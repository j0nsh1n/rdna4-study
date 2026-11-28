# Round 13 hardware microbenchmarks: preregistered expectations

Drafted: 2026-11-27, BEFORE any Round 13 GPU run.
Target card: AMD Radeon RX 9070 XT (`gfx1201`, 64 CUs / 32 WGPs / 128 SIMDs, 256 TMUs, 32 KiB L0/CU, 8 MiB L2, 64 MiB MALL, 16 GiB 256-bit GDDR6), ROCm 10.1 (`hipcc`, `-mcumode`), Mesa RADV `27.0.0-devel` (ACO).
Sources: `src/round13/msad.hip`, `src/round13/vk_tex.c`, `src/round13/vk_tex_fp16.c`, `src/round13/vk_async.c`, `scripts/round13/build_probes.sh`, `scripts/round13/build_async.sh`, `scripts/round13/run_probes.sh`.
Labels: MEASURED (cited from earlier study files), READ (ISA / compiler source / hardware table), INFERRED (prediction before running).

---

## 1. Probe A — `v_msad_u8`, `v_sad_u8`, `v_sad_u16`, `v_sad_hi_u8` (`msad_w32`, `msad_w64`)

### Why this matters
In AMD's published MIT FSR 3.1.6 Optical Flow (`ffx_opticalflow_compute_optical_flow_v5.h`), the inner search loop executes 16 `msad4` (`v_msad_u8`) instructions per block in Wave64 with 0% FP16 instructions. The study's execution-core pass (`measurements/execution-core.md`) measured basic INT32 ALU (1 cyc issue / 5 cyc latency), 32-bit integer multiply `v_mul_lo_u32` (4 cyc issue / 8 cyc latency), `v_mul_u32_u24` (1 cyc / 5 cyc), and `v_dot2_f32_f16` (1 cyc / 6 cyc), but never measured any SAD/MSAD instruction.

### Preregistered numeric predictions (INFERRED)
1. **FP32 FMA calibration (`k_fma_base`, VOPD disabled):**
   - Wave32: `1.00` cyc/op at `c = 16`, `5.00` cyc/op at `c = 1` (MEASURED anchor from `measurements/execution-core.md` §2).
   - Wave64: `1.00` cyc/op at `c = 16` if the 64-lane SIMD VALU executes one wave64 FMA per clock (`measurements/vulkan-view.md` §3: wave64 `v_fmac_f32_e32` reached 0.944 of 64 FMAs/SIMD/clock, INFERRED 1 cyc/inst), or `2.00` cyc/op if issued over two 32-lane passes; dependent chain `c = 1` predicted at `5.00–6.00` cyc/op.
2. **SAD / MSAD issue rate (`c = 16`) and dependent-chain latency (`c = 1`):**
   - LLVM's `VOP3Instructions.td` (READ) assigns `V_MSAD_U8`, `V_SAD_U8`, `V_SAD_U16`, and `V_SAD_HI_U8` to the default 32-bit VALU class (`Write32Bit`, 5 cyc latency, no `WriteQuarterRate32` override), unlike `V_MQSAD_PK_U16_U8` which carries `GFX1250BlockingCycles = 4`.
   - **Primary prediction (full-rate VALU):** all four instructions issue at **`1.00` cyc/op** (`c = 16`) in wave32 (and `1.00–2.00` cyc/op in wave64 matching the FMA wave64 ratio), with dependent-chain latency **`5.00–6.00` cyc/op** (`c = 1`, both `acc` and `src0` chains equal within `0.1` cyc; if accumulator forwarding differs, `acc` may be 1 cycle shorter than `src0`).
   - **Explicit alternative hypothesis (quarter-rate integer pipe, like `v_mul_lo_u32`):** if RDNA 4 places legacy SAD/MSAD on the quarter-rate integer multiplier pipe (`measurements/execution-core.md` §2: `v_mul_lo_u32` = `4.00` cyc issue, `8.00` cyc latency), then `c = 16` will land at **`4.00` cyc/op** (wave32) and `c = 1` at **`7.00–8.00` cyc/op**. Intermediate chains `c = 2, 4, 8` will follow `max(issue, latency / c)`.
3. **Co-issue mixed with FP32 FMA (`msad+1fma`, `msad+3fma`, `sad*+1fma`, VOPD disabled):**
   - `v_msad_u8` and `v_sad_*` are VOP3-only and have no VOPD dual-issue encoding (READ, `notes/research-1-instructions.md` §2).
   - **Prediction:** **No co-issue** with FP32 FMA (unlike transcendentals `v_exp_f32 + 3 FMA = 4.0 cyc`, `measurements/execution-core.md` §3). The mixed pattern cost will equal the **sum** of its parts: `cost(1 SAD + 1 FMA) = issue(SAD) + issue(FMA)` (`~2.00` cyc/pair if full-rate, `~5.00` cyc/pair if quarter-rate), and `cost(1 MSAD + 3 FMA) = issue(MSAD) + 3 * issue(FMA)` (`~4.00` cyc if full-rate, `~7.00` cyc if quarter-rate).
   - **What would prove co-issue:** `cost(1 MSAD + 1 FMA) <= max(issue(MSAD), issue(FMA)) + 0.15` (e.g., `~1.0` cyc if full-rate, or `~4.0` cyc for both `1+1` and `1+3` if quarter-rate on a separate pipe like TRANS).

### Controls, static proof, and runtime
- **Static check (verified on CPU):** all `93` timed kernels in `msad_w32.s` (`wavefront_size32=1`) and `93` in `msad_w64.s` (`wavefront_size32=0`) contain ONLY the intended instruction (`16 * U` ops, or `16 * U` SAD + `16 * U * NFMA` `v_fmac_f32_e32`), zero `v_mov` and zero spills.
- **Exactness control:** `k_sad_exact` tests 256 vectors per lane (including asymmetric zero bytes in `src0` vs `src1` confirming `src1` is the `v_msad_u8` mask, plus 32-bit wrap), and every timed kernel checks all 16 chains across all 32/64 lanes bit-for-bit against `src/round13/msad_ref.h`. `--break` corrupts expected chain/table bits and calibration and must exit non-zero (`FAIL`).
- **Expected GPU runtime:** `~3–5 seconds` total for both `msad_w32` and `msad_w64` including breaks.

---

## 2. Probe B — Vulkan compute texture sampling throughput (`vk_tex`)

### Why this matters
Almost every FidelityFX pass (FSR 1/2/3, Frame Interpolation, CACAO, Denoiser, Lens, Brixelizer GI) is bounded by `texelFetch`, `textureLod` bilinear, or `textureGather` across `R8_UNORM`, `RGBA8_UNORM`, `R16G16_SFLOAT`, `R16G16B16A16_SFLOAT`, `R32G32B32A32_SFLOAT`, and `B10G11R11_UFLOAT`. Round 11 Item 3's graphics texture sweep failed its memory anchor and suffered `>10%` spread on unpinned clocks (`measurements/round11.md` §Slot 2). This compute probe isolates TMU + cache-hierarchy throughput cleanly across L0 (`16 KiB`), L2 (`2 MiB`), MALL (`32 MiB`), and DRAM (`256 MiB`) in both wave32 and wave64, with per-run `gpu_metrics` SCLK/UCLK.

### Preregistered numeric predictions (INFERRED)
Hardware ceilings (READ `scripts/round11/expect/expect-3.md` & MEASURED `measurements/memory-system.md` §2a, `measurements/round7.md`, `measurements/blindspot-followups.md` §4):
- **TMUs:** 256 TMUs = `4.0` TMUs/CU. At `3.0–3.4 GHz` SCLK, `4.0 samp/clk/CU` = `768–870 GSamp/s`.
- **L0 vector cache:** `32 KiB/CU`, `128 B/cyc/CU` read port (`~19 TB/s` card-wide for shared data, `measurements/memory-system.md` §2a).
- **L2 cache:** `8 MiB`, `7.4–9.5 TB/s` (`~40–50 B/cyc/CU` at `3.0 GHz`, `measurements/memory-system.md` §2a).
- **MALL (Infinity Cache):** `64 MiB`, `~2.0 TB/s` (`~10 B/cyc/CU`, `measurements/memory-system.md` §2a; note tiled 2D locality in `32 MiB` can exceed `2.0 TB/s` up to `~3–5 TB/s` if neighboring waves hit in L2, as `lds-and-special-loads.md` §5 saw `5.3–5.4 TB/s` on a 32 MiB tiled window).
- **DRAM (GDDR6):** `633 GB/s` at `1258 MHz` UCLK (`measurements/memory-system.md` §1) and `~700 GB/s` at `1408 MHz` UCLK (`measurements/round7.md`). Any incompressible `256 MiB` DRAM cell reporting `> 735 GB/s` (`720.9 * 1.02`) would indicate either DCC compression or a cache-coverage bug.

Predicted steady-state rates (`samp/clk/CU` and effective `GB/s` at `~3.0–3.3 GHz` SCLK, `1408 MHz` UCLK):

| Format (B/texel) | L0 (`16 KiB`) `texelFetch` | L0 (`16 KiB`) `textureLod` bilinear | L0 (`16 KiB`) `textureGather` | L2 (`2 MiB`) `textureLod` | MALL (`32 MiB`) `textureLod` | DRAM (`256 MiB`) `textureLod` |
|---|---|---|---|---|---|---|
| `R8_UNORM` (1 B) | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.0–4.0 /clk/CU` (~0.7 TB/s) | `2.5–3.8 /clk/CU` (~0.6 TB/s) | `2.0–3.2 /clk/CU` (`450–680 GB/s`) |
| `RGBA8_UNORM` (4 B) | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.0–4.0 /clk/CU` (~2.8 TB/s) | `1.8–3.0 /clk/CU` (~1.8–2.5 TB/s) | `0.75–0.90 /clk/CU` (`600–700 GB/s`) |
| `R16G16_SFLOAT` (4 B) | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.0–4.0 /clk/CU` (~2.8 TB/s) | `1.8–3.0 /clk/CU` (~1.8–2.5 TB/s) | `0.75–0.90 /clk/CU` (`600–700 GB/s`) |
| `B10G11R11_UFLOAT` (4 B) | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.5–4.0 /clk/CU` | `3.0–4.0 /clk/CU` (~2.8 TB/s) | `1.8–3.0 /clk/CU` (~1.8–2.5 TB/s) | `0.75–0.90 /clk/CU` (`600–700 GB/s`) |
| `R16G16B16A16_SFLOAT` (8 B) | `3.0–4.0` (`fetch` / `gather`) | **`1.8–2.0 /clk/CU`** (64-bit half-rate filter) | `3.0–4.0 /clk/CU` (single channel) | `1.8–2.0 /clk/CU` (~3.0 TB/s) | `1.0–1.6 /clk/CU` (~1.8–2.5 TB/s) | `0.38–0.45 /clk/CU` (`600–700 GB/s`) |
| `R32G32B32A32_SFLOAT` (16 B) | **`1.8–2.0 /clk/CU`** (L0 128 B/cyc/CU & TA b128) | **`0.9–1.0 /clk/CU`** (128-bit quarter-rate filter) | `1.8–2.0 /clk/CU` (16 B fetch, 1 ch) | `0.9–1.0 /clk/CU` (~3.0 TB/s) | `0.5–0.9 /clk/CU` (~1.8–2.5 TB/s) | `0.19–0.23 /clk/CU` (`600–700 GB/s`) |

Key structural predictions:
1. **Format filter-rate steps at L0:** `32-bit` (`RGBA8`, `R16G16F`, `B10G11R11`) bilinear hits `~3.5–4.0 samp/clk/CU`; `64-bit` (`RGBA16F`) bilinear drops to **`~2.0 samp/clk/CU` (0.5x)**; `128-bit` (`RGBA32F`) bilinear drops to **`~1.0 samp/clk/CU` (0.25x)**.
2. **`textureGather` vs `textureLod` on `RGBA16F`:** because `textureGather(..., 0)` extracts only 1 component (16 bits/texel) without 4-channel FP16 filtering, it is predicted to run **faster than `textureLod` on `RGBA16F` (`~3.0–4.0` vs `~2.0 samp/clk/CU`)**, matching why FSR 1 EASU uses `GatherRed/Green/Blue`.
3. **Wave32 vs Wave64:** from `measurements/blindspot-followups.md` §4 (TA issue rate = `~8 cyc/load` wave32 vs `~16 cyc/load` wave64, i.e., equal lanes/cycle at 4 lanes/clk/CU for 32-bit loads!), **wave32 and wave64 should match within ~10%** at L0 and DRAM (where 1024–4096 workgroups saturate occupancy), with wave32 slightly ahead at L2/MALL if lower per-wave latency hiding hurts wave64 (`4` vs `8` waves/WG).
4. **DRAM convergence:** at `256 MiB`, all formats with `bpp >= 4` (`RGBA8`, `R16G16F`, `B10G11R11`, `RGBA16F`, `RGBA32F`) converge to the GDDR6 ceiling (**`~600–700 GB/s`** at `1408 MHz` UCLK, or `~550–630 GB/s` at `1258 MHz`), whereas `R8_UNORM` (`1 B/texel`) may remain filter/address-bound below DRAM saturation (`~450–650 GB/s`).

### Controls, static proof, and runtime
- **SPIR-V check (verified on CPU):** `vk_tex_fetch.spv` (`16` `OpImageFetch`), `vk_tex_lod.spv` (`16` `OpImageSampleExplicitLod`), `vk_tex_gather.spv` (`16` `OpImageGather`) all pass `spirv-val --target-env vulkan1.3`.
- **ACO check (verified at runtime / `--dump-asm`):** `vk_tex` captures ACO disassembly via `VK_KHR_pipeline_executable_properties` and asserts `>= 16` `image_load` / `image_sample` / `image_gather4` in the inner loop.
- **Exactness control & break:** exact FP16-representable bilinear (`weights = 0.375, 0.125, 0.375, 0.125`) checked bit-exact (`err == 0`) on all 6 formats (`exact_mode=1` control) and on all 4 float formats in every timed run; `--break` shifts x by `+1` texel and must fail (`exact=FAIL`).
- **Expected GPU runtime:** `~10–25 seconds` full sweep (`144` cells + uploads + break).

---

## 3. Probe C — Texture sampling overlapped with packed FP16 / FP32 FMAs (`vk_tex_fp16`)

### Why this matters
Analytic FidelityFX shaders (FSR 1.2, FSR 2.3.4, FSR 3.1.6, CAS, CACAO, Denoiser) interleave packed FP16 (`f16vec2` -> `v_pk_fma_f16`) or FP32 (`v_fma_f32`) math with `textureLod` sampling. Knowing whether wall time follows `max(T_tex, T_alu)` (true hardware overlap between TMU/VMEM and VALU) or `T_tex + T_alu` (serialized) determines the ALU-to-texture balance point on `gfx1201`.

### Preregistered numeric predictions (INFERRED)
From `measurements/execution-core.md` §2 & `measurements/vulkan-view.md` §3:
- **Pure ALU baseline (`DO_TEX = 0`, 64 CUs / 128 SIMDs):**
  - **FP16 (`v_pk_fma_f16`):** `1` instruction/cycle/SIMD (`2` FP16 FMAs/lane/cyc). In wave32: `128 SIMDs * 32 lanes * 1 inst/cyc = 4,096` `v_pk_fma_f16`/clk = **`64 inst/clk/CU`** = `~50–55 TFLOPS` at `3.1–3.4 GHz` (if wave64 issues `v_pk_fma_f16` over 1 cycle for 64 lanes, **`128 inst/clk/CU`** = `~100–110 TFLOPS`; if over 2 cycles, `64 inst/clk/CU` matching wave32).
  - **FP32 (`v_fma_f32` / `v_dual_fmac_f32`):** `64–128 inst/clk/CU` depending on whether ACO pairs `v_dual_fmac_f32` in wave32 (`measurements/vulkan-view.md` §3 measured `50.4 TFLOPS` wave32 dual-issue and `49.4 TFLOPS` wave64 single-issue at `3.2 GHz`, i.e., **`~128 FMAs/clk/CU`**!).
- **Overlap behavior (`vs_max = T_mix / max(T_tex, T_alu)` vs `vs_sum = T_mix / (T_tex + T_alu)`):**
  - Because `gfx1201` has independent VALU and TMU/VMEM pipes with a dedicated 6-bit `SAMPLE_CNT` wait counter (`notes/research-10-compilers.md` §5) and 1024–4096 workgroups saturate occupancy across SIMDs, **time is predicted to follow `max(T_tex, T_alu)` (strong overlap: `vs_max = 1.00–1.20`, `vs_sum = 0.55–0.75` near the crossover `N*`)**, NOT `T_tex + T_alu`!
  - **Predicted crossover `N*` (FMA instructions per texture sample where `T_alu == T_tex`):**
    - `N* = (VALU inst/clk/CU) / (TMU samp/clk/CU)`:
    - **L0 `RGBA8` (`~3.8 samp/clk/CU`):** `N* ~ 16–32` `v_pk_fma_f16` (or `~32` FP32 FMAs) per texture tap. At `N <= 8`, `T_mix` stays within **`+15%`** of `T_tex` (`vs_max ~ 1.0–1.15`).
    - **L0 `RGBA16F` (`~2.0 samp/clk/CU`):** `N* ~ 32–64` `v_pk_fma_f16` per texture tap. At `N <= 16`, `T_mix` stays within **`+15%`** of `T_tex`.
    - **DRAM `RGBA16F` (`~0.42 samp/clk/CU`):** `T_tex` is `~5x` longer than L0, so **`N = 64` `v_pk_fma_f16` per texture tap will still be 100% hidden behind DRAM bandwidth** (`vs_max ~ 1.00–1.10`, `T_mix` nearly flat from `N = 0` to `N = 64`!).
  - **What would contradict strong overlap:** `vs_max > 1.35` (approaching `vs_sum ~ 1.0`) at the crossover `N*`, which would indicate either register-file/VGPR-bank contention, `s_wait_samplecnt` serialization in ACO, or low wave occupancy (visible immediately in the captured ACO `.s` `VGPRs` and `Subgroups per SIMD` stats).

### Controls, static proof, and runtime
- **SPIR-V check (verified on CPU):** all 20 SPIR-V shaders (`vk_mix_{fp16,fp32}_n{0,2,4,8,16,32,64}.spv` + `vk_alu_{fp16,fp32}_n{4,16,64}.spv`) pass `spirv-val` with exact `OpImageSampleExplicitLod` (`4` or `0`) and exact `4 * N + 5` (N > 0; `0` at N = 0; the `+ 5` is the post-loop parity step added after review 2026-11-27) `OpExtInst %v2half Fma` / `%float Fma` counts.
- **ACO check (verified at runtime / `--dump-asm`):** asserts `image_sample >= 4` (for `DO_TEX=1`) and `v_pk_fma_f16 >= 4 * N` (for `fp16`) or `fma_f32 + 2 * dual_fmac_f32 >= 4 * N` (for `fp32`).
- **Exactness control & break:** 5-chain period-2 oscillating FMA (`a = fma(a, -1.0, ak)`) with odd `iters` (or, for even `iters`, one extra 5-FMA parity step per chain after the loop, so the SPIR-V count is `4 * N + 5`; added after review 2026-11-27) guarantees bit-exact FP16 & FP32 verification with zero overflow and non-trivial parity on every `N`; `--break` shifts x by `+1` and corrupts expected ALU parity and must fail (`exact=FAIL`).
- **Expected GPU runtime:** `~10–20 seconds` full sweep (`12` pure-ALU + `168` interleaved cells + uploads + break).

---

## 4. Probe D (P4) — All-Vulkan async compute beside graphics (`vk_async`)

### Why this matters
Games run FidelityFX passes (frame generation, optical flow, denoisers, FSR) on Vulkan/D3D12 async compute queues beside graphics within a single device. Round 12 (`measurements/blindspot-followups.md` §2) measured only a separate-process HIP FP8 WMMA job beside Vulkan 2560x1440 RGBA16F graphics (`src/round12/gfx_load.c:16-18, 167-220, 243-264, 333-397` and `scripts/round12/run_async_overlap.sh:83-88`): under `SAT-100%` (`in_flight=2`, `iters=6450`, `3.79 ms/frame`), a monolithic HIP kernel (`chain=1`, `9.5 ms` alone) slowed only `1.25–1.29x` (`11.9–12.3 ms`), while an 80-dispatch chain (`chain=80`, `10.8 ms` alone) starved `13.6–14.0x` (`147–151 ms`) with default stream priority until hardware queue priority 2 was enabled; under `PACE-60` (`in_flight=1`, `pace_ms=16.6`, `iters=10791`, `5.69 ms` draw), `chain=80` added `+5.5–5.7 ms` (`1.51–1.53x`, one draw pass). `vk_async` tests whether an **all-Vulkan single-`VkDevice`** app using a separate compute-only queue family and `VK_KHR_global_priority` shows the same dispatch-boundary starvation across four realistic FidelityFX-class workloads:
- **(a) `fp16`:** packed FP16 FMA ALU (`v_pk_fma_f16`, reusing P2's `vk_tex_fp16.comp` with `USE_FP16=1, DO_TEX=0, FMA_PER_TEX=64`).
- **(b) `sad`:** integer SAD-like ALU (`vk_async_sad.comp`; **explicit note:** GLSL/SPIR-V has no `msad4`/`v_msad_u8` instruction, so this tests unpacked 4-byte `max - min` integer SAD ALU).
- **(c) `tex`:** texture-sampling heavy (`32 MiB` `RGBA16F` MALL `textureLod`, reusing P2's `vk_tex.comp` `OP_MODE=1`).
- **(d) `stream`:** `128 MiB -> 128 MiB` (`256 MiB > 64 MiB` MALL) coalesced `uvec4` GDDR6 streaming (`5120 MiB` = `5.0 GiB` total traffic per chain; *2026-11-27 post-review unit correction from `5.12 GiB`*).

### Preregistered numeric predictions (INFERRED)
1. **Queue families & `VK_KHR_global_priority` without root (READ + INFERRED):**
   - RADV on RX 9070 XT (`gfx1201`) is expected to expose a dedicated compute-only queue family (`COMPUTE | TRANSFER`, `separate_compute=1`, `count=4`, `ts_bits=64`) distinct from the universal graphics queue family (`GRAPHICS | COMPUTE | TRANSFER`, `count=1`).
   - Without `CAP_SYS_NICE` / root, `amdgpu` DRM context creation allows **`LOW` (`128`) and `MEDIUM` (`256`)**, while **`HIGH` (`512`) and `REALTIME` (`1024`) are predicted to be refused with `VK_ERROR_NOT_PERMITTED` (`-1000174001`)** (if the kernel/environment grants `HIGH` without root, `HIGH` will join the ABBA schedule automatically).
2. **Alone baselines (`~3.1–3.3 GHz` SCLK, `1408 MHz` UCLK):**
   - Graphics alone (anchored by Round 12 MEASURED `measurements/blindspot-followups.md` §2): `SAT-100%` `dt_avg ~ 3.7–4.0 ms`, `gpu_p50 ~ 3.6–4.0 ms`; `PACE-60` `dt_avg ~ 16.6 ms`, `gpu_p50 ~ 5.5–6.0 ms`.
   - Compute alone (sized to `~6–10 ms` per chain at both `N = 1` and `N = 80`): `fp16` `~6–10 ms`, `sad` `~6–11 ms`, `tex` `~6–11 ms`, `stream` `~7.5–9.0 ms` (`5120 MiB` = `5.0 GiB` at `~600–700 GB/s`; *2026-11-27 post-review unit correction from `5.12 GiB`*, prediction band unchanged). `N = 80` alone adds `< 1.0 ms` barrier/dispatch overhead over `N = 1` alone (`~0.5–10 us/barrier`).
3. **Monolithic (`N = 1`) vs chained (`N = 80`) overlap under `SAT-100%` (`in_flight=2`) at `MEDIUM` priority:**
   - **Monolithic `N = 1`:** all 4 jobs are predicted to overlap well with graphics (**`1.15–1.80x` compute slowdown**):
     - `stream` (`N = 1`) slows least (**`1.05–1.35x`**) because `gdraw.frag` is pure ALU (`29.5 MiB` render target write-only, low GDDR6 traffic), leaving memory bandwidth mostly free.
     - `fp16` and `sad` (`N = 1`) compete for VALU/CUs with `gdraw.frag` (**`1.20–1.70x`** compute slowdown; graphics `dt_avg` rises **`+10–40%`**).
     - `tex` (`N = 1`, `32 MiB` MALL) competes with both CUs and the `29.5 MiB` `RGBA16F` render target in the `64 MiB` MALL (**`1.25–1.90x`** compute slowdown).
   - **Chained `N = 80`:** compute-to-compute barriers (`CS_PARTIAL_FLUSH`) create 80 wave-drain/relaunch boundaries where the saturated graphics ring (`in_flight=2`) can hold CUs.
     - **Primary prediction:** `N = 80` suffers strong dispatch-boundary starvation (**`3–15x` compute slowdown**, i.e. `~30–150 ms`), qualitatively reproducing Round 12's HIP `13.8x` starvation on all 4 jobs (with `fp16`/`sad`/`tex` worst and `stream` also starving at barrier boundaries because new compute waves cannot launch while graphics occupies all CUs), while graphics `dt_avg` stays near baseline (**`0.95–1.10x`**).
     - **Explicit alternative hypothesis:** if RADV's same-process MEC async compute pipe arbitrates more fairly against its own GFX ring than cross-API KFD/AQL queues did in Round 12, `N = 80` slowdown will land lower (**`1.5–3.0x`**) while graphics `dt_avg` rises correspondingly (`+15–35%`).
4. **Paced graphics (`PACE-60`, `16.6 ms` cadence, `~5.7 ms` draw + `~10.9 ms` idle gap) at `MEDIUM` priority:**
   - `N = 1` slows **`1.05–1.40x`** (`+0.5–3.5 ms`, depending on phase alignment with the 5.7 ms draw).
   - `N = 80` adds **`+4.0–7.0 ms` (`1.45–1.90x` median)**, matching one overlapping draw-pass duration (anchored by Round 12 MEASURED `+5.5–5.7 ms`), while graphics `dt_avg` stays pinned at **`~16.6 ms`** (and `dt_p95` stays `< 18.5 ms` as long as overlapped `gfx_gpu` stays well below `16.6 ms`).
5. **Global priority effect (`LOW` vs `MEDIUM`, plus `HIGH` if permitted without root, ABBA order):**
   - `LOW` (`AMDGPU_CTX_PRIORITY_LOW`) vs `MEDIUM` (`AMDGPU_CTX_PRIORITY_NORMAL`): **`LOW` is predicted to be equal or slower than `MEDIUM`** (by **`0–30%`** on `N = 1` and up to **`1.1–2.0x`** on `N = 80` if the hardware scheduler deprioritizes low-priority compute waves behind graphics), while protecting graphics `dt_p95` slightly better.
   - If `HIGH` is permitted without root: **`HIGH` is predicted to cut `N = 80` `SAT-100%` starvation dramatically** (down toward `1.3–2.5x`) while visibly increasing graphics `dt_avg` and `dt_p95` (`+20–60%`), matching Round 12's hardware priority 2 result.

### Controls, static proof, and runtime
- **SPIR-V check (verified on CPU via `scripts/round13/build_async.sh`):** all 7 SPIR-V shaders (`gdraw_vert`, `gdraw_frag`, `vk_alu_fp16_n64`, `vk_async_sad`, `vk_tex_lod`, `vk_async_stream_init`, `vk_async_stream`) pass `spirv-val --target-env vulkan1.3`, have distinct SHA-256 hashes, and match exact instruction counts (`261` `%v2half Fma` (`256` in the loop plus the 5-FMA parity step, after review 2026-11-27), `>=64` integer `UMax`/`UMin`/`OpISub` with `0` float FMAs, `16` `OpImageSampleExplicitLod`, `>=4` `%v4uint` load/XOR/store, `>=2` graphics FP32 `Fma`).
- **ACO check (verified at runtime / `--dump-asm`):** captures ACO disassembly via `VK_KHR_pipeline_executable_properties` for all 5 timed pipelines (`fp16`, `sad`, `tex`, `stream`, `gdraw`) and asserts intended instructions (`v_pk_fma_f16 >= 256`, `int_alu >= 64` & `v_msad == 0`, `image_sample >= 16`, `ld_b128 >= 4` & `st_b128 >= 4`, graphics `fma_f32 >= 2`).
- **Exactness control, deliberate break, and fail-closed thermal stop:** every compute job verifies bit-exact (`bad=0`) on CPU for both `N = 1` and `N = 80`; `vk_async --break` corrupts GPU inputs/coords on all 4 jobs and must fail 4/4 (`VERDICT vk_async: FAIL`); `vka_check_hotspot_file` fails closed (unreadable/corrupt sensor or `>= 95 C` for this probe and the runner since the owner's 2026-11-27 change, item 11 below; the texture probes keep `>= 90 C` => immediate stop) in both graphics and compute submission loops, verified by `vk_async_selftest` + 12 deliberate breaks in `--cpu-checks`.
- **Expected GPU runtime:** **`~25–65 seconds`** full run (`~20–25 s` if `LOW`/`MEDIUM` allowed -> 4 ABBA passes; `~30–40 s` if `HIGH` also allowed -> 6 ABBA passes; `< 65 s` even with worst-case 150 ms `N = 80` starvation + thermal cooldowns + `--break`, well under the 3-minute cap).

---

## 5. Build, CPU-check, and GPU-run commands

- **Build Probes A–C (CPU only):** `bash scripts/round13/build_probes.sh`
- **Build Probe D / `vk_async` (CPU only):** `bash scripts/round13/build_async.sh`
- **CPU self-tests & breaks (CPU only, includes `vk_async`):** `bash scripts/round13/run_probes.sh --cpu-checks`
- **ACO disassembly verification only (no GPU execution):** `bash scripts/round13/run_probes.sh <run_dir> --dump-asm` (or `--only vk_async --dump-asm`)
- **Run `vk_async` only on GPU (lead):** `bash scripts/round13/run_probes.sh <run_dir> --only vk_async` (expected GPU time: **`~25–65 seconds`**, well under 3 minutes; `--smoke` runs in `~5 seconds`)
- **Full GPU run across all four probes (lead):** `bash scripts/round13/run_probes.sh <run_dir>` (total expected GPU time: **`~60–120 seconds`** across all four probes + breaks).

---

## 6. Addendum (2026-11-27, before re-running `vk_tex` and `vk_tex_fp16`): DRAM visit-once & in-shader clock fixes

Added before any re-run of Probes B (`vk_tex`) and C (`vk_tex_fp16`) to close the two Slot 2 measurement caveats (strip-wrap MALL hits on DRAM cells and 10 ms `gpu_metrics` polling missing 0.3–9 ms bursts).

### What changed
1. **DRAM visit-once footprint (`src/round13/vk_tex_ref.h`, `vk_tex.c`, `vk_tex_fp16.c`):**
   - `VKT_LVL_DRAM` increased to **`512 MiB`** (`8x` the 64 MiB MALL) for all `bpp >= 4` formats (`16384x8192` for 4 B, `8192x8192` for 8 B, `8192x4096` for 16 B), and stays at the hardware `maxImageDimension2D = 16384` cap **`256 MiB`** (`16384x16384`, `4x` MALL) for 1 B/texel `R8_UNORM`.
   - `vkt_dram_schedule()` sets `groups * 256 * iters * taps == w * h` with `y_tile_step = row_span = groups / strips_x` and `row_span * iters == tiles_y`, visiting **every texel strictly once per dispatch** with zero strip wrap and zero tile reuse. Proven on CPU by `vk_tex_selftest` (tile-level across all 6+2 DRAM configs, direct texel-by-texel on full 512 MiB `RGBA32F` and `RGBA16F`, negative controls catching the old Slot 2 4x wrap/overlap, and `--break 7`).
2. **In-shader core clock + warmed timed window + `< 5` samples refusal (`vk_tex.comp`, `vk_tex_fp16.comp`, `vk_tex_common.c`, `vk_tex.c`, `vk_tex_fp16.c`):**
   - Primary SCLK now uses RADV's exposed `VK_KHR_shader_clock` (`GL_ARB_shader_clock` `clockARB()` -> `HW_REG_SHADER_CYCLES_LO/HI`, over `GL_EXT_shader_realtime_clock` `clockRealtimeEXT()` -> 100 MHz `MSG_RTN_GET_REALTIME`, already verified on this card in `measurements/vulkan-view.md` §3 to match `pp_dpm_sclk` to `0.03%`), sampled outside the inner loop on 64 evenly-spaced workgroups per dispatch. Backward-compatible with `vk_async` (`out_stride_wg = 4` -> `clk_slots = 0`).
   - Each cell runs a bounded `~20 ms` warmup (`3–40` dispatches) followed by a bounded `~30 ms` timed window (`8–48` dispatches, hard 60 ms wall cap) under fail-closed `90 C` thermal stop and `< 60 C` pre-cell cooldowns, with `gpu_metrics` polled every **`2 ms`** strictly inside the timed window (no post-burst idle fallback) as an independent `gm_sclk` / `uclk` cross-check.
   - Every `RESULT` line reports `clk_n` (valid timed-window clock samples) and `gm_n` (in-window `gpu_metrics` samples) and **refuses** to print clock-normalised values (`samp/clk/CU`, `inst/clk/CU`, `T_alu_est`, `vs_max`, `vs_sum` -> `N/A`) whenever `clk_n < 5`.

### Updated preregistered expectations (INFERRED, before re-run)
- **`vk_tex` DRAM (`512 MiB`, `256 MiB` for `R8_UNORM`):**
  - Full-entropy `RGBA8_UNORM` (4 B) DRAM bandwidth will drop from Slot 2's MALL-cached `1,300–1,840 GB/s` to the GDDR6 ceiling (**`~550–700 GB/s`** at `1408 MHz` UCLK; `<= 735 GB/s` ceiling).
  - If 7-bit dyadic float textures (`R16G16F`, `B10G11R11`, `RGBA16F`, `RGBA32F`) remain near **`~550–700 GB/s`**, DRAM is cleanly uncached; if any float format exceeds `735 GB/s` while `RGBA8_UNORM` stays `<= 700 GB/s`, that isolates hardware DCC compression on `VK_IMAGE_TILING_OPTIMAL` (since visit-once rules out MALL reuse).
  - `R8_UNORM` (1 B, `256 MiB`) may stay below GDDR6 saturation if filter/address bound (`~350–650 GB/s`, `<= 735 GB/s`).
- **Clock-normalised metrics (`vk_tex` & `vk_tex_fp16`):**
  - Every cell will report `clk_n >= 8` (and `gm_n >= 8`), active in-shader `sclk` in **`~2,600–3,500 MHz`**, `SPREAD>10%` near zero, and `samp/clk/CU <= 4.1` on every cell (with L0 `texelFetch` `~3.8–4.0` or `~4.0` if 4 TMU/CU, `textureLod` `1–8 B` `~1.9–2.1` or `~3.8` depending on active boost clock vs Slot 2's ~1.8–2.4 GHz tier, and `RGBA32F` at `0.5x` of `RGBA16F`).
  - `vk_tex_fp16` `vs_max` will land cleanly in **`~1.00–1.25`** near crossover without any low-SCLK `< 0.5` corruption, and DRAM `RGBA16F` / `RGBA8` `N = 0` will sit near the GDDR6 ceiling (`~550–700 GB/s` on `RGBA8`, hiding FP16/FP32 FMAs cleanly).
- **Expected GPU runtime:**
  - `bash scripts/round13/run_probes.sh <run_dir> --only vk_tex`: **`~15–25 seconds`** (worst case `< 45 s`, well under 90 s).
  - `bash scripts/round13/run_probes.sh <run_dir> --only vk_tex_fp16`: **`~20–40 seconds`** including `< 60 C` cooldowns on heavy `N = 64` cells (worst case `< 60 s`, well under 90 s).

---

## 7. Post-review method fixes (2026-11-27)

Recorded after an outside code review of Probes A–D. No preregistered numeric prediction band in §§1–6 was changed; only probe method defects and the `5120 MiB` (`5.0 GiB`) stream unit label were fixed:

1. **Probe D (`vk_async`) SAT-100 chains and equal total work (Finding 1):** Restored `VKA_CHAIN_SAT = 80` so both `SAT-100` and `PACE-60` compare preregistered `N = 1` vs `N = 80` chains, and matched per-job total work strictly between `N = 1` and `N = 80` (`fp16`: `880` vs `80 x 11`; `sad`: `560` vs `80 x 7`; `tex`: `160` vs `80 x 2`; `stream`: `20 x 4` vs `80 x 1` quarters = `5120 MiB` = `5.0 GiB`), printed at startup (`WORK_TOTAL`) and on every `BASELINE_COMP` / `RESULT` line.
2. **Probe C (`vk_tex_fp16`) DRAM ALU exactness (Finding 2):** Kept visit-once `512 MiB` DRAM geometry and added one post-loop FMA step per chain in `vk_tex_fp16.comp` whenever `FMA_PER_TEX > 0` and `pc.iters` is positive and even, forcing odd per-chain parity; `vkt_verify_mix_alu_lane` now also rejects any cell whose stored ALU output equals the untouched initial accumulator sum when FMA work was requested.
3. **Probe A (`msad`) wave64 FMA calibration gate (Finding 3):** Updated `msad_ref::fma_calib_ok` to accept `1.00` or `2.00 +- 0.10` cyc/op on wave64 only (matching §1), while keeping wave32 at `1.00 +- 0.10` and still requiring bit-exact outputs and linearity.
4. **Hard `< 60 C` pre-cell cooldown enforcement (Finding 4):** Shared `vka_precell_cooldown_action` across `vk_tex`, `vk_tex_fp16`, and `vk_async` (baselines and arms) so a missed `< 60 C` cooldown never submits the cell, exits `4` if the wait ended at or above the probe's stop (`95 C` for `vk_async` since item 11, `90 C` for `vk_tex` and `vk_tex_fp16`) or on a sensor error, and otherwise fails the cell and verdict (exit `1`).
5. **End shader-clock ordering and minimum span check (Finding 5):** Marked `OutBuf` `coherent` in `vk_tex.comp` and `vk_tex_fp16.comp`, stored the loop accumulator to the clock slot, issued `memoryBarrierBuffer()`, reloaded it, and guarded the end `clockARB()` / `clockRealtimeEXT()` reads behind a branch on the reloaded value; `vkt_decode_shader_clk_mhz` rejects spans `< 1000 ns` or `< 2000` cycles, `build_probes.sh` checks SPIR-V order across all 23 shaders, and `vkt_asm_clock_order_ok` checks post-loop reload+clock order in ACO listings at runtime.
6. **GPU lock path (Finding 6):** Left at the local shared GPU lock during the private run (public `run_probes.sh` defaults to `$ROOT/runs/gpu.lock`, override via `STUDY_GPU_LOCK`).
7. **Vulkan compute dispatch timestamps (Finding 7):** Updated `vk_tex_common.c` `vkt_dispatch` to use `vkCmdWriteTimestamp2` at `VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT` immediately before `vkCmdDispatch` and `VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT` after it, matching `vk_async.c`'s `record_gfx_frame`.
8. **Stream traffic unit correction (Finding 8):** Corrected `5.12 GiB` to `5120 MiB` (`5.0 GiB`) in `vk_async_stream.comp`, `vk_async.c` printout, and §4 above (prediction bands unchanged).
9. **Runner pre-flight hotspot guard (Finding 9):** Updated `scripts/round13/run_probes.sh` `check_hot_file` to reject readings `<= 0` and return `4` before any job starts.
10. **Round-2 review (2026-11-27):** Probe D `tex` uses `N = 1` at `80` iterations and `N = 80` at `1` (the nearest
    equal pair to the registered monolithic size of `64`; F2 had resized it to `160`, which the review rejected). The
    `fp16` job's even `N = 1` chain carries the 5-FMA parity step, so its total work is counted with it and differs
    from `N = 80` by under 0.01 %. Prediction bands unchanged. DRAM `T_alu_est` in Probe C does not add the parity
    step (5 FMAs per thread against thousands per cell); noted, not corrected.
11. **Hotspot stop for Probe D and the runner (owner, 2026-11-27):** raised from `90 C` to the study's standing `95 C` limit after the first rerun stopped Probe D at `90 C` (card went from `43 C` to `90 C` within one SAT-100 repetition). Probes B and C keep their in-probe `90 C` stop. Prediction bands unchanged.
