# Round 13 (P1): Compiled-Code Statistics of AMD MIT FidelityFX SDK v1.1.4 Shaders on RDNA 4 (`gfx1201`)

- **Date:** 2026-11-27 (draft)
- **Hardware & driver (`MEASURED`):** AMD Radeon RX 9070 XT (`1002:7550`, `gfx1201`), Mesa RADV `27.0.0-devel` (`radv`, Vulkan loader `1.4.341`), ACO compiler backend (`sg=[32,64]`).
- **Compiler & validator (`MEASURED`):** `glslc` (`shaderc v2026.1`), `spirv-val` (`SPIRV-Tools v2026.1`).
- **Upstream shader source (`READ`):** `GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`, tag `v1.1.4`, commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`, MIT license (`LICENSE.txt`, `sdk/LICENSE.txt`, `docs/license.md`; 5.2 MiB sparse checkout of shader headers, Vulkan GLSL entry points, CMake shader lists, and blob accessors kept outside this repository). No AMD source was copied into this repository; no closed AMD DLLs or leaked sources were touched.
- **Tools in this repo:** `src/round13/vk_pipestats.c` (compute pipeline creation only — zero dispatches, verified statically and via `0` amdgpu kernel faults; 41 C pre-run hotspot under shared lock), `scripts/round13/build_mit_stats.sh`, `scripts/round13/isa_classify.py`.
- **Labels:** `MEASURED` (compiled on this machine's RADV/ACO driver), `READ` (inspected in MIT SDK source), `INFERRED` (architectural implication from static ISA + prior `gfx1201` measurements).

> **Static vs. runtime warning:** Every instruction count and class share in this document is a **STATIC** count from the compiled ACO machine-code listing — **NOT** a dynamic execution count or runtime cost. Loops, divergent control flow, texture/L2/MALL misses, image/LDS atomic contention, and `s_wait_*` stalls can cause a small number of static instructions to dominate wall-clock execution time.

---

## Plain answers to the P1 question

1. **Does AMD's published analytic frame generation (FSR 3.1 Optical Flow + Frame Interpolation) compile to FP16 work at all on `gfx1201`? (`MEASURED` + `READ`)**
   - **No — 0.0% FP16 VALU across every single pass (0 packed `v_pk_*_f16` and 0 scalar `*_f16` instructions across all 7 Optical Flow passes and all 11 Frame Interpolation passes, in all 76 compiled pipelines).**
   - This holds not only for the SDK-dispatched permutations (`ffx_opticalflow_shaderblobs.cpp:72-196` and `ffx_frameinterpolation_shaderblobs.cpp:87-283` both leave `is16bit` unnamed/unused in all 18 pass getters), but **even when forcibly compiled with `-DFFX_HALF=1`**: every Optical Flow and Frame Interpolation shader produces **byte-for-byte identical ACO pipeline hashes and instruction counts** for `FFX_HALF=0` vs `FFX_HALF=1`.
   - **What analytic Frame Generation actually compiles to on RADV/ACO (`MEASURED`):**
     - **Optical Flow (`optical_flow_v5`) is overwhelmingly INT32 VALU (`1,365` INT32 instructions = `78.1%` of static instructions in wave64, `0` FP32, `0` FP16) + LDS (`2,048 B`, `26–30` `ds_*` ops), with ZERO `v_msad_u8` / `v_sad_*` instructions.** Source inspection (`READ`: `sdk/include/FidelityFX/gpu/opticalflow/ffx_opticalflow_callbacks_glsl.h:28-29`) shows why: AMD explicitly sets `#define FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION 0` ("no msad4 in glsl"), so the Vulkan GLSL shader emulates packed 8-bit SAD in software (`v_sub_nc_u32` x265, `v_max_i32` x260, `v_subrev_nc_u32` x256, `v_bfe_u32` x178, `v_add3_u32` x126). `scd_histogram` adds `10,240 B` LDS + `4` LDS atomics + `1` image atomic.
     - **Frame Interpolation is FP32 VALU + INT32 VALU + image loads/stores/atomics + LDS pyramids:** `interpolation` is `31.8%` FP32 (`720` insns w64), `13.8%` INT32, `72` `image_load`/`image_store`, `0%` FP16; `disocclusion_mask` is `33.4%` FP32 (`810` insns w64), `56` `image_load`/`image_store`, `8` `image_sample`; `game_motion_vector_field` emits `10` `image_atomic_*` and `optical_flow_vector_field` emits `8` `image_atomic_*`; `game_mv_inpainting_pyramid` and `inpainting_pyramid` each use `5,120 B` LDS (`40` `ds_*` ops) + `1` global buffer atomic.

2. **How does analytic Frame Generation compare with FidelityFX passes that DO use FP16? (`MEASURED` + `READ`)**
   - **Classic spatial/post-process/reflection/downsampling passes genuinely saturate packed FP16 (`v_pk_*_f16`) on RADV/ACO:**
     - **CAS 1.2 (`FFX_HALF=1`, w64):** `sharpen_only_srgb` reaches **`71.6%` FP16** (`72.7%` w32; `300 v_pk_*_f16 + 144 v_log/exp_f16`, `84.4%` of VALU); `scale_sharpen_linear` reaches **`44.7%` FP16** (`245 v_pk_*_f16 + 6 s16`); `sharpen_only_linear` reaches **`29.3%` FP16** (`60 v_pk_*_f16`).
     - **FSR 1.2 (`FFX_HALF=1`, w64):** `easu` reaches **`69.8%` FP16** (`72.0%` w32; `728 v_pk_*_f16 + 122 s16`, `89.3%` of VALU, plus `48 image_gather4_lz`), cutting static instructions by `-32.0%` (`1,789 -> 1,217`) and wave64 VGPRs `96 -> 72`. `rcas` reaches **`56.9%` FP16** (`58.5%` w32; `182 v_pk_*_f16 + 24 s16`, `0` FP32).
     - **LPM 1.4 (`FFX_HALF=1`, w64):** **`63.3%` FP16** (`63.8%` w32; `228 v_pk_*_f16 + 168 s16`, `0` FP32), cutting static instructions `-30.8%` (`904 -> 626`).
     - **SPD 2.2 (`FFX_HALF=1`, w64):** **`19.6–24.0%` FP16** (`126–168 v_pk_*_f16`, `0` FP32), cutting LDS allocation by `-40%` (`5,120 -> 3,072 B`) and halving `ds_*` LDS ops (`40 -> 21` in wave-DPP mode; `128 -> 65` in LDS-interop mode).
     - **Denoiser 1.3 Reflections (`FFX_HALF=1`, w64):** **`55.0%` FP16** in `resolve_temporal_reflections` (`266 pk16 + 305 s16`), **`40.7%`** in `prefilter_reflections`, **`39.3%`** in `reproject_reflections`.
     - **CACAO 1.4 (`FFX_HALF=1`, w64):** **`47.3%` FP16** in `upscale_bilateral_5x5` (`333 pk16 + 108 s16`) and **`23.6–36.0%`** across `edge_sensitive_blur_1..8` (cutting w64 VGPRs `120 -> 96`), while AO generation (`generate_q0..q3`) stays `0.0%` FP16.
   - **FSR 3.1 Upscaler (`FFX_HALF=1`) uses far less FP16 than expected from marketing summaries (`MEASURED` + `READ`):**
     - Only **one** SDK-dispatched pass (`accumulate` / `accumulate_sharpen`) actually executes FP16 arithmetic on PC/Vulkan: **`8.8%` (w32) / `7.9%` (w64)** pure FP16 (`68 v_pk_*_f16 + 70–72 scalar *_f16`), plus **`92` `v_fma_mix_f32` (`5.2%`)** fusing FP16 inputs into FP32 accumulation — while pure FP32 VALU remains `50.6%` (w32) / `56.7%` (w64). Source (`READ`: `CMakeCompileFSR3UpscalerShaders.txt:26-30`) explains why: AMD hardcoded `# Only reprojection is to do half for now` (`UPSAMPLE/ACCUMULATE/POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0`, only `REPROJECT_SAMPLERS_USE_DATA_HALF=1`). Even so, `FFX_HALF=1` trims `accumulate` w64 VGPRs from `96 -> 72`.
     - The other three passes that select `_16bit` blobs (`shading_change`, `prepare_reactivity`, `autogen_reactive`) compile to **`0.0%` FP16 VALU** (`autogen_reactive` adds only `2` conversions; the other two are byte-for-byte identical to `FFX_HALF=0`).
     - `rcas` in FSR 3.1 gates its 16-bit path behind `#if FFX_HALF && defined(__XBOX_SCARLETT)` (`ffx_fsr3upscaler_rcas.h:28`), leaving PC/Vulkan at **`0.0%` FP16**.
     - `luma_pyramid` and `shading_change_pyramid` have empty `#if FFX_HALF` stubs (`SpdStoreH {}` in `ffx_fsr3upscaler_luma_pyramid.h:136-148`), which dead-code-eliminate to 306 instructions if forced to `FFX_HALF=1` — explaining why `ffx_fsr3upscaler_shaderblobs.cpp:235,289` ignores `is16bit` for them.

3. **Cross-cutting RDNA 4 (`gfx1201`) RADV/ACO findings (`MEASURED`):**
   - **Wave32 VOPD vs. Wave64 VGPR pressure:** ACO emits `v_dual_*` (VOPD dual-issue) **exclusively in Wave32** (`0` VOPD lines across all `182` Wave64 pipelines; up to `392` VOPD lines in `fsr1__easu__fp32__w32`, `298` in `fsr3upscaler__shading_change_pyramid__fp32__w32`, `277` in `fsr3upscaler__accumulate__fp32__w32`). Conversely, Wave64 frequently cuts VGPR allocation in half (e.g., `optical_flow_v5` `96 -> 48`, `fsr1__rcas` `96 -> 48`, `cas` `96 -> 48`, `parallelsort__scatter` `96 -> 48`, `fsr3upscaler__accumulate__fp16` `96 -> 72`) and doubles `Subgroups per SIMD` where VGPR-bound.
   - **Zero spills / zero WMMA / zero `v_dot*` / zero `v_msad_u8`:** Across all `364` MIT v1.1.4 Vulkan pipelines: `Scratch size = 0`, `Spilled VGPRs = 0`, `Spilled SGPRs = 0`, `wmma = 0`, `dot = 0`, `msad_sad = 0`, `unknown_ops = 0`.
   - **Negative FP16 code-size case (`blur`):** In `blur_k3` and `blur_k15`, `FFX_HALF=1` emits `10.3–18.7%` FP16 VALU, but adds `41–196` `v_cvt_*` / `v_pack_b32_f16` conversion instructions (`+303` static instructions on `blur_k15` w64: `1,315 -> 1,618`; `+390` in w32: `1,226 -> 1,616`) with no VGPR or LDS reduction.

---

## Method, controls, and preregistered expectations scorecard

### Build & validation pipeline
- Every GLSL compute shader under `sdk/src/backends/vk/shaders/<component>/` was compiled with `glslc -fshader-stage=comp --target-env=vulkan1.2 -Os -DFFX_GLSL=1 -DFFX_GPU=1 -DFFX_HALF={0,1}` plus the exact component defines from `sdk/include/FidelityFX/gpu/<comp>/CMakeCompile*Shaders.txt` (cited in `scripts/round13/build_mit_stats.sh`).
- Every SPIR-V binary passed `spirv-val --target-env vulkan1.2` (`182/182 PASS`).
- In FidelityFX's Vulkan backend (`READ`: `sdk/src/backends/vk/ffx_vk.cpp:3625-3633`), Wave64 is selected at pipeline creation via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (`requiredSubgroupSize = 64`) on the same SPIR-V binary. `src/round13/vk_pipestats.c` reflects SPIR-V descriptor bindings directly, creates each compute pipeline with `VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR` for both `req_sg=32` and `req_sg=64`, captures all 26 RADV/ACO statistics + AMD GCN ISA disassembly, and never creates command buffers or dispatches (`vkCmdDispatch` absent, statically verified).
- **Controls & deliberate breaks (`MEASURED`):**
  - `isa_classify.py --self-test`: 33 synthetic instructions covering all 15 classes + ACO `(then repeated N times)` expansion -> `15/15` exact, `other=0`, deliberate wrong-label check (`v_msad_u8` labeled as `packed_fp16_valu`) caught (`FAIL` as expected).
  - `vk_pipestats` CPU reflection break + GPU pipeline break (`--entry nonexistent_ep`): both caught (`FAIL` as expected).
  - Cross-check against RADV/ACO's own `Instructions` statistic across all `364` pipelines: `347/364` exact match (`0` difference), `0` undercounts, remaining `17` differ by `+1..+10` because ACO's `Instructions` count omits most `s_nop` lines (on 13 of the 17 the expanded `s_nop` count equals the difference; on `blur__blur_k15__fp32__w64`, `blur__blur_k15__fp16__w64`, `parallelsort__sum__fp32__w64` and `parallelsort__sum__fp16__w64` one `s_nop` sits inside ACO's count, and the nops are not only at the end of the listing). `req_sg == sg_actual` on `364/364` pipelines. `0` amdgpu kernel faults.

### Preregistered expectations scorecard (written pre-run in scratch `expectations.md`)

| Expectation (pre-run, `READ`) | Measured (`MEASURED`) | Verdict |
|---|---|---|
| Optical Flow: `0.0%` FP16 in all 7 passes (`fp32` & `fp16`), `0` `v_msad_u8` (GLSL fallback to INT32+LDS) | `0.0%` FP16 across all 32 pipelines; `0` `v_msad_u8`; `optical_flow_v5` is `78.1%` INT32 (`1,365` insns) | **Confirmed** |
| Frame Interpolation: `0.0%` FP16 across all 11 SDK passes (`<2%` even if forced `FFX_HALF=1`) | `0.0%` FP16 across all 44 pipelines (byte-identical for `FFX_HALF=0` vs `1`); `10`/`8` image atomics in MV field passes | **Confirmed** |
| FSR 3.1 Upscaler: `>=15%` FP16 in the 4 `_16bit`-selected passes (`accumulate`, `prepare_reactivity`, `shading_change`, `autogen_reactive`) | **Missed magnitude:** only `accumulate` uses FP16 (`7.9–8.8%` pure FP16 + `5.2%` `v_fma_mix_f32`; w64 VGPR `96->72`); the other 3 compile to **`0.0%` FP16** because `CMakeCompileFSR3UpscalerShaders.txt:26-30` disables `DATA_HALF` outside reprojection | **Refuted high-FP16 assumption; explained by CMake defines** |
| FSR 1 / CAS / SPD (`FFX_HALF=1`): high packed FP16 (`>=20%` FSR1/CAS, `>=10%` SPD), `image_gather4` in EASU, fewer `ds_*` in SPD wave mode | CAS `29.3–72.7%`, FSR1 EASU `69.8–72.0%` (`48 gather4`), RCAS `56.9–58.5%`, SPD `19.6–24.3%` (`21 vs 65 ds_*`, LDS `5120->3072 B`) | **Confirmed** |
| Group 2: high FP16 in Blur/LPM/Denoiser/CACAO-blur, low in SSSR, `0%` FP16/FP32 in Parallel Sort | LPM `63.3%`, Denoiser refl `39.3–56.4%`, CACAO bilateral/blur `23.6–47.3%`, Blur `10.3–18.7%` (with heavy `cvt` overhead), SSSR `0–0.9%`, Parallel Sort `0%` FP16 (`<=1` FP32 op in `reduce`/`scan_add`); discovered Denoiser shadow filters stub out `FFX_HALF=0` | **Mostly confirmed; discovered Blur `cvt` cost & Denoiser `FFX_HALF=0` stub** |
| Classifier `unknown_ops == {}`, `sg_actual == req_sg`, `0` faults | `unknown_ops == {}`, `364/364` `sg_actual == req_sg`, `0` faults | **Confirmed** |

---

## Group 1 Components (Full Pass x Permutation Tables, `MEASURED`)

*Column key: `Wave` = actual subgroup size (`sg_actual`); `VGPR`/`SGPR`/`LDS (B)` = RADV/ACO pipeline stats (`Scratch = 0` and `Spilled VGPRs = 0` on every pipeline); `Static Insns` = classified static instructions (`our_n`); `pk16` = `v_pk_*_f16`; `s16` = scalar-lane `*_f16`; `FP16%` = `(pk16 + s16) / Static Insns`; `tex/img%` = `image_sample + image_gather4 + image_load_store`; `mem/lds%` = `buffer/global/flat/scratch + ds_*`; `salu/wb%` = `scalar + waits_branches`; `VOPD` = dual-issue `v_dual_*` lines (wave32 only).*

### 1. FSR 3.1 Upscaler (`fsr3upscaler`, 10 passes + `accumulate_sharpen`, 44 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/fsr3upscaler/CMakeCompileFSR3UpscalerShaders.txt:23-42` (`UPSAMPLE/ACCUMULATE/POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0`, `REPROJECT_SAMPLERS_USE_DATA_HALF=1`, `UPSAMPLE_USE_LANCZOS_TYPE=2`, `REPROJECT_USE_LANCZOS_TYPE=0`, `HDR=1`, `LOW_RES_MV=1`, `JITTERED_MV=0`, `INVERTED_DEPTH=1`, `APPLY_SHARPENING={0,1}`). Blob selection (`READ`): `ffx_fsr3upscaler_shaderblobs.cpp:85-341`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prepare_inputs` | `fp32` | 32 | 48 | 108 | 0 | 222 | 0 | 0 | **0.0%** | 24.8% | 18.0% | 1.8% | 8.6% | 0.0% | 46.8% | 23 |
| `prepare_inputs` | `fp32` | 64 | 48 | 108 | 0 | 248 | 0 | 0 | **0.0%** | 26.6% | 21.0% | 1.6% | 7.7% | 0.0% | 43.1% | 0 |
| `prepare_inputs` | `fp16`* | 32 | 48 | 108 | 0 | 222 | 0 | 0 | **0.0%** | 24.8% | 18.0% | 1.8% | 8.6% | 0.0% | 46.8% | 23 |
| `prepare_inputs` | `fp16`* | 64 | 48 | 108 | 0 | 248 | 0 | 0 | **0.0%** | 26.6% | 21.0% | 1.6% | 7.7% | 0.0% | 43.1% | 0 |
| `luma_pyramid` | `fp32` | 32 | 48 | 108 | 5120 | 1605 | 0 | 0 | **0.0%** | 27.6% | 16.4% | 0.0% | 6.0% | 2.4% | 47.6% | 89 |
| `luma_pyramid` | `fp32` | 64 | 36 | 108 | 5120 | 1674 | 0 | 0 | **0.0%** | 30.2% | 17.3% | 0.0% | 5.8% | 2.3% | 44.4% | 0 |
| `luma_pyramid` | `fp16`* | 32 | 24 | 108 | 3072 | 306 | 48 | 0 | **15.7%** | 0.0% | 36.6% | 0.0% | 0.7% | 6.2% | 40.8% | 2 |
| `luma_pyramid` | `fp16`* | 64 | 24 | 108 | 3072 | 308 | 48 | 0 | **15.6%** | 0.0% | 37.0% | 0.0% | 0.6% | 6.2% | 40.6% | 0 |
| `shading_change_pyramid` | `fp32` | 32 | 72 | 108 | 5120 | 6316 | 0 | 0 | **0.0%** | 32.0% | 25.0% | 0.8% | 3.3% | 0.6% | 38.4% | 298 |
| `shading_change_pyramid` | `fp32` | 64 | 60 | 108 | 5120 | 6649 | 0 | 0 | **0.0%** | 34.4% | 24.2% | 0.7% | 3.1% | 0.6% | 37.0% | 0 |
| `shading_change_pyramid` | `fp16`* | 32 | 24 | 108 | 3072 | 306 | 48 | 0 | **15.7%** | 0.0% | 36.6% | 0.0% | 0.7% | 6.2% | 40.8% | 2 |
| `shading_change_pyramid` | `fp16`* | 64 | 24 | 108 | 3072 | 308 | 48 | 0 | **15.6%** | 0.0% | 37.0% | 0.0% | 0.6% | 6.2% | 40.6% | 0 |
| `shading_change` | `fp32` | 32 | 24 | 108 | 0 | 108 | 0 | 0 | **0.0%** | 20.4% | 8.3% | 1.9% | 3.7% | 0.0% | 65.7% | 4 |
| `shading_change` | `fp32` | 64 | 12 | 108 | 0 | 113 | 0 | 0 | **0.0%** | 22.1% | 8.8% | 1.8% | 3.5% | 0.0% | 63.7% | 0 |
| `shading_change` | `fp16` | 32 | 24 | 108 | 0 | 108 | 0 | 0 | **0.0%** | 20.4% | 8.3% | 1.9% | 3.7% | 0.0% | 65.7% | 4 |
| `shading_change` | `fp16` | 64 | 12 | 108 | 0 | 113 | 0 | 0 | **0.0%** | 22.1% | 8.8% | 1.8% | 3.5% | 0.0% | 63.7% | 0 |
| `prepare_reactivity` | `fp32` | 32 | 48 | 108 | 0 | 619 | 0 | 0 | **0.0%** | 33.4% | 10.5% | 1.3% | 5.3% | 0.0% | 49.4% | 42 |
| `prepare_reactivity` | `fp32` | 64 | 48 | 108 | 0 | 667 | 0 | 0 | **0.0%** | 35.7% | 11.4% | 1.2% | 4.9% | 0.0% | 46.8% | 0 |
| `prepare_reactivity` | `fp16` | 32 | 48 | 108 | 0 | 619 | 0 | 0 | **0.0%** | 33.4% | 10.5% | 1.3% | 5.3% | 0.0% | 49.4% | 42 |
| `prepare_reactivity` | `fp16` | 64 | 48 | 108 | 0 | 667 | 0 | 0 | **0.0%** | 35.7% | 11.4% | 1.2% | 4.9% | 0.0% | 46.8% | 0 |
| `luma_instability` | `fp32` | 32 | 48 | 108 | 0 | 205 | 0 | 0 | **0.0%** | 39.5% | 8.3% | 1.0% | 3.4% | 0.0% | 47.8% | 27 |
| `luma_instability` | `fp32` | 64 | 36 | 108 | 0 | 233 | 0 | 0 | **0.0%** | 43.3% | 10.3% | 0.9% | 3.0% | 0.0% | 42.5% | 0 |
| `luma_instability` | `fp16`* | 32 | 48 | 108 | 0 | 205 | 0 | 0 | **0.0%** | 39.5% | 8.3% | 1.0% | 3.4% | 0.0% | 47.8% | 27 |
| `luma_instability` | `fp16`* | 64 | 36 | 108 | 0 | 233 | 0 | 0 | **0.0%** | 43.3% | 10.3% | 0.9% | 3.0% | 0.0% | 42.5% | 0 |
| `accumulate` | `fp32` | 32 | 96 | 108 | 0 | 1586 | 0 | 0 | **0.0%** | 57.1% | 6.7% | 0.5% | 2.2% | 0.0% | 33.5% | 277 |
| `accumulate` | `fp32` | 64 | 96 | 108 | 0 | 1818 | 0 | 0 | **0.0%** | 64.1% | 6.7% | 0.4% | 1.9% | 0.0% | 26.8% | 0 |
| `accumulate` | `fp16` | 32 | 96 | 108 | 0 | 1576 | 68 | 70 | **8.8%** | 50.6% | 5.6% | 1.1% | 2.2% | 0.0% | 31.7% | 216 |
| `accumulate` | `fp16` | 64 | 72 | 108 | 0 | 1775 | 68 | 72 | **7.9%** | 56.7% | 5.8% | 0.5% | 2.0% | 0.0% | 27.2% | 0 |
| `accumulate_sharpen` | `fp32` | 32 | 96 | 108 | 0 | 1581 | 0 | 0 | **0.0%** | 57.2% | 6.6% | 0.5% | 2.2% | 0.0% | 33.6% | 276 |
| `accumulate_sharpen` | `fp32` | 64 | 96 | 108 | 0 | 1813 | 0 | 0 | **0.0%** | 64.3% | 6.5% | 0.4% | 1.9% | 0.0% | 26.9% | 0 |
| `accumulate_sharpen` | `fp16` | 32 | 96 | 108 | 0 | 1571 | 68 | 70 | **8.8%** | 50.7% | 5.5% | 1.1% | 2.2% | 0.0% | 31.8% | 215 |
| `accumulate_sharpen` | `fp16` | 64 | 72 | 108 | 0 | 1770 | 68 | 72 | **7.9%** | 56.8% | 5.6% | 0.5% | 1.9% | 0.0% | 27.3% | 0 |
| `rcas` | `fp32` | 32 | 96 | 108 | 0 | 416 | 0 | 0 | **0.0%** | 66.3% | 5.8% | 0.0% | 6.0% | 0.0% | 21.9% | 134 |
| `rcas` | `fp32` | 64 | 48 | 108 | 0 | 552 | 0 | 0 | **0.0%** | 73.4% | 6.2% | 0.0% | 4.5% | 0.0% | 15.9% | 0 |
| `rcas` | `fp16`* | 32 | 96 | 108 | 0 | 432 | 0 | 0 | **0.0%** | 63.9% | 5.6% | 1.9% | 5.8% | 0.0% | 22.9% | 134 |
| `rcas` | `fp16`* | 64 | 60 | 108 | 0 | 547 | 0 | 0 | **0.0%** | 74.0% | 5.5% | 1.5% | 4.6% | 0.0% | 14.4% | 0 |
| `autogen_reactive` | `fp32` | 32 | 24 | 108 | 0 | 77 | 0 | 0 | **0.0%** | 40.3% | 9.1% | 0.0% | 3.9% | 0.0% | 46.8% | 7 |
| `autogen_reactive` | `fp32` | 64 | 12 | 108 | 0 | 83 | 0 | 0 | **0.0%** | 45.8% | 8.4% | 0.0% | 3.6% | 0.0% | 42.2% | 0 |
| `autogen_reactive` | `fp16` | 32 | 24 | 108 | 0 | 79 | 0 | 0 | **0.0%** | 39.2% | 8.9% | 2.5% | 3.8% | 0.0% | 45.6% | 7 |
| `autogen_reactive` | `fp16` | 64 | 12 | 108 | 0 | 85 | 0 | 0 | **0.0%** | 44.7% | 8.2% | 2.4% | 3.5% | 0.0% | 41.2% | 0 |
| `debug_view` | `fp32` | 32 | 24 | 108 | 0 | 349 | 0 | 0 | **0.0%** | 18.6% | 8.9% | 3.4% | 3.4% | 0.0% | 65.6% | 27 |
| `debug_view` | `fp32` | 64 | 12 | 108 | 0 | 377 | 0 | 0 | **0.0%** | 23.3% | 9.3% | 3.2% | 3.2% | 0.0% | 61.0% | 0 |
| `debug_view` | `fp16`* | 32 | 24 | 108 | 0 | 349 | 0 | 0 | **0.0%** | 18.6% | 8.9% | 3.4% | 3.4% | 0.0% | 65.6% | 27 |
| `debug_view` | `fp16`* | 64 | 12 | 108 | 0 | 377 | 0 | 0 | **0.0%** | 23.3% | 9.3% | 3.2% | 3.2% | 0.0% | 61.0% | 0 |

*Rows marked `*` (`fp16*`) are built with `-DFFX_HALF=1` as a counterfactual check; the SDK's `*_shaderblobs.cpp` ignores `is16bit` for that pass and never dispatches the 16-bit blob.*

*Notes (`READ` + `MEASURED`):*
- `luma_pyramid` and `shading_change_pyramid` `fp16*` rows (`306–308` insns, `48 pk16`) are dead-code-eliminated shells from empty `#if FFX_HALF` stubs (`SpdStoreH {}` in `ffx_fsr3upscaler_luma_pyramid.h:136-148`) and are never selected by `ffx_fsr3upscaler_shaderblobs.cpp`.
- `accumulate` / `accumulate_sharpen` (`fp16` w64) also contain `92` `v_fma_mix_f32` (`5.2%`, counted in `fp32%`) fusing FP16 inputs into FP32 accumulation (vs `0` in `fp32`). `prepare_inputs` emits `4` `image_atomic_*`; `luma_pyramid` and `shading_change_pyramid` emit `1` `image_atomic_*` each.

### 2. FSR 3.1 Optical Flow (`opticalflow`, 7 passes + SDR/HDR `prepare_luma`, 32 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/opticalflow/CMakeCompileOpticalflowShaders.txt:23-28` (`HDR_COLOR_INPUT={0,1}`). Blob selection (`READ`): `ffx_opticalflow_shaderblobs.cpp:72-196` (ignores `is16bit` in all 7 passes).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prepare_luma_hdr` | `fp32` | 32 | 24 | 108 | 0 | 457 | 0 | 0 | **0.0%** | 49.9% | 4.4% | 0.9% | 1.8% | 0.0% | 43.1% | 16 |
| `prepare_luma_hdr` | `fp32` | 64 | 12 | 108 | 0 | 469 | 0 | 0 | **0.0%** | 52.0% | 4.3% | 0.9% | 1.7% | 0.0% | 41.2% | 0 |
| `prepare_luma_hdr` | `fp16`* | 32 | 24 | 108 | 0 | 457 | 0 | 0 | **0.0%** | 49.9% | 4.4% | 0.9% | 1.8% | 0.0% | 43.1% | 16 |
| `prepare_luma_hdr` | `fp16`* | 64 | 12 | 108 | 0 | 469 | 0 | 0 | **0.0%** | 52.0% | 4.3% | 0.9% | 1.7% | 0.0% | 41.2% | 0 |
| `prepare_luma_sdr` | `fp32` | 32 | 24 | 108 | 0 | 457 | 0 | 0 | **0.0%** | 49.9% | 4.4% | 0.9% | 1.8% | 0.0% | 43.1% | 16 |
| `prepare_luma_sdr` | `fp32` | 64 | 12 | 108 | 0 | 469 | 0 | 0 | **0.0%** | 52.0% | 4.3% | 0.9% | 1.7% | 0.0% | 41.2% | 0 |
| `prepare_luma_sdr` | `fp16`* | 32 | 24 | 108 | 0 | 457 | 0 | 0 | **0.0%** | 49.9% | 4.4% | 0.9% | 1.8% | 0.0% | 43.1% | 16 |
| `prepare_luma_sdr` | `fp16`* | 64 | 12 | 108 | 0 | 469 | 0 | 0 | **0.0%** | 52.0% | 4.3% | 0.9% | 1.7% | 0.0% | 41.2% | 0 |
| `luminance_pyramid` | `fp32` | 32 | 48 | 108 | 4096 | 333 | 0 | 0 | **0.0%** | 24.3% | 18.9% | 8.4% | 8.4% | 6.3% | 33.6% | 8 |
| `luminance_pyramid` | `fp32` | 64 | 36 | 108 | 4096 | 338 | 0 | 0 | **0.0%** | 24.9% | 20.1% | 8.3% | 8.3% | 6.2% | 32.2% | 0 |
| `luminance_pyramid` | `fp16`* | 32 | 48 | 108 | 4096 | 333 | 0 | 0 | **0.0%** | 24.3% | 18.9% | 8.4% | 8.4% | 6.3% | 33.6% | 8 |
| `luminance_pyramid` | `fp16`* | 64 | 36 | 108 | 4096 | 338 | 0 | 0 | **0.0%** | 24.9% | 20.1% | 8.3% | 8.3% | 6.2% | 32.2% | 0 |
| `scd_histogram` | `fp32` | 32 | 24 | 108 | 10240 | 123 | 0 | 0 | **0.0%** | 0.0% | 33.3% | 0.0% | 4.1% | 9.8% | 52.8% | 3 |
| `scd_histogram` | `fp32` | 64 | 12 | 108 | 10240 | 125 | 0 | 0 | **0.0%** | 0.0% | 36.0% | 0.0% | 4.0% | 9.6% | 50.4% | 0 |
| `scd_histogram` | `fp16`* | 32 | 24 | 108 | 10240 | 123 | 0 | 0 | **0.0%** | 0.0% | 33.3% | 0.0% | 4.1% | 9.8% | 52.8% | 3 |
| `scd_histogram` | `fp16`* | 64 | 12 | 108 | 10240 | 125 | 0 | 0 | **0.0%** | 0.0% | 36.0% | 0.0% | 4.0% | 9.6% | 50.4% | 0 |
| `scd_divergence` | `fp32` | 32 | 24 | 108 | 5120 | 383 | 0 | 0 | **0.0%** | 11.7% | 21.7% | 0.3% | 4.2% | 14.4% | 47.8% | 11 |
| `scd_divergence` | `fp32` | 64 | 24 | 108 | 5120 | 394 | 0 | 0 | **0.0%** | 11.7% | 23.6% | 0.3% | 4.1% | 14.0% | 46.4% | 0 |
| `scd_divergence` | `fp16`* | 32 | 24 | 108 | 5120 | 383 | 0 | 0 | **0.0%** | 11.7% | 21.7% | 0.3% | 4.2% | 14.4% | 47.8% | 11 |
| `scd_divergence` | `fp16`* | 64 | 24 | 108 | 5120 | 394 | 0 | 0 | **0.0%** | 11.7% | 23.6% | 0.3% | 4.1% | 14.0% | 46.4% | 0 |
| `optical_flow_v5` | `fp32` | 32 | 96 | 108 | 2048 | 1757 | 0 | 0 | **0.0%** | 0.0% | 78.1% | 0.0% | 0.9% | 1.7% | 19.3% | 4 |
| `optical_flow_v5` | `fp32` | 64 | 48 | 108 | 2048 | 1748 | 0 | 0 | **0.0%** | 0.0% | 78.1% | 0.0% | 0.9% | 1.5% | 19.5% | 0 |
| `optical_flow_v5` | `fp16`* | 32 | 96 | 108 | 2048 | 1757 | 0 | 0 | **0.0%** | 0.0% | 78.1% | 0.0% | 0.9% | 1.7% | 19.3% | 4 |
| `optical_flow_v5` | `fp16`* | 64 | 48 | 108 | 2048 | 1748 | 0 | 0 | **0.0%** | 0.0% | 78.1% | 0.0% | 0.9% | 1.5% | 19.5% | 0 |
| `filter_optical_flow` | `fp32` | 32 | 96 | 108 | 0 | 435 | 0 | 0 | **0.0%** | 0.0% | 90.8% | 0.0% | 2.3% | 0.0% | 6.9% | 6 |
| `filter_optical_flow` | `fp32` | 64 | 48 | 108 | 0 | 441 | 0 | 0 | **0.0%** | 0.0% | 90.9% | 0.0% | 2.3% | 0.0% | 6.8% | 0 |
| `filter_optical_flow` | `fp16`* | 32 | 96 | 108 | 0 | 435 | 0 | 0 | **0.0%** | 0.0% | 90.8% | 0.0% | 2.3% | 0.0% | 6.9% | 6 |
| `filter_optical_flow` | `fp16`* | 64 | 48 | 108 | 0 | 441 | 0 | 0 | **0.0%** | 0.0% | 90.9% | 0.0% | 2.3% | 0.0% | 6.8% | 0 |
| `scale_optical_flow` | `fp32` | 32 | 48 | 108 | 1024 | 562 | 0 | 0 | **0.0%** | 0.0% | 54.4% | 0.0% | 4.3% | 1.8% | 39.5% | 0 |
| `scale_optical_flow` | `fp32` | 64 | 36 | 108 | 1024 | 571 | 0 | 0 | **0.0%** | 0.0% | 53.4% | 0.0% | 4.2% | 1.8% | 40.6% | 0 |
| `scale_optical_flow` | `fp16`* | 32 | 48 | 108 | 1024 | 562 | 0 | 0 | **0.0%** | 0.0% | 54.4% | 0.0% | 4.3% | 1.8% | 39.5% | 0 |
| `scale_optical_flow` | `fp16`* | 64 | 36 | 108 | 1024 | 571 | 0 | 0 | **0.0%** | 0.0% | 53.4% | 0.0% | 4.2% | 1.8% | 40.6% | 0 |

*Rows marked `*` (`fp16*`) are built with `-DFFX_HALF=1` as a counterfactual check; the SDK's `*_shaderblobs.cpp` ignores `is16bit` for that pass and never dispatches the 16-bit blob.*

*Notes (`READ` + `MEASURED`):*
- Every pass is **0.0% FP16** and **byte-for-byte identical** between `fp32` and `fp16*`.
- `optical_flow_v5` emits **0 `v_msad_u8` / `v_sad_*`** because `sdk/include/FidelityFX/gpu/opticalflow/ffx_opticalflow_callbacks_glsl.h:28-29` sets `#define FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION 0` for GLSL; ACO emits `1,365–1,372` INT32 VALU ops (`78.1%`), dominated by `v_sub_nc_u32` (`265`), `v_max_i32` (`260`), `v_subrev_nc_u32` (`256`), `v_bfe_u32` (`178`), `v_add3_u32` (`126`).
- `scd_histogram` emits `4` LDS atomics (`ds_*`) + `1` `image_atomic_*` (`10,240 B` LDS); `scd_divergence` emits `2` `image_atomic_*`.

### 3. FSR 3.1 Frame Interpolation (`frameinterpolation`, 11 passes, 44 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/frameinterpolation/CMakeCompileFrameinterpolationShaders.txt:23-35` (`LOW_RES_MOTION_VECTORS=1`, `JITTER_MOTION_VECTORS=0`, `INVERTED_DEPTH=1`, plus base sampler flags). Blob selection (`READ`): `ffx_frameinterpolation_shaderblobs.cpp:87-283` (ignores `is16bit` in all 11 passes).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reconstruct_and_dilate` | `fp32` | 32 | 48 | 108 | 0 | 183 | 0 | 0 | **0.0%** | 18.0% | 21.9% | 2.2% | 8.7% | 0.0% | 49.2% | 13 |
| `reconstruct_and_dilate` | `fp32` | 64 | 36 | 108 | 0 | 196 | 0 | 0 | **0.0%** | 20.4% | 23.5% | 2.0% | 8.2% | 0.0% | 45.9% | 0 |
| `reconstruct_and_dilate` | `fp16`* | 32 | 48 | 108 | 0 | 183 | 0 | 0 | **0.0%** | 18.0% | 21.9% | 2.2% | 8.7% | 0.0% | 49.2% | 13 |
| `reconstruct_and_dilate` | `fp16`* | 64 | 36 | 108 | 0 | 196 | 0 | 0 | **0.0%** | 20.4% | 23.5% | 2.0% | 8.2% | 0.0% | 45.9% | 0 |
| `setup` | `fp32` | 32 | 24 | 108 | 0 | 55 | 0 | 0 | **0.0%** | 0.0% | 21.8% | 0.0% | 10.9% | 7.3% | 60.0% | 1 |
| `setup` | `fp32` | 64 | 12 | 108 | 0 | 56 | 0 | 0 | **0.0%** | 0.0% | 23.2% | 0.0% | 10.7% | 7.1% | 58.9% | 0 |
| `setup` | `fp16`* | 32 | 24 | 108 | 0 | 55 | 0 | 0 | **0.0%** | 0.0% | 21.8% | 0.0% | 10.9% | 7.3% | 60.0% | 1 |
| `setup` | `fp16`* | 64 | 12 | 108 | 0 | 56 | 0 | 0 | **0.0%** | 0.0% | 23.2% | 0.0% | 10.7% | 7.1% | 58.9% | 0 |
| `reconstruct_previous_depth` | `fp32` | 32 | 24 | 108 | 0 | 105 | 0 | 0 | **0.0%** | 19.0% | 11.4% | 5.7% | 6.7% | 0.0% | 57.1% | 6 |
| `reconstruct_previous_depth` | `fp32` | 64 | 12 | 108 | 0 | 110 | 0 | 0 | **0.0%** | 23.6% | 10.9% | 5.5% | 6.4% | 0.0% | 53.6% | 0 |
| `reconstruct_previous_depth` | `fp16`* | 32 | 24 | 108 | 0 | 105 | 0 | 0 | **0.0%** | 19.0% | 11.4% | 5.7% | 6.7% | 0.0% | 57.1% | 6 |
| `reconstruct_previous_depth` | `fp16`* | 64 | 12 | 108 | 0 | 110 | 0 | 0 | **0.0%** | 23.6% | 10.9% | 5.5% | 6.4% | 0.0% | 53.6% | 0 |
| `game_motion_vector_field` | `fp32` | 32 | 48 | 108 | 0 | 573 | 0 | 0 | **0.0%** | 43.8% | 6.3% | 2.3% | 2.6% | 0.0% | 45.0% | 27 |
| `game_motion_vector_field` | `fp32` | 64 | 36 | 108 | 0 | 585 | 0 | 0 | **0.0%** | 47.0% | 6.7% | 2.2% | 2.6% | 0.0% | 41.5% | 0 |
| `game_motion_vector_field` | `fp16`* | 32 | 48 | 108 | 0 | 573 | 0 | 0 | **0.0%** | 43.8% | 6.3% | 2.3% | 2.6% | 0.0% | 45.0% | 27 |
| `game_motion_vector_field` | `fp16`* | 64 | 36 | 108 | 0 | 585 | 0 | 0 | **0.0%** | 47.0% | 6.7% | 2.2% | 2.6% | 0.0% | 41.5% | 0 |
| `game_mv_inpainting_pyramid` | `fp32` | 32 | 96 | 108 | 5120 | 1431 | 0 | 0 | **0.0%** | 37.6% | 22.9% | 4.5% | 4.8% | 2.9% | 27.3% | 235 |
| `game_mv_inpainting_pyramid` | `fp32` | 64 | 48 | 108 | 5120 | 1701 | 0 | 0 | **0.0%** | 41.4% | 25.2% | 1.9% | 4.1% | 2.5% | 25.0% | 0 |
| `game_mv_inpainting_pyramid` | `fp16`* | 32 | 96 | 108 | 5120 | 1431 | 0 | 0 | **0.0%** | 37.6% | 22.9% | 4.5% | 4.8% | 2.9% | 27.3% | 235 |
| `game_mv_inpainting_pyramid` | `fp16`* | 64 | 48 | 108 | 5120 | 1701 | 0 | 0 | **0.0%** | 41.4% | 25.2% | 1.9% | 4.1% | 2.5% | 25.0% | 0 |
| `optical_flow_vector_field` | `fp32` | 32 | 72 | 108 | 0 | 712 | 0 | 0 | **0.0%** | 50.0% | 3.9% | 3.7% | 2.7% | 0.0% | 39.7% | 93 |
| `optical_flow_vector_field` | `fp32` | 64 | 48 | 108 | 0 | 787 | 0 | 0 | **0.0%** | 55.9% | 4.7% | 3.3% | 2.4% | 0.0% | 33.7% | 0 |
| `optical_flow_vector_field` | `fp16`* | 32 | 72 | 108 | 0 | 712 | 0 | 0 | **0.0%** | 50.0% | 3.9% | 3.7% | 2.7% | 0.0% | 39.7% | 93 |
| `optical_flow_vector_field` | `fp16`* | 64 | 48 | 108 | 0 | 787 | 0 | 0 | **0.0%** | 55.9% | 4.7% | 3.3% | 2.4% | 0.0% | 33.7% | 0 |
| `disocclusion_mask` | `fp32` | 32 | 48 | 108 | 0 | 2283 | 0 | 0 | **0.0%** | 29.3% | 10.5% | 2.7% | 2.8% | 0.0% | 54.8% | 159 |
| `disocclusion_mask` | `fp32` | 64 | 36 | 108 | 0 | 2424 | 0 | 0 | **0.0%** | 33.4% | 10.7% | 2.4% | 2.6% | 0.0% | 50.8% | 0 |
| `disocclusion_mask` | `fp16`* | 32 | 48 | 108 | 0 | 2283 | 0 | 0 | **0.0%** | 29.3% | 10.5% | 2.7% | 2.8% | 0.0% | 54.8% | 159 |
| `disocclusion_mask` | `fp16`* | 64 | 36 | 108 | 0 | 2424 | 0 | 0 | **0.0%** | 33.4% | 10.7% | 2.4% | 2.6% | 0.0% | 50.8% | 0 |
| `interpolation` | `fp32` | 32 | 48 | 108 | 0 | 2048 | 0 | 0 | **0.0%** | 26.6% | 13.2% | 2.0% | 3.6% | 0.0% | 54.6% | 215 |
| `interpolation` | `fp32` | 64 | 36 | 108 | 0 | 2262 | 0 | 0 | **0.0%** | 31.8% | 13.8% | 1.7% | 3.2% | 0.0% | 49.4% | 0 |
| `interpolation` | `fp16`* | 32 | 48 | 108 | 0 | 2048 | 0 | 0 | **0.0%** | 26.6% | 13.2% | 2.0% | 3.6% | 0.0% | 54.6% | 215 |
| `interpolation` | `fp16`* | 64 | 36 | 108 | 0 | 2262 | 0 | 0 | **0.0%** | 31.8% | 13.8% | 1.7% | 3.2% | 0.0% | 49.4% | 0 |
| `inpainting_pyramid` | `fp32` | 32 | 72 | 108 | 5120 | 1286 | 0 | 0 | **0.0%** | 25.7% | 32.7% | 0.0% | 4.1% | 3.3% | 34.2% | 292 |
| `inpainting_pyramid` | `fp32` | 64 | 48 | 108 | 5120 | 1579 | 0 | 0 | **0.0%** | 34.3% | 31.9% | 0.0% | 3.4% | 2.7% | 27.9% | 0 |
| `inpainting_pyramid` | `fp16`* | 32 | 72 | 108 | 5120 | 1286 | 0 | 0 | **0.0%** | 25.7% | 32.7% | 0.0% | 4.1% | 3.3% | 34.2% | 292 |
| `inpainting_pyramid` | `fp16`* | 64 | 48 | 108 | 5120 | 1579 | 0 | 0 | **0.0%** | 34.3% | 31.9% | 0.0% | 3.4% | 2.7% | 27.9% | 0 |
| `inpainting` | `fp32` | 32 | 48 | 108 | 0 | 1551 | 0 | 0 | **0.0%** | 29.5% | 12.2% | 1.4% | 2.9% | 0.0% | 54.0% | 158 |
| `inpainting` | `fp32` | 64 | 36 | 108 | 0 | 1698 | 0 | 0 | **0.0%** | 35.2% | 12.2% | 1.3% | 2.7% | 0.0% | 48.6% | 0 |
| `inpainting` | `fp16`* | 32 | 48 | 108 | 0 | 1551 | 0 | 0 | **0.0%** | 29.5% | 12.2% | 1.4% | 2.9% | 0.0% | 54.0% | 158 |
| `inpainting` | `fp16`* | 64 | 36 | 108 | 0 | 1698 | 0 | 0 | **0.0%** | 35.2% | 12.2% | 1.3% | 2.7% | 0.0% | 48.6% | 0 |
| `debug_view` | `fp32` | 32 | 24 | 108 | 0 | 3037 | 0 | 0 | **0.0%** | 24.3% | 13.4% | 2.2% | 3.6% | 0.0% | 56.6% | 226 |
| `debug_view` | `fp32` | 64 | 24 | 108 | 0 | 3276 | 0 | 0 | **0.0%** | 28.3% | 13.6% | 2.0% | 3.3% | 0.0% | 52.8% | 0 |
| `debug_view` | `fp16`* | 32 | 24 | 108 | 0 | 3037 | 0 | 0 | **0.0%** | 24.3% | 13.4% | 2.2% | 3.6% | 0.0% | 56.6% | 226 |
| `debug_view` | `fp16`* | 64 | 24 | 108 | 0 | 3276 | 0 | 0 | **0.0%** | 28.3% | 13.6% | 2.0% | 3.3% | 0.0% | 52.8% | 0 |

*Rows marked `*` (`fp16*`) are built with `-DFFX_HALF=1` as a counterfactual check; the SDK's `*_shaderblobs.cpp` ignores `is16bit` for that pass and never dispatches the 16-bit blob.*

*Notes (`READ` + `MEASURED`):*
- Every pass is **0.0% FP16** and **byte-for-byte identical** between `fp32` and `fp16*`.
- Image/buffer atomics (`MEASURED`): `game_motion_vector_field` (`10` `image_atomic_*`), `optical_flow_vector_field` (`8`), `reconstruct_and_dilate` (`4`), `reconstruct_previous_depth` (`4`), `game_mv_inpainting_pyramid` (`1` buffer atomic), `inpainting_pyramid` (`1` buffer atomic).

### 4. FSR 1.2 Spatial (`fsr1`, EASU + RCAS, 8 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/fsr1/CMakeCompileFSR1Shaders.txt:23-30` (`APPLY_RCAS={0,1}`, `RCAS_PASSTHROUGH_ALPHA=0`, `SRGB_CONVERSIONS=0`). Blob selection (`READ`): `ffx_fsr1_shaderblobs.cpp:50-111`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `easu` | `fp32` | 32 | 96 | 108 | 0 | 1470 | 0 | 0 | **0.0%** | 67.3% | 5.6% | 0.3% | 3.5% | 0.0% | 23.3% | 392 |
| `easu` | `fp32` | 64 | 96 | 108 | 0 | 1789 | 0 | 0 | **0.0%** | 76.2% | 5.3% | 0.2% | 2.9% | 0.0% | 15.3% | 0 |
| `easu` | `fp16` | 32 | 96 | 108 | 0 | 1180 | 728 | 122 | **72.0%** | 1.7% | 5.3% | 0.3% | 4.4% | 0.0% | 16.2% | 9 |
| `easu` | `fp16` | 64 | 72 | 108 | 0 | 1217 | 728 | 122 | **69.8%** | 2.0% | 6.1% | 0.3% | 4.3% | 0.0% | 17.5% | 0 |
| `rcas` | `fp32` | 32 | 96 | 108 | 0 | 375 | 0 | 0 | **0.0%** | 64.0% | 7.7% | 0.0% | 6.4% | 0.0% | 21.9% | 102 |
| `rcas` | `fp32` | 64 | 48 | 108 | 0 | 487 | 0 | 0 | **0.0%** | 68.2% | 9.2% | 0.0% | 4.9% | 0.0% | 17.7% | 0 |
| `rcas` | `fp16` | 32 | 96 | 108 | 0 | 352 | 182 | 24 | **58.5%** | 0.0% | 19.0% | 1.1% | 6.8% | 0.0% | 14.5% | 2 |
| `rcas` | `fp16` | 64 | 48 | 108 | 0 | 362 | 182 | 24 | **56.9%** | 0.0% | 19.9% | 1.1% | 6.6% | 0.0% | 15.5% | 0 |

*Notes (`MEASURED`):* `easu` emits `48` `image_gather4_lz` + `4` `image_store` across all 4 permutations. With `FFX_HALF=1`, `easu` converts `990–1,364` FP32 VALU ops into `728 v_pk_*_f16 + 122 scalar *_f16` (`69.8–72.0%` of the shader, `89.3%` of VALU) and drops wave64 VGPRs `96 -> 72`.

### 5. CAS 1.2 (`cas`, 3 modes x 4 perms, 12 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/cas/CMakeCompileCASShaders.txt:23-28` (`SHARPEN_ONLY={0,1}`, `COLOR_SPACE_CONVERSION={0,4}`). Blob selection (`READ`): `ffx_cas_shaderblobs.cpp:54-85`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_sharpen_linear` | `fp32` | 32 | 96 | 108 | 0 | 616 | 0 | 0 | **0.0%** | 55.2% | 12.8% | 1.3% | 8.4% | 0.0% | 22.2% | 117 |
| `scale_sharpen_linear` | `fp32` | 64 | 84 | 108 | 0 | 721 | 0 | 0 | **0.0%** | 61.0% | 14.3% | 1.1% | 7.2% | 0.0% | 16.4% | 0 |
| `scale_sharpen_linear` | `fp16` | 32 | 96 | 108 | 0 | 553 | 245 | 6 | **45.4%** | 1.4% | 26.2% | 3.4% | 9.4% | 0.0% | 14.1% | 0 |
| `scale_sharpen_linear` | `fp16` | 64 | 72 | 108 | 0 | 561 | 245 | 6 | **44.7%** | 3.6% | 25.7% | 1.2% | 9.3% | 0.0% | 15.5% | 0 |
| `sharpen_only_linear` | `fp32` | 32 | 96 | 108 | 0 | 193 | 0 | 0 | **0.0%** | 45.1% | 15.5% | 0.0% | 12.4% | 0.0% | 26.9% | 22 |
| `sharpen_only_linear` | `fp32` | 64 | 48 | 108 | 0 | 216 | 0 | 0 | **0.0%** | 48.1% | 16.2% | 0.0% | 11.1% | 0.0% | 24.5% | 0 |
| `sharpen_only_linear` | `fp16` | 32 | 96 | 108 | 0 | 204 | 60 | 0 | **29.4%** | 0.0% | 32.4% | 7.4% | 11.8% | 0.0% | 19.1% | 1 |
| `sharpen_only_linear` | `fp16` | 64 | 48 | 108 | 0 | 205 | 60 | 0 | **29.3%** | 5.9% | 32.7% | 1.5% | 11.7% | 0.0% | 19.0% | 0 |
| `sharpen_only_srgb` | `fp32` | 32 | 96 | 108 | 0 | 783 | 0 | 0 | **0.0%** | 77.7% | 4.3% | 0.0% | 3.1% | 0.0% | 14.9% | 109 |
| `sharpen_only_srgb` | `fp32` | 64 | 48 | 108 | 0 | 919 | 0 | 0 | **0.0%** | 77.9% | 3.8% | 0.0% | 2.6% | 0.0% | 15.7% | 0 |
| `sharpen_only_srgb` | `fp16` | 32 | 96 | 108 | 0 | 611 | 300 | 144 | **72.7%** | 0.0% | 11.0% | 2.5% | 3.9% | 0.0% | 10.0% | 0 |
| `sharpen_only_srgb` | `fp16` | 64 | 48 | 108 | 0 | 620 | 300 | 144 | **71.6%** | 1.9% | 10.8% | 0.5% | 3.9% | 0.0% | 11.3% | 0 |

*Notes (`MEASURED`):* `0` LDS and `0` `image_sample` across all permutations (`24` image load/stores in sharpen-only, `52` in scale+sharpen). In `sharpen_only_srgb` (`fp16`), sRGB transfer functions compile to `72 v_log_f16_e32 + 72 v_exp_f16_e32` alongside `300 v_pk_*_f16`.

### 6. SPD 2.2 (`spd`, wave-DPP vs LDS-interop & mean vs max, 12 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/spd/CMakeCompileSPDShaders.txt:23-29` (`LINEAR_SAMPLE=0`, `WAVE_INTEROP_LDS={0,1}`, `DOWNSAMPLE_FILTER={0,2}`). Blob selection (`READ`): `ffx_spd_shaderblobs.cpp:43-85`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `downsample_wave_mean` | `fp32` | 32 | 48 | 108 | 5120 | 755 | 0 | 0 | **0.0%** | 32.1% | 18.9% | 0.0% | 7.0% | 5.6% | 36.4% | 128 |
| `downsample_wave_mean` | `fp32` | 64 | 48 | 108 | 5120 | 872 | 0 | 0 | **0.0%** | 38.5% | 20.3% | 0.0% | 6.1% | 4.8% | 30.3% | 0 |
| `downsample_wave_mean` | `fp16` | 32 | 48 | 108 | 3072 | 692 | 168 | 0 | **24.3%** | 0.0% | 28.8% | 0.0% | 7.7% | 3.3% | 36.0% | 18 |
| `downsample_wave_mean` | `fp16` | 64 | 48 | 108 | 3072 | 701 | 168 | 0 | **24.0%** | 0.0% | 31.0% | 0.0% | 7.6% | 3.3% | 34.2% | 0 |
| `downsample_lds_mean` | `fp32` | 32 | 48 | 108 | 5120 | 1007 | 0 | 0 | **0.0%** | 22.4% | 21.4% | 0.0% | 5.3% | 12.9% | 38.0% | 114 |
| `downsample_lds_mean` | `fp32` | 64 | 48 | 108 | 5120 | 1120 | 0 | 0 | **0.0%** | 27.1% | 22.4% | 0.0% | 4.7% | 11.6% | 34.1% | 0 |
| `downsample_lds_mean` | `fp16` | 32 | 48 | 108 | 3072 | 721 | 168 | 0 | **23.3%** | 0.0% | 18.7% | 0.0% | 7.4% | 9.3% | 41.3% | 17 |
| `downsample_lds_mean` | `fp16` | 64 | 36 | 108 | 3072 | 739 | 168 | 0 | **22.7%** | 0.0% | 20.6% | 0.0% | 7.2% | 9.1% | 40.5% | 0 |
| `downsample_wave_max` | `fp32` | 32 | 72 | 108 | 5120 | 685 | 0 | 0 | **0.0%** | 21.9% | 29.3% | 0.0% | 7.7% | 6.1% | 34.9% | 59 |
| `downsample_wave_max` | `fp32` | 64 | 48 | 108 | 5120 | 727 | 0 | 0 | **0.0%** | 23.1% | 33.3% | 0.0% | 7.3% | 5.8% | 30.5% | 0 |
| `downsample_wave_max` | `fp16` | 32 | 48 | 108 | 3072 | 630 | 126 | 0 | **20.0%** | 0.0% | 32.2% | 0.0% | 8.4% | 3.7% | 35.7% | 22 |
| `downsample_wave_max` | `fp16` | 64 | 48 | 108 | 3072 | 643 | 126 | 0 | **19.6%** | 0.0% | 35.0% | 0.0% | 8.2% | 3.6% | 33.6% | 0 |

*Notes (`MEASURED`):* Every permutation has `53` `image_load`/`image_store` and `1` buffer atomic (`spdCounter`). `WAVE_INTEROP_LDS=0` uses subgroup DPP (`72 v_mov_b32_dpp` in `fp16`), cutting `ds_*` LDS ops from `128 -> 40` (`fp32`) and `65 -> 21` (`fp16`). `FFX_HALF=1` cuts LDS bytes `5,120 -> 3,072 B` and replaces all FP32 math with `126–168` `v_pk_*_f16`.

---

## Group 2 Components (Extended Survey, `MEASURED`)

### 7. Blur 1.1 (`blur`, 3x3 and 15x15 kernels, 8 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/blur/CMakeCompileBlurShaders.txt:23-28` (`KERNEL_DIMENSION={3,15}`, `KERNEL_PERMUTATION=0`).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blur_k3` | `fp32` | 32 | 48 | 108 | 2048 | 298 | 0 | 10 | **3.4%** | 15.4% | 21.5% | 4.0% | 5.7% | 3.7% | 46.3% | 21 |
| `blur_k3` | `fp32` | 64 | 36 | 108 | 2048 | 307 | 0 | 10 | **3.3%** | 19.2% | 23.5% | 3.9% | 5.5% | 3.6% | 41.0% | 0 |
| `blur_k3` | `fp16` | 32 | 48 | 108 | 2048 | 429 | 21 | 23 | **10.3%** | 1.4% | 25.9% | 12.1% | 4.0% | 2.6% | 43.8% | 3 |
| `blur_k3` | `fp16` | 64 | 48 | 108 | 2048 | 423 | 21 | 29 | **11.8%** | 2.6% | 27.0% | 9.7% | 4.0% | 2.6% | 42.3% | 0 |
| `blur_k15` | `fp32` | 32 | 96 | 108 | 4096 | 1226 | 0 | 14 | **1.1%** | 25.0% | 21.9% | 4.9% | 8.7% | 3.0% | 35.2% | 137 |
| `blur_k15` | `fp32` | 64 | 96 | 108 | 4096 | 1315 | 0 | 14 | **1.1%** | 32.0% | 22.2% | 4.6% | 8.1% | 2.8% | 29.2% | 0 |
| `blur_k15` | `fp16` | 32 | 96 | 108 | 4096 | 1616 | 135 | 137 | **16.8%** | 1.9% | 27.4% | 12.1% | 6.6% | 2.3% | 32.9% | 4 |
| `blur_k15` | `fp16` | 64 | 96 | 108 | 4096 | 1618 | 135 | 167 | **18.7%** | 2.3% | 27.6% | 9.8% | 6.6% | 2.3% | 32.7% | 0 |

*Notes (`MEASURED`):* `blur_k15` (`fp16`) emits `159–196` conversions (`v_cvt_*` / `v_pack_b32_f16`), inflating static instruction count by `+23–32%` vs `fp32` (`1,226 -> 1,616` w32; `1,315 -> 1,618` w64) without reducing VGPRs or LDS.

### 8. LPM 1.4 (`lpm`, 4 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/lpm/CMakeCompileLPMShaders.txt:23-25`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `filter` | `fp32` | 32 | 24 | 108 | 0 | 851 | 0 | 0 | **0.0%** | 60.2% | 6.2% | 0.0% | 0.9% | 0.0% | 32.7% | 83 |
| `filter` | `fp32` | 64 | 24 | 108 | 0 | 904 | 0 | 0 | **0.0%** | 64.2% | 7.5% | 0.0% | 0.9% | 0.0% | 27.4% | 0 |
| `filter` | `fp16` | 32 | 24 | 108 | 0 | 621 | 228 | 168 | **63.8%** | 0.0% | 9.0% | 0.0% | 1.3% | 0.0% | 25.9% | 7 |
| `filter` | `fp16` | 64 | 24 | 108 | 0 | 626 | 228 | 168 | **63.3%** | 0.0% | 10.1% | 0.0% | 1.3% | 0.0% | 25.4% | 0 |


### 9. Denoiser 1.3 (`denoiser`, 8 passes, 32 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/denoiser/CMakeCompileDenoiserShaders.txt:23-27` (`INVERTED_DEPTH=1`).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prepare_shadow_mask` | `fp32` | 32 | 96 | 108 | 0 | 437 | 0 | 0 | **0.0%** | 0.0% | 42.6% | 0.0% | 3.7% | 3.7% | 50.1% | 18 |
| `prepare_shadow_mask` | `fp32` | 64 | 48 | 108 | 0 | 539 | 0 | 0 | **0.0%** | 0.0% | 44.0% | 0.0% | 3.0% | 3.0% | 50.1% | 0 |
| `prepare_shadow_mask` | `fp16` | 32 | 96 | 108 | 0 | 437 | 0 | 0 | **0.0%** | 0.0% | 42.6% | 0.0% | 3.7% | 3.7% | 50.1% | 18 |
| `prepare_shadow_mask` | `fp16` | 64 | 48 | 108 | 0 | 539 | 0 | 0 | **0.0%** | 0.0% | 44.0% | 0.0% | 3.0% | 3.0% | 50.1% | 0 |
| `shadows_tile_classification` | `fp32` | 32 | 48 | 108 | 1024 | 929 | 0 | 3 | **0.3%** | 22.5% | 18.6% | 0.6% | 1.3% | 2.7% | 53.9% | 37 |
| `shadows_tile_classification` | `fp32` | 64 | 36 | 108 | 1024 | 921 | 0 | 3 | **0.3%** | 25.6% | 18.2% | 0.7% | 1.3% | 2.1% | 51.8% | 0 |
| `shadows_tile_classification` | `fp16` | 32 | 48 | 108 | 1024 | 929 | 0 | 3 | **0.3%** | 22.5% | 18.6% | 0.6% | 1.3% | 2.7% | 53.9% | 37 |
| `shadows_tile_classification` | `fp16` | 64 | 36 | 108 | 1024 | 921 | 0 | 3 | **0.3%** | 25.6% | 18.2% | 0.7% | 1.3% | 2.1% | 51.8% | 0 |
| `filter_soft_shadows_0` | `fp32` | 32 | 24 | 108 | 0 | 37 | 0 | 0 | **0.0%** | 0.0% | 13.5% | 0.0% | 2.7% | 0.0% | 83.8% | 1 |
| `filter_soft_shadows_0` | `fp32` | 64 | 12 | 108 | 0 | 36 | 0 | 0 | **0.0%** | 0.0% | 16.7% | 0.0% | 2.8% | 0.0% | 80.6% | 0 |
| `filter_soft_shadows_0` | `fp16` | 32 | 72 | 108 | 4096 | 520 | 0 | 8 | **1.5%** | 54.6% | 5.8% | 2.7% | 2.7% | 5.0% | 27.7% | 36 |
| `filter_soft_shadows_0` | `fp16` | 64 | 72 | 108 | 4096 | 543 | 0 | 8 | **1.5%** | 57.8% | 6.6% | 2.6% | 2.6% | 4.8% | 24.1% | 0 |
| `filter_soft_shadows_1` | `fp32` | 32 | 24 | 108 | 0 | 29 | 0 | 0 | **0.0%** | 0.0% | 17.2% | 0.0% | 3.4% | 0.0% | 79.3% | 0 |
| `filter_soft_shadows_1` | `fp32` | 64 | 12 | 108 | 0 | 27 | 0 | 0 | **0.0%** | 0.0% | 18.5% | 0.0% | 3.7% | 0.0% | 77.8% | 0 |
| `filter_soft_shadows_1` | `fp16` | 32 | 72 | 108 | 4096 | 519 | 0 | 8 | **1.5%** | 54.1% | 5.8% | 2.7% | 2.7% | 5.8% | 27.4% | 37 |
| `filter_soft_shadows_1` | `fp16` | 64 | 72 | 108 | 4096 | 545 | 0 | 8 | **1.5%** | 57.6% | 6.2% | 2.6% | 2.6% | 5.5% | 24.0% | 0 |
| `filter_soft_shadows_2` | `fp32` | 32 | 24 | 108 | 0 | 43 | 0 | 0 | **0.0%** | 4.7% | 14.0% | 0.0% | 2.3% | 0.0% | 79.1% | 2 |
| `filter_soft_shadows_2` | `fp32` | 64 | 12 | 108 | 0 | 43 | 0 | 0 | **0.0%** | 4.7% | 18.6% | 0.0% | 2.3% | 0.0% | 74.4% | 0 |
| `filter_soft_shadows_2` | `fp16` | 32 | 72 | 108 | 4096 | 539 | 0 | 8 | **1.5%** | 54.7% | 5.9% | 2.6% | 2.6% | 5.6% | 27.1% | 27 |
| `filter_soft_shadows_2` | `fp16` | 64 | 60 | 108 | 4096 | 556 | 0 | 8 | **1.4%** | 57.4% | 6.3% | 2.5% | 2.5% | 5.4% | 24.5% | 0 |
| `reproject_reflections` | `fp32` | 32 | 96 | 108 | 2048 | 2139 | 0 | 12 | **0.6%** | 57.9% | 4.7% | 0.7% | 2.8% | 4.8% | 28.6% | 229 |
| `reproject_reflections` | `fp32` | 64 | 60 | 108 | 2048 | 2304 | 0 | 12 | **0.5%** | 62.8% | 4.8% | 0.6% | 2.6% | 4.4% | 24.2% | 0 |
| `reproject_reflections` | `fp16` | 32 | 96 | 108 | 2048 | 2030 | 314 | 481 | **39.2%** | 16.4% | 6.5% | 1.3% | 3.0% | 5.0% | 28.7% | 54 |
| `reproject_reflections` | `fp16` | 64 | 60 | 108 | 2048 | 2061 | 314 | 497 | **39.3%** | 18.4% | 6.5% | 0.2% | 2.9% | 4.9% | 27.7% | 0 |
| `prefilter_reflections` | `fp32` | 32 | 120 | 108 | 5120 | 815 | 0 | 8 | **1.0%** | 58.8% | 4.4% | 2.7% | 2.5% | 6.3% | 24.4% | 56 |
| `prefilter_reflections` | `fp32` | 64 | 120 | 108 | 5120 | 862 | 0 | 8 | **0.9%** | 62.4% | 4.4% | 2.1% | 2.3% | 5.9% | 21.9% | 0 |
| `prefilter_reflections` | `fp16` | 32 | 120 | 108 | 5120 | 922 | 68 | 273 | **37.0%** | 19.5% | 5.3% | 7.5% | 2.2% | 5.5% | 23.0% | 16 |
| `prefilter_reflections` | `fp16` | 64 | 120 | 108 | 5120 | 952 | 68 | 319 | **40.7%** | 22.2% | 5.5% | 0.6% | 2.1% | 5.4% | 23.6% | 0 |
| `resolve_temporal_reflections` | `fp32` | 32 | 96 | 108 | 2048 | 1181 | 0 | 0 | **0.0%** | 64.4% | 3.2% | 1.1% | 1.1% | 7.2% | 22.9% | 111 |
| `resolve_temporal_reflections` | `fp32` | 64 | 48 | 108 | 2048 | 1252 | 0 | 0 | **0.0%** | 69.3% | 3.7% | 0.8% | 1.0% | 6.8% | 18.4% | 0 |
| `resolve_temporal_reflections` | `fp16` | 32 | 96 | 108 | 2048 | 1007 | 266 | 302 | **56.4%** | 4.2% | 4.3% | 0.7% | 1.3% | 8.4% | 24.7% | 9 |
| `resolve_temporal_reflections` | `fp16` | 64 | 48 | 108 | 2048 | 1039 | 266 | 305 | **55.0%** | 4.7% | 4.6% | 0.2% | 1.3% | 8.2% | 26.1% | 0 |

*Notes (`READ` + `MEASURED`):*
- `filter_soft_shadows_{0,1,2}` `#else` branches in `sdk/include/FidelityFX/gpu/denoiser/ffx_denoiser_shadows_filter.h:127-128` return `0` when `FFX_HALF=0`, causing ACO to dead-code-eliminate the `fp32` builds to `27–43` instructions. Only the `fp16` builds contain the real `519–556`-instruction shadow filter (which itself is mostly FP32 VALU + `4,096 B` LDS + `8` scalar FP16 ops).
- In contrast, the three reflection passes (`reproject`, `prefilter`, `resolve_temporal`) execute heavily in FP16 (`37.0–56.4%`) when `FFX_HALF=1`, alongside `2,048–5,120 B` LDS (`51–102` `ds_*` ops) and `29` `image_sample` in `reproject_reflections`.

### 10. CACAO 1.4 (`cacao`, 31 passes, 124 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/cacao/CMakeCompileCACAOShaders.txt:23-28` (`APPLY_SMART=1`).
To keep the table readable while reporting every pass, the table below lists all 31 passes in Wave64 (`fp32` & `fp16`, matching `FFX_PREFER_WAVE64`), plus Wave32 rows for the passes whose FP16 or VGPR behavior changes meaningfully (`edge_sensitive_blur_1`, `edge_sensitive_blur_8`, `upscale_bilateral_5x5`, `generate_q2`, `generate_q3`). Full 124-pipeline stats are preserved in `summary.json`.


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `apply_non_smart_half` | `fp32` | 64 | 12 | 108 | 0 | 37 | 0 | 0 | **0.0%** | 8.1% | 16.2% | 5.4% | 10.8% | 0.0% | 59.5% | 0 |
| `apply_non_smart_half` | `fp16` | 64 | 12 | 108 | 0 | 37 | 0 | 0 | **0.0%** | 8.1% | 16.2% | 5.4% | 10.8% | 0.0% | 59.5% | 0 |
| `apply_non_smart` | `fp32` | 64 | 12 | 108 | 0 | 48 | 0 | 0 | **0.0%** | 12.5% | 16.7% | 4.2% | 12.5% | 0.0% | 54.2% | 0 |
| `apply_non_smart` | `fp16` | 64 | 12 | 108 | 0 | 48 | 0 | 0 | **0.0%** | 12.5% | 16.7% | 4.2% | 12.5% | 0.0% | 54.2% | 0 |
| `apply` | `fp32` | 64 | 36 | 108 | 0 | 116 | 0 | 0 | **0.0%** | 32.8% | 17.2% | 10.3% | 5.2% | 0.0% | 34.5% | 0 |
| `apply` | `fp16` | 64 | 36 | 108 | 0 | 116 | 0 | 0 | **0.0%** | 32.8% | 17.2% | 10.3% | 5.2% | 0.0% | 34.5% | 0 |
| `clear_load_counter` | `fp32` | 64 | 12 | 108 | 0 | 7 | 0 | 0 | **0.0%** | 0.0% | 28.6% | 0.0% | 14.3% | 0.0% | 57.1% | 0 |
| `clear_load_counter` | `fp16` | 64 | 12 | 108 | 0 | 7 | 0 | 0 | **0.0%** | 0.0% | 28.6% | 0.0% | 14.3% | 0.0% | 57.1% | 0 |
| `edge_sensitive_blur_1` | `fp32` | 32 | 96 | 108 | 13312 | 531 | 0 | 12 | **2.3%** | 34.7% | 12.8% | 16.0% | 4.5% | 3.2% | 26.6% | 16 |
| `edge_sensitive_blur_1` | `fp32` | 64 | 96 | 108 | 13312 | 550 | 0 | 12 | **2.2%** | 37.5% | 13.5% | 13.3% | 4.4% | 3.1% | 26.2% | 0 |
| `edge_sensitive_blur_1` | `fp16` | 32 | 96 | 108 | 13312 | 499 | 60 | 60 | **24.0%** | 4.8% | 16.2% | 17.0% | 4.8% | 3.4% | 29.7% | 10 |
| `edge_sensitive_blur_1` | `fp16` | 64 | 96 | 108 | 13312 | 508 | 60 | 60 | **23.6%** | 7.5% | 17.5% | 14.4% | 4.7% | 3.3% | 28.9% | 0 |
| `edge_sensitive_blur_2` | `fp32` | 64 | 96 | 108 | 13312 | 663 | 0 | 24 | **3.6%** | 40.1% | 12.7% | 11.0% | 3.6% | 4.2% | 24.7% | 0 |
| `edge_sensitive_blur_2` | `fp16` | 64 | 96 | 108 | 13312 | 591 | 96 | 60 | **26.4%** | 6.4% | 17.6% | 12.4% | 4.1% | 4.7% | 28.4% | 0 |
| `edge_sensitive_blur_3` | `fp32` | 64 | 120 | 108 | 13312 | 778 | 0 | 36 | **4.6%** | 41.9% | 10.2% | 9.4% | 3.1% | 5.0% | 25.8% | 0 |
| `edge_sensitive_blur_3` | `fp16` | 64 | 96 | 108 | 13312 | 703 | 132 | 60 | **27.3%** | 5.4% | 17.6% | 10.4% | 3.4% | 5.5% | 30.3% | 0 |
| `edge_sensitive_blur_4` | `fp32` | 64 | 120 | 108 | 13312 | 839 | 0 | 48 | **5.7%** | 46.0% | 9.1% | 8.7% | 2.9% | 6.0% | 21.7% | 0 |
| `edge_sensitive_blur_4` | `fp16` | 64 | 96 | 108 | 13312 | 741 | 168 | 60 | **30.8%** | 5.1% | 18.4% | 9.9% | 3.2% | 6.7% | 25.9% | 0 |
| `edge_sensitive_blur_5` | `fp32` | 64 | 120 | 108 | 13312 | 968 | 0 | 60 | **6.2%** | 46.1% | 8.2% | 7.5% | 2.5% | 6.3% | 23.2% | 0 |
| `edge_sensitive_blur_5` | `fp16` | 64 | 96 | 108 | 13312 | 852 | 204 | 60 | **31.0%** | 4.5% | 18.1% | 8.6% | 2.8% | 7.2% | 27.9% | 0 |
| `edge_sensitive_blur_6` | `fp32` | 64 | 120 | 108 | 13312 | 1033 | 0 | 72 | **7.0%** | 49.0% | 7.4% | 7.1% | 2.3% | 7.0% | 20.3% | 0 |
| `edge_sensitive_blur_6` | `fp16` | 64 | 96 | 108 | 13312 | 895 | 240 | 60 | **33.5%** | 4.2% | 18.5% | 8.2% | 2.7% | 8.0% | 24.8% | 0 |
| `edge_sensitive_blur_7` | `fp32` | 64 | 120 | 108 | 13312 | 1153 | 0 | 84 | **7.3%** | 49.1% | 6.9% | 6.3% | 2.1% | 7.2% | 21.2% | 0 |
| `edge_sensitive_blur_7` | `fp16` | 64 | 96 | 108 | 13312 | 993 | 276 | 60 | **33.8%** | 3.8% | 18.5% | 7.4% | 2.4% | 8.4% | 25.7% | 0 |
| `edge_sensitive_blur_8` | `fp32` | 32 | 120 | 108 | 13312 | 1200 | 0 | 96 | **8.0%** | 49.5% | 5.7% | 7.1% | 2.0% | 7.8% | 19.9% | 28 |
| `edge_sensitive_blur_8` | `fp32` | 64 | 120 | 108 | 13312 | 1214 | 0 | 96 | **7.9%** | 51.6% | 6.3% | 6.0% | 2.0% | 7.7% | 18.5% | 0 |
| `edge_sensitive_blur_8` | `fp16` | 32 | 96 | 108 | 13312 | 1018 | 312 | 60 | **36.5%** | 2.4% | 17.2% | 8.3% | 2.4% | 9.2% | 24.0% | 23 |
| `edge_sensitive_blur_8` | `fp16` | 64 | 96 | 108 | 13312 | 1034 | 312 | 60 | **36.0%** | 3.7% | 19.0% | 7.1% | 2.3% | 9.1% | 22.9% | 0 |
| `generate_importance_map_a` | `fp32` | 64 | 12 | 108 | 0 | 58 | 0 | 0 | **0.0%** | 34.5% | 6.9% | 3.4% | 8.6% | 0.0% | 46.6% | 0 |
| `generate_importance_map_a` | `fp16` | 64 | 12 | 108 | 0 | 58 | 0 | 0 | **0.0%** | 34.5% | 6.9% | 3.4% | 8.6% | 0.0% | 46.6% | 0 |
| `generate_importance_map_b` | `fp32` | 64 | 48 | 108 | 0 | 100 | 0 | 0 | **0.0%** | 22.0% | 26.0% | 3.0% | 6.0% | 0.0% | 43.0% | 0 |
| `generate_importance_map_b` | `fp16` | 64 | 48 | 108 | 0 | 100 | 0 | 0 | **0.0%** | 22.0% | 26.0% | 3.0% | 6.0% | 0.0% | 43.0% | 0 |
| `generate_importance_map` | `fp32` | 64 | 24 | 108 | 0 | 140 | 0 | 0 | **0.0%** | 65.7% | 7.1% | 1.4% | 3.6% | 0.0% | 22.1% | 0 |
| `generate_importance_map` | `fp16` | 64 | 24 | 108 | 0 | 140 | 0 | 0 | **0.0%** | 65.7% | 7.1% | 1.4% | 3.6% | 0.0% | 22.1% | 0 |
| `generate_q0` | `fp32` | 64 | 48 | 108 | 0 | 319 | 0 | 0 | **0.0%** | 57.1% | 6.3% | 0.9% | 2.8% | 0.3% | 32.6% | 0 |
| `generate_q0` | `fp16` | 64 | 48 | 108 | 0 | 319 | 0 | 0 | **0.0%** | 57.1% | 6.3% | 0.9% | 2.8% | 0.3% | 32.6% | 0 |
| `generate_q1` | `fp32` | 64 | 48 | 108 | 0 | 604 | 0 | 0 | **0.0%** | 71.5% | 3.1% | 0.5% | 2.3% | 0.2% | 22.4% | 0 |
| `generate_q1` | `fp16` | 64 | 48 | 108 | 0 | 604 | 0 | 0 | **0.0%** | 71.5% | 3.1% | 0.5% | 2.3% | 0.2% | 22.4% | 0 |
| `generate_q2` | `fp32` | 32 | 96 | 108 | 0 | 923 | 0 | 0 | **0.0%** | 72.0% | 1.6% | 0.3% | 3.5% | 0.1% | 22.4% | 189 |
| `generate_q2` | `fp32` | 64 | 48 | 108 | 0 | 1086 | 0 | 0 | **0.0%** | 78.5% | 1.6% | 0.3% | 2.9% | 0.1% | 16.6% | 0 |
| `generate_q2` | `fp16` | 32 | 96 | 108 | 0 | 923 | 0 | 0 | **0.0%** | 72.0% | 1.6% | 0.3% | 3.5% | 0.1% | 22.4% | 189 |
| `generate_q2` | `fp16` | 64 | 48 | 108 | 0 | 1086 | 0 | 0 | **0.0%** | 78.5% | 1.6% | 0.3% | 2.9% | 0.1% | 16.6% | 0 |
| `generate_q3_base` | `fp32` | 64 | 48 | 108 | 0 | 439 | 0 | 0 | **0.0%** | 73.6% | 2.7% | 0.7% | 3.0% | 0.2% | 19.8% | 0 |
| `generate_q3_base` | `fp16` | 64 | 48 | 108 | 0 | 439 | 0 | 0 | **0.0%** | 73.6% | 2.7% | 0.7% | 3.0% | 0.2% | 19.8% | 0 |
| `generate_q3` | `fp32` | 32 | 96 | 108 | 0 | 452 | 0 | 0 | **0.0%** | 51.1% | 4.9% | 0.9% | 3.3% | 0.2% | 39.6% | 65 |
| `generate_q3` | `fp32` | 64 | 48 | 108 | 0 | 504 | 0 | 0 | **0.0%** | 57.5% | 6.3% | 0.8% | 3.0% | 0.2% | 32.1% | 0 |
| `generate_q3` | `fp16` | 32 | 96 | 108 | 0 | 452 | 0 | 0 | **0.0%** | 51.1% | 4.9% | 0.9% | 3.3% | 0.2% | 39.6% | 65 |
| `generate_q3` | `fp16` | 64 | 48 | 108 | 0 | 504 | 0 | 0 | **0.0%** | 57.5% | 6.3% | 0.8% | 3.0% | 0.2% | 32.1% | 0 |
| `prepare_downsampled_depths_and_mips` | `fp32` | 64 | 36 | 108 | 1024 | 214 | 0 | 0 | **0.0%** | 45.3% | 13.6% | 0.9% | 5.1% | 4.7% | 30.4% | 0 |
| `prepare_downsampled_depths_and_mips` | `fp16` | 64 | 36 | 108 | 1024 | 214 | 0 | 0 | **0.0%** | 45.3% | 13.6% | 0.9% | 5.1% | 4.7% | 30.4% | 0 |
| `prepare_downsampled_depths_half` | `fp32` | 64 | 12 | 108 | 0 | 49 | 0 | 0 | **0.0%** | 36.7% | 20.4% | 0.0% | 8.2% | 0.0% | 34.7% | 0 |
| `prepare_downsampled_depths_half` | `fp16` | 64 | 12 | 108 | 0 | 49 | 0 | 0 | **0.0%** | 36.7% | 20.4% | 0.0% | 8.2% | 0.0% | 34.7% | 0 |
| `prepare_downsampled_depths` | `fp32` | 64 | 24 | 108 | 0 | 97 | 0 | 0 | **0.0%** | 41.2% | 13.4% | 2.1% | 8.2% | 0.0% | 35.1% | 0 |
| `prepare_downsampled_depths` | `fp16` | 64 | 24 | 108 | 0 | 97 | 0 | 0 | **0.0%** | 41.2% | 13.4% | 2.1% | 8.2% | 0.0% | 35.1% | 0 |
| `prepare_downsampled_normals_from_input_normals` | `fp32` | 64 | 36 | 108 | 0 | 94 | 0 | 0 | **0.0%** | 51.1% | 17.0% | 0.0% | 8.5% | 0.0% | 23.4% | 0 |
| `prepare_downsampled_normals_from_input_normals` | `fp16` | 64 | 36 | 108 | 0 | 94 | 0 | 0 | **0.0%** | 51.1% | 17.0% | 0.0% | 8.5% | 0.0% | 23.4% | 0 |
| `prepare_downsampled_normals` | `fp32` | 64 | 84 | 108 | 0 | 641 | 0 | 0 | **0.0%** | 83.3% | 3.4% | 0.3% | 2.5% | 0.0% | 10.5% | 0 |
| `prepare_downsampled_normals` | `fp16` | 64 | 84 | 108 | 0 | 641 | 0 | 0 | **0.0%** | 83.3% | 3.4% | 0.3% | 2.5% | 0.0% | 10.5% | 0 |
| `prepare_native_depths_and_mips` | `fp32` | 64 | 36 | 108 | 1024 | 205 | 0 | 0 | **0.0%** | 47.3% | 12.7% | 1.0% | 3.9% | 4.9% | 30.2% | 0 |
| `prepare_native_depths_and_mips` | `fp16` | 64 | 36 | 108 | 1024 | 205 | 0 | 0 | **0.0%** | 47.3% | 12.7% | 1.0% | 3.9% | 4.9% | 30.2% | 0 |
| `prepare_native_depths_half` | `fp32` | 64 | 12 | 108 | 0 | 49 | 0 | 0 | **0.0%** | 36.7% | 20.4% | 0.0% | 8.2% | 0.0% | 34.7% | 0 |
| `prepare_native_depths_half` | `fp16` | 64 | 12 | 108 | 0 | 49 | 0 | 0 | **0.0%** | 36.7% | 20.4% | 0.0% | 8.2% | 0.0% | 34.7% | 0 |
| `prepare_native_depths` | `fp32` | 64 | 24 | 108 | 0 | 87 | 0 | 0 | **0.0%** | 46.0% | 11.5% | 2.3% | 5.7% | 0.0% | 34.5% | 0 |
| `prepare_native_depths` | `fp16` | 64 | 24 | 108 | 0 | 87 | 0 | 0 | **0.0%** | 46.0% | 11.5% | 2.3% | 5.7% | 0.0% | 34.5% | 0 |
| `prepare_native_normals_from_input_normals` | `fp32` | 64 | 36 | 108 | 0 | 94 | 0 | 0 | **0.0%** | 51.1% | 17.0% | 0.0% | 8.5% | 0.0% | 23.4% | 0 |
| `prepare_native_normals_from_input_normals` | `fp16` | 64 | 36 | 108 | 0 | 94 | 0 | 0 | **0.0%** | 51.1% | 17.0% | 0.0% | 8.5% | 0.0% | 23.4% | 0 |
| `prepare_native_normals` | `fp32` | 64 | 84 | 108 | 0 | 643 | 0 | 0 | **0.0%** | 85.1% | 3.0% | 0.6% | 1.2% | 0.0% | 10.1% | 0 |
| `prepare_native_normals` | `fp16` | 64 | 84 | 108 | 0 | 643 | 0 | 0 | **0.0%** | 85.1% | 3.0% | 0.6% | 1.2% | 0.0% | 10.1% | 0 |
| `upscale_bilateral_5x5` | `fp32` | 32 | 96 | 108 | 2048 | 1075 | 0 | 0 | **0.0%** | 62.3% | 6.3% | 3.9% | 2.5% | 2.5% | 22.4% | 194 |
| `upscale_bilateral_5x5` | `fp32` | 64 | 48 | 108 | 2048 | 1228 | 0 | 0 | **0.0%** | 69.6% | 6.4% | 3.4% | 2.2% | 2.2% | 16.2% | 0 |
| `upscale_bilateral_5x5` | `fp16` | 32 | 96 | 108 | 2048 | 880 | 333 | 108 | **50.1%** | 10.9% | 7.7% | 5.2% | 3.1% | 3.1% | 19.9% | 46 |
| `upscale_bilateral_5x5` | `fp16` | 64 | 48 | 108 | 2048 | 932 | 333 | 108 | **47.3%** | 14.7% | 9.1% | 4.5% | 2.9% | 2.9% | 18.6% | 0 |

*Notes (`MEASURED`):* Across CACAO's 31 passes, FP16 is concentrated in `upscale_bilateral_5x5` (`47.3%` w64, `333 pk16 + 108 s16`) and `edge_sensitive_blur_1..8` (`23.6–36.0%` w64, `13,312 B` LDS, cutting w64 VGPRs `120 -> 96` on passes `3..8`). All 5 AO generation passes (`generate_q0..q3_base`) and all depth/normal prepare passes compile to `0.0%` FP16 (pure FP32 + up to `24` `image_sample` in `generate_q2`).

### 11. SSSR 1.5 (`sssr`, 5 passes, 20 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/sssr/CMakeCompileSSSRShaders.txt:23-27` (`INVERTED_DEPTH=1`).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `classify_tiles` | `fp32` | 32 | 48 | 108 | 1024 | 327 | 0 | 1 | **0.3%** | 19.0% | 15.3% | 0.9% | 2.1% | 2.1% | 60.2% | 21 |
| `classify_tiles` | `fp32` | 64 | 36 | 108 | 0 | 325 | 0 | 3 | **0.9%** | 24.9% | 12.3% | 0.9% | 2.2% | 1.2% | 57.5% | 0 |
| `classify_tiles` | `fp16` | 32 | 48 | 108 | 1024 | 327 | 0 | 1 | **0.3%** | 19.0% | 15.3% | 0.9% | 2.1% | 2.1% | 60.2% | 21 |
| `classify_tiles` | `fp16` | 64 | 36 | 108 | 0 | 325 | 0 | 3 | **0.9%** | 24.9% | 12.3% | 0.9% | 2.2% | 1.2% | 57.5% | 0 |
| `depth_downsample` | `fp32` | 32 | 48 | 108 | 2048 | 653 | 0 | 0 | **0.0%** | 6.6% | 21.3% | 0.0% | 13.0% | 3.2% | 55.9% | 4 |
| `depth_downsample` | `fp32` | 64 | 36 | 108 | 2048 | 649 | 0 | 0 | **0.0%** | 6.6% | 22.2% | 0.0% | 13.1% | 3.2% | 54.9% | 0 |
| `depth_downsample` | `fp16` | 32 | 48 | 108 | 2048 | 653 | 0 | 0 | **0.0%** | 6.6% | 21.3% | 0.0% | 13.0% | 3.2% | 55.9% | 4 |
| `depth_downsample` | `fp16` | 64 | 36 | 108 | 2048 | 649 | 0 | 0 | **0.0%** | 6.6% | 22.2% | 0.0% | 13.1% | 3.2% | 54.9% | 0 |
| `intersect` | `fp32` | 32 | 72 | 108 | 0 | 622 | 0 | 0 | **0.0%** | 47.7% | 9.3% | 1.8% | 2.1% | 0.2% | 38.9% | 85 |
| `intersect` | `fp32` | 64 | 48 | 108 | 0 | 696 | 0 | 0 | **0.0%** | 54.3% | 9.1% | 1.6% | 1.9% | 0.1% | 33.0% | 0 |
| `intersect` | `fp16` | 32 | 72 | 108 | 0 | 622 | 0 | 0 | **0.0%** | 47.7% | 9.3% | 1.8% | 2.1% | 0.2% | 38.9% | 85 |
| `intersect` | `fp16` | 64 | 48 | 108 | 0 | 696 | 0 | 0 | **0.0%** | 54.3% | 9.1% | 1.6% | 1.9% | 0.1% | 33.0% | 0 |
| `prepare_blue_noise_texture` | `fp32` | 32 | 24 | 108 | 0 | 51 | 0 | 0 | **0.0%** | 11.8% | 29.4% | 3.9% | 9.8% | 0.0% | 45.1% | 2 |
| `prepare_blue_noise_texture` | `fp32` | 64 | 12 | 108 | 0 | 52 | 0 | 0 | **0.0%** | 11.5% | 32.7% | 3.8% | 9.6% | 0.0% | 42.3% | 0 |
| `prepare_blue_noise_texture` | `fp16` | 32 | 24 | 108 | 0 | 51 | 0 | 0 | **0.0%** | 11.8% | 29.4% | 3.9% | 9.8% | 0.0% | 45.1% | 2 |
| `prepare_blue_noise_texture` | `fp16` | 64 | 12 | 108 | 0 | 52 | 0 | 0 | **0.0%** | 11.5% | 32.7% | 3.8% | 9.6% | 0.0% | 42.3% | 0 |
| `prepare_indirect_args` | `fp32` | 32 | 24 | 108 | 0 | 22 | 0 | 0 | **0.0%** | 0.0% | 27.3% | 0.0% | 0.0% | 22.7% | 50.0% | 3 |
| `prepare_indirect_args` | `fp32` | 64 | 12 | 108 | 0 | 25 | 0 | 0 | **0.0%** | 0.0% | 36.0% | 0.0% | 0.0% | 20.0% | 44.0% | 0 |
| `prepare_indirect_args` | `fp16` | 32 | 24 | 108 | 0 | 22 | 0 | 0 | **0.0%** | 0.0% | 27.3% | 0.0% | 0.0% | 22.7% | 50.0% | 3 |
| `prepare_indirect_args` | `fp16` | 64 | 12 | 108 | 0 | 25 | 0 | 0 | **0.0%** | 0.0% | 36.0% | 0.0% | 0.0% | 20.0% | 44.0% | 0 |

*Notes (`MEASURED`):* `classify_tiles` drops LDS from `1,024 B` (`3 ds_*`) in wave32 to `0 B` (`0 ds_*`) in wave64 because the 8x8 tile fits in a single wave64 without LDS reduction. All 5 passes are effectively FP32/INT32 (`<=0.9%` FP16) even with `FFX_HALF=1`.

### 12. Parallel Sort 1.3 (`parallelsort`, 6 passes, 24 pipelines)
Defines (`READ`): `sdk/include/FidelityFX/gpu/parallelsort/CMakeCompileParallelSortShaders.txt:23-27` (`HAS_PAYLOAD=1`). Blob selection (`READ`): `ffx_parallelsort_shaderblobs.cpp:51-160` (ignores `is16bit`).


| Pass | Perm | Wave | VGPR | SGPR | LDS (B) | Static Insns | `pk16` | `s16` | FP16% | `fp32%` | `int%` | `cvt%` | `tex/img%` | `mem/lds%` | `salu/wb%` | VOPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `setup_indirect_args` | `fp32` | 32 | 24 | 108 | 0 | 16 | 0 | 0 | **0.0%** | 0.0% | 25.0% | 0.0% | 0.0% | 12.5% | 62.5% | 2 |
| `setup_indirect_args` | `fp32` | 64 | 12 | 108 | 0 | 18 | 0 | 0 | **0.0%** | 0.0% | 33.3% | 0.0% | 0.0% | 11.1% | 55.6% | 0 |
| `setup_indirect_args` | `fp16`* | 32 | 24 | 108 | 0 | 16 | 0 | 0 | **0.0%** | 0.0% | 25.0% | 0.0% | 0.0% | 12.5% | 62.5% | 2 |
| `setup_indirect_args` | `fp16`* | 64 | 12 | 108 | 0 | 18 | 0 | 0 | **0.0%** | 0.0% | 33.3% | 0.0% | 0.0% | 11.1% | 55.6% | 0 |
| `sum` | `fp32` | 32 | 72 | 108 | 8192 | 364 | 0 | 0 | **0.0%** | 0.0% | 29.1% | 0.0% | 0.0% | 22.3% | 48.6% | 3 |
| `sum` | `fp32` | 64 | 96 | 108 | 8192 | 369 | 0 | 0 | **0.0%** | 0.0% | 29.5% | 0.0% | 0.0% | 22.0% | 48.5% | 0 |
| `sum` | `fp16`* | 32 | 72 | 108 | 8192 | 364 | 0 | 0 | **0.0%** | 0.0% | 29.1% | 0.0% | 0.0% | 22.3% | 48.6% | 3 |
| `sum` | `fp16`* | 64 | 96 | 108 | 8192 | 369 | 0 | 0 | **0.0%** | 0.0% | 29.5% | 0.0% | 0.0% | 22.0% | 48.5% | 0 |
| `reduce` | `fp32` | 32 | 96 | 108 | 1024 | 158 | 0 | 0 | **0.0%** | 0.6% | 27.2% | 0.0% | 0.0% | 4.4% | 67.7% | 1 |
| `reduce` | `fp32` | 64 | 48 | 108 | 1024 | 167 | 0 | 0 | **0.0%** | 0.6% | 28.7% | 0.0% | 0.0% | 4.2% | 66.5% | 0 |
| `reduce` | `fp16`* | 32 | 96 | 108 | 1024 | 158 | 0 | 0 | **0.0%** | 0.6% | 27.2% | 0.0% | 0.0% | 4.4% | 67.7% | 1 |
| `reduce` | `fp16`* | 64 | 48 | 108 | 1024 | 167 | 0 | 0 | **0.0%** | 0.6% | 28.7% | 0.0% | 0.0% | 4.2% | 66.5% | 0 |
| `scan` | `fp32` | 32 | 96 | 108 | 3072 | 185 | 0 | 0 | **0.0%** | 0.0% | 30.3% | 0.0% | 0.0% | 15.1% | 54.6% | 0 |
| `scan` | `fp32` | 64 | 48 | 108 | 3072 | 193 | 0 | 0 | **0.0%** | 0.0% | 31.1% | 0.0% | 0.0% | 14.5% | 54.4% | 0 |
| `scan` | `fp16`* | 32 | 96 | 108 | 3072 | 185 | 0 | 0 | **0.0%** | 0.0% | 30.3% | 0.0% | 0.0% | 15.1% | 54.6% | 0 |
| `scan` | `fp16`* | 64 | 48 | 108 | 3072 | 193 | 0 | 0 | **0.0%** | 0.0% | 31.1% | 0.0% | 0.0% | 14.5% | 54.4% | 0 |
| `scan_add` | `fp32` | 32 | 96 | 108 | 3072 | 233 | 0 | 0 | **0.0%** | 0.4% | 25.8% | 0.0% | 0.0% | 12.0% | 61.8% | 0 |
| `scan_add` | `fp32` | 64 | 48 | 108 | 3072 | 242 | 0 | 0 | **0.0%** | 0.4% | 26.4% | 0.0% | 0.0% | 11.6% | 61.6% | 0 |
| `scan_add` | `fp16`* | 32 | 96 | 108 | 3072 | 233 | 0 | 0 | **0.0%** | 0.4% | 25.8% | 0.0% | 0.0% | 12.0% | 61.8% | 0 |
| `scan_add` | `fp16`* | 64 | 48 | 108 | 3072 | 242 | 0 | 0 | **0.0%** | 0.4% | 26.4% | 0.0% | 0.0% | 11.6% | 61.6% | 0 |
| `scatter` | `fp32` | 32 | 96 | 108 | 2048 | 1196 | 0 | 0 | **0.0%** | 0.0% | 29.7% | 0.0% | 0.0% | 10.6% | 59.7% | 6 |
| `scatter` | `fp32` | 64 | 48 | 108 | 2048 | 1299 | 0 | 0 | **0.0%** | 0.0% | 30.9% | 0.0% | 0.0% | 9.8% | 59.4% | 0 |
| `scatter` | `fp16`* | 32 | 96 | 108 | 2048 | 1196 | 0 | 0 | **0.0%** | 0.0% | 29.7% | 0.0% | 0.0% | 10.6% | 59.7% | 6 |
| `scatter` | `fp16`* | 64 | 48 | 108 | 2048 | 1299 | 0 | 0 | **0.0%** | 0.0% | 30.9% | 0.0% | 0.0% | 9.8% | 59.4% | 0 |

*Rows marked `*` (`fp16*`) are built with `-DFFX_HALF=1` as a counterfactual check; the SDK's `*_shaderblobs.cpp` ignores `is16bit` for that pass and never dispatches the 16-bit blob.*

*Notes (`MEASURED`):* Pure INT32 + LDS (`1,024–8,192 B`, up to `110 ds_*` in `scatter`) + buffer loads/stores; `0.0%` FP16 across all 24 pipelines (`reduce` and `scan_add` contain `1` `v_rcp`/float helper op each). Wave64 halves VGPRs (`96 -> 48`) on `reduce`, `scan`, `scan_add`, and `scatter`.

---

## Which passes deserve runtime timing next (`INFERRED` from `MEASURED` + Round 12)

Static counts identify which hardware units each shader *contains*, not which unit gates its wall-clock time on `gfx1201`. Based on the ISA signatures above and Round 12's `gfx1201` hardware laws (`measurements/blindspot-followups.md`), these six targets offer the highest information per GPU minute next:

1. **`opticalflow__optical_flow_v5` (Vulkan GLSL software-SAD vs. DX12 HLSL `v_msad_u8`, w32 vs. w64):** `1,365` static INT32 instructions emulating packed 8-bit SAD in software (`0` `v_msad_u8`) vs. HLSL's `16x msad4`. Directly quantifies the real-world cost of GLSL lacking `msad4` and whether Wave64's `96 -> 48` VGPR drop outweighs Wave64's `2x` TA load-issue rate.
2. **`frameinterpolation__interpolation`, `disocclusion_mask`, and `game_motion_vector_field` (w32 vs. w64, MCLK sweep):** Pure FP32 + heavy `image_load`/`image_store` (`56–72` ops) + `8–10` global `image_atomic_*` in MV field passes. Tests global image-atomic contention and whether Wave32 VOPD (`159–215` pairs) + half TA load cost beats Wave64 (`48 -> 36` VGPRs).
3. **`fsr3upscaler__accumulate` (`fp16` vs. `fp32`, `w32` vs. `w64`):** Tests whether the modest `7.9–8.8%` FP16 + `5.2%` `v_fma_mix_f32` reprojection path (and w64 `96 -> 72` VGPR savings) actually beats `fp32` Wave32 with `277` VOPD dual-issue pairs, or whether 32 image loads + 3 filtered samples pin it to L2/MCLK anyway.
4. **`fsr1__easu` & `cas__sharpen_only_srgb` (`fp16` vs. `fp32`, `w32` vs. `w64`):** Highest packed-FP16 density in the entire SDK (`70–73%` FP16, `245–728` `v_pk_*_f16`), pairing packed FP16 VALU with `48` `image_gather4_lz` (EASU) or `24` image loads/stores (CAS). Clean real-shader test of packed FP16 throughput vs. TMU/MALL bandwidth.
5. **`spd__downsample` (`wave` vs. `lds`, `fp16` vs. `fp32`) & `parallelsort__scatter` (`w32` vs. `w64`):** Tests CU-mode LDS bandwidth (`104 B/cyc/CU`) and `v_mov_b32_dpp` wave reductions (`21` vs `65` `ds_*`, `3,072` vs `5,120 B` LDS) against Wave64 VGPR halving (`96 -> 48` in `scatter`).
6. **`blur__blur_k15` (`fp16` vs. `fp32`):** Direct empirical check of the conversion-bloat anomaly (`+303` static instructions in w64, `+390` in w32, and `159–196` `v_cvt_*` / `v_pack_b32_f16` in `fp16` vs. `137` VOPD pairs in `fp32`) — does `FFX_HALF=1` help or hurt runtime here?

