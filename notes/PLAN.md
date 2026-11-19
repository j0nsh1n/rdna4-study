# RDNA 4 deep-dive: starting point

Local only, no remote. Card: RX 9070 XT (gfx1201), replacing the RX 7900 XT (gfx1100) in the same machine.
Goal: the same kind of deep-dive as `~/Projects/rdna3-fp8`, for RDNA 4. Scope and order are for the session that
runs it (the user plans to switch this thread to Fable) to settle with the user.

## Ready before the swap (checked 2026-11-16)

- ROCm 7.1.1 (Fedora packages): rocBLAS, hipBLASLt ship gfx1201 kernels. `hipcc --offload-arch=gfx1201` builds
  (`rdna3-fp8/examples/bandwidth/bandwidth.hip` compiled).
- PyTorch: `toolchains/torch-rocm10-venv` (2026-11-17): torch 2.15.0+rocm10.0 from
  `https://download.pytorch.org/whl/rocm10.0/`, which brings its own ROCm 10.0.0 as pip packages (18 GB; not in
  git; package list in `toolchains/torch-rocm10-freeze.txt`). Tested on the RX 9070 XT: FP16 matmul and
  `torch._scaled_mm` FP8 run and are correct (first numbers on the Testing profile: 131.8 TFLOPS FP8, 102.9 FP16,
  4096 x 4096 x 512, one run while other jobs were using the CPU). Do not enable the allocator option
  `expandable_segments` (open PyTorch issue on gfx1201, `notes/research-7-rocm-upgrade-and-fp8-software.md`).
  The earlier 2.13.0+rocm7.1 environment in `rdna3-fp8/build/torch-venv` was deleted on 2026-11-17.
- llama.cpp master `f477c1b90`, built for gfx1201 into `build-gfx1201/` of both
  `rdna3-fp8/build/llama.cpp-latest` (unmodified) and `rdna3-fp8/build/llama.cpp-latest-patched` (our RDNA 3 patch).
  See the build result at the end of this file.
- Models: `rdna3-fp8/models` links to `/var/mnt/games/LLM_Models/rdna3-fp8` (3B and 8B: FP8, FP8 with Q8_0 output
  layer, Q8_0; the 8B HF checkpoint).
- RDNA 3 baseline on the same llama.cpp master, for cross-card comparison: `rdna3-fp8/build/latest-runs/`
  (script `rdna3-fp8/build/test-latest.sh`). The full RDNA 3 numbers are in `rdna3-fp8/README.md`.

## After the swap: first checks

The checks below are now one script: `scripts/first-hour.sh` (usage in `scripts/README.md`). Status on 2026-11-17:
the RX 9070 XT is installed; `info` ran and `build` compiles everything for gfx1201 and gfx1100. No GPU phase has
run yet (owner's hold), so there is no measurement on either card from this script.

0. Before any GPU load: `scripts/first-hour.sh info`. It saves the KFD node's `properties`, the `caches/` table and
   `mem_banks/`, clocks and power state (`pp_dpm_sclk`, `pp_dpm_mclk`, `pp_power_profile_mode`,
   `power_dpm_force_performance_level`, power cap), rocminfo, vulkaninfo's FP8/BF16 matrix features, versions and
   firmware into a new `runs/<time>-<arch>/`, and writes `card.env`. Compare with `baseline-7900xt-kfd-topology.txt`.
1. The card record holds the HIP device index (rocminfo order; checked with the HIP runtime before the first GPU
   job), so the integrated GPU is never timed. Every program refuses a card of another arch (exit 3).
2. `scripts/first-hour.sh build`, then the GPU phases in order: `bandwidth` (ceiling, buffers above 4x the level-3
   cache), `probes` (matrix instruction cycles and overlap, matrix exactness, FP8 conversions), `decode` (the RDNA 3
   decode prototype), `llama` (kernel tests, NaN check, speed on both llama.cpp builds), `libs` (hipBLASLt FP8,
   `torch._scaled_mm`). Or `all`.
3. Every GPU job takes the shared lock (`GPU_LOCK` in `rdna3-fp8/build/local.env`) and logs clocks before and after.

## What is already known about FP8 on RDNA 4 (from reading code and the RDNA 3 work)

- RDNA 4 has FP8 WMMA (`v_wmma_f32_16x16x16_fp8_fp8` and mixed FP8/BF8 forms) and FP8 conversion instructions.
- llama.cpp master does NOT use them on AMD. `ggml_cuda_mul_mat_fp8` is a stub that returns false under HIP
  (`ggml/src/ggml-cuda/fp8.cu`, the `#else` branch), so FP8 weights are widened to FP16 for prompts and decoded in
  software for generation, as on RDNA 3.
- On RDNA 4 the cuBLAS path already uses FP32 output for FP16 compute (`prefer_f32_output` includes RDNA 4), so the
  RDNA 3 NaN bug should not appear. INFERRED, to be measured.
- Our patch's WMMA prompt kernel is RDNA 3 only. Its decode kernel is generic HIP and should run on gfx1201.
- The RDNA 4 FP8 WMMA rounding model used in `rdna3-fp8/measurements/errors-after-quantizing.md` (M-g8) is a fit,
  not a measurement on this card. RDNA 4's accumulator layout differs from RDNA 3's (row `8h + i`, see
  `rdna3-fp8/measurements/kernels-and-drivers.md`).
- `kernels-and-drivers.md` also notes the matrix-op-inside-a-lane-mask pattern exists in gfx12 code objects; whether
  it misbehaves on RDNA 4 is unknown.

## The RDNA 3 method, as a template

What the RDNA 3 deep-dive did, in order. Each step has an RDNA 4 counterpart:

1. Survey: what software supports FP8 on the card, and what it does with it (llama.cpp, PyTorch, Vulkan, FSR 4).
2. Measure the hardware: bandwidth ceiling, cache trap, matrix instruction cost and rounding, conversion cost.
3. Go/no-go: llama.cpp FP8 vs Q8_0 vs F16 on a large out-of-cache weight, decode and prompt.
4. Prototype kernels in `examples/` style, count instructions with `examples/tools/loop_isa.py`, then time.
5. Integrate into llama.cpp as a patch; gate with `test-backend-ops`, a deliberate break, perplexity, KL divergence.
6. Real models (3B, 8B): speed at 1-8 streams, prompt speed, memory, real prompts.
7. Side tracks: PyTorch FP8 layer, FSR 4.
8. Write-up: limits, workarounds, results, with MEASURED / INFERRED labels.

## Open questions for the user

- The FSR 4 comparison (RDNA 3 emulation vs RDNA 4 native) needs RDNA 3 captures from the 7900 XT. If that comparison
  matters, it has to happen before the swap; it was on hold.
- Whether this becomes its own repo later or a section of `rdna3-fp8`.

## Build result

Both gfx1201 builds succeeded (2026-11-16): `llama-perplexity`, `llama-bench`, `llama-batched-bench`,
`test-backend-ops`, `llama-completion` in `build-gfx1201/bin/` of `llama.cpp-latest` and `llama.cpp-latest-patched`.
Built for gfx1201 only; not run (no RDNA 4 card was installed yet). Build script: `rdna3-fp8/build/build-gfx1201.sh`.

## Toolchains on this machine (2026-11-17)

- System: ROCm 7.1.1 from Fedora 44 (hipcc, clang 20), now with `hipblaslt-devel` and `ShellCheck` installed.
  `scripts/first-hour.sh` uses this one.
- ROCm 10.1.0, self-contained, unpacked in `toolchains/rocm-10.1.0/install/` (AMD's stable gfx120X tarball of
  2026-10-05, HIP 7.16, AMD clang 24; 9.9 GB; ignored by git; our own checksum in `SHA256SUMS`). Nothing uses it by
  default and the system is unchanged. Compile-only check: `wmma_time`, `cvt_fp8`, `bandwidth` and `fp8_gemv` build
  with it into `build/gfx1201-rocm10/`. Not run. To use it for one command:
  `R=$PWD/toolchains/rocm-10.1.0/install; env ROCM_PATH=$R HIP_PATH=$R PATH=$R/bin:$PATH $R/bin/hipcc ... -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`.
- Order agreed with the owner: measure on 7.1.1 first, then repeat with 10.1, so the effect of the upgrade is a
  result of its own.
