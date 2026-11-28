# Preregistered expectations: pinned memory-clock scaling (cache latency, FSR 4.1.2) and vLLM FP8 split sweep (2026-11-26)

Written before running any GPU jobs in this batch. Card: ASUS Prime RX 9070 XT (`1002:7550`, `gfx1201`), LACT profile `Default` (`overclock`: +150 MHz core, -50 mV, 374 W cap, 1409 MHz memory request, 1.2x regulator mod). Monitors active (`dpms=On`) for Parts A, B, and C; Part D waits for a `dpms=Off` window.

## Part A: Pinned cache latency versus memory clock (`r9_lat`)

- **Method:** Run `build/round9/bin/r9_lat lat 16,1024,32768,1048576 5 both` (L0 16 KiB, L2 1 MiB, MALL 32 MiB random and page, GDDR6 1 GiB random, fresh, and page; zero background load) under `scripts/round10/pin_mclk.sh` at `--level 1` (`456 MHz`), `--level 3` (`875 MHz`), and `--level 5` (`1124 MHz`), plus one locked unpinned `auto` run. Record `r9_sampler` (50 ms) beside every job and check `journalctl -k` for GPU faults.
- **Predictions:**
  1. `r9_sampler` holds `uclk_cur_MHz` flat at each pinned level (or exposes any driver clamping, as seen earlier when `--level 5` held flat at `1124 MHz`).
  2. `L0` (16 KiB) and `L2` (1 MiB) stay invariant in cycles across pinned MCLK levels (within ~5%; L0 near 81–84 cycles, L2 near 210–250 cycles).
  3. `GDDR6` (1 GiB random and page) latency in nanoseconds decreases monotonically as pinned MCLK increases (`456 -> 875 -> 1124 MHz`).
  4. `MALL` (32 MiB) cleanly separates whether the memory-side cache tracks UCLK/fabric (monotonic ns drop with MCLK) or shader/L2 clock (flat cycles).
- **Controls / failure criteria:** `pointer=ok` on 100% of rows; `RESTORE-OK level=auto mask=all` and no `RESTORE-FAILED` after every pinned run; 0 amdgpu faults. Any run whose sampled `uclk` drifts from its pinned level is flagged and not pooled.

## Part B: Pinned FSR 4.1.2 memory-clock scaling (outside view)

- **Method:** AMD FidelityFX SDK sample (`Release240`, outside view only; `run_sample.sh` forbidden-env guard active, gamescope headless, Quality `n6=0`, frame generation off `n7=1`, `1500` frames per cell) at `1920x1080` and `3840x2160` across three memory-clock conditions in symmetric interleaved order: `auto` (~1408 MHz under load), `pin_mclk.sh --level 5` (`1124 MHz`), `pin_mclk.sh --level 3` (`875 MHz`), then reverse (`level 3`, `level 5`, `auto`). `r9_sampler` (10 ms) runs beside every cell; `Cauldron.log` and JSON timestamps are archived per cell.
- **Predictions:**
  1. Pinning MCLK to `1124 MHz` (`--level 5`) against `auto` (`1408 MHz`) causally reproduces Round 8's observed **+12% to +19%** 4K Quality slowdown (`~1.66 ms -> ~1.88–1.96 ms`).
  2. Pinning MCLK further down to `875 MHz` (`--level 3`) increases 4K Quality upscaler time monotonically beyond `1124 MHz` (predicted roughly **+35% to +55%** above `1408 MHz`).
  3. 1080p Quality shows a smaller percentage slowdown than 4K Quality if small-grid overhead / fixed dispatch latency occupies a larger share at 1080p, or comparable scaling if memory-bandwidth bound at both sizes.
  4. Forward and reverse repeats at the same MCLK state agree within **<= 3%**.
- **Controls / failure criteria:** Every cell must show `qualityMode 1` in `Cauldron.log`, `restarts >= 1`, no `Frame Generation` GPU label, window median `gfxclk >= 3000 MHz`, exact expected `uclk` median, `RESTORE-OK` after every pinned window, hotspot `< 95 C`, and 0 amdgpu faults.

## Part C: vLLM FP8 two-level decode split sweep (ROCm 10.2, vLLM `0.32.1`, no sudo)

- **Method:** `toolchains/vllm-rocm102-venv` with `scripts/llm4_vllm.py bench --model llama32-3b-redhat-fp8 --prefill 512 --streams 1 4 8 --steps 128 --reps 2`, gated via `scripts/study-llm.sh dev` (shared lock, `MemoryMax=10G`, 95 C hotspot guard, LACT `Default` check) with `r9_sampler` (200 ms) and full stdout/stderr captured per fresh process start. Five conditions (17 fresh starts total):
  1. `default` (4 starts): default `VLLM_COMPILE` + default cudagraphs + default AOT/cache.
  2. `no_aot_nocache` (4 starts): `VLLM_USE_AOT_COMPILE=0`, `VLLM_USE_STANDALONE_COMPILE=0`, `VLLM_DISABLE_COMPILE_CACHE=1`, isolated empty `VLLM_CACHE_ROOT` per start.
  3. `cg_piecewise` (3 starts): `--compilation-mode VLLM_COMPILE --cudagraph-mode PIECEWISE`.
  4. `cg_none_compiled` (3 starts): `--compilation-mode VLLM_COMPILE --cudagraph-mode NONE` (compile active, cudagraph capture/replay off).
  5. `eager` (3 starts): `--eager` (`enforce_eager=True`).
- **Predictions (testing ROCm/TheRock #9228 hypotheses):**
  1. `default` reproduces the bimodal decode-1 split (~159 vs ~120 tok/s, >10% spread across starts) at identical active clocks (~3.45 GHz SCLK, 1408 MHz MCLK).
  2. If AOT / compile-cache loading is NOT the sole trigger (per TheRock #9228 updates), `no_aot_nocache` still shows the split (or locks consistently to one band, which cleanly falsifies or confirms cache involvement).
  3. Comparing `cg_piecewise`, `cg_none_compiled`, and `eager` isolates whether the split requires full-decode cudagraph replay, any cudagraph replay, `torch.compile`, or disappears completely in `eager` (where eager is predicted to be slower than both graph bands, ~90–115 tok/s, with <3% start-to-start spread).
- **Controls / failure criteria:** Every job exits 0 with valid `RESULT` JSON, 0 amdgpu faults, hotspot `< 95 C`, and `uclk` / `gfxclk` recorded from `r9_sampler` to verify no clock confound separates fast and slow starts.

## Part D: Warmed 3-repeat pinned Item 3 Step 1 (staged for `dpms=Off`)

- **Method:** `r10_cl3` with `warm=150` (matching `vkr10`'s 150 ms warm-up spin) across 3 repeats at pinned `1124 MHz` MCLK once displays blank (`dpms=Off`).
- **Prediction:** `hip-base-r` SCLK moves from ~1.98 GHz to ~3.2 GHz (`cell_clock PASS`), all repeat spreads fall `<= 5%`, and `vk-base-p` vs `hip-base-p` either passes all controls to establish a clean non-clock gap or identifies the remaining variance source.
