# Top-5 GPU Replication Protocol and Expectations

Preregistered replication protocol, acceptance/rejection criteria, and invalidation controls for an independent operator running from a clean checkout of this repository.

Supported hardware targets:
- Primary target: AMD Radeon RX 9070 XT (`gfx1201`, PCI `0x1002:0x7550`, 64 CUs / 32 WGPs / 128 SIMDs).
- Related RDNA 4 SKUs: AMD Radeon AI PRO R9700 (`gfx1201`) and RX 9060 XT (`gfx1200`/`gfx1201`, fewer CUs; judge per-CU rates and ratios rather than raw whole-card totals).
- Control target: RDNA 3 (`gfx1100`, e.g., RX 7900 XTX) where applicable for architecture comparison (Probes 1-3 can serve as contrast controls if built for `gfx1100`; Probe 4 `ld_mat2` requires `gfx1201` FP8 WMMA instructions and PCI `0x7550`, and Probe 5 `run_item3_bisect.sh` targets `gfx1201` / PCI `0x7550`, so `p4` and `p5` are skipped on other cards).

## Operator Checklist

1. **Clean checkout**: Verify `git status --short` is clean before running. Record the commit SHA and dirty flag via `scripts/env_manifest.py`.
2. **Configuration track (`env_manifest.py`)**: Run `python3 -I scripts/env_manifest.py --repo . --device <pci_id> --out <run_dir>/env-manifest-before.json` before GPU work and again afterward. Record `gpu.track`:
   - `stock`: core offset `0`, voltage offset `0`, `OD_MCLK` level 1 `<= 1259` MHz.
   - `overclock`: any nonzero offset or `OD_MCLK` level 1 `> 1259` MHz.
   - `unknown`: sysfs overdrive table incomplete or unreadable.
   Cycle-counter and bit-exact checks (Probes 1, 3, and cycle columns of Probes 2 and 4) are clock-invariant across stock and overclock tracks; wall-clock rates (GB/s, TB/s, ns, watts) must always carry the recorded `track` and `mclk`.
3. **Pre-run safety gates** (enforced by `scripts/replicate_top5.sh`):
   - `MemAvailable >= 8 GiB` before every job.
   - No `RESTORE-FAILED` sentinel file present (if a prior clock-pin restore failed, inspect sysfs by hand and clear it before running).
   - Hotspot temperature `<= 58 C` before starting every probe; post-probe check aborts at `>= 95 C`, and Probes 4 and 5 also enforce active `95 C` stops during execution.
   - Kernel log (`journalctl -k`) readable before starting, and scanned after every probe for `amdgpu` page faults, GPU resets, or ring timeouts.
4. **What invalidates a replication run**:
   - Missing or failed listing/disassembly check (`haz_hazscan.py`, `mem_loopcheck.py`, `check_ld_mat2.py`, `check_item3_listing.py`).
   - Any `amdgpu` page fault, reset, or ring timeout in `journalctl -k` during a job.
   - Hotspot `> 58 C` at launch start (Probes 4-5) or thermal guard trip (`>= 95 C`).
   - Unpinned memory-clock (`uclk` / `mclk`) drift or ramp contaminating wall-clock comparisons without matched-clock grouping (Probe 5).
   - Card configuration (`power1_cap` or `pp_od_clk_voltage`) changing mid-run.
   - Any deliberate break (`--break` or aim break) failing to report `FAIL` / close the control gap.
5. **Archiving results**: Package every completed run directory with `scripts/archive_runs.py`:
   ```sh
   python3 -I scripts/archive_runs.py pack --run-dir <run_dir> --out-dir <archive_dir> --repo . --name <bundle_name>
   python3 -I scripts/archive_runs.py verify --out-dir <archive_dir> --name <bundle_name>
   ```
   To cross-reference markdown tables in `measurements/*.md` and `README.md` against local run directories:
   ```sh
   python3 -I scripts/archive_runs.py manifest --repo . --out <manifest.json> [--runs-root <dir>]
   ```

---

## Probe `p1` — Non-inlined WMMA Call-Boundary Compiler Failure

- **Source & checker**: `src/study/haz_hazard.hip`, `scripts/haz_hazscan.py`, `scripts/haz-expect/hazscan-haz_hazard.txt`, reference driver `scripts/study-compiler.sh`.
- **What it tests**: Whether AMD clang omits the required VALU (`v_nop`) separation across a non-inlined device call (`first_mm_tied`) that ends with `v_wmma_f16_16x16x16_f16` and returns its result untouched to a caller (`h_call3`, `h_call4`) whose next instruction is `v_wmma_f32_16x16x16_f16` reading that result as A.
- **Exact expected result (`gfx1201`)**:
  - Disassembly (`haz_hazscan.py`): `first_mm_tied return == 1`, `h_call3 call == 1`, `h_call4 call == 1`, `h_asm_raw hazard == 1`, `h_asm_nop hazard == 0`, straight-line/inlined shapes `hazard == 0` and `call == 0`.
  - Hardware execution (`haz_hazard all`):
    - `hazard_straight`, `hazard_bsrc`, `hazard_call`, `hazard_call_tied`, `hazard_asm_vnop`, `hazard_branch`, `hazard_mixed`, `hazard_loop`: `0 of 256` differ from `src/study/arith_model.h` (`PASS`).
    - `hazard_call_miss` (`h_call3` and `h_call4`): `64 of 256` differ from the model, and `0 of 256` differ from `h_asm_raw` (`hazard_asm_no_vnop`), which itself produces `64 of 256` wrong against the model (bit-identical to the forbidden no-`v_nop` sequence).
  - Deliberate break (`haz_hazard break`): `hazard_asm_no_vnop` reports `FAIL` (`64 of 256` differ, exit 1).
- **Rejection criteria**:
  - `hazard_asm_vnop` differs from `0/256` wrong, or `haz_hazard break` fails to exit 1.
  - `h_asm_raw` gives `0/256` wrong on `gfx1201` (hardware hazard failed to manifest).
- **Invalidation controls (compiler fixed vs measurement failed)**:
  - **Always inspect the `.s` disassembly and `haz_hazscan.py` output.**
  - If `h_call3` / `h_call4` compute `0/256` wrong AND disassembly shows either `v_nop`/VALU emitted before `s_setpc_b64` in `first_mm_tied` (`return == 0`) or a VALU emitted between `s_swappc_b64` and `v_wmma_f32` in the caller (`call == 0`, as seen with incidental `v_add_co_u32` in upstream clang 22.1.8), classify as **compiler fixed or caller scheduled VALU**, not a failed measurement.
  - If disassembly still shows `return == 1` and `call == 1` with no intervening VALU, and `h_asm_raw` gives `64/256` wrong while `h_call4` gives `0/256` wrong, the run is **invalidated** (binary/disassembly mismatch).
  - On RDNA 3 (`gfx1100` control via `src/extcheck/haz_gfx11.hip`), compare using the gfx11 WMMA builtin.

---

## Probe `p2` — LDS CU vs WGP Bandwidth Difference

- **Source & checker**: `src/study/lds_probe.hip` (built twice: `lds_probe` in default WGP mode, and `lds_probe_cu` with `-mcumode -DUSE_CU_KEY=1`), expectation `scripts/lds-expect/lds_probe.txt` via `scripts/mem_loopcheck.py`, reference driver `scripts/study-lds.sh`.
- **What it tests**: Whether HIP's default WGP mode caps a work-group's LDS bandwidth at 128 bytes/cycle per WGP (64 bytes/cycle per CU across 2 CUs), whereas CU mode (`-mcumode`) unlocks each CU's own 128 bytes/cycle LDS port with 32-bit loads (`ds_load_b32`).
- **Exact expected numbers/ratios**:
  - Loop check: `mem_loopcheck.py` passes against `scripts/lds-expect/lds_probe.txt`.
  - Calibration (`calib`): `1.00` cycle per independent `v_fma_f32` (`+- 6%`, `U=6` midpoint deviation `< 1.5` cycles/iter).
  - One work-group (`waves`):
    - WGP mode: 4..32 waves reach `~127.8-128.0` B/cycle (`ds_load_b128`) and `~126.1-128.0` B/cycle (`ds_load_b32`: `126.1` at 4 waves, `127.6` at 8, `127.9` at 16) across 2 CUs.
    - CU mode: 4..32 waves stay on 1 CU and reach `~85.2-85.3` B/cycle (`ds_load_b128`) and `~122.4-126.7` B/cycle (`ds_load_b32`).
  - Whole card (`card`, 512 groups x 16 waves x 32 KiB):
    - WGP `ds_load_b128`: `~64.1-64.2` B/cycle/CU (`~4105-4110` B/cycle card, `~11.4` TB/s on 64-CU RX 9070 XT).
    - WGP `ds_load_b32`: `~65.4` B/cycle/CU (`~4186` B/cycle card, `~11.5` TB/s).
    - CU `ds_load_b128`: `~85.1-85.2` B/cycle/CU (`~5449-5450` B/cycle card, `~15.0` TB/s; ~1.33x WGP).
    - CU `ds_load_b32`: `~125.4` B/cycle/CU (`~8025` B/cycle card, `~21.9` TB/s; **~1.9x WGP**).
  - Deliberate breaks: `lds_probe calib --break` (expects 2.00 cyc/FMA, must exit 1 `FAIL`) and `lds_probe pattern --break` (stride 32 replaced by stride 33, must exit 1 `FAIL`).
- **Rejection criteria**:
  - `calib` fails, `mem_loopcheck.py` fails, or either `--break` exits 0.
  - CU `ds_load_b32` per-CU bytes/cycle is not clearly higher than WGP `ds_load_b32` (expected ratio `>= 1.75x`, reference `~1.92x`).
- **Invalidation controls**:
  - Verify `cus_used` and `simds_used` in `RESULT lds waves` and `RESULT lds card` (WGP 4-wave uses 2 CUs / 4 SIMDs; CU mode stays on 1 CU / 2 SIMDs; whole-card uses all CUs of the target GPU).
  - On smaller/larger SKUs (e.g., RX 9060 XT), scale total card B/cycle by CU count and judge per-CU B/cycle and CU/WGP ratio.
  - Clock-invariant in bytes/cycle; wall-clock TB/s requires recording SCLK and track.

---

## Probe `p3` — Neighboring-SIMD L0/Memory Interference

- **Source & checker**: `src/study/mem_inflight.hip` (built with `-mcumode`, mode `waits` and `--break`), expectation `scripts/mem-expect/mem_inflight.txt` via `scripts/mem_loopcheck.py`, reference driver `scripts/study-memory.sh` (results documented in `measurements/memory-system.md` section 7).
- **What it tests**: Whether slow memory-chasing waves (1 GiB random pointer chase, `~770-800` cycles/load) on the same SIMD vs the neighboring SIMD of the same CU slow down a fast wave performing batches of 8 L0-hit (`16 KiB`) or L2-hit (`1 MiB`) loads.
- **Exact expected slowdown / scaling (`gfx1201`)**:
  - Fast wave alone (`0, 0`): L0 `~159.0` cycles/batch (`1.000x`), L2 `~320.0` cycles/batch (`1.000x`).
  - Slow waves on the **same SIMD** (`1, 0` up to `15, 0`): ratio stays within `0.98x-1.02x` (`same1_ratio <= 1.20` gate).
  - Slow waves on the **other SIMD** up to 8 (`0, 1` / `0, 4` / `0, 8`): ratio stays near `1.00x-1.06x`.
  - Cliff at **`>= 10` slow waves on the other SIMD** (`0, 10`..`0, 16`, `12, 12`, `15, 16`): fast wave slows to **`~3.93x-4.00x`** on L0 (`~625-636` cycles/batch) and **`~1.95x-1.97x`** on L2 (`~623-630` cycles/batch), passing `worst_ratio >= 2.0`.
  - Deliberate break (`mem_inflight waits --break`): claims `worst_ratio <= 1.20` and must exit 1 with `VERDICT mem_inflight_waits: FAIL`.
- **Rejection criteria**:
  - `same1_ratio > 1.20`, or `worst_ratio < 2.0` across high slow-wave counts, or `mem_inflight waits --break` exits 0.
  - `mem_loopcheck.py` fails on `scripts/mem-expect/mem_inflight.txt`.
- **Invalidation controls**:
  - Confirm `-mcumode` was applied so the 1024-thread work-group sits on one CU (check `slow_waves` matches `n_same + n_other`).
  - Confirm `slow_cycles_per_load` is in the GDDR6 latency range (`~650-850` cycles) rather than cached (`< 400` cycles).
  - Clock-invariant (shader cycle counter on one CU). On an RDNA 3 control (`gfx1100`), absence of the `>= 2.0x` cliff indicates architectural difference rather than harness failure if `slow_cycles_per_load` is confirmed at GDDR6 latency.

---

## Probe `p4` — Load Beside FP8 WMMA (`ld_mat2`)

- **Source & checker**: `src/round10/ld_mat2.hip`, listing checker `scripts/round10/check_ld_mat2.py`, driver `scripts/round10/run_ld_mat2_nopin.sh`, preregistered details in `measurements/round10.md`.
- **What it tests**: Per-WMMA cost of one 256-byte `global_load_b64` group per `K` `v_wmma_f32_16x16x16_fp8_fp8` instructions at `L0` (`8 KiB`) and `L2` (`1 MiB`) across lookahead depths `D`, for one wave (`R=209`) and whole card (`R=209` = 7 waves/SIMD, `R=51` = 16 waves/SIMD). (`MALL` and `GDDR` cells refuse without `--pinned`.)
- **Exact expected results (`gfx1201`)**:
  - **Listing check (`c5`)**: `check_ld_mat2.py` passes on extracted code object (`65` kernels checked, 0 extra VALU address ops between WMMAs, exact ring/wait counts, footprint == span), plus `check_ld_mat2.py selftest` passes all deliberate breaks.
  - **Reference gate (`c1_ref_8`)**: matrix-only reference (`ref`) runs first in every job and must land in **`7.9-8.1` cyc/mm** for one wave (`8.055` measured with 256-op loop) and **`7.9-8.3` cyc/mm** for whole card (`8.043` at `R=209`, `8.054` at `R=51`). Ideal FP8 WMMA rate is `8.0` cyc/mm.
  - **One-wave 25-cycle-class L0/L2 transition (`one-L0L2`, `R=209`)**:
    - `L0`: `K=1 D=0` `~104.1` (+96.0), `D=1` `~52.1`, `D=2` `~34.7`, plateau `D=4..54` `~33.06` (**~25.0 cyc/mm penalty**). `K=4 D=12` `~14.27` (+6.22); `K=8 D=12` `~11.15` (+3.09).
    - `L2` anchors (`c3`, `c4`): `K=1 D=54` `33.65 +- 1.0` (`~33.13` measured), `K=4 D=12` `14.4 +- 0.5` (`~14.29` measured), `K=8 D=12` `11.4 +- 0.4` (`~11.15` measured). (Note: short-lookahead one-wave L2 cells `K=1 D<=4`, `K=4 D<=2`, `K=8 D=0` may exceed 5% burst-to-burst spread on `c10_agree`.)
  - **Whole-card `R=209` and `R=51` (`card209-*`, `card51-*`)**:
    - `R=209`: `L0 K=1 D=0` `~15.08` -> `D=60` `~10.03` (+1.98); `L2 K=1 D=0` `~38.53` -> `D=60` `~9.90` (+1.86); `L2 K=8 D=14` `~8.50` (+0.46).
    - `R=51`: `L0 K=1 D=0` `~10.03` -> `D=6` `~9.99` (+1.94); `L2 K=1 D=0` `~16.90` -> `D=6` `~10.02` (+1.97); `L2 K=8 D=6` `~8.38` (+0.33).
  - **Deliberate aim break (`c6_aim_break`)**: `break-one` and `break-card` (L2 label, loads aimed at L0, `K=1 D=0`) MUST close **`>= 90%` (`>= 0.90`)** of the `(L2 - L0)` gap (measured `1.000`).
- **Rejection criteria**:
  - `ref` outside `7.9-8.1` (one-wave) / `7.9-8.3` (whole-card) -> exit 3, stop immediately.
  - `c3`/`c4` anchor fail, `c5` listing fail, `c6_aim_break < 0.90`, `c7_counters` flags bursts, or `c9_placed` misses SIMD/wave counts.
- **Invalidation controls**:
  - Hotspot `> 58 C` at start of any launch (`c8_hot` fail / wrapper exit 4).
  - Unpinned MCLK spread / clock drift: record `mclk` and `CLOCKS` lines; do not assign wall-clock GHz or wattage claims across unpinned runs, and do not make claims about `MALL` or `GDDR` without `--pinned`.
  - `RESTORE-FAILED` present or kernel log GPU fault (exit 11/12).

---

## Probe `p5` — HIP vs Vulkan Clock and Cold-Read Behavior

- **Source & checker**: `src/round10/item3/` (`build.sh`, `vkr10.c`, `r10_cl3.hip`, `check_item3_listing.py`, `analyse_item3.py`), driver `scripts/round10/run_item3_bisect.sh` (`--no-pin --steps 2-6` default; `--pin <level> --steps 1` when pinned MCLK is authorized).
- **What it tests**: Two-way causal bisect comparing cold dependent pointer-chase latency between Vulkan (`vkr10`) and HIP (`r10_cl3`) across random (`r`) and page-order (`p`) walks, controlling for clock ramp, loop form, allocation layout, eviction size, scope, and wave size.
- **Exact expected behavior & bands**:
  - **Baseline bands (`analyse_item3.py`)**:
    - `hip-base-r`: `690-720` cycles/step (`~218 ns` post-ramp); `hip-base-p`: `470-560` cycles/step (`~152 ns`).
    - `vk-base-r`: `740-900` cycles/step (`~232-239 ns` post-ramp); `vk-base-p`: `720-880` cycles/step (`~237-241 ns`).
    - Fused cache-hit controls: `hip-fbase` `200-270` cycles (`~231`), `vk-fbase` `260-340` cycles (`~294-302`).
  - **Unpinned vs pinned separation**:
    - Without `--pin`, Vulkan cells typically ramp `uclk` from `456` MHz up to `1124`/`1408` MHz while HIP starts near `1408` MHz.
    - `analyse_item3.py` outputs a **matched-clock view** (only repetitions where `uclk0 == uclk1`, `>= 3` reps per label) in both cycles and ns.
    - Unpinned runs that fail `repeats` (`> 5%` spread across process repeats) or baseline/eviction controls MUST print `CONTROLS FAILED` and refuse causal closure (as in Round 10 Slot 1: 17 PASS, 6 FAIL, no closure interpreted).
    - Pinned MCLK (`--pin <level> --steps 1`) is required before claiming a non-clock root cause for any remaining gap.
  - **Deliberate breaks & sanity controls**:
    - `vk_noevict_break` / `hip_noevict_break` (`order=noevict,evict=0`): must read `<= 0.60 * cold` (cache level).
    - `vk_loop8_break` / `hip_loop8_break` (8-line footprint): must read `<= 0.50 * cold`.
    - `vk_placement_break` (`vk-gtt-r`) / `hip_placement_break` (`hip-host-r`): system memory must read `>= 2.5 * VRAM cold`.
    - `pointer_ok`: all counted repetitions `ok=1`; `start_temp`: all cells `<= 58 C`, `0 NOT RUN`.
- **Rejection criteria**:
  - `check_item3_listing.py` fails (and its `--selftest` fails to catch a break).
  - Any control (`pointer_ok`, `start_temp`, `repeats`, `*_base_cold_*`, `*_noevict_break_*`, `*_loop8_break_*`, `*_placement_break`, `*_stride*`) fails: leave it failed in the record and refuse closure interpretation.
- **Invalidation controls**:
  - **Clock mismatch**: `uclk0 != uclk1` or unpinned MCLK ramp contaminating medians -> only compare matched-clock (`uclk0 == uclk1`) or pinned (`--pin`) repetitions in both cycles and ns.
  - **Missing eviction break**: if `vk-noev` or `hip-noev` reads `> 0.60 * cold`, eviction was not the cause of coldness and the run is invalidated for cold-read claims.
  - **Kernel faults / hot start**: any `amdgpu` fault in `journalctl -k` (`replicate_top5.sh` exits `11`; `run_item3_bisect.sh` itself exits `13`), `hotspot > 58 C` (exit 4), or `RESTORE-FAILED` (exit 2) invalidates the run.
