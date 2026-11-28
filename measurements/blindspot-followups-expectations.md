# Blind-spot follow-ups (Round 12): preregistered expectations

Drafted: 2026-11-26, BEFORE the Blind Spot #3, #4, and #10 GPU runs (note: first committed to git alongside the Round 12 write-up in the same commit, so pre-run authorship is not independently provable from git commit timestamps alone).
Card: Radeon RX 9070 XT (`gfx1201`), LACT Default (`auto`, no sudo), ROCm 10.1 (`hipcc`, `-mcumode`), Mesa RADV `27.0.0-devel`.

## 1. Blind Spot #3 — Store-hint MALL residency & cross-wave 32 B write combining (`store_mall`)

### Part 3A: Store-hint MALL allocation (`after_hinted_store`)
- **Question:** When a producer writes a `4 MiB` or `32 MiB` buffer with `global_store_b128` using each valid `gfx1201` store cache hint (`rt`, `nt`, `ht`, `wb`, `nt_rt`, `rt_nt`, `nt_ht`, `rt_sys`, `nt_sys`), does a separate consumer launch hit in the 64 MB MALL (~360–420 cycles random chase; `> 1000 GB/s` 16 MiB streaming read) or miss to GDDR6 (~500–560 cycles; `~630–700 GB/s`)?
- **Preregistered expectation:** Mirroring the measured load-hint split (`memory-system.md` §8), `th:TH_STORE_NT`, `th:TH_STORE_RT_NT`, and `th:TH_STORE_NT scope:SCOPE_SYS` will **not** allocate in MALL (consumer falls to near `cold_ref` GDDR6 latency and ~650–700 GB/s), whereas `rt`, `ht`, `wb`, `nt_rt`, `nt_ht`, and `rt_sys` will allocate in MALL (consumer near `warm_ref` latency and `> 1000 GB/s`). Establishing whether `TH_STORE_NT` bypasses MALL determines when non-temporal stores benefit write-only outputs without evicting hot producer->consumer activations.
- **Controls & break:** `cold_ref` (flushed 256 MiB) must be `>= 80` cycles slower than `warm_ref` (read twice with `TH_LOAD_RT`). Every pointer-chase trial verifies exact final pointer. `--break` corrupts expected pointer & checksum and must exit non-zero (`FAIL`).

### Part 3B: Store-hint MALL eviction (`after_hinted_stream_store`)
- **Question:** Does streaming `256 MiB` of stores with hint `H` evict a hot `32 MiB` activation buffer from MALL?
- **Preregistered expectation:** `nt`, `rt_nt`, and `nt_sys` leave the hot `32 MiB` chase near `warm_ref`; `rt`, `ht`, `wb`, `nt_rt`, `nt_ht`, and `rt_sys` evict it toward `cold_ref`.

### Part 3C: Cross-wave 32 B piece write combining (4-wave shared-line pattern)
- **Question:** Writing `512 MiB` (`4M` × `128 B` lines) where 4 waves each write one 32 B quarter of the same 128 B lines: does `xwave_same_wg_32B` (4 waves in the same workgroup/CU) combine near full-line bandwidth (`full_b128` / `full_b32`), and does `xwave_diff_wg_32B` (4 different workgroups across CUs) lose combining?
- **Preregistered expectation:** `full_b128` and `full_b32` reach `>= 550 GB/s` (at 1408 MHz MCLK). Because `gfx1201` commits in 32 B sectors (`memory-system.md` §10), `quarter_32B_only` (32 B written per 128 B line) should avoid the 16 B read-modify-write penalty (`eighth_16B_only` much slower per written byte). `xwave_same_wg_32B` is expected to reach `>= 70%` of `full_b32` (combining in L2/MALL sectors), while `xwave_diff_wg_32B` may show lower efficiency due to cross-CU arrival skew. Every pattern verifies deterministic written words on GPU/host.

## 2. Blind Spot #4 — Load-landing cost vs load width (`b128`/`b64`/`b32`) and clustering (`load_cluster`)
- **Question:** At equal bytes per WMMA group (`L0` 8 KiB and `L2` 1 MiB, 1 wave and whole card, wave32 and wave64), does issuing fewer wider loads (`1 x b128` vs `2 x b64` vs `4 x b32`) or clustering 4 loads before 4 WMMAs (`4 x b128` clumped vs `1 x b128` before each WMMA) reduce cycles/WMMA?
- **Preregistered expectation:**
  - Reference `ref` (register-fed FP8 WMMA) must land in `7.9–8.3` cyc/mm in both wave32 and wave64 (gate).
  - In 1 wave, because landing cost is per landing event (`matrix-timing.md` §8a: 1 load `25.7 cyc` vs 4 loads `32.2 cyc`), clustering 4 loads before 4 WMMAs should cut per-WMMA overhead by `~2–3x` compared to spreading 1 load per WMMA, and `1 x b128` should beat `2 x b64` and `4 x b32` at equal bytes.
  - Across the whole card (8 waves/SIMD wave32, 4 waves/SIMD wave64), multi-wave hiding compresses all deltas to `< 3 cyc/mm`, with wave64 showing slightly larger load overhead than wave32 (`vulkan-view.md`).
- **Controls & break:** Exact FP8 WMMA arithmetic verified on every cell (`E4M3 1.0 * 1.0`); `--break` must report `FAIL` and exit non-zero.

## 3. Blind Spot #10 — Vulkan RADV wave32 vs wave64 `ld_mat2` (`vk_ld_mat2`)
- **Question:** Under Mesa RADV + ACO on `gfx1201`, how do wave64 and wave32 compare on register-fed (`ref`) and load-fed (`K=1`, `K=8` at `L0` 8 KiB and `L2` 1 MiB, 1 wave and whole card) FP8 cooperative-matrix loops?
- **Preregistered expectation:**
  - `ref` hits `~8.0 cyc/mm` in both wave32 and wave64 (`vulkan-view.md` §1).
  - Under `K=1` load-fed loops (where ACO inserts descriptor/load waits before each `coopMatMulAdd`), wave64 will be slower per WMMA than wave32 in 1 wave and slightly slower (~5–15%) across the whole card due to double VGPR usage per wave (halving wave count per SIMD) and address-ALU overhead; at `K=8` reuse, both approach `8.0–9.5 cyc/mm` across the whole card.
- **Controls & break:** Captured ACO disassembly confirms `v_wmma_f32_16x16x16_fp8_fp8` inside the loop and reports `VGPRs` + `Subgroups per SIMD`; exact FP8 output verified; `--break` must fail.
