# RDNA 4 shared memory, scalar loads and special loads (lane 3)

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201, 64 CUs in 32 WGPs, 128 SIMDs, wave32 under HIP), ROCm 10.1.0
(hipcc, clang 24), kernel 7.3.4. Run folder: `runs/20261117T160439Z-lds/` (`scripts/study-lds.sh all`). Every number is MEASURED
in that run unless labelled READ (from a document) or INFERRED (reasoning). Cycles are shader-clock cycles from the
wave's own counter and do not depend on the clock. The few wall-clock numbers (marked GB/s or TB/s) were taken on LACT
profile "Testing" (stock clocks, 317 W cap); a regulator-level power-limit mod (1.2x) is in the card, so the driver's
power reading is scaled down. The script recorded the profile, `pp_dpm_sclk` and `pp_dpm_mclk` before every job.

This lane was written in a staging folder outside the repository and copied in unchanged by the lead ("the lead"
in the section titles below is the lane's owner, who reran its helpers' probes). The study lead then reran the
whole script from the repository (`runs/20261117T161139Z-lds/`): 25 jobs passed and all 8 breaks failed as in the
owner's run; one job failed, the calibration of `lds_stage feed` (the register-only FP16 loop read 17.66 cycles
in the one launch that check reads, at a counter clock of 2.77 GHz, and 16.01 in the other two launches at 2.47
and 2.51 GHz). The timing leftovers pass settled it (`measurements/execution-core.md` item 9): the launch held
40,550 extra cycles (14.6 us), ten times the card's regular 1.1 us stall, so not the regular stall (INFERRED: a desktop event or a desktop wave on
that SIMD); a clock effect is unlikely (cycles per instruction stay the same while the clock climbs from 2.79 to
3.24 GHz, MEASURED; the climb from the idle clock to 2.79 GHz was not covered). Five
repeats of that job passed (`lds_stage-feed-repeat-*.log` in that folder). Over the owner's run, the rerun and the
five repeats, the per-SIMD medians of all 30 staging configurations agree within 1.5 %, except FP8 with a global
and a transposing load at 16 waves (11.15 cycles in six runs, 12.71 in the rerun).

## In short

- LDS is fast and simple: a 32-bit load costs one cycle per wave (128 bytes per cycle) and its data arrives 32
  cycles after issue. An L0 cache hit of a global load takes 81 cycles (lane 2), so LDS is 2.5 times quicker to
  answer. Bank conflicts follow a plain 32-bank model exactly: every extra pass through a bank costs one cycle.
- The surprises: (1) wide LDS loads are not faster per byte. 64-bit loads reach the same 128 bytes per cycle as
  32-bit ones, 128-bit loads only 85 (6 cycles per load). (2) The LDS port belongs to the CU in CU mode, but in HIP's
  default WGP mode a work-group's LDS traffic for the whole WGP (two CUs) is capped at 128 bytes per cycle per
  WGP instead of 128 per CU (a single CU's path still limits 128-bit loads to 85: two waves on one CU get 85, four
  waves on the two CUs of a WGP get 128). The whole card reads LDS at 11.5 TB/s in WGP mode and 21.9 TB/s in CU mode
  (32-bit loads). (3) Feeding the matrix unit by loading an A and a B tile for every instruction (the probe reloads the same
  data; the loads are in the loop, their values are not checked) reaches only about two thirds of its rate at
  best (FP16 24 cycles per instruction against 16). The cached global route is one plain 128-bit load plus one
  transposing load per instruction, both L0 hits. In CU mode LDS is close to it (26.9 cycles against 24.0 to 24.4
  at 8 and 16 waves); in the default WGP mode LDS is clearly slower (32 to 34 against 24 to 27). A single wave
  reaches about a third.
- The transposing load in one picture: each group of 8 lanes (16-bit data) loads an 8 x 8 block of 16-bit elements,
  one row of 8 per lane, and every lane comes back holding one column of that block. With the lane addresses given
  below, `global_load_tr_b128` turns a row-major B (or column-major A) tile straight into the matrix operand, and a
  multiply with it gives the exact product with no shuffle; `global_load_tr_b64` does the same for 8-bit data (16
  lanes, 8 x 8 bytes). It costs the same as a plain load from DRAM and the memory-side cache, 10 % less bandwidth
  from L2, and from L0 it issues at half the rate of a plain 128-bit load (8 against 4 cycles). The 8-bit form costs
  exactly what a plain 64-bit load costs.
- Scalar loads: a 16 KB scalar cache answers in about 26 to 36 cycles; past it they take 140 to 210 cycles (below
  the vector L2's 230). Scalar loads are not delayed by vector loads in flight.
- Block loads (`global_load_block`) work as the guide says, but are slower than eight 128-bit loads for the same
  bytes (L0, four loads per batch: 15 bytes per cycle per wave against 46 for per-lane 128-bit loads and 73 for
  coalesced ones). No way was found to use dynamic register allocation from HIP: a normal kernel runs with the mode
  off and `s_alloc_vgpr` changes nothing.
- A spill that is reloaded right after it was written (timed with hand-written scratch stores and loads) costs
  about 270 to 340 cycles (the store goes out
  to L2 and the reload waits for it); independent reloads of older spills cost 5 to 8 cycles each.
- What a kernel author should do: read LDS with 32- or 64-bit loads rather than 128-bit ones when one CU's LDS
  path is the limit (CU mode: 128 against 85 bytes per cycle; in WGP mode the whole card reads the same with
  either), lay data out so that the 32 lanes
  (64-bit: each half-wave) touch 32 different banks; build kernels that lean on LDS bandwidth in CU mode
  (`-mcumode`), or know that WGP mode halves the LDS bandwidth per CU; feed matrix work from registers (reuse each
  loaded tile for at least two instructions) rather than reloading A and B for every instruction; use the
  transposing loads to read B row-major (or A column-major) tiles directly instead of shuffling; do not expect
  `global_load_block` to beat plain 128-bit loads.

## 1. LDS as a data store (one wave alone; `lds_probe width`, `chain`, `inflight`)

Lane-sequential addresses (no bank conflicts). "Per access" is the differential cost of one more instruction in a
long block of independent accesses; "issue to data" is the same instruction followed by `s_wait_dscnt 0`.
Spread: lowest to highest repetition, 3 launches x 7 repetitions per build (U = 4, 6, 8).

| Instruction | Bytes per lane | Cycles per access | Bytes per cycle (one wave) | Issue to data, stores: to completion (cycles) |
|---|---|---|---|---|
| `ds_load_u8` | 1 | 1.00 (1.00-1.00) | 32.0 | 32.00 |
| `ds_load_u16` | 2 | 1.00 (1.00-1.00) | 64.0 | 32.00 |
| `ds_load_b32` | 4 | 1.00 (1.00-1.00) | 128.0 | 32.00 |
| `ds_load_b64` | 8 | 2.00 (2.00-2.00) | 128.0 | 34.00 |
| `ds_load_b96` | 12 | 4.00 (4.00-4.17) | 96.0 | 37.00 |
| `ds_load_b128` | 16 | 6.00 (5.95-6.00) | 85.3 | 38.00 |
| `ds_load_2addr_b32` | 8 | 2.00 (2.00-2.00) | 128.0 | 33.00 |
| `ds_load_2addr_b64` | 16 | 5.00 (5.00-5.00) | 102.4 | 36.00 |
| `ds_store_b8` | 1 | 2.03 (1.97-2.03) | 15.8 | 34.00 |
| `ds_store_b16` | 2 | 2.03 (1.97-2.03) | 31.5 | 34.00 |
| `ds_store_b32` | 4 | 2.00 (1.97-2.38) | 64.0 | 34.00 |
| `ds_store_b64` | 8 | 3.00 (3.00-3.00) | 85.3 | 36.00 |
| `ds_store_b96` | 12 | 5.00 (5.00-5.00) | 76.8 | 39.00 |
| `ds_store_b128` | 16 | 5.00 (5.00-5.00) | 102.4 | 40.00 |
| `ds_store_2addr_b32` | 8 | 3.00 (3.00-3.00) | 85.3 | 36.00 |
| `ds_store_2addr_b64` | 16 | 5.00 (5.00-5.16) | 102.4 | 40.00 |

- Loads: 8-, 16- and 32-bit loads cost one cycle; 64-bit two (the same 128 bytes per cycle); 96-bit four and
  128-bit six (85 bytes per cycle). The two-address forms behave like their total width (2addr_b32 = b64, 2 cycles;
  2addr_b64 5 cycles, better than b128's 6). Stores cost more: 32-bit two cycles (64 bytes per cycle), 128-bit five
  (102). Data arrives 32 cycles after a 32-bit load, 38 after a 128-bit load.

| Dependent chain (one register) | Cycles per link |
|---|---|
| load->address | 32.00 (31.93-32.07) |
| load->v_add->address | 49.00 (48.94-49.07) |
| store+load same address | 36.00 (35.86-36.24) |
| v_add alone | 5.00 (5.00-5.00) |

- A load whose data is the next address: 32 cycles (L0 hit of a global load: 81, lane 2). Load, then a vector add,
  then the next load: 49 cycles, which is the 32 + 5 (the add) + 12: the same unexplained 12 to 18 cycles lane 4
  saw after every wait for memory data. Store then load of the same address in the same wave: 36 cycles, and the
  load returns the stored value (checked).

| Loads in flight (N independent `ds_load_b32`, one wait) | Cycles per batch |
|---|---|
| N = 1 | 52.3 (52.3-52.3) |
| N = 2 | 52.6 (52.6-52.6) |
| N = 4 | 54.3 (54.3-54.3) |
| N = 8 | 58.3 (58.3-58.3) |
| N = 16 | 66.3 (66.3-66.3) |
| N = 24 | 74.3 (74.3-74.3) |
| N = 32 | 82.3 (82.3-82.3) |
| N = 48 | 98.3 (98.3-98.3) |
| N = 56 | 106.3 (106.3-106.3) |
| N = 62 | 112.3 (112.3-112.3) |
| N = 63 | 113.3 (113.3-113.3) |
| N = 64 | 114.3 (114.3-124.3) |
| N = 65 | 119.8 (119.8-119.8) |
| N = 72 | 126.9 (126.9-126.9) |
| N = 96 | 150.9 (150.9-150.9) |

- Each extra load adds one cycle up to N = 64, then about 1.0 to 1.1 more. A 32-bit LDS load completes in 32 cycles
  and the wave issues one per cycle, so no more than about 32 are ever in flight and the 6-bit DS counter (63, READ)
  is never reached in this test: the counter limit could not be shown with conflict-free loads. INFERRED: a wave
  stalls on the counter only with slow (conflicting) loads.
- Waits the compiler inserted: none inside the timed loops of `lds_probe` except one `s_wait_loadcnt 0` per loop
  iteration (nothing is outstanding then; it is present in the U = 4 and U = 8 builds alike, so it cancels).

Several waves (cycle counters; one work-group, `waves`; aggregate bytes per cycle over the span of all waves):

| Mode | Load | Waves | CUs | SIMDs | LDS allocated | Bytes per cycle (all waves) |
|---|---|---|---|---|---|---|
| WGP | `ds_load_b128` | 1 | 1 | 1 | 64 KiB | 84.9 |
| WGP | `ds_load_b128` | 2 | 1 | 2 | 64 KiB | 85.1 |
| WGP | `ds_load_b128` | 4 | 2 | 4 | 64 KiB | 127.8 |
| WGP | `ds_load_b128` | 8 | 2 | 4 | 64 KiB | 127.9 |
| WGP | `ds_load_b128` | 16 | 2 | 4 | 64 KiB | 127.9 |
| WGP | `ds_load_b128` | 32 | 2 | 4 | 64 KiB | 128.0 |
| WGP | `ds_load_b32` | 1 | 1 | 1 | 64 KiB | 59.7 |
| WGP | `ds_load_b32` | 2 | 1 | 2 | 64 KiB | 119.2 |
| WGP | `ds_load_b32` | 4 | 2 | 4 | 64 KiB | 126.1 |
| WGP | `ds_load_b32` | 8 | 2 | 4 | 64 KiB | 127.6 |
| WGP | `ds_load_b32` | 16 | 2 | 4 | 64 KiB | 127.9 |
| WGP | `ds_load_b32` | 32 | 2 | 4 | 64 KiB | 128.0 |
| WGP | `ds_load_b128` | 1 | 1 | 1 | 8 KiB | 84.9 |
| WGP | `ds_load_b128` | 4 | 2 | 4 | 8 KiB | 127.7 |
| CU | `ds_load_b128` | 1 | 1 | 1 | 64 KiB | 84.9 |
| CU | `ds_load_b128` | 2 | 1 | 2 | 64 KiB | 85.2 |
| CU | `ds_load_b128` | 4 | 1 | 2 | 64 KiB | 85.2 |
| CU | `ds_load_b128` | 8 | 1 | 2 | 64 KiB | 85.3 |
| CU | `ds_load_b128` | 16 | 1 | 2 | 64 KiB | 85.3 |
| CU | `ds_load_b128` | 32 | 1 | 2 | 64 KiB | 85.3 |
| CU | `ds_load_b32` | 1 | 1 | 1 | 64 KiB | 56.4 |
| CU | `ds_load_b32` | 2 | 1 | 2 | 64 KiB | 101.6 |
| CU | `ds_load_b32` | 4 | 1 | 2 | 64 KiB | 122.4 |
| CU | `ds_load_b32` | 8 | 1 | 2 | 64 KiB | 124.8 |
| CU | `ds_load_b32` | 16 | 1 | 2 | 64 KiB | 126.0 |
| CU | `ds_load_b32` | 32 | 1 | 2 | 64 KiB | 126.7 |
| CU | `ds_load_b128` | 1 | 1 | 1 | 8 KiB | 84.9 |
| CU | `ds_load_b128` | 4 | 1 | 2 | 8 KiB | 85.2 |

- One CU's LDS port gives 128 bytes per cycle with 32-bit loads and 85 with 128-bit loads, whatever the number of
  waves (2 to 32) or SIMDs: the 128-bit limit is per CU, not per wave.
- In WGP mode (the HIP default) two waves land on the two SIMDs of one CU; four waves use both CUs, and still get
  only 128 bytes per cycle together.

Whole card (`card`, 512 groups of 16 waves, 32 KiB LDS each, REALTIME span):

| Mode | Load | Bytes per cycle, card | Per CU | TB/s (clock in GHz) |
|---|---|---|---|---|
| WGP | `ds_load_b128` | 4105 | 64.1 | 11.4 (2.77) |
| WGP | `ds_load_b128` | 4110 | 64.2 | 11.4 (2.77) |
| WGP | `ds_load_b32` | 4186 | 65.4 | 11.5 (2.76) |
| CU | `ds_load_b128` | 5450 | 85.2 | 15.0 (2.76) |
| CU | `ds_load_b128` | 5449 | 85.1 | 15.0 (2.76) |
| CU | `ds_load_b32` | 8025 | 125.4 | 21.9 (2.74) |

- CU mode: 85 bytes per cycle per CU with 128-bit loads, 125 with 32-bit loads (64 CUs). WGP mode: 64 per CU (128
  per WGP) for both widths. INFERRED: a WGP-mode work-group's LDS space is reached through one port for the WGP; the
  guide says the 128 KB of a WGP are two sets of 32 banks, each serving one pair of SIMDs (READ), which matches CU
  mode, where each CU gets its own set.
- Against lane 2's global loads: one wave reads 128 bytes per cycle from L0 with 16-byte loads (4 cycles per load),
  and 85 to 128 from LDS; LDS's advantage is latency (32 against 81 cycles), conflict-free random access, and its
  data does not need to be in a cache.

## 2. LDS access patterns and bank conflicts (`lds_probe pattern`)

Model: 32 banks of 4 bytes, bank = (byte address / 4) mod 32; a load takes as many passes as its most used bank,
where lanes reading the same dword count once. For 64-bit loads the model is applied to each half-wave and the two
added; for 128-bit loads to each quarter-wave (8 lanes), with a floor of 6 cycles.

| Load | Pattern | Model passes (wave / half / quarter) | Cycles per access | Issue to data |
|---|---|---|---|---|
| `ds_load_b32` | same_address | 1 / 2 / 4 | 1.00 (1.00-1.00) | 32.00 |
| `ds_load_b32` | sequential | 1 / 2 / 4 | 1.00 (1.00-1.19) | 32.00 |
| `ds_load_b32` | stride_2 | 2 / 2 / 4 | 2.00 (2.00-2.00) | 33.00 |
| `ds_load_b32` | stride_3 | 1 / 2 / 4 | 1.00 (1.00-1.00) | 32.00 |
| `ds_load_b32` | stride_4 | 4 / 4 / 4 | 4.00 (4.00-4.00) | 35.00 |
| `ds_load_b32` | stride_8 | 8 / 8 / 8 | 8.00 (8.00-8.00) | 39.00 |
| `ds_load_b32` | stride_16 | 16 / 16 / 16 | 16.00 (16.00-16.04) | 47.00 |
| `ds_load_b32` | stride_32 | 32 / 32 / 32 | 32.00 (32.00-32.00) | 63.00 |
| `ds_load_b32` | stride_33 | 1 / 2 / 4 | 1.00 (1.00-1.00) | 32.00 |
| `ds_load_b32` | stride_64 | 32 / 32 / 32 | 32.00 (32.00-32.00) | 63.00 |
| `ds_load_b32` | one_pair_collides | 2 / 3 / 5 | 2.00 (2.00-2.00) | 33.00 |
| `ds_load_b32` | halves_same | 1 / 2 / 4 | 1.00 (1.00-1.00) | 32.00 |
| `ds_load_b32` | random_0 | 3 / 4 / 7 | 3.00 (3.00-3.00) | 34.00 |
| `ds_load_b32` | random_1 | 3 / 5 / 8 | 3.00 (3.00-3.00) | 34.00 |
| `ds_load_b32` | random_2 | 3 / 4 / 6 | 3.00 (3.00-3.24) | 34.00 |
| `ds_load_b32` | random_3 | 4 / 4 / 7 | 4.00 (4.00-4.00) | 35.00 |
| `ds_load_b32` | random_4 | 4 / 5 / 7 | 4.00 (4.00-4.00) | 35.00 |
| `ds_load_b32` | random_5 | 3 / 5 / 6 | 3.00 (3.00-3.00) | 34.00 |
| `ds_load_b64` | same_address | 1 / 2 / 4 | 2.00 (1.92-2.00) | 34.00 |
| `ds_load_b64` | sequential | 2 / 2 / 4 | 2.00 (2.00-2.00) | 34.00 |
| `ds_load_b64` | stride_2 | 4 / 4 / 4 | 4.00 (4.00-4.00) | 36.00 |
| `ds_load_b64` | stride_3 | 2 / 2 / 4 | 2.00 (2.00-2.00) | 34.00 |
| `ds_load_b64` | stride_4 | 8 / 8 / 8 | 8.00 (8.00-8.00) | 40.00 |
| `ds_load_b64` | stride_8 | 16 / 16 / 16 | 16.00 (16.00-16.00) | 48.00 |
| `ds_load_b64` | stride_33 | 2 / 2 / 4 | 2.00 (2.00-2.00) | 34.00 |
| `ds_load_b64` | one_pair_collides | 3 / 3 / 5 | 3.00 (3.00-3.00) | 35.00 |
| `ds_load_b64` | halves_same | 1 / 2 / 4 | 2.00 (2.00-2.00) | 34.00 |
| `ds_load_b64` | random_0 | 5 / 6 / 7 | 6.00 (6.00-6.00) | 38.00 |
| `ds_load_b64` | random_1 | 4 / 7 / 10 | 7.00 (7.00-7.00) | 39.00 |
| `ds_load_b64` | random_2 | 4 / 6 / 7 | 6.00 (6.00-6.00) | 38.00 |
| `ds_load_b64` | random_3 | 4 / 7 / 9 | 7.00 (7.00-7.00) | 39.00 |
| `ds_load_b64` | random_4 | 4 / 5 / 9 | 5.00 (5.00-5.00) | 37.00 |
| `ds_load_b64` | random_5 | 4 / 7 / 8 | 7.00 (7.00-7.00) | 39.00 |
| `ds_load_b128` | same_address | 1 / 2 / 4 | 6.00 (6.00-6.00) | 38.00 |
| `ds_load_b128` | sequential | 4 / 4 / 4 | 6.00 (6.00-6.00) | 38.00 |
| `ds_load_b128` | stride_2 | 8 / 8 / 8 | 8.00 (8.00-8.00) | 42.00 |
| `ds_load_b128` | stride_3 | 4 / 4 / 4 | 6.00 (6.00-6.13) | 38.00 |
| `ds_load_b128` | stride_4 | 16 / 16 / 16 | 16.00 (16.00-16.02) | 50.00 |
| `ds_load_b128` | stride_8 | 32 / 32 / 32 | 32.00 (32.00-32.00) | 66.00 |
| `ds_load_b128` | stride_33 | 4 / 4 / 4 | 6.00 (6.00-6.00) | 38.00 |
| `ds_load_b128` | one_pair_collides | 5 / 5 / 5 | 6.50 (6.50-6.50) | 39.00 |
| `ds_load_b128` | halves_same | 2 / 4 / 4 | 6.00 (5.93-6.00) | 38.00 |
| `ds_load_b128` | random_0 | 8 / 10 / 10 | 10.00 (9.94-10.08) | 44.00 |
| `ds_load_b128` | random_1 | 6 / 11 / 12 | 12.00 (12.00-12.04) | 46.00 |
| `ds_load_b128` | random_2 | 6 / 9 / 10 | 10.00 (9.98-10.09) | 44.00 |
| `ds_load_b128` | random_3 | 6 / 8 / 12 | 12.00 (12.00-12.05) | 46.00 |
| `ds_load_b128` | random_4 | 7 / 8 / 10 | 10.00 (9.95-10.11) | 44.00 |
| `ds_load_b128` | random_5 | 7 / 7 / 8 | 8.00 (8.00-8.00) | 42.00 |

- 32-bit loads: all 18 patterns match the model exactly (the probe's verdict text says "of 20"; 18 were run), one cycle per pass (stride 2 dwords: 2, stride 4: 4, ...,
  stride 32 or 64: 32 cycles; stride 3 and 33: 1). All lanes on one address, or two half-waves on the same 16
  dwords, cost 1: equal dwords are broadcast. Random addresses: 3 cycles (3 passes). The issue-to-data time grows by
  the same number of cycles minus one.
- 64-bit loads: all 15 patterns match the model applied per half-wave (stride 2 elements: 4 cycles, 8: 16).
- 128-bit loads: the quarter-wave model with a 6-cycle floor fits 14 of 15 patterns (the one-pair collision costs
  6.5, not the modelled 5 passes; this fit is not judged by the probe); random addresses 8 to 12 cycles.
- Alignment: every pattern here is aligned to its access size. Misaligned LDS access was not tested.
- Store-to-load: see item 1 (36 cycles, correct value). Another wave's store is visible after a barrier (used by every
  probe's LDS fill; not timed).

Size limits and placement (`size`, both modes; READ from the runtime: `sharedMemPerBlock` 65536,
`maxSharedMemoryPerMultiProcessor` 131072):

| Mode | Request | Result |
|---|---|---|
| WGP | launch with 64 KiB | no error |
| WGP | launch with 65 KiB | invalid argument |
| WGP | launch with 128 KiB | invalid argument |
| WGP | one-wave groups at once, 1 KiB each | 64 per WGP (all 32 units) |
| WGP | one-wave groups at once, 16 KiB each | 8 per WGP (all 32 units) |
| WGP | one-wave groups at once, 32 KiB each | 4 per WGP (all 32 units) |
| WGP | one-wave groups at once, 48 KiB each | 2 per WGP (all 32 units) |
| WGP | one-wave groups at once, 64 KiB each | 2 per WGP (all 32 units) |
| CU | launch with 64 KiB | no error |
| CU | launch with 65 KiB | invalid argument |
| CU | launch with 128 KiB | invalid argument |
| CU | one-wave groups at once, 1 KiB each | 32 per CU (all 64 units) |
| CU | one-wave groups at once, 16 KiB each | 4 per CU (all 64 units) |
| CU | one-wave groups at once, 32 KiB each | 2 per CU (all 64 units) |
| CU | one-wave groups at once, 48 KiB each | 1 per CU (all 64 units) |
| CU | one-wave groups at once, 64 KiB each | 1 per CU (all 64 units) |

- HIP accepts 64 KiB of LDS per work-group; 65 KiB and 128 KiB are refused at launch ("invalid argument").
- Work-groups per WGP = floor(128 KiB / allocation), in both modes (CU mode: floor(64 KiB / allocation) per CU), until
  the wave limit (64 one-wave groups per WGP in WGP mode, 32 per CU in CU mode) takes over. A 48 KiB group already
  allows only 2 per WGP, like 64 KiB.
- The allocation size does not change speed: one wave reads at 84.9 bytes per cycle with 8 KiB or 64 KiB allocated,
  four waves at 127.5 and 127.7.

## 3. LDS as a staging buffer for matrix work (`lds_stage feed`)

Every matrix instruction is preceded by loads of an A and a B tile: FP16 1024 bytes (two 512-byte tiles), FP8 512 bytes. The probe reloads the same data each time and does not check the loaded values or the matrix results; the expectation file pins the instruction counts of the loop (for example 112 matrix instructions and 224 `ds_load_b128`), so the loads are there. That reusing tiles from registers reaches the full rate is INFERRED from the register-only rows; no reuse variant was run. Four operand
sets rotate: wait for the oldest set, multiply, reload the set used one instruction earlier. Per-SIMD cycles per
instruction come from the REALTIME span of all waves (median of 3 launches; the busiest SIMD). Lane 4: 16 cycles FP16,
8 cycles FP8. (The brief's "512 bytes of A and B per 16 cycles" is one operand; A and B together are 1024.)

| Mode | Operands from | Waves | Waves per SIMD | Cycles per instruction per SIMD (3 launches) | Of the matrix rate | Operand bytes per cycle per SIMD |
|---|---|---|---|---|---|---|
| WGP | f16_reg | 1 | 1 | 16.0 (16.0-16.0) | 100 % | 0.0 |
| WGP | f16_reg | 2 | 1 | 16.0 (16.0-16.0) | 100 % | 0.0 |
| WGP | f16_reg | 4 | 1 | 16.0 (16.0-16.0) | 100 % | 0.0 |
| WGP | f16_reg | 8 | 2 | 16.4 (16.4-16.4) | 98 % | 0.0 |
| WGP | f16_reg | 16 | 4 | 16.5 (16.5-16.5) | 97 % | 0.0 |
| WGP | f16_lds_b128 | 1 | 1 | 45.8 (45.8-54.3) | 35 % | 22.4 |
| WGP | f16_lds_b128 | 2 | 1 | 45.8 (45.8-45.8) | 35 % | 22.4 |
| WGP | f16_lds_b128 | 4 | 1 | 45.4 (45.4-53.7) | 35 % | 22.5 |
| WGP | f16_lds_b128 | 8 | 2 | 32.0 (32.0-33.6) | 50 % | 32.0 |
| WGP | f16_lds_b128 | 16 | 4 | 34.4 (34.4-34.4) | 47 % | 29.8 |
| WGP | f16_global_b128+tr_b128 | 1 | 1 | 44.9 (44.9-44.9) | 36 % | 22.8 |
| WGP | f16_global_b128+tr_b128 | 2 | 1 | 44.9 (44.9-44.9) | 36 % | 22.8 |
| WGP | f16_global_b128+tr_b128 | 4 | 1 | 44.9 (44.9-44.9) | 36 % | 22.8 |
| WGP | f16_global_b128+tr_b128 | 8 | 2 | 24.0 (24.0-24.0) | 67 % | 42.6 |
| WGP | f16_global_b128+tr_b128 | 16 | 4 | 24.0 (24.0-26.8) | 67 % | 42.7 |
| WGP | fp8_reg | 1 | 1 | 8.0 (8.0-8.0) | 100 % | 0.0 |
| WGP | fp8_reg | 2 | 1 | 8.0 (8.0-8.0) | 100 % | 0.0 |
| WGP | fp8_reg | 4 | 1 | 8.0 (8.0-8.0) | 100 % | 0.0 |
| WGP | fp8_reg | 8 | 2 | 8.3 (8.3-8.3) | 96 % | 0.0 |
| WGP | fp8_reg | 16 | 4 | 8.4 (8.4-8.4) | 95 % | 0.0 |
| WGP | fp8_lds_b64 | 1 | 1 | 36.3 (36.3-36.8) | 22 % | 14.1 |
| WGP | fp8_lds_b64 | 2 | 1 | 36.8 (36.8-36.8) | 22 % | 13.9 |
| WGP | fp8_lds_b64 | 4 | 1 | 36.3 (36.3-36.4) | 22 % | 14.1 |
| WGP | fp8_lds_b64 | 8 | 2 | 18.3 (18.3-18.3) | 44 % | 28.0 |
| WGP | fp8_lds_b64 | 16 | 4 | 17.8 (17.7-17.8) | 45 % | 28.8 |
| WGP | fp8_global_b64+tr_b64 | 1 | 1 | 36.7 (36.7-36.7) | 22 % | 14.0 |
| WGP | fp8_global_b64+tr_b64 | 2 | 1 | 36.7 (36.7-38.5) | 22 % | 14.0 |
| WGP | fp8_global_b64+tr_b64 | 4 | 1 | 36.7 (36.7-36.7) | 22 % | 14.0 |
| WGP | fp8_global_b64+tr_b64 | 8 | 2 | 18.5 (18.4-19.3) | 43 % | 27.7 |
| WGP | fp8_global_b64+tr_b64 | 16 | 4 | 11.2 (11.1-11.2) | 72 % | 45.9 |
| CU | f16_reg | 1 | 1 | 16.0 (16.0-16.0) | 100 % | 0.0 |
| CU | f16_reg | 2 | 1 | 16.0 (16.0-16.0) | 100 % | 0.0 |
| CU | f16_reg | 4 | 2 | 16.4 (16.4-16.4) | 98 % | 0.0 |
| CU | f16_reg | 8 | 4 | 16.5 (16.5-16.5) | 97 % | 0.0 |
| CU | f16_reg | 16 | 8 | 16.6 (16.6-16.6) | 96 % | 0.0 |
| CU | f16_lds_b128 | 1 | 1 | 45.8 (45.8-50.9) | 35 % | 22.4 |
| CU | f16_lds_b128 | 2 | 1 | 45.8 (45.8-45.8) | 35 % | 22.4 |
| CU | f16_lds_b128 | 4 | 2 | 24.2 (24.0-28.1) | 66 % | 42.4 |
| CU | f16_lds_b128 | 8 | 4 | 26.8 (26.8-26.9) | 60 % | 38.2 |
| CU | f16_lds_b128 | 16 | 8 | 26.9 (26.9-27.0) | 59 % | 38.0 |
| CU | f16_global_b128+tr_b128 | 1 | 1 | 44.8 (44.8-47.2) | 36 % | 22.8 |
| CU | f16_global_b128+tr_b128 | 2 | 1 | 44.9 (44.8-44.9) | 36 % | 22.8 |
| CU | f16_global_b128+tr_b128 | 4 | 2 | 24.0 (24.0-28.7) | 67 % | 42.6 |
| CU | f16_global_b128+tr_b128 | 8 | 4 | 24.0 (24.0-27.0) | 67 % | 42.7 |
| CU | f16_global_b128+tr_b128 | 16 | 8 | 24.4 (24.0-25.4) | 66 % | 41.9 |
| CU | fp8_reg | 1 | 1 | 8.0 (8.0-8.0) | 100 % | 0.0 |
| CU | fp8_reg | 2 | 1 | 8.0 (8.0-8.0) | 100 % | 0.0 |
| CU | fp8_reg | 4 | 2 | 8.3 (8.3-8.3) | 96 % | 0.0 |
| CU | fp8_reg | 8 | 4 | 8.4 (8.4-8.4) | 95 % | 0.0 |
| CU | fp8_reg | 16 | 8 | 8.5 (8.5-8.5) | 94 % | 0.0 |
| CU | fp8_lds_b64 | 1 | 1 | 36.3 (36.3-36.8) | 22 % | 14.1 |
| CU | fp8_lds_b64 | 2 | 1 | 36.8 (36.8-36.8) | 22 % | 13.9 |
| CU | fp8_lds_b64 | 4 | 2 | 18.4 (18.4-18.4) | 43 % | 27.8 |
| CU | fp8_lds_b64 | 8 | 4 | 11.0 (11.0-12.0) | 73 % | 46.7 |
| CU | fp8_lds_b64 | 16 | 8 | 10.8 (10.8-10.8) | 74 % | 47.4 |
| CU | fp8_global_b64+tr_b64 | 1 | 1 | 36.6 (36.6-36.7) | 22 % | 14.0 |
| CU | fp8_global_b64+tr_b64 | 2 | 1 | 36.7 (36.6-36.7) | 22 % | 14.0 |
| CU | fp8_global_b64+tr_b64 | 4 | 2 | 18.4 (18.4-19.3) | 43 % | 27.8 |
| CU | fp8_global_b64+tr_b64 | 8 | 4 | 11.1 (11.1-11.1) | 72 % | 46.0 |
| CU | fp8_global_b64+tr_b64 | 16 | 8 | 11.0 (11.0-11.7) | 73 % | 46.6 |

- One wave per SIMD: 45 to 46 cycles per FP16 instruction from LDS or from L0 (35 % of the matrix rate), 36 to 37 per
  FP8 instruction (22 %). The wave cannot issue its loads while a matrix instruction runs (lane 4, item 8), and each
  arrival of load data costs it 12 to 18 cycles of issue.
- Two or more waves per SIMD: FP16 24 to 27 cycles per instruction (59 to 67 %) from LDS in CU mode, 24 (67 %) from
  L0 in either mode; FP8 11 cycles (72 to 74 %) with 4 or more waves per SIMD from LDS in CU mode or from L0.
  In WGP mode LDS gives FP16 32 to 34 cycles and FP8 18 (45 %).
- The FP16 limits match the measured ports: in CU mode two SIMDs need 2 x 1024 bytes per 24 cycles = 85 bytes per
  cycle, which is the CU's 128-bit LDS rate; L0 delivers A (4 cycles per 512 bytes) plus the transposing B (8 cycles)
  = 85 bytes per cycle per CU as well. In WGP mode four SIMDs share 128 bytes per cycle: 32 cycles per FP16
  instruction (measured 32.0 to 34.4) and 16 per FP8 instruction (measured 17.8 to 18.3).
- INFERRED: LDS staging does not feed the matrix unit at full rate without reuse in registers. At full FP16 rate one
  CU needs 128 bytes per cycle (two SIMDs x 1024 bytes / 16 cycles), FP8 256. LDS gives 85 (128-bit) or 128 (32- and
  64-bit loads) per CU in CU mode, 128 per WGP in WGP mode. A kernel that reuses each A and B tile for two
  instructions (2 x 2 register blocking) halves the need: FP16 then fits LDS in CU mode with 64-bit loads; FP8 needs
  4 x 4 blocking. What LDS buys over global loads is the data exchange between waves and conflict-free random access,
  not bandwidth: from L0, global loads deliver the same 85 bytes per cycle per CU for this pattern.
- Staging cost (INFERRED from item 1): writing one 512-byte tile into LDS costs one `ds_store_b128` (5 cycles of the
  CU's LDS port, plus the global load that brought it), reading it back one `ds_load_b128` (6 cycles) or two
  `ds_load_b64` (4 cycles).

## 4. Scalar memory loads (`lds_smem`, written by a helper, rerun by the lead)

Scalar pointer chase (`s_load_b64` + `s_wait_kmcnt 0`, random cycle, warmed; cycles per load, median of repetitions with min-max):

| Footprint | node every 32 B | node every 64 B | node every 128 B | node every 256 B |
|---|---|---|---|---|
| 1 KiB | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) |
| 2 KiB | 36.5 (36.5-37.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-37.0) |
| 4 KiB | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) |
| 8 KiB | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-39.3) | 36.5 (36.5-36.5) |
| 12 KiB | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) |
| 16 KiB | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) | 36.5 (36.5-36.5) |
| 24 KiB | 137.6 (137.3-142.3) | 195.5 (195.5-195.5) | 189.5 (189.5-192.6) | 36.5 (36.5-36.5) |
| 32 KiB | 143.5 (143.5-148.2) | 199.0 (199.0-199.0) | 213.5 (213.5-217.6) | 36.5 (36.5-36.5) |
| 48 KiB | 183.8 (183.4-189.7) | 217.0 (217.0-223.0) | 189.5 (189.5-195.4) | 213.5 (213.5-219.3) |
| 64 KiB | 174.5 (174.5-182.7) | 199.0 (199.0-207.1) | 189.5 (189.5-197.2) | 207.5 (207.5-215.1) |
| 128 KiB | 189.7 (189.7-199.7) | 199.0 (199.0-214.4) | 189.5 (189.5-204.8) | 189.5 (189.5-189.5) |
| 256 KiB | 211.3 (211.3-242.2) | 218.0 (218.0-255.9) | 217.0 (217.0-247.4) | 207.5 (207.5-236.6) |
| 1024 KiB | 265.3 (197.7-292.7) | 250.4 (212.0-307.3) | 207.5 (207.5-319.2) | 207.5 (207.5-273.0) |
| 4096 KiB | 330.0 (258.8-369.9) | 317.8 (271.8-365.7) | 318.8 (214.4-369.6) | 406.7 (189.6-418.2) |
| 16384 KiB | 395.3 (391.3-410.7) | 408.8 (366.3-419.7) | 428.3 (420.7-441.5) | 444.2 (443.5-445.5) |
| 65536 KiB | 440.5 (436.1-445.8) | 441.7 (435.2-443.3) | 440.1 (436.7-443.4) | 506.6 (503.1-510.7) |

```
lds_smem chase_step stride 32 base_cycles 36.5 last_footprint_KiB_below_1.3x 16 first_footprint_KiB_above_1.3x 24 lines_at_last_low 512
lds_smem chase_step stride 64 base_cycles 36.5 last_footprint_KiB_below_1.3x 16 first_footprint_KiB_above_1.3x 24 lines_at_last_low 256
lds_smem chase_step stride 128 base_cycles 36.5 last_footprint_KiB_below_1.3x 16 first_footprint_KiB_above_1.3x 24 lines_at_last_low 128
lds_smem chase_step stride 256 base_cycles 36.5 last_footprint_KiB_below_1.3x 32 first_footprint_KiB_above_1.3x 48 lines_at_last_low 128
lds_smem chase mode cold stride 64 footprint_KiB 256 nodes 4096 cycles_per_load 351.8 min 351.8 max 351.8 final_pointer ok
lds_smem chase mode cold stride 64 footprint_KiB 65536 nodes 1048576 cycles_per_load 701.8 min 701.8 max 701.8 final_pointer ok
lds_smem chase mode cold stride 64 footprint_KiB 256 nodes 4096 cycles_per_load 315.9 min 315.9 max 315.9 final_pointer ok
lds_smem chase mode cold stride 64 footprint_KiB 65536 nodes 1048576 cycles_per_load 666.3 min 666.3 max 666.3 final_pointer ok
lds_smem chase mode cold stride 64 footprint_KiB 256 nodes 4096 cycles_per_load 316.0 min 316.0 max 316.0 final_pointer ok
lds_smem chase mode cold stride 64 footprint_KiB 65536 nodes 1048576 cycles_per_load 653.0 min 653.0 max 653.0 final_pointer ok
```

Independent scalar loads from a hot 4 KiB buffer, N per batch then one wait (cycles per batch include about 20 cycles of loop and wait):

| Width | Batch of 1 (cycles) | Batch of 8 | Batch of 16 | Largest batch | Cycles per extra load (1 to largest) | Best bytes per cycle |
|---|---|---|---|---|---|---|
| `s_load_b32` | 46.1 | 61.1 | 69.1 | N = 31: 83.1 | 1.23 | 1.97 |
| `s_load_b64` | 46.1 | 61.1 | 75.1 | N = 31: 102.1 | 1.87 | 3.02 |
| `s_load_b96` | 46.1 | 61.1 | 75.1 | N = 16: 75.1 | 1.93 | 3.49 |
| `s_load_b128` | 46.1 | 61.1 | 75.1 | N = 16: 75.1 | 1.93 | 4.65 |
| `s_load_b256` | 47.1 | 62.1 | nan | N = 8: 62.1 | 2.14 | 6.09 |
| `s_load_b512` | 49.1 | nan | nan | N = 4: 62.1 | 4.33 | 6.09 |
| `s_load_u8` | 46.1 | 61.1 | 69.1 | N = 31: 83.1 | 1.23 | 0.49 |
| `s_load_u16` | 46.1 | 61.1 | 69.1 | N = 31: 83.1 | 1.23 | 0.98 |

Latency and the trip through the vector unit, and scalar loads next to vector loads in flight (raw lines):

```
lds_smem width empty_batch_cycles 20.1 (one s_wait_kmcnt 0 with nothing outstanding + loop)
lds_smem lat load_wait cycles_per_step 34.31 min 34.31 max 34.56 final_s 16 (want 16)
lds_smem lat load_wait_vmov_vadd cycles_per_step 41.81 min 41.81 max 42.31 final_s 16 (want 16)
lds_smem lat load_wait_vmov_readfirstlane_to_soffset cycles_per_step 57.31 min 57.31 max 57.55 final_s 0 (want 0)
lds_smem lat load_wait_vmov_vadd_readfirstlane_to_soffset cycles_per_step 62.31 min 62.31 max 62.80 final_s 0 (want 0)
lds_smem lat_summary load_wait 34.31 plus_vmov_vadd_issue 41.81 round_trip_vmov_rfl 57.31 round_trip_vmov_vadd_rfl 62.31 extra_cost_of_vector_round_trip 23.00
lds_smem calib cycles_per_fma 0.999 min 0.916 max 0.999 samples 21
lds_smem mixed vector_loads 0 scalar hot issue_cycles 28 scalar_load_wait_cycles 62 (min 55 max 64) rest_of_vector_wait_cycles 29 (min 29 max 29) wrong 0
lds_smem mixed vector_loads 16 scalar hot issue_cycles 68 scalar_load_wait_cycles 68 (min 62 max 70) rest_of_vector_wait_cycles 662 (min 629 max 1054) wrong 0
lds_smem mixed vector_loads 32 scalar hot issue_cycles 161 scalar_load_wait_cycles 70 (min 51 max 71) rest_of_vector_wait_cycles 693 (min 578 max 741) wrong 0
lds_smem mixed vector_loads 0 scalar cold issue_cycles 28 scalar_load_wait_cycles 698 (min 598 max 775) rest_of_vector_wait_cycles 29 (min 29 max 29) wrong 0
lds_smem mixed vector_loads 16 scalar cold issue_cycles 68 scalar_load_wait_cycles 638 (min 594 max 772) rest_of_vector_wait_cycles 95 (min 36 max 248) wrong 0
lds_smem mixed vector_loads 32 scalar cold issue_cycles 145 scalar_load_wait_cycles 630 (min 580 max 775) rest_of_vector_wait_cycles 101 (min 28 max 311) wrong 0
lds_smem mixed_summary hot_scalar_alone 62 with16 68 with32 70 | cold_scalar_alone 698 with16 638 with32 630 | vector_wait_after_scalar16 662 vector_wait_after_scalar32 693
```

- The scalar cache holds 16 KiB: 36.5 cycles per dependent load (load, wait and loop; about 26 net of the loop) up to
  16 KiB with nodes every 32, 64 or 128 bytes. With nodes every 256 bytes it still holds 32 KiB (128 nodes), and the
  step comes at 128 nodes whenever nodes are 128 or more bytes apart. INFERRED: 128 lines of 128 bytes (the driver's
  cache table says 64-byte lines, READ; with 64-byte lines, 16 KiB at 64-byte spacing would be 256 lines and 24 KiB
  at 128-byte spacing would fit, and neither matches).
- Past the scalar cache: 140 to 220 cycles up to about 256 KiB, below the vector L2 latency (230 to 255, lane 2);
  320 to 440 cycles at 4 to 16 MiB; 440 to 650 at 64 MiB; a cold (invalidated, flushed) load from memory 650 to 700.
  INFERRED: a second scalar level (the memory model's "scalar L0 per WGP", READ, is the first) or a shorter path
  into L2; not resolved.
- Throughput from the scalar cache: each extra independent load costs about 1.2 cycles (32-bit and smaller), 1.9
  (64 to 128 bits), 2.1 (256) and 4.3 (512); one wave peaks near 6 bytes per cycle. Scalar loads are for
  constants and addresses, not for data. All widths, including the new 8- and 16-bit and 96-bit forms, return the
  right values.
- Scalar after vector: a hot scalar load waited for 62 cycles alone and 68 to 70 with 16 or 32 vector loads to memory
  in flight: scalar loads do not queue behind vector loads (separate counters and paths). A cold one: medians of 630 to 698 in the mixed test.
- The round trip: load and wait 34.3 cycles; plus `v_mov` and a dependent vector add 41.8; back into a scalar
  register with `v_readfirstlane` and into the next load's offset 57.3 (23 cycles for the vector round trip; lane 1:
  25 to 32).

## 5. The transposing loads (`lds_tr`)

Layout (MEASURED with marker data, 8 random sets of per-lane addresses, every element checked):

- `global_load_tr_b128` (wave32, 16-bit elements, 4 VGPRs per lane): lane l, element i (i = 0..7; element i is the
  low half of VGPR i / 2 when i is even, the high half when odd) = 16-bit element (l mod 8) of the 16 bytes at the
  address of lane 8 * (l / 8) + i. Each group of 8 lanes transposes an 8 x 8 block of 16-bit elements.
- `global_load_tr_b64` (wave32, 8-bit elements, 2 VGPRs per lane): lane l, byte i (i = 0..7; bytes 0-3 in the first
  VGPR) = byte (l mod 8) of the 8 bytes at the address of lane 16 * (l / 16) + 4 * ((l / 8) mod 2) + (i mod 4) +
  8 * (i / 4). Each group of 16 lanes transposes 16 rows of 8 bytes into 16 columns, interleaved.
- Every lane supplies its own address; nothing else about the addresses matters (random addresses all over 32 KiB
  give the model's result). Misaligned addresses are used exactly as given: a b128 lane address off by 2, 4 or 8
  bytes, or a b64 address off by 1, 2 or 4, returns the bytes at that exact address (256 of 256 elements), with no
  fault.
- The matrix operand (wave32 B layout: lane c + 16 * (k / 8) holds K = 8 * (k / 8) to 8 * (k / 8) + 7 of column c;
  A is the same with rows) comes out of a row-major K x N matrix B with leading dimension ld when lane s points at:
  16-bit: row 8 * (s / 16) + s mod 8, column 8 * ((s / 8) mod 2); 8-bit: row 8 * (s / 16) + 4 * ((s / 8) mod 2) +
  s mod 4, column 8 * ((s / 4) mod 2). The same addresses give a column-major A (INFERRED from the identical A and B register layouts; only B was run here. The Vulkan lane saw Mesa use the transposing load for row-major B and column-major A). Row-major A and column-major B are
  plain loads (lane r + 16 * (k / 8) reads row r, K from 8 * (k / 8)): Mesa's choice (READ, vulkan-view item 5).
- Products (MEASURED): FP16 and FP8 16 x 16 x 16, A row-major through a plain load, B row-major through the
  transposing load, leading dimension 16 and 64: all 256 outputs equal the CPU result, with no shuffle. The control,
  B through a plain load, gets 253 to 255 of 256 wrong.

| Level | Load | N in flight | Cycles per batch (min-max) | Bytes per cycle |
|---|---|---|---|---|
| L0 | b128 | 1 | 127.2 (127.2-127.2) | 4.0 |
| L0 | b128 | 8 | 155.4 (155.4-155.4) | 26.4 |
| L0 | b128 | 24 | 223.6 (223.6-223.6) | 55.0 |
| L0 | tr_b128 | 1 | 131.2 (131.2-131.2) | 3.9 |
| L0 | tr_b128 | 8 | 187.4 (187.4-187.4) | 21.9 |
| L0 | tr_b128 | 24 | 319.6 (319.6-319.6) | 38.5 |
| L0 | b64 | 1 | 123.2 (123.2-123.2) | 2.1 |
| L0 | b64 | 8 | 137.3 (137.3-137.3) | 14.9 |
| L0 | b64 | 24 | 173.4 (173.4-173.4) | 35.4 |
| L0 | tr_b64 | 1 | 123.2 (123.2-123.2) | 2.1 |
| L0 | tr_b64 | 8 | 137.3 (137.3-137.3) | 14.9 |
| L0 | tr_b64 | 24 | 173.4 (173.4-173.4) | 35.4 |
| L2 | b128 | 1 | 285.2 (279.2-500.4) | 1.8 |
| L2 | b128 | 8 | 317.3 (311.3-340.5) | 12.9 |
| L2 | b128 | 24 | 401.4 (384.3-408.8) | 30.6 |
| L2 | tr_b128 | 1 | 295.2 (289.2-499.5) | 1.7 |
| L2 | tr_b128 | 8 | 352.6 (346.6-358.6) | 11.6 |
| L2 | tr_b128 | 24 | 481.9 (475.9-498.3) | 25.5 |
| L2 | b64 | 1 | 289.2 (280.7-502.6) | 0.9 |
| L2 | b64 | 8 | 305.4 (298.7-311.4) | 6.7 |
| L2 | b64 | 24 | 343.8 (337.8-519.9) | 17.9 |
| L2 | tr_b64 | 1 | 292.7 (280.7-514.9) | 0.9 |
| L2 | tr_b64 | 8 | 310.7 (292.5-317.4) | 6.6 |
| L2 | tr_b64 | 24 | 347.1 (340.0-353.1) | 17.7 |
| memory | b128 | 1 | 803.5 (790.7-826.7) | 0.6 |
| memory | b128 | 8 | 838.1 (828.6-851.8) | 4.9 |
| memory | b128 | 24 | 972.9 (930.9-1098.8) | 12.6 |
| memory | tr_b128 | 1 | 812.3 (804.5-817.6) | 0.6 |
| memory | tr_b128 | 8 | 872.8 (737.7-879.1) | 4.7 |
| memory | tr_b128 | 24 | 1024.5 (1002.2-1132.2) | 12.0 |
| memory | b64 | 1 | 863.9 (847.9-876.2) | 0.3 |
| memory | b64 | 8 | 819.5 (810.0-837.9) | 2.5 |
| memory | b64 | 24 | 866.1 (837.4-878.6) | 7.1 |
| memory | tr_b64 | 1 | 853.9 (842.8-859.0) | 0.3 |
| memory | tr_b64 | 8 | 821.7 (798.0-830.9) | 2.5 |
| memory | tr_b64 | 24 | 873.9 (867.7-897.8) | 7.0 |

| Window (whole card, 4 loads in flight per wave, 4 waves per SIMD) | Load | GB/s (min-max) |
|---|---|---|
| L0_16KiB | b128 | 16800 (16364-17223) |
| L0_16KiB | tr_b128 | 10143 (9982-10226) |
| L0_16KiB | b64 | 11583 (11506-11594) |
| L0_16KiB | tr_b64 | 11564 (11318-11583) |
| L2_2MiB | b128 | 9523 (9503-9619) |
| L2_2MiB | tr_b128 | 8576 (8467-8766) |
| L2_2MiB | b64 | 11474 (9271-11532) |
| L2_2MiB | tr_b64 | 11586 (11511-11629) |
| MALL_32MiB | b128 | 5409 (5179-7206) |
| MALL_32MiB | tr_b128 | 5346 (5152-7129) |
| MALL_32MiB | b64 | 5399 (5287-6363) |
| MALL_32MiB | tr_b64 | 5322 (4639-5753) |
| DRAM_512MiB | b128 | 629 (628-630) |
| DRAM_512MiB | tr_b128 | 629 (628-630) |
| DRAM_512MiB | b64 | 628 (626-630) |
| DRAM_512MiB | tr_b64 | 629 (626-630) |

- Latency (N = 1, issue to data): the transposing b128 load is 4 to 10 cycles slower than the plain one (L0 131
  against 127; L2 295 against 285; memory 812 against 804); the b64 forms are equal within the spread.
- Issue rate from L0: each extra transposing b128 load costs 8 cycles against 4 for a plain b128 (one wave reaches
  38.5 against 55 bytes per cycle at N = 24); the b64 forms both cost 2. From L2 the b128 difference is smaller
  (25.5 against 30.6 bytes per cycle at N = 24); from memory 12.0 against 12.6.
- Whole card: from DRAM and from a 32 MiB window both forms read at the same speed (628 to 629 GB/s; 5.3 to
  5.4 TB/s). From a 2 MiB window (L2) the b128 transposing load reaches 8.6 against 9.5 TB/s, from a 16 KiB window
  (L0) 10.1 against 16.8 TB/s. The 32 MiB window reads faster than lane 2's memory-side cache figure (about 2.0 TB/s)
  because neighbouring waves read neighbouring tiles at the same time; it is not comparable and its cause was not
  studied.
- Loads in flight: the same as plain loads (tested up to N = 24 per wave; lane 2's limit of 63 was not retested).
- What a kernel author gains: B row-major (or A column-major) goes straight from memory into the operand registers,
  with no LDS round trip and no shuffle; RDNA 3 had no such load (its write-up and AMD's GPUOpen article, READ, build
  transposes from shuffles or an identity multiply). From DRAM it is free; from L0 it halves the issue rate of that
  operand, which matters only for kernels that reload operands from L0 for every instruction.

## 6. Block register loads and stores (`lds_block`, written by a helper, rerun by the lead)

Layout (marker data; mismatches against the guide's rule and against addresses rounded down to 4 bytes):

```
lds_block layout_load aligned_full mask=0xFFFFFFFF stride=128 extra=0 mismatches_vs_rule=0 mismatches_vs_floor=0
lds_block layout_load aligned_low16 mask=0x0000FFFF stride=128 extra=0 mismatches_vs_rule=0 mismatches_vs_floor=0
lds_block layout_load aligned_odd_regs mask=0xAAAAAAAA stride=128 extra=0 mismatches_vs_rule=0 mismatches_vs_floor=0
lds_block layout_load aligned_one_reg mask=0x00000001 stride=128 extra=0 mismatches_vs_rule=0 mismatches_vs_floor=0
lds_block layout_store aligned_full mask=0xFFFFFFFF stride=128 extra=0 words_changed=1024 diff_exact_lane0first=0 diff_exact_lane31first=0 diff_floor=0
lds_block layout_store aligned_low16 mask=0x0000FFFF stride=128 extra=0 words_changed=512 diff_exact_lane0first=0 diff_exact_lane31first=0 diff_floor=0
lds_block layout_store aligned_odd_regs mask=0xAAAAAAAA stride=128 extra=0 words_changed=512 diff_exact_lane0first=0 diff_exact_lane31first=0 diff_floor=0
lds_block layout_store aligned_one_reg mask=0x00000001 stride=128 extra=0 words_changed=32 diff_exact_lane0first=0 diff_exact_lane31first=0 diff_floor=0
lds_block layout_load misaligned_by_2 mask=0xFFFFFFFF stride=128 extra=2 mismatches_vs_rule=0 mismatches_vs_floor=1024
lds_block layout_load misaligned_by_1 mask=0xFFFFFFFF stride=128 extra=1 mismatches_vs_rule=0 mismatches_vs_floor=1024
lds_block layout_load overlap_16B_lanes mask=0xFFFFFFFF stride=16 extra=0 mismatches_vs_rule=0 mismatches_vs_floor=0
lds_block layout_store misaligned_by_2 mask=0xFFFFFFFF stride=128 extra=2 words_changed=1025 diff_exact_lane0first=0 diff_exact_lane31first=0 diff_floor=1025
lds_block layout_store overlap_16B_lanes mask=0xFFFFFFFF stride=16 extra=0 words_changed=156 diff_exact_lane0first=148 diff_exact_lane31first=0 diff_floor=148
```

Cost (one wave; 'blk32' = one `global_load_block` of 32 VGPRs, 128 bytes per lane; 'blk16' = mask 0x0000FFFF; 'b128_lane128' = eight `global_load_b128` covering the same 128 bytes per lane; 'b128_coalesced' = eight `global_load_b128` with consecutive lanes on consecutive 16 bytes):

| Data | Load | N per batch | Cycles per batch (min-max) | Bytes per cycle |
|---|---|---|---|---|
| hot | blk32 | 1 | 360.0 (360.0-360.5) | 11.38 |
| hot | blk32 | 2 | 616.0 (616.0-616.4) | 13.30 |
| hot | blk32 | 4 | 1125.0 (1125.0-1125.1) | 14.56 |
| hot | blk32 | 8 | 2149.0 (2149.0-2149.3) | 15.25 |
| hot | blk16 | 1 | 233.0 (233.0-233.0) | 8.79 |
| hot | blk16 | 2 | 360.0 (360.0-360.0) | 11.38 |
| hot | blk16 | 4 | 614.0 (614.0-614.0) | 13.34 |
| hot | blk16 | 8 | 1127.0 (1127.0-1127.4) | 14.54 |
| hot | b128_lane128 | 1 | 172.0 (172.0-172.0) | 23.81 |
| hot | b128_lane128 | 2 | 236.0 (236.0-236.0) | 34.71 |
| hot | b128_lane128 | 4 | 356.0 (356.0-356.0) | 46.02 |
| hot | b128_lane128 | 8 | 612.0 (612.0-612.8) | 53.54 |
| hot | b128_coalesced | 1 | 137.0 (137.0-137.0) | 29.89 |
| hot | b128_coalesced | 2 | 169.0 (169.0-169.0) | 48.47 |
| hot | b128_coalesced | 4 | 225.0 (225.0-225.0) | 72.81 |
| hot | b128_coalesced | 8 | 353.0 (353.0-353.0) | 92.82 |
| cold | blk32 | 1 | 885.7 (883.7-911.4) | 4.62 |
| cold | blk32 | 2 | 1145.3 (1137.1-1174.0) | 7.15 |
| cold | blk32 | 4 | 1637.4 (1634.5-1640.0) | 10.01 |
| cold | blk16 | 1 | 839.3 (832.9-862.3) | 2.44 |
| cold | blk16 | 2 | 928.8 (919.5-935.8) | 4.41 |
| cold | blk16 | 4 | 1180.3 (1178.2-1184.5) | 6.94 |
| cold | b128_lane128 | 1 | 827.7 (815.2-857.7) | 4.95 |
| cold | b128_lane128 | 2 | 875.6 (872.6-885.8) | 9.36 |
| cold | b128_lane128 | 4 | 1027.8 (1016.7-1034.4) | 15.94 |
| cold | b128_coalesced | 1 | 791.9 (787.6-811.6) | 5.17 |
| cold | b128_coalesced | 2 | 842.1 (828.5-858.5) | 9.73 |
| cold | b128_coalesced | 4 | 934.3 (916.2-949.6) | 17.54 |

- What it does (MEASURED, 4 masks, loads and stores): VGPR base + n receives the dword at address + 4n when bit n of
  M0 is set; with the bit clear the register keeps its value and the address still advances. Stores write only the
  masked dwords. One instruction moves up to 32 registers (128 bytes per lane, 4 KiB per wave).
- Addresses are used exactly as given: a lane address off by 1 or 2 bytes loads and stores the bytes at that exact
  address, no fault. When lanes overlap (16 bytes apart, 128-byte blocks), a store leaves the lowest lane's data in
  every overlapped dword (the helper's model "lane 31 written first" matches all 1024 dwords).
- Cost: much slower than plain loads for the same bytes. From L0, 8 block loads per batch reach 15 bytes per cycle per
  wave; eight `global_load_b128` covering the same per-lane 128 bytes reach 54, and coalesced `global_load_b128`
  (consecutive lanes on consecutive 16 bytes) 93. Each extra 32-register block costs about 256 cycles from L0 (8
  cycles per register, 16 bytes per cycle); why it is this slow was not established. From memory (N = 4): 10
  against 16 to 17 bytes per cycle.
- Caveat: in the hot rows with N = 8, plain and block alike, destination registers are reused while earlier loads
  are still in flight (the probe uses at most 4 register tuples; the last batch's data still checked correct), so
  the N = 8 figures (15 against 54 and 93) are less clean than the N = 4 ones: block 14.6, per-lane 128-bit loads
  46.0, coalesced 72.8 bytes per cycle. Either way block loads are several times slower.
- Restriction found: none beyond the guide's (EXEC, M0 mask, GV/GVS addressing). Use: saving and restoring a range of
  registers (spills, a trap handler), not data streaming.

## 7. Dynamic register allocation (`lds_dvgpr`, written by a helper, rerun by the lead)

```
RESULT lds_dvgpr before status=0x00010040 dyn_vgpr_en=0 state_priv=0x00000000 wave64=0
RESULT lds_dvgpr alloc vgprs=64 scc=1 status=0x00010040 dyn_vgpr_en=0 state_priv=0x00000200
RESULT lds_dvgpr alloc vgprs=256 scc=1 status=0x00010040 dyn_vgpr_en=0 state_priv=0x00000200
RESULT lds_dvgpr alloc vgprs=512 scc=1 status=0x00010040 dyn_vgpr_en=0 state_priv=0x00000200
RESULT lds_dvgpr kernel_completed=1
VERDICT dvgpr: INFO ordinary kernel: DYN_VGPR_EN=0, s_alloc_vgpr left STATUS unchanged and the kernel completed; consistent with the guide (allocation instructions ignored when not in dynamic mode)
```

- Hardware: an ordinary HIP kernel runs with STATUS.DYN_VGPR_EN (bit 30) = 0. `s_alloc_vgpr` with 64, 256 or 512
  registers changes nothing (STATUS unchanged, SCC as set before, the kernel completes with correct results), as the
  guide says for waves not launched in dynamic mode (READ).
- Compiler: clang 24 has no HIP attribute for it (`amdgpu_dynamic_vgpr_block_size` is "unknown attribute ...
  ignored"). The following experiment was done by hand during the study and its files and commands were not kept
  in the run folder: adding the LLVM function attribute `"amdgpu-dynamic-vgpr-block-size"="16"` to the IR makes the backend
  emit `s_alloc_vgpr 0` before `s_endpgm` and a short prologue that reads HW_ID2, but the kernel descriptor is
  byte-for-byte the same as without it (the dynamic-VGPR bit in `compute_pgm_rsrc3` is reserved on gfx120x, READ
  research-10), so the wave still starts in normal mode; that code object loads and runs correctly.
- Why it stays off (INFERRED; nothing was seen to refuse): neither the compiler's descriptor nor the ROCm runtime
  asks for dynamic mode on gfx1201, so the waves-per-SIMD rule lane 2 measured (min(16, 1536 / registers rounded up to 24)) is the only
  one that applies to HIP. Whether the hardware supports the mode for compute was not tested (it would need a
  hand-patched descriptor or driver change, outside this lane's limits).

## 8. Scratch (spill) loads and stores (`lds_smem scratch`)

```
lds_smem calib cycles_per_fma 0.999 min 0.999 max 0.999 samples 21
lds_smem scratch baseline_dependent_add_per_step cycles 6.67 min 6.67 max 6.67 mismatches 0
lds_smem scratch b32_store_reload_chain_per_round_trip cycles 279.88 min 266.64 max 288.88 mismatches 0
lds_smem scratch b32_store_wait_storecnt_reload_chain_per_round_trip cycles 341.64 min 338.64 max 347.88 mismatches 0
lds_smem scratch b128_store_reload_chain_per_round_trip cycles 310.88 min 302.88 max 319.64 mismatches 0
lds_smem scratch empty_batch_one_wait_per_batch cycles 26.47 min 26.47 max 27.45 mismatches 0
lds_smem scratch b32_16_independent_reloads_per_batch cycles 113.44 min 113.44 max 113.44 mismatches 0
lds_smem scratch b128_16_independent_reloads_per_batch cycles 161.44 min 161.44 max 161.44 mismatches 0
lds_smem scratch_summary chain_b32 279.9 chain_b32_net_of_add 273.2 chain_b32_with_storecnt_wait 341.6 chain_b128 310.9 | 16_reloads_b32 batch 113.4 net 87.0 per_load 5.44 | 16_reloads_b128 batch 161.4 net 135.0 per_load 8.44 bytes_per_cycle_per_wave 60.7
lds_smem spill_kernels array_alloca_result ok register_spill_result ok (instructions: see listing)
```

- A register spilled and reloaded at once (store, load of the same address, dependent use): 270 to 280 cycles per
  round trip for 32-bit, 300 to 320 for 128-bit; with `s_wait_storecnt 0` between store and load, 340. INFERRED: the
  store goes out to L2 (the vector L0 does not keep written data, lane 2) and the load waits for it; this is L2
  latency, not L0.
- Reloads of values spilled long before (independent, 16 per batch): 5.4 cycles per 32-bit reload and 8.4 per
  128-bit reload, 61 bytes per cycle per wave; the first reload pays about 87 cycles (an L0 hit, lane 2: 81).
- What clang emits for a spill: `scratch_store_b32` / `scratch_load_b32` with immediate offsets (a test kernel that
  needs more than 256 registers spilled 47 registers: 47 stores and 47 loads, `.amdhsa_private_segment_fixed_size`
  above 0), and a private array indexed at run time also became `scratch_load_b32`. Both kernels give the host's
  result.
- For lane 8: price a spill that is reloaded inside the same loop iteration at about 280 cycles of latency (hidden
  only by other waves), and a spill reloaded much later at about 5 cycles of issue plus an L0 hit.

## Checks and deliberate breaks

Known answers reproduced in the cited run: lane 1's FMA (1.000 cycle in `lds_probe`, `lds_tr`, `lds_stage`; 0.999 in
the helpers' probes); lane 2's plain 16-byte load batch (127 / 285 / 804 cycles from L0 / L2 / memory against 135 /
314 / 772, `tr_inflight_calib`); lane 4's matrix instruction (16.01 and 8.01 cycles); DRAM bandwidth 628 to 629 GB/s
(lane 2: 633). Every layout claim is checked with marker data against a CPU model; the transposing-load products
against a CPU reference. Every timed loop passed its assembly check (8 binaries, 0 expectations failed).

Deliberate breaks (`scripts/study-lds.sh breaks`), each must FAIL; all did:

| Probe and mode | What is broken | Result |
|---|---|---|
| `lds_probe calib` | calibration expects 2.00 cycles per FMA | FAIL as intended: VERDICT calib: FAIL independent v_fma_f32 1.000 cycles (expected 2.00; lane 1: 1.00), U=6 off midpoint by 0.00 cycles/iter |
| `lds_tr layout` | transposing-load model with lanes 3 and 5 swapped | FAIL as intended: VERDICT tr_layout: FAIL 8 trials of random per-lane addresses: tr_b128 128 of 2048 elements differ from the model, tr_b64 128 of 2048 (break: model with lanes 3 and 5 swapped) |
| `lds_probe pattern` | "colliding" stride-32 pattern run as stride 33 (no collision) | FAIL as intended: VERDICT lds_bank: FAIL ds_load_b32: 1 of 20 patterns differ from the 32-bank model (one cycle per pass); ds_load_b64: 0 of 15 differ from the same model applied to each half-wave; stride 32 dwords cos |
| `lds_tr wmma` | CPU reference off by one in one output | FAIL as intended: VERDICT tr_wmma_f16_ld16: FAIL FP16 A row-major x B row-major (ld 16) via global_load_tr_b128: 1 of 256 outputs differ from the CPU; control with a plain load of B: 254 differ (must be > 0) |
| `lds_stage feed` | register-only matrix loop expected at 32 cycles | FAIL as intended: VERDICT stage_calib: FAIL register-only loop: FP16 16.01 cycles (expected 32; lane 4: 16.1), FP8 8.01 (expected 8; lane 4: 8.1) |
| `lds_smem calib` | calibration expects 2.00 cycles | FAIL as intended: VERDICT calib: FAIL v_fma_f32 x16 independent: 0.999 cycles per FMA, accepted 1.90..2.10 (deliberate break: expects 2.00) |
| `lds_block layout` | block-load rule shifted by one dword | FAIL as intended: VERDICT layout: FAIL block load or store layout differs from the guide's rule |
| `lds_dvgpr probe` | expects DYN_VGPR_EN = 1 | FAIL as intended: VERDICT dvgpr: FAIL break: expected DYN_ |

## Method

Probes (all in `src/study/`, shared code in `lds_common.h`, which reuses lane 1's `core_common.h` and lane 2's
`mem_common.h`):

| Probe | Items | Modes |
|---|---|---|
| `lds_probe.hip` (built for WGP mode and, as `lds_probe_cu`, with `-mcumode`) | 1, 2 | calib, width, chain, inflight, pattern, waves, card, size |
| `lds_tr.hip` | 5 | calib, layout, wmma, inflight, bw |
| `lds_stage.hip` (WGP and CU mode builds) | 3 | calib, feed |
| `lds_smem.hip` (helper) | 4, 8 | calib, chase, width, mixed, scratch |
| `lds_block.hip` (helper) | 6 | calib, layout, cost |
| `lds_dvgpr.hip` (helper) | 7 | probe |

- Cycle probes use one wave alone on its SIMD and lane 1's differential method (blocks of 16 inline-assembly
  instructions, U = 4, 6 and 8 blocks per loop iteration; U = 6 must sit midway). Every probe first reproduces lane
  1's FMA (1.00 cycle). The global-load probes reproduce lane 2's L0, L2 and memory batch times for a plain 16-byte
  load (`tr_inflight_calib`), and `lds_stage` reproduces lane 4's 16 and 8 cycles per matrix instruction.
- LDS is filled so that every dword holds its own byte address; loads can then feed the next address, and any
  wrong address shows in the chain's end value.
- Multi-wave figures use the span from the first wave's start to the last wave's end on the 100 MHz REALTIME
  counter, converted to cycles with the waves' own cycle counts; per-wave best times are not used (waves sharing a
  SIMD do not run in parallel, so per-wave times overstate a shared rate).
- Every timed loop is checked in the `-save-temps` listing with lane 2's `mem_loopcheck.py` against
  `scripts/lds-expect/<probe>.txt`: which instructions, how many, which waits, no scratch (except where scratch is
  the subject). The helpers' probes have their own expectation files.
- Thermal: whole-card jobs (`card`, `bw`) last well under a second of load; every job started below 65 C. Highest
  hotspot in the cited run: 54 C.

## How to rerun

From the study repository root (after copying this lane in): `scripts/study-lds.sh all` (or `build`, `run`,
`breaks`). From the staging folder: `STUDY_REPO=<checkout of this repository> scripts/study-lds.sh all`.
One probe by hand: `HIP_VISIBLE_DEVICES=0 CARD_ARCH=gfx1201 flock <lock> build/lds/lds_probe/lds_probe pattern`.

## Not measured

Still open:

- The `ds_swizzle` rotate and FFT modes (results recorded, no rule); the rest of the old bullet on LDS misaligned
  loads, atomics, `ds_bpermute` and `ds_swizzle` is answered below.
  Answered in part (leftovers C, LD-1d, `measurements/leftovers-c-a.md`): the bit-mask and quad-permute modes match
  the guide's formula in all 32 lanes (7 offsets); every swizzle costs 1.00 cycle per access, 32 cycles from issue to
  data. Rotate mode (offset 0xc07f) gave every lane its own value; LLVM's `swizzle(ROTATE,0,5)` gave sources 4, 7, 6,
  9, 8, ... (not a plain rotation); FFT mode 0xe005 gave sources 0, 5, 2, 7, 5, 5, 7, 7, repeating every 8.
  MEASURED, no rule.
- The DS counter limit (63 LDS operations in flight, READ): conflict-free loads finish too fast for the counter to
  fill. A test with deliberately conflicting 128-bit loads would show it; not run.
  Answered in part (leftovers C, LD-2, `measurements/leftovers-c-a.md`): the test was run and did not show the
  counter limit. With slow loads (conflicting or 128-bit, 6 to 32 cycles each) the wave issues one load per cycle
  until about 17 128-bit (about 22 32-bit) loads are waiting, then each further load waits for one to leave; nothing
  changes at 63 or 64. This is shown for slow loads only (fast loads issued 96 in a row without a stop in item 1),
  so it is not a fixed queue size and not a byte count. MEASURED; a queue is INFERRED. Open: what stops the issue,
  and whether faster-draining conflicting loads stop at another count.
- The 32 MiB window in the transposing-load bandwidth table reads at 5.3 to 5.4 TB/s, above lane 2's memory-side
  cache figure (about 2.0 TB/s); the cause (neighbouring waves sharing lines in L2) was not studied, so that row is
  only a like-for-like comparison of the two loads.
  Answered in part (leftovers C, LD-9, `measurements/leftovers-c-b.md`): the 5.3 TB/s was not reproduced: the same
  walk read 9.5 TB/s at 32 MiB (and at 4 and 16 MiB) in two bursts and in the lead's rerun, while the DRAM
  calibration reproduced (634 GB/s); why the earlier figure was 5.3 is unexplained. The sharing explanation is "not
  supported in this run" (not ruled out, because the calibration failed): a walk in which no wave rereads its data
  read 7.2 to 7.9 TB/s at 32 MiB, far above the 2.4 TB/s it predicted, but that walk still shares data between waves
  over a launch and its bursts were shorter (clock ramp). At 64 MiB the same walk read 4.7 TB/s against 2.4. Rates of
  7 to 10 TB/s are far above the 2 TB/s memory-side-cache rate, so what serves them is not determined. The
  transposing and plain loads read at the same rate in every pattern (within about 8 %). Single bursts; MEASURED.
- A full staged GEMM (global to LDS to registers with barriers and register blocking): item 3 measures the feed rates
  that bound it and infers the staging cost from item 1; a kernel was not written. Not taken in leftovers C (plan,
  LD-5: the largest build, cut first; out of that pass by the brief, `measurements/leftovers-c-b.md`).
- Transposing loads from L0: a 16 KiB window cannot hold batches of 24 or more tiles (leftovers C, LD-6a,
  `measurements/leftovers-c-b.md`, left it out).
- Scalar loads: batches of more than 16 independent `s_load_b96`/`b128`, 8 `b256` or 4 `b512` were not run: they
  exceed the SGPR budget (a b96 batch of 31 spilled SGPRs and returned wrong words in development, so it was dropped).
  Low value (leftovers C plan, LD-7e): a batch with reused destinations gives no usable data; only the counter limit
  would show.
- Scalar loads, what is left of the sharing and cost rows: why the other CU of the WGP finds a region at 46 cycles
  against 37 for the own region, and the cold `s_buffer_load` (leftovers C, LD-7b and LD-7c,
  `measurements/leftovers-c-b.md`; the 64 KiB-region rows moved by up to 34 cycles between runs and do not separate
  the partners).
- Dynamic VGPRs on the hardware: not reachable from HIP (item 7); enabling the mode would need a descriptor or driver
  change, which this lane may not make. Not reachable (leftovers C plan, LD-8): a wrong start could hang the card.
- RDNA 3: its write-up (`rdna3-fp8/README.md`) did not measure LDS latency, bandwidth, banks or scalar loads; it only
  notes that an LDS lookup table was slower than bit arithmetic for FP8 widening there (lane 6 here found 4.6 cycles
  per weight for the same idea on RDNA 4) and that LLVM split FP16 copies out of LDS into halves. No RDNA 3 numbers
  to compare. Not reachable (leftovers C plan, LD-10): it needs the RX 7900 XT.
- New from leftovers C: why `ds_bpermute_b32` costs 5.06 cycles per access in execution-core's compiler-allocated
  block and 3.00 with fixed registers (INFERRED: that block's register choice; not re-measured; LD-1d,
  `measurements/leftovers-c-a.md`).

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, LD-1a, `measurements/leftovers-c-a.md`): LDS loads from misaligned addresses: ROCm runs the
  LDS in UNALIGNED mode, so every misaligned load returns exactly the bytes at its address in all 32 lanes, but any
  access not aligned to its own size (96-bit: 16 bytes) costs 32 cycles (one lane per cycle). A 64-bit load at a 4-byte
  offset is 16 times slower than aligned; 96-bit data packed at 12 bytes is 8 times slower than padded to 16. MEASURED.
- Answered (leftovers C, LD-1b, `measurements/leftovers-c-a.md`): LDS atomics: every result is right (sums, returned
  values, max, compare-and-store). On one word the 32 lanes are served in lane order, 2 cycles each (64 for 32 lanes);
  otherwise an atomic costs what a store costs and follows the load's bank model; compare-and-store costs about twice
  (INFERRED: two words per lane). MEASURED.
- Answered in lds-and-special-loads.md and execution-core.md, item 2, and for address patterns by leftovers C (LD-1c,
  LD-1d, `measurements/leftovers-c-a.md`): `ds_bpermute_b32` has 36 cycles from issue to data; 3.00 cycles per access
  for every index pattern tried (identity, all from lane 0, strides 2 to 32, reverse, random): it does not depend on
  the pattern. MEASURED.
- Answered (leftovers C, LD-3a and LD-3b, `measurements/leftovers-c-a.md`): LDS stores and 96-bit loads under
  conflicts follow one rule: the number of passes of the 32-bank model, applied to the whole wave for 32-bit accesses,
  to each half-wave for 64-bit and to each quarter-wave for 96- and 128-bit, with a floor per form (loads 1, 2, 4, 6
  cycles; stores 2, 3, 5, 5). Loads and stores follow the same model; 96-bit accesses behave like 128-bit ones under
  conflicts. Lanes storing to one address leave the highest lane's value. MEASURED.
- Answered (leftovers C, LD-4, `measurements/leftovers-c-a.md`): cross-wave store-to-load timing, measured by polling
  (not after a barrier): through LDS a round trip costs 272 cycles (about 136 per direction against 110 for a wave
  reading its own store), on one SIMD, on the two SIMDs of a CU, and 277 across the two CUs of a WGP. Through global
  memory at device scope between two CUs a round trip costs 912 cycles (871 to 952 over three runs; 326 per step for the
  own-store loop). MEASURED.
- Answered (leftovers C, LD-6a and LD-6b, `measurements/leftovers-c-b.md`): transposing loads past 24 in flight: for
  the b64 forms the 64th load costs a whole L2 or memory trip (from memory +527 cycles for the transposing form,
  against 1.1 to 6.8 cycles per load from 48 to 63): the same 63-load counter as plain loads. The b128 form was run to
  56 (4 VGPRs each) and keeps its higher issue cost (1.25 to 1.43 times the plain batch from L2). With EXEC not all
  ones, both transposing loads write every lane's destination with the full-mask result, under every mask including 0:
  they ignore EXEC like the matrix instructions. MEASURED.
- Answered (leftovers C, LD-7a, `measurements/leftovers-c-b.md`): the scalar cache line is 64 bytes, not the 128
  inferred: after a miss at X the next 64 bytes are a miss (184 cycles against 37 for a hit). MEASURED; item 4's capacity
  steps must come from the cache's organisation (INFERRED). Behind it, the 256-byte block around X is present (the
  L2 line size; that it is L2 is INFERRED).
- Answered (leftovers C, LD-7b, `measurements/leftovers-c-b.md`): the level between the scalar cache and memory: a
  region one CU touched is found at 46 cycles by the other CU of the same WGP and at about 216 cycles by any other CU,
  whatever its shader array or engine. So the scalar cache is shared by the two CUs of a WGP and the level behind it is
  card-wide (INFERRED: L2). MEASURED.
- Answered (leftovers C, LD-7c, `measurements/leftovers-c-b.md`): `s_buffer_load` returns the right data; a lone load
  takes 53 cycles against 46 for `s_load`, a batch of 8 costs less (54 against 61, all widths up to b128). MEASURED.
- Answered in memory-system.md, item 9: `s_prefetch_data` works under ROCm; a scalar load falls from 723 to 194 cycles
  at 4 steps ahead and to 70 at 16 (INFERRED: scalar loads then hit the scalar cache), and the instruction does nothing
  with the MODE bit cleared. The scalar cache's line size is the 64 bytes of LD-7a (leftovers C,
  `measurements/leftovers-c-b.md`).
- Answered in compiler-and-hazards.md, item 5, and in this file, item 1: the compiler never splits a 16-byte read
  (forced 4 x `b32` is 2.9 times slower for one wave, warm, and no different for the whole card); on LDS four 32-bit
  loads move the same bytes as one 128-bit load in 4 cycles against 6.
