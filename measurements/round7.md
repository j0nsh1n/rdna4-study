# Round 7: the memory clock under load, and 32- against 64-wide waves

Three items in three run folders (git-ignored), all on 2026-11-24, card RX 9070 XT (gfx1201, `lspci` 1002:7550), kernel 7.3.4-202:

- **A3, the memory clock under load:** `runs/20261124T053703Z-r7mem/` (read, write, copy, duty-cycle and compute-only loads
  with a 5 ms clock sampler).
- **A, wave64 (A1, A2, A2b):** `runs/20261124T065618Z-r7w64/` (instruction costs in wave32 and wave64, one SIMD and the
  whole card, and vector work beside a matrix wave).
- **B, FSR 4 from the outside:** `runs/20261124T090844Z-r7fsr-b1/` (AMD's SDK sample, 45 timing cells and 6 clock cells; section B at the end).

All three runs ran on the LACT profile "Default" (overclock: +150 MHz core, -50 mV, 374 W cap; the power the driver reports is
about 1.2x too low because of the regulator mod). Every wall-clock number below (GB/s, T MAC/s, GHz, W) describes this card
in that state, not a stock card. Cycle counts do not depend on the clock. Labels as in `round5.md`: MEASURED (a number in a
log of the run folder), READ (a source file or a listing), INFERRED (worked out from measured numbers, not observed). The
numbers were checked against the runs' own `RESULTS.md` and an independent read-only review (recomputation from the raw logs);
where the review found a number or a sentence too strong, the corrected form is used here and the change is named.

## Plain summary

The card's memory clock is no longer stuck at 1258 MHz. Any streaming load, from a 0.05 s burst to 20 s, takes it to 1408 MHz
(the overclock request) within one to three of the driver's 10 ms table refreshes, and a read pass then runs at 700 GB/s,
about 10 to 11 % more than the 633 GB/s measured at 1258 MHz, so older numbers taken while the memory clock was held at 1258 MHz
are about 10 % low for memory-bound work (each table states its own memory clock). After a load the clock steps back down to 772 MHz within about 0.15 s, and it often stays at 772
instead of going on to 456; why is untested. On the compute side, a 64-wide wave does twice the dot-product work of a 32-wide
wave per instruction (the dot instructions cost one cycle in both sizes), so the 8-bit dot path reaches 99 T MAC/s on the whole
card in wave64 against 50 in wave32, while the FP8 conversions cost twice as many cycles in wave64 and gain nothing. The matrix
unit's lead over the best 8-bit dot path therefore drops from 4x to 2x (200.5 T MAC/s against 50.1 and 99.1), and a matrix wave
still keeps the SIMD to itself in wave64: the vector partner makes no progress while it runs, and `s_setprio` decides who goes
first, as in wave32. One control failed: the whole-card matrix rate came out at 400-403 TOPS, not the 420 TOPS the brief asked
for, and the cycle counter's clock (3072 MHz) is 4.8 % above the clock the driver reports (2917-2932 MHz); a likely cause
(the reported clock leaves out the +150 MHz offset) is INFERRED and untested.
FSR 4, seen from the outside with AMD's SDK sample, costs about 0.34 ms at 1080p output, 0.63 ms at 1440p and 1.64 ms at 4K on
this card (MEASURED, undisturbed cells), 0.97x and 1.25x AMD's published 1080p and 4K figures (READ), a rough match only, the 1080p agreement possibly coincidental (FSR
4.1.1 against 4.1.2, RCAS on, frame generation on, overclocked card); the inferred preset changes it by 10 to 20 % at most, though the
preset key presses were unreliable and the preset was inferred, and the core clock sat at 2.3 GHz at 1080p and 1440p against
3.2 GHz at 4K, so the small-size numbers are not at peak clocks.

## A3. The memory clock under load, now that it is unlocked

Run folder `runs/20261124T053703Z-r7mem/` (`RESULTS.md`, `analysis.md`, `expect.md` written before any GPU job, raw data in
`loads/*/memload.out`, `loads/*/watch/samples.csv`, `gaps/*.txt`, `idle-record/samples.csv`).

**Question.** Earlier rounds held the memory clock at 1258 MHz whatever the load (`scheduling-and-clocks.md`, `memory-system.md`),
and 1408 MHz appeared only after the displays blanked. One monitor now runs on the iGPU. What does the
memory clock do now when a load starts, runs, changes shape and ends, and what does bandwidth do with it? The run itself made no
display change (the owner's setup is not logged by the run).

**Method.** `memload.hip` runs one timed load per gated job (`scripts/study-llm.sh r4dev`): read, write, copy (read and write
bytes counted), duty cycles (busy and idle spans) or an FP16 matrix loop, 1 GiB buffer, cold start (the process allocates and
launches at once). A separate read-only sampler (`uclk_sampler.cpp`, sysfs, 5 ms period, reads `gpu_metrics`) logs the memory
clock from 1 s before the process to 3 s after. 27 traced loads (18 read loads, 3 of them warm; 2 write; 2 copy; 4 duty; 1
FP16), 3 loads without the sampler, a 60 s sampler-only idle record, and a 100 ms poll after each process ended. The
`gpu_metrics` table refreshes every 10 ms (READ in `expect.md`), so every clock time below is good to about +-10 ms.

### What the clock does (MEASURED, `analysis.md` sections 1, 2, 9)

- The top reading was 1408 MHz in every streaming load (read, write, copy, both duty patterns, 0.05 s to 20 s). Once there it
  stayed for the whole load (no fall back in 20 s). 1258 MHz never appeared in any of the 27 traces or the 30 idle polls.
  `pp_dpm_mclk` still lists no 1408 level (READ: 96, 456, 772, 875, 1124, 1258); the reading comes from `gpu_metrics`.
- The clock ramp, in the logged clock: first step up 2-16 ms after the first launch (median 7), 1408 reached 7-29 ms after
  it (median 16). That is one to three table refreshes, so the times are only good to +-10 ms, and the logged clock lags the
  real one (passes inside a single 875 or 1124 MHz reading read about 700 GB/s, INFERRED).
- The ramp in the bandwidth of the first passes (no sampler involved, 18 read loads): median pass 1 273 GB/s (188-380), pass 2
  381 (320-463), pass 3 486 (390-699), pass 4 604 (482-702), pass 5 699, pass 6 700. Full speed (three passes in a row within
  5 % of the load's median) after a median 11.3 ms (5.3-74.6 ms) over 22 loads. This is the cleaner view of the ramp.
- Where the ramp starts matters. Most loads did not start from idle: starting the process lifted the clock from 456 to 772
  before the first launch in several loads (for example `read-0.05s-r1`: 214 of 214 samples before the process read 456, the
  reading at the first launch was 772; INFERRED, the cause is not separated). Only the 8 loads that started at 456 show a ramp
  from idle:

| reading before the first launch | loads | first rise ms (median) | 1408 reached ms (median) |
|---|---|---|---|
| 456 | 8 | 4-9 (6) | 14-19 (16) |
| 772 | 15 | 2-16 (7) | 7-29 (16) |
| 875 | 3 | 5-10 (7) | 10-17 (15) |

### Read loads of five lengths, three repeats each (cold start)

All-pass GB/s is the program's own figure over every pass including pass 0, the first pass after launch, which carries most of
the ramp (`GBps_all` in `loads/*/memload.out`). **Correction:** `RESULTS.md` summary item 2 and `analysis.md` section 4 give
the figure without pass 0 (612.1 / 605.4 / 664.3 for the 0.05 s loads); the lead's review note in `RESULTS.md` replaces them
with the all-pass figures used here. The 5 s and 20 s loads differ by 0.1-0.5 GB/s between the two ways of counting.

| load length | reading before launch (r1 / r2 / r3) | first rise ms | 1408 reached ms | all-pass GB/s | passes |
|---|---|---|---|---|---|
| 0.05 s | 772 / 772 / 456 | 7 / 8 / 4 | 17 / 8 / 14 | 585.7 / 586.5 / 630.0 | 29 / 30 / 32 |
| 0.2 s | 456 / 875 / 456 | 7 / 5 / 4 | 17 / 15 / 15 | 677.4 / 657.6 / 678.7 | 129 / 125 / 129 |
| 1 s | 456 / 772 / 772 | 8 / 8 / 9 | 18 / 18 / 9 | 695.1 / 697.8 / 626.6 | 650 / 652 / 586 |
| 5 s | 456 / 456 / 772 | 8 / 9 / 7 | 18 / 19 / 17 | 658.9 / 688.3 / 685.6 | 3071 / 3208 / 3195 |
| 20 s | 456 / 772 / 772 | 4 / 16 / 6 | 14 / 16 / 16 | 645.4 / 673.2 / 686.8 | 12024 / 12541 / 12795 |

A 0.05 s burst spent 0.037, 0.047 and 0.040 s of its window at 1408 (74-94 %) and read 586-630 GB/s as a whole. Whole-load
rates sit below the per-pass median (700) because some passes are slow: 1.1-37.9 % of the passes of a load run below 95 % of
its median (266-460 GB/s). The same happens with no sampler running (2.4, 5.0 and 19.9 % in the three sampler-free loads), so
the sampler is not the cause; INFERRED: other work on the card (the desktop, a Steam client), not tested.

### Bandwidth against the clock (MEASURED, `analysis.md` section 4)

Passes that lie inside one 1408 MHz reading with 15 ms of the same reading on both sides:

| pattern | passes | median GB/s | at 1258 MHz earlier | ratio to earlier x 1408/1258 |
|---|---|---|---|---|
| read | 49,699 (per-load medians 698.0-701.6) | 700.3 | 633 (`memory-system.md`) | 0.988 |
| write | 5,646 | 638.7 | 581 (`memory-system.md`) | 0.982 |
| copy (read and write bytes) | 2,779 | 608.9 | 547 (`memory-system.md`) | 0.995 |

At 1408 MHz the pass rates are about 10 to 11 % higher than at 1258 (700.3 / 633 = +10.6 %, 638.7 / 581 = +9.9 %,
608.9 / 547 = +11.3 %; computed here), close to the clock ratio of 1.119. Only 1408 lasted long enough for a pass to sit
inside one reading; the lower readings do not predict bandwidth (see the ramp above). `analysis.md` section 4 also prints
0.902 (write) and 0.859 (copy) "ratio to 633/1258 x clock"; those use the read baseline for all three and are not used here.

### After the load, and at idle (MEASURED, sections 3 and 8)

- After the last pass the clock steps down 1124, 875, 772 and reaches 772 within about 0.15 s; the first reading below the top
  came 25-76 ms after the last pass.
- It then goes on to 456 only sometimes. In 11 of 30 idle polls (100 ms, 30-120 s) it never read 456; in the others the first
  456 came 0.0 to 111 s after the process ended.
- Idle level, sampler running: 456 in 3413 of 5241 samples (65 %) before the processes, 772 in the other 1828; in the 60 s
  sampler-only record 456 for 10.6 s and then 772 for 49.4 s (456 in 17.7 % of 12,012 samples).
- Cause of the 772 idle level: **untested**. `watch_card clients` listed Xwayland, steam and steamwebhelper on the card in the
  setup tests (`smoke-keep/`, `old-smoke1/`, `attempt1-watchcard/`; not in the 27 main loads), which fits a desktop client
  holding the clock (INFERRED). The 5 ms `gpu_metrics` reader itself is an equally untested alternative; nothing in the run
  separates the two.

### Duty cycles and a compute-only load

- Duty cycles (5 s each): busy and idle spans of 2.07 ms / 2 ms (1250 cycles) and 5.20 ms / 11 ms (313 cycles, 31 % busy; about
  the shape of a 60 Hz frame, `expect.md`) with `--grid 256`, plus the first pair with a grid of 1024 that was not all
  resident (busy 8.07 and 20.07 ms, kept as extra data). All four read 1408 MHz for the whole 5 s window, busy spans and
  gaps alike (4.985 s of the 5.0 s at 1408 in the 2/2 ms load). The 10 ms table refresh means a 2 ms cycle is only seen as an
  average. Bandwidth while busy: 651.5 and 661.5 GB/s (grid 256), 702.8 and 704.4 (grid 1024).
- FP16 matrix loop, 5 s (215.2 TFLOPS, gfx activity 100 %, core 3392 MHz): the memory clock stayed at 772 MHz (875 for 0.38 s).
  Hotspot rose from 44 to 92 C in 5 s (the guard stops at 95 C; no other load went above 62 C). **Softened by the review:**
  this is one 5 s load that started at 772, not at 456, so it shows that the memory clock did not rise above 875 here; it does
  not show that the driver follows memory demand and not core activity. The 875 bumps repeat every second and look like
  desktop activity (INFERRED).

### Controls and their outcomes

| Control | Outcome |
|---|---|
| 1: at 1258 MHz a pass must read 632-633 GB/s | **Could not be run**: 1258 MHz never appeared. |
| 1b: at 1408 MHz a pass must read 687-703 GB/s | Pass: median 700.3 over 49,699 passes, every load's median 698.0-701.6. |
| 2: deliberate break (shift every reading to the next level, so 1408 reads as "1258") | Caught, narrowly: ratio 1.107 against an 8 % tolerance; one trace (`read-20s-r3`), one shift. The real column passed (0.989). |
| 3: the sampler reads 456 at idle | **Failed as worded**: 456 in 599 of 599 samples in the first 3 s idle test, but only 3413 of 5241 (65 %) before the loads. The sampler reads the same field as the gate; the card's idle level was not always 456. |
| Does the sampler slow the stream? (3 loads without it) | No visible effect: medians 701.2, 701.2, 699.3 against 700.1-701.0 with it; slow-pass shares in the same range. |
| Did the gate refuse a job over a changed memory clock? | No: 30 of 30 jobs exit 0 (review); the gate only labels "MEMORY CLOCK uclk=a/b differs". |

Not done: the fresh-login and resolution-change tests (skipped by the plan). The study's `watch_card record` refused to start
on the idle card (it exits 3 when `pp_dpm_sclk` shows the sleep state), so the sampler `uclk_sampler.cpp` was written instead
(`attempt1-watchcard/` keeps the first try). No display setting, LACT setting or sysfs value was changed.

### What changed against the earlier documents

- The memory clock is not pinned at 1258 MHz any more: it follows the load, goes to 1408 MHz and falls back to 772 or 456.
  Earlier rounds saw 1258 "at rest and under every load" (`scheduling-and-clocks.md`) and 1408 only after a display change
  (`README.md`); a 1408 reading now needs no blanking by this run.
- Bandwidth at 1408 MHz (read 700, write 639, copy 609 GB/s) replaces the 1258 MHz figures for this card in its current
  state. Taken at 1258 MHz (so about 10 % low for memory-bound work): the hardware lanes' bandwidth and latency files
  (`memory-system.md`, `scheduling-and-clocks.md`), round 5's 590 jobs (`round5.md`, 1258 before and after every job) and LLM
  round 4 after its GPU reset. Taken at 1408 MHz per `README.md`: LLM rounds 1 to 3, the exact 8B step and round 4 before the
  reset. Each job's own `current_uclk` decides.
- Short bursts reach full speed after about 16 ms; a 0.05 s burst still averages 586-630 GB/s. The "short job at a low memory clock"
  caveat of earlier rounds applies mainly to the first 10-30 ms.

### Open

- Why the idle level is 772 and not 456 about 35 % of the time, and why the clock sometimes does not fall to 456 for two
  minutes after a load: untested (a desktop client and the sampler's own reads are both candidates).
- Whether the monitor on the iGPU is what freed the clock: INFERRED from the setup change, never switched back.
- Whether a real game (frame loads rather than streaming) holds 1408: only streaming loads and two duty patterns ran.
- Whether the driver follows core activity at all: one compute-only load that started at 772.

## A. Wave64: A1, A2 and A2b

Run folder `runs/20261124T065618Z-r7w64/` (`RESULTS.md`, `analysis.md`, `expect.md` written before the first GPU job, with
dated addenda before the first A2 and A2b jobs; logs `a1-*.log`, `a2-simd-*.log`, `card-*.log`, `a2b-*.log`, jobs and their
memory clock in `jobs.log` and `.meta`). Toolchain ROCm 10.1 (`hipcc` 7.16, clang 24), `-O3 --offload-arch=gfx1201 -mcumode`,
wave64 builds with `-mwavefrontsize64`. Every GPU job ran through the gate.

**Questions.** A1: what do the dot-product and conversion instructions cost in a wave of 64 lanes, against 32? A2: do the
costs hold with 1 to 8 waves on one SIMD and on the whole card, and what does the 8-bit matrix unit's lead over the dot path
become? A2b: does vector work still stall beside a matrix wave in wave64?

**Method.**

| Part | Method |
|---|---|
| A1 | The SC-10 method of `left_core.hip`: one wave, 16-instruction blocks in one inline-asm statement with the counter reads and the loop, R = 2, 4, 6 launched interleaved, cost = (R=6 minus R=2) / (4 x 16 x 16), 45 samples per block; 52 blocks in wave32, 48 in wave64. |
| A2, one SIMD | SIMD 0 of one CU, W = 1, 2, 4, 8 waves, cost = (span at 24 copies minus span at 12) / (3000 x 12 x 16 x W) in the SIMD's cycle counter, 15 repetitions. |
| A2, whole card | 64 groups of 1024 threads (one per CU), W waves per SIMD, bursts of about 100 ms (50 ms for random data), 3 per cell, each started below 50 C; the placement (128 SIMDs x W waves) was checked from every wave's `HW_ID1` on every burst. |
| A2b | `left_prio.hip`'s roles (rank 0 older) in a new harness `r7_prio.hip` (partners FMA, `v_dot4_i32_iu8`, `v_cvt_pk_fp8_f32`; wave32 and wave64; 30 pairs per row), plus the unchanged `left_prio.hip` for wave32. |

### A1: one wave alone (MEASURED, cycles per instruction)

| Instruction | wave32 independent | wave32 chained | wave64 independent | wave64 chained | wave64 sources in one bank |
|---|---|---|---|---|---|
| the 8 dot forms: `v_dot4_i32_iu8`, `v_dot8_i32_iu4`, the four FP8/BF8 `v_dot4_f32_*`, `v_dot2_f32_f16`, `v_dot2_f32_bf16` | 1.000 | 6.000 | 1.000 | 6.000 | 1.188 (wave32: 1.000) |
| `v_cvt_pk_fp8_f32`, `v_cvt_pk_bf8_f32`, `v_cvt_f32_fp8`, `v_cvt_f32_bf8` | 1.000 | 5.000 | 2.000 | 5.000 | 2.000 |
| `v_s_exp_f32`, `v_s_rcp_f32` (SGPR result) | 1.875 | 19.000 | 2.125 * | 19.000 | n/a |
| `v_exp_f32` | 4.000 | 10.000 | 8.000 | 14.000 | n/a |

`*` the midpoint rule failed for these two blocks: R = 2, 4, 6 gave 1,619 / 2,771 / 3,795 cycles per region, 2.25 per
instruction from R = 2 to 4 and 2.00 from R = 4 to 6; 2.00 (eight SGPR writes per 16 cycles) is the INFERRED steady state. A
wave-size dependence of an op that computes one scalar is odd and not understood. The wave32 dual form `v_dual_dot2acc_f32_f16`
costs 1.000 per dual instruction, so 0.500 per dot2, against 1.000 per ordinary dot2 (VOPD has no wave64 form).

Both repetitions agree (largest difference 0.0000 over the six numbers of each row). Reading: a wave64 dot instruction costs
the same one cycle as in wave32 and does twice the lane work, so a SIMD does 256 dot4 multiply-adds per cycle in wave64
against 128 in wave32 (`v_dot8_i32_iu4`: 512 against 256); the FP8 conversions cost twice as many cycles, so the same lanes
per cycle as wave32. **The expectation of 2.00 for the dot products in `expect.md` was wrong.**

### A2: one SIMD and the whole card (MEASURED)

One SIMD, W waves: the cost stays within 0.5 % of the W = 1 value (dots 1.000-1.005, matrix 8.002-8.021). Selected cells,
cycles per instruction (multiply-adds per cycle per SIMD in brackets):

| Instruction | w32 W=1 | w32 W=8 | w64 W=1 | w64 W=8 |
|---|---|---|---|---|
| `v_dot4_i32_iu8` | 1.000 (128.0) | 1.002 (127.8) | 1.000 (256.0) | 1.004 (255.1) |
| `v_dot4_f32_fp8_fp8` | 1.000 (128.0) | 1.001 (127.9) | 1.000 (256.0) | 1.002 (255.5) |
| `v_dot2_f32_f16` | 1.000 (64.0) | 1.001 (63.9) | 1.000 (128.0) | 1.002 (127.8) |
| `v_dot8_i32_iu4` | 1.000 (256.0) | 1.001 (255.8) | 1.000 (512.0) | 1.002 (510.8) |
| `v_cvt_pk_fp8_f32` | 1.000 | 1.001 | 2.000 | 2.001 |
| `v_wmma_i32_16x16x16_iu8` (chained) | 8.008 (511.5) | 8.021 (510.7) | 8.006 (511.6) | 8.020 (510.7) |

Whole card, 64 CUs, W = 8, medians of 3 bursts (T MAC/s per burst in brackets). The cycle counts include the loop control, so
the cost is 1-2 % above the instruction rate of A1 (1.006 and 1.017 at W = 8).

| Size | Instruction | cycles per SIMD instr | MAC/cycle/SIMD | cycle-counter clock MHz | T MAC/s |
|---|---|---|---|---|---|
| w32 | `v_dot4_i32_iu8` | 1.006 | 127.2 | 3075 | 50.1 (50.1-51.5) |
| w32 | `v_dot4_f32_fp8_fp8` | 1.006 | 127.2 | 3077 | 50.1 (50.0-50.1) |
| w32 | `v_dual_dot2acc_f32_f16` | 1.006 | 127.2 | 3075 | 50.1 (49.9-50.1) |
| w32 | `v_wmma_i32_16x16x16_iu8` | 8.034 | 509.9 | 3072 | 200.5 (200.2-201.6) |
| w64 | `v_dot4_i32_iu8` | 1.017 | 251.8 | 3075 | 99.1 (99.1-101.2) |
| w64 | `v_dot4_f32_fp8_fp8` | 1.017 | 251.8 | 3066 | 98.8 (98.6-99.0) |
| w64 | `v_dot2_f32_f16` | 1.017 | 125.9 | 3053 | 49.2 (49.2-49.3) |
| w64 | `v_wmma_i32_16x16x16_iu8` | 8.034 | 509.9 | 3074 | 200.6 (200.4-200.8) |

- Matrix lead over the best 8-bit dot path, whole card at W = 8: 4.0x in wave32 (200.5 / 50.1) and 2.02x in wave64
  (200.6 / 99.1); at W = 4 4.04x and 2.04x. `v_dot8_i32_iu4` in wave64 reaches 511-512 multiply-adds per cycle on one SIMD,
  level with the matrix unit, but only that single-SIMD row was run, not the whole card.
- Coverage: A1 (one wave) covers all the forms in the first table. One SIMD with 1 to 8 waves and the whole card cover only
  `v_dot4_i32_iu8`, `v_dot4_f32_fp8_fp8` (the representative of the four FP8 forms), `v_dot2_f32_f16`, the wave32 dual form
  and the matrix instruction; `v_dot8_i32_iu4` and the conversion run on one SIMD only. `v_dot2_f32_bf16`, the other FP8/BF8
  forms, `v_cvt_pk_fp8_f32` and the pseudo-scalar forms were not run on the whole card.
- At W = 1 the dots reach 0.9505 (w32) and 0.9481 (w64) of 128 x clock x the A1 rate (the loop control and the burst
  edges); at W = 4 and 8 0.979-0.994 for the dots, and for the matrix instruction 1.000 at W = 1 and 0.996-0.998 at W = 4 and 8.
- Random register data (50 ms bursts, W = 8; highest instantaneous `gpu_metrics` power in the burst, above the 374 W cap, so
  not sustained power): `v_dot4_i32_iu8` w64 2735 MHz, 88.1 T MAC/s, 516 W reported, against 3075 MHz, 99.1 T MAC/s, 347 W with
  constant data in 100 ms bursts; matrix w64 2861 MHz, 186.7 T MAC/s, 358 W. INFERRED: a wave64 dot path costs the clock more
  than the same work on the matrix unit; the bursts differ in length, so this is a hint, not a measurement of equal loads.
  Conversion-heavy code (dequantise, then multiply) gets no per-cycle gain from wave64 on the conversion side, only on the
  multiply side (INFERRED).

### A2b: vector work beside a matrix wave (MEASURED)

Two waves on one SIMD, rank 0 older. Matrix wave: 64 x 16 chained FP16 matrix instructions. Cycles per instruction, matrix
wave / partner, no `s_setprio` unless stated:

| Partner | w32 matrix older | w32 partner older | w64 matrix older | w64 partner older |
|---|---|---|---|---|
| FMA, banks spread (alone 1.078 / 1.082) | 16.36 / 2.09 | 31.91 / 1.50 | 16.46 / 2.10 | 32.02 / 1.34 |
| `v_dot4_i32_iu8` (alone as FMA) | 16.36 / 2.09 | 31.91 / 1.50 | 16.47 / 2.10 | 32.02 / 1.34 |
| `v_cvt_pk_fp8_f32` (alone 1.078 / 2.078) | 16.24 / 2.08 | 31.85 / 1.49 | 16.28 / 4.09 | 31.62 / 2.33 |
| FMA, harness text (half the sources in one bank) | 16.43 / 2.10 | 31.91 / 1.50 | 16.47 / 2.34 | 35.90 / 1.59 |

The matrix wave alone costs 16.058 (w32) and 16.024 (w64) cycles per instruction. With `s_setprio 3` on the matrix wave it
keeps 16.2-16.5 whichever wave is older; with `s_setprio 3` on the partner the matrix wave is held to 31.6-35.9 whichever is
older (all rows `a2b-*.log`). In the matrix-older rows no partner made progress during the matrix wave's run: the partner's
cost minus its alone cost, times its instructions, equals the matrix wave's whole run to within 1.1 % (0.989-0.995 in the
five rows). Not repeated for wave64: the 3-wave and round-robin-bit rows of `left_prio`.

### Controls and their outcomes

| Control | Outcome |
|---|---|
| A1 calibration (bank-spread FMA 1.000, dependent FMA 5.000, `v_add_nc_u32` 1.000 / 2.000, chained FP16 matrix 16.0, chained 8-bit matrix 8.0) and SC-10 rows (`v_pk_fma_f16`, `v_cvt_f32_i32`, `v_exp_f32`) | Pass in both sizes, both repetitions. |
| A1 deliberate break (`v_add_nc_u32` in wave64 judged at 1.00; wave32 FMA judged at 2.00) | `a1_calib` fails, as it must. |
| Listing checks of the code object of each binary that ran (opcodes, registers as written, bank layout, `s_setprio`; wave64 decoded with `--mattr=+wavefrontsize64`); `.wavefront_size: 64` in the three wave64 objects (review) | 936 / 864 timed regions in wave32 / wave64, 16 and 14 in the card program, 24 each in `r7_prio`, 21 kernels in `left_prio`: 0 mismatches; each check fails against a deliberately wrong header. |
| A2 one SIMD calibration (FMA, chained matrix) | Pass in both sizes; break fails. |
| A2 whole-card placement | 128 SIMDs with W waves each in all 93 bursts without a break (95 bursts with the two break bursts). |
| **A2 whole-card 8-bit matrix, W = 8, absolute: 420 TOPS +-1.5 %** | **Failed.** Smoke runs 374.9 and 391.5 TOPS, main runs 400.3-403.1 (w32) and 400.7-401.7 (w64) at 3.07-3.09 GHz; with random data 367-373 TOPS at 2.81-2.86 GHz. |
| A2 whole-card matrix control per GHz of the cycle-counter clock: 131.07 +-1.5 % | Pass: 130.50-130.53 (-0.4 %). **This form was adopted after the absolute control failed** and divides out the clock, so it only re-checks 8.0 cycles per instruction (nearly circular); it is not a pass of the absolute control. The break (judged at 262.14 per GHz) fails. |
| `left_prio` (unchanged, wave32) | Pass: 16.356 / 2.096 and 31.690 / 1.488, matching `matrix-timing.md` lines 452-463 to three digits. |
| `r7prio` harness, both sizes | Pass; both breaks fail. |

### The clock gap (INFERRED, untested)

The cycle counter's clock in the whole-card bursts was 3072-3075 MHz (matrix) while the driver's `pp_dpm_sclk`, sampled every
40 ms, read 2917-2932 MHz, 140-155 MHz lower (4.8 % in `card-w64-W8-const.log`). `matrix-timing.md` found 0.3 % agreement on
the "Testing" profile. The gap matches the +150 MHz core offset of "Default". INFERRED: the driver's reported clock
(`pp_dpm_sclk`, `gpu_metrics` `current_gfxclk`) leaves the overclock offset out, while the shader clock includes it. The
wall-clock rate itself says the logged value is the low one (200.5 T MAC/s / (128 x 512 x 2.92 GHz) = 1.048, impossible if 512 is
the peak). Not tested: a burst on "Testing" in the same harness would confirm it (needs the owner's OK for a profile
switch). Until then, "the card ran at 3.07 GHz" means the cycle-counter clock, and this also explains the 400-403 against
420 TOPS (3.2 GHz) only as far as the clock is concerned, not as a tested cause.

### What went wrong on the way (all logs kept)

- First A1 run: a bug in the `u_split*` blocks and no per-R medians; fixed, rerun (`a1-first-run/`). A relative binary path
  made the first gate call exit 1 without GPU work.
- First A2 attempt: 104 VGPRs per wave with 1024-thread groups is beyond what the hardware accepts (16 waves per SIMD need at
  most 96): the queue refused the 4 dispatches with `HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS`, the kernel log shows 4
  "Illegal opcode in command stream" lines (one per rejected dispatch, 23:08:48-49 local), no GPU reset, no ring timeout, no
  hang (review: journal read). The kernels now use 64 VGPRs (`attr-w32.log`, `attr-w64.log`); no reported number comes from
  before the fix. The four journal lines are quoted in `RESULTS.md` but not saved in the run folder (`a2-failed-launch/`
  holds the failed logs). The four jobs were fired within three seconds, so the same error came back four times before
  anyone could react.
- The one-SIMD slope over the iteration count contained the loop control (1.052 for 1.000); replaced by the R difference.
- 104 register-initialisation instructions in front of the two A2b timed regions made the two waves start at different
  times (matrix older 16.45 / 1.67); the final harness has none (`a2b-init-variant/`).
- The disassembler decodes wave64 objects as wave32 unless told `--mattr=+wavefrontsize64`; `scripts/leftc_asmcheck.py`
  compares opcodes only, so it does not show this.

### What changed against the earlier documents

- `conversions-and-dots.md` gave dot costs for wave32 only (1.00 independent, 6.00 chained). They hold in wave64 as cycles
  per instruction; the work per instruction doubles. Chained cost is 6.000 in wave64 too (expected 7), conversions chained
  5.000 in both sizes (expected 6 in wave64), and the dot products with both sources in one bank cost 1.188 in wave64
  (expected 2.00 to 2.5). FP8 conversions cost 2.00 cycles in wave64.
- The 4x lead of the matrix unit over the dot path, implied by `conversions-and-dots.md` (dot4 4 multiply-adds per lane per
  cycle, matrix about 16) and not stated as such in it or in `matrix-timing.md`, is a wave32 figure; in wave64 it is 2x.
- RDNA 3's int8 dot ran at half rate in wave32 (READ, `conversions-and-dots.md`); `RESULTS.md` says a wave64 gain for the dot
  path is also seen there, by a different mechanism (not rechecked here). On this card wave32 is already at one cycle and
  wave64 gains by lane work per cycle.
- A matrix wave still excludes vector work on its SIMD in wave64, as in wave32 (`matrix-timing.md` lines 452-463).
- `matrix-timing.md` says the counter's clock equals the driver's reported clock within 0.3 %. On "Default" that does not
  hold (4.8 % gap above).

### Open

- Whether the reported clock omits the overclock offset (INFERRED above, one profile switch away from a test).
- Whole-card runs for `v_cvt_pk_fp8_f32`, `v_dot8_i32_iu4`, the three FP8/BF8 dot4 mixes and the pseudo-scalar forms.
- Why `v_s_exp_f32` and `v_s_rcp_f32` cost 2.125 (steady state 2.00) in wave64 against 1.875 in wave32.
- A sustained load: all whole-card bursts were 50-103 ms, highest hotspot reading 78 C.
- The memory clock moved between 456 and 1124 MHz between jobs of this run (`jobs.log`); none of the jobs is memory-bound, so
  no row depends on it.

## B. FSR 4 from the outside

> **Correction, 2026-11-24 (round 8, [round8.md](round8.md)).** Parts of this section are superseded:
> - The clock-scaled estimate (1.47-1.58 ms at 4K; 0.22-0.30 ms at 1080p) is **withdrawn**. It assumed the cost is inverse to the core clock; round 8 at 3.4 GHz gave 0.346 / 0.652 / 1.663 ms
>   against 0.349 / 0.643 / 1.661 here at 2.3 / 2.3 / 3.2 GHz (MEASURED, agreeing within 1.5 %). That comparison also changes the sample's version and frame generation, so
>   whether the cost follows the core clock is open (INFERRED: this comparison gives no support for the inverse-clock assumption, but is not a clean test).
> - The sample's version was **4.1.1** (the "Caveats" line says so); round 8 used 4.1.2.
> - The presets were **inferred** from the Lighting label, not read; round 8 reads them from the sample's log.
> - Frame generation was on in every cell here and the GPU was only 70-82 % busy (MEASURED); frame generation is associated with the low busy share and possibly with the low clocks at 1080p and 1440p (INFERRED),
>   but round 8's own frame-generation cells still clocked 3.4 GHz, so its part is not isolated.
>
> The text below is kept as written.

Run folder `runs/20261124T090844Z-r7fsr-b1/` (`RESULTS.md`, `analysis.md`, `analyse.py`, per-cell `.log` / `.meta` /
`.guard.csv`, clock cells in `clk/`); how the sample was started and tested: `runs/20261124T073232Z-r7fsr-b0/`.

**Outside-view rule followed.** FSR 4 was studied from the outside only (the owner's rule): AMD's own FidelityFX SDK sample
(v2.3.0 prebuilt, MIT sample code, FSR 4.1.1) was run under GE-Proton and timed with the sample's own GPU timestamps, the
core clock was sampled beside it, and nothing was traced; no shader or DLL was opened and none of the shader dump, override or debug variables
the start script refuses (a name list) was set.

**Method (MEASURED).** 45 cells: 3 output sizes (1920x1080, 2560x1440, 3840x2160, each in a gamescope display of exactly that
size) x 5 requested presets x 3 repeats, interleaved, each through the study gate (shared lock, 20 ms hotspot guard; hotspot
maximum 49-76 C). The number is the label `FFX API FSR Upscaler` (average ms over about 1500 frames). Two things went wrong
and are handled, not hidden: the F6 key presses that select the preset were unreliable, and some cells were disturbed.

**Preset inferred, not trusted.** The requested preset was what ran in only 5 of 45 cells (shifts are modulo 5, so dropped and extra presses cannot be told apart; one or two short is most common,
consistent with dropped presses, not proven; the known-preset check cells ran outside gamescope, the grid inside it). Lighting and GBuffer scale with render size, RasterShadow does not; per output
size, Lighting was fitted as `a + k x render Mpx` (a = 0.016-0.019 ms, k = 0.091-0.094 ms per Mpx, rms log residual 0.02-0.04)
and each cell took the nearest preset. Checks: the known Quality test cell (0.161 ms at 2560x1418, predicted 0.168) and the
known Performance test cell (0.0975, predicted 0.101) come out right; GBuffer and FG prepare agree with Lighting (r 0.99-1.00),
the control RasterShadow does not. Three cells fit no preset within 20 % and are unassigned.

**Disturbed cells.** Rule: RasterShadow above 1.10 x its median (1.709 ms), so above 1.879 ms; 9 of 45 cells are flagged
(1.05 would flag 11, 1.20 flags 5). Results are given with and without them.

**FSR 4 ms (MEASURED), median [min - max], n cells, undisturbed cells only; Native AA and Ultra Performance bracket the preset effect:**

| output size | Native AA | Quality | Balanced | Performance | Ultra Performance | all presets | all presets, disturbed included |
|---|---|---|---|---|---|---|---|
| 1920x1080 | 0.362 n=2 | 0.349 n=2 | 0.342 n=1 | 0.334 n=1 | 0.330 n=3 | 0.342 [0.326-0.363] n=9 | 0.360 [0.326-0.406] n=15 |
| 2560x1440 | 0.713 n=2 | 0.643 n=3 | 0.631 n=4 | 0.620 n=1 | 0.597 n=3 | 0.633 [0.589-0.716] n=13 | 0.632 [0.589-0.716] n=15 |
| 3840x2160 | 1.770 n=2 | 1.661 n=4 | 1.642 n=4 | 1.625 n=3 | 1.597 n=1 | 1.643 [1.597-1.776] n=14 | 1.643 [1.597-1.940] n=15 |

Small n in several columns: read those as single readings (1080p Quality includes one borderline cell, `r3-1920x1080-performance`, 0.5 % under the disturbance line; 4K Ultra Performance is one cell). The all-presets columns mix a different set of presets per size, so ratios carry a few percent of mix bias. The preset matters little: Native AA costs 10 % (1080p), 20 % (1440p)
and 11 % (4K) more than Ultra Performance, and the cost rises with render size by 0.018-0.036 ms per render Mpx on top of a
fixed part (0.33, 0.59, 1.57 ms). Disturbed cells have a slower upscaler too (1080p Quality 0.385 against 0.349), so they are
removed, not only noted.

**Against AMD (READ: gpuopen, FSR Upscaling 4.1.2 manual, performance table, RX 9070 XT, Full FP8, RCAS off, peak GPU clocks;
the table names no quality mode).** 3840x2160: AMD 1316 us, ours 1643 us (1.25x; Quality-only cells 1.26x). 1920x1080:
AMD 352 us, ours 342 us (0.97x; Quality-only 0.99x). No 2560x1440 figure was read. A rough match at best, and the 1080p agreement is possibly coincidental (the clock was far below peak there); AMD's figures were READ from the manual on 2026-11-19 (per the study's notes) and not re-confirmed on 2026-11-24, and the table's mode is not stated in what we have. Not a test of AMD's table, because of the caveats below.

**Clocks (MEASURED; scaled figures INFERRED).** The sample does not keep the card busy at the small sizes: median core clock
2.34 GHz (1080p) and 2.28 GHz (1440p) at about 70 % activity, 3.22 GHz at 82 % at 4K; highest sample 3.59 GHz (6 clock cells,
5 ms sampler, samples with activity of at least 30 % only). Scaling the measured cost to the highest sampled clock, assuming
cost inverse to clock, gives 1.47-1.58 ms at 4K (1.12-1.20x AMD); at 1080p it gives 0.22-0.30 ms, below AMD's 0.352, so the
assumption does not hold there (the cost is not simply clock-bound) and that figure is a floor, not an estimate. The sampler
sees the clock around the 0.3-1.6 ms dispatch, not during it. Three of the six clock cells are flagged disturbed by the rule above.

**Caveats (all cells).** Sample FSR 4.1.1 against AMD's 4.1.2; RCAS sharpening on (AMD: off); frame generation was on in every grid cell (the grid never pressed F7); one test cell pressed F7 once and the FG-prepare label stayed, a dropped key is not excluded and it was not repeated (the prepare pass is a separate label, not in the upscaler number); all cells inside gamescope;
the card is overclocked (+150 MHz, -50 mV, 374 W), so every wall-clock number is for that state; the memory clock changed
between 456 and 772 MHz during cells (logged, quoted, not merged).

### Open

- Whether the small-size cost is clock-bound (the simple scaling fails at 1080p).
- Presets with n=1-2 are single readings; a switch that does not rely on key presses would fill them.
- Frame generation off, RCAS off and FSR 4.1.2 each remove one difference from AMD's table; not possible from outside here.
- Dispatch counts: NOT OBTAINED. The counting layer hooked only submit, present, begin-command-buffer and get-queue under Proton, and all four layer runs counted 0 (`runs/20261124T090844Z-r7fsr-b1/layer*/`).
