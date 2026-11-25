# RDNA 4 scheduling, launch, clocks and sharing (lane 7)

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201, 64 CUs = 128 SIMDs, wave32), ROCm 10.1.0 (hipcc, clang 24),
kernel 7.3.4, RADV for item 6. Lane 7 of `notes/STUDY-PLAN.md`. Written in a staging folder and laid out to be
copied into the study repository unchanged.

Run folders (all `scripts/study-scheduling.sh`, under `runs/`):

- `20261117T165519Z-sched`: build, every short probe (items 1 to 4, 6 to 8) and all 8 deliberate breaks. All checks
  passed. Numbers for those items come from here unless a line says otherwise.
- `20261117T164432Z-sched`: the same probes once more with earlier sources, plus the item 5 traces idle, light,
  stream, FP16 and FP8. Every check passed. After its last break the script stopped with a syntax error, because I
  edited the script file while that run was executing it; every job had already finished and logged.
- `20261117T163033Z-sched`: the first trace run (idle, light, stream, FMA, FP16, FP8; its script also ended with a
  syntax error after the last trace, for the same reason, after every trace had finished). Its FMA trace reached 95 C
  after 2.7 s, so under the thermal rule FMA was not traced again; the FMA row of item 5 comes from this run. Its
  other traces agree with the later run (FP16 216.8 against 216.0 TFLOPS, FP8 416.3 against 416.4, clocks within
  2 MHz).

- `20261117T171514Z-sched`: the study lead's rerun from the repository after the copy: build, every short probe
  and the deliberate breaks, without the item 5 traces (the two trace runs above agree, and the lead did not load
  the card for 20 seconds again while the owner of the machine was away). All checks passed; every check has the
  same status as in `20261117T165519Z-sched` (88 PASS verdict lines, 8 breaks failing as intended), and the cycle
  figures spot-checked are identical (for example the WGP-mode barrier costs 93.7, 93.7, 93.2, 94.7, 106.7, 120.5).

Labels: MEASURED (seen in those runs), READ (from a document or a file), INFERRED (reasoning). Wall-clock and
clock numbers: profile "Testing" (stock clocks, 317 W cap, memory 1258 MHz, performance level auto), checked and
logged before every job; a regulator-level power-limit mod (1.2x) is in the card, so every watt below is the
driver's reading, scaled down on the main rail, not true board power; the desktop was in use the whole time
(25 DRM clients on the card: the compositor and ordinary desktop applications).
- A fourth run, `20261117T165251Z-sched` (earlier sources, a moment of heavy desktop drawing), is the source of the
  Vulkan queue table in item 6 and of the higher drawing correlation in item 7; it is named where it is used.

## In short

- **What a launch costs.** Launching an empty kernel and waiting for it with `hipDeviceSynchronize` takes 17 to
  22 us wall time (median); events see 6 to 7 us of it on the GPU. Launches queued back to back without waiting
  cost 2.1 us each (empty) or 3.4 us (a tiny one-wave kernel). The first launch of a kernel costs 330 to 490 us
  (code loading). After the card has been idle 1 ms or more, the same launch takes 55 to 71 us. Up to about 4,000
  groups the grid size does not matter; beyond that an empty kernel costs about 12 waves per ns, and at most 5.6
  groups per ns (2^20 groups of 1024 threads: 2.6 ms).
- **How fast the clock comes back.** The kernel's own cycle counter against REALTIME shows the same ramp after 10,
  100 or 1000 ms of idle: about 2.79 GHz (one wave) or 2.69 GHz (whole card) for the first 8 ms, then a climb to
  3.2 GHz, 95 % reached at 14 to 15 ms. `pp_dpm_sclk` lags this by tens of milliseconds. A light load then creeps
  up to 3.43 GHz over about 1.5 s; heavy whole-card loads stay at 3.18 to 3.32 GHz.
- **How the card fills up.** Groups are spread over all 4 shader engines and both shader arrays at once. The
  highest-numbered work-group processor (WGP 3) of each array fills first, then WGP 2, 1, 0. Up to 16 waves fit on a
  SIMD; the register rule `min(16, floor(1536 / VGPRs rounded up to 24))` holds exactly, counted in whole groups.
  For wave64 the registers count double. Group placement differs from launch to launch: only 0 to 16 % of groups
  land on the same CU as in the previous launch. A freed slot is refilled after 0.2 to 0.6 us (median). FMA
  throughput scales exactly with the CUs in use (0.39 TFMA/s per CU, 25.1 at 64). One CU streams about 70 GB/s,
  so 9 to 16 CUs already reach the memory bus (623 GB/s).
- **Barriers** cost 27 cycles per wave in CU mode and 94 in the default WGP mode for small groups, rising to 105
  and 120 at 32 waves. A wave waiting at a barrier costs the other waves on its SIMD nothing.
- **Clock, power, temperature.** At rest: about 110 MHz, 47 W reported, hotspot 48 C, fan stopped, memory clock
  stays at 1258 MHz (round 7, `round7.md`: no longer; it follows the load and idles at 456 or 772 MHz). FP16 matrix for 20 s: 216 TFLOPS at 3.32 GHz, 306 W, hotspot steady at 90 to 92 C once the fan
  reaches about 3,650 rpm. FP8 matrix: 416 TFLOPS at 3.18 GHz, against the 317 W limit (the driver's
  throttle bit PPT0 is set in 393 of 400 samples: every one after the first 0.34 s). Dual-issue FMA reached 95 C after 2.7 s and was stopped. After any load
  the hotspot falls below 55 C within 0.4 s. FP8 costs 0.76 pJ per flop in reported power, FP16 1.42, streaming
  260 pJ per byte.
- **Overclock profile (owner's, short bursts only).** In 0.1 s bursts FP16 matrix work, FMA work and memory reads
  run at the same rate as on the stock profile; FP8 matrix work is 0.6 to 1.9 % faster (421.5 against 414 to 419
  TFLOPS) because the clock does not sag; the power the driver reports (a scaled reading) is 6 to 15 % lower. The
  memory overclock does not raise
  the memory clock (it stays at 1258 MHz) or the read rate. This is one overclock round against five clean stock
  rounds. Loads lasting seconds were not measured on that profile (item 5a). Round 7 (`round7.md`): the memory clock now
  reaches 1408 MHz under load.
- **How much the desktop disturbs a measurement, and how to tell.** A spinning wave loses about 1.1 us roughly
  every 2.8 ms (0.05 % of its time). These stalls hit every SIMD at once: 99 % of the stall clusters coincide within 1 us between
  two processes, and 96 % across 16 waves on all 4 shader engines. Desktop drawing seems to add more, on weak
  evidence: lost time per 20 ms was about 9 us in quiet windows and 18 us in the 9 of 200 windows in which other
  programs drew (39 us in another run with more drawing). The cycle counter keeps
  counting through a stall, so a stalled sample simply shows about 3,900 extra cycles. **Rule:** time short chunks
  of known work (below about 7,000 cycles, or a typical stall no longer stands out), and treat any chunk that
  takes more than 1.5 times the median chunk plus 64 cycles as disturbed.
  Drop or repeat such samples, or quote the median of several repetitions.
- **Two compute processes.** Two whole-card processes never share the card: whole launches alternate, A then B,
  with 25 us between them, and each process gets half the throughput. Two 16-CU processes run side by side (both on
  the card 97 to 99 % of the time). A single wave beside a busy process keeps its 480 cycles per step, except when
  the other process's older waves share its SIMD: then it starves for a whole launch (1.7 to 2.0 ms seen).
- **Vulkan queues.** The compute-only queue family shows two peaks in submit-to-completion time, about 30 us and
  about 100 us. When it runs alone, 30 to 42 % of its submissions fall in the slow peak (four runs; median about
  104 us). The graphics queue has one peak near 30 us. A neighbouring project saw a compute-queue stall of 0.24 ms
  on RDNA 3; this is in the same direction and about 70 us.
- **For a kernel author or a benchmarker:**
  - Warm up for at least 20 ms before timing; a light load needs about 1.5 s to reach its final clock.
  - Make each timed launch at least 1 ms, or queue launches back to back. A synchronize after every launch costs
    about 18 us, and after idle about 60 us.
  - Use CU mode (`-mcumode`) for barrier-heavy groups.
  - Keep VGPRs at or below 96 for 16 waves per SIMD, and remember that wave64 needs twice the registers.
  - Never poll REALTIME from many waves: it delayed other waves' REALTIME reads by whole spin lengths.
  - Expect another process's kernels to take turns with yours, not to overlap, unless both fit on the card.
  - Treat cycle-counter samples with the chunk rule above.

## 1. Launch overhead under HIP (`sched_launch`)

Empty kernel, launch then `hipDeviceSynchronize`, host wall time (steady_clock) and HIP event time around it.
200 samples (50 for the two largest grids) after 5 warm-up launches. MEASURED, run `20261117T165519Z-sched`.

| Grid (groups) | Block | Wall us (median / p10 / p90 / max) | GPU events us (median / p10 / p90 / max) |
|---|---|---|---|
| 1 | 32 | 21.6 / 21.4 / 24.2 / 72.2 | 6.9 / 6.7 / 7.0 / 7.1 |
| 4096 | 32 | 21.3 / 21.0 / 21.8 / 79.1 | 7.0 / 6.8 / 7.1 / 7.2 |
| 65536 | 32 | 30.8 / 30.5 / 31.3 / 35.7 | 16.5 / 16.4 / 16.6 / 16.7 |
| 1048576 | 32 | 201.1 / 200.6 / 204.7 / 214.3 | 186.6 / 186.3 / 189.5 / 190.0 |
| 1 | 256 | 20.9 / 20.6 / 21.9 / 227.1 | 6.4 / 6.2 / 6.5 / 6.8 |
| 4096 | 256 | 23.4 / 23.1 / 24.5 / 155.0 | 8.8 / 8.7 / 9.0 / 9.3 |
| 65536 | 256 | 82.5 / 76.0 / 86.2 / 148.3 | 67.9 / 61.2 / 68.9 / 71.7 |
| 1048576 | 256 | 698.9 / 690.2 / 703.1 / 723.8 | 683.3 / 674.9 / 686.1 / 686.4 |
| 1 | 1024 | 21.3 / 21.0 / 21.7 / 76.8 | 6.8 / 6.6 / 6.9 / 23.2 |
| 4096 | 1024 | 30.3 / 30.1 / 30.6 / 45.3 | 16.2 / 16.1 / 16.2 / 16.4 |
| 65536 | 1024 | 201.0 / 200.6 / 204.8 / 217.6 | 186.6 / 186.5 / 189.5 / 192.2 |
| 1048576 | 1024 | 2778.2 / 2766.1 / 2790.9 / 2884.6 | 2638.8 / 2638.6 / 2639.2 / 2642.1 |

(16 and 256 groups: within 1 us of the 1-group rows in every block size; full table in the run folder's
`tables.md`.)

| Case | Result (MEASURED) |
|---|---|
| First launch of a kernel never launched before (empty) | 328.5 us (390 and 490 us in two earlier runs); the same kernel afterwards 17.4 us (p10 17.0, p90 17.7) |
| One-wave kernel (1024 FMAs per lane, one store), launch + synchronize | 22.2 us (p10 21.9, p90 22.8); GPU events 7.8 us; body 0.88 us |
| Launch + synchronize only (no events), empty / one wave | 17.3 / 18.7 us median (700 samples each) |
| Back to back, 2000 launches then one synchronize, empty / one wave | 2.13 / 3.36 us per launch (7 repetitions, range 2.12-2.14 / 3.35-3.37) |
| Host enqueue time alone (the launch call returns), empty / one wave | 0.49 / 0.53 us per launch |
| Event-to-event spacing of 200 back-to-back empty kernels with an event after each | 5.64 us (p10 5.60, p90 5.68) |

- Fixed cost: about 17 us of wall time per synchronized launch, of which 6 to 7 us shows between GPU events
  (INFERRED: the rest is host-side submission, the completion signal and the wake-up of the waiting thread). A
  launch without waiting costs 0.5 us on the host and 2.1 us on the GPU queue. Recording an event after each kernel
  raises the GPU spacing to 5.6 us.
- Grid size costs nothing up to about 4,000 groups. Beyond that the time grows with the number of waves, not
  groups: 2^20 groups of 32 threads (2^20 waves) take 187 us, 2^20 of 256 (2^23 waves) 683 us, 2^20 of 1024 (2^25
  waves) 2639 us: 12.3 to 12.7 empty waves per ns (INFERRED from the rows). With 32-thread groups the limit is
  groups, not waves: 2^20 groups in 187 us = 5.6 groups per ns.

After idle (one-wave kernel; idle = host sleep before each launch; 100, 100, 100, 30, 8 samples):

| Idle before | Wall us (median / p10 / p90 / max) | GPU events us (median / p90) | Clock inside the kernel (MHz, median / p10 / p90) |
|---|---|---|---|
| 0 ms | 24.4 / 24.1 / 25.0 / 30.4 | 9.4 / 9.6 | 2253 / 2253 / 2347 |
| 1 ms | 54.8 / 31.3 / 346.5 / 371.9 | 9.5 / 9.6 | 2253 / 2253 / 2347 |
| 10 ms | 65.0 / 58.8 / 189.2 / 370.8 | 9.5 / 9.7 | 2253 / 2253 / 2347 |
| 100 ms | 65.5 / 58.0 / 336.2 / 367.0 | 9.5 / 9.6 | 2253 / 2253 / 2253 |
| 1000 ms | 71.2 / 68.1 / 76.1 / 76.6 | 9.7 / 9.8 | 2253 / 2166 / 2253 |

- After 1 ms or more of idle the wall time triples (55 to 71 us median, tails to 370 us), while the GPU-side time
  stays 9.5 us. INFERRED: the cost is in waking the host thread and the GPU from idle states, not in the dispatch.
- The kernel body (0.9 us, too short to resolve the clock better than one REALTIME tick, about 4 %) shows about
  2.25 GHz in every case. That figure is lower than the 2.79 GHz of the ramp and probably includes a fixed read
  latency, so do not read it as a physical clock; what holds is that a lone short kernel never sees the loaded
  clock.

Clock ramp seen from inside: after the idle time, a kernel runs 150 ms; wave 0 of group 0 records its cycle counter
and REALTIME every 8 chunks of 128 FMAs (about 0.3 us); clock per window = cycles / REALTIME. One wave, or the whole
card (64 groups of 1024 threads). The hotspot went 51 to 74 C during the 150 ms whole-card ramp.

| ms from kernel start | one wave, idle 10 ms | one wave, idle 100 ms | one wave, idle 1000 ms | whole card, idle 1000 ms |
|---|---|---|---|---|
| 0 | 2786 | 2787 | 2786 | 2688 |
| 4 | 2786 | 2787 | 2786 | 2687 |
| 8 | 2787 | 2786 | 2787 | 2746 |
| 10 | 2848 | 2830 | 2832 | 2821 |
| 12 | 2966 | 2939 | 2941 | 2941 |
| 14 | 3121 | 3082 | 3087 | 3097 |
| 17 | 3236 | 3238 | 3238 | 3226 |
| 30 / 100 / 145 | 3228-3229 | 3228-3229 | 3227-3228 | 3213-3223 |
| time to 95 % of final | 14 ms | 14 ms | 14 ms | 14 ms (15 ms in the final run) |

(MHz; rows at 0.25 ms resolution below 30 ms are in the logs. Rows from `20261117T164432Z-sched`; the final run
`20261117T165519Z-sched` agrees within 5 MHz at 0 ms and after 17 ms, for example 2787, 2785, 2786, 2693 at 0 ms.)

- The ramp is a fixed sequence that does not depend on how long the card was idle (10 ms to 1 s): a plateau near
  2.79 GHz for 8 ms (2.69 GHz with every SIMD busy), then a 6 ms climb to 3.23 GHz. MEASURED. `pp_dpm_sclk`
  sampled every 5 ms showed 2837 MHz at 15 ms and 3199 at 30 ms, so the driver's reading lags the real clock.
- `pp_dpm_sclk` at 50 ms sampling reached 97 % of its steady value 0.1 s (FMA, FP8), 0.5 s (FP16) or 1.4 s (light,
  stream) after the start of a long load (item 5), because those loads keep climbing slowly after the fast ramp.

## 2. Dispatch and placement (`sched_place`)

Each wave records `HW_ID1`, the shader engine (message `GET_SE_AID_ID`) and REALTIME at start and end; it waits
64,000 cycles on its own counter (about 20 us). 5 launches per row, both builds (WGP mode, the HIP default, and CU
mode). Concurrency = largest number of start-end intervals open at once. "Slot gap" = time from the last end of a
group on a CU to the start of a group that began later than the launch's first round. MEASURED, run
`20261117T165519Z-sched`, WGP mode unless the row says CU.

| Grid | Block | SIMDs used | Max waves per SIMD | Max groups per CU / per WGP | First 64 groups started within | Span | Slot gap us (median / p90) | Same CU as launch 0 |
|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 16 | 1 | 1 / 1 | 0.08 us | 23.1 us | - | 0 % |
| 128 | 32 | 107 | 2 | 2 / 4 | 0.12 | 23.4 | - | 0 % |
| 512 | 32 | 128 | 5 | 8 / 16 | 0.24 | 24.5 | - | 0 % |
| 4096 | 32 | 128 | 16 | 32 / 64 | 0.12 | 55.5 | 0.60 / 1.36 | 3 % |
| 4096 | 64 | 128 | 16 | 16 / 32 | 0.20 | 103.7 | 0.20 / 1.00 | 5 % |
| 64 | 256 | 128 | 4 | 2 / 2 | 1.24 | 24.4 | - | 0 % |
| 4096 | 256 | 128 | 16 | 8 / 8 | 1.20 | 390.2 | 0.20 / 0.36 | 14 % |
| 64 | 1024 | 128 | 16 | 2 / 2 | 5.40 | 29.1 | - | 50 % |
| 4096 | 1024 | 128 | 16 | 2 / 2 | 5.32 | 1600.7 | 0.40 / 0.48 | 16 % |
| CU 4096 | 256 | 128 | 16 | 4 / 8 | 1.04 | 391.8 | 0.32 / 0.44 | 6 % |
| CU 4096 | 1024 | 128 | 16 | 1 / 2 | 5.04 | 1623.6 | 0.36 / 0.44 | 5 % |

(All 40 rows, both modes: run folder `tables.md`.)

- Fill order (`ORDER` lines): the first groups of a launch go to WGP 3 of every shader array of all 4 shader engines,
  then WGP 2, then 1, then 0; within a few ticks of each other for small groups. With 1024-thread groups the WGPs
  fill about 0.6 to 1.5 us apart (WGP 3 at 0 to 0.6 us, WGP 2 at 1.4 to 2.4 us, WGP 1 from 2.8 us). MEASURED.
- Spread: a 16-group grid of single waves used 16 different SIMDs on 16 different CUs over 16 WGPs on all 4 shader
  engines in every launch (checked, verdict `placement16`). A 128-group grid of single waves used only 107 of 128
  SIMDs (2 waves on some SIMDs while others stayed empty): the dispatcher balances CUs and WGPs, not SIMDs.
- Start spread: the first 64 groups of 1024 threads (2048 waves) carry start stamps within 5.0 to 5.4 us of the
  first; 64 groups of 256 (512 waves) within 1.2 us; 64 of 64 (128 waves) within 0.2 to 0.3 us, about 2.5 ns per
  wave. This is not the dispatcher's limit: empty waves are placed at about 12 per ns (item 1). INFERRED: the stamp
  follows two message reads (shader engine, REALTIME) per wave, and many waves sending messages at once delay each
  other (see Method).
- WGP mode against CU mode: in WGP mode a group of 8 or more waves spreads over the 4 SIMDs of a WGP (2 CUs); in CU
  mode over the 2 SIMDs of one CU. A 1024-thread group fills a WGP's 64 wave slots twice in WGP mode (2 groups per
  WGP) and one CU in CU mode (2 per WGP, 1 per CU). The slot gap is the same in both (0.2 to 0.6 us).
- Not repeatable: from one launch to the next only 0 to 16 % of groups (50 % for 64 groups of 1024, which have only
  two choices per WGP) land on the same CU. Code must not assume a group-to-CU mapping.
- Registers (`regs`, 256-thread groups, 4096 of them):

| VGPRs (reported) | Rule `min(16, floor(1536 / rounded))` | Measured waves per SIMD, WGP mode | CU mode |
|---|---|---|---|
| 11, 44, 68, 92 | 16 | 16 | 16 |
| 124 (rounded 144) | 10 | 10 | 8 |
| 154 (168) | 9 | 8 | 8 |
| 184 (192) | 8 | 8 | 8 |
| 244 (264) | 5 | 4 | 4 |

  The rule holds once whole groups are counted: a 256-thread group adds 2 waves per SIMD in WGP mode and 4 in CU
  mode, so the count is the rule rounded down to that step (verdict `sched_regs`, every row). With single-wave
  groups (helper's probe, item 4) the rule holds exactly at every count from 25 to 254 VGPRs.
- Throughput against CUs (`scale`, one 512-thread group per CU forced by 40 KiB of unused LDS, CU mode, 8 waves per
  SIMD; 0.2 s warm-up; loaded `pp_dpm_sclk` median 3226 MHz):

| CUs | FMA (dual issue) TFMA/s | per CU | Stream (512 MiB, 16-byte loads) GB/s | per CU |
|---|---|---|---|---|
| 1 | 0.390 | 0.390 | 71.4 | 71.4 |
| 2 | 0.779 | 0.390 | 138.8 | 69.4 |
| 4 | 1.558 | 0.390 | 281.3 | 70.3 |
| 8 | 3.123 | 0.390 | 543.1 | 67.9 |
| 16 | 6.262 | 0.391 | 623.2 | 39.0 |
| 32 | 12.529 | 0.392 | 625.2 | 19.5 |
| 64 | 25.105 | 0.392 | 622.7 | 9.7 |

  FMA scales linearly: 0.389 to 0.392 TFMA/s per CU at every count, which is 0.95 of 128 lane-FMAs per cycle per CU
  (2 SIMDs x 32 lanes x dual issue) at the loaded 3.23 GHz (INFERRED; lane 4 found 0.945). The whole-card value,
  25.1 TFMA/s, is within 2 % of lane 4's 24.7 (verdict `whole_card_fma`). The FMA trace of item 5 reached only
  23.7 TFMA/s (0.90 of that peak) with the same kernel shape but 50 ms launches; not explained. A first version with 1 ms launches gave 19.7 TFMA/s: short launches run at 2.5 to 2.8 GHz.
  One CU with 32 waves streams 70 GB/s, so the 623 GB/s bus is full with 9 to 16 CUs.

## 3. Barriers and cross-wave communication (`sched_barrier`, helper; rerun here)

One work-group of N waves; cycles per `__syncthreads()` per wave from each wave's own counter, two loop lengths
differenced. Calibration: 1.000 cycles per FMA in both builds. MEASURED, run `20261117T165519Z-sched`, 6 launches
per row, 0 disturbed samples.

| N waves | WGP mode: cycles per barrier (SIMDs / CUs spanned) | CU mode (SIMDs / CUs) |
|---|---|---|
| 1 | 93.7 (1 / 1) | 27.3 (1 / 1) |
| 2 | 93.7 (2 / 1) | 27.3 (2 / 1) |
| 4 | 93.2 (4 / 2) | 28.1 (2 / 1) |
| 8 | 94.7 (4 / 2) | 37.6 (2 / 1) |
| 16 | 106.7 (4 / 2) | 58.7 (2 / 1) |
| 32 | 120.5 (4 / 2) | 104.6 (2 / 1) |

- A barrier costs 27 cycles in CU mode and 94 in WGP mode even for a single wave on one SIMD: the mode, not the
  span, sets the base cost (INFERRED: WGP mode synchronizes through the WGP-level barrier logic). It grows with the
  number of waves, more in CU mode, where more waves share a SIMD.
- With work between barriers the overhead mostly disappears: WGP mode, 4 waves, 16 / 64 / 256 FMAs between
  barriers: 31 / 1.8 / 4.0 cycles extra per iteration; CU mode 8.5 / 1.8 / -1.4. With 8 waves on 2 SIMDs (CU mode)
  the median wave waits 65 to 254 cycles more per iteration while the slowest waits nothing: that is the waves
  queueing for their SIMD, not barrier cost.
- Imbalance: with wave 0 doing 256 FMAs per iteration and the others none, every wave takes wave 0's 301 to 305
  cycles per iteration (they wait in the barrier).
- A waiting wave gives its SIMD away completely: an FMA wave whose SIMD partner is blocked in `s_barrier_wait` runs
  0.984 cycles per FMA, the same as with the partner gone (0.984). The control, where the partner also runs FMAs,
  shows the probe can see sharing (12.2 cycles per FMA for the slowest single loop, 1.045 per FMA for the SIMD).
- Dependent atomic add, cycles per atomic per wave (WGP mode): LDS 56.0 (distinct or the same address, 1 to 32
  waves; CU mode 57 to 64); global, distinct addresses 270 to 280; global, one address for all waves 2,121 to 2,206
  (CU mode, 32 waves: 2,726). INFERRED: a same-address global atomic is serialized at the L2, about 0.65 us each
  at 3.3 GHz.

## 4. Wave32 against wave64 under HIP (`sched_wave`, helper; rerun here)

Built twice in CU mode: wave32 and `-mwavefrontsize64 -DW64=1`. Lane 8 (`compiler-and-hazards.md` item 8) found
wave64 builds and runs, WMMA at 16 cycles, and the host `warpSize` stuck at 32. This item extends that. MEASURED,
run `20261117T165519Z-sched`.

| Question | wave32 | wave64 |
|---|---|---|
| Device `warpSize`, `__builtin_amdgcn_wavefrontsize()` | 32, 32 | 64, 64 |
| `__AMDGCN_WAVEFRONT_SIZE` | not defined (-1) | not defined (-1) |
| Host `hipDeviceProp_t::warpSize` | 32 | 32 (wrong for this build) |
| Waves in a 64-thread block (placement records) | 2 | 1 |
| Independent `v_fma_f32`, one wave alone: lane-FMAs per cycle per SIMD | 32.00 (1 instruction per cycle) | 51.85 (0.81 instructions per cycle) |
| Same with 2, 4, 8 waves on the SIMD | 32.02-32.05 | 50.34, 49.90, 49.66 |
| C++ accumulate form (clang pairs it into `v_dual_fmac_f32` in wave32) | 64.00 (63.7-65.3 with 2-8 waves) | 51.20 (50.8-51.2) |
| `v_wmma_f32_16x16x16_f16` chained | 16.000 cycles | 16.000 cycles |
| Streaming 256 MiB, 8 x 1024 / 256 x 256 / 1024 x 256 threads | 505 / 618 / 619 GB/s | 496 / 624 / 620 GB/s |
| Waves per SIMD against VGPRs | `min(16, floor(1536 / roundup24(VGPRs)))` | `min(16, floor(1536 / roundup24(2 x VGPRs)))` (25, 49, 97, 130, 146-179, 195-252 VGPRs: 16, 12, 7, 5, 4, 3 waves) |

- Wave64 vector FMAs run at 0.81 instructions per cycle (51.85 lane-FMAs per cycle): faster than two 32-lane passes
  (32 per cycle), slower than one 64-lane pass. Clang emits no dual issue in wave64, so the best wave32 code
  (dual-issue FMA, 64 lane-FMAs per cycle) beats wave64 by 23 %. Not explained here (INFERRED candidates: some wave64
  VALU instructions issue in one pass, others in two).
- Matrix and memory rates per SIMD are the same for both sizes.
- Registers: a wave64 wave with the same per-lane VGPR count takes twice the register file, rounded to 24 after
  doubling (verdict `waves_per_simd_rule`, every count; for example 97 VGPRs: 7 waves, not 6): the 1536 limit is
  per 32 lanes. Wave64 kernels in 8-wave groups spilled to scratch at high register counts in the helper's first
  attempt; the table uses 1-wave groups.
- Check added by the lead: in my first two runs the script built this probe without `-mcumode`; the multi-wave FMA
  rows then read 0.00 and every verdict still passed. The probe now fails when any multi-wave row is below 90 % of
  the one-wave rate (`fma_rows`), and the script runs that wrong build as a deliberate break.

## 5. Clock, power and temperature against load and time (`sched_trace`, read-only)

Each trace: 1 s at rest, then launches of about 50 ms until 20 s or the 95 C stop, then 30 s of cool-down; sysfs
every 50 ms (`pp_dpm_sclk`, `pp_dpm_mclk`, reported power, edge, hotspot and memory temperature, fan, busy; and from
`gpu_metrics` the throttle bits and the graphics voltage); every launch records each wave's placement, cycle counter
and REALTIME. Loads: FP16 and FP8 matrix (one wave on each of the 128 SIMDs, as lane 4), dual-issue FMA (8 waves
per SIMD), streaming reads of 1 GiB, and "light" (the FMA kernel on 4 CUs). MEASURED.

| Load | Ran | sclk (steady median) | Counter clock | Rate | Reported power (median) | Hotspot max | Edge / memory max | Fan | Throttle bits (`indep_throttle_status`) | Reported energy |
|---|---|---|---|---|---|---|---|---|---|---|
| idle (10 s) | - | 111 MHz (107-149) | - | - | 47 W | 49 C | 48 / 52 C | 0 rpm | none | - |
| light, 4 CUs | 20 s | 3426 | 3426 | 3.16 TFLOPS | 111 W | 63 C | 49 / 52 | 1,700-2,500 | none | 35 pJ/flop |
| stream, 1 GiB | 20 s | 3422 | 3421 | 638 GB/s | 166 W | 60 C | 52 / 74 | 0, then 2,250 | none | 260 pJ/byte |
| FMA, whole card | 2.7 s, stopped at 95 C | 3218 | 3219 | 47.4 TFLOPS (23.7 TFMA/s) | 317 W | 95 C | 54 / 54 | 621, then 3,335 | (not recorded in that run) | 6.7 pJ/flop |
| FP16 matrix | 20 s | 3320 | 3319 | 216.0 TFLOPS | 306 W | 92 C | 55 / 58 | 3,640-3,660 | bit 36 in 8 of the first 40 samples, then none | 1.42 pJ/flop |
| FP8 matrix | 20 s | 3179 | 3180 | 416.4 TFLOPS | 317 W | 94 C | 55 / 60 | 3,640-3,660 | PPT0 in 393 of 400 samples (all but the first 7, up to 0.34 s); bit 36 also in 15 samples of the first 2 s | 0.76 pJ/flop |

(Watts: driver reading, scaled down by the power-limit mod. Energy = reported power / rate, so it carries the same
scale error. FMA row from run `20261117T163033Z-sched`, others from `20261117T164432Z-sched`.)

Lines from the FP8 trace (2-second windows):

```
WINDOW fp8 0-2 s: sclk_med=3180 counter_clk_med=3179 rate_TFLOPS=416.375 power_W_reported_med=316 hotspot_max=93 edge_max=54 mem_C_max=54 fan_rpm_med=1617 mclk_med=1258 vgfx_mV_med=1056 throttle=PPT0+TEMP_HOTSPOT(0x1000000001, in 39 of 40 samples)
WINDOW fp8 2-4 s: sclk_med=3179 counter_clk_med=3179 rate_TFLOPS=416.341 power_W_reported_med=317 hotspot_max=93 edge_max=55 mem_C_max=56 fan_rpm_med=3545 mclk_med=1258 vgfx_mV_med=1051 throttle=PPT0(0x1, in 40 of 40 samples)
WINDOW fp8 18-20 s: sclk_med=3179 counter_clk_med=3180 rate_TFLOPS=416.383 power_W_reported_med=317 hotspot_max=93 edge_max=54 mem_C_max=60 fan_rpm_med=3641 mclk_med=1258 vgfx_mV_med=1052 throttle=PPT0(0x1, in 40 of 40 samples)
COOL fp8 +0.5 s: hotspot=54 edge=51 sclk=112 power_W_reported=48 fan=1810
COOL fp8 +1.0 s: hotspot=52 edge=49 sclk=110 power_W_reported=47 fan=0
COOL fp8 +5.0 s: hotspot=48 edge=47 sclk=111 power_W_reported=47 fan=0
```

(In a WINDOW line the throttle mask is the OR over the window's samples and "in N of M samples" counts the samples
with any bit set. In the 0-2 s window: PPT0 alone in 24 samples, PPT0 with bit 36 (`TEMP_HOTSPOT`) in 9, bit 36
alone in 6, none in 1.)

From the FMA trace (the one that reached 95 C):

```
WINDOW fma 0-1 s: sclk_med=3215 counter_clk_med=3213 rate_TFLOPS=47.324 power_W_reported_med=305 hotspot_max=92 edge_max=53 mem_C_max=54 fan_rpm_med=621 mclk_med=1258
WINDOW fma 1-2 s: sclk_med=3217 counter_clk_med=3218 rate_TFLOPS=47.438 power_W_reported_med=317 hotspot_max=93 edge_max=54 mem_C_max=54 fan_rpm_med=2591 mclk_med=1258
WINDOW fma 2-3 s: sclk_med=3220 counter_clk_med=3220 rate_TFLOPS=47.462 power_W_reported_med=318 hotspot_max=95 edge_max=54 mem_C_max=54 fan_rpm_med=3335 mclk_med=1258
RESULT trace fma load_s=2.71 launches=64 stopped_at_95C=yes after 2.713470 s
COOL fma +0.5 s: hotspot=53 edge=51 sclk=111 power_W_reported=48 fan=3431
```

- Time to full clock: from inside the kernel 14 to 15 ms (item 1). `pp_dpm_sclk` at 50 ms sampling: 0.1 s for FMA and
  FP8, 0.5 s for FP16; light and stream loads first reach about 3.29 GHz and then climb to 3.42 GHz over about 1.4 s.
- Steady clock depends on the load: 3.42-3.43 GHz for light and memory-bound loads, 3.32 GHz for FP16 matrix,
  3.22 GHz for dual-issue FMA, 3.18 GHz for FP8 matrix. Inside one 20 s trace the clock does not step down: FP16
  3322 to 3319 MHz, FP8 3179 to 3180, light and stream flat after their ramp.
- Which limit acts (READ from `gpu_metrics`; bit names READ from the kernel's `SMU_THROTTLER_*` list in
  `drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h`, Linux master, 2026-11-17: bit 0 is `PPT0`, a power throttler,
  and bit 36 is `TEMP_HOTSPOT`, a temperature throttler; in `smu_v14_0_2_ppt.c` a bit is set when the firmware
  reports a non-zero throttling percentage for that throttler; that this card runs that file's code is INFERRED
  from the kernel log's SMU 14 line): FP8 runs at the power limit (PPT0)
  from 0.34 s on (393 of 400 samples), at 317 W reported, equal to `power1_cap`. FP16 shows no bit after the first 2 s. During the first
  2 s of both matrix loads bit 36, the hotspot throttler, is set in some samples, at hotspot readings from 57 to
  92 C in the 50 ms samples, so it does not wait for the sensor reading to reach a limit near 95 C: why it acts
  that early (a faster or different internal reading, or a controller that acts on the rate of rise) is not known.
  The probe's name table had `PROCHOT_CPU` and `PROCHOT_GFX` at bits 48 and 49 (the kernel has 46 and 47); those
  bits never appeared in a trace, and the table is corrected. FMA reached the cap (317 W) within 1 s and 95 C
  after 2.7 s. INFERRED: its 16 FMA chains in every lane heat the die more per watt in one spot than the matrix
  units; that run predates the throttle-bit sampler.
- Comparison with lane 4: lane 4's 0.17 s launches gave FP16 211 TFLOPS at 3.22 GHz and FP8 420 at 3.20. Over 20 s
  FP16 runs faster (216 TFLOPS, 3.32 GHz) and FP8 slightly slower (416, 3.18 GHz); the first 1.5 s of each trace
  agrees with lane 4 within 1 % (verdicts `rate`).
- The counter's clock equals `pp_dpm_sclk` within 0.03 % in every trace once both see the same window (verdicts
  `counter_vs_sclk`; ratios 0.9999 to 1.0003).
  Round 7 (`round7.md`): on the "Default" profile the counter's clock is 140-155 MHz (4.8 %) above `pp_dpm_sclk`, which matches
  the +150 MHz core offset; INFERRED that the reported clock omits the overclock offset, untested.
- Memory clock stays at 1258 MHz at rest and under every load (READ in every sample): this card does not lower its
  memory clock at idle on this desktop (INFERRED: the attached displays keep it up).
  Round 7 (`round7.md`): the lock is gone since one monitor was moved to the iGPU (owner's statement; the link is INFERRED): idle reads 456 or 772 MHz and a load takes it to
  1408 MHz within about 16 ms; the cause of the lock is not tested further.
- Cool-down: after every load the hotspot is below 60 C within 0.15 s and below 55 C within 0.4 s, and back at about
  48 C within 5 s. The fan stops (0 rpm) within 1 s of the load ending. The edge sensor never exceeded 55 C: the
  hotspot is a local reading.
- Idle state and power settings (READ, nothing changed): `power_dpm_force_performance_level` auto;
  `pp_power_profile_mode` offers BOOTUP_DEFAULT (selected), 3D_FULL_SCREEN, POWER_SAVING, VIDEO, VR, COMPUTE and
  CUSTOM, with per-clock activity parameters (for example COMPUTE: GFXCLK minimum active frequency type 4 at
  600 MHz). MEASURED: while a ROCm kernel runs, BOOTUP_DEFAULT stays selected; it does not switch to COMPUTE.
  `notes/research-9-kernel-driver.md` (READ) says KFD requests COMPUTE when queues are active, and that the
  automatic switching is skipped when overdrive is enabled. INFERRED: LACT's overdrive is the reason.
  `pp_od_clk_voltage`: SCLK offset 0 MHz (range -500 to +1000), memory 97-1259 MHz (range up to 1500), voltage
  offset 0 mV (range -200 to 0); `power1_cap` 317 W (default 317, maximum 374); temperature limits: edge and
  junction critical 110 C, emergency 115 C; memory 108 / 113 C. What changing them would do was not measured
  (brief).

## 5a. The owner's overclock profile against the stock profile, short bursts only (`scripts/study-overclock.sh`)

In plain words: in bursts of about 0.1 s the overclock profile changes very little. FP16 matrix work, FMA work and
memory reads run at the same rate on both profiles. FP8 matrix work is 0.6 to 1.9 % faster on the overclock profile,
because there the clock holds at 3.22 GHz while on the stock profile it sags 19 to 47 MHz below that. The memory overclock
does not raise the memory clock (the driver keeps it at 1258 MHz) or the read rate. The power the driver reports
(a scaled reading, not true board power) is 6 to 15 % lower on the overclock profile for the same work. The
overclock side is a single round, compared with five clean stock rounds. What the profile does to a load that
lasts seconds was not measured (owner decision: short bursts only on that profile).
Round 7 (`round7.md`) measured loads of 0.05 s to 20 s on "Default": the memory clock now reaches 1408 MHz (the 1409 request applies).

Profiles (READ from sysfs at the start of each round, `profile-<round>.txt`):

| | "Testing" (stock) | "Default" (overclock) |
|---|---|---|
| `power1_cap` | 317 W | 374 W |
| `OD_SCLK_OFFSET` | 0 MHz | +150 MHz |
| `OD_VDDGFX_OFFSET` | 0 mV | -50 mV |
| `OD_MCLK` maximum requested | 1259 MHz | 1409 MHz |
| top level in `pp_dpm_mclk` (the active one) | 1258 MHz | 1258 MHz |

Rounds: A (stock), B (overclock), C (stock again) in run `20261117T182423Z-overclock`. Rounds A and C were partly
disturbed (below), so five more stock rounds were taken with no profile switch at all: S1 and S2 in run
`20261117T182919Z-overclock`, five minutes later; S3 (run `20261117T184031Z-overclock`) and S4, S5 (run
`20261117T184115Z-overclock`), 16 to 17 minutes later, while testing the reworked script (see "Checks of the
script" below). S3 ran with a stand-in for `lact` (its profile label comes from the stand-in; that the card was on
the stock settings is READ from sysfs in its `profile-A-testing.txt`: cap 317 W, offsets 0); S4 and S5 used the real
`lact`. "Clean stock rounds" below means S1 to S5; for FP16, which was not disturbed in round A, also A; for FMA,
which was not disturbed in round C, also C.
Every job is the `--short` form of `mat_card` (lane 4's whole-card probe: one warm launch and three timed launches of
about 85 ms per setting; W = waves per SIMD, 1 to 16) or the streaming-read test (`bandwidth 1024 50`, three times).
Clock = the kernel's cycle counter against REALTIME, median over the waves. Power = the driver's `power1_average`
(an averaged sensor, scaled down by the power-limit mod), read every 50 ms by a host thread, so about five readings
per setting; the tables give the highest of them. MEASURED.

FP16 matrix (`reg:f16`), five settings W = 1, 2, 4, 8, 16 per round:

| Round | Profile | TFLOPS at W=1 | TFLOPS, all W | Clock, all W (MHz) | Highest reported power, all W (W) | Hotspot max (C) |
|---|---|---|---|---|---|---|
| A | stock | 210.8 | 205.8-210.8 | 3218-3219 | 256-271 | 77-80 |
| B | overclock | 210.1 | 205.4-210.1 | 3213-3216 | 228-243 | 71-74 |
| C | stock | 191.4 (disturbed) | 191.4-206.8 | 2923-3216 | 178-260 | 59-75 |
| S1 | stock | 210.9 | 205.6-210.9 | 3219-3220 | 256-273 | 76-80 |
| S2 | stock | 210.8 | 205.7-210.8 | 3218-3220 | 264-273 | 77-79 |
| S3 | stock | 210.9 | 205.8-210.9 | 3219-3220 | 257-272 | 79-81 |
| S4 | stock | 210.7 | 205.8-210.7 | 3219-3219 | 255-274 | 77-80 |
| S5 | stock | 210.9 | 205.7-210.9 | 3219-3221 | 266-271 | 77-78 |

FP8 matrix (`reg:fp8`):

| Round | Profile | TFLOPS at W=1 | TFLOPS, all W | Clock, all W (MHz) | Highest reported power, all W (W) | Hotspot max (C) |
|---|---|---|---|---|---|---|
| A | stock | 418.7 | 398.2-418.7 (disturbed from W=4) | 3013-3196 | 328-347 | 84-90 |
| B | overclock | 421.4 | 421.4-421.5 | 3216-3218 | 305-319 | 85-86 |
| C | stock | 397.3 (disturbed) | 396.0-418.7 | 2890-3211 | 259-348 | 72-87 |
| S1 | stock | 418.9 | 415.4-418.9 | 3172-3200 | 337-347 | 89-90 |
| S2 | stock | 418.9 | 415.3-418.9 | 3178-3197 | 337-347 | 88-89 |
| S3 | stock | 417.1 | 413.6-417.1 | 3174-3185 | 341-351 | 90-91 |
| S4 | stock | 418.2 | 415.4-418.2 | 3178-3193 | 339-344 | 90-90 |
| S5 | stock | 418.5 | 416.2-418.5 | 3186-3196 | 337-342 | 88-89 |

Dual-issue FMA (8 waves per SIMD), streaming reads inside `mat_card`, and the streaming test:

| Round | Profile | FMA (G FMA/s) | FMA clock (MHz) | FMA highest reported power (W) | FMA hotspot max (C) | `mat_card stream` (GB/s) | Read, 3 runs (GB/s) | Copy, 3 runs (GB/s) |
|---|---|---|---|---|---|---|---|---|
| A | stock | 23,490 (disturbed) | 2946 | 339 | 82 | 636.0 | 632.9, 632.8, 632.9 | 549.4, 549.1, 549.1 |
| B | overclock | 24,877 | 3208 | 309 | 85 | 635.9 | 632.8, 632.7, 632.8 | 550.2, 550.8, 550.5 |
| C | stock | 24,893 | 3212 | 326 | 85 | 635.9 | 632.7, 632.7, 632.6 | 549.1, 549.3, 549.1 |
| S1 | stock | 24,845 | 3208 | 354 | 90 | 633.7 | 632.9, 632.8, 632.8 | 549.1, 548.8, 549.2 |
| S2 | stock | 24,745 | 3201 | 346 | 89 | 633.5 | 632.8, 632.8, 632.8 | 549.1, 549.2, 548.7 |
| S3 | stock | 24,792 | 3203 | 345 | 91 | 635.4 | 632.9, 632.9, 632.8 | 548.6, 549.4, 549.4 |
| S4 | stock | 24,800 | 3203 | 347 | 90 | 634.7 | 632.9, 632.9, 632.6 | 549.4, 549.6, 549.0 |
| S5 | stock | 24,837 | 3207 | 347 | 90 | 635.9 | 632.8, 632.9, 632.8 | 549.8, 549.8, 549.5 |

Two lines as printed (FP8, W = 8; stock round S1, then the overclock round):

```
RESULT mat_card_reg:fp8 W=8 waves=1024 iters=... ms=85.7 Ginstr_s=50.8 TFLOPS=415.7 ... ratio=0.999 ... clock_MHz=3175 sclk_MHz(min/med/max)=3060/3178/3220 power_W(min/max)=287/340 hotspot_C(min/max)=71/89
RESULT mat_card_reg:fp8 W=8 waves=1024 iters=... ms=90.0 Ginstr_s=51.5 TFLOPS=421.5 ... ratio=1.000 ... clock_MHz=3216 sclk_MHz(min/med/max)=3115/3217/3217 power_W(min/max)=263/319 hotspot_C(min/max)=69/86
```

What the rounds show:

- **Memory.** The requested 1409 MHz does not take effect: `pp_dpm_mclk` lists 1258 MHz as its top and active level
  on both profiles (READ), and the read rate is the same in all eight rounds (632.6 to 632.9 GB/s, 24 runs). The
  copy rate is 550.2 to 550.8 GB/s in the overclock round against 548.6 to 549.8 in 21 stock runs, 0.07 to 0.40 %
  higher and above every stock run. That is one round of three runs, and it is no larger than the spread between
  stock rounds of the other memory test (`mat_card stream`: 633.5 to 636.0, 0.4 %), so it is weak evidence at
  most; its cause is not known.
  This looks like a fault in how the setting is applied, not a limit of the card (open, the owner will look into
  it): the overdrive table accepts the request (`OD_MCLK` level 1 reads 1409 MHz on that profile) while the running
  level does not move; the owner recalls a higher memory clock after setting it by hand, which none of this
  study's records show (they start at the boot of 2026-11-17 08:45 UTC, and show 1258 MHz also in the morning on
  the owner's own profile); and READ from the `lactd` journal, every profile switch of the day (11:29 UTC by the
  owner, 18:24 UTC twice by the script) logs `could not reset the clocks table: io error: Invalid argument (os
  error 22)`. Whether that error is the cause is not known. The kernel log has the usual start-up note that the
  SMU driver interface (0x2e) is older than the firmware's (0x33).
  Tested by the owner at the machine on 2026-11-18 (MEASURED): (1) switching to the overclock profile in LACT's
  window (01:33 UTC; 374 W, +150 MHz, -50 mV and `OD_MCLK` 1409 MHz all read back): the active memory level was
  1258 MHz in every 50 ms sample during three read tests (best 630.5 to 632.0 GB/s; the medians, 583 to 588, were
  lowered by desktop use); (2) setting the memory clock by hand in LACT and applying (01:38 UTC; the request again
  reads 1409 MHz): active level 1258 MHz in every sample, read 632.7, 632.9, 632.8 GB/s, the stock figure. So the
  memory clock setting has no effect on this card with this kernel (7.3.4-201) and firmware (SMU 104.80.0), by
  profile or by hand. The by-hand apply logged no "could not reset the clocks table" error (that error comes only
  with profile switches), so the error is not what blocks it. READ from the system journal: the error is new since
  the card was put back in on 2026-11-17 (the RX 7900 XT was installed from 2026-11-14 to 2026-11-17); no boot with
  this card before that logged it, on the same kernel and unchanged firmware and LACT packages; LACT's settings
  now also hold an entry for the other card. The owner recalls the setting working before that swap; no record of
  the memory clock from then exists. Logs: `rdna3-fp8/build/lead-scratch/memclock/`.
  Then found (2026-11-18, kernel 7.3.4-202, same firmware; MEASURED): the request is applied once the displays
  are turned off and on. With the overclock profile active and the monitors on, the memory clock read 1258 MHz
  in all 9,000 samples of a 3-minute recording (it also never dropped below its top level at idle, about 6 %
  busy). The owner then switched the monitors off and on (03:07 UTC): from that moment `pp_dpm_mclk` shows no
  active level (1408 MHz is not one of its listed levels) and `gpu_metrics` reads current_uclk 1408 MHz (average
  2803, the double-rate figure). Three read tests: 702.7, 688.5, 687.3 GB/s (best 706.4) against 632.7 to 632.9 at
  1258 MHz, +9 to +11 % for a clock ratio of 1.119; copy 594 to 599 against 549 GB/s. Memory temperature rose from
  60 to 74 C over the three tests. INFERRED: the display engine holds the memory clock at its level while it
  scans out, so a new maximum takes effect only at a display mode change; an Arch Linux forum report of the same
  card and symptom says it applies "after quitting the session" or "turning off a monitor". Consequence for
  measurements: the memory clock must be read from `gpu_metrics` (current_uclk), not from `pp_dpm_mclk`, which
  only marks its listed levels. Log: `rdna3-fp8/build/lead-scratch/memclock/memclock-after-monitor-off.log`.
- **FP16 matrix and FMA: no difference.** FP16 at W=1: 210.7 to 210.9 in the six clean stock rounds against 210.1
  on the overclock profile, with the clock at 3218 to 3221 MHz on stock and 3213 to 3216 on the overclock profile;
  at the same W the overclock round is between 0.4 % lower and 0.1 % higher. FMA: 24,745 to 24,893 on stock (C and
  S1 to S5) against 24,877. The +150 MHz offset is not visible in any burst: no sample of `pp_dpm_sclk` in the
  overclock round is above 3219 MHz, and the stock rounds reach up to 3225.
- **FP8 matrix: 0.6 to 1.9 % faster on the overclock profile** (same W, against each of the five clean stock
  rounds). There all five settings give 421.4 to 421.5 TFLOPS at 3216 to 3218 MHz. The clean stock rounds give
  413.6 to 418.9 at 3172 to 3200 MHz. The rate follows the clock: instructions per cycle are the same (ratio to
  128 SIMDs x clock / 8 cycles: 0.994 to 1.000 in S1 to S5 and B).
- **Reported power is 6 to 15 % lower on the overclock profile for the same work.** Highest reading per setting,
  the overclock round against each clean stock round at the same W: FP16 228 to 243 W against 255 to 274 W, 9 to
  15 % lower at the same clock (3213 to 3216 against 3218 to 3221 MHz); FP8 305 to 319 W against 337 to 351 W, 6 to
  11 % lower at a clock 18 to 45 MHz higher. INFERRED: the -50 mV offset is the likely main cause (power falls
  roughly with the square of the voltage). Not separated from it: the overclock round ran cooler (hotspot maximum 71 to 74 C against
  76 to 81 C for FP16, 85 to 86 against 88 to 91 for FP8), which also lowers the power, and it is one round. The
  graphics voltage was not sampled in these runs; earlier records have it at different clocks only (1051 to 1056 mV
  at 3.18 GHz on stock in item 5's FP8 trace, 1117 to 1120 mV at 3.44 to 3.46 GHz on the overclock profile in the
  once-a-second log), so they do not settle it.
- **Why FP8 differs.** INFERRED from the power readings and from item 5, not shown directly (throttle bits were not
  sampled in these runs). On the stock profile the FP8 bursts peak at 337 to 351 W reported, above the 317 W cap,
  and item 5 shows the power limit (PPT0) acting on FP8 from 0.34 s of a long load and holding it at 3.18 GHz. In
  the first 2 s of that trace (40 samples of `indep_throttle_status` in `trace-fp8-sysfs.csv`) PPT0 is set in 33,
  bit 36, the hotspot throttler, in 15 (9 of them together with PPT0), and neither in 1 (item 5). So the sag in a
  burst may be the power limit, the hotspot throttler, or both. On the overclock profile the same bursts
  peak at 305 to 319 W and 85 to 86 C, and the clock does not sag. Those peaks are at or below even the stock cap
  of 317 W, so the lower voltage alone could account for it: these data cannot say whether the higher cap (374 W)
  plays any part. The +150 MHz offset shows no visible effect in these bursts; that it contributes nothing is not
  shown either. Also open: the overclock round is a single round, run within seconds of the profile switch.
- **Hotspot.** No job reached 95 C; the highest reading was 91 C (FP8 and FMA on stock, round S3) and 86 C on the
  overclock profile. Every job started below 60 C (highest reading before a job: 58 C).
- **Disturbed rows.** Round A from FP8 W=4 on and its FMA job, and several rows of round C, ran at a lower clock
  (2890 to 3111 MHz) and gave lower rates; S1 to S5, taken 5 to 17 minutes later on the same profile, did not. In
  the disturbed matrix rows the rate still follows the clock (ratio 0.999 to 1.045 against the clock of the same
  launches; the clock is a median over waves and launches, the rate comes from the median launch time). Two of round C's rows (FP16 W=1, FP8 W=1)
  were also short (55 and 40 ms instead of 85) because the probe sized them from a slower trial launch. Round C
  started within seconds of the switch back to the stock profile. The cause of the lower clock is not known.

What this run cannot show:

- **The effect of the +150 MHz offset on a steady load.** A burst of 0.1 s ends before the clock has finished
  climbing (item 5: a light load needs about 1.4 s to go from 3.29 to 3.42 GHz on the stock profile). READ from
  earlier records: the same probe (`core_counter`, one busy wave, three 1-second runs) ran on both profiles. On the
  overclock profile (`runs/20261117T110801Z-core`, 11:08 UTC; `notes/STUDY-PLAN.md` says the card was on that
  profile until 11:30 UTC) the counter gave 3399, 3484 and 3570 MHz; the log lists the `pp_dpm_sclk` readings of
  each run by value, not in time order, and in the third run they span 3482 to 3589 with 3584 the most frequent (6
  of 20 readings). On the stock profile (`runs/20261117T133209Z-core`, 13:32 UTC) it gave 3329, 3408 and 3430 MHz,
  with 3430 the most frequent reading of the third run (13 of 20). The once-a-second log, on the overclock
  profile, shows a full load at 10:43 UTC (100 % busy from 10:43:36 to 10:43:46 and again from 10:43:48; which load is
  not recorded); in the first stretch, from 10:43:40 to 10:43:45, 3440 to 3464 MHz, 290 to 313 W reported and a hotspot
  of 88 to 92 C (`clock-log-20261117T104212Z.csv`).
  INFERRED: the offset raises the clock of a light load by 70 to 140 MHz as averaged over these 1-second runs, and
  by about the 150 MHz it asks for where the clock dwells longest (3584 against 3430). How much rate that gives a
  whole-card matrix load was not measured.
- Power, temperature and throttling over seconds on the overclock profile: the notes record 102 C under a
  sustained load on that profile before the fan curve was changed (`notes/STUDY-PLAN.md`).

Card state: the script switched the profile with `lact cli profile set`, and back. After the comparison run the
card was checked by hand, and every self-test run below ends with a line read from the real `lact` and sysfs:
profile "Testing", cap 317 W, clock offset 0 MHz, voltage offset 0 mV.

Checks of the script. The measured rounds A, B, C, S1 and S2 were taken with its first form (commits 939d40d and
78739d8). Three reviews then asked for stricter guards and tests, and the script was reworked. Rounds S3 to S5
come from hand-run tests of the first rework (runs `183855Z` to `184321Z` of the same day; their exit codes were
not stored). The final form is tested by `scripts/study-overclock-selftest.sh`: each test runs the script with a
stand-in for `lact` first in `PATH` that records every call and never talks to the daemon, so no card setting
changes; t13 uses the real `lact`, which the stock form only reads. Run `20261117T194003Z-overclock-selftest`
(`results.txt`; 472 s; the two scripts had the same checksums before and after as the committed files); every row
is MEASURED there, with the stand-in's call record and the script's output stored per test:

| Test | Situation | Expected and seen |
|---|---|---|
| t01 | stock form, hotspot sensor unreadable | exit 1, "hotspot unreadable", 0 jobs, 0 `profile set` calls |
| t02 | stock form, profile reads "Default" after the start check | exit 4, 0 jobs, 0 `profile set` calls |
| t03 | comparison form, profile is "Default" at the start | exit 3, 0 jobs, 0 `profile set` calls |
| t04 | comparison form, sensors unreadable, the switch back always fails | exit 5, "RESTORE FAILED", 2 tries, 0 jobs |
| t05 | comparison form, sensors unreadable, the switch back works | exit 1, "restored", 1 try, 0 jobs |
| t06 | comparison form, the first switch back fails, the second works | exit 1, "restored", 2 tries, 0 jobs |
| t07 | stock form, hotspot stays at 70 C | gives up after 180 s, exit 1, 0 jobs |
| t08 | stock form, the shared lock is held by another process for 75 s | gives up after 60 s, exit 1, 0 jobs |
| t09 | comparison form, TERM while it waits for the lock (held 40 s) | exit 130 after 45 s, "restored", 0 jobs |
| t10 | comparison form, HUP while it waits for the lock (held 40 s) | exit 129 after 45 s, "restored", 0 jobs |
| t11 | comparison form, TERM while the first GPU job runs | the job finishes (its VERDICT line is there), then exit 130, "restored", no further job |
| t12 | comparison form, the switch to "Default" does not take | round A runs (7 jobs), the lock is held during the switch (another `flock -n` fails), stop, "restored", exit 1, no round B or C |
| t13 | stock form with the real `lact`, normal | exit 0, 14 jobs, each exit 0 |
| t14 | comparison form, whole flow: the stand-in's switch changes only a file that stands for the power cap, so all three rounds run while the card stays on "Testing" | exit 0, 21 jobs, each exit 0 (the overclock round's 7 run with the lock already held); lock held during the switch and during the switch back, free at the end; one `set Default` and one `set Testing` call |

Deliberate breaks of the self-test (a changed copy of the script, removed afterwards), each must FAIL and does:
the stock form allowed to switch again: t02 FAILS (1 `profile set` call; run `194757Z`); the lock not kept over
the overclock round: t12 FAILS (lock free during the switch; `194806Z`); the lock given up before the switch
back: t14 FAILS (lock free during the switch back; `194852Z`). A run that names no existing test fails too.
Earlier self-test runs of the day (`185815Z` to `193903Z`) are from its development. In one of them (`185835Z`)
t09, t10 and t11 were reported failed because the self-test had signalled a wrapper process instead of the
script; the three script copies ran on to their normal end with the stand-in, so no card setting changed.

Not tested: the comparison form end to end with the real `lact` after the rework (it needs a new OK from the
owner; t14 covers its flow with the stand-in); the memory check (below 8 GB, or unreadable); the second check with
the lock held failing three times; a reading of 95 C or more after a job; the 240 s limit of a job; a `lact` call
that hangs until its 20 s limit; the lock file missing or unreadable; INT; the lock not being free before the
switch to the overclock profile; a signal during the switch to the overclock profile or during the switch back
(the latter is ignored by design: a signal that arrives then is lost); exit 4 (the card not on "Testing" at the
end although the switch back reported success). If the switch back fails (exit 5), the shared lock is given up
when the script ends although the card may still be on the overclock profile; the script cannot prevent that,
and its summary then says RESTORE FAILED.

## 6. Vulkan queues (`vksched`, helper; rerun here)

RADV exposes queue family 0 (graphics, compute, transfer; 1 queue) and family 1 (compute and transfer, 4 queues).
The same dispatch (one 64-thread group, one store per invocation) is pre-recorded for one queue of each, submitted
with a fence and waited on, interleaved graphics then compute (order alternating), 2,000 samples per queue (600 for
the 10 ms pause) after 50 warm-ups. Wall = `vkQueueSubmit` to fence signalled; GPU = timestamps around the
dispatch. MEASURED; the table is from run `20261117T165251Z-sched`, in which the desktop was drawing heavily. The
other runs agree in shape with different figures: the compute queue alone has a median of 34 to 48 us with 30 to
42 % of submissions in the slow peak (35.3 % in the table; 36.0 %, 42.4 % and 30.0 % in `164432Z`, `165519Z` and
the lead's `171514Z`); in `165519Z` the no-pause medians are 31.0 (graphics) and 93.5 us (compute).

| Mode | Queue | Wall us median | p10 | p90 | p99 | Share above 2 x median (its median) | GPU us median / p90 |
|---|---|---|---|---|---|---|---|
| no pause | graphics | 29.8 | 20.3 | 38.9 | 430 | 4.4 % (108 us) | 0.28 / 0.32 |
| no pause | compute | 87.2 | 29.9 | 104.6 | 192 | 1.1 % (994 us) | 0.56 / 0.92 |
| 1 ms pause | graphics | 65.6 | 41.2 | 77.7 | 89 | 0.05 % | - |
| 1 ms pause | compute | 115.9 | 67.6 | 141.2 | 153 | 0 % | - |
| 10 ms pause | graphics | 78.2 | 52.4 | 481 | 1627 | 15.7 % (653 us) | - |
| 10 ms pause | compute | 112.0 | 58.4 | 345 | 1379 | 15.3 % (638 us) | - |
| alone (compute queue only, then graphics only) | graphics | 32.1 | 24.8 | 35.8 | 58 | 1.0 % | 0.28 / 0.32 |
| alone | compute | 34.3 | 30.5 | 103.8 | 115 | 35.3 % (103 us) | 0.52 / 0.96 |

Histogram of the compute queue, no pause (wall, log bins): 25-40 us: 237, 40-63: 375, 63-100: 423, 100-158: 957;
graphics: 25-40 us: 1,724 of 2,000 (run `20261117T164432Z-sched`).

- Yes, there is a counterpart. The compute-only queue's times fall into two groups, one near 30 us and one near
  100 us; when it runs alone, 30 to 42 % of its submissions take the slow path (median about 104 us against 34 to
  48 us). Compute was measured before graphics in the "alone" mode. The graphics
  queue has a single peak near 30 us. The extra delay is about 70 us, smaller than the 0.24 ms the neighbouring
  project saw on RDNA 3, and in the same direction: the compute queue is the slow one. The dispatch itself also
  takes longer on the compute queue (0.52 to 0.56 us against 0.28 us GPU time, p90 0.9 against 0.3 us).
- With a 1 ms pause both queues slow down (66 / 116 us). The long tail at a 10 ms pause in this table (15 % above
  twice the median, around 0.65 ms) appears only in this run, with heavy desktop drawing; the runs `165519Z` and
  `171514Z` show 0.3 to 2.7 % there. So it is desktop disturbance in this run, not a property of the pause.
- INFERRED, not tested: on RADV, using the graphics queue for latency-sensitive small dispatches avoids the slow
  path.

## 7. Sharing the card (`sched_share`, `scripts/sched_pair.py`)

(a) With the desktop. One wave runs steps of 256 independent FMAs (480 cycles alone at 3.36 GHz) for 4 s and records
cycles, REALTIME and `HW_ID1` after every step (28 million steps); a host thread reads `gpu_busy_percent` and every
other process's `drm-engine-gfx` time every 20 ms. MEASURED, run `20261117T165519Z-sched` unless noted.

| Quantity | Value |
|---|---|
| Disturbed events | 1,561 in 4.0 s (390 per s); 0.046 % of the time lost |
| Event length | median 1.12 us (p10 1.04, p90 1.20, max 9.8); extra cycles median 3,902 (p10 3,496, p90 4,109) |
| Spacing | median 2,773 us (p10 1,545, p90 2,890): a regular stall about every 2.8 ms in this run; the rate differs between runs (390 per s here, 476 per s in the pair run, 1,023 per s in the run with heavy drawing) |
| Wave moved during a stall (`HW_ID1` changed) | never |
| Counter during the stall | kept counting: extra cycles equal the lost time x the clock within 10 % in 1,516 of 1,561 events |
| Lost time per 20 ms window, quiet / other programs drawing | 8.8 us / 18.1 us; only 9 of the 200 windows had other programs drawing (1.05 ms of their graphics time in all), so this is weak evidence (correlation 0.46, computed by hand from `spin-windows.csv`; the script prints only the correlation with `gpu_busy_percent`). Another run with more drawing (`20261117T165251Z-sched`): 11.0 / 39.3 us, correlation 0.84, 1,023 events per s, 0.11 % lost |
| `gpu_busy_percent` | 100 % throughout (the spinning wave itself counts as busy); it does not reveal desktop load |
| Two waves in two processes (`pair small small`, different WGPs) | 476 and 478 events by the in-kernel detector; the coincidence script counts row clusters instead (1,246 and 1,282, because the two or three rows of one event lie more than its 20-tick merge gap apart), and 98.8 % of A's clusters are within 1 us of one of B's. INFERRED: with events this sparse a chance hit within 1 us is about 0.3 %; a shifted-trace control was not run |
| 16 waves at once, all 4 shader engines (`multi`) | share of a wave's events within 1 us of another wave's event: same shader engine 96.2 % (median of 48 pairs), different shader engines 95.6 % (192 pairs) |

- So the regular stall (about 1.1 us every 2.8 ms) stops every SIMD of the card at once. INFERRED: a card-level
  event (firmware or scheduler activity), not desktop drawing on that SIMD. Desktop drawing adds more short events
  on top, in proportion to how much the other programs draw.
- What an interruption does to a cycle-counter timing: the counter keeps counting, the wave stays on its SIMD, so the
  timed region simply includes about 3,900 extra cycles (up to 34,000 seen). A region of 100,000 cycles (30 us)
  meets the regular stall with a chance of about 1 % per repetition, and more while the desktop draws. That fits
  lane 6's disturbed repeats (a dependent chain reading 5.101 cycles per instruction instead of 5.000, with single
  samples up to 12.5) and why medians held; whether the size matches needs the chain length, which the timing
  leftovers check.
- Rule for detecting a disturbed sample from inside the kernel: time the work in chunks of fixed size (a few hundred
  to a few thousand cycles), and flag a chunk whose cycle count exceeds 1.5 x the median chunk + 64 cycles. The
  rule has a size limit: a stall adds about 3,900 cycles (p10 3,496), so a chunk longer than about 7,000 cycles
  does not cross the threshold and the stall goes unflagged; for longer regions compare against the median plus
  about 3,000 cycles instead, or split the region. A
  change of `HW_ID1` also flags it. REALTIME growing without extra cycles is a clock change, which does not disturb
  a cycle count. The detector is checked three ways in every run: a clean synthetic trace gives 0 events, the same
  trace with an injected 50 us gap gives exactly 1 event of 50.0 us, and so does a clean 20,000-step stretch of the
  real trace with the gap injected.

(b) Two compute processes of my own, started together inside one locked job (REALTIME is chip-wide, so the two
processes' records share one time base). Each "big" launch: 64 groups of 512 threads with 40 KiB of unused LDS (one
group per CU, every CU taken), 2.1 ms alone; "part": the same on 16 groups; "small": the spin wave of (a).

| Pair | Result (MEASURED, run `20261117T165519Z-sched`) |
|---|---|
| big alone | 2.20 ms per launch (276 launches in 0.6 s) |
| big + big | each process 4.4 ms per launch (half the throughput); the two were never on the card together (both-on-card fraction 0.000, 136 + 140 launches); launches alternate A, B, A, B, each a single launch (runs of 2.11 ms); gap at a switch median 27 us (p10 18, p90 293) |
| part + part (16 CUs each) | both on the card 97 % of the time, never on the same CU; 2.22 ms per launch each, the same as alone |
| small + big | the spin wave's cycles per step inside B's launches 480.03 (p90 481.2) against 480.00 outside; but at the start it shared its SIMD with B's older waves and made about one step per 210 us for the first 1.86 ms (one B launch), then ran normally. Lost 0.52 % of 0.6 s (event max 1,718 us); 2,015 us in an earlier run |
| small + small | 0.07 % lost each; their events coincide (above) |

- The card does not time-slice two compute processes on this scale: it runs whatever fits. Two whole-card kernels
  alternate at launch boundaries (each launch runs to the end), and two that fit side by side run at once at full
  speed. No preemption was seen within 0.6 s (INFERRED from the uninterrupted 2.1 ms runs; longer kernels not
  tested).
- Another process disturbs a single-wave cycle timing only through its waves on the same SIMD. Those take the
  vector unit when they are older (lane 1's oldest-first rule holds across processes) and can then hold the timed
  wave for a whole launch of theirs.

## 8. What a kernel can read about itself (`sched_place ids`)

One launch of 4096 groups of 1024 threads (131,072 waves), WGP mode. MEASURED.

| Field | Bits | Distinct values on this card |
|---|---|---|
| Wave slot | `HW_ID1` 4:0 | 16 (0-15) |
| SIMD | `HW_ID1` 9:8 (bit 0 = CU within the WGP, bit 1 = SIMD within the CU) | 4 |
| WGP | `HW_ID1` 13:10 | 4 (0-3) in every shader array |
| Shader array | `HW_ID1` 16 | 2 |
| Shader engine | `HW_ID1` 19:18 | 4, equal to the `GET_SE_AID_ID` message (bits 3:0) in all 131,072 waves |
| Constant bits | `HW_ID1` 7:5 = 0, 15:14 = 0, bit 29 = 1, other high bits 0; `GET_SE_AID_ID` bits 31:4 = 0 | 1 |
| SE.SA.WGP / .CU / .SIMD combinations | - | 32 / 64 / 128 |

- The card shows 4 shader engines x 2 shader arrays x 4 WGPs x 2 CUs x 2 SIMDs = 128 SIMDs and 64 CUs, the driver's
  topology (`simd_count 128`, `array_count 8`, `cu_per_simd_array 8`; verdict `topology`).
- Correction to the study's helper (`mem_common.h`, which says RDNA 4 dropped SE_ID from `HW_ID1`): on this card
  bits 19:18 hold the shader engine, so one `s_getreg` is enough; the message gives the same value.

## Checks and deliberate breaks

Every probe prints `VERDICT` lines and exits non-zero on a FAIL. Known answers reproduced in run
`20261117T165519Z-sched`: one independent FMA 0.9996 cycles (calibration; 1.000 in the helpers' probes); REALTIME
99.970 MHz against host time; the counter's clock within 0.03 % of `pp_dpm_sclk` under steady load (5 traces);
whole-card FMA 25.11 TFMA/s against lane 4's 24.7; FP16 213.3 and FP8 416.4 TFLOPS in the first 1.5 s against
lane 4's 211.4 and 419.5, with all 128 SIMDs placed; stream 638 GB/s against 633; WMMA 16.000 cycles in both wave
sizes; placement checked from the records in every launch.

Deliberate breaks, each run by the script and each failing as intended (`summary.md` of the run):

```
- break-calib: FAILED as intended: VERDICT sched_launch calib_fma: FAIL cycles per independent v_fma_f32 = 0.9996 (expected 2.00 +- 0.02)
- break-placement: FAILED as intended: VERDICT sched_place placement16: FAIL 16 groups of one wave: 16 distinct SIMDs, expected 32
- break-detector-clean: FAILED as intended: VERDICT sched_share detector_clean: FAIL clean synthetic trace (200000 steps): 0 events, expected 1
- break-detector-injected: FAILED as intended: VERDICT sched_share detector_injected: FAIL same trace with one 50 us gap injected: 1 events (50.0 us), expected 0
- break-barrier-calib: FAILED as intended: VERDICT sched_barrier calib_fma: FAIL ... 1.000 cycles per FMA (expected 2.00 +- 0.02 ...)
- break-wave-wmma: FAILED as intended: VERDICT sched_wave wmma_rate: FAIL wave32 v_wmma_f32_16x16x16_f16: 16.000 cycles per instruction ... (expected 8.0 +- 3 %)
- break-wave-rows-wgp-build: FAILED as intended: VERDICT sched_wave fma_rows: FAIL wave32: every multi-wave FMA row reaches 90 % of its one-wave rate: NO
- break-vksched: FAILED as intended: VERDICT vksched break_compute_wall_under_1us: FAIL deliberate break: compute wall median 91.97 us claimed under 1 us
```

The detector's positive controls pass in every run (clean synthetic 0 events, injected 1 event of 50.0 us, clean
real stretch 0, with injection 1 of 50.0 us).

## Method

- Probes (all under `src/study/`, shared code in `sched_common.h`): `sched_launch` (item 1), `sched_place` built in
  WGP mode and in CU mode (items 2 and 8), `sched_barrier` built in both modes (item 3, helper), `sched_wave` built
  for wave32 and wave64, both in CU mode (item 4, helper), `sched_trace` in CU mode (item 5), `sched_share` in CU mode
  (item 7) with `scripts/sched_pair.py`; `src/vk/vksched.c` with `src/vk/shaders/vksched_one.comp` (item 6, helper).
  `scripts/sched_tables.py` prints the tables of this file from a run folder.
- Time sources inside a kernel: the wave's 64-bit shader cycle counter (`SHADER_CYCLES_HI/LO`, per SIMD, runs at the
  shader clock) and REALTIME (`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, chip-wide, 100 MHz). Placement from
  `HW_REG_HW_ID1` and `MSG_RTN_GET_SE_AID_ID`, decoded as in the study's `mem::placement`. On the host:
  `std::chrono::steady_clock` and HIP events.
- Calibration before trusting a cycle figure: 16 independent `v_fma_f32` per block, U = 4 and 8 blocks per loop
  iteration, (cycles(8) - cycles(4)) / (4 x iterations x 16) must be 1.00 +- 0.02 (lane 1). REALTIME against host
  time within 0.5 % of 100 MHz. Whole-card FMA within 5 % of lane 4's 24.7 TFMA/s; FP16 and FP8 matrix within 5 %
  of lane 4 over the first 1.5 s; the counter's clock within 1 % of `pp_dpm_sclk` over the same window.
- Concurrency per SIMD, CU or WGP: the largest number of start-end REALTIME intervals open at once. Every wave of the
  placement kernel waits 64,000 cycles on its own cycle counter. A first version polled REALTIME instead; with many
  waves doing that, other waves' start stamps were delayed by whole wait lengths (groups seemed to start 20, 40,
  60 us late). So: never poll REALTIME from many waves; read it once at each end.
- Sysfs, read-only, every 50 ms from a host thread (5 ms in the ramp): `pp_dpm_sclk` and `pp_dpm_mclk` ('*' line),
  `gpu_busy_percent`, hwmon `power1_average` (PPT; scaled down by the mod), `temp1_input` edge, `temp2_input`
  junction (hotspot), `temp3_input` memory, `fan1_input`, and `gpu_metrics` (format 1, content 3, 120 bytes:
  `throttle_status` at byte 68, `indep_throttle_status` at 112, graphics voltage at 106; layout checked against
  hwmon power and `pp_dpm_sclk` on this card). Per-process graphics time from `/proc/<pid>/fdinfo/*`
  (`drm-engine-gfx`, one fd per `drm-client-id`, for `drm-pdev: 0000:03:00.0`).
- Thermal rule: traces start below 55 C (60 C for the light load and the other whole-card jobs); an in-process
  sampler sets a stop flag at 95 C that ends the load between launches of about 50 ms (and ends the process at
  100 C as a last resort, never reached). Traces: until 95 C or 20 s, once per load type per run, at least 60 s and
  a hotspot below 55 C between them, 30 s of cool-down sampled after.
- Disturbance detector (item 7): see the rule in item 7(a).
- Two processes (item 7b): the script starts both in one locked job, with the same wall-clock start time;
  `sched_pair.py` merges their files on REALTIME.

## How to rerun

From the lane folder (after the copy into the study repository, `STUDY_REPO` defaults to the repository root):

```
export ROCM_PATH=$HOME/.local/opt/rocm
STUDY_REPO=<checkout of this repository> scripts/study-scheduling.sh all   # about 15 minutes, 6 traces
scripts/study-scheduling.sh build | run | traces | breaks                        # one phase
SCHED_RUN_DIR=<folder> scripts/study-scheduling.sh <phase>                       # several phases into one folder
SCHED_TRACES="idle light stream f16 fp8" scripts/study-scheduling.sh traces      # leave out a load that hit 95 C
python3 scripts/sched_tables.py runs/<folder>                                    # the tables of this file
ROUNDS=stock scripts/study-overclock.sh     # item 5a, two stock rounds; never switches the profile; under a minute
scripts/study-overclock-selftest.sh         # item 5a, the script's guards with a stand-in for lact; 8 minutes
scripts/study-overclock.sh                  # item 5a, stock / overclock / stock; SWITCHES THE LACT PROFILE and back:
                                            # needs the owner's OK each time (the 2026-11-17 OK was for one run)
```

Results land in `runs/<UTC time>-sched/`: one log per job, `config.txt` (read-only configuration record),
`summary.md` (every VERDICT line), trace CSVs (`trace-<load>-sysfs.csv`, `trace-<load>-launch.csv`), the spin
trace's `spin-events.csv` and `spin-windows.csv`, the pair files, and the Vulkan samples under `vk/`.

## Not measured

Still open:

- How well the detection rule works on real traces: how many disturbed samples it catches and how many clean ones
  it flags. It is validated only by an injected 50 us gap and a clean stretch (0 events). What is open now: the ground
  truth of a real trace's small events.
  Answered in part (leftovers C, SC-1, `measurements/leftovers-c-a.md`): with injected sleeps the rule behaves as its
  threshold says: every gap of 420 cycles or more was found, none of 261 or less, and a 326-cycle gap, 16 below the
  limit, in 27 % of its steps. So for a region of C cycles it misses any disturbance smaller than about 0.5 x C + 64
  cycles. Outside the injected steps it flags about 10.4 per 100,000 steps, consistent with the regular stall (about 6.5
  per 100,000); 3.2 per 100,000 have an excess below 1,500 cycles and are not the regular stall (desktop drawing,
  INFERRED). MEASURED.
- Why the hotspot throttler (throttle bit 36) is already active at hotspot readings from 57 to 92 C.
  Answered in part (leftovers C, SC-3 and SC-5, `measurements/leftovers-c-b.md`): in the kept traces the bit is set in
  23 samples, all within 0.8 s of the start of the FP16 and FP8 loads, none in the stream, light and idle traces. In
  four short bursts (FMA twice, FP8, FP16) the only throttle bit is 36 (`TEMP_HOTSPOT`), first set about 50 ms into
  the load (0.050 to 0.061 s) at a hotspot of 66 to 67 C; PPT0 does not appear within 0.1 s. A fast rise separates the
  bit well (AUC 0.946), time since the load started separates it better, and a threshold near 66 C fits the bursts too:
  the data do not separate "time since the load started", "rate of rise" and "a threshold". MEASURED (association
  only); the cause is open.
- Sustained loads beyond 20 s, and whether the clock steps down later: forbidden by the thermal rule. Needs the owner
  (leftovers C plan, SC-6: thermal rule).
- What the power profiles and performance levels would do if changed: out of scope (read-only), except the one
  short-burst comparison of the owner's overclock profile in item 5a. Sustained loads on that profile, its steady
  clock under a whole-card load, and its throttle bits were not measured. Needs the owner (leftovers C plan, SC-7:
  profile switch).
- Item 5a: the memory overclock applies only after the displays are switched off and on (found 2026-11-18). Not
  measured: whether a fresh login or a resolution change does the same, whether the memory clock falls back to
  1258 MHz later, and what the higher memory clock does for the matrix and upscaler loads of this study.
- Item 5a: whether the 0.07 to 0.40 % higher copy rate in the overclock round is real. Needs the owner (leftovers C
  plan, SC-8a: it needs the profile switch).
- Item 5a: the cause of the lower clock in the disturbed rows of rounds A and C. Low value (leftovers C plan, SC-8b):
  desktop activity cannot be separated afterwards.
- The cause of the card-wide stall every 2.8 ms (firmware, MES, power management): the probes see it, nothing
  read-only names it. Not reachable (leftovers C plan, SC-9).
- Preemption between processes for kernels longer than about 2 ms, and compute-queue priority (READ: disabled in
  the GC 12 queue manager since 2026-02): busy waves instead of sleeping ones.
  Answered in part (leftovers C, SC-11, `measurements/leftovers-c-b.md`): a first kernel whose waves hold every SIMD
  while sleeping (`s_sleep`) was not preempted by a second kernel, from another process or from a higher-priority
  stream: the second kernel started after the first ended in 64 of 64 trials per setting, for kernels of 4 to 180 ms,
  and stream priority changed nothing; no gap above 4.08 us in the first kernel's waves. MEASURED, for that state only.
  Open: a positive control (a second kernel that fits beside the first and is seen to start early; the overlap
  detection has never been seen to fire in that lane), and a first kernel of busy waves.
- The Vulkan compute queue's slow path: not traced further than the timing (which RADV or firmware step adds the
  70 us is unknown). The exact RADV or firmware step is not reachable (leftovers C plan, SC-12).
  Answered in part (leftovers C, SC-12, `measurements/leftovers-c-b.md`): with a second thread keeping the graphics
  queue submitting tiny dispatches, the median fell from 31.1 to 18.8 us and the share of slow submissions from 0.253
  to 0.115 (counted above twice the alone run's median; 0.198 against the run's own median; the lead's rerun gave 0.303
  to 0.207, and 0.299 against its own median). The slow path is not removed. MEASURED for this weak keep-alive;
  whether a continuously busy card removes it is not settled.
- True board power: the power-limit mod scales the reading. Not reachable (leftovers C plan, SC-13).
- New from leftovers C: what sets the rate of the 2.8 ms stall: the lane's runs saw 380 to 420 events per second and the
  lead's rerun, 4 minutes later, 520 to 560 (earlier item 7: 390 to 1,023), with the same loss per event; the cause was
  not tested (SC-2, `measurements/leftovers-c-a.md`).
- New from leftovers C: why the stall costs 24 to 27 % more cycles per event when the wave stores nothing per step (SC-2,
  `measurements/leftovers-c-a.md`).
- New from leftovers C: the rule behind the wave64 register-bank costs: blocks in which every instruction reads two
  registers of one bank cost 1.0 to 2.5 cycles per instruction depending on what the neighbouring instructions read
  (SC-10, T1, `measurements/leftovers-c-a.md`); and the register layout of Vulkan's 1.06 cycles per wave64 FMAC, whose
  listing was not re-read.

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, SC-2, `measurements/leftovers-c-a.md`): the stall without the REALTIME read: it is there at the
  same rate per second (R, with REALTIME every step: 392 events per second; N, a store but no REALTIME: 389 to 396; Q,
  neither: 382 to 386) and, with a store per step, the same size (N against R extra cycles 1.00 to 1.02 in three runs).
  So the REALTIME message does not cause it. Without a store per step the stall loses more per event (1.24 to 1.27
  times), why not established. That it hits the whole card rests on item 7; this job used one wave. MEASURED.
- Answered (leftovers C, SC-4, `measurements/leftovers-c-b.md`): the shifted-trace control: the stalls of two processes
  coincide in 98.8 % of events when the traces are aligned, and in 0.000 to 0.004 when one is shifted by 5 to 100 ms
  (chance 0.0026). So the coincidence is not an artefact of the event rate. MEASURED (offline).
- Answered (leftovers C, SC-5, `measurements/leftovers-c-b.md`): throttle bits in the first moments of a load: FMA
  (0.1 s, two bursts), FP8 and FP16 bursts set only bit 36 from about 50 ms on, at 66 to 67 C; the long FMA trace
  itself was not repeated. See the throttle bullet above for the open cause. MEASURED.
- Answered (leftovers C, SC-10, `measurements/leftovers-c-a.md`): why wave64 FMAs issued at 0.81 per cycle: the register
  layout. Half of that probe's FMAs read two registers of one bank (measured cause); the same kernel with only its
  registers moved, so that none does, ran at exactly 1.000 (64.00 lane-FMAs per cycle against 51.85, i.e. 1.234 cycles
  per FMA). In wave64, 32-bit float add, multiply, FMA, FMAC and `v_mov_b32` take one cycle for all 64 lanes; integer
  add, packed FP16, conversions, transcendentals and FP64 take twice their wave32 cycles. MEASURED.

## Helpers

- Claude Sonnet 5.5 (medium effort), two sealed briefs. (1) `src/vk/vksched.c` and `src/vk/shaders/vksched_one.comp`
  (item 6). (2) `src/study/sched_barrier.hip` and `src/study/sched_wave.hip` (items 3 and 4). The lead read their
  code and logs, rebuilt both with the lane script, and reran them in three script runs; every number above comes
  from those reruns. The lead's change: the wave probe's CU-mode build flags in the script, and the new `fma_rows`
  check with its deliberate break (item 4).
- Everything else (the shared header, `sched_launch`, `sched_place`, `sched_trace`, `sched_share`, the scripts and
  this file): the lead.
