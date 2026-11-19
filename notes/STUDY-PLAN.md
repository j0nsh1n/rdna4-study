# RDNA 4 hardware study: plan

Owner request (2026-11-17): understand the actual hardware of the RDNA 4 architecture at the depth of the RDNA 3
measurements, by measuring it. Card: RX 9070 XT (gfx1201), ROCm 10.1.0, kernel 7.3.4. This file is the lead's plan;
each lane below is one sealed piece of work with its own measurement file under `measurements/`.

## Rules for every lane

- A probe is trusted only after it reproduces a known answer: a value from AMD's guide or calculator, a result in
  `notes/first-hour-9070xt-rocm10.md`, or a CPU model. Each probe prints `VERDICT` lines and has a way to fail.
- Cycle-counter measurements (one wave alone on its SIMD, `src/harness.h`) do not depend on the clock frequency and
  are preferred. Wall-clock measurements log `pp_dpm_sclk` and `pp_dpm_mclk` before and after and use buffers above
  the cache trap (256 MB) unless the cache itself is the subject.
- Every GPU job: the shared lock, the device index from `card.env`, `nice`, memory floor 8 GB. One GPU job at a time.
- Check the generated assembly (`-save-temps`, `rdna3-fp8/examples/tools/loop_isa.py`) so the loop contains what
  was meant. Compiler: ROCm 10.1 `hipcc` (clang 24), `ROCM_PATH=$HOME/.local/opt/rocm`.
- Results: MEASURED (seen in a run here), READ (from a document), INFERRED (reasoning). Give the spread.
- Write each measurement file with a plain summary first, then tables, then method and how to rerun.
- Disturbed timings (timing leftovers item 1, `tim_disturb`, `measurements/execution-core.md` item 9): every SIMD
  of the card stops for about 1.1 us (about 3,000 to 4,000 cycles at 3.2 GHz) roughly every 2.8 ms, and desktop
  drawing adds longer events. A timed region of C cycles meets the regular stall with a chance of about
  C / 9,000,000 at 3.2 GHz (1 % at 90,000 cycles, 12 % at 1,100,000). The lane 7 rule (drop a sample above
  1.5 x median + 64 cycles) sees a single stall only in regions shorter than about 7,500 cycles, so keep a region
  that is judged sample by sample below about 5,000 cycles. For longer regions quote the median of at least 5
  samples (it never moved here: the medians with and without the rule are equal), or the best sample for a
  minimum; never the mean or the worst, and never base a verdict on the extreme samples of regions above about
  50,000 cycles. The clock climb from 2.79 to 3.24 GHz after idle changes the clock, not the cycles per instruction, so it does
  not disturb a cycle count (the first climb from the idle clock was not tested). Keep a timed loop body below 4 KB of code: above it the loop itself costs more.

## Lanes

| Lane | Subject | Questions | Output |
|---|---|---|---|
| 1 | Execution core | Cycles per instruction class when independent and when chained (latency against throughput): float, integer, 16-bit packed, conversions, transcendental, 64-bit, compares, lane permutes. Which dual-issue pairs really run two per cycle. Scalar unit speed and the scalar-to-vector hazard. How several waves share one SIMD (1 to 16 waves). Branch and loop overhead. | `measurements/execution-core.md` |
| 2 | Memory system, global | Bandwidth and latency against buffer size (find every cache level and its size). Load width (32 to 128 bits per lane). Loads in flight per wave before waiting. Waves per SIMD and register count against occupancy (the 24-register blocks, the 96-register line). Cache hints (regular, non-temporal, last use, and the mixed ones if reachable). Software prefetch (is it enabled under ROCm). Alignment to 128-byte lines. Store and copy. Whether one wave's slow loads delay another wave. | `measurements/memory-system.md` |
| 3 | Shared memory, scalar loads, special loads | LDS bandwidth, latency, bank conflicts, load widths. Scalar memory loads. Block register loads and stores. The transposing matrix loads (layout and speed). Dynamic register allocation (`S_ALLOC_VGPR`) under ROCm. | `measurements/lds-and-special-loads.md` |
| 4 | Matrix accelerator, timing | Every dense and sparse form (FP16 and BF16 accumulators, 4-bit, K = 32, sparse K = 32 and 64). Cost of the documented hazards and stalls. Chained against independent accumulators. Several waves on one SIMD, one compute unit, the whole card (measured TFLOPS against the theoretical rate). Inline-constant accumulator. | `measurements/matrix-timing.md` |
| 5 | Matrix accelerator, arithmetic | What each form computes bit for bit: product exactness, order and grouping of additions, rounding points, fused or not; size of the differences from an exact sum; denormals, NaN, infinity, signed zero; the negate and absolute-value modifiers; effect of the MODE rounding and denormal bits; FP8, BF8 and mixed forms; sparse forms. Then how often the differences change an FP8 or FP16 result after rounding. | `measurements/matrix-arithmetic.md`, `measurements/errors-after-quantizing.md` |
| 6 | Conversions, dot products, small formats | All FP8, BF8, FP16, BF16 conversions over all inputs (rounding, overflow mode, NaN, stochastic rounding). `V_DOT4` FP8/BF8 and `V_DOT2` FP16/BF16: exactness, order, cost. Packed 16-bit math. Cost per weight of each way to widen FP8. | `measurements/conversions-and-dots.md` |
| 7 | Scheduling, launch, clocks, power | Launch and dispatch overhead. Scaling with compute units and work-group size. Wave32 against wave64. Barriers. Core clock against load over time, cycle counter against wall clock, power draw and temperature (read-only). What the power profile and performance level do, if the owner allows changing them. | `measurements/scheduling-and-clocks.md` |
| 8 | Compiler and hazards on gfx12 | The masked-lane matrix problem from RDNA 3, retested. Which hazards clang 24 inserts and what they cost. The RDNA 3 compiler pitfalls retested (16-bit splitting, copies into half vectors, unrolling until registers spill). Lane permute with the fetch-inactive bit. | `measurements/compiler-and-hazards.md` |
| 9 | The Vulkan view | Cooperative-matrix property table from RADV. Wave size RADV picks. A few of the same measurements through Vulkan compute, to separate hardware from the ROCm stack. | `measurements/vulkan-view.md` |

## Order

One lane at a time: helpers started from the lead's thread share this one checkout. Each lane's owner (Opus) may
use Sonnet 5.5 helpers for clearly specified pieces and Haiku 5.5 for sifting logs (owner permission, 2026-11-17).
Order: 1, 2, 4, 5, 3, 6, 7, 8, 9. The lead reads each lane's diff, reruns its probes, and merges.

Status: lane 1 done (`measurements/execution-core.md`, merged, rerun by the lead).
Lane 2 done (`measurements/memory-system.md`; rerun by the lead; Sonnet review FIX-PASS, two fix rounds, then PASS on
fde2427; merged).
Lane 4 done (`measurements/matrix-timing.md`; rerun by the lead; Sonnet review FIX-PASS, one fix round, then PASS on
9163ba1; merged). It found a counting mistake in lane 1's wave-sharing probe: fixed, rerun, and item 6 of
`measurements/execution-core.md` corrected (several waves on one SIMD lose nothing).
Lane 5 done (`measurements/matrix-arithmetic.md`, `measurements/errors-after-quantizing.md`; two halves, three
review rounds; the lead found and fixed two model gaps with full-range BF16 inputs; test plan in
`notes/plans/lane5-test-plan.md`). Lane 6 done (`measurements/conversions-and-dots.md`). Lane 9 done
(`measurements/vulkan-view.md`). Lane 3 done (`measurements/lds-and-special-loads.md`). Lane 8 done
(`measurements/compiler-and-hazards.md`). Each was rerun by the lead and reviewed by Sonnet (FIX-PASS, fixes made,
re-checked) before its local merge. Lanes 6, 9, 3 and 8 were written in staging folders under
`rdna3-fp8/build/rdna4-runs/` and copied in, so that several could run at once.
Lane 7 done (`measurements/scheduling-and-clocks.md`), rerun by the lead without its 20 s traces, reviewed by
Sonnet, merged. So all nine lanes are measured, reviewed and in master.
Leftovers: the matrix-unit items (`scripts/study-leftovers-matrix.sh`) and the timing items
(`scripts/study-leftovers-timing.sh`) are done, reviewed and merged; each measurement file's "Not measured" list
says what is still open.
Overclock comparison done (`scripts/study-overclock.sh`, short bursts only, owner decision 2026-11-17; run
`20261117T182423Z-overclock`, with stock-only rounds in the `-overclock` runs after it;
`measurements/scheduling-and-clocks.md` item 5a). The
card is back on the "Testing" profile. Any further switch of the profile needs a new OK from the owner.
On this machine the desktop keeps the card partly busy; see "Disturbed timings" under the rules above.
Timing leftovers pass (branch `study/leftovers-timing`, `scripts/study-leftovers-timing.sh`, run
`runs/20261117T175731Z-timing`): the timing oddities and the remaining non-matrix items, written into new sections of
the measurement files (execution-core 8 and 9, matrix-timing 8a and 9a, memory-system 2b-2 and 7a, vulkan-view 5,
conversions-and-dots 3, compiler-and-hazards 3) and the rule on disturbed timings above. Rerun from the committed
code (c0605c8): `runs/20261117T180204Z-timing`, every verdict the same, all 9 deliberate breaks FAIL. The study lead's rerun
after rebasing onto master (`runs/20261117T180457Z-timing`): the same statuses, 9 breaks FAIL.
Gaming round (2026-11-17 and 18), asked for by the owner and scoped with the upscaler's developers: two lanes,
each reviewed twice. `measurements/gaming-route.md`: an upscaler's neural network (model 1, model 2) run headless on the route
games use (vkd3d-proton, dxil-spirv, RADV), with the predictor tables in `measurements/gaming-tables/`.
`measurements/gaming-probes.md`: plain-Vulkan and HIP probes (`scripts/study-gaming-probes.sh`) of dispatch gaps,
queues, the clock under bursts and preemption. In-game recordings with the owner (Cyberpunk 2077, 2026-11-18) are in
`notes/home-session.md`'s expected pattern and in `scheduling-and-clocks.md` item 5a for the memory clock. The
home-session kit (`scripts/study-home.sh`, `notes/home-session.md`) prepares the tests that need the owner at the
machine. Their staging folders and run data stay outside the repository (`rdna3-fp8/build/rdna4-runs/gaming/`).

Leftovers C, the last pass over the open items, is done: lane A (`scripts/study-leftovers-c-a.sh`, 33 items,
`measurements/leftovers-c-a.md`) and lane B (`scripts/study-leftovers-c-b.sh`, 28 items,
`measurements/leftovers-c-b.md`) were measured, rerun by the study lead, reviewed twice and corrected, and merged in
master (6948a7f). In each of the ten measurement files the "Not measured" list now puts the items still open first and
says under "Answered since (leftovers C, 2026-11-17)" what the pass answered, with the item's ID and its label
(MEASURED, READ, INFERRED); the new open questions the lanes raised that belong to a file's subject were added to its
list, with a pointer to the lane file. What stays open: items that need the owner (llama.cpp kernel work, real
activations, sustained loads, profile switches, the 1409 MHz memory request, the overclock copy rate), items not
reachable from HIP or Vulkan, and the parts the lane write-ups list as not settled.

LLM data dive done (2026-11-18, branch `study/llm-dive`, `scripts/study-llm.sh`, `measurements/llm-workload.md`,
expectations in `measurements/llm-workload-expectations.md`, written before the first GPU job): llama.cpp (unmodified
and with the FP8 decode kernel) on Llama 3.2 3B and 3.1 8B as FP8 and Q8_0: kernel traces, prompt-size, depth, stream
and quality grids, and 20 s clock and power recordings, on the "Testing" profile with the memory clock at 1408 MHz
(held by the displays). Run `runs/20261118T110108Z-llm` (rerun of the committed fix-round code; first run
`runs/20261118T094213Z-llm`), 130 GPU jobs each, no thermal stop. Reviewed (FIX-PASS, fixes made), every number
traced to the raw logs, rerun. Parked: an exact FP16 reference of the 3B (the converter needs packages not installed).

LLM data dive, round 2 done (2026-11-18, owner's request to redo what failed; converter packages installed with the
owner's OK in `toolchains/convert-venv`, git-ignored): run `runs/20261118T151433Z-llm2`, fix round
`runs/20261118T165826Z-llm2fix`, `measurements/llm-workload-2.md`, expectations in `llm-workload-2-expectations.md`
and `-7.md`, written before the GPU jobs. Exact-weights reference for the 3B (FP8 routes 3e-6 KL divergence from exact,
Q8_0 7e-4); the 8B only against an F16 reference (its F32 version did not fit this machine's memory). The first-hour
37 TFLOPS explained (the shape); the rerun's no-overlap FAIL most likely a timestamp artifact (INFERRED); counters
mostly read 0 on gfx1201; the micro-batch 2048 gap explained; repeated streams; real weight shapes through a separate
copy of llama.cpp with test-only changes. Since the fix round every GPU job of `scripts/study-llm.sh` runs under a
10 GB memory cap with no swap. Reviewed (FIX-PASS, fixes made), numbers traced to the raw logs. Memory clock 1408 MHz.

LLM data dive, round 3 done (2026-11-18): run `runs/20261118T175910Z-llm3`, fix round `runs/20261118T191700Z-llm3fix`,
`measurements/llm-workload-3.md`, expectations in `llm-workload-3-expectations.md` and `-q1d.md`, written before the
GPU jobs. The extra 57 W of FP8 decode is explained to 30 to 33 W by its instruction mix and block shape (the rest is
open); the mixed F16/F32 exact reference for the 8B is 27.9 GB and does not fit under the 10 GB job cap; the rebuilt
cache probe's 4 % lower reading follows the 1 GiB read before its rows (machine code identical; the cache state as the
cause is INFERRED); ROCm 10.0's profiler aborts on every counter set, so whether 10.1's dead counters are new is still
open; the sub-microsecond kernel overlaps are profiler timestamp artifacts on a synthetic pair probe (an in-kernel
clock shows none in 323,982 pairs; INFERRED for llama.cpp's own five). Reviewed (FIX-PASS, fixes made), numbers traced
to the raw logs. Memory clock 1408 MHz.

LLM data dive, round 4 done (2026-11-18, on the owner's overclock profile "Default": 374 W, +150 MHz, -50 mV, memory
1409 MHz): PyTorch (`torch._scaled_mm`) and vLLM 0.29 (AMD's ROCm 10.1 wheels) next to llama.cpp on the 3B and 8B,
`measurements/llm-runtimes.md`, run `runs/20261118T202715Z-llm4`; FP8 matrix instructions confirmed in both runtimes'
prompt kernels; quality against Meta's 16-bit 3B (downloaded with the owner's OK). PyTorch's FP8 path hung the GPU
four times (full resets; the debugging limit was breached after the third, stated in the write-up) and is parked by
the owner. Then `measurements/llm-8b-exact.md`: the 8B against its FP8 weights converted exactly (one job with a
16 GB cap, the web browser closed with the owner's OK). Since then every GPU job of `scripts/study-llm.sh` runs with a 20 ms
thermal guard (hotspot 95 C, memory 100 C, unreadable sensor = stop, guard lost = stop, the job's whole scope
stopped). Reviewed (FIX-PASS twice, fixes made), numbers traced to the raw logs.

## Already measured (first hour)

Read 634 GB/s; FP16 and BF16 matrix instruction 16.1 cycles, FP8 and 8-bit integer 8.1; no overlap with vector work;
operand layout confirmed; FP8 conversions exact, NaN always 0xFF, infinity not clamped in saturation mode;
decode prototype 615 GB/s at one column.

## Card configuration (owner, 2026-11-17)

Since 2026-11-17 11:30 UTC the LACT profile "Testing" is active for the study: stock core clock and voltage, the
driver's default 317 W cap, memory 1258 MHz, performance level auto, a steeper fan curve. The card is an ASUS Prime
RX 9070 XT with Samsung GDDR6. The regulator-level power-limit mod (rdna4p, 1.2x) stays in the card, so the driver's
power reading is scaled down on the main rail. Hotspot under full load on this profile: 62 to 82 C. Lanes 2 onward
ran on this profile; lane 1 (cycle counts only) and the first-hour run were on the earlier profile described next.

Earlier profile (until 11:30 UTC): the card was overclocked and its power limit raised: LACT profile (power cap 374 W, core clock offset +150 MHz,
voltage offset -50 mV, requested maximum memory clock 1409 MHz, performance level auto) and regulator-level mods
applied with `rdna4p.iso` at 1.2x. The driver still reports the memory clock at 1258 MHz. Consequences for every
lane: wall-clock numbers are for this configuration; the driver's power reading is scaled down on the main rail and
is not true board power; hotspot temperature reached 102 C under sustained load before the fan curve was changed, so
keep GPU jobs short. Do not change LACT, sysfs power files or the performance level (the one exception so far: the
owner's OK of 2026-11-17 for the overclock comparison, see the status above).
A once-a-second log of clocks, reported power and temperatures covers 10:42 to 12:11 UTC of 2026-11-17
(`rdna3-fp8/build/rdna4-runs/clock-log-20261117T104212Z.csv`); the logger is no longer running.
