# RDNA 4 measurement study: the AMD RX 9070 XT (gfx1201)

This repository measures how the AMD Radeon RX 9070 XT behaves. Small probes time or inspect one part of the card at a
time, from matrix instructions to memory caches and clock changes. A separate workload study uses large language models
to see how those hardware rules appear in real programs.

The goal is to understand the hardware, not to speed up a program. Each measurement file starts with a plain-language
summary, then gives the tables, method, and rerun commands.

## Key findings

Each bullet states the result. The linked file contains the evidence, spread, and limits of the measurement.

- **Matrix instruction time depends on the input type, not the data.** FP16 and BF16 inputs take 16 cycles. FP8, BF8,
  and 8-bit integer inputs take 8 cycles. Chaining through the accumulator adds no time, while feeding a result back as
  a multiplicand adds 3 cycles. Whole-card rates are about 211 TFLOPS for FP16 and BF16, and 420 TFLOPS for FP8 and
  8-bit integer work. Sparse forms do twice the work in the same time.
  ([matrix-timing.md](measurements/matrix-timing.md))
- **CPU models reproduce the matrix instructions bit for bit, and show that their additions are not exact.** All 11
  dense and all 11 sparse forms match the models. The card adds 16-bit-input products in 4 steps of 4, and 8-bit-input
  products in 2 steps of 8. Each fixed-point step cuts low bits toward minus infinity, then rounds the sum to FP32.
  A small term next to a large one can vanish.
  AMD's rocjitsu emulator (2026-10-22) and a separate public report (2026-10-31) describe the same FP8 structure; a later
  CPU comparison found rocjitsu and this model agree bit for bit ([ext-review-checks.md](measurements/ext-review-checks.md)).
  ([matrix-arithmetic.md](measurements/matrix-arithmetic.md))
- **The study's matrix-arithmetic model agrees bit for bit with AMD's rocjitsu emulator.** The CPU-only comparison
  covered all 11 dense forms, with about 1 million outputs each, and 9 of the 11 sparse forms (all but the two 4-bit
  ones) with valid indices. Both models
  also match all 278,635 recorded card outputs. They differ only for the 8-bit integer sparse form with invalid indices, on AMD's
  SIMD code paths, which AMD's header calls
  unspecified. AMD's rocjitsu code is in ROCm/rocm-systems under the MIT licence.
  ([ext-review-checks.md](measurements/ext-review-checks.md))
- **The matrix adder's error rarely survives rounding to FP8.** Only 2 of 1,446,000 model-shaped outputs differed from
  the exact sum after E4M3 rounding. With FP8 weights, FP16 output differed in 0.004 % of cases at K = 16 and 0.18 % at
  K = 4096.
  ([errors-after-quantizing.md](measurements/errors-after-quantizing.md))
- **FP8 conversions are exact, but their edge cases need care.** Widening to FP32 is exact for every code. Narrowing
  rounds to nearest even over all 2^32 FP32 inputs. Narrowing a NaN always gives 0xFF, overflow gives NaN, and infinity
  still gives NaN with saturation enabled. Widening costs 1.0 cycle per weight to FP32 and 1.5 cycles to FP16 pairs.
  ([conversions-and-dots.md](measurements/conversions-and-dots.md))
- **Most vector instructions issue once per cycle, but a dependent instruction waits 5 cycles.** Every dual-issue pair
  allowed by the guide runs at two operations per cycle. FP64 operations take 32 cycles. Branches cost 9 cycles when
  not taken and 15 when taken.
  ([execution-core.md](measurements/execution-core.md))
- **Streaming reads nearly fill the memory bus, and the cache levels are distinct.** Plain reads reach 633 GB/s against
  a 640 GB/s bus limit regardless of data content. The card has a 32 KB first-level cache per compute unit, 8 MB of L2
  emptied at each kernel launch, a 64 MB memory-side cache, and 16 GB of GDDR6. A first-level hit takes 81 cycles;
  a GDDR6 read takes 485 to 545.
  ([memory-system.md](measurements/memory-system.md))
- **Shared memory answers 2.5 times faster than the first-level cache.** A 32-bit LDS load issues in one cycle per wave
  and returns data after 32 cycles. HIP's default work-group-processor mode halves bandwidth per compute unit: the whole
  card reads 11.5 TB/s in that mode and 21.9 TB/s in compute-unit mode.
  ([lds-and-special-loads.md](measurements/lds-and-special-loads.md))
- **The whole card pauses hundreds of times a second.** The rate changes from run to run: 380 to 1,023 pauses per
  second in the runs read here, so "about every 2.8 ms" fits only the quieter ones. A spinning wave loses about 1.1 us per pause, and 96 % of stall clusters coincide
  across waves on all 4 shader engines. The cycle counter keeps running during the pause, so timing code needs short
  regions or medians.
  In the two saved two-wave runs, 98.8 % and 97.9 % of stall events fell within the window of the other wave's events,
  against about 0.15 % after random shifts (95 % ranges up to 0.6 % and 1.9 %), so the coincidence is far above chance;
  the 16-wave run saved no event times, so its published 96 % figure could not be rechecked.
  ([scheduling-and-clocks.md](measurements/scheduling-and-clocks.md))
- **Vulkan reaches the same hardware limits as HIP.** Register-fed matrix work measured 210.6 TFLOPS for FP16 and 421.0
  for FP8 through Vulkan, against 211 and 420 through HIP. Vulkan read 636 GB/s against HIP's 633. RADV uses wave64 for
  compute shaders with 64 or more invocations, which costs nothing for register-fed matrix code.
  ([vulkan-view.md](measurements/vulkan-view.md))
- **Matrix instructions ignore the lane mask, and the compiler can miss the consequences.** All lanes are read and
  written in wave32 and wave64 across 20 tested forms. A matrix builtin inside a per-lane `if` can therefore give wrong
  results. The compiler also omitted a required hazard fix across a non-inlined call, producing 64 wrong results among
  256 in the probe.
  ([compiler-and-hazards.md](measurements/compiler-and-hazards.md),
  [leftovers-c-b.md](measurements/leftovers-c-b.md))
- **The compiler still omits a required matrix-instruction hazard fix across a non-inlined call.** In listing-only
  checks, no V_NOP sits between a non-inlined callee's final WMMA and the caller's next WMMA that reads that result.
  The gap remains in ROCm 10.2's AMD clang, for gfx1201 and gfx1200, at -O3 and -O1. In upstream clang 22.1.8 the
  callee still returns straight after its WMMA, though in this probe an unrelated instruction happened to fall between
  the two. A gfx11 port also shows the gap for gfx1150 and gfx1151, and the same gap was measured on an RDNA 3 card in the
  companion study (rdna3-fp8, `measurements/compiler-hazards-rdna3.md`). These listings were not run on the card.
  ([ext-review-checks.md](measurements/ext-review-checks.md))
- **Game-style work starts at a lower clock through Vulkan than through HIP and steps up only when idle gaps are short.**
  The same loop starts at 1.7 to 1.8 GHz through Vulkan and 2.7 to 2.8 through HIP. An upscaler's neural network took
  about 2.8 ms per frame at the low clock and 1.6 ms at full clock. On the game route's graphics queue, gaps of 2.1 ms
  or less always allowed a step up, while gaps of 3 ms or more always stayed low. These were headless tests, not a
  measurement of a running game.
  ([gaming-route.md](measurements/gaming-route.md), [gaming-probes.md](measurements/gaming-probes.md))
  Round 5 (2026-11-20): on the profile "Default" the same Vulkan probe starts at 2.8 GHz, with or without an idle HIP process;
  the kernel skips workload-profile requests for graphics rings when overdrive is enabled, while HIP's queue creation
  requests COMPUTE. What moves the start clock between "Testing" and "Default" is open.
  ([round5.md](measurements/round5.md))
- **llama.cpp's FP8 prompts are slower because of the code path, not the FP8 format.** FP8 cannot use the integer matrix
  kernel available to Q8_0, so llama.cpp first widens each weight matrix to FP16. In a 3B prompt of 512 tokens, the
  matrix operations took 26.9 ms for FP8 and 25.6 ms for Q8_0. The 17.8 ms gap came from the extra memory passes for
  widening, activation conversion, and scaling. Q8_0 forced through the same path was as slow as FP8.
  ([llm-workload.md](measurements/llm-workload.md))
- **Generation leaves part of the memory bus unused, and FP8 decode has a real power cost.** Generation streamed 71 to
  83 % of the read rate at a 1408 MHz memory clock. Prompt loops sat at the 317 W reported cap. Patched FP8 generation
  used 56 to 57 W more than Q8_0 at nearly the same bandwidth. Matched test kernels, run at 659 to 692 GB/s against
  about 584 GB/s in real decode, drew 30 to 33 W more for the FP8 instruction mix and block shape. That shows the
  mix costs power, but the test kernels do not run under the real conditions, so how much of the real 57 W they
  account for is open. Round 4 saw the gap (294 against 240 W) with no power throttle flag, so the power cap is not
  needed to produce it.
  ([llm-workload.md](measurements/llm-workload.md), [llm-workload-3.md](measurements/llm-workload-3.md))
  Round 5: the real FP8 decode kernel loop alone draws +47 W over Q8_0 at the 374 W cap, cache-resident, so a lower bound.
  ([round5.md](measurements/round5.md))
- **PyTorch and vLLM use the FP8 matrix instruction for prompts, but this card hung during PyTorch's FP8 jobs.** Their large matrix
  products run hipBLASLt kernels containing `v_wmma_f32_16x16x16_fp8_fp8`, reaching 236 to 259 TFLOPS on the overclocked
  profile. vLLM's one-token decode instead uses the FP8 dot product on the vector unit. The overclocked GPU hung
  four times during PyTorch FP8 jobs, and the jobs are parked. A fifth reset on 2026-11-19, with no study job
  running, named an ordinary desktop app as the guilty process on the same profile (+150 MHz, -50 mV). So the
  overclock and undervolt are as likely a cause as PyTorch's FP8 path. On 2026-11-20 the command of one of the hangs
  ran clean 3 times of 3 on the stock profile, which points at the overclock profile (small numbers); it did not test
  the NaNs, and nothing ran on PyTorch FP8 under ROCm 10.2.
  ([llm-runtimes.md](measurements/llm-runtimes.md), [round5.md](measurements/round5.md))
- **The overclock profile computed correctly in every bit-exact check, which makes the undervolt a less likely cause of the
  resets but does not exclude it.** 522 jobs of the arithmetic and conversion checks at two clock states, three heavy
  bursts beside a GPU-resident bit check (3.0 x 10^10 matrix instructions) and a 2 s whole-card run: zero mismatches, and
  identical results from run to run. The checks are short exact-value checks; a sustained heavy load was not tested.
  ([round5.md](measurements/round5.md))
- **vLLM's 32-against-128-token decode gap did not reproduce.** In one warmed engine the step time was 6.82 ms at every
  context length (151.5 and 147.7 tokens/s by the old method, which differs only by how it counts steps). The engine-busy
  sampler used for outside-view measurements reads 0.04 to 0.13 ms per frame high and does not see HIP work.
  ([round5.md](measurements/round5.md), [llm-runtimes.md](measurements/llm-runtimes.md))
- **Against the true 3B weights, llama.cpp's FP8 and Q8_0 files have similar error.** Mean KL divergence from the 16-bit
  reference was 3.7e-3 for FP8 and 4.3e-3 for Q8_0, against 2e-5 for llama.cpp's own 16-bit file. The earlier, much
  larger ratio compared both files with the exact FP8 checkpoint weights, not with the true model.
  ([llm-runtimes.md](measurements/llm-runtimes.md), [llm-workload-2.md](measurements/llm-workload-2.md))
- **Most profiler counters read zero in the default power state, not because of the ROCm version, and the small kernel
  overlaps are timestamp artifacts in the probe.** Only 4 of 13 counter sets returned numbers under ROCm 10.1 (and 10.2)
  with the performance level on `auto`; memory-side and instruction counters read zero. With the level set to
  `profile_standard` 15 counters (10.1) and 14 (10.2) count; checked against the known work 7 and 6 are exact, 7 and 7
  plausible, and `VALUBusy` is 4 times too high (2026-11-20, [round5.md](measurements/round5.md)). An
  in-kernel 100 MHz clock showed no overlap in 323,982 kernel pairs (162,000 dependent, 161,982 neighbours), while two
  kernels that really ran at the same time overlapped on both clocks. The profiler reported 3 overlaps (465 to 8,040
  ns, so not all under a microsecond), but the in-kernel clock placed the kernels 3.4 us apart. llama.cpp's overlaps
  were not retimed, so applying the artifact finding to them remains INFERRED. Round 4's much larger trace overlaps
  (PyTorch and vLLM, up to 1,393 pairs) are not explained by this finding.
  ([llm-workload-2.md](measurements/llm-workload-2.md), [llm-workload-3.md](measurements/llm-workload-3.md))
- **ROCm 10.2 gives the same results as 10.1, and vLLM's FP8 decode speed comes in two levels that the ROCm version does not explain.**
  FP8 perplexity is identical to the last digit (9.725647880183923), vLLM bf16 speed and all 16 hipBLASLt GEMM speeds are within
  about 2 %, and hipBLASLt picks the same tiles under renamed kernels. vLLM FP8 decode ran at about 159 or about 120 tokens/s at
  one stream depending on the engine start, in both versions, with the same clocks; the cause is unknown. The vLLM version changed
  with ROCm (0.29.1 against 0.32.1.dev0), so ROCm alone is not isolated (2026-11-20).
  A public ROCm/TheRock issue (2026-06-11) reports the same kind of split, fixed at process start, for vLLM on an R9700;
  its cause is open there too.
  ([round6.md](measurements/round6.md), [llm-runtimes.md](measurements/llm-runtimes.md))
- **Decode reads the model's weights exactly once per token, in every format tested.** With the profiler counters working, the
  matrix kernels of llama.cpp (Q8_0, FP8, FP8 with a bf16 head) and vLLM (FP8, bf16) fetch 1.000 to 1.001 times the weight bytes
  of the model file per token. llama.cpp's FP8 reads 5 % fewer bytes than Q8_0 but needs 2.1 times the vector instructions, and
  its attention kernel appears to fetch the key/value cache for 256 positions at a context of about 152 (inferred). Cycle counts there are at a fixed
  1.65 GHz core and a lower memory clock, so they rank kernels and are not real speed (2026-11-20).
  ([round6.md](measurements/round6.md))
- **The memory clock is no longer stuck at 1258 MHz, and a 64-wide wave doubles the 8-bit dot work.** Round 7 (2026-11-24):
  any streaming load from 0.05 s to 20 s took the memory clock to 1408 MHz within 7 to 29 ms, and a read pass then ran at
  about 700 GB/s, 10 to 11 % above the 633 GB/s measured at 1258 MHz, so earlier memory-bound wall-clock numbers are
  about 10 % low. After a load the clock fell to 772 MHz within about 0.15 s and often stayed there. In wave64 the dot
  instructions cost the same cycle but do twice the work, so the 8-bit dot path reaches 99 T MAC/s on the whole card
  against 50 in wave32, and the matrix unit's lead over it drops from 4x to 2x (200.5 against 99.1). The FP8 conversions
  cost twice as many cycles in wave64 and gain nothing. A matrix wave still keeps its SIMD to itself in wave64. One control
  failed: the whole-card matrix rate was 400 to 403 TOPS, not 420, and the cycle counter's clock is 4.8 % above the
  driver's reported clock (cause inferred, not tested).
  ([round7.md](measurements/round7.md))
- **FSR 4 costs about 0.34 ms at 1080p output, 0.63 ms at 1440p and 1.64 ms at 4K on this card, a rough match to AMD's
  published figures.** This is AMD's own FidelityFX SDK sample, timed from the outside with the sample's GPU timestamps
  and the core clock sampled beside it; nothing was traced and no AMD shader or library was opened. The numbers are 0.97
  times AMD's 1080p figure and 1.25 times its 4K figure. The agreement is rough: the sample is FSR 4.1.1 against AMD's
  4.1.2, sharpening and frame generation were on, the card is overclocked, and the core clock sat at 2.3 GHz at the small
  sizes against 3.2 GHz at 4K. The preset was inferred, because the key presses that select it were unreliable, and it moves
  the cost by only 10 to 20 %. The predictions were written first and kept unedited.
  ([round7.md](measurements/round7.md), [fsr4-predictions.md](measurements/fsr4-predictions.md))
- **FSR 4.1.2 costs about 0.35 ms at 1080p output, 0.65 ms at 1440p and 1.66 ms at 4K when the core clock is near 3.4 GHz, and
  whether that cost follows the core clock is open.** This is AMD's FidelityFX SDK sample again (frame generation off, Quality;
  the other presets within about 10 %), timed with the sample's own GPU timestamps. Round 7's low clocks (about 2.3 GHz) came
  with frame generation present and the GPU only 70-82 % busy. Round 7 and round 8 agree within 1.5 %, but that comparison also changes the FSR
  version and frame generation, and a clean test at a lower core clock did not take hold, so whether the cost follows the core clock
  is open; round 7's estimates scaled to peak clocks are withdrawn. Six runs (five grid cells and one clock test) that ran with the
  memory clock at 1124 MHz instead of 1408 were 13 to 17 % slower, which fits a memory-clock effect but was observed, not set. A third
  party's exact rewrite of the load order gave matching picture fingerprints and came out 2.2 to 4.1 % faster at Balanced (0.1 to 2.4 %
  at Ultra Performance) on this card, on a noisy session, against 7.2 to 9.6 % reported for moving Balanced on Windows. By that project's
  own scanner, RADV's compiler output shows no serialized load waits in any of the 12 model passes; compile counts are published, AMD's
  shader code is not. Timed per dispatch with that project's kit (one instrumented run), the first and last full-size dispatches took
  48 % of the dispatch time at 1080p and 60 % at 1440p and 4K, the 12 dispatches inferred to be the model passes 48 % and 36 %, and
  the 13 small dispatches under 1 %.
  ([round8.md](measurements/round8.md))
- **Loads come back in the order they were issued, and a load that lands during FP8 matrix work costs its wave about 25 cycles.**
  In every pattern tested, a wait that allows one load still out returned only when the older load had arrived (195,840 lane checks,
  no early return). On one wave, one load per 4 or 8 matrix instructions cut the extra to a quarter and an eighth, so the cost follows
  the loads, not the waits. The whole-card version of this probe failed its own reference checks, so no whole-card figure is given.
  ([round9.md](measurements/round9.md))
- **A shader sees a steady core clock at a 60 Hz cadence, but Vulkan graphics runs at a lower clock than HIP and has dips.** One short
  dispatch every 16.7 ms saw 2960 MHz through HIP and 2022 MHz through Vulkan; six of seven Vulkan runs also had a few dispatches
  near 1180 MHz that a median hides. `profile_standard` held 1623 MHz with no dips in the six runs that started, and it also changed
  the memory clock, so that comparison is not core-clock-only. Alone, the card read GDDR6 at 705 GB/s at a memory clock of 1408 MHz and L0 at 81 cycles
  per load. What one dispatch leaves in L2 for the next, and whether cache latency depends on the memory clock, were not
  answered: the controls failed or the clock could not be separated from the load that raises it.
  ([round9.md](measurements/round9.md))

Update (2026-11-25): three no-card checks cover the compiler gap at a call boundary, bit-for-bit agreement with AMD's
rocjitsu arithmetic model, and a chance control for the pause coincidence
([ext-review-checks.md](measurements/ext-review-checks.md)). A new dry-run-only driver for later memory-clock pinning
keeps every settings change under the shared lock, restores the setting even if the job is killed, and blocks further
jobs if the restore fails. It refuses real runs without an explicit switch and the right card. Its 22 self-test cases
used stand-in files; it has not run on the card.
([round10-p0.md](measurements/round10-p0.md))

## Card and setup

- Card: ASUS Prime RX 9070 XT, 16 GB Samsung GDDR6, 64 compute units, gfx1201, wave32 under HIP.
- Compiler and runtime: ROCm 10.1, HIP 7.16, and AMD clang 24. The Vulkan probes use Mesa RADV 27.0.0-devel.
- Operating system: Linux kernel 7.3.4.
- Round 6 (`round6.md`) also ran ROCm 10.2 beside 10.1 for its comparison item; the other results used ROCm 10.1.

The configuration matters for every wall-clock result:

- Most hardware probes used the LACT profile "Testing" with stock core clock and voltage, a 317 W cap, and memory at
  1258 MHz. The first-hour run and lane 1's cycle counts used an earlier overclocked profile. Cycle counts come from
  the wave's own counter and do not depend on the clock frequency.
- LLM rounds 1 to 3 used "Testing" with the memory clock at 1408 MHz.
- Round 5 (`round5.md`) used "Default" for all 590 jobs, with the memory clock at 1258 MHz.
- Round 6 (`round6.md`): item R (ROCm 10.2 against 10.1) used "Default" with the memory clock at 1408 MHz; item C (counters) set the performance level to `profile_standard` (core near 1.65 GHz, memory 1124 MHz), with "Default" as the LACT profile, and set it back to `auto`.
- Round 7 (`round7.md`) used "Default" for all three items. The memory clock followed the load and reached 1408 MHz.
- Round 8 (`round8.md`) used "Default". The memory clock was observed, not set: it read 1408 MHz in most cells and 1124 MHz in the six slow ones.
- Round 9 (`round9.md`) used "Default" for most items; item G1 set the performance level to `profile_standard`. The memory clock moved between and
  inside jobs (456 to 1408 MHz), and each cell states it where it matters.
- LLM round 4 and the exact 8B step used the overclocked "Default" profile: a 374 W cap, +150 MHz core offset, -50 mV,
  and a requested memory maximum of 1409 MHz. Round 4 read 1408 MHz until a GPU reset and 1258 MHz afterward. The exact
  8B step read 1408 MHz throughout. Each wall-clock table in those files states its memory clock.
- A 1.2x regulator-level power-limit mod is installed. Reported watts are scaled down on the main rail and are not true
  board power. The study notes document the mod; these probes did not measure it.

Wall-clock numbers in GB/s, TFLOPS, GHz, and watts describe this card in these configurations, not a stock card. Later
rounds correct earlier interpretations where the files say "Updated in round N"; the findings above use the corrected
results.

A new memory overclock took effect only after the displays blanked once. Before that, the active memory level stayed at
1258 MHz. After switching the displays off and on, `gpu_metrics` reported 1408 MHz. Read `current_uclk` from
`gpu_metrics` after a change rather than relying on the listed levels in `pp_dpm_mclk`.
Round 7 ([round7.md](measurements/round7.md)): a 1408 MHz reading under load now needs no blanking (that run made no display
change); the memory clock follows the load.
([scheduling-and-clocks.md](measurements/scheduling-and-clocks.md))

## How results are labelled

- **MEASURED** means observed in a run on this card. The write-up names the run folder.
- **READ** means taken from a document, source file, machine-code listing, or driver dump.
- **INFERRED** means reasoned from measurements rather than observed directly.

A probe is trusted only after it reproduces a known answer from AMD's documentation, an earlier run, or a CPU model.
Expectations are written before the runs and kept in the `*-expectations.md` files. Probes print `VERDICT` lines and
include deliberate breaks that must report `FAIL`. A check that cannot fail proves nothing. The write-ups report the
spread as well as the median.

AMD's official [RDNA 4 Instruction Set Architecture: Reference Guide](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
is the main instruction reference.

## Repository layout

| Folder | What it holds |
|---|---|
| `measurements/` | Measurement write-ups, expectations written before runs, and `gaming-tables/` with three derived CSV tables. |
| `notes/` | The study plan, first-hour record, session guide, and reading notes on the hardware and software stack. |
| `notes/plans/` | The matrix-arithmetic test plan. |
| `scripts/` | One `study-*.sh` entry point per study lane, Python checkers and table builders, and expected machine-code patterns. `scripts/round5/`, `scripts/round6/` and `scripts/round9/` hold the rounds' own wrappers and analysis scripts. |
| `scripts/extcheck/` | Build, scan, extraction, and pause-chance scripts for the three no-card checks. |
| `scripts/round10/` | The dry-run-only memory-clock pinning driver and its self-test. |
| `src/` | HIP probes, Vulkan host programs, and Vulkan shaders. `src/round5/`, `src/round6/` and `src/round9/` hold the rounds' own probes. |
| `src/extcheck/` | The gfx11 compiler probe, arithmetic-model comparison harness, and rocjitsu build wrapper. |

`runs/`, `build/`, `toolchains/`, and `sources/` are git-ignored. Raw run data is not in the repository. Each write-up
names its run folder and the script and mode that produced it. The committed tables preserve the results; rerun the
script to check them against new raw data.

## Measurement files

| File | Subject |
|---|---|
| `execution-core.md` | Instruction issue and latency, dual issue, scalar operations, branches, and waves sharing a SIMD. |
| `memory-system.md` | Global-memory bandwidth and latency, cache levels, loads in flight, stores, and cache hints. |
| `lds-and-special-loads.md` | Shared memory, bank conflicts, scalar loads, block loads, and transposing loads. |
| `matrix-timing.md` | Cost of each matrix form, operand hazards, sharing, and whole-card rates. |
| `matrix-arithmetic.md` | Bit-exact models of the matrix instructions and their addition rules. |
| `errors-after-quantizing.md` | How often matrix-addition error survives rounding to FP16 or FP8. |
| `conversions-and-dots.md` | Small-format conversions, dot products, and the cost of widening FP8. |
| `scheduling-and-clocks.md` | Launch cost, work placement, barriers, clocks, power, the regular pause, and process sharing. |
| `compiler-and-hazards.md` | Hazard handling, lane masks, and wrong or slow compiler output. |
| `vulkan-view.md` | The same hardware through Vulkan compute and RADV. |
| `fsr4-predictions.md` | Predictions for the cost of FSR 4, written before any measurement. |
| `round7.md` | Round 7: the memory clock under load (1408 MHz, unlocked), 32- against 64-wide waves, and FSR 4 cost seen from the outside with AMD's SDK sample. |
| `round8.md` | Round 8: FSR 4.1.2 cost per size and preset, the timing-kit result, a time for each of its 28 dispatches (one instrumented run, names inferred), an observed slowdown with a lower memory clock, and a look at how the compiler built its passes (counts only, no AMD shader code); withdraws round 7's clock-scaled FSR estimate. |
| `round9.md` | Round 9: load order, loads beside matrix work (raw tables and a one-wave load ratio), data across dispatches (observations), the core clock at a fixed cadence and under `profile_standard`, and memory bandwidth at 1408 MHz. |
| `ext-review-checks.md` | Three no-card checks: compiler listings at a call boundary, bit-for-bit comparison with AMD's rocjitsu arithmetic model, and a chance control for pause coincidence. |
| `round10-p0.md` | Dry run only: a memory-clock pinning driver with locked settings changes, watchdog restore, a failure gate, and 22 self-test cases using stand-in files. |
| `leftovers-c-a.md` | Follow-up probes of the execution core, memory, LDS, and compiler hazards. |
| `leftovers-c-b.md` | Follow-up probes of matrix rules, scheduling, special loads, Vulkan, and arithmetic error. |
| `gaming-route.md` | An upscaler's neural network on the headless vkd3d-proton, dxil-spirv, and RADV route. |
| `gaming-probes.md` | Vulkan and HIP probes of dispatch gaps, queues, bursts, clock steps, and preemption. |
| `gaming-probes-f-expectations.md` | Expectations for the final gaming probe round, written before its runs. |
| `llm-workload.md` | LLM round 1: llama.cpp kernel traces, speed, quality, clocks, power, and bandwidth. |
| `llm-workload-expectations.md` | Expectations for LLM round 1. |
| `llm-workload-2.md` | LLM round 2: exact-weight references, matrix shapes, profiler counters, cache behavior, and streams. |
| `llm-workload-2-expectations.md` | Expectations for the first part of LLM round 2. |
| `llm-workload-2-expectations-7.md` | Expectations for the stream and matrix-shape items in LLM round 2. |
| `llm-workload-3.md` | LLM round 3: FP8 decode power, the exact 8B reference attempt, cache state, counters, and overlap timestamps. |
| `llm-workload-3-expectations.md` | Expectations for LLM round 3. |
| `llm-workload-3-expectations-q1d.md` | Expectations for the random-data power follow-up. |
| `llm-runtimes.md` | LLM round 4: PyTorch, vLLM, and llama.cpp on the overclocked profile, including the PyTorch hangs. |
| `llm-runtimes-expectations.md` | Expectations for LLM round 4. |
| `llm-8b-exact.md` | The 8B files against the FP8 checkpoint's own weights converted exactly. |
| `llm-8b-exact-expectations.md` | Expectations for the exact 8B step. |
| `round5.md` | Round 5: overclock bit-exact checks, the engine sampler's offset, real FP8 decode kernel power, the vLLM decode gap, and the start clock. |
| `round6.md` | Round 6: ROCm 10.2 against 10.1 (vLLM speed, FP8 perplexity, hipBLASLt GEMMs) and profiler counters on the real decode kernels. |

Two earlier follow-up passes have no separate measurement file. Their results appear in new sections of the lane files.

## Running the probes

Read the usage text at the top of the relevant `scripts/study-*.sh` file first. Most hardware lanes accept `build`, `run`,
`breaks`, or `all`. The `all` mode builds, runs the probes, then runs the deliberate breaks.

```sh
scripts/study-core.sh all
scripts/study-memory.sh all
scripts/study-vulkan.sh all
```

The scripts read `local.env` in the repository root when that file exists. Copy the committed `local.env.example` and
set the paths for your machine:

| Setting | Purpose |
|---|---|
| `STUDY_GPU_LOCK` | Lock file that serializes GPU jobs. The default is `runs/gpu.lock`. |
| `LLM_BUILD_ROOT` | Folder containing the llama.cpp builds and the perplexity text. |
| `LLM_MODELS` | Folder containing the GGUF models. |
| `RDNA3_DIR` | Optional: a checkout of the RDNA 3 companion repository, used by `scripts/first-hour.sh`. |
| `STUDY_UPSCALER_LOG` | Optional: the path of an upscaler's log inside a game folder, read by `scripts/study-home.sh`. |

ROCm 10.1 is found through `ROCM_PATH` in the environment, not through `local.env`.

The Vulkan probes also need Mesa RADV and `glslc`. The LLM study needs the external builds, models, and runtime
environments described in its measurement files.

Run `scripts/first-hour.sh info` before the other probes. This read-only step writes `runs/<time>-gfx1201/card.env`;
later scripts read the card record through `runs/latest/card.env`.

Every GPU job takes the shared lock, runs under `nice`, refuses to start below 8 GB of free memory, and logs the clocks
before and after; whole-card jobs keep to a hotspot limit. The LLM study's script (`scripts/study-llm.sh`) adds more:
a game check, a `systemd-run` scope with a 10 GB host-memory cap and no swap, and a 20 ms thermal guard that stops the
job at a 95 C hotspot or 100 C memory temperature, if the hotspot sensor becomes unreadable, or if the guard itself is
lost. An unreadable memory temperature only logs a warning and the job goes on.

Whole-card loads can heat this card quickly, especially on an overclock. Keep added loads short and do not bypass the
thermal guard. `scripts/study-overclock.sh`, `scripts/round9/stable_clocks.sh` and some `scripts/study-home.sh` steps change card settings; they are not
part of a normal probe run.

## Status and open work

The nine hardware lanes and their follow-up passes are measured, reviewed, and rerun. The gaming studies and four LLM
rounds are complete and reviewed. The exact 8B comparison is also complete. Rounds 5 to 9 followed up on open items from
those: the overclock checks, the profiler counters, ROCm 10.2, the memory clock, wave64, FSR 4 from the outside and a look at its compile counts,
load order, and the core clock at a fixed cadence.

The open lists in the measurement files still include sustained loads and some profile changes, measurements during a
running game (FSR 4's cost was measured only in AMD's sample, not in a game, and whether it is bound by the core clock is open), what loads cost beside matrix work on the whole card, what one dispatch leaves in L2 for the next, whether cache latency depends on the memory clock, a time per FSR 4 pass, questions not reachable through HIP or Vulkan, and which part of `profile_standard` makes the profiler counters
count (the ROCm version does not). The remaining part of FP8 decode's extra power is unexplained. PyTorch's FP8 hangs, NaNs,
and non-reproducibility remain parked rather than diagnosed; the hangs may well belong to the overclock profile, since one
hanging command ran clean three times on stock. Other GPUs, multi-GPU runs, and real chat workloads are outside the measured scope.
