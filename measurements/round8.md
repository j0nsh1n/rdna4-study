# Round 8: FSR 4.1.2 on the RX 9070 XT (outside timing, the timing kit, a look inside)

Three parts, all on 2026-11-24, card RX 9070 XT (gfx1201, `lspci` 1002:7550), kernel 7.3.4-202:

- **B, the timing grid:** `runs/20261124T193633Z-r8fsr/` (56 cells, plus the pilot `exp-e1/` and the first test `test1/`; `RESULTS.md`,
  `analysis.md`, `analyse.py`).
- **Timing kit:** the project's Linux timing kit run on this card (report summary `results-20261124-130943.txt`, kit 2026-11-14, report DE4301C9).
- **Inside:** `runs/20261124T193633Z-r8fsr/inside/` (`RESULTS-inside.md`, `PLAN.md`; compile statistics and a static scan, CPU only).

The grid and the kit ran on the LACT profile "Default" (overclock: +150 MHz core, -50 mV, 374 W cap; the power the driver reports is about
1.2x too low because of the regulator mod). Every wall-clock number of ours (ms, GHz) describes this card in that state, not a stock card; figures quoted from other sources (AMD, PR 57's
Windows contributor) keep the conditions their source states.
Labels as in `round7.md`: MEASURED (a number in a log of the run folder), READ (a source file, a listing or someone else's text),
INFERRED (worked out from measured numbers, not observed). Run folders are git-ignored; AMD's shader code (dumps, machine code, SPIR-V)
was read on this machine only and none of it is in this repository: this write-up holds counts and conclusions.

## Plain summary

Each bullet carries its own label.

- **MEASURED:** with frame generation off and the core clock near 3.4 GHz, FSR 4.1.2 took about 0.35 ms per frame at 1080p output, 0.65 ms at 1440p
  and 1.66 ms at 4K (Quality; the other presets are within about 10 % of that), measured with AMD's own sample.
- **MEASURED:** round 7's low core clocks (about 2.3 GHz at 1080p and 1440p) came with frame generation present in every cell and a GPU 70-82 % busy
  (round 7's own activity medians); the two coincided in round 7, and its files do not show which one caused the other. Round 8's retained frame-generation grid cells
  clocked 3428, 3432 and 3401 MHz at 81, 80 and 95 % busy (the pilot's cells are reported separately below). **INFERRED:** frame
  generation is associated with the low busy share, and its part in the low clocks is not isolated.
- **MEASURED:** round 7 (about 2.3 GHz) and round 8 (about 3.4 GHz) agree within 1.5 %, but that comparison also changes version, frame generation and
  backend, and a clean test at a fixed lower core clock did not take hold. So whether the cost follows the core clock is open, and round 7's "scale to peak
  clocks" estimates are withdrawn (**INFERRED:** they assumed the cost is inverse to the clock).
- **MEASURED:** five slow grid cells and one slow clock-pair cell had a median memory clock of 1124 MHz instead of 1408 and ran the upscaler 13-17 % slower than
  comparable cells at 1408. **INFERRED:** this is consistent with a memory-clock effect; the clock state was observed, not set, and the five grid cells were
  picked out after the run from their times. One other disturbed grid cell (1080p Quality with frame generation on, memory at 1408) is unexplained.
- **MEASURED (the kit's report):** on the timing kit, all 12 picture fingerprints of an exact load-reorder rewrite (PR 57 of the overrides project) match the
  reference's. The noisy Linux means were 2.2 to 4.1 % lower in Balanced and 0.1 to 2.4 % lower at Ultra Performance. **READ:** PR 57 reports 7.2 to 9.6 %
  lower times for moving Balanced 4.1.2 on a Windows 9070 XT at stable clocks.
- **MEASURED:** the Linux compiler (ACO) has no serialized load waits by PR 57's scan in any of the 12 model passes at any of the 3 sizes (36 rows). **READ:** PR 57's
  Windows-compiler output has 24 to 39 in five passes (4.1.1 table) and 110 in pass 9 (4.1.2 example; 113 in the 4.1.1 table). **INFERRED:** this fits the smaller gain
  here. **MEASURED against READ:** ACO uses more registers, and pass 9 has a static limit of 12 waves per SIMD where PR 57's figures give 16.
- **MEASURED:** the timing kit's `--passes` mode gives a time for each of the 28 dispatches of one upscale call (moving Balanced, one run, instrumented with timestamp queries, on the overclocked card). On AMD's 4.1.2,
  dispatches 1 and 27 (full-size grids) took 48 % of the dispatch time at 1080p and 60 % at 1440p and 4K, the 12 odd dispatches 3-25 took 48 % at 1080p and 36 % at 1440p and 4K, and the 13 one-row even dispatches 2-26
  took under 1 %. From the first dispatch's start to the call-end timestamp, the exact rewrite changed the total by -0.7 % (1080p), -0.8 % (1440p) and +0.1 % (4K) in this run, against -3.5, -3.6 and -2.2 % for the kit's
  earlier noisy normal-run means; these totals do not depend on which dispatch is which pass, and the measurements do not isolate a cause for the difference. **INFERRED** (from dispatch order and grid sizes, not from any
  name in the DLL): dispatches 1 and 27 are the prepass and postpass, the odd dispatches 3-25 are model passes 1 to 12, the even dispatches 2-26 are the 13 small post shaders, and dispatch 0 is the extra small shader; see "Per-step timing".

## Setup

- **Sample (MEASURED).** AMD's FidelityFX SDK v2.3.0 prebuilt sample with the signed v2.4.0 DLLs, which the log's version id reports as FSR
  Upscaling 4.1.2 and Frame Generation 4.0.2 (round 7 had 4.1.1). Run under GE-Proton inside gamescope's **headless** backend (no window), timed with the
  sample's own GPU timestamps; the number is the label `FFX API FSR Upscaler`. Outside view: no DLL or shader was opened in the grid.
- **Method (MEASURED).** 56 cells = 3 output sizes (1920x1080, 2560x1440, 3840x2160) x (5 presets with frame generation off, plus Quality with
  frame generation on) x 3 repeats (repeats 1 and 3 forward, repeat 2 in reverse order), plus `ref-first` and `ref-last` (1440p Quality, off).
  Every cell ran through the study gate (shared lock, 20 ms hotspot guard).
- **Keys (MEASURED).** F6 presses pick the preset from Quality; F7 turns frame generation off, which changes the timing labels and restarts the
  sample's averages, so the numbers cover only the chosen preset at full load. The preset that ran is **read from the sample's log** (the last
  `qualityMode` before the final restart), not inferred: all 56 cells ran the preset they were named for, and the Lighting label (which scales with
  render size) agrees at every size.

## The pilot: why round 7 clocked low

`runs/20261124T193633Z-r8fsr/exp-e1/`, 1080p Quality, 2500 frames per cell (MEASURED; one cell each, a pilot):

| cell | frame generation | gamescope extras | AvgFPS | upscaler ms | GPU activity p50 | `EndFrame` CPU ms | core clock p50 (MHz, samples above 20 % activity) |
|---|---|---|---|---|---|---|---|
| `base` | on | none | 171.7 | 0.338 | 75 % | 3.79 | 3261 |
| `r1000` | on | refresh 1000, immediate flips | 179.8 | 0.343 | 76 % | 3.67 | 3352 |
| `fg0` | off | none | 233.9 | 0.346 | 100 % | 0.08 | 3325 |
| `r1000fg0` | off | refresh 1000, immediate flips | 234.7 | 0.349 | 100 % | 0.07 | 3430 |

MEASURED: with frame generation on, the GPU is about 75 % busy and the CPU sits about 3.7-3.8 ms per frame in `EndFrame`; with it off the GPU is
100 % busy and the wait is gone. Letting gamescope run at a refresh of 1000 Hz with immediate flips changed little. The pilot cells are not a preset
or a size study, and one cell each is a single reading. Also in the folder: `presetcheck` showed that two F6 presses give `qualityMode` 3 at 960x540,
which is how the log-read preset check was fixed.

INFERRED: round 7's grid never pressed F7, so every cell ran with frame generation on and a GPU that was only 70-82 % busy (round 7's own activity medians: 72, 70 and 82 % at 1080p, 1440p and 4K); low busy share is the
state the driver tends to clock down in. **But** the pilot's own `base` cell clocked 3261 MHz at 75 % busy (samples above 20 % activity), and the grid's frame-generation cells (below) 3401-3432 MHz at 80-95 % busy,
so busy share alone does not give round 7's 2.3 GHz. Round 7 also used a windowed gamescope and FSR 4.1.1; those differences
were changed together and were not separated. The first test cell `test1/` (frame generation on, 1080p Quality) shows the low-clock state:
upscaler 0.366 ms, 70 % busy, 134.5 FPS, core clock median 2.08 GHz over all busy samples of the file (2.70 GHz inside the benchmark window, p10 1.97 GHz).

## The grid

Upscaler ms, frame generation off, preset-checked cells; median (min-max, n) **without** the disturbed cells (see "Open"). MEASURED.

| Output | Native AA | Quality | Balanced | Performance | Ultra Performance |
|---|---|---|---|---|---|
| 1920x1080 | 0.370 (0.365-0.370, 3) | 0.346 (0.344-0.350, 3) | 0.347 (0.342-0.349, 3) | 0.343 (0.337-0.343, 3) | 0.338 (0.335-0.339, 3) |
| 2560x1440 | 0.709 (0.708-0.714, 3) | 0.652 (0.646-0.658, 5) | 0.642 (0.637-0.642, 3) | 0.633 (0.628-0.635, 3) | 0.604 (0.603-0.604, 2) |
| 3840x2160 | 1.788 (1.785-1.790, 2) | 1.663 (1.660-1.667, 2) | 1.644 (1.639-1.648, 2) | 1.625 (1.622-1.625, 3) | 1.596 (1.595-1.608, 3) |

The 1440p Quality n=5 includes `ref-first` and `ref-last`. At n=2 or 3, read the "median" as a typical value, not a distribution. With the disturbed
cells included the 4K medians move by at most 0.3 % (Native AA 1.788 to 1.790, Quality 1.663 to 1.667, Balanced 1.644 to 1.648), while the maxima jump by 13-17 % (single
slow repeats: 2.086, 1.948, 1.921 at 4K; 1440p Ultra Performance reads 0.604 with a maximum of 0.681); the full table is in `RESULTS.md`.

- **The preset trims, size sets the cost (MEASURED).** Against Quality = 100 %: Native AA 107 / 109 / 108 % and Ultra Performance 98 / 93 / 96 % at
  1080p / 1440p / 4K. Render sizes run from 1.0x (Native AA) to 3.0x smaller (Ultra Performance); the cost falls only slightly as the render size
  shrinks, so most of it follows the output size (INFERRED).
- **Against AMD's published 4.1.2 figures (READ: gpuopen FSR 4.1.2 manual, RX 9070 XT, RCAS off, peak clocks, no mode named).** 1080p: AMD 0.352 ms,
  our Quality 0.346 (0.98x). 4K: AMD 1.316 ms, our Quality 1.663 (1.26x). No 1440p figure was read. These were READ on 2026-11-19 (per the study's
  notes) and not re-read; not a test of AMD's table: our sample has RCAS on, the card is overclocked and the table's mode is unnamed. Part of the 4K gap may be
  the sharpening pass (INFERRED; the sample has no separate label for it).
- **Controls (MEASURED).** No off cell's median core clock was below 3.0 GHz (lowest 3.19 GHz, `r1-1920x1080-nativeaa`; five off cells below 3.3 GHz, most others 3.37-3.45 GHz; window p10 mostly about 3.15-3.2 GHz, but 1.97-2.02 GHz in `r1-1920x1080-nativeaa`, `r1-1920x1080-balanced` and `ref-first`). Window = the last `Runtime` seconds of the clock file minus a 1.5 s tail (`analysis.md`). Drift:
  `ref-first` 0.6464 ms against `ref-last` 0.6582 ms (+1.8 %, inside the 3 % line). Forward against reverse repeats agree within 1.5 % for 13 of 18
  cell types; the other five are the disturbed cells below.

## Frame generation (MEASURED, Quality, undisturbed cells, medians)

| Output | FG | AvgFPS | upscaler ms | core clock p50 MHz (busy) | activity p50 % | `EndFrame` CPU ms | n |
|---|---|---|---|---|---|---|---|
| 1920x1080 | off | 225.4 | 0.346 | 3352 | 100 | 0.07 | 3 |
| 1920x1080 | on | 206.4 | 0.348 | 3428 | 81 | 3.48 | 2 |
| 2560x1440 | off | 194.1 | 0.652 | 3403 | 100 | 0.05 | 5 |
| 2560x1440 | on | 156.2 | 0.649 | 3432 | 80 | 4.39 | 2 |
| 3840x2160 | off | 147.4 | 1.663 | 3409 | 100 | 0.06 | 2 |
| 3840x2160 | on | 119.9 | 1.668 | 3401 | 95 | 6.55 | 3 |

- The upscaler's own time is the same on or off (within 0.5 %), and so is the core clock (about 3.4 GHz).
- With frame generation on, the GPU is 80-95 % busy (cell medians), the CPU waits 3.5-6.6 ms per frame in `EndFrame`, and the sample's AvgFPS is 8 % lower at 1080p, 20 % at
  1440p and 19 % at 4K. This run does not say whether that AvgFPS counts rendered or presented frames, so "FPS fell" means that number fell. INFERRED: the GPU
  waits for the generated-frame work to line up, the CPU sits waiting.
- The label `FFX API FSR Frame Generation Prepare` is the only frame-generation label the sample reports: 0.033 ms (1080p), 0.064 (1440p), 0.110 (4K). The
  dispatch of the frame generation itself has no label of its own.

## The core clock: no clear dependence seen, not cleanly tested (MEASURED costs; INFERRED readings)

| Comparison | Core clock | Quality cost (ms) | Source |
|---|---|---|---|
| round 7, 1080p | about 2.3 GHz | 0.349 | `runs/20261124T090844Z-r7fsr-b1/` |
| round 8, 1080p | about 3.4 GHz | 0.346 | this grid |
| round 7, 1440p | about 2.3 GHz | 0.643 | same |
| round 8, 1440p | about 3.4 GHz | 0.652 | this grid |
| round 7, 4K | about 3.2 GHz | 1.661 | same |
| round 8, 4K | about 3.4 GHz | 1.663 | this grid |
| `test1`, 1080p (FG on) | about 2.1 GHz (all busy samples; 2.7 GHz in the window) | 0.366 | `test1/` |
| round 8 grid, 1080p Quality (FG off) | about 3.4 GHz | 0.346 | this grid |

MEASURED: across these setups (a 1.5x higher core clock at 1080p and 1440p, 2.3 to 3.4 GHz, plus the changes in version, frame generation and backend) the Quality cost agrees within 1.5 % of round 7's (1080p -0.9 %, 1440p +1.4 %, 4K +0.1 %). The 2.1 against 3.4 GHz pair is a weaker point: `test1` was a first test with frame generation on and a clock that
moved a lot inside the window (p10 1.97, median 2.70 GHz), so it shows only that this one cell, with a clock 1.3 to 1.6x lower and frame generation on, read 5.7 % higher (0.366 ms against 0.346 ms in the grid's 1080p Quality cells with it off); frame generation and the unsteady clock are not separated from the core-clock difference. INFERRED: time that a
faster core clock does not shorten fits waiting on loads (latency, caches, memory) rather than on arithmetic; the card's own load latencies and the
inside look (below) fit that, but nothing here tests it (cycle counts from the probes are in core cycles, and an equal elapsed latency counts as different cycles at different core
clocks, so those counts depend on clock state too), and the sampler sees the clock around a 0.3-1.7 ms dispatch, not during it. The round 7
against round 8 rows also change the version (4.1.1 against 4.1.2; the timing kit puts 4.1.2 0.3-5.1 % faster on this card in Balanced and 1.8-7.7 % at Ultra Performance, on a noisy session) and frame generation, so they
are not a clean clock test either.

**A clean test that did not take hold (2026-11-24, owner-approved sudo).** `clockpair.sh` wrote `profile_standard` (which held the core at 1623 MHz
in round 9's Vulkan probe, `round9.md` G1) before each of four cells and `auto` before four others, alternated, at 1080p and 4K, frame generation
off (`exp/cp-*`, `cp-clockpair.log`). The sysfs readback matched the written level each time, but the core ran at 3.40-3.45 GHz (median in the benchmark window)
in the "standard" cells too: `profile_standard` did not yield a distinct sampled core clock in this test, and the files do not show why. The level was restored to `auto` and checked.
Result for the core clock: not tested. The upscaler times of the cells with the memory clock at 1408 agree within 0.8 % at 1080p (0.3487-0.3514 ms) and 0.3 % at 4K (1.663-1.669 ms), which is the run-to-run noise of this method.

## The memory clock: slow cells are associated with it (MEASURED; the cause is INFERRED)

| Cell | Core clock (window median) | Memory clock (window median) | Upscaler (ms) | Same cell type at 1408 MHz memory (ms) | Change |
|---|---|---|---|---|---|
| `cp-3840x2160-profile_standard-7` | 3.42 GHz | 1124 MHz | 1.897 | 1.663-1.669 (`cp-*` 4K) | +14 % |
| `r2-3840x2160-quality` | about 3.4 GHz | 1124 MHz | 1.948 | 1.660-1.667 (r1, r3) | +17 % |
| `r2-3840x2160-balanced` | about 3.4 GHz | 1124 MHz | 1.921 | 1.639-1.648 | +17 % |
| `r2-3840x2160-nativeaa` | about 3.4 GHz | 1124 MHz | 2.086 | 1.785-1.790 | +17 % |
| `r2-2560x1440-ultraperf` | about 3.4 GHz | 1124 MHz | 0.681 | 0.603-0.604 | +13 % |
| `r2-2560x1440-quality-fgon` | about 3.4 GHz | 1124 MHz | 0.742 | 0.648-0.651 | +14 % |

The memory clock of each cell is the median of the sampler (asked for every 10 ms; median step 11 ms, because each read takes time) inside the benchmark window; in these six cells it read 1408 in about 1-40 % of samples (about 1-25 % in the five grid cells),
in every other grid cell above 80 %. MEASURED: in these six cells the memory clock was 20 % lower (1124 against 1408 MHz) and the upscaler ran 13-17 % slower at the same core clock. INFERRED: this is
consistent with FSR 4.1.2 on this card being limited mostly by memory (bandwidth or latency) rather than arithmetic; it is a hypothesis, because the clock state was
observed and not set, the five grid cells were picked out after the run from their times, and which of bandwidth and latency is not separated.
Why the card's power management left the memory at 1124 MHz under full load in these cells, and at 1408 in all others, is open (round 7's A3 saw any
streaming load reach 1408).

## Against round 7

| | Round 7 | Round 8 |
|---|---|---|
| FSR version | 4.1.1 | 4.1.2 (signed v2.4.0 DLLs) |
| Frame generation | on in every cell | off in 47 of 56 cells; on only in the labelled Quality cells |
| Backend | gamescope with a window | gamescope headless |
| Presets | inferred from Lighting; 5 of 45 as requested | read from the log; all 56 as named; Lighting agrees |
| Core clock | about 2.3 GHz at 1080p/1440p, 3.2 GHz at 4K | about 3.4 GHz everywhere |
| Quality cost (ms) 1080p / 1440p / 4K | 0.349 / 0.643 / 1.661 | 0.346 / 0.652 / 1.663 |

MEASURED: the numbers agree within 1.5 %. What changed is the reliable presets, the full-load clocks and the frame-generation state, not the cost.
The version change (4.1.1 to 4.1.2) is not visible at this accuracy, but it cannot be isolated: the other changes were made together (INFERRED).
Round 7's clock-scaled estimates (0.22-0.30 ms at 1080p; 1.47-1.58 ms at 4K) are withdrawn: they assumed the cost is inverse to the clock.

## The timing kit (PR 57's exact rewrite on this card)

**Context (READ).** PR 57 of `zgauthier2000/fsr4-shader-performance-overrides` is an exact load-reorder rewrite of FSR 4's FP8 passes, reported -7.2 to -9.6 % for moving Balanced 4.1.2 on a
Windows 9070 XT at stable clocks (1868 MHz) (an RGP capture of all 28 dispatches, per-pass microseconds and an ISA scan are part of that PR). The project's Linux timing kit
(`fsr4-timing-kit-linux`, kit 2026-11-14) times each test DLL's upscale call with GPU timestamps, 8 rounds over 29 minutes with every other round in reverse
order, at 1080p, 1440p and 4K, still and moving, Balanced and Ultra Performance ("UP", render size a third of the output). The report holds many more test DLLs
than the two used here (their own builds, including lossy and Lite ones and other cards' reference rows); those rows are their work, not ours, and are not
quoted. Setup (READ from the report header): GE-Proton11-14, Mesa 27.0.0-devel (RADV), the card above; the owner posted the result as a comment on PR 57
on 2026-11-24.

**Result (MEASURED here; the kit's own report).** Rows `r4-amd-412-reference` (AMD's 4.1.2) against `r4-exact-412-test` (the exact rewrite), ms, mean of the DLL's runs:

| | still, reference -> rewrite | moving, reference -> rewrite |
|---|---|---|
| 1920x1080 | 0.340 -> 0.330 (-2.9 %) | 0.341 -> 0.329 (-3.5 %) |
| 2560x1440 | 0.687 -> 0.659 (-4.1 %) | 0.693 -> 0.668 (-3.6 %) |
| 3840x2160 | 1.560 -> 1.526 (-2.2 %) | 1.567 -> 1.532 (-2.2 %) |
| 1920x1080 UP | 0.331 -> 0.323 (-2.4 %) | 0.331 -> 0.326 (-1.5 %) |
| 2560x1440 UP | 0.639 -> 0.633 (-0.9 %) | 0.644 -> 0.640 (-0.6 %) |
| 3840x2160 UP | 1.490 -> 1.488 (-0.1 %) | 1.495 -> 1.490 (-0.3 %) |

The percentages are worked out here from the rounded kit figures. The rows without "UP" are Balanced (as the owner's comment states them).

- **Same picture (MEASURED, the kit's fingerprints).** The rewrite's picture fingerprint equals the reference's in all 12 rows. The report holds fingerprints only, no byte comparison; PR 57's "byte for byte" is its own claim (READ).
- **Smaller than on Windows.** MEASURED: 2.2 to 4.1 % lower in Balanced and 0.1 to 2.4 % lower at Ultra Performance, on a NOISY session. READ: PR 57 reports 7.2 to 9.6 % lower (moving Balanced, 4.1.2, stable clocks). Possible reasons (INFERRED, none tested): scheduling, the compiler (below), the clock.
- **Drift (MEASURED).** The kit flagged the session NOISY: the first DLL's first and last runs differ by -13.7 % / -24.9 % / -13.3 % (1080p / 1440p / 4K, moving) and
  -9.9 % / -24.4 % / -12.6 % at UP; the card got faster during the session. The kit alternates the order of the rounds so the drift spreads over both DLLs, but the
  percentages are less reliable than a quiet run's, and the run-to-run spread inside the kit's own rows (for example 0.331 to 0.392 ms for the reference at 1080p
  moving) is larger than the rewrite's gain at the small sizes.
- The kit ran Linux, so AMD's 4.1.2 shaders were compiled by RADV's ACO, not by AMD's Windows compiler. That is the lead into the next part (INFERRED).

## Per-step timing (the timing kit's --passes mode)

**What was run (MEASURED; folder `runs/20261125T0250Z-r8passes/`, git-ignored; run on 2026-11-24, local evening).** The project's own timing kit (`zgauthier2000/fsr4-shader-performance-overrides`, kit
2026-11-14) in `--passes` mode with `FSR4TIME_PASSESONLY=1` and `FSR4_PASSLIST=r4-amd-412-reference,r4-exact-412-test`, hidden (gamescope headless), GE-Proton11-14, the card above on the same
overclocked profile. Moving scene, Balanced, 1080p, 1440p and 4K output, 200 timed frames per DLL and size (100 in the kit's even group and 100 in its odd group, after 40 warm-up frames); the figures are the mean of the even and the odd group. **READ** (the kit's `fsr4time.c`, lines 589-640): it replaces
`Dispatch` in the command list's function table and writes a GPU timestamp before and after every dispatch the DLL records. The kit reports, per dispatch, the mean time of the dispatch and the mean time from
its end to the next dispatch's start (the "gap"). For the last dispatch (`fsr4time.c:774`) the gap runs from its end timestamp to the call-end timestamp. The interval from the call's start to the first dispatch's start is not printed, so the dispatch times plus all the gaps span the first dispatch's start to the call end, which is what the totals below are labelled. Every number below comes from `analyse_passes.py` in that folder (standard library, `python3 -I`; output in `analyse_passes.out`).
The kit saw 28 dispatches in every cell, with the same reported group counts for both DLLs and both frame groups. The kit waits on a fence after every call.

**Limits.** One run. The only repeat is even against odd frames. The card is overclocked (+150 MHz, -50 mV, raised cap), so microseconds describe this card in that state. The added timestamp queries may perturb GPU scheduling (the kit's own source says they may disturb the totals), so these shares describe the instrumented run, not the normal run. The core clock during this run is unverified.

**Which dispatch is which pass (INFERRED).** The DLL gives no names. The reading used here: dispatch 1 is the prepass, dispatch 27 the postpass, the odd dispatches 3, 5, ..., 25 are model passes 1 to 12 in order,
the even dispatches 2, 4, ..., 26 are the 13 small "post" shaders, and dispatch 0 is a small first dispatch (the extra small shader of "Inside"; small in code, but it takes 10.6 to 63 us and grows with the output size). Evidence: the order; dispatches 1 and 27 have the full output size as
their grid (30x540 groups at 1080p, 40x720, 60x1080); the 13 even-index slots d2 to d26 are one-row grids of 47-94 groups at 1080p, 214-848 at 1440p and 47-188 at 4K, with times of 0.15-0.17 us each; and the inside capture has 28 shaders (12 model + 2 pre/post + 14 small).
It is not a measurement of which shader ran.

**Totals against the kit's normal run (MEASURED, reference).** The normal figure is the kit's whole-call mean from report DE4301C9 (moving Balanced, `r4-amd-412-reference`, READ from the report). The totals here span the first dispatch's start to the call-end timestamp and leave out the interval before the first dispatch.

| output | dispatches | gaps | first dispatch to call end | normal kit whole call | difference |
|---|---|---|---|---|---|
| 1920x1080 | 301.4 us | 29.7 us | 331.1 us | 341 us | -2.9 % |
| 2560x1440 | 635.1 us | 38.0 us | 673.1 us | 693 us | -2.9 % |
| 3840x2160 | 1498.3 us | 34.0 us | 1532.4 us | 1567 us | -2.2 % |

Gaps are 9.0 %, 5.6 % and 2.2 % of that interval; the last dispatch's trailing gap is 3.4, 4.8 and 4.1 us of them. They include (INFERRED) the cost of the timestamps themselves and any barrier between steps. **The totals come out 2-3 % below the whole-call figure; the cause of the difference is not separated.** Not separated (none tested): the interval before the first dispatch, which the rows do not hold; the two runs' different clock and temperature states (the normal session was flagged NOISY); and the effect of the timestamps.

**Shares of the dispatch time (MEASURED numbers; their names INFERRED).** Reference, mean of even and odd.

| | 1080p | 1440p | 4K |
|---|---|---|---|
| d1, prepass | 66.3 us, 22.0 % | 212.3 us, 33.4 % | 426.5 us, 28.5 % |
| d27, postpass | 77.3 us, 25.7 % | 166.9 us, 26.3 % | 468.7 us, 31.3 % |
| d1 + d27 | 47.7 % | 59.7 % | 59.7 % |
| the 12 odd dispatches 3-25 (model passes) | 145.1 us, 48.1 % | 227.2 us, 35.8 % | 538.1 us, 35.9 % |
| d0 | 10.6 us, 3.5 % | 26.6 us, 4.2 % | 63.0 us, 4.2 % |
| the 13 small dispatches 2-26 | 2.10 us, 0.70 % | 2.04 us, 0.32 % | 2.04 us, 0.14 % |

The shares add to 100 %. At 4K the 12 odd dispatches in dispatch order are 94.1, 63.6, 25.1, 32.2, 32.3, 14.7, 25.5, 25.6, 46.6, 31.9, 79.7 and 66.8 us. The two full-size dispatches carry the most time and their share
rises with the output size from 1080p to 1440p; the 12 odd dispatches together take 36 % of the dispatch time at 1440p and 4K.
PR 57 reports 93 us each for passes 1, 2 and 12 and equal metrics for them (READ; equal metrics do not establish identical shader code). Here the slots inferred as those passes, d3, d5 and d25, took 94.1, 63.6 and 66.8 us at 4K. At 1080p and 1440p the three are close (18.6, 18.5, 18.1 us and 29.9, 29.5, 28.9 us), with d3 1 to 4 % slower than the others, so the large gap appears only at 4K. The three slots inferred as passes 4, 5 and 10 (32.2, 32.3, 31.9 us) are close to one another, as are the two inferred as 7 and 8 (25.5, 25.6 us). Whether the d3 slot is mislabelled, or the same code takes different time with different input, is not known (INFERRED mapping; compare the open question about `pass1` in "Inside").

**Gaps (MEASURED).** Mean gap after a small dispatch against after the other 14 (the last dispatch's trailing value left out): 0.96 against 0.98 us at 1080p, 0.59 against 1.82 us at 1440p, 0.41 against 1.76 us at 4K. The gap sums are
12.5 / 7.7 / 5.3 us after the 13 small ones and 17.1 / 30.3 / 28.8 us after the other 15 (trailing value included), at 1080p / 1440p / 4K. So at 1440p and 4K the gap after a small dispatch is a quarter to a third of the gap after a big one; at 1080p they are alike. Why is not tested.

**The exact rewrite against the reference (MEASURED; one run).**

| | dispatch sum | gaps | first dispatch to call end | the kit's normal run (moving Balanced) |
|---|---|---|---|---|
| 1920x1080 | 301.4 -> 297.9 us (-1.1 %) | 29.7 -> 30.9 | 331.1 -> 328.9 us (-0.66 %) | -3.5 % |
| 2560x1440 | 635.1 -> 630.2 us (-0.8 %) | 38.0 -> 37.6 | 673.1 -> 667.9 us (-0.78 %) | -3.6 % |
| 3840x2160 | 1498.3 -> 1499.6 us (+0.1 %) | 34.0 -> 34.3 | 1532.4 -> 1533.9 us (+0.10 %) | -2.2 % |

Per dispatch, the biggest single change is d1 at 1080p (-3.0 us, -4.5 %) and 1440p (-4.1 us, -1.9 %), which is the prepass on the inferred reading; at 4K d1 changes by -1.2 us. Among the 12 odd dispatches the changes are mostly within +-1 %; the clearest
is d13 (pass 6 on the inferred reading): -0.5 us (-7.3 %) at 1440p and -1.3 us (-8.6 %) at 4K, but +0.2 us (+3.3 %) at 1080p. d19 (pass 9, inferred) moves -4.4 %, -1.7 % and +1.2 % at the three sizes. At 4K ten of the 12 odd dispatches are
slightly slower (+0.3 to +1.2 %). So the rewrite shows a gain in the normal run, a smaller one here at 1080p and 1440p and none at 4K. Possible reasons (INFERRED, none tested): the added timestamps may perturb scheduling; the order of the
two DLLs in the run and any drift were not separated; one run. The single instrumented run had total changes of -0.66 %, -0.78 % and +0.10 %, and the earlier noisy normal-run means changed by -3.5 %, -3.6 % and -2.2 %; the measurements do not isolate a cause. The 13 small dispatches took 2.04 to 2.10 us combined in both DLLs.

**Even against odd frames (MEASURED).** The first-dispatch-to-call-end totals differ by 0.35 / 0.09 / -0.04 % (reference) and -0.91 / +0.44 / +0.03 % (rewrite) at 1080p / 1440p / 4K. For the 15 non-small dispatches the mean absolute even-odd difference is 0.86 / 0.24 / 0.21 % (reference) and
0.98 / 0.39 / 0.10 % (rewrite); the largest single difference among those 15 is 6.99 % (reference) and 4.74 % (rewrite), both d0 at 1080p, and 0.97 % (reference) and 2.09 % (rewrite) at 1440p, 0.52 % and 0.25 % at 4K. Over all 28 dispatches the largest even-odd difference per DLL is 6.99 / 6.06 / 6.45 % (reference) and 6.45 / 6.45 / 6.06 % (rewrite) at 1080p / 1440p / 4K. The 13 small dispatches differ by about 6.5 % at a 0.01 us print step (for example 0.15 against 0.16 us). At 4K the 12 odd dispatches differ between even and odd frames by at most 0.40 % (reference) and 0.25 % (rewrite); the largest reference-to-rewrite change among them is d13 (-8.6 %), and the other 11 (+0.3 to +1.2 %, and -0.6 % for d15) each exceed that slot's even-odd difference in either DLL. The two frame groups come from the same run and cannot show variation between runs.

**Against PR 57's per-pass table (READ; ratios and shares only).** PR 57's table (research/rdna4/README.md at commit 470b6c3c) is AMD 4.1.1 at 4K, 1868 MHz stable, from an RGP capture on Windows. Ours is 4.1.2 on the overclocked card, Linux with RADV's ACO compiler, and timestamp queries around dispatches instead of RGP; the core clock during this run is unverified. So absolute microseconds are not comparable (here/PR 57 is 0.41-1.01 per pass), and only the shares of the 11 passes PR 57 lists (pass 3 is not in its table; mine is 25.1 us) are compared:

| pass (inferred slot) | PR 57 us | share of the 11 | here us | share of the 11 | share here / share PR 57 |
|---|---|---|---|---|---|
| 1 | 93 | 12.0 % | 94.1 | 18.3 % | 1.53 |
| 2 | 93 | 12.0 % | 63.6 | 12.4 % | 1.03 |
| 4, 5, 10 | 50-52 (51 used) | 6.6 % each | 32.2, 32.3, 31.9 | 6.3, 6.3, 6.2 % | 0.94-0.96 |
| 6 | 35 | 4.5 % | 14.7 | 2.9 % | 0.64 |
| 7, 8 | 35 | 4.5 % each | 25.5, 25.6 | 5.0 % each | 1.10 |
| 9 | 114 | 14.7 % | 46.6 | 9.1 % | 0.62 |
| 11 | 123 | 15.9 % | 79.7 | 15.5 % | 0.98 |
| 12 | 93 | 12.0 % | 66.8 | 13.0 % | 1.08 |

INFERRED: passes 9 and 6 (slots d19 and d13), the two PR 57 marks with serialized waits at 113 and 39, are the two passes that are relatively fastest here (shares 0.62 and 0.64 of PR 57's). Passes 4, 5, 10 and 11, the others with waits in PR 57's table, sit at 0.94-0.98, so the pattern is not
clean. It depends on the inferred mapping (and the d3 slot disagrees with the d5 and d25 slots), and it does not show the cause. The differences in version, clock, OS and driver, and tool listed above could each move these shares.

## Inside: compile statistics and PR 57's scan

**Method (MEASURED).** One captured sample run per output size, on the real card, with vkd3d's shader dump and RADV's compile statistics switched on; the SPIR-V
dumps and the compiled code were read on this machine, CPU only, and are not published. Identity of each shader: the name vkd3d gives the dump file and, independently,
the hash of the dumped SPIR-V; they agree wherever both survive. A parser fault (RADV prints from several compile threads without a lock, and each shader's
statistics follow the next shader's header) had caused a mismatch in an earlier attempt; it was fixed and all 28 FSR 4 shaders map one to one at every size.
The 1440p and 4K dumps are byte-identical; the 1080p ones differ only in file name and 0-9 constants per shader; the compile statistics are identical across the
three sizes except for 4 bytes of code in four shaders (1080p smaller), so one table serves all sizes. Four of the 84 shader-size records (`pass4_post`, `pass5` and `pass7_post` at 1440p; `pass6_post` at 4K) took their statistics from the other size's record because the log lost their identity keys (`RESULTS-inside.md`); that rests on the identical dumps. Names follow creation order and are less sure for the second of two
shaders with identical statistics (the table in `RESULTS-inside.md` rates each).

**Per-pass compile table (counts only; MEASURED, RADV's own numbers).** Waves per SIMD is RADV's static wave limit, not achieved occupancy; it equals the study's rule
(`measurements/memory-system.md`) for all 28 shaders. No spills and no scratch anywhere; no backward branch in any model pass.

| pass | VGPRs | WMMA | waves/SIMD | instructions | vector loads |
|---|---|---|---|---|---|
| prepass | 96 | 4 | 16 | 1808 | 109 |
| 1, 2, 12 | 192 | 104 | 8 | 779 / 804 / 804 | 65 |
| 3 | 72 | 16 | 16 | 239 | 24 |
| 4, 5, 10 | 192 | 100 | 8 | 772 | 77 |
| 6 | 96 | 64 | 16 | 547 | 64 |
| 7, 8 | 240 | 164 | 6 | 1081 | 166 |
| 9 | 120 | 114 | 12 | 1099 | 206 |
| 11 | 96 | 33 | 16 | 489 | 79 |
| postpass | 96 | 30 | 16 | 1558 | 92 |
| 13 small shaders (`pass1_post` to `pass13_post`) | 24 | 0 | 16 | 116-128 | 0 |

- **Native FP8 matrix instruction in all 12 model passes (MEASURED):** the WMMA count equals the SPIR-V's cooperative-matrix multiply-add count in each (104, 104, 16, 100,
  100, 64, 164, 164, 114, 100, 33, 104). The prepass has 4 and the postpass 30 against 27 in the SPIR-V.
- All 28 shaders are 32 wide (subgroup size 32 in their NIR print). Counted as 12 model + 2 pre/post + 14 small = 28, which matches PR 57's 28 dispatches (READ).
- The heavy passes (192 and 240 VGPRs: passes 1, 2, 4, 5, 7, 8, 10 and 12) have static limits of 6 to 8 waves per SIMD (RADV's own limit, not achieved occupancy) and
  65 to 166 vector loads each. Pass 9 (120 VGPRs, static limit 12 waves) has the most loads, 206. The loads are issued in clauses. INFERRED: that is how a
  latency-bound pass hides loads; not tested.

**The scan against PR 57 (MEASURED here; Windows counts READ from PR 57, where the pass 9 figure of 110 is its 4.1.2 example and the other five come from its 4.1.1 table).** PR 57's `scan_isa.py`, unchanged, counts a "serialized wait": a wait for a counter at zero that
retires at most two loads with fewer than 24 instructions between issue and wait. It was run on ACO's output for every shader at every size. The scanner prints a count line only when it is above zero, so a zero is the absence of that line: each of the 36 model-pass rows was rechecked with an exit status of 0, a parsed instruction count and a parsed load count, so none is a missing result. Control: on pass 9's real code
it gives 0; with a forced wait at zero after every load it gives 206 of 206 loads; it also fires on real code elsewhere (prepass 4, postpass 3, one small shader 1), so it reads
ACO's text format.

| pass | Windows compiler (PR 57; pass 9 from the 4.1.2 example, the rest from the 4.1.1 table): serialized waits | ACO, Linux (here) | PR 57 VGPRs | ACO VGPRs |
|---|---|---|---|---|
| 9 | 110 | **0** | 90 | 120 |
| 11 | 33 | **0** | 51 | 96 |
| 6 | 39 | **0** | 79 | 96 |
| 4, 5, 10 | 24 each | **0** each | 189 | 192 |
| 1, 2, 12 | 0 | 0 | 167 | 192 |
| 7, 8 | 0 | 0 | 209 | 240 |

- MEASURED: by this scanner's definition, ACO's code has no serialized load wait in any model pass (36 of 36 rows). In pass 9 the 206 loads go out in clauses and are waited for by 80 wait instructions, 77 of them
  partial counts (above zero), so loads overlap with matrix work. The other model passes look the same.
- INFERRED: the Windows compiler leaves loads next to their waits in passes 4, 5, 6, 9, 10 and 11 (PR 57's counts for 4.1.1, and 4.1.2 for pass 9) and ACO does not, so a source-level load reorder has little left to fix
  in ACO's output; that fits 2.2 to 4.1 % here (Balanced) against PR 57's 7.2 to 9.6 % on Windows. It does not prove the cause: the driver, the clock and the code length also differ (pass 9 is 1099
  instructions under ACO against 1079 in PR 57's Windows listing).
- **More registers, lower static wave limit for pass 9 (MEASURED against READ; PR 57's registers are its 4.1.1 table).** ACO's counts are multiples of 12 and PR 57's are raw, so gaps up to about 12 are rounding. Pass 9 uses
  120 VGPRs (12 waves per SIMD) against PR 57's 90 (16 waves); passes 1, 2, 12 and 7, 8 use more as well (192 vs 167: 8 against 9 waves; 240 vs 209: 6 against 7); passes
  6 and 11 (96 vs 79 and 51) still give 16 waves; passes 4, 5, 10 are close (192 vs 189, 8 waves both). Whether pass 9's lower static wave limit costs anything is not measured.
- The scan is a static order check, not a timing, and it ignores one combined wait form (2 in the prepass, none in a model pass).

**The 28th shader (INFERRED, from structure).** `pass0_post` is unlike the other 13 small shaders: 2 bindings against 3, 27 instructions against 116-128, one load and one
store, and the only small shader with a loop. It was created 144 ms before the prepass while the other 13 each appear next to a model pass. So the docs' "13 post shaders" are the 13
three-binding ones and this extra small kernel is probably the 28th dispatch in PR 57's capture. Where it runs in the frame cannot be told without a dispatch trace.

**Other inside findings (MEASURED, with the first one READ from the machine code).**
- Prepass and postpass stores are rows of 32 consecutive pixels per store instruction (each store takes its x from the lane id; READ from the instruction listing). How the memory
  system lands them (full 64- or 128-byte lines, or not) is not shown and needs counters.
- 0 backward branches in every model pass, the prepass and the postpass (MEASURED), though their SPIR-V holds 3 to 25 loops each. The count shows that no loop is left in
  the compiled code; it cannot show that every source loop survived to be unrolled (some may have been removed or folded away).
- Pass identities: 3, 6, 9, 11 are sure (unique WMMA counts; pass 9 equals the 114 in PR 57's own example); which of two passes with identical statistics is which (7/8; 4/5/10;
  1/2/12) rests on creation order; `pass1` compiles to 779 instructions against 804 for `pass2` and `pass12`, although PR 57 says the three share code on Windows, so either the
  Linux path differs slightly for one or our `pass1` is not PR 57's pass 1.

## What is open

- **Per-step timing is done once, not settled.** The kit's `--passes` mode (see "Per-step timing") gives a time per dispatch, but from one run, and the names of the dispatches
  are inferred. RADV's GPU tracepoints are still compiled out of this Mesa build and no RGP viewer is installed, so there is no capture like PR 57's here; nothing
  yet shows which dispatch is which pass, why the first-dispatch-to-call-end totals sit 2-3 % under the kit's normal whole-call figure, or why the times of d3, d5 and d25 differ (below). Everything in the "inside" part is
  compile-time structure, not time.
- **Store density:** whether the 32-pixel rows reach memory as full lines needs cache counters.
- **Which post shader follows which pass, and where the extra small shader runs:** the per-step timing is consistent with the extra small shader being the first dispatch (d0), but only by order and grid size (INFERRED); a trace that names each dispatch's shader would settle it. The identity of pass 1 against PR 57's is unsettled.
- **AMD's Windows compiler is not here:** the comparison is ACO's code against PR 57's published counts, not two compiles of the same input.
- **The disturbed grid cells (an association found for five; one unexplained).** One cell, `r1-1920x1080-quality-fgon`, was flagged by RasterShadow (1.11x its size's median); its memory clock was
  1408 and its cause is unknown. Five cells that the RasterShadow rule misses ran 13-17 % slower in the upscaler than their other repeats: `r2-3840x2160-balanced`,
  `-nativeaa`, `-quality`, `r2-2560x1440-ultraperf`, `r2-2560x1440-quality-fgon`; they ran back to back (about 20:00-20:03 UTC) with the core clock, activity and
  temperatures normal, and with the memory clock at 1124 MHz instead of 1408 (lead's check, 2026-11-24; see "The memory clock"; an association, the cause is INFERRED). Why the memory stayed at 1124 is open. A sixth slow cell at 1124 MHz is the clock-pair cell `cp-3840x2160-profile_standard-7`. Four of the six disturbed grid cells are left out of the frame-generation-off headline table and two out of the frame-generation-on table; the second rule ("more than 5 % above the other repeats")
  was added after seeing the data. The run's driver script was stopped at 19:48 UTC and restarted at about 19:53; `ref-first` ran before it, every other cell after.
- RCAS is on (the sample's default; not a separate label), so the upscaler time may include sharpening, which AMD's figures leave out. The headless gamescope path may differ from
  native. n is 2 to 3 per cell type. The meaning of AvgFPS under frame generation is not settled. The cause of round 7's low clocks is not isolated (see "The pilot").
