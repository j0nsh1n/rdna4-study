# FSR 4 predictions, written before the measurement

Written 2026-11-19 (UTC), before any run with a real game and FSR 4. The commit time of this file is the evidence
that these predictions came first. They are outside-view predictions: no GPU job ran for them, and nothing here says
how FSR 4 works inside beyond AMD's public text. Every number is either copied from a named source or worked out in
the arithmetic below. INFERRED means worked out, not measured.

## AMD's numbers (READ)

Source: AMD FSR Upscaling 4.1.2 manual, "Performance" section,
<https://gpuopen.com/manuals/fsr_sdk/techniques/super-resolution-ml/#performance>, opened and read as a page on
2026-11-19. The text above the table says: "reference performance numbers for running FSR4 in performance mode for
recommended target resolution", "measured with RCAS sharpening disabled and peak GPU clocks", and that the target
resolution mostly sets the cost. Performance mode is a 2.0x per-dimension scale (the same page's scaling table).

| GPU | ML model | Target resolution | Microseconds |
|---|---|---|---|
| AMD Radeon RX 9070 XT | Full FP8 | 3840x2160 | 1316 us |
| AMD Radeon RX 9070 XT | Full FP8 | 1920x1080 | 352 us |

There is no 1440p row for this card. The same page lists a 4K working set of 318 MB (rounded, measured on an
RX 9070 XT, "subject to change").

## Conditions: AMD's against tonight's

| | AMD's table | Tonight |
|---|---|---|
| Card | RX 9070 XT, stock | the same card model on the overclock profile "Default" (+150 MHz, -50 mV, 374 W limit) unless changed |
| FSR version, mode | 4.1.2, performance mode, sharpening off | whatever the game ships; unknown until the game is running |
| Core clock | "peak GPU clocks" (no number given) | the card's own state: about 1.65 to 1.80 GHz at the low state, about 3.1 to 3.5 GHz at the full state (`gaming-route.md`, `gaming-probes.md`; `matrix-timing.md` read 3.2 GHz under a matrix load) |
| Memory clock | not stated | 1258 MHz, unless the displays are blanked first (earlier jobs ran at 1408 MHz) |
| Work | FSR 4 alone | FSR 4 inside a real game frame, with the game's own work around it |

So AMD's times are the best-case time for the full state. Tonight's full state could be a few percent faster than
AMD's (higher clock, offset) or slower (shared with game work). I do not know AMD's clock, so I do not correct for it.

## Prediction 1: cost per frame

**Full clock.** Interpolate only between AMD's own two rows, linear in pixels (INFERRED):
- 4K = 8.294 MP, 1080p = 2.074 MP, 1440p = 3.686 MP.
- Slope = (1316 - 352) / (8.294 - 2.074) = 155.0 us per MP; fixed part = 352 - 155.0 x 2.074 = 30.7 us.
- 1440p: 30.7 + 155.0 x 3.686 = 602 us. Pure scaling from the 1080p row instead gives 352 x 3.686 / 2.074 = 626 us.

Prediction: **1440p about 0.60 to 0.63 ms, 4K about 1.3 ms** (AMD's 1.316 ms, with a few percent of room either way
for the clock difference above).

**Low clock.** The study's own timings of a different, upscaler-style network on this card: a frame took 2.8 ms at
about 1.7 GHz and 1.6 ms at full clock, a ratio of 1.76 (`gaming-route.md`). The stages of that network followed the
clock by 1.62 to 1.84 times, while one small kernel followed it by 1.05 times. That network is not FSR 4, so this is
only a guide to how much a clock-bound pass slows. If FSR 4 is clock-bound, it slows by 1.6 to 1.85 times; if
memory-bound, by close to 1 (memory clocks did not change with the core clock in that test).

| Output | Full clock | Low clock, ratio 1.62 to 1.84 (middle 1.76) |
|---|---|---|
| 1440p | 0.60 to 0.63 ms | 0.98 to 1.1 ms (middle 1.06) |
| 4K | 1.3 ms | 2.1 to 2.4 ms (middle 2.3: 1316 x 1.76 = 2316 us) |

Prediction: **the low-clock cost is 1.6 to 1.85 times the full-clock cost.** A ratio near 1 would mean FSR 4 is
memory-bound or latency-bound, which I think is unlikely for an FP8 network.

## Prediction 2: share of a 60 fps frame

A 60 fps frame is 16.67 ms.

| Output | Full clock | Low clock |
|---|---|---|
| 1440p | 0.60 / 16.67 = 3.6 % | 1.06 / 16.67 = 6.4 % (5.9 to 6.6 %) |
| 4K | 1.316 / 16.67 = 7.9 % | 2.13 to 2.42 / 16.67 = 12.8 to 14.5 % |

This is the share on the engine that runs it, if it ran alone. If other work overlaps it, the frame grows by less.

## Prediction 3: upper bound on matrix work and memory traffic that fits in AMD's 4K time

**FP8 matrix work.** The study measured the dense FP8 rate at 419.5 TFLOPS at a 3.201 GHz counter clock
(`matrix-timing.md`, section 11 table, W = 1), which is 131.1 TFLOPS per GHz (419.5 / 3.201). At that clock, 1.316 ms
holds at most 419.5 TFLOPS x 1.316 ms = **0.55 TFLOP**, which is 66,600 flops (33,300 multiply-adds) per 4K output
pixel (0.552e12 / 8.294e6). This is a ceiling. It assumes every cycle of the 1.316 ms issues dense matrix
instructions with data already in registers. Real kernels do far less: the study's best hipBLASLt FP8 GEMM reached
170 TFLOPS, 41 % of the ceiling (`matrix-timing.md`), and AMD's time also covers non-matrix passes. A realistic
figure is therefore 0.2 TFLOP or less; the 0.55 TFLOP is what cannot be exceeded. If the clock is lower, the ceiling
falls in proportion (131.1 TFLOPS per GHz).

**Memory traffic.** The measured read rate is 633 GB/s of a 640 GB/s bus limit at 1258 MHz (`memory-system.md`).
633 GB/s x 1.316 ms = **0.83 GB**, about 100 bytes per 4K pixel (0.833e9 / 8.294e6 = 100). From the memory-side
cache, the study saw about 2.0 TB/s up to 64 MiB (`memory-system.md`), so traffic that stays in the cache could be as
high as 2.0 TB/s x 1.316 ms = 2.6 GB; I treat that as a loose ceiling because it needs the data to fit in 64 MiB.
For comparison, AMD's 318 MB working set read once is 38 % of the 0.83 GB DRAM ceiling, so the memory ceiling is not
tight and does not by itself decide whether FSR 4 is memory-bound.

**Power.** At 0.76 pJ per flop for FP8 in reported power (`scheduling-and-clocks.md`), 0.55 TFLOP is at most 0.42 J
per frame, or 25 W averaged at 60 fps. During the 1.3 ms burst the same work would read about 319 W (0.42 J / 1.316
ms), but the reported power is a smoothed value and the real work is below the ceiling, so I expect no visible 25 W
step in a game that draws 150 W or more.

## Prediction 4: what an outside view can and cannot tell apart

Outside view means: per-process engine busy time (the kernel's per-client counters), frame rate, the 20 ms clock and
power sampler, with FSR 4 on and off at the same input resolution.

**It can tell:**
- the extra busy time per frame when FSR 4 is on, to within the repeat noise; I predict 0.6 to 0.63 ms at 1440p and
  about 1.3 ms at 4K at the full state, and 1.6 to 1.85 times that at the low state;
- which clock state the game sat in (the clock reading in each window), and so which row of the tables above applies;
- whether the frame rate changes by about that amount when the game is limited by the GPU;
- whether the engine busy time follows the clock between two states (a ratio of about 1.76 would fit a clock-bound
  pass).

**It cannot tell:**
- which part of the extra time is matrix work, vector work or memory traffic: the busy counter is one number per
  engine;
- whether the extra time is FSR 4 alone or also changed game work (the game may react to the setting);
- anything inside a frame: the sampler runs at 20 ms and the pass lasts about 1 ms;
- whether a reading of 25 W more is FSR 4, because the power reading is smoothed and noisy against a game's load;
- whether a capped game (60 fps) hides the cost: with idle time left, the frame rate does not move.

A result that would break these predictions: an on-minus-off busy time more than 30 % away from the table at the
state seen, or a clock ratio near 1 between states. The first would say the page's numbers do not carry over to a
game on this profile; the second would say FSR 4 is not clock-bound.
