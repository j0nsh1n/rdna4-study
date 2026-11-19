# First measurements on the RX 9070 XT

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201). ROCm 10.1.0 (HIP 7.16, clang 24) in `~/.local/opt/rocm`, kernel
7.3.4, Mesa 27.0.0-devel. Run folder: `runs/20261117T101216Z-gfx1201/` (script `scripts/first-hour.sh all`, commit
e766c3c). Everything below is MEASURED in that one run unless marked otherwise. One run, 3 rounds or 3 repetitions
per item; treat small differences as within spread. The memory clock was at its top level (1258) throughout; the core
clock was not pinned (performance level "auto", profile "BOOTUP_DEFAULT").

The card is overclocked: a LACT profile (core clock offset +150 MHz, -50 mV, power cap at the driver's maximum)
and regulator-level power-limit mods applied with `rdna4p.iso` at 1.2x (owner, 2026-11-17). Every number measured
in seconds below (GB/s, tokens/s, TFLOPS) is for that configuration, not a stock card; the driver's power reading is
scaled down on the main rail and is not true board power. Cycle counts are not affected.

The RX 7900 XT numbers for comparison are from 2026-11-16 on ROCm 7.1.1 (`rdna3-fp8/build/latest-runs/`) and from
the RDNA 3 write-up. Different ROCm version and compiler, so a difference between the cards is not purely the card.

## In short

- The documents were right about the matrix instructions: 16 cycles for FP16 and BF16, 8 cycles for FP8 and 8-bit
  integers, and other vector work does not overlap with them.
- Memory reads reach 99 % of the card's bus (measured on all-zero buffers), and our exact decode prototype reads
  random weights at 96 % of the bus for one stream.
- The RDNA 3 NaN bug does not exist on this card.
- With our RDNA 3 patch, FP8 generation is 8 % faster than Q8_0 and 1.9 times faster than unpatched FP8.
- Prompt processing is where FP8 now loses: about 27 % slower than Q8_0, because llama.cpp gives Q8_0 a matrix
  instruction kernel on this card and sends FP8 through a slow FP16 library call.
- AMD's FP8 matrix library is fast here: 170 TFLOPS against 37 for the FP16 routine llama.cpp's FP8 prompt path
  uses. That 37 is a slow route; PyTorch's FP16 matmul reaches 123, so FP8 hardware itself is worth about 1.4x.
- The hardware FP32 to FP8 conversion is exact, with two things to know: every NaN becomes the code 0xFF, and
  infinity is not clamped even when the saturation mode is on.

## Card and memory

| | RX 9070 XT | RX 7900 XT |
|---|---|---|
| Read bandwidth, 1 GiB buffers | 633.7 GB/s (630.7 to 634.2) | 754 GB/s |
| Copy bandwidth | 550.4 GB/s (546.9 to 550.7) | 603 GB/s |
| Share of the theoretical bus (640 / 800 GB/s) | 99 % | 94 % |

## Matrix instructions (`wmma_time`, one wave alone on its SIMD, cycle counter)

| Instruction | Cycles each, chained | AMD's figure |
|---|---|---|
| FP16 x FP16 -> FP32 | 16.11 | 16 |
| BF16 x BF16 -> FP32 | 16.11 | 16 |
| FP8 x FP8 -> FP32 | 8.12 | 8 |
| 8-bit integer | 8.12 | 8 |
| One independent vector FMA (calibration) | 1.08 | 1 |

Overlap test: adding k vector FMAs per matrix instruction adds their full cost (FP16: 16.1, 22.1, 25.7, 33.7, 50.7
cycles for k = 0, 4, 8, 16, 32; FP8: 8.1, 15.3, 18.2, 26.2, 43.0). The overlap fraction is 0.00. So on RDNA 4, as on
RDNA 3, vector work and matrix work take turns inside one wave. Widening FP8 weights in registers costs time on top
of the matrix instruction; it cannot hide inside it. (Not tested here: a second wave on the same SIMD doing the
vector work.)

## What the matrix instruction computes (`wmma_exact`)

- The identity-matrix check confirms rows, columns and the accumulator layout for FP16, BF16 and FP8, and the full
  8-bit operand layout. It cannot tell the order of the K values inside a lane for 16-bit data, where AMD's guide
  and AMD's sample code differ; that order is still undetermined.
- No simple model reproduces the results bit for bit. Best fits, with the sample-code K order and (in brackets) the
  guide's K order: FP16 "groups of 4" 51 % (56 %) of random outputs and 79 % (82 %) of adversarial ones; BF16
  groups of 4, 76 % (87 %) and 90 % (96 %); FP8 groups of 8, 89 % of random and 100 % of adversarial outputs. The
  FP8 adversarial figure does not discriminate: an exact sum also scores 99 % on that set.
- So, as on RDNA 3, results are deterministic but not equal to an exact sum rounded once, and an "exact" kernel can
  only be exact up to the instruction's own rounding. How large the differences are has not been measured yet, and
  "deterministic" has not been tested by repeating runs. The probe's PASS only asserts the layout check.

## FP8 conversions (`cvt_fp8`, all inputs)

- FP8 to FP32: exact for all 256 codes. The byte select is plain: select 0 reads byte 0, 1 reads byte 1, and so on;
  the packed form converts the low byte first. (The guide's description of the select looked reversed; the builtin
  is not.)
- FP32 to FP8, over all 2^32 inputs, in both slots: 0 mismatches for ties, subnormals, zeros, values above 448 and
  infinities in the default mode. The only difference from our reference: every NaN becomes 0xFF, where the
  reference kept the sign (0x7F for positive NaN). Both are NaN codes. The probe reports FAIL because of that; the
  reference should accept either NaN code.
- With the saturation mode bit set, finite values above 448 saturate to 448 as documented, but plus and minus
  infinity do not (4 mismatches: 2 values in 2 slots). This confirms Mesa's comment that the guide is wrong on this
  point. Clamp before converting when saturation is wanted.

## Decode prototype (exact, FP8 weights in memory, 16384 x 14336 weight)

Weight bytes read per second. Variant a keeps FP32 activations (exact); variant b uses FP16 activations.

| Columns | a, best rows-per-wave | b, best rows-per-wave | Share of 640 GB/s (a) |
|---|---|---|---|
| 1 | 614.8 GB/s | 608.9 | 96 % |
| 4 | 499.1 | 604.4 | 78 % |
| 8 | 447.1 | 463.3 | 70 % |

The correctness modes pass (plain, NaN rows, overflow fallback), but they were run for one setting only (2 rows per
wave, 4 columns); the timed settings were not checked for correctness. The weight (235 MB) is slightly below the
256 MB cache-trap size the script uses elsewhere. One setting is bad: variant a with 1 row per wave
at 8 columns reads only 130 GB/s. This is the unchanged RDNA 3 prototype; nothing was tuned for this card.

## llama.cpp, Llama 3.2 3B (master f477c1b90, ROCm 10.1)

| | Prompt (pp512), tokens/s | Generation (tg128), tokens/s | Perplexity, 4 chunks |
|---|---|---|---|
| FP8, unmodified | 8536 +- 181 | 76.8 +- 1.6 | 9.7049 (default batch), 9.7042 (batch 8) |
| FP8, with our RDNA 3 patch | 9051 +- 153 | 146.0 +- 0.8 | 9.7037, 9.7041 |
| Q8_0, unmodified build | 12123 +- 1094 | 134.7 +- 0.5 | 9.7140 |
| Q8_0, patched build | 12478 +- 547 | 134.8 +- 0.8 | 9.7140 |

- `test-backend-ops -p f8_e4m3`: 448 of 448 pass on both builds.
- No NaN at the default batch: the RDNA 3 bug is absent, as predicted from the source.
- Generation: patched FP8 is 8 % faster than Q8_0 and 1.9 times faster than unpatched FP8. The decode kernel from
  RDNA 3 carries over without changes.
- Prompt: FP8 is 27 to 30 % slower than Q8_0. Q8_0 uses llama.cpp's RDNA 4 integer matrix kernel; FP8 is converted to
  FP16 and sent to hipBLAS. Our RDNA 3 prompt kernel is not built for this card. This is the gap to close.
- Against the RX 7900 XT (ROCm 7.1.1, yesterday): Q8_0 generation 134.7 against 148.3 (0.91x, the bus is 0.8x);
  Q8_0 prompt 12123 against 7332 (1.65x); patched FP8 generation 146.0 against 163.3 (0.89x).

## Libraries (4096 x 4096 x 512)

| Path | Speed | Check |
|---|---|---|
| hipBLASLt FP8 x FP8 -> FP32 | 170.1 TFLOPS | matches the CPU reference (3.0e-07) |
| hipBLAS FP16 GEMM (what llama.cpp's FP8 prompt path uses) | 37.3 TFLOPS | |
| `torch._scaled_mm` FP8 | 166.1 TFLOPS | matches (2.1e-08) |
| PyTorch FP16 matmul | 123.2 TFLOPS | |

`torch._scaled_mm` works on this card; on the RX 7900 XT it raised an error. The PyTorch used is 2.13.0+rocm7.1 with
its own bundled ROCm 7.1 libraries.

## Predictions from the research, checked

| Prediction | Result |
|---|---|
| FP16 matrix instruction 16 cycles, FP8 8 cycles | confirmed |
| Matrix and vector instructions do not overlap | confirmed |
| Operand layout as in the guide | confirmed |
| No NaN bug on RDNA 4 | confirmed |
| Infinity not clamped in saturation mode | confirmed |
| Guide's byte-select order reversed | not seen; the builtin is in plain order |
| One-stream generation about 0.8x of the 7900 XT | 0.89 to 0.91x measured (different ROCm) |
| Prompt processing clearly faster than the 7900 XT | Q8_0 1.65x |

## Not done in this run

- No clock pinning; one run only. The 8B model was not run.
- The decode kernel was not tuned for this card (registers, cache hints, more waves).
- Size of the matrix instruction's rounding differences.
- A fix for the probe's NaN reference.
