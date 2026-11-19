# How often the matrix instruction's error survives rounding to FP16 or FP8 (RX 9070 XT, gfx1201)

Lane 5 of `notes/STUDY-PLAN.md`, item 4. Labels: MEASURED (seen in a run here), INFERRED (reasoning).
Run: `runs/20261117T141221Z-arith/arith_stats-quant.log`; sections 3 and 4 (second part): `runs/20261117T145413Z-arith/`
(`arith_qworst.log`, `arith_realw-build_arith_realw_rows.bin.log`).

## Summary

- **Rounded to E4M3, the card's result and the exact result agree almost always.** Of 1,446,000 model-shaped
  outputs, 2 differed (FP16 form, K = 256, 2 of 60,000); every other row was 0. The instruction's error is about
  2^-22 of the sum of magnitudes, far below half an E4M3 step (2^-4 of the value).
- **Rounded to FP16, a few in ten thousand change.** FP16 form with FP8 weights: 0.004 % at K = 16, 0.10 % at
  K = 1024, 0.18 % at K = 4096 (rising with K). FP8 x FP8: 0 at K = 16 and 64, then 1 of 60,000, 3 of 16,000 and
  1 of 6,000, too few events to call a trend. BF16: 1 of 200,000 up to 4 of 6,000. A change is one FP16 step,
  except one case of 6,000 at K = 4096.
- **Real FP8 weights behave like the Gaussian stand-in (MEASURED on the model).** 3,072 rows of a real E4M3 tensor
  (Llama 3.2 3B, layer 10 FFN down projection) give the same picture: rounded to E4M3 at most 0.017 % changed,
  rounded to FP16 at most 0.18 %, the share differing from the exact FP32 sum within a few points of the stand-in
  (section 4).
- **Built to fail, it fails by little in chains, by much only with cancellation and subnormal inputs.** A search for
  inputs that make the card round to a different value than the exact sum found, for K = 256 chains, at most 1 E4M3
  step and 18-34 FP16 steps (all with sums cancelling to far below the sum of magnitudes); for one instruction up to
  6 E4M3 steps and hundreds to thousands of FP16 steps, again only with cancellation and subnormal inputs. Rates near
  50 % are reachable by placing C next to a rounding midpoint on purpose (section 3).
- **What a kernel author should do.** For FP8 outputs the matrix instruction's own error does not matter. For FP16
  outputs from long K the FP16 form changes about 1 in 500 to 1 in 1000 values by one step (FP8 x FP8 far fewer); if bit-identical FP16 output to an
  exact reference is required, keep the long accumulation outside the FP32 chain through C (most of this error is
  the per-instruction FP32 rounding of the running sum, the same as an FP32 FMA loop; see
  `measurements/matrix-arithmetic.md` section 6).

## 1. Table (MEASURED on the model, which equals the card bit for bit; `arith_stats quant`)

The card result (FP32) is scaled by a power of two (exact) and rounded to FP16 (RNE) and to E4M3 (RNE, saturating at
448); the exact sum of the same products is scaled the same way and rounded once. Scales: FP16 so that the RMS is
about 1; E4M3 so that the RMS is about 64 (the share that saturates is given). Interval: 95 %, normal approximation
with a floor of 1/N.

| Path | K | N | FP16 differs | E4M3 differs | E4M3 saturated |
|---|---|---|---|---|---|
| FP8 x FP8 (activations rounded to E4M3) | 16 | 200,000 | 0 (< 0.001 %) | 0 | 0.25 % |
| | 64 | 200,000 | 0 | 0 | 0.26 % |
| | 256 | 60,000 | 0.0017 % (1) | 0 | 0.10 % |
| | 1024 | 16,000 | 0.019 % (3) | 0 | 0.02 % |
| | 4096 | 6,000 | 0.017 % (1) | 0 | 0 |
| FP16 (E4M3 weights widened exactly, FP16 activations) | 16 | 200,000 | 0.0040 % +- 0.0028 | 0 | 0.14 % |
| | 64 | 200,000 | 0.013 % +- 0.005 | 0 | 0.08 % |
| | 256 | 60,000 | 0.028 % +- 0.014 | 0.0033 % (2) | 0.01 % |
| | 1024 | 16,000 | 0.10 % +- 0.05 | 0 | 0 |
| | 4096 | 6,000 | 0.18 % +- 0.11 (1 by two steps) | 0 | 0 |
| BF16 (weights widened, activations rounded to BF16) | 16 | 200,000 | 0.0005 % (1) | 0 | 0.15 % |
| | 64 | 200,000 | 0.0040 % +- 0.0028 | 0 | 0.08 % |
| | 256 | 60,000 | 0.027 % +- 0.013 | 0 | 0 |
| | 1024 | 16,000 | 0.013 % (2) | 0 | 0 |
| | 4096 | 6,000 | 0.067 % (4) | 0 | 0 |

Without cancellation the rates come from values whose exact sum lies within the instruction's error of an FP16
rounding boundary; the FP16 rate grows with K because the FP32 running sum collects one rounding per instruction
(INFERRED from section 6 of the arithmetic file: the FMA loop shows the same growth).

## 2. Control (MEASURED, every row)

The counter is checked two ways on the same data: (1) the exact result against itself changes nothing (0 in every
row); (2) the exact FP32 value against the next FP32 value up must change exactly where a target rounding boundary
lies between them, counted directly from the target's midpoints (including the ties a value sits on). Counts agree
in every row, for example FP8 x FP8 at K = 16: 11,609 by the counter and 11,609 directly for FP16, 360 and 360 for
E4M3; FP16 form at K = 16: 398 and 398. The high count for FP8 data is real: its sums lie on a coarse grid and often
fall exactly on an FP16 midpoint. `arith_stats quant --break` adds one to the direct count and the verdict FAILs.

## 3. Worst cases by search (MEASURED on the model; `arith_qworst`, written by a helper and checked here)

Four paths (FP16 inputs; E4M3 weights widened exactly to FP16; BF16; E4M3 x E4M3), one instruction (K = 16, any FP32
C) and 16-instruction chains through C (K = 256, C = 0), targets FP16 and E4M3 (saturating). Card result and exact
sum are scaled by the same power of two (sum of magnitudes M in [1, 2) for FP16, [64, 128) for E4M3) and rounded once
each. Two searches: (1) random-restart hill climbing on the raw input codes for the largest distance in target steps,
then the best case shrunk by zeroing products; (2) a generator family (exponent spread, cancelling partners, zeros,
coarse mantissas, subnormals, C placed near a rounding midpoint) hill-climbed on its parameters for the highest rate,
then re-measured on fresh samples (N = 400,000 for K = 16, 59,968 for K = 256).

| Path | K | Target | Worst steps found | Exact sum / M there | Best rate found (95 % interval) |
|---|---|---|---|---|---|
| FP16 | 16 | FP16 | 943 | 2^-14.1 | 49.5 % (49.3-49.6) |
| FP16 | 16 | E4M3 | 6 | 2^-12.2 | 49.7 % |
| FP16 | 256 | FP16 | 34 | 2^-14.3 | 1.07 % (0.99-1.15) |
| FP16 | 256 | E4M3 | 1 | 2^-8.6 | 0.04 % |
| FP16 form, E4M3 weights | 16 | FP16 | 25 | 2^-13.5 | 48.3 % |
| FP16 form, E4M3 weights | 16 | E4M3 | 1 | 2^-12.1 | 45.7 % |
| FP16 form, E4M3 weights | 256 | FP16 | 33 | 2^-14.7 | 0.15 % (0.13-0.19) |
| FP16 form, E4M3 weights | 256 | E4M3 | 1 | 2^-14.6 | 0 (< 0.007 %) |
| BF16 | 16 | FP16 | 7110 | 2^-12.7 (a sum near 2^-137) | 49.1 % |
| BF16 | 16 | E4M3 | 1 | 2^-11.6 | 47.8 % |
| BF16 | 256 | FP16 | 21 | 2^-16.1 | 9.3 % (9.1-9.6) |
| BF16 | 256 | E4M3 | 1 | 2^-12.3 | 4.1 % (3.9-4.2) |
| E4M3 x E4M3 | 16 | FP16 | 5 | 2^-16.1 | 49.4 % |
| E4M3 x E4M3 | 16 | E4M3 | 1 | 2^-12.8 | 45.6 % |
| E4M3 x E4M3 | 256 | FP16 | 18 | 2^-14.7 | 0.013 % |
| E4M3 x E4M3 | 256 | E4M3 | 1 | 2^-12.4 | 0 (< 0.007 %) |

Reading (INFERRED from the table and the model's rules):

- Every case with more than one step has an exact sum far below the sum of magnitudes (2^-8 to 2^-16 of M), so the
  target's steps there are tiny compared with the instruction's floor unit, which follows the largest term.
- The FP16 cases with hundreds of steps contain subnormal inputs. A subnormal counts with the minimum exponent
  (`matrix-arithmetic.md` section 1), so its product's floor unit is set by the exponent field, not by its much
  smaller value: the error relative to M reaches about 2^-14, against 2^-21 with normal inputs. The 7110-step BF16
  case is a sum near 2^-137, below the smallest normal FP32 value; real but of no practical interest.
- The 45-50 % rates at K = 16 come from C chosen right next to a target rounding midpoint; they show what a hostile
  C can do, not what data does. Chains (no free C) reach 0-9 %: highest for BF16 inputs, whose 16-bit products are
  rarely exact.
- For E4M3 outputs the worst found in chains is one step. Searches are heuristic: larger distances may exist, above
  all for K = 256 where each climb gets fewer iterations.

Checks: the step counter against a brute-force rounding with a sorted table of every target value (no code shared
with the model's rounding), 400,000 values per target including 240,752 exact midpoints: 0 mismatches; each of the
16 worst cases recomputed from its printed hex (a fresh chain of the model, the exact sum by 128-bit integers):
16 of 16 equal. `arith_qworst --break` adds one to the expected count: FAIL. Each worst case's inputs in hex are in
the log; two short ones: FP16 target, FP16 form with E4M3 weights, 25 steps, K = 16,
`C=0x348CCFFF 1:5C/8057 2:8D/2A9 4:79/8056 5:F/3F1 7:7/279 8:87/8CB4 9:7/279 10:7B/8012 11:F/3F1 12:F/3F1 13:FC/8056 14:F/71`
(slot:weight code/activation bits); E4M3 target, BF16, 1 step, `C=0x3662FFFF 3:4/FD83 5:24/75C2 14:8004/FD83`.

## 4. Real FP8 weights (MEASURED on the model; `arith_realw`, `scripts/gguf_rows.py`, written by a helper)

The FP8 Llama 3.2 3B file of the rdna3-fp8 project (`llama32-3b-fp8-outq8.gguf`, read only) stores its 196
projection matrices as ggml type 43, a non-upstream id: raw E4M3, one byte per element, no blocks (sizes from tensor
offsets are exactly one byte per element; no NaN code; sign bit on 50.0 %), with one FP32 scale per matrix in a
separate `<layer>.scale` tensor. The scale is a factor outside the instruction and does not change the grid of the
products; it was not applied. Used: all 3,072 rows x 8,192 columns of `blk.10.ffn_down.weight`, as row segments of
length K. Activations: the synthetic model of section "Data and method", identical draws for both weight sources.

| Path | K | Differs from exact: real | stand-in | FP16 differs: real | stand-in | E4M3 differs: real | stand-in |
|---|---|---|---|---|---|---|---|
| FP8 x FP8 | 16 | 0.0020 % | 0.0055 % | 0 | 0 | 0 | 0 |
| FP8 x FP8 | 256 | 1.15 % | 1.80 % | 0.0067 % | 0.0017 % | 0 | 0 |
| FP8 x FP8 | 4096 | 54.3 % | 56.8 % | 0.050 % | 0.017 % | 0 | 0 |
| FP16 (weights widened) | 16 | 2.15 % | 2.21 % | 0.0020 % | 0.0040 % | 0 | 0 |
| FP16 (weights widened) | 256 | 61.6 % | 61.8 % | 0.053 % | 0.028 % | 0 | 0.0033 % |
| FP16 (weights widened) | 4096 | 92.0 % | 92.7 % | 0.167 % | 0.183 % | 0.017 % | 0 |
| BF16 (weights widened) | 16 | 0.28 % | 0.34 % | 0 | 0.0005 % | 0 | 0 |
| BF16 (weights widened) | 256 | 29.7 % | 30.8 % | 0.032 % | 0.027 % | 0 | 0 |
| BF16 (weights widened) | 4096 | 85.5 % | 85.2 % | 0.183 % | 0.067 % | 0 | 0 |

N as in section 1 (200,000 at K = 16, 60,000 at 256, 6,000 at 4096). The stand-in columns equal `arith_stats`
output exactly (same code path, same seeds), which checks the new program. Real weight codes sit about three
binades lower than the stand-in's (exponent field peak 10 against 13) and have more zeros (0.006 % against 0.001 %);
that narrows each instruction's product spread a little, so the FP8 path differs from the exact sum somewhat less
often. Rounded to FP16 or E4M3 the two sources agree within the intervals (a few counts each). So the Gaussian
stand-in is a fair model for this question (INFERRED from one tensor of one model with synthetic activations).
Control as in section 2, in every cell; `--break` FAILs.

## Data and method

Weights: E4M3 codes from a Gaussian scaled so 4.5 standard deviations reach 448 (saturating rounding). Activations:
a standard Gaussian with 0.1 % of values multiplied by 20, rounded to FP16; for the FP8 path scaled so the outliers
reach about 448 and rounded to E4M3; for the BF16 path rounded to BF16. Each output is a dot product of K products,
computed as K/16 model instructions chained through C (the model equals the card for chains up to K = 4096,
`arith_check chain`). Real model tensors were not used.

Rerun: `scripts/study-matrix-arith.sh` (all), or `build/arith/bin/arith_stats quant` after `build`.

## Not measured

Still open:

- Real activations (section 4 pairs real weights with synthetic activations). Needs the owner (leftovers C plan,
  EQ-1b): it needs a model forward pass.
- The worst-case search beyond K = 1024; the K = 1024 figures below are with subnormal inputs allowed and come from a
  search with a quarter of the iterations of K = 256, so a lower figure may only mean a less thorough search.

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, EQ-1a, `measurements/leftovers-c-b.md`): other tensors and layers, and the per-tensor scale
  applied. 12 real E4M3 tensors (layers 0, 10, 20, 27; `attn_q`, `ffn_gate`, `ffn_down`) give the same picture as
  layer 10. Applying the stored scale in FP32 adds one rounding: pooled FP16 changes, unscaled to scaled, at K = 16:
  FP16 path 56 to 155 (2.8 times), BF16 path 8 to 81 (10 times), FP8 path 0 to 22 of 2.4 million (no ratio); at
  K = 256: FP8 path 12 to 49 of 720 k (about 4 times, small counts), FP16 path 306 to 254, BF16 path 152 to 136; at
  K = 4096 no change is visible in 24 k samples. At most 0.0117 % of outputs change in any tensor. MEASURED on the
  model, which equals the card; that the scale's rounding causes the extra changes is INFERRED.
- Answered (leftovers C, EQ-2, `measurements/leftovers-c-b.md`): the worst-case search without subnormal inputs: one
  instruction reaches at most 5 FP16 steps (against up to 7,110 with subnormals), chains at K = 256 reach 31 to 35
  steps for the FP16, FP16-form and BF16 paths and 14 for E4M3 x E4M3 (cancellation across the chain, not
  subnormals). With non-finite inputs (880,000 cases) the card's result class (NaN, +inf, -inf, finite) differs from
  the IEEE class of the exact sum in 0. K = 1024 (subnormals allowed): worst cases of 46 FP16 steps for the FP16 path
  against 34 at K = 256, and 26 against 18 for E4M3 x E4M3; the other two paths are lower (22 against 33, 19 against
  21); E4M3 steps stay at 1. MEASURED on the model; the searches are heuristic, so larger worst cases may exist.
- Answered (leftovers C, EQ-3, `measurements/leftovers-c-b.md`): outputs rounded to BF16 change about a tenth as many
  values as FP16: 11 BF16 changes against 114 FP16 changes pooled over all cells (ratio 0.096; the expected range was
  1/16 to 1/4), every BF16 change one step. The counts per cell are small (0 to 3). MEASURED on the model.
