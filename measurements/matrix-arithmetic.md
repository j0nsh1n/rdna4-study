# What the RDNA 4 matrix instructions compute (RX 9070 XT, gfx1201)

Lane 5 of `notes/STUDY-PLAN.md`. Labels: MEASURED (seen on this card in a run here), READ (from a document),
INFERRED (reasoning from measurements). Card configuration: LACT profile "Testing" (results are bits, not times).
Run: `runs/20261117T141221Z-arith/` (every mode PASS, every deliberate break FAIL). The lead reran the whole script
(`runs/20261117T141759Z-arith/`): the same verdicts and counts in every mode and break. Sections 8 to 13 (second
part: 16-bit accumulators, modifiers, sparse and integer forms, BF8 chains, the forbidden back-to-back read) come
from `runs/20261117T145413Z-arith/`, which also repeats every first-part mode with unchanged counts. The owner's
first run was made from a tree with uncommitted changes; the lead's reruns are the ones that match committed code.
After the review the lead tested full-range inputs, found two gaps in the model, fixed them and added the test as
the mode `wide` (section 2a). The first full run with the fixed model, `runs/20261117T152603Z-arith/`
(22 jobs PASS, 18 breaks FAIL, every earlier line unchanged), was made on the tree before it was committed. The
run that matches committed code is `runs/20261117T160250Z-arith/` (commit 9995208, after the second review's
fixes): 23 jobs PASS, 19 breaks FAIL, every earlier line unchanged; it is the source of the `wide` controls (smallest
unit 2^-151: 758 and 167 of 1,000,192 BF16 elements differ; 2^-153: 101 and 25; BF16 into BF16 without the
minus-zero rule: 551 and 144) and of the `edge` results.
Leftovers pass (branch `study/leftovers-matrix`, `scripts/study-leftovers-matrix.sh`, `src/study/left_mat.hip`): the
two 4-bit sparse forms (section 10a), the dense 4-bit K = 32 pairing (section 11), the modifiers not covered before
(section 9), the forbidden back-to-back read for more forms (section 13) and the lane mask of more opcodes
(`compiler-and-hazards.md` section 1) come from `runs/20261117T165116Z-leftmat/` (every mode PASS, every deliberate break FAIL). The lead reran the
leftovers script from the branch (`runs/20261117T170450Z-leftmat/`): 11 jobs PASS, 10 breaks FAIL, every check with
the same status; the only verdict text that moved is a timing figure of the wave-order probe (16.36 in the owner's run, 16.37 in
the rerun). After the review the hazard and wave-order verdicts were tightened (the stale-word mask, the busy
placement, the priority and round-robin rates are now asserted) and a round-robin break was added; the full rerun
`runs/20261117T171227Z-leftmat/` gives 11 jobs PASS and 11 breaks FAIL. That run's
summary says "with uncommitted changes": those were two helpers' files of other items (`mat_card.hip`,
`left_dot.hip`), not the `left_*` programs, which were committed (49e0672).

## Summary

- **The matrix instructions are known bit for bit on every input set tried.** CPU models
  (`src/study/arith_model.h`) reproduce the card exactly for all 11 dense forms and all 11 sparse forms
  (MEASURED; the two 4-bit sparse forms in the leftovers pass, section 10a; wave64 through HIP was not tested here; a partial
  EXEC changes nothing for the five opcodes tested, one of them sparse, `compiler-and-hazards.md` section 1): the six float forms with an FP32 accumulator (0 mismatches on 10^6 to 10^7 random elements
  each, adversarial sets, special values, chains to K = 4096), FP16 into FP16 and BF16 into BF16 (section 8), the
  8-bit and 4-bit integer forms with every sign and clamp setting (section 11), and the sparse forms (section 10).
- **How the float forms add (MEASURED).** The 16 products are added in a fixed order of steps: 16-bit inputs in 4
  steps of 4, 8-bit inputs in 2 steps of 8. Each step adds the running sum (C in the first step) and its products in
  a fixed-point adder with one unit for the whole step, set by the step's largest term: 2^-24 of the largest
  product's exponent or 2^-26 of the running sum's exponent, whichever is larger (and never below 2^-152). Bits
  below the unit are cut off toward minus infinity (floor), not rounded. Products are first added exactly in pairs
  (16-bit inputs) or quads (8-bit inputs; inside a quad each product is first floored at 2^-24 of the quad's own
  largest product), then each pair or quad sum is floored once. After every step the sum is rounded to FP32 (round
  to nearest even). A running sum smaller than one unit of the next step vanishes.
- **16-bit accumulators use the same adder, rounded to 16 bits after every step (MEASURED).** FP16 into FP16 rounds
  each step to FP16 to nearest even; BF16 into BF16 rounds each step to FP32 and then simply drops the low 16 bits
  (toward zero). So a BF16 accumulator loses up to a whole BF16 step per instruction step, always toward zero; a
  long BF16-accumulated sum drifts toward zero, and a tiny negative result of the last step comes out as -0
  (every other zero result of any form is +0). FP16 overflow gives infinity, or the largest value when the MODE bit
  FP16_OVFL is set (the only MODE bit that changes any matrix result); BF16 ignores it.
- **Modifiers are plain sign changes (MEASURED, except where marked).** On the FP16 and BF16 forms NEG negates the even slots (low halves)
  of A or B, NEG_HI the odd slots, and on C NEG flips, ABS clears the sign, also for NaN, infinity and zero. On the
  four 8-bit float forms the C bits are accepted by the assembler and ignored by the card. On the sparse float forms
  NEG and NEG_HI act on A's slots as on the dense forms, and on B they give the same result as on A (MEASURED), which fits their acting on the gathered B values by the
  stored slot they meet (INFERRED; the outputs cannot tell that from NEG on A); the C bits there are refused by the assembler but work when encoded by hand.
  NEG_HI on the integer forms (refused by the assembler, undefined in the guide) is ignored where it was run (on A
  for every form tried; on B for signed 8-bit only).
- **Sparse forms are the dense instruction after a gather (MEASURED).** Each stored A element picks one of four B
  slots by a 2-bit field of the index register (the 4-bit forms too, section 10a); the 16 picked products then go through the dense adder exactly
  (dense and sparse instructions give identical bits). Index values the documents call invalid are used as given.
- **Integer forms are exact (MEASURED).** Without the clamp bit the exact sum wraps to 32 bits. With it the running
  sum saturates to the signed 32-bit range after each half of the K slots, not once at the end, also for unsigned
  inputs.
- **When it is exact (MEASURED rule, INFERRED proof).** If all inputs are normal numbers, no partial sum leaves
  the FP32 normal range, all products and C lie on a common grid q (multiples of q) and the sum of their magnitudes
  is below 2^24 q, the result is the exact sum. The first two conditions are needed: a subnormal input counts with
  the format's minimum exponent, so the unit can be coarser than q (FP16: 2^-24 x 2^-1 + 2^-24 x 2^-24 gives
  0x33000000 where the exact sum is 0x33000001, card and model alike), and sixteen BF16 products of 2^-153 give 0
  where the exact sum is 2^-149. For FP8 x FP8 with C = 0 the rule holds whenever the exponents (ea + eb) of the
  16 products span at most 12 binades, subnormal E4M3 codes included (the reviewer's check: 0 bad rows in 800,000
  with subnormals, zeros and C); here 0 of 1.2 million random rows of normal codes with spreads of 8, 12 and 13
  differed. FP8 weights widened to FP16 obey the same rule in the FP16 form. Ordinary FP16 or BF16 data with 22- or
  16-bit products is generally not exact.
- **How wrong it can be.** Relative to the sum of magnitudes |C| + sum |p|, the largest difference found on one
  instruction with normal inputs is 2^-21.2 (FP16), 2^-21.4 (BF16, FP8); these come from a random search and are
  lower bounds on the worst case, not the worst case. It can be far larger in two situations. With subnormal inputs:
  FP16 A = {2^-24, -2^-24}, B = {2^15, 2^-6} gives 0x3AFFFC00 on the card where the exact sum is 0x3AFFFFF8, an
  error of 2^-14 of the sum of magnitudes (`arith_more edge`). With BF16 inputs whose step sums fall in the FP32
  subnormal range, where FP32 itself has few bits left: the rounding search's worst BF16 case is off by 2^-11.1 of
  the sum of magnitudes (the model gives 0x80000002 where the exact sum rounds to 0x00000001; recomputed by the
  reviewer and again by the lead from the example printed in `arith_qworst.log`, line "worst path bf16 K 16 target
  fp16"; the 2^-12.7 printed on that line is the exact sum over the sum of magnitudes, a different quantity). INFERRED cause of the first: a subnormal counts with the minimum exponent, so the unit of the floor
  follows the exponent field, not the much smaller value; `errors-after-quantizing.md` section 3). In ulps of the
  result it is unbounded under cancellation: +L - L + tiny gives one unit of 2^-24 L instead of the tiny value. On
  model-shaped data the card is about as close to the exact sum as a sequential FP32 FMA loop over the same K
  (section 6).
- **MODE does not matter, except FP16_OVFL on FP16 into FP16.** Rounding mode and denormal mode leave every matrix
  output unchanged, while the same MODE writes change VALU results (MEASURED). Denormal inputs and outputs are always
  kept.
- **What a kernel author should do.** Treat each instruction as "floor to 2^-24 of the largest term, then FP32
  rounding per step"; the error bound per instruction is about 2^-20 of the sum of magnitudes for normal inputs
  whose partial sums stay in the FP32 normal range (INFERRED). For bit-exact FP8 results, keep each instruction's product exponents within 12 binades (per-block
  scaling does this) and keep C on the same grid, or accumulate partial sums outside the matrix unit. Prefer an FP32
  accumulator over BF16 into BF16 for anything long. Never feed a matrix result into A or B of the very next matrix
  instruction: the card then reads stale register contents (section 13; the same operand words in every form tried
  and with a second wave on the SIMD), which the guide forbids.
- **Compared with RDNA 3:** the RDNA 3 one's-complement artefacts are gone (1 + (-1) = +0 here; it was -2^-24).

## 1. Structure of the adder (MEASURED)

Slots are register positions (`src/study/arith_gpu.h`): for 16-bit data slot s of an A row (B column) sits in lane
half s >> 3, VGPR (s & 7) >> 1, half s & 1; for 8-bit data in lane half s >> 3, VGPR (s & 7) >> 2, byte s & 3.

| Form | Steps, in order (slots) | Products added exactly in | Unit of the floor | Experiment |
|---|---|---|---|---|
| FP16, BF16 into FP32 | {0-3}, {8-11}, {4-7}, {12-15}; C enters with the first | pairs (0,1), (2,3) of each step (one VGPR) | 2^(E-26), E = max(ea + eb + 2 over the step's products, exponent of the running sum) | loss matrix, pair probes |
| FP8, BF8 and mixed into FP32 | {0-7}, {8-15}; C enters with the first | quads {0-3}, {4-7} of each step, each aligned to its own largest product first | same | loss matrix, quad probes |

Details, each with its evidence (scripts in `build/arith/explore/` during the study; the decisive cases are
reproduced by `arith_check selftest` and `adv`):

- **Order of steps.** +L at slot x, -L at slot z and a small term s (gap 2^-30) elsewhere: s survives exactly when it
  is added after +L and -L have cancelled. The full 17 x 16 x 16 loss matrix (`loss_f16.txt`) fits only the step
  order {0-3}, {8-11}, {4-7}, {12-15} with C in the first step; the FP8 matrix fits {0-7}, {8-15}. Swapping slots 4
  and 8 changes 172 of 3,360 entries of the fingerprint (the probe can see the order).
- **Exponents only.** The floor unit depends on ea + eb, not on the product's value: 1 x 1, 1.5 x 1.5 and
  1.999 x 1.999 (x 2^16) all floor a tiny negative product to -2^-8. Subnormal inputs count with the minimum
  exponent (FP16 3 x 2^-24 times 2^10 floors at 2^-28).
- **C and the running sum.** C behaves as the running sum: it sets the step's unit to 2^-26 of its exponent when
  it is the largest term, two bits finer than a product of the same exponent would. A running sum whose exponent
  is below the unit's contributes 0 (C = -0.9375 units against +L - L gives 0, C = -1.0625 units gives -2 units).
  A subnormal FP32 running sum counts with the minimum exponent (-126) for both rules, so it is floored, not
  dropped, when the unit is below 2^-126: BF16 2^-51 x 2^-51 with C = 0x80400000 gives 0x0C7FFFFF on the card
  (`arith_more edge`, case 2; the exact sum is 0x0C800000).
- **Smallest unit (found after the review).** The unit never goes below 2^-152, three bits below the smallest FP32
  value. It only shows with BF16 inputs, when every term of a step is below 2^-126: a single product in the FP32
  subnormal range is then floored at 2^-152 before the rounding, so the result is not always the nearest subnormal
  (-49.43 units of 2^-149 gives -50; `arith_more edge`, case 4). A sweep of 45,056 single products across 2^-156
  to 2^-140 (`arith_more edge`) fits 2^-152 and neither neighbour: 0 mismatches, against 2,249 with 2^-151, 1,112
  with 2^-153 and 2,128 with no smallest unit.
- **Pairs and quads.** With +L - L in slots 0, 1, two small products in slots 2, 3 are added exactly before the floor
  (-0.5 - 0.5 units gives -1; separately floored would give -2); FP8: two small products in the same quad add
  exactly, in different quads or in L's quad they are floored separately.
- **Rounding per step.** C = L, s in step 1, -L in step 2: the result is RNE32(L + floor(s)) - L, so each step
  ends with an FP32 round to nearest even. With no cancellation, products below half an FP32 ulp are lost.
- **Overflow and underflow per step.** BF16 products +2^200 and -2^200 in different steps give +inf; sixteen
  products 2^-153 give 0 (each step's 2^-151 rounds to 0), while the exact sum is 2^-149.

### K order labels

| Slot | 0-3 | 4-7 | 8-11 | 12-15 |
|---|---|---|---|---|
| K under AMD's guide table (16-bit) | 0-3 | 8-11 | 4-7 | 12-15 |
| K under the sample-code order (GPUOpen, LLVM, llama.cpp; 16-bit) | 0-3 | 4-7 | 8-11 | 12-15 |
| Step (16-bit forms) | 1 | 3 | 2 | 4 |
| K (8-bit forms; guide and sample code agree) | 0-3 | 4-7 | 8-11 | 12-15 |

Under the guide's labels the 16-bit adder takes K 0-3, 4-7, 8-11, 12-15 in sequence. That is only a way to read
the table; nothing measured here says which labelling the hardware designers meant. Both labellings give the same products (identity checks in
`src/wmma_exact.hip`), so the choice only matters for which partial sums are floored together.

Cross-check through another software stack (lead): the Vulkan lane ran 2,560 FP16 elements (10 sets of a
16 x 16 tile, random data plus special cases) through RADV's cooperative-matrix multiply-add, in wave32 and wave64.
`src/study/arith_vkcheck.cpp` feeds the same A, B and C to this model. MEASURED: with the matrix's K index placed
in slot K (the sample-code order, K contiguous in registers) the model reproduces all 2,560 result words; with the
guide's labelling 2,050; the exact sum rounded once 1,384 (only these two labellings were tried). INFERRED: through
RADV a tile's products are added in the steps K {0-3}, {8-11}, {4-7}, {12-15}, so Mesa lays operands out
K-contiguously, and the model holds outside HIP and in wave64. One corrupted result word turns the check's verdict
to FAIL (2,559 of 2,560; run by hand by the lead, not part of the script's logs).

## 2. Agreement of the model (MEASURED, `arith_check`)

| Set | FP16 | BF16 | FP8 x FP8 | FP8 x BF8 | BF8 x FP8 | BF8 x BF8 |
|---|---|---|---|---|---|---|
| random (narrow, wide, cancelling C, big and small, all codes) | 0 of 10,000,128 | 0 of 10,000,128 | 0 of 10,000,128 | 0 of 1,000,192 | 0 of 1,000,192 | 0 of 1,000,192 |
| adversarial (cancelling pairs, gaps 0-40, all slots and C) | 0 of 1,999,872 | 0 of 1,999,872 | 0 of 1,999,872 | 0 of 499,968 | 0 of 499,968 | 0 of 499,968 |
| special values (40,960 each) | 0 | 0 | 0 | 0 | 0 | 0 |
| chains through C, K = 64 / 1024 / 4096 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 (section 12) | 0 / 0 / 0 | 0 / 0 / 0 |
| card equals the exact sum (random set) | 31.6 % | 49.4 % | 66.5 % | 52.1 % | 45.7 % | 44.1 % |

### 2a. Full-range inputs (lead, after the review; `arith_more wide`)

The sets above keep BF16 exponents within +-60 and have few subnormals, so no step ever worked below the FP32
normal range. The mode `wide` uses any finite bit pattern as input, with and without a 40 % share of subnormal
inputs, a zero, subnormal or full-range accumulator, and 1 to 16 slots in use: 1,000,192 elements per form and
share, 16 million in all.

| Form | Mismatches before the two fixes (131,072 elements per share; the lead's first test by hand, not in the run logs) | After (`wide`, 2 x 1,000,192) |
|---|---|---|
| FP16 into FP32, the four 8-bit forms, FP16 into FP16 | 0 | 0 |
| BF16 into FP32 | 21 and 7 (subnormal results off by one unit) | 0 |
| BF16 into BF16 | 48 and 18 (a tiny negative result in the last step: the card gives -0) | 0 |

The two fixes: the smallest unit 2^-152 (section 1), and for BF16 into BF16 a negative step result that truncates
to zero keeps its sign (0x8000), because the card keeps the high half of the FP32 bits (`arith_more edge`, case 5).
So the first version of the model was wrong for about 1 element in 6,000 to 19,000 of full-range data for BF16 into
FP32 and 1 in 2,700 to 7,300 for BF16 into BF16, and right on everything inside the earlier sets' ranges; FP8 and
FP16 inputs cannot reach the affected range. The deliberate break (the model without the smallest unit) FAILs: 46
of 400,384 BF16 elements differ. Controls inside `wide`, on the same card outputs: the smallest unit moved to
2^-151 or 2^-153, and the BF16 accumulator form without the minus-zero rule, must each disagree with the card (the
counts are in the run's `arith_more-wide.log`).

Controls (MEASURED): the model with one stage changed disagrees with the card on the random set: unit one bit finer
44 % (FP16), 32 % (BF16), 15 % (FP8); products floored one by one 30 %, 24 %, 1.4 %; FP8 without the quad-local
alignment 0.67 %. For pairs the local alignment cannot change a result (the larger product of a pair is always exact
in its own window, and flooring one small term cannot cross a step unit differently; INFERRED), so the FP16 and BF16
rows show 0 there, as expected. The `selftest` variants (unit, one-by-one floor, exact steps, K-contiguous order,
small running sum floored) each disagree with the measured model on 141 to 7,348 of 16,384 random elements; that
compares model with model and shows only that the variants are different models. The reviewer reran 17 one-rule
variants against the logged card outputs of the random set: each disagrees on at least 0.5 % of elements (order of
the last two steps 44 % FP16 and 26 % FP8; floor against truncation 54 %, 35 %, 11 %; FP32 rounding per step
against truncation 69 %, 57 %, 30 %), except "zero products do not take part in the anchor", which the random set
barely tests (970, 2 and 0 of 1,048,576); the `wide` set has zero operands in every problem.

## 3. Special values (MEASURED, all six float forms)

| Case | Result |
|---|---|
| NaN in A or B (any slot, payload, sign; both E4M3 NaN codes 0x7F, 0xFF; E5M2 NaN codes) | 0xFFC00A3D, always |
| inf x 0, or +inf and -inf among a step's products (FP16, BF16) | 0xFFC00000 |
| the same in the 8-bit forms | 0xFFC00A3D (0xFFC00000 never appears there) |
| NaN in C | C quieted, payload and sign kept (0x7F800001 gives 0x7FC00001) |
| NaN in A or B together with NaN in C, same step | 0xFFC00A3D |
| NaN made in step 1, inf x 0 in step 4 | 0xFFC00000 (a later step's NaN replaces an earlier one) |
| inf + C = -inf; running sum -inf + product +inf | 0xFFC00000 |
| inf - inf in different steps with C = NaN | C's NaN (the step with C made it first) |
| exact zero result (any mix of signed zeros, x - x, underflow of a negative result) | +0, never -0 |
| FP32 overflow at any step | infinity, kept to the end (round to nearest even: FLT_MAX + half an ulp gives inf) |
| subnormal inputs (FP16, BF16, FP8) and subnormal results | kept, under every MODE setting |
| E4M3 range | all 254 finite codes act as their values; 448 x 448 = 200704 exact |

## 4. MODE register (MEASURED, `arith_check mode`)

Default MODE on this stack: 0x010000F0 (round to nearest even, denormals kept, bit 24 SCALAR_PREFETCH_EN set).
The probe writes bits [7:0] and 23 with `s_setreg_b32`, reads them back, runs the matrix, and restores MODE.

| Setting | MODE written | VALU control (5 operations) | Matrix outputs changed (FP16 / BF16 / FP8, of 65,536 each) |
|---|---|---|---|
| default, written back | 0x000000F0 | 0 changed | 0 / 0 / 0 |
| round toward +inf | 0x000000F5 | 1 changed (FP16 add) | 0 / 0 / 0 |
| round toward -inf | 0x000000FA | 2 changed | 0 / 0 / 0 |
| round toward zero | 0x000000FF | 1 changed | 0 / 0 / 0 |
| denormals flushed (single and FP16) | 0x00000000 | 2 changed | 0 / 0 / 0 |
| single-precision denormals flushed | 0x000000C0 | 1 changed | 0 / 0 / 0 |
| FP16_OVFL set | 0x008000F0 | 0 (no FP16 overflow in the control) | 0 / 0 / 0 |

The MODE write takes effect (read back in the VALU kernel and VALU changes; the matrix kernel's own read-back is
not checked, and this table's FP16_OVFL row has no working control, section 8 has one); the matrix instruction
ignores rounding and denormal mode.
The test data include FP16 subnormal inputs, BF16 products near 2^-140 and FP32 denormal C. FP16_OVFL matters only
for FP16 into FP16 (section 8).

## 5. Determinism (MEASURED, `arith_check determ`)

524,288 elements per form (FP16, BF16, FP8): 0 bit changes over 3 launches, with problems moved to other waves
(reversed order), with A rows and B columns reversed (each value in other lanes and registers), and after a launch
of another form with other data. No hidden state.

## 6. Size of the differences (MEASURED on the model, which equals the card; `arith_stats`)

Data shaped like model data: E4M3 weights from a Gaussian (4.5 sigma at 448); activations FP16 from a Gaussian with
0.1 % outliers at 20x; separately activations scaled and rounded to E4M3; BF16 path with activations rounded to BF16.
"exact" is the exact sum of the same inputs rounded once to FP32, so only the instruction's own error is counted.
Chained through C in steps of 16. Ulps are FP32 ulps of the result; "rel" is |card - exact| / sum |p|.

| Path | K | differ from exact | ulps 0 / 1 / 2 / 3-4 / 5-16 / >16 (share of N) | max rel | FP32 FMA loop differs |
|---|---|---|---|---|---|
| FP8 x FP8 | 16 | 0.006 % | 99.99 / 0.002 / 0.004 / 0 / 0 / 0 | 2^-24.4 | 0.014 % |
| FP8 x FP8 | 256 | 1.8 % | 98.2 / 1.2 / 0.31 / 0.13 / 0.10 / 0.03 | 2^-24.2 | 2.1 % |
| FP8 x FP8 | 4096 | 56.8 % | 43.2 / 33.5 / 12.1 / 5.7 / 4.3 / 1.2 | 2^-25.5 | 58.1 % |
| FP16 (FP8 weights widened) | 16 | 2.2 % | 97.8 / 1.6 / 0.32 / 0.15 / 0.11 / 0.05 | 2^-23.1 | 3.3 % |
| FP16 (FP8 weights widened) | 256 | 61.8 % | 38.2 / 36.8 / 12.3 / 6.3 / 4.7 / 1.6 | 2^-22.2 | 63.8 % |
| FP16 (FP8 weights widened) | 4096 | 92.7 % | 7.3 / 13.4 / 12.4 / 20.2 / 35.5 / 11.3 | 2^-22.8 | 92.9 % |
| BF16 | 16 | 0.34 % | 99.66 / 0.23 / 0.06 / 0.03 / 0.02 / 0.01 | 2^-23.1 | 0.51 % |
| BF16 | 256 | 30.8 % | 69.2 / 20.6 / 5.6 / 2.2 / 1.8 / 0.5 | 2^-22.4 | 32.3 % |
| BF16 | 4096 | 85.2 % | 14.9 / 25.4 / 17.8 / 20.4 / 16.5 / 5.0 | 2^-23.7 | 85.1 % |

Synthetic data only in this table (real weights: `errors-after-quantizing.md` section 4). Rows per K: 200,000 at
K = 16 and 64, 60,000 at 256, 16,000 at 1024, 6,000 at 4096, so the smallest shares rest on few events (FP8 x FP8
at K = 16: 11 elements). All K values (16, 64, 256, 1024, 4096) are in the run's `arith_stats-diff.log`. Large ulp counts come from sums
that nearly cancel; relative to the sum of magnitudes the error stays below 2^-22. For K beyond 16 most of the
difference is the FP32 rounding of the running sum, the same thing an FP32 FMA loop does; the card differs from
the exact sum less often than that loop in every row but one (BF16 at K = 4096: 85.2 % against 85.1 %).

Worst cases by search (64 random restarts x 2 objectives x 20,000 bit flips, single instruction): relative to the
sum of magnitudes 2^-21.17 (FP16), 2^-21.42 (BF16, FP8). A later search for rounding differences (second part,
`arith_qworst`) found cases with subnormal inputs at about 2^-14 of the sum of magnitudes; the bound below holds for
normal inputs only (`errors-after-quantizing.md` section 3). INFERRED bound: per step at most 3 units (two pair floors
and the running-sum floor; FP8: two quad floors and the running sum) plus half an FP32 ulp, a unit being at most
2^-24 of the largest term, so about 4 x (3 + 1) x 2^-24 = 2^-20 for the FP16 form.

## 7. Comparison with RDNA 3 (RX 7900 XT, `rdna3-fp8/measurements/matrix-arithmetic.md`)

| Question | RDNA 3 (F16 into F32) | RDNA 4 (this file) |
|---|---|---|
| 1 + (-1), C = 0 | -2^-24 | +0 |
| 8 + (-1) | 7 - 2^-21 | 7 |
| Best simple model | none fits (best 37-75 % on random data) | exact model, 100 % on 10^7 per form |
| Mixed-sign loss | one's complement without carry-in | two's complement floor at 2^-24 of the largest term |
| Slot position matters | yes (K slot dependent) | yes (steps {0-3}, {8-11}, {4-7}, {12-15}) |
| Deterministic, no hidden state | yes | yes |
| Exact zero sign | +0 | +0 |

## 8. 16-bit accumulators: FP16 into FP16, BF16 into BF16 (MEASURED, `arith_more acc16`, `arith_more ovfl`)

Same adder as the FP32 forms (same steps, slots, pairs, floor unit), but the running sum is a 16-bit value: every
step ends with a rounding to the accumulator format.

| Form | Rounding after every step | Exact model | Controls (other rounding, same data; share disagreeing) |
|---|---|---|---|
| FP16 into FP16 | the step's fixed-point sum rounded to FP16, nearest even | 0 mismatches | FP32 steps rounded once at the end 55.7 %; last step only 55.7 %; toward zero 86.3 % |
| BF16 into BF16 | the step's sum rounded to FP32 (nearest even), then the low 16 bits dropped (toward zero) | 0 mismatches | rounded once at the end 51.8 %; to nearest even 68.3 %; toward zero from the exact step sum 27.3 % |

| Set (per form) | FP16 into FP16 | BF16 into BF16 |
|---|---|---|
| random (narrow, wide, cancelling C, big and small) | 0 of 1,000,192 | 0 of 1,000,192 |
| adversarial (cancelling big terms, small terms at gaps 0-40) | 0 of 499,968 | 0 of 499,968 |
| special values (NaN payloads and signs, infinities, signed zeros, subnormal and overflowing results) | 0 of 102,400 | 0 of 102,400 |
| chains through C, K = 64 / 1024 | 0 of 131,072 / 0 of 16,384 | 0 of 131,072 / 0 of 16,384 |
| card equals the exact sum rounded once (random set) | 44.3 % | 26.4 % |

Details (each found on the card by shrinking a failing element, then held by the sets above):

- **Where C enters.** C is widened exactly and enters the first step like the FP32 forms' C. One difference: a
  subnormal 16-bit running sum counts with the format's minimum exponent (2^-14 for FP16) for the anchor and for the
  "below one unit vanishes" rule, like subnormal inputs. Example: C = -2^-24, products 2787, -314.06 and 9.73 in
  steps 1, 3, 4 (a case shrunk by hand during the study, not kept in a log): the card gives 2482 (C floored to -2^-14, which breaks the tie 2787 toward 2786); with the FP32
  rule (C vanishes) it would be 2484.
- **BF16 is truncated, through FP32.** 1 + 0.75 units gives 1, -17.56 units gives -17, 1 - 0.25 units gives the next
  value below 1. A tiny negative product floored to -2^-26 of the running sum is absorbed by the FP32 rounding first,
  so it does not pull the BF16 result down a step (rounding straight from the fixed-point sum toward zero would).
- **NaN.** A NaN in A or B gives 0xFE3D (FP16) or 0xFFFD (BF16): the quiet NaN with low bits 0x3D, the 16-bit
  counterpart of 0xFFC00A3D. A NaN in C keeps its sign and payload (quieted); inf x 0 and +inf - inf give the
  default NaN of the format with the sign set (0xFE00, 0xFFC0). Exact zero results are +0. One exception, BF16
  into BF16 only: a negative result of the last step that is cut to zero is -0 (0x8000; section 2a).
- **Overflow.** A finite FP16 step result beyond 65504 gives infinity, which stays. BF16 overflows only where FP32
  does.
- **MODE.** FP16_OVFL set: a finite FP16 overflow at a step gives +-65504 instead (6,196 of 132,096 outputs of the
  overflow-heavy set change, infinities drop from 16,074 to 10,516; the model with that rule matches all 132,096).
  True infinity inputs stay infinite. BF16 into BF16 ignores FP16_OVFL. Round toward zero and toward +inf change no
  matrix output in either form. Each write is proved by reading MODE back and by the VALU control (v_add_f16 of
  65504 + 65504 gives 0x7C00 by default, 0x7BFF with FP16_OVFL or round toward zero); MODE is restored after.

## 9. Modifiers NEG, NEG_HI, ABS (MEASURED, `arith_more mods`)

Method: the card with the modifier on the original data against the card without it on data whose sign bits were
changed in memory. Half of the problems carry NaN (several payloads, both signs), infinities, signed zeros and
subnormals in A, B and C. 1,000,192 elements per form and modifier.

| Modifier (assembly) | Acts on | FP16, BF16 into FP32 | FP16 into FP16, BF16 into BF16 | FP8 x FP8 |
|---|---|---|---|---|
| `neg_lo:[1,0,0]` (NEG A) | sign of A slots 0, 2, 4, ..., 14 (low halves of each VGPR) | 0 differ | 0 differ | not allowed |
| `neg_hi:[1,0,0]` (NEG_HI A) | sign of A slots 1, 3, ..., 15 (high halves) | 0 differ | 0 differ | not allowed |
| `neg_lo:[0,1,0]`, `neg_hi:[0,1,0]` | the same on B | 0 differ | 0 differ | not allowed |
| `neg_lo:[0,0,1]` (NEG C) | sign of C flipped | 0 differ | 0 differ | ignored: equals no modifier |
| `neg_hi:[0,0,1]` (ABS C) | sign of C cleared | 0 differ | 0 differ | ignored |
| both on C | sign of C set (-abs) | 0 differ | 0 differ | ignored |

Controls (must differ, on 262,144 elements each): the other halves of the same operand, C instead of A or B, the
other C change, and on FP8 the real C sign changes all differ, on 20-78 % of the elements. The deliberate break (C as the operand of an A/B
modifier) FAILs. NEG on A cannot be told from NEG on B by any output: only the products' signs matter (INFERRED; the
A-versus-B control shows 0 differences, as it must). On FP8 the disassembly shows the bits set (`neg_lo:[0,0,1]`
sets bit 63, `neg_hi:[0,0,1]` bit 10; read by hand, not in a log), so the card ignores them on FP8 x FP8 (the BF8 forms have no modifier kernels here); the guide says the FP8 forms have no NEG or ABS
(READ). So each modifier is exactly a sign change before the adder, including for NaN (a NaN C keeps its payload
with the new sign), infinity and zero.

### 9a. Modifiers on the other forms (MEASURED, `left_mat mods`, `runs/20261117T165116Z-leftmat/`)

Same method (1,000,192 elements per case, half the problems with special values; controls on 262,144 elements).
Assembler acceptance was read from `llvm-mc -mcpu=gfx1201` (READ): NEG/ABS on C are accepted for the BF8 forms;
NEG and NEG_HI on A and B are accepted for the four 16-bit sparse forms; the C bits are refused on every sparse form
("invalid neg_lo operand"; C is D there); NEG_HI is refused on the integer forms. Refused bits were set by hand
(`.long` words with fixed registers; `llvm-objdump` cannot decode those words either).

| Form | Modifier | Equal to (0 differ) | Controls (differ) |
|---|---|---|---|
| FP8 x BF8, BF8 x FP8, BF8 x BF8 | `neg_lo:[0,0,1]`, `neg_hi:[0,0,1]`, both | no modifier on unchanged data: ignored, as on FP8 x FP8 | the real C changes: 89,422 to 190,207 of 262,144 |
| sparse FP16, BF16 into FP32; FP16 into FP16; BF16 into BF16 | `neg_lo:[1,0,0]` (NEG A) | A sign flipped in the even stored slots | odd slots, C flipped: 191,397 to 206,142 |
| same | `neg_hi:[1,0,0]` (NEG_HI A) | A sign flipped in the odd stored slots | even slots, C flipped |
| same | `neg_lo:[0,1,0]` (NEG B) | A sign flipped in the even stored slots (the products of even stored slots negated) | B's even register halves flipped: 182,551 to 193,338; odd stored slots |
| same | `neg_hi:[0,1,0]` (NEG_HI B) | A sign flipped in the odd stored slots | B's odd register halves; even stored slots |
| sparse FP16 into FP32, hand-encoded | NEG on C (bit 63) | C sign flipped | C cleared 85,849; no change 171,854 (of 250,112) |
| same | ABS on C (bit 10) | C sign cleared | C flipped 85,849; no change 86,005 |
| IU8 signed (on A and on B), IU8 unsigned, IU4 K = 16 (signed, unsigned) and IU4 K = 32 (signed) (on A only), hand-encoded | NEG_HI | the instruction without it: ignored (0 of 250,112) | the other sign settings: 242,644 to 250,112 |

The hand-encoded words without the extra bits equal the assembled instructions (0 of 250,112, sparse FP16 and IU8),
so the encodings are right. A scan of 15 candidate meanings of NEG B on the sparse FP16 form (B halves, B register
groups, B lane halves, B slot pairs, A, C) leaves only "the products of even stored slots" (0 of 262,144; every
other candidate 192,464 to 206,142). INFERRED: on the sparse forms the B modifier is applied after the gather, to
the selected B value of stored slot j by the parity of j; no output can tell it from the same bit on A.

## 10. Sparse forms (SWMMAC) (MEASURED, `arith_more sparsemap`, `arith_more sparse`)

Layout: A holds 16 stored elements per row in the dense A image of its width (slot j: lane r + 16 (j >> 3), as in
section 1); B holds 32 slots per column, slot t at lane c + 16 (t >> 4), position t & 15 in the dense order (16-bit:
VGPR (t & 15) >> 1, half t & 1; 8-bit: VGPR (t & 15) >> 2, byte t & 3). C is also D. The index is one VGPR per lane;
`index_key` selects its low (0) or high (1) 16 bits.

**What the index selects.** Stored element j belongs to group g = ((j & 7) >> 1) + 4 (j >> 3) of four B slots
(4g .. 4g + 3) and takes slot 4g + i, where i is the 2-bit field number (j & 7) + 8 index_key of the index word in
the lane that holds j (lane r for j < 8, lane r + 16 for j >= 8). Each field value 0-3 selects its slot as given,
including the cases the documents call invalid (both elements of a pair naming the same slot, or out of order): the
two products then use the same B value.

| Experiment | Result |
|---|---|
| one-hot: A row r holds 1 at stored slot j = r, B holds a distinct value in each of its 32 slots, 64 index words (0, all ones, 0x44444444, 0xEEEEEEEE, 0xD8D8D8D8, 59 random), both index_key values | 2,048 of 2,048 results per form as the rule above, for all 9 forms run |
| the rule with the other key's fields (deliberate break) | 672 of 2,048 per form: FAIL |
| random problems with random index words in every lane (invalid values included), model = gather by the rule, then the dense model | 0 mismatches, 500,224 elements per form and index_key |
| the same 16 products through the dense instruction and the sparse one (valid index, unselected B slots filled with unrelated values; adversarial and random data) | 0 of 125,184 differ per form and index_key |

Forms run: FP16, BF16, FP8 x FP8, FP8 x BF8, BF8 x FP8, BF8 x BF8 into FP32; FP16 into FP16; BF16 into BF16; signed
8-bit integer. So the arithmetic on the selected products is exactly the dense adder: same steps, same slots (stored
slot j acts as dense slot j), same rounding. The 4-bit integer sparse forms follow the same rule (section 10a).

### 10a. The 4-bit sparse forms (MEASURED, `left_mat s4map`, `left_mat s4`, `runs/20261117T165116Z-leftmat/`)

`v_swmmac_i32_16x16x32_iu4`: A 1 VGPR (16 stored nibbles per row), B 2 VGPRs (32 per column), index 1 VGPR,
`index_key` 0 or 1. `v_swmmac_i32_16x16x64_iu4`: A 2 VGPRs (32 stored), B 4 VGPRs (64), index 1 VGPR; the assembler
refuses `index_key` (READ). Positions in natural order: position p of an operand with n VGPRs per lane, for row
(column) r, is lane r + 16 floor(p / 8n), VGPR floor((p mod 8n) / 8), nibble p mod 8.

**Rule.** Stored A position j meets B position 4g + f:
- K = 32: g = ((j & 7) >> 1) + 4 (j >> 3), f = the 2-bit field (j & 7) + 8 index_key of the index word of lane
  r + 16 (j >> 3); the rule of the 16-bit and 8-bit sparse forms, with nibbles for slots.
- K = 64: g = ((j & 15) >> 1) + 8 (j >> 4), f = field j & 15 of the index word of lane r + 16 (j >> 4) (all 32 bits
  of the word; that is why there is no index_key).

The 16 or 32 gathered products then go through the integer model of section 11: exact sum wrapped to 32 bits; with
clamp the running sum saturates to the signed 32-bit range after each lane half (8 products for K = 32, 16 for
K = 64), also for unsigned inputs.

| Experiment | Result |
|---|---|
| one-hot: A row r holds 1 at position r (K = 64: also r + 16), every B column holds its position as two digits (two waves), 64 index sets (6 uniform words, 58 with a different random word in every lane) | 1,024 (K = 32, each key) and 2,048 (K = 64) observations, all as the rule; each A position fits exactly one 2-bit field of one lane, values 0-3 giving B positions 4g to 4g + 3 |
| the rule with the other half of the index word (deliberate break) | 338 of 1,024 (K = 32) and 670 of 2,048 (K = 64): FAIL |
| random problems, random index words (every value, the "invalid" ones included), 4 sign settings x with and without clamp | 0 mismatches, 500,224 elements per form, index_key and setting (24 settings, 12.0 million elements); 7,135 to 60,299 outputs per clamp setting at a 32-bit limit |
| controls on the same card outputs | the other index half: 441,668 to 498,860 disagree per setting; saturation once at the end 21,124 and per quarter 27,570 of 6,002,688 clamped elements |
| the same 32 products through the dense K = 32 form and the sparse K = 64 form (valid index, signed, clamp) | 0 of 125,184 differ |

So all 11 sparse forms are the dense instruction after a gather with one common rule (groups of four B slots per
pair of stored elements, 2-bit fields in the lane that holds the element).

## 11. Integer forms (MEASURED, `arith_more int`)

`v_wmma_i32_16x16x16_iu8`, `v_wmma_i32_16x16x16_iu4` and `v_wmma_i32_16x16x32_iu4`; NEG[0] makes A signed, NEG[1]
B; all four sign settings, with and without `clamp`. 1,000,192 elements per form and setting (uniform codes and C;
extreme codes with C near the 32-bit limits; small values crossing zero; structured rows with C near a limit and a
run of slots pushing past it), 24 settings, 0 mismatches in every one.

| Setting | Result |
|---|---|
| no clamp | exact sum C + sum a b (each value w-bit two's complement when signed, else unsigned), wrapped to 32 bits |
| clamp | the running sum, starting at C, saturated to [-2^31, 2^31 - 1] after each step; a step is one lane half of A and B: slots 0-7 then 8-15 (IU8, IU4 K = 16), 0-15 then 16-31 (IU4 K = 32) |
| clamp, unsigned x unsigned | the same signed bounds (C is read as signed; the result never saturates to 2^32 - 1) |

Controls on the same card outputs: saturation once at the end disagrees on 6,451 elements (all in settings with a
signed input; with both inputs unsigned every product is positive, so the two cannot differ); unsigned bounds for
unsigned x unsigned disagree on 407,678; a saturating model on a setting without clamp disagrees on 2,321 to 18,696
elements per setting. The deliberate break (saturation only at the end) FAILs. The IU4 K = 32 register layout used
here (16 nibbles per lane, slot s at lane r + 16 (s >> 4), VGPR (s & 15) >> 3, nibble s & 7) was an assumption; the
leftovers pass measured it (`left_mat s4map`, `runs/20261117T165116Z-leftmat/`): with A row r holding 1 at one position and every B
position holding its own number, A position p meets B position p in both dense 4-bit forms (1,024 observations for
K = 16, 2,048 for K = 32, all as that layout), and the dense K = 32 form equals the sparse K = 64 form after the gather
(section 10a). Which K index a position stands for remains a labelling question (as for the other forms); the
pairing and the clamp steps (lane halves) are measured. The guide says NEG_HI is undefined for these forms (READ):
on the card it is ignored (section 9a).

## 12. Chains through C for the BF8 forms (MEASURED, `arith_more chain8`)

| Form | K = 64 | K = 1024 | K = 4096 |
|---|---|---|---|
| FP8 x BF8 | 0 of 32,768 | 0 of 32,768 | 0 of 8,192 |
| BF8 x FP8 | 0 of 32,768 | 0 of 32,768 | 0 of 8,192 |
| BF8 x BF8 | 0 of 32,768 | 0 of 32,768 | 0 of 8,192 |

Narrow, wide and all-codes inputs. The model with the unit one bit wider (break) disagrees on 42-100 %: FAIL.

## 13. The forbidden back-to-back read (MEASURED, `arith_more hazard`; a curiosity)

AMD's guide requires a V_NOP or an independent VALU instruction between a matrix instruction and a following one
that reads its result as A or B (READ; the accumulator chain needs nothing). Here an FP16-into-FP16 result D1 feeds
the next FP16-into-FP32 instruction as A, or as B, with nothing between (checked in the disassembly), against the
same with a V_NOP; D1's registers start with known values. 64 waves, 16,384 results each way.

| D1 used as | Results differing from the V_NOP version | What was read instead |
|---|---|---|
| A | 4,096 (rows 0, 1, 8, 9 of the second product) | the old contents of D1's VGPRs 0 and 1 in lanes 0, 1, 8, 9; everything else new |
| B | 16,384 (all) | the old contents of D1's VGPRs 0 and 1 in lanes 0-15; everything else new |

The same in all 64 waves (deterministic for this sequence); D1 itself is written correctly, and the V_NOP version
equals the model in every wave. The stale words are D1 rows 0-3 (VGPRs 0, 1 of the lower lanes), so the first
instruction's first result rows reach the register file after the second instruction has read them. Wrong results,
never partial values: each stale word is exactly the old register content (INFERRED from the exact model fit). This
breaks the guide's rule on purpose; no kernel should rely on it, and the pattern may change with clocks, other
forms or other waves in flight (only single waves on an otherwise idle card were tried in this first test).

### 13a. Other forms and a busy SIMD (MEASURED, `left_mat hazard`, `runs/20261117T165116Z-leftmat/`)

Each pair in inline assembly with fixed registers, nothing between (checked in the listing) against one V_NOP; 64
waves. "Alone": one wave per work-group. "Busy": work-groups of 8 waves where waves 4-7 run a chain of 32 FP16 matrix
instructions while waves 0-3 run the pair; HW_ID1 shows every pair wave on the same SIMD as a busy wave (64 of 64).
Which words were read stale was found on the card: the second instruction alone, fed every mix of new and old words
of a row's (column's) two lanes and its operand VGPRs, against the forbidden version's result.

| First result used as | Results differing from the V_NOP version (alone / busy) | Stale words |
|---|---|---|
| FP16 into FP16 as A of FP16 (section 13's case) | 4,096 / 4,096 of 16,384 | operand VGPRs 0, 1 in lanes 0, 1, 8, 9 |
| FP16 into FP16 as B of FP16 | 16,384 / 16,384 | operand VGPRs 0, 1 in lanes 0-15 |
| BF16 into BF16 as A of BF16 into BF16 | 4,056 / 4,054 of 16,384 | operand VGPRs 0, 1 in lanes 0, 1, 8, 9 |
| BF16 into BF16 as B of BF16 into BF16 | 16,256 / 16,254 | operand VGPRs 0, 1 in lanes 0-15 |
| 8-bit integer result, VGPRs 0-1 as A of the 8-bit form | 4,096 / 4,096 | operand VGPRs 0, 1 in lanes 0, 1, 8, 9 |
| 8-bit integer result, VGPRs 6-7 as A | 4,096 / 4,096 | operand VGPRs 0, 1 (result VGPRs 6, 7) in lanes 0, 1, 8, 9 |
| 8-bit integer result, VGPRs 0-1 as B | 16,384 / 16,384 | operand VGPRs 0, 1 in lanes 0-15 |
| sparse FP16 into FP16 as A of sparse FP16 into FP32 | 4,096 / 4,096 | operand VGPRs 0, 1 in lanes 0, 1, 8, 9 |

The pattern is the same as section 13 in every form, alone and beside a busy wave, in all 64 waves; the V_NOP version
equals the second instruction run alone on the written result in every wave, and the first result is written
correctly. A few BF16 results equal the right ones by chance (rounding to BF16), hence 4,056 rather than 4,096. New
here: reading result VGPRs 6-7 gives the same stale positions as VGPRs 0-1, so the stale words are fixed by where
the second instruction reads (its first two operand VGPRs, in the first lanes it reads), not by which result rows the
first instruction writes last; INFERRED: the second instruction fetches those operand words before the first
instruction's result has reached any of its registers.

## 14. Not measured

Still open:

- The back-to-back read under whole-card load (only one busy wave on the SIMD): the cause of the split below, and
  whether it repeats.
  Answered in part (leftovers C, MA-2, `measurements/leftovers-c-b.md`): in one launch of 512 pair waves per pair on
  all 128 SIMDs (four pair waves and four long FP16 matrix chain waves per SIMD), a little under half of the waves
  (222 to 245 of 512, in each of 16 pairs) read the stale words, exactly the rule's words, and the rest read the right
  (new) values, never a mixture. The stale counts are lower bounds where old and new words can coincide. One launch,
  no repeat and no deliberate break of its own. The simplest reading is that four pair waves on one SIMD interleave
  their instructions, so a wave's second instruction often does not directly follow its first (INFERRED, untested: a
  launch with one pair wave per SIMD next to the same chain waves would separate it from the card-wide load).
  MEASURED (the split); the cause is untested.
- New from leftovers C: why, in wave64, the second instruction reads old words in lanes 0, 4, 8, 12 when the stale
  result is its A operand (rows 0, 4, 8, 12 wrong instead of 0, 1, 8, 9). The lane set was fitted to the first run
  and then reproduced exactly (one launch of 64 identical waves per case): a fit that repeats, not a prediction
  (MA-2, `measurements/leftovers-c-b.md`).

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, MA-1, `measurements/leftovers-c-b.md`): NEG_HI on B for the 4-bit integer forms (K = 16 and
  K = 32, signed and unsigned) and for unsigned 8-bit is ignored: the result equals the plain instruction on all 500,224
  elements, while the other sign settings differ on 485,484 to 500,224. MEASURED.
- Answered (leftovers C, MA-2, `measurements/leftovers-c-b.md`): the back-to-back read with another form as the second
  instruction: for 8 new pairs of different forms (FP16 into FP32, FP8, 8-bit integer, FP16 into FP16 feeding FP8, and
  more), as A and as B, alone and beside a busy wave on the SIMD, the second instruction reads its first two operand
  VGPRs of lanes 0 to 15 before the first instruction has written them: 0 results differ from the predicted forbidden
  result. In wave64, as B the stale words are the same register positions; as A they sit in lanes 0, 4, 8, 12 (and
  32, 36, 40, 44 for the 8-bit form's second half), so rows 0, 4, 8, 12 are wrong instead of 0, 1, 8, 9. MEASURED.
- Answered (leftovers C, MA-3, `measurements/leftovers-c-b.md`): the C bits on the sparse forms other than FP16 into
  FP32: on sparse BF16 into FP32, FP16 into FP16 and BF16 into BF16, NEG C flips and ABS C clears the sign of C, as
  on the dense 16-bit forms; on sparse FP8 and BF8 both are ignored, as on their dense twins. NEG_HI on the unsigned
  IU4 K = 32 form is ignored, on A and on B. MEASURED.

- (Done in the leftovers pass: the 4-bit sparse forms, section 10a; modifiers on the sparse, BF8 and integer forms,
  section 9a; the back-to-back read for BF16, 8-bit integer and sparse forms and beside a busy wave, section 13a;
  partial EXEC for five matrix opcodes, `compiler-and-hazards.md` section 1.)

## Method and how to rerun

`scripts/study-matrix-arith.sh` builds everything into `build/arith/bin`, runs each `arith_check` mode as its own
GPU job under the shared lock (single waves; the 10^7 random set takes about 20 s), the CPU statistics, and the
deliberate breaks; results in `runs/<UTC time>-arith/`. Programs:

| Program | Role |
|---|---|
| `src/study/arith_model.h` | the model, the exact reference (1024-bit fixed point), rounding to FP32/FP16/BF16/E4M3 |
| `src/study/arith_gpu.h` | inline-assembly kernels for every dense and sparse form and modifier, the back-to-back sequence, MODE write and restore, register images |
| `src/study/arith_check.hip` | acceptance: selftest, random, adv, special, determ, chain, mode |
| `src/study/arith_more.hip` | second part: acc16, ovfl, mods, sparsemap, sparse, int, chain8, hazard |
| `src/study/left_mat.hip`, `scripts/study-leftovers-matrix.sh` | leftovers pass: s4map, s4 (4-bit sparse), mask, mods, hazard |
| `src/study/arith_qworst.cpp`, `arith_realw.cpp`, `scripts/gguf_rows.py` | rounding worst cases; real FP8 weights (`errors-after-quantizing.md`) |
| `src/study/arith_stats.cpp` | exactness rule, differences, worst-case search, quantizing (CPU, 8 threads) |
| `src/study/arith_run.hip`, `arith_eval.cpp`, `scripts/arith_lib.py` | the exploration runner: element files in, card or model bits out |
| `src/study/arith_dump.hip`, `arith_fit.cpp` | random dumps and offline model fitting with parameter variants |

How the structure was found: targeted experiments first (two-term cases, cancellation sweeps over gaps 20-60, the
full loss matrix, transfer functions of one small term next to a big product or C), then the parameterised model
fitted against random dumps, then shrinking each failing element to its smallest form on the card (which found the
pairs and the quads). Each rule above has a hand-worked case in `arith_check selftest`.
