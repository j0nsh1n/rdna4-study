# Conversions, dot products and small formats on RDNA 4 (lane 6)

Lane 6 of `notes/STUDY-PLAN.md`. Card: Radeon RX 9070 XT (gfx1201, wave32), ROCm 10.1.0 (hipcc, clang 24), kernel
7.3.4, LACT profile "Testing" (recorded in every run summary). Run folder: `runs/20261117T151742Z-conv/`
(`scripts/study-conversions.sh all`). Labels: MEASURED (seen on this card in that run), READ (from a document),
INFERRED (reasoning). Bit results do not depend on clocks; cycle counts come from the wave's own counter (lane 1's
method), so they do not depend on the clock either.

This lane was written in a staging folder outside the repository and copied in unchanged by the lead, who reran the
whole script from the repository (`runs/20261117T153940Z-conv/`): 40 of 41 jobs and all 7 breaks gave the same
verdicts, and the same counts except in the dot table's columns that use lane 5's model, which had been fixed in
between (item 5). The timing job (`cvt_time`) failed in that rerun: its medians were the same, but several
samples were disturbed (for example the dependent FMA chain 5.000 with a range of 5.000 to 12.454) and two blocks
were flagged as not linear. The lead then repeated that job ten times: 7 passed and 3 failed the same way (logs of
repeats 4 to 10 are in that folder as `time-repeat-*.log`; the first three, all passes, were not kept). The clean repeats (5, 6, 8, 9) reproduce the quoted values: single-instruction rows exactly, except one latency
row at 5.025 against 5.000 in repeat 8, and multi-instruction rows within about 0.1 cycle (for example 18.345
against 18.347, 6.031 against 6.000, 18.586 against 18.463). In
the disturbed runs some dependent-chain (latency) rows and the LDS rows move: for example the FMA chain 5.101, a
dot4 chain 7.844 against 6.000, the LDS widening row 20.4 to 23.8 against 18.3. During those repeats the driver reported the card 17 to 45 % busy
with desktop programs (the compositor and several applications use the graphics engine all the time), which do not
take the study's lock. So the quoted cycle figures are those of undisturbed runs, a disturbed run usually fails the probe's check (repeat
12 passed although some of its samples were disturbed: its medians held), and a failed `cvt_time` verdict on this
machine means "repeat it". The timing leftovers pass sized it (`measurements/execution-core.md` item 9): one
`cvt_time` sample of a 5-cycle chain spans about 82,000 cycles and meets the card's regular stall (about 3,900
cycles) with a chance of 0.9 %; the worst disturbed sample above held about 61,000 extra cycles, 16 stalls' worth,
so desktop drawing, not the regular stall and not the clock ramp. Checks that use the extreme samples of regions
this long fail now and then by design; the medians are what to quote.

After the review (FIX-PASS) the lead changed two checks and reran those jobs (logs in the same folder): the
`cvt_dot mode` jobs now fail if any result differs from the rule under a changed MODE (`dot-mode-*-after-review.log`:
0 of 8 x 10^6 in all five settings, PASS), and the `cvt_time` verdict text names the calibration and the linearity
count separately (repeats 11 to 13: PASS).

Agreement with lane 5 (lead): the two BF16 rules this lane had to add to lane 5's adder (the anchor never below
-126; minus zero handled before the BF16 cut) are the two gaps the lead found independently in the matrix model
with full-range inputs and fixed there (unit never below 2^-152 = 2^(-126 - 26); `measurements/matrix-arithmetic.md`
section 2a). Two routes, one rule.

## In short

- **FP8 and BF8 to FP32 is exact** for every code at every byte position (MEASURED). The byte select picks byte 0 to
  3 in plain order; the packed form converts the low byte of the chosen half first. NaN codes come out as
  0xFFC00000 whatever their sign.
- **FP32 to FP8 and BF8 (`v_cvt_pk_*`) is exact round-to-nearest-even** over all 2^32 inputs in both slots
  (MEASURED). It ignores the MODE rounding and denormal bits. Things a kernel author must know:
  1. A NaN input always becomes 0xFF (FP8) or 0xFE (BF8): the sign is lost.
  2. Overflow gives NaN for FP8 and infinity for BF8. With MODE.FP16_OVFL set, finite values saturate (448, 57344),
     but **infinity still gives NaN (FP8) or infinity (BF8)**: clamp before converting if you want saturation.
  3. Values in (448, 464] round down to 448 (464 is a tie and goes to the even code); above 464 is overflow.
- **Stochastic rounding (`v_cvt_sr_*`) is real stochastic rounding, and the guide's entry for the instruction,
  as this lane read it, does not describe it** (MEASURED). The top 20 bits (FP8) or 21 bits (BF8) of the second operand are a random number added just below the
  last kept bit; the result is truncated. The probability of rounding up equals the input's position between its
  two codes, exactly, in steps of 2^-20 (2^-21), also for subnormal results. The low 12 (11) bits of the operand
  are ignored. A rule built from the guide's instruction entry (add without carry into the exponent, then round to nearest)
  disagrees with the card on 47 % (FP8) and 51 % (BF8) of the conversions in the test set; the guide's general
  section on these conversions says the result is truncated after the seed is added, which is what the card does.
- **Cheapest exact way to widen FP8 weights:** `v_cvt_f32_fp8` four times or `v_cvt_pk_f32_fp8` twice per 32-bit
  word, both **1.0 cycle per weight** to FP32 (the packed form is half the instructions but each takes 2 cycles).
  To FP16 pairs (the operand of the FP16 matrix instruction and of `v_dot2_f32_f16`): packed conversion plus
  `v_cvt_pk_rtz_f16_f32`, **1.5 cycles per weight, exact, and a NaN code stays a NaN** (its sign and payload were not compared), cheaper than the RDNA 3 shift trick
  (1.875, NaN codes become +-480). There is no direct FP8-to-FP16 instruction. Not widening at all,
  `v_dot4_f32_fp8_fp8` takes four weights per cycle, but the activations must then be FP8 too.
- **The dot instructions are not exact dot products: they use the matrix unit's adder** (MEASURED). Lane 5's model
  of one adder step reproduces all eight float forms bit for bit, NaN results compared only as NaN here and bit for
  bit in the leftovers pass (item 5a: the matrix instructions' NaN words) (0 mismatches on 2 x 10^7 rows per FP8/BF8 form and
  2 x 10^6 per 16-bit form), once two rules are added that only BF16 inputs can reach (below). So the
  products and the accumulator are added in a fixed-point window whose last bit is about 2^-24 of the largest
  product, the bits below are cut off (floor), and the sum is rounded once. A small term next to a large one disappears even when the
  large ones cancel: 0 + 2^-18 + 448^2 - 448^2 gives 0, not 2^-18. None of the guide's orderings fits.
  MODE rounding and denormal bits change nothing; FP16_OVFL clamps the FP16 form.
- **What they are good for:** `v_dot4_f32_fp8_fp8` issues one per cycle (4 multiply-adds per lane per cycle, latency
  6), four times an FMA and twice `v_dot2_f32_f16`; the FP8 matrix instruction does about 16 per lane per cycle.
  Use dot4 for FP8 x FP8 work too small for the matrix unit (one-column decode with FP8 activations); widen to FP16
  for anything that needs FP16 or FP32 activations.
- **FP16 is ordinary IEEE** (item 6, helper-written probe, rerun here): FP32 to FP16 rounds to nearest even and,
  unlike the FP8 conversions, follows the MODE FP16 rounding field; packed FP16 FMA is fused; the mixed FMAs round
  once. There is no BF16 conversion instruction; the compiler's integer sequence is exact.
- **Integer dots** (`v_dot4_i32_iu8`, `v_dot8_i32_iu4`) are exact and wrap around at 32 bits without the clamp bit;
  with it the exact sum saturates to the signed 32-bit range, also for unsigned x unsigned; `neg_lo` bit 0 makes A
  signed, bit 1 B (item 7). One per cycle, latency 6 (MEASURED). On RDNA 3 the int8 dot ran at half rate.
- **Dot modifiers** (item 5a): NEG and NEG_HI on A or B negate the low or high element before the adder; on C,
  NEG flips the sign and NEG_HI takes the absolute value (also on the FP8 forms, where the matrix instruction ignores
  the same bits); `clamp` on the FP32-output forms is ignored; OPSEL selects halves.

## 1. FP8 and BF8 to FP32

`cvt_exact decode`: every code in every byte position (the other bytes hold unrelated codes), each of the 4 selects,
and all 65536 byte pairs in each half for the packed forms, against `cvt_ref.h` (itself checked against llama.cpp's
E4M3 decoder and the CPU's FP16 conversion by `cvt_refcheck`).

| Instruction | Result (MEASURED) | Cycles: independent / chained |
|---|---|---|
| `v_cvt_f32_fp8`, `_bf8` (VOP1, byte 0) | exact, 256 codes x 4 positions | 1.00 / 5.00 |
| `v_cvt_f32_fp8_e64 ... byte_sel:n` | select n reads byte n (as numbered) | 1.00 / 5.00 |
| `v_cvt_pk_f32_fp8`, `_bf8` | exact, all 65536 pairs, low byte to the first register; op_sel picks the high half | 2.00 / 5.00 |

- NaN codes (FP8 0x7F/0xFF, BF8 0x7D-0x7F and 0xFD-0xFF) all give 0xFFC00000; BF8 infinities give +-infinity;
  -0 gives -0; subnormal codes give their exact value (MEASURED).
- The packed form writes two registers and occupies the unit for 2 cycles, so it converts at the same rate per
  value as the single form (MEASURED, item 4).
- RDNA 3 has no FP8 instructions; this whole item was software there (READ, rdna3-fp8 README).

## 2. FP32 to FP8 and BF8 (`v_cvt_pk_fp8_f32`, `v_cvt_pk_bf8_f32`)

`cvt_exact encode <fmt> <setting>`: all 2^32 FP32 inputs in the first slot and (input XOR 0x9E3779B9) in the second,
both word selects (the other half of the destination must survive), judged in place by `cvt_ref.h` (round to nearest
even; either NaN code accepted and counted). Input classes: `tie` and `tie_sub` are exact halfway points in the
normal and subnormal target range; `max_to_mid` is (max, midpoint to the next code), `at_mid` the midpoint itself
(464 for FP8, 61440 for BF8), `above_mid` beyond it.

| Setting (MODE bits written) | FP8: mismatches / 2^33 | BF8: mismatches / 2^33 | Control |
|---|---|---|---|
| plain (as launched, 0x010000F0) | 0 | 0 | ok |
| ovfl (FP16_OVFL = 1), reference saturating finite values only | 0 | 0 | ok (`v_add_f16` 65504+65504 = 0x7BFF) |
| rup / rdn / rtz (both rounding fields) | 0 against nearest even | 0 against nearest even | ok (`v_add_f32` rounds as set) |
| noden (both denormal fields 0) | 0 | 0 | ok (`v_mul_f32` 2^-127 = 0) |
| ovfl + rtz | 0 | 0 | ok |

Against a directed-rounding reference the same runs differ on 1.7 x 10^8 to 2.2 x 10^9 first-slot inputs,
depending on the setting, so a MODE effect would have shown. The NaN rule (the sign is lost) is read from the
histogram below; the check itself accepts either NaN code for a NaN input. Histogram of the codes the hardware wrote (first slot, plain setting):

| Class | FP8 output codes | BF8 output codes |
|---|---|---|
| NaN (16,777,214 inputs) | 0xFF only | 0xFE only |
| +inf / -inf | 0x7F / 0xFF (NaN) | 0x7C / 0xFC (infinity) |
| above_mid | 0x7F, 0xFF (NaN, sign kept) | 0x7C, 0xFC |
| at_mid | 0x7E, 0xFE (tie goes to 448) | 0x7C, 0xFC (tie goes to infinity: 0x7B is odd) |
| max_to_mid | 0x7E, 0xFE | 0x7B, 0xFB |
| FP32 subnormals | 0x00, 0x80 (signed zero) | 0x00, 0x80 |
| zero | 0x00, 0x80 | 0x00, 0x80 |

- With FP16_OVFL: above_mid and at_mid give 0x7E/0xFE (FP8) and 0x7B/0xFB (BF8); infinity gives 0x7F/0xFF (FP8,
  NaN) and 0x7C/0xFC (BF8, infinity), against the guide's table (READ: +-max for both). This confirms the first
  hour's finding and Mesa's comment, and extends it to BF8.
- Cycles: 1.00 independent, 5.00 chained (MEASURED; the instruction also reads its destination, which keeps the
  untouched half).
- Two gaps of the older reference are closed: either NaN code is accepted (counted apart: 8,388,607 positive NaN
  inputs per slot get the 0xFF code), and the class (448, 464] is counted (1,048,574 inputs, 0 mismatches).

## 3. Stochastic rounding (`v_cvt_sr_fp8_f32`, `v_cvt_sr_bf8_f32`)

`cvt_sr rule`: 56,514 (FP8) / 56,502 (BF8) inputs, each converted with every value of the operand's top 20 / 21 bits
(5.9 x 10^10 and 1.2 x 10^11 conversions). `cvt_sr sweep`: all 2^32 inputs with a pseudo-random operand, byte select
cycling 0-3, each converted twice with the operand's low bits flipped.

| Candidate rule (`cvt_ref.h` sr8) | FP8: mismatching conversions | BF8 |
|---|---|---|
| the guide's instruction entry as read: add to the mantissa without carry, then round to nearest | 2.78 x 10^10 (46.9 %) | 6.06 x 10^10 (51.2 %) |
| add with carry, round to nearest | 2.56 x 10^10 (43.3 %) | 5.18 x 10^10 (43.7 %) |
| add without carry, truncate | 9.34 x 10^9 (15.8 %) | 2.41 x 10^10 (20.3 %) |
| add with carry, truncate | 6.54 x 10^9 (11.0 %; subnormal targets and overflow only) | 1.25 x 10^10 (10.5 %) |
| **random aligned below the kept bit, truncate, overflow as for round to nearest** | **0** | **0** |

(Percentages of this test set, which is weighted toward edge cases; MEASURED.)

The accepted rule (MEASURED):
- Let the result keep k mantissa bits. The operand's top 20 (FP8) or 21 (BF8) bits form R. R is added to the input's
  magnitude so that R's top bit sits just below the result's last kept bit, a carry moves into the exponent, and the
  bits below the kept ones are dropped. In the subnormal target range the same alignment holds (R is shifted down
  with the result's step); bits of the input below R's last bit are dropped first.
- So the number of operand values that round up equals the input's position between its two codes, in units of
  2^-20 (2^-21) of a step, exactly: 0 inputs out of 56,514 deviate. Exact codes never round up; ties round up for
  exactly half of the operands.
- The operand's low 12 (FP8) or 11 (BF8) bits change nothing (0 of 2^32 results changed when flipped).
- A sum above the largest finite code gives NaN (FP8) or infinity (BF8); with FP16_OVFL it gives the largest finite
  code, but infinity again stays NaN / infinity. NaN gives 0xFF / 0xFE. The three other destination bytes are kept.
- MODE rounding and denormal bits: no effect (sweeps under rtz and noden: 0 mismatches).
- Cycles: 1.00 independent, 5.00 chained (MEASURED).
- For a kernel: give each lane a fresh 32-bit random word per conversion; only its top 20 bits matter.

Added by the timing leftovers pass (`src/study/tim_sr.hip`, `tim_bf8search.cpp`, written by a Sonnet helper from a
sealed brief, read, rebuilt and rerun by the lane owner; run `runs/20261117T175731Z-timing/`, MEASURED):
- Byte selects 1, 2 and 3 (`tim_sr select`): 4,448 FP8 and 4,424 BF8 inputs (exact codes, ties in the normal and
  the subnormal target range and their FP32 neighbours, the overflow midpoint classes, ordinary and subnormal
  values, both signs, the specials), each converted with every value of the operand's top 20 / 21 bits (the bits
  below cycling through 0, all ones and a hash), at every select: 1.4 x 10^10 and 2.8 x 10^10 conversions, 0
  mismatches against the accepted rule in every class and select, and the three other destination bytes kept in
  every conversion. NaN inputs give the hardware's NaN code (0xFF / 0xFE) as at select 0. Judged against the
  "add with carry, round to nearest" candidate instead (`--break`): 6.2 x 10^9 and 1.2 x 10^10 mismatches, FAIL.
- How many random bits (`tim_sr bits`), pinned by changed-rule controls rather than by exclusion: the accepted rule
  rewritten with the operand cut to its top B bits (equal to the accepted rule at B = 20 / 21 on 1.5 x 10^8 CPU
  cases), and the hardware compared at B - 1, B and B + 1 on 18,648 inputs x 2^20 / 2^21 operands:

| Format | B - 1 | B | B + 1 |
|---|---|---|---|
| FP8 (19 / 20 / 21 bits) | 9,213 mismatching conversions | 0 | 2,534 |
| BF8 (20 / 21 / 22 bits) | 9,212 | 0 | 2,547 |

  Both controls fail, so the count is pinned at 20 (FP8) and 21 (BF8). The B + 1 control can only fail where the
  target is subnormal: in the normal range a 21st (22nd) bit would sit below the FP32 input's last bit, so it cannot
  change the sum there; the B + 1 mismatches come from the subnormal-target and deep-subnormal inputs (FP8 1,011
  and 1,509) and from 14 tie neighbours.
- The BF8 encode reference checked on all 2^32 inputs against the independent search over the sorted code values
  (`tim_bf8search`, CPU, 8 threads, 15.5 s), in all 8 modes (4 rounding directions, saturation on and off): 0
  mismatches. With the reference rounding ties away (`--break`): 124 mismatches in each of the two nearest-even
  modes, FAIL. Both 8-bit encode references are now checked exhaustively.

## 4. Widening E4M3 weights

Every method checked on all 2^32 words (`cvt_widen`, the same instruction sequences as the timed blocks): 1.7 x 10^10
non-NaN weights each. Timed with 8 independent words per block, instructions scheduled across the words (`cvt_time`,
widen_* tests), and again with the instructions that consume the result (use_* tests: one accumulator per word,
activation in a register).

| Method | Instructions per weight | Cycles per weight (widen only) | With its consumer | Exact | Registers out per word | Feeds |
|---|---|---|---|---|---|---|
| (a) 4 x `v_cvt_f32_fp8` | 1 | 1.00 | 2.41 (+4 `v_fmac_f32`) | yes; a NaN code stays a NaN (sign and payload not compared) | 4 (FP32) | vector FMA (dual issue possible, INFERRED) |
| (b) 2 x `v_cvt_pk_f32_fp8` | 0.5 | 1.00 | 2.41 (+4 `v_fmac_f32`) | yes; a NaN code stays a NaN (sign and payload not compared) | 4 (FP32) | vector FMA |
| (c) shifts and masks (RDNA 3) | 1.75 | 1.875 | 2.5 to 2.9 (+2 `v_dot2_f32_f16`; this row is not linear in the block count: 2.53 per weight between 4 and 6 blocks, 2.89 between 6 and 8, 2.71 overall) | yes x 2^-8, NaN codes become +-480 x 2^-8 | 2 (FP16 pairs, order 0,2 / 1,3) | FP16 matrix, `v_dot2_f32_f16`, `v_pk_fma_f16`, `v_fma_mix` |
| (d) 2 x pk + 2 x `v_cvt_pk_rtz_f16_f32` | 1 | 1.50 | 2.00 (+2 `v_dot2_f32_f16`) | yes; a NaN code stays a NaN (sign and payload not compared) | 2 (FP16 pairs) | same as (c) |
| (e) FP16 table in LDS, 4 x `ds_load_u16_d16(_hi)` | 2 VALU + 1 LDS | 4.59 | 4.62 | yes; a NaN code stays a NaN (sign and payload not compared) | 2 (FP16 pairs) | same as (c) |
| (f) no widening: `v_dot4_f32_fp8_fp8` | 0.25 | - | 0.25 | the adder of item 5 | - | itself (needs FP8 activations) |

- (d) is the cheapest exact path to FP16 and the only cheap one that keeps NaN codes. (c) needs no FP8 hardware and
  stays the choice for RDNA 3 (where it cost about 35 cycles per 8 codes in the Vulkan probe, READ, rdna3-fp8
  README).
- (a) and (b) cost the same per weight: the packed conversion occupies the unit for 2 cycles (item 1). (b) uses half
  the instruction bytes.
- The consumers: 4 `v_fmac_f32` per word took 5.6 cycles instead of 4 here (an accumulator, a weight and the
  activation sometimes fall in one register bank, lane 1 item 3); with `v_dual_fmac_f32` two FMAs would issue per
  cycle (INFERRED, not timed here).
- (e) is latency-bound here (one wait per block of 32 loads) and its timing varies by about 5 % between builds of
  4, 6 and 8 blocks; it is clearly the slowest, so it was not refined.
- Exactness: (a), (b), (d), (e) 0 wrong of 17,045,651,456; NaN codes stay NaN (134,217,728 each). (c) 0 wrong, NaN
  codes not NaN (known from RDNA 3). The deliberate break (one table entry wrong in (e)) gives 67,108,864 wrong
  weights = 4 positions x 2^24 words, as it must.
- The FP16 matrix instruction takes FP16 operands; (c), (d), (e) produce them directly. FP32 results (a, b) feed only
  the vector unit.

## 5. Dot products

`cvt_dot run`: per form, targeted rows (a big product B, its negative, a tiny product t and zeros in every
arrangement of the slots, with C = 0, B, -B or t), then random rows (10^7 per FP8/BF8 form, 10^6 per 16-bit form;
ordinary values, values near zero, random bit patterns, infinities and NaNs, cancelling pairs, C near the products,
C = -sum). Repeated on a second sample (seed 777). Rules (`cvt_dot_model.h`, `cvt_dot_adder.h`):

| Form | exact, one rounding | guide order (FMA chain) | reversed | products then C | lane 5's adder before its fix | **adder (final rule)** | adder, 2 steps (changed rule) |
|---|---|---|---|---|---|---|---|
| `v_dot4_f32_fp8_fp8` (10^7) | 729,282 | 826,853 | 827,079 | 775,262 | 0 | **0** | 392,849 |
| `v_dot4_f32_fp8_bf8` | 1,601,326 | 1,607,809 | 1,608,244 | 1,523,567 | 0 | **0** | 954,742 |
| `v_dot4_f32_bf8_fp8` | 1,600,599 | 1,606,974 | 1,608,537 | 1,523,306 | 0 | **0** | 954,649 |
| `v_dot4_f32_bf8_bf8` | 1,855,945 | 1,782,115 | 1,783,132 | 1,707,049 | 0 | **0** | 1,133,057 |
| `v_dot2_f32_f16` (10^6) | 133,111 | 115,737 | 115,863 | 124,709 | 0 | **0** | 92,200 |
| `v_dot2_f32_bf16` | 112,442 | 94,793 | 94,660 | 99,470 | 3,715 | **0** | 89,007 |
| `v_dot2_f16_f16` | 17,749 | 272,043 | 272,233 | 148,702 | 0 | **0** | 248,580 |
| `v_dot2_bf16_bf16` | 238,451 | 425,052 | 425,170 | 309,336 | 2,565 | **0** | 254,274 |

(MEASURED in the owner's run, seed 1000. The seed-777 sample: final rule 0 in every form; lane 5's adder before
its fix 3,629 and 2,651 on the two BF16 forms. In the lead's rerun lane 5's model had been fixed: that column is 0
for `v_dot2_f32_bf16` in both samples and 2,488 (seed 777: 2,567) for `v_dot2_bf16_bf16`; its "2 steps" control
counts moved to 74,477 and 253,444. The remaining rows are not a difference between the models: the probe feeds
the two products to the first step of the matrix model and leaves the later steps empty, and an empty matrix step
turns -0 into +0, which a one-step dot instruction does not do. The first reviewer reported running both
models on 400,000 random rows per form with the products placed in the last step, all eight forms agreeing on every
row; that check is in the review, not in a log here.)

The rule (MEASURED; lane 5's adder, one step):
- Anchor E = the largest of (ea + eb + 2) over the nonzero products (exponent fields, subnormals at the minimum
  exponent) and C's exponent, **but never below -126**. Unit U = 2^(E - 26).
- Products are summed exactly in runs (the four products of an 8-bit form together, the pair of a 16-bit form), each
  first floored to its run's own unit 2^(E1 - 26) (not limited by -126); each run sum is floored to U. C is floored
  to U, or contributes 0 when |C| < U.
- The integer sum is rounded once: FP32 and FP16 outputs to nearest even, BF16 output to FP32 (nearest even) and
  then cut to its top 16 bits (toward zero). A -0 from the FP32 or FP16 rounding becomes +0; a tiny negative FP32
  result cut to BF16 stays -0. FP16 output under FP16_OVFL: overflow gives 65504.
- Special values follow lane 5's rules (NaN input or 0 x inf or +inf - inf gives NaN, infinity otherwise wins).
  NaN results were compared as a class: their payloads were not checked here.
- The two new parts (anchor at least -126; -0 rule before the BF16 cut) change only BF16 rows whose sums fall in the
  FP32 subnormal range. They were found on a first sample (lane 5's model missed 6 + 1,169 rows there), then the
  generator got more near-zero inputs and two new samples were drawn: lane 5's model before its fix missed 3,715 +
  2,565 and 3,629 + 2,651 rows, the rule above none. INFERRED: the adder's window never extends below the FP32 subnormal range plus
  three guard bits.
- `v_dot2_f16_f16` and `v_dot2_bf16_bf16` write the low half and keep the high half of the destination (MEASURED).
- Order: inside one instruction there is no order; all terms meet in one fixed-point sum. The targeted rows show it
  directly: the adder rule agrees with all 60 (dot4) and 8 (dot2) targeted rows, and every arrangement of
  {t, B, -B, 0} with C = 0 gives 0 (exact would be t = 2^-18 for FP8; the guide's FMA chain keeps t whenever t comes
  after both B and -B).
- Error size (INFERRED from the rule, as lane 5): at most about 2^-24 of the largest term per instruction, plus the
  final rounding; exact when all terms lie on a grid of 2^-26 of the largest term.

MODE (`cvt_dot mode`, 10^6 rows per form): under rtz, rup, rdn and noden every float form matches the adder rule
with 0 mismatches (the one-rounding rule in the set direction differs on 1.8 to 46 % of rows), so rounding and denormal
bits are ignored. Under FP16_OVFL only `v_dot2_f16_f16` changes (41,608 rows), all explained by overflow giving 65504.

Cost (MEASURED, cycles per instruction, `cvt_time`):

| Instruction | Independent | Chained through the accumulator | Multiply-adds per lane per cycle |
|---|---|---|---|
| `v_dot4_f32_fp8_fp8` and the 3 mixes | 1.00 | 6.00 | 4 |
| `v_dot2_f32_f16`, `v_dot2_f32_bf16` | 1.00 | 6.00 (lane 1: 6) | 2 (4 with `v_dual_dot2acc`, lane 1) |
| `v_dot2_f16_f16`, `v_dot2_bf16_bf16` | 1.00 | 6.00 | 2 |
| `v_dot4_i32_iu8` | 1.00 | 6.00 | 4 |
| `v_dot8_i32_iu4` | 1.00 | 6.00 | 8 |
| `v_fma_f32` (calibration) | 1.00 | 5.00 | 1 (2 with dual issue) |
| FP8 matrix instruction 16x16x16 (READ, first hour: 8.1 cycles) | | | about 16 |
| FP16 matrix instruction (READ, first hour: 16.1 cycles) | | | about 8 |

INFERRED: six independent accumulators keep dot4 at full rate. The dot4 FP8 forms have no dual-issue encoding.

### 5a. NaN words and modifier bits (leftovers pass, MEASURED, `runs/20261117T165551Z-leftmat/`)

`src/study/left_dot.hip` (written by a Sonnet 5.5 helper from a sealed brief; the lead read it, rebuilt it from
`scripts/study-leftovers-matrix.sh` and reran it: the run above, every mode PASS, every deliberate break FAIL, the
same verdicts and counts as the helper's own logs).

NaN results, every case in every slot position, two launches identical (1,134 cases over the 8 forms):

| Case | FP32 output (dot4 FP8/BF8 forms; dot2 FP16 and BF16 into FP32) | FP16 output | BF16 output |
|---|---|---|---|
| NaN in A, in B, or both (any payload, sign, both E4M3 NaN codes), also with a NaN C | 0xFFC00A3D | 0xFE3D | 0xFFFD |
| 0 x inf, +inf and -inf products, an infinite product against an opposite infinite C | 0xFFC00A3D (8-bit inputs), 0xFFC00000 (16-bit inputs) | 0xFE00 | 0xFFC0 |
| NaN in C only | C quieted, payload and sign kept (0x7F800001 gives 0x7FC00001, 0xFFC12345 stays) | the same in 16 bits | the same |
| controls: one infinity, infinite C, finite rows | never NaN | | |

Every class equals the matrix instructions' value (`matrix-arithmetic.md` sections 3 and 8): the dot instructions
make NaNs exactly like one matrix step. FP8 x FP8 has no infinity code, so its infinity classes do not exist.
Deliberate break (expect 0x7FC00000 for NaN inputs): 304 cases differ, FAIL.

Modifier bits. Accepted by `llvm-mc -mcpu=gfx1201` (READ; 80 of 250 candidate spellings): on the FP8/BF8 dot4
forms only `neg_lo:[0,0,1]` and `neg_hi:[0,0,1]`; on `v_dot2_f32_f16` and `_bf16` `neg_lo`, `neg_hi`, `op_sel`,
`op_sel_hi`, `clamp`; on `v_dot2_f16_f16` and `v_dot2_bf16_bf16` the source modifiers `-x`, `|x|` and
`op_sel:[0,0,1]`, `op_sel:[0,0,0,1]`; on the integer forms `neg_lo`, `neg_hi`, `clamp`. Each accepted bit on the
card, against the plain instruction on data changed in memory (10^6 rows; every claim below has 0 mismatches, and
the plain instruction differs from the modified one on 209,083 to 785,261 rows unless the row says "ignored"):

| Bit | Forms | Effect |
|---|---|---|
| `neg_lo:[1,0,0]` / `[0,1,0]` | dot2 into FP32 | sign of the low element of A (or of B) flipped; A and B give the same result |
| `neg_hi:[1,0,0]` / `[0,1,0]` | dot2 into FP32 | sign of the high element flipped |
| `neg_lo:[0,0,1]` | dot4 FP8/BF8, dot2 into FP32 | sign of C flipped |
| `neg_hi:[0,0,1]` | dot4 FP8/BF8, dot2 into FP32 | absolute value of C (not a flip; the matrix FP8 forms ignore this bit, `matrix-arithmetic.md` 9a) |
| `op_sel:[1,0,0]` / `[0,1,0]` | dot2 into FP32 | both elements read the high half of A (B) |
| `op_sel_hi:[0,1,1]` / `[1,0,1]` | dot2 into FP32 | both elements read the low half of A (B) |
| `op_sel:[0,0,1]`, `op_sel_hi:[1,1,0]`, `clamp` | dot2 into FP32 | ignored: equal to the plain result on all 10^6 rows (the clamp bit is set in the encoding; a [0, 1] clamp would have changed 794,091 and 818,407 rows) |
| `op_sel:[0,0,0,1]` | dot2 into FP32 | the assembler drops it (same encoding as plain) |
| `-x`, `|x|` on A or B | dot2 into FP16 / BF16 | both elements negated, or both signs cleared |
| `-x`, `|x|` on C | dot2 into FP16 / BF16 | act on the low half of C (the half that is read) |
| `op_sel:[0,0,1]` | dot2 into FP16 / BF16 | C is read from its high half (marker data: C = {2.0, 4.0}, A x B = 1 gives 5.0) |
| `op_sel:[0,0,0,1]` | dot2 into FP16 / BF16 | the result goes to the high half of D, the low half keeps the old D |

Deliberate break (every modifier judged "ignored"): 44 modifiers disagree, FAIL. Not run on the card: `neg_hi` on
the integer forms, the all-ones `op_sel_hi`, combined bits.

## 6. FP16 and BF16 conversions, packed 16-bit math, mixed FMA

`cvt_f16` (written by a Sonnet helper; reviewed, rebuilt and rerun here). Each reference is integer-only and
self-tested on the CPU first (x86 F16C conversions in all four rounding modes, double arithmetic, `fmaf`, a
nearest-value rule). Timing of these instructions is lane 1's (`v_cvt_f16_f32`, `v_cvt_f32_f16`,
`v_cvt_pk_rtz_f16_f32`, `v_pk_*_f16`, `v_fma_mix_f32`: 1 cycle, latency 5).

| Operation | Inputs | Result (MEASURED) |
|---|---|---|
| `v_cvt_f16_f32` (and the `(_Float16)` cast) | all 2^32 | exact round to nearest even in the default MODE; NaN keeps sign and top payload bits, quieted |
| same, MODE rounding field [3:2] = up / down / toward zero | all 2^32 each | follows the set direction exactly (0 mismatches); the FP32 field [1:0] has no effect |
| same, FP16_OVFL = 1 | all 2^32 | finite overflow gives 0x7BFF (65504); infinity stays infinity, in every rounding mode |
| `v_cvt_pk_rtz_f16_f32` | all 2^32 in both halves | toward zero always; MODE rounding has no effect |
| `v_cvt_f32_f16` | all 65536 | exact; NaN quiet bit set, payload kept; subnormal inputs flushed to zero only when MODE[7:6] (FP16/FP64 denormals) = 0, not by [5:4] |
| FP32 to BF16 | | no instruction on gfx1201 (READ: ISA text and builtins). The compiler emits about 6 integer instructions (`v_bfe_u32` of bit 16, `v_add3_u32` with 0x7FFF, an unordered compare, `v_or_b32` 0x400000 for NaN, `v_cndmask_b32`, shift right 16; READ from the listing); it is exact round to nearest even on all 2^32 inputs, NaN quieted with payload |
| BF16 to FP32 | all 65536 | the compiler's shift: exact |
| `v_pk_fma_f16` | 1.2 x 10^7 words | fused (one rounding): 0 mismatches; in 2.1 x 10^6 cases where an unfused result differs, the card never gives it |
| `v_pk_add_f16`, `v_pk_mul_f16` | 1.2 x 10^7 words each | IEEE, round to nearest even, halves independent: 0 mismatches |
| packed NaN results | | the first NaN operand, quieted, sign kept; an invalid operation (inf x 0, inf - inf) gives 0xFE00 |
| `v_fma_mix_f32` | 1.08 x 10^6, all 27 operand-source patterns | FP16 sources widened exactly, one rounding to FP32: 0 mismatches |
| `v_fma_mixlo_f16`, `v_fma_mixhi_f16` | 1.08 x 10^6 each | one rounding straight to FP16 (in 45,000 cases where rounding to FP32 first would differ, the card never does); the other half of the destination is kept |

For FP8 weights: every E4M3 value is an FP16 value, so the FP16 path loses nothing; only the RDNA 3 shift trick
needs the 2^-8 scale (item 4).

## 7. Integer dot products (Q8_0 comparison point)

`v_dot4_i32_iu8` and `v_dot8_i32_iu4`, signed x signed (`neg_lo:[1,1,0]`), no clamp bit: 0 mismatches on 10^6 rows
each against the exact sum wrapped to 32 bits; a saturating rule disagrees on 248,128 and 124,131 rows (accumulators
near INT32_MAX/MIN were included) (MEASURED).

Sign settings and clamp (leftovers pass, MEASURED, `left_dot int`, `runs/20261117T165551Z-leftmat/`; 10^6 rows per form and setting,
uniform codes and extremes with C near INT32_MIN, INT32_MAX, 0 and UINT32_MAX): `neg_lo` bit 0 makes A signed and
bit 1 B (the swapped reading disagrees on 833,921 to 864,888 rows in the mixed settings). Without `clamp` every
setting is the exact sum wrapped to 32 bits. With `clamp` the exact sum, C read as signed, saturates to
[INT32_MIN, INT32_MAX] in all four settings, unsigned x unsigned included (0 mismatches); wrap-around disagrees on
2,295 to 81,753 rows per setting, unsigned saturation on 4,482 to 163,350. The dot instruction saturates once (one
step), as the matrix integer forms do per step. Signed x signed without clamp equals item 7's rule (0 of 10^6).
Deliberate break (clamp rows judged by wrap-around): FAIL.

Both issue one per cycle with a 6-cycle chain (MEASURED); on RDNA 3 the
int8 dot ran at half rate (READ, rdna3-fp8 README: 32 instructions in 62.6 cycles). So per instruction an FP8 dot4
and an int8 dot4 cost the same; the int8 one is exact, the FP8 one goes through the adder of item 5.

## Method

- References: `src/study/cvt_ref.h` (integer-only FP8/BF8 decode, encode in four rounding directions with or
  without saturation, input classes, stochastic-rounding candidates), tested by `cvt_refcheck` on the CPU: decode
  against llama.cpp's own E4M3 conversion and the CPU's FP16 conversion (BF8 is the top byte of an FP16); encode
  against a search over the sorted code values on 268 million boundary and random inputs in all modes; FP8 encode
  against the study's older `e4m3_ref.h` on all 2^32 inputs, BF8 encode against the search on all 2^32 inputs in all 8
  modes (leftovers pass, `tim_bf8search`); the stochastic rule against "round-up count = position".
  Dot rules: `cvt_dot selftest` checks the sequential rule against the CPU's `fmaf` and the one-rounding rule against
  double arithmetic (exact in the tested range). The accepted adder rule reuses lane 5's helpers
  (`arith_model.h`) and agrees with lane 5's own model wherever no BF16 term reaches the FP32 subnormal range.
- MODE: written with `s_setreg_b32` (rounding [3:0], denormals [7:4], FP16_OVFL [23]) at the start of each kernel and
  restored at the end; every kernel also runs the vector-ALU control (`cvt_gpu.h`) and reads MODE back, and the
  verdict fails if the control does not show the setting.
- Conversions inside MODE-sensitive kernels are inline assembly, so the compiler cannot move or fold them.
- Timing: lane 1's differential method (`core_common.h`): one wave alone, blocks built at U = 4, 6, 8, cost =
  (U=8 - U=4) / 4 blocks; U = 6 must sit midway (within 1.5 cycles per iteration or 5 %). The blocks are generated
  with fixed registers (`scripts/cvt_gen_timing.py` -> `src/study/cvt_time_blocks.h`; the script checks the header
  matches). Every timed loop was checked in the `.lto.s` listing with `core_loopcheck.py`: 0 extra instructions.
  Calibration `v_fma_f32` 1.00 / 5.00 reproduces lane 1.
- Whole-card sweeps (2^32 inputs) take well under a second each; the script waits for the hotspot to fall below 65 C
  before each and would stop at 95 C. Highest hotspot in the final run: 81 C read by the script after a job; never above 65 C at the start of a
  whole-card job (MEASURED).
- Deliberate breaks (each must FAIL; run by the script):

| Break | Result (final run) |
|---|---|
| `cvt_refcheck --break`: reference rounds ties away | FAIL: 648 mismatches against the search, 254 against `e4m3_ref.h` |
| `cvt_exact encode fp8 plain --break`: GPU reference rounds ties away | FAIL: 256 mismatches (the tie classes) |
| `cvt_widen --break`: one LDS table entry wrong | FAIL: 67,108,864 wrong weights (4 positions x 2^24 words) |
| `cvt_dot run --break`: dot model rounds once more (products over two adder steps) | FAIL in every float form, e.g. 392,849 of 10^7 for FP8 x FP8 |
| `cvt_dot selftest --break`: sequential rule with the additions reversed | FAIL: 186,901 mismatches against `fmaf` |
| `cvt_sr sweep fp8 plain --break`: judged against the guide's rule | FAIL: 167,939,720 mismatches |
| `cvt_f16 all --break`: RNE reference rounds ties away; packed FMA reference rounds twice | FAIL: 31,744 and 1,502,877 mismatches |

## How to rerun

```
cd <study repository>
scripts/study-conversions.sh all        # build, CPU self-tests, GPU runs, breaks; about 5 minutes
scripts/study-conversions.sh run        # GPU runs only (after build)
```
Requires the card record `runs/latest/card.env`, ROCm 10.1 at `$HOME/.local/opt/rocm`, and llama.cpp's
`libggml-base` (`LLAMA_DIR`, default the rdna3-fp8 build) for `cvt_refcheck`. Results land in
`runs/<UTC time>-conv/summary.md`. Exploration switches: `cvt_dot run --form N --show RULE --seed S`,
`cvt_time <test-name-part>`.

## Not measured

Still open:

- Dual-issue (VOPD) consumers in the widening loops: timed with plain `v_fmac_f32` only; `v_dual_fmac_f32` would
  halve the FMA cost (INFERRED from lane 1). What is open now: what keeps the dual-issue consumer at 2.19 cycles per
  weight instead of the 1.5 issue bound.
  Answered in part (leftovers C, CD-1, `measurements/leftovers-c-a.md`): with the consumers as `v_dual_fmac_f32`
  pairs a widened weight costs 2.19 cycles instead of 2.41, for both widening forms (the FMA cost is not halved). The
  32 conversions and 16 pairs of a block take 70 cycles for 48 instructions; INFERRED (not established): the
  conversions' results feeding the pairs. MEASURED.
- The LDS table method with loads issued ahead of their use (software pipelining): only the simple form was timed.
  What is open now: why the method is bimodal and why issuing its loads ahead is slower.
  Answered in part (leftovers C, CD-2, `measurements/leftovers-c-a.md`): issuing the loads ahead is slower, not
  faster: 5.5 cycles per weight against 4.6 for the simple form in its fast mode (the simple form has a slow mode of
  5.15 to 5.91; against that mode "ahead" was slower in one of three runs only). A form with at most 16 loads in
  flight is slower still (7.19, against 5.50 for 32 in flight; test T4), so the test does not support the issue stop
  of lds-and-special-loads LD-2 as the cause. The half-register candidate (the `_d16` / `_d16_hi` pairs write halves of
  the same registers) is untested. MEASURED.
- Dot products: `neg_hi` on the integer forms and combined modifier bits were not run on the card (the single bits
  were, item 5a). What is left: the full set of 2^k modifier combinations; 64 chosen combinations were run, of 256 for
  the two sources' bits alone.
  Answered in part (leftovers C, CD-3, `measurements/leftovers-c-a.md`): for `v_dot2_f32_f16` all 64 combinations
  equal the composition model on every row (0 mismatches; the bits act independently: half selection, then the sign of
  each element, and |C| before its negation). On `v_dot4_i32_iu8` and `v_dot8_i32_iu4`, `neg_hi` is ignored on A, B
  and C, as on the matrix integer forms. MEASURED.
- New from leftovers C: why the LDS table method is bimodal, also inside one process (2 and 5 of 12 samples slow in
  two processes, 0 in the third; CD-2, `measurements/leftovers-c-a.md`).

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, CD-4, `measurements/leftovers-c-a.md`): stochastic rounding with a random-number generator in
  the loop: `v_cvt_sr_fp8_f32` plus a 24-bit LCG costs 3.000 cycles per conversion, one shift per conversion (one
  random word for four) 2.000, a 32-bit LCG 6.12, xorshift32 7.25. The cost is additive and issue-bound when the
  conversion's value and the random operand sit in different register banks; with both sources in one bank every
  conversion costs a quarter cycle more. The cheapest generator is one random word shifted for four conversions. The
  final rerun moved two of these (xorshift 7.42, 32-bit LCG 6.00). MEASURED.
