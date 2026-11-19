# Lane 5 test plan: what the RDNA 4 matrix instructions compute (draft for the lead)

Status: DRAFT by the design sub-agent; nothing was run. "To check" marks guesses. Outputs: measurements/matrix-arithmetic.md and
measurements/errors-after-quantizing.md.

## 0. Method

- One output element: d = C + sum of 16 products p_s = a_s * b_s. Leaves: 16 slots plus C (sparse: 32 slots, 16 selected).
- SLOT, not K. A slot is a place in the operand image (lane group, register, half or byte). The adder only sees slots. If A and B use
  the same slot-to-K table, the result does not depend on which K a slot stands for. So the open "K order inside a lane" question is a
  labelling question: measure the adder over slots, then state what each published K order (guide, sample code) implies. (To check with the
  lead: that the two orders differ by one permutation applied to both A and B. The identity check catches an A/B mismatch.)
- Trial packing: trial = row m of A; B = 1 (or powers of two) in all columns, so the 16 columns repeat the trial (free determinism check).
  16 trials per instruction. Instructions cost microseconds, so 10^4 to 10^6 are affordable. Chain only through C (documented hazard
  when D feeds A or B).
- Model family, parameters fixed by experiment and never by fitting random data: tree over 17 leaves; per node a width W (bits kept after
  aligning to the larger exponent), a drop rule (truncate magnitude, floor of two's complement, one's complement without carry-in, RNE,
  sticky or not); a normalise and final-round rule. Known RDNA 3 answers: 1 + (-1) = -2^-24; 8 + (-1) = 7 - 2^-21
  (rdna3-fp8/measurements/matrix-arithmetic.md 3.1).
- Each experiment: inputs, outcome per hypothesis, control (a way to fail), falsifier. FP16 first: its products span 2^-48..2^31 (gaps to about
  78 bits). FP8 products span 2^-18..2^17.6 (gap about 35), so for FP8 use C (FP32) as the large term.

## 1. Structure probes

E0 Fingerprint and replay (first; about 40 instructions). Replay the RDNA 3 table 3.1 pairs with B = 1 in every ordered slot pair (i,j):
F[i][j] = result. If F depends on (i,j) only through classes, those classes sketch the tree (RDNA 3: slots 0,1 lose 2^-24, slots 0,8 lose
2^-25). If all F are exact, the adder is wider than FP32; E3 will show it. Control: all-positive rows must equal the exact model; the model
with the drop rule changed by one bit must disagree on the mixed-sign rows. Falsifier: F changes with anything but slots (E1).

E1 Determinism and independence. Same trial in all rows, other rows random; 100 repeats; another wave and CU; after an instruction with other
data and after another matrix opcode; partial EXEC (guide forces all ones; to check). Control: a slot-permuted copy of the trial must change
the E0 result. Falsifier: any bit change, which means hidden state to pin down before anything else.

E2 Are products exact? a = 1 + 2^-i, b = 1 + 2^-j, p = 1 + 2^-i + 2^-j + 2^-(i+j), C = -(1 + 2^-i + 2^-j); output 2^-(i+j) if the whole product
reaches the adder. Sweep t = i+j: FP16 2..20, BF16 2..14, FP8 2..6, BF8 2..4. Exact product: 2^-t for all t. Product cut to w bits: 0 beyond w.
Control: CPU model with the product rounded to 16 bits must disagree. Falsifier: a partial value (not 0 or 2^-t) means alignment cuts the
product, so continue with E3.

E3 Width and drop rule (core). Leaf pair (x,y): L = 1 at x, small s at y, gap g = 1..60.
- E3a: s = (2^11 - 1) * 2^-(g+10) (all-ones pattern, top bit 2^-g). D - 1 shows the kept bits: floor at W gives the pattern cut at 2^-W;
  RN gives a carry to 2^-W; sticky adds a step above a tie. W = the g where D - 1 first reaches 0.
- E3b: all four sign combinations of (L, s). One's complement without carry-in gives 1 - trunc(s) - 2^-W; two's complement floor gives 1 - ceil(s);
  truncation toward zero gives 1 - trunc(s). This is the RDNA 3 -2^-24 mechanism; report if RDNA 4 does not show it.
- E3c: normalisation: terms +1, -(1 - 2^-k) and s at gap g; do bits of s survive the left shift after cancellation? Separates "aligned once then
  normalised" from "FP32 rounding at each node".
- E3d: final round: C = 1, one product m * 2^-25 for m = 1..5, with and without a second product 2^-60 (sticky), both signs. The table (tie, odd
  tie, just below, just above) separates RNE, RNA, RZ, floor and no-sticky.
- Repeat per leaf pair class from E4 (stages may differ by depth), with L = C and s = product, and the reverse.
- Falsifier of "one common W": W differs between pairs or between C and a product.

E4 Grouping by rooted triples (full schedule for K = 16). For leaves (x; y, z): L = 1 at x, equal smalls u = 2^-g at y and z (second pattern
1.5 * 2^-g to avoid ties), g around the E3 threshold. g1*(x,y) = the largest g where one small term still changes D; g2*(x;y,z) = the same with
both. delta = g2* - g1*. delta = 1: y and z were added before meeting x. delta = 0: each was cut separately (single aligned pass, or different
branches). A chain in slot order gives delta = 1 only when y and z both precede x; groups of 4 give delta = 1 inside a group of 4. All triples
determine a rooted tree (a node with more than 2 children shows delta = 0 for all its child pairs).
Schedule: all C(17,3) = 680 triples with the large term in each of its 3 roles = 2040 cases, about 25 gaps, 2 sign patterns: about 10^5 trials, about
6000 instructions. Cheap first cut: x in {slot 0, 5, 8, 15, C} with all pairs (y,z) (about 100 cases each).
Expectations: "groups of 4" (FP16) delta = 1 exactly within a group of 4; "groups of 8" (FP8) within 8; FMA chain as above; very wide exact adder: no
threshold in range. Control: the same cases through a model with two slots swapped between groups; the delta matrix must differ in the affected
triples. E4 alone is not enough: the model must also reproduce E3 numbers. Falsifier: triples that no tree explains (adder not a tree of independent
nodes) leads to E5.

E5 Shared alignment (only if E4 is inconsistent). Same triples plus a huge unrelated term H = 2^e (e = 0..40) at a fourth slot w, or at C. If g1*(x,y)
moves with H, alignment is to a global (or per-group) maximum exponent. Falsifier of a tree-only model: g1* moves with a leaf off the path between x
and y.

E6 C and the 16-bit accumulator forms. C enters as a leaf in E4 (triples with C). For F16 and BF16 accumulate: C and output are 16-bit. Output
rounding: C = 1, sum = 1 + 2^-11 (exact tie in FP16): 1 if RNE; plus 2^-30 sticky: 1 + 2^-10 if sticky kept; FP16 denormal outputs; 65504 + 16 and
65520 (inf or max; with FP16_OVFL, section 3). Inner precision: run E3 with FP16-width gaps and compare with the FP32-accumulate form. Control: the RDNA 3
model "round to FP16 after each pair" must give different E4 deltas from the measured ones, or the probe cannot tell them apart.

E7 Fused or not. BF16 a = 2^-77, b = 2^-76 (product 2^-153) in all 16 slots, C = 0: exact sum 2^-149 (smallest FP32 denormal) if products are exact
and output denormals kept; 0 if products are rounded or flushed first, or the output is flushed. Run with denormal bits 0 and 3. Overflow cancel (BF16
products reach 2^254): +2^200 at one slot and -2^200 at another, others 0: exact 0; products rounded to FP32 first give NaN (inf - inf). Also C =
-FLT_MAX plus a +2^200 product. Falsifier of "fused": NaN in the cancel case.

## 2. Special values (each case at each leaf position, each form; also two NaNs at once)

| Case | Inputs | Read |
|---|---|---|
| inf x 0 | a = inf, b = 0 | NaN and its bits |
| inf - inf | +inf at one slot, -inf at another or in C | NaN; overflow to inf |
| NaN | one NaN, then two with different payloads and signs, in A, B, C; quiet and signalling | payload and sign kept or canonical; which wins by slot |
| Signed zero | every mix of +0, -0 in C and products; x - x with C = +0, -0 | sign of zero |
| Denormal in | FP16 and BF16 subnormals, denormal bits 0 and 3 | flushed or kept |
| Denormal out | results 2^-130..2^-149 | flushed, kept (E7) |
| Overflow | FP32 sums beyond 3.4e38 (BF16); FP16 accumulate 65504 + 65504 | inf or max; FP16_OVFL 0 and 1 |
| FP8 | codes 0x7F, 0xFF (NaN), 0x7E (448), 0x00, 0x80, 0x01..0x07 in A, B, both | NaN result; value range |
| BF8 | 0x7C, 0xFC (inf), 0x7D..0x7F (NaN), 0x7B (57344) | inf as inf; inf x 0 |

Control: same inputs through a VALU op with known behaviour (v_mul_f32; v_cvt for FP8) proves the input arrived. Falsifier: an answer that depends
on slot where it should not.

## 3. MODE register
Bits (guide-A): FP_ROUND [3:0] (2 bits single, 2 bits FP16/double; 0 RNE, 1 +inf, 2 -inf, 3 zero); FP_DENORM [7:4] (single [5:4], FP16 [7:6]; 0 flush
both, 1 input only, 2 output only, 3 both); FP16_OVFL bit 23. No IEEE or DX10_CLAMP on gfx12. Set with s_setreg (syntax to check with llvm-mc), read,
modify, write, restore. Rerun E3d, E7, E6 overflow and FP16 denormal inputs at the corners: rounding {0,1,2,3} x denormal {0,3}, single and FP16
fields separately, FP16_OVFL 0 and 1. Control: v_add_f32, v_mul_f32, v_add_f16 on the same data must change with the setting (guide: affects VALU); if not,
the setreg did not work and the matrix result means nothing. Expectation: no effect (fixed hardware rule) or final step only; a change in E3 widths would
mean inner stages. Falsifier: matrix identical across modes while the VALU control differs means "ignores MODE" (a measured result).

## 4. Modifier bits
Which bits exist: assemble each form offline with llvm-mc -triple=amdgcn -mcpu=gfx1201 -show-encoding using neg_lo, neg_hi, abs, clamp, op_sel, reuse flags
(to check). The guide (rdna4.txt about lines 4780, 4820, 5333): F16/BF16: neg_lo[1:0] on low halves of SRC1 and SRC0, neg_hi[1:0] on high halves,
{neg_hi[2], neg_lo[2]} on SRC2 as {abs, neg}; FP8/BF8: all zero; IU: neg_lo[1:0] = signed flags, clamp. The .td files in sources/ hold only feature
flags, not WMMA definitions, so the assembler is the reference. The float builtins have no modifier arguments: use inline asm, or fneg in IR and check
-save-temps for the folded modifier.
Test: random operands, modifier against the same operand with the sign bit flipped (or cleared for abs on C) in memory: bit-identical outputs, 10^6
elements. neg_lo[0] alone flips only even (low-half) slots of A. Special: negated NaN (sign flips, payload kept), -0, abs of -0, -inf, NaN in C.
Control: wrong half flag must differ. Falsifier: one mismatching element means the modifier is more than a sign change at the input.

## 5. Integer forms (short; droppable)
IU8 K = 16; IU4 K = 16 and 32. 4 signedness combos, clamp 0/1. Extreme inputs (127, -128, 255, 7, -8, 15 in all slots; C = INT_MAX, INT_MIN, 0), 10^6
random per form. Expected: int64 sum wrapped; clamp compared with both "clamp the final" and "clamp each step" (RDNA 3: wrap exact; clamp differed from
"clamp the final" on 2,241 outputs, 2,200 of them overflows). Control: wrong signedness flag disagrees.

## 6. Sparse forms
Guide 7.12.3: two 2-bit indices per group of 4 K values (idx0 < idx1); OPSEL selects among index sets per VGPR (wave32: OPSEL[0]). Builtin index
operand is a 16-bit short; the mapping to the guide's VGPR format is to check.
- E8 mapping: one packed nonzero, all 6 legal index pairs per group, B = distinct powers of two per K row; output shows which expanded K was used.
  Control: dense WMMA on the guide's expansion must match bit for bit. Illegal indices (idx0 >= idx1) recorded as measured.
- E9 arithmetic: the same 16 selected products through dense and sparse; then E3, E4, E7 on the 32-slot form with a thinned schedule (x in 6 slots,
  pairs within and across halves, C). Falsifier of "same adder": different delta for the same selected leaves.

## 7. Acceptance of the final model
100 % agreement, no tolerance, on (a) at least 10^7 random elements per form (FP16 exponents including subnormal, BF16 full range, FP8/BF8 all codes except
NaN in the main set, C from 2^-149 to 2^100, mixed signs, cancelling C, large-small mixes), (b) all adversarial sets of sections 1 and 2, (c) every
special-value case, (d) three launches identical. Below 100 % the model is wrong. Localise by shrinking a failing element: zero one leaf at a time while
it still fails; reduce operands to the nearest power of two, then fewer mantissa bits; move the two largest exponents together; stop at 2 or 3 leaves;
that case becomes a new E3 or E4 row. Control: a model with one drop rule changed (or W + 1) must fail the random set at a visible rate; if not, the set
is too tame.

## 8. Size of the differences (after the model is exact)
Compare hardware (CPU model) with the exact sum rounded once to FP32, for FP8xFP8, FP8xFP16, FP16xFP16, BF16. Data: weights from E4M3 codes drawn from a
Gaussian, activations FP16 from a Gaussian with about 0.1 % outliers at 20x; real tensors from the FP8 Llama 3.2 3B model if usable (to check). Chained
through C in steps of 16: K = 16, 64, 256, 1024, 4096. Also compare with an FP32 sequential FMA chain (a plain kernel). Report per K: share of elements
different, ulp distribution (0, 1, 2, 3-4, 5-16, >16) in ulps of the result and relative to sum |p| (cancellation makes ulps of the result misleading).
Worst case by hill climbing and random restarts over 16-term rows and 4096 chains, objective: error in ulps of the result and relative to sum |p|, seeded from
E3 and E4 cases. Control: the model must equal the hardware chain bit for bit first; an exact-accumulator chain has zero error by construction.

## 9. Errors after quantizing
Round hardware and exact results to FP16 (RNE) and E4M3 (RNE, saturating) and count changes. Report: rate per element with a binomial interval; maximum and
mean change in ulps of the target format when it changes; for K = 16..4096, with and without cancellation. Worst case: search with objective "rounds to a
different target value", seeded near ties of the target. Control: exact against itself changes nothing; exact against exact plus 1 FP32 ulp must give a rate
equal to the directly counted tie-neighbourhood share on the same data, proving the counter works.

## 10. Order
1. E0, E1 (cheap; decide whether hidden state exists; give the slot picture).
2. E2, E3a/b on FP16, then E3d and E7.
3. E4 cheap cut, then full schedule; E5 only if E4 contradicts a tree.
4. Sections 2 and 3 (MODE needs the E3d table).
5. FP8, BF8, mixed: repeat E3 and E4 with C as the large term; do not assume the structure is shared.
6. Section 4, then the CPU model, then section 7.
7. Section 6 after dense is closed. 8. Sections 8 and 9.
Drop if short: section 5, E5, the full E4 schedule (keep the cut), BF8 mixes beyond one cross form, real-tensor data.
Helper-suitable from a precise description: trial packer (slot tables exist in wmma_exact.hip), gap-sweep generators for E2/E3/E4, special-value runner,
modifier equality test, statistics scripts (8, 9), shrinker, integer and sparse-expansion CPU models. Lead keeps: reading E4 into a tree, model design,
MODE register writes, modifier decisions, acceptance.

## Lead notes (checked against the code, 2026-11-17)

- Section 0, slots: confirmed. `src/wmma_exact.hip` (header lines 4 to 9, `guide_k_of_contig` at line 122) treats the two published 16-bit
  K orders as one permutation applied to A and B alike, so the adder can be measured over slots. Report the slot tree, then say which K
  each slot holds under each published order. The K label matters again only for the transposing loads (lane 3).
- Priority of forms: FP8 x FP8 into FP32, FP16 and BF16 into FP32 first (the forms an FP8 kernel can use). Then BF8 and the mixed forms,
  the 16-bit accumulator forms, sparse, integers.
- Section 3: no probe in the repository writes MODE yet. Reading a hardware register through inline assembly works
  (`src/study/mem_common.h` line 34, `s_getreg_b32`); `s_setreg_b32` is to be shown working by the VALU control before any matrix result
  under a changed MODE is reported. Check the field layout against AMD's RDNA 4 instruction set guide rather than this draft.
- Section 1, E1 partial EXEC: only record what happens; the masked-lane question is lane 8.
- Section 8: "FP8 x FP16" is not a hardware form. It means FP8 weights widened exactly to FP16 and fed to the FP16 form, the route an
  exact kernel takes. Add the FP8 x FP8 form with activations rounded to FP8 as a separate, lossy row.
- Add to section 8: state for which inputs each form IS exact (for example: FP8 x FP8 products have 8-bit significands; if the measured
  internal width covers the exponent spread of 16 such products, say so as a rule a kernel author can rely on).
- Length of any single GPU job stays small: these are single-wave runs, exempt from the thermal rule, but still under the shared lock.
