# RDNA 4 matrix accelerator: timing (lane 4)

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201, 64 CUs = 128 SIMDs, wave32), ROCm 10.1.0 (hipcc, clang 24),
kernel 7.3.4. Run folder: `runs/20261117T131818Z-matrix/` (`scripts/study-matrix-timing.sh all`); the deliberate
breaks were rerun after a fix to the script in `runs/20261117T132259Z-matrix/` (see Method, "Runs"). The lead reran the whole script
(`runs/20261117T132721Z-matrix/`): every verdict the same, whole-card rates within 0.2 % (211.2, 210.9, 418.8 TFLOPS,
419.4 TOPS), highest hotspot 92 C. Every number is
MEASURED in that run unless it is labelled READ (from a document) or INFERRED (reasoning). Cycles are shader-clock
cycles from the wave's own counter, so they do not depend on the clock frequency. Unless an item says otherwise, one
wave runs alone on its SIMD. Card configuration for every job: LACT profile "Testing" (stock core clock and voltage,
317 W cap, memory 1258 MHz, performance level auto), checked and logged before each job; the regulator-level
power-limit mod (1.2x) is in the card, so the driver's power reading is scaled down on the main rail and is not true
board power.

Leftovers pass (branch `study/leftovers-matrix`): item 11a (whole-card rates of the 4-bit and sparse forms) is from
the full rerun of this lane's script `runs/20261117T165700Z-matrix/`; items 5d and 9 below gained a cause and a
control of wave order, from `runs/20261117T165116Z-leftmat/` (`scripts/study-leftovers-matrix.sh`, `src/study/left_rot.hip`, `left_prio.hip`; every mode PASS,
every deliberate break FAIL; same profile, cycle counts only).

## In short

- One matrix instruction occupies the SIMD for a fixed time that depends only on the input type: 16 cycles for
  FP16 and BF16 inputs (whatever the accumulator, dense or sparse), 8 cycles for FP8, BF8, 8-bit and 4-bit integer
  inputs (dense or sparse, K = 16, 32 or 64). All 22 instructions (11 dense, 11 sparse) take exactly the cycles AMD's calculator lists, and the 7 sign and
  clamp settings of the 8-bit instruction, which the calculator does not list, cost the same. A sparse
  instruction does twice the multiply-adds of its dense form in the same time (FP16: 512 against 256 per cycle;
  8-bit: 1024 against 512; 4-bit K = 64: 2048).
- A chain through the accumulator (C) costs exactly the issue time: 1, 2, 4 or 8 independent accumulators are no
  faster. The unit runs one instruction at a time and (INFERRED) forwards C to the next instruction of the same
  kind for free.
  The real result latency is longer: a result used as A or B gives 19 cycles per FP16 instruction and 11 per FP8
  instruction (with the required V_NOP),
  and the last result register reaches a vector instruction about 5 cycles after the first.
- What costs cycles: anything that breaks the C forward path costs 2 cycles per instruction (a different matrix
  form, even dense after sparse of the same type; a negate or absolute modifier on C); the first vector instruction
  after a matrix instruction costs 1 cycle extra; with a vector instruction between matrix instructions, a B operand
  that differs from the previous instruction's B costs 1 cycle more (A changes are free; item 5d); a vector
  instruction writing A, B or C right before costs 3; a
  result used as A or B (with the required V_NOP) costs 3 over the plain chain (FP16 into FP32: 19 against 16; FP16
  into FP16: 20). Clang's own code for a result used as A costs 26.5 in the FP16-into-FP16 form, where the
  hand-written V_NOP sequence costs 20: it copies the result with vector moves instead of using V_NOP. The two tests
  do not have the same dependence structure (see 5a), so read the 6.5 cycles as an indication, not an exact price.
- Nothing else in the same wave runs during a matrix instruction except scalar instructions and branches (8 scalar adds per FP16 instruction are free, 16 cost one cycle). Vector FMAs, V_NOPs and transcendentals add their full time.
  Loads issued before a matrix chain do progress while it runs.
- Data content (zeros, ones, random, denormals, NaN, infinity, huge cancelling products), the sparse index content,
  the signedness and clamp bits, an inline-constant accumulator and the negate modifiers on A and B change nothing.
- Up to 16 waves on one SIMD share its single matrix unit with no loss (16.0 cycles per FP16 instruction for the
  SIMD whatever the wave count). A vector-FMA wave on the same SIMD does not run during the matrix wave's
  instructions: they take turns, and when the vector wave is the older one both slow down (matrix 40 cycles per
  instruction in every run; the FMA wave's rate varied between runs, 1.9 to 2.7 cycles per FMA in the single-FMA pair). A memory-bound wave keeps its load rate beside a matrix
  wave. The two SIMDs of a CU and the four of a work-group processor have independent matrix units. `s_setprio`
  decides which of a matrix wave and an FMA wave goes first, whichever arrived first (one workload pair, one
  placement, priority levels 0 and 3); the kernel's round-robin bit
  (WG_RR_EN) removes the age order but leaves `s_setprio` in charge (item 9).
- Whole card (register-only operands, wall clock, profile "Testing" with stock clocks and the 317 W cap, power-limit
  mod present): FP16 and BF16 211 TFLOPS, FP8 and 8-bit integer 420 TFLOPS (dense); in the leftovers pass 4-bit
  K = 16 422 TOPS, 4-bit K = 32 841, sparse FP16 422, sparse FP8 842 and sparse 8-bit 839 (dense-equivalent; item 11a). That equals 128 SIMDs x the clock
  the cycle counter shows in the same waves (3.20 to 3.23 GHz) / 16 or 8 cycles (ratio 1.000 with one wave per
  SIMD), which means no SIMD loses time to anything else; it is not an independent check of the absolute rate (the
  independent part is that the counter's clock equals the driver's reported clock within 0.3 %). The operands were
  constant values; power and clock with random operands were not measured. More waves per SIMD give 1 to 2 % less. A naive kernel that loads both operands for every instruction reaches 19 to 38 TFLOPS
  from the caches and 5 (FP16) or 10 (FP8) TFLOPS from DRAM, where it is exactly bus-bound (634-638 GB/s).
  The core clock barely moves with the load: 3.16 to 3.24 GHz under register-only matrix, FMA and streaming loads
  (up to 3.41 GHz in the light memory-fed matrix runs).
- For a kernel author: chain through C freely (no need for several accumulators to hide latency); keep one form per
  accumulator chain; do not interleave vector work with matrix work in one wave expecting it to hide (it adds), but
  scalar address arithmetic and branches are free; feed operands from registers loaded well ahead (a load's
  results cost 12 to 18 cycles at the first vector or matrix instruction after the wait); never use a result as
  the next A or B if a C chain can do the same; 1 wave per SIMD already reaches the full matrix rate.

## 1. Known answers

| Check | Measured | Known | Label |
|---|---|---|---|
| First-hour probe `wmma_time`, FP16 chain (loop control included) | 16.11 | 16.11 (first hour) | MEASURED |
| `wmma_time`, BF16 | 16.11 | 16.11 | MEASURED |
| `wmma_time`, FP8 | 8.12 | 8.12 | MEASURED |
| `wmma_time`, 8-bit integer | 8.12 | 8.12 | MEASURED |
| `wmma_time`, one independent FMA | 1.078 | 1.08 | MEASURED |
| `mat_forms` calibration: 16 independent `v_fma_f32` (differential, loop control removed) | 1.000 | 1.00 (lane 1) | MEASURED |
| `mat_forms` FP16 chain (differential) | 16.000 | 16 (AMD's calculator, READ) | MEASURED |

The first-hour figures include that probe's loop control (about 0.1 cycle per instruction); the differential
method of this lane removes it, and every matrix figure below is an exact multiple of a cycle.

## 2. Dense forms

Cycles per instruction, one chain (every instruction reads the previous result as C) and 4 independent
accumulators. Spread: range from the extreme repetitions (7 timed repetitions in each of 3 launches per build, the
two ends of the U = 8 minus U = 4 difference taken at their extremes). Multiply-adds per cycle: M x N x K / cycles.

| Instruction | Form | Chain | 4 accumulators | AMD calculator | Multiply-adds per cycle |
|---|---|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` | FP16 -> FP32 | 16.00 (16.00-16.05) | 16.00 | 16 | 256 |
| `v_wmma_f32_16x16x16_bf16` | BF16 -> FP32 | 16.00 (16.00-16.08) | 16.00 (16.00-16.07) | 16 | 256 |
| `v_wmma_f16_16x16x16_f16` | FP16 -> FP16 | 16.00 | 16.00 (15.95-16.00) | 16 | 256 |
| `v_wmma_bf16_16x16x16_bf16` | BF16 -> BF16 | 16.00 (16.00-16.06) | 16.00 (16.00-16.08) | 16 | 256 |
| `v_wmma_f32_16x16x16_fp8_fp8` | FP8 x FP8 -> FP32 | 8.00 (8.00-8.14) | 8.00 | 8 | 512 |
| `v_wmma_f32_16x16x16_fp8_bf8` | FP8 x BF8 | 8.00 | 8.00 (7.93-8.00) | 8 | 512 |
| `v_wmma_f32_16x16x16_bf8_fp8` | BF8 x FP8 | 8.00 | 8.00 (7.93-8.00) | 8 | 512 |
| `v_wmma_f32_16x16x16_bf8_bf8` | BF8 x BF8 | 8.00 | 8.00 (7.92-8.00) | 8 | 512 |
| `v_wmma_i32_16x16x16_iu8` | 8-bit, unsigned x unsigned | 8.00 | 8.00 | 8 | 512 |
| `... neg_lo:[1,0,0]` | 8-bit, signed A | 8.00 | 8.00 | 8 | 512 |
| `... neg_lo:[0,1,0]` | 8-bit, signed B | 8.00 | 8.00 | 8 | 512 |
| `... neg_lo:[1,1,0]` | 8-bit, signed x signed | 8.00 | 8.00 | 8 | 512 |
| `... clamp` | 8-bit, unsigned, clamp | 8.00 (8.00-8.10) | 8.00 (8.00-8.14) | 8 | 512 |
| `... neg_lo:[1,0,0] clamp` | 8-bit, signed A, clamp | 8.00 (8.00-8.15) | 8.00 | 8 | 512 |
| `... neg_lo:[0,1,0] clamp` | 8-bit, signed B, clamp | 8.00 (8.00-8.12) | 8.00 (8.00-8.16) | 8 | 512 |
| `... neg_lo:[1,1,0] clamp` | 8-bit, signed, clamp | 8.00 (7.94-8.00) | 8.00 | 8 | 512 |
| `v_wmma_i32_16x16x16_iu4` | 4-bit, K = 16 | 8.00 (8.00-8.18) | 8.00 (8.00-8.10) | 8 | 512 |
| `v_wmma_i32_16x16x32_iu4` | 4-bit, K = 32 | 8.00 | 8.00 | 8 | 1024 |

The 11 dense instructions match AMD's calculator (`sources/matrix_calc.py`, rdna4 entries, READ: 22 entries, 11
dense and 11 sparse; the probe's expected values are typed into `mat_forms.hip` and equal the calculator's). The 7
further rows are sign and clamp settings of the 8-bit instruction, which the calculator does not list. The four signed and
unsigned settings and the clamp bit of the 8-bit form, and the K = 32 4-bit form (twice the work of K = 16 in the
same 8 cycles), cost the same.

## 3. Sparse forms (SWMMAC)

| Instruction | Form | Chain | 4 accumulators | AMD calculator | Multiply-adds per cycle |
|---|---|---|---|---|---|
| `v_swmmac_f32_16x16x32_f16` | FP16 -> FP32 | 16.00 (15.94-16.07) | 16.00 (16.00-16.05) | 16 | 512 |
| `v_swmmac_f32_16x16x32_bf16` | BF16 -> FP32 | 16.00 | 16.00 | 16 | 512 |
| `v_swmmac_f16_16x16x32_f16` | FP16 -> FP16 | 16.00 | 16.00 (15.95-16.06) | 16 | 512 |
| `v_swmmac_bf16_16x16x32_bf16` | BF16 -> BF16 | 16.00 | 16.00 | 16 | 512 |
| `v_swmmac_i32_16x16x32_iu8` | 8-bit, K = 32 | 8.00 | 8.00 (7.95-8.00) | 8 | 1024 |
| `v_swmmac_i32_16x16x32_iu4` | 4-bit, K = 32 | 8.00 (7.89-8.00) | 8.00 | 8 | 1024 |
| `v_swmmac_i32_16x16x64_iu4` | 4-bit, K = 64 | 8.00 (7.89-8.00) | 8.00 (7.95-8.00) | 8 | 2048 |
| `v_swmmac_f32_16x16x32_fp8_fp8` | FP8 x FP8 | 8.00 (7.89-8.00) | 8.00 | 8 | 1024 |
| `v_swmmac_f32_16x16x32_fp8_bf8` | FP8 x BF8 | 8.00 | 8.00 | 8 | 1024 |
| `v_swmmac_f32_16x16x32_bf8_fp8` | BF8 x FP8 | 8.00 | 8.00 | 8 | 1024 |
| `v_swmmac_f32_16x16x32_bf8_bf8` | BF8 x BF8 | 8.00 | 8.00 | 8 | 1024 |

Index content: FP16 sparse, 8-bit sparse (chain and x4), FP8 sparse, 4-bit K = 64 and the `index_key:1` forms
with the index register holding 0x44444444 (elements 0 and 1 of every four), 0xEEEEEEEE (2 and 3), 0xD8D8D8D8,
random valid pairs per lane, and the invalid patterns 0x00000000 (0 and 0), 0xFFFFFFFF (3 and 3) and random bits:
56 measurements, medians 8.000 to 16.000 cycles (16 for the FP16 forms, 8 for the others in every case). The
index content does not change the time.

## 4. Latency against throughput

Cycles per instruction:

| Form | 1 chain | 2 accumulators | 4 | 8 | A and B rotate (1 chain) | 8 accumulators, A and B rotate | chain through A (+ V_NOP) |
|---|---|---|---|---|---|---|---|
| FP16 -> FP32 | 16.00 (16.00-16.05) | 16.00 (15.95-16.05) | 16.00 (15.95-16.06) | 16.00 | 16.00 | 16.00 | 19.00 (19.00-19.14) |
| FP8 -> FP32 | 8.00 (8.00-8.13) | 8.00 | 8.00 (7.94-8.00) | 8.00 (7.92-8.00) | 8.00 | 8.00 (7.89-8.00) | 11.00 (11.00-11.22) |
| 8-bit signed | 8.00 (8.00-8.15) | 8.00 (8.00-8.13) | 8.00 (8.00-8.14) | 8.00 | 8.00 | 8.00 (7.93-8.00) | 11.00 (10.89-11.23) |
| FP16 -> FP16 | 16.00 | 16.00 | 16.00 (15.96-16.00) | 16.00 (16.00-16.08) | 16.00 (16.00-16.06) | 16.00 | 20.00 |

- Chained through C, an instruction issues as soon as the previous one has finished issuing; more accumulators do
  not help, so the matrix unit is not pipelined in the sense of starting a second instruction before the first one
  ends: it processes one instruction at a time, back to back, and C is forwarded inside the unit.
- The true result latency shows when the result feeds A (or B) of the next instruction: 19 cycles per FP16
  instruction including the required V_NOP, 11 for FP8 and 8-bit integer, 20 for the FP16-accumulator form.
  Without the V_NOP (forbidden, see item 5) the chain runs at 18 and 10. INFERRED: the result is complete about
  2 cycles (FP16 and FP8 with FP32 results) after the instruction's issue time, 3 to 4 for 16-bit results.
- A and B changing every instruction costs nothing (rot columns).

## 5. Hazards and stalls (cycles added)

(a) Result used as A or B of the next instruction (`mat_pipe hazard`, two accumulators alternating, each
instruction's A = the other's previous result; per matrix instruction, the V_NOP included):

| Sequence | FP16 | FP8 |
|---|---|---|
| plain chain through C (reference) | 16 | 8 |
| matrix + V_NOP, A unrelated (cost of the V_NOP after a matrix instruction) | 18 | 10 |
| matrix + V_NOP, A = previous result (what the documents require) | 19 | 11 |
| matrix + `v_mov`, A = previous result (any VALU satisfies the rule) | 19 | 11 |
| B = previous result + V_NOP (FP16) | 19 | |
| no V_NOP, A = previous result: FORBIDDEN by AMD's guide and LLVM; timed only, results not used | 18 | 10 |
| clang, builtin `wmma_f16(acc, b, acc)` (16-bit accumulator as A) | 26.5 | |
| the same form through C (builtin) | 16 | |

The forbidden sequence did not hang or fault; the hardware still waits 2 cycles, but what it reads was not checked
(lane 5's subject) and the documents say it is not safe. Clang does not use V_NOP here: it copies the 4 result
registers into the A registers with two `v_dual_mov_b32` and adds `s_delay_alu`, which costs 26.5 cycles per
instruction against 20 for the hand-written V_NOP form (FP16-accumulator form). The two are not like for like: the
builtin kernel is one chain in which A and C are both the previous result (and clang's copy of C by a vector move
costs 3 cycles on its own, 5d), while the hand-written test alternates two accumulators, so its C is two
instructions old. Data through C is the cheap path.

(b) Previous result used as C by a different matrix form (pairs of instructions on one accumulator, against the
same pair on two accumulators; cycles per pair):

| Pair | Dependent | Independent | Added per change of form |
|---|---|---|---|
| FP16 -> FP32 and FP8 -> FP32 | 28.00 | 24.00 | 2.00 |
| FP16 and BF16 (both FP32 results) | 36.00 | 32.00 | 2.00 |
| FP8 and BF8 | 20.00 | 16.00 | 2.00 |
| FP16 (float) and 8-bit integer | 28.00 | 24.00 | 2.00 |
| FP8 (float) and 8-bit integer | 20.00 | 16.00 | 2.00 |
| FP16 dense and FP16 sparse | 36.00 | 32.00 | 2.00 |

Every change of form on a C chain costs 2 cycles, whatever the types, including FP16 dense to FP16 sparse and FP8 to
BF8 (same result type). The C forward path works only between identical instructions. The negate and absolute
modifiers on C (item 6) cost the same 2 cycles in a chain and nothing with independent accumulators: INFERRED,
they also take C through the slow path.

(c) A vector instruction reading the result right after the matrix instruction (8 accumulators rotate, so matrix
instructions never wait for each other; per matrix + `v_mov` pair):

| `v_mov` reads | FP16 | FP8 |
|---|---|---|
| nothing (`v130 <- v131`) | 18 | 10 |
| the result one instruction older | 18 | 10 |
| the first register of the fresh result | 20 | 12 |
| the last register of the fresh result | 23 | 15 |
| result -> `v_add` -> next matrix instruction reads it as C (one chain) | 24 | 16 |

So any VALU after a matrix instruction costs 2 cycles (1 for itself, 1 switch); reading the fresh result costs 2
more for its first register and 5 more for its last. INFERRED: the result registers are written in order, the
first one about 2 cycles and the last about 5 cycles after the issue time; a VALU reading the result sees it 18
(first register) to 21 (last) cycles after an FP16 instruction issues, 10 to 13 after an FP8 one.

(d) A vector instruction writing A, B or C right before the matrix instruction (A/B sets rotate over 4,
accumulators over 8):

| `v_mov` writes | FP16 | FP8 |
|---|---|---|
| nothing, A and B fixed | 18 | |
| nothing, A and B rotating | 19 | 11 |
| A (first register) | 22 | 14 |
| B (first register) | 22 | |
| C (first register) | 22 | 14 |
| C (last register) | 19 | |

A VALU result feeding A, B or the first register of C costs 3 cycles (INFERRED: the VALU-to-matrix operand
latency is about 4 cycles where VALU-to-VALU is 5 with dependent issue); feeding the last register of C costs
nothing, so the unit reads C late and in order. With a VALU in between, rotating A and B costs 1 cycle (19 against
18), which it does not without the VALU (item 4).

What triggers that cycle (leftovers pass, MEASURED, `left_rot`, `runs/20261117T165116Z-leftmat/`): one wave, a period of 8 matrix
instructions on 8 rotating accumulators, an optional instruction before each; cycles per matrix instruction from the
difference of two unroll lengths (the loop's own cost cancels); 9 launches, medians, spread at most 0.002 except
one launch of one row (17.06).

Cycles per matrix instruction; in brackets the numbers of register sets rotated; "-" not run.

| Between the matrix instructions | A and B fixed | A rotates | B rotates | B changes every 2nd | A and B rotate (4) |
|---|---|---|---|---|---|
| nothing (FP16) | 16 | 16 (4) | 16 (4) | - | - |
| `s_nop 0` | 16 | 16 (4) | 16 (4) | - | - |
| `v_mov_b32` of a VGPR | 18 | 18 (2, 4, 8; also A changing every 2nd or 4th) | 19 (2, 4, 8) | 18.5 | 19 |
| `v_mov_b32` of a constant | 18 | 18 (4) | 19 (4) | - | - |
| `v_nop` | 18 | 18 (4) | 19 (4) | - | - |
| `v_fma_f32` (3 VGPR reads) | 18 | 18 (4) | - | - | - |
| `v_mov_b32` reading A's own first register | 18 | 18 (4) | - | - | - |
| nothing (FP8) | 8 | - | 8 (4) | - | - |
| `v_mov_b32` of a VGPR (FP8) | 10 | 10 (4) | 11 (4) | - | - |

So the cycle is real and its trigger is narrow: a matrix instruction whose B registers differ from the previous
matrix instruction's B, when any vector instruction (even `v_nop`, which reads no register) was issued between them;
B changing on every second instruction costs half a cycle per instruction, so it is paid per change. A changes,
the number of register sets, and scalar instructions between make no difference; the vector instruction's own
register reads (none for `v_nop` and a constant, one or three otherwise) changed nothing where tried. Lane 4's 19 (FP16) and 11 (FP8) rows rotated A and B together, so the cycle was B's. INFERRED: the matrix
unit keeps the previous B and can skip fetching it again; a new B is fetched in the issue slot and that fetch
overlaps the previous instruction only when nothing else was issued in between. For a kernel: when vector work must
sit between matrix instructions, keep B fixed over a run of instructions and rotate A (passing the operands the other
way round computes the transposed tile from the same products, so a loop can often choose which one stays fixed).

(e) A load delivering the operand (`global_load_b128` from the 64 KiB buffer, an L0 hit, then `s_wait_loadcnt 0`,
then the consumer; cycles per pattern): matrix instruction reading the loaded A 117.0, matrix instruction not reading
it 117.0; `v_mov` reading the loaded register 99.0, not reading it 98.0. The matrix instruction waits for the load
only through the wait instruction, like any VALU; it adds no wait of its own. The first VALU or matrix instruction
after the wait costs 12 to 15 cycles more than its issue time whether it reads the loaded data or not (see item 8;
the timing leftovers pass measured what this charge depends on, item 8a).

## 6. Inline-constant accumulator and modifiers

| Variant (FP16 unless named) | Cycles | Against |
|---|---|---|
| C = 0 (inline), one destination | 16.00 | 16.00 (F16_x4) |
| C = 0, 7 destinations rotating | 16.00 | 16.00 (F16_x4) |
| C = 1.0 (inline), rotating | 16.00 | 16.00 (F16_x4) |
| C from a register, one destination | 16.00 | 16.00 (F16_x4) |
| C from a register, rotating | 16.00 | 16.00 (F16_x4) |
| FP8, C = 0, rotating | 8.00 | 8.00 (FF_x4) |
| FP8, C from a register | 8.00 | 8.00 (FF_x4) |
| 8-bit, C = 0, rotating | 8.00 | 8.00 (IU8_S_x4) |
| 8-bit, C from a register | 8.00 | 8.00 (IU8_S_x4) |
| NEG on A, chain | 16.00 | 16.00 (F16_c1) |
| NEG_HI on A, chain | 16.00 | 16.00 (F16_c1) |
| NEG on B, chain | 16.00 | 16.00 (F16_c1) |
| NEG on C, chain | 18.00 | 16.00 (F16_c1) |
| ABS on C (NEG_HI[2]), chain | 18.00 | 16.00 (F16_c1) |
| all NEG and NEG_HI bits, chain | 18.00 | 16.00 (F16_c1) |
| NEG on A, 4 accumulators | 16.00 | 16.00 (F16_x4) |
| ABS on C, 4 accumulators | 16.00 | 16.00 (F16_x4) |
| all bits, 4 accumulators | 16.00 | 16.00 (F16_x4) |

C as the inline constant 0 or 1.0, or from a register no instruction writes, costs the same as an ordinary
independent accumulator. NEG on A, NEG_HI on A and NEG on B cost nothing. NEG or ABS on C cost 2 cycles per
instruction in a dependent chain and nothing with 4 accumulators (see 5b). The 8-bit sign bits and clamp are in
item 2 (no cost).

## 7. Data content

Cycles per instruction, chain and 8 accumulators, for FP16 -> FP32, BF16 -> FP32, FP16 -> FP16, FP8 -> FP32, BF8
-> FP32, signed 8-bit and signed 8-bit with clamp. Contents: zeros; ones; random finite values in [-1, 1] (random
finite codes for FP8/BF8, random bytes for integers); denormal inputs in the input format (integers: +127/-128 so
the sums overflow); tiny products (BF16 2^-70, products 2^-140, FP32 denormals; FP16 2^-10, products FP16
denormals in the FP16-accumulator form); NaN in A; infinity in A (E4M3 has no infinity: its largest value 448);
infinity times zero (NaN products); large cancellation (A alternates +big and -big in every row, B +big, C = 1).

| Form | Contents measured | Chain: lowest / highest median | 8 accumulators: lowest / highest |
|---|---|---|---|
| F16 | 9 | 16.000 / 16.000 | 16.000 / 16.000 |
| BF16 | 9 | 16.000 / 16.000 | 16.000 / 16.000 |
| F16H | 9 | 16.000 / 16.000 | 16.000 / 16.000 |
| FF | 9 | 8.000 / 8.000 | 8.000 / 8.000 |
| BB | 9 | 8.000 / 8.000 | 8.000 / 8.000 |
| IU8_S | 9 | 8.000 / 8.000 | 8.000 / 8.000 |
| IU8_SC | 9 | 8.000 / 8.000 | 8.000 / 8.000 |

The time per instruction does not move with the data in any form (largest change 0.00 % in every form).

## 8. What overlaps with a matrix instruction in one wave

Cycles per pattern (one FP16 or FP8 matrix instruction in a chain plus the extra work); "hidden" = (matrix alone +
other alone - together) / the smaller of the two (1: fully hidden, 0 or less: adds):

| Pattern | Together | Matrix alone | Other alone | Hidden |
|---|---|---|---|---|
| f16+4 scalar adds | 16.00 | 16.00 | 4.00 | 1.00 |
| f16+8 scalar adds | 16.00 | 16.00 | 8.00 | 1.00 |
| f16+16 scalar adds | 17.00 | 16.00 | 16.00 | 0.94 |
| f16+1 v_exp_f32 | 22.00 | 16.00 | 4.00 | -0.50 |
| f16+4 v_exp_f32 | 34.00 | 16.00 | 16.00 | -0.12 |
| f16+4 FMAs | 23.00 | 16.00 | 4.00 | -0.75 |
| f16+4 v_nop | 21.00 | 16.00 | 4.00 | -0.25 |
| f16+s_branch | 16.00 | 16.00 | 9.00 | 1.00 |
| fp8+4 scalar adds | 8.00 | 8.00 | 4.00 | 1.00 |
| fp8+1 v_exp_f32 | 14.00 | 8.00 | 4.00 | -0.50 |

- Scalar instructions run beside the matrix instruction: 8 scalar adds per FP16 instruction are free, 16 cost 1
  cycle; 4 per FP8 instruction are free. A branch (`s_branch`, 9 cycles alone) is free.
- Vector work does not overlap: FMAs, V_NOPs and `v_exp_f32` add their full time plus 1 to 3 cycles for the switch.
  (The first-hour overlap test said the same for FMAs: 0.00 overlap.) AMD's calculator lists no co-execution for
  RDNA 4 forms (`coexec: False`, READ).
- Loads: two 16-byte loads (L0 hits, 86 cycles alone with their wait) issued before a chain of 4, 8 or 16 FP16
  instructions (64, 128, 256 cycles) and waited for after it: chain mm4: together 90, chain alone 64, sum 150; chain mm8: together 154, chain alone 128, sum 214; chain mm16: together 282, chain alone 256, sum 342 cycles per block. The loads progress while the matrix
  chain runs: the total is the longer of the two plus 4 cycles when the loads take longer, and the chain plus 26
  cycles when the chain takes longer. With the wait before the chain: 168 (= 86 + 64 + 18). INFERRED: the arrival
  of load data costs the wave 12 to 26 cycles of issue (the same 12 to 18 cycles appear for the first vector or
  matrix instruction after any `s_wait_loadcnt`: 4 V_NOPs then the loads cost 104 against 86 + 4); the mechanism
  is not established.

### 8a. What the cost around a load wait depends on (timing leftovers, `tim_wait`)

One wave; each pattern ends with nothing outstanding and is timed with the differential method (2, 4 and 6
patterns per loop iteration). Loads: 4-byte global loads that hit L0 (77 cycles with the wait) or 4-byte LDS loads
(32 cycles). "Extra" = the pattern minus the same pattern without X, minus X's own issue time (1 cycle; 16 for a
matrix instruction). Run `runs/20261117T175731Z-timing/`, MEASURED; 1 of 2,610 samples dropped by the disturbance rule (an earlier run, `20261117T175038Z-timing`, agrees within 0.6 cycles).

| After the wait that blocked (the wave slept until the data came) | Extra cycles |
|---|---|
| `v_fma_f32` (not reading the loaded register) after 1, 4 or 16 global loads; after an LDS load | 15.3, 14.7, 14.6; 15.2 |
| 4 FMAs (the charge is paid once) | 15.2 |
| FP16 matrix instruction, after a global or an LDS load | 18.2, 18.2 |
| `s_add_co_i32` (scalar ALU) | 8.2 |
| `s_nop 0`; another global load; an LDS load | 0.3; 0.4; 0.1 |
| `s_nop 7` or `s_nop 15` first, then the FMA; `s_add` first, then the FMA | FMA still 14.9 / 15.2; both together 15.2 |

| Other cases | Extra cycles |
|---|---|
| data already back when the wait is reached (192 cycles of `s_nop` between load and wait), then FMA / `s_nop` / matrix / LDS case | 0.9 / -0.4 / 0.9 / 1.0 |
| a wait with nothing outstanding (`s_wait_loadcnt 0` or `s_wait_dscnt 0`), then FMA | 0.0 |
| a barrier in a one-wave group (nobody to wait for), then FMA / `s_nop` | 0.1 / -0.2 |
| `s_sleep 1`, then 64 FMAs | 0 (pattern 128 cycles = 64 + 64; s_sleep wakes on 64-cycle ticks, so a 15-cycle charge would have made it 192) |
| load data arriving during 8 or 16 chained FP16 matrix instructions, wait after the chain (no blocking) | +26.0 / +26.0 (b128 load 25.7; 4 loads 32.2; LDS 25.0) |
| load data arriving during 26 dependent FMAs | +19.0 |

- The charge appears only when the wave actually slept in `s_wait_loadcnt` or `s_wait_dscnt` (`tim_wait_charge`
  verdict). It does not depend on how many loads were outstanding (1, 4, 16), on LDS against global, or on whether the
  next instruction reads the loaded register. It is paid by the first vector-ALU or matrix instruction after the
  wake, wherever that comes (16 cycles of `s_nop` before it do not hide it), and once (4 FMAs pay it once). Scalar
  `s_nop` and memory instructions do not pay it; a scalar ALU instruction pays about half, and an FMA after it pays
  the rest. It is not a general cost of leaving and re-entering the SIMD: `s_sleep` and a barrier cost nothing.
- So the mechanism, INFERRED: a wave woken by returning load data must have its vector (and scalar ALU) issue
  re-enabled after the return is written into its registers, which takes about 15 cycles for the vector ALU and 18
  for the matrix unit; a wave that did not sleep finds the return already absorbed. The arrival of data while the
  vector or matrix unit is busy (the second table's last rows) costs the unit about the same: 19 cycles under FMAs,
  26 under matrix instructions, a little more for 4 separate returns. Which result would have contradicted it: a
  charge after `s_sleep` or after a wait that did not block, or a charge that hides behind 16 cycles of `s_nop`.
- For a kernel: waiting for loads costs about 15 cycles per wake on top of the latency; issue loads early enough that
  the wave does not block (data already back costs nothing), and expect a little loss when data returns under a
  matrix chain.

## 9. Several waves on one SIMD

`mat_share`, one 1024-thread group on one CU (CU mode), 1 to 16 waves active on one SIMD. "Own" = cycles per
instruction one wave sees between its own counter reads; "SIMD" = span of all active waves on that SIMD (its own
counter) divided by all their instructions; "overlap" = sum of the waves' REALTIME durations / REALTIME span (all
waves take their start stamp together, so n waves finishing one after another give about (n + 1) / 2 and n waves
all busy to the end give n).

| Waves | FP16 chain: own | SIMD | overlap | FP16 x4: SIMD | FP8 chain: SIMD |
|---|---|---|---|---|---|
| 1 | 16.000 | 15.999 | 1.00 | 16.000 | 8.000 |
| 2 | 16.000 | 16.001 | 1.51 | 16.001 | 8.000 |
| 4 | 16.179 | 16.016 | 2.53 | 16.016 | 8.009 |
| 8 | 16.468 | 16.983 | 4.45 | - | - |
| 16 | not reproducible (49.5; 2.3 in the lead's rerun) | 16.091 | 8.72 | 16.159 | 8.046 |

The SIMD has one matrix unit and any number of waves keep it fully busy: 16.0 cycles per FP16 instruction and 8.0
per FP8 instruction for the SIMD with 1 to 16 waves (within 1 %), except the 8-wave FP16 row of this run (median
16.98 with a wide spread, 16.04 to 18.29; 16.04 in a development run and 16.041 in the lead's rerun
`runs/20261117T132721Z-matrix/`, so the 16.98 is an outlier of that run: the timing leftovers pass sized it at about
246,000 extra cycles in a 4.2-million-cycle launch, 130 times the card's regular stall, so another process's waves on
that SIMD, not the clock ramp; `measurements/execution-core.md` item 9). One wave alone already reaches
that rate, so more waves only help to hide other latencies. Each wave alone sees 16.0 to 16.5 cycles up to 8
waves. With 16 waves the "own" figure (the slowest wave's difference of medians) is not usable: 49.5 in the main run
and 2.3 in the lead's rerun; whether the 16 waves interleave is not established (the overlap figure, 8.7 against 8.5
for one after another, does not show it). The SIMD figure (16.09) reproduces.

A method note for lane 1: a span that starts at repetition 1 includes, when waves run one after another, the
other waves' untimed repetition 0. That inflates the SIMD's cycles per operation by exactly (8n - 1) / (7n) for n
waves: 1.071, 1.107, 1.125, 1.134 for 2, 4, 8, 16 waves. Measured with such a span in a development run (not kept in the run folders), the matrix
SIMD showed 17.14, 17.73, 18.05, 18.25 cycles (16 x those factors). `measurements/execution-core.md` item 6 reported
the same factors (FMA 1.070, 1.107, 1.125, 1.134 cycles per op) as a 7 to 13 % loss. The lead fixed that probe the
same way and reran it: 0.999 to 1.000 cycles per FMA with 2 to 16 waves, so the loss was this artifact, and item 6
there is corrected. This lane's probe starts the span at repetition 0 and counts every repetition.

Two waves, one SIMD, different work (rates per wave; "rank 0" is the wave that took the first rank on the SIMD,
INFERRED to be the older one):

| Pair | Matrix wave | Other wave | Overlap |
|---|---|---|---|
| matrix alone / FMA alone / DRAM loads alone | 16.000 cycles per instruction | 1.000 cycles per FMA; 2.52 to 2.61 bytes per cycle (two runs) | |
| mm (rank 0) + fma | 16.000 cycles per instruction | 1.000 cycles per FMA | 1.46 |
| fma (rank 0) + mm | 40.188 cycles per instruction (40.218 in the rerun) | 1.883 cycles per FMA (2.686 in the rerun) | 2.00 |
| fp8 (rank 0) + fma | 8.000 cycles per instruction | 1.000 cycles per FMA | 1.47 |
| mm_x4 (rank 0) + fma | 15.994 cycles per instruction | 0.984 cycles per FMA | 1.46 |
| mm (rank 0) + fma x3 | 16.026 cycles per instruction | 0.984 cycles per FMA | 2.52 |
| mm x4 + fma x4 alternating | 40.691 cycles per instruction (40.916 in the rerun) | 2.023 cycles per FMA (1.035 in the rerun) | 4.82 |
| mm (rank 0) + mem | 15.998 cycles per instruction | 2.62 bytes/cycle | 1.41 |
| mem (rank 0) + mm | 16.234 cycles per instruction | 2.46 bytes/cycle | 1.55 |

- A matrix wave and a vector-FMA wave take turns. When the matrix wave is older it runs undisturbed and the FMA wave
  runs only around it (overlap 1.46: their timed runs barely overlap). When the FMA wave is older they interleave
  (overlap 2.0) and both slow down: the matrix wave takes 40.2 cycles per instruction in every run (16 alone), and
  the FMA wave 1.9 to 2.7 cycles per FMA (1.88 in the main run, 2.69 in the lead's rerun; 1.0 alone). Only the
  matrix side reproduces, so the split of those 40 cycles between FMAs and idle switching is not established. If
  FMAs could issue during a matrix instruction, both waves would keep close to their alone rates (16 and 1.0). So
  the FMA wave's work does not proceed during the matrix wave's instructions. What is not separated: whether the
  vector unit is blocked, or the SIMD's front end simply does not switch waves during a matrix instruction.
- Wave order (leftovers pass, MEASURED, `left_prio`, `runs/20261117T165116Z-leftmat/`): the same placement (one CU-mode group, two waves
  on one SIMD, roles by rank), each wave timed over its own region: the matrix wave 64 x 16 chained FP16
  instructions, the FMA wave 64 x 256 FMAs (16.0 and 1.08 cycles per operation alone; the 0.08 is the loop). 30
  pairs per configuration (15 launches x 2 SIMDs); "ends first" in 30 of 30 or 0 of 30 in every row.

  | Code object, s_setprio | Matrix wave older: matrix / FMA cycles per op | FMA wave older: matrix / FMA | Which ends first |
  |---|---|---|---|
  | normal, none | 16.36 / 2.09 | 31.69 / 1.49 | the older wave's work |
  | normal, matrix wave 3 (or matrix 3, FMA 0) | 16.36 / 2.09 | 16.36 / 2.09 | matrix |
  | normal, FMA wave 3 (or matrix 0, FMA 3) | 32.21 / 1.51 | 31.69 / 1.49 | FMA |
  | round-robin bit, none | 19.27 / 2.21 | 19.29 / 2.22 | matrix |
  | round-robin bit, matrix wave 3 | 16.33 / 2.09 | 16.33 / 2.09 | matrix |
  | round-robin bit, FMA wave 3 | 31.94 / 1.50 | 31.93 / 1.49 | FMA |

  So the "older wave first" reading of the pairs above holds, and both controls work: `s_setprio` makes the raised
  wave behave as the older one in either order, and the round-robin bit makes the order of creation irrelevant
  (the same numbers with either wave older) while `s_setprio` still decides. With the bit and equal priorities the
  matrix wave still mostly goes first (19.3 cycles per instruction against 16.0 alone). The FMA wave beside a
  favoured matrix wave makes almost no progress during the matrix wave's run: its 2.09 cycles per FMA equals the
  matrix wave's region plus its own alone time (INFERRED from 16.4 x 1,024 + 1.08 x 16,384 cycles over 16,384
  FMAs = 2.10). Neither control gives both waves their alone rates. The bit was set with the assembler directive
  `.amdhsa_round_robin_scheduling 1` (clang 24 accepts no source attribute or option for it; READ from the LLVM
  strings) in a copy of the kernels' listing, assembled into a code object loaded with `hipModuleLoad`; the
  descriptors read back with the bit set in 5 of 5 kernels of that object and clear in the other (bit 21 of
  COMPUTE_PGM_RSRC1). Rates differ from lane 4's table (40 against 32 cycles) because the two waves' run lengths
  differ; the rule is the same.
- A memory-bound wave (four 16-byte loads per lane from DRAM, then a wait) keeps its rate beside a matrix wave
  (2.46 to 2.62 bytes per cycle against 2.52 to 2.61 alone) and the matrix wave keeps 16.0 to 16.2 cycles: loads issue
  rarely and proceed in the background.

### 9a. Do waves interleave matrix chains, and what does a hand-over cost? (timing leftovers, `tim_mat`)

One work-group on one CU (CU mode), W waves on each SIMD; every wave runs its own chain of 256 matrix instructions
(16 chunks of 16; chained through C, or four accumulators in turn as in `mat_card`), all released together by a
barrier. The SIMD's cycles per instruction come from the first start and the last end of its waves (cycle counter
reads before and after the chain only). Run `runs/20261117T175731Z-timing/`, 5 launches x 3 repetitions per row, MEASURED.

| W | FP16 chain | FP16, 4 accumulators | FP8 chain | FP8, 4 accumulators | Order the waves finish in |
|---|---|---|---|---|---|
| 1 | 16.086 | 16.086 | 8.082 | 8.086 | - |
| 2 | 16.248 | 16.250 | 8.066 | 8.072 | oldest first, one after another |
| 4 | 16.333 | 16.336 | 8.057 | 8.065 | the same |
| 8 | 16.384 | 16.390 | 8.055 | 8.062 | the same |
| 8, youngest wave at `s_setprio 3` | 16.385 | 16.391 | 8.055 | 8.062 | the raised wave first, then oldest first |

- Waves run their matrix chains one after another, not interleaved (INFERRED): the cost per hand-over is constant,
  105, 106 and 109 cycles at 2, 4 and 8 waves, which fits a serial model, and the waves finish oldest first. The
  probe does not show it directly: its switch counts exist only in the rows with counter reads (the zeros in the
  other rows are placeholders), and the finish order has no end times, so two interleaving waves would also finish
  oldest first. Also with 8 cycles of `s_nop` or a taken branch after every 16 instructions (both run
  beside the matrix unit) the rows are the same within 0.02 and no wave switched. With a cycle-counter read after
  every 16 instructions (it costs the wave about 20 cycles) the waves do alternate, 15 to 80 times per repetition,
  and the read's cost is then partly filled by the other waves: so a counter read inside a chain changes what it
  measures.
- A hand-over from one wave's finished chain to the next wave's costs about 106 cycles for FP16 (105, 106, 109 per
  hand-over at W = 2, 4, 8, from the spans) and about 13 for FP8. Why FP16 costs eight times more is not explained
  (both forms forward C and finish 2 cycles after issue, item 4).
- `s_setprio 3` on the youngest wave makes it run first; the rate does not change.
- What this means for the whole-card loss of item 11 (more than one wave per SIMD loses 1.2 to 2.3 % for FP16, none
  for FP8): `mat_card`'s waves run their whole loop one after another the same way, so a launch of about 0.17 s
  holds W - 1 hand-overs per SIMD, about 0.2 us in total, far below 1 % (INFERRED from the per-hand-over cost).
  Excluded as the cause: interleaving at chunk boundaries (none happens), a broken C forward at every switch
  (no switches), wave order (`s_setprio` changes the order, not the rate). Not established: what at card level
  (launch and end of launch with more waves, or something outside one SIMD) costs FP16 the 1 to 2 %.

## 10. SIMDs of one CU and of different CUs

`mat_share` in WGP mode (the group spreads over the 4 SIMDs of a work-group processor; SIMD_ID bit 0 = CU, bit 1 =
SIMD within the CU). SIMD cycles per instruction:

| Work, waves per SIMD | One SIMD | Two SIMDs of one CU | Two SIMDs of two CUs | All four |
|---|---|---|---|---|
| mm_f16_chain x1 | 16.000 | 15.999 | 15.999 | 16.000 |
| mm_f16_chain x4 | 16.015 | 16.016 | 16.016 | 16.016 |
| mm_f16_x4 x1 | 16.000 | 16.001 | 15.998 | 16.000 |
| mm_f16_x4 x4 | 16.015 | 16.016 | 16.016 | 16.016 |
| mm_fp8_chain x1 | 8.000 | 8.001 | 7.999 | 8.001 |
| mm_fp8_chain x4 | 8.008 | 8.009 | 8.009 | 8.009 |

Every SIMD keeps its one-SIMD rate when the other SIMD of its CU, or all four SIMDs of the work-group processor,
run matrix work at the same time. No matrix resource is shared within a CU or a WGP.

## 11. The whole card

`mat_card` (written by a helper, reviewed and changed by the lead; see Helpers). 64 work-groups of 64 x W threads,
each with 40 KiB of unused LDS so only one fits on a CU: W waves on each of the 128 SIMDs, checked for every launch
from each wave's placement record (128 distinct SIMDs, exactly W waves each, for every W). Each wave runs a loop of
16 matrix instructions over 4 independent accumulators, A and B fixed in registers (no memory traffic). Timed by
HIP events over 3 launches of about 0.17 s after a warm-up launch; the clock is the median over waves of cycle
counter ticks / REALTIME seconds. Profile "Testing" (stock clocks, 317 W cap), power-limit mod present: reported
power is the driver's reading, scaled down on the main rail.

TFLOPS = matrix instructions per second x 16 x 16 x 16 multiply-adds x 2 operations (8192 per dense K = 16
instruction; TOPS for 8-bit integer). Theoretical = 128 SIMDs x clock / cycles per instruction (16 or 8, item 2).

| Form | W | G instructions/s | TFLOPS | Clock (counter) | Theoretical G instr/s | Ratio | sclk median | Reported power | Hotspot |
|---|---|---|---|---|---|---|---|---|---|
| FP16 -> FP32 | 1 | 25.8 | 211.4 | 3227 MHz | 25.8 | 1.000 | 3221 | 245-271 W | 76-88 C |
| FP16 -> FP32 | 2 | 25.5 | 208.8 | 3224 | 25.8 | 0.988 | 3219 | 241-280 | 76-89 |
| FP16 -> FP32 | 4 | 25.3 | 207.6 | 3222 | 25.8 | 0.983 | 3219 | 248-280 | 75-88 |
| FP16 -> FP32 | 8 | 25.3 | 206.9 | 3221 | 25.8 | 0.980 | 3218 | 263-279 | 77-87 |
| FP16 -> FP32 | 16 | 25.2 | 206.2 | 3220 | 25.8 | 0.977 | 3219 | 244-282 | 74-87 |
| BF16 -> FP32 | 1 | 25.8 | 211.1 | 3222 | 25.8 | 1.000 | 3221 | 246-261 | 76-83 |
| BF16 -> FP32 | 16 | 25.2 | 206.2 | 3222 | 25.8 | 0.977 | 3221 | 262-280 | 79-85 |
| FP8 -> FP32 | 1 | 51.2 | 419.5 | 3201 | 51.2 | 1.000 | 3199 | 303-337 | 77-91 |
| FP8 -> FP32 | 2 | 50.7 | 415.1 | 3168 | 50.7 | 1.000 | 3167 | 315-327 | 80-91 |
| FP8 -> FP32 | 16 | 50.6 | 414.8 | 3163 | 50.6 | 1.000 | 3164 | 313-331 | 81-92 |
| 8-bit integer | 1 | 51.2 | 419.8 TOPS | 3204 | 51.3 | 1.000 | 3199 | 306-326 | 76-91 |
| 8-bit integer | 16 | 50.7 | 415.5 TOPS | 3171 | 50.7 | 1.000 | 3171 | 308-330 | 80-91 |

(BF16 W = 2, 4, 8: 208.5, 207.7, 206.9 TFLOPS, ratios 0.987, 0.983, 0.980; FP8 W = 4, 8: 414.3, 415.2; 8-bit
W = 2, 4, 8: 416.4, 416.0, 415.8. All in the run folder's `mat_card-reg_*.log`.)

- One wave per SIMD gives the highest whole-card rate for every form, at ratio 1.000 to the rate computed from the
  clock the same waves' cycle counters show: the card is 128 independent matrix units each issuing back to back,
  with no time lost at card level. The ratio is not an independent check of the absolute rate; the clock itself is
  cross-checked against `pp_dpm_sclk` (item 12). A and B were constant values in these runs. More waves per SIMD lose 1.2 to 2.3 % of wall
  clock for FP16 and BF16 (not for FP8 and 8-bit, whose clock drops instead, see item 12); the cause is not
  established. The leftovers pass makes wave switching on one SIMD an unlikely cause (item 9a: the chains look serial, a
  hand-over costs about 106 cycles, far too rare to matter by the arithmetic of a 256-instruction chain, and
  `s_setprio` changes nothing); launch and
  end-of-launch effects of more waves remain the candidate (INFERRED, not tested).
- FP8 and 8-bit integer draw more reported power (297 to 337 W, around the 317 W cap figure; the reading is
  scaled down by the power-limit mod) and their clock is 1 to 2 % lower than under FP16 (3.16 to 3.20 against
  3.22 GHz); FP16 reports 241 to 284 W. INFERRED, not shown: the lower clock is the power limit acting.

- Theoretical rate at another clock (INFERRED from the per-SIMD cycle counts): dense FP16/BF16 512 operations per
  SIMD per clock (1024 per CU), dense FP8/BF8/8-bit 2048 per CU, dense 4-bit K = 32 and sparse 8-bit/FP8 4096 per CU,
  sparse 4-bit K = 64 8192 per CU. At the 2.97 GHz boost clock in AMD's specification that is 195 TFLOPS FP16,
  389 FP8, 779 sparse FP8 and 1557 sparse INT4: this settles the conflicting labels noted in
  `notes/research-8-architecture-and-family.md` (389 is dense FP8 or sparse FP16, not dense FP16). This card runs
  at 3.2 GHz under matrix load, so it exceeds those figures by about 8 %.

### 11a. The 4-bit and sparse forms on the whole card (leftovers pass, MEASURED, `runs/20261117T165700Z-matrix/`)

Five register-only modes added to `mat_card` (written by a Sonnet 5.5 helper from a sealed brief; the lead read the
diff, and the numbers below are from the lead's full rerun of `scripts/study-matrix-timing.sh`, every verdict as in
`runs/20261117T132721Z-matrix/`, the four existing whole-card modes within 0.6 % of that run: FP16 211.7, BF16 211.0,
FP8 421.1, 8-bit 420.2 at W = 1). Same kernel shape: 16 instructions of one form over 4 accumulators per loop
iteration, A, B and the index fixed in registers (the listing check finds exactly 16 of the form per loop, no other
matrix instruction, no scratch). Thermal rule as above (bursts of about 0.17 s, each started below 60 C; highest
hotspot 89 C). Operations per instruction count the dense-equivalent work (2 x 16 x 16 x K); a sparse instruction
stores half of it.

| Form | Cycles (item 2) | W = 1: G instr/s | TOPS / TFLOPS | Theoretical (128 x clock / cycles) | Ratio | Counter clock | W = 16 | Reported power | Hotspot |
|---|---|---|---|---|---|---|---|---|---|
| 4-bit integer, K = 16 | 8 | 51.5 | 421.8 | 51.5 | 0.999 | 3221 MHz | 417.5 (1.000) | 313-321 W | 78-88 C |
| 4-bit integer, K = 32 | 8 | 51.3 | 840.8 | 51.4 | 0.999 | 3210 | 831.6 (0.999) | 304-323 | 73-88 |
| sparse FP16 -> FP32, K = 32 | 16 | 25.8 | 422.1 (stored 211.1) | 25.8 | 1.000 | 3221 | 415.8 (0.985) | 250-276 | 67-80 |
| sparse FP8 -> FP32, K = 32 | 8 | 51.4 | 841.6 (stored 420.8) | 51.4 | 1.000 | 3211 | 815.1 (0.985) | 309-333 | 79-88 |
| sparse 8-bit integer, K = 32 | 8 | 51.2 | 838.8 (stored 419.4) | 51.3 | 0.998 | 3205 | 817.9 (0.986) | 305-325 | 73-88 |

So lane 4's inference from the single-SIMD cycle counts holds: every form reaches 128 SIMDs x clock / cycles within
0.2 % with one wave per SIMD. The helper's own runs (`runs/helper-leftcard/`, not from the script) agree within
1 %. Deliberate breaks of the five new modes (half the cycles claimed) were run only in the helper's runs, not through
the script: ratio 0.500, FAIL for all five. The script's own break covers the FP16 mode. The sparse
FP16 form reports less power than the 8-bit forms, like dense FP16. A and B were constant values here too.
- Against AMD's library (first hour, earlier overclock profile, READ from `notes/first-hour-9070xt-rocm10.md`):
  hipBLASLt FP8 GEMM reached 170 TFLOPS, 41 % of the 420 TFLOPS register-only FP8 ceiling; PyTorch FP16 matmul 123
  TFLOPS, 58 % of 211.

Operands loaded from memory every iteration (the simplest kernel: each iteration loads its A and B fragment, 16
bytes per lane each for FP16, 8 for FP8, with ordinary vector loads, then one matrix instruction into one
accumulator; compiler-built loop unrolled 4 times):

| Form | Source | W | G instr/s | TFLOPS | Share of register-only rate | Bytes read per s |
|---|---|---|---|---|---|---|
| FP16 | 64 KiB per wave in a 32 MiB buffer (cache) | 1 | 3.7 | 30.1 | 14 % | 3765 GB/s |
| FP16 | cache | 4 | 2.3 | 18.9 | 9 % | 2366 GB/s |
| FP16 | cache (16 KiB per wave) | 16 | 2.4 | 19.3 | 9 % | 2412 GB/s |
| FP16 | 1 GiB, no reuse (DRAM) | 1 / 4 / 16 | 0.6 | 5.1 / 5.1 / 5.0 | 2.4 % | 638 / 638 / 630 GB/s |
| FP8 | cache | 1 | 3.8 | 31.1 | 7 % | 1942 GB/s |
| FP8 | cache | 4 / 16 | 4.6 | 37.7 | 9 % | 2354 / 2355 GB/s |
| FP8 | DRAM | 1 / 4 / 16 | 1.2 | 10.1 / 10.2 / 10.1 | 2.4 % | 634 / 638 / 629 GB/s |

Why (INFERRED arithmetic on the MEASURED rows): such a kernel reads 1024 bytes (FP16) or 512 bytes (FP8) per
8192-flop instruction, 8 or 16 flops per byte. From DRAM it runs exactly at the bus (634 to 638 GB/s, as the
`stream` mode's 636 GB/s and lane 2's 633 GB/s), so 638 GB/s x 8 = 5.1 TFLOPS and x 16 = 10.2 TFLOPS. From the
caches (the 32 MiB buffer sits in the 64 MiB memory-side cache; at W = 1 the 8 MiB touched may also fit L2) it is
bound by 1.9 to 3.8 TB/s of cache bandwidth. To reach the register-only ceiling an FP16 kernel needs about 330 flops
per DRAM byte (211 TFLOPS / 638 GB/s) and about 70 per cache byte: operands must be reused many times from
registers or LDS (tiling); one load per instruction can never feed the unit.

## 12. Core clock under load

Read-only: `pp_dpm_sclk` sampled every 50 ms by the program during the timed launches (min / median / max), and
the cycle counter over REALTIME per wave (median). Same run, profile "Testing", power-limit mod present.

| Whole-card load | sclk (min / median / max) | Counter clock | Reported power | Hotspot |
|---|---|---|---|---|
| FP16 matrix, W = 1..16 | 3119-3185 / 3218-3221 / 3243-3250 MHz | 3220-3227 MHz | 241-284 W | 74-89 C |
| BF16 matrix | 3217-3218 / 3221-3222 / 3243-3250 | 3222-3225 | 246-280 | 76-85 |
| FP8 matrix | 3119-3178 / 3164-3199 / 3173-3222 | 3163-3201 | 303-337 | 77-92 |
| 8-bit integer matrix | 3126-3179 / 3171-3199 / 3180-3221 | 3171-3204 | 297-333 | 76-91 |
| FP32 FMA, dual issue, W = 8 (24.7 TFMA/s = 0.945 of 2 FMAs per lane per cycle) | 3169 / 3191 / 3221 | 3195 | 269-346 | 76-92 |
| memory-bound read, 1 GiB, 636 GB/s | 3217 / 3230 / 3258 | 3236 | 143-149 | 53-55 |
| matrix fed from DRAM, W = 1 / 4 / 16 | median 3233 / 3301 / 3411 | 3239 / 3304 / 3412 | 140-175 | 54-59 |

- The core clock under whole-card matrix load is 3.16 to 3.23 GHz; under the FMA load 3.19 and under a
  memory-bound load 3.23 GHz. The matrix load does not lower the clock against the other heavy loads by more than
  about 2 %, and only for the forms that reach the power cap (FP8, 8-bit). The memory-fed matrix kernel with 16 waves
  per SIMD ran at 3.41 GHz, higher than the streaming read (3.24) at similar power; not explained.
- Cycle counter against wall clock: the counter's clock equals the `pp_dpm_sclk` median within 0.3 % in every row,
  so the counter runs at the shader clock under load (lane 1 found the same with one wave).
- Between jobs the clock fell to the idle level (sclk "S: 75-102 MHz" before each job). The 1-second clock log
  (`clock-log.csv`) is too coarse for the sub-second single-wave jobs; their clock was not recorded.

## Method

Probes (all `src/study/`, built and run by `scripts/study-matrix-timing.sh`):

| Probe | Items | What it runs |
|---|---|---|
| `src/wmma_time.hip` (first hour, unchanged) | 1 | builtin chains, the known answers |
| `mat_forms.hip` | 1, 2, 3, 6, 7 | modes `forms`, `index`, `mods`, `data` |
| `mat_pipe.hip` | 4, 5, 8 | modes `lat`, `hazard`, `overlap` |
| `mat_share.hip` (CU and WGP builds) | 9, 10 | roles per wave in one work-group |
| `mat_card.hip` | 11, 12 | whole card, wall clock (written by a helper, see below) |
| `mat_common.h` | | register map, kernel macro, differential cost |

- Single-wave probes (`mat_common.h`): one wave of 32 lanes; a loop of 128 iterations, each with U blocks of
  inline assembly; built with U = 4, 6 and 8; cost of a block = (cycles at U = 8 - cycles at U = 4) / (4 x 128).
  U = 6 must land midway (within 1.5 cycles per iteration plus 0.5 % of the blocks), or the probe reports the loop
  as non-linear and its verdict fails. 8 repetitions per launch (the first dropped), 3 launches: 21 samples per
  build; tables give the median and the range from the extreme samples.
- Every operand is bound to a fixed register range with an explicit-register constraint (`"+{v[16:23]}"`), so
  the assembly names every register (a VALU reading one register of a result is possible) and the compiler only
  moves data around the loop. A, B and C never overlap a matrix destination except in the hazard tests that say so.
- `mat_share` writes its whole timed loop (counter read, loop control, blocks, counter read) in one assembly
  statement: with compiler-built loops at the 96-register limit of 16 waves per SIMD, clang copied and spilled
  registers inside them.
- Assembly checks (in the build phase of the script; output `corecheck-*.txt` and `loopcheck-*.txt` in the run
  folder): `scripts/core_loopcheck.py` (every loop holds only the assembly blocks and the loop control
  `s_add_co_i32`/`s_cmp_lg_u32`/`s_cbranch_scc1`; nothing the compiler adds grows with U) for `mat_forms` and
  `mat_pipe`; `scripts/mem_loopcheck.py` with `scripts/mat-expect/<probe>.txt` for the exact count of each
  matrix instruction, with its modifiers, in every timed loop (and V_NOPs, moves, loads, waits where they matter;
  no scratch anywhere). The compiler added no V_NOP, move or wait to any inline-assembly loop. In the builtin
  kernels (`wmma_time`, `bi_*` in `mat_pipe`) the loops hold exactly the intended matrix instructions; the
  compiler's chain through A carries two `v_dual_mov_b32` and one or two `s_delay_alu` per instruction.
- Loops over about 4 KiB that issue an 8-byte instruction every cycle lose 12 to 14 cycles per iteration at U = 8
  (measured with 64 FMAs per block; INFERRED instruction fetch); the FMA reference blocks were made smaller.
- Verdicts: each probe prints `VERDICT <name>: PASS|FAIL` and exits 1 on FAIL. They are tied to known answers
  (the calibration, the first-hour cycle counts, lane 1's FMA, exp, scalar-add and branch costs), to the
  one-matrix-unit limit, and to stated expectations (independent sets at the issue time; data, index, modifier
  and placement checks). The C modifiers in a chain are reported and not judged (no document gives their cost).

Deliberate breaks (run by the script in the `breaks` phase; each must FAIL):

| Probe and break | What is claimed | Output (`runs/20261117T132259Z-matrix/`) |
|---|---|---|
| `wmma_time --expect f16=8` | FP16 takes 8 cycles | `FAIL ... f16 chain 16.1 cycles (x4 16.0), expected 8 MISMATCH` |
| `mat_forms forms --break` | FP16 takes 8 cycles | `FAIL ... F16 chain 16.000 cycles, expected 8 (first hour) FAILED` |
| `mat_pipe lat --break` | the 8-accumulator set is independent, but it is the chain through A | `FAIL ... F16_c8 19.000 (issue 16) FAILED; ... FF_c8 11.000 (issue 8) ...` |
| `mat_share_cu --break` | two matrix waves on one SIMD, really one on each SIMD of the CU | `FAIL ... two matrix waves claimed on one SIMD: 8.000 cycles per instruction for the SIMD, one matrix unit allows no less than 15.52 FAILED` |
| `mat_card reg:f16 --break` | FP16 takes 8 cycles (theoretical rate doubled) | `FAIL ... best W=1 ratio 0.500 outside [0.80, 1.02]` |

Each was restored (the normal runs of the main folder PASS with the same builds). The assembly checker was broken
once by hand (an expected count of 63 instead of 64 matrix instructions): it reported `FAIL ... (got 64)` and exit
status 1.

Runs: the main run (`runs/20261117T131818Z-matrix/`) was made from commit 5a33425 with `mat_card.hip` and its
expectation file not yet committed (they were committed unchanged afterwards in 5b57906). In that run
every job and four breaks behaved as intended; the `mat_card` break printed FAIL and exit status 1, but the script's
check missed the colon in its verdict name and reported "did NOT fail". After that fix (commit 93dc5d3) the breaks
phase was rerun alone (`runs/20261117T132259Z-matrix/`) and all five reported FAIL as intended.

## How to rerun

```
scripts/study-matrix-timing.sh            # build, run, breaks; results in runs/<UTC>-matrix/
scripts/study-matrix-timing.sh build      # build and assembly checks only
MAT_DEBUG=1 build/study/bin/mat_pipe lat  # prints cycles per iteration of every U build (under the GPU lock)
```

The script needs `runs/latest/card.env` (from `scripts/first-hour.sh info`), takes the shared GPU lock for every
job, refuses to run timed work unless the LACT profile is "Testing", and stops on a configuration change or a
hotspot of 95 C.

## Not measured

Still open:

- The cost around a load wait (item 8a) is measured in detail; the mechanism (re-enabling issue after a wake) is
  INFERRED. Not reachable (leftovers C plan, MT-5a): it is internal issue logic and item 8a measured everything
  visible.
- Why a hand-over between waves costs 106 cycles for FP16 and 13 for FP8 (item 9a), and what costs the whole card 1 to
  2 % with more waves for FP16 (item 11), are not established.
  Answered in part (leftovers C, MT-5b, `measurements/leftovers-c-b.md`): the hand-over cost goes with the
  instruction's issue timing, not its type: every dense form with a 16-cycle instruction costs 105 to 110 cycles per
  hand-over (FP16 into FP32, BF16 into FP32, FP16 into FP16), every 8-cycle form (FP8, BF8, 8-bit integer) 12 to 13,
  and sparse FP16, also 16 cycles, 76 to 79. MEASURED. Open: why a 16-cycle instruction costs eight times more than
  an 8-cycle one; chain lengths other than 256 were not run.
  Answered in part (leftovers C, MT-5c, `measurements/leftovers-c-b.md`): the FP16 loss with 16 waves per SIMD is absent
  in 50 ms launches (-0.2 %) and is 1.0 % at 100 ms and 1.7 % at 200 ms; it is associated with longer launches and a
  higher clock (waves' counter clock 3015, 3114, 3166 MHz), and these bursts do not separate the two. At one wave per
  SIMD the clock rises the same way with no loss. The clock estimate of 16 waves carries about 1.5 % uncertainty, as
  large as the loss, so it is a measured difference between two ratios, not a firm wall-clock figure; 3 bursts per
  cell. MEASURED; the cause is open.
- Why a new B costs a cycle only with a vector instruction between (item 5d): the trigger is measured (leftovers
  pass), the mechanism is INFERRED. Low value (leftovers C plan, MT-6).
  Answered in part (leftovers C, MT-6, `measurements/leftovers-c-b.md`): the extra cycle belongs to the matrix
  instruction that directly follows a vector instruction and has a B different from its predecessor's: with the
  `v_mov` before every second instruction the cost is 17.5 cycles per block, so a B change right after a matrix
  instruction stays free; the vector instruction's encoding (VOP3 or not) does not matter; a B that overlaps the
  previous one by half costs half a cycle on average (18.5, not explained). MEASURED; the mechanism is open.
- Wave order beyond three waves of one SIMD, and memory-bound partners (item 9 already shows a load wave neither slows
  nor is slowed): not tried.
  Answered in part (leftovers C, MT-7, `measurements/leftovers-c-b.md`): `s_setprio` levels 0 to 3 form a strict order:
  in all 24 unequal pairs and both arrival orders the higher level runs first, in the normal code object and with the
  round-robin bit; at equal levels the older wave goes first in the normal object and the matrix wave goes first with
  the round-robin bit. With three waves (a matrix wave and two FMA waves) the normal object keeps strict age order
  when nobody is raised. MEASURED (30 of 30 per row); the mechanism is not established.
- Hotspot: whole-card FP8, 8-bit and dual-FMA loads lifted the hotspot from about 55 C to 91-92 C within about one
  second of load (each configuration then cooled to below 60 C in 0.3 s before the next). The highest reading of the
  study was 92 C, below the 95 C stop, but above the 62-82 C the owner noted for full load on this profile. Longer
  whole-card runs were not made for that reason. Needs the owner (leftovers C plan, MT-9): sustained load. Leftovers C
  ran only short bursts (about 0.2 s or less; highest hotspot found 85 C, `measurements/leftovers-c-b.md`).
- New from leftovers C: with both FMA waves raised to level 3, the matrix wave (older, lower level) finishes before the
  second FMA wave in the normal object, while the round-robin object finishes both FMA waves first; the mechanism is
  not established (MT-7, `measurements/leftovers-c-b.md`).
- New from leftovers C: why sparse FP16 costs 76 to 79 cycles per hand-over when dense FP16 costs 105 to 110 (MT-5b,
  `measurements/leftovers-c-b.md`).

Answered since (leftovers C, 2026-11-17):

- Answered in matrix-arithmetic.md, items 13 and 13a, and by leftovers C (MT-1, MA-2, `measurements/leftovers-c-b.md`):
  what the forbidden sequence (result as A or B without a V_NOP) reads: the old contents of the first result's VGPRs 0
  and 1, in lanes 0, 1, 8, 9 as A and lanes 0 to 15 as B, everything else new, for every form pair tried, alone and
  beside a busy wave. MEASURED.
- Answered in compiler-and-hazards.md, item 8, scheduling-and-clocks.md, item 4 and vulkan-view.md, item 4, and by
  leftovers C (MT-2, CH-6, `measurements/leftovers-c-b.md`): the wave64 forms: the FP16 form chained costs 16.01 and
  16.000 cycles in wave64, the same as wave32; in Vulkan, wave64 FP16, BF16, FP8 and 8-bit run at full rate (FP16
  210.6 TFLOPS, FP8 421.0, 8-bit 421.3 TOPS at W = 8). Leftovers C ran 20 forms in wave64 with EXEC full or one lane:
  16.000 cycles for the 16-bit-input forms and 8.000 for the 8-bit and 4-bit forms, the same as wave32. MEASURED.
- Answered (leftovers C, MT-3, `measurements/leftovers-c-b.md`): whole-card rates of the remaining forms at one wave
  per SIMD, ratio to 128 x clock / cycles: sparse BF16 into FP32 0.9988, sparse BF8 0.9995, BF8 0.9996, FP16 into FP16
  0.9997, BF16 into BF16 0.9990, sparse FP16 into FP16 0.9996 (within 0.12 %). At 16 waves the ratios range from 0.9875
  to 1.0151 in one burst each, read as carrying about 1.5 % of uncertainty. MEASURED.
- Answered in matrix-timing.md, item 9a: 8 waves per SIMD (W = 8): FP16 chain 16.384 cycles per instruction, four
  accumulators 16.390; FP8 chain 8.055, four accumulators 8.062; the waves finish oldest first.
- Answered in execution-core.md, items 7 and 9 (and scheduling-and-clocks.md, item 1, as the leftovers C plan names
  it): the single-wave clock is read from inside the kernel (cycle counter against REALTIME), so a job shorter than the
  1-second clock log is covered: for example 3,484 MHz median over three 1-second spins (item 7), and the climb from
  the idle clock (item 9).

## Helpers

`src/study/mat_card.hip` and `scripts/mat-expect/mat_card.txt` were written by a Sonnet 5.5 helper from a sealed
brief (placement with unused LDS, the W sweep, wall-clock rate against 128 SIMDs x the counter clock, the 50 ms
sampler, the thermal stop at 95 C, the verdict and its break). The lead read the whole file, rebuilt it, checked the
listing (each register-only loop holds exactly 16 matrix instructions of its form and only `s_add_co_i32`,
`s_cmp_eq_u32`, `s_cbranch_scc0` besides; A and B never overlap an accumulator; no scratch) and changed three
things before the final run: the FMA load used `x = x * k0 + k1`, which compiled to unpaired `v_fma_f32` and gave
0.48 of the dual-issue rate (now `x += k0 * k1`, all `v_dual_fmac_f32`, 0.945); the timed launches were 0.5 s and
a configuration reached 91 C in 2 s (now 0.2 s, and each configuration starts below 60 C); an expectation for the FMA
loop was added. Every number in items 11 and 12 is from the lead's run of the script, not from the helper's runs
(`runs/dev-matrix-card/`), which agree within 3 % (FP16 213-216 TFLOPS at a 3.33 GHz clock in a cooler state; FP8
414-419).
