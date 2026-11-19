# RDNA 4 execution core (lane 1)

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201, 64 CUs, wave32), ROCm 10.1.0 (hipcc, clang 24), kernel 7.3.4.
Run folder: `runs/20261117T110801Z-core/` (`scripts/study-core.sh all`, commit 9f5ad7a). Every number below is
MEASURED in that run unless it is labelled READ (from a document) or INFERRED (reasoning). Item 6 was corrected
later the same day and its numbers are from `runs/20261117T133209Z-core/` (see the note in item 6). Cycles are shader-clock
cycles from the wave's own cycle counter, so they do not depend on the clock frequency. Unless an item says
otherwise, one wave runs alone on its SIMD.

## In short

- The vector unit issues one instruction per cycle per wave. Most 32-bit operations (FP32 add, multiply, FMA, min and
  max, integer add, shift and logic, packed FP16, `v_fma_mix_f32`, conversions, `v_perm_b32`, the permlane forms)
  take that one cycle and need 5 cycles before a dependent instruction can use the result. `v_dot2_f32_f16` and its
  BF16 form also issue every cycle; their result takes 6.
- Dual issue works for every pair the guide allows: two operations per cycle, for example 2 FP32 FMAs per lane per
  cycle (INFERRED: 64 CUs x 2 SIMDs x 32 lanes x 2 FMAs x 2 flops = 16,384 flops per clock for the card; the clock
  under a full-card load is lane 7's subject). The compiler forms these pairs by itself for a plain FMA loop. Pairs
  outside the list cannot be encoded and run as two instructions.
- Slower classes: 32-bit integer multiply, `v_mad_co_u64_u32` and every transcendental (rcp, rsq, sqrt, exp, log,
  sin, cos, FP16 rcp) issue every 4 cycles. 64-bit shift and DPP8 issue every 2. FP64 add, multiply and FMA take
  32 cycles each (1/32 rate) with a 38-cycle latency, which is exactly LLVM's number.
- A transcendental does not block the vector unit: an exp plus three independent FMAs costs 4 cycles, the same as the
  exp alone.
- Surprises: (1) waves sharing a SIMD do not hide each other's latency unless every dependent instruction is preceded
  by `s_delay_alu`. Without it, 16 waves each running a dependent chain get 0.20 operations per cycle, the same as
  one wave alone; with it, 5 waves fill the SIMD (1.0); with 6 to 16 the five oldest run first and the rest form a
  tail, so the average over a run drops to 0.60 to 0.80 (item 8f). (2) An FMA whose three sources are three
  different registers of one register bank costs 5 cycles instead of 1, and other waves do not fill them (item 8g). (3) Vector-to-scalar-to-vector round trips cost 25 to 32
  cycles. (4) Every branch is expensive: 9 cycles when not taken, 15 when taken; vector instructions with EXEC = 0
  still take their full time, and skipping a masked-off region with a branch only pays from about 9 instructions up.
- The two SIMDs of a CU, and the four SIMDs of a work-group processor, are fully independent: one wave per SIMD gives
  exactly 2x and 4x the single-SIMD rate.
- The cycle counter is 64 bits wide and runs at the shader clock (3.40 to 3.57 GHz in this run with one wave busy,
  matching `pp_dpm_sclk`). It keeps counting while a wave waits for memory or sleeps. Its value differs between
  SIMDs by up to 4e9 cycles, so only differences within one wave (or one SIMD) mean anything.

## 1. Calibration

| Loop | Cycles per FMA | Known answer |
|---|---|---|
| First-hour loop (`src/wmma_time.hip` "valu", 8 chains, loop control included) | 1.080 (1.080-1.080) | 1.08 (first hour) |
| Inline-assembly `v_fma_f32`, 16 independent chains, differential (loop control removed) | 1.000 (1.000-1.000) | 1.00 |

Both reproduce. The extra 0.08 cycles per FMA in the first-hour figure is that loop's own control and counter reads
(41 cycles per loop iteration of 512 FMAs); the differential method removes it. The inline-assembly loops' control
costs 19.2 cycles per iteration (`s_add_co_i32`, `s_wait_alu`, `s_cmp`, `s_cbranch`; item 5 explains why that much).

## 2. Throughput against latency per instruction class

Columns: cycles per instruction with 1 chain (each instruction waits for the previous: the latency a dependent
instruction sees), and with 16 independent chains (the issue cost). Spread: the range from the extreme repetitions
(7 timed repetitions in each of 3 launches per build, both ends of the differential taken at their extremes); where
the range is under 0.02 only the median is shown. "LLVM" is the
`GFX12SpeedModel` latency of the class I matched the instruction to (READ, `notes/research-10-compilers.md`; the
instruction-to-class mapping is INFERRED from the class names, not read from LLVM's tables).

| Instruction | Class | 1 chain (latency) | 16 chains (issue) | Ops per cycle | LLVM |
|---|---|---|---|---|---|
| `v_add_f32` | fp32 | 5.00 | 1.00 | 1.00 | 5 |
| `v_mul_f32` | fp32 | 5.00 | 1.00 | 1.00 | 5 |
| `v_fma_f32` | fp32 | 5.00 | 1.00 | 1.00 | 5 |
| `v_fmac_f32` | fp32 | 5.00 (5.00-5.16) | 1.00 | 1.00 | 5 |
| `v_min_num_f32` | fp32 | 5.00 (5.00-5.16) | 1.00 | 1.00 | 5 |
| `v_max_num_f32` | fp32 | 5.00 (5.00-5.23) | 1.00 (1.00-1.11) | 1.00 | 5 |
| `v_minimum_f32` (NaN-propagating) | fp32 | 5.00 | 1.00 | 1.00 | 5 |
| `v_maximum_f32` (NaN-propagating) | fp32 | 5.00 (5.00-5.48) | 1.00 | 1.00 | 5 |
| `v_cndmask_b32` (VCC) | fp32 | 5.00 (5.00-5.26) | 1.00 (1.00-1.18) | 1.00 | 5 |
| `v_cmp_lt_f32` + v_cndmask_b32 (per instruction) | fp32 | 3.00 (3.00-3.15) | 1.21 (1.13-1.21) | 0.83 | 5 |
| `v_add_f32` (one SGPR operand) | fp32 | 5.00 (5.00-5.17) | 1.00 | 1.00 | 5 |
| `v_fma_f32` (two SGPR operands) | fp32 | 5.00 | 1.00 | 1.00 | 5 |
| `v_add_nc_u32` | int | 5.00 (5.00-5.26) | 1.00 | 1.00 | 5 |
| `v_lshlrev_b32` | int | 5.00 (5.00-5.15) | 1.00 | 1.00 | 5 |
| `v_and_b32` | int | 5.00 (5.00-5.23) | 1.00 | 1.00 | 5 |
| `v_or_b32` | int | 5.00 | 1.00 | 1.00 | 5 |
| `v_mul_u32_u24` | int | 5.00 (5.00-5.19) | 1.00 | 1.00 | 5 |
| `v_mul_lo_u32` | int | 8.00 (7.71-8.00) | 4.00 (3.89-4.64) | 0.25 | 6 |
| `v_mul_hi_u32` | int | 8.00 | 4.00 (3.79-4.00) | 0.25 | 6 |
| `v_mad_co_u64_u32` | int64 | 7.00 (7.00-7.15) | 4.00 | 0.25 | 6 |
| `v_lshlrev_b64` | int64 | 6.00 (6.00-6.20) | 2.00 | 0.50 | 6 |
| `v_pk_add_f16` | fp16x2 | 5.00 (4.86-5.00) | 1.00 | 1.00 | 5 |
| `v_pk_mul_f16` | fp16x2 | 5.00 (4.82-5.00) | 1.00 | 1.00 | 5 |
| `v_pk_fma_f16` | fp16x2 | 5.00 (4.84-5.46) | 1.00 | 1.00 | 5 |
| `v_fma_mix_f32` | mixed | 5.00 (4.85-5.00) | 1.00 | 1.00 | 5 |
| `v_dot2_f32_f16` | dot | 6.00 (5.82-6.38) | 1.00 | 1.00 | 5 |
| `v_dot2_f32_bf16` | dot | 6.00 (6.00-6.31) | 1.00 | 1.00 | 5 |
| `v_cvt_f32_i32` | cvt | 5.00 (5.00-5.15) | 1.00 | 1.00 | 5 |
| `v_cvt_i32_f32` | cvt | 5.00 | 1.00 | 1.00 | 5 |
| `v_cvt_f16_f32` | cvt | 5.00 | 1.00 (0.76-1.00) | 1.00 | 5 |
| `v_cvt_f32_f16` | cvt | 5.00 | 1.00 | 1.00 | 5 |
| `v_cvt_pk_rtz_f16_f32` | cvt | 5.00 | 1.00 | 1.00 | 5 |
| `v_rcp_f32` | trans | 10.00 (10.00-10.37) | 4.00 | 0.25 | 9 |
| `v_rsq_f32` | trans | 10.00 (9.88-10.21) | 4.00 | 0.25 | 9 |
| `v_sqrt_f32` | trans | 10.00 | 4.00 | 0.25 | 9 |
| `v_exp_f32` | trans | 10.00 | 4.00 | 0.25 | 9 |
| `v_log_f32` | trans | 10.00 (9.92-10.00) | 4.00 | 0.25 | 9 |
| `v_sin_f32` | trans | 10.00 | 4.00 (3.94-4.08) | 0.25 | 9 |
| `v_cos_f32` | trans | 10.00 | 4.00 | 0.25 | 9 |
| `v_rcp_f16` | trans | 10.00 | 4.00 | 0.25 | 9 |
| `v_add_f64` | fp64 | 38.00 (37.97-38.04) | 32.00 (31.94-32.06) | 0.03 | 38 |
| `v_mul_f64` | fp64 | 38.00 (38.00-38.03) | 32.00 (31.94-32.06) | 0.03 | 38 |
| `v_fma_f64` | fp64 | 38.00 (37.97-38.04) | 32.00 (31.93-32.00) | 0.03 | 38 |
| `v_rcp_f64` | fp64 | 40.00 (40.00-40.02) | 32.00 | 0.03 | 40 |
| `v_perm_b32` | lane | 5.00 | 1.00 | 1.00 | 5 |
| `v_mov_b32` DPP quad_perm | lane | 5.00 | 1.00 | 1.00 | 5 |
| `v_mov_b32` DPP8 | lane | 6.00 | 2.00 | 0.50 | 5 |
| `v_permlane16_var_b32` | lane | 5.00 | 1.00 | 1.00 | 5 |
| `v_permlanex16_var_b32` | lane | 5.00 | 1.00 | 1.00 | 5 |
| `v_permlane16_b32` (SGPR selects) | lane | 5.00 (5.00-9.05) | 1.00 | 1.00 | 5 |
| `v_permlanex16_b32` (SGPR selects) | lane | 5.00 (5.00-5.17) | 1.00 | 1.00 | 5 |
| `ds_bpermute_b32` (c16: 16 issued, one wait) | lane | 36.00 (35.84-36.11) | 5.06 | 0.20 | 20 |
| `v_s_rcp_f32` (chain incl. s_wait_alu) | pseudo | 19.00 (18.90-19.60) | 1.88 | 0.53 | 8 |
| `v_s_rsq_f32` | pseudo | 19.00 (19.00-19.14) | 1.88 (1.88-2.35) | 0.53 | 8 |
| `v_s_sqrt_f32` | pseudo | 19.00 (19.00-19.40) | 1.88 | 0.53 | 8 |
| `v_s_exp_f32` | pseudo | 19.00 (19.00-19.16) | 1.88 (1.88-2.07) | 0.53 | 8 |
| `v_s_log_f32` | pseudo | 19.00 (19.00-19.13) | 1.88 | 0.53 | 8 |

Chains 2, 4 and 8 sit where they should: with c chains a latency-L op costs max(issue, L / c) per instruction (for
example FP32 add: 5.00, 2.50, 1.25, 1.00, 1.00 for c = 1, 2, 4, 8, 16). INFERRED from that rule: 5 independent
instructions keep a full-rate class busy, and 3 a quarter-rate one (10 / 4).

Where the hardware differs from LLVM's model:
- Quarter-rate 32-bit integer multiply (`v_mul_lo_u32`, `v_mul_hi_u32`): latency 8, issue 4. LLVM's quarter-rate
  class says 6 and has no issue cost, so it schedules multiplies as if they were nearly free.
- Transcendentals: latency 10 (LLVM 9) and issue every 4 cycles, which LLVM does not model (it has no separate
  transcendental unit for gfx12). They do overlap with other vector work (item 3).
- 64-bit shift and DPP8: issue every 2 cycles, latency 6; LLVM says 6 for 64-bit and 5 for DPP8 with no issue cost.
- `v_dot2_f32_f16`/`bf16`: latency 6, not 5.
- FP64: latency 38 and 40 (rcp) match LLVM exactly; LLVM does not model the 32-cycle issue cost.
- `ds_bpermute_b32`: 36 cycles latency (LLVM's LDS class: 20).
- Pseudo-scalar `v_s_*` (SGPR in and out): a dependent chain costs 19 cycles per op including the `s_wait_alu`
  that the SGPR hazard rule requires (LLVM: 8). Independent ones cost 1.88 cycles each (30 cycles per 16; INFERRED:
  issued at roughly the vector rate, slowed by the scalar destination writes; not explained further).
- Reading SGPR operands costs nothing: `v_add_f32` with one SGPR and `v_fma_f32` with two SGPRs run at 1.00, like the
  VGPR forms; `v_cndmask_b32` reading VCC also runs at 1.00.
- A compare followed by a select that reads its VCC (`v_cmp_lt_f32` then `v_cndmask_b32`) costs 6 cycles per pair
  as a chain (3.00 per instruction), so VCC reaches the select quickly; 16 such pairs that all write VCC cost 2.4
  cycles per pair (1.2 per instruction).

Values: the chains start at 1.0. The exp and log chains run into infinity or NaN after a few steps, while rcp, rsq,
sqrt, sin and cos chains stay finite (1.0 or 0.0); all transcendentals time the same, so special values did not change
the timing here.

## 3. Dual issue (VOPD) and register banks

One wave, 16 independent chains, fixed registers so that the guide's bank rules hold (`src/study/core_dual.hip`).
A "pattern" is one `v_dual_X :: v_dual_Y` instruction (two operations) or the same two operations as two ordinary
instructions. Values are cycles per pattern. The dual figures read 0.984 rather than 1.000 because the 4-block loop
of this probe costs one cycle per iteration more than the 8-block loop (63 against 64 cycles per 64 patterns); the
resolution of this probe is therefore +-0.016.

| Pattern | Cycles per pattern | Range | What it is |
|---|---|---|---|
| du_fmac | 0.984 | 0.984-0.984 | v_dual_fmac_f32 x2 |
| du_fmaak | 1.233 | 1.233-1.233 | v_dual_fmaak_f32 x2 (shared literal, 96-bit instruction) |
| du_fmamk | 1.233 | 1.233-1.233 | v_dual_fmamk_f32 x2 (shared literal, 96-bit instruction) |
| du_mul | 0.984 | 0.984-0.984 | v_dual_mul_f32 x2 |
| du_add | 0.984 | 0.984-0.984 | v_dual_add_f32 x2 |
| du_sub | 0.984 | 0.984-0.984 | v_dual_sub_f32 x2 |
| du_subrev | 0.984 | 0.984-0.984 | v_dual_subrev_f32 x2 |
| du_muldx9 | 0.984 | 0.984-0.984 | v_dual_mul_dx9_zero_f32 x2 |
| du_mov | 0.984 | 0.984-0.984 | v_dual_mov_b32 x2 |
| du_cnd | 1.046 | 1.046-1.046 | v_dual_cndmask_b32 x2 (VCC) |
| du_max | 0.984 | 0.984-0.984 | v_dual_max_num_f32 x2 |
| du_min | 0.984 | 0.984-0.984 | v_dual_min_num_f32 x2 |
| du_dotf16 | 0.984 | 0.984-0.984 | v_dual_dot2acc_f32_f16 x2 |
| du_dotbf16 | 0.984 | 0.984-0.984 | v_dual_dot2acc_f32_bf16 x2 |
| du_addnc | 0.984 | 0.984-0.984 | v_dual_add_f32 :: v_dual_add_nc_u32 |
| du_lshl | 0.984 | 0.984-0.984 | v_dual_add_f32 :: v_dual_lshlrev_b32 |
| du_and | 0.984 | 0.984-0.984 | v_dual_add_f32 :: v_dual_and_b32 |
| du_fmac_addnc | 0.984 | 0.984-0.984 | v_dual_fmac_f32 :: v_dual_add_nc_u32 |
| du_mul_samesrc0 | 0.984 | 0.984-0.984 | v_dual_mul_f32 x2, both src0 = v1 (gfx12 same-VGPR exception) |
| sp_fmac | 1.984 | 1.984-1.984 | v_fmac_f32 + v_fmac_f32 as two instructions |
| sp_mul | 1.984 | 1.984-1.984 | v_mul_f32 + v_mul_f32 as two instructions |
| sp_addnc | 1.984 | 1.984-1.984 | v_add_f32 + v_add_nc_u32 as two instructions |
| sp_dotf16 | 1.984 | 1.984-1.984 | v_dot2_f32_f16 x2 as two VOP3P instructions |
| sp_fmaak | 1.984 | 1.984-1.984 | 2 x v_fmaak_f32 (VOP2 + literal, 64-bit instructions) |
| sp_fma3lit | 1.984 | 1.984-1.984 | 2 x v_fma_f32 with a literal (VOP3 + literal, 96-bit instructions) |
| sp_fma3nolit | 1.984 | 1.984-1.984 | 2 x v_fma_f32 with an inline constant (64-bit instructions) |
| tr_exp2 | 7.968 | 7.968-7.968 | 2 x v_exp_f32 (known: 4 cycles each) |
| tr_exp_fma | 3.968 | 3.968-4.898 | v_exp_f32 + 1 v_fma_f32 (4 if they overlap, 5 if not) |
| tr_exp_fma3 | 3.984 | 3.984-3.984 | v_exp_f32 + 3 v_fma_f32 (4 if they overlap, 7 if not) |
| sp_mix_add | 1.984 | 1.984-2.300 | v_fma_mix_f32 + v_add_f32 (fma_mix has no dual form) |
| sp_fma3_add | 1.984 | 1.984-1.984 | v_fma_f32 (VOP3) + v_add_f32 |
| sp_mullo_add | 4.984 | 4.984-4.984 | v_mul_lo_u32 + v_add_f32 |
| c_fmac | 0.500 | 0.500-0.500 | compiler-built: 16 chains x = fmaf(k0, k1, x), dual issue on (pattern = one FMA) |
| du_fmac_dep | 4.999 | 4.999-4.999 | v_dual_fmac_f32 x2, every pattern on the same chain pair (dependent) |

Findings:
- All 14 first-slot operations and the 3 second-slot-only operations pair at one instruction per cycle, so two
  operations per cycle. The same work as two ordinary instructions takes 2 cycles. So the hardware does not pair
  ordinary instructions by itself: only the VOPD encoding dual-issues.
- `v_fma_mix_f32`, VOP3 `v_fma_f32` and `v_mul_lo_u32` have no dual form (the assembler rejects them); beside
  `v_add_f32` they cost the sum of their parts.
- Pairs carrying a literal (`v_dual_fmaak_f32`, `v_dual_fmamk_f32`, a 96-bit instruction) cost 1.23 cycles. Ordinary
  instructions with a literal (64 or 96 bits) cost 1.00, so it is the combination of VOPD and literal; the cause is
  not known (not explained by instruction size alone).
- `v_dual_cndmask_b32` in both halves (both read VCC): 1.05 cycles.
- The gfx12 relaxation that lets both halves read the same VGPR in the same slot (`AllowSameVGPR` in LLVM) costs
  nothing (0.984), and the compiler uses it: its own loop pairs `v_dual_fmac_f32 vA, v4, v7 :: v_dual_fmac_f32 vB,
  v4, v7` and reaches 0.500 cycles per FMA.
- Bank rules the assembler enforces (`llvm-mc`, run folder `vopd-assembler.txt`): src0 of both halves in different
  banks (unless the same VGPR), src1 likewise, destinations of different parity, the operands read through the SRC2
  port of different parity (tested with FMAMK), one shared literal. It also rejects second-slot-only ops in the
  first slot and ops without a dual form. It accepts a pair whose second half reads the first half's destination,
  which the guide forbids ("OPX must not overwrite sources of OPY"); what the hardware then reads was not tested.
  So a violating register assignment that the assembler accepts could not be built for the bank rules themselves,
  and the bank effect was measured on ordinary instructions instead:

| Two `v_fma_f32`, sources (bank = register mod 4) | Cycles per pair |
|---|---|
| three banks each (v1,v2,v3) then (v5,v6,v4) | 1.984 |
| two sources in bank 1, changing or repeated | 1.984 / 1.984 |
| one register three times, or (v1,v1,v5) | 1.984 / 1.984 |
| three bank-1 registers, the same triple in both | 2.968 |
| a bank-2/3 instruction, then a bank-1 triple | 1.984 |
| (v1,v5,v9) then (v13,v17,v2) | 7.999 |
| (v1,v5,v9) then (v13,v17,v21): two different bank-1 triples | 9.999 |

  So two sources in one bank are free, three different registers in one bank cost an extra half cycle when the
  instruction repeats, and consecutive instructions that each need three or two-plus registers from the same bank
  cost 4 to 5 cycles each. INFERRED: each bank delivers a limited number of reads per cycle and a small operand
  cache absorbs repeats; the 5-cycle case looks like a pipeline stall rather than a queue, but the mechanism is not
  established. Compilers do not avoid this pattern for VOP3 (LLVM checks banks only for VOPD).
- Transcendental beside FMAs: `v_exp_f32` alone 4 cycles; exp + 1 FMA 3.97; exp + 3 FMAs 3.98. The FMAs issue
  while the transcendental unit is busy, as on RDNA 3.

## 4. The scalar unit, hand-offs between the units, and `s_delay_alu`

`src/study/core_scalar.hip`. Cycles per operation (chains: one dependent chain; independent: 16 SGPR chains).

| Operation | Dependent chain | 16 independent | LLVM |
|---|---|---|---|
| `s_add_co_i32` | 2.00 | 1.19 | 2 |
| `s_and_b32` | 2.00 | 1.19 | 2 |
| `s_lshl_b32` | 2.00 | 1.19 | 2 |
| `s_mul_i32` | 3.00 | 2.00 | 2 |
| `s_mul_hi_u32` | 3.00 (3.00-3.30) | 2.00 | 2 |
| `s_add_nc_u64` | 2.00 (1.98-2.00) | 1.12 | 2 |
| `s_mul_u64` | 6.00 | 5.00 (5.00-5.05) | 2 |
| `s_add_f32` | 4.00 | 1.19 | 4 |
| `s_mul_f32` | 4.00 | 1.19 (1.19-1.29) | 4 |
| `s_fmac_f32` | 4.00 | 1.19 | 4 |
| `s_cvt_f32_i32` | 4.00 (3.83-4.00) | 1.00 | 4 |

| Sequence (one pattern = one step of a closed dependency loop) | Cycles per step |
|---|---|
| `s_mul_i32` chain with `s_wait_alu depctr_sa_sdst(0)` after each | 10.00 (10.00-10.12) |
| `s_mul_i32` chain alone (mixed kernel) | 3.00 |
| the same + `s_wait_alu` (mixed kernel) | 10.00 |
| `v_readfirstlane` -> `s_mul_i32` -> wait -> `v_add_nc_u32` (compiler form) | 32.00 (32.00-32.13) |
| the same without the wait | 28.00 (27.92-28.47) |
| `s_mul_i32` chain, each result read by a vector op, wait (compiler form) | 15.00 (14.85-15.29) |
| the same without the wait | 10.00 |
| `v_cmp` -> VCC -> `s_and_b32` -> wait -> next `v_cmp` | 25.00 (24.47-25.00) |
| the same without the wait | 25.00 (25.00-25.11) |
| `v_readfirstlane` -> `s_add_co_i32` -> wait -> `v_mov_b32` | 30.00 (29.91-30.10) |
| `v_add_f32` chain | 5.00 |
| `v_add_f32` chain, `s_delay_alu instid0(VALU_DEP_1)` before each | 5.00 (5.00-5.18) |
| `v_exp_f32` chain | 10.00 |
| `v_exp_f32` chain, `s_delay_alu instid0(TRANS32_DEP_1)` before each | 10.00 |
| 16 independent `v_add_f32` | 1.00 |
| 16 independent `v_add_f32`, each behind `s_delay_alu instid0(VALU_DEP_1)` | 5.00 |

- Scalar integer add, logic and shift: 2 cycles to a dependent scalar instruction (LLVM `WriteSALU` 2: matches).
  Scalar float (`s_add_f32`, `s_mul_f32`, `s_fmac_f32`, `s_cvt_f32_i32`): 4 (LLVM `WriteSFPU` 4: matches).
  `s_mul_i32` and `s_mul_hi_u32`: latency 3, one every 2 cycles. `s_mul_u64`: latency 6, one every 5.
- Independent scalar instructions cost 1.19 cycles each (19 cycles per 16; 64-bit add 18 per 16; `s_cvt_f32_i32`
  16 per 16). The extra cycles are not explained.
- `s_wait_alu depctr_sa_sdst(0)` after a scalar result costs about 7 cycles more than the plain dependency (a
  `s_mul_i32` chain: 3 cycles per step, 10 with the wait). clang 24 inserts this wait for the gfx12 SGPR read hazard
  (READ, guide and research-10) in many places, among them the loop control of every probe loop here (after the loop
  counter's `s_add_co_i32`), which is part of why a loop iteration costs 19 cycles.
- Round trips: vector -> `v_readfirstlane` -> scalar multiply -> back into a vector add: 32 cycles with the guard the
  compiler inserts, 28 without. A scalar multiply chain feeding a vector op each step: 15 with the guard, 10 without.
  Vector compare -> VCC -> scalar AND -> next compare: 25 cycles either way. Vector -> scalar -> `v_mov` back: 30.
  The "no guard" timings are timing only: the guide says the value may be wrong without the wait.
- `s_delay_alu`, one wave alone: in front of each instruction of a dependent chain it changes nothing (FP32 add chain
  5.00 with and without; exp chain 10.00 with and without). In front of independent instructions it costs the full
  latency: 16 independent adds behind `s_delay_alu instid0(VALU_DEP_1)` cost 5.00 cycles each instead of 1.00. The
  hardware obeys the hint literally. Its real effect shows with several waves (item 6).

## 5. Control flow

`src/study/core_flow.hip` (written by a helper, reviewed and rerun here). A pattern is the whole group in the
first column; 16 patterns per block, method as above. The patterns use local labels (`1:` and `1f`/`1b`), so each
branch goes to the instruction right after it or back to the loop head; checked in the `.lto.s`.

| Pattern | Cycles per pattern |
|---|---|
| `s_cmp_eq_u32` alone | 1.00 |
| `s_cmp` + `s_cbranch_scc1`, not taken | 10.00 (9.92-10.12) |
| `s_cmp` + `s_cbranch_scc1`, taken (to the next instruction) | 16.00 (15.62-16.12) |
| `s_branch` to the next instruction | 9.00 (9.00-9.04) |
| `s_branch` over one `v_add_f32` | 9.00 (9.00-9.58) |
| `v_add_f32` alone (reference) | 1.00 |
| `s_add_co_i32` + `s_cmp_lg_u32` (loop counter, no branch) | 3.00 (3.00-3.16) |
| assembly loop, 2 trips, empty body (`s_mov` + 2 x add/cmp/branch) | 34.00 (34.00-34.51) |
| the same, 4 trips | 74.00 (73.81-74.21) |
| 2 trips, one `v_fma_f32` in the body | 35.62 (35.47-35.99) |
| 4 trips, one `v_fma_f32` in the body | 78.88 (78.44-79.00) |
| `v_fma_f32` alone (reference) | 1.00 |
| `v_fma_f32` alone, all operands the chain register (reference for the next two) | 1.00 |
| `s_and_saveexec_b32` + `v_fma_f32` + `s_or_b32 exec_lo`, mask all lanes | 12.00 (12.00-12.63) |
| the same, mask 16 of 32 lanes | 12.00 (11.44-13.16) |
| `s_mov_b32 exec_lo, -1` alone, back to back | 2.38 |
| 8 independent `v_fma_f32` | 8.00 |
| `s_mov exec_lo, -1` + 8 FMAs + `s_mov exec_lo, -1` | 16.00 (16.00-19.79) |
| `s_mov exec_lo, 0` + 8 FMAs + `s_mov exec_lo, -1` (FMAs run with EXEC = 0) | 16.00 (13.30-19.78) |
| `s_mov exec_lo, 0` + `s_cbranch_execz` over the 8 FMAs (taken) + restore | 17.06 (17.06-21.81) |
| `s_mov exec_lo, -1` + `s_cbranch_execz` (not taken) + 8 FMAs + restore | 19.00 (16.44-22.19) |
| `s_mov exec_lo, 0` + `s_cbranch_execz` taken, nothing to skip + restore | 17.06 (17.06-17.43) |

Findings (differences between rows are INFERRED arithmetic on the MEASURED rows):
- A branch costs 9 cycles even when not taken (`s_cmp` + not-taken `s_cbranch_scc1` 10, `s_cmp` alone 1). Taken:
  15. An unconditional `s_branch`: 9. The instruction a branch jumps over costs nothing (9 with or without a
  skipped `v_add_f32`). LLVM's `WriteBranch` is 32 (READ): on this card a branch is cheaper than that, but far
  more expensive than an ALU instruction.
- Loop back-edge: each extra trip of an assembly loop (`s_add_co_i32`, `s_cmp_lg_u32`, taken `s_cbranch_scc1`)
  costs 20 cycles ((74 - 34) / 2); the counter update and compare alone cost 3, so the taken back-edge branch is
  about 17. This is where the 19 cycles of loop control per iteration in every probe come from. An FMA inside such
  a loop adds only 1.2 to 0.8 cycles per trip: it issues in the shadow of the branch.
- Execution mask: `s_and_saveexec_b32` + one FMA + `s_or_b32 exec_lo` costs 12 cycles against 1 for the FMA, the same
  with all lanes or half of them on (masked-off lanes save nothing). Back-to-back `s_mov_b32 exec_lo` cost 2.38
  cycles each, but two EXEC writes around 8 FMAs add 8 cycles (16 against 8): INFERRED, the first vector
  instruction after an EXEC write waits for it.
- Empty mask: 8 FMAs with EXEC = 0 cost exactly what they cost with all lanes on (16.00 and 16.00). The hardware does
  not skip vector instructions when EXEC is zero (the guide says it may, READ; it does not here). Jumping over them
  with `s_cbranch_execz` costs 17.06 whether there are 8 FMAs to skip or none, against 16.00 for running the 8 with
  EXEC = 0, and 19.00 when the branch is not taken. So for 8 FMAs the skip branch costs more than running them
  masked off; INFERRED: it pays only for regions longer than about 9 full-rate instructions, and costs 3 cycles
  when the region does run.
- Caveat: in the `--break` run of the same build, `loop4` (74.07) was reported non-linear (U = 6 off by 2.3 cycles
  per iteration, limit 1.5); in the normal run it was linear. Treat the loop figures as +-1 cycle.

## 6. Several waves on one SIMD; SIMDs of a CU and of a work-group processor

`src/study/core_share.hip`. One work-group of 1024 threads (32 waves). With `-mcumode` it sits on one CU, 16 waves on
each of its 2 SIMDs; without (WGP mode, the HIP default) 8 waves on each of the 4 SIMDs of a work-group processor
(MEASURED from `HW_REG_HW_ID1` in every launch). What could be controlled: how many waves of the group run on each
SIMD the group received (the others end at once after one barrier), and CU against WGP mode. What could not: which
CU or work-group processor the group gets; in CU mode the group lands on either CU of a WGP from one launch to the
next (SIMD ids 0 and 2 or 1 and 3), so SIMDs are selected by their position within the CU.

Correction (lead, 2026-11-17, found by lane 4): the first version of this item reported that several waves on one
SIMD lose 7 to 13 % (1.070, 1.107, 1.125, 1.134 cycles per FMA for 2, 4, 8, 16 waves). That was a counting mistake
in the probe, not hardware. The span started at each wave's repetition 1 while the waves' repetition 0 was not
counted as work; with waves finishing one after another the span then holds 8n - 1 repetitions for 7n counted, a
factor (8n - 1) / (7n) = 1.071, 1.107, 1.125, 1.134. The probe now starts the span at repetition 0 and counts all 8
repetitions. Everything below is from the rerun `runs/20261117T133209Z-core/`.

Two measures: "own" = cycles per op one wave sees between its own counter reads; "SIMD" = the span from the first
active wave's start to the last one's end on that SIMD's counter, divided by all the waves' operations (cycles per
op of the SIMD). "Overlap" = sum of the waves' REALTIME durations / REALTIME span. All active waves take their start
stamp within a few tens of cycles of each other, so n waves that finish one after another give about (n + 1) / 2,
and n waves that all stay busy to the end give n.

CU mode, one SIMD: cycles per op of the SIMD (own cycles per op of one wave; overlap factor)

| Workload | 1 wave | 2 waves | 4 waves | 8 waves | 16 waves |
|---|---|---|---|---|---|
| ind_fma | 1.000 (1.00; 1.00) | 0.999 (1.00; 1.54) | 1.000 (1.00; 2.56) | 1.000 (1.00; 4.57) | 1.000 (1.01; 8.58) |
| dep_fma | 5.000 (5.00; 1.00) | 5.000 (5.00; 1.52) | 4.992 (5.00; 2.52) | 4.979 (5.00; 4.54) | 4.959 (5.03; 8.56) |
| dep_fma_delayalu | 5.000 (5.00; 1.00) | 2.500 (5.00; 2.00) | 1.250 (5.00; 4.00) | 1.250 (5.02; 5.44) | 1.222 (5.03; 9.06) |
| ind_exp | 4.000 (4.00; 1.00) | 4.004 (4.00; 1.51) | 4.002 (4.01; 2.51) | 4.001 (4.02; 4.52) | 4.001 (4.05; 8.56) |
| ind_dual_fmac | 0.500 (0.50; 1.00) | 0.510 (0.50; 1.54) | 0.504 (0.49; 2.56) | 0.502 (0.50; 4.57) | 0.501 (0.50; 8.59) |
| mix_exp_fma | 4.000 (4.00; 1.00) | 2.000 (4.00; 1.36) | 2.001 (4.00; 2.06) | 2.002 (4.03; 3.44) | 2.002 (4.04; 6.15) |
| mix_exp_depfma_da | 4.000 (4.00; 1.00) | 2.635 (5.33; 1.77) | 2.220 (5.43; 2.80) | 2.439 (9.13; 4.53) | 2.464 (9.47; 7.84) |

Total operations per cycle: 1 wave per SIMD on one SIMD, on both SIMDs of a CU (CU mode), on all 4 SIMDs of a WGP
(WGP mode); then the same with 8 waves per SIMD

| Workload | 1 SIMD | 2 SIMDs (CU) | 4 SIMDs (WGP) | 1 SIMD, 8 waves | 2 SIMDs, 8 waves each | 4 SIMDs, 8 waves each |
|---|---|---|---|---|---|---|
| ind_fma | 1.000 | 2.000 | 4.000 | 1.000 | 1.999 | 3.996 |
| dep_fma | 0.200 | 0.400 | 0.800 | 0.201 | 0.402 | 0.803 |
| dep_fma_delayalu | 0.200 | 0.400 | 0.800 | 0.800 | 1.600 | 3.199 |
| ind_exp | 0.250 | 0.500 | 1.000 | 0.250 | 0.500 | 1.000 |
| ind_dual_fmac | 2.000 | 4.000 | 8.000 | 1.993 | 3.985 | 7.992 |
| mix_exp_fma | 0.250 | 0.500 | 1.000 | 0.499 | 0.999 | 1.998 |
| mix_exp_depfma_da | 0.250 | 0.500 | 1.000 | 0.410 | 0.819 | 1.631 |

Findings:
- Waves that could each issue every cycle (independent FMAs, dual FMACs, exps) share the SIMD without loss: the SIMD
  delivers the single-wave rate whatever the number of waves (FMA 0.999 to 1.000 cycles per op with 2 to 16 waves;
  dual FMAC within 2 %; exp within 0.1 %). They finish one after another (overlap about (n + 1) / 2), and each wave
  sees the single-wave rate in its own timed repetitions: the oldest wave keeps the vector unit (INFERRED: the
  default oldest-first wave order; LLVM's kernel descriptor bit `WG_RR_EN` selects round robin, READ, not tried).
  The per-wave time stamps show a younger wave advancing while the older one runs (two FMA waves in a separate
  `SHARE_DEBUG=1` run by the lead, not kept in the run folder: the first ends after 86,384 cycles, the second after
  149,881, not 172,768); INFERRED: it uses the cycles the older wave spends
  in loop branches and counter reads, which the differential method removes from the per-op figures.
- Dependent chains without `s_delay_alu`: the waves do not fill each other's stall cycles. 2 to 16 waves give 4.96 to
  5.00 cycles per op for the SIMD, the same as one wave (5.00). This is what Mesa's comment says (READ, research-10):
  the SIMD does not switch to another wave on an ALU stall unless `s_delay_alu` tells it to.
- The same chains with `s_delay_alu instid0(VALU_DEP_1)` before each FMA: perfect interleaving up to 4 waves (2.50
  cycles per op with 2 waves, 1.25 with 4), and no further gain: 1.25 with 8 and 1.22 with 16 (0.80 to 0.82 ops per
  cycle, where 5 or more waves could in principle reach 1.0; not explained). So the compiler's `s_delay_alu` is what
  lets waves hide each other's latency. Hand-written assembly without it loses that.
- Mixed pair, exp wave + FMA wave on one SIMD: 2.000 cycles per op averaged over both, which is full overlap (one
  after the other would be 2.5): the FMA wave runs in the cycles the exp wave leaves free, as inside one wave.
- Independence: one wave on each of 2 SIMDs gives exactly 2x the total of one SIMD (FMA 2.000, exp 0.500, dual FMAC
  4.000 ops per cycle); one on each of the 4 SIMDs of a WGP, 4x (FMA 4.000). Every per-SIMD figure with 2 or 4 SIMDs
  active equals the 1-SIMD figure for the same number of waves per SIMD. So the SIMDs of a CU and of a WGP do not
  share issue bandwidth for vector work.

## 7. The cycle counter

`src/study/core_counter.hip` (written by a helper, reviewed and rerun here). Raw `s_getreg_b32` reads of
`SHADER_CYCLES_LO`/`_HI` and `s_sendmsg_rtn_b64` REALTIME reads in inline assembly; the host times the launches with
`std::chrono::steady_clock` and reads `pp_dpm_sclk` every 50 ms from a second thread while a 1-second kernel spins.

| Question | Answer | Label |
|---|---|---|
| Width and wrap | 64 bits: over a 1.28 s spin sampled every 1.0e6 cycles, LO wrapped past 2^32 once and HI went from 0 to 1 at exactly that sample; 4140 samples, none inconsistent; largest value 4.33e9 | MEASURED |
| Cost of one read | 8 cycles between two back-to-back `s_getreg_b32` reads (the guide says about 8, READ) | MEASURED |
| REALTIME rate | 99.992 MHz against host time (99.966 to 99.994 over 3 runs), i.e. 100 MHz within 0.01 % | MEASURED |
| Core clock from the counter | 3484 MHz median (3399, 3484, 3570 in the three 1-second runs); the same from host wall time | MEASURED |
| `pp_dpm_sclk` at the same time | 3332 to 3589 MHz in 51 readings, rising over the three runs like the counter's rate (run 1: 3332 to 3437 against 3399 derived; run 3: 3482 to 3589 against 3570) | MEASURED |
| Counts while waiting for memory | yes: during a dependent pointer chase through 512 MB (481 cycles, 137 ns per load) the counter advanced at 3584 MHz against REALTIME, the same as a vector loop just before (3581 MHz, ratio 1.001) | MEASURED |
| Counts during `s_sleep` | yes: `s_sleep 127` took 8128.0 cycles, exactly 64 x 127 (READ, guide), at 3579 MHz against 3583 for the control (ratio 0.999) | MEASURED |
| Same counter on all SIMDs | no: 128 waves on different CUs started within 0.16 us of each other (REALTIME), yet their counters differed by -5.7e7 to +4.1e9 cycles after removing the start skew; only 1 of 128 within 2000 cycles of the median. All of them tick at the same rate (35.69 cycles per REALTIME tick, 3569 MHz). INFERRED: each SIMD's counter starts at a different time (reset or power-up), so only differences taken on one SIMD mean anything | MEASURED |

The counter therefore measures time in shader-clock cycles, not instructions: cycle figures in this file are
independent of the clock, and seconds follow by dividing by the clock of the moment. pp_dpm_sclk shows the clock in
this run was 3.3 to 3.6 GHz with one wave busy (the card was not pinned; performance level "auto").

Raw lines from the run:

```
RESULT counter wrap samples=4140 run_s=1.281 span_cycles=4.174e+09 median_step=1000672 wraps_seen=1 bad=0 max_value=0x0000000101e85b05 (4326972165) max_hi=1
RESULT counter read_cost cycles_per_s_getreg median=8.0 min=8 max=1735
DATA counter rate_rep=0 wall_s=1.000059 drealtime=100000008 dcycles=3398604715 sclk_readings=3332MHz:x1,3353MHz:x1,3363MHz:x1,3364MHz:x1,3368MHz:x1,3369MHz:x1,3378MHz:x2,3389MHz:x1,3390MHz:x1,3394MHz:x1,3395MHz:x1,3408MHz:x1,3410MHz:x1,3419MHz:x2,3429MHz:x2,3435MHz:x1,3437MHz:x1
DATA counter sclk_raw 0: 500Mhz |1: 3332Mhz *|2: 2460Mhz |
DATA counter rate_rep=1 wall_s=1.000077 drealtime=100000032 dcycles=3483559147 sclk_readings=3430MHz:x1,3435MHz:x1,3436MHz:x1,3445MHz:x1,3446MHz:x1,3459MHz:x1,3460MHz:x1,3470MHz:x1,3472MHz:x1,3480MHz:x1,3481MHz:x1,3487MHz:x1,3489MHz:x1,3497MHz:x1,3499MHz:x1,3506MHz:x1,3509MHz:x1,3520MHz:x1,3522MHz:x1,3527MHz:x1
DATA counter rate_rep=2 wall_s=1.000342 drealtime=100000024 dcycles=3570071307 sclk_readings=3482MHz:x1,3527MHz:x1,3529MHz:x1,3538MHz:x1,3539MHz:x1,3548MHz:x1,3552MHz:x1,3578MHz:x1,3579MHz:x1,3580MHz:x2,3583MHz:x1,3584MHz:x6,3587MHz:x1,3589MHz:x1
RESULT counter realtime_rate_per_host_s median=99992357 min=99965788 max=99994131 Hz (expected 100000000, off by -0.008 %)
RESULT counter core_clock_MHz (cycles / REALTIME at 100000000 Hz) median=3483.6 min=3398.6 max=3570.1
RESULT counter core_clock_MHz (cycles / host wall time) median=3483.3 min=3398.4 max=3568.8
RESULT counter chase loads=4096 trials=8 cycles_per_load median=480.7 min=442.0 max=490.4
RESULT counter chase ns_per_load median=136.8 min=133.5 max=149.3 (REALTIME)
RESULT counter chase MHz_during_chase median=3583.5 min=2965.0 max=3588.2 ; VALU control just before: median=3581.2 min=2963.4 max=3584.1 ; chase/control=1.001
RESULT counter sleep sleeps=32000 cycles_per_s_sleep_127 median=8128.0 min=8128.0 max=8128.0 (guide: 64*127 = 8128)
RESULT counter sleep us_per_s_sleep_127 median=2.271 min=2.267 max=2.273 (REALTIME)
RESULT counter sleep MHz_during_sleep median=3578.7 min=3575.5 max=3584.7 ; VALU control just before: median=3583.0 min=3573.5 max=3583.3 ; sleep/control=0.999
RESULT counter simd waves=128 start_spread_ticks=16 (0.16 us) raw_c0_spread_cycles=4.173e+09
RESULT counter simd cycles_per_realtime_tick median=35.6902 min=35.6885 max=35.6917 (x100000000 Hz = 3569 MHz)
RESULT counter simd offset_vs_wave6_after_removing_start_skew_cycles min=-5.739e+07 median=-1.006e+07 max=4.116e+09 within_2000_of_median=1/128
```

## 8. The unexplained items, tested (timing leftovers, `tim_core`, `tim_disturb`)

Added by the leftovers pass. Run folder `runs/20261117T175731Z-timing/` (`scripts/study-leftovers-timing.sh`), MEASURED there
unless labelled otherwise. Method as above, with two differences: every timed region is one inline-assembly
statement holding the counter reads and the loop (as in `mat_share`), so nothing the compiler adds can land in it,
and the registers are fixed in the instruction text (`scripts/tim_gen_core.py` writes the 79 blocks of 16
instructions into `src/study/tim_core_blocks.h`), so each operand's register bank (register number mod 4) is set
by the source. Cost per instruction = (R = 6 blocks per iteration - R = 2) / (4 x 16 x 16 iterations), R = 4 must
sit midway. Samples above 1.5 x median + 64 cycles are dropped before the medians (lane 7's rule; the counts are in
each log, 0 to 9 of 75 per block). Calibration: independent `v_fma_f32` with inline constants 1.0000, VOP2
`v_add_f32` 1.0000 (PASS).

(a) The 1.23-cycle VOPD pairs with a literal follow the instruction length, not VOPD and not the literal.

| Block (16 instructions, independent) | Bytes each | Cycles per instruction |
|---|---|---|
| `v_fma_f32` with a real literal (0x3f400000), VOP3 | 12 | 1.2578 |
| the same with a VGPR as third source | 12 | 1.2578 |
| `v_mov_b32_e64` with a literal (VOP3) | 12 | 1.2578 |
| `v_dual_fmaak_f32` x2 (lane 1's case) | 12 | 1.2578 |
| `v_dual_fmaak_f32` :: `v_dual_mov_b32` | 12 | 1.2578 |
| `v_dual_mov_b32` with a real literal :: `v_dual_mov_b32` | 12 | 1.2578 |
| `v_dual_mov_b32` with 1.0 (0x3f800000, encoded as an inline constant) | 8 | 1.0000 |
| `v_mov_b32_e32` with a literal (VOP1) | 8 | 1.0000 |
| `v_add_f32_e32` with a literal (VOP2) | 8 | 1.0000 |
| `v_fmaak_f32` (VOP2, literal) | 8 | 1.0000 |
| `v_dual_fmac_f32` x2, `v_add_f32_e64`, `v_fma_f32` with inline constants | 8 | 1.0000 |
| 12-byte runs between 8-byte instructions: 1 of 4 / runs of 2 / 3 of 4 / runs of 4 / runs of 8 | 12 and 8 | 1.0010 / 1.0088 / 1.1309 / 1.1260 / 1.1250 |

- Any 12-byte instruction back to back costs 1.2578 cycles (161 per 128), whatever its kind; an 8-byte instruction
  costs 1.00 with or without a literal. A 12-byte instruction between shorter ones is free, and the cost grows with
  the length of a run of 12-byte instructions (about 2 cycles per run of 4 to 8). INFERRED: instruction fetch
  delivers a little under 12 bytes per cycle on average (about 9.5), and a small buffer covers short runs; the exact
  rule (buffer size, fetch width) is not established.
- Lane 1's two readings follow from this: its "VOP3 with a literal costs 1.00" used 0x3f800000 (1.0), which the
  assembler encodes as an inline constant, so those instructions were 8 bytes (READ, `llvm-mc -show-encoding`:
  `v_fma_f32 v1, v2, 0x3f800000, v3` is 8 bytes, with 0x3f400000 it is 12); its literal VOPD pairs, whose constant
  is always a separate literal word, were the only real 12-byte instructions it timed. (Lane 1 read 1.233 for them in a different
  loop; INFERRED: the code alignment differs; not tested.) What would have contradicted it: a 12-byte VOP3 at 1.00,
  or an 8-byte instruction with a literal above 1.00.

(b) Code size. Raw cycles per instruction (loop control included) against the loop body's size, R = 2 to 64
blocks per iteration (`tim_core size`). Loop control per iteration = raw cycles - instructions x their own cost:

| Body | 1 KB | 2 KB | 3 KB | 3.5 KB | 4 KB | 5 KB | 6 KB | 8 KB |
|---|---|---|---|---|---|---|---|---|
| 8-byte `v_fma_f32`: loop control per iteration (cycles) | 21 | 21 | 21 | 21 | 34 | 34 | 43 / 197 (two runs) | about 800, bimodal |
| 4-byte `v_add_f32`, `s_nop` | 20 (0.5 KB) | 20 | 20 (3 KB) | - | 33 | - | - | - |

| 12-byte `v_fma_f32` with a literal | 1.5 KB | 3 KB | 3.8 KB | 4.6 KB | 5.4 KB | 7.7 KB | 9.2 KB | 12.3 KB |
|---|---|---|---|---|---|---|---|---|
| cycles per instruction, loop control included (two runs) | 1.412 | 1.335 | 1.319 | 1.43-1.56 | 1.59-1.70 | 1.46-1.52 | 2.14-2.57 | 2.58-2.63 |

- Up to just below 4 KB of loop body the loop control costs a constant 20 to 21 cycles per iteration. From 4 KB on
  the loop costs more: 34 cycles of control per iteration at 4 and 5 KB, and beyond that it varies between runs
  (6 KB: 43 in one run, 197 in the other) and becomes bimodal; from about 8 KB (8-byte instructions) or 9 KB (12-byte)
  the loop runs at 1.8 and 2.1 to 2.6 cycles per instruction. INFERRED: a loop of up to 4 KB is served from a buffer
  close to the wave (the instruction cache line is 128 bytes, READ), and larger loops are fetched again every
  iteration, partly hidden; the 8 KB step may be a second level. Which structures these are is not established.
- This probably accounts for most of the first-hour calibration loop's 41 cycles of control per iteration (item
  1): its body is 512 FMAs of 8 bytes, 4 KB, and the sweep here shows 34 cycles of loop control at 4 KB. For the study's probes: keep a timed loop body below 4 KB.

(c) Independent scalar instructions at 1.19 cycles: the cost follows the SGPR banks of the two sources.

| Block (16 `s_add_co_i32`, destinations s16-s31) | Cycles | Predicted |
|---|---|---|
| sR = sR + 1 (inline constant), sR = sR + literal, `s_mov`, `s_nop`, `s_cvt_f32_i32` | 1.000 | 1.000 |
| sR = sR + s40 / + s42 / + s44 | 1.125 / 1.125 / 1.125 | 1.125 |
| sR = s40 + s41 / s40 + s56 / s40 + s40 | 2.000 / 2.016 / 2.000 | 2.000 |
| sR = s40 + s42 / s40 + s44 / s40 + s48 / s41 + s43 | 1.000 each | 1.000 |

- Model: SGPRs sit in 8 banks of 64-bit pairs (bank = (register >> 1) mod 8, so 16 registers per round); a scalar
  instruction whose two sources are in one bank costs one extra cycle, even when it is the same register twice.
  Every one of the 11 blocks matches its prediction within 0.02 (verdict `tim_core_sgpr_banks`). The decisive pair:
  s40 + s48 (pairs 20 and 24) costs nothing and s40 + s56 (pairs 20 and 28) costs a cycle, so there are 8 banks of
  pairs, not 4. Lane 1's 1.19 (19 per 16) is three such conflicts per 16 instructions with the registers its
  allocator chose (INFERRED; its listing was not re-read). A scalar instruction with one SGPR source and an inline
  constant or literal costs 1.00. What would have contradicted it: s40 + s48 at 2.0, or s40 + s42 at 2.0.

(d) Independent pseudo-scalar operations at 1.88 cycles: the cost belongs to VALU instructions that write an SGPR,
issued back to back.

| Block (16) | Cycles |
|---|---|
| `v_s_exp_f32` sR, s40 / with an inline-constant source / 8 destinations reused | 1.875 / 1.875 / 1.875 |
| `v_readfirstlane_b32` sR, vR | 1.875 |
| `v_cmp_gt_f32_e64` sR, vR, 0.5 | 1.625 |
| `v_exp_f32` (vector, for comparison) | 4.000 |
| `v_s_exp_f32` + `v_fma_f32` (per pair) / + `s_nop 0` (per pair) | 2.000 / 2.000 |
| `v_s_exp_f32` + `s_add_co_i32` on another SGPR (per pair) | 25.94 |

- So it is neither the transcendental unit (the vector exp issues every 4 cycles, the pseudo-scalar one at 1.875)
  nor the source read (an inline constant source costs the same): two SGPR-writing VALU instructions in a row cost
  about 2 cycles each, and any instruction between (an FMA, even `s_nop 0`) absorbs the extra cycle. INFERRED: one
  VALU-to-SGPR write path that accepts a write every second cycle; the fractions (1.875, 1.625) are not explained.
- New and costly: a scalar ALU instruction right after a VALU instruction that wrote an SGPR waits about 24 cycles,
  even when it touches a different SGPR (the hardware waits for every outstanding VALU write of an SGPR before a
  SALU instruction, INFERRED). Keep scalar ALU work away from `v_readfirstlane` and `v_s_*` results.

(e) Back-to-back FMAs writing one register: 2, 4, 8 and 16 consecutive `v_fma_f32` with the same destination and
independent sources cost 1.0000 each (verdict `tim_core_waw`), so writing one register repeatedly costs nothing. A
first version of the 16-write block whose first instruction read the register written by the previous block's last
one cost 1.25 (4 cycles per 16: the 5-cycle latency minus one): lane 1's non-linear timings came from such a
dependency, INFERRED (its blocks were not rebuilt).

(f) Dependent chains behind `s_delay_alu` stopping at 0.80 operations per cycle with 5 or more waves: explained by
oldest-first issue and a tail.

| Waves per SIMD (CU mode, one per wave: a single FMA chain, `s_delay_alu instid0(VALU_DEP_1)` before each) | 1 | 2 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| Cycles per FMA of the SIMD (span of its waves) | 5.000 | 2.500 | 1.250 | 1.000 | 1.676 | 1.254 |
| Model: groups of up to 5 waves, each group 5 cycles per FMA | 5.000 | 2.500 | 1.250 | 1.000 | 1.667 | 1.250 |
| Wave end times in units of one wave alone (one launch) | | | 1.04 x4 | 1.04 x5 | 1.04 x5, 1.97 | 1.04 x5, 1.97, 2.02, 2.05 |

- Five waves fill the SIMD exactly (1.000 cycles per FMA: the FMA latency is 5 cycles). With more waves five of them
  finish together and the rest run afterwards as a tail with fewer than five waves (the probe records end times
  without the wave's identity, so that the five are the oldest is INFERRED from the oldest-first order seen
  elsewhere).
  So the SIMD's rate while five or more waves are active is the full 1.0; 0.80 is the average over the whole run,
  set by the tail. The same holds for `v_add_f32`, with an `s_nop` after each FMA, and for independent FMAs each
  behind the hint (all within 0.01 of the model). With two chains per wave (`VALU_DEP_2`) one wave gives 0.40, two 0.80,
  and the tail again keeps the average below 1.0 (0.87 to 0.91 with 5 to 8 waves). Verdict `tim_core_delay_tail`:
  measured within 3 % of the model at every wave count (0.6 % worst). What would have contradicted it: 6 waves at
  1.25 (a fixed 0.80 limit) or the waves finishing together.
- Lane 1's 16-wave figure (1.22) fits the same model (four groups: 20 x one wave's time for 16 waves' work, 1.25).

(g) Register banks with several waves (compiler-and-hazards item 3's 0.72): an FMA reading three different
registers of one bank costs 5.0 cycles and so does the SIMD with 8 waves (5.0000): other waves do not fill those
cycles. VOPD FMAC pairs with per-lane sources: 1.5 cycles per pair when the six reads spread over the banks with at
most two per bank, 2.5 when three different registers of one bank are read across the pair, at 1 and at 8 waves
(verdict `tim_core_vopd_banks`); with sources shared by every pair (`du_fmac`) 1.0. Other per-lane FMA shapes with
at most two registers per bank cost 1.00 to 1.02 at every wave count.

## 9. Disturbed samples and the clock ramp (timing leftovers, `tim_disturb`)

Lane 7 found that every SIMD of the card stops for about 1.1 us roughly every 2.8 ms, and that desktop drawing adds
longer events (`measurements/scheduling-and-clocks.md` item 7). Here, one wave, after 30 ms of the same work (past
the clock ramp), timed one calibration loop (256 independent FMAs and the loop control per iteration, 277 cycles)
over and over and kept every sample, with its start counter and REALTIME. Run `runs/20261117T175731Z-timing/`, MEASURED.

Two runs: `20261117T175038Z-timing` (desktop quiet) and `20261117T175731Z-timing` (the desktop drew more). "Holding a
stall": excess over the best sample of at least about 2,000 cycles.

| Loop | Samples (timed) | Median, all = after the rule | Worst, all / after the rule | Flagged by the rule | Holding a stall or more |
|---|---|---|---|---|---|
| short, 2 iterations (512 FMAs) | 200,000 (34 ms) | 551 / 551 | 3,975 / 737; 4,244 / 802 | 20 (0.010 %); 69 (0.035 %) | 14; 54 |
| medium, 40 iterations | 30,000 (103 ms) | 11,077 / 11,077 | 18,808 / 15,809; 18,281 / 16,677 | 4 (0.013 %); 20 (0.067 %) | 42 (0.14 %); 108 (0.36 %) |
| long, 4,000 iterations | 400 (138 ms) | 1,107,997 (= best) | 1,122,595 (+1.3 %); 1,127,421 (+1.8 %) | 0; 0 | 52 (13 %): 49 about one stall; 62 (16 %): 11 one, 51 two to five or more |

- In the quiet run the disturbances are the regular stall lane 7 saw: one per 2.7 to 2.8 ms of timed work (14 in 34
  ms, 42 in 103 ms, 52 in 138 ms; flagged short samples spaced 2.76 ms apart), each adding about 3,000 to 3,900
  cycles. A sample of C cycles holds one with a chance of about C / 9,000,000 at 3.2 GHz: 0.006 % at 551, 0.12 % at
  11,077, 12 % at 1.1 million (measured 0.007 %, 0.14 %, 13 %). In the second run desktop drawing added events on top
  (2 to 5 stalls' worth in the long samples, flagged short samples in clusters 0 ms apart), two to three times as many.
- The rule (above 1.5 x median + 64 cycles) catches a stall only when the sample is shorter than about 2 x the stall:
  in the medium loop it flagged 4 of 42 (20 of 108) disturbed samples and in the long loop none. In the short loop it
  caught every whole stall, and also samples holding part of one (excess 400 to 2,000).
- The median never moved (verdict `tim_disturb_median`: equal with and without the rule in all three loops, and the
  long loop's median equals its best sample). The worst sample and the mean do move.
- Clock ramp: after 0.5 s of idle and one warm-up launch (which took the climb from the idle clock, about 0.1 GHz,
  to 2.79 GHz: that part is not covered here), one wave timed chunks of 4,096 FMAs from the first cycle: the clock
  climbed from 2.786 GHz (first 9 ms) to 3.238 GHz (16 ms), and every 0.5 ms bin had the same median 1.0813 cycles per FMA (loop
  control included), during the ramp and after it (verdict `tim_ramp_cycles`). So between 2.79 and 3.24 GHz the ramp changes how long a cycle
  is, not how many cycles an instruction takes: a cycle-counter measurement made during that part of the ramp is
  not disturbed. The first climb, from the idle clock to 2.79 GHz, was not covered.
  (Bins with a stall show a worst chunk of 1.70 to 1.86 cycles per FMA, every 2.8 ms, during the ramp as after it.)
- What this settles from earlier lanes (each region size from the probe's source, the excess from its log):

| Earlier oddity | Timed region | Excess | One regular stall would add | Verdict |
|---|---|---|---|---|
| `lds_stage feed` calibration 17.66 against 16.01 cycles, one launch (lane 3 rerun) | 6 repetitions of 4,096 FP16 matrix instructions, 393,216 cycles per launch | 40,550 cycles (14.6 us at 2.77 GHz) | 3,900 cycles, with a chance of 6 % in 157 us | ten stalls' worth, so not the regular stall; INFERRED: a desktop event or a desktop wave on the SIMD, not the regular stall; probably not the clock (the job did start after idle, sclk 81 MHz, and the clock changed between launches; cycles per instruction did not change between 2.79 and 3.24 GHz, but the climb from the idle clock was not tested) |
| `cvt_time`: 3 of 10 repeats failed calibration or linearity; worst FMA-chain sample 12.454 against 5.000 | one repetition at U = 8: 16,384 patterns, 82,000 cycles for a 5-cycle chain | up to 61,000 cycles in one sample; the 5.101 median meant about 800 cycles across many samples | 3,900, chance 0.9 % per sample | 16 stalls' worth in one sample: desktop drawing (the driver showed the card 17 to 45 % busy with other programs in those repeats); its checks used the extreme samples, which fail at that sample size (above) |
| `mat_share` 8-wave FP16 row 16.98 (16.04 to 18.29) | 8 waves x 8 repetitions x 65,536 cycles = 4.2 million cycles per launch (1.3 ms) | about 246,000 cycles per launch (77 us) | 0.5 stalls per launch, 1,800 cycles | 130 times the regular stall, so not the regular stall; INFERRED: another process's waves on that SIMD (lane 7 saw them hold a SIMD for a whole launch); probably not the clock (as in the first row) |

- Practical rule, added to `notes/STUDY-PLAN.md` ("Rules for every lane"): judge single samples only for regions
  below about 5,000 cycles; above that, quote the median of at least 5 samples (or the best for a minimum), and
  never base a verdict on the extreme samples of regions above about 50,000 cycles.
- Also found here: the loop body of the first version (4 KB) cost 1.0667 cycles per FMA; the loop-size step of item
  8b. The probe uses a 2 KB body.

## Method

- Harness: `src/harness.h` (arch guard, statistics, the cycle-counter read). On gfx12 `__builtin_readcyclecounter`
  reads `SHADER_CYCLES_HI`, `_LO`, `_HI` and drops the low word when the high word changed between the reads.
- Loops (`src/study/core_common.h`): one inline-assembly block of 16 "patterns" (one instruction or a fixed short
  group) over c = 1, 2, 4, 8 or 16 chains. Operands are `"+v"`/`"+s"` constraints (or fixed `{vN}` registers in
  `core_dual`), so the compiler chooses registers only; it cannot delete, merge, reorder or pair the instructions. The
  constants are pinned in registers before the loop with an empty `asm volatile`. Each test is built at U = 4, 6 and
  8 blocks per loop iteration, 128 iterations, 8 timed repetitions per launch (the first dropped), 3 launches.
  Cost per pattern = (cycles(U=8) - cycles(U=4)) / (4 x 128 x 16): the loop control, counter reads and pipeline drain
  cancel. U = 6 must land midway within 1.5 cycles per iteration or the test is reported non-linear and the verdict
  fails. U = 1 was avoided: measured, its loop costs one cycle per iteration more than the U >= 3 loops, which would
  read as 0.97 cycles per FMA.
- Assembly check: every probe is built with `-save-temps`; `scripts/core_loopcheck.py` reads the `.lto.s` and, for
  each timed loop, lists the instructions the compiler added outside the assembly blocks. Only loop control is
  allowed, and that list must be identical at U = 4, 6 and 8 (otherwise it would not cancel). In the run folder,
  `loopcheck-<probe>.txt` lists every loop with the opcode counts of its assembly blocks. The compiler interleaves
  its loop control (`s_add_co_i32`, `s_wait_alu`, `s_cmp`, `s_cbranch`) between the blocks, once per iteration.
- Compiler-built loops: `c_fmac` in `core_dual` (plain C, dual issue on) is not checked by the script; its loop holds
  32, 48 and 64 `v_dual_fmac_f32` at U = 4, 6, 8 and the loop control, nothing else (read in the `.lto.s`).
- Build flags: `-O3 --offload-arch=gfx1201`; `core_valu`, `core_scalar` and `core_flow` with
  `-mllvm -amdgpu-enable-vopd=0` (as the first hour) so the compiler-built calibration loop is not paired;
  `core_share` with and without `-mcumode`.
- Checks that can fail: each probe prints `VERDICT <name>: PASS|FAIL`. Known answers: the calibration (item 1), the
  dependent and independent FMA (5.00 and 1.00) in `core_scalar` and `core_flow`, the dual and split rates in
  `core_dual`, the single-wave issue rates as the bound for several waves in `core_share`, REALTIME at 100 MHz against
  the host clock in `core_counter`. Deliberate breaks (`--break`, all run by the script, all must FAIL):

| Probe and break | Verdict |
|---|---|
| `core_valu --break` | FAIL as intended. VERDICT core_calibration: FAIL first-hour loop 1.080 cycles/FMA (known 1.08), asm differential 5.000 (expect 1.00)  |
| `core_dual --break` | FAIL as intended. VERDICT core_dual: FAIL 1 patterns with an expected rate, 1 mismatches or non-linear loops: du_fmac  |
| `core_share_cu --break` | FAIL as intended. VERDICT core_share_cu: FAIL ind_fma simds 1 waves 2: 0.999 SIMD cycles/op, expected 0.48 to 0.53; (rerun `runs/20261117T133209Z-core/`) |
| `core_flow --break` | FAIL as intended. VERDICT core_flow: FAIL independent FMA 1.000 (expect 2.00 [--break]), dependent FMA 5.000 (expect 5.00 within 4.9-5.1), 1 of 24 tests not linear in block count  |
| `core_counter --break` | FAIL as intended. VERDICT core_counter: FAIL REALTIME runs at 99.989921 MHz against host time, not within 2% of 50.000000 MHz  |

## How to rerun

```
scripts/study-core.sh            # build, run, breaks: about 2 minutes of building, about 10 s of GPU time
scripts/study-core.sh build      # build and loop checks only (no GPU)
```

It needs the card record `runs/latest/card.env` (`scripts/first-hour.sh info`), ROCm 10.1 in `~/.local/opt/rocm`,
python3. Results in `runs/<UTC time>-core/summary.md`; one log per probe with the clocks before and after. Each GPU
job takes the shared GPU lock file (`STUDY_GPU_LOCK` in `local.env`; default `runs/gpu.lock`), runs under `nice`, uses the card
record's device index and does not start below 8 GB of available memory.

## Not measured, or not explained

Still open:

- Explained or narrowed in item 8 (timing leftovers): the 1.23-cycle literal VOPD pairs (instruction length), the
  1.19-cycle scalar instructions (SGPR banks), the 1.88-cycle pseudo-scalar operations (back-to-back VALU writes of
  SGPRs), the 0.80 limit with `s_delay_alu` (oldest-first tail), the back-to-back writes of one register (no cost).
  Still not explained: the mechanism behind the 5-cycle bank case (it holds with 8 waves too), the exact rule for
  runs of 12-byte instructions, the structures behind the 4 KB and 8 KB loop-size steps, and the fractions 1.875 and
  1.625 of the SGPR-writing VALU instructions. Where leftovers C went further, and what stays open:
  - 5-cycle bank case. Answered in part (leftovers C, EC-5, `measurements/leftovers-c-a.md`): with a different
    triple of bank-1 registers each time a period costs max(k + 2, 5) cycles (k other FMAs between): the instruction
    costs one cycle more and the next one of its kind cannot start until 5 cycles after it, and ordinary instructions
    fill those cycles; with the same triple repeated there is no cost once anything sits between. So it is a shared
    limit, not a fixed stall of each instruction (a recovery time of the bank or a small operand cache that the
    repeated triple hits: INFERRED). MEASURED. Open: the mechanism; why the same triple repeated is free; the
    8-waves-per-SIMD case was not run.
  - Runs of 12-byte instructions. Still open (leftovers C, EC-6): the rule. From runs of 4 on, the cost falls smoothly
    as the gap of 8-byte instructions grows, as a fetch limit a little under 12 bytes per cycle predicts, but for
    short runs it is not monotonic (one 12-byte instruction among 16 short ones costs 1.235 per instruction, more
    than runs of 4 among 16, 1.049). A fetch-rate model with a buffer (largest error 0.235 cycles per instruction)
    and a boundary-crossing model (0.16) both fail on the 42 points; the table is in `measurements/leftovers-c-a.md`.
  - The 4 KB and 8 KB steps. Answered in part (leftovers C, EC-7, `measurements/leftovers-c-a.md`): the 4 KB step is
    not a fixed size limit: a 4 KB loop costs 34 cycles of loop control when it starts at a 4 KB boundary (to 128
    bytes past it) and 127 or 236 when it starts elsewhere (one wave; INFERRED: a structure indexed by address).
    Two different 3 KB loops on the two SIMDs of a compute unit slowed each other (30 against 20 to 21 cycles of loop
    control) in one build and not in another whose loops sat 68 bytes further on, and across the two compute units
    of a work-group processor the slow side follows the code copy, not the compute unit (test T2). MEASURED. Open: the
    size and organisation of the structure; why one copy loses and the other does not; the 8 KB step was not looked
    at.
  - The fractions 1.875 and 1.625. Answered in part (leftovers C, EC-8, `measurements/leftovers-c-a.md`): the cost is
    a throughput: the path from the VALU to the SGPRs takes 8 writes per 15 cycles (`v_readfirstlane_b32`,
    `v_s_exp_f32`) or 8 per 13 (`v_cmp` to an SGPR), and independent FMAs fill the idle cycles, so a period of n such
    instructions and m FMAs costs max(n + m, 1.875 n) (1.625 n for compares); 57 of 63 points within 0.05 cycles.
    MEASURED (rates), INFERRED (the write-path reading). Open: the six misses, all runs of 3 with fillers, and runs of
    3 without fillers that are not linear in the block count.
- `exec` cannot be declared as clobbered in inline assembly (the compiler reserves it), so the control-flow patterns
  restore EXEC themselves; the timed loops run with all 32 lanes on. Not reachable (leftovers C plan, EC-9): the
  compiler reserves it.
- 64-bit integer add and `v_div_scale` timing, what is left: the pair with a VCC carry and back-to-back
  `v_div_scale_f32` are not linear in the number of blocks (two increments reported), and the cost of the compiler's
  full division sequence was not timed.
  Answered in part (leftovers C, EC-2d and EC-2e, `measurements/leftovers-c-a.md`): see the lines below.
- New from leftovers C: why `v_sqrt_f64` gives only about 26 correct bits (`sqrt(1.0)` returned 1.0000000149; T3,
  `measurements/leftovers-c-a.md`).
- New from leftovers C: why a block in which every instruction reads and writes register bank 0 costs 1.343 cycles per
  instruction in both wave sizes, while reading or writing bank 0 alone costs 1.000 (T1,
  `measurements/leftovers-c-a.md`).
- New from leftovers C: why a loop of 2 KB costs a single wave 198 cycles of loop control when copies of the loop are
  aligned at the same offset inside 4 KB, even when only one copy runs (EC-7, `measurements/leftovers-c-a.md`).

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, EC-1, `measurements/leftovers-c-a.md`): `v_permlane64_b32` in wave32 is a full-rate no-op: 1.000
  cycle per instruction, result 1 cycle later, all 32 lanes unchanged. MEASURED.
- Answered (leftovers C, EC-2a, `measurements/leftovers-c-a.md`): the FP16 pseudo-scalar forms (`v_s_rcp_f16`,
  `v_s_rsq_f16`, `v_s_sqrt_f16`, `v_s_log_f16`, `v_s_exp_f16`) cost exactly their FP32 twins: 1.875 cycles back to back
  and 19.00 through a chain. MEASURED.
- Answered (leftovers C, EC-2b, `measurements/leftovers-c-a.md`): `v_sqrt_f64` and `v_rsq_f64` cost what `v_rcp_f64`
  costs: one every 32.00 cycles, result after 40.00. The costs do not depend on the data (zero, infinity, NaN, the
  smallest subnormal, 1.0 and 1.7: 29 blocks, six data sets, test T3; no positive control for a data-dependent cost
  was run). MEASURED.
- Answered (leftovers C, EC-2c, `measurements/leftovers-c-a.md`): `v_dot2_f16_f16` and `v_dot2_bf16_bf16` cost what
  the FP32-result dot product costs: 1.000 cycle per instruction, 6.000 through a chain. MEASURED.
- Answered in part (leftovers C, EC-2d, `measurements/leftovers-c-a.md`): there is no single 64-bit add on gfx1201
  (`v_add_nc_u64` and `v_lshl_add_u64` are refused by the assembler: READ). The `v_add_co_u32` + `v_add_co_ci_u32`
  pair costs 5 cycles through one chain, 4.5 to 4.9 per pair for eight independent pairs sharing VCC, and 3.5 per
  pair with each pair's carry in its own SGPR. MEASURED; the VCC form has no single per-instruction cost.
- Answered in part (leftovers C, EC-2e, `measurements/leftovers-c-a.md`): `v_div_fmas_f32` costs 1.000 cycle like an
  FMA; `v_div_scale_f32` 2.1 to 2.4 back to back (not linear); a `v_div_scale` directly followed by a `v_div_fmas`
  that reads its VCC costs 17 cycles per pair. The compiler's full division sequence gave the correctly rounded
  quotient in 32 of 32 lanes. MEASURED.
- Answered (leftovers C, EC-3, `measurements/leftovers-c-a.md`): when a VOPD pair breaks the guide's rule, both halves
  still read their sources before either writes: the second half reads the old value in 32,000 of 32,000 lane results
  in each of the five forbidden shapes, the same as the allowed reverse case (the same two operations as two
  instructions give the new value). Write-write to one register cannot be encoded. The guide does not promise it, so
  a later chip may differ. MEASURED.
- Answered in scheduling-and-clocks.md, item 4, and since then by leftovers C (EC-4, SC-10,
  `measurements/leftovers-c-a.md`): wave64 under HIP: independent `v_fma_f32` ran at 0.81 instructions per cycle
  (51.85 lane-FMAs per cycle), the chained matrix instruction at 16.000 cycles as in wave32. The 0.81 came from the
  register layout (measured cause: half the FMAs read two registers of one bank): the same kernel with only its
  registers moved ran at exactly 1.000 (64 lane-FMAs per cycle). MEASURED.

- Helper work: `core_flow.hip` and `core_counter.hip` were written by Sonnet 5.5 helpers from sealed briefs; their
  code was read, two comments and one plausibility bound were corrected, and every number here comes from the
  rerun in the run folder above, not from the helpers' runs.
