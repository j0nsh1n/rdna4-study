# Compiler and hazards on gfx12 (lane 8)

What ROCm 10.1 `hipcc` (AMD clang 24.0.0git, d6f6cb69) does to kernel code for the RX 9070 XT (gfx1201, wave32):
where it protects the hardware's rules, what that costs, and where it produces slow or wrong code. Run folder:
`runs/20261117T160358Z-haz/` (LACT profile "Testing", hotspot 50 C before and after every job). Labels: MEASURED (seen
in that run), READ (from a document or a listing), INFERRED (reasoning). Cycles are the wave's shader cycle counter,
one wave alone on its SIMD unless a row says otherwise.

This lane was written in a staging folder outside the repository and copied in unchanged by the lead, who reran
the whole script from the repository (`runs/20261117T161110Z-haz/`): every check has the same status as in the
owner's run (136 distinct check-and-status pairs over 482 verdict lines: 78 PASS, 21 INFO, 37 FAIL; every FAIL is
in a deliberate-break log; the script reports 0 unexpected outcomes). The checks of known wrong behaviour
(`hazard_call_miss`, the masked matrix kernels, the lane permutes) print PASS or INFO: a PASS there means "the
behaviour described here is still present", not "the code is right". What an identical status does not show: the
wrong-result counts of the masked kernels are INFO lines and moved between the two runs (section 1 gives both).
Timing figures agree within 3 % except twelve multi-wave rows that moved by up to 11 % (the desktop was using the
card; see the note in `measurements/conversions-and-dots.md`); where the two runs differ the tables below give both. One deviation the owner reported after its final message: a helper wrote
one scratch file to `/tmp/x` and deleted it at once, against the study's no-/tmp rule; nothing depends on it.

## Summary

**What the compiler protects you from.** Every source shape tried that feeds a matrix result into the next matrix
instruction's A or B gets the required separation inside one function: a V_NOP, or vector copies (straight line, as B,
across a loop back-edge, across basic blocks, after inlining, FP16 bits read as BF16). The scalar-register read hazard
gets its `s_wait_alu` waits. All of these computed right on the card against the CPU model.

**What it does not protect you from.**
1. *A matrix instruction ignores the lane mask (EXEC) completely* (MEASURED in wave32, 7 masks including EXEC = 0,
   for `v_wmma_f32_16x16x16_f16` here and, in the leftovers pass, for the FP8 x FP8, 8-bit integer, FP16-accumulator
   and sparse FP16 forms; the dot instructions beside them obey the mask; AMD's guide describes the rule for all
   matrix opcodes, READ): it reads
   every lane's operands and writes every lane's result, at the same 16.7 cycles (the timing was measured for the
   FP16 form only). The compiler does not know this. A
   matrix builtin inside a per-lane `if`, after an early `return`, or in a loop with a per-lane trip count runs under
   a partial mask; the switched-off lanes' operand registers then hold whatever was there (results wrong, and how
   many are wrong changes from run to run), and their result registers are overwritten (lanes that left a per-lane loop end with
   the result of the lanes that stayed: 128 of 128 values). Same failure class as gfx11, but now also proven at the
   hardware level.
2. *The matrix hazard fix is missing across a function call* (MEASURED): a non-inlined device function that ends with
   a matrix instruction and returns its result untouched, called by code whose next instruction is a matrix
   instruction using that result as A. No V_NOP on either side; 64 of 256 results wrong, bit-identical to the forbidden
   inline-assembly sequence, in both committed runs (two kernels each).
   Plain C++, no inline assembly needed, but it does need a callee that is not inlined and ends in a matrix
   instruction; llama.cpp's gfx12 kernel shows no such site (both scanners clean on its listing).
3. *A lane permute with the fetch-inactive bit inside a per-lane branch reads stale data* (MEASURED; of the 5 mask
   shapes, 3 per permute leave lanes whose partner is switched off, and all of those lanes show it):
   the hardware does fetch switched-off lanes, but clang computes the permute's source value inside the branch, so a
   switched-off partner's register holds something else (100 % of such lanes wrong, 0 % once the value is pinned
   before the branch).

**Pitfalls gone since gfx11** (same kind of source, gfx12 listing): 32-bit bit tricks feeding FP16 vectors stay
32-bit (no 16-bit splitting); LDS words copied into an FP16 vector by `memcpy` become one 128-bit LDS load, as does
`__builtin_bit_cast`; 16-byte global reads written any natural way become one 128-bit load (no need to mask low address
bits; masking turns them into flat loads, which is slower for one wave and no faster for the whole card). **Pitfalls that remain:** few FMAs are paired for dual issue when operands are uniform
(scalar registers block pairing and halve the FMA rate); `__builtin_nontemporal_load` still rejects HIP's `uint4` (seen by the owner and a helper as a compile error; no failing build is kept in the run folder);
register caps and large tiles spill, and spilling costs 5 to 7 times per loop step (4x8 tiles 3202 cycles against 609 for 4x6; a 16-wave register cap 3357 against 476; owner's run).

**Checklist for a matrix kernel on gfx12.**
- Never put a matrix builtin under a per-lane condition. Make loop bounds and early exits wave-uniform
  (`__builtin_amdgcn_readfirstlane`, or a bound that only depends on `blockIdx`); pad partial tiles with a select, not
  a branch. Run `scripts/haz_maskscan.py` on the listing; it follows the mask within one function only, so a
  matrix instruction in a non-inlined function that is CALLED under a partial mask is not reported (the reviewer
  showed this with a synthetic listing): keep matrix code inlined or check call sites by hand.
- Keep the matrix loop in one function (`__forceinline__`), or put a VALU instruction after a call that returns a
  matrix result. Run `scripts/haz_hazscan.py` (a heuristic: it flags every RETURN after a matrix instruction and
  every matrix instruction right after a CALL, whether or not the result is used).
- For a lane permute with fetch-inactive in masked code, compute its source before the branch and pin it
  (`asm volatile("" : "+v"(x))`), or do the permute outside the branch.
- Keep uniform FMA operands in vector registers if the loop is FMA-bound (one `asm volatile("" : "+v"(k))` before the
  loop; INFERRED, that exact source was not built): the measured pair is 0.47 FMAs per cycle per SIMD with uniform
  operands in scalar registers against 1.48 with per-lane operands loaded from memory, 8 waves.
- Accumulators decide registers: 8 per 16x16 FP32 tile. 4x6 tiles (239 VGPRs, 6 waves) is the largest that does not
  spill; do not cap with `amdgpu_waves_per_eu` below what the accumulators need.
- Do not mask pointer bits to get wide loads; it turns global loads into flat loads.

## 1. Matrix instructions under a lane mask

Source `src/study/haz_mask.hip`; scan `scripts/haz_maskscan.py` with `scripts/haz-expect/maskscan-haz_mask.txt`.

The hardware rule (MEASURED, inline assembly, EXEC set around one `v_wmma_f32_16x16x16_f16`, D preset to a sentinel):

| EXEC | results equal to the full-wave model | switched-off lanes' D written | cycles per chained instruction |
|---|---|---|---|
| all ones | 256 of 256 | (none off) | 16.70 |
| 0x0000FFFF, 0xFFFF0000, 0x55555555 | 256 of 256 | 128 of 128 | 16.70 |
| 0x11111111 | 256 of 256 | 192 of 192 | 16.70 |
| 0x00000001 | 256 of 256 | 248 of 248 | 16.70 |
| 0 | 256 of 256 | 256 of 256 | 16.70 |

So the instruction behaves as AMD's guide's pseudocode says (EXEC forced to all ones; READ): all lanes' operands are
read, all lanes' results written, no time difference, deterministic. 16.70 includes the two EXEC writes per 64.

Other opcodes (leftovers pass, MEASURED, `left_mat mask`, `runs/20261117T165116Z-leftmat/`): the same test (EXEC set around one
instruction in inline assembly, the 7 masks above, D preset to a sentinel, everything stored under full EXEC
afterwards, EXEC restored, each launch run twice).

| Instruction | Every result word equal to the full-EXEC result | Switched-off lanes' D words written | Repeat identical |
|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` (the known answer above) | 256 of 256 for every mask | all (128, 128, 128, 192, 248, 256 for the 6 partial masks) | yes |
| `v_wmma_f32_16x16x16_fp8_fp8` | 256 of 256 for every mask | all | yes |
| `v_wmma_i32_16x16x16_iu8 neg_lo:[1,1,0]` (llama.cpp's form) | 256 of 256 for every mask | all | yes |
| `v_wmma_f16_16x16x16_f16` (16-bit accumulator, 4 VGPRs) | 128 of 128 for every mask | all (64, 64, 64, 96, 124, 128) | yes |
| `v_swmmac_f32_16x16x32_f16` (sparse; D is C, so the untouched C is the sentinel) | 256 of 256 for every mask | all | yes |
| `v_dot4_f32_fp8_fp8` (control) | only the enabled lanes' words (16, 16, 16, 8, 1, 0) | none (0 of 16, 16, 16, 24, 31, 32) | yes |
| `v_dot2_f32_f16` (control) | only the enabled lanes' words | none | yes |

The dot instructions are ordinary vector instructions and obey EXEC, which shows the test can see a masked
instruction; the deliberate break (expecting the dot instructions to ignore the mask too) FAILs. The equality with
the full-EXEC result needs every lane's operands, so all five matrix opcodes read the switched-off lanes' A, B, C
and index registers. Not timed here (lane 8's timing covers the FP16 form only).

What clang emits and what the card computes (natural HIP, builtin `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`):

| Source shape | Emitted (READ) | Scan | On the card (MEASURED, against the CPU model) | Good form |
|---|---|---|---|---|
| operands loaded by all lanes, result stored under `if (lane < n)` | matrix instruction before `s_and_saveexec` (not sunk) | clean | right (0 wrong, n = 8..32) | (already good) |
| everything under `if (lane < n)` | loads and matrix instruction inside the `s_and_saveexec` region | 1 site | wrong: 64 of 64 (n = 8), 128 of 128 in the owner's run and 120 of 128 in the rerun (16), 144 of 160 (20), 23 of 248 (31); the count is unstable: over 3 launches it changed for n = 20 and 31 in the owner's run and for every n below 32 in the rerun | pad by select, all lanes run it |
| early `return` for lanes past the end | same as above | 1 site | wrong, same pattern (n = 20: 144 of 160 in the owner's run, 140 in the rerun), unstable | wave-uniform bound |
| loop with per-lane trip count (lanes 0-15 two trips, 16-31 one) | `v_cmpx` at loop entry, matrix instruction in the masked loop | 1 site | lanes 0-15 equal "stale operands of lanes 16-31" in 128 of 128; lanes 16-31 overwritten with that result in 128 of 128; repeatable | uniform trip count |
| padding by select `ok ? A[l] : 0` | branch around the load only; matrix instruction under full EXEC | clean | right (0 of 256) | (good form) |
| wave-uniform early return on `blockIdx` | scalar branch | clean | right | (good form) |

gfx11 comparison: on RDNA 3 the compiler moved a matrix chain under the store's mask (README, "Compiler pitfalls"); here
clang 24 did not sink the instruction into the store branch (first row), but the shapes where the source itself puts
the builtin under a lane condition remain wrong, and the per-lane loop shows a second effect (other lanes' results
clobbered) that a register-level model of the compiler cannot see.

llama.cpp (READ, compile only, `scripts/study-compiler.sh llama`): `mmq-instance-q8_0.cu` and `-q4_0.cu` for gfx1201
use `v_wmma_i32_16x16x16_iu8` (276 matrix instructions in the q8_0 listing); the scan flags 0 of them and the hazard
scan 0. The partial masks there (tile-edge and valid-row conditions, about 1,550 `s_and_saveexec`/`v_cmpx`; the reviewer's own build of the same kernel counted 1,557) guard only loads,
LDS stores and output stores (helper's reading of one kernel by hand; the scan agrees on both files; counts are
for the q8_0 listing).

## 2. Hazards the compiler fixes, and a missed one

Source `src/study/haz_hazard.hip`; scan `scripts/haz_hazscan.py` with `scripts/haz-expect/hazscan-haz_hazard.txt`.

| Source shape | Emitted between the two matrix instructions (READ) | On the card (MEASURED) |
|---|---|---|
| FP16 result as next A, straight line | `v_nop` + `s_delay_alu` | right (0 of 256) |
| result as B | `v_nop` | right |
| across a loop back-edge (`x = wmma(x, b, c)`) | 2 `v_dual_mov_b32` copies (no V_NOP) | right (n = 1, 2, 5) |
| across a wave-uniform branch | `v_nop` | right (both paths) |
| FP16 result bit-cast to BF16 A | `v_nop` | right |
| non-inlined callee, result copied to return registers | 2 `v_dual_mov` in the callee | right |
| non-inlined callee ending in the matrix instruction, caller's next instruction a matrix instruction | **nothing**: callee `v_wmma v[0:3] ...; s_setpc_b64`, caller `s_swappc_b64; v_wmma ... v[0:3] ...` | **wrong: 64 of 256**, equal bit for bit to the no-V_NOP inline assembly; both committed runs |
| inline assembly with V_NOP | (written) | right |
| inline assembly without V_NOP (the deliberate break) | (written) | wrong: 64 of 256 |

Cost of the fix (READ from lane 4, `measurements/matrix-timing.md` item 5, same card): V_NOP form 19 cycles against 16
for the plain chain (FP16 into FP32); clang's copy form for a 16-bit accumulator used as A 26.5 against 20 for the
V_NOP form.

The scalar-register read hazard (a scalar value read by VALU, rewritten by SALU, read again):

| Loop | Waits in the loop (READ) | Cycles per iteration (MEASURED) | Right |
|---|---|---|---|
| clang | 2 `s_wait_alu depctr_sa_sdst(0)` | 47.0 | yes |
| inline assembly, one wait | 1 | 42.0 | yes |
| inline assembly, no wait | 0 | 33.0 | yes (the hazard did not show in this sequence) |

So one wait costs about 9 cycles here; the case without it still computed right, so I cannot say the wait was needed
in this exact sequence (INFERRED: the SALU-to-VALU distance was long enough).

`v_cmpx` followed by `v_permlane16`: the closest plain-C++ shape (a per-lane trip count with the permute first in the
loop) produced `v_cmpx`, then `s_cbranch_execz`, a scalar move, and a vector add before the permute (READ); with a VALU
in between there is no hazard to fix, and the result was right (n = 7, 16, 31). A direct `v_cmpx`-permute pair was not
reached from source; LLVM's fix for it (a `v_mov`) is READ only (`notes/research-10-compilers.md` section 2).

gfx11 comparison: the V_NOP rule and the permute rule exist on gfx11 too; the scalar-register wait is new on gfx12
(READ). The call-boundary gap was not looked for on gfx11.

## 3. `s_delay_alu` and waits

Source `src/study/haz_issue.hip`, built four ways (default; `-mllvm --amdgpu-enable-delay-alu=0`; `-mllvm
--amdgpu-enable-vopd=0`; `-mllvm --amdgpu-expert-scheduling-mode`). 16 operations per loop iteration. "Per SIMD"
divides by the number of waves per SIMD, assuming the work-group spreads evenly over the 4 SIMDs of one WGP as lane 1
measured (INFERRED for this probe; placement not recorded here).

Cycles per operation for the slowest wave / operations per cycle per SIMD (MEASURED):

| Loop | Where clang puts `s_delay_alu` (READ) | default, 1 wave | no delay, 1 wave | default, 8 waves/SIMD | no delay, 8 waves/SIMD | expert mode, 8 waves/SIMD |
|---|---|---|---|---|---|---|
| one dependent FMA chain | before every second FMA (9 per 16) | 5.63 | 5.00 | 0.49 | 0.23 | 0.48 |
| four interleaved chains | 2 per 16 (loop length difference between the default and the no-delay build) | 2.38 | 2.31 | 0.50 | 0.50 | 0.50 |
| 8 accumulators, per-lane operands, dual issue | 2 per 8 pairs | 1.88 | 1.75 | 1.48 (1.49 in the rerun) | 1.54 (1.64) | 1.58 (1.58) |
| 8 accumulators, own per-lane sources | 3 per 16 | 2.06 | 1.81 | 0.72 | 0.79 | 0.72 |
| 16 loads folded into one chain | 7 per 16 | 13.1 | 13.2 | 0.37 (0.35 in the rerun) | 0.20 (0.20) | 0.40 (0.36) |
| 16 chained matrix instructions | 2 | 15.9 | 16.0 | 0.060 | 0.062 | 0.061 |

- One wave alone: every `s_delay_alu` in front of a dependent instruction costs a little (chain: 5.63 against 5.00
  cycles, 12 %; independent sources: 14 %). Without other waves there is nothing to switch to.
- Several waves: the delay hint is what lets the SIMD switch waves at a dependency. Without it, a dependent chain runs
  at 0.23 instead of 0.49 operations per cycle per SIMD and the load-fed loop at 0.20 instead of 0.37. Where the
  instructions are independent anyway the hint costs 0 to about 12 % (four chains 0; paired accumulators 4 % in the
  owner's run and 10 % in the rerun; own sources 11 to 12 %).
- The switch exists: `-mllvm --amdgpu-enable-delay-alu=0` removes every `s_delay_alu` (listing check). A kernel author
  would want it only for a kernel that always runs one wave per SIMD (or has no dependent VALU chains), and the gain
  is at most about 12 %.
- Expert scheduling mode (`-mllvm --amdgpu-expert-scheduling-mode`): it compiles, writes `HW_REG_WAVE_SCHED_MODE` = 2 at
  kernel entry, replaces some delays with `s_wait_alu depctr_va_vdst/vm_vsrc` waits, and every result was right. No
  gain that reproduces: the load-fed loop at 8 waves showed 0.40 against 0.37 in the owner's run and 0.36 against
  0.35 in the rerun; tested on these kernels only.
- Matrix chains: one instruction per 16 cycles per SIMD whatever the number of waves or the switches.
- Per-lane FMA sources at 0.72 per cycle per SIMD (timing leftovers, `measurements/execution-core.md` item 8g,
  run `runs/20261117T175731Z-timing/`): register banks, now isolated. A VOPD FMAC pair that reads three different registers
  of one bank across its two halves costs 2.5 cycles, one whose six reads spread over the banks 1.5, at 1 and at 8
  waves per SIMD (MEASURED; other waves do not fill those cycles). clang's loop for this case (READ, the `.lto.s` of
  the default build) has 7 VOPD pairs, 5 of them with such a triple (for example `v_dual_fmac_f32 v2, v10, v17 ::
  v_dual_fmac_f32 v3, v11, v18` reads v2, v10 and v18 from bank 2), plus 2 single FMACs and the loop control. With the
  measured costs: 5 x 2.5 + 2 x 1.5 + 2 x 1 = 17.5 cycles per 16 FMAs, plus about 3 for `s_add`, `s_cmp` and the
  branch, gives 0.78 FMAs per cycle against 0.79 measured without `s_delay_alu`; the default build's 3 delay hints
  per iteration bring it to the measured 0.72 (INFERRED; the loop itself was not rerun). LLVM checks VOPD bank rules
  only within each source slot (src0 against src0, src1 against src1), not this triple across the slots.
- Waits seen: `s_wait_loadcnt` per load in a load-fed loop (16 per 16), `s_wait_alu depctr_va_vcc(0)` after
  carry-producing address arithmetic, `s_wait_alu depctr_sa_sdst(0)` around scalar values reused by VALU (section 2),
  and a `s_wait_loadcnt 0x0` that clang left inside the scalar-hazard loop although only the first iteration needs it.

gfx11 comparison: the same `s_delay_alu` pass serves both (READ, research-10 section 6); the measurements above are
gfx12 only.

## 4. The RDNA 3 pitfalls, retested

Source `src/study/haz_pitfalls.hip` (a, b, f), `haz_issue.hip` (c), `haz_memshape.hip` (d), `haz_regs.hip` (e).

| Pitfall | gfx11 (RDNA 3 README) | gfx12 (this card) | Card result | Good code |
|---|---|---|---|---|
| (a) E4M3-to-FP16 shift trick feeding an FP16 vector | split into 16-bit ops (8.4 ALU per weight instead of 2.6) | GONE: per-half construction, `__builtin_bit_cast` and the asm barrier all give 14 32-bit bit operations per 8 weights (1.75 per weight), no 16-bit op (READ, listing check) | right, 6 chained matrix instructions against the model | any |
| (b) LDS words into an FP16 vector by `memcpy` | 784 vector instructions per 16 matrix instructions | GONE: `memcpy` via word array, `memcpy` from LDS and `bit_cast` all give one `ds_load_b128` per fragment | right | any |
| (c) dual issue in straightforward FMA code | few pairs | REMAINS, different cause: uniform `k`, `c` in scalar registers give `v_fmac_f32_e64 v, s4, s5`, never paired, 3.19 cycles per FMA alone, 0.47 per cycle per SIMD at 8 waves; per-lane operands: all 16 paired, 1.48 per cycle per SIMD (0.92 with pairing off); a literal multiplier still pairs (1.18); per-lane sources in allocator-chosen registers reach 0.72 (INFERRED cause: register-bank triples across VOPD halves, section 3; measured there: 2.5 cycles per such pair, and 0.78 predicted against 0.79 measured without the delay hints) | all right | move uniform operands into VGPRs before the loop |
| (d) `__builtin_nontemporal_load` and vector types | needs native `ext_vector_type` | REMAINS: on HIP `uint4` it does not compile; on an `ext_vector_type` it gives `global_load_b128 ... th:TH_LOAD_NT` (READ, helper, rebuilt) | right | use `ext_vector_type` |
| (e) register cap / big tiles spill | `waves_per_eu(16)`: 96 VGPRs + 68 scratch | REMAINS: see section 6 | right (all sizes) | no cap below need |
| (f) `v_permlanex16` with fetch-inactive in masked code | reads the off lane's register (safe) | hardware SAME (fetches off lanes); clang now computes the source inside the branch, so the off partner's register is stale: 100 % of the lanes with a switched-off partner wrong, for `v_permlane16` (3 of the 5 mask shapes have such lanes: odd lanes, every 4th, data-dependent) and `v_permlanex16` (3 of 5: lanes 0-15, lanes 16-31, data-dependent); with the value pinned before the branch 100 % right | wrong unless pinned | pin the source or permute outside the branch |

Lane permute detail (MEASURED; 5 shapes per permute, of which 3 each leave lanes with a switched-off partner; the
check named `pit_perm` passes when partners that are on are read and off lanes are untouched, so its PASS says
nothing about the stale reads, which are the counts in its text): fetch-inactive off: a lane whose partner is off keeps its
own old value; fetch-inactive on: it gets the partner register's content, which clang filled only in executing lanes.
Lanes that are off are never written.

## 5. Memory access shapes (helper, rerun by me)

Source `src/study/haz_memshape.hip`, expectations `scripts/haz-expect/haz_memshape.txt` (53 kernels).

| Shape (8, 16, 32, 64 bytes per lane) | Emitted (READ) | On card | Cost (MEASURED) |
|---|---|---|---|
| struct copy, `memcpy` from `const char*` + run-time offset, `ext_vector_type`, `uint4`, `float4`, `__builtin_assume_aligned`, loop of 4 words | one `global_load_b64` / `b128` / 2 x `b128` / 4 x `b128`; never split | right | 16 B: 49 cycles per load cold, 13.9 warm, single wave; 632 GB/s from a 1 GiB buffer |
| low 4 address bits masked (`p & ~15`) | `flat_load_b128` (the address space is lost) | right | 210 cycles per load cold, 40.7 warm (4.3 and 2.9 times); same 632 GB/s whole-card |
| forced split control (4 x `b32`) | 4 x `global_load_b32` | right | warm 2.9 times slower single wave, no whole-card difference |

No gfx12/HIP counterpart of the neighbouring project's "mask the low bits" finding: clang already emits the wide
load, and masking hurts. (64 MiB rows read about 1600 GB/s: INFERRED, the buffer sits in the 64 MB cache.)

## 6. Register pressure (helper, rerun by me)

Source `src/study/haz_regs.hip` (one wave per tile, FP16 into FP32), expectations `scripts/haz-expect/haz_regs.txt`.
Waves per SIMD from lane 2's rule min(16, 1536 / VGPRs rounded up to 24) (INFERRED).

| Kernel | VGPRs (READ) | Scratch bytes | Spill loads/stores per K step | Waves/SIMD | Cycles per K step / per matrix instruction (MEASURED) |
|---|---|---|---|---|---|
| 2x2 tiles | 55 | 0 | 0 | 16 | 286 / 71.4 (load latency bound) |
| 4x4 | 167 | 0 | 0 | 9 | 476 / 29.8 |
| 4x6 | 239 | 0 | 0 | 6 | 609 / 25.4 |
| 4x8 | 256 | 1312 | 15 / 14 | 5 | 3202 / 100 (rerun: 3117 / 97) |
| 8x8 | 256 | 3328 | 80 / 78 | 5 | 20972 / 328 (rerun: 20357 / 318) |
| 4x4, `amdgpu_waves_per_eu(16)` | 96 | 360 | 18 / 16 | 16 | 3357 / 210 (rerun: 3389 / 212) |
| 4x4, `amdgpu_waves_per_eu(12)` | 120 | 264 | 12 / 10 | 12 | 2712 / 169 (rerun: 2667 / 167) |
| 4x4, `#pragma unroll 4` | 196 | 0 | 0 | 7 | 489 / 30.6 |

What sets the count (READ): 8 VGPRs per accumulator tile, 4 per A or B fragment held, about 7 for addresses; no
operand copies in the non-spilling loops. Unrolling holds more fragments in flight (+29 VGPRs). All results right.

llama.cpp `mul_mat_q` for gfx1201 (READ, compile only): 110 to 214 VGPRs for the kernels without bounds checks (7 to
12 waves per SIMD), up to 256 with bounds checks; one kernel spills (q8_0, mmq_x = 32, bounds checks: 256 VGPRs,
48 bytes scratch, 11 loads and 11 stores). Other tile widths compile to a 39-VGPR stub without matrix instructions.

gfx11 comparison: the same mechanism (cap or accumulators past 256 forces scratch); on gfx12 a 4x6 tile of FP32
accumulators is the largest that fits.

## 7. Float division and other slow paths (helper, rerun by me)

Source `src/study/haz_slow.hip`. Cycles per operation in a dependent loop, including a 20-cycle loop floor (MEASURED;
spread under 0.3 cycles).

| Operation | Emitted (READ) | Cycles | Data-dependent? |
|---|---|---|---|
| `a / b`, `1.0f / b`, `__fdividef` | the same 11-instruction correctly rounded sequence (`v_div_scale`, `v_rcp`, FMAs, `v_div_fmas`, `v_div_fixup`) | 80 | no: zero numerator, denormals, NaN, infinity, 1e30/1e-30 all 80 |
| `sqrtf` | `v_sqrt_f32` plus scaling, compares and selects | 119 | no |
| `expf` / `__expf` | range reduction + `v_exp_f32` / `v_mul` + `v_exp_f32` | 69 / 23 | no |
| 32-bit signed / unsigned `/` and `%` | 33 to 46 instructions around `v_rcp_f32` | 124 to 149 | no |
| 64-bit multiply | `v_mad_co_u64_u32` + `v_mul_lo` + add | 49 | no |
| 64-bit unsigned `/` | branch: 32-bit path or long path | 205 or 672 | YES, 3.3 times when either operand has high bits |
| int to float, float to int | one `v_cvt_*` | 27 | no |

All bit-exact against the CPU inside the checked ranges (expf within 1 ulp; 5 inputs outside the checked range differ, `haz_slow_range` in the log). gfx11's slow general path for zero numerators (neighbouring project,
Mesa) does not appear here: clang's division sequence is branch-free.

## 8. Wave64 under HIP

Source `src/study/haz_w64.hip`, built with `-mwavefrontsize64`. It compiles (`.wavefront_size: 64`) and runs; the
`_w64_gfx12` builtin gives the model's bits (0 of 256 wrong) under this layout (MEASURED): the wave32 register image
with each lane's upper half moved to lane + 32 (A row r slot s: lane r + 16 (s >> 3) + 32 ((s >> 2) & 1), half s & 3;
D (r, c): lane 16 (r >> 3) + c + 32 ((r >> 2) & 1), float r & 3). My first guess (slot groups of 4 over 4 lane
groups) gave 128 of 256 wrong. Chained: 16.01 cycles per instruction, the same as wave32, so the same rate per SIMD
(agrees with the Vulkan lane). Pitfall: the host's `hipDeviceProp_t::warpSize` still says 32.

## Method

- Every claim about emitted code comes from the `-save-temps` listing (`build/tmp-<probe>/*.img.lto.s`) through a
  script with an expectation file: `scripts/haz_maskscan.py` (forward data-flow of the EXEC state over each
  function's blocks; flags matrix instructions and lane permutes that can run under a partial mask),
  `scripts/haz_hazscan.py` (from each matrix instruction, any path to the next matrix instruction without a VALU;
  flags A/B overlap, a return or a call in between), and the study's `scripts/mem_loopcheck.py` (instruction counts
  per loop and per kernel, VGPRs, scratch).
- Card results are judged against `src/study/arith_model.h` (bit-exact model of the matrix instructions) or a CPU
  reference; inputs are small integers where exactness makes the reference independent of rounding.
- Timing: the shader cycle counter around a loop, difference of two trip counts.

Deliberate breaks (all fail as they must, `runs/20261117T160358Z-haz/summary.txt`): mask-scan expectation that calls
the per-lane-if kernel clean, and one that calls the clean kernel masked; hazard-scan expectation that calls the
no-V_NOP assembly clean; a loop-count expectation off by one; the matrix-mask rule checked as "switched-off lanes
untouched"; the hazard pair without V_NOP (64 of 256 wrong); the widening checked with RDNA 3's broken odd mask; an
FMA chain checked one FMA short; wave64 checked under the wrong layout; and the three helpers' breaks (division
against `a * b`, shifted memory reference, shifted tile reference).

## How to rerun

```
STUDY_REPO=<rdna4-fp8 checkout> scripts/study-compiler.sh all     # build, listing checks, card runs, breaks (~1 min)
STUDY_REPO=<rdna4-fp8 checkout> scripts/study-compiler.sh llama   # optional: llama.cpp mmq listings and scans
```

`bash -n` and `shellcheck -S warning` are clean on the script.

## Not measured

Still open:

- The scalar-register hazard showing a wrong result: the no-wait sequence I wrote computed right; a sequence that
  fails without the wait was not found in the time. Low value.
  Answered in part (leftovers C, CH-1, `measurements/leftovers-c-a.md`): 0 of 60 no-wait sequences differed from
  their waited twins, down to an SALU write directly before the VALU read; on this card the VALU read appears to wait
  for the SALU write by itself in these shapes (INFERRED). MEASURED; "never needed" is not shown.
- A `v_cmpx` directly before a lane permute from plain C++ (clang always put a VALU between in the shapes tried).
  Not reachable (leftovers C plan, CH-2b): no source shape produces it. The hardware side is answered below (CH-2a).
- llama.cpp kernels on the card (compile only, by the brief); a ragged-shape run of its bounds-checked kernels.
  Needs the owner (leftovers C plan, CH-4): it lifts the "no llama.cpp kernel work" limit.
- Misaligned per-lane addresses for wide loads, for byte offsets that are not multiples of 4: not tried (possible
  memory fault).
  Answered in part (leftovers C, CH-5a, `measurements/leftovers-c-a.md`): with 4-, 8- and 12-byte offsets,
  `global_load_b128` returned the right bytes (0 wrong words of 262,144 per offset). A misaligned 16-byte load costs
  about 5 cycles more per load from the near caches (19 against 13.9, one wave, warm), about 3 more from memory, and
  nothing when the whole card streams from memory (632 GB/s at every offset). MEASURED.
- Optimisation levels other than -O3, what is left: the hazard waits at -O0 are not judged; `haz_memshape` was not
  run on the card at the other levels (listing only); the -O0 listing of the permute pinning pitfall was not read.
  Answered in part (leftovers C, CH-5b, `measurements/leftovers-c-a.md`): at -O1, -O2 and -Os every hazard shape the
  compiler handles at -O3 is handled, with the same one gap at a call boundary (64 of 256 wrong); the unsafe mask
  shapes of item 1 (`mask_ifload`, `mask_early`) give wrong results at every level, -O3 included; the small
  differences between levels in counts are not shown to be level effects. At -O0 the question cannot be judged: the
  "the hazard waits hold at -O0" reading was WITHDRAWN, because the -O0 code keeps dependent matrix instructions 51 or
  more instructions apart (4 at the other levels), so no wait is needed and none is tested (T5). At -O0 lane permutes
  that read switched-off lanes get stale values (through memory: read in one probe, inferred in the other).
- The lane mask on the two sparse matrix forms that leftovers C did not run (it ran 9 of the 11 sparse forms).
  Answered in part (leftovers C, CH-6, `measurements/leftovers-c-b.md`): all 20 forms tried (11 dense, 9 sparse:
  BF16, BF8, 4-bit and the rest) read and write every lane whatever EXEC says, in wave32 and wave64, and the time is
  the same to the cycle with EXEC full or only lane 0 (16.000 for the 16-bit-input forms, 8.000 for the 8-bit and
  4-bit forms). MEASURED.

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, CH-2a, `measurements/leftovers-c-a.md`): a `v_cmpx` directly before `v_permlane16_b32` /
  `v_permlanex16_b32`, written in inline assembly: 0 lanes differ from the model in all 60 cases (2 permutes, fetch-
  inactive on and off, nothing, `v_mov` or `s_nop` between, 5 masks). The permute sees the new EXEC without the guard;
  the hazard LLVM guards against did not show. MEASURED.
- Answered (leftovers C, CH-3, `measurements/leftovers-c-a.md`): wave placement in the multi-wave timing: every
  kernel, both sizes, all 5 launches had 4 waves on each of the 4 SIMDs (512 threads) and 8 on each (1,024 threads),
  so item 3's per-SIMD figures stand as computed. MEASURED.
