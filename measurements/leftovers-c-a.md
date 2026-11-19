# Leftovers C, lane A: instruction level, memory, LDS, hazards

Card: RX 9070 XT (gfx1201), ROCm 10.1 `hipcc` (clang 24), LACT profile "Testing". Branch `study/leftovers-c-a`.
Script: `scripts/study-leftovers-c-a.sh`. Plan: `rdna3-fp8/build/rdna4-runs/plans/leftovers-c-plan.md`, section 3
"Lane A". Labels: MEASURED (seen in a run here), READ (from a document), INFERRED (reasoning).

## In short

- Wave64 runs 32-bit float math at full speed only for some register layouts; integer, 16-bit and 64-bit math run
  at half speed per instruction. The 0.81-per-cycle figure seen under HIP was the register layout: the same kernel
  with only its registers moved, so that no instruction reads two registers of one bank, ran at 1.00 per cycle
  (test T1, measured). There is no simple rule for the other layouts: blocks in which every instruction reads two
  registers of one bank cost anywhere from 1.0 to 2.5 cycles per instruction in wave64, depending on what the
  neighbouring instructions read.
- Reading a fresh piece of memory takes about 240 to 260 ns whatever the load looks like. The earlier probes
  disagreed in cycles because neighbouring reads shared the cache's 256-byte lines, and because the clock differed.
- A pause of about 3,300 cycles shows up 380 to 420 times a second in the lane's runs (520 to 560 in the lead's
  rerun) also when the wave never reads the timer. That it hits the whole card comes from the earlier
  scheduling-and-clocks item 7; this test used one wave. The study's rule for spotting disturbed samples catches
  what its threshold says it should.
- Shared memory (LDS): misaligned accesses give the right bytes but can be 32 times slower. A wave stops issuing LDS
  loads once about 17 (128-bit) to 22 (32-bit) slow loads are waiting; fast loads issued 96 in a row without a stop
  in the earlier LDS lane, so this shows only when loads drain slowly. Atomics, stores and lane shuffles behave as
  the documentation and the bank model say.
- One wave's address translation covers about 64 MiB: about 1,024 nodes when the nodes are 64 KiB or more apart
  (at 32 KiB spacing 2,048 nodes still hit); two points do not fit (16 KiB spacing slows at 3,072 nodes, 2 MiB spacing already at 1,024).
  One wave alone slows down somewhere between about 4,000 and 11,000 L2 lines (it wobbles from run to run);
  eight or sixteen waves together keep L2 speed up to about 45,000 lines in all, in all three runs. So one wave
  alone does not get all of L2, and many waves do.
- Each compute unit's first-level cache delivers 128 bytes per clock even with all 64 units busy: 25.8 TB/s.
- Several timing oddities now have rules (VALU writes to scalar registers run at 8 per 15 cycles; three inputs from
  one register bank cost a cycle and need a gap; a 4 KB loop is cheap only at a 4 KB boundary). Where loop code sits
  matters more than its size alone: two different 3 KB loops on the two SIMDs of a compute unit slowed each other in
  one build and not in another whose loops sat 68 bytes further on; across the two compute units of a WGP the slow
  side follows the code copy, not the compute unit (test T2). Long runs of 12-byte instructions still have no rule.
- The cost of FP64 square root, reciprocal square root and reciprocal, and of the scalar-result transcendentals, does
  not depend on the data (1.7, 1, 0, infinity, the smallest subnormal, NaN: test T3).
- The LDS table method for widening FP8 does not get faster by keeping fewer loads in flight: with at most 16 in
  flight it is slower (7.2 per weight) than with 32 (5.5) or the simple form (4.7) (test T4).
- Compiler: the hazard fixes hold at -O1, -O2 and -Os (as at -O3, with the same gap at a call boundary). At -O0
  the question cannot be judged: a copy of the hazard probe builds and gives right results there (test T5), but the
  -O0 code never puts two dependent matrix instructions near each other (51 or more instructions apart, against 4
  at the other levels), so no hazard wait is needed and none is tested. At -O0, lane permutes that read
  switched-off lanes get stale values in two probes; in one of them the listing shows the value going through
  memory, for the other that cause is inferred. Two hazards the
  compiler guards against could not be made to go wrong on this card. Every cache hint, and the last-use hint, can be
  reached from HIP or LLVM IR.
- Every item was run, some only in part; the parts not run, and what stays open, are under "Not measured".

## SC-10. Why wave64 FMAs issued at 0.81 per cycle under HIP

Question: scheduling-and-clocks item 4 measured 51.85 lane-FMAs per cycle for wave64 `v_fma_f32` (1.234 cycles per
instruction), where Vulkan's wave64 FMAC loop reached 0.944 of 64 (vulkan-view item 3, 1.06 cycles). Why?

Expected: no prior answer (exploratory). Known answers used as checks: wave32 independent FMA 1.00, dependent FMA
5.00 (execution-core items 1 and 2), chained FP16 matrix instruction 16.0 cycles in both wave sizes (matrix-timing
item 2, scheduling item 4).

Probe: `src/study/left_core.hip` mode `w64`, built twice (wave32, and `-mwavefrontsize64 -DW64=1`), one wave alone.
Blocks of 16 instructions with registers fixed in the text (`scripts/leftc_gen_core.py`), counter reads and loop in
one inline-assembly statement (method of execution-core item 8); cost = (R = 6 - R = 2) / (4 x 16 x 16 iterations),
R = 4 midway. Listing check: every timed region holds exactly R copies of the block plus 4 loop-control
instructions (`scripts/leftc_asmcheck.py`, 3,510 regions in the wave32 object and 432 in the wave64 one, 0 mismatched,
in the final rerun `runs/20261117T215755Z-leftc-a`, `asmcheck-left_core.txt`).

MEASURED, run `runs/20261117T201152Z-leftc-a`, 45 samples per kernel, 0 dropped; cycles per instruction (range from
the extreme samples):

| Block (16 independent instructions unless noted) | wave32 | wave64 |
|---|---|---|
| `v_fma_f32` vR, vR, 0.5, 0.5 | 1.000 | 1.000 (0.985-1.015) |
| dependent `v_fma_f32` chain | 5.000 | 5.000 |
| `v_fma_f32` vR, vR, v32, 0.5 | 1.000 | 1.000 |
| `v_fma_f32` vR, vR, v32, v33 (8 of 16 have two sources in one bank) | 1.000 | 1.250 (1.235-1.265) |
| the same with the two constants in the banks after vR's (at most one source per bank) | 1.000 | 1.000 |
| `v_fmac_f32` (VOP2) vR, v32, v33 | 1.000 | 1.250 |
| `v_fmac_f32` with banks spread / with an inline constant | 1.000 / 1.000 | 1.000 / 1.000 |
| `v_fmac_f32_e64` vR, v32, v33 | 1.000 | 1.250 |
| `v_add_f32` (VOP2, VOP3, two VGPRs), `v_mul_f32`, `v_mov_b32`, `v_fmaak_f32` | 1.000 | 1.000 |
| `v_add_nc_u32` | 1.000 | 2.000 |
| `v_pk_fma_f16` | 1.000 | 2.000 |
| `v_cvt_f32_i32` | 1.000 | 2.000 |
| `v_exp_f32` | 4.000 | 8.000 |
| `v_fma_f64` | 32.00 | 64.00 |
| three bank-1 registers, the same triple every time | 1.500 | 1.500 |
| two different bank-1 triples alternating | 5.000 | 5.000 |
| chained `v_wmma_f32_16x16x16_f16` | 16.000 | 16.000 |

Answer:
- In wave64, FP32 add, multiply, FMA, FMAC, FMAAK and `v_mov_b32` cover all 64 lanes in one cycle (64 lane-ops per
  cycle per SIMD); integer add, packed FP16, conversion, transcendental and FP64 instructions take exactly twice
  their wave32 cycles. MEASURED.
- The block in which 8 of 16 FMAs read two VGPRs of one register bank (bank = register number mod 4) costs 1.250
  cycles per instruction in wave64 and 1.000 in wave32; spreading the two constants over the banks removes the extra
  cost, VOP2 or VOP3 alike (45 samples per block, ranges in the table). MEASURED. The first reading of this, "each
  such instruction costs 1.5 cycles" (8 of 16 at an extra 0.5 each), was derived from this one block, not measured;
  test T1 below measured blocks in which all 16 instructions read two VGPRs of one bank and found 1.000 to 2.500,
  so it is not a rule. Three registers of one bank cost the same in both wave sizes (1.5 when the triple repeats, 5.0
  when it changes).
- The 0.81 of scheduling item 4: its listing (`build/tmp-sched_wave64/...lto.s`, READ) has the constants in v16 and
  v17 (banks 0 and 1) and the 16 chains in v0-v15, so half the FMAs read two registers of one bank, the pattern of
  the `v32, v33` row (1.250 here against 1.234 there). Test T1 settles it on that probe itself: its kernel copied
  unchanged gives 1.234 again, and the same kernel with only the registers moved, so that no FMA reads two registers
  of one bank, gives 1.000. MEASURED. Vulkan's 1.06 was not re-read (its listing was not opened; see "Not
  measured").
- Practical rule: wave64 FP32 math ran at full speed in every block whose instructions read their VGPR sources from
  different banks (the "spread" rows here, and T1); other layouts cost between 1.0 and 2.5 cycles per instruction in
  the blocks measured, without a rule found. Integer and 16-bit work runs at half the wave32 lane rate per
  instruction (the same lanes per cycle as wave32). LLVM checks banks only for VOPD (execution-core item 3), so
  wave64 code from clang can lose speed on FP32; the 25 % of the HIP probe is one case, not a bound.

**Test T1 (fix round): the 0.81 probe itself, and blocks where every instruction has two sources in one bank.**
Expectation written before the run: if the 0.81 comes from the register banks, the earlier probe's kernel with
only its registers moved (no instruction reading two registers of one bank) runs at 1.00 cycle per FMA (64
lane-FMAs per cycle); a contradiction would be any rate clearly below 64. If "1.5 cycles per instruction with two
sources in one bank" is a per-instruction rule, every block where all 16 instructions have two sources in one bank
costs 1.5 in wave64 and 1.0 in wave32; a contradiction would be any such block away from 1.5.

Probes: `src/study/left_w64fma.hip`, a copy of the FMA section of `sched_wave.hip` (item 4's kernel unchanged as
mode 0, and the same kernel with its 16 chains and 2 constants pinned to chosen registers by register constraints;
`sched_wave`'s measurement, one wave, 7 repetitions, run 3 times interleaved: 21 samples per mode), wave64 CU-mode
build as `sched_wave64`. Listing check (`scripts/leftc_bankcheck.py`): each timed loop holds U x 16 FMAs with the
intended number of shared-bank instructions; the pinned modes also hold 10 to 24 `v_mov_b32` copies per loop
iteration, the same at U = 4 and U = 8, so the U difference cancels them. Control (verdict `w64fma_control`): mode
0 reproduces item 4's 51.85 lane-FMAs per cycle, and the same layout pinned gives the same; break: mode 0 expected
at 64: FAIL. And `src/study/left_bank.hip` mode `bank`, built in both wave sizes: blocks of `left_core`'s method
(`LEFT_T_BLOCKS` in `scripts/leftc_gen_core.py`), with calibration copies (independent FMA 1.000, dependent 5.000,
the 8-of-16 block 1.250 in wave64 and 1.000 in wave32; verdict `bank_calib`; break: the FMA expected at 2.00: FAIL).

MEASURED, `left_w64fma` (21 samples per mode; medians, 10th and 90th percentiles equal in every mode; single
outliers exist: one sample of the no-shared-bank mode read 80.01 lane-FMAs per cycle, above the 64 that are
possible, in one of its three launches, and one sample of the unchanged kernel read 44.88), final run `runs/20261117T224107Z-leftc-a`
(the development run `runs/20261117T222442Z-leftc-a` gave the same five values):

| Mode (wave64, one wave) | Lane-FMAs per cycle | Cycles per FMA |
|---|---|---|
| item 4's kernel unchanged (compiler's registers: chains v0-v15, constants v16, v17; 8 of 16 share a bank) | 51.85 | 1.234 |
| the same registers pinned | 51.85 | 1.234 |
| chains in banks 2 and 3, constants v32 (bank 0) and v33 (bank 1): no instruction with two sources in one bank | 64.00 | 1.000 |
| chains in banks 1 to 3, constants v24 and v28 (both bank 0): all 16 share a bank | 53.89 | 1.188 |
| chains in bank 0, constants v64 (bank 0) and v65: all 16 share a bank | 32.00 | 2.000 |

MEASURED, `left_bank` and `left_bank64`, final run `runs/20261117T224107Z-leftc-a` (the same values in
`runs/20261117T223438Z-leftc-a`), 45 samples per block, 0 dropped, cycles per instruction (every range within +-0.015
of the median):

| Block of 16 (registers) | Two sources in one bank | wave32 | wave64 |
|---|---|---|---|
| `v_fma_f32 vR, vR, v32, v33`, vR = v16-v31 (the SC-10 block) | 8 of 16 | 1.000 | 1.250 |
| `v_fma_f32 vR, vR, v32, v36`, vR in banks 1-3 (constants share bank 0) | 16 | 1.000 | 1.125 |
| `v_fma_f32 vR, vR, v32, v33`, vR all in bank 0 | 16 | 1.343 | 2.000 |
| `v_fmac_f32 vR, v32, v33`, vR all in bank 0 | 16 | 1.343 | 2.000 |
| `v_fma_f32 vR, vR, v32, 0.5`, vR all in bank 0 (two VGPR sources) | 16 | 1.343 | 1.343 |
| `v_add_f32 vR, v32, vR`, vR all in bank 0 (VOP2) | 16 | 1.343 | 1.343 |
| `v_fma_f32 vR, vR, vR, 0.5` (one register read twice) | (same register) | 1.000 | 1.000 |
| chain and first constant in one bank, the bank rotating (`vR, vR, v(84 + R mod 4), v(84 + (R+1) mod 4)`) | 16 | 1.000 | 1.000 |
| the two constants in one bank, the bank rotating (`vR, vR, v(88 + (R+1) mod 4), v(92 + (R+1) mod 4)`) | 16 | 1.000 | 2.500 |
| destinations all in bank 0, sources in banks 1-3 (one VGPR source) | 0 | 1.000 | 1.000 |
| sources all in bank 0, destinations in banks 1-3 (one VGPR source) | 0 | 1.000 | 1.000 |

Answer (T1):
- The 0.81 of scheduling item 4 came from the register layout: the same kernel with only the registers moved runs
  at 1.000. MEASURED (this replaces the earlier INFERRED match).
- "1.5 cycles per instruction with two sources in one bank" is contradicted: blocks where every instruction has two
  sources in one bank cost 1.000, 1.125, 1.188, 1.343, 2.000 or 2.500 in wave64, depending on which registers the
  neighbouring instructions read and whether the shared bank is also the one being written. The cheapest such
  block (1.000) re-reads in each instruction a constant the instruction before also read; the dearest (2.500) reads
  two constants that the instruction before did not read. But the blocks that read the same two constants in every
  instruction cost 1.125 to 1.250, so "re-reading is free" does not explain them either. The mechanism is not
  established.
- A separate effect that does not depend on the wave size: when every instruction reads and writes bank 0 (16
  chains all in bank 0), the block costs 1.343 even in wave32 and even with only two VGPR sources, while reading
  bank 0 only or writing bank 0 only costs 1.000. MEASURED; INFERRED: reads and result writes of one bank compete.
- The two all-16 "constants in one bank" measurements differ (1.125 in `left_bank`, 1.188 in the copy of item 4's
  kernel, with different registers and the copy's compiler-built loop), which again says the cost depends on more
  than the bank count.

Label: MEASURED (rates), INFERRED (mechanism, not established). Verdicts `w64_calib`, `w64_calib64`,
`w64fma_control`, `bank_calib` PASS; `w64_rates`, `w64fma_rates`, `bank_all` INFO.

## MS-3 and MS-4. Memory latency across probes, and the in-flight probe's N = 1 and N = 2 rows

Questions: (MS-3) the chase gave 485-545 cycles from memory, the in-flight probe 549-772, the prefetch probe 694
(memory-system "Not measured"): is it the clock, as memory-system INFERRED, or the access shape? (MS-4) in the
in-flight probe 4-byte loads with 1 or 2 in flight cost 549 and 845 cycles per batch, 2 being slower than 4 (777):
does that vanish when each load reads its own L2 line?

Expected: within one launch the clock is the same for every segment, so a cycle difference there is not the clock.
The in-flight probe's source (READ, `mem_inflight.hip`: load K at offset K x 128, batches back to back) puts two
consecutive loads in one 256-byte L2 line (memory-system 2b, 2b-2 and 4: memory moves in 256-byte units). Lane 2's
rule (memory-system item 5): a batch costs about one latency for N up to 63, so with distinct lines N = 2 <= 1.10 x
N = 1 and N = 4 <= 1.10 x N = 2.

Probe: `src/study/left_mem.hip` mode `ns`. One wave; seven launches, each after a 512 MiB flush read; inside each
launch, in order, segments that each read the cycle counter and REALTIME (10 ns ticks) at start and end, every
region fresh (not read since the flush): a pointer chase of 1,024 steps (random inside 64 KiB blocks, 128-byte
nodes, all lanes one address: memory-system 2b's "random in 64 KiB blocks"), the same chase stamped per step (128
steps), both again with 256-byte nodes, then 128 batches each of N = 1, 2, 4, 8 loads of 4 bytes per lane in the old
layout (consecutive 128-byte lines), with every load on its own 256-byte line, and 4 KiB apart, and 16-byte loads
(512 bytes per load) consecutive and 4 KiB apart, then the first chase again. Every batch is stamped on its own
(cycles kept in LDS); the listing check (`scripts/leftc_regioncheck.py`) confirms each stamped batch holds only its
loads, the wait and the counter reads. Calibration: L0 chase (16 KiB random) 81 cycles +-10 % (tim_l2), and the old
layout's per-batch mean ratios N2/N1 = 1.540 and N4/N1 = 1.415 (mem_inflight) within 8 %, ratios because cycle
counts of different jobs depend on the clock.

MEASURED, run `runs/20261117T201833Z-leftc-a`, clock 2.79-2.81 GHz in every segment (cycles / ns of the segment); medians of the 7 x
128 per-batch stamps (chase rows without "stamped": mean per step over 1,024 steps, median of 7 launches):

| Shape | Cycles | ns | Share below 450 cycles (L2 hits) |
|---|---|---|---|
| L0 calibration chase | 80.7 | - | - |
| chase, 128-byte nodes, 1,024 steps (two full 64 KiB blocks) | 463 (444-496) | 163 (154-166) | - |
| chase, 128-byte nodes, stamped, first 128 steps of a block | median 704, mean 682 | 253 | 12 % |
| chase, 256-byte nodes, 1,024 steps | 660 (645-749) | 236 (231-247) | - |
| chase, 256-byte nodes, stamped | median 715 | 257 | 0 % |
| old layout N = 1 (two batches per L2 line) | median 605, mean 512 (p10 245, p90 833) | 216 | 50 % |
| old layout N = 2 / 4 / 8 | 738 / 680 / 687 | 263 / 242 / 244 | 0 % |
| each load on its own 256-byte line, N = 1 / 2 / 4 / 8 | 724 / 680 / 682 / 705 | 258 / 242 / 243 / 251 | 0 % |
| loads 4 KiB apart, N = 1 / 2 / 4 / 8 | 667 / 681 / 688 / 707 | 237 / 243 / 245 / 252 | 0 % |
| 16 bytes per lane (512 bytes per load), consecutive / 4 KiB apart, N = 1 | 707 / 693 | 251 / 247 | 0 % |

Answer:
- MS-3: in one launch at one clock the chase with 128-byte nodes costs 463 cycles per step and the one with 256-byte
  nodes 660, a single load on its own line 667-724. So the shape decides, not the clock: with 128-byte nodes two
  nodes share a 256-byte L2 line, and once a chase has walked a whole 64 KiB block (or walks in increasing order)
  half its steps find the line already fetched (about 245 cycles) and half go to memory (about 700): mean 463. The
  stamped first 128 steps of a block show the same thing early (12 % hits, as expected when about k / 512 of the
  partners have been visited). In nanoseconds the earlier rows agree with this: memory-system 2b's increasing and
  block orders read 149-158 ns, this chase 160-163 ns. Every shape here with one read per L2 line costs 237-258 ns
  at N = 1; the in-flight probe's 772 cycles (16 bytes per lane) is that at a higher clock (251 ns x 3.08 GHz =
  772, INFERRED: that job's clock was not logged per segment). MEASURED. The 694 of the prefetch probe was not
  rerun (its shape was not re-read; INFERRED: one read per line at about 2.8 GHz).
- MS-4: the old layout's N = 1 row is a mix: exactly 50 % of its batches are L2 hits (p10 245 cycles) because
  consecutive batches read the two halves of one 256-byte line, so its mean (512 here, 549 there) is half an L2
  latency plus half a memory latency. With each load on its own line, N = 2 costs no more than N = 1 (680 against
  724 cycles; 4 KiB apart 681 against 667) and N = 4 no more than N = 2: lane 2's rule holds (verdict
  `mem_distinct_lines` PASS). The old layout's N = 2 (both loads in one 256-byte line) stays slower than N = 4 here
  too (738 against 680, medians), about 55 cycles (20 ns) above the distinct-line N = 2: two loads to one L2 line
  that is in flight cost more than two loads to two lines. MEASURED; why is not established (INFERRED: the second
  request to a line that is being filled waits for the fill and is answered after it).
- Rule for the study: memory latency is about 240-260 ns per read of a fresh 256-byte line, at any clock; quote
  memory latencies in ns, and count L2 lines (256 bytes) when a layout reads 128-byte pieces.

Label: MEASURED (table), INFERRED (the in-flight-request mechanism and the 772 clock). Verdicts `mem_ns_calib` PASS,
`mem_distinct_lines` PASS, `mem_clock_explanation` INFO.

## EC-3. A dual-issue pair that breaks the guide's rule

Question: the RDNA 4 guide says "OPX must not overwrite sources of OPY; OPY may overwrite sources of OPX"
(READ, AMD's RDNA 4 instruction set guide, 7.8). The RDNA 3 guide added that both halves execute at the same
time and reads see the old value; the RDNA 4 guide dropped that sentence. The assembler accepts the forbidden pairs
(it enforces register parity and banks only). What does the hardware do?

Expected: unknown (exploratory). Requirements for the controls: the same two operations as two ordinary
instructions give the new value; the allowed reverse case (OPY overwrites OPX's source) gives OPX the old value.

Probe: `src/study/left_vopd.hip`, one wave, every lane its own exact FP32 data, 1,000 repetitions of each variant in
one launch; the result of the reading half is classified as old, new or other. The object holds the seven pairs as
written, in order (listing check in the script).

MEASURED, run `runs/20261117T202103Z-leftc-a` (32,000 lane results per variant):

| Variant (X :: Y) | Old | New | Other |
|---|---|---|---|
| `v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23` (Y src0 = X dst) | 32,000 | 0 | 0 |
| the same with v10 as Y's src1 | 32,000 | 0 | 0 |
| `v_dual_add_f32 v10, ... :: v_dual_mul_f32 v11, v10, v23` | 32,000 | 0 | 0 |
| `v_dual_fmac_f32 v10, ... :: v_dual_fmac_f32 v11, v10, v23` | 32,000 | 0 | 0 |
| `v_dual_mov_b32 v10, v21 :: v_dual_mov_b32 v11, v10` (Y's mov reads through the SRC2 port) | 32,000 | 0 | 0 |
| the first pair with v10 written by the instruction right before (no `s_nop`) | 32,000 | 0 | 0 |
| allowed: `v_dual_add_f32 v10, v21, v22 :: v_dual_mov_b32 v21, v23` (X's view of v21) | 32,000 | 0 | 0 |
| control: `v_mov_b32` then `v_add_f32`, two instructions | 0 | 32,000 | 0 |

Answer: on gfx1201 both halves of a VOPD pair read their sources before either writes, whichever half writes and
whichever operand slot reads, and whether the source was written just before the pair or long before. The forbidden
direction behaves exactly like the allowed one: deterministic, old value. MEASURED for these five forbidden shapes
(write-write to one register cannot be encoded: the destinations must differ in parity). Practical meaning: code
that relies on it works on this card, but the guide does not promise it, so a later chip may differ.

Label: MEASURED. Verdicts `vopd_control` PASS, `vopd_deterministic` PASS; the answer (old) is INFO.

## SC-2 and SC-1. The stall without the REALTIME read, and the detection rule on real traces

Questions: (SC-2) the stall of about 1.1 us every 2.8 ms (scheduling-and-clocks item 7) was found by waves that read
REALTIME (`s_sendmsg_rtn_b64`) after every step; is it there without that message? (SC-1) how well does the rule
"a step above 1.5 x the median step + 64 cycles is disturbed" work on a real trace?

Expected: (SC-2) if the stall is a card-level event it appears at the same rate and size without the message
(draft's bands: rate ratio 0.7 to 1.4, extra-cycle ratio 0.85 to 1.15; earlier 390 to 1,023 events per second).
(SC-1) by construction the rule flags a step whose excess over the median is above 0.5 x median + 64 cycles; a gap
well above that must be found, one well below must not.

Probe: `src/study/left_spin.hip`, one wave, steps of 256 independent FMAs (sched_share's spin step), every step's
cycles kept. Variants in turn R, N, Q, R, N, Q in one job, 10 million steps (1.1-1.35 s) each, the first 200,000
steps (about 30 ms, the clock ramp) dropped: R reads REALTIME and stores 8 bytes per step (as sched_share), N stores
the same 8 bytes but reads no REALTIME, Q reads nothing and stores only the steps above the limit. Then one N launch
with `s_sleep k` after about 200 steps for each k = 2, 3, 4, 5, 6, 8, 32, 127, the steps picked by a hash of the step
number computed in the kernel (the listing check confirms no memory read in any spin loop: a first version read the
schedule from memory every step, which made steps slower and noisier; that run is not used). Checks: synthetic
trace clean 0 events and with a 50 us gap 1 event; `s_sleep 127` alone 8,098 cycles (core_counter 8,128, +-64).

MEASURED, run `runs/20261117T202732Z-leftc-a` (two rounds; the run before it, `runs/20261117T202649Z-leftc-a`, gave the same within 8 %):

| Variant | Median step (cycles) | Events per s | Events per million steps | Extra cycles per event (median) |
|---|---|---|---|---|
| R: REALTIME every step | 448 / 448 | 392 / 392 | 53.9 / 51.1 | 3,303 / 3,409 |
| N: no REALTIME, a store every step | 379 / 379 | 396 / 389 | 45.2 / 43.4 | 3,350 / 3,466 |
| Q: no REALTIME, no store per step | 379 / 379 | 382 / 386 | 42.7 / 42.8 | 4,273 / 4,243 |

Clocks of the rates: the R rate divides by the REALTIME span of the steps kept; N and Q, which read no REALTIME,
divide by the launch's wall time measured on the host (`seconds ... (launch wall time)` in the log). The two
differ by 1 to 2 % (a launch includes its start and the 200,000 dropped steps).

The same job in the two later reruns, events per second, round 0 / round 1 (extra cycles per event, the median of
both rounds, in brackets):

| Run | R | N | Q | Q / R extra cycles |
|---|---|---|---|---|
| lane, `runs/20261117T202732Z-leftc-a` (above) | 392 / 392 (3,356) | 396 / 389 (3,408) | 382 / 386 (4,258) | 1.27 |
| lane's final rerun, `runs/20261117T215755Z-leftc-a` | 400 / 397 (3,343) | 422 / 399 (3,329) | 384 / 388 (4,147) | 1.24 |
| lead's rerun, `runs/20261117T220137Z-leftc-a` | 562 / 526 (3,236) | 537 / 537 (3,280) | 536 / 518 (4,004) | 1.24 |

Injected sleeps (median step 556 cycles in that launch, which computes the schedule hash each step; limit 898, so
an excess above 342 cycles is flagged):

| `s_sleep k`, nominal cycles | 128 | 192 | 256 | 320 | 384 | 512 | 2,048 | 8,128 |
|---|---|---|---|---|---|---|---|---|
| Real extra cycles of the step (median) | 96 | 158 | 223 | 261 | 326 | 420 | 1,970 | 8,051 |
| Found by the rule | 0 / 199 | 0 / 230 | 0 / 214 | 0 / 222 | 51 / 191 | 207 / 207 | 191 / 191 | 213 / 213 |

Steps flagged outside the injected ones: 10.4 per 100,000 steps, 3.2 per 100,000 with an excess below 1,500 cycles.

Answer:
- SC-2: the stall is there without the REALTIME read, at the same rate per second within each run (R 392, N
  389-396, Q 382-386 in the named run) and the same size when the wave also stores every step (N/R extra 1.00 to
  1.02 in the three runs). Per million steps the rate reads lower without REALTIME only because those steps are
  shorter (the message costs about 70 cycles per step). Q, with no store per step, loses more per event in all
  three runs (1.24 to 1.27 x, outside the draft's band). MEASURED; why is not established (INFERRED candidate: with a
  store per step the wave has a store in flight when the stall begins and part of the stall overlaps it). So the
  stall is not something the REALTIME message causes. That it is a card-level event rests on scheduling-and-clocks
  item 7 (waves on several SIMDs saw it together); this job runs one wave and does not show it by itself.
- The rate is not a fixed property: the lead's rerun, 4 minutes after the lane's final rerun, saw 518 to 562
  events per second in all three variants, 30 to 40 % above the lane's 382 to 422, with the same loss per event.
  So the variants still agree with each other, and the number of events per second changes from run to run (the
  earlier item 7 saw 390 to 1,023). What sets the rate was not tested here (the desktop's activity is one
  candidate, not checked).
- SC-1: a sleep overlaps the work still in flight, so a step loses 60 to 90 cycles less than the nominal 64 k. With
  that real loss, the rule behaves as its threshold says: every gap of 420 cycles or more (78 above the limit) was
  found, nothing of 261 or less (81 below it), and the 326-cycle gap, 16 below the limit, was found in 27 % of its
  steps (step-to-step jitter). Outside the injected steps it flags about 10 steps per 100,000, consistent with the
  regular stall (about 6.5 events per 100,000 steps of this length, one or two steps each); 3 per 100,000 have an
  excess below 1,500 cycles and are not the regular stall (desktop drawing, INFERRED: the ground truth is not
  known). So for a region of C cycles the rule misses any disturbance smaller than about 0.5 x C + 64 cycles; the
  STUDY-PLAN rule's 5,000-cycle limit for judging single samples follows from the 3,300 to 4,300-cycle stall.
- Label: MEASURED (rates, recall), INFERRED (the Q difference and the small events). Verdicts `spin_detector`,
  `spin_events_seen`, `spin_inject_recall` PASS; `spin_no_rt` and `spin_false_flags` INFO.

## LD-2, LD-1a, LD-1b, LD-1d, LD-3a, LD-3b. LDS: counter limit, misaligned access, atomics, swizzle, stores, 96-bit

Probe: `src/study/left_lds.hip` (new; the draft planned modes in `lds_probe.hip`), one wave alone. Each timed region
is one inline-assembly statement: counter read, a loop of 8 iterations of R copies of 16 identical accesses, the
final `s_wait_dscnt 0`, counter read; the lane addresses are data in v1 (so one kernel per instruction serves every
pattern); cost per access = (R = 6 - R = 2) / (4 x 16 x 8), R = 4 midway. LDS is filled before timing; a kernel
reads it after the timing so the compiler keeps the 64 KiB allocation (a first build had none: the compiler had
removed the fill and the allocation, and every load read out of range at 1 cycle; the listing check now fails any
timed kernel with less than 64 KiB, and that run's timings were thrown away). Listing check: every timed region
holds exactly the source's block (`scripts/leftc-expect/left_lds.txt`), 0 mismatched. Correctness uses a separate
kernel that runs one instruction and dumps the returned registers and LDS. Calibration (verdict
`lds_banks_calib` PASS): `ds_load_b32` stride 32 = 32.00, stride 2 = 2.00, sequential 1.00, stride 33 1.00, and
`ds_load_b128` sequential 6.00 (lds item 2). MEASURED, run `runs/20261117T203911Z-leftc-a`, 15 samples per point; every cost below is
the median and the spread across samples was at most +-0.05 cycles except where a range is given.

**LD-2: does the 64th outstanding LDS load stall the issue?** Expected (READ): the DS counter has 6 bits (63). The
draft predicted the step at load 64 for slow loads.

| N loads issued with no wait | 1 | 8 | 16 | 17 | 18 | 19 | 20 | 22 | 24 | 32 | 63 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ds_load_b128` stride 8 (32 cycles each): cycles to issue all | 9 | 16 | 24 | 25 | 68 | 69 | 132 | 196 | 260 | 516 | 1,477 | 1,540 | 2,564 |
| `ds_load_b128` sequential (6 cycles each) | 9 | 16 | 24 | 25 | 33 | 34 | 45 | 57 | 69 | 117 | 298 | 309 | 501 |
| `ds_load_b32` stride 32 (32 cycles each) | 9 | 16 | 24 | 25 | 26 | 27 | 28 | 30 | 76 | 332 | 1,293 | 1,356 | 2,380 |

The data arrive 32.00 (stride 8) and 6.00 (sequential) cycles apart per extra load between N = 32 and 96 (verdict
`lds_inflight_calib` PASS, lds item 2's costs). Answer: with these slow loads (32 or 6 cycles each) the wave issues
LDS loads at one per cycle until about 17 128-bit loads (22 32-bit loads) are waiting, then each further load waits
for one to leave: the issue time grows by the access cost per load (in steps of two loads). Nothing changes at
N = 63 or 64: the 63-entry DS counter was not the limit in any of the three series. MEASURED (15 samples per point,
spread within +-0.05 cycles).

Scope: all three series use slow loads. With conflict-free 32-bit loads, which leave at one per cycle, the earlier
LDS lane saw the wave issue one load per cycle up to N = 96 with no stop (`measurements/lds-and-special-loads.md`,
the table above its "Waits the compiler inserted" note: 66.3 cycles at N = 16 to 150.9 at N = 96, one cycle per
load, a step of about 5 cycles at N = 64). So the stop depends on how fast the loads drain; a fixed queue size is
not shown (INFERRED: some queue in front of the LDS that fast loads never fill; its size, if it is one, is not the
same count for 32- and 128-bit loads, so it is not a byte count either). For code: with slow (conflicting or 128-bit)
loads, more than about 16 to 20 LDS loads issued ahead of their use stall the issue; with fast loads that limit was
not seen.

Link to the earlier LDS lane: that file left the DS counter limit open ("could not be shown with conflict-free
loads", and "a test with deliberately conflicting 128-bit loads would show it" under its "Not measured"). LD-2 is
that test. It did not show the counter limit; it found the issue stop above, which comes long before 63 loads are
outstanding. This counter is the LDS one (DS); the other lane's "63 transposing global loads" concerns the counter
of global loads, a different counter.

**LD-1a: misaligned loads.** Expected (READ, guide 3.3.4 and 3.3.5.1): in UNALIGNED mode any byte address works;
otherwise the address is masked to 4, 8 or 16 bytes.

| Load (lane address = lane x element + offset) | +0 | +1, +2, +3 | +4 | +8 |
|---|---|---|---|---|
| `ds_load_b32`: bytes / cycles | right / 1 | byte-exact / 32 | right / 1 | right / 1 |
| `ds_load_b64` | right / 2 | byte-exact / 32 | byte-exact / 32 | right / 2 |
| `ds_load_b96` (16-byte element) | right / 4 | byte-exact / 32 | byte-exact / 32 | byte-exact / 32 |
| `ds_load_b128` | right / 6 | byte-exact / 32 | byte-exact / 32 | byte-exact / 32 |
| `ds_load_b96` / `ds_store_b96` packed at 12-byte spacing | 32 / 32 | | | |

Answer: ROCm runs the LDS in UNALIGNED mode: every misaligned load returns exactly the bytes at its address, in all
32 lanes (verdict `lds_misalign` PASS). But any access not aligned to its own size (96-bit: 16 bytes) costs 32
cycles: one lane per cycle, whatever the bank pattern. MEASURED. So a 64-bit load at a 4-byte offset is 16 times
slower than aligned, and 96-bit data packed at 12 bytes is 8 times slower than padded to 16.

**LD-1b: atomics.** Requirement: 32 lanes adding 1 to one word give +32, returned values distinct.

| Pattern | `ds_add_u32` | `ds_add_rtn_u32` (issue to data) | `ds_max_u32` | `ds_cmpstore_b32` | `ds_load_b32` |
|---|---|---|---|---|---|
| all lanes one word | 64 | 64 (96) | 64 | 64 | 1 |
| own word, sequential | 2 | 2 (34) | 2 | 3 | 1 |
| stride 2 / 4 / 8 / 16 / 32 dwords | 2 / 4 / 8 / 16 / 32 | the same (35 / 37 / 41 / 49 / 65) | the same | 4 / 8 / 16 / 32 / 32 | 2 / 4 / 8 / 16 / 32 |

Answer: every result is right for every pattern (sums, returned sets, max, compare-and-store; verdict `lds_atomic`
PASS). On one word the returned values are 0, 1, ..., 31 in lane order, and the compare-and-store winner is lane 0:
the lanes are served in lane order, two cycles each (64 for 32 lanes). Otherwise an atomic costs what a store costs
(at least 2 cycles) and follows the load's bank model; compare-and-store costs about twice (two words per lane
read: data and compare value, INFERRED). Operand order of `ds_cmpstore_b32 vaddr, v_new, v_cmp` confirmed (the word
takes the first data operand when it equals the second). MEASURED.

**LD-1d: swizzle and bpermute.** Expected (READ, guide `DS_SWIZZLE_B32`): bit-mask mode `j = ((i & and) | or) ^
xor`, quad-permute mode 2 bits per lane; rotate and FFT modes described without a formula that could be checked.

- Seven offsets (swap 1, reverse 32, broadcast lane 5, a mixed and/or/xor, swap 16, two quad permutes): the source
  lanes equal the guide's formula in all 32 lanes (verdict `lds_swizzle` PASS). Rotate mode, offset 0xc07f (amount 3,
  mask 0x1f as the guide's fields read): every lane got its own value; LLVM's `swizzle(ROTATE,0,5)` encoding 0xc0a1:
  sources 4, 7, 6, 9, 8, ... (not a plain rotation). FFT 0xe005: sources 0, 5, 2, 7, 5, 5, 7, 7, repeating every 8.
  These two modes are recorded, not explained (INFO).
- Cost: every swizzle 1.00 cycle per access, 32 cycles issue to data (like a 32-bit LDS load).
- `ds_bpermute_b32`: 3.00 cycles per access and 36.00 issue to data for every index pattern (identity, all from lane
  0, strides 2 to 32, reverse, random): it does not depend on the pattern (verdict `lds_bperm_calib` PASS: 36 as
  execution-core item 2). Execution-core's 5.06 per access was measured with 16 different data registers in a
  compiler-allocated block; with fixed registers here it is 3.00 (INFERRED: that block's register choice; not
  re-measured).

**LD-3a, LD-3b: stores, 96-bit loads under conflicts.** Expected: loads follow the bank model (lds item 2); stores
and 96-bit: unknown.

| Pattern (element = access size; 96-bit at 16 bytes) | b32 load / store | b64 load / store | b96 load / store | b128 load / store |
|---|---|---|---|---|
| same address | 1 / 2 | 2 / 3 | 4 / 5 | 6 / 5 |
| sequential, stride 3, stride 33 | 1 / 2 | 2 / 3 | 4 / 5 | 6 / 5 |
| stride 2 | 2 / 2 | 4 / 4 | 8 / 8 | 8 / 8 |
| stride 4 / 8 / 16 / 32 | 4 / 8 / 16 / 32 | 8 / 16 / 32 / 32 | 16 / 32 / 32 / 32 | 16 / 32 / 32 / 32 |
| random 0 / 1 / 2 | 4 / 4 / 3 (both) | 6 / 7 / 5 (both) | 9 / 13 / 11 (both) | 9 / 13 / 11 (both) |

- Every point fits one rule: the number of passes of the 32-bank model, applied to the whole wave for 32-bit
  accesses, to each half-wave for 64-bit, to each quarter-wave (8 lanes) for 96- and 128-bit, with a floor per form:
  loads 1, 2, 4, 6 cycles, stores 2, 3, 5, 5 (the floors are lds item 1's conflict-free costs). Loads and stores
  follow the same model. 96-bit accesses behave exactly like 128-bit ones under conflicts (the random rows equal),
  apart from the lower floor. MEASURED. Stores show small non-linearities (one point at 1.75 cycles per iteration off
  the midpoint, others at 0.88; loads none), reported, not judged.
- Lanes storing to one address: the highest lane's value is left (lane 31, for 32-, 64- and 128-bit stores).
  MEASURED.

Label: MEASURED unless marked. Verdicts `lds_banks_calib`, `lds_inflight_calib`, `lds_misalign`, `lds_atomic`,
`lds_swizzle`, `lds_bperm_calib` PASS; `lds_issue_queue` INFO.

## EC-1 and EC-2a to EC-2e. Instructions not timed before

Question: the issue cost and latency of `v_permlane64_b32` (wave32), `v_s_{rcp,rsq,sqrt,log,exp}_f16`, `v_sqrt_f64`,
`v_rsq_f64`, `v_dot2_f16_f16`, `v_dot2_bf16_bf16`, the 64-bit integer add, `v_div_scale_f32` and `v_div_fmas_f32`.

Expected: (EC-1) READ: a no-op in wave32, data unchanged. (EC-2a) the FP32 twins measured in the same run
(execution-core: chain 19.0, independent 1.88). The others exploratory, with neighbours measured in the same run
(`v_rcp_f64`, `v_dot2_f32_f16`). Requirements for the checks: the add pair carries, and the FP32 division sequence
built from `v_div_scale`, `v_div_fmas` and `v_div_fixup` gives the correctly rounded quotient.

Probe: `left_core` mode `insn` (the draft planned rows in `core_valu.hip`; the fixed-register method here also fixes
the banks), one wave; `_c16` / `_c8` independent results, `_c1` one dependent chain. Chained pseudo-scalar forms
wait for the SGPR write before the next read, as the compiler emits them (`s_wait_alu depctr_va_sdst(0)`). The
inputs of every block are whatever the registers held: v16-v103 are only declared as clobbers in `left_core.hip`,
never set, so this table does not exclude a data-dependent fast path (test T3 below sets them). MEASURED, run
`runs/20261117T204427Z-leftc-a`, 45 samples per kernel, spread within +-0.03 cycles unless noted:

| Instruction | Independent (cycles per instruction) | Chained (cycles per link) |
|---|---|---|
| `v_permlane64_b32` (wave32) | 1.000 | 1.000 |
| `v_s_rcp_f16`, `v_s_rsq_f16`, `v_s_sqrt_f16`, `v_s_log_f16`, `v_s_exp_f16` | 1.875 each | 19.00 each |
| their FP32 twins | 1.875 each | 19.00 each |
| `v_sqrt_f64`, `v_rsq_f64` | 32.00 | 40.00 |
| `v_rcp_f64` (known: 32 / 40) | 32.00 | 40.00 |
| `v_dot2_f16_f16`, `v_dot2_bf16_bf16` | 1.000 | 6.000 |
| `v_dot2_f32_f16` (known: 1 / 6) | 1.000 | 6.000 |
| 64-bit add: `v_add_co_u32` + `v_add_co_ci_u32`, carry through VCC, 8 independent pairs | 2.44 then 2.25 (not linear, see below) | 2.50 (one chain) |
| the same with the carry in a different SGPR per pair (`_e64`, carry-out of the second to `null`) | 1.750 | - |
| `v_div_scale_f32` x16 (writes VCC) | 2.41 then 2.13 (not linear) | - |
| `v_div_fmas_f32` x16 (reads VCC) | 1.000 | - |
| `v_div_scale_f32` then `v_div_fmas_f32`, 8 pairs | 8.50 per instruction (17 per pair) | - |

Checks: `v_permlane64_b32` left all 32 lanes unchanged; 0xFFFFFFFF + 1 carried in all lanes; the division sequence
gave the correctly rounded IEEE quotient in 32 of 32 lanes over exponents from about 2^-50 to 2^+50, where the
approximate `v_rcp_f32` x numerator differs in 12 (so the check can fail). Verdict `ins_calib` PASS (FMA 1.00,
linear, all checks right).

Answer:
- EC-1: `v_permlane64_b32` in wave32 is a full-rate no-op: 1 cycle, result 1 cycle later, data unchanged. MEASURED.
- EC-2a: the FP16 pseudo-scalar transcendentals cost exactly their FP32 twins: 1.875 back to back (the
  SGPR-writing VALU rule of execution-core 8d), 19 cycles through a chain. MEASURED.
- EC-2b: `v_sqrt_f64` and `v_rsq_f64` cost what `v_rcp_f64` costs: one every 32 cycles, result after 40. MEASURED.
- EC-2c: the FP16- and BF16-result dot products cost what the FP32-result one costs: full rate, 6 cycles latency
  (one more than an FMA). MEASURED.
- EC-2d: there is no single 64-bit add on gfx1201 (`v_add_nc_u64` and `v_lshl_add_u64` are refused by the
  assembler: READ from `llvm-mc`). The pair costs 5 cycles through one chain, and 4.5 to 4.9 per pair when eight
  independent pairs share VCC (the VCC carry orders them); with each pair's carry in its own SGPR, 3.5 cycles per
  pair. MEASURED; the cost of the VCC form changes with how many pairs are in a row (2.44 per instruction from 2 to
  4 blocks, 2.25 from 4 to 6), so it has no single per-instruction cost. INFERRED: the VCC write by the first
  instruction and its read by the second (the SGPR-writing VALU of execution-core 8d) set it.
- EC-2e: `v_div_fmas_f32` (reads VCC) costs 1 cycle like an FMA. `v_div_scale_f32` (writes VCC) costs 2.1 to 2.4
  back to back (not linear), and a `v_div_scale` followed directly by a `v_div_fmas` that reads its VCC costs 17
  cycles per pair: the read waits for the VALU write of VCC. MEASURED. In the compiler's division sequence the two
  are separated by about 7 other instructions (the sequence above), which hides most of that wait (INFERRED; the
  sequence itself was checked for correctness, not timed).

**Test T3 (fix round): the FP64 and pseudo-scalar rows with known data.** Expectation written before the run: if
the costs above do not depend on the data, every block costs the same with every data set (within the sample
spread, +-0.03 cycles); a contradiction would be any block whose cost moves with the data. Probe:
`src/study/left_bank.hip` mode `insdata` (wave32): the 26 FP64 and pseudo-scalar blocks of the table (`v_sqrt_f64`,
`v_rsq_f64`, `v_rcp_f64` independent and chained; the five `v_s_*` in FP16 and FP32, independent and chained), plus
three FP64 blocks whose 16 instructions all read one fixed input pair `v[48:49]` (so the input never changes), each
in a kernel that first sets v16-v31, v48-v49, s16-s31 and s40 to one value, inside the timed statement but before
the first counter read (listing check: every timed region as written, 0 mismatched). Six data sets: 1.7, 1.0, 0,
+infinity, the smallest subnormal, a quiet NaN (FP64, FP32 or FP16 encodings). Control that the data reached the
instructions: after the timed loop the kernel copies out `v[16:17]` and `s16`; for every block with a fixed input
and every data set except the subnormal, the result must equal the CPU's value (sqrt, 1/x, 1/sqrt, log2, exp2;
within 1e-5, FP16 2e-3): 65 of 65 right. Calibration (verdict `insdata_calib`): `v_rcp_f64` with input 1.7 at 32.0
independent and 40.0 chained (EC-2 above); break: 16.0 expected: FAIL.

MEASURED, final run `runs/20261117T224107Z-leftc-a` (the same in `runs/20261117T223438Z-leftc-a`), 45 samples per
block and data set: every one of the 29 blocks cost
the same with all six data sets (verdict `insdata_costs` INFO: 0 of 29 differ by more than 0.05 cycles between data
sets; in the final run the costs are equal to four decimals for every block and data set; in the development run
`runs/20261117T223438Z-leftc-a` four chained pseudo-scalar rows differed by 0.015 cycles per instruction (15 cycles
in a region of about 19,500), in both directions: twice with NaN, once with infinity, once with 1.0; three rows
have 1 or 2 dropped samples of 45). The test has no positive control: no instruction with a known data-dependent
cost was run to show that the method would see one. The FP64 blocks cost 32.00 per instruction independent (with a fixed input and with the moving input of the
table) and 40.00 chained; the pseudo-scalar blocks 1.875 and 19.00, as in the table above. Two by-products of the
control: `v_sqrt_f64` has about 26 correct bits (`sqrt(1.0)` returned 1.0000000149, bits `3ff0000004000000`;
`sqrt(1.7)` returned `3ff4dc87d3200000` against the correctly rounded `3ff4dc87d61451da`; a first version of the
check expected correct rounding and failed, `runs/20261117T222442Z-leftc-a`), and the subnormal inputs were not
flushed (FP64 `sqrt(4.9e-324)` gave 2.2e-162). Chained rows move through the values their function produces (the
FP32 `exp2` chain from 1.7, for example, reaches infinity after 4 links and stays there), so they test the data only
loosely; the fixed-input rows test it exactly.

Answer (T3): no data dependence in these instructions' costs: zero, infinity, NaN and subnormal inputs cost the same
as ordinary ones. MEASURED.

Label: MEASURED (costs, checks), INFERRED where marked. Verdicts `ins_calib`, `insdata_calib` PASS; `ins_costs`,
`ins_nonlinear`, `insdata_costs` INFO.

## MS-5 and MS-2a. Translation reach, and whether the one-wave L2 limit is shared

Questions: (MS-5) what is the reach of address translation (entries and page size)? (MS-2a) is the one-wave L2 limit
(a step at about 4,000 to 11,000 lines, memory-system 2b-2) per wave or shared by all waves?

Expected: unknown (exploratory). Known points used as checks: L0 chase 81 cycles; one wave's L2 plateau 231-256
cycles (memory-system 2b; tim_l2 233 at 2,048 nodes of 4 KiB); 463 at 2,048 nodes of 64 KiB (tim_l2). A translation
cache of E entries of P bytes makes a chase step at N = E nodes once the spacing is at least P.

Probe: `left_mem` modes `tlb` and `multi` (the draft planned them in `tim_l2.hip`; same method, kept with the other
memory items). Each point: the chain is built in random order, one full cycle of warm-up and 8,192 timed steps in the
same launch, median of 5 launches (tlb) or of 3 chain seeds x 3 launches (multi), final pointers checked (all
right). `multi` runs W one-wave groups at once in CU mode with 40 KiB of LDS each (one group per CU; every launch put
each wave on its own CU, 2 to 4 shader engines), each chasing its own region at 256-byte spacing.

MS-5, MEASURED, run `runs/20261117T205135Z-leftc-a` (the run before it, `runs/20261117T204955Z-leftc-a`, gave the
same steps), cycles per dependent load:

| Spacing | 256 nodes | 512 | 1,024 | 1,152 | 2,048 | 3,072 |
|---|---|---|---|---|---|---|
| 4 KiB | 81 | 232 | 233 | - | 247 | 227 |
| 8 KiB | 81 | 248 | 239 | - | 238 | 248 |
| 16 KiB | 81 | 244 | 235 | - | 235 | 445 |
| 32 KiB | 81 | 233 | 238 | - | 240 | 449 |
| 64 KiB | 81 | 252 | 256 | 461 | 447 | - |
| 128 KiB | 81 | 235 | 230 | 461 | 475 | - |
| 256 KiB | 81 | 251 | 273 | 449 | 455 | - |
| 1 MiB | 244 | 248 | 284 | 462 | 465 | - |
| 2 MiB | 234 | 246 | 378 | 386 | 466 | - |

(The 81-cycle column is L0: up to 256 lines fit its 32 KiB; at 1 and 2 MiB spacing L0 sets conflict. Calibration
verdict `mem_tlb_calib` PASS: 80.7, 247 and 447 cycles.)

Answer (MS-5):
- At every spacing from 64 KiB to 1 MiB, 1,024 nodes run at L2 latency and 1,152 nodes miss on every step (450 to
  460 cycles, about 210 more); the fine steps between (1,152 to 1,920 nodes, 64 to 256 KiB spacing) all miss. At
  32 KiB spacing 2,048 nodes still hit and 3,072 miss, so two nodes 32 KiB apart share one entry. MEASURED. The
  step is bracketed only between 1,024 and 1,152 nodes (no point between), and it holds for spacings of 32 KiB or
  more; at 16 KiB the slowdown comes at 3,072 nodes (48 MiB) and at 2 MiB already at 1,024 nodes (378 cycles), both
  not explained (below).
- INFERRED: a translation cache of about 1,024 entries, each covering 64 KiB (so pages, or page fragments, of 64 KiB
  under ROCm), a reach of about 64 MiB for one wave; a miss with the data in L2 costs about 210 cycles. A cyclic
  walk over more entries than it holds misses every time (LRU-like). At 2 MiB spacing 1,024 nodes already miss partly
  (378): INFERRED set conflicts when the address bits above 64 KiB that select the set do not vary.
- Not explained: 16 KiB spacing misses at 3,072 nodes (768 regions of 64 KiB, inside the reach), in both runs; 3,072
  nodes at 4 and 8 KiB spacing do not. Memory-system 2b-2's second rise at 4 KiB spacing (between 120 and 165 MiB)
  lies above the 64 MiB found here; there the data already came from beyond L2, which may hide part of the miss
  (INFERRED).

MS-2a, MEASURED, run `runs/20261117T204955Z-leftc-a`, median wave of W, cycles per load (lines in all = W x nodes):

| Nodes per wave | 2,048 | 2,880 | 4,032 | 5,632 | 7,872 | 11,008 | 15,424 | 21,568 |
|---|---|---|---|---|---|---|---|---|
| W = 1 | 244 | 252 | 234 | 320 | 355 | 398 | 340 | 472 |
| W = 2 | 249 | 251 | 298 | 256 | 384 | 406 | 333 | 460 |
| W = 4 | 246 | 250 | 322 | 253 | 330 | 455 | 459 | 459 |
| W = 8 | 244 | 246 | 263 | 454 | 453 | 453 | 454 | 452 |
| W = 16 | 246 | 451 | 450 | 451 | 452 | 454 | 457 | 612 |

The same job in the lane's final rerun and in the lead's rerun (`left_mem-multi.log` of each; "step" is the
first node count at 1.3 x the lowest value of the row, as the probe prints it):

| Run | Step per wave, W = 1 / 2 / 4 / 8 / 16 (nodes) | Lines in all at the step, W = 8 / 16 |
|---|---|---|
| `runs/20261117T204955Z-leftc-a` (the table above) | 5,632 / 7,872 / 4,032 / 5,632 / 2,880 | 45,056 / 46,080 |
| `runs/20261117T215755Z-leftc-a` (lane's final rerun) | 7,872 / 7,872 / 11,008 / 5,632 / 2,880 | 45,056 / 46,080 |
| `runs/20261117T220137Z-leftc-a` (lead's rerun) | 7,872 / 7,872 / 7,872 / 5,632 / 2,880 | 45,056 / 46,080 |

Answer (MS-2a): two limits show.
- A total near 45,000 lines: eight waves step at 5,632 nodes each and sixteen at 2,880 in all three runs (45,056
  and 46,080 lines in all, 449 to 458 cycles after the step), and W = 4 at 11,008 nodes (44,032 lines) is at 455 to
  458 in all three. Below that total, eight waves at 4,032 nodes each (32,256 lines) read at 247 to 263 cycles.
  MEASURED. That total lies above the 32,768 256-byte lines of the 8 MiB L2; how much of each 256-byte node L2 keeps
  was not measured, so whether this total is L2's capacity is not settled (INFERRED either way).
- A one-wave limit around 4,000 to 11,000 nodes (the fix round's final run steps at 4,032 for W = 1: 318.8 cycles
  there, back to 254 at 5,632): one, two or four waves slow down well before that total (W = 1 at
  5,632 or 7,872 nodes, W = 2 at 7,872, W = 4 at 4,032 to 11,008), with wide spreads inside a point (for example
  W = 1 at 7,872 nodes: 241 to 465 cycles over the 9 samples of one run). So the step position for few waves moves
  from run to run, as in memory-system 2b-2. MEASURED.
- So a single wave does not get the whole L2 at L2 latency, and many waves together do. Whether the one-wave limit
  belongs to the wave, its CU or its shader array was not separated (the dispatcher spread the waves over 2 to 4
  shader engines). INFERRED, and "per wave" stays a reading, not a result.
- Not explained, in all three runs: one and two waves read faster at 15,424 nodes than at 11,008 (W = 1: 340, 352,
  336 against 398, 416, 413 cycles).

Verdict `mem_multi_calib` PASS in all three runs (W = 1 steps inside 2,867 to 15,421 nodes); `mem_multi_rule` INFO.

Label: MEASURED (tables), INFERRED (entry count and size, per-wave reading).

## LD-4. How long until another wave sees a store

Question: how long from one wave's store to another wave's load seeing it, through LDS (same SIMD, other SIMD of the
CU, other CU of the WGP) and through global memory between two CUs?

Expected: a lower bound from lds item 1: a store then a load in one wave takes 36 cycles, so a round trip between
two waves cannot be shorter than 72. The rest exploratory.

Probe: `src/study/left_comm.hip`, built in CU mode and in WGP mode. Wave A stores 2i+1 and polls until it reads 2i+2;
wave B polls for 2i+1 and answers 2i+2; 20,000 round trips, A's cycle counter and REALTIME, median of 5 launches.
The poll loops are inline assembly (load, wait, `v_readfirstlane_b32`, wait for the SGPR, compare, branch, with a cap
of 2 million polls): a first version written in C++ compiled into a long scalar loop and gave 1,550 cycles, which
measured the loop, not the hardware. The same loop run by one wave on its own store gives the loop's floor. LDS pairs
are picked by `HW_ID1` from a group of 8 waves; the global pair is two one-wave groups with 40 KiB of LDS each (on
different CUs, here on different shader engines), device-scope loads and stores (`scope:SCOPE_DEV`). Calibration:
a store then a load of the same word, the value plus one stored next: 53.7 cycles per link. The expected 53 is not a
documented value: it is lds item 1's measured 36 (store then load) plus 17, derived from its 49-cycle
load-add-address chain (49 - 32 = 17 for the add between an LDS load and the next LDS operation). INFERRED as an
expectation; it matched to 0.7 cycles.

MEASURED, run `runs/20261117T205831Z-leftc-a` (CU-mode and WGP-mode builds agree within 1 cycle):

| Path | Round trip (cycles) | ns | One wave on its own store, per step |
|---|---|---|---|
| LDS, both waves on one SIMD | 272.0 (272.0-273.4) | 97 | 110.0 |
| LDS, the two SIMDs of one CU | 272.1 | 86 | 110.0 |
| LDS, the two CUs of one WGP (WGP mode) | 277.4 | 86 | 110.0 |
| global word, device scope, two CUs on different shader engines | 912 (892-952) | 287 | 326 (320-338) |

The global row in the two reruns (`left_comm-global.log`): 895 (882-937) cycles, 274 ns, own-store loop 330, in the
final rerun `runs/20261117T215755Z-leftc-a`; 891 (871-910), 273 ns, own-store 329, in the lead's
`runs/20261117T220137Z-leftc-a`. So the global round trip ranges over 871 to 952 cycles across the three runs.

Answer: through LDS a round trip costs 272 cycles, about 136 per direction, where the same loop reading its own store
costs 110: another wave sees an LDS store about 26 cycles later than the storing wave itself would, whether it is on
the same SIMD, the other SIMD or (5 cycles more) the other CU of the WGP. Through global memory at device scope a
direction costs about 450 cycles against 326 for the own-store loop (L2 round trip): about 125 cycles more to reach
another CU. No timeout in any run. MEASURED. Same-SIMD and other-SIMD pairs take the same cycles, but the same-SIMD
pair's time in ns is longer (97 against 86 ns): the clock was lower in those launches (INFERRED from cycles / ns;
the clock was not logged per launch).

Label: MEASURED. Verdict `comm_bound` PASS (calibration 53.7, every round trip above 65 cycles, no timeout).

## EC-5, EC-6, EC-8. The timing oddities: register-bank stall, runs of 12-byte instructions, SGPR writes

Method: `left_core` modes `bank`, `run12`, `sgprw` (one wave, fixed-register blocks; the draft planned new modes in
`tim_core.hip`, the blocks are in `left_core_blocks.h` instead). All three are exploratory: the verdicts judge the
known values of execution-core items 3 and 8, the findings are INFO. MEASURED, runs
`runs/20261117T205944Z-leftc-a` (bank, run12) and `runs/20261117T210205Z-leftc-a` (sgprw).

**EC-5: the 5-cycle bank case.** Question (draft question 4): is the 5-cycle cost of an FMA reading three different
registers of one bank (execution-core 8g) a stall of that instruction, or something shared with the next one? My
reading of execution-core item 3, settled from its table: "5" is the cost of such an instruction when the one before
it also read a different triple of the same bank (9.999 per pair of two different bank-1 triples); the same triple
repeated costs 2.968 per pair. Calibration (verdict `bank_calib` PASS): 3.000, 10.000 and 8.000 per pair against
2.968, 9.999 and 7.999.

| X = FMA reading three bank-1 registers, then k other FMAs (period k + 1) | k = 0 | 1 | 3 | 7 | 15 |
|---|---|---|---|---|---|
| X with a different triple each time: cycles per period | 5.0 | 5.0 | 5.0 | 9.0 | 16.0 (one X per block: the same triple each time) |
| X with the same triple each time | 1.5 | 2.0 | 4.0 | 8.0 | 16.0 |
| the fillers reading two SGPRs instead of one VGPR (k = 1, 3; different / same triple) | 5.0 / 2.0 | | 5.0 / 4.0 | | |

Answer: a period costs max(k + 2, 5) with different triples: such an instruction costs one cycle more than an
ordinary one, and the next one of its kind cannot start until 5 cycles after it; ordinary instructions in between
fill those cycles. With the same triple repeated there is no cost once anything sits between (k >= 1), and half a
cycle when back to back. So it is a shared limit (a recovery time of the bank, or a small operand cache that the
repeated triple hits; INFERRED), not a fixed stall of each instruction. MEASURED (the 8-wave case of the draft was
not run: execution-core 8g already shows other waves do not fill those cycles).

**EC-8: the fractions 1.875 and 1.625.** n consecutive SGPR-writing VALU instructions, then m FMAs (rotating over 16
registers, so independent), cycles per period (n + m instructions):

| n | m = 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| `v_readfirstlane_b32` / `v_s_exp_f32`, 1 | 1.875 | 2.0 | 3.0 | 5.0 |
| 2 | 3.75 | 3.75 / 3.79 | 4.0 | 6.0 |
| 3 | 5.63 | 5.88 / 6.12 | 6.13 / 6.37 | 7.0 |
| 4 | 7.5 | 7.5 | 7.5 | 8.0 |
| 8 | 15.0 | 15.0 | 15.0 | 15.0 |
| `v_cmp_gt_f32_e64` to an SGPR, 1 / 2 / 4 / 8 (m = 0) | 1.625 / 3.25 / 6.5 / 13.0 | | | |

Destination patterns (16 writes to one SGPR, to even SGPRs, alternating two) cost the same as 16 different ones.
Answer: the cost is a throughput, not a per-run effect: the path from the VALU to the SGPRs takes 8 writes per 15
cycles (`v_readfirstlane_b32`, `v_s_exp_f32`) or 8 per 13 (`v_cmp` to an SGPR), and independent FMAs fill the idle
cycles, so a period costs max(n + m, 1.875 n) (1.625 n for compares): 57 of 63 points within 0.05 cycles. The six
misses are all runs of 3 with fillers (0.25 to 0.75 cycles per period more), and the runs of 3 without fillers are
not linear in the block count, so short odd runs have some extra rule not found here. The draft's "first free, each
later one 2 cycles" model does not fit (it predicts 1.9375 for 16). MEASURED (rates), INFERRED (the write-path
reading). Verdict `sgprw_calib` PASS.

**EC-6: runs of 12-byte instructions.** L consecutive 12-byte `v_fma_f32` (real literal) between G 8-byte ones,
cycles per instruction (calibration `run12_calib` PASS: all six known values of execution-core 8a within 0.0001):

| L \ G | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | 1.008 | 1.154 | 1.039 | 1.180 | 1.235 |
| 2 | 1.170 | 1.009 | 1.154 | 1.039 | 1.180 |
| 3 | 1.131 | 1.046 | 1.006 | 0.992 | 1.105 |
| 4 | 1.205 | 1.170 | 1.126 | 1.083 | 1.049 |
| 8 | 1.227 | 1.205 | 1.170 | 1.125 | 1.083 |
| 12 | 1.236 | 1.218 | 1.190 | 1.153 | 1.109 |
| 16 (all 12-byte: 1.258) | 1.227 | 1.227 | 1.204 | 1.169 | 1.127 |

(L = 1, G = 3: 1.001.) Answer: not settled. From L = 4 on the cost falls smoothly as G grows, as a fetch limit of a
little under 12 bytes per cycle with a buffer predicts. For short runs it is not monotonic: one 12-byte instruction
among 16 short ones costs 1.235 per instruction, more than runs of 4 among 16 (1.049), and L = 2 alternates between
about 1.01 and 1.17 with G. Two models were fitted to all 42 points and neither fits: a fetch of F bytes per cycle
into a Q-byte buffer (best F 8.25, Q 20: largest error 0.235 cycles per instruction) and a fixed cost per instruction
that crosses a 16-, 32- or 64-byte boundary, using the real code addresses (largest error 0.16). INFERRED: where the
12-byte instructions land relative to the fetch blocks matters in a way neither model captures. MEASURED (table).

## EC-7. What holds the code of a loop (the 4 KB step)

Question: does the step at 4 KB of loop body (execution-core 8b: 20-21 cycles of loop control per iteration below
4 KB, 34 at 4 and 5 KB, bimodal from 6 KB) belong to the wave, the SIMD, the CU or the work-group processor?

Expected: unknown (exploratory). Known: one wave at 2 KB and 3 KB 20-21 cycles of loop control, at 4 KB 34.

Probe: `left_core` mode `fsize`, built in CU mode and in WGP mode (`left_core_wgp`). Loops of independent 8-byte FMAs;
loop control per iteration = cycles per iteration minus the instructions. Two 3 KB copies of the same loop sit at
different addresses in one kernel; the waves of one work-group run either all copy A, all copy B, or different
copies chosen by their placement (`HW_ID1`). A 4 KB loop is also started 0, 64, 128 and 2,048 bytes past a 4 KB
boundary (padding of `s_nop` before it, run once). MEASURED, run `runs/20261117T210816Z-leftc-a`, medians of 15
samples:

| Case (3 KB loop each unless noted) | Loop control per iteration (cycles) |
|---|---|
| one wave, 2 KB / 3 KB | 20.8 / 20.8 (WGP build 19.9 / 19.9) |
| one wave, 4 KB starting 0, 64, 128 bytes past a 4 KB boundary | 34.4, 34.4, 34.4 |
| one wave, 4 KB starting 2,048 bytes past it | 236 |
| one wave, 4 KB in the two-copy kernel (wherever the compiler put it) / 5 KB there | 127 / 473 |
| one wave alone on copy A / on copy B | 20.8 / 19.9 |
| two waves, one on each SIMD of a CU: both copy A / both copy B / different copies | 20.8, 20.8 / 19.9, 19.9 / 30.5, 30.0 |
| four waves, two per SIMD: first wave of each SIMD, same copy / different copies by rank | 20.9 / 47-53 |
| the same four waves: second wave of each SIMD, all copy A / all copy B / different copies | 404.8, 404.8 / 404.2, 404.2 / 404.6, 410.6 |
| WGP mode, one wave on each SIMD of both CUs: all copy A / all copy B | 19.9 x 4 / 21.0 x 4 |
| WGP mode, CU 0 on copy A and CU 1 on copy B | CU 0: 19.9, 19.9; CU 1: 40.6, 40.8 |

Answer:
- The cost of a loop of about 4 KB depends on where it starts: 34 cycles of loop control when it starts at a 4 KB
  boundary (to 128 bytes past it), 127 or 236 when it starts elsewhere. So execution-core's "4 KB step" is not a fixed
  size limit: the code's position relative to 4 KB boundaries decides it (INFERRED: a structure indexed by address,
  such as an instruction cache with 4 KB per way). MEASURED. Practical: keep timed loops well below 4 KB (the study's
  rule), and a 4 KB loop is cheap only when aligned to 4 KB.
- In this build, two different 3 KB loops on the two SIMDs of a CU slowed each other (30 against 20-21 cycles of
  loop control), the same 3 KB loop on both did not. MEASURED in this run and in the final rerun (30.4 / 30.1). But
  in the fix round's build, whose loops all sit 68 bytes further on (see test T2), the same case cost 20.8 / 20.8:
  no interference. So whether two loops on the two SIMDs of a CU interfere depends on where they sit, and "shared by
  the two SIMDs of a CU" holds only for some placements. MEASURED (two placements); the structure is not
  established.
- Across the two CUs of a WGP, different code also interferes, but only one side lost (CU 1: 41; CU 0 unchanged).
  With the controls (each copy alone on both CUs: 20-21) this is the interaction, not the placement of copy B in
  memory. MEASURED. The earlier reading, "the first CU's fetches served first", is contradicted by test T2: the side
  that loses follows the copy, not the CU.
- The four-wave rows: the second wave on each SIMD costs about 404 cycles of loop control in all three cases,
  controls included, against 20-21 for the first. Its iteration takes about 788 cycles (404 + 384 instructions),
  about twice the first wave's 405. INFERRED: the two waves of a SIMD share its issue, and the second wave gets most
  of its slots only after the first has finished (the first wave's figure is that of a wave alone); plain equal
  sharing would give both waves about 788. So the "different copies by rank" row (47-53) concerns the first waves
  only; the second waves on different copies lose little or nothing more (404.6 and 410.6 here, 406.8 and 407.1 in
  the final rerun, against 404.2 to 404.8 on one copy).

**Test T2 (fix round): does the slow side follow the CU or the copy?** Expectation written before the run: if
the first CU's fetches are served first, the side that loses stays CU 1 when the copies are swapped; if it follows
the copy, CU 0 loses when it runs copy B. Probe: `left_core_wgp fsize`, a fifth selection that puts copy B on
CU 0 and copy A on CU 1 (HW_ID1 bit 8), next to the existing "CU 0 copy A, CU 1 copy B" row, which is the control
that the test can see the effect (CU 1 lost in every earlier run). Run in the same job; break: the job's
calibration with the 2 KB loop expected at 40 cycles: FAIL.

A first attempt changed the object enough to move the loops: adding the fix round's kernels to `left_core.hip` moved
copy A across a boundary (3 KB alone: 41 cycles of loop control instead of 21, calibration FAIL, run
`runs/20261117T222442Z-leftc-a`), and placing every copy at a 4 KB boundary made it worse (one wave on a 2 KB loop:
198 cycles of loop control, 3 KB 277; run `runs/20261117T223054Z-leftc-a`): copies of the same loop at the same
offset inside 4 KB slow each other even when only one of them runs (MEASURED, not explained). The fix round's other
kernels then moved to `left_bank.hip`, so `left_core`'s object differs from the earlier one only by the selection
code; its loops sit 68 bytes later than before (`llvm-objdump`, the 12 loop copies of `k_fs<24>`: the compiler
unrolled the 6 repetitions, so each "copy" is 6 loops at different addresses).

MEASURED, WGP build, loop control per iteration per wave (CU 0 SIMD 0, CU 0 SIMD 1 / CU 1 SIMD 0, CU 1 SIMD 1):

| Run | Each copy alone on all four (A / B) | CU 0 on A, CU 1 on B | CU 0 on B, CU 1 on A (swapped) |
|---|---|---|---|
| final rerun `runs/20261117T224107Z-leftc-a` | 20.8-20.9 / 20.0-20.1 | 20.9, 20.8 / 40.6, 40.8 | 41.9, 41.9 / 20.9, 20.9 |
| `runs/20261117T223438Z-leftc-a` | 20.8-21.1 / 19.9-20.1 | 20.8, 20.9 / 41.2, 41.1 | 39.9, 39.8 / 20.8, 20.8 |
| `runs/20261117T222442Z-leftc-a` (the shifted object: copy A alone 41) | 42.2-42.4 / 20.0-20.1 | 52.5, 52.1 / 71.2, 71.4 | 71.5, 71.4 / 52.9, 53.1 |

Answer (T2): the waves on copy B lose, on whichever CU they run (CU 1 on B: 40.6-41.2; CU 0 on B: 39.9-41.9), and
the waves on copy A keep their own cost; in the shifted object too, the extra cost follows the copy (B +51, A +11
whichever CU). MEASURED in three runs and two placements. This contradicts the earlier inference that the first CU
is served first. Why copy B and not A is not established (INFERRED: where B sits relative to A).
- Label: MEASURED (table), INFERRED (structures). Verdict `fsize_calib` PASS (2 KB 20.8, 3 KB 20.8, 4 KB aligned
  34.4); the rest INFO.

## CD-1, CD-2, CD-4. Widening with dual-issue consumers, the LDS table with loads issued ahead, stochastic rounding with a generator

Probe: new blocks in `scripts/cvt_gen_timing.py` (`src/study/cvt_time_blocks.h` regenerated; the merged `cvt_time`
probe gains them too, and the shared initialisation and clobber list of every block grow by v138-v143: six more
`v_mov_b32` before the timed loops, outside them). Fix round: the merged `cvt_time` was rebuilt through the script
that owns it (`scripts/study-conversions.sh build`, run `runs/20261117T223759Z-conv`): the header matches its
generator, `cvt_time` builds, and its timed loops pass the loop check (1,536 loops, 0 not clean, 0 kernels whose
added instructions change with U). It was not rerun on the card: the timed block texts of the merged tests are
unchanged and the change lies outside the timed loops; `left_cvt` runs the same header and reproduces the
conversions figures it shares with `cvt_time` (FMA 1.000 / 5.000, widening (a) 1.000, (a) with `v_fmac_f32` 2.406
against 2.41, `v_cvt_sr_fp8_f32` 1.000 / 5.000). The blocks are timed by a new driver `src/study/left_cvt.hip` with `cvt_time`'s method (lane 1's U = 4, 6,
8 differential, U = 6 midway) and its own verdicts; the timed loops pass lane 1's loop check (`core_loopcheck.py`,
456 loops clean). Calibration in the same run (verdict `cvt_calib` PASS): `v_fma_f32` 1.000 / 5.000, widening (a)
alone 1.000 per weight, (a) with `v_fmac_f32` consumers 2.406 (conversions item 4: 2.41), `v_cvt_sr_fp8_f32`
1.000 / 5.000. MEASURED, run `runs/20261117T212039Z-leftc-a` (and the four runs before it, the same within 0.01
except where noted).

| Form | Cycles per weight (or conversion) | Draft's expectation |
|---|---|---|
| (a) 4 x `v_cvt_f32_fp8` + 4 `v_fmac_f32` | 2.406 | known 2.41 |
| (a) + 2 `v_dual_fmac_f32` pairs | 2.188 | 1.45 to 2.41 |
| (b) 2 x `v_cvt_pk_f32_fp8` + 2 `v_dual_fmac_f32` pairs | 2.188 | 1.45 to 2.41 |
| (e) FP16 table in LDS, simple (one wait per 32 loads) | 4.59 to 4.63 in six development runs, the final rerun (4.624) and the lead's (4.626); 5.15, 5.81, 5.91 in three development runs (bimodal samples) | known 4.59 |
| (e) loads issued ahead (two half-blocks, each waiting only for the other's loads) | 5.48 to 5.54 in eight development runs, the final rerun and the lead's; 6.69 in one | 2.95 to 4.2 |
| `v_cvt_sr_fp8_f32` + xorshift32 (6 VALU) | 7.252 (sources in one bank: 7.500) | 7 |
| + 24-bit LCG (`v_mul_u32_u24`, `v_add_nc_u32`) | 3.000 (3.250) | 3 |
| + 32-bit LCG (`v_mul_lo_u32`, `v_add_nc_u32`) | 6.124 (6.250) | 6 |
| + one shift per conversion (one random word for four) | 2.000 (2.250) | 2 |

Answer:
- CD-1: with the consumers as dual-issue pairs (registers chosen so every VOPD bank and parity rule holds) a widened
  weight costs 2.19 cycles instead of 2.41, for both widening forms; the issue bound is 1.5. So most of the gap to the
  bound is not the FMAs' issue: the 32 conversions and 16 pairs of a block take 70 cycles for 48 instructions.
  MEASURED; INFERRED: the conversions' results feeding the pairs (latency and banks), not established.
- CD-2: issuing the table loads ahead does not help: 5.5 per weight against 4.6 for the simple form in its fast
  mode. The simple form has a slow mode (5.15 to 5.91 in 3 of the 9 development runs counted from the logs of
  `runs/20261117T2112*` to `runs/20261117T2121*`, one value per process; single samples up to twice the median). In
  its slow mode it was slower than the ahead form in two of the three runs (5.91 against 5.54, 5.81 against 5.49)
  and faster in one (5.15 against 5.54), so "slower" holds only against the simple form's fast mode. The LDS table
  method stays the slowest way to widen FP8 on this card. MEASURED. Why the ahead form is slower: test T4 below
  removes one candidate. The candidates were (INFERRED): the `ds_load_u16_d16` / `_hi` pairs write halves of the same
  registers, so the loads of one half-block wait for each other; or the issue stop of LD-2, since the ahead form
  keeps up to 32 LDS loads outstanding (16 of each half), above the 17 to 22 at which LD-2's wave stopped issuing.

**Test T4 (fix round): fewer loads in flight.** Expectation written before the run: if LD-2's issue stop is what
makes the ahead form slower, a form that keeps at most 16 loads in flight is faster than the ahead form (32 in
flight), towards the simple form or below it; a contradiction would be the 16-in-flight form at or above the ahead
form. Probe: `use_e_lds_ahead8` (`scripts/leftc_gen_cd2.py`, header `src/study/left_cd2_blocks.h`, the same
instructions and registers as the ahead form): the block's 8 words in four quarters of 2 words; each quarter issues
its 8 loads, waits until at most 8 are outstanding (`s_wait_dscnt 8`) and consumes the previous quarter. `left_cvt`
mode `cd2` times the simple, the ahead and this form interleaved, 12 rounds per process, three processes (jobs
`cd2`, `cd2 b2`, `cd2 b3`): 36 samples per form, each sample one U = 4, 6, 8 differential. Listing check: lane 1's
loop check (`core_loopcheck.py`) on every timed loop, 0 not clean; the generated header matches its generator.
Control (verdict `cd2_bound`): FMA 1.00 in the same process, every form at or above its issue bound 3.45 per
weight, the table at LDS address 0; that the test can resolve the forms is shown by the simple and ahead forms
themselves (0.8 per weight apart in every process). Break: the new form claimed at 1.00 per weight: FAIL.

MEASURED, final run `runs/20261117T224107Z-leftc-a` (the development run `runs/20261117T222442Z-leftc-a` gave
medians 4.69, 5.46-5.52, 7.19-8.46), cycles per weight, 36 samples per form:

| Form | Loads in flight at most | Median | Main group (samples) | Slow samples |
|---|---|---|---|---|
| simple (one wait per block) | 32, all waited for before use | 4.689 | 4.62-4.71 (29) | 5.32-6.10 (7; 2, 0 and 5 per process) |
| ahead (two halves) | 32 | 5.503 | 5.35-5.52 (25) | 6.00-6.05 (3), 6.83-6.95 (8) |
| ahead8 (four quarters) | 16 | 7.187 | 7.00-7.19 (25) | 7.57-7.79 (3), 8.44-8.71 (8) |

Answer (T4): keeping at most 16 loads in flight makes the method slower, not faster: 7.19 against 5.50 per weight.
This contradicts LD-2's issue stop as the cause of the ahead form's cost; what the shorter look-ahead shows instead
(INFERRED) is the load latency that it no longer hides: each quarter now waits for loads issued only one quarter
earlier. The half-register candidate is untouched (all three forms use the same `_d16` / `_d16_hi` pairs). Also
seen: the simple form's slow mode shows inside one process too (2 and 5 of 12 samples in two processes, 0 in the
third), so it is not a per-process state; the ahead forms have slow samples as well (11 of 36 each).
- CD-4: the cost is additive and issue-bound once the conversion's value and random operand sit in different register
  banks: 24-bit LCG 3.000, shift 2.000, 32-bit LCG 6.12, xorshift 7.25 (the last two 2 and 3.6 % above the sum). With
  both sources in one bank every conversion costs a quarter cycle more (one extra cycle per four, on top of
  execution-core item 3's finding that two sources in one bank are free for an FMA). Cheapest generator: one random
  word shifted for four conversions, 2 cycles per conversion. MEASURED.
- Verdicts: `cd1_dual`, `cd2_ahead`, `cd4_sr` PASS (each at or above its issue bound); the draft's ranges are INFO:
  CD-1 inside, CD-2 outside (slower than the simple form), CD-4 3 of 4 within 3 % of the sum.

## CD-3. Combined modifier bits on the dot products, and `neg_hi` on the integer forms

Question: what do `neg_hi` on `v_dot4_i32_iu8` and `v_dot8_i32_iu4`, and combinations of the accepted bits on
`v_dot2_f32_f16`, do? Expected: for `v_dot2_f32_f16`, the composition of the single-bit effects measured in
conversions-and-dots 5a (element k of a source reads the half chosen by `op_sel` (low element) or `op_sel_hi` (high
element) and is negated by `neg_lo` or `neg_hi`; C: `neg_lo:[0,0,1]` flips its sign, `neg_hi:[0,0,1]` takes |C|). For
the integer forms two candidates: `neg_hi` ignored (as on the matrix integer forms, matrix-arithmetic 9a), or `neg_hi`
acting as a second sign setting (OR-ed into `neg_lo`).

Probe: `src/study/left_dotm.hip` (the draft planned rows in the helper-written `left_dot.hip`; a separate file kept
that program unchanged), with the variants in `src/study/left_dotm_variants.h`: for `v_dot2_f32_f16` all 16
combinations of `neg_lo`, `neg_hi`, `op_sel` and a cleared `op_sel_hi` on A, the same on B, and 32 random combinations
on A, B and C (64 in all; the assembler accepts every one when `op_sel`/`op_sel_hi` are written before `neg_lo`/
`neg_hi`, and refuses the other order with "not a valid operand"); for the integer forms every `neg_hi` setting on A,
B and C beside every `neg_lo` setting (32 cases). Each variant runs on 131,072 rows of random finite data and is
compared bit for bit with the plain instruction on data changed by the model, so no model of the adder is needed.

MEASURED, run `runs/20261117T212550Z-leftc-a`:
- `v_dot2_f32_f16`: all 64 combinations equal the composition model on every row (0 mismatches); every combination
  with a bit set differs from the plain instruction on at least 119,948 rows, so the comparison can see a wrong
  model (the deliberate break, `op_sel` ignored when a neg bit is set, gives mismatches in 44 combinations). So the
  bits act independently: half selection first, then the sign of each element, and |C| before its negation.
- Integer forms: in every case where the two candidates differ (16 of 32), the result equals the instruction without
  `neg_hi` on all rows and differs from the OR-ed reading on 122,452 to 131,069 rows: `neg_hi` is ignored on A, B
  and C of both integer dot forms, as on the matrix integer forms. In the other 16 cases (`neg_hi` on C, or on a
  source that `neg_lo` already marks signed) both candidates give the same result.
- Label: MEASURED. Verdicts `dotm_compose` and `dotm_int_neghi` PASS.

## MS-9. L0 bandwidth of the whole card, no two waves on one line

Question: what does L0 serve at whole-card level when no two waves read the same line? Expected: at most 128 bytes per
cycle per CU (memory-system item 5: each extra 16-byte L0 load of one wave costs 4 cycles: 512 bytes per 4 cycles),
that is 64 x 128 x clock = 25.8 TB/s at 3.15 GHz; the earlier 19 TB/s included request merging (memory-system 2a).

Probe: `left_mem` mode `l0card W`, 64 groups (one per CU: 40 KiB of LDS each, CU mode) of W waves; each wave reads
only its own 2 KiB slice (no two waves on one line; 16 waves fill the 32 KiB L0), for 0.1 s of REALTIME, in batches
of 8 `global_load_b128` and one wait. The 8 loads of a batch read four 512-byte chunks of the slice (offsets 0,
512, 1,024, 1,536 from the lane's address) twice each: loads 5 to 8 repeat the addresses of loads 1 to 4 (READ,
`k_l0card` in `left_mem.hip`). The byte count includes the repeated loads (8 x 512 bytes per batch). One burst per
job, three jobs per W; the script waits outside the GPU lock until the hotspot is below 55 C (it never had to wait
more than 1 s), and inside a job a burst would be skipped above 55 C (none was). Bytes per cycle per CU = bytes /
REALTIME span / 64 / clock, the clock from the waves' own counters (3.15 GHz in every burst). Calibration (job
`l0card 0`, verdict `mem_l0card_calib` PASS): one wave per CU, batches of 8 and 16 loads on 8 or 16 different
512-byte chunks: 137.2 and 169.3 cycles per batch, 4.01 cycles per extra load (item 5: 4.0). The first calibration
tried, item 5's absolute 156.1 cycles per batch of 8, does not transfer: that probe's batch costs 19 cycles more
than this calibration's 137.2 (its loop also computes a modulo per batch, READ; that the 19 cycles are that
address work is INFERRED); the differential cancels whatever it is. The burst itself costs 113.1 cycles per batch at
W = 1, 24 cycles less than the calibration's 137.2 for the same number of loads: not explained. INFERRED candidate:
the burst's batch reads only 4 distinct chunks (each twice), the calibration's 8; the listings differ otherwise only
in the loop control (one REALTIME read per 64 batches in the burst).

MEASURED, run `runs/20261117T213358Z-leftc-a` (three bursts per W, all within 0.1 of each other; the run before,
`runs/20261117T213230Z-leftc-a`, the same):

| Waves per CU | Bytes per cycle per CU | Card TB/s | Cycles per batch per wave | Hotspot after the burst |
|---|---|---|---|---|
| 1 | 36.2 | 7.3 | 113.1 | 56-57 C |
| 2 | 72.5 | 14.6 | 113.1 | 62-63 C |
| 4 | 126.4 | 25.5 | 129.6 | 71 C |
| 8 | 128.0 | 25.8 | 256.1 | 71 C |
| 16 | 127.9 | 25.8 | 512.1 | 70-71 C |

Answer: every CU's L0 delivers exactly 128 bytes per cycle at whole-card level, with every CU busy at once: 25.8 TB/s
at 3.15 GHz (verdict `mem_l0card_ceiling` PASS in every burst: no burst above the ceiling). Half of those loads
repeat an address the same wave loaded a few instructions before; since the total stays at the 128 bytes per cycle
that the calibration's distinct loads cost (4 cycles per 512-byte load), the repeats were not served for free at
that rate (INFERRED). One wave per CU is
latency-bound (8 loads in flight: 113 cycles per 4 KiB); two waves double it; four waves reach 99 % and eight 100 %.
So the 19 TB/s of memory-system 2a was below the L0 limit, not above it: L0 alone, without request merging, gives
more when the access pattern lets every CU work at once. MEASURED. Thermal: a 0.1 s burst lifted the hotspot from
54 C to 57 to 73 C in all but three bursts by the reading taken right after the burst (`hotspot_after`; the
`CLOCKS end` reading a moment later is up to 8 C higher, 76 to 81 C in about half of the logs); the first W = 4 burst of a run reached 80 C in the fix round's final
run and 82 C in the lead's rerun, and 77 C once in a development run (`hotspot_after` in the `l0card` logs); none
came near the 95 C stop. Bursts: each run takes three per W (jobs `W`, `W b2`, `W b3`) plus the break at W = 16, so
four at W = 16; over the seven runs that ran `l0card` (development runs from `runs/20261117T212844Z-leftc-a`, the
final reruns and the lead's) that is 19 bursts at each of W = 2, 4, 8, 22 at W = 1 (the first calibration's three
jobs in `runs/20261117T213034Z-leftc-a` were W = 1 bursts) and 26 at W = 16. The rule "at most 3
timed bursts per configuration per run" was read as three timed bursts per W per run; the break's burst at W = 16 is
a fourth, which the rule did not allow for (see "Corrections after review").

Label: MEASURED.

## MS-6. Cache hints through the buffer-load builtins

Question: do `__builtin_amdgcn_raw_buffer_load_*`'s cache-policy operand reach the same hints as the inline-assembly
encodings of memory-system item 8, and with the same effect? Expected (READ, research-10 section 4 from
`IntrinsicsAMDGPU.td`): "gfx12+: bits [0-2] = th, bits [3-4] = scope"; the effect as item 8 (chase after a hinted 4 MiB
read: RT 405.1, NT 637.5, HT 460.7, LU 704.9 cycles).

Probe: `src/study/left_hint.hip`. Listing: the builtin instantiated for all 32 policy values; a check
(`scripts/leftc_hintcheck.py`) reads each kernel's `buffer_load_b128` and compares its `th:` and `scope:` with the
bits: 32 of 32 right. Hint 3 is printed `TH_LOAD_LU`, under system scope `TH_LOAD_BYPASS` (the same bits, READ:
`TH_BYPASS = 3`, only with scope 3); hint 7 has no name and prints as `th:0x7`. On the card: flush (512 MiB read), a
whole-card hinted read of a 4 MiB region, then in a new launch (L2 empty) a one-wave chase over that region (256-byte
nodes, random order, 4,096 steps); builtin and inline-assembly trials alternate, 5 each; the hinted read's checksum
equals a plain read's (the buffer descriptor, dword 3 = 0x31004000 as composable_kernel uses for gfx11 and gfx12,
works).

MEASURED, run `runs/20261117T213657Z-leftc-a`, cycles per chase load (median of 5, range):

| Hint | Inline assembly | Builtin (aux) |
|---|---|---|
| RT (0) | 438.6 (430.6-457.6) | 434.9 (428.5-459.7) |
| NT (1) | 754.8 (746.0-757.7) | 763.6 (746.0-776.1) |
| HT (2) | 460.7 (455.9-483.9) | 467.4 (458.7-485.5) |
| LU (3) | 748.3 (739.7-751.0) | 754.3 (742.2-782.1) |
| RT with scope SE / DEV / SYS (8, 16, 24) | - | 458.6 / 465.3 / 455.8 |
| NT with scope SE / SYS (9, 25) | - | 742.3 / 748.9 |

Answer: the builtin's operand maps one to one onto the instruction's hint and scope fields, and each hint has the
same effect through the builtin as through assembly (within 2 %; verdict `hint_builtin` PASS): after an RT or HT read
the data is in the memory-side cache (435-467 cycles), after an NT or LU read it is not (748-764). The scope bits do
not change where a read leaves data. The assembly path reproduces item 8's pattern (verdict `hint_control` PASS; NT
and LU at 1.7 x RT here, 1.6 and 1.7 x there). So HIP code can ask for every hint, including those plain pointer loads
cannot express (`TH_HT`, `TH_NT_HT`, ...), through these builtins. MEASURED (effect, listing), READ (bit layout).

## MS-7. Last use through the compiler's `!amdgpu.last.use` metadata

Question: does `!amdgpu.last.use` produce a last-use load? Expected (READ, research-10 section 4): the metadata exists
for gfx12 and has no source-level builtin; memory-system reached `TH_LOAD_LU` only through inline assembly.

Probe: `src/study/left_lastuse.ll`, a hand-written LLVM IR kernel with three global loads (plain, `!amdgpu.last.use`,
and `!nontemporal` as the control), compiled by `scripts/leftc_lastuse.py` with `clang -x ir -mcpu=gfx1201 -O3`; the
loads are told apart by their offsets. CPU only.

MEASURED (listing), run `runs/20261117T213856Z-leftc-a`: the plain load has no hint, the `!amdgpu.last.use` load is
`global_load_b32 ... offset:16384 th:TH_LOAD_LU`, the `!nontemporal` load `th:TH_LOAD_NT` (verdict `lastuse_listing`
PASS). So the metadata reaches the last-use hint from IR. Its effect on the card is the LU row of MS-6 (data not left
in the memory-side cache, 748-754 cycles in the chase): the hardware effect through this path was not run again (the
draft's `hipModuleLoad` run was cut as low value: the instruction it emits is the one measured).

## CH-3. Wave placement in compiler-and-hazards item 3

Question: item 3 divided by the number of waves per SIMD assuming its 512- and 1,024-thread groups spread evenly over
the 4 SIMDs of one WGP (lane 1's finding); the placement was not recorded. Expected: 4 and 8 waves on each SIMD.

Probe: `src/study/haz_issue.hip` with `-DHAZ_PLACEMENT` (new, guarded by the macro: without it the kernels are
unchanged; the placement build passes the default build's listing expectations, run by this lane's script). Fix
round: the four merged builds were rebuilt through the script that owns them (`scripts/study-compiler.sh build`, run
`runs/20261117T223858Z-haz`): `loops-haz_issue-default`, `-nodelay`, `-novopd`, `-expert` PASS against
`scripts/haz-expect/haz_issue-*.txt`, and the other lane-8 checks (mask scan, hazard scan, pitfall, slow, memshape
and register loops) PASS too, 0 failures. Not rerun on the card (their timed loops are unchanged). Each of the 8 kernels records every
wave's `HW_ID1` after its loop; mode `place` launches each with 512 and 1,024 threads, 5 times.

MEASURED, run `runs/20261117T213956Z-leftc-a`: every kernel, both sizes, all 5 launches: 4 waves on each of the 4
SIMDs (512 threads) and 8 on each (1,024 threads). Verdict `issue_placement` PASS. So item 3's per-SIMD figures stand
as computed.

## CH-5a. Wide loads from per-lane addresses that are not 16-byte aligned

Question: do `global_load_b128` loads from 4-byte-aligned (not 16-byte-aligned) lane addresses return the right bytes,
and what do they cost? Expected: right bytes (requirement); cost unknown (a lane address crossing a 128-byte line
touches two lines).

Probe: `src/study/left_misal.hip` (the draft planned new kernels in `haz_memshape.hip`; the new file copies its readers
and method): a run-time byte offset of 0, 4, 8 or 12 is added to the base, so clang still emits one
`global_load_b128` per item (listing check: one, four and 64 wide loads in the three kernels, none split). One item
in 8 then crosses a 128-byte line and one in 16 a 256-byte line. Byte offsets that are not multiples of 4 were not
tried (they could fault depending on the alignment mode, which is not worth a memory fault on this card).
Correctness: 65,536 items of random words per offset. Cost: one wave, 64 independent loads per pass, a cold pass
after a 512 MiB flush and a warm pass (median of 9); whole card, 1 GiB, 4 items per thread, 3 bursts of 20 launches
(about 35 ms each). Calibration (verdict `misal_calib` PASS): offset 0 gives 13.9 cycles per warm load and 632.3 GB/s
(compiler-and-hazards item 5: 13.9 and 632).

MEASURED, run `runs/20261117T214259Z-leftc-a`:

| Offset (bytes) | Words wrong | One wave, cold (cycles per load) | One wave, warm | Whole card, 1 GiB (GB/s) |
|---|---|---|---|---|
| 0 | 0 of 262,144 | 45.5 | 13.9 | 632.3 (632.1-632.6) |
| 4 | 0 | 47.8 | 19.0 | 632.1 (632.0-632.3) |
| 8 | 0 | 48.1 | 18.9 | 632.1 (632.1-632.2) |
| 12 | 0 | 48.9 | 19.4 | 632.2 (632.0-632.3) |

Thermal exception: in this run and the runs before it the program had no thermal gate of its own: 4 offsets x 3
bursts x 22 launches of about 1.7 ms (about 37 ms per burst) ran back to back in one job, and the script's wait for
a hotspot below 55 C gave up after 120 s and started the job anyway. That broke the lane's thermal rule (start
below 55 C, stop at 95 C, checked per burst). The temperatures seen were within it (52 to 54 C at the start and
55 to 56 C at the end, in the `clocks` lines of the three runs that ran it: `runs/20261117T214259Z-leftc-a`, the
final rerun and the lead's), so the results stand, recorded as an exception. Fixed in the fix round:
`left_misal` now checks the hotspot before each burst (55 C) and after it (95 C) with `left_mem l0card`'s helper,
each burst is its own job (`misal bw <offset>`, three per offset) so the cool-down happens outside the GPU lock,
and the script refuses to start a whole-card job whose cool-down wait timed out (NOT STARTED). A first version with
the gate but still three bursts per job skipped 11 of 12 bursts, because one burst lifts the hotspot from 54 to
55 C (`runs/20261117T222442Z-leftc-a`).

MEASURED again with the gate, final run `runs/20261117T224107Z-leftc-a` (every burst started at 54 C and ended at
54 or 55 C; the break, the reference shifted by one word, FAILs):

| Offset (bytes) | Words wrong | One wave, cold | One wave, warm | Whole card, 1 GiB (GB/s), three bursts |
|---|---|---|---|---|
| 0 | 0 of 262,144 | 44.5 | 13.9 | 632.1, 632.3, 631.8 |
| 4 | 0 | 47.0 | 19.0 | 631.6, 632.2, 632.1 |
| 8 | 0 | 48.1 | 18.9 | 632.0, 631.6, 631.4 |
| 12 | 0 | 48.2 | 19.0 | 631.8, 631.1, 631.4 |

Answer: the bytes are always right (verdict `misal_bytes` PASS, both runs). A misaligned 16-byte load costs about 5
cycles more per load when the data is in the near caches (19 against 13.9, +37 %), about 3 cycles more from memory
for one wave, and nothing when the whole card streams from memory (the bus is the limit). MEASURED in both runs.
Practical: misalignment by 4 to 12 bytes matters only for cache-resident, latency-bound reads.

## CH-5b. Optimisation levels other than -O3

Question: do the hazard and mask results of compiler-and-hazards change at -O0, -O1, -O2 and -Os? Expected
(requirement): the hazard recognizer must still insert every required wait (a missing one gives wrong results); the
mask and memory shapes: unknown.

Probe: `haz_mask`, `haz_hazard`, `haz_pitfalls` and `haz_memshape` (unchanged sources) rebuilt by
`scripts/study-leftovers-c-a.sh` at -O0, -O1, -O2, -Os and -O3 (the control); the listing scans of compiler-and-hazards
(`haz_maskscan.py`, `haz_hazscan.py`, `mem_loopcheck.py`) run against the -O3 expectation files (CPU job
`leftc_olevel`; verdict `olevel_control` PASS: the -O3 builds match all four files); the first three probes run on the
card at every level (`haz_memshape`'s run includes whole-card streaming and was not repeated: listing only); the
deliberate no-`V_NOP` break run at every level. MEASURED, runs `runs/20261117T214835Z-leftc-a` (scans and card runs)
and `runs/20261117T214933Z-leftc-a` (breaks):

| Level | Builds | Card: hazard shapes | Card: masks, pitfalls | Listing against the -O3 expectations |
|---|---|---|---|---|
| -O3 (control) | all | all right; the call-boundary gap 64 of 256 wrong, as before | masks: the same wrong-result pattern as before (the known unsafe shapes, see below); pitfalls right | all match |
| -O2 | all | the same as -O3 | masks: the -O3 pattern; pitfalls right | all match |
| -O1 | all | the same as -O3 | masks: the -O3 pattern; pitfalls right | hazard scan matches; mask scan: `k_asm_time` has 64 matrix instructions, not 256 (less unrolling, not masking); `w_lds` shows 0 or 1 `ds_load_b128` instead of 9 and one narrow LDS load (static counts of a loop clang did not unroll); single-wave `kc_*` kernels of `haz_memshape` spill (52 to 92 bytes of scratch) |
| -Os | all | the same as -O3 | masks: the -O3 pattern; pitfalls right | hazard scan matches; `k_asm_time` as -O1; `w_lds` 1 `ds_load_b128` (not unrolled); `haz_memshape` (`olevel-memshape-Os.txt`): `k_loop64` (4-word reads) holds one `global_load_b32` inside a loop clang did not unroll (no `global_load_b128`: the reads are not merged into a wide load); the `kb_*` kernels keep their `global_load_b128` but in a loop that is not unrolled (1 static load instead of 4); `kc_split16` uses 256 VGPRs and spills |
| -O0 | `haz_hazard` does not compile: "illegal VGPR to SGPR copy" in `s_hazard_wait`, then the linker aborts (core dump): the probe's inline assembly takes a per-lane value through an "s" operand at -O0 (see T5); the other three build | (T5: all right) | masks: the -O3 pattern, and `mask_select` and `mask_uniform` right (0 of 256 wrong) although the listing puts a matrix instruction under a lane mask in them; pitfall (f): `pit_perm_pinned` FAILS: with the permute source "pinned" before the branch, 0 of 7 to 16 lanes with a switched-off partner get its value, in all 6 shapes (100 % right at -O1 to -O3) | mask scan: `k_select` and `k_uniform` each get a matrix instruction under a lane mask (0 at -O3); every kernel uses scratch |

Every level's no-`V_NOP` break FAILS (64 of 256 wrong), and every hazard shape the compiler handles at -O3 is handled
at -O1, -O2 and -Os, with the same one gap (a non-inlined callee that ends in a matrix instruction: 64 of 256 wrong,
equal to the no-`V_NOP` result, at every level). So the hazard fixes do not depend on the level from -O1 to -O3 (and
at -O0, test T5 below).

Masks: "right" in an earlier version of this table was wrong. `mask_ifload` and `mask_early`, the two shapes
compiler-and-hazards item 1 found unsafe, differ from the CPU model at every level, -O3 included (final rerun
`runs/20261117T215755Z-leftc-a`, `haz_mask_O3-all.log`: n = 8 64 of 64, n = 16 128 of 128, n = 20 144 of 160,
n = 31 23 of 248; n = 32 0 of 256), and the other mask shapes are right at every level. The small differences
between levels (for example `mask_early` n = 20: 141 of 160 at -O2 against 144 elsewhere; `mask_ifload` n = 8
"deterministic over 3 runs" no at -O0 and -O2, yes at -O1, -Os and -O3) are not shown to be level effects: the same counts and
flags change from run to run at one level (three runs, the named one, the final rerun and the lead's: -O3's
n = 8 is "yes", "yes", "no"; `mask_early` n = 20 at -Os is 142, 144, 138 of 160). -O0 differs in what it reads
besides: `mask_loop_perlane` no longer matches the stale-operand model (127 of 128 differ from it, 0 at the other
levels), and `mask_asm_time` costs 37 cycles per matrix instruction (16.7 at the other levels: its loop is compiler
code at -O0).

At -O0 the mask scan finds a matrix instruction under a lane mask in `k_select` and `k_uniform` (0 at -O3), that is,
compiler-and-hazards item 1's condition, but the card still gave `mask_select` and `mask_uniform` 0 of 256 wrong.
So the condition is reached in the listing, and no wrong result followed in these runs. MEASURED (listing and card).

The pinning workaround for fetch-inactive permutes fails at -O0 (`pit_perm_pinned`), and so does the `v_cmpx` /
permute test of the hazard probe at -O0 (test T5: exactly the lanes whose partner is switched off are wrong). For
the second, the -O0 listing was read (READ, `left_hazard_O0`, kernel `p_cmpx`): there is no `v_cmpx` at all at -O0;
the value lives in memory and only the active lanes reload it (`flat_load_b32 v2`) right before
`v_permlane16_b32 v2, v2, ...`, so fetch-inactive reads the switched-off lanes' stale `v2`. The pitfall's listing was
not read; the same mechanism there is INFERRED.

-Os differs from -O3 in one memory shape of item 5: `k_loop64`'s 4-word reads are one 32-bit load inside a loop that
is not unrolled, not a 128-bit load (the earlier text said "4 narrow loads"; the listing check counts one). MEASURED
(card results, listing counts), INFERRED where marked.

**Test T5 (fix round): the hazard probe at -O0.** The -O0 build error above is a fault of the probe, not of the
compiler's hazard handling: `s_hazard_wait`'s inline assembly passes a variable through an "s" (scalar register)
operand, and at -O0 that variable lives on the stack as a per-lane value. Expectation written before the run: if the
compiler's hazard waits hold at -O0, every hazard shape gives 0 of 256 wrong, and the no-`V_NOP` inline-assembly
break still gives 64 of 256 wrong (the test can see the hazard); a contradiction would be any wrong result in a
compiler-built shape. Probe: `src/study/left_hazard.hip`, a copy of `haz_hazard.hip` whose only change feeds the
scalar operands from `__builtin_amdgcn_readfirstlane` and copies them into the loop registers inside the statement;
built at -O0 and at -O3 (the control: its listing passes `haz_hazard`'s scan expectations, and its card run must
equal `haz_hazard` at -O3).

MEASURED, final run `runs/20261117T224107Z-leftc-a` (the same in `runs/20261117T222442Z-leftc-a`), -O0:
- every compiler-built hazard shape right: straight, B source, call, tied call, branch (both flags), mixed FP16/BF16,
  loop n = 1, 2, 5, each 0 of 256 wrong; the SGPR hazard 0 of 32 lanes wrong with the compiler's code and with the
  waited assembly;
- the -O3 call-boundary miss does not occur: `hazard_call_miss` 0 of 256 differ from the model, in both forms (at
  -O3 64 of 256; the -O0 listing scan finds none of the three missed sites `first_mm_tied`, `h_call3`, `h_call4`);
- the no-`V_NOP` break FAILS (64 of 256 wrong), so the test sees the hazard at -O0;
- `hazard_cmpx_permlane` FAILS: n = 7 7 of 32 lanes wrong, n = 31 1 of 32, n = 16 0 of 32: exactly the lanes whose
  partner has left the loop (the -O0 memory round trip read above, not the `v_cmpx` hazard).
The -O3 build of the copy gives the results of `haz_hazard` at -O3 (call-gap miss 64 of 256, `v_cmpx` test 0 wrong).
The -O0 job is marked "expected to fail" in the script (`hazard_call_miss` and `hazard_cmpx_permlane` FAIL there).

Answer (T5), corrected after the re-check: the results above are MEASURED, but they do not test the compiler's
hazard handling at -O0. READ from the -O0 listing (`build/leftc-a/tmp-left_hazard_O0/dis.txt`): the two dependent
matrix instructions of `h_straight`, `h_bsrc` and `h_mixed` are 51 instructions apart and those of `h_branch` 191
(the lead's count; the re-checker counted 59 and 218 with another rule), with loads and stores to the stack between
them, and each call shape holds a single matrix instruction per function; at -O1, -O2, -Os and -O3 the pair is 4
instructions apart with one `v_nop`. So at -O0 no hazard wait is needed in these shapes, the "missed" call-boundary
sites do not exist, and 0 of 256 wrong says nothing about the hazard recognizer. Only the inline-assembly break
(the pair 1 apart) exercises the hazard at -O0, and it fails as it must. What is shown for -O0: the probe builds,
its results are right, and the unoptimised code keeps dependent matrix instructions far apart. What goes wrong at
-O0 is unrelated to hazards: values read from switched-off lanes are stale; in `left_hazard`'s kernel `p_cmpx` the
listing shows the value reloaded from the stack right before the permute (READ), and the wrong lanes are exactly
those whose partner is off (MEASURED count); for the pitfall probe the same cause is INFERRED (its listing was
not read).

The -O0 pitfalls job is marked "expected to fail" in the script: its FAIL is this finding, and the run counts it as an
error if it ever passes.

## CH-2a. `v_cmpx` directly before a lane permute

Question: with nothing between a `v_cmpx` (which writes EXEC) and `v_permlane16_b32` / `v_permlanex16_b32`, does the
permute use a stale EXEC? Expected (READ, research-10 section 2): LLVM inserts a `v_mov` there (its
`VcmpxPermlaneHazard`); what the hardware does without it is unknown.

Probe: `src/study/left_haz.hip` mode `cmpx` (the draft placed it in `haz_hazard.hip`). Inline assembly: EXEC from
`v_cmpx_gt_u32` on per-lane data (5 masks: odd lanes, every 4th, lanes 0-15, lanes 16-31, data-dependent), then the
permute (row reversal) with fetch-inactive on and off, with nothing, a `v_mov` or an `s_nop` between (listing check:
the instruction after the `v_cmpx` is as written in all 12 kernels). CPU model: a lane that is on gets its partner's
value; when the partner is off, with fetch-inactive it still gets it, without it keeps its own old value
(compiler-and-hazards item 4); lanes that are off keep a sentinel. A stale EXEC (all lanes on) would show as written
off lanes, or, without fetch-inactive, as off partners being read.

MEASURED, run `runs/20261117T215508Z-leftc-a`: 0 lanes differ from the model in all 60 cases (2 permutes x 2
fetch-inactive settings x 3 gaps x 5 masks); the deliberate break (fetch-inactive ignored in the model) differs in 96
lanes. Answer: on gfx1201 the permute right after `v_cmpx` sees the new EXEC, with or without fetch-inactive; the hazard
LLVM guards against did not show in these sequences (the guard costs one `v_mov`). Verdict `cmpx_control` PASS;
the no-gap result is INFO.

## CH-1. The scalar-register hazard with a wrong result

Question: is there a sequence where the missing `s_wait_alu depctr_sa_sdst(0)` (a VALU reading an SGPR that SALU has
just rewritten) gives a wrong result? compiler-and-hazards item 2's no-wait loop computed right.

Probe: `left_haz` mode `sgpr`: an LCG in SALU rewrites s20 (and s21) and a VALU instruction reads it D2 = 0, 1, 2, 4
or 8 `s_nop`s later; the previous read is D1 = 0, 2 or 8 `s_nop`s (plus 2 to 4 SALU instructions of the next LCG
step) before the rewrite (the body is unrolled twice); four kinds of reading instruction (`v_fma_f32` VOP3,
`v_add_f32` VOP2, `v_fma_f64` on s[20:21], `v_exp_f32`), 64 iterations, with and without the wait; every no-wait
result is compared bit for bit with its waited twin, and the waited `v_fma_f32` with a CPU model.

MEASURED, run `runs/20261117T215323Z-leftc-a`: the waited `v_fma_f32` equals the CPU model in all lanes of all 15
distance pairs (verdict `sgpr_control` PASS; the break with the LCG one step behind differs in 480 lanes), and 0 of
60 no-wait sequences differ from their waited twins, down to the SALU write directly before the VALU read. Answer: no
wrong result was found; on this card the VALU read waits for an SALU write of an SGPR by itself in all these shapes
(INFERRED: the hardware interlocks; the `s_wait_alu` the compiler inserts costs about 9 cycles per use, compiler-and-
hazards item 2, and may guard a case not reached here). Label: MEASURED (no wrong result in 60 sequences); "never
needed" is not shown.

## Checks and deliberate breaks

| Probe and mode | Break | Result |
|---|---|---|
| `left_core w64` (wave32) | the independent FMA expected at 2.00 cycles | FAIL as intended |
| `left_core64 w64` (wave64) | the chained matrix instruction expected at 8.0 cycles | FAIL as intended |
| `left_mem ns` | the L0 calibration chase replaced by independent loads (8.7 cycles) | FAIL as intended |
| `left_vopd` | the sequential control expected to give the old value | FAIL as intended |
| `left_spin` | 0 events expected in the R launches, and the 128-cycle sleeps expected to be found | both FAIL as intended |
| `left_lds banks` | stride 33 expected to collide like stride 32 | FAIL as intended |
| `left_lds inflight2` | stride-8 loads expected 6 cycles apart | FAIL as intended |
| `left_lds misalign` | the aligned b32 at +4 judged by the masked-to-16 rule | FAIL as intended |
| `left_lds atomic` | 32 lanes adding 1 to one word expected to give 31 | FAIL as intended |
| `left_lds swz` | the bit-mask model with and-mask bit 0 flipped (48 mismatches) | FAIL as intended |
| `left_core insn` | the independent FMA expected at 2.00 | FAIL as intended |
| `left_mem tlb` | the L0 calibration chase replaced by independent loads | FAIL as intended |
| `left_mem multi` | the same | FAIL as intended |
| `left_core bank` | the same-triple pair expected at 2.0 cycles | FAIL as intended |
| `left_core run12` | all-12-byte instructions expected at 1.00 | FAIL as intended |
| `left_core sgprw` | `v_readfirstlane` x16 expected at 1.00 | FAIL as intended |
| `left_core fsize` | the 2 KB loop control expected at 40 cycles | FAIL as intended |
| `left_cvt` | the dual-issue consumer claimed at 1.00 cycle per weight | FAIL as intended |
| `left_dotm` | the composition model with `op_sel` ignored when a neg bit is set (44 combinations differ) | FAIL as intended |
| `left_mem l0card 16` | the ceiling claimed at 64 bytes per cycle per CU | FAIL as intended |
| `left_hint` | the builtin NT expected to behave as RT | FAIL as intended |
| `leftc_lastuse` (CPU) | the plain load expected to carry `TH_LOAD_LU` | FAIL as intended |
| `haz_issue_place place` | 3 and 5 waves claimed on two SIMDs | FAIL as intended |
| `left_misal misal` | the reference shifted by one word (1,048,576 words differ) | FAIL as intended |
| `haz_hazard_O1`, `_O2`, `_Os`, `_O3` `break` | the inline-assembly matrix pair without `V_NOP` judged as right | FAIL as intended at every level |
| `leftc_olevel` (CPU) | none of its own: the -O3 control must match the committed expectations | (a control, not a break) |
| `left_haz cmpx` | the model without fetch-inactive for the `v_mov` variant (96 lanes differ) | FAIL as intended |
| `left_haz sgpr` | the CPU model one LCG step behind (480 lanes differ) | FAIL as intended |
| `left_comm lds`, `left_comm global` | a round trip claimed to be at most one store-to-load link (36 cycles) | FAIL as intended (a first break, "expect a 30-cycle round trip", could not fail against a lower bound; replaced) |
| `left_w64fma` (T1) | item 4's kernel expected at 64 lane-FMAs per cycle | FAIL as intended |
| `left_bank bank`, `left_bank64 bank` (T1) | the independent FMA expected at 2.00 | FAIL as intended in both builds |
| `left_bank insdata` (T3) | `v_rcp_f64` expected at 16.0 cycles | FAIL as intended |
| `left_cvt cd2` (T4) | the 16-in-flight form claimed at 1.00 per weight | FAIL as intended |
| `left_hazard_O0`, `left_hazard_O3` `break` (T5) | the inline-assembly matrix pair without `V_NOP` judged as right | FAIL as intended at both levels |
| `left_core_wgp fsize` (T2) | no break of its own; the job's calibration break is `left_core fsize`'s (the same probe) | - |
| `leftc_bankcheck.py` (T1 listing) | run on the original `sched_wave64` listing, which lacks the pinned loops | exit 1 (checked by hand once, not a job) |

## How to rerun

`scripts/study-leftovers-c-a.sh all` (build, every job, every break). The build takes about 1.5 minutes (at most 4
compilers at once, under nice); 81 jobs and 39 breaks, almost all single-wave or a few waves (the longest, `spin`,
about 9 s); the whole-card jobs are the 0.1 s L0 bursts and the 35 ms streams of `misal bw`, one burst per job, each
started only after the hotspot is below 55 C (the script waits outside the GPU lock, at most 120 s, and does not
start the job otherwise); the whole run took 3 minutes in the fix round (22:41 to 22:44 UTC). One item: `scripts/study-leftovers-c-a.sh run
<mode>` (the mode is the first argument of a job, for example `w64`, `ns`, `spin`, `banks`, `l0card`, `all` for the
-O-level runs). Results in `runs/<UTC time>-leftc-a/summary.md`, one log per job.

Rerun from the committed code (git status clean, commit 7e66557): `runs/20261117T215755Z-leftc-a`. Every verdict as
in the runs named in the sections (numbers within their spreads for the examples given here; after the fix round added kernels to `left_cvt`,
four of its rows moved in the final run beyond the earlier spread: the simple LDS form 4.607-4.626 earlier and
4.679 now, `sr_xorshift` 7.252 to 7.422, `sr_lcg32` 6.124 to 6.000, `cvt_sr` 1.000 to 0.984, so CD-4's "2 and
3.6 % above the sum" is not reproduced there; code position can matter, see EC-7; for example wave64 FMA 1.250 with banks shared,
memory 718 / 686 / 682 cycles per batch with distinct lines, L0 128.0 bytes per cycle per CU, round trip 272.0
cycles); 59 jobs ok, the -O0 pitfalls job failing as found, all 32 deliberate breaks FAIL, no job not started or
timed out. The single-wave L2 step of `multi` moved within its known wobble (W = 1 at 7,872 nodes, W = 4 at 11,008,
W = 8 at 5,632, W = 16 at 2,880).

Fix round: rerun from the committed code (git status clean, commit bcce6f3): `runs/20261117T224107Z-leftc-a`. 81
jobs: 79 ok, and the two "expected to fail" jobs (`haz_pitfalls_O0`, `left_hazard_O0`) failing as found; all 39
deliberate breaks FAIL; no job not started or timed out. The values of the earlier sections reproduce within their
spreads (for example wave64 FMA 1.250 with banks shared, L0 chase 80.7 cycles, L0 128.0 bytes per cycle per CU, LDS
round trip 272.0, the fsize calibration 19.9 / 19.9 / 34.4); the T1 to T5 tables above are from this run.

## Not measured

Every item of the list was run, some only in part. Parts not run:
- MS-7: the draft's `hipModuleLoad` run of the IR last-use load on the card (listing only; the instruction is the
  one MS-6 measured).
- EC-5: the draft's 8-waves-per-SIMD case of the bank stall.
- CH-5b: `haz_memshape` on the card at -O0, -O1, -O2 and -Os (listing only; its run streams the whole card for each
  shape).
- MS-3: the prefetch probe's 694 cycles was not rerun, and its access shape was not re-read.
- SC-10: Vulkan's 1.06 cycles per wave64 FMAC: its listing was not re-read, so which register layout it had is not
  known.
- CD-3: the full set of 2^k modifier combinations (64 chosen combinations were run, of 256 for the two sources' bits
  alone).

Left open or not settled:
- EC-6: the rule for runs of 12-byte instructions (neither a fetch-rate model nor a boundary-crossing model fits the
  42 points; the table is reported).
- EC-7: the size and organisation of the structure that holds a loop's code; why the copy B side loses and copy A
  does not (T2); why the two-SIMD interference shows in one placement and not in one 68 bytes away; why copies of a
  loop aligned at the same offset inside 4 KB slow one wave that runs only one of them (198 cycles of loop control at
  2 KB).
- SC-10 / T1: the rule behind the wave64 register-bank costs (1.0 to 2.5 per instruction for blocks with two sources
  in one bank), and the wave-size-independent 1.343 of instructions that read and write bank 0.
- EC-5: the 8-waves-per-SIMD case of the bank stall (execution-core 8g already shows other waves do not fill it); why
  the same triple repeated is free.
- EC-8: the extra cost of runs of three SGPR-writing instructions with fillers (6 of 63 points off the rule).
- EC-2d, EC-2e: the 64-bit add pair with a VCC carry and back-to-back `v_div_scale_f32` are not linear in the number
  of blocks (two increments reported); the cost of the compiler's full division sequence was not timed.
- MS-2a: which structure sets the one-wave L2 limit (per wave, per CU or per shader array: the dispatcher spread the
  waves over shader engines, so these were not separated); whether the 45,000-line total is L2's capacity; why one
  and two waves are faster at 15,424 nodes than at 11,008.
- MS-4: why two loads in flight to one L2 line cost about 20 ns more than to two lines.
- MS-5: why 16 KiB spacing misses at 3,072 nodes (inside the 64 MiB reach) and 2 MiB spacing already at 1,024; where
  between 1,024 and 1,152 nodes the step lies; the second rise of memory-system 2b-2 above 120 MiB.
- MS-9: why the burst's batch (4 distinct chunks, each read twice) costs 113 cycles against the calibration's 137 for
  8 distinct chunks.
- MS-7: the effect of the IR last-use load on the card (the instruction is the one MS-6 measured).
- SC-2: why the stall costs 24 to 27 % more cycles when the wave stores nothing per step; what sets the event rate
  (380 to 420 per second in the lane's runs, 520 to 560 in the lead's).
- CD-1: what keeps the dual-issue consumer at 2.19 cycles per weight instead of the 1.5 issue bound. CD-2: why the
  LDS table method is bimodal (inside one process too) and why issuing its loads ahead is slower (T4 does not support LD-2's
  issue stop as the cause, and the simple form, the fastest, issues 32 loads at once; the ahead8 form also has 4
  waits against 2, so T4 alone does not exclude it; the half-register candidate is untested).
- LD-1d: the swizzle rotate and FFT modes (results recorded, no rule).
- LD-2: what stops the issue at about 17 slow 128-bit or 22 slow 32-bit loads (a queue is INFERRED, its size not
  shown); whether faster-draining conflicting loads stop at a different count.
- CH-5a: byte-misaligned global loads (not tried: possible memory fault).
- CH-5b: the -O0 listing of the permute pinning pitfall was not read (the T5 probe's -O0 listing was); `haz_hazard`
  itself still does not build at -O0 (its inline assembly; the copy `left_hazard.hip` does).
- CH-1: a sequence where the missing SGPR wait gives a wrong result (none in 60).

## Changes to the draft plan

The lead's questions on the draft, settled by reading (the lead's decision):
- Question 4 (the "5 cycles" bank case): READ from execution-core item 3's table: 5 cycles is the cost of an
  instruction reading three different registers of one bank when the previous one read a different triple of that
  bank (9.999 per such pair; the same triple repeated costs 2.968 per pair). EC-5 measured the rule behind it.
- Question 5 (matrix-timing item 11): READ: the ratio column is the HIP-event instruction rate divided by 128 SIMDs x
  the counter clock of the same launches, and the counter clock is the same at W = 1 and W = 16 (3,227 and 3,220 MHz),
  so the 1 to 2 % loss is a real loss at equal clock, as the draft assumed (that item, MT-5c, is lane B's).
- Question 6 (the in-flight probe's "L2-line hits"): the draft's reading was right: MS-4 measured that half the
  N = 1 batches of the old layout are L2 hits because two 128-byte loads share one 256-byte L2 line.

Changes per item:

- SC-10: the draft's verdict failed any wave64 FMA cost other than 1.06 or 1.234 as "unexplained"; the study rule
  for exploratory answers is that the verdict judges only calibration and linearity, so the rates are INFO and the
  wave64 build carries its own known answer (the 16-cycle matrix instruction). The probe is a new file
  `left_core.hip` (not `left_w64.hip`) because the same one-statement method also serves EC-5, EC-6 and EC-8; those
  blocks are in a new generated header rather than in `tim_core_blocks.h`, so the merged `tim_core` probe and its
  committed header stay as they are.
- MS-3 and MS-4: the draft compared segment means and planned an old-layout check against 549 / 845 cycles. The
  segment means here include the loop around the stamps (about 300 cycles per batch: the segment descriptor is
  reloaded), so the verdicts use the per-batch stamps: medians for the latency rule, per-batch means for the ratio
  check (mem_inflight's figures are means), and ratios rather than cycle counts because the clock differs between
  jobs. Added: chases with 256-byte nodes and per-step stamps, which settle MS-3 directly.
- SC-2 and SC-1: a third variant Q (no per-step store) was added, because R and N differ in the message only if the
  per-step store stays; the draft's "false flags at most 1 per 100,000" is INFO, not a verdict, because the ground
  truth of a real trace's small events is unknown; the injection uses a hash computed in the kernel instead of a
  table read from memory (the table read changed the steps it was meant to measure); the recall verdict uses the
  measured limit (0.5 x the median step + 64) instead of the draft's fixed 304 cycles, because the step here is 379
  to 556 cycles, not 480.
- LDS items: a new probe `left_lds.hip` instead of new modes in `lds_probe.hip` (one tim-style kernel per instruction
  with the addresses as data serves all patterns; the merged probe stays unchanged). LD-2's verdict: the draft's
  "step at 64" was the hypothesis under test, and the hardware stops issuing at about 17 to 22 loads, so the verdict
  checks the calibration (data spacing 32 and 6 cycles) and the issue limit is INFO; a 32-bit series was added to
  tell an instruction count from a byte count. LD-1a also tests 96-bit loads and costs; LD-1b uses
  `ds_cmpstore_rtn_b32` for the result (the cost row uses the non-returning form). LD-1d's rotate and FFT modes have
  no formula in the guide that could be checked, so they are reported.
- EC-1, EC-2: blocks in `left_core` instead of rows in `core_valu.hip` (fixed registers and banks; the merged probe
  stays as it is). The draft's verdict thresholds (FP16 forms within 0.2 of FP32, add pair >= 1.95, division pair
  >= 1.95) are INFO comparisons here, because the expected values were inferred, not documented; the verdict checks
  the calibration, linearity of the method and the three correctness checks. Added: the FP32-result dot product and
  `v_rcp_f64` as neighbours in the same run, an SGPR-carry form of the 64-bit add, and the full division sequence
  as the "known quotient" check.
- MS-5 and MS-2a: modes of `left_mem` instead of `tim_l2`. The draft's step rule (1.25 x the value at 512 nodes or
  the smallest N) first caught the L0 capacity (81 to 240 cycles at 512 nodes); the rule now measures from the L2
  plateau at 512 nodes and only up to 2,048 nodes (above about 4,000 lines one wave also leaves L2). Finer node
  counts (1,152 to 1,920) were added where the step lay. The calibration point "2,048 nodes at 4 KiB = 233 +-10 %"
  failed by 1 cycle in one run (257); it is now the documented one-wave L2 plateau of memory-system 2b (231-256)
  +-5 %, because 233 was one sample of that range. `multi` uses W = 1, 2, 4, 8, 16 (draft: 1, 4, 16, 64; 64 waves
  exceed L2 at 512 nodes each and say nothing about the limit) and three chain seeds per point.
- LD-4: inline-assembly poll loops and a one-wave floor added; a WGP-mode build for the other CU of the WGP; the
  calibration is 53 cycles (lds item 1's two numbers for this loop shape), not 36, because the link includes the add
  between the load and the next store; the break claims an upper bound instead of a lower one.
- EC-5, EC-6, EC-8: blocks in `left_core` instead of `tim_core`. EC-5 uses k = 0, 1, 3, 7, 15 (whole periods in a
  block of 16; the draft's 2 and 4 do not divide it) and no 8-wave run. EC-6 adds G = 0 and G = 3 for the known
  values; its fit residual is INFO as planned. EC-8: the linearity check covers the calibration blocks; the three
  non-linear blocks (runs of 3) are reported. A first EC-8 run had all filler FMAs on one register (a dependent
  chain, 5 cycles per period); the calibration caught it and the generator was fixed before the run used here.
- EC-7: the 4 KB calibration point is the loop that starts at a 4 KB boundary, because the run showed the cost of a
  4 KB loop depends on its start (127 in the two-copy kernel); "all on copy B" controls were added. The draft's
  loops on two CUs of a WGP use a WGP-mode build; waves were placed by `HW_ID1` rather than by wave count alone.
- CD-1, CD-2, CD-4: a new driver `left_cvt.hip` (the draft extended `cvt_time.hip`; that probe has no `--break` and
  one generic verdict). The draft's ranges were predictions, not documented values, so the verdicts are the issue
  bounds and the ranges are INFO. CD-2's bound is 3.5, not 3.0: the method needs 2 VALU instructions per weight for
  the addresses plus half a dot product, and 1 LDS load. The simple LDS form is no longer a calibration point: it is
  bimodal (4.6 or 5.2 to 5.9 per weight between runs). CD-4 adds one-bank forms after the first run showed a
  quarter-cycle excess in every generator.
- CD-3: a new probe `left_dotm.hip` instead of rows in `left_dot.hip`; 131,072 rows per variant instead of 10^6 (every
  variant's result differs from the plain one on at least 119,948 rows, so the smaller count still separates the
  models); the combinations are 64 chosen ones, not all 2^k (2^8 for the two sources' bits alone).
- MS-9: W = 1, 2, 4, 8, 16 waves per CU (the draft: 8 waves with 4 loads in flight); one 0.1 s burst per job with the
  cool-down outside the GPU lock (a first version waited for the cool-down inside the program, which would have held
  the lock; it was changed before any burst ran with a wait); the calibration is item 5's per-load differential (4.0
  cycles), not "one wave per CU reproduces 128 bytes per cycle": one wave with a few loads in flight is
  latency-bound (36 bytes per cycle here).
- MS-6: a new probe `left_hint.hip` (the draft put it in `left_mem.hip`); the chase uses 256-byte nodes (each node its
  own L2 line, see MS-3) and the effect is compared builtin against assembly in the same run, so the verdict does not
  depend on matching item 8's absolute cycle counts.
- CH-5a: a new probe `left_misal.hip` instead of kernels in `haz_memshape.hip`; whole-card bursts of about 35 ms
  instead of 0.2 s (enough for a stable median at 632 GB/s); only 4-byte-aligned offsets.
- CH-5b: `haz_memshape` at other levels: listing only (its run streams the whole card for each shape). `haz_hazard`
  at -O0 does not build; the first version of this file called that a toolchain limit, but the cause is the probe's
  inline assembly (a per-lane value in an "s" operand at -O0); the fix round's copy `left_hazard.hip` builds and runs
  at -O0 (T5).
- CH-2a: fetch-inactive off added (with it on, a stale EXEC could only show through written off lanes); a new probe
  file instead of `haz_hazard.hip`. CH-1: reading instructions and distances as the draft lists, plus VOP2 `v_add_f32`;
  the "k instructions before the SALU rewrite" are `s_nop`s; no-wait results are compared with their waited twins on
  the card instead of a CPU model for every kind (only `v_fma_f32` has one).
- MS-7: listing only (no `hipModuleLoad` run on the card): the emitted instruction is the one whose effect MS-6
  measured.

## Corrections after review (2026-11-17)

An independent review (FIX-PASS) found the numbers matching the logs but several statements stronger than the
evidence, one probe without a thermal gate, stale counts, and five places where a cheap test could replace an
inference. What changed, and why (fix round, branch `study/leftovers-c-a`, code commits d99b3fa and bcce6f3, final
run `runs/20261117T224107Z-leftc-a`):

1. SC-10: "In short" no longer states the cause of the 0.81 as fact on the strength of an analogy; test T1 measured
   it on the earlier probe itself (moving only the registers gives 1.000). The "1.5 cycles per conflicting
   instruction" was derived from one block (8 of 16 conflicting), not measured; T1's blocks with all 16 conflicting
   cost 1.000 to 2.500, so it is withdrawn as a rule. "Can lose up to 25 %" was that probe's case, not a bound:
   reworded. New: a 1.343 cost, in both wave sizes, for blocks that read and write bank 0 in every instruction.
2. CH-5b: "masks right" was wrong: `mask_ifload` and `mask_early` give the same wrong-result pattern as at -O3 at
   every level (the known unsafe shapes). The level-to-level differences in the determinism flag and in small counts
   are run-to-run differences: they change between three runs at one level, -O3 included (my reading differs from
   the review's "determinism flips between levels"). At -O0, `k_select` and `k_uniform` have a matrix instruction
   under a lane mask, yet the card gave 0 of 256 wrong: now said. The -Os memory-shape sentence ("stays 4 narrow
   loads") contradicted its own check (one 32-bit load in a loop that is not unrolled); corrected, and the `kb_*`
   kernels' drop to one wide load in a loop is mentioned.
3. LD-2: "a wave can only queue about 17 to 22 loads" and "more than about 16 LDS loads ahead gain nothing" were
   stated for all loads; all three series used slow loads, and the earlier LDS lane saw fast loads issue 96 in a row.
   Now scoped to slow loads, with the link to the earlier lane's open counter question (LD-2 is that test; it found
   the issue stop instead of the counter limit) and kept apart from the other lane's global-load counter. CD-2's
   connection named as a candidate; test T4 then ruled it out. CD-2 "slower" holds only against the simple form's
   fast mode: in one of its three slow runs (5.15) the simple form was still faster than "ahead" (5.54) (the review
   said slower in all three; the logs say two of three). The run counts were recounted from the logs (6 fast and 3
   slow development runs, plus the final rerun's 4.624 and the lead's 4.626).
4. Thermal rule: `left_misal` had no in-process 55 C start gate or 95 C stop, and the script's 120 s wait started the
   job anyway when it timed out. Both fixed (the gate and stop of `left_mem l0card`; one burst per job; a timed-out
   wait now gives NOT STARTED), the probe and its break rerun, and the earlier runs recorded as an exception (their
   temperatures stayed within the rule: 52-54 C at the start, 55-56 C at the end). Bursts per configuration: read as
   "at most 3 timed bursts per W per run"; each run did three per W plus the break's fourth at W = 16, which the rule
   did not allow for; over the seven runs that ran `l0card`: 19 to 26 bursts per W in all (MS-9). The highest
   hotspot after an `l0card` burst was 82 C (the lead's rerun), not the 77 C stated before; corrected.
5. Counts: 3,510 timed regions in the wave32 object (not 2,790; final rerun's `asmcheck-left_core.txt`). "Nothing on
   the list was cut" and "every item was run" contradicted the text: every omission (MS-7's card run, EC-5's 8-wave
   case, `haz_memshape` on the card at other levels, the prefetch probe's 694, Vulkan's 1.06 listing, CD-3's full
   set) is now listed under "Not measured", and both sentences say "some only in part".
6. MS-5: the reach is now qualified (spacings of 32 KiB or more; the step bracketed only between 1,024 and 1,152
   nodes), with the two points that do not fit (16 KiB at 3,072 nodes, 2 MiB at 1,024) in "In short".
7. MS-2a: two limits are described (a total near 45,000 lines, the same in three runs; a one-wave limit that moves
   between about 5,600 and 11,000 nodes), from the named run, the final rerun and the lead's rerun; "per wave" is
   softened to a reading. New, in all three runs: one and two waves are faster at 15,424 nodes than at 11,008.
8. MS-9: the loop is now described as it is (each batch reads 4 chunks twice); "about 40 cycles of address work"
   was 156 - 113, the wrong pair: item 5's batch is 19 cycles above this calibration (137.2), and the burst's 113.1
   against 137.2 is not explained (candidate given as INFERRED).
9. EC-7: the second wave of each SIMD in the four-wave rows (about 404 cycles of loop control in all cases, controls
   included) is now in the table, with the reading INFERRED. My reading differs from the review's ("384 FMAs plus 20:
   plain issue sharing"): with equal sharing both waves would show about 788 cycles per iteration, but the first
   wave shows 405 (its alone value), so the second wave appears to get the SIMD mostly after the first. Test T2
   (the swap) contradicts the earlier "first CU served first". New: the two-SIMD interference of this item vanished
   when the loops moved by 68 bytes.
10. EC-2: the inputs of the FP64 and pseudo-scalar rows were never set (clobbers only); said beside the table, and
    test T3 measured with set data: no data dependence.
11. CH-5b: "a toolchain limit" was wrong: the -O0 build fails because the probe passes a per-lane value through an
    "s" operand at -O0. Test T5 fixed the copy and ran it at -O0.
12. Merged probes: `cvt_time` and the four `haz_issue` builds were rebuilt and listing-checked through their own
    scripts (`runs/20261117T223759Z-conv`, `runs/20261117T223858Z-haz`), all clean; not rerun on the card (reasons in
    CD-1 and CH-3).
13. SC-2: which clock each rate uses is stated (REALTIME for R, launch wall time for N and Q); "card-wide" is credited
    to scheduling item 7; the lead's rerun's rates (518 to 562 per second against the lane's 382 to 422, the same loss
    per event) are reported without a cause.
14. LD-4: the 53-cycle calibration is 36 plus a derived 17, now said; the global round trip's range is given per run
    (871 to 952 cycles across three runs; the final rerun's smallest was 882, outside the old "892 to 952").

Tests run in the fix round (each with its expectation written before the run, a control, a break that FAILs and a
listing check; tables in the sections named):
- T1 (SC-10): the 0.81 is the register layout (1.234 to 1.000 by moving registers only); the per-instruction
  "1.5" is not a rule (1.000 to 2.500).
- T2 (EC-7): the losing side follows the code copy, not the CU (three runs, two placements).
- T3 (EC-2): no data dependence in the FP64 and pseudo-scalar costs (six data sets, 29 blocks); `v_sqrt_f64` has
  about 26 correct bits.
- T4 (CD-2): at most 16 loads in flight is slower (7.19 per weight) than 32 (5.50): this does not support LD-2's
  issue stop as the cause.
- T5 (CH-5b): the probe builds and is right at -O0, but its -O0 code keeps dependent matrix instructions 51 or
  more instructions apart, so the hazard waits are not tested there; the -O0 failures are stale values of
  switched-off lanes (through memory: read in one probe, inferred in the other).

Corrections after the re-check of the fix round (2026-11-17, by the study lead; text only, no new run):
- T5 and the "In short" compiler line: "the hazard waits hold at -O0" and "the call-boundary gap is closed at -O0"
  withdrawn; the -O0 listing shows why the test cannot judge it (the lead recounted the instruction distances).
- T1: "every one of the 21 samples equal" was false for two modes (one sample of 80.01 and one of 44.88).
- T3: the 0.015 differences belong to the development run 223438Z; the final run has none; three rows have dropped
  samples; no positive control.
- MS-5 "In short": 64 KiB, not 32 KiB (2,048 nodes still hit at 32 KiB). MS-2a: the final run's one-wave step at
  4,032 nodes widens the range to about 4,000 to 11,000.
- CH-5b: level differences are "not shown to be" level effects (-O3's counts did not move in four runs; -O0 and
  -O1 deviate). MS-9: which hotspot reading the burst figures use. CD-2: "rules out" replaced by "does not
  support". The rerun paragraph: four `left_cvt` rows moved after kernels were added.
- Not checkable: that each test's expectation was written before its run (the development runs used uncommitted
  code; the expectations are in the fix round's brief and in the text). The expected-failure logic of the script
  accepts any FAIL verdict in the two marked jobs; `mem::hotspot_c()` returns -1 when HWMON is unset, which would
  open the gate (it did not happen: the logs show real temperatures). Both are left as they are and noted here.

Also found while doing this: adding kernels to `left_core.hip` moved the fsize loops and broke that job's
calibration (copy A alone 41 instead of 21), and aligning the loops at 4 KB made a single wave on a 2 KB loop pay
198 cycles of loop control (`runs/20261117T222442Z-leftc-a`, `runs/20261117T223054Z-leftc-a`). The fix round's
kernels were moved to `left_bank.hip` so `left_core`'s object stays as before (EC-7, T2).
