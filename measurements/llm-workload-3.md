# LLM data dive, round 3: the questions round 2 left open

Measured 2026-11-18 on the LACT profile "Testing" (stock core clock, 317 W cap), memory clock 1408 MHz in every GPU job of this round (before and after each job, `uclk_before`/`uclk_after` in
every `.meta`), performance level auto. A regulator-level power-limit mod (1.2x) is in the card, so reported watts are scaled down on the main rail: only differences between runs are quoted, and every wall-clock
number is for this configuration. Software as in round 2 (llama.cpp master f477c1b90, ROCm 10.1; unmodified and "patched" builds); ROCm 10.0's `rocprofv3` from the pip packages in `toolchains/torch-rocm10-venv` for Q4.
Run folder `runs/20261118T175910Z-llm3/` (git-ignored). Expectations were committed before the GPU jobs of each question: `llm-workload-3-expectations.md` (Q1 to Q5) and
`llm-workload-3-expectations-q1d.md` (the random-data follow-up of Q1, added after the first Q1 result and committed before its jobs). Labels: MEASURED (a run file is named), READ (source, listing, file),
INFERRED (reasoning, not tested). Medians everywhere; every number names its run file; numbers were written from the raw files.
All GPU jobs went through the gated runner (shared lock, `nice`, time limit, device index from the card record, 8 GB floor, the 10 GB systemd cap with no swap, hotspot start limit 60 C for loops over 10 s and 70 C
for short jobs, stop at 95 C, game check, LACT profile "Testing" only): 195 jobs with a `.meta` (18 of them the ROCm 10.0 counter jobs that fail by design, 6 the probes' deliberate breaks that must exit 1, every other one exit 0), memory clock 1408 MHz before and after every one. Highest hotspot 81 C (`r3m-*`, the 10 s power loops) (stop limit 95 C never reached), no game, no thermal stop, no memory-cap kill, LACT and sysfs untouched.

## Plain summary

1. **Q1, the extra 57 W of FP8 generation: a bit more than half explained (30 to 33 W), the rest not.** The ISA shows that the patched FP8 decode loop does 3.9 to 4.3 vector instructions per weight and the Q8_0 loop 1.5 to 2.0 (1.9 to 2.9 times as many over all pairings of the loops, not the 8 times I expected).
   Two synthetic kernels that stream the same bytes at the same bandwidth (672 GB/s, within 0.5 %) with these instruction mixes in the thread-block shape of the real kernels draw 32 W more for the FP8 mix (285 against 253 W); without the block shape (loop only) the difference is 13 W.
   With random data instead of constant data both kernels draw about 30 W more and the difference stays at 30 W (314 against 284 W). The FP8 probe was at the power limit in 89 % of its samples there (as the real run was), so its random-data draw is capped and the +30 W is a lower bound: whether data-dependent switching explains part of the remaining 24 to 27 W is not decided. The counters that work (SIMD utilisation 0.997 in both, SQ busy cycles) cannot tell the two kernels apart.
2. **Q2, the exact 8B comparison: does not fit under the 10 GB job cap.** Only 31 of the 224 FP8 tensors of the 8B hold every FP8 x scale product exactly in F16, so a mixed F16/F32 reference is 27.9 GB (F32 alone is 30.0 GB). The card has 13.7 GB free and the job cap is 10 GB: at least 4.2 GB short even with
   zero overhead. Nothing was run; the 8B stays against the F16 reference.
   > **Updated after round 4 (`llm-8b-exact.md`):** the owner later allowed one job under a 16 GB cap (and the web browser closed first), and the exact 8B reference was built and run; the reference is the FP8 checkpoint's own weights converted exactly, not Meta's 16-bit 8B.
3. **Q3, the 4 % lower reading of the rebuilt probe: neither the code nor the build, the 1 GiB read before the rows.** The machine code of the timed kernels is byte-identical in the old and new binaries. In 8 interleaved repetitions the new binary reads 3.7 % lower on the widen-with-reused-destination row (ranges separate),
   while the same binary with only that pre-read switched off equals the old one (-0.5 %, ranges overlap). Only that one row moves; the other four rows are equal within 1 %. So the "gain" of a reused destination is 9 % after a cold start and 4 % after a 1 GiB read: both are real, they are two cache states (INFERRED; the two readings are MEASURED).
4. **Q4, dead counters and ROCm 10.0: no help.** The 10.0 profiler cannot collect any of the 13 counter sets on this card (it aborts at start-up with "aqlprofile API table load failed"), including the four that work with 10.1. Its kernel trace works. The counter lists differ by one entry (`SIMD_UTILIZATION`, only in 10.1). The kernel is 7.3.4; nothing readable
   without root (perf-event setting, amdgpu and kfd parameters) is about counters. So the zeros of 10.1 stay unexplained.
5. **Q5, the sub-microsecond kernel overlaps: a timestamp artifact (MEASURED on a synthetic pair probe).** In 323,982 dependent and independent kernel pairs (plain launches and HIP graphs, kernels of 1 to 20 us) under the profiler, an independent in-kernel clock (the 100 MHz REALTIME counter, 10 ns per tick) shows no overlap
   at all, and the shortest gap between the last wave of one kernel and the first wave of the next is 3.2 us (1.8 us without the profiler). The profiler showed 3 overlaps (465, 490 and 8,040 ns), and in all three the waves of the second kernel started 3.4 us after the first kernel had finished. Two independent kernels on two streams, as a control, overlap on both clocks in 1000 of 1000 rounds;
   with a host sync between the kernels neither clock shows any. The overlaps of round 2 (5 in 60,984 pairs, 29 to 815 ns) are probably the same kind (INFERRED: llama.cpp itself was not re-timed in-kernel; the 323,982 pairs are MEASURED on the synthetic probe). The `no-overlap` check is not changed.

## Expectations against results (round 3)

| # | Expectation | Result | Verdict |
|---|---|---|---|
| E1a | patched FP8 loop 4 to 7 vector ALU instructions per weight, Q8_0 loop 0.3 to 0.6, ratio at least 8 | FP8 4.31 (no gate) and 3.88 (gate and up fused) per weight, Q8_0 1.75 (plain), 2.0 and 1.5 (fused kernel): ratio 1.9 to 2.9 (all pairings); the F16-to-F32 conversion is folded into the FMA (`v_fma_mix_f32`), there is no separate conversion | right on the FP8 count for the unfused loop (3.88 for the fused loop is below 4, the "fails if under 4" case), wrong on Q8_0 and on the ratio |
| E1b | SIMD utilisation and SQ busy cycles about 1.0 for both, no useful difference; fails if SQ_BUSY_CYCLES differs by more than 5 % | SIMD_UTILIZATION 0.9969 (FP8 matvec), 0.9965 (Q8_0 fused), 0.9976 (Q8_0 plain); SQ_BUSY_CYCLES per us 117,187, 106,329 and 157,142 (10 % and 34 % apart, but the dispatches have different shapes) | right in spirit, wrong by the letter of the 5 % rule |
| E1c | the FP8 mix draws 30 to 60 reported W more at matched bandwidth (the whole 57 W within 25 %: 55 %; half or more: 75 %) | loop-only mixes +13 W; thread-block-shaped mixes +30 to +33 W (53 to 58 % of 57 W); with random data +30 W (314 against 284 W, the FP8 probe at the power limit in 89 % of samples) | wrong for the loop-only probe (13 W is below 30); the block-shaped result is post hoc: those variants were added after the loop-only probe missed, and no expectation was committed before their first run (`r3m-five/`, 10:32), so "right" is not claimed; the whole 57 W not reached |
| E1d | (follow-up, random data) `fp8row` +15 to 40 W, `q8row` +5 to 15 W, `read` 0 to 10 W; the difference grows to +40 to 55 W (55 %) or stays within +-5 W of +33 W (30 %); bandwidth stays matched and unchanged by random data | `fp8row` +29.0 W, `q8row` +30.5 W, `read` +24.8 W; difference +30.0 W (stayed); the pair's bandwidth agrees (0.0 %) but both lose 1.7 to 2.2 % and `read` 0.8 %; the FP8 variant hit the power limit in 89 % of samples (the "could fail" case) | right on `fp8row` and on "stays" (the 30 % case); wrong on `q8row` and `read` (30 and 25 W, not 5 to 15 and 0 to 10) and on the memory rate |
| E2a | 10 to 40 % of the 8B's FP8 tensors exact in F16 (bits and range) | 31 of 224 (13.8 %; 15.4 % of the elements) | right |
| E2b | the mixed file is 20 to 28 GB and does not fit (75 %) | 27.86 GB; does not fit | right |
| E3 | machine code of the timed loops identical; the 4 % persists in an interleaved run; the cause is the state before the rows | identical for all three kernels (cmp of the machine code); -3.7 % (8 pairs, ranges separate); removing the 1 GiB known-answer read removes it (-0.5 %); compile command of the old binary not recorded anywhere | right (the compile-flag part could not be scored) |
| E4 | the same four counters work with 10.0 (70 %) | none works with 10.0 (abort at start-up); trace and listing work | wrong (closest is the 10 % case of the expectations file, "10.0 does not work with the system's runtime libraries at all"; it also failed on its own 10.0 libraries, so that reason is not what was seen) |
| E5a | a dependent A to B pair never overlaps on the in-kernel clock, plain or graph | 0 in 323,982 profiled and about 108,000 unprofiled pairs | right |
| E5b | the profiler shows overlaps in a tiny share of pairs, the in-kernel clock none: artifact (65 %) | 3 profiler overlaps in 323,982 pairs (one run of 18), in-kernel none: artifact | right |
| E5c | positive control overlaps on both clocks; negative control on neither | 1000 of 1000 on both; 3,999 pairs, none on either | right |
| E5d | clocks cannot be aligned to better than about 1 us, so the order within the GPU clock decides | a constant offset does not fit (0.6 to 12 us infeasible, plain and graph runs); the rates differ by 13 to 54 ppm; with the rate taken out it still does not fit: it misses by 35 to 115 ns in 11 of 12 runs of the 256 and 8192 KiB kernels, by 10.7 us in one (`r3p-graph-8192-prof-r2`), and by about 5 us for 2048 KiB | partly right (the expectation was an alignment error of about 1 us; no constant offset exists, and removing the rate difference nearly closes the gap for 256 and 8192 KiB only) |

## Q1: why FP8 patched generation draws about 57 W more than Q8_0

**What was open.** Round 1: the 8B patched FP8 generation sat at the 316 W cap (PPT0 in 1007 of 1157 samples of `beh/tg-8b-fp8-patched`, core 3.31 GHz) while Q8_0 drew 259 to 260 W (two sets of runs; the 3B Q8_0 drew 248 W) at 3.39 GHz, at 584 against 581 GB/s (bytes streamed per token times tokens per second, first set: 7,538,571,520 B x 77.53 tok/s = 584.5 for FP8 and 7,974,764,800 B x 72.83 tok/s = 580.8 for Q8_0; the second set gives 585.3 and 581.7; `models.txt` and `beh-tg-8b-*.log` of `20261118T094213Z-llm` and `20261118T110108Z-llm`). The extra power is 57 W in the first set and 56 W in the second (316 against 259 and 260 W). The instruction and memory counters read 0, so the cause was INFERRED (instruction mix).

**(a) The ISA of the two decode kernels (READ).** The code objects of the two builds (`libggml-hip.so`, sha256 of the extracted bundles in `q1/fatbin-evidence.txt`) were split and disassembled. The decode kernels are, from the round-1 kernel traces (`trace-tg128-8b-*`): patched FP8 `mul_mat_vec_f<ggml_fp8_e4m3_t, float, 1, 256, true, false>`
(88.3 % of the token's kernel time) and Q8_0 `mul_mat_vec_q<Q8_0, 1, true, false, false>` (76.2 %) and `<..., false, ...>` (18.2 %). Hot loops, counted by `scripts/llm3_isaclass.py` (full tables in `q1/q1-isa-counts.txt`; disassembly `q1/*.s`):

| loop | weights per iteration | vector instructions | per weight | by class (per weight) |
|---|---|---|---|---|
| patched FP8, no gate (one 16-byte load) | 16 | 69 | **4.31** | 1.0 `v_fma_mix_f32` (converts F16 to F32 inside the FMA), 2.19 integer bit operations (shift, and, or), 1.0 other (NaN check, address) |
| patched FP8, gate and up fused | 32 | 124 | **3.88** | 1.0 fma_mix, 2.19 bit operations, 0.7 other |
| Q8_0, `has_fusion=false` | 8 (two `v_dot4_i32_iu8`) | 14 | **1.75** | 0.25 dot, 0.25 fma, 0.125 convert, 1.0 add/compare (64-bit address arithmetic), 0.125 other |
| Q8_0, `has_fusion=true`, no gate | 8 | 16 | **2.0** | |
| Q8_0, `has_fusion=true`, gate and up fused | 16 | 24 | **1.5** | |

So the FP8 loop does about 1.9 to 2.9 times as many vector instructions per weight (4.31 or 3.88 against 2.0, 1.75 or 1.5: 1.9 at the lowest, 2.9 at the highest), not the 8 times of my expectation. What the counts leave out: the prologue and the block reduction (static kernel lengths 1022 instructions for the patched FP8 kernel, 210 and 654 for the two Q8_0 ones, all paths) and the clock. In a 4096-weight
row the FP8 kernel runs its loop once per thread, so the reduction is a large part of its work. Arithmetic from READ numbers (INFERRED): at 584 GB/s the FP8 loop issues about 2.5e12 vector lane-instructions per second (5.84e11 weights x 4.31), against 1.0 to 1.1e12 for Q8_0 (5.5e11 weights x 1.75 to 2.0); with the study's 6.7 pJ per flop of a fully loaded FMA card
(`scheduling-and-clocks.md`, 13.4 pJ per lane-FMA in reported power) that is about 33 W and 14 W: a difference of about 19 W, a third of the 57. The probe below tests this directly.

**(b) The counters that work, on the real decode kernels (MEASURED, `r3k-*`, `r3k-table.txt`; 8B, 16 generated tokens, medians over all dispatches of the kernel family).**

| kernel | dispatches | SIMD_UTILIZATION | SQ_BUSY_CYCLES per us | SQ_WAVES | GRBM_GUI_ACTIVE (cycles) |
|---|---|---|---|---|---|
| patched FP8 matvec_f | 3072 | 0.9969 | 117,187 | 32,768 | 94,035 |
| Q8_0 matvec_q, fused | 1520 | 0.9965 | 106,329 | 32,768 | 310,690 |
| Q8_0 matvec_q, plain | 1568 | 0.9976 | 157,142 | 8,192 | 43,957 |

Utilisation is 0.997 for all (the kernels are memory-bound and the counter saturates); the busy-cycle rate differs between kernels with the dispatch shape, not with the instruction mix. These counters cannot separate the two decode kernels.

**(c) The synthetic probe pair (MEASURED).** `src/power_mix.hip`: three streaming kernels over the same 1 GiB per launch (far above the 64 MB cache) with the real loops' instruction mixes: `fp8` (the patched kernel's widening function copied from its source plus its 16 FMAs and the NaN check), `q8` (Q8_0 blocks of 34 bytes, two int8 dots, one convert, one FMA),
`read` (a plain 16-byte read, the bandwidth probe's kernel); plus `fp8row` and `q8row`, the same loop bodies in the thread-block shape of the real kernels (one block per row of 4096 weights, the shuffle and shared-memory reduction at the end, one store per row).
Checks: known answers of every result (weights 1.0, activations 1.0, the sum must be exactly the number of weights; a deliberate break must FAIL), the read kernel within 10 % of 706 GB/s (the bandwidth probe's read rate in round 1: 705.0 median, 707.7 best), the ISA of the loops (`scripts/llm3_mix_isacheck.py`; the probe's FP8 loop has 41 bit operations and 16 `v_fma_mix_f32` per 16 weights, 4.0 vector instructions per weight; its Q8_0 loop 20 per 8 weights, 2.5 per weight,
slightly more than the real kernel's 1.75 to 2.0, which makes the FP8 excess in the table below smaller if anything). Each variant ran 10 s, six repetitions interleaved (`r3m-*`, the 20 ms sampler windowed by the probe's loop markers after the first second; the share of samples at the power limit is in the last column of the table; memory clock 1408 MHz). Paired differences (same repetition) in reported W:

| variant (6 repetitions, interleaved) | data | GB/s | reported W | core MHz | at the power limit |
|---|---|---|---|---|---|
| `fp8` (loop only) | constant | 683.5 | 260.0 [259 to 261] | 3394 | 0 % |
| `q8` (loop only) | constant | 678.9 | 247.0 [246 to 248] | 3405 | 0 % |
| `fp8row` (block shape) | constant | 673.3 | 285.0 [285 to 286] | 3366 | 0 % |
| `q8row` (block shape) | constant | 670.3 | 253.0 [252 to 254] | 3396 | 0 % |
| `read` | constant | 691.8 | 174.0 [173 to 174] | 3420 | 0 % |
| `fp8row` | random | 658.7 | 314.0 [314 to 315] | 3342 | **89 %** (78 to 94 %) |
| `q8row` | random | 659.2 | 284.0 [283 to 284] | 3374 | 0 % |
| `read` | random | 686.0 | 198.5 [198 to 199] | 3419 | 0 % |

(The two loop-only rows are from `r3m-five/` (the second run of that day, `r3m-five/r3m-table.txt`: `fp8` 260.0 W against `q8` 247.0 W); the very first run of the three loop-only variants is `r3m-first/` and gave 261 and 247 W at 675.6 GB/s. The block-shape and random rows are from the last run, `r3m-table.txt`, all six variants interleaved; the same block-shape variants in `r3m-five/` gave 287.5 and 255.0 W.)

| paired difference, same repetition | reported W median [min to max] | bandwidth difference | core clock difference |
|---|---|---|---|
| `fp8` minus `q8`, loop only, constant data (`r3m-five`; first run `r3m-first`: +13.5) | **+13.0** [12.0 to 15.0] | +0.5 % | -11 MHz |
| `fp8row` minus `q8row`, constant data (`r3m-five`: +33.0 [31 to 33]) | **+32.0** [31.0 to 33.0] | +0.5 % | -30 MHz |
| `fp8row` minus `q8row`, random data | **+30.0** [30.0 to 32.0] | 0.0 % | -32 MHz |
| random minus constant: `fp8row` / `q8row` / `read` | +29.0 / +30.5 / +24.8 | -2.2 / -1.7 / -0.8 % | -24 / -22 / -1 MHz |
| `fp8row` / `q8row` minus `read` (constant) | +111.0 / +79.2 | -2.6 / -3.1 % | |


Reading: at the same bandwidth (the pairs agree within 1 %) the instruction mix alone costs +13 W for the loop-only FP8 kernel over Q8_0 and +30 to +33 W in the real kernels' block shape (the block reduction and the store per row matter: they add 28 W to the FP8 kernel and 8 W to the Q8_0 one, `r3m-five`: `fp8row` minus `fp8` +27.5, `q8row` minus `q8` +8.0).
The core clock (run medians) was 3342 to 3420 MHz in all probe runs (the read kernel 3419 to 3421), so the clock does not explain the paired differences, except in the random FP8 run, which was held at the power limit (clock 3342 MHz, 32 MHz below the Q8_0 one: at equal clock its draw, and the difference, would be somewhat higher; how much was not measured).
**Random data** (the follow-up, expectations in `llm-workload-3-expectations-q1d.md`) raise every variant by 25 to 30 W (`read` +24.8, `fp8row` +29.0, `q8row` +30.5 reported W; bandwidth down 0.8 to 2.2 %) and leave the difference at +30 W: data-dependent switching is large (uniform random bits are a worst case; real weights are Gaussian-looking and toggle less, INFERRED) and it raised both kernels alike. For the FP8 kernel this is a lower bound: its random-data run was held at the power limit (89 % of samples, clock 32 MHz lower), so its uncapped draw and the true difference are higher by an amount that was not measured. Whether data-dependent switching explains part of the remaining 24 to 27 W is therefore not decided.
The real runs cannot be compared with the probes by absolute watts: they stream at 584 GB/s (round 1), the probes at 659 to 692 GB/s, and real FP8 decode (316 W) and the random-data `fp8row` (314 W) are both at the cap, so their match says nothing about the data. Real 8B Q8_0 decode (259 to 260 W) is near the constant-data `q8row` (253 W), also not a test of the data. So **the instruction mix and the block structure explain 30 to 33 of the 57 W (53 to 58 %)**; the other 24 to 27 W are not explained.
What is not explained, INFERRED and untested: the real kernels' extra paths (the fused gate and up path, which the probe does not have), address arithmetic, and the fact that the real FP8 run hit the 316 W cap (so its true demand at 3.39 GHz would be higher than the measured 57 W difference).

## Q2: the exact 8B comparison (mixed F16/F32 reference)

**What was open.** Round 2 built a 32 GB F32 reference for the 8B that could not be run on this machine (memory). The idea: store F16 where every FP8 x scale product of a tensor is exactly representable in F16, F32 elsewhere, and see whether the smaller file fits.
**What was measured (`scripts/llm3_mixed_ref.py plan`, run with the convert venv under the 10 GB cap; `q2-plan-8b.tsv`, `q2-vram.txt`).** For each of the 224 FP8 tensors of `llama31-8b-auto.gguf` the products of the codes present in that tensor with the tensor's scale were computed in float64 and each product round-tripped through float16 (this tests the significand bits and the exponent range at once; products below the F16 normal range lose bits as subnormals);
F32 is exact for all 224 (no NaN codes). Selftest (`scripts/llm3_mixed_ref.py selftest`, run by `study-llm.sh selftest`): scales of 1, 7 and 8 bits, a 12-bit product, a scale of 2^-20 (range), a scale of 2^8, a tensor that holds only 0 and 1 with an 8-bit scale (exact), plus the significand-bit function; a deliberately wrong expectation is part of it.
- **31 tensors** (1,077,936,128 elements, 15.4 % of the FP8 weights) are exact in F16; **193 tensors** (5,901,385,728 elements) need F32. Of the 193, 116 have a product that needs more than 11 significant bits (of the 193 tensors the scales have 8 bits in 127 and 7 bits in 64; all 116 have an 8-bit scale) and 77 need at most 11 bits but fall outside the F16 normal range (smallest product 3.2e-7 to 4.0e-6, median 1.7e-6, against the normal limit 6.1e-5).
- **Sizes:** F16 tensors 2.156 GB, F32 tensors 23.606 GB, the other tensors as stored (embedding and output in BF16, norms; the scale tensors dropped) 2.102 GB: **27.864 GB** (all F32: 30.020 GB; all F16: 16.061 GB).
- **Fit:** the card has 17.10 GB of memory with 3.43 GB used by the desktop at the time of the reading, so 13.67 GB free; the job cap is 10 GB with no swap (this includes the process's own memory and the page cache of the mapped file). 13.67 + 10 = 23.67 GB against 27.86 GB: **4.2 GB short with zero overhead** (the KV cache, the 262 MB of logits and the compute buffers are not counted, so the real shortfall is larger; by how much was not computed).
- **Stopped here.** No file was written, no model was loaded, no perplexity run, no bit-check (nothing to check), no swap use, no over-cap run. The 8B stays against the F16 reference of round 2 (indirect support only). An option for the owner: a machine with 64 GB, or a run with nothing else resident (the cap would then have to be raised, which is the owner's call).

## Q3: why the rebuilt `widen_cache` read about 4 % lower than the round-2 binary

**What was open.** In the fix round the same session gave 663.5 and 660.0 GB/s (round-2 binary) against 637.7 and 635.6 (new) on the widen row with the reused destination (S = 25 MiB).
**Code and build (READ, `q3/`).** The device code objects were extracted from both binaries (`llvm-objdump --offloading`) and disassembled: `widen_kernel`, `read_kernel` and `write_kernel` are **byte-identical** (67, 53 and 46 instruction words, `cmp` of the machine code), the same sizes (396, 304 and 248 bytes) and 256-byte aligned in both (their offsets in the code section
differ by 0x200 because the new binary also holds `empty_kernel`); the new binary adds only that kernel. Both carry the same compiler (AMD clang 24.0.0git d6f6cb69). The compile command of the round-2 binary was never recorded (the round-2 script said "build it with hipcc from src/widen_cache.hip"); the new one is `hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH="gfx1201" -save-temps`
(`r2mallbuild`, same flags as the round-3 `r3build` used for the third binary). The source differs only in host code (56 added, 9 removed lines: the 1 GiB known-answer read, the launch floor, `VERDICT` lines).
**Run (MEASURED, `r3q3-*`, `r3q3-table.txt`).** Three binaries through the gated runner at S = 25 MiB, 200 calls, 8 repetitions each, interleaved with the order rotated per repetition (24 jobs): `old` (round 2), `new` (the fix round's), `nocal` (the current source with the 1 GiB known-answer read skipped, a new argument; its timed kernels are byte-identical to `new`'s, `r3build`'s loop check passes).

| row (GB/s, median [min to max] of 8) | old | new | nocal | new against old (paired) | nocal against old (paired) |
|---|---|---|---|---|---|
| widen, destination reused | 660.2 [655.1 to 665.1] | 634.8 [633.0 to 637.9] | 657.2 [653.8 to 662.2] | **-3.7 % (-4.8 to -3.1; ranges separate)** | -0.5 % (ranges overlap) |
| widen, destination rotating | 607.5 [604.0 to 608.1] | 607.9 [604.0 to 609.6] | 602.1 [601.1 to 602.7] | +0.2 % | -0.9 % |
| read, rotating | 536.5 | 535.4 | 536.3 | -0.2 % | 0.0 % |
| write, reused | 1528.5 | 1528.5 | 1530.3 | 0.0 % | 0.0 % |
| write, rotating | 609.9 | 609.9 | 609.5 | 0.0 % | -0.1 % |

**Answer (MEASURED for what, INFERRED for why).** It is neither the code nor the build: it is the run-time state the 1 GiB known-answer read leaves behind, and it only touches the one row that reuses a destination while reading from a rotating pool. The spread of the "gain of a reused destination at S = 25" (12 % in round 2: 681 against 610 GB/s; 4.3 to 9.1 % in the fix round: 5.1 and 4.3 % with the new binary, 9.1 and 8.6 % with the round-2 binary) is, INFERRED, two states of the card: 9.2 % after a cold start (657.2 against 602.1 for `nocal`, medians) and 4.4 % after the 1 GiB read (634.8 against 607.9). Why the read changes the state (contents of the memory-side cache, address translation) is INFERRED and not examined.

## Q4: is the dead-counter result specific to ROCm 10.1

**What was open.** Round 2: of 13 counter sets only GRBM, SQ_WAVES, SQ_BUSY_CYCLES and SIMD_UTILIZATION return numbers (ROCm 10.1 `rocprofv3`, against the 256 MiB read kernel of the bandwidth probe); the memory and instruction counters read 0.
**What was measured (`r3c-*`, `r3sys.txt`).** The same 13 sets, the same kernel, with 10.1 (control, same session) and with 10.0's `rocprofv3` 1.3.5 from the venv (10.1 is 1.4.1). 10.0 was run "pure" (the app on the venv's 10.0 runtime libraries by `LD_LIBRARY_PATH`; `ldd` confirmed the 10.0 `libamdhip64` and `libhsa-runtime64`) for all 13 sets, "as is" (the app's own 10.1 runtime) for 3 sets,
and once with `ROCR_VISIBLE_DEVICES=0`. Jobs were limited to 25 s; a failed 10.0 run hangs in its signal handler until the time limit (exit 137 after 45 s). Two exploratory attempts before the mode existed (`q4-try1`, `q4-try2`, both exit 137) are in `summary.md` as FAILED; their durations and logs were not kept.

| counter set | 10.1 (same session) | 10.0 |
|---|---|---|
| GRBM_COUNT GRBM_GUI_ACTIVE | counts: 1.11e6 and 1.11e6 | no result: abort at start-up |
| SQ_WAVES SQ_BUSY_CYCLES | counts: 8192 and 3.47e7 | no result |
| SIMD_UTILIZATION | counts: 0.9994 | no result |
| the other 10 sets (SQ_INSTS_*, SQ_WAVE*, SQ_INST_CYCLES_*, GL2C_*, FETCH_SIZE, TCP_*, L2CacheHit, MemUnitBusy, VALUBusy) | accepted, read 0 on a kernel that moves 256 MiB (NONFUNCTIONAL), as in round 2 | no result |

10.0 prints `aqlprofile API table load failed: HSA_STATUS_ERROR` right after "HSA version 1.21.0 initialized", then SIGABRT, in all three variants (pure, as is, one device). Its **kernel trace works** (8 `read_kernel` rows, as 10.1). **Lists:** `rocprofv3 --list-avail` gives 146 counters for gfx1201 with 10.1 and 145 with 10.0; the only difference is `SIMD_UTILIZATION` (10.1 only). `rocprofv3-avail` of 10.0 does not run (it looks for `librocprofv3-list-avail.so`; the package holds `.so.1`: a packaging error;
not repaired, nothing was installed or linked), 10.1's does.
**Kernel and settings, read only (`r3sys.txt`):** kernel 7.3.4-202.nobara.fc44; `perf_event_paranoid` 2; `power_dpm_force_performance_level` auto; amdgpu parameters at their defaults except `ppfeaturemask=0xfff7ffff` (the overdrive bit, set for LACT); none of the readable amdgpu or kfd parameters is about counters; the card's directory has no perf-counter files; the kfd node of the card reports `debug_prop` 1495 (the integrated GPU the same).
**What differs / the answer.** The 10.0 profiler fails before any counter is read, on every set, so **the 10.0 result cannot show whether the zeros are a 10.1 matter**: it only shows that the one other profiler available here is worse. The cause of the 10.0 abort is not examined (a guess, INFERRED: the packaged `libhsa-amd-aqlprofile64` and the runtime of this driver do not fit). The cause of the zeros in 10.1 stays unknown.

## Q5: the sub-microsecond overlaps: real or timestamp artifact

**What was open.** Round 2 item 3: 5 of 60,984 neighbouring kernel pairs of one llama.cpp trace overlapped by 29 to 815 ns on one queue, none in 706,017 rows of 11 repeats; called "most likely a timestamp artifact" (INFERRED), because the `GGML_CUDA_GRAPH_OPT` control could not tell jitter from a small real overlap.
**The probe (`src/pair_overlap.hip`, MEASURED).** Kernel A writes a buffer and kernel B (same stream) reads what A's last blocks wrote and checks every value (B reads A's last blocks first, so an early B would see stale values); every wave of every kernel reads the REALTIME counter first and, after its stores completed (`s_wait_storecnt`), last. Sequence A0 B0 A1 B1 ...:
(A_i, B_i) are dependent pairs (like a KV write followed by the attention that reads it), (B_i, A_i+1) independent neighbours. The counter: gfx12 has no `s_memrealtime` (the compiler refuses the builtin for gfx1201); `__builtin_readsteadycounter()` lowers to `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, the same 100 MHz counter, 10 ns per tick (deviation from the task text, stated).
Checks: the counter ticks 100.00 to 100.09 MHz against the host clock in the 28 runs (known answer, 1 %; unprofiled runs 100.003 to 100.005, profiled up to 100.087); the data check passes in every run and a deliberate break (B expects the wrong tag) FAILs (8,192,000 mismatches; **the first version of the break did not fail**: an error counter reset after the loop made the data check vacuous; found by the break, fixed, committed after); ISA check of the order of the clock reads and the stores (`llm3_pair_isacheck.py`, two deliberate breaks FAIL).
Kernel shapes: 256 KiB, 2 MiB and 8 MiB per kernel, 32, 256 and 1024 blocks of 256 threads: in-kernel spans (first wave start to last wave end) of 0.8 to 1.1 us, 5.1 us and 20.4 us. Plain launches on one stream and a HIP graph of 100 pairs (200 kernel nodes, launched many times; host sync between graph launches). The neighbour counts of the graph rows include the pairs across a launch boundary (a host sync lies between them, about 1 %: 59,997 in the join against 59,400 in the probe's own count); the overlap counts do not change, and the three profiler overlaps have in-kernel gaps of 3.4 us, as inside a launch, not 16 us or more.
Under `rocprofv3 --kernel-trace` (three runs of every shape) and without it (one run). `scripts/llm3_pairs.py` joins the probe's per-kernel stamps (earliest first stamp, latest last stamp over the waves) with the trace rows (the last N rows of the same names, kinds checked).

**Per pair, two gaps on two clocks (no alignment needed for the verdict):** profiler gap = start(B) - end(A); in-kernel gap = first REALTIME read of B's earliest wave - last read of A's latest wave. A gap below 0 is an overlap on that clock; within one tick (10 ns) it would be ambiguous (none occurred).

| run (profiled, 3 repetitions summed) | pairs | profiler overlaps | in-kernel overlaps | both | profiler only |
|---|---|---|---|---|---|
| plain, 256 KiB | 60,000 dependent + 59,997 neighbour | 0 | 0 | 0 | 0 |
| plain, 2 MiB | 15,000 + 14,997 | 0 | 0 | 0 | 0 |
| plain, 8 MiB | 6,000 + 5,997 | 0 | 0 | 0 | 0 |
| graph, 256 KiB | 60,000 + 59,997 | 0 | 0 | 0 | 0 |
| graph, 2 MiB | 15,000 + 14,997 | 0 | 0 | 0 | 0 |
| graph, 8 MiB | 6,000 + 5,997 | **3** (2 dependent, 1 neighbour: 490, 465 and 8,040 ns) | 0 | 0 | **3** |
| **total plain and graph** | **323,982** | **3** | **0** | **0** | **3** |
| negative control: host sync between all kernels (`r3p-hostsync-256-prof-r1`) | 2,000 + 1,999 | 0 | 0 | 0 | 0 |
| positive control: two independent spin kernels on two streams, 200 us each (`r3p-twostreams-256-prof-r1`) | 1,000 rounds | 1,000 (200 to 820 us) | 1,000 (median 400 us) | 1,000 | 0 |

The three profiler-only overlaps are all in one run (`r3p-graph-8192-prof-r2`; the other two runs of that shape have none). The in-kernel gaps of those three pairs are **+3,360, +3,400 and +3,440 ns**: the second kernel's first wave started 3.4 us after the first kernel's last wave had finished, and the profiler still put the end of the first kernel 465 to 8,040 ns after the start of the second.
Normally the profiler's gap is the in-kernel gap minus a margin (the stamp of the last wave to the profiler's end stamp, plus the profiler's start stamp to the first wave's stamp): median 440 ns, minimum 320 to 360 ns for the 256 KiB and 8 MiB kernels in every run but one (`r3p-graph-8192-prof-r2`, the run with the three overlaps: minimum -8,370 and -2,945 ns), 5.6 to 6.0 us (median) for the 2 MiB kernels (a fixed offset of unknown cause; it is also why the clocks do not align there).
The margin has a long upper tail in every run (maximum 3.1 to 15.0 us, `margin_ns` lines of `r3p-join.txt`): the profiler's gap can be much smaller than the waves' gap. In these three pairs it was 3,850, 3,865 and 11,480 ns; elsewhere in that run it also reached -8.4 and -2.9 us (the profiler's gap larger than the waves'). So the profiler's stamps jitter by several us in both directions,
which is what makes a 29 to 815 ns "overlap" possible without any overlap of the waves. Under the profiler the in-kernel gap is larger (median 3.4 us, smallest 3.2 us: the profiler slows the dispatch). Without the profiler (`r3p-*-noprof-r1`: 108,000 pairs) there is no in-kernel overlap either and the smallest gap is 1.8 us (median 1.84 to 1.88 us for 256 KiB and 8 MiB kernels, 7.2 to 7.5 us for 2 MiB; 16.4 us minimum with a host sync). The data check passed in all 28 pair runs.
**Mapping between the clocks (d).** The REALTIME counter x 10 ns and the profiler's nanoseconds do not share a constant offset: the interval of offsets that keeps every dispatch's first stamp after its profiler start and last stamp before its profiler end is empty in every run (by 0.6 to 12 us), and the run spans differ by 13 to 54 ppm (in-kernel / profiler 0.999946 to 0.999987 over 98 to 208 ms). With that rate taken out a single offset still does not fit: the interval is empty by 35 to 115 ns in 11 of the 12 runs of the 256 KiB and 8 MiB shapes (`NO-OFFSET-FITS` in `r3p-join.txt`), by 10.7 us in the twelfth (`r3p-graph-8192-prof-r2`) and by about 5 us in the 2 MiB runs. So the clocks were **not aligned** beyond a few us; the verdict compares the order inside each clock. Resolution: in-kernel 10 ns; the profiler's nanosecond stamps are not that precise (the jitter above is 3.9 us or more in the cases seen).

**Answer.** Profiler overlaps between neighbouring dispatches of the synthetic probe are **timestamp artifacts** (MEASURED; INFERRED, not tested, for llama.cpp's own five of round 2): where the profiler showed one, the independent clock showed the waves 3.4 us apart, in 323,982 profiled and about 108,000 unprofiled pairs not one pair overlapped in execution, and a real overlap (positive control) is seen on both clocks. The artifact rate is 3 in 323,982 pairs (9e-6, in 1 of 18 profiled runs of plain and graph kernels), against 5 in 60,984 (8e-5) in one llama.cpp trace of round 2; the sizes agree in kind (465 and 490 ns against 29 to 815 ns; one of 8,040 ns), which is the reason to infer the same for llama.cpp's five.
Limits: the probe's kernels are not llama.cpp's, so the five overlaps of round 2 themselves were not re-timed in-kernel (that would need a change in llama.cpp); the evidence is that the same profiler produces such overlaps at similar rates on kernels with a known, independent order. The artifacts appeared in one run only (the 8 MiB graph shape, run 2 of 3).
**Round 2 and the check.** `llm-workload-2.md` item 3 and plain-summary item 3 now carry an "Updated in round 3" note (MEASURED for the synthetic probe; the decision for llama.cpp's own five stays INFERRED, now with a measured basis). The `no-overlap` check is **not changed**: the evidence says the profiler's stamps overlap rarely without any execution overlap, and the check reports exactly that (a `FAIL` with a list from `llm_kernels.py overlap`); it is a false-alarm rate of about 1e-5 per pair, not a wrong check.

## Deliberate breaks and selftest (round 3)

`scripts/study-llm.sh selftest` (about 90 s, no GPU) ends "every deliberate break failed as intended, every control passed" and now also runs the selftests of `llm3_pairs.py` (a clean pair set, a planted profiler-only overlap, a planted real overlap, wrong kind order and too few rows are refused), `llm3_cnt.py` (known median, a regex that matches nothing fails), `llm3_power.py` (the paired difference of 300 against 250 W is +50.0, a window with 30 samples and a log without loop markers are refused),
`llm3_q3.py` (a planted 4 % gap is found, five repetitions of a binary are refused) and `llm3_mixed_ref.py` (hand-worked exactness cases, the 12-bit product, range cases, a wrong expectation). The `r3build` phase (CPU) builds the probes and checks their ISA, with FAIL seen for: the pair probe's order and count breaks, the power probe's dot-count break. The `r3check` phase (GPU) runs the probes' own breaks on the card, FAIL seen for each:
the pair probe's data check, the power probe's known answers (constant data for `fp8` and `q8row`, random data for `fp8row` and `q8row`) and the clock known answer PASS. Fix round: the power probe's known answer must not depend on fresh VRAM being zero. It now zeroes its result buffer and sums only the entries the kernel writes; the `dirty` argument fills the buffer with 1.0f first. Phase `r3fix` (`runs/20261118T191700Z-llm3fix/`, through the gated runner): the code before the fix with only `dirty` added FAILS on `q8row` (sum 1,010,592,829 against 1,010,577,408); the fixed probe PASSES on the dirty buffer and on a clean one, with the same sums as round 3 (no number moved). Not done this round: the round-2 style mutation test of every new check in a copy (each check was seen to fail on a deliberate break instead).

## Parked, skipped and open

- **8B exact reference**: stopped by the numbers (Q2); needs 64 GB or the owner raising the job cap.
- **Why the 10.0 profiler aborts and why 10.1's memory and instruction counters read 0**: not examined (Q4); no install was allowed.
- **The remaining 24 W of the 57** (Q1) after the instruction-mix probe: the fused gate and up path and the power cap are the candidates, untested.
- **Why the 1 GiB read lowers the reused-destination widen row** (Q3): INFERRED cache state, not examined.
- **The llama.cpp overlaps themselves** were not re-timed in-kernel (Q5); the evidence is from a synthetic probe.
- The two exploratory counter attempts before `r3counters` existed (both exit 137; durations not kept) appear as FAILED in `summary.md`; the 10.0 counter jobs all end FAILED by design (the finding) and do not count against the run.

## How to rerun

```
scripts/study-llm.sh r3sys && scripts/study-llm.sh r3counters                         # Q4 (about 13 minutes of job time, from the `r3c-*.meta` wall times: the 10.0 jobs hang until their 25 s limit)
scripts/study-llm.sh r3build && scripts/study-llm.sh r3check                          # probes, ISA checks, breaks (CPU, then GPU); r3fix (after r3build): known answer on a dirty buffer
scripts/study-llm.sh r3q3                                                             # Q3: three binaries interleaved (needs the round-2 binary in runs/20261118T165826Z-llm2fix/)
scripts/study-llm.sh r3pairs                                                          # Q5 (python3 -I scripts/llm3_pairs.py join <dump> <trace> joins one run again)
R3P_VARIANTS="fp8row q8row read fp8rowR q8rowR readR" scripts/study-llm.sh r3power    # Q1c (default variants: fp8 q8 read fp8row q8row), table by scripts/llm3_power.py
scripts/study-llm.sh r3cnt                                                            # Q1b
toolchains/convert-venv/bin/python -I scripts/llm3_mixed_ref.py plan <fp8.gguf>       # Q2 (CPU); the disassembly of Q1a: scripts/llm3_isaclass.py <objdump of one kernel> --loop 0 --weights-per-iter 16
```
All with `LLM_RUN=runs/<folder>` to continue in one folder; do not edit `scripts/study-llm.sh` while a phase runs.

## Proposed lines for `README.md` and `notes/STUDY-PLAN.md` (not applied; those are the lead's files)

**STUDY-PLAN.md, status paragraph (add after the LLM data dive round 2 line):**

> LLM data dive, round 3 done (branch `study/llm-dive`, run `runs/20261118T175910Z-llm3`, `measurements/llm-workload-3.md`, expectations in `llm-workload-3-expectations.md` and `-q1d.md`, written before the GPU jobs): the extra 57 W of FP8 decode
> is explained to 30 to 33 W by its instruction mix and block structure (ISA read, matched-bandwidth probe pair, random and constant data; the random-data difference is a lower bound because the FP8 probe was power-capped), the mixed F16/F32 reference for the 8B is 27.9 GB and does not fit under the 10 GB job cap, the 4 % lower reading of the rebuilt cache probe is the 1 GiB read
> before its rows (machine code identical), ROCm 10.0's profiler cannot read any counter on this card, and the sub-microsecond kernel overlaps are timestamp artifacts on a synthetic pair probe (an independent in-kernel clock shows none in 323,982 pairs; INFERRED, not re-timed, for llama.cpp's own five). Memory clock 1408 MHz throughout.

**README.md, key-finding bullets:**

> - **The extra 57 W of FP8 decode is half instruction mix.** The patched FP8 decode loop does 3.9 to 4.3 vector instructions per weight against 1.5 to 2.0 for Q8_0 (the F16 conversion is folded into the FMA); synthetic kernels with those mixes in the real kernels' thread-block shape, streaming the same bytes at the same bandwidth,
>   differ by 30 to 33 W, with constant or random data alike. The remaining 24 to 27 W are not explained. ([llm-workload-3.md](measurements/llm-workload-3.md))
> - **Profiler kernel overlaps of under a microsecond are timestamp artifacts (on a synthetic pair probe).** An in-kernel 100 MHz counter shows no overlap in 323,982 dependent kernel pairs; the three overlaps the profiler reported had the real kernels 3.4 us apart. Real concurrency shows on both clocks. llama.cpp's own five overlaps were not re-timed (INFERRED to be the same kind).

**README.md, measurement-file table rows:**

> | `llm-workload-3.md` | LLM data dive, round 3: the 57 W of FP8 decode (ISA and a power probe), the mixed exact reference (does not fit), the cache probe's 4 %, ROCm 10.0 counters, sub-microsecond overlaps on an independent clock |
> | `llm-workload-3-expectations.md`, `llm-workload-3-expectations-q1d.md` | The expectations for round 3, written before the GPU jobs of each question |
