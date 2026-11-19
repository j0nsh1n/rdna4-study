# Leftovers C, lane B: matrix rules, scheduling, special loads, Vulkan, arithmetic errors

Card: RX 9070 XT (gfx1201), LACT profile "Testing", ROCm 10.1 toolchain, kernel 7.3.4. Measured 2026-11-17 by the
lane B agent of "leftovers C" (plan: `rdna3-fp8/build/rdna4-runs/plans/leftovers-c-plan.md`, section 3 "Lane B").
Script: `scripts/study-leftovers-c-b.sh`. Final runs: `runs/20261117T210256Z-leftc-b` (every item without a whole-card load,
and the breaks) and `runs/*-leftc-b-card` (the whole-card bursts, see "Not measured"). Folders named `runs/dev-*` are
development runs, cited where they differ.

## In short

- A long kernel whose waves hold their SIMDs while sleeping in `s_sleep` is not preempted by another process's kernel,
  nor by one on a higher-priority stream: the second kernel always started after the first ended (kernels of 4 to
  180 ms, 64 of 64 trials per setting; no positive control for the overlap detection, see "Not measured").
- The forbidden back-to-back matrix read follows the earlier rule exactly for mixed forms too: the second instruction
  reads its first two operand registers of lanes 0-15 before the first instruction has written them. In wave64 the
  "as B" case maps over unchanged; "as A" hits rows 0, 4, 8, 12 instead of 0, 1, 8, 9 (that lane set was fitted to the
  first run's observation, then reproduced: one run of 64 identical waves).
- Every matrix instruction tried (20 forms, wave32 and wave64) ignores the lane mask: all lanes are read and written and
  the time does not change. The transposing loads ignore the mask too.
- The ignored sign bits (NEG_HI) stay ignored on the remaining integer forms; the accumulator sign bits work on every
  16-bit sparse form and are ignored on the FP8 and BF8 sparse forms, as on their dense twins.
- The scalar cache fetches 64-byte lines (not 128 as inferred before) and is shared by the two compute units of a
  WGP; behind it sits one card-wide level (about 200 cycles, the same from every other compute unit).
- Up to 63 transposing loads can be in flight, the same counter limit as plain loads.
- Applying the per-tensor FP8 scale in FP32 adds one rounding. Pooled over 12 tensors, the (very rare) FP16 changes go
  up about 3 times (FP16 path), 10 times (BF16) at K = 16, and for the FP8 path from 0 to 22 of 2.4 million (no ratio);
  at K = 256 they go from 12 to 49 of 720 k for the FP8 path (about 4 times) and do not rise for the FP16 and BF16
  paths (0.83 and 0.89 times); at K = 4096 no change is visible in 24 k samples. Rounding to BF16 changes about a tenth as
  many values as rounding to FP16.
- Without subnormal inputs the worst single-instruction case shrinks from up to 7,110 FP16 steps to 5; chains of
  1,024 products reach at most 46 FP16 steps and 1 E4M3 step. NaN and infinity follow the IEEE classes in 880,000 of
  880,000 cases.
- `s_setprio` levels 0 to 3 form a strict order: in every pair of levels the higher one runs first. A hand-over
  between two waves' matrix chains costs about 105 to 110 cycles for every 16-cycle dense form, 13 for every 8-cycle
  form and 77 for sparse FP16.
- In Vulkan, the accumulator rounding rule (round to nearest even after each group of 4 or 8 products) predicts all
  8,448 results with odd accumulator values too.
- Under whole-card load (one launch, four pair waves and four chain waves on every SIMD) the forbidden matrix read is
  no longer certain: 222 to 245 of 512 waves per pair read the stale words (exactly the rule's; lower bounds where old
  and new words can coincide), the rest the right values, never a mixture. Whether the cause is the busy card or the
  four pair waves sharing a SIMD is not separated.
- Every remaining matrix form reaches 128 SIMDs x clock / cycles: within 0.12 % in HIP (MT-3), and within 1.1 % in
  Vulkan for the mixed FP8/BF8 and saturating 8-bit forms (as FP8 itself there). The 1 to 2 % FP16 loss with 16 waves
  per SIMD is absent in 50 ms launches (-0.2 %) and is 1.0 % at 100 ms and 1.7 % at 200 ms; it goes with a longer
  launch and a higher clock together, and these runs do not separate the two (3 bursts per cell, the lead's rerun
  agrees in direction).
- In the first 0.1 s of a heavy load the only throttle bit is the hotspot one, set from about 50 ms on (at 66 to 67 C
  in the four short bursts).
- Vulkan: 8 added per-tile vector operations cost wave64 much more than wave32 in both passes (second pass 19 % against
  1.3 % per clock; first pass 45 % against 1.7 % in operand rate); that wave64's 8 % loss in the load loop shrinks to 3.7 %
  with a shorter loop holds in the second pass and the rerun but not in the first pass (11 %); a GLSL software pipeline
  does not survive ACO, but restructuring the loop made ACO move the address code out of it (39 to 22 instructions) and
  the loop faster by 5 to 20 % in four of five cases (not for FP16 from the 32 MiB window).
- The transposing-load "32 MiB" figure of 5.3 TB/s was not reproduced (9.5 TB/s here, and in the lead's rerun) and is
  unexplained; with the calibration failed, "neighbouring waves share lines" is not supported in this run (the unique
  walk read 7.3 TB/s against the 2.4 TB/s it predicted), not ruled out.
- The card-wide stalls of two processes coincide only when the traces are aligned: shifted by 5 to 100 ms the
  coincidence falls from 98.8 % to at most 0.4 % (chance: 0.26 %). The hotspot throttle bit appears early in a load
  (all 23 set samples of the kept traces within 0.8 s of a load's start, from two load onsets; all four short bursts at
  about 50 ms and 66 to 67 C). The data do not separate "time since the load started", "rate of rise" and "a threshold
  near 66 C".

## SC-11. Preemption of long kernels and compute-queue priority

Question: does a second kernel start while a long kernel of another process (or of the same process, from a
higher-priority stream) is running?

Expected: READ (research-9, section 7): queue priority in the GC 12 queue manager is disabled since 2026-02. The
queue descriptor does enable a time quantum (`cp_hqd_quantum = QUANTUM_EN ...`, research-9 section 3.1), which
governs how long one queue dispatches before another, not whether running waves are saved. MEASURED earlier
(scheduling item 7b): two whole-card kernels of 2 ms alternate whole launches. Expectation: the second kernel waits
for the first to end, at every length and priority.

Probe: `sched_share preempt` (new mode). Kernel A: 64 groups of 2 waves, 40 KiB of LDS each (CU mode: one group per
CU, and no second group of this shape fits beside it); its waves sleep in `s_sleep 127` steps and read REALTIME until
T_A has passed. Kernel B: the same shape for 1 ms, launched 1 ms after A. Every wave records REALTIME at start and
end and its longest gap between two REALTIME reads (a wave that is saved and restored shows a gap). Every fifth trial
A launches nothing: B alone is the control. 16 overlap and 4 control trials per T_A of 4, 16, 64 and 180 ms. Three
settings: two processes with B on a default stream; two processes with B on the highest-priority stream
(`hipStreamCreateWithPriority`, range 1 to -1); one process, A on the lowest and B on the highest priority. The
waves sleep, so the card is expected to draw near-idle power (not recorded during the jobs); every kernel stays below 0.2 s.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (`sc11-pair-normal.log`, `sc11-pair-high.log`,
`sc11-same-process.log`; the table below is rebuilt from them; the development run `runs/dev-sc11b` also gave 64 of 64
in its two settings; the lead's rerun gave 63 of 63 usable trials in each of its settings and in one trial of the
default-priority setting at T_A = 4 ms A had not yet started when B launched):

| Setting | Trials where B waited for A's end | B start minus A end, us (median / p10 / p90 / min / max, over the 4 T_A) | A's longest wave gap |
|---|---|---|---|
| two processes, default priority | 64 of 64 | 27.5 to 28.7 / 19.4 to 23.0 / 269 to 310 / -1.48 / 311.5 | median 3.1 to 3.2 (largest 3.52) |
| two processes, B highest priority | 64 of 64 | 25.5 to 28.2 / 18.2 to 21.9 / 122 to 297 / -1.36 / 322.5 | median 3.0 to 3.2 (largest 3.64) |
| one process, A lowest, B highest | 64 of 64 | 28.1 to 31.0 / 20.5 to 23.8 / 62 to 299 / 2.56 / 304.1 | median 3.0 to 3.2 (largest 4.08) |
| control: B alone, launch to first wave | (16 per setting) | median 47 to 55 us; largest 320 to 352 us | - |

(Ranges are over the four T_A values; min and max are over all 64 trials of the setting; per-T_A rows are in the logs.
"A's longest wave gap" is the median over the 16 trials per T_A, then its range; the largest single value in
brackets. The lead's rerun gave medians of 23.3 to 32.6 us in all three settings.)

Answer: B started when A's groups ended in every trial, for kernels of 4 to 180 ms, and stream priority changed
nothing, whether B came from another process or from a higher-priority stream of the same process. This was seen with
A's waves sleeping in `s_sleep 127` steps: they hold their SIMDs and a CU's LDS while asleep, so the result is limited to
that state: "a kernel whose waves hold their SIMDs while sleeping is not preempted". It does not show that a kernel
of busy waves would behave the same (the plan's FMA chain was replaced by the sleep, see "Changes to the draft plan").
The few slightly negative values (down to -1.48 us) are B's first group starting on a CU that one of A's groups had already
freed while A's last group was still finishing (the groups do not end at the same instant). The gap is bimodal
(about 20 us or about 300 us), like B's own launch after idle (scheduling item 1: medians 55 to 71 us, tails to
372 us). MEASURED. That the time quantum in the queue descriptor does not lead to wave save and
restore at these lengths is INFERRED from the absence of any gap in A's waves (largest gap 4.08 us in the final run).

There is no positive control: nothing in this item shows that the overlap detection can report an early start (a
second kernel that fits beside A, seen to start early, was not run), and the only deliberate break changes the
calibration limit (to 1 us), not the detection. So "never overlapped" is a result of this probe as built, not a
tested ability to see overlap.

Side finding (MEASURED, first version of the probe, `runs/dev-sc11`): with `s_sleep 4` between REALTIME reads, the
second wave of a group often did not get its first REALTIME read for the whole 1 ms its sibling was polling (both in
B alone and in A), which looked like preemption (2.8 ms gaps, B starting 2 ms before "A's end"). With `s_sleep 127`
every wave of a group starts within 0.2 us (development run) and no gap exceeds 4.1 us (final run; 3.3 us in the
development run). So one wave polling REALTIME tightly can hold
another wave's REALTIME message for a long time (one development run; not repeated in the final runs); probes that poll
it tightly should not judge other waves' start times.

Label: MEASURED (no preemption and no priority effect, for sleeping waves, up to 180 ms); INFERRED (why).

## MA-2. The forbidden back-to-back read: other forms, wave64, a busy card

Question: does the forbidden read of matrix-arithmetic section 13 (a matrix result used as A or B by the next matrix
instruction with nothing between) give the same stale words when the second instruction is another form, in wave64,
and under whole-card load?

Expected (matrix-arithmetic 13a, MEASURED): the stale words are fixed by where the second instruction reads: its
operand VGPRs 0 and 1 of lanes 0-15 are read old; as A only rows 0, 1, 8, 9 show it (those rows' operands), as B every
column does. Made exact here: the second instruction alone, fed exactly those words old and everything else new,
must give the forbidden result word for word. How many results differ depends on the data (an old word can give the
same product), so counts are reported, not predicted.

Probe: `left_mat hazard2` (wave32: 8 new pairs with the second instruction of another form, each alone and beside a
busy wave on the same SIMD; section 13's two cases as calibration), `left_mat hazard2 --card` (every pair under
whole-card load: 128 groups of 8 waves, 4 running the pair and 4 a long FP16 matrix chain on the same SIMDs), new
`left_mat64 hazard` (wave64 build: FP16 into FP16 feeding FP16 into FP32, and the 8-bit form feeding itself, as A and
as B). Fixed registers, one inline-assembly statement per pair; the code object holds every pair back to back and
the control with one `v_nop` (listing check). 64 waves each.

MEASURED, wave32, final run `runs/20261117T210256Z-leftc-b` (`ma2-hazard2.log`; every count equals the development run `runs/dev-ma2b`):

| First result used as | Results differing from the V_NOP version (alone / busy partner) | Differ from the predicted forbidden result |
|---|---|---|
| FP16 into FP16 as A of FP16 (calibration, section 13) | 4,096 / - of 16,384 | 0 |
| FP16 into FP16 as B of FP16 (calibration) | 16,384 / - | 0 |
| FP16 into FP32, VGPRs 0-3 as A of FP16 into FP16 | 2,547 / 2,637 | 0 / 0 |
| same, as B | 10,113 / 10,132 | 0 / 0 |
| FP8 result, VGPRs 0-3 as A of FP16 into FP32 | 3,360 / 3,440 | 0 / 0 |
| same, as B | 13,567 / 13,471 | 0 / 0 |
| 8-bit integer result, VGPRs 0-3 as A of BF16 into BF16 | 1,397 / 1,504 | 0 / 0 |
| same, as B | 5,657 / 5,752 | 0 / 0 |
| FP16 into FP16, VGPRs 0-1 as A of FP8 | 4,016 / 4,016 | 0 / 0 |
| same, as B | 15,904 / 15,856 | 0 / 0 |

In every case the first result itself is written correctly and the V_NOP version equals the second instruction run
alone on it.

MEASURED, wave64, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-w64c`) (64 waves, 16,384 result words per case):

| Case | Words differing from the V_NOP version | Rows (A) or columns (B) affected | Wave32 rule mapped through the wave64 layout | Fitted: lanes 0, 4, 8, 12 |
|---|---|---|---|---|
| FP16 into FP16 as A of FP16 | 4,096 | rows 0, 4, 8, 12 | does not hold (4,096 differ) | exact (0 differ) |
| FP16 into FP16 as B of FP16 | 16,384 | all 16 columns | exact (0 differ) | - |
| 8-bit integer as A of itself | 4,096 | rows 0, 4, 8, 12 | does not hold | exact |
| 8-bit integer as B of itself | 16,384 | all 16 columns | exact | - |

(The mapping uses the wave64 register layout of compiler-and-hazards section 8: wave32 lane l, VGPR k of an operand
of 2n VGPRs sits in wave64 lane l + 32 (k / n), VGPR k % n.)

Answer: in wave32 the rule holds for every mixed pair tried, alone and beside a busy wave: the second instruction
reads operand VGPRs 0 and 1 of lanes 0-15 before the first instruction's result lands, whatever the two forms are
(MEASURED). In wave64 the B case is the same register positions; the A case reads old words in lanes 0, 4, 8, 12
(and 32, 36, 40, 44 for the 8-bit form's second half), so rows 0, 4, 8, 12 are wrong instead of 0, 1, 8, 9 (MEASURED;
the lane set was fitted to the first run's observation and then reproduced exactly by the final run: that is a fit
that repeats, not an independent prediction; one launch of 64 identical waves per case).

Whole-card load, MEASURED, run `runs/20261117T210336Z-leftc-b-card` (`ma2-hazard2-card.log`; one launch, 512 pair waves per pair on
all 128 SIMDs, so four pair waves per SIMD, each SIMD also holding four waves of a long FP16 matrix chain; each wave
classified against the rule's prediction and against the all-new result):

| Pairs | Waves reading old words exactly by the rule | Waves reading only new words | Neither |
|---|---|---|---|
| all 16 pairs (section 13's two, section 13a's six, the eight new ones) | 222 to 245 of 512 | 267 to 290 of 512 | 0 |

Under whole-card load the forbidden read is no longer certain: in every pair a little under half of the waves (222 to
245 of 512) read the stale words, exactly the rule's words, and the rest read the right (new) values, never a
mixture (0 of 512 in every pair). Beside one busy wave on its SIMD (section 13a and the table above) every wave read
stale words. Two cautions on the counts. First, a wave whose stale words happen to equal the new ones is counted as
"new", so the stale counts are lower bounds where the data can make the old and new words equal: the 8-bit-integer
into BF16 pair has the lowest count as A (222; as B 239) and the fewest differing result words (5,059 of 131,072 as A);
the other 14 pairs read 236 to 245 (as A and as B). Second, the whole-card setting puts four pair waves on every SIMD
at once, and the single launch has no repeat and no deliberate break of its own (`break-ma2-hazard2` runs without
`--card`). Why only about half read stale words is therefore open. The explanation that fits what was seen and is the
simplest is that four pair waves on one SIMD interleave their instructions, so a wave's second instruction often does
not directly follow its first (INFERRED, not tested: a launch with one pair wave per SIMD next to the same chain waves
would separate it from the card-wide load). The earlier reading, "the matrix unit's issue timing under contention" of
the rest of the card, is not supported over that one. A first version of this job in the same folder (its verdict lines
remain in that folder's `summary.md`; the rerun replaced its log) showed this in its counts (15,232 of at most 32,768
results as A), and its control verdict failed on my own check: it compared the 512 waves that share a SIMD with a busy
wave against the 64 fitted waves. Fixed and rerun.

Label: MEASURED (wave32 and wave64 rules; the half-and-half split in this one launch); INFERRED, untested (why).

## CH-6. Matrix instructions under a lane mask: every form, wave32 and wave64, and their time

Question: does every matrix opcode ignore EXEC, in wave32 and wave64, and does the time change?

Expected: AMD's guide forces EXEC to all ones in the matrix pseudocode (READ); measured for five opcodes in wave32
(compiler-and-hazards section 1). Chained costs from matrix-timing sections 2 and 3: 16 cycles for the 16-bit-input
forms, 8 for the 8-bit and 4-bit forms.

Probe: new header `src/study/left_mask.h`, used by `left_mat mask2` (wave32) and `left_mat64 mask` (wave64): 20
matrix opcodes (all 11 dense and 9 of the 11 sparse forms: FP16, BF16, FP16 into FP16, BF16 into BF16, FP8, FP8 x
BF8, BF8 x FP8, BF8, 8-bit, 4-bit K = 16 and 32; sparse FP16, BF16, FP16 into FP16, BF16 into BF16, FP8, BF8, 8-bit,
4-bit K = 32 and 64) and two dot instructions as the controls that obey the mask. EXEC set around one instruction
(7 masks, 0 included), D preset to a sentinel, each launch twice. Timing: chains of 16 and 48 instructions between
two reads of the cycle counter with EXEC all ones and with only lane 0, best of 7 each.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development runs `runs/dev-ma2b`, `runs/dev-w64b`):

| | Wave32 | Wave64 |
|---|---|---|
| Matrix forms whose every result word equals the full-EXEC result under all 7 masks (switched-off lanes written) | 20 of 20 | 20 of 20 |
| Dot controls obeying the mask (switched-off lanes untouched) | 2 of 2 | 2 of 2 |
| Chained cycles per instruction, full EXEC / one lane | 16.000 / 16.000 (16-bit inputs), 8.000 / 8.000 (8- and 4-bit) for every form | the same |

Answer: every matrix opcode reads every lane's operands and writes every lane's result whatever EXEC says, in wave32
and wave64, and the time is the same to the cycle. MEASURED.

## MA-1 and MA-3. Modifier bits on the remaining forms

Question: (MA-1) is NEG_HI on B ignored on the 4-bit forms and unsigned 8-bit, as on A? (MA-3) do the C bits act on
the sparse forms other than FP16 into FP32, and is NEG_HI ignored on the unsigned 4-bit K = 32 form?

Expected: NEG_HI ignored (matrix-arithmetic 9a, MEASURED on A for these forms, on A and B for signed 8-bit). C bits:
NEG flips and ABS clears C's sign on the 16-bit sparse forms, as on the sparse FP16-into-FP32 form and every dense
16-bit form (9 and 9a); FP8 and BF8 sparse: unknown (their dense twins ignore the bits).

Probe: `left_mat mods2` (new mode). Hand-encoded words, the same fields as 9a (NEG_HI on A bit 8, on B bit 9; ABS C
bit 10; NEG C bit 63) on base words that equal `llvm-mc`'s encoding of the plain instruction with the same registers
(READ, checked at build) and that give the assembled instruction's results on the card (0 differ, every form).
Integer forms: 500,224 random elements per case against the plain instruction and the other sign settings. Float
sparse forms: 125,184 elements, half the problems with NaN, infinities, signed zeros and subnormals, against the
plain instruction on data whose C sign was changed in memory.

New hand-encoded words in this pass (`src/study/left_mat.hip`, lines 181 to 218): the sparse forms
`v_swmmac_f32_16x16x32_bf16` 0xcc514000, `_f16_16x16x32_f16` 0xcc524000, `_bf16_16x16x32_bf16` 0xcc534000,
`_f32_16x16x32_fp8_fp8` 0xcc574000 and `_bf8_bf8` 0xcc5a4000 (second word 0x1c521908; with ABS C the first word
gains 0x400, with NEG C the second becomes 0x9c521908), and the 4-bit words 0xcc454000 (K = 16) and 0xcc4a4000 (K = 32,
signed second word 0x7c021508) with their NEG_HI variants (0x...4200 on B, 0x...4100 on A), plus the unsigned 8-bit
NEG_HI-on-B word 0xcc444200. Validation: the base words come from `llvm-mc` for the plain instruction with the same
registers (READ, the encoding given in the source comment); on the card the raw word without any bit equals the
assembled instruction in every element (0 of 500,224 integer and 0 of 125,184 sparse elements differ, in
`ma1-ma3-mods2.log`: the "raw" and "nothing changed" rows), so where an assembled form exists, 0 words differ.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-ma1`):

| Form | Bit | Equal to (0 differ) | Controls (differ) |
|---|---|---|---|
| unsigned 8-bit | NEG_HI on B | the plain instruction: ignored | other sign settings 500,208 to 500,224 |
| 4-bit K = 16, signed and unsigned | NEG_HI on B | ignored | 485,484 to 500,224 |
| 4-bit K = 32, signed and unsigned | NEG_HI on B | ignored | 490,287 to 500,224 |
| 4-bit K = 32, unsigned | NEG_HI on A | ignored | 500,224 |
| sparse BF16 into FP32, FP16 into FP16, BF16 into BF16 | NEG C | C sign flipped | C cleared 21,196 to 42,856; no change 42,295 to 85,722 |
| same | ABS C | C sign cleared | C flipped; no change |
| sparse FP8, sparse BF8 | NEG C, ABS C | the plain instruction: ignored | the real C changes 43,174 to 93,310 |

Answer: NEG_HI is ignored on every integer form and both operands; the C bits act on every 16-bit sparse form exactly
as on the dense ones, and are ignored on the 8-bit float sparse forms, as on the dense FP8 and BF8 forms. MEASURED.

## LD-6a and LD-6b. Transposing loads: more than 24 in flight, and a partial lane mask

Question: (6a) do `global_load_tr_b128/b64` behave like plain loads past 24 in flight per wave? (6b) what does
`global_load_tr_b128` (and `_b64`) do with EXEC not all ones?

Expected: (6a) the load counter holds 63 (lane 2 item 5: a step at the 64th load); b128 results take 4 VGPRs, so only
56 fit. (6b) the guide calls the result undefined (READ); unknown: exploratory.

Probe: `lds_tr inflight2` (one wave, N loads of 16 x 16 tiles then one wait; plain and transposing; data in L2 and
from memory; 6 repetitions, first dropped; the listing holds 56 and 72 loads before one wait, no spill),
`lds_tr mask` (marker data, sentinel-filled destination, all lanes' addresses valid; 7 masks).

MEASURED, final run `runs/20261117T210256Z-leftc-b` (`ld6a-inflight2.log`; rebuilt from it; the development run `runs/dev-ld6` differs in several cells) (cycles per batch, median of 5 repetitions; in brackets the lead's rerun, `runs/lead/rerun/ld6a-inflight2.log`):

| Level | N | b128 | tr_b128 | b64 | tr_b64 |
|---|---|---|---|---|---|
| L2 | 1 | 294 (295) | 301 (305) | 291 (280) | 292 (291) |
| L2 | 24 | 396 (393) | 495 (488) | 341 (353) | 347 (344) |
| L2 | 48 | 497 (494) | 686 (689) | 403 (399) | 400 (404) |
| L2 | 56 | 524 (530) | 748 (753) | - | - |
| L2 | 63 | - | - | 424 (427) | 434 (427) |
| L2 | 64 | - | - | 535 (525) | 533 (533) |
| L2 | 72 | - | - | 579 (566) | 561 (559) |
| memory | 1 | 844 (853) | 867 (854) | 896 (897) | 882 (783) |
| memory | 24 | 948 (950) | 1,059 (1,034) | 892 (880) | 892 (882) |
| memory | 48 | 1,190 (1,203) | 1,339 (1,360) | 933 (993) | 954 (951) |
| memory | 56 | 1,340 (1,348) | 1,462 (1,432) | - | - |
| memory | 63 | - | - | 1,034 (1,132) | 970 (969) |
| memory | 64 | - | - | 1,493 (1,496) | 1,498 (1,486) |
| memory | 72 | - | - | 1,589 (1,564) | 1,560 (1,555) |

- The 64th load costs a whole memory or L2 trip for both b64 forms (from memory +459 plain, +527 transposing in the
  final run, rerun +364 and +517; from L2 +112 and +99, rerun +98 and +106), against 1.1 to 6.8 cycles per load from 48
  to 63 (final run): the same 63-load counter. MEASURED.
- Up to 56 the transposing b128 load keeps its higher issue cost: from L2 1.25 to 1.43 times the plain b128 batch
  (N = 24 to 56; 7.9 against 4.0 cycles per extra load between N = 24 and 56, as section 5 found at 24), from memory
  1.09 to 1.18 times (rerun 1.06 to 1.13). MEASURED.
- Partial EXEC (6b): for both transposing loads, under every mask including 0, every lane's destination receives the
  full-mask result, and on-lane elements whose source lane is switched off still hold that lane's data (64 of 64,
  48 of 48, 7 of 7). So the transposing loads ignore EXEC like the matrix instructions do: every lane's address is
  read and every lane written. MEASURED (one marker pattern per mask).

## LD-7a, LD-7b, LD-7c. The scalar cache

Questions: (7a) the scalar cache's line size (INFERRED 128 bytes in item 4 from capacity steps); (7b) what the level
between the scalar cache and memory is, and who shares it; (7c) do `s_buffer_load` forms cost what `s_load` costs?

Probes (`lds_smem`, new modes): `line`: load X (4 KiB aligned, fresh each trial), invalidate the scalar cache
(`s_dcache_inv`), load X again (from the level behind), then time X + d; 24 trials per d, in random order, from X and
from X + 64. `share` (CU-mode build): 64 single-wave groups, one per CU; each chases its own fresh region once,
then after a barrier chases the region of a partner (other CU of its WGP, other WGP of its shader array, other array of
its engine, other engine), then a fresh region of its own and its own region again; 4 KiB and 64 KiB regions, 3
launches per relation. `sbuf`: `s_buffer_load` through a descriptor (base, stride 0, size; the fields scalar loads use,
READ, guide B) against `s_load`, item 4's batch method.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development runs `runs/dev-ld7`, `runs/dev-ld7b`):

| d (bytes after X) | 4 to 48 | 64, 96, 128, 192 | 256, 512 |
|---|---|---|---|
| X + d, cycles (median, from X) | 37 | 184 to 185 | 450 to 511 |
| X + d, cycles (from X + 64) | 37 (d below 64) | 184 (d = 64, 96, 128) | 452 to 506 (d = 192 and more) |
| Calibration: X reloaded / X after invalidation | 35 / 182 | | |

(Final run; the development run `runs/dev-ld7` had the same steps at 199 to 202 cycles for the middle column.)

| Partner region (who touched it) | 4 KiB regions, cycles per load | 64 KiB regions |
|---|---|---|
| own region, right after touching it | 37.0 | 205.8 |
| other CU of the same WGP | 46.5 | 256.0 |
| other WGP, same shader array | 215.8 | 246.5 |
| other array, same engine | 215.1 | 252.2 |
| other engine | 215.7 | 253.1 |
| a fresh region (just written by the host-side builder) | 324 to 336 | 341 to 385 |

(Medians of 192 pairs per row, from `ld7b-share.log` of the final run, rebuilt from it. The lead's rerun, same folder
name `ld7b-share.log`, gives 46.4 / 215.7 / 215.5 / 215.8 for the 4 KiB column and 254.5 / 212.9 / 252.4 / 252.1 for the 64 KiB
column: the 4 KiB column repeats within 0.5 cycles, the 64 KiB column moves by up to 34 cycles in one row, so the 64 KiB
rows do not separate the partners. The fresh region at 64 KiB reads 341 to 385 in the final run, 336 to 385 in the rerun.)

| Width | s_load, N = 1 / 8 / 16 (cycles per batch) | s_buffer_load, N = 1 / 8 / 16 |
|---|---|---|
| b32 | 46.1 / 61.1 / 69.1 | 53.1 / 54.1 / 62.1 |
| b64 | 46.1 / 61.1 / 75.1 | 53.1 / 54.1 / 74.1 |
| b128 | 46.1 / 61.1 / 75.1 | 53.1 / 54.1 / 74.1 |
| b256 | 47.1 / 62.1 / - | 54.1 / 61.1 / - |

Answers:
- (7a) A scalar miss fills 64 bytes, aligned: the next 64 bytes are a miss (MEASURED). The driver's table says 64-byte
  lines (READ, item 4), so item 4's 128-byte inference from the capacity steps was wrong; those steps must come from
  the cache's organisation (INFERRED). Behind the scalar cache, the 256-byte block around X is present after X's miss
  (184 cycles for d = 64 to 192 from X, a memory trip at d = 256; from X + 64 the block ends at X + 256): that level
  fills 256 bytes, the L2 line size of memory-system item 2b (MEASURED; that it is L2 is INFERRED).
- (7b) A region one CU touched is found at 46 cycles by the other CU of the same WGP, and at about 216 cycles by any
  other CU, whatever its shader array or engine. So the scalar cache is shared by the two CUs of a WGP (the memory
  model's "scalar L0 per WGP", READ in item 4) and the level behind it is card-wide (INFERRED: L2). Supporting it: the
  own 4 KiB region was still a hit at the end only when the partner was in the same WGP (both groups' regions then
  fit 16 KiB together). The 10-cycle difference against the own region (46 against 37) is not explained here.
- (7c) `s_buffer_load` returns the right data; a lone load takes 7 cycles more than `s_load` (53 against 46), but a
  batch of 8 costs less (54 against 61, all widths up to b128), and the 32-bit form adds about 0.6 cycles per extra
  load against 1.5. MEASURED; why is not known.

Labels: MEASURED (line fill, sharing, costs); INFERRED (which named levels).

## EQ-1a. Other tensors and layers, and the per-tensor scale applied

Question: do real E4M3 tensors of other layers, and the stored per-tensor scale applied in FP32, change the rates of
errors-after-quantizing section 4?

Expected: layer 10's rates match the Gaussian stand-in within a few points (section 4, MEASURED); the scale multiply
adds one FP32 rounding of relative size 2^-24, comparable to the instruction's own error, so at most about twice the
unscaled FP16-differs rate (INFERRED in the plan).

Probe: `arith_realw` with `--scale` (new: the card's FP32 result multiplied by the stored FP32 scale in FP32, as a
kernel would; the exact sum multiplied by the scale exactly and rounded once) and `--real-only`; rows of 12 tensors
(layers 0, 10, 20, 27; `attn_q`, `ffn_gate`, `ffn_down`; 3,072 rows each, extracted read-only by
`scripts/gguf_rows.py`; scales 0.00058 to 0.0023, none a power of two). N as in section 1. Synthetic activations as
before. CPU only, on the bit model that equals the card.

MEASURED on the model, final run `runs/20261117T210256Z-leftc-b` (identical to the development run `runs/dev-eq` and to the lead's rerun `runs/lead/rerun`) (ranges over the 12 tensors; K = 4096 only `ffn_down`, the only tensor with
rows that long; pooled counts over the tensors):

| Path | K | Differs from exact: unscaled / scaled | FP16 differs: unscaled / scaled | Pooled FP16 count unscaled / scaled |
|---|---|---|---|---|
| FP8 x FP8 | 16 | 0.00-0.01 % / 0.01-0.03 % | 0 / 0-0.002 % | 0 / 22 of 2.4 M |
| FP8 x FP8 | 256 | 1.2-2.8 % / 2.2-4.8 % | 0-0.007 % / 0.002-0.012 % | 12 / 49 of 720 k |
| FP8 x FP8 | 4096 | 52.5-54.5 % / 57.2-58.4 % | 0.03-0.07 % / 0-0.07 % | 11 / 11 of 24 k |
| FP16 (weights widened) | 16 | 2.1-3.8 % / 4.8-8.1 % | 0.0005-0.007 % / 0.0035-0.0085 % | 56 / 155 of 2.4 M |
| FP16 | 256 | 61.1-66.9 % / 64.2-69.2 % | 0.028-0.053 % / 0.023-0.045 % | 306 / 254 of 720 k |
| FP16 | 4096 | 92.0-93.4 % / 92.0-93.1 % | 0.12-0.22 % / 0.15-0.22 % | 40 / 43 of 24 k |
| BF16 (weights widened) | 16 | 0.27-0.52 % / 0.70-1.36 % | 0-0.001 % / 0.001-0.0065 % | 8 / 81 of 2.4 M |
| BF16 | 256 | 28.6-39.9 % / 35.8-46.2 % | 0.008-0.032 % / 0.013-0.030 % | 152 / 136 of 720 k |
| BF16 | 4096 | 84.7-85.5 % / 84.8-85.4 % | 0.07-0.18 % / 0.07-0.15 % | 30 / 27 of 24 k |

E4M3 differs at most 0.0167 % unscaled (FP16 path, K = 4096, 1 of 6,000) and 0.033 % scaled (the same cell, 2 of
6,000), a few counts in every cell. Layer 0's `attn_q` stands out
(FP16 path at K = 16: 3.8 % differ from exact against 2.1 to 2.2 % for the others). The control of section 2 holds in
every cell.

Answer: other tensors and layers give the same picture as layer 10. Applying the scale adds its own rounding, and the
logs show it as follows (pooled FP16 changes, unscaled -> scaled): at K = 16, where the instruction's error is tiny,
FP16 path 56 -> 155 (2.8 times), BF16 path 8 -> 81 (10 times), FP8 path 0 -> 22 (no ratio: the unscaled count is
zero); at most 0.0117 % of outputs in any tensor. At K = 256: FP8 path 12 -> 49 of 720 k (4.1 times; "differs from
exact" 1.15-2.83 % -> 2.15-4.83 %), FP16 path 306 -> 254 (0.83), BF16 path 152 -> 136 (0.89); for the BF16 path the
share that differs from exact also rises (28.6-39.9 % -> 35.8-46.2 %). At K = 4096: 11 -> 11, 40 -> 43, 30 -> 27 of
24 k, no visible change. So the plan's "at most twice" fails at K = 16 for the FP16 and BF16 paths and, for the FP8
path, at K = 256 as well (about 4 times, small counts); it holds for the FP16 and BF16 paths from K = 256 on. The
counts are small (a Poisson spread of about +-14 % at 49 and +-29 % at 12), so the 4.1 times at K = 256 is
real in direction but loose in size. MEASURED on the model; that the extra changes come from the scale's FP32 rounding
is INFERRED (it is the only difference between the two runs).

## EQ-3. Outputs rounded to BF16

Question: how often do the card's FP32 result and the exact sum round to different BF16 values?

Expected (INFERRED in the plan): BF16's step is 8 times FP16's, so about one eighth of the FP16 rate (accepted range
1/16 to 1/4).

Probe: `arith_stats quantbf16` (new mode): the outputs, scale and control of section 1, target BF16 (RNE), with FP16 on
the same outputs.

MEASURED on the model, final run `runs/20261117T210256Z-leftc-b` (identical to the development run `runs/dev-eq`):

| Path | BF16 differs at K = 16 / 64 / 256 / 1024 / 4096 (count) | FP16 differs (count) |
|---|---|---|
| FP8 x FP8 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 1 / 3 / 1 |
| FP16 (weights widened) | 0 / 3 / 3 / 1 / 0 | 8 / 26 / 17 / 16 / 11 |
| BF16 (weights widened) | 0 / 2 / 1 / 1 / 0 | 1 / 8 / 16 / 2 / 4 |

Pooled over all cells: 11 BF16 changes against 114 FP16 changes, a ratio of 0.096 (about 1/10), inside the expected
range; every BF16 change is one step. Counts are small, so the ratio per cell is not meaningful. MEASURED on the model.

## EQ-2. Worst cases without subnormal inputs, for K = 1024, and with non-finite inputs

Question: how bad can it get (section 3's search) when no input is subnormal, for chains of 64 instructions, and do
NaN and infinity follow the IEEE rules?

Expected: without subnormals the error relative to the sum of magnitudes falls to about 2^-21 (section 3's reading),
so the large single-instruction step counts should vanish; K = 1024: unknown; non-finite: the class of the result
(NaN, +inf, -inf, finite) equals the IEEE class of the exact sum (a NaN input, inf x 0, or +inf with -inf give NaN).

Probe: `arith_qworst --nosub` (no subnormal input codes and no subnormal FP32 C anywhere in the search),
`arith_qworst --k1024` (chains of 64 instructions; a quarter of the K = 256 iterations), `arith_qworst --nonfinite`
(random cases with NaN and infinite inputs and C, 200,000 per path at K = 16, 20,000 at K = 256). Controls of section
3 (step counter against a brute-force table; every worst case recomputed from its printed hex).

MEASURED on the model, final run `runs/20261117T210256Z-leftc-b` (identical to the development run `runs/dev-eq`):

| Path | Worst FP16 steps, K = 16: with subnormals (section 3) / without | K = 256: with / without | K = 1024 (subnormals allowed) | Worst E4M3 steps, any of these |
|---|---|---|---|---|
| FP16 | 943 / 5 | 34 / 31 | 46 | 1 |
| FP16 form, E4M3 weights | 25 / 5 | 33 / 35 | 22 | 1 |
| BF16 | 7,110 / 5 | 21 / 31 | 19 | 1 |
| E4M3 x E4M3 | 5 / 5 | 18 / 14 | 26 | 1 |

Rates at K = 1024 (best generator found, 14,976 fresh samples): FP16 target 0.99 % (FP16), 0.24 % (E4M3 weights),
10.6 % (BF16), 0.007 % (E4M3 x E4M3); E4M3 target 0 except BF16 6.6 %. Every worst case with more than one step has
an exact sum 2^-13 to 2^-22 of the sum of magnitudes (cancellation).

Non-finite: 880,000 cases (expected classes: 405,955 NaN, 43,675 +inf, 43,720 -inf, the rest finite); the card's
class differs in 0.

Answer: the hundreds-to-thousands-of-steps cases of section 3 need subnormal inputs; without them one instruction
reaches 5 FP16 steps at worst, and chains 31 to 35 (those come from cancellation across the chain, not from
subnormals). K = 1024 (subnormals allowed, a quarter of the search iterations of K = 256) found worst cases of 46
FP16 steps for the FP16 path against 34 at K = 256, and 26 for E4M3 x E4M3 against 18; the FP16 form with E4M3
weights (22 against 33) and BF16 (19 against 21) found fewer. So K = 1024 is not "no worse": two paths are worse, two
are lower, and with a smaller search a lower figure may only mean a less thorough search. E4M3 steps stay at 1 at
K = 1024 in every path. The lead's rerun found the same eight figures. NaN and infinity follow IEEE classes exactly. MEASURED on the model (which equals the card for chains and special values, matrix-arithmetic 2, 3);
the searches are heuristic, so larger worst cases may exist.

## MT-7. Wave order: every `s_setprio` level, three waves, the round-robin bit

Question: is `s_setprio` a total order over its levels 0 to 3, and what happens with three waves on a SIMD?

Expected: matrix-timing item 9 (MEASURED): level 3 against none makes the raised wave behave as the older one; with
equal levels the older wave goes first (normal code object) or, with the round-robin bit, the matrix wave mostly goes
first. A total order predicts the higher level first in every pair.

Probe: `left_prio --levels` (16 new kernels, every pair of levels for the matrix wave and the FMA wave; two arrival
orders; 15 launches x 2 SIMDs = 30 pairs per row), for the normal code object and the one with the round-robin bit
set in every descriptor (21 of 21 descriptors checked at build). Then three running waves per SIMD (matrix and two
FMA waves) in two arrival orders, with and without `s_setprio 3`.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-mt7`):

| Levels (matrix, FMA) | Normal object: which wave ends first (30 of 30 in every row) | Round-robin object |
|---|---|---|
| matrix higher (6 pairs) | matrix (16.34 to 16.38 cycles per instruction) | matrix (16.33 to 16.45) |
| FMA higher (6 pairs) | FMA (matrix 31.7 to 32.2) | FMA (31.9 to 32.2) |
| equal (4 levels) | the older wave (matrix 16.4 when older, 31.7 when younger) | matrix (20.02 to 20.21, either order) |

| Three waves (ranks in arrival order) | Normal object: end order | Round-robin object |
|---|---|---|
| matrix, FMA, FMA; no s_setprio | M F F (30 of 30) | M F F |
| FMA, matrix, FMA; no s_setprio | F M F | M F F |
| FMA, matrix, FMA; matrix at 3 | M F F | M F F |
| matrix, FMA, FMA; both FMA waves at 3 | F M F | F F M |

Answer: the four levels are a strict order (24 of 24 unequal pairs and both arrival orders in each object); at equal
levels the age rule holds in the normal object and the matrix wave goes first with the round-robin bit, at every
level. With three waves the normal object keeps strict age order when nobody is raised. Raising both FMA waves to 3
lets the first FMA wave finish first, but then the matrix wave (older, lower level) finishes before the second FMA wave
in the normal object, while the round-robin object finishes both FMA waves first: so with equal raised waves the
normal object does not keep the lower wave waiting to the end. MEASURED (30 of 30 per row); the mechanism is not
established. The memory-bound partner was not repeated (item 9 already shows a load wave neither slows nor is slowed).

Label: MEASURED.

## MT-5b. What a hand-over between two waves' matrix chains costs, per form

Question: does the hand-over cost (item 9a: about 106 cycles for FP16, 13 for FP8) follow the instruction's cycles or
its type?

Expected: item 9a (MEASURED) for FP16 and FP8; if cycle-tied, BF16 about 106 and the 8-bit forms about 13.

Probe: `tim_mat handover` (new mode): W = 1, 2, 4, 8 waves per SIMD each running a 256-instruction chain, released
together; cost per hand-over = (span(W) - span(1) - (W - 1) x 256 x the chained cost) / (W - 1), item 9a's model
(chained costs 16 and 8 from matrix-timing items 2 and 3). 5 launches x 3 repetitions per row, medians.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-mt5b`) (cycles per hand-over at W = 2 / 4 / 8):

| Form | Chained cost | Hand-over |
|---|---|---|
| FP16 into FP32 (calibration) | 16 | 105 / 106 / 110 |
| BF16 into FP32 | 16 | 105 / 106 / 109 |
| FP16 into FP16 | 16 | 106 / 107 / 109 |
| sparse FP16 into FP32 (K = 32) | 16 | 76 / 77 / 79 |
| FP8 (calibration) | 8 | 13 / 12 / 13 |
| BF8 | 8 | 13 / 12 / 13 |
| 8-bit integer | 8 | 13 / 12 / 13 |

Answer: every dense form with a 16-cycle instruction costs the same 105 to 110 cycles per hand-over and every 8-cycle
form 12 to 13, whatever the data type; sparse FP16, also 16 cycles, costs 76 to 79. So the cost goes with the
instruction's issue timing, not with the type, and the sparse form differs. MEASURED; why 16-cycle instructions cost
eight times more than 8-cycle ones is still not explained. Chain lengths other than 256 were not run.

Label: MEASURED.

## MT-6. Why a new B costs a cycle only with a vector instruction between

Question: what exactly makes a changed B operand cost one cycle when a vector instruction sits between two matrix
instructions (item 5d)?

Expected: unknown, exploratory. Known (item 5d): 16 / 18 / 19 cycles per block for nothing / `v_mov` / `v_mov` with B
rotating.

Probe: `left_rot`, five new variants (one wave, 8 rotating accumulators, cycles per block from the difference of two
unroll lengths, 9 launches, medians; the listing holds every one of the 3,552 matrix instructions as written).

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-mt6b`) (cycles per block; known answers 16.00, 18.00, 19.00 reproduced):

| Variant | Cycles per block |
|---|---|
| `v_mov` between, B alternating between v[96:99] and v[98:101] (shifted by 2, overlapping) | 18.5 |
| `v_mov` only before every second matrix instruction, B rotating over 4 | 17.5 |
| `v_mov` in VOP3 form between, B rotating over 4 | 19.0 |
| `v_nop` in VOP3 form between, B rotating over 4 / A and B fixed | 19.0 / 18.0 |

Answer: the extra cycle belongs to the matrix instruction that directly follows a vector instruction and has a B
different from its predecessor's: with the `v_mov` before every second instruction the cost is (19 + 16) / 2 = 17.5,
so a B change right after a matrix instruction stays free. The encoding of the vector instruction does not matter
(VOP3 forms cost the same). A B that overlaps the previous one by half costs half the cycle on average (18.5), which
fits "only every second change counts" but is not explained. Every A and B set holds the same values, so the cost is
the register change, not the data. MEASURED; the mechanism remains open.

Label: MEASURED (exploratory).

## VK-5. The lost-product effect with odd accumulator values (Vulkan)

Question: does the rounding rule behind vulkan-view item 6's lost products (the running sum rounded to FP32, nearest
even, after each group of 4 products for 16-bit inputs and of 8 for 8-bit) also predict results with odd C?

Expected: that model (matrix-arithmetic, lane 5; vulkan-view item 6) predicts every result bit: at C = 2^22 + 0.5 (an
odd last bit) a 16-bit step of 0.25 is a tie that rounds up to 2^22 + 1, and from there each further 0.25 ties back to
the even value, so 2^22 + 1 (not the exact 2^22 + 1.5).

Probe: `vkstudy check2` (new mode): the `check_big` shaders (every A and B element 0.25, C uniform per set, 256 result
words per set) with 11 values of C, the three old even ones as calibration and 8 new ones; every word against the step
model computed on the host. FP16, BF16, FP8.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (development run `runs/dev-vk5`):

| C | FP16 and BF16 result (model) | FP8 result (model) | Exact C + 1 |
|---|---|---|---|
| 2^21 (old) | 2^21 + 1 | 2^21 + 1 | 2^21 + 1 |
| 2^22 (old) | 2^22 (lost) | 2^22 + 1 | 2^22 + 1 |
| 2^23 (old) | 2^23 (lost) | 2^23 (lost) | 2^23 + 1 |
| 2^21 + 0.25 | 2^21 + 1.25 | 2^21 + 1.25 | the same |
| 2^22 + 0.5, + 1.5, + 2.5 | 2^22 + 1, + 2, + 3 (half lost) | 2^22 + 1.5, + 2.5, + 3.5 | 2^22 + 1.5, ... |
| 2^23 + 1, 2^23 + 3 | unchanged (lost) | 2^23 + 2, 2^23 + 4 | 2^23 + 2, + 4 |
| 2^24 + 2, 2^24 + 6 | unchanged | unchanged | + 1 |

Answer: the model reproduces all 8,448 result words (0 differ); with round toward zero in the model 2,048 differ (the
deliberate break). An odd C makes the first tie round up, so FP16 and BF16 keep half of the 1.0 at C = 2^22 + 0.5 and
FP8 keeps all of it at 2^23 + 1 (the first 0.5 step is a tie that rounds up to the even 2^23 + 2, the second a tie
that rounds back down to it, which is the exact result). MEASURED.

## VK-3. A software-pipelined cooperative-matrix load loop (Vulkan)

Question: does a load loop that fetches tile t + 1 before the multiply-add of tile t beat ACO's naive loop from the
L2 (4 MiB) and memory-side-cache (32 MiB) windows?

Expected: unknown, exploratory; the naive loop waits for both loads every tile (vulkan-view item 5, listing).

Probe: `load.comp -DPIPE` (new variant: the next tile's `coopMatLoad`s before this tile's `coopMatMulAdd`, the
fragments carried in variables), `vkstudy load ... --variant pipe`; one timed dispatch per job (VKS_TARGET_S 0.075,
1 repetition; the timed dispatches lasted 0.031 to 0.042 s), W = 4, wave64 (wave32 for FP16 from 4 MiB); exact
checksum as the existing load mode. The load mode now prints the counter clock, so rates are compared per GHz (these
short bursts run at 2.8 to 3.1 GHz, in the clock ramp).

MEASURED, run `runs/20261117T210336Z-leftc-b-card`. Every cell ran in two passes (the folder's `summary.md` keeps the
operand rates of both; the first pass's logs were replaced by the second, so only its GB/s of operands survive, no
clocks). The table gives the second pass (the logs in the folder) as TFLOPS per GHz, and operand GB/s of both passes:

| Case | Naive loop: instructions, TFLOPS per GHz (GB/s first / second pass) | "Pipelined" loop: instructions, TFLOPS per GHz (GB/s first / second pass) | Change in operand GB/s, first / second pass (per GHz, second pass) |
|---|---|---|---|
| FP16, 4 MiB, wave64 | 39, 9.12 (3,077 / 3,310) | 22, 10.24 (3,412 / 3,720) | +11 % / +12 % (+12 %) |
| FP16, 4 MiB, wave32 | 35, 9.88 (3,598 / 3,532) | 22, 10.63 (3,788 / 3,838) | +5 % / +9 % (+8 %) |
| FP16, 32 MiB, wave64 | 39, 6.03 (2,003 / 2,194) | 22, 6.13 (1,744 / 2,215) | -13 % / +1 % (+2 %) |
| FP8, 4 MiB, wave64 | 38, 10.34 (1,791 / 1,812) | 17, 12.13 (2,125 / 2,172) | +19 % / +20 % (+17 %) |
| FP8, 32 MiB, wave64 | 38, 6.92 (1,317 / 1,336) | 17, 7.62 (1,455 / 1,481) | +10 % / +11 % (+10 %) |

(The lead's rerun, `runs/lead/rerun`, gives per GHz: 9.12 / 10.15, 9.89 / 10.53, 6.02 / 6.26, 10.34 / 12.13, 6.93 / 7.63
for the five cases, so +11, +6, +4, +17 and +10 %.)

Listing of the "pipelined" loop (FP16, wave64): `buffer_load_b64` and `global_load_tr_b128` of the next tile, then
`s_wait_loadcnt 0x0`, then the `v_wmma`, then four `v_mov` that move the new fragments into the operand registers. So
ACO did not pipeline: it waits for the next tile's loads before this tile's multiply-add, exactly as in the naive
loop. What changed is the loop body: the per-lane address work (`v_mbcnt`, `v_bfrev`, shifts, `v_mul_lo_u32`) left
the loop, 22 instead of 39 instructions.

Answer: a GLSL-level software pipeline does not reach the machine code (ACO places a full wait before the
multiply-add). The restructured loop is faster in four of the five cases in both passes and the rerun: FP16 from the
4 MiB window by 5 to 12 % and FP8 by 10 to 20 % (operand GB/s, both passes), the difference seen in the listing is the
shorter loop body (no test of cause). For the FP16 32 MiB window the first pass gave -13 %, the second +1 % and the rerun
+4 %: no effect is established there. The same naive configuration (FP16, 4 MiB, wave64) was also measured under VK-4
and gave 3,342 against 3,077 GB/s in the first pass (8.6 % apart) and 3,341 against 3,310 in the second (1 %): that is
the size of the pass-to-pass spread of a single burst, so differences of a few per cent are not separated. MEASURED
(single bursts per pass, two passes and a rerun). The break (the next tile taken one tile too far) gives a wrong
checksum: FAIL as intended.

## VK-4. Why wave64 is 8 % slower in the load-per-tile loop (Vulkan)

Question: is vulkan-view item 5's wave64 loss from 4 MiB the extra per-tile vector instructions (16 against 13)?

Expected: INFERRED in vulkan-view item 5: yes; then a loop with fewer per-tile vector instructions should close the
gap, and added per-tile vector work should cost wave64 more than wave32.

Probe: the plan's "lean" shader could not be written in GLSL (see "Not measured"), but VK-3's restructured loop is
one (22 instructions in both wave sizes); and a dose test, `load.comp -DEXTRA=8` (8 dependent integer multiply-adds
per tile on a value kept alive: 10 `v_mul_lo_u32` more per tile in the listing). FP16, 4 MiB, W = 4.

MEASURED, run `runs/20261117T210336Z-leftc-b-card`, two passes as in VK-3 (second pass as TFLOPS per GHz; the naive
loops were measured in VK-3 and here, with the same command; GB/s of operands of both passes in brackets, first pass
first):

| Loop | Wave32 | Wave64 | Wave32 / wave64 (per GHz; first / second pass in GB/s) |
|---|---|---|---|
| naive (35 and 39 instructions) | 9.88, 9.89 (3,598 and 3,569 / 3,532 and 3,608) | 9.12, 9.12 (3,077 and 3,342 / 3,310 and 3,341) | 1.084 (1.07 / 1.08) |
| restructured (22) | 10.63 (3,788 / 3,838) | 10.24 (3,412 / 3,720) | 1.037 (1.11 / 1.03) |
| naive + 8 multiply-adds per tile (59 and 61) | 9.76, -1.3 % (3,508 / 3,537; -1.7 % / -2.0 %) | 7.42, -19 % (1,853 / 2,746; -45 % / -18 %) | 1.315 |

(Changes against the naive loop of VK-4's own command; the rerun gives 9.76 and 7.41 TFLOPS per GHz, so the same -1.3 and -19 %.)

Answer: two things hold in both passes. The added per-tile vector work costs wave64 much more than wave32 (-45 % and
-18 % against -1.7 % and -2.0 % in operand GB/s; -19 % against -1.3 % per GHz in the second pass and the rerun), which
fits "wave64 runs every vector instruction in two halves, wave32 hides it under the loads" (INFERRED). The naive
loops show wave32 faster than wave64 by 7 to 8 % in both passes (item 5: 1.083). What does not hold in both passes
is that the gap closes with less per-tile vector work: 1.037 in the second pass and the rerun (10.63 against 10.24
and 10.53 against 10.15 per GHz), but 1.11 in the first pass (3,788 against 3,412 GB/s). The first pass's -45 % for
the extra work is also far from the second pass's -18 %; the first pass's clocks were not kept, so the difference is
unexplained. So the plan's reading, that the per-tile vector instructions explain wave64's loss, is supported for the
dose test and not settled for the shorter loop. MEASURED (rates, single bursts); INFERRED (the mechanism).

## VK-7a. Mixed FP8/BF8 and saturating 8-bit forms (Vulkan)

Question: do the mixed E4M3 x E5M2 forms and the saturating signed 8-bit form run at the FP8 rate?

Expected: 8 cycles per instruction for every 8-bit form (matrix-timing item 2), so 128 x clock / 8.

Probe: `mat.comp` with new defines (`T_FP8BF8`, `T_BF8FP8`, `T_BF8`; B read from a second binding of the same
buffer so A and B can differ in type; `T_I8SAT` with `gl_MatrixOperandsSaturatingAccumulation`), `vkstudy mat`, W = 8,
wave64, one 0.04 s timed dispatch per job, two jobs per form; exact totals checked. Verdict: ratio to 128 x the
counter clock / 8 within 3 % (the plan's "within 3 % of 421 TFLOPS" assumed the steady 3.22 GHz clock; these bursts
ran at 3.12 to 3.21 GHz, and the first version of the verdict failed one burst at 0.965 of 420 (405.3 TFLOPS, bf8 x fp8, second burst of the first
pass) for that reason only; the criterion was changed after that failing burst). The first pass's ten bursts gave 405.3
to 416.4 TFLOPS against 420; the table below is the second pass, whose ten bursts (two per form) would also pass the
old criterion: all within +-3 % of 420, the lowest 411.0 TFLOPS (0.979).

MEASURED, run `runs/20261117T210336Z-leftc-b-card`:

| Form | Ratio to 128 x clock / 8 (two bursts) | TFLOPS (clock) | Totals |
|---|---|---|---|
| FP8 x FP8 (control) | 0.997, 0.996 (from the REP lines) | 419.5 (3211 MHz), 415.3 (3180) | exact |
| FP8 x BF8 | 0.997, 0.997 | 416.7 (3190 MHz), 414.8 (3175) | exact |
| BF8 x FP8 | 0.989, 0.998 | 412.0 (3177), 417.3 (3189) | exact |
| BF8 x BF8 | 0.989, 0.993 | 411.4 (3175), 411.0 (3159) | exact |
| signed 8-bit, saturating | 0.998, 0.997 | 415.7 (3179), 416.3 (3185) | exact |

Answer: every mixed and saturating form runs at the FP8 rate per clock (0.989 to 0.998 of 128 x clock / 8, the FP8
control 0.996 to 0.997). MEASURED.

## VK-2. Matrix repetitions of about 0.1 s against the clock (Vulkan)

Question: do longer matrix repetitions agree with the HIP clock figures (counter clock equal to `pp_dpm_sclk`)?

Probe: `vkstudy mat f16` and `bf16`, VKS_TARGET_S 0.15, one repetition per job (the timed dispatch lasted 0.08 s:
the calibration dispatch overestimates by about 2x, vulkan-view "Not measured"). 50 ms sysfs sampler.

MEASURED, run `runs/20261117T210336Z-leftc-b-card`: FP16 210.0 TFLOPS at a counter clock of 3220 MHz (ratio to theory 0.995), BF16 210.6
at 3220 MHz (0.998); the sampler caught 3 samples per repetition with `pp_dpm_sclk` 1759 / 2599 / 3220 MHz (min /
median / max): the highest sample equals the counter clock within 0.03 %, the others fall in the ramp before the
timed dispatch.

Answer: the counter clock equals the driver's clock reading once the clock has ramped; with 0.08 s dispatches the
50 ms sampler is too slow to show more (the plan's 4 samples per repetition were not reached). MEASURED; low value,
as the plan expected.

## SC-4. A shifted-trace control for the coincidence of stalls between two processes

Question: do the two processes' stalls (scheduling item 7: 98.8 % within 1 us) coincide more than by chance?

Expected: chance coincidence about 0.3 % (item 7, INFERRED from the event rate).

Probe: `scripts/sched_pair.py --shift` on the kept files of `runs/20261117T165519Z-sched` (`pair-small-small-A.csv`,
`-B.csv`, read only): B's events moved by +-5, 10, 20, 50 and 100 ms and A's events counted again.

MEASURED (offline): unshifted 0.988 (1,246 events of A, 1,282 of B); shifted 0.000 to 0.004 (largest at +-100 ms);
chance for independent events 0.0026.

Answer: the coincidence exists only for the aligned traces, so it is not an artefact of the event rate. MEASURED.

## SC-3. Why the hotspot throttle bit 36 is set at 57 to 92 C

Question: what goes with throttle bit 36 (`TEMP_HOTSPOT`) being set at hotspot readings as low as 57 C?

Expected: unknown, exploratory; candidates: rate of rise, another internal sensor.

Probe: new `scripts/leftc_throttle.py` (CPU) over the kept 50 ms traces of `runs/20261117T164432Z-sched` (FP16, FP8,
stream, light, idle; 4,305 samples): per sample the bit, the hotspot, its rise over the last 200 ms, reported power,
sclk and time since the load started; AUC of each variable for the bit. Control: PPT0 against reported power (known
to be set at the 317 W cap) and the same with shuffled labels.

MEASURED (offline): the bit is set in 23 samples, all in the first 0.8 s of the FP16 (8) and FP8 (15) loads, none in
the stream, light and idle traces.

| Variable | AUC for bit 36 | Share of samples with the bit |
|---|---|---|
| hotspot rise over 200 ms | 0.946 | rise >= 50 C/s: 10 of 10; 20 to 50: 7 of 81; below 20: 6 of 3,692 |
| time since the load started | 0.006 (early) | first 0.1 s: 2 of 10; 0.1-0.5 s: 15 of 40; 0.5-1 s: 6 of 50; later: 0 of 4,104 |
| reported power | 0.905 | |
| hotspot | 0.833 | 70-90 C: 14 of 31; 90 C and above: 8 of 770 |
| sclk | 0.714 | |

Control: PPT0 against power AUC 0.999, shuffled labels 0.507.

Where the 23 samples sit (`sc3-throttle.log`): the FP16 trace's 8 at t = 0.05 to 0.40 s, hotspot 57 to 82 C, rise 25 to
135 C/s, power 179 to 265 W; the FP8 trace's 15 at t = 0.09 to 0.79 s, hotspot 72 to 92 C, rise 0 to 176 C/s, power 309
to 326 W. Of the 23 set samples, 6 have a rise below 20 C/s. The AUC for "time since the load started" (0.006, which
is 0.994 in the other direction) separates the bit better than the rise does (0.946). Everything comes from two load
onsets (the FP16 and the FP8 trace); the stream, light and idle traces have no onset of this kind and no bit.

Answer: the bit appears in the first 0.8 s of the two heavy loads and nowhere else in these traces. A fast rise
separates it well (AUC 0.946; 10 of 10 samples with a rise of 50 C/s or more have it), a sustained 90 to 93 C rarely
shows it (8 of 770), but time since the load started separates it still better, and with only two onsets the three
cannot be told apart: "early in a load", "a fast rise" and a temperature or power threshold all fit (the FP8 trace's
bits sit at 72 to 92 C and 309 to 326 W with rises from 0 to 176 C/s). SC-5 below adds four more onsets but they
do not separate the candidates either. MEASURED (association only). INFERRED, weakly: the firmware's hotspot limiter
acts on something that is high shortly after a load starts (a predicted or rate-based temperature is one reading;
a time or a threshold are others).

Label: MEASURED (association); the cause is not established.

## SC-12. The Vulkan compute queue's slow path with the card kept busy

Question: does the compute queue's slow path (scheduling item 6: 30 to 42 % of submissions take about 100 us instead
of about 30 when the queue runs alone) go away when the card is never idle?

Expected: INFERRED in item 6: the slow path follows an idle state of the GPU or the queue.

Probe: `vksched --keepalive` (new option): the compute queue's one-group dispatch, submitted and fence-waited 250
times per block, 8 blocks alternating without and with a second thread that resubmits the graphics queue's dispatch
back to back (fence-waited). The keep-alive is a stream of tiny dispatches (about one every 40 us of host time, 0.3 us
of GPU work each), not a continuously busy card: no long-running shader was written.

MEASURED, final run `runs/20261117T210256Z-leftc-b` (1,000 samples each; "slow" = above twice the median without keep-alive, 31.1 us; the
development run `runs/dev-sc12` gave 0.272 and 0.174):

| | Wall median (p10 / p90) | Slow share | GPU time median |
|---|---|---|---|
| compute queue alone | 31.1 us (19.9 / 102.7) | 0.253 | 0.48 us |
| with the graphics-queue keep-alive | 18.8 us (17.3 / 64.3) | 0.115 | 0.40 us |

Answer: keeping the other queue submitting lowers the median by 40 % (31.1 to 18.8 us) and the slow share
from 0.253 to 0.115, counted above 62.2 us (twice the alone run's median, as in the table); counted against each run's own
median (2 x 18.8 us for the "on" run) the "on" run's slow share is 0.198, so on that measure the fall is from 0.253 to 0.198.
The slow path is not removed. The spread between runs is large: the lead's rerun (`runs/lead/rerun/sc12-keepalive.log`)
gave medians 30.6 and 19.8 us and slow shares 0.303 (alone) and 0.207 (with keep-alive, counted above 2 x 30.6 us; 0.299
against its own median); the development run gave 0.272 and 0.174. So the median fall (35 to 40 %) repeats; the share above the alone run's threshold falls in all three runs
(0.253 to 0.115, 0.303 to 0.207, 0.272 to 0.174; "a tenth" is the lowest of three); against each run's own median it
falls from 0.253 to 0.198 in the final run and hardly at all in the rerun (0.303 to 0.299). MEASURED for this weak
keep-alive; whether a continuously busy card removes the slow path is not settled.

Label: MEASURED.

## MT-5c. What costs the whole card 1 to 2 % of FP16 with more waves?

Question: is the FP16 loss with 16 waves per SIMD (matrix-timing item 11: ratio 0.977) a fixed cost per launch or a
cost per instruction?

Expected: unknown, exploratory. A fixed per-launch cost shrinks with the launch length (loss(50 ms) / loss(200 ms) of
2.5 or more); a per-instruction cost stays (ratio near 1).

Probe: `mat_card burst f16` (new mode): one launch per locked job of 50, 100 or 200 ms at W = 1 and W = 16 (64 groups
with 40 KiB of LDS, every SIMD with W waves, checked per burst), three bursts per configuration, each started below
55 C; rate from HIP events and from the waves' REALTIME span, against 128 x the waves' counter clock / 16.

MEASURED, run `runs/20261117T210336Z-leftc-b-card` (medians of 3 bursts; ratio from HIP events, the REALTIME span gives the same within 0.1 %):

| Launch | W = 1 ratio (clock) | W = 16 ratio (clock) | Loss |
|---|---|---|---|
| 50 ms | 0.9993 (3095 MHz) | 1.0012 (3015 MHz) | -0.2 % |
| 100 ms | 0.9984 (3155) | 0.9886 (3114) | 1.0 % |
| 200 ms | 0.9991 (3188) | 0.9823 (3166) | 1.7 % |

Answer: the loss is absent in 50 ms launches (-0.2 %) and is 1.0 % at 100 ms and 1.7 % at 200 ms (item 11 saw 2.3 %
at 0.17 s after a warm-up launch). The ratio of the two long launches is not the 2.5 or more of a fixed per-launch
cost (loss(50 ms) is negative). But the W = 16 clock also rises with the launch (waves' counter clock 3015, 3114,
3166 MHz at 50, 100, 200 ms), so the loss is associated with longer launches and with a higher clock, and these
bursts do not tell which of the two matters. At W = 1 the clock rises the same way (3095, 3155, 3188 MHz) with no
loss (0.998 to 0.999), so the clock alone does not produce it; whether the cause is the length or an effect of 16
waves on the clock estimate (MT-3: each wave's counter covers only its own part of the burst, up to 1.5 % uncertainty
in W = 16 ratios) is open. Because that uncertainty is as large as the 1.0 to 1.7 % itself, the loss is a
measured difference between two ratios, not a firm wall-clock figure. Spread between bursts (min to max of 3, ratio
from HIP events): W = 1: 0.0003 at 50 ms, 0.0027 at 100 ms (0.9969 to 0.9996), 0.0003 at 200 ms; W = 16: 0.0050 at
50 ms (0.9967 to 1.0017), 0.0020 at 100 ms (0.9871 to 0.9891), 0.0008 at 200 ms (0.9818 to 0.9826). The REALTIME-span
ratio agrees with the event ratio within 0.15 % (W = 1 at 100 ms: 0.9999 against 0.9984). The calibration passes
narrowly (0.9823 against 0.977 +-0.006: the edge of the band is 0.9830). The lead's rerun (`runs/lead/rerun`,
`mt5c-analysis.log`) gives W = 16: 1.0007 / 0.9862 / 0.9826 at 50 / 100 / 200 ms, and W = 1: 0.9991 / 0.9996 / 0.9998
(one W = 1 burst at 200 ms read 0.9775). MEASURED; cause open.

Label: MEASURED (loss against launch length and clock, together); the cause is not established.

## MT-3. Whole-card rates of the remaining forms

Question: do sparse BF16, sparse BF8, dense BF8 and the 16-bit-accumulator forms reach 128 SIMDs x clock / cycles?

Expected: cycles from matrix-timing items 2 and 3 (16 for 16-bit inputs, 8 for 8-bit); ratio 1.000 +-0.01 at W = 1,
as for the nine forms of items 11 and 11a.

Probe: `mat_card burst <form>`: 0.1 s bursts, W = 1 twice and W = 16 once per form; 16 instructions of the form per
loop iteration over 4 accumulators (listing: a run of exactly 16 per form).

MEASURED, run `runs/20261117T210336Z-leftc-b-card`:

| Form | Cycles | W = 1 ratio | W = 1 rate (dense-equivalent) | W = 16 ratio |
|---|---|---|---|---|
| sparse BF16 into FP32 | 16 | 0.9988 | 413.3 TFLOPS | 0.9941 |
| sparse BF8 into FP32 | 8 | 0.9995 | 826.1 TFLOPS | 1.0013 |
| BF8 into FP32 | 8 | 0.9996 | 413.1 TFLOPS | 1.0151 |
| FP16 into FP16 | 16 | 0.9997 | 206.9 TFLOPS | 0.9888 |
| BF16 into BF16 | 16 | 0.9990 | 206.6 TFLOPS | 0.9875 |
| sparse FP16 into FP16 | 16 | 0.9996 | 413.4 TFLOPS | 0.9967 |

(The rates are at the clock of these short bursts, 3.15 to 3.16 GHz; the ratio is the comparable figure.)

Answer: every form reaches 128 x clock / cycles within 0.12 % at one wave per SIMD. At 16 waves the 16-bit dense
forms lose 1.1 to 1.3 % as FP16 does (MT-5c); the 8-bit forms show ratios slightly above 1 (up to 1.015), which
would fit a clock-estimate effect of 16 waves sharing a SIMD in turn (each wave's counter clock covers only its own
part of the burst; INFERRED, not tested here), so the W = 16 ratios are read as carrying about 1.5 % of uncertainty.
The W = 16 ratios of the six forms range from 0.9875 to 1.0151 in one burst each. MEASURED.

## LD-9. The 32 MiB window of the transposing-load bandwidth

Question: why did the 32 MiB window in lds section 5 read at 5.3 to 5.4 TB/s, above lane 2's memory-side-cache rate
of about 2.0 TB/s?

Expected (plan): if neighbouring waves share lines in L2, a walk where no two waves read the same tile in a pass must
fall to at most 2.4 TB/s. Reading `k_bw` showed more: at a 32 MiB window each wave alternates between the same two
4-tile groups (8,192 waves x 2 KiB per step cover the window in two steps), so each wave rereads its own 4 KiB about
30 times in a launch. So the "unique" walk was defined as: wave w reads the run of groups w x steps + s, so no wave
ever rereads its own data.

Probe: `lds_tr bw2` (new mode), one burst per job sized for 0.1 s from an expected rate (the bursts actually ran 25 to
55 ms at 4 to 32 MiB, 223 ms (shared) and 54 ms (unique) at 64 MiB, 99 ms at 512 MiB), windows 4, 16, 32, 64 and 512
MiB, the `bw` walk ("shared") against the unique walk, transposing and plain b128. In the unique walk no wave rereads
its own data, but it is not free of sharing between waves: each of the 8,192 waves sweeps 14,901 of the window's
16,384 groups per launch, so every group is read by thousands of waves over the launch (never by two at the same
step).

MEASURED, run `runs/20261117T210336Z-leftc-b-card` (GB/s, one burst per cell; two at 32 and 512 MiB for the transposing load; burst length in ms in brackets for the 32 MiB row):

| Window | tr_b128 bw walk | tr_b128 unique walk | b128 bw walk | b128 unique walk |
|---|---|---|---|---|
| 4 MiB | 9,554 | 10,027 | | |
| 16 MiB | 9,547 | 9,262 | | |
| 32 MiB | 9,532 and 9,555 (55.6, 55.5 ms) | 7,248 and 7,395 (34.5, 33.8 ms) | 9,574 (55.4 ms) | 7,891 (31.7 ms) |
| 64 MiB | 2,378 | 4,666 | | |
| 512 MiB (calibration: DRAM) | 634 and 634 | | | |

Answer: the calibration failed for the 32 MiB window: section 5's 5.3 TB/s was not reproduced; the same walk read
9.5 TB/s (9,532 and 9,555 GB/s) in both bursts, the same as the 4 and 16 MiB windows, while the DRAM calibration
(629 GB/s) reproduced (634). The lead's rerun (`runs/lead/rerun`, same sources) reproduced the 9.5 TB/s (9,525 and
9,532 GB/s; unique walk 7,404 and 7,435; plain b128 9,582 shared and 8,028 unique; DRAM 638). The earlier 5.3 TB/s of
lds-and-special-loads is therefore unexplained: its method differs (7 timed launches after 2 warm-up launches, a wide
spread of 5.2 to 7.2 TB/s) from this one (one 55 ms launch after a one-step warm-up), but no test here shows that
this accounts for the difference.

What the within-run comparison shows: the unique walk at 32 MiB (7.2 to 7.9 TB/s) is far above the 2.4 TB/s bound the
sharing explanation predicted, so that prediction is rejected in this run, and no more than that. "Neighbouring waves
share lines" is not supported here; it is not ruled out, because the calibration failed (the walk the explanation was
about did not reproduce) and because the unique walk still shares data between waves over a launch. Two
confounds limit the comparison at 32 MiB: the unique bursts ran 32 to 34 ms against 55 ms for the shared walk, so a
larger part of each unique burst is clock ramp (the 7.3 against 9.5 TB/s gap is not separated from that), and the
rates are single bursts (two for the transposing load). At 64 MiB the unique walk (4.7 TB/s, 54 ms) is twice as fast as
the bw walk (2.4 TB/s, 223 ms): per-wave sequential runs help there, MEASURED, with the same length caveat. The
statement that the windows up to 32 MiB are "served by L2 and the memory-side cache together" is not supported: rates
of 7 to 10 TB/s are far above the 2 TB/s memory-side-cache rate of lane 2, and what serves them is not determined
here. The transposing and plain loads read at the same rate in every pattern tried (9.5 against 9.6 TB/s shared, 7.3
against 7.9 unique at 32 MiB: within about 8 %). Verdict `ld9_calib` FAIL (see "Checks").

Label: MEASURED (rates); the explanation of section 5's 5.3 TB/s is open.

## SC-5. Throttle bits during short FMA, FP8 and FP16 loads

Question: which throttle bits does the FMA load show in its first moments (the long FMA trace predates the sampler)?

Expected: unknown, exploratory. Known: FP8 and FP16 traces showed bit 36 (`TEMP_HOTSPOT`) in the first 2 s and PPT0
for FP8 from 0.34 s; SC-3 above found bit 36 only early in the two heavy loads.

Probe: `sched_trace <load> --burst-ms L --sample-ms 5` (new): one load of 0.1 s (FMA, FP8) or 0.2 s (FP16), started
below 55 C and stopped at 90 C; `gpu_metrics` throttle bits, hotspot and reported power every 5 ms from 0.2 s before
to 0.5 s after. Idle control. The load is a series of launches (11 to 23) after one calibration launch that ran in the
0.2 s before the burst: the first sample (t = -0.2 s) shows 71 and 153 W in the two FMA bursts, 55 W for FP8 and 109 W
for FP16, so the burst is not a cold start from idle, and the idle control had no calibration launch.

MEASURED, run `runs/20261117T210336Z-leftc-b-card`:

| Load | Bits seen | Bit 36 first at | Samples with a bit (from load start) | Hotspot start / max | Reported power max |
|---|---|---|---|---|---|
| idle (control) | none | - | 0 of 20 in the window | 51 / 51 C | 50 W |
| FMA, 0.1 s (two bursts) | `TEMP_HOTSPOT` only | 0.050 and 0.051 s | 20 and 20 | 52 / 77 C | 314 to 316 W |
| FP8, 0.106 s | `TEMP_HOTSPOT` only | 0.051 s | 22 | 51 / 79 C | 316 W |
| FP16, 0.204 s | `TEMP_HOTSPOT` only | 0.061 s | 34 | 51 / 76 C | 259 W |

(Each row is from the matching `sc5-*.log`; the lead's rerun gives the same bit at 0.050 to 0.061 s, hotspot starts of
50 to 52 C and maxima of 76 to 79 C.)

In the four loaded bursts the bit first appears when the hotspot reads 66 to 67 C (FMA 67 and 66, FP8 67, FP16 66),
0.050 to 0.061 s after the load starts, with the hotspot at 49 to 50 C 0.2 s earlier (an average rise of 85 to 90 C/s over
those 0.2 s, about 300 C/s during the 50 ms of load). It stays set to the end of the load and the last set sample is 45 ms (FMA, both
bursts), 49 ms (FP8) and 21 ms (FP16) after the load ended, when the hotspot has fallen to 54 C (61 C for FP16). PPT0 does not appear within 0.1 s although the reported power reaches the 317 W cap figure.

Answer: in the first 0.1 s of FMA, FP8 and FP16 loads the only throttle bit is 36, set from about 50 ms on, in every
burst, at 66 to 67 C. All four bursts set it at the same temperature, so a threshold near 66 C fits these bursts as
well as a rate of rise; together with SC-3 the data do not separate "time since the load started", "rate of rise"
and "a threshold near 66 C". The idle control (no calibration launch) shows no bit, which says only that the bit is
not set at rest. MEASURED. The `metric_changes` counter counts changes between consecutive 5 ms samples of the two
throttle words and of `vgfx_mV` (not "the content of `gpu_metrics`"): 51 to 62 changes in 0.8 s under load, and 41 at
idle in the final run (the lead's rerun: 52 at idle, 53 to 62 under load; the development run: 13 to 19 at idle), all
the idle ones in `vgfx_mV`. So the earlier statement that the content changes only every 50 ms at idle is not
supported by the final run.

## Lead's question 5. Is the FP16 loss with more waves a wall-clock loss at equal clock?

Reading of matrix-timing item 11 and its run logs (`runs/20261117T165700Z-matrix/mat_card-reg_f16.log`, READ): W = 1
gave 211.7 TFLOPS at a counter clock of 3231 MHz (sclk median 3230); W = 16 gave 206.0 TFLOPS at 3223 MHz (3222).
The ratio column divides the event-timed rate by 128 x the same waves' counter clock / 16, so the 0.975 is a
wall-clock loss at that clock; the clock itself differs by 0.25 %, which explains a tenth of the 2.7 % loss. So yes:
a wall-clock loss at (nearly) equal clock. MT-5c above measured it against launch length.

## Checks and deliberate breaks

Final runs: `runs/20261117T210256Z-leftc-b` (every item without a whole-card load; MA-2, MA-1, CH-6 and VK-5 were run a second time
into the same folder after the last fixes to `left_mat.hip` and `vkstudy.c`, so their logs are from the final
sources) and `runs/20261117T210336Z-leftc-b-card` (the whole-card bursts; MA-2's whole-card job and the Vulkan bursts were run a second
time after the fixes, and the burst analyses were rerun by hand with the final `leftc_bursts.py`; `summary.md` keeps
the lines of both passes). Every check below PASSED in the final pass except LD-9's calibration; all 29 deliberate
breaks FAILED as they must (22 in the first folder, 6 of them run again after the fixes; 7 in the second).

| Item | Check (VERDICT) | Deliberate break | Result |
|---|---|---|---|
| SC-11 | `preempt_calib`: B alone starts after a median of at most 81 us and at most 382 us (scheduling item 1 plus 10 us mapping); `preempt_consistent`: every B started after A's end or during a gap of A's waves; all 128 waves on 64 CUs | limits set to 1 us (changes only the calibration limit, not the overlap detection; no positive control) | PASS; break FAIL |
| MA-2 | `left_mat_hazard2_prediction`: every forbidden result equals the rule's prediction (fitted waves; under card load every wave by the rule or all new); V_NOP version right; D1 right; every pair wave beside a busy wave; `left_mat64_hazard`: controls | A cases predicted with every row stale; wave64: claims no stale read (both without `--card`: the whole-card classification has no break of its own) | PASS; breaks FAIL |
| CH-6 | `left_mat_mask2`, `left_mat64_mask`: matrix forms ignore EXEC, dot controls obey it, times equal the known cost and are equal with one lane | the dot controls expected to ignore EXEC | PASS; breaks FAIL |
| MA-1, MA-3 | `left_mat_mods2`: raw words equal the assembled ones; NEG_HI ignored; C bits as predicted on 16-bit sparse forms | NEG_HI expected to act; C bits expected ignored | PASS; break FAIL |
| LD-6a | `tr_inflight2_calib` (plain b128 N = 1 against lane 2), `tr_inflight2_data` | expected xor off by one | PASS; break FAIL |
| LD-6b | `tr_mask_full` (full EXEC equals the layout model) | lanes 3 and 5 swapped in the model | PASS; break FAIL |
| LD-7a | `line_calib` (reload a hit, load after invalidation a miss) | load after invalidation claimed a hit | PASS; break FAIL |
| LD-7b | `share_calib` (own region a hit right after touching, fresh region not, all groups on their own CU) | fresh region claimed a hit | PASS; break FAIL |
| LD-7c | `sbuf_calib` (s_load reproduces item 4 within 10 %; every value right) | item 4's figures doubled | PASS; break FAIL |
| EQ-1a | `arith_realw` control (section 2), every tensor, scaled and unscaled | direct count plus one | PASS (25 jobs); break FAIL |
| EQ-3 | `arith_stats_quantbf16` control for BF16 | direct count plus one | PASS; break FAIL |
| EQ-2 | `arith_qworst_nosub`, `_k1024` controls and recheck; `arith_qworst_nonfinite` | control expectation off by one; NaN expected finite | PASS; breaks FAIL |
| MT-7 | `left_prio_levels`: calibration alone, higher level first in every unequal pair, age rule at equal levels (normal object) | lower level expected first | PASS (both objects); break FAIL |
| MT-5b | `tim_mat_handover_calib`: FP16 95..120, FP8 8..18 cycles per hand-over at W = 2, 4, 8 (item 9a) | FP16 expected 13 | PASS; break FAIL |
| MT-6 | `left_rot`: known answers 16, 18, 19 | 18 expected for A and B rotating | PASS; break FAIL |
| VK-5 | `check2`: every word equals the step model | model with round toward zero | PASS; break FAIL |
| SC-4 | `shift_control`: unshifted 0.988 +-0.01, every shift <= 3 % | the unshifted data judged by the 3 % limit | PASS; break FAIL |
| SC-3 | `throttle_control`: PPT0 by power AUC >= 0.9, shuffled 0.4..0.6 | shuffled labels for the first check | PASS; break FAIL |
| SC-12 | `keepalive_calib`: compute queue alone slow share 0.20..0.52 (item 6: 0.30 to 0.42 +-0.10), keep-alive submissions > 0 | claims 0 % slow when alone | PASS; break FAIL |
| SC-5 | `burst_idle`: no throttle bit at rest; loaded bursts: >= 20 samples and >= 3 metric changes | idle expects a bit | PASS (5 bursts); break FAIL |
| MT-5c | `mt5c_calib`: W = 1 at 200 ms 1.000 +-0.005, W = 16 at 200 ms 0.977 +-0.006 (item 11) | W = 16 claimed equal to W = 1 within 0.2 % | PASS (0.9991, 0.9823; the W = 16 limit is 0.977 +-0.006, so 0.9823 is inside by 0.0007); break FAIL |
| MT-3 | `mt3_ratio`: every form at W = 1 within 0.99..1.01; every burst placed W waves on all 128 SIMDs | half the cycles claimed | PASS; break FAIL |
| LD-9 | `ld9_calib`: tr_b128 at 32 MiB 5346 GB/s +-10 % (section 5), DRAM 629 +-5 % | twice the DRAM rate claimed | FAIL: 9,543 GB/s at 32 MiB (DRAM 634 passes); the break fails too, but with the calibration already failing it proves nothing |
| VK-3 | the existing load checksum (exact) and the 9.5 TB/s ceiling | the pipelined shader with a one-tile offset error | PASS (10 bursts); break FAIL (checksum wrong) |
| VK-4 | as VK-3 | (VK-3's) | PASS (4 bursts) |
| VK-7a | ratio to 128 x counter clock / 8 within 3 %, exact totals | 4 cycles claimed (ratio 2 expected) | PASS (10 bursts, two per form; the first pass's TFLOPS verdict failed one burst and the criterion was changed, see the section; the second pass would also pass the old one); break FAIL |
| VK-2 | the existing `mat` verdict (HIP figure +-10 %, exact) | half the HIP figure | PASS; break FAIL |
| listings | every back-to-back pair and V_NOP control in the code object; every timed chain (16 and 48 per form, wave32 and wave64); 56 and 72 transposing loads before one wait; 16 per form in `mat_card`'s loops; 1,024 per form in `tim_mat`; 3,552 in `left_rot` | a wrong pair and a run of 47, done by hand; they left no log in either run folder and cannot be checked from them | PASS |

Not checked by a control of its own: LD-6b has no plain-load control in the same kernel (the EXEC-setting assembly is
the one CH-6's dot controls validate); VK-3's and VK-4's single bursts have no repeat inside a pass (two passes and the
lead's rerun exist and are reported in their sections); LD-9's cells are single bursts, two at 32 and 512 MiB, and
VK-3, VK-4 and LD-9 use single bursts where the other items use medians.

## Where the evidence is, and the rerun from the repository tree

This lane was built and run in a staging folder (`rdna3-fp8/build/rdna4-runs/leftc-b/`) against a snapshot of
master bab895c, and its files were copied into the repository by the study lead after review. Its run folders are
kept beside the repository's other runs: `runs/20261117T210256Z-leftc-b` and `runs/20261117T210336Z-leftc-b-card`
(the lane's final runs), `runs/20261117T211450Z-leftc-b-leadrerun` (the lead's rerun in the staging folder, which
the text below calls `runs/lead/rerun`). After the copy, the lead built and ran everything from the repository tree
(commit 51ec14a, on top of lane A): `runs/20261117T230300Z-leftc-b`, 187 VERDICT lines, 29 of 29 breaks FAIL as
intended, no job skipped, the only failing check again `ld9_calib` (9,547 GB/s against the earlier 5,346), highest
hotspot reading 83 C, profile "Testing" afterwards.

## How to rerun

```
scripts/study-leftovers-c-b.sh all                         # build, every job, every break (whole-card jobs wait for 55 C)
scripts/study-leftovers-c-b.sh run SC-11 MA-2              # selected items (IDs as in this file)
scripts/study-leftovers-c-b.sh breaks SC-11                # their deliberate breaks
LCB_NOCARD=1 scripts/study-leftovers-c-b.sh all            # everything without a whole-card load
```

Environment: `STUDY_REPO` (shared headers and unchanged sources; default the repository), `CARD_ENV`, `SC_RUNS`
(the kept runs SC-3 and SC-4 read; default `$STUDY_REPO/runs`), `REALW_MODEL` (the FP8 GGUF for EQ-1a, read only),
`LCB_BIN` (binaries, default `build/lcb`), `LCB_RUN_DIR`, `LCB_GATE_S` (how long a whole-card job waits for 55 C,
default 300 s). Every GPU job takes the shared lock, runs under `nice -n 10` and a time limit, needs 8 GB available
and the LACT profile "Testing" (`lact cli profile get`, read only). The final runs used `STUDY_REPO` = a `git
archive` of master bab895c and `LCB_GATE_S=3600` for the whole-card folder.

## Not measured

- Thermal gate (a note, not a gap): every whole-card burst had to start below 55 C hotspot, and for the first hour the
  idle card sat at 55 to 60 C with its fan stopped (zero-RPM mode; the fan's stop temperature is 55 C, read-only from
  `gpu_od/fan_ctrl`); three development attempts waited 300 s each without seeing 54 C. In the final runs the hotspot
  dipped to 54 C, the first bursts started the fan, and from then on the gate opened within seconds between bursts.
  Every whole-card burst of the final runs started at a gate reading of 49 to 54 C (the bursts' own start readings, 50 to
  54 C; the lead's rerun 48 to 54 C). Highest hotspot found in the logs: first run folder 58 C (SC-11, sleeping waves;
  57 to 58 C there), whole-card folder 84 C (MT-3 BF8 and sparse BF8 at W = 16; 71 C in its breaks), the lead's rerun 85 C
  (MT-3 sparse BF8 at W = 16). The script's own summary lines said 79 C for the whole-card folder and 79 C for the rerun
  because the script read only the `HOTTEST` lines of `sched_trace` and not the `THERMAL highest` lines of `mat_card`,
  `lds_tr` and `vkstudy`; the script now reads both kinds (bookkeeping only; the summary files of the runs keep the old
  figure).
- VK-4 as planned (a "lean" shader with the per-lane address computed once, written directly in GLSL): the per-lane
  address code is generated by ACO inside `coopMatLoad`, and GLSL has no way to write or place it; a buffer-reference
  variant would need the device-address feature, which the shared `vkh.c` does not enable. VK-3 shows that
  restructuring the loop can still make ACO hoist that code out of the loop (listing: `v_mbcnt`, `v_bfrev`,
  `v_mul_lo_u32` before the loop only; 39 to 22 instructions), so a lean loop was reached through VK-3's variant, not
  through a lean shader. Replaced by that variant plus a dose test (8 extra dependent vector instructions per tile in
  wave32 and wave64), itself a whole-card item.
- MT-5b at chain lengths other than 256; MT-7's memory-bound partner (item 9 already shows a load wave neither slows
  nor is slowed); LD-7c's cold loads (the miss path is item 7a's); LD-6a from L0 (a 16 KiB window cannot hold batches
  of 24 or more tiles).
- SC-12 with a continuously busy card: the keep-alive was a stream of tiny dispatches, not a long-running shader.
- SC-11 positive control: a second kernel that fits beside the first (so it must start early) and is seen to start
  early; without it the overlap detection has never been seen to fire. SC-11 with busy (not sleeping) waves.
- MA-2 under whole-card load: a deliberate break of the card-level classification, repeat launches (the claim rests on
  one launch of 512 waves per pair), and a run with one pair wave per SIMD next to the chain waves to separate four
  interleaved pair waves from the load of the whole card.
- SC-3 and SC-5: a load whose onset is not preceded by a calibration launch and that varies the rise rate or the
  start temperature independently of the time since the load started.
- LD-5 (staged GEMM), EQ-1b, LD-8 and LD-10: out of this pass by the brief.

## Changes to the draft plan

- SC-11: the holding kernel sleeps instead of running a dependent FMA chain (near-idle power, the same occupancy);
  180 ms instead of 64 ms as the longest (kept below the 0.2 s burst limit); a same-process two-stream setting added;
  the classification uses each wave's longest REALTIME gap, and the polling interval was raised after the first
  version showed polling starvation; the calibration limit uses scheduling item 1's after-idle tail (372 us), not
  70 us, because B launches after 40 ms of idle in the control trials. The SC-11 jobs are gated like few-wave jobs
  (no 55 C wait): their waves sleep, so the card was expected to stay near idle power (final runs: hotspot 55 to 56 C at the start, highest
  reading 57 to 58 C, reported power 45 to 46 W before and 49 to 50 W after each job; power during the jobs was not
  recorded).
- MA-2: the plan's count prediction (64 per wave as A, 256 as B, within 1 %) ignored the data: an old word can give
  the same product, so counts fall below those numbers. Replaced by an exact word-for-word prediction from the same
  rule. Wave64 in a separate source (`left_mat64.hip`, one binary per wave size) instead of a code object loaded with
  `hipModuleLoad`. The whole-card setting runs the pair while every SIMD's matrix unit is busy for about 80 us, not a
  0.1 s burst (the pair runs once, so a longer load adds heat and nothing else).
- CH-6: the forms were put in a shared header so wave32 and wave64 run the same test; the timing uses the best of 7
  samples (a desktop disturbance hit the median once in development).
- MA-1 and MA-3 run as one job (`mods2`).
- LD-6a: L2 and memory only (the 16 KiB L0 window cannot hold batches of 24 or more tiles); the step at 64 and the
  ratio to plain loads are INFO, not verdicts (exploratory).
- LD-7a: the hit level is the reload of X, the miss level the load after `s_dcache_inv`, both measured in every trial;
  a second series from X + 64 checks the alignment.
- LD-7b: placement-chosen partners in one launch per relation (every group both touches and reads) instead of a single
  toucher, so each relation has 192 samples; the "fresh" region is in L2 or the memory-side cache (just written by the
  host-side builder), not cold DRAM, so it serves as a "not a hit" control only.
- LD-7c: hot loads only (the cold comparison was dropped: the miss path is the same cache, item 7a).
- MT-5b: the cost per hand-over uses item 9a's model (span(W) - span(1) - (W - 1) x 256 x the nominal chained cost);
  my first formula used the measured one-wave cycles per instruction and gave 83 instead of 105 (caught by the
  calibration verdict).
- MT-7: in `left_prio.hip` (as planned); a `roles` argument was added to the kernels for the three-wave rows; 16
  level-pair kernels; both code objects (21 of 21 descriptors checked).
- MT-6: five new variants in `left_rot.hip` (B shifted by 2, the vector instruction before every second matrix
  instruction, VOP3 forms); "B changed to a copy with the same values" needs no variant: every set holds the same
  values already.
- VK-5: a new `check2` mode with the step model on the host instead of `arith_vkcheck.cpp` (every A and B element is
  0.25, so the model is four or two rounded additions per element).
- SC-12: the keep-alive is a second thread resubmitting the graphics queue's tiny dispatch (no new shader).
- SC-3: the planned rise-rate, power and time variables, with AUC as the measure of association and the PPT0-power
  pair as the known answer; SC-4: as planned.
- Whole-card items: one burst per locked job (the plan's probes looped over bursts inside one process, which would
  hold the GPU lock through the cool-down waits); MT-3 at 0.1 s for every form, W = 1 twice and W = 16 once;
  LD-9 with a "unique" walk in which no wave rereads its own data (reading the code showed that in `bw` each wave
  alternates between the same two 2 KiB groups at a 32 MiB window, a hypothesis at the time that the 5.3 TB/s was
  per-wave reuse in L0/L2 rather than neighbouring waves sharing lines; LD-9's result neither supports nor rejects
  it, the calibration having failed); SC-5 at 5 ms sampling, 0.1 s for FMA and FP8 and 0.2 s for FP16 (the FMA- and
  FP8-class limit is stricter than the lead's 0.2 s), stopped at 90 C; the development run counted only 13 to 19 changes
  of the throttle words and `vgfx_mV` in 0.8 s at idle, the final run 41 (rerun 52), so how often the values update
  is not settled and 5 ms sampling may repeat values.
- EQ-1a: the stand-in was run once (it does not depend on the tensor); 12 tensors from 4 layers. EQ-2: three separate
  runs (`--nosub` at K = 16 and 256, `--k1024` with subnormals allowed, `--nonfinite`).

## Deviations from the run rules

Checked against the logs of `runs/20261117T210336Z-leftc-b-card` (and the first folder where named). No probe was
changed for any of these.

- Bursts longer than their limits (0.1 s for FMA- and FP8-class loads, 0.2 s for FP16-class): MT-3's 8-bit forms ran
  101.5 to 103.2 ms (all MT-3 bursts 101.4 to 103.9 ms); MT-5c's 200 ms target with 16 waves ran 205.7 to 205.9 ms
  (W = 1: 200.9 to 201.0 ms; the 100 ms and 50 ms targets 101.4 to 103.9 and 51.7 to 53.3 ms); SC-5's FP8 burst
  106.1 ms and FP16 burst 203.9 ms; LD-9's shared 64 MiB burst 222.8 ms (above 0.2 s: its length was sized from an
  expected rate of 5.3 TB/s and the walk read 2.4 TB/s). For `mat_card` the cause is that its iteration count is
  sized for a clock of 3.2 GHz (source comment, `mat_card.hip` line 20) and the bursts ran at 3.0 to 3.2 GHz, so a
  slower clock overshoots.
- More than 3 bursts of one configuration over the two passes and the breaks: VK-7a FP8 x BF8 six (two per pass and
  the break of each pass), the other four VK-7a forms four (two per pass); VK-2 FP16 four (one per pass and the break of
  each pass); VK-3 and VK-4 share the same commands for FP16 4 MiB with wave64 (naive) and wave32 (naive), four bursts
  each (two passes, once under each item name).
- The idle control of SC-5 had no calibration launch before it, the loaded bursts did (see SC-5).

## Files changed and files new

Changed (copied from master, then edited): `src/study/sched_share.hip`, `scripts/sched_pair.py`,
`src/study/left_mat.hip`, `src/study/lds_tr.hip`, `src/study/lds_smem.hip`, `src/study/arith_realw.cpp`,
`src/study/arith_stats.cpp`, `src/study/arith_qworst.cpp`, `src/study/left_prio.hip`, `src/study/tim_mat.hip`,
`src/study/left_rot.hip`, `src/study/mat_card.hip`, `src/study/sched_trace.hip`, `src/vk/vkstudy.c`,
`src/vk/shaders/load.comp`, `src/vk/shaders/mat.comp`, `src/vk/vksched.c`.
New: `scripts/study-leftovers-c-b.sh`, `scripts/leftc_listing.py`, `scripts/leftc_bursts.py`,
`scripts/leftc_throttle.py`, `src/study/left_mask.h`, `src/study/left_mat64.hip`, `measurements/leftovers-c-b.md`.

## Corrections after review (2026-11-17)

After an independent review of the first version of this file (verdict FIX-PASS: no answer overturned). Each correction
was checked against the logs of the two final run folders before it was made; the lead's rerun (`runs/lead/rerun`) is
cited where it gives a second value.

- EQ-1a (A): "the scale changes nothing measurable from K = 256" was false for FP8 (FP16 changes 12 to 49 of 720 k,
  "differs from exact" 1.15-2.83 % to 2.15-4.83 %; BF16 28.6-39.9 % to 35.8-46.2 %); "20 times" at K = 16 is 0 to 22 (no
  ratio); the E4M3 maximum is 0.0167 % unscaled, not 0.0017 %. Recomputed from the 24 logs.
- EQ-2 (B): "K = 1024 is no worse than K = 256" replaced by the figures (FP16 46 against 34, E4M3 x E4M3 26 against 18,
  the other two paths lower) and the note that the K = 1024 search had a quarter of the iterations.
- MT-5c and MT-3 (C): the loss is associated with longer launches and a higher clock, not separated; "neither per
  launch nor per instruction" removed; the real spreads given; the narrow calibration pass noted; the W = 1 clock
  rise without loss added.
- SC-3 and SC-5, and both "In short" bullets (D): "fast rise, not high temperature" reduced to what was seen; time since
  load start separates better (AUC 0.994 against 0.946); two load onsets only; all four SC-5 bursts set the bit at 66 to
  67 C; SC-5's calibration launch precedes the burst and the idle control had none.
- SC-5 metrics (E): idle is 41 changes (rerun 52) in the final run, not 13 to 19 (development run); the counter counts the
  throttle words and `vgfx_mV`, not the `gpu_metrics` content.
- Tables from development runs (F): SC-11, LD-6a and LD-7b (64 KiB column and fresh region) rebuilt from the final logs; the
  MA-2 wave32 table relabelled (its counts equal the final run's); every other table was checked against the final
  logs (two or more cells each, most in full): CH-6, MA-1 and MA-3, MA-2 wave64 and whole-card, LD-7a, LD-7c, EQ-1a, EQ-2,
  EQ-3, MT-7, MT-5b, MT-6, VK-5, VK-7a, VK-2, SC-3, SC-4, SC-12, MT-3, MT-5c, LD-9 and SC-5 agree, with small
  rounding fixes in MT-7's round-robin column.
- VK-3 and VK-4 (G): both passes shown side by side; conclusions lowered to what holds in both (FP16 32 MiB window and
  the shorter-loop gap are not); "GLSL cannot move the address code" reconciled with VK-3's 39 to 22 instruction listing.
- Hotspot bookkeeping (H): the script now also reads the `THERMAL highest` lines; true maxima per run folder stated (58,
  84 and, for the rerun, 85 C); the burst start range is 49 to 54 C; SC-11's "58 to 62 C" replaced by 55 to 56 C at the
  start and 57 to 58 C at the highest, with power during the jobs not recorded.
- Rule deviations (I): new section "Deviations from the run rules" (burst lengths above the limits, bursts per
  configuration), all figures taken from the logs.
- LD-9 (J): burst lengths stated (25 to 55 ms, 99 ms DRAM, 223 ms shared 64 MiB), the unique walk's shared reads stated, the
  gap at 32 MiB marked as confounded with the clock ramp, "L2 and MALL together" dropped, "ruled out" reduced to "not
  supported in this run", the rerun's 9.5 TB/s and the unexplained 5.3 TB/s stated.
- VK-7a (K): 10 bursts, criterion change kept, the second pass would also pass the old criterion (lowest 411.0 TFLOPS).
- SC-11 method (L): conclusion limited to sleeping waves; the missing positive control listed under "Not measured"; the
  note on one wave's tight REALTIME polling kept.
- MA-2 under whole-card load (M): one launch; stale counts are lower bounds; no card-level break; four pair waves per SIMD
  named as the simpler explanation; wave64 "as A" is a fit that repeats; the missing break and repeats listed.
- Listing breaks (N): the by-hand breaks are stated to have left no log.
- Optional points (O): SC-12 against the "on" run's own median (0.198 against 0.253) and the rerun's values; single bursts
  named for VK-3, VK-4 and LD-9.
- Hand-encoded words (P): the new words and their validation added to MA-1 and MA-3.
- Smaller changes made while checking: SC-11's largest wave gap is 4.08 us (not 3.3); LD-6a's memory ratio is 1.09 to 1.18;
  MT-5c's REALTIME-span agreement is within 0.15 %, not 0.1 %.
