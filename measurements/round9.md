# Round 9: load order, loads beside matrix work, data across dispatches, the core clock at a fixed cadence, memory bandwidth at 1408 MHz

Checked by the lead against the run folders. Reviewed in three rounds by GPT-6.2 Sol (Codex) and Gemini 4 Neon (the last round's blocking findings were applied and checked by the lead; no fourth round), with a keep-or-cut judgement by Fable 5.2 after round 2; what was cut or marked not interpretable is said per item.

Card RX 9070 XT (gfx1201), 2026-11-24, run folders in `runs/` (git-ignored). Three chains and some single runs:

| Chain / run | Folders | Used for |
|---|---|---|
| First chain, 21:55-22:10 UTC (`runs/r9-chain-20261124T215518Z.log`) | `20261124T215518Z-r9-ldorder` (B), `20261124T215531Z-r9C` (C), `20261124T215643Z-r9-ldmat` (A), `20261124T215916Z-r9E` (E), `20261124T220342Z-r9D` (D) | B and D stand. A, C and E of this chain are **superseded** by the third chain (code faults found by the reviews, listed per item). |
| G1, 22:11-22:14 UTC | `20261124T221141Z-r9G1` (first run, 3 cells), `20261124T221243Z-r9G1` (rerun; log `runs/r9-G1b-20261124T221243Z.log`) | G1 |
| Second chain, 22:28-22:32 UTC (`runs/r9-chain2-20261124T222849Z.log`) | `20261124T222849Z-r9-ldmatk` (A load ratio), `20261124T222854Z-r9C2` (C2, superseded), `20261124T223006Z-r9E2` (E2) | ld_mat_k, E2 |
| Third chain, 23:39-23:48 UTC (`runs/r9-chain3-20261124T233903Z.log`) and the ld_mat rerun (`runs/r9-ldmat-rerun2-20261124T234911Z.log`) | `20261124T234112Z-r9C`, `20261124T234234Z-r9C2`, `20261124T234356Z-r9E`, `20261124T234911Z-r9-ldmat` | C, C2, E, A on the fixed code. The ld_mat attempt of this chain (`20261124T233903Z-r9-ldmat`) ran the one-wave cells and the first whole-card cell (the matrix-only reference, 3 bursts) and stopped at the second whole-card cell (L0, D = 0), whose start check (hotspot below 50 C at that time) refused it at 53 C after 120 s; the check was changed to 58 C and the rerun used. The thresholds in the program: a recorded whole-card burst starts below 58 C (a one-wave burst below 60 C); the three calibration launches before a cell's bursts have no such check; the gate starts a job below 60 C and stops it at 95 C. In the rerun the 300 whole-card bursts started at 45-57 C (169 of them at 50 C or above) and reached at most 72 C; the calibration launches' temperatures were not recorded. |

Every analysis was run again on its archived logs after the fixes (no new GPU run): B after the first round, D, G1, E2, E, A and the load-ratio follow-up after the second
; C and C2 give the same output as before. The verdict counts below are the new ones. Each run folder holds the new `analysis.txt` (`ana-ld_mat.txt`, `ana-ld_mat_k.txt`) and beside it the file as it was before
(`analysis-pre-fix.txt`, `ana-ld_mat-pre-fix.txt`, `ana-ld_mat_k-pre-fix.txt`; for B `ana-ld_order-corrected.txt`). The saved D, G1 and E2 files had still printed 62 of 66, 19 of 23
and 4 of 8; the new ones print 61 of 66, 34 of 39 and 5 of 9. The fixes are listed under "Files changed".

Labels as in `round5.md`: MEASURED (a number in a log of a run folder), READ (a source file, a listing or an earlier measurement
of this study), INFERRED (worked out, not observed).

**Caveats.** Our wall-clock numbers (GB/s, MHz, ns) are for this overclocked card, which runs the LACT profile "Default":
+150 MHz core offset, -50 mV, 374 W cap (the reported power is about 1.2x too low because of the regulator mod), memory request
1409 MHz. Cycle counts are probe core cycles: the same elapsed latency is a different number of core cycles at a different core clock, so counts from different clock
states are quoted with their elapsed time. Matrix-rate figures hold for the instruction streams tested. Every cell
states its memory clock where it matters; the memory clock moved between and inside jobs (456 to 1408 MHz). External figures
(PR 57, the Windows tools) are quoted with the conditions their source states.

## Plain summary

1. Loads come back in the order they were issued: in every pattern tested, a wait that allows one load still out returned only when the older load had arrived, and no
   early return appeared in 195,840 checks (MEASURED).
2. A load that lands while FP8 matrix work runs costs its wave about 25 cycles, and the cost follows the number of loads, not the number of waits: one load per 4 or 8
   matrix instructions cut the extra to a quarter and an eighth (MEASURED on one wave; "per landing load" is INFERRED).
3. With one short dispatch every 16.7 ms the shader saw a steady core clock, 2960 MHz through HIP and 2022 MHz through Vulkan, but six of seven Vulkan runs also had a few
   dispatches near 1180 MHz that a median hides; `profile_standard` held 1623 MHz with no such dips in the six runs that started, and it also changed the memory clock
   (MEASURED).
4. Through HIP, data written by one launch was read by the next no faster than data nobody had touched; through Vulkan, the next dispatch read at same-kernel speed with no
   barrier or an execution-only barrier and more than 200 cycles per load slower with the five other barrier kinds; no cache level is named for either (MEASURED).
5. Alone, the card read GDDR6 at 705 GB/s with the memory clock at 1408 MHz, and L0 at 81 cycles per load (MEASURED).
6. Not answered this round: what loads cost beside matrix work on the whole card and how much lookahead they need (the probe failed its own reference checks after its fix),
   whether cache or memory latency depends on the memory clock (neither attempt could separate the clock from the load that raises it), and why the Vulkan cold read is
   slower than HIP's.

### Status by item

- **B, load order (done; 11 of 12 controls passed as written, `pure_hit` failed its written gate and is read for the reason given under Item B).** MEASURED: in the tested
  patterns (global loads against global, LDS and scalar loads; one hit with one miss; five-load mixes), a wait for "at most 1 load still out" returned when the older of two
  loads had returned. A cache hit issued after a GDDR6 miss waited for the miss (636-647 cycles); a hit issued before it did not (75). No early return was detected in
  97,920 lane checks per repeat (195,840 in total) for these load types and widths. The first control (one cache hit, expected 129-135) failed because the expectation took
  the wrong row of an older table (READ); the probe gave 75 against 77 in `matrix-timing.md` 8a.
- **A, loads feeding FP8 matrix work (not trustworthy by its own rule; cells reported, not interpreted).** The first run had two probe faults (READ) and is superseded. The
  fixed rerun fails 3 of its 6 controls: the matrix-only reference reads 15.0-15.1 cycles against 7.9-8.1 (one wave) and 7.9-8.3 (whole card) (the fix put two dependent
  vector instructions between the matrix instructions; their cost was not isolated), and the D = 60 cell reads 39.3 against 15.1. The expectation file says that when either
  fails the cells are not interpreted, so the tables in Item A are raw numbers: no cost per load, no lookahead threshold and no whole-card or GDDR6 conclusion is drawn from
  them. One of 300 whole-card bursts printed an impossible clock (58 GHz; R 51, L0, D = 2, burst 2); the cell's median of three did not move. The cost of a landing load
  comes from follow-up A.
- **Follow-up A, load ratio (done; reference 8.06 in its gate; the 13 controls that need no outside number pass; 6 more compare with the superseded first item A run and are
  not counted).** MEASURED, one wave, 256-byte loads: with one load per 1, 4 and 8 matrix instructions the extra over matrix-only is 25.6, 6.3 and 3.3-3.4 cycles per
  instruction at L2 (25.0, 6.3 and 3.1 at the 8 KiB span, where the one-per-instruction cells reach past L0), the same whether a wait stands before every instruction or only
  the first of a group. INFERRED: consistent with a cost per landing load; arrival timing and issue cost were not measured separately. INFERRED: on one wave, at these spans
  and this load width, a matrix loop with fewer loads pays less. Issuing a load earlier covers its latency but, by these numbers and 8a's, not the roughly 25 cycles.
- **C, does the data one dispatch wrote stay in L2 for the next? (kept as observations; 18 of 23 controls pass; the cold, two-launch and HIP fused-clock controls FAIL; the
  item's question is not answered).** MEASURED: as cell medians the Vulkan cold chase reads 812-842 cycles per load (random walk) and 799-802 (page order), against the HIP
  cold chase's 700 and 488. Every Vulkan cell starts with the memory clock at 456 MHz and ends at 1124 or 1408; its repetitions 1-4 read 1006-1106 (repetition 0, not counted, 1225-1402). In repetitions 12-19
  the Vulkan cold chase reads 777-819 and 764-780 (run medians), in time 232-239 and 237-241 ns against HIP's 218 and 152 ns. The HIP cold cell starts at 1408. The two probes
  share the walk and the dependent load and differ otherwise (index width, a mask and a stamp sum inside the timed loop, the size of the eviction pass), so this is a
  difference between two probes; its cause is open. Through HIP, a consumer launch reads the region the producer launch wrote at 697 cycles per load (random) and 489 (page
  order), against 700 and 488 for a region nobody wrote: consistent with HIP's evicted reference; the HIP check is of the final address only; no cache level and no write-back
  is concluded. On Vulkan, with barrier kinds none and exec the consumer reads at 302-303 cycles per load, the same-kernel level (302); with b2, legacy, vv7, full and event it
  reads 530-559 as cell medians (513-534 in repetitions 12-19, after the memory-clock ramp). The stamp sum and final slot were right in 57 of 57 repetitions of every normal
  Vulkan cell and wrong in 57 of 57 of the race break. None, exec and vv7 give no write-to-read guarantee, so their passes describe these runs only. No cache operation was
  measured for any kind. The item's question, whether the data stays in L2, is not answered.
- **Follow-up C2, page-order walk (kept as observations, see Item C).** MEASURED: HIP cold 488, HIP two-launch 489. Vulkan cold 799-802 as cell medians (764-780 in
  repetitions 12-19); b2 533-535 (520-524 in repetitions 12-19), 264-269 below the Vulkan cold medians.
- **E, memory hierarchy (done on fixed code; 3 of 8 controls fail; bandwidth and L0 stand, the latency question is not answered).** MEASURED: the FP16 background loop ran
  through the probes in 3 of 3 jobs (it paused at its 76 C cap for 24-33 % of its 2.35-2.37 s). L0 reads 75-90 cycles in 18 of 18 points (median 80.6). The L2 chase reads
  253-305 cycles at 1 MiB and 247-433 at 2-4 MiB across the three steady jobs; the control band 215-270 fails (5 of 30 points inside). Reported as measured; nothing about the
  memory clock is concluded from it. Of 123 points, 20 carry a clean memory-clock label. 63 are "too short" (their window held fewer than 3 readings of the 5 ms sampler; all
  of them last 0.3-11.4 ms), 32 have a reading before the point that differs from the one after, and 8 are "mixed" (the clock moved inside the window). So no latency comparison
  across memory-clock labels exists. GDDR6 read bandwidth alone is 705.1-705.9 GB/s at the 1408 label, within 0.8 % of round 7's 700.3. Beside the loop the 1 GiB read gave
  282.5, 373.1 and 352.1 GB/s (the median of each job's timed launches; single launches ranged from 219 to 702 GB/s). The loop pauses for 24-33 % of its run; the files record the pause time per
  job and the launch rates, not which launch overlapped a pause, so concurrent execution per launch is unconfirmed: this is not a steady rate beside a running loop.
- **D, call then wait at a 16.7 ms cadence (done; 61 of 66 controls pass).** MEASURED: the shader sees 2960 MHz through HIP (one run per length) and 2022 MHz through Vulkan
  graphics (seven runs), steady to 0.2-0.5 % between the 5th and 95th percentile, at planned dispatch lengths of 0.3 to 1.7 ms (measured 0.34 to 1.73 ms). Six of seven Vulkan
  runs also had dispatches near 1180 MHz (50 of 1,197 in one run, 20 in another). The five failures: four are about the planned dispatch length (they do not enter the clock
  figure), one is the probe's own cadence verdict (one late submission of 1169 us in `d-vk-1.0-r3`).
- **G1, steady clocks (done; 34 of 39 controls pass, 5 FAIL: four cells never ran, one run failed the probe's cadence verdict).** MEASURED: run medians were stable under both
  levels: 1623 MHz (+-0.07 % over 6 runs) under `profile_standard`, 2022 MHz (+-0.05 % over 10) under `auto`. Several auto runs had low-clock dispatches: within one run the
  5th-95th percentile spread is 0.18-0.23 % under standard and 0.31-0.51 % under auto, but 41 % in `auto-5` and 4.7 % in `auto-6`. Standard also ran the memory at 1124 MHz in
  every sample and auto mostly at 456, so the comparison is not core-clock-only. The Windows figures of PR 57 (1868 MHz; timing spreads) are a different quantity and are not
  compared.
- **Follow-up E2, latency at the 1124 and 1408 labels (not interpretable; 5 of 9 controls pass).** Three of the four controls that say "the background is light" fail (140 and
  550 GB/s against 70; the 1 MiB chase outside 215-270 and 161 % above idle; the 32 MiB page chase at or above 470 in 6 of 8 chases); the fourth, the 1 GiB page chase equal at
  two background rates, passes. The run, the table and the control results are kept; no conclusion is drawn. MEASURED: beside the lightest background read (30 GB/s) the 1 MiB
  chase read about 427 cycles (119-120 ns) against 240 (74-76 ns) idle. The memory clock, the core clock (about 3.6 against 3.2 GHz) and the background traffic all differ
  between the two, so no cause is separated.

## Item B: does `s_wait_loadcnt N` retire loads in order? (`ld_order`)

**Method (READ: `scripts/round9/gen_ld_order.py`).** One wave. Each pattern is a few loads and waits in one inline-assembly
statement, timed with the in-kernel cycle counter minus an empty pattern. Far load (M): 4-byte `global_load_b32` (saddr form,
vector offset) of a line untouched since a 512 MiB flush (GDDR6). Near load (H): the same instruction on a line the prelude just
loaded (L0). Also `ds_load_b32` and `s_load_b32`. Every destination register holds a sentinel before the timed region, and a
snapshot after the wait copies every register, so an early return shows as a sentinel in a register the wait should cover.
Two process repeats (`r9-ldo-1`, `r9-ldo-2`).

**Numbers (MEASURED, `ana-ld_order.txt`, both repeats within a few cycles).**

| Pattern | Cycles |
|---|---|
| one L0 hit and its wait | 75.0 |
| one GDDR6 miss and its wait | 647.0 / 646.0 |
| [miss, hit], wait until 1 left | 636.0 / 647.0 (the miss) |
| [hit, miss], wait until 1 left | 75.0 / 75.0 (the hit) |
| wait 0 after [miss, hit] / [hit, miss] | 639 / 652 and 652 / 682 (about the miss) |
| miss + LDS: wait dscnt / wait loadcnt | 38 (LDS alone 25) / 697 and 690 |
| miss + scalar hit: wait kmcnt / wait loadcnt | 28 (scalar hit alone 27) / 671 and 675 |

All 12 five-load mixes fit the in-order model within max(25 cycles, 12 %); six of them also fit an out-of-order model because
their cost is the same either way. Verdict printed by the analysis: IN ORDER (both repeats). No early return was detected in
either repeat (97,920 lane checks of registers that must hold data each, 195,840 in total; 0 wrong words) for these load types
and widths (MEASURED). The two deliberate breaks (wrong counter waited) show wrong words in 100 % of 1,920 checks each, so the
probe can see an early return. The verdict now needs both break rows and all 12 mixes in the log; a missing one gives
"UNANSWERED" (analysis self-test), and the `stale_visible` control compares the numeric count, which can fail at 100 % (the
old substring test could not).

**The `pure_hit` control, judged.** The first expectation was 129-135 cycles (gate 124-140); the probe measured 75.0 in both repeats.
- The rule that was triggered (READ, `expect-ld_order.md`): a `pure_hit` outside its gate means "report, do not interpret the rest". It was outside in both repeats, so 11 of 12
  controls passed as written in each. The rest is read here for the reason in the last point below; the old control line is `ana-ld_order.txt`, the corrected one
  `ana-ld_order-corrected.txt`.
- Instruction form (READ): `ld_order` uses `global_load_b32 v, voff, sbase` (saddr form, 4 bytes), the form `mem_inflight` also
  uses (`src/study/mem_inflight.hip` line 33).
- Where 129 came from (READ): `memory-system.md` section 5, row L0, 4 bytes, N = 1, 129.1 cycles per batch. `mem_inflight` times a
  repeating loop (`k_inflight`, lines 77-81): each batch period also holds the address arithmetic (a modulo), the loop branch and
  the wake-up charge after the wait. The first expectation's "135.1 at W = 16, 4-byte loads" is the 16-byte row of that table.
- What this probe times is one load and its wait between two counter reads, with the copies after the second read, the same shape
  as `matrix-timing.md` 8a (line 354: "4-byte global loads that hit L0 (77 cycles with the wait)", differential method).
- Why the result is still read: `pure_hit` failed its written gate (124-140) with 75.0 and passes a gate of 69-85 set after the run from the earlier 77 of `matrix-timing.md`
  8a. That gate exists to show the near load is an L0 hit: 75 cycles is below every other level this study has measured and 2 cycles from the earlier figure taken the same way,
  and the order verdict rests on differences inside the probe (636-647 against 75) and on the register checks, not on the hit's absolute cost.
- Verdict: the expectation was wrong, not the probe. 75.0 against 77 is 2 cycles apart. Only 8a's own number is used for the
  correction (the band is 77 +- 8); the old band stays in `expect-ld_order.md` and in the analysis text, and the control
  re-judged on both repeats passes (`ana-ld_order-corrected.txt`; the current analysis script, run again on the same logs,
  gives 12 controls, 0 failed). INFERRED: the gap between 75 and 129 is about 54 cycles of loop and wake-up cost in the older
  figure; not isolated.

**What is open.** Order is shown for global loads against global loads (one hit and one miss), global against LDS, and global
against scalar. Not tested: loads of different widths in one wait, stores, and more than five loads.

## Item A: load, wait and FP8 matrix (`ld_mat`)

**Item verdict: not trustworthy by its own rule; the cells are raw numbers and no finding.** `scripts/round9/expect-ld_mat.md` says: "If (a) or (c) fails the probe is not
trustworthy: report, do not interpret the cells." Both fail below (`a_ref_one` 15.124 and `a_ref_card` 15.006-15.027 against the gates 7.9-8.1 and 7.9-8.3; `c_back` 39.308
against 15.124). The explanation offered for the 15 is untested, and a second one fits as well (below). The fix also changed what the probe is: a stream at half the matrix
rate with vector work in it. So the tables below are reported, and no cost per load, no lookahead threshold and no whole-card or GDDR6 conclusion is drawn from them. The cost
of a landing load comes from follow-up A.

**Superseded tables.** The first run of this item (`20261124T215643Z-r9-ldmat`) had two probe faults found by the reviews and
confirmed in the generator (READ, `scripts/round9/gen_ld_mat.py`): the ring of load registers had `K = D + 2` slots and the loop
body 114 instructions, so for 5 of 6 lookaheads the load for the next iteration's first positions overwrote a pair a later matrix
instruction still had to read; and only the per-lane offset was masked into the level's span while the immediate offsets added
up to 43 KiB, so the "L0" cell read a 36-51 KiB footprint. Its tables (ring bug, L0 overflow) are superseded and not used here.
The rerun (`20261124T234911Z-r9-ldmat`) uses a loop body that is a multiple of the ring size (at least 114) and masks the whole
address `(voff + 256 g)` into the span (a listing check of 28 blocks found 0 failures; ring and mask breaks fail as they must).

**Method (READ: `scripts/round9/gen_ld_mat.py`).** A loop body of `v_wmma_f32_16x16x16_fp8_fp8`, chained through the accumulator.
Position i is: an address add and mask (two vector instructions), one `global_load_b64` (256 bytes per wave) for matrix instruction
i + D, then `s_wait_loadcnt D`, then matrix instruction i. So every instruction has one load issued and one load landing in
steady state. D = 0, 1, 2, 4, 8 and "all" (the largest the registers allow, at most 60: 10, 30, 60, 60 at R = 51, 90, 167, 209
registers). Four levels named by their buffer span (8 KiB, 1 MiB, 32 MiB, 1 GiB; "L0", "L2", "memory-side cache" and "GDDR6"
after `memory-system.md`, not calibrated here). One wave (R 209), and the whole card at 16, 16, 9 and 7 waves per SIMD (R 51,
90, 167, 209), 3 bursts per cell, median reported. The design is a steady pipeline, not "all loads issued once up front".

**Controls (MEASURED, `ana-ld_mat.txt`).**

| Control | Result |
|---|---|
| a_ref_one: matrix only, one wave | **FAIL**, 15.124 against the band 7.9-8.1 (reference 8.0) |
| a_ref_card: matrix only, whole card | **FAIL**, 15.006 / 15.006 / 15.019 / 15.027 at R 51 / 90 / 167 / 209 against the band 7.9-8.3 |
| b_8a: one wave, L0, D = 0 | PASS, extra 96.0 against 75.0 (ld_order hit) + 18 = 93.0, 3.2 % apart (gate 15 %) |
| c_back: one wave, L0, D = all | **FAIL**, 39.308 against 15.124 (gate 1 %); see below |
| d_break_L2: L2 label aimed at L0 | PASS, D = 0 cost 110.9 (L2 314.4, L0 111.1), moved 203.5 of 203.3 |
| placed | PASS, 300 whole-card bursts all placed as W waves on 128 SIMDs |

Summary: 6 controls, 3 failed. The expected values were not edited.

**One bad burst (MEASURED, `ana-ld_mat.txt`).** The analysis now rejects a burst whose `clock_ghz` is outside 0.5-4.0 or whose per-SIMD min or max is more than 20 % from its
median, and prints how many it rejected (the control count does not change; a cell whose every burst is rejected fails a separate data check, `cells_survive`, which is not one of the six controls and which no cell fails in the archived run). Of the 378 bursts (300 whole-card, 75
one-wave, 3 of the break) exactly one is rejected: `r9-ldm-R51-L0.log`, whole card, R 51, L0, D = 2, burst 2 (of 0-2): clock 58.076 GHz, median 312.036 cycles per matrix
instruction, min 14.157, max 466.754, started at 57 C; its two siblings read 15.267 and 15.267 (2.971 and 2.942 GHz). The cell's value is the median of the bursts that remain,
15.267, which is what the median of all three gave: no cell median changed. Cause open: a wrap of the 32-bit cycle counter was suspected (Neon round 2), but the code's unsigned
subtraction survives a plain wrap, and the largest per-SIMD span of this burst is about 2^32 (466.754 x 16 x 4956 x 116 = 4.293e9 against 4.295e9), which would mean an end
stamp earlier than the start stamp used for its SIMD. A 64-bit or cross-checked counter belongs to the redesign.

**The reference reads 15, not 8, and its controls stay FAIL.** The band 7.9-8.1 is the chained rate of a matrix instruction alone (8.0, round 7). In this probe the reference also
carries the address add and mask between the matrix instructions (the generator writes the same arithmetic into a register nothing reads; READ, `gen_ld_mat.py`: a
`v_add_nc_u32` and then a `v_and_b32` on the same register, a dependent pair), and the reference reads 15.1, 7.0 cycles higher. The fix put two dependent vector instructions
between the matrix instructions; their cost was not isolated. The reference and the loaded
cells carry the same arithmetic; but this stream runs at about half the matrix rate of the old 8-cycle
design, so a whole-card cell asks the memory for about half the bytes per cycle and spaces each wave's loads about twice as far apart: whole-card cells are not comparable with
an 8-cycle stream (Neon round 2). The two matrix-only controls stay FAIL against the 8.0 band.

**The one-wave table (MEASURED, R 209, cycles per matrix instruction; reference 15.12; D = 60 is "all"; raw numbers, not interpreted).**

| level | D = 0 | D = 1 | D = 2 | D = 4 | D = 8 | D = 60 |
|---|---|---|---|---|---|---|
| L0 (8 KiB) | 111.13 | 55.43 | 39.46 | 39.44 | 39.35 | 39.31 |
| L2 (1 MiB) | 314.39 | 147.86 | 98.30 | 57.24 | 39.93 | 39.33 |
| memory-side cache (32 MiB) | 496.93 | 235.07 | 158.74 | 98.06 | 52.74 | 39.34 |
| GDDR6 (1 GiB) | 896.49 | 490.28 | 347.17 | 206.33 | 96.97 | 39.36 |

At D = 60 the four levels read 39.31, 39.33, 39.34 and 39.36 against the reference's 15.12. These are the numbers of a probe that failed its own reference checks: no
extra per landing load and no lookahead each level needs is read from this table. (The thresholds would also depend on this stream's own pace, 39 cycles per position, and
would not carry to an 8-cycle stream.)

**The whole card (MEASURED, cycles per matrix instruction per SIMD; reference 15.01-15.03; raw numbers, not interpreted).**

| waves per SIMD (R) | level | D = 0 | D = 1 | D = 2 | D = 4 | D = 8 | D = all |
|---|---|---|---|---|---|---|---|
| 16 (51) | L0 | 16.33 | 15.56 | 15.27 | 15.26 | 15.26 | 15.26 |
| 16 (51) | L2 | 17.46 | 18.07 | 16.97 | 15.68 | 15.28 | 15.26 |
| 16 (51) | memory-side cache | 29.54 | 15.29 | 19.40 | 17.12 | 19.88 | 23.59 |
| 16 (51) | GDDR6 | 148.22 | 143.82 | 140.79 | 140.26 | 137.63 | 133.27 |
| 16 (90) | L0 | 16.32 | 15.55 | 15.26 | 15.26 | 15.26 | 15.25 |
| 16 (90) | L2 | 17.46 | 18.08 | 16.96 | 15.69 | 15.26 | 15.25 |
| 16 (90) | memory-side cache | 29.62 | 15.26 | 19.33 | 17.14 | 19.65 | 30.49 |
| 16 (90) | GDDR6 | 147.08 | 143.20 | 139.46 | 141.07 | 137.36 | 128.05 |
| 9 (167) | L0 | 15.40 | 16.08 | 15.79 | 15.79 | 15.78 | 15.78 |
| 9 (167) | L2 | 30.71 | 15.55 | 19.99 | 16.47 | 15.79 | 15.79 |
| 9 (167) | memory-side cache | 51.09 | 25.73 | 17.70 | 19.83 | 16.56 | 29.13 |
| 9 (167) | GDDR6 | 141.26 | 142.33 | 136.95 | 140.75 | 139.86 | 137.48 |
| 7 (209) | L0 | 16.03 | 17.22 | 15.76 | 15.75 | 15.74 | 15.73 |
| 7 (209) | L2 | 39.33 | 19.84 | 17.27 | 17.16 | 15.74 | 15.73 |
| 7 (209) | memory-side cache | 67.08 | 33.92 | 22.62 | 20.00 | 17.52 | 28.99 |
| 7 (209) | GDDR6 | 146.96 | 141.22 | 137.94 | 137.94 | 142.11 | 133.08 |

The memory clock was 456 MHz at the start of every job and 772 to 1408 at the end of several (`summary.md`), so the GDDR6 rows are not at one memory clock. Nothing is drawn
from this table: not the lookahead each level needs, not that the charge is hidden at L0 and L2, not that GDDR6 is bandwidth-bound, not that no lookahead helps there (D = all
is 3-13 % below D = 0 at every register count in the GDDR6 rows, so that statement of the first version contradicted its own table).

**Shapes in the raw table, not explained.** The memory-side-cache rows are not monotonic in D (R 51: 29.5, 15.3, 19.4, 17.1, 19.9, 23.6), and the L2 rows at R 51 and 90 are
higher at D = 1 than at D = 0 (18.07 against 17.46). The first run showed the same shapes with the ring fault in place, so the ring was not their cause. Whether they are the
memory system or an artifact of the 15-cycle masked loop is not known.

**The `c_back` control (MEASURED, FAIL; not edited).**
- What the control assumed (READ, `expect-ld_mat.md`): at D = 60 the data is back before its use, so no wait blocks and the cost is the matrix-only rate within 1 %; the premise
  cited `matrix-timing.md` 8a, which says data already back when the wait is reached costs 0.9 / -0.4 / 0.9 / 1.0 cycles.
- What 8a says that the control left out (READ, `matrix-timing.md` 8a, "Other cases" table): "load data arriving during 8 or 16 chained FP16 matrix instructions, wait after the
  chain (no blocking): +26.0 / +26.0 (b128 load 25.7; 4 loads 32.2; LDS 25.0)". Whether this probe is in that case is open: the run recorded no load arrival time and no separate timing of the two added vector instructions.
  The 0.9-cycle rows are the case where 192 cycles of `s_nop` (no matrix work running) sit between the load and the wait. Whether the control's premise holds for this stream was not measured.
- The probe is nevertheless not trusted: the control fails by its own written rule, and a_ref_one and a_ref_card fail with it. The 39.308 of the D = 60 cell is quoted as a raw
  number. 8a's +26.0 on 15.124 would be 41.12 (arithmetic only).
- The control is not edited (it stays FAIL in `ana-ld_mat.txt`); its text points here. The old model "extra(D) = max(0, extra(0) - 8 D)" that the analysis prints does not
  fit (L0 D = 1: 40.3 measured against 88.0; D = 2: 24.3 against 80.0) and is not used.

**What is open.** (1) A probe with the addresses computed once outside the matrix stream, which should put the reference back at 8 cycles: not run, and the redesign that this item
needs before any whole-card or lookahead statement. (2) D between 8 and 60 for the 32 MiB and 1 GiB spans. (3) The cause of the 15-cycle reference. (4) The cause of the one 58 GHz burst. (5) The per-load
cost of the load-ratio follow-up (25.00 cycles at L0 and 25.58 at L2, over a reference of 8.064; `ld_mat_k` only) was not tested with other matrix instructions.

## Item C: does the data one dispatch wrote sit in L2 for the next dispatch? (`vkr9 cbar`, `r9_cl2`)

**Item verdict: kept as observations; the item's question is not answered.** The Vulkan cold reference fails its band in C2 for a cause not found, and `expect-C.md` says that
a HIP two-launch read outside 330-540 means "the probe or the card state differs from 2c". So no cell below is placed in a cache level.

**First run superseded.** The first run (`20261124T215531Z-r9C`) gave the Vulkan cold chase an arithmetic next address (a load
whose value only fed a dependency, READ: `r9_chase.comp`) while the HIP cold chase and the Vulkan normal cells followed the loaded
links; the Vulkan cold cell was therefore not matched to the HIP cold cell, and the analysis used the HIP two-launch median as
a "memory-side cache" reference although it sat in the slow band. Both are fixed (a cold Vulkan chase follows loaded links; the
analysis prints the measured cell as the nearest control and names no cache level). The graphics-queue b2 value of the first C2 run
(435) was a low-clock artifact: its first repetitions ran at 1.19 GHz (382 us for 454,472 cycles in repetition 0). The rerun has a
150 ms warm-up dispatch and a per-cell clock check. The first run's analysis input path was also wrong (it read the gate summary
instead of the job log); fixed, and the analyses were rerun on the old logs.

**Method (READ).** A producer dispatch writes a 4 MiB region (32,768 lines of 128 bytes); a consumer dispatch chases 1,024 of
them with dependent vector loads. Cells: Vulkan on the graphics and the compute queue, barrier kinds none, exec, b2, legacy, vv7,
full, event; orders normal, cold (a region whose links were written once before the run and pushed out by a flush before every
repetition, then chased by its loaded links), fused (the same
kernel writes and chases), race (consumer before producer, a break); HIP: two launches, cold, fused, race. 3 process repeats,
20 repetitions each. Every producer write carries a stamp of its repetition in the upper 17 bits of dword 0; the consumer adds
up the stamps of the values it loads, and a repetition is `ok` only if the final slot and the sum are right (READ, `vkr9.c`). The HIP probe has no stamp: its `ok` is the final address alone (READ, `r9_cl2.hip`).

**Numbers (MEASURED, run `20261124T234112Z-r9C`, `analysis.txt`, cycles per dependent load, random walk).**

| Cell | Compute queue | Graphics queue | HIP |
|---|---|---|---|
| cold (flushed region, loaded links) | 842 (run medians 822 855 842) | 812 (824 804 812) | 700 (699 707 700) |
| fused (same kernel) | 302 | 302 | 231 |
| two launches (producer kernel, then consumer kernel) | - | - | 697 (697 708 671, spread 5.4 %) |
| barrier kind none | 303 | 303 | - |
| barrier kind exec | 303 | 302 | - |
| b2 / legacy / vv7 / full / event | 558 / 550 / 555 / 559 / 541 | 540 / 530 / 546 / 542 / 532 | - |

Stamp check (Vulkan: stamp sum and final slot; HIP: final address only): 57 of 57 repetitions `ok` in every normal cell (none and exec included); the race break fails it in
57 of 57 repetitions on both Vulkan queues and on HIP. The repeat controls (cold, fused and two-launch cells) agree within 5.4 % (limit 10 %).

**Controls judged (18 PASS, 5 FAIL).**
- `cold` FAIL on all three (HIP 700, compute 842, graphics 812 against the band 440-600) and `twolaunch` FAIL (697 against
  330-540). READ, `memory-system.md` line 37 and 350: 485-545 cycles is GDDR6 in page order and a random walk after a long walk
  reads 700-820; this chase is a random permutation, so 700-842 is that row. The expectation took the wrong row of an older table; that places no cell of this item in a
  cache level. The bands were not edited.
- `clock_hip` FAIL: the HIP fused cells' core clock moves by more than 10 % inside the cell (run 1 2665-3228 MHz, run 2 2638-3227,
  run 3 2556-3227, median 3225). INFERRED: the short fused cell starts while the clock is still rising; the effect on the
  231 is not separated (three runs agree: 231, 234, 231). The Vulkan clock checks pass (60 cell runs).
- `fused`, `race`, `b2_pointer` and all repeats pass.

**What it shows (observations only).**
- MEASURED: as cell medians the Vulkan cold chase reads 812-842 cycles per load (random walk) and 799-802 (page order, item C2), against the HIP cold chase's 700 and 488.
  Every Vulkan cell starts with the memory clock at 456 MHz and ends at 1124 or 1408 (`uclk_start`, `uclk_end` of each cell, all Vulkan cells of C and C2); its first
  repetitions are slower (repetitions 1-4 read a median of 1006-1106 cycles in the four cold cells of C and C2; repetition 0, not counted, 1225-1402). In repetitions 12-19 the
  Vulkan cold chase reads 777-819 (random walk) and 764-780 (page order) as run medians, in time 232-239 and 237-241 ns against HIP's 218 and 152 ns. The HIP cold cell
  starts at 1408. The two probes share the walk and the dependent load and differ otherwise (index width, a mask and a stamp sum inside the timed loop, the size of the
  eviction pass), so this is a difference between two probes; its cause is open. The gap is real after the ramp but smaller than the cell medians say.
- MEASURED: through HIP, a consumer launch reads the region the producer launch wrote at 697 cycles per load (random) and 489 (page order), against 700 and 488 for a region
  nobody wrote: consistent with HIP's evicted reference. The HIP check is of the final address only. No cache level and no write-back is concluded. (`memory-system.md` 2c
  measured data a previous kernel read, which is a different thing.)
- MEASURED: on Vulkan, with barrier kinds none and exec the consumer reads at 302-303 cycles per load, the same-kernel level (302); with b2, legacy, vv7, full and event it
  reads 530-559 as cell medians (513-534 in repetitions 12-19, after the memory-clock ramp). Neither queue nor kind separates the five beyond 6 %. The stamp sum and final slot
  were right in 57 of 57 repetitions of every cell and wrong in 57 of 57 of the race break. None, exec and vv7 give no write-to-read guarantee (READ, `vkr9.c`: none and exec
  record no memory dependency; vv7's destination access is shader writes only), so their passes describe these runs only. No cache operation was measured for any kind.
- The item's question, whether the data stays in L2, is not answered. No level is named for 530-559, and no L2 write-back is claimed.

**What is open.** The Vulkan cold gap to HIP cold after the ramp; the kind of barrier needed for visibility (not tested beyond these runs); the HIP fused clock ramp. The 150 ms
warm-up before every Vulkan cell is compute-only and lets the memory clock fall to 456 MHz before the cell (INFERRED); a warm-up that keeps the memory clock up, or
discarding repetitions until it settles, belongs to the next C probe.

## Item E: the hierarchy at the memory clock the card reaches (`r9_lat`, `r9_bw`, `r9_bg`)

**First run superseded.** In the first run (`20261124T215916Z-r9E`) the FP16 loop in the "beside a loop" jobs had stopped before
the probes measured anything: `r9_bg` ended at 2.0-2.4 s with a hotspot of 84-88 C (its 88 C tripwire) while the probes ran for
longer (the log shows power 68 W and a 50 C hotspot at the end of `r9_lat`). Its tables are superseded. In the rerun
(`20261124T234356Z-r9E`) `r9_bg` stops at a 76 C cap and resumes, and the driver checks it is alive after the start, after `r9_lat`
and after `r9_bw` (3 of 3 jobs have all three lines; launches 355, 515, 538 in `e-bg-r2` for example). Over its 2.35-2.37 s run it
paused 0.78, 0.70 and 0.56 s at the cap (READ, `e-bg-r?.bg.out`), that is 33, 30 and 24 % of its run: for a quarter to a third of the loop's time it was not running.

**Method (READ, `expect.md` of the run folder).** 3 steady latency jobs (`e-lat`), 3 bandwidth jobs (`e-bw`), 12 cold-start
chases (`e-cold`, half started after the memory clock fell to 456), 3 jobs with the FP16 loop beside the probes (`e-bg`). A 5 ms sampler
runs beside every job. Each point is labelled by the sampler rows inside its own window. A point is clean (carries a memory-clock label) only if its reading before and its
reading after agree and the window holds at least 3 sampler rows, at least 90 % of them at that clock. Every other point is kept apart under one of three states: **too short**
(the endpoints agree, the window holds fewer than 3 rows), **endpoints differ** (the reading before differs from the one after; the window is not evaluated), or **mixed** (at
least 3 rows and less than 90 % at one clock: the clock moved inside the window). Job states are never pooled.

**How the 123 points divide (MEASURED, `analysis.txt`, "90 % rule" line).** 20 clean. Of the 91 evaluated against their window (123 minus the 32 whose endpoints differ), 63 are
too short and 8 are mixed. The first version of this note said the points were mostly "mixed" because the memory clock moved inside the jobs; that is wrong for 63 points:
`run_E.sh` sets no minimum length for a chase, and all 63 too-short points last 0.3-11.4 ms, which a 5 ms sampler covers with 0-2 rows. The mislabel only hid clean
points, so no number changes. A minimum point length in `run_E.sh` belongs to a rerun. (It was also first written that "123 of 123 points were checked against a window": the
window was evaluated for 91; the other 32 were labelled from their endpoints alone.)

**Numbers (MEASURED, `analysis.txt`; median over all points of a job state, cycles; the label states of the points in brackets; per-label medians are in `analysis.txt`).**

| Size | steady (3 jobs) | cold-start (12 jobs) | beside the loop (3 jobs) |
|---|---|---|---|
| 16 KiB (L0) | 80.6 (3 too short) | 80.6 (12 too short) | 80.6 (3 too short) |
| 1 MiB (L2) | 267.8; per job 267.8, 253.1, 305.3 (3 too short) | - | 255.5 (3 too short) |
| 2 MiB | 384.0; per job 384.0, 387.4, 246.7 (3 clean at 456) | - | 400.1 (3 too short) |
| 4 MiB | 430.7; per job 408.4, 430.7, 433.4 (3 clean at 456) | 411.4 (12 too short) | 386.2 (3 clean at 456) |
| 32 MiB | 469.8 (1 clean at 456, 2 mixed) | 466.9 (1 clean at 456, 11 endpoints differ) | 462.4 (2 endpoints differ, 1 mixed) |
| 512 MiB random | 1037.2 (2 endpoints differ, 1 mixed) | 795.8 (8 endpoints differ, 4 too short) | 861.6 (3 mixed) |
| 512 MiB fresh | 481.6 (3 too short) | - | 481.6 (3 too short) |
| 1 GiB fresh | 485.7 (1 clean at 456, 2 too short) | - | - |
| 1 GiB random | 1052.3 (2 endpoints differ, 1 mixed) | - | - |

Bandwidth ("split" read, GB/s): steady 1 GiB 705.9, 705.1, 705.2, all three with a clean 1408 label; 512 MiB 708.3 (3 endpoints differ); 32 MiB 2035.2 (2 too short, 1
endpoints differ); 1 MiB 17,125; 2 MiB 13,523; 4 MiB 13,675 (too short or endpoints differ). Beside the FP16 loop: 32 MiB 2185 (2 clean at 1408, 1 too short); 1 GiB 282.5, 373.1
and 352.1 at a clean 1408 label. Each of these is the median of the job's 5 timed launches (READ, `r9-e-bg-r?.log`, `r9-e-bw-r?.log`: each 1 GiB launch makes 2 passes over the buffer and each 32 MiB launch makes 64); single launches beside the
loop ranged from 219.1 to 293.2 GB/s (lowest) and 700.6 to 702.1 GB/s (highest) (READ in `r9-e-bg-r?.log`). The loop pauses for 24-33 % of its run; the files record the aggregate pause time
and the launch-rate ranges, not which launch overlapped a pause, so concurrent execution per launch is unconfirmed: this is not a steady rate beside a running loop.

**Controls (5 PASS, 3 FAIL).** `complete` PASS (21 jobs present in the expected file slots, each with its expected points: 9 latency chases in `e-lat`, 7 bandwidth points in `e-bw`, 4 chases
in `e-cold`, 7 chases and 2 bandwidth points in `e-bg`; the analysis fails a duplicate or a missing point inside a job, and does not check a job's identity). `bg_running` PASS (3 of 3). `l0_chase` PASS (75-90
cycles in 18 of 18 points). `pointers` PASS (96 of 96). `label_shift_break_caught` PASS. The failures:
- `l2_chase` FAIL: 1-4 MiB chase between 215 and 270 cycles in 5 of 30 points (all job states). The band was written from "MEASURED 230-255", which READ
  in `memory-system.md` 2b holds up to about 1 MiB; at 2, 3 and 4 MiB that table reads 297-358, 368-443 and 362-418. The L2 chase reads 253-305 cycles at 1 MiB
  (267.8, 253.1, 305.3) and 247-433 at 2-4 MiB across the three steady jobs, and the first run read 237-406 for the same sizes (no label comparison between them). The 1 MiB points
  scatter as well, so the band's range was not wrong for 2-4 MiB alone. Reported as measured; nothing about the memory clock is concluded from it; the control is not edited.
- `gddr6_bw_1408` FAIL and `label_vs_bandwidth` FAIL: both pool the three beside-the-loop jobs (282.5, 373.1, 352.1 GB/s,
  0.40-0.53 of 0.5031 GB/s per MHz); the three steady jobs are within 0.8 % of 700.3 GB/s. The lower bandwidth beside a running loop is
  expected (the two share the card, and the loop was paused for 24-33 % of the run); INFERRED, the part of the drop that comes from the loop's pauses or from shared CUs
  against shared memory is not separated.

**What it shows.** MEASURED: L0 reads 75-90 cycles in 18 of 18 points (every median 80.6). GDDR6 read bandwidth alone is 705.1-705.9 GB/s at the 1408 label, three jobs with
clean windows, within 0.8 % of round 7's 700.3. The latency analysis finds no size and job state with two clean labels to compare (`latency_vs_endpoint_label` prints "only 1
endpoint label seen, no comparison" four times; every other group has no clean label at all): of 123 points, 20 carry a clean memory-clock label, 63 are too short, 32 have
endpoints that differ and 8 are mixed. The memory clock did move during most jobs (for example `e-lat-r1` 456 MHz in 300 samples, 772 in 97, 875 in 20), but that is the cause
of only the 8 mixed points and part of the 32 whose endpoints differ. So E gives no latency comparison across memory-clock labels: it cannot say that L2 or GDDR6 latency does or
does not depend on the memory clock, and the earlier statements to that effect are withdrawn. The 512 MiB random walk reads 796-1037 depending on the job state, with no clean
label in any of its points; not a label effect that this run can show.

**What is open.** No latency point exists at a clean 1124 or 1408 label: latency chases do not raise the memory clock themselves,
and the bandwidth passes run in a separate job. A chase beside a streaming read was tried afterwards (E2, below) and is not interpretable. A minimum point length in `run_E.sh`
(for example 30-60 ms, as `e2_inner.sh` uses) would give every point a window the sampler can judge: later.

## Item D: the core clock at a fixed cadence (`r9_clk`, `vkr9 clk`)

**Method (READ, `analyse_clk.py`).** A dispatch of a planned length (0.3, 0.6, 1.0, 1.7 ms) is submitted every 16.7 ms and waited on;
the clock of a dispatch is (wave 0's cycle counter difference) / (the REALTIME counter difference, 100 MHz ticks) over the whole
dispatch, its first quarter and its last quarter. Cells: HIP (one run per length) and Vulkan graphics (one per length, four for 1.0
ms). A calibration dispatch of 30 ms is checked against the queue's own timestamp timer. The probe also prints its own cadence verdict
(p95 and the largest late submission against a limit of 1000 us); the analysis now reads it.

**Numbers (MEASURED, `analysis.txt`; planned dispatch lengths 0.3-1.7 ms, measured 0.34-1.73 ms).**

| Path | Core clock seen by the shader | 5th-95th percentile spread | Notes |
|---|---|---|---|
| HIP, 0.3 / 0.6 / 1.0 / 1.7 ms | 2961 / 2961 / 2960 / 2960 MHz | 0.5 / 0.4 / 0.2 / 0.2 % | sysfs `gfxclk` 2972, 2971, 2761, 2753 |
| Vulkan graphics, 0.3 / 0.6 / 1.0 / 1.7 ms | 2023 / 2022 / 2022 / 2024 MHz | 0.5 / 0.5 / 0.4 / 0.5 % | sysfs `gfxclk` 2034, 1924, 2034, 2036 |
| Vulkan graphics 1.0 ms, four runs | 2022, 2024, 2023, 2022 MHz | run to run 0.1 % | |

No throttle flag in any sample; hotspot at most 54 C. First and last quarter agree within 0.2 %, no step inside a dispatch.

**Low-clock dispatches (MEASURED, minima of the call records).** Six of the seven Vulkan runs have a minimum near 1180 MHz
(1177-1187); `d-vk-1.0-r2` has 2014. Counting dispatches below 1400 MHz: 50 of 1,197 in `d-vk-0.6-r1` (9 of them among the first 20),
20 in `d-vk-1.0-r3` (calls 442-502), 1 each in `d-vk-0.3-r1`, `d-vk-1.0-r1` and `d-vk-1.0-r4`, 2 in `d-vk-1.7-r1`, none in `d-vk-1.0-r2`.
The percentile spread (0.4-0.5 %) does not show them because they are fewer than 5 % of the dispatches in every run (4.2 % at most).

**Controls (61 PASS, 5 FAIL of 66; the analysis file saved before the analysis fixes printed 62 of 66, and was regenerated).** `rt_shift` (a REALTIME x 1.05 shift of the parsed numbers) is a self-test of the analysis, not
a control of the run, and is no longer counted. All 11 validity controls (1,197 of 1,197 dispatches usable in every cell), 11
calibrations, 11 wave-agreement controls, 11 median-cadence controls, 10 of 11 probe-cadence controls and 7 of 11 length controls pass. Calibration
against the timestamp timer: Vulkan 0.01-0.13 % apart, HIP event timer 0.21-1.21 % apart (limit 1.5 %).

The five FAILs:
- `probe_d-vk-1.0-r3`: the probe's own cadence verdict, one submission late by 1169.4 us against the limit of 1000 (p95 0.2 us against 200).
  The python median-cadence control passes on all cells; the probe verdict is the stricter one.
- `length_*` (four): the median dispatch length (the span between wave 0's start and end REALTIME readings) against the planned
  length: HIP 0.344 against 0.300 and 0.635 against 0.600; Vulkan 0.359 against 0.300 and 0.647 against 0.600. The gate is
  max(3 %, 0.03 ms), so the excess (0.035-0.059 ms) fails at 0.3 and 0.6 ms and passes at 1.0 and 1.7 ms (Vulkan 1.007, 1.726; HIP 1.023, 1.703).
  The planned length does not enter the clock figure (READ, `analyse_clk.py`, `pipeline()`: the cycle count is divided by the
  REALTIME ticks of the same span). INFERRED: the busy loop was sized for a clock the card did not run at, and a fixed start-up and
  finish cost of tens of microseconds adds to short dispatches; quote "0.3 ms dispatches" as "about 0.34-0.36 ms".

**Why the HIP and Vulkan clocks differ.** The 2960 against 2022 MHz is MEASURED (the sysfs readings agree for Vulkan; for HIP at 1.0 and
1.7 ms sysfs is 7 % lower, 2761 and 2753 MHz, probably because its 5 ms samples include idle between dispatches; INFERRED). Why the
two APIs run at different clocks at the same 16.7 ms cadence is not separated (INFERRED candidates: the card's dispatch-load estimate,
a different queue, the shader's power use).

**What is open.** The cause and timing of the low-clock dispatches (they sit in runs, not at fixed places); the idle clock between
dispatches; steady-state behaviour over seconds (G1).

## Item G1: steady clocks with `profile_standard`

**Method (READ: `scripts/round9/g1_inner.sh`, `g1_lib.sh`, `stable_clocks.sh`, `analyse_stable.py`).** One gated job holds the GPU
lock. Inside it the script writes `profile_standard` to `power_dpm_force_performance_level` (through sudo, with the owner's approval;
G2, the other profile switch, is not part of it), runs ten times the item D cell (Vulkan graphics queue, 1.0 ms dispatch, called and
waited on every 16.7 ms, 4 s each, `vkr9 clk`), writes `auto`, and runs the same ten cells again. The clock of a run is the median over
its dispatches (after the first 20) of (wave 0's cycle counter) / (REALTIME ticks) as in item D; steadiness is (max - min) / 2 /
median of the run medians, limit +-0.3 %. A trap restores `auto` on EXIT, INT, TERM, HUP and QUIT; the restore now checks the write
status and reads the level back, and runs right after the gate returns. If the job is killed, that recovery write runs after the GPU lock is released (a known gap, Codex round 2
finding 1, not exercised in these runs: both recorded runs read `auto` after the job); `stable_clocks.sh` therefore refuses a real run without `G1_REDESIGNED=1` until the
recovery runs under the same lock, and G1 is not run again before that. The password path no longer appears in the logged command
line (fixed after the review); the run used here predates that fix, so its first log line holds the path of the password file, which
this note does not quote.

**First run (`runs/20261124T221141Z-r9G1`, stopped after 3 cells; MEASURED, `runs/r9-G1-20261124T221141Z.log`).** Three standard cells
ran (1625, 1625, 1624 MHz, +-0.05 %). Cell 3 printed a failed cadence verdict (the largest late submission was 1061.5 us against a limit of
1000), and the script then left its loop (`exit 1`), so the other 17 cells never started; `auto` was restored by the trap. The level check
read the wrong file and printed an empty level. Both faults are fixed in a later commit.

**Rerun (`runs/20261124T221243Z-r9G1`, log `runs/r9-G1b-20261124T221243Z.log`; MEASURED, analysis rerun on the new code).**

| Level | Cells that ran | Run medians (MHz) | Median | Spread of run medians |
|---|---|---|---|---|
| `profile_standard` | 6 of 10 (runs 1-6) | 1624 1624 1624 1623 1623 1622 | 1623 | +-0.07 % |
| `auto` | 10 of 10 | 2022 2022 2023 2022 2021 2021 2022 2021 2021 2021 | 2022 | +-0.05 % (max - min 0.1 %) |

Standard against auto: ratio 0.803. The two level controls pass (read before and after: `auto`; read inside the job after the write:
`profile_standard`). Every cell that ran has 239 dispatches, 100 % usable. **34 PASS, 5 FAIL, 0 not judged**: the four cells that never
ran (runs 7-10, below) and the probe's own cadence verdict of `std-1` (largest late submission 1526.4 us against 1000; its median
clock, 1624 MHz, is the same as the other runs'). The steadiness verdict for standard passes (+-0.07 % against +-0.3 %); the ten auto runs
have no limit.

**Within a run (MEASURED, calls 21 onward of each run: 219 dispatches).**

| Level | 5th-95th percentile spread within a run | Dispatches below 1400 MHz | Median dispatch length |
|---|---|---|---|
| `profile_standard`, 6 runs | 0.18-0.23 % | 0 | 1.075 ms |
| `auto`, 10 runs | 0.31-0.51 % in 8 runs; 4.74 % in `auto-6`; **41.43 % in `auto-5`** | 13 in `auto-5` (calls 124-238), 11 in `auto-6`, 3 in `auto-10`, 1 each in `auto-7` and `auto-9`, 0 in the other five | 1.007-1.009 ms |

So the run medians were stable under both levels, and several auto runs had low-clock dispatches (near 1180 MHz) that the median hides.
Standard's median dispatch is 1.075 ms against auto's 1.008 for the same planned 1.0 ms.

**Memory clock (MEASURED, `loads/*.watch/samples.csv`, the 5 ms sampler, and `uclk_start`/`uclk_end` of each cell).** Under
`profile_standard` the memory clock was 1124 MHz in every sample of all six runs (1225-1230 samples each) and `uclk_start` and `uclk_end` read 1124.
Under `auto` it was 456 in most samples (all 1225-1226 of them in six runs), with 772 MHz in 316 samples of `auto-1` and 141 of `auto-3`, and
772, 875 and 1124 in 150 samples of `auto-5`. So the two levels differ in the memory clock as well as the core clock, and the
comparison is not core-clock-only.

**The four standard cells with no dispatches (runs 7-10; MEASURED from their `.out` files).** Each of `loads/std-7.out` ... `std-10.out`
holds the same lines and no cell: `START hotspot 58 C`, then `NOT RUN cell: hotspot 58 C (a cell starts below 58 C)`. The refusal is the
start guard of `vkr9 clk` (`src/round9/vkr9.c`, `hotspot() >= 58.0`). The hotspot at the start of the ten standard cells climbed 53, 55, 55,
56, 57, 57, 58, 58, 58, 58 C, so cells 7 to 10 were refused and the script went on; the analysis found no `*-clk-cell.csv` for them and
printed `0 dispatches, 0 % usable`: the four FAILs are the validity control doing its job. The auto cells started at 55, 54, 54, 54,
54, 53, 53, 53, 53, 52 C and all ran.
- Cause (INFERRED from the numbers): under `profile_standard` the card does not drop its clock between cells. The one sample of the
  refused cell `std-7` reads 1635 MHz core, about 82 W reported (about 1.2x too low, regulator mod) with nothing running; under
  auto the card idles down between 4 s cells and the start hotspot fell from 55 to 52 C. The 0.5 s idle between cells is not enough
  to cool a card held at 1635 MHz.
- Consequence: the standard result rests on 6 run medians, not 10, and all six started at 53-57 C, the cooler part of the sequence.
  The clock under standard might read lower in the refused cells (hotter silicon); not measured. The +-0.07 % is over 6 runs and is not
  directly comparable in count with the auto figure over 10.
- Not changed here: the 58 C start limit belongs to every item that uses `vkr9 clk`, and a longer rest between cells (for example 20 s)
  is the lead's choice of cost.

**Against PR 57's Windows measurements.** What the local runs show on their own (MEASURED): 1623 MHz under standard; under auto, run medians within 0.1 %, a within-run
5th-95th percentile spread of 0.31-0.51 % in eight runs, 4.7 % in `auto-6` and 41 % in `auto-5` (13 dispatches below 1400 MHz; minimum 1176.7 against a median of 2020.7). READ
(PR 57's research README, not re-opened for this note): PR 57 reports 1868 MHz for its stable mode on Windows and timing spreads of +-0.3 % and 20-40 %. The 1868 MHz is a clock;
the +-0.3 % and 20-40 % are spreads of repeated DLL timings, not clock spreads. These are different quantities and are not compared. The cell is a 6 % duty, 1 ms call-then-wait
load; a heavier or burstier load may behave differently and this item does not test it. INFERRED: standard pins the clock the card's own table gives for that state, not the
clock PR 57's tool picked; not tested.

**What is open.** A rerun with enough rest between cells (or a cooler start) to get 10 standard medians; a burstier or heavier load
under both levels; the cause of the low-clock dispatches under auto (they are 1-13 of 219 in the affected runs and sit in runs of calls).

## Files changed in this branch for this round

Code fixes after the reviews (the ld_mat ring and address mask; the item E loop and labels; ld_order; the drivers, Vulkan cold chase, stamp
and cadence; the G1 sudo path and restore; two superseded pre-run draft notes removed; the ld_mat start guard) and the earlier ones: `scripts/round9/run_C.sh`, `run_E.sh` (analysis input
path), `ana_ld_order.py` and `expect-ld_order.md` (the `pure_hit` band, old value kept in the text), `ana_ld_mat.py` (note in the `c_back`
text only), `src/round9/ld_mat_k.hip`, `r9_bgread.hip`, `vkr9.c`, `r9_cl2.hip` (the page walk), `r9_lat.hip`, `scripts/round9/` drivers
and analyses. After review round 2 (scripts only, no GPU; a later commit changed the header comment of `src/round9/ld_mat.hip`, with no rebuild and no GPU): `ana_ld_mat.py` rejects a burst with an impossible clock or a min/max far from its median;
`ana_ld_mat_k.py` prints the six controls that compare with the superseded first run as NOT COUNTED; `r9lib.py`, `analyse_hier.py` and `analyse_E2.py` label a window with fewer than
3 sampler rows "too short" and keep too short, endpoints differ, mixed and unchecked apart (an endpoint label without a window is never clean); `analyse_cbar.py`,
`analyse_hier.py`, `analyse_E2.py`, `analyse_clk.py` and `analyse_stable.py` check cell and point names, repetition or call ids and a probe cadence verdict for every
usable CSV (a renamed cell and a CSV without its `.out` now fail; a duplicate repetition id fails in C; each has a deliberate break in the self-test, with the limits listed below); the "PR 57's
swing seen" label is removed from `analyse_clk.py`; `stable_clocks.sh` refuses a real run without `G1_REDESIGNED=1`, `g1_lib.sh` ignores signals while it restores and
`g1_inner.sh` removes the password-file variable from the environment of each cell; the comment of `run_ld_mat.sh` names the 58 C rule.

What the driver scripts enforce: they exit with the analysis status. The drivers supply the expected file slots. C checks cell names and repetition ids (a duplicate repetition id
fails). E and E2 check the point identities and counts inside each accepted job. D and G1 check contiguous observed ids, some header fields and the cadence sidecar. Not checked:
the identity of a whole job (a repeated or substituted job in E, E2, D or G1) and the expected call totals of D and G1; both are deferred to the redesign. Re-judged on the archived
logs: `complete` passes in C, C2, E and E2; D and G1 give 61 of 66 and 34 of 39 as before. A and its load-ratio follow-up and B check that the cells they judge are present, not exact identities. Not changed: the executable code of `src/round9/ld_mat.hip` (only its header comment changed; the calibration launches still have no 58 C check, and the
comment says so) and a minimum point length in `run_E.sh`. No expected value or control verdict of items A, C, D or E was edited; the controls
that failed still fail in the analysis files; the six `ld_mat_k` controls were taken out of the count after the round 2 judgement, not edited.

## Follow-up cells (code and drivers, then run)

Each has its expectation file written first, an analysis self-test with deliberate breaks that must fail, and compiles (build and listing
checks, CPU only). The lead ran them in the second chain (`runs/r9-chain2-20261124T222849Z.log`); elapsed wall times from that log:

| Driver | What it does | Elapsed (chain log) |
|---|---|---|
| `scripts/round9/run_ld_mat_k.sh` | Item A load ratio (`ld_mat_k`): one load group per 1, 4 or 8 matrix instructions at L0 and L2, one wave; variant s (one wait per group) and w (a wait before every instruction); predictions 8.08 + 26/K or 34.1 in `expect-ld_mat_k.md` | 5 s |
| `scripts/round9/run_C2.sh` | Item C with a page-order walk (increasing, next = slot + 1): cold, fused, b2 and the race break on both queues, and the HIP cold, fused, two-launch and race cells; judged with the page-order bands (GDDR6 470-560, memory-side cache 350-400) | 72 s (first run); rerun in chain 3 |
| `scripts/round9/run_E2.sh` | Item E at the 1124 and 1408 labels: single-wave chases at 1 MiB, 32 MiB (random and page) and 1 GiB (random, fresh, page) beside a second process that streams a different 1 GiB buffer with 2, 8 or 32 work-groups (`r9_bgread`, rate printed); each chase labelled by the sampler's readings during it; two idle jobs | 131 s |

Limits of the design, known before the run: C2's page walk repeats the same links every repetition (the stamp added later makes the
check able to see stale data); E2 reaches the 1124 and 1408 labels only if the card's own policy raises the memory clock for a few-block
read; A's loop has 112 matrix instructions, not 114, so that every group size divides it. The memory clock label at the start of every
follow-up job was 456 MHz (`between-jobs.log`, each job's `.meta`).

### Follow-up A: the load ratio (`ld_mat_k`; MEASURED, `ana-ld_mat_k.txt`, run folder `20261124T222849Z-r9-ldmatk`)

**Design (READ, `expect-ld_mat_k.md`).** One wave, L0 and L2. One 256-byte load group per K = 1, 4 or 8 matrix instructions, so the
loads are the same width and there are fewer of them as K grows. Variant s: one `s_wait_loadcnt` in front of the first instruction of
a group. Variant w: a wait in front of every instruction (it never blocks, the group has landed). Cells: `k<K><s|w>_d<lookahead>`.
Three jobs: two repeats and one deliberate break. This design has no address arithmetic between matrix instructions: the reference reads 8.06.

**Controls.** 13 counted controls PASS, 0 failed: matrix only 8.064 (gate 7.9-8.1); the aim break moved 21.92 of 22.10; every cell agrees with its other lookahead within 0.93
cycles (gate 2.0); 153 bursts placed. Six more compare with the first `ld_mat` run, which is superseded (Item A), so they prove nothing now and are printed as NOT COUNTED in
`ana-ld_mat_k.txt`: `k1s_d54` 33.07 at L0 and 33.65 at L2 against that run's 33.60 and 35.70 (gate +-3.0), and the latency cells `k4s_d0` and `k8s_d0` giving 96.4 and 96.5
extra cycles per group at L0 and 273.7 and 273.3 at L2 against its 104.8 and 308.2 (gate 20 %). Against the ld_mat rerun, `b_k1` would fail as written (33.07 against 39.31 at
L0). The verdict does not need them: it rests on ratios inside this probe, the reference at 8.064 in its gate, and the cover, aim and placement controls. The analysis self-test
passed before the first job; the listing check passed on 13 blocks. The result is quoted from the L2 rows, because the L0 one-load-per-instruction cells reach 42-49.5 KiB, past
L0 (footprints below).

**Numbers (MEASURED, cycles per matrix instruction, extra over matrix only 8.06 in brackets).**

| Cell | L0 | L2 |
|---|---|---|
| K = 1, d54 | 33.07 (+25.00) | 33.65 (+25.58) |
| K = 4, s, d24 | 14.33 (+6.26) | 14.41 (+6.34) |
| K = 4, w, d24 | 14.31 (+6.24) | 14.39 (+6.32) |
| K = 8, s, d12 | 11.21 (+3.14) | 11.43 (+3.37) |
| K = 8, w, d12 | 11.19 (+3.13) | 11.39 (+3.32) |

The shorter lookahead of each cell (K = 1 d24, K = 4 d12, K = 8 d6) is within 0.92 cycles of these (largest: `k1s_d24` at L2, 34.57 against 33.65).

**Footprints of the L0 cells (READ, the immediate offsets in the generated header, plus the 8 KiB span).** The loads read
`voff + 256 g` and only `voff` is wrapped into the span, so the footprint is the span plus the load window. Six cells reach beyond
the 8 KiB L0 span: `k1s_d24` (about 42 KiB), `k1s_d54` (49.5), `k4s_d12` and `k4w_d12` (18), `k4s_d24` and `k4w_d24` (21). The two K = 1
cells exceed the 32 KiB L0 (READ, `memory-system.md` line 33), so their L0 rows are partly L2-served; the K = 4 cells fit 32 KiB but
not the 8 KiB label (INFERRED: still L0-resident; not tested). The K = 8 cells (13-14.5 KiB) and the D = 0 cells (11.5 and 15 KiB)
fit. The L2 rows have the right span and stand; L0 and L2 differ by 0.1-0.6 cycles at these cells, and the conclusions below rest on the L2 rows.

**Verdict printed by the analysis, both levels: PER LANDING LOAD.** L0: e1 = 25.0, the extra as a fraction of e1 is 0.25 for K = 4
(s and w) and 0.13 for K = 8 (s and w). L2: e1 = 25.6, the same fractions. A cost per wait or per load in flight would give 1.00 for
the w variants; they gave about 0.25 and 0.13 (L2: 0.247 and 0.130 for w, 0.248 and 0.132 for s; L0 w: 0.250 and 0.125). MEASURED: the extra falls as 1/K for K = 1, 4, 8 at both levels, and it is the same with a wait
before every instruction or only before a group's first.

**What it supports (INFERRED).** On one wave, at these spans and this load width, a matrix loop with fewer loads pays less. Issuing a load earlier covers its latency but, by these
numbers and 8a's, not the roughly 25 cycles. The extra scales with load count in the tested schedules, consistent with a cost per landing load; the number of waits had little
effect. The tested per-wait and constant-per-load-in-flight models are rejected (the w variants gave 0.25 and 0.13 where those models give 1.00). Arrival timing and issue cost were
not measured independently (both variants issue the same loads). The load width was fixed (256 bytes per wave) and wider loads were not tested; `matrix-timing.md` 8a (READ) has
one b128 load at 25.7 and four loads at 32.2, a different design. Two sources agree on the size: this follow-up gives 25.0-25.6 with one load per instruction (MEASURED) and
`matrix-timing.md` 8a reports 26 (READ); the `ld_mat` rerun's 39.31-39.36 against 15.12 is reported and not counted.

**What is open.** Memory-side cache and GDDR6 were not run in this follow-up, and one wave only; the whole card is not repeated.

### Follow-up C2: the page-order walk (`vkr9 cbar`, `r9_cl2`; MEASURED, run folder `20261124T234234Z-r9C2`, the rerun)

**Design (READ, `expect.md`).** Item C again with the walk in page order (increasing, next = slot + 1, 1,024 steps = 128 KiB of the 4
MiB region), 20 repetitions, 3 process repeats, Vulkan graphics and compute queue (b2 barrier) and HIP. Judged with page-order bands:
GDDR6 470-560, memory-side cache 350-400 (READ, `memory-system.md`). The first C2 run (`20261124T222854Z-r9C2`) is superseded (arithmetic
Vulkan cold chase; graphics b2 435 a low-clock artifact, see item C).

**Numbers (MEASURED, cycles per step).**

| Cell | HIP | Vulkan compute | Vulkan graphics |
|---|---|---|---|
| cold (evicted region, loaded links) | 488 (run medians 489 488 488) | 799 (799 799 806) | 802 (794 831 802) |
| fused (same kernel writes and chases) | 231 (231 235 231) | 294 (294 294 295) | 295 (295 294 302) |
| two launches (producer launch, then consumer launch) | 489 (483 497 489) | - | - |
| b2, producer then consumer | - | 535 | 533 |

Stamp checks: b2 passes in 57 of 57 repetitions on both Vulkan queues (stamp sum and final slot); the race break fails in 57 of 57 on both queues and, by its final address, on
HIP. Every Vulkan cell of C and C2 shows the memory-clock ramp of its start (456 MHz at the start, 1124 or 1408 at the end): for example the first ten repetitions of the graphics
b2 cell of run 1 read 552-681 cycles per step, the last ten 515-533, at a steady 3.23 GHz, and the cold cells read 1006-1104 in repetitions 1-4 and 764-780 in repetitions 12-19.
The cell medians are quoted, with the repetitions 12-19 beside them where the ramp matters. Controls: 20 PASS, 3 FAIL.

**Controls judged.**
- HIP cold 488 and HIP two-launch 489 both pass, in the page-order GDDR6 band (470-560). The two-launch control passes in either band and
  the band was the finding. HIP's two-launch 489 is consistent with its 488-cycle evicted reference.
- `cold_vulkan/compute` 799 and `cold_vulkan/graphics` 802 **FAIL** the band 470-560. The probe now follows loaded links as HIP does, and
  the Vulkan cold cell still sits 311-314 cycles above HIP's as cell medians (276-292 in repetitions 12-19). The cause is open; untested candidates are the Vulkan buffer's
  placement and its address translation entries.
- `clock_hip` FAIL: the HIP fused cells' clock moves by more than 10 % in the cell (run 1 2494-3228 MHz, run 2 2522-3228, run 3
  2492-3229, median 3225). The Vulkan clock checks pass.
- fused, race, b2 pointer and all repeats pass.

**What it shows (observations only).**
- MEASURED: HIP cold 488, HIP two-launch 489. Vulkan cold 799-802 as cell medians (764-780 in repetitions 12-19); b2 533-535 (520-524 in repetitions 12-19), 264-269 below the
  Vulkan cold medians (compute 799 - 535, graphics 802 - 533; the first version said 240). HIP's two-launch 489 is consistent with its 488-cycle evicted reference, as in random
  order (697 against 700); the HIP check is of the final address only. No cache level, no write-back and no statement about where the written data was is concluded: this
  differs from `memory-system.md` 2c, which measured data a previous kernel *read*, and one probe that does both in one program was not run.
- The Vulkan cold cell cannot serve as the matched evicted reference until its gap to HIP cold is explained.

### Follow-up E2: latency at the 1124 and 1408 labels (`r9_lat`, `r9_bgread`; MEASURED, run folder `20261124T223006Z-r9E2`; analysis rerun on the new code)

**Not interpretable.** Three of the four controls that say "the background is light" fail (the background rate, the L2 chase band and its 5 % of idle, the 32 MiB page chase
below 470); the fourth, the 1 GiB page chase equal at two background rates, passes. The run, the table and the control results are kept; no conclusion is drawn.

**Design (READ, `expect.md`).** A single-wave chase at 1 MiB, 32 MiB and 1 GiB in a second process beside a background read of another
1 GiB buffer with 2, 8 or 32 work-groups, two repeats of each, and two idle jobs. Each chase is labelled by the memory-clock readings of the
sampler during it. Four controls say the background is light (at most 70 GB/s; L2 1 MiB within 215-270 and 5 % of idle; the 32 MiB page
chase below 470; the 1 GiB page chase equal at two background rates).

**Numbers (MEASURED).** Background rates: 2 blocks 30.4 and 30.2, 8 blocks 140.7 and 139.4, 32 blocks 546.3 and 554.4 GB/s. 56 of 56
chase points have the right pointer. Clean-label chases: 2 at 1124, 28 at 1408. Median cycles by size and walk:

| Size, walk | idle, 456 (background 0) | background 2 blocks (30 GB/s), label 875 | background 8 and 32 blocks (140-554 GB/s), label 1408 |
|---|---|---|---|
| 1 MiB, random (L2) | 240.1 | 427.2 | 668.2 |
| 1 MiB, page | 242.6 | 360.4 (875), 357.7 (1124) | 481.9 |
| 32 MiB, page | 360.9 | 529.7 (mixed label) | 482.0 |
| 32 MiB, random | 433.9 (n = 1) | 824.4 (mixed label; pools the idle and 2-block jobs, n = 3) | 724.5 |
| 1 GiB, page | - | 604.4 (mixed label; pools the idle and 2-block jobs, n = 4) | 478.6 |
| 1 GiB, random fresh | 473.5 | 853.0 (875), 785.0 (1124) | 725.8 |
| 1 GiB, random | - | 843.0 (875, n = 1); 1037.2 mixed (pools idle and 2-block, n = 3) | 722.1 |

**Controls (5 PASS, 4 FAIL).** Pass: `complete` (8 of 8 jobs in the expected file slots, each with its 7 chases; a job's identity is not checked), `pointers`, a clean 1124 label, a clean 1408 label, and the 1 GiB
page chase equal at 139.4 and 554.4 GB/s (479.6 against 477.5, ratio 0.996). That pair is one of four 1408-labelled 1 GiB page results
(the others are 470.1 and 494.8). Fail:
- `bg_light` FAIL: the 8-block and 32-block jobs read 140 and 550 GB/s (limit 70). Only the 2-block job (30 GB/s) was within it.
- `l2_band` FAIL: the 1 MiB L2 chase is in 215-270 in 2 of 8 chases (the idle ones); outside are 428 and 426 (2 blocks), 629 and 624 (8 blocks) and 707,
  708 (32 blocks).
- `l2_unchanged` FAIL: the L2 1 MiB median is 626.7 beside the background against 240.1 idle (161 %; limit 5 %). Beside the lightest background (30 GB/s) it reads 428.4 and 426.1 (about 427). The memory clock and the core clock also differ between the idle and background jobs, so no cause is separated.
- `mall_level` FAIL: the 32 MiB page chase is at or above 470 in 6 of 8 chases (471.5 to 539.1) and 355.0 and 366.8 only idle. No cache residency is concluded: the
  timings alone cannot separate contention on the path from where the lines were.

**Reading (MEASURED only).** Observed timings: the 1 MiB random chase reads 240 cycles (74-76 ns) idle at the 456 label, about 427 cycles (119-120 ns) beside the 2-block background
read (30 GB/s, label 875) and a median of 668 cycles beside the 8- and 32-block reads (140-554 GB/s, label 1408). The memory clock, the core clock (about 3.2 GHz idle, about
3.6 beside the 2-block read) and the background traffic all differ between these jobs, so no cause is separated: not the memory clock, not the traffic, not cache residency. The
32 MiB page chases show the same kind of shift (355.0 and 366.8 idle; 471.5-539.1 with a background). The `E2 latency_vs_memory_clock` lines (ratios 1.17 to 2.78; the analysis
says "varies across the observed labels") are associations only: the background rate and the label change together in these jobs. The 1 GiB page equality (0.996 between 139.4
and 554.4 GB/s at the 1408 label) says only that the two chases read 479.6 and 477.5 cycles under these two joint conditions (background rate and label both differ), within the control band; the other
1408-labelled 1 GiB page results at these background sizes read 470.1 and 494.8, and no idle run carries that label, so no response to the background rate is read from it. Clock-only latency remains unmeasured, and there is no clean latency at the 1124 or 1408 label without a background. The statement of the first
version that the L2 chase "should not move with the memory clock" was withdrawn in item E and is not used; neither is its inference from a failed band that the lines left the
memory-side cache. The run does contain clean sampled labels (2 chases at 1124, 28 at 1408) beside a running background with different core rates.

**What a clean design would need.** The memory clock held high without traffic on the measured path, for example by setting the driver's power
level (not approved for this item), so the chase runs alone at the raised label. A background stream cannot do it here: the lightest one that holds the
clock above 456 MHz was read with different L2 and 32 MiB chase timings from the idle job, with the core clock also different.

## What this round does not show

- Anything about FP8 dot paths or the LLM runtimes; those are in earlier files.
- A value for load order with different load widths, stores or more than five loads.
- A cost per load beside matrix work on the whole card, a lookahead a stream needs, or anything about GDDR6 against matrix work (item A did not pass its own reference checks).
- Latency at a clean 1124 or 1408 label, or any statement that latency does or does not depend on the memory clock (item E: of 123 points 63 are too short for a label, 32 have
  endpoints that differ and 8 are mixed; the E2 run is not interpretable).
- Whether the data one dispatch wrote stays in L2 for the next: the cells of C and C2 are observations and name no cache level.
- Why Vulkan cold reads slower than HIP cold with the same walk (item C and C2), and whether any Vulkan barrier kind gives a memory dependency
  beyond the tested runs.
- Memory-side cache or GDDR6 rows of the load ratio (follow-up A ran L0 and L2 only), and D between 8 and 60 at those spans.
- Anything about the standard level under a heavier or burstier load than the 6 % duty cell, or 10 standard medians (G1 has 6), or a
  comparison with PR 57's Windows timing spreads.

## Review record and follow-ups (2026-11-24)

The write-up and code were reviewed in three rounds by GPT-6.2 Sol (Codex) and Gemini 4 Neon, read-only. Between rounds 2
and 3, at the owner's request, Fable 5.2 judged which claims to keep (keep, keep with change, not interpretable, drop). The
blocking findings of round 3 were applied and checked by the lead against the reviewers' evidence; no fourth review round
was run (the owner's rules allow three). These findings were left as follow-ups for the next probe design:
- `scripts/round9/stable_clocks.sh` is blocked (it refuses to run without `G1_REDESIGNED=1`): its last-chance restore runs
  outside the shared GPU lock and can write the level when this run never changed it. Redesign before any further use.
- `ld_mat`: 64-bit or cross-checked cycle counters (one burst read an impossible 58 GHz and was rejected; cause open),
  the 58 C check also on the calibration launches, and address arithmetic kept out of the timed stream.
- `vkr9`: warm the memory clock or discard ramp repetitions before timing C/C2 cells (every Vulkan cell starts at 456 MHz).
- `run_E.sh`: a minimum point length so short chases get at least 3 sampler readings.
- `analyse_hier.py`: handle a missing `t1_ns`, and separate a dominant-but-disagreeing window from variation within it.
- Run logs that once held the password file's path were redacted; the file was deleted on 2026-11-24.
