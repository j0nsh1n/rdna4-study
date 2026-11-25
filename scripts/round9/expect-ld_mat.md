# Expectations for ld_mat (round 9, item A), written before any GPU job

Source: `DRAFT-r9-plan.md` section A, `measurements/matrix-timing.md` 8a and 9, `measurements/memory-system.md`
sections 2b, 5, 6, `measurements/round7.md` A2. All numbers INFERRED; nothing has been measured with this probe.
Cycles are per matrix instruction (`v_wmma_f32_16x16x16_fp8_fp8`, 8.0 cycles at the matrix unit's own rate), per SIMD.

## Question

In a stream of FP8 matrix instructions fed by global loads, what does the stream cost per matrix instruction when each
load is issued D matrix instructions ahead of its use (D = 0, 1, 2, 4, 8, "all upfront"), by memory level (L0, L2,
memory-side cache, GDDR6), by registers per wave (51, 90, 167, 209: 16, 16, 9, 7 waves per SIMD)?

## Design facts that differ from the draft's wording

- One load per matrix instruction (a "group" is one instruction), so the stream asks for 256 bytes per 8 cycles per wave:
  on the whole card most cells are limited by cache or memory bandwidth, not by latency. The one-wave cells show latency.
- "All upfront" is the largest lookahead the registers allow (and at most 60, the load counter holds 63): D = 10 at 51
  registers, 30 at 90, 60 at 167 and 209. Loading all 114 loads of an iteration at once is impossible (228 registers; 63
  loads in flight). At D = 60 the cover is 480 cycles, not enough for GDDR6 latency.
- After the first review (2026-11-24) the streams changed shape, not the expectations below: the loop body is 114 rounded up to
  a multiple of the register ring (114, 114, 116, 114, 120, 120, 128, 124 matrix instructions for D = 0 ... 60), because with a
  ring that did not divide 114 the load for position 114 overwrote a pair that a later matrix instruction still read (the
  listing check now simulates this); every load takes a masked address `(voff + 256 g) & mask`, so each level's footprint stays
  inside its span (the L0 cell had reached 36-51 KiB of an 8 KiB level). The matrix-only reference carries the same address
  arithmetic, so control (a) now also proves it costs nothing. The numbers below were written before any run and are unchanged.
- The work-groups carry `64 x W` threads (1024, 1024, 576, 448), one group per CU, so the waves per SIMD are exactly
  16, 16, 9, 7. The brief's 256-thread groups cannot give 9 or 7 waves per SIMD on every CU.

## Numbers expected

| Cell | Cycles per matrix instruction | Why |
|---|---|---|
| matrix only, one wave | 8.0 (7.9-8.1) | `round7.md` A1, `matrix-timing.md` 9 |
| matrix only, whole card | 8.03 (7.9-8.3) | `round7.md` A2: 8.034 at W = 8 |
| one wave, L0, D = 0 | 8 + extra, extra = hit cost + 18 | wait charge of 8a: 18 cycles for the matrix unit; with the 77-cycle hit of 8a (no saddr form) 95, with the saddr-form hit of `mem_inflight` (about 130) 148; `ld_order`'s `pure_hit` of the same session decides (`--hit`) |
| one wave, L2, D = 0 | extra about 250-330 | L2 latency 230-255 plus the wake charge |
| one wave, memory-side cache, D = 0 | extra about 400-520 | |
| one wave, GDDR6, D = 0 | extra about 500-800 | |
| one wave, D | extra(D) = max(0, extra(0) - 8 D) | 8 cycles of matrix work between issue and use per step of D |
| one wave, L0, D = all (60) | 8.0 within 1 % | data is back before use, no wait blocks |
| whole card, 16 or 9 waves per SIMD, L0 | close to 8 + extra0 / W up to the point where the matrix unit is saturated, then 8.03 | other waves fill the matrix unit while one waits |
| whole card, L2 and further | bandwidth limit: 256 B per instruction per wave; L2 about 7.4-9.5 TB/s, memory-side cache about 2 TB/s, GDDR6 about 640 GB/s over 128 SIMDs | the cost per instruction cannot go below bytes moved / bandwidth: (128 SIMDs x W x 256 B x clock / cost) |
| gain from D = 0 to D = 4 | largest at 7 to 9 waves and L0 or L2; near zero at 16 waves | hypothesis (the draft's, to test, not a number) |

## Controls (a to e) and what they prove

| Control | Pass rule | Proves |
|---|---|---|
| a_ref | one wave 7.9-8.1; whole card 7.9-8.3 | the stream without loads costs the matrix unit's own rate: the loop, the address arithmetic and the wmma chain add nothing |
| b_8a | one wave, L0, D = 0: extra = `ld_order` pure_hit + 18, within 15 % (without `--hit`: near 95 or near 148) | the probe measures the same load wait and wake charge as 8a |
| c_back | one wave, L0, D = 60: within 1 % of (a) | when the data is back before use, the probe adds nothing: the streams differ only by the loads and waits |
| d_break | loads aimed one level lower than the label (L2 label, L0 span): the D = 0 cost drops by at least 70 % of the L2 - L0 difference | the buffer span really picks the level the cost follows |
| placed | every whole-card burst: 128 SIMDs with W waves each | the waves per SIMD are the ones claimed |
| e (listing) | `build_round9.sh`: opcodes, order, registers, `.wavefront_size` 32, `.vgpr_count` = padding, `.max_flat_workgroup_size` = group size, group fits the register file; wrong-header breaks must fail | the code object holds the streams as written and the padded kernels are launchable |

If (a) or (c) fails the probe is not trustworthy: report, do not interpret the cells. If (b) fails, the L0 D = 0 cell is
not the 8a case (check the load form): report both numbers.
