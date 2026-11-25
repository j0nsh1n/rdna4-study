# Expectations for ld_mat_k (round 9, item A, load-ratio variant), written before any GPU job of this variant

Source: `measurements/round9.md` item A (MEASURED: one wave, 209 registers, 114 matrix instructions per iteration, one load per
instruction, D = 60: L0 33.62, L2 35.74 cycles per matrix instruction; matrix only 8.08; D = 0: L0 112.91, L2 316.33) and
`measurements/matrix-timing.md` 8a (+26.0 for load data landing during a matrix chain). Nothing has been measured with this probe.
Cycles are per matrix instruction (`v_wmma_f32_16x16x16_fp8_fp8`, 8.08 at the matrix unit's own rate), one wave, L0 and L2.

## Question

The 26-cycle charge of load data that lands while matrix instructions run: is it paid per landing load, or per matrix
instruction that has a wait in front of it, or whenever loads are in flight? One load group per G = 1, 4, 8 matrix instructions.
Variant s: the wait only in front of the first instruction of a group. Variant w: the same wait, same count, in front of every
instruction (it never blocks: the group has landed). Lookahead D in load groups, large enough to cover the latency (D x G x ~11
instruction-cycles is at least 500 cycles; the two D of one cell must agree, control e).

## Predictions (e1 = cost of k1s_d54 minus matrix only: about 25.5 at L0 and 27.7 at L2 by the earlier cell)

| Hypothesis | k1 (G = 1) | G = 4, s | G = 4, w | G = 8, s | G = 8, w |
|---|---|---|---|---|---|
| per landing load | 8.08 + 26 = 34.1 | 8.08 + 26/4 = 14.6 | 14.6 | 8.08 + 26/8 = 11.3 | 11.3 |
| per instruction with a wait in front | 34.1 | 14.6 (one wait per group) | 8.08 + 26 = 34.1 | 11.3 | 34.1 |
| whenever loads are in flight | 34.1 | 34.1 | 34.1 | 34.1 | 34.1 |

The cells with D = 0 (k4s_d0, k8s_d0): the group costs the same extra as one ld_mat D = 0 position (L0 104.8, L2 308.2 cycles
per group: the latency plus the charge), so per instruction ref + 104.8 / G (L0: G = 4 34.3, G = 8 21.2; L2: G = 4 85.1, G = 8 46.6).

No hypothesis is preferred before the run. The analysis names the world (per landing / per wait / in flight) from the fractions
(cost - ref) / e1 of the larger-D cells: per landing is 1/G for both variants; per wait is 1/G for s and 1 for w; in flight is 1 for
all. When the numbers sit between worlds, the verdict says "NO WORLD FITS" and the fractions are reported without a story.

## Controls (analysis `ana_ld_mat_k.py`, self-test with breaks runs before the first job)

| Control | Pass rule |
|---|---|
| a_ref | matrix only, one wave: 7.9-8.1 |
| b_k1 | k1s_d54 reproduces the ld_mat D = 60 cell: L0 33.6 +- 3, L2 35.7 +- 3 (one load per instruction; the loop here has 112 instructions, not 114) |
| c_latency | k4s_d0 and k8s_d0: extra per group (cost - ref) x G within 20 % of L0 104.8, L2 308.2 |
| d_break | k8s_d0 labelled L2 with loads aimed at L0: the cost moves at least 70 % of the L2 - L0 difference |
| e_cover | the two D of one (G, variant): within 2.0 cycles at each level |
| placed | every burst: its one wave placed |
| listing | `build_round9.sh ld_mat_k`: every code object holds the streams as written; deliberate breaks fail |

If a, b or e fails the verdict is not trusted: report the numbers, do not interpret them.

## Known limits of the design (for the reviewer)

- The footprint of one loop iteration differs by G (28.7 KB at G = 1, 7.2 KB at G = 4, 3.6 KB at G = 8 of 256-byte loads); at L0
  the span is 8 KiB plus the iteration's window, so G = 1 may see more L0 misses than G = 8. Control b says whether G = 1 sits where
  ld_mat's D = 60 cell sat; the G = 4 and 8 cells at L0 cannot be slower from misses than G = 1.
- Per-load bytes are the same (256 per wave), so bandwidth per instruction falls with G; with one wave the stream is latency-
  and charge-limited, not bandwidth-limited.
- One wave only. The whole-card effect was seen in ld_mat and is not repeated.
