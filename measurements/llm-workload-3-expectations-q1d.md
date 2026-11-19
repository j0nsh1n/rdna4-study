# LLM data dive, round 3, Q1 follow-up: random data in the power probe (expectations, written before its GPU jobs)

Added after the first Q1c results (committed results are in `llm-workload-3.md`; this file is committed before any random-data job and not edited afterwards).
What the first runs showed (run `runs/20261118T175910Z-llm3`, files `r3m-first/` and `r3m-table.txt`): at matched bandwidth the loop-only mixes differ by +13 W (fp8 minus q8),
the row-structured ones by +33 W (fp8row 287.5 W, q8row 255 W, both at about 672 GB/s); the real 8B decode kernels differ by about 57 W (316 W at the power cap against 248 to 260 W). Those probe
runs use constant data (every weight 1.0, every activation 1.0), so the ALUs see almost no bit toggling. The power of an ALU depends on how many bits change, so constant data
under-reports it, and it does so more for a kernel that does many bit operations on every word (the FP8 widen) than for one dot product per four weights.

Expectation (INFERRED, written now): with random valid FP8 codes, random int8 and random F16 scales in [1, 2), and small random integer activations:
- fp8row rises by 15 to 40 W, q8row by 5 to 15 W, read by 0 to 10 W (random bytes on the bus); the difference fp8row minus q8row grows from +33 W to +40 to +55 W
  (so 70 to 95 % of the 57 W would be explained), probability 55 %. It stays within +-5 W of +33 W (random data does not matter) with probability 30 %; it falls (15 %).
- Bandwidth stays matched within 2 % (random data does not change the memory rate).
Could fail: the sampler's power reading is averaged by the card, so differences under 3 W are not trusted; the card reaches the power limit (flag PPT0) in the fp8 variant, then the core clock falls and
the watts understate the demand (the table prints the power-limit share and the clock); the CPU known answer (256 sampled rows) fails, which rejects the run.
Not covered: the real kernels' extra paths (the fused gate and up path, bias), which this probe does not have. The remainder after this test is INFERRED to be those.
