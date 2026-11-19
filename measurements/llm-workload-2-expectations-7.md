# LLM data dive, round 2, items 7a and 7b: expectations, written before their GPU jobs

Written 2026-11-18 after the lead made item 7 two full items (7a streams, 7b matrix shapes); same rules as
`llm-workload-2-expectations.md`: not edited afterwards. Labels INFERRED/READ as there. Memory clock expected 1408 MHz.

## 7a. Streams 1 to 8, six interleaved repetitions per cell

Round 1 (one run per cell) said: 3B patched FP8 leads Q8_0 at 1, 2 and 8 streams and trails from 3 to 7; 8B patched FP8 leads
only at 1 stream; 3B Q8_0 flattens from 7 to 8 streams (+4.9 %).
E7a1. The spread (min to max over six repetitions) of a cell is 1 to 4 % of its median; cells above 6 % are the exception.
E7a2. The 3B shape holds: patched leads at 1 and 2 streams and trails Q8_0 somewhere in 3 to 7 (60 %). The lead at 8
streams (+1.9 % in round 1) does not hold as a statement: it is inside the spread, so the median difference will be
within +-3 % of zero (60 %). The 8B statement (patched leads only at 1) holds (70 %).
E7a3. The 3B Q8_0 flattening from 7 to 8 streams (+4.9 %) is real: its median ratio stays below 1.06 (55 %).
Could fail: a noisy cell (another program using the card) would widen a spread; cells with a spread above 6 % are flagged, not dropped.

## 7b. test-backend-ops MUL_MAT at the real 3B and 8B weight shapes (n = 1, 8, 32, 512, 2048; f8_e4m3, f16, q8_0)

The widen kernel reads 1 byte and writes 2 per element (3 m k bytes). Round 1 had one shape (m 4096, k 14336) and found
f8 - f16 = 240.6 us, 732 GB/s.
E7b1. For n >= 32 the difference f8 - f16 behaves as a widen plus a scale multiply: the implied rate 3 m k / (f8 - f16) is
650 to 800 GB/s for the big matrices (m k above 10 million elements: gate, up, down, output) and lower for the small ones (k and v:
1024 x 3072 and 1024 x 4096 are 3 to 4 MB of FP8; launch and tail effects, I expect under 500 GB/s) (65 %). The implied
rate does not depend on n (it is a per-call cost): the same within 10 % for n = 32, 512, 2048 (70 %).
E7b2. f16 rate (2 m n k / time) grows with n and with m k: above 100 TFLOPS for n = 512 and 2048 at the big shapes, below 40 TFLOPS at n = 32.
E7b3. At n = 1 and 8 (decode kernels) Q8_0 is faster than the unmodified f8_e4m3, which is slower than f16 or equal (the generic FP8
decode kernel is the slow one in this build): f8 / q8_0 time ratio 2 to 5 at the big shapes.
E7b4. A3 (f8 = f16 + widen) holds for every shape with n >= 32 within 25 % of the traced widen time; it fails for the small k and v matrices.
Could fail: a shape where a different hipBLASLt kernel is chosen for f8 than for f16 (the test builds different graphs).
