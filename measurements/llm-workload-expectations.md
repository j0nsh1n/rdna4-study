# LLM data dive: expectations, written before any GPU run

Written 2026-11-18 (file time below), before the first GPU job of the dive (the script's selftest and the setup record
use no GPU). Nothing in this file is changed after the runs; what the runs showed goes into `llm-workload.md`, each line
marked right or wrong. Labels: INFERRED means reasoning from earlier measurements in this study, READ means from the
llama.cpp source. Reference setup: llama.cpp master f477c1b90 built with ROCm 10.1 (unmodified build, and the patched
build that adds a FP8 decode kernel), LACT profile "Testing" (stock core clock, 317 W cap), memory clock expected to
read 1408 MHz (held by the displays; 1258 MHz if they blank). Earlier llama.cpp numbers on this card (the
"known answers") were taken at 1258 MHz memory, so memory-bound rows should be about 9 to 12 % faster now.

Models: Llama 3.2 3B and Llama 3.1 8B as FP8 (E4M3 weights, Q8_0 output layer), as Q8_0, and as "auto" (FP8 everywhere,
used only for the quality runs).

## A. Where the FP8 prompt time goes

A1. Per-kernel time of one pp512 pass, 3B. FP8 takes about 56.6 ms and Q8_0 about 41.1 ms, a gap of about 15.5 ms
(from 9051 and 12123 tokens/s, 1258 MHz). Widening the 2.8 G weights (read 1 byte, write 2 bytes: 8.4 GB) takes about
13 ms at 650 GB/s, so the widen kernels alone explain most of the gap. If the widen kernels sum to under 6 ms, this
story is wrong and the FP16 GEMM against the int8 MMQ kernel is the cause. If both matter, the split is reported. The
widen cost is per call, not per token, so FP8's loss shrinks with prompt size (pp128 against pp2048) and is largest at
pp32. (INFERRED, and READ from `ggml-cuda.cu` and `mmq.cu`: FP8 cannot use MMQ; the NVIDIA-only FP8 path is not
compiled in for AMD; the route is widen to FP16 into a pool buffer, then one hipBLAS FP16 GEMM per matmul.)

A2. GEMM rate llama.cpp really gets, per shape (3B: K = 3072, N = 3072, 1024, 8192; 8B: K = 4096, N up to 14336). The
FP16 GEMMs reach well below the 123 TFLOPS of PyTorch's FP16 and the 170 TFLOPS of hipBLASLt FP8, in the range from
the 37 TFLOPS of the 4096-cubed probe up to about 80, and none of the three routes comes near the matrix peaks (211
TFLOPS FP16, 420 TOPS 8-bit integer). The Q8_0 pass of 41 ms for about 2.9 TFLOP means at most 70 TOPS averaged over
the whole pass, MMQ itself a modest share of 420. (INFERRED)

A3. `test-backend-ops perf` MUL_MAT at n = 512: time(f8_e4m3) = time(f16) + widen time. The f16 row is what the hipBLAS
FP16 route costs once weights are already FP16; f8 minus f16 should match the widen time from A1 within 25 %. (INFERRED)

A4. A hipBLASLt-FP8-class GEMM (170 TFLOPS) on the measured GEMM FLOPs would take a small fraction of the measured GEMM
time; this is arithmetic only and says how much of the gap is the routine and how much the format. (INFERRED)

A5. Q8_0 forced through dequantize + hipBLAS (compile switch, no kernel change) lands near FP8's prompt speed, within
15 %, because both then pay the widen to FP16 and the FP16 GEMM. (INFERRED)

## B. More numbers

B1. Q8_0 generation and patched-FP8 generation both reach 85 to 95 % of the bus at the logged memory clock
(tokens/s x file size / bus; 3B Q8_0 earlier: 134.7 t/s x 3.84 GB = 517 GB/s, 82 % at 1258 MHz). At 1408 MHz expect
+9 to +11 % in generation (the study's copy and read gain), not +12 %. Unmodified FP8 generation is about half of
patched (76.8 against 146.0 earlier). Prompt: FP8 loses most at small pp (widen per call), least at 2048 and above.

B2. Parallel streams 1 to 8: S_TG grows nearly linearly while weight bytes dominate, and bends where the generic-HIP
decode kernel's column loop starts to cost (decode prototype: 96 % of the bus at 1 column, 70 to 78 % at 4 to 8).
The Q8_0 against FP8 crossover in streams is the interesting number; the earlier `llama-more` run (1258 MHz) is the
known answer: 3B Q8_0 98.5 / 186.6 / 306.9 / 461.2 / 563.3 / 651.2 / 717.8 / 653.5 t/s for 1 to 8 streams, 3B patched
FP8 145.2 to 785.5, 8B Q8_0 65.4 to 427.1, 8B patched 71.2 to 409.7.

B3. Quality. Known answers at 4 chunks: 3B FP8 9.7049 (default batch), 9.7042 (batch 8), patched 9.7037 and 9.7041,
Q8_0 9.7140; no NaN. KL divergence against the unmodified FP8 default-batch run (hipBLAS FP16 on exactly widened
weights): (a) the same run again gives exactly 0 (determinism, not tested for llama.cpp so far); (b) FP8 with batch 8
(other kernel, other summation order) and (c) patched FP8 are tiny, mean KLD below 1e-4; (d) Q8_0 is larger, about
1e-3, and the only one whose error comes from rounding activations. If (b) is not far below (d), reconsider. A KLD
against a true F16 reference is not possible for the 8B (no file, would not fit); for the 3B it is possible if the
F16 file is made (open item).

B4. Context depth: generation at depth 512 and 4096 declines in proportion to the extra KV-cache bytes read.

## C. How the card behaves while llama.cpp runs

C1. Prompts: core clock 3.2 to 3.3 GHz (the study's heavy matrix loads: 3.18 to 3.32), power-limited at 317 W only for
the sustained FP8-class math. Generation: clock at or above 3.3 GHz, reported power far below the matrix loads (3B Q8_0
at 517 GB/s is about 135 W reported, from 260 pJ per byte). Memory clock exactly the logged current_uclk. Hotspot
below 80 C in 20 s loads.

C2. Bandwidth reached in generation (model bytes x tokens/s, plus KV bytes at depth) against the probe bus numbers at
the SAME logged memory clock (633 GB/s at 1258 MHz, about 690 to 705 at 1408). From per-kernel times of a 128-token
generation trace, the big matmul kernels reach 90 % or more of the bus, the small kernels (norm, rope, copies) far
below, and the gaps between kernels account for the rest.

C3. Launch gaps in the trace: the study predicts 2.1 us per queued empty launch, 3.4 us for a tiny kernel, 17 to 22 us
for launch plus sync; inside a HIP graph probably less. About 350 kernels per 3B token x 3 us = 1 ms of 7.4 ms
(about 14 %). Gaps of 1 to 4 us inside a graph; if the trace shows over 8 us, the idle gap, the clock or the profiler's
own cost is the explanation to test (the overhead control: profiled against unprofiled).

C4. Clock ramp after idle: no visible effect on tg128 (about 0.9 s on the 3B; the ramp finishes in 15 ms); a few percent
on the first pp512 after 2 s of idle (41 to 57 ms long) against the tenth in a loop.

C5. The displays keep the memory clock at 1408 MHz throughout. Any job that logs 1258 is quoted separately and rows
at different memory clocks are never mixed.

## D. Profiler and tool questions (decided by step 2, no number expected)

D1. `rocprofv3 --kernel-trace` shows the kernels inside HIP graphs, or only the graph launches. I do not know, and lean
(55 %) that it shows them (the runtime-trace options exist for it). D2. Hardware counters (`--pmc`) work on gfx1201 in
ROCm 10.1: lean no (60 %). D3. The trace's kernel duration of the study's 1 GiB read probe equals the probe's own timing
within 5 %. (INFERRED from the 1.7 ms duration; short kernels may show a larger relative gap.)
