# LLM data dive, round 2: expectations, written before the GPU jobs of each item

Written 2026-11-18 (file time of the commit), before any GPU job of round 2. Not edited afterwards; what the runs showed
goes into `llm-workload-2.md`, each line marked right or wrong. Labels: INFERRED (reasoning from round 1 or earlier
study results), READ (from source, a listing or a file). Reference setup as round 1: llama.cpp master f477c1b90 on
ROCm 10.1, unmodified and patched builds, LACT profile "Testing", memory clock expected 1408 MHz (held by the displays).

## 1. Exact-weights quality reference

Facts READ before the runs: the 8B checkpoint stores one BF16 scale per tensor (8 significant bits); the 3B GGUF stores
one F32 scale per tensor that also has 8 significant bits (so it was BF16 in the original). An E4M3 value has 4
significant bits, so a product has up to 12 significant bits: F16 (11) and BF16 (8) cannot hold every product, F32 (24)
can. On one 3B tensor (`blk.0.ffn_down.weight`) a check in float64 found 100 % of the products exact in F32 and 53 % in F16.
The 3B reference is built from `llama32-3b-auto.gguf`, not `-fp8-outq8`: its output and embedding tensors are BF16 (the
original values); `-outq8` stores the output tensor as Q8_0, which has already lost information, so a reference built
from it would not be the original model. (Deviation from the task text, stated here.)

E1a. Exactness: F16 is not exact for the 3B and 8B (expect 30 to 70 % of products exact in F16 on every tensor, never
100 %); F32 is exact for 100 % of products in every tensor. Could fail if a scale had more than 8 bits (then F32 is
not exact either; the check reports it).
E1b. Sample rows compared bit for bit against the FP8 file times the scale (in float64 rounded to F32 equals the F32
reference): all equal. A deliberately damaged reference row must be caught.
E1c. Perplexity at 50 chunks of 512: reference within 0.003 of the FP8 unmodified run (9.70 for the 3B at 4 chunks in
round 1; the 50-chunk values are new). FP8 unmodified and patched differ from the reference by mean KL divergence
below 1e-4, most likely 1e-5 to 5e-5: their weights are exactly the reference's (FP8 widens to FP16 without loss, the scale
is applied to the FP32 output), the error is FP16 rounding of the activations and the summation order. Q8_0 about
6e-4 to 8e-4 (round 1 measured 6e-4 to 7e-4 against the FP8 run, which was itself close to exact). If the FP8 value is
above 1e-4 the FP16 activations are the larger error than assumed; if Q8_0 is below 3e-4 round 1's reading was wrong.
Q8_0 minus FP8 in PPL: about +0.010 (3B, as at 4 chunks).
E1d. The 8B reference as F32 (about 32 GB) does not fit the 16 GB card: partial offload with `-ngl` about half the
layers, the rest computed on the CPU from the memory-mapped file. It runs, at some minutes for 50 chunks. Could fail on
memory (file larger than free RAM plus the 10 GB cap) or time; then fewer chunks, and the count is stated.

## 2. Why the first-hour "hipBLAS FP16 GEMM 37 TFLOPS" disagrees with 78 to 135 TFLOPS

READ: the first-hour probe `hipblaslt_fp8 4096 4096 512` is M = 4096, N = 4096, K = 512 (output 4096 x 4096 in F32, a
64 MB write per call), while llama.cpp's prompt GEMMs have M = weight rows, N = 512 tokens, K = 3072 to 14336 with
a small output. The probe's hipBLAS call is `hipblasGemmEx` with F16 inputs, F32 output, compute 32F. My guess
(INFERRED, 60 %): the 37 TFLOPS is the shape (a short K and a large F32 output make it memory- and latency-bound), not a
different library path: the same library with the llama.cpp shape reaches above 80 TFLOPS in the same probe. Competing
guesses: a different kernel family (rocBLAS Tensile against hipBLASLt) chosen by output type or by the library's
heuristic (25 %); the older library of the first-hour note (15 %). The rocprofv3 kernel names settle which. Could
fail: the probe binary was built against a different library than the one now loaded (the trace names the library path).

## 3. The `no-overlap` FAIL (5 of 60,984 kernel pairs, 29 to 815 ns)

INFERRED, 55 % real concurrency: llama.cpp master can run independent branches of a graph on forked streams
(Q, K and V projections of a layer), and a HIP graph can then run two kernels at once. 45 % a timestamp artifact (the
profiler's start and end stamps of back-to-back kernels on one queue jitter by tens of ns). Evidence that decides: if the
overlapping pairs always involve the same two kernel types on different queue or stream ids and recur in every repetition
at the same positions, it is real concurrency; if they are same-queue neighbours at random places, differing between
repetitions, with overlaps of tens of ns, it is a timestamp artifact. The check is changed only if the evidence shows the
check is wrong (a real concurrent graph is not a defect of the trace).

## 4. Hardware counters on gfx1201

INFERRED: some counters work, as in round 1 (GRBM_COUNT did); counters of the memory-side cache (the 64 MB MALL) are not
exposed (70 %), so "do widen writes stay in the cache" can at best be bounded by L2 and memory-request counters, not
answered directly. If requests to memory counters exist (GL2C or EA read and write request counts), the widen kernel's
DRAM bytes would show as below 3 bytes per widened element if the writes were absorbed, and 3 bytes if not; I expect 3
(no absorption visible at the counter level; the 770 to 800 GB/s then comes from the profiler's kernel time, not from
cache absorption), 50 %. For the FP8 decode kernel's extra 57 W: expect more vector ALU instructions per byte than the
Q8_0 kernel (SQ_INSTS_VALU about 2 to 4 times higher), 65 %. If counters fail, the exact error is shown.

## 5. The unexplained 4.8 ms at micro-batch 2048

INFERRED: the FP16 GEMM rate is higher at N = 2048 than at N = 512 (larger tiles, less per-call overhead), saving 2 to 4
ms of the 4.8; the rest from the activation convert, scale multiply and gaps (fewer launches per token). A cause that
would make this wrong: attention (flash attention is per micro-batch and may scale differently), or the MMQ/other
kernels being unchanged and the 4.8 ms sitting in gaps between the 1500 kernels of a pass. Per-group times of the two
traces (pp 2048 at ubatch 2048 against 4 x 512) decide.

## 6. The 8B pp16 to pp64 rows that moved up to 2x

INFERRED, 50 %: the rows depend on where the core clock sits when the short job starts. The sampler will show a core
clock near idle (below 1 GHz) at the start of those short tests in the slow repetitions and above 2.5 GHz in the fast
ones. Not expected: that a high clock at the start predicts a fast row perfectly (other causes: the first-call set-up
cost, which is about 5x for FP8). If the clock shows no relation, the idea is wrong and the cause stays open.

## 7. Lower priority

If time remains: the 3B Q8_0 streams curve (flattening 7 to 8 streams, +4.9 %) with more repetitions: expect it to stay
flat within 5 % (a real effect) rather than a noise artifact, 55 %. The one-shape limit of `test-backend-ops perf`: the
tool's perf cases are a fixed list; a regex over the parameters selects only among them, so other shapes cannot be run
without changing the tool (not allowed): expected to stay as it is.
