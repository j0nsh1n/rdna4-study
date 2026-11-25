# LLM data dive, round 2: redoing what failed, and why the open ones failed

Measured 2026-11-18 on the LACT profile "Testing" (stock core clock, 317 W cap), memory clock 1408 MHz in every GPU job of
this round (held there by the displays; no row at 1258 MHz was taken, as agreed), performance level auto. A regulator-level
power-limit mod (1.2x) is in the card, so reported watts are scaled down on the main rail. Every wall-clock number in
this file is for that configuration. Software as in round 1 (llama.cpp master f477c1b90, ROCm 10.1, unmodified and
"patched" builds), plus a separate copy of the unmodified tree with extra test cases (item 7b). Run folder:
`runs/20261118T151433Z-llm2/` (git-ignored). Expectations were written and committed before the GPU jobs of each item:
`llm-workload-2-expectations.md` (items 1 to 7) and `llm-workload-2-expectations-7.md` (7a and 7b, added when the lead
made them full items). Labels: MEASURED (a run file is named), READ (source, listing), INFERRED (reasoning, not tested).
Medians everywhere (a median hides single-sample dips; item 6 says where); every number names its run file. After the review, a fix round
added a memory cap, reran the cache probe and corrected numbers: `runs/20261118T165826Z-llm2fix/` (section "Fix round" below). 119 GPU jobs in the main folder, every one with exit 0 and the memory
clock 1408 before and after, highest hotspot 93 C (the 3B F32 reference run; stop limit 95 C, never reached), no thermal
stop, no game running, the LACT profile "Testing" throughout.

## Plain summary

1. **The exact-weights check is done for the 3B and only approximated for the 8B.** For the 3B, an F32 file holding every
   FP8 weight times its scale exactly (bit-checked against the FP8 file) is the reference. The FP8 route differs from it by
   a mean KL divergence of 3e-6 (perplexity equal to the reference's own within 0.0005), Q8_0 by 7e-4 (perplexity +0.010), about 230 times more. So round 1's
   route-against-route numbers were already close to the truth. The 8B exact file (32 GB) was built and verified but could
   not be run on this machine (memory and swap ran out; two attempts, see item 1); the 8B reference is an F16 file instead (converter output, not bit-checked against the checkpoint), whose
   own distance from exact was measured on the 3B (3e-6).
   > **Updated in round 3 (Q2):** a mixed reference (F16 where every product is exact in F16, else F32) is 27.9 GB, not 16 GB: only 31 of the 224 FP8 tensors qualify. It does not fit (GPU 13.7 GB free plus the 10 GB job cap); the 8B stays against the F16 reference. Side finding: the patched build's prompt path folds the scale into the
   FP16 weights while widening, so its weights are exactly the F16-rounded products: its numbers equal the F16 reference's to every printed digit.
   > **Updated in round 4 (`llm-runtimes.md`, R5):** the 3e-6 (FP8) and 7e-4 (Q8_0) above are distances from the exact FP8 weights, not from the true model. Against Meta's 16-bit 3B (an Unsloth re-upload of it), run in F32 over 8 chunks, the FP8 route is 3.7e-3 and Q8_0 4.3e-3, a ratio of 1.16 and not 230 (llama.cpp's own 16-bit file: 2e-5): the error of both is in the FP8 checkpoint's weights, not in the route.
   > **Updated after round 4 (`llm-8b-exact.md`):** the 8B exact reference was built and run (mixed F16/F32, 27.87 GB, bit-checked, one job under a 16 GB cap). That reference is the 8B FP8 checkpoint's own weights converted exactly, **not Meta's 16-bit 8B** (not on disk), and the reference run itself ran 22 of 32 layers on the CPU in F32; so the 8B numbers that follow are distances from the exact FP8 weights, a different question from the 3B figures of the round 4 note above (3.7e-3 and 4.3e-3, measured against Meta's 16-bit weights); the 8B checkpoint's own distance from the true model is not measured. Against it the plain FP8 file is 5e-6 (equal to the 5e-6 against the F16 reference above), the `outq8` FP8 file 8.1e-5 (its Q8_0 output matrix) and Q8_0 5.2e-4 over 8 chunks (6.1e-4 over 50 chunks against the F16 reference above).
2. **The "37 TFLOPS" of the first hour was the shape, not the library.** The same library and kernel family reach 108 to 136
   TFLOPS at llama.cpp's shape; at the first-hour shape (K = 512 with a 4096 x 4096 output) the library's heuristic picks a tiny 16 x 16
   tile that runs at 36.5 TFLOPS (the first hour's number reproduced).
3. **The no-overlap FAIL was most likely a timestamp artifact (INFERRED), not concurrent execution.** 0 overlapping pairs in 706,017 kernel rows
   of 11 further tg128 traces; the five overlaps of the rerun were 29 to 815 ns, one queue, neighbouring dispatches; three are a KV-cache write followed by the
   attention kernel that reads it (at one position of three different tokens), two are matvec pairs. llama.cpp does run concurrent branches only when
   `GGML_CUDA_GRAPH_OPT=1` is set, which no run did. The check stays as it is.
   > **Updated in round 3 (Q5, `llm-workload-3.md`):** settled with an independent clock. In 323,982 dependent and independent kernel pairs under the profiler (plain launches and HIP graphs) the in-kernel
   > REALTIME counter shows no overlap in any pair; the profiler showed 3 overlaps (465, 490 and 8,040 ns) and in all three the waves of the second kernel started 3.4 us after the last wave of the first one
   > had finished. A real overlap (positive control) shows on both clocks. So the profiler overlaps are timestamp artifacts (MEASURED on a synthetic pair probe; llama.cpp's own five were not re-timed in-kernel).
4. **Hardware counters mostly do not count on this card with this stack.** Of 13 counter sets tried, only GRBM, `SQ_WAVES`, `SQ_BUSY_CYCLES` and
   `SIMD_UTILIZATION` return numbers; every memory-side and instruction counter reads 0 even for a kernel that reads 256 MiB. So the two
   counter questions stay open, but the cache question has a direct answer from a new probe: writes to a reused buffer up to
   about 50 MiB are absorbed (1524 GB/s write rate at 50 MiB, against 611 for a rotating buffer).
5. **The 4.8 ms at micro-batch 2048 is the matrix kernel getting faster (and fewer gaps).** Per 2048 tokens the widen drops by 27.1 ms
   (exactly four times fewer calls), the FP16 GEMM by 21.0 ms (103 to 128 TFLOPS), the launch gaps by 7.8 ms; attention gets 4.5 ms slower.
6. **The "clock ramp" explanation for the moving 8B rows is not supported.** In 12 repetitions, 11 reproduced every row within 3 % by the median of their five samples;
   in the one that did not, five of seven rows were 8 to 25 % slower and the card was 68 to 96 % busy in the test windows (probably host-side stalls), not slow at the start.
   Single slow samples also hit two other runs (Q8_0 run 6 down to 0.28 of the median); a median hides them, the mean shows them.
7. **Streams (7a)**: with six interleaved repetitions the round-1 crossover statements hold, including the 3B's 8-stream lead of the patched build
   (+2.8 %, ranges do not overlap). **Shapes (7b)**: f8 - f16 behaves as a widen at n = 512 for the big matrices (718 to 842 GB/s), but at
   n = 2048 the GEMM after a widen runs 6 % faster than on a stored F16 matrix, which hides most of the widen cost; the rest of the table is below.

## Expectations against results (round 2)

| # | Expectation | Result | Verdict |
|---|---|---|---|
| E1a | F16 exact for 30 to 70 % of products on every tensor, never 100 %; F32 exact for 100 % | F32 exact for 100 % in all 196 FP8 tensors of the 3B (and the scales of both models are F32 values of at most 8 significant bits, READ from the GGUF files, `llm2fix/scale-bits.txt`). F16: lowest tensor 52.5 %, highest tensor 100.0 %, i.e. at least one tensor is fully exact in F16; BF16 lowest 13.3 % (`ref3b-build.log`) | right on F32, wrong on "never 100 %" |
| E1b | Sample rows bit-identical; a damaged row is caught | 3B: 3528 rows and 256 other tensors identical; 8B: 2240 rows and 67 other tensors equal (`ref3b-build.log`, `ref8b-compare.log`); the selftest catches a one-bit change and an F16 round trip | right |
| E1c | FP8 mean KLD below 1e-4, most likely 1e-5 to 5e-5; Q8_0 6e-4 to 8e-4; PPL gap Q8_0 about +0.010; reference within 0.003 of FP8 | FP8 unmodified 3e-6, patched 3e-6 (3B); 5e-6 (8B, F16 reference). Q8_0 7.05e-4 (3B), 6.11e-4 (8B). PPL against the reference run's own perplexity (9.4622): -0.0004 and +0.0001 (FP8), +0.0100 (Q8_0, 3B); against the re-read logits file the table gives +0.0018, +0.0022, +0.0122 (a storage shift of about 0.002) | right in kind; the FP8 value is 3 times below the low end of my range (1e-5) |
| E1d | the 8B F32 reference runs with partial offload in some minutes | built and verified, but no pass finished in 5 minutes (attempt 1, stopped at 301 s) and none made progress in 4.5 minutes (attempt 2, stopped at 272 s; swap full, seen live, not logged); the F16 reference ran in 60 s | wrong |
| E2 | the cause is the shape (60 %); same library | same library and kernel family, the first-hour shape gets a 16 x 16 tile (36.5 TFLOPS), llama.cpp's shapes 64 x 64 and 128 x 128 tiles (108 to 136) | right |
| E3 | real concurrency (55 %) | 0 overlaps in 706,017 rows; the rerun's five were ns-scale, one queue, neighbouring dispatches (three KV-write then attention, two matvec pairs); concurrency needs an env switch no run set | wrong (the 45 % case); most likely a timestamp artifact (INFERRED) |
| E4 | some counters work; memory-cache counters not exposed; VALU counts 2 to 4 times higher for the FP8 decode kernel | only GRBM and a few SQ counters count; memory and VALU counters all read 0. Neither question answered by counters | right that the cache counters fail, wrong that VALU counters work |
| E5 | GEMM rate higher at N = 2048 saves 2 to 4 ms of the 4.8; rest gaps and others | per 512 tokens: GEMM 5.2 ms, gaps 1.95 ms, decode-size matvec 0.6 ms, minus attention 1.1 ms and activation convert plus elementwise 0.6 ms | right on the cause, 1 ms over on its size |
| E6 | the slow rows start at a low clock; fast ones above 2.5 GHz | not supported: the one slow run was 8 to 25 % slow in five of seven rows (0.3 and 3 % in the other two) with the card 68 to 96 % busy in its windows (runs 1 to 4: 93 to 100 %); clock at the start does not predict speed | wrong |
| E7a1 | spread 1 to 4 % per cell, above 6 % the exception | 3B cells 0.5 to 2.6 %, 8B above 2 streams 0.2 to 5.6 %; 8B Q8_0 and FP8 unmodified at 1 and 2 streams 11 to 20 % | right (the 8B Q8_0 at 4 and 5 streams reaches 5.6 and 4.5 %) |
| E7a2 | 3B: leads at 1, 2, trails somewhere in 3 to 7; the 8-stream lead is inside the spread; 8B leads only at 1 | 3B: +9.4, +3.6 %, then -3.8 to -5.8 % from 3 to 7, and +2.8 % at 8 with separate ranges; 8B: leads only at 1 (+7.5 %) | right on the 3B crossings and on the 8B; "the 8-stream lead is inside the spread": wrong (the ranges are separate); "median within +-3 % of zero": right (+2.8 %) |
| E7a3 | 3B Q8_0 7 to 8 streams below +6 % | 777.4 to 807.5 tokens/s = +3.9 % | right |
| E7b1 | implied widen 650 to 800 GB/s for big matrices (above 10 million elements), below 500 for small ones; same for n = 32, 512, 2048 | n = 512: the seven matrices of 16.8 million elements and up 718 to 842 (above 800 for three: 842, 804, 803); 3B q/o (9.4 million) 1424, k and v (3 to 4 million) 954 to 1322, higher not lower (cache); n = 32 and 2048 differ strongly | right at n = 512 for the biggest matrices, wrong for the small ones and for n-independence |
| E7b2 | f16 above 100 TFLOPS at n = 512 and 2048 for big shapes, below 40 at n = 32 | n = 512: 101.7, 72.5 (3B down), 136, 120, 109, 117; n = 2048: 92 to 144; n = 32: 20 to 48.5 | mostly right |
| E7b3 | f8 / q8_0 time ratio 2 to 5 at n = 1 and 8 | 2.5 to 3.9 at n = 1 and 2.4 to 3.9 at n = 8 (shapes of 9 million elements and up) | right |
| E7b4 | f8 = f16 + widen within 25 % for n >= 32, fails for small k and v | holds at n = 512 for the big shapes (traced widen 99.4 us at 3B gate/up against 89.7 us measured difference; the GEMM after the widen is 14 us faster); n = 32 was not traced (gate/up, down and output imply 627 to 744 GB/s there, q/o 1086 to 1162); fails at n = 2048 (3B gate/up 48.1 us measured against a traced widen of 97 us; 8B down 28.4 us against about 223 us implied by the 791 GB/s of round 1's widen) without the GEMM correction | right at n = 512, wrong at n = 2048 (a correction was found) |

## Item 1: exact-weights reference (quality)

**What failed before.** The reference was parked: `llama-quantize` cannot apply the separate scale tensors, and the
converter's Python packages were missing. **Now.** The convert venv (`toolchains/convert-venv`) is used; `scripts/llm_ref.py` (run with that venv)
writes and checks the files, `scripts/study-llm.sh r2q` runs the quality jobs.

- **Exactness (MEASURED, `ref3b-build.log`).** Both models store one scale per tensor as an F32 value with at most 8 significant bits (READ from the GGUF files:
  392 and 448 scale values, 1 to 8 bits, `llm2fix/scale-bits.txt`; E4M3 has 4, a format fact, READ), so a product
  has at most 12 significant bits. F32 holds all of them (100 % of the products in all 196 FP8 tensors of the 3B); F16 holds 52.5 to 100 % per tensor and BF16
  13 to 100 %. The check reports a scale with more than 8 bits as not exact in F32 (selftest break).
- **3B reference** = `llama32-3b-auto.gguf` with every FP8 tensor replaced by FP8 x scale in F32 (the scale tensors dropped), all other tensors and
  the metadata copied byte for byte (embedding and output stay BF16, the original values). The task text named `llama32-3b-fp8-outq8.gguf`; that file stores the output matrix as Q8_0,
  which has already lost information, so the reference is built from the "auto" file whose output is BF16 (deviation, stated). 12.9 GB. Bit compare: 16 random rows plus the first and
  last of every FP8 tensor (3528 rows), float64 product equal to the stored F32 and the F32 bits equal; the 256 other tensors byte-identical.
- **8B**: `convert_hf_to_gguf.py` from the FP8 checkpoint with `--outtype f32` (32.1 GB), from a scratch folder of symlinks whose `config.json` has one label changed
  (`naive-quantized` to `float-quantized`, because the converter asserts on the first; llama.cpp itself untouched). Bit compare against the checkpoint: 2240 rows
  (the converter's q/k row permutation applied to the checkpoint rows) and 67 other tensors equal; `rope_freqs` is generated by the converter and not compared.
- **8B F32 evaluation failed (parked).** Attempt 1 (`-ngl 12`, the rest mapped from the file, default op offload): the loader copied every CPU-resident layer to the card from mapped
  pages on every pass; no pass finished in 5 minutes (stopped at 301 s, exit 143, `attempt1-8b-op-offload/r2q-llama31-8b-ref.meta`; process in `hipMemcpy`; heavy page faulting and I/O pressure, seen live, not logged). Attempt 2 (`--no-op-offload`): after 4.5 minutes (272 s, `.meta`)
  the logits file still held only its header, the process in disk wait, `kswapd` busy and the swap full (seen live, not logged; the machine has other programs resident); I stopped it by PID to protect the
  owner's session (`attempt2-8b-f32-no-op-offload/`). Both stopped by the debugging limit. **The 10 GB memory cap of the fix round was not in force during these two attempts (or the F16 run)**: runs of that kind now start under it. Cause of the stall: INFERRED to be memory (30 GB machine, a 32 GB file, other programs resident; how much they held was not logged).
  A side finding, MEASURED: on the 3B, loading an F32 file with the default memory-mapped load took minutes (the card upload pins mapped pages one tensor at a time), while `-lm none` took seconds.
- **8B reference used: F16** (`--outtype f16`, 16.1 GB, a separate converter invocation from the F32 one: `refs/convert8b-f16.log`; its embedding and output tensors are BF16 values rounded to F16 as well), `-ngl 27` (the rest on the CPU), `--no-op-offload`.
  **Converter F16, not bit-checked:** `compare8b` handles only the F32 file, so the F32 file's verification does not transfer, and the F16 file was deleted after its sha256 was logged. The FP8 routes' KLD of 5e-6 against it is indirect support only. NOT exact: F16 rounds products that need 12 bits. How far from exact:
  on the 3B the F16 version (made by the same writer, `make3b ... f16`) rounds 12.96 % of all products, worst relative error 6.7e-2 (products below the F16 normal range), and its KL divergence
  against the exact reference is 3e-6 (below, row f16ref) against 7e-4 for Q8_0. For the 8B the same size of error is expected (INFERRED, not measured).

3B (50 chunks of 512; reference llama32-3b-exact-f32.gguf; the reference run's own perplexity 9.4622, and 9.460064 re-read from its saved 16-bit logits file; memory clock 1408 MHz):

| run | PPL | PPL - re-read reference | mean KLD | median KLD | 99 % KLD | max KLD | same top token % |
|---|---|---|---|---|---|---|---|
| fp8-unmod | 9.461824 | +0.0018 | 3e-06 | 1e-06 | 4.5e-05 | 0.000149 | 99.922 |
| fp8-patched | 9.462283 | +0.0022 | 3e-06 | 1e-06 | 4.3e-05 | 0.000603 | 99.945 |
| q8_0 | 9.472246 | +0.0122 | 0.000705 | 0.000379 | 0.005693 | 0.040883 | 98.682 |
| f16ref | 9.462283 | +0.0022 | 3e-06 | 1e-06 | 4.3e-05 | 0.000603 | 99.945 |

8B (50 chunks of 512; reference llama31-8b-f16.gguf, converter F16, not bit-checked; the reference run's own perplexity 7.2670, and 7.264332 re-read from its saved logits file; memory clock 1408 MHz):

| run | PPL | PPL - re-read reference | mean KLD | median KLD | 99 % KLD | max KLD | same top token % |
|---|---|---|---|---|---|---|---|
| fp8-unmod | 7.266773 | +0.0024 | 5e-06 | 1e-06 | 5.2e-05 | 0.000889 | 99.875 |
| fp8-patched | 7.267018 | +0.0027 | 5e-06 | 1e-06 | 5.2e-05 | 0.000849 | 99.867 |
| q8_0 | 7.274153 | +0.0098 | 0.000611 | 0.000332 | 0.00483 | 0.051544 | 98.651 |



Notes. Reading the tables: "PPL - re-read reference" is the run's mean perplexity minus `Mean PPL(base)`, the reference's perplexity re-read from the saved 16-bit logits file (3B 9.460064, 8B 7.264332). The reference run's own
`Final estimate` is 9.4622 and 7.2670, about 0.002 higher: that storage shift is inside the deltas. Against the reference's own perplexity the 3B FP8 routes are -0.0004 and +0.0001 and Q8_0 +0.0100; the 8B FP8 routes -0.0002 and +0.0000 and Q8_0 +0.0072.
The printed standard error of the table's difference is 0.0010 (3B FP8 rows), 0.0038 (3B Q8_0), 0.0016 (8B FP8 rows) and 0.0029 (8B Q8_0), so every FP8 delta is inside two standard errors. Mean KLD prints with 6 decimals, so 3e-06 means 2.5e-6 to 3.5e-6. The 3B base logits file (3270732420 bytes, sha256 a3e87a5d...) and the 8B base (same size, 24d25a79...) were deleted after the runs;
the evidence is in `r2q-llama32-3b.base.txt` and `r2q-llama31-8b.base.txt`. The 3B f16ref row equals the patched row in every digit
(PPL 9.462283, max KLD 0.000603): READ from the round-1 trace, the patched build's prompt path runs `convert_fp8_scaled_f16` (FP8 x scale to FP16 in one kernel) and has no separate scale multiply, so its weights
are the F16-rounded products, exactly what the F16 file holds. The unmodified route widens without the scale and multiplies the result: exact weights, and the slightly smaller worst-case KLD (0.000149 against 0.000603).
Round 1's route-against-route numbers stand: against the true weights the FP8 routes are about 3e-6, Q8_0 about 7e-4. The writer and the checker both decode FP8 with torch's `float8_e4m3fn`; if that disagreed with ggml's decode they would
agree on a wrong file, and the 3e-6 KL divergence of the real FP8 routes against the reference is the end-to-end check that it does not.

Files: `r2q-*.log`, `r2q-*.meta`, `refs/sha256.txt` (F32 files), `refs/sha256-f16.txt`, `refs/sizes.txt`, `ref3b-build.log`, `ref3b-f16-build.log`, `ref8b-compare.log`, `refs/convert8b*.log`. Not used by any table:
`first-3b-pass/` (an identical first pass of the 3B quality jobs, same digits) and the `r2c-x-*`, `r2c-y-*` counter folders (an earlier exploratory counter pass with the same zeros, no logs kept).
Disk: the four reference files together took 68.3 GB (`refs/sizes.txt`; the peak size of the whole run folder was not logged); they were deleted after the numbers and sha256 were logged (reproducible with `llm_ref.py` and the converter); the run folder is now 0.4 GB, the separate llama.cpp copy of item 7b 0.9 GB.

## Item 2: why the first-hour hipBLAS FP16 GEMM said 37 TFLOPS

**What failed before.** Round 1 reported 109 TFLOPS (`test-backend-ops` f16) and 78 to 135 in the real GEMMs against the first hour's 37.3, with a guess ("a different library path, version or shape") labelled INFERRED.
**Measured now.** The first-hour probe (`build/gfx1201/hipblaslt_fp8 M N K`; M = weight rows, N = tokens, K = inputs, the same convention as llama.cpp's call) under `rocprofv3 --kernel-trace` and
rocBLAS's call log (`ROCBLAS_LAYER=2`), with the libraries it loads (`r2libs-ldd.txt`: libhipblas.so.3, librocblas.so.5, libhipblaslt.so.1 of the ROCm 10.1 tree, the same ones llama.cpp loads).

| shape M x N x K | hipBLAS F16 TFLOPS (F16 in, F32 out) | kernel picked (tile) | kernel time |
|---|---|---|---|
| 4096 x 4096 x 512 (the first-hour shape) | **36.5** | `Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32...` | 460 us |
| 4096 x 512 x 4096 (llama.cpp-like) | 107.8 | `..._MT64x64x64...` | 146 us |
| 3072 x 512 x 8192 (3B down) | 79.9 | `..._MT128x128x32...` | 309 us |
| 4096 x 512 x 14336 (8B down) | 136.2 | `..._MT128x128x32...` | 415 us |
| 4096 x 4096 x 4096 (cube) | 128.1 | `..._MT128x128x32...` | 1046 us |
| 4096 x 512 x 512 | 64.4 | `..._MT128x128x32...` | 19.5 us |
| control: 4096 x 512 x 4096 with `ROCBLAS_USE_HIPBLASLT=0` | 11.5 | `Cijk_..._HSS_BH_MT64x32x8...` (a different, Tensile-only kernel) | 1471 us |

rocBLAS's log shows `gemm_ex` with `f16_r` inputs, `f32_r` output and `f32_r` compute type for every row (math mode: FP16 multiplies, FP32 accumulate, FP32 result). llama.cpp's own f16 route
at m = 4096, n = 512, k = 14336 (`test-backend-ops`, trace `r2libs-bops-f16.d`) runs the same kernel family (`Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32`); the real GEMMs of the models
in the round-1 traces carry the same names (`MT64x64x64` and `MT128x128x32`). **Cause (MEASURED, the shape sweep and the control):** one library and one kernel family; the library's choice of tile depends on the shape,
and for K = 512 with a 4096 x 4096 F32 output it chooses a 16 x 16 tile, which is 3 to 4 times slower. Why the heuristic chooses so is INFERRED (not looked into). The "different library version" guess of round 1 is not needed (the
probe reproduces the first-hour number, 36.5 against 37.3, with the current libraries). The hipBLASLt FP8 line of the probe reads 167 TFLOPS at the first-hour shape (`r2libs-orig.log:67`; the first hour's "170" is READ from its note), 166 to 245 at the shapes with k of 4096 and more, and 57.5 at 4096 x 512 x 512 (`r2libs-n512k512.log:67`).
Files: `r2libs-*.log`, `r2libs-*.d/t_kernel_trace.csv`, `r2libs-ldd.txt`.

## Item 3: the no-overlap FAIL (5 of 60,984 kernel pairs overlap by 29 to 815 ns)

**Round-1 rerun, re-examined (the saved trace).** All five pairs are on one queue and stream (queue 2), dispatch ids differ by one, and three of five are `k_set_rows` (the KV-cache write) followed by `flash_attn_tile` (which reads that cache), at
position 7 of 485 of three different tokens (layer 0), so those three cannot be concurrent by data dependence; the other two are `mul_mat_vec_f` followed by `rope_norm` (29 ns) and a 176 us `mul_mat_vec_f` followed by an 86 us one (303 ns; gate then up in the graph, which do not depend on each other). Overlaps 29, 303, 380, 659 and 815 ns, against kernels of 1 to 176 us.
**Repeats (MEASURED, `r2ov-*.overlap.txt`).** The same trace taken five more times (8B patched, 312,825 kernel rows), plus the 3B patched twice (109,650 rows), the 8B Q8_0 twice (158,412 rows) and the 8B patched with HIP graphs
switched off (`GGML_CUDA_DISABLE_GRAPHS=1`, twice, 125,130 rows): **0 overlapping pairs in 706,017 rows.** The round-1 first run also had 0, so 5 overlaps appeared in one of the two round-1 runs of that trace and in none of the 11 repetitions here.
**What llama.cpp does (READ, `ggml-cuda.cu`):** concurrent branches (forked streams in the captured graph) only run when `GGML_CUDA_GRAPH_OPT=1`, which neither round set; a default graph is one linear chain.
**Positive control (MEASURED):** with `GGML_CUDA_GRAPH_OPT=1` the finder reports 12 overlaps in one of two traces, one of them 785 us long (a 1.4 us `quantize_q8_1` running inside the 789 us output-matrix kernel, dispatch ids
reversed): real concurrency looks like that, and even then the trace reports one queue and one stream id, so queue ids do not tell the two apart.
In the positive-control trace 11 of the 12 overlaps are the same size class (292 to 2316 ns) and include that same 170/85 us pair at one position, so the control does not tell jitter from a small real overlap for the ns-scale ones; it only shows what a long one looks like.
> **Updated in round 3:** a synthetic pair probe with an independent in-kernel clock MEASURED the same kind of profiler overlap as an artifact (see `llm-workload-3.md`, Q5); the decision below for llama.cpp's own five overlaps stays INFERRED (they were not re-timed in-kernel).
**Decision.** Most likely a timestamp artifact (INFERRED from: ns-scale, one queue, neighbouring dispatches, none in 706,017 rows, none with graphs off, no concurrency without the env switch; three of the five are data-dependent pairs, two are not), not concurrent execution. The check
(`no-overlap`, no tolerance) is **not changed**: it flags a rare jitter of the profiler's timestamps, and nothing that the analysis rests on moves (the wall and gap numbers of that trace did not change). A rerun of `all` can show this FAIL again;
it should be read as "one timestamp jitter", with `scripts/llm_kernels.py overlap <trace.csv>` giving the list. Tool: new `overlap` and `topk` commands of `llm_kernels.py` (selftest: planted overlaps on one and on two queues, the running-maximum case, a clean trace).

## Item 4: hardware counters, and the cache question

**Counters (MEASURED, `r2counters`, `r2c-*.pmc.txt`).** `rocprofv3-avail list` lists the counters of gfx1201 (`r2c-list.log`). Each set was run against the bandwidth probe, whose read and copy kernels move 256 MiB; a counter that reads 0 in every dispatch
of such a kernel counts nothing. **Count**: `GRBM_COUNT`, `GRBM_GUI_ACTIVE` (1.08 million cycles for the read kernel), `SQ_WAVES` (8192 = the launched waves), `SQ_BUSY_CYCLES`, `SIMD_UTILIZATION` (0.993). **Read 0 on a kernel that moves 256 MiB:**
`GL2C_EA_RDREQ_sum`, `GL2C_EA_WRREQ_sum`, `GL2C_HIT_sum`, `GL2C_MISS_sum`, `GL2C_EA_*_128B/64B`, `FETCH_SIZE`, `TCP_REQ`, `TCP_REQ_MISS`, `L2CacheHit`, `MemUnitBusy`, `VALUBusy`, `SQ_INSTS_VALU`, `SQ_INSTS_WAVE32_VALU`,
`SQ_INSTS_FLAT`, `SQ_WAVE32_INSTS`, `SQ_INSTS_LDS`, `SQ_INST_CYCLES_VMEM`, `SQ_WAVE_CYCLES`, `SQ_INST_CYCLES_VALU`, `SQ_INSTS_TEX_STORE` (the profiler accepts them and returns zeros; no error message; the profiler also prints a warning for the integrated GPU, which is
unrelated). **Consequence:** neither open question can be tested with counters. Why they read 0 is not known (a profiler or firmware limit of this stack is a guess; not examined).
> **Updated in round 3 (Q4):** ROCm 10.0's rocprofv3 cannot collect any of the 13 sets on this card (it aborts at counter initialisation, "aqlprofile API table load failed"), including the four that work with 10.1; its kernel trace works. So the zeros are not a 10.1-only effect that 10.0 would fix; nothing was learned about why they are zero.
**The "does the widen's write stay in the 64 MB cache" question, answered by a probe instead** (`src/widen_cache.hip`, built by `scripts/study-llm.sh r2mallbuild` into `build/gfx1201-rocm10/widen_cache`, `r2mall-*`; read S MiB from a rotating 1 GiB pool and write 2 S MiB, the destination either one buffer reused every call, as llama.cpp's
scratch pool is, or rotating through 1 GiB; 3 repetitions of 200 calls; GB/s counts bytes moved, events around each launch):

| S read / written (MiB) | widen, destination reused | widen, destination rotating | read only | write only, reused | write only, rotating |
|---|---|---|---|---|---|
| 8 / 16 | 635 | 500 | 292 | 1030 | 530 |
| 25 / 50 (a 3B gate matrix) | 681 | 610 | 535 | **1524** | 611 |
| 50 / 100 | 629 | 630 | 593 | 632 | 632 |
| 100 / 200 | 635 | 636 | 656 | 644 | 644 |

Writes to a buffer that is rewritten again and again are absorbed while the buffer is up to about 50 MiB (write rate 1524 GB/s at 50 MiB, 2.5 times the 611 of a rotating buffer; gone at 100 MiB). MEASURED, and stable (1524, 1520.6, 1524 in the three runs; 1525.9 and 1527.6 in the fix-round reruns). For a widen-shaped kernel
the gain is 12 % in round 2 (S = 25 MiB: 681 against 610 GB/s; 635 against 500 = 27 % at S = 8, where the fix round measured the launch overhead at 14 to 24 % of the time), but the fix-round reruns move it: 4 to 9 % at S = 25 (see "Fix round"). That gain is the least stable number of this item. The probe's best widen rate (681 GB/s) is still below the 770 to 800 GB/s of llama.cpp's widen kernels in the traces; the rest of that gap is not explained here
(the real kernel's access pattern and the 8 MiB L2 are not the probe's), and which cache level absorbs the writes (the 8 MB L2 or the 64 MB memory-side cache) is not separated by this probe; the size dependence (50 MiB absorbed) points to the 64 MB cache (INFERRED). The 8 MB L2 and the 64 MB memory-side cache are device facts, READ from `measurements/memory-system.md` (table of the cache levels) and the card record (`L2_KB`, `L3_KB`).
**The FP8 decode kernel's extra 57 W**: needs instruction counters, which read 0: **still INFERRED, untested.** Not attempted by other means (reading the kernels' ISA is a possible next step, parked).
> **Updated in round 3 (Q1):** the ISA was read and a matched-bandwidth probe pair was run; see `llm-workload-3.md`, Q1, for how much of the 57 W is explained.

## Item 5: the 4.8 ms at micro-batch 2048

3B patched FP8, `r2ub-trace-512-*` (4 x 512) against `r2ub-trace-2048-*` (one batch of 2048), per-group medians scaled to 2048 tokens (`r2ub-3b-fp8-patched.groupcmp.txt`, 3 traced passes each, unprofiled controls 9667 and 12266 tokens/s):

| group (ms per 2048 tokens) | 4 x 512 | 1 x 2048 | difference |
|---|---|---|---|
| widen | 36.19 | 9.04 | -27.15 |
| FP16 GEMM | 111.51 | 90.53 | **-20.99** (103.5 to 127.5 TFLOPS on the layer matrices) |
| activation convert | 12.07 | 13.27 | +1.20 |
| attention | 18.38 | 22.87 | +4.49 |
| elementwise | 21.37 | 22.65 | +1.28 |
| decode-size matvec | 3.0 | 0.75 | -2.25 |
| gaps between kernels | 10.42 | 2.64 | -7.78 |
| pass wall | 212.94 | 161.60 | **-51.34** |

Per 512 tokens the traced saving is 12.8 ms (wall 51.34 / 4), which closes as: the widen (4 times fewer calls, 36.19 to 9.04 = exactly a quarter, MEASURED) 6.8 ms (round 1's number, confirmed) + the larger GEMM rate 5.2 ms + fewer launch gaps 1.95 ms + the matvec
kernels 0.6 ms = 14.55, minus attention 1.1 ms (a longer attention per batch), minus activation convert and elementwise 0.6 ms (+1.20 and +1.28 per 2048 tokens) = 12.8. Which 'saving' is which: round 1's 11.6 ms came from unprofiled tokens/s (53.8 - 42.2); the unprofiled controls here give 11.2 ms
(52.96 - 41.74, medians of five samples) and the traced passes 12.8 ms (the 1.6 ms difference is profiling overhead, INFERRED). **So the "unexplained 4.8 ms" is the FP16 GEMM running at a higher rate for N = 2048 (MEASURED: 103.5 to 127.5 TFLOPS),
plus fewer gaps; why the larger N helps (bigger tiles; the trace shows `MT128x128x32` at N = 2048 where N = 512 also uses `MT64x64x64`, see item 7b) is INFERRED.** Q8_0 does not change (158.8 to 155.4 ms; MMQ 112.7 to 117.0 TOPS; 12785 against 12735 tokens/s), which is why FP8's
share of Q8_0 rises from 76 % to 96 % here (9667 / 12785 and 12266 / 12735 tokens/s in these controls).

## Item 6: the 8B prompt rows that moved

Round 1 saw the 8B rows pp16 to pp64 differ by up to 2x between two runs and put it down to the clock at the start of a short test (INFERRED). Now the round-1 command (`-p 16,32,64,128,256,512,1024 -n 0 -r 5`) was run six times each for the 8B patched FP8 and the 8B
Q8_0, interleaved, with 6 to 20 s between the jobs, under the 20 ms sampler (`r2clock/`, `r2clock-*.log`). The clock inside each test is found by rebuilding the tests' windows backwards from the last busy sample (the tests run back to back; sampler resolution 20 ms,
alignment good to a few samples; selftest on a canned recording).

8b-fp8-patched (6 runs, memory clock held):

| row | runs | median tok/s | min to max | max/min | median clock in window MHz | clock min to max | r(speed, mean clock) | r(speed, clock at start) | GPU busy % in window | r(speed, busy %) |
|---|---|---|---|---|---|---|---|---|---|---|
| pp16 | 6 | 296 | 224 to 297 | 1.33 | 3200 | 2833 to 3203 | 1.00 | 1.00 | 73 to 99 | 0.99 |
| pp32 | 6 | 594 | 498 to 595 | 1.19 | 3208 | 3028 to 3219 | 1.00 | 0.33 | 82 to 100 | 1.00 |
| pp64 | 6 | 1113 | 956 to 1114 | 1.17 | 3214 | 2484 to 3224 | 0.13 | 0.70 | 78 to 100 | 0.53 |
| pp128 | 6 | 2016 | 1854 to 2018 | 1.09 | 3201 | 2784 to 3222 | -0.00 | 0.53 | 82 to 99 | -0.03 |
| pp256 | 6 | 3329 | 3318 to 3341 | 1.01 | 3100 | 2392 to 3129 | 0.70 | 0.61 | 68 to 97 | 0.64 |
| pp512 | 6 | 4858 | 4707 to 4877 | 1.04 | 2979 | 2472 to 3001 | 0.97 | -0.44 | 78 to 100 | 0.98 |
| pp1024 | 6 | 4831 | 3736 to 4844 | 1.30 | 2970 | 2786 to 2977 | 0.97 | -0.98 | 89 to 99 | 0.92 |

8b-q8_0 (6 runs, memory clock held):

| row | runs | median tok/s | min to max | max/min | median clock in window MHz | clock min to max | r(speed, mean clock) | r(speed, clock at start) | GPU busy % in window | r(speed, busy %) |
|---|---|---|---|---|---|---|---|---|---|---|
| pp16 | 6 | 973 | 954 to 978 | 1.03 | 3184 | 3178 to 3196 | 0.11 | -0.67 | 82 to 99 | -0.09 |
| pp32 | 6 | 1747 | 1740 to 1749 | 1.01 | 3159 | 2584 to 3168 | -0.53 | 0.30 | 68 to 98 | -0.45 |
| pp64 | 6 | 2811 | 2739 to 2821 | 1.03 | 3170 | 1935 to 3171 | 0.98 | 0.98 | 51 to 96 | 0.98 |
| pp128 | 6 | 3918 | 3885 to 3930 | 1.01 | 3182 | 3112 to 3184 | 0.92 | 0.94 | 96 to 97 | -0.05 |
| pp256 | 6 | 5465 | 5428 to 5468 | 1.01 | 3073 | 2719 to 3083 | -0.18 | 0.66 | 77 to 98 | -0.21 |
| pp512 | 6 | 5945 | 5932 to 5952 | 1.00 | 2968 | 2869 to 2985 | -0.60 | -0.29 | 89 to 98 | -0.67 |
| pp1024 | 6 | 5896 | 5885 to 5907 | 1.00 | 2980 | 2969 to 2982 | 0.69 | -0.03 | 97 to 99 | -0.28 |



Reading: **eleven of the twelve runs reproduce every row within 3 % by the median of their five samples** (all six Q8_0 runs, and patched runs 1 to 4 and 6; the largest single deviation there is pp64 of Q8_0 run 6, -2.6 % against the row median, -2.9 % against the fastest run). In the one that does not (patched run 5: five of seven rows 8 to 25 % slow by the median, pp16 224 against 296, pp1024 3736 against 4834; pp256 and pp512 only -0.3 and -3.1 %)
the **GPU was 68 to 96 % busy in the test windows** (runs 1 to 4: 93 to 100 %) with a few samples at the idle clock inside the windows, which says the card waited for the host (launch stalls from other load on the machine; INFERRED), not that it started at a low clock: the clock at the
start of the test does not predict speed (the correlation columns swing from -0.98 to 1.00 across rows with only six points, each driven by the one slow run). Limit of this evidence: patched run 6 also has low-busy windows (pp64 78 %, pp128 82 %) without losing speed by the median, and Q8_0 run 6 has 51 % busy at pp64 with the median only -2.6 %; in run 5 the lowest busy window (pp256, 68 %) is the row that lost 0.3 % by the median. So the window alignment (rebuilt from durations) is not exact enough to call the busy share a measurement per test, and row by row it does not line up with the loss; the whole-run difference of run 5 is what carries the conclusion. The mean clock in the window falls too in the slow runs (2833 against 3200 MHz at pp16), most likely a consequence of the idle gaps (INFERRED).
**The clock-ramp idea of round 1 is not supported (the cause of the 2x between runs stays open; a host-load cause is the best-supported guess, INFERRED).**
**Single-sample dips (the tables show medians of five samples, which hide them).** In 3 of the 12 runs single samples fall far below the median: Q8_0 run 6 has samples at 0.28x (pp64, 780 against 2811 tok/s), 0.35x (pp128), 0.43x (pp256) and 0.59x (pp512) of the row median; patched run 5 has samples at 0.42 to 0.86x of the median in all seven rows; patched run 6 at 0.76x in pp1024. Also the first sample of pp16 in every Q8_0 run is at 0.66 to 0.81x of the median (first-call cost). By the mean of the five samples (`avg_ts`, what llama-bench prints), against the median of the six runs' means: Q8_0 run 6 is 15 % low at pp64 and pp128, 11 % at pp256 and 7.5 % at pp512; patched run 5 is 8.6 to 28 % low in every row (pp16: 1.40x slower than normal); patched run 6 is 4.5 % low at pp1024; the other nine runs stay within 3.4 % (Q8_0 run 2 pp16, -3.4 %). So "reproduces every row within 3 %" is true for medians only, and Q8_0 has dips too. This matters for round 1's 2x: it may have been one dip inside a mean row, not a clock effect (INFERRED; I did not recheck which statistic round 1 quoted).
Honest limit: 6 runs per variant; the 2x seen in round 1 itself (two runs hours apart) was not reproduced, the worst row here is 1.33x by the median (1.40x by the mean).

## Item 7a: streams, repeated and interleaved

llama-batched-bench, `-npp 128 -ntg 128 -npl 1..8`, six repetitions of each cell in the order 3B (unmod, patched, Q8_0), 8B (unmod, patched, Q8_0), repeated six times (36 jobs; `r2streams-<config>-r<k>.log`).

Memory clock 1408 MHz in every job.

| config | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 3b fp8-unmod | 76.8 [76-77] | 130.6 [130-131] | 178.0 [177-179] | 220.0 [219-220] | 254.8 [254-255] | 286.0 [285-286] | 313.2 [306-314] | 333.2 [332-334] |
| 3b fp8-patched | 156.4 [155-157] | 266.0 [263-267] | 366.4 [363-368] | 463.0 [461-467] | 566.8 [560-569] | 669.7 [663-673] | 739.9 [732-743] | 829.8 [823-834] |
| 3b q8_0 | 142.9 [142-144] | 256.7 [254-258] | 380.8 [376-382] | 491.2 [485-496] | 602.0 [593-603] | 698.2 [690-700] | 777.4 [769-779] | 807.5 [797-814] |

3b: patched FP8 against Q8_0 per stream count:

| streams | patched | Q8_0 | patched vs Q8_0 | the six repetitions of each |
|---|---|---|---|---|
| 1 | 156.4 | 142.9 | +9.4 % | separate |
| 2 | 266.0 | 256.7 | +3.6 % | separate |
| 3 | 366.4 | 380.8 | -3.8 % | separate |
| 4 | 463.0 | 491.2 | -5.7 % | separate |
| 5 | 566.8 | 602.0 | -5.8 % | separate |
| 6 | 669.7 | 698.2 | -4.1 % | separate |
| 7 | 739.9 | 777.4 | -4.8 % | separate |
| 8 | 829.8 | 807.5 | +2.8 % | separate |

| config | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 8b fp8-unmod | 37.2 [30-37]* | 63.1 [55-63]* | 85.5 [82-86] | 106.1 [102-106] | 123.1 [122-123] | 136.3 [135-137] | 149.8 [148-150] | 160.1 [160-160] |
| 8b fp8-patched | 77.4 [77-78] | 127.3 [126-128] | 181.1 [181-182] | 230.9 [230-232] | 276.3 [275-277] | 341.1 [339-343] | 394.9 [393-397] | 433.1 [425-435] |
| 8b q8_0 | 72.0 [59-72]* | 133.9 [120-135]* | 197.0 [192-199] | 258.0 [246-261] | 312.9 [306-320] | 374.2 [372-377] | 419.8 [418-422] | 450.5 [441-453] |

8b: patched FP8 against Q8_0 per stream count:

| streams | patched | Q8_0 | patched vs Q8_0 | the six repetitions of each |
|---|---|---|---|---|
| 1 | 77.4 | 72.0 | +7.5 % | separate |
| 2 | 127.3 | 133.9 | -4.9 % | ranges overlap |
| 3 | 181.1 | 197.0 | -8.1 % | separate |
| 4 | 230.9 | 258.0 | -10.5 % | separate |
| 5 | 276.3 | 312.9 | -11.7 % | separate |
| 6 | 341.1 | 374.2 | -8.8 % | separate |
| 7 | 394.9 | 419.8 | -5.9 % | separate |
| 8 | 433.1 | 450.5 | -3.9 % | separate |



Round-1 statements: **hold.** 3B: patched leads at 1 stream (+9.4 %) and 2 (+3.6 %), trails Q8_0 from 3 to 7 streams (-3.8 to -5.8 %), and leads again at 8 (+2.8 %); in all 3B rows the six repetitions of the two builds do not overlap, so the 8-stream
lead is real, though small (round 1 had +1.9 %). 8B: patched leads only at 1 stream (+7.5 %); at 2 streams the ranges overlap (-4.9 %); from 3 on Q8_0 leads by 3.9 % (8 streams) to 11.7 % (5 streams). 3B Q8_0 from 7 to 8 streams: 777.4 to 807.5 tokens/s, +3.9 %: the flattening is real
and smaller than round 1's single run said it was (+4.9 %). Spreads (max minus min over the median): 0.5 to 2.6 % for the 3B and 0.2 to 5.6 % for the 8B above 2 streams (8B Q8_0 at 4 and 5 streams 5.6 and 4.5 %); cells with a spread above 6 % are marked, all of them are the 8B Q8_0 and unmodified FP8 at 1 and 2 streams (11.2 to 20.2 %; some repetitions were much slower: 59 against 72 tokens/s; the first-call cost of round 1's "Smaller points"). The round-1 known answer for the 3B Q8_0 single stream (98.5 tokens/s, noisy) is replaced by 142.9 [142 to 144].

## Item 7b: more matrix shapes in `test-backend-ops perf`

**What blocked it.** The tool's perf list has one shape family (m = 4096, k = 14336). **Now:** `scripts/llm2-test-shapes.patch` (15 added lines in `tests/test-backend-ops.cpp`) adds MUL_MAT cases for every (m, k) of the 3B and 8B (q/o, k/v, gate/up, down, output),
n = 1, 8, 32, 512, 2048, f8_e4m3, f16, q8_0 (150 cases). It was applied to a separate clone of the unmodified tree (`llama.cpp-llm2-shapes` next to `llama.cpp-latest` under the build root; HEAD f477c1b90), and only `test-backend-ops` was built, for gfx1201 with ROCm 10.1, in the copy's
own folder (nice, -j8, 10 GB cap). **Proof that only test code changed (MEASURED, `r2build-diff.txt`):** all 4189 files that git tracks in the original tree compared byte for byte with the copy except `tests/test-backend-ops.cpp`: 0 differ, 0 files under `ggml/` differ. The check was made to FAIL
once on purpose (one line appended to `ggml/CMakeLists.txt` in the copy, the phase refused, the copy was restored). **The evidence of that FAIL is the summary line only**: `break-r2build/summary.md` shows "the copy differs from the original outside tests/test-backend-ops.cpp ... (FAILED)", but the failing diff text was not kept (`break-r2build/r2build-diff.txt` is the later passing rerun, "0 differ"; "DIFFERS: <file>" is the format the check prints, not a kept line), and the first false alarm printed the same summary line. The check compares the files git tracks in the original tree; a new untracked file in the copy is listed in the file's status lines but not compared. Rebuild from the patch: `git apply scripts/llm2-test-shapes.patch` in a fresh clone, then `scripts/study-llm.sh r2build`. Three repetitions (`r2shapes-r1..3`, 159 result lines each: 150 of ours and 9 of the tool's own m = 4096, k = 14336 cases, which repeat one of ours and are averaged with it).

3B:

| m x k | n | f8_e4m3 us | f16 us | q8_0 us | f16 TFLOPS | q8_0 TFLOPS | f8 - f16 us | implied widen GB/s (3 m k / diff) |
|---|---|---|---|---|---|---|---|---|
| 3072 x 3072 (q, o) | 1 | 35.1 | 29.3 | 13.9 | 0.6 | 1.4 | 5.8 | - |
| 3072 x 3072 (q, o) | 8 | 69.5 | 13.1 | 21.2 | 11.5 | 7.1 | 56.4 | - |
| 3072 x 3072 (q, o) | 32 | 42.6 | 16.6 | 19.1 | 36.5 | 31.6 | 26.1 | 1086 |
| 3072 x 3072 (q, o) | 512 | 109.8 | 89.9 | 102.5 | 107.5 | 94.3 | 19.9 | 1424 |
| 3072 x 3072 (q, o) | 2048 | 375.9 | 369.0 | 386.2 | 104.8 | 100.1 | 6.9 | 4079 |
| 1024 x 3072 (k, v) | 1 | 21.9 | 17.4 | 7.9 | 0.4 | 0.8 | 4.6 | - |
| 1024 x 3072 (k, v) | 8 | 30.2 | 7.4 | 11.9 | 6.8 | 4.2 | 22.8 | - |
| 1024 x 3072 (k, v) | 32 | 19.3 | 9.6 | 16.2 | 21.0 | 12.4 | 9.7 | 971 |
| 1024 x 3072 (k, v) | 512 | 50.0 | 42.9 | 46.4 | 75.1 | 69.4 | 7.1 | 1322 |
| 1024 x 3072 (k, v) | 2048 | 138.2 | 138.6 | 148.1 | 93.0 | 87.0 | -0.4 | - |
| 8192 x 3072 (gate, up) | 1 | 88.3 | 49.5 | 29.2 | 1.0 | 1.7 | 38.9 | - |
| 8192 x 3072 (gate, up) | 8 | 175.5 | 31.8 | 47.1 | 12.6 | 8.5 | 143.6 | - |
| 8192 x 3072 (gate, up) | 32 | 138.3 | 33.2 | 32.0 | 48.5 | 50.3 | 105.1 | 718 |
| 8192 x 3072 (gate, up) | 512 | 343.2 | 253.5 | 250.3 | 101.7 | 103.0 | 89.7 | 842 |
| 8192 x 3072 (gate, up) | 2048 | 959.1 | 911.0 | 1019.8 | 113.2 | 101.1 | 48.1 | 1569 |
| 3072 x 8192 (down) | 1 | 76.4 | 41.2 | 22.2 | 1.2 | 2.3 | 35.2 | - |
| 3072 x 8192 (down) | 8 | 135.9 | 48.1 | 56.1 | 8.4 | 7.2 | 87.8 | - |
| 3072 x 8192 (down) | 32 | 144.6 | 43.1 | 52.0 | 37.3 | 31.0 | 101.4 | 744 |
| 3072 x 8192 (down) | 512 | 450.8 | 355.4 | 263.6 | 72.5 | 97.7 | 95.4 | 791 |
| 3072 x 8192 (down) | 2048 | 1216.5 | 1116.1 | 1082.8 | 92.4 | 95.2 | 100.4 | 752 |
| 128256 x 3072 (output) | 1 | 1616.6 | 1161.8 | 639.0 | 0.7 | 1.2 | 454.9 | - |
| 128256 x 3072 (output) | 8 | 3002.9 | 1175.9 | 763.9 | 5.4 | 8.3 | 1827.1 | - |
| 128256 x 3072 (output) | 32 | 3019.4 | 1259.6 | 682.5 | 20.0 | 36.9 | 1759.8 | 672 |
| 128256 x 3072 (output) | 512 | 4436.1 | 2964.1 | 3779.5 | 136.1 | 106.7 | 1472.0 | 803 |
| 128256 x 3072 (output) | 2048 | 12232.6 | 11204.1 | 14517.0 | 144.0 | 111.2 | 1028.6 | 1149 |

8B:

| m x k | n | f8_e4m3 us | f16 us | q8_0 us | f16 TFLOPS | q8_0 TFLOPS | f8 - f16 us | implied widen GB/s (3 m k / diff) |
|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 (q, o) | 1 | 57.1 | 36.6 | 18.1 | 0.9 | 1.9 | 20.6 | - |
| 4096 x 4096 (q, o) | 8 | 107.9 | 22.9 | 34.9 | 11.7 | 7.7 | 85.0 | - |
| 4096 x 4096 (q, o) | 32 | 67.5 | 24.2 | 24.9 | 44.4 | 43.2 | 43.3 | 1162 |
| 4096 x 4096 (q, o) | 512 | 238.1 | 167.9 | 173.6 | 102.3 | 99.0 | 70.1 | 718 |
| 4096 x 4096 (q, o) | 2048 | 698.0 | 672.0 | 712.7 | 102.3 | 96.4 | 25.9 | 1940 |
| 1024 x 4096 (k, v) | 1 | 24.6 | 18.6 | 8.3 | 0.5 | 1.0 | 6.0 | - |
| 1024 x 4096 (k, v) | 8 | 33.7 | 9.9 | 13.7 | 6.8 | 4.9 | 23.9 | - |
| 1024 x 4096 (k, v) | 32 | 24.0 | 10.9 | 19.6 | 24.6 | 13.7 | 13.1 | 958 |
| 1024 x 4096 (k, v) | 512 | 66.4 | 53.2 | 60.2 | 80.7 | 71.3 | 13.2 | 954 |
| 1024 x 4096 (k, v) | 2048 | 226.3 | 173.7 | 199.3 | 98.9 | 86.2 | 52.6 | 239 |
| 14336 x 4096 (gate, up) | 1 | 189.0 | 176.9 | 51.3 | 0.7 | 2.3 | 12.1 | - |
| 14336 x 4096 (gate, up) | 8 | 364.6 | 179.7 | 100.1 | 5.2 | 9.4 | 184.9 | - |
| 14336 x 4096 (gate, up) | 32 | 465.9 | 184.9 | 64.1 | 20.3 | 58.7 | 281.0 | 627 |
| 14336 x 4096 (gate, up) | 512 | 718.9 | 499.7 | 592.5 | 120.3 | 101.5 | 219.2 | 804 |
| 14336 x 4096 (gate, up) | 2048 | 2387.8 | 2205.0 | 2332.4 | 109.1 | 103.1 | 182.8 | 964 |
| 4096 x 14336 (down) | 1 | 162.1 | 175.7 | 41.9 | 0.7 | 2.8 | -13.6 | - |
| 4096 x 14336 (down) | 8 | 285.5 | 172.9 | 103.6 | 5.4 | 9.1 | 112.6 | - |
| 4096 x 14336 (down) | 32 | 442.6 | 186.2 | 73.7 | 20.2 | 51.0 | 256.4 | 687 |
| 4096 x 14336 (down) | 512 | 789.9 | 551.1 | 637.0 | 109.1 | 94.4 | 238.8 | 738 |
| 4096 x 14336 (down) | 2048 | 2278.0 | 2249.5 | 2505.1 | 106.9 | 96.0 | 28.4 | 6192 |
| 128256 x 4096 (output) | 1 | 2007.3 | 1530.9 | 810.6 | 0.7 | 1.3 | 476.4 | - |
| 128256 x 4096 (output) | 8 | 3611.9 | 1560.6 | 1199.8 | 5.4 | 7.0 | 2051.4 | - |
| 128256 x 4096 (output) | 32 | 3971.1 | 1645.5 | 890.3 | 20.4 | 37.8 | 2325.6 | 678 |
| 128256 x 4096 (output) | 512 | 6610.2 | 4579.5 | 5048.6 | 117.5 | 106.6 | 2030.7 | 776 |
| 128256 x 4096 (output) | 2048 | 19027.6 | 17393.3 | 19209.6 | 123.7 | 112.0 | 1634.3 | 964 |



**A3 redone (f8 = f16 + widen, per shape).** Reading the last column: n = 1 and 8 run the decode kernels (no widen): FP8 is 2.4 to 3.9 times slower than Q8_0 and often slower than f16, as in round 1. For n >= 32 the column is 3 m k / (f8 - f16).
At **n = 512** the matrices of 17 million elements and up give 718 to 842 GB/s (3B gate/up 842, down 791, output 803; 8B gate/up 804, down 738, output 776, q/o 718), within 10 % of the 791 GB/s of the 8B trace widen: A3 holds for them. The 3B q/o (9.4 million elements) gives 1424 and the small k and v matrices 954 to 1322, above any
DRAM rate (the widened matrix fits in cache), so A3 fails for them as expected. At **n = 32 and n = 2048 the column swings** (n = 2048: 1569, 4079, 6192, 239, ...): a widen costs the same at every n, so the difference f8 - f16 must have another part, and the trace says what
(`r2sht-*`, m = 8192, k = 3072, `r2build` binary, per-kernel medians):

| case | widen kernel | FP16 GEMM kernel | activation convert | tile |
|---|---|---|---|---|
| f8_e4m3, n = 512 | 99.4 us | 214.0 us | 16.5 us | MT64x64x64 |
| f16, n = 512 | - | 228.3 us | 13.2 us | MT64x64x64 |
| f8_e4m3, n = 2048 | 97.0 us | 747.3 us | 69.0 us | MT128x128x32 |
| f16, n = 2048 | - | 796.1 us | 57.0 us | MT128x128x32 |

The widen kernel takes the same 97 to 99 us at both n (75.5 MB in 97 us = 778 GB/s) and the **same GEMM kernel runs 14 us (6 %) faster at n = 512 and 49 us (6 %) faster at n = 2048 after a widen than on a stored F16 matrix**, so
f8 - f16 = widen - GEMM gain + convert difference: 99.4 - 14.3 + 3.3 = 88.4 us (measured difference 89.7) at n = 512 and 97.0 - 48.8 + 12.0 = 60 us at n = 2048 against 48.1 measured in the perf run (25 % off: the traced run and the perf run are different runs of the same case). MEASURED. Why the GEMM is faster after a widen is INFERRED: the freshly written
50 MB matrix is still in the memory-side cache (item 4's probe shows writes of that size stay cached); the f16 test's matrix is read from memory. So A3 holds as "f8 = f16 + widen - a cache gain on the GEMM", and the cache gain grows with n. Consequence for llama.cpp's FP8 prompts: part of
the widen cost is paid back by the GEMM that follows it, more at larger micro-batches (consistent with item 5). Dependent claims of round 1 that used the "widen = f8 - f16" reading: A3's 732 GB/s (one shape) becomes 718 to 842 at n = 512 for the biggest matrices.

## Deliberate breaks and selftest (round 2)

All run by `scripts/study-llm.sh selftest` (about 1.5 minutes, no GPU; timed once in the fix round: 90 s); every break must FAIL and every control pass. New in round 2, FAIL seen yes for each:
1. `llm_ref.py`: a reference with one bit changed in the last row, a reference rounded through F16, damage in a middle row (all rows sampled), a scale with 24 significant bits (F32 not exact), and one ulp apart in a tensor of the 8B comparison; controls: the exact file passes.
2. `llm_kernels.py`: overlap finder (a clean trace must give no hit, a planted 300 ns overlap on the same queue and on another queue is found with the right ids, two kernels inside one long kernel are both found, the neighbours-only version finds one), counter summary
   (a counter that reads 0 on a kernel that moves data is NONFUNCTIONAL, no rows for the named kernel fails).
3. `llm_tables.py`: the item 1 table (Q8_0 label with the FP8 file, patched label with the unmodified build, 4 chunks where 50 are claimed, a reference run of 4 chunks, comparisons made against another reference, the reference run loading a Q8_0 file, one job at 1258 MHz), the streams table
   (missing repetition, wrong file under a label, one repetition at another memory clock), the shapes table (a missing type, two of three repetitions; a known answer: 302/102 us, 8192 x 3072 gives 377 GB/s), the clock windows (three back-to-back tests, a shifted window, a window outside the recording, Pearson sign).
4. `r2build`: the "only the test file differs" proof FAILs when one line of `ggml/CMakeLists.txt` is changed (done by hand once, restored; evidence is the summary line only, see item 7b).
5. Fix round (see below): a dummy job allocating 800 MB under a 300M memory cap is killed (control: 100 MB under 300M and 800 MB under 2G pass; mutation: with the cap removed the break goes red); `widen_cache` known answer on 8 MiB instead of 1 GiB FAILs; a wrong ISA expectation for its loops FAILs.
**Mutation check** (each new check removed in a copy and the selftest went red every time, `runs/.../mutation/`): the running-maximum in the overlap finder, the "report a hit" condition, the dead-counter flag, the chunk-count check, the same-reference check, the label-file check of the quality table, and the bit compares of `llm_ref.py` (both weakened). A first
mutation attempt of the table checks tested nothing (the copies lived outside `scripts/` and could not import their helper); found, moved, repeated.

## Fix round (after the review of round 2)

Run folder `runs/20261118T165826Z-llm2fix/` (git-ignored): every GPU job went through the gated runner (shared lock, `nice`, device index from the card record, game check, LACT profile "Testing" only, hotspot start limit 70 C, memory clock logged;
all with exit as stated below, memory clock 1408 MHz before and after, highest hotspot 72 C). LACT and sysfs settings were not touched.

**Memory cap (review fix 8).** Every GPU job of `scripts/study-llm.sh` now runs in a systemd user scope with `MemoryMax=10G` and `MemorySwapMax=0` (the user systemd instance works on this machine); if no scope can be made the job is not started (exit 9), so a job never runs uncapped.
A job killed before its time limit is logged ("MEMORY CAP", exit 137) and summarised. Selftest: a dummy job touching 800 MB under a fake 300M cap is killed (exit 137); controls pass (100 MB under 300M; 800 MB under 2G); with the scope removed from the job command (a mutated copy) the break goes red, as it should.
Real job under the cap: the 3B Q8_0 `llama-bench` known answer (`fixbench-3b-q8_0.log`, `.meta` records `mem_cap=10G swap_cap=0`, 4.5 s, exit 0): pp512 12785 tokens/s against round 1's 12827 (-0.3 %), tg128 145.2 against 145.9 (-0.5 %).
**The cap was not in force during round 2**: the two 8B F32 attempts (and the F16 reference run) ran without it; the 10 GB figure is the machine rule, the runner did not enforce it then.

**`src/widen_cache.hip` (review fix 7).** It now (1) reads the whole 1 GiB pool before the rows and must land within 10 % of the 706 GB/s of the study's bandwidth probe (round 1, `runs/20261118T094213Z-llm/probe-bw-plain.log`: 705.0 and 705.7; `scheduling-and-clocks.md`: best 706.4);
VERDICT lines, exit 1 on FAIL: the fix-round runs read 703.0, 702.6, 702.7, 702.5 and 701.4 GB/s, PASS; the deliberate break (argument `break`, 8 MiB instead of 1 GiB) reads 371.2 GB/s, VERDICT known-answer FAIL, exit 1 (`r2mallcheck-break.log`). (2) times an empty kernel as the launch floor (6.8 to 7.0 us here) and marks rows where it is 10 % or more of the time LAUNCH-BOUND, with a net rate (floor taken out; INFERRED, the floor is not exactly additive) beside the raw one.
(3) ISA of the three timed loops (`scripts/mem-expect/widen_cache.txt`, `r2mallbuild-loopcheck.txt`, `scripts/mem_loopcheck.py`): widen loop 1 `global_load_b128` and 2 `global_store_b128` (16 VGPRs), read 1 load (10), write 1 store (8), no scratch, no other global access; a wrong expectation (3 stores) FAILs (`r2mallbuild-loopcheck-break.txt`).
`S_MiB` in the new output is the MiB read; the bytes moved are `moved_MiB` (the round-2 logs printed 3 S for a widen row).
Rerun (one repetition of 200 calls per size, `r2mall-*-r1.log` in the fix folder; raw GB/s, net in brackets, LB = LAUNCH-BOUND):

| S read / written (MiB) | widen, destination reused | widen, destination rotating | read only | write only, reused | write only, rotating |
|---|---|---|---|---|---|
| 8 / 16 | 655 [799] LB | 503 [584] LB | 294 [388] LB | 1033 [1808] LB | 530 [679] LB |
| 25 / 50 | 638 [676] | 607 [641] | 529 [615] LB | **1528** [1916] LB | 609 [662] |
| 50 / 100 | 625 [643] | 625 [643] | 622 [678] | 623 [650] | 624 [651] |
| 100 / 200 | 622 [631] | 620 [629] | 657 [686] | 632 [646] | 631 [645] |

What moved against the table of item 4: the write-only absorption holds (1528 against 609 at S = 25; 1033 against 530 at S = 8) and the 100 MiB row is flat as before. The widen row at S = 25 moved: the gain of the reused destination is 12 % in round 2 (681 against 610), 9.1 and 8.6 % when the round-2 binary was run twice in the fix round (663.5 against 608.1, 660.0 against 607.6; `r2mall-25-round2binary-*.log`),
and 5.1 and 4.3 % with the new binary (637.7 against 606.6; 635.6 against 609.4). Why the new binary reads about 4 % lower than the old one in the same session is not examined (INFERRED: the 1 GiB known-answer read and the empty launches before the first row change the state of the cache or clocks). *(Round 3, Q3: MEASURED, the 1 GiB known-answer read before the rows is the cause; the kernels' machine code is byte-identical. See `llm-workload-3.md`.)* So "a widen-shaped kernel gains 9 to 27 %" is withdrawn: **4 to 12 % at S = 25, about 30 % raw (37 % net) at S = 8, where the rows are launch-bound (18 % of the time for the widen rows) and the best and worst sample differ by 44 %; the S = 8 gain is the least certain number of the item.**
The probe's best widen rate is 638 to 681 GB/s, still below the 770 to 800 of the traced kernels.

**Review fixes 1 to 6 and 9 to 10, and the number check.** Wording, arithmetic and evidence corrections are in the places they belong (perplexity headers and deltas, the verdict column, item 3's strength, item 5's arithmetic, item 6's dips, the 8B F16 reference label, the r2build break evidence, the round-1 notes in `llm-workload.md`);
all numbers of the review's check list were re-derived against the raw files first. Facts seen live but never logged (page faults, I/O pressure, swap use, memory held by other programs, the disk peak) are marked "seen live, not logged" or removed.

## Parked, skipped and open

- **8B exact (F32) quality reference: not evaluated** (item 1): built, verified and deleted; the machine's memory does not allow it with partial offload (the two attempts ran without the 10 GB cap that the fix round added). *(Round 3: a mixed F16/F32 reference is 27.9 GB and does not fit either.)* An option the owner could allow: run it when nothing else is open (no browser, no other agents), or on a machine with 64 GB.
  Until then the 8B reference is F16, whose error is measured on the 3B only.
- **8B F16 reference is not bit-checked** against the checkpoint (item 1); an F16 variant of `compare8b` is needed before it is used again.
- **The cache gain of a widen-shaped kernel at S = 25 moves between runs** (4 to 12 %, see "Fix round"); the write-only 1524 against 611 does not. Why the new binary reads about 4 % lower than the round-2 binary in the same session is not examined. *(Round 3: answered, the known-answer read before the rows; the cause of that state effect is still INFERRED.)*
- **FP8 decode kernel's extra 57 W (INFERRED, untested)**: instruction counters read 0; reading the kernels' ISA would give a static VALU count per byte (not done). *(Round 3: done, see `llm-workload-3.md` Q1.)*
- **Where the 8B pp16 to pp64 2x came from**: not reproduced (worst here 1.33x, a host-load cause is the best guess); an interleaved long soak with the machine in use is the way.
- **Why the library picks the 16 x 16 tile at K = 512 (item 2)**: not looked into (heuristic of the library).
- **Whether the cache that absorbs writes is the L2 or the 64 MB cache** (item 4): the size dependence points to the 64 MB cache; no direct way found.
- **Rows at 1258 MHz memory**: out of scope (needs the displays blanked).
- The first `r2build` attempt of the run folder reports "differs ... (FAILED)": that was a false alarm of my first version of the check (it compared untracked files); the check was rewritten to compare git-tracked files and the second run passes (both records are in `summary.md`).
- `summary.md` of the folder also holds the FAILED line of the killed 8B F32 attempt (item 1).

## How to rerun

```
R2_8B_NGL=27 scripts/study-llm.sh r2q   # needs refs/ built with toolchains/convert-venv/bin/python scripts/llm_ref.py (make3b, make3b ... f16, compare3b/8b) and the converter, see item 1
scripts/study-llm.sh r2libs       # item 2
scripts/study-llm.sh r2overlap    # item 3 (13 traces), then python3 -I scripts/llm_kernels.py overlap <csv...>
scripts/study-llm.sh r2counters && scripts/study-llm.sh r2mallbuild && scripts/study-llm.sh r2mallcheck && scripts/study-llm.sh r2mall    # item 4 (r2mallbuild makes build/gfx1201-rocm10/widen_cache from src/widen_cache.hip and checks its ISA)
scripts/study-llm.sh r2ub         # item 5
scripts/study-llm.sh r2clock      # item 6, then python3 -I scripts/llm_tables.py r2clock <run>
scripts/study-llm.sh r2streams    # item 7a, then python3 -I scripts/llm_tables.py r2streams <run>
scripts/study-llm.sh r2build && scripts/study-llm.sh r2shapes && scripts/study-llm.sh r2shapes-trace   # item 7b, then llm_tables.py r2shapes <run>
```
All with `LLM_RUN=runs/<folder>` to continue in one folder. Do not edit `scripts/study-llm.sh` while a phase runs: bash reads the script as it goes and the locked runner re-enters the same file.

## Proposed lines for `notes/STUDY-PLAN.md` and `README.md` (not applied; those are the lead's files)

**STUDY-PLAN.md, status paragraph (add after the LLM data dive line):**

> LLM data dive, round 2 done (branch `study/llm-dive`, run `runs/20261118T151433Z-llm2`, `measurements/llm-workload-2.md`, expectations in `llm-workload-2-expectations.md` and `-7.md`, written before the GPU jobs): exact-weights quality
> reference for the 3B (FP8 routes 3e-6 KL divergence from exact, Q8_0 7e-4; the 8B only against an F16 reference, its F32 version could not be run on this machine), the first-hour 37 TFLOPS explained (the shape), the no-overlap FAIL decided (timestamp artifact),
> the counters tried (mostly non-functional on gfx1201), the micro-batch 2048 gap explained, repeated streams and the real weight shapes in `test-backend-ops perf` (a separate patched copy of llama.cpp, tests only). Memory clock 1408 MHz throughout.

**README.md, key-finding bullet:**

> - **The 37 TFLOPS FP16 GEMM of the first hour was the shape, and the exact-weights check agrees with round 1.** The same hipBLAS call runs at 108 to 136 TFLOPS at llama.cpp's shapes; at K = 512 with a large output the library picks a 16 x 16 tile (36.5).
>   Against an exact F32 copy of the 3B's weights, FP8 prompts differ by a mean KL divergence of 3e-6 and Q8_0 by 7e-4. Writes to a reused buffer of up to about 50 MiB stay in the cache (1524 against 611 GB/s), which is, INFERRED, why the GEMM right after a widen runs 6 % faster.
>   ([llm-workload-2.md](llm-workload-2.md))

**README.md, measurement-file table rows:**

> | `llm-workload-2.md` | LLM data dive, round 2: exact-weights quality reference, why the first-hour GEMM number was low, overlap and counters, micro-batch gap, repeated streams, real weight shapes |
> | `llm-workload-2-expectations.md`, `llm-workload-2-expectations-7.md` | The expectations for round 2, written before the GPU jobs of each item |
