# LLM data dive: what llama.cpp shows about the RX 9070 XT

Measured 2026-11-18 on the LACT profile "Testing" (stock core clock, 317 W power cap), with the memory clock reading
1408 MHz in every job (held there by the displays; the other study files were taken at 1258 MHz), performance level
auto. A regulator-level power-limit mod (1.2x) is in the card, so reported watts are scaled down on the main rail and
are not board power. Every wall-clock number in this file is for that configuration. Software: llama.cpp master
f477c1b90 built with ROCm 10.1 (an unmodified build and a build that adds an FP8 decode kernel, called "patched"
below), models Llama 3.2 3B and Llama 3.1 8B as FP8, as Q8_0 and ("auto") FP8 everywhere. Nothing in llama.cpp was
changed and nothing here tries to make it faster: llama.cpp is the load, the card is the subject. The expectations
for each question were written before any GPU job and are in `llm-workload-expectations.md`; this file marks each
right or wrong. Labels: MEASURED (seen in a run, the run file is named), READ (from source or a listing), INFERRED
(reasoning; an explanation nobody tested is INFERRED).

Every number below is from the rerun of the committed code (run `20261118T110108Z-llm`); the first run
(`20261118T094213Z-llm`, made with the first version of the scripts) agreed within its spread on the headline numbers,
and the section "Rerun" says what moved. Where a quoted value differs from the first run's, the first run's value is
given in brackets.

## Summary: what the card did under an LLM

1. **FP8 prompts are slow because of the route, not the format, and not the matrix hardware.** On this card
   llama.cpp sends Q8_0 prompts to an integer matrix kernel (MMQ), and FP8 (which cannot use it) to a chain: widen
   every weight matrix to FP16 into a scratch buffer, convert the activations to FP16, run one hipBLASLt FP16
   GEMM, multiply by a scale. In a 3B prompt pass of 512 tokens the FP16 GEMMs took 26.9 ms against 25.6 ms for MMQ;
   the whole difference between FP8 (57.5 ms) and Q8_0 (39.7 ms) is the extra passes over memory: widening 10.5 ms, the
   activation convert 2.8 ms, the scale multiply 3.4 ms (MEASURED, kernel traces). Strong evidence that it is the
   route (the numbers are MEASURED, the conclusion is INFERRED): Q8_0 forced through the same dequantize-then-hipBLAS
   route (a compile-time switch) lands at 53.5 ms, equal to FP8 patched (53.6 ms), and at 9573 tokens/s against FP8
   patched's 9521 (MEASURED; first run 9611 against 9529).
2. **FP8 gets closer to Q8_0 with bigger batches (MEASURED); that the widen is paid per call is INFERRED, and explains
   only about 60 % of it.** With a 2048-token micro-batch the FP8 patched prompt speed is 95 % of Q8_0's (3B; first run
   97 %), against 77 % at 512 and 53 % at 16 tokens (MEASURED). FP8 patched needs 53.8 ms per 512 tokens at a micro-batch
   of 512 and 42.2 ms at 2048 (from tokens/s), a saving of 11.6 ms; spreading the traced 9.05 ms widen over four times
   fewer calls would save 6.8 ms. The other 4.8 ms has no tested explanation (a better GEMM rate at N = 2048 is a guess).
   > **Updated in round 2 (item 5):** settled by a per-group trace comparison. The 11.6 ms are round 1's number from unprofiled tokens/s; round 2's unprofiled controls give 11.2 ms (52.96 against 41.74) and the traced passes 12.8 ms per 512 tokens (profiling adds about 1.6). The traced 12.8 are widen 6.8 (confirmed) + FP16 GEMM 5.2 (103.5 to 127.5 TFLOPS) + launch gaps 1.95 + small matvec kernels 0.6, minus attention 1.1 and activation convert plus elementwise 0.6. The guess was right. See `llm-workload-2.md`.
3. **The FP16 matrix routine is much faster than the first hour suggested.** The GEMMs llama.cpp actually runs reach
   78 to 135 TFLOPS (hipBLASLt kernels), not the 37 TFLOPS of the first-hour library probe: up to 64 % of the 211 TFLOPS
   FP16 peak (the 8B gate and up matrices, 134.7). The int8 kernel reaches 82 to 114 TOPS, 20
   to 27 % of the 420 TOPS peak (about a quarter) (MEASURED; the peaks are READ from the expectations file).
   > **Updated in round 2 (item 2):** the first-hour probe's shape (K = 512, 4096 x 4096 F32 output) makes the same library pick a 16 x 16 tile (36.5 TFLOPS); llama.cpp's shapes get 64 x 64 and 128 x 128 tiles (108 to 136). One library and kernel family; the cause is the shape (MEASURED).
4. **Generation uses 71 to 83 % of the read rate, counting the bytes decode streams: 73 and 83 % for FP8 patched
   (3B, 8B), 71 and 82 % for Q8_0.** The bytes are the GGUF tensors minus `token_embd.weight`, which decode does not
   stream (the first run divided by the whole file and got 80 to 95 %, which overstated it; see Rerun). The big weight
   kernels reach 95 to 99 % of the bus; the rest is the small kernels and the 2 to 3 us gaps between the 425 to 614
   kernels of a token (5 to 21 % of a token, with the profiler). The gaps are 1.8 to 1.9 us without the profiler
   (INFERRED from unprofiled token time; first run 1.8 to 2.1) and 2.9 to 3.0 us with it. HIP graphs do not hide kernels
   from the profiler.
5. **The card runs against its power limit in prompts and also in FP8 generation.** Sustained prompt loops sat at the
   317 W cap (power-limit flag PPT0 in 89 to 91 % of samples; first run 77 to 92 %) with the core at 2.94 to 3.08 GHz
   (median while awake), hotspot up to 90 C. The reported power peaks of 382 to 399 W in those loops are instantaneous
   readings; the cap acts on an average (INFERRED). Patched-FP8 generation of the 8B also sat at the cap (316 W) at
   3.31 GHz, while the 8B Q8_0 generation drew 259 to 260 W at 3.39 GHz (the 3B Q8_0 248 W) at nearly the same bandwidth (584 against 581 GB/s on
   the 8B, 516 against 503 on the 3B): the FP8 decode kernel costs 56 to 57 W more for the same bytes (MEASURED; why is
   INFERRED, see "Not measured").
6. **Same program, same answer.** Two identical 50-chunk perplexity runs wrote logits files that `cmp` found
   byte-identical on both models (3270732420 bytes each; sizes, sha256 and the `cmp` exit status are kept in
   `det-*.compare.log`). The saved files are 16-bit compressed, so that is the strict statement. Compared with its own
   saved file, the same run gives mean KL divergence 0.000000 (six decimals) but a 99th percentile of 3.4e-5 and a maximum
   of 6.3e-5: the noise floor of the compressed file (INFERRED), which the pipeline's check allows for (mean below 1e-6,
   maximum below 2e-4, same top token at least 99.9 %). Against the unmodified FP8 run, the other FP8 kernels differ by a
   mean KL divergence of 3e-6 to 7e-6; Q8_0 (activations rounded to int8) by 6e-4 to 7e-4, about 90 to 240 times more.

Smaller points: the first prompt after loading a model ran 4.8 times (FP8) or 2.1 times (Q8_0) slower than the later
ones (first run 5.2 and 2.3). After 2 s of idle the following prompts ran 5 % (FP8) and 10 % (Q8_0) slower than in a
back-to-back loop (medians of tests 2 to 6 against tests 3 to 12; the core sat at 2.79 GHz median in those recordings
against 3.07 to 3.11 in the loop), while generation after 2 s idle ran as fast as in the grid (159.0 against 158.7 and
146.0 against 146.9 tokens/s). The large first-call penalty after loading is first-call set-up, not the clock ramp
(INFERRED: idle did not repeat it). The memory clock never moved from 1408 MHz in all 130 GPU jobs of the run.

## Expectations against results

Each line is the expectation from `llm-workload-expectations.md`, then what was measured. "Right" and "wrong" are
judged against the numbers written there, not softened afterwards.

| # | Expectation | Result | Verdict |
|---|---|---|---|
| A1 | The 3B pass gap FP8 against Q8_0 is about 15.5 ms; widening alone explains most of it (about 13 ms); under 6 ms would mean the story is wrong | Gap 17.8 ms in the traces (57.5 against 39.7). Widen 10.5 ms (59 % of the gap), activation convert 2.8 and scale multiply 3.4 more; FP16 GEMM against MMQ only +1.3 ms. Widen above 6 ms, so the story holds, but widen is 10.5, not 13 | right in kind, 19 to 20 % low on the number |
| A1 | Widen cost is per call: FP8's loss shrinks with prompt size, largest at pp32 | FP8 patched / Q8_0: 53 % at pp16, 55 % at pp32, 70 % at pp128, 77 % at pp512, 95 % at pp2048, 103 % at pp4096 (3B). The shrinking is MEASURED; that the widen is per call is INFERRED and accounts for about 60 % of the 2048 gain (summary 2). The loss is not largest at pp32 but flat from pp16 to pp32 | right on the shrinking, the mechanism only partly |
| A2 | GEMM rate well below the 123 TFLOPS of PyTorch, in the range 37 up to about 80 TFLOPS; none near peak | FP16 GEMMs 78 to 135 TFLOPS: most shapes above 80, the 8B gate and up matrices above PyTorch's 123, up to 64 % of the 211 TFLOPS peak. The 37 TFLOPS of the first-hour probe is not what llama.cpp's route runs | wrong on the range and on "none near peak" (64 % is as near as the study's own PyTorch result, 58 %) |
| A2 | Q8_0 MMQ a modest share of the 420 TOPS peak, at most 70 TOPS averaged over the pass | MMQ matrices 82 to 114 TOPS (20 to 27 % of 420) while running; averaged over the pass 2.9 TFLOP in 39.7 ms = 73 TOPS | right (the 70 is slightly passed) |
| A3 | f8 time = f16 time + widen time, widen matching A1 within 25 % | Only one shape exists in the tool (m=4096, k=14336, n=512): f8 792.3 us, f16 551.7 us, widen 240.6 us = 732 GB/s; the 8B trace widen is 791 GB/s over all matrices (8 % apart) | right (one shape only) |
| A4 | A hipBLASLt-FP8-class GEMM (170 TFLOPS) would take a small fraction of the measured GEMM time | 3B: 2.9 TFLOP at 170 TFLOPS = 17 ms against 26.9 ms measured (63 %, not a small fraction). Without the widen (10.5 ms) a pass would be about 37 ms against Q8_0's 39.7 (INFERRED, arithmetic only; the activation quantize is not counted) | wrong ("small fraction"); the conclusion that the routine, not the format, is the loss holds |
| A5 | Q8_0 via hipBLAS lands within 15 % of FP8's prompt speed | 3B pp512 9573 against FP8 patched 9521 (+0.5 %) and unmodified 8975; 8B 4793 against 4676 and 4578. Trace: 53.5 ms, the same as FP8 patched (53.6) | right |
| B1 | Q8_0 and patched-FP8 generation are 85 to 95 % of the bus | Counting the bytes decode streams (tensors minus `token_embd`) against the 706 GB/s read rate: patched FP8 73 % (3B) and 83 % (8B); Q8_0 71 % (3B) and 82 % (8B). The first run's 91, 95, 80 and 88 % divided by the whole file and are withdrawn | wrong for all four (71 to 83 %); the per-matrix kernel rates (95 to 99 %) are unaffected |
| B1 | At 1408 MHz generation is +9 to +11 % over the 1258 MHz run | tg128 against the first-hour table (READ, `notes/first-hour-9070xt-rocm10.md`; that run had the core overclocked, so it is not like for like): Q8_0 +8.3 %, patched FP8 +8.3 %, unmodified FP8 +2.2 % (compute-bound kernel) | wrong, about 1 point low for the two memory-bound kernels, and the unmodified one gains almost nothing |
| B1 | Unmodified FP8 generation about half of patched | 78.7 against 158.7 (3B), 37.6 against 77.2 (8B): 0.50 and 0.49 | right |
| B2 | S_TG nearly linear while weights dominate, bending where the generic decode kernel's column loop costs; Q8_0 against FP8 crossover is the interesting number | 3B patched FP8 scales 5.3x from 1 to 8 streams. It leads Q8_0 at 1 and 2 streams (158.1 against 145.0, 268.4 against 258.7), trails from 3 to 7 (367.0 against 383.0 at 3, 740.5 against 777.4 at 7) and leads again at 8 (831.1 against 815.3). The 8B patched FP8 leads only at 1 stream (77.2 against 72.5) and trails from 2 (433.9 against 452.6 at 8). 3B Q8_0 flattens from 7 to 8 streams (+4.9 %). Earlier known answers reproduced at the new memory clock (3B Q8_0 at 7 streams 777 against 718) | right on the shape; crossovers at 2 to 3 and 7 to 8 streams for the 3B, at 1 to 2 for the 8B. **Round 2 (item 7a, six interleaved repetitions): holds**, including the 3B's 8-stream lead (+2.8 %, ranges do not overlap); 3B Q8_0 7 to 8 streams +3.9 % (round 1 +4.9 %); the 3B Q8_0 single stream is 142.9, not the noisy 98.5 |
| B3 | Known perplexities reproduced (the dive plan's tolerance was 0.002; the plan file is not in the repository) | 3B 9.7049 / 9.7042 / 9.7037 / 9.7041 / 9.7140 exactly as before; 8B 7.4407 / 7.4408 / 7.4397 / 7.4404 / 7.4489, no NaN | right (identical to four decimals) |
| B3 | Same run again gives exactly 0 | Mean KLD 0.000000 (six decimals) but not exactly 0 in every statistic: 99 % KLD 3.4e-5, maximum 6.3e-5 (3B) and 5.5e-5 (8B), same top token 100 and 99.992 %. The two saved logits files are byte-identical (`cmp` exit 0, sha256 equal, `det-*.compare.log`). The residue is INFERRED to come from the 16-bit compression of the saved logits (the same size of numbers appears in (b) and (c)), not from the GPU | right for the files, wrong as worded for the KL statistics; noise floor named |
| B3 | (b) and (c) tiny, mean KLD below 1e-4; (d) about 1e-3; (b) must be far below (d) | (b) 7e-6 both models, (c) 3e-6 and 5e-6, (d) 7.05e-4 and 6.14e-4: 88 to 235 times larger than (b) and (c) | right (d is 0.6 to 0.7e-3, not 1e-3) |
| B4 | Decline at depth proportional to the extra KV bytes | The F16 KV cache at 4096 adds 0.470 GB (3B) and 0.537 GB (8B) to the streamed weight bytes: predicted -12.7 % and -12.1 % (3B FP8, Q8_0), -6.6 % and -6.3 % (8B); measured -11.2 % and -9.9 % (3B patched FP8, Q8_0), -7.2 % and -6.3 % (8B) | right for the 8B (within 0.6 points); the 3B falls 1.5 and 2.2 points short of the predicted decline |
| C1 | Prompts: core 3.2 to 3.3 GHz, power-limited at 317 W | Core 2.94 to 3.08 GHz (median while awake), power at 315 to 317 W median, power-limit flag PPT0 in 89 to 91 % of samples (first run 77 to 92 %); the peak was 3.23 GHz | wrong on the clock (about 0.2 GHz lower), right on the cap |
| C1 | Generation: clock at or above 3.3 GHz, power far lower than matrix loads (3B Q8_0 about 135 W) | Clock 3.31 to 3.39 GHz, right. Power 248 to 260 W for Q8_0, 305 to 316 W for FP8 patched (the 8B at the cap): not far below the prompts | right on the clock, wrong on the power (about twice) |
| C1 | Memory clock exactly the logged uclk | 1408 MHz in every one of the 130 jobs and in every sample | right |
| C1 | Hotspot below 80 C in 20 s loads | Prompt loops 83 to 90 C, generation loops 70 to 80 C; the thermal guard (95 C) never fired | wrong for prompts, right for generation |
| C2 | Big matmul kernels at 90 % or more of the bus, small kernels far below, gaps account for the rest | Q8_0 gate+up 687 to 697 GB/s (97 to 99 %), down 668 to 686 (95 to 97 %); q, o 594 to 653; k, v 489 to 558 (69 to 79 %). Gaps 11 to 21 % of a token for the patched and Q8_0 runs (profiled) | right |
| C3 | Gaps of 1 to 4 us inside a graph; over 8 us would need an explanation; about 14 % of a 3B token | Median gap 2.9 to 3.0 us (p95 3.0 to 3.2) with the profiler; 1.8 to 1.9 us without (INFERRED); 3B patched: 425 kernels, gaps 18.7 % of the traced token, about 13 % implied without the profiler. A few gaps above 8 us exist (largest 11 to 24 us; first run up to 135 us) | right |
| C4 | No visible effect on tg128 after idle; a few percent on the first pp512 after 2 s idle | First tg128 after load 150.1 against 159.0 (-5.6 %, FP8) and 139.5 against 146.0 (-4.5 %, Q8_0); first pp512 after load 1909 against about 9100 (FP8) and 5451 against about 11300 (Q8_0). Tests 2 to 6 after 2 s idle ran 5 % (FP8) and 10 % (Q8_0) slower than a back-to-back loop (9139 against 9623; 11317 against 12587), with a lower core clock (2.79 against 3.07 to 3.11 GHz median); tg128 after 2 s idle was not slower. The first-call penalty after loading is far larger than a clock ramp (INFERRED: idle did not repeat it) | right for the size of the idle effect (5 to 10 %, a few single runs each); wrong in size for the first call after loading |
| C5 | The displays keep the memory clock at 1408 | Held throughout | right |
| D1 | Kernel trace sees kernels inside HIP graphs (leaned 55 % yes) | Yes: 9307 kernel rows from 32 `hipGraphLaunch` calls; no graphs-off build was needed | right |
| D2 | Counters (`--pmc`) work on gfx1201 in ROCm 10.1 (leaned no) | `GRBM_COUNT` returned 22 rows; the listing of available counters includes `SQ_WAVES` and `L2CacheHit`. Only this one counter was tried | wrong (round 2: true only for GRBM and a few SQ counters; see item 4 of `llm-workload-2.md`) |
| D3 | Trace kernel duration equals the probe's own timing within 5 % | read kernel 1.5263 against 1.5214 ms (+0.32 %), copy 3.5182 against 3.5092 (+0.26 %) (first run -0.64 % and -0.08 %) | right |

Plan items not turned into expectations but checked: the grouping known answer (a plain f16 MUL_MAT run should put
more than 90 % of time in the GEMM group) gave **88.4 %** (`trace-bops-f16.report.txt`, both runs): missed, the other
11.6 % is the test's own B-matrix convert to FP16, which the grouping files under "activation convert". The 90 % was the
plan's guess. After seeing this the check was changed to "GEMM plus activation convert above 95 %" (100.0 %, PASS), the
same in `profile` and `reanalyse` (in the first run `reanalyse` ended "all checks passed" on the missed 88.4 %, while `all`
reported FAILED); the change was made after the number was known, for that reason, and the 88.4 % is still the result.

## Tables

All numbers are medians; min and max are given where they matter. Every table's source file is named in its heading or
in the run file list below. Profile "Testing" (stock core, 317 W cap), memory clock from the first table.
Reading notes: the prompt sizes up to 1024 run with the default micro-batch of 512 tokens (so pp1024 is two micro-batches);
pp2048 and pp4096 use a micro-batch equal to the prompt, which is why pp2048 is faster than pp1024. "TFLOPS" in table A2 is
2 x 512 x K x N over the kernel time, for MMQ read it as int8 TOPS. In C1, "awake" leaves out the samples where the
core sat at its idle clock between tests; the power throttle flags are those of the card's metrics table (PPT0 is the
power limit, TEMP_HOTSPOT the temperature).

### Memory clock of every GPU job (current_uclk before/after; the run's first reading is the reference)

- 1408/1408 MHz: 130 jobs

### Profiler known answer (bandwidth probe, 1 GiB read and 2 GiB copy, 50 reps)

| kernel | probe's own median ms (events) | rocprofv3 median ms | difference |
|---|---|---|---|
| read | 1.5214 | 1.5263 | 0.32 % |
| copy | 3.5092 | 3.5182 | 0.26 % |

### Known answers

| Piece | Known answer | Result |
|---|---|---|
| Profiler | kernel duration within 5 % of the probe's own timing | +0.32 % (read), +0.26 % (copy); first run -0.64 % and -0.08 % |
| `llama-bench`, 3B, 1408 MHz against the first hour (1258 MHz, overclocked core) | within 8 %, differences explained | pp512 +5.0 / +6.4 / +5.8 % (FP8 unmodified / patched / Q8_0), tg128 +2.2 / +8.3 / +8.3 %. Four of six cells are within 8 %; the two generation cells are just outside. The pp cells are +5 to +6 % in spite of the lower core clock (stock, not +150 MHz); why is INFERRED and untested: prompt GEMMs are compute-bound, so the 12 % higher memory clock is a doubtful cause, and other changes since the first hour (the build is the same, the library stack and run conditions may not be) are as likely |
| `test-backend-ops` | 448/448 f8_e4m3, all Q8_0 MUL_MAT | 448/448 on both builds, 122/122 Q8_0, 383/383 f16 |
| Perplexity at 4 chunks | 9.7049 / 9.7042 / 9.7037 / 9.7041 / 9.7140 (the dive plan's tolerance, 0.002, is not in the repository) | identical to four decimals; no NaN |
| Batched bench | earlier streams table (1258 MHz, READ from the expectations file); one stream equals B1's depth-0 tg128 within 5 % | one stream 145.0 / 158.1 against tg128 147.2 / 159.5 (3B Q8_0 / patched, -1.5 % and -0.9 %). Against the earlier table: patched FP8 +8.9 % (3B), +8.4 % (8B), 8B Q8_0 +10.9 %; the earlier 3B Q8_0 row (98.5 at one stream) was noisy (it was below its own tg128) and is not a usable known answer |
| Sampler | idle clock near the lowest state (the job logs read `sclk=S: 103Mhz *` before each job), memory 1408 | sampler idle median 95 MHz over all samples (first run 98), memory 1408 in 551 of 551 samples; busy prompt loops 2.9 to 3.2 GHz |
| Kernel grouping | an f16 MUL_MAT run puts more than 90 % in the GEMM group | 88.4 %, missed; the rest (11.6 %) is the convert of the B matrix; the check is now GEMM plus that convert above 95 % (100.0 %) |
| Library GEMM at 4096 x 4096 x 512 (first hour) | 37.3 TFLOPS "hipBLAS FP16" | not reproduced as llama.cpp's route: test-backend-ops f16 at m=4096, k=14336, n=512 gives 109.0 TFLOPS. Why is INFERRED and untested: a different library path in the first-hour probe, or a different library version (the first-hour note's PyTorch used bundled ROCm 7.1 libraries, the study runs ROCm 10.1), or the probe's shape |

> **Updated in round 2 (item 2):** ~~Why is INFERRED and untested~~ settled: the probe's shape. The same call at 4096 x 512 x 4096 gives 107.8 TFLOPS and at 4096 x 512 x 14336 136.2, with the same libraries; the first-hour 37 TFLOPS reproduces (36.5) at its own shape with a 16 x 16 tile. Not a different library path or version.


## Deliberate breaks (each must FAIL), selftest and mutation checks

All run by `scripts/study-llm.sh selftest` (about 1.5 minutes, no GPU; the rerun's setup began 11:01:08Z and its first GPU
job 11:02:54Z), FAIL seen yes for every one, and each of the checks below also named in the fix round was removed or
weakened in a copy of the code and the selftest went red:
1. fake `/proc` with a card client named GameThread, and with a process holding a `steamapps/common` path: job skipped, exit 7.
2. fake meminfo at 4 GB: not run, exit 5. 3. lock held by another process for longer than the wait: exit 6.
4. LACT profile read as Default: the next job is not run, all GPU work stopped. 5a. sampler limit set low: a dummy process
   was stopped by PID within a second; 5b. hotspot read as 96 C after a job: all further work stopped; 5c. hotspot
   75 C at the start: not started, exit 4. 6. canned trace with one kernel row deleted: `total-kernel-count` FAILs;
   a kernel name the groups do not know in one pass only: `passes-identical` FAILs; the same unknown name in every pass:
   `other-below-5pct` FAILs. These and 11 run `cmd_report`, the code the pipeline runs, on a canned trace file, and each break
   must trip the check it names.
7. Q8_0 results under an FP8 label: FAIL. 8. same-run-again KL divergence: a mean of 1.2e-4, a maximum of 2e-3 or a top-token
   agreement of 99.5 % each FAIL the noise-floor check; the real-data values (mean 0, maximum 6.3e-5, top 99.992 %) pass;
   3B logits against 8B (mean KLD 5.2): not tiny FAILs.
9. memory clock 1408 then 1258 inside one job: labelled `uclk=1408/1258` (script). A table with one job at 1258 MHz among
   1408 MHz jobs, the bus share with the probe at another clock, and a job whose clock changed inside it each raise
   (the real builders `sec_b4` and `sec_bw` on a canned run folder).
10. a profiled command that launches no kernel produces no trace file: `kernel-rows-found` FAILs (seen in `trace-nomatch`).
11. profiler overhead above 15 %: the 50 % case FAILs `profiler-overhead-below-15pct` and the 2 % case passes.
12. (fix round) the bus share uses streamed bytes: a canned GGUF with known tensor sizes gives 2240 - 640 = 1600 bytes, not
    the file size; a GGUF without `token_embd.weight` or without `output.weight` (tied embeddings) is refused; a models file
    without `streamed` lines is refused instead of falling back to the file size.
13. (fix round) gates after the lock wait: while another job holds the lock, a hotspot of 80 C, a game process, a LACT profile
    change or 4 GB of memory appears; each job is then not run after the wait (exits 4, 7, 3, 5) and the summary says it was seen
    after the wait. A control with nothing changed runs. Removing the post-lock checks makes all four go red.
14. (fix round) determinism evidence: identical canned files give `cmp` exit 0, equal sha256 and sizes in the evidence file and
    are then deleted; one changed byte gives exit 1 and different sha256; a missing file fails; an evidence file that cannot
    be written fails and the logits files are kept. Forcing the `cmp` status to 0, or deleting before the evidence is written,
    makes the selftest red.
   Each deliberate-break case has a control without the fault that must pass; all controls pass.
- **Mutation check:** each gate check was removed from a copy of the script in turn (game scan, memory floor, lock,
  profile and configuration check, poll limit, post-job limit, start limit, memory-clock label) and the selftest went red
  every time in the first run (the profile check and the configuration check cover each other: both had to be removed for
  the break to show). A first mutation round tested nothing because the copies called the original script; found, fixed
  (the copy now calls itself), repeated; the same trap was avoided in the fix round. In the fix round the following were
  weakened and went red: the overhead limit (1.15 to 9.99), the `other` limit (5 % to 99 %), the group-share sum, the
  one-clock check, the streamed-bytes subtraction, the mean and maximum limits of the same-run check.
- A first version of the profile step used the profiler's `--kernel-include-regex` for break 10; in the step-2 trace
  that option did not filter (the trace still held kernel rows; the count was not kept), so the break was replaced by a command
  with no kernels. The first analysis of the generation traces also failed on "128 passes kept" (126 to 127 kept: a few
  tokens split at host pauses); the minimum was lowered to 115 and the traces were analysed again offline (`reanalyse`),
  so the `profile` section of the first run's `summary.md` shows those FAILs and the later `reanalyse` section the passes.
- Not an independent check: `kernels-plus-gaps-equal-wall` is an identity for sorted, non-overlapping kernel rows (kernel
  sum plus gaps always equals last end minus first start), so it can only fail through overlap. The check on the wall time
  is the profiler-overhead ratio against an unprofiled run.

## Rerun

Two runs, both on the "Testing" profile at memory clock 1408 MHz in every job:
- first run `20261118T094213Z-llm` (09:43 to 10:17Z), from the first scripts (1fbba0e); `reanalyse`, `determinism`, `a5` and the
  table builders were added after it;
- rerun `20261118T110108Z-llm` (11:01 to 11:37Z), from the committed fix-round head (b497f7d; later commits changed only the
  output formatting of one summary line and the A5 table block): `all`, then `a5`, `determinism` and `reanalyse` in the
  same folder. 130 GPU jobs (120 from `all`, 6 `a5`, 4 `determinism`), every one with exit 0 and the memory clock 1408 before and
  after, highest hotspot 89 C (sampler 90 C), no thermal stop, the card configuration unchanged, the LACT profile "Testing"
  throughout, no game, hotspot before each job 45 to 56 C.

Checks. Every analysis check has the same status in both runs except one: the rerun's `no-overlap` check FAILED on
`trace-tg128-8b-fp8-patched` (5 kernel pairs of 60,984 overlap by 29 to 815 ns; the first run had 0). The check has no tolerance
and was left as it is; the traced numbers of that trace did not move (wall 13.198 against 13.165 ms, median gap 2.92 against
2.96 us). So the rerun's `all` and `reanalyse` end "something FAILED" on that one check, and the first run's ended "all checks
passed" on `reanalyse` (while the 88.4 % grouping answer, missed, was not counted; see above). The first run's profile section
also holds four `passes-count` FAILs that the later `reanalyse` replaced; the rerun, with the lowered minimum, has none.
The grouping known answer, the 448/448, 122/122 and 383/383 test-backend-ops results, all perplexities (identical to four
decimals), all KL divergence tables (identical to the digits shown) and both byte-identical determinism comparisons agree.
> **Updated in round 2 (item 3):** the five overlaps were most likely a timestamp artifact (INFERRED; 0 overlaps in 706,017 kernel rows of 11 further tg128 traces; one queue, ns-scale, neighbouring dispatches; three are the KV-cache write followed by the attention kernel that reads it, two are matvec pairs; concurrency only exists with `GGML_CUDA_GRAPH_OPT=1`, never set). The check was left unchanged. See `llm-workload-2.md`.

Headline numbers, first run against rerun (medians; "spread" = the first run's min to max of repetitions):

| quantity | first run | rerun | |
|---|---|---|---|
| A split, 3B FP8 unmodified pass: wall / GEMM / widen / act convert / scale mul / gaps (ms) | 57.61 / 26.94 / 10.54 / 2.77 / 3.40 / 3.19 | 57.54 / 26.91 / 10.55 / 2.75 / 3.40 / 3.16 | within 1 % |
| 3B Q8_0 pass wall / MMQ (ms) | 39.74 / 25.65 | 39.74 / 25.64 | same |
| A5, 3B: traced wall (ms) / pp512 (tokens/s); FP8 patched wall | 53.42 / 9611; 53.43 | 53.54 / 9573; 53.60 | within 0.5 %, pp512 inside the first run's spread |
| A5, 8B: pp512 (tokens/s) | 4768 | 4793 | +0.5 % |
| B1 pp512, 3B: unmodified / patched / Q8_0 | 8957 / 9529 / 12483 | 8975 / 9521 / 12438 | within 0.4 % |
| B1 tg128, 3B: unmodified / patched / Q8_0 | 78.5 / 157.9 / 146.1 | 78.7 / 158.7 / 146.9 | within 0.6 % |
| B1 pp512, 8B: unmodified / patched / Q8_0 | 4471 / 4689 / 5917 | 4578 / 4676 / 5947 | unmodified +2.4 % (first run had a cold repetition), others within 0.5 % |
| B1 tg128, 8B: unmodified / patched / Q8_0 | 37.6 / 77.2 / 72.6 | 37.6 / 77.2 / 72.6 | same |
| prompt sizes, 3B patched / Q8_0 at pp16 / pp512 / pp2048 | 53 / 76 / 97 % | 53 / 77 / 95 % | pp2048 is 1.4 points lower |
| generation share of the bus (streamed bytes, depth 0): 3B patched / 3B Q8_0 / 8B patched / 8B Q8_0 | 73 / 71 / 83 / 82 % (first run's data, recomputed) | 73 / 71 / 83 / 82 % | same |
| C2 gaps: median gap (us), share of a token (3B patched / 8B Q8_0) | 2.92, 18.7 % / 2.96, 12.7 % | 2.92, 18.7 % / 2.92, 12.7 % | same; largest gap fell from up to 135 us to 11 to 24 us |
| C1 prompts: core MHz median awake, power W median (3B patched / 8B Q8_0) | 3081, 317 / 2942, 316 | 3078, 316 / 2941, 316 | same |
| C1 generation: core MHz, power W (8B patched / 3B Q8_0) | 3311, 316 / 3388, 248 | 3311, 316 / 3388, 248 | same |
| PPT0 flag share of samples, prompt loops | 77 to 92 % | 89 to 91 % | the first run's 3B patched recording had 1102 samples against 1032 now; the extra were not examined |

What moved: (1) the 8B prompt rows pp16 to pp64 (up to twice; see the notes under the tables), caused by where the core clock
sat when a short job started (INFERRED); (2) the largest gap per token (a few outliers in the first run); (3) the first-call
set-up cost after loading (first prompt 1736 against 1909 tokens/s, 4945 against 5451), which varies by 10 % between runs; (4) the
pp2048 ratio, 97 against 95 %; (5) the profiler read/copy agreement changed sign (-0.64 against +0.32 %), both inside 1 %.
Everything the summary rests on agreed. Where the two differ this file quotes the rerun, which ran the committed code.
> **Updated in round 2 (item 6), on (1):** ~~caused by where the core clock sat when a short job started (INFERRED)~~ not supported: in 12 repetitions 11 reproduced every row within 3 % by the median of the five samples; the one slow run (patched run 5) was 8 to 25 % slow in five of seven rows (median) with the card 68 to 96 % busy in its windows (host stalls, INFERRED); the clock at the start of a test does not predict speed. Single-sample dips also hit Q8_0 run 6 and patched run 6, which a median hides; by the mean the 2x may have been one dip in a mean row (INFERRED). See `llm-workload-2.md`.

## Run folders

All in `runs/` (git-ignored, so not in the repository):
- `20261118T094213Z-llm/`: the first run of the dive (`all`), then `trace-nomatch`, `determinism`, `builds`, `a5` and `reanalyse`
  added to the same folder. 130 GPU jobs, memory clock 1408 MHz before and after every one. The first run's `models.txt` has no
  `streamed` lines and its determinism step kept no `cmp` evidence (both fixed in the rerun).
- `20261118T110108Z-llm/`: the rerun (see Rerun): `all`, `a5`, `determinism`, `reanalyse`. Logs are `<job>.log`
  with `<job>.meta`, `summary.md`, `config.txt`, `builds.txt`, `models.txt` (sizes and sha256 of the six GGUF files and the
  streamed bytes of the four decode files), `det-<model>.compare.log` (the determinism evidence), traces in `trace-*.d/`,
  sampler recordings in `beh/`, analyses in `*.analysis.json` and `*.report.txt`.
- `dev-llm-*`: development runs of the script (probe and first profile attempts, selftest and mutation runs); no number
  in this file comes from them, except that the step-2 look at the traces decided the kernel groups before the first run.
- Builds: the two existing build folders (unmodified, patched; ROCm 10.1 clang 24, Release, `GGML_HIP_GRAPHS=ON`,
  source HEAD f477c1b90, binaries of 2026-11-17) were used as they are. The A5 build is a new folder
  `llm-dive-forcecublas` next to them (`-DGGML_CUDA_FORCE_CUBLAS=ON`, same source tree, built with `nice`, -j8, 10 GB cap).

## How to rerun

```
scripts/study-llm.sh all            # 34 min wall in the rerun (11:01 to 11:35Z; the 130 jobs' own time was 23 min, the rest is cool-downs and the selftest): setup, selftest, probe, known, shapes, profile, grid, quality, behaviour
scripts/study-llm.sh determinism    # two 50-chunk runs per model, logits files compared with cmp, evidence in det-<model>.compare.log
scripts/study-llm.sh builds && scripts/study-llm.sh a5    # the A5 build (once), then its jobs
LLM_RUN=runs/<folder> scripts/study-llm.sh reanalyse      # also writes the A5 trace analyses
python3 -I scripts/llm_tables.py tables runs/<folder>
```
`determinism`, `a5` and `reanalyse` are not part of `all`; run them with `LLM_RUN` set to the same folder.
It needs the card record (`scripts/first-hour.sh info`), the sampler (`build/study/bin/watch_card`), the bandwidth probe,
the shared GPU lock path in the build folder's `local.env`, and the builds and models described above (`LLM_BUILD_ROOT`,
`LLM_MODELS`). `LLM_BUILD_ROOT` defaults to a folder under the user's home next to the study's older scripts' default;
it is the one script default that names a sibling project, kept for consistency with the older scripts and listed for the
scrub before any publishing.

## Parked, skipped and open

- **3B F16 reference (parked).** The plan's route (`llama-quantize` from the existing tree) cannot make an exact-weights
  file: the FP8 files keep their scales as separate tensors, which re-quantizing would not apply, and the converter
  from the original checkpoint needs Python packages that are not installed (no installs allowed). The exact-weights
  KLD reference is therefore not measured; the KLD numbers are route against route (hipBLAS FP16 on exactly widened
  weights is the reference), not distance to the true model. A stdlib-only writer (FP8 times scale to F16 per row) is
  possible; not done. Owner decision: accept route against route, or allow that writer.
  > **Updated in round 2 (item 1):** done for the 3B with an exact F32 reference (every FP8 weight times its scale, bit-checked): FP8 routes 3e-6 mean KL divergence, Q8_0 7e-4 (`llm-workload-2.md`). For the 8B the F32 reference (built and verified) could not be run on this machine; an F16 reference was used instead.
- **Only one shape in `test-backend-ops perf`** (m=4096, k=14336): arbitrary shapes cannot be given to the tool without
  changing it; A3 rests on that one shape, A1 and A2 cover the others.
  > **Updated in round 2 (item 7b):** done in a separate copy of the unmodified tree with only test code added (`scripts/llm2-test-shapes.patch`): every 3B and 8B weight shape at n = 1 to 2048. A3 holds at n = 512 for the biggest matrices (718 to 842 GB/s); at larger n the GEMM after a widen is 6 % faster than on a stored F16 matrix.
- **Counters:** only `GRBM_COUNT` was run; none of the counters was used in the analysis.
  > **Updated in round 2 (item 4):** 13 counter sets run against a kernel that reads 256 MiB: only GRBM, `SQ_WAVES`, `SQ_BUSY_CYCLES` and `SIMD_UTILIZATION` count; every memory-side and instruction counter reads 0 (no error). ~~D2 "counters work"~~ is only true for those few.
- **The rerun's `no-overlap` FAIL** on `trace-tg128-8b-fp8-patched` (see Rerun) is reported, not tuned away.
- **Stale line in the first run's `summary.md`:** its first `profile` section holds "break10 ... did NOT fail" from the abandoned
  regex version of the break; its `reanalyse` section has the replacement (FAIL seen). The rerun has no such line.
- **`first-hour.sh` line 479** (not 478) links against `$LLAMA_UNMOD/build-gfx1201/bin`, a build folder that no longer exists
  (the current one is `build-gfx1201-rocm10`; READ with `ls`): out of scope, not changed. It does not name a ROCm 7 path.
- No game ran during either run; the owner's desktop was in use (hotspot before each job 45 to 56 C).

## Not measured

- FP8 matrix instruction use by an LLM kernel (no kernel changes were allowed); a hipBLASLt-FP8 prompt path inside
  llama.cpp (arithmetic only, A4).
- True F16 reference perplexity and KL divergence (see parked).
- Llama 70B or a mixture-of-experts model; contexts beyond 4096; the Vulkan backend of llama.cpp (not built); another
  ROCm version; a newer llama.cpp than f477c1b90.
- Power at the wall (reported watts are scaled down by the 1.2x mod); the memory clock at 1258 MHz (the displays never
  blanked, so no job ran at it; the comparison with 1258 MHz numbers is against earlier runs only); thermals beyond
  20 s loads.
- Hardware counters in the analysis; per-wave occupancy inside the llama.cpp kernels; why the FP8 decode kernel draws
  56 to 57 W more than the Q8_0 kernel at nearly the same bandwidth (INFERRED: FP8 widening and floating-point math against
  integer dot products; not tested); whether the part of the widen traffic above the read rate (about 770 to 800 GB/s: 802 over all
  matrices of the 3B, 772 to 796 depending on which matrices count; the last layer's single-row matrices are not widened) stays in the 64 MB memory-side cache (INFERRED, not tested); multi-GPU; real chat workloads.
  > **Updated in round 2 (item 4):** the cache question was tested with a probe (counters being dead): writes to a buffer reused every call stay cached up to about 50 MiB (write rate 1524 against 611 GB/s); a widen-shaped kernel gains 4 to 12 % at 25 MiB (three runs of two binaries) and about 30 % at 8 MiB, where launch overhead is 14 to 18 % of the widen rows' time. The kernels' 770 to 800 GB/s is not reproduced (probe 637 to 681). The 57 W question stays INFERRED.

## Proposed lines for `notes/STUDY-PLAN.md` and `README.md` (not applied; those are the lead's files)

**STUDY-PLAN.md, status paragraph:**

> LLM data dive done (branch `study/llm-dive`, `scripts/study-llm.sh`, run `runs/20261118T110108Z-llm` (rerun of the committed fix-round code; first run `runs/20261118T094213Z-llm`),
> `measurements/llm-workload.md`, expectations in `measurements/llm-workload-expectations.md`, written before the
> first GPU job): llama.cpp (unmodified and with the FP8 decode kernel) on Llama 3.2 3B and 3.1 8B as FP8 and Q8_0,
> kernel traces, prompt-size, depth, stream and quality grids, and 20 s clock and power recordings, at memory clock
> 1408 MHz on the "Testing" profile. 130 GPU jobs, no thermal stop, the card configuration unchanged.

**README.md, key-finding bullet:**

> - **llama.cpp's FP8 prompt loss is the route, not the format.** FP8 weights cannot use the int8 matrix kernel Q8_0
>   gets, so each matrix is widened to FP16 in memory, then one FP16 GEMM runs. In a 3B pass of 512 tokens the GEMMs cost
>   the same (26.9 against 25.6 ms); the 17.8 ms difference is the widen (10.5 ms), an activation convert and a scale
>   multiply. Q8_0 forced through the same route is as slow as FP8, and a 2048-token micro-batch brings FP8 to 95 % of
>   Q8_0. Generation reaches 73 to 83 % of the read bandwidth with the FP8 decode kernel (bytes decode streams); the card sits at its 317 W
>   limit in prompts and in FP8 generation of the 8B, and two identical runs give byte-identical saved logits.
>   ([llm-workload.md](measurements/llm-workload.md))

**README.md, measurement-file table rows:**

> | `llm-workload.md` | LLM data dive: where llama.cpp's FP8 and Q8_0 prompt and generation time goes (kernel traces), speed and quality grids, clocks, power and bandwidth under the load |
> | `llm-workload-expectations.md` | The expectations for the LLM dive, written before any of its GPU runs |
