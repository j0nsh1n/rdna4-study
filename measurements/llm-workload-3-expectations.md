# LLM data dive, round 3: expectations, written before the GPU jobs of each question

Written 2026-11-18 (file time of the commit), before any GPU job of round 3. Not edited afterwards; what the runs showed
goes into `llm-workload-3.md`, each line marked right or wrong. Labels: INFERRED (reasoning from earlier results), READ
(source, listing, file). Setup as round 2: llama.cpp master f477c1b90, ROCm 10.1, LACT profile "Testing", memory clock
expected 1408 MHz. Q5 was added by the lead after Q1 to Q4 were assigned; its expectations are in the last section and
were committed before any Q5 GPU job (the Q1 to Q4 expectations were committed before their own).

## Q1. Why FP8 patched generation draws about 57 W more than Q8_0 at the same bandwidth

Known: 8B patched FP8 tg128 sat at the 316 W cap at 3.31 GHz, Q8_0 at 248 to 260 W at 3.39 GHz, 584 against 581 GB/s
(round 1). Reported watts are scaled by the regulator mod, so only differences are quoted and every one is "reported W".
The source of the patched kernel (READ, `mmvf.cu` diff): per 16 weights, per 32-bit word, about 5 integer bit operations
(shifts, ands, or) to put FP8 bits into F16 position, four F16-to-F32 conversions, four FMAs, and a NaN check. Q8_0
decode (llama.cpp's integer matvec) does one 4-way int8 dot per 4 weights and one float multiply per 32-weight block.
E1a (ISA count). Per weight the patched FP8 kernel's loop has 4 to 7 vector ALU instructions (about 1 conversion, 1 FMA,
2 to 4 bit operations), the Q8_0 loop about 0.3 to 0.6 (the dot, plus scale work and the activation loads). Ratio at least
8 to 1. Could fail: the compiler packs two F16 conversions (packed conversion instructions) or uses a dot-product form, giving a ratio under 4.
E1b (counters that work). SIMD_UTILIZATION and SQ_BUSY_CYCLES are about 1.0 for both and cannot tell them apart (both
are busy waiting on memory); I expect no useful difference (70 %). Could fail if SQ_BUSY_CYCLES differs by more than 5 %.
E1c (synthetic probe pair at matched bandwidth). At the same streamed bytes per second, the FP8 instruction mix draws
30 to 60 reported W more than the dot4 mix (the whole 57 W would be explained, within 25 %: 55 %; half or more: 75 %).
The remainder, if any, is the lower core clock under the cap and the different access pattern (INFERRED). Could fail:
the probe does not reach the same bandwidth (the mixes are then not matched; the run is rejected, not reported), the
compiler removes the work (ISA loop check catches it), the sampler's power reading is too coarse (20 ms, averaged by the card)
so differences under 5 W are not trusted, or the card is power-capped in the FP8 probe (then the clock falls and bandwidth may not match).

## Q2. The exact 8B comparison (mixed F16 and F32 reference)

READ: the 8B scales have at most 8 significant bits, E4M3 has 4, so a product has up to 12; F16 holds 11. A product
of an FP8 value and a scale is exact in F16 only if its significand needs at most 11 bits AND its exponent lies in the F16 range
(normal down to 2^-14, subnormals down to 2^-24 with fewer bits; maximum 65504). Round 2 found on the 3B that F16 is
exact for 52.5 to 100 % of products per tensor, 100 % in at least one tensor.
E2a. Per tensor, by computation over all 256 FP8 codes times that tensor's scale: a minority of the 8B's FP8 tensors
(10 to 40 %) are exact in F16 on both tests; most (60 to 90 %) need F32. Tensors with a scale of at most 7 bits pass the bit test
(4 + 7 = 11); the exponent test fails for tensors with a very small scale (products below 2^-24). Could fail: if most scales
have 7 or fewer bits nearly every tensor passes (then the file is near 16 GB and fits).
E2b. The mixed file is 20 to 28 GB (F32 tensors 4 bytes per weight, F16 2, the embedding BF16). It does NOT fit as 16 GB
card plus the 10 GB job cap: the card needs room for the KV cache, compute buffers and the desktop (about 1.5 to 2.5 GB), and
the process needs host memory beyond the weights, so about 21 GB of weights is the limit and the file is above it (75 %).
Then the item stops with the numbers (file size, VRAM free, cap). If it does fit (25 %), the run uses `-ngl` partial
offload, 50 chunks of 512 as round 2 (chunk count stated), sample rows bit-checked.
Could fail: the F16 share is above 60 % by size (file about 18 GB).

## Q3. Why the rebuilt widen_cache reads about 4 % lower than the round-2 binary

READ: both binaries come from the same source except the round-2 binary lacks the 1 GiB known-answer read and the empty
launches; the fix round added them (`widen_cache.hip` header). The widen row at S = 25 differed 663.5 and 660.0 (old) against
637.7 and 635.6 (new) in the same session.
E3. The timed loops' ISA is identical between the two binaries (the loop check passes for the new one; the old one has the
same source in the timed functions) (80 %); the compile flags are the same except `-DHARNESS_ARCH` (maybe) (60 %). The difference
is then the code around (the new binary's extra work before the rows changes the state of the cache or the clocks) or the
build's code alignment, not the loops. In an interleaved A B A B run of at least 6 each, the 4 % persists with non-overlapping ranges
(65 %); the other 35 %: it vanishes (run-to-run spread, seen once in the same session, 6 to 8 % of spread in the S = 25 widen row of round 2).
Could fail: the old binary no longer exists as built (then it is rebuilt from the round-2 commit and that is stated).

## Q4. Is the dead-counter result specific to ROCm 10.1

READ: ROCm 10.0 rocprofv3 ships in `toolchains/torch-rocm10-venv/bin` (pip packages). The round-2 result: of 13 sets only GRBM,
SQ_WAVES, SQ_BUSY_CYCLES and SIMD_UTILIZATION return numbers.
E4. The same 13 sets give the same pattern with 10.0 (70 %): the same four work, the rest read 0 or error, because the
cause is the kernel driver or firmware, not the profiler. The counter lists differ little (both show the same gfx1201 counters, 10.1 maybe more).
Other outcomes: 10.0 reads some memory-side counters (20 %); 10.0 does not work with the system's runtime libraries at all (10 %, a
mismatch between its pip libraries and the card). Kernel settings readable without root (perf_event_paranoid, the
amdgpu module parameters, power_dpm_force_performance_level) are not expected to differ from what round 2 assumed; I expect
no setting readable here to explain the zeros (80 %).

## Q5. The sub-microsecond overlaps: real or timestamp artifact

READ: round 2 item 3: 5 overlaps of 29 to 815 ns between neighbouring dispatches on one queue in one trace, 0 in 706,017
rows of 11 repeats; called "most likely a timestamp artifact" (INFERRED). The in-kernel REALTIME counter (`s_memrealtime`,
100 MHz, so 10 ns per tick) is independent of the profiler's start and end stamps.
E5a. A plain (non-graph) A to B pair with B reading A's output never overlaps on the in-kernel clock: the queue's barrier
bit between dependent dispatches (the runtime sets it) holds B until A has finished, so last(A) is before first(B) in
every pair (85 %). In a HIP graph the same (80 %).
E5b. Under the profiler, the profiler's own stamps show end(A) later than start(B) in a small fraction of pairs
(rate 0 to 0.1 %, sizes tens to hundreds of ns, as in round 2) in some runs and none in others; when it does, the in-kernel clock
shows no overlap, i.e. an artifact (65 %). If an overlap shows on BOTH clocks the answer is real overlap (15 %), and B's data check
then fails or passes depending on whether B's early waves read stale data. Neither: the profiler never shows an overlap in my probe's
thousands of pairs (20 %); then the probe could not reproduce the effect, the answer stays "no evidence for real, not shown as artifact either".
E5c. Positive control: two independent kernels on two streams, each long enough (hundreds of us) overlap on both clocks (90 %).
Negative control: with a host sync between A and B neither clock shows overlap (95 %).
E5d. Aligning the clocks: the profiler's stamps are in the system's ns timebase (kernel-trace); the in-kernel counter is a GPU
REALTIME counter. They can be put on one axis by an offset fitted from the first and last pairs (host reads `clock_gettime` and the device
reads `wall_clock64`/`s_memrealtime` in one probe launch); the alignment error will be around 1 us, larger than the overlaps in question,
so the comparison that counts is the ORDER within the GPU clock (last(A) against first(B)), not the absolute alignment against the profiler (80 %).
Could fail: kernels that finish within one REALTIME tick of each other (10 ns) leave order ambiguous: such pairs are counted as ambiguous, not as overlaps.
