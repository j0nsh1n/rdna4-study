# Round 6: ROCm 10.2 against 10.1, and profiler counters on the real decode kernels

Two items, each in its own run folder (git-ignored), both on 2026-11-20:

- **Item R, ROCm 10.2 against 10.1:** `runs/20261120T161509Z-r6cmp/` (vLLM speed, FP8 perplexity, hipBLASLt GEMM speed).
- **Item C, counters on the decode kernels:** `runs/20261120T161536Z-r6ctr/` (llama.cpp and vLLM, one-token decode).

Every job of item R ran on the LACT profile "Default" (overclocked core, raised power limit), with the memory clock at 1408 MHz
in every vLLM job. Item C ran with the performance level set to `profile_standard` (set before the run, set back to `auto`
afterwards, read back each time), LACT profile "Default" throughout; the jobs' own logs show the level `profile_standard` before and
after every one of the 10 jobs (`out/jobs-profile_standard.tsv`). No GPU reset in either run. Labels as in `round5.md`: MEASURED
(read from a log or counter file), INFERRED (worked out from measured numbers), READ (a source file). Each run's own
`RESULTS.md` was checked against the logs and CSVs it names; where they disagree the log wins and the difference is listed at the end.

All wall-clock numbers in item R come from an overclocked card with a raised power limit. Item C's cycle counts come from a
core clock held near 1.65 GHz and a memory clock of 1124 MHz, so they rank kernels and are not real speed.

## Plain summary

- **ROCm 10.2 gives the same results as 10.1.** FP8 perplexity is 9.725647880183923 on both, to every printed digit (10.1 twice
  in this round, 10.2 twice). vLLM bf16 speed and all 16 hipBLASLt GEMM speeds are within about 2 % of 10.1.
- **vLLM FP8 decode speed has two levels, and it is not the ROCm version.** One engine start gives about 159 tokens/s at one
  stream and another gives about 120, on the same day, with the same clocks, in both versions (10.1: 159, 120, 120; 10.2: 118, 159,
  159, over three starts each). At the same level the two versions agree (one stream within 1.5 %). The cause is unknown. Because
  of this, a plain "median of 3 starts" per version would be wrong and is not used. The vLLM version changed with ROCm (0.29.1 on
  10.1, 0.32.1.dev0 on 10.2), so ROCm alone is not isolated.
- **hipBLASLt picks the same macro tile for all 16 shapes on 10.2**, under renamed kernels (new tuning fields in the names).
- **No `hipblaslt-bench` ships in either install**, so a small own program calls hipBLASLt directly (`src/round6/lt_bench.cpp`).
- **Decode reads the model's weights exactly once per token, and no more.** In the fixed power state the memory counters count.
  For all five decode workloads (four formats: Q8_0, FP8, FP8 with a bf16 head, bf16) the matrix kernels fetch 1.000 to 1.001 times the weight bytes that the model file's header
  gives. So these kernels do not waste memory traffic; the formats differ in how many bytes the weights take and in how many
  instructions it takes to turn the stored numbers into something the card can multiply.
- **FP8 reads about 5 % fewer bytes than Q8_0 in llama.cpp but needs about twice the vector instructions** (2.10 times, same
  number of dispatches). vLLM's FP8 decode kernel uses about a sixth of llama.cpp's patched FP8 instructions.
- **llama.cpp's attention fetches the key/value cache for 256 positions, not the ~152 in use** (INFERRED, two numbers support it);
  at this short context that is 0.9 % of the decode bytes.

## Caveats that apply throughout

- The vLLM version changed with ROCm (0.29.1 on 10.1, 0.32.1.dev0 on 10.2), so a vLLM-level difference can come from either.
- FP8 decode speed in vLLM came out at two levels in both versions (see item R); no cause found.
- Item C's cycle counts and "bytes per cycle" are at about 1.65 GHz core and a lower memory clock (1124 MHz against 1408 in
  `auto`). Byte and instruction counts do not depend on the clock.
- `llama32-3b-auto.gguf`, the third llama.cpp workload of item C, is **FP8 matrices with a bf16 head, not 16-bit, and it ran on the
  unmodified llama.cpp build**, while the "FP8" workload ran on the patched build (`ctr.sh`: `f16|$UNMOD`). There is no true
  16-bit llama.cpp workload in this round; the 16-bit comparison is vLLM bf16.
- vLLM in item C ran `--enforce-eager` at one stream, so its graph mode and batches above 1 are not covered.

## Item R. ROCm 10.2 against 10.1

Stacks: 10.1 = vLLM 0.29.1 (torch 2.13); 10.2 = vLLM 0.32.1.dev0 (torch 2.14). 3B model, round 4's settings (max length 4096,
16 sequences, 3 repetitions inside one warmed engine, 128 steps, prefix caching off); the numbers are medians of the 3 repetitions.

### vLLM bf16 (one engine per version, 10.1 first) - MEASURED

| metric | 10.1 | 10.2 | change % |
|---|---|---|---|
| decode 1 stream (tokens/s) | 97.6 | 98.4 | +0.8 |
| decode 4 streams | 372.9 | 375.3 | +0.6 |
| decode 8 streams | 702.8 | 704.2 | +0.2 |
| prompt, 512 tokens | 15,252 | 15,746 | +3.2 |

One run per version, so the +3 % on the prompt number is inside the spread I would expect between runs (the three repetitions of
one 10.1 engine ran 13,832 to 15,600). Whether bf16 also has two levels is not excluded: one start each.

### vLLM FP8 (three engine starts per version, order 10.1/10.2, 10.2/10.1, 10.1/10.2) - MEASURED

| job (run order) | version | decode 1 | decode 4 | decode 8 | prompt 512 | level |
|---|---|---|---|---|---|---|
| r6b | 10.1 | 158.8 | 613.9 | 998.9 | 22,971 | fast |
| r6b | 10.2 | 118.2 | 449.2 | 745.5 | 19,524 | slow |
| r6b2 | 10.2 | 158.6 | 615.0 | 986.6 | 23,017 | fast |
| r6b2 | 10.1 | 120.0 | 471.0 | 806.2 | 20,538 | slow |
| r6b3 | 10.1 | 119.8 | 470.1 | 807.7 | 20,491 | slow |
| r6b3 | 10.2 | 158.5 | 613.3 | 983.9 | 23,048 | fast |

At the fast level 10.2 against 10.1 (decode 1 / 4 / 8, prompt): -0.1 / +0.2 / -1.2 / +0.2 % (r6b2) and -0.2 / -0.1 / -1.5 / +0.3 %
(r6b3). At the slow level the single 10.2 run is -1.3 to -1.5 % at one stream and 4.4 to 7.7 % below the two 10.1 runs at 4 and 8
streams. One run is not enough to call that a real difference (OPEN, below).

- Round 4's 10.1 numbers (111.7 / 436.3 / 749.1 decode, prompt 18,956, memory 1258 MHz) sit at or below the slow level; the same-day
  10.1 runs here are 7 to 42 % higher at a different memory clock, so they are not comparable (INFERRED; different day and clock).
  Round 4's 111.7 (128 steps) against 145.7 (32 steps) at the same 1258 MHz, and round 5 item H's 146.5 tokens/s in one warmed engine
  (`round5.md`), fit the same picture: an engine that lands on a slow or a fast level at start, with a context-length effect that did
  not reproduce (INFERRED; not tested, and the memory clock there was 1258 MHz against 1408 here, so the absolute levels differ).
- Clocks do not separate the levels (MEASURED, 200 ms sampler, `clocks.csv`): the core clock median while active was 3455 to 3482 MHz in
  every job that has samples, power while active 175 to 209 W, memory clock 1408 MHz. The first job `r6b` (10.1) has no samples (the
  sampler started right after it).

### vLLM FP8 perplexity, 4 chunks - MEASURED

All four runs (`r6p-101-fp8-{1,2}.log`, `r6p-102-fp8-{1,2}.log`) print `ppl 9.725647880183923` (1020 scored tokens): identical to the last
digit, 10.2 repeatable. Round 4 gave 9.725648 on 10.1.

### hipBLASLt GEMM speed, 3B projection shapes - MEASURED

`src/round6/lt_bench.cpp` calls `hipblasLtMatmul` directly: FP8 is e4m3 in and bf16 out with f32 compute and no scale pointers; bf16 is
bf16 in and out; layout `y = x W^T`. One heuristic pick per shape (top-1 `hipblasLtMatmulAlgoGetHeuristic`), 3 bursts of about 0.2 s 2 s
apart, median of the 3 (the bursts in brackets). Shapes (N x K): qkv 5120 x 3072, o 3072 x 3072, gate+up 16384 x 3072, down 3072 x 8192.
It is not `hipblaslt-bench`'s own search.

| dtype | M | layer | 10.1 TFLOPS | 10.2 TFLOPS | change % |
|---|---|---|---|---|---|
| fp8 | 512 | qkv | 178.8 (183/179/179) | 179.6 (183/179/180) | +0.4 |
| fp8 | 512 | o | 177.8 (181/178/178) | 180.3 (184/180/180) | +1.4 |
| fp8 | 512 | gateup | 195.0 (201/195/195) | 197.8 (201/198/197) | +1.4 |
| fp8 | 512 | down | 130.5 (133/131/130) | 132.2 (134/132/132) | +1.3 |
| fp8 | 16 | qkv | 11.8 | 11.8 | -0.6 |
| fp8 | 16 | o | 8.8 | 9.0 | +1.8 |
| fp8 | 16 | gateup | 16.8 | 16.8 | -0.4 |
| fp8 | 16 | down | 13.7 | 13.8 | +1.0 |
| bf16 | 512 | qkv | 93.4 | 93.4 | +0.0 |
| bf16 | 512 | o | 123.1 | 122.5 | -0.5 |
| bf16 | 512 | gateup | 146.9 | 146.9 | -0.0 |
| bf16 | 512 | down | 108.2 | 108.1 | -0.1 |
| bf16 | 16 | qkv | 23.9 | 23.9 | +0.0 |
| bf16 | 16 | o | 19.9 | 19.9 | +0.1 |
| bf16 | 16 | gateup | 10.4 | 10.4 | -0.1 |
| bf16 | 16 | down | 28.3 | 28.3 | -0.0 |

(Change % is as printed in `gemm_table.md`, which I checked row by row against these values.) For all 16 shapes the macro tile and MFMA size are the same on both
versions (for example FP8 M=512: `MT128x128x32_MI16x16x1` for every layer; FP8 M=16: `MT16x128x32` for qkv and gate+up,
`MT128x128x32` for o and down; bf16 M=512: `MT256x256x32` for qkv and gate+up, `MT128x128x32` for o, `MT64x128x128` for down), but the
full names differ by new tuning fields (`IA0`, `G0_NT`, `G0_NV`, `RAP0`, ...). So the library's kernel set was rebuilt and chose the
same tiles. Build line (reconstructed; the run folder does not record the flags): `hipcc -O2 --offload-arch=gfx1201
-I$ROCM/include src/round6/lt_bench.cpp -L$ROCM/lib -lhipblaslt -o lt_bench_<version>`, once per ROCm tree with that tree's own
`hipcc` and `libhipblaslt`; run through `scripts/round6/lt_shapes.sh`.

Memory-clock caveat (INFERRED, not MEASURED, for FP8): these jobs are short and the memory clock moved with the load (96 to 1408 MHz). For
FP8 M=512 the 10.1 job saw 772 to 1124 MHz while active and the 10.2 job 1124 and 1408 MHz, so those pairs are not strictly
comparable. M=512 is compute-bound and the versions agree within 1.4 %. The bf16 pairs had the same clock mix and agree within 0.5 %.

## Item C. Counters on the real decode kernels (`profile_standard`)

Ten jobs: llama.cpp Q8_0 (unmodified build), llama.cpp FP8 (`llama32-3b-fp8-outq8`, patched build), llama.cpp "auto" (FP8 matrices and a
bf16 head, unmodified build), vLLM FP8 and vLLM bf16 (eager, one stream). Each workload runs at 16 and at 32 generated tokens after a
128-token context. The per-token value is the difference of the two runs divided by the extra tokens (16 for llama.cpp; 32 for vLLM,
which decodes each job twice), so prompt and start-up cancel. Counters (the ones round 5 item I found exact): `GRBM_COUNT`,
`GRBM_GUI_ACTIVE`, `SQ_INSTS_VALU`, `SQ_WAVE32_INSTS`, `FETCH_SIZE`, `TCP_REQ`, `TCP_REQ_MISS`, plus `SQ_WAVES` as a control; one
profiler pass per group. `scripts/round6/ctr.sh` runs it, `analyse.py` makes the tables, `verify.py` re-adds the CSVs.

Checks before writing (MEASURED): I reran `analyse.py` on the raw CSVs and its output equals `table-profile_standard.md`. `verify.py`
(plain sums, no code shared with `analyse.py`) reproduces the llama.cpp matrix-kernel FETCH_SIZE, VALU, wave32, TCP_REQ and TCP_REQ_MISS
to 6 digits; for vLLM it sums only the `wvSplitK` kernels and its FETCH_SIZE is 0.0 % to 0.1 % below the table's matrix family (see the end).
Every counter grows from N=16 to N=32 in every family. The unit of FETCH_SIZE is KiB (INFERRED: with x1024 it lands on the file's weight
bytes to 0.1 %, which no other factor would).

### 1. Matrix-kernel bytes per token against the weights - MEASURED (ratio INFERRED from the header sizes)

| workload | FETCH_SIZE bytes/token | weight bytes (file header) | ratio | TCP_REQ/token |
|---|---|---|---|---|
| llama.cpp Q8_0 | 3,414,842,880 | 3,413,360,640 | 1.000 | 225,401,728 |
| llama.cpp FP8 (patched) | 3,239,946,640 | 3,237,206,144 | 1.001 | 316,753,792 |
| llama.cpp "auto" (FP8 + bf16 head, unmodified) | 3,609,439,768 | 3,606,583,424 | 1.001 | 173,282,560 |
| vLLM FP8 (eager) | 3,611,354,692 | 3,606,577,544 | 1.001 | 29,138,048 |
| vLLM bf16 (eager) | 6,429,834,992 | 6,425,149,440 | 1.001 | 50,892,928 |

Weight bytes are the layer matrices plus their scales plus the tensor the head reads (`output.weight`, or the tied embedding for
vLLM); the llama.cpp embedding table is not read by decode and is not counted (READ: headers, via `analyse.py`).
INFERRED: every runtime reads the weights once per token, with no re-reads and no skipped reads; the 0.1 % above 1.000 is of the size of
the small tensors in the sum and was not chased. llama.cpp issues many small requests (10 to 21 bytes per `TCP_REQ`), vLLM's `wvSplitK`
about 124 to 126; the request unit is not documented here, so read this as a comparison between rows, not as a line size.
Bytes per `GRBM_GUI_ACTIVE` cycle (INFERRED): Q8_0 87, FP8 89, vLLM FP8 91, vLLM bf16 86, "auto" 62.

### 2. FP8 against Q8_0, bytes - MEASURED, INFERRED

llama.cpp FP8 reads 174,896,240 bytes (5.1 %) fewer per token than Q8_0. The head is the same Q8_0 tensor (418,631,424 bytes) in both
files, so the whole saving is in the layers: 2,818,574,720 bytes for FP8 (1 byte per weight plus F32 scales) against 2,994,729,216 for
Q8_0 (34 bytes per 32 weights) (INFERRED, from the header sizes minus the head).

### 3. Vector instructions per token in the matrix kernels - MEASURED (per weight INFERRED)

The weight count read per token is 3,212,574,720 (28 layers x (2 x 3072^2 + 2 x 3072 x 1024 + 3 x 3072 x 8192) + 3072 x 128,256).

| workload | SQ_INSTS_VALU/token | per weight | SQ_WAVE32_INSTS/token | per weight | GRBM_GUI_ACTIVE/token |
|---|---|---|---|---|---|
| llama.cpp Q8_0 | 303,019,259 | 0.094 | 1,087,587,068 | 0.339 | 39,173,759 |
| llama.cpp FP8, patched | 636,968,225 | 0.198 | 1,750,209,277 | 0.545 | 36,488,127 |
| llama.cpp "auto", unmodified | 2,189,351,680 | 0.682 | 4,736,022,016 | 1.474 | 57,997,727 |
| vLLM FP8 | 116,827,904 | 0.036 | 224,146,816 | 0.070 | 39,804,914 |
| vLLM bf16 | 349,139,200 | 0.109 | 556,116,608 | 0.173 | 75,117,189 |

- MEASURED: patched FP8 uses 2.10 times Q8_0's VALU and 1.61 times its wave32 instructions in the same 169 dispatches.
- MEASURED: the same FP8 kernel (168 dispatches, 2,821,301,424 bytes unmodified and 2,821,296,112 patched) uses 2,088,927,232 VALU on the
  unmodified build and 594,771,968 on the patched one, 3.5 times. INFERRED: the patch's saving is the FP8-to-float conversion; bytes are
  identical, so the saving is all arithmetic. The counters do not show which instructions.
- vLLM FP8 is the lowest (0.036 per weight; 18 % of llama.cpp's patched VALU, 39 % of Q8_0's). Its decode kernel uses the packed FP8 dot
  product `v_dot4_f32_fp8_fp8` (READ in `llm-runtimes.md`); the counters here show low counts, not the opcode.
- INFERRED: converting a format costs instructions, not bandwidth. Whether instruction counts limit speed is not decided here: all
  kernels except the unmodified one run at 86 to 91 bytes per cycle.

### 4. Miss shares (`TCP_REQ_MISS / TCP_REQ`) - MEASURED

| workload | matvec | attention | other |
|---|---|---|---|
| llama.cpp Q8_0 | 19.4 % | 96.4 % | 97.7 % |
| llama.cpp FP8 patched | 28.4 % | 96.5 % | 96.2 % |
| llama.cpp "auto" unmodified | 48.1 % | 96.5 % | 96.1 % |
| vLLM FP8 | 100.1 % | 43.3 % | 82.2 % |
| vLLM bf16 | 100.0 % | 43.3 % | 86.0 % |

vLLM's matrix-kernel misses equal its requests (29,164,506 against 29,138,048 for FP8) and one elementwise kernel shows 109.1 %. INFERRED: a
share at or above 100 % means the two counters do not count the same unit, or wide streaming loads bypass the cache; it is not a hit rate.
What holds is that weight streams almost never hit in this cache. llama.cpp's lower share comes with many more, smaller requests per byte
(INFERRED: repeated small reads of the same line; not proven).

### 5. Attention against the key/value cache - MEASURED, INFERRED

| workload | attention FETCH bytes/token | main kernel alone | KV bytes at mean context 152.5 | ratio |
|---|---|---|---|---|
| llama.cpp (three files) | 32,530,192 / 32,536,392 / 32,542,032 | `flash_attn_tile` 29,732,864 / 29,740,016 / 29,740,000 | 17,489,920 | 1.860 (family), 1.70 (tile kernel) |
| vLLM (FP8, bf16) | 18,760,960 / 18,760,976 | `paged_attention_ll4mi_QKV` 18,552,576 / 18,552,328 | 17,489,920 | 1.073 (family), 1.061 (main kernel) |

- INFERRED, from two numbers: llama.cpp's tile kernel reads the cache for 256 positions, not the ~152 in use. K and V at 256 positions are
  29,360,128 bytes; the kernel fetched 29,732,864, 1.013 times that. READ: `FATTN_KQ_STRIDE 256` in `ggml/src/ggml-cuda/fattn-common.cuh`
  (line 9 of the checkout used). All counted positions (145 to 160) fall in the same 256-block, which fits the flat per-token bytes. The
  remaining 1.3 % and the 2.8 MB of `flash_attn_combine_results` I cannot attribute. Whether the padded half is read by design or the
  kernel skips work is not shown; the bytes were fetched.
- INFERRED: vLLM's paged attention reads whole 16-token blocks, so 145 to 160 tokens round to about 160 (1.05 times); 1.061 is of that order.
  vLLM's block size was not read in this round.
- INFERRED: at this short context attention is 0.9 % of llama.cpp's decode bytes (32.5 MB of 3.45 GB); the 256-position floor matters
  only below a context of 256.

### 6. vLLM FP8 against llama.cpp FP8: why 3.607e9 against 3.237e9 bytes - MEASURED, INFERRED

vLLM's matrix kernels read 371,408,052 more bytes per token than llama.cpp's patched FP8 (3,611,354,692 against 3,239,946,640). The heads
account for 369,380,096 of that: vLLM's bf16 head reads 788,011,520 bytes (tied embedding, 3072 x 128,256 x 2), llama.cpp's Q8_0 head
418,631,424 (MEASURED). That is 99.5 % of the gap (INFERRED); the layers are the same size in both (about 2.8186e9 bytes after taking out
the head; same 196 FP8 matrices; bf16 scales in vLLM, F32 in llama.cpp). So the gap is the output layer's format (the llama.cpp FP8
file quantises the head to Q8_0; the vLLM model keeps it bf16), not the FP8 path. vLLM also has fewer matrix dispatches per token (113
against 169, a different kernel split).

### 7. Clocks in item C - MEASURED

In `profile_standard` the "clocks before/after" lines of all 10 job logs read a memory clock of 1,124 MHz in every one of the 20 lines, a
core clock of 1,642 to 1,657 MHz and reported power of 73 to 77 W. In `auto` the dry run read about 1,408 MHz memory and a core clock near
3.5 GHz (from `RESULTS.md`, not re-read in a log). Every cycle count above is for the lower clocks.

## Files

`scripts/round6/` and `src/round6/` hold **this round's own copies** of the programs the two items used, with private paths replaced by
environment variables (set before running; each script's header names them). They are not the run folders' files byte for byte. I ran
`analyse.py` from this copy on the raw CSVs and its output equals the run's table.

| File | What it is |
|---|---|
| `src/round6/lt_bench.cpp` | The direct hipBLASLt GEMM timer (`lt_bench fp8\|bf16 M N K`), item R. |
| `scripts/round6/lt_shapes.sh` | Runs it for the four 3B projection shapes; `LT_BIN_DIR`, `ROCM_PARENT`. |
| `scripts/round6/ctr.sh` | Item C's ten gated profiler jobs; refuses if the performance level is not the label; `LLM_BUILD_ROOT`, `LLM_MODELS`, `VLLM_VENV`, `ROCM_DIR`. |
| `scripts/round6/analyse.py` | Item C's tables from the raw CSVs; `LLM_MODELS`. |
| `scripts/round6/verify.py` | Item C's independent re-sums, run from the run folder. |

## Where RESULTS.md and the logs disagree, or could not be confirmed

- Item C's `RESULTS.md` says `verify.py` reproduces the table's FETCH_SIZE for all 5 workloads to 6 digits. For llama.cpp it does. For vLLM it
  matches `TCP_REQ`, `VALU` and wave32, but its FETCH_SIZE is 3.60716e9 (FP8) and 6.42616e9 (bf16), against the table's 3,611,354,692 and
  6,429,834,992 (ratios to the file bytes 1.0000 and 1.0001, against 1.001 in the table), because it sums only the `wvSplitK` kernels and the
  table's matrix family also holds other GEMM-named kernels (about 4 MB per token). The table's definition is used above.
- Item C's `RESULTS.md` says vLLM FP8 "needs about a third of llama.cpp's" matrix instructions. From the counters it is 0.18 of the patched
  FP8 VALU and 0.13 of its wave32 instructions (a third is 0.39 of Q8_0's VALU). The numbers above are used.
- Item C's `RESULTS.md` clock figures for `auto` (memory 1,408 MHz, core about 3.5 GHz) were not re-read here.
- The build line of `lt_bench.cpp` is not recorded in either run folder; the one above is reconstructed.
- Everything else I checked against the logs agreed with item R's `RESULTS.md`: the six FP8 and two bf16 vLLM speed tables, the perplexity
  lines, and the byte, instruction and ratio arithmetic of item C (recomputed from the table).

## Parked, skipped and open

- **Why vLLM FP8 decode and prompt speed come in two levels** (about 159 and about 120 tokens/s at one stream) in both versions: not clocks,
  power or memory clock. Untested: host CPU contention (the jobs were pinned to cores 0 to 7 while the machine was in use), a choice made
  at engine start inside vLLM or its kernel cache.
- **Whether 10.2 is 5 to 8 % slower at the slow level at 4 and 8 streams:** one 10.2 run against two 10.1 runs. An A-B-B-A order with 3 starts
  each would settle it.
- **ROCm alone is not isolated:** the vLLM versions differ, and the 10.2 vLLM runs a torch 2.14 and triton mix that AMD does not state.
- **The bf16 pair has one run per version;** the GEMM numbers are one heuristic pick, not `hipblaslt-bench`'s search; no kernel trace inside
  vLLM was taken for item R.
- **A true 16-bit llama.cpp workload** (needs a conversion of the model first, a 6.4 GB CPU job): not done.
- **Graph mode and batches above 1 under counters** (vLLM ran eager at one stream): not done.
- **The opcodes behind the instruction counts and the 256-position padding** (no machine code was read in item C), and vLLM's block size: not
  examined.
- **Which part of `profile_standard` makes the counters count:** still open from round 5.
