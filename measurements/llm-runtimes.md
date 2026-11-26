# LLM data dive, round 4: PyTorch and vLLM next to llama.cpp, and the FP8 matrix hardware under an LLM

Measured 2026-11-18 on the LACT profile "Default" (the owner's overclock: 374 W cap, core offset +150 MHz, -50 mV, memory maximum
1409 MHz; the 1.2x regulator mod is in the card (the owner's statement, `notes/first-hour-9070xt-rocm10.md`), so reported watts are scaled down). **Every wall-clock number in this file is for the
overclocked profile with the power limit raised.** It is not comparable with rounds 1 to 3 (profile "Testing", stock clocks, 317 W) except where
the text says so. Memory clock: 1408 MHz until a GPU reset at 20:51:51 UTC (see "GPU hangs"); 1258 MHz in every job after it (`uclk_before` of each `.meta`; 22 of the 36 jobs at 1408 MHz are in `runs/20261118T202715Z-llm4/uclk1408/`, the 14 earliest (smoke tests, `t1` to `t4`, the first trace and vLLM smoke runs) are in the top folder of that run). Each table says which, and the text says it for every wall-clock number. The
expectations were committed before the first GPU job (`llm-runtimes-expectations.md`, never edited). Labels: MEASURED (a run file is named),
READ (a listing, a disassembly, a document), INFERRED (reasoning, not tested). Medians; every number is from a raw run file (the tables are
built by `scripts/llm4_tables.py` from `runs/20261118T202715Z-llm4/`, git-ignored; the build lists are `tables-1258.md` there and `tables-1408.md`, built from the `uclk1408/` folder).
Software: llama.cpp master f477c1b90 (build folder `build-gfx1201-rocm10`; the ROCm version of that build is not logged, the system's `rocprofv3` reports 10.1, `r4-config.txt`; unmodified and "patched" builds); PyTorch 2.15.0+rocm10.0 with transformers 5.23.0
(HuggingFace Llama modules, eager); vLLM 0.29.1.dev0+rocm10.1.0 on PyTorch 2.13.0+rocm10.1.0 (AMD's wheels). Models: Red Hat's FP8 checkpoints
of Llama 3.2 3B and Llama 3.1 8B (static per-tensor activation and weight scales), Meta's 16-bit 3B (an Unsloth re-upload of it; equality with Meta's own release was not checked, the repository is gated), and the study's GGUF files.

## Plain summary

1. **The FP8 matrix hardware is used, but only by the big GEMMs.** PyTorch's `torch._scaled_mm` and vLLM's prompt path call hipBLASLt kernels
   whose machine code holds `v_wmma_f32_16x16x16_fp8_fp8` (READ, disassembly of the code objects the traces name). vLLM's one-token decode
   uses a different kernel (`wvSplitKQ`, vLLM's own) that works on FP8 weights with `v_dot4_f32_fp8_fp8`, a vector-unit dot product, not the matrix unit
   (READ). llama.cpp never uses either (it widens FP8 or uses int8).
2. **Speed, 3B, prompt of 512 tokens (tokens/s, memory 1258 MHz):** llama.cpp Q8_0 12,651, FP8 patched 9,280; PyTorch FP8 13,748 (bf16 13,675: no gain); vLLM FP8 18,956
   (bf16 15,113). **Generation, one stream (also memory 1258 MHz):** llama.cpp FP8 patched 145 to 147, vLLM FP8 112 (146 in a shorter test), PyTorch eager 60 (launch-bound; bf16 62).
   The FP8 matrix GEMMs alone reach up to 236 TFLOPS (3B shapes) and 259 (8B), about 56 to 67 % of the 420 TFLOPS FP8 peak, depending on the clock assumed (3.19 GHz, the sampler's 99th percentile, or 2.95 GHz, its awake median in the 8B job).
3. **PyTorch's FP8 path is not trustworthy on this card as set up, and the hangs may not be its fault.** The GPU hung four times during its jobs (a full GPU reset each time, `runs/llm4-meta/kernel-log-gpu-resets.txt`; the desktop shares that graphics ring, INFERRED),
   gave NaN in 3 of 12 perplexity runs, and gave different numbers from run to run (9.69 to 9.73 on a 4-chunk set), while the same PyTorch in bf16 is bit-identical
   across runs, repeated FP8 GEMMs are bit-identical, and vLLM FP8 (same hipBLASLt kernel family on ROCm 10.1) is identical across runs. Cause not found. **The debugging limit was breached:** it was reached at hang 3 and about 20 more PyTorch FP8 jobs were started afterwards (see "GPU hangs"). PyTorch FP8 is parked by the owner.
4. **Against the true weights, FP8 and Q8_0 are about equally far, and both are about 180 to 210 times farther than 16-bit.** Mean KL divergence from the F32 run of Meta's 3B
   (8 chunks): llama.cpp 16-bit 2e-5, FP8 3.7e-3 (both builds), Q8_0 4.3e-3; PyTorch bf16 3.6e-4, PyTorch FP8 2.2e-2 (one run), vLLM FP8 only through the target-token
   log-probability (0.113 against 0.015 for bf16). Round 2's "Q8_0 is 230 times worse than FP8" was distance from the exact FP8 weights; from the true model the two are the same size (`llm-workload-2.md` now carries an "Updated in round 4" note).
5. **Everything runs hot on this profile, and the runner's thermal stop was too slow (now fixed).** From a 50 to 53 C start the 20 ms sampler saw: the llama.cpp Q8_0 prompt loop reach 95 C
   at 2.3 s (the job was stopped at 5.0 s of wall time); the llama.cpp FP8 patched loop only at 20.3 s of 21.4 s (no stop; the runner's poll read 93 C, the sampler 96 C); the vLLM prompt loop never (peak 86 C).
   Core medians while awake 3.33 GHz (llama.cpp FP8 patched), 3.21 (Q8_0), 3.12 (vLLM); generation runs at 3.5 to 3.57 GHz at 240 to 294 W (llama.cpp; vLLM 142 W).
   The FP8 GEMM jobs reached **103 C (3B) and 95 C (8B)** in the sampler while the runner's 0.5 s poll read 63 and 81 C and stopped nothing (about 20 C per second; the critical limit is 110 C).
   The stop is now a 20 ms guard that ends the job's process group at 95 C (section "Thermal guard").

## Expectations against results

| # | Expectation | Result | Verdict |
|---|---|---|---|
| O1 | Red Hat 3B FP8 = the 3B FP8 GGUF's source | all 196 FP8 tensors bit-identical (after the q/k reorder), 392 scales, 58 other tensors equal; only `output.weight` (Q8_0 in the GGUF) and `rope_freqs` have no twin (`runs/20261118T202715Z-llm4/r4-origin-3b.txt`) | right |
| O2 | vLLM runs this FP8 checkpoint on gfx1201 (65 %); may not fit the 10 GB host cap (20 %) | runs, under the 10 GB cap, 3B and 16-bit; the 8B ran with `--gpu-util 0.93 --max-len 2560` (`r4v-8b-fp8.log`: 1.64 GiB of KV cache in use); an earlier 0.80 attempt that left no KV cache is not logged and is not evidence | right |
| R1 | torch: hipBLASLt FP8 kernels with FP8 WMMA (90 %); vLLM hipBLASLt FP8 (50 %) or Triton/AITER (20 %) | torch and vLLM prompt: `Cijk_..._F8BS_...` kernels, `v_wmma_f32_16x16x16_fp8_fp8` in 9 of 9 distinct FP8 kernels of the torch traces and 12 of 12 of the vLLM traces (every one the final traces name); the traced bf16 kernels (8 torch, 13 vLLM) hold no FP8 WMMA. vLLM decode: `wvSplitKQ_hf_sml_` with `v_dot4_f32_fp8_fp8` (48), no WMMA | right for the GEMMs; wrong in that vLLM's decode kernel is a third kind |
| R2a | torch pp512 3B FP8 8,000 to 16,000 (best 11,000), FP8 1.0 to 1.3 times bf16 | 13,748 and 13,675: 1.005 times | right on range, bf16 gain smaller than 1.0 to 1.3 (the 1.0 edge) |
| R2b | torch decode 1 stream 20 to 50 tok/s; FP8 not faster than bf16; near-linear streams | 59.8 FP8 and 61.7 bf16 (above 50: wrong on the range); FP8 not faster (right); bf16 1 to 8 streams 61.7 to 463.9 (7.5 times, right) | partly right |
| R2c | vLLM 3B FP8 1 stream 110 to 200; 8 streams 500 to 1,000; pp512 8,000 to 20,000 | 111.7 (146 in a 32-step test); 749.1; 18,956 | right |
| R2d | llama.cpp on this profile: pp +2 to +10 %, tg within +-3 % | against round 1: pp512 3B -0.3 / +1.6 / +3.3 % (unmod, patched, Q8_0) at 1408 MHz; tg128 +2.5 / +0.6 / +0.5 %; at 1258 MHz tg128 is 7 to 8 % lower for patched FP8, Q8_0 and bf16 but only 2.2 % (3B) and 2.6 % (8B) for the unmodified FP8 build | pp: mostly below the range (wrong low end); tg right at 1408 |
| R2e | best FP8 GEMM below 50 % of peak; first-hour 166 TFLOPS reproduces within 15 % | best 236 (3B) and 259 (8B) TFLOPS = 56 to 67 % of the scaled 420 (clock assumed 3.19 or 2.95 GHz) | wrong (above 50 %); the first-hour shape (4096 x 4096 x 512) was not rerun |
| R3a | torch prefill 512: FP8 GEMMs 40 to 65 % of kernel time, quantize plus elementwise 15 to 30 % | gemm_fp8 46.4 %; quantize casts 10.7 %, other elementwise 25.7 %, copies 10.3 % | right |
| R3b | torch decode: gaps over 50 % of the token | gaps 32.9 % (3B FP8, union of kernel intervals, profiler timestamps) | wrong |
| R3c | vLLM decode gaps under 15 % | 3B FP8 31.8 %, 3B bf16 11.2 %, 8B FP8 9.2 % | right for two of three |
| R3d | no widen pass in torch or vLLM | none seen | right |
| R4a | prompt loops at the cap (PPT0 at least half the samples), core 3.0 to 3.4 GHz, hotspot 85 to 95 C, one stop possible | llama.cpp FP8 patched: PPT0 873 of 1,070 samples, core 3.33 GHz, 373 W, sampler hotspot 91 to 94 C from 19.0 s to 20.28 s and one 96 C sample at 20.30 s (the runner's poll read 93 C: no stop); Q8_0 loop: 96 C, stopped at 5.0 s; vLLM prefill: 3.12 GHz, 293 W reported median, hotspot 86 | right; torch FP8 loops hung instead |
| R4b | eager decode under 200 W | torch decode loop not measured (FP8 hung); bf16 and FP8 llama.cpp decode 240 to 294 W | not tested |
| R4c | memory clock 1408 in every job | 1408 until the GPU reset, 1258 after | wrong |
| R5a | torch 16-bit perplexity 9.3 to 9.9; FP8 within 0.15 | 9.7223 (bf16, bit-identical in 5 runs); FP8 9.69 to 9.73 | right |
| R5b | KL from true: FP8 1e-3 to 1e-2; Q8_0 within 1.5 times of FP8 | 3.69e-3 and 4.28e-3 (1.16 times) | right |
| R5c | torch FP8 1.2 to 3 times llama.cpp FP8; torch bf16 below 5e-4, top token 99.5 % | torch FP8 5.9 times (2.19e-2, one run of a path that is not reproducible); torch bf16 3.6e-4 and 98.7 % top token | wrong on the FP8 factor (as far as one run says) and on the top-token figure |
| R5d | vLLM FP8 perplexity within 0.05 of torch FP8 | 9.7256 against 9.69 to 9.73 (4 chunks) | right |

## R1: do the FP8 paths use the FP8 matrix instructions?

- **PyTorch (MEASURED trace, READ code).** Each FP8 linear layer calls `torch._scaled_mm` (activation quantised with the checkpoint's static scale, weights FP8 with their scale);
  the traced kernels are hipBLASLt Tensile kernels `Cijk_Alik_Bljk_F8BS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs_MT{128x128x64, 64x64x128, 64x64x64}_...`. Their code objects were cut out of the
  `.co` bundles (`clang-offload-bundler`), the kernel symbol's code disassembled to the next function symbol (`scripts/llm4_isa.py`): the FP8 kernels contain
  `v_wmma_f32_16x16x16_fp8_fp8`: all 9 distinct FP8 `Cijk_` kernels of the final PyTorch traces (28 to 448 each, `isa-final/isa-final-torch-fp8.txt`; the first, smaller listing `isa-torch-hipblaslt.txt` held 4 of them); the 8 traced bf16 kernels hold only `v_wmma_f32_16x16x16_bf16` (`isa-final-torch-16bit.txt`).
  In a prompt pass of the 3B, 183 FP8 GEMM kernels, 46.4 % of the kernel time (`r4tr-t-prefill-3b-fp8`). In one-token decode PyTorch pads the one row to 16 and uses the same kernels (195 per token, 57.3 % of kernel time).
- **vLLM prompt (MEASURED, READ).** The log says `Selected ROCmFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8`; the traced kernels are the same hipBLASLt F8 family (ROCm 10.1 library), 112 per pass,
  61.9 % of kernel time; ISA: `v_wmma_f32_16x16x16_fp8_fp8` in all 12 distinct FP8 kernels of the final vLLM traces, 160 to 448 each (`isa-final/isa-final-vllm-fp8.txt`; the first listing held 8 of them); the 13 traced bf16 kernels hold no FP8 WMMA (`isa-final-vllm-16bit.txt`). The activation quantise is fused with the norm and the activation in Triton kernels (`triton_red_fused_..._rms_norm_...`).
- **vLLM decode (MEASURED, READ).** The one-token matrix products run in `wvSplitKQ_hf_sml_<bf16, Float8_e4m3fn, 32, 2, 16, 16, 2, 1>` (vLLM's `_rocm_C`, 112 launches, 4.66 ms of 6.92 ms of kernel time, 67 %); in its gfx1201 code
  there are 48 `v_dot4_f32_fp8_fp8` and no `v_wmma` (the bundle was cut out of `_rocm_C.abi3.so` with `llvm-objdump --offloading`; the disassembly of the traced bf16 variant and of the fp16 twin is saved in `isa-final/isa-vllm-wvSplitKQ.txt` and the two `-listing-` files, 487 and 502 lines, 48 dot4 in each). So decode uses the FP8 dot-product instruction on the vector unit; the matrix unit sits idle. The LM head stays 16-bit (`wvSplitK_hf_sml_`, bf16).
- **16-bit paths (READ):** bf16 `Cijk_..._BBS_...` kernels hold `v_wmma_f32_16x16x16_bf16` only; vLLM bf16 decode runs `wvSplitK_hf_sml_` (bf16) for 94.1 % of its kernel time (95.9 % with the other 16-bit GEMM kernels).
- Not read: the attention kernels' (Triton `_fwd_kernel`, `paged_attention_*`, torch's `flyc_attn_fwd`) instructions.

## R2: speed (overclocked profile, power limit raised; tokens/s)

Prompt (3B unless marked; memory 1258 MHz, `tables-1258.md`; llama.cpp medians of 5, torch medians of 7, vLLM medians of 3):

| runtime | pp128 | pp512 | pp2048 |
|---|---|---|---|
| llama.cpp FP8 unmodified | 4,319 | 8,746 | 11,091 |
| llama.cpp FP8 patched | 4,750 | 9,280 | 12,058 |
| llama.cpp Q8_0 | 6,834 | 12,651 | 13,040 |
| llama.cpp 16-bit (bf16 GGUF of Meta's weights) | 6,241 | 12,362 | 13,575 |
| PyTorch FP8 | 5,903 | 13,748 | not run (hang) |
| PyTorch bf16 | 6,244 | 13,675 | 16,411 (median of 5 repetitions, a development run: `r4t-pp2048-3b-bf16`, samples 15,816 to 16,497) |
| vLLM FP8 | 8,320 | 18,956 | 21,863 |
| vLLM bf16 | 7,036 | 15,113 | 16,662 |
| 8B: llama.cpp FP8 patched / Q8_0 | 1,816 / 3,076 | 4,550 / 6,134 | 5,693 / 6,332 |
| 8B: vLLM FP8 | 5,793 | 11,163 | 11,562 |

(The 8B PyTorch prompt rows were not run after the hangs; the 8B PyTorch FP8 perplexity ran once, 7.4752: one run of a path that is not reproducible.) pp2048 for llama.cpp uses a micro-batch of 2048.

Generation, 128 prompt + 128 generated tokens per stream, streams 1 / 4 / 8 (tokens/s; llama.cpp S_TG of `llama-batched-bench`; memory 1258 MHz):

| runtime | 1 | 4 | 8 |
|---|---|---|---|
| llama.cpp 3B FP8 patched | 145 | 443 | 800 |
| llama.cpp 3B Q8_0 | 133 | 467 | 797 |
| llama.cpp 3B 16-bit | 78 | 300 | 565 |
| PyTorch 3B FP8 | 59.8 (streams 1 to 3: 59.8, 117.0, 171.8; hung at 4) | - | - |
| PyTorch 3B bf16 | 61.7 | 240.6 | 463.9 |
| vLLM 3B FP8 | 111.7 | 436.3 | 749.1 |
| vLLM 3B bf16 | 89.6 | 340.8 | 639.3 |
| llama.cpp 8B FP8 patched / Q8_0 | 72 / 66 | 216 / 246 | 420 / 437 |
| vLLM 8B FP8 | 71.9 | 278.6 | 513.3 |

- **The memory clock matters and changed during the round.** llama.cpp at 1408 MHz (jobs before the reset, `tables-1408.md`): 3B pp512 FP8 unmod / patched / Q8_0 8,951 / 9,671 / 12,848; tg128 80.7 / 159.7 / 147.6; 8B pp512 4,491 / 4,759 / 5,924 (Q8_0 73.0 tg128).
  Against round 1 (profile Testing, 1408 MHz): pp512 -0.3 / +1.6 / +3.3 %, tg128 +2.5 / +0.6 / +0.5 % (3B), so the overclock buys almost nothing in llama.cpp prompts or generation; the 150 MHz core offset
  is spent against the 374 W cap. Memory 1408 to 1258 costs llama.cpp generation 7 to 8 % for the patched FP8, Q8_0 and bf16 files (159.7 to 147.4 for the patched 3B; 8B patched 78.7 to 73.0, 8B Q8_0 73.0 to 67.3, 3B bf16 85.7 to 79.0), as the bandwidth ratio predicts, but only 2.2 % (3B, 80.7 to 78.9) and 2.6 % (8B, 38.6 to 37.6) for the unmodified FP8 build (INFERRED: that build's generation is bound by something other than memory bandwidth). Both clocks: `uclk1408/` and the top folder.
- **vLLM's 112 tokens/s is protocol-dependent (INFERRED, unexplained).** At 128 generated tokens, one stream: 111.7 (112.1 and 112.0 in two controls, detokenising on or off); with 32 generated tokens at the same
  clock 145.7; at 1408 MHz and 32 tokens 158.7 (`vllm-smoke3`). The 128-token protocol matches llama-batched-bench; why it is 23 % slower per token than 32 was not found.
  *Update, round 5 (2026-11-20, `round5.md` item H):* the gap was **not reproduced**. In one warmed engine (profile Default, memory 1258 MHz) 32 and 128 generated
  tokens gave 151.5 and 147.7 tokens/s by this same subtraction method, and the host step time was 6.82 ms (146.5 tokens/s) at every context length (KV 129 to 288,
  +0.15 %). The 2.5 % between 32 and 128 is the method (N tokens over N - 1 extra steps). The 111.7 and 145.7 above are not shown to come from context length;
  the likely source is the state of that run's engine or the requests before the timed ones (INFERRED, the old protocol was not rerun).
  *Update, round 6 (2026-11-20, `round6.md` item R):* vLLM FP8 decode is **bimodal per engine start**: the same command gave about 159 or about 120 tokens/s at one stream (same day, same clocks, memory 1408 MHz) in both ROCm 10.1 (vLLM 0.29.1: 159, 120, 120) and ROCm 10.2 (vLLM 0.32.1.dev0: 118, 159, 159), over three starts each, with no clock or power difference between the levels. That fits round 4's 111.7 against 146 and round 5 item H's 146.5, and round 4's numbers sit at the slow level (INFERRED; the cause is unknown and the memory clocks differ). The R2 vLLM decode numbers above are therefore one of two levels, not a typical value.
  *Related public report (READ 2026-11-25):* [ROCm/TheRock issue 9228](https://github.com/ROCm/TheRock/issues/9228), opened 2026-06-11, reports the same kind of split on a Radeon AI PRO R9700 (gfx1201) with vLLM 0.22.1 and Qwen3-14B-FP8: about 33 or about 26 tokens/s at one stream, fixed at process start, with no clock or temperature difference and correct output in both states. Card, model, versions and ratio differ (26/33 there, 120/159 here). Comments there tie a similar split to cudagraph replay and to torch's compiled-graph cache (`VLLM_USE_AOT_COMPILE=0`); this study did not test either. Same mechanism: not shown.
- **PyTorch eager decode is launch-bound (MEASURED):** 16.7 ms per token step for the 3B at 1 stream, nearly flat from 1 to 3 streams (16.7, 17.1, 17.5 ms), 2,310 kernels per token with FP8 (1,136 with bf16).
- **FP8 against 16-bit:** in llama.cpp FP8 patched generation is 1.9 times bf16 (the bytes), prompts 0.75 times (the widen); in vLLM 1.25 times for both (the weight bytes alone would suggest more; no byte calculation was made for this round); in PyTorch eager 0.97 times (59.8 against 61.7; launch-bound).
- **GEMM rate (MEASURED, `r4t-gemm-3b`, `-8b`, 1258 MHz, median of 50 calls with random data; TFLOPS = 2 M N K / time):**

| tokens N | 3B shapes: FP8 best / bf16 best | 8B shapes: FP8 best / bf16 best |
|---|---|---|
| 16 | 30 (gate/up) / 25 | 42 (down) / 20 |
| 128 | 164 / 109 | 200 / 97 |
| 512 | 201 / 134 | 221 / 129 |
| 2048 | 236 / 137 | 259 / 151 |

  The FP8 peak is 420 TFLOPS at 3.2 GHz (`matrix-timing.md`; 131 TFLOPS per GHz is INFERRED from that). The clock during each timed call was not recorded; the sampler of the same jobs shows (`beh/r4t-gemm-*`): 99th percentile 3.19 GHz in both jobs, awake median 2.95 GHz in the 8B job (45 awake samples), 90th percentile 3.02 GHz in the 3B job. So 236 TFLOPS is 56 % of the scaled peak at 3.19 GHz and 259 TFLOPS is 67 % at 2.95 GHz: **56 to 67 %, depending on the clock assumed** (INFERRED). The first-hour 166 TFLOPS shape was not rerun.
  The GEMM job ran at both memory clocks, and the rates agree only partly (FP8 rows, 1408 MHz in `uclk1408/r4t-gemm-*`, 1258 MHz in the top folder): the 3B shapes agree within 3 % in 17 of 20 rows (3B gate/up at N = 512: 206 at 1408 and 201 at 1258 TFLOPS; N = 2048: 241 and 236; the largest difference is `q` at N = 16, 12.4 against 5.8), the 8B shapes in only 9 of 20, with the N = 128 and 512 rows up to 32 % apart (8B `q` at N = 128: 125.7 against 85.4; `o` -24 %; gate/up -21 %). The 8B best at N = 512 is 221 (1258) against 247 (1408). These are 2 to 5 s jobs under the power limit with throttle flags in 20 (3B) and 34 (8B) sampler samples, so clock and power noise is the likely reason (INFERRED); the table above is the 1258 MHz set. Random data, one data pattern: real activations may draw more power.

## R3: where the time goes (kernel traces, rocprofv3; 3B FP8 unless marked)

`scripts/llm4_trace.py` cuts the region between two marker kernels. **Caveat (MEASURED): kernel timestamps overlap in these traces, and the round 3 artifact finding does not explain it** (round 3 covers overlaps of up to 8 us in a synthetic probe; these traces overlap so much that the kernel intervals cover only 18 % of a 3B vLLM prompt's wall time) (PyTorch and vLLM traces: 544 to 1,393 overlapping neighbour pairs in the prompt traces and 858 to 1,251 in the PyTorch decode traces (447 to 908 in the vLLM decode traces); the sum of kernel durations is 0.68 to 0.95 of the wall time of a pass, e.g. 0.80 to 0.89 for PyTorch prompts, while the union of the kernel intervals is only 18 % of the wall time for the 3B vLLM prompt, 52 % for the 8B vLLM prompt and 52 to 69 % for PyTorch prompts; negative gaps also appear in the llama.cpp traces). All kernels are on one queue, so this is not multi-queue concurrency. In round 1 the first llama.cpp traces had no overlaps and the rerun of `trace-tg128-8b-fp8-patched` had 5 overlapping pairs and failed its `no-overlap` check (`llm-workload.md`). So shares of kernel time and kernel counts are used; **gap figures of prompt traces are unreliable** (marked in the table); decode traces overlap little (union 67.1 % against a kernel sum of 67.7 % in the 3B PyTorch FP8 decode), so their gaps are sound. A cross-check of the share numbers: 14.63 ms of FP8 GEMM per 3B prompt pass is 2.89 TFLOP (2 x 2.82 billion weights x 512 tokens, from the model's shapes) in 14.63 ms = 197 TFLOPS, against 201 TFLOPS measured by the GEMM job at N = 512, so the GEMM durations are credible; the small elementwise kernels' shares are less certain. All traces in this section ran at memory 1258 MHz (`uclk_before=1258` in the `r4tr-*.meta`) on the profile Default. Why the overlaps appear now is not known (profile, the GPU reset and a different library state are all untested). llama.cpp's own round-1 pipeline run on this profile (`trace-*`) fails its `no-overlap`, `total-kernel-count` and `passes-identical` checks, and its absolute times are not quoted.

| trace (per pass or per token) | wall ms | kernels | gaps (union) | largest groups, share of kernel time |
|---|---|---|---|---|
| PyTorch FP8 prompt 512 | 39.6 | 1,785 | 41 % (unreliable) | FP8 GEMM 46.4, other elementwise 25.7, activation quantise casts 10.7, copies 10.3 |
| PyTorch bf16 prompt 512 | 39.2 | 1,092 | 48 % (unreliable) | bf16 GEMM 73.6, elementwise 18.4 |
| PyTorch FP8 decode, 1 stream | 21.4 | 2,310 | 33 % | FP8 GEMM 57.3, elementwise 13.4, copies 11.6, LM head 9.1 |
| PyTorch bf16 decode, 1 stream | 18.7 | 1,136 | 20 % | (see `r4tr-t-decode-3b-bf16.report.txt`) |
| vLLM FP8 prompt 512 | 26.5 | 395 | (unreliable) | FP8 GEMM 61.9, attention 16.0, fused Triton 13.7 |
| vLLM FP8 decode, 1 stream | 9.8 | 366 | 32 % | FP8 GEMM group 69.1 (`wvSplitKQ` alone 67.3, plus a few hipBLASLt F8 kernels), bf16 LM head 18.3, attention 5.5, fused Triton 4.2 |
| vLLM bf16 decode | 12.2 | 340 | 11 % | bf16 matrix-vector 95.9 (`wvSplitK_hf_sml_` alone 94.1) |
| 8B: PyTorch FP8 prompt / vLLM FP8 prompt | 69.9 / 46.1 | 2,173 / 447 | 31 % / 48 % (both unreliable) | FP8 GEMM 55.6 / 76 (see reports) |
| 8B: PyTorch FP8 decode / vLLM FP8 decode | 31.7 / 15.0 | 2,646 / 415 | 25 % / 9 % | FP8 GEMM 71 / 83 (vLLM `wvSplitKQ` alone 81.5, bf16 LM head 12) |

- PyTorch needs 1,785 kernels for a 28-layer prompt pass (64 per layer) against vLLM's 395: the eager path launches the quantise, clamp, cast, norm, rope and residual steps as separate kernels.
  The activation quantise (cast and clamp) alone is 10.7 % of the kernel time; the FP8 GEMMs are less than half.
- PyTorch eager FP8 decode is 21.4 ms per token under the profiler (16.7 unprofiled) and spends a third of it between kernels; vLLM's graphs bring this to 9.8 ms (9 to 11 % gaps for the 8B and bf16).
- The `__amd_rocclr_fillBufferUnAligned` kernel appears about once per 68 kernels in the PyTorch 3B prompt trace (26 per pass of 1,785 kernels; none in the decode region) (377 copy/fill kernels per 3B prompt pass, 457 for the 8B, 815 per 3B token); it comes with `_scaled_mm` (INFERRED: a hipBLASLt workspace or argument fill).

## R4: clocks, power, hotspot (20 ms sampler, `beh/r4b-*` and `beh/r4t-gemm-*`, memory 1258 MHz, loops cut short by the 95 C stop)

| recording | core MHz median awake / max | reported W median / max | hotspot max C (**20 ms sampler**; the runner's own poll value is in the text) | throttle flags (samples) |
|---|---|---|---|---|
| llama.cpp FP8 patched, prompts (21.4 s loop, no stop; runner poll max 93 C) | 3,333 / 3,415 | 373 / 429 | 96 (95 C first reached at 20.3 s) | PPT0 873 of 1,070; TEMP_HOTSPOT 7 |
| llama.cpp Q8_0, prompts (stopped by the poll at 5.0 s of wall time; 95 C first reached at 2.3 s) | 3,210 / 3,274 | 309 / 395 | 96 | PPT0 31; TEMP_HOTSPOT 21 of 132 |
| llama.cpp FP8 patched, generation | 3,476 / 3,497 | 294 / 312 | 78 | none |
| llama.cpp Q8_0, generation | 3,529 / 3,539 | 240 / 267 | 69 | none |
| vLLM FP8, prompts | 3,124 / 3,219 | 293 / 424 | 86 | TEMP_HOTSPOT 17 of 1,807 |
| vLLM FP8, generation | 3,569 / 3,574 | 142 / 422 | 84 | TEMP_HOTSPOT 7 |
| FP8 GEMM job, 3B shapes (`r4t-gemm-3b`, 3.7 s) | p99 3,187 | 410 to 435 in the stalls | **103** (runner poll: 63) | TEMP_HOTSPOT 20, PPT0 7 of 185 |
| FP8 GEMM job, 8B shapes (`r4t-gemm-8b`, 4.7 s) | awake median 2,946 / p99 3,191 | - | **95** (runner poll: 81) | TEMP_HOTSPOT 31, PPT0 18 of 233 |
| PyTorch FP8, prompts | hung after 10.9 s (see below) | | | |

- Against round 1 (Testing profile, 317 W cap, memory 1408): llama.cpp prompts ran at 2.94 to 3.08 GHz and 316 W; now 3.33 GHz at a reported 373 W (the 374 W cap) with the hotspot at the stop within seconds.
  Generation ran at 3.31 to 3.39 GHz and 248 to 316 W; now 3.48 to 3.53 GHz and 240 to 294 W. The FP8 patched decode still draws about 54 W more than Q8_0 at the same bandwidth (294 against 240 median), as in round 3 (57 W).
- vLLM's generation draws far less than its prompts (median 142 W) but the core sits at 3.57 GHz, the highest clock any job reached in a running state.
- Heat, per job (start 50 to 53 C): llama.cpp Q8_0 prompts reached 95 C at 2.3 s of the sampler record (the runner stopped the job at 5.0 s of wall time); llama.cpp FP8 patched prompts only at 20.3 s of 21.4 s, and **were not stopped** (the runner's poll peaked at 93 C, `hotspot_max=93`, `thermal_stop=0`; the sampler read 91 to 94 C from 19.0 s to 20.28 s, one sample of 96 C at 20.30 s, then the loop ended); the vLLM prompt loop (36 s) peaked at 86 C. Stops by the runner: `r4t-detgemm` at 9.1 s of wall time (96 C), the F32 reference run of the 3B at 8.0 s (95 C, `stale/r4q-base-3b-true-f32`, 12 of 25 chunks; the rerun with 8 chunks, `stale2/` and the top folder, ended without a stop at 84 and 87 C).
- **The runner's guard was blind to short bursts.** In the two FP8 GEMM jobs the sampler saw 103 C (3B: 57 C until 1.82 s of the sampler record, 103 C at 2.16 s: 46 C in 0.34 s, at 410 to 435 W, 0.14 s above 90 C) and 95 C (8B) while the runner's 0.5 s poll recorded 63 and 81 C and stopped nothing. The hotspot rose about 20 C per second in the prompt loops (Q8_0: 50 to 95 C in 2.3 s) and about 135 C per second in that GEMM burst, so a 0.5 s poll can overshoot a 95 C stop by 10 C or more, and the card's critical limit is 110 C. Fixed before any further GPU job: see "Thermal guard" below.

## Thermal guard (added after round 4, measured without a GPU job except one real check)

- Mechanism: `watch_card guard` (`src/study/watch_card.cpp`) runs beside every gated GPU job and reads the hotspot file and the memory (VRAM) temperature file (`temp3`, critical 108 C) every 20 ms (measured period: median 20.00 ms, p99 20.05 ms, longest 20.06 ms in the first real check). A reading counts only when the whole file is digits: `nan`, `-nan`, `inf`, `96abc`, an empty or missing file and a reading of 0 are all "unreadable" (an earlier version let `nan` and `96abc` through). At a hotspot of 95 C, or when the hotspot file is unreadable (the runner's old 999 C value was exactly that: an unreadable sensor, not a temperature), or at a memory temperature of 100 C, it sends TERM to the job's process group (the job leads one because `timeout` does) and to the job's whole systemd scope (`systemctl --user kill`, so a child that left the group with `setsid` is stopped too), KILL to both 0.5 s later if anything is left (the guard ends only when the group and the scope are empty), writes one `THERMAL STOP:` line with the reading, the time and the UTC time to the job log, and the runner marks the job `STOPPED` (exit 8, `thermal_stop=1` in the `.meta`). An unreadable memory sensor with a good hotspot is one `WARNING` line, not a stop. The `.meta` `hotspot_max` comes from this 20 ms guard, and every job leaves a `<job>.guard.csv` (time, hotspot, memory temperature, read time).
- If the guard itself ends while the job still runs (crash, SIGKILL, any status but 0 and 8), the runner notices at once (`wait -n` on both), logs `GUARD LOST`, stops the job (TERM, KILL after 0.5 s, group and scope), records `guard_lost=1` and exits 10. A guard that hangs in a blocked sysfs read without exiting is not detected (parked).
- Measured with a dummy job and fake sensor files (`scripts/study-llm.sh selftest`, `selftest-guard`; not the card): hotspot to TERM received by the job, four trials per full run: 19, 9, 22 and 7 ms in the first full run (`runs/llm4-fix/stfull.out`), 13, 9, 13 and 19 ms in the second (`stfull2.out`), 10, 2, 10 and 20 ms in the third (`stfull3.out`), and 15, 12, 11 and 6 ms in the run of the last fix round (`runs/fix5/full.out`); the limit asked for is 100 ms (one period of 20 ms plus the read; the 22 ms includes the dummy's `date` call). In that last run: memory temperature file to TERM 7 ms; a `setsid` child got TERM 12 ms after the hotspot change (and one that ignores TERM was ended by the scope's KILL); the guard killed 2 s into a job: the job received TERM 1 ms later (the clock read of the test has about 1 ms resolution), runner exit 10. Also: the job's child process ended too; a process in the runner's own group was untouched; a vanished file, `n/a`, `nan`, `-nan`, `inf`, `96abc`, `0` and a file reading 999 C each stopped the job. Controls: a stream up to 94 C and a memory temperature of 99 C do not stop a job; an unreadable memory sensor with a good hotspot only logs a warning. **Each new check was seen to fail on its break** (records in `runs/fix5/breaks/`, git-ignored): scope signals removed (the `setsid` children survive: `break5i-*` fail), scope KILL removed (the TERM-ignoring `setsid` child survives), the old `strtod` reading (`nan`, `-nan`, `96abc`, `0` not stopped), memory sensor not given to the guard (`break5h-*` fail), unreadable memory as a stop (`break5h-mem-control` and `-warning` fail), the runner waiting on the job alone (`break5j-*` fail). Earlier breaks: a 400 ms period (TERM after 298 to 392 ms), a signal to the group leader only, a lowered limit (false stop of the control). In the break runs the surviving children kept the job lock, so some later cases of the same run failed too (lock wait); the cases named are the ones that fail by themselves.
- Real check (`runs/20261119-llm4fix/guard-real-q8-pp`, 3B Q8_0 prompt loop, gated runner, no change to the 95 C limit): the loop started at 53 C; the guard read the hotspot 128 times in 2.54 s at the 20 ms rate (no missed slot), read 95 C at 2.500 s and sent TERM 0.1 ms later; the job ended with exit 143, `hotspot_max=95` (the runner's old poll stopped the same loop at 5.0 s with the sampler at 96 C). Memory clock 1408 MHz before and after (the owner's screens were on).
- Real check of the final guard (`runs/fix5/real2/guard-real2-q8-pp`, 3B Q8_0 `llama-bench` prompt loop, 512 tokens, through the gated runner, memory clock 1408 MHz before and after): 151 guard samples in 3.001 s at the 20 ms rate (period median 20.00 ms, p99 20.61 ms, longest 21.53 ms, no missed slot); hotspot 53 C until about 2.0 s, 89 C at 2.44 s, 94 C at 2.94 s and 95 C at 2.960 s, when the guard sent TERM to the process group and the job's scope 0.1 ms after the reading; the memory temperature read 58 to 60 C at the start and 64 C at the end (`guard.csv` third column, maximum 64 C against the 100 C limit); wall time 3.0 s, exit 8, `thermal_stop=1`, `guard_lost=0`, no scope left behind. The first attempt of this check waited 600 s in the game gate (a game was running) and was skipped; this one ran after the game ended.

## GPU hangs (PyTorch FP8; parked by the owner)

**Update, 2026-11-19 (`journalctl -k`, read by the lead).** At 19:59:44 UTC the card had a fifth reset this boot (`ring gfx_0.0.0 timeout`, MODE1 reset, `VRAM is lost due to GPU reset!`). The guilty process was a desktop app (an Electron-based code editor's GPU thread). No study job was running, and the profile was "Default" (+150 MHz, -50 mV). GPU compute jobs of another project had run on the card until about 3 minutes before (their last log line at 19:56:56 UTC, all 20 jobs ended with exit 0); none was logged as running at the reset. So an ordinary desktop app also hung the card on this profile, and the overclock and undervolt are as likely a cause as PyTorch's FP8 path. MEASURED for the reset, INFERRED for the cause; untested. A stock-profile test is planned. The four hangs below are the PyTorch ones of 2026-11-18.

**Update, 2026-11-20 (`round5.md` item K).** The exact command of hang 2 (`r4t-tg-3b-fp8-a`, same venv) ran clean three times on the stock profile "Testing" (exit 0 each, the kernel log's reset count stayed at 5), against 2 hangs in 2 runs of the 4-stream decode on "Default" (hang 2, and hang 3 on another library stack). That implicates the overclock/undervolt profile (INFERRED; small numbers). It does not test the NaNs or the unequal results (no perplexity ran) or the prompt path (`prefill` came back empty), and nothing ran on ROCm 10.2's PyTorch FP8.

Four GPU hangs, each ended by a full GPU reset. Kernel log (`journalctl -k`, saved by the lead as `runs/llm4-meta/kernel-log-gpu-resets.txt`; the log is in the machine's local time, UTC minus 8 h): `ring gfx_0.0.0 timeout`, `Ring gfx_0.0.0 reset failed`, `MODE1 reset`, `GPU reset succeeded`, `VRAM is lost due to GPU reset!`; for hang 1 also `Queues reset on process python`. The graphics ring is the desktop's too (INFERRED); a few seconds of disturbance were expected each time and are not logged. The runner reported each as `THERMAL STOP hotspot 999 C`: **999 is the runner's value for an unreadable sensor, not a temperature**; the card could not be read while it was being reset.

| # | reset in the kernel log (UTC) | job | what it was doing | evidence |
|---|---|---|---|---|
| 1 | 20:51:51 | `r4t-pp-3b-fp8` (first run) | torch 2.15 + ROCm 10.0, prefill 128 and 512 printed (6,551 / 14,450 tok/s), hung in the 2048 prefill | `uclk1408/r4t-pp-3b-fp8.log` (GPU Hang, exit 134) |
| 2 | 21:04:02 | `r4t-tg-3b-fp8-a` | same venv, decode with 1, 2, 3 streams printed, hung at the 4th | `r4t-tg-3b-fp8-a.log`, `.meta` (`hotspot_max=999`) |
| 3 | 21:10:50 | `r4t101-3b-fp8` | same harness on torch 2.13 + ROCm 10.1, decode, hung at 4 streams | `r4t101-3b-fp8.log` |
| 4 | 21:44:29 | `r4b-t-prefill-3b-fp8` | 20 s prompt loop under the 20 ms sampler | `stale2/r4b-t-prefill-3b-fp8/` (`samples.csv`, `summary.txt`) |

Hang 4 in the sampler record: the loop starts at about 4.2 s (the busy share is 100 % from 4.20 s); from 4.4 to 6.2 s the card works normally (core 3.06 to 3.22 GHz, 225 to 310 W, hotspot 73 to 82 C, 982 to 1,048 mV; the run's maximum, 82 C, at 6.1 s; two samples carry the hotspot throttle bit). At 6.36 s the power falls from about 300 W to 118 W and then stays at 99 to 111 W (6.5 to 10.5 s) with the card still 100 % busy, while the core clock goes to 3.37 GHz and then 3.42 to 3.59 GHz (1,124 to 1,205 mV) and the hotspot falls to 75 C at 6.36 s and 56 to 62 C from 6.5 s; some samples cannot be read (-1). That last phase lasted until the reset, 4.6 s.

- **The debugging limit was breached (stated plainly).** The rule is: stop at once when the same error comes back after a fix, after two different fixes fail, or after about 10 minutes. Hang 2 came back with nothing changed, hang 3 came back after the change of library stack (a fix that failed), and 18 minutes had passed since hang 1: the limit was reached at hang 3 (21:10Z) at the latest. Instead, PyTorch FP8 GPU jobs were started again and again: counted from the `# start` lines of the logs, 20 after hang 2 and **19 after hang 3** (18 `llm4_torch.py` jobs and `r4t-detgemm`): four trace jobs (21:39 to 21:43Z), the 20 s loop that caused hang 4 (21:44Z, 34 minutes after hang 3 with three hangs behind it), and from 22:39Z the perplexity, KL and NaN-hunting reruns (`r4q-t-*`, 13 jobs) and `r4t-detgemm` (23:09Z). The earlier write-up said the limit was reached "after the second different change" (the sync switch ran at 22:59Z, 1 h 49 min after hang 3); that was wrong. The NaN problem (first seen at 22:39Z) can be called a separate problem, which covers the NaN reruns (`-r2` to `-r5`, `-sync-r1` to `-sync-r6`) and `detgemm`, but not the trace jobs, the loop that hung, or the 3B/8B perplexity and KL runs of the hang-prone path. PyTorch FP8 is now parked: the owner forbids any GPU job on PyTorch's FP8 path.
- What was tried and not seen (INFERRED where it is a negative; every one of these is a short probe, not a proof): not seen when repeating one FP8 GEMM (`r4t-detgemm`: 2 shapes of 3 planned, N = 512, 2 s per row, 7,653 and 8,568 repetitions, bit-identical, no NaN; stopped at 96 C, 3 of 6 rows done; the hangs are at N = 16 padded and N = 2048); not seen in the separate GEMM job (`r4t-gemm-*`: N = 16 to 2048, 50 calls each, no hang; outputs not compared); not seen in one SDPA probe (5 lengths, default backend only, `attn-probe`); not seen in bf16 (the same harness: 0 hangs, 5 bit-identical perplexity runs of 9.72225615). The ROCm 10.0 / 10.1 split does not separate the cases: hang 3 is eager PyTorch on 10.1.
- The FP8 harness also gives NaN in 3 of 12 4-chunk perplexity runs and unequal results over nine finite runs. All twelve values, in run order (`ppl-smoke-fp8.log`: 9.7347 at memory 1408 MHz, before the first hang, the only FP8 perplexity at that clock; then at 1258 MHz `r4q-t-ppl4-3b-fp8` NaN, `-r2` NaN, `-r3` 9.7087, `-r4` 9.7303, `-r5` 9.7279; then with `LLM4_SYNC_AFTER_MM=1`, a debugging switch that waits for the GPU after every `_scaled_mm`: `-sync-r1` NaN, `-sync-r2` 9.7121, `-r3` 9.7216, `-r4` 9.7267, `-r5` 9.6911, `-r6` 9.6987). The sync changed nothing (1 NaN in 6).
  vLLM FP8 (same kernel family, ROCm 10.1): four runs, 9.725648, 9.725648, 9.725648, 9.725648 (differences below 1e-7).
- Hang 4's state before the stall was normal (3.1 GHz, 225 to 310 W, 73 to 82 C). The 3.58 GHz and 108 W belong to the stalled state (a stuck busy card draws little and the clock governor raises the frequency, INFERRED). The earlier hypothesis that the card is unstable "at the 3.5 GHz it reaches when the load is light" is **not supported**: that clock was reached only after the stall began. The cause stays untested: overclock and undervolt against software cannot be told apart here (no job ran on the Testing profile), and nothing about the PyTorch FP8 harness was changed to hide this; the affected rows are marked. A reset also sets the memory clock to 1258 MHz (it was 1408 before; the gate does not check the memory maximum, so it went unnoticed by the gate and was caught by the label).
- Not tested, INFERRED reading of the reviewer: the many `__amd_rocclr_fillBufferUnAligned` kernels that come with `_scaled_mm` have timestamps that overlap the GEMMs; a race between the fills and the GEMM would fit "unequal but never systematically wrong". The sync switch (after the GEMM, not after the fill) did not remove the NaN, which argues against it.

## R5: quality (same text, 512-token chunks, scored on positions 256 to 510 with BOS first, as llama-perplexity does)

Perplexity, 4 chunks (1,020 scored tokens; memory 1258 MHz; the table covers `r4q-*ppl4*`):

| model | llama.cpp unmod / patched | llama.cpp Q8_0 | llama.cpp 16-bit true | F32 true | PyTorch FP8 | PyTorch bf16 | vLLM FP8 | vLLM bf16 |
|---|---|---|---|---|---|---|---|---|
| 3B | 9.7113 / 9.7104 | 9.7140 | 9.7112 | 9.7084 | 3 NaN and 9 finite runs, 9.6911 to 9.7347 (all twelve values and files: "GPU hangs") | 9.7223 | 9.7256 | 9.7140 |
| 8B | 7.4451 / 7.4439 | 7.4489 | - | - | 7.4752 (one run, PyTorch FP8 not reproducible) | - | 7.4566 | - |

The text tokenises to 132,871 tokens with the HF tokenizer, the count the PyTorch runs print (`n_tokens_text` in `r4q-t-ppl4-3b-fp8.log`; llama-perplexity prints no token count); the token ids themselves were not compared (no token dump was kept), so "torch and vLLM score the same tokens" rests on the same count and the same chunking rule.
Distance from the true weights (Meta's 3B as F32 run by llama.cpp, saved logits, **8 chunks** = 2,040 positions; a first reference run with 25 chunks was stopped at 95 C after 12 chunks (8.0 s, `stale/r4q-base-3b-true-f32`), because F32 GEMMs run hot, and the run was redone with 8; reference perplexity 10.4977 is the KL tool's PPL of the base over the 8 chunks (the saved logits are 16-bit compressed; the F32 run's own running estimate after chunk 8 prints 10.5011); its file sha256 `b7d371fb2ff999f9e5ed8569441011578b65875aa33e816fb4f1cce6cbf73466`, 523,317,204 bytes, deleted after use):

| route | perplexity | mean KL divergence | 99 % KL | same top token % |
|---|---|---|---|---|
| llama.cpp 16-bit (bf16 GGUF) | 10.5017 | 0.000020 | 0.000162 | 99.56 |
| llama.cpp FP8 unmodified / patched | 10.5110 / 10.5116 | 0.003687 / 0.003687 | 0.0294 / 0.0285 | 96.47 / 96.37 |
| llama.cpp Q8_0 | 10.5146 | 0.004277 | 0.0306 | 96.08 |
| PyTorch bf16 | 10.5094 | 0.000357 | - | 98.73 |
| PyTorch FP8 (one run, not reproducible) | 10.6849 | 0.021864 | - | 93.82 |
| vLLM bf16 / FP8 | 10.5047 / 10.7413 | no distributions | - | mean abs log-prob difference of the target token 0.0155 / 0.1133 |

- The 16-bit file's distance is the floor of each runtime's arithmetic (llama.cpp bf16 2e-5, PyTorch bf16 3.6e-4: PyTorch's bf16 route is 18 times noisier). FP8 weights from Red Hat's checkpoint are 184 times (FP8) and 214 times (Q8_0) farther (3.7e-3 and 4.3e-3); Q8_0 (made from that checkpoint) is 1.16 times the FP8 figure. So the error is in the FP8 checkpoint's weights, not in the route; round 2's 3e-6 measured the route against the exact FP8 weights.
- Quantising the activations to FP8 with static scales (PyTorch, vLLM) costs about 6 times more than llama.cpp's FP8 weights with FP16 activations (2.2e-2 against 3.7e-3 for PyTorch: about 6 times in the one PyTorch run, a path that is not reproducible; vLLM's target-token log-probability error is 7 times its own bf16's, 0.113 against 0.015, not a comparison with llama.cpp's FP8) and costs about +0.2 perplexity on the 8 chunks (10.68 and 10.74 against 10.50 true; both runtimes support the +0.2).
- The 8B has no 16-bit original on disk (about 16 GB, from two bytes per parameter of an 8-billion-parameter model: an external estimate, not measured here; it would not fit the card), so it has no true-weights rows in this file (see `llm-8b-exact.md` for the exact-weights comparison made later); its perplexities are comparable between runtimes only (7.4439 to 7.4752).
  > **Updated (`llm-8b-exact.md`):** the 8B now has an exact reference: the FP8 checkpoint's own weights converted exactly (every FP8 value times its scale, stored without rounding; 8 chunks of the same text), **not Meta's 16-bit 8B** (not on disk; the reference run itself ran 22 of 32 layers on the CPU in F32). So the 8B figures below measure the distance from the exact FP8 weights, a different question from the 3B rows above (3.7e-3 and 4.3e-3), which measure the distance from Meta's 16-bit weights; how far the 8B FP8 checkpoint is from Meta's 16-bit 8B is not measured. Against it, llama.cpp's `llama31-8b-fp8-outq8` file has a mean KL divergence of 8.1e-5 (unmodified build) and 8.2e-5 (patched), the plain FP8 file `llama31-8b-auto` 5e-6, and the Q8_0 file 5.2e-4; the `outq8` file's Q8_0 output matrix accounts for the 8e-5.
- Chunking: llama-perplexity reads the text, tokenises with BOS, takes consecutive 512-token chunks, replaces the first token of each by BOS, evaluates all 512 in one pass and scores positions 256 to 510 against the following token (255 per chunk); `llm4_torch.py` and `llm4_vllm.py` do the same.
  The KL divergence counts only reference entries above -16 in log-probability, as llama-perplexity does (llama.cpp source, READ: `tools/perplexity/perplexity.cpp`, `p_log_base > -16.f`; an earlier version of the PyTorch harness without that cut is not logged).

## Not measured, parked and open

- The cause of the PyTorch FP8 hangs, NaN and non-reproducibility (above); PyTorch FP8 prompts of 2048 tokens, decode at 4 or more streams, the 8B prompt and decode rows, a PyTorch FP8 20 s behaviour loop (all hung or would risk another reset).
- The kernel traces' overlapping timestamps (R3 caveat), and llama.cpp's kernel breakdown on this profile (the round-1 pipeline fails its own checks here).
- vLLM's 23 % slower decode at 128 generated tokens than at 32; vLLM attention kernels' instructions; Triton fused kernels' ISA.
- KL divergence over 50 chunks (8 were used; no 50-chunk run was made: the 25-chunk try stopped at 95 C); the 8B against any reference (done later, `llm-8b-exact.md`); vLLM KL divergence (no full distributions); the FP8 peak measured at the logged clock (it is the READ 420 scaled).
- The overclock's role in any of this: no job ran on the Testing profile in this round.

## Install and download records (undo with the commands given)

- Downloads (Hugging Face, no token), in `models/`: `llama32-3b-redhat-fp8/` (`RedHatAI/Llama-3.2-3B-Instruct-FP8`, 4.40 GB) and `llama32-3b-instruct-16bit/` (`unsloth/Llama-3.2-3B-Instruct`, 6.44 GB). Both repositories exist under those names. File lists, sizes and sha256: `runs/llm4-meta/downloads.txt`.
  The Unsloth copy has no LICENSE file; its `README.md` names the Llama 3.2 Community License (`license: llama3.2`) and is kept. Converted here with `convert_hf_to_gguf.py`: `llama32-3b-true-f32.gguf` (12.9 GB, sha256 in `downloads.txt`) and `llama32-3b-true-bf16.gguf` (6.4 GB). Undo: delete those four folders and files.
- PyTorch venv: `transformers==5.23.0` plus its dependencies (27 packages added, none removed or changed, `toolchains/llm4-torch-install-diff.txt`: tokenizers 0.23.3, safetensors 0.8.0, huggingface_hub 2.4.0 and 23 small ones; accelerate not installed) with torch 2.15.0+rocm10.0 and numpy 2.5.5 pinned by a constraints file. `pip freeze` before and after: `toolchains/torch-rocm10-freeze-before-llm4.txt`, `-after-llm4.txt`; the added lines: `toolchains/llm4-torch-install-diff.txt`. These records are in the git-ignored `toolchains/` folder, so they are not in the commit (like `runs/`). Undo: `pip uninstall` the added names.
- vLLM venv `toolchains/vllm-rocm101-venv` (Python 3.14, AMD's route): torch 2.13.0+rocm10.1.0 and torchvision 0.28.0+rocm10.1.0 from `stable.repo.amd.com/rocm/whl-next/`; flash-attn 2.8.3, amd-aiter 0.1.22.post2 and vLLM 0.29.1.dev0+rocm10.1.0 from `rocm.frameworks.amd.com/whl-multi-arch/vllm/`; tensorizer 2.12.1. Freezes: `toolchains/vllm-venv-freeze-before.txt`, `-after.txt`. Size: 19 GB (`du`; the uv cache `~/.cache/uv` is 20 GB, of which the same files may be hard links: not checked; no before-measurement of the free space was kept). vLLM's own caches `~/.cache/vllm` (0.2 GB) and `~/.triton`. Undo: delete the venv, `~/.cache/vllm`, and either the whole uv cache `~/.cache/uv` (it is shared with any other uv use and was created on 2026-11-17, before this install; it can be rebuilt, but it is not only this install's) or, next time, install with a project cache folder instead.
- vLLM route: AMD's documented pip route for ROCm 10.1 (AMD's page lists the RX 9070 XT, gfx1201); no container, no sudo, no system change. It needs `PYTHONPATH` set to the venv's `_rocm_sdk_core/share/amd_smi` (the platform check needs `amdsmi`) and was run in one process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the profiler sees its kernels. The AMD page said nothing about FP8 on RDNA (and vLLM's own page lists the RX 9000 series as supported).
- Profiler: `rocprofv3` crashes when Triton's LLVM is imported next to its own ("registered more than once"); the PyTorch harness sets `LLM4_NO_TRITON=1` when traced (the eager HF path needs no Triton).
- Disk (this round): the four new model files and folders add 30.1 GB (28.1 GiB: Red Hat FP8 folder 4.40, Unsloth folder 6.44, F32 GGUF 12.86, bf16 GGUF 6.43 GB); the free disk space was not recorded then; the run folder 0.4 GB. The free space before the round was not recorded.

## How to rerun

```
scripts/study-llm.sh selftest                        # no GPU; the Default-profile gate cases and the thermal guard cases are in it (selftest-guard: only the guard)
LLM_RUN=runs/<folder> scripts/study-llm.sh r4setup    # versions, models, origin check
scripts/study-llm.sh r4known | r4torch | r4vllm | r4q | r4trace | r4beh | r4llamatrace   # each gated; modes starting r4 expect the profile Default
R4_KL_CHUNKS=8 R4_KL_ONLY=1 R4_SKIP_TORCH_FP8=1 scripts/study-llm.sh r4q   # the quality run as it was made here
python3 -I scripts/llm4_tables.py tables runs/<folder> --uclk 1258
```
The PyTorch FP8 jobs can hang the GPU on the profile Default: leave them for last (`R4_SKIP_TORCH_FP8=1`).

## Proposed lines for `notes/STUDY-PLAN.md` and `README.md` (not applied)

**STUDY-PLAN.md, status paragraph:**

> LLM data dive, round 4 done (2026-11-18, branch `study/llm-dive`, `measurements/llm-runtimes.md`, expectations in `llm-runtimes-expectations.md` written before the GPU jobs): PyTorch (`torch._scaled_mm`, eager) and vLLM 0.29 (AMD's ROCm 10.1 wheels) on Llama 3.2 3B and 3.1 8B next to llama.cpp, on the profile "Default" (overclock, 374 W): FP8 WMMA confirmed in the hipBLASLt kernels both use (vLLM's decode uses `v_dot4_f32_fp8_fp8`); quality against Meta's 16-bit 3B run in F32. Run `runs/20261118T202715Z-llm4`. PyTorch FP8 hung the GPU four times (full resets, memory clock 1408 to 1258 MHz) and is not reproducible; parked.

**README.md, key-finding bullet:**

> - **The FP8 matrix instruction is used by the big GEMMs of PyTorch and vLLM, not by their one-token decode.** `torch._scaled_mm` and vLLM's prompt path run hipBLASLt kernels with `v_wmma_f32_16x16x16_fp8_fp8` (up to 236 to 259 TFLOPS, about 60 % of the 420 peak); vLLM's decode kernel uses the FP8 dot product `v_dot4_f32_fp8_fp8` on the vector unit. vLLM FP8 prompts run at 18,956 tokens/s for the 3B (llama.cpp Q8_0 12,651, FP8 9,280). Against the true weights FP8 and Q8_0 files are equally far (KL divergence 3.7e-3 and 4.3e-3, 16-bit 2e-5). PyTorch's FP8 path hung the overclocked card four times and is not bit-reproducible.
>   ([llm-runtimes.md](llm-runtimes.md))

**README.md, measurement-file rows:**

> | `llm-runtimes.md` | LLM dive, round 4: PyTorch and vLLM with real FP8 matrix maths next to llama.cpp on the overclock profile: kernels and instructions, speed, clocks and power, quality against the true weights, the PyTorch FP8 hangs |
> | `llm-runtimes-expectations.md` | The expectations for round 4, written before any GPU job |
