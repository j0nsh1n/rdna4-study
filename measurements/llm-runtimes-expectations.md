# LLM data dive, round 4: expectations (written 2026-11-18 before any GPU job of this round; never edited afterwards)

Round 4 runs the same models through PyTorch (real FP8 matrix maths, `torch._scaled_mm`) and vLLM, next to llama.cpp, on the LACT
profile "Default" (the owner's overclock: 374 W cap, core offset +150 MHz, -50 mV, memory maximum 1409 MHz, the 1.2x regulator mod in the
card). Every wall-clock number of this round is "overclocked profile, power limit raised" and is not compared with round 1 to 3's
stock-profile ("Testing") numbers except where the file says so. Models: Llama 3.2 3B Instruct and Llama 3.1 8B Instruct as Red Hat FP8
checkpoints (static per-tensor activation scales in the checkpoint, per-tensor weight scales), and Meta's 16-bit 3B (Unsloth's public copy).
The 8B has no 16-bit original on disk (and its 16 GB would not fit the 16 GB card), so the 8B has no 16-bit rows and no true-weights quality rows.
Percentages in brackets are my confidence. "Fails if" says what would make the expectation wrong.

## Before the runs (no GPU)

- **O1 Origin.** The Red Hat 3B FP8 checkpoint is the one the existing 3B FP8 GGUF was made from: all 196 FP8 weight tensors bit-identical (after the
  converter's q/k row reorder), all 392 scales and the norms and embedding equal; only `output.weight` differs (the GGUF has it as Q8_0, the checkpoint
  holds `lm_head` in BF16) (90 %). Fails if any FP8 byte differs.
- **O2 Support for vLLM.** AMD's page lists gfx1201 and gives a pip route (vLLM 0.29 wheel, ROCm 10.1, torch 2.13.0+rocm10.1.0, Python 3.14). Whether it
  runs this FP8 checkpoint on gfx1201 is open: works through `torch._scaled_mm` or a Triton/AITER FP8 kernel (65 %); starts but falls back to widening FP8
  weights to 16-bit (15 %); does not start on this card or fails under the 10 GB host-memory cap (20 %).

## R1: does the FP8 path use the FP8 matrix instructions?

- PyTorch: each FP8 linear layer calls `torch._scaled_mm`, which goes to hipBLASLt; the kernel trace shows hipBLASLt (Tensile) GEMM kernels whose names carry
  an FP8 type, and their machine code contains `v_wmma_f32_16x16x16_fp8_fp8` (90 %). Fails if the GEMM kernels are 16-bit ones or the code object holds no FP8 WMMA.
- vLLM: its FP8 GEMMs are also hipBLASLt through `torch._scaled_mm` (50 %), a Triton or AITER FP8 kernel with FP8 WMMA (20 %), a non-FP8 fallback (15 %), or
  vLLM does not run (15 %). Whatever it is, I expect the trace names a library and a kernel and the ISA shows FP8 WMMA unless it is the fallback.
- The 16-bit paths (PyTorch bf16, vLLM bf16) use `v_wmma_f32_16x16x16_bf16` and no FP8 instruction (95 %).

## R2: speed (overclocked profile, power limit raised)

- PyTorch (HuggingFace eager, batch 1 per stream): prompt of 512 tokens on the 3B 8,000 to 16,000 tokens/s for FP8, best guess 11,000; FP8 is 1.0 to 1.3 times the
  bf16 speed, not 2 times, because the activation quantize, the casts and the eager elementwise kernels cost time next to the GEMMs (70 %). Fails if FP8
  is below bf16 or above 1.5 times.
- PyTorch generation, one stream, 3B: 20 to 50 tokens/s, limited by launch and Python overhead, not by memory (80 %); FP8 is not faster than bf16 there (ratio 0.8 to 1.2, 70 %);
  stream scaling from 1 to 8 streams close to linear (6 to 8 times) because the step cost does not depend on the batch (75 %).
- vLLM (graphs, fused kernels), 3B FP8, one stream: 110 to 200 tokens/s, near the memory limit (the 3B's 4 GB at 600 to 640 GB/s is about 150 to 160), and 8 streams 500 to 1000 (60 %, conditional on it running). Prompt of 512: 8,000 to 20,000 tokens/s.
- llama.cpp on this profile against round 1: prompt speed +2 to +10 % (the power cap is higher, the core clock higher), generation within +-3 % (same memory clock 1408; memory-bound) (75 %).
- FP8 GEMM rate in PyTorch at the models' shapes (M = weight rows, N = tokens 128 to 2048, K = inputs): 60 to 200 TFLOPS; best below 50 % of the FP8 peak scaled to the logged core clock (80 %). The first hour's 166 TFLOPS for 4096 x 4096 x 512 reproduces within 15 % (65 %).
  Fails if any real shape exceeds 60 % of the scaled peak.

## R3: where the time goes

- PyTorch prefill 512, 3B FP8: FP8 GEMMs 40 to 65 % of the kernel time; activation quantize plus other elementwise kernels 15 to 30 %; attention under 15 % (65 %).
- PyTorch decode, 3B, one stream: the GPU is idle for more than half of the token time between kernels (launch gaps over 50 %), with 28 layers times 20 or more kernels (80 %).
- vLLM decode: gaps under 15 % of a token with graphs; the weight GEMMs read at 70 to 95 % of the bus rate when running (55 %, conditional).
- The PyTorch and vLLM FP8 prompt paths have no separate widen pass (the weights stay FP8), unlike llama.cpp's FP8 path (90 %).

## R4: clocks, power, hotspot

- PyTorch and vLLM prompt loops sit at the 374 W reported cap (PPT0 flag in at least half of the samples) with the core at 3.0 to 3.4 GHz median while awake (60 %); hotspot reaches 85 to 95 C in 20 s loops
  and the 95 C stop may fire once (35 %). Eager decode draws much less (under 200 W reported) because the card idles between launches (75 %). llama.cpp prompts on this profile 330 to 374 W with a clock about 0.1 to 0.3 GHz above round 1's.
- Memory clock 1408 MHz in every job (the displays hold it) (85 %); a different value is recorded and labelled.

## R5: quality (same text, 512-token chunks, scored on the second half of each chunk, first token set to BOS, as llama-perplexity does)

- Torch perplexity of Meta's 16-bit 3B (the true reference) on the 4-chunk set: 9.3 to 9.9; the FP8 checkpoint within 0.15 of it (70 %); llama.cpp's FP8 files within 0.02 of torch FP8's value (60 %: torch quantizes the activations to FP8 too, llama.cpp does not).
- Against the TRUE weights (F32 llama.cpp reference from Meta's file), the FP8 weights are far from exact: mean KL divergence 1e-3 to 1e-2 for llama.cpp's FP8 routes, not round 2's 3e-6 (that was distance from the exact FP8 weights) (75 %).
  Q8_0 (made from the FP8 checkpoint) is about as far as FP8, within a factor of 1.5, because the error comes from the FP8 checkpoint, not from the route (65 %). So round 2's "Q8_0 is 230 times worse" turns round when the ruler is the true model.
  Fails if Q8_0 is more than 3 times the FP8 value or less than a third of it.
- Torch FP8 (activations also in FP8, static scales) is 1.2 to 3 times as far from true as llama.cpp's FP8 (60 %). Torch bf16 of the same Meta weights against the F32 reference: mean KL divergence below 5e-4 (85 %), and the same top token in 99.5 % or more.
- vLLM: only per-token log-probabilities of the actual next token are available (no full distributions), so its quality is perplexity and the mean absolute difference of the target-token log-probability to the reference; its FP8 perplexity within 0.05 of torch FP8's (50 %, conditional).

## How the expectations could fail through the setup, not the card

- vLLM may not fit under the 10 GB host-memory cap (the cap stays); then the vLLM rows are missing and the reason is reported.
- The HuggingFace eager path's decode speed depends on the library version and Python overhead; the number is for that stack only.
- The core clock under the overclock profile may sit lower than 3.3 GHz when the power cap acts; the FP8 peak is scaled by the logged clock and the scaled figure is an estimate (READ peak times a clock ratio), not a measured peak.
