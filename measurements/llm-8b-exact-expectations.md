# The exact 8B comparison: expectations (written before any GPU job of this step, never edited after its commit)

Written 2026-11-18 on the branch `study/llm-dive`, before the reference file is built and before any GPU job of this step. Setting: the LACT profile "Default" (374 W cap, +150 MHz, -50 mV), the gated runner with the new 20 ms thermal guard (stop at 95 C), memory cap 16 GB for the one reference job (owner-approved, MemoryMax=16G, no swap) and 10 GB for every other job.

**Question.** How far are llama.cpp's 8B FP8 route (unmodified and patched builds, the `llama31-8b-fp8-outq8` file) and the 8B Q8_0 file from the *exact* FP8 weights (every FP8 value times its scale, stored without rounding: F16 where that is exact, else F32)? Round 2 could only compare against an F16 file for the 8B (FP8 5e-6, Q8_0 6.1e-4 mean KL divergence; the F16 file was not bit-checked and its own error is unknown).

**Method.** `scripts/llm3_mixed_ref.py build` (27.864 GB planned in round 3: 31 tensors F16, 193 tensors F32, the rest as stored) from `llama31-8b-auto.gguf`, `compare` with 16 sampled rows per FP8 tensor (rows 0 and last always) and every other tensor byte for byte, sha256 logged. One reference job: `llama-perplexity` with saved logits, 8 chunks of 512 tokens of the same text as the 3B true-reference rows of round 4 (`ppl-text.txt`), partial offload with the layer count chosen from the card's free memory with at least 2.5 GB margin. Then `--kl-divergence` of the three files at the normal cap. The reference GGUF and the logits are deleted at the end after size and sha256 are logged.

| # | Expectation | Confidence |
|---|---|---|
| E1 | The built file is 27.86 GB within 0.5 % (27.7 to 28.0 GB), with 31 F16 and 193 F32 FP8 tensors, and the bit check passes on every sampled row (no tensor differs). | 90 % |
| E2 | The build runs under the 10 GB host cap without being killed and takes under 30 minutes. | 80 % |
| E3 | the web browser's main process ends within 30 s of SIGTERM, and available memory rises by at least 1 GB. | 70 % |
| E4 | The reference job fits the card with 12 to 17 of the 32 layers (plus the output layer) on the GPU, finishes the 8 chunks within 50 minutes (about 10 minutes most likely) and is not killed by the 16 GB cap. | 65 % |
| E5 | The reference job does not reach the 95 C stop (peak under 92 C), because part of the work runs on the CPU. If it stops on heat twice, the job is parked. | 70 % |
| E6 | The reference's own perplexity over the 8 chunks is within 0.01 of the unmodified FP8 file's (the same weights; difference only the rounding of the activations path and the Q8_0 output matrix of the `outq8` file). | 85 % |
| E7 | Mean KL divergence of the unmodified FP8 file against the exact reference: 1e-6 to 3e-5 (round 2: 3e-6 on the 3B and 5e-6 on the 8B against F16 references). Most likely 5e-6 to 2e-5, slightly more than round 2 because the `outq8` file holds its output matrix in Q8_0 and the reference in its original 16-bit form. | 75 % |
| E8 | The patched build is within a factor of 2 of the unmodified one (round 2: equal at 3e-6). | 85 % |
| E9 | Q8_0: mean KL divergence 4e-4 to 8e-4 (round 2: 6.1e-4 against the F16 reference), at least 20 times the FP8 figure. | 80 % |
| E10 | "Same top token": FP8 files 99.7 % or more, Q8_0 98.5 % or more (a rounded-weights file differs more from the reference than the FP8 route does). | 65 % |
| E11 | Peak host memory of the reference job stays under 16 GB (cgroup), without swap. | 80 % |
| E12 | The memory clock reads 1258 or 1408 MHz in every job and is logged per job; it does not change the KL numbers (KL does not depend on speed), only the times. | 90 % |

Not predicted, to be measured: the layer count that fits, the reference job's time, the peak hotspot and host memory, the number of tokens per second.
