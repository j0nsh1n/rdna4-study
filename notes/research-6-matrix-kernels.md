# RDNA 4 research, part 6: AMD's matrix guides and existing kernel designs

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 deep research: AMD matrix guides
and kernel designs"); items marked CHECKED were re-read by the lead. READ = stated in the source, seen by the helper
only. INFERRED = reasoning.

Sources:
- [W1-3] AMD GPUOpen, "WMMA guide for AMD RDNA 4 architecture GPUs", parts 1 to 3, Hui Zhang, 2026-06-02,
  <https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/> (and `part-2`, `part-3`).
- [MC4] AMD GPUOpen, "Using the Matrix Cores of AMD RDNA 4", 2025-07-11,
  <https://gpuopen.com/learn/using_matrix_core_amd_rdna4/>
- [PR] llama.cpp pull requests, <https://github.com/ggml-org/llama.cpp/pull/NNNNN>.
- [MESA] Mesa main, `src/amd/compiler/` and `src/amd/vulkan/nir/radv_nir_lower_cooperative_matrix.c`.
- [CALC] AMD matrix instruction calculator (part 1).

## Operand layout in practice

- CHECKED [W1]: "Both matrix A and B are K-major, with each thread holding 8 contiguous elements." "Matrix D,
  however, is M-major." One layout for all data types.
- READ [W1], from the sample code: `lane = lIdx % 16; laneGroup = lIdx / 16`; a lane loads its 8 elements from
  `base + lane * row_stride + laneGroup * 8`. So lanes 0 to 15 take K 0 to 7 and lanes 16 to 31 take K 8 to 15, each
  with one contiguous load. No replication across the two halves as on RDNA 3 ([MC4]: "we needed to duplicate some
  elements for A and B, but it is removed for RDNA 4").
- READ [MESA], header comment of the cooperative-matrix lowering: the guide's A and B layouts "are identical to the C
  layout with the exception of the order of the rows (columns for A). And as long as these are swapped in the same
  way for both A and B, the muladd result will be the same. So we use the C layout for all uses." Mesa uses one
  layout for everything: 8 elements per lane for every type.
- READ [W1]: because the result layout is the transpose of the A layout, calling the instruction with A and B swapped
  gives a result that can feed the next multiply as A without moving data between lanes (only works for 16x16).
  "This technique has been deployed in Llama.cpp to implement Flash Attention on RDNA 4."

## Load widths and the "double-K" trick

- CHECKED [W2]: "FP16 saturates the 128-bit memory interface (8 x 16 bits), while FP8 and INT8 utilize only 64-bit
  loads (8 x 8 bits)". Loads and stores work at up to 128 bits (16 bytes) per lane.
- CHECKED [W2]: fuse two matrix instructions into one step with K = 32. Each lane loads 16 bytes; bytes 0 to 7 feed
  the first instruction, bytes 8 to 15 the second; both accumulate into the same C. Data layout:
  `row * K_stride + k * 32 + lane_group * 16`.
- READ [W2]: the sample is for 8-bit integers, checked against hipBLAS with exact comparison on small random
  integers. The article says the result "remains bit-identical ... due to the associativity of matrix
  multiplication". INFERRED: that argument holds for integers. For FP8 into FP32 the order of additions can change
  the last bit; not shown either way.
- READ [W1]: the FP16 fused example is checked with a tolerance of 0.001, not bit-exact.
- READ: the samples run one wave, load straight from global memory, keep everything in registers, and use no
  shared memory. No timings, no hazard discussion.

## Transposing

- READ [W3]: AMD's article says RDNA 4 lacks "shared-memory transpose loading and in-register matrix transpose", and
  builds a transpose by multiplying with an identity matrix (exact, F16 output).
- READ [MESA]: Mesa does use the transposing loads `global_load_tr_b128` (16-bit elements) and `global_load_tr_b64`
  (8-bit elements) on gfx12, for row-major matrices loaded from buffers, never from shared memory. So a transposing
  load from global memory exists and is used; only the shared-memory and register cases are missing. The clang
  builtin exists too (part 1).
- READ [MESA]: its own software transpose is several lane shuffles ("Transpose is a mess...").

## Rounding traps

- READ [MC4]: the sample converts activations with `__builtin_amdgcn_cvt_pkrtz`, which rounds toward zero. An exact
  kernel must choose round-to-nearest on purpose.
- READ [PR 30885, merged 2026-10-23]: the BF16 matrix builtins take vectors of `short`, not `__bf16`; builds broke
  with a newer LLVM until this was fixed.
- READ [CALC]: the FP8 matrix instructions accept no negate modifier; 16-bit float ones do.

## What llama.cpp already does on RDNA 4

| PR | What | Date |
|---|---|---|
| 12372 | gfx1200/gfx1201 targets | 2025-03-26 |
| 14202 | disable rocWMMA on gfx12 | 2025-06-16 |
| 14323 | vector flash attention on RDNA 4 | 2025-06-22 |
| 17077 | "RDNA4 tensor core support for MMF" (F16 matrix kernel) | 2025-11-21, CHECKED |
| 17156 | "WMMA-MMQ kernels for RDNA 4" (8-bit integer matrix kernel for quantized types) | 2025-11-24, CHECKED |
| 18537, 26199, 26284 | tuning of when to use these kernels or rocBLAS on RDNA 3 and 4 | 2026 |
| 28102 | flash attention tuning for gfx1201 | 2026-09-11 |
| 25940 (open), 29536 (closed) | work around register spills in the RDNA 4 kernels | 2026 |

- READ [PR 17156]: Q8_0 and Q4_0 weights become 8-bit integers and activations are quantized to 8 bits (Q8_1), then
  the integer matrix instruction is used: two 8-byte loads per lane from shared memory, two instructions per 16x8
  tile. Benchmark by the llama.cpp maintainer on an RX 9060 XT, 8B model, prompt of 2048 tokens: Q8_0 at batch 512
  went from 345 to 2110 tokens/s; batches of 1 and 8 did not change (32.7 and 181.4 before and after).
- INFERRED: llama.cpp's own data says the matrix instructions do nothing for decode-size batches; decode stays a
  memory problem.
- CHECKED [PR 17077]: the F16 kernel was slower than hipBLAS at batches 2 to 16 at first. The maintainer: "On RDNA
  the WMMA instructions do to my knowledge not increase peak FLOPS, they only reduce I/O and register usage." And on
  shared memory: loading 4-byte chunks per thread hits only half the banks, "you only get 50% of the memory
  bandwidth". The author: "hip compiler doesn't acquire register very well".
- READ [PR 29536, 25940]: the ROCm compiler unrolls loops in these kernels until registers spill to memory ("5 of 64
  iq4_xs kernels spilled 2387 VGPRs"; prompt speed 298 against 1250 tokens/s once fixed). Turning off unrolling lets
  it reuse registers. The same class of compiler problem met on RDNA 3.
- No FP8 matrix path exists in llama.cpp for AMD (CHECKED locally, part 1).

## Hazards as the compilers see them

- READ [MESA `aco_insert_NOPs.cpp`]: a `v_nop` is inserted when a matrix instruction reads, as A or B (or the sparse
  index), a register written by the matrix instruction just before it. Chaining the accumulator needs nothing. LLVM
  does the same (part 1). In HIP the compiler handles it; it only matters for hand-written assembly.
- READ [MESA]: a zero or constant accumulator can be an inline constant operand, so no instruction is spent
  clearing it.

## What this means for an exact FP8-weight prompt kernel (INFERRED)

1. The natural tile is 16x16 with K = 32 per step: 16 weight bytes per lane per load, as AMD recommends.
2. With F16 or BF16 activations the weights must be widened first, so the 16-bit instruction (16 cycles) is used and
   each 16-byte load of weights feeds two of them. Widening costs vector instructions that cannot overlap with the
   matrix instruction if RDNA 4 behaves as the tools say.
3. The transposing 8-bit load may let weights stored row by row go straight into the K-major layout without a pass
   through shared memory. Untested anywhere public.
4. Shared-memory layouts must spread accesses over all banks; AMD's samples avoid shared memory altogether.
5. Watch for register spills from loop unrolling with the ROCm compiler; check the generated code as on RDNA 3.
6. Bit-exactness against a reference depends on addition order; define the order and test it.

## Not read

- The figures in the GPUOpen articles and the benchmark images in PR 17156.
- The flash-attention pull request the GPUOpen articles refer to.
