# RDNA 4 research, part 1: instruction set and compiler

Date: 2026-11-16. Reading only; no RDNA 4 card was installed. Labels: READ = stated in the source; CHECKED = I ran a
local, compile-only or file-listing check on this machine; INFERRED = my reasoning.

Sources:
- [G4] AMD, "RDNA4" Instruction Set Architecture Reference Guide, 7 April 2025,
  <https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture>. Its licence forbids copying parts of it, so
  this file paraphrases and gives section numbers.
- [G3] AMD, "RDNA3" Instruction Set Architecture Reference Guide, 15 August 2023.
- [MC] AMD Matrix Instruction Calculator, `matrix_calculator.py` (MIT),
  <https://github.com/ROCm/amd_matrix_instruction_calculator>, fetched 2026-11-16 to `sources/matrix_calc.py`.
- [L20] LLVM `llvmorg-20.1.0`: `llvm/lib/Target/AMDGPU/AMDGPU.td`, `clang/include/clang/Basic/BuiltinsAMDGPU.def`
  (copies in `sources/`). ROCm's clang 20 is AMD's fork; this tag is the nearest public reference.
- [HIP] `/usr/include/hip/amd_detail/amd_hip_fp8.h` from the installed ROCm 7.1.1.
- [LC] llama.cpp master `f477c1b90` (`rdna3-fp8/build/llama.cpp-latest`).

## 1. What RDNA 4 adds for 8-bit floats

- READ [G4 7.5]: two 8-bit float formats. FP8 is sign, 4 exponent bits, 3 mantissa bits, bias 7, largest value 448, no
  infinity, NaN codes 0x7F and 0xFF. BF8 is sign, 5 exponent, 2 mantissa, bias 15, largest 57344, with infinities.
  They are supported in dot-product, matrix (WMMA) and convert instructions only. Single bytes in a register cannot
  be addressed by ordinary instructions.
- READ [HIP lines 33-50]: for gfx1200 and gfx1201 the HIP FP8 type is the OCP one (E4M3 with 448 as the largest
  value), with a hardware conversion fast path. gfx942 (MI300) uses the other variant ("fnuz", largest 240).
  INFERRED: RDNA 4's FP8 is the same format as llama.cpp's F8_E4M3 and the compressed-tensors FP8 checkpoints, so
  model bytes can be fed to the hardware unchanged. (The guide's table also shows a bias-8, max-240 mode; the guide
  does not say how a shader would select it.)

### Matrix instructions (WMMA) [G4 7.12, table 41]

| Instruction | A x B | Accumulator | In RDNA 3? |
|---|---|---|---|
| `V_WMMA_F32_16X16X16_F16`, `_BF16` | F16 or BF16 | F32 | yes |
| `V_WMMA_F16_16X16X16_F16`, `V_WMMA_BF16_..._BF16` | F16 or BF16 | F16 or BF16 | yes |
| `V_WMMA_I32_16X16X16_IU8`, `_IU4` | 8-bit or 4-bit integers | I32 | yes |
| `V_WMMA_I32_16X16X32_IU4` | 4-bit integers, K = 32 | I32 | new |
| `V_WMMA_F32_16X16X16_{FP8,BF8}_{FP8,BF8}` (4 forms) | 8-bit floats | F32 | new |
| `V_SWMMAC_*_16X16X32_*` (11 forms, FP8 included) | sparse A (2 of every 4 elements zero), K = 32 | F32, F16, BF16 or I32 | new |

- READ: there is NO mixed form with FP8 on one side and F16, BF16 or F32 on the other. Both operands of an FP8 matrix
  instruction are 8-bit floats. The FP8 forms accept no negate, absolute value, clamp or overflow-mode modifiers.
  A and B must be registers; C may be a register or an inline constant.
- READ [G4 7.12.2, table 43]: in wave32 a 16x16 FP8 matrix takes 2 registers (a 16-bit one takes 4, an accumulator 8).
  RDNA 4 does not replicate the operand across the two lane halves: the gfx12 compiler builtins take 8 halves per
  lane where the gfx11 ones take 16 [L20 lines 314 vs 517]. llama.cpp calls these "128-bit" and "256-bit" WMMA [LC
  `common.cuh:292-298`].
- READ: operand layout. A: one matrix row runs across registers inside a lane. B, C, D: one row runs across lanes
  inside a register. For 8-bit data in wave32: A element (row, col) is in lane `{col[3], row[3:0]}`, register
  `col[2]`, byte `col[1:0]`; B is the mirror image. Accumulator element (row, col) is in lane `(row / 8) * 16 + col`,
  register `row % 8`.
- READ [G4 7.12.1]: hazards. If a matrix instruction's A, B or index operand overlaps the previous matrix
  instruction's result, at least one unrelated vector instruction (or `V_NOP`) must sit between them, or the result
  is wrong. Chaining the result as the next C is allowed; it stalls when the two instructions are of different
  types.
- READ [G4 16.10]: as on RDNA 3, the matrix instructions run with the execution mask forced to all lanes.
- READ [G4 11.6]: `GLOBAL_LOAD_TR_B64` and `GLOBAL_LOAD_TR_B128` load a 16x16 tile of 8-bit or 16-bit data and
  transpose it between row-major and column-major on the way into registers. They need the execution mask all ones.
- READ [G4 7.12.3]: sparse form. A is stored packed (half the columns) plus one register of 2-bit index pairs that say
  which 2 of each 4 elements are non-zero. B must be dense, K x N, column-major.

### Speed of the matrix instructions (AMD's own table) [MC, per-instruction `cycles` field]

| Instruction | RDNA 3 | RDNA 4 |
|---|---|---|
| F16 or BF16, 16x16x16 | 32 cycles | 16 cycles |
| IU8, 16x16x16 | 32 | 8 |
| IU4, 16x16x16 | 16 | 8 |
| FP8 / BF8, 16x16x16 | not available | 8 |
| Sparse F16 / BF16, 16x16x32 | not available | 16 |
| Sparse FP8 / BF8 / IU8, 16x16x32 | not available | 8 |
| Sparse or dense IU4 with K = 32 or 64 | not available | 8 |

- The RDNA 3 figure matches what was measured on the RX 7900 XT (32 to 33 cycles per F16 WMMA), which is some
  evidence the tool's numbers are real cycle counts. The RDNA 4 numbers are READ, not measured.
- INFERRED: per instruction, an FP8 WMMA on RDNA 4 does the work of an RDNA 3 F16 WMMA in a quarter of the cycles,
  and an F16 WMMA in half. Whole-card throughput also depends on the number of compute units (fewer on the 9070 XT
  than on the 7900 XT; see the hardware part) and on whether matrix and ordinary vector instructions can overlap,
  which the guide does not say.

### Dot products and conversions

- READ [G4 16.10, opcode 38-41]: `V_DOT4_F32_{FP8,BF8}_{FP8,BF8}`: four 8-bit float products from two 32-bit words,
  summed in F32 and added to an F32 third operand. Round to nearest even. Both operands must be 8-bit floats.
- READ [G4 7.6]: conversions.
  - 8-bit to F32: `V_CVT_F32_FP8` / `_BF8` (one byte, selectable by a byte-select field) and
    `V_CVT_PK_F32_FP8` / `_BF8` (two bytes to two F32 values in two registers). Exact, no rounding needed.
  - F32 to 8-bit: `V_CVT_PK_FP8_F32` / `_BF8_F32` (two F32 to two bytes, round to nearest even) and
    `V_CVT_SR_FP8_F32` / `_BF8_F32` (one value with stochastic rounding: adds random bits from a second register to
    the mantissa, then truncates).
  - Overflow follows the MODE register's FP16_OVFL bit: when set, values beyond the largest FP8 number (and
    infinities) saturate to +-448; when clear they become NaN. NaN stays NaN.
  - There is NO direct FP8 to F16 or FP8 to BF16 conversion, and none from F16 to FP8.
- READ [L20 `AMDGPU.td:804-809, 1629, 1640`]: the erratum feature "cvt-fp8-vop1-bug" (one encoding of the FP8 to F32
  conversion is unreliable) is attached to gfx940 and gfx941 (MI300 family), not to gfx12.

### What that means for FP8-weight models (INFERRED, to be tested)

1. The native FP8 matrix instruction needs FP8 activations too. llama.cpp keeps activations in F32, so using
   `V_WMMA_F32_16X16X16_FP8_FP8` means rounding activations to 8 bits (3 mantissa bits) per layer. That is a
   different, lossy calculation from what llama.cpp does today, not a faster version of the same one. It is what
   FSR 4 does by design.
2. To stay exact (FP8 weights times F16 or F32 activations) the weights still have to be widened in software, as on
   RDNA 3. Options on RDNA 4: the RDNA 3 bit-shift widening to F16 (should carry over unchanged); or
   `V_CVT_PK_F32_FP8` to F32 (2 values per instruction, exact).
3. BF16 is a new option worth testing: FP8 widens to BF16 exactly (BF16 has F32's exponent range, so no overflow of
   the kind behind the RDNA 3 NaN bug), and F32 activations become BF16 by dropping the low 16 bits. The price is 8
   significant bits for activations instead of F16's 11.
4. The F16 WMMA itself is listed at half the cycles of RDNA 3, so the RDNA 3 prompt kernel design should gain on
   RDNA 4 even without native FP8, if it is not memory-bound.

## 2. Other instruction-set changes that matter for kernels

- READ [G4 7.8]: dual-issue (VOPD) pairs: the list is the same as RDNA 3 (FMAC/FMA with constant/MUL/ADD/SUB/MOV/
  CNDMASK/MAX/MIN, `DOT2ACC_F32_F16`, `DOT2ACC_F32_BF16`, and three integer ops in the second slot). `V_FMA_MIX_F32`
  is still not in it. Wave32 only.
- READ [G4 4.1.1]: every memory instruction now carries a cache scope and a 3-bit temporal hint: regular,
  non-temporal, high-priority, last-use, and three mixed near/far policies. RDNA 3's kernels gained from
  non-temporal loads; RDNA 4 gives finer control.
- READ [G4 8.5]: software prefetch: `S_PREFETCH_DATA` (and buffer and PC-relative forms) prefetch 1 to 32 chunks of
  128 bytes into the first-level cache. Ignored unless the MODE register enables scalar prefetch. Exposed as
  `__builtin_amdgcn_s_prefetch_data` [L20 line 495].
- READ [G4 11.5]: block loads and stores move up to 32 consecutive registers per lane in one instruction.
- READ [G4 3.3.3]: compute waves can be launched in a mode where they allocate and free vector registers at run
  time (`S_ALLOC_VGPR`, blocks of 16 or 32, wave32 only). Whether ROCm launches HIP kernels that way is unknown.
- READ [G4 6.8, 7.10]: scalar float arithmetic (`S_ADD_F32`, `S_FMAC_F32`, conversions, compares) and
  single-lane transcendental ops (`V_S_EXP_F32`, `V_S_RCP_F32`, ...).
- READ [G4 5.7]: the wait counters are split (`S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_DSCNT`, ...) in place of
  RDNA 3's combined `S_WAITCNT`.
- READ [G4 5.8]: `S_DELAY_ALU` hints still exist; the compiler inserts them.
- READ [G4 1.2.2.1, 12.1]: local data share is 128 kB per work-group processor in 64 banks; one work-group can use up
  to 64 kB. (Same as RDNA 3.) The guide gives no cache sizes.
- READ [L20 `AMDGPU.td:1822-1858`]: gfx12's compiler features include the FP8 conversions, the FP8 and BF16 dot
  products, scalar float instructions, "LDS bank count 32", and "50 % more physical vector registers and 50 %
  larger allocation granule" (a feature gfx1100 also has).

## 3. Toolchain on this machine (CHECKED 2026-11-16)

- `hipcc --offload-arch=gfx1201` (ROCm 7.1.1, clang 20) compiles `sources/probe.hip` and emits
  `v_wmma_f32_16x16x16_fp8_fp8`, `v_dot4_f32_fp8_fp8`, `v_cvt_f32_fp8` and `v_cvt_pk_fp8_f32` from the builtins
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, `__builtin_amdgcn_dot4_f32_fp8_fp8`,
  `__builtin_amdgcn_cvt_f32_fp8`, `__builtin_amdgcn_cvt_pk_fp8_f32`. Compile only; nothing was run.
- hipBLASLt 7.1.1 ships gfx1201 kernels for 8-bit floats: every combination of F8 and B8 inputs with F32, F16, BF16,
  F8 or B8 output (20 type sets, file names `TensileLibrary_F8F8_SF8_..._gfx1201`). For gfx1100 it ships none.
  INFERRED: an FP8 GEMM through hipBLASLt should work on RDNA 4 out of the box.
- rocBLAS 7.1.1 ships gfx1201 kernels (56 files against 96 for gfx1100).
- PyTorch 2.13.0+rocm7.1 lists gfx1200 and gfx1201 in its architecture list.

## 4. llama.cpp master and RDNA 4 [LC] (READ)

- It detects RDNA 4 (`GGML_CUDA_CC_RDNA4`, `common.cuh:86`) and uses the gfx12 WMMA builtins for quantized
  matrix products and attention (F16, BF16 and 8-bit integer forms, `mma.cuh:1174-1474`).
- It does not use FP8 hardware on RDNA 4. `ggml_cuda_mul_mat_fp8` (hipBLASLt-style FP8 GEMM) is compiled only for
  NVIDIA CUDA; under HIP it is a stub that returns false (`fp8.cu`). The hardware FP8-to-F32 conversion path covers
  only CDNA 3 (`common.cuh:899`); RDNA 4 takes the software `ldexpf` path like RDNA 3.
- For FP8 weights in the F16 cuBLAS path it already asks for F32 output on RDNA 4 (`ggml-cuda.cu:1934`), so the
  RDNA 3 NaN bug should not occur. INFERRED until run.

## 5. To measure once the card is in

- Cycles per FP8, F16 and BF16 WMMA, and whether vector instructions overlap with them.
- Exactness and rounding of the FP8 WMMA and `V_DOT4` sums (the RDNA 3 tables used a fitted model for RDNA 4).
- Cost per weight of: shift widening to F16, `V_CVT_PK_F32_FP8`, and the transposed 8-bit tile load.
- Memory bandwidth ceiling and cache trap size; effect of the temporal hints and of software prefetch.
- llama.cpp FP8 against Q8_0 and F16, unmodified and with the RDNA 3 patch's decode kernel.
- hipBLASLt FP8 GEMM speed against its F16 GEMM.

## 6. Addendum (2026-11-16, second round): compiler model, cache hints, llama.cpp's FP8 path

Sources: LLVM main, `llvm/lib/Target/AMDGPU/` files `SISchedule.td`, `GCNFeatures.td`, `GCNProcessors.td`,
`GCNHazardRecognizer.cpp`, `SIMemoryLegalizer.cpp`, `SIDefines.h` (copies in `sources/llvm_*`); llama.cpp master
`f477c1b90`, `ggml/src/ggml-cuda/fp8.cu`. All READ by the lead.

- Matrix and vector instructions do not overlap on RDNA 4, as far as the compiler knows. LLVM has a feature
  "wmma-coexecution-hazards" ("Hazards when WMMA co-executes with other WMMA/VALU instruction") and a separate matrix
  execution resource in its timing model, but only for the later targets gfx1250 and gfx1251
  (`GCNFeatures.td:2599-2734`). gfx1200 and gfx1201 use `GFX12SpeedModel`, which has no matrix entries at all.
  INFERRED: on RDNA 4 a matrix instruction occupies the vector unit for its whole duration, as measured on RDNA 3.
  To confirm on the card.
- The same file shows what the later chip adds, and so what RDNA 4 lacks: 16x16x64 FP8 matrix forms, mixed
  FP8/FP6/FP4 forms with scaling (16x16x128), and an FP32 matrix form (`SISchedule.td:474-520`).
- `GFX12SpeedModel` against `GFX11SpeedModel` (`SISchedule.td`): identical for ordinary vector ops (5), FMA (5),
  conversions (5), integer multiply (8), LDS (20), scalar memory (20), vector memory (320). Different: transcendental
  ops 9 against 10 and no longer on a separate resource, "quarter-rate" ops 6 against 8, and new single-lane
  transcendental ops at 8. The comment says the numbers are "1 / (operations / cycle)" plus one stall cycle for the
  register read, so they are a scheduling model, not measured latencies.
- Matrix hazard in the compiler (`GCNHazardRecognizer.cpp:2657-2700`): if A or B of a matrix instruction overlaps the
  previous matrix instruction's result, a `V_NOP` is inserted unless a vector instruction sits between them. For
  gfx12 the comment says overlap of C with the previous result is allowed ("hardware will stall"), but a sparse
  index register must not overlap it. Matches the guide.
- gfx12's feature list on main adds names the LLVM 20 list did not have: "wmma-128b-insts",
  "swmmac-gfx1200-insts", "ocp-fp8-conversion-insts", "cvt-fp8-byte-sel", "1536-physical-vgprs" ("Has 1536 physical
  VGPRs per SIMD"; gfx1100 has the same). 1536 registers x 32 lanes x 4 bytes = 192 KB, which matches the register
  file size reported by Chips and Cheese.
- Cache hints from source code (`SIMemoryLegalizer.cpp:2202-2223`, `SIDefines.h:610-618`): on gfx12 a load marked
  non-temporal gets the hint TH_NT, and one marked with the `amdgpu.last.use` metadata gets TH_LU. The mixed hints
  (non-temporal near / regular far, and so on; "far" is the memory-attached last-level cache, i.e. the Infinity
  Cache) and the high-priority hint are defined but not set by this code. INFERRED: from HIP,
  `__builtin_nontemporal_load` reaches TH_NT; the other hints need assembly or a compiler change.
- llama.cpp's NVIDIA FP8 path is not exact. `ggml_cuda_mul_mat_fp8` (`fp8.cu:26-190`) finds the largest absolute
  activation, divides by it so the range fits 448, rounds the activations to FP8 (`quantize_fp8_e4m3`), and calls
  cuBLASLt with FP8 for both operands, FP32 compute and output, passing the activation scale and the weight scale as
  scale pointers. So upstream already accepts rounding activations to 8 bits where FP8 hardware exists. An exact
  RDNA 4 path would be stricter than what llama.cpp does on NVIDIA.

## 7. Correction (2026-11-16): saturating FP32 to FP8 conversion

Section 1 says, from the guide, that with the MODE bit FP16_OVFL set, out-of-range values and infinities saturate to
+-448 in `V_CVT_PK_FP8_F32`. Mesa's compiler authors found the hardware does not do that for infinity. Their code
(`src/amd/compiler/instruction_selection/aco_select_nir_alu.cpp`, read by the lead from a GitHub fork of Mesa) says:
"Ideally we would want to use FP16_OVFL for the sat variants, but the ISA doc is wrong and Inf isn't clamped to
max_float." Mesa clamps to +-448 (or +-57344 for BF8) with `v_minimummaximum_f32`, which keeps NaN, and then
converts. HIP's own FP8 header does the same with a median-of-three clamp before `__builtin_amdgcn_cvt_pk_fp8_f32`
(`amd_hip_fp8.h:564-587`). For an exact FP32 to FP8 store on RDNA 4: clamp first, then convert; and test the
conversion over all inputs on the card, as was done for the software version on RDNA 3.
