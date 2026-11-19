# RDNA 4 research, part 4: ROCm and ML software

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 research: ROCm and ML software");
items marked CHECKED were re-read by the lead against the source. READ = stated in the source, read by the helper
only. INFERRED = reasoning. The helper read pages through a page-reading helper of its own.

## ROCm

- CHECKED: the current ROCm is 10.1.0 (<https://rocm.docs.amd.com/en/latest/about/release-notes.html>). Its support
  lists name gfx1200 and gfx1201 for the core SDK on Linux and for PyTorch, vLLM 0.29.0, ONNX Runtime 1.29.0 and
  MIGraphX. This machine has ROCm 7.1.1 from Fedora, three major versions behind.
- READ: RX 9070 support first appeared in ROCm 6.4.1 (May 2025), from a mirror of the release notes.
- READ: a comment in a llama.cpp thread says WMMA for RDNA 4 was not implemented before ROCm 7.
- READ: no open known issue in the 10.1.0 notes names gfx1201. Fixed ones: GPU resets during PyTorch training on RX
  9070, "memory coherency issues on gfx1201" (7.2.x), wrong rocBLAS `trsv`/`asum`/`nrm2` results on gfx12 (later
  release). INFERRED: with 7.1.1 some of these fixes are missing; worth knowing before trusting a wrong result.

## hipBLASLt (AMD's matrix library with FP8)

- CHECKED in part: the data-type page says the FNUZ FP8 types are "only supported on the gfx942 platform" and that
  `hipblaslt_f8` and `hipblaslt_bf8` (the OCP types, `HIP_R_8F_E4M3` and `HIP_R_8F_E5M2`) are only supported on
  gfx950 and gfx12 (<https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html>).
- CHECKED locally (part 1): the installed hipBLASLt 7.1.1 ships gfx1201 kernels for FP8/BF8 inputs with F32, F16,
  BF16, FP8 or BF8 output.
- READ: FP8 GEMM has per-tensor scale options (`scaleA,B` in the API table), an "amax of D" output option, and
  Stream-K for FP8 since 1.0.0 (ROCm 7.0). A later release mentions improved "fp8, bf8, and int8 TN GEMM performance
  on gfx1201" and another turns off uniform summation order on RDNA
  (<https://raw.githubusercontent.com/ROCm/rocm-libraries/develop/projects/hipblaslt/CHANGELOG.md>).
- INFERRED: a hipBLASLt FP8 GEMM needs both operands in FP8, like the instruction it uses. It is the natural
  baseline for any FP8 x FP8 kernel written here, as hipBLAS F16 was on RDNA 3.

## Other ROCm libraries

- CHECKED in part: rocWMMA's data-type table lists gfx1200 and gfx1201, and says "f8 NANOO (optimized) format is only
  supported on gfx942, otherwise f8 OCP is assumed on targets that support f8 datatypes". Older rocWMMA guides show
  FP8 as CDNA only, so FP8 on gfx12 depends on the rocWMMA version.
- READ: MIGraphX supports gfx1200/gfx1201 since 2.12 (ROCm 6.4.0) and "hardware specific FP8 datatypes (FP8 OCP and
  FP8 FNUZ)"; 2.14 restricted FP8 quantization to `dot` and `convolution`.

## Compiler

- CHECKED locally (part 1): the installed clang emits the gfx12 FP8 instructions from builtins.
- READ: LLVM main names further subtarget features: `wmma-128b-insts`, `swmmac-gfx1200-insts`,
  `ocp-fp8-conversion-insts`, `wmma-coexecution-hazards`
  (<https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/GCNFeatures.td>). The last name
  suggests the compiler knows about matrix instructions running alongside other vector instructions; what it does was
  not read.
- READ: the FP8 conversion erratum is a gfx940 (MI300) problem. An LLVM reviewer: "GFX12 does not have SDWA and hence
  does not have the bug."
- READ: the gfx12 WMMA and sparse WMMA instructions entered LLVM in January 2024 (PR #77795).

## PyTorch

- CHECKED: `torch._scaled_mm` is allowed on gfx1200 and gfx1201 when PyTorch is built against ROCm 6.3 or newer
  (`aten/src/ATen/native/cuda/ScaledBlasDeviceUtils.cpp`, main: the list is gfx942, then gfx1200 and gfx1201 under
  `ROCM_VERSION >= 60300`, gfx950, gfx1250, gfx1170/1171). The error text still says "ROCm MI300+", which is stale.
- CHECKED: on ROCm the FNUZ variant is used only on gfx942; other devices use `float8_e4m3fn`, the model-file format
  (`ScaledBlas.cpp:73-75`).
- CHECKED: PyTorch's hipBLASLt architecture lists include gfx1200 and gfx1201 (`CUDAHooks.cpp:548-575`).
- READ: `CUDABlas.cpp` notes "gfx11 and gfx12 have no CK fp32 GEMM".
- INFERRED: the PyTorch 2.13.0+rocm7.1 already installed here should run `torch._scaled_mm` with FP8 on the 9070
  XT. On the 7900 XT it raised. First thing to try for the PyTorch side.

## llama.cpp

- CHECKED locally (part 1): RDNA 4 detection, gfx12 WMMA builtins for F16, BF16 and 8-bit integers, FP32 output for
  the F16 cuBLAS path on RDNA 4, and no FP8 hardware use under HIP.
- READ: the RDNA 4 WMMA kernels for quantized models came from PR #17156, merged 2025-11-24
  (<https://github.com/ggml-org/llama.cpp/pull/17156>). No PR for native FP8 on AMD was found.

## Not verified

- The first ROCm release with Windows support for the RX 9070.
- The full hipBLASLt table of FP8 input/output type combinations (the local kernel file names in part 1 are better
  evidence).
- Any FP8-specific RDNA 4 documentation for vLLM, llm-compressor, Lemonade or ONNX Runtime.
