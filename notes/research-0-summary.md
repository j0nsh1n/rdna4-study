# RDNA 4 documentation research: summary and index

Date: 2026-11-16, two rounds. Everything here comes from documents, source code, compile-only checks and read-only
system files. No RDNA 4 card was installed, so nothing about the RX 9070 XT is measured. Each part says which items
the lead re-checked against the source and which are a helper's reading only.

## Index

| File | What it covers |
|---|---|
| [research-1-instructions.md](research-1-instructions.md) | FP8, matrix, dot-product and convert instructions; compiler builtins; local toolchain checks; compiler timing model; cache hints; llama.cpp's FP8 paths; a correction on FP8 saturation |
| [research-2-drivers.md](research-2-drivers.md) | Mesa, Vulkan extensions for FP8 and BF16, FSR 4 |
| [research-3-hardware.md](research-3-hardware.md) | The card against the RX 7900 XT, AMD's matrix rates, community llama.cpp numbers |
| [research-4-compute-stack.md](research-4-compute-stack.md) | ROCm, hipBLASLt, PyTorch, llama.cpp support |
| [research-5-memory-and-decode.md](research-5-memory-and-decode.md) | Inside a work-group processor, caches, how waves wait for memory, registers and wave slots, clocks and power, one-stream speed |
| [research-6-matrix-kernels.md](research-6-matrix-kernels.md) | AMD's RDNA 4 matrix guides, the "double-K" load pattern, what llama.cpp's RDNA 4 kernels do and where they struggle |
| [research-7-rocm-upgrade-and-fp8-software.md](research-7-rocm-upgrade-and-fp8-software.md) | How to get ROCm 10 on Fedora 44, PyTorch for ROCm 10, FP8 in llama.cpp and hipBLASLt, prior art |
| [research-8-architecture-and-family.md](research-8-architecture-and-family.md) | The RDNA 4 family, ray tracing and other blocks, AMD's AI claims, successors (gfx1250, gfx13) |
| [research-9-kernel-driver.md](research-9-kernel-driver.md) | What the Linux driver and the ROCm runtime encode about the chip; power profiles; known bugs |
| [research-10-compilers.md](research-10-compilers.md) | Every gfx12 feature, hazard, counter limit and cache-control rule in LLVM and Mesa |
| [guide/](guide/) A to D | A full digest of AMD's RDNA 4 instruction guide, with changes from RDNA 3 per chapter and per opcode group |
| [baseline-7900xt-kfd-topology.txt](baseline-7900xt-kfd-topology.txt) | The RX 7900 XT's cache and SIMD table as the driver reports it, saved before the swap |
| [PLAN.md](PLAN.md) | What is ready on this machine and the checklist for after the swap |
| [../scripts/first-hour.sh](../scripts/first-hour.sh) | The first-hour checks as one command (card record, build, bandwidth, probes, decode, llama.cpp, libraries); usage in [../scripts/README.md](../scripts/README.md) |

## The picture in short

1. RDNA 4 has real FP8 hardware (matrix, dot product, conversions), and the Linux driver, AMD's math library and
   PyTorch use it. llama.cpp does not use it on AMD.
2. The FP8 matrix instruction multiplies FP8 by FP8 only. An exact path (FP8 weights, activations at higher
   precision) still has to widen the weights, as on RDNA 3. A lossy path that rounds activations to FP8 is what
   llama.cpp already does on NVIDIA.
3. Matrix work should be much faster than on the 7900 XT: AMD lists 16 cycles for an FP16 matrix instruction and 8
   for an 8-bit one, against 32 on RDNA 3 (which matches our measurement there). Whole card, dense FP16: about 190
   TFLOPS against about 100.
4. One-stream generation is limited by memory, and this card has 80 % of the 7900 XT's bus (640 against 800 GB/s).
   llama.cpp reaches only 73 to 79 % of that bus today, where our RDNA 3 kernel reached 94 % of its bus. That gap is
   the opportunity for decode.
5. The memory system changed in ways that favour more waves: the L1 level is gone, waves no longer wait for each
   other's slow loads, and a kernel can use up to 96 registers and still fill all 16 wave slots.
6. Nobody has published measurements of the FP8 instructions, the memory bandwidth of a compute kernel, or an exact
   FP8-weight kernel on RDNA 4.

## Facts worth remembering

| Fact | Where | Status |
|---|---|---|
| FP8 is E4M3 with largest value 448, the same format as model files | part 1 | read |
| No FP8 x FP16 matrix form; no direct FP8 to FP16 conversion | part 1 | read |
| FP8 to FP32 conversion exists, one or two values per instruction, with a byte select | part 1, guide C | read |
| Hardware FP32 to FP8 saturation does not clamp infinity as the guide says; clamp first | part 1 section 7 | Mesa source, checked |
| Matrix instruction cycles: FP8 8, FP16 16 | parts 1 and 3 | AMD's calculator and table |
| Whether matrix and vector instructions overlap: AMD's tools say no, AMD's slides show a separate matrix block | parts 1, 5 | open |
| Each lane holds 8 consecutive K elements; no replication; FP8 needs two fused instructions to use a full 16-byte load | part 6 | AMD guide, checked |
| A transposing 16x16 tile load from global memory exists and Mesa uses it | parts 1, 6 | read |
| Registers: 1536 per SIMD, 16 wave slots, blocks of 24; up to 96 per wave keeps all slots | parts 5, 10 | checked |
| Wait counters allow 63 outstanding loads per wave | part 10, guide A | read (compilers and guide agree) |
| Caches: 32 KB L0 per compute unit, 8 MB L2, 64 MB Infinity Cache, 128-byte lines | part 5 | AMD slides, checked |
| Only "non-temporal" and "last use" cache hints are reachable from plain HIP code; the others need buffer builtins or assembly | parts 1, 10 | read |
| Prefetch instructions exist but need a mode bit that may not be set | guide B | read |
| The old min/max instructions were renamed `_NUM`; new MINIMUM/MAXIMUM pass NaN through | guide D | checked |
| hipBLASLt ships FP8 kernels for gfx1201; `torch._scaled_mm` allows gfx1201 | parts 1, 4 | checked |
| ROCm 10.1 is current; Fedora 44 stays on 7.1; a self-contained ROCm 10 is the low-risk route | part 7 | checked |
| Memory clock may not reach its top level on some kernels; power settings swing results | part 5 | community reports |
| The driver exposes the card's real cache table in `/sys/class/kfd/kfd/topology` | part 9 | checked on the 7900 XT |

## To measure once the card is in (in order)

1. Read-only first: the driver's topology and cache table, clocks, power profile, `vulkaninfo` matrix types.
2. Bandwidth ceiling of a plain read kernel, with clocks logged; cache trap size.
3. llama.cpp unmodified on ROCm 7.1.1: FP8 against Q8_0 and F16, kernel tests, the NaN check.
4. Free baselines: hipBLASLt FP8 GEMM, `torch._scaled_mm`.
5. Instruction measurements: cycles per matrix instruction; whether vector instructions overlap with it; rounding and
   order of the FP8 and FP16 matrix sums; the FP8 conversions over all inputs (byte select, saturation).
6. Decode kernel: port the RDNA 3 kernel; registers at 64, 96, 120, 144; non-temporal against regular loads; more
   waves; 128-byte alignment.
7. Prompt kernel, exact: FP8 weights widened to FP16 (and to BF16), 16 weight bytes per lane feeding two matrix
   instructions; try the transposing load; watch for register spills.
8. Then upgrade to ROCm 10.1 in the project folder and repeat 2, 3, 6 and 7 to see what the upgrade changes.
9. Later, the lossy route: FP8 activations with the native FP8 matrix instruction, with quality measured.

## Decisions (owner, 2026-11-16)

- No FSR 4 comparison between RDNA 3 emulation and RDNA 4. The 7900 XT can come out without captures.
- ROCm will be upgraded after the card swap, not before.
- Exact methods first (FP8 weights, activations kept at higher precision). The lossy FP8-activation route is
  interesting but comes second.
- The outdated FSR 4 line in the public rdna3-fp8 README was corrected (commit 788113a).

## Open points for the owner

- Kernel: no update needed. Correction of an earlier statement: the August 2026 fix for compute jobs dropped under
  queue oversubscription (commit 288cc4a54a00) is contained in Linux 7.3, checked against the kernel history on
  2026-11-17; this machine runs 7.3.4. The card's MES firmware is 0x97, newer than the 0x8b the idle-power fix needs.
- Which ROCm 10 route to use after the swap (part 7 recommends a self-contained copy in the project folder).
