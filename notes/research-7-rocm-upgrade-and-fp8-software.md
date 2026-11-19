# RDNA 4 research, part 7: the ROCm upgrade, and FP8 in existing software

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 deep research: ROCm upgrade and FP8
software paths"); items marked CHECKED were re-read by the lead. READ = stated in the source, seen by the helper
only. INFERRED = reasoning. This machine: a Fedora 44-based distribution, kernel 7.3, ROCm 7.1.1 from Fedora packages,
PyTorch 2.13.0+rocm7.1 in a virtual environment.

## A. Upgrading ROCm (to do after the card swap; owner decision 2026-11-16)

### Versions

- READ (<https://rocm.docs.amd.com/en/latest/release/versions.html>): 7.1.0 (2025-10-30), 7.1.1 (2025-11-26), 7.2.0
  to 7.2.4 (January to May 2026), 7.14.0 (2026-07-15), 10.0.0 (2026-08-26), 7.14.1 (2026-09-02), 10.1.0 (2026-10-05).
  There is no 8 or 9. AMD: "ROCm 10.0 marks that decade with a major version bump, and the jump is deliberate"
  (ten years since ROCm 1.0). The new build system, TheRock, became the production one in 7.14.
- READ: ROCm 10.1 moves the compiler "from LLVM 23 to LLVM 24". The installed hipcc is clang 20. HIP itself is
  7.16.0, hipBLASLt 1.4.1, rocBLAS 5.7.0 in the 10.1.0 matrix.
- INFERRED: four LLVM versions of compiler changes sit between what was used on RDNA 3 and ROCm 10.1. Kernel code
  generation (register use, loop unrolling, instruction pairing) may differ in either direction; the instruction
  counts from the RDNA 3 work should be re-checked, not assumed.

### What AMD supports

- READ (<https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>): for RDNA 4 on Linux: Ubuntu
  26.04, 24.04, 22.04 and RHEL 10.2, 9.8. No Fedora. gfx1201 is listed ("AMD Radeon RX 9070 XT (gfx1201)").
- READ: install methods: package manager, `amdgpu-install`, pip, tarball, runfile. AMD documents its own kernel
  driver package (DKMS) for Radeon cards; the driver built into a newer kernel is not a documented path, though it
  is what Fedora uses.

### Routes on Fedora 44

| Route | What it is | Notes |
|---|---|---|
| Stay on Fedora's packages | ROCm 7.1.1 | CHECKED: Fedora 44 has 7.1.1, Fedora 45 has 7.2.x, Rawhide (Fedora 46) has 10.1.0. Fedora's policy locks a release to its ROCm minor version, so Fedora 44 will not get 10.x. |
| TheRock tarball | Unpack ROCm 10.1 into a directory of our choice, leave the system packages alone | READ: tarballs "do not come with 'install' steps"; layout mirrors `/opt/rocm`. Lowest risk to the system; point the build at it with paths. |
| TheRock pip packages | `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-devel`, `rocm-sdk-device-gfx1201` in a virtual environment | READ. Same packages the PyTorch ROCm 10 wheels use. |
| AMD's RPM repository | `dnf install amdrocm` from `stable.repo.amd.com` | READ. Built for RHEL; would sit beside or replace Fedora's packages. Highest risk of conflicts. |

Source: <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>. TheRock lists gfx1201 as build-passing,
sanity-tested and release-ready.

- INFERRED risks: two ROCm installs fighting over library paths (the Fedora one in `/usr`); hipcc picking up the
  wrong LLVM; a kernel (7.3) newer than any AMD lists. None is documented as a known failure.
- INFERRED recommendation, for the owner to decide after the swap: the tarball or pip route into the project's own
  folder, keeping Fedora's 7.1.1 installed. That keeps the RDNA 3 baseline reproducible and makes the upgrade
  reversible. Measure with 7.1.1 first, then with 10.1, so the effect of the upgrade is itself a result.
- Not read: COPR (blocked), the distribution's own notes.

### PyTorch for ROCm 10

- CHECKED: `https://download.pytorch.org/whl/rocm10.0/` has `torch-2.15.0+rocm10.0` only, and carries the ROCm
  libraries as pip packages (`rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1201`, ...). So a ROCm 10
  PyTorch brings its own ROCm and does not use the system one.
- READ: pip can pick a `rocm7` build over `rocm10` because of how the version strings sort; pin the exact version.
- READ: an open PyTorch issue on gfx1201 (2026-11-10): with `expandable_segments:True` the allocator "silently
  overwrites live tensors" (torch 2.14.1+rocm7.2, RX 9070). Do not enable that option.

### llama.cpp

- READ: its build notes give no ROCm version range; no ROCm 10 build problem was found in a first search.

## B. FP8 in existing software, and what "exact" means there

### llama.cpp (CHECKED by the lead in the source and the pull request)

- #27512 "Quant: OCP FP8 E4M3 support" (2026-10-29), #28392 "Add FP8 KV-cache support" (2026-11-10), #28413 "CUDA: Add
  OCP FP8 support" (2026-11-11).
- Matrix-vector path (1 to 8 columns): FP8 weights, FP32 activations, FP32 sum. Commit titles in #28413: "Do W8A16 for
  FP8 in GEMV case", then "Just do FP8 x FP32 x FP32 in MMVF as its fast enough". This is the exact path, and the one
  our RDNA 3 kernel replaced.
- Matrix-matrix path on NVIDIA with FP8 hardware: activations are scaled by their largest value and rounded to FP8,
  then cuBLASLt multiplies FP8 by FP8 with FP32 output (`fp8.cu`, part 1 section 6). Lossy by design.
- Matrix-matrix path elsewhere (all AMD): FP8 weights are converted to F16 and the F16 library is used.
- A reviewer's note in #27512: "AMD FNUZ is only CDNA3", "RDNA4 and CDNA4 use FP8 OCP". No plan for AMD FP8 hardware
  is stated. Follow-ups exist for Vulkan ("vulkan: add fp8 and scaled matmul support", #29791).
- INFERRED: there are two honest targets on RDNA 4. "Exact" means matching llama.cpp's matrix-vector path and its
  F16 path's inputs with FP32 sums. "Same as NVIDIA" means the FP8 x FP8 path with rounded activations, which
  hipBLASLt could provide with little new code. The owner chose exact first.

### hipBLASLt

- CHECKED (part 4): OCP FP8 types are supported on gfx12.
- READ: the documented type table has FP8 x FP8 with FP32 compute; no row with FP8 on one side and F16, BF16 or F32
  on the other was found. Treat mixed precision as unsupported until a call proves otherwise.
- READ: the scale pointers are single scalars for the whole tensor by default
  (<https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/datatypes.html>). Nothing says whether the sum
  is exactly FP32 at each step.

### Prior art for exact FP8-weight kernels

| Project | Hardware | How weights are widened | Exact? |
|---|---|---|---|
| vLLM "FP8 Marlin" (<https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin>) | NVIDIA Turing and Ampere, weight-only FP8 | Bit operations on four bytes in one register, then a multiply by 2^8; scale factors absorb the exponent offset | not stated |
| feather-ops (<https://github.com/woct0rdho/feather-ops>) | AMD RDNA 3 and 3.5, tested on gfx1151 | Raw FP8 bytes go to shared memory; each wave reads 128 bits and widens in registers before the F16 matrix instruction. E5M2 is "free" (same exponent bias as F16); E4M3 needs a rebias | No: "Denormals are dropped in every form." |
| Our RDNA 3 kernels (rdna3-fp8) | gfx1100 | Shift widening to F16 times 2^-8, the 2^8 folded into the scale | Yes, subnormals included, checked over all inputs |

- READ: no exact FP8-weight kernel for RDNA 4 was found anywhere.
- READ: vLLM lists "Radeon RX 9000 series (gfx1200/1201)" as supported on ROCm, with ROCm 10 wheels, but documents
  FP8 only for NVIDIA and MI300. Which kernel it would pick for FP8 on gfx1201 is unknown.

## Not verified

- Whether a COPR or distribution package for ROCm 10 exists.
- hipBLASLt's full FP8 type table and the order of its sums.
- vLLM FP8 on gfx1201 in practice.
