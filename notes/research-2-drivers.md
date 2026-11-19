# RDNA 4 research, part 2: drivers, Vulkan and FSR

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 research: drivers and graphics
stack"); the items marked CHECKED were re-read by the lead against the source page. READ = stated in the source.
INFERRED = reasoning. Mesa's GitLab and Phoronix blocked automated reads, so merge-request pages were not read; the
Mesa facts come from the official release notes.

## Mesa (RADV, ACO)

| Mesa | Date | What it says | Status |
|---|---|---|---|
| 25.0.0 | 2025-02-19 | "initial GFX12 (RDNA4) support on RADV"; "radv: disable VK_KHR_cooperative_matrix on GFX12" | CHECKED |
| 25.1.0 | 2025-05-07 | "radv: advertise VK_KHR_cooperative_matrix on GFX12" | CHECKED |
| 25.2.0 | 2025-08-06 | "aco: emit float8 wmma"; "aco/lower_to_hw: support saturating fp8 conversions"; "spirv,nir: emit saturating float8 cmat convert"; "radv: expose VK_EXT_shader_float8"; "radv/gfx12+: enable VK_KHR_shader_bfloat16"; cooperative_matrix2 conversions "behind an env var" | CHECKED (first five) |
| 26.1.0 | 2026-05-06 | "aco: allow dpp for fp8/bf8 dot4"; a Valve mixed-float dot-product extension on RADV | CHECKED (first) |
| 26.2.0 | 2026-08-05 | "radv/gfx11.7: enable float8"; "util: Fix float8 denorm rounding to min-normal"; gfx11.7 and gfx12.1 chips appear | CHECKED |

Sources: <https://docs.mesa3d.org/relnotes/25.0.0.html> and the same path for the other versions.

- INFERRED: Mesa 25.2.0 is the first release in which FP8 cooperative matrices work on RDNA 4. The Mesa run on this
  machine during the RDNA 3 work was 27.0.0-devel, so it is new enough.
- READ: BF16 cooperative matrices on gfx11 to gfx11.5 are an opt-in debug option (<https://docs.mesa3d.org/envvars.html>).
- Not found: the full table of cooperative-matrix types RADV reports on gfx12 (INT8, E5M2, BF16 shapes). vkd3d-proton
  requires at least F16 x F16 -> F32 and E4M3 x E4M3 -> F32 at 16x16x16, subgroup scope
  (`libs/vkd3d/device.c`, master). `vulkaninfo` on the card will settle it.
- Known Mesa issues listed for RDNA 4 are about DX11 rendering and video encode, none about FP8 or matrices.

## Vulkan and SPIR-V

- READ: `VK_EXT_shader_float8` adds E4M3 and E5M2 as 8-bit float types, usable in conversions and cooperative
  matrices (<https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_shader_float8.html>).
- READ: `SPV_EXT_float8` (rev 1, 2025-04-16): without the saturation decoration, an out-of-range conversion gives NaN
  for E4M3 and infinity for E5M2; with it, the largest finite value. No dot product on FP8 in SPIR-V; arithmetic is
  cooperative-matrix multiply-add and conversions only
  (<https://github.khronos.org/SPIRV-Registry/extensions/EXT/SPV_EXT_float8.html>). This matches the hardware
  FP16_OVFL behaviour in part 1.
- READ: `VK_KHR_shader_bfloat16` has separate feature bits for the type, dot product and cooperative matrix.
- READ: `VK_NV_cooperative_matrix2` adds use conversions, reductions and workgroup-scope matrices; RADV exposes its
  conversions only behind an environment variable (name not found).

## Kernel and firmware

- No primary source found. Press reports say kernel 6.12 or 6.13 and Mesa 25.0 as the minimum. This machine runs
  kernel 7.3, so it is not a concern.

## FSR 4 and FP8

- CHECKED, AMD, 2026-10-28, <https://gpuopen.com/learn/amd-fsr-sdk-2-4-rx-6000-ryzen-apus/>: FSR Upscaling 4.1.2
  "keeps the models already shipping on AMD Radeon RX 9000 Series (AMD RDNA 4 architecture, full FP8) and Radeon RX
  7000 Series discrete (RDNA 3 architecture, full INT8) GPUs", and adds the INT8 model for RX 6000 and some Ryzen
  APUs. SDK 2.4 also "adds Vulkan support at feature parity with DirectX 12" for upscaling and frame generation.
- CHECKED, AMD, 2026-06-24, <https://gpuopen.com/learn/amd-fsr-sdk-2-3-blog/>: SDK 2.3 made ML upscaling (FSR 4.1.1)
  official on RX 7000. Frame generation and ray regeneration stay RDNA 4 only.
- So AMD's own answer to "FP8 on RDNA 3" for FSR is a separate INT8 model, not FP8 emulation. The statement in the
  public rdna3-fp8 README that FSR 4 is officially RDNA 4 only is out of date.
- READ: AMD does not document the AGS matrix intrinsics FSR 4 uses on DirectX 12. What is public comes from the
  vkd3d-proton author's write-up (<https://themaister.net/blog/2025/05/>, May 2025): FSR 4 relies on FP8 WMMA and on
  FP32 to FP8 conversion between stages.
- READ: vkd3d-proton 3.0 (2025-11-17) maps those intrinsics to `VK_KHR_cooperative_matrix` and the float8 extension
  on RDNA 4, and has a slower emulation path for older cards that official builds do not enable
  (<https://github.com/HansKristian-Work/vkd3d-proton/releases/tag/v3.0>).

## Windows, briefly

- READ: DirectX cooperative vectors have an FP8 E4M3 data type and AMD shipped a preview driver for them
  (<https://devblogs.microsoft.com/directx/cooperative-vector/>, 2025-06-02). The wider HLSL matrix API was deferred.
  Whether AMD's retail driver maps them to FP8 hardware was not found.

## Not verified

- Mesa merge-request numbers and the exact instructions ACO emits.
- Any gpuinfo.org report for an RX 9070 XT (the tables load by script).
- What the 2025 FSR 4 source release contained beyond community reports of an INT8 model.
