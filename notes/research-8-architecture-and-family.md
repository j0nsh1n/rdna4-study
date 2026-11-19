> Provenance: written by a Sonnet helper from web sources (2026-11-16); several outlets blocked page reads, so parts
> rest on search summaries, as marked. Checked by the lead: the chip figures agree with AMD's Hot Chips slides (parts
> 3 and 5 of these notes); the successor targets gfx1250, gfx1251 and gfx1310 exist in LLVM's processor list; the FSR
> 4 on RX 7000 statement is confirmed from AMD's own page (part 2). Not checked: prices, dates and the product list.
>
> Lead's note on the conflicting matrix figures: 64 compute units x 2.97 GHz gives 190 TFLOPS at 1024 operations per
> clock per unit (dense FP16, AMD's GPUOpen table), 380 at 2048 (dense 8-bit, or FP16 with sparsity), 760 at 4096 and
> 1520 at 8192. So "389" is most likely FP16 with sparsity or dense 8-bit, and "1557" a 4-bit figure with sparsity.
> The dense FP16 rate of about 190 to 195 TFLOPS is the one that matters for an exact kernel.

# RDNA 4 architecture and product family: research notes

Compiled 2026-11-16. Method: web search plus page fetches. Labels: READ = stated in a source I opened (or in a search-result summary of it, marked "search summary"); INFERRED = my deduction. Quotes are short and exact as returned by the fetch tool. Several primary sites blocked automated fetches (TechPowerUp, Phoronix, HotHardware, Guru3D and TweakTown returned HTTP 403; Tom's Hardware returned only headlines; AMD product pages timed out), so much of AMD's slide content is reported via press. Those gaps are listed at the end.

Source key (used below):
- CC-RT = Chips and Cheese, "RDNA 4's Raytracing Improvements", 2025-04-14, https://chipsandcheese.com/p/rdna-4s-raytracing-improvements
- CC-VID = Chips and Cheese, "AMD's RDNA4 architecture (video)", 2025-03-05, https://old.chipsandcheese.com/2025/03/05/amds-rdna4-architecture-video/
- CC-OOO = Chips and Cheese, "RDNA 4's Out-of-Order Memory Accesses", 2025-03-23, https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses
- CC-DYN = Chips and Cheese, "Dynamic Register Allocation on AMD's RDNA 4", 2025-04-05, https://old.chipsandcheese.com/2025/04/05/dynamic-register-allocation-on-amds-rdna-4-gpu-architecture/
- CC-1250 = Chips and Cheese, "Scrying the AMD GFX1250 LLVM Tea Leaves", 2026-07-19, https://chipsandcheese.com/i/207602737/similarities-to-rdna4
- WCC-DD = Wccftech, "AMD RDNA 4 Architecture Deep-Dive", 2025-02-28, https://wccftech.com/amd-rdna-4-architecture-deep-dive-new-compute-units-raytracing-cores-ai-enhancements-path-tracing/ (reproduces AMD's slides)
- STH-HC = ServeTheHome, AMD RDNA 4 at Hot Chips 2025, 2025-08-25, https://www.servethehome.com/?p=89447
- WCC-HC = Wccftech, RDNA 4 modular SoC (Hot Chips 2025), 2025-08-26, https://wccftech.com/amd-rdna-4-modular-soc-flexible-configurability-helps-spawn-smaller-gpus-navi-44-features-to-reduce-memory-bandwidth/

---

## 1. Product family and silicon

### Silicon (READ unless noted)
Two dies. gfx1201 = Navi 48, gfx1200 = Navi 44 (search summary of press on ROCm/LLVM IDs; the LLVM docs ELF table lists GFX1200 = 0x048 and GFX1201 = 0x04e, https://llvm.org/docs/AMDGPUUsage.html, read via fetch summary, no date).

| Die | gfx target | Full config | Memory | Notes |
|---|---|---|---|---|
| Navi 48 | gfx1201 | 64 CU, 4 shader engines, 4096 shaders, 64 MB Infinity Cache, 8 MB L2 | 256-bit GDDR6, up to 20 Gbps | 53.9 B transistors, 356.5 mm2, TSMC 4 nm (WCC-DD) |
| Navi 44 | gfx1200 | 32 CU, 2 shader engines, 2048 shaders | 128-bit GDDR6 | about 199 mm2 (search summary only, not a primary number) |

Quote, Navi 44 layout (WCC-HC): "configured for the Navi 44 with two Shader Engines and four GDDR6 (128-bit) memory controllers". Quote on SKU count (WCC-HC, AMD Hot Chips): "AMD has four Navi 48 SKUs and three Navi 44 SKUs available".

Design stance (STH-HC, 2025-08-25): "AMD designed Navi 48 in such a way that it can essentially be cut in half to make a smaller GPU". Harvest levels per WCC-HC: SE harvest, WGP harvest, asymmetric harvest, and memory-device harvest at 64-bit granularity.

### Products
| Product | Chip / target | CU | Memory | Board power | Clocks (game / boost) | Price, launch | Source |
|---|---|---|---|---|---|---|---|
| RX 9070 XT | Navi 48 / gfx1201 | 64 | 16 GB GDDR6, 256-bit, 20 Gbps, 640 GB/s | 304 W | 2400 / 2970 MHz | $599, 2025-03-06 | search summary of Jon Peddie, Tom's Hardware, KitGuru (links in search result); 640 GB/s is derived (256 bit x 20 Gbps /8) |
| RX 9070 | Navi 48 / gfx1201 | 56 (CC-VID: "8 fewer Compute Units") | 16 GB | 220 W | boost about 2.52 GHz | $549, 2025-03-06 | same; 2.52 GHz is a revised figure per search summary (slides said 2.54) |
| RX 9070 GRE | Navi 48 "XL" / gfx1201 | 48 | 12 GB GDDR6 18 Gbps, 192-bit, 432 GB/s, 48 MB Infinity Cache, 96 ROPs | 220 W | 2220 / 2790 MHz | China only first: preorders 2025-04-30, shipments 2025-05-08, 4199 RMB; later western release (see Conflicts) | hwcooling.net 2025-04-30, https://www.hwcooling.net/en/radeon-rx-9070-gre-officially-unveiled-only-for-china-for-now/ |
| RX 9060 XT 8 GB / 16 GB | Navi 44 / gfx1200 | 32 | GDDR6 20 Gbps, 128-bit | 150-182 W (table says 142-182 W) | 2530 / 3130 MHz | $299 / $349, on sale 2025-06-05 | Wccftech 2025-05-20 https://wccftech.com/amd-radeon-rx-9060-xt-unveiled-navi-44-gpu-2048-cores-3-1-ghz-clocks-16-gb-8-gb/ ; Tom's Hardware headline "launches on June 5, starting at $299" |
| RX 9060 | Navi 44 / gfx1200 | 28 (1792 shaders) | 8 GB GDDR6 18 Gbps, 128-bit | 132 W | 2400 / 2990 MHz | $329, table says "2026"; one database says OEM-only 2025-08-05 | Wccftech table (edited after publication) |
| RX 9050 8 GB | Navi 44 cut / gfx1200 | 16 (1024 shaders) | 8 GB, 128-bit, about 288 GB/s | 92 W | 1920 / 2600 MHz | $279 SEP, July 2026, APJ and Latin America listed | Wccftech 2026-07-28 https://wccftech.com/amd-preps-entry-level-radeon-rx-9050-gpu-comes-in-8-gb-4-gb-flavors/amp/ |
| RX 9050 4 GB | same | 16 | 4 GB, 64-bit | "75W?" | - | OEM only: "There are currently no plans to sell the 4GB variant outside of the OEM system." | same |
| RX 9060 XT LP | gfx1200 | not found | not found | not found | - | listed in ROCm 7.14 table | ROCm docs, 2026-07-15 (see section 8) |

### Workstation / compute cards (all Navi 48 / gfx1201 unless noted)
| Product | CU | Memory | Power | Form | Date / price | Source |
|---|---|---|---|---|---|---|
| Radeon AI PRO R9700 | 64 (4096 shaders, 128 AI accelerators, 64 MB Infinity Cache) | 32 GB GDDR6, 256-bit, 20 Gbps, 640 GB/s | 300 W | dual-slot blower, 4x DP 2.1 | announced Computex 2025-05-20/21; retail from 2025-10-27, $1,299 | Club386, 2025-10-25, https://www.club386.com/amds-retail-launch-of-the-1299-radeon-ai-pro-r9700-32gb-gpu-is-imminent/ ; PCGH 2025-05-21 |
| Radeon AI PRO R9700S | 64 | 32 GB, 640 GB/s | up to 300 W | passive, for racks | announced 2025-12-11, no price at announcement | Notebookcheck 2025-12-11 (updated 2026-10-05), https://www.notebookcheck.net/AMD-unveils-two-new-Radeon-Pro-GPUs-for-AI-workloads.1183454.0.html |
| Radeon AI PRO R9600D | 48 (3072 shaders) | 32 GB, 256-bit | 150 W | passive | same | same |
| Radeon AI PRO R9600 | not found | not found | not found | not found | ROCm 10.1.0 notes add it, 2026-10-05 | ROCm release notes (section 8) |

R9700 vs the gaming card (READ, Club386 and PCGH): same full Navi 48 (64 CU, 4096 shaders), same 300 W class; the difference is 32 GB versus 16 GB, a blower multi-GPU-friendly design ("The reference model is designed for multi-GPU setups"), and $1,299 versus $599. ECC: Notebookcheck says the R9700S and R9600D support ECC "but only on Linux" (READ, secondhand paraphrase). I found no source stating ECC, ISV certification or driver branch for the R9700 itself (see Could not verify).

### Mobile and integrated
- No RDNA 4 laptop discrete GPU (RX 9000M or similar) was found; I could not search for it effectively. NOT VERIFIED.
- LLVM lists gfx1170, gfx1171, gfx1172 as "GCN GFX11.7 (RDNA 4m)" APUs with product name TBA (MacMyths 2026-09-23, https://macmyths.com/amds-llvm-backend-lists-gfx1171-and-gfx1172-in-its-rdna-4m-apu-branch/, secondary site citing LLVM docs). Quote: "AMD has not publicly tied gfx1171 or gfx1172 to a named processor." So gfx11.7 is an RDNA 4-branded integrated-GPU family, not announced as a product. The article notes Phoronix suggested a possible link to the rumored Medusa Point family; this is speculation.
- Strix Halo is RDNA 3.5 (gfx1151); that is not RDNA 4. Not verified in this session; treat as background knowledge (INFERRED, from memory).

---

## 2. Compute unit and shader core

AMD's own wording (WCC-DD, AMD slides as reproduced, 2025-02-28): "Dual SIMD32 Vector Units and Enhanced Matrix Operations." Also listed: "Matrix load w/transpose" and "New 8b Float Data Types."

Layout (READ, WCC-HC 2025-08-26): "The Shader Engine houses multiple WGPs or Work Group Processors, each with dual Compute Units." Navi 48 therefore has 4 SE x 8 WGP x 2 CU = 64 CU (INFERRED arithmetic from 64 CU and 4 SE; WCC-DD says 4 matrix engines and 2 ray accelerators per DCU, i.e. per WGP... that is "4 Matrix Acceleration Engines per DCU and 2 Ray Accelerators per DCU").

Stated per-CU gain (WCC-DD): "almost a 2x uplift in rasterization, close to 2.5x uplift in raytracing and a 3.5x uplift in ML (FP16 dense matrix) workloads per compute unit." Note: this baseline is RDNA 2, not RDNA 3 (the quote text says "vs RDNA 2" in the article's context per the fetch summary). Another claim from press summaries: up to 40 percent per-CU gain mostly from higher clocks (search summary, outlet not named; weak).

### Scheduling and registers
- Dual-issue is not improved: CC-VID: "Dual-issue is not improved" (paraphrase in the fetch report, not a verified verbatim quote). Treat as READ-by-paraphrase.
- Register file: CC-DYN: "RDNA 4 desktop GPUs have a 192 KB register file per SIMD." 256 VGPRs max per wave, 16 wave slots per SIMD, full occupancy at 96 or fewer VGPRs.
- Dynamic VGPR mode: `SQ_DYN_VGPR`, threads request registers with `s_alloc_vgpr`; blocks of 16 or 32 registers, up to 8 blocks per wave. Limits: "can only be used with wave32 compute shaders" (CC-DYN, paraphrase); graphics shaders and wave64 use the normal mode. A dynamic workgroup takes over its WGP/CU. 7 VGPR blocks per SIMD are reserved for deadlock avoidance, yet deadlock is still possible if two waves depend on each other. Chips and Cheese saw the driver use it only for indirect RT mode, not inlined RT.
- STH-HC: "RDNA 3 would allocate registers based on the worst case scenario." (READ, motivation).
- Scalar unit: "the scalar units have added floating point support" (STH-HC). CC-VID says this carried over from RDNA 3.5.
- Barriers: CC-VID: split barriers added; Register Block Spill/Fill works on up to 32 registers. AMD's slide listed named barriers but CC-VID says "Named Barriers are not a part of RDNA4" (CONFLICT with the slide, see Conflicts; later CC-1250 says RDNA 4 got "split barriers and barrier objects", a further conflict).
- Out-of-order memory (CC-OOO, 2025-03-23): AMD slide says RDNA 4 "adds new out-of-order queues for memory requests." RDNA 3 had cross-wave false dependencies on `vmcnt`; Chips and Cheese tests show these are gone. Counters were split: `vmcnt` into several counters, scalar loads on `kmcnt`, LDS on `dscnt` (split from `lgkmcnt`). The article calls these "generational tweaks."
- Wave modes: wave32 and wave64 both exist in RDNA 4 (LLVM docs feature list gfx1200 "wavefrontsize64", per fetch summary of llvm.org). Dynamic VGPR is wave32 only.
- Instruction cache, constant cache sizes, and clock comparison with RDNA 3 were not found in the sources I could read. WCC-DD lists "2MB Aggregate CU cache" with no breakdown (INFERRED: likely sum of L0 vector, scalar and instruction caches across the chip; unverified).
- Clocks: 9070 XT boost 2.97 GHz; 9060 XT 3.13 GHz (see products). KitGuru measured an average 2726 MHz (Quiet BIOS) and 2830 MHz (Performance BIOS) in a 30-minute game run, 2025-03-05 (page 28 of the review). RDNA 3's top boost (RX 7900 XTX) was about 2.5 GHz (INFERRED from memory, not sourced here). So RDNA 4 clocks are roughly 15-20 percent higher; the "40 percent per CU" claim leans on that.

---

## 3. Memory and cache

| Level | RDNA 4 | Source |
|---|---|---|
| L0 vector cache | 32 KB per CU (CC-RT says "The L0 is 32 KB") | CC-RT 2025-04-14 |
| L1 (mid-level) | dropped. CC-RT: "RDNA 4 drops the mid-level L1 cache of prior RDNA generations". CC-VID: "the L1 cache no longer exists on RDNA4" and acts as a read/write-coalescing buffer | CC-RT, CC-VID |
| L2 | 8 MB on Navi 48 (4 SE); CC-VID: banks per slice doubled, L2 per SE doubled. STH-HC: "The L2 cache was enlarged this generation to better prep the architecture for RT workloads." | CC-VID, STH-HC |
| Infinity Cache | up to 64 MB, third generation (STH-HC: "This also marks the 3rd generation of AMD's Infinity Cache."). CC-RT: "Even an Infinity Cache hit adds 60-70 ns of extra latency over a L2 hit." | WCC-DD, STH-HC, CC-RT |
| Infinity Fabric | "up to 1KB/clock", range 1.5-2.5 GHz | STH-HC, WCC-HC |
| DRAM | GDDR6 up to 20 Gbps, 256-bit (Navi 48), 128-bit (Navi 44) | WCC-DD |

Fabric and compression (WCC-HC, AMD Hot Chips slide as reported): "AMD's RDNA 4 GPUs offer 15% performance improvement in some raster workloads, deliver 25% reduction in fabric bandwidth". STH-HC: "AMD has seen a ~25% reduction in fabric bandwidth usage... entirely transparent to software". INFERRED: this is lossless data compression on the fabric, applied on both Navi dies; Navi 44 coverage mentions "features to reduce memory bandwidth" (WCC-HC headline).
Navi 44 versus bandwidth: AMD answered a question about lower bandwidth per CU (WCC-HC): "memory bandwidth depends on the workloads, and they have done enough tuning" and on LPDDR: "while LPDDR offers lower power, it still doesn't provide sufficient bandwidth".
GDDR6 choice: AMD stayed on GDDR6 (20 Gbps) rather than GDDR7 for the whole generation; no AMD statement explaining why was found (INFERRED: cost and availability). RX 9070 XT bandwidth 640 GB/s is derived arithmetic (256 bit x 20 Gbps / 8).

---

## 4. Ray tracing hardware (RT IP 3.1 per CC-RT)

CC-RT (2025-04-14): RT IP 3.1 is "doubled intersection testing, oriented bounding boxes, primitive compression."

- Intersection engines: two per Ray Accelerator; 8 box or 2 triangle tests per cycle versus 4 and 1 on RDNA 2/3 (CC-RT says RDNA 2/3 "presumably" had one engine). AMD slide (WCC-DD): "2x box & triangle intersection units", "2x ray traversal performance compared to RDNA 3 at equal clock rates and bandwidth." Ray intersection rates are "doubled."
- BVH8: box nodes widen from 4 to 8 children. BVH memory "less than 60% versus RDNA 3" (WCC-DD). STH-HC adds the 2x RT claim: "allowing for ~2x RT performance versus RDNA 3" (combined effect of OoO memory, hardware transform and OBBs, per the STH summary).
- OBBs: node stores an index into "a table of 104 predefined matrices"; each matrix is 9 packed 6-bit indices into 26 FP32 values; about 800 bytes. Only one rotation per box node ("best-effort"). AMD slide: rotated boxes improve traversal "by 10%" (WCC-DD).
- Ray transform block: "RDNA 4 adds a ray transform block to the Ray Accelerator" for TLAS-to-BLAS transitions. CC-RT notes AMD does not tie it to OBBs.
- Primitive node: 128-byte compressed node, up to 8 triangle pairs or 16 unique vertices; box extents quantized to 12 bits.
- "Pack and Sort" lets BVH results be stored out of order and re-sorted (CC-VID).
- Instructions (CC-RT): `IMAGE_BVH8_INTERSECT_RAY` (one ray, one 8-wide node, drives both engines), `IMAGE_BVH_DUAL_INTERSECT_RAY` (a pair of 4-wide nodes, the "BVH4x2" path), `DS_BVH_STACK_PUSH8_POP1_RTN_B32` and `DS_BVH_STACK_PUSH8_POP2_RTN_B64`. Inherited: `IMAGE_BVH_INTERSECT_RAY`, `IMAGE_BVH64_INTERSECT_RAY`.
- Still inline RT: a compute thread manages traversal (RDNA 2 model); no full hardware traversal. CC-RT hedges this ("RDNA 4 still uses the RDNA 2-style inline model").
- Tooling: HIP RT 3.0.9 lists compressed BVH8, intersectable instances and OBBs for HQ builds (search summary of GPUOpen https://gpuopen.com/hiprt/, date unknown).

---

## 5. Matrix cores / "AI accelerators"

Generation by generation (INFERRED from sources plus memory, flagged):
- RDNA 3 (2022): WMMA instructions executed on the vector ALUs; AMD called these "AI accelerators", 2 per CU. Not a separate matrix engine (INFERRED/background, not sourced in this session).
- RDNA 4 (2025): dedicated "Matrix Acceleration Engines" (WCC-DD: 4 per DCU/WGP). Dense rates (WCC-DD): "2x-16b & 4x-8b/4b dense matrix rates." New FP8 (E4M3) and BF8 (E5M2): CC-VID says they run "at the same throughput as INT8". Sparsity: "4:2 Structured Sparsity for +2x rate" (WCC-DD). STH-HC: "RDNA 4 added FP8 capabilities, as well as structured sparsity." WMMA shape: M=N=K=16 except 4-bit (CC-1250). Transpose loads are new ("Matrix load w/transpose").
- gfx1250 (CDNA-class, 2026): K=32 for FP16, K=128 for FP8, scaled (OCP MX) formats, FP4/FP6 (CC-1250, see section 10).

Published AMD numbers:
| Claim | Value | Source / notes |
|---|---|---|
| RX 9070 XT peak matrix | "389 (1557)" in a TFLOPS FP16 (FP4/FP8) column | Tom's Hardware spec table as relayed by search summary; Jon Peddie (search summary) says peak INT4 1557 TOPS and INT8 779 TOPS. Labels disagree |
| R9700 | "383TFLOPs of half precision (FP16 Matrix), 766TFLOPs of 8-bit precision (FP8 Matrix), and 1,531TOPs of 4-bit precision (INT4 Matrix)" | Club386 2025-10-25. PCGH 2025-05-21 labels 1,531 as "Sparsity". Wccftech (2025-05-20) lists 96 TFLOPs FP16 and 1531 INT4 sparse, 48 FP32 |
| Per CU vs RDNA 2 | 3.5x FP16 dense ML | WCC-DD |
| Image generation | "2x boost in image generation performance (SDXL 1.5)" vs RDNA 3, normalized, FP16 | WCC-DD, no test notes obtained |
| R9700 LLM | "up to 496% faster in handling large AI models compared to Nvidia's GeForce RTX 5080" | Club386; "in-house benchmarks", no model or footnote in text. Wccftech: "up to 5x faster" than RTX 5080 16 GB (VRAM limit), and 2x PRO W7800 32 GB on DeepSeek R1 |
| Models AMD named | DeepSeek R1 Distill Qwen 32B Q6, Mistral Small 3.1 24B Q8, Flux 1 Schnell, SD 3.5 Medium; 4-GPU PCIe 5.0 pool of 128 GB | Wccftech 2025-05-20 |

INFERRED arithmetic: 389 TFLOPS at 2.97 GHz over 64 CU = 2048 ops per CU per clock; 1557 is 4x of the 389... no: 1557/389 = 4.0, so the 1557 column is 4x, consistent with the numbers being dense FP16 (389?) versus INT4 sparse. Because AMD's own slide claims "2x-16b" over RDNA 3 (about 512/CU/clk dense), the dense FP16 figure would be about 1024/CU/clk = about 195 TFLOPS, making 389 the sparse FP16 or dense FP8 number. I could not resolve this from sources; treat precision labels as unreliable. KitGuru measured SDXL 7 percent behind RTX 5070 Ti and Llama 3.1 "almost 50% slower" (2025-03-05, page 27, runtime unnamed).

Software use (READ):
- FSR 4 (ML upscaler, FP8) shipped March 2025, RDNA 4 only at first. FSR Redstone, hwcooling.net 2025-12-10: components "FSR Frame Generation, FSR Ray Regeneration, and FSR Radiance Caching" plus FSR 4 upscaling; AMD's table lists ML features as RDNA 4 only. Radiance Caching to arrive in titles in 2026 (TechRadar via search summary). Adrenalin 25.12.1 (search summary).
- FSR 4.1 on RX 7000 using an INT8 model: Tom's Hardware, Club386 and HWBusters (all via search summary, 2026): "FP8 instructions ... are missing from the older hardware, so AMD rebuilt the model around INT8". Club386: arrived 2026-06-22. RDNA 2 in 2027 per AMD announcement. This supports the repo's own commit message; I could not open these pages to confirm the date.
- Amuse and LM Studio: AMD named LM Studio/Amuse in launch material per my background knowledge; no source read in this session. NOT VERIFIED.

---

## 6. Media, display and other blocks

| Block | RDNA 4 | Source |
|---|---|---|
| Media engines | two on Navi 48 ("AMD has two such media engines in the Navi 48 GPU", STH-HC) | STH-HC |
| Encode | AV1 B-frames added, lower latency; AMD slide: "25% gain in H.264 low latency encode quality", "11% HEVC encode quality improvement", up to 30% encode boost at 720p | STH-HC; WCC-DD |
| Decode / playback | "VCN low power video playback (50% performance uplift for AV1 & VP9)" | WCC-DD |
| Display outputs | DisplayPort 2.1a, HDMI 2.1b | WCC-DD |
| Display engine | Radeon Image Sharpening 2 integrated into the display block, "rather than processing it as a shader effect" | STH-HC |
| Bus | PCIe 5.0 x16 | WCC-DD ("PCIe Gen5 compliant") |
| Power modes | FreeSync power modes | WCC-DD |

DP 2.1a UHBR rate (UHBR13.5 versus UHBR20) was not stated in what I read (NOT VERIFIED).

---

## 7. Manufacturing and physical design

- Process: TSMC N4P per Chips and Cheese (CC-VID, 2025-03-05): "TSMC N4P, 53.9 billion transistors, 356.5 mm2". A search summary of TechPowerUp's deep dive says N4C in one place and N4P in another (CONFLICT; TPU pages blocked). WCC-DD: "TSMC 4nm".
- Density (computed): 53.9 B / 356.5 mm2 = 151 M per mm2. Tom's Hardware headline: "Navi 48 is 25% denser than Nvidia Blackwell GPUs" (headline only; body unreadable). Search summary: GB203 45.6 B, about 120 M per mm2.
- Versus RDNA 3: Navi 31 is a chiplet design with a 5 nm GCD and 6 nm MCDs; Navi 48 is monolithic (TechPowerUp via search summary: "traditional monolithic die"). RDNA 3 Navi 32/33 density figures were not sourced here. Navi 48 early leak estimated 390 mm2 (Tom's Hardware, Jan 2025, an overestimate).
- Navi 44: about 199 mm2 (search summary).
- Power: 9070 XT 304 W, 9070 220 W, 9060 XT 150-182 W, R9700 300 W. HotHardware (search summary): AMD "pushed ... frequencies and voltages fairly aggressively". KitGuru measured 311 W average (12 games, 1080p, ASRock Taichi, PCAT) and under 450 W wall total at 4K. Voltages not found.

---

## 8. Software platform facts

Linux and ROCm (READ):
- ROCm 6.4.0 did not support RDNA 4 (Tom's Hardware headline, "AMD's latest ROCm 6.4 release continues to lack RDNA 4 support"; PCGH).
- ROCm 6.4.2 docs: Radeon Software for Linux 25.10.2.1 with ROCm 6.4.2 "expanded ML support to RDNA 4" for 9070, 9070 XT, 9070 GRE (search summary of AMD docs).
- Radeon/Ryzen docs: "ROCm 7.2.1 supports the latest Radeon 9000 Series (RDNA 4)" (undated).
- ROCm 7.14.0 table, dated 2026-07-15 on AMD pages: gfx1201 = RX 9070 XT, 9070 GRE, 9070, Radeon AI PRO R9700, R9600D; gfx1200 = RX 9060 XT LP, 9060 XT, 9060; footnote OS list Ubuntu 24.04.4, 22.04.5, RHEL 10.1, 9.7. RX 7000 (gfx1100/1101) is also listed.
- ROCm 10.1.0 (release notes 2026-10-05; matrix 2026-08-26): adds RX 9050 (gfx1200) and "support for AMD Radeon AI PRO R9600 GPUs"; build system TheRock since 7.14; PyTorch 2.14.0; Windows 11 25H2 listed; Adrenalin 26.10.41.05. Table text was garbled in fetch.
- Mesa/RADV Vulkan: RDNA 4 launched with open-source Linux support at launch (Notebookcheck headline "quasi perfect Linux support on launch day", search summary only).

Windows:
- Radeon/Ryzen docs: "Radeon GPUs (9000 & select 7000 Series) | Windows | PyTorch"; "PyTorch on Windows updated with ROCm 7.2.1". HIP SDK for Windows page not found, so RDNA 4 HIP SDK status is NOT VERIFIED. DirectML, ONNX and WinML recommendations were not found; ONNX Runtime with MIGraphX appears under Linux only.
- Adrenalin 25.12.1 brought Redstone (search summary, 2025-12). FSR 4.1 INT8 RX 7000 2026-06-22.

Graphics APIs: nothing RDNA 4-unique beyond hardware RT features was sourced; Vulkan/DX12 feature lists NOT VERIFIED.

---

## 9. Known weaknesses and criticisms

- Memory bandwidth per CU: 640 GB/s over 64 CU = 10 GB/s per CU (computed). Compared with an RX 7900 XTX (about 960 GB/s over 96 CU = 10 GB/s per CU; unsourced background). No reviewer text with this ratio was found; the claim is INFERRED and not a stated criticism.
- LLM decoding: KitGuru "for Llama 3.1 performance the 9070 XT is almost 50% slower" than the RTX 5070 Ti (2025-03-05). Community: Vulkan beats ROCm on token generation (about 20% on an 8B model, roughly 77-79% of theoretical bandwidth versus 63-72% on ROCm, one blog); llama.cpp discussion #15021 holds detailed numbers. Conflicting bandwidth quoted at 608 GB/s by one guide (likely wrong).
- Power: HotHardware and Tom's Hardware (search summary): the 9070 XT draws more than the RTX 5070 Ti.
- Idle power: not a clear criticism; PCGH says idle power improved over earlier Radeons; multi-monitor idle for RDNA 4 was not found (the hwcooling.net multi-monitor article measured an RX 6900 XT).
- RT: still inline traversal; OBBs best-effort one rotation per node (CC-RT). L1 dropped, so CC-RT suggests "first level caches could be larger".
- VRAM: only 16 GB on the flagship (R9700 doubles it).
- Dynamic VGPR limited to wave32 compute (CC-DYN).
- Software: ROCm did not ship RDNA 4 support at launch (see section 8). FSR 4 RDNA 4-only for 15 months, then INT8 port.
- Dual-issue: not improved (CC-VID).

---

## 10. What comes next

- gfx1250 / gfx1251 (LLVM "GFX12.5"): CC-1250 (2026-07-19, reading LLVM commits and an emulator config): "GFX1250 is an accelerator chip oriented at the machine learning market called MI455X." gfx1251 is the HPC part (MI430X). Phoronix (search summary, 2026): "AMD GFX1250 IP is what's used by the new Instinct MI450 series". So gfx12.5 is Instinct, not a consumer RDNA 4 refresh. The LLVM docs list it under "GCN GFX12 (RDNA 4)" (model-extracted); CONFLICT in naming only.
- What gfx1250 has that RDNA 4 lacks (CC-1250, READ unless noted):
  - Wave32 only: "GFX1250 operates only in Wave32 mode".
  - 20 waves per SIMD ("four more than RDNA4", hedged), up to 1024 VGPRs per wave (RDNA 4: 256).
  - Dynamic VGPR allocation is dropped: "The dynamic VGPR allocation feature that was heavily touted in RDNA4 is missing."
  - LDS up to 320 KB per workgroup (RDNA: 64 KB); a single 448 KB structure merges LDS and vector L0.
  - WMMA K=128 for FP8, K=32 for FP16; OCP MX scaling "previously available solely on CDNA4"; FP64 only on gfx1251; transpose loads incl. FP4/FP6 LDS transpose.
  - Clusters and cluster barriers (Hopper-style), Tensor Data Mover ("loads data from global memory into LDS without using the VGPRS"), 16 named barrier objects, monitors on L2 lines, 128B cooperative atomics, explicit vector prefetch, new counters ASYNCcnt, TENSORcnt, Xcnt.
  - 57-bit VA (LLVM PR #152373, 2025-08-06: "gfx1250 has fixed GETPC bug and also extended VA to 57 bits").
  - Dropped (like CDNA): no export instructions or rasterizer, no image/texture instructions, no BVH instructions, no MUBUF/MTBUF.
- gfx1310 / GFX13 (RDNA 5 or UDNA): LLVM 23 gets initial GFX13 target; Phoronix headline "presumably RDNA5" (search summary, date not captured; AMDGPU kernel patches also posted for GFX 13.0.1 and 13.0.4). A January 2026 LLVM change added FeatureGFX13 (PR #177567). Commit 5e4f44041d "Clean up GFX13 feature list (#208416)", 2026-07-16: "GFX13 features were previously mostly inherited from GFX12/GFX1250" and removes F4/F6 transpose loads copied from gfx12. LLVM docs list gfx1310 as "GCN GFX13 (RDNA 5)" (model-extracted, one search summary says Igor's Lab confirms labeling). Features per LLVM docs: cumode, wavefrontsize64, architected flat scratch, packed work-item IDs (so GFX13 keeps wave64, unlike gfx1250). Wccftech: AMD has not confirmed the marketing name; RDNA 5 GPUs "not likely before 2027" (search summary).
- UDNA: no AMD primary source for the name was read in this session. NOT VERIFIED.

---

## Could not verify

- TechPowerUp RDNA 4 deep dive and R9700S page, Phoronix (gfx13, gfx1250), HotHardware, Guru3D, TweakTown: HTTP 403, not read. Facts attributed to them come only from search-result summaries.
- Tom's Hardware pages (Hot Chips two-dies, density, spec table, FSR Redstone, RX 9070 review): only headlines returned.
- AMD's own pages (product pages for RX 9070 XT, R9700, R9700S, R9600D, RX 9050; GPUOpen FSR 4/Redstone; HIP SDK for Windows; RDNA 4 ISA guide): timed out or not found. R9700 AMD footnotes not obtained.
- Chips and Cheese "Examining AMD's RDNA 4 Changes in LLVM": not located. No Chips and Cheese RDNA 4 cache-latency article found.
- RDNA 4 laptop GPUs; Strix Halo status; Windows HIP SDK RDNA 4; DirectML/ONNX recommendations; Vulkan/DX12 unique features; instruction and constant cache sizes; L0 per-CU breakdown of "2 MB aggregate"; voltages; Amuse and LM Studio claims; UDNA statements; ECC and ISV status of R9700; DP 2.1a link rate; exact date Phoronix reported GFX13 and gfx1171/1172.
- FSR 4.1 RX 7000 date (2026-06-22) is from Club386 via search summary only.

## Conflicts

1. Process node: TechPowerUp (via search summary) says N4P in review but N4C in the deep dive; Chips and Cheese says N4P; Wccftech says "4nm". Use N4P.
2. 1557 / 389 / 1531 / 383 matrix figures: precision labels differ (Jon Peddie: INT4 1557, INT8 779; Tom's table: FP4/FP8 column; Club386: 383 FP16, 766 FP8, 1531 INT4; PCGH: 1531 "Sparsity"; Wccftech: 96 TFLOPS FP16). Dense versus sparse unresolved.
3. R9700 availability: PCGH/Wccftech (May 2025) said July 2025; Club386 (2025-10-25) says retail 2025-10-27 after supply constraints and OEM-only sales.
4. RX 9070 GRE: hwcooling.net says China-only at launch (May 2025); NanoReview lists 2025-08-27 and $549; PCGH says "13 months after China" it reached western markets (about mid-2026). Wccftech table lists $549. Western launch date unresolved.
5. RX 9060 XT TBP: 150-182 W (Wccftech text) vs 142-182 W (table); 150 W for 8 GB and 160 W for 16 GB (Yahoo/Tom's summary).
6. RX 9060 launch: Wccftech table says "2026"; another database says 2025-08-05 OEM-only.
7. Named barriers: AMD slide lists them; CC-VID says "Named Barriers are not a part of RDNA4"; CC-1250 says RDNA 4 already has "split barriers and barrier objects".
8. Naming of gfx1250: LLVM docs headings (model-extracted) put it under "RDNA 4"; Chips and Cheese and Phoronix say it is Instinct MI450/MI455X hardware.
9. Per-CU gain baseline: WCC-DD quotes AMD vs RDNA 2 (2x raster, 2.5x RT, 3.5x ML); another outlet states "up to 40%" per CU versus RDNA 3 (baseline not named).
10. Memory bandwidth of 9070 XT: most sources 640 GB/s; one guide 608 GB/s (derived 640 is correct for 20 Gbps).
11. Pre-launch Navi 48 size estimate 390 mm2 vs official 356.5 mm2.
