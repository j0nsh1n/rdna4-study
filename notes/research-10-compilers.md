> Provenance: written by a Sonnet helper from LLVM main and Mesa main (2026-11-16/17). Checked by the lead against
> the sources: the matrix-instruction hazard rule and the absence of matrix timings in `GFX12SpeedModel` (part 1,
> section 6), the mapping of non-temporal loads to `TH_NT` in LLVM, the wait-counter sizes (they agree with the
> guide's counter widths in guide digest A), and Mesa's comment and clamp for saturating FP8 conversion
> (`aco_select_nir_alu.cpp`: "the ISA doc is wrong and Inf isn't clamped to max_float"). The rest is the helper's
> reading; no commit dates were collected for the errata.

# What LLVM and Mesa ACO encode about RDNA 4 (gfx1200 / gfx1201, "GFX12")

Research draft, source-reading only. No GPU was touched.

## Versions read

- LLVM: github.com/llvm/llvm-project, branch `main`, read 2026-11-16/17 (head at feed time: commit 32d3230be28e, feed stamped 2026-11-17 02:50 UTC; the file `GCNHazardRecognizer.cpp` last changed 2026-11-16 19:43 UTC). Everything below is `llvm/lib/Target/AMDGPU/...` unless noted. This tree already has gfx1250, gfx1251 and a gfx1310 ("GFX13") target.
- Mesa: gitlab.freedesktop.org/mesa/mesa, branch `main`, head commit e8f491e8, 2026-11-17 02:40 UTC. The GitHub mirror named in the task (`Mesa3D/mesa`, `mesa3d/mesa`) now returns 404 for every file; the GitLab raw-file URLs answered normally (HTTP 200, real content), so I used those. Mesa also has a `GFX12_1` level (gfx1210) and `GFX13`; this report is about plain `GFX12` (gfx1200/gfx1201).
- LLVM's `llvm/docs/AMDGPUUsage.rst` no longer exists at that path (404). I read the rendered page https://llvm.org/docs/AMDGPUUsage.html instead (plain page fetch, converted to text in memory).

Labels: READ = I saw the line (file and short quote given). INFERRED = my reading of what it implies.

Naming: gfx1200 = RX 9060 class, gfx1201 = RX 9070 class (LLVM docs table lists "Radeon RX 9060 / 9060 XT" under gfx1200 and "RX 9070 / 9070 XT / 9070 GRE" under gfx1201; Mesa `amd_family.h`: `CHIP_GFX1200, /* Radeon 9060 */`, `CHIP_GFX1201, /* Radeon 9070 */`). Neither compiler uses the names Navi 44 / Navi 48 in the code I read, and neither has a single rule that differs between gfx1200 and gfx1201 (both map to the same feature set `FeatureISAVersion12`; Mesa's `identify_chip(GFX1200); identify_chip(GFX1201);` share one `GFX12` level).

---

## 1. LLVM feature list for gfx1200 / gfx1201 (FeatureISAVersion12)

READ: `GCNProcessors.td`:
`def GFX1200 : AMDGPUProcessorModel<"gfx1200", GFX12SpeedModel, FeatureISAVersion12.Features, [12, 0, 0]>` and the same for GFX1201 with `[12, 0, 1]`. `gfx12-generic` = `FeatureISAVersion12_Generic` = the same list plus `FeatureRequiresCOV6` (generic targets need code object v6), covering gfx1200 and gfx1201.

Description strings are the second string of each `defm X : AMDGPUSubtargetFeature<"name", "description">` in `GCNFeatures.td`. Entries with only a number are `SubtargetFeature...<N>` helper classes with no text.

### Generation-wide features pulled in by `FeatureGFX12` (READ, `GCNFeatures.td`)
The `def FeatureGFX12 : GCNSubtargetFeatureGeneration<"GFX12", "gfx12", [...]>` list contains, among others: FP64, MIMG_R128, SupportsWave32, FlatAddressSpace, 16BitInsts, Inv2PiInlineImm, ApertureRegs, GetDoorbellID, GFX8..GFX11Insts and GFX12Insts, VOP3PInsts, **VOPDInsts**, Movrel, FastFMAF32, DPP, DPP8, IntClamp, FlatInstOffsets, FlatGlobalInsts, FlatScratchInsts, AddNoCarryInsts, FmaMixInsts, NoSdstCMPX, Vscnt, VOP3Literal, Permlane16Insts, **NoDataDepHazard**, PkFmacF16Inst, A16, FastDenormalF32, G16, UnalignedBufferAccess, UnalignedScratchAccess, UnalignedDSAccess, True16BitInsts, DefaultComponentBroadcast, MaxHardClauseLength32, AtomicFMinFMaxF32Global/FlatInsts, AtomicDsCondSubClampInsts, AtomicCondSubClampFlatInsts, IEEEMinimumMaximumInsts, SALUMinimumMaximumInsts, Minimum3Maximum3F32/F16, AgentScopeFineGrainedRemoteMemoryAtomics, FlatOffsetBits24, FlatSignedOffset, InstCacheLineSize128, **MaxWavesPerEU16**.

### Per-chip list (`FeatureISAVersion12`), grouped by theme

Wave and memory geometry:

| Feature | Description / value |
|---|---|
| SupportsWave64 | "Hardware supports wave64 mode" |
| SupportsWGP | "Hardware supports WGP mode" |
| BackOffBarrier | "Hardware supports backing off s_barrier if an exception occurs" |
| 1536VGPRs | "Has 1536 physical VGPRs per SIMD" |
| AddressableLocalMemorySize65536 | 65536 bytes of LDS per workgroup |
| LDSAllocGranularity1024 / LDSEncodingGranularity512 | allocation in 1024 B steps; descriptor encoded in 512 B steps |
| HalfAddressablePhysicalLocalMemory | "A work-group can address only half of the physical LDS block" (so the WGP has 128 KB, a workgroup sees 64 KB) |
| LDSBankCount32 | 32 LDS banks |
| MaxHardClauseLength32 | s_clause up to 32 instructions |
| DataCacheLineSize128 | 128 B data cache line (generation also sets InstCacheLineSize128) |
| 32BitNumRecordsBufferResource | buffer descriptor num_records is 32 bit |
| RestrictedSOffset | "Has restricted SOffset (immediate not supported)." |
| ArchitectedFlatScratch | "Flat Scratch register is a readonly SPI initialized architected register" |
| ArchitectedSGPRs | "Enable the architected SGPRs" |
| PackedTID | "Workitem IDs are packed into v0 at kernel launch" |
| WaitsBeforeSystemScopeStores | "Target requires waits for loads and atomics before system scope stores" |
| MemoryAtomicFAddF32DenormalSupport | "global/flat/buffer atomic fadd for float supports denormal handling" |
| ShaderCyclesHiLoRegisters | "Has SHADER_CYCLES_HI/LO hardware registers" |

Arithmetic and ALU:

| Feature | Description |
|---|---|
| SALUFloatInsts | "Has SALU floating point instructions" |
| PseudoScalarTrans | "Has Pseudo Scalar Transcendental instructions" |
| NoF16PseudoScalarTransInlineConstants | "Inline constants are not supported for F16 pseudo scalar transcendentals" |
| IEEEMinimumMaximumInsts | "Has v_minimum/maximum_f16/f32/f64, v_minimummaximum/maximumminimum_f16/f32, v_pk_minmax/maxmin_num_f16, v_min/max_num_f16/f32/f64 and v_pk_min/max_num_f16 instructions" |
| DPPSrc1SGPR | "Support SGPR for Src1 of DPP instructions" |
| ScalarDwordx3Loads | "Has 96-bit scalar load instructions" |
| SmemPrefetchInsts | "Has s_prefetch_inst, s_prefetch_inst_pc_rel, s_prefetch_data and s_prefetch_data_pc_rel instructions" |
| RealTrue16Insts | "Use true 16-bit registers" |
| D16Writes32BitVgpr | "D16 instructions potentially have 32-bit data dependencies" |
| CubeInsts, LerpInst, SadInsts, QsadInsts, MsadInsts, MqsadInsts, MqsadPkInsts | "Has v_cube* / v_lerp_u8 / v_sad* / v_qsad* / v_msad_u8 / v_mqsad_u32_u8 / v_mqsad_pk_u16_u8 instructions" (these legacy ops are listed explicitly because gfx12 is the first gen where they are not implied by an older generation list) |
| CvtNormInsts, CvtPkNormVOP2Insts, CvtPkNormVOP3Insts | "Has v_cvt_norm* instructions", "Has v_cvt_pk_norm_*f32 instructions/Has v_cvt_pk_norm_*_f16 instructions" |
| VcmpxPermlaneHazard | "TODO: describe me" (the description string is literally a TODO) |

Dot, matrix and low precision:

| Feature | Description |
|---|---|
| DLInsts | "Has v_fmac_f32 and v_xnor_b32 instructions" |
| Dot7Insts | "Has v_dot4_u32_u8, v_dot8_u32_u4 instructions" |
| Dot8Insts | "Has v_dot4_i32_iu8, v_dot8_i32_iu4 instructions" |
| Dot9Insts | "Has v_dot2_f16_f16, v_dot2_bf16_bf16 instructions" |
| Dot10Insts | "Has v_dot2_f32_f16 instruction" |
| Dot11Insts | "Has v_dot4_f32_fp8_fp8, v_dot4_f32_fp8_bf8, v_dot4_f32_bf8_fp8, v_dot4_f32_bf8_bf8 instructions" |
| Dot12Insts | "Has v_dot2_f32_bf16 instructions" |
| FP8ConversionInsts | "Has fp8 and bf8 conversion instructions" |
| CvtFP8ByteSel | "FP8/BF8 conversions take the byte index from a byte_sel operand" |
| OCPFP8ConversionInsts | "Has OCP fp8 (E4M3FN, E5M2) conversion instructions" |
| WMMA128bInsts | "Has WMMA instructions where A and B matrices do not have duplicated data" |
| SWMMACGfx1200Insts | "Has GFX1200 SWMMAC instructions" (sparse WMMA) |

Atomics:

| Feature | Description |
|---|---|
| AtomicFaddRtnInsts / AtomicFaddNoRtnInsts | "Has buffer_atomic_add_f32 and global_atomic_add_f32 instructions that return / don't return original value" |
| FlatAtomicFaddF32Inst | "Has flat_atomic_add_f32 instruction" |
| AtomicDsPkAdd16Insts | "Has ds_pk_add_bf16, ds_pk_add_f16, ds_pk_add_rtn_bf16, ds_pk_add_rtn_f16 instructions" |
| AtomicFlatPkAdd16Insts | "Has flat_atomic_pk_add_f16 and flat_atomic_pk_add_bf16 instructions" |
| AtomicBufferGlobalPkAddF16Insts | "Has buffer_atomic_pk_add_f16 and global_atomic_pk_add_f16 instructions that can return original value" |
| AtomicGlobalPkAddBF16Inst / AtomicBufferPkAddBF16Inst | "Has global_atomic_pk_add_bf16 instruction" / "Has buffer_atomic_pk_add_bf16 instruction" |

Graphics-flavoured (listed for completeness): ImageInsts "Support image instructions", ExtendedImageInsts "Support mips != 0, lod != 0, gather4, and get_lod", NSAEncoding "Support NSA encoding for image instructions", PartialNSAEncoding, MSAALoadInsts "Has image_msaa_load instructions", BVHRayTracingInsts "Has BVH ray tracing instructions", BVHDualAndBVH8Insts "Has image_bvh_dual_intersect_ray and image_bvh8_intersect_ray instructions", MTBUFInsts "Has memory typed buffer instructions.", FormattedMUBUFInsts "Has formatted memory untyped buffer instructions."

### Difference from gfx1100 (`FeatureISAVersion11_0_0` = 11_Common + 11_0_Common + two)

READ (from `GCNFeatures.td`, set difference I computed):

- In gfx1100 but NOT in gfx12: `DX10ClampAndIEEEMode` ("Target has DX10_CLAMP and IEEE_MODE kernel descriptor bits"), `Dot5Insts` ("Has v_dot2c_f32_f16 instruction"), `WMMA256bInsts` ("Has WMMA instructions where A and B matrices have duplicated data"), `ShaderCyclesRegister` (gfx12 has the Hi/Lo pair instead), and five hardware-bug flags: `MSAALoadDstSelBug` ("MSAA loads not honoring dst_sel bug"), `VALUTransUseHazard` ("Hazard when TRANS instructions are closely followed by a use of the result"), `VOPDInterlockHazard` ("VOPD destination buffer forwarding drops the interlock when SRC0 or SRC1 ..."), `MADIntraFwdBug` ("MAD_U64/I64 intra instruction forwarding bug"), `PrivEnabledTrap2NopBug` ("Hardware that runs with PRIV=1 interpreting 's_trap 2' as a nop bug"), and `UserSGPRInit16Bug` ("Bug requiring at least 16 user+system SGPRs to be enabled").
- INFERRED: gfx12 fixed (or hid) every gfx11.0 data-hazard bug flag LLVM tracks, except `VcmpxPermlaneHazard`, which both generations keep. gfx12 also has no `RequiredExportPriority` (that flag is only in the gfx11 generic and 11.5 lists, described "Export priority must be explicitly manipulated on GFX11.5").
- In gfx12 but not gfx1100: wave64 explicit support, WGP support flag, 64 KB LDS addressing plus LDS granularities, architected SGPRs, the packed/bf16 atomics, BVH8/dual, FP8/BF8/OCP FP8 conversion, Dot11, WMMA128b, SWMMAC, IEEE minimum/maximum, SALU float, pseudo-scalar transcendentals, restricted SOffset, scalar dwordx3 loads, DPP src1 SGPR, hard clause 32, s_prefetch, WaitsBeforeSystemScopeStores, DataCacheLineSize128, plus the legacy-op flags (cube/lerp/sad/qsad/cvt-norm/mtbuf/formatted-mubuf), which are only bookkeeping.
- Same in both: 1536 VGPRs per SIMD, DL insts, Dot7..Dot10, Dot12, BackOffBarrier, 32 LDS banks, NSA, PackedTID, ArchitectedFlatScratch, D16Writes32BitVgpr, RealTrue16Insts, MemoryAtomicFAddF32DenormalSupport, VcmpxPermlaneHazard.

### What the newer gfx1250 list adds (what gfx12 lacks)

READ: `FeatureISAVersion12_50` has 129 entries and does NOT contain the gfx1200 graphics set (ImageInsts, NSA, BVH, MSAA, MTBUF, formatted MUBUF, Dot9..Dot12, WMMA128b, SWMMACGfx1200, 1536VGPRs, 32 bank LDS, 64 KB LDS). What gfx1250 adds that gfx1200 lacks (all `defm` descriptions from `GCNFeatures.td`):

- Memory model/ISA plumbing: `Clusters` ("clusters of workgroups"), `TDMInsts` ("tensor_load_to_lds/tensor_store_from_lds"), `AsyncLoadToLDSInsts`/`AsyncStoreFromLDSInsts`, `Asynccnt`, `WaitXcnt` ("s_wait_xcnt"), `LoadMonitorInsts`, `McastLoadInsts`, `VmemPrefInsts` ("flat_prefetch_b8 and global_prefetch_b8"), `ScaleOffset`, `64BitLiterals`, `VGPRMSBs` ("MSB registers to address VGPRs that don't fit in the instruction encodings", 1024 VGPRs), `GloballyAddressableScratch`, `KernargPreload`, `EmulatedSystemScopeAtomics` ("System scope atomics unsupported by the PCI-e are emulated in HW via CAS"), `NoMTypeNC`, larger LDS (`AddressableLocalMemorySize196608/327680`, LDS bank count 64).
- Arithmetic: `BitOp3Insts`, 64-bit VALU (`VMovB64Inst`, `VMulU64Inst`, `AddSubU64Insts`, `MinMaxI64Insts`, `LshlAddU64Inst`), `FullRate64Ops`, bf16 packed/trans/cvt, `TanhInsts`, `PrngInst`, `Permlane16Swap`, `WaveMatchInsts`, `ExclusiveScanInsts`, FP6/BF6/FP4 conversion-scale and WMMA F4 (`WMMAF4Insts`, `WMMA16x16x128FP8Insts`, `WMMAN16Insts`), `FP8E5M3Insts`, `SWMMACGfx1250Insts`, `SetPrioIncWgInst`, `SWakeupBarrier`, `LdsBarrierArriveAtomic`, `FlatBufferGlobalAtomicFaddF64Inst`.
- Its own hazards: `TransCoexecutionHazard` ("Hazard when TRANS co-executes with other VALU instruction"), `WMMACoexecutionHazards`, `VPermPk16Hazard`, `VOPD3F64OPYSrcHazard`, `VOP3PX2IncrementsVaVdstTwice`, `RequiresInitialUnclausedVmem`, `NeedsTDMDrain`, `McastFullMaskBug`, `SetregVGPRMSBFixup`, `RequiresAlignedVGPRs`, `GFX1250VALUBlockingCycles`, `SlowMaxMinMulI64Insts`.
- INFERRED: gfx120x has none of: clusters, tensor/async copies to LDS, 64-bit VALU moves/multiplies, FP6/FP4, bitop3, tanh, hardware prefetch of data into L2 (only the SMEM prefetches), more than 64 KB LDS per workgroup, more than 256 addressable VGPRs, or kernel-argument preload.

---

## 2. Hazards handled by LLVM that apply to gfx12

Entry point READ: `GCNHazardRecognizer::fixHazards` calls, in order: fixVMEMtoScalarWriteHazards, fixVcmpxPermlaneHazards, fixSMEMtoVectorWriteHazards, fixVcmpxExecWARHazard, fixLdsBranchVmemWARHazard, fixLdsDirectVALUHazard/VMEMHazard (if `hasLdsDirect()`), fixVALUPartialForwardingHazard, fixVALUTransUseHazard, fixVALUTransCoexecutionHazards, fixWMMAHazards, fixWMMACoexecutionHazards, fixShift64HighRegBug, fixVALUMaskWriteHazard, fixRequiredExportPriority, and gfx1250-only extras. Each function starts with a feature guard; the guard decides applicability.

Legend for "gfx12?": YES, NO, and the guard.

| Hazard (function) | gfx12? | gfx11? | Trigger | What LLVM inserts | LLVM's comment |
|---|---|---|---|---|---|
| fixVcmpxPermlaneHazards | YES (`hasVcmpxPermlaneHazard`, in the gfx12 list) | YES (gfx11 common list) | A v_permlane* after a VOPC/VOP3 compare that writes EXEC (v_cmpx), with no real VALU in between (V_NOP does not count) | `v_mov_b32 vX, vX` on the permlane's src0 | "V_NOP will be discarded by SQ. Use V_MOV_B32 v?, v?. Register must be alive so use src0 of V_PERMLANE*" |
| fixWMMAHazards | YES | YES | A WMMA/SWMMAC whose A or B source overlaps the destination of the previous WMMA/SWMMAC, with no VALU in between | one `V_NOP_e32` | "Src0(matrix A) or Src1(matrix B) of the current wmma instruction overlaps with the dest(matrix D) of the previous wmma." Also READ: "GFX12+ allows overlap of matrix C with PrevDstReg (hardware will stall) but Index can't overlap with PrevDstReg." (SWMMAC index operand, src2) |
| SGPR read hazard (file `AMDGPUWaitSGPRHazards.cpp`, `hasVALUReadSGPRHazard()` = `HasGFX12Insts && !HasGFX1250Insts`) | YES, gfx12 only (not gfx1250) | NO (gfx11 uses fixVALUMaskWriteHazard, wave64 only) | An SGPR that was read by a VALU is later written by a VALU or SALU, and then read again by a VALU/SALU. VCC is tracked separately | `s_waitcnt_depctr` (s_wait_alu) with the right field: `sa_sdst=0` after SALU writes, `va_sdst=0` after VALU writes, `va_vcc=0` for VCC. A forward dataflow pass with optional "culls" (a run of DS_NOPs: 4 in wave32, 8 in wave64) to reset tracking | File header: "Insert s_wait_alu instructions to mitigate SGPR read hazards on GFX12." SMEM/VMEM (non-flat) clear all hazards. At calls/returns/indirect branches every pending write is flushed |
| fixVALUMaskWriteHazard | NO (`hasVALUMaskWriteHazard` = generation == GFX11) | YES, wave64 only | VALU reads SGPR as lane mask, later written by SALU/VALU, then read | s_waitcnt_depctr | "The hazard sequence is three instructions: 1. VALU reads SGPR as mask 2. VALU/SALU writes SGPR 3. VALU/SALU reads SGPR" and "hazard can expire if the distance between 2 and 3 is sufficient, or (2) is VALU and (3) is SALU. In practice this happens <10% of the time" so it always assumes the hazard |
| fixVALUTransUseHazard | NO (`VALUTransUseHazard` is only in the gfx11.0/generic lists) | YES (not 11.5) | VALU uses a VGPR written by a transcendental with <=5 VALUs / <=1 TRANS between | `s_waitcnt_depctr` | "Va <- TRANS VALU ... intv <= 5 VALUs / 1 TRANS" |
| fixVALUPartialForwardingHazard | NO (`generation == GFX11`) | YES, wave64 | VALU partially forwarded operands after an EXEC write | `s_waitcnt_depctr` | pattern comment: Va VALU, SALU exec write, Vb VALU, user of both (intervals <=2 and <=4 VALUs) |
| fixLdsDirectVALUHazard / VMEMHazard | YES in code (`hasLdsDirect` = gen >= GFX11) but LDSDIR is a graphics (pixel shader) instruction | YES | LDSDIR (LDS_PARAM_LOAD / DIRECT_LOAD) writes a VGPR still in use by earlier VALU/VMEM/DS | VALU: sets the LDSDIR `waitvdst` field (15 wait states max). VMEM: on gfx12 sets the `waitvsrc` field (gfx11: an s_waitcnt_depctr) | `// TODO: On GFX12 the hazard should expire on S_WAIT_LOADCNT/SAMPLECNT/BVHCNT according to the type of VMEM instruction.` Not relevant for compute kernels |
| fixRequiredExportPriority | NO (feature not in gfx12 list; description "...must be explicitly manipulated on GFX11.5") | only 11.5 / gfx11-generic | exports at end of graphics shaders | `s_setprio` games | skips CS and kernel calling conventions anyway |
| fixShift64HighRegBug | NO (`HasGFX90AInsts`) | NO | 64-bit shift amount in last VGPR of an allocation block | - | "Shift amount of a 64 bit shift cannot be a highest allocated register if also at the end of the allocation block." |
| fixLdsBranchVmemWARHazard, fixVcmpxExecWARHazard, fixVMEMtoScalarWriteHazards, fixSMEMtoVectorWriteHazards | NO (each asserts `!hasExtendedWaitCounts()`, i.e. gfx10-only) | NO | - | - | READ the asserts; gfx12 always has extended counters |
| s_nop before `s_sendmsg dealloc_vgprs` (in `SIInsertWaitcnts.cpp`, `requiresNopBeforeDeallocVGPRs()` = `!HasGFX1250Insts`) | YES | YES | The wave-end message that frees VGPRs early (only emitted when the kernel is VGPR-limited in occupancy, or has calls) | `s_nop 0` immediately before it | `// returns true if the subtarget has a hazard requiring an "s_nop 0" instruction before "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)"` |
| Waits before system-scope stores (`SIMemoryLegalizer.cpp`, `WaitsBeforeSystemScopeStores`) | YES (gfx12.0 only) | NO | A store with SCOPE_SYS (volatile, or system-scope atomic ordering) | `s_wait_loadcnt 0`, `s_wait_samplecnt 0`, `s_wait_bvhcnt 0`, `s_wait_kmcnt 0`, `s_wait_storecnt 0` before it | "Stores with system scope (SCOPE_SYS) need to wait for: loads or atomics(returning)... non-returning-atomics - wait for STORECNT==0. There is no need to wait if memory is cached (mtype != UC)." and `// GFX12.0 only: Extra waits needed before system scope stores.` |

Also READ (not hazards, but on gfx12): LLVM's `hasExpertSchedulingMode()` = `getGeneration() >= GFX12`, doc: "supports expert scheduling mode 2 which relies on the compiler to insert waits to avoid hazards between VMEM and VALU instructions in some instances." Function attribute `amdgpu-expert-scheduling-mode`. In expert mode LLVM skips VA_VDST waits before VALU ("hardware handles VALU->VGPR->VALU hazards in expert scheduling mode"). INFERRED: whether gfx120x silicon actually supports the SCHED_MODE register is not proven by the `>= GFX12` predicate; the docs say "introduced in GFX12".

Mesa agreement is discussed in section 8.

---

## 3. Scheduling model and VOPD

### GFX12SpeedModel (READ, `SISchedule.td`, `let SchedModel = GFX12SpeedModel`)

The comment sits on the GFX11 block and applies by copy: "The latency values are 1 / (operations / cycle). Add 1 stall cycle for VGPR read." (so a full-rate VALU op is 1 issue cycle + 4 pipeline = 5.) The GFX12 block has no comment of its own.

| SchedWrite | GFX10 | GFX11 | GFX12 | Resources on GFX12 |
|---|---|---|---|---|
| Write32Bit | 5 | 5 | 5 | HWVALU, HWRC |
| WriteFloatCvt | 5 | 5 | 5 | HWVALU, HWRC |
| Write64Bit | 6 | 6 | 6 | HWVALU, HWRC |
| WriteTrans32 | 10 | 10 (HWTransVALU) | **9** | HWVALU, HWRC (no separate TRANS unit in the model) |
| WriteQuarterRate32 | 8 | 8 | **6** | HWVALU, HWRC |
| WriteFloatFMA | 5 | 5 | 5 | HWVALU, HWRC |
| WriteDouble / WriteDoubleAdd / WriteDoubleCvt | 22 | 38 | 38 | HWVALU, HWRC |
| WriteIntMul | 8 | 8 | 8 | HWVALU, HWRC |
| WriteTrans64 | 24 | 40 | 40 | HWVALU, HWRC |
| WritePseudoScalarTrans | unsupported | unsupported | **8** | HWVALU, HWRC |
| WriteSFPU | unsupported | 4 | 4 | HWSALU, HWRC |
| WriteSALU | 2 | 2 | 2 | HWSALU, HWRC |
| WriteBranch | 32 | 32 | 32 | HWBranch |
| WriteExport | 16 | 16 | 16 | HWExport, HWRC |
| WriteLDS | 20 | 20 | 20 | HWLGKM, HWRC |
| WriteSMEM | 20 | 20 | 20 | HWLGKM, HWRC |
| WriteVMEM | 320 | 320 | 320 | HWVMEM, HWRC |
| WriteBarrier | 2000 | 2000 | 2000 | HWBranch |
| WriteVALUDummy / WriteSALUDummy (zero micro-ops) | 5 / 2 | 5 / 2 | 5 / 2 | HWVALU / HWSALU |

All resources have `ProcResource<1>` with `BufferSize = 1`; the source comment above them reads `// XXX: Are the resource counts correct?`.

INFERRED:
- The gfx12 model differs from gfx11 in exactly three places: transcendental 9 (was 10), quarter-rate 6 (was 8), pseudo-scalar transcendentals exist (8). The doubles and branch/memory numbers are copied from gfx11 and are probably not measured for RDNA 4 (the 320 / 20 / 2000 numbers are round placeholders).
- There is no WMMA scheduling write in the GFX12 model (the WMMA `Write*PassWMMA` and `WriteXDL*` classes are only bound for the gfx125x models), so LLVM's scheduler treats gfx12 WMMA like an ordinary 5-cycle op.
- These numbers also feed `s_delay_alu` placement (`computeOperandLatency` in `AMDGPUInsertDelayAlu.cpp`), so they matter for generated stalls (section 6). gfx13 reuses `GFX12SpeedModel`.
- I did not find the definition line of `GFX12SpeedModel` itself (grep found only its `let SchedModel` uses and processor entries); its issue width and buffer settings are not reported here.

### VOPD (dual issue) legality on gfx12 (READ: `GCNVOPDUtils.cpp`, `Utils/AMDGPUBaseInfo.h/.cpp`)

- Pairing pass: `tryMatchVOPDPair` looks at two adjacent instructions; either can be OPX or OPY if the opcode tables allow it (`getCanBeVOPD`). A RAW dependency from first to second kills it (`hasRAWDependency`).
- Operand rules (`checkVOPDRegConstraints`): no DPP; at most one unique literal; `UniqueLiterals + UniqueScalarRegs <= 2` (comment: "Literals also count against scalar bus limit"; VCC read by v_cndmask counts as a scalar); mandatory literals (fmamk/fmaak) count.
- VGPR bank rules (`VOPD_VGPR_BANK_MASKS[] = {1, 3, 3, 1}` for dst, src0, src1, src2): the two destinations must have different parity (mask 1); src0 of X and src0 of Y must be in different banks of 4 (register number mod 4; mask 3); same for src1; src2 (the tied accumulator of fmamk/dot2acc) parity only. For 64-bit register operands both halves are checked. Different MSB groups also reject (`getVGPREncodingMSBs`, only matters on gfx1250).
- gfx12-specific, READ: `bool AllowSameVGPR = ST.hasGFX12Insts();` For gfx12 the bank check is waived if X and Y read the same VGPR in the same source slot (header comment: "If AllowSameVGPR is set then same VGPRs are allowed for X and Y sources even though it violates requirement to be from different banks"). The relaxation is only enabled when there is no anti-dependency: `AllowSameVGPR &= !IsAntiDep` and, with an anti-dependency, `if (IsAntiDep && !TII.isVOPDAntidependencyAllowed(SecondMI)) return std::nullopt;`
- gfx11-only extra: with `FeatureVOPDInterlockHazard` (gfx11.0, gfx11 generic) all four masks become 1 (comment: "GFX11 VOPD interlock hazard requires SRC0/SRC1 to have different parities, not just on different banks. Else, non-deterministic forwarding error may occur."). gfx12 does not have this flag, so it gets the looser 4-bank rule.
- `SkipSrc` (no source bank check at all) when both X and Y are `V_MOV_B32_e32` and `hasGFX11_7Insts() || hasGFX12Insts()` (comment: "On GFX1170+ if both OpX and OpY are V_MOV_B32 then OPY uses SRC2 source-cache").
- VOP3P dot2: `canMapVOP3PToVOPD` maps only `V_DOT2_F32_F16` / `V_DOT2_F32_BF16` with no modifiers, src2 == dst, no clamp (they become `v_dual_dot2acc_*`).
- VOPD3 is gfx1250 (`hasVOPD3`), not gfx12.

---

## 4. Memory model: SIGfx12CacheControl (READ, `SIMemoryLegalizer.cpp`)

Instruction cache-policy field (`CPol` in `SIDefines.h`, READ):
- TH (bits 0-2): `TH_RT=0` regular, `TH_NT=1` non-temporal, `TH_HT=2` high-temporal, `TH_LU=3` last use (loads) / `TH_WB=3` write-back (stores), `TH_NT_RT=4` "non-temporal (CU, SE), regular (MALL)", `TH_RT_NT=5` "regular (CU, SE), non-temporal (MALL)", `TH_NT_HT=6`, `TH_NT_WB=7`; `TH_BYPASS = 3 // only to be used with scope = 3`. Atomics: bit0 return, bit1 non-temporal, bit2 cascade.
- SCOPE (bits 3-4): `SCOPE_CU=0`, `SCOPE_SE=1`, `SCOPE_DEV=2`, `SCOPE_SYS=3`. Doc: "SCOPE_CU: Compute Unit (NOTE: not affected by CU/WGP mode)".
- Other bits: NV (5), SWZ (6), CFS (7-8, gfx1250), SCAL (9).

Per access kind (code of `enableVolatileAndOrNonTemporal`, `finalizeStore`, `setAtomicScope`):

| Access | What gets set |
|---|---|
| normal load/store | nothing (TH_RT, SCOPE_CU) |
| `nontemporal` (IR metadata) | TH = `TH_NT` ("Set non-temporal hint for all cache levels"); a load shows in the asm as `th:TH_LOAD_NT`, a store `th:TH_STORE_NT` (docs table) |
| last use (IR `!amdgpu.last.use`) | TH = `TH_LU` for loads; "Takes priority over nontemporal hint" |
| volatile load/store | SCOPE = `SCOPE_SYS`, then an `s_wait_loadcnt 0` (loads) or `s_wait_storecnt 0` (stores) after the access, "to cause all volatile operations to be visible outside the program in a global order"; docs: "Must happen before any following volatile global/generic load/store." For a volatile store, because of `WaitsBeforeSystemScopeStores`, the five waits listed in section 2 are first placed before it |
| atomic load/store/RMW, per syncscope | `setAtomicScope`: system -> `SCOPE_SYS`; agent -> `SCOPE_DEV`; (cluster -> SCOPE_SE); workgroup -> `SCOPE_SE` in WGP mode, nothing in CU mode; wavefront/singlethread -> nothing. LDS (`ds_*`) ops have no scope |
| atomic returning | `TH_ATOMIC_RETURN` (docs: "If atomic with return, use th:TH_ATOMIC_RETURN") |
| scratch / private | never needs a cache bypass or invalidate (comment: "all memory operations by the same thread are sequentially consistent, and no other thread can access scratch memory") |

Waits (`insertWait`): global/scratch at system, agent or cluster scope: `s_wait_loadcnt` for loads, `s_wait_storecnt` for stores; for acquire the sampler/BVH counters are waited first unless `AtomicsOnly`; workgroup scope: waits only if WGP mode or release-or-stronger ("Note that we still need to wait when performing a release in this mode to respect the transitivity of happens-before"), because "In WGP mode the waves of a work-group can be executing on either CU of the WGP... the L0 is per CU"; wavefront: "The L0 cache keeps all memory operations in order". LDS: `s_wait_dscnt 0` only when cross-address-space ordering is requested ("LDS operations for all waves are executed in a total global ordering").

Invalidate (`insertAcquire`, emits `global_inv scope:X`): system -> SCOPE_SYS; agent -> SCOPE_DEV; workgroup -> SCOPE_SE in WGP mode only ("In WGP mode ... we need to invalidate the L0 which is per CU. Otherwise in CU mode all waves of a work-group are on the same CU, and so the L0 does not need to be invalidated."); wave/thread: nothing. Not emitted if the `InsertCacheInv` option is off.

Write-back (`insertWriteback`, emits `global_wb scope:X`): only for system scope on gfx12.0 ("global_wb is only necessary at system scope for GFX12.0. They're also necessary at device scope for GFX12.5 as stores cannot report completion earlier than L2... Emitting it for lower scopes is a slow no-op, so we omit it for performance."). Docs: "global_wb can be omitted for scopes other than SCOPE_SYS in gfx120x."

Docs memory-model text (READ, AMDGPUUsage, "Memory Model GFX12"): per-CU L0 vector cache; scalar L0 per WGP, not coherent with vector L0; L1 buffer per shader array "with independent quadrants"; L2 per agent; MALL for GPU memory. Completion order: "Completion of a vector memory operation is reported to a wavefront in-order within a type, but may be out of order between types" (types: LDS s_wait_dscnt, loads s_wait_loadcnt, stores s_wait_storecnt, sample/gather4 s_wait_samplecnt, BVH s_wait_bvhcnt). "global_inv invalidates caches whose scope is strictly smaller than the instruction's." Scalar memory only touches data "proven to not change during the execution of the kernel dispatch". Scratch backing memory is MTYPE NC; kernarg on dGPU is MTYPE UC.

Syncscope table (docs "Instruction Scopes", CU mode / WGP mode): none/system -> SYS/SYS; agent -> DEV/DEV; workgroup -> none / SE; wavefront and singlethread -> none; the `one-as` variants map the same way. The docs also keep a stale line: "TODO: Currently the compiler does not support WGP mode on gfx12+. Hence, 'workgroup' scope currently maps to CU scope (no bits). When WGP mode is enabled this should map to scope:SCOPE_SE." (the code does implement the WGP-mode branch, so the TODO text and the table/code disagree.)

How a programmer reaches each hint:
- `__builtin_nontemporal_load` / `__builtin_nontemporal_store` (Clang) -> `!nontemporal` metadata -> `TH_NT` (value 1 = "NT for all levels"). INFERRED from the metadata name (clang side not read).
- `volatile` -> SCOPE_SYS plus waits. Atomics with `__ATOMIC_*` and HIP scopes (`__HIP_MEMORY_SCOPE_*` / syncscope strings) -> scope bits as above.
- `!amdgpu.last.use !{}` metadata on a load -> `TH_LU` (docs: `%val = load i32, ptr %in, align 4, !amdgpu.last.use !{}`). READ in `SIISelLowering.cpp`: `if (I.getMetadata("amdgpu.last.use")) Flags |= MOLastUse;`. INFERRED: I found no Clang builtin for it; it needs LLVM IR (or inline asm).
- Any TH/scope value, including `TH_HT`, `TH_NT_RT`, `TH_RT_NT`, `TH_NT_HT`, `TH_WB`/`NT_WB`, `SCOPE_SE`, and the swizzle bit, via the `aux` / cache-policy operand of the buffer builtins: READ `BuiltinsAMDGPU.td`: `__builtin_amdgcn_raw_buffer_load_b32(__amdgpu_buffer_rsrc_t, int, int, _Constant int)` (and b8/b16/b64/b96/b128, stores, format variants) and `IntrinsicsAMDGPU.td`: "auxiliary/cachepolicy(imm) ... gfx12+: bits [0-2] = th, bits [3-4] = scope, bit 6 = swz" and "Note: volatile bit is **not** permitted here." INFERRED consequence: those builtins are the only portable way to emit `th:TH_HT`/`TH_NT_HT`-style hints from HIP; plain pointer loads cannot ask for them.
- Not reachable from source (INFERRED): TH_BYPASS (3 with scope 3, documented as MALL bypass; Mesa warns not to use it in shaders), `TH_WB` stores except via buffer aux, CFS and NV bits (gfx1250 only).

---

## 5. Wait counters on gfx12 (READ, `SIInsertWaitcnts.cpp`, `Utils/AMDGPUBaseInfo.cpp`)

gfx12 splits the old `vmcnt` and `lgkmcnt`. `WaitcntGeneratorGFX12Plus` event map, in counter order: LOAD_CNT (VMEM reads, invalidates), DS_CNT (LDS/GDS), EXP_CNT (exports, GPR locks), STORE_CNT (VMEM writes, scratch writes), SAMPLE_CNT (sampler reads), BVH_CNT (BVH reads), KM_CNT (SMEM, `s_sendmsg`, SCC write events). X_CNT, ASYNC_CNT, TENSOR_CNT and the VA_VDST/VM_VSRC pseudo counters exist in the table but are gfx1250 or `s_wait_alu`-only.

Bit widths (the wait-immediate range; LLVM uses them as the counter limits via `HardwareLimits`):

| Counter | Instruction | Bits | Max value |
|---|---|---|---|
| LOAD_CNT | `s_wait_loadcnt` | 6 | 63 |
| SAMPLE_CNT | `s_wait_samplecnt` | 6 | 63 |
| BVH_CNT | `s_wait_bvhcnt` | 3 | 7 |
| DS_CNT | `s_wait_dscnt` | 6 | 63 |
| STORE_CNT | `s_wait_storecnt` | 6 | 63 |
| KM_CNT | `s_wait_kmcnt` | 5 | 31 |
| EXP_CNT | `s_wait_expcnt` | 3 | 7 |

Mesa agrees exactly (`aco_ir.cpp` `wait_imm::max`: `vm 63, exp 7, lgkm 63, vs 63, sample 63, bvh 7, km 31`). INFERRED: more outstanding operations than the max cannot be tracked precisely, so a wave can have at least 63 loads, 63 stores, 63 LDS ops, 63 sampler ops, 7 BVH ops and 31 scalar loads in flight before the compiler must start waiting; the hardware's own counter limits may be the same widths but I did not see a statement of that. For comparison (READ, same file): gfx11 has vmcnt 6 bits, lgkmcnt 6, expcnt 3, vscnt 6; gfx9/10 vmcnt is 6 bits via 4+2.

gfx12-specific rules READ:
- Combined waits exist: `s_wait_loadcnt_dscnt` and `s_wait_storecnt_dscnt` (`(vm << 8) | lgkm`); others are one instruction per counter ("For GFX12+, we use separate wait instructions, which makes expansion ...").
- `counterOutOfOrder`: SMEM "can go out of order" on its counter, so only waiting to 0 is exact; LOAD_CNT is in order unless several event kinds are pending (inv/sample/bvh mixed counts as out of order); GLOBAL_INV/BUFFER_INV "complete in-order with VMEM_READ_ACCESS".
- Because loads and samplers have separate counters, in-order register-write assumptions differ. Mesa comment: "Before GFX12, they also write VGPRs in order if they're of the same type. We can do this for GFX12 and different types for GFX11 if we know that the two VMEM loads do not write the same register half or the same lanes." (so on gfx12 two loads to the same VGPR may complete out of order even within a type; ACO therefore does not skip waits for same-type pairs on GFX12: `(event_matches && type_matches && ctx.gfx_level < GFX12)`).
- LLVM loop heuristics are GFX12-only: "DS optimization only applies to GFX12+ where DS_CNT is separate" (preheader flush of DS reads, prefetch pattern with "flush points"). `amdgpu-expand-waitcnt-profiling` splits multi-counter waits for PC sampling.
- Mesa: `mem_ordered` config bit is set only for `gfx_level >= GFX10 && < GFX12`.

---

## 6. s_delay_alu, kernel descriptor, dynamic VGPR, limits, docs

### S_DELAY_ALU (READ, `AMDGPUInsertDelayAlu.cpp`; `hasDelayAlu` = `HasGFX11Insts`, so gfx11 and gfx12 identical)

- Producer classes: VALU, TRANS (non-F64 transcendentals; on gfx1250 also XDL WMMA), SALU, OTHER.
- Encodable limits: `VALU_MAX = 5` (can wait for 1..4 VALUs back), `TRANS_MAX = 4` (1..3), `SALU_CYCLES_MAX = 4` (1..3 cycles; "Guard against pseudo-instructions like SI_CALL which are marked as SALU but with a very high latency").
- Latency source: `SchedModel->computeOperandLatency` (so GFX12SpeedModel numbers above); issue time per instruction from `getNumWaitStates`. TODOs in code: "Use a more advanced model that accounts for instructions that take multiple cycles to issue" and "In wave64 mode, double the number of cycles for VALU and VMEM instructions".
- Instructions that already wait for VALU (VA_VDST==0): DS, EXP, FLAT, MIMG, buffer, `s_sendmsg_rtn_*`, `s_waitcnt_depctr va_vdst=0`. Instructions that wait for SGPR writes (VA_SDST==0): SMEM, and SALU with a register operand.
- Mesa's header comment explains why it exists: "On GFX11+ the SIMD frontend doesn't switch to issuing instructions from a different wave if there is an ALU stall. Hence we have an instruction (s_delay_alu) to signal that we should switch to a different wave... This seems to apply only for ALU->ALU dependencies... if we do not emit s_delay_alu things will still be correct, but the wave will stall in the ALU (and the ALU will be doing nothing else)." Mesa constants: `valu_nop = 5`, `trans_nop = 4` (same as LLVM).

### Kernel descriptor / program info (READ, AMDGPUUsage "compute_pgm_rsrc1/3")

- `compute_pgm_rsrc1`: VGPR granule field is "GFX10-GFX12 wave32: max(0, ceil(vgprs_used / 8) - 1)", wave64 `ceil(vgprs_used/4)`; max 256 VGPRs per wave per the table (gfx12.5 uses 16-granules up to 1024). Note the code (`getVGPRAllocGranule`) says 1536-VGPR targets allocate in **24** (wave32) or **12** (wave64) register granules; the docs table text for rsrc1 is the older 8/4 description (INFERRED: doc not updated for the 1536-VGPR change; the encoding granularity is `getVGPREncodingGranule` = 8 / 4).
- SGPRs: "GFX10-GFX12: Reserved, must be 0. (128 SGPRs always allocated.)"
- Bit 21: on gfx9-gfx11 `ENABLE_DX10_CLAMP`; on gfx12 it is `WG_RR_EN` ("If 1, wavefronts are scheduled in a round-robin fashion with respect to the other wavefronts of the SIMD. Otherwise, wavefronts are scheduled in oldest age order."). Bit 23: IEEE mode on gfx9-11, "Reserved. Must be 0" on gfx12 (matches `DX10ClampAndIEEEMode` absent from the gfx12 feature list: gfx12 has no DX10_CLAMP/IEEE_MODE bits; INFERRED: IEEE behaviour is fixed on).
- `compute_pgm_rsrc3` for gfx12 (rows marked GFX120* are reserved on gfx1200/1201): bits 11:4 `INST_PREF_SIZE` ("Number of instruction bytes to prefetch, starting at the kernel's entry point instruction, before wavefront starts execution. The value is 0..255 with a granularity of 128 bytes"), bit 13 `GLG_EN` ("group launch guarantee"). Named barrier count, `ENABLE_DYNAMIC_VGPR`, `TCP_SPLIT` and `ENABLE_DIDT_THROTTLE` are all "GFX120*: Reserved, must be 0" and only meaningful on gfx125*.
- Scratch size field in the program-info register: `SetBits(ScratchBlocks, Mask=0x3FFFF, Shift=12)` for GFX12 vs 0x7FFF on GFX11 and 0x1FFF before (`AMDGPUAsmPrinter.cpp`).
- Code object: `gfx12-generic` requires COV6 (`FeatureRequiresCOV6`); ELF machine values `EF_AMDGPU_MACH_AMDGCN_GFX1200` / `GFX1201` / `GFX12_GENERIC` exist. The doc processor table lists both chips as dGPU with target features `cumode` and `wavefrontsize64` selectable (default wave32), "Architected flat scratch", "Packed work-item IDs".

### Dynamic VGPR

READ (docs "amdgpu-dynamic-vgpr-block-size"): "Represents the size of a VGPR block in the 'Dynamic VGPR' hardware mode, introduced in GFX12. A value of 0 (default) means that dynamic VGPRs are not enabled. Valid values for GFX12+ are 16 and 32. Waves launched in this mode may allocate or deallocate the VGPRs using dedicated instructions, but may not send the DEALLOC_VGPRS message. If a shader has this attribute, then all its callees must match its value. An amd_cs_chain CC function with this enabled has an extra symbol prefixed with '_dvgpr$'..." Code: `getDynamicVGPRBlockSize` accepts only 16 or 32; `MaxDynamicVGPRBlocks = 8` (so max 128 or 256 VGPRs); in dynamic mode `getMaxNumVGPRs` ignores waves-per-EU; end-of-program uses `S_ALLOC_VGPR 0` instead of the `s_sendmsg` dealloc; the frame pointer reserves scratch "for any dynamic VGPRs that might need to be saved by the CWSR trap handler" (`SIFrameLowering`, docs: "when dynamic VGPRs are enabled, the prologue will check if the kernel is running on a compute queue"); the `_dvgpr$` symbol encodes `(NumBlocks - 1) << 3`, an error is raised when blocks exceed 8. Calling convention involved: `AMDGPU_CS_Chain` (and `AMDGPU_CS_ChainPreserve`) shaders. INFERRED: the dynamic-VGPR enable bit in `compute_pgm_rsrc3` is reserved on gfx120x ("GFX120*: Reserved") and the only per-chip signal is the CS-chain path, so ordinary HIP kernels on gfx1200/1201 cannot use it; LLVM comments mention gfx12 mainly for graphics/raytracing chained shaders. Mesa: only a comment, `On GFX12+, wave32 will also be required for a future dynamic VGPR allocation implementation.` (no implementation in ac_gpu_info/RADV).

### Register limits per wave size (READ)

- Total VGPRs per SIMD (`getTotalNumVGPRs`, gfx12 has `1536VGPRs`): wave32 = 1536, wave64 = 768 (Mesa: `info->gfx_level == GFX12 -> num_physical_wave64_vgprs_per_simd = 768`). Allocation granule: wave32 24, wave64 12. Addressable per wave: 256 (gfx12 has no VGPR MSB feature). So wave32 at 192 VGPRs gives 1536/192 = 8 waves (the docs example: "On a gfx12 target such as gfx1200 (wave32, 4 workgroup SIMDs), the default ABI occupancy is ... 8, which caps VGPRs at getMaxNumVGPRs(8) = 192").
- Max waves per SIMD: 16 (`FeatureMaxWavesPerEU16`; Mesa `max_waves_per_simd = 16` for GFX10_3+).
- SGPRs: addressable 106 (`getAddressableNumSGPRs`, gfx10+), 108 allocated (Mesa `sgpr_alloc_granularity = 108`: "includes VCC, which can be treated as s[106-107] on GFX10+"). LLVM `getMaxNumSGPRs` returns 108 for non-addressable. SGPRs are not an occupancy limiter on gfx10+.
- LDS: 64 KB per workgroup (both compilers); WGP has 128 KB.

---

## 7. Errata and workarounds specific to gfx12 (collected, with provenance)

| Item | Where | Quote / evidence | Date or commit |
|---|---|---|---|
| VcmpxPermlane needs a real VALU (v_mov) not a v_nop | `GCNHazardRecognizer.cpp` | "V_NOP will be discarded by SQ." | not found (no blame access) |
| SGPR read hazard wait (s_wait_alu) | `AMDGPUWaitSGPRHazards.cpp` | "Insert s_wait_alu instructions to mitigate SGPR read hazards on GFX12." Options `amdgpu-sgpr-hazard-boundary-cull`, `amdgpu-sgpr-hazard-mem-wait-cull`, `amdgpu-sgpr-hazard-mem-wait-cull-threshold` (default 8) tune it | not found |
| WMMA operand overlap needs one VALU between | `fixWMMAHazards` | see section 2 | not found |
| Waits before system-scope stores | `SIGfx12CacheControl::insertWaitsBeforeSystemScopeStore` | "`// TODO: implement flag for frontend to give us a hint not to insert waits.`" and "`TODO: SIInsertWaitcnts will not always be able to remove STORECNT waits since it does not distinguish atomics-with-return from regular stores.`" | not found |
| s_nop 0 before dealloc-VGPRs message | `SIInsertWaitcnts.cpp` + Mesa NOPs | quoted above | not found |
| Uncached L2 reads/writes to the same address can be reordered on gfx12 | Mesa `ac_gpu_info.c` | "On GFX12, uncached reads and writes to the same address can execute out of order. That's OK for DRI_PRIME blits, but exposure via APIs should be disallowed if general shader access can occur. Alternatively, the shader compiler can add extra s_wait to enforce ordering." (`has_out_of_order_uncached_l2 = gfx_level == GFX12`) | not found |
| SMEM on NULL PRT page hangs | Mesa `ac_gpu_info.c` | "On GFX10-12, SMEM loads on a NULL PRT page throws a VM fault and hangs the GPU." (`has_smem_with_null_prt_bug`) | not found |
| Random hangs when the last VGT stage exports VRS rates | Mesa | "GFX12 is affected by random GPU hangs when VRS rates are exported by the last VGT stage under some conditions that are unclear. One possible workaround is to emit BOP events after every draw that exports VRS rates." | not found |
| Stencil force-valid bug (graphics) | Mesa | `has_db_force_stencil_valid_bug = gfx_level == GFX12` | not found |
| draw-opaque / GE bug 11.5 and 12 | Mesa | `has_draw_opaque_ge_bug = gfx_level >= GFX11_5 && gfx_level <= GFX12` ("see drawOpaqueSqNonEvents in PAL") | not found |
| HiZ workaround setting | RADV | `gfx12_hiz_wa` default `RADV_GFX12_HIZ_WA_FULL`, user override "disabled/partial/full/full_rez" | not found |
| CP/SDMA/GE not coherent with GL2 (system scope needed) | Mesa `ac_shader_util.h` | `cp_sdma_ge_use_system_memory_scope = gfx_level == GFX12` (not set for GFX12.1) | not found |
| FP8 conversion "saturate" does not clamp Inf | Mesa `aco_select_nir_alu.cpp` | "Ideally we would want to use FP16_OVFL for the sat variants, but the ISA doc is wrong and Inf isn't clamped to max_float." | not found |
| s_setpc source possibly hit by SGPR read hazard | Mesa `aco_insert_NOPs.cpp` | "there are GPU hangs in the wild suggesting that the s_setpc source may be susceptible to VALUReadSGPRHazard ... TODO: This hypothesis is not yet conclusively proven." | not found |

I could not do a "blame" or commit-title lookup (no git, GitLab blame not fetched), so no commit title/date is attached to any line above.

---

## 8. Mesa ACO hazards for GFX12 (READ, `aco_insert_NOPs.cpp`)

ACO runs one context type for all GFX11+ (`NOP_ctx_gfx11`) and gates inside it. In `insert_NOPs`, GFX12 uses `mitigate_hazards<NOP_ctx_gfx11, handle_instruction_gfx11, resolve_all_gfx11>`. `required_export_priority` runs only `if (program->gfx_level == GFX11_5 && (NGG/pixel shader))`, with the comment `/* FeatureRequiredExportPriority in LLVM */`.

| Hazard | ACO trigger / fix | gfx12? | LLVM agreement |
|---|---|---|---|
| s_nop before dealloc | `Due to a hazard, an s_nop is needed before "s_sendmsg sendmsg_dealloc_vgprs"` -> `s_nop 0` if the previous inserted instruction is not already an s_nop | YES | agrees |
| VcmpxPermlaneHazard | v_cmpx writing exec followed by permlane16/permlanex16/permlane64/permlane16_var/permlanex16_var with no VALU between -> insert `v_nop` ("Unlike on GFX10, v_nop should resolve the hazard on GFX11") | YES | **disagree**: LLVM inserts `v_mov_b32 vX, vX` because "V_NOP will be discarded by SQ" and its expiry function excludes `V_NOP_e32/e64/sdwa`; ACO's fix is a v_nop and treats any VALU (including its own inserted v_nop) as clearing the state. One of the two comments is wrong or they model different silicon revisions. Mesa has run many shaders on gfx11/gfx12 with v_nop, which suggests v_nop is sufficient (INFERRED) |
| VALUReadSGPRHazard | READ: "VALU reads SGPR and later written by VALU/SALU cannot safely be read by VALU/SALU." Tracks `sgpr_read_by_valu` (SGPR pairs), `sgpr_read_by_valu_then_wr_by_valu`, and `sgpr_read_by_valu_then_wr_by_salu` as a counter map (13 values). A SALU write expires by itself after 10 SALU instructions before a SALU reader, 11 before a VALU reader (`expiry_count = instr->isSALU() ? 10 : 11`). Fix = `s_waitcnt_depctr` with mask bits: `0xfffe` (sa_sdst), `0xfffd` (va_vcc), `0xf1ff` (va_sdst). SGPR >= `sgpr_null`, exec, m0, scc ignored | YES (gfx12) | **partial disagreement**: LLVM has no expiry counting (it always waits after a conflicting write unless it sees an existing s_wait_alu / SMEM / VMEM) so ACO emits fewer waits; LLVM adds an optional DS_NOP "cull" scheme ACO does not have |
| VALUMaskWriteHazard | `if (state.program->gfx_level < GFX12)`, wave64 only | NO (GFX11 only) | agrees (LLVM: generation == GFX11) |
| VALUTransUseHazard | `gfx_level < GFX11_5` | NO | agrees |
| VALUPartialForwardingHazard | `gfx_level < GFX12` | NO | agrees (LLVM: generation == GFX11) |
| LdsDirectVALU/VMEM | gfx12: sets `wait_vsrc = 0` on the LDSDIR instead of s_waitcnt_depctr; gfx12 distinguishes VMEM load / sample / bvh / store / ds | YES (irrelevant to compute) | agrees; LLVM TODO says it should distinguish counters per VMEM type |
| WMMAHazards | READ: "`test_vgpr_bitset(ctx.vgpr_written_by_wmma, instr->operands[0]) || ... operands[1] ... || (is_swmma && operands[2])`" -> `v_nop`; any VALU clears. For SWMMAC operand index 2 is the sparsity index | YES | agrees (LLVM: A/B overlap, plus SWMMAC index on gfx12) |
| resolve at s_setpc/s_swappc/s_call/p_end_with_regs | `resolve_all_gfx11` flushes everything; for gfx12 shaders with a previous part ("has_previous_part": epilog, prolog, separately compiled merged shaders, raytracing) it assumes all SGPRs were read by VALU (`initial_ctx.sgpr_read_by_valu.flip()`) and forces a wait at the start | YES | LLVM treats callee entry blocks as "track all SGPRs" |
| `v_xor3_b32 v0, v0, s0, s0` + `s_waitcnt_depctr 0x0fff` at resolve | "workaround possible LdsDirectVALUHazard/VALUPartialForwardingHazard" | gfx11 only branch | - |

---

## 9. What `ac_gpu_info.c` records for GFX12 (READ)

- Family mapping: `FAMILY_NV4` with IP gfx version minor 0 -> `identify_chip(GFX1200)` and `identify_chip(GFX1201)`; minor 1 -> `CHIP_GFX1210` (GFX12_1, a different level); `ver_major == 12 && ver_minor == 0` -> `info->gfx_level = GFX12`. Navi 44 / Navi 48 names do not appear; no per-chip flag differs between gfx1200 and gfx1201 in this file. Per-chip variation comes only from the CU/SE/RB masks the kernel reports.
- Compiler info (`ac_compiler_info`): `max_waves_per_simd = 16`; `num_simd_per_compute_unit = 2` (gfx10+); `lds_size_per_workgroup = 64 KB` with the comment "GFX10+: LDS is 128KB in WGP mode, but a workgroup can only use up to 64KB."; SGPR: 108 (min/max/granularity); `num_physical_wave64_vgprs_per_simd = 768` ("Navi31/32/Strix Halo/GFX12"); `wave64_vgpr_alloc_granularity = 768/64 = 12`, `wave64_vgpr_encode_granularity = 4`; `max_vgpr_alloc = 256`; `has_lds_bank_count_16` false.
- Chip flags that are gfx12-specific or gfx12-inclusive: `has_out_of_order_uncached_l2 = GFX12`; `has_smem_with_null_prt_bug` (gfx10-12 except gfx9 and gfx1156); `has_db_force_stencil_valid_bug = GFX12`; `has_vrs_export_bug = GFX12`; `has_draw_opaque_ge_bug` (11.5 to 12); `cp_sdma_ge_use_system_memory_scope = GFX12`; `cp_dma_use_L2 = false` when that is set; `gfx12_supports_dcc_write_compress_disable = gfx12 && drm_minor >= 60`; `tcc_cache_line_size = 256` (128 on gfx10/11); `rt_ip_version = RT_3_1` (gfx11 `RT_2_0`); `tcc_rb_non_coherent` false on gfx12; `has_set_context_pairs / set_sh_pairs / set_uconfig_pairs = true`.
- Not set on gfx12 (gfx11 only): `has_attr_ring_wait_bug` (gfx11 only), `has_export_conflict_bug` (gfx11 only), `pc_lines / pbb_max_alloc_count` ("Gfx12 doesn't use pc_lines and pbb_max_alloc_count"), `has_tc_compatible_htile`, display-DCC retile flags, late-alloc CU masking ("Gfx12 doesn't need to mask CUs for late alloc").
- CU mask / occupancy: CU enable mask is 32 bits per SE ("The CU mask has 32 bits and is per SE, not per SA" in `ac_shader_util.c`). `COMPUTE_RESOURCE_LIMITS`: "On GFX12+, WAVES_PER_SH means waves per SE" (multiplied by `max_sa_per_se`).
- Attribute/pos/prim ring sizes for graphics: `attribute_ring_size_per_se = 1400 KB` with the Navi48 benchmark comment (1024 vs 1400 vs 1900); irrelevant to compute.
- Instruction prefetch: `instr_prefetch_distance = 3` for gfx10+ non-CDNA3/MI200 (feeds `INST_PREF_SIZE` in RADV, `radv_get_inst_pref_size`).
- RADV compute wave size (`radv_physical_device.c`, `radv_shader.c`): the code first sets `cs_wave_size = ps_wave_size = ge_wave_size = rt_wave_size = 64` for every chip; for gfx10+ only the perftest flags `RADV_PERFTEST_CS_WAVE_32` / PS / GE switch to 32, while rt defaults to 32 unless `RADV_PERFTEST_RT_WAVE_64`. So READ: **by default RADV compiles ordinary compute shaders as wave64 on gfx12**; wave32 only if the app requests subgroup size 32 (`requiredSubgroupSize`), the workgroup has `local_size <= 32` ("Use wave32 for small workgroups"), the shader uses ray queries (rt wave size), or the user sets `cswave32`. If `cs_wave_size` is 32 but the shader needs full subgroups (wide subgroup ops or cooperative matrix, workgroup size divisible by 64) it is promoted to wave64. The comment: "Default to 32 on RDNA as that gives better perf due to less issues with divergence. On GFX12+, wave32 will also be required for a future dynamic VGPR allocation implementation." (this comment sits on the RT size only).
- Compute dispatch registers on gfx12 (RADV): `NUM_THREAD_FULL_GFX12`; `INTERLEAVE_BITS_X/Y = 1` when the shader declares quad derivative groups; "On GFX12, TRAP_PRESENT doesn't exist for compute shaders and it's enabled by default."

---

## 10. Cooperative matrix on gfx12 in RADV

Enabled when `gfx_level >= GFX11 && !use_llvm` (`radv_cooperative_matrix_enabled`); supported stage is compute only (`cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT`); `cooperativeMatrixRobustBufferAccess = false`. `radv_bfloat16_enabled`: experimental flag OR gfx12+ (BF16 is always on for gfx12; `shaderBFloat16CooperativeMatrix`, `shaderFloat8CooperativeMatrix` follow the coop-matrix switch).

Property table (READ, `fill_array_sizes_structs`): all entries are scope SUBGROUP, M = N = K = 16.

| A type | B type | C = Result type | Saturating | Condition |
|---|---|---|---|---|
| FLOAT8_E4M3 or E5M2 (4 combos) | E4M3 or E5M2 | FLOAT32 | no | `gfx_level >= GFX11_7` (includes GFX12) |
| UINT8 / SINT8 (each) | UINT8 / SINT8 | UINT32 / SINT32 | yes and no; unsigned-acc saturating is skipped ("The HW only supports signed acc.") | all |
| FLOAT16 | FLOAT16 | FLOAT32 or FLOAT16 | no | all |
| BFLOAT16 | BFLOAT16 | FLOAT32 or BFLOAT16 | no | only if `radv_bfloat16_enabled` (true on gfx12) |

Order matters because the Vulkan spec says preferred types are listed first: FP8 first, then int8, then fp16/bf16. 4-bit integer and 16x16x32 shapes exist in hardware (opcodes below) but are not exposed here.

Layout/lowering (READ, `radv_nir_lower_cooperative_matrix.c`):
- Header: "On GFX11, the A&B matrices needs to be replicated... On GFX11.7+, there is no data replication" and a lane/row table: wave32 lanes 0..15 hold rows 0,1,2,..7 across `v0 lo, v0 hi, v1 lo...`, lanes 16..31 hold rows 8..15; wave64 groups of 16 lanes hold row sets (0-3), (8-11), (4-7), (12-15). "Note that the GFX12 ISA doc describes other layouts for A/B, but they are identical to the C layout with the exception of the order of the rows (columns for A). And as long as these are swapped in the same way for both A and B, the muladd result will be the same. So we use the C layout for all uses."
- Length per lane = `256 / wave_size` elements.
- Transposed loads: `use_tr_load = gfx_level >= GFX12 && row-major && is_load && bits < 32 && (global/ubo/ssbo)` -> `load_deref_transpose_amd`, which becomes `global_load_tr_b128` / `global_load_tr_b64` (gfx12-only opcodes). `get_load_tr_row_col` computes the per-lane row/column (8 columns/lane for 16-bit data...).
- Conversions (`convert_base_type`): bf16 <-> f32 via `nir_bf2f` / `nir_f2bf`; FP8 widening: `e4m3fn2f`, `e5m22f`; FP8 narrowing: `f2e4m3fn`, `f2e5m2`, saturating forms `f2e4m3fn_sat`, `f2e5m2_sat`. ACO selection (`aco_select_nir_alu.cpp`): narrowing uses `v_cvt_pk_fp8_f32` / `v_cvt_pk_bf8_f32` (two values per instruction); widening uses `v_cvt_f32_fp8`, `v_cvt_f32_bf8` or the packed `v_cvt_pk_f32_fp8` / `v_cvt_pk_f32_bf8`. Saturation: ACO does not use the FP16_OVFL mode ("the ISA doc is wrong and Inf isn't clamped to max_float"); instead it clamps first with `v_minimummaximum_f32 src, max, max` with the third operand negated, and the max constants are `448.0` (E4M3FN) and `57344.0` (E5M2), "use minimum variant because it preserves NaN". A third variant `f2e4m3fn_satfn` uses a pseudo opcode `p_v_cvt_pk_fp8_f32_ovfl` plus an FP-mode insertion (`needs_fp_mode_insertion`, field `mode_fp16_ovfl`).
- FP8 dot: `e4m3fn_dot4_fadd`, `e5m2_dot4_fadd`, `e4m3fn_e5m2_dot4_fadd` -> `v_dot4_f32_{fp8_fp8, bf8_bf8, fp8_bf8}`.
- Integer saturating accumulation is passed straight through as `.saturate` on `nir_cmat_muladd_amd`.
- Statistics model (`aco_statistics.cpp`, a compiler guess, not measured): gfx11 WMMA cost 32 (16 for iu4); gfx12: "Half the cost of GFX11, int4/8 and (b)f8 twice as fast as (b)f16": f16/bf16 (including swmmac 16x16x32) cost 16, everything else cost 8 VALU cycles, plus 4 latency.

### gfx12-only (more precisely gfx11_7-and-up) opcodes in `aco_opcodes.py`

READ: `Opcode` fields are `gfx6 ... gfx11, gfx11_7, gfx12`, and `op()` copies the previous field forward (`enc[i] = enc[i - 1]`), so an opcode declared with `gfx11_7=X` has opcode X on gfx12 too, unless `gfx12=` overrides.

New opcodes with GFX12 numbers (including inherited-from-11_7 ones), instruction classes in parentheses:

| Opcode (GFX12 number) | Notes |
|---|---|
| `v_wmma_f32_16x16x16_f16` 0x40, `_bf16` 0x41, `v_wmma_f16_16x16x16_f16` 0x42, `v_wmma_bf16_16x16x16_bf16` 0x43, `v_wmma_i32_16x16x16_iu8` 0x44, `_iu4` 0x45 (class WMMA) | gfx11 numbers, carried to gfx12 |
| `v_wmma_f32_16x16x16_{fp8_fp8 0x46, fp8_bf8 0x47, bf8_fp8 0x48, bf8_bf8 0x49}`, `v_wmma_i32_16x16x32_iu4` 0x4a (WMMA) | declared `gfx11_7`, inherited |
| `v_swmmac_f32_16x16x32_f16` 0x50, `_bf16` 0x51, `v_swmmac_f16_16x16x32_f16` 0x52, `v_swmmac_bf16_16x16x32_bf16` 0x53, `v_swmmac_i32_16x16x32_iu8` 0x54, `_iu4` 0x55, `v_swmmac_i32_16x16x64_iu4` 0x56, fp8 variants 0x57-0x5a (WMMA) | sparse |
| `v_cvt_f32_fp8` 0x6c, `v_cvt_f32_bf8` 0x6d, `v_cvt_pk_f32_fp8` 0x6e, `v_cvt_pk_f32_bf8` 0x6f (VOP1) | |
| `v_cvt_pk_fp8_f32` 0x369, `v_cvt_pk_bf8_f32` 0x36a, `v_cvt_sr_fp8_f32` 0x36b, `v_cvt_sr_bf8_f32` 0x36c (VOP3) | `p_v_cvt_pk_fp8_f32_ovfl` is a pseudo with no hardware opcode |
| `v_dot4_f32_fp8_bf8` 0x24, `bf8_fp8` 0x25, `fp8_fp8` 0x26, `bf8_bf8` 0x27 (VOP3P) | |
| `v_s_exp_f32/f16` 0x280/0x281, `v_s_log_*` 0x282/0x283, `v_s_rcp_*` 0x284/0x285, `v_s_rsq_*` 0x286/0x287, `v_s_sqrt_*` 0x288/0x289 (class `ValuPseudoScalarTrans`) | pseudo-scalar transcendentals: VALU ops whose result is a scalar value; ACO's cycle model gives them 7 |
| `v_permlane16_var_b32` 0x30f, `v_permlanex16_var_b32` 0x310 | |
| `v_add_f64` 0x02, `v_mul_f64` 0x06, `v_min_f64` 0x0d, `v_max_f64` 0x0e, `v_lshlrev_b64` 0x1f (VOP2 now; the `_e64` forms are gone) | |
| `v_minimum*/maximum*` (IEEE-754-2019) and renamed `v_min_num_*`, e.g. `v_min_f32` 0x15, `v_max_f32` 0x16, `v_min3_f32` 0x229, `v_max3_f32` 0x22a, `v_minmax_f32` 0x268, `v_maxmin_f32` 0x269 | |
| SALU: `s_minimum_f32` 0x4f, `s_maximum_f32` 0x50, `s_minimum_f16` 0x51, `s_maximum_f16` 0x52 (class SFPU), `s_add_u64` 0x53, `s_sub_u64` 0x54, `s_mul_u64` 0x55 | |
| Barriers/wait: `s_barrier_signal` 0x4e, `s_barrier_signal_isfirst` 0x4f, `s_get_barrier_state` 0x50, `s_barrier_init` 0x51, `s_barrier_join` 0x52, `s_wakeup_barrier` 0x57, `s_sleep_var` 0x58, `s_barrier_wait` 0x14, `s_barrier_leave` 0x15, `s_wait_loadcnt` 0x40, `_storecnt` 0x41, `_samplecnt` 0x42, `_bvhcnt` 0x43, `_expcnt` 0x44, `_dscnt` 0x46, `_kmcnt` 0x47, `_loadcnt_dscnt` 0x48, `_storecnt_dscnt` 0x49 | The old `s_waitcnt*`, `s_cmpk_*`, `s_subvector_loop_*`, `s_gl1_inv` are removed (gfx12=-1) |
| Prefetch: `s_prefetch_inst` 0x24, `_pc_rel` 0x25, `s_prefetch_data` 0x26, `s_buffer_prefetch_data` 0x27, `s_prefetch_data_pc_rel` 0x28 | |
| Memory: `global_inv` 0x2b, `global_wb` 0x2c, `global_wbinv` 0x4f, `global_load_tr_b128` 0x57, `global_load_tr_b64` 0x58, `global_atomic_pk_add_f16` 0x59, `_bf16` 0x5a, `global_atomic_cond_sub_u32` 0x50, `global_atomic_ordered_add_b64` 0x73, `flat_atomic_csub_u32` 0x37, `flat_atomic_cond_sub_u32` 0x50, `flat_atomic_pk_add_f16/bf16` 0x59/0x5a, `buffer_atomic_pk_add_f16/bf16` 0x59/0x5a, `image_atomic_add_flt` 0x83, `image_atomic_pk_add_f16/bf16` 0x86/0x87 | `global_inv/wb/wbinv` replace `buffer_gl0_inv` etc. |
| LDS: `ds_cond_sub_u32` 0x98, `ds_sub_clamp_u32` 0x99, `ds_pk_add_f16` 0x9a, `ds_pk_add_bf16` 0x9b, `ds_cond_sub_rtn` 0xa8, `ds_sub_clamp_rtn_u32` 0xa9, `ds_pk_add_rtn_f16/bf16` 0xaa/0xab, `ds_bvh_stack_push4_pop1_rtn_b32` 0xe0, `push8_pop1_rtn_b32` 0xe1, `push8_pop2_rtn_b64` 0xe2 | GWS and `ds_ordered_count` are gone |
| Ray tracing: `image_bvh_dual_intersect_ray` 0x80, `image_bvh8_intersect_ray` 0x81 | |
| SMEM: `s_load_dwordx3` (b96) 0x05, signed/unsigned byte/short loads 0x08-0x0b, `s_buffer_load_*` renumbered (b32 0x10 ... b512 0x14, b96 0x15, byte/short 0x18-0x1b) | |

---

## 11. Where the two compilers disagree, and where comments admit doubt

1. VcmpxPermlane fix: LLVM v_mov, Mesa v_nop (see section 8). Mesa itself says "Unlike on GFX10, v_nop should resolve the hazard on GFX11".
2. SGPR read hazard: Mesa counts SALU instructions (expiry 10 for SALU consumer, 11 for VALU consumer) and so waits less often than LLVM, whose pass assumes any conflicting write needs a wait. Which is right is not stated in either. Mesa has a hypothesis comment: "TODO: This hypothesis is not yet conclusively proven."
3. Non-temporal loads: LLVM sets `TH_NT` (1, non-temporal at all levels); Mesa's `ac_get_hw_cache_flags` sets `gfx12_load_near_non_temporal_far_regular_temporal` (= TH 4 "NT_RT"), with the reason "Don't use non_temporal for SMEM because it can't set regular_temporal for MALL." So LLVM kernels evict from the infinity cache too; RADV shaders keep MALL regular.
4. Volatile / coherent scope: LLVM volatile -> SCOPE_SYS plus wait; Mesa `scope_is_device = ACCESS_COHERENT | ACCESS_VOLATILE` -> SCOPE_DEV (SCOPE_SYS "memory" scope is used only for CP/GE coherent accesses).
5. System-scope write-back: LLVM treats global_wb at system scope as needed on gfx12.0; Mesa describes `cp_sdma_ge_use_system_memory_scope` for the CP/SDMA/GE clients only.
6. Mesa sets `has_out_of_order_uncached_l2` and suggests "the shader compiler can add extra s_wait to enforce ordering"; I found no LLVM code that does this beyond the volatile waits (LLVM's volatile path does wait; non-volatile UC accesses are unordered).
7. VOPD interlock hazard range: LLVM puts `VOPDInterlockHazard` only on gfx11.0/gfx11-generic (11.5 list does not contain it); Mesa uses `has_interlock_hazard = ctx.program->gfx_level <= GFX11_5`, so Mesa is more conservative on 11.5. Not gfx12-relevant but shows the same chip family is modelled differently.
8. gfx12 VOPD WAR: Mesa quotes the ISA doc: "The OPX instruction must not overwrite sources of the OPY instruction" and enforces it ("WaR dependencies are not a concern before GFX12"). LLVM's pairing code has an equivalent (antidependency handling with `isVOPDAntidependencyAllowed`), not the same logic.
9. LLVM doc vs code on WGP mode: docs say the compiler does not support WGP mode on gfx12+, the code and the docs' own table do. Also the doc `compute_pgm_rsrc1` granularity text predates 1536-VGPR allocation (granule 24/12 in code).
10. LLVM `hasExpertSchedulingMode() = gen >= GFX12`: broader than any evidence for gfx120x silicon; a comment elsewhere calls ENABLE_DYNAMIC_VGPR "GFX120*: Reserved" while the attribute doc says dynamic VGPR was "introduced in GFX12".
11. LLVM leaves the `VcmpxPermlaneHazard` description as "TODO: describe me". LLVM also has `TODO: SIInsertWaitcnts will not always be able to remove STORECNT waits`, and `TODO: On GFX12 the hazard should expire on S_WAIT_LOADCNT/SAMPLECNT/BVHCNT according to the type of VMEM instruction`.
12. Saturating FP8 conversion: Mesa's ISA-doc claim is wrong per its own comment ("the ISA doc is wrong and Inf isn't clamped to max_float").
13. Mesa admits uncertainty in cycle costs: "I'm pretty sure our cycle info is wrong at times (necessarily so, e.g. wave64 VALU instructions can take a different number of cycles based on the exec mask)" and "These numbers are all a bit nonsense. LDS/VMEM/SMEM/EXP performance depends a lot on the situation." The "fp64 might be incorrect" comment also applies to gfx10+ ALU costs shared with gfx12.
14. Mesa's ILP/cycle model gives gfx12 ALU costs identical to gfx10/gfx11 (valu 5, quarter-rate 8, trans 10, double 22), but LLVM's GFX12SpeedModel says 6 / 9 / 38; the two cost tables disagree on gfx12 numbers.

---

## Could not read

- `llvm/docs/AMDGPUUsage.rst` at the raw GitHub path (404, file moved); I read the rendered HTML instead, so exact RST wording may differ slightly from the quotes.
- The GitHub mirror of Mesa (404 for all files); GitLab raw worked instead.
- `SIProgramInfo` file itself (no dedicated file found by that name; its fields were read through `AMDGPUAsmPrinter.cpp` and the docs).
- The `def GFX12SpeedModel : SchedMachineModel` definition line (not found in the files I grepped), so issue width / micro-op buffer size of the model are unknown.
- `SIInstrInfo.td`, `VOPDInstructions.td`, `VOP3PInstructions.td`: not read in full; VOPD legality was taken from `GCNVOPDUtils.cpp` and `AMDGPUBaseInfo`.
- `SIFrameLowering.cpp` read only for dynamic-VGPR bits; the rest of gfx12 frame/scratch behaviour was not examined.
- Mesa `aco_form_hard_clauses.cpp` and the instruction_selection VMEM files: not read; ACO's s_clause use on gfx12 is not covered.
- Blame/commit history for every item in section 7: no git access allowed, no web blame fetched, so no commit titles or dates.
- Clang's mapping of `__builtin_nontemporal_*` to `!nontemporal` (clang code not read; INFERRED).
- Whether ACO's NSA/VSAMPLE and ray-tracing paths (`max_nsa_vgprs = 3`, `bvh8`) matter for compute: not followed.
- Any statement about measured RDNA 4 timing: both compilers only hold modelled numbers; nothing here is a measurement.

## What this implies for someone writing gfx12 compute kernels

1. Default wave size differs between the two compilers' pipelines. HIP/LLVM builds gfx1200/gfx1201 kernels as wave32 unless you ask for wave64 (`-mwavefrontsize64`). RADV compiles Vulkan compute as wave64 by default and only drops to wave32 for workgroups of 32 or fewer invocations, a required subgroup size of 32, or the `cswave32` option. VGPR budget (1536 per SIMD in wave32, 768 in wave64) and the WMMA register layout follow the wave size you end up with.
2. VGPR budget: 1536 physical VGPRs per SIMD in wave32 with a 256-register addressing cap; the compiler's occupancy examples give 8 waves at 192 VGPRs (1536/192). Max 16 waves per SIMD. Allocation steps in 24-register units (wave32), so request sizes that fill the granule.
3. gfx12 has far fewer data hazards than gfx11. The remaining ones that your generated code will carry are: `s_wait_alu` waits around SGPR reuse (the "SGPR read hazard", easy to hit if a kernel keeps rewriting scalar values that vector code also reads, especially VCC), a `v_nop` (or `v_mov` in LLVM) between a `v_cmpx` and a permlane, one VALU between dependent WMMAs when a result feeds A or B (accumulator C feeding the next WMMA is fine, the hardware stalls), and `s_nop 0` before the deallocate-VGPRs message. Chaining WMMAs through C in a loop is the cheap pattern; feeding D to A/B directly costs a v_nop.
4. WMMA/SWMMAC: LLVM exposes `__builtin_amdgcn_wmma_*_w32_gfx12` / `_w64_gfx12` (f16, bf16, iu8, iu4, fp8/bf8 combos, 16x16x16, plus `iu4 16x16x32`) and SWMMAC; RADV exposes 16x16x16 subgroup-scope matrices for fp8 (E4M3/E5M2, f32 accumulate), int8 (signed accumulators, optional saturation), fp16 and bf16. A and B are not replicated on gfx12, so the same loaded data feeds both layouts (use row/column conventions consistently). Transposed global loads (`global_load_tr_b128/b64`, wave32/wave64 builtins) let row-major global data load straight into the WMMA layout.
5. FP8: hardware conversion is `v_cvt_pk_fp8_f32` / `v_cvt_f32_fp8` family with OCP encodings (E4M3FN, E5M2); the saturate behaviour is not what the ISA doc says. Mesa had to add an explicit min/max clamp (448.0 for E4M3FN, 57344.0 for E5M2) before the convert to keep Inf from escaping. If you need saturating FP8 converts, clamp in FP32 first.
6. Memory hints: normal accesses are CU-scope, regular temporal. Stronger ordering costs waits: volatile and system scope add a `s_wait_loadcnt`/`s_wait_storecnt` after, and on gfx12.0 system-scope stores add up to five waits before. Device-scope atomics and agent fences cost a `global_inv scope:SCOPE_DEV`, and a `global_wb` appears only at system scope. `__builtin_nontemporal_*` gives TH_NT (all levels); to reach other TH values (HT, NT_RT, NT_HT, WB) or SCOPE_SE you need the buffer builtins with an explicit aux immediate; last-use needs `!amdgpu.last.use` in LLVM IR.
7. Workgroup scope in WGP mode costs more: with waves of one workgroup possibly on both CUs of a WGP, workgroup acquire adds `global_inv scope:SCOPE_SE` and workgroup release adds waits. In CU mode it is free. Pick CU mode (`-mcumode`) for workgroup-synchronised kernels if the launch geometry allows.
8. Up to 63 outstanding loads, 63 stores, 63 LDS ops, 63 sampler ops, 31 scalar loads, 7 BVH ops are trackable per wave; a prefetch loop deeper than that gets conservative waits. Loads and LDS use separate counters, so overlapping a global load with LDS traffic does not force the same wait. On gfx12 do not assume two loads to the same register return in order.
9. Instruction scheduling hints the compiler uses (model, not measurement): VALU 5 cycles, transcendental 9, pseudo-scalar transcendental (`v_s_exp/log/rcp/rsq/sqrt`) 8, double precision 38, LDS 20, VMEM 320. Dual issue (VOPD) requires different destination parity, src banks mod 4 differ (equal registers allowed on gfx12), at most one literal, and "OPX must not overwrite sources of OPY".
10. Do not count on dynamic VGPRs, clusters, tensor/async copy, 64-bit VALU moves or FP4/FP6 on gfx1200/1201; those are gfx1250-only in LLVM's feature lists. Do not count on two compilers behaving the same: LLVM and ACO already differ on non-temporal encoding, volatile scope, and how many SGPR hazard waits they insert, so keep per-compiler microbenchmarks.
