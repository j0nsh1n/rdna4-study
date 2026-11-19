> Provenance: written by a Sonnet helper from Linux master (7.4-rc3) and ROCm source (2026-11-16). Checked by the
> lead against the kernel source: `gfx_target_version` 120000/120001 for GC 12.0.0/12.0.1 (`kfd_device.c:465-469`),
> the 0x60000 register save size shared by gfx1100, gfx1151, gfx1200 and gfx1201 (`kfd_queue.c:451-454`), the
> absence of a static cache table for GC 11 and 12 (`kfd_crat.c`), and the firmware 0x8b oversubscription setting
> (`mes_v12_0.c:1099`). The rest is the helper's reading.
>
> Lead's addition: because the cache table is built at run time from the card, the real values can be read from
> `/sys/class/kfd/kfd/topology/nodes/<n>/` without any GPU load. The RX 7900 XT's values were saved on 2026-11-16 in
> `baseline-7900xt-kfd-topology.txt`: 168 SIMDs (84 compute units), 16 waves per SIMD, 64 KB LDS, level-1 entries of
> 32 KB with 128-byte lines, 12 entries of 256 KB (one per shader array), one 6144 KB level 2, one 81920 KB level 3
> (the 80 MB Infinity Cache), memory width 320 bits, maximum memory clock 1249. Reading the same files on the RX
> 9070 XT is the first step after the swap.

# RDNA 4 (Navi 48 / Navi 44, gfx1201 / gfx1200) as seen by the Linux amdgpu driver, KFD, ROCR and CLR

Sources read on 2026-11-16.

- Linux: `torvalds/linux` master via raw.githubusercontent.com. `Makefile` says `VERSION = 7`, `PATCHLEVEL = 4`, `EXTRAVERSION = -rc3` ("7.4-rc3"). The GitHub commit feed for master showed its newest entry at 2026-11-16T17:35Z. Commit dates in section 7 come from per-file GitHub commit feeds.
- ROCm user space: `ROCm/rocm-systems`, branch `develop`, paths `projects/rocr-runtime/...` and `projects/clr/...`. No commit id was available; this is a "develop as of 2026-11-16" snapshot.
- PCI names: `pciutils/pciids` master `pci.ids`.
- Kernel paths below are relative to `drivers/gpu/drm/amd/`.
- READ = the line was seen. INFERRED = my deduction from lines I saw (stated with the reasoning).

Key caveat up front: for GC 12.0.x the kernel does NOT contain per-chip numbers for cache sizes, CU counts, wave slots or LDS size. These come from the IP-discovery table stored on the card (read by `amdgpu_discovery.c`). The source tells you which fields exist and how they are combined, not their values. Where I could not get a value from source, I say so.

---

## 1. Identification

### 1.1 PCI IDs

The kernel has no per-device-ID list for these chips. `amdgpu/amdgpu_drv.c` (pciidlist) ends with class-based catch-all entries:

```
{ PCI_DEVICE(0x1002, PCI_ANY_ID), .class = PCI_CLASS_DISPLAY_VGA << 8, .class_mask = 0xffffff, .driver_data = CHIP_IP_DISCOVERY },
```

(same for `PCI_CLASS_DISPLAY_OTHER` and `PCI_CLASS_ACCELERATOR_PROCESSING`). READ. Meaning: any AMD (vendor 0x1002) display or accelerator device is claimed and its IP blocks are taken from the discovery table on the card. The last explicitly listed ID block I saw is Beige Goby (0x7420 to 0x743F). So Navi 4x IDs are not in the driver; only `pci.ids` has them.

From `pci.ids` (READ):

| Device ID | Name in pci.ids |
|---|---|
| 0x7550 | `Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]` |
| 0x7551 | `Navi 48 [Radeon AI PRO R9700]` |
| 0x7590 | `Navi 44 [Radeon RX 9050 / 9060 XT]` |

### 1.2 gfx target numbers KFD reports

`amdkfd/kfd_device.c`, `kgd2kfd_probe()`:

```
case IP_VERSION(12, 0, 0):
	gfx_target_version = 120000;
	f2g = &gfx_v12_kfd2kgd;
case IP_VERSION(12, 0, 1):
	gfx_target_version = 120001;
	f2g = &gfx_v12_kfd2kgd;
case IP_VERSION(12, 1, 0):
	gfx_target_version = 120500;
	f2g = &gfx_v12_1_kfd2kgd;
```

READ. So GC 12.0.0 reports 120000 (gfx1200), GC 12.0.1 reports 120001 (gfx1201). Both use the same `gfx_v12_kfd2kgd` function table. GC 12.1.0 (gfx1250, a different, datacenter-style part) has its own table and is not RDNA 4 desktop.

The mapping gfx1200 = Navi 44 and gfx1201 = Navi 48 is INFERRED. The kernel never says "Navi 48". Evidence for the GC version to gfx name link is READ in ROCR `core/runtime/isa.cpp`: `ISAREG_ENTRY_GEN("gfx1200", 12, 0, 0, ...)` and `("gfx1201", 12, 0, 1, ...)`. The chip-name link to gfx name comes from the public naming (pci.ids chip names plus the AMD target list), not from a source line I read.

### 1.3 IP block versions

`amdgpu/amdgpu_discovery.c` (the `amdgpu_discovery_set_*_ip_blocks` functions). READ for which versions the driver accepts and which driver file handles them:

| Block | Version(s) accepted for GC 12.0.x parts | Driver file / ip_block |
|---|---|---|
| GC (graphics + compute) | 12.0.0, 12.0.1 | `gfx_v12_0.c`, `gfx_v12_0_ip_block` |
| Common SoC | 12.0.0, 12.0.1 (this is the NBIO/SOC "soc24" block) | `soc24.c`, `soc24_common_ip_block` |
| GMC / memory controller | 12.0.0, 12.0.1 | `gmc_v12_0.c` |
| UMC (memory channel RAS) | 8.14.0 | `umc_v8_14.h` (`case IP_VERSION(8, 14, 0)` in `gmc_v12_0.c`) |
| PSP | 14.0.2, 14.0.3 and others (14.0.5 is an APU) | `psp_v14_0.c` |
| SMU | 14.0.2, 14.0.3 (14.0.0/14.0.1/14.0.4/14.0.5 are APUs) | `pm/swsmu/smu14/` |
| SDMA | 7.0.0, 7.0.1 (also 7.1.0 for gfx1250) | `sdma_v7_0.c` |
| MES | 12.x | `mes_v12_0.c` |
| IMU | 12.0.0, 12.0.1 | `imu_v12_0.c` |
| VCN / JPEG | 5.0.0 supported in the switch | `vcn_v5_0_0`, `jpeg_v5_0_0` |
| Display | DCN 4.0.1 | `display/` (not read in detail) |

What I could NOT determine from source: which exact IP version numbers Navi 48 vs Navi 44 report for SMU, PSP, SDMA, VCN and DCN. The discovery table on the card holds them and the kernel only has switch statements. The firmware names in section 6 hint at SMU/PSP 14.0.2 for the common case, but the 12.0.0 vs 12.0.1 split of 14.0.2 vs 14.0.3 is not stated anywhere I read.

Other identifying facts (READ):

- `gfx_v12_0.c` `gfx_v12_0_gpu_early_init()`: for 12.0.0 and 12.0.1 `max_hw_contexts = 8`, `sc_prim_fifo_size_frontend = 0x20`, `sc_prim_fifo_size_backend = 0x100`, `sc_earlyz_tile_fifo_size = 0x4C0`. Any other GC version prints `Unsupported GC version`. These are graphics-side values, same for both chips.
- `amdkfd/kfd_topology.c`: the topology node `name` is the lowercase `amdgpu_asic_name[asic_type]`; `vendor_id` and `device_id` come from the PCI device. `gfx_target_version` is the value above.

---

## 2. Cache topology and compute-unit counts

### 2.1 There is no static cache table for GC 11 or GC 12

`amdkfd/kfd_crat.c` has static tables up to Aldebaran, Navi 1x/2x (`navi10_cache_info`, `sienna_cichlid_cache_info`, ... `gc_10_3_6_cache_info`). In `kfd_get_gpu_cache_info()` the GC 11 and GC 12 versions take a different path (READ):

```
case IP_VERSION(11, 0, 0): ... case IP_VERSION(11, 7, 1):
	/* Cacheline size not available in IP discovery for gc11.
	 * kfd_fill_gpu_cache_info_from_gfx_config to hard code it
	 */
	cache_line_size_missing = true;
	fallthrough;
case IP_VERSION(12, 0, 0):
case IP_VERSION(12, 0, 1):
case IP_VERSION(12, 1, 0):
	num_of_cache_types = kfd_fill_gpu_cache_info_from_gfx_config(kdev->kfd, cache_line_size_missing, *pcache_info);
```

So Navi 31 (GC 11.0.0) and Navi 48/44 (GC 12.0.0/12.0.1) use the same function. Cache sizes come from `adev->gfx.config.*` fields filled from the discovery table (`amdgpu_discovery.c`, `gc_info_v1_2` / `v1_3`). READ. There is no "name the table" answer for GC 12.0.x: the table is built at runtime. The only difference is that GC 11 falls through with `cache_line_size_missing = true` (the discovery table lacks line sizes) while GC 12 does not set it, so GC 12 uses the line sizes the card reports.

### 2.2 How the levels are built (READ, `kfd_fill_gpu_cache_info_from_gfx_config`)

| Level | Size field | Line size | `num_cu_shared` | Quoted line |
|---|---|---|---|---|
| L1 vector data (TCP), per CU | `gc_tcp_l1_size` | `gc_tcp_cache_line_size` (GC 11 forced 128) | `gc_num_tcp_per_wpg / 2` | `/* TCP L1 Cache per CU */` ... `num_cu_shared = adev->gfx.config.gc_num_tcp_per_wpg / 2;` |
| L1 scalar instruction (SQC) | `gc_l1_instruction_cache_size_per_sqc` | `gc_instruction_cache_line_size` (GC 11: 128) | `gc_num_sqc_per_wgp * 2` | `/* Scalar L1 Instruction Cache per SQC */` |
| L1 scalar data (SQC) | `gc_l1_data_cache_size_per_sqc` | `gc_scalar_data_cache_line_size` (GC 11: 64) | `gc_num_sqc_per_wgp * 2` | `/* Scalar L1 Data Cache per SQC */` |
| "GL1" per shader array | `gc_gl1c_per_sa * gc_gl1c_size_per_instance` | forced 128 only if `cache_line_size_missing` | `max_cu_per_sh` | `/* GL1 Data Cache per SA */`, only emitted if both fields non-zero |
| L2 (whole GPU) | `gc_gl2c_per_gpu` | `gc_tcc_cache_line_size` (GC 11: 128) | `max_cu_per_sh` | `/* L2 Data Cache per GPU (Total Tex Cache) */` |
| L3 / MALL (Infinity Cache) | `adev->gmc.mall_size / 1024` | hard-coded 64 | `max_cu_per_sh` | `/* L3 Data Cache per GPU */`, only if `mall_size != 0` |

Two things to notice:

- The `num_cu_shared` for L2 and L3 is set to `max_cu_per_sh` (CUs per shader array), not "all CUs". That is a KFD reporting convention; do not read it as "the L2 is split per array". INFERRED.
- A WGP holds 2 CUs. "CUs per TCP" is `gc_num_tcp_per_wpg / 2`, which suggests there is one TCP per WGP or per two CUs on this IP, depending on the value the card reports. I could not see the value. INFERRED.

Commit "drm/amdkfd: hard-code MALL cacheline size for gfx11, gfx12" (2024-12-10) is the origin of the "64" for L3 (READ in feed title; body not read).

### 2.3 Numbers for comparison (a static table that does exist)

`sienna_cichlid_cache_info` (Navi 21, GC 10.3.0), READ:

| Level | Size (KB) | Line | Shared by (CUs) |
|---|---|---|---|
| L1 data (TCP) | 16 | 128 | 1 |
| L1 instruction (SQC) | 32 | 64 | 2 |
| L1 scalar data (SQC) | 16 | 64 | 2 |
| GL1 | 128 | 128 | 2 |
| L2 | 4096 | 128 | 10 |
| L3 (Infinity Cache) | 128*1024 | 64 | 10 |

I did not find numbers for GC 11.0.0 or GC 12.0.x in any source file. Public knowledge (Navi 31 and Navi 48 specs) is not primary source and is not included here. The real values are visible only on a live system under `/sys/class/kfd/kfd/topology/nodes/N/caches/*`, a runtime check I was not allowed to do.

### 2.4 Compute-unit, SIMD, wave and LDS numbers

How KFD derives them:

- `gfx_v12_0.c` `gfx_v12_0_get_cu_info()` (READ): WGP bitmaps are expanded to CUs: `/* if there is one WGP enabled, it means 2 CUs will be enabled */  cu_bitmap_per_wgp = 3 << (2 * wgp_idx);`. At the end: `cu_info->number = active_cu_number; cu_info->simd_per_cu = NUM_SIMD_PER_CU;`. So the kernel's "CU" for RDNA is half a WGP, and `number` counts those CUs. I did not find where `NUM_SIMD_PER_CU` is defined in the files I searched (gfx_v12_0.c, gfx_v11_0.c, amdgpu.h, amdgpu_gfx.h). The CLR ISA table below uses 2, which is consistent. INFERRED: 2 SIMDs per CU, 4 per WGP.
- `amdgpu_discovery.c` (READ), GC info v1.x:
  - `adev->gfx.cu_info.wave_front_size = le32_to_cpu(gc_info->v1.gc_wave_size);`
  - `adev->gfx.cu_info.max_waves_per_simd = le32_to_cpu(gc_info->v1.gc_max_waves_per_simd);`
  - `adev->gfx.cu_info.max_scratch_slots_per_cu = ...gc_max_scratch_slots_per_cu;`
  - `adev->gfx.cu_info.lds_size = le32_to_cpu(gc_info->v1.gc_lds_size);`
  - `max_cu_per_sh = 2 * (gc_num_wgp0_per_sa + gc_num_wgp1_per_sa)`, `max_sh_per_se = gc_num_sa_per_se`, `max_shader_engines = gc_num_se`.
  - For v1.5 the code takes the larger of two shader arrays' WGP counts: `num_wgps_per_sa >= num_wgps_per_sa1 ? ...` (READ, so some arrays can have fewer WGPs enabled than the max).
- `amdkfd/kfd_crat.c` `kfd_create_vcrat_image_gpu` (READ):
  ```
  cu->num_simd_per_cu = cu_info->simd_per_cu;
  cu->num_simd_cores = cu_info->simd_per_cu * (cu_info->number / kdev->kfd->num_nodes);
  cu->max_waves_simd = cu_info->max_waves_per_simd;
  cu->wave_front_size = cu_info->wave_front_size;
  cu->array_count = gfx_info->max_sh_per_se * gfx_info->max_shader_engines;
  cu->num_cu_per_array = gfx_info->max_cu_per_sh;
  cu->max_slots_scatch_cu = cu_info->max_scratch_slots_per_cu;
  cu->lds_size_in_kb = cu_info->lds_size;
  ```
- `amdkfd/kfd_topology.c` prints these as sysfs properties: `simd_count`, `max_waves_per_simd`, `lds_size_in_kb`, `wave_front_size`, `array_count`, `simd_arrays_per_engine` (= `max_sh_per_se`), `cu_per_simd_array`, `simd_per_cu`, `max_slots_scratch_cu`, `gfx_target_version`, `num_cp_queues`, `cwsr_size`, `ctl_stack_size`, `num_gws`.
- `kfd_device.c` (READ): for GC >= 12.0.0, `node->adev->gds.gws_size = 64;` when HWS is on. So 64 GWS barriers/slots are advertised.
- Queues: `gfx_v12_0.c` `gfx_v12_0_sw_init()` (READ): `me.num_me = 1, num_pipe_per_me = 1, num_queue_per_pipe = 8` for graphics and `mec.num_mec = 1; mec.num_pipe_per_mec = 2; mec.num_queue_per_pipe = 4;` for compute. So the hardware has 2 compute pipes x 4 queues = 8 hardware queue slots (HQDs) on the compute microengine for GC 12.0.x. (The default case uses 4 pipes x 8 queues, for later parts.) KFD's `device_info.max_no_of_hqd = 24` is a generic default, not GC 12 specific.

Values I cannot give from source: the actual `max_waves_per_simd`, `lds_size`, `max_cu_per_sh`, and `max_slots_scratch_cu` for Navi 48/44. From the KFD CWSR sizing code below one can deduce waves: see 2.5.

### 2.5 What the CWSR sizing code implies about the chip (READ + INFERRED)

`amdkfd/kfd_queue.c`:

```
static u32 kfd_get_vgpr_size_per_cu(u32 gfxv)
{
	u32 vgpr_size = 0x40000;
	...
	else if (gfxv == 110000 ||		/* GFX_VERSION_PLUM_BONITO */
		 gfxv == 110001 ||		/* GFX_VERSION_WHEAT_NAS */
		 gfxv == 110501 ||		/* GFX_VERSION_GFX1151 */
		 gfxv == 120000 ||		/* GFX_VERSION_GFX1200 */
		 gfxv == 120001)		/* GFX_VERSION_GFX1201 */
		vgpr_size = 0x60000;
```

- Default VGPR bytes saved per CU: 0x40000 = 256 KiB (RDNA 1/2 class). For gfx1100, gfx1101, gfx1151, gfx1200 and gfx1201: 0x60000 = 384 KiB per CU. Note gfx1102 (Navi 33) is not in the list and would use 0x40000 (INFERRED: smaller VGPR file there).
- INFERRED: 2 SIMDs per CU means 192 KiB of VGPRs per SIMD, i.e. 1536 VGPRs of 32 lanes x 4 bytes, versus 1024 on RDNA 2. Same file size as RDNA 3's large-register parts.
- `static u32 kfd_get_sgpr_size_per_cu()`: 0x4000 (16 KiB) for non-12.5. `kfd_get_hwreg_size_per_cu()`: 0x1000. `kfd_get_lds_size_per_cu()`: 0x10000 (64 KiB) except gfx9.5 and gfx12.5, which take `props->lds_size_in_kb << 10`. So for gfx1200/1201 KFD hard-codes 64 KiB LDS per CU in the save-area size (READ). The same 64 KiB is in the CLR table (section 8). Whether the hardware reports 64 KiB per CU or per WGP: the ISA table calls it "local memory size per CU"; KFD's CU here is half a WGP; I did not verify what `gc_lds_size` says. Open question.
- `get_num_waves()`: `else if (gfxv < 120500) wave_num = cu_num * 32;`. READ. So for gfx1200/1201 KFD sizes for 32 waves per CU, i.e. 16 waves per SIMD (with 2 SIMDs/CU). INFERRED: max 16 waves per SIMD, same as RDNA 3.
- `CNTL_STACK_BYTES_PER_WAVE(gfxv) = (gfxv >= 100100 ? 12 : 8)`: 12 bytes of control stack per wave.
- `ctl_stack_size = wave_num * 12 + 8`, then plus a 40-byte header, aligned to `AMDGPU_GPU_PAGE_SIZE` (4 KiB). `debug_memory_size = ALIGN(wave_num * 32, 64)`. `cwsr_size = ALIGN(ctl_stack_size + wg_data_size, PAGE_SIZE)`, where `wg_data_size = ALIGN(cu_num * (VGPR + SGPR + LDS + HWREG), 4096)`.
- Worked example (INFERRED arithmetic, depends on the card's CU count): per CU 0x60000 + 0x4000 + 0x10000 + 0x1000 = 0x75000 = 479,232 bytes = 468 KiB. At 64 CUs (a 9070 XT is publicly 64 CUs; not read from source) the save area is about 30 MB per queue, plus a control stack of 64*32*12 + 8 + 40 bytes = 24,624 B rounded up to 28 KiB.
- `props->eop_buffer_size = 4096` for gfx >= 80000.

---

## 3. Compute scheduling and context save

### 3.1 Queue model

- GC 12 uses MES (the Micro Engine Scheduler) to map queues; KFD uses HWS. `amdgpu/mes_v12_0.c`. MES firmware: `gc_12_0_{0,1}_mes.bin`, `_mes1.bin`, `_uni_mes.bin` (see section 6).
- `gfx_v12_0.c`: user-mode queues through MES are enabled when `me_fw >= 3090, pfp_fw >= 3190, mec_fw >= 3450, mes fw[0] >= 147` (READ; `userq_mes_funcs` for GFX and COMPUTE).
- The "cleaner shader" (RUN_CLEANER_SHADER packet, clears VGPR/LDS between processes) is enabled when `me_fw >= 2480, pfp >= 2530, mec >= 2680, mes >= 100` (READ, `adev->gfx.enable_cleaner_shader = true`).
- Per-queue reset is enabled for compute and gfx when `me_fw >= 2660 && mec_fw >= 2920` and no SR-IOV (READ, `AMDGPU_RESET_TYPE_PER_QUEUE`). The per-pipe reset is commented out: `/* TODO: verify CP warm-reset support before enabling PER_PIPE. */`.
- `amdkfd/kfd_mqd_manager_v12.c` `init_mqd()` (READ):
  - `m->cp_hqd_pq_control = 5 << RPTR_BLOCK_SIZE__SHIFT; ... |= CP_HQD_PQ_CONTROL__UNORD_DISPATCH_MASK;`
  - `m->cp_hqd_quantum = QUANTUM_EN | (1 << QUANTUM_SCALE) | (1 << QUANTUM_DURATION)`
  - `m->cp_hqd_persistent_state = PRELOAD_REQ | 0x55 << PRELOAD_SIZE`
  - `compute_static_thread_mgmt_se0..se7 = 0xFFFFFFFF` (all CUs enabled per SE by default; `update_cu_mask` narrows it via `mqd_symmetrically_map_cu_mask`).
  - Debug: "Set cp_hqd_hq_status0.c_queue_debug_en to 1 to have the CP set up the DISPATCH_PTR. This is required for the kfd debugger".

### 3.2 CWSR (compute wave save/restore)

- `kfd_device.c`: `supports_cwsr = true` for all SOC15+ (`doorbell_size = 8` path). For GC 12.0.x, `cwsr_isa = cwsr_trap_gfx12_hex` from `cwsr_trap_handler_gfx12.asm` (shared with a newer chip it calls `CHIP_GC_12_0_3`; GC 12.1.0 has its own `cwsr_trap_gfx12_1_0_hex`). READ. `BUILD_BUG_ON(sizeof(cwsr_trap_gfx12_hex) > KFD_CWSR_TMA_OFFSET)`: the handler must fit in `KFD_CWSR_TMA_OFFSET = AMDGPU_GPU_PAGE_SIZE + 2048` = 6144 bytes.
- `cwsr_trap_handler_gfx12.asm` header (READ): `#define CHIP_GFX12 37`, `#define CHIP_GC_12_0_3 38`, `#define RELAXED_SCHEDULING_IN_TRAP (ASIC_FAMILY == CHIP_GFX12)`, `#define HAVE_VALU_SGPR_HAZARD (ASIC_FAMILY == CHIP_GFX12)`, `#define HAVE_HT_TRAP_ID_WA (ASIC_FAMILY >= CHIP_GFX12)`, `#define SINGLE_STEP_MISSED_WORKAROUND 1  //workaround for lost TRAP_AFTER_INST exception when SAVECTX raised`. So the plain-GFX12 flavour (our chips) has: a VALU/SGPR hazard workaround in the handler, a "relaxed scheduling in trap" mode, a high-trap-ID workaround, and a single-step workaround. Features like `HAVE_XNACK`, `HAVE_BANKED_VGPRS`, `NUM_NAMED_BARRIERS`, `HAVE_CLUSTER_BARRIER`, `WAVE32_ONLY` are enabled only for CHIP_GC_12_0_3 (gfx1250-class), not for our chips.
- Save-area layout facts (READ in the asm): VGPR count is read from `HW_REG_WAVE_GPR_ALLOC` field `VGPR_SIZE` (8 bits, shift 12): `s_lshl_b32 s_save_alloc_size, s_save_alloc_size, 2  //Number of VGPRs = (vgpr_size + 1) * 4`. So VGPR allocation granularity is 4 registers on these chips (matches ROCR below, `vgpr_gran = 4` for non-12.5). LDS alloc field is 9 bits with granularity 2^9 or 2^10 depending on chip (`SQ_WAVE_LDS_ALLOC_GRANULARITY = 9` / `10`; I did not see which branch applies to GC 12.0).
- Wave64 is handled in the trap handler (`SQ_WAVE_STATUS_WAVE64_SHIFT = 29`), i.e. wave64 remains a hardware mode on these chips. READ.

### 3.3 Dynamic VGPR

- READ: `gfx_v12_0_read_wave_data()` dumps `ixSQ_WAVE_DVGPR_ALLOC_LO` and `ixSQ_WAVE_DVGPR_ALLOC_HI` (plus `SQ_WAVE_SCHED_MODE`) for the wave debug interface, so the registers exist in GC 12.0 headers.
- READ: grep of both `cwsr_trap_handler_gfx12.asm` and `cwsr_trap_handler_gfx12_1_0.asm` for "dvgpr" returns 0 matches, and KFD's save-area size has no term for dynamically allocated VGPR state.
- I found no `SQ_DYN_VGPR`-style enable, no `DVGPR` setting in `gfx_v12_0.c` compute init, in `kfd_mqd_manager_v12.c`, or in the KFD queue code. I could not run a repo-wide code search (rate-limited API), so this is a finding limited to the files I read.
- INFERRED: the Linux kernel does not enable dynamic VGPR allocation for compute on GC 12.0.x. If it is used, a user-space-built compute wave would have to set it up itself (and the CWSR handler would not preserve that state). The register presence says the hardware has it; the lack of any enable says the compute stack ignores it.

### 3.4 Debug / trap support

`amdgpu/amdgpu_amdkfd_gfx_v12.c` (READ): implements `kgd_gfx_v12_enable_debug_trap`, `..._set_wave_launch_trap_override`, `..._set_wave_launch_mode`, `..._set_address_watch`, `wave_control_execute_v12`, `get_wave_count` via `SPI_CSQ_WF_ACTIVE_COUNT`. Trap mask bits supported: `FP_INVALID` and others (list partly read). Address watch: `num_of_watch_points = 4` (`kfd_device.c`). Wave launch modes via `SPI_GDBG_PER_VMID_CNTL` (`TRAP_ON_START`, `TRAP_ON_END`, `EXCP_REPLACE`). `gfx_v12_0.c` also sets `SPI_GDBG_PER_VMID_CNTL TRAP_EN` (line ~1820).

### 3.5 Per-chip quirks coded for GC 12.0.0 / 12.0.1

Almost every switch lists 12.0.0 and 12.0.1 together, so there are few per-chip differences. The ones I found (READ):

- `gmc_v12_0.c`: `if (amdgpu_ip_version(adev, GC_HWIP, 0) == IP_VERSION(12, 0, 0)) r = amdgpu_irq_add_id(adev, SOC21_IH_CLIENTID_DF, 0, &adev->gmc.ecc_irq);` Only GC 12.0.0 routes DF ECC interrupts through `SOC21_IH_CLIENTID_DF`; others use `SOC_V1_0_IH_CLIENTID_DF`. Minor.
- Firmware: `gc_12_0_1_rlc_kicker.bin` and `gc_12_0_1_imu_kicker.bin` exist only for 12.0.1 (see section 6).
- `kfd_device.c` event interrupts for 12.0.0/12.0.1: `/* GFX12_TODO: Change to v12 version. */  event_interrupt_class = &event_interrupt_class_v11;`.
- PCIe atomics: `kfd_device.c`, `gc_version < IP_VERSION(13, 0, 0)` branch: `needs_pci_atomics = true; no_atomic_fw_version = 2090;` for GC 12. So KFD wants PCIe atomics unless MEC firmware is >= 2090 (READ). Commit "drm/amdkfd: Enable atomic support for GFX12" (2024-05-02) turned the atomic bit on in the MQD.

---

## 4. Memory

### 4.1 What the driver describes

`amdgpu/gmc_v12_0.c` (READ unless noted):

- Size: `adev->gmc.mc_vram_size = adev->nbio.funcs->get_memsize(adev) * 1024ULL * 1024ULL;`. Resizable BAR is attempted for dGPUs: `r = amdgpu_device_resize_fb_bar(adev);`.
- Aperture from BAR 0: `aper_base = pci_resource_start(adev->pdev, 0); aper_size = pci_resource_len(adev->pdev, 0);`; `visible_vram_size` is clamped to `real_vram_size`.
- GART: `amdgpu_gmc_set_gart_size(adev, SZ_512M);`
- VM size for GC 12.0.x: `amdgpu_vm_adjust_size(adev, 256 * 1024, 9, 3, 48);` with comment "To fulfill 4-level page support, vm size is 256TB (48bit), maximum size, block size 512 (9bit)". `pte_addr_mask = 0x0000FFFFFFFFF000ULL; /* 48 bit PA */`, `dma_addr_bits = 44;` (GC 12.1.0 differs: 57-bit VA, 52-bit PA). The "3" is the number of page-table levels past the root, per the helper's argument names (INFERRED from the comment).
- Memory type and width come from VBIOS/ATOM tables: `amdgpu_gmc_get_vram_info()` (in `amdgpu_gmc.c`) for GC 12.0.0/12.0.1 calls `amdgpu_atomfirmware_get_umc_info(adev, vram_width, vram_type, vram_vendor)`. `amdgpu_atomfirmware.c` `convert_atom_mem_type_to_vram_type()` maps `ATOM_DGPU_VRAM_TYPE_GDDR6` to `AMDGPU_VRAM_TYPE_GDDR6` and HBM variants to HBM; I saw GDDR5, GDDR6, HBM2/2E/3 and HBM3E, and no GDDR7 branch. Consistent with RDNA 4 desktop cards using GDDR6. The bus width (and so the number of 32-bit channels) is read at runtime, not coded. Kernel file confirms only that `kfd_crat.c` forwards `local_mem_info->vram_width` as `sub_type_hdr->width` and ROCR/CLR use it (section 8: `globalMemChannels_ = vramBusBitWidth_ / 32`).
- Memory channels: `umc_v8_14.h`: `UMC_V8_14_CHANNEL_INSTANCE_NUM 2` (2 channels per UMC instance), `UMC_V8_14_PER_CHANNEL_OFFSET 0x400`, `UMC_V8_14_INST_DIST 0x40000`. `UMC_V8_14_TOTAL_CHANNEL_NUM(adev)` multiplies UMC instance count by channel count. Instance count is set at runtime. READ (no numeric bus width).
- Infinity Cache ("MALL"): size from the discovery MALL table in `amdgpu_discovery_get_mall_info()` (`adev->gmc.mall_size = (uint64_t)mall_size_per_umc * adev->gmc.num_umc;` for v2, or per-UMC with half/double adjustments for v1). Exposed to KFD as the L3 entry (section 2.2). No fixed number in source. READ.
- L2 slices: `adev->gfx.config.max_texture_channel_caches` = `gc_num_gl2c` (v1) or `gc_num_tccs` (v2). `gmc_v12_0_get_dcc_alignment()` for 12.0.0/12.0.1 uses `max_tex_channel_caches / 4` (or `roundup_pow_of_two`) for DCC alignment: tells you the L2 slice count matters for compressed-surface layout (graphics, not compute). READ.

### 4.2 Page-table format and translation

From the comment block in `gmc_v12_0.c` (READ):

```
PTE format: 63 P | 58 D | 57 G | 56 T | 55:54 M | 53:52 SW | 47:12 4k physical page base | 11:7 fragment | 6 write | 5 read | 4 exe | 3 Z | 2 snooped | 1 system | 0 valid
PDE format: 63 P | 62:58 block fragment size | 56 A | 55:54 M | 47:6 base of PD/PTE | 2 C | 1 system | 0 valid
```

- Fragment field is 5 bits (11:7): a PTE can describe a power-of-two run of 4 KiB pages as one TLB entry. The kernel PDE code sets `AMDGPU_PDE_BFS_GFX12(0x9)` (block fragment size field = 9, meaning 2^9 x 4 KiB = 2 MiB) when `translate_further` is off and the level is `AMDGPU_VM_PDB0`. INFERRED meaning of 9 from the "block size 512 (9bit)" comment. READ for the 0x9.
- MTYPE for default GPU VM mappings: `case AMDGPU_VM_MTYPE_DEFAULT: *flags = AMDGPU_PTE_MTYPE_GFX12(*flags, MTYPE_NC);`. So default (and NC) VRAM/GTT mappings are MTYPE_NC; UC is only if explicitly asked. `AMDGPU_VM_PAGE_NOALLOC` sets `AMDGPU_PTE_NOALLOC` (do not allocate in MALL/Infinity Cache) (READ). This is the bit for "streaming" accesses that skip the big cache; user-space gets it through the VM flags of `amdgpu_gem_va`.
- Retry faults: `noretry_flags = AMDGPU_VM_NORETRY_FLAGS_GFX12`. The retry-fault handler exists (`amdgpu_gmc_handle_retry_fault`), but GC 12.0.x has no XNACK in the CWSR flags (XNACK handling is only for GC_12_0_3). INFERRED: demand paging (SVM with page faults on GPU) is limited on these chips.
- TLB flush: `gmc_v12_0_flush_gpu_tlb_pasid` can invalidate through MES (`adev->mes.funcs->invalidate_tlbs_pasid`) (READ).
- Recent fix: "drm/amdgpu/gmc12: properly pass flush_type to gmc_v12_0_flush_vm_hub()" (2026-10-01) (title only).

### 4.3 Address-space layout KFD gives a compute process

`amdkfd/kfd_flat_memory.c` `kfd_init_apertures_v12()` (READ):

```
pdd->lds_base = pdd->dev->adev->gmc.shared_aperture_start;
pdd->lds_limit = pdd->dev->adev->gmc.shared_aperture_end;
pdd->gpuvm_base = AMDGPU_VA_RESERVED_BOTTOM;
pdd->gpuvm_limit = pdd->dev->kfd->shared_resources.gpuvm_size - 1;
pdd->scratch_base = pdd->dev->adev->gmc.private_aperture_start;
pdd->scratch_limit = pdd->dev->adev->gmc.private_aperture_end;
pdd->qpd.cwsr_base = AMDGPU_VA_RESERVED_TRAP_START(pdd->dev->adev);
```

with `gmc_v12_0.c`:

```
adev->gmc.shared_aperture_start = 0x2000000000000000ULL;   (end = start + (4ULL << 30) - 1)
adev->gmc.private_aperture_start = 0x1000000000000000ULL;  (end = start + (4ULL << 30) - 1, unless GC >= 12.1.0)
```

and `amdgpu_vm.h`: `AMDGPU_VA_RESERVED_BOTTOM = 1ULL << 16` (64 KiB), `AMDGPU_VA_RESERVED_TRAP_SIZE = 1ULL << 16`, `AMDGPU_VA_RESERVED_SEQ64_SIZE = 2 MiB`, `AMDGPU_VA_RESERVED_CSA_SIZE = 2 MiB`.

Summary for a GC 12.0.x process:

| Region | Address | Size |
|---|---|---|
| Unmapped bottom | 0 to 0xFFFF | 64 KiB |
| GPUVM (SVM-compatible VA) | 0x10000 up to `gpuvm_size - 1` (VM is 48-bit, 256 TB, minus the reserved top) | about 256 TB |
| LDS ("shared") aperture | 0x2000_0000_0000_0000 to +4 GiB | 4 GiB window in the flat address space |
| Scratch ("private") aperture | 0x1000_0000_0000_0000 to +4 GiB | 4 GiB window |
| Trap handler (TBA/TMA) | top of VA, below SEQ64 and CSA reserved regions | 64 KiB |

The LDS and scratch bases are in the non-canonical hole of the 48-bit space. Flat pointers whose upper bits match these windows hit LDS or scratch. INFERRED from the aperture definitions.

---

## 5. Power and clocks

### 5.1 Which SMU code

`pm/swsmu/smu14/smu_v14_0_2_ppt.c` handles SMU 14.0.2 (and 14.0.3, e.g. `if (mp1_ver == IP_VERSION(14, 0, 3) && smu->smc_fw_version >= 0x00685000)`). Driver interface version: `#define SMU14_DRIVER_IF_VERSION_SMU_V14_0_2 0x2E`. READ.

### 5.2 Clock domains and DPM features

- Clock map (READ): `GFXCLK`/`SCLK` -> `PPCLK_GFXCLK`, `SOCCLK`, `FCLK`, `UCLK`/`MCLK` -> `PPCLK_UCLK`, `DCEFCLK` -> `PPCLK_DCFCLK`. The PMFW interface header lists `PPCLK_GFXCLK, SOCCLK, UCLK, FCLK, DCLK_0, VCLK_0, DISPCLK, DPPCLK, DPREFCLK, DCFCLK, DTBCLK`.
- DPM features always allowed: `FEATURE_DPM_GFXCLK`, `DPM_UCLK`, `DPM_LINK`, `DPM_SOCCLK`, `DPM_FCLK` (`smu_v14_0_2_dpm_features`). Other features mapped: `DS_GFXCLK/SOCCLK/FCLK/DCFCLK/UCLK` (deep sleep), `GFXOFF`, `VR0HOT`, `FAN_CONTROL`, `GFXCLK_SPREAD_SPECTRUM`, `MEM_TEMP_READ`, `FAN_ABNORMAL`. READ.
- Clock/voltage defaults live in the PMFW pptable (`PPTable_t`, `SkuTable`, `CustomSkuTable`) read from the card (`smu_v14_0_2_get_pptable_from_pmfw`), not in source. The only source-embedded thermal numbers are:
  `static const struct smu_temperature_range smu14_thermal_policy[] = { {-273150, 99000, 99000, -273150, 99000, 99000, -273150, 99000, 99000}, { 120000, 120000, ...} };` (millidegrees C). So the fallback policy is 99 C critical / 120 C emergency. The actual limits are overridden by `pptable->CustomSkuTable.TemperatureLimit[TEMP_EDGE/HOTSPOT/MEM]` plus offsets `CTF_OFFSET_EDGE/HOTSPOT/MEM` (READ, `smu_v14_0_2_get_thermal_temperature_range`). I found no hot-spot number in source.
- Sensors: edge, hotspot, memory, VR GFX, VR SoC temps (`metrics->AvgTemperature[TEMP_*]`), `AverageGfxActivity`, `AverageUclkActivity`. A `SMU_14_0_2_BUSY_THRESHOLD 5` decides between "pre-deep-sleep" and "post-deep-sleep" frequency reporting (READ).

### 5.3 Power profiles (READ)

```
WORKLOAD_MAP(PP_SMC_POWER_PROFILE_BOOTUP_DEFAULT, WORKLOAD_PPLIB_DEFAULT_BIT),
..._FULLSCREEN3D, _POWERSAVING, _VIDEO, _VR, _COMPUTE, _CUSTOM, _WINDOW3D
```

So `pp_power_profile_mode` offers: BOOTUP_DEFAULT, 3D_FULL_SCREEN, POWER_SAVING, VIDEO, VR, COMPUTE, CUSTOM, WINDOW_3D. The listing prints, per profile and per clock (GFXCLK, FCLK): `FPS, MinActiveFreqType, MinActiveFreq, BoosterFreqType, BoosterFreq, PD_Data_limit_c, PD_Data_error_coeff, PD_Data_error_rate_coeff` (activity-monitor coefficients stored in firmware). CUSTOM takes `SMU_14_0_2_CUSTOM_PARAMS_COUNT = 9` parameters per clock for 2 clocks.

Compute-specific code in `smu_v14_0_2_set_power_profile_mode()`:

```
/* disable deep sleep if compute is enabled */
if (workload_mask & (1 << PP_SMC_POWER_PROFILE_COMPUTE))
	smu_v14_0_deep_sleep_control(smu, false);
else
	smu_v14_0_deep_sleep_control(smu, true);
```

So picking the COMPUTE profile turns off clock deep sleep (the DS_* features), which removes wake-up latency between bursts of work. READ.

Who selects the profile:

- KFD: `amdgpu_amdkfd.c` `amdgpu_amdkfd_set_compute_idle()` ends with `amdgpu_dpm_switch_power_profile(adev, PP_SMC_POWER_PROFILE_COMPUTE, !idle);`. READ. So KFD (ROCm) workloads turn on the COMPUTE profile when queues are active.
- amdgpu kernel rings: `amdgpu_gfx.c` `amdgpu_gfx_profile_ring_begin_use()`: `if (adev->gfx.num_gfx_rings) profile = PP_SMC_POWER_PROFILE_FULLSCREEN3D; else profile = PP_SMC_POWER_PROFILE_COMPUTE;`. A dGPU with a graphics ring (Navi 48 has one: `GFX12_NUM_GFX_RINGS 1`) gets FULLSCREEN3D for submissions through the DRM scheduler (Vulkan/OpenCL via Mesa), not COMPUTE. `if (amdgpu_dpm_is_overdrive_enabled(adev)) return;`: with overdrive on, this auto profile switching is skipped. READ.

### 5.4 Sysfs knobs that touch compute speed

I read the SMU-side implementation; the generic sysfs files are in `pm/amdgpu_pm.c` (not read in detail).

- `power_dpm_force_performance_level`: auto, low, high, manual, profile_standard/min_sclk/min_mclk/peak. Generic. (Not read; INFERRED from kernel ABI knowledge.)
- `pp_power_profile_mode`: see 5.3.
- `pp_od_clk_voltage` (SMU 14.0.2 overdrive, READ in `smu_v14_0_2_ppt.c`): features listed in the source:

  ```
  PP_OD_FEATURE_GFXCLK_FMIN 0, GFXCLK_FMAX 1, UCLK_FMIN 2, UCLK_FMAX 3, GFX_VF_CURVE 4,
  FAN_CURVE_TEMP 5, FAN_CURVE_PWM 6, FAN_ACOUSTIC_LIMIT 7, FAN_ACOUSTIC_TARGET 8,
  FAN_TARGET_TEMPERATURE 9, FAN_MINIMUM_PWM 10, FAN_ZERO_RPM_ENABLE 11, FAN_ZERO_RPM_STOP_TEMP 12
  ```

  The printout uses `GfxclkFoffset` (a clock offset), `UclkFmin/UclkFmax`, `VoltageOffsetPerZoneBoundary[0]` (voltage offset), and PPT percent `OverDriveTable.Ppt`. If the PMFW's `FeatureCtrlMask` limits are zero, `smu->od_enabled = false`. Overdrive must be enabled at boot (`amdgpu.ppfeaturemask` bit 14) in practice (INFERRED from kernel docs knowledge, not read).
- Power cap (`power1_cap`): `smu_v14_0_2_init_ppt_limits()` builds default/min/max: `default_value = SocketPowerLimitAc[PPT0]` (or `Dc`), `max = max(pp_limit, MsgLimits.Power)`, `min = min(...) * (100 + od_percent_lower) / 100`, `od_max = max_limit * (100 + od_percent_upper) / 100`. So the card's pptable fixes the default and the OD percent range (READ). `smu_v14_0_2_get_ppt_limit` multiplies by `(100 + OverDriveTable.Ppt) / 100`.
- Recent commits on this: "drm/amd/pm: fix smu14 power limit range calculation" (2026-07-08), "drm/amd/pm: account for OD percentage in effective PPT limits" (2026-08-06), "drm/amd/pm: preserve committed OD features for SMU 14.0.2" (2026-10-27), "drm/amd/pm: add zero fan stop temperature support on smu 14.0.2" (2026-10-27).

---

## 6. Firmware file names

READ in `MODULE_FIRMWARE` lines (kernel 7.4-rc3 master):

| Block | Files |
|---|---|
| GFX (`gfx_v12_0.c`) | `amdgpu/gc_12_0_0_{pfp,me,mec,rlc,toc}.bin`; `amdgpu/gc_12_0_1_{pfp,me,mec,rlc,toc}.bin`; `amdgpu/gc_12_0_1_rlc_kicker.bin` |
| IMU (`imu_v12_0.c`) | `amdgpu/gc_12_0_0_imu.bin`, `gc_12_0_1_imu.bin`, `gc_12_0_1_imu_kicker.bin` |
| MES (`mes_v12_0.c`) | `amdgpu/gc_12_0_{0,1}_mes.bin`, `_mes1.bin`, `_uni_mes.bin` |
| SDMA (`sdma_v7_0.c`) | `amdgpu/sdma_7_0_0.bin`, `sdma_7_0_1.bin` |
| PSP (`psp_v14_0.c`) | `amdgpu/psp_14_0_2_sos.bin`, `psp_14_0_2_ta.bin`, `psp_14_0_3_sos.bin`, `psp_14_0_3_sos_kicker.bin`, `psp_14_0_3_ta.bin`, `psp_14_0_3_ta_kicker.bin` (plus `psp_14_0_5_*` for an APU) |
| SMU (`smu_v14_0.c`) | `amdgpu/smu_14_0_2.bin`, `smu_14_0_3.bin`, `smu_14_0_3_kicker.bin` |
| VCN / JPEG | `MODULE_FIRMWARE` lines not found in `vcn_v5_0_0.c` or `jpeg_v5_0_0.c`; the VCN firmware names are probably in a shared file (`amdgpu_vcn.c`). Not found. |

The "kicker" files are a firmware variant: `gc_12_0_1_*_kicker` is selected for some boards with GC 12.0.1. The commit "drm/amdgpu: add kicker fws loading for gfx12/smu14/psp14" (2025-07-28) added them. The exact selection rule is in code I did not read; INFERRED: it depends on a revision or SKU check. Practical meaning: Navi 48 boards can need newer firmware in `linux-firmware`.

Firmware versions that matter to features (READ, section 3): MEC >= 2090 for atomics without PCIe atomics, `me 2480/pfp 2530/mec 2680/mes 100` for cleaner shader, `me 2660/mec 2920` for per-queue reset, `me 3090/pfp 3190/mec 3450/mes 147` for user queues. Also "fix gpu idle power consumption issue for gfx v12" (2026-03-06) says: "Older versions of the MES firmware may cause abnormal GPU power consumption. When performing inference tasks on the GPU (e.g., with Ollama using ROCm), the GPU may show abnormal power consumption in idle state and incorrect GPU load information. This issue has been fixed in firmware version 0x8b and newer."

---

## 7. Known problems, fixes and workarounds (commit titles with dates)

Dates come from GitHub per-file commit feeds on master. Bodies I read are quoted; others are titles only.

Hangs, resets, scheduling:

- 2026-08-06 `drm/amdgpu/mes12: fix dropped dispatches under queue oversubscription`. READ body: with more ready queues than hardware queue slots, MES maps a queue when its doorbell rings; with "unmapped-doorbell mode 1 (basic)" a ring "does not read the per-queue bitmap or clear the ringing queue's work_done. A queue still marked done from a prior dispatch is skipped and its new work is never fetched: the completion never signals, while queue creation succeeds and no dmesg error is reported." Fix: use mode 2 (basic+). Relevant to anyone running many compute queues/streams on a GC 12 card: silent lost dispatches.
- 2026-03-30 `drm/amdkfd: Fix queue preemption/eviction failures by aligning control stack size to GPU page size`. READ body: a control stack aligned to 64 KB caused "Queue preemption failed ... Failed to evict process queues ... GPU reset begin!" in RCCL tests (observed on 64 KB-page systems). Generic KFD, but the sizing code is the one gfx1200/1201 use.
- 2026-09-23 `drm/amdgpu/mes12: clear event log on resume to avoid MES page fault`. READ body: stale MES log region causes a recoverable CPC page fault on resume with `mes_log_enable=1`.
- 2026-10-01 `drm/amdgpu/gfx12: implement detect_hung_queue`, `drm/amdgpu/gfx12: defer gfx pipe reset enablement`, `drm/amdgpu/mes12: gate gfx pipe reset on PER_PIPE`, `drm/amdgpu/mes12: restore collateral gfx queues after pipe reset` (titles only): queue and pipe reset work is still moving.
- 2026-06-17 `drm/amdgpu/mes12: enable compute MMIO pipe reset` (title only).
- 2026-11-04 `drm/amdgpu: Report MES ring wait timeouts.` (title only).
- 2026-08-06 `drm/amdgpu/gfx12: add priv-fault user-queue recovery worker` (title only).
- 2024-08-23 `drm/amdgpu/gfx12: set UNORD_DISPATCH in compute MQDs`. READ body: "This needs to be set to 1 to avoid a potential deadlock in the GC 10.x and newer ... This can lead to hangs in some mixed graphics and compute workloads." Closes freedesktop drm/amd issue 3575. This is already in the MQD code in section 3.1.
- 2026-02-19 `drm/amdkfd: Disable MQD queue priority` (title; follow-up "Removed commented line for MQD queue priority" 2026-02-25). So compute queue priority via the MQD is turned off in the GC 12 manager.
- 2026-11-04 `drm/amdgpu/gfx12: disallow GFXOFF while updating the SPM VMID`. READ body: on "navi48", the RLC_SPM_MC_CNTL update does not stick in GFXOFF; a job reads 0 instead of the reserved VMID; related to page faults and secure/reserved-VMID tests.

Power and idle state:

- 2026-03-06 `drm/amdgpu: fix gpu idle power consumption issue for gfx v12`: see section 6; fixed in MES firmware 0x8b and newer; closes ROCm issue 5706.
- 2025-12-16 `drm/amd/pm: restore SCLK settings after S0ix resume` (title only; SMUv14 file).
- 2026-01-27 `drm/amd/pm: fix smu v14 soft clock frequency setting issue` (title only).
- 2025-02-27 `drm/amd/pm: add fan abnormal detection` (title only).
- 2026-06-03 `drm/amd/pm: mark metrics.energy_accumulator is invalid for smu 14.0.2` (title only): the energy counter on SMU 14.0.2 is not usable, relevant when you compute joules per op.
- 2026-04-03 `drm/amd/pm: correct mem_busy_percent display due to calculation errors` (title only).

Memory / VM:

- 2026-10-01 `drm/amdgpu/gmc12: properly pass flush_type to gmc_v12_0_flush_vm_hub()`; 2026-09-02 `drm/amdgpu/gmc12: Pass cam_index to retry fault handler`; 2026-08-25 `drm/amdgpu: Update no-retry PTE flags for GFX12` (titles only).
- 2026-07-14 `drm/amdkfd: fix 32-bit overflow in CWSR total size calculation` and 2026-07-28 `drm/amdkfd: fix uint32_t overflow in EOP ring buffer size alignment` (titles only).

A reader's note: I could not run a text search of git history for "Navi 48" or "Navi 44". The feed-based list above is by file, so fixes that are in other files (display, ttm, firmware loading) are missing.

---

## 8. User space: ROCR runtime and CLR (HIP)

### 8.1 ROCR (`rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/`)

- ISA registry (`isa.cpp`, READ): `ISAREG_ENTRY_GEN("gfx1200", 12, 0, 0, unsupported, unsupported, 32, "gfx12-generic")`, `("gfx1201", 12, 0, 1, ..., 32, "gfx12-generic")`, `("gfx1250", 12, 5, 0, ..., 32, "gfx12-5-generic")`. The parameter `32` is the wavefront size. So both chips are registered as wave32 and share the `gfx12-generic` target (code objects built for `gfx12-generic` run on both).
- Workgroup limits are hard-coded and not per device: `HSA_AGENT_INFO_WORKGROUP_MAX_DIM = {1024, 1024, 1024}` and `HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 1024` with `// TODO: must be per-device` (`amd_gpu_agent.cpp`).
- `HSA_AGENT_INFO_WAVEFRONT_SIZE` returns `properties_.WaveFrontSize` (from KFD). `HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU` returns `properties_.NumSIMDPerCU`; `HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = NumSIMDPerCU * MaxWavesPerSIMD`; `HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = NumFComputeCores / NumSIMDPerCU`; `HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = NumShaderBanks`. READ.
- Cache line: `HSA_AMD_AGENT_INFO_CACHELINE_SIZE` returns the line size of the level-2 cache from KFD; `// Fallback for when KFD is returning zero. *value = 256;`. READ.
- Scratch: `#define MAX_WAVE_SCRATCH_GFX12 67106816 // 2MB stack size per wave` and `max_wave_scratch_ = (...GetMajorVersion() >= 12) ? MAX_WAVE_SCRATCH_GFX12 : MAX_WAVE_SCRATCH;`. READ. So gfx12 allows a larger per-wave scratch than earlier parts. (The comment says 2 MB per wave while the number is about 64 MiB; I could not reconcile them, likely 32 lanes x 2 MiB. INFERRED.)
- Trap handler selection table has `// GFX12_TODO: Using one for GFX10 for now.` for the V1 trap handler row (`kCodeTrapHandler10` for gfx12) and a dedicated `kCodeTrapHandlerV2_12` row. Blit kernels have separate gfx12 variants: `kCodeCopyAligned12`, `kCodeCopyMisaligned12`, `kCodeFill12`. READ.
- Blit shader kernel descriptor: `vgpr_gran = 4` for gfx12.0 (16 only for 12.5). Matches the kernel handler's "(vgpr_size + 1) * 4". READ.
- "A0" silicon: `if (properties_.Capability.ui32.ASICRevision == 0 && !...disable_gfx12_strict())` the agent switches to the `-strict` ISA variant by name. In the registry I only saw a strict variant for gfx1250. For gfx1200/1201 `strict_isa` is null (no such registered ISA), so they keep the base target. READ the code; the "null for 1200/1201" is INFERRED from the registry lines I grepped (`gfx1250-strict` only).
- Fine-grain memory: `if (!(Major == 12 && Minor == 0)) regions_.push_back(... extended fine-grain region ...)`. So gfx1200/1201 do NOT expose an extended-scope fine-grained VRAM region (the same rule appears in CLR: "gfx120x does not support extended-scope fine-grained memory. So explicitly force the uncached flag."). READ.
- Workgroup clusters and extended AQL dispatch are only set for `Major == 12 && Minor >= 5` (`extended_aql_dispatch_supported_`, `workgroup_clusters_supported_`). Not for 12.0.

### 8.2 CLR / HIP (`rocm-systems/projects/clr/`)

ISA table, `rocclr/device/device.cpp` (READ). The columns follow the `Isa` constructor in `device.hpp` (`simdPerCU, simdWidth, simdInstructionWidth, memChannelBankWidth, localMemSizePerCU, localMemBanks, ldsAlignment`):

```
{"gfx1200", true, true, 12, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
{"gfx1201", true, true, 12, 0, 1, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
{"gfx1100", true, true, 11, 0, 0, NONE, NONE, 2, 32, 1, 256, 64 * Ki, 32, 1024},
{"gfx1250", ..., 4, 32, 1, 256, 320 * Ki, 64, 1024},
```

Meaning for gfx1200/gfx1201: 2 SIMDs per CU, SIMD width 32, instruction width 1, memory channel bank width 256 bytes, LDS 64 KiB per CU (65,536 bytes), 32 LDS banks, LDS alignment 1024. The table is identical for gfx1100 and gfx1030 (RDNA 2/3/4 share it). xnack and sramecc are both `NONE` (feature not present).

Settings (`rocclr/device/rocm/rocsettings.cpp`, READ):

- `if (gfxipMajor >= 10) { enableWave32Mode_ = true; ... enableWgpMode_ = GPU_ENABLE_WGP_MODE; }` (WGP mode is disabled only for 12.5). So HIP compiles for wave32 by default and, by default, `GPU_ENABLE_WGP_MODE` decides whether CUs are counted per WGP (`maxComputeUnits_ / 2`, `simdPerCU_ *= 2`, `max_waves_per_cu *= 2`).
- `barrier_value_packet_ = true; queue_pipe_dist_ = dynamic_queues_ >= 1;` for gfx12+.
- `setKernelArgImpl()`: for gfx12+ `isGfx12Plus` true: `kernelArgImpl = KernelArgImpl::DeviceKernelArgsReadback` ("Implement the kernel argument readback workaround (write all args -> sfence -> write last byte -> mfence -> read last byte)") when `pcieStoresNeedFlush`, and `kernel_arg_impl_ = kernelArgImpl; kernel_arg_opt_ = true; aql_device_ring_buf_ = DEBUG_CLR_AQL_DEV_QUEUE > 0;`. So on gfx12 HIP puts kernel arguments in device memory and uses the readback trick for write-ordering over PCIe. Kernel-argument size limit: I did not find a numeric limit in the files I read.
- `gwsInitSupported_ = false` for gfx > 10.

Device properties HIP reports (`rocclr/device/rocm/rocdevice.cpp`, `hipamd/src/hip_device.cpp`, READ):

- `deviceProps.warpSize = info.wavefrontWidth_` (from ROCR; 32 in wave32 mode).
- `sharedMemPerBlock = info.localMemSizePerCU_` (= `isa().localMemSizePerCU()` = 65,536 for gfx1200/1201); `sharedMemPerBlockOptin` the same; `maxSharedMemoryPerMultiProcessor = ldsPerMultiprocessor` (computed in lines I did not fully read, may double in WGP mode).
- `regsPerBlock = info.availableRegistersPerCU_ = vgprsPerSimd_ * simdPerCU_ * wavefrontWidth_`, where `vgprsPerSimd_` comes from the ISA metadata key `TotalNumVGPRs` (`getValueFromIsaMeta`). The metadata value for gfx1200/1201 is in the compiler component and was not read.
- `multiProcessorCount = info.maxComputeUnits_` (KFD CUs, or half in WGP mode); `maxThreadsPerMultiProcessor = wavefrontWidth_ * max_waves_per_cu`; `maxBlocksPerMultiProcessor = maxThreadsPerCU_ / wavefrontWidth_`.
- `l2CacheSize = info.l2CacheSize_` from `HSA_AGENT_INFO_CACHE_SIZE[1]` (a FIXIT comment says "Seems like hardcoded in HSA backend so 0": it may be 0).
- Cache line: `globalMemCacheLineSize_ = HSA_AMD_AGENT_INFO_CACHELINE_SIZE` or 64 if 0; raised to 256 only for 12.5 and up. So gfx1200/1201 report whatever KFD says for L2 (or 256 via ROCR's fallback; or 64 in CLR if 0). READ.
- `memoryBusWidth = vramBusBitWidth_`; `globalMemChannels_ = vramBusBitWidth_ / 32`.
- `imagePitchAlignment_ = (isa().versionMajor() >= 12) ? 128 : 256;`.
- `deviceProps.arch.hasDynamicParallelism = 0`, `hasFunnelShift = 0`, `hasSyncThreadsExt = 0`, `hasSurfaceFuncs = 0`; atomics flags all 1; `hostNativeAtomicSupported = info.pcie_atomics_`. `gcnArchName` is `isa.targetId()` (for example `gfx1201`). READ.
- Uncached coherent memory: `if (uncached && isa().versionMajor() == 12 && isa().versionMinor() == 0)` the flag is forced (two places) because gfx120x has no extended-scope fine-grain. READ.
- AQL queue ring placement and kernarg in device memory by default on gfx12+ (above). CU masks are merged with a WGP-mode expansion (`rocdevice.cpp` ~3763-3790).

---

## Could not find

- Numeric cache sizes (L0/L1/L2/Infinity Cache) and line sizes for GC 11.0.0 or GC 12.0.x. The kernel builds them at runtime from the card's discovery table (section 2). There is no per-chip table.
- The IP versions (SMU, PSP, SDMA, VCN, JPEG, DCN, NBIO, MP0/MP1) each Navi 48 / Navi 44 card reports. Only the driver's accepted-version switches are visible. The mapping of 14.0.2 vs 14.0.3 to a specific chip was not found.
- Actual `gc_max_waves_per_simd`, `gc_lds_size`, `gc_num_*` values: not in source. The CWSR sizing suggests 16 waves per SIMD and 64 KiB LDS per CU (INFERRED).
- The definition of `NUM_SIMD_PER_CU` (value 2 inferred from CLR's table).
- Hardware dynamic VGPR: no enable found, but I could only read selected files and had no repo-wide search (GitHub code search not available unauthenticated). Treat "not enabled in kernel" as strong but not exhaustive.
- Default clock/voltage/power numbers (they live in the on-card PMFW pptable), and thermal limits other than the fallback 99 C / 120 C policy table.
- VCN/JPEG firmware names (not in the VCN v5 files I looked at).
- Chip-name to GC-version link for Navi 48/44 in primary source (only inferred from pci.ids names plus standard gfx names).
- Kernel-argument size limits and AQL queue size limits in CLR/ROCR for gfx12 (not located).
- The `HIP`/`ROCR` ISA metadata (`TotalNumVGPRs`, `AddressableNumVGPRs`) for gfx1200/1201: kept in the compiler's metadata, not read.
- A git-history search by keyword (Navi 48, Navi 44, SMU 14): not possible without API access; section 7 is built from per-file feeds, so fixes in other directories (display, firmware loader, TTM, DC) are missing.
- Text of several commit bodies (those marked "title only").

---

## What this implies for compute kernels

1. Treat each CU as half a WGP. KFD counts CUs as WGP x 2 (`gfx_v12_0_get_cu_info`), while HIP, in WGP mode, halves the count again and doubles SIMDs per "CU". Use `multiProcessorCount` knowing which mode your build uses (`GPU_ENABLE_WGP_MODE`). Rates per "CU" differ by a factor of two between these views.
2. The VGPR file is 192 KiB per SIMD (1536 x 32 lanes x 4 B), derived from KFD saving 0x60000 bytes per 2-SIMD CU (INFERRED from READ constants). Only gfx1100/1101/1151/1200/1201 use this larger size in KFD's table. That affects how many waves you can keep resident at a given VGPR count; the allocation granularity is 4 VGPRs on these chips (READ in trap handler and ROCR).
3. Wave slots: KFD sizes for 32 waves per CU (16 per SIMD). Occupancy above that is not available. LDS per CU: 64 KiB in both KFD and CLR (verify with `lds_size_in_kb` on the live node; this is the only place the kernel actually reports the hardware value).
4. Wave32 is the default HIP compile mode, but wave64 is still a hardware mode (the trap handler saves WAVE64 state). Code objects are `gfx12-generic` compatible.
5. L2 and Infinity Cache sizes are not in source; read them from `/sys/class/kfd/kfd/topology/nodes/*/caches/` or `rocminfo` on the machine. `hipDeviceProp.l2CacheSize` may come back 0 (CLR has a FIXIT comment about it).
6. `AMDGPU_PTE_NOALLOC` is the page-level control for "do not allocate in Infinity Cache"; default mappings are `MTYPE_NC`. Fine-grain, extended-scope VRAM is not offered on gfx1200/1201, so system-scope coherent atomics need the uncached path.
7. Context-save cost: a compute preemption has to save roughly 468 KiB per CU (INFERRED: 0x75000 bytes), tens of MB per queue. Long-running kernels that are preempted (time slicing, another process, debugger) pay this. Control stack size alignment bugs have caused eviction hangs (section 7).
8. Many queues and streams: on a kernel without the 2026-08-06 MES12 fix (it is in master as of this read; I did not check which release carries it), oversubscribing the 8 hardware queue slots (2 pipes x 4 queues) can lose dispatches silently. Keep concurrent HW queues at or below 8 for stability on older kernels, or run a kernel containing that fix.
9. Power: the ROCm path switches on the COMPUTE profile, which disables clock deep sleep (more consistent latency, a little more idle power). Using graphics rings (Vulkan, or kernel-submitted work) selects FULLSCREEN3D instead. Do not benchmark compute with overdrive off and the profile set by hand and expect the same clocks as the auto profile. PPT (power cap) range and defaults are in the card's pptable and exposed as `power1_cap`; the OD range uses percent of that limit.
10. Firmware matters: MES firmware older than 0x8b can keep the GPU's idle power high after ROCm inference; per-queue reset, the cleaner shader and user queues only turn on at the firmware versions listed in section 6. Check `dmesg` firmware versions before attributing performance differences to the kernel.
11. Kernel arguments: on gfx12 HIP places kernel args in device memory and uses a write-readback sequence for PCIe ordering (small extra latency per launch); small-kernel launch rate is therefore bound by this path. AQL rings may also live in device memory.
12. Dynamic VGPR is a hardware feature visible in debug registers (`SQ_WAVE_DVGPR_ALLOC_*`), but the kernel's save/restore code does not handle it. Do not rely on it for kernels that can be preempted.
13. Page faults: GPU retry faults are handled in the driver, but there is no XNACK flag in the gfx1200/1201 ISA table (`xnack NONE`). Plan on all memory being resident (no on-demand migration in kernels).
