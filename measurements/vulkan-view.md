# Lane 9: the RX 9070 XT through Vulkan compute

Date: 2026-11-17. Card: RX 9070 XT (gfx1201, PCI 1002:7550), Mesa RADV 27.0.0-devel with its ACO compiler, kernel
7.3.4. Run: `runs/20261117T141203Z-vulkan/` (every job PASS, all three deliberate breaks FAIL as intended, highest
hotspot 87 C). The lead reran the whole script (`runs/20261117T141716Z-vulkan/`): every verdict the same, FP16
210.1, FP8 420.1 TFLOPS, 8-bit 421.4 TOPS, read 636.5 GB/s, highest hotspot 87 C. This lane was written in a staging
folder outside the repository and copied in unchanged by the lead. After the review the lead changed two checks in
`vkstudy.c` (the cross-check prints the random sets' error separately; the load ceilings are 650 GB/s and 9.5
TB/s) and reran everything once more (`runs/20261117T142940Z-vulkan/`): all verdicts PASS, three breaks FAIL,
highest hotspot 88 C. Wall-clock numbers: LACT profile "Testing" (stock clocks, 317 W cap), memory 1258 MHz, a 1.2x
regulator-level power-limit mod is in the card, so the reported power is scaled down and is not true board power.

## In short

- Vulkan reaches the same hardware limits as HIP where both were measured the same way, except FP8 loaded from
  DRAM (last sentence of this point). Matrix instructions with
  operands in registers: FP16 210.6, BF16 210.5, FP8 421.0 TFLOPS and 8-bit integer 421.3 TOPS (HIP: 211, 211, 420,
  420), which is 128 SIMDs x the clock the same shader's cycle counter shows / 16 or 8 cycles within 0.2 to 0.3 %
  (that ratio shows no lost time; it is not an independent check of the absolute rate). Read bandwidth 636 GB/s
  (HIP 633). FP32 FMA 25.2 T FMA/s with dual issue (HIP 24.7). The naive load-then-multiply kernel: 5.06 TFLOPS FP16
  from DRAM at 633 GB/s (HIP 5.1) and 29 to 33 TFLOPS from a 4 MiB window (HIP 19 to 38 from the caches, too wide
  a range to compare closely). One gap: FP8 from DRAM reaches 9.2 TFLOPS (574 GB/s) against HIP's 10.1 to 10.2 with
  4 subgroups per SIMD; with 8 it reaches 10.1 (631 GB/s, the bus): too little data in flight (item 5).
- The difference is the compiler and the driver's defaults, not the hardware:
  - RADV compiles compute shaders as wave64 whenever the work-group has 64 or more invocations, matrix shaders
    included. Wave64 matrix code with operands in registers runs at full rate (one 16x16x16 instruction per 16 or
    8 cycles per SIMD, as in wave32), so wave64 costs nothing there. In the load-per-tile kernel from a 4 MiB
    window wave64 is 8 % slower than wave32 (29.35 against 31.80 TFLOPS); from DRAM they are equal.
  - ACO keeps a uniform matrix (one value everywhere) in scalar registers and copies it into vector registers
    inside the loop: 4 `v_mov` per 16 `v_wmma`, 2 % (wave64) to 1.4 % (wave32) slower. A and B loaded once from
    memory give a loop with only the 16 `v_wmma`.
  - ACO formed dual-issue FMA pairs in wave32 when the multiplicands were per-lane values (kept listing). An
    earlier build with uniform multiplicands and another operand order gave single `v_fma_f32` and half the rate;
    its listing was not kept and it changed two things at once, so which of the two matters is not established.
  - Cooperative-matrix loads get one transposing load (`global_load_tr_b128`, or `_b64` for 8-bit) and one plain
    `buffer_load` per tile in every layout. ACO puts no shuffles around them, but in the loop it reloads the
    descriptors, recomputes the per-lane address (two `v_mul_lo_u32`) and waits for both loads before every
    `v_wmma`: nothing is in flight across tiles.
- Wave size does not change the arithmetic: the FP16 cross-check gives identical bits in wave32 and wave64.
- Two observations for other lanes (MEASURED here, not this lane's subject):
  - Matrix arithmetic (lane 5): an FP32 accumulator of 2^22 or more (FP16, BF16) or 2^23 or more (FP8) loses 16
    products of 1/16 entirely. 2^22 + 16 x (0.25 x 0.25) gives 2^22, not the representable 2^22 + 1. Lane 5's model
    explains it (item 6), and reproduces every result word of this lane's cross-check.
  - Scheduling (lane 7): with several matrix waves per SIMD, a few waves took nearly all the work (which wave is
    oldest and where it sits cannot be read through Vulkan). With
    work stealing, the median wave got 1 chunk and the busiest about 93 % of its SIMD's share.

## 1. What the driver reports (READ through the API; `info.log`)

| Property | Value |
|---|---|
| Driver | radv, Mesa 27.0.0-devel, conformance 1.4.5.3, API 1.4.366 |
| Cooperative matrix | 20 entries, all 16x16x16, subgroup scope, compute stage only; same table as `coopmat-radv-9070xt.txt` (FP8 E4M3/E5M2 all four pairs -> FP32; u8/s8 pairs -> u32/s32, saturating only with signed result; FP16 -> FP16/FP32; BF16 -> BF16/FP32); robust buffer access not supported |
| Feature bits | shaderFloat8 1, shaderFloat8CooperativeMatrix 1; shaderBFloat16Type 1, DotProduct 1, CooperativeMatrix 1; shaderSubgroupClock 1, shaderDeviceClock 1; pipelineExecutableInfo 1 |
| Subgroup size | default 64; min 32, max 64; required size allowed in fragment, compute, task, mesh; maxComputeWorkgroupSubgroups 4294967295 |
| Compute limits | work-group size 1024 x 1024 x 1024, 1024 invocations; work-group count 4294967295 x 65535 x 65535; 64 KiB shared memory; 256 bytes of push constants |
| Timestamps | period 10 ns, valid on compute queues (family 1, compute only, was used) |
| Memory heaps | 15.92 GiB device-local, 15.11 GiB system |

## 2. Wave size (`wave.log`, `asm/`)

| Shader | Work-group | Required size | Driver's choice (pipeline property) = `gl_SubgroupSize` seen at run time |
|---|---|---|---|
| plain | 32 | none | 32 |
| plain | 64, 128, 256, 1024 | none | 64 |
| plain | 64, 256 | 32 | 32 |
| plain | 64, 256 | 64 | 64 |
| cooperative matrix | 32 | none | 32 |
| cooperative matrix | 64, 128, 256 | none | 64 |
| cooperative matrix | 64 | 32 / 64 | 32 / 64 |

MEASURED: the run-time `gl_SubgroupSize` matched the pipeline's reported size in all 16 cases, and every required
size was honoured. Machine code of the same FP16 multiply-add (`wave_coop`):

```
wave32: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[0:3], 0     ; A, B 4 registers, C/D 8
wave64: v_wmma_f32_16x16x16_f16 v[4:7],  v[0:1], v[0:1], 0     ; A, B 2 registers, C/D 4
```

The wave64 form is the same 16x16x16 instruction with half the registers per lane. Its timing is in item 4.

## 3. Known answers (`bw.log`, `fma_sg_*.log`)

| Probe | Vulkan, MEASURED (median, min-max) | HIP | Verdict |
|---|---|---|---|
| Read, 1 GiB random data, 16-byte loads, 4 in flight, wave64 | 636.4 GB/s (635.0-637.4), 24 dispatches | 633 GB/s | within 10 % of 633: PASS |
| FP32 FMA, wave32, 8 `v_dual_fmac_f32` per 16 FMAs | 25.22 T FMA/s = 50.44 TFLOPS (50.09-50.89) at 3174 MHz; 0.970 (0.965-0.970) of 128 x 64 FMAs x clock | 24.7 T FMA/s, 0.945 at 3191 MHz (dual issue, W = 8) | within 0.90-1.02 of the machine-code rate: PASS |
| FP32 FMA, wave64, 16 `v_fmac_f32_e32` | 24.72 T FMA/s = 49.43 TFLOPS (49.36-49.60) at 3197 MHz; 0.944 (0.944) of 128 x 64 x clock | - | PASS |

- Wave64 FMA: one instruction covers 64 lanes per cycle (INFERRED from 0.944 of 64 FMAs per SIMD per cycle), the
  same rate as wave32 with every FMA paired. Wave64 cannot encode the dual-issue form.
- A first build used `x = fma(x, a, b)` with uniform `a`, `b`; ACO emitted 16 VOP3 `v_fma_f32` with scalar
  operands even in wave32: no dual issue, so half the wave32 rate. The `asm/` of that build was not kept, and it
  differs from the final probe in both the operand order and the uniform operands, so this is an observation, not
  a rule. The probe uses `x = fma(a, b, x)` with `a`, `b` loaded per lane.
- Clock: the shader's 64-bit cycle counter (`s_getreg_b32 HW_REG_SHADER_CYCLES_LO/HI`) over the 100 MHz realtime
  counter (`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`). The realtime span matched the timestamp span (ratio 0.997 to
  1.002 in every repetition), so the realtime counter runs at 100 MHz.
- FMA power: up to 352 W reported in 50 ms samples. A trial 0.1 s dual-issue FMA burst lifted the hotspot from 57 to
  91 C, so FMA repetitions were kept to about 0.05 s.

## 4. Matrix unit with operands in registers (`mat_*.log`)

A and B loaded once from memory (every element 0.25; 1 for 8-bit integers), 4 independent accumulators, 16
`v_wmma` per loop iteration, chunks of 4096 instructions handed out by work stealing. Wall time is from GPU
timestamps around the timed dispatch, which runs after a warm-up dispatch in the same submission. Clock: the shader
counter. Theoretical = 128 x clock / 16 (FP16, BF16) or / 8 (FP8, 8-bit). TFLOPS = instructions x 8192 / s. 5
repetitions of about 0.1 s each; W = subgroups per SIMD on average.

| Form | Wave | W | TFLOPS median (min-max) | HIP | Ratio to HIP | Ratio to theoretical | Counter clock | Reported power / hotspot |
|---|---|---|---|---|---|---|---|---|
| FP16 -> FP32 | 64 | 8 | 210.6 (210.2-210.7) | 211.4 | 0.996 | 0.998 | 3220 MHz | 259-261 W / 77-79 C |
| FP16 -> FP32 | 32 | 8 | 202.5 (201.9-202.6) | 211.4 | 0.958 | 0.960 | 3219 | 256-266 / 75-77 |
| FP16 -> FP32 | 64 | 1 | 209.5 (209.1-210.0) | 211.4 | 0.991 | 0.993 | 3220 | |
| FP16 -> FP32 | 32 | 1 | 209.5 (209.0-210.0) | 211.4 | 0.991 | 0.992 | 3221 | |
| FP16 -> FP32 | 32 | 16 | 202.5 (202.4-202.6) | 211.4 | 0.958 | 0.960 | 3221 | |
| BF16 -> FP32 | 64 | 8 | 210.5 (210.0-210.7) | 211.1 | 0.997 | 0.997 | 3220 | 264-274 / 76-79 |
| BF16 -> FP32 | 32 | 8 | 202.0 (202.0-202.6) | 211.1 | 0.957 | 0.957 | 3220 | |
| FP8 -> FP32 | 64 | 8 | 421.0 (419.9-421.0) | 419.5 | 1.004 | 0.997 | 3221 | 325-333 / 83-86 |
| FP8 -> FP32 | 32 | 8 | 420.9 (420.1-421.2) | 419.5 | 1.003 | 0.997 | 3223 | |
| FP8 -> FP32 | 32 | 1 | 415.7 (414.7-415.9) | 419.5 | 0.991 | 0.985 | 3221 | |
| 8-bit integer -> 32 | 64 | 8 | 421.3 TOPS (420.2-421.4) | 419.8 | 1.004 | 0.998 | 3221 | 311-318 / 78-81 |
| 8-bit integer -> 32 | 32 | 8 | 421.5 TOPS (420.4-422.6) | 419.8 | 1.004 | 0.998 | 3221 | |
| FP16, uniform splat A, B | 64 | 8 | 206.7 (206.4-206.8) | 211.4 | 0.978 | 0.980 | 3220 | |
| FP16, uniform splat A, B | 32 | 8 | 199.7 (199.3-199.8) | 211.4 | 0.945 | 0.947 | 3220 | |
| 8-bit, uniform splat A, B | 32 | 8 | 411.8 TOPS (411.7-412.6) | 419.8 | 0.981 | 0.975 | 3222 | |

Every repetition's results were exact (each lane's per-chunk totals against the host's count). All verdicts are
within 10 % of the HIP figure. "Ratio to HIP" divides by the HIP column, which holds HIP's rate at one wave per
SIMD (its best); the program's own verdict lines compare against the rounded 211 and 420.

Machine code (`asm/mat_f16_sg0.s`, `asm/mat_f16_splat_sg0.s`, `compile.log`):

```
loaded A, B (all four types, both wave sizes): 16 x v_wmma_*, s_add_co_u32, s_wait_alu, s_cmp_ge_u32, s_cbranch_scc0
    v_wmma_f32_16x16x16_f16 v[12:15], v[2:3], v[4:5], v[12:15]
    v_wmma_i32_16x16x16_iu8 v[8:11], v2, v4, v[8:11] neg_lo:[1,1,0]     ; signed A and B
splat A, B (FP16, BF16): 4 x v_mov_b32 (wave32: v_dual_mov_b32) from s3/s4 + 1 s_delay_alu + 16 x v_wmma
splat A, B (8-bit): 2 moves; (FP8): no moves
```

No V_NOP, no extra wait, no copy of an accumulator, no spill (`compile.log`: 36 VGPRs in wave64 and 72 in wave32
for the FP16 and BF16 loops, 24 and 48 for FP8 and 8-bit).

- With operands in registers wave64 costs nothing. Wave32 with 8 or 16 subgroups per SIMD loses 4 % for FP16 and
  BF16, but not for FP8 or 8-bit, and not with one subgroup per SIMD. HIP showed the same direction with a smaller
  size: FP16 0.977-0.988 with W > 1 (1 to 2 % lost), FP8 1.000.
  Cause not established. INFERRED candidate: an occasional instruction from a starved younger wave breaks the
  accumulator forwarding (HIP item 5: 2 cycles), and it hits the 16-cycle forms differently.
- The splat cost (2 % in wave64) is the 4 moves and the delay per 16 instructions, matching HIP's "vector
  instruction writing A, B or C right before costs 3".
- Placement: Vulkan gives no way to read which SIMD a wave runs on, so W is an average. W = 1 may put two subgroups
  on one SIMD; with work stealing that costs little (0.993).
- Chunks per subgroup (W = 8, FP16 wave64): min 0, median 1, max about 4828, where the average per SIMD is about
  5200. A few subgroups did nearly all the work, and up to 200 of 1024 subgroups got none. INFERRED: one wave per
  SIMD, the oldest, wins the matrix unit, as in HIP item 9 (neither a wave's age nor its SIMD can be read here).

## 5. Cooperative-matrix loads (`load_*.log`)

The table is from the owner's run (`runs/20261117T141203Z-vulkan/`); the two later runs agree within 0.2 % except
FP8 from DRAM, which was 9.15, 9.19 and 9.41 TFLOPS (572 to 588 GB/s) across the three runs.
Each tile: `coopMatLoad` A, `coopMatLoad` B, one multiply-add into one accumulator (the naive form). Data: random -1,
0, 1. Check: the sum of every output element against sum over tiles of colsum(A) . rowsum(B), computed on the host
from the buffers with the layout applied. Tight = 16 x 16 tile stored alone (stride 16); wide = tile inside rows of
4096 elements. Window = bytes of A plus B the chunks cycle through: 4 MiB (L2), 32 MiB (memory-side cache), 1 GiB
(DRAM, four passes). Wave64 unless marked, W = 4. 5 repetitions.

| Type, layout, stride | Source | TFLOPS median (min-max) | Operand GB/s | HIP |
|---|---|---|---|---|
| FP16 row, tight | L2 (4 MiB) | 29.35 (29.34-29.36) | 3668 | cache 19-30 |
| FP16 row, tight, wave32 | L2 | 31.80 (31.80-31.81) | 3976 | |
| FP16 col, tight | L2 | 29.43 (29.42-29.45) | 3679 | |
| FP16 row, wide | L2 | 29.54 (29.53-29.55) | 3693 | |
| FP16 row, tight | memory-side cache (32 MiB) | 18.80 (18.73-18.96) | 2350 | 18.9-30.1 (32 MiB) |
| FP16 row, tight | DRAM | 5.06 (5.06-5.06) | 633 | 5.0-5.1 (630-638 GB/s) |
| FP16 col, tight | DRAM | 5.06 (5.06-5.06) | 633 | |
| FP16 row, tight, wave32 | DRAM | 5.06 (4.85-5.27) | 633 | |
| FP16 row, wide | DRAM | 4.11 (4.09-4.12) | 513 | |
| FP16 col, wide | DRAM | 3.95 (3.92-3.96) | 494 | |
| FP8 row, tight | L2 | 33.26 (33.16-33.36) | 2079 | cache 31-38 |
| FP8 row, tight | memory-side cache | 22.03 (22.03-22.03) | 1377 | |
| FP8 row, tight | DRAM | 9.19 (9.18-9.21) | 574 | 10.1-10.2 (629-638 GB/s) |
| FP8 col, tight | DRAM | 9.25 (9.22-9.33) | 578 | |
| FP8 row, wide | DRAM | 8.04 (8.00-8.07) | 503 | |

Machine code (`asm/load_f16_row_sg0.s`, 39 instructions per tile):

```
s_load_b128 s[20:23], s[16:17], 0x20      ; descriptors reloaded every tile
s_load_b64  s[16:17], s[16:17], 0x30
v_mbcnt_lo/hi, v_and, v_lshrrev, v_mul_lo_u32 x2, ...   ; per-lane address recomputed every tile
buffer_load_b64 v[8:9], v7, s[20:23], null offen        ; one operand: plain load
global_load_tr_b128 v[4:5], v10, s[16:17]               ; other operand: transposing load
s_wait_loadcnt 0x0                                      ; both waited for before the multiply
v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[4:5], v[0:3]
```

- Which operand gets the transposing load: row-major B and column-major A (in the listing above A is `v[8:9]`
  from the plain load and B is `v[4:5]` from `global_load_tr_b128`; the column-major listing is the mirror).
  Row-major A and column-major B use a plain `buffer_load` (b64 in wave64, b128 in wave32; b32 / b64 for FP8). There is never a shuffle or permute, so
  both layouts cost the same (29.35 against 29.43 TFLOPS from L2, 5.06 against 5.06 from DRAM).
- From DRAM, FP16 tight tiles run exactly at the bus (633 GB/s, the same as HIP). FP8 reaches 574-578 GB/s (90 %),
  and wide strides 494-513 GB/s. For wide strides, INFERRED: each 16-row tile touches 16 separate 32-byte (FP16) or
  16-byte (FP8) row pieces. Tight FP8 tiles: explained by the data in flight (timing leftovers pass, run
  `runs/20261117T175731Z-timing/`, MEASURED, repetitions of 7 ms, hotspot 48 C): the loop waits for its two loads every tile,
  so each subgroup has one tile pair in flight, and an FP8 pair is half the bytes of an FP16 pair. With more
  subgroups per SIMD FP8 reaches the bus:

| FP8 row, tight, DRAM | W = 4 (as above) | W = 8 | W = 16 | W = 8, wave32 | FP16 W = 8 (control) |
|---|---|---|---|---|---|
| GB/s of operands (median of 5), run `175731Z` | 586 | 632 | 614 | 633 | 632 |
| the same, earlier run `175038Z` | 578 | 631 | 615 | 631 | 632 |

  So 4 wave64 subgroups per SIMD with 512 bytes each in flight (256 KiB for the card) are not enough, and 8 are (or
  FP16's 1 KiB per subgroup at W = 4). W = 4 varies between runs from 578 to 590 GB/s (the two reruns: 590 and 583). What would have contradicted it:
  FP8 staying near that at W = 8.
- The loop is naive: the descriptor reload, address recompute and full wait each tile mean only other waves hide
  load latency. A software-pipelined or unrolled loop was not tried (see Not measured).

## 6. Arithmetic cross-check (`check.log`, `coopmat-f16-check-check_sg{0,32,64}.txt`)

10 sets of D = A x B + C (A, B FP16 16x16, C, D FP32, all row-major). Set 0: row 0 of A = (1, 0, ...), column 0 of B
= (1, 0, ...), C[0][0] = -1, everything else random normal FP16 in +-[2^-4, 4) and C in +-4. Sets 1-7 random. Sets 8
and 9: A = B = 0.25 everywhere, C = 2^22 and 2^21.

- MEASURED: D[0][0] of set 0 = +0 (bits 0x00000000). Every element of the random sets 0-7 is within 1.04e-7 of the exact
  sum, relative to the sum of magnitudes (`runs/20261117T142940Z-vulkan/check.log`, after the program was changed
  to print the random sets separately); over all ten sets the largest is 2.38e-7 = 1 / 4194305, the lost products
  of set 8. Checksum (FNV-1a over the 2560 D words, little endian, sets in order):
  0x43d4ea15, the same for wave32, wave64 and the driver's default. The files hold A, B, C and D bits in hex for the
  lead to compare with the HIP instruction.
- MEASURED, large accumulators (C + 16 x (0.25 x 0.25), exact result C + 1):

| Input type | C = 2^21 | 2^22 | 2^22 + 2 | 2^23 | 2^23 + 4 | 16777200 |
|---|---|---|---|---|---|---|
| FP16 | exact | lost | lost | lost | lost | lost |
| BF16 | exact | lost | lost | lost | lost | lost |
| FP8 E4M3 | exact | exact | exact | lost | lost | lost |

  "Lost" means D = C: all 16 products disappeared, although C + 1 is representable. Explanation from lane 5
  (`measurements/matrix-arithmetic.md`, added by the lead): the instruction adds its products in steps (4 steps of 4
  for 16-bit inputs, 2 steps of 8 for 8-bit inputs) and rounds the running sum to FP32, nearest even, after each
  step. At C = 2^22 the FP32 spacing is 0.5; a 16-bit step adds 4 x 1/16 = 0.25, an exact tie that rounds back to
  the even value C, four times. An 8-bit step adds 0.5, which is representable, so FP8 survives at 2^22 and fails at
  2^23 (spacing 1, tie again). This limits exact counting in long matrix loops, which is why the timed shaders
  restart their accumulators every chunk.
- Model cross-check (lead, MEASURED): `src/study/arith_vkcheck.cpp` runs lane 5's CPU model, which was built from
  the HIP instruction, on the A, B, C of these files. It reproduces all 2,560 result words (wave32, wave64 and the
  driver's default) when the matrix's K index is placed K-contiguously in the operand registers; the exact sum
  rounded once matches only 1,384, and the guide's K labelling 2,050. So a model built from the HIP instruction
  reproduces 2,560 of 2,560 Vulkan result words; no HIP run used these same inputs.

## 7. Dispatch overhead (`dispatch.log`)

200 samples each after 10 warm-ups; one submission per sample, waited on with a fence. Wall = vkQueueSubmit to
fence signalled. GPU = timestamps before and after the dispatch.

| Case | Wall median (p10-p90) | GPU time median (p10-p90) |
|---|---|---|
| empty command buffer | 21.8 us (21.5-24.9) | - |
| dispatch of zero work-groups | 101.9 us (82.9-103.6) | 0.56 us (0.56-0.60) |
| empty shader, 1 work-group of 64 | 101.5 us (47.3-104.0) | 0.56 us (0.48-0.60) |
| one store, 1 work-group of 64 | 20.7 us (19.7-22.9) | 0.44 us (0.36-0.60) |
| empty shader, 1024 work-groups | 78.3 us (64.5-81.0) | 0.48 us (0.36-0.60) |
| 100 one-work-group dispatches with barriers in one command buffer | - | 1.13 us per dispatch (1.12-1.14, 5 buffers) |

- On the GPU a dispatch costs about 0.5 us, and 1.1 us back to back with a barrier.
- Submit-to-completion wall time is 20 to 100 us. It does not follow the work (zero work-groups is as slow as one,
  and one store is faster than an empty shader). INFERRED: it depends on whether the GPU dropped to an idle state
  between submissions (the sampled sclk was 90-100 MHz between jobs). The desktop shares the card, so these
  wall-clock figures are an upper range, not a constant.

## 8. Generated code for dot products and FP8 conversion (`asm/dot.s`)

| Source | ACO emits |
|---|---|
| `dot(f16vec2, f16vec2)` -> FP16 | `v_pk_mul_f16` + `v_add_f16` (two roundings, as written; no `v_dot2_f16_f16`) |
| FP16 x FP16 summed in FP32 (`float(x.x)*float(y.x) + float(x.y)*float(y.y)`) | 2 x `v_fma_mix_f32` (no `v_dot2_f32_f16`, which rounds once) |
| `dot(bf16vec2, bf16vec2)` -> BF16 | `v_dot2_bf16_bf16` |
| E4M3 / E5M2 -> FP32 | `v_cvt_f32_fp8_e32`, `v_cvt_f32_bf8_e32` (bytes placed with `v_perm_b32`) |
| FP32 -> E4M3 / E5M2 | `v_cvt_pk_fp8_f32`, `v_cvt_pk_bf8_f32` |
| saturating FP32 -> E4M3 | `v_minimummaximum_f32 v, x, 448.0, -448.0` then `v_cvt_pk_fp8_f32` (as `notes/research-10-compilers.md` read in the Mesa source) |
| FP16 -> FP32 | `v_fma_mix_f32 v, x, 1.0, -0` (a mixed FMA used as the conversion) |

READ from the listing (no timing). ACO uses the RDNA 4 conversion instructions and the BF16 dot, and keeps the FP16
forms exact to the source rather than using the fused dot instructions.

## Method

- `src/vk/vkstudy.c` (modes `info`, `compile`, `wave`, `bw`, `fma`, `mat`, `load`, `check`, `dispatch`) and
  `src/vk/vkh.c` (device, buffers, pipelines, timing, sampler). Shaders in `src/vk/shaders/`, GLSL compiled by
  `glslc` (shaderc 2026.1). It expresses the FP8 and BF16 cooperative-matrix types directly, so no SPIR-V assembly
  was needed. Every SPIR-V file passes `spirv-val`.
- Device: chosen by vendor and device id (1002:7550). The program prints it and refuses anything else. Its sysfs node
  is matched by PCI address. Work runs on the compute-only queue family (1), with no window and no swapchain.
- Machine code: this Mesa build has no `RADV_DEBUG=shaders` or `asm` (the option list in the driver binary has
  neither). The listings come from `VK_KHR_pipeline_executable_properties` (internal representation "Assembly", plus
  statistics), written to `asm/<pipeline>.s`. The program finds the hottest loop (a backward branch) and counts its
  instructions; `compile` judges the matrix and FMA loops.
- Timed jobs (items 3b, 4, 5): work stealing over chunks (one atomic per subgroup per chunk). A warm-up dispatch runs
  in the same submission so the timed dispatch starts at full clock. Without it, a 0.08 s matrix dispatch averaged
  2.65-2.95 GHz and 174-194 TFLOPS. Each repetition: wait below 60 C hotspot, sample sclk, power and hotspot every
  50 ms (stop at 95 C), check every result word.
- Checks that cannot be fooled by deleted work: matrix and FMA shaders restart their accumulators each chunk from a
  value that depends on the chunk count and add each chunk's result into integer totals, which the host recomputes
  from each subgroup's chunk count. Bandwidth: the 32-bit sum of the 1 GiB. Loads: the layout-aware checksum plus a
  ceiling on the median rate (DRAM: 650 GB/s, lane 2's bus figure of 640-644 plus 1 %; cache windows: 9.5 TB/s,
  lane 2's highest L2 rate, a loose ceiling that only catches skipped loads). Whether the 4 MiB and 32 MiB windows
  really sit in L2 and in the memory-side cache was not verified.
- Deliberate breaks (`*-break.log`), each FAIL with exit 1:

| Break | Output |
|---|---|
| `bw --expect 1266` | `FAIL median 636.4 GB/s against expected 1266.0 GB/s (+-10 %); sum check exact` |
| `mat f16 --unused` (results never stored; ACO removes every `v_wmma`, loop holds 0) | `FAIL median 34597.6 TFLOPS against the HIP figure 211 (+-10 %): 163.970; results WRONG (work deleted or miscounted)` |
| `load f16 row tight l2 --swap-check` (host reads the layout the wrong way round) | `FAIL checksum over every output element WRONG (351285248 tiles)` |

## How to rerun

```
scripts/study-vulkan.sh            # build, all jobs, breaks; results in runs/<UTC>-vulkan/
scripts/study-vulkan.sh build      # shaders and host program only
VKS_SPV=build/spv VKS_OUT=<dir> flock <gpu lock> build/vkstudy mat fp8 --sg 32 --w 1 --reps 5
```

Every job holds the shared GPU lock under `nice` with a time limit. A job does not start below 8 GB of available
memory, or unless the LACT profile is "Testing" and the card configuration is unchanged since the start of the run.
`VKS_TARGET_S` sets the load per repetition (default 0.2 s for matrix, 0.1 s FMA, 0.25 s loads, at most 1 s).

## Not measured

Still open:

- Placement of waves on SIMDs: Vulkan exposes no hardware id, so W is an average, not checked per SIMD as in HIP. Not
  reachable (leftovers C plan, VK-1).
- Sustained clocks: repetitions were about 0.1 s (matrix) and 0.05 s (FMA), shorter than the 0.2 s the brief
  suggests. The calibration dispatch overestimated the time per chunk by about 2x, and FMA heats the card fastest.
  The 50 ms sampler therefore caught only 1 to 4 samples per repetition, and its sclk figures include idle; the
  shader counter clock is the reliable value. Low value (leftovers C plan, VK-2).
  Answered in part (leftovers C, VK-2, `measurements/leftovers-c-b.md`): matrix repetitions with a 0.15 s target still
  ran 0.08 s (the calibration overestimate remains): FP16 210.0 TFLOPS and BF16 210.6 at a counter clock of 3220 MHz
  (0.995 and 0.998 of theory); the highest sampled `pp_dpm_sclk` equals the counter clock within 0.03 %, the other
  samples fall in the ramp. The 50 ms sampler is too slow to show more. MEASURED.
- A software-pipelined or unrolled cooperative-matrix load loop, so whether ACO's naive loop costs anything beyond
  the bus is open for the cache cases. What is open now: a loop in which ACO really overlaps the next tile's loads with
  the multiply-add.
  Answered in part (leftovers C, VK-3, `measurements/leftovers-c-b.md`): a GLSL-level software pipeline does not reach
  the machine code: ACO waits for the next tile's loads before the multiply-add, as in the naive loop. Restructuring
  the loop did make ACO move the per-lane address code out of it (39 to 22 instructions), and that loop is faster in
  four of five cases in both passes and the rerun: FP16 from the 4 MiB window by 5 to 12 % and FP8 by 10 to 20 % (operand
  GB/s; no test of cause). For FP16 from the 32 MiB window no effect is established (-13 %, +1 %, +4 %). Single
  bursts; differences of a few per cent are within the pass-to-pass spread. MEASURED.
- Why wave64 is 8 % slower than wave32 in the load-per-tile kernel from a 4 MiB window: narrowed from the machine
  code (READ, `asm/load_f16_row_sg0.s` and `_sg32.s` of `runs/20261117T142940Z-vulkan/`): both loops issue the same 2
  loads per tile (wave64 `buffer_load_b64` + `global_load_tr_b128` into 2 registers, wave32 `buffer_load_b128` +
  `global_load_tr_b128` into 4), wait for both every tile, and reload the descriptors the same way, so loads per tile
  and loads in flight are excluded. The difference is the per-tile address arithmetic: 16 vector instructions
  besides the matrix instruction in wave64 against 13 in wave32 (wave64 adds `v_mbcnt_hi`, `v_bfrev_b32` and a
  shift for the lane index), each executed as two 32-lane halves in wave64, among them two `v_mul_lo_u32` (4 cycles
  each in wave32, lane 1). INFERRED: the extra vector issue time per tile (about 44 against 19 cycles per subgroup)
  is what wave64 loses where the loop is not bound by memory; not tested with a modified shader. (FP8 from DRAM at 90 %: explained in item 5.)
  Answered in part (leftovers C, VK-4, `measurements/leftovers-c-b.md`): a dose test of 8 added dependent integer
  multiply-adds per tile costs wave64 much more than wave32 (-19 % against -1.3 % per GHz in the second pass and the
  rerun; -45 % and -18 % against -1.7 % and -2.0 % in operand GB/s in the first and second pass), which fits "wave64 runs every vector
  instruction in two halves" (INFERRED). Not settled: that a shorter loop closes the gap (wave32 over wave64 1.037 in
  the second pass and the rerun, 1.11 in the first pass); a "lean" shader could not be written in GLSL. MEASURED
  (single bursts); the mechanism is INFERRED.
- 4-bit, K = 32 and sparse forms are not exposed by RADV. Not reachable (leftovers C plan, VK-7b).
- `VK_NV_cooperative_matrix2` is not exposed. Not reachable (leftovers C plan, VK-8).
- New from leftovers C: why the first Vulkan pass's -45 % for the added per-tile work differs so far from the second
  pass's -18 % (the first pass's clocks were not kept; VK-4, `measurements/leftovers-c-b.md`).

Answered since (leftovers C, 2026-11-17):

- Answered (leftovers C, VK-5, `measurements/leftovers-c-b.md`): the lost-product table with odd C: the rounding rule
  (the running sum rounded to FP32, nearest even, after each group of 4 products for 16-bit inputs and 8 for 8-bit)
  predicts all 8,448 result words (0 differ; with round toward zero in the model, 2,048 differ). An odd C makes the
  first tie round up: FP16 and BF16 keep half of the 1.0 at C = 2^22 + 0.5, and FP8 keeps all of it at 2^23 + 1.
  MEASURED.
- Answered in vulkan-view.md, item 7: a zero-work-group dispatch reads 0.56 us of GPU time, the same as an empty
  shader, so about 0.5 us is the floor of a timestamp pair; the stated figure is 1.13 us per dispatch back to back with
  a barrier.
- Answered (leftovers C, VK-7a, `measurements/leftovers-c-b.md`): mixed FP8/BF8 and the saturating 8-bit form run at
  the FP8 rate per clock: 0.989 to 0.998 of 128 x clock / 8 for FP8 x BF8, BF8 x FP8, BF8 x BF8 and signed 8-bit
  saturating (FP8 control 0.996 to 0.997), totals exact. MEASURED.

## Helpers

None. All code, runs and this file are the lane owner's (Opus).
