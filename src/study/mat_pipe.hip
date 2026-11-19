// Lane 4, items 4, 5 and 8: latency against throughput of the matrix unit, the documented hazards and stalls, and
// what overlaps with a matrix instruction inside one wave. Method and registers: mat_common.h. One wave alone.
//
// Usage: mat_pipe lat|hazard|overlap [--break]
//   lat      1, 2, 4, 8 independent accumulators; A and B rotating over 4 register sets; a true dependence through A
//            (the next instruction's A is the previous result). PASS when every independent set and every rotation
//            costs the issue time measured by mat_forms (16 / 8 cycles, within 3 %) and the A-dependence chain is
//            slower than that (it waits for the whole result, so it measures the result latency). --break swaps the
//            8-accumulator test for the A-dependence chain under the same name: an "independent" set that is really
//            one chain, which must FAIL.
//   hazard   (a) A or B from the previous result with V_NOP (what the compiler emits), with another VALU, and without
//            (forbidden by the documents: timing only, results not used); a compiler-built chain through A for
//            comparison; (b) the previous result used as C by a matrix instruction of another type; (c) a VALU
//            reading the result right after; (d) a VALU writing A, B or C right before; (e) a load delivering A.
//            PASS when the reference rows reproduce the known costs (plain chain 16 / 8 cycles, a v_mov 1 cycle).
//   overlap  scalar adds, v_exp_f32, FMAs, v_nop, a branch, and loads issued before a matrix chain. PASS when the
//            "alone" references reproduce lane 1 (FMA 1, exp 4, scalar add 1 to 1.2, branch about 9 cycles).
#include "mat_common.h"

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));

// ---- item 4 ------------------------------------------------------------------------------------------------------
#define LAT(F)                          \
    MAT_TEST(F##_c1, MAT_C1(MF_##F))   \
    MAT_TEST(F##_c2, MAT_C2(MF_##F))   \
    MAT_TEST(F##_c4, MAT_C4(MF_##F))   \
    MAT_TEST(F##_c8, MAT_C8(MF_##F))   \
    MAT_TEST(F##_rot1, MAT_ROT1(MF_##F)) \
    MAT_TEST(F##_rot8, MAT_ROT8(MF_##F))
LAT(F16)
LAT(FF)
LAT(IU8_S)
LAT(F16H)

// True dependence through A: two accumulators D0/D1; each instruction's A is the low registers of the other
// accumulator, which the instruction just before wrote. One V_NOP after each (the documented requirement, and what
// clang inserts). Each instruction also reads its own accumulator as C, written two instructions earlier.
#define AD_F16(nop)                                                                          \
    "v_wmma_f32_16x16x16_f16 v[24:31], v[16:19], " MAT_B4(0) ", v[24:31]\n" nop             \
    "v_wmma_f32_16x16x16_f16 v[16:23], v[24:27], " MAT_B4(0) ", v[16:23]\n" nop
#define AD_FF(nop)                                                                           \
    "v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[16:17], " MAT_B2(0) ", v[24:31]\n" nop         \
    "v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[24:25], " MAT_B2(0) ", v[16:23]\n" nop
#define AD_IU8(nop)                                                                          \
    "v_wmma_i32_16x16x16_iu8 v[24:31], v[16:17], " MAT_B2(0) ", v[24:31] neg_lo:[1,1,0]\n" nop \
    "v_wmma_i32_16x16x16_iu8 v[16:23], v[24:25], " MAT_B2(0) ", v[16:23] neg_lo:[1,1,0]\n" nop
#define AD_F16H(nop)                                                                         \
    "v_wmma_f16_16x16x16_f16 v[24:27], v[16:19], " MAT_B4(0) ", v[24:27]\n" nop             \
    "v_wmma_f16_16x16x16_f16 v[16:19], v[24:27], " MAT_B4(0) ", v[16:19]\n" nop
// Through B instead of A.
#define BD_F16(nop)                                                                          \
    "v_wmma_f32_16x16x16_f16 v[24:31], " MAT_A4(0) ", v[16:19], v[24:31]\n" nop             \
    "v_wmma_f32_16x16x16_f16 v[16:23], " MAT_A4(0) ", v[24:27], v[16:23]\n" nop
// The same instruction stream with A from an unrelated register: the cost of the V_NOP alone.
#define AN_F16(nop)                                                                          \
    "v_wmma_f32_16x16x16_f16 v[24:31], " MAT_A4(0) ", " MAT_B4(0) ", v[24:31]\n" nop        \
    "v_wmma_f32_16x16x16_f16 v[16:23], " MAT_A4(0) ", " MAT_B4(0) ", v[16:23]\n" nop
#define AN_FF(nop)                                                                           \
    "v_wmma_f32_16x16x16_fp8_fp8 v[24:31], " MAT_A2(0) ", " MAT_B2(0) ", v[24:31]\n" nop    \
    "v_wmma_f32_16x16x16_fp8_fp8 v[16:23], " MAT_A2(0) ", " MAT_B2(0) ", v[16:23]\n" nop
#define NOP "v_nop\n"
#define VALU1 "v_mov_b32 v130, v131\n"
#define NONE ""
MAT_TEST(adep_f16_nop, MAT_R8(AD_F16(NOP)))
MAT_TEST(adep_ff_nop, MAT_R8(AD_FF(NOP)))
MAT_TEST(adep_iu8_nop, MAT_R8(AD_IU8(NOP)))
MAT_TEST(adep_f16h_nop, MAT_R8(AD_F16H(NOP)))
MAT_TEST(bdep_f16_nop, MAT_R8(BD_F16(NOP)))
MAT_TEST(adep_f16_valu, MAT_R8(AD_F16(VALU1)))
MAT_TEST(adep_ff_valu, MAT_R8(AD_FF(VALU1)))
// Forbidden by the documents (no instruction between): timing only.
MAT_TEST(adep_f16_none, MAT_R8(AD_F16(NONE)))
MAT_TEST(adep_ff_none, MAT_R8(AD_FF(NONE)))
MAT_TEST(bdep_f16_none, MAT_R8(BD_F16(NONE)))
MAT_TEST(anodep_f16_nop, MAT_R8(AN_F16(NOP)))
MAT_TEST(anodep_ff_nop, MAT_R8(AN_FF(NOP)))
MAT_TEST(anodep_f16_valu, MAT_R8(AN_F16(VALU1)))

// Compiler-built chains with builtins (a marker asm keeps the loop visible to the loop checker): through A of the
// 16-bit-accumulator form (clang must insert V_NOP), and through C (no hazard).
#define MAT_BUILTIN_KERNEL(NAME, T, INIT, STEP)                                                                  \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void NAME(const uint32_t * __restrict__ in, uint64_t * __restrict__ out,      \
                                               uint32_t * __restrict__ sink, const uint32_t *) {                  \
        const int lane = threadIdx.x;                                                                             \
        h8 a = mat::ld<h8>(in + mat::IN_A + lane * 8), b = mat::ld<h8>(in + mat::IN_B + lane * 8);                \
        T acc = INIT;                                                                                             \
        asm volatile("" : "+v"(a), "+v"(b), "+v"(acc));                                                           \
        for (int rep = 0; rep < mat::REPS; ++rep) {                                                               \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < mat::ITERS; ++it) {                                         \
                asm volatile(";compiler-built loop");                                                             \
                _Pragma("unroll") for (int u = 0; u < 16 * U; ++u) { STEP; }                                      \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (lane == 0) {                                                                                      \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        uint32_t s = 0;                                                                                           \
        for (int i = 0; i < 8; ++i) {                                                                             \
            s += core::fold(acc[i]) + core::fold(a[i]);                                                           \
        }                                                                                                         \
        /* a small constant: the folded halves keep s below 2^21, so a large one lets clang drop the loop */      \
        if (s == 12345u) {                                                                                        \
            sink[lane] = s;                                                                                       \
        }                                                                                                         \
    }
MAT_BUILTIN_KERNEL(bi_f16h_achain, h8, a, acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(acc, b, acc))
MAT_BUILTIN_KERNEL(bi_f16h_cchain, h8, a, acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, acc))
MAT_BUILTIN_KERNEL(bi_f16_cchain, f8v, (f8v){}, acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc))

// ---- item 5 (b): result used as C by another type -----------------------------------------------------------------
// dep: both instructions on D0 (each reads the other's result as C); ind: the second type on D1, so no instruction
// reads a result of the other type.
#define MIX(F1, F2, d2) MAT_R8(F1(0, 0) F2(d2, 0))
MAT_TEST(mix_f16_ff_dep, MIX(MF_F16, MF_FF, 0))
MAT_TEST(mix_f16_ff_ind, MIX(MF_F16, MF_FF, 1))
MAT_TEST(mix_f16_bf16_dep, MIX(MF_F16, MF_BF16, 0))
MAT_TEST(mix_f16_bf16_ind, MIX(MF_F16, MF_BF16, 1))
MAT_TEST(mix_ff_bb_dep, MIX(MF_FF, MF_BB, 0))
MAT_TEST(mix_ff_bb_ind, MIX(MF_FF, MF_BB, 1))
MAT_TEST(mix_f16_iu8_dep, MIX(MF_F16, MF_IU8_S, 0))
MAT_TEST(mix_f16_iu8_ind, MIX(MF_F16, MF_IU8_S, 1))
MAT_TEST(mix_ff_iu8_dep, MIX(MF_FF, MF_IU8_S, 0))
MAT_TEST(mix_ff_iu8_ind, MIX(MF_FF, MF_IU8_S, 1))
// dense and sparse of the same type on one accumulator
MAT_TEST(mix_f16_sf16_dep, MIX(MF_F16, MS_F16, 0))
MAT_TEST(mix_f16_sf16_ind, MIX(MF_F16, MS_F16, 1))

// ---- item 5 (c): VALU reading the result right after -------------------------------------------------------------
// Eight accumulators rotate, so matrix instructions never depend on each other; after each one a v_mov reads a
// register of the result it just wrote (first or last register), of the result one instruction older, or nothing.
#define RD(F, n, reg) F(n, 0) "v_mov_b32 v130, " reg "\n"
#define RD8(F, r0, r1, r2, r3, r4, r5, r6, r7) \
    MAT_R2(RD(F, 0, r0) RD(F, 1, r1) RD(F, 2, r2) RD(F, 3, r3) RD(F, 4, r4) RD(F, 5, r5) RD(F, 6, r6) RD(F, 7, r7))
#define RD_FIRST(F) RD8(F, "v16", "v24", "v32", "v40", "v48", "v56", "v64", "v72")
#define RD_LAST(F) RD8(F, "v23", "v31", "v39", "v47", "v55", "v63", "v71", "v79")
#define RD_PREV(F) RD8(F, "v72", "v16", "v24", "v32", "v40", "v48", "v56", "v64")
#define RD_NONE(F) RD8(F, "v131", "v131", "v131", "v131", "v131", "v131", "v131", "v131")
MAT_TEST(rd_f16_first, RD_FIRST(MF_F16))
MAT_TEST(rd_f16_last, RD_LAST(MF_F16))
MAT_TEST(rd_f16_prev, RD_PREV(MF_F16))
MAT_TEST(rd_f16_none, RD_NONE(MF_F16))
MAT_TEST(rd_ff_first, RD_FIRST(MF_FF))
MAT_TEST(rd_ff_last, RD_LAST(MF_FF))
MAT_TEST(rd_ff_prev, RD_PREV(MF_FF))
MAT_TEST(rd_ff_none, RD_NONE(MF_FF))
// Result to a VALU and back: v_add on the first register of the result, which the next matrix instruction of the
// same chain reads as C (one chain).
#define RT(F) F(0, 0) "v_add_f32 v16, v16, v138\n"
MAT_TEST(rt_f16, MAT_R16(RT(MF_F16)))
MAT_TEST(rt_ff, MAT_R16(RT(MF_FF)))

// ---- item 5 (d): VALU writing A, B or C right before -------------------------------------------------------------
// A and B sets rotate (4) and accumulators rotate (8), so only the VALU-to-matrix dependence remains.
#define WR(F, n, m, reg) "v_mov_b32 " reg ", v131\n" F(n, m)
#define WR8(F, r0, r1, r2, r3, r4, r5, r6, r7)                                                                  \
    MAT_R2(WR(F, 0, 0, r0) WR(F, 1, 1, r1) WR(F, 2, 2, r2) WR(F, 3, 3, r3) WR(F, 4, 0, r4) WR(F, 5, 1, r5)        \
               WR(F, 6, 2, r6) WR(F, 7, 3, r7))
#define WR_A(F) WR8(F, "v80", "v84", "v88", "v92", "v80", "v84", "v88", "v92")
#define WR_B(F) WR8(F, "v96", "v104", "v112", "v120", "v96", "v104", "v112", "v120")
#define WR_C(F) WR8(F, "v16", "v24", "v32", "v40", "v48", "v56", "v64", "v72")
#define WR_CLAST(F) WR8(F, "v23", "v31", "v39", "v47", "v55", "v63", "v71", "v79")
#define WR_NONE(F) WR8(F, "v130", "v130", "v130", "v130", "v130", "v130", "v130", "v130")
// the same with A and B fixed (set 0): does the rotation of A and B cost anything when a VALU sits in between
#define WRF(F, n) "v_mov_b32 v130, v131\n" F(n, 0)
#define WR_NONE_FIXED(F) MAT_R2(WRF(F, 0) WRF(F, 1) WRF(F, 2) WRF(F, 3) WRF(F, 4) WRF(F, 5) WRF(F, 6) WRF(F, 7))
MAT_TEST(wr_f16_A, WR_A(MF_F16))
MAT_TEST(wr_f16_B, WR_B(MF_F16))
MAT_TEST(wr_f16_C, WR_C(MF_F16))
MAT_TEST(wr_f16_Clast, WR_CLAST(MF_F16))
MAT_TEST(wr_f16_none, WR_NONE(MF_F16))
MAT_TEST(wr_f16_none_fixed, WR_NONE_FIXED(MF_F16))
MAT_TEST(wr_ff_A, WR_A(MF_FF))
MAT_TEST(wr_ff_C, WR_C(MF_FF))
MAT_TEST(wr_ff_none, WR_NONE(MF_FF))

// ---- item 5 (e): a load delivering the operand -------------------------------------------------------------------
// Each pattern: one 16-byte load per lane from the 64 KiB buffer (an L0 hit after the first pass), the wait, then
// the consumer. A-dependent matrix instruction against one that does not read the loaded registers; the same for a
// VALU consumer. The loaded data are FP16 ones, a sensible A.
#define LDW(dst) "global_load_b128 " dst ", v129, s[44:45]\ns_wait_loadcnt 0x0\n"
#define LD_A_MM(n) LDW("v[80:83]") MF_F16(n, 0)
#define LD_X_MM(n) LDW("v[132:135]") MF_F16(n, 0)
#define LD_X_VA LDW("v[132:135]") "v_mov_b32 v130, v132\n"
#define LD_X_VN LDW("v[132:135]") "v_mov_b32 v130, v131\n"
MAT_TEST(ld_mm_dep, MAT_R2(LD_A_MM(0) LD_A_MM(1) LD_A_MM(2) LD_A_MM(3) LD_A_MM(4) LD_A_MM(5) LD_A_MM(6) LD_A_MM(7)))
MAT_TEST(ld_mm_ind, MAT_R2(LD_X_MM(0) LD_X_MM(1) LD_X_MM(2) LD_X_MM(3) LD_X_MM(4) LD_X_MM(5) LD_X_MM(6) LD_X_MM(7)))
MAT_TEST(ld_valu_dep, MAT_R16(LD_X_VA))
MAT_TEST(ld_valu_ind, MAT_R16(LD_X_VN))
MAT_TEST(f16_c8, MAT_C8(MF_F16))
MAT_TEST(valu_mov, MAT_R16("v_mov_b32 v130, v131\n"))

// ---- item 8: what overlaps with a matrix instruction in one wave -------------------------------------------------
#define SA4 "s_add_co_i32 s40, s40, 1\ns_add_co_i32 s41, s41, 1\ns_add_co_i32 s42, s42, 1\ns_add_co_i32 s43, s43, 1\n"
#define EX1 "v_exp_f32 v130, v131\n"
#define EX4 "v_exp_f32 v130, v131\nv_exp_f32 v132, v133\nv_exp_f32 v134, v135\nv_exp_f32 v136, v137\n"
// Two groups of four FMAs on different registers, alternated, so each register is reused 8 FMAs later (beyond the
// 5-cycle latency).
#define FM4A                                                                                                   \
    "v_fma_f32 v130, v130, v138, v139\nv_fma_f32 v131, v131, v138, v139\nv_fma_f32 v132, v132, v138, v139\n" \
    "v_fma_f32 v133, v133, v138, v139\n"
#define FM4B                                                                                                   \
    "v_fma_f32 v134, v134, v138, v139\nv_fma_f32 v135, v135, v138, v139\nv_fma_f32 v136, v136, v138, v139\n" \
    "v_fma_f32 v137, v137, v138, v139\n"
#define VN4 "v_nop\nv_nop\nv_nop\nv_nop\n"
#define BR "s_branch 1f\n1:\n"
MAT_TEST(ov_mm, MAT_R16(MF_F16(0, 0)))
MAT_TEST(ov_mm_sa4, MAT_R16(MF_F16(0, 0) SA4))
MAT_TEST(ov_mm_sa8, MAT_R16(MF_F16(0, 0) SA4 SA4))
MAT_TEST(ov_mm_sa16, MAT_R16(MF_F16(0, 0) SA4 SA4 SA4 SA4))
MAT_TEST(ov_sa4, MAT_R16(SA4))
MAT_TEST(ov_mm_ex1, MAT_R16(MF_F16(0, 0) EX1))
MAT_TEST(ov_mm_ex4, MAT_R16(MF_F16(0, 0) EX4))
MAT_TEST(ov_ex4, MAT_R16(EX4))
// Smaller blocks for the FMA rows: measured, loops over about 4 KiB that issue an 8-byte instruction every cycle
// lose 12 to 14 cycles per iteration at U = 8 (instruction fetch, INFERRED), which breaks the linearity check.
MAT_TEST(ov_mm_fm4, MAT_R4(MF_F16(0, 0) FM4A MF_F16(0, 0) FM4B))
MAT_TEST(ov_fm4, FM4A FM4B)
MAT_TEST(ov_mm_vn4, MAT_R16(MF_F16(0, 0) VN4))
MAT_TEST(ov_vn4, MAT_R16(VN4))
MAT_TEST(ov_mm_br, MAT_R16(MF_F16(0, 0) BR))
MAT_TEST(ov_br, MAT_R16(BR))
// FP8 form (8 cycles): the same with scalar adds and exps
MAT_TEST(ov_ff, MAT_R16(MF_FF(0, 0)))
MAT_TEST(ov_ff_sa4, MAT_R16(MF_FF(0, 0) SA4))
MAT_TEST(ov_ff_ex1, MAT_R16(MF_FF(0, 0) EX1))
// Loads issued before a matrix chain, waited for after it. 2 loads of 16 bytes per lane (4 KiB apart, L0 hits).
#define LD2 "global_load_b128 v[130:133], v129, s[44:45]\nglobal_load_b128 v[134:137], v129, s[44:45] offset:4096\n"
#define WT "s_wait_loadcnt 0x0\n"
#define MM4 MF_F16(0, 0) MF_F16(0, 0) MF_F16(0, 0) MF_F16(0, 0)
#define MM8 MM4 MM4
MAT_TEST(ov_ld2, LD2 WT)
MAT_TEST(ov_mm4, MM4)
MAT_TEST(ov_mm8, MM8)
MAT_TEST(ov_ld2_mm4, LD2 MM4 WT)
MAT_TEST(ov_ld2_mm8, LD2 MM8 WT)
MAT_TEST(ov_ld2_wt_mm4, LD2 WT MM4)
MAT_TEST(ov_ld2_mm16, LD2 MM8 MM8 WT)
MAT_TEST(ov_mm16, MM8 MM8)
// The loads issued right behind matrix work, and behind matrix work plus 4 V_NOPs, to see whether a load waits for
// the matrix instructions before it.
MAT_TEST(ov_mm4_ld2_wt, MM4 LD2 WT)
MAT_TEST(ov_mm4_nop4_ld2_wt, MM4 VN4 LD2 WT)
MAT_TEST(ov_nop4_ld2_wt, VN4 LD2 WT)

#define BI_ENTRY(NAME) mat::Test{#NAME, NAME<mat::ULO>, NAME<mat::UMID>, NAME<mat::UHI>}

static int mode_lat(bool brk) {
    mat::set_input(mat::default_input());
    mat::Checks ck;
    struct L {
        const char * f;
        double issue;
        std::vector<mat::Test> t;
        mat::Test adep;
    };
    std::vector<L> ls = {
        {"F16", 16, {MAT_ENTRY(F16_c1), MAT_ENTRY(F16_c2), MAT_ENTRY(F16_c4), MAT_ENTRY(F16_c8), MAT_ENTRY(F16_rot1), MAT_ENTRY(F16_rot8)}, MAT_ENTRY(adep_f16_nop)},
        {"FF", 8, {MAT_ENTRY(FF_c1), MAT_ENTRY(FF_c2), MAT_ENTRY(FF_c4), MAT_ENTRY(FF_c8), MAT_ENTRY(FF_rot1), MAT_ENTRY(FF_rot8)}, MAT_ENTRY(adep_ff_nop)},
        {"IU8_S", 8, {MAT_ENTRY(IU8_S_c1), MAT_ENTRY(IU8_S_c2), MAT_ENTRY(IU8_S_c4), MAT_ENTRY(IU8_S_c8), MAT_ENTRY(IU8_S_rot1), MAT_ENTRY(IU8_S_rot8)}, MAT_ENTRY(adep_iu8_nop)},
        {"F16H", 16, {MAT_ENTRY(F16H_c1), MAT_ENTRY(F16H_c2), MAT_ENTRY(F16H_c4), MAT_ENTRY(F16H_c8), MAT_ENTRY(F16H_rot1), MAT_ENTRY(F16H_rot8)}, MAT_ENTRY(adep_f16h_nop)},
    };
    for (L & l : ls) {
        if (brk) {
            // the "8 independent accumulators" entry is really one chain through A
            l.t[3] = mat::Test{l.t[3].name, l.adep.lo, l.adep.mid, l.adep.hi};
        }
        auto m = mat::measure("mat_pipe", l.t, 16);
        const auto a = mat::measure("mat_pipe", {l.adep}, 16);
        // the A-dependence block has 16 matrix instructions and 16 V_NOPs
        const double adep = a.at(l.adep.name).per;
        for (const mat::Test & t : l.t) {
            const mat::Measured & v = m.at(t.name);
            ck.add(mat::near(v.per, l.issue, 0.03 * l.issue) && v.lin, "%s %.3f (issue %.0f)", t.name, v.per, l.issue);
        }
        std::printf("RESULT mat_pipe lat %s chain_through_C %.3f independent_x8 %.3f chain_through_A_with_vnop %.3f "
                    "result_latency_estimate %.3f\n",
                    l.f, m.at(l.t[0].name).per, m.at(l.t[3].name).per, adep, adep - 1.0);
        ck.add(adep > 1.1 * l.issue, "%s chain through A %.3f per instruction+V_NOP, above the issue time", l.f, adep);
    }
    return ck.finish("mat_pipe_lat");
}

static int mode_hazard() {
    mat::set_input(mat::default_input());
    mat::Checks ck;
    // (a): 16 matrix instructions per block in every test; per instruction, the V_NOP/VALU included.
    const auto a = mat::measure(
        "mat_pipe",
        {MAT_ENTRY(F16_c1), MAT_ENTRY(FF_c1), MAT_ENTRY(adep_f16_nop), MAT_ENTRY(adep_ff_nop), MAT_ENTRY(adep_iu8_nop),
         MAT_ENTRY(adep_f16h_nop), MAT_ENTRY(bdep_f16_nop), MAT_ENTRY(adep_f16_valu), MAT_ENTRY(adep_ff_valu),
         MAT_ENTRY(adep_f16_none), MAT_ENTRY(adep_ff_none), MAT_ENTRY(bdep_f16_none), MAT_ENTRY(anodep_f16_nop),
         MAT_ENTRY(anodep_ff_nop), MAT_ENTRY(anodep_f16_valu), BI_ENTRY(bi_f16h_achain), BI_ENTRY(bi_f16h_cchain),
         BI_ENTRY(bi_f16_cchain)},
        16);
    ck.add(mat::near(a.at("F16_c1").per, 16, 0.3) && mat::near(a.at("FF_c1").per, 8, 0.2),
           "reference chains %.3f and %.3f (16 and 8 known)", a.at("F16_c1").per, a.at("FF_c1").per);
    std::printf("RESULT mat_pipe hazard_a f16: A from previous result + V_NOP %.3f; V_NOP without dependence %.3f; "
                "added by the dependence %.3f; with v_mov instead of V_NOP %.3f; no V_NOP (FORBIDDEN, timing only) %.3f\n",
                a.at("adep_f16_nop").per, a.at("anodep_f16_nop").per,
                a.at("adep_f16_nop").per - a.at("anodep_f16_nop").per, a.at("adep_f16_valu").per,
                a.at("adep_f16_none").per);
    std::printf("RESULT mat_pipe hazard_a fp8: A from previous result + V_NOP %.3f; V_NOP without dependence %.3f; "
                "added %.3f; no V_NOP (FORBIDDEN) %.3f\n",
                a.at("adep_ff_nop").per, a.at("anodep_ff_nop").per,
                a.at("adep_ff_nop").per - a.at("anodep_ff_nop").per, a.at("adep_ff_none").per);
    // (b)
    const auto b = mat::measure(
        "mat_pipe",
        {MAT_ENTRY(mix_f16_ff_dep), MAT_ENTRY(mix_f16_ff_ind), MAT_ENTRY(mix_f16_bf16_dep), MAT_ENTRY(mix_f16_bf16_ind),
         MAT_ENTRY(mix_ff_bb_dep), MAT_ENTRY(mix_ff_bb_ind), MAT_ENTRY(mix_f16_iu8_dep), MAT_ENTRY(mix_f16_iu8_ind),
         MAT_ENTRY(mix_ff_iu8_dep), MAT_ENTRY(mix_ff_iu8_ind), MAT_ENTRY(mix_f16_sf16_dep),
         MAT_ENTRY(mix_f16_sf16_ind)},
        8, "pair");
    for (const char * p : {"mix_f16_ff", "mix_f16_bf16", "mix_ff_bb", "mix_f16_iu8", "mix_ff_iu8", "mix_f16_sf16"}) {
        const double dep = b.at(std::string(p) + "_dep").per, ind = b.at(std::string(p) + "_ind").per;
        std::printf("RESULT mat_pipe hazard_b %-13s pair_dependent %7.3f pair_independent %7.3f added_per_type_change "
                    "%6.3f\n", p, dep, ind, (dep - ind) / 2);
    }
    // (c), (d), (e): 16 patterns per block
    const auto c = mat::measure(
        "mat_pipe",
        {MAT_ENTRY(f16_c8), MAT_ENTRY(valu_mov), MAT_ENTRY(rd_f16_first), MAT_ENTRY(rd_f16_last), MAT_ENTRY(rd_f16_prev),
         MAT_ENTRY(rd_f16_none), MAT_ENTRY(rd_ff_first), MAT_ENTRY(rd_ff_last), MAT_ENTRY(rd_ff_prev),
         MAT_ENTRY(rd_ff_none), MAT_ENTRY(rt_f16), MAT_ENTRY(rt_ff), MAT_ENTRY(wr_f16_A), MAT_ENTRY(wr_f16_B),
         MAT_ENTRY(wr_f16_C), MAT_ENTRY(wr_f16_Clast), MAT_ENTRY(wr_f16_none), MAT_ENTRY(wr_f16_none_fixed), MAT_ENTRY(wr_ff_A), MAT_ENTRY(wr_ff_C),
         MAT_ENTRY(wr_ff_none), MAT_ENTRY(ld_mm_dep), MAT_ENTRY(ld_mm_ind), MAT_ENTRY(ld_valu_dep),
         MAT_ENTRY(ld_valu_ind)},
        16, "pattern");
    ck.add(mat::near(c.at("valu_mov").per, 1.0, 0.05), "v_mov alone %.3f (1 known)", c.at("valu_mov").per);
    ck.add(mat::near(c.at("f16_c8").per, 16, 0.3), "8 accumulators %.3f (16 known)", c.at("f16_c8").per);
    for (const char * f : {"f16", "ff"}) {
        const std::string F = f;
        std::printf("RESULT mat_pipe hazard_c %s: matrix + v_mov reading its first register %.3f, its last register "
                    "%.3f, the previous result %.3f, nothing %.3f; added by reading the fresh result %.3f\n",
                    f, c.at("rd_" + F + "_first").per, c.at("rd_" + F + "_last").per, c.at("rd_" + F + "_prev").per,
                    c.at("rd_" + F + "_none").per, c.at("rd_" + F + "_first").per - c.at("rd_" + F + "_none").per);
        std::printf("RESULT mat_pipe hazard_d %s: v_mov writing A %.3f, C %.3f, nothing %.3f%s\n", f,
                    c.at("wr_" + F + "_A").per, c.at("wr_" + F + "_C").per, c.at("wr_" + F + "_none").per,
                    F == "f16" ? "" : "");
    }
    std::printf("RESULT mat_pipe hazard_d f16: v_mov writing B %.3f, last register of C %.3f, nothing with A and B "
                "fixed %.3f\n", c.at("wr_f16_B").per, c.at("wr_f16_Clast").per, c.at("wr_f16_none_fixed").per);
    std::printf("RESULT mat_pipe round_trip matrix -> v_add on the result -> next matrix reads it as C: f16 %.3f, "
                "fp8 %.3f per pair\n", c.at("rt_f16").per, c.at("rt_ff").per);
    std::printf("RESULT mat_pipe hazard_e load+wait then matrix reading the loaded A %.3f, not reading it %.3f; then "
                "VALU reading it %.3f, not reading it %.3f\n",
                c.at("ld_mm_dep").per, c.at("ld_mm_ind").per, c.at("ld_valu_dep").per, c.at("ld_valu_ind").per);
    return ck.finish("mat_pipe_hazard");
}

static int mode_overlap() {
    mat::set_input(mat::default_input());
    mat::Checks ck;
    const auto m = mat::measure(
        "mat_pipe",
        {MAT_ENTRY(ov_mm), MAT_ENTRY(ov_mm_sa4), MAT_ENTRY(ov_mm_sa8), MAT_ENTRY(ov_mm_sa16), MAT_ENTRY(ov_sa4),
         MAT_ENTRY(ov_mm_ex1), MAT_ENTRY(ov_mm_ex4), MAT_ENTRY(ov_ex4), MAT_ENTRY(ov_mm_vn4), MAT_ENTRY(ov_vn4),
         MAT_ENTRY(ov_mm_br), MAT_ENTRY(ov_br), MAT_ENTRY(ov_ff), MAT_ENTRY(ov_ff_sa4), MAT_ENTRY(ov_ff_ex1)},
        16, "pattern");
    const auto mf = mat::measure("mat_pipe", {MAT_ENTRY(ov_mm_fm4)}, 8, "pattern");
    const auto f2 = mat::measure("mat_pipe", {MAT_ENTRY(ov_fm4)}, 2, "pattern");
    bool lin = true;
    for (const auto * mm : {&m, &mf, &f2}) {
        for (const auto & [n, v] : *mm) {
            lin = lin && v.lin;
        }
    }
    ck.add(lin, "all loops linear");
    const double mm = m.at("ov_mm").per, sa = m.at("ov_sa4").per / 4, ex = m.at("ov_ex4").per / 4,
                 fm = f2.at("ov_fm4").per / 4, vn = m.at("ov_vn4").per / 4, br = m.at("ov_br").per;
    ck.add(mat::near(fm, 1.0, 0.05), "FMA alone %.3f (1 known)", fm);
    ck.add(mat::near(ex, 4.0, 0.15), "v_exp_f32 alone %.3f (4 known)", ex);
    ck.add(sa >= 0.95 && sa <= 1.3, "s_add_co_i32 alone %.3f (1.0 to 1.19 known)", sa);
    ck.add(br >= 8 && br <= 11, "s_branch alone %.3f (9 known)", br);
    struct O {
        const char * name;
        double with, base, alone;
    };
    const std::vector<O> os = {
        {"f16+4 scalar adds", m.at("ov_mm_sa4").per, mm, 4 * sa},
        {"f16+8 scalar adds", m.at("ov_mm_sa8").per, mm, 8 * sa},
        {"f16+16 scalar adds", m.at("ov_mm_sa16").per, mm, 16 * sa},
        {"f16+1 v_exp_f32", m.at("ov_mm_ex1").per, mm, ex},
        {"f16+4 v_exp_f32", m.at("ov_mm_ex4").per, mm, 4 * ex},
        {"f16+4 FMAs", mf.at("ov_mm_fm4").per, mm, 4 * fm},
        {"f16+4 v_nop", m.at("ov_mm_vn4").per, mm, 4 * vn},
        {"f16+s_branch", m.at("ov_mm_br").per, mm, br},
        {"fp8+4 scalar adds", m.at("ov_ff_sa4").per, m.at("ov_ff").per, 4 * sa},
        {"fp8+1 v_exp_f32", m.at("ov_ff_ex1").per, m.at("ov_ff").per, ex},
    };
    for (const O & o : os) {
        // fraction 0: the extra work adds its full time; 1: it hides completely inside the matrix instruction.
        const double frac = (o.base + o.alone - o.with) / std::min(o.base, o.alone);
        std::printf("RESULT mat_pipe overlap %-20s together %7.3f matrix_alone %7.3f other_alone %7.3f sum %7.3f "
                    "hidden_fraction %5.2f\n", o.name, o.with, o.base, o.alone, o.base + o.alone, frac);
    }
    const auto l = mat::measure("mat_pipe",
                                {MAT_ENTRY(ov_ld2), MAT_ENTRY(ov_mm4), MAT_ENTRY(ov_mm8), MAT_ENTRY(ov_ld2_mm4),
                                 MAT_ENTRY(ov_ld2_mm8), MAT_ENTRY(ov_ld2_wt_mm4), MAT_ENTRY(ov_ld2_mm16),
                                 MAT_ENTRY(ov_mm16), MAT_ENTRY(ov_mm4_ld2_wt), MAT_ENTRY(ov_mm4_nop4_ld2_wt),
                                 MAT_ENTRY(ov_nop4_ld2_wt)},
                                1, "block");
    for (const char * mmn : {"ov_mm4", "ov_mm8", "ov_mm16"}) {
        const std::string both = std::string("ov_ld2_") + (mmn + 3);
        std::printf("RESULT mat_pipe overlap loads_then_%s together %7.2f loads_alone %7.2f chain_alone %7.2f sum %7.2f "
                    "max %7.2f\n", mmn + 3, l.at(both).per, l.at("ov_ld2").per, l.at(mmn).per,
                    l.at("ov_ld2").per + l.at(mmn).per, std::max(l.at("ov_ld2").per, l.at(mmn).per));
    }
    std::printf("RESULT mat_pipe overlap loads_wait_then_mm4 %7.2f (reference: no overlap possible)\n",
                l.at("ov_ld2_wt_mm4").per);
    std::printf("RESULT mat_pipe overlap mm4_then_loads_wait %7.2f mm4_nop4_then_loads_wait %7.2f "
                "nop4_then_loads_wait %7.2f\n",
                l.at("ov_mm4_ld2_wt").per, l.at("ov_mm4_nop4_ld2_wt").per, l.at("ov_nop4_ld2_wt").per);
    return ck.finish("mat_pipe_overlap");
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "lat";
    const bool brk = mat::flag(argc, argv, "--break");
    if (mode == "lat") {
        return mode_lat(brk);
    }
    if (mode == "hazard") {
        return mode_hazard();
    }
    if (mode == "overlap") {
        return mode_overlap();
    }
    std::fprintf(stderr, "usage: mat_pipe lat|hazard|overlap [--break]\n");
    return 2;
}
