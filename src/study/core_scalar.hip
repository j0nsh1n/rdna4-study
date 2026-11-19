// Lane 1, item 4: the scalar unit, the vector/scalar hand-offs and their s_wait_alu guards, and s_delay_alu, one
// wave alone on its SIMD (method in core_common.h).
//
// Part A: scalar ALU operations as 16 independent SGPR chains (throughput) and one dependent chain (latency).
// Part B: round trips between the units, each pattern one closed dependency loop, timed with the guard the compiler
// inserts (s_wait_alu depctr_sa_sdst(0) where gfx12's SGPR read hazard applies; checked against clang 24 output) and
// with the guard removed ("_noguard": timing only; the guide says the unguarded value may be wrong).
// Part C: s_delay_alu in front of each dependent vector instruction against none; one wave alone should not care
// (the two-wave case is in core_share.hip).
#include "core_common.h"

typedef unsigned int u32;
typedef unsigned long long u64;

#define P_S_DK(op, i) op " %" #i ", %" #i ", %16\n"
#define P_S_KL(op, i) op " %" #i ", %16, %17\n"  // s_fmac: d += k0 * k1
#define P_S_D(op, i) op " %" #i ", %" #i "\n"
#define P_S_DK_W(op, i) op " %" #i ", %" #i ", %16\ns_wait_alu depctr_sa_sdst(0)\n"

#define SDEF(id, op, TX, TK, P)                                              \
    CORE_KERNEL(id##_c1, TX, TK, "+s", "s", CORE_B1(P, op), "")             \
    CORE_KERNEL(id##_c16, TX, TK, "+s", "s", CORE_B16(P, op), "")

SDEF(sadd, "s_add_co_i32", u32, u32, P_S_DK)
SDEF(sand, "s_and_b32", u32, u32, P_S_DK)
SDEF(sshl, "s_lshl_b32", u32, u32, P_S_DK)
SDEF(smul, "s_mul_i32", u32, u32, P_S_DK)
SDEF(smulhi, "s_mul_hi_u32", u32, u32, P_S_DK)
SDEF(sadd64, "s_add_nc_u64", u64, u64, P_S_DK)
SDEF(smul64, "s_mul_u64", u64, u64, P_S_DK)
SDEF(sfadd, "s_add_f32", float, float, P_S_DK)
SDEF(sfmul, "s_mul_f32", float, float, P_S_DK)
SDEF(sfmac, "s_fmac_f32", float, float, P_S_KL)
SDEF(scvt, "s_cvt_f32_i32", u32, u32, P_S_D)
SDEF(smul_w, "s_mul_i32", u32, u32, P_S_DK_W)

// Mixed kernel: %0..%7 VGPR chains (u32), %8..%15 SGPR chains (u32), %16 a VGPR constant, %17 an SGPR constant.
#define MIX_KERNEL(NAME, BODY)                                                                                    \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,      \
                                               uint32_t * __restrict__ sink) {                                    \
        u32 v[8], s[8];                                                                                           \
        for (int i = 0; i < 8; ++i) {                                                                             \
            v[i] = (u32) in[0];                                                                                   \
            s[i] = (u32) in[0];                                                                                   \
        }                                                                                                         \
        u32 kv = (u32) in[1], ks = (u32) in[2];                                                                   \
        asm volatile("" : "+v"(kv), "+s"(ks));                                                                    \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                              \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                        \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(BODY                                                                             \
                                 : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]), "+v"(v[4]), "+v"(v[5]),         \
                                   "+v"(v[6]), "+v"(v[7]), "+s"(s[0]), "+s"(s[1]), "+s"(s[2]), "+s"(s[3]),         \
                                   "+s"(s[4]), "+s"(s[5]), "+s"(s[6]), "+s"(s[7])                                  \
                                 : "v"(kv), "s"(ks)                                                               \
                                 : "vcc", "scc");                                                                 \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        u32 x = 0;                                                                                                \
        for (int i = 0; i < 8; ++i) {                                                                             \
            x ^= v[i] ^ s[i];                                                                                     \
        }                                                                                                         \
        sink[threadIdx.x] = x;                                                                                    \
    }

#define R16(p) p p p p p p p p p p p p p p p p
// B1: vector -> scalar -> vector: readfirstlane, a scalar multiply, back into the vector chain (the compiler's
// sequence for s = readfirstlane(v); s = s * k; v = v + s).
#define RT_VSV "v_readfirstlane_b32 %8, %0\ns_mul_i32 %8, %8, %17\ns_wait_alu depctr_sa_sdst(0)\nv_add_nc_u32 %0, %8, %0\n"
#define RT_VSV_NG "v_readfirstlane_b32 %8, %0\ns_mul_i32 %8, %8, %17\nv_add_nc_u32 %0, %8, %0\n"
// B2: scalar chain feeding a vector op every step (s = s * k; v = v * s), the vector chain independent of itself.
#define RT_SV "s_mul_i32 %8, %8, %17\ns_wait_alu depctr_sa_sdst(0)\nv_mul_lo_u16 %0, %8, %16\n"
#define RT_SV_NG "s_mul_i32 %8, %8, %17\nv_mul_lo_u16 %0, %8, %16\n"
// B3: vector compare -> scalar use of VCC -> back into the compare (VALU reads the SGPR the SALU wrote).
#define RT_CMP "v_cmp_ne_u32 vcc_lo, %8, %0\ns_and_b32 %8, vcc_lo, %8\ns_wait_alu depctr_sa_sdst(0)\n"
#define RT_CMP_NG "v_cmp_ne_u32 vcc_lo, %8, %0\ns_and_b32 %8, vcc_lo, %8\n"
// B4: vector -> scalar only (readfirstlane result consumed by a scalar op, which feeds the next readfirstlane's
// vector source through v_mov): the vector-written SGPR read by the SALU needs no guard.
#define RT_VS "v_readfirstlane_b32 %8, %0\ns_add_co_i32 %8, %8, %17\ns_wait_alu depctr_sa_sdst(0)\nv_mov_b32 %0, %8\n"
// B5: one scalar op alone in the same kernel shape (reference for B2), and the bare s_wait_alu after it.
#define RT_S "s_mul_i32 %8, %8, %17\n"
#define RT_SW "s_mul_i32 %8, %8, %17\ns_wait_alu depctr_sa_sdst(0)\n"
MIX_KERNEL(rt_vsv, R16(RT_VSV))
MIX_KERNEL(rt_vsv_ng, R16(RT_VSV_NG))
MIX_KERNEL(rt_sv, R16(RT_SV))
MIX_KERNEL(rt_sv_ng, R16(RT_SV_NG))
MIX_KERNEL(rt_cmp, R16(RT_CMP))
MIX_KERNEL(rt_cmp_ng, R16(RT_CMP_NG))
MIX_KERNEL(rt_vs, R16(RT_VS))
MIX_KERNEL(rt_s, R16(RT_S))
MIX_KERNEL(rt_sw, R16(RT_SW))

// Part C: dependent vector chains with and without s_delay_alu (the compiler's form names the producer).
#define P_V_DK(op, i) op " %" #i ", %" #i ", %16\n"
#define P_V_DK_DA(op, i) "s_delay_alu instid0(VALU_DEP_1)\n" op " %" #i ", %" #i ", %16\n"
#define P_V_D(op, i) op " %" #i ", %" #i "\n"
#define P_V_D_DA(op, i) "s_delay_alu instid0(TRANS32_DEP_1)\n" op " %" #i ", %" #i "\n"
CORE_KERNEL(da_add_no, float, float, "+v", "v", CORE_B1(P_V_DK, "v_add_f32"), "")
CORE_KERNEL(da_add_yes, float, float, "+v", "v", CORE_B1(P_V_DK_DA, "v_add_f32"), "")
CORE_KERNEL(da_exp_no, float, float, "+v", "v", CORE_B1(P_V_D, "v_exp_f32"), "")
CORE_KERNEL(da_exp_yes, float, float, "+v", "v", CORE_B1(P_V_D_DA, "v_exp_f32"), "")
// s_delay_alu in front of independent work costs an issue slot or not?
CORE_KERNEL(da_ind_yes, float, float, "+v", "v", CORE_B16(P_V_DK_DA, "v_add_f32"), "")
CORE_KERNEL(da_ind_no, float, float, "+v", "v", CORE_B16(P_V_DK, "v_add_f32"), "")

struct T {
    const char * id;
    const char * what;
    core::Kern k[3];
    double expect;  // cycles per pattern for a known answer; 0 = information only
    core::Init init;
};

#define K3(id) \
    { id<core::ULO>, id<core::UMID>, id<core::UHI> }

int main(int argc, char ** argv) {
    require_arch();
    const core::Init I{3, 1, 1}, F{core::f32(1.0f), core::f32(1.0f), core::f32(0.0f)};
    const core::Init D{core::f64(1.0), core::f64(1.0), core::f64(0.0)};
    (void) D;
    std::vector<T> ts = {
        {"sadd_lat", "s_add_co_i32 dependent chain", K3(sadd_c1), 0, I},
        {"sadd_thr", "s_add_co_i32, 16 independent SGPRs", K3(sadd_c16), 0, I},
        {"sand_lat", "s_and_b32 chain", K3(sand_c1), 0, I},
        {"sand_thr", "s_and_b32 independent", K3(sand_c16), 0, I},
        {"sshl_lat", "s_lshl_b32 chain", K3(sshl_c1), 0, I},
        {"sshl_thr", "s_lshl_b32 independent", K3(sshl_c16), 0, I},
        {"smul_lat", "s_mul_i32 chain", K3(smul_c1), 0, I},
        {"smul_thr", "s_mul_i32 independent", K3(smul_c16), 0, I},
        {"smulhi_lat", "s_mul_hi_u32 chain", K3(smulhi_c1), 0, I},
        {"smulhi_thr", "s_mul_hi_u32 independent", K3(smulhi_c16), 0, I},
        {"sadd64_lat", "s_add_nc_u64 chain", K3(sadd64_c1), 0, I},
        {"sadd64_thr", "s_add_nc_u64 independent", K3(sadd64_c16), 0, I},
        {"smul64_lat", "s_mul_u64 chain", K3(smul64_c1), 0, I},
        {"smul64_thr", "s_mul_u64 independent", K3(smul64_c16), 0, I},
        {"sfadd_lat", "s_add_f32 chain", K3(sfadd_c1), 0, F},
        {"sfadd_thr", "s_add_f32 independent", K3(sfadd_c16), 0, F},
        {"sfmul_lat", "s_mul_f32 chain", K3(sfmul_c1), 0, F},
        {"sfmul_thr", "s_mul_f32 independent", K3(sfmul_c16), 0, F},
        {"sfmac_lat", "s_fmac_f32 chain", K3(sfmac_c1), 0, F},
        {"sfmac_thr", "s_fmac_f32 independent", K3(sfmac_c16), 0, F},
        {"scvt_lat", "s_cvt_f32_i32 chain", K3(scvt_c1), 0, I},
        {"scvt_thr", "s_cvt_f32_i32 independent", K3(scvt_c16), 0, I},
        {"smul_wait", "s_mul_i32 chain with s_wait_alu depctr_sa_sdst(0) after each", K3(smul_w_c1), 0, I},
        {"rt_vsv", "v_readfirstlane -> s_mul -> [wait] -> v_add (compiler's form)", K3(rt_vsv), 0, I},
        {"rt_vsv_noguard", "same without the s_wait_alu", K3(rt_vsv_ng), 0, I},
        {"rt_sv", "s_mul chain, each result read by a VALU, [wait] (compiler's form)", K3(rt_sv), 0, I},
        {"rt_sv_noguard", "same without the s_wait_alu", K3(rt_sv_ng), 0, I},
        {"rt_cmp", "v_cmp -> VCC -> s_and -> [wait] -> v_cmp (VALU->SALU->VALU loop)", K3(rt_cmp), 0, I},
        {"rt_cmp_noguard", "same without the s_wait_alu", K3(rt_cmp_ng), 0, I},
        {"rt_vs", "v_readfirstlane -> s_add -> [wait] -> v_mov (scalar result back to a VGPR)", K3(rt_vs), 0, I},
        {"rt_s", "s_mul_i32 chain alone in the mixed kernel", K3(rt_s), 0, I},
        {"rt_sw", "s_mul_i32 chain + s_wait_alu in the mixed kernel", K3(rt_sw), 0, I},
        // Known answers from core_valu: dependent v_add_f32 5 cycles, v_exp_f32 10, independent v_add_f32 1.
        {"delay_add_no", "v_add_f32 chain, no s_delay_alu", K3(da_add_no), 5, F},
        {"delay_add_yes", "v_add_f32 chain, s_delay_alu VALU_DEP_1 before each", K3(da_add_yes), 0, F},
        {"delay_exp_no", "v_exp_f32 chain, no s_delay_alu", K3(da_exp_no), 10, F},
        {"delay_exp_yes", "v_exp_f32 chain, s_delay_alu TRANS32_DEP_1 before each", K3(da_exp_yes), 0, F},
        {"delay_ind_no", "16 independent v_add_f32", K3(da_ind_no), 1, F},
        {"delay_ind_yes", "16 independent v_add_f32, each behind s_delay_alu VALU_DEP_1", K3(da_ind_yes), 0, F},
    };
    bool ok = true;
    int failed = 0, checked = 0;
    std::string fails;
    for (const T & t : ts) {
        core::set_init(t.init);
        core::Cost r;
        bool pass = false;
        for (int attempt = 0; attempt < 2 && !pass; ++attempt) {
            if (attempt) {
                std::printf("RESULT scalar %-15s first measurement %.3f (mid %.1f), measuring again\n", t.id, r.per, r.mid);
            }
            r = core::pattern_cost(t.k[0], t.k[1], t.k[2]);
            pass = std::fabs(r.mid) <= 1.5 && (t.expect <= 0 || std::fabs(r.per - t.expect) <= 0.03 * t.expect + 0.02);
        }
        if (t.expect > 0) {
            ++checked;
        }
        if (!pass) {
            ++failed;
            fails += std::string(" ") + t.id;
        }
        std::printf("RESULT scalar %-15s cycles_per_pattern %.3f [%.3f-%.3f] loop %.0f%s%s  (%s)\n", t.id, r.per, r.lo,
                    r.hi, r.ovh, t.expect > 0 ? (pass ? " ok" : " MISMATCH") : "",
                    std::fabs(r.mid) <= 1.5 ? "" : " NONLINEAR", t.what);
    }
    ok = failed == 0;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%zu patterns, %d with a known answer; %d mismatches or non-linear loops:%s",
                  ts.size(), checked, failed, failed ? fails.c_str() : " none");
    verdict("core_scalar", ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}
