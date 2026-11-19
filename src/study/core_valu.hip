// Lane 1, items 1 and 2: calibration, then throughput against latency for each vector instruction class.
//
// For each instruction the block holds 16 patterns spread over c = 1, 2, 4, 8 or 16 independent chains
// (core_common.h). c = 1 is one dependent chain: cycles per op = the latency as seen by a dependent instruction.
// c = 16 gives enough independent work to reach the issue rate: cycles per op = 1 / throughput. The intermediate
// values show where the dependency stops limiting.
//
// Calibration (item 1): "calib_c" is the first-hour loop of src/wmma_time.hip (32 v_fma_f32 per iteration over 8
// chains, loop control included, built with dual issue off); the first hour measured 1.08 cycles per FMA. The
// inline-assembly FMA at c = 16 must come out at 1.00 (one full-rate op per cycle, loop control subtracted).
//
// Usage: core_valu [--break]. --break swaps the calibration kernel for a dependent chain while still expecting the
// independent rate, so the calibration verdict must FAIL.
#include "core_common.h"

#include <map>

typedef unsigned int u32;
typedef unsigned long long u64;

// Pattern shapes. %i is the chain register, %16 and %17 the constants (k0, k1).
#define P_D_D(op, i) op " %" #i ", %" #i "\n"                       // d = op(d)
#define P_D_DK(op, i) op " %" #i ", %" #i ", %16\n"                 // d = d op k0
#define P_D_KD(op, i) op " %" #i ", %16, %" #i "\n"                 // d = k0 op d (shift amount first)
#define P_D_DKL(op, i) op " %" #i ", %" #i ", %16, %17\n"           // d = fma(d, k0, k1)
#define P_D_KLD(op, i) op " %" #i ", %16, %17, %" #i "\n"           // d = k0 * k1 + d (three-operand forms)
#define P_D_KL(op, i) op " %" #i ", %16, %17\n"                    // d += k0 * k1 (two-source accumulate)
#define P_D_DD(op, i) op " %" #i ", %" #i ", %" #i "\n"             // d = d op d
#define P_MIX(op, i) op " %" #i ", %16, %17, %" #i " op_sel_hi:[1,1,0]\n"  // f32 += f16(k0.lo) * f16(k1.lo)
#define P_CND(op, i) op " %" #i ", %16, %" #i ", vcc_lo\n"          // d = vcc ? d : k0 (vcc fixed)
#define P_PL16(op, i) op " %" #i ", %" #i ", %16, %17\n"            // permlane16 with SGPR selects
#define P_DPP(op, i) "v_mov_b32_dpp %" #i ", %" #i " " op "\n"       // op carries the DPP control
#define P_MAD64(op, i) op " %" #i ", null, %16, %17, %" #i "\n"     // u64 d = k0 * k1 + d
// compare + select as one pattern (two instructions): d = (d < k0) ? k1 : d, the select reading the compare's VCC
#define P_CMPSEL(op, i) op " vcc_lo, %" #i ", %16\nv_cndmask_b32 %" #i ", %" #i ", %17, vcc_lo\n"
// float <-> int round trip as one pattern (two instructions)
#define P_RT(op, i) op "\n"

// X(id, asm op, chain type, constant type, pattern, instructions per pattern, init, LLVM GFX12SpeedModel latency
// for this class (research-10), class label)
#define FLT(v) core::f32(v)
#define F_ONE \
    core::Init { FLT(1.0f), FLT(1.0f), FLT(0.0f) }
#define F_ADD \
    core::Init { FLT(1.0f), FLT(0.0f), FLT(0.0f) }
#define I_SM \
    core::Init { 3, 0, 1 }
#define H_ONE \
    core::Init { core::h2(1.0f, 1.0f), core::h2(1.0f, 1.0f), core::h2(0.0f, 0.0f) }
#define D_ONE \
    core::Init { core::f64(1.0), core::f64(1.0), core::f64(0.0) }

#define OPS(X)                                                                                          \
    X(fadd, "v_add_f32", float, float, P_D_DK, 1, F_ADD, 5, "fp32")                                     \
    X(fmul, "v_mul_f32", float, float, P_D_DK, 1, F_ONE, 5, "fp32")                                     \
    X(ffma, "v_fma_f32", float, float, P_D_DKL, 1, F_ONE, 5, "fp32")                                    \
    X(ffmac, "v_fmac_f32", float, float, P_D_KL, 1, F_ADD, 5, "fp32")                                  \
    X(fminnum, "v_min_num_f32", float, float, P_D_DK, 1, F_ONE, 5, "fp32")                              \
    X(fmaxnum, "v_max_num_f32", float, float, P_D_DK, 1, F_ONE, 5, "fp32")                              \
    X(fminimum, "v_minimum_f32", float, float, P_D_DK, 1, F_ONE, 5, "fp32")                             \
    X(fmaximum, "v_maximum_f32", float, float, P_D_DK, 1, F_ONE, 5, "fp32")                             \
    X(fcmpsel, "v_cmp_lt_f32", float, float, P_CMPSEL, 2, F_ONE, 5, "fp32")                             \
    X(cndmask, "v_cndmask_b32", float, float, P_CND, 1, F_ONE, 5, "fp32")                               \
    X(iadd, "v_add_nc_u32", u32, u32, P_D_DK, 1, I_SM, 5, "int")                                        \
    X(ishl, "v_lshlrev_b32", u32, u32, P_D_KD, 1, I_SM, 5, "int")                                       \
    X(iand, "v_and_b32", u32, u32, P_D_DK, 1, I_SM, 5, "int")                                           \
    X(ior, "v_or_b32", u32, u32, P_D_DK, 1, I_SM, 5, "int")                                             \
    X(imul24, "v_mul_u32_u24", u32, u32, P_D_DK, 1, I_SM, 5, "int")                                     \
    X(imullo, "v_mul_lo_u32", u32, u32, P_D_DK, 1, I_SM, 6, "int")                                      \
    X(imulhi, "v_mul_hi_u32", u32, u32, P_D_DK, 1, I_SM, 6, "int")                                      \
    X(imad64, "v_mad_co_u64_u32", u64, u32, P_MAD64, 1, I_SM, 6, "int64")                               \
    X(ishl64, "v_lshlrev_b64", u64, u32, P_D_KD, 1, I_SM, 6, "int64")                                   \
    X(pkadd, "v_pk_add_f16", u32, u32, P_D_DK, 1, H_ONE, 5, "fp16x2")                                   \
    X(pkmul, "v_pk_mul_f16", u32, u32, P_D_DK, 1, H_ONE, 5, "fp16x2")                                   \
    X(pkfma, "v_pk_fma_f16", u32, u32, P_D_DKL, 1, H_ONE, 5, "fp16x2")                                  \
    X(fmamix, "v_fma_mix_f32", float, u32, P_MIX, 1, H_ONE, 5, "mixed")                                 \
    X(dot2f16, "v_dot2_f32_f16", float, u32, P_D_KLD, 1, H_ONE, 5, "dot")                               \
    X(dot2bf16, "v_dot2_f32_bf16", float, u32, P_D_KLD, 1, H_ONE, 5, "dot")                             \
    X(cvtf32i32, "v_cvt_f32_i32", u32, u32, P_D_D, 1, I_SM, 5, "cvt")                                   \
    X(cvti32f32, "v_cvt_i32_f32", u32, u32, P_D_D, 1, I_SM, 5, "cvt")                                   \
    X(cvtf16f32, "v_cvt_f16_f32", u32, u32, P_D_D, 1, I_SM, 5, "cvt")                                   \
    X(cvtf32f16, "v_cvt_f32_f16", u32, u32, P_D_D, 1, I_SM, 5, "cvt")                                   \
    X(cvtpkrtz, "v_cvt_pk_rtz_f16_f32", u32, u32, P_D_DK, 1, I_SM, 5, "cvt")                            \
    X(rcp, "v_rcp_f32", float, float, P_D_D, 1, F_ONE, 9, "trans")                                      \
    X(rsq, "v_rsq_f32", float, float, P_D_D, 1, F_ONE, 9, "trans")                                      \
    X(sqrt, "v_sqrt_f32", float, float, P_D_D, 1, F_ONE, 9, "trans")                                    \
    X(exp, "v_exp_f32", float, float, P_D_D, 1, F_ONE, 9, "trans")                                      \
    X(log, "v_log_f32", float, float, P_D_D, 1, F_ONE, 9, "trans")                                      \
    X(sin, "v_sin_f32", float, float, P_D_D, 1, F_ADD, 9, "trans")                                      \
    X(cos, "v_cos_f32", float, float, P_D_D, 1, F_ADD, 9, "trans")                                      \
    X(rcpf16, "v_rcp_f16", u32, u32, P_D_D, 1, H_ONE, 9, "trans")                                       \
    X(dadd, "v_add_f64", double, double, P_D_DK, 1, D_ONE, 38, "fp64")                                  \
    X(dmul, "v_mul_f64", double, double, P_D_DK, 1, D_ONE, 38, "fp64")                                  \
    X(dfma, "v_fma_f64", double, double, P_D_DKL, 1, D_ONE, 38, "fp64")                                 \
    X(rcpf64, "v_rcp_f64", double, double, P_D_D, 1, D_ONE, 40, "fp64")                                 \
    X(perm, "v_perm_b32", u32, u32, P_D_DKL, 1, I_SM, 5, "lane")                                        \
    X(dpp_qp, "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", u32, u32, P_DPP, 1, I_SM, 5, "lane")    \
    X(dpp8, "dpp8:[1,0,3,2,5,4,7,6]", u32, u32, P_DPP, 1, I_SM, 5, "lane")                              \
    X(pl16var, "v_permlane16_var_b32", u32, u32, P_D_DK, 1, I_SM, 5, "lane")                            \
    X(plx16var, "v_permlanex16_var_b32", u32, u32, P_D_DK, 1, I_SM, 5, "lane")

#define DEFINE_OP(id, op, TX, TK, P, npi, init, model, cls)                                   \
    CORE_KERNEL(id##_c1, TX, TK, "+v", "v", CORE_B1(P, op), "")                              \
    CORE_KERNEL(id##_c2, TX, TK, "+v", "v", CORE_B2(P, op), "")                              \
    CORE_KERNEL(id##_c4, TX, TK, "+v", "v", CORE_B4(P, op), "")                              \
    CORE_KERNEL(id##_c8, TX, TK, "+v", "v", CORE_B8(P, op), "")                              \
    CORE_KERNEL(id##_c16, TX, TK, "+v", "v", CORE_B16(P, op), "")
OPS(DEFINE_OP)

// Forms whose constants come from SGPRs ("s" constraint): v_permlane16_b32 and v_permlanex16_b32 must take their
// lane selects from SGPRs; fadd_s and ffma_s are the FP32 add and FMA with one and two SGPR operands, to see whether
// reading SGPRs costs vector issue cycles.
#define SGPR_OP(id, op, P, TX)                                                  \
    CORE_KERNEL(id##_c1, TX, TX, "+v", "s", CORE_B1(P, op), "")                \
    CORE_KERNEL(id##_c2, TX, TX, "+v", "s", CORE_B2(P, op), "")                \
    CORE_KERNEL(id##_c4, TX, TX, "+v", "s", CORE_B4(P, op), "")                \
    CORE_KERNEL(id##_c8, TX, TX, "+v", "s", CORE_B8(P, op), "")                \
    CORE_KERNEL(id##_c16, TX, TX, "+v", "s", CORE_B16(P, op), "")
SGPR_OP(pl16, "v_permlane16_b32", P_PL16, u32)
SGPR_OP(plx16, "v_permlanex16_b32", P_PL16, u32)
SGPR_OP(fadd_s, "v_add_f32", P_D_DK, float)
SGPR_OP(ffma_s, "v_fma_f32", P_D_DKL, float)

// ds_bpermute_b32 goes through the LDS crossbar: for one chain each pattern waits for its own result; for
// independent chains the 16 requests are issued and one wait closes the block. The address operand is k0.
#define P_BPERM(op, i) op " %" #i ", %16, %" #i "\n"
#define P_BPERM_W(op, i) op " %" #i ", %16, %" #i "\ns_wait_dscnt 0\n"
CORE_KERNEL(bperm_c1, u32, u32, "+v", "v", CORE_B1(P_BPERM_W, "ds_bpermute_b32"), "")
CORE_KERNEL(bperm_c16, u32, u32, "+v", "v", CORE_B16(P_BPERM, "ds_bpermute_b32"), "s_wait_dscnt 0\n")

// Pseudo-scalar transcendentals: SGPR in, SGPR out, VALU pipeline. Independent form: 16 destinations from one
// constant SGPR (never written, so no SGPR read hazard). Chained form: each result feeds the next; the guide's SGPR
// rule (a VALU-written SGPR read again by a VALU after a VALU read of that pair) requires s_wait_alu va_sdst(0)
// before the reuse, so the chain includes it, as the compiler emits it.
#define P_S_K(op, i) op " %" #i ", %16\n"
#define P_S_DW(op, i) op " %" #i ", %" #i "\ns_wait_alu depctr_va_sdst(0)\n"
#define PSEUDO(id, op)                                                                \
    CORE_KERNEL(id##_c1, float, float, "+s", "s", CORE_B1(P_S_DW, op), "")           \
    CORE_KERNEL(id##_c16, float, float, "+s", "s", CORE_B16(P_S_K, op), "")
PSEUDO(s_rcp, "v_s_rcp_f32")
PSEUDO(s_rsq, "v_s_rsq_f32")
PSEUDO(s_sqrt, "v_s_sqrt_f32")
PSEUDO(s_exp, "v_s_exp_f32")
PSEUDO(s_log, "v_s_log_f32")

// The first hour's calibration loop, unchanged (src/wmma_time.hip, kind "valu", 32 FMAs per iteration).
__global__ __launch_bounds__(32) void calib_c(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,
                                              uint32_t * __restrict__ sink) {
    constexpr int IT = 512;
    float f[8];
    for (int i = 0; i < 8; ++i) {
        f[i] = 1.0f;
    }
    float m = core::from_bits<float>(in[1]), c = core::from_bits<float>(in[2]);
    asm volatile("" : "+v"(m), "+v"(c));
    for (int rep = 0; rep < core::REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 16
        for (int it = 0; it < IT; ++it) {
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                f[k & 7] = fmaf(f[k & 7], m, c);
            }
        }
        float dep;
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep) : "v"(f[7]));
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
        if (dep == 1234.5f) {
            sink[0] = 1;
        }
    }
    float s = 0;
    for (int i = 0; i < 8; ++i) {
        s += f[i];
    }
    sink[threadIdx.x] = __float_as_uint(s);
}

struct Op {
    std::string id, cls;
    core::Kern k[5][3];  // chains 1, 2, 4, 8, 16; U = ULO, UMID, UHI (null where a chain count is not built)
    int npi;
    core::Init init;
    double model;
};

static const int CHAINS[5] = {1, 2, 4, 8, 16};

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = core::flag(argc, argv, "--break");
    std::vector<Op> ops;
#define REG_OP(id, op, TX, TK, P, npi, init, model, cls)                                                       \
    ops.push_back(Op{#id, cls,                                                                                 \
                     {{id##_c1<core::ULO>, id##_c1<core::UMID>, id##_c1<core::UHI>},                                                                \
                      {id##_c2<core::ULO>, id##_c2<core::UMID>, id##_c2<core::UHI>},                                                                \
                      {id##_c4<core::ULO>, id##_c4<core::UMID>, id##_c4<core::UHI>},                                                                \
                      {id##_c8<core::ULO>, id##_c8<core::UMID>, id##_c8<core::UHI>},                                                                \
                      {id##_c16<core::ULO>, id##_c16<core::UMID>, id##_c16<core::UHI>}},                                                             \
                     npi, init, model});
    OPS(REG_OP)
    REG_OP(pl16, "", u32, u32, P_PL16, 1, (core::Init{3, 0x76543210u, 0xfedcba98u}), 5, "lane")
    REG_OP(plx16, "", u32, u32, P_PL16, 1, (core::Init{3, 0x76543210u, 0xfedcba98u}), 5, "lane")
    REG_OP(fadd_s, "", float, float, P_D_DK, 1, F_ADD, 5, "fp32")
    REG_OP(ffma_s, "", float, float, P_D_DKL, 1, F_ONE, 5, "fp32")
#define REG_2(id, cls, model, init) \
    ops.push_back(Op{#id, cls, {{id##_c1<core::ULO>, id##_c1<core::UMID>, id##_c1<core::UHI>}, {}, {}, {}, {id##_c16<core::ULO>, id##_c16<core::UMID>, id##_c16<core::UHI>}}, 1, init, model});
    REG_2(bperm, "lane", 20, (core::Init{3, 4, 0}))
    REG_2(s_rcp, "pseudo", 8, F_ONE)
    REG_2(s_rsq, "pseudo", 8, F_ONE)
    REG_2(s_sqrt, "pseudo", 8, F_ONE)
    REG_2(s_exp, "pseudo", 8, F_ONE)
    REG_2(s_log, "pseudo", 8, F_ONE)

    bool ok = true;
    std::string text;
    char buf[512];

    // Item 1: calibration.
    core::set_init(core::Init{FLT(1.0f), FLT(0.999f), FLT(0.001f)});
    const Stats cal = stats_of(core::run_one_wave(calib_c));
    const double cal_fma = cal.median / (512.0 * 32);
    std::printf("RESULT calib_c cycles_per_fma %.3f [%.3f-%.3f] (first hour: 1.08)\n", cal_fma, cal.min / (512.0 * 32),
                cal.max / (512.0 * 32));
    core::set_init(F_ONE);
    const core::Cost fcal = brk ? core::pattern_cost(ffma_c1<core::ULO>, ffma_c1<core::UMID>, ffma_c1<core::UHI>) : core::pattern_cost(ffma_c16<core::ULO>, ffma_c16<core::UMID>, ffma_c16<core::UHI>);
    std::printf("RESULT calib_asm%s cycles_per_fma %.3f [%.3f-%.3f] loop control %.1f cycles/iteration (expect 1.00)\n",
                brk ? "(BROKEN: dependent chain)" : "", fcal.per, fcal.lo, fcal.hi, fcal.ovh);
    const bool cal_ok = std::fabs(cal_fma - 1.08) <= 0.05 && std::fabs(fcal.per - 1.0) <= 0.03;
    std::snprintf(buf, sizeof buf, "first-hour loop %.3f cycles/FMA (known 1.08), asm differential %.3f (expect 1.00)",
                  cal_fma, fcal.per);
    verdict("core_calibration", cal_ok ? "PASS" : "FAIL", buf);
    ok = ok && cal_ok;
    if (brk) {
        return cal_ok ? 0 : 1;
    }

    // Item 2.
    int nonlinear = 0, inverted = 0;
    double ovh_max = -1e9;
    for (const Op & o : ops) {
        core::set_init(o.init);
        double lat = 0, thr = 1e9;
        std::printf("RESULT op %-9s class %-6s", o.id.c_str(), o.cls.c_str());
        for (int c = 0; c < 5; ++c) {
            if (!o.k[c][0]) {
                continue;
            }
            const core::Cost r = core::pattern_cost(o.k[c][0], o.k[c][1], o.k[c][2], core::PATTERNS * o.npi);
            std::printf(" c%d %.2f [%.2f-%.2f] loop %.0f;", CHAINS[c], r.per, r.lo, r.hi, r.ovh);
            ovh_max = std::max(ovh_max, r.ovh);
            if (std::fabs(r.mid) > 1.5) {
                ++nonlinear;
                std::printf(" NONLINEAR(%.1f)", r.mid);
            }
            if (c == 0) {
                lat = r.per;
            }
            thr = std::min(thr, r.per);
        }
        std::printf("\n");
        std::printf("SUMMARY %-9s %-6s latency_cycles %.2f throughput_cycles_per_op %.2f ops_per_cycle %.3f llvm_model %.0f\n",
                    o.id.c_str(), o.cls.c_str(), lat * o.npi, thr * o.npi, 1.0 / (thr * o.npi), o.model);
        if (lat + 0.1 < thr) {
            ++inverted;
        }
    }
    std::snprintf(buf, sizeof buf, "%zu instruction classes; %d chain counts not linear in block count; %d classes "
                  "where the dependent chain was faster than the independent streams; largest loop control cost %.1f cycles "
                  "per iteration", ops.size(), nonlinear, inverted, ovh_max);
    text = buf;
    ok = ok && nonlinear == 0 && inverted == 0;
    verdict("core_valu", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
