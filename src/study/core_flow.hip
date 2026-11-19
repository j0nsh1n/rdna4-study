// Lane 1, control flow: what branches, loop back-edges and execution-mask changes cost, one wave alone on its SIMD.
//
// Method as in core_common.h (16-pattern blocks, U = 4/6/8, pattern_cost differential, linearity check). A "pattern"
// here is the whole group named in the test (for example compare + branch), so the result is cycles per group. The
// kernels are CORE_KERNEL with extra clobbers: scc and the scratch SGPRs s20/s21 that the patterns write. exec cannot
// be declared (it is reserved; the clobber is ignored with a warning), so every pattern restores exec to all lanes
// itself; the timed loop runs with all 32 lanes on.
// Differences between tests (branch cost = "compare + branch" minus "compare alone") are derived by reading the
// RESULT lines; nothing is subtracted in the program.
//
// Usage: core_flow [--break]. --break expects 2.00 cycles for the independent FMA baseline, so the verdict must FAIL.
#include "core_common.h"

typedef unsigned int u32;

// 16 copies of a pattern string that does not depend on the chain register.
#define REP16(s) s s s s s s s s s s s s s s s s

// Chain patterns: %i is chain register i, %16 and %17 the constants.
#define P_FMA(op, i) op " %" #i ", %" #i ", %16, %17\n"
#define P_ADD(op, i) op " %" #i ", %" #i ", %16\n"
#define P_FMAK(op, i) op " %" #i ", %16, %17, %16\n"                // writes chain i, reads only constants
#define P_FMA4(op, i) op " %" #i ", %" #i ", %" #i ", %" #i "\n"     // no SGPR operands (constants are masks)
#define P_SKIP(op, i) "s_branch 1f\n" op " %" #i ", %" #i ", %16\n1:\n"
#define P_LOOP4(op, i) "s_mov_b32 s20, 4\n1:\ns_add_co_i32 s20, s20, -1\ns_cmp_lg_u32 s20, 0\ns_cbranch_scc1 1b\n"
#define P_LOOP4F(op, i) "s_mov_b32 s20, 4\n1:\n" op " %" #i ", %16, %17, %16\ns_add_co_i32 s20, s20, -1\n" \
                        "s_cmp_lg_u32 s20, 0\ns_cbranch_scc1 1b\n"
#define P_LOOP2(op, i) "s_mov_b32 s20, 2\n1:\ns_add_co_i32 s20, s20, -1\ns_cmp_lg_u32 s20, 0\ns_cbranch_scc1 1b\n"
#define P_LOOP2F(op, i) "s_mov_b32 s20, 2\n1:\n" op " %" #i ", %16, %17, %16\ns_add_co_i32 s20, s20, -1\n" \
                        "s_cmp_lg_u32 s20, 0\ns_cbranch_scc1 1b\n"
#define P_SAVEEXEC(op, i) "s_and_saveexec_b32 s20, %16\n" op " %" #i ", %" #i ", %" #i ", %" #i "\n" \
                          "s_or_b32 exec_lo, exec_lo, s20\n"

// Eight independent FMAs on chains 0..7 (the pattern group of item 9).
#define F8 "v_fma_f32 %0, %0, %0, %0\nv_fma_f32 %1, %1, %1, %1\nv_fma_f32 %2, %2, %2, %2\nv_fma_f32 %3, %3, %3, %3\n" \
           "v_fma_f32 %4, %4, %4, %4\nv_fma_f32 %5, %5, %5, %5\nv_fma_f32 %6, %6, %6, %6\nv_fma_f32 %7, %7, %7, %7\n"

#define FLT(v) core::f32(v)
#define F_ONE \
    core::Init { FLT(1.0f), FLT(1.0f), FLT(0.0f) }
#define F_ADD \
    core::Init { FLT(1.0f), FLT(0.0f), FLT(0.0f) }
#define M_ALL \
    core::Init { FLT(1.0f), 0xffffffffu, 0 }
#define M_HALF \
    core::Init { FLT(1.0f), 0x0000ffffu, 0 }

// Like CORE_KERNEL, with the scratch SGPRs declared as clobbered (exec: see the top of the file).
#define FLOW_KERNEL(NAME, TX, TK, XC, KC, BODY)                                                                   \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,      \
                                               uint32_t * __restrict__ sink) {                                    \
        TX x[16];                                                                                                 \
        for (int i = 0; i < 16; ++i) {                                                                            \
            x[i] = core::from_bits<TX>(in[0]);                                                                    \
        }                                                                                                         \
        TK k0 = core::from_bits<TK>(in[1]), k1 = core::from_bits<TK>(in[2]);                                      \
        asm volatile("" : "+" KC(k0), "+" KC(k1));                                                                \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                              \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                        \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(BODY : CORE_X16(XC) : KC(k0), KC(k1) : "vcc", "scc", "s20", "s21");      \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        uint32_t s = 0;                                                                                           \
        for (int i = 0; i < 16; ++i) {                                                                            \
            s ^= core::fold(x[i]);                                                                                \
        }                                                                                                         \
        sink[threadIdx.x] = s;                                                                                    \
    }

// X(id, constant type, constant constraint, block text, init, item)
#define TESTS(X)                                                                                                    \
    X(fma_ind, float, "v", CORE_B16(P_FMA, "v_fma_f32"), F_ONE, "1")                                                \
    X(fma_dep, float, "v", CORE_B1(P_FMA, "v_fma_f32"), F_ONE, "1")                                                 \
    X(scmp, float, "v", REP16("s_cmp_eq_u32 0, 0\n"), F_ONE, "2")                                                   \
    X(br_nottaken, float, "v", REP16("s_cmp_eq_u32 0, 1\ns_cbranch_scc1 1f\n1:\n"), F_ONE, "3")                      \
    X(br_taken, float, "v", REP16("s_cmp_eq_u32 0, 0\ns_cbranch_scc1 1f\n1:\n"), F_ONE, "4")                         \
    X(sbranch, float, "v", REP16("s_branch 1f\n1:\n"), F_ONE, "5")                                                  \
    X(add_ind, float, "v", CORE_B16(P_ADD, "v_add_f32"), F_ADD, "6")                                                \
    X(sbranch_skip_add, float, "v", CORE_B16(P_SKIP, "v_add_f32"), F_ADD, "6")                                      \
    X(salu_pair, float, "v", REP16("s_add_co_i32 s20, s20, -1\ns_cmp_lg_u32 s20, 0\n"), F_ONE, "7")                  \
    X(fma_k, float, "v", CORE_B16(P_FMAK, "v_fma_f32"), F_ONE, "7")                                                 \
    X(loop4, float, "v", CORE_B16(P_LOOP4, ""), F_ONE, "7")                                                         \
    X(loop4_fma, float, "v", CORE_B16(P_LOOP4F, "v_fma_f32"), F_ONE, "7")                                           \
    X(loop2, float, "v", CORE_B16(P_LOOP2, ""), F_ONE, "7")                                                         \
    X(loop2_fma, float, "v", CORE_B16(P_LOOP2F, "v_fma_f32"), F_ONE, "7")                                           \
    X(fma_self, float, "v", CORE_B16(P_FMA4, "v_fma_f32"), F_ONE, "8")                                              \
    X(exec_write, float, "v", REP16("s_mov_b32 exec_lo, -1\n"), F_ONE, "9")                    \
    X(fma8_plain, float, "v", REP16(F8), F_ONE, "9")                                                                \
    X(fma8_exec1, float, "v", REP16("s_mov_b32 exec_lo, -1\n" F8 "s_mov_b32 exec_lo, -1\n"), F_ONE, "9")            \
    X(fma8_exec0, float, "v", REP16("s_mov_b32 exec_lo, 0\n" F8 "s_mov_b32 exec_lo, -1\n"), F_ONE, "9")             \
    X(skip_execz_taken, float, "v",                                                                                 \
      REP16("s_mov_b32 exec_lo, 0\ns_cbranch_execz 1f\n" F8 "1:\ns_mov_b32 exec_lo, -1\n"), F_ONE, "9")             \
    X(skip_execz_nottaken, float, "v",                                                                              \
      REP16("s_mov_b32 exec_lo, -1\ns_cbranch_execz 1f\n" F8 "1:\ns_mov_b32 exec_lo, -1\n"), F_ONE, "9")            \
    X(execz_taken_nov, float, "v", REP16("s_mov_b32 exec_lo, 0\ns_cbranch_execz 1f\n1:\ns_mov_b32 exec_lo, -1\n"),   \
      F_ONE, "9")

#define DEFINE_T(id, TK, KC, body, init, item) FLOW_KERNEL(id, float, TK, "+v", KC, body)
TESTS(DEFINE_T)
// Item 8 uses the mask as an SGPR operand: one kernel, run with two initial values.
FLOW_KERNEL(saveexec, float, u32, "+v", "s", CORE_B16(P_SAVEEXEC, "v_fma_f32"))

struct Test {
    const char * name;
    const char * item;
    core::Init init;
    core::Kern k[3];
};

struct Res {
    std::string name;
    core::Cost c;
};

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = core::flag(argc, argv, "--break");
    std::vector<Test> tests;
#define REG_T(id, TK, KC, body, init, item) \
    tests.push_back(Test{#id, item, init, {id<core::ULO>, id<core::UMID>, id<core::UHI>}});
    TESTS(REG_T)
    tests.push_back(Test{"saveexec_all", "8", M_ALL, {saveexec<core::ULO>, saveexec<core::UMID>, saveexec<core::UHI>}});
    tests.push_back(Test{"saveexec_half", "8", M_HALF, {saveexec<core::ULO>, saveexec<core::UMID>, saveexec<core::UHI>}});

    int nonlinear = 0;
    double ind = -1, dep = -1;
    for (const Test & t : tests) {
        core::set_init(t.init);
        const core::Cost r = core::pattern_cost(t.k[0], t.k[1], t.k[2]);
        std::printf("RESULT flow %s cycles_per_pattern %.2f [%.2f-%.2f] loop %.1f (item %s)", t.name, r.per, r.lo, r.hi,
                    r.ovh, t.item);
        if (std::fabs(r.mid) > 1.5) {
            ++nonlinear;
            std::printf(" NONLINEAR(%.1f)", r.mid);
        }
        std::printf("\n");
        if (std::strcmp(t.name, "fma_ind") == 0) {
            ind = r.per;
        }
        if (std::strcmp(t.name, "fma_dep") == 0) {
            dep = r.per;
        }
    }
    const double want_ind_lo = brk ? 1.97 : 0.97, want_ind_hi = brk ? 2.03 : 1.03;
    const bool ind_ok = ind >= want_ind_lo && ind <= want_ind_hi;
    const bool dep_ok = dep >= 4.9 && dep <= 5.1;
    const bool ok = ind_ok && dep_ok && nonlinear == 0;
    char buf[400];
    std::snprintf(buf, sizeof buf, "independent FMA %.3f (expect %s), dependent FMA %.3f (expect 5.00 within 4.9-5.1), "
                  "%d of %zu tests not linear in block count", ind, brk ? "2.00 [--break]" : "1.00 within 0.97-1.03", dep,
                  nonlinear, tests.size());
    verdict("core_flow", ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}
