// Timing leftovers, item 2: the cost around load waits (matrix-timing.md items 5e and 8).
//
// Usage: tim_wait run [--break]
//   One wave. Each pattern below is a short sequence that ends with nothing outstanding; it is timed with lane 1's
//   differential method (loops of R = 2, 4, 6 patterns per iteration, cost = (R6 - R2) / 4 patterns), so the loop
//   control cancels. Loads are 4-byte global loads that hit L0 (all lanes read one 128-byte line) or 4-byte LDS
//   loads. "fill" is 12 x s_nop 15 (192 cycles alone), long enough for an L0 hit (81 cycles, lane 2) to be back.
//   What each comparison isolates:
//     wave waited or not:  ld+wait+X against ld+wait (the wave sleeps in the wait) and ld+fill+wait+X against
//                          ld+fill+X+wait (data already back when the wait is reached)
//     kind of X:           s_nop 0 (scalar), v_fma_f32 (vector), v_wmma f16 (matrix), another load, ds_load
//     outstanding loads:   1, 4, 16 loads before the wait
//     LDS against global:  ds_load_b32 + s_wait_dscnt
//     wait with nothing outstanding: fill + wait + X against fill + X
//     other ways off the SIMD: s_sleep 1, and a barrier in a one-wave group (nobody to wait for)
//     data arriving under a matrix chain: load, 8 FP16 matrix instructions, wait, against the chain alone; the same
//                          with 128 FMAs instead of the chain
//   --break claims the independent FMA costs 2 cycles: the calibration verdict must FAIL.
#include "tim_common.h"

#include <map>

namespace {

constexpr int REPS = 6;
constexpr int LAUNCHES = 3;
constexpr uint32_t NIT = 16;

#define FMA1 "v_fma_f32 v16, v16, 0.5, 0.5\n"
#define FMA4 FMA1 "v_fma_f32 v17, v17, 0.5, 0.5\n v_fma_f32 v18, v18, 0.5, 0.5\n v_fma_f32 v19, v19, 0.5, 0.5\n"
#define FMA16 TIM_FMA16
#define DEP26 ".rept 26\n v_fma_f32 v16, v16, 0.5, 0.5\n .endr\n"
#define FMA64 TIM_FMA16 TIM_FMA16 TIM_FMA16 TIM_FMA16
#define GLD128 "global_load_b128 v[40:43], v[60:61], off\n"
#define MM16 MM8 MM8
#define GLD "global_load_b32 v40, v[60:61], off\n"
#define GLD4 "global_load_b32 v40, v[60:61], off\n global_load_b32 v41, v[60:61], off offset:128\n" \
             " global_load_b32 v42, v[60:61], off offset:256\n global_load_b32 v43, v[60:61], off offset:384\n"
#define GLD16 GLD4 GLD4 GLD4 GLD4
#define WL "s_wait_loadcnt 0\n"
#define LLD "ds_load_b32 v41, v62\n"
#define WD "s_wait_dscnt 0\n"
#define FILL ".rept 12\n s_nop 15\n .endr\n"
#define MM "v_wmma_f32_16x16x16_f16 v[64:71], v[72:75], v[76:79], v[64:71]\n"
#define MM8 MM MM MM MM MM MM MM MM
#define SADD "s_add_co_i32 s16, s16, 1\n"
#define NOP "s_nop 0\n"
#define BAR "s_barrier_signal -1\n s_barrier_wait -1\n"
#define SLEEP "s_sleep 1\n"

// name, text
#define TIM_WAIT_PATTERNS(X)                         \
    X(fma, FMA1)                                     \
    X(fma16, FMA16)                                  \
    X(nop, NOP)                                      \
    X(sadd, SADD)                                    \
    X(mm, MM)                                        \
    X(mm8, MM8)                                      \
    X(g_wait, GLD WL)                                \
    X(g_wait_nop, GLD WL NOP)                        \
    X(g_wait_sadd, GLD WL SADD)                      \
    X(g_wait_fma, GLD WL FMA1)                       \
    X(g_wait_fma4, GLD WL FMA4)                      \
    X(g_wait_mm, GLD WL MM)                          \
    X(g_wait_g, GLD WL GLD WL)                       \
    X(g_wait_l, GLD WL LLD WD)                       \
    X(g4_wait, GLD4 WL)                              \
    X(g4_wait_fma, GLD4 WL FMA1)                     \
    X(g16_wait, GLD16 WL)                            \
    X(g16_wait_fma, GLD16 WL FMA1)                   \
    X(g_fill_wait_fma, GLD FILL WL FMA1)             \
    X(g_fill_fma_wait, GLD FILL FMA1 WL)             \
    X(g_fill_wait_nop, GLD FILL WL NOP)              \
    X(g_fill_nop_wait, GLD FILL NOP WL)              \
    X(g_fill_wait_mm, GLD FILL WL MM)                \
    X(g_fill_mm_wait, GLD FILL MM WL)                \
    X(fill, FILL)                                    \
    X(fill_fma, FILL FMA1)                           \
    X(fill_wait_fma, FILL WL FMA1)                   \
    X(fill_waitd_fma, FILL WD FMA1)                  \
    X(l_wait, LLD WD)                                \
    X(l_wait_nop, LLD WD NOP)                        \
    X(l_wait_fma, LLD WD FMA1)                       \
    X(l_wait_mm, LLD WD MM)                          \
    X(l_fill_wait_fma, LLD FILL WD FMA1)             \
    X(l_fill_fma_wait, LLD FILL FMA1 WD)             \
    X(sleep, SLEEP)                                  \
    X(sleep_nop, SLEEP NOP)                          \
    X(sleep_fma, SLEEP FMA1)                         \
    X(sleep_mm, SLEEP MM)                            \
    X(bar, BAR)                                      \
    X(bar_nop, BAR NOP)                              \
    X(bar_fma, BAR FMA1)                             \
    X(g_mm8_wait, GLD MM8 WL)                        \
    X(l_mm8_wait, LLD MM8 WD)                        \
    X(g_dep26_wait, GLD DEP26 WL)                    \
    X(dep26, DEP26)                                  \
    X(g128_mm8_wait, GLD128 MM8 WL)                  \
    X(g4_mm8_wait, GLD4 MM8 WL)                      \
    X(mm8_g_wait, MM8 GLD WL)                        \
    X(g_mm16_wait, GLD MM16 WL)                      \
    X(mm16, MM16)                                    \
    X(sleep_fma64, SLEEP FMA64)                      \
    X(fma64, FMA64)                                  \
    X(g_wait_nop7_fma, GLD WL "s_nop 7\n" FMA1)       \
    X(g_wait_nop15_fma, GLD WL "s_nop 15\n" FMA1)     \
    X(g_wait_sadd_fma, GLD WL SADD FMA1)             \
    X(g_wait_g_fma, GLD WL GLD FMA1 WL)              \
    X(g_mm8_wait_fma, GLD MM8 WL FMA1)               \
    X(mm8_fma, MM8 FMA1)

#define TIM_WAIT_CLOBBERS                                                                                            \
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", \
        "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54",    \
        "v55", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v77",    \
        "v78", "v79", "s16", "scc", "memory"

typedef unsigned u2 __attribute__((ext_vector_type(2)));

#define TIM_WAIT_K(NAME, TEXT, R)                                                                                    \
    __global__ __launch_bounds__(32) void k_##NAME##_##R(const uint32_t * buf, uint32_t n_iter, uint32_t * out) {   \
        extern __shared__ uint32_t lds[];                                                                            \
        lds[threadIdx.x] = threadIdx.x;                                                                              \
        __syncthreads();                                                                                             \
        uint64_t p = (uint64_t) (buf + threadIdx.x);                                                                 \
        u2 addr = {(uint32_t) p, (uint32_t) (p >> 32)};                                                              \
        uint32_t la = threadIdx.x * 4;                                                                               \
        for (int rep = 0; rep < REPS; ++rep) {                                                                       \
            uint32_t t0, t1, cnt;                                                                                    \
            asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n"       \
                         ".rept " #R "\n" TEXT ".endr\n"                                                             \
                         " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"           \
                         " s_wait_loadcnt 0\n s_wait_dscnt 0\n"                                                      \
                         " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                     \
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), "+{v[60:61]}"(addr), "+{v62}"(la)    \
                         : [n] "s"(n_iter)                                                                           \
                         : TIM_WAIT_CLOBBERS);                                                                       \
            if (threadIdx.x == 0) {                                                                                  \
                out[rep] = t1 - t0;                                                                                  \
            }                                                                                                        \
        }                                                                                                            \
    }
#define TIM_WAIT_KS(NAME, TEXT) TIM_WAIT_K(NAME, TEXT, 2) TIM_WAIT_K(NAME, TEXT, 4) TIM_WAIT_K(NAME, TEXT, 6)
TIM_WAIT_PATTERNS(TIM_WAIT_KS)

using Kern = void (*)(const uint32_t *, uint32_t, uint32_t *);
struct Pat {
    const char * name;
    Kern k[3];
};
#define TIM_WAIT_ROW(NAME, TEXT) {#NAME, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6}},
const Pat pats[] = {TIM_WAIT_PATTERNS(TIM_WAIT_ROW)};

uint32_t *g_buf = nullptr, *g_out = nullptr;

std::vector<double> samples(Kern k) {
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 4096, 0, g_buf, NIT, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, REPS);
        for (int r = 1; r < REPS; ++r) {
            v.push_back(h[r]);
        }
    }
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = tim::flag(argc, argv, "--break");
    if (argc < 2 || std::strcmp(argv[1], "run") != 0) {
        std::fprintf(stderr, "usage: tim_wait run [--break]\n");
        return 2;
    }
    CHECK(hipMalloc(&g_buf, 1 << 16));
    CHECK(hipMemset(g_buf, 0, 1 << 16));
    g_out = tim::dev_u32(REPS);
    mem::clocks("start");
    std::map<std::string, double> c;
    size_t n = 0, dropped = 0, nonlin = 0;
    for (const Pat & p : pats) {
        tim::Clean m[3];
        for (int i = 0; i < 3; ++i) {
            m[i] = tim::clean_of(samples(p.k[i]));
            n += m[i].n;
            dropped += m[i].dropped;
        }
        const double per = (m[2].med - m[0].med) / (4.0 * NIT);
        const double mid = (m[1].med - 0.5 * (m[0].med + m[2].med)) / NIT;
        nonlin += std::fabs(mid) > 1.5;
        c[p.name] = per;
        std::printf("RESULT wait %-16s cycles_per_pattern %8.3f range %.3f-%.3f midpoint_dev_per_it %.2f\n", p.name, per,
                    (m[2].lo - m[0].hi) / (4.0 * NIT), (m[2].hi - m[0].lo) / (4.0 * NIT), mid);
        std::fflush(stdout);
    }
    std::printf("RESULT wait disturbance rule: %zu of %zu samples dropped; %zu patterns off the midpoint by > 1.5 cycles "
                "per iteration\n",
                dropped, n, nonlin);
    // Derived: the extra cycles of X after a wait, beyond the pattern without X and X's own issue cost (1 cycle for
    // an FMA, s_nop 0 or s_add; 16 for a matrix instruction; the pattern itself for a second load).
    auto ex = [&](const char * with, const char * without, double issue) {
        const double e = c[with] - c[without] - issue;
        std::printf("RESULT wait extra %-16s = %-16s - %-10s - %5.2f = %7.2f\n", with, with, without, issue, e);
        return e;
    };
    const double e1 = ex("g_wait_fma", "g_wait", 1);
    ex("g_wait_fma4", "g_wait", 4);
    ex("g_wait_nop", "g_wait", 1);
    ex("g_wait_sadd", "g_wait", 1);
    ex("g_wait_mm", "g_wait", 16);
    ex("g_wait_g", "g_wait", c["g_wait"]);
    ex("g_wait_l", "g_wait", c["l_wait"]);
    const double e4 = ex("g4_wait_fma", "g4_wait", 1);
    const double e16 = ex("g16_wait_fma", "g16_wait", 1);
    const double el = ex("l_wait_fma", "l_wait", 1);
    ex("l_wait_nop", "l_wait", 1);
    ex("l_wait_mm", "l_wait", 16);
    ex("bar_fma", "bar", 1);
    ex("bar_nop", "bar", 1);
    ex("fill_fma", "fill", 1);
    ex("fill_wait_fma", "fill", 2);
    ex("fill_waitd_fma", "fill", 2);
    const double e_nop15 = ex("g_wait_nop15_fma", "g_wait", 17);
    ex("g_wait_nop7_fma", "g_wait", 9);
    ex("g_wait_sadd_fma", "g_wait", 2);
    std::printf("RESULT wait data_back g_fill_wait_fma - g_fill_fma_wait = %.2f; nop %.2f; mm %.2f; lds fma %.2f\n",
                c["g_fill_wait_fma"] - c["g_fill_fma_wait"], c["g_fill_wait_nop"] - c["g_fill_nop_wait"],
                c["g_fill_wait_mm"] - c["g_fill_mm_wait"], c["l_fill_wait_fma"] - c["l_fill_fma_wait"]);
    std::printf("RESULT wait under_chain g_mm8_wait - mm8 = %.2f (load alone with wait %.2f); l_mm8_wait - mm8 = %.2f; "
                "g_dep26_wait - dep26 = %.2f; g_mm8_wait_fma - mm8_fma - (g_mm8_wait - mm8) = %.2f\n",
                c["g_mm8_wait"] - c["mm8"], c["g_wait"], c["l_mm8_wait"] - c["mm8"], c["g_dep26_wait"] - c["dep26"],
                c["g_mm8_wait_fma"] - c["mm8_fma"] - (c["g_mm8_wait"] - c["mm8"]));
    std::printf("RESULT wait under_chain2 g128_mm8_wait - mm8 = %.2f; g4_mm8_wait - mm8 = %.2f; mm8_g_wait - mm8 = %.2f; "
                "g_mm16_wait - mm16 = %.2f\n",
                c["g128_mm8_wait"] - c["mm8"], c["g4_mm8_wait"] - c["mm8"], c["mm8_g_wait"] - c["mm8"],
                c["g_mm16_wait"] - c["mm16"]);
    std::printf("RESULT wait gap g_wait_nop7_fma - g_wait = %.2f (8 + 1 if the extra hides behind the nop); "
                "g_wait_nop15_fma - g_wait = %.2f (16 + 1); g_wait_sadd_fma - g_wait = %.2f; sleep_fma64 - fma64 = %.2f "
                "(64 with no resume cost, s_sleep wakes on 64-cycle ticks)\n",
                c["g_wait_nop7_fma"] - c["g_wait"], c["g_wait_nop15_fma"] - c["g_wait"], c["g_wait_sadd_fma"] - c["g_wait"],
                c["sleep_fma64"] - c["fma64"]);
    const double want = brk ? 2.0 : 1.0;
    tim::check("tim_wait_calib", std::fabs(c["fma16"] / 16 - want) < 0.02 && std::fabs(c["mm"] - 16.0) < 0.3,
               "independent FMA %.3f cycles (expected %.2f%s), FP16 matrix chain %.2f (expected 16)", c["fma16"] / 16,
               want, brk ? ", deliberate break" : "", c["mm"]);
    const double back = c["g_fill_wait_fma"] - c["g_fill_fma_wait"];
    const double lo = std::min(std::min(e1, e4), std::min(e16, el)), hi = std::max(std::max(e1, e4), std::max(e16, el));
    tim::check("tim_wait_charge", lo > 10 && hi < 20 && hi - lo < 3 && e_nop15 > 10 && e_nop15 < 20 && std::fabs(back) < 1.5,
               "first FMA after a wait that blocked: %.1f (1 global load), %.1f (4), %.1f (16), %.1f (LDS) extra cycles "
               "(claim: 10-20, the same within 3); still %.1f with 16 cycles of s_nop between (claim: not hidden); "
               "data already back when the wait is reached: %.1f (claim: |x| < 1.5)",
               e1, e4, e16, el, e_nop15, back);
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
