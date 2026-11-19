// Leftovers C, lane A, CD-1, CD-2, CD-4: widened FP8 weights with dual-issue consumers, the LDS table method with
// loads issued ahead, and stochastic rounding with a random-number generator in the loop.
//
// Usage: left_cvt [--break]
//   Times blocks of src/study/cvt_time_blocks.h (scripts/cvt_gen_timing.py) with cvt_time.hip's method (lane 1's
//   differential: U = 4, 6, 8 copies of a block per iteration, U = 6 midway), here with verdicts:
//   cvt_calib       in the same run: v_fma_f32 1.00 / 5.00, (a) widening alone 1.00 per weight and with v_fmac_f32
//                   consumers 2.41 (+-3 %), v_cvt_sr_fp8_f32 1.00 / 5.00 (conversions items 3 and 4); every test linear
//                   except the two LDS table forms (e), whose samples are bimodal (18 to 37 cycles per pattern within
//                   one run, 4.6 or 5.8 per weight between runs for the simple form, 5.5 or 6.7 for the ahead form):
//                   they are printed, not judged for linearity
//   The verdicts of the three items are the issue bounds (one wave issues at most one instruction, or one dual pair,
//   per cycle; a cost below the bound means the probe is broken); how close each comes to its bound, and the
//   comparisons with the draft's predictions, are INFO:
//   cd1_dual        (a) and (b) with v_dual_fmac_f32 consumers: per weight >= 1.45 (4 conversions at 1.00 or 2 packed at
//                   2.00, plus 2 dual pairs: 6 cycles per 4 weights = 1.5, less 3 %); the draft's range ends at the plain
//                   consumer's 2.41
//   cd2_ahead       the LDS table with loads issued ahead: >= 3.45 (2.5 VALU and 1 LDS instruction per weight = 3.5,
//                   less 3 %); the draft's range ends at the simple form's 4.59
//   cd4_sr          conversion + generator: >= 0.97 x the sum of their issue costs (execution-core item 2: 1.00 per
//                   32-bit VALU, 4.00 for v_mul_lo_u32): xorshift 7, 24-bit LCG 3, 32-bit LCG 6, shift 2; the draft's
//                   "within 3 % of the sum" is INFO
//   --break         cd1_dual claims 1.00 per weight (below the issue bound): FAIL.
// Usage: left_cvt cd2 [--break]   (fix round, test T4)
//   The simple LDS form (use_e_lds_dot2), the ahead form (use_e_lds_ahead: up to 32 loads in flight) and
//   use_e_lds_ahead8 (src/study/left_cd2_blocks.h: at most 16 in flight), interleaved, 12 rounds in one process; every
//   sample's cost per weight is printed. Verdict cd2_bound: every form at or above its issue bound 3.45 per weight,
//   v_fma_f32 1.00 in the same process, the table at LDS 0; the comparison is INFO. --break claims ahead8 <= 1.00: FAIL.
#include "core_common.h"
#include "cvt_ref.h"
#include "cvt_time_blocks.h"
#include "left_cd2_blocks.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <map>
#include <string>

__host__ __device__ inline uint16_t f16_of_e4m3(uint32_t c) {
    const uint32_t f = cr::decode8(cr::FP8, c);
    const uint32_t s = (f >> 16) & 0x8000u, a = f & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return (uint16_t) (s | 0x7E00u);
    }
    if (a == 0) {
        return (uint16_t) s;
    }
    return (uint16_t) (s | (((a >> 23) - 127 + 15) << 10) | ((a >> 13) & 0x3FFu));
}

// cvt_time.hip's kernel, unchanged apart from the name (the FP16 table sits at LDS address 0)
#define CVT_KERNEL(NAME, P, W, DESC)                                                                              \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void k_##NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out, \
                                                   uint32_t * __restrict__ sink) {                                \
        __shared__ uint16_t tab[256];                                                                             \
        for (int i = threadIdx.x; i < 256; i += 32) {                                                             \
            tab[i] = f16_of_e4m3(i);                                                                              \
        }                                                                                                         \
        __syncthreads();                                                                                          \
        asm volatile(CVT_INIT ::"v"((uint32_t) in[0]), "v"((uint32_t) in[1]), "v"((uint32_t) in[2])              \
                     : CVT_CLOBBERS, "memory");                                                                   \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                              \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                        \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(CVT_BLOCK_##NAME ::: CVT_CLOBBERS, "vcc", "memory");                             \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        uint32_t s;                                                                                               \
        asm volatile("v_xor_b32 %0, v64, v128" : "=v"(s)::CVT_CLOBBERS);                                          \
        sink[threadIdx.x] = s;                                                                                    \
        if (threadIdx.x == 0) {                                                                                   \
            sink[64] = (uint32_t) (size_t) tab;                                                                   \
        }                                                                                                         \
    }
#define LEFT_CVT_TESTS(X)                                                                                         \
    X(fma_f32_c16, 16, 0, "") X(fma_f32_c1, 16, 0, "") X(widen_a_cvt4, 8, 4, "") X(use_a_cvt4_fma, 8, 4, "")    \
        X(use_b_pk2_fma, 8, 4, "") X(use_e_lds_dot2, 8, 4, "") X(cvt_sr_fp8_f32_c16, 16, 0, "")                  \
            X(cvt_sr_fp8_f32_c1, 16, 0, "") X(use_a_cvt4_dual, 8, 4, "") X(use_b_pk2_dual, 8, 4, "")             \
                X(use_e_lds_ahead, 8, 4, "") X(sr_xorshift, 16, 1, "") X(sr_lcg24, 16, 1, "")                     \
                    X(sr_lcg32, 16, 1, "") X(sr_shift, 16, 1, "") X(sr_xorshift_b0, 16, 1, "")                    \
                        X(sr_lcg24_b0, 16, 1, "") X(sr_lcg32_b0, 16, 1, "") X(sr_shift_b0, 16, 1, "")
LEFT_CVT_TESTS(CVT_KERNEL)
CVT_KERNEL(use_e_lds_ahead8, 8, 4, "")

struct Test {
    const char * name;
    int patterns, wpp;
    core::Kern k4, k6, k8;
};
#define CVT_ENTRY(NAME, P, W, DESC) {#NAME, P, W, k_##NAME<4>, k_##NAME<6>, k_##NAME<8>},
static const Test TESTS[] = {LEFT_CVT_TESTS(CVT_ENTRY)};

static int fails = 0;
static void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
static void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

// T4: the three LDS forms interleaved
static int mode_cd2(bool brk) {
    const Test fma = TESTS[0];
    const Test forms[3] = {{"use_e_lds_dot2", 8, 4, k_use_e_lds_dot2<4>, k_use_e_lds_dot2<6>, k_use_e_lds_dot2<8>},
                           {"use_e_lds_ahead", 8, 4, k_use_e_lds_ahead<4>, k_use_e_lds_ahead<6>, k_use_e_lds_ahead<8>},
                           {"use_e_lds_ahead8", 8, 4, k_use_e_lds_ahead8<4>, k_use_e_lds_ahead8<6>,
                            k_use_e_lds_ahead8<8>}};
    const double f = core::pattern_cost(fma.k4, fma.k6, fma.k8, fma.patterns).per;
    std::vector<double> v[3];
    bool at0 = true;
    for (int round = 0; round < 12; ++round) {
        for (int i = 0; i < 3; ++i) {
            const core::Cost c = core::pattern_cost(forms[i].k4, forms[i].k6, forms[i].k8, forms[i].patterns);
            uint32_t lds_addr = 1;
            CHECK(hipMemcpy(&lds_addr, core::buffers().sink + 64, 4, hipMemcpyDeviceToHost));
            at0 &= lds_addr == 0;
            v[i].push_back(c.per / forms[i].wpp);
            std::printf("RESULT cd2 round %2d %-17s per_weight %.3f (pattern %.3f, range %.3f-%.3f, midpoint_dev %.2f)\n",
                        round, forms[i].name, c.per / forms[i].wpp, c.per, c.lo, c.hi, c.mid);
            std::fflush(stdout);
        }
    }
    std::string txt;
    bool bound = true;
    for (int i = 0; i < 3; ++i) {
        std::vector<double> s = v[i];
        std::sort(s.begin(), s.end());
        int slow = 0;
        for (double x : s) slow += x > 5.0;
        char b[256];
        std::snprintf(b, sizeof b, " %s: median %.3f, min %.3f, max %.3f, %d of %zu samples above 5.0;", forms[i].name,
                      s[s.size() / 2], s.front(), s.back(), slow, s.size());
        txt += b;
        bound &= i == 2 && brk ? s.back() <= 1.0 : s.front() >= 3.45;
    }
    check("cd2_bound", bound && std::fabs(f - 1) < 0.05 && at0,
          "v_fma_f32 %.3f (1.00); every LDS form >= 3.45 per weight (issue bound 3.5%s); table at LDS 0: %s", f,
          brk ? "; deliberate break: ahead8 claimed <= 1.00" : "", at0 ? "yes" : "NO");
    std::printf("VERDICT cd2_forms: INFO per weight over 12 interleaved rounds:%s\n", txt.c_str());
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = core::flag(argc, argv, "--break");
    // FP8 word with four ordinary codes (cvt_time.hip's init)
    core::set_init({0x40B03A38u, 0x3F800000u, 0x3C003C00u});
    if (argc > 1 && std::string(argv[1]) == "cd2") {
        return mode_cd2(brk);
    }
    std::map<std::string, double> per, wt;
    int nlin = 0;
    for (const Test & t : TESTS) {
        const core::Cost c = core::pattern_cost(t.k4, t.k6, t.k8, t.patterns);
        uint32_t lds_addr = 1;
        CHECK(hipMemcpy(&lds_addr, core::buffers().sink + 64, 4, hipMemcpyDeviceToHost));
        const bool lin = std::fabs(c.mid) <= std::max(1.5, 0.05 * c.it_mid);
        nlin += (!lin && std::string(t.name).rfind("use_e_lds", 0) != 0) || lds_addr != 0;
        per[t.name] = c.per;
        wt[t.name] = t.wpp ? c.per / t.wpp : c.per;
        std::printf("RESULT %-20s cycles_per_pattern %.3f (%.3f-%.3f) per_weight %.3f midpoint_dev %.2f%s%s\n", t.name,
                    c.per, c.lo, c.hi, wt[t.name], c.mid, lin ? "" : " NOT LINEAR", lds_addr ? " LDS TABLE NOT AT 0" : "");
        std::fflush(stdout);
    }
    auto within = [](double v, double want, double rel) { return std::fabs(v / want - 1) <= rel; };
    check("cvt_calib",
          std::fabs(per["fma_f32_c16"] - 1) < 0.05 && std::fabs(per["fma_f32_c1"] - 5) < 0.1 &&
              std::fabs(wt["widen_a_cvt4"] - 1) < 0.05 && within(wt["use_a_cvt4_fma"], 2.41, 0.03) &&
              std::fabs(per["cvt_sr_fp8_f32_c16"] - 1) < 0.05 &&
              std::fabs(per["cvt_sr_fp8_f32_c1"] - 5) < 0.1 && nlin == 0,
          "FMA %.3f / %.3f (1.00 / 5.00); widen (a) %.3f per weight (1.00); (a) + v_fmac_f32 %.3f (2.41 +-3 %%); "
          "v_cvt_sr_fp8_f32 %.3f / %.3f (1.00 / 5.00); %d tests not linear (LDS table %.3f per weight, not judged)",
          per["fma_f32_c16"], per["fma_f32_c1"], wt["widen_a_cvt4"], wt["use_a_cvt4_fma"],
          per["cvt_sr_fp8_f32_c16"], per["cvt_sr_fp8_f32_c1"], nlin, wt["use_e_lds_dot2"]);
    const double a = wt["use_a_cvt4_dual"], b = wt["use_b_pk2_dual"];
    const bool d1 = brk ? (a <= 1.05 && b <= 1.05) : (a >= 1.45 && b >= 1.45);
    check("cd1_dual", d1,
          "per weight with v_dual_fmac_f32 consumers: (a) %.3f, (b) %.3f; expect %s (plain v_fmac_f32: (a) %.3f, (b) "
          "%.3f; draft's range [1.45, 2.41]: %s)",
          a, b, brk ? "<= 1.05: deliberate break, 1.00 claimed" : ">= 1.45 (issue bound 1.5)", wt["use_a_cvt4_fma"],
          wt["use_b_pk2_fma"], a <= 2.41 && b <= 2.41 ? "inside" : "outside");
    const double e = wt["use_e_lds_ahead"];
    check("cd2_ahead", e >= 3.45,
          "LDS table with loads issued ahead: %.3f per weight (expect >= 3.45, issue bound 3.5); simple form %.3f; the "
          "draft's range [3.45, 4.59]: %s",
          e, wt["use_e_lds_dot2"], e <= 4.59 ? "inside" : "outside (slower than the simple form)");
    const struct {
        const char * n;
        double want;
    } sr[] = {{"sr_xorshift", 7}, {"sr_lcg24", 3}, {"sr_lcg32", 6}, {"sr_shift", 2}};
    bool srok = true;
    int add3 = 0;
    std::string txt;
    for (const auto & x : sr) {
        srok &= per[x.n] >= 0.97 * x.want;
        add3 += within(per[x.n], x.want, 0.03);
        char b2[128];
        std::snprintf(b2, sizeof b2, " %s %.3f (sum %.0f; sources in one bank %.3f)", x.n + 3, per[x.n], x.want,
                      per[std::string(x.n) + "_b0"]);
        txt += b2;
    }
    check("cd4_sr", srok, "cycles per conversion with its generator, measured:%s; each >= 0.97 x the sum (issue bound); "
          "within 3 %% of the sum (draft's prediction, INFO): %d of 4", txt.c_str(), add3);
    return fails ? 1 : 0;
}
