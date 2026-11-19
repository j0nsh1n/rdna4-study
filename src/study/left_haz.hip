// Leftovers C, lane A: CH-2a (v_cmpx directly before a lane permute) and CH-1 (the scalar-register hazard: a VALU
// read of an SGPR after an SALU rewrite, without the wait).
//
// Usage: left_haz <cmpx|sgpr> [--break]
//   cmpx  inline assembly: EXEC is set by v_cmpx_gt_u32 from per-lane data (5 masks as compiler-and-hazards item 4),
//         then v_permlane16_b32 or v_permlanex16_b32 with fetch-inactive (op_sel:[1,0]) reads a register that every
//         lane filled before; three variants: nothing between, LLVM's fix (a v_mov_b32 between; READ, research-10
//         section 2) and an s_nop between; each with fetch-inactive on and off. CPU model: a lane that is on gets the
//         selected lane's value; when that lane is off, with fetch-inactive it still gets it, without it keeps its own
//         old value (compiler-and-hazards item 4); a lane that is off keeps its sentinel. A stale EXEC (all lanes on)
//         would show as written off lanes, or as off partners read without fetch-inactive. Verdict cmpx_control: the v_mov variant
//         equals the model in every lane of every mask; the no-gap and s_nop results are INFO. --break judges the
//         v_mov variant against a model that ignores fetch-inactive (an off source lane gives the sentinel): FAIL.
//   sgpr  CH-1: s20 is rewritten by SALU (an LCG) and read by a VALU instruction D2 instructions later, the previous
//         read D1 instructions before the rewrite (the body is unrolled twice, so D1 can be 0), for four kinds of
//         reading instruction (v_fma_f32 VOP3, v_add_f32 VOP2, v_fma_f64 on s[20:21], v_exp_f32; the next half's
//         LCG step puts 2 SALU instructions before the first rewrite of s21 and 4 before s20); each with and
//         without `s_wait_alu depctr_sa_sdst(0)` before the read. Verdict sgpr_control: every wait variant of v_fma_f32
//         equals the CPU model; every no-wait variant is compared bit for bit with its wait variant (a difference is a
//         wrong result: INFO with the sequence). --break compares against a model with the LCG one step behind: FAIL.
#include "../harness.h"

#include <cmath>
#include <cstdarg>
#include <cstring>
#include <map>
#include <string>

namespace {

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1200];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

// ---- CH-2a -------------------------------------------------------------------------------------------------------
constexpr uint32_t SENT = 0xDEAD0000u;
#define CMPX_K(NAME, GAP, PERM, FI)                                                                                 \
    __global__ __launch_bounds__(32) void NAME(const uint32_t * key, const uint32_t * val, uint32_t thr,            \
                                               uint32_t * out) {                                                    \
        const uint32_t l = threadIdx.x;                                                                             \
        uint32_t k = key[l], v = val[l], r = SENT + l, sv;                                                          \
        asm volatile("s_mov_b32 %[sv], exec_lo\n"                                                                   \
                     "v_cmpx_gt_u32 %[k], %[thr]\n" GAP PERM " %[r], %[v], %[slo], %[shi]" FI "\n"                   \
                     "s_mov_b32 exec_lo, %[sv]\n"                                                                   \
                     : [r] "+v"(r), [sv] "=&s"(sv)                                                                  \
                     : [k] "v"(k), [thr] "v"(thr), [v] "v"(v), [slo] "s"(0x89ABCDEFu), [shi] "s"(0x01234567u)       \
                     : "v0", "memory");                                                                            \
        out[l] = r;                                                                                                 \
    }
// fetch-inactive on (op_sel:[1,0]) and off
CMPX_K(c16_none, "", "v_permlane16_b32", " op_sel:[1,0]")
CMPX_K(c16_mov, "v_mov_b32 v0, v0\n", "v_permlane16_b32", " op_sel:[1,0]")
CMPX_K(c16_nop, "s_nop 0\n", "v_permlane16_b32", " op_sel:[1,0]")
CMPX_K(cx16_none, "", "v_permlanex16_b32", " op_sel:[1,0]")
CMPX_K(cx16_mov, "v_mov_b32 v0, v0\n", "v_permlanex16_b32", " op_sel:[1,0]")
CMPX_K(cx16_nop, "s_nop 0\n", "v_permlanex16_b32", " op_sel:[1,0]")
CMPX_K(c16f0_none, "", "v_permlane16_b32", "")
CMPX_K(c16f0_mov, "v_mov_b32 v0, v0\n", "v_permlane16_b32", "")
CMPX_K(c16f0_nop, "s_nop 0\n", "v_permlane16_b32", "")
CMPX_K(cx16f0_none, "", "v_permlanex16_b32", "")
CMPX_K(cx16f0_mov, "v_mov_b32 v0, v0\n", "v_permlanex16_b32", "")
CMPX_K(cx16f0_nop, "s_nop 0\n", "v_permlanex16_b32", "")

using CK = void (*)(const uint32_t *, const uint32_t *, uint32_t, uint32_t *);

int mode_cmpx(bool brk) {
    uint32_t *dk, *dv, *dout;
    CHECK(hipMalloc(&dk, 128));
    CHECK(hipMalloc(&dv, 128));
    CHECK(hipMalloc(&dout, 128));
    // masks (lane on when key > 100): odd lanes, every 4th, lanes 0-15, lanes 16-31, data-dependent
    const char * mname[5] = {"odd lanes", "every 4th", "lanes 0-15", "lanes 16-31", "data-dependent"};
    struct V {
        const char * name;
        CK k;
        int x, gap, fi;
    } vs[] = {{"permlane16 fi, nothing between", c16_none, 0, 0, 1},  {"permlane16 fi, v_mov between", c16_mov, 0, 1, 1},
              {"permlane16 fi, s_nop between", c16_nop, 0, 2, 1},    {"permlanex16 fi, nothing between", cx16_none, 1, 0, 1},
              {"permlanex16 fi, v_mov between", cx16_mov, 1, 1, 1},  {"permlanex16 fi, s_nop between", cx16_nop, 1, 2, 1},
              {"permlane16, nothing between", c16f0_none, 0, 0, 0},  {"permlane16, v_mov between", c16f0_mov, 0, 1, 0},
              {"permlane16, s_nop between", c16f0_nop, 0, 2, 0},     {"permlanex16, nothing between", cx16f0_none, 1, 0, 0},
              {"permlanex16, v_mov between", cx16f0_mov, 1, 1, 0},   {"permlanex16, s_nop between", cx16f0_nop, 1, 2, 0}};
    int ctl_bad = 0;
    std::map<std::string, int> other_bad;
    for (int m = 0; m < 5; ++m) {
        uint32_t key[32], val[32];
        for (int l = 0; l < 32; ++l) {
            bool on = m == 0 ? (l & 1) : m == 1 ? (l % 4 == 0) : m == 2 ? (l < 16) : m == 3 ? (l >= 16) : ((l * 7 + 3) % 5 < 2);
            key[l] = on ? 200 + l : 50;
            val[l] = 0x1000 + l * 3;
        }
        CHECK(hipMemcpy(dk, key, 128, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dv, val, 128, hipMemcpyHostToDevice));
        for (const V & v : vs) {
            hipLaunchKernelGGL(v.k, dim3(1), dim3(32), 0, 0, dk, dv, 100u, dout);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            uint32_t r[32];
            CHECK(hipMemcpy(r, dout, 128, hipMemcpyDeviceToHost));
            int bad = 0, written_off = 0, stale = 0;
            for (int l = 0; l < 32; ++l) {
                const bool on = key[l] > 100;
                // row reversal within 16 lanes; permlanex16 reads the other half of the 32
                const int src = (v.x ? (l ^ 16) & ~15 : l & ~15) | (15 - (l & 15));
                uint32_t want;
                // fetch-inactive off: a lane whose partner is off keeps its own old value (compiler-and-hazards 4)
                const bool partner_off = key[src] <= 100;
                if (!on) {
                    want = SENT + l;
                } else if (partner_off && (!v.fi || (brk && v.gap == 1))) {
                    want = SENT + l;  // (with fetch-inactive on, the deliberate break uses this rule)
                } else {
                    want = val[src];
                }
                bad += r[l] != want;
                written_off += !on && r[l] != SENT + (uint32_t) l;
                stale += on && r[l] != want;
            }
            std::printf("RESULT cmpx %-30s mask %-15s lanes on %2d: %2d of 32 differ from the model (%d off lanes "
                        "written, %d on lanes wrong)\n",
                        v.name, mname[m], (int) std::count_if(key, key + 32, [](uint32_t x) { return x > 100; }), bad,
                        written_off, stale);
            if (v.gap == 1) {
                ctl_bad += bad;
            } else {
                other_bad[v.name] += bad;
            }
        }
    }
    check("cmpx_control", ctl_bad == 0,
          "v_cmpx then v_mov then the permute (LLVM's fix), 2 permutes x fetch-inactive on and off x 5 masks: %d lanes "
          "differ from the model%s",
          ctl_bad, brk ? " (deliberate break: model without fetch-inactive)" : "");
    std::string t;
    for (auto & kv : other_bad) t += " " + kv.first + ": " + std::to_string(kv.second) + ";";
    std::printf("VERDICT cmpx_nogap: INFO lanes that differ from the model over the 5 masks:%s\n", t.c_str());
    return fails ? 1 : 0;
}

// ---- CH-1 --------------------------------------------------------------------------------------------------------
// One half of the unrolled body: the LCG step on %[s], s20 (and s21) written from it by SALU, D2 nops, the optional
// wait, the VALU read, D1 nops. KIND 0: v_fma_f32 x, x, 0.5, s20; 1: v_add_f32 x, s20, x (VOP2); 2: v_fma_f64 on
// s[20:21]; 3: v_exp_f32 t, s20 then v_add_f32 x, x, t.
#define S_WRITE "s_mul_i32 %[s], %[s], 0x19660d\n s_add_co_i32 %[s], %[s], 0x3c6ef35f\n" \
                "s_and_b32 s21, %[s], 0xfffff\n s_or_b32 s21, s21, 0x3ff00000\n"           \
                "s_and_b32 s20, %[s], 0x7fffff\n s_or_b32 s20, s20, 0x3f800000\n"
#define S_READ0 "v_fma_f32 %[x], %[x], 0.5, s20\n"
#define S_READ1 "v_add_f32 %[x], s20, %[x]\n"
#define S_READ2 "v_fma_f64 %[d], %[d], 0.5, s[20:21]\n"
#define S_READ3 "v_exp_f32 %[t], s20\n v_add_f32 %[x], %[x], %[t]\n"
#define S_WAIT "s_wait_alu depctr_sa_sdst(0)\n"

template <int KIND, int D1, int D2, bool WAIT>
__global__ __launch_bounds__(32) void k_sgpr(uint32_t seed, int n, float * outf, double * outd) {
    float x = 1.0f + threadIdx.x * 0.001f, t = 0;
    double d = 1.0 + threadIdx.x * 0.001;
    uint32_t s = seed;
    int k = n;
#define S_LOOP(READ, WT)                                                                                            \
    asm volatile("1:\n" S_HALF(READ) S_HALF(READ) " s_add_co_i32 %[k], %[k], -1\n s_cmp_lg_u32 %[k], 0\n"           \
                 " s_cbranch_scc1 1b\n"                                                                             \
                 : [x] "+v"(x), [d] "+v"(d), [t] "+v"(t), [s] "+s"(s), [k] "+s"(k)                                  \
                 : [d1] "i"(D1), [d2] "i"(D2)                                                                       \
                 : "s20", "s21", "scc")
    // the optional wait is spliced in through two definitions of the text (an operand cannot carry text)
    if constexpr (WAIT) {
#define S_HALF(READ) S_WRITE ".rept %c[d2]\n s_nop 0\n .endr\n" S_WAIT READ ".rept %c[d1]\n s_nop 0\n .endr\n"
        if constexpr (KIND == 0) S_LOOP(S_READ0, 1);
        else if constexpr (KIND == 1) S_LOOP(S_READ1, 1);
        else if constexpr (KIND == 2) S_LOOP(S_READ2, 1);
        else S_LOOP(S_READ3, 1);
    } else {
#undef S_HALF
#define S_HALF(READ) S_WRITE ".rept %c[d2]\n s_nop 0\n .endr\n" READ ".rept %c[d1]\n s_nop 0\n .endr\n"
        if constexpr (KIND == 0) S_LOOP(S_READ0, 0);
        else if constexpr (KIND == 1) S_LOOP(S_READ1, 0);
        else if constexpr (KIND == 2) S_LOOP(S_READ2, 0);
        else S_LOOP(S_READ3, 0);
    }
    outf[threadIdx.x] = x;
    outd[threadIdx.x] = d;
}

using SK = void (*)(uint32_t, int, float *, double *);
template <int KIND, int D1, int D2>
void pair(SK & w, SK & nw) {
    w = k_sgpr<KIND, D1, D2, true>;
    nw = k_sgpr<KIND, D1, D2, false>;
}

int mode_sgpr(bool brk) {
    float * df;
    double * dd;
    CHECK(hipMalloc(&df, 128));
    CHECK(hipMalloc(&dd, 256));
    const int N = 64;  // loop iterations (two reads each)
    const uint32_t seed = 12345u;
    // CPU model of the FMA kind: x = fma(x, 0.5, f(s)) per half, with f the s20 value built from the LCG state
    auto model = [&](int lane, int lag) {
        float x = 1.0f + lane * 0.001f;
        uint32_t s = seed, prev = seed;
        for (int i = 0; i < 2 * N; ++i) {
            prev = s;
            s = s * 0x19660du + 0x3c6ef35fu;
            const uint32_t use = lag ? prev : s;
            uint32_t b = (use & 0x7fffffu) | 0x3f800000u;
            float v;
            std::memcpy(&v, &b, 4);
            x = std::fmaf(x, 0.5f, v);
        }
        return x;
    };
    const char * kn[4] = {"v_fma_f32 (VOP3)", "v_add_f32 (VOP2)", "v_fma_f64 s[20:21]", "v_exp_f32"};
    int ctl_bad = 0, wrong_cases = 0, cases = 0;
    std::string wrong;
#define RUN_KIND(KIND)                                                                                              \
    {                                                                                                               \
        SK ws[3][5], nws[3][5];                                                                                     \
        pair<KIND, 0, 0>(ws[0][0], nws[0][0]); pair<KIND, 0, 1>(ws[0][1], nws[0][1]);                              \
        pair<KIND, 0, 2>(ws[0][2], nws[0][2]); pair<KIND, 0, 4>(ws[0][3], nws[0][3]);                              \
        pair<KIND, 0, 8>(ws[0][4], nws[0][4]); pair<KIND, 2, 0>(ws[1][0], nws[1][0]);                              \
        pair<KIND, 2, 1>(ws[1][1], nws[1][1]); pair<KIND, 2, 2>(ws[1][2], nws[1][2]);                              \
        pair<KIND, 2, 4>(ws[1][3], nws[1][3]); pair<KIND, 2, 8>(ws[1][4], nws[1][4]);                              \
        pair<KIND, 8, 0>(ws[2][0], nws[2][0]); pair<KIND, 8, 1>(ws[2][1], nws[2][1]);                              \
        pair<KIND, 8, 2>(ws[2][2], nws[2][2]); pair<KIND, 8, 4>(ws[2][3], nws[2][3]);                              \
        pair<KIND, 8, 8>(ws[2][4], nws[2][4]);                                                                      \
        const int d1s[3] = {0, 2, 8}, d2s[5] = {0, 1, 2, 4, 8};                                                     \
        for (int a = 0; a < 3; ++a) {                                                                               \
            for (int b = 0; b < 5; ++b) {                                                                           \
                float fw[32], fn[32];                                                                               \
                double dw[32], dn[32];                                                                              \
                hipLaunchKernelGGL(ws[a][b], dim3(1), dim3(32), 0, 0, seed, N, df, dd);                             \
                CHECK(hipDeviceSynchronize());                                                                      \
                CHECK(hipMemcpy(fw, df, 128, hipMemcpyDeviceToHost));                                               \
                CHECK(hipMemcpy(dw, dd, 256, hipMemcpyDeviceToHost));                                               \
                hipLaunchKernelGGL(nws[a][b], dim3(1), dim3(32), 0, 0, seed, N, df, dd);                            \
                CHECK(hipDeviceSynchronize());                                                                      \
                CHECK(hipMemcpy(fn, df, 128, hipMemcpyDeviceToHost));                                               \
                CHECK(hipMemcpy(dn, dd, 256, hipMemcpyDeviceToHost));                                               \
                int diff = 0, cb = 0;                                                                               \
                for (int l = 0; l < 32; ++l) {                                                                      \
                    diff += std::memcmp(&fw[l], &fn[l], 4) != 0 || std::memcmp(&dw[l], &dn[l], 8) != 0;             \
                    if (KIND == 0) {                                                                                \
                        const float m = model(l, brk ? 1 : 0);                                                      \
                        cb += std::memcmp(&fw[l], &m, 4) != 0;                                                      \
                    }                                                                                               \
                }                                                                                                   \
                ctl_bad += cb;                                                                                      \
                ++cases;                                                                                            \
                if (diff) {                                                                                         \
                    ++wrong_cases;                                                                                  \
                    wrong += std::string(" ") + kn[KIND] + " d1=" + std::to_string(d1s[a]) + " d2=" +              \
                             std::to_string(d2s[b]) + ";";                                                          \
                }                                                                                                   \
                std::printf("RESULT sgpr %-18s d1 %d d2 %d: no-wait differs from wait in %2d of 32 lanes%s\n",      \
                            kn[KIND], d1s[a], d2s[b], diff, KIND == 0 ? (cb ? "; WAIT VARIANT WRONG" : "; wait = model") : ""); \
            }                                                                                                       \
        }                                                                                                           \
    }
    RUN_KIND(0) RUN_KIND(1) RUN_KIND(2) RUN_KIND(3)
    check("sgpr_control", ctl_bad == 0,
          "the waited v_fma_f32 sequences equal the CPU model in all lanes of all 15 distance pairs%s: %d lanes differ",
          brk ? " (deliberate break: model one LCG step behind)" : "", ctl_bad);
    std::printf("VERDICT sgpr_nowait: INFO %d of %d sequences without the wait give a different result:%s\n", wrong_cases,
                cases, wrong.empty() ? " none" : wrong.c_str());
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    bool brk = false;
    for (int i = 1; i < argc; ++i) brk |= std::strcmp(argv[i], "--break") == 0;
    if (mode == "cmpx") return mode_cmpx(brk);
    if (mode == "sgpr") return mode_sgpr(brk);
    std::fprintf(stderr, "usage: left_haz <cmpx|sgpr> [--break]\n");
    return 2;
}
