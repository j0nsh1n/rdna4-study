// Leftovers pass, matrix unit, item 7: who runs when a matrix wave and a vector-FMA wave share one SIMD, and whether
// s_setprio or the kernel descriptor's round-robin bit (WG_RR_EN, assembler directive
// .amdhsa_round_robin_scheduling; clang 24 has no source attribute for it) changes that.
//
// Placement as in mat_share.hip: one 1024-thread group built with -mcumode sits on one CU, 16 waves on each SIMD;
// each wave takes a rank on its SIMD with an LDS atomic; ranks 0 and 1 run, the rest end after one barrier. The
// kernels are loaded from a code object (hipModuleLoad), so the same code can be loaded with and without the
// round-robin bit: scripts/study-leftovers-matrix.sh builds left_prio.hsaco and left_prio_rr.hsaco from one
// assembly listing, the second with the bit set in every kernel descriptor.
//
// Usage: left_prio <code object> [--rr] [--break]
//        left_prio <code object> --levels [--rr] [--break]   (leftovers C, MT-7)
//   --levels  every pair of s_setprio levels 0..3 (matrix wave, FMA wave), both arrival orders: a total order predicts
//             that the wave with the higher level ends first in every pair (30 of 30), and at equal levels the age
//             rule of item 9 (normal object: the older wave first). Then three running waves per SIMD (matrix + two
//             FMA, in two arrival orders, without s_setprio and with the matrix wave at level 3): the end order is
//             reported (INFO). --break expects the lower level to end first, so the verdict must FAIL.
// Per configuration: each running wave's cycles per operation over its own timed region (SHADER_CYCLES of its SIMD)
// and the overlap of the two regions (intersection / shorter region: 0 = one after the other, 1 = fully side by
// side). Known answers judged: each role alone (16.0 cycles per FP16 matrix instruction, 1.0 per FMA plus the loop's
// 0.08), and without --rr lane 4's age rule (matrix wave older: it keeps 16 cycles; FMA wave older: the matrix wave
// slows to well above 16). The rest is reported. --break expects the FMA-first pair to keep the matrix wave at its
// alone rate, so the verdict must FAIL.
#include "../harness.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#define MM16                                                                                                       \
    "v_wmma_f32_16x16x16_f16 v[16:23], v[48:51], v[52:55], v[16:23]\n"                                             \
    "v_wmma_f32_16x16x16_f16 v[16:23], v[48:51], v[52:55], v[16:23]\n"                                             \
    "v_wmma_f32_16x16x16_f16 v[16:23], v[48:51], v[52:55], v[16:23]\n"                                             \
    "v_wmma_f32_16x16x16_f16 v[16:23], v[48:51], v[52:55], v[16:23]\n"
#define FMA8                                                                                                       \
    "v_fma_f32 v56, v56, v64, v65\nv_fma_f32 v57, v57, v64, v65\nv_fma_f32 v58, v58, v64, v65\n"                   \
    "v_fma_f32 v59, v59, v64, v65\nv_fma_f32 v60, v60, v64, v65\nv_fma_f32 v61, v61, v64, v65\n"                   \
    "v_fma_f32 v62, v62, v64, v65\nv_fma_f32 v63, v63, v64, v65\n"

// 256 FMAs per loop iteration: the loop's own cost (about 20 cycles per taken branch) is then below 0.1 cycle per FMA
#define FMA64 FMA8 FMA8 FMA8 FMA8 FMA8 FMA8 FMA8 FMA8
#define FMA256 FMA64 FMA64 FMA64 FMA64
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));

// One timed loop: counter, n iterations of BODY (16 matrix instructions or 256 FMAs), counter. One asm statement.
#define TIMED(BODY)                                                                                                \
    asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                            \
                 "s_mov_b32 %[cnt], %[n]\n"                                                                        \
                 "2:\n" BODY                                                                                       \
                 "s_add_co_i32 %[cnt], %[cnt], -1\n"                                                               \
                 "s_cmp_lg_u32 %[cnt], 0\n"                                                                        \
                 "s_cbranch_scc1 2b\n"                                                                             \
                 "s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                            \
                 : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), "+{v[16:23]}"(d), "+{v[56:63]}"(x)            \
                 : [n] "s"(n), "{v[48:51]}"(a), "{v[52:55]}"(b), "{v[64:65]}"(k)                                   \
                 : "scc")

// PM, PF: s_setprio level of the matrix wave and of the FMA wave (-1: no s_setprio). out per wave (8 words): HW_ID1,
// rank, running, role (0 matrix, 1 FMA), t0, t1. mm_first: rank 0 runs the matrix role.
#define PRIO_KERNEL(NAME, PM, PF)                                                                                  \
    extern "C" __global__ __launch_bounds__(1024) void NAME(uint32_t * out, int mm_first, int n_mm, int n_fma, int nact,    \
                                                            int roles) {                                                  \
        __shared__ int slots[2];                                                                                     \
        const int w = threadIdx.x / 32, lane = threadIdx.x % 32;                                                   \
        if (threadIdx.x < 2) {                                                                                     \
            slots[threadIdx.x] = 0;                                                                                  \
        }                                                                                                          \
        __syncthreads();                                                                                           \
        uint32_t hw;                                                                                               \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));                                          \
        const int simd = (hw >> 9) & 1;                                                                            \
        int rank = 0;                                                                                              \
        if (lane == 0) {                                                                                           \
            rank = atomicAdd(&slots[simd], 1);                                                                       \
        }                                                                                                          \
        rank = __builtin_amdgcn_readfirstlane(rank);                                                               \
        __syncthreads();                                                                                           \
        const bool on = rank < nact;                                                                                  \
        /* roles != 0 (leftovers C): bit r gives rank r's role (1 = FMA) */                                        \
        const int role = roles ? (roles >> rank) & 1 : (rank == 0) == (mm_first != 0) ? 0 : 1;                     \
        if (lane == 0) {                                                                                           \
            out[w * 8 + 0] = hw;                                                                                   \
            out[w * 8 + 1] = rank;                                                                                 \
            out[w * 8 + 2] = on;                                                                                   \
            out[w * 8 + 3] = role;                                                                                 \
        }                                                                                                          \
        if (!on) {                                                                                                 \
            return;                                                                                                \
        }                                                                                                          \
        v8f d = {};                                                                                                \
        v8h a, b;                                                                                                  \
        for (int i = 0; i < 8; ++i) {                                                                              \
            a[i] = (_Float16) (0.001f * (lane + i));                                                               \
            b[i] = (_Float16) (0.002f * (lane - i));                                                               \
            d[i] = 0.5f * lane;                                                                                    \
        }                                                                                                          \
        v8f x = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};                                                          \
        v2f k = {0.999f, 0.001f};                                                                                  \
        uint32_t t0, t1, cnt;                                                                                      \
        if (role == 0) {                                                                                           \
            if (PM >= 0) {                                                                                         \
                asm volatile("s_setprio %0" ::"n"(PM < 0 ? 0 : PM));                                               \
            }                                                                                                      \
            const int n = n_mm;                                                                                    \
            TIMED(MM16 MM16 MM16 MM16);                                                                            \
        } else {                                                                                                   \
            if (PF >= 0) {                                                                                         \
                asm volatile("s_setprio %0" ::"n"(PF < 0 ? 0 : PF));                                               \
            }                                                                                                      \
            const int n = n_fma;                                                                                   \
            TIMED(FMA256);                                                                                      \
        }                                                                                                          \
        if (lane == 0) {                                                                                           \
            out[w * 8 + 4] = t0;                                                                                   \
            out[w * 8 + 5] = t1;                                                                                   \
        }                                                                                                          \
        float s = 0;                                                                                               \
        for (int i = 0; i < 8; ++i) {                                                                              \
            s += d[i] + x[i];                                                                                      \
        }                                                                                                          \
        if (s == 1.2345f) {                                                                                        \
            out[1023] = 1;                                                                                         \
        }                                                                                                          \
    }

PRIO_KERNEL(prio_none, -1, -1)
PRIO_KERNEL(prio_mm3, 3, -1)
PRIO_KERNEL(prio_fma3, -1, 3)
PRIO_KERNEL(prio_mm3_fma0, 3, 0)
PRIO_KERNEL(prio_mm0_fma3, 0, 3)
// leftovers C, MT-7: every pair of levels
PRIO_KERNEL(prio_l00, 0, 0)
PRIO_KERNEL(prio_l01, 0, 1)
PRIO_KERNEL(prio_l02, 0, 2)
PRIO_KERNEL(prio_l03, 0, 3)
PRIO_KERNEL(prio_l10, 1, 0)
PRIO_KERNEL(prio_l11, 1, 1)
PRIO_KERNEL(prio_l12, 1, 2)
PRIO_KERNEL(prio_l13, 1, 3)
PRIO_KERNEL(prio_l20, 2, 0)
PRIO_KERNEL(prio_l21, 2, 1)
PRIO_KERNEL(prio_l22, 2, 2)
PRIO_KERNEL(prio_l23, 2, 3)
PRIO_KERNEL(prio_l30, 3, 0)
PRIO_KERNEL(prio_l31, 3, 1)
PRIO_KERNEL(prio_l32, 3, 2)
PRIO_KERNEL(prio_l33, 3, 3)

#ifndef LEFT_PRIO_DEVICE_ONLY
struct Res {
    double mm_cpo = 0, fma_cpo = 0, ovl = 0, mm_first_frac = 0;  // medians; share of pairs where the matrix wave ended first
    int pairs = 0;
};

static Res run_cfg(hipFunction_t f, int mm_first, int n_mm, int n_fma, int launches, int nact = 2,
                   std::vector<std::string> * orders = nullptr, int roles = 0) {
    uint32_t * dout;
    CHECK(hipMalloc(&dout, 1024 * 4));
    std::vector<double> mm, fm, ov;
    int mm_end_first = 0, pairs = 0;
    for (int l = 0; l < launches; ++l) {
        CHECK(hipMemset(dout, 0, 1024 * 4));
        struct {
            uint32_t * out;
            int mm_first, n_mm, n_fma, nact, roles;
        } args{dout, mm_first, n_mm, n_fma, nact, roles};
        size_t sz = sizeof args;
        void * cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CHECK(hipModuleLaunchKernel(f, 1, 1, 1, 1024, 1, 1, 0, 0, nullptr, cfg));
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> o(1024);
        CHECK(hipMemcpy(o.data(), dout, 1024 * 4, hipMemcpyDeviceToHost));
        for (int s = 0; s < 2; ++s) {
            if (orders && nact > 2) {
                // three or more running waves: the roles in the order their regions end, e.g. "M F F"
                std::vector<std::pair<uint32_t, int>> ends;
                for (int w = 0; w < 32; ++w) {
                    if (o[w * 8 + 2] && (int) ((o[w * 8] >> 9) & 1) == s) {
                        ends.push_back({o[w * 8 + 5], (int) o[w * 8 + 3]});
                    }
                }
                std::sort(ends.begin(), ends.end());
                std::string t;
                for (auto & e : ends) {
                    t += e.second == 0 ? "M" : "F";
                }
                orders->push_back(t);
                ++pairs;
                continue;
            }
            int wm = -1, wf = -1;
            for (int w = 0; w < 32; ++w) {
                if (o[w * 8 + 2] && (int) ((o[w * 8] >> 9) & 1) == s) {
                    (o[w * 8 + 3] == 0 ? wm : wf) = w;
                }
            }
            if (nact == 1 && (wm >= 0 || wf >= 0)) {
                const int w1 = wm >= 0 ? wm : wf;
                const double c = ((double) o[w1 * 8 + 5] - o[w1 * 8 + 4]) / ((wm >= 0 ? 16.0 : 256.0) * (wm >= 0 ? n_mm : n_fma));
                (wm >= 0 ? mm : fm).push_back(c);
                ++pairs;
                continue;
            }
            if (wm < 0 || wf < 0) {
                continue;
            }
            const double m0 = o[wm * 8 + 4], m1 = o[wm * 8 + 5], f0 = o[wf * 8 + 4], f1 = o[wf * 8 + 5];
            mm.push_back((m1 - m0) / (16.0 * n_mm));
            fm.push_back((f1 - f0) / (256.0 * n_fma));
            const double inter = std::max(0.0, std::min(m1, f1) - std::max(m0, f0));
            ov.push_back(inter / std::min(m1 - m0, f1 - f0));
            mm_end_first += m1 < f1;
            ++pairs;
        }
    }
    CHECK(hipFree(dout));
    auto med = [](std::vector<double> v) {
        if (v.empty()) {
            return 0.0;
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    Res r;
    r.mm_cpo = med(mm);
    r.fma_cpo = med(fm);
    r.ovl = med(ov);
    r.pairs = pairs;
    r.mm_first_frac = pairs ? (double) mm_end_first / pairs : 0;
    return r;
}

// MT-7: levels 0..3 for each wave; three waves.
static int mode_levels(hipModule_t mod, const char * obj, bool rr, bool brk) {
    const int n_mm = 64, n_fma = 64, launches = 15;
    bool ok = true;
    int agree = 0, total = 0;
    {
        hipFunction_t f;
        CHECK(hipModuleGetFunction(&f, mod, "prio_none"));
        const Res rm = run_cfg(f, 1, n_mm, n_fma, launches, 1), rf = run_cfg(f, 0, n_mm, n_fma, launches, 1);
        std::printf("RESULT prio %s alone: matrix cycles/instr %.3f (%d waves), FMA cycles/FMA %.3f (%d waves)\n", obj, rm.mm_cpo,
                    rm.pairs, rf.fma_cpo, rf.pairs);
        ok = ok && std::abs(rm.mm_cpo - 16.0) < 0.6 && std::abs(rf.fma_cpo - 1.08) < 0.1;
    }
    for (int pm = 0; pm < 4; ++pm) {
        for (int pf = 0; pf < 4; ++pf) {
            char name[16];
            std::snprintf(name, sizeof name, "prio_l%d%d", pm, pf);
            hipFunction_t f;
            CHECK(hipModuleGetFunction(&f, mod, name));
            for (int mm_first = 1; mm_first >= 0; --mm_first) {
                const Res r = run_cfg(f, mm_first, n_mm, n_fma, launches);
                // expected share of pairs where the matrix wave ends first: 1 (pm > pf), 0 (pm < pf); equal levels:
                // the older wave (normal object), INFO for the round-robin object. --break swaps the expectation.
                double want = pm > pf ? 1.0 : pm < pf ? 0.0 : (rr ? -1.0 : (mm_first ? 1.0 : 0.0));
                if (brk && want >= 0) {
                    want = 1.0 - want;
                }
                const bool good = want < 0 || (std::abs(r.mm_first_frac - want) < 1e-9 && r.pairs >= launches);
                if (want >= 0) {
                    ++total;
                    agree += good;
                }
                ok = ok && good;
                std::printf("RESULT prio %s levels matrix %d FMA %d rank 0 = %-6s pairs %d | matrix cycles/instr %.3f | FMA "
                            "cycles/FMA %.3f | overlap %.2f | matrix wave ended first in %.0f %%%s\n",
                            obj, pm, pf, mm_first ? "matrix" : "FMA", r.pairs, r.mm_cpo, r.fma_cpo, r.ovl, 100 * r.mm_first_frac,
                            want < 0 ? " (no expectation)" : good ? "" : " NOT as expected");
            }
        }
    }
    // three running waves per SIMD (INFO): roles by rank as given
    struct T3 {
        const char * kernel;
        int roles;
        const char * what;
    };
    const T3 t3[] = {{"prio_none", 0b110, "ranks M F F, no s_setprio"},
                     {"prio_none", 0b101, "ranks F M F, no s_setprio"},
                     {"prio_mm3", 0b101, "ranks F M F, matrix wave at 3"},
                     {"prio_fma3", 0b110, "ranks M F F, FMA waves at 3"}};
    for (const T3 & t : t3) {
        hipFunction_t f;
        CHECK(hipModuleGetFunction(&f, mod, t.kernel));
        std::vector<std::string> orders;
        const Res r = run_cfg(f, 1, n_mm, n_fma, launches, 3, &orders, t.roles);
        std::map<std::string, int> h;
        for (auto & o : orders) {
            ++h[o];
        }
        std::string hs;
        for (auto & kv : h) {
            hs += " " + kv.first + ":" + std::to_string(kv.second);
        }
        std::printf("INFO prio %s three waves %-30s SIMDs %d end order (first to last) counts:%s\n", obj, t.what, r.pairs, hs.c_str());
    }
    std::printf("VERDICT left_prio_levels: %s calibration alone and %d of %d level pairs and arrival orders as a total order "
                "with the age rule at equal levels predicts (%s)%s\n",
                ok ? "PASS" : "FAIL", agree, total, obj, brk ? " (broken on purpose)" : "");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: left_prio <code object> [--break]\n");
        return 2;
    }
    bool brk = false, rr = false;
    for (int i = 2; i < argc; ++i) {
        brk = brk || std::string(argv[i]) == "--break";
        rr = rr || std::string(argv[i]) == "--rr";
    }
    require_arch();
    hipModule_t mod;
    CHECK(hipModuleLoad(&mod, argv[1]));
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--levels") {
            return mode_levels(mod, argv[1], rr, brk);
        }
    }
    const char * names[] = {"prio_none", "prio_mm3", "prio_fma3", "prio_mm3_fma0", "prio_mm0_fma3"};
    const char * what[] = {"no s_setprio", "matrix wave s_setprio 3", "FMA wave s_setprio 3", "matrix 3, FMA 0",
                           "matrix 0, FMA 3"};
    const int n_mm = 64, n_fma = 64, launches = 15;  // both regions about 16,000 cycles when alone
    bool ok = true;
    std::string text;
    {
        // each role alone on its SIMD: the known answers 16.0 cycles per FP16 matrix instruction and 1.0 per FMA
        hipFunction_t f;
        CHECK(hipModuleGetFunction(&f, mod, "prio_none"));
        const Res rm = run_cfg(f, 1, n_mm, n_fma, launches, 1), rf = run_cfg(f, 0, n_mm, n_fma, launches, 1);
        std::printf("RESULT prio %s alone: matrix cycles/instr %.3f (%d waves), FMA cycles/FMA %.3f (%d waves)\n", argv[1], rm.mm_cpo,
                    rm.pairs, rf.fma_cpo, rf.pairs);
        ok = ok && std::abs(rm.mm_cpo - 16.0) < 0.6 && std::abs(rf.fma_cpo - 1.08) < 0.1;
    }
    for (int i = 0; i < 5; ++i) {
        hipFunction_t f;
        CHECK(hipModuleGetFunction(&f, mod, names[i]));
        for (int mm_first = 1; mm_first >= 0; --mm_first) {
            const Res r = run_cfg(f, mm_first, n_mm, n_fma, launches);
            std::printf("RESULT prio %s %-24s rank 0 = %-6s pairs %d | matrix cycles/instr %.3f | FMA cycles/FMA %.3f | "
                        "overlap %.2f | matrix wave ended first in %.0f %%\n",
                        argv[1], what[i], mm_first ? "matrix" : "FMA", r.pairs, r.mm_cpo, r.fma_cpo, r.ovl, 100 * r.mm_first_frac);
            if (i == 0 && !rr) {
                // the known answer (matrix-timing.md item 9): the older matrix wave keeps 16 cycles; beside an older
                // FMA wave it slows down (40 there, with other run lengths). --break expects 16 in the second case.
                const bool good = mm_first ? std::abs(r.mm_cpo - 16.0) < 0.6
                                           : (brk ? std::abs(r.mm_cpo - 16.0) < 0.6 : r.mm_cpo > 24.0);
                ok = ok && good && r.pairs >= launches;
                char buf[160];
                std::snprintf(buf, sizeof buf, "%s%s first: matrix %.2f, FMA %.3f", text.empty() ? "" : "; ", mm_first ? "matrix" : "FMA",
                              r.mm_cpo, r.fma_cpo);
                text += buf;
            }
            // Pinned for both code objects (found in the leftovers pass, pinned after its review): a wave with
            // s_setprio 3 goes first whatever the arrival order, so the matrix wave keeps about 16 cycles when it has
            // the priority and is held back (above 24) when the FMA wave has it.
            if (i >= 1) {
                const bool mm_prio = i == 1 || i == 3;
                const bool good = mm_prio ? std::abs(r.mm_cpo - 16.0) < 0.8 : r.mm_cpo > 24.0;
                ok = ok && good && r.pairs >= launches;
                if (!good) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "%s%s, rank 0 %s: matrix %.2f NOT as pinned", text.empty() ? "" : "; ", what[i],
                                  mm_first ? "matrix" : "FMA", r.mm_cpo);
                    text += buf;
                }
            }
            // Round-robin object without s_setprio: no age order, the matrix wave costs about 19.3 cycles whichever
            // wave arrived first. --break (with --rr) expects the age order of the normal object instead.
            if (i == 0 && rr) {
                const bool good = brk ? (mm_first ? std::abs(r.mm_cpo - 16.0) < 0.6 : r.mm_cpo > 24.0) : std::abs(r.mm_cpo - 19.3) < 1.5;
                ok = ok && good && r.pairs >= launches;
                char buf[160];
                std::snprintf(buf, sizeof buf, "%sround robin, %s first: matrix %.2f, FMA %.3f", text.empty() ? "" : "; ",
                              mm_first ? "matrix" : "FMA", r.mm_cpo, r.fma_cpo);
                text += buf;
            }
        }
    }
    if (ok && !brk) {
        text += "; s_setprio 3 decides in all four priority settings";
    }
    std::printf("VERDICT left_prio: %s %s (%s)%s\n", ok ? "PASS" : "FAIL", text.c_str(), argv[1], brk ? " (broken on purpose)" : "");
    return ok ? 0 : 1;
}
#endif
