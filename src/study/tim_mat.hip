// Timing leftovers, item 5: why more than one wave per SIMD loses 1 to 2 % of the FP16 matrix rate (matrix-timing.md
// item 11) but nothing for FP8. One work-group on one CU (built with -mcumode), W = 1, 2, 4, 8 waves on each SIMD.
//
// Usage: tim_mat run [--break]
//   Every wave runs its own dependent chain of 256 matrix instructions (FP16 16x16x16, 16 cycles; FP8, 8 cycles) in
//   16 chunks of 16, and reads the cycle counter into an SGPR after every chunk (scalar reads run beside the matrix
//   unit, matrix-timing item 8, so the stamps do not break the chain). From the stamps of all waves on one SIMD:
//   the SIMD's cycles per matrix instruction (first start to last end), how often consecutive chunks on the SIMD
//   belong to different waves (switches), which chunks were stretched by another wave's work, and the cost per
//   switch. The same with the first wave raised to s_setprio 3.
//   --break claims a FP16 chain costs 32 cycles: the calibration verdict must FAIL.
// Usage: tim_mat handover [--break]   (leftovers C, MT-5b)
//   The same release of W = 1, 2, 4, 8 waves per SIMD with no stamps inside the chain, for FP16, FP8, BF16, FP16 into
//   FP16, BF8, the 8-bit integer form and sparse FP16: the cost of one hand-over = (SIMD span - W x 256 x the one-wave
//   cycles per instruction) / (W - 1). Known (item 9a): FP16 105 to 109, FP8 about 13. --break expects 13 for FP16.
#include "tim_common.h"

#include <map>

namespace {

constexpr int REPS = 4;
constexpr int CHUNKS = 16, PER_CHUNK = 16;
constexpr int MAXW = 16;

#define MM16 "v_wmma_f32_16x16x16_f16 v[64:71], v[72:75], v[76:79], v[64:71]\n"
#define MM8 "v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[72:73], v[74:75], v[64:71]\n"
// four independent accumulators in turn, as in mat_card's whole-card loop (matrix-timing item 11)
#define MM16X4 "v_wmma_f32_16x16x16_f16 v[64:71], v[72:75], v[76:79], v[64:71]\n"                               \
               "v_wmma_f32_16x16x16_f16 v[80:87], v[72:75], v[76:79], v[80:87]\n"                               \
               "v_wmma_f32_16x16x16_f16 v[88:95], v[72:75], v[76:79], v[88:95]\n"                               \
               "v_wmma_f32_16x16x16_f16 v[96:103], v[72:75], v[76:79], v[96:103]\n"
#define MM8X4 "v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[72:73], v[74:75], v[64:71]\n"                              \
              "v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[72:73], v[74:75], v[80:87]\n"                              \
              "v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[72:73], v[74:75], v[88:95]\n"                              \
              "v_wmma_f32_16x16x16_fp8_fp8 v[96:103], v[72:73], v[74:75], v[96:103]\n"
// GAP: what follows each chunk of 16 matrix instructions. 0: a scalar move (runs beside the matrix unit, no gap);
// 1: s_nop 7 (8 cycles in which the wave issues nothing); 2: a taken branch to the next instruction (16 cycles,
// execution-core item 5); 3: a cycle-counter read into an SGPR (the stamps that show the interleaving).
#define TIM_GAP0 "s_mov_b32 s\\i, 0\n"
#define TIM_GAP1 "s_mov_b32 s\\i, 0\n s_nop 7\n"
#define TIM_GAP2 "s_mov_b32 s\\i, 0\n s_cmp_eq_u32 0, 0\n s_cbranch_scc1 1f\n 1:\n"
#define TIM_GAP3 "s_getreg_b32 s\\i, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
#define TIM_MAT_ASM(MM, REPT, GAPTEXT)                                                                                     \
    asm volatile("s_getreg_b32 s39, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                               \
                 ".irp i,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55\n"                                         \
                 ".rept " REPT "\n" MM ".endr\n" GAPTEXT ".endr\n"                                                       \
                 "s_getreg_b32 s56, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                               \
                 : "={s39}"(st[0]), "={s40}"(st[1]), "={s41}"(st[2]), "={s42}"(st[3]), "={s43}"(st[4]),             \
                   "={s44}"(st[5]), "={s45}"(st[6]), "={s46}"(st[7]), "={s47}"(st[8]), "={s48}"(st[9]),              \
                   "={s49}"(st[10]), "={s50}"(st[11]), "={s51}"(st[12]), "={s52}"(st[13]), "={s53}"(st[14]),         \
                   "={s54}"(st[15]), "={s55}"(st[16]), "={s56}"(st[17])                                             \
                 :                                                                                                   \
                 : "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v77", \
                   "v78", "v79", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v91", \
                   "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "scc",   \
                   "memory")

// MT-5b: more forms, chained through C, no stamps (gap 0).
#define MMBF "v_wmma_f32_16x16x16_bf16 v[64:71], v[72:75], v[76:79], v[64:71]\n"
#define MMSF "v_swmmac_f32_16x16x32_f16 v[64:71], v[72:75], v[76:83], v84\n"
#define MMI8 "v_wmma_i32_16x16x16_iu8 v[64:71], v[72:73], v[74:75], v[64:71]\n"
#define MMHH "v_wmma_f16_16x16x16_f16 v[64:67], v[72:75], v[76:79], v[64:67]\n"
#define MMB8 "v_wmma_f32_16x16x16_bf8_bf8 v[64:71], v[72:73], v[74:75], v[64:71]\n"

// out per wave: [rep][start, 16 chunk stamps (gap 3; else 0), end], then HW_ID1.
template <int FP8, int GAP, int X4>
__global__ __launch_bounds__(1024) void k_mat(int prio, uint32_t * out) {
    const int w = threadIdx.x / 32;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    if (prio && w >= (int) (blockDim.x / 32) - 2) {
        // the youngest wave of each SIMD (waves alternate between the two SIMDs; checked from HW_ID1 on the host)
        asm volatile("s_setprio 3");
    }
    for (int rep = 0; rep < REPS; ++rep) {
        __syncthreads();
        uint32_t st[18];
        constexpr const char * dummy = "";
        (void) dummy;
        if constexpr (X4 && FP8 && GAP == 0) TIM_MAT_ASM(MM8X4, "4", TIM_GAP0);
        else if constexpr (X4 && FP8) TIM_MAT_ASM(MM8X4, "4", TIM_GAP3);
        else if constexpr (X4 && GAP == 0) TIM_MAT_ASM(MM16X4, "4", TIM_GAP0);
        else if constexpr (X4) TIM_MAT_ASM(MM16X4, "4", TIM_GAP3);
        else if constexpr (FP8 && GAP == 0) TIM_MAT_ASM(MM8, "16", TIM_GAP0);
        else if constexpr (FP8 && GAP == 1) TIM_MAT_ASM(MM8, "16", TIM_GAP1);
        else if constexpr (FP8 && GAP == 2) TIM_MAT_ASM(MM8, "16", TIM_GAP2);
        else if constexpr (FP8) TIM_MAT_ASM(MM8, "16", TIM_GAP3);
        else if constexpr (GAP == 0) TIM_MAT_ASM(MM16, "16", TIM_GAP0);
        else if constexpr (GAP == 1) TIM_MAT_ASM(MM16, "16", TIM_GAP1);
        else if constexpr (GAP == 2) TIM_MAT_ASM(MM16, "16", TIM_GAP2);
        else TIM_MAT_ASM(MM16, "16", TIM_GAP3);
        if (threadIdx.x % 32 == 0) {
            for (int i = 0; i < 18; ++i) {
                out[(w * REPS + rep) * 18 + i] = st[i];
            }
        }
    }
    if (threadIdx.x % 32 == 0) {
        out[MAXW * 2 * REPS * 18 + w] = hw;
    }
}

// FORM: 0 FP16, 1 FP8, 2 BF16, 3 sparse FP16, 4 8-bit integer, 5 FP16 into FP16, 6 BF8.
template <int FORM>
__global__ __launch_bounds__(1024) void k_ho(int prio, uint32_t * out) {
    const int w = threadIdx.x / 32;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    (void) prio;
    for (int rep = 0; rep < REPS; ++rep) {
        __syncthreads();
        uint32_t st[18];
        if constexpr (FORM == 0) TIM_MAT_ASM(MM16, "16", TIM_GAP0);
        else if constexpr (FORM == 1) TIM_MAT_ASM(MM8, "16", TIM_GAP0);
        else if constexpr (FORM == 2) TIM_MAT_ASM(MMBF, "16", TIM_GAP0);
        else if constexpr (FORM == 3) TIM_MAT_ASM(MMSF, "16", TIM_GAP0);
        else if constexpr (FORM == 4) TIM_MAT_ASM(MMI8, "16", TIM_GAP0);
        else if constexpr (FORM == 5) TIM_MAT_ASM(MMHH, "16", TIM_GAP0);
        else TIM_MAT_ASM(MMB8, "16", TIM_GAP0);
        if (threadIdx.x % 32 == 0) {
            for (int i = 0; i < 18; ++i) {
                out[(w * REPS + rep) * 18 + i] = st[i];
            }
        }
    }
    if (threadIdx.x % 32 == 0) {
        out[MAXW * 2 * REPS * 18 + w] = hw;
    }
}

struct Res {
    double simd_cycles_per_mm, switches_per_rep, stretched_share;
    int nsimd;
    std::vector<int> finish_order;  // waves by finish time, repetition 1
};

using Kern = void (*)(int, uint32_t *);
// gap 4 and 5: the four-accumulator chunk without and with stamps
Kern kern(int fp8, int gap) {
    static const Kern k[2][6] = {
        {k_mat<0, 0, 0>, k_mat<0, 1, 0>, k_mat<0, 2, 0>, k_mat<0, 3, 0>, k_mat<0, 0, 1>, k_mat<0, 3, 1>},
        {k_mat<1, 0, 0>, k_mat<1, 1, 0>, k_mat<1, 2, 0>, k_mat<1, 3, 0>, k_mat<1, 0, 1>, k_mat<1, 3, 1>}};
    return k[fp8][gap];
}

Res run(int fp8, int gap, int waves_per_simd, int prio, uint32_t * d, Kern kk = nullptr) {
    const int nw = 2 * waves_per_simd;
    hipLaunchKernelGGL(kk ? kk : kern(fp8, gap), dim3(1), dim3(32 * nw), 0, 0, prio, d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> h = tim::host_u32(d, MAXW * 2 * REPS * 18 + MAXW * 2);
    std::map<uint32_t, std::vector<int>> by;
    for (int w = 0; w < nw; ++w) {
        by[(h[MAXW * 2 * REPS * 18 + w] >> 8) & 3].push_back(w);
    }
    const std::vector<int> & v = by.begin()->second;
    const double mmc = fp8 ? 8.0 : 16.0;
    Res r{};
    r.nsimd = (int) v.size();
    std::vector<double> spans;
    double sw = 0, stretched = 0, chunks = 0;
    for (int rep = 1; rep < REPS; ++rep) {
        const uint32_t base = h[(v[0] * REPS + rep) * 18];
        std::vector<std::pair<double, int>> ends, fin;
        double lo = 1e18, hi = 0;
        for (int w : v) {
            const uint32_t * s = &h[(w * REPS + rep) * 18];
            lo = std::min(lo, (double) (s[0] - base + 0x80000000u));
            hi = std::max(hi, (double) (s[17] - base + 0x80000000u));
            fin.push_back({(double) (s[17] - base + 0x80000000u), w});
            if (gap == 3 || gap == 5) {
                for (int i = 1; i <= CHUNKS; ++i) {
                    ends.push_back({(double) (s[i] - base + 0x80000000u), w});
                    stretched += (double) (s[i] - s[i - 1]) > PER_CHUNK * mmc + 40;
                    chunks += 1;
                }
            }
        }
        std::sort(ends.begin(), ends.end());
        for (size_t i = 1; i < ends.size(); ++i) {
            sw += ends[i].second != ends[i - 1].second;
        }
        spans.push_back(hi - lo);
        if (rep == 1) {
            std::sort(fin.begin(), fin.end());
            for (auto & f : fin) {
                r.finish_order.push_back(f.second);
            }
        }
    }
    r.simd_cycles_per_mm = tim::median_of(spans) / ((double) v.size() * CHUNKS * PER_CHUNK);
    r.switches_per_rep = sw / (REPS - 1);
    r.stretched_share = chunks > 0 ? stretched / chunks : 0;
    return r;
}

}  // namespace

// MT-5b: hand-over cost per form.
int mode_handover(bool brk) {
    uint32_t * d = tim::dev_u32(MAXW * 2 * REPS * 18 + MAXW * 2);
    mem::clocks("start");
    const Kern ks[7] = {k_ho<0>, k_ho<1>, k_ho<2>, k_ho<3>, k_ho<4>, k_ho<5>, k_ho<6>};
    const char * names[7] = {"fp16", "fp8", "bf16", "sparse_fp16", "iu8", "fp16_into_fp16", "bf8"};
    double ho[7][4] = {};
    for (int f = 0; f < 7; ++f) {
        double one = 0;
        int wi = 0;
        for (int w : {1, 2, 4, 8}) {
            std::vector<double> per;
            for (int l = 0; l < 5; ++l) {
                per.push_back(run(f == 1 || f == 4 || f == 6, 0, w, 0, d, ks[f]).simd_cycles_per_mm);
            }
            const tim::Clean pc = tim::clean_of(per);
            if (w == 1) {
                one = pc.med;
            }
            // item 9a's model: span(W) = span(1) + (W - 1) x (256 x the chained cost + one hand-over), with the chained
            // cost of matrix-timing items 2 and 3 (16 or 8 cycles); span(1) holds the fixed start and end
            const double nominal = (f == 1 || f == 4 || f == 6) ? 8.0 : 16.0, n = CHUNKS * PER_CHUNK;
            const double per_ho = w > 1 ? (pc.med * w * n - one * n - (w - 1) * n * nominal) / (w - 1) : 0;
            ho[f][wi++] = per_ho;
            std::printf("RESULT handover %-15s waves_on_simd %d simd_cycles_per_mm %.3f (min %.3f max %.3f) cycles_per_handover "
                        "%.1f\n",
                        names[f], w, pc.med, pc.lo, pc.hi, per_ho);
            std::fflush(stdout);
        }
    }
    // calibration (item 9a): FP16 105 to 109 per hand-over, FP8 about 13, at W = 2, 4, 8; --break expects 13 for FP16
    bool ok = true;
    for (int wi = 1; wi < 4; ++wi) {
        ok = ok && (brk ? std::fabs(ho[0][wi] - 13) <= 3 : ho[0][wi] >= 95 && ho[0][wi] <= 120) && ho[1][wi] >= 8 && ho[1][wi] <= 18;
    }
    tim::check("tim_mat_handover_calib", ok, "cycles per hand-over FP16 %.0f %.0f %.0f (item 9a: 105 to 109%s), FP8 %.0f %.0f %.0f "
               "(item 9a: about 13) at W = 2, 4, 8", ho[0][1], ho[0][2], ho[0][3], brk ? "; break: expects 13" : "", ho[1][1],
               ho[1][2], ho[1][3]);
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = tim::flag(argc, argv, "--break");
    if (argc >= 2 && std::strcmp(argv[1], "handover") == 0) {
        return mode_handover(brk);
    }
    if (argc < 2 || std::strcmp(argv[1], "run") != 0) {
        std::fprintf(stderr, "usage: tim_mat run|handover [--break]\n");
        return 2;
    }
    uint32_t * d = tim::dev_u32(MAXW * 2 * REPS * 18 + MAXW * 2);
    mem::clocks("start");
    std::map<std::string, double> got;
    const char * gap_name[6] = {"none", "nop8", "branch", "stamp", "x4", "x4stamp"};
    size_t dropped = 0, total = 0;
    for (int fp8 : {0, 1}) {
        for (int gap : {0, 1, 2, 3, 4, 5}) {
            for (int prio : {0, 1}) {
                for (int w : {1, 2, 4, 8}) {
                    if (prio && w == 1) {
                        continue;
                    }
                    std::vector<double> per, swr, str;
                    Res last{};
                    for (int l = 0; l < 5; ++l) {
                        last = run(fp8, gap, w, prio, d);
                        per.push_back(last.simd_cycles_per_mm);
                        swr.push_back(last.switches_per_rep);
                        str.push_back(last.stretched_share);
                    }
                    const tim::Clean pc = tim::clean_of(per);
                    dropped += pc.dropped;
                    total += pc.n;
                    std::printf("RESULT mat %s gap %-7s prio %d waves_on_simd %d simd_cycles_per_mm %.3f (min %.3f max %.3f) "
                                "switches_per_rep %.1f stretched_chunks %.3f finish_order",
                                fp8 ? "fp8 " : "fp16", gap_name[gap], prio, last.nsimd, pc.med, pc.lo, pc.hi,
                                tim::median_of(swr), tim::median_of(str));
                    for (int x : last.finish_order) {
                        std::printf(" %d", x);
                    }
                    std::printf("\n");
                    std::fflush(stdout);
                    got[std::string(fp8 ? "fp8_" : "fp16_") + gap_name[gap] + "_" + std::to_string(prio) + "_" +
                        std::to_string(w)] = pc.med;
                }
            }
        }
    }
    std::printf("RESULT mat disturbance rule: %zu of %zu launch medians dropped\n", dropped, total);
    const double want = brk ? 32.0 : 16.0;
    tim::check("tim_mat_calib", std::fabs(got["fp16_none_0_1"] - want) < 0.1 && std::fabs(got["fp8_none_0_1"] - 8.0) < 0.1,
               "one wave: FP16 chain %.3f cycles (expected %.0f%s), FP8 %.3f (expected 8)", got["fp16_none_0_1"], want,
               brk ? ", deliberate break" : "", got["fp8_none_0_1"]);
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
