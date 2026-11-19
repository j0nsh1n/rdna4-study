// Leftovers C, lane A: instruction-issue items timed with one wave (measurements/leftovers-c-a.md).
//
// Usage: left_core <w64|bank|run12|sgprw> [--break]
//   Built twice by scripts/study-leftovers-c-a.sh: wave32 (left_core) and -mwavefrontsize64 -DW64=1 (left_core64).
//   w64     SC-10: the blocks of LEFT_W_BLOCKS (FMA forms, encodings, register banks, other classes) in this build's
//           wave size; the wave64 build against the wave32 build gives the wave64 issue cost per instruction kind.
//   bank    EC-5 (wave32): an FMA reading three different registers of one bank, separated by k other FMAs.
//   run12   EC-6 (wave32): runs of L 12-byte instructions between G 8-byte ones; a fetch model is fitted (INFO).
//   sgprw   EC-8 (wave32): n consecutive VALU instructions writing an SGPR, then m FMAs.
//   insn    EC-1, EC-2a to EC-2e (wave32): v_permlane64_b32, the pseudo-scalar FP16 transcendentals beside their FP32
//           twins, v_sqrt_f64 and v_rsq_f64 beside v_rcp_f64, v_dot2_f16_f16 and v_dot2_bf16_bf16 beside
//           v_dot2_f32_f16, the 64-bit add pair (carry through VCC or an SGPR), v_div_scale_f32 and v_div_fmas_f32;
//           independent (_c16, _c8) and chained (_c1). Correctness in the same run: v_permlane64_b32 leaves every
//           lane's value in wave32 (guide: a no-op), the add pair carries (0xFFFFFFFF + 1), and the full FP32 division
//           sequence (div_scale, rcp, fma steps, div_fmas, div_fixup) gives the correctly rounded quotient for 32
//           lanes of test values. Verdict ins_calib: the run's independent FMA 1.00, every block linear, every check
//           right; the costs are INFO. --break expects the FMA at 2.00: FAIL.
//   fsize   EC-7: where the step at 4 KB of loop body (execution-core 8b: 20-21 cycles of loop control per iteration
//           below 4 KB, 34 at 4 and 5 KB) belongs. One wave at 2, 3, 4, 5 KB (calibration); two 3 KB copies of the same
//           loop at different addresses, run by waves on one SIMD, on the two SIMDs of a CU (CU-mode build) and on the
//           two CUs of a WGP (left_core_wgp, WGP mode), each either all on one copy or on different copies; a 4 KB loop
//           started 0, 64, 128 and 2,048 bytes past a 4 KB boundary. Loop control per iteration = cycles per iteration
//           minus the instructions (one cycle each). The table is INFO.
//   --break the mode's calibration expects a wrong value, so its verdict must FAIL.
// Method (tim_core.hip, measurements/execution-core.md item 8): one wave alone; each timed region is one
// inline-assembly statement holding both counter reads and the loop of NIT iterations of R copies of one block, with
// the registers named in the block text (scripts/leftc_gen_core.py), so nothing the compiler adds can land inside.
// Cost per instruction = (cycles(R=6) - cycles(R=2)) / (4 x n x NIT), n instructions per block; R = 4 must sit midway
// (within 1.5 cycles per iteration). Samples above 1.5 x median + 64 cycles are dropped (lane 7's rule).
#include "tim_common.h"
#include "left_core_blocks.h"

#include <map>

#ifndef W64
#define W64 0
#endif
#ifndef WGP_BUILD
#define WGP_BUILD 0
#endif

namespace {

constexpr int WS = W64 ? 64 : 32;
constexpr int REPS = 6;  // per launch; repetition 0 warms the instruction cache and is dropped
constexpr int LAUNCHES = 3;
constexpr uint32_t NIT = 16;

#define LEFT_CLOBBERS                                                                                                \
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", \
        "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46",    \
        "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61",    \
        "v62", "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76",    \
        "v77", "v78", "v79", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v91",    \
        "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "s16", "s17", "s18", \
        "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "s32", "s33",    \
        "s34", "s35", "s36", "s37", "s38", "s39", "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "vcc",    \
        "scc", "memory"

// out: per repetition [t0, t1] of the wave (lane 0).
#define LEFT_K(NAME, TEXT, R)                                                                                        \
    __global__ __launch_bounds__(64) void k_##NAME##_##R(uint32_t n_iter, uint32_t * out) {                         \
        for (int rep = 0; rep < REPS; ++rep) {                                                                       \
            uint32_t t0, t1, cnt;                                                                                    \
            asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n"       \
                         ".rept " #R "\n" TEXT ".endr\n"                                                             \
                         " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"           \
                         " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                     \
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt)                                         \
                         : [n] "s"(n_iter)                                                                           \
                         : LEFT_CLOBBERS);                                                                           \
            if (threadIdx.x == 0) {                                                                                  \
                out[2 * rep] = t0;                                                                                   \
                out[2 * rep + 1] = t1;                                                                               \
            }                                                                                                        \
        }                                                                                                            \
    }
#define LEFT_KS(NAME, N, TEXT) LEFT_K(NAME, TEXT, 2) LEFT_K(NAME, TEXT, 4) LEFT_K(NAME, TEXT, 6)

using Kern = void (*)(uint32_t, uint32_t *);
struct Block {
    const char * name;
    int n;
    Kern k[3];
};
#define LEFT_ROW(NAME, N, TEXT) {#NAME, N, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6}},

LEFT_W_BLOCKS(LEFT_KS)
#if W64
LEFT_W64_BLOCKS(LEFT_KS)
const Block wblocks[] = {LEFT_W_BLOCKS(LEFT_ROW) LEFT_W64_BLOCKS(LEFT_ROW)};
#else
LEFT_W32_BLOCKS(LEFT_KS)
LEFT_E_BLOCKS(LEFT_KS)
const Block wblocks[] = {LEFT_W_BLOCKS(LEFT_ROW) LEFT_W32_BLOCKS(LEFT_ROW)};
const Block eblocks[] = {LEFT_E_BLOCKS(LEFT_ROW)};
#endif

uint32_t * g_out = nullptr;

std::vector<double> samples(Kern k) {
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(WS), 0, 0, NIT, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 2 * REPS);
        for (int rep = 1; rep < REPS; ++rep) {
            v.push_back((double) (h[2 * rep + 1] - h[2 * rep]));
        }
    }
    return v;
}

struct Cost {
    double per, mid, lo, hi;
    size_t n, dropped;
    double inc24, inc46;  // cycles per instruction from R = 2 to 4 and from 4 to 6
};

Cost cost(const Block & b) {
    tim::Clean m[3];
    Cost c{};
    for (int i = 0; i < 3; ++i) {
        m[i] = tim::clean_of(samples(b.k[i]));
        c.n += m[i].n;
        c.dropped += m[i].dropped;
    }
    const double d = 4.0 * b.n * NIT;
    c.per = (m[2].med - m[0].med) / d;
    c.lo = (m[2].lo - m[0].hi) / d;
    c.hi = (m[2].hi - m[0].lo) / d;
    c.mid = (m[1].med - 0.5 * (m[0].med + m[2].med)) / NIT;
    c.inc24 = (m[1].med - m[0].med) / (2.0 * b.n * NIT);
    c.inc46 = (m[2].med - m[1].med) / (2.0 * b.n * NIT);
    return c;
}

std::map<std::string, Cost> run_all(const Block * blocks, size_t nb, const char * tag, const char * prefix,
                                    int & nonlin) {
    std::map<std::string, Cost> got;
    nonlin = 0;
    for (size_t i = 0; i < nb; ++i) {
        const Block & b = blocks[i];
        if (prefix && std::strncmp(b.name, prefix, std::strlen(prefix)) != 0) {
            continue;
        }
        const Cost c = cost(b);
        got[b.name] = c;
        nonlin += std::fabs(c.mid) > 1.5;
        std::printf("RESULT %s wave%d %-22s n %2d cycles_per_instr %.4f range %.4f-%.4f midpoint_dev_per_it %.2f "
                    "samples %zu dropped %zu\n",
                    tag, WS, b.name, b.n, c.per, c.lo, c.hi, c.mid, c.n, c.dropped);
        std::fflush(stdout);
    }
    return got;
}

bool near(double v, double want, double tol) { return std::fabs(v - want) <= tol; }

int mode_w64(bool brk) {
    int nonlin = 0;
    auto got = run_all(wblocks, sizeof wblocks / sizeof wblocks[0], "w64", nullptr, nonlin);
    const double want = brk ? 2.0 : 1.0;
    if (!W64) {
        tim::check("w64_calib", near(got["w_fma_ic"].per, want, 0.02) && near(got["w_dep"].per, 5.0, 0.05) &&
                                    near(got["w_wmma"].per, 16.0, 0.1) && nonlin == 0,
                   "wave32: independent FMA %.4f (expect %.2f%s), dependent FMA %.4f (expect 5.00), chained FP16 matrix "
                   "%.3f (expect 16.0); %d blocks off the midpoint",
                   got["w_fma_ic"].per, want, brk ? ", deliberate break" : "", got["w_dep"].per, got["w_wmma"].per,
                   nonlin);
    } else {
        tim::check("w64_calib64", near(got["w_wmma"].per, brk ? 8.0 : 16.0, 0.1) && nonlin == 0,
                   "wave64: chained FP16 matrix %.3f cycles (expect %.1f: sched item 4, 16.000 in wave64%s); %d blocks "
                   "off the midpoint",
                   got["w_wmma"].per, brk ? 8.0 : 16.0, brk ? ", deliberate break" : "", nonlin);
        std::printf("VERDICT w64_rates: INFO wave64 cycles per instruction: fma_ic %.3f fma_v1 %.3f fma_vv %.3f "
                    "fma_vv_spread %.3f fmac(e32) %.3f fmac_spread %.3f fmac_e64 %.3f add_e32 %.3f add_e64 %.3f mov %.3f "
                    "addu %.3f (HIP sched_wave form 1.234, Vulkan fmac 1.06)\n",
                    got["w_fma_ic"].per, got["w_fma_v1"].per, got["w_fma_vv"].per, got["w_fma_vv_spread"].per,
                    got["w_fmac"].per, got["w_fmac_spread"].per, got["w_fmac_e64"].per, got["w_add_e32"].per,
                    got["w_add_e64"].per, got["w_mov"].per, got["w_addu"].per);
    }
    return 0;
}

#if !W64
// ---- EC-7 -----------------------------------------------------------------------------------------------------------
#define FS_BODY                                                                                                        \
    "s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n .rept %c[r]\n" FS_FMA16          \
    ".endr\n s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"                          \
    " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
#define FS_FMA16 "v_fma_f32 v16, v16, 0.5, 0.5\nv_fma_f32 v17, v17, 0.5, 0.5\nv_fma_f32 v18, v18, 0.5, 0.5\nv_fma_f32 v19, v19, 0.5, 0.5\nv_fma_f32 v20, v20, 0.5, 0.5\nv_fma_f32 v21, v21, 0.5, 0.5\nv_fma_f32 v22, v22, 0.5, 0.5\nv_fma_f32 v23, v23, 0.5, 0.5\nv_fma_f32 v24, v24, 0.5, 0.5\nv_fma_f32 v25, v25, 0.5, 0.5\nv_fma_f32 v26, v26, 0.5, 0.5\nv_fma_f32 v27, v27, 0.5, 0.5\nv_fma_f32 v28, v28, 0.5, 0.5\nv_fma_f32 v29, v29, 0.5, 0.5\nv_fma_f32 v30, v30, 0.5, 0.5\nv_fma_f32 v31, v31, 0.5, 0.5\n"
#define FS_OPS : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt) : [n] "s"(n_iter), [r] "i"(R) : LEFT_CLOBBERS

constexpr int FS_W = 8;
// out per wave w and repetition: [2 (w REPS + rep)] t0, t1; then [2 FS_W REPS + w] HW_ID1, [.. + FS_W + w] copy used.
// sel: 0 every wave on copy A; 1 copy by SIMD in CU (HW_ID1 bit 9); 2 by CU in WGP (bit 8); 3 by rank on the SIMD;
// 4 every wave on copy B (the control for copy B's own placement in memory); 5 by CU in WGP, swapped (CU 0 copy B,
// CU 1 copy A: does the slow side follow the CU or the copy?). Fix round, test T2.
template <int R>
__global__ __launch_bounds__(256) void k_fs(uint32_t n_iter, int sel, uint32_t * out) {
    __shared__ uint32_t rank_ctr[4];
    const int w = threadIdx.x / 32;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    if (threadIdx.x < 4) rank_ctr[threadIdx.x] = 0;
    __syncthreads();
    uint32_t rank = 0;
    if (threadIdx.x % 32 == 0) rank = atomicAdd(&rank_ctr[(hw >> 8) & 3], 1u);
    rank = __builtin_amdgcn_readfirstlane(__shfl(rank, 0));
    const int copy = sel == 0 ? 0 : sel == 1 ? (hw >> 9) & 1 : sel == 2 ? (hw >> 8) & 1 : sel == 3 ? (int) (rank & 1) : sel == 4 ? 1 : 1 - (int) ((hw >> 8) & 1);
    for (int rep = 0; rep < REPS; ++rep) {
        __syncthreads();
        uint32_t t0, t1, cnt;
        if (copy == 0) {
            asm volatile(FS_BODY FS_OPS);
        } else {
            asm volatile(FS_BODY FS_OPS);
        }
        if (threadIdx.x % 32 == 0) {
            out[2 * (w * REPS + rep)] = t0;
            out[2 * (w * REPS + rep) + 1] = t1;
        }
    }
    if (threadIdx.x % 32 == 0) {
        out[2 * FS_W * REPS + w] = hw;
        out[2 * FS_W * REPS + FS_W + w] = copy;
    }
}

// one wave, a 4 KB loop whose first instruction lies OFF bytes past a 4 KB boundary (s_nop padding executed once)
template <int OFF>
__global__ __launch_bounds__(32) void k_fs_align(uint32_t n_iter, uint32_t * out) {
    for (int rep = 0; rep < REPS; ++rep) {
        uint32_t t0, t1, cnt;
        asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n s_branch 3f\n"
                     " .p2align 12\n .rept %c[pad]\n s_nop 0\n .endr\n 3:\n 2:\n .rept 32\n" FS_FMA16
                     ".endr\n s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"
                     " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                     : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt)
                     : [n] "s"(n_iter), [pad] "i"(OFF / 4)
                     : LEFT_CLOBBERS);
        if (threadIdx.x == 0) {
            out[2 * rep] = t0;
            out[2 * rep + 1] = t1;
        }
    }
}

// Loop control per iteration for each wave (median over repetitions 1..), grouped by copy. Returns per-wave values.
struct FsRow {
    std::vector<double> ctl;  // per wave: cycles per iteration minus the instructions of one iteration
    std::vector<int> copy, simd;
};
FsRow fs_run(void (*k)(uint32_t, int, uint32_t *), int waves, int sel, int R, uint32_t * d) {
    FsRow r;
    std::vector<std::vector<double>> per(waves);
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32 * waves), 0, 0, NIT, sel, d);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(d, 2 * FS_W * REPS + 2 * FS_W);
        for (int w = 0; w < waves; ++w) {
            for (int rep = 1; rep < REPS; ++rep) {
                per[w].push_back((double) (h[2 * (w * REPS + rep) + 1] - h[2 * (w * REPS + rep)]) / NIT);
            }
            if (l == 0) {
                r.copy.push_back((int) h[2 * FS_W * REPS + FS_W + w]);
                r.simd.push_back((int) ((h[2 * FS_W * REPS + w] >> 8) & 3));
            }
        }
    }
    for (int w = 0; w < waves; ++w) r.ctl.push_back(tim::clean_of(per[w]).med - 16.0 * R);
    return r;
}

int mode_fsize(bool brk) {
    uint32_t * d;
    CHECK(hipMalloc(&d, (2 * FS_W * REPS + 2 * FS_W) * 4));
    // calibration: one wave
    std::map<int, double> one;
    const int rs[4] = {16, 24, 32, 40};
    void (*ks[4])(uint32_t, int, uint32_t *) = {k_fs<16>, k_fs<24>, k_fs<32>, k_fs<40>};
    for (int i = 0; i < 4; ++i) {
        one[rs[i]] = fs_run(ks[i], 1, 0, rs[i], d).ctl[0];
        std::printf("RESULT fsize one_wave body_KB %d loop_control_per_iteration %.1f\n", rs[i] * 128 / 1024, one[rs[i]]);
    }
    const double w2 = brk ? 40.0 : 20.5;
    // the 4 KB point is judged on the loop that starts at a 4 KB boundary (offset 0 below): the cost of a 4 KB loop
    // depends on where it starts, which execution-core 8b did not control
    double al0 = 0;
    {
        std::vector<double> v;
        for (int l = 0; l < LAUNCHES; ++l) {
            hipLaunchKernelGGL(k_fs_align<0>, dim3(1), dim3(32), 0, 0, NIT, d);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const std::vector<uint32_t> h = tim::host_u32(d, 2 * REPS);
            for (int rep = 1; rep < REPS; ++rep) v.push_back((double) (h[2 * rep + 1] - h[2 * rep]) / NIT);
        }
        al0 = tim::clean_of(v).med - 16.0 * 32;
    }
    tim::check("fsize_calib", std::fabs(one[16] - w2) <= 2 && std::fabs(one[24] - 21) <= 2 && std::fabs(al0 - 34) <= 2,
               "one wave, loop control per iteration: 2 KB %.1f (expect %.1f%s), 3 KB %.1f (21), 4 KB starting at a 4 KB "
               "boundary %.1f (34); +-2 (execution-core 8b); 4 KB in the two-copy kernel %.1f (INFO)",
               one[16], w2, brk ? ", deliberate break" : "", one[24], al0, one[32]);
    for (int sel : {0, 4}) {
        const FsRow r = fs_run(k_fs<24>, 1, sel, 24, d);
        std::printf("RESULT fsize one wave alone on copy %s, 3 KB: loop control per iteration %.1f\n", sel ? "B" : "A",
                    r.ctl[0]);
    }
    // two copies at 3 KB: waves together on one copy, or on different copies
    struct C {
        const char * name;
        int waves, sel_diff;
    } cs[] = {{"2 waves, one per SIMD of the CU", 2, 1}, {"4 waves, 2 per SIMD", 4, 3}, {"2 waves, two CUs (WGP build)", 4, 2}};
    for (const C & c : cs) {
        if (std::strstr(c.name, "WGP") && !WGP_BUILD) continue;
        if (!std::strstr(c.name, "WGP") && WGP_BUILD) continue;
        for (int diff = 0; diff < (WGP_BUILD ? 4 : 3); ++diff) {
            const FsRow r = fs_run(k_fs<24>, c.waves, diff == 1 ? c.sel_diff : diff == 2 ? 4 : diff == 3 ? 5 : 0, 24, d);
            std::string txt;
            for (size_t w = 0; w < r.ctl.size(); ++w) {
                char b[64];
                std::snprintf(b, sizeof b, " [simd %d copy %d: %.1f]", r.simd[w], r.copy[w], r.ctl[w]);
                txt += b;
            }
            std::printf("RESULT fsize %s, %s, 3 KB each: loop control per iteration per wave%s\n", c.name,
                        diff == 1 ? "different copies" : diff == 2 ? "all on copy B"
                                        : diff == 3 ? "different copies swapped (CU 0 copy B, CU 1 copy A)"
                                                    : "all on copy A",
                        txt.c_str());
        }
    }
    // alignment of a 4 KB loop
    if (!WGP_BUILD) {
        void (*ka[4])(uint32_t, uint32_t *) = {k_fs_align<0>, k_fs_align<64>, k_fs_align<128>, k_fs_align<2048>};
        const int offs[4] = {0, 64, 128, 2048};
        for (int i = 0; i < 4; ++i) {
            std::vector<double> v;
            for (int l = 0; l < LAUNCHES; ++l) {
                hipLaunchKernelGGL(ka[i], dim3(1), dim3(32), 0, 0, NIT, d);
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                const std::vector<uint32_t> h = tim::host_u32(d, 2 * REPS);
                for (int rep = 1; rep < REPS; ++rep) v.push_back((double) (h[2 * rep + 1] - h[2 * rep]) / NIT);
            }
            std::printf("RESULT fsize 4KB_loop start_offset %d loop_control_per_iteration %.1f\n", offs[i],
                        tim::clean_of(v).med - 16.0 * 32);
        }
    }
    return 0;
}

// Steady-state cycles per instruction of a periodic sequence of instruction sizes, for a fetch of F bytes per cycle
// into a buffer of Q bytes; one instruction issues per cycle when all its bytes are in the buffer.
double fetch_model(const std::vector<int> & sz, double F, double Q) {
    double buf = Q;
    size_t idx = 0;
    long issued = 0, cyc = 0, warm = 0, warm_cyc = 0;
    const long total = 4000;
    while (issued < total) {
        ++cyc;
        buf = std::min(Q, buf + F);
        if (buf >= sz[idx % sz.size()]) {
            buf -= sz[idx % sz.size()];
            ++idx;
            ++issued;
            if (issued == 1000) {
                warm = issued;
                warm_cyc = cyc;
            }
        }
    }
    return double(cyc - warm_cyc) / double(issued - warm);
}

int mode_run12(bool brk) {
    int nonlin = 0;
    auto got = run_all(eblocks, sizeof eblocks / sizeof eblocks[0], "run12", "run_", nonlin);
    // known values, execution-core item 8a
    struct Kn {
        const char * name;
        double v;
    } kn[] = {{"run_L16_G0", 1.2578}, {"run_L1_G3", 1.0010}, {"run_L2_G2", 1.0088}, {"run_L3_G1", 1.1309},
              {"run_L4_G4", 1.1260},  {"run_L8_G8", 1.1250}};
    double worst = 0;
    std::string txt;
    for (const Kn & k : kn) {
        const double want = brk && std::strcmp(k.name, "run_L16_G0") == 0 ? 1.0 : k.v;
        worst = std::max(worst, std::fabs(got[k.name].per - want));
        char b[96];
        std::snprintf(b, sizeof b, " %s %.4f/%.4f", k.name + 4, got[k.name].per, want);
        txt += b;
    }
    tim::check("run12_calib", worst <= 0.01 && nonlin == 0,
               "measured/known (execution-core 8a%s):%s; worst %.4f (limit 0.01); %d blocks off the midpoint",
               brk ? ", deliberate break: all-12-byte expected at 1.00" : "", txt.c_str(), worst, nonlin);
    // fit of the fetch model over every point
    double bestF = 0, bestQ = 0, bestErr = 1e9;
    for (double F = 8.0; F <= 16.0; F += 0.125) {
        for (double Q = 12; Q <= 256; Q += 4) {
            double err = 0;
            for (auto & kv : got) {
                int L = 0, G = 0;
                std::sscanf(kv.first.c_str(), "run_L%d_G%d", &L, &G);
                std::vector<int> sz;
                for (int i = 0; i < L; ++i) sz.push_back(12);
                for (int i = 0; i < G; ++i) sz.push_back(8);
                err = std::max(err, std::fabs(fetch_model(sz, F, Q) - kv.second.per));
            }
            if (err < bestErr) {
                bestErr = err;
                bestF = F;
                bestQ = Q;
            }
        }
    }
    std::printf("VERDICT run12_fit: INFO best fetch model (F bytes per cycle into a Q-byte buffer, one instruction per "
                "cycle): F %.3f Q %.0f, largest error %.4f cycles per instruction over %zu points\n",
                bestF, bestQ, bestErr, got.size());
    for (auto & kv : got) {
        int L = 0, G = 0;
        std::sscanf(kv.first.c_str(), "run_L%d_G%d", &L, &G);
        std::vector<int> sz;
        for (int i = 0; i < L; ++i) sz.push_back(12);
        for (int i = 0; i < G; ++i) sz.push_back(8);
        std::printf("RESULT run12_table L %2d G %2d measured %.4f model %.4f\n", L, G, kv.second.per,
                    fetch_model(sz, bestF, bestQ));
    }
    return 0;
}

int mode_bank(bool brk) {
    int nonlin = 0;
    auto got = run_all(eblocks, sizeof eblocks / sizeof eblocks[0], "bank", "bk_", nonlin);
    const double s = got["bk_cal_same"].per * 2, a = got["bk_cal_alt"].per * 2, m = got["bk_cal_mixed"].per * 2;
    const double ws = brk ? 2.0 : 2.968;
    tim::check("bank_calib", near(s, ws, 0.05) && near(a, 9.999, 0.05) && near(m, 7.999, 0.05) && nonlin == 0,
               "per pair: same bank-1 triple %.3f (known 2.968%s), two bank-1 triples alternating %.3f (9.999), triple "
               "then (v45,v49,v34) %.3f (7.999); %d blocks off the midpoint",
               s, brk ? ", deliberate break: expected 2.0" : "", a, m, nonlin);
    for (auto & kv : got) {
        int k = -1;
        char tri[8] = {0}, kind = 0;
        if (std::sscanf(kv.first.c_str(), "bk_%5[a-z]_%c_k%d", tri, &kind, &k) == 3) {
            const double period = kv.second.per * (k + 1);
            std::printf("RESULT bank_extra triple %-5s fill %c k %2d cycles_per_period %.3f extra_cycles_of_X %.3f\n", tri,
                        kind, k, period, period - (k + 1));
        }
    }
    return 0;
}

// out per lane: [0] permlane64 result, [1] 64-bit add low, [2] high, [3] quotient bits, [4] approximate quotient
// (v_rcp_f32 x numerator): the counter-example that shows the division check can fail
__global__ __launch_bounds__(32) void k_ins_check(const float * num, const float * den, uint32_t * out) {
    const uint32_t l = threadIdx.x;
    uint32_t p = l * 3 + 1, lo = 0xFFFFFFFFu, hi = l, one = 1, zero = 0;
    asm volatile("v_permlane64_b32 %0, %0" : "+v"(p));
    asm volatile("v_add_co_u32 %0, vcc_lo, %0, %2\n v_add_co_ci_u32 %1, vcc_lo, %1, %3, vcc_lo"
                 : "+v"(lo), "+v"(hi)
                 : "v"(one), "v"(zero)
                 : "vcc");
    float q;
    const float n = num[l], d = den[l];
    // the division sequence LLVM emits for an IEEE FP32 divide (denormals kept)
    asm volatile("v_div_scale_f32 v40, null, %2, %2, %1\n"
                 "v_div_scale_f32 v41, vcc_lo, %1, %2, %1\n"
                 "v_rcp_f32 v42, v40\n"
                 "v_fma_f32 v43, -v40, v42, 1.0\n"
                 "v_fmac_f32 v42, v43, v42\n"
                 "v_mul_f32 v43, v41, v42\n"
                 "v_fma_f32 v44, -v40, v43, v41\n"
                 "v_fmac_f32 v43, v44, v42\n"
                 "v_fma_f32 v40, -v40, v43, v41\n"
                 "v_div_fmas_f32 v40, v40, v42, v43\n"
                 "v_div_fixup_f32 %0, v40, %2, %1\n"
                 : "=v"(q)
                 : "v"(n), "v"(d)
                 : "v40", "v41", "v42", "v43", "v44", "vcc");
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(d));
    out[l * 4 + 0] = p;
    out[l * 4 + 1] = lo;
    out[l * 4 + 2] = hi;
    out[l * 4 + 3] = __float_as_uint(q);
    out[128 + l] = __float_as_uint(n * r);
}

int mode_insn(bool brk) {
    int nonlin = 0, n2 = 0;
    auto cal = run_all(wblocks, sizeof wblocks / sizeof wblocks[0], "insn", "w_fma_ic", n2);
    auto got = run_all(eblocks, sizeof eblocks / sizeof eblocks[0], "insn", "ins_", nonlin);
    // correctness
    float hn[32], hd[32];
    for (int i = 0; i < 32; ++i) {
        hn[i] = std::ldexp(1.0f + i * 0.0371f, (i % 7) * 9 - 30);
        hd[i] = std::ldexp(3.0f - i * 0.0517f, (i % 5) * 11 - 20);
    }
    hn[0] = 1.0f, hd[0] = 3.0f;
    hn[1] = 7.0f, hd[1] = 7.0f;
    float *dn, *dd;
    uint32_t * dout;
    CHECK(hipMalloc(&dn, 128));
    CHECK(hipMalloc(&dd, 128));
    CHECK(hipMalloc(&dout, 640));
    CHECK(hipMemcpy(dn, hn, 128, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dd, hd, 128, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_ins_check, dim3(1), dim3(32), 0, 0, dn, dd, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint32_t h[160];
    CHECK(hipMemcpy(h, dout, 640, hipMemcpyDeviceToHost));
    int pl_bad = 0, add_bad = 0, div_bad = 0, approx_diff = 0;
    for (int i = 0; i < 32; ++i) {
        pl_bad += h[i * 4] != (uint32_t) (i * 3 + 1);
        add_bad += h[i * 4 + 1] != 0 || h[i * 4 + 2] != (uint32_t) i + 1;
        const float want = hn[i] / hd[i];  // the host divides in IEEE single precision, round to nearest even
        uint32_t wb;
        std::memcpy(&wb, &want, 4);
        div_bad += h[i * 4 + 3] != wb;
        approx_diff += h[128 + i] != wb;
    }
    std::printf("RESULT insn checks: v_permlane64_b32 changed %d of 32 lanes; 64-bit add 0xFFFFFFFF + 1 wrong in %d; "
                "division sequence differs from the correctly rounded quotient in %d of 32 lanes (the approximate "
                "v_rcp_f32 x numerator differs in %d: the check can fail)\n",
                pl_bad, add_bad, div_bad, approx_diff);
    const double want = brk ? 2.0 : 1.0;
    // The method's linearity is judged on the calibration block; a measured block that is not linear has no single
    // cost per instruction, and is reported with its two increments.
    std::string nl;
    for (auto & kv : got) {
        if (std::fabs(kv.second.mid) > 1.5) {
            char b[160];
            std::snprintf(b, sizeof b, " %s (R 2->4: %.3f, 4->6: %.3f cycles per instruction)", kv.first.c_str(),
                          kv.second.inc24, kv.second.inc46);
            nl += b;
        }
    }
    std::printf("VERDICT ins_nonlinear: INFO blocks whose cost changes with the number of blocks in a row:%s\n",
                nl.empty() ? " none" : nl.c_str());
    tim::check("ins_calib",
               near(cal["w_fma_ic"].per, want, 0.02) && n2 == 0 && pl_bad == 0 && add_bad == 0 && div_bad == 0 &&
                   approx_diff > 0,
               "independent FMA %.4f (expect %.2f%s), linear: %s; permlane64 changed %d lanes, 64-bit add wrong in %d, "
               "division wrong in %d (all expect 0)",
               cal["w_fma_ic"].per, want, brk ? ", deliberate break" : "", n2 == 0 ? "yes" : "NO", pl_bad, add_bad,
               div_bad);
    std::printf("VERDICT ins_costs: INFO cycles per instruction, independent / chained: permlane64 %.3f / %.3f; "
                "v_s_rcp f16 %.3f / %.3f, f32 %.3f / %.3f; v_s_exp f16 %.3f / %.3f, f32 %.3f / %.3f; sqrt_f64 %.2f / "
                "%.2f; rsq_f64 %.2f / %.2f; rcp_f64 %.2f / %.2f; dot2_f16_f16 %.3f / %.3f; dot2_bf16_bf16 %.3f / %.3f; "
                "dot2_f32_f16 %.3f / %.3f; add64 pair (VCC) %.3f per instruction, chained %.3f; add64 (SGPR carry) "
                "%.3f; div_scale %.3f, div_fmas %.3f, pair %.3f per instruction\n",
                got["ins_permlane64_c16"].per, got["ins_permlane64_c1"].per, got["ins_s_rcp_f16_c16"].per,
                got["ins_s_rcp_f16_c1"].per, got["ins_s_rcp_f32_c16"].per, got["ins_s_rcp_f32_c1"].per,
                got["ins_s_exp_f16_c16"].per, got["ins_s_exp_f16_c1"].per, got["ins_s_exp_f32_c16"].per,
                got["ins_s_exp_f32_c1"].per, got["ins_sqrt_f64_c8"].per, got["ins_sqrt_f64_c1"].per,
                got["ins_rsq_f64_c8"].per, got["ins_rsq_f64_c1"].per, got["ins_rcp_f64_c8"].per,
                got["ins_rcp_f64_c1"].per, got["ins_dot2f16f16_c16"].per, got["ins_dot2f16f16_c1"].per,
                got["ins_dot2bf16bf16_c16"].per, got["ins_dot2bf16bf16_c1"].per, got["ins_dot2f32f16_c16"].per,
                got["ins_dot2f32f16_c1"].per, got["ins_add64_vcc_c8"].per, got["ins_add64_vcc_c1"].per,
                got["ins_add64_sgpr_c8"].per, got["ins_divscale_c16"].per, got["ins_divfmas_c16"].per,
                got["ins_divpair_c8"].per);
    return 0;
}

int mode_sgprw(bool brk) {
    int nonlin = 0;
    auto got = run_all(eblocks, sizeof eblocks / sizeof eblocks[0], "sgprw", "sw_", nonlin);
    const double r = got["sw_rfl_n16_m0"].per, c = got["sw_cmp_n16_m0"].per, e = got["sw_sexp_n1_m1"].per * 2;
    const double wr = brk ? 1.0 : 1.875;
    // linearity is judged on the three calibration blocks; other blocks off the midpoint are reported
    const bool lin = std::fabs(got["sw_rfl_n16_m0"].mid) <= 1.5 && std::fabs(got["sw_cmp_n16_m0"].mid) <= 1.5 &&
                     std::fabs(got["sw_sexp_n1_m1"].mid) <= 1.5;
    tim::check("sgprw_calib", near(r, wr, 0.02) && near(c, 1.625, 0.02) && near(e, 2.0, 0.02) && lin,
               "v_readfirstlane x16 %.4f (known 1.875%s), v_cmp to SGPR x16 %.4f (1.625), v_s_exp + FMA per pair %.4f "
               "(2.000); calibration blocks linear: %s; other blocks off the midpoint (INFO): %d",
               r, brk ? ", deliberate break: expected 1.00" : "", c, e, lin ? "yes" : "NO", nonlin);
    // model: the SGPR write path takes 8 writes per 15 cycles (8 per 13 for v_cmp); FMAs fill the idle cycles
    int fit = 0, pts = 0;
    for (auto & kv : got) {
        int n = 0, m = 0;
        char nm[8] = {0};
        if (std::sscanf(kv.first.c_str(), "sw_%4[a-z]_n%d_m%d", nm, &n, &m) == 3) {
            const double w = std::strcmp(nm, "cmp") == 0 ? 13.0 / 8 : 15.0 / 8;
            const double pred = std::max<double>(n + m, w * n), meas = kv.second.per * (n + m);
            ++pts;
            fit += std::fabs(meas - pred) <= 0.05;
            if (std::fabs(meas - pred) > 0.05) {
                std::printf("RESULT sgprw_model_miss %s n %d m %d measured %.3f predicted %.3f per period\n", nm, n, m, meas,
                            pred);
            }
        }
    }
    std::printf("VERDICT sgprw_model: INFO max(n + m, w x n) per period, w = 15/8 (readfirstlane, v_s_exp) or 13/8 "
                "(v_cmp): %d of %d points within 0.05 cycles\n",
                fit, pts);
    for (auto & kv : got) {
        int n = 0, m = 0;
        char nm[8] = {0};
        if (std::sscanf(kv.first.c_str(), "sw_%4[a-z]_n%d_m%d", nm, &n, &m) == 3) {
            const double period = kv.second.per * (n + m);
            std::printf("RESULT sgprw_table %-4s n %2d m %d cycles_per_period %.3f per_sgpr_write_after_fmas %.3f\n", nm,
                        n, m, period, (period - m) / n);
        }
    }
    return 0;
}
#endif

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = tim::flag(argc, argv, "--break");
    CHECK(hipMalloc(&g_out, 2 * REPS * 4));
    mem::clocks("start");
    if (mode == "w64") {
        mode_w64(brk);
#if !W64
    } else if (mode == "bank") {
        mode_bank(brk);
    } else if (mode == "run12") {
        mode_run12(brk);
    } else if (mode == "sgprw") {
        mode_sgprw(brk);
    } else if (mode == "insn") {
        mode_insn(brk);
    } else if (mode == "fsize") {
        mode_fsize(brk);
#endif
    } else {
        std::fprintf(stderr, "usage: left_core <w64%s> [--break]\n", W64 ? "" : "|bank|run12|sgprw|insn");
        return 2;
    }
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
