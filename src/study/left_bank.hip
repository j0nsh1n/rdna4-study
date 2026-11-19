// Leftovers C, lane A, fix round (measurements/leftovers-c-a.md): tests T1 and T3, kept out of left_core.hip so the
// left_core objects, whose code addresses EC-7 depends on, stay as they were.
//
// Usage: left_bank <bank|insdata> [--break]
//   Built twice by scripts/study-leftovers-c-a.sh: wave32 (left_bank) and -mwavefrontsize64 -DW64=1 (left_bank64).
//   bank     T1 (SC-10): the blocks of LEFT_T_BLOCKS (scripts/leftc_gen_core.py): every instruction with two VGPR
//            sources in one register bank (bank = register number mod 4), with three or two VGPR sources, with the
//            banks fixed or rotating, and blocks that only write or only read bank 0. Calibration (verdict
//            bank_calib): independent FMA 1.000 and dependent FMA 5.000 in both sizes, and the 8-of-16 block of SC-10
//            (v_fma_f32 vR, vR, v32, v33 over v16-v31) 1.250 in wave64 and 1.000 in wave32 (run 20261117T201152Z);
//            --break expects the independent FMA at 2.00. The other costs are INFO.
//   insdata  T3 (EC-2, wave32 only): the FP64 and pseudo-scalar blocks with every input register set to a chosen value
//            first (1.7, 1.0, 0, infinity, the smallest subnormal, a quiet NaN); the outputs of fixed inputs are
//            checked against the CPU (the data reached the instructions). Verdict insdata_calib: v_rcp_f64 with input
//            1.7 at 32.0 independent and 40.0 chained (EC-2) and every checked output right; --break expects 16.0.
//            Whether the costs depend on the data is INFO (insdata_costs).
// Method: left_core.hip's (one wave alone, one inline-assembly statement per timed region, registers named in the
// block text; cost per instruction = (cycles(R=6) - cycles(R=2)) / (4 x n x NIT), R = 4 midway).
#include "tim_common.h"
#include "left_core_blocks.h"

#include <map>

#ifndef W64
#define W64 0
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

// T3 (fix round): the same timed region, with every input register of the FP64 and pseudo-scalar blocks set first
// (inside the statement, before the first counter read): v16-v31 and v48-v49 to the FP64 value (lo, hi words),
// s16-s31 and s40 to sv (FP32 bits, or FP16 bits in the low half). After the second counter read v[16:17] and s16
// are copied out, so the host can check that the data reached the instructions.
#define LEFT_DINIT "v_mov_b32 v16, %[vlo]\n" "v_mov_b32 v17, %[vhi]\n" "v_mov_b32 v18, %[vlo]\n" "v_mov_b32 v19, %[vhi]\n" "v_mov_b32 v20, %[vlo]\n" "v_mov_b32 v21, %[vhi]\n" "v_mov_b32 v22, %[vlo]\n" "v_mov_b32 v23, %[vhi]\n" "v_mov_b32 v24, %[vlo]\n" "v_mov_b32 v25, %[vhi]\n" "v_mov_b32 v26, %[vlo]\n" "v_mov_b32 v27, %[vhi]\n" "v_mov_b32 v28, %[vlo]\n" "v_mov_b32 v29, %[vhi]\n" "v_mov_b32 v30, %[vlo]\n" "v_mov_b32 v31, %[vhi]\n" "v_mov_b32 v48, %[vlo]\n" "v_mov_b32 v49, %[vhi]\n"  "s_mov_b32 s16, %[sv]\n" "s_mov_b32 s17, %[sv]\n" "s_mov_b32 s18, %[sv]\n" "s_mov_b32 s19, %[sv]\n" "s_mov_b32 s20, %[sv]\n" "s_mov_b32 s21, %[sv]\n" "s_mov_b32 s22, %[sv]\n" "s_mov_b32 s23, %[sv]\n" "s_mov_b32 s24, %[sv]\n" "s_mov_b32 s25, %[sv]\n" "s_mov_b32 s26, %[sv]\n" "s_mov_b32 s27, %[sv]\n" "s_mov_b32 s28, %[sv]\n" "s_mov_b32 s29, %[sv]\n" "s_mov_b32 s30, %[sv]\n" "s_mov_b32 s31, %[sv]\n" "s_mov_b32 s40, %[sv]\n" 
#define LEFT_KD(NAME, TEXT, R)                                                                                       \
    __global__ __launch_bounds__(64) void k_##NAME##_##R(uint32_t n_iter, uint32_t * out, uint32_t vlo, uint32_t vhi, \
                                                         uint32_t sv) {                                               \
        for (int rep = 0; rep < REPS; ++rep) {                                                                       \
            uint32_t t0, t1, cnt, r0, r1, r2;                                                                        \
            asm volatile(LEFT_DINIT "s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n" \
                         ".rept " #R "\n" TEXT ".endr\n"                                                             \
                         " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"           \
                         " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                     \
                         " v_mov_b32 %[r0], v16\n v_mov_b32 %[r1], v17\n s_mov_b32 %[r2], s16\n"                   \
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), [r0] "=&v"(r0), [r1] "=&v"(r1),          \
                           [r2] "=&s"(r2)                                                                            \
                         : [n] "s"(n_iter), [vlo] "s"(vlo), [vhi] "s"(vhi), [sv] "s"(sv)                             \
                         : LEFT_CLOBBERS);                                                                           \
            if (threadIdx.x == 0) {                                                                                  \
                out[2 * rep] = t0;                                                                                   \
                out[2 * rep + 1] = t1;                                                                               \
                out[2 * REPS] = r0;                                                                                  \
                out[2 * REPS + 1] = r1;                                                                              \
                out[2 * REPS + 2] = r2;                                                                              \
            }                                                                                                        \
        }                                                                                                            \
    }
#define LEFT_KDS(NAME, N, TEXT) LEFT_KD(NAME, TEXT, 2) LEFT_KD(NAME, TEXT, 4) LEFT_KD(NAME, TEXT, 6)

using Kern = void (*)(uint32_t, uint32_t *);
struct Block {
    const char * name;
    int n;
    Kern k[3];
};
#define LEFT_ROW(NAME, N, TEXT) {#NAME, N, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6}},

LEFT_T_BLOCKS(LEFT_KS)
const Block tblocks[] = {LEFT_T_BLOCKS(LEFT_ROW)};
#if !W64
LEFT_D_BLOCKS(LEFT_KDS)
using KernD = void (*)(uint32_t, uint32_t *, uint32_t, uint32_t, uint32_t);
struct BlockD {
    const char * name;
    int n;
    KernD k[3];
};
#define LEFT_ROWD(NAME, N, TEXT) {#NAME, N, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6}},
const BlockD dblocks[] = {LEFT_D_BLOCKS(LEFT_ROWD)};
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

int mode_bank(bool brk) {
    int nonlin = 0;
    auto got = run_all(tblocks, sizeof tblocks / sizeof tblocks[0], "bank", nullptr, nonlin);
    const double want = brk ? 2.0 : 1.0, vv = W64 ? 1.25 : 1.0;
    tim::check("bank_calib", near(got["t_fma_ic"].per, want, 0.02) && near(got["t_dep"].per, 5.0, 0.05) &&
                                 near(got["t_fma_vv"].per, vv, 0.02) && nonlin == 0,
               "wave%d: independent FMA %.4f (expect %.2f%s), dependent FMA %.4f (expect 5.00), 8 of 16 with two VGPR "
               "sources in one bank %.4f (expect %.3f: SC-10, run 20261117T201152Z); %d blocks off the midpoint",
               WS, got["t_fma_ic"].per, want, brk ? ", deliberate break" : "", got["t_dep"].per, got["t_fma_vv"].per, vv,
               nonlin);
    // the per-instruction reading of SC-10 ("two VGPR sources in one bank cost 1.5 cycles in wave64") predicts 1.5 for
    // every all-16 block in wave64 and 1.0 for every block in wave32
    std::printf("VERDICT bank_all: INFO wave%d cycles per instruction, all 16 instructions with two VGPR sources in "
                "one bank: three sources: constants in one bank %.3f, chain and constant %.3f, v_fmac %.3f; two sources: "
                "FMA with an inline constant %.3f, VOP2 add %.3f; one register read twice %.3f; banks rotating: chain "
                "and constant %.3f, the two constants %.3f; every write to bank 0 (sources in banks 1-3) %.3f; every "
                "read from bank 0 (writes to banks 1-3) %.3f\n",
                WS, got["t_fma_kk_same"].per, got["t_fma_vv_all"].per, got["t_fmac_all"].per, got["t_fma_v1_all"].per,
                got["t_add_vv_all"].per, got["t_fma_xx"].per, got["t_fma_cc_rot"].per, got["t_fma_kk_rot"].per,
                got["t_wr_b0"].per, got["t_rd_b0"].per);
    return 0;
}

#if !W64
// ---- T3 (fix round): EC-2's FP64 and pseudo-scalar rows with known input data -------------------------------------
struct DataSet {
    const char * name;
    uint64_t f64;
    uint32_t f32;
    uint16_t f16;
};
const DataSet kData[] = {
    {"finite_1.7", 0x3FFB333333333333ull, 0x3FD9999Au, 0x3ECDu}, {"one", 0x3FF0000000000000ull, 0x3F800000u, 0x3C00u},
    {"zero", 0, 0, 0}, {"inf", 0x7FF0000000000000ull, 0x7F800000u, 0x7C00u},
    {"subnormal_min", 1ull, 1u, 1u}, {"qnan", 0x7FF8000000000000ull, 0x7FC00000u, 0x7E00u}};

float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3FFu;
    float v;
    if (e == 0) {
        v = std::ldexp((float) m, -24);
    } else if (e == 31) {
        v = m ? NAN : INFINITY;
    } else {
        v = std::ldexp((float) (m | 0x400u), (int) e - 25);
    }
    return s ? -v : v;
}

// the value an op gives for input x (CPU reference; rcp, rsq, log, exp are approximations on the card)
double ref_op(const std::string & op, double x) {
    if (op == "sqrt") return std::sqrt(x);
    if (op == "rcp") return 1.0 / x;
    if (op == "rsq") return 1.0 / std::sqrt(x);
    if (op == "log") return std::log2(x);
    return std::exp2(x);
}

// 0: matches (special values exactly, finite ones within rel), 1: differs
bool close_to(double got, double want, double rel) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want) || want == 0) return got == want;
    return std::fabs(got / want - 1) <= rel;
}

int mode_insdata(bool brk) {
    uint32_t * d;
    CHECK(hipMalloc(&d, (2 * REPS + 3) * 4));
    std::map<std::string, std::map<std::string, Cost>> got;  // block -> data -> cost
    int nonlin = 0, checked = 0, wrong = 0;
    std::string wrong_txt;
    for (const BlockD & b : dblocks) {
        const std::string nm = b.name;
        const bool f16 = nm.find("_f16_") != std::string::npos, f64 = nm.find("_f64_") != std::string::npos;
        const bool c16 = nm.size() > 4 && nm.compare(nm.size() - 4, 4, "_c16") == 0;
        for (const DataSet & ds : kData) {
            const uint32_t sv = f16 ? ds.f16 : ds.f32;
            tim::Clean m[3];
            Cost c{};
            uint32_t res[3] = {};
            for (int i = 0; i < 3; ++i) {
                std::vector<double> v;
                for (int l = 0; l < LAUNCHES; ++l) {
                    hipLaunchKernelGGL(b.k[i], dim3(1), dim3(32), 0, 0, NIT, d, (uint32_t) ds.f64,
                                       (uint32_t) (ds.f64 >> 32), sv);
                    CHECK(hipGetLastError());
                    CHECK(hipDeviceSynchronize());
                    const std::vector<uint32_t> h = tim::host_u32(d, 2 * REPS + 3);
                    for (int rep = 1; rep < REPS; ++rep) v.push_back((double) (h[2 * rep + 1] - h[2 * rep]));
                    res[0] = h[2 * REPS];
                    res[1] = h[2 * REPS + 1];
                    res[2] = h[2 * REPS + 2];
                }
                m[i] = tim::clean_of(v);
                c.n += m[i].n;
                c.dropped += m[i].dropped;
            }
            const double dd = 4.0 * b.n * NIT;
            c.per = (m[2].med - m[0].med) / dd;
            c.lo = (m[2].lo - m[0].hi) / dd;
            c.hi = (m[2].hi - m[0].lo) / dd;
            c.mid = (m[1].med - 0.5 * (m[0].med + m[2].med)) / NIT;
            nonlin += std::fabs(c.mid) > 1.5;
            got[nm][ds.name] = c;
            // what the instruction left: v[16:17] (FP64 blocks) or s16 (pseudo-scalar), against the CPU for the inputs
            // that stay fixed (dfx blocks, c16 blocks); chains are printed only
            double out = 0;
            if (f64) {
                uint64_t u = ((uint64_t) res[1] << 32) | res[0];
                std::memcpy(&out, &u, 8);
            } else if (f16) {
                out = f16_to_f32((uint16_t) res[2]);
            } else {
                float f;
                std::memcpy(&f, &res[2], 4);
                out = f;
            }
            double in = 0;
            if (f64) {
                std::memcpy(&in, &ds.f64, 8);
            } else if (f16) {
                in = f16_to_f32(ds.f16);
            } else {
                float f;
                std::memcpy(&f, &ds.f32, 4);
                in = f;
            }
            const std::string op = nm.find("sqrt") != std::string::npos ? "sqrt"
                                   : nm.find("rsq") != std::string::npos ? "rsq"
                                   : nm.find("rcp") != std::string::npos ? "rcp"
                                   : nm.find("log") != std::string::npos ? "log"
                                                                         : "exp";
            std::string judged = "chain: not judged";
            const bool fixed_in = nm.rfind("dat_dfx_", 0) == 0 || c16;
            if (fixed_in && std::strcmp(ds.name, "subnormal_min") != 0) {
                const double want = ref_op(op, in);
                // v_sqrt_f64 is not correctly rounded on this card (a first run: sqrt(1.0) did not return exactly
                // 1.0), so every op is judged within 1e-5 (FP16 2e-3); the result bits are printed
                const bool ok = close_to(out, want, f16 ? 2e-3 : 1e-5);
                ++checked;
                wrong += !ok;
                if (!ok) wrong_txt += " " + nm + "/" + ds.name;
                judged = ok ? "right" : "WRONG";
            } else if (fixed_in) {
                judged = "subnormal input: reported";
            }
            std::printf("RESULT insdata %-26s data %-13s cycles_per_instr %.4f range %.4f-%.4f midpoint_dev_per_it %.2f "
                        "samples %zu dropped %zu input %.6g result %.17g (bits %08x%08x / s16 %08x) (%s)\n",
                        b.name, ds.name, c.per, c.lo, c.hi, c.mid, c.n, c.dropped, in, out, res[1], res[0], res[2],
                        judged.c_str());
            std::fflush(stdout);
        }
    }
    const double want8 = brk ? 16.0 : 32.0;
    const Cost & r8 = got["dat_ins_rcp_f64_c8"]["finite_1.7"];
    const Cost & r1 = got["dat_ins_rcp_f64_c1"]["finite_1.7"];
    tim::check("insdata_calib",
               near(r8.per, want8, 0.1) && near(r1.per, 40.0, 0.2) && nonlin == 0 && checked > 0 && wrong == 0,
               "v_rcp_f64 with input 1.7: independent %.3f (expect %.1f%s), chained %.3f (expect 40.0; EC-2, run "
               "20261117T204427Z); %d blocks off the midpoint; results of fixed inputs against the CPU: %d checked, %d "
               "wrong%s",
               r8.per, want8, brk ? ", deliberate break" : "", r1.per, nonlin, checked, wrong, wrong_txt.c_str());
    // does the cost depend on the data? the largest difference between data sets, per block
    std::string txt;
    int differ = 0;
    for (auto & kv : got) {
        double lo = 1e9, hi = -1e9;
        std::string lon, hin;
        for (auto & dv : kv.second) {
            if (dv.second.per < lo) lo = dv.second.per, lon = dv.first;
            if (dv.second.per > hi) hi = dv.second.per, hin = dv.first;
        }
        if (hi - lo > 0.05) {
            ++differ;
            char b[160];
            std::snprintf(b, sizeof b, " %s %.3f (%s) to %.3f (%s);", kv.first.c_str() + 4, lo, lon.c_str(), hi,
                          hin.c_str());
            txt += b;
        }
    }
    std::printf("VERDICT insdata_costs: INFO blocks whose cost differs by more than 0.05 cycles between data sets: %d of "
                "%zu;%s\n",
                differ, got.size(), txt.c_str());
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
    if (mode == "bank") {
        mode_bank(brk);
#if !W64
    } else if (mode == "insdata") {
        mode_insdata(brk);
#endif
    } else {
        std::fprintf(stderr, "usage: left_bank <bank%s> [--break]\n", W64 ? "" : "|insdata");
        return 2;
    }
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
