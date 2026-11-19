// Timing leftovers, item 3: the execution-core oddities (measurements/execution-core.md "not explained",
// compiler-and-hazards.md section 3), each varied in isolation.
//
// Usage: tim_core <single|waves> [--break]      (built with -mcumode: a work-group stays on one CU)
//   single  one wave; every block of tim_core_blocks.h (16 instructions, fixed registers) timed with lane 1's
//           differential method: loops of R = 2, 4, 6 blocks per iteration, cost per instruction =
//           (cycles(R=6) - cycles(R=2)) / (4 blocks x 16 x iterations); R = 4 must sit midway. The same at R = 24 and
//           28 (a loop body 12 to 14 times longer: 3.5 to 5.3 KB of code) shows any dependence on code size.
//   waves   1, 2, 4, 5, 6, 8 waves on each SIMD of one CU (2 W waves per group): cycles per instruction of the
//           SIMD from the span of its waves (first start to last end, waves released together by a barrier).
//   size    raw cycles per instruction against the loop body's size (R = 2 to 64 blocks per iteration) for 4-, 8- and
//           12-byte instructions: does the cost follow the code size, and from which size on?
//   --break claims the independent FMA costs 2 cycles: the calibration verdict must FAIL.
// Disturbed samples (lane 7 rule) are dropped before the medians; the counts are printed.
#include "tim_common.h"
#include "tim_core_blocks.h"

#include <map>

namespace {

constexpr int REPS = 6;  // per launch; repetition 0 warms the instruction cache and is dropped
constexpr int LAUNCHES = 3;
constexpr uint32_t NIT = 16;
constexpr int MAXW = 32;

#define TIM_CORE_CLOBBERS                                                                                            \
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", \
        "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46",    \
        "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61",    \
        "v62", "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28",    \
        "s29", "s30", "s31", "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39", "s40", "s41", "s42", "s43",    \
        "s44", "s45", "s46", "s47", "vcc", "scc", "memory"

// out: per wave and repetition [t0, t1]; then per wave HW_ID1.
#define TIM_CORE_K(NAME, TEXT, R)                                                                                    \
    __global__ __launch_bounds__(1024) void k_##NAME##_##R(uint32_t n_iter, uint32_t * out) {                       \
        const int w = threadIdx.x / 32;                                                                              \
        uint32_t hw;                                                                                                 \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));                                           \
        for (int rep = 0; rep < REPS; ++rep) {                                                                       \
            __syncthreads();                                                                                         \
            uint32_t t0, t1, cnt;                                                                                    \
            asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n"       \
                         ".rept " #R "\n" TEXT ".endr\n"                                                             \
                         " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"           \
                         " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                     \
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt)                                         \
                         : [n] "s"(n_iter)                                                                           \
                         : TIM_CORE_CLOBBERS);                                                                       \
            if (threadIdx.x % 32 == 0) {                                                                             \
                out[2 * (w * REPS + rep)] = t0;                                                                      \
                out[2 * (w * REPS + rep) + 1] = t1;                                                                  \
            }                                                                                                        \
        }                                                                                                            \
        if (threadIdx.x % 32 == 0) {                                                                                 \
            out[2 * MAXW * REPS + w] = hw;                                                                           \
        }                                                                                                            \
    }

#define TIM_CORE_KS(NAME, TEXT) \
    TIM_CORE_K(NAME, TEXT, 2) TIM_CORE_K(NAME, TEXT, 4) TIM_CORE_K(NAME, TEXT, 6) TIM_CORE_K(NAME, TEXT, 24) \
        TIM_CORE_K(NAME, TEXT, 28)
TIM_CORE_BLOCKS(TIM_CORE_KS)

// Code-size sweep: the same block at R = 2 to 64 per loop iteration (loop body 128 B to 12 KB).
#define TIM_SIZE_K(NAME, TEXT, R) TIM_CORE_K(z_##NAME, TEXT, R)
#define TIM_SIZE_KS(NAME, TEXT)                                                                                      \
    TIM_SIZE_K(NAME, TEXT, 2) TIM_SIZE_K(NAME, TEXT, 8) TIM_SIZE_K(NAME, TEXT, 16) TIM_SIZE_K(NAME, TEXT, 20)        \
        TIM_SIZE_K(NAME, TEXT, 24) TIM_SIZE_K(NAME, TEXT, 28) TIM_SIZE_K(NAME, TEXT, 32) TIM_SIZE_K(NAME, TEXT, 40)  \
            TIM_SIZE_K(NAME, TEXT, 48) TIM_SIZE_K(NAME, TEXT, 64)
TIM_SIZE_BLOCKS(TIM_SIZE_KS)
constexpr int NSIZE = 10;
constexpr int SIZES[NSIZE] = {2, 8, 16, 20, 24, 28, 32, 40, 48, 64};

using Kern = void (*)(uint32_t, uint32_t *);
struct SizeRow {
    const char * name;
    int bytes;  // per instruction
    Kern k[NSIZE];
};
#define TIM_SIZE_ROW(NAME, TEXT)                                                                                     \
    {#NAME, 0, {k_z_##NAME##_2, k_z_##NAME##_8, k_z_##NAME##_16, k_z_##NAME##_20, k_z_##NAME##_24, k_z_##NAME##_28, \
                k_z_##NAME##_32, k_z_##NAME##_40, k_z_##NAME##_48, k_z_##NAME##_64}},
SizeRow size_rows[] = {TIM_SIZE_BLOCKS(TIM_SIZE_ROW)};
struct Block {
    const char * name;
    Kern k[5];  // R = 2, 4, 6, 24, 28
};
#define TIM_CORE_ROW(NAME, TEXT) {#NAME, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6, k_##NAME##_24, k_##NAME##_28}},
const Block blocks[] = {TIM_CORE_BLOCKS(TIM_CORE_ROW)};

const Block & find(const std::string & n) {
    for (const Block & b : blocks) {
        if (n == b.name) {
            return b;
        }
    }
    std::fprintf(stderr, "no block %s\n", n.c_str());
    std::exit(2);
}

uint32_t * g_out = nullptr;

// Span samples (cycles) of the SIMD holding the most waves; nsimd: that SIMD's wave count.
std::vector<double> spans(Kern k, int waves_per_group, int & nsimd, size_t & dropped_total) {
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32 * waves_per_group), 0, 0, NIT, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 2 * MAXW * REPS + MAXW);
        std::map<uint32_t, std::vector<int>> by;
        for (int w = 0; w < waves_per_group; ++w) {
            const uint32_t hw = h[2 * MAXW * REPS + w];
            by[(hw >> 8) & 3].push_back(w);  // SIMD within the WGP (CU and SIMD-in-CU bits)
        }
        const std::vector<int> * best = nullptr;
        for (auto & kv : by) {
            if (!best || kv.second.size() > best->size()) {
                best = &kv.second;
            }
        }
        nsimd = (int) best->size();
        for (int rep = 1; rep < REPS; ++rep) {
            const uint32_t base = h[2 * ((*best)[0] * REPS + rep)];
            uint32_t lo = 0xFFFFFFFFu, hi = 0;
            for (int w : *best) {
                lo = std::min(lo, h[2 * (w * REPS + rep)] - base + 0x80000000u);
                hi = std::max(hi, h[2 * (w * REPS + rep) + 1] - base + 0x80000000u);
            }
            v.push_back((double) (hi - lo));
        }
    }
    (void) dropped_total;
    return v;
}

struct Cost {
    double per, big, mid;
    size_t n, dropped;
    int nsimd;
};

// Cost per instruction of the SIMD (one wave: of the wave) from R = 2, 4, 6 (and 24, 28 when big).
Cost cost(const Block & b, int waves_per_group, bool big) {
    Cost c{};
    tim::Clean m[5];
    const int rs[5] = {2, 4, 6, 24, 28};
    for (int i = 0; i < 5; ++i) {
        if (i >= 3 && !big) {
            continue;
        }
        size_t d = 0;
        m[i] = tim::clean_of(spans(b.k[i], waves_per_group, c.nsimd, d));
        c.n += m[i].n;
        c.dropped += m[i].dropped;
    }
    const double per_block_it = 16.0 * NIT * c.nsimd;
    c.per = (m[2].med - m[0].med) / ((rs[2] - rs[0]) * per_block_it);
    c.mid = (m[1].med - 0.5 * (m[0].med + m[2].med)) / NIT;
    c.big = big ? (m[4].med - m[3].med) / ((rs[4] - rs[3]) * per_block_it) : NAN;
    return c;
}

int mode_single(bool brk) {
    std::map<std::string, double> got;
    size_t n = 0, dropped = 0, nonlin = 0;
    for (const Block & b : blocks) {
        const Cost c = cost(b, 1, true);
        got[b.name] = c.per;
        n += c.n;
        dropped += c.dropped;
        nonlin += std::fabs(c.mid) > 1.5;
        std::printf("RESULT core %-14s cycles_per_instr %.4f big_loop %.4f midpoint_dev_per_it %.2f samples %zu dropped %zu\n",
                    b.name, c.per, c.big, c.mid, c.n, c.dropped);
        std::fflush(stdout);
    }
    std::printf("RESULT core disturbance rule: %zu of %zu samples dropped; %zu blocks off the midpoint by > 1.5 cycles per "
                "iteration\n",
                dropped, n, nonlin);
    auto near = [&](std::initializer_list<const char *> names, double lo, double hi, double & mn, double & mx) {
        mn = 1e9;
        mx = -1e9;
        for (const char * nm : names) {
            mn = std::min(mn, got[nm]);
            mx = std::max(mx, got[nm]);
        }
        return mn >= lo && mx <= hi;
    };
    double a0, a1, b0, b1, c0, c1;
    // Instruction length: every 12-byte instruction back to back costs the same, VOP3 or VOPD, literal or not;
    // 8-byte instructions with a literal cost 1.00, and so does a 12-byte one between 8-byte ones.
    const bool l12 = near({"fma_lit", "fma_lit_2v", "fma_lit_e64mov", "du_fmaak", "du_fmaak_ic", "du_fmaak_mov",
                           "du_mov_reallit"},
                          1.20, 1.30, a0, a1) &&
                     a1 - a0 < 0.01;
    const bool l8 = near({"mov_lit", "add_lit", "fmaak_vop2", "du_mov_lit", "add_e64", "du_fmac"}, 0.99, 1.01, b0, b1);
    tim::check("tim_core_length", l12 && l8 && got["fma_lit_1in4"] < 1.02,
               "back-to-back 12-byte instructions (7 kinds: VOP3 and VOPD, with and without a VGPR beside the literal) "
               "%.4f-%.4f cycles; 8-byte ones with or without a literal %.4f-%.4f; one 12-byte in four %.4f",
               a0, a1, b0, b1, got["fma_lit_1in4"]);
    // SGPR read banks, model: bank = (register >> 1) mod 8 (64-bit pairs, 16 registers per round); two scalar sources
    // in one bank cost one extra cycle, the same register twice included. Predicted per block from its registers.
    struct Sb {
        const char * name;
        int a, b;  // source registers; -1: the chain register s16..s31 itself
    };
    const Sb sb[] = {{"s2_40_41", 40, 41}, {"s2_40_42", 40, 42}, {"s2_40_44", 40, 44}, {"s2_40_48", 40, 48},
                     {"s2_40_56", 40, 56}, {"s2_40_40", 40, 40}, {"s2_41_43", 41, 43}, {"s_add_sreg", -1, 40},
                     {"s_add_s42", -1, 42}, {"s_add_s44", -1, 44}, {"s_and_sreg", -1, 40}};
    double worst = 0;
    std::string pred;
    for (const Sb & x : sb) {
        int conflicts = 0;
        for (int r = 16; r < 32; ++r) {
            const int a = x.a < 0 ? r : x.a;
            conflicts += ((a >> 1) & 7) == ((x.b >> 1) & 7);
        }
        const double p = 1.0 + conflicts / 16.0;
        worst = std::max(worst, std::fabs(got[x.name] - p));
        char buf[64];
        std::snprintf(buf, sizeof buf, " %s %.3f/%.3f", x.name, got[x.name], p);
        pred += buf;
    }
    // the same banks for a vector instruction reading two SGPRs
    const double vp = std::max(std::fabs(got["fma_ss"] - 2.0), std::fabs(got["fma_ss2"] - 1.0));
    tim::check("tim_core_sgpr_banks", worst < 0.03 && vp < 0.03,
               "scalar adds, measured/predicted by 8 banks of SGPR pairs:%s; worst deviation %.3f; v_fma_f32 with s40, s41 "
               "%.3f (predicted 2), with s40, s42 %.3f (predicted 1)",
               pred.c_str(), worst, got["fma_ss"], got["fma_ss2"]);
    // A VALU instruction that writes an SGPR costs about 2 cycles when the next one also writes an SGPR, 1 when any
    // other instruction comes between; a scalar instruction right after it waits for the write.
    const bool vs = near({"ps_exp", "ps_exp_ic", "ps_exp2", "readlane"}, 1.80, 1.95, c0, c1) &&
                    got["ps_exp_fma"] < 2.02 && got["ps_exp_snop"] < 2.02 && got["ps_exp_sadd"] > 20;
    tim::check("tim_core_valu_sgpr", vs,
               "back to back: v_s_exp %.3f (inline source %.3f), v_readfirstlane %.3f, v_cmp to SGPR %.3f; with an FMA "
               "between, per pair %.3f, with s_nop %.3f; v_s_exp then s_add per pair %.2f",
               got["ps_exp"], got["ps_exp_ic"], got["readlane"], got["cmp_sgpr"], got["ps_exp_fma"], got["ps_exp_snop"],
               got["ps_exp_sadd"]);
    tim::check("tim_core_waw", near({"waw2", "waw4", "waw8", "waw16", "waw4_add"}, 0.99, 1.01, c0, c1),
               "2, 4, 8, 16 back-to-back writes of one register with independent sources: %.4f-%.4f cycles", c0, c1);
    const double want = brk ? 2.0 : 1.0;
    tim::check("tim_core_calib", std::fabs(got["fma_ic"] - want) < 0.02 && std::fabs(got["add_vop2"] - want) < 0.02,
               "independent FMA (inline constants) %.4f, VOP2 add %.4f cycles; expected %.2f%s", got["fma_ic"],
               got["add_vop2"], want, brk ? " (deliberate break)" : "");
    return 0;
}

// Finish order of the waves on the busiest SIMD in one launch (R = 6 kernel, repetition 1): each wave's end, from
// the first start, in units of one wave's work alone (`alone` cycles). Oldest-first issue with s_delay_alu predicts
// waves finishing in groups of five (the FMA latency), not together.
void finish_order(const Block & b, int waves_per_group, double alone) {
    hipLaunchKernelGGL(b.k[2], dim3(1), dim3(32 * waves_per_group), 0, 0, NIT, g_out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> h = tim::host_u32(g_out, 2 * MAXW * REPS + MAXW);
    std::map<uint32_t, std::vector<int>> by;
    for (int w = 0; w < waves_per_group; ++w) {
        by[(h[2 * MAXW * REPS + w] >> 8) & 3].push_back(w);
    }
    const std::vector<int> & v = by.begin()->second;
    const int rep = 1;
    const uint32_t base = h[2 * (v[0] * REPS + rep)];
    uint32_t lo = 0xFFFFFFFFu;
    for (int w : v) {
        lo = std::min(lo, h[2 * (w * REPS + rep)] - base + 0x80000000u);
    }
    std::vector<double> ends;
    for (int w : v) {
        ends.push_back((double) (h[2 * (w * REPS + rep) + 1] - base + 0x80000000u - lo) / alone);
    }
    std::sort(ends.begin(), ends.end());
    std::printf("RESULT finish %-8s waves_on_simd %zu ends_in_units_of_one_wave_alone", b.name, v.size());
    for (double e : ends) {
        std::printf(" %.2f", e);
    }
    std::printf("\n");
}

int mode_waves() {
    const char * names[] = {"fma_ic",   "dep",     "dep_da",     "dep2_da",    "ind_da",  "dep_add_da", "dep_da_nop",
                            "fma_vv",   "pl_good", "pl_twobank", "pl_onebank", "pl_fmac", "du_pl_good", "du_pl_bad"};
    std::map<std::string, std::map<int, double>> got;
    for (const char * nm : names) {
        const Block & b = find(nm);
        for (int w : {1, 2, 4, 5, 6, 8}) {
            const Cost c = cost(b, 2 * w, false);
            got[nm][w] = c.per;
            std::printf("RESULT waves %-11s waves_per_simd %d cycles_per_instr_simd %.4f instr_per_cycle %.3f "
                        "midpoint_dev_per_it %.2f samples %zu dropped %zu\n",
                        nm, c.nsimd, c.per, 1.0 / c.per, c.mid, c.n, c.dropped);
            std::fflush(stdout);
        }
    }
    // one wave of dep_da alone at R = 6: 6 blocks x 16 x NIT instructions at 5 cycles
    for (int w : {4, 5, 6, 8}) {
        finish_order(find("dep_da"), 2 * w, 6.0 * 16 * NIT * 5.0);
    }
    // the model: while 5 or more waves are active the SIMD issues one FMA per cycle; the oldest five run first, and
    // the remainder runs as a tail at (waves left) / 5. Predicted cycles per instruction of the SIMD for n waves:
    auto model = [](int n) {
        double t = 0;
        for (int left = n; left > 0; left -= std::min(left, 5)) {
            t += 5.0;  // each group of up to five waves takes one wave's time alone (5 cycles per instruction)
        }
        return t / n;
    };
    double worst = 0;
    for (int w : {1, 2, 4, 5, 6, 8}) {
        worst = std::max(worst, std::fabs(got["dep_da"][w] - model(w)) / model(w));
    }
    tim::check("tim_core_delay_tail", worst < 0.03,
               "dep_da against the oldest-first tail model: 1, 2, 4, 5, 6, 8 waves measured %.3f %.3f %.3f %.3f %.3f %.3f, "
               "model %.3f %.3f %.3f %.3f %.3f %.3f cycles per FMA of the SIMD; worst deviation %.1f %% (limit 3 %%)",
               got["dep_da"][1], got["dep_da"][2], got["dep_da"][4], got["dep_da"][5], got["dep_da"][6], got["dep_da"][8],
               model(1), model(2), model(4), model(5), model(6), model(8), 100 * worst);
    // VOPD FMAC pairs with per-lane sources: three different registers of one bank across the pair cost 2.5 cycles,
    // six registers spread over the banks 1.5, and other waves on the SIMD do not fill those cycles.
    tim::check("tim_core_vopd_banks",
               std::fabs(got["du_pl_bad"][1] - 2.5) < 0.05 && std::fabs(got["du_pl_bad"][8] - 2.5) < 0.05 &&
                   std::fabs(got["du_pl_good"][1] - 1.5) < 0.05 && std::fabs(got["du_pl_good"][8] - 1.5) < 0.05 &&
                   std::fabs(got["pl_onebank"][8] - 5.0) < 0.05,
               "VOPD FMAC pair with a bank triple %.3f (1 wave) / %.3f (8 waves) cycles per pair; without %.3f / %.3f; "
               "single FMA with a bank triple %.3f / %.3f",
               got["du_pl_bad"][1], got["du_pl_bad"][8], got["du_pl_good"][1], got["du_pl_good"][8],
               got["pl_onebank"][1], got["pl_onebank"][8]);
    tim::check("tim_core_waves_calib", std::fabs(got["fma_ic"][1] - 1.0) < 0.02 && std::fabs(got["fma_ic"][8] - 1.0) < 0.02,
               "independent FMA: %.4f cycles per instruction with 1 wave, %.4f with 8 waves on the SIMD; expected 1.00",
               got["fma_ic"][1], got["fma_ic"][8]);
    return 0;
}

// Raw cycles per instruction (loop control included, so small R reads a little high) against the loop body's size.
// The bytes per instruction come from the code object (the script reads them from the disassembly and passes them
// as "name=bytes" arguments); without them the size column is left out.
int mode_size(int argc, char ** argv) {
    for (SizeRow & r : size_rows) {
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            const size_t eq = a.find('=');
            if (eq != std::string::npos && a.substr(0, eq) == r.name) {
                r.bytes = std::atoi(a.c_str() + eq + 1);
            }
        }
        for (int i = 0; i < NSIZE; ++i) {
            int ns = 0;
            size_t d = 0;
            const tim::Clean c = tim::clean_of(spans(r.k[i], 1, ns, d));
            const double instr = 16.0 * SIZES[i] * NIT;
            std::printf("RESULT size %-9s R %2d body_bytes %6d cycles_per_instr %.4f range %.4f-%.4f dropped %zu of %zu\n",
                        r.name, SIZES[i], r.bytes * 16 * SIZES[i], c.med / instr, c.lo / instr, c.hi / instr, c.dropped, c.n);
        }
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = tim::flag(argc, argv, "--break");
    CHECK(hipMalloc(&g_out, (2 * MAXW * REPS + MAXW) * 4));
    mem::clocks("start");
    if (mode == "single") {
        mode_single(brk);
    } else if (mode == "waves") {
        mode_waves();
    } else if (mode == "size") {
        mode_size(argc, argv);
    } else {
        std::fprintf(stderr, "usage: tim_core <single|waves|size [name=bytes...]> [--break]\n");
        return 2;
    }
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
