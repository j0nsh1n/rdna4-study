// Lane 4, items 9 and 10: several waves on one SIMD running matrix chains; a matrix wave beside a vector-FMA wave or
// a memory-bound wave on the same SIMD; both SIMDs of one compute unit (CU) against SIMDs of different CUs.
//
// Placement (as in core_share.hip): one work-group of 1024 threads (32 waves). Built with -mcumode (CUMODE=1) it sits
// on one CU, 16 waves on each of its 2 SIMDs; built without (WGP mode) it spreads over the 4 SIMDs of a work-group
// processor, 8 waves each. Each wave reads HW_REG_HW_ID1 (SIMD_ID bits 9:8: bit 0 = CU within the WGP, bit 1 = SIMD
// within the CU) and takes a rank on its SIMD with an LDS atomic; waves whose rank is not below active[simd] end
// after one barrier. Every active wave runs the timed loop of its role, chosen by its rank.
//
// Roles (one inline-assembly block each, explicit registers; the per-role iteration count iters[role] is set so
// every role alone runs about as long as the matrix chain): 0 mm: 16 chained v_wmma_f32_16x16x16_f16; 1 fma: 16
// independent v_fma_f32 (8 registers, reuse distance 8); 2 mem: four 16-byte loads per lane from DRAM (each wave
// streams its own 128 MiB region of a 2 GiB buffer, see MEM_BYTES) then a wait; 3 mmx4: 16 f16 matrix
// instructions over 4 independent accumulators; 4 ff: 16 chained v_wmma_f32_16x16x16_fp8_fp8.
// Cost: U = 4 and 8 blocks per iteration, differences per role ("own" = cycles per operation the slowest wave of that role sees
// between its own counter reads). "SIMD" = span of all active waves on one SIMD (one SIMD's counter) per operation.
//
// Usage: mat_share [--break]. CU mode: matrix-only wave counts 1..16, mixed pairs. WGP mode: SIMD pairs.
// --break (CU mode) places one matrix wave on each of the two SIMDs of the CU but evaluates them as if both were
// on one SIMD (both waves' instructions over one SIMD's span); the one-matrix-unit check must FAIL.
#include "mat_common.h"

#ifndef CUMODE
#error "build with -DCUMODE=1 -mcumode or -DCUMODE=0"
#endif

constexpr int MAXW = 32;
constexpr int NROLE = 5;
// 128 MiB per wave and a start offset that moves 4 MiB per launch: a wave meets its data again only after 128 MiB
// of its own traffic, beyond the 64 MiB memory-side cache, so its loads come from DRAM.
constexpr size_t MEM_BYTES = size_t(2048) << 20;
constexpr size_t REGION = size_t(128) << 20;

__device__ __forceinline__ uint64_t realtime() {
    uint64_t t;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0" : "=s"(t));
    return t;
}

// Small register map (16 waves per SIMD allow 96 VGPRs): D0..D3 v[16:47], A v[48:51], B v[52:55], X v[56:63],
// K v[64:65], load offset v66, offset mask v67, buffer address s[44:45].
#define SH_MM(d) "v_wmma_f32_16x16x16_f16 " d ", v[48:51], v[52:55], " d "\n"
#define SH_FF(d) "v_wmma_f32_16x16x16_fp8_fp8 " d ", v[48:49], v[52:53], " d "\n"
#define SH_FMA8                                                                                                \
    "v_fma_f32 v56, v56, v64, v65\nv_fma_f32 v57, v57, v64, v65\nv_fma_f32 v58, v58, v64, v65\n"             \
    "v_fma_f32 v59, v59, v64, v65\nv_fma_f32 v60, v60, v64, v65\nv_fma_f32 v61, v61, v64, v65\n"             \
    "v_fma_f32 v62, v62, v64, v65\nv_fma_f32 v63, v63, v64, v65\n"
#define SH_LD(dst) "global_load_b128 " dst ", v66, s[44:45]\nv_add_nc_u32 v66, 0x200, v66\n"
#define BODY_MM MAT_R16(SH_MM("v[16:23]"))
#define BODY_FMA SH_FMA8 SH_FMA8
#define BODY_MEM SH_LD("v[56:59]") SH_LD("v[60:63]") SH_LD("v[56:59]") SH_LD("v[60:63]") \
    "v_and_b32 v66, v66, v67\ns_wait_loadcnt 0x0\n"
#define BODY_MMX4 MAT_R4(SH_MM("v[16:23]") SH_MM("v[24:31]") SH_MM("v[32:39]") SH_MM("v[40:47]"))
#define BODY_FF MAT_R16(SH_FF("v[16:23]"))

// operations per block: matrix instructions, FMAs, bytes read by the wave (4 loads x 32 lanes x 16 bytes)
constexpr double OPS[NROLE] = {16, 16, 2048, 16, 16};
const char * const ROLE_NAME[NROLE] = {"mm_f16_chain", "fma", "mem_dram", "mm_f16_x4", "mm_fp8_chain"};

// The whole timed region is one assembly statement: counter read, loop of n_it iterations of U blocks, counter
// read. So nothing the compiler adds (copies, waits, spills; measured with compiler-built loops at the 96-register
// limit of 16 waves per SIMD) can land inside it. The loop control is s_add_co_i32 / s_cmp_lg_u32 /
// s_cbranch_scc1 (cancels in the U = 8 minus U = 4 difference). The 32-bit low half of the counter is enough: a
// timed region lasts far less than 2^32 cycles.
#define SH_ASM(TEXT)                                                                                               \
    asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                            \
                 "s_mov_b32 %[cnt], %[n]\n"                                                                        \
                 "2:\n" TEXT                                                                                       \
                 "s_add_co_i32 %[cnt], %[cnt], -1\n"                                                               \
                 "s_cmp_lg_u32 %[cnt], 0\n"                                                                        \
                 "s_cbranch_scc1 2b\n"                                                                             \
                 "s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                            \
                 : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), "+{v[16:23]}"(d0), "+{v[24:31]}"(d1),         \
                   "+{v[32:39]}"(d2), "+{v[40:47]}"(d3), "+{v[48:51]}"(a), "+{v[52:55]}"(b), "+{v[56:63]}"(x),     \
                   "+{v[64:65]}"(k), "+{v66}"(off), "+{s[44:45]}"(pm), "+{v67}"(mask)                              \
                 : [n] "s"(n_it)                                                                                   \
                 : "vcc", "scc", "memory")

#define SH_LOOP(BODY)                                                                                             \
    for (int rep = 0; rep < core::REPS; ++rep) {                                                                  \
        /* the span starts at repetition 0: when waves run one after another, a span from repetition 1 would      \
           include the other waves' uncounted repetition 0 */                                                     \
        if (rep == 0) {                                                                                           \
            span_r0 = realtime();                                                                                 \
            span_c0 = wave_cycles();                                                                              \
        }                                                                                                         \
        uint32_t t0, t1, cnt;                                                                                     \
        if constexpr (U == 4) {                                                                                   \
            SH_ASM(MAT_R4(BODY));                                                                                 \
        } else {                                                                                                  \
            SH_ASM(MAT_R8(BODY));                                                                                 \
        }                                                                                                         \
        if (lane == 0) {                                                                                          \
            out[w * core::REPS + rep] = (uint32_t) (t1 - t0);                                                     \
        }                                                                                                         \
    }

#define ROLE_LOOP(R)                     \
    if constexpr (R == 0) {              \
        SH_LOOP(BODY_MM)                 \
    } else if constexpr (R == 1) {       \
        SH_LOOP(BODY_FMA)                \
    } else if constexpr (R == 2) {       \
        SH_LOOP(BODY_MEM)                \
    } else if constexpr (R == 3) {       \
        SH_LOOP(BODY_MMX4)               \
    } else {                             \
        SH_LOOP(BODY_FF)                 \
    }

template <int U, int RA, int RB>
__global__ __launch_bounds__(1024) void share(const uint32_t * __restrict__ in, uint64_t * __restrict__ out,
                                              uint32_t * __restrict__ info, const int * __restrict__ ctl,
                                              const uint8_t * mem) {
    __shared__ int cnt[4];
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x < 4) {
        cnt[threadIdx.x] = 0;
    }
    __syncthreads();
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    // CU mode: the group lands on either CU of the WGP, so select by SIMD within the CU (bit 1); WGP mode: full id
    const int simd = CUMODE ? ((hw >> 9) & 1) : ((hw >> 8) & 3);
    int rank = 0;
    if (lane == 0) {
        rank = atomicAdd(&cnt[simd], 1);
    }
    rank = __builtin_amdgcn_readfirstlane(rank);
    __syncthreads();
    // ctl: [0..3] active waves per SIMD key, [4] ranks running RA, [5] alternate by parity, [6..10] iterations per role,
    // [11] start offset of the memory role
    const bool on = rank < ctl[simd];
    const bool is_a = ctl[5] ? (rank & 1) == 0 : rank < ctl[4];
    const int role = is_a ? RA : RB;
    if (lane == 0) {
        info[w] = hw;
        info[MAXW + w] = rank;
        info[2 * MAXW + w] = on;
        info[3 * MAXW + w] = role;
    }
    if (!on) {
        return;
    }
    const int n_it = ctl[6 + role];
    mat_v8i d0 = mat::ld<mat_v8i>(in + mat::IN_D + lane * 8), d1 = d0, d2 = d0, d3 = d0;
    mat_v4i a = mat::ld<mat_v4i>(in + mat::IN_A + lane * 8), b = mat::ld<mat_v4i>(in + mat::IN_B + lane * 8);
    mat_v8i x = mat::ld<mat_v8i>(in + mat::IN_X);
    mat_v2i k = mat::ld<mat_v2i>(in + mat::IN_X + 8);
    int off = (lane * 16 + ctl[11]) & (int) (REGION - 1);
    int mask = (int) (REGION - 1);
    const uint8_t * pm = mem + (size_t) (__builtin_amdgcn_readfirstlane(w) % 16) * REGION;
    uint64_t span_c0 = 0, span_r0 = 0;
    // the role choice is uniform per wave and stays outside the timed loops
    if (is_a) {
        ROLE_LOOP(RA)
    } else {
        ROLE_LOOP(RB)
    }
    const uint64_t span_c1 = wave_cycles();
    const uint64_t span_r1 = realtime();
    if (lane == 0) {
        uint64_t * sp = out + MAXW * core::REPS + w * 4;
        sp[0] = span_c0;
        sp[1] = span_c1;
        sp[2] = span_r0;
        sp[3] = span_r1;
    }
    uint32_t h = 0;
    for (int i = 0; i < 8; ++i) {
        h = h * 31u + (uint32_t) d0[i] + (uint32_t) d1[i] + (uint32_t) d2[i] + (uint32_t) d3[i] + (uint32_t) x[i];
    }
    h += (uint32_t) a[0] + (uint32_t) b[0] + (uint32_t) k[0] + (uint32_t) off;
    if (h == 0x9e3779b9u) {
        info[4 * MAXW] = h;
    }
}

using SKern = void (*)(const uint32_t *, uint64_t *, uint32_t *, const int *, const uint8_t *);

struct Pair {
    int ra, rb;
    SKern lo, hi;
};
#define PAIR(A, B) Pair{A, B, share<4, A, B>, share<8, A, B>}
// role pairs used below (0 mm, 1 fma, 2 mem, 3 mmx4, 4 ff)
static const Pair PAIRS[] = {PAIR(0, 0), PAIR(1, 1), PAIR(2, 2), PAIR(3, 3), PAIR(4, 4), PAIR(0, 1),
                             PAIR(1, 0), PAIR(0, 2), PAIR(2, 0), PAIR(4, 1), PAIR(3, 1)};

static const Pair & pair_of(int ra, int rb) {
    for (const Pair & p : PAIRS) {
        if (p.ra == ra && p.rb == rb) {
            return p;
        }
    }
    std::fprintf(stderr, "no kernel for roles %d/%d\n", ra, rb);
    std::exit(2);
}

struct Launch {
    std::vector<uint64_t> c;
    std::vector<uint32_t> info;
};

struct Cfg {
    std::string name;
    int active[4];
    int ra, rb = -1;  // rb < 0: every wave runs ra
    int n_a = 16;     // ranks below n_a run ra
    bool alt = false;  // even ranks ra, odd ranks rb
    bool claim_one_simd = false;  // --break: count all active waves' work against one SIMD's span
    int role_of(int rank) const {
        if (rb < 0) {
            return ra;
        }
        return (alt ? (rank & 1) == 0 : rank < n_a) ? ra : rb;
    }
};

struct Res {
    double own[NROLE];    // cycles per op per wave of each role (0 if absent)
    double simd_cpo = 0;  // cycles per op on the busiest SIMD, single-role configs
    double simd_lo = 0, simd_hi = 0;
    double overlap = 0;
    int waves_on[4] = {0, 0, 0, 0};
    bool placed = true;
};

static int g_iters[NROLE];

static std::vector<Launch> run_cfg(SKern k, const Cfg & c, int * dctl, uint64_t * dout, uint32_t * dinfo,
                                   const uint8_t * dmem) {
    int ctl[16] = {0};
    for (int s = 0; s < 4; ++s) {
        ctl[s] = c.active[s];
    }
    ctl[4] = c.rb < 0 ? 16 : c.n_a;
    ctl[5] = c.alt;
    static int launches = 0;
    for (int r = 0; r < NROLE; ++r) {
        ctl[6 + r] = g_iters[r];
    }
    CHECK(hipMemcpy(dctl, ctl, sizeof ctl, hipMemcpyHostToDevice));
    constexpr int NOUT = MAXW * core::REPS + MAXW * 4;
    std::vector<Launch> v;
    for (int l = 0; l < core::LAUNCHES; ++l) {
        ctl[11] = (int) (((size_t) launches++ * (size_t(4) << 20)) % REGION);
        CHECK(hipMemcpy(dctl, ctl, sizeof ctl, hipMemcpyHostToDevice));
        CHECK(hipMemset(dout, 0, NOUT * 8));
        hipLaunchKernelGGL(k, dim3(1), dim3(1024), 0, 0, mat::buffers().in, dout, dinfo, dctl, dmem);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        Launch L;
        L.c.resize(NOUT);
        L.info.resize(4 * MAXW + 1);
        CHECK(hipMemcpy(L.c.data(), dout, NOUT * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(L.info.data(), dinfo, L.info.size() * 4, hipMemcpyDeviceToHost));
        v.push_back(L);
    }
    return v;
}

static int simd_key(uint32_t hw) { return CUMODE ? ((hw >> 9) & 1) : ((hw >> 8) & 3); }

// Per role: per repetition, the slowest wave of that role. Per launch: the largest per-SIMD span (each SIMD's span
// on its own counter: counters differ between SIMDs).
static void collect(const std::vector<Launch> & ls, const Cfg & c, std::vector<double> own[NROLE],
                    std::vector<double> & span, std::vector<double> & ovl, Res & res) {
    for (const Launch & L : ls) {
        uint64_t c0[4] = {~0ull, ~0ull, ~0ull, ~0ull}, c1[4] = {0, 0, 0, 0};
        uint64_t r0 = ~0ull, r1 = 0;
        double rsum = 0;
        int on[4] = {0, 0, 0, 0};
        for (int w = 0; w < MAXW; ++w) {
            if (!L.info[2 * MAXW + w]) {
                continue;
            }
            const int s = simd_key(L.info[w]);
            on[s]++;
            const uint64_t * sp = &L.c[MAXW * core::REPS + w * 4];
            c0[s] = std::min(c0[s], sp[0]);
            c1[s] = std::max(c1[s], sp[1]);
            r0 = std::min(r0, sp[2]);
            r1 = std::max(r1, sp[3]);
            rsum += (double) (sp[3] - sp[2]);
        }
        double sp_max = 0;
        for (int s = 0; s < 4; ++s) {
            if (c1[s]) {
                sp_max = std::max(sp_max, (double) (c1[s] - c0[s]));
            }
            res.placed = res.placed && on[s] == c.active[s];
            res.waves_on[s] = on[s];
        }
        span.push_back(sp_max);
        ovl.push_back(r1 > r0 ? rsum / (double) (r1 - r0) : 0);
        for (int r = 1; r < core::REPS; ++r) {
            double worst[NROLE] = {0};
            for (int w = 0; w < MAXW; ++w) {
                if (L.info[2 * MAXW + w]) {
                    const int role = (int) L.info[3 * MAXW + w];
                    worst[role] = std::max(worst[role], (double) L.c[w * core::REPS + r]);
                }
            }
            for (int ro = 0; ro < NROLE; ++ro) {
                if (worst[ro] > 0) {
                    own[ro].push_back(worst[ro]);
                }
            }
        }
    }
}

static Res measure_cfg(const Cfg & c, int * dctl, uint64_t * dout, uint32_t * dinfo, const uint8_t * dmem) {
    Res res;
    const Pair & p = pair_of(c.ra, c.rb < 0 ? c.ra : c.rb);
    std::vector<double> own_a[NROLE], own_b[NROLE], span_a, span_b, ovl_a, ovl_b;
    collect(run_cfg(p.lo, c, dctl, dout, dinfo, dmem), c, own_a, span_a, ovl_a, res);
    collect(run_cfg(p.hi, c, dctl, dout, dinfo, dmem), c, own_b, span_b, ovl_b, res);
    bool single_role = true;
    int total_on = 0, n_max = 0;
    for (int s = 0; s < 4; ++s) {
        total_on += c.active[s];
        n_max = std::max(n_max, c.active[s]);
        for (int r = 0; r < c.active[s]; ++r) {
            single_role = single_role && c.role_of(r) == c.role_of(0);
        }
    }
    for (int ro = 0; ro < NROLE; ++ro) {
        res.own[ro] = 0;
        if (!own_a[ro].empty() && !own_b[ro].empty()) {
            res.own[ro] = (stats_of(own_b[ro]).median - stats_of(own_a[ro]).median) / (4.0 * g_iters[ro] * OPS[ro]);
        }
    }
    if (single_role) {
        const int ro = c.role_of(0);
        const int n = c.claim_one_simd ? total_on : n_max;
        const double nops = 4.0 * g_iters[ro] * OPS[ro] * core::REPS * n;
        const Stats sa = stats_of(span_a), sb = stats_of(span_b);
        res.simd_cpo = (sb.median - sa.median) / nops;
        res.simd_lo = (sb.min - sa.max) / nops;
        res.simd_hi = (sb.max - sa.min) / nops;
    }
    res.overlap = stats_of(ovl_b).median;
    return res;
}

static void print_res(const Cfg & c, const Res & r) {
    std::printf("RESULT mat_share %s %-30s active [%d %d %d %d] placed [%d %d %d %d]%s", CUMODE ? "CU " : "WGP",
                c.name.c_str(), c.active[0], c.active[1], c.active[2], c.active[3], r.waves_on[0], r.waves_on[1],
                r.waves_on[2], r.waves_on[3], r.placed ? "" : " WRONG");
    for (int ro = 0; ro < NROLE; ++ro) {
        if (r.own[ro] > 0) {
            if (ro == 2) {
                std::printf(" own_%s %.2f bytes/cycle per wave", ROLE_NAME[ro], 1.0 / r.own[ro]);
            } else {
                std::printf(" own_%s %.3f", ROLE_NAME[ro], r.own[ro]);
            }
        }
    }
    if (r.simd_cpo > 0) {
        std::printf(" simd_cycles_per_op %.3f [%.3f-%.3f]", r.simd_cpo, r.simd_lo, r.simd_hi);
    }
    std::printf(" overlap %.2f\n", r.overlap);
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = mat::flag(argc, argv, "--break");
    mat::set_input(mat::default_input());
    // iterations per role so each role alone runs about as long: matrix chain block 256 cycles, FMA block 16, memory
    // block several hundred (DRAM latency with 4 loads in flight), x4 256, fp8 128
    g_iters[0] = 32;
    g_iters[1] = 512;
    g_iters[2] = 16;
    g_iters[3] = 32;
    g_iters[4] = 64;
    uint8_t * dmem;
    CHECK(hipMalloc(&dmem, MEM_BYTES));
    CHECK(hipMemset(dmem, 0, MEM_BYTES));
    int * dctl;
    uint64_t * dout;
    uint32_t * dinfo;
    CHECK(hipMalloc(&dctl, 16 * sizeof(int)));
    CHECK(hipMalloc(&dout, (MAXW * core::REPS + MAXW * 4) * 8));
    CHECK(hipMalloc(&dinfo, (4 * MAXW + 1) * 4));

    // which SIMD keys exist in this mode
    Cfg probe{"probe", {16, 16, 16, 16}, 1};
    const auto pl = run_cfg(pair_of(1, 1).lo, probe, dctl, dout, dinfo, dmem);
    int seen[4] = {0, 0, 0, 0};
    for (int w = 0; w < MAXW; ++w) {
        seen[simd_key(pl[0].info[w])]++;
    }
    std::printf("RESULT mat_share mode %s: waves per SIMD key [%d %d %d %d] (WGP id %u)\n", CUMODE ? "CU" : "WGP",
                seen[0], seen[1], seen[2], seen[3], (pl[0].info[0] >> 10) & 15);

    mat::Checks ck;
    auto run = [&](const Cfg & c) {
        const Res r = measure_cfg(c, dctl, dout, dinfo, dmem);
        print_res(c, r);
        ck.add(r.placed, "%s placement", c.name.c_str());
        return r;
    };
    if (CUMODE) {
        if (brk) {
            Cfg c{"BREAK mm on 2 SIMDs claimed 1", {1, 1, 0, 0}, 0};
            c.claim_one_simd = true;
            const Res r = run(c);
            ck.add(r.simd_cpo >= 0.97 * 16, "two matrix waves claimed on one SIMD: %.3f cycles per instruction for "
                   "the SIMD, one matrix unit allows no less than %.2f", r.simd_cpo, 0.97 * 16);
            return ck.finish("mat_share_cu");
        }
        for (int n : {1, 2, 4, 8, 16}) {
            const Res r = run({"mm_f16_chain x" + std::to_string(n), {n, 0, 0, 0}, 0});
            // one matrix unit per SIMD at 16 cycles per f16 instruction: n waves cannot beat it, and if they share it
            // without loss they keep it busy
            ck.add(r.simd_cpo >= 0.97 * 16 && r.simd_cpo <= 1.1 * 16, "f16 chains, %d waves: %.3f SIMD cycles per "
                   "instruction (one unit kept busy: 15.5 to 17.6)", n, r.simd_cpo);
        }
        for (int n : {1, 2, 4, 16}) {
            run({"mm_f16_x4 x" + std::to_string(n), {n, 0, 0, 0}, 3});
            run({"mm_fp8_chain x" + std::to_string(n), {n, 0, 0, 0}, 4});
        }
        const Res f1 = run({"fma x1", {1, 0, 0, 0}, 1});
        ck.add(mat::near(f1.own[1], 1.0, 0.05), "FMA wave alone %.3f cycles per FMA (1 known)", f1.own[1]);
        run({"fma x2", {2, 0, 0, 0}, 1});
        run({"mem x1", {1, 0, 0, 0}, 2});
        run({"mem x2", {2, 0, 0, 0}, 2});
        run({"mm (rank 0) + fma", {2, 0, 0, 0}, 0, 1, 1});
        run({"fma (rank 0) + mm", {2, 0, 0, 0}, 1, 0, 1});
        run({"mm (rank 0) + mem", {2, 0, 0, 0}, 0, 2, 1});
        run({"mem (rank 0) + mm", {2, 0, 0, 0}, 2, 0, 1});
        run({"fp8 (rank 0) + fma", {2, 0, 0, 0}, 4, 1, 1});
        run({"mm_x4 (rank 0) + fma", {2, 0, 0, 0}, 3, 1, 1});
        run({"mm (rank 0) + fma x3", {4, 0, 0, 0}, 0, 1, 1});
        run({"mm x4 + fma x4 alternating", {8, 0, 0, 0}, 0, 1, 16, true});
        // matrix waves on both SIMDs of the CU at once
        run({"mm_f16_chain x1 both SIMDs", {1, 1, 0, 0}, 0});
        run({"mm_f16_x4 x4 both SIMDs", {4, 4, 0, 0}, 3});
        return ck.finish("mat_share_cu");
    }
    // WGP mode: SIMD keys 0..3 = (SIMD within CU << 1) | CU within WGP. Same CU: keys 0 and 2; different CUs: 0 and 1.
    struct P {
        const char * name;
        int a[4];
    };
    const std::vector<P> ps = {{"one SIMD", {1, 0, 0, 0}},          {"two SIMDs of one CU", {1, 0, 1, 0}},
                               {"two SIMDs of two CUs", {1, 1, 0, 0}}, {"all four SIMDs", {1, 1, 1, 1}}};
    for (int role : {0, 3, 4}) {
        for (int n : {1, 4}) {
            double base = 0;
            for (const P & p : ps) {
                Cfg c{std::string(ROLE_NAME[role]) + " x" + std::to_string(n) + " " + p.name, {0, 0, 0, 0}, role};
                for (int s = 0; s < 4; ++s) {
                    c.active[s] = p.a[s] * n;
                }
                const Res r = run(c);
                if (base == 0) {
                    base = r.simd_cpo;
                }
                // independent matrix units: every SIMD keeps the one-SIMD rate
                ck.add(std::fabs(r.simd_cpo / base - 1) <= 0.03, "%s: %.3f SIMD cycles per instruction against %.3f "
                       "on one SIMD", c.name.c_str(), r.simd_cpo, base);
            }
        }
    }
    return ck.finish("mat_share_wgp");
}
