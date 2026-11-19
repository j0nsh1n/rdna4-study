// Lane 1, item 6 (and the two-wave half of item 4): how several waves share one SIMD, and whether the SIMDs of a
// compute unit (CU) and of a work-group processor (WGP) are independent.
//
// Placement: one work-group of 1024 threads (32 waves). Built with -mcumode (CUMODE=1) the whole group sits on one CU
// (2 SIMDs, 16 waves each); built without it (WGP mode, the HIP default) it spreads over the 4 SIMDs of one WGP
// (8 waves each). Every wave reads HW_REG_HW_ID1 (SIMD_ID in bits 9:8: bit 0 = CU within the WGP, bit 1 = SIMD
// within the CU, per AMD's guide) and takes a rank on its SIMD with an LDS atomic. Measured: in CU mode a group lands
// on either CU of a WGP from one launch to the next, so in CU mode the SIMD is identified by bit 1 only. After one barrier, a wave whose
// rank is at least active[simd] ends at once; the others run the timed loop together. So the program cannot choose
// which hardware SIMD gets the group, but it chooses exactly how many waves run on each SIMD the group received.
// The placement of every wave is recorded and printed.
//
// Cost: as in core_common.h, the timed loop is built with U = 4 and U = 8 blocks and differences are taken. Two views:
// "own" = the cycles one wave sees between its own counter reads (slowest active wave); "simd" = the SIMD's span
// from the first active wave's start to the last one's end (same per-SIMD counter), divided by all the ops of all
// waves on it. If waves run one after another, own stays at the single-wave figure while the span grows; REALTIME
// (fixed rate, chip-wide) start and end stamps give the overlap factor that tells the two cases apart.
//
// Usage: core_share [--break]. --break claims that 2 independent-FMA waves make one SIMD issue 2 FMAs per cycle;
// the verdict must FAIL. SHARE_DEBUG=1 prints every active wave's SIMD, rank and time stamps.
#include "core_common.h"

#ifndef CUMODE
#error "build with -DCUMODE=1 -mcumode or -DCUMODE=0"
#endif

#define P_D_DKL(op, i) op " %" #i ", %" #i ", %16, %17\n"
#define P_D_DK(op, i) op " %" #i ", %" #i ", %16\n"
#define P_D_D(op, i) op " %" #i ", %" #i "\n"
#define P_DA(op, i) "s_delay_alu instid0(VALU_DEP_1)\n" op " %" #i ", %" #i ", %16, %17\n"

constexpr int MAXW = 32;

// REALTIME: the fixed-frequency counter shared by the whole chip (AMD's guide), to see whether waves overlap in time.
__device__ __forceinline__ uint64_t realtime() {
    uint64_t t;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\ns_wait_kmcnt 0" : "=s"(t));
    return t;
}

#define SHARE_KERNEL(NAME, BODY)                                                                                   \
    template <int U>                                                                                               \
    __global__ __launch_bounds__(1024) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,     \
                                                 uint32_t * __restrict__ info, const int * __restrict__ active) {  \
        __shared__ int cnt[4];                                                                                     \
        const int w = threadIdx.x / 32;                                                                            \
        if (threadIdx.x < 4) {                                                                                     \
            cnt[threadIdx.x] = 0;                                                                                  \
        }                                                                                                          \
        __syncthreads();                                                                                           \
        uint32_t hw;                                                                                               \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));                                          \
        /* CU mode: the group may land on either CU of the WGP (SIMD ids 0/2 or 1/3), so select by SIMD within  \
           the CU (bit 1); WGP mode: by the full SIMD id */                                                        \
        const int simd = CUMODE ? ((hw >> 9) & 1) : ((hw >> 8) & 3);                                               \
        int rank = 0;                                                                                              \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            rank = atomicAdd(&cnt[simd], 1);                                                                       \
        }                                                                                                          \
        rank = __builtin_amdgcn_readfirstlane(rank);                                                               \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            info[w] = hw;                                                                                          \
            info[MAXW + w] = rank;                                                                                 \
        }                                                                                                          \
        __syncthreads();                                                                                           \
        const bool on = rank < active[simd];                                                                       \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            info[2 * MAXW + w] = on;                                                                               \
        }                                                                                                          \
        if (!on) {                                                                                                 \
            return;                                                                                                \
        }                                                                                                          \
        float x[16];                                                                                               \
        for (int i = 0; i < 16; ++i) {                                                                             \
            x[i] = core::from_bits<float>(in[0]);                                                                  \
        }                                                                                                          \
        float k0 = core::from_bits<float>(in[1]), k1 = core::from_bits<float>(in[2]);                              \
        asm volatile("" : "+v"(k0), "+v"(k1));                                                                     \
        uint64_t span_c0 = 0, span_r0 = 0;                                                                         \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                               \
            if (rep == 0) {                                                                                        \
                span_r0 = realtime();                                                                              \
                span_c0 = wave_cycles();                                                                           \
            }                                                                                                      \
            const uint64_t t0 = wave_cycles();                                                                     \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                         \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                    \
                    asm volatile(BODY : CORE_X16("+v") : "v"(k0), "v"(k1) : "vcc", "scc");                         \
                }                                                                                                  \
            }                                                                                                      \
            const uint64_t t1 = wave_cycles();                                                                     \
            if ((threadIdx.x & 31) == 0) {                                                                         \
                out[w * core::REPS + rep] = cycle_diff(t0, t1);                                                    \
            }                                                                                                      \
        }                                                                                                          \
        const uint64_t span_c1 = wave_cycles();                                                                    \
        const uint64_t span_r1 = realtime();                                                                       \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            uint64_t * sp = out + MAXW * core::REPS + w * 4;                                                       \
            sp[0] = span_c0;                                                                                       \
            sp[1] = span_c1;                                                                                       \
            sp[2] = span_r0;                                                                                       \
            sp[3] = span_r1;                                                                                       \
        }                                                                                                          \
        uint32_t s = 0;                                                                                            \
        for (int i = 0; i < 16; ++i) {                                                                             \
            s ^= core::fold(x[i]);                                                                                 \
        }                                                                                                          \
        if (s == 0x12345678u) {                                                                                    \
            info[3 * MAXW] = s;                                                                                    \
        }                                                                                                          \
    }

// Same, but waves with an odd rank on their SIMD run BODY_ODD (the branch is uniform per wave, outside the asm).
#define SHARE_KERNEL2(NAME, BODY, BODY_ODD)                                                                                   \
    template <int U>                                                                                               \
    __global__ __launch_bounds__(1024) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,     \
                                                 uint32_t * __restrict__ info, const int * __restrict__ active) {  \
        __shared__ int cnt[4];                                                                                     \
        const int w = threadIdx.x / 32;                                                                            \
        if (threadIdx.x < 4) {                                                                                     \
            cnt[threadIdx.x] = 0;                                                                                  \
        }                                                                                                          \
        __syncthreads();                                                                                           \
        uint32_t hw;                                                                                               \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));                                          \
        /* CU mode: the group may land on either CU of the WGP (SIMD ids 0/2 or 1/3), so select by SIMD within  \
           the CU (bit 1); WGP mode: by the full SIMD id */                                                        \
        const int simd = CUMODE ? ((hw >> 9) & 1) : ((hw >> 8) & 3);                                               \
        int rank = 0;                                                                                              \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            rank = atomicAdd(&cnt[simd], 1);                                                                       \
        }                                                                                                          \
        rank = __builtin_amdgcn_readfirstlane(rank);                                                               \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            info[w] = hw;                                                                                          \
            info[MAXW + w] = rank;                                                                                 \
        }                                                                                                          \
        __syncthreads();                                                                                           \
        const bool on = rank < active[simd];                                                                       \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            info[2 * MAXW + w] = on;                                                                               \
        }                                                                                                          \
        if (!on) {                                                                                                 \
            return;                                                                                                \
        }                                                                                                          \
        float x[16];                                                                                               \
        for (int i = 0; i < 16; ++i) {                                                                             \
            x[i] = core::from_bits<float>(in[0]);                                                                  \
        }                                                                                                          \
        float k0 = core::from_bits<float>(in[1]), k1 = core::from_bits<float>(in[2]);                              \
        asm volatile("" : "+v"(k0), "+v"(k1));                                                                     \
        uint64_t span_c0 = 0, span_r0 = 0;                                                                         \
        /* the per-wave choice stays outside the timed loops */                                                  \
        if (rank & 1) {                                                                                            \
            for (int rep = 0; rep < core::REPS; ++rep) {                                                               \
                if (rep == 0) {                                                                                        \
                    span_r0 = realtime();                                                                              \
                    span_c0 = wave_cycles();                                                                           \
                }                                                                                                      \
                const uint64_t t0 = wave_cycles();                                                                     \
                _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                         \
                    _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                    \
                        asm volatile(BODY_ODD : CORE_X16("+v") : "v"(k0), "v"(k1) : "vcc", "scc");                      \
                    }                                                                                                  \
                }                                                                                                      \
                const uint64_t t1 = wave_cycles();                                                                     \
                if ((threadIdx.x & 31) == 0) {                                                                         \
                    out[w * core::REPS + rep] = cycle_diff(t0, t1);                                                    \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                   \
            for (int rep = 0; rep < core::REPS; ++rep) {                                                               \
                if (rep == 0) {                                                                                        \
                    span_r0 = realtime();                                                                              \
                    span_c0 = wave_cycles();                                                                           \
                }                                                                                                      \
                const uint64_t t0 = wave_cycles();                                                                     \
                _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                         \
                    _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                    \
                        asm volatile(BODY : CORE_X16("+v") : "v"(k0), "v"(k1) : "vcc", "scc");                      \
                    }                                                                                                  \
                }                                                                                                      \
                const uint64_t t1 = wave_cycles();                                                                     \
                if ((threadIdx.x & 31) == 0) {                                                                         \
                    out[w * core::REPS + rep] = cycle_diff(t0, t1);                                                    \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                          \
        const uint64_t span_c1 = wave_cycles();                                                                    \
        const uint64_t span_r1 = realtime();                                                                       \
        if ((threadIdx.x & 31) == 0) {                                                                             \
            uint64_t * sp = out + MAXW * core::REPS + w * 4;                                                       \
            sp[0] = span_c0;                                                                                       \
            sp[1] = span_c1;                                                                                       \
            sp[2] = span_r0;                                                                                       \
            sp[3] = span_r1;                                                                                       \
        }                                                                                                          \
        uint32_t s = 0;                                                                                            \
        for (int i = 0; i < 16; ++i) {                                                                             \
            s ^= core::fold(x[i]);                                                                                 \
        }                                                                                                          \
        if (s == 0x12345678u) {                                                                                    \
            info[3 * MAXW] = s;                                                                                    \
        }                                                                                                          \
    }

// Workloads (16 patterns per block): independent FMAs (issue-bound), one dependent FMA chain (latency-bound), the
// same chain with the compiler-style s_delay_alu before each FMA, independent dual-issue FMAC pairs, independent
// v_exp_f32 (quarter rate).
SHARE_KERNEL(w_ind, CORE_B16(P_D_DKL, "v_fma_f32"))
SHARE_KERNEL(w_dep, CORE_B1(P_D_DKL, "v_fma_f32"))
SHARE_KERNEL(w_dep_da, CORE_B1(P_DA, "v_fma_f32"))
SHARE_KERNEL(w_exp, CORE_B16(P_D_D, "v_exp_f32"))
#define DUALS                                                                                                       \
    "v_dual_fmac_f32 %0, %16, %17 :: v_dual_fmac_f32 %1, %16, %17\nv_dual_fmac_f32 %2, %16, %17 :: v_dual_fmac_f32 %3, %16, %17\n" \
    "v_dual_fmac_f32 %4, %16, %17 :: v_dual_fmac_f32 %5, %16, %17\nv_dual_fmac_f32 %6, %16, %17 :: v_dual_fmac_f32 %7, %16, %17\n" \
    "v_dual_fmac_f32 %8, %16, %17 :: v_dual_fmac_f32 %9, %16, %17\nv_dual_fmac_f32 %10, %16, %17 :: v_dual_fmac_f32 %11, %16, %17\n" \
    "v_dual_fmac_f32 %12, %16, %17 :: v_dual_fmac_f32 %13, %16, %17\nv_dual_fmac_f32 %14, %16, %17 :: v_dual_fmac_f32 %15, %16, %17\n"
// 16 patterns of one dual instruction each (two FMAs per pattern). Both halves read the same two VGPRs (allowed on
// gfx12); the destinations are consecutive chain registers, so their parity differs (the assembler checks it).
SHARE_KERNEL(w_dual, DUALS DUALS)
// Mixed pair on one SIMD: even-rank waves independent v_exp_f32 (quarter rate), odd-rank waves independent FMAs.
// One after another: per op averaged over both waves (4 + 1) / 2 = 2.5 cycles; fully overlapped: 4 / 2 = 2.0.
SHARE_KERNEL2(w_mix, CORE_B16(P_D_D, "v_exp_f32"), CORE_B16(P_D_DKL, "v_fma_f32"))
// The same with the FMAs dependent (one chain) behind s_delay_alu, and the exps independent.
SHARE_KERNEL2(w_mix_dep, CORE_B16(P_D_D, "v_exp_f32"), CORE_B1(P_DA, "v_fma_f32"))

using SKern = void (*)(const uint64_t *, uint64_t *, uint32_t *, const int *);

struct Placement {
    int on[4] = {0, 0, 0, 0};  // active waves per SIMD key
    int seen[4] = {0, 0, 0, 0};
    uint32_t wgp = 0;
    uint32_t cu_mask = 0;  // which CUs of the WGP the group's waves were on (bit 0 of SIMD_ID)
};

// One configuration, LAUNCHES launches. own: per repetition, the slowest active wave's own timed cycles. span: per
// launch, the largest over SIMDs of (last end - first start) of the active waves on that SIMD, in that SIMD's cycle
// counter, over all REPS repetitions (a span starting at repetition 1 would, when waves run one after another,
// include the other waves' repetition 0 without counting its work). overlap: per launch, the sum of the active waves' REALTIME durations divided by
// the REALTIME span of the group (1 = the waves ran one after another, n = all n ran at the same time).
struct GroupRun {
    std::vector<double> own, span, overlap;
    Placement pl;
};

static GroupRun run_group(SKern k, const int * dactive) {
    core::Buffers & b = core::buffers();
    static uint64_t * dout = nullptr;
    static uint32_t * dinfo = nullptr;
    constexpr int NOUT = MAXW * core::REPS + MAXW * 4;
    if (!dout) {
        CHECK(hipMalloc(&dout, NOUT * 8));
        CHECK(hipMalloc(&dinfo, (4 * MAXW + 1) * 4));
    }
    GroupRun g;
    for (int l = 0; l < core::LAUNCHES; ++l) {
        CHECK(hipMemset(dout, 0, NOUT * 8));
        CHECK(hipMemset(dinfo, 0, (4 * MAXW + 1) * 4));
        hipLaunchKernelGGL(k, dim3(1), dim3(1024), 0, 0, b.in, dout, dinfo, dactive);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> c(NOUT);
        uint32_t info[4 * MAXW + 1];
        CHECK(hipMemcpy(c.data(), dout, NOUT * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(info, dinfo, sizeof info, hipMemcpyDeviceToHost));
        g.pl = Placement{};
        if (std::getenv("SHARE_DEBUG")) {
            std::printf("debug launch %d:", l);
            for (int w = 0; w < MAXW; ++w) {
                if (info[2 * MAXW + w]) {
                    const uint64_t * sp = &c[MAXW * core::REPS + w * 4];
                    std::printf(" w%d:simd%u r%u cyc[%llu..%llu] rt[%llu..%llu]", w, (info[w] >> 8) & 3,
                                info[MAXW + w], (unsigned long long) sp[0], (unsigned long long) sp[1],
                                (unsigned long long) sp[2], (unsigned long long) sp[3]);
                }
            }
            std::printf("\n");
        }
        uint64_t c0[4] = {~0ull, ~0ull, ~0ull, ~0ull}, c1[4] = {0, 0, 0, 0};
        uint64_t r0 = ~0ull, r1 = 0;
        double rsum = 0;
        for (int w = 0; w < MAXW; ++w) {
            const int simd = CUMODE ? ((info[w] >> 9) & 1) : ((info[w] >> 8) & 3);
            g.pl.seen[simd]++;
            g.pl.wgp = (info[w] >> 10) & 15;
            g.pl.cu_mask |= 1u << ((info[w] >> 8) & 1);
            if (!info[2 * MAXW + w]) {
                continue;
            }
            g.pl.on[simd]++;
            const uint64_t * sp = &c[MAXW * core::REPS + w * 4];
            c0[simd] = std::min(c0[simd], sp[0]);
            c1[simd] = std::max(c1[simd], sp[1]);
            r0 = std::min(r0, sp[2]);
            r1 = std::max(r1, sp[3]);
            rsum += (double) (sp[3] - sp[2]);
        }
        double span = 0;
        for (int s = 0; s < 4; ++s) {
            if (c1[s]) {
                span = std::max(span, (double) (c1[s] - c0[s]));
            }
        }
        g.span.push_back(span);
        g.overlap.push_back(r1 > r0 ? rsum / (double) (r1 - r0) : 0);
        for (int r = 1; r < core::REPS; ++r) {
            double worst = 0;
            for (int w = 0; w < MAXW; ++w) {
                if (info[2 * MAXW + w]) {
                    worst = std::max(worst, (double) c[w * core::REPS + r]);
                }
            }
            g.own.push_back(worst);
        }
    }
    return g;
}

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = core::flag(argc, argv, "--break");
    core::set_init(core::Init{core::f32(1.0f), core::f32(1.0f), core::f32(0.0f)});
    // expect: SIMD cycles per op when the SIMD is the limit (independent work, any number of waves): FMA 1 (one
    // issue per cycle), dual FMAC 0.5, v_exp_f32 4 (quarter rate). 0 = no known answer (latency-bound chains).
    struct Wl {
        const char * id;
        SKern lo, hi;
        double ops_per_pattern, expect;
    };
    std::vector<Wl> wls = {
        {"ind_fma", w_ind<core::ULO>, w_ind<core::UHI>, 1, 1.0},
        {"dep_fma", w_dep<core::ULO>, w_dep<core::UHI>, 1, 0},
        {"dep_fma_delayalu", w_dep_da<core::ULO>, w_dep_da<core::UHI>, 1, 0},
        {"ind_exp", w_exp<core::ULO>, w_exp<core::UHI>, 1, 4.0},
        {"ind_dual_fmac", w_dual<core::ULO>, w_dual<core::UHI>, 2, 0.5},
        {"mix_exp_fma", w_mix<core::ULO>, w_mix<core::UHI>, 1, 0},
        {"mix_exp_depfma_da", w_mix_dep<core::ULO>, w_mix_dep<core::UHI>, 1, 0},
    };
    int * dactive;
    CHECK(hipMalloc(&dactive, 4 * sizeof(int)));

    GroupRun probe;
    {
        const int all[4] = {16, 16, 16, 16};
        CHECK(hipMemcpy(dactive, all, sizeof all, hipMemcpyHostToDevice));
        probe = run_group(w_ind<core::ULO>, dactive);
    }
    std::printf("RESULT share mode %s: 32 waves of one group per SIMD key (%s): [0] %d [1] %d [2] %d [3] %d (WGP id %u, "
                "CU mask %u)\n", CUMODE ? "CU" : "WGP", CUMODE ? "SIMD within the CU" : "SIMD id", probe.pl.seen[0],
                probe.pl.seen[1], probe.pl.seen[2], probe.pl.seen[3], probe.pl.wgp, probe.pl.cu_mask);
    std::vector<int> simds;
    for (int s = 0; s < 4; ++s) {
        if (probe.pl.seen[s]) {
            simds.push_back(s);
        }
    }
    const int per_simd_max = probe.pl.seen[simds[0]];

    struct Cfg {
        int nsimd, n;
    };
    std::vector<Cfg> cfgs;
    for (int n : {1, 2, 4, 8, 16}) {
        if (n <= per_simd_max) {
            cfgs.push_back({1, n});
        }
    }
    for (int n : {1, 2, 4, 8, 16}) {
        if (n <= per_simd_max && (int) simds.size() >= 2) {
            cfgs.push_back({(int) simds.size(), n});
        }
    }
    if (brk) {
        // Claim two waves of independent FMAs make one SIMD issue 2 FMAs per cycle: must FAIL.
        cfgs = {{1, 2}};
        wls = {wls[0]};
        wls[0].expect = 0.5;
    }

    bool ok = true;
    std::string text;
    char buf[512];
    for (const Wl & wl : wls) {
        for (const Cfg & c : cfgs) {
            int act[4] = {0, 0, 0, 0};
            for (int i = 0; i < c.nsimd; ++i) {
                act[simds[i]] = c.n;
            }
            CHECK(hipMemcpy(dactive, act, sizeof act, hipMemcpyHostToDevice));
            const GroupRun A = run_group(wl.lo, dactive);
            const GroupRun B = run_group(wl.hi, dactive);
            const double ops_rep = double(core::UHI - core::ULO) * core::ITERS * core::PATTERNS * wl.ops_per_pattern;
            const Stats oa = stats_of(A.own), ob = stats_of(B.own);
            const Stats sa = stats_of(A.span), sb = stats_of(B.span);
            const double own = (ob.median - oa.median) / ops_rep;  // cycles per op as one wave sees its own run
            // the SIMD: c.n waves did REPS repetitions each in the span
            const double nops = ops_rep * core::REPS * c.n;
            const double simd_cpo = (sb.median - sa.median) / nops;
            const double simd_lo = (sb.min - sa.max) / nops, simd_hi = (sb.max - sa.min) / nops;
            const double ovl = stats_of(B.overlap).median;
            bool placed = true;
            for (int s = 0; s < 4; ++s) {
                placed = placed && A.pl.on[s] == act[s] && B.pl.on[s] == act[s];
            }
            std::printf("RESULT share %-3s %-16s simds %d waves_per_simd %2d own_cycles_per_op_per_wave %7.3f "
                        "simd_cycles_per_op %6.3f [%.3f-%.3f] simd_ops_per_cycle %.3f total_ops_per_cycle %.3f "
                        "overlap %.2f placement %s\n",
                        CUMODE ? "CU" : "WGP", wl.id, c.nsimd, c.n, own, simd_cpo, simd_lo, simd_hi, 1.0 / simd_cpo,
                        c.nsimd / simd_cpo, ovl, placed ? "ok" : "WRONG");
            ok = ok && placed;
            // Known answer: a SIMD cannot beat the single-wave issue rate of core_valu/core_dual, and waves that can
            // each issue every cycle should keep it busy. Outside [0.97, 1.05] x the issue rate the check fails.
            if (wl.expect > 0 && (simd_cpo < 0.97 * wl.expect || simd_cpo > 1.05 * wl.expect)) {
                ok = false;
                std::snprintf(buf, sizeof buf, " %s simds %d waves %d: %.3f SIMD cycles/op, expected %.2f to %.2f;",
                              wl.id, c.nsimd, c.n, simd_cpo, 0.97 * wl.expect, 1.05 * wl.expect);
                text += buf;
            }
        }
    }
    if (text.empty()) {
        text = " SIMD-limited workloads stay between the single-wave issue rate and 5 % below it for every wave "
               "count and SIMD count (FMA 1, dual FMAC 0.5, exp 4 cycles per op); placement as requested";
    }
    verdict(CUMODE ? "core_share_cu" : "core_share_wgp", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
