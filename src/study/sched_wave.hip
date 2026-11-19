// Lane 7, item 4: wave32 against wave64 under HIP on gfx1201.
//
// One source, two builds (both in CU mode so a work-group stays on one CU):
//   wave32: -mcumode -DCUMODE=1
//   wave64: -mcumode -DCUMODE=1 -mwavefrontsize64 -DW64=1
// Sections: what the build reports (warpSize, builtins, kernel attributes, waves per 64 threads), FMA throughput per
// SIMD with 1/2/4/8 waves (inline-asm v_fma_f32 and the accumulate form the compiler may pair into v_dual_fmac_f32),
// streaming-load bandwidth, WMMA cycles per instruction, and how many waves of each width fit on one SIMD at register
// counts 24..256 (from REALTIME start/end stamps and placement of every wave).
// Usage: sched_wave [--break] [--quick]. --break claims WMMA takes 8 cycles per instruction; it must FAIL.
#include "sched_common.h"

#include <cstdarg>
#include <set>

#ifndef W64
#define W64 0
#endif
#ifndef CUMODE
#define CUMODE 1
#endif

namespace {

constexpr int WS = W64 ? 64 : 32;  // lanes per wave in this build
constexpr int REC = 32;
constexpr int MAXW = 32;

#define F1(i) "v_fma_f32 %" #i ", %" #i ", %16, %17\n"
#define FMA16 F1(0) F1(1) F1(2) F1(3) F1(4) F1(5) F1(6) F1(7) F1(8) F1(9) F1(10) F1(11) F1(12) F1(13) F1(14) F1(15)
#define XOPS                                                                                                      \
    "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), "+v"(x[8]), \
        "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])
#define FMA_BLOCK() asm volatile(FMA16 : XOPS : "v"(k0), "v"(k1))

__device__ __forceinline__ bool lane0() { return (threadIdx.x % WS) == 0; }

// ---------------------------------------------------------------------------------------------------------------
// Build report
__global__ __launch_bounds__(64) void k_info(int * dev, uint64_t * pl) {
    if (threadIdx.x == 0) {
        dev[0] = warpSize;
        dev[1] = __builtin_amdgcn_wavefrontsize();
#ifdef __AMDGCN_WAVEFRONT_SIZE
        dev[2] = __AMDGCN_WAVEFRONT_SIZE;
#else
        dev[2] = -1;
#endif
    }
    if (lane0()) {
        const int i = atomicAdd(&dev[3], 1);
        pl[i] = sched::placement();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// FMA throughput on one SIMD. All waves of the group take a rank on their SIMD (bit 1 of SIMD_ID in CU mode); waves on
// SIMD 0 with rank < nact run the loop, everyone else leaves. Every active wave records its first start and last end on
// the SIMD's own cycle counter, so the SIMD span is comparable; the cost per iteration is the slope of the span over
// the iteration count (two launches), which cancels the loop control and the start skew.
// MODE 0: inline-asm v_fma_f32, 16 independent registers (cannot be paired). MODE 1: accumulate form in C++ (the
// compiler may emit v_fmac_f32 and pair it into v_dual_fmac_f32).
// U blocks of 16 per iteration; the host builds U = 4 and 8 and differences them so the loop control cancels (as lane 1).
template <int MODE, int U>
__global__ __launch_bounds__(1024) void k_fma(const float * in, uint64_t * out, float * sink, int nact, int n) {
    __shared__ int cnt[2];
    if (threadIdx.x < 2) {
        cnt[threadIdx.x] = 0;
    }
    __syncthreads();
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    const int simd = (hw >> 9) & 1;
    int rank = 0;
    if (lane0()) {
        rank = atomicAdd(&cnt[simd], 1);
    }
    rank = __builtin_amdgcn_readfirstlane(rank);
    const int w = threadIdx.x / WS;
    const uint32_t pl = sched::placement();
    if (lane0()) {
        out[w * REC] = pl;
        out[w * REC + 3] = rank;
        out[w * REC + 4] = simd;
    }
    __syncthreads();
    if (simd != 0 || rank >= nact) {
        return;
    }
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[i];
    }
    float k0 = in[16], k1 = in[17];
    asm volatile("" : "+v"(k0), "+v"(k1));
    const uint64_t s0 = sched::cyc();
    if (MODE == 0) {
#pragma unroll 1
        for (int it = 0; it < n; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                FMA_BLOCK();
            }
        }
    } else {
#pragma unroll 1
        for (int it = 0; it < n; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    x[i] = __builtin_fmaf(k0, k1, x[i]);
                }
            }
        }
    }
    const uint64_t e0 = sched::cyc();
    if (lane0()) {
        out[w * REC + 1] = s0;
        out[w * REC + 2] = e0;
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    sink[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------------------------------------------
// Streaming 16-byte loads
__global__ void k_stream(const uint4 * __restrict__ p, size_t n16, uint32_t * sink) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t a = 0, b = 0, c = 0, d = 0;
#pragma unroll 4
    for (; i < n16; i += stride) {
        const uint4 v = p[i];
        a ^= v.x;
        b ^= v.y;
        c ^= v.z;
        d ^= v.w;
    }
    if ((a ^ b ^ c ^ d) == 0x9e3779b9u) {
        sink[0] = a;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// WMMA chain: one wave. CHAIN 1 = every instruction depends on the previous one's accumulator; CHAIN 2 = two
// independent accumulators alternate.
#if W64
typedef _Float16 vh __attribute__((ext_vector_type(4)));
typedef float vf __attribute__((ext_vector_type(4)));
#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(a, b, c)
#else
typedef _Float16 vh __attribute__((ext_vector_type(8)));
typedef float vf __attribute__((ext_vector_type(8)));
#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
#endif

template <int CHAIN, int U>
__global__ __launch_bounds__(WS) void k_wmma(const float * in, uint64_t * out, float * sink, int n1, int n2) {
    vh a, b;
    vf c0, c1;
    for (int i = 0; i < (int) (sizeof(vh) / 2); ++i) {
        a[i] = (_Float16) in[i];
        b[i] = (_Float16) in[8 + i];
    }
    for (int i = 0; i < (int) (sizeof(vf) / 4); ++i) {
        c0[i] = in[16 + i];
        c1[i] = in[24 + i];
    }
    auto run = [&](int n) {
#pragma unroll 1
        for (int it = 0; it < n; ++it) {
#pragma unroll
            for (int j = 0; j < U; ++j) {
                if (CHAIN == 1) {
                    c0 = WMMA(a, b, c0);
                } else if (j & 1) {
                    c1 = WMMA(a, b, c1);
                } else {
                    c0 = WMMA(a, b, c0);
                }
            }
        }
    };
    for (int rep = 0; rep < 6; ++rep) {
        const uint64_t t0 = sched::cyc();
        run(n1);
        const uint64_t t1 = sched::cyc();
        run(n2);
        const uint64_t t2 = sched::cyc();
        if (threadIdx.x == 0) {
            out[1 + 2 * rep] = t1 - t0;
            out[2 + 2 * rep] = t2 - t1;
        }
    }
    float s = 0;
    for (int i = 0; i < (int) (sizeof(vf) / 4); ++i) {
        s += c0[i] + c1[i];
    }
    sink[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------------------------------------------
// Waves per SIMD. R values are kept live in VGPRs by the compiler (one asm touch each per spin iteration); each wave
// spins about 50 us of REALTIME and records placement and REALTIME start/end. Groups of 8 waves, many groups.
constexpr int SPIN_TICKS = 5000;  // 100 MHz

// G waves per group: 1 measures the per-SIMD cap itself; larger groups in CU mode need room on both SIMDs of a CU.
template <int L, int G>
__global__ __launch_bounds__(G * WS) void k_regs(const float * in, uint64_t * rec) {
    float x[L];
    for (int i = 0; i < L; ++i) {
        x[i] = in[i & 15] + (float) i;
    }
    const float c = in[17];
    const uint64_t t0 = sched::rt();
    uint64_t t;
    do {
#pragma unroll
        for (int i = 0; i < L; ++i) {
            asm volatile("v_add_f32 %0, %0, %1" : "+v"(x[i]) : "v"(c));
        }
        t = sched::rt();
    } while (t - t0 < SPIN_TICKS);
    const uint32_t pl = sched::placement();
    if (lane0()) {
        const size_t w = ((size_t) blockIdx.x * G + threadIdx.x / WS);
        rec[w * 3] = t0;
        rec[w * 3 + 1] = t;
        rec[w * 3 + 2] = pl;
    }
    float s = 0;
    for (int i = 0; i < L; ++i) {
        s += x[i];
    }
    if (s == 1.2345f) {
        rec[0] = 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host

uint64_t * d_out;
float * d_in;
float * d_sink;

std::vector<uint64_t> fetch(int nwaves) {
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> h((size_t) nwaves * REC);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * 8, hipMemcpyDeviceToHost));
    return h;
}
void clear_out() { CHECK(hipMemset(d_out, 0, (size_t) MAXW * REC * 8)); }

struct Verdicts {
    int fails = 0;
    void put(const char * name, bool ok, const std::string & text) {
        if (!ok) {
            ++fails;
        }
        std::printf("VERDICT sched_wave %s: %s %s\n", name, ok ? "PASS" : "FAIL", text.c_str());
        std::fflush(stdout);
    }
};

std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char * f, ...) {
    char b[768];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

int regs_rule32(int vgprs) {  // lane 2: min(16, floor(1536 / VGPRs rounded up to 24))
    const int r = (vgprs + 23) / 24 * 24;
    return std::min(16, 1536 / r);
}

// Max number of intervals overlapping on one SIMD, from records [t0, t1, placement] of n waves.
struct Occ {
    int maxres = 0;
    int waves = 0;
    int simds = 0;
    int simds_at_max = 0;
};
Occ occupancy(const std::vector<uint64_t> & rec, int n) {
    std::map<uint32_t, std::vector<std::pair<uint64_t, int>>> ev;
    for (int w = 0; w < n; ++w) {
        if (rec[(size_t) w * 3 + 1] == 0) {
            continue;
        }
        const uint32_t k = sched::pl_simd((uint32_t) rec[(size_t) w * 3 + 2]);
        ev[k].push_back({rec[(size_t) w * 3], +1});
        ev[k].push_back({rec[(size_t) w * 3 + 1], -1});
    }
    Occ o;
    o.waves = 0;
    for (int w = 0; w < n; ++w) {
        o.waves += rec[(size_t) w * 3 + 1] != 0;
    }
    std::vector<int> per;
    for (auto & kv : ev) {
        auto & v = kv.second;
        std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.first != b.first ? a.first < b.first : a.second < b.second; });
        int cur = 0, mx = 0;
        for (auto & e : v) {
            cur += e.second;
            mx = std::max(mx, cur);
        }
        per.push_back(mx);
        o.maxres = std::max(o.maxres, mx);
    }
    o.simds = (int) ev.size();
    for (int m : per) {
        o.simds_at_max += m == o.maxres;
    }
    return o;
}

}  // namespace

template <int L, int G>
static Occ run_regs(int groups, int & numregs, int & spill, int & maxthr, size_t & scratch) {
    hipFuncAttributes at;
    CHECK(hipFuncGetAttributes(&at, (const void *) k_regs<L, G>));
    numregs = at.numRegs;
    spill = (int) at.localSizeBytes;
    maxthr = at.maxThreadsPerBlock;
    scratch = (size_t) at.localSizeBytes;
    const int nw = groups * G;
    uint64_t * d;
    CHECK(hipMalloc(&d, (size_t) nw * 3 * 8));
    CHECK(hipMemset(d, 0, (size_t) nw * 3 * 8));
    hipLaunchKernelGGL((k_regs<L, G>), dim3(groups), dim3(G * WS), 0, 0, d_in, d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> h((size_t) nw * 3);
    CHECK(hipMemcpy(h.data(), d, h.size() * 8, hipMemcpyDeviceToHost));
    CHECK(hipFree(d));
    return occupancy(h, nw);
}

template <int G>
static bool regs_table(int groups) {
    std::printf("\n## Waves resident on one SIMD against VGPRs (wave%d, %d groups of %d wave(s))\n| live values requested | VGPRs reported (hipFuncGetAttributes numRegs) | scratch bytes | max waves resident on one SIMD | SIMDs seen | SIMDs at that maximum | wave32 rule min(16, 1536/roundup24(VGPRs)) | same rule with 2 x VGPRs |\n|---|---|---|---|---|---|---|---|\n",
                WS, groups, G);
    struct Row {
        int L, regs, spill, maxthr;
        Occ o;
    };
    std::vector<Row> rows;
    size_t scr = 0;
#define REGROW(LV)                                                                                                       \
    {                                                                                                                    \
        Row r{};                                                                                                         \
        r.L = LV;                                                                                                        \
        r.o = run_regs<LV, G>(groups, r.regs, r.spill, r.maxthr, scr);                                                   \
        rows.push_back(r);                                                                                               \
    }
    REGROW(16) REGROW(40) REGROW(88) REGROW(120) REGROW(136) REGROW(152) REGROW(168) REGROW(184) REGROW(200) REGROW(216) REGROW(240)
#undef REGROW
    bool match_same = true;
    for (const Row & r : rows) {
        const int rule32 = regs_rule32(r.regs), rule2 = regs_rule32(2 * r.regs);
        const int rule = WS == 32 ? rule32 : rule2;
        std::printf("RESULT sched_wave regs wave%d group_waves %d live %d vgprs %d scratch %d maxres %d simds %d at_max %d rule32 %d\n", WS,
                    G, r.L, r.regs, r.spill, r.o.maxres, r.o.simds, r.o.simds_at_max, rule32);
        std::printf("| %d | %d | %d | %d | %d | %d | %d | %d |\n", r.L, r.regs, r.spill, r.o.maxres, r.o.simds, r.o.simds_at_max, rule32,
                    rule2);
        match_same = match_same && r.o.maxres == rule;
    }
    return match_same;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const bool brk = sched::has_flag(argc, argv, "--break");
    const bool quick = sched::has_flag(argc, argv, "--quick");
    Verdicts V;
    std::printf("# sched_wave: build wave%d, CU mode, device %s\n", WS, prop.gcnArchName);
    CHECK(hipMalloc(&d_out, (size_t) MAXW * REC * 8));
    CHECK(hipMalloc(&d_in, 64 * 4));
    CHECK(hipMalloc(&d_sink, 1024 * 4));
    {
        float h[64];
        for (int i = 0; i < 64; ++i) {
            h[i] = 1.0f + 0.01f * (float) (i & 15);
        }
        h[16] = 1.0f;  // k0 for the asm FMAs: x = x * 1 + 0 stays put
        h[17] = 0.0f;
        CHECK(hipMemcpy(d_in, h, sizeof h, hipMemcpyHostToDevice));
    }

    // ---------------- what the build reports ------------------------------------------------------------------
    {
        int * dd;
        uint64_t * pl;
        CHECK(hipMalloc(&dd, 16));
        CHECK(hipMalloc(&pl, 64 * 8));
        CHECK(hipMemset(dd, 0, 16));
        CHECK(hipMemset(pl, 0, 64 * 8));
        hipLaunchKernelGGL(k_info, dim3(1), dim3(64), 0, 0, dd, pl);
        CHECK(hipDeviceSynchronize());
        int h[4];
        uint64_t p[64];
        CHECK(hipMemcpy(h, dd, 16, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(p, pl, sizeof p, hipMemcpyDeviceToHost));
        hipFuncAttributes at;
        CHECK(hipFuncGetAttributes(&at, (const void *) k_info));
        std::printf("RESULT sched_wave report build wave%d device_warpSize %d builtin_wavefrontsize %d __AMDGCN_WAVEFRONT_SIZE %d "
                    "host_warpSize %d k_info_numRegs %d maxThreadsPerBlock %d\n",
                    WS, h[0], h[1], h[2], prop.warpSize, at.numRegs, at.maxThreadsPerBlock);
        V.put("warpsize_report", h[0] == WS && h[1] == WS,
              fmt("wave%d build: device warpSize %d, __builtin_amdgcn_wavefrontsize() %d, __AMDGCN_WAVEFRONT_SIZE %d, host "
                  "hipDeviceProp_t::warpSize %d (device values must equal %d)",
                  WS, h[0], h[1], h[2], prop.warpSize, WS));
        std::set<std::pair<uint32_t, int>> slots;
        for (int i = 0; i < h[3]; ++i) {
            slots.insert({sched::pl_simd((uint32_t) p[i]), sched::pl_slot((uint32_t) p[i])});
        }
        const int want = 64 / WS;
        std::printf("RESULT sched_wave report block 64 threads: lane-0 threads %d, distinct (SIMD, wave slot) %zu\n", h[3],
                    slots.size());
        V.put("waves_per_64_threads", h[3] == want && (int) slots.size() == want,
              fmt("a block of 64 threads is %d wave(s) in wave%d (lane-0 count %d, distinct SIMD+slot %zu, expected %d)", h[3], WS,
                  h[3], slots.size(), want));
        CHECK(hipFree(dd));
        CHECK(hipFree(pl));
    }

    // ---------------- FMA throughput per SIMD -----------------------------------------------------------------
    {
        const int nacts[] = {1, 2, 4, 8};
        const int n1 = 64, n2 = 192;
        std::printf("\n## Lane-FMAs per cycle per SIMD (wave%d)\n| form | waves on the SIMD | lane-FMAs per cycle | range | cycles per wave-instruction (all waves) |\n|---|---|---|---|---|\n", WS);
        double asm1 = 0;
        double one_rate[2] = {0, 0};
        bool rows_ok = true;
        for (int mode = 0; mode < 2; ++mode) {
            for (int nact : nacts) {
                std::vector<double> thr;
                int got_min = 99;
                for (int rep = 0; rep < 7; ++rep) {
                    double per_iter[2];  // [0] U = 4, [1] U = 8
                    for (int ui = 0; ui < 2; ++ui) {
                        double span[2];
                        for (int k = 0; k < 2; ++k) {
                            clear_out();
                            const int n = k ? n2 : n1;
                            const int sel = mode * 2 + ui;
                            if (sel == 0) {
                                hipLaunchKernelGGL((k_fma<0, 4>), dim3(1), dim3(1024), 0, 0, d_in, d_out, d_sink, nact, n);
                            } else if (sel == 1) {
                                hipLaunchKernelGGL((k_fma<0, 8>), dim3(1), dim3(1024), 0, 0, d_in, d_out, d_sink, nact, n);
                            } else if (sel == 2) {
                                hipLaunchKernelGGL((k_fma<1, 4>), dim3(1), dim3(1024), 0, 0, d_in, d_out, d_sink, nact, n);
                            } else {
                                hipLaunchKernelGGL((k_fma<1, 8>), dim3(1), dim3(1024), 0, 0, d_in, d_out, d_sink, nact, n);
                            }
                            const auto h = fetch(1024 / WS);
                            uint64_t mn = ~0ull, mx = 0;
                            int got = 0;
                            for (int w = 0; w < 1024 / WS; ++w) {
                                if (h[(size_t) w * REC + 2] == 0) {
                                    continue;
                                }
                                ++got;
                                mn = std::min(mn, h[(size_t) w * REC + 1]);
                                mx = std::max(mx, h[(size_t) w * REC + 2]);
                            }
                            got_min = std::min(got_min, got);
                            span[k] = (double) (mx - mn);
                        }
                        per_iter[ui] = (span[1] - span[0]) / (n2 - n1);
                    }
                    thr.push_back((double) got_min * WS * 4 * 16 / (per_iter[1] - per_iter[0]));
                }
                const double m = sched::pct(thr, 0.5);
                if (mode == 0 && nact == 1) {
                    asm1 = m;
                }
                // Lead's addition: several waves on one SIMD lose nothing (lane 1), so every row must reach 90 % of the
                // one-wave rate of its form; a build without -mcumode gave 0.00 here and every verdict still passed.
                if (nact == 1) {
                    one_rate[mode] = m;
                } else if (!(m >= 0.9 * one_rate[mode])) {
                    rows_ok = false;
                }
                const char * nm = mode == 0 ? "asm v_fma_f32 (independent)" : "C++ fmaf accumulate";
                std::printf("RESULT sched_wave fma wave%d %s waves_on_simd %d (launched/active %d) lane_fma_per_cycle %s\n", WS, nm,
                            nact, got_min, sched::dist(thr, "%.2f").c_str());
                std::printf("| %s | %d (got %d) | %.2f | %.2f..%.2f | %.3f |\n", nm, nact, got_min, m, sched::pct(thr, 0),
                            sched::pct(thr, 1), (double) nact * WS / m);
            }
        }
        V.put("fma_rows", rows_ok,
              fmt("wave%d: every multi-wave FMA row reaches 90 %% of its one-wave rate: %s", WS, rows_ok ? "yes" : "NO"));
        // one wave alone issues one FMA per cycle: WS lane-FMAs per cycle
        const double instr_per_cycle = asm1 / WS;
        const bool ok = WS == 32 ? std::fabs(instr_per_cycle - 1.0) <= 0.03
                                 : (asm1 >= 32.0 * 0.97 && asm1 <= 64.0 * 1.03);
        V.put("fma_one_wave", ok,
              fmt("wave%d: one wave alone on a SIMD, independent v_fma_f32: %.2f lane-FMAs per cycle = %.3f wave-instructions per cycle "
                  "(wave32 expects 1.00 +- 3 %%; wave64 passes between 32 and 64 lane-FMAs per cycle, the bounds of one half-wave pass per cycle and two)",
                  WS, asm1, instr_per_cycle));
    }

    // ---------------- streaming loads -------------------------------------------------------------------------
    if (!quick) {
        const size_t bytes = 256ull << 20;
        uint4 * buf;
        uint32_t * sk;
        CHECK(hipMalloc(&buf, bytes));
        CHECK(hipMalloc(&sk, 64));
        CHECK(hipMemset(buf, 0x5a, bytes));
        const int cfg[][2] = {{8, 1024}, {256, 256}, {1024, 256}};
        std::printf("\n## Streaming 16-byte loads over 256 MiB (wave%d)\n| grid x block | total threads | GB/s median | range |\n|---|---|---|---|\n", WS);
        for (auto & c : cfg) {
            auto ms = time_ms([&] { hipLaunchKernelGGL(k_stream, dim3(c[0]), dim3(c[1]), 0, 0, buf, bytes / 16, sk); }, 9, 3);
            std::vector<double> gbs;
            for (double m : ms) {
                gbs.push_back((double) bytes / (m * 1e-3) / 1e9);
            }
            std::printf("RESULT sched_wave stream wave%d grid %d block %d GB/s %s\n", WS, c[0], c[1], sched::dist(gbs, "%.1f").c_str());
            std::printf("| %d x %d | %d | %.1f | %.1f..%.1f |\n", c[0], c[1], c[0] * c[1], sched::pct(gbs, 0.5), sched::pct(gbs, 0),
                        sched::pct(gbs, 1));
        }
        V.put("stream_ran", true, fmt("wave%d streaming sweeps done (table above)", WS));
        CHECK(hipFree(buf));
        CHECK(hipFree(sk));
    }

    // ---------------- WMMA ------------------------------------------------------------------------------------
    {
        const int n1 = 64, n2 = 192;
        double per[2] = {};
        double lo[2] = {}, hi[2] = {};
        for (int ch = 1; ch <= 2; ++ch) {
            std::vector<double> v;
            std::vector<double> slope[2];  // [0] 16 instructions per iteration, [1] 32
            for (int ui = 0; ui < 2; ++ui) {
                for (int launch = 0; launch < 5; ++launch) {
                    clear_out();
                    const int sel = (ch - 1) * 2 + ui;
                    if (sel == 0) {
                        hipLaunchKernelGGL((k_wmma<1, 16>), dim3(1), dim3(WS), 0, 0, d_in, d_out, d_sink, n1, n2);
                    } else if (sel == 1) {
                        hipLaunchKernelGGL((k_wmma<1, 32>), dim3(1), dim3(WS), 0, 0, d_in, d_out, d_sink, n1, n2);
                    } else if (sel == 2) {
                        hipLaunchKernelGGL((k_wmma<2, 16>), dim3(1), dim3(WS), 0, 0, d_in, d_out, d_sink, n1, n2);
                    } else {
                        hipLaunchKernelGGL((k_wmma<2, 32>), dim3(1), dim3(WS), 0, 0, d_in, d_out, d_sink, n1, n2);
                    }
                    const auto h = fetch(1);
                    for (int rep = 1; rep < 6; ++rep) {  // rep 0 warms the instruction cache
                        slope[ui].push_back(((double) h[2 + 2 * rep] - (double) h[1 + 2 * rep]) / (n2 - n1));
                    }
                }
            }
            for (size_t i = 0; i < slope[0].size(); ++i) {
                v.push_back((slope[1][i] - slope[0][i]) / 16.0);
            }
            per[ch - 1] = sched::pct(v, 0.5);
            lo[ch - 1] = sched::pct(v, 0);
            hi[ch - 1] = sched::pct(v, 1);
            std::printf("RESULT sched_wave wmma wave%d %s cycles_per_instruction %s\n", WS,
                        ch == 1 ? "dependent chain" : "two chains", sched::dist(v, "%.3f").c_str());
        }
        std::printf("\n## WMMA v_wmma_f32_16x16x16_f16 (wave%d)\n| form | cycles per instruction | range |\n|---|---|---|\n| dependent chain | %.3f | %.3f..%.3f |\n| two independent chains | %.3f | %.3f..%.3f |\n",
                    WS, per[0], lo[0], hi[0], per[1], lo[1], hi[1]);
        const double want = brk ? 8.0 : 16.0;
        const double best = std::min(per[0], per[1]);
        V.put("wmma_rate", std::fabs(best - want) <= 0.03 * want,
              fmt("wave%d v_wmma_f32_16x16x16_f16: %.3f cycles per instruction in the faster form (dependent %.3f, two chains %.3f; "
                  "expected %.1f +- 3 %%)",
                  WS, best, per[0], per[1], want));
    }

    // ---------------- registers and waves per SIMD ------------------------------------------------------------
    {
        const bool ok1 = regs_table<1>(quick ? 1024 : 8192);
        regs_table<4>(quick ? 256 : 2048);
        // The rule is judged on 1-wave groups, which measure the per-SIMD cap without the pairing a 4-wave CU-mode group needs.
        V.put("waves_per_simd_rule", ok1,
              WS == 32 ? fmt("wave32, 1-wave groups: measured max waves per SIMD equals min(16, floor(1536 / VGPRs rounded up to 24)) at every register count: %s",
                             ok1 ? "yes" : "no (see table)")
                       : fmt("wave64, 1-wave groups: measured max waves per SIMD equals min(16, floor(1536 / (2 x VGPRs rounded up to 24))), i.e. a wave64 wave takes twice the registers of a wave32 wave with the same per-lane count, at every register count: %s",
                             ok1 ? "yes" : "no"));
    }
    return V.fails ? 1 : 0;
}
