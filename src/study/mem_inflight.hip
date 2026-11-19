// Lane 2, items 5 and 7 (one wave, or a few waves on one CU; cycle counter). Built with -mcumode, so a work-group
// sits on one CU (2 SIMDs) and the waves of item 7 can be placed on the same SIMD or on the other one.
//
// Usage: mem_inflight <inflight|waits> [--break]
//   inflight  one wave issues N independent loads (W bytes per lane, consecutive lanes consecutive, so one load of
//             W = 4 is one 128-byte line and one load of W = 16 is four), then waits once (s_wait_loadcnt 0), and
//             repeats. Cycles per batch give the throughput against N. Data in memory (fresh lines after a flush),
//             in L2 (a 1 MiB region read before) and in L0 (a 16 KiB region). N runs past 63, the limit of the
//             6-bit load counter in the wait instruction: the values of the last batch are checked against the
//             data (a ramp), so a wait that returned early would show as wrong values.
//   waits     item 7: a "fast" wave does batches of 8 independent loads that hit L0 (or L2) while 0 to 31 "slow" waves
//             on the same SIMD and on the other SIMD of the CU follow a random pointer chain through 1 GiB (each load
//             about 700 cycles). Does the fast wave slow down when slow loads are in flight around it?
//             --break claims the fast wave runs at 3x its measured alone-speed: the verdict must FAIL.
#include "mem_common.h"

typedef unsigned u4 __attribute__((ext_vector_type(4)));

template <int W>
struct V;
template <>
struct V<4> {
    using T = unsigned;
};
template <>
struct V<16> {
    using T = u4;
};

template <int W, int K>
__device__ __forceinline__ void ldk(typename V<W>::T & r, const char * base, uint32_t lane_off) {
    if constexpr (W == 4) {
        asm volatile("global_load_b32 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 128) : "memory");
    } else {
        asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 512) : "memory");
    }
}

template <int W, int K, int N>
__device__ __forceinline__ void issue(typename V<W>::T (&r)[N], const char * base, uint32_t lane_off) {
    if constexpr (K < N) {
        ldk<W, K>(r[K], base, lane_off);
        issue<W, K + 1, N>(r, base, lane_off);
    }
}

// The loads are inline assembly, so the compiler neither waits for them nor knows when their registers arrive. After
// the wait, an empty statement re-defines each destination: every use stays below the wait and no destination is
// reused while its load is in flight.
template <int W, int N>
__device__ __forceinline__ void wait_all(typename V<W>::T (&r)[N]) {
    asm volatile("s_wait_loadcnt 0" ::: "memory");
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" : "+v"(r[k]));
    }
}

// out: [0] cycles for `batches` batches, [1] mismatches in the last batch, [2] sum.
// Batch b reads at base + (b % wrap) * N * 32 * W (wrap = batches for fresh memory, small for a cached region).
template <int W, int N>
__global__ __launch_bounds__(32) void k_inflight(const char * base, uint32_t base_word, uint32_t batches, uint32_t wrap,
                                                 uint32_t warm, uint64_t * out) {
    using T = typename V<W>::T;
    const uint32_t lane_off = (threadIdx.x % 32) * W;
    constexpr uint32_t BATCH_BYTES = N * 32 * W;
    T r[N];
    unsigned acc = 0;
    // untimed pass over a cached region: L2 starts every launch empty (mode persist of mem_latency)
#pragma unroll 1
    for (uint32_t b = 0; b < warm; ++b) {
        issue<W, 0, N>(r, base + (size_t) (b % wrap) * BATCH_BYTES, lane_off);
        wait_all<W, N>(r);
    }
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t b = 0; b < batches; ++b) {
        const char * q = base + (size_t) (b % wrap) * BATCH_BYTES;
        issue<W, 0, N>(r, q, lane_off);
        wait_all<W, N>(r);
    }
    const uint64_t c1 = wave_cycles();
    // the data is a ramp of 32-bit words: word i holds i
    const uint32_t last = (batches - 1) % wrap;
    uint32_t bad = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        const uint32_t word = base_word + (uint32_t) (((size_t) last * BATCH_BYTES + (size_t) k * 32 * W + lane_off) / 4);
        if constexpr (W == 4) {
            bad += r[k] != word;
            acc += r[k];
        } else {
            bad += r[k].x != word || r[k].w != word + 3;
            acc += r[k].x;
        }
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = bad;
        out[2] = acc;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Item 7. One work-group of 1024 threads (32 waves, one CU in CU mode: 16 per SIMD). Each wave takes a rank on its
// SIMD; roles: fast = rank 0 on SIMD A (the SIMD of wave 0); slow = ranks 1..n_same on SIMD A and ranks 0..n_other-1
// on the other SIMD. Inactive waves exit after the barrier. fast_bytes: the fast wave's region (16 KiB = L0,
// 1 MiB = L2). out: [0] fast cycles, [1] fast batches, [2] slow loads done, [3] slow cycles per load (median wave),
// [4] number of slow waves, [5] placement of the fast wave.
constexpr int FAST_BATCHES = 4096;

__global__ __launch_bounds__(1024) void k_waits(uint32_t n_same, uint32_t n_other, const char * fast_base, uint32_t fast_bytes, uint64_t chase_start,
                                                uint64_t * out) {
    __shared__ uint32_t rank_ctr[2];
    __shared__ uint32_t simd_a;
    __shared__ volatile uint32_t done;
    __shared__ uint32_t slow_loads, slow_waves;
    __shared__ uint64_t slow_cyc;
    const uint32_t pl = mem::placement();
    const uint32_t simd = (pl >> 10) & 1u;
    if (threadIdx.x == 0) {
        rank_ctr[0] = rank_ctr[1] = 0;
        simd_a = simd;
        done = 0;
        slow_loads = slow_waves = 0;
        slow_cyc = 0;
    }
    __syncthreads();
    uint32_t rank = 0;
    if (threadIdx.x % 32 == 0) {
        rank = atomicAdd(&rank_ctr[simd], 1u);
    }
    rank = __shfl(rank, 0);
    __syncthreads();
    const bool same = simd == simd_a;
    const bool fast = same && rank == 0;
    const bool slow = same ? (rank >= 1 && rank <= n_same) : rank < n_other;
    const uint32_t lane_off = (threadIdx.x % 32) * 4;
    if (fast) {
        unsigned acc = 0;
#pragma unroll 1
        for (uint32_t b = 0; b < fast_bytes / 1024; ++b) {  // untimed warm-up pass over the region
            unsigned r[8];
            issue<4, 0, 8>(r, fast_base + b * 1024u, lane_off);
            wait_all<4, 8>(r);
            acc += r[0];
        }
        const uint64_t c0 = wave_cycles();
#pragma unroll 1
        for (uint32_t b = 0; b < FAST_BATCHES; ++b) {
            const char * q = fast_base + (b * 8u * 128u) % fast_bytes;
            unsigned r[8];
            issue<4, 0, 8>(r, q, lane_off);
            wait_all<4, 8>(r);
            acc += r[0] + r[7];
        }
        const uint64_t c1 = wave_cycles();
        done = 1;
        if (threadIdx.x == 0) {
            out[0] = c1 - c0;
            out[1] = FAST_BATCHES;
            out[5] = pl;
            out[6] = acc;
        }
    } else if (slow) {
        uint64_t p = chase_start;
        uint32_t n = 0;
        const uint64_t c0 = wave_cycles();
        while (!done) {
            asm volatile("global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
                         "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
                         "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
                         "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0"
                         : "+v"(p)::"memory");
            n += 4;
        }
        const uint64_t c1 = wave_cycles();
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&slow_loads, n);
            atomicAdd(&slow_waves, 1u);
            atomicAdd((unsigned long long *) &slow_cyc, (unsigned long long) ((c1 - c0) / (n ? n : 1)));
        }
        if (p == 0) {
            out[7] = p;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out[2] = slow_loads;
        out[3] = slow_waves ? slow_cyc / slow_waves : 0;
        out[4] = slow_waves;
    }
}

// ---------------------------------------------------------------------------------------------------------------
struct Ctx {
    char * buf = nullptr;  // 1 GiB ramp
    char * flush = nullptr;
    uint64_t * out = nullptr;
    unsigned * sink = nullptr;
    bool fail = false;
};

__global__ void k_flush(const u4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const u4 v = p[i];
        acc += v.x;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static void do_flush(Ctx & c) {
    hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const u4 *) c.flush, (512u << 20) / 16, c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

// where: 0 memory (fresh lines, flushed first), 1 L2 (1 MiB region, warmed in the same launch by the first pass),
// 2 L0 (16 KiB region).
template <int W, int N>
static void inflight_one(Ctx & c, int where, size_t & region) {
    const uint32_t batch_bytes = N * 32 * W;
    uint32_t batches, wrap;
    if (where == 0) {
        batches = std::max<uint32_t>(64, (uint32_t) ((8u << 20) / batch_bytes));
        wrap = batches;
    } else {
        const uint32_t span = where == 1 ? (1u << 20) : (16u << 10);
        wrap = std::max<uint32_t>(1, span / batch_bytes);
        batches = std::max<uint32_t>(wrap * 8, 1024);
    }
    std::vector<double> cyc;
    uint64_t o[3] = {0, 0, 0};
    uint64_t bad = 0;
    for (int t = 0; t < 3; ++t) {
        if (where == 0) {
            if (region + (size_t) batches * batch_bytes > (1ull << 30)) {
                region = 0;
            }
            do_flush(c);
        }
        const char * base = c.buf + (where == 0 ? region : 0);
        hipLaunchKernelGGL((k_inflight<W, N>), dim3(1), dim3(32), 0, 0, base, (uint32_t) ((base - c.buf) / 4), batches,
                           wrap, where == 0 ? 0 : wrap, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        cyc.push_back((double) o[0] / batches);
        bad += o[1];
        if (where == 0) {
            region += (size_t) batches * batch_bytes;
        }
    }
    const Stats s = stats_of(cyc);
    const char * wn[] = {"memory", "L2", "L0"};
    std::printf("RESULT mem_inflight inflight %s W %d N %d cycles_per_batch %.1f (%.1f-%.1f) cycles_per_load %.2f "
                "bytes_per_cycle %.1f wrong_values %llu\n",
                wn[where], W, N, s.median, s.min, s.max, s.median / N, (double) batch_bytes / s.median,
                (unsigned long long) bad);
    std::fflush(stdout);
    if (bad) {
        std::printf("NOTE wrong values with N = %d: the wait returned before all loads arrived\n", N);
    }
}

template <int W, int... Ns>
static void inflight_list(Ctx & c, int where, size_t & region) {
    (inflight_one<W, Ns>(c, where, region), ...);
}

static void mode_inflight(Ctx & c) {
    mem::clocks("inflight-start");
    size_t region = 0;
    for (int where = 0; where < 3; ++where) {
        inflight_list<4, 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 63, 64, 72, 80, 96>(c, where, region);
        // N above 24 spills registers for 16-byte loads (measured in the .lto.s)
        inflight_list<16, 1, 2, 4, 8, 12, 16, 24>(c, where, region);
    }
    mem::clocks("inflight-end");
}

static void mode_waits(Ctx & c, bool brk) {
    mem::clocks("waits-start");
    // slow chain: random cycle through 1 GiB at 128-byte nodes (built over the ramp buffer's second half region)
    char * chase_buf;
    CHECK(hipMalloc(&chase_buf, 1ull << 30));
    const uint64_t start = mem::build_chase(chase_buf, (1ull << 30) / 128, 128, true, 23);
    // (slow waves on the fast wave's SIMD, slow waves on the other SIMD of the CU)
    const int cfgs[][2] = {{0, 0},  {1, 0},  {0, 1},  {15, 0}, {0, 4},  {0, 8},   {0, 10},  {0, 12},
                           {0, 14}, {0, 16}, {4, 4},  {8, 8},  {15, 8}, {12, 12}, {15, 16}};
    const int ncfg = sizeof cfgs / sizeof cfgs[0];
    double alone_l0 = 0, worst_ratio = 0, same1_ratio = 0;
    for (uint32_t fb : {16u << 10, 1u << 20}) {
        double alone = 0;
        for (int cfg = 0; cfg < ncfg; ++cfg) {
            std::vector<double> f, s;
            uint64_t o[8];
            for (int t = 0; t < 5; ++t) {
                do_flush(c);
                hipLaunchKernelGGL(k_waits, dim3(1), dim3(1024), 0, 0, (uint32_t) cfgs[cfg][0], (uint32_t) cfgs[cfg][1],
                                   (const char *) c.buf, fb, start, c.out);
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
                f.push_back((double) o[0] / o[1]);
                s.push_back((double) o[3]);
            }
            const Stats fs = stats_of(f), ss = stats_of(s);
            if (cfg == 0) {
                alone = fs.median;
                if (fb == (16u << 10)) {
                    alone_l0 = alone;
                }
            } else if (cfg == 1) {
                same1_ratio = std::max(same1_ratio, fs.median / alone);
            }
            if (cfg > 0) {
                worst_ratio = std::max(worst_ratio, fs.median / alone);
            }
            std::printf("RESULT mem_inflight waits fast_region_KiB %u slow_same_simd %d slow_other_simd %d "
                        "fast_cycles_per_batch_of_8 %.1f (%.1f-%.1f) ratio_to_alone %.3f slow_waves %llu "
                        "slow_cycles_per_load %.0f\n",
                        fb >> 10, cfgs[cfg][0], cfgs[cfg][1], fs.median, fs.min, fs.max, fs.median / alone,
                        (unsigned long long) o[4], ss.median);
            std::fflush(stdout);
        }
    }
    char t[200];
    // Two measured ratios carry the verdict: one slow wave on the same SIMD must leave the fast wave within 20 %,
    // and many slow waves must slow it at least 2x. The break claims the opposite of the second (that many slow
    // waves are also harmless), so it fails on the measured value, not on a constant.
    const bool one_harmless = same1_ratio > 0 && same1_ratio <= 1.20;
    const bool many_harmful = worst_ratio >= 2.0;
    const bool pass = alone_l0 > 0 && one_harmless && (brk ? worst_ratio <= 1.20 : many_harmful);
    std::snprintf(t, sizeof t,
                  "fast wave alone %.1f cycles per batch of 8 L0 loads; with one slow wave on its SIMD x%.3f "
                  "(must be <= 1.20); worst with many slow waves x%.2f (%s)",
                  alone_l0, same1_ratio, worst_ratio,
                  brk ? "deliberate break: claimed <= 1.20" : "must be >= 2.0");
    verdict("mem_inflight_waits", pass ? "PASS" : "FAIL", t);
    c.fail |= !pass;
    CHECK(hipFree(chase_buf));
    mem::clocks("waits-end");
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: mem_inflight <inflight|waits> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode %s%s\n", prop.name, prop.gcnArchName, mode.c_str(), brk ? " --break" : "");
    Ctx c;
    CHECK(hipMalloc(&c.buf, 1ull << 30));
    mem::fill(c.buf, 1ull << 30, mem::RAMP);
    CHECK(hipMalloc(&c.flush, 512u << 20));
    mem::fill(c.flush, 512u << 20, mem::RANDOM);
    CHECK(hipMalloc(&c.out, 64));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    if (mode == "inflight") {
        mode_inflight(c);
    } else if (mode == "waits") {
        mode_waits(c, brk);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::report_hottest();
    return c.fail ? 1 : 0;
}
