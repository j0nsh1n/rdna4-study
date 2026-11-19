// Timing study, part B: when do slow waves on the neighbouring SIMD slow a fast wave (mem-system section 7: 10 or more
// slow waves, 4x on L0 data; 8 nothing)? Is the trigger a fixed number of outstanding misses (a shared queue) or a fixed
// number of waves, and does it need address-translation misses?
//
// Usage: tim_slow waits [--break]       (build with -mcumode: one work-group of 32 waves sits on one CU, 16 per SIMD)
//   waits  The "fast" wave (rank 0 on SIMD A) does batches of 8 independent loads that hit L0 (16 KiB). Slow waves run
//          ONLY on the other SIMD. Each slow wave holds C independent pointer chains (C = 1, 2 or 4; each lane holds
//          the same addresses, so one 128-byte line per load), issues one load per chain, waits for all, and repeats, so
//          outstanding misses = slow waves x C. Every chain starts at its own node of the cycle (spread evenly along it),
//          so no two chains read the same line at the same time; one extra configuration shares one start node for
//          comparison with mem_inflight, where all waves did. Footprint of the chains: 1 GiB random (every load also a
//          translation miss), 16 MiB random (translations stay cached after the first touch; about half the loads still
//          miss the 8 MiB L2), 1 MiB random (L2 hits). The fast wave's sum of loaded words is checked against the host.
//          Verdict: calibration (fast wave alone 159 cycles per batch of 8, +-10 %) and 12 slow waves with one chain on
//          1 GiB give at least 2x slowdown. --break claims the fast wave alone is 3x faster than measured: must FAIL.
#include "mem_common.h"

#include <unistd.h>

constexpr int FAST_BATCHES = 4096;
constexpr uint32_t FAST_BYTES = 16 << 10;
// The region size is a kernel argument, as in mem_inflight: with a compile-time power of two the modulo becomes a mask
// and the batch costs about 121 instead of 159 cycles (the known value this probe is calibrated against).
constexpr int WARM_PASSES = 4;  // also gives the slow waves time to fill their loads before the timed region
constexpr int REPS = 5;
constexpr int NSTART = 64;

#define LD1 "global_load_b64 %0, %0, off\n\t"
#define LD2 "global_load_b64 %1, %1, off\n\t"
#define LD3 "global_load_b64 %2, %2, off\n\t"
#define LD4 "global_load_b64 %3, %3, off\n\t"

template <int K>
__device__ __forceinline__ void ldk(unsigned & r, const char * base, uint32_t lane_off) {
    asm volatile("global_load_b32 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 128) : "memory");
}

__device__ __forceinline__ void fast_batch(unsigned (&r)[8], const char * q, uint32_t lane_off) {
    ldk<0>(r[0], q, lane_off);
    ldk<1>(r[1], q, lane_off);
    ldk<2>(r[2], q, lane_off);
    ldk<3>(r[3], q, lane_off);
    ldk<4>(r[4], q, lane_off);
    ldk<5>(r[5], q, lane_off);
    ldk<6>(r[6], q, lane_off);
    ldk<7>(r[7], q, lane_off);
    asm volatile("s_wait_loadcnt 0" ::: "memory");
    for (int k = 0; k < 8; ++k) {
        asm volatile("" : "+v"(r[k]));
    }
}

// One round of a slow wave: C independent loads, one wait.
template <int C>
__device__ __forceinline__ void slow_round(uint64_t (&p)[C]) {
    if constexpr (C == 1) {
        asm volatile(LD1 "s_wait_loadcnt 0" : "+v"(p[0])::"memory");
    } else if constexpr (C == 2) {
        asm volatile(LD1 LD2 "s_wait_loadcnt 0" : "+v"(p[0]), "+v"(p[1])::"memory");
    } else {
        asm volatile(LD1 LD2 LD3 LD4 "s_wait_loadcnt 0" : "+v"(p[0]), "+v"(p[1]), "+v"(p[2]), "+v"(p[3])::"memory");
    }
}

// out: [0] fast cycles, [1] fast REALTIME ticks, [2] slow waves seen, [3] mean slow cycles per round, [4] fast sum,
// [5] slow rounds done, [6] placement of the fast wave.
template <int C>
__global__ __launch_bounds__(1024) void k_slow(uint32_t n_slow, const char * fast_base, uint32_t fast_bytes,
                                               const uint64_t * starts, uint32_t same_start, const char * warm_base,
                                               uint32_t warm_bytes, uint64_t * out) {
    __shared__ uint32_t rank_ctr[2];
    __shared__ uint32_t simd_a;
    __shared__ volatile uint32_t done;
    __shared__ uint32_t slow_rounds, slow_waves;
    __shared__ uint64_t slow_cyc;
    // Every wave of the group reads the footprint once (before the roles are split), so a 1 MiB footprint sits in L2 and
    // a 16 MiB one has its translations cached when the timed region starts. 0 bytes: nothing (the 1 GiB footprint).
    uint32_t wacc = 0;
    for (uint32_t off = threadIdx.x * 16u; off < warm_bytes; off += 1024u * 16u) {
        wacc += ((const uint4 *) (warm_base + off))->x;
    }
    if (wacc == 0x9e3779b9u) {
        out[8] = wacc;
    }
    const uint32_t pl = mem::placement();
    const uint32_t simd = (pl >> 10) & 1u;
    if (threadIdx.x == 0) {
        rank_ctr[0] = rank_ctr[1] = 0;
        simd_a = simd;
        done = 0;
        slow_rounds = slow_waves = 0;
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
    const bool slow = !same && rank < n_slow;
    const uint32_t lane_off = (threadIdx.x % 32) * 4;
    if (fast) {
        unsigned acc = 0;
#pragma unroll 1
        for (int w = 0; w < WARM_PASSES; ++w) {
#pragma unroll 1
            for (uint32_t b = 0; b < fast_bytes / 1024; ++b) {
                unsigned r[8];
                fast_batch(r, fast_base + b * 1024u, lane_off);
                acc += r[0];
            }
        }
        const uint64_t t0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
#pragma unroll 1
        for (uint32_t b = 0; b < FAST_BATCHES; ++b) {
            unsigned r[8];
            fast_batch(r, fast_base + (b * 8u * 128u) % fast_bytes, lane_off);
            acc += r[0] + r[7];
        }
        const uint64_t c1 = wave_cycles();
        const uint64_t t1 = mem::realtime();
        done = 1;
        if (threadIdx.x == 0) {
            out[0] = c1 - c0;
            out[1] = t1 - t0;
            out[4] = acc;
            out[6] = pl;
        }
    } else if (slow) {
        uint64_t p[C];
        for (int c = 0; c < C; ++c) {
            p[c] = starts[same_start ? 0 : (rank * C + c) % NSTART];
        }
        uint32_t n = 0;
        const uint64_t c0 = wave_cycles();
        while (!done) {
            slow_round<C>(p);
            ++n;
        }
        const uint64_t c1 = wave_cycles();
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&slow_rounds, n);
            atomicAdd(&slow_waves, 1u);
            atomicAdd((unsigned long long *) &slow_cyc, (unsigned long long) ((c1 - c0) / (n ? n : 1)));
        }
        uint64_t x = 0;
        for (int c = 0; c < C; ++c) {
            x |= p[c];
        }
        if (x == 0) {
            out[7] = x;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out[2] = slow_waves;
        out[3] = slow_waves ? slow_cyc / slow_waves : 0;
        out[5] = slow_rounds;
    }
}

__global__ void k_flush(const uint4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        acc += p[i].x;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

struct Foot {
    const char * name;
    size_t bytes;
    uint32_t warm;  // bytes every wave reads before the timed region
    char * buf = nullptr;
    uint64_t * starts = nullptr;  // device array of NSTART start addresses
};

struct Ctx {
    char * fast = nullptr;
    char * flush = nullptr;
    uint64_t * out = nullptr;
    unsigned * sink = nullptr;
    bool bad = false;
};

// The flush is the one whole-card launch of this probe (512 groups reading 512 MiB, about 1 ms, about 400 times).
// It runs near the full memory bandwidth but only for about a millisecond, and it gets the study's gate: start below 60 C, stop the program at 95 C.
static void thermal_gate() {
    double t = mem::hotspot_c();
    for (int i = 0; t >= 60.0 && i < 600; ++i) {
        usleep(100000);
        t = mem::hotspot_c();
    }
    if (t >= 95.0) {
        std::printf("THERMAL hotspot %.0f C: stopping\n", t);
        std::exit(1);
    }
    if (t >= 60.0) {
        std::printf("THERMAL hotspot still %.0f C after a 60 s wait: continuing (below the 95 C stop)\n", t);
    }
    mem::hottest() = std::max(mem::hottest(), t);
}

static void do_flush(Ctx & c) {
    thermal_gate();
    hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const uint4 *) c.flush, (512u << 20) / 16, c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

static void build_foot(Foot & f, uint32_t seed) {
    CHECK(hipMalloc(&f.buf, f.bytes));
    const size_t nodes = f.bytes / 128;
    std::vector<uint32_t> order;
    mem::build_chase(f.buf, nodes, 128, true, seed, &order);
    std::vector<uint64_t> st(NSTART);
    for (int i = 0; i < NSTART; ++i) {
        st[i] = (uint64_t) f.buf + (uint64_t) order[(size_t) i * (nodes / NSTART)] * 128;
    }
    CHECK(hipMalloc(&f.starts, NSTART * 8));
    CHECK(hipMemcpy(f.starts, st.data(), NSTART * 8, hipMemcpyHostToDevice));
}

// Expected sum of words the fast wave's lane 0 loads (the data is a ramp of 32-bit words).
static uint32_t fast_expect() {
    uint32_t e = 0;
    for (int w = 0; w < WARM_PASSES; ++w) {
        for (uint32_t b = 0; b < FAST_BYTES / 1024; ++b) {
            e += b * 1024u / 4;
        }
    }
    for (uint32_t b = 0; b < (uint32_t) FAST_BATCHES; ++b) {
        const uint32_t w0 = ((b * 8u * 128u) % FAST_BYTES) / 4;
        e += w0 + (w0 + 7 * 32);
    }
    return e;
}

struct Res {
    double cyc, cmin, cmax, ns, nmin, nmax, slow_cyc;
    uint64_t waves;
    bool ok;
};

static Res run_cfg(Ctx & c, Foot & f, uint32_t n_slow, int C, uint32_t same_start) {
    std::vector<double> cy, ns, sc;
    uint64_t o[16] = {};
    bool ok = true;
    const uint32_t expect = fast_expect();
    for (int t = 0; t < REPS; ++t) {
        do_flush(c);
        if (C == 1) {
            hipLaunchKernelGGL(k_slow<1>, dim3(1), dim3(1024), 0, 0, n_slow, (const char *) c.fast, FAST_BYTES, f.starts, same_start, f.buf, f.warm, c.out);
        } else if (C == 2) {
            hipLaunchKernelGGL(k_slow<2>, dim3(1), dim3(1024), 0, 0, n_slow, (const char *) c.fast, FAST_BYTES, f.starts, same_start, f.buf, f.warm, c.out);
        } else {
            hipLaunchKernelGGL(k_slow<4>, dim3(1), dim3(1024), 0, 0, n_slow, (const char *) c.fast, FAST_BYTES, f.starts, same_start, f.buf, f.warm, c.out);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        cy.push_back((double) o[0] / FAST_BATCHES);
        ns.push_back((double) o[1] * 10.0 / FAST_BATCHES);
        sc.push_back((double) o[3]);
        ok &= (uint32_t) o[4] == expect && o[2] == n_slow;
    }
    const Stats a = stats_of(cy), b = stats_of(ns), s = stats_of(sc);
    return {a.median, a.min, a.max, b.median, b.min, b.max, s.median, o[2], ok};
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2 || std::strcmp(argv[1], "waits") != 0) {
        std::fprintf(stderr, "usage: tim_slow waits [--break]\n");
        return 2;
    }
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode waits%s\n", prop.name, prop.gcnArchName, brk ? " --break" : "");
    Ctx c;
    CHECK(hipMalloc(&c.fast, 64 << 10));
    mem::fill(c.fast, 64 << 10, mem::RAMP);
    CHECK(hipMalloc(&c.flush, 512u << 20));
    mem::fill(c.flush, 512u << 20, mem::RANDOM);
    CHECK(hipMalloc(&c.out, 256));
    CHECK(hipMemset(c.out, 0, 256));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    Foot foots[3] = {{"1GiB", 1ull << 30, 0}, {"16MiB", 16ull << 20, 16u << 20}, {"1MiB", 1ull << 20, 1u << 20}};
    for (int i = 0; i < 3; ++i) {
        build_foot(foots[i], 23 + i);
    }
    mem::clocks("tim_slow-start");

    const Res alone = run_cfg(c, foots[0], 0, 1, 0);
    std::printf("RESULT tim_slow waits footprint - slow_waves 0 chains 0 outstanding 0 fast_cycles_per_batch_of_8 %.1f (%.1f-%.1f) "
                "ns %.1f (%.1f-%.1f) ratio 1.000 fast_sum %s\n",
                alone.cyc, alone.cmin, alone.cmax, alone.ns, alone.nmin, alone.nmax, alone.ok ? "ok" : "WRONG");
    bool all_ok = alone.ok;
    double ratio12 = 0;
    const double claimed = brk ? alone.cyc / 3.0 : alone.cyc;

    std::vector<std::pair<int, int>> cfgs;  // (slow waves, chains)
    for (int w : {6, 8, 9, 10, 12}) cfgs.push_back({w, 1});
    for (int w : {2, 3, 4, 5}) cfgs.push_back({w, 1});
    for (int C : {2, 4}) {
        for (int w : {2, 3, 4, 5, 6}) cfgs.push_back({w, C});
    }
    const int nfoot = brk ? 1 : 3;
    for (int fi = 0; fi < nfoot; ++fi) {
        for (auto [w, C] : cfgs) {
            if (brk && !(w == 12 && C == 1)) {
                continue;
            }
            const Res r = run_cfg(c, foots[fi], (uint32_t) w, C, 0);
            all_ok &= r.ok;
            std::printf("RESULT tim_slow waits footprint %s slow_waves %d chains %d outstanding %d fast_cycles_per_batch_of_8 %.1f "
                        "(%.1f-%.1f) ns %.1f (%.1f-%.1f) ratio %.3f slow_cycles_per_round %.0f seen_waves %llu fast_sum %s\n",
                        foots[fi].name, w, C, w * C, r.cyc, r.cmin, r.cmax, r.ns, r.nmin, r.nmax, r.cyc / alone.cyc,
                        r.slow_cyc, (unsigned long long) r.waves, r.ok ? "ok" : "WRONG");
            std::fflush(stdout);
            if (fi == 0 && w == 12 && C == 1) {
                ratio12 = r.cyc / alone.cyc;
            }
        }
    }
    if (!brk) {
        // all waves on one start node, as in mem_inflight (equal addresses at the same moment)
        for (int w : {8, 10, 12}) {
            const Res r = run_cfg(c, foots[0], (uint32_t) w, 1, 1);
            all_ok &= r.ok;
            std::printf("RESULT tim_slow waits footprint 1GiB_same_start slow_waves %d chains 1 outstanding %d "
                        "fast_cycles_per_batch_of_8 %.1f (%.1f-%.1f) ns %.1f (%.1f-%.1f) ratio %.3f slow_cycles_per_round %.0f "
                        "seen_waves %llu fast_sum %s\n",
                        w, w, r.cyc, r.cmin, r.cmax, r.ns, r.nmin, r.nmax, r.cyc / alone.cyc, r.slow_cyc,
                        (unsigned long long) r.waves, r.ok ? "ok" : "WRONG");
            std::fflush(stdout);
        }
    }
    const bool cal_ok = std::fabs(claimed - 159.0) <= 15.9;
    const bool slow_ok = ratio12 >= 2.0;
    char t[300];
    std::snprintf(t, sizeof t,
                  "fast wave alone %.1f cycles per batch of 8 L0 loads%s, known 159 +-10%%: %s; 12 slow waves x 1 chain on "
                  "1 GiB: x%.2f (must be >= 2.0): %s; fast sums and wave counts %s",
                  claimed, brk ? " (deliberate break: claimed 3x faster than measured)" : "", cal_ok ? "yes" : "no", ratio12,
                  slow_ok ? "yes" : "no", all_ok ? "all ok" : "WRONG");
    const bool pass = cal_ok && slow_ok && all_ok;
    verdict("tim_slow_waits", pass ? "PASS" : "FAIL", t);
    mem::clocks("tim_slow-end");
    mem::report_hottest();
    return pass ? 0 : 1;
}
