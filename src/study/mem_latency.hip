// Lane 2, latency probes: item 2b (one wave doing dependent loads against buffer size), the sharing of each cache
// level (which waves see each other's cached data), whether L2 contents survive a kernel launch, item 8's effect of
// cache hints on a later read and on a neighbour's cached data, and item 9 (software prefetch).
//
// Usage: mem_latency <chase|scope|persist|hints|hints_l2|prefetch> [--break]
//   chase    one wave (32 lanes, one address) follows a pointer cycle: each node holds the address of the next, one
//            node per 128 bytes, random order (Sattolo), increasing order, or random inside 64 KiB blocks visited in
//            order (same lines, few page changes: separates address translation from cache capacity). Every load
//            is followed by s_wait_loadcnt 0, so loads cannot overlap. Cycles from the wave's counter, nanoseconds
//            from REALTIME (100 MHz). Warm-up: one full cycle (up to 512 Ki nodes) before the timed 8192 steps;
//            above 64 MiB the random case is also timed after a 4096-step warm-up ("fresh", see mode_chase). The
//            final pointer is checked against the host's walk. Known answer: 137 ns per load at 512 MB random (lane 1,
//            core_counter, earlier overclocked profile). --break replaces the chain with independent loads (the
//            address does not depend on the previous load, one wait per 16): the verdict must FAIL.
//   scope    128 one-wave work-groups take turns (a flag in memory) chasing the same small buffer once each and then
//            again; each records its placement. A reader whose first pass is fast found the data in a cache that an
//            earlier reader filled: grouping by the nearest earlier reader (same CU, same WGP, same shader array,
//            same shader engine, other) shows which level is shared by whom.
//   persist  a buffer chased in one launch, then chased again in a second launch, against a second pass in the same
//            launch: does data stay in L0, L2 and the memory-side cache across kernel launches?
//   hints    for each load hint: (a) flush, the whole card reads buffer H once with the hint, then one wave chases H:
//            where did the hinted read leave H? (b) flush, H read twice normally, the whole card streams 256 MiB with
//            the hint, then one wave chases H: did the hinted stream evict H? H = 4 MiB (L2-sized) and 32 MiB.
//   hints_l2 the same two questions inside one launch with one wave (H = 1 MiB, stream 16 MiB), so the L2 is seen.
//   prefetch reads MODE, sets SCALAR_PREFETCH_EN (MODE bit 24) where asked, and times a latency-bound loop (one
//            128-byte load per 512-byte step through memory, each waited before the next) with and without
//            s_prefetch_data some steps ahead, and with a vector-load prefetch for comparison.
#include "mem_common.h"

#include <set>

// ---------------------------------------------------------------------------------------------------------------
// Chase kernels. out: [0] cycles, [1] REALTIME ticks, [2] final pointer, [3] placement.
#define CH1 "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
#define CH16 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1

__device__ __forceinline__ uint64_t chase(uint64_t p, uint32_t steps) {
#pragma unroll 1
    for (uint32_t i = 0; i < steps / 16; ++i) {
        asm volatile(CH16 : "+v"(p)::"memory");
    }
    return p;
}

__global__ __launch_bounds__(32) void k_chase(uint64_t start, uint32_t warm, uint32_t timed, uint64_t * out) {
    uint64_t p = chase(start, warm);
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
    p = chase(p, timed);
    const uint64_t c1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    const uint32_t pl = mem::placement();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = p;
        out[3] = pl;
    }
}

// Deliberate break: the same loads, but each reads from the start node into a scratch register, so they are
// independent and only every 16th is waited for.
__global__ __launch_bounds__(32) void k_chase_broken(uint64_t start, uint32_t warm, uint32_t timed, uint64_t * out) {
    (void) warm;
    uint64_t p = start, d = 0;
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
#define BR1 "global_load_b64 %1, %0, off\n\t"
#pragma unroll 1
    for (uint32_t i = 0; i < timed / 16; ++i) {
        asm volatile(BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 "s_wait_loadcnt 0"
                     : "+v"(p), "=&v"(d)::"memory");
    }
#undef BR1
    const uint64_t c1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = p ^ (d & 0);
        out[3] = mem::placement();
    }
}

// Turn-taking readers. The flag is read and written with relaxed device-scope atomics: an acquire would add a
// cache invalidate (global_inv) and wipe the very L0 contents this test looks for.
__global__ __launch_bounds__(32) void k_scope(uint64_t start, uint32_t steps, uint32_t * flag, uint64_t * out) {
    const uint32_t k = blockIdx.x;
    const uint32_t pl = mem::placement();
    const uint64_t t0 = mem::realtime();
    uint32_t timeout = 0;
    while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != k) {
        __builtin_amdgcn_s_sleep(4);
        if (mem::realtime() - t0 > 300000000ull) {  // 3 s
            timeout = 1;
            break;
        }
    }
    uint64_t p = start;
    const uint64_t c0 = wave_cycles();
    p = chase(p, steps);
    const uint64_t c1 = wave_cycles();
    p = chase(p, steps);
    const uint64_t c2 = wave_cycles();
    if (threadIdx.x == 0) {
        out[k * 4 + 0] = pl | ((uint64_t) timeout << 32);
        out[k * 4 + 1] = c1 - c0;
        out[k * 4 + 2] = c2 - c1;
        out[k * 4 + 3] = p;
    }
    __hip_atomic_store(flag, k + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Whole-card reads with a load hint (16 bytes per lane, one load and wait per element).
template <int H>
__global__ __launch_bounds__(256) void k_hread(const char * __restrict__ base, uint32_t n16, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
        typedef unsigned u4 __attribute__((ext_vector_type(4)));
        u4 v;
        const uint32_t off = i * 16;
#define X(j, n, s) \
    if constexpr (H == j) asm volatile("global_load_b128 %0, %1, %2" s "\n\ts_wait_loadcnt 0" : "=v"(v) : "v"(off), "s"(base) : "memory");
        MEM_HINTS(X)
#undef X
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Hints against L2, inside one launch (L2 does not keep data across launches; see mode persist). One wave:
// (a) chases H once with hinted loads, then once with regular loads (timed): where did the hinted pass leave H?
// (b) chases H twice regularly, streams `sbytes` with hinted 16-byte loads (8 in flight), then chases H (timed).
// out: [0] cycles of (a)'s timed pass, [1] cycles of (b)'s timed pass, [2] pointer, [3] stream sum.
template <int H>
__device__ __forceinline__ uint64_t chase_hint(uint64_t p, uint32_t steps) {
#pragma unroll 1
    for (uint32_t i = 0; i < steps / 4; ++i) {
#define X(j, n, s)                                                                                      \
    if constexpr (H == j) {                                                                             \
        asm volatile("global_load_b64 %0, %0, off" s "\n\ts_wait_loadcnt 0\n\t"                         \
                     "global_load_b64 %0, %0, off" s "\n\ts_wait_loadcnt 0\n\t"                         \
                     "global_load_b64 %0, %0, off" s "\n\ts_wait_loadcnt 0\n\t"                         \
                     "global_load_b64 %0, %0, off" s "\n\ts_wait_loadcnt 0"                               \
                     : "+v"(p)::"memory");                                                              \
    }
        MEM_HINTS(X)
#undef X
    }
    return p;
}

template <int H>
__global__ __launch_bounds__(32) void k_hint_l2(uint64_t start, uint32_t steps, const char * stream, uint32_t sbytes,
                                                uint64_t * out) {
    typedef unsigned u4 __attribute__((ext_vector_type(4)));
    uint64_t p = chase_hint<H>(start, steps);
    uint64_t c0 = wave_cycles();
    p = chase(p, steps);
    const uint64_t ta = wave_cycles() - c0;
    p = chase(p, steps);
    p = chase(p, steps);
    const uint32_t lane_off = (threadIdx.x % 32) * 16;
    unsigned acc = 0;
#pragma unroll 1
    for (uint32_t off = 0; off < sbytes; off += 8 * 512) {
        const char * q = stream + off;
        u4 v0, v1, v2, v3, v4, v5, v6, v7;
#define X(j, n, s)                                                                                                  \
    if constexpr (H == j) {                                                                                         \
        asm volatile("global_load_b128 %0, %8, %9 offset:0" s "\n\tglobal_load_b128 %1, %8, %9 offset:512" s         \
                     "\n\tglobal_load_b128 %2, %8, %9 offset:1024" s "\n\tglobal_load_b128 %3, %8, %9 offset:1536" s \
                     "\n\tglobal_load_b128 %4, %8, %9 offset:2048" s "\n\tglobal_load_b128 %5, %8, %9 offset:2560" s \
                     "\n\tglobal_load_b128 %6, %8, %9 offset:3072" s "\n\tglobal_load_b128 %7, %8, %9 offset:3584" s \
                     "\n\ts_wait_loadcnt 0"                                                                          \
                     : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3), "=&v"(v4), "=&v"(v5), "=&v"(v6), "=&v"(v7)      \
                     : "v"(lane_off), "s"(q)                                                                        \
                     : "memory");                                                                               \
    }
        MEM_HINTS(X)
#undef X
        acc += v0.x + v1.x + v2.x + v3.x + v4.x + v5.x + v6.x + v7.x;
    }
    c0 = wave_cycles();
    p = chase(p, steps);
    const uint64_t tb = wave_cycles() - c0;
    if (threadIdx.x == 0) {
        out[0] = ta;
        out[1] = tb;
        out[2] = p;
        out[3] = acc;
    }
}

// Prefetch test. VAR: 0 plain; 1 s_prefetch_data D steps ahead; 2 vector-load prefetch D steps ahead (load i is
// issued, then the prefetch for i + D, then s_wait_loadcnt 1, which waits for load i only because loads complete in
// order); 3 scalar loads (s_load_b32) without prefetch; 4 scalar loads with s_prefetch_data D steps ahead.
// MODEOP: 0 leave MODE, 1 set bit 24, 2 clear bit 24. out: [0] cycles, [1] MODE at start, [2] MODE after, [3] sink.
template <int VAR, int MODEOP>
__global__ __launch_bounds__(32) void k_pf(const char * base, uint32_t iters, uint32_t dist, uint64_t * out) {
    uint32_t m0, m1;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE)" : "=s"(m0));
    if (MODEOP == 1) {
        asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 24, 1), 1" ::: "memory");
    } else if (MODEOP == 2) {
        asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 24, 1), 0" ::: "memory");
    }
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE)" : "=s"(m1));
    const uint32_t lane_off = (threadIdx.x % 32) * 4;
    constexpr uint32_t STEP = 512;
    unsigned acc = 0, dummy = 0;
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t i = 0; i < iters; ++i) {
        const char * line = base + (size_t) i * STEP;
        const char * ahead = base + (size_t) (i + dist) * STEP;
        unsigned v;
        if (VAR == 0) {
            asm volatile("global_load_b32 %0, %1, %2\n\ts_wait_loadcnt 0" : "=v"(v) : "v"(lane_off), "s"(line) : "memory");
        } else if (VAR == 1) {
            asm volatile("s_prefetch_data %1, 0x0, null, 0\n\tglobal_load_b32 %0, %2, %3\n\ts_wait_loadcnt 0"
                         : "=v"(v)
                         : "s"(ahead), "v"(lane_off), "s"(line)
                         : "memory");
        } else if (VAR == 2) {
            asm volatile("global_load_b32 %0, %2, %3\n\tglobal_load_b32 %1, %2, %4\n\ts_wait_loadcnt 1"
                         : "=&v"(v), "=&v"(dummy)
                         : "v"(lane_off), "s"(line), "s"(ahead)
                         : "memory");
        } else if (VAR == 3) {
            uint32_t s;
            asm volatile("s_load_b32 %0, %1, 0x0\n\ts_wait_kmcnt 0" : "=s"(s) : "s"(line) : "memory");
            v = s;
        } else {
            uint32_t s;
            asm volatile("s_prefetch_data %1, 0x0, null, 0\n\ts_load_b32 %0, %2, 0x0\n\ts_wait_kmcnt 0"
                         : "=s"(s)
                         : "s"(ahead), "s"(line)
                         : "memory");
            v = s;
        }
        acc += v;
    }
    asm volatile("s_wait_loadcnt 0" ::: "memory");
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = m0;
        out[2] = m1;
        out[3] = acc + (dummy & 0);
    }
}

// ---------------------------------------------------------------------------------------------------------------
struct Ctx {
    char * buf = nullptr;  // chase buffer
    size_t bytes = 0;
    char * flush = nullptr;  // 512 MiB, read to push other data out of L2 and the memory-side cache
    uint64_t * out = nullptr;
    unsigned * sink = nullptr;
    bool fail = false;
};

static void do_flush(Ctx & c) {
    hipLaunchKernelGGL(k_hread<0>, dim3(512), dim3(256), 0, 0, c.flush, (uint32_t) ((512u << 20) / 16), c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

struct Lat {
    double cyc, ns;
    uint32_t pl;
    bool ok;
};

// Runs k_chase `trials` times; median cycles and ns per load; ok = final pointer as the host expects.
static Lat run_chase(Ctx & c, uint64_t start, uint64_t expect, uint32_t warm, uint32_t timed, int trials,
                     bool broken = false) {
    std::vector<double> cy, ns;
    bool ok = true;
    uint32_t pl = 0;
    for (int t = 0; t < trials; ++t) {
        if (broken) {
            hipLaunchKernelGGL(k_chase_broken, dim3(1), dim3(32), 0, 0, start, warm, timed, c.out);
        } else {
            hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, warm, timed, c.out);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[4];
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        cy.push_back((double) o[0] / timed);
        ns.push_back((double) o[1] * 10.0 / timed);
        ok &= o[2] == expect;
        pl = (uint32_t) o[3];
    }
    return {stats_of(cy).median, stats_of(ns).median, pl, ok};
}

static void mode_chase(Ctx & c, bool brk) {
    mem::clocks("chase-start");
    const std::vector<double> kib = {4,     8,     12,     16,     20,     24,     28,     30,     32,     34,
                                     36,    40,    48,     56,     64,     96,     128,    160,    192,    224,
                                     240,   256,   272,    288,    320,    384,    448,    512,    768,    1024,
                                     2048,  3072,  4096,   5120,   6144,   7168,   7680,   8192,   8704,   9216,
                                     10240, 12288, 16384,  24576,  32768,  40960,  49152,  57344,  61440,  65536,
                                     69632, 73728, 81920,  98304,  131072, 196608, 262144, 524288, 1048576, 4194304};
    const uint32_t timed = 8192;
    double lat512 = -1, lat16k = -1;
    bool all_ok = true;
    // seq 0: random over the whole buffer; 1: increasing order; 2: random inside 64 KiB blocks, blocks in order
    for (int seq = 0; seq < 3; ++seq) {
        for (double k : kib) {
            if ((brk || std::getenv("CHASE_ONLY_512M")) && (seq || k != 524288)) {
                continue;
            }
            if (seq && k > 1048576) {
                continue;
            }
            if (seq == 2 && k < 1024) {
                continue;
            }
            const size_t bytes = (size_t) (k * 1024);
            const size_t nodes = bytes / 128;
            std::vector<uint32_t> order;
            const uint64_t start = mem::build_chase(c.buf, nodes, 128, seq != 1, 7, &order, seq == 2 ? 512 : 0);
            uint32_t warm = (uint32_t) std::min<size_t>((nodes + 15) / 16 * 16, 524288);
            if (const char * w = std::getenv("CHASE_WARM")) {  // diagnostic override, multiple of 16
                warm = (uint32_t) std::strtoul(w, nullptr, 10) / 16 * 16;
            }
            const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[(warm + timed) % nodes] * 128;
            const Lat l = run_chase(c, start, expect, warm, timed, 3, brk);
            std::printf("RESULT mem_latency chase %s KiB %.0f cycles %.1f ns %.1f pointer %s%s\n",
                        seq == 1 ? "seq128" : seq == 2 ? "random_in_64k" : "random", k, l.cyc, l.ns, l.ok ? "ok" : "WRONG", brk ? " (BREAK)" : "");
            std::fflush(stdout);
            all_ok &= l.ok;
            // Above the memory-side cache a long random walk is slower than a short one (measured: 150 ns after
            // 4 Ki steps, 230 ns after 192 Ki or more). "fresh" repeats the point with a 4096-step warm-up, as lane 1
            // timed it (no warm-up there).
            if (!seq && k >= 65536) {
                const uint64_t ex = (uint64_t) c.buf + (uint64_t) order[(4096 + timed) % nodes] * 128;
                const Lat f = run_chase(c, start, ex, 4096, timed, 3, brk);
                std::printf("RESULT mem_latency chase random_fresh KiB %.0f cycles %.1f ns %.1f pointer %s%s\n", k,
                            f.cyc, f.ns, f.ok ? "ok" : "WRONG", brk ? " (BREAK)" : "");
                all_ok &= f.ok;
                if (k == 524288) {
                    lat512 = f.ns;
                }
            }
            if (!seq && k == 16) {
                lat16k = l.ns;
            }
        }
    }
    char t[300];
    const bool near = std::fabs(lat512 - 137.0) <= 0.15 * 137.0;
    const bool order = brk || (lat16k > 0 && lat16k < 0.5 * lat512);
    std::snprintf(t, sizeof t,
                  "512 MiB random chase (4096-step warm-up) %.1f ns per load; known answer 137 ns (lane 1, earlier profile) +-15%%: %s; "
                  "16 KiB faster than half of that: %s; final pointers %s%s",
                  lat512, near ? "yes" : "no", brk ? "n/a" : (order ? "yes" : "no"), all_ok ? "all ok" : "WRONG",
                  brk ? " (deliberate break: independent loads)" : "");
    const bool pass = near && order && all_ok;
    verdict("mem_latency_chase", pass ? "PASS" : "FAIL", t);
    c.fail |= !pass;
    mem::clocks("chase-end");
}

static const char * relation(uint32_t a, uint32_t b) {
    if (mem::pl_cu(a) == mem::pl_cu(b)) return "same_cu";
    if (mem::pl_wgp(a) == mem::pl_wgp(b)) return "same_wgp";
    if (mem::pl_sa(a) == mem::pl_sa(b)) return "same_sa";
    if (mem::pl_se(a) == mem::pl_se(b)) return "same_se";
    return "other_se";
}
static int rel_rank(const char * r) {
    const char * names[] = {"same_cu", "same_wgp", "same_sa", "same_se", "other_se"};
    for (int i = 0; i < 5; ++i) {
        if (std::strcmp(r, names[i]) == 0) return i;
    }
    return 5;
}

static void mode_scope(Ctx & c) {
    mem::clocks("scope-start");
    const int G = 128;
    uint32_t * flag;
    uint64_t * out;
    CHECK(hipMalloc(&flag, 4));
    CHECK(hipMalloc(&out, G * 4 * 8));
    for (double kib : {16.0, 128.0, 1024.0}) {
        const size_t nodes = (size_t) (kib * 1024) / 128;
        std::vector<uint32_t> order;
        const uint64_t start = mem::build_chase(c.buf, nodes, 128, true, 11, &order);
        const uint32_t steps = (uint32_t) ((nodes + 15) / 16 * 16);
        do_flush(c);
        CHECK(hipMemset(flag, 0, 4));
        hipLaunchKernelGGL(k_scope, dim3(G), dim3(32), 0, 0, start, steps, flag, out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> o(G * 4);
        CHECK(hipMemcpy(o.data(), out, o.size() * 8, hipMemcpyDeviceToHost));
        const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[(2 * steps) % nodes] * 128;
        std::vector<std::vector<double>> first(6), second(6);
        int timeouts = 0, wrong = 0;
        std::vector<uint32_t> seen;
        std::vector<int> distinct_cu;
        for (int k = 0; k < G; ++k) {
            const uint32_t pl = (uint32_t) o[k * 4];
            timeouts += (int) (o[k * 4] >> 32);
            wrong += o[k * 4 + 3] != expect;
            const char * best = "first";
            int br = 6;
            for (uint32_t q : seen) {
                const char * r = relation(pl, q);
                if (rel_rank(r) < br) {
                    br = rel_rank(r);
                    best = r;
                }
            }
            const double f = (double) o[k * 4 + 1] / steps, s = (double) o[k * 4 + 2] / steps;
            if (k > 0) {
                first[br].push_back(f);
                second[br].push_back(s);
            }
            if (std::getenv("SCOPE_DEBUG")) {
                std::printf("DATA scope KiB %.0f k %d se %u sa %u wgp %u cu %u simd %u nearest %s first %.1f second %.1f\n",
                            kib, k, mem::pl_se(pl), (pl >> 4) & 1, (pl >> 5) & 15, (pl >> 9) & 1, (pl >> 10) & 1, best,
                            f, s);
            }
            seen.push_back(pl);
        }
        std::set<uint32_t> cus, sas, ses;
        for (uint32_t q : seen) {
            cus.insert(mem::pl_cu(q));
            sas.insert(mem::pl_sa(q));
            ses.insert(mem::pl_se(q));
        }
        std::printf("RESULT mem_latency scope KiB %.0f readers %d cus %zu shader_arrays %zu shader_engines %zu "
                    "timeouts %d wrong_pointers %d warmer_first_pass %.1f cycles\n",
                    kib, G, cus.size(), sas.size(), ses.size(), timeouts, wrong, (double) o[1] / steps);
        const char * names[] = {"same_cu", "same_wgp", "same_sa", "same_se", "other_se"};
        for (int r = 0; r < 5; ++r) {
            if (first[r].empty()) {
                continue;
            }
            const Stats f = stats_of(first[r]), s = stats_of(second[r]);
            std::printf("RESULT mem_latency scope KiB %.0f nearest_earlier_reader %s n %zu first_pass_cycles %.1f "
                        "(%.1f-%.1f) second_pass_cycles %.1f (%.1f-%.1f)\n",
                        kib, names[r], first[r].size(), f.median, f.min, f.max, s.median, s.min, s.max);
        }
        c.fail |= timeouts > 0 || wrong > 0;
    }
    CHECK(hipFree(flag));
    CHECK(hipFree(out));
    mem::clocks("scope-end");
}

static void mode_persist(Ctx & c) {
    mem::clocks("persist-start");
    for (double kib : {16.0, 1024.0, 4096.0, 32768.0}) {
        const size_t nodes = (size_t) (kib * 1024) / 128;
        std::vector<uint32_t> order;
        const uint64_t start = mem::build_chase(c.buf, nodes, 128, true, 13, &order);
        const uint32_t steps = (uint32_t) ((nodes + 15) / 16 * 16);
        const uint32_t timed = std::min<uint32_t>(steps, 8192);
        auto at = [&](uint64_t k) { return (uint64_t) c.buf + (uint64_t) order[k % nodes] * 128; };
        std::vector<double> in, cross, cold;
        uint32_t pl_a = 0, pl_b = 0;
        for (int t = 0; t < 3; ++t) {
            // cold: flushed, first touch
            do_flush(c);
            cold.push_back(run_chase(c, start, at(timed), 0, timed, 1).cyc);
            // in-kernel: a full warming pass, then timed, same launch
            do_flush(c);
            in.push_back(run_chase(c, start, at(steps + timed), steps, timed, 1).cyc);
            // across launches: one launch warms (full pass), the next times
            do_flush(c);
            const Lat a = run_chase(c, start, at(steps + 16), steps, 16, 1);
            const Lat b = run_chase(c, at(steps + 16), at(steps + 16 + timed), 0, timed, 1);
            cross.push_back(b.cyc);
            pl_a = a.pl;
            pl_b = b.pl;
        }
        std::printf("RESULT mem_latency persist KiB %.0f cold %.1f same_launch %.1f next_launch %.1f cycles "
                    "(medians of 3; next launch on %s CU as the warming launch in the last trial)\n",
                    kib, stats_of(cold).median, stats_of(in).median, stats_of(cross).median,
                    mem::pl_cu(pl_a) == mem::pl_cu(pl_b) ? "the same" : "a different");
    }
    mem::clocks("persist-end");
}

template <int H>
static void hint_launch(Ctx & c, const char * p, size_t bytes) {
    hipLaunchKernelGGL(k_hread<H>, dim3(512), dim3(256), 0, 0, p, (uint32_t) (bytes / 16), c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

template <int H>
static void hint_tests(Ctx & c, double kib, uint64_t start, const std::vector<uint32_t> & order, size_t nodes,
                       char * stream) {
    const uint32_t timed = 8192;
    const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[timed % nodes] * 128;
    const size_t hb = (size_t) (kib * 1024);
    mem::cool_down();
    // (a) where does a hinted read leave H?
    do_flush(c);
    hint_launch<H>(c, c.buf, hb);
    const Lat a = run_chase(c, start, expect, 0, timed, 1);
    // (b) does a hinted 256 MiB stream evict H?
    do_flush(c);
    hint_launch<0>(c, c.buf, hb);
    hint_launch<0>(c, c.buf, hb);
    hint_launch<H>(c, stream, 256u << 20);
    const Lat b = run_chase(c, start, expect, 0, timed, 1);
    std::printf("RESULT mem_latency hints KiB %.0f hint %s after_hinted_read %.1f cycles after_hinted_stream %.1f "
                "cycles pointers %s\n",
                kib, mem::hint_name(H), a.cyc, b.cyc, a.ok && b.ok ? "ok" : "WRONG");
    std::fflush(stdout);
    c.fail |= !(a.ok && b.ok);
    if constexpr (H + 1 < mem::NHINTS) {
        hint_tests<H + 1>(c, kib, start, order, nodes, stream);
    }
}

static void mode_hints(Ctx & c) {
    mem::clocks("hints-start");
    char * stream;
    CHECK(hipMalloc(&stream, 256u << 20));
    mem::fill(stream, 256u << 20, mem::RANDOM);
    for (double kib : {4096.0, 32768.0}) {
        const size_t nodes = (size_t) (kib * 1024) / 128;
        std::vector<uint32_t> order;
        const uint64_t start = mem::build_chase(c.buf, nodes, 128, true, 17, &order);
        const uint32_t timed = 8192;
        const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[timed % nodes] * 128;
        // references: H in memory only (flushed), and H read twice normally (no stream after)
        do_flush(c);
        const Lat cold = run_chase(c, start, expect, 0, timed, 1);
        do_flush(c);
        hint_launch<0>(c, c.buf, nodes * 128);
        hint_launch<0>(c, c.buf, nodes * 128);
        const Lat warm = run_chase(c, start, expect, 0, timed, 1);
        std::printf("RESULT mem_latency hints KiB %.0f reference flushed %.1f cycles read_twice %.1f cycles\n", kib,
                    cold.cyc, warm.cyc);
        hint_tests<0>(c, kib, start, order, nodes, stream);
    }
    CHECK(hipFree(stream));
    mem::clocks("hints-end");
}

template <int VAR, int MODEOP>
static void pf_one(Ctx & c, const char * name, uint32_t dist, size_t & region) {
    const uint32_t iters = 2048;
    const size_t span = (size_t) (iters + dist + 1) * 512;
    if (region + span > c.bytes) {
        region = 0;
    }
    do_flush(c);
    std::vector<double> cy;
    uint64_t o[4];
    for (int t = 0; t < 3; ++t) {
        do_flush(c);
        hipLaunchKernelGGL((k_pf<VAR, MODEOP>), dim3(1), dim3(32), 0, 0, c.buf + region, iters, dist, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        cy.push_back((double) o[0] / iters);
        region += span;  // fresh lines every trial
    }
    const Stats s = stats_of(cy);
    std::printf("RESULT mem_latency prefetch %s dist %u cycles_per_step %.1f (%.1f-%.1f) MODE_before 0x%08x "
                "MODE_after 0x%08x prefetch_enable_bit %u\n",
                name, dist, s.median, s.min, s.max, (uint32_t) o[1], (uint32_t) o[2], (uint32_t) (o[2] >> 24) & 1u);
    std::fflush(stdout);
}

static void mode_prefetch(Ctx & c) {
    mem::clocks("prefetch-start");
    size_t region = 0;
    pf_one<0, 0>(c, "vector_plain", 0, region);
    for (uint32_t d : {4u, 16u}) {
        pf_one<1, 0>(c, "vector_sprefetch_mode_as_is", d, region);
        pf_one<1, 1>(c, "vector_sprefetch_mode_bit_set", d, region);
        pf_one<1, 2>(c, "vector_sprefetch_mode_bit_clear", d, region);
        pf_one<2, 0>(c, "vector_loadprefetch", d, region);
    }
    pf_one<3, 0>(c, "scalar_plain", 0, region);
    for (uint32_t d : {4u, 16u}) {
        pf_one<4, 0>(c, "scalar_sprefetch_mode_as_is", d, region);
        pf_one<4, 1>(c, "scalar_sprefetch_mode_bit_set", d, region);
        pf_one<4, 2>(c, "scalar_sprefetch_mode_bit_clear", d, region);
    }
    mem::clocks("prefetch-end");
}

template <int H>
static void hint_l2_one(Ctx & c, uint64_t start, uint64_t expect, uint32_t steps, char * stream, uint32_t sbytes) {
    std::vector<double> a, b;
    bool ok = true;
    for (int t = 0; t < 3; ++t) {
        do_flush(c);
        hipLaunchKernelGGL(k_hint_l2<H>, dim3(1), dim3(32), 0, 0, start, steps, stream, sbytes, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[4];
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        a.push_back((double) o[0] / steps);
        b.push_back((double) o[1] / steps);
        ok &= o[2] == expect;
    }
    std::printf("RESULT mem_latency hints_l2 KiB %u hint %s after_hinted_chase %.1f cycles after_hinted_stream_%uMiB "
                "%.1f cycles pointers %s\n",
                (unsigned) (steps * 128 / 1024), mem::hint_name(H), stats_of(a).median, sbytes >> 20,
                stats_of(b).median, ok ? "ok" : "WRONG");
    std::fflush(stdout);
    c.fail |= !ok;
    if constexpr (H + 1 < mem::NHINTS) {
        hint_l2_one<H + 1>(c, start, expect, steps, stream, sbytes);
    }
}

static void mode_hints_l2(Ctx & c) {
    mem::clocks("hints_l2-start");
    const uint32_t sbytes = 16u << 20;
    char * stream;
    CHECK(hipMalloc(&stream, sbytes));
    mem::fill(stream, sbytes, mem::RANDOM);
    // 1 MiB: a single wave's random walk is slower above about 2 MiB (mode chase), so H stays well below that
    const size_t nodes = (1u << 20) / 128;
    std::vector<uint32_t> order;
    const uint64_t start = mem::build_chase(c.buf, nodes, 128, true, 19, &order);
    const uint32_t steps = (uint32_t) nodes;
    // the kernel walks the cycle 5 times in all
    const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[(5ull * steps) % nodes] * 128;
    hint_l2_one<0>(c, start, expect, steps, stream, sbytes);
    CHECK(hipFree(stream));
    mem::clocks("hints_l2-end");
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: mem_latency <chase|scope|persist|hints|hints_l2|prefetch> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode %s%s\n", prop.name, prop.gcnArchName, mode.c_str(), brk ? " --break" : "");
    Ctx c;
    c.bytes = mode == "chase" ? (4ull << 30) : (1ull << 30);
    CHECK(hipMalloc(&c.buf, c.bytes));
    CHECK(hipMalloc(&c.flush, 512u << 20));
    mem::fill(c.flush, 512u << 20, mem::RANDOM);
    if (mode == "prefetch") {
        mem::fill(c.buf, c.bytes, mem::RANDOM);
    }
    CHECK(hipMalloc(&c.out, 64));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    if (mode == "chase") {
        mode_chase(c, brk);
    } else if (mode == "scope") {
        mode_scope(c);
    } else if (mode == "persist") {
        mode_persist(c);
    } else if (mode == "hints") {
        mode_hints(c);
    } else if (mode == "prefetch") {
        mode_prefetch(c);
    } else if (mode == "hints_l2") {
        mode_hints_l2(c);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::report_hottest();
    CHECK(hipFree(c.buf));
    CHECK(hipFree(c.flush));
    return c.fail ? 1 : 0;
}
