// Does the write half of a widen (FP8 weights read, FP16 written to a scratch buffer that is reused for every matrix) stay in
// the last-level cache? Same access shape as llama.cpp's widen kernel: read S bytes, write 2 S bytes. The source rotates through a
// 1 GiB pool (so its reads come from memory); the destination is either ONE buffer reused every call (as llama.cpp's scratch pool
// is) or rotates through the same 1 GiB pool (never reused while still cached). Also a read-only and a write-only kernel at the same size.
// Usage: widen_cache [S in MiB] [reps] [break|nocal]. Defaults 25 MiB (a 3B gate matrix) and 200 reps. "nocal" (round 3) skips check 1 and prints VERDICT widen_cache NOCAL.
//
// Checks (exit 1 if one fails; every number below it is only as good as these):
//  1. KNOWN ANSWER: before the rows, a read of the whole 1 GiB pool in one launch, 30 reps, must land within 10 % of the 706 GB/s
//     the study's bandwidth probe reads at the 1408 MHz memory clock (round 1, runs/20261118T094213Z-llm/probe-bw-plain.log:
//     705.0 and 705.7 GB/s; measurements/scheduling-and-clocks.md: best 706.4). A card at another memory clock, a broken
//     timing or a kernel that does not read what it claims fails here, and so does nothing else.
//  2. LAUNCH FLOOR: an empty kernel with the same grid, timed the same way (events around one launch), gives the per-launch
//     overhead. Rows where the floor is 10 % or more of the median are marked LAUNCH-BOUND, and a second rate with the floor
//     taken out (GBps_net, INFERRED: the floor is not exactly additive) is printed beside the raw one.
//  3. The deliberate break (third argument "break") runs the known answer on 8 MiB instead of 1 GiB: it must print
//     VERDICT known-answer FAIL and exit 1. scripts/study-llm.sh r2mallcheck runs it and requires that.
//  4. The ISA of the three loops is checked by scripts/mem_loopcheck.py against scripts/mem-expect/widen_cache.txt
//     (scripts/study-llm.sh r2mallbuild).
#include "harness.h"

#include <cmath>

__global__ void widen_kernel(const uint4 * __restrict__ src, uint4 * __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];  // 16 FP8 bytes in, 16 F16 values (32 bytes) out
        uint4 a = make_uint4(v.x & 0x00ff00ffu, (v.x >> 8) & 0x00ff00ffu, v.y & 0x00ff00ffu, (v.y >> 8) & 0x00ff00ffu);
        uint4 b = make_uint4(v.z & 0x00ff00ffu, (v.z >> 8) & 0x00ff00ffu, v.w & 0x00ff00ffu, (v.w >> 8) & 0x00ff00ffu);
        dst[2 * i] = a;
        dst[2 * i + 1] = b;
    }
}

__global__ void read_kernel(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) {
        *sink = acc;
    }
}

__global__ void write_kernel(uint4 * __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = make_uint4(1u, 2u, 3u, 4u);
    }
}

__global__ void empty_kernel(unsigned * __restrict__ sink) {
    if (sink == nullptr) {
        *sink = 0;
    }
}

static double g_floor_ms = 0.0;

// s_mib: the MiB read (or written, for the write-only rows); moved_mib: all bytes the row counts (3 S for a widen).
static void report(const char * name, const std::vector<double> & ms, double bytes, double s_mib) {
    const Stats s = stats_of(ms);
    const double floor_pct = 100.0 * g_floor_ms / s.median;
    const bool bound = floor_pct >= 10.0;
    const double net = s.median > g_floor_ms ? bytes / ((s.median - g_floor_ms) * 1e6) : 0.0;
    std::printf("RESULT %s S_MiB %.0f moved_MiB %.0f median_ms %.4f GBps %.1f best_GBps %.1f worst_GBps %.1f floor_pct %.1f GBps_net %.1f%s\n", name,
                s_mib, bytes / (1 << 20), s.median, bytes / (s.median * 1e6), bytes / (s.min * 1e6), bytes / (s.max * 1e6), floor_pct, net,
                bound ? " LAUNCH-BOUND" : "");
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const size_t s_bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 25ull) << 20;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    const bool do_break = argc > 3 && std::strcmp(argv[3], "break") == 0;
    const bool do_nocal = argc > 3 && std::strcmp(argv[3], "nocal") == 0;  // round 3, Q3: skip the 1 GiB known-answer read; the rows are then NOT trusted (comparison runs only)
    const size_t pool = 1ull << 30;
    const size_t slots = pool / s_bytes;
    const size_t n = s_bytes / sizeof(uint4);
    std::printf("device %s (%s), S %zu MiB, source pool %zu slots, %d reps\n", prop.name, prop.gcnArchName, s_bytes >> 20, slots, reps);
    uint8_t *src, *dst_pool, *dst_one;
    unsigned * sink;
    CHECK(hipMalloc(&src, pool));
    CHECK(hipMalloc(&dst_pool, 2 * pool));
    CHECK(hipMalloc(&dst_one, 2 * s_bytes));
    CHECK(hipMalloc(&sink, sizeof(unsigned)));
    CHECK(hipMemset(src, 0x3c, pool));
    CHECK(hipMemset(dst_pool, 0, 2 * pool));
    CHECK(hipMemset(dst_one, 0, 2 * s_bytes));
    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;
    // 1. known answer: the whole pool read in one launch (the bandwidth probe's 1 GiB read); the break reads 8 MiB instead
    if (do_nocal) {
        std::printf("VERDICT known-answer SKIPPED (nocal: the 1 GiB known-answer read is not run; rows are for the binary comparison of round 3, not trusted alone)\n");
    } else {
        const double want = 706.0, tol = 0.10;
        const size_t cal_bytes = do_break ? (8ull << 20) : pool;
        const Stats c = stats_of(time_ms([&] { read_kernel<<<grid, block>>>((const uint4 *) src, cal_bytes / sizeof(uint4), sink); }, 30));
        const double got = cal_bytes / (c.median * 1e6);
        const bool ok = std::fabs(got - want) <= tol * want;
        std::printf("RESULT known_answer_read MiB %zu median_ms %.4f GBps %.1f (expected %.0f +- %.0f %%)\n", cal_bytes >> 20, c.median, got, want, tol * 100);
        std::printf("VERDICT known-answer %s%s\n", ok ? "PASS" : "FAIL", do_break ? " (deliberate break: 8 MiB read, must FAIL)" : "");
        if (!ok) {
            std::printf("VERDICT widen_cache FAIL: the rows below are not trusted\n");
            return 1;
        }
    }
    // 2. launch floor: an empty kernel, same grid, same event bracket
    g_floor_ms = stats_of(time_ms([&] { empty_kernel<<<grid, block>>>(sink); }, reps)).median;
    std::printf("RESULT launch_floor median_ms %.4f (empty kernel, grid %d x %d)\n", g_floor_ms, grid, block);
    size_t k = 0;
    auto slot = [&] { return (k++) % slots; };
    report("widen_dst_reused", time_ms([&] { widen_kernel<<<grid, block>>>((const uint4 *) (src + slot() * s_bytes), (uint4 *) dst_one, n); }, reps), 3.0 * s_bytes, s_bytes / 1048576.0);
    k = 0;
    report("widen_dst_rotating", time_ms([&] { const size_t j = slot(); widen_kernel<<<grid, block>>>((const uint4 *) (src + j * s_bytes), (uint4 *) (dst_pool + j * 2 * s_bytes), n); }, reps), 3.0 * s_bytes, s_bytes / 1048576.0);
    k = 0;
    report("read_rotating", time_ms([&] { read_kernel<<<grid, block>>>((const uint4 *) (src + slot() * s_bytes), n, sink); }, reps), (double) s_bytes, s_bytes / 1048576.0);
    k = 0;
    report("write_reused", time_ms([&] { write_kernel<<<grid, block>>>((uint4 *) dst_one, 2 * n); }, reps), 2.0 * s_bytes, s_bytes / 1048576.0);
    k = 0;
    report("write_rotating", time_ms([&] { write_kernel<<<grid, block>>>((uint4 *) (dst_pool + slot() * 2 * s_bytes), 2 * n); }, reps), 2.0 * s_bytes, s_bytes / 1048576.0);
    CHECK(hipFree(src));
    CHECK(hipFree(dst_pool));
    CHECK(hipFree(dst_one));
    CHECK(hipFree(sink));
    if (do_nocal) {
        std::printf("VERDICT widen_cache NOCAL (no known answer was run)\n");
        return 0;
    }
    std::printf("VERDICT widen_cache PASS (known answer held; rows marked LAUNCH-BOUND carry the launch floor)\n");
    return 0;
}
