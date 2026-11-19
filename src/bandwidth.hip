// Device memory bandwidth: a read-only pass (what a decode GEMV does with its weights) and a copy, both with 16-byte
// loads in a grid-stride loop. Adapted from rdna3-fp8/examples/bandwidth/bandwidth.hip; same kernels and launch
// shape, so numbers compare across the two cards.
// Usage: bandwidth [MiB per buffer] [reps]. Keep the buffer well above the last-level cache (the script uses at least
// 4x the level-3 size from the card record).
#include "harness.h"

__global__ void read_kernel(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // The data is zero, so this never writes; it keeps the loads from being removed.
    if (acc == 0xdeadbeefu) {
        *sink = acc;
    }
}

__global__ void copy_kernel(const uint4 * __restrict__ src, uint4 * __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static void report(const char * name, const std::vector<double> & ms, double bytes) {
    const Stats s = stats_of(ms);
    std::printf("RESULT %s MiB %.0f median_ms %.4f GBps %.1f best_GBps %.1f worst_GBps %.1f\n", name,
                bytes / (1 << 20), s.median, bytes / (s.median * 1e6), bytes / (s.min * 1e6), bytes / (s.max * 1e6));
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const size_t bytes = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024ull) << 20;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    const size_t n = bytes / sizeof(uint4);
    std::printf("device %s (%s), %d CUs, buffer %zu MiB, %d reps\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount, bytes >> 20, reps);

    uint4 *src, *dst;
    unsigned * sink;
    CHECK(hipMalloc(&src, bytes));
    CHECK(hipMalloc(&dst, bytes));
    CHECK(hipMalloc(&sink, sizeof(unsigned)));
    CHECK(hipMemset(src, 0, bytes));
    CHECK(hipMemset(dst, 0, bytes));

    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;
    report("read", time_ms([&] { read_kernel<<<grid, block>>>(src, n, sink); }, reps), (double) bytes);
    report("copy", time_ms([&] { copy_kernel<<<grid, block>>>(src, dst, n); }, reps), 2.0 * bytes);

    CHECK(hipFree(src));
    CHECK(hipFree(dst));
    CHECK(hipFree(sink));
    return 0;
}
