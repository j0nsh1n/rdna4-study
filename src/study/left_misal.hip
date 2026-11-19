// Leftovers C, lane A, CH-5a: 16-byte loads (global_load_b128) from per-lane addresses that are 4-byte aligned but
// not 16-byte aligned. The method and readers of haz_memshape.hip (compiler-and-hazards item 5), with a run-time byte
// offset of 0, 4, 8 or 12 added to the base, so the compiler still emits the wide load (it assumes the vector type's
// alignment). Byte offsets that are not multiples of 4 are not tried: depending on the alignment mode they could
// fault, and a memory fault on this card is not worth the answer.
//
// Usage: left_misal misal [--break]      correctness and the single-wave cost (no whole-card load)
//        left_misal misal bw <offset> [tag]   one whole-card burst of 20 launches at that offset (fix round: one
//                                             burst per job, so the cool-down to 55 C happens outside the GPU lock)
//   correctness: n items, lane i reads 16 bytes at base + off + 16 i (one item in 8 then crosses a 128-byte line, one
//                in 16 a 256-byte line); every word must equal the input (verdict misal_bytes). --break compares
//                against the input shifted by one word: FAIL.
//   single wave: 64 independent 16-byte loads per pass, a cold pass after a 512 MiB flush and a warm pass (cycles
//                per load); whole card: a 1 GiB buffer, 4 items per thread, 20 launches per offset in 3 bursts
//                (about 35 ms each). Calibration (verdict misal_calib): offset 0 reproduces item 5: warm 13.9 cycles
//                per load +-10 % (mode misal), 632 GB/s +-5 % (verdict misal_bw_calib, mode bw 0). The cost of the
//                offsets is INFO.
// Thermal rule (fix round, as left_mem l0card): a whole-card burst starts only below 55 C (no waiting here: the job
// holds the GPU lock; a skipped burst is printed and the calibration then fails), and after a burst at 95 C or more
// no further burst starts. The script waits outside the lock until the hotspot is below 55 C before it starts the job.
#include "mem_common.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

typedef uint32_t v4u __attribute__((ext_vector_type(4)));

namespace {

__device__ __forceinline__ v4u rd(const char * base, size_t i) { return reinterpret_cast<const v4u *>(base)[i]; }

__global__ __launch_bounds__(256) void k_copy(const char * in, uint32_t * out, size_t off, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const v4u v = rd(in + off, i);
        out[(size_t) i * 4 + 0] = v.x;
        out[(size_t) i * 4 + 1] = v.y;
        out[(size_t) i * 4 + 2] = v.z;
        out[(size_t) i * 4 + 3] = v.w;
    }
}

__global__ __launch_bounds__(256) void k_bw(const char * in, uint32_t * out, size_t off, size_t q, uint32_t sentinel) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc = 0;
    for (int j = 0; j < 4; ++j) {
        const v4u v = rd(in + off, i + j * q);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == sentinel) out[0] = acc;
}

// the second pass reads through a run-time base (off2 = off), so the compiler cannot merge it with the first
__global__ __launch_bounds__(32) void k_cyc(const char * in, uint64_t * res, size_t off, size_t off2, uint32_t * sink) {
    const int lane = threadIdx.x;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint64_t cy[2];
    for (int pass = 0; pass < 2; ++pass) {
        const char * b = in + (pass ? off2 : off);
        asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));
        const uint64_t t0 = wave_cycles();
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            const v4u v = rd(b, (size_t) j * 32 + lane);
            a0 ^= v.x;
            a1 ^= v.y;
            a2 ^= v.z;
            a3 ^= v.w;
        }
        asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));
        cy[pass] = cycle_diff(t0, wave_cycles());
    }
    if (lane == 0) {
        res[0] = cy[0];
        res[1] = cy[1];
    }
    sink[lane] = a0 ^ a1 ^ a2 ^ a3;
}

__global__ __launch_bounds__(256) void k_flush(const uint4 * p, uint32_t * out, size_t n, uint32_t sentinel) {
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == sentinel) out[0] = acc;
}

uint32_t lcg(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return s ^ (s >> 15);
}

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

}  // namespace

// one whole-card burst at byte offset off: 2 warm-up and 20 timed launches over 1 GiB (about 35 ms)
static int burst(size_t off) {
    char * d_big;
    uint32_t * d_sink;
    CHECK(hipMalloc(&d_big, (1u << 30) + 64));
    CHECK(hipMalloc(&d_sink, 4096));
    CHECK(hipMemset(d_big, 0x5a, (1u << 30) + 64));
    CHECK(hipDeviceSynchronize());
    const double before = mem::hotspot_c();
    if (before >= 55) {
        std::printf("THERMAL %.0f C: offset %zu burst skipped\n", before, off);
        check("misal_bw_gate", false, "burst at offset %zu not run: hotspot %.0f C at its start (gate 55 C)", off, before);
        return fails ? 1 : 0;
    }
    const size_t items = (1u << 30) / 16 - 1, q = items / 4;
    auto ms = time_ms([&] { hipLaunchKernelGGL(k_bw, dim3((int) (q / 256)), dim3(256), 0, 0, d_big, d_sink, off, q, 0xdeadbeefu); }, 20, 2);
    const double after = mem::hotspot_c();
    std::sort(ms.begin(), ms.end());
    const double gbs = (double) q * 4 * 16 / (stats_of(ms).median * 1e6);
    std::printf("THERMAL offset %zu burst: hotspot %.0f C before, %.0f C after%s\n", off, before, after,
                after >= 95 ? " (95 C: the script's next whole-card job waits for 55 C)" : "");
    std::printf("RESULT misal bw offset %2zu: whole card %.1f GB/s (launches %.3f-%.3f ms, median %.3f)\n", off, gbs,
                ms.front(), ms.back(), stats_of(ms).median);
    if (off == 0) {
        check("misal_bw_calib", std::fabs(gbs / 632 - 1) <= 0.05, "offset 0: whole card %.1f GB/s (632 +-5 %%)", gbs);
    }
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    const bool brk = argc > 1 && std::strcmp(argv[argc - 1], "--break") == 0;  // first argument: the mode name
    require_arch();
    if (argc > 3 && std::strcmp(argv[2], "bw") == 0) {
        return burst((size_t) std::strtoul(argv[3], nullptr, 10));
    }
    const size_t offs[4] = {0, 4, 8, 12};
    // correctness
    const int n = 1 << 16;
    const size_t in_words = (size_t) n * 4 + 64;
    std::vector<uint32_t> h_in(in_words);
    uint32_t seed = 12345u;
    for (auto & w : h_in) w = lcg(seed);
    char * d_in;
    uint32_t * d_out;
    CHECK(hipMalloc(&d_in, in_words * 4));
    CHECK(hipMalloc(&d_out, (size_t) n * 16));
    CHECK(hipMemcpy(d_in, h_in.data(), in_words * 4, hipMemcpyHostToDevice));
    size_t bad_all = 0;
    for (size_t off : offs) {
        hipLaunchKernelGGL(k_copy, dim3(n / 256), dim3(256), 0, 0, d_in, d_out, off, n);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h((size_t) n * 4);
        CHECK(hipMemcpy(h.data(), d_out, h.size() * 4, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t k = 0; k < h.size(); ++k) bad += h[k] != h_in[off / 4 + k + (brk ? 1 : 0)];
        bad_all += bad;
        std::printf("RESULT misal correctness offset %zu: %zu of %zu words differ from the input\n", off, bad, h.size());
    }
    check("misal_bytes", bad_all == 0, "16-byte loads at offsets 0, 4, 8, 12 (one item in 8 crossing a 128-byte line): %zu "
          "words wrong%s", bad_all, brk ? " (deliberate break: reference shifted by one word)" : "");
    if (brk) return fails ? 1 : 0;
    // cost
    char * d_big;
    uint32_t * d_sink;
    uint64_t * d_res;
    CHECK(hipMalloc(&d_big, (1u << 30) + (512u << 20) + 64));
    CHECK(hipMalloc(&d_sink, 4096));
    CHECK(hipMalloc(&d_res, 16));
    CHECK(hipMemset(d_big, 0x5a, (1u << 30) + (512u << 20) + 64));
    double warm0 = 0;
    for (size_t off : offs) {
        // single wave
        std::vector<uint64_t> p1, p2;
        for (int rep = 0; rep < 9; ++rep) {
            hipLaunchKernelGGL(k_flush, dim3(4096), dim3(256), 0, 0, (const uint4 *) (d_big + (1u << 30)), d_sink,
                               (size_t) (512u << 20) / 16, 0xdeadbeefu);
            hipLaunchKernelGGL(k_cyc, dim3(1), dim3(32), 0, 0, d_big, d_res, off, off, d_sink + 64);
            CHECK(hipGetLastError());
            uint64_t h[2];
            CHECK(hipMemcpy(h, d_res, 16, hipMemcpyDeviceToHost));
            p1.push_back(h[0]);
            p2.push_back(h[1]);
        }
        std::sort(p1.begin(), p1.end());
        std::sort(p2.begin(), p2.end());
        std::printf("RESULT misal offset %2zu: single wave cold %.1f cycles per load, warm %.1f\n", off, p1[4] / 64.0,
                    p2[4] / 64.0);
        std::fflush(stdout);
        if (off == 0) {
            warm0 = p2[4] / 64.0;
        }
    }
    check("misal_calib", std::fabs(warm0 / 13.9 - 1) <= 0.10,
          "offset 0: warm %.1f cycles per load (compiler-and-hazards item 5: 13.9 +-10 %%)", warm0);
    return fails ? 1 : 0;
}
