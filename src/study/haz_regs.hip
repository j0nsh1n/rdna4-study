// haz_regs: how the register count of a tiled matrix kernel (one wave, MT x NT output fragments of 16x16, K loop) grows
// with the tile, where it spills to scratch, and what a spill costs per K step on one wave alone.
// VGPR / scratch / spill-instruction counts are read from the -save-temps listing by scripts/mem_loopcheck.py
// (haz-expect/haz_regs.txt); correctness and cycles come from the card.
// Run: haz_regs [break]   ("break" checks one result against a deliberately wrong reference and must report FAIL)
#include "harness.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int KMAX = 128;                // K steps of 16 in the largest run
constexpr int KROW = 16 * KMAX;          // row stride of A and of the transposed B, in halves

// A is M x KROW row-major, Bt is N x KROW (B transposed), so a lane's 8 K-values of its row/column are 16 contiguous
// bytes: one global_load_b128 per fragment. D is M x N floats. Lane l feeds row/column l & 15 with K half l >> 4.
// MODE 0: default loop, 1: `#pragma unroll 4`.
template <int MT, int NT, int MODE>
__device__ __forceinline__ void tile_body(const _Float16 * A, const _Float16 * Bt, float * D, int ks,
                                          unsigned long long * cyc) {
    const int lane = threadIdx.x;
    const int rl = lane & 15, kh = lane >> 4;
    v8f acc[MT][NT];
    for (int i = 0; i < MT; ++i)
        for (int j = 0; j < NT; ++j)
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    const unsigned long long t0 = wave_cycles();
    if constexpr (MODE == 1) {
#pragma unroll 4
        for (int k = 0; k < ks; ++k) {
            v8h a[MT], b[NT];
            for (int i = 0; i < MT; ++i) a[i] = *reinterpret_cast<const v8h *>(A + (size_t)(16 * i + rl) * KROW + 16 * k + 8 * kh);
            for (int j = 0; j < NT; ++j) b[j] = *reinterpret_cast<const v8h *>(Bt + (size_t)(16 * j + rl) * KROW + 16 * k + 8 * kh);
            for (int i = 0; i < MT; ++i)
                for (int j = 0; j < NT; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    } else {
        for (int k = 0; k < ks; ++k) {
            v8h a[MT], b[NT];
            for (int i = 0; i < MT; ++i) a[i] = *reinterpret_cast<const v8h *>(A + (size_t)(16 * i + rl) * KROW + 16 * k + 8 * kh);
            for (int j = 0; j < NT; ++j) b[j] = *reinterpret_cast<const v8h *>(Bt + (size_t)(16 * j + rl) * KROW + 16 * k + 8 * kh);
            for (int i = 0; i < MT; ++i)
                for (int j = 0; j < NT; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    }
    // Keeps every result live here so the matrix instructions cannot be moved past the second counter read.
    for (int i = 0; i < MT; ++i)
        for (int j = 0; j < NT; ++j) asm volatile("" : "+v"(acc[i][j]));
    const unsigned long long t1 = wave_cycles();
    if (lane == 0) *cyc = cycle_diff(t0, t1);

    const int N = 16 * NT;
    for (int i = 0; i < MT; ++i)
        for (int j = 0; j < NT; ++j)
            for (int e = 0; e < 8; ++e) D[(size_t)(16 * i + 8 * kh + e) * N + 16 * j + rl] = acc[i][j][e];
}

template <int MT, int NT> __global__ __launch_bounds__(32) void regs_kernel(const _Float16 * A, const _Float16 * Bt, float * D, int ks, unsigned long long * cyc) {
    tile_body<MT, NT, 0>(A, Bt, D, ks, cyc);
}
template <int MT, int NT> __global__ __launch_bounds__(32) __attribute__((amdgpu_waves_per_eu(16))) void regs_kernel_w16(const _Float16 * A, const _Float16 * Bt, float * D, int ks, unsigned long long * cyc) {
    tile_body<MT, NT, 0>(A, Bt, D, ks, cyc);
}
template <int MT, int NT> __global__ __launch_bounds__(32) __attribute__((amdgpu_waves_per_eu(12))) void regs_kernel_w12(const _Float16 * A, const _Float16 * Bt, float * D, int ks, unsigned long long * cyc) {
    tile_body<MT, NT, 0>(A, Bt, D, ks, cyc);
}
template <int MT, int NT> __global__ __launch_bounds__(32) void regs_kernel_u4(const _Float16 * A, const _Float16 * Bt, float * D, int ks, unsigned long long * cyc) {
    tile_body<MT, NT, 1>(A, Bt, D, ks, cyc);
}

typedef void (*KFn)(const _Float16 *, const _Float16 *, float *, int, unsigned long long *);

struct Case {
    std::string name;
    int MT, NT;
    KFn fn;
};

static int g_fails = 0;
static bool g_break = false;

static void report(const std::string & name, bool ok, const std::string & text) {
    verdict(("haz_regs_" + name).c_str(), ok ? "PASS" : "FAIL", text.c_str());
    if (!ok) ++g_fails;
}

static float h2f(uint16_t u) {
    _Float16 h;
    std::memcpy(&h, &u, 2);
    return (float) h;
}

int main(int argc, char ** argv) {
    g_break = argc > 1 && std::strcmp(argv[1], "break") == 0;
    const hipDeviceProp_t prop = require_arch();
    std::printf("device 0: %s (%s)%s\n", prop.name, prop.gcnArchName, g_break ? "  [break mode: reference is shifted on purpose]" : "");

    const int MMAX = 16 * 8, NMAX = 16 * 8;
    // Small integers -3..3: every product and sum is exact in fp16 inputs / fp32 accumulation, and in double.
    std::vector<_Float16> hA((size_t)MMAX * KROW), hB((size_t)NMAX * KROW);
    uint32_t x = 12345u;
    auto rnd = [&]() {
        x = x * 1664525u + 1013904223u;
        return (int) ((x >> 16) % 7) - 3;
    };
    for (auto & v : hA) v = (_Float16) rnd();
    for (auto & v : hB) v = (_Float16) rnd();
    _Float16 *dA, *dB;
    float * dD;
    unsigned long long * dC;
    CHECK(hipMalloc(&dA, hA.size() * 2));
    CHECK(hipMalloc(&dB, hB.size() * 2));
    CHECK(hipMalloc(&dD, (size_t)MMAX * NMAX * 4));
    CHECK(hipMalloc(&dC, 8));
    CHECK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));

    std::vector<Case> cases = {
        {"2x2", 2, 2, regs_kernel<2, 2>},       {"4x4", 4, 4, regs_kernel<4, 4>},
        {"4x6", 4, 6, regs_kernel<4, 6>},       {"4x8", 4, 8, regs_kernel<4, 8>},
        {"8x8", 8, 8, regs_kernel<8, 8>},       {"4x4_w16", 4, 4, regs_kernel_w16<4, 4>},
        {"4x4_w12", 4, 4, regs_kernel_w12<4, 4>}, {"4x4_u4", 4, 4, regs_kernel_u4<4, 4>},
    };

    auto launch = [&](const Case & c, int ks) {
        c.fn<<<1, 32>>>(dA, dB, dD, ks, dC);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        unsigned long long cy = 0;
        CHECK(hipMemcpy(&cy, dC, 8, hipMemcpyDeviceToHost));
        return cy;
    };

    for (const Case & c : cases) {
        const int M = 16 * c.MT, N = 16 * c.NT;
        bool ok = true;
        long bad = 0;
        for (int ks : {64, 128}) {
            CHECK(hipMemset(dD, 0xFF, (size_t)M * N * 4));
            launch(c, ks);
            std::vector<float> got((size_t)M * N);
            CHECK(hipMemcpy(got.data(), dD, got.size() * 4, hipMemcpyDeviceToHost));
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    double ref = 0;
                    for (int k = 0; k < 16 * ks; ++k) ref += (double) (float) hA[(size_t)m * KROW + k] * (double) (float) hB[(size_t)n * KROW + k];
                    if (g_break && c.name == "2x2" && m == 0 && n == 0) ref += 1.0;
                    if ((double) got[(size_t)m * N + n] != ref) ++bad;
                }
        }
        ok = bad == 0;
        report(c.name, ok, ok ? "all outputs equal the double reference at K steps 64 and 128"
                              : ("mismatching outputs: " + std::to_string(bad)).c_str());
    }

    // One wave alone: warm runs, then the median of 9 for each K step count; the difference removes the fixed
    // part (first load latency, drain after the last matrix instruction).
    for (const Case & c : cases) {
        double med[2];
        int idx = 0;
        for (int ks : {64, 128}) {
            for (int w = 0; w < 3; ++w) launch(c, ks);
            std::vector<double> v;
            for (int r = 0; r < 9; ++r) v.push_back((double) launch(c, ks));
            med[idx++] = stats_of(v).median;
        }
        const double per_step = (med[1] - med[0]) / 64.0;
        char buf[256];
        std::snprintf(buf, sizeof buf, "cycles K64 %.0f K128 %.0f  per K step %.1f  per wmma %.2f  (%d wmma per step)", med[0], med[1],
                      per_step, per_step / (c.MT * c.NT), c.MT * c.NT);
        verdict(("haz_regs_" + c.name).c_str(), "INFO", buf);
    }
    std::printf("haz_regs: %d FAIL\n", g_fails);
    return g_fails ? 1 : 0;
}
