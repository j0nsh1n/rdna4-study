// Lane 8, item 8: wave64 under HIP (-mwavefrontsize64). One wave of 64 lanes runs the _w64_gfx12 matrix builtin;
// the result is checked against the CPU model under a layout hypothesis, and a chain of 64 instructions is timed.
// Layout (H2, checked here): the wave32 image with each lane's upper half moved to lane + 32:
//   A row r, slot s: lane r + 16 * (s >> 3) + 32 * ((s >> 2) & 1), half (s & 3);  B column c likewise;
//   D (r, c): lane 16 * (r >> 3) + c + 32 * ((r >> 2) & 1), float (r & 3).
// The first guess (H1: slot groups of 4 over lane groups of 16, lane r + 16 * (s >> 2); D lane 16 * (r >> 2) + c)
// gave 128 of 256 wrong. Usage: haz_w64 [all|break]. "break" checks against H1; it must fail.
#include "haz_common.h"

using namespace haz;

typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));

__global__ void w64_one(const v4h * A, const v4h * B, const v4f * C, v4f * D) {
    const int l = threadIdx.x;
    D[l] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(A[l], B[l], C[l]);
}

__global__ void w64_chain(const v4h * A, const v4h * B, const v4f * C, v4f * D, int n, uint64_t * cyc) {
    const int l = threadIdx.x;
    const v4h a = A[l], b = B[l];
    v4f acc = C[l];
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(a, b, acc);
        }
    }
    const uint64_t t1 = wave_cycles();
    D[l] = acc;
    if (l == 0) {
        *cyc = cycle_diff(t0, t1);
    }
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    const hipDeviceProp_t prop = require_arch();
    Verdicts v;
    v.info("w64_device", fmt("warpSize reported %d", prop.warpSize));
    // Matrices by element; small integers, so every sum is exact.
    uint32_t s = 5;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return (float) ((int) ((s >> 16) % 7) - 3);
    };
    float Am[16][16], Bm[16][16], Cm[16][16];
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            Am[i][j] = rnd();
            Bm[i][j] = rnd();
            Cm[i][j] = rnd();
        }
    }
    const bool h1 = mode == "break";
    auto ab_at = [&](int r, int k) {
        return h1 ? (r + 16 * (k >> 2)) * 4 + (k & 3) : (r + 16 * (k >> 3) + 32 * ((k >> 2) & 1)) * 4 + (k & 3);
    };
    auto d_at = [&](int r, int col) {
        return h1 ? (16 * (r >> 2) + col) * 4 + (r & 3) : (16 * (r >> 3) + col + 32 * ((r >> 2) & 1)) * 4 + (r & 3);
    };
    std::vector<uint16_t> a(256), b(256);
    std::vector<uint32_t> c(256);
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 16; ++k) {
            a[ab_at(r, k)] = h16(Am[r][k]);
            b[ab_at(r, k)] = h16(Bm[k][r]);  // B column r, slot k
        }
        for (int col = 0; col < 16; ++col) {
            c[d_at(r, col)] = f32b(Cm[r][col]);
        }
    }
    // Reference through the bit-exact model (slot order = K order; all values exact anyway).
    const am::Params P = am::measured_params(am::F_F16);
    float ref[16][16];
    for (int r = 0; r < 16; ++r) {
        for (int col = 0; col < 16; ++col) {
            uint32_t as[16], bs[16];
            for (int k = 0; k < 16; ++k) {
                as[k] = h16(Am[r][k]);
                bs[k] = h16(Bm[k][col]);
            }
            ref[r][col] = bf32(am::model_f32(am::F_F16, as, bs, f32b(Cm[r][col]), P));
        }
    }
    v4h * dA = (v4h *) dev_copy(a);
    v4h * dB = (v4h *) dev_copy(b);
    v4f * dC = (v4f *) dev_copy(c);
    v4f * dD;
    uint64_t * dcyc;
    CHECK(hipMalloc(&dD, 64 * 16));
    CHECK(hipMalloc(&dcyc, 8));
    hipLaunchKernelGGL(w64_one, dim3(1), dim3(64), 0, 0, dA, dB, dC, dD);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const auto d = host_copy((const uint32_t *) dD, 256);
    int w = 0;
    for (int r = 0; r < 16; ++r) {
        for (int col = 0; col < 16; ++col) {
            const uint32_t got = d[d_at(r, col)];
            w += got != f32b(ref[r][col]);
        }
    }
    v.put(mode == "break" ? "w64_bits_break" : "w64_bits", w == 0,
          fmt("wave64 builtin, 16x16x16 FP16: %d of 256 results differ from the model under layout %s", w, h1 ? "H1" : "H2"));
    if (mode == "break") {
        return v.exit_code();
    }
    double cy[2];
    for (int j = 0; j < 2; ++j) {
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            hipLaunchKernelGGL(w64_chain, dim3(1), dim3(64), 0, 0, dA, dB, dC, dD, 16 << j, dcyc);
            CHECK(hipDeviceSynchronize());
            best = std::min(best, (double) host_copy(dcyc, 1)[0]);
        }
        cy[j] = best;
    }
    v.info("w64_time", fmt("chained wave64 matrix instruction: %.2f cycles each (n=32 minus n=16 iterations of 16)",
                           (cy[1] - cy[0]) / 256.0));
    return v.exit_code();
}
