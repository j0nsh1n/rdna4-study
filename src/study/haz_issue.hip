// Lane 8, items 3 and 4c: what the compiler's s_delay_alu, waits and dual-issue pairing do to ordinary loops.
// The same source is built four ways by scripts/study-compiler.sh: default, with s_delay_alu insertion off
// (-mllvm --amdgpu-enable-delay-alu=0), with dual issue off (-mllvm --amdgpu-enable-vopd=0) and in expert scheduling
// mode (-mllvm --amdgpu-expert-scheduling-mode); the build is named
// by HAZ_VARIANT. Each kernel runs a loop of FMAs (or matrix instructions); every wave reads its own cycle counter
// around the loop. A work-group of 32 threads is one wave alone on a SIMD; 512 and 1024 threads put 4 and 8 waves
// on each SIMD of one work-group processor (WGP mode spreads a group over its 4 SIMDs; lane 1 recorded the
// placement). Reported: cycles per FMA for the slowest wave, and FMAs per cycle per SIMD = waves * ops / cycles.
// Usage: haz_issue [all|break]. "break" compares the chain result with a reference using one FMA fewer.
// Built with -DHAZ_PLACEMENT (leftovers C, lane A, CH-3; scripts/study-leftovers-c-a.sh): every kernel also records
// each wave's HW_ID1, and "place [--break]" launches each kernel with 512 and 1,024 threads and checks that every
// SIMD of the WGP gets the same number of waves (the even spread item 3's per-SIMD figures assume); --break claims
// 5 and 3 waves on two SIMDs, so the verdict must FAIL. Without the macro the kernels are unchanged.
#include "haz_common.h"

using namespace haz;

#ifndef HAZ_VARIANT
#define HAZ_VARIANT "default"
#endif

#ifdef HAZ_PLACEMENT
// after the timed loop: HW_ID1 of the wave into cyc[32 + wave]
#define HAZ_PLACE(t)                                                                  \
    if (((t) & 31) == 0) {                                                            \
        uint32_t hw_;                                                                 \
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw_));            \
        cyc[32 + ((t) >> 5)] = hw_;                                                   \
    }
#else
#define HAZ_PLACE(t)
#endif

constexpr int OPS = 16;  // FMAs per loop iteration in every FMA kernel

// One dependent chain: each FMA needs the previous result (s_delay_alu is needed in front of each).
__global__ void i_chain1(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float x = X[t & 31];
    const float k = X[32], c = X[33];
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            x = fmaf(x, k, c);
        }
    }
    const uint64_t t1 = wave_cycles();
    Y[t] = x;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// Four independent chains interleaved: each FMA depends on the one 4 back.
__global__ void i_chain4(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float x0 = X[t & 31], x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
    const float k = X[32], c = X[33];
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS / 4; ++j) {
            x0 = fmaf(x0, k, c);
            x1 = fmaf(x1, k, c);
            x2 = fmaf(x2, k, c);
            x3 = fmaf(x3, k, c);
        }
    }
    const uint64_t t1 = wave_cycles();
    Y[t] = x0 + x1 + x2 + x3;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// Eight independent accumulators fed from two loop-invariant registers: the straightforward dual-issue candidate.
__global__ void i_acc8(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = X[t & 31] + (float) j;
    }
    const float k = X[32], c = X[33];
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            a[j & 7] = fmaf(k, c, a[j & 7]);
        }
    }
    const uint64_t t1 = wave_cycles();
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += a[j];
    }
    Y[t] = s;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// The same with per-lane (VGPR) k and c: X[64 + lane] and X[96 + lane] hold the same values in every lane.
__global__ void i_acc8v(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = X[t & 31] + (float) j;
    }
    const float k = X[64 + (t & 31)], c = X[96 + (t & 31)];
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            a[j & 7] = fmaf(k, c, a[j & 7]);
        }
    }
    const uint64_t t1 = wave_cycles();
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += a[j];
    }
    Y[t] = s;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// The same with a literal multiplier (one literal per pair is allowed; two different literals are not).
__global__ void i_acc8lit(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = X[t & 31] + (float) j;
    }
    const float c = X[96 + (t & 31)];
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            a[j & 7] = fmaf((j & 1) ? 1.0009765625f : 0.9990234375f, c, a[j & 7]);
        }
    }
    const uint64_t t1 = wave_cycles();
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += a[j];
    }
    Y[t] = s;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// Eight accumulators, each with its own pair of per-lane sources (register banks as the allocator picks them).
__global__ void i_acc8var(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float a[8], p[8], q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = X[t & 31] + (float) j;
        p[j] = X[128 + j * 32 + (t & 31)];
        q[j] = X[384 + j * 32 + (t & 31)];
    }
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            a[j & 7] = fmaf(p[j & 7], q[j & 7], a[j & 7]);
        }
    }
    const uint64_t t1 = wave_cycles();
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += a[j];
    }
    Y[t] = s;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// Loads feeding ALU: each iteration loads 16 values and folds them into one chain.
__global__ void i_load(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    float x = X[t & 31];
    const float k = X[32];
    const float * P = X + 64;
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            x = fmaf(x, k, P[((i * OPS + j) & 255) * 32 + (t & 31)]);
        }
    }
    const uint64_t t1 = wave_cycles();
    Y[t] = x;
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

// Matrix chain through C, 16 per iteration.
__global__ void i_mm(const float * X, float * Y, int n, uint64_t * cyc) {
    const int t = threadIdx.x;
    v8h a, b;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = (_Float16) ((t + j) % 3 - 1);
        b[j] = (_Float16) ((t * 3 + j) % 3 - 1);
    }
    v8f acc = {};
    __syncthreads();
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < OPS; ++j) {
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        }
    }
    const uint64_t t1 = wave_cycles();
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += acc[j];
    }
    Y[t] = s * X[34];
    if ((t & 31) == 0) {
        cyc[t >> 5] = cycle_diff(t0, t1);
    }
    HAZ_PLACE(t)
}

namespace {

typedef void (*Kern)(const float *, float *, int, uint64_t *);

// CPU reference of each kernel's output for lane l (fmaf on the host is the correctly rounded fused operation).
float ref_out(const std::string & k, const std::vector<float> & X, int l, int n, int drop) {
    const float kk = X[32], c = X[33];
    if (k == "chain1") {
        float x = X[l];
        for (int i = 0; i < n * OPS - drop; ++i) {
            x = std::fma(x, kk, c);
        }
        return x;
    }
    if (k == "chain4") {
        float x[4] = {X[l], X[l] + 1.0f, X[l] + 2.0f, X[l] + 3.0f};
        for (int i = 0; i < n * OPS / 4; ++i) {
            for (float & v : x) {
                v = std::fma(v, kk, c);
            }
        }
        return x[0] + x[1] + x[2] + x[3];
    }
    if (k == "acc8" || k == "acc8v" || k == "acc8lit" || k == "acc8var") {
        float a[8];
        for (int j = 0; j < 8; ++j) {
            a[j] = X[l] + (float) j;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < OPS; ++j) {
                float p = kk, q = c;
                if (k == "acc8lit") {
                    p = (j & 1) ? 1.0009765625f : 0.9990234375f;
                } else if (k == "acc8var") {
                    p = X[32] + (float) (j & 7) * 0.0009765625f;
                    q = X[33] - (float) (j & 7) * 0.0009765625f;
                }
                a[j & 7] = std::fma(p, q, a[j & 7]);
            }
        }
        float s = 0;
        for (float v : a) {
            s += v;
        }
        return s;
    }
    if (k == "load") {
        float x = X[l];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < OPS; ++j) {
                x = std::fma(x, kk, X[64 + ((i * OPS + j) & 255) * 32 + l]);
            }
        }
        return x;
    }
    return NAN;  // matrix chain: checked against the model separately
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    require_arch();
    Verdicts v;
    std::vector<float> X(64 + 256 * 32);
    for (int l = 0; l < 32; ++l) {
        X[l] = 0.25f + 0.01f * (float) l;
    }
    X[32] = 0.9990234375f;  // k: keeps the chains finite and away from denormals
    X[33] = 0.125f;
    X[34] = 1.0f;
    for (size_t i = 64; i < X.size(); ++i) {
        X[i] = (float) (i % 17) * 0.0078125f;
    }
    // Per-lane copies of k and c, and per-lane sources of acc8var (lane-independent values, so the reference is simple).
    for (int l = 0; l < 32; ++l) {
        X[64 + l] = X[32];
        X[96 + l] = X[33];
        for (int j = 0; j < 8; ++j) {
            X[128 + j * 32 + l] = X[32] + (float) j * 0.0009765625f;
            X[384 + j * 32 + l] = X[33] - (float) j * 0.0009765625f;
        }
    }
    float * dX = dev_copy(X);
    float * dY;
    uint64_t * dC;
    CHECK(hipMalloc(&dY, 1024 * 4));
    CHECK(hipMalloc(&dC, 64 * 8));

    struct K {
        const char * name;
        Kern k;
    } ks[] = {{"chain1", i_chain1}, {"chain4", i_chain4}, {"acc8", i_acc8}, {"acc8v", i_acc8v}, {"acc8lit", i_acc8lit},
              {"acc8var", i_acc8var}, {"load", i_load},   {"mm", i_mm}};
#ifdef HAZ_PLACEMENT
    if (mode == "place") {
        const bool brk = argc > 2 && std::string(argv[2]) == "--break";
        bool even = true;
        for (const K & k : ks) {
            for (int W : {4, 8}) {
                const int threads = 128 * W, nw = threads / 32;
                for (int r = 0; r < 5; ++r) {
                    hipLaunchKernelGGL(k.k, dim3(1), dim3(threads), 0, 0, dX, dY, 64, dC);
                    CHECK(hipDeviceSynchronize());
                    const auto hw = host_copy(dC, 64);
                    int per[4] = {0, 0, 0, 0};
                    for (int w = 0; w < nw; ++w) per[(hw[32 + w] >> 8) & 3]++;  // SIMD_ID: bit 0 CU in WGP, bit 1 SIMD
                    std::sort(per, per + 4);
                    even &= brk ? (per[0] == 3 && per[3] == 5) : (per[0] == W && per[3] == W);
                    if (r == 0) {
                        std::printf("RESULT place %s threads %d waves per SIMD (sorted) %d %d %d %d\n", k.name, threads,
                                    per[0], per[1], per[2], per[3]);
                    }
                }
            }
        }
        v.put("issue_placement", even,
              fmt("every kernel, 512 and 1,024 threads, 5 launches each: %s", brk ? "deliberate break: 3 and 5 waves "
                  "expected on two SIMDs" : "every SIMD of the WGP got the same number of waves (4 or 8)"));
        return v.exit_code();
    }
#endif
    // Correctness, one wave, n = 8.
    for (const K & k : ks) {
        if (std::string(k.name) == "mm") {
            continue;
        }
        hipLaunchKernelGGL(k.k, dim3(1), dim3(32), 0, 0, dX, dY, 8, dC);
        CHECK(hipDeviceSynchronize());
        const auto y = host_copy(dY, 32);
        int w = 0;
        for (int l = 0; l < 32; ++l) {
            w += f32b(y[l]) != f32b(ref_out(k.name, X, l, 8, mode == "break" && std::string(k.name) == "chain1"));
        }
        v.put(mode == "break" ? "issue_result_break" : "issue_result", w == 0,
              fmt("[%s] %s: %d of 32 lanes differ from the CPU", HAZ_VARIANT, k.name, w));
    }
    if (mode == "break") {
        return v.exit_code();
    }
    // Timing: n = 64 and 128 iterations, difference / 64 / OPS.
    const int waves_per_simd[] = {1, 4, 8};
    for (const K & k : ks) {
        for (int W : waves_per_simd) {
            const int threads = W == 1 ? 32 : 128 * W;
            const int nw = threads / 32;
            double c[2];
            for (int j = 0; j < 2; ++j) {
                double best = 1e30;
                for (int r = 0; r < 5; ++r) {
                    hipLaunchKernelGGL(k.k, dim3(1), dim3(threads), 0, 0, dX, dY, 64 << j, dC);
                    CHECK(hipDeviceSynchronize());
                    const auto cy = host_copy(dC, nw);
                    best = std::min(best, (double) *std::max_element(cy.begin(), cy.end()));
                }
                c[j] = best;
            }
            const double per = (c[1] - c[0]) / (64.0 * OPS);
            v.info("issue_time", fmt("[%s] %s waves/SIMD=%d: %.3f cycles per op (slowest wave), %.3f ops per cycle "
                                     "per SIMD",
                                     HAZ_VARIANT, k.name, W, per, W / per));
        }
    }
    return v.exit_code();
}
