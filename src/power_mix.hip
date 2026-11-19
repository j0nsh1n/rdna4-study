// Round 3, Q1c: does the instruction mix of the patched FP8 decode kernel explain its extra ~57 W (reported watts) over Q8_0 decode at the same bandwidth?
// Three streaming kernels read the same 1 GiB of "weights" per launch (memory-bound, far above the 64 MB cache) with different per-byte work:
//   fp8   the loop of llama.cpp's patched mul_mat_vec_f for FP8 (one 16-byte non-temporal load, 16 FP8 weights widened with shifts/and/or, 16 mixed F16xF32 FMAs,
//         the NaN-code check; the widening function is copied from the patched tree's mmvf.cu), activations from a 16 KiB cached vector
//   q8    the loop of llama.cpp's mul_mat_vec_q for Q8_0 (34-byte blocks: F16 scale + 32 int8, 8 weights per thread per step, two 4-way int8 dots, one convert, one FMA)
//   read  a pure 16-byte read with an XOR (the study's bandwidth probe), the baseline for the memory power
//   fp8row, q8row   the same loop bodies in the thread-block shape of the real kernels (one block per 4096-weight row, block reduction at the end, one store per row)
// Usage: power_mix <fp8|q8|read|fp8row|q8row> <seconds> [break] [rand] [dirty] [rand]
//   rand (row variants and read only, added after the first constant-data result): the weights, scales and activations are random (valid FP8 codes, random int8, random F16 scales in [1, 2),
//   activations small random integers), because the dynamic power of an ALU depends on how many bits toggle and constant data under-reports it. The known answer then is a CPU
//   reference (double precision) of 256 sampled rows, within 1e-4 of the row's sum of absolute terms; "break" perturbs the expectation and must FAIL.
// Each launch reads the whole buffer once; the loop runs for <seconds> (events around groups of 10 launches); prints LOOP_START/LOOP_END with the host monotonic
// clock so the sampler's rows can be windowed, then RESULT lines. Checks (exit 1 on FAIL):
//  1. KNOWN ANSWER of every kernel's result: the buffer holds weights 1.0 (fp8 0x38; q8: int8 1 with scale 1.0) and the activations are 1.0, so the sum of all
//     partial results is exactly the number of weights (the FP8 kernel's is 16 per chunk, the q8 kernel's 8 per group): a kernel that skips work or reads the wrong bytes fails.
//     The argument "break" runs the check with a wrong expectation (+1): must FAIL.
//  2. BANDWIDTH KNOWN ANSWER (read variant only): within 10 % of 706 GB/s (the study's bandwidth probe at the 1408 MHz memory clock).
//  3. the ISA of the loops (scripts/llm3_mix_isacheck.py): the instruction mix is what is claimed (counts per weight), so a compiler that folds the work away fails there.
#include "harness.h"

#include <hip/hip_fp16.h>

#include <cmath>
#include <ctime>
#include <random>
#include <string>

// copied from llama.cpp's patched ggml/src/ggml-cuda/mmvf.cu (mmvf_fma_fp8x16): E4M3 bits shifted into F16 position give value * 2^-8, the sum is scaled by 256 at the end
static __device__ __forceinline__ uint32_t fma_fp8x16(float & sum, const uint4 x, const float4 * y) {
    const uint32_t xw[4] = {x.x, x.y, x.z, x.w};
    uint32_t nan_bytes = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w = xw[i];
        const uint32_t even = ((w << 8) & 0x80008000u) | ((w << 7) & 0x3F803F80u);
        const uint32_t odd = (w & 0x80008000u) | ((w >> 1) & 0x3F803F80u);
        nan_bytes |= ((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
        const float4 yi = y[i];
        sum = fmaf(__half2float(__ushort_as_half((unsigned short) (even & 0xFFFF))), yi.x, sum);
        sum = fmaf(__half2float(__ushort_as_half((unsigned short) (odd & 0xFFFF))), yi.y, sum);
        sum = fmaf(__half2float(__ushort_as_half((unsigned short) (even >> 16))), yi.z, sum);
        sum = fmaf(__half2float(__ushort_as_half((unsigned short) (odd >> 16))), yi.w, sum);
    }
    return nan_bytes;
}

__global__ void mix_fp8(const uint4 * __restrict__ w, size_t n, const float4 * __restrict__ y, unsigned ymask, float * __restrict__ out) {
    float sum = 0.0f;
    uint32_t nan = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        typedef uint32_t u4 __attribute__((ext_vector_type(4)));
        const u4 v = __builtin_nontemporal_load((const u4 *) w + i);
        nan |= fma_fp8x16(sum, make_uint4(v.x, v.y, v.z, v.w), y + 4 * (i & ymask));
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = nan ? NAN : 256.0f * sum;
}

// Q8_0: block = 2 bytes F16 scale + 32 int8 (34 bytes); a thread handles 8 weights (two ints) of a block per step; activations are Q8_1-like: 32 int8 + F16 d, per block
struct y_block {
    int8_t q[32];
    __half d;
    __half s;
};

static __device__ __forceinline__ int get_int_b2(const void * x, int i) {  // two 16-bit loads, as llama.cpp's get_int_b2 for 2-byte aligned blocks
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) x16[2 * i] | ((int) x16[2 * i + 1] << 16);
}

__global__ void mix_q8(const char * __restrict__ w, size_t ngroups, const y_block * __restrict__ y, unsigned ymask, float * __restrict__ out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ngroups; i += (size_t) gridDim.x * blockDim.x) {
        const size_t b = i >> 2;
        const int g = (int) (i & 3);
        const char * xb = w + b * 34;
        const y_block * yb = y + (b & ymask);
        const int x0 = get_int_b2(xb + 2 + g * 8, 0), x1 = get_int_b2(xb + 2 + g * 8, 1);
        const int y0 = ((const int *) yb->q)[g * 2], y1 = ((const int *) yb->q)[g * 2 + 1];
        int sumi = __builtin_amdgcn_sudot4(true, x0, true, y0, 0, false);
        sumi = __builtin_amdgcn_sudot4(true, x1, true, y1, sumi, false);
        const float dx = __half2float(*(const __half *) xb);
        acc = fmaf(dx * __half2float(yb->d), (float) sumi, acc);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Row-structured variants (the first two kernels above are loop-only): one thread block per weight row of 4096 weights, as in llama.cpp's decode kernels, with the
// block reduction at the end (shuffles, shared memory, a barrier) and one store per row. Same loop bodies.
// fp8row: 256 threads, one 16-byte chunk (16 weights) per thread, no loop (mul_mat_vec_f's shape for ncols = 4096);
// q8row: 128 threads, 4 threads per 32-weight block, 4 iterations of 32 blocks per row (mul_mat_vec_q's shape for ncols = 4096, one column).
__global__ void mix_fp8row(const uint4 * __restrict__ w, const float4 * __restrict__ y, float * __restrict__ out) {
    const int tid = threadIdx.x;
    typedef uint32_t u4 __attribute__((ext_vector_type(4)));
    const u4 v = __builtin_nontemporal_load((const u4 *) w + (size_t) blockIdx.x * 256 + tid);
    float sum = 0.0f;
    const uint32_t nan = fma_fp8x16(sum, make_uint4(v.x, v.y, v.z, v.w), y + 4 * tid);
    sum = nan ? NAN : 256.0f * sum;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor(sum, off, 32);
    }
    __shared__ float part[8];
    if ((tid & 31) == 0) {
        part[tid >> 5] = sum;
    }
    __syncthreads();
    if (tid < 32) {
        float t = tid < 8 ? part[tid] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            t += __shfl_xor(t, off, 32);
        }
        if (tid == 0) {
            out[blockIdx.x] = t;
        }
    }
}

__global__ void mix_q8row(const char * __restrict__ w, const y_block * __restrict__ y, float * __restrict__ out) {
    const int tid = threadIdx.x;
    const char * row = w + (size_t) blockIdx.x * 128 * 34;
    const int g = tid & 3;
    float acc = 0.0f;
    for (int kbx = tid >> 2; kbx < 128; kbx += 32) {
        const char * xb = row + kbx * 34;
        const y_block * yb = y + kbx;
        const int x0 = get_int_b2(xb + 2 + g * 8, 0), x1 = get_int_b2(xb + 2 + g * 8, 1);
        const int y0 = ((const int *) yb->q)[g * 2], y1 = ((const int *) yb->q)[g * 2 + 1];
        int sumi = __builtin_amdgcn_sudot4(true, x0, true, y0, 0, false);
        sumi = __builtin_amdgcn_sudot4(true, x1, true, y1, sumi, false);
        const float dx = __half2float(*(const __half *) xb);
        acc = fmaf(dx * __half2float(yb->d), (float) sumi, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, 32);
    }
    __shared__ float part[4];
    if ((tid & 31) == 0) {
        part[tid >> 5] = acc;
    }
    __syncthreads();
    if (tid < 32) {
        float t = tid < 4 ? part[tid] : 0.0f;
#pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            t += __shfl_xor(t, off, 32);
        }
        if (tid == 0) {
            out[blockIdx.x] = t;
        }
    }
}

__global__ void mix_read(const uint4 * __restrict__ w, size_t n, float * __restrict__ out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = w[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = (float) (acc & 1u);
}

static double mono_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 3) {
        std::fprintf(stderr, "usage: power_mix <fp8|q8|read|fp8row|q8row> <seconds> [break] [rand] [dirty]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const double seconds = std::atof(argv[2]);
    bool do_break = false, do_rand = false, do_dirty = false;
    for (int i = 3; i < argc; ++i) {
        do_break |= std::strcmp(argv[i], "break") == 0;
        do_rand |= std::strcmp(argv[i], "rand") == 0;
        do_dirty |= std::strcmp(argv[i], "dirty") == 0;
    }
    const size_t bytes = 1ull << 30;
    const int block = 256;
    const int grid = prop.multiProcessorCount * 32;
    const size_t nthreads = (size_t) grid * block;
    std::printf("device %s (%s), variant %s, %.1f s, grid %d x %d%s%s\n", prop.name, prop.gcnArchName, mode.c_str(), seconds, grid, block, do_break ? ", BREAK" : "", do_rand ? ", RANDOM DATA" : "");
    if (do_rand && !(mode == "fp8row" || mode == "q8row" || mode == "read")) {
        std::fprintf(stderr, "rand is for fp8row, q8row and read only\n");
        return 2;
    }
    uint8_t * w;
    float * out;
    CHECK(hipMalloc(&w, bytes));
    const bool is_fp8 = mode == "fp8" || mode == "fp8row";
    const bool is_q8 = mode == "q8" || mode == "q8row";
    const bool is_row = mode == "fp8row" || mode == "q8row";
    const size_t rows_fp8 = bytes / 4096;
    const size_t rows_q8 = bytes / (34 * 128);
    const size_t nrows = mode == "fp8row" ? rows_fp8 : (mode == "q8row" ? rows_q8 : 0);
    const size_t nout = std::max(nthreads, nrows);
    CHECK(hipMalloc(&out, nout * sizeof(float)));
    CHECK(hipMemset(out, 0, nout * sizeof(float)));
    if (do_dirty) {
        // deliberate dirt (the check must not depend on fresh memory being zero): every entry starts as 1.0f
        std::vector<float> junk(nout, 1.0f);
        CHECK(hipMemcpy(out, junk.data(), nout * sizeof(float), hipMemcpyHostToDevice));
    }
    float4 * y4 = nullptr;
    y_block * yq = nullptr;
    const unsigned yvecs = 1024;  // 16 KiB of float activations, 1024 float4 groups of 4
    const unsigned yblocks = 128;
    CHECK(hipMalloc(&y4, yvecs * 4 * sizeof(float4)));
    CHECK(hipMalloc(&yq, yblocks * sizeof(y_block)));
    // known-answer data: fp8 1.0 = 0x38; Q8_0 blocks with scale 1.0 (0x3c00) and int8 1
    std::vector<uint8_t> hw(1 << 20);
    if (is_q8) {
        for (size_t i = 0; i < hw.size(); ++i) {
            const size_t r = i % 34;
            hw[i] = r == 0 ? 0x00 : (r == 1 ? 0x3c : 0x01);
        }
    } else {
        std::fill(hw.begin(), hw.end(), is_fp8 ? 0x38 : 0x01);
    }
    if (is_q8) {
        // the 1 MiB pattern repeats every 34 bytes; 1 GiB / 34 is not a whole number of blocks, so the tail block is partial: the group count below uses whole blocks only
        for (size_t off = 0; off < bytes; off += hw.size()) {
            const size_t len = std::min(hw.size(), bytes - off);
            // keep the 34-byte phase: the pattern buffer length is not a multiple of 34, so copy with the phase of `off`
            std::vector<uint8_t> tmp(len);
            for (size_t i = 0; i < len; ++i) {
                const size_t r = (off + i) % 34;
                tmp[i] = r == 0 ? 0x00 : (r == 1 ? 0x3c : 0x01);
            }
            CHECK(hipMemcpy(w + off, tmp.data(), len, hipMemcpyHostToDevice));
        }
    } else {
        for (size_t off = 0; off < bytes; off += hw.size()) {
            CHECK(hipMemcpy(w + off, hw.data(), std::min(hw.size(), bytes - off), hipMemcpyHostToDevice));
        }
    }
    {
        std::vector<float4> hy(yvecs * 4, make_float4(1.f, 1.f, 1.f, 1.f));
        CHECK(hipMemcpy(y4, hy.data(), hy.size() * sizeof(float4), hipMemcpyHostToDevice));
        std::vector<y_block> hq(yblocks);
        for (auto & b : hq) {
            for (int j = 0; j < 32; ++j) {
                b.q[j] = 1;
            }
            b.d = __float2half(1.0f);
            b.s = __float2half(0.0f);
        }
        CHECK(hipMemcpy(yq, hq.data(), hq.size() * sizeof(y_block), hipMemcpyHostToDevice));
    }
    // random data (round 3 follow-up): kept on the host for the CPU reference
    std::vector<uint8_t> rw;
    std::vector<float> ry_f;
    std::vector<y_block> ry_q;
    if (do_rand) {
        std::mt19937 rng(12345);
        rw.resize(bytes);
        if (is_q8) {
            for (size_t i = 0; i < bytes; ++i) {
                const size_t r = i % 34;
                const uint32_t x = (uint32_t) rng();
                rw[i] = r == 0 ? (uint8_t) x : (r == 1 ? (uint8_t) (0x3c | (x & 3)) : (uint8_t) x);  // scale F16 in [1, 2), int8 weights
            }
        } else {
            for (size_t i = 0; i + 4 <= bytes; i += 4) {
                const uint32_t x = (uint32_t) rng();
                std::memcpy(&rw[i], &x, 4);
            }
            if (is_fp8) {
                for (size_t i = 0; i < bytes; ++i) {
                    if ((rw[i] & 0x7f) == 0x7f) {
                        rw[i] ^= 0x40;  // never an E4M3 NaN code
                    }
                }
            }
        }
        for (size_t off = 0; off < bytes; off += (64u << 20)) {
            CHECK(hipMemcpy(w + off, rw.data() + off, std::min<size_t>(64u << 20, bytes - off), hipMemcpyHostToDevice));
        }
        ry_f.resize(yvecs * 16);
        for (auto & v : ry_f) {
            v = (float) ((int) (rng() % 9) - 4);
        }
        CHECK(hipMemcpy(y4, ry_f.data(), ry_f.size() * sizeof(float), hipMemcpyHostToDevice));
        ry_q.resize(yblocks);
        for (auto & b : ry_q) {
            for (int j = 0; j < 32; ++j) {
                b.q[j] = (int8_t) (rng() & 0xff);
            }
            uint16_t hb = (uint16_t) (0x3c00 | (rng() & 0x3ff));
            std::memcpy(&b.d, &hb, 2);
            b.s = __float2half(0.0f);
        }
        CHECK(hipMemcpy(yq, ry_q.data(), ry_q.size() * sizeof(y_block), hipMemcpyHostToDevice));
    }
    const size_t n16 = bytes / 16;
    const size_t nblocks34 = bytes / 34;
    const size_t ngroups = nblocks34 * 4;
    auto launch = [&] {
        if (mode == "fp8row") {
            mix_fp8row<<<(unsigned) rows_fp8, 256>>>((const uint4 *) w, y4, out);
        } else if (mode == "q8row") {
            mix_q8row<<<(unsigned) rows_q8, 128>>>((const char *) w, yq, out);
        } else if (mode == "fp8") {
            mix_fp8<<<grid, block>>>((const uint4 *) w, n16, y4, yvecs - 1, out);
        } else if (mode == "q8") {
            mix_q8<<<grid, block>>>((const char *) w, ngroups, yq, yblocks - 1, out);
        } else {
            mix_read<<<grid, block>>>((const uint4 *) w, n16, out);
        }
    };
    // 1. known answer of the result
    launch();
    CHECK(hipDeviceSynchronize());
    std::vector<float> ho(nout);
    CHECK(hipMemcpy(ho.data(), out, nout * sizeof(float), hipMemcpyDeviceToHost));
    // only the entries the kernel writes: one per thread for fp8, q8 and read, one per row for the row variants
    const size_t nwritten = is_row ? nrows : nthreads;
    double total = 0;
    for (size_t i = 0; i < nwritten; ++i) {
        total += ho[i];
    }
    double want = mode == "fp8" ? 16.0 * (double) n16 : (mode == "q8" ? 8.0 * (double) ngroups : (is_row ? 4096.0 * (double) nrows : 0.0));
    if (do_break) {
        want += 1.0;
    }
    if (mode == "read") {
        std::printf("RESULT known_answer_result read variant: no numeric result\n");
    } else if (do_rand) {
        // CPU reference (double) of 256 sampled rows
        auto dec8 = [](uint8_t b) {
            const int e = (b >> 3) & 15, m = b & 7;
            const double v = e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
            return (b & 0x80) ? -v : v;
        };
        std::mt19937 pick(777);
        int bad = 0;
        const size_t nsamp = 256;
        for (size_t t = 0; t < nsamp; ++t) {
            const size_t r = pick() % nrows;
            double ref = 0, absref = 0;
            if (mode == "fp8row") {
                for (int j = 0; j < 4096; ++j) {
                    const double term = dec8(rw[r * 4096 + j]) * (double) ry_f[j];
                    ref += term;
                    absref += std::fabs(term);
                }
            } else {
                for (int kb = 0; kb < 128; ++kb) {
                    const uint8_t * xb = &rw[r * 4352 + (size_t) kb * 34];
                    uint16_t hx;
                    std::memcpy(&hx, xb, 2);
                    __half dxh;
                    std::memcpy(&dxh, &hx, 2);
                    long sumi = 0;
                    for (int i = 0; i < 32; ++i) {
                        sumi += (long) (int8_t) xb[2 + i] * (long) ry_q[kb].q[i];
                    }
                    const double term = (double) __half2float(dxh) * (double) __half2float(ry_q[kb].d) * (double) sumi;
                    ref += term;
                    absref += std::fabs(term);
                }
            }
            if (do_break) {
                ref += 1e-2 * absref;  // the deliberate break: a wrong expectation
            }
            if (std::fabs((double) ho[r] - ref) > 1e-4 * absref) {
                ++bad;
            }
        }
        std::printf("RESULT known_answer_result random data: %d of %zu sampled rows differ from the CPU reference by more than 1e-4 of the sum of |terms|\n", bad, nsamp);
        std::printf("VERDICT result-known-answer %s%s\n", bad == 0 ? "PASS" : "FAIL", do_break ? " (deliberate break: expectation off by 1 %, must FAIL)" : "");
        if (bad) {
            std::printf("VERDICT power_mix FAIL: the kernel does not compute what it claims\n");
            return 1;
        }
    } else {
        const bool ok = total == want;
        std::printf("RESULT known_answer_result sum %.1f expected %.1f\n", total, want);
        std::printf("VERDICT result-known-answer %s%s\n", ok ? "PASS" : "FAIL", do_break ? " (deliberate break: expectation + 1, must FAIL)" : "");
        if (!ok) {
            std::printf("VERDICT power_mix FAIL: the kernel does not compute what it claims\n");
            return 1;
        }
    }
    // 2. the timed loop
    for (int i = 0; i < 30; ++i) {
        launch();
    }
    CHECK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    std::vector<double> gbps;
    const double t_start = mono_s();
    std::printf("LOOP_START mono_s %.6f\n", t_start);
    std::fflush(stdout);
    long launches = 0;
    while (mono_s() - t_start < seconds) {
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < 10; ++i) {
            launch();
        }
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        gbps.push_back(10.0 * (double) bytes / (ms * 1e6));
        launches += 10;
    }
    const double t_end = mono_s();
    std::printf("LOOP_END mono_s %.6f\n", t_end);
    std::sort(gbps.begin(), gbps.end());
    const double med = gbps[gbps.size() / 2];
    std::printf("RESULT %s launches %ld loop_s %.3f GBps_median %.1f min %.1f max %.1f groups %zu\n", mode.c_str(), launches, t_end - t_start, med, gbps.front(), gbps.back(),
                gbps.size());
    bool pass = true;
    if (mode == "read") {
        const bool ok = std::fabs(med - 706.0) <= 70.6;
        std::printf("VERDICT bandwidth-known-answer %s (read %.1f GB/s against 706 +- 10 %%)\n", ok ? "PASS" : "FAIL", med);
        pass = ok;
    }
    std::printf("VERDICT power_mix %s\n", pass ? "PASS" : "FAIL");
    CHECK(hipFree(w));
    CHECK(hipFree(out));
    CHECK(hipFree(y4));
    CHECK(hipFree(yq));
    return pass ? 0 : 1;
}
