// Cycles per matrix (WMMA) instruction in a single wave, and whether vector FMAs overlap with it.
//
// One wave is launched, so it is alone on its SIMD. The timed region is a loop of ITERS iterations, each one WMMA
// that accumulates into the same C (a dependent chain) plus NFMA independent v_fma_f32 spread over 8 chains. The
// wave's shader clock counter is read before and after (harness.h, wave_cycles). After the loop a v_mov reads the
// last results, so the second read cannot happen before the chain has finished.
// Kinds: gfx1100: f16 (gfx11 builtin, operands replicated in both half-waves). gfx1201: f16, bf16, fp8, iu8 (gfx12
// builtins). "x4" variants run four independent accumulators instead of one chain (throughput rather than latency).
// "valu" runs the FMAs alone: its cycles per FMA calibrate the counter (one VALU per cycle expected, with dual issue
// switched off at build time by -mllvm -amdgpu-enable-vopd=0).
//
// Usage: wmma_time [--expect kind=cycles ...]. Prints RESULT lines and one VERDICT line.
// Expected cycles: gfx1100 f16 32 (AMD's calculator; 32 to 33 measured on the RX 7900 XT). gfx1201: AMD's
// calculator says f16 16, bf16 16, fp8 8, iu8 8 (not yet measured anywhere).
#include "harness.h"

#include <cmath>
#include <map>
#include <string>

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v8s __attribute__((ext_vector_type(8)));

enum Kind { K_VALU = 0, K_F16 = 1, K_BF16 = 2, K_FP8 = 3, K_IU8 = 4 };

constexpr int ITERS = 512;
constexpr int REPS = 6;  // the first one warms the instruction cache and is dropped

// One matrix instruction of the given kind. Operand registers come from raw words so every kind loads alike.
template <int KIND>
__device__ __forceinline__ void mm(const uint32_t (&w)[8], v8f & acc, v8i & iacc) {
#if defined(__GFX11__)
    if constexpr (KIND == K_F16) {
        v16h a;
        __builtin_memcpy(&a, w, 16);
        __builtin_memcpy(reinterpret_cast<char *>(&a) + 16, w, 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, a, acc);
    }
#elif defined(__GFX12__)
    if constexpr (KIND == K_F16) {
        v8h a;
        __builtin_memcpy(&a, w, 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, a, acc);
    } else if constexpr (KIND == K_BF16) {
        v8s a;
        __builtin_memcpy(&a, w, 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, a, acc);
    } else if constexpr (KIND == K_FP8) {
        const v2i a = {(int) w[0], (int) w[1]};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, a, acc);
    } else if constexpr (KIND == K_IU8) {
        const v2i a = {(int) w[0], (int) w[1]};
        iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, a, iacc, false);
    }
#endif
}

template <int KIND, int NFMA, int NACC>
__global__ __launch_bounds__(32) void chain(const uint32_t * __restrict__ in, uint64_t * __restrict__ out,
                                           float * __restrict__ sink) {
    const int lane = threadIdx.x;
    uint32_t w[8];
    for (int i = 0; i < 8; ++i) {
        w[i] = in[lane * 8 + i];
    }
    v8f acc[NACC];
    v8i iacc[NACC];
    for (int j = 0; j < NACC; ++j) {
        for (int i = 0; i < 8; ++i) {
            acc[j][i] = 0.0f;
            iacc[j][i] = 0;
        }
    }
    float f[8];
    for (int i = 0; i < 8; ++i) {
        f[i] = __uint_as_float(in[256 + lane * 8 + i]);
    }
    float m = __uint_as_float(in[512]), c = __uint_as_float(in[513]);
    asm volatile("" : "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3]), "+v"(w[4]), "+v"(w[5]), "+v"(w[6]), "+v"(w[7]));
    asm volatile("" : "+v"(m), "+v"(c));
    for (int rep = 0; rep < REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 16
        for (int it = 0; it < ITERS; ++it) {
            if constexpr (KIND != K_VALU) {
#pragma unroll
                for (int j = 0; j < NACC; ++j) {
                    mm<KIND>(w, acc[j], iacc[j]);
                }
            }
#pragma unroll
            for (int k = 0; k < NFMA; ++k) {
                f[k & 7] = fmaf(f[k & 7], m, c);
            }
        }
        float dep0, dep1, dep2;
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep0) : "v"(acc[NACC - 1][7]));
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep1) : "v"(iacc[NACC - 1][7]));
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep2) : "v"(f[(NFMA + 7) & 7]));
        const uint64_t t1 = wave_cycles();
        if (lane == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
        if (dep0 + dep1 + dep2 == 1234.5f) {
            sink[lane] = 1.0f;
        }
    }
    float s = 0;
    for (int j = 0; j < NACC; ++j) {
        for (int i = 0; i < 8; ++i) {
            s += acc[j][i] + (float) iacc[j][i];
        }
    }
    for (int i = 0; i < 8; ++i) {
        s += f[i];
    }
    if (s == 1234.5f) {
        sink[lane] = s;
    }
}

struct Run {
    std::string name;
    void (*kernel)(const uint32_t *, uint64_t *, float *);
    int kind, nfma, nacc;
};

#define RUN(K, NF, NA, NAME) Run{NAME, chain<K, NF, NA>, K, NF, NA}

int main(int argc, char ** argv) {
    require_arch();
    const bool g11 = arch_is_gfx11();
    std::map<std::string, double> expect;
    if (g11) {
        expect["f16"] = 32;
    } else {
        expect["f16"] = 16;
        expect["bf16"] = 16;
        expect["fp8"] = 8;
        expect["iu8"] = 8;
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--expect") {
            const std::string kv = argv[i + 1];
            const size_t eq = kv.find('=');
            expect[kv.substr(0, eq)] = std::atof(kv.c_str() + eq + 1);
        }
    }

    // Small finite values; with a = b the chains grow slowly and never reach infinity in ITERS * REPS steps.
    std::vector<uint32_t> h(1024);
    for (int i = 0; i < 256; ++i) {
        h[i] = 0x24002400u;  // f16 pairs of 2^-6; read as bf16, fp8 or int8 the values are small too
    }
    for (int i = 256; i < 512; ++i) {
        h[i] = 0x3f800000u;  // FMA chains start at 1
    }
    const float mval = 0.999f, cval = 0.001f;
    std::memcpy(&h[512], &mval, 4);
    std::memcpy(&h[513], &cval, 4);

    std::vector<Run> runs = {RUN(K_VALU, 4, 1, "valu"), RUN(K_VALU, 8, 1, "valu"), RUN(K_VALU, 16, 1, "valu"),
                             RUN(K_VALU, 32, 1, "valu")};
    if (g11) {
        runs.insert(runs.end(), {RUN(K_F16, 0, 1, "f16"), RUN(K_F16, 0, 4, "f16"), RUN(K_F16, 4, 1, "f16"),
                                 RUN(K_F16, 8, 1, "f16"), RUN(K_F16, 16, 1, "f16"), RUN(K_F16, 32, 1, "f16")});
    } else {
        runs.insert(runs.end(), {RUN(K_F16, 0, 1, "f16"), RUN(K_F16, 0, 4, "f16"), RUN(K_F16, 4, 1, "f16"),
                                 RUN(K_F16, 8, 1, "f16"), RUN(K_F16, 16, 1, "f16"), RUN(K_F16, 32, 1, "f16"),
                                 RUN(K_BF16, 0, 1, "bf16"), RUN(K_BF16, 0, 4, "bf16"), RUN(K_FP8, 0, 1, "fp8"),
                                 RUN(K_FP8, 0, 4, "fp8"), RUN(K_FP8, 4, 1, "fp8"), RUN(K_FP8, 8, 1, "fp8"),
                                 RUN(K_FP8, 16, 1, "fp8"), RUN(K_FP8, 32, 1, "fp8"), RUN(K_IU8, 0, 1, "iu8"),
                                 RUN(K_IU8, 0, 4, "iu8")});
    }

    uint32_t * din;
    uint64_t * dout;
    float * dsink;
    CHECK(hipMalloc(&din, h.size() * 4));
    CHECK(hipMalloc(&dout, REPS * 8));
    CHECK(hipMalloc(&dsink, 32 * 4));
    CHECK(hipMemcpy(din, h.data(), h.size() * 4, hipMemcpyHostToDevice));

    // cycles per iteration, median over 3 launches x (REPS - 1) timed repetitions
    std::map<std::string, double> per_iter;
    auto key = [](const std::string & n, int nfma, int nacc) {
        return n + (nacc > 1 ? "_x4" : "") + "+" + std::to_string(nfma);
    };
    for (const Run & r : runs) {
        std::vector<double> v;
        for (int launch = 0; launch < 3; ++launch) {
            hipLaunchKernelGGL(r.kernel, dim3(1), dim3(32), 0, 0, din, dout, dsink);
            CHECK(hipDeviceSynchronize());
            uint64_t c[REPS];
            CHECK(hipMemcpy(c, dout, sizeof c, hipMemcpyDeviceToHost));
            for (int i = 1; i < REPS; ++i) {
                v.push_back((double) c[i] / ITERS);
            }
        }
        const Stats s = stats_of(v);
        per_iter[key(r.name, r.nfma, r.nacc)] = s.median;
        const int per = r.kind == K_VALU ? 0 : r.nacc;
        std::printf("RESULT %-5s acc %s fma_per_iter %2d cycles_per_iter %7.2f [%.2f-%.2f]", r.name.c_str(),
                    r.nacc > 1 ? "x4   " : "chain", r.nfma, s.median, s.min, s.max);
        if (per) {
            std::printf("  cycles_per_wmma %.2f", s.median / per);
        } else {
            std::printf("  cycles_per_fma %.3f", s.median / r.nfma);
        }
        std::printf("\n");
    }

    bool ok = true;
    std::string text;
    char buf[256];
    const double cal = per_iter["valu+32"] / 32;
    std::snprintf(buf, sizeof buf, "counter: %.2f cycles per independent FMA (expect about 1)", cal);
    text += buf;
    if (cal < 0.8 || cal > 1.6) {
        ok = false;
        text += " OUT OF RANGE";
    }
    for (const auto & [kind, want] : expect) {
        if (!per_iter.count(kind + "+0")) {
            continue;
        }
        const double got = per_iter[kind + "+0"];
        const bool match = std::fabs(got - want) <= 0.15 * want + 1.0;
        std::snprintf(buf, sizeof buf, "; %s chain %.1f cycles (x4 %.1f), expected %.0f%s", kind.c_str(), got,
                      per_iter[kind + "_x4+0"] / 4, want, match ? "" : " MISMATCH");
        text += buf;
        ok = ok && match;
    }
    for (const std::string kind : {"f16", "fp8"}) {
        if (!per_iter.count(kind + "+32")) {
            continue;
        }
        std::printf("overlap %s: k, wmma+k FMAs, wmma alone + FMAs alone, max of the two, overlap fraction\n",
                    kind.c_str());
        double frac_hi = 0;
        for (int k : {4, 8, 16, 32}) {
            const double tw = per_iter[kind + "+0"], tv = per_iter["valu+" + std::to_string(k)];
            const double tm = per_iter[kind + "+" + std::to_string(k)];
            const double frac = (tw + tv - tm) / std::min(tw, tv);
            std::printf("RESULT overlap %s k %2d mixed %7.2f sum %7.2f max %7.2f fraction %.2f\n", kind.c_str(), k, tm,
                        tw + tv, std::max(tw, tv), frac);
            if (k >= 16) {
                frac_hi = std::max(frac_hi, frac);
            }
        }
        // fraction 0: the FMAs add their full time; 1: they hide completely inside the matrix instruction's time.
        const char * word = frac_hi < 0.25 ? "adds, no overlap" : frac_hi > 0.75 ? "hides, overlaps" : "partial overlap";
        std::snprintf(buf, sizeof buf, "; %s+FMA %s (fraction %.2f at k>=16)", kind.c_str(), word, frac_hi);
        text += buf;
        if (g11 && kind == "f16" && frac_hi >= 0.25) {
            ok = false;
            text += " UNEXPECTED on gfx1100";
        }
    }
    verdict("wmma_time", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
