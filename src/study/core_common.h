// Shared parts of the execution-core probes (lane 1 of notes/STUDY-PLAN.md).
//
// Method: one wave of 32 lanes is launched, so it is alone on its SIMD. The timed region is a loop of ITERS
// iterations; each iteration runs U copies of one inline-assembly block of 16 "patterns" (a pattern is one
// instruction, or a fixed short group). The wave's shader cycle counter is read before and after (harness.h).
// Every test is built with U = 4, 6 and 8, and the cost of one pattern is
//     (cycles(U=8) - cycles(U=4)) / (4 * ITERS * 16),
// which cancels the loop control, the counter reads and the pipeline drain at the end. (U = 1 is avoided: measured,
// its loop costs about one cycle per iteration more than the U >= 3 loops, which all add exactly the block's cost
// per extra block.) U = 6 must land midway between U = 4 and U = 8 (within 1.5 cycles per iteration), or the loop is
// not linear in the block count and the verdict fails.
// Inline assembly is used so the compiler cannot delete, merge, reorder or pair the instructions; the operands are
// "+v" (or "+s") constraints, so the compiler only chooses registers. The generated loop is checked in the
// -save-temps assembly by scripts/study-core.sh (no extra instructions inside the timed loop).
#pragma once
#include "../harness.h"

#include <cmath>
#include <string>
#include <vector>

namespace core {

constexpr int ITERS = 128;
constexpr int REPS = 8;  // timed repetitions per launch; the first warms the instruction cache and is dropped
constexpr int LAUNCHES = 3;
constexpr int PATTERNS = 16;

using Kern = void (*)(const uint64_t *, uint64_t *, uint32_t *);

// Initial bit patterns: x for the 16 chain registers, k0 and k1 for the two constant operands.
struct Init {
    uint64_t x, k0, k1;
};

inline uint64_t f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline uint64_t f64(double d) {
    uint64_t u;
    std::memcpy(&u, &d, 8);
    return u;
}
inline uint64_t h2(float lo, float hi) {  // two halves as packed fp16
    const _Float16 a = (_Float16) lo, b = (_Float16) hi;
    uint16_t ua, ub;
    std::memcpy(&ua, &a, 2);
    std::memcpy(&ub, &b, 2);
    return ua | ((uint64_t) ub << 16);
}

template <class T>
__device__ __forceinline__ T from_bits(uint64_t b) {
    T t;
    __builtin_memcpy(&t, &b, sizeof(T));
    return t;
}

template <class T>
__device__ __forceinline__ uint32_t fold(const T & t) {
    uint64_t b = 0;
    __builtin_memcpy(&b, &t, sizeof(T));
    return (uint32_t) (b ^ (b >> 32));
}

struct Buffers {
    uint64_t * in = nullptr;
    uint64_t * out = nullptr;
    uint32_t * sink = nullptr;
};

inline Buffers & buffers() {
    static Buffers b;
    if (!b.in) {
        CHECK(hipMalloc(&b.in, 64 * 8));
        CHECK(hipMalloc(&b.out, 1024 * 8));
        CHECK(hipMalloc(&b.sink, 1024 * 4));
    }
    return b;
}

inline void set_init(const Init & i) {
    std::vector<uint64_t> h(64, 0);
    h[0] = i.x;
    h[1] = i.k0;
    h[2] = i.k1;
    CHECK(hipMemcpy(buffers().in, h.data(), h.size() * 8, hipMemcpyHostToDevice));
}

// Cycles of the timed region for LAUNCHES x (REPS - 1) repetitions of one single-wave launch.
inline std::vector<double> run_one_wave(Kern k) {
    Buffers & b = buffers();
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, b.in, b.out, b.sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t c[REPS];
        CHECK(hipMemcpy(c, b.out, sizeof c, hipMemcpyDeviceToHost));
        for (int r = 1; r < REPS; ++r) {
            v.push_back((double) c[r]);
        }
    }
    return v;
}

// Cost of one pattern from the U=4 and U=8 builds; the U=6 build tests linearity: it must sit midway.
// per: cycles per pattern; lo/hi: conservative range from the extreme repetitions; raw: cycles(U=8) per pattern with
// the loop control included; ovh: loop control and counter reads in cycles per loop iteration; mid: how far U=6 is
// from the midpoint, in cycles per iteration.
struct Cost {
    double per, lo, hi, raw, ovh, mid;
    double it_lo, it_mid, it_hi;  // median cycles per loop iteration at U = ULO, UMID, UHI
};

constexpr int ULO = 4, UMID = 6, UHI = 8;

inline Cost pattern_cost(Kern klo, Kern kmid, Kern khi, double patterns_per_block = PATTERNS) {
    const Stats a = stats_of(run_one_wave(klo));
    const Stats m = stats_of(run_one_wave(kmid));
    const Stats b = stats_of(run_one_wave(khi));
    const double n = double(UHI - ULO) * ITERS * patterns_per_block;
    const double per = (b.median - a.median) / n;
    return {per,
            (b.min - a.max) / n,
            (b.max - a.min) / n,
            b.median / (double(UHI) * ITERS * patterns_per_block),
            (b.median - per * UHI * ITERS * patterns_per_block) / ITERS,
            (m.median - 0.5 * (a.median + b.median)) / ITERS,
            a.median / ITERS,
            m.median / ITERS,
            b.median / ITERS};
}

inline bool flag(int argc, char ** argv, const char * f) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace core

// The 16-pattern block for c independent chains: pattern j works on chain j mod c, so c = 1 is one dependent
// chain and c = 16 gives 16 independent ones. P(op, i) expands to the instruction text for chain register %i.
#define CORE_B1(P, op) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) \
                       P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0) P(op, 0)
#define CORE_B2(P, op) P(op, 0) P(op, 1) P(op, 0) P(op, 1) P(op, 0) P(op, 1) P(op, 0) P(op, 1) \
                       P(op, 0) P(op, 1) P(op, 0) P(op, 1) P(op, 0) P(op, 1) P(op, 0) P(op, 1)
#define CORE_B4(P, op) P(op, 0) P(op, 1) P(op, 2) P(op, 3) P(op, 0) P(op, 1) P(op, 2) P(op, 3) \
                       P(op, 0) P(op, 1) P(op, 2) P(op, 3) P(op, 0) P(op, 1) P(op, 2) P(op, 3)
#define CORE_B8(P, op) P(op, 0) P(op, 1) P(op, 2) P(op, 3) P(op, 4) P(op, 5) P(op, 6) P(op, 7) \
                       P(op, 0) P(op, 1) P(op, 2) P(op, 3) P(op, 4) P(op, 5) P(op, 6) P(op, 7)
#define CORE_B16(P, op) P(op, 0) P(op, 1) P(op, 2) P(op, 3) P(op, 4) P(op, 5) P(op, 6) P(op, 7) \
                        P(op, 8) P(op, 9) P(op, 10) P(op, 11) P(op, 12) P(op, 13) P(op, 14) P(op, 15)

#define CORE_X16(C)                                                                                                 \
    C(x[0]), C(x[1]), C(x[2]), C(x[3]), C(x[4]), C(x[5]), C(x[6]), C(x[7]), C(x[8]), C(x[9]), C(x[10]), C(x[11]), \
        C(x[12]), C(x[13]), C(x[14]), C(x[15])

// One test kernel. TX: chain register type; TK: constant operand type; XC/KC: constraint strings ("+v"/"v" or
// "+s"/"s"); BODY: the block text; TAIL: text appended to each block (for example a wait counter). Operands: %0..%15
// the chains, %16 and %17 the constants.
#define CORE_KERNEL(NAME, TX, TK, XC, KC, BODY, TAIL)                                                             \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,      \
                                               uint32_t * __restrict__ sink) {                                    \
        TX x[16];                                                                                                 \
        for (int i = 0; i < 16; ++i) {                                                                            \
            x[i] = core::from_bits<TX>(in[0]);                                                                    \
        }                                                                                                         \
        TK k0 = core::from_bits<TK>(in[1]), k1 = core::from_bits<TK>(in[2]);                                      \
        /* pin the constants in registers so the loop does not re-materialise them */                             \
        asm volatile("" : "+" KC(k0), "+" KC(k1));                                                                \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                              \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                        \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(BODY TAIL : CORE_X16(XC) : KC(k0), KC(k1) : "vcc", "scc");                       \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        uint32_t s = 0;                                                                                           \
        for (int i = 0; i < 16; ++i) {                                                                            \
            s ^= core::fold(x[i]);                                                                                \
        }                                                                                                         \
        sink[threadIdx.x] = s;                                                                                    \
    }
