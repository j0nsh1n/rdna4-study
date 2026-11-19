// Shared parts of the matrix-timing probes (lane 4 of notes/STUDY-PLAN.md).
//
// Method (the same differential as core_common.h): one wave of 32 lanes is launched, so it is alone on its SIMD. The
// timed region is a loop of ITERS iterations; each iteration runs U copies of one inline-assembly block. The wave's
// cycle counter is read before and after. Every test is built with U = 4, 6 and 8 and the cost of one block is
//     (cycles(U=8) - cycles(U=4)) / (4 * ITERS),
// which cancels the loop control, the counter reads and the drain at the end. U = 6 must land midway (linearity).
//
// Registers: every operand is bound to a fixed register range with an explicit-register constraint ("+{v[16:23]}"),
// so the assembly text can name sub-registers (a VALU reading one register of a matrix result) and the compiler only
// moves data in and out around the loop. scripts/study-matrix-timing.sh checks in the -save-temps listing that the
// loop holds nothing but the blocks and the loop control (scripts/core_loopcheck.py) and counts the matrix
// instructions in each loop (scripts/mem_loopcheck.py with scripts/mat-expect/<probe>.txt).
//
//   D0..D7  accumulators, 8 VGPRs each: v[16:23] v[24:31] ... v[72:79]; 16-bit accumulator forms use the low 4
//   A0..A3  v[80:83] v[84:87] v[88:91] v[92:95]; forms with 2- or 1-register A use the low part
//   B0..B3  v[96:103] v[104:111] v[112:119] v[120:127]; dense forms use the low 4, 2 or 1, sparse 16-bit forms all 8
//   IDX     v128 (sparse index), VOFF v129 (lane offset for loads), X v[130:137] (VALU scratch), K v[138:139]
//   S       s[40:43] (scalar scratch), P s[44:45] (address of the 64 KiB load buffer)
// Matrix A or B operands never overlap a matrix destination except in the hazard tests that say so.
#pragma once
#include "core_common.h"

#include <cstdarg>
#include <map>

typedef int mat_v8i __attribute__((ext_vector_type(8)));
typedef int mat_v4i __attribute__((ext_vector_type(4)));
typedef int mat_v2i __attribute__((ext_vector_type(2)));

namespace mat {

constexpr int ITERS = 128;
constexpr int REPS = 8;  // the first repetition warms the instruction cache and is dropped
constexpr int LAUNCHES = 3;
constexpr int ULO = 4, UMID = 6, UHI = 8;

// Layout of the input words (per lane where it says so).
constexpr int IN_D = 0;      // 32 lanes x 8 words: initial accumulator content (every D gets the same)
constexpr int IN_A = 256;    // 32 x 8: A content (every A set gets the first 4 words)
constexpr int IN_B = 512;    // 32 x 8: B content (every B set)
constexpr int IN_IDX = 768;  // 32: sparse index
constexpr int IN_X = 800;    // 8 words: X and K (uniform floats)
constexpr int IN_S = 816;    // 4 words: S (uniform)
constexpr int IN_WORDS = 1024;
constexpr int MEM_WORDS = 16384;  // 64 KiB load buffer

struct Regs {
    mat_v8i d0, d1, d2, d3, d4, d5, d6, d7;
    mat_v4i a0, a1, a2, a3;
    mat_v8i b0, b1, b2, b3;
    int idx, voff;
    mat_v8i x;
    mat_v2i k;
    mat_v4i s;
};

template <class T>
__device__ __forceinline__ T ld(const uint32_t * w) {
    T t;
    __builtin_memcpy(&t, w, sizeof(T));
    return t;
}

__device__ __forceinline__ void load_regs(Regs & r, const uint32_t * in) {
    const int lane = threadIdx.x;
    r.d0 = r.d1 = r.d2 = r.d3 = r.d4 = r.d5 = r.d6 = r.d7 = ld<mat_v8i>(in + IN_D + lane * 8);
    r.a0 = r.a1 = r.a2 = r.a3 = ld<mat_v4i>(in + IN_A + lane * 8);
    r.b0 = r.b1 = r.b2 = r.b3 = ld<mat_v8i>(in + IN_B + lane * 8);
    r.idx = (int) in[IN_IDX + lane];
    r.voff = lane * 16;
    r.x = ld<mat_v8i>(in + IN_X);
    r.k = ld<mat_v2i>(in + IN_X + 8);
    r.s = ld<mat_v4i>(in + IN_S);
}

__device__ __forceinline__ uint32_t fold_regs(const Regs & r) {
    uint32_t h = 0;
#define MAT_FOLD8(v) for (int i = 0; i < 8; ++i) h = h * 31u + (uint32_t) (v)[i];
#define MAT_FOLD4(v) for (int i = 0; i < 4; ++i) h = h * 31u + (uint32_t) (v)[i];
    MAT_FOLD8(r.d0) MAT_FOLD8(r.d1) MAT_FOLD8(r.d2) MAT_FOLD8(r.d3) MAT_FOLD8(r.d4) MAT_FOLD8(r.d5) MAT_FOLD8(r.d6)
    MAT_FOLD8(r.d7) MAT_FOLD4(r.a0) MAT_FOLD4(r.a1) MAT_FOLD4(r.a2) MAT_FOLD4(r.a3) MAT_FOLD8(r.b0) MAT_FOLD8(r.b1)
    MAT_FOLD8(r.b2) MAT_FOLD8(r.b3) MAT_FOLD8(r.x) MAT_FOLD4(r.s)
#undef MAT_FOLD8
#undef MAT_FOLD4
    return h ^ (uint32_t) r.idx ^ (uint32_t) r.k[0];
}

}  // namespace mat

#define MAT_OUTS(r)                                                                                                \
    "+{v[16:23]}"(r.d0), "+{v[24:31]}"(r.d1), "+{v[32:39]}"(r.d2), "+{v[40:47]}"(r.d3), "+{v[48:55]}"(r.d4),     \
        "+{v[56:63]}"(r.d5), "+{v[64:71]}"(r.d6), "+{v[72:79]}"(r.d7), "+{v[80:83]}"(r.a0), "+{v[84:87]}"(r.a1), \
        "+{v[88:91]}"(r.a2), "+{v[92:95]}"(r.a3), "+{v[96:103]}"(r.b0), "+{v[104:111]}"(r.b1),                 \
        "+{v[112:119]}"(r.b2), "+{v[120:127]}"(r.b3), "+{v128}"(r.idx), "+{v[130:137]}"(r.x),                 \
        "+{v[138:139]}"(r.k), "+{s[40:43]}"(r.s)
#define MAT_INS(r) "{v129}"(r.voff)

// One test kernel: U blocks of BODY per loop iteration. The address of the load buffer is passed as `mem`.
#define MAT_KERNEL(NAME, BODY)                                                                                    \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void NAME(const uint32_t * __restrict__ in, uint64_t * __restrict__ out,      \
                                               uint32_t * __restrict__ sink, const uint32_t * mem) {              \
        mat::Regs r;                                                                                              \
        mat::load_regs(r, in);                                                                                    \
        const uint32_t * pm = mem;                                                                                \
        /* binds every register before the timed loop, so the initial loads are waited for here, not inside it */ \
        asm volatile("" : MAT_OUTS(r), "+{s[44:45]}"(pm) : MAT_INS(r));                                          \
        for (int rep = 0; rep < mat::REPS; ++rep) {                                                               \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < mat::ITERS; ++it) {                                         \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(BODY : MAT_OUTS(r), "+{s[44:45]}"(pm) : MAT_INS(r) : "vcc", "scc", "memory");   \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        const uint32_t h = mat::fold_regs(r);                                                                     \
        if (h == 0x9e3779b9u) {                                                                                   \
            sink[threadIdx.x] = h;                                                                                \
        }                                                                                                         \
    }

// Register names. D(n): 8-register accumulator n; H(n): its low 4 registers (16-bit accumulator forms).
#define MAT_D_0 "v[16:23]"
#define MAT_D_1 "v[24:31]"
#define MAT_D_2 "v[32:39]"
#define MAT_D_3 "v[40:47]"
#define MAT_D_4 "v[48:55]"
#define MAT_D_5 "v[56:63]"
#define MAT_D_6 "v[64:71]"
#define MAT_D_7 "v[72:79]"
#define MAT_H_0 "v[16:19]"
#define MAT_H_1 "v[24:27]"
#define MAT_H_2 "v[32:35]"
#define MAT_H_3 "v[40:43]"
#define MAT_H_4 "v[48:51]"
#define MAT_H_5 "v[56:59]"
#define MAT_H_6 "v[64:67]"
#define MAT_H_7 "v[72:75]"
#define MAT_A4_0 "v[80:83]"
#define MAT_A4_1 "v[84:87]"
#define MAT_A4_2 "v[88:91]"
#define MAT_A4_3 "v[92:95]"
#define MAT_A2_0 "v[80:81]"
#define MAT_A2_1 "v[84:85]"
#define MAT_A2_2 "v[88:89]"
#define MAT_A2_3 "v[92:93]"
#define MAT_A1_0 "v80"
#define MAT_A1_1 "v84"
#define MAT_A1_2 "v88"
#define MAT_A1_3 "v92"
#define MAT_B8_0 "v[96:103]"
#define MAT_B8_1 "v[104:111]"
#define MAT_B8_2 "v[112:119]"
#define MAT_B8_3 "v[120:127]"
#define MAT_B4_0 "v[96:99]"
#define MAT_B4_1 "v[104:107]"
#define MAT_B4_2 "v[112:115]"
#define MAT_B4_3 "v[120:123]"
#define MAT_B2_0 "v[96:97]"
#define MAT_B2_1 "v[104:105]"
#define MAT_B2_2 "v[112:113]"
#define MAT_B2_3 "v[120:121]"
#define MAT_B1_0 "v96"
#define MAT_B1_1 "v104"
#define MAT_B1_2 "v112"
#define MAT_B1_3 "v120"
#define MAT_D(n) MAT_D_##n
#define MAT_H(n) MAT_H_##n
#define MAT_A4(m) MAT_A4_##m
#define MAT_A2(m) MAT_A2_##m
#define MAT_A1(m) MAT_A1_##m
#define MAT_B8(m) MAT_B8_##m
#define MAT_B4(m) MAT_B4_##m
#define MAT_B2(m) MAT_B2_##m
#define MAT_B1(m) MAT_B1_##m

#define MAT_R2(x) x x
#define MAT_R4(x) MAT_R2(x) MAT_R2(x)
#define MAT_R8(x) MAT_R4(x) MAT_R4(x)
#define MAT_R16(x) MAT_R8(x) MAT_R8(x)

// 16-instruction blocks for a form macro F(n, m) (accumulator n, operand set m). C1: one chain (every instruction
// reads the previous one's result as C); C2/C4/C8: 2, 4, 8 independent accumulators interleaved; ROT1: one chain
// whose A and B registers change every instruction (4 sets); ROT8: 8 accumulators and rotating A and B.
#define MAT_C1(F) MAT_R16(F(0, 0))
#define MAT_C2(F) MAT_R8(F(0, 0) F(1, 0))
#define MAT_C4(F) MAT_R4(F(0, 0) F(1, 0) F(2, 0) F(3, 0))
#define MAT_C8(F) MAT_R2(F(0, 0) F(1, 0) F(2, 0) F(3, 0) F(4, 0) F(5, 0) F(6, 0) F(7, 0))
#define MAT_ROT1(F) MAT_R4(F(0, 0) F(0, 1) F(0, 2) F(0, 3))
#define MAT_ROT8(F) MAT_R2(F(0, 0) F(1, 1) F(2, 2) F(3, 3) F(4, 0) F(5, 1) F(6, 2) F(7, 3))

// Dense forms. 16-bit inputs: A and B 4 registers; 8-bit and K=32 4-bit: 2; K=16 4-bit: 1.
#define MF_F16(n, m) "v_wmma_f32_16x16x16_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", " MAT_D(n) "\n"
#define MF_BF16(n, m) "v_wmma_f32_16x16x16_bf16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", " MAT_D(n) "\n"
#define MF_F16H(n, m) "v_wmma_f16_16x16x16_f16 " MAT_H(n) ", " MAT_A4(m) ", " MAT_B4(m) ", " MAT_H(n) "\n"
#define MF_BF16H(n, m) "v_wmma_bf16_16x16x16_bf16 " MAT_H(n) ", " MAT_A4(m) ", " MAT_B4(m) ", " MAT_H(n) "\n"
#define MF_FP8(OP, n, m) OP " " MAT_D(n) ", " MAT_A2(m) ", " MAT_B2(m) ", " MAT_D(n)
#define MF_FF(n, m) MF_FP8("v_wmma_f32_16x16x16_fp8_fp8", n, m) "\n"
#define MF_FB(n, m) MF_FP8("v_wmma_f32_16x16x16_fp8_bf8", n, m) "\n"
#define MF_BF(n, m) MF_FP8("v_wmma_f32_16x16x16_bf8_fp8", n, m) "\n"
#define MF_BB(n, m) MF_FP8("v_wmma_f32_16x16x16_bf8_bf8", n, m) "\n"
#define MF_IU8(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) "\n"
#define MF_IU8_S(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[1,1,0]\n"
#define MF_IU8_SA(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[1,0,0]\n"
#define MF_IU8_SB(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[0,1,0]\n"
#define MF_IU8_C(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " clamp\n"
#define MF_IU8_SC(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[1,1,0] clamp\n"
#define MF_IU8_SAC(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[1,0,0] clamp\n"
#define MF_IU8_SBC(n, m) MF_FP8("v_wmma_i32_16x16x16_iu8", n, m) " neg_lo:[0,1,0] clamp\n"
#define MF_IU4(n, m) "v_wmma_i32_16x16x16_iu4 " MAT_D(n) ", " MAT_A1(m) ", " MAT_B1(m) ", " MAT_D(n) " neg_lo:[1,1,0]\n"
#define MF_IU4K32(n, m) MF_FP8("v_wmma_i32_16x16x32_iu4", n, m) " neg_lo:[1,1,0]\n"

// Sparse forms: D is also C; IDX (v128) is the index operand.
#define MS_F16(n, m) "v_swmmac_f32_16x16x32_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B8(m) ", v128\n"
#define MS_BF16(n, m) "v_swmmac_f32_16x16x32_bf16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B8(m) ", v128\n"
#define MS_F16H(n, m) "v_swmmac_f16_16x16x32_f16 " MAT_H(n) ", " MAT_A4(m) ", " MAT_B8(m) ", v128\n"
#define MS_BF16H(n, m) "v_swmmac_bf16_16x16x32_bf16 " MAT_H(n) ", " MAT_A4(m) ", " MAT_B8(m) ", v128\n"
#define MS_IU8(n, m) "v_swmmac_i32_16x16x32_iu8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B4(m) ", v128 neg_lo:[1,1,0]\n"
#define MS_IU4K32(n, m) "v_swmmac_i32_16x16x32_iu4 " MAT_D(n) ", " MAT_A1(m) ", " MAT_B2(m) ", v128 neg_lo:[1,1,0]\n"
#define MS_IU4K64(n, m) "v_swmmac_i32_16x16x64_iu4 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B4(m) ", v128 neg_lo:[1,1,0]\n"
#define MS_FP8(OP, n, m) OP " " MAT_D(n) ", " MAT_A2(m) ", " MAT_B4(m) ", v128\n"
#define MS_FF(n, m) MS_FP8("v_swmmac_f32_16x16x32_fp8_fp8", n, m)
#define MS_FB(n, m) MS_FP8("v_swmmac_f32_16x16x32_fp8_bf8", n, m)
#define MS_BF(n, m) MS_FP8("v_swmmac_f32_16x16x32_bf8_fp8", n, m)
#define MS_BB(n, m) MS_FP8("v_swmmac_f32_16x16x32_bf8_bf8", n, m)

// Declares the three builds of one test and a table entry for it.
#define MAT_TEST(NAME, BODY) MAT_KERNEL(NAME, BODY)
#define MAT_ENTRY(NAME) mat::Test{#NAME, NAME<mat::ULO>, NAME<mat::UMID>, NAME<mat::UHI>}

namespace mat {

using Kern = void (*)(const uint32_t *, uint64_t *, uint32_t *, const uint32_t *);

struct Test {
    const char * name;
    Kern lo, mid, hi;
};

struct Buffers {
    uint32_t * in = nullptr;
    uint64_t * out = nullptr;
    uint32_t * sink = nullptr;
    uint32_t * mem = nullptr;
};

inline Buffers & buffers() {
    static Buffers b;
    if (!b.in) {
        CHECK(hipMalloc(&b.in, IN_WORDS * 4));
        CHECK(hipMalloc(&b.out, 64 * 8));
        CHECK(hipMalloc(&b.sink, 64 * 4));
        CHECK(hipMalloc(&b.mem, MEM_WORDS * 4));
        std::vector<uint32_t> m(MEM_WORDS, 0x3c003c00u);  // f16 pairs of 1.0
        CHECK(hipMemcpy(b.mem, m.data(), m.size() * 4, hipMemcpyHostToDevice));
    }
    return b;
}

// Host image of the input words.
struct Input {
    std::vector<uint32_t> w = std::vector<uint32_t>(IN_WORDS, 0);
    void fill_d(uint32_t v) { std::fill(w.begin() + IN_D, w.begin() + IN_D + 256, v); }
    void fill_a(uint32_t v) { std::fill(w.begin() + IN_A, w.begin() + IN_A + 256, v); }
    void fill_b(uint32_t v) { std::fill(w.begin() + IN_B, w.begin() + IN_B + 256, v); }
    void fill_idx(uint32_t v) { std::fill(w.begin() + IN_IDX, w.begin() + IN_IDX + 32, v); }
};

// Default content: f16 pairs of 2^-6 (as bf16, fp8 or int8 the values are small too), accumulators zero, sparse
// index 0x4444 (every group of four keeps elements 0 and 1), X = 1.0, K = (0.999, 0.001), S = 1.
inline Input default_input() {
    Input i;
    i.fill_a(0x24002400u);
    i.fill_b(0x24002400u);
    i.fill_idx(0x44444444u);
    for (int j = 0; j < 8; ++j) {
        i.w[IN_X + j] = 0x3f800000u;
    }
    const float k0 = 0.999f, k1 = 0.001f;
    std::memcpy(&i.w[IN_X + 8], &k0, 4);
    std::memcpy(&i.w[IN_X + 9], &k1, 4);
    for (int j = 0; j < 4; ++j) {
        i.w[IN_S + j] = 1;
    }
    return i;
}

inline void set_input(const Input & i) {
    CHECK(hipMemcpy(buffers().in, i.w.data(), IN_WORDS * 4, hipMemcpyHostToDevice));
}

// Cycles of the timed region for LAUNCHES x (REPS - 1) repetitions of one single-wave launch.
inline std::vector<double> run_one_wave(Kern k) {
    Buffers & b = buffers();
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, b.in, b.out, b.sink, b.mem);
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

// per: cycles per block; lo/hi: conservative range from the extreme repetitions; mid: distance of U = 6 from the
// midpoint of U = 4 and U = 8, in cycles per loop iteration; ovh: loop control per iteration.
struct Cost {
    double per, lo, hi, mid, ovh;
};

inline Cost block_cost(const Test & t) {
    const Stats a = stats_of(run_one_wave(t.lo));
    const Stats m = stats_of(run_one_wave(t.mid));
    const Stats b = stats_of(run_one_wave(t.hi));
    const double n = double(UHI - ULO) * ITERS;
    const double per = (b.median - a.median) / n;
    if (std::getenv("MAT_DEBUG")) {
        std::printf("debug %s cycles per iteration: U=4 %.1f [%.1f-%.1f] U=6 %.1f [%.1f-%.1f] U=8 %.1f [%.1f-%.1f]\n",
                    t.name, a.median / ITERS, a.min / ITERS, a.max / ITERS, m.median / ITERS, m.min / ITERS,
                    m.max / ITERS, b.median / ITERS, b.min / ITERS, b.max / ITERS);
    }
    return {per, (b.min - a.max) / n, (b.max - a.min) / n, (m.median - 0.5 * (a.median + b.median)) / ITERS,
            (b.median - per * UHI * ITERS) / ITERS};
}

// Linearity limit for the U = 6 build, in cycles per iteration (core_common.h uses 1.5 for 16-cycle blocks; matrix
// blocks are 100 to 1000 cycles long, so 0.5 % of a block is allowed on top).
inline bool linear(const Cost & c) { return std::fabs(c.mid) <= 1.5 + 0.005 * c.per * (UMID - ULO); }

// Measures every test once, prints one RESULT line each ("per" = cycles per block / units), returns per-unit costs.
// Units: how many instructions of interest one block holds (for a 16-instruction block, 16).
struct Measured {
    double per, lo, hi;
    bool lin;
};

inline std::map<std::string, Measured> measure(const char * probe, const std::vector<Test> & tests, double units,
                                               const char * unit_name = "instr") {
    std::map<std::string, Measured> m;
    for (const Test & t : tests) {
        const Cost c = block_cost(t);
        const bool lin = linear(c);
        m[t.name] = {c.per / units, c.lo / units, c.hi / units, lin};
        std::printf("RESULT %s %-28s cycles_per_%s %8.3f [%.3f-%.3f] block %8.2f loop_ovh %6.2f u6_off %6.2f%s\n",
                    probe, t.name, unit_name, c.per / units, c.lo / units, c.hi / units, c.per, c.ovh, c.mid,
                    lin ? "" : " NONLINEAR");
        std::fflush(stdout);
    }
    return m;
}

inline bool flag(int argc, char ** argv, const char * f) { return core::flag(argc, argv, f); }

// Verdict helper: appends a check to text and clears ok if it fails.
struct Checks {
    bool ok = true;
    std::string text;
    void add(bool pass, const char * fmt, ...) __attribute__((format(printf, 3, 4))) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        text += text.empty() ? "" : "; ";
        text += buf;
        if (!pass) {
            text += " FAILED";
            ok = false;
        }
    }
    int finish(const char * probe) {
        verdict(probe, ok ? "PASS" : "FAIL", text.c_str());
        return ok ? 0 : 1;
    }
};

inline bool near(double got, double want, double tol) { return std::fabs(got - want) <= tol; }

}  // namespace mat
