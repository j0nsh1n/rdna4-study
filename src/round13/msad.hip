// Round 13 Probe A: gfx1201 v_msad_u8, v_sad_u8, v_sad_u16, v_sad_hi_u8 throughput,
// dependent-chain latency (accumulator and src0 chains), wave32 and wave64, and co-issue
// mixed with FP32 FMA. Exact CPU verification of every timed kernel output and of a
// dedicated 256-vector-per-lane edge-case table.
//
// Build twice with -mcumode and -mllvm -amdgpu-enable-vopd=0:
//   wave32: hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -mno-wavefrontsize64 -DW64=0 ...
//   wave64: hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -mwavefrontsize64 -DW64=1 ...
//
// Usage: msad_w32 [--break]
//        msad_w64 [--break]
#include "../harness.h"
#include "msad_ref.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef W64
#define W64 0
#endif

#if W64
constexpr int WSIZE = 64;
#else
constexpr int WSIZE = 32;
#endif

constexpr int ITERS = 128;
constexpr int REPS = 8;
constexpr int LAUNCHES = 3;
constexpr int PATTERNS = 16;
constexpr int ULO = 4;
constexpr int UMID = 6;
constexpr int UHI = 8;

template <int OP>
__device__ __forceinline__ uint32_t dev_sad(uint32_t s0, uint32_t s1, uint32_t s2) {
    if constexpr (OP == msad_ref::OP_MSAD_U8) {
        return __builtin_amdgcn_msad_u8(s0, s1, s2);
    } else if constexpr (OP == msad_ref::OP_SAD_U8) {
        return __builtin_amdgcn_sad_u8(s0, s1, s2);
    } else if constexpr (OP == msad_ref::OP_SAD_U16) {
        return __builtin_amdgcn_sad_u16(s0, s1, s2);
    } else {
        return __builtin_amdgcn_sad_hi_u8(s0, s1, s2);
    }
}

// Pin 16 scalar/vector values in registers across compiler barriers without emitting any instruction.
__device__ __forceinline__ void pin_u32(uint32_t & v) { asm volatile("" : "+v"(v)); }
__device__ __forceinline__ void pin_f32(float & v) { asm volatile("" : "+v"(v)); }

// Pure SAD/MSAD timing kernel: U blocks of 16 operations over C independent accumulator chains.
// Every repetition resets x[0..15] outside the timed region so final x[0..15] matches exact CPU
// reference for U * ITERS * (16 / C) steps per active chain.
template <int OP, int CHAINS, int U>
__global__ __launch_bounds__(WSIZE) void k_sad_acc(const uint32_t * __restrict__ in_s0,
                                                   const uint32_t * __restrict__ in_s1,
                                                   const uint32_t * __restrict__ in_x,
                                                   uint64_t * __restrict__ out_cyc,
                                                   uint32_t * __restrict__ out_val) {
    const uint32_t lane = threadIdx.x;
    uint32_t a = in_s0[lane];
    uint32_t b = in_s1[lane];
    pin_u32(a);
    pin_u32(b);

    uint32_t x[16];
    for (int rep = 0; rep < REPS; ++rep) {
        for (int i = 0; i < 16; ++i) {
            x[i] = in_x[lane * 16u + static_cast<uint32_t>(i)];
            pin_u32(x[i]);
        }
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int p = 0; p < PATTERNS; ++p) {
                    const int idx = p % CHAINS;
                    x[idx] = dev_sad<OP>(a, b, x[idx]);
                }
            }
        }
        // Data-dependent barrier so t1 waits for the last vector result to land.
        uint32_t dep = x[(PATTERNS - 1) % CHAINS];
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep) : "v"(dep));
        const uint64_t t1 = wave_cycles();
        if (lane == 0u) {
            out_cyc[rep] = cycle_diff(t0, t1);
        }
        if (dep == 0x7f7f7f7fu && rep == -1) {
            out_val[0] = dep;
        }
    }
    for (int i = 0; i < 16; ++i) {
        out_val[lane * 16u + static_cast<uint32_t>(i)] = x[i];
    }
}

// Dependent chain through src0 (x = op(x, s1, s2)) to test whether src0->dst latency differs
// from src2(accumulator)->dst latency.
template <int OP, int U>
__global__ __launch_bounds__(WSIZE) void k_sad_src0(const uint32_t * __restrict__ in_s0,
                                                    const uint32_t * __restrict__ in_s1,
                                                    const uint32_t * __restrict__ in_x,
                                                    uint64_t * __restrict__ out_cyc,
                                                    uint32_t * __restrict__ out_val) {
    const uint32_t lane = threadIdx.x;
    uint32_t b = in_s1[lane];
    uint32_t c = in_s0[lane];
    pin_u32(b);
    pin_u32(c);

    uint32_t x = 0u;
    for (int rep = 0; rep < REPS; ++rep) {
        x = in_x[lane * 16u];
        pin_u32(x);
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int p = 0; p < PATTERNS; ++p) {
                    x = dev_sad<OP>(x, b, c);
                }
            }
        }
        uint32_t dep = x;
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep) : "v"(dep));
        const uint64_t t1 = wave_cycles();
        if (lane == 0u) {
            out_cyc[rep] = cycle_diff(t0, t1);
        }
        if (dep == 0x7f7f7f7fu && rep == -1) {
            out_val[0] = dep;
        }
    }
    out_val[lane * 16u] = x;
}

// Pure FP32 FMA baseline (calibration: expect 1.00 cyc/op at c=16, 5.00 cyc/op at c=1 in both
// wave32 and wave64). Pin fa=v32 (bank 0), fb=v33 (bank 1), and accumulators in banks 2 & 3 so
// 0/16 FMAs hit a same-bank VGPR read conflict in wave64 on gfx1201.
#define FMA_F16_CONSTRAINTS                                                                                   \
    "+{v2}"(f[0]), "+{v3}"(f[1]), "+{v6}"(f[2]), "+{v7}"(f[3]), "+{v10}"(f[4]), "+{v11}"(f[5]),               \
        "+{v14}"(f[6]), "+{v15}"(f[7]), "+{v18}"(f[8]), "+{v19}"(f[9]), "+{v22}"(f[10]), "+{v23}"(f[11]),     \
        "+{v26}"(f[12]), "+{v27}"(f[13]), "+{v30}"(f[14]), "+{v31}"(f[15])

#define FMA_BLOCK_C16                                                                                         \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %1, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %2, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %3, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %4, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %5, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %6, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %7, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %8, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %9, %[fa], %[fb]\n"                                                                       \
    "v_fmac_f32_e32 %10, %[fa], %[fb]\n"                                                                      \
    "v_fmac_f32_e32 %11, %[fa], %[fb]\n"                                                                      \
    "v_fmac_f32_e32 %12, %[fa], %[fb]\n"                                                                      \
    "v_fmac_f32_e32 %13, %[fa], %[fb]\n"                                                                      \
    "v_fmac_f32_e32 %14, %[fa], %[fb]\n"                                                                      \
    "v_fmac_f32_e32 %15, %[fa], %[fb]\n"

#define FMA_BLOCK_C1                                                                                          \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"                                   \
    "v_fmac_f32_e32 %0, %[fa], %[fb]\n" "v_fmac_f32_e32 %0, %[fa], %[fb]\n"

template <int CHAINS, int U>
__global__ __launch_bounds__(WSIZE) void k_fma_base(const float * __restrict__ in_fa,
                                                    const float * __restrict__ in_fb,
                                                    const float * __restrict__ in_f,
                                                    uint64_t * __restrict__ out_cyc,
                                                    float * __restrict__ out_f) {
    static_assert(CHAINS == 1 || CHAINS == 16, "k_fma_base supports CHAINS=1 or 16");
    const uint32_t lane = threadIdx.x;
    float fa = in_fa[lane];
    float fb = in_fb[lane];
    pin_f32(fa);
    pin_f32(fb);

    float f[16];
    for (int rep = 0; rep < REPS; ++rep) {
        if constexpr (CHAINS == 16) {
            for (int i = 0; i < 16; ++i) {
                f[i] = in_f[lane * 16u + static_cast<uint32_t>(i)];
            }
        } else {
            f[0] = in_f[lane * 16u];
        }
        const uint64_t t0 = wave_cycles();
        if constexpr (CHAINS == 16) {
            // Put the ITERS loop inside inline asm so LLVM cannot insert register-copy v_mov
            // instructions across the loop backedge when pinning 16 bank-separated VGPRs.
            uint32_t cnt = static_cast<uint32_t>(ITERS);
            static_assert(U == 4 || U == 6 || U == 8, "k_fma_base expects U in {4,6,8}");
            if constexpr (U == 4) {
                asm volatile(
                    ".Lfma16_u4_%=:\n"
                    FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16
                    "s_sub_u32 %[cnt], %[cnt], 1\n"
                    "s_cmp_lg_u32 %[cnt], 0\n"
                    "s_cbranch_scc1 .Lfma16_u4_%=\n"
                    : FMA_F16_CONSTRAINTS, [cnt] "+s"(cnt)
                    : [fa] "{v32}"(fa), [fb] "{v33}"(fb)
                    : "scc");
            } else if constexpr (U == 6) {
                asm volatile(
                    ".Lfma16_u6_%=:\n"
                    FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16
                    "s_sub_u32 %[cnt], %[cnt], 1\n"
                    "s_cmp_lg_u32 %[cnt], 0\n"
                    "s_cbranch_scc1 .Lfma16_u6_%=\n"
                    : FMA_F16_CONSTRAINTS, [cnt] "+s"(cnt)
                    : [fa] "{v32}"(fa), [fb] "{v33}"(fb)
                    : "scc");
            } else {
                asm volatile(
                    ".Lfma16_u8_%=:\n"
                    FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16
                    FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16 FMA_BLOCK_C16
                    "s_sub_u32 %[cnt], %[cnt], 1\n"
                    "s_cmp_lg_u32 %[cnt], 0\n"
                    "s_cbranch_scc1 .Lfma16_u8_%=\n"
                    : FMA_F16_CONSTRAINTS, [cnt] "+s"(cnt)
                    : [fa] "{v32}"(fa), [fb] "{v33}"(fb)
                    : "scc");
            }
        } else {
#pragma unroll 1
            for (int it = 0; it < ITERS; ++it) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    asm volatile(FMA_BLOCK_C1 : "+{v2}"(f[0]) : [fa] "{v32}"(fa), [fb] "{v33}"(fb));
                }
            }
        }
        float dep = f[(PATTERNS - 1) % CHAINS];
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep) : "v"(dep));
        const uint64_t t1 = wave_cycles();
        if (lane == 0u) {
            out_cyc[rep] = cycle_diff(t0, t1);
        }
        if (dep == 123456.75f && rep == -1) {
            out_f[0] = dep;
        }
    }
    for (int i = 0; i < CHAINS; ++i) {
        out_f[lane * 16u + static_cast<uint32_t>(i)] = f[i];
    }
}

// Mixed SAD + NFMA FP32 FMAs per pattern (16 patterns per block, 8 independent SAD chains +
// 8 independent FMA chains) to test whether SAD/MSAD co-issues with FP32 FMA.
// Explicit register pinning and inline asm enforce:
//   1. Zero same-bank VGPR read conflicts in wave64 (constants in banks 0 & 1: fa=v32, fb=v33,
//      a=v36, b=v37; all 16 accumulators in banks 2 & 3: f[0..7]=v2,v3,v6,v7,v10,v11,v14,v15,
//      x[0..7]=v18,v19,v22,v23,v26,v27,v30,v31).
//   2. Exact 1:NFMA instruction interleave (prevents LLVM from hoisting FMAs into a front block)
//      with cyclic FMA chain rotation so every chain has >= 8-instruction reuse distance.
#define MIX_XF_CONSTRAINTS                                                                                    \
    "+{v2}"(f[0]), "+{v3}"(f[1]), "+{v6}"(f[2]), "+{v7}"(f[3]), "+{v10}"(f[4]), "+{v11}"(f[5]),               \
        "+{v14}"(f[6]), "+{v15}"(f[7]), "+{v18}"(x[0]), "+{v19}"(x[1]), "+{v22}"(x[2]), "+{v23}"(x[3]),       \
        "+{v26}"(x[4]), "+{v27}"(x[5]), "+{v30}"(x[6]), "+{v31}"(x[7])

#define MIX_P1(OP, XV, FV)                                                                                    \
    OP " " XV ", v36, v37, " XV "\n"                                                                          \
    "v_fmac_f32_e32 " FV ", v32, v33\n"

#define MIX_P3(OP, XV, F0, F1, F2)                                                                            \
    OP " " XV ", v36, v37, " XV "\n"                                                                          \
    "v_fmac_f32_e32 " F0 ", v32, v33\n"                                                                       \
    "v_fmac_f32_e32 " F1 ", v32, v33\n"                                                                       \
    "v_fmac_f32_e32 " F2 ", v32, v33\n"

#define MIX_BLOCK_1(OP)                                                                                       \
    MIX_P1(OP, "v18", "v2")  MIX_P1(OP, "v19", "v3")  MIX_P1(OP, "v22", "v6")  MIX_P1(OP, "v23", "v7")        \
    MIX_P1(OP, "v26", "v10") MIX_P1(OP, "v27", "v11") MIX_P1(OP, "v30", "v14") MIX_P1(OP, "v31", "v15")       \
    MIX_P1(OP, "v18", "v2")  MIX_P1(OP, "v19", "v3")  MIX_P1(OP, "v22", "v6")  MIX_P1(OP, "v23", "v7")        \
    MIX_P1(OP, "v26", "v10") MIX_P1(OP, "v27", "v11") MIX_P1(OP, "v30", "v14") MIX_P1(OP, "v31", "v15")

#define MIX_BLOCK_3(OP)                                                                                       \
    MIX_P3(OP, "v18", "v2",  "v3",  "v6")  MIX_P3(OP, "v19", "v7",  "v10", "v11")                             \
    MIX_P3(OP, "v22", "v14", "v15", "v2")  MIX_P3(OP, "v23", "v3",  "v6",  "v7")                              \
    MIX_P3(OP, "v26", "v10", "v11", "v14") MIX_P3(OP, "v27", "v15", "v2",  "v3")                              \
    MIX_P3(OP, "v30", "v6",  "v7",  "v10") MIX_P3(OP, "v31", "v11", "v14", "v15")                             \
    MIX_P3(OP, "v18", "v2",  "v3",  "v6")  MIX_P3(OP, "v19", "v7",  "v10", "v11")                             \
    MIX_P3(OP, "v22", "v14", "v15", "v2")  MIX_P3(OP, "v23", "v3",  "v6",  "v7")                              \
    MIX_P3(OP, "v26", "v10", "v11", "v14") MIX_P3(OP, "v27", "v15", "v2",  "v3")                              \
    MIX_P3(OP, "v30", "v6",  "v7",  "v10") MIX_P3(OP, "v31", "v11", "v14", "v15")

#define MIX_LOOP_U(BODY)                                                                                      \
    ".Lmix_loop_%=:\n"                                                                                        \
    BODY                                                                                                      \
    "s_sub_u32 %[cnt], %[cnt], 1\n"                                                                           \
    "s_cmp_lg_u32 %[cnt], 0\n"                                                                                \
    "s_cbranch_scc1 .Lmix_loop_%=\n"

#define EMIT_MIX_LOOP(BLK, UVAL)                                                                              \
    do {                                                                                                      \
        uint32_t cnt = static_cast<uint32_t>(ITERS);                                                          \
        if constexpr ((UVAL) == 4) {                                                                          \
            asm volatile(MIX_LOOP_U(BLK BLK BLK BLK)                                                          \
                         : MIX_XF_CONSTRAINTS, [cnt] "+s"(cnt)                                                \
                         : "{v36}"(a), "{v37}"(b), "{v32}"(fa), "{v33}"(fb)                                   \
                         : "scc");                                                                            \
        } else if constexpr ((UVAL) == 6) {                                                                   \
            asm volatile(MIX_LOOP_U(BLK BLK BLK BLK BLK BLK)                                                  \
                         : MIX_XF_CONSTRAINTS, [cnt] "+s"(cnt)                                                \
                         : "{v36}"(a), "{v37}"(b), "{v32}"(fa), "{v33}"(fb)                                   \
                         : "scc");                                                                            \
        } else {                                                                                              \
            asm volatile(MIX_LOOP_U(BLK BLK BLK BLK BLK BLK BLK BLK)                                          \
                         : MIX_XF_CONSTRAINTS, [cnt] "+s"(cnt)                                                \
                         : "{v36}"(a), "{v37}"(b), "{v32}"(fa), "{v33}"(fb)                                   \
                         : "scc");                                                                            \
        }                                                                                                     \
    } while (0)

template <int OP, int NFMA, int U>
__device__ __forceinline__ void emit_mix_loop(uint32_t (&x)[8], float (&f)[8],
                                              uint32_t a, uint32_t b, float fa, float fb) {
    static_assert(NFMA == 1 || NFMA == 3, "k_sad_fma_mix supports NFMA=1 or 3");
    static_assert(U == 4 || U == 6 || U == 8, "k_sad_fma_mix expects U in {4,6,8}");
    if constexpr (NFMA == 1) {
        if constexpr (OP == msad_ref::OP_MSAD_U8) {
            EMIT_MIX_LOOP(MIX_BLOCK_1("v_msad_u8"), U);
        } else if constexpr (OP == msad_ref::OP_SAD_U8) {
            EMIT_MIX_LOOP(MIX_BLOCK_1("v_sad_u8"), U);
        } else if constexpr (OP == msad_ref::OP_SAD_U16) {
            EMIT_MIX_LOOP(MIX_BLOCK_1("v_sad_u16"), U);
        } else {
            EMIT_MIX_LOOP(MIX_BLOCK_1("v_sad_hi_u8"), U);
        }
    } else {
        if constexpr (OP == msad_ref::OP_MSAD_U8) {
            EMIT_MIX_LOOP(MIX_BLOCK_3("v_msad_u8"), U);
        } else if constexpr (OP == msad_ref::OP_SAD_U8) {
            EMIT_MIX_LOOP(MIX_BLOCK_3("v_sad_u8"), U);
        } else if constexpr (OP == msad_ref::OP_SAD_U16) {
            EMIT_MIX_LOOP(MIX_BLOCK_3("v_sad_u16"), U);
        } else {
            EMIT_MIX_LOOP(MIX_BLOCK_3("v_sad_hi_u8"), U);
        }
    }
}

template <int OP, int NFMA, int U>
__global__ __launch_bounds__(WSIZE) void k_sad_fma_mix(const uint32_t * __restrict__ in_s0,
                                                       const uint32_t * __restrict__ in_s1,
                                                       const uint32_t * __restrict__ in_x,
                                                       const float * __restrict__ in_fa,
                                                       const float * __restrict__ in_fb,
                                                       const float * __restrict__ in_f,
                                                       uint64_t * __restrict__ out_cyc,
                                                       uint32_t * __restrict__ out_val,
                                                       float * __restrict__ out_f) {
    const uint32_t lane = threadIdx.x;
    uint32_t a = in_s0[lane];
    uint32_t b = in_s1[lane];
    float fa = in_fa[lane];
    float fb = in_fb[lane];
    pin_u32(a);
    pin_u32(b);
    pin_f32(fa);
    pin_f32(fb);

    uint32_t x[8];
    float f[8];
    for (int rep = 0; rep < REPS; ++rep) {
        for (int i = 0; i < 8; ++i) {
            x[i] = in_x[lane * 16u + static_cast<uint32_t>(i)];
            f[i] = in_f[lane * 16u + static_cast<uint32_t>(i)];
        }
        const uint64_t t0 = wave_cycles();
        emit_mix_loop<OP, NFMA, U>(x, f, a, b, fa, fb);
        uint32_t dep_u = x[7];
        float dep_f = f[7];
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep_u) : "v"(dep_u));
        asm volatile("v_mov_b32 %0, %1" : "=v"(dep_f) : "v"(dep_f));
        const uint64_t t1 = wave_cycles();
        if (lane == 0u) {
            out_cyc[rep] = cycle_diff(t0, t1);
        }
        if (dep_u == 0x7f7f7f7fu && dep_f == 123456.75f && rep == -1) {
            out_val[0] = dep_u;
        }
    }
    for (int i = 0; i < 8; ++i) {
        out_val[lane * 16u + static_cast<uint32_t>(i)] = x[i];
        out_f[lane * 16u + static_cast<uint32_t>(i)] = f[i];
    }
}

// Dedicated exactness table: 256 vectors per lane covering zero-byte masks in src0 only,
// src1 only, both, neither, extreme byte differences, and 32-bit wrap-around.
__global__ __launch_bounds__(WSIZE) void k_sad_exact(const uint32_t * __restrict__ v_s0,
                                                     const uint32_t * __restrict__ v_s1,
                                                     const uint32_t * __restrict__ v_s2,
                                                     uint32_t * __restrict__ out4,
                                                     uint32_t n_per_lane) {
    const uint32_t lane = threadIdx.x;
    for (uint32_t i = 0; i < n_per_lane; ++i) {
        const uint32_t idx = lane * n_per_lane + i;
        const uint32_t a = v_s0[idx];
        const uint32_t b = v_s1[idx];
        const uint32_t c = v_s2[idx];
        out4[idx * 4u + 0u] = __builtin_amdgcn_msad_u8(a, b, c);
        out4[idx * 4u + 1u] = __builtin_amdgcn_sad_u8(a, b, c);
        out4[idx * 4u + 2u] = __builtin_amdgcn_sad_u16(a, b, c);
        out4[idx * 4u + 3u] = __builtin_amdgcn_sad_hi_u8(a, b, c);
    }
}

struct DevBufs {
    uint32_t * s0 = nullptr;
    uint32_t * s1 = nullptr;
    uint32_t * x = nullptr;
    float * fa = nullptr;
    float * fb = nullptr;
    float * f = nullptr;
    uint64_t * cyc = nullptr;
    uint32_t * out_val = nullptr;
    float * out_f = nullptr;
};

template <typename LaunchFn>
static Stats measure_kernel(LaunchFn launch, uint64_t * d_cyc) {
    std::vector<double> samples;
    for (int l = 0; l < LAUNCHES; ++l) {
        launch();
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t h_cyc[REPS];
        CHECK(hipMemcpy(h_cyc, d_cyc, sizeof(h_cyc), hipMemcpyDeviceToHost));
        for (int r = 1; r < REPS; ++r) {
            samples.push_back(static_cast<double>(h_cyc[r]));
        }
    }
    return stats_of(samples);
}

static int verify_sad_acc(int op, int chains, int U, const uint32_t * d_out, int do_break) {
    std::vector<uint32_t> h(WSIZE * 16);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    int bad = 0;
    const uint32_t steps_per_chain = static_cast<uint32_t>(U * ITERS * (PATTERNS / chains));
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(WSIZE); ++lane) {
        const uint32_t a = msad_ref::lane_s0(lane);
        const uint32_t b = msad_ref::lane_s1(lane);
        for (int i = 0; i < chains; ++i) {
            const uint32_t init_x = msad_ref::lane_init_acc(lane, static_cast<uint32_t>(i));
            uint32_t want = msad_ref::run_acc_chain(op, a, b, init_x, steps_per_chain);
            if (do_break && lane == 0u && i == 0) want ^= 1u;
            if (h[lane * 16u + static_cast<uint32_t>(i)] != want) ++bad;
        }
    }
    return bad;
}

static int verify_sad_src0(int op, int U, const uint32_t * d_out, int do_break) {
    std::vector<uint32_t> h(WSIZE * 16);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    int bad = 0;
    const uint32_t steps = static_cast<uint32_t>(U * ITERS * PATTERNS);
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(WSIZE); ++lane) {
        const uint32_t init_x = msad_ref::lane_init_acc(lane, 0u);
        const uint32_t b = msad_ref::lane_s1(lane);
        const uint32_t c = msad_ref::lane_s0(lane);
        uint32_t want = msad_ref::run_src0_chain(op, init_x, b, c, steps);
        if (do_break && lane == 0u) want ^= 1u;
        if (h[lane * 16u] != want) ++bad;
    }
    return bad;
}

static int verify_fma_base(int chains, int U, const float * d_out, int do_break) {
    std::vector<float> h(WSIZE * 16);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * sizeof(float), hipMemcpyDeviceToHost));
    int bad = 0;
    const uint32_t steps_per_chain = static_cast<uint32_t>(U * ITERS * (PATTERNS / chains));
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(WSIZE); ++lane) {
        for (int i = 0; i < chains; ++i) {
            const float f0 = msad_ref::lane_init_f(lane, static_cast<uint32_t>(i));
            float want = msad_ref::run_fma_chain(f0, 1.0f, 0.0625f, steps_per_chain);
            if (do_break && lane == 0u && i == 0) want += 0.0625f;
            if (h[lane * 16u + static_cast<uint32_t>(i)] != want) ++bad;
        }
    }
    return bad;
}

static int verify_mix(int op, int nfma, int U, const uint32_t * d_val, const float * d_f, int do_break) {
    std::vector<uint32_t> hu(WSIZE * 16);
    std::vector<float> hf(WSIZE * 16);
    CHECK(hipMemcpy(hu.data(), d_val, hu.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hf.data(), d_f, hf.size() * sizeof(float), hipMemcpyDeviceToHost));
    int bad = 0;
    const uint32_t sad_steps = static_cast<uint32_t>(U * ITERS * 2);  // 16 patterns / 8 chains = 2
    const uint32_t fma_steps = static_cast<uint32_t>(U * ITERS * 2 * nfma);
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(WSIZE); ++lane) {
        const uint32_t a = msad_ref::lane_s0(lane);
        const uint32_t b = msad_ref::lane_s1(lane);
        for (int i = 0; i < 8; ++i) {
            uint32_t want_u = msad_ref::run_acc_chain(op, a, b, msad_ref::lane_init_acc(lane, static_cast<uint32_t>(i)), sad_steps);
            float want_f = msad_ref::run_fma_chain(msad_ref::lane_init_f(lane, static_cast<uint32_t>(i)), 1.0f, 0.0625f, fma_steps);
            if (do_break && lane == 0u && i == 0) want_u ^= 1u;
            if (hu[lane * 16u + static_cast<uint32_t>(i)] != want_u) ++bad;
            if (hf[lane * 16u + static_cast<uint32_t>(i)] != want_f) ++bad;
        }
    }
    return bad;
}

template <int OP, int CHAINS>
static int run_sad_acc_case(const char * name, const DevBufs & d, int do_break,
                            double * out_per = nullptr, int * out_nonlin = nullptr) {
    auto s_lo = measure_kernel([&] { hipLaunchKernelGGL((k_sad_acc<OP, CHAINS, ULO>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_lo = verify_sad_acc(OP, CHAINS, ULO, d.out_val, do_break);
    auto s_mid = measure_kernel([&] { hipLaunchKernelGGL((k_sad_acc<OP, CHAINS, UMID>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_mid = verify_sad_acc(OP, CHAINS, UMID, d.out_val, do_break);
    auto s_hi = measure_kernel([&] { hipLaunchKernelGGL((k_sad_acc<OP, CHAINS, UHI>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_hi = verify_sad_acc(OP, CHAINS, UHI, d.out_val, do_break);

    const msad_ref::DiffCost dc = msad_ref::diff_cost(s_lo.median, s_mid.median, s_hi.median, ULO, UHI, ITERS, PATTERNS);
    const double n = static_cast<double>(UHI - ULO) * ITERS * PATTERNS;
    const double lo = (s_hi.min - s_lo.max) / n;
    const double hi = (s_hi.max - s_lo.min) / n;
    const int bad = b_lo + b_mid + b_hi;
    if (out_per) *out_per = dc.per;
    if (out_nonlin && !dc.linear) ++(*out_nonlin);
    std::printf("RESULT msad w=%d %-14s c=%2d cyc/op=%.3f [%.3f-%.3f] ops/cyc=%.3f mid=%.2f ovh=%.1f exact=%s%s\n",
                WSIZE, name, CHAINS, dc.per, lo, hi, 1.0 / dc.per, dc.mid, dc.ovh,
                bad == 0 ? "ok" : "FAIL", dc.linear ? "" : " NONLINEAR");
    std::fflush(stdout);
    return bad == 0 ? 0 : 1;
}

template <int OP>
static int run_sad_src0_case(const char * name, const DevBufs & d, int do_break, int * out_nonlin) {
    auto s_lo = measure_kernel([&] { hipLaunchKernelGGL((k_sad_src0<OP, ULO>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_lo = verify_sad_src0(OP, ULO, d.out_val, do_break);
    auto s_mid = measure_kernel([&] { hipLaunchKernelGGL((k_sad_src0<OP, UMID>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_mid = verify_sad_src0(OP, UMID, d.out_val, do_break);
    auto s_hi = measure_kernel([&] { hipLaunchKernelGGL((k_sad_src0<OP, UHI>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.cyc, d.out_val); }, d.cyc);
    int b_hi = verify_sad_src0(OP, UHI, d.out_val, do_break);

    const msad_ref::DiffCost dc = msad_ref::diff_cost(s_lo.median, s_mid.median, s_hi.median, ULO, UHI, ITERS, PATTERNS);
    const double n = static_cast<double>(UHI - ULO) * ITERS * PATTERNS;
    const double lo = (s_hi.min - s_lo.max) / n;
    const double hi = (s_hi.max - s_lo.min) / n;
    const int bad = b_lo + b_mid + b_hi;
    if (out_nonlin && !dc.linear) ++(*out_nonlin);
    std::printf("RESULT msad w=%d %-14s c= 1 cyc/op=%.3f [%.3f-%.3f] ops/cyc=%.3f mid=%.2f ovh=%.1f exact=%s%s\n",
                WSIZE, name, dc.per, lo, hi, 1.0 / dc.per, dc.mid, dc.ovh,
                bad == 0 ? "ok" : "FAIL", dc.linear ? "" : " NONLINEAR");
    std::fflush(stdout);
    return bad == 0 ? 0 : 1;
}

template <int OP, int NFMA>
static int run_mix_case(const char * name, const DevBufs & d, int do_break, int * out_nonlin) {
    auto s_lo = measure_kernel([&] { hipLaunchKernelGGL((k_sad_fma_mix<OP, NFMA, ULO>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.fa, d.fb, d.f, d.cyc, d.out_val, d.out_f); }, d.cyc);
    int b_lo = verify_mix(OP, NFMA, ULO, d.out_val, d.out_f, do_break);
    auto s_mid = measure_kernel([&] { hipLaunchKernelGGL((k_sad_fma_mix<OP, NFMA, UMID>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.fa, d.fb, d.f, d.cyc, d.out_val, d.out_f); }, d.cyc);
    int b_mid = verify_mix(OP, NFMA, UMID, d.out_val, d.out_f, do_break);
    auto s_hi = measure_kernel([&] { hipLaunchKernelGGL((k_sad_fma_mix<OP, NFMA, UHI>), dim3(1), dim3(WSIZE), 0, 0, d.s0, d.s1, d.x, d.fa, d.fb, d.f, d.cyc, d.out_val, d.out_f); }, d.cyc);
    int b_hi = verify_mix(OP, NFMA, UHI, d.out_val, d.out_f, do_break);

    const msad_ref::DiffCost dc = msad_ref::diff_cost(s_lo.median, s_mid.median, s_hi.median, ULO, UHI, ITERS, PATTERNS);
    const double n = static_cast<double>(UHI - ULO) * ITERS * PATTERNS;
    const double lo = (s_hi.min - s_lo.max) / n;
    const double hi = (s_hi.max - s_lo.min) / n;
    const int bad = b_lo + b_mid + b_hi;
    if (out_nonlin && !dc.linear) ++(*out_nonlin);
    std::printf("RESULT msad_mix w=%d %-18s nfma=%d cyc/pat=%.3f [%.3f-%.3f] mid=%.2f ovh=%.1f exact=%s%s\n",
                WSIZE, name, NFMA, dc.per, lo, hi, dc.mid, dc.ovh,
                bad == 0 ? "ok" : "FAIL", dc.linear ? "" : " NONLINEAR");
    std::fflush(stdout);
    return bad == 0 ? 0 : 1;
}

static int run_exact_table(int do_break) {
    const uint32_t n_per_lane = 256u;
    const uint32_t total = static_cast<uint32_t>(WSIZE) * n_per_lane;
    std::vector<uint32_t> h_s0(total), h_s1(total), h_s2(total), h_out(total * 4u);
    uint64_t rng = 0x20261127c0ffee13ull;
    auto next32 = [&]() -> uint32_t {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(rng >> 32);
    };
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t a = next32(), b = next32(), c = next32();
        if ((i & 7u) == 0u) a &= 0xff00ff00u;           // zero bytes in s0 only
        else if ((i & 7u) == 1u) b &= 0x00ff00ffu;      // zero bytes in s1 only
        else if ((i & 7u) == 2u) { a = 0u; b = 0xffffffffu; }
        else if ((i & 7u) == 3u) { a = 0xffffffffu; b = 0u; }
        else if ((i & 7u) == 4u) { c = 0xfffffff0u; }   // 32-bit wrap
        h_s0[i] = a;
        h_s1[i] = b;
        h_s2[i] = c;
    }
    uint32_t * d_s0 = nullptr;
    uint32_t * d_s1 = nullptr;
    uint32_t * d_s2 = nullptr;
    uint32_t * d_out = nullptr;
    CHECK(hipMalloc(&d_s0, total * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_s1, total * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_s2, total * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_out, total * 4u * sizeof(uint32_t)));
    CHECK(hipMemcpy(d_s0, h_s0.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_s1, h_s1.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_s2, h_s2.data(), total * sizeof(uint32_t), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(k_sad_exact, dim3(1), dim3(WSIZE), 0, 0, d_s0, d_s1, d_s2, d_out, n_per_lane);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h_out.data(), d_out, total * 4u * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int bad[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t want[4] = {
            msad_ref::msad_u8(h_s0[i], h_s1[i], h_s2[i]),
            msad_ref::sad_u8(h_s0[i], h_s1[i], h_s2[i]),
            msad_ref::sad_u16(h_s0[i], h_s1[i], h_s2[i]),
            msad_ref::sad_hi_u8(h_s0[i], h_s1[i], h_s2[i]),
        };
        if (do_break && i == 0u) want[0] ^= 1u;
        for (int k = 0; k < 4; ++k) {
            if (h_out[i * 4u + static_cast<uint32_t>(k)] != want[k]) ++bad[k];
        }
    }
    CHECK(hipFree(d_s0));
    CHECK(hipFree(d_s1));
    CHECK(hipFree(d_s2));
    CHECK(hipFree(d_out));
    const int tot_bad = bad[0] + bad[1] + bad[2] + bad[3];
    std::printf("CONTROL msad_exact w=%d vectors=%u bad_msad=%d bad_sad8=%d bad_sad16=%d bad_sadhi8=%d exact=%s\n",
                WSIZE, total, bad[0], bad[1], bad[2], bad[3], tot_bad == 0 ? "ok" : "FAIL");
    std::fflush(stdout);
    return tot_bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    int do_break = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--break") == 0) do_break = 1;
        else {
            std::fprintf(stderr, "usage: msad_w%d [--break]\n", WSIZE);
            return 2;
        }
    }
    require_arch();

    std::vector<uint32_t> h_s0(WSIZE), h_s1(WSIZE), h_x(WSIZE * 16);
    std::vector<float> h_fa(WSIZE, 1.0f), h_fb(WSIZE, 0.0625f), h_f(WSIZE * 16);
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(WSIZE); ++lane) {
        h_s0[lane] = msad_ref::lane_s0(lane);
        h_s1[lane] = msad_ref::lane_s1(lane);
        for (uint32_t i = 0; i < 16u; ++i) {
            h_x[lane * 16u + i] = msad_ref::lane_init_acc(lane, i);
            h_f[lane * 16u + i] = msad_ref::lane_init_f(lane, i);
        }
    }

    DevBufs d;
    CHECK(hipMalloc(&d.s0, h_s0.size() * sizeof(uint32_t)));
    CHECK(hipMalloc(&d.s1, h_s1.size() * sizeof(uint32_t)));
    CHECK(hipMalloc(&d.x, h_x.size() * sizeof(uint32_t)));
    CHECK(hipMalloc(&d.fa, h_fa.size() * sizeof(float)));
    CHECK(hipMalloc(&d.fb, h_fb.size() * sizeof(float)));
    CHECK(hipMalloc(&d.f, h_f.size() * sizeof(float)));
    CHECK(hipMalloc(&d.cyc, REPS * sizeof(uint64_t)));
    CHECK(hipMalloc(&d.out_val, WSIZE * 16 * sizeof(uint32_t)));
    CHECK(hipMalloc(&d.out_f, WSIZE * 16 * sizeof(float)));

    CHECK(hipMemcpy(d.s0, h_s0.data(), h_s0.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.s1, h_s1.data(), h_s1.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.x, h_x.data(), h_x.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.fa, h_fa.data(), h_fa.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.fb, h_fb.data(), h_fb.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.f, h_f.data(), h_f.size() * sizeof(float), hipMemcpyHostToDevice));

    int fails = 0;
    int nonlin = 0;

    // 1. Exhaustive exactness control table.
    fails += run_exact_table(do_break);

    // 2. FP32 FMA calibration (both wave32 and wave64 expect 1.00 cyc/op at c=16 and 5.00 cyc/op
    //    at c=1 when VGPR banks are conflict-free: fa=v32 bank 0, fb=v33 bank 1, acc in banks 2/3).
    {
        auto s_lo = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<16, ULO>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int b_lo = verify_fma_base(16, ULO, d.out_f, do_break);
        auto s_mid = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<16, UMID>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int b_mid = verify_fma_base(16, UMID, d.out_f, do_break);
        auto s_hi = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<16, UHI>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int b_hi = verify_fma_base(16, UHI, d.out_f, do_break);
        const msad_ref::DiffCost dc16 = msad_ref::diff_cost(s_lo.median, s_mid.median, s_hi.median, ULO, UHI, ITERS, PATTERNS);

        auto d_lo = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<1, ULO>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int db_lo = verify_fma_base(1, ULO, d.out_f, do_break);
        auto d_mid = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<1, UMID>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int db_mid = verify_fma_base(1, UMID, d.out_f, do_break);
        auto d_hi = measure_kernel([&] { hipLaunchKernelGGL((k_fma_base<1, UHI>), dim3(1), dim3(WSIZE), 0, 0, d.fa, d.fb, d.f, d.cyc, d.out_f); }, d.cyc);
        int db_hi = verify_fma_base(1, UHI, d.out_f, do_break);
        const msad_ref::DiffCost dc1 = msad_ref::diff_cost(d_lo.median, d_mid.median, d_hi.median, ULO, UHI, ITERS, PATTERNS);

        const int bad = b_lo + b_mid + b_hi + db_lo + db_mid + db_hi;
        // Preregistered FMA calibration: wave32 accepts 1.00 +- 0.10; wave64 accepts 1.00 or 2.00 +- 0.10.
        const double cal_per = do_break ? dc1.per : dc16.per;
        const bool cal_ok = msad_ref::fma_calib_ok(WSIZE, bad, dc16, dc1, cal_per);
        std::printf("CONTROL msad_calib_fma w=%d c16=%.3f c1=%.3f mid16=%.2f mid1=%.2f exact=%s cal=%s\n",
                    WSIZE, dc16.per, dc1.per, dc16.mid, dc1.mid,
                    bad == 0 ? "ok" : "FAIL", cal_ok ? "PASS" : "FAIL");
        std::fflush(stdout);
        if (!cal_ok) ++fails;
    }

    // 3. All four SAD/MSAD instructions: c = 1 (acc), c = 1 (src0), c = 2, 4, 8, 16.
#define RUN_SAD_SUITE(OP, NAME)                                                           \
    do {                                                                                  \
        fails += run_sad_acc_case<OP, 1>(NAME "_acc", d, do_break, nullptr, &nonlin);     \
        fails += run_sad_src0_case<OP>(NAME "_src0", d, do_break, &nonlin);               \
        fails += run_sad_acc_case<OP, 2>(NAME "_acc", d, do_break, nullptr, &nonlin);     \
        fails += run_sad_acc_case<OP, 4>(NAME "_acc", d, do_break, nullptr, &nonlin);     \
        fails += run_sad_acc_case<OP, 8>(NAME "_acc", d, do_break, nullptr, &nonlin);     \
        fails += run_sad_acc_case<OP, 16>(NAME "_acc", d, do_break, nullptr, &nonlin);    \
    } while (0)

    RUN_SAD_SUITE(msad_ref::OP_MSAD_U8, "v_msad_u8");
    RUN_SAD_SUITE(msad_ref::OP_SAD_U8, "v_sad_u8");
    RUN_SAD_SUITE(msad_ref::OP_SAD_U16, "v_sad_u16");
    RUN_SAD_SUITE(msad_ref::OP_SAD_HI_U8, "v_sad_hi_u8");
#undef RUN_SAD_SUITE

    // 4. Co-issue tests mixed with FP32 FMA (1:1 and 1:3 for msad_u8, 1:1 for the other three).
    fails += run_mix_case<msad_ref::OP_MSAD_U8, 1>("msad_u8+1fma", d, do_break, &nonlin);
    fails += run_mix_case<msad_ref::OP_MSAD_U8, 3>("msad_u8+3fma", d, do_break, &nonlin);
    fails += run_mix_case<msad_ref::OP_SAD_U8, 1>("sad_u8+1fma", d, do_break, &nonlin);
    fails += run_mix_case<msad_ref::OP_SAD_U16, 1>("sad_u16+1fma", d, do_break, &nonlin);
    fails += run_mix_case<msad_ref::OP_SAD_HI_U8, 1>("sad_hi_u8+1fma", d, do_break, &nonlin);

    CHECK(hipFree(d.s0));
    CHECK(hipFree(d.s1));
    CHECK(hipFree(d.x));
    CHECK(hipFree(d.fa));
    CHECK(hipFree(d.fb));
    CHECK(hipFree(d.f));
    CHECK(hipFree(d.cyc));
    CHECK(hipFree(d.out_val));
    CHECK(hipFree(d.out_f));

    char vbuf[160];
    std::snprintf(vbuf, sizeof(vbuf), "w=%d fails=%d nonlin=%d%s",
                  WSIZE, fails, nonlin, do_break ? " (deliberate break)" : "");
    const bool ok = (fails == 0 && nonlin == 0);
    verdict(W64 ? "msad_w64" : "msad_w32", ok ? "PASS" : "FAIL", vbuf);
    return ok ? 0 : 1;
}
