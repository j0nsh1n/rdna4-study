// Blind Spot #4: Load-landing cost vs load width (b128 vs b64 vs b32 at equal bytes) and load
// clustering (loads clumped before 4 WMMAs vs spread before each WMMA), wave32 and wave64,
// L0 (8 KiB) and L2 (1 MiB), 1 wave and whole card (including matched 8 waves/SIMD for wave64).
//
// Build twice with -mcumode:
//   wave32: hipcc -O3 --offload-arch=gfx1201 -mcumode -mno-wavefrontsize64 -DHARNESS_ARCH=\"gfx1201\" load_cluster.hip -o load_cluster_w32
//   wave64: hipcc -O3 --offload-arch=gfx1201 -mcumode -mwavefrontsize64 -DW64=1 -DHARNESS_ARCH=\"gfx1201\" load_cluster.hip -o load_cluster_w64
#include "../study/mem_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef W64
#define W64 0
#endif

__device__ __forceinline__ uint64_t shader_cycles() {
    uint32_t hi1 = 0, lo = 0, hi2 = 0;
    asm volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)\n"
        "s_getreg_b32 %1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
        "s_getreg_b32 %2, hwreg(HW_REG_SHADER_CYCLES_HI)\n"
        : "=s"(hi1), "=s"(lo), "=s"(hi2));
    uint32_t hi = (hi1 == hi2) ? hi1 : ((lo & 0x80000000u) ? hi1 : hi2);
    return ((uint64_t) hi << 32) | lo;
}

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
#if W64
typedef float f32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ f32x4 wmma_fp8(i32x2 a, i32x2 b, f32x4 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(a[0], b[0], c);
}
constexpr int WSIZE = 64;
constexpr int CARD_WG_THREADS = 256;
constexpr int MAX_GRID = 256; // 256 WGs x 4 wave64 = 1024 wave64 = 8 waves/SIMD
#else
typedef float f32x8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ f32x8 wmma_fp8(i32x2 a, i32x2 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
constexpr int WSIZE = 32;
constexpr int CARD_WG_THREADS = 256;
constexpr int MAX_GRID = 128; // 128 WGs x 8 wave32 = 1024 wave32 = 8 waves/SIMD
#endif

struct Rec {
    uint64_t cyc;
    uint64_t rt;
    float check;
    uint32_t pl;
};

// Early-clobber SGPR-base + 32-bit VGPR-offset loads (no 64-bit carry math, no src/dst overlap).
__device__ __forceinline__ u4 ld128(const char * base, uint32_t off) {
    u4 v;
    asm volatile("global_load_b128 %0, %1, %2" : "=&v"(v) : "v"(off), "s"(base) : "memory");
    return v;
}
__device__ __forceinline__ u2 ld64(const char * base, uint32_t off) {
    u2 v;
    asm volatile("global_load_b64 %0, %1, %2" : "=&v"(v) : "v"(off), "s"(base) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld32(const char * base, uint32_t off) {
    unsigned v;
    asm volatile("global_load_b32 %0, %1, %2" : "=&v"(v) : "v"(off), "s"(base) : "memory");
    return v;
}

template <typename T>
__device__ __forceinline__ void wait_load_0(T & v) {
    asm volatile("s_wait_loadcnt 0" : "+v"(v) :: "memory");
}
__device__ __forceinline__ void wait_load_0_only() {
    asm volatile("s_wait_loadcnt 0" ::: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_1(T & v) {
    asm volatile("s_wait_loadcnt 1" : "+v"(v) :: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_2(T & v0, T & v1) {
    asm volatile("s_wait_loadcnt 2" : "+v"(v0), "+v"(v1) :: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_4_one(T & v0) {
    asm volatile("s_wait_loadcnt 4" : "+v"(v0) :: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_4(T & v0, T & v1, T & v2, T & v3) {
    asm volatile("s_wait_loadcnt 4" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3) :: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_8_pair(T & v0, T & v1) {
    asm volatile("s_wait_loadcnt 8" : "+v"(v0), "+v"(v1) :: "memory");
}
template <typename T>
__device__ __forceinline__ void wait_load_8(T & v0, T & v1, T & v2, T & v3,
                                            T & v4, T & v5, T & v6, T & v7) {
    asm volatile("s_wait_loadcnt 8"
                 : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3),
                   "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7)
                 :: "memory");
}
template <typename T>
__device__ __forceinline__ void keep_live(const T & v) {
    asm volatile("" :: "v"(v) : "memory");
}
template <typename Acc>
__device__ __forceinline__ void pin_acc(Acc & c) {
    asm volatile("" : "+v"(c) :: "memory");
}
// Pin ALL four accumulators across a clumped-half boundary so LLVM cannot float
// the next half's loads between the four v_wmma instructions.
template <typename Acc>
__device__ __forceinline__ void pin_accs4(Acc & c0, Acc & c1, Acc & c2, Acc & c3) {
    asm volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) :: "memory");
}

template <typename Acc>
__device__ __forceinline__ Acc wmma_u4(const u4 & v, Acc c) {
    Acc r = wmma_fp8(i32x2{(int) v.x, (int) v.y}, i32x2{(int) v.z, (int) v.w}, c);
    keep_live(v);
    return r;
}

// Patterns (all execute 8 WMMAs per loop iteration via ping-pong double buffering = 8*iters WMMAs):
//   0: ref            (register-fed only, 0 loads -> ~8.0 cyc/mm gate)
//   1: b128_1_spread  (1 x b128 spread before each WMMA; 16 B/lane = 512 B w32 / 1024 B w64)
//   2: b128_4_clump   (4 x b128 clumped before 4 WMMAs; same bytes as PAT 1, pinned schedule)
//   3: b64_2_spread   (2 x b64 spread before each WMMA; same 16 B/lane as PAT 1)
//   4: b64_8_clump    (8 x b64 clumped before 4 WMMAs; same 16 B/lane as PAT 2, pinned schedule)
//   5: b32_4_spread   (4 x b32 spread before each WMMA; same 16 B/lane as PAT 1)
//   6: b128_1_per4    (K=4: 1 x b128 per 4 WMMAs = 4 B/lane/WMMA)
//   7: b64_2_per4     (K=4: 2 x b64 per 4 WMMAs = 4 B/lane/WMMA)
//   8: b32_4_per4     (K=4: 4 x b32 per 4 WMMAs = 4 B/lane/WMMA)
//   9: b64_1_spread   (1 x b64 spread before each WMMA; 8 B/lane = 512 B/WMMA in w64 = exact A+B tile!)
//  10: b64_4_clump    (4 x b64 clumped before 4 WMMAs; 8 B/lane = 512 B/WMMA in w64, pinned schedule)
template <int PAT>
__global__ __launch_bounds__(256) void k_cluster(const char * __restrict__ buf, uint32_t span_mask,
                                                 uint32_t iters, Rec * __restrict__ recs) {
    const uint32_t lane = threadIdx.x & (WSIZE - 1);
    const uint32_t wid_global = (blockIdx.x * blockDim.x + threadIdx.x) / WSIZE;
    i32x2 b = {0x38383838, 0x38383838};
    asm volatile("" : "+v"(b));
#if W64
    f32x4 c0 = {0,0,0,0}, c1 = {0,0,0,0}, c2 = {0,0,0,0}, c3 = {0,0,0,0};
#else
    f32x8 c0 = {0,0,0,0,0,0,0,0}, c1 = {0,0,0,0,0,0,0,0}, c2 = {0,0,0,0,0,0,0,0}, c3 = {0,0,0,0,0,0,0,0};
#endif

    const uint32_t off_mask = span_mask;
    uint32_t off = (wid_global * 256u + lane * 16u) & off_mask;
    const uint32_t s16 = WSIZE * 16u;
    const uint32_t s8  = WSIZE * 8u;
    const uint32_t s4  = WSIZE * 4u;

    // Zero-seed pre-prime registers (NOT 0x38!) so missed loads/waits produce 0.0 instead of expect.
    u4 a0 = {0u, 0u, 0u, 0u}, a1 = a0, a2 = a0, a3 = a0;
    u4 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    u2 qa0 = {0u, 0u}, qa1 = qa0, qa2 = qa0, qa3 = qa0, qa4 = qa0, qa5 = qa0, qa6 = qa0, qa7 = qa0;
    u2 qb0 = qa0, qb1 = qa0, qb2 = qa0, qb3 = qa0, qb4 = qa0, qb5 = qa0, qb6 = qa0, qb7 = qa0;
    unsigned ra0 = 0u, ra1 = 0u, ra2 = 0u, ra3 = 0u;
    unsigned rb0 = 0u, rb1 = 0u, rb2 = 0u, rb3 = 0u;

    // Prime in-flight FIFO for ping-pong steady state (t=0 dispatch warms caches).
    if constexpr (PAT == 0) {
        a0 = ld128(buf, off);
        wait_load_0(a0);
    } else if constexpr (PAT == 1 || PAT == 2) {
        a0 = ld128(buf, off & off_mask); off += s16;
        a1 = ld128(buf, off & off_mask); off += s16;
        a2 = ld128(buf, off & off_mask); off += s16;
        a3 = ld128(buf, off & off_mask); off += s16;
    } else if constexpr (PAT == 3 || PAT == 4) {
        qa0 = ld64(buf, off & off_mask); off += s8;
        qa1 = ld64(buf, off & off_mask); off += s8;
        qa2 = ld64(buf, off & off_mask); off += s8;
        qa3 = ld64(buf, off & off_mask); off += s8;
        qa4 = ld64(buf, off & off_mask); off += s8;
        qa5 = ld64(buf, off & off_mask); off += s8;
        qa6 = ld64(buf, off & off_mask); off += s8;
        qa7 = ld64(buf, off & off_mask); off += s8;
    } else if constexpr (PAT == 5 || PAT == 8) {
        ra0 = ld32(buf, off & off_mask); off += s4;
        ra1 = ld32(buf, off & off_mask); off += s4;
        ra2 = ld32(buf, off & off_mask); off += s4;
        ra3 = ld32(buf, off & off_mask); off += s4;
    } else if constexpr (PAT == 6) {
        a0 = ld128(buf, off & off_mask); off += s16;
    } else if constexpr (PAT == 7) {
        qa0 = ld64(buf, off & off_mask); off += s8;
        qa1 = ld64(buf, off & off_mask); off += s8;
    } else if constexpr (PAT == 9 || PAT == 10) {
        qa0 = ld64(buf, off & off_mask); off += s8;
        qa1 = ld64(buf, off & off_mask); off += s8;
        qa2 = ld64(buf, off & off_mask); off += s8;
        qa3 = ld64(buf, off & off_mask); off += s8;
    }

    __syncthreads();
    const uint64_t t0 = shader_cycles();
    const uint64_t rt0 = mem::realtime();

#pragma unroll 1
    for (uint32_t i = 0; i < iters; i++) {
        if constexpr (PAT == 0) {
            i32x2 aa = {(int) a0.x, (int) a0.y};
            c0 = wmma_fp8(aa, b, c0);
            c1 = wmma_fp8(aa, b, c1);
            c2 = wmma_fp8(aa, b, c2);
            c3 = wmma_fp8(aa, b, c3);
            c0 = wmma_fp8(aa, b, c0);
            c1 = wmma_fp8(aa, b, c1);
            c2 = wmma_fp8(aa, b, c2);
            c3 = wmma_fp8(aa, b, c3);
        } else if constexpr (PAT == 1) {
            b0 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(a0); c0 = wmma_u4(a0, c0); pin_acc(c0);
            b1 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(a1); c1 = wmma_u4(a1, c1); pin_acc(c1);
            b2 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(a2); c2 = wmma_u4(a2, c2); pin_acc(c2);
            b3 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(a3); c3 = wmma_u4(a3, c3); pin_acc(c3);
            a0 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(b0); c0 = wmma_u4(b0, c0); pin_acc(c0);
            a1 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(b1); c1 = wmma_u4(b1, c1); pin_acc(c1);
            a2 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(b2); c2 = wmma_u4(b2, c2); pin_acc(c2);
            a3 = ld128(buf, off & off_mask); off += s16; wait_load_4_one(b3); c3 = wmma_u4(b3, c3); pin_acc(c3);
        } else if constexpr (PAT == 2) {
            // True clumped: 4 loads, 1 wait, 4 WMMAs pinned together, repeat
            b0 = ld128(buf, off & off_mask); off += s16;
            b1 = ld128(buf, off & off_mask); off += s16;
            b2 = ld128(buf, off & off_mask); off += s16;
            b3 = ld128(buf, off & off_mask); off += s16;
            wait_load_4(a0, a1, a2, a3);
            c0 = wmma_u4(a0, c0);
            c1 = wmma_u4(a1, c1);
            c2 = wmma_u4(a2, c2);
            c3 = wmma_u4(a3, c3);
            pin_accs4(c0, c1, c2, c3);
            a0 = ld128(buf, off & off_mask); off += s16;
            a1 = ld128(buf, off & off_mask); off += s16;
            a2 = ld128(buf, off & off_mask); off += s16;
            a3 = ld128(buf, off & off_mask); off += s16;
            wait_load_4(b0, b1, b2, b3);
            c0 = wmma_u4(b0, c0);
            c1 = wmma_u4(b1, c1);
            c2 = wmma_u4(b2, c2);
            c3 = wmma_u4(b3, c3);
            pin_accs4(c0, c1, c2, c3);
        } else if constexpr (PAT == 3) {
            qb0 = ld64(buf, off & off_mask); off += s8; qb1 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qa0, qa1); c0 = wmma_fp8(i32x2{(int) qa0.x, (int) qa0.y}, i32x2{(int) qa1.x, (int) qa1.y}, c0); keep_live(qa0); keep_live(qa1); pin_acc(c0);
            qb2 = ld64(buf, off & off_mask); off += s8; qb3 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qa2, qa3); c1 = wmma_fp8(i32x2{(int) qa2.x, (int) qa2.y}, i32x2{(int) qa3.x, (int) qa3.y}, c1); keep_live(qa2); keep_live(qa3); pin_acc(c1);
            qb4 = ld64(buf, off & off_mask); off += s8; qb5 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qa4, qa5); c2 = wmma_fp8(i32x2{(int) qa4.x, (int) qa4.y}, i32x2{(int) qa5.x, (int) qa5.y}, c2); keep_live(qa4); keep_live(qa5); pin_acc(c2);
            qb6 = ld64(buf, off & off_mask); off += s8; qb7 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qa6, qa7); c3 = wmma_fp8(i32x2{(int) qa6.x, (int) qa6.y}, i32x2{(int) qa7.x, (int) qa7.y}, c3); keep_live(qa6); keep_live(qa7); pin_acc(c3);
            qa0 = ld64(buf, off & off_mask); off += s8; qa1 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qb0, qb1); c0 = wmma_fp8(i32x2{(int) qb0.x, (int) qb0.y}, i32x2{(int) qb1.x, (int) qb1.y}, c0); keep_live(qb0); keep_live(qb1); pin_acc(c0);
            qa2 = ld64(buf, off & off_mask); off += s8; qa3 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qb2, qb3); c1 = wmma_fp8(i32x2{(int) qb2.x, (int) qb2.y}, i32x2{(int) qb3.x, (int) qb3.y}, c1); keep_live(qb2); keep_live(qb3); pin_acc(c1);
            qa4 = ld64(buf, off & off_mask); off += s8; qa5 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qb4, qb5); c2 = wmma_fp8(i32x2{(int) qb4.x, (int) qb4.y}, i32x2{(int) qb5.x, (int) qb5.y}, c2); keep_live(qb4); keep_live(qb5); pin_acc(c2);
            qa6 = ld64(buf, off & off_mask); off += s8; qa7 = ld64(buf, off & off_mask); off += s8;
            wait_load_8_pair(qb6, qb7); c3 = wmma_fp8(i32x2{(int) qb6.x, (int) qb6.y}, i32x2{(int) qb7.x, (int) qb7.y}, c3); keep_live(qb6); keep_live(qb7); pin_acc(c3);
        } else if constexpr (PAT == 4) {
            qb0 = ld64(buf, off & off_mask); off += s8; qb1 = ld64(buf, off & off_mask); off += s8;
            qb2 = ld64(buf, off & off_mask); off += s8; qb3 = ld64(buf, off & off_mask); off += s8;
            qb4 = ld64(buf, off & off_mask); off += s8; qb5 = ld64(buf, off & off_mask); off += s8;
            qb6 = ld64(buf, off & off_mask); off += s8; qb7 = ld64(buf, off & off_mask); off += s8;
            wait_load_8(qa0, qa1, qa2, qa3, qa4, qa5, qa6, qa7);
            c0 = wmma_fp8(i32x2{(int) qa0.x, (int) qa0.y}, i32x2{(int) qa1.x, (int) qa1.y}, c0);
            c1 = wmma_fp8(i32x2{(int) qa2.x, (int) qa2.y}, i32x2{(int) qa3.x, (int) qa3.y}, c1);
            c2 = wmma_fp8(i32x2{(int) qa4.x, (int) qa4.y}, i32x2{(int) qa5.x, (int) qa5.y}, c2);
            c3 = wmma_fp8(i32x2{(int) qa6.x, (int) qa6.y}, i32x2{(int) qa7.x, (int) qa7.y}, c3);
            keep_live(qa0); keep_live(qa1); keep_live(qa2); keep_live(qa3);
            keep_live(qa4); keep_live(qa5); keep_live(qa6); keep_live(qa7);
            pin_accs4(c0, c1, c2, c3);
            qa0 = ld64(buf, off & off_mask); off += s8; qa1 = ld64(buf, off & off_mask); off += s8;
            qa2 = ld64(buf, off & off_mask); off += s8; qa3 = ld64(buf, off & off_mask); off += s8;
            qa4 = ld64(buf, off & off_mask); off += s8; qa5 = ld64(buf, off & off_mask); off += s8;
            qa6 = ld64(buf, off & off_mask); off += s8; qa7 = ld64(buf, off & off_mask); off += s8;
            wait_load_8(qb0, qb1, qb2, qb3, qb4, qb5, qb6, qb7);
            c0 = wmma_fp8(i32x2{(int) qb0.x, (int) qb0.y}, i32x2{(int) qb1.x, (int) qb1.y}, c0);
            c1 = wmma_fp8(i32x2{(int) qb2.x, (int) qb2.y}, i32x2{(int) qb3.x, (int) qb3.y}, c1);
            c2 = wmma_fp8(i32x2{(int) qb4.x, (int) qb4.y}, i32x2{(int) qb5.x, (int) qb5.y}, c2);
            c3 = wmma_fp8(i32x2{(int) qb6.x, (int) qb6.y}, i32x2{(int) qb7.x, (int) qb7.y}, c3);
            keep_live(qb0); keep_live(qb1); keep_live(qb2); keep_live(qb3);
            keep_live(qb4); keep_live(qb5); keep_live(qb6); keep_live(qb7);
            pin_accs4(c0, c1, c2, c3);
        } else if constexpr (PAT == 5) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                rb0 = ld32(buf, off & off_mask); off += s4; rb1 = ld32(buf, off & off_mask); off += s4;
                rb2 = ld32(buf, off & off_mask); off += s4; rb3 = ld32(buf, off & off_mask); off += s4;
                wait_load_4(ra0, ra1, ra2, ra3);
                i32x2 aa = {(int) ra0, (int) ra1}, bb = {(int) ra2, (int) ra3};
                if (k == 0) { c0 = wmma_fp8(aa, bb, c0); pin_acc(c0); }
                else if (k == 1) { c1 = wmma_fp8(aa, bb, c1); pin_acc(c1); }
                else if (k == 2) { c2 = wmma_fp8(aa, bb, c2); pin_acc(c2); }
                else { c3 = wmma_fp8(aa, bb, c3); pin_acc(c3); }
                keep_live(ra0); keep_live(ra1); keep_live(ra2); keep_live(ra3);

                ra0 = ld32(buf, off & off_mask); off += s4; ra1 = ld32(buf, off & off_mask); off += s4;
                ra2 = ld32(buf, off & off_mask); off += s4; ra3 = ld32(buf, off & off_mask); off += s4;
                wait_load_4(rb0, rb1, rb2, rb3);
                i32x2 aa2 = {(int) rb0, (int) rb1}, bb2 = {(int) rb2, (int) rb3};
                if (k == 0) { c0 = wmma_fp8(aa2, bb2, c0); pin_acc(c0); }
                else if (k == 1) { c1 = wmma_fp8(aa2, bb2, c1); pin_acc(c1); }
                else if (k == 2) { c2 = wmma_fp8(aa2, bb2, c2); pin_acc(c2); }
                else { c3 = wmma_fp8(aa2, bb2, c3); pin_acc(c3); }
                keep_live(rb0); keep_live(rb1); keep_live(rb2); keep_live(rb3);
            }
        } else if constexpr (PAT == 6) {
            b0 = ld128(buf, off & off_mask); off += s16; wait_load_1(a0);
            c0 = wmma_u4(a0, c0); c1 = wmma_u4(a0, c1); c2 = wmma_u4(a0, c2); c3 = wmma_u4(a0, c3); pin_accs4(c0, c1, c2, c3);
            a0 = ld128(buf, off & off_mask); off += s16; wait_load_1(b0);
            c0 = wmma_u4(b0, c0); c1 = wmma_u4(b0, c1); c2 = wmma_u4(b0, c2); c3 = wmma_u4(b0, c3); pin_accs4(c0, c1, c2, c3);
        } else if constexpr (PAT == 7) {
            qb0 = ld64(buf, off & off_mask); off += s8; qb1 = ld64(buf, off & off_mask); off += s8; wait_load_2(qa0, qa1);
            i32x2 aa = {(int) qa0.x, (int) qa0.y}, bb = {(int) qa1.x, (int) qa1.y};
            c0 = wmma_fp8(aa, bb, c0); c1 = wmma_fp8(aa, bb, c1); c2 = wmma_fp8(aa, bb, c2); c3 = wmma_fp8(aa, bb, c3);
            keep_live(qa0); keep_live(qa1); pin_accs4(c0, c1, c2, c3);
            qa0 = ld64(buf, off & off_mask); off += s8; qa1 = ld64(buf, off & off_mask); off += s8; wait_load_2(qb0, qb1);
            i32x2 aa2 = {(int) qb0.x, (int) qb0.y}, bb2 = {(int) qb1.x, (int) qb1.y};
            c0 = wmma_fp8(aa2, bb2, c0); c1 = wmma_fp8(aa2, bb2, c1); c2 = wmma_fp8(aa2, bb2, c2); c3 = wmma_fp8(aa2, bb2, c3);
            keep_live(qb0); keep_live(qb1); pin_accs4(c0, c1, c2, c3);
        } else if constexpr (PAT == 8) {
            rb0 = ld32(buf, off & off_mask); off += s4; rb1 = ld32(buf, off & off_mask); off += s4;
            rb2 = ld32(buf, off & off_mask); off += s4; rb3 = ld32(buf, off & off_mask); off += s4; wait_load_4(ra0, ra1, ra2, ra3);
            i32x2 aa = {(int) ra0, (int) ra1}, bb = {(int) ra2, (int) ra3};
            c0 = wmma_fp8(aa, bb, c0); c1 = wmma_fp8(aa, bb, c1); c2 = wmma_fp8(aa, bb, c2); c3 = wmma_fp8(aa, bb, c3);
            keep_live(ra0); keep_live(ra1); keep_live(ra2); keep_live(ra3); pin_accs4(c0, c1, c2, c3);
            ra0 = ld32(buf, off & off_mask); off += s4; ra1 = ld32(buf, off & off_mask); off += s4;
            ra2 = ld32(buf, off & off_mask); off += s4; ra3 = ld32(buf, off & off_mask); off += s4; wait_load_4(rb0, rb1, rb2, rb3);
            i32x2 aa2 = {(int) rb0, (int) rb1}, bb2 = {(int) rb2, (int) rb3};
            c0 = wmma_fp8(aa2, bb2, c0); c1 = wmma_fp8(aa2, bb2, c1); c2 = wmma_fp8(aa2, bb2, c2); c3 = wmma_fp8(aa2, bb2, c3);
            keep_live(rb0); keep_live(rb1); keep_live(rb2); keep_live(rb3); pin_accs4(c0, c1, c2, c3);
        } else if constexpr (PAT == 9) {
            // 1 x b64 spread before each WMMA (8 B/lane/WMMA = 512 B/WMMA in wave64 = exact A+B tile!)
            qb0 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qa0); c0 = wmma_fp8(i32x2{(int) qa0.x, (int) qa0.y}, i32x2{(int) qa0.y, (int) qa0.x}, c0); keep_live(qa0); pin_acc(c0);
            qb1 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qa1); c1 = wmma_fp8(i32x2{(int) qa1.x, (int) qa1.y}, i32x2{(int) qa1.y, (int) qa1.x}, c1); keep_live(qa1); pin_acc(c1);
            qb2 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qa2); c2 = wmma_fp8(i32x2{(int) qa2.x, (int) qa2.y}, i32x2{(int) qa2.y, (int) qa2.x}, c2); keep_live(qa2); pin_acc(c2);
            qb3 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qa3); c3 = wmma_fp8(i32x2{(int) qa3.x, (int) qa3.y}, i32x2{(int) qa3.y, (int) qa3.x}, c3); keep_live(qa3); pin_acc(c3);
            qa0 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qb0); c0 = wmma_fp8(i32x2{(int) qb0.x, (int) qb0.y}, i32x2{(int) qb0.y, (int) qb0.x}, c0); keep_live(qb0); pin_acc(c0);
            qa1 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qb1); c1 = wmma_fp8(i32x2{(int) qb1.x, (int) qb1.y}, i32x2{(int) qb1.y, (int) qb1.x}, c1); keep_live(qb1); pin_acc(c1);
            qa2 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qb2); c2 = wmma_fp8(i32x2{(int) qb2.x, (int) qb2.y}, i32x2{(int) qb2.y, (int) qb2.x}, c2); keep_live(qb2); pin_acc(c2);
            qa3 = ld64(buf, off & off_mask); off += s8; wait_load_4_one(qb3); c3 = wmma_fp8(i32x2{(int) qb3.x, (int) qb3.y}, i32x2{(int) qb3.y, (int) qb3.x}, c3); keep_live(qb3); pin_acc(c3);
        } else if constexpr (PAT == 10) {
            // 4 x b64 clumped before 4 WMMAs (8 B/lane/WMMA = 512 B/WMMA in wave64, pinned schedule)
            qb0 = ld64(buf, off & off_mask); off += s8;
            qb1 = ld64(buf, off & off_mask); off += s8;
            qb2 = ld64(buf, off & off_mask); off += s8;
            qb3 = ld64(buf, off & off_mask); off += s8;
            wait_load_4(qa0, qa1, qa2, qa3);
            c0 = wmma_fp8(i32x2{(int) qa0.x, (int) qa0.y}, i32x2{(int) qa0.y, (int) qa0.x}, c0);
            c1 = wmma_fp8(i32x2{(int) qa1.x, (int) qa1.y}, i32x2{(int) qa1.y, (int) qa1.x}, c1);
            c2 = wmma_fp8(i32x2{(int) qa2.x, (int) qa2.y}, i32x2{(int) qa2.y, (int) qa2.x}, c2);
            c3 = wmma_fp8(i32x2{(int) qa3.x, (int) qa3.y}, i32x2{(int) qa3.y, (int) qa3.x}, c3);
            keep_live(qa0); keep_live(qa1); keep_live(qa2); keep_live(qa3);
            pin_accs4(c0, c1, c2, c3);
            qa0 = ld64(buf, off & off_mask); off += s8;
            qa1 = ld64(buf, off & off_mask); off += s8;
            qa2 = ld64(buf, off & off_mask); off += s8;
            qa3 = ld64(buf, off & off_mask); off += s8;
            wait_load_4(qb0, qb1, qb2, qb3);
            c0 = wmma_fp8(i32x2{(int) qb0.x, (int) qb0.y}, i32x2{(int) qb0.y, (int) qb0.x}, c0);
            c1 = wmma_fp8(i32x2{(int) qb1.x, (int) qb1.y}, i32x2{(int) qb1.y, (int) qb1.x}, c1);
            c2 = wmma_fp8(i32x2{(int) qb2.x, (int) qb2.y}, i32x2{(int) qb2.y, (int) qb2.x}, c2);
            c3 = wmma_fp8(i32x2{(int) qb3.x, (int) qb3.y}, i32x2{(int) qb3.y, (int) qb3.x}, c3);
            keep_live(qb0); keep_live(qb1); keep_live(qb2); keep_live(qb3);
            pin_accs4(c0, c1, c2, c3);
        }
    }
    wait_load_0_only();
    const uint64_t t1 = shader_cycles();
    const uint64_t rt1 = mem::realtime();

    if (lane == 0) {
        recs[wid_global] = {t1 - t0, rt1 - rt0, c0[0] + c1[0] + c2[0] + c3[0], mem::placement()};
    }
}

static double median_v(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <int PAT>
static void run_pat(const char * pname, const char * mode, const char * lvl, const char * d_buf,
                    uint32_t span_bytes, uint32_t iters, int grid, int blk, bool do_break,
                    Rec * d_rec, std::vector<Rec> & h_rec, int & fail) {
    const int nwaves = (grid * blk) / WSIZE;
    const bool card = (grid > 1);
    const uint32_t mask = span_bytes - 16u;
    const double expect = 128.0 * (double) iters + (do_break ? 1.0 : 0.0);

    std::vector<double> cyc_mm, ghz_v;
    bool ok = true;
    for (int t = 0; t < 4; t++) {
        hipLaunchKernelGGL(k_cluster<PAT>, dim3(grid), dim3(blk), 0, 0, d_buf, mask, iters, d_rec);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        if (t == 0) continue; // warm
        CHECK(hipMemcpy(h_rec.data(), d_rec, nwaves * sizeof(Rec), hipMemcpyDeviceToHost));
        double total_mm = 8.0 * (double) iters;
        if (!card) {
            cyc_mm.push_back((double) h_rec[0].cyc / total_mm);
            ghz_v.push_back((double) h_rec[0].cyc / ((double) h_rec[0].rt * 10.0));
            if (std::fabs(h_rec[0].check - expect) > 0.5) ok = false;
        } else {
            std::vector<double> per_simd;
            std::vector<uint32_t> simds;
            for (int w = 0; w < nwaves; w++) {
                if (std::fabs(h_rec[w].check - expect) > 0.5) ok = false;
                simds.push_back(h_rec[w].pl & 2047u);
            }
            std::sort(simds.begin(), simds.end());
            simds.erase(std::unique(simds.begin(), simds.end()), simds.end());
            for (uint32_t sd : simds) {
                int cnt = 0;
                uint64_t cmax = 0;
                for (int w = 0; w < nwaves; w++) {
                    if ((h_rec[w].pl & 2047u) == sd) {
                        cnt++;
                        if (h_rec[w].cyc > cmax) cmax = h_rec[w].cyc;
                    }
                }
                if (cnt > 0) per_simd.push_back((double) cmax / ((double) cnt * total_mm));
            }
            cyc_mm.push_back(median_v(per_simd));
            ghz_v.push_back((double) h_rec[0].cyc / ((double) h_rec[0].rt * 10.0));
        }
    }
    double m = median_v(cyc_mm), lo = *std::min_element(cyc_mm.begin(), cyc_mm.end()), hi = *std::max_element(cyc_mm.begin(), cyc_mm.end());
    std::printf("RESULT load_cluster wave=%d mode=%-7s w/simd=%d lvl=%-2s pat=%-14s cyc/mm=%.2f (%.2f-%.2f) sclk=%.2fGHz exact=%s\n",
                WSIZE, mode, card ? (nwaves / 128) : 1, lvl, pname, m, lo, hi, median_v(ghz_v), ok ? "ok" : "FAIL");
    std::fflush(stdout);
    if (!ok) fail++;
    if (PAT == 0) {
        // Strict preregistered 7.9-8.3 gate for single-wave ref; multi-wave card cmax includes launch/barrier skew (7.9-9.2)
        double lo_b = 7.9, hi_b = card ? 9.2 : 8.3;
        if (m < lo_b || m > hi_b) {
            std::printf("CONTROL load_cluster_ref (%s): FAIL expected %.1f-%.1f cyc/mm, got %.2f\n", mode, lo_b, hi_b, m);
            fail++;
        }
    }
}

int main(int argc, char ** argv) {
    bool do_break = false;
    for (int i = 1; i < argc; i++) if (!std::strcmp(argv[i], "--break")) do_break = true;

    const size_t kBuf = 2u << 20;
    char * d_buf = nullptr;
    Rec * d_rec = nullptr;
    CHECK(hipMalloc(&d_buf, kBuf));
    CHECK(hipMemset(d_buf, 0x38, kBuf)); // 0x38 = E4M3 1.0f everywhere
    const int max_waves = (MAX_GRID * CARD_WG_THREADS) / WSIZE;
    CHECK(hipMalloc(&d_rec, max_waves * sizeof(Rec)));
    std::vector<Rec> h_rec(max_waves);

    int fail = 0;
    const uint32_t iters = 2048; // 8 WMMAs/iter -> 16384 WMMAs per wave

    struct Cfg { const char * name; int grid; int blk; };
    std::vector<Cfg> cfgs;
    cfgs.push_back({"one", 1, WSIZE});
#if W64
    cfgs.push_back({"card_4w", 128, CARD_WG_THREADS}); // equal thread count (4 wave64/SIMD)
    cfgs.push_back({"card_8w", 256, CARD_WG_THREADS}); // matched wave occupancy (8 wave64/SIMD)
#else
    cfgs.push_back({"card_8w", 128, CARD_WG_THREADS}); // 8 wave32/SIMD
#endif

    for (const auto & cfg : cfgs) {
        run_pat<0>("ref", cfg.name, "RG", d_buf, 8192, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
        for (uint32_t span : {8192u, 1048576u}) {
            const char * lname = (span == 8192u) ? "L0" : "L2";
            run_pat<1>("b128_1_spread", cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<2>("b128_4_clump",  cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<3>("b64_2_spread",  cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<4>("b64_8_clump",   cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<5>("b32_4_spread",  cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<6>("b128_1_per4",   cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<7>("b64_2_per4",    cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<8>("b32_4_per4",    cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<9>("b64_1_spread",  cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
            run_pat<10>("b64_4_clump",  cfg.name, lname, d_buf, span, iters, cfg.grid, cfg.blk, do_break, d_rec, h_rec, fail);
        }
    }

    CHECK(hipFree(d_buf));
    CHECK(hipFree(d_rec));
    std::printf("VERDICT load_cluster_w%d: %s (fail=%d%s)\n", WSIZE, fail == 0 ? "PASS" : "FAIL", fail,
                do_break ? ", deliberate break" : "");
    return fail == 0 ? 0 : 1;
}
