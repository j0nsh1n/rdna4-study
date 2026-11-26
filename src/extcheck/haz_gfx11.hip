// Check 1, RDNA 3.5 (gfx1150/gfx1151): the call-boundary and per-lane-if shapes of haz_hazard.hip / haz_mask.hip
// rewritten with the gfx11 matrix builtins (16-element A and B). Listing only; never run on a card here.
#include <hip/hip_runtime.h>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define WMMA16(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32((a), (b), (c), false)
#define WMMA32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))

__global__ void h_straight(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h d1 = WMMA16(A[l], B[l], C16[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

__device__ __attribute__((noinline)) v16h first_mm_tied(v16h c, v16h a, v16h b) { return WMMA16(a, b, c); }

__global__ void h_call3(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v16h b2 = B2[l];
    v8f c2 = C2[l];
    asm volatile("" : "+v"(b2), "+v"(c2));
    const v16h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, b2, c2);
}

__global__ void h_call4(const v16h * A, const v16h * B, const v16h * C16, const v16h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v16h b2 = B2[l];
    const v8f c2 = C2[l];
    const v16h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, b2, c2);
}

// Everything, loads included, under a per-lane if (as k_ifload in haz_mask.hip).
__global__ void k_ifload(const v16h * A, const v16h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    if (l < n) {
        D[l] = WMMA32(A[l], B[l], C[l]);
    }
}
