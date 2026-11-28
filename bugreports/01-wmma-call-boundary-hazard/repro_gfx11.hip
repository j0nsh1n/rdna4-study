// gfx11 (RDNA 3 / 3.5) form of repro.hip, using the gfx11 WMMA builtins (16-element A and B).
// Build (device code only):
//   hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx1151 repro_gfx11.hip -o repro_gfx11.s
#include <hip/hip_runtime.h>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define WMMA_F16(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32((a), (b), (c), false)
#define WMMA_F32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))

__device__ __attribute__((noinline)) v16h first_mm(v16h c, v16h a, v16h b) { return WMMA_F16(a, b, c); }
__device__ __attribute__((always_inline)) v16h first_mm_inl(v16h c, v16h a, v16h b) { return WMMA_F16(a, b, c); }

// BUG: callee ends with a WMMA, caller's next WMMA reads its result as A.
__global__ void caller_plain(const v16h *A, const v16h *B, const v16h *C16, const v16h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v16h b2 = B2[l];
    const v8f c2 = C2[l];
    const v16h d1 = first_mm(C16[l], A[l], B[l]);
    D[l] = WMMA_F32(d1, b2, c2);
}

// Workaround (listing-level; not run on hardware): an asm V_NOP tied to the returned value sits between the
// call and the second WMMA because the asm both reads and writes d1.
__global__ void workaround_asm_nop(const v16h *A, const v16h *B, const v16h *C16, const v16h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v16h b2 = B2[l];
    const v8f c2 = C2[l];
    v16h d1 = first_mm(C16[l], A[l], B[l]);
    asm volatile("v_nop" : "+v"(d1));
    D[l] = WMMA_F32(d1, b2, c2);
}

// Control: callee inlined. Expect a V_NOP between the two WMMAs.
__global__ void control_inline(const v16h *A, const v16h *B, const v16h *C16, const v16h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v16h d1 = first_mm_inl(C16[l], A[l], B[l]);
    D[l] = WMMA_F32(d1, B2[l], C2[l]);
}
