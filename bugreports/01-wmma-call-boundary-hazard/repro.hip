// Reproducer for: AMDGPU WMMA read-after-write hazard is not handled across a call boundary (gfx12).
//
// A WMMA whose result is used as the A or B operand of the next WMMA needs a V_NOP (or other VALU
// instruction) between the two. The hazard recognizer inserts one when both WMMAs are in the same
// function. When the first WMMA is the last instruction of a non-inlined callee and the second WMMA
// directly follows the call in the caller, nothing is inserted on either side of the call.
//
// Build (device code only):
//   hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx1201 repro.hip -o repro.s
// The kernels use only the builtin work-item id, so no device library is needed.
#include <hip/hip_runtime.h>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define WMMA_F16(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12((a), (b), (c))
#define WMMA_F32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

// Callee: ends with a WMMA and returns its result untouched (C is the first argument, so no copy is needed).
__device__ __attribute__((noinline)) v8h first_mm(v8h c, v8h a, v8h b) { return WMMA_F16(a, b, c); }

// Same callee, but always inlined: the compiler is expected to insert the V_NOP here (control).
__device__ __attribute__((always_inline)) v8h first_mm_inl(v8h c, v8h a, v8h b) { return WMMA_F16(a, b, c); }

// BUG: plain C++. The second WMMA reads the callee's result as its A operand, right after the call.
__global__ void caller_plain(const v8h *A, const v8h *B, const v8h *C16, const v8h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v8h b2 = B2[l];
    const v8f c2 = C2[l];
    const v8h d1 = first_mm(C16[l], A[l], B[l]);
    D[l] = WMMA_F32(d1, b2, c2);
}

// Workaround (listing-level; not run on hardware): an asm V_NOP tied to the returned value sits between the
// call and the second WMMA because the asm both reads and writes d1.
__global__ void workaround_asm_nop(const v8h *A, const v8h *B, const v8h *C16, const v8h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v8h b2 = B2[l];
    const v8f c2 = C2[l];
    v8h d1 = first_mm(C16[l], A[l], B[l]);
    asm volatile("v_nop" : "+v"(d1));
    D[l] = WMMA_F32(d1, b2, c2);
}

// Control: callee inlined. Expect a V_NOP between the two WMMAs.
__global__ void control_inline(const v8h *A, const v8h *B, const v8h *C16, const v8h *B2, const v8f *C2, v8f *D) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v8h d1 = first_mm_inl(C16[l], A[l], B[l]);
    D[l] = WMMA_F32(d1, B2[l], C2[l]);
}
