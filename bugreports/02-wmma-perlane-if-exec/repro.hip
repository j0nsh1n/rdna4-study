// Reproducer for: WMMA/SWMMAC builtins are accepted and compiled inside lane-divergent control flow
// (gfx11 and gfx12), although the instruction ignores EXEC.
//
// ISA rule (AMD RDNA 3/4 ISA guide, WMMA section; confirmed on gfx1201 by measurement): a WMMA/SWMMAC
// instruction executes as if EXEC were all ones. It reads every lane's A, B and C registers and writes every
// lane's D registers, whatever the current EXEC mask is.
//
// Compiler behaviour: when the builtin is under a per-lane condition, the compiler emits the instruction
// inside the s_and_saveexec / v_cmpx region with no diagnostic. It then assumes that the switched-off lanes'
// registers are untouched. They are not: their operand registers are read (so they must hold meaningful
// values) and their result registers are overwritten.
//
// Build (device code only):
//   hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx1201 repro.hip -o repro.s
// The kernels use only the builtin work-item id, so no device library is needed.
#include <hip/hip_runtime.h>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

// One wave32 kernel (launch with 32 threads). Lane l owns operand slots A[l], B[l], C[l] and result D[l].

// BAD 1: everything under a per-lane if. The WMMA lands inside the s_and_saveexec region.
__global__ void bad_if(const v8h *A, const v8h *B, const v8f *C, v8f *D, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    if (l < n) {
        D[l] = WMMA(A[l], B[l], C[l]);
    }
}

// BAD 2: early return for the lanes past the end of a partly filled tile. Same result.
__global__ void bad_early_return(const v8h *A, const v8h *B, const v8f *C, v8f *D, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    if (l >= n) {
        return;
    }
    D[l] = WMMA(A[l], B[l], C[l]);
}

// BAD 3: loop with a per-lane trip count. The WMMA runs under v_cmpx; lanes that left the loop early still
// receive the result of the lanes that stayed.
__global__ void bad_perlane_loop(const v8h *A, const v8h *B, const v8f *C, v8f *D, const int *trip) {
    const int l = __builtin_amdgcn_workitem_id_x();
    v8f c = C[l];
    for (int i = 0; i < trip[l]; ++i) {
        c = WMMA(A[i * 32 + l], B[i * 32 + l], c);
    }
    D[l] = c;
}

// GOOD (control): pad the partial tile with a select. Every lane executes the WMMA under full EXEC.
__global__ void good_select(const v8h *A, const v8h *B, const v8f *C, v8f *D, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const v8h zero = {};
    const v8h a = (l & 15) < n ? A[l] : zero;
    D[l] = WMMA(a, B[l], C[l]);
}
