// Reproducer for: the source operand of v_permlane16 / v_permlanex16 with fetch-inactive is computed inside the
// divergent region, so a switched-off partner lane never writes it.
//
// Semantics (RDNA 3/4 ISA guide; confirmed on gfx1201 by measurement): with the fetch-inactive bit set, the
// permute reads its source VGPR from lanes that are switched off in EXEC. A programmer who writes
//     v = f(x[lane]);                       // every lane computes v
//     if (cond(lane)) y = permlane16(y, v, sel0, sel1, /*fetch_inactive=*/true, false);
// therefore expects an active lane to receive the partner's v even if the partner failed cond.
//
// Compiler behaviour: v has no use outside the branch, so the compiler sinks its computation (the load of x and
// the arithmetic) into the s_and_saveexec region. A switched-off partner lane never executes it; its source
// VGPR still holds whatever the register had before (here: the initial value of y, or garbage), and that is
// what the active lane receives.
//
// Build (device code only):
//   hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx1201 repro.hip -o repro.s
// The kernels use only the builtin work-item id, so no device library is needed.
#include <hip/hip_runtime.h>

// Wave32, one wave per launch (32 threads). sel0/sel1 reverse each row of 16 lanes.
#define PERMLANE16(old, src) __builtin_amdgcn_permlane16((old), (src), 0x89ABCDEFu, 0x01234567u, /*fetch_inactive=*/true, false)
// Lane l reads lane l ^ 16.
#define PERMLANEX16(old, src) __builtin_amdgcn_permlanex16((old), (src), 0x76543210u, 0xFEDCBA98u, /*fetch_inactive=*/true, false)

// BAD: v is computed in natural C++ before the branch. The compiler sinks it into the branch.
__global__ void bad_sunk(const unsigned *X, unsigned *Y, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const unsigned v = X[l] * 3u + 1u;
    unsigned y = 0xDEAD0000u + l;
    if (l < n) {
        y = PERMLANE16(y, v);
    }
    Y[l] = y;
}

// GOOD: the empty asm pins v's definition in front of the branch (every lane executes it).
__global__ void good_pinned(const unsigned *X, unsigned *Y, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    unsigned v = X[l] * 3u + 1u;
    asm volatile("" : "+v"(v));
    unsigned y = 0xDEAD0000u + l;
    if (l < n) {
        y = PERMLANE16(y, v);
    }
    Y[l] = y;
}

// BAD, permlanex16 form (lane l reads lane l ^ 16).
__global__ void bad_sunk_x(const unsigned *X, unsigned *Y, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const unsigned v = X[l] * 3u + 1u;
    unsigned y = 0xDEAD0000u + l;
    if (l < n) {
        y = PERMLANEX16(y, v);
    }
    Y[l] = y;
}

// GOOD, permlanex16 form.
__global__ void good_pinned_x(const unsigned *X, unsigned *Y, int n) {
    const int l = __builtin_amdgcn_workitem_id_x();
    unsigned v = X[l] * 3u + 1u;
    asm volatile("" : "+v"(v));
    unsigned y = 0xDEAD0000u + l;
    if (l < n) {
        y = PERMLANEX16(y, v);
    }
    Y[l] = y;
}
