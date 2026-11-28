// CPU reference for Round 13 Probe A (msad): exact semantics of gfx1201 v_msad_u8,
// v_sad_u8, v_sad_u16, v_sad_hi_u8, exact dyadic FP32 FMA chains, and differential timing math.
// No HIP dependency: included by both src/round13/msad.hip and src/round13/msad_selftest.cpp.
#pragma once

#include <cmath>
#include <cstdint>

namespace msad_ref {

enum OpKind {
    OP_MSAD_U8 = 0,
    OP_SAD_U8 = 1,
    OP_SAD_U16 = 2,
    OP_SAD_HI_U8 = 3,
    OP_FMA_F32 = 4
};

// AMD RDNA ISA v_msad_u8 dst, src0, src1, src2:
// src1 is the masked reference word; any byte of src1 equal to 0 is excluded from the sum.
// Matches Mesa ACO (aco_select_nir_alu.cpp swaps NIR msad_4x8 src0/src1 into v_msad_u8) and
// LLVM __builtin_amdgcn_msad_u8(src0, src1, src2).
inline uint32_t msad_u8(uint32_t s0, uint32_t s1, uint32_t s2) {
    uint32_t acc = s2;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b0 = (s0 >> (8 * i)) & 0xffu;
        const uint32_t b1 = (s1 >> (8 * i)) & 0xffu;
        if (b1 != 0u) {
            acc += (b0 > b1) ? (b0 - b1) : (b1 - b0);
        }
    }
    return acc;
}

inline uint32_t sad_u8(uint32_t s0, uint32_t s1, uint32_t s2) {
    uint32_t acc = s2;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b0 = (s0 >> (8 * i)) & 0xffu;
        const uint32_t b1 = (s1 >> (8 * i)) & 0xffu;
        acc += (b0 > b1) ? (b0 - b1) : (b1 - b0);
    }
    return acc;
}

inline uint32_t sad_u16(uint32_t s0, uint32_t s1, uint32_t s2) {
    const uint32_t a0 = s0 & 0xffffu;
    const uint32_t a1 = s0 >> 16;
    const uint32_t b0 = s1 & 0xffffu;
    const uint32_t b1 = s1 >> 16;
    const uint32_t d0 = (a0 > b0) ? (a0 - b0) : (b0 - a0);
    const uint32_t d1 = (a1 > b1) ? (a1 - b1) : (b1 - a1);
    return s2 + d0 + d1;
}

inline uint32_t sad_hi_u8(uint32_t s0, uint32_t s1, uint32_t s2) {
    uint32_t d = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b0 = (s0 >> (8 * i)) & 0xffu;
        const uint32_t b1 = (s1 >> (8 * i)) & 0xffu;
        d += (b0 > b1) ? (b0 - b1) : (b1 - b0);
    }
    return s2 + (d << 16);
}

inline uint32_t eval_sad_op(int kind, uint32_t s0, uint32_t s1, uint32_t s2) {
    switch (kind) {
        case OP_MSAD_U8: return msad_u8(s0, s1, s2);
        case OP_SAD_U8: return sad_u8(s0, s1, s2);
        case OP_SAD_U16: return sad_u16(s0, s1, s2);
        case OP_SAD_HI_U8: return sad_hi_u8(s0, s1, s2);
        default: return 0u;
    }
}

// Deterministic per-lane test operands with asymmetric zero bytes in s0 vs s1 so msad_u8
// masking on src1 (and non-masking on src0) is exercised on every lane.
inline uint32_t lane_s0(uint32_t lane) {
    uint32_t b0 = ((lane * 37u + 13u) & 0xffu);
    uint32_t b1 = (lane & 1u) ? 0u : ((lane * 53u + 91u) & 0xffu);
    uint32_t b2 = ((lane * 71u + 200u) & 0xffu);
    uint32_t b3 = ((lane * 19u + 7u) & 0xffu);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

inline uint32_t lane_s1(uint32_t lane) {
    uint32_t b0 = ((lane * 29u + 41u) & 0xffu) | 1u;
    uint32_t b1 = ((lane * 61u + 17u) & 0xffu) | 1u;
    uint32_t b2 = (lane & 2u) ? 0u : (((lane * 83u + 5u) & 0xffu) | 1u);
    uint32_t b3 = (lane & 4u) ? 0u : (((lane * 11u + 129u) & 0xffu) | 1u);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

inline uint32_t lane_init_acc(uint32_t lane, uint32_t chain_idx) {
    return 0x10203040u ^ (lane * 0x01010101u) ^ (chain_idx * 0x00110011u);
}

// Exact accumulator-chain result after `steps` repetitions of x = op(s0, s1, x).
inline uint32_t run_acc_chain(int kind, uint32_t s0, uint32_t s1, uint32_t init_x, uint32_t steps) {
    const uint32_t delta = eval_sad_op(kind, s0, s1, 0u);
    return init_x + delta * steps;
}

// Exact src0-dependent chain result after `steps` repetitions of x = op(x, s1, s2).
inline uint32_t run_src0_chain(int kind, uint32_t init_x, uint32_t s1, uint32_t s2, uint32_t steps) {
    uint32_t x = init_x;
    for (uint32_t i = 0; i < steps; ++i) {
        x = eval_sad_op(kind, x, s1, s2);
    }
    return x;
}

// Exact dyadic FP32 FMA chain: f_{k+1} = fmaf(fa, fb, f_k) where fa = 1.0f, fb = 0.0625f.
// With steps <= 2^18 and f_0 in [0, 64] on a 1/16 grid, every addition is bit-exact in FP32.
inline float lane_init_f(uint32_t lane, uint32_t chain_idx) {
    return static_cast<float>((lane & 15u) + chain_idx) * 0.25f;
}

inline float run_fma_chain(float init_f, float fa, float fb, uint32_t steps) {
    float x = init_f;
    for (uint32_t i = 0; i < steps; ++i) {
        x = std::fmaf(fa, fb, x);
    }
    return x;
}

struct DiffCost {
    double per;   // cycles per pattern
    double mid;   // linearity residual at UMID (cycles per loop iteration)
    double ovh;   // loop control overhead (cycles per loop iteration)
    bool linear;  // |mid| <= 1.5 cycles/iteration
};

inline DiffCost diff_cost(double cyc_lo, double cyc_mid, double cyc_hi,
                          int u_lo, int u_hi, int iters, double patterns_per_block) {
    const double n = static_cast<double>(u_hi - u_lo) * iters * patterns_per_block;
    const double per = (cyc_hi - cyc_lo) / n;
    const double mid = (cyc_mid - 0.5 * (cyc_lo + cyc_hi)) / static_cast<double>(iters);
    const double ovh = (cyc_hi - per * static_cast<double>(u_hi) * iters * patterns_per_block) / static_cast<double>(iters);
    return {per, mid, ovh, std::fabs(mid) <= 1.5};
}

// Preregistered FP32 FMA c=16 calibration gate (Expectations §1):
//   wave32: 1.00 +- 0.10 cyc/op
//   wave64: 1.00 +- 0.10 OR 2.00 +- 0.10 cyc/op (64-wide 1-pass or dual 32-lane pass)
// Always requires exact bits (bad == 0) and linearity on both c=16 and c=1.
inline bool fma_calib_ok(int wave_size, int bad, const DiffCost & dc16, const DiffCost & dc1, double cal_per) {
    if (bad != 0 || !dc16.linear || !dc1.linear) return false;
    const bool near_1 = std::fabs(cal_per - 1.0) <= 0.10;
    if (wave_size == 64) {
        const bool near_2 = std::fabs(cal_per - 2.0) <= 0.10;
        return near_1 || near_2;
    }
    return near_1;
}

}  // namespace msad_ref
