// Lane 6: the vector dot-product instructions as one step of the matrix unit's adder (lane 5,
// src/study/arith_model.h and measurements/matrix-arithmetic.md), with two additions found here, both visible only
// with BF16 inputs (the only ones whose products reach the FP32 subnormal range, which lane 5's data never did):
// the anchor is never below -126, the FP32 minimum exponent (emin_anchor), so the unit is never finer than 2^-152;
// and the rule "-0 gives +0" applies to the FP32 rounding, before a BF16 result is cut to 16 bits.
//
// One output: products p_i = a_i * b_i (exact) and the accumulator C.
//   anchor E = max(ea + eb + 2 over the nonzero products, exponent of C), exponents of subnormals counted as the
//   format's minimum, and E at least emin_anchor; unit U = 2^(E - 26).
//   Products are summed exactly in runs (pairs for 16-bit inputs, the whole quad for 8-bit inputs), each product
//   first floored to its run's own unit 2^(E1 - 26) (E1: the run's largest anchor, not limited), then
//   each run sum is floored to U. C is floored to U, or contributes 0 when |C| < U. The integer sum is rounded once:
//   to FP32
//   (nearest even), to FP16 (nearest even), or for BF16 to FP32 (nearest even) and then cut to its top 16 bits.
//   A result of -0 after rounding to FP32 or FP16 gives +0; a nonzero FP32 result cut to a BF16 -0 stays -0.
// Special values: lane 5's special_step, then lane 5's 16-bit NaN conversion (model_acc16).
#pragma once
#include "arith_model.h"

namespace da {

constexpr int NO_CLAMP = -100000;

// in_a, in_b: am::Fmt of the inputs; n products from words a, b; out: am::F32, am::F16 or am::BF16; c: bits of out.
// emin_anchor: the lowest anchor (-126 measured; NO_CLAMP gives lane 5's model). sat16: MODE.FP16_OVFL set; a
// finite FP16 result beyond the range gives the largest finite value (lane 5's rule for the matrix unit).
inline uint32_t dot(am::Fmt in_a, am::Fmt in_b, int n, am::Fmt out, uint32_t a, uint32_t b, uint32_t c,
                    int emin_anchor = -126, bool sat16 = false) {
    const int w = 32 / n;
    const uint32_t mask = (1u << w) - 1;
    am::Dec da[8], db[8];
    int slots[8];
    for (int k = 0; k < n; ++k) {
        da[k] = am::decode(in_a, (a >> (w * k)) & mask);
        db[k] = am::decode(in_b, (b >> (w * k)) & mask);
        slots[k] = k;
    }
    const bool eight_bit = w == 8;
    uint32_t c32 = c;
    am::FmtInfo fo{};
    uint32_t sign16 = 0, inf16 = 0;
    int sh16 = 0;
    if (out != am::F32) {
        fo = am::info(out);
        sh16 = 23 - fo.mbits;
        sign16 = 1u << (fo.ebits + fo.mbits);
        inf16 = ((1u << fo.ebits) - 1) << fo.mbits;
        c &= 0xFFFFu;
        c32 = am::decode(out, c).nan ? ((c & sign16) ? 0x80000000u : 0u) | 0x7F800000u | ((c & ((1u << fo.mbits) - 1)) << sh16)
                                    : am::convert(out, c, am::F32);
    }
    bool handled = false;
    uint32_t r32 = am::special_step(da, db, slots, n, c32, handled, eight_bit);
    if (handled) {
        if (out == am::F32) {
            return r32;
        }
        const am::Dec d = am::decode(am::F32, r32);
        if (d.nan) {
            if (r32 == am::NAN_INPUT) {
                return sign16 | inf16 | (1u << (fo.mbits - 1)) | 0x3Du;
            }
            return ((r32 >> 31) ? sign16 : 0) | inf16 | ((r32 & 0x7FFFFFu) >> sh16);
        }
        return am::convert(am::F32, r32, out);
    }
    const am::Dec dc = am::decode(am::F32, c32);
    // a 16-bit C counts with the 16-bit format's minimum exponent when subnormal there
    const int ce = out == am::F32 ? dc.e : std::max(dc.e, 1 - fo.bias);
    int E = dc.zero ? NO_CLAMP : ce;
    am::Prod pr[8];
    for (int k = 0; k < n; ++k) {
        pr[k] = am::product(da[k], db[k], 2);
        if (!pr[k].zero) {
            E = std::max(E, pr[k].anchor);
        }
    }
    if (E == NO_CLAMP) {
        return 0u;  // every term zero: +0 (lane 5)
    }
    const int ue = std::max(E, emin_anchor) - 26;
    __int128 S = 0;
    if (!dc.zero) {
        const int64_t m = dc.neg ? -dc.m : dc.m;
        const int x = dc.e - dc.mbits;
        if (x >= ue) {
            S += (__int128) m << (x - ue);
        } else if (ce >= ue) {
            S += am::floor_shift(m, ue - x);
        }
    }
    const int sub = eight_bit ? n : 2;
    for (int j = 0; j < n; j += sub) {
        int E1 = NO_CLAMP;
        for (int i = j; i < j + sub; ++i) {
            if (!pr[i].zero) {
                E1 = std::max(E1, pr[i].anchor);
            }
        }
        if (E1 == NO_CLAMP) {
            continue;
        }
        const int ue1 = E1 - 26;  // measured: the run's own unit is not limited by emin_anchor
        am::Exact ex;
        for (int i = j; i < j + sub; ++i) {
            if (pr[i].zero) {
                continue;
            }
            if (pr[i].x < ue1) {
                const int64_t q = am::floor_shift(pr[i].m, ue1 - pr[i].x);
                ex.add(q < 0, (uint64_t) (q < 0 ? -q : q), ue1);
            } else {
                ex.add(pr[i].m < 0, (uint64_t) (pr[i].m < 0 ? -pr[i].m : pr[i].m), pr[i].x);
            }
        }
        S += ex.floor_at(ue);
    }
    if (out == am::F32) {
        const uint32_t r = am::round_int(am::F32, S, ue, false);
        return r == 0x80000000u ? 0u : r;
    }
    if (out == am::F16) {
        const uint32_t r = am::round_int(am::F16, S, ue, false, sat16);
        return r == 0x8000u ? 0u : r;  // -0 gives +0, as for FP32 (lane 5)
    }
    // BF16: nearest even to FP32 (-0 gives +0), then the top 16 bits (toward zero)
    uint32_t r = am::round_int(am::F32, S, ue, false);
    if (r == 0x80000000u) {
        r = 0;
    }
    const am::Dec d32 = am::decode(am::F32, r);
    if (d32.inf) {
        return am::convert(am::F32, r, am::BF16);
    }
    const __int128 m32 = d32.neg ? -(__int128) d32.m : (__int128) d32.m;
    return am::round_int(am::BF16, m32, d32.e - d32.mbits, d32.neg, false, true);
}

}  // namespace da
