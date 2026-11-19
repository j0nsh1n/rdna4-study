// Lane 6: integer-only references for the 8-bit float formats (FP8 = OCP E4M3, BF8 = OCP E5M2), usable on the host
// and on the GPU, so the GPU probes can judge every input in place. src/study/cvt_refcheck.cpp tests them on the CPU
// against definitions that share no code with this file (llama.cpp's E4M3 decoder, the CPU's FP16 conversion for
// E5M2, a nearest-value search over the sorted code values for encoding).
//
// FP8 (E4M3): bias 7, no infinity, 0x7F and 0xFF are NaN, largest finite 448 (0x7E).
// BF8 (E5M2): bias 15, 0x7C is infinity, 0x7D-0x7F NaN, largest finite 57344 (0x7B).
#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define CR_FN __host__ __device__ inline
#else
#define CR_FN inline
#endif

namespace cr {

enum Fmt8 { FP8 = 0, BF8 = 1 };
// MODE register rounding encoding (FP_ROUND field): 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero.
enum Rnd { RNE = 0, RUP = 1, RDN = 2, RTZ = 3 };

struct F8 {
    int M, emin, bias;
    uint32_t maxcode;
};
CR_FN F8 f8info(int f) { return f == FP8 ? F8{3, -6, 7, 0x7Eu} : F8{2, -14, 15, 0x7Bu}; }

CR_FN bool is_nan8(int f, uint32_t c) {
    c &= 0x7Fu;
    return f == FP8 ? c == 0x7Fu : c > 0x7Cu;
}

CR_FN int top_bit(uint32_t v) {
    int t = 0;
    while (v >>= 1) {
        ++t;
    }
    return t;
}

// 8-bit code to FP32 bits. NaN gives sign | 0x7FC00000 (the probes compare NaN by class, not by payload).
CR_FN uint32_t decode8(int f, uint32_t c) {
    const F8 i = f8info(f);
    const uint32_t s = (c & 0x80u) << 24, a = c & 0x7Fu;
    if (is_nan8(f, a)) {
        return s | 0x7FC00000u;
    }
    if (f == BF8 && a == 0x7Cu) {
        return s | 0x7F800000u;
    }
    const uint32_t e = a >> i.M, m = a & ((1u << i.M) - 1);
    if (e == 0) {
        if (m == 0) {
            return s;
        }
        const int t = top_bit(m);
        return s | ((uint32_t) (i.emin - i.M + t + 127) << 23) | ((m << (23 - t)) & 0x7FFFFFu);
    }
    return s | ((uint32_t) ((int) e - i.bias + 127) << 23) | (m << (23 - i.M));
}

// How a magnitude is rounded: nearest even, up (away from zero), down (toward zero); TIES_AWAY exists only so the
// self-test and the probes' --break can show that a wrong tie rule is caught.
enum Mag { M_NE = 0, M_UP = 1, M_DN = 2, M_TIES_AWAY = 3 };

CR_FN Mag mag_mode(int rnd, bool neg) {
    if (rnd == RNE) {
        return M_NE;
    }
    if (rnd == RTZ) {
        return M_DN;
    }
    return ((rnd == RUP) != neg) ? M_UP : M_DN;
}

CR_FN uint32_t rshift_round(uint32_t v, int sh, Mag how) {
    if (sh <= 0) {
        return v;
    }
    const uint64_t vv = v;
    const uint64_t q = sh >= 40 ? 0 : vv >> sh;
    const uint64_t rem = sh >= 40 ? vv : vv - (q << sh);
    if (rem == 0) {
        return (uint32_t) q;
    }
    if (how == M_UP) {
        return (uint32_t) q + 1;
    }
    if (how == M_DN) {
        return (uint32_t) q;
    }
    if (sh > 40) {
        return (uint32_t) q;  // remainder far below half
    }
    const uint64_t half = 1ull << (sh - 1);
    if (rem > half || (rem == half && (how == M_TIES_AWAY || (q & 1)))) {
        return (uint32_t) q + 1;
    }
    return (uint32_t) q;
}

// FP32 bits to an 8-bit code. sat: out-of-range finite values and infinity give the largest finite code. Without
// sat, FP8 gives NaN (sign | 0x7F) and BF8 gives infinity, except that a directed rounding toward zero gives the
// largest finite code as IEEE does. NaN gives sign | 0x7F (FP8) or sign | 0x7E (BF8). sat_inf = false is the
// hardware's rule (MEASURED): saturation applies to finite inputs only, infinity still gives NaN (FP8) or infinity.
CR_FN uint32_t encode8(int f, uint32_t bits, bool sat, int rnd = RNE, bool ties_away = false, bool sat_inf = true) {
    const F8 i = f8info(f);
    const bool neg = bits >> 31;
    const uint32_t s = neg ? 0x80u : 0u, a = bits & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return s | (f == FP8 ? 0x7Fu : 0x7Eu);
    }
    const Mag how = ties_away ? M_TIES_AWAY : mag_mode(rnd, neg);
    const uint32_t ovf = (sat || how == M_DN) ? i.maxcode : (f == FP8 ? 0x7Fu : 0x7Cu);
    if (a == 0x7F800000u) {
        return s | ((sat && sat_inf) ? i.maxcode : (f == FP8 ? 0x7Fu : 0x7Cu));
    }
    if (a == 0) {
        return s;
    }
    const int ef = (int) (a >> 23);
    const int exp = ef ? ef - 127 : -126;
    const uint32_t m24 = (a & 0x7FFFFFu) | (ef ? 0x800000u : 0u);
    uint32_t code;
    if (exp < i.emin) {
        code = rshift_round(m24, 23 + i.emin - i.M - exp, how);  // units of the smallest subnormal step
    } else {
        // q in [2^M, 2^(M+1)]; adding it to (e - 1) << M carries into the exponent by itself
        code = ((uint32_t) (exp + i.bias - 1) << i.M) + rshift_round(m24, 23 - i.M, how);
    }
    return s | (code > i.maxcode ? ovf : code);
}

// Input classes for counting: what kind of rounding the input needs.
enum Cls { C_NAN, C_INF, C_OVER, C_MID, C_TOMAX, C_TIE, C_TIESUB, C_SUB, C_FSUB, C_ZERO, C_OTHER, NCLS };
static const char * const cls_name[NCLS] = {"nan",     "inf",       "above_mid", "at_mid", "max_to_mid", "tie",
                                            "tie_sub", "subnormal", "f32_sub",   "zero",   "other"};

// max_to_mid: above the largest finite value but below the midpoint to the next (absent) code; at_mid: exactly that
// midpoint (448..464 for FP8, 57344..61440 for BF8); above_mid: beyond it. tie / tie_sub: exactly half a step in
// the normal / subnormal target range. subnormal: other inputs below the smallest normal target value.
CR_FN int class8(int f, uint32_t bits) {
    const F8 i = f8info(f);
    const uint32_t a = bits & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return C_NAN;
    }
    if (a == 0x7F800000u) {
        return C_INF;
    }
    if (a == 0) {
        return C_ZERO;
    }
    if (a < 0x00800000u) {
        return C_FSUB;
    }
    const uint32_t maxbits = decode8(f, i.maxcode);
    const uint32_t mid = maxbits + (1u << (22 - i.M));
    if (a > mid) {
        return C_OVER;
    }
    if (a == mid) {
        return C_MID;
    }
    if (a > maxbits) {
        return C_TOMAX;
    }
    const int exp = (int) (a >> 23) - 127;
    const uint32_t m24 = (a & 0x7FFFFFu) | 0x800000u;
    if (exp < i.emin) {
        const int sh = 23 + i.emin - i.M - exp;
        if (sh <= 24 && (m24 & ((1u << sh) - 1)) == (1u << (sh - 1))) {
            return C_TIESUB;
        }
        return C_SUB;
    }
    const uint32_t low = a & ((1u << (23 - i.M)) - 1);
    return low == (1u << (22 - i.M)) ? C_TIE : C_OTHER;
}

// Stochastic rounding candidates (V_CVT_SR_*). r is the second operand; the guide's pseudo-code adds r's top
// 23 - M bits (r >> 12 for FP8, r >> 11 for BF8) to the FP32 mantissa, "wrapping" on overflow, then converts.
//   SR_GUIDE: literal guide text: mantissa + bits mod 2^23 (no carry into the exponent), then round to nearest even.
//   SR_CARRY_TRUNC: add the bits to the whole magnitude (a carry moves into the exponent), then truncate.
//   SR_CARRY_RNE: the same add, then round to nearest even.
//   SR_WRAP_TRUNC: the guide's wrapping add, then truncate.
//   SR_ALIGNED: the operand's top 23 - M bits are added just below the last kept bit of the target (in the
//   subnormal target range too, where more than 23 - M bits are dropped; the bits below the random ones are cut),
//   then the result is truncated; a result above the largest finite code overflows like round to nearest (NaN for
//   FP8, infinity for BF8, the largest finite code when sat). In the normal range it equals SR_CARRY_TRUNC.
enum SrVar { SR_GUIDE = 0, SR_CARRY_TRUNC, SR_CARRY_RNE, SR_WRAP_TRUNC, SR_ALIGNED, NSRVAR };
static const char * const sr_name[NSRVAR] = {"guide(wrap,rne)", "carry,trunc", "carry,rne", "wrap,trunc",
                                             "aligned,trunc,ovf"};

CR_FN uint32_t sr_add(int f, uint32_t bits, uint32_t r, bool carry) {
    const uint32_t add = r >> (f == FP8 ? 12 : 11);
    if (carry) {
        const uint32_t a = bits & 0x7FFFFFFFu;
        return (bits & 0x80000000u) | (a + add);
    }
    return (bits & 0xFF800000u) | (((bits & 0x7FFFFFu) + add) & 0x7FFFFFu);
}

// Number of dropped bits (from the 24-bit significand) for a finite nonzero magnitude a, and its significand.
CR_FN int dropped_bits(int f, uint32_t a, uint32_t & m24, int & exp) {
    const F8 i = f8info(f);
    const int ef = (int) (a >> 23);
    exp = ef ? ef - 127 : -126;
    m24 = (a & 0x7FFFFFu) | (ef ? 0x800000u : 0u);
    return exp < i.emin ? 23 + i.emin - i.M - exp : 23 - i.M;
}

CR_FN uint32_t sr_aligned(int f, uint32_t bits, uint32_t r, bool sat) {
    const F8 i = f8info(f);
    const uint32_t a = bits & 0x7FFFFFFFu, s = (bits >> 31) ? 0x80u : 0u;
    if (a >= 0x7F800000u) {
        return encode8(f, bits, sat, RNE, false, false);
    }
    if (a == 0) {
        return s;
    }
    uint32_t m24;
    int exp;
    const int sh = dropped_bits(f, a, m24, exp), rb = 23 - i.M;
    const uint64_t R = r >> (32 - rb);
    uint64_t q;
    if (sh - rb >= 24) {
        q = 0;  // m24 is below one unit of the random bits, so the sum stays below one step
    } else {
        q = ((uint64_t) m24 + (R << (sh - rb))) >> sh;
    }
    const uint32_t code = exp < i.emin ? (uint32_t) q : ((uint32_t) (exp + i.bias - 1) << i.M) + (uint32_t) q;
    return s | (code > i.maxcode ? (sat ? i.maxcode : (f == FP8 ? 0x7Fu : 0x7Cu)) : code);
}

CR_FN uint32_t sr8(int f, uint32_t bits, uint32_t r, int var, bool sat) {
    if (var == SR_ALIGNED) {
        return sr_aligned(f, bits, r, sat);
    }
    const bool carry = var != SR_GUIDE && var != SR_WRAP_TRUNC;
    const bool trunc = var != SR_GUIDE && var != SR_CARRY_RNE;
    const uint32_t a = bits & 0x7FFFFFFFu;
    if (a >= 0x7F800000u && carry) {
        return encode8(f, bits, sat);
    }
    return encode8(f, sr_add(f, bits, r, carry), sat, trunc ? RTZ : RNE);
}

}  // namespace cr
