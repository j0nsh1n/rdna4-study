// Lane 6, item 5: CPU models of the vector dot-product instructions (V_DOT4_F32_{FP8,BF8}_{FP8,BF8},
// V_DOT2_F32_F16, V_DOT2_F32_BF16, V_DOT2_F16_F16, V_DOT2_BF16_BF16). Host code only.
//
// Every input is decoded to an exact term (-1)^neg * m * 2^e; products are exact (m up to 22 bits). Sums are formed
// exactly in a 640-bit two's-complement fixed-point number (lowest bit 2^-320, enough for BF16 products down to
// 2^-266 and up to 2^256) and rounded with one of the four IEEE directions, with subnormals and overflow.
// Candidate rules (cvt_dot probes the hardware against each):
//   EXACT     C + sum of products, rounded once to the output format;
//   SEQ       the guide's text: acc = C, then acc = round(acc + p_i) for i = 0, 1, ... (each step a fused
//             multiply-add rounded to the step format), finally rounded to the output format;
//   SEQ_REV   the same with the products in reverse order;
//   PROD_C    the products summed exactly and rounded, then C added and rounded again.
// Special values follow IEEE: a NaN input or 0 x infinity gives NaN, +inf + -inf gives NaN, an exact zero sum is
// -0 only when every term is -0 (round to nearest, up, toward zero) or when the rounding is downward and the terms
// cancel.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dm {

enum Fmt { FP8 = 0, BF8, F16, BF16, F32 };
enum Rule { EXACT = 0, SEQ, SEQ_REV, PROD_C, NRULE };
static const char * const rule_name[NRULE] = {"exact,one rounding", "sequential (guide)", "sequential reversed",
                                              "products then C"};

struct FI {
    int eb, mb, bias;
};
inline FI fi(Fmt f) {
    switch (f) {
    case FP8: return {4, 3, 7};
    case BF8: return {5, 2, 15};
    case F16: return {5, 10, 15};
    case BF16: return {8, 7, 127};
    default: return {8, 23, 127};
    }
}
inline int width(Fmt f) { return f == FP8 || f == BF8 ? 8 : (f == F32 ? 32 : 16); }

struct Term {
    bool nan = false, inf = false, neg = false;
    uint64_t m = 0;  // 0 means zero
    int e = 0;
};

inline Term dec(Fmt f, uint32_t bits) {
    const FI i = fi(f);
    Term t;
    t.neg = (bits >> (i.eb + i.mb)) & 1;
    const uint32_t ef = (bits >> i.mb) & ((1u << i.eb) - 1), mf = bits & ((1u << i.mb) - 1);
    const uint32_t emax = (1u << i.eb) - 1;
    if (f == FP8) {
        if (ef == 15 && mf == 7) {
            t.nan = true;
            return t;
        }
    } else if (ef == emax) {
        (mf ? t.nan : t.inf) = true;
        return t;
    }
    t.m = ef ? (mf | (1u << i.mb)) : mf;
    t.e = (ef ? (int) ef : 1) - i.bias - i.mb;
    return t;
}

inline Term mul(const Term & a, const Term & b) {
    Term t;
    t.neg = a.neg != b.neg;
    if (a.nan || b.nan || (a.inf && b.m == 0 && !b.inf) || (b.inf && a.m == 0 && !a.inf)) {
        t.nan = true;
        return t;
    }
    if (a.inf || b.inf) {
        t.inf = true;
        return t;
    }
    t.m = a.m * b.m;
    t.e = a.e + b.e;
    return t;
}

constexpr int NL = 10, LSB = -320;

struct Big {
    uint64_t w[NL] = {};
};

inline void big_add(Big & s, const Term & t) {
    if (t.m == 0) {
        return;
    }
    const int pos = t.e - LSB;
    uint64_t add[NL] = {};
    const int k = pos / 64, off = pos % 64;
    add[k] = t.m << off;
    if (off && k + 1 < NL) {
        add[k + 1] = t.m >> (64 - off);
    }
    if (!t.neg) {
        unsigned __int128 c = 0;
        for (int i = 0; i < NL; ++i) {
            c += (unsigned __int128) s.w[i] + add[i];
            s.w[i] = (uint64_t) c;
            c >>= 64;
        }
    } else {
        uint64_t borrow = 0;
        for (int i = 0; i < NL; ++i) {
            const unsigned __int128 sub = (unsigned __int128) add[i] + borrow;
            borrow = (unsigned __int128) s.w[i] < sub;
            s.w[i] = (uint64_t) ((unsigned __int128) s.w[i] - sub);
        }
    }
}

inline bool big_bit(const Big & s, int i) { return (s.w[i / 64] >> (i % 64)) & 1; }

// Rounding directions as in the MODE register: 0 nearest even, 1 up, 2 down, 3 toward zero.
// zero_neg: the sign to give an exact zero.
inline uint32_t big_round(Big s, Fmt out, int rnd, bool zero_neg) {
    const FI i = fi(out);
    const uint32_t sign_bit = 1u << (i.eb + i.mb);
    const bool neg = s.w[NL - 1] >> 63;
    if (neg) {
        uint64_t c = 1;
        for (int k = 0; k < NL; ++k) {
            s.w[k] = ~s.w[k] + c;
            c = c && s.w[k] == 0;
        }
    }
    int t = -1;
    for (int k = NL - 1; k >= 0 && t < 0; --k) {
        if (s.w[k]) {
            t = k * 64 + 63 - __builtin_clzll(s.w[k]);
        }
    }
    if (t < 0) {
        return zero_neg ? sign_bit : 0;
    }
    const int emin = 1 - i.bias;
    const int L = std::max(t + LSB - i.mb, emin - i.mb);  // exponent of the last kept bit
    const int lp = L - LSB;
    uint64_t q = 0;
    for (int b = t; b >= lp && b >= 0; --b) {
        q = (q << 1) | (uint64_t) big_bit(s, b);
    }
    if (lp > t) {
        q = 0;
    }
    bool half = false, sticky = false;
    if (lp > 0) {
        half = lp - 1 <= t ? big_bit(s, lp - 1) : false;
        for (int b = std::min(lp - 2, t); b >= 0 && !sticky; --b) {
            if (b % 64 == 63 && s.w[b / 64] == 0) {
                b -= 63;
                continue;
            }
            sticky = big_bit(s, b);
        }
    }
    bool up = false;
    if (half || sticky) {
        if (rnd == 0) {
            up = half && (sticky || (q & 1));
        } else if (rnd == 1) {
            up = !neg;
        } else if (rnd == 2) {
            up = neg;
        }
    }
    int Lq = L;
    if (up) {
        ++q;
        if (q >> (i.mb + 1)) {
            q >>= 1;
            ++Lq;
        }
    }
    const uint32_t sgn = neg ? sign_bit : 0;
    if (q == 0) {
        return sgn;
    }
    uint32_t efield = 0;
    if (q >> i.mb) {
        efield = (uint32_t) (Lq + i.mb + i.bias);
    }
    const uint32_t emaxf = (1u << i.eb) - 1;
    if (efield >= emaxf) {
        const bool to_inf = rnd == 0 || (rnd == 1 && !neg) || (rnd == 2 && neg);
        return sgn | (to_inf ? (emaxf << i.mb) : (((emaxf - 1) << i.mb) | ((1u << i.mb) - 1)));
    }
    return sgn | (efield << i.mb) | ((uint32_t) q & ((1u << i.mb) - 1));
}

inline uint32_t nan_bits(Fmt out) {
    const FI i = fi(out);
    return (((1u << i.eb) - 1) << i.mb) | (1u << (i.mb - 1));
}

// Sum of terms with IEEE special handling; returns the rounded bits.
inline uint32_t sum_round(const std::vector<Term> & ts, Fmt out, int rnd) {
    bool nan = false, pinf = false, ninf = false, all_negzero = true;
    Big s;
    for (const Term & t : ts) {
        nan |= t.nan;
        if (t.inf) {
            (t.neg ? ninf : pinf) = true;
        }
        if (t.nan || t.inf || t.m != 0 || !t.neg) {
            all_negzero = false;
        }
        if (!t.nan && !t.inf) {
            big_add(s, t);
        }
    }
    const FI i = fi(out);
    if (nan || (pinf && ninf)) {
        return nan_bits(out);
    }
    if (pinf || ninf) {
        return (ninf ? 1u << (i.eb + i.mb) : 0u) | (((1u << i.eb) - 1) << i.mb);
    }
    // an exact zero from cancelling nonzero terms is -0 only when rounding down
    bool any_nonzero = false;
    for (const Term & t : ts) {
        any_nonzero |= t.m != 0;
    }
    return big_round(s, out, rnd, any_nonzero ? rnd == 2 : all_negzero);
}

// Evaluate one rule. a, b: the packed source words; c: the accumulator bits (FP32, or FP16/BF16 in the low half).
// in: element format; n: elements per word; out: output format.
inline uint32_t eval(Rule r, Fmt in, int n, Fmt out, uint32_t a, uint32_t b, uint32_t c, int rnd = 0,
                     Fmt in_b = (Fmt) -1) {
    if (in_b == (Fmt) -1) {
        in_b = in;
    }
    const int w = width(in);
    std::vector<Term> p;
    for (int k = 0; k < n; ++k) {
        const uint32_t mask = (1u << w) - 1;
        p.push_back(mul(dec(in, (a >> (w * k)) & mask), dec(in_b, (b >> (w * k)) & mask)));
    }
    const Term ct = dec(out, out == F32 ? c : c & 0xFFFFu);
    if (r == EXACT) {
        std::vector<Term> ts = p;
        ts.insert(ts.begin(), ct);
        return sum_round(ts, out, rnd);
    }
    if (r == PROD_C) {
        const uint32_t ps = sum_round(p, out, rnd);
        return sum_round({dec(out, ps), ct}, out, rnd);
    }
    // sequential: each step rounded to FP32 when the output is FP32, else to the output format
    uint32_t acc = out == F32 ? c : c & 0xFFFFu;
    for (int k = 0; k < n; ++k) {
        const Term & pk = p[r == SEQ ? k : n - 1 - k];
        acc = sum_round({dec(out, acc), pk}, out, rnd);
    }
    return acc;
}

}  // namespace dm
