// Lane 5: CPU model of what the RDNA 4 (gfx1201) dense matrix instructions compute, bit for bit, and an exact
// reference. Plain C++17, no HIP: used by the CPU programs (arith_fit, arith_eval, arith_stats) and by the GPU
// probes; `arith_check selftest` tests it on the CPU against hand-worked cases.
//
// Model (measured; measurements/matrix-arithmetic.md has the evidence for each rule). For one output element with
// slots s = 0..15 (src/study/arith_gpu.h has the register layout of a slot) and accumulator C:
//   1. The 16 products are taken in 4 steps of 4 slots, in the order {0-3}, {8-11}, {4-7}, {12-15}. Under AMD's
//      guide K order for 16-bit data these are K 0-3, 4-7, 8-11, 12-15 (sequential); under the sample-code order
//      (K contiguous per lane half) they are K {0-3}, {8-11}, {4-7}, {12-15}. For 8-bit data the slot is K.
//      8-bit inputs: 2 steps of 8 slots in slot order.
//   2. Each step adds the running sum (C for the first step) and its products in one fixed-point adder with one
//      unit for the whole step: anchor E = max(exponent of the running sum, ea + eb + P over the step's nonzero
//      products), where ea, eb are the exponent fields of the inputs (subnormals count with the minimum exponent,
//      also a subnormal FP32 running sum) and P is a per-form offset; unit U = 2^(E - W), never below 2^-152.
//      The products are first added in runs (16-bit inputs: the 2 products of one register; 8-bit inputs: 4
//      products), each product first floored to the run's own unit 2^(E1 - W), E1 the run's largest anchor (for a
//      pair that floor never changes the result, so a pair is in effect added exactly); each run sum
//      is then floored to a multiple of U (two's complement floor: a small negative sum becomes -U). The running
//      sum is floored to U too, except that one whose exponent is below U's contributes 0.
//   3. The integer sum is rounded once to FP32 (round to nearest even) and becomes the next running sum.
//   The result is the running sum after the last step.
// Special values (NaN, infinity, zero sign, overflow) are handled per step; see special_step().
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace am {

enum Fmt { F16 = 0, BF16, E4M3, E5M2, F32 };

struct FmtInfo {
    int ebits, mbits, bias;
    bool e4m3;
};
inline FmtInfo info(Fmt f) {
    switch (f) {
    case F16: return {5, 10, 15, false};
    case BF16: return {8, 7, 127, false};
    case E4M3: return {4, 3, 7, true};
    case E5M2: return {5, 2, 15, false};
    default: return {8, 23, 127, false};
    }
}

// A decoded input: value = (neg ? -1 : 1) * m * 2^(e - mbits), with e the exponent field's value (subnormal and
// zero: 1 - bias) and m the significand with its implicit bit.
struct Dec {
    bool neg, zero, inf, nan;
    int e;
    int64_t m;
    int mbits;
};

inline Dec decode(Fmt f, uint32_t bits) {
    const FmtInfo fi = info(f);
    Dec d{};
    d.mbits = fi.mbits;
    d.neg = (bits >> (fi.ebits + fi.mbits)) & 1;
    const uint32_t ef = (bits >> fi.mbits) & ((1u << fi.ebits) - 1), mf = bits & ((1u << fi.mbits) - 1);
    const uint32_t emax = (1u << fi.ebits) - 1;
    if (fi.e4m3 ? (ef == emax && mf == 7) : (ef == emax && mf)) {
        d.nan = true;
        return d;
    }
    if (!fi.e4m3 && ef == emax) {
        d.inf = true;
        return d;
    }
    d.e = ef ? (int) ef - fi.bias : 1 - fi.bias;
    d.m = ef ? (int64_t) (mf | (1u << fi.mbits)) : (int64_t) mf;
    d.zero = d.m == 0;
    return d;
}

inline double value(Fmt f, uint32_t bits) {
    const Dec d = decode(f, bits);
    if (d.nan) {
        return NAN;
    }
    if (d.inf) {
        return d.neg ? -INFINITY : INFINITY;
    }
    const double v = std::ldexp((double) d.m, d.e - d.mbits);
    return d.neg ? -v : v;
}

// ---------------------------------------------------------------------------------------------------------------
// Rounding an integer S * 2^ue to a float format, round to nearest even, with subnormals; overflow gives infinity
// (or the largest finite value when sat). E4M3 has no infinity: overflow gives NaN (0x7F) unless sat.
// neg_zero: the sign to give an exact zero.

inline int bitlen(unsigned __int128 v) {
    int n = 0;
    while (v) {
        v >>= 1;
        ++n;
    }
    return n;
}

// rtz: round toward zero instead (overflow then gives the largest finite value).
inline uint32_t round_int(Fmt f, __int128 S, int ue, bool neg_zero = false, bool sat = false, bool rtz = false) {
    const FmtInfo fi = info(f);
    const uint32_t sign_bit = 1u << (fi.ebits + fi.mbits);
    if (S == 0) {
        return neg_zero ? sign_bit : 0;
    }
    const bool neg = S < 0;
    unsigned __int128 a = neg ? (unsigned __int128) (-S) : (unsigned __int128) S;
    const int n = bitlen(a);
    const int emin = 1 - fi.bias;
    int e = ue + n - 1;                                // exponent of the leading bit
    int lsb = std::max(e - fi.mbits, emin - fi.mbits);  // weight of the last kept bit
    unsigned __int128 q;
    if (lsb <= ue) {
        q = a << (ue - lsb);
    } else {
        const int sh = lsb - ue;
        if (sh >= 128) {
            q = 0;
        } else {
            q = a >> sh;
            const unsigned __int128 rem = a - (q << sh);
            const unsigned __int128 half = (unsigned __int128) 1 << (sh - 1);
            if (!rtz && (rem > half || (rem == half && (q & 1)))) {
                ++q;
            }
        }
    }
    // q is the significand at weight 2^lsb; renormalise after a carry
    if (q >> (fi.mbits + 1)) {
        q >>= 1;
        ++lsb;
    }
    uint32_t out;
    const int ebias_max = fi.e4m3 ? 15 : (1 << fi.ebits) - 2;  // largest finite biased exponent
    if (q == 0) {
        return neg ? sign_bit : 0;
    }
    if (q >> fi.mbits) {
        const int be = lsb + fi.mbits + fi.bias;
        const uint32_t mf = (uint32_t) q & ((1u << fi.mbits) - 1);
        if (be > ebias_max || (fi.e4m3 && be == 15 && mf == 7)) {
            if (fi.e4m3) {
                out = sat ? 0x7E : 0x7F;
            } else {
                out = (sat || rtz) ? (((uint32_t) ebias_max << fi.mbits) | ((1u << fi.mbits) - 1)) : ((uint32_t) (ebias_max + 1) << fi.mbits);
            }
        } else {
            out = ((uint32_t) be << fi.mbits) | mf;
        }
    } else {
        out = (uint32_t) q;  // subnormal
    }
    return out | (neg ? sign_bit : 0);
}

// Round a float (given as bits of format `from`) to format `to`, RNE (sat as in round_int). NaN stays NaN.
inline uint32_t convert(Fmt from, uint32_t bits, Fmt to, bool sat = false) {
    const Dec d = decode(from, bits);
    const FmtInfo ti = info(to);
    const uint32_t sign_bit = 1u << (ti.ebits + ti.mbits);
    if (d.nan) {
        return to == E4M3 ? (d.neg ? 0xFFu : 0x7Fu) : (((1u << ti.ebits) - 1) << ti.mbits) | (1u << (ti.mbits - 1)) | (d.neg ? sign_bit : 0);
    }
    if (d.inf) {
        if (to == E4M3) {
            return (d.neg ? 0x80u : 0u) | (sat ? 0x7Eu : 0x7Fu);
        }
        return (((1u << ti.ebits) - 1) << ti.mbits) | (d.neg ? sign_bit : 0);
    }
    return round_int(to, d.neg ? -(__int128) d.m : (__int128) d.m, d.e - d.mbits, d.neg, sat);
}

// ---------------------------------------------------------------------------------------------------------------
// Exact reference: a sum of terms m * 2^x accumulated in a 1024-bit fixed-point integer (unit 2^-600), then
// rounded once.

struct Exact {
    static constexpr int L = 16, OFF = 600;
    uint64_t pos[L], neg[L];
    bool any_pos_zero, any_term;
    Exact() { clear(); }
    void clear() {
        std::memset(pos, 0, sizeof pos);
        std::memset(neg, 0, sizeof neg);
        any_pos_zero = false;
        any_term = false;
    }
    static void add_mag(uint64_t * acc, uint64_t m, int sh) {
        // m * 2^sh with sh >= 0: m << (sh % 64) fits in 128 bits
        const int w = sh >> 6, b = sh & 63;
        const unsigned __int128 v = (unsigned __int128) m << b;
        const uint64_t parts[2] = {(uint64_t) v, (uint64_t) (v >> 64)};
        unsigned __int128 carry = 0;
        for (int i = 0; w + i < L && (i < 2 || carry); ++i) {
            const unsigned __int128 t = (unsigned __int128) acc[w + i] + (i < 2 ? parts[i] : 0) + carry;
            acc[w + i] = (uint64_t) t;
            carry = t >> 64;
        }
    }
    // Adds (neg ? -1 : 1) * m * 2^x; a zero term records its sign for the sign of an exact-zero result.
    void add(bool negative, uint64_t m, int x) {
        if (m == 0) {
            any_pos_zero |= !negative;
            return;
        }
        any_term = true;
        add_mag(negative ? neg : pos, m, x + OFF);
    }
    // floor(exact value / 2^ue), for results that fit in 120 bits.
    __int128 floor_at(int ue) const {
        uint64_t d[L];
        int cmp = 0;
        for (int i = L - 1; i >= 0 && !cmp; --i) {
            cmp = pos[i] > neg[i] ? 1 : pos[i] < neg[i] ? -1 : 0;
        }
        const uint64_t *big = cmp >= 0 ? pos : neg, *small = cmp >= 0 ? neg : pos;
        uint64_t borrow = 0;
        for (int i = 0; i < L; ++i) {
            const unsigned __int128 t = (unsigned __int128) big[i] - small[i] - borrow;
            d[i] = (uint64_t) t;
            borrow = (t >> 64) ? 1 : 0;
        }
        const int sh = ue + OFF;  // bit index of the unit
        unsigned __int128 q = 0;
        for (int i = L * 64 - 1; i >= sh; --i) {
            if (i < 0) {
                break;
            }
            q = (q << 1) | ((d[i >> 6] >> (i & 63)) & 1);
        }
        bool rest = false;
        for (int i = 0; i < sh && i < L * 64 && !rest; ++i) {
            rest = (d[i >> 6] >> (i & 63)) & 1;
        }
        if (cmp >= 0) {
            return (__int128) q;
        }
        return -(__int128) q - (rest ? 1 : 0);
    }

    // The exact value rounded once to f (round to nearest even).
    uint32_t round(Fmt f, bool sat = false) const {
        uint64_t d[L];
        int cmp = 0;
        for (int i = L - 1; i >= 0 && !cmp; --i) {
            cmp = pos[i] > neg[i] ? 1 : pos[i] < neg[i] ? -1 : 0;
        }
        const uint64_t *big = cmp >= 0 ? pos : neg, *small = cmp >= 0 ? neg : pos;
        uint64_t borrow = 0;
        for (int i = 0; i < L; ++i) {
            const unsigned __int128 s = (unsigned __int128) big[i] - small[i] - borrow;
            d[i] = (uint64_t) s;
            borrow = (s >> 64) ? 1 : 0;
        }
        int p = -1;
        for (int i = L - 1; i >= 0; --i) {
            if (d[i]) {
                p = i * 64 + 63 - __builtin_clzll(d[i]);
                break;
            }
        }
        if (p < 0) {
            // exact zero: -0 only if every term was -0 (round to nearest)
            return round_int(f, 0, 0, !any_pos_zero && !any_term, sat);
        }
        // take the top 100 bits plus a sticky bit, enough for any target format here
        const int lo = std::max(0, p - 100);
        unsigned __int128 v = 0;
        for (int i = p; i >= lo; --i) {
            v = (v << 1) | ((d[i >> 6] >> (i & 63)) & 1);
        }
        bool sticky = false;
        for (int i = 0; i < lo && !sticky; ++i) {
            sticky = (d[i >> 6] >> (i & 63)) & 1;
        }
        v = (v << 1) | (sticky ? 1 : 0);
        __int128 S = (__int128) v;
        return round_int(f, cmp < 0 ? -S : S, lo - 1 - OFF, false, sat);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// The model.

enum Form { F_F16 = 0, F_BF16, F_FF, F_FB, F_BF, F_BB };

inline Fmt fmt_a(int form) {
    static const Fmt t[] = {F16, BF16, E4M3, E4M3, E5M2, E5M2};
    return t[form];
}
inline Fmt fmt_b(int form) {
    static const Fmt t[] = {F16, BF16, E4M3, E5M2, E4M3, E5M2};
    return t[form];
}

struct Params {
    int order[16];      // slots in the order they are added
    int group;          // slots per step
    int prod_off;       // P: anchor contribution of a product is ea + eb + P
    int width;          // W: unit = 2^(E - W)
    bool acc_floor_small;  // false (measured): |running sum| < U contributes 0; true: floored like a product
    bool zero_products_anchor;  // whether a product with a zero factor takes part in the anchor
    bool prod_trunc;    // false (measured): products floored; true: truncated toward zero
    int acc_off;        // anchor contribution of the running sum is its exponent + acc_off
    bool prod_each;     // true: each product floored separately (overrides sub)
    int sub;            // products are summed exactly in runs of `sub` consecutive entries of order[] (measured:
                        // 2, one register of 16-bit data), each run sum floored to the unit
    bool local_floor;   // products floored to the run's own unit 2^(E1 - W) before the run sum (E1: the run's
                        // largest anchor)
    // 16-bit accumulator forms (FP16 into FP16, BF16 into BF16). 0 (the FP32 forms): every step rounds to FP32.
    // 1: as 0, then the FP32 result is rounded to out16 (double rounding). 2: the last step rounds its integer sum
    // directly to out16. 3: every step rounds to out16.
    int round16;
    Fmt out16;
    int rtz16;  // how a step rounds to out16: 0 to nearest even (measured for FP16 into FP16); 1 toward zero from
                // the step's integer sum; 2 to FP32 (nearest even) first, then toward zero, which for BF16 is
                // keeping the high half of the FP32 bits (measured for BF16 into BF16)
    int min_unit;  // the unit never goes below 2^min_unit (measured: -152, three bits below the smallest FP32
                   // value; only BF16 inputs reach it, when a step's largest term is below 2^-126: the terms are
                   // then floored at 2^-152 before the FP32 rounding, so a subnormal result is not always the
                   // nearest value)
    bool neg_zero16;  // true (measured): with rtz16 == 2 a negative step result that truncates to zero is -0
    bool sat16;  // MODE.FP16_OVFL set: a finite step result beyond the 16-bit range gives the largest finite value
};

inline Params measured_params(int form);
inline Params measured_params(int form) {
    Params p{};
    const int ord16[16] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
    for (int i = 0; i < 16; ++i) {
        p.order[i] = ord16[i];
    }
    p.group = 4;
    p.prod_off = 2;
    p.width = 26;
    p.acc_floor_small = false;
    p.zero_products_anchor = false;
    p.prod_trunc = false;
    p.acc_off = 0;
    p.prod_each = false;
    p.sub = 2;
    p.local_floor = true;
    p.round16 = 0;
    p.out16 = F32;
    p.rtz16 = 0;
    p.min_unit = -152;
    p.neg_zero16 = true;
    p.sat16 = false;
    if (form >= F_FF) {
        // 8-bit forms: two steps of 8 slots in K order, products in runs of 4
        for (int i = 0; i < 16; ++i) {
            p.order[i] = i;
        }
        p.group = 8;
        p.sub = 4;
    }
    return p;
}

inline int64_t floor_shift(int64_t v, int sh) {  // floor(v / 2^sh) for sh >= 0
    if (sh >= 63) {
        return v < 0 ? -1 : 0;
    }
    return v >> sh;  // arithmetic shift on this compiler is floor
}

inline int64_t trunc_shift(int64_t v, int sh) {
    if (sh >= 63) {
        return 0;
    }
    return v < 0 ? -((-v) >> sh) : v >> sh;
}

// Product of two decoded inputs as a signed integer m and weight x (value m * 2^x), and its anchor exponent.
struct Prod {
    int64_t m;
    int x, anchor;
    bool zero;
};

inline Prod product(const Dec & a, const Dec & b, int off) {
    Prod p;
    p.m = a.m * b.m;
    if (a.neg != b.neg) {
        p.m = -p.m;
    }
    p.x = a.e - a.mbits + b.e - b.mbits;
    p.anchor = a.e + b.e + off;
    p.zero = p.m == 0;
    return p;
}

// One output of the instruction. a, b: the 16 slot bits; c: FP32 bits. Returns FP32 bits.
inline uint32_t special_step(const struct Dec * da, const struct Dec * db, const int * slots, int n, uint32_t acc,
                             bool & handled, bool eight_bit);

inline uint32_t model_f32(int form, const uint32_t * a, const uint32_t * b, uint32_t c, const Params & P) {
    Dec da[16], db[16];
    for (int s = 0; s < 16; ++s) {
        da[s] = decode(fmt_a(form), a[s]);
        db[s] = decode(fmt_b(form), b[s]);
    }
    uint32_t acc = c;
    for (int g = 0; g < 16; g += P.group) {
        bool handled = false;
        const uint32_t sp = special_step(da, db, P.order + g, P.group, acc, handled, form >= F_FF);
        if (handled) {
            acc = sp;
            continue;
        }
        const Dec dc = decode(F32, acc);
        // a 16-bit running sum: a subnormal counts with the 16-bit format's minimum exponent (measured), for the
        // anchor and for the "below one unit" test; its value is unchanged
        const int ce = P.round16 == 3 ? std::max(dc.e, 1 - info(P.out16).bias) : dc.e;
        bool have = !dc.zero;
        int E = dc.zero ? -100000 : ce + P.acc_off;
        Prod pr[16];
        for (int i = 0; i < P.group; ++i) {
            const int s = P.order[g + i];
            pr[i] = product(da[s], db[s], P.prod_off);
            if (!pr[i].zero || P.zero_products_anchor) {
                E = std::max(E, pr[i].anchor);
                have = true;
            }
        }
        if (!have) {
            acc = 0u;  // measured: an exact zero result is +0 even when every term is -0
            continue;
        }
        const int ue = std::max(E - P.width, P.min_unit);
        __int128 S = 0;
        if (!dc.zero) {
            const int64_t m = dc.neg ? -dc.m : dc.m;
            const int x = dc.e - dc.mbits;
            if (x >= ue) {
                S += (__int128) m << (x - ue);
            } else if (P.acc_floor_small || ce >= ue) {
                S += floor_shift(m, ue - x);
            }
        }
        if (P.prod_each) {
            for (int i = 0; i < P.group; ++i) {
                if (pr[i].zero) {
                    continue;
                }
                if (pr[i].x >= ue) {
                    S += (__int128) pr[i].m << (pr[i].x - ue);
                } else {
                    S += P.prod_trunc ? trunc_shift(pr[i].m, ue - pr[i].x) : floor_shift(pr[i].m, ue - pr[i].x);
                }
            }
        } else {
            for (int j = 0; j < P.group; j += P.sub) {
                // the run is aligned to its own largest anchor first (local unit), then floored to the step unit
                int E1 = -100000;
                for (int i = j; i < j + P.sub; ++i) {
                    if (!pr[i].zero) {
                        E1 = std::max(E1, pr[i].anchor);
                    }
                }
                if (E1 == -100000) {
                    continue;
                }
                const int ue1 = E1 - P.width;
                Exact ex;
                for (int i = j; i < j + P.sub; ++i) {
                    if (pr[i].zero) {
                        continue;
                    }
                    if (P.local_floor && pr[i].x < ue1) {
                        const int64_t q = floor_shift(pr[i].m, ue1 - pr[i].x);
                        ex.add(q < 0, (uint64_t) (q < 0 ? -q : q), ue1);
                    } else {
                        ex.add(pr[i].m < 0, (uint64_t) (pr[i].m < 0 ? -pr[i].m : pr[i].m), pr[i].x);
                    }
                }
                S += ex.floor_at(ue);
            }
        }
        bool keep_neg_zero = false;
        if (P.round16 == 3 || (P.round16 == 2 && g + P.group >= 16)) {
            if (P.rtz16 == 2) {
                const Dec d32 = decode(F32, round_int(F32, S, ue, false));
                const __int128 m32 = d32.inf ? 0 : d32.neg ? -(__int128) d32.m : (__int128) d32.m;
                acc = d32.inf ? round_int(F32, S, ue, false)
                              : convert(P.out16, round_int(P.out16, m32, d32.e - d32.mbits, false, false, true), F32);
                // measured (BF16 into BF16, full-range inputs): keeping the high half of a small negative FP32 value
                // leaves its sign, so a negative step result that truncates to zero is -0, not +0
                if (P.neg_zero16 && !d32.inf && !d32.zero && d32.neg && (acc & 0x7FFFFFFFu) == 0) {
                    acc = 0x80000000u;
                    keep_neg_zero = true;
                }
            } else {
                acc = convert(P.out16, round_int(P.out16, S, ue, false, P.sat16, P.rtz16 == 1), F32);
            }
        } else {
            acc = round_int(F32, S, ue, false);
        }
        if (acc == 0x80000000u && !keep_neg_zero) {
            acc = 0u;  // measured: a negative result that rounds to zero gives +0
        }
    }
    return acc;
}

// The exact sum of the 16 products and C, rounded once to `to`.
inline uint32_t exact_round(int form, const uint32_t * a, const uint32_t * b, uint32_t c, Fmt to, bool sat = false) {
    Exact ex;
    for (int s = 0; s < 16; ++s) {
        const Dec da = decode(fmt_a(form), a[s]), db = decode(fmt_b(form), b[s]);
        ex.add(da.neg != db.neg, (uint64_t) (da.m * db.m), da.e - da.mbits + db.e - db.mbits);
    }
    const Dec dc = decode(F32, c);
    ex.add(dc.neg, (uint64_t) dc.m, dc.e - dc.mbits);
    return ex.round(to, sat);
}

// Non-finite values, one step at a time (measured, matrix-arithmetic.md section 4). Within a step, in this order:
// a NaN input in A or B (any payload or sign, both FP8 NaN codes) gives 0xFFC00A3D; inf x 0 in a product, or +inf
// and -inf among the products, gives 0xFFC00000; a NaN running sum (C, or an earlier step's NaN) passes through
// quieted with payload and sign; a running-sum infinity against a product infinity of the other sign gives
// 0xFFC00000; otherwise an infinity gives that infinity. In the 8-bit forms every NaN made inside the instruction
// is 0xFFC00A3D (0xFFC00000 never appears there). A later step's
// NaN replaces an earlier one. Finite overflow at a step gives infinity, which then stays.
constexpr uint32_t NAN_INPUT = 0xFFC00A3Du, NAN_INVALID = 0xFFC00000u;

inline uint32_t special_step(const Dec * da, const Dec * db, const int * slots, int n, uint32_t acc, bool & handled,
                             bool eight_bit) {
    bool nan_in = false, invalid = false, pinf = false, ninf = false;
    for (int i = 0; i < n; ++i) {
        const Dec & x = da[slots[i]];
        const Dec & y = db[slots[i]];
        if (x.nan || y.nan) {
            nan_in = true;
        } else if ((x.inf && y.zero) || (y.inf && x.zero)) {
            invalid = true;
        } else if (x.inf || y.inf) {
            (x.neg != y.neg ? ninf : pinf) = true;
        }
    }
    const Dec dc = decode(F32, acc);
    handled = nan_in || invalid || dc.nan || dc.inf || pinf || ninf;
    const uint32_t invalid_nan = eight_bit ? NAN_INPUT : NAN_INVALID;
    if (nan_in) {
        return NAN_INPUT;
    }
    if (invalid || (pinf && ninf)) {
        return invalid_nan;
    }
    if (dc.nan) {
        return acc | 0x00400000u;
    }
    if (dc.inf) {
        if ((dc.neg && pinf) || (!dc.neg && ninf)) {
            return invalid_nan;
        }
        return acc;
    }
    return pinf ? 0x7F800000u : 0xFF800000u;
}

// ---------------------------------------------------------------------------------------------------------------
// 16-bit accumulator forms: c and the result are bits of P.out16 (FP16 or BF16); the inputs are those of `form`
// (F_F16 or F_BF16). C is widened exactly and runs through the FP32-form steps; P.round16 says where the result
// is rounded to 16 bits.
// NaNs: a NaN C keeps its sign and payload (widened by shifting the significand up, narrowed back); a NaN input in
// A or B gives the quiet NaN with low bits 0x3D and the sign set (0xFE3D, 0xFFFD), like 0xFFC00A3D in the FP32 forms.
inline uint32_t model_acc16(int form, const uint32_t * a, const uint32_t * b, uint32_t c, const Params & P) {
    const FmtInfo fo = info(P.out16);
    const int sh = 23 - fo.mbits;
    const uint32_t sign16 = 1u << (fo.ebits + fo.mbits), inf16 = ((1u << fo.ebits) - 1) << fo.mbits;
    const uint32_t c32 = decode(P.out16, c).nan
                             ? ((c & sign16) ? 0x80000000u : 0u) | 0x7F800000u | ((c & ((1u << fo.mbits) - 1)) << sh)
                             : convert(P.out16, c, F32);
    const uint32_t r = model_f32(form, a, b, c32, P);
    const Dec d = decode(F32, r);
    if (d.nan) {
        if (r == NAN_INPUT) {
            return sign16 | inf16 | (1u << (fo.mbits - 1)) | 0x3Du;
        }
        return ((r >> 31) ? sign16 : 0) | inf16 | ((r & 0x7FFFFFu) >> sh);
    }
    if (d.inf) {
        return convert(F32, r, P.out16);
    }
    return round_int(P.out16, d.neg ? -(__int128) d.m : (__int128) d.m, d.e - d.mbits, d.neg, false, P.rtz16);
}

inline Params acc16_params(int form, int round16) {
    Params p = measured_params(form);
    p.round16 = round16;
    p.out16 = form == F_BF16 ? BF16 : F16;
    return p;
}

// ---------------------------------------------------------------------------------------------------------------
// Sparse forms (SWMMAC, measured by one-hot experiments, matrix-arithmetic.md section 11). Stored element j
// (0..15) of an A row multiplies B slot 4 g + i, where g = ((j & 7) >> 1) + 4 (j >> 3) is its group of four and i
// is the 2-bit field (j & 7) + 8 * index_key of the index register in the lane that holds j (row r: lane r for
// j < 8, lane r + 16 for j >= 8). Every field value is used as given, including two stored elements naming the
// same slot. The 16 selected products then go through the dense model as dense slots 0..15.
inline int sparse_slot(int j, uint32_t idx_lane, int key) {
    const int g = ((j & 7) >> 1) + 4 * (j >> 3);
    return 4 * g + (int) ((idx_lane >> (2 * ((j & 7) + 8 * key))) & 3);
}

// b32: the column's 32 B slots; idx_lo, idx_hi: the index registers of lanes r and r + 16. b_sel: 16 slots.
inline void sparse_select(const uint32_t * b32, uint32_t idx_lo, uint32_t idx_hi, int key, uint32_t * b_sel) {
    for (int j = 0; j < 16; ++j) {
        b_sel[j] = b32[sparse_slot(j, j < 8 ? idx_lo : idx_hi, key)];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Integer forms (IU8, IU4; matrix-arithmetic.md section 12). Slot values are w-bit two's complement when signed,
// else unsigned. The sum C + sum of the k products is exact (64-bit); clamp_mode says what happens beyond 32 bits:
// 0 wrap around (no clamp bit); 1 the exact final sum saturated to [INT32_MIN, INT32_MAX]; 2 saturated after every
// step of `group` slots; 3 saturated to [0, UINT32_MAX] when both inputs are unsigned, else as 1.
inline int64_t int_val(uint32_t bits, int w, bool sgn) {
    const uint32_t m = (1u << w) - 1, v = bits & m;
    return sgn && (v >> (w - 1)) ? (int64_t) v - (int64_t) (m + 1) : (int64_t) v;
}

inline int64_t sat32(int64_t v, bool uns) {
    if (uns) {
        return std::min<int64_t>(std::max<int64_t>(v, 0), 0xFFFFFFFFll);
    }
    return std::min<int64_t>(std::max<int64_t>(v, INT32_MIN), INT32_MAX);
}

inline uint32_t model_int(const uint32_t * a, const uint32_t * b, int k, int w, bool sa, bool sb, uint32_t c, int clamp_mode,
                          int group = 8) {
    int64_t acc = (int64_t) (int32_t) c;
    if (clamp_mode == 3 && !sa && !sb) {
        acc = (int64_t) c;
    }
    for (int g = 0; g < k; g += group) {
        for (int s = g; s < g + group && s < k; ++s) {
            acc += int_val(a[s], w, sa) * int_val(b[s], w, sb);
        }
        if (clamp_mode == 2) {
            acc = sat32(acc, false);
        }
    }
    if (clamp_mode == 1) {
        acc = sat32(acc, false);
    } else if (clamp_mode == 3) {
        acc = sat32(acc, !sa && !sb);
    }
    return (uint32_t) acc;
}

}  // namespace am
