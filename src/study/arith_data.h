// Lane 5: input generators for the matrix-arithmetic probes (CPU only). One problem = A (16 rows x 16 slots),
// B (16 columns x 16 slots), C (16 x 16 FP32 bits). Element (r, c) uses A row r, B column c and C[r][c].
#pragma once
#include "arith_model.h"

#include <random>
#include <vector>

namespace ad {

struct Problem {
    uint32_t a[16][16], b[16][16], c[16][16];  // a[row][slot], b[col][slot]
};

enum Dist { WIDE = 0, NARROW, CANCEL, BIGSMALL, ALLCODES, NDIST };
static const char * const dist_name[NDIST] = {"wide", "narrow", "cancel", "bigsmall", "allcodes"};

// Exponent range (unbiased, normal numbers) used for "wide" inputs per format, chosen so that FP32 does not
// overflow: products and C stay below 2^126.
inline void wide_range(am::Fmt f, int & lo, int & hi) {
    switch (f) {
    case am::F16: lo = -14; hi = 15; break;
    case am::BF16: lo = -60; hi = 60; break;
    case am::E4M3: lo = -6; hi = 8; break;
    case am::E5M2: lo = -14; hi = 15; break;
    default: lo = -20; hi = 20;
    }
}

// Random finite bits of format f with exponent in [lo, hi]; a share of zeros and subnormals.
inline uint32_t rand_val(am::Fmt f, std::mt19937_64 & rng, int lo, int hi, int zero_per_64 = 2, int sub_per_64 = 3) {
    const am::FmtInfo fi = am::info(f);
    const uint32_t sign = (uint32_t) (rng() & 1) << (fi.ebits + fi.mbits);
    const int pick = (int) (rng() & 63);
    const uint32_t mant = (uint32_t) (rng() & ((1u << fi.mbits) - 1));
    if (pick < zero_per_64) {
        return sign;
    }
    if (pick < zero_per_64 + sub_per_64 && lo <= 1 - fi.bias) {
        return sign | (mant ? mant : 1);
    }
    for (;;) {
        const int e = lo + (int) (rng() % (uint64_t) (hi - lo + 1));
        const int be = e + fi.bias;
        if (be < 1 || be > (fi.e4m3 ? 15 : (1 << fi.ebits) - 2)) {
            continue;
        }
        if (fi.e4m3 && be == 15 && mant == 7) {
            continue;
        }
        return sign | ((uint32_t) be << fi.mbits) | mant;
    }
}

inline uint32_t rand_f32(std::mt19937_64 & rng, int lo, int hi) {
    const int e = lo + (int) (rng() % (uint64_t) (hi - lo + 1));
    return (uint32_t) ((rng() & 1) << 31) | ((uint32_t) (e + 127) << 23) | (uint32_t) (rng() & 0x7FFFFF);
}

// Any finite code of an 8-bit format (FP8 forms: "allcodes").
inline uint32_t rand_code8(am::Fmt f, std::mt19937_64 & rng) {
    for (;;) {
        const uint32_t v = (uint32_t) (rng() & 0xFF);
        const am::Dec d = am::decode(f, v);
        if (!d.nan && !d.inf) {
            return v;
        }
    }
}

// Exponent of the largest product a format pair can produce (for C ranges).
inline void prod_range(int form, int & lo, int & hi) {
    int alo, ahi, blo, bhi;
    wide_range(am::fmt_a(form), alo, ahi);
    wide_range(am::fmt_b(form), blo, bhi);
    lo = alo + blo - 6;
    hi = ahi + bhi + 2;
}

inline Problem make(int form, Dist dist, std::mt19937_64 & rng) {
    Problem p{};
    const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
    int alo, ahi, blo, bhi, plo, phi;
    wide_range(fa, alo, ahi);
    wide_range(fb, blo, bhi);
    prod_range(form, plo, phi);
    if (dist == NARROW) {
        alo = blo = -4;
        ahi = bhi = 4;
        plo = -10;
        phi = 10;
    }
    for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 16; ++s) {
            if (dist == ALLCODES && am::info(fa).mbits <= 3) {
                p.a[r][s] = rand_code8(fa, rng);
                p.b[r][s] = rand_code8(fb, rng);
            } else if (dist == BIGSMALL) {
                // per slot either near the top or far down, so gaps of every size occur inside one step
                const bool big = rng() & 1;
                const int mid = (alo + ahi) / 2;
                p.a[r][s] = big ? rand_val(fa, rng, mid, ahi, 0, 0) : rand_val(fa, rng, alo, mid, 4, 6);
                p.b[r][s] = rand_val(fb, rng, (blo + bhi) / 2 - 2, (blo + bhi) / 2 + 2, 0, 0);
            } else {
                p.a[r][s] = rand_val(fa, rng, alo, ahi);
                p.b[r][s] = rand_val(fb, rng, blo, bhi);
            }
        }
    }
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            const int pick = (int) (rng() & 15);
            if (pick == 0) {
                p.c[r][c] = (rng() & 1) ? 0x80000000u : 0u;
            } else if (dist == CANCEL || (pick == 1 && dist != NARROW)) {
                // C = minus the exactly rounded product sum, so the result is the small remainder
                uint32_t zero = 0;
                const uint32_t s = am::exact_round(form, p.a[r], p.b[c], zero, am::F32);
                p.c[r][c] = s ^ 0x80000000u;
                if ((rng() & 3) == 0) {
                    p.c[r][c] ^= 1;  // off by one unit
                }
            } else {
                p.c[r][c] = rand_f32(rng, plo, phi);
            }
        }
    }
    return p;
}

// Adversarial: a big term at X (a slot or C), its exact negation at Z, 1 to 3 small terms at other slots with
// random gaps 0..40 below the big term, random signs; C is the big term, small, zero or random.
// hi_a, hi_b: largest exponents of the big terms (default: the format's wide range).
inline Problem adversarial(int form, std::mt19937_64 & rng, int hi_a = 1000, int hi_b = 1000) {
    Problem p{};
    const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
    int alo, ahi, blo, bhi;
    wide_range(fa, alo, ahi);
    wide_range(fb, blo, bhi);
    ahi = std::min(ahi, hi_a);
    bhi = std::min(bhi, hi_b);
    const am::FmtInfo ia = am::info(fa), ib = am::info(fb);
    auto mk = [&](const am::FmtInfo & fi, int e, bool neg) {
        e = std::max(1 - fi.bias, std::min(e, (fi.e4m3 ? 8 : (1 << fi.ebits) - 2 - fi.bias)));
        uint32_t m = (uint32_t) (rng() & ((1u << fi.mbits) - 1));
        if (fi.e4m3 && e == 8 && m == 7) {
            m = 6;
        }
        return ((uint32_t) neg << (fi.ebits + fi.mbits)) | ((uint32_t) (e + fi.bias) << fi.mbits) | m;
    };
    for (int r = 0; r < 16; ++r) {
        const int X = (int) (rng() % 17) - 1, Z = (int) (rng() % 16);  // X = -1: C
        const int ea = ahi - (int) (rng() % 4), eb = bhi - (int) (rng() % 4);
        const uint32_t abig = mk(ia, ea, rng() & 1), bbig = mk(ib, eb, false);
        for (int s = 0; s < 16; ++s) {
            p.a[r][s] = 0;
        }
        p.a[r][Z] = abig ^ (1u << (ia.ebits + ia.mbits));
        const int nsmall = 1 + (int) (rng() % 3);
        int sl[3];
        for (int k = 0; k < nsmall; ++k) {
            sl[k] = (int) (rng() % 16);
            const int gap = (int) (rng() % 41);
            p.a[r][sl[k]] = mk(ia, ea - gap / 2, rng() & 1);
        }
        if (X >= 0 && X != Z) {
            p.a[r][X] = abig;
        }
        for (int c = 0; c < 16; ++c) {
            // column c of B: the big pair's b at X and Z, the small terms' b varied per column
            for (int s = 0; s < 16; ++s) {
                p.b[c][s] = mk(ib, eb - (int) (rng() % 21) / 2, rng() & 1);
            }
            p.b[c][Z] = bbig;
            if (X >= 0) {
                p.b[c][X] = bbig;
            }
            const int pick = (int) (rng() % 3);
            if (X < 0 || X == Z) {
                // C = the big product exactly (fits FP32: at most 22 significant bits), cancelled by Z
                const am::Dec da = am::decode(fa, abig), db = am::decode(fb, bbig);
                p.c[r][c] = am::round_int(am::F32, (da.neg ? -1 : 1) * (__int128) (da.m * db.m),
                                          da.e - da.mbits + db.e - db.mbits);
            } else if (pick == 0) {
                p.c[r][c] = rand_f32(rng, ea + eb - 40, ea + eb - 10);
            } else if (pick == 1) {
                p.c[r][c] = 0;
            } else {
                p.c[r][c] = rand_f32(rng, ea + eb - 5, ea + eb + 2);
            }
        }
    }
    return p;
}

}  // namespace ad
