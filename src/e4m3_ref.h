// Integer-only reference for FP8 E4M3 (OCP: sign, 4 exponent bits with bias 7, 3 mantissa bits, subnormals,
// 0x7F and 0xFF are NaN, no infinity, largest finite value 448). Usable on the host and on the GPU, so the GPU probe
// can compare every input in place; src/cvt_ref_check.cpp tests these functions on the CPU against independent
// definitions.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define E4M3_FN __host__ __device__ inline
#else
#define E4M3_FN inline
#endif

E4M3_FN float e4m3_decode(uint8_t x) {
    const uint32_t sign = (uint32_t) (x & 0x80) << 24;
    const uint32_t e = (x >> 3) & 15, m = x & 7;
    uint32_t bits;
    if (e == 15 && m == 7) {
        bits = sign | 0x7FC00000u;
    } else if (e == 0) {
        // value m * 2^-9: exact in float; built from the bit pattern to stay integer-only
        if (m == 0) {
            bits = sign;
        } else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 4)) {
                mm <<= 1;
                ++sh;
            }
            // m = 1.xx * 2^(2 - sh), value = 2^(2 - sh - 9)
            bits = sign | ((uint32_t) (127 - 7 - sh) << 23) | ((mm & 3) << 21);
        }
    } else {
        bits = sign | ((e - 7 + 127) << 23) | (m << 20);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round the integer v right by s bits, to nearest, ties to even. ties_away exists only so cvt_ref_check --perturb
// can show that its check fails on a wrong reference.
E4M3_FN uint32_t rne_shift(uint32_t v, int s, bool ties_away = false) {
    if (s <= 0) {
        return v;
    }
    if (s >= 32) {
        return 0;
    }
    const uint32_t q = v >> s, rem = v & ((1u << s) - 1), half = 1u << (s - 1);
    if (rem > half || (rem == half && (ties_away || (q & 1)))) {
        return q + 1;
    }
    return q;
}

// FP32 bits to E4M3 with round to nearest even. sat = false: the OCP non-saturating rule (infinity and anything
// that rounds above 448 become NaN, sign kept). sat = true: those become +-448 instead. NaN stays NaN.
E4M3_FN uint8_t e4m3_encode(uint32_t bits, bool sat, bool ties_away = false) {
    const uint8_t sign = (uint8_t) ((bits >> 31) << 7);
    const uint32_t a = bits & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return sign | 0x7F;
    }
    if (a == 0x7F800000u) {
        return sign | (sat ? 0x7E : 0x7F);
    }
    if (a < 0x00800000u) {
        return sign;  // float subnormals are far below half of E4M3's smallest step (2^-10)
    }
    const int exp = (int) (a >> 23) - 127;
    const uint32_t m24 = (a & 0x7FFFFFu) | 0x800000u;
    if (exp < -6) {
        // subnormal target, in units of 2^-9: m24 * 2^(exp - 23 + 9); q = 8 is the smallest normal, code 0x08
        return sign | (uint8_t) rne_shift(m24, 14 - exp, ties_away);
    }
    uint32_t q = rne_shift(m24, 20, ties_away);  // 1.mmm in [8, 16]
    int e = exp + 7;
    if (q == 16) {
        q = 8;
        ++e;
    }
    if (e > 15 || (e == 15 && (q & 7) == 7)) {
        return sign | (sat ? 0x7E : 0x7F);
    }
    return sign | (uint8_t) ((e << 3) | (q & 7));
}
