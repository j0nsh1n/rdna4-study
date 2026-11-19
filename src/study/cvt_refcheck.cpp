// CPU-only test of src/study/cvt_ref.h against definitions that share none of its code:
// - FP8 decode: llama.cpp's own E4M3 conversion (ggml type traits) and the study's older e4m3_ref.h, all 256 codes;
// - BF8 decode: the CPU's FP16 conversion of the code placed in the top byte of an FP16 (E5M2 is FP16 cut to 8 bits);
// - encode, both formats, all four rounding directions, saturating or not: the code found by a search over the
//   sorted code values (nearest, ties to the even code; or floor / ceiling), with the next value past the largest
//   standing in for "overflow"; on every boundary value (each code value, each midpoint, and their FP32
//   neighbours), zeros, infinities, NaNs and 2^24 random inputs;
// - FP8 encode, all 2^32 inputs, against e4m3_ref.h's e4m3_encode (round to nearest even, saturating or not);
// - stochastic rounding: for the carry-then-truncate rule, the number of the 2^20 (FP8) or 2^21 (BF8) operand
//   values that round up equals the input's position between its two codes in those units (the definition of
//   stochastic rounding), on 4096 random inputs.
// Usage: cvt_refcheck [--break]. --break makes the reference under test round ties away from zero; the check must
// then fail.
#include "cvt_ref.h"
#include "e4m3_ref.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static bool g_break = false;

static float fl(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
static uint32_t bi(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// Positive values in code order 0..maxcode, plus one more: the next value the format would have.
static std::vector<double> grid(int f) {
    std::vector<double> v;
    const int M = f == cr::FP8 ? 3 : 2, bias = f == cr::FP8 ? 7 : 15;
    const int maxcode = f == cr::FP8 ? 0x7E : 0x7B;
    for (int c = 0; c <= maxcode + 1; ++c) {
        const int e = c >> M, m = c & ((1 << M) - 1);
        v.push_back(e == 0 ? std::ldexp((double) m, 1 - bias - M) : std::ldexp((double) (m + (1 << M)), e - bias - M));
    }
    return v;
}

// rnd: cr::Rnd. Returns the code by search over the grid.
static uint32_t search_encode(int f, uint32_t bits, bool sat, int rnd) {
    static const std::vector<double> G[2] = {grid(cr::FP8), grid(cr::BF8)};
    const std::vector<double> & v = G[f];
    const uint32_t maxcode = (uint32_t) v.size() - 2;
    const uint32_t s = (bits >> 31) ? 0x80u : 0u;
    const float x = fl(bits);
    if (std::isnan(x)) {
        return s | (f == cr::FP8 ? 0x7Fu : 0x7Eu);
    }
    const uint32_t over = f == cr::FP8 ? 0x7Fu : 0x7Cu;
    if (std::isinf(x)) {
        return s | (sat ? maxcode : over);
    }
    const double ax = std::fabs((double) x);
    const bool neg = s != 0;
    // floor index
    size_t lo = (size_t) (std::upper_bound(v.begin(), v.end(), ax) - v.begin()) - 1;
    uint32_t code;
    if (lo >= v.size() - 1) {
        // at or beyond the "next" value
        const bool toward_zero = rnd == cr::RTZ || (rnd == cr::RUP && neg) || (rnd == cr::RDN && !neg);
        code = toward_zero ? maxcode : maxcode + 1;
    } else if (v[lo] == ax) {
        code = (uint32_t) lo;
    } else {
        const size_t hi = lo + 1;
        bool up;
        if (rnd == cr::RNE) {
            const double dl = ax - v[lo], dh = v[hi] - ax;
            up = dh < dl || (dh == dl && (g_break || (hi & 1) == 0));
        } else if (rnd == cr::RTZ) {
            up = false;
        } else {
            up = (rnd == cr::RUP) != neg;
        }
        code = (uint32_t) (up ? hi : lo);
    }
    if (code > maxcode) {
        code = sat ? maxcode : over;
    }
    return s | code;
}

int main(int argc, char ** argv) {
    g_break = argc > 1 && std::strcmp(argv[1], "--break") == 0;
    // In --break mode the search stays correct and the reference under test rounds ties away.
    const bool ta = g_break;
    g_break = false;
    long bad = 0;
    char text[512];

    // decode
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    float ref[256];
    uint8_t codes[256];
    for (int c = 0; c < 256; ++c) {
        codes[c] = (uint8_t) c;
    }
    traits->to_float(codes, ref, 256);
    long bad_dec = 0;
    for (int c = 0; c < 256; ++c) {
        const float a = fl(cr::decode8(cr::FP8, c));
        const float o = e4m3_decode((uint8_t) c);
        const bool ok8 = (std::isnan(a) && std::isnan(ref[c]) && std::isnan(o)) ||
                         (bi(a) == bi(ref[c]) && bi(a) == bi(o));
        bad_dec += !ok8;
        const uint16_t hbits = (uint16_t) (c << 8);
        _Float16 h;
        std::memcpy(&h, &hbits, 2);
        const float hb = (float) h;
        const float b = fl(cr::decode8(cr::BF8, c));
        bad_dec += !((std::isnan(hb) && std::isnan(b)) || bi(hb) == bi(b));
    }
    std::printf("RESULT decode: %ld mismatches over 2 x 256 codes (FP8 vs llama.cpp and e4m3_ref.h; BF8 vs FP16)\n",
                bad_dec);
    bad += bad_dec;

    // encode against the search
    std::vector<uint32_t> inputs;
    for (int f = 0; f < 2; ++f) {
        const std::vector<double> v = grid(f);
        for (size_t i = 0; i < v.size(); ++i) {
            std::vector<double> pts{v[i]};
            if (i + 1 < v.size()) {
                pts.push_back(0.5 * (v[i] + v[i + 1]));
            }
            for (double p : pts) {
                const uint32_t b = bi((float) p);
                for (int d = -3; d <= 3; ++d) {
                    inputs.push_back(b + d);
                    inputs.push_back((b + d) | 0x80000000u);
                }
            }
        }
    }
    for (uint32_t b : {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F800001u, 1u,
                       0x807FFFFFu, 0x7F7FFFFFu}) {
        inputs.push_back(b);
    }
    std::mt19937_64 rng(12345);
    for (int i = 0; i < (1 << 24); ++i) {
        inputs.push_back((uint32_t) rng());
    }
    long bad_enc = 0, n_enc = 0;
    for (int f = 0; f < 2; ++f) {
        for (int sat = 0; sat < 2; ++sat) {
            for (int rnd = 0; rnd < 4; ++rnd) {
                long b = 0;
                for (uint32_t x : inputs) {
                    const uint32_t want = search_encode(f, x, sat, rnd);
                    const uint32_t got = cr::encode8(f, x, sat, rnd, ta && rnd == cr::RNE);
                    if (want != got) {
                        if (++b <= 3) {
                            std::printf("  MISMATCH fmt %d sat %d rnd %d x=0x%08x (%g): search 0x%02x ref 0x%02x\n", f,
                                        sat, rnd, x, (double) fl(x), want, got);
                        }
                    }
                    ++n_enc;
                }
                bad_enc += b;
            }
        }
    }
    std::printf("RESULT encode vs search: %ld mismatches of %ld (2 formats x sat x 4 rounding modes)\n", bad_enc, n_enc);
    bad += bad_enc;

    // all 2^32 against e4m3_ref.h, 8 threads
    std::atomic<long> bad_all{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t) {
        th.emplace_back([t, ta, &bad_all] {
            long b = 0;
            for (uint64_t x = (uint64_t) t << 29; x < (uint64_t) (t + 1) << 29; ++x) {
                for (int sat = 0; sat < 2; ++sat) {
                    b += cr::encode8(cr::FP8, (uint32_t) x, sat, cr::RNE, ta) != e4m3_encode((uint32_t) x, sat);
                }
            }
            bad_all += b;
        });
    }
    for (auto & t : th) {
        t.join();
    }
    std::printf("RESULT FP8 encode vs e4m3_ref.h, all 2^32 inputs x {sat, non-sat}: %ld mismatches\n", bad_all.load());
    bad += bad_all;

    // stochastic rounding rule: the number of operand values that round up equals the input's position between its
    // two codes, in units of 2^-rb of a step (normal and subnormal targets)
    long bad_sr = 0;
    for (int f = 0; f < 2; ++f) {
        const int rb = f == cr::FP8 ? 20 : 21, sh = 32 - rb;
        const cr::F8 fi = cr::f8info(f);
        for (int i = 0; i < 4096; ++i) {
            uint32_t x = (uint32_t) rng();
            const uint32_t lo_exp = 127 + fi.emin - fi.M - 4, hi_exp = 127 + fi.bias;
            x = (x & 0x807FFFFFu) | ((lo_exp + (uint32_t) (rng() % (hi_exp - lo_exp))) << 23);
            const double ax = std::fabs((double) fl(x));
            const double step = std::ldexp(1.0, std::max((int) std::floor(std::log2(ax)), fi.emin) - fi.M);
            const double frac = ax / step - std::floor(ax / step);
            const long want = (long) std::floor(frac * (double) (1u << rb));
            const uint32_t down = search_encode(f, x, false, cr::RTZ);
            long ups = 0;
            for (uint32_t r = 0; r < (1u << rb); ++r) {
                ups += cr::sr8(f, x, r << sh, cr::SR_ALIGNED, false) != down;
            }
            bad_sr += ups != want;
        }
    }
    std::printf("RESULT stochastic rule (aligned, truncate): %ld of 8192 inputs where round-up count != position\n",
                bad_sr);
    bad += bad_sr;

    std::snprintf(text, sizeof text, "decode %ld, encode-vs-search %ld, fp8-all-vs-e4m3_ref %ld, sr-rule %ld%s",
                  bad_dec, bad_enc, bad_all.load(), bad_sr, ta ? " (--break: ties away)" : "");
    std::printf("VERDICT cvt_refcheck: %s %s\n", bad ? "FAIL" : "PASS", text);
    return bad ? 1 : 0;
}
