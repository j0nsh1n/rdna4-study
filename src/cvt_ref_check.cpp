// CPU-only test of the E4M3 reference in e4m3_ref.h, against definitions that do not share its code:
// - decode: llama.cpp's own FP8 E4M3 conversion (ggml type traits), all 256 codes;
// - encode: the code whose decoded value is nearest to the input (binary search over the sorted finite values),
//   ties to the even code, 480 standing in for "rounds above the largest value" (then NaN, or 448 saturated),
//   sign kept on zero; for every boundary value (each finite value, its neighbours in FP32, each midpoint and its
//   neighbours, the overflow edge, zeros, infinities, NaNs) and a large random sample.
// Usage: cvt_ref_check [--perturb]. --perturb makes the reference round ties away from zero, and the check must fail.
#include "e4m3_ref.h"
#include "ggml.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

static float float_of(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// Positive finite values in code order 0x00..0x7E, from the definition of the format (not from e4m3_decode).
static std::vector<double> grid() {
    std::vector<double> g;
    for (int c = 0; c <= 0x7E; ++c) {
        const int e = c >> 3, m = c & 7;
        g.push_back(e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7));
    }
    return g;
}

static uint8_t expected_encode(float x, bool sat, const std::vector<double> & g) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    if (std::isnan(x)) {
        return sign | 0x7F;
    }
    if (std::isinf(x)) {
        return sign | (sat ? 0x7E : 0x7F);
    }
    const double a = std::fabs((double) x);
    // candidates: codes 0..0x7E, and 0x7F meaning 480 (overflow)
    auto value = [&](int c) { return c == 0x7F ? 480.0 : g[c]; };
    int lo = 0, hi = 0x7F;
    if (a >= 480.0) {
        return sign | (sat ? 0x7E : 0x7F);
    }
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        (value(mid) <= a ? lo : hi) = mid;
    }
    const double dl = a - value(lo), dh = value(hi) - a;
    int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) == 0 ? lo : hi);
    if (c == 0x7F) {
        return sign | (sat ? 0x7E : 0x7F);
    }
    return sign | (uint8_t) c;
}

int main(int argc, char ** argv) {
    const bool perturb = argc > 1 && std::strcmp(argv[1], "--perturb") == 0;
    long fails = 0;

    // decode, all codes, against llama.cpp
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    uint8_t codes[256];
    float ref[256];
    for (int c = 0; c < 256; ++c) {
        codes[c] = (uint8_t) c;
    }
    traits->to_float(codes, ref, 256);
    long dec_fail = 0;
    for (int c = 0; c < 256; ++c) {
        const float got = e4m3_decode((uint8_t) c);
        const bool same = std::isnan(ref[c]) ? std::isnan(got) : bits_of(got) == bits_of(ref[c]);
        if (!same) {
            if (dec_fail < 5) {
                std::printf("decode mismatch code 0x%02x: reference %a, llama.cpp %a\n", c, got, ref[c]);
            }
            ++dec_fail;
        }
    }
    std::printf("decode: %ld of 256 codes differ from llama.cpp\n", dec_fail);
    fails += dec_fail;

    // encode
    const std::vector<double> g = grid();
    std::vector<uint32_t> inputs;
    auto add = [&](float f) {
        for (float s : {f, -f}) {
            inputs.push_back(bits_of(s));
            inputs.push_back(bits_of(std::nextafter(s, INFINITY)));
            inputs.push_back(bits_of(std::nextafter(s, -INFINITY)));
        }
    };
    for (size_t c = 0; c < g.size(); ++c) {
        add((float) g[c]);
        const double next = c + 1 < g.size() ? g[c + 1] : 480.0;
        add((float) ((g[c] + next) / 2));  // every midpoint is exact in FP32
    }
    for (float f : {0.0f, 448.0f, 464.0f, 480.0f, 500.0f, 1e6f, FLT_MAX, FLT_MIN, std::ldexp(1.0f, -149),
                    std::ldexp(1.0f, -10), std::ldexp(1.0f, -11), INFINITY}) {
        add(f);
    }
    for (uint32_t nan : {0x7FC00000u, 0x7F800001u, 0x7FFFFFFFu}) {
        inputs.push_back(nan);
        inputs.push_back(nan | 0x80000000u);
    }
    const size_t n_boundary = inputs.size();
    std::mt19937_64 rng(42);
    for (int i = 0; i < (1 << 25); ++i) {
        inputs.push_back((uint32_t) rng());
    }
    std::uniform_real_distribution<float> wide(-600.0f, 600.0f), unit(-1.0f, 1.0f);
    for (int i = 0; i < (1 << 23); ++i) {
        inputs.push_back(bits_of(wide(rng)));
        inputs.push_back(bits_of(std::ldexp(unit(rng), -(int) (rng() % 16))));
    }
    long enc_fail[2] = {0, 0}, boundary_fail = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const float x = float_of(inputs[i]);
        for (int sat = 0; sat < 2; ++sat) {
            const uint8_t want = expected_encode(x, sat, g);
            const uint8_t got = e4m3_encode(inputs[i], sat, perturb);
            if (want != got) {
                if (enc_fail[0] + enc_fail[1] < 8) {
                    std::printf("encode mismatch x=%a (0x%08x) sat=%d: reference 0x%02x, nearest-value rule 0x%02x\n",
                                x, inputs[i], sat, got, want);
                }
                ++enc_fail[sat];
                boundary_fail += i < n_boundary;
            }
        }
    }
    std::printf("encode: %zu inputs (%zu boundary values), mismatches plain %ld saturating %ld (boundary %ld)\n",
                inputs.size(), n_boundary, enc_fail[0], enc_fail[1], boundary_fail);
    fails += enc_fail[0] + enc_fail[1];

    char text[200];
    std::snprintf(text, sizeof text, "E4M3 reference: decode %ld/256 differ from llama.cpp, encode %ld mismatches%s",
                  dec_fail, enc_fail[0] + enc_fail[1], perturb ? " (reference perturbed on purpose)" : "");
    std::printf("VERDICT cvt_ref: %s %s\n", fails == 0 ? "PASS" : "FAIL", text);
    return fails == 0 ? 0 : 1;
}
