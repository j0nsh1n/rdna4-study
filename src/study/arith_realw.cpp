// Real weights against the Gaussian stand-in: repeats the "differs from exact" (diff) and "rounds differently"
// (quant) tables of arith_stats.cpp with E4M3 weight codes read from a real model tensor (scripts/gguf_rows.py
// writes ROWS_FILE). CPU only; uses the bit model in arith_model.h.
//
// Usage: arith_realw ROWS_FILE [--break] [--scale] [--real-only]
//   --scale      (leftovers C, EQ-1a) applies the stored per-tensor scale S the way a kernel would: the card's FP32
//                result is multiplied by S in FP32 (one more rounding), against the exact sum times S rounded once;
//                every output column then describes the scaled result
//   --real-only  skips the stand-in rows (they do not depend on the tensor)
//
// Weights: the real E4M3 codes, taken as row segments of length K (K = 16, 256, 4096 where the row is long
// enough). The per-tensor scale stored next to the weights is not applied: the FP8 x FP8 instruction sees the
// codes only, and a scale applied outside it (one factor per tensor, here one F32; a power of two or any other
// factor per row or block) multiplies every product of a row by the same number, so it does not change the grid
// the products sit on. It would change the output magnitude, which is handled as in arith_stats: a power-of-two
// output scale chosen from the rms of the exact result (log2_rms), computed separately for each weight source.
// Activations: the same synthetic model as arith_stats sample() (Gaussian, 0.1 % outliers at 20x), rounded to
// FP16 / E4M3 / BF16 per path. The stand-in run draws weights as arith_stats does (Gaussian scaled so that 4.5
// sigma is 448, rounded to E4M3); the real run draws the same random numbers and then replaces the weight code,
// so both sources see identical activations for the same sample index.
// Control as in arith_stats quant; --break adds one to the direct count so the verdict must FAIL.
#include "arith_model.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

static bool g_break = false;
// EQ-1a: the per-tensor scale as an exact integer mantissa and exponent (S = g_sm * 2^g_se), and its float value.
static bool g_scaled = false;
static uint64_t g_sm = 1;
static int g_se = 0;
static float g_s = 1.0f;

static void par_for(size_t n, const std::function<void(size_t, int)> & f) {
    const int T = 8;
    std::vector<std::thread> th;
    std::atomic<size_t> next{0};
    for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < n;) {
                f(i, t);
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
}

static float f32(uint32_t b) { return __builtin_bit_cast(float, b); }
static uint32_t bits(float f) { return __builtin_bit_cast(uint32_t, f); }

static int64_t ulps(uint32_t a, uint32_t b) {
    auto ord = [](uint32_t x) { return (x >> 31) ? -(int64_t) (x & 0x7FFFFFFFu) : (int64_t) x; };
    return std::llabs(ord(a) - ord(b));
}

struct Path {
    const char * name;
    const char * tag;
    int form;
};
static const Path paths[] = {{"fp8_fp8 (activations rounded to E4M3)", "fp8", am::F_FF},
                             {"f16 (E4M3 weights widened, FP16 activations)", "f16", am::F_F16},
                             {"bf16 (E4M3 weights widened, BF16 activations)", "bf16", am::F_BF16}};

// Real weights: rows of E4M3 codes. segment() picks the row segment for sample index i.
struct Weights {
    uint32_t rows = 0, cols = 0, kind = 0, type = 0;
    float scale = 0;
    std::vector<uint8_t> codes;
    bool load(const char * path) {
        FILE * f = std::fopen(path, "rb");
        if (!f) {
            return false;
        }
        char magic[8];
        bool ok = std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, "E4M3ROWS", 8);
        ok = ok && std::fread(&rows, 4, 1, f) == 1 && std::fread(&cols, 4, 1, f) == 1 && std::fread(&kind, 4, 1, f) == 1 &&
             std::fread(&type, 4, 1, f) == 1 && std::fread(&scale, 4, 1, f) == 1;
        if (ok) {
            codes.resize((size_t) rows * cols);
            ok = std::fread(codes.data(), 1, codes.size(), f) == codes.size();
        }
        std::fclose(f);
        return ok;
    }
    // Start offset of the segment for sample i. Stride 7919 is prime, so the first `total` indices are distinct.
    size_t segment(size_t i, int K) const {
        const size_t total = (size_t) rows * (cols / K);
        return (i * 7919ull + 13) % total * K;
    }
};

struct Source {
    const Weights * real;  // null: Gaussian stand-in
};

// One weight code and one activation as the form's input bits. wc_real < 0: use the Gaussian stand-in code.
static void sample(const Path & p, std::mt19937_64 & rng, int wc_real, uint32_t & wa, uint32_t & xb) {
    std::normal_distribution<double> n01(0, 1);
    std::uniform_real_distribution<double> u01(0, 1);
    const float w = (float) (n01(rng) * 448.0 / 4.5);
    const uint32_t wcode = wc_real >= 0 ? (uint32_t) wc_real : am::convert(am::F32, bits(w), am::E4M3, true);
    double x = n01(rng);
    if (u01(rng) < 0.001) {
        x *= 20;
    }
    if (p.form == am::F_FF) {
        wa = wcode;
        xb = am::convert(am::F32, bits((float) (x * 448.0 / 80.0)), am::E4M3, true);
    } else if (p.form == am::F_F16) {
        wa = am::convert(am::E4M3, wcode, am::F16);
        xb = am::convert(am::F32, bits((float) x), am::F16);
    } else {
        wa = am::convert(am::E4M3, wcode, am::BF16);
        xb = am::convert(am::F32, bits((float) x), am::BF16);
    }
}

struct Elem {
    uint32_t hw, exact;
    double exact_val;
    uint32_t exact16, exact8;
};

static Elem element(const Path & p, int K, std::mt19937_64 & rng, int s16, int s8, const Source & src, size_t idx) {
    const am::Params P = am::measured_params(p.form);
    uint32_t acc = 0;
    am::Exact ex, ex16, ex8;
    const size_t base = src.real ? src.real->segment(idx, K) : 0;
    for (int s = 0; s < K / 16; ++s) {
        uint32_t a[16], b[16];
        for (int k = 0; k < 16; ++k) {
            const int wc = src.real ? src.real->codes[base + (size_t) s * 16 + k] : -1;
            sample(p, rng, wc, a[k], b[k]);
            const am::Dec da = am::decode(am::fmt_a(p.form), a[k]), db = am::decode(am::fmt_b(p.form), b[k]);
            const bool neg = (da.neg != db.neg) != (g_scaled && g_s < 0);
            const uint64_t m = (uint64_t) (da.m * db.m) * (g_scaled ? g_sm : 1);
            const int x = da.e - da.mbits + db.e - db.mbits + (g_scaled ? g_se : 0);
            ex.add(neg, m, x);
            ex16.add(neg, m, x + s16);
            ex8.add(neg, m, x + s8);
        }
        acc = am::model_f32(p.form, a, b, acc, P);
    }
    Elem e;
    // with --scale the kernel's last step: one FP32 multiply by S (RNE, as v_mul_f32)
    e.hw = g_scaled ? bits(f32(acc) * g_s) : acc;
    e.exact = ex.round(am::F32);
    e.exact_val = (double) f32(e.exact);
    e.exact16 = ex16.round(am::F16);
    e.exact8 = ex8.round(am::E4M3, true);
    return e;
}

static int log2_rms(const Path & p, int K, const Source & src) {
    std::mt19937_64 rng(99);
    double s = 0;
    for (int i = 0; i < 400; ++i) {
        const Elem e = element(p, K, rng, 0, 0, src, (size_t) i);
        s += e.exact_val * e.exact_val;
    }
    return (int) std::lround(std::log2(std::sqrt(s / 400)));
}

static uint32_t scaled_f32(uint32_t v, int e) { return bits(std::ldexp(f32(v), e)); }

struct Cell {
    long N, d, bins[6], ch16, ch8, big16, big8, self16, ctl16, ctl8, dir16, dir8, sat8;
    int64_t umax;
    int s16, s8;
};

static Cell run_cell(const Path & p, int K, long N, const Source & src) {
    Cell c = {};
    c.N = N;
    const int lr = log2_rms(p, K, src);
    c.s16 = -lr;
    c.s8 = 6 - lr;
    std::vector<Elem> es(N);
    par_for((size_t) N, [&](size_t i, int) {
        std::mt19937_64 rng(1000003ull * K + i * 7919 + p.form);
        es[i] = element(p, K, rng, c.s16, c.s8, src, i);
    });
    const int s16 = c.s16, s8 = c.s8;
    auto bin = [](int64_t x) { return x == 0 ? 0 : x == 1 ? 1 : x == 2 ? 2 : x <= 4 ? 3 : x <= 16 ? 4 : 5; };
    for (const Elem & e : es) {
        const int64_t u = ulps(e.hw, e.exact);
        c.d += u != 0;
        ++c.bins[bin(u)];
        c.umax = std::max(c.umax, u);
        const uint32_t h16 = am::convert(am::F32, scaled_f32(e.hw, s16), am::F16);
        const uint32_t h8 = am::convert(am::F32, scaled_f32(e.hw, s8), am::E4M3, true);
        if (h16 != e.exact16) {
            ++c.ch16;
            c.big16 += std::llabs((int64_t) (h16 & 0x7FFF) - (int64_t) (e.exact16 & 0x7FFF)) > 1;
        }
        if (h8 != e.exact8) {
            ++c.ch8;
            c.big8 += std::abs((int) (h8 & 0x7F) - (int) (e.exact8 & 0x7F)) > 1;
        }
        c.sat8 += (e.exact8 & 0x7F) == 0x7E;
        // Control: counter on (exact FP32, next FP32 up) against a direct count of target midpoints between them.
        const uint32_t v = scaled_f32(e.exact, s16), vu = bits(std::nextafter(f32(v), INFINITY));
        c.self16 += am::convert(am::F32, v, am::F16) != am::convert(am::F32, v, am::F16);
        c.ctl16 += am::convert(am::F32, v, am::F16) != am::convert(am::F32, vu, am::F16);
        auto direct = [](uint32_t vb, uint32_t vub, am::Fmt f, bool sat) {
            // t = the target value v rounds to; the only boundary that can lie in (v, v+] is the midpoint
            // between t and the next larger target value.
            const double x = f32(vb), xu = f32(vub);
            const uint32_t t = am::convert(am::F32, vb, f, sat);
            const uint32_t sign = 1u << (am::info(f).ebits + am::info(f).mbits), mag = t & ~sign;
            const uint32_t above = !(t & sign) ? t + 1 : (mag ? (sign | (mag - 1)) : 1u);
            const double av = am::value(f, above);
            if (!std::isfinite(av)) {
                return false;
            }
            const double mid = 0.5 * (am::value(f, t) + av);
            if (x == mid || (x < mid && mid < xu)) {
                return true;
            }
            return mid == xu && (above & 1) == 0;
        };
        c.dir16 += direct(v, vu, am::F16, false);
        const uint32_t v8 = scaled_f32(e.exact, s8), vu8 = bits(std::nextafter(f32(v8), INFINITY));
        c.ctl8 += am::convert(am::F32, v8, am::E4M3, true) != am::convert(am::F32, vu8, am::E4M3, true);
        c.dir8 += direct(v8, vu8, am::E4M3, true);
    }
    if (g_break) {
        c.dir16 += 1;
    }
    return c;
}

static double ci(long k, long N) {
    const double ph = (double) k / N;
    return 1.96 * std::sqrt(std::max(ph * (1 - ph), 1.0 / N) / N);
}

static void code_summary(const char * what, const std::vector<uint32_t> & codes) {
    // exp = bits 6..3 of the E4M3 code; field 0 is zero and subnormals
    long zero = 0, nan = 0, ex[16] = {}, neg = 0;
    for (uint32_t c : codes) {
        zero += (c & 0x7F) == 0;
        nan += (c & 0x7F) == 0x7F;
        neg += (c >> 7) & 1;
        ++ex[(c >> 3) & 15];
    }
    const double n = (double) codes.size();
    std::printf("RESULT codes %-8s n %zu zero_code %.4f%% nan_code %ld negative %.2f%% exponent_field_share_%%[0..15]:", what,
                codes.size(), 100.0 * zero / n, nan, 100.0 * neg / n);
    for (int i = 0; i < 16; ++i) {
        std::printf(" %.2f", 100.0 * ex[i] / n);
    }
    std::printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: arith_realw ROWS_FILE [--break]\n");
        return 2;
    }
    bool real_only = false;
    for (int i = 2; i < argc; ++i) {
        g_break = g_break || std::string(argv[i]) == "--break";
        g_scaled = g_scaled || std::string(argv[i]) == "--scale";
        real_only = real_only || std::string(argv[i]) == "--real-only";
    }
    Weights W;
    if (!W.load(argv[1])) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::printf("INFO rows %u cols %u scale_kind %u ggml_type %u per_tensor_scale %g (%s)\n", W.rows, W.cols, W.kind, W.type,
                W.scale, g_scaled ? "APPLIED: FP32 multiply of the card result, exact product for the reference" : "not applied: see header comment");
    if (g_scaled) {
        if (W.kind != 1 || !(std::fabs(W.scale) > 0) || !std::isfinite(W.scale)) {
            std::fprintf(stderr, "--scale needs a finite nonzero per-tensor scale in the rows file\n");
            return 2;
        }
        int e2 = 0;
        const double fr = std::frexp((double) std::fabs(W.scale), &e2);  // |S| = fr * 2^e2, fr in [0.5, 1)
        g_sm = (uint64_t) std::ldexp(fr, 24);
        g_se = e2 - 24;
        g_s = W.scale;
        if (std::ldexp((double) g_sm, g_se) != (double) std::fabs(W.scale)) {
            std::fprintf(stderr, "scale decomposition is not exact\n");
            return 2;
        }
    }

    std::vector<uint32_t> real_codes(W.codes.begin(), W.codes.end()), stand;
    {
        std::mt19937_64 rng(5);
        std::normal_distribution<double> n01(0, 1);
        for (size_t i = 0; i < real_codes.size(); ++i) {
            stand.push_back(am::convert(am::F32, bits((float) (n01(rng) * 448.0 / 4.5)), am::E4M3, true));
        }
    }
    code_summary("real", real_codes);
    code_summary("standin", stand);
    bool ok = true;
    long nan_codes = 0;
    for (uint32_t c : real_codes) {
        nan_codes += (c & 0x7F) == 0x7F;
    }
    ok = ok && nan_codes == 0;  // the instruction would turn a NaN code into a NaN result and bias the table

    std::string text;
    long ctl_total = 0;
    for (const Path & p : paths) {
        for (int K : {16, 256, 4096}) {
            if ((int) W.cols < K) {
                continue;
            }
            const long N = K <= 64 ? 200000 : K == 256 ? 60000 : 6000;
            for (int s = 0; s < (real_only ? 1 : 2); ++s) {
                const Source src{s == 0 ? &W : nullptr};
                const char * sn = s == 0 ? "real" : "standin";
                const Cell c = run_cell(p, K, N, src);
                std::printf("RESULT table %-7s %-5s K %4d N %6ld differ_from_exact %.4f%% ulps[0,1,2,3-4,5-16,>16] %ld %ld %ld %ld %ld %ld "
                            "max_ulps %lld | fp16_changed %.4f%% (+-%.4f) e4m3_changed %.4f%% (+-%.4f) fp16_by_more_than_1_step %ld "
                            "e4m3_by_more_than_1_step %ld e4m3_saturated %.2f%% | out_scale_log2 fp16 %d e4m3 %d | control self %ld "
                            "next_up fp16 %ld direct %ld e4m3 %ld direct %ld\n",
                            sn, p.tag, K, N, 100.0 * c.d / N, c.bins[0], c.bins[1], c.bins[2], c.bins[3], c.bins[4], c.bins[5],
                            (long long) c.umax, 100.0 * c.ch16 / N, 100 * ci(c.ch16, N), 100.0 * c.ch8 / N, 100 * ci(c.ch8, N),
                            c.big16, c.big8, 100.0 * c.sat8 / N, c.s16, c.s8, c.self16, c.ctl16, c.dir16, c.ctl8, c.dir8);
                ok = ok && c.self16 == 0 && c.ctl16 == c.dir16 && c.ctl8 == c.dir8;
                ctl_total += c.ctl16;
                char buf[120];
                std::snprintf(buf, sizeof buf, "%s%s %s K%d %.2f%%/%.2f%%/%.2f%%", text.empty() ? "" : "; ", sn, p.tag, K,
                              100.0 * c.d / N, 100.0 * c.ch16 / N, 100.0 * c.ch8 / N);
                text += buf;
            }
        }
    }
    ok = ok && ctl_total > 0;
    if (g_break) {
        text += " (broken on purpose)";
    }
    std::printf("VERDICT arith_realw: %s %s\n", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
