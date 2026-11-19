// Lane 5, items 3 and 4: how far the matrix instruction is from an exact sum, and how often that changes a result
// after rounding to FP16 or E4M3. CPU only: uses the model (arith_model.h), which arith_check showed equal to the
// card on every tested element, single instructions and chains through C up to K = 4096.
//
// Usage: arith_stats exact|diff|quant|quantbf16|worst [--break]
//   quantbf16 (leftovers C, EQ-3): the quant table with BF16 as the target (RNE), beside FP16 on the same outputs
//            and the same scale (rms about 1); control as quant, for BF16
//   exact  the exactness rule: normal inputs on a common grid q with sum of magnitudes below 2^24 q, and no
//          partial sum outside the FP32 normal range, give the exact sum; random FP8 rows of normal codes with a
//          bounded spread of product exponents are counted against the rule.
//   diff   model-shaped data (E4M3 weights from a Gaussian; activations FP16 from a Gaussian with 0.1 % outliers
//          at 20x; separately activations rounded to E4M3), chained through C for K = 16 .. 4096, for the forms
//          fp8_fp8, f16 (FP8 weights widened exactly) and bf16 (weights widened, activations rounded to BF16).
//          Card result against the exact sum rounded once to FP32; also a sequential FP32 FMA chain.
//   quant  the same data: card result and exact result rounded to FP16 and to E4M3 (saturating); share of
//          elements whose rounded value differs, with a 95 % interval. Control: the counter on exact against the
//          next FP32 value up must equal a directly counted share of rounding boundaries in between.
//   worst  random-restart hill climbing for the largest difference (ulps of the exact result; relative to the
//          sum of magnitudes) on single instructions.
//   --break  the exactness and quant controls are evaluated with a wrong expectation; the verdict must FAIL.
#include "arith_model.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

static bool g_break = false;

static void verdict(const char * probe, bool ok, const std::string & text) {
    std::printf("VERDICT %s: %s %s\n", probe, ok ? "PASS" : "FAIL", text.c_str());
}

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

// Distance in representable FP32 values (ordered bits).
static int64_t ulps(uint32_t a, uint32_t b) {
    auto ord = [](uint32_t x) { return (x >> 31) ? -(int64_t) (x & 0x7FFFFFFFu) : (int64_t) x; };
    return std::llabs(ord(a) - ord(b));
}

// ---------------------------------------------------------------------------------------------------------------
// Data shaped like model data.

struct Path {
    const char * name;
    int form;
    bool act_fp8;
};
static const Path paths[] = {{"fp8_fp8 (activations rounded to E4M3)", am::F_FF, true},
                             {"f16 (E4M3 weights widened, FP16 activations)", am::F_F16, false},
                             {"bf16 (E4M3 weights widened, BF16 activations)", am::F_BF16, false}};

// One weight code (E4M3) and one activation as the form's input bits.
static void sample(const Path & p, std::mt19937_64 & rng, uint32_t & wa, uint32_t & xb) {
    std::normal_distribution<double> n01(0, 1);
    std::uniform_real_distribution<double> u01(0, 1);
    // per-tensor scale so that 4.5 standard deviations reach 448
    const float w = (float) (n01(rng) * 448.0 / 4.5);
    const uint32_t wcode = am::convert(am::F32, bits(w), am::E4M3, true);
    double x = n01(rng);
    if (u01(rng) < 0.001) {
        x *= 20;
    }
    if (p.form == am::F_FF) {
        wa = wcode;
        // activations: per-tensor scale with the outliers (about 4 x 20) at 448
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
    uint32_t hw, exact, fma;
    double sumabs, exact_val;
    uint32_t exact16, exact8;  // exact sum rounded once to FP16 and to E4M3 after the power-of-two scale
    uint32_t exactb;           // (leftovers C, EQ-3) the same as exact16, rounded once to BF16
};

// K/16 chained instructions; exact sum over all K products. scale: power of two applied before quantizing.
static Elem element(const Path & p, int K, std::mt19937_64 & rng, int scale_exp16, int scale_exp8) {
    const am::Params P = am::measured_params(p.form);
    uint32_t acc = 0;
    am::Exact ex, ex16, ex8;
    float fma = 0;
    double sumabs = 0;
    for (int s = 0; s < K / 16; ++s) {
        uint32_t a[16], b[16];
        for (int k = 0; k < 16; ++k) {
            sample(p, rng, a[k], b[k]);
            const am::Dec da = am::decode(am::fmt_a(p.form), a[k]), db = am::decode(am::fmt_b(p.form), b[k]);
            const bool neg = da.neg != db.neg;
            const uint64_t m = (uint64_t) (da.m * db.m);
            const int x = da.e - da.mbits + db.e - db.mbits;
            ex.add(neg, m, x);
            ex16.add(neg, m, x + scale_exp16);
            ex8.add(neg, m, x + scale_exp8);
            const double pv = std::ldexp((double) m, x) * (neg ? -1 : 1);
            sumabs += std::fabs(pv);
            fma = std::fmaf((float) am::value(am::fmt_a(p.form), a[k]), (float) am::value(am::fmt_b(p.form), b[k]), fma);
        }
        acc = am::model_f32(p.form, a, b, acc, P);
    }
    Elem e;
    e.hw = acc;
    e.exact = ex.round(am::F32);
    e.fma = bits(fma);
    e.sumabs = sumabs;
    e.exact_val = (double) f32(e.exact);
    e.exact16 = ex16.round(am::F16);
    e.exact8 = ex8.round(am::E4M3, true);
    e.exactb = ex16.round(am::BF16);
    return e;
}

static int log2_rms(const Path & p, int K) {
    std::mt19937_64 rng(99);
    double s = 0;
    for (int i = 0; i < 400; ++i) {
        const Elem e = element(p, K, rng, 0, 0);
        s += e.exact_val * e.exact_val;
    }
    return (int) std::lround(std::log2(std::sqrt(s / 400)));
}

static uint32_t scaled_f32(uint32_t v, int e) { return bits(std::ldexp(f32(v), e)); }

// ---------------------------------------------------------------------------------------------------------------

static bool mode_exact(std::string & text) {
    // FP8 rows of normal codes: product exponents e_a + e_b inside a window of `spread`; C = 0. The rule
    // (matrix-arithmetic.md, Summary) needs normal inputs and no FP32 underflow or overflow, both true here; it
    // predicts exactness when every product is a multiple of q = 2^(emin - 6) and the sum of magnitudes stays
    // below 2^24 q; with 16 products of at most 15 * 15 * 2^(emax - 6) that holds for spread <= 12.
    bool ok = true;
    for (int form : {am::F_FF, am::F_F16}) {
        for (int spread : {8, 12, 13, 16, 20}) {
            std::mt19937_64 rng(spread * 13 + form);
            const am::Params P = am::measured_params(form);
            long n = 0, diff = 0, rule_true = 0, rule_true_diff = 0;
            for (int it = 0; it < 200000; ++it) {
                uint32_t a[16], b[16];
                const int base = -3;
                for (int k = 0; k < 16; ++k) {
                    // E4M3 normal values with exponent base .. base + spread split between a and b
                    const int e = base + (int) (rng() % (spread + 1));
                    const int ea = std::min(8, std::max(-6, e / 2)), eb = std::min(8, std::max(-6, e - e / 2));
                    // mantissa 7 at the top exponent would be the NaN code
                    const uint32_t ma = (uint32_t) (rng() & 7), mb = (uint32_t) (rng() & 7);
                    const uint32_t ca = ((uint32_t) (rng() & 1) << 7) | ((uint32_t) (ea + 7) << 3) | (ea == 8 && ma == 7 ? 6 : ma);
                    const uint32_t cb = ((uint32_t) (eb + 7) << 3) | (eb == 8 && mb == 7 ? 6 : mb);
                    a[k] = form == am::F_FF ? ca : am::convert(am::E4M3, ca, am::F16);
                    b[k] = form == am::F_FF ? cb : am::convert(am::E4M3, cb, am::F16);
                }
                const uint32_t hw = am::model_f32(form, a, b, 0, P);
                const uint32_t exv = am::exact_round(form, a, b, 0, am::F32);
                // rule: grid q = smallest product LSB; sum of magnitudes below 2^24 q
                int qe = 1000;
                double sm = 0;
                for (int k = 0; k < 16; ++k) {
                    const am::Dec da = am::decode(am::fmt_a(form), a[k]), db = am::decode(am::fmt_b(form), b[k]);
                    if (da.m && db.m) {
                        int64_t m = da.m * db.m;
                        int x = da.e - da.mbits + db.e - db.mbits;
                        while (!(m & 1)) {
                            m >>= 1;
                            ++x;
                        }
                        qe = std::min(qe, x);
                        sm += std::ldexp((double) (da.m * db.m), da.e - da.mbits + db.e - db.mbits);
                    }
                }
                const bool rule = qe == 1000 || sm < std::ldexp(1.0, 24 + qe);
                ++n;
                diff += hw != exv;
                rule_true += rule;
                rule_true_diff += rule && hw != exv;
            }
            std::printf("RESULT exact %-4s product_exponent_spread %2d rows %ld differ_from_exact %ld rule_holds %ld "
                        "rule_holds_but_differs %ld\n",
                        form == am::F_FF ? "fp8" : "f16w", spread, n, diff, rule_true, rule_true_diff);
            ok = ok && rule_true_diff == 0;
            if (spread <= 12 && form == am::F_FF) {
                ok = ok && diff == 0;
            }
            char buf[120];
            std::snprintf(buf, sizeof buf, "%s%s spread %d: %ld/%ld differ", text.empty() ? "" : "; ",
                          form == am::F_FF ? "fp8" : "f16w", spread, diff, n);
            text += buf;
        }
    }
    if (g_break) {
        ok = false;
    }
    return ok;
}

static bool mode_diff_quant(bool quant, std::string & text) {
    bool ok = true;
    long ctl_total = 0;  // the control must count something somewhere, or it shows nothing
    for (const Path & p : paths) {
        for (int K : {16, 64, 256, 1024, 4096}) {
            const long N = K <= 64 ? 200000 : K == 256 ? 60000 : K == 1024 ? 16000 : 6000;
            const int lr = log2_rms(p, K);
            const int s16 = -lr, s8 = 6 - lr;  // FP16: rms about 1; E4M3: rms about 64 (outliers saturate)
            std::vector<Elem> es(N);
            par_for((size_t) N, [&](size_t i, int) {
                std::mt19937_64 rng(1000003ull * K + i * 7919 + p.form);
                es[i] = element(p, K, rng, s16, s8);
            });
            if (!quant) {
                long d = 0, dfma = 0, bins[6] = {}, binsf[6] = {};
                int64_t umax = 0, ufmax = 0;
                double rel_max = 0, rel_sum = 0, relf_max = 0;
                for (const Elem & e : es) {
                    const int64_t u = ulps(e.hw, e.exact), uf = ulps(e.fma, e.exact);
                    d += u != 0;
                    dfma += uf != 0;
                    auto bin = [](int64_t x) { return x == 0 ? 0 : x == 1 ? 1 : x == 2 ? 2 : x <= 4 ? 3 : x <= 16 ? 4 : 5; };
                    ++bins[bin(u)];
                    ++binsf[bin(uf)];
                    umax = std::max(umax, u);
                    ufmax = std::max(ufmax, uf);
                    const double rel = std::fabs((double) f32(e.hw) - e.exact_val) / e.sumabs;
                    rel_max = std::max(rel_max, rel);
                    rel_sum += rel;
                    relf_max = std::max(relf_max, std::fabs((double) f32(e.fma) - e.exact_val) / e.sumabs);
                }
                std::printf("RESULT diff %-48s K %4d N %6ld differ %.4f%% ulps[0,1,2,3-4,5-16,>16] %ld %ld %ld %ld %ld %ld "
                            "max_ulps %lld max_rel_to_sumabs %.3g (2^%.1f) mean_rel %.3g | fma_chain differ %.4f%% max_ulps "
                            "%lld max_rel %.3g bins %ld %ld %ld %ld %ld %ld\n",
                            p.name, K, N, 100.0 * d / N, bins[0], bins[1], bins[2], bins[3], bins[4], bins[5],
                            (long long) umax, rel_max, std::log2(rel_max > 0 ? rel_max : 1e-300), rel_sum / N,
                            100.0 * dfma / N, (long long) ufmax, relf_max, binsf[0], binsf[1], binsf[2], binsf[3],
                            binsf[4], binsf[5]);
                char buf[100];
                std::snprintf(buf, sizeof buf, "%s%s K%d %.2f%%", text.empty() ? "" : "; ", p.form == am::F_FF ? "fp8" : p.form == am::F_F16 ? "f16" : "bf16", K, 100.0 * d / N);
                text += buf;
            } else {
                long ch16 = 0, ch8 = 0, big16 = 0, big8 = 0, self16 = 0, ctl16 = 0, ctl8 = 0, dir16 = 0, dir8 = 0, sat8 = 0;
                for (const Elem & e : es) {
                    const uint32_t h16 = am::convert(am::F32, scaled_f32(e.hw, s16), am::F16);
                    const uint32_t h8 = am::convert(am::F32, scaled_f32(e.hw, s8), am::E4M3, true);
                    if (h16 != e.exact16) {
                        ++ch16;
                        big16 += std::llabs((int64_t) (h16 & 0x7FFF) - (int64_t) (e.exact16 & 0x7FFF)) > 1;
                    }
                    if (h8 != e.exact8) {
                        ++ch8;
                        big8 += std::abs((int) (h8 & 0x7F) - (int) (e.exact8 & 0x7F)) > 1;
                    }
                    sat8 += (e.exact8 & 0x7F) == 0x7E;
                    // controls: the counter on (exact FP32, the next FP32 value up), and the same share counted
                    // directly as "a rounding boundary of the target lies above v and at or below v+"
                    const uint32_t v = scaled_f32(e.exact, s16), vu = bits(std::nextafter(f32(v), INFINITY));
                    self16 += am::convert(am::F32, v, am::F16) != am::convert(am::F32, v, am::F16);
                    ctl16 += am::convert(am::F32, v, am::F16) != am::convert(am::F32, vu, am::F16);
                    auto direct = [](uint32_t vb, uint32_t vub, am::Fmt f, bool sat) {
                        // t = the target value v rounds to; the only rounding boundary that can lie in (v, v+] is
                        // the midpoint between t and the next larger target value
                        const double x = f32(vb), xu = f32(vub);
                        const uint32_t t = am::convert(am::F32, vb, f, sat);
                        const uint32_t sign = 1u << (am::info(f).ebits + am::info(f).mbits), mag = t & ~sign;
                        const uint32_t above = !(t & sign) ? t + 1 : (mag ? (sign | (mag - 1)) : 1u);
                        const double av = am::value(f, above);
                        if (!std::isfinite(av)) {
                            return false;  // saturating E4M3 at 448, or FP16 at its largest value: no boundary
                        }
                        const double mid = 0.5 * (am::value(f, t) + av);
                        if (x == mid || (x < mid && mid < xu)) {
                            return true;  // v on the midpoint rounds down to t (even), v+ rounds up
                        }
                        return mid == xu && (above & 1) == 0;  // a tie at v+ goes to the even code
                    };
                    dir16 += direct(v, vu, am::F16, false);
                    const uint32_t v8 = scaled_f32(e.exact, s8), vu8 = bits(std::nextafter(f32(v8), INFINITY));
                    ctl8 += am::convert(am::F32, v8, am::E4M3, true) != am::convert(am::F32, vu8, am::E4M3, true);
                    dir8 += direct(v8, vu8, am::E4M3, true);
                }
                auto ci = [&](long k) {
                    const double ph = (double) k / N;
                    return 1.96 * std::sqrt(std::max(ph * (1 - ph), 1.0 / N) / N);
                };
                if (g_break) {
                    dir16 += 1;
                }
                std::printf("RESULT quant %-48s K %4d N %6ld fp16_changed %.4f%% (+-%.4f) by_more_than_1_step %ld | "
                            "e4m3_changed %.4f%% (+-%.4f) by_more_than_1_step %ld saturated %.2f%% | control self %ld "
                            "next_up fp16 %ld direct %ld e4m3 %ld direct %ld\n",
                            p.name, K, N, 100.0 * ch16 / N, 100 * ci(ch16), big16, 100.0 * ch8 / N, 100 * ci(ch8), big8,
                            100.0 * sat8 / N, self16, ctl16, dir16, ctl8, dir8);
                ok = ok && self16 == 0 && ctl16 == dir16 && ctl8 == dir8;
                ctl_total += ctl16;
                char buf[120];
                std::snprintf(buf, sizeof buf, "%s%s K%d %.3f%%/%.3f%%", text.empty() ? "" : "; ",
                              p.form == am::F_FF ? "fp8" : p.form == am::F_F16 ? "f16" : "bf16", K, 100.0 * ch16 / N,
                              100.0 * ch8 / N);
                text += buf;
            }
        }
    }
    return ok && (!quant || ctl_total > 0);
}

// Hill climbing for the largest difference on one instruction.
static bool mode_worst(std::string & text) {
    for (int form : {am::F_F16, am::F_BF16, am::F_FF}) {
        const am::Params P = am::measured_params(form);
        const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
        const int ab = am::info(fa).ebits + am::info(fa).mbits + 1;
        double best_rel = 0;
        int64_t best_ulp = 0;
        std::string best_case;
        std::mutex mu;
        par_for(64, [&](size_t restart, int) {
            std::mt19937_64 rng(restart * 31 + form);
            uint32_t a[16], b[16], c;
            auto finite = [&](am::Fmt f, uint32_t v) {
                const am::Dec d = am::decode(f, v);
                return !d.nan && !d.inf;
            };
            auto rnd = [&](am::Fmt f) {
                for (;;) {
                    const uint32_t v = (uint32_t) (rng() & ((1u << ab) - 1));
                    if (finite(f, v)) {
                        return v;
                    }
                }
            };
            for (int k = 0; k < 16; ++k) {
                a[k] = rnd(fa);
                b[k] = rnd(fb);
            }
            c = 0;
            auto score = [&](bool relmode) {
                const uint32_t hw = am::model_f32(form, a, b, c, P), ex = am::exact_round(form, a, b, c, am::F32);
                if (!finite(am::F32, hw) || !finite(am::F32, ex)) {
                    return -1.0;
                }
                double sm = std::fabs((double) f32(c));
                for (int k = 0; k < 16; ++k) {
                    sm += std::fabs(am::value(fa, a[k]) * am::value(fb, b[k]));
                }
                if (sm == 0) {
                    return 0.0;
                }
                return relmode ? std::fabs((double) f32(hw) - (double) f32(ex)) / sm : (double) ulps(hw, ex);
            };
            for (int mode = 0; mode < 2; ++mode) {
                double cur = score(mode == 1);
                for (int it = 0; it < 20000; ++it) {
                    uint32_t sa[16], sb[16], sc = c;
                    std::copy(a, a + 16, sa);
                    std::copy(b, b + 16, sb);
                    const int which = (int) (rng() % 33), bit = (int) (rng() % (which == 32 ? 32 : ab));
                    if (which < 16) {
                        a[which] ^= 1u << bit;
                    } else if (which < 32) {
                        b[which - 16] ^= 1u << bit;
                    } else {
                        c ^= 1u << bit;
                    }
                    bool fin = finite(am::F32, c);
                    for (int k = 0; k < 16; ++k) {
                        fin = fin && finite(fa, a[k]) && finite(fb, b[k]);
                    }
                    const double s = fin ? score(mode == 1) : -1;
                    if (s >= cur) {
                        cur = s;
                    } else {
                        std::copy(sa, sa + 16, a);
                        std::copy(sb, sb + 16, b);
                        c = sc;
                    }
                }
                std::lock_guard<std::mutex> g(mu);
                if (mode == 1 && cur > best_rel) {
                    best_rel = cur;
                    char buf[200];
                    const uint32_t hw = am::model_f32(form, a, b, c, P), ex = am::exact_round(form, a, b, c, am::F32);
                    std::snprintf(buf, sizeof buf, "card %a exact %a C %a", (double) f32(hw), (double) f32(ex), (double) f32(c));
                    best_case = buf;
                }
                if (mode == 0) {
                    best_ulp = std::max(best_ulp, (int64_t) cur);
                }
            }
        });
        std::printf("RESULT worst %-8s max_rel_to_sumabs %.4g (2^%.2f) max_ulps_of_exact %lld example: %s\n",
                    form == am::F_FF ? "fp8_fp8" : form == am::F_F16 ? "f16" : "bf16", best_rel, std::log2(best_rel),
                    (long long) best_ulp, best_case.c_str());
        char buf[100];
        std::snprintf(buf, sizeof buf, "%s%s 2^%.2f", text.empty() ? "" : "; ", form == am::F_FF ? "fp8" : form == am::F_F16 ? "f16" : "bf16",
                      std::log2(best_rel));
        text += buf;
    }
    return true;
}

// EQ-3: how often the card's FP32 result and the exact sum round to different BF16 values. Same outputs, scale and
// control as mode quant (the control counted for BF16 midpoints); --break adds one to the direct BF16 count.
static bool mode_quant_bf16(std::string & text) {
    bool ok = true;
    long ctl_total = 0;
    for (const Path & p : paths) {
        for (int K : {16, 64, 256, 1024, 4096}) {
            const long N = K <= 64 ? 200000 : K == 256 ? 60000 : K == 1024 ? 16000 : 6000;
            const int lr = log2_rms(p, K);
            const int s16 = -lr, s8 = 6 - lr;
            std::vector<Elem> es(N);
            par_for((size_t) N, [&](size_t i, int) {
                std::mt19937_64 rng(1000003ull * K + i * 7919 + p.form);
                es[i] = element(p, K, rng, s16, s8);
            });
            long ch16 = 0, chb = 0, bigb = 0, selfb = 0, ctlb = 0, dirb = 0;
            for (const Elem & e : es) {
                const uint32_t v = scaled_f32(e.hw, s16);
                ch16 += am::convert(am::F32, v, am::F16) != e.exact16;
                const uint32_t hb = am::convert(am::F32, v, am::BF16);
                if (hb != e.exactb) {
                    ++chb;
                    bigb += std::llabs((int64_t) (hb & 0x7FFF) - (int64_t) (e.exactb & 0x7FFF)) > 1;
                }
                const uint32_t x = scaled_f32(e.exact, s16), xu = bits(std::nextafter(f32(x), INFINITY));
                selfb += am::convert(am::F32, x, am::BF16) != am::convert(am::F32, x, am::BF16);
                ctlb += am::convert(am::F32, x, am::BF16) != am::convert(am::F32, xu, am::BF16);
                // direct: the midpoint between t (what x rounds to) and the next larger BF16 value lies in (x, x+]
                const uint32_t t = am::convert(am::F32, x, am::BF16), sign = 0x8000u, mag = t & 0x7FFFu;
                const uint32_t above = !(t & sign) ? t + 1 : (mag ? (sign | (mag - 1)) : 1u);
                const double av = am::value(am::BF16, above);
                if (std::isfinite(av)) {
                    const double mid = 0.5 * (am::value(am::BF16, t) + av), xv = f32(x), xuv = f32(xu);
                    dirb += (xv == mid || (xv < mid && mid < xuv)) || (mid == xuv && (above & 1) == 0);
                }
            }
            if (g_break) {
                dirb += 1;
            }
            const double ratio = ch16 ? (double) chb / ch16 : NAN;
            std::printf("RESULT quantbf16 %-48s K %4d N %6ld bf16_changed %.4f%% (%ld) by_more_than_1_step %ld | fp16_changed "
                        "%.4f%% (%ld) | ratio bf16/fp16 %.3f | control self %ld next_up bf16 %ld direct %ld\n",
                        p.name, K, N, 100.0 * chb / N, chb, bigb, 100.0 * ch16 / N, ch16, ratio, selfb, ctlb, dirb);
            ok = ok && selfb == 0 && ctlb == dirb;
            ctl_total += ctlb;
            char buf[120];
            std::snprintf(buf, sizeof buf, "%s%s K%d %ld/%ld", text.empty() ? "" : "; ",
                          p.form == am::F_FF ? "fp8" : p.form == am::F_F16 ? "f16" : "bf16", K, chb, ch16);
            text += buf;
        }
    }
    return ok && ctl_total > 0;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    for (int i = 2; i < argc; ++i) {
        g_break = g_break || std::string(argv[i]) == "--break";
    }
    std::string text;
    bool ok;
    if (mode == "exact") {
        ok = mode_exact(text);
    } else if (mode == "diff") {
        ok = mode_diff_quant(false, text);
    } else if (mode == "quant") {
        ok = mode_diff_quant(true, text);
    } else if (mode == "quantbf16") {
        ok = mode_quant_bf16(text);
    } else if (mode == "worst") {
        ok = mode_worst(text);
    } else {
        std::fprintf(stderr, "usage: arith_stats exact|diff|quant|quantbf16|worst [--break]\n");
        return 2;
    }
    if (g_break) {
        text += " (broken on purpose)";
    }
    verdict(("arith_stats_" + mode).c_str(), ok, text);
    return ok ? 0 : 1;
}
