// Lane 6, items 5 and 7: what the vector dot-product instructions compute, bit for bit, on gfx1201, against the
// CPU rules in cvt_dot_model.h (the way lane 5 established the matrix model: targeted cases, then large random sets,
// with a control that a changed rule disagrees).
//
// cvt_dot selftest           CPU only: the rules against independent definitions (the CPU's fmaf for the
//                            sequential rule; double arithmetic, exact in the tested range, for the one-rounding rule).
// cvt_dot run [--break]      every form: targeted rows (a large product cancelled by its negative, a tiny product,
//                            in every arrangement of the slots and with C), then random rows (10^7 per FP8/BF8 form,
//                            10^6 per 16-bit form); every rule's mismatch count; the accepted rule must have 0 and a
//                            changed rule (reordered, or one rounding more) must not. --break judges with the changed
//                            rule instead, so the verdict must be FAIL.
// cvt_dot run [--form N] [--show RULE] [--seed S]  one form only; print 40 mismatches of one rule; another random
//                            sample (the rule was fitted on seed 1000; a second seed is the confirmation).
// cvt_dot mode <setting>     10^6 random rows per float form under one MODE setting (cvt_gpu.h), with the vector-ALU
//                            control; reports whether the results follow the setting's rounding direction or ignore it.
// Integer forms (item 7): v_dot4_i32_iu8 and v_dot8_i32_iu4, signed x signed (neg_lo:[1,1,0]), no clamp: the rule
// is the exact integer sum wrapped to 32 bits, checked on 10^6 random rows with extreme accumulators.
#include "arith_model.h"
#include "cvt_dot_adder.h"
#include "cvt_dot_model.h"
#include "cvt_gpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Form {
    const char * name;
    dm::Fmt in, in_b, out;
    int n;
    bool integer;
};
static const Form FORMS[] = {
    {"dot4_f32_fp8_fp8", dm::FP8, dm::FP8, dm::F32, 4, false},
    {"dot4_f32_fp8_bf8", dm::FP8, dm::BF8, dm::F32, 4, false},
    {"dot4_f32_bf8_fp8", dm::BF8, dm::FP8, dm::F32, 4, false},
    {"dot4_f32_bf8_bf8", dm::BF8, dm::BF8, dm::F32, 4, false},
    {"dot2_f32_f16", dm::F16, dm::F16, dm::F32, 2, false},
    {"dot2_f32_bf16", dm::BF16, dm::BF16, dm::F32, 2, false},
    {"dot2_f16_f16", dm::F16, dm::F16, dm::F16, 2, false},
    {"dot2_bf16_bf16", dm::BF16, dm::BF16, dm::BF16, 2, false},
    {"dot4_i32_iu8", dm::FP8, dm::FP8, dm::F32, 4, true},
    {"dot8_i32_iu4", dm::FP8, dm::FP8, dm::F32, 8, true},
};
constexpr int NFORMS = sizeof(FORMS) / sizeof(FORMS[0]);

static constexpr uint32_t HIGH_FILL = 0xA5A50000u;  // high half of a 16-bit destination before the instruction

template <int F>
__device__ __forceinline__ uint32_t dot_hw(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d = HIGH_FILL | (c & 0xFFFFu);
    if (F == 0) {
        asm volatile("v_dot4_f32_fp8_fp8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 1) {
        asm volatile("v_dot4_f32_fp8_bf8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 2) {
        asm volatile("v_dot4_f32_bf8_fp8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 3) {
        asm volatile("v_dot4_f32_bf8_bf8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 4) {
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 5) {
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 6) {
        asm volatile("v_dot2_f16_f16 %0, %1, %2, %3" : "+v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 7) {
        asm volatile("v_dot2_bf16_bf16 %0, %1, %2, %3" : "+v"(d) : "v"(a), "v"(b), "v"(c));
    } else if (F == 8) {
        asm volatile("v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    } else {
        asm volatile("v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,1,0]" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    }
    return d;
}

template <int F>
__global__ void dot_kernel(const uint32_t * a, const uint32_t * b, const uint32_t * c, uint32_t * d, int n, int write,
                           uint32_t mode, uint32_t * ctrl) {
    const uint32_t launch = cvt::mode_read();
    uint32_t old = 0;
    if (write) {
        old = cvt::mode_set(mode);
    }
    if (blockIdx.x == 0) {
        uint32_t cc[cvt::NCTRL];
        cvt::control(cc);
        if (threadIdx.x == 0) {
            for (int i = 0; i < cvt::NCTRL; ++i) {
                ctrl[i] = cc[i];
            }
            ctrl[cvt::NCTRL] = launch;
        }
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        d[i] = dot_hw<F>(a[i], b[i], c[i]);
    }
    if (write) {
        cvt::mode_restore(old);
    }
}

struct Rows {
    std::vector<uint32_t> a, b, c;
    std::vector<std::string> tag;  // targeted rows only
};

// Runs one form on the card; returns d and fills ctrl (NCTRL control values + launch MODE).
static std::vector<uint32_t> run_gpu(int f, const Rows & r, int write, uint32_t mode, uint32_t * ctrl_out) {
    const int n = (int) r.a.size();
    uint32_t *da, *db, *dc, *dd, *dctrl;
    CHECK(hipMalloc(&da, n * 4));
    CHECK(hipMalloc(&db, n * 4));
    CHECK(hipMalloc(&dc, n * 4));
    CHECK(hipMalloc(&dd, n * 4));
    CHECK(hipMalloc(&dctrl, 64));
    CHECK(hipMemcpy(da, r.a.data(), n * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, r.b.data(), n * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, r.c.data(), n * 4, hipMemcpyHostToDevice));
    const int blocks = std::min(4096, (n + 255) / 256);
    switch (f) {
#define L(F)                                                                     \
    case F:                                                                      \
        dot_kernel<F><<<blocks, 256>>>(da, db, dc, dd, n, write, mode, dctrl); \
        break;
        L(0) L(1) L(2) L(3) L(4) L(5) L(6) L(7) L(8) L(9)
#undef L
    }
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> d(n);
    CHECK(hipMemcpy(d.data(), dd, n * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(ctrl_out, dctrl, (cvt::NCTRL + 1) * 4, hipMemcpyDeviceToHost));
    for (uint32_t * p : {da, db, dc, dd, dctrl}) {
        CHECK(hipFree(p));
    }
    return d;
}

static bool is_nan_out(dm::Fmt out, uint32_t v) {
    const dm::FI i = dm::fi(out);
    const uint32_t mask = (1u << (i.eb + i.mb)) - 1;
    return (v & mask) > (((1u << i.eb) - 1) << i.mb);
}

static uint32_t out_bits(const Form & fm, uint32_t d) { return dm::width(fm.out) == 16 ? d & 0xFFFFu : d; }

// Rules beyond cvt_dot_model.h: lane 5's model of the matrix unit's adder (src/study/arith_model.h, measured
// parameters) applied to one instruction: the products occupy the first slots of one adder step, the other slots are
// zero. ADDER_SPLIT (the changed rule): the same adder with the products spread over two steps, so the sum is rounded
// to FP32 (or the 16-bit format) once more in the middle. ADDER_MIN: cvt_dot_adder.h, one step of the same adder
// with the anchor never below -126 (unit never finer than 2^-152).
constexpr int ADDER = dm::NRULE, ADDER_SPLIT = dm::NRULE + 1, ADDER_MIN = dm::NRULE + 2, NFLOATRULE = dm::NRULE + 3;

static am::Fmt am_fmt(dm::Fmt f) {
    const am::Fmt t[] = {am::E4M3, am::E5M2, am::F16, am::BF16, am::F32};
    return t[f];
}

static uint32_t adder_model(const Form & fm, uint32_t a, uint32_t b, uint32_t c, bool split) {
    int form;
    if (fm.in == dm::F16) {
        form = am::F_F16;
    } else if (fm.in == dm::BF16) {
        form = am::F_BF16;
    } else {
        form = fm.in == dm::FP8 ? (fm.in_b == dm::FP8 ? am::F_FF : am::F_FB) : (fm.in_b == dm::FP8 ? am::F_BF : am::F_BB);
    }
    uint32_t A[16] = {}, B[16] = {};
    const int w = 32 / fm.n;
    const uint32_t mask = (1u << w) - 1;
    for (int k = 0; k < fm.n; ++k) {
        // slots 0..n-1 are in the first step of every form; the split rule moves the second half to slot 8 on,
        // which is in a later step for both the 8-bit (steps of 8) and the 16-bit (steps {0-3}, {8-11}, ...) forms
        const int slot = split && k >= fm.n / 2 ? 8 + k - fm.n / 2 : k;
        A[slot] = (a >> (w * k)) & mask;
        B[slot] = (b >> (w * k)) & mask;
    }
    if (fm.out == dm::F32) {
        return am::model_f32(form, A, B, c, am::measured_params(form));
    }
    am::Params P = am::acc16_params(form, 3);
    P.rtz16 = fm.out == dm::BF16 ? 2 : 0;
    return am::model_acc16(form, A, B, c & 0xFFFFu, P);
}

static bool g_sat16 = false;  // the MODE run under FP16_OVFL

static uint32_t model(const Form & fm, int rule, uint32_t a, uint32_t b, uint32_t c, int rnd) {
    if (fm.integer) {
        int64_t s = (int32_t) c;
        const int w = 32 / fm.n;
        for (int k = 0; k < fm.n; ++k) {
            const int sh = 32 - w;
            const int32_t x = ((int32_t) (a << (sh - w * k))) >> sh, y = ((int32_t) (b << (sh - w * k))) >> sh;
            s += (int64_t) x * y;
        }
        // rule 0 wraps to 32 bits; rule 1 (the changed rule) saturates
        if (rule == 1) {
            s = std::max<int64_t>(INT32_MIN, std::min<int64_t>(INT32_MAX, s));
        }
        return (uint32_t) s;
    }
    if (rule == ADDER_MIN) {
        return da::dot(am_fmt(fm.in), am_fmt(fm.in_b), fm.n, am_fmt(fm.out), a, b, c, -126, g_sat16);
    }
    if (rule >= dm::NRULE) {
        return adder_model(fm, a, b, c, rule == ADDER_SPLIT);
    }
    return dm::eval((dm::Rule) rule, fm.in, fm.n, fm.out, a, b, c, rnd, fm.in_b);
}

static bool same(const Form & fm, uint32_t hw, uint32_t m) {
    if (fm.integer) {
        return hw == m;
    }
    return hw == m || (is_nan_out(fm.out, hw) && is_nan_out(fm.out, m));
}

// Mismatches of every rule over rows [from, n), 8 threads.
static std::vector<long> count_rules(const Form & fm, const Rows & r, const std::vector<uint32_t> & d, int nrule,
                                     int rnd, size_t from = 0) {
    std::vector<std::atomic<long>> cnt(nrule);
    std::vector<std::thread> th;
    const size_t n = r.a.size();
    for (int t = 0; t < 8; ++t) {
        th.emplace_back([&, t] {
            std::vector<long> loc(nrule, 0);
            for (size_t i = from + t; i < n; i += 8) {
                const uint32_t hw = out_bits(fm, d[i]);
                for (int k = 0; k < nrule; ++k) {
                    loc[k] += !same(fm, hw, model(fm, k, r.a[i], r.b[i], r.c[i], rnd));
                }
            }
            for (int k = 0; k < nrule; ++k) {
                cnt[k] += loc[k];
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
    std::vector<long> out(nrule);
    for (int k = 0; k < nrule; ++k) {
        out[k] = cnt[k];
    }
    return out;
}

// ---------------------------------------------------------------- inputs
static uint32_t rand_elem(dm::Fmt f, std::mt19937_64 & g) {
    const int w = dm::width(f);
    const uint32_t mask = (1u << w) - 1;
    const uint32_t u = (uint32_t) g();
    if (w == 8) {
        return u & mask;
    }
    const dm::FI i = dm::fi(f);
    const int kind = (int) (g() % 100);
    if (kind < 60) {
        // ordinary values around 1: exponent within +-12 of the bias
        const uint32_t e = (uint32_t) (i.bias - 12 + (int) (g() % 25));
        return ((u & 1) << (i.eb + i.mb)) | (e << i.mb) | ((u >> 1) & ((1u << i.mb) - 1));
    }
    if (kind < 72) {
        // near the bottom of the range: exponent field 0..12 (subnormals included)
        return ((u & 1) << (i.eb + i.mb)) | ((uint32_t) (g() % 13) << i.mb) | ((u >> 1) & ((1u << i.mb) - 1));
    }
    if (kind < 88) {
        return u & mask;
    }
    const uint32_t sp[6] = {0u, 1u << (w - 1), 1u, ((1u << i.eb) - 1) << i.mb,
                            (((1u << i.eb) - 2) << i.mb) | ((1u << i.mb) - 1), ((1u << i.eb) - 1) << i.mb | 1u};
    return sp[g() % 6] | ((u & 1) << (w - 1));
}

static double val(dm::Fmt f, uint32_t bits) {
    const dm::Term t = dm::dec(f, bits);
    if (t.nan) {
        return NAN;
    }
    if (t.inf) {
        return t.neg ? -INFINITY : INFINITY;
    }
    const double v = std::ldexp((double) t.m, t.e);
    return t.neg ? -v : v;
}

static uint32_t to_out(dm::Fmt out, double v) {
    if (out == dm::F32) {
        const float f = (float) v;
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return u;
    }
    // nearest 16-bit value via the exact rounding of the model
    const float f = (float) v;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return dm::sum_round({dm::dec(dm::F32, u)}, out, 0);
}

static void add_random(const Form & fm, Rows & r, size_t n, uint64_t seed) {
    std::mt19937_64 g(seed);
    const int w = 32 / fm.n;
    for (size_t row = 0; row < n; ++row) {
        uint32_t a = 0, b = 0;
        if (fm.integer) {
            a = (uint32_t) g();
            b = (uint32_t) g();
        } else {
            for (int k = 0; k < fm.n; ++k) {
                a |= rand_elem(fm.in, g) << (w * k);
                b |= rand_elem(fm.in_b, g) << (w * k);
            }
            if (g() % 10 < 3) {
                // cancellation: slot k+1 = -(slot k) with the same b
                const int k = (int) (g() % (fm.n - 1));
                const uint32_t m = (1u << w) - 1;
                const uint32_t ak = (a >> (w * k)) & m, bk = (b >> (w * k)) & m;
                const uint32_t sign = (1u << (w - 1));
                const uint32_t ak1 = ak ^ sign;
                a = (a & ~(m << (w * (k + 1)))) | (ak1 << (w * (k + 1)));
                b = (b & ~(m << (w * (k + 1)))) | (bk << (w * (k + 1)));
            }
        }
        uint32_t c;
        const int kind = (int) (g() % 100);
        if (fm.integer) {
            const uint32_t sp[4] = {0x7FFFFFFFu, 0x80000000u, 0x7FFFFF00u, 0x80000100u};
            c = kind < 50 ? (uint32_t) g() : sp[g() % 4];
        } else {
            // sum of the products in double (not exact in general, which is fine for building inputs)
            double s = 0;
            for (int k = 0; k < fm.n; ++k) {
                const uint32_t m = (1u << w) - 1;
                s += val(fm.in, (a >> (w * k)) & m) * val(fm.in_b, (b >> (w * k)) & m);
            }
            if (kind < 20) {
                c = (g() & 1) ? (dm::width(fm.out) == 32 ? 0x80000000u : 0x8000u) : 0u;
            } else if (kind < 60 && std::isfinite(s) && s != 0) {
                // a value near the products' magnitude
                c = to_out(fm.out, std::ldexp(s, (int) (g() % 13) - 6) * ((g() & 1) ? -1 : 1) *
                                       (1 + (double) (g() % 1000) / 1000));
            } else if (kind < 75 && std::isfinite(s)) {
                c = to_out(fm.out, -s);  // cancels the products up to rounding
            } else if (kind < 90) {
                c = dm::width(fm.out) == 32 ? (uint32_t) g() : rand_elem(fm.out, g);
            } else {
                const uint32_t sp32[6] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F7FFFFFu, 0x00000001u, 0x00800000u};
                const uint32_t sp16[6] = {0x7C00u, 0xFC00u, 0x7E00u, 0x7BFFu, 0x0001u, 0x0400u};
                const uint32_t spb[6] = {0x7F80u, 0xFF80u, 0x7FC0u, 0x7F7Fu, 0x0001u, 0x0080u};
                c = fm.out == dm::F32 ? sp32[g() % 6] : (fm.out == dm::F16 ? sp16[g() % 6] : spb[g() % 6]);
            }
        }
        r.a.push_back(a);
        r.b.push_back(b);
        r.c.push_back(c);
    }
}

// Targeted rows: a big product B, its negative, a tiny product t, zeros; every arrangement of the slots, with C
// given as 0, +B, -B or t. Which rule survives tells the order and the number of roundings.
static void add_targeted(const Form & fm, Rows & r) {
    if (fm.integer) {
        return;
    }
    struct BT {
        uint32_t big, tiny;
    };
    auto bt = [](dm::Fmt in, dm::Fmt out) -> BT {
        if (out == dm::F16) {
            return {0x4800u, 0x2C00u};  // 8 and 2^-4: products 64 and 2^-8
        }
        if (out == dm::BF16) {
            return {0x4200u, 0x3C00u};  // 32 and 2^-7
        }
        switch (in) {
        case dm::FP8: return {0x7Eu, 0x01u};
        case dm::BF8: return {0x7Bu, 0x01u};
        case dm::F16: return {0x7BFFu, 0x0001u};
        default: return {0x4700u, 0x3300u};
        }
    };
    const BT A = bt(fm.in, fm.out), B = bt(fm.in_b, fm.out);
    const int w = 32 / fm.n;
    const uint32_t sa = 1u << (w - 1);
    // the products as terms: big = A.big * B.big, tiny = A.tiny * B.tiny
    const uint32_t bigp = dm::sum_round({dm::mul(dm::dec(fm.in, A.big), dm::dec(fm.in_b, B.big))}, fm.out, 0);
    const uint32_t tinyp = dm::sum_round({dm::mul(dm::dec(fm.in, A.tiny), dm::dec(fm.in_b, B.tiny))}, fm.out, 0);
    const uint32_t osign = dm::width(fm.out) == 32 ? 0x80000000u : 0x8000u;
    // element pairs: 0 zero, 1 +big, 2 -big, 3 tiny
    auto pair_of = [&](int kind, uint32_t & ea, uint32_t & eb) {
        ea = kind == 0 ? 0 : (kind == 3 ? A.tiny : (kind == 2 ? A.big | sa : A.big));
        eb = kind == 0 ? 0 : (kind == 3 ? B.tiny : B.big);
    };
    struct Pat {
        const char * name;
        int c;  // 0: zero, 1: +big, 2: -big, 3: tiny
        std::vector<int> slots;
    };
    std::vector<Pat> pats;
    if (fm.n == 4) {
        pats = {{"C=0 {t,B,-B,0}", 0, {3, 1, 2, 0}}, {"C=B {-B,t,0,0}", 1, {2, 3, 0, 0}},
                {"C=t {B,-B,0,0}", 3, {1, 2, 0, 0}}, {"C=-B {B,t,t,0}", 2, {1, 3, 3, 0}}};
    } else {
        pats = {{"C=B {-B,t}", 1, {2, 3}}, {"C=t {B,-B}", 3, {1, 2}}, {"C=-B {B,t}", 2, {1, 3}}, {"C=t {t,0}", 3, {3, 0}}};
    }
    for (const Pat & p : pats) {
        std::vector<int> s = p.slots;
        std::sort(s.begin(), s.end());
        do {
            uint32_t a = 0, b = 0;
            std::string tag = std::string(p.name) + " as [";
            for (int k = 0; k < fm.n; ++k) {
                uint32_t ea, eb;
                pair_of(s[k], ea, eb);
                a |= ea << (w * k);
                b |= eb << (w * k);
                tag += s[k] == 2 ? std::string("-B") : std::string(1, "0B-t"[s[k]]);
                tag += k + 1 < fm.n ? "," : "]";
            }
            const uint32_t c = p.c == 0 ? 0 : (p.c == 1 ? bigp : (p.c == 2 ? bigp ^ osign : tinyp));
            r.a.push_back(a);
            r.b.push_back(b);
            r.c.push_back(c);
            r.tag.push_back(tag);
        } while (std::next_permutation(s.begin(), s.end()));
    }
}

// ---------------------------------------------------------------- modes
// only_form >= 0: run that form only; show_rule >= 0: print the first mismatches of that rule instead of the judged
// one (exploration).
static int run_forms(bool brk, int only_form = -1, int show_rule = -1, uint64_t seed = 1000) {
    bool ok = true;
    for (int f = 0; f < NFORMS; ++f) {
        if (only_form >= 0 && f != only_form) {
            continue;
        }
        const Form & fm = FORMS[f];
        Rows r;
        add_targeted(fm, r);
        const size_t ntarget = r.a.size();
        const size_t nrand = (fm.in == dm::FP8 || fm.in == dm::BF8) && !fm.integer ? 10000000 : 1000000;
        add_random(fm, r, nrand, seed + f);
        uint32_t ctrl[cvt::NCTRL + 1];
        const std::vector<uint32_t> d = run_gpu(f, r, 0, 0, ctrl);
        const int nrule = fm.integer ? 2 : NFLOATRULE;
        const char * iname[2] = {"wrap", "saturate"};
        auto rname = [&](int k) {
            return fm.integer ? iname[k]
                              : (k == ADDER ? "lane-5 adder"
                                            : (k == ADDER_SPLIT ? "adder, 2 steps"
                                                                : (k == ADDER_MIN ? "adder, anchor >= -126" : dm::rule_name[k])));
        };
        // targeted fingerprint
        if (ntarget) {
            std::printf("TARGET %s: %zu rows; per rule, rows that agree:", fm.name, ntarget);
            Rows t{std::vector<uint32_t>(r.a.begin(), r.a.begin() + ntarget),
                   std::vector<uint32_t>(r.b.begin(), r.b.begin() + ntarget),
                   std::vector<uint32_t>(r.c.begin(), r.c.begin() + ntarget), {}};
            const std::vector<long> tm = count_rules(fm, t, d, nrule, 0);
            for (int k = 0; k < nrule; ++k) {
                std::printf(" [%s] %ld", rname(k), (long) ntarget - tm[k]);
            }
            std::printf("\n");
            for (size_t i = 0; i < ntarget && i < 6; ++i) {
                std::printf("  %-34s hardware 0x%08x; rules:", r.tag[i].c_str(), out_bits(fm, d[i]));
                for (int k = 0; k < nrule; ++k) {
                    std::printf(" 0x%08x", model(fm, k, r.a[i], r.b[i], r.c[i], 0));
                }
                std::printf("\n");
            }
        }
        const std::vector<long> m = count_rules(fm, r, d, nrule, 0);
        int best = 0;
        // ties go to the later rule: ADDER_MIN (the full rule) over lane 5's model where both have 0
        for (int k = 1; k < nrule; ++k) {
            if (m[k] <= m[best]) {
                best = k;
            }
        }
        // the changed rule: one rounding too many (when the accepted rule rounds once) or the order reversed
        const int changed = fm.integer ? 1
                                       : (best >= dm::NRULE ? ADDER_SPLIT : (best == dm::EXACT ? dm::PROD_C : dm::SEQ_REV));
        const int judge = brk ? changed : best;
        long high_changed = 0;
        if (dm::width(fm.out) == 16) {
            for (uint32_t v : d) {
                high_changed += (v & 0xFFFF0000u) != HIGH_FILL;
            }
        }
        std::printf("RESULT %s: %zu rows;", fm.name, r.a.size());
        for (int k = 0; k < nrule; ++k) {
            std::printf(" [%s] %ld", rname(k), m[k]);
        }
        std::printf("%s\n", dm::width(fm.out) == 16
                                ? (" (destination high half " + std::string(high_changed ? "CHANGED" : "kept") + ")").c_str()
                                : "");
        // first mismatches of the judged rule
        int shown = 0;
        const int show = show_rule >= 0 ? show_rule : judge;
        for (size_t i = 0; i < r.a.size() && shown < (show_rule >= 0 ? 40 : 4); ++i) {
            const uint32_t mv = model(fm, show, r.a[i], r.b[i], r.c[i], 0);
            if (!same(fm, out_bits(fm, d[i]), mv)) {
                std::printf("  mismatch a=0x%08x b=0x%08x c=0x%08x hardware 0x%08x rule 0x%08x\n", r.a[i], r.b[i],
                            r.c[i], out_bits(fm, d[i]), mv);
                ++shown;
            }
        }
        const bool fok = m[judge] == 0 && m[changed] > 0 && high_changed == 0;
        char buf[300];
        std::snprintf(buf, sizeof buf, "%zu rows: rule '%s' %ld mismatches; changed rule '%s' %ld%s", r.a.size(),
                      rname(judge), m[judge], rname(changed), m[changed], brk ? " (--break)" : "");
        verdict(("cvt_dot_" + std::string(fm.name)).c_str(), fok ? "PASS" : "FAIL", buf);
        ok = ok && fok;
    }
    return ok ? 0 : 1;
}

static int run_mode(int si) {
    const cvt::Setting & s = cvt::SETTINGS[si];
    const int rnd = s.write ? (int) (s.mode & 3) : 0;
    bool ok = true;
    long nchanged = 0;
    for (int f = 0; f < NFORMS; ++f) {
        const Form & fm = FORMS[f];
        if (fm.integer) {
            continue;
        }
        Rows r;
        add_random(fm, r, 1000000, 5000 + f);
        uint32_t ctrl[cvt::NCTRL + 1];
        const std::vector<uint32_t> d = run_gpu(f, r, s.write, s.mode, ctrl);
        std::string why;
        const bool cok = cvt::control_ok(s, ctrl, ctrl[cvt::NCTRL], why);
        // FP16_OVFL: the rule is evaluated with and without the 16-bit saturation
        const bool ovfl = s.write && ((s.mode >> 23) & 1);
        g_sat16 = false;
        const long m_nosat = ovfl ? count_rules(fm, r, d, NFLOATRULE, 0)[ADDER_MIN] : 0;
        g_sat16 = ovfl;
        // the accepted rule (from `run`) is the adder (ADDER_MIN) for every float form; it has no rounding-direction
        // input, so a MODE effect shows as mismatches; the one-rounding rule in the setting's direction is printed
        // for comparison
        const std::vector<long> mm = count_rules(fm, r, d, NFLOATRULE, 0);
        const long m_rne = mm[ADDER_MIN];
        const long m_dir = rnd ? count_rules(fm, r, d, 1, rnd)[0] : mm[dm::EXACT];
        // denormal handling: rows whose result or inputs are subnormal are where flushing would show
        std::printf("RESULT mode %s %s: mismatches against the adder rule %ld%s; one rounding in direction %d %ld; "
                    "control %s %s\n",
                    s.name, fm.name, m_rne,
                    ovfl ? (" (with FP16 saturation; without it " + std::to_string(m_nosat) + ")").c_str() : "", rnd,
                    m_dir, cok ? "ok" : "FAILED", why.c_str());
        g_sat16 = false;
        ok = ok && cok;
        nchanged += m_rne;
    }
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "%ld of 8 x 10^6 results differ from the adder rule under MODE %s (rule without rounding or denormal "
                  "input; FP16 saturation applied when FP16_OVFL is set)%s",
                  nchanged, s.name, ok ? "" : "; MODE control FAILED");
    // a result that differs from the rule under a changed MODE fails the job: the claim is "MODE changes nothing"
    const bool pass = ok && nchanged == 0;
    verdict(("cvt_dot_mode_" + std::string(s.name)).c_str(), pass ? "PASS" : "FAIL", buf);
    return pass ? 0 : 1;
}

// CPU self-test of the rules against independent definitions.
static int selftest(bool brk) {
    std::mt19937_64 g(99);
    long bad_seq = 0, bad_exact = 0, n = 0;
    for (int it = 0; it < 2000000; ++it) {
        uint32_t a = 0, b = 0;
        for (int k = 0; k < 4; ++k) {
            a |= (uint32_t) (g() & 0xFF) << (8 * k);
            b |= (uint32_t) (g() & 0xFF) << (8 * k);
        }
        // C within 2^-10 .. 2^18 so that the exact sum fits in a double
        const float cf = std::ldexp((float) (1 + (g() % 8388608)) / 8388608.0f, (int) (g() % 28) - 10) *
                         ((g() & 1) ? -1.0f : 1.0f);
        uint32_t c;
        std::memcpy(&c, &cf, 4);
        float acc = cf;
        double exact = cf;
        bool any_nan = false;
        for (int k = 0; k < 4; ++k) {
            const double x = val(dm::FP8, (a >> (8 * k)) & 0xFF), y = val(dm::FP8, (b >> (8 * k)) & 0xFF);
            any_nan |= std::isnan(x) || std::isnan(y);
            acc = std::fmaf((float) x, (float) y, acc);
            exact += x * y;
        }
        if (any_nan) {
            continue;
        }
        ++n;
        const float ef = (float) exact;
        uint32_t ua, ue;
        std::memcpy(&ua, &acc, 4);
        std::memcpy(&ue, &ef, 4);
        const uint32_t ms = dm::eval(brk ? dm::SEQ_REV : dm::SEQ, dm::FP8, 4, dm::F32, a, b, c);
        const uint32_t me = dm::eval(dm::EXACT, dm::FP8, 4, dm::F32, a, b, c);
        bad_seq += ms != ua && !(acc == 0 && ms == 0x80000000u && ua == 0);
        bad_exact += me != ue && !(ef == 0 && ((me | ue) & 0x7FFFFFFFu) == 0);
    }
    // F16 inputs: sequential rule against fmaf (products of FP16 are exact in FP32)
    long bad16 = 0, n16 = 0;
    for (int it = 0; it < 1000000; ++it) {
        const uint32_t a = rand_elem(dm::F16, g) | rand_elem(dm::F16, g) << 16;
        const uint32_t b = rand_elem(dm::F16, g) | rand_elem(dm::F16, g) << 16;
        const uint32_t c = (uint32_t) g();
        float acc;
        std::memcpy(&acc, &c, 4);
        for (int k = 0; k < 2; ++k) {
            acc = std::fmaf((float) val(dm::F16, (a >> (16 * k)) & 0xFFFF), (float) val(dm::F16, (b >> (16 * k)) & 0xFFFF),
                            acc);
        }
        uint32_t ua;
        std::memcpy(&ua, &acc, 4);
        const uint32_t ms = dm::eval(dm::SEQ, dm::F16, 2, dm::F32, a, b, c);
        float mf;
        std::memcpy(&mf, &ms, 4);
        ++n16;
        bad16 += !(ms == ua || (std::isnan(mf) && std::isnan(acc)) || (mf == 0 && acc == 0 && ms == ua));
    }
    std::printf("RESULT selftest: FP8 sequential vs fmaf %ld of %ld, FP8 one rounding vs double %ld of %ld, F16 "
                "sequential vs fmaf %ld of %ld\n",
                bad_seq, n, bad_exact, n, bad16, n16);
    const bool ok = bad_seq == 0 && bad_exact == 0 && bad16 == 0;
    char buf[200];
    std::snprintf(buf, sizeof buf, "rules against fmaf and double: %ld, %ld, %ld mismatches%s", bad_seq, bad_exact, bad16,
                  brk ? " (--break: sequential rule reversed)" : "");
    verdict("cvt_dot_selftest", ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const std::string what = argc > 1 ? argv[1] : "run";
    const bool brk = argc > 2 && std::strcmp(argv[argc - 1], "--break") == 0;
    if (what == "selftest") {
        return selftest(brk);
    }
    require_arch();
    if (what == "run") {
        int only = -1, show = -1;
        uint64_t seed = 1000;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--form") == 0) {
                only = std::atoi(argv[i + 1]);
            }
            if (std::strcmp(argv[i], "--show") == 0) {
                show = std::atoi(argv[i + 1]);
            }
            if (std::strcmp(argv[i], "--seed") == 0) {
                seed = std::strtoull(argv[i + 1], nullptr, 10);
            }
        }
        return run_forms(brk, only, show, seed);
    }
    if (what == "mode" && argc > 2) {
        const int si = cvt::setting_index(argv[2]);
        if (si >= 0) {
            return run_mode(si);
        }
    }
    std::fprintf(stderr, "usage: cvt_dot selftest|run [--break] | mode <setting>\n");
    return 2;
}
