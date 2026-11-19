// Lane 8: which scalar-math operations compile to slow sequences on gfx1201, and which take data-dependent time.
// Method: one wave of 32 lanes (alone on its SIMD). Each kernel runs n iterations of  x = op(dep(x, a), k)  where dep()
// is one v_and_or_b32 that makes the operand depend on the previous result but returns the fixed input a, so the
// operand stays in one chosen data class for the whole loop and the compiler cannot hoist or fold the op (a and k are
// loaded from memory). The timed cost of one op is (cycles(n=512) - cycles(n=256)) / 256; k_dep (the chain with no op)
// is timed too so the cost of the chain link itself can be subtracted. Because dep() resets the operand, the result
// after any n >= 1 is exactly op(a, k), so the same kernels also give the correctness check against the CPU.
// The integer divisors pass through dep() as well, otherwise the compiler hoists the divisor's reciprocal out of the loop
// and the timed loop would miss the part of the division that depends on the divisor alone.
// Usage: haz_slow [all|break]. "break" points the fdiv check at a wrong reference (a*b) and must fail.
#include "harness.h"

#include <cmath>
#include <functional>
#include <random>
#include <string>

struct Row {
    uint32_t a, ahi, k, khi;
};
using KernFn = void (*)(const Row *, uint32_t *, uint64_t *, int);

__device__ __forceinline__ uint32_t dep(uint32_t x, uint32_t a) {
    uint32_t r;
    asm volatile("v_and_or_b32 %0, %1, 0, %2" : "=v"(r) : "v"(x), "v"(a));
    return r;
}
__device__ __forceinline__ float F(uint32_t u) { return __uint_as_float(u); }
__device__ __forceinline__ uint32_t U(float f) { return __float_as_uint(f); }
__device__ __forceinline__ uint64_t Q(uint32_t hi, uint32_t lo) { return ((uint64_t) hi << 32) | lo; }
__device__ __forceinline__ uint32_t fold(uint64_t p) { return (uint32_t) p ^ (uint32_t) (p >> 32); }

#define KERN(NAME, EXPR)                                                                      \
    __global__ void NAME(const Row * in, uint32_t * res, uint64_t * cyc, int n) {             \
        const Row r = in[threadIdx.x];                                                        \
        const uint32_t A = r.a, AH = r.ahi, K = r.k, KH = r.khi;                              \
        (void) AH;                                                                            \
        (void) K;                                                                             \
        (void) KH;                                                                            \
        uint32_t x = 0;                                                                       \
        const uint64_t t0 = wave_cycles();                                                    \
        _Pragma("unroll 1") for (int i = 0; i < n; ++i) {                                     \
            const uint32_t D = dep(x, A);                                                     \
            x = (EXPR);                                                                       \
        }                                                                                     \
        asm volatile("" : "+v"(x));                                                           \
        const uint64_t t1 = wave_cycles();                                                    \
        res[threadIdx.x] = x;                                                                 \
        if (threadIdx.x == 0) {                                                               \
            cyc[0] = cycle_diff(t0, t1);                                                      \
        }                                                                                     \
    }

KERN(k_dep, D)
KERN(k_fdiv, U(F(D) / F(K)))
KERN(k_frcp, U(1.0f / F(D)))
KERN(k_fdividef, U(__fdividef(F(D), F(K))))
KERN(k_sqrt, U(sqrtf(F(D))))
KERN(k_exp, U(expf(F(D))))
KERN(k_expfast, U(__expf(F(D))))
KERN(k_s32div, (uint32_t) ((int) D / (int) dep(x, K)))
KERN(k_s32rem, (uint32_t) ((int) D % (int) dep(x, K)))
KERN(k_u32div, D / dep(x, K))
KERN(k_u32rem, D % dep(x, K))
KERN(k_mul64, fold(Q(AH, D) * Q(KH, K)))
KERN(k_u64div, fold(Q(AH, D) / Q(KH, dep(x, K))))
KERN(k_i2f, U((float) (int) D))
KERN(k_u2f, U((float) D))
KERN(k_f2i, (uint32_t) (int) F(D))
KERN(k_f2u, (uint32_t) F(D))

// ---------------------------------------------------------------- host side

static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static float flt(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static Row fr(float a, float k) { return {bits(a), 0, bits(k), 0}; }
static Row ir(uint32_t a, uint32_t k) { return {a, 0, k, 0}; }
static Row lr(uint64_t a, uint64_t k) { return {(uint32_t) a, (uint32_t) (a >> 32), (uint32_t) k, (uint32_t) (k >> 32)}; }

struct Cls {
    const char * name;
    Row r;
};

static Row * d_in;
static uint32_t * d_res;
static uint64_t * d_cyc;

static void upload(const std::vector<Row> & v) {
    CHECK(hipMemcpy(d_in, v.data(), 32 * sizeof(Row), hipMemcpyHostToDevice));
}

static double launch_cycles(KernFn k, int n) {
    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, d_in, d_res, d_cyc, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint64_t c = 0;
    CHECK(hipMemcpy(&c, d_cyc, 8, hipMemcpyDeviceToHost));
    return (double) c;
}

constexpr int REPS = 5;

static Stats time_class(KernFn k, const Row & r) {
    upload(std::vector<Row>(32, r));
    launch_cycles(k, 256);  // warms the instruction cache
    std::vector<double> v;
    for (int i = 0; i < REPS; ++i) {
        const double c1 = launch_cycles(k, 256);
        const double c2 = launch_cycles(k, 512);
        v.push_back((c2 - c1) / 256.0);
    }
    return stats_of(v);
}

static void print_time(const char * op, const char * cls, const Stats & s) {
    char b[200];
    std::snprintf(b, sizeof b, "op=%s class=%s cycles=%.2f range=%.2f-%.2f", op, cls, s.median, s.min, s.max);
    verdict("haz_slow_time", "INFO", b);
}

// Per-lane results of kernel k for up to 32 rows (one launch, n = 1).
static std::vector<uint32_t> run_rows(KernFn k, const std::vector<Row> & rows) {
    std::vector<Row> v(rows);
    v.resize(32, rows.back());
    upload(v);
    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, d_in, d_res, d_cyc, 1);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(32);
    CHECK(hipMemcpy(out.data(), d_res, 32 * 4, hipMemcpyDeviceToHost));
    out.resize(rows.size());
    return out;
}

// Distance in float steps; NaN against NaN is 0, NaN against a number is huge.
static int64_t ulp_dist(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return (std::isnan(a) && std::isnan(b)) ? 0 : (int64_t) 1 << 40;
    }
    auto ord = [](float f) {
        int32_t i;
        std::memcpy(&i, &f, 4);
        return (int64_t) (i < 0 ? (int32_t) 0x80000000 - i : i);
    };
    return std::llabs(ord(a) - ord(b));
}

// A check returns 0 = pass, 1 = fail, 2 = deviation in a range where the op is allowed to differ (reported only).
struct Op {
    const char * name;
    KernFn k;
    std::vector<Cls> classes;
    std::vector<Row> special;
    std::function<Row(std::mt19937 &)> gen;
    std::function<int(const Row &, uint32_t, std::string &, int64_t &)> check;
};

static bool g_break = false;

static int run_check(const Op & op, std::mt19937 & rng, bool & any_fail) {
    std::vector<Row> rows = op.special;
    for (int i = 0; i < 4096; ++i) {
        rows.push_back(op.gen(rng));
    }
    int64_t worst = 0;
    int fails = 0, infos = 0;
    std::string first_fail, first_info;
    for (size_t s = 0; s < rows.size(); s += 32) {
        std::vector<Row> b(rows.begin() + s, rows.begin() + std::min(rows.size(), s + 32));
        const std::vector<uint32_t> got = run_rows(op.k, b);
        for (size_t i = 0; i < b.size(); ++i) {
            std::string note;
            int64_t u = 0;
            const int c = op.check(b[i], got[i], note, u);
            if (c == 1) {
                if (!fails) {
                    first_fail = note;
                }
                ++fails;
            } else if (c == 2) {
                if (!infos) {
                    first_info = note;
                }
                ++infos;
            }
            if (c == 0) {
                worst = std::max(worst, u);
            }
        }
    }
    char name[64], text[600];
    std::snprintf(name, sizeof name, "haz_slow_%s", op.name);
    if (fails) {
        std::snprintf(text, sizeof text, "%d of %zu mismatches; first: %s", fails, rows.size(), first_fail.c_str());
        verdict(name, "FAIL", text);
        any_fail = true;
    } else {
        std::snprintf(text, sizeof text, "%zu inputs match the CPU reference; largest error in passing range %lld ulp",
                      rows.size(), (long long) worst);
        verdict(name, "PASS", text);
    }
    if (infos) {
        std::snprintf(text, sizeof text, "op=%s %d inputs outside the checked range differ from the reference; first: %s",
                      op.name, infos, first_info.c_str());
        verdict("haz_slow_range", "INFO", text);
    }
    return fails;
}

static const float SPECF[] = {0.0f, -0.0f, 1e-40f, -1e-40f, 1.4e-45f, 1.17549435e-38f, 1.0f, -1.0f, 3.7f, 1e30f,
                              1e-30f, 3e38f, INFINITY, -INFINITY, NAN};

static float rnd_float(std::mt19937 & g, int lo_exp, int hi_exp) {
    std::uniform_real_distribution<double> m(1.0, 2.0);
    std::uniform_int_distribution<int> e(lo_exp, hi_exp);
    const double v = std::ldexp(m(g), e(g));
    return (g() & 1) ? (float) -v : (float) v;
}

static uint32_t rnd_width(std::mt19937 & g) {  // random value with a random number of significant bits
    const int w = 1 + (int) (g() % 32);
    return (uint32_t) g() >> (32 - w);
}
static uint64_t rnd_width64(std::mt19937 & g) {
    const int w = 1 + (int) (g() % 64);
    return (((uint64_t) g() << 32) | g()) >> (64 - w);
}

static std::string hexf(float f) {
    char b[48];
    std::snprintf(b, sizeof b, "%a", (double) f);
    return b;
}

static bool is_norm(float f) { return std::isnormal(f); }

static std::vector<Op> make_ops() {
    std::vector<Op> ops;
    std::vector<Row> pairs;
    for (float a : SPECF) {
        for (float b : SPECF) {
            pairs.push_back(fr(a, b));
        }
    }
    std::vector<Row> singles;
    for (float a : SPECF) {
        singles.push_back(fr(a, a));
    }
    auto gen2 = [](std::mt19937 & g) { return fr(rnd_float(g, -60, 60), rnd_float(g, -60, 60)); };

    // exact float check: bit-equal to ref (any NaN equals any NaN)
    auto exact_f = [](const char * nm, std::function<float(const Row &)> ref) {
        return [nm, ref](const Row & r, uint32_t got, std::string & note, int64_t & u) {
            const float e = ref(r), g = flt(got);
            u = ulp_dist(e, g);
            if (u == 0 && (std::isnan(e) || bits(e) == got)) {
                return 0;
            }
            note = std::string(nm) + " a=" + hexf(flt(r.a)) + " k=" + hexf(flt(r.k)) + " got=" + hexf(g) +
                   " want=" + hexf(e);
            return 1;
        };
    };
    // approximate op: error bound in the ordinary range, report-only elsewhere
    auto approx_f = [](const char * nm, std::function<double(const Row &)> ref, std::function<bool(const Row &)> inrange,
                       std::function<double(const Row &)> tol) {
        return [=](const Row & r, uint32_t got, std::string & note, int64_t & u) {
            const float e = (float) ref(r), g = flt(got);
            u = ulp_dist(e, g);
            const bool in = inrange(r);
            if (in && u <= (int64_t) tol(r)) {
                return 0;
            }
            if (!in && u == 0) {
                return 0;
            }
            note = std::string(nm) + " a=" + hexf(flt(r.a)) + " k=" + hexf(flt(r.k)) + " got=" + hexf(g) +
                   " want=" + hexf(e) + " ulp=" + std::to_string((long long) u);
            return in ? 1 : 2;
        };
    };

    ops.push_back({"fdiv", k_fdiv,
                   {{"ordinary", fr(3.7f, 1.3f)}, {"zero_num", fr(0.0f, 1.3f)}, {"denorm_num", fr(1e-40f, 1.3f)},
                    {"denorm_div", fr(3.7f, 1e-40f)}, {"denorm_both", fr(1e-40f, 3e-40f)},
                    {"nan_num", fr(NAN, 1.3f)}, {"inf_num", fr(INFINITY, 1.3f)}, {"huge_quot", fr(1e30f, 1e-30f)},
                    {"div_zero", fr(3.7f, 0.0f)}, {"div_inf", fr(3.7f, INFINITY)}, {"nan_div", fr(3.7f, NAN)}},
                   pairs, gen2,
                   exact_f("fdiv", [](const Row & r) { return g_break ? flt(r.a) * flt(r.k) : flt(r.a) / flt(r.k); })});
    ops.push_back({"frcp", k_frcp,
                   {{"ordinary", fr(1.3f, 0)}, {"denorm_div", fr(1e-40f, 0)}, {"zero", fr(0.0f, 0)},
                    {"inf", fr(INFINITY, 0)}, {"nan", fr(NAN, 0)}, {"huge_in", fr(1e30f, 0)},
                    {"tiny_in", fr(1e-30f, 0)}, {"max_in", fr(3e38f, 0)}},
                   singles, [](std::mt19937 & g) { return fr(rnd_float(g, -126, 127), 0); },
                   exact_f("frcp", [](const Row & r) { return 1.0f / flt(r.a); })});
    auto fdv_in = [](const Row & r) {
        const float a = flt(r.a), b = flt(r.k), q = a / b;
        return is_norm(a) && is_norm(b) && is_norm(q) && std::fabs(b) < 1e37f && std::fabs(q) < 1e37f;
    };
    ops.push_back({"fdividef", k_fdividef,
                   {{"ordinary", fr(3.7f, 1.3f)}, {"zero_num", fr(0.0f, 1.3f)}, {"denorm_num", fr(1e-40f, 1.3f)},
                    {"denorm_div", fr(3.7f, 1e-40f)}, {"denorm_both", fr(1e-40f, 3e-40f)},
                    {"nan_num", fr(NAN, 1.3f)}, {"inf_num", fr(INFINITY, 1.3f)}, {"huge_quot", fr(1e30f, 1e-30f)},
                    {"div_zero", fr(3.7f, 0.0f)}, {"div_inf", fr(3.7f, INFINITY)}, {"nan_div", fr(3.7f, NAN)}},
                   pairs, gen2,
                   approx_f("fdividef", [](const Row & r) { return (double) flt(r.a) / (double) flt(r.k); }, fdv_in,
                            [](const Row &) { return 2.0; })});
    ops.push_back({"sqrt", k_sqrt,
                   {{"ordinary", fr(3.7f, 0)}, {"one", fr(1.0f, 0)}, {"zero", fr(0.0f, 0)}, {"denorm", fr(1e-40f, 0)},
                    {"nan", fr(NAN, 0)}, {"inf", fr(INFINITY, 0)}, {"negative", fr(-3.7f, 0)}, {"huge", fr(1e30f, 0)}},
                   singles, [](std::mt19937 & g) { return fr(std::fabs(rnd_float(g, -126, 127)), 0); },
                   exact_f("sqrt", [](const Row & r) { return std::sqrt(flt(r.a)); })});
    std::vector<Cls> expc = {{"ordinary", fr(0.5f, 0)}, {"ordinary_large", fr(20.0f, 0)}, {"zero", fr(0.0f, 0)},
                             {"denorm", fr(1e-40f, 0)}, {"nan", fr(NAN, 0)}, {"inf", fr(INFINITY, 0)},
                             {"neg_inf", fr(-INFINITY, 0)}, {"overflow", fr(100.0f, 0)}, {"underflow", fr(-100.0f, 0)},
                             {"denorm_result", fr(-90.0f, 0)}};
    auto exp_gen = [](std::mt19937 & g) { return fr(std::uniform_real_distribution<float>(-85.0f, 88.0f)(g), 0); };
    auto exp_ref = [](const Row & r) { return std::exp((double) flt(r.a)); };
    ops.push_back({"exp", k_exp, expc, singles, exp_gen,
                   approx_f("expf", exp_ref,
                            [](const Row & r) {
                                const float a = flt(r.a);
                                return std::fabs(a) < 85.0f;
                            },
                            [](const Row &) { return 2.0; })});
    ops.push_back({"expfast", k_expfast, expc, singles,
                   [](std::mt19937 & g) { return fr(std::uniform_real_distribution<float>(-20.0f, 20.0f)(g), 0); },
                   approx_f("__expf", exp_ref,
                            [](const Row & r) {
                                const float a = flt(r.a);
                                return std::fabs(a) <= 20.0f;
                            },
                            [](const Row & r) { return 4.0 + 2.0 * std::fabs(flt(r.a)); })});

    // integer division: exact against the host; divisor never zero, INT_MIN / -1 left out
    std::vector<Cls> ic = {{"small_small", ir(7, 3)}, {"small_over_large", ir(7, 0x7fffffffu)},
                           {"large_over_small", ir(0x7ffffff1u, 3)}, {"large_large", ir(0x7ffffff1u, 0x1000001u)},
                           {"zero_num", ir(0, 3)}, {"hibit_over_small", ir(0xfffffff1u, 3)},
                           {"hibit_over_hibit", ir(0xfffffff1u, 0x80000001u)}};
    std::vector<Row> isp;
    const uint32_t IV[] = {0u, 1u, 7u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0xfffffff1u, 3u, 0x10001u, 2u};
    for (uint32_t a : IV) {
        for (uint32_t b : IV) {
            if (b) {
                isp.push_back(ir(a, b));
            }
        }
    }
    auto igen = [](std::mt19937 & g) {
        uint32_t k = rnd_width(g);
        if (!k) {
            k = 1;
        }
        return ir(rnd_width(g), k);
    };
    auto icheck = [](const char * nm, bool sgn, bool rem) {
        return [=](const Row & r, uint32_t got, std::string & note, int64_t & u) {
            u = 0;
            uint32_t want;
            if (sgn) {
                const int32_t a = (int32_t) r.a, b = (int32_t) r.k;
                want = (a == INT32_MIN && b == -1) ? got : (uint32_t) (rem ? a % b : a / b);
            } else {
                want = rem ? r.a % r.k : r.a / r.k;
            }
            if (want == got) {
                return 0;
            }
            note = std::string(nm) + " a=" + std::to_string(r.a) + " k=" + std::to_string(r.k) +
                   " got=" + std::to_string(got) + " want=" + std::to_string(want);
            return 1;
        };
    };
    ops.push_back({"s32div", k_s32div, ic, isp, igen, icheck("s32div", true, false)});
    ops.push_back({"s32rem", k_s32rem, ic, isp, igen, icheck("s32rem", true, true)});
    ops.push_back({"u32div", k_u32div, ic, isp, igen, icheck("u32div", false, false)});
    ops.push_back({"u32rem", k_u32rem, ic, isp, igen, icheck("u32rem", false, true)});

    // 64-bit: multiply and unsigned divide, checked on the folded (lo xor hi) result
    const uint64_t LV[] = {0ull, 1ull, 7ull, 3ull, 0xfffffff1ull, 0x123456789abcdef1ull, 0xfedcba9876543210ull,
                           0x8000000000000001ull, 0x10001ull, 0xffffffffffffffffull};
    std::vector<Row> lsp;
    for (uint64_t a : LV) {
        for (uint64_t b : LV) {
            if (b) {
                lsp.push_back(lr(a, b));
            }
        }
    }
    auto lgen = [](std::mt19937 & g) {
        uint64_t k = rnd_width64(g);
        if (!k) {
            k = 1;
        }
        return lr(rnd_width64(g), k);
    };
    auto lcheck = [](const char * nm, bool div) {
        return [=](const Row & r, uint32_t got, std::string & note, int64_t & u) {
            u = 0;
            const uint64_t a = ((uint64_t) r.ahi << 32) | r.a, k = ((uint64_t) r.khi << 32) | r.k;
            const uint64_t p = div ? a / k : a * k;
            const uint32_t want = (uint32_t) p ^ (uint32_t) (p >> 32);
            if (want == got) {
                return 0;
            }
            note = std::string(nm) + " a=" + std::to_string(a) + " k=" + std::to_string(k) +
                   " got_fold=" + std::to_string(got) + " want_fold=" + std::to_string(want);
            return 1;
        };
    };
    ops.push_back({"mul64", k_mul64,
                   {{"ordinary", lr(0x123456789abcdef1ull, 0x0fedcba987654321ull)}, {"zero_num", lr(0, 0x0fedcba987654321ull)},
                    {"small", lr(7, 3)}, {"one", lr(1, 1)}},
                   lsp, lgen, lcheck("mul64", false)});
    ops.push_back({"u64div", k_u64div,
                   {{"small_small", lr(7, 3)}, {"a32_over_small", lr(0xfffffff1ull, 3)},
                    {"a64_over_small", lr(0x123456789abcdef1ull, 3)}, {"a64_over_k32", lr(0x123456789abcdef1ull, 0x10001ull)},
                    {"a64_over_k64", lr(0x123456789abcdef1ull, 0x0123456789ull)},
                    {"a64_over_k64_huge", lr(0xfedcba9876543210ull, 0x8000000000000001ull)}, {"zero_num", lr(0, 3)},
                    {"a_less_k", lr(7, 0x123456789abcdef1ull)}},
                   lsp, lgen, lcheck("u64div", true)});

    // conversions
    ops.push_back({"i2f", k_i2f,
                   {{"small", ir(7, 0)}, {"large", ir(0x7fffffffu, 0)}, {"negative", ir((uint32_t) -7, 0)},
                    {"zero", ir(0, 0)}, {"int_min", ir(0x80000000u, 0)}},
                   isp, [](std::mt19937 & g) { return ir(rnd_width(g), 0); },
                   exact_f("i2f", [](const Row & r) { return (float) (int32_t) r.a; })});
    ops.push_back({"u2f", k_u2f, {{"small", ir(7, 0)}, {"large", ir(0xffffffffu, 0)}, {"zero", ir(0, 0)}}, isp,
                   [](std::mt19937 & g) { return ir(rnd_width(g), 0); },
                   exact_f("u2f", [](const Row & r) { return (float) r.a; })});
    std::vector<Cls> f2c = {{"ordinary", fr(3.7f, 0)}, {"negative", fr(-3.7f, 0)}, {"zero", fr(0.0f, 0)},
                            {"denorm", fr(1e-40f, 0)}, {"large_inrange", fr(2.0e9f, 0)}, {"saturate", fr(1e30f, 0)},
                            {"nan", fr(NAN, 0)}};
    auto cvt_check = [](const char * nm, bool sgn) {
        return [=](const Row & r, uint32_t got, std::string & note, int64_t & u) {
            u = 0;
            const float a = flt(r.a);
            const bool in = sgn ? (std::fabs(a) < 2147483648.0f) : (a > -1.0f && a < 4294967296.0f);
            uint32_t want;
            if (in) {
                want = sgn ? (uint32_t) (int32_t) a : (uint32_t) (uint64_t) (double) a;
            } else {  // saturating, NaN to 0: what the hardware conversion does; report-only because C++ leaves it undefined
                want = std::isnan(a) ? 0u
                       : sgn         ? (a > 0 ? 0x7fffffffu : 0x80000000u)
                                     : (a > 0 ? 0xffffffffu : 0u);
            }
            if (want == got) {
                return 0;
            }
            note = std::string(nm) + " a=" + hexf(a) + " got=" + std::to_string(got) + " want=" + std::to_string(want);
            return in ? 1 : 2;
        };
    };
    auto fgen = [](std::mt19937 & g) { return fr(rnd_float(g, -10, 33), 0); };
    ops.push_back({"f2i", k_f2i, f2c, singles, fgen, cvt_check("f2i", true)});
    ops.push_back({"f2u", k_f2u, f2c, singles, fgen, cvt_check("f2u", false)});
    return ops;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    std::printf("device %s %s\n", prop.name, prop.gcnArchName);
    const bool brk = argc > 1 && std::strcmp(argv[1], "break") == 0;
    CHECK(hipMalloc(&d_in, 32 * sizeof(Row)));
    CHECK(hipMalloc(&d_res, 32 * 4));
    CHECK(hipMalloc(&d_cyc, 8));
    std::mt19937 rng(12345);
    std::vector<Op> ops = make_ops();
    bool fail = false;

    if (brk) {
        g_break = true;
        run_check(ops[0], rng, fail);  // fdiv against a*b
        std::printf("break mode: exit %d\n", fail ? 1 : 0);
        return fail ? 1 : 0;
    }

    for (const Op & op : ops) {
        run_check(op, rng, fail);
    }
    print_time("dep", "chain_only", time_class(k_dep, fr(3.7f, 1.3f)));
    for (const Op & op : ops) {
        for (const Cls & c : op.classes) {
            print_time(op.name, c.name, time_class(op.k, c.r));
        }
    }
    return fail ? 1 : 0;
}
