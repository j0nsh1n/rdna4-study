// Lane 5, item 7: a worst-case search for the quantizing question. CPU only: uses the model (arith_model.h), which
// arith_check showed equal to the card, single instructions and chains through C.
//
// Question: after the card's FP32 result and the exact sum are both rounded to FP16 (or to E4M3, saturating), how
// often do the two target values differ, and by how many target steps at worst?
//
// Usage: arith_qworst [--break] [--nosub] [--k1024] [--nonfinite]
//   --nosub      (leftovers C, EQ-2) no subnormal input codes anywhere in the search (and no subnormal FP32 C)
//   --k1024      (EQ-2) chains of 64 instructions (K = 1024) instead of K = 16 and 256
//   --nonfinite  (EQ-2) no search: random cases with NaN and infinite inputs; the class of the card's result (NaN,
//                +inf, -inf, finite) must equal the class of the exact sum under IEEE rules (a NaN input, inf x 0
//                or +inf + -inf give NaN; otherwise any infinite term gives that infinity; otherwise finite)
//
// Paths: FP16 form with FP16 inputs; FP16 form with E4M3 weights widened exactly; BF16 form with BF16 inputs;
// FP8 x FP8. Sizes: (a) one instruction (K = 16, C any FP32), (b) a chain of 16 instructions (K = 256, C starts 0).
//
// Scale: arith_stats scales each path by the RMS of its results. Searched inputs have no common RMS, so the scale
// is chosen per case from the sum of magnitudes M (products and C): FP16 target so that M is in [1, 2), E4M3 target
// so that M is in [64, 128) (the same two operating points as arith_stats). Nothing saturates or overflows then, and
// cancellation shows up as results far below M, where the target steps are small relative to M.
//
// Two searches per path x K x target:
//   worst  random-restart hill climbing on the raw input codes for the largest step distance (ordered-code distance
//          of the two rounded codes), with a smooth tie-break (error in target steps) to give a slope; the best case
//          is then shrunk by zeroing products while the distance stays, and printed as hex.
//   rate   hill climbing on 7 parameters of an input generator (exponent spread, share of cancelling pairs, share of
//          zero products, mantissa bits kept, share of subnormal inputs, how C is chosen, how close to a rounding
//          midpoint C puts the sum), with common random numbers per evaluation; the best parameters are re-measured
//          on fresh samples (N given) for the reported rate with a Wilson 95 % interval.
// Chains have no free C, so the generator's C parameters do nothing there (C = 0).
//
// Verdict: the controls below pass and every reported worst case is recomputed from its printed hex (card by a
// fresh chain of the model, exact sum by a 128-bit integer sum that does not use am::Exact where it fits).
//   --break  the control's expectation is perturbed; the verdict must FAIL.
#include "arith_model.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

static bool g_break = false;
static bool g_nosub = false;  // EQ-2: no subnormal input codes
constexpr int KMAX = 1024;

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

static uint32_t bits32(float f) { return __builtin_bit_cast(uint32_t, f); }

// ---------------------------------------------------------------------------------------------------------------
// Paths and targets

struct PathDef {
    const char * name;
    const char * tag;
    int form;
    am::Fmt sa, sb;  // formats the generator and the search work in (source codes)
    bool widen;      // sa codes are E4M3, widened exactly to FP16 before the instruction
};
static const PathDef PATHS[4] = {{"FP16 x FP16", "f16", am::F_F16, am::F16, am::F16, false},
                                 {"E4M3 widened x FP16", "f16w", am::F_F16, am::E4M3, am::F16, true},
                                 {"BF16 x BF16", "bf16", am::F_BF16, am::BF16, am::BF16, false},
                                 {"E4M3 x E4M3", "fp8", am::F_FF, am::E4M3, am::E4M3, false}};

static am::Fmt tfmt(int T) { return T == 0 ? am::F16 : am::E4M3; }
static bool tsat(int T) { return T == 1; }
static const char * tname(int T) { return T == 0 ? "fp16" : "e4m3"; }

static int nbits(am::Fmt f) { return am::info(f).ebits + am::info(f).mbits + 1; }
static bool finite_code(am::Fmt f, uint32_t v) {
    const am::Dec d = am::decode(f, v);
    return !d.nan && !d.inf;
}

static bool subnormal_code(am::Fmt f, uint32_t v) {
    const am::FmtInfo fi = am::info(f);
    return ((v >> fi.mbits) & ((1u << fi.ebits) - 1)) == 0 && (v & ((1u << fi.mbits) - 1)) != 0;
}
// A code the search may use: finite, and with --nosub not subnormal.
static bool usable(am::Fmt f, uint32_t v) { return finite_code(f, v) && !(g_nosub && subnormal_code(f, v)); }

static uint32_t form_a(const PathDef & p, uint32_t a) { return p.widen ? am::convert(am::E4M3, a, am::F16) : a; }

// Ordered code: negative codes below zero, +0 and -0 equal.
static int64_t ordc(int T, uint32_t c) {
    const uint32_t sb = T == 0 ? 0x8000u : 0x80u;
    const int64_t mag = c & (sb - 1);
    return (c & sb) ? -mag : mag;
}

// FP32 bits scaled by 2^s and rounded once to the target.
static uint32_t round_scaled(uint32_t f32bits, int s, int T) {
    const am::Dec d = am::decode(am::F32, f32bits);
    return am::round_int(tfmt(T), d.neg ? -(__int128) d.m : (__int128) d.m, d.e - d.mbits + s, d.neg, tsat(T));
}

static int scale_for(double sumabs, int T) {
    int e2;
    std::frexp(sumabs, &e2);
    const int fl = e2 - 1;  // floor(log2(sumabs))
    return T == 0 ? -fl : 6 - fl;
}

// ---------------------------------------------------------------------------------------------------------------
// One case and its evaluation

struct Case {
    int K;
    uint32_t c;
    uint32_t a[KMAX], b[KMAX];
};

struct Res {
    bool ok = false;
    int64_t dist = 0;
    uint32_t hwc = 0, exc = 0;
    double smooth = 0;
    int s = 0;
    double sumabs = 0;
};

struct Term {
    bool neg;
    uint64_t m;
    int x;
};

static Term term_of(const PathDef & p, uint32_t a, uint32_t b) {
    const am::Dec da = am::decode(am::fmt_a(p.form), form_a(p, a)), db = am::decode(am::fmt_b(p.form), b);
    return {da.neg != db.neg, (uint64_t) (da.m * db.m), da.e - da.mbits + db.e - db.mbits};
}

static double sum_of_magnitudes(const PathDef & p, const Case & cs) {
    double sm = 0;
    for (int k = 0; k < cs.K; ++k) {
        const Term t = term_of(p, cs.a[k], cs.b[k]);
        sm += std::ldexp((double) t.m, t.x);
    }
    const am::Dec dc = am::decode(am::F32, cs.c);
    return sm + std::ldexp((double) dc.m, dc.e - dc.mbits);
}

static Res evaluate(const PathDef & p, const Case & cs, int T, bool smooth) {
    Res r;
    const am::Dec dc = am::decode(am::F32, cs.c);
    if (dc.nan || dc.inf) {
        return r;
    }
    const am::Params P = am::measured_params(p.form);
    uint32_t acc = cs.c;
    for (int s = 0; s < cs.K / 16; ++s) {
        uint32_t A[16], B[16];
        for (int k = 0; k < 16; ++k) {
            A[k] = form_a(p, cs.a[16 * s + k]);
            B[k] = cs.b[16 * s + k];
        }
        acc = am::model_f32(p.form, A, B, acc, P);
    }
    const am::Dec dh = am::decode(am::F32, acc);
    if (dh.nan || dh.inf) {
        return r;
    }
    r.ok = true;
    const double sm = sum_of_magnitudes(p, cs);
    r.sumabs = sm;
    if (sm == 0) {
        return r;
    }
    const int s = scale_for(sm, T);
    r.s = s;
    am::Exact ex;
    for (int k = 0; k < cs.K; ++k) {
        const Term t = term_of(p, cs.a[k], cs.b[k]);
        ex.add(t.neg, t.m, t.x + s);
    }
    ex.add(dc.neg, (uint64_t) dc.m, dc.e - dc.mbits + s);
    r.hwc = round_scaled(acc, s, T);
    r.exc = ex.round(tfmt(T), tsat(T));
    r.dist = std::llabs(ordc(T, r.hwc) - ordc(T, r.exc));
    if (smooth) {
        const am::Dec dx = am::decode(am::F32, ex.round(am::F32));
        const double exv = std::ldexp((double) (dx.neg ? -dx.m : dx.m), dx.e - dx.mbits);
        const double hwv = std::ldexp((double) (dh.neg ? -dh.m : dh.m), dh.e - dh.mbits + s);
        const uint32_t sb = T == 0 ? 0x8000u : 0x80u;
        const uint32_t mag = r.exc & (sb - 1);
        const am::Fmt tf = tfmt(T);
        double hi = am::value(tf, mag + 1), lo = am::value(tf, mag);
        if (!std::isfinite(hi) || (tf == am::E4M3 && mag + 1 > 0x7E)) {
            hi = lo;
            lo = am::value(tf, mag - 1);
        }
        const double ulp = std::fabs(hi - lo);
        const double d = std::fabs(hwv - exv) / ulp;
        r.smooth = d / (1 + d);
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Generator family

struct Par {
    int spread;  // exponent spread of the products (field offsets below the top)
    int cancel;  // share of slots (%, about) turned into a cancelling or near-cancelling partner of another slot
    int zero;    // share of zero products (%)
    int mant;    // mantissa bits kept (coarse grids make sums land on rounding midpoints)
    int subp;    // share of subnormal inputs (%)
    int cmode;   // 0: C = 0, 1: C random near the sum, 2: C puts the exact sum near a target midpoint
    int jc;      // cmode 2: the sum is put within 2^-jc of the sum of magnitudes of the midpoint
};
static const int PAR_LO[7] = {0, 0, 0, 0, 0, 0, 8}, PAR_HI[7] = {32, 100, 90, 10, 60, 2, 30};
static int & par_at(Par & q, int i) {
    int * v[7] = {&q.spread, &q.cancel, &q.zero, &q.mant, &q.subp, &q.cmode, &q.jc};
    return *v[i];
}

static Par random_par(std::mt19937_64 & rng) {
    Par q{};
    for (int i = 0; i < 7; ++i) {
        par_at(q, i) = PAR_LO[i] + (int) (rng() % (uint64_t) (PAR_HI[i] - PAR_LO[i] + 1));
    }
    return q;
}

static void mutate_par(Par & q, std::mt19937_64 & rng) {
    const int i = (int) (rng() % 7);
    const int span = PAR_HI[i] - PAR_LO[i];
    int & v = par_at(q, i);
    if (rng() % 4 == 0) {
        v = PAR_LO[i] + (int) (rng() % (uint64_t) (span + 1));
    } else {
        const int step = 1 + (int) (rng() % (uint64_t) std::max(1, span / 4));
        v += (rng() & 1) ? step : -step;
        v = std::min(PAR_HI[i], std::max(PAR_LO[i], v));
    }
}

static int top_field(am::Fmt f) { return f == am::F16 ? 15 : f == am::BF16 ? 127 : 9; }

static uint32_t make_code(am::Fmt f, std::mt19937_64 & rng, int off, int mant, int subp) {
    const am::FmtInfo fi = am::info(f);
    const int maxf = fi.e4m3 ? 15 : (1 << fi.ebits) - 2;
    int field = std::min(maxf, std::max(g_nosub ? 1 : 0, top_field(f) - off));
    if ((int) (rng() % 100) < subp && !g_nosub) {
        field = 0;
    }
    uint32_t m = (uint32_t) rng() & ((1u << fi.mbits) - 1);
    const int keep = std::min(mant, fi.mbits);
    m &= ~((1u << (fi.mbits - keep)) - 1);
    if (fi.e4m3 && field == 15 && m == 7) {
        m = 6;
    }
    const uint32_t sign = (uint32_t) (rng() & 1) << (fi.ebits + fi.mbits);
    return sign | ((uint32_t) field << fi.mbits) | m;
}

static uint32_t random_finite(am::Fmt f, std::mt19937_64 & rng) {
    for (;;) {
        const uint32_t v = (uint32_t) rng() & ((1u << nbits(f)) - 1);
        if (usable(f, v)) {
            return v;
        }
    }
}

static uint32_t sign_of(am::Fmt f) { return 1u << (nbits(f) - 1); }

static void gen(const PathDef & p, int K, const Par & q, int T, std::mt19937_64 & rng, Case & cs) {
    cs.K = K;
    cs.c = 0;
    std::fill(cs.a, cs.a + KMAX, 0u);
    std::fill(cs.b, cs.b + KMAX, 0u);
    const int sa = q.spread / 2, sb = q.spread - sa;
    for (int k = 0; k < K; ++k) {
        if ((int) (rng() % 100) < q.zero) {
            cs.a[k] = 0;
            cs.b[k] = make_code(p.sb, rng, (int) (rng() % (uint64_t) (sb + 1)), q.mant, q.subp);
        } else {
            cs.a[k] = make_code(p.sa, rng, (int) (rng() % (uint64_t) (sa + 1)), q.mant, q.subp);
            cs.b[k] = make_code(p.sb, rng, (int) (rng() % (uint64_t) (sb + 1)), q.mant, q.subp);
        }
    }
    for (int k = 0; k < K; ++k) {
        if ((int) (rng() % 200) < q.cancel) {
            const int j = (int) (rng() % (uint64_t) K);
            if (j == k) {
                continue;
            }
            cs.a[j] = cs.a[k] ^ sign_of(p.sa);
            cs.b[j] = cs.b[k];
            if (rng() % 5 < 2) {
                cs.a[j] ^= 1u << (rng() % (uint64_t) am::info(p.sa).mbits);  // near cancel: one mantissa bit off
            }
        }
    }
    if (g_nosub) {
        for (int k = 0; k < K; ++k) {  // a near-cancel bit flip of a zero code makes a subnormal: back to zero
            cs.a[k] = subnormal_code(p.sa, cs.a[k]) ? cs.a[k] & sign_of(p.sa) : cs.a[k];
            cs.b[k] = subnormal_code(p.sb, cs.b[k]) ? cs.b[k] & sign_of(p.sb) : cs.b[k];
        }
    }
    if (K != 16 || q.cmode == 0) {
        return;
    }
    double sm = sum_of_magnitudes(p, cs);
    if (sm == 0) {
        return;
    }
    long double S = 0;
    for (int k = 0; k < K; ++k) {
        const Term t = term_of(p, cs.a[k], cs.b[k]);
        S += std::ldexp((long double) t.m, t.x) * (t.neg ? -1 : 1);
    }
    std::uniform_real_distribution<double> u01(0, 1);
    if (q.cmode == 1) {
        const double c = sm * std::ldexp(1.0, -(int) (rng() % 5)) * u01(rng) * ((rng() & 1) ? 1 : -1);
        cs.c = bits32((float) c);
        if (g_nosub && subnormal_code(am::F32, cs.c)) {
            cs.c = 0;
        }
        return;
    }
    const int s = scale_for(sm, T);
    const long double Ss = std::ldexp(S, s);
    const int e = T == 0 ? std::max(-14, std::ilogb((double) Ss == 0 ? 1e-300 : (double) Ss)) : std::max(-6, std::ilogb((double) Ss == 0 ? 1e-300 : (double) Ss));
    const long double u = std::ldexp((long double) 1, e - (T == 0 ? 10 : 3));
    const long double mid = (std::floor(Ss / u) + 0.5L) * u;
    const long double amp = std::ldexp((long double) sm, s - q.jc);
    const long double Cs = mid - Ss + amp * (2 * u01(rng) - 1);
    cs.c = bits32((float) std::ldexp(Cs, -s));
    if (!usable(am::F32, cs.c)) {
        cs.c = 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Hill climbing on raw codes (worst) and on generator parameters (rate)

static void mutate_case(const PathDef & p, Case & cs, std::mt19937_64 & rng) {
    const int K = cs.K;
    const int k = (int) (rng() % (uint64_t) K), j = (int) (rng() % (uint64_t) K);
    const int op = (int) (rng() % 9);
    auto pick = [&](bool a) -> uint32_t & { return a ? cs.a[k] : cs.b[k]; };
    const bool useA = rng() & 1;
    const am::Fmt f = useA ? p.sa : p.sb;
    switch (op) {
    case 0:
    case 1: {
        const uint32_t old = pick(useA);
        for (int t = 0; t < 8; ++t) {
            const uint32_t v = old ^ (1u << (rng() % (uint64_t) nbits(f)));
            if (usable(f, v)) {
                pick(useA) = v;
                break;
            }
        }
        break;
    }
    case 2:
        pick(useA) = random_finite(f, rng);
        break;
    case 3:
        cs.a[k] = 0;
        break;
    case 4:
        cs.a[k] = cs.a[j] ^ sign_of(p.sa);
        cs.b[k] = cs.b[j];
        break;
    case 5:
        cs.a[k] = cs.a[j] ^ sign_of(p.sa) ^ (1u << (rng() % (uint64_t) am::info(p.sa).mbits));
        cs.b[k] = cs.b[j];
        break;
    case 6:
        cs.a[k] = cs.a[j];
        cs.b[k] = cs.b[j];
        break;
    case 7:
        std::swap(cs.a[k], cs.a[j]);
        std::swap(cs.b[k], cs.b[j]);
        break;
    default:
        if (K == 16) {
            const uint32_t v = cs.c ^ (1u << (rng() % 31));
            if (usable(am::F32, v)) {
                cs.c = v;
            }
        } else {
            pick(useA) = random_finite(f, rng);
        }
        break;
    }
}

static bool has_subnormal(const PathDef & p, const Case & cs) {
    for (int k = 0; k < cs.K; ++k) {
        if (subnormal_code(p.sa, cs.a[k]) || subnormal_code(p.sb, cs.b[k])) {
            return true;
        }
    }
    return subnormal_code(am::F32, cs.c);
}

struct Best {
    Case cs;
    Res r;
    double score = -1;
};

static Best climb(const PathDef & p, int T, Case start, int iters, std::mt19937_64 & rng) {
    Best b;
    b.cs = start;
    b.r = evaluate(p, start, T, true);
    b.score = b.r.ok ? (double) b.r.dist + b.r.smooth : -1;
    for (int it = 0; it < iters; ++it) {
        Case c = b.cs;
        mutate_case(p, c, rng);
        if (rng() % 3 == 0) {
            mutate_case(p, c, rng);
        }
        if (g_nosub && has_subnormal(p, c)) {
            continue;  // a sign-and-bit copy of a zero code (mutations 4 and 5) can make a subnormal
        }
        const Res r = evaluate(p, c, T, true);
        if (!r.ok) {
            continue;
        }
        const double sc = (double) r.dist + r.smooth;
        if (sc >= b.score) {
            b.cs = c;
            b.r = r;
            b.score = sc;
        }
    }
    return b;
}

// ---------------------------------------------------------------------------------------------------------------
// Printing and re-reading a case (sparse hex: only slots with a nonzero code)

static std::string ser(const Case & cs) {
    std::string s;
    char buf[64];
    std::snprintf(buf, sizeof buf, "K=%d C=0x%08X", cs.K, cs.c);
    s += buf;
    for (int k = 0; k < cs.K; ++k) {
        if (cs.a[k] || cs.b[k]) {
            std::snprintf(buf, sizeof buf, " %d:%X/%X", k, cs.a[k], cs.b[k]);
            s += buf;
        }
    }
    return s;
}

static bool parse(const std::string & s, Case & cs) {
    unsigned K = 0, c = 0;
    int used = 0;
    if (std::sscanf(s.c_str(), "K=%u C=0x%x%n", &K, &c, &used) != 2 || (K != 16 && K != 256 && K != KMAX)) {
        return false;
    }
    cs.K = (int) K;
    cs.c = c;
    std::fill(cs.a, cs.a + KMAX, 0u);
    std::fill(cs.b, cs.b + KMAX, 0u);
    const char * q = s.c_str() + used;
    int k, n;
    unsigned a, b;
    while (std::sscanf(q, " %d:%x/%x%n", &k, &a, &b, &n) == 3) {
        if (k < 0 || k >= cs.K) {
            return false;
        }
        cs.a[k] = a;
        cs.b[k] = b;
        q += n;
    }
    return true;
}

// Zero products while the distance stays; slots with a zero product are then written as 0/0.
static void shrink(const PathDef & p, int T, Best & b) {
    const int64_t D = b.r.dist;
    for (int pass = 0; pass < 4; ++pass) {
        bool changed = false;
        for (int k = 0; k < b.cs.K; ++k) {
            if (b.cs.a[k] == 0 && b.cs.b[k] == 0) {
                continue;
            }
            Case c = b.cs;
            c.a[k] = 0;
            c.b[k] = 0;
            const Res r = evaluate(p, c, T, true);
            if (r.ok && r.dist >= D && r.sumabs > 0) {
                b.cs = c;
                b.r = r;
                changed = true;
            }
        }
        if (b.cs.c) {
            Case c = b.cs;
            c.c = 0;
            const Res r = evaluate(p, c, T, true);
            if (r.ok && r.dist >= D && r.sumabs > 0) {
                b.cs = c;
                b.r = r;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
}

static int nonzero_products(const PathDef & p, const Case & cs) {
    int n = 0;
    for (int k = 0; k < cs.K; ++k) {
        n += term_of(p, cs.a[k], cs.b[k]).m != 0;
    }
    return n;
}

// ---------------------------------------------------------------------------------------------------------------
// Independent re-check of a printed case: a fresh chain of the model for the card, and for the exact sum a 128-bit
// integer sum (falls back to am::Exact when the terms span more than 120 bits).

struct Check {
    bool ok;
    bool integer_path;
    int64_t dist;
    uint32_t hwc, exc;
};

static Check recheck(const PathDef & p, int T, const std::string & text) {
    Check ck{false, false, 0, 0, 0};
    Case cs;
    if (!parse(text, cs)) {
        return ck;
    }
    const am::Params P = am::measured_params(p.form);
    uint32_t acc = cs.c;
    std::vector<Term> terms;
    for (int s = 0; s < cs.K / 16; ++s) {
        uint32_t A[16], B[16];
        for (int k = 0; k < 16; ++k) {
            A[k] = form_a(p, cs.a[16 * s + k]);
            B[k] = cs.b[16 * s + k];
            terms.push_back(term_of(p, cs.a[16 * s + k], cs.b[16 * s + k]));
        }
        acc = am::model_f32(p.form, A, B, acc, P);
    }
    const am::Dec dc = am::decode(am::F32, cs.c);
    terms.push_back({dc.neg, (uint64_t) dc.m, dc.e - dc.mbits});
    long double sm = 0;
    int xmin = 1 << 20, xtop = -(1 << 20);
    for (const Term & t : terms) {
        if (t.m == 0) {
            continue;
        }
        sm += std::ldexp((long double) t.m, t.x);
        xmin = std::min(xmin, t.x);
        xtop = std::max(xtop, t.x + am::bitlen(t.m));
    }
    if (sm == 0) {
        ck.ok = true;
        return ck;
    }
    const int s = scale_for((double) sm, T);
    ck.hwc = round_scaled(acc, s, T);
    if (xtop - xmin <= 120) {
        __int128 S = 0;
        for (const Term & t : terms) {
            if (t.m) {
                const __int128 v = (__int128) t.m << (t.x - xmin);
                S += t.neg ? -v : v;
            }
        }
        ck.exc = am::round_int(tfmt(T), S, xmin + s, false, tsat(T));
        ck.integer_path = true;
    } else {
        am::Exact ex;
        for (const Term & t : terms) {
            ex.add(t.neg, t.m, t.x + s);
        }
        ck.exc = ex.round(tfmt(T), tsat(T));
    }
    ck.dist = std::llabs(ordc(T, ck.hwc) - ordc(T, ck.exc));
    ck.ok = true;
    return ck;
}

// ---------------------------------------------------------------------------------------------------------------
// Control: the counting must see exactly the rounding boundaries. For an FP32 value v and the next FP32 value up,
// the two rounded codes are one step apart exactly when a target rounding boundary lies between them (a boundary
// at v rounds to the even code, one at the upper value goes to the even code too). The expectation comes from a
// brute-force search of a sorted table of all target values, which shares no code with am::round_int.

static bool control(std::string & text) {
    bool ok = true;
    long total_ones = 0, total_ties = 0, total_mismatch = 0, total_big = 0;
    for (int T = 0; T < 2; ++T) {
        std::vector<double> val;
        const uint32_t last = T == 0 ? 0x7BFFu : 0x7Eu;
        for (uint32_t c = 0; c <= last; ++c) {
            val.push_back(am::value(tfmt(T), c));
        }
        auto brute = [&](double x) -> int64_t {  // ordered code of x rounded to nearest, ties to the even code
            const bool neg = x < 0;
            const double ax = std::fabs(x);
            size_t i = (size_t) (std::upper_bound(val.begin(), val.end(), ax) - val.begin()) - 1;
            int64_t code;
            if (i + 1 >= val.size()) {
                code = (int64_t) i;
            } else {
                const double d1 = ax - val[i], d2 = val[i + 1] - ax;
                code = d1 < d2 ? (int64_t) i : d2 < d1 ? (int64_t) i + 1 : ((i & 1) ? (int64_t) i + 1 : (int64_t) i);
            }
            return neg ? -code : code;
        };
        std::mt19937_64 rng(777 + T);
        long ones_p = 0, ones_b = 0, ties = 0, mismatch = 0, big = 0;
        const int N = 400000;
        for (int i = 0; i < N; ++i) {
            const size_t ci = (size_t) (rng() % (last));  // interval [val[ci], val[ci+1]]
            const double lo = val[ci], hi = val[ci + 1];
            double x;
            const int kind = (int) (rng() % 5);
            if (kind == 0) {
                x = lo + (hi - lo) * ((double) (rng() & 0xFFFFFF) / 16777216.0);
            } else if (kind == 1) {
                x = lo + (hi - lo) * ((double) (rng() % 17) / 16.0);
            } else {
                x = 0.5 * (lo + hi);
            }
            float v = (float) x;
            if (kind == 3) {
                v = std::nextafter(v, -INFINITY);
            } else if (kind == 4 && (rng() & 1)) {
                v = std::nextafter(v, INFINITY);
            }
            if (v == 0) {
                continue;
            }
            const bool on_mid = kind >= 2 && (double) v == x;
            if (rng() & 1) {
                v = -v;
            }
            const float vu = std::nextafter(v, INFINITY);
            const int64_t dp = std::llabs(ordc(T, round_scaled(bits32(v), 0, T)) - ordc(T, round_scaled(bits32(vu), 0, T)));
            const int64_t db = std::llabs(brute((double) v) - brute((double) vu));
            mismatch += dp != db;
            big += dp > 1;
            ones_p += dp == 1;
            ones_b += db == 1;
            ties += on_mid;
        }
        if (g_break) {
            ones_b += 1;  // wrong expectation on purpose
        }
        std::printf("RESULT control target %s samples %d counted_one_step %ld expected_one_step %ld exact_midpoints_tested %ld "
                    "mismatches %ld two_or_more_steps %ld\n",
                    tname(T), N, ones_p, ones_b, ties, mismatch, big);
        ok = ok && mismatch == 0 && big == 0 && ones_p == ones_b && ones_p > 0 && ties > 0;
        total_ones += ones_p;
        total_ties += ties;
        total_mismatch += mismatch;
        total_big += big;
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "control one-step pairs %ld, midpoints %ld, mismatches %ld", total_ones, total_ties,
                  total_mismatch + total_big);
    text = buf;
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Rate search

struct RateOut {
    Par par;
    long hit = 0, N = 0;
    int64_t maxd = 0;
    double sumd = 0;
};

static void rate_count(const PathDef & p, int K, int T, const Par & q, uint64_t seed, long N, long & hit, int64_t & maxd,
                       double & sumd) {
    hit = 0;
    maxd = 0;
    sumd = 0;
    Case cs;
    for (long i = 0; i < N; ++i) {
        std::mt19937_64 rng(seed * 1000003ull + (uint64_t) i * 7919ull + 17);
        gen(p, K, q, T, rng, cs);
        const Res r = evaluate(p, cs, T, false);
        if (r.ok && r.dist >= 1) {
            ++hit;
            sumd += (double) r.dist;
            maxd = std::max(maxd, r.dist);
        }
    }
}

static void wilson(long k, long n, double & lo, double & hi) {
    const double z = 1.96, ph = (double) k / n;
    const double den = 1 + z * z / n, ctr = ph + z * z / (2 * n);
    const double rad = z * std::sqrt(ph * (1 - ph) / n + z * z / (4.0 * n * n));
    lo = (ctr - rad) / den;
    hi = (ctr + rad) / den;
}

// ---------------------------------------------------------------------------------------------------------------

// EQ-2 --nonfinite: class of the card's result against the IEEE class of the exact sum.
enum Cls { C_FIN = 0, C_PINF, C_NINF, C_NAN };
static const char * const cls_name[4] = {"finite", "+inf", "-inf", "NaN"};
static int cls_of(uint32_t f32b) {
    const am::Dec d = am::decode(am::F32, f32b);
    return d.nan ? C_NAN : d.inf ? (d.neg ? C_NINF : C_PINF) : C_FIN;
}

static uint32_t special_code(am::Fmt f, std::mt19937_64 & rng) {
    const am::FmtInfo fi = am::info(f);
    const uint32_t sign = (uint32_t) (rng() & 1) << (fi.ebits + fi.mbits);
    const uint32_t emax = (1u << fi.ebits) - 1;
    if (fi.e4m3) {
        return sign | 0x7Fu;  // E4M3 has no infinity: NaN only
    }
    if (rng() & 1) {
        return sign | (emax << fi.mbits);  // infinity
    }
    return sign | (emax << fi.mbits) | (1u + (uint32_t) (rng() % ((1u << fi.mbits) - 1)));  // NaN with a payload
}

static bool nonfinite_test(std::string & text) {
    bool ok = true;
    long total = 0, total_bad = 0, by_cls[4] = {};
    for (int pi = 0; pi < 4; ++pi) {
        const PathDef & p = PATHS[pi];
        for (int K : {16, 256}) {
            const long N = K == 16 ? 200000 : 20000;
            std::vector<long> bad(64, 0), cnt(64 * 4, 0);
            std::vector<std::string> ex(64);
            par_for(64, [&](size_t ch, int) {
                for (long i = (long) ch; i < N; i += 64) {
                    std::mt19937_64 rng(5150000ull + (uint64_t) pi * 1000003ull + (uint64_t) K * 7919ull + (uint64_t) i);
                    Case cs;
                    Par q{};
                    q.spread = 6;
                    q.mant = 10;
                    gen(p, K, q, 0, rng, cs);
                    // specials: each input slot 1 % (one case in four: 5 %), C 10 %; one case in eight pairs an
                    // infinity with a zero on purpose
                    const int per = rng() % 4 == 0 ? 5 : 1;
                    for (int k = 0; k < K; ++k) {
                        if ((int) (rng() % 100) < per) cs.a[k] = special_code(p.sa, rng);
                        if ((int) (rng() % 100) < per) cs.b[k] = special_code(p.sb, rng);
                    }
                    if (rng() % 10 == 0) {
                        cs.c = (rng() % 3 == 0) ? 0x7FC00000u + (uint32_t) (rng() & 0xFFFF) : (rng() & 1) ? 0x7F800000u : 0xFF800000u;
                    }
                    if (rng() % 8 == 0 && !am::info(p.sb).e4m3) {
                        const int k = (int) (rng() % (uint64_t) K);
                        cs.a[k] = 0;
                        cs.b[k] = (uint32_t) (am::info(p.sb).ebits ? ((1u << am::info(p.sb).ebits) - 1) << am::info(p.sb).mbits : 0);
                    }
                    // the card (model) result
                    const am::Params P = am::measured_params(p.form);
                    uint32_t acc = cs.c;
                    for (int s2 = 0; s2 < K / 16; ++s2) {
                        uint32_t A[16], B[16];
                        for (int k = 0; k < 16; ++k) {
                            A[k] = form_a(p, cs.a[16 * s2 + k]);
                            B[k] = cs.b[16 * s2 + k];
                        }
                        acc = am::model_f32(p.form, A, B, acc, P);
                    }
                    // the IEEE class of the exact sum
                    bool nan = false, pinf = false, ninf = false;
                    am::Exact exs;
                    for (int k = 0; k < K; ++k) {
                        const am::Dec da = am::decode(am::fmt_a(p.form), form_a(p, cs.a[k])), db = am::decode(am::fmt_b(p.form), cs.b[k]);
                        if (da.nan || db.nan || (da.inf && db.zero) || (db.inf && da.zero)) {
                            nan = true;
                        } else if (da.inf || db.inf) {
                            ((da.neg != db.neg) ? ninf : pinf) = true;
                        } else {
                            exs.add(da.neg != db.neg, (uint64_t) (da.m * db.m), da.e - da.mbits + db.e - db.mbits);
                        }
                    }
                    const am::Dec dc = am::decode(am::F32, cs.c);
                    if (dc.nan) {
                        nan = true;
                    } else if (dc.inf) {
                        (dc.neg ? ninf : pinf) = true;
                    } else {
                        exs.add(dc.neg, (uint64_t) dc.m, dc.e - dc.mbits);
                    }
                    int ref = nan || (pinf && ninf) ? C_NAN : pinf ? C_PINF : ninf ? C_NINF : C_FIN;
                    if (ref == C_FIN) {
                        ref = cls_of(exs.round(am::F32));  // an exact sum beyond the FP32 range rounds to infinity
                    }
                    if (g_break && ref == C_NAN) {
                        ref = C_FIN;
                    }
                    const int got = cls_of(acc);
                    ++cnt[ch * 4 + ref];
                    if (got != ref) {
                        if (bad[ch] == 0) {
                            char b2[160];
                            std::snprintf(b2, sizeof b2, "case %ld: card %s, IEEE %s, C=0x%08X", i, cls_name[got], cls_name[ref], cs.c);
                            ex[ch] = b2;
                        }
                        ++bad[ch];
                    }
                }
            });
            long nb = 0, c4[4] = {};
            std::string first;
            for (int ch = 0; ch < 64; ++ch) {
                nb += bad[ch];
                for (int c = 0; c < 4; ++c) c4[c] += cnt[ch * 4 + c];
                if (first.empty() && !ex[ch].empty()) first = ex[ch];
            }
            std::printf("RESULT nonfinite path %s K %d cases %ld expected classes finite %ld +inf %ld -inf %ld NaN %ld | card class "
                        "differs %ld%s%s\n",
                        p.tag, K, N, c4[0], c4[1], c4[2], c4[3], nb, first.empty() ? "" : " first: ", first.c_str());
            ok = ok && nb == 0 && c4[3] > 0 && (am::info(p.sa).e4m3 && am::info(p.sb).e4m3 ? true : c4[1] > 0 && c4[2] > 0);
            total += N;
            total_bad += nb;
            for (int c = 0; c < 4; ++c) by_cls[c] += c4[c];
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "non-finite inputs: %ld cases (NaN %ld, +inf %ld, -inf %ld expected), card class differs in %ld",
                  total, by_cls[3], by_cls[1], by_cls[2], total_bad);
    text = buf;
    return ok;
}

struct Report {
    int path, K, T;
    std::string text;
    Res r;
    int nz;
};

int main(int argc, char ** argv) {
    bool k1024 = false, nonfinite = false;
    for (int i = 1; i < argc; ++i) {
        g_break = g_break || std::string(argv[i]) == "--break";
        g_nosub = g_nosub || std::string(argv[i]) == "--nosub";
        k1024 = k1024 || std::string(argv[i]) == "--k1024";
        nonfinite = nonfinite || std::string(argv[i]) == "--nonfinite";
    }
    if (nonfinite) {
        std::string t;
        const bool ok = nonfinite_test(t);
        std::printf("VERDICT arith_qworst_nonfinite: %s %s%s\n", ok ? "PASS" : "FAIL", t.c_str(), g_break ? " (broken on purpose)" : "");
        return ok ? 0 : 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    std::string ctl_text;
    bool ok = control(ctl_text);

    // sizes: restarts, climbing iterations per restart, rate-search restarts / iterations / samples per evaluation,
    // and the number of fresh samples for the reported rate
    struct Size {
        int w_restarts, w_iters, r_restarts, r_iters;
        long r_n, r_final;
    };
#ifdef QW_QUICK
    const Size SZ[2] = {{16, 300, 8, 6, 100, 2000}, {16, 20, 8, 4, 10, 200}};
#else
    const Size SZ[2] = {{32, 150000, 16, 120, 1500, 400000}, {32, 7000, 16, 60, 150, 60000}};
#endif
    // --k1024: chains of 64 instructions; a quarter of the K = 256 iterations and samples (each costs 4 times more)
    const Size SZ1024 = {32, 1750, 16, 30, 40, 15000};

    std::vector<Report> reps;
    for (int pi = 0; pi < 4; ++pi) {
        const PathDef & p = PATHS[pi];
        for (int ki = k1024 ? 1 : 0; ki < 2; ++ki) {
            const int K = k1024 ? KMAX : ki == 0 ? 16 : 256;
            const Size & z = k1024 ? SZ1024 : SZ[ki];
            for (int T = 0; T < 2; ++T) {
                // ---- worst: largest step distance
                std::vector<Best> bests(z.w_restarts);
                par_for((size_t) z.w_restarts, [&](size_t rs, int) {
                    std::mt19937_64 rng(1000ull * (pi * 4 + ki * 2 + T) + rs + 5);
                    Best best;
                    for (int tries = 0; tries < 200 && best.score < 0; ++tries) {
                        Case cs;
                        gen(p, K, random_par(rng), T, rng, cs);
                        best = climb(p, T, cs, 0, rng);
                    }
                    bests[rs] = climb(p, T, best.cs, z.w_iters, rng);
                });
                size_t bi = 0;
                for (size_t i = 1; i < bests.size(); ++i) {
                    if (bests[i].score > bests[bi].score) {
                        bi = i;
                    }
                }
                Best w = bests[bi];
                shrink(p, T, w);
                Report rep{pi, K, T, ser(w.cs), w.r, nonzero_products(p, w.cs)};
                reps.push_back(rep);
                std::printf("RESULT worst path %s K %d target %s max_steps %lld nonzero_products %d scale 2^%d card_code 0x%X "
                            "exact_code 0x%X exact_over_sum_of_magnitudes 2^%.1f example: %s\n",
                            p.tag, K, tname(T), (long long) w.r.dist, rep.nz, w.r.s, w.r.hwc, w.r.exc,
                            std::log2(std::fabs(am::value(tfmt(T), w.r.exc)) / std::ldexp(w.r.sumabs, w.r.s)), rep.text.c_str());
                std::fflush(stdout);

                // ---- rate: how often the rounded values differ
                std::vector<RateOut> outs(z.r_restarts);
                par_for((size_t) z.r_restarts, [&](size_t rs, int) {
                    std::mt19937_64 rng(777ull * (pi * 4 + ki * 2 + T) + rs + 11);
                    Par q = random_par(rng);
                    long hit;
                    int64_t md;
                    double sd;
                    rate_count(p, K, T, q, 4242, z.r_n, hit, md, sd);
                    for (int it = 0; it < z.r_iters; ++it) {
                        Par q2 = q;
                        mutate_par(q2, rng);
                        long h2;
                        rate_count(p, K, T, q2, 4242, z.r_n, h2, md, sd);
                        if (h2 >= hit) {
                            hit = h2;
                            q = q2;
                        }
                    }
                    outs[rs].par = q;
                    outs[rs].hit = hit;
                });
                size_t ri = 0;
                for (size_t i = 1; i < outs.size(); ++i) {
                    if (outs[i].hit > outs[ri].hit) {
                        ri = i;
                    }
                }
                // fresh samples for the best parameters
                const int chunks = 64;
                std::vector<long> hits(chunks, 0);
                std::vector<int64_t> mds(chunks, 0);
                std::vector<double> sds(chunks, 0);
                const long per = z.r_final / chunks;
                par_for(chunks, [&](size_t c, int) { rate_count(p, K, T, outs[ri].par, 9000 + c, per, hits[c], mds[c], sds[c]); });
                long hit = 0;
                int64_t md = 0;
                double sd = 0;
                for (int c = 0; c < chunks; ++c) {
                    hit += hits[c];
                    md = std::max(md, mds[c]);
                    sd += sds[c];
                }
                const long N = per * chunks;
                double lo, hi;
                wilson(hit, N, lo, hi);
                const Par & q = outs[ri].par;
                std::printf("RESULT rate path %s K %d target %s differs %.4f%% (%.4f-%.4f) hits %ld N %ld mean_steps_when_differs %.3f "
                            "max_steps_seen %lld params spread %d cancel %d zero %d mant %d subnormal %d cmode %d jc %d "
                            "search_best_on_search_samples %ld/%ld\n",
                            p.tag, K, tname(T), 100.0 * hit / N, 100 * lo, 100 * hi, hit, N, hit ? sd / hit : 0.0, (long long) md,
                            q.spread, q.cancel, q.zero, q.mant, q.subp, q.cmode, q.jc, outs[ri].hit, z.r_n);
                std::fflush(stdout);
            }
        }
    }

    // every reported worst case is recomputed from its printed text
    int rechecked = 0, integer = 0, bad = 0;
    int64_t overall = 0;
    for (const Report & r : reps) {
        const Check ck = recheck(PATHS[r.path], r.T, r.text);
        const bool same = ck.ok && ck.dist == r.r.dist && ck.hwc == r.r.hwc && ck.exc == r.r.exc;
        ++rechecked;
        integer += ck.integer_path;
        bad += !same;
        overall = std::max(overall, r.r.dist);
        if (!same) {
            std::printf("RESULT recheck FAILED path %s K %d target %s stored dist %lld codes 0x%X/0x%X recomputed dist %lld codes "
                        "0x%X/0x%X\n",
                        PATHS[r.path].tag, r.K, tname(r.T), (long long) r.r.dist, r.r.hwc, r.r.exc, (long long) ck.dist, ck.hwc,
                        ck.exc);
        }
    }
    std::printf("RESULT recheck cases %d recomputed_equal %d by_integer_sum %d largest_step_distance_any %lld seconds %.1f\n",
                rechecked, rechecked - bad, integer, (long long) overall, secs());
    ok = ok && bad == 0 && rechecked == (k1024 ? 8 : 16);

    char buf[256];
    std::snprintf(buf, sizeof buf, "%s; %d/%d worst cases recomputed equal; largest distance %lld steps%s", ctl_text.c_str(),
                  rechecked - bad, rechecked, (long long) overall, g_break ? " (broken on purpose)" : "");
    std::printf("VERDICT arith_qworst%s: %s %s%s%s\n", g_nosub ? "_nosub" : k1024 ? "_k1024" : "", ok ? "PASS" : "FAIL", buf,
                g_nosub ? " (no subnormal inputs)" : "", k1024 ? " (K = 1024)" : "");
    return ok ? 0 : 1;
}
