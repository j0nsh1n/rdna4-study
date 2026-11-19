// Lane 5, second part: the matrix forms and features the first part did not measure (measurements/
// matrix-arithmetic.md sections 9-14). Each mode compares the card with a CPU model (arith_model.h) bit for bit.
//
// Usage: arith_more MODE [--n N] [--break]
//   acc16     FP16 x FP16 into FP16 and BF16 x BF16 into BF16: random, adversarial, special values, chains through
//             C, MODE.FP16_OVFL (with a VALU control); controls: the other places of the 16-bit rounding.
//   ovfl      MODE.FP16_OVFL and rounding modes on the 16-bit accumulator forms, proved by a VALU control.
//   mods      NEG and NEG_HI on A and B, NEG and ABS on C: the card with the modifier against the card without it
//             on operands with sign bits changed in memory; controls: the other halves, the other operand.
//   sparsemap one-hot experiments: which B slot each stored A element meets, for every index field value; against
//             the rule in arith_model.h (sparse_slot).
//   sparse    SWMMAC FP16, BF16, FP8 into FP32 and FP16 into FP16, index_key 0 and 1: random problems with random
//             per-lane index words (invalid values included) against the model; the same 16 products through the
//             dense and the sparse instruction, card against card.
//   int       IU8 (K = 16) and IU4 (K = 16 and 32), every sign setting with and without clamp: random, extreme and
//             structured inputs against the exact integer model.
//   chain8    chains through C (K = 64, 1024, 4096) for FP8 x BF8, BF8 x FP8, BF8 x BF8.
//   hazard    a 16-bit result used as A or B of the very next matrix instruction (forbidden by the guide) against
//             the same with a V_NOP; what was read instead (stale contents, a mix of registers).
//   --break   perturbs the model or the expectation; the verdict must turn to FAIL.
#include "arith_data.h"
#include "arith_gpu.h"

#include <atomic>
#include <cstring>
#include <cstdarg>
#include <functional>
#include <string>
#include <thread>

static bool g_break = false;

static void verdict(const std::string & probe, bool ok, const std::string & text) {
    std::printf("VERDICT %s: %s %s%s\n", probe.c_str(), ok ? "PASS" : "FAIL", text.c_str(),
                g_break ? " (broken on purpose)" : "");
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

static void add_text(std::string & text, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
static void add_text(std::string & text, const char * fmt, ...) {
    char buf[400];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!text.empty()) {
        text += "; ";
    }
    text += buf;
}

// Problems through the card, 8192 waves per launch; steps > 1 chains each wave's problems through C (the
// problem list is then [wave][step], C taken from the first step's problem). Output [wave][r][c].
static std::vector<uint32_t> run_problems(int form, const std::vector<ad::Problem> & ps, int mod = 0, int steps = 1,
                                          bool use_mode = false, uint32_t mode = 0) {
    const size_t nwav = ps.size() / steps;
    std::vector<uint32_t> out(nwav * 256);
    const size_t CH = 8192;
    for (size_t base = 0; base < nwav; base += CH) {
        const int n = (int) std::min(CH, nwav - base);
        arith::Batch bt;
        bt.mod = mod;
        bt.init(form, n, steps);
        for (int w = 0; w < n; ++w) {
            for (int s = 0; s < steps; ++s) {
                const ad::Problem & p = ps[(base + w) * steps + s];
                for (int r = 0; r < 16; ++r) {
                    for (int k = 0; k < 16; ++k) {
                        bt.set_a(w, s, r, k, p.a[r][k]);
                        bt.set_b(w, s, r, k, p.b[r][k]);
                    }
                }
            }
            const ad::Problem & p0 = ps[(base + w) * steps];
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    bt.set_c(w, r, c, p0.c[r][c]);
                }
            }
        }
        arith::run(bt, use_mode, mode);
        for (int w = 0; w < n; ++w) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    out[(base + w) * 256 + r * 16 + c] = bt.get_d(w, r, c);
                }
            }
        }
    }
    return out;
}

static int g_dump = 0, g_dump_form = 0;
static am::Fmt g_dump_out = am::F32;

// Debug print of one element: every nonzero product (slot, value), C, card and model.
static void dump_elem(const ad::Problem * p, int steps, int r, int c, uint32_t got, uint32_t want) {
    std::printf("DUMP card %08x (%.10g) model %08x (%.10g) C %08x (%.10g)\n", got, am::value(g_dump_out, got), want,
                am::value(g_dump_out, want), p[0].c[r][c], am::value(g_dump_out, p[0].c[r][c]));
    for (int s = 0; s < steps; ++s) {
        for (int k = 0; k < 16; ++k) {
            const double v = am::value(am::fmt_a(g_dump_form), p[s].a[r][k]) * am::value(am::fmt_b(g_dump_form), p[s].b[c][k]);
            if (v != 0) {
                std::printf("  step %d slot %2d a %04x b %04x product %.17g (2^%.2f)\n", s, k, p[s].a[r][k], p[s].b[c][k], v,
                            std::log2(std::fabs(v)));
            }
        }
    }
}

// Element model: inputs of problem p (chain: steps problems), row r, column c -> expected output bits.
typedef std::function<uint32_t(const ad::Problem * p, int steps, int r, int c)> ElemModel;

struct Tally {
    long n = 0, bad = 0;
    std::string first;
};

static Tally compare(const std::vector<ad::Problem> & ps, int steps, const std::vector<uint32_t> & d,
                     const ElemModel & m) {
    const size_t nwav = ps.size() / steps;
    std::vector<Tally> part(8);
    // dumps come from a serial pass over the first waves so they are complete and in order
    if (g_dump > 0) {
        Tally d0;
        for (size_t w = 0; w < std::min<size_t>(nwav, 64) && d0.bad < g_dump; ++w) {
            for (int r = 0; r < 16 && d0.bad < g_dump; ++r) {
                for (int c = 0; c < 16 && d0.bad < g_dump; ++c) {
                    const uint32_t want = m(&ps[w * steps], steps, r, c), got = d[w * 256 + r * 16 + c];
                    if (got != want) {
                        dump_elem(&ps[w * steps], steps, r, c, got, want);
                        ++d0.bad;
                    }
                }
            }
        }
    }
    par_for(nwav, [&](size_t w, int t) {
        Tally & ta = part[t];
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                const uint32_t got = d[w * 256 + r * 16 + c];
                const uint32_t want = m(&ps[w * steps], steps, r, c);
                ++ta.n;
                if (got != want) {
                    if (!ta.bad) {
                        char buf[200];
                        std::snprintf(buf, sizeof buf, "wave %zu (%d,%d): card %08x model %08x", w, r, c, got, want);
                        ta.first = buf;
                    }
                    ++ta.bad;
                }
            }
        }
    });
    Tally all;
    for (auto & t : part) {
        all.n += t.n;
        all.bad += t.bad;
        if (all.first.empty()) {
            all.first = t.first;
        }
    }
    return all;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 1: 16-bit accumulators.

static int in_form(int f16form) { return f16form == arith::F16H ? am::F_F16 : am::F_BF16; }

// Random problems for a 16-bit accumulator form; C in the output format.
static ad::Problem make16(int f16form, int dist, std::mt19937_64 & rng) {
    const int form = in_form(f16form);
    const am::Fmt fo = form == am::F_F16 ? am::F16 : am::BF16, fi = fo;
    ad::Problem p;
    if (dist == 0 || dist == 3 || form == am::F_BF16) {
        p = ad::make(form, dist == 0 ? ad::NARROW : dist == 3 ? ad::BIGSMALL : ad::WIDE, rng);
    } else {
        // FP16 "wide" scaled so most sums stay below 65504 (some overflow on purpose)
        for (int r = 0; r < 16; ++r) {
            for (int s = 0; s < 16; ++s) {
                p.a[r][s] = ad::rand_val(fi, rng, -14, 6);
                p.b[r][s] = ad::rand_val(fi, rng, -14, 6);
            }
        }
    }
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            const int pick = (int) (rng() & 7);
            uint32_t cv;
            if (pick == 0) {
                cv = (rng() & 1) ? 0x8000u : 0u;
            } else if (pick <= 2 || dist == 2) {
                // minus the sum rounded to 16 bits, sometimes one unit off: the result is the small remainder
                cv = am::exact_round(form, p.a[r], p.b[c], 0, fo) ^ 0x8000u;
                if (rng() & 1) {
                    cv ^= 1;
                }
            } else {
                cv = am::convert(am::F32, p.c[r][c], fo);
            }
            if (am::decode(fo, cv).inf || am::decode(fo, cv).nan) {
                cv = 0;
            }
            p.c[r][c] = cv;
        }
    }
    return p;
}

static ElemModel model16(int f16form, int round16, int rtz) {
    const int form = in_form(f16form);
    am::Params P = am::acc16_params(form, round16);
    P.rtz16 = rtz;
    if (g_break) {
        P.width += 1;
    }
    return [=](const ad::Problem * p, int steps, int r, int c) {
        uint32_t acc = p[0].c[r][c];
        for (int s = 0; s < steps; ++s) {
            acc = am::model_acc16(form, p[s].a[r], p[s].b[c], acc, P);
        }
        return acc;
    };
}

// Measured rounding per form: FP16 rounds every step to FP16 (nearest even); BF16 rounds every step to FP32 and
// then truncates to BF16. --break: the FP32 steps rounded once at the end instead.
static ElemModel measured16(int f, bool ovfl = false) {
    if (!ovfl) {
        return model16(f, g_break ? 1 : 3, f == arith::BF16H ? 2 : 0);
    }
    const int form = in_form(f);
    am::Params P = am::acc16_params(form, 3);
    P.rtz16 = f == arith::BF16H ? 2 : 0;
    P.sat16 = f == arith::F16H && !g_break;
    return [=](const ad::Problem * p, int steps, int r, int c) {
        uint32_t acc = p[0].c[r][c];
        for (int s = 0; s < steps; ++s) {
            acc = am::model_acc16(form, p[s].a[r], p[s].b[c], acc, P);
        }
        return acc;
    };
}

static uint32_t conv16(int f, uint32_t f32bits) { return am::convert(am::F32, f32bits, f == arith::F16H ? am::F16 : am::BF16); }

// Special values for a 16-bit accumulator form: NaNs with payloads and signs, infinities, signed zeros, subnormals,
// the largest finite value, in A, B and C; also overflow (products near the top of FP16) and subnormal results.
static std::vector<ad::Problem> special16(int f, int np, std::mt19937_64 & rng) {
    const am::Fmt fo = f == arith::F16H ? am::F16 : am::BF16;
    const am::FmtInfo fi = am::info(fo);
    const uint32_t sb = 1u << (fi.ebits + fi.mbits), emax = (1u << fi.ebits) - 1, inf = emax << fi.mbits;
    const uint32_t maxf = inf - 1, qnan = inf | (1u << (fi.mbits - 1));
    const std::vector<uint32_t> sv = {0,        sb,       1,           sb | 1,   (1u << fi.mbits) - 1, inf, sb | inf,
                                      inf | 1,  sb | qnan, qnan | 5,   maxf,     sb | maxf};
    std::vector<ad::Problem> ps(np);
    for (int i = 0; i < np; ++i) {
        ad::Problem & p = ps[i];
        p = make16(f, i % 4, rng);
        const int kind = i % 5;
        for (int r = 0; r < 16; ++r) {
            for (int j = 0, k = (int) (rng() % 3); j < k; ++j) {
                p.a[r][rng() % 16] = sv[rng() % sv.size()];
            }
            if (kind == 3) {
                // overflow: a few products near the largest value, all of one sign or mixed
                for (int j = 0; j < 6; ++j) {
                    p.a[r][rng() % 16] = (fo == am::F16 ? 0x7800u : 0x7F00u) | (uint32_t) (rng() & 0xFF) | ((rng() & 7) == 0 ? sb : 0);
                }
            }
            if (kind == 4) {
                // subnormal results: every product tiny
                for (int s = 0; s < 16; ++s) {
                    p.a[r][s] = (uint32_t) (rng() % (fo == am::F16 ? 0x0800u : 0x0400u)) | ((rng() & 1) ? sb : 0);
                }
            }
        }
        for (int c = 0; c < 16; ++c) {
            for (int j = 0, k = (int) (rng() % 2); j < k; ++j) {
                p.b[c][rng() % 16] = sv[rng() % sv.size()];
            }
            if (kind >= 3) {
                for (int s = 0; s < 16; ++s) {
                    p.b[c][s] = kind == 3 ? (fo == am::F16 ? 0x3C00u : 0x3F80u) | (uint32_t) (rng() & 0x7F)
                                          : (fo == am::F16 ? 0x1000u : 0x0800u) | (uint32_t) (rng() & 0x3FF) | ((rng() & 1) ? sb : 0);
                }
            }
            for (int r = 0; r < 16; ++r) {
                if (rng() % 3 == 0) {
                    p.c[r][c] = sv[rng() % sv.size()];
                }
                if (kind == 4 && (rng() & 1)) {
                    p.c[r][c] = (uint32_t) (rng() % 0x0400u) | ((rng() & 1) ? sb : 0);
                }
            }
        }
        if (i % 10 == 9) {
            // all-zero products with signed zeros and a signed-zero C
            for (int r = 0; r < 16; ++r) {
                for (int s = 0; s < 16; ++s) {
                    p.a[r][s] = (rng() & 1) ? sb : 0;
                }
                for (int c = 0; c < 16; ++c) {
                    p.c[r][c] = (rng() & 1) ? sb : 0;
                }
            }
        }
    }
    return ps;
}

static bool mode_acc16(long n, std::string & text) {
    bool ok = true;
    for (int f : {arith::F16H, arith::BF16H}) {
        const char * fn = arith::form_name[f];
        g_dump_form = in_form(f);
        g_dump_out = f == arith::F16H ? am::F16 : am::BF16;
        const ElemModel M = measured16(f);
        // random
        const long np = (n + 255) / 256;
        std::mt19937_64 rng(77 + f);
        std::vector<ad::Problem> ps(np);
        for (long i = 0; i < np; ++i) {
            ps[i] = make16(f, (int) (i % 4), rng);
        }
        std::vector<uint32_t> d = run_problems(f, ps);
        Tally t = compare(ps, 1, d, M);
        long exact_eq = 0;
        for (long i = 0; i < np; ++i) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    exact_eq += d[i * 256 + r * 16 + c] ==
                                am::exact_round(in_form(f), ps[i].a[r], ps[i].b[c], am::convert(g_dump_out, ps[i].c[r][c], am::F32), g_dump_out);
                }
            }
        }
        std::printf("RESULT acc16 %-9s random elements %ld model_mismatches %ld card_equals_exact_rounded_once %.2f%%%s%s\n", fn,
                    t.n, t.bad, 100.0 * exact_eq / t.n, t.bad ? " first: " : "", t.first.c_str());
        ok = ok && t.bad == 0;
        add_text(text, "%s random %ld/%ld", fn, t.n - t.bad, t.n);
        if (!g_break) {
            // controls: the 16-bit rounding in another place or of another kind must disagree visibly
            const std::vector<ad::Problem> sub(ps.begin(), ps.begin() + std::min<long>(np, 2048));
            const int meas_rtz = f == arith::BF16H ? 2 : 0;
            const struct {
                const char * name;
                int round16, rtz;
            } ctl[] = {{"FP32 steps, rounded once at the end", 1, meas_rtz},
                       {"last step rounded directly", 2, meas_rtz},
                       {"every step, other rounding", 3, meas_rtz == 2 ? 0 : 2},
                       {"every step toward zero from the exact step sum", 3, 1}};
            for (const auto & cv : ctl) {
                const Tally tc = compare(sub, 1, d, model16(f, cv.round16, cv.rtz));
                std::printf("RESULT acc16 %-9s control %-48s mismatches %ld of %ld (%.2f%%)\n", fn, cv.name, tc.bad, tc.n,
                            100.0 * tc.bad / tc.n);
                ok = ok && tc.bad > 0;
            }
        }
        // adversarial: cancelling big terms; FP16 kept below its overflow (big products at most about 2^14)
        {
            const long na = std::max(256L, n / 2) / 256;
            std::vector<ad::Problem> pa(na);
            std::mt19937_64 rng2(91 + f);
            for (long i = 0; i < na; ++i) {
                pa[i] = f == arith::F16H ? ad::adversarial(in_form(f), rng2, 7, 7) : ad::adversarial(in_form(f), rng2);
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        pa[i].c[r][c] = conv16(f, pa[i].c[r][c]);
                    }
                }
            }
            const std::vector<uint32_t> da = run_problems(f, pa);
            const Tally ta = compare(pa, 1, da, M);
            std::printf("RESULT acc16 %-9s adversarial elements %ld model_mismatches %ld%s%s\n", fn, ta.n, ta.bad,
                        ta.bad ? " first: " : "", ta.first.c_str());
            ok = ok && ta.bad == 0;
            add_text(text, "%s adv %ld/%ld", fn, ta.n - ta.bad, ta.n);
        }
        // special values
        {
            std::mt19937_64 rng3(55 + f);
            const std::vector<ad::Problem> pspec = special16(f, 400, rng3);
            const std::vector<uint32_t> ds = run_problems(f, pspec);
            const Tally ts = compare(pspec, 1, ds, M);
            long nan = 0, inf = 0, maxv = 0, neg0 = 0, sub = 0;
            for (uint32_t v : ds) {
                const am::Dec dv = am::decode(g_dump_out, v);
                nan += dv.nan;
                inf += dv.inf;
                maxv += (v & 0x7FFFu) == (f == arith::F16H ? 0x7BFFu : 0x7F7Fu);
                neg0 += v == 0x8000u;
                sub += !dv.nan && !dv.inf && !dv.zero && (v & 0x7FFFu) < (f == arith::F16H ? 0x0400u : 0x0080u);
            }
            std::printf("RESULT acc16 %-9s special elements %ld model_mismatches %ld nan %ld inf %ld largest_finite %ld "
                        "negative_zero %ld subnormal %ld%s%s\n",
                        fn, ts.n, ts.bad, nan, inf, maxv, neg0, sub, ts.bad ? " first: " : "", ts.first.c_str());
            ok = ok && ts.bad == 0;
            add_text(text, "%s special %ld/%ld", fn, ts.n - ts.bad, ts.n);
        }
        // chains through C (K = 64 and 1024)
        for (int steps : {4, 64}) {
            const int waves = steps == 4 ? 512 : 64;
            std::mt19937_64 rng4(steps + f);
            std::vector<ad::Problem> pc((size_t) waves * steps);
            for (auto & p : pc) {
                p = make16(f, (int) (rng4() % 2) * 3, rng4);  // narrow or big/small
            }
            const std::vector<uint32_t> dc = run_problems(f, pc, 0, steps);
            const Tally tc = compare(pc, steps, dc, M);
            std::printf("RESULT acc16 %-9s chain K %d elements %ld model_mismatches %ld%s%s\n", fn, steps * 16, tc.n, tc.bad,
                        tc.bad ? " first: " : "", tc.first.c_str());
            ok = ok && tc.bad == 0;
            add_text(text, "%s chain K%d %ld/%ld", fn, steps * 16, tc.n - tc.bad, tc.n);
        }
    }
    return ok;
}

// MODE.FP16_OVFL (bit 23) and the rounding mode on the 16-bit accumulator forms. --break: the model ignores
// FP16_OVFL. The VALU control (v_add_f16 of
// 65504 + 65504, and of 1 + 2^-12 under round toward +inf) proves each write; the matrix outputs on data that
// overflow FP16 and on random data are compared with the default MODE.
static bool mode_ovfl(std::string & text) {
    bool ok = true;
    const uint32_t dflt = 0x000000F0u;
    const struct {
        const char * name;
        uint32_t mode;
    } sets[] = {{"default written back", dflt}, {"FP16_OVFL", 0x008000F0u}, {"round toward zero", 0x000000FFu},
                {"round toward +inf", 0x000000F5u}, {"FP16_OVFL and round toward zero", 0x008000FFu}};
    // VALU control inputs: x4 = x5 = 65504 (0x7BFF) for the FP16 add; x6, x7 = 1, 2^-12 for the second FP16 add
    const uint32_t x[8] = {0x3F800000u, 0x33C00000u, 0xBF800001u, 0x3F800001u, 0x7BFFu, 0x7BFFu, 0x3C00u, 0x0C00u};
    const std::vector<uint32_t> v0 = arith::run_valu(x, true, dflt);
    for (int f : {arith::F16H, arith::BF16H}) {
        std::mt19937_64 rng(5 + f);
        std::vector<ad::Problem> ps(512);
        for (size_t i = 0; i < ps.size(); ++i) {
            ps[i] = i % 2 ? special16(f, 1, rng)[0] : make16(f, (int) (i % 4), rng);
        }
        // half of the problems overflow on purpose: special16 kind 3 problems
        std::mt19937_64 rng5(9);
        const std::vector<ad::Problem> over = special16(f, 20, rng5);
        for (size_t i = 0; i < over.size(); ++i) {
            if (i % 5 == 3) {
                ps.push_back(over[i]);
            }
        }
        const std::vector<uint32_t> base = run_problems(f, ps, 0, 1, true, dflt);
        g_dump_form = in_form(f);
        g_dump_out = f == arith::F16H ? am::F16 : am::BF16;
        const Tally tb = compare(ps, 1, base, measured16(f));
        long inf_base = 0;
        for (uint32_t v : base) {
            inf_base += am::decode(g_dump_out, v).inf;
        }
        for (const auto & st : sets) {
            const std::vector<uint32_t> v = arith::run_valu(x, true, st.mode);
            int vchg = 0;
            for (int k = 0; k < 5; ++k) {
                vchg += v[k] != v0[k];
            }
            const std::vector<uint32_t> d = run_problems(f, ps, 0, 1, true, st.mode);
            long chg = 0, inf = 0;
            for (size_t i = 0; i < d.size(); ++i) {
                chg += d[i] != base[i];
                inf += am::decode(g_dump_out, d[i]).inf;
            }
            const bool seen = (v[5] & 0x008000FFu) == st.mode;
            // the model with FP16_OVFL as read back (rounding mode ignored, as measured)
            const Tally tm = compare(ps, 1, d, measured16(f, (st.mode >> 23) & 1));
            std::printf("RESULT ovfl %-9s %-32s model (FP16_OVFL %s) mismatches %ld of %ld%s%s\n", arith::form_name[f],
                        st.name, (st.mode >> 23) & 1 ? "clamps finite overflow" : "off", tm.bad, tm.n,
                        tm.bad ? " first: " : "", tm.first.c_str());
            ok = ok && tm.bad == 0;
            std::printf("RESULT ovfl %-9s %-32s mode_read_back %08x valu_changed %d (fp16 add 65504+65504 = %04x) "
                        "matrix_changed %ld of %zu infinities %ld (default %ld)\n",
                        arith::form_name[f], st.name, v[5], vchg, v[2], chg, d.size(), inf, inf_base);
            // the write must be visible (read back) and, except for the default, change the VALU control
            ok = ok && seen && (st.mode == dflt ? vchg == 0 : vchg > 0);
            if (f == arith::F16H) {
                add_text(text, "%s: valu %d changed, f16_f16 %ld changed", st.name, vchg, chg);
            } else {
                add_text(text, "bf16_bf16 %ld changed", chg);
            }
        }
        std::printf("RESULT ovfl %-9s model under default MODE: mismatches %ld of %ld; overflowing outputs %ld\n",
                    arith::form_name[f], tb.bad, tb.n, inf_base);
        ok = ok && tb.bad == 0;
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 2: modifier bits. A transform changes sign bits of one operand in memory; the card with the modifier on the
// original data must equal the card without it on the transformed data, bit for bit.

enum Xf { X_A_EVEN = 0, X_A_ODD, X_B_EVEN, X_B_ODD, X_C_NEG, X_C_ABS, X_C_NEGABS, X_NONE, X_A_ALL, X_A_LO8, X_A_HI8, NXF };
static const char * const xf_name[NXF] = {"A sign flipped, even slots", "A sign flipped, odd slots", "B sign flipped, even slots",
                                          "B sign flipped, odd slots", "C sign flipped", "C sign cleared", "C sign set",
                                          "nothing changed", "A sign flipped, all slots", "A sign flipped, slots 0-7", "A sign flipped, slots 8-15"};

static void apply_xf(int f, ad::Problem & p, int x) {
    const int eb = arith::elem_bits(f), cb = arith::acc16(f) ? 16 : 32;
    const uint32_t sa = 1u << (eb - 1), sc = 1u << (cb - 1);
    for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 16; ++s) {
            const bool even = (s & 1) == 0;
            if ((x == X_A_EVEN && even) || (x == X_A_ODD && !even) || x == X_A_ALL || (x == X_A_LO8 && s < 8) ||
                (x == X_A_HI8 && s >= 8)) {
                p.a[r][s] ^= sa;
            }
            if ((x == X_B_EVEN && even) || (x == X_B_ODD && !even)) {
                p.b[r][s] ^= sa;
            }
        }
        for (int c = 0; c < 16; ++c) {
            if (x == X_C_NEG) {
                p.c[r][c] ^= sc;
            } else if (x == X_C_ABS) {
                p.c[r][c] &= ~sc;
            } else if (x == X_C_NEGABS) {
                p.c[r][c] |= sc;
            }
        }
    }
}

// Random problems with special values mixed in (NaN with payloads and both signs, infinities, signed zeros,
// subnormals) in A, B and C.
static std::vector<ad::Problem> mod_problems(int f, long np, std::mt19937_64 & rng) {
    std::vector<ad::Problem> ps(np);
    const bool h = arith::acc16(f);
    const int form = h ? in_form(f) : f;
    std::vector<ad::Problem> sp = h ? special16(f, (int) std::max(1L, np / 2), rng) : std::vector<ad::Problem>();
    const am::Fmt fa = am::fmt_a(form);
    const am::FmtInfo fi = am::info(fa);
    const uint32_t sb = 1u << (fi.ebits + fi.mbits), emax = (1u << fi.ebits) - 1;
    std::vector<uint32_t> sv = {0, sb, 1, sb | 1};
    if (fi.e4m3) {
        sv.insert(sv.end(), {0x7F, 0xFF});
    } else {
        sv.insert(sv.end(), {emax << fi.mbits, sb | (emax << fi.mbits), (emax << fi.mbits) | 3, sb | (emax << fi.mbits) | (1u << (fi.mbits - 1))});
    }
    const std::vector<uint32_t> sc = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC12345u, 0xFF800001u, 1u, 0x80000001u};
    for (long i = 0; i < np; ++i) {
        if (h) {
            ps[i] = (i & 1) ? sp[i / 2 % sp.size()] : make16(f, (int) (i % 4), rng);
            continue;
        }
        ps[i] = ad::make(form, (ad::Dist) (i % 4), rng);
        if (i & 1) {
            for (int r = 0; r < 16; ++r) {
                ps[i].a[r][rng() % 16] = sv[rng() % sv.size()];
                ps[i].b[r][rng() % 16] = sv[rng() % sv.size()];
                for (int c = 0; c < 16; ++c) {
                    if (rng() % 4 == 0) {
                        ps[i].c[r][c] = sc[rng() % sc.size()];
                    }
                }
            }
        }
    }
    return ps;
}

static long count_diff(const std::vector<uint32_t> & x, const std::vector<uint32_t> & y, size_t n) {
    long d = 0;
    for (size_t i = 0; i < n; ++i) {
        d += x[i] != y[i];
    }
    return d;
}

static bool mode_mods(long n, std::string & text) {
    bool ok = true;
    struct Case {
        int mod, expect, ctl1, ctl2;
    };
    // expected transform (the guide: NEG on the low halves = even slots, NEG_HI on the high halves = odd slots,
    // {NEG_HI[2], NEG[2]} = {ABS, NEG} on C) and two controls that must disagree: the other halves, the other operand
    const Case c16[] = {{arith::M_NEGA, X_A_EVEN, X_A_ODD, X_C_NEG},      {arith::M_NEGHIA, X_A_ODD, X_A_EVEN, X_C_NEG},
                        {arith::M_NEGB, X_B_EVEN, X_B_ODD, X_C_NEG},      {arith::M_NEGHIB, X_B_ODD, X_B_EVEN, X_C_NEG},
                        {arith::M_NEGC, X_C_NEG, X_C_ABS, X_A_EVEN},      {arith::M_ABSC, X_C_ABS, X_C_NEG, X_B_ODD},
                        {arith::M_NEGABSC, X_C_NEGABS, X_C_NEG, X_C_ABS}};
    for (int f : {arith::F16, arith::BF16, arith::F16H, arith::BF16H, arith::FF}) {
        const long np = (n + 255) / 256;
        std::mt19937_64 rng(300 + f);
        const std::vector<ad::Problem> ps = mod_problems(f, np, rng);
        // FP8: the guide gives these forms no NEG or ABS; the encoded C bits are ignored (measured), so the
        // expectation is "nothing changed" and the controls are the C sign changes
        const Case c8[] = {{arith::M_NEGC, X_NONE, X_C_NEG, X_C_ABS},
                           {arith::M_ABSC, X_NONE, X_C_ABS, X_C_NEG},
                           {arith::M_NEGABSC, X_NONE, X_C_NEGABS, X_C_NEG}};
        for (const Case & cs : f == arith::FF ? std::vector<Case>(c8, c8 + 3) : std::vector<Case>(c16, c16 + 7)) {
            if (!arith::mod_valid(f, cs.mod)) {
                continue;
            }
            const std::vector<uint32_t> dm = run_problems(f, ps, cs.mod);
            const int xs[3] = {g_break ? cs.ctl2 : cs.expect, cs.ctl1, cs.ctl2};
            long diff[3];
            for (int k = 0; k < 3; ++k) {
                // controls on the first 1024 problems (262,144 elements) are enough to show a visible rate
                const size_t m = k == 0 ? ps.size() : std::min<size_t>(ps.size(), 1024);
                std::vector<ad::Problem> px(ps.begin(), ps.begin() + m);
                for (auto & p : px) {
                    apply_xf(f, p, xs[k]);
                }
                const std::vector<uint32_t> db = run_problems(f, px);
                diff[k] = count_diff(dm, db, m * 256);
                std::printf("RESULT mods %-9s %-8s vs no modifier with %-28s elements %zu differ %ld%s\n", arith::form_name[f],
                            arith::mod_name[cs.mod], xf_name[xs[k]], m * 256, diff[k], k ? " (control)" : "");
            }
            ok = ok && diff[0] == 0 && diff[1] > 0 && diff[2] > 0;
            add_text(text, "%s %s = %s: %ld differ of %ld", arith::form_name[f], arith::mod_name[cs.mod], xf_name[xs[0]], diff[0],
                     np * 256);
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 4: integer forms. Problems hold up to 32 slots.

struct IProblem {
    uint32_t a[16][32], b[16][32], c[16][16];
};

static int int_k(int f) { return f == arith::IU4K32 ? 32 : 16; }

static std::vector<uint32_t> run_int(int f, int mod, const std::vector<IProblem> & ps) {
    std::vector<uint32_t> out(ps.size() * 256);
    const size_t CH = 8192;
    for (size_t base = 0; base < ps.size(); base += CH) {
        const int n = (int) std::min(CH, ps.size() - base);
        arith::Batch bt;
        bt.mod = mod;
        bt.init(f, n, 1);
        for (int w = 0; w < n; ++w) {
            const IProblem & p = ps[base + w];
            for (int r = 0; r < 16; ++r) {
                for (int s = 0; s < int_k(f); ++s) {
                    bt.set_a(w, 0, r, s, p.a[r][s]);
                    bt.set_b(w, 0, r, s, p.b[r][s]);
                }
                for (int c = 0; c < 16; ++c) {
                    bt.set_c(w, r, c, p.c[r][c]);
                }
            }
        }
        arith::run(bt);
        for (int w = 0; w < n; ++w) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    out[(base + w) * 256 + r * 16 + c] = bt.get_d(w, r, c);
                }
            }
        }
    }
    return out;
}

// kind 0: all codes uniform, C uniform; 1: extremes (largest and most negative codes, C near the 32-bit limits);
// 2: small C and codes, so sums cross zero; 3: C near a limit and a run of slots (start and length random) pushing
// past it with the opposite sign elsewhere, which separates saturation after each step from saturation at the end.
static IProblem make_int(int f, int kind, std::mt19937_64 & rng) {
    IProblem p;
    const int w = arith::elem_bits(f), k = int_k(f);
    const uint32_t m = (1u << w) - 1;
    const uint32_t ext[] = {m, m >> 1, (m >> 1) + 1, 0, 1, m - 1};
    for (int r = 0; r < 16; ++r) {
        const int s0 = (int) (rng() % k), len = 1 + (int) (rng() % k);
        for (int s = 0; s < k; ++s) {
            if (kind == 3) {
                // A: largest positive (signed view) inside the run, most negative outside; B: largest positive
                const bool in = ((s - s0 + k) % k) < len;
                p.a[r][s] = in ? (m >> 1) : (m >> 1) + 1;
                p.b[r][s] = (rng() & 3) ? (m >> 1) : (uint32_t) rng() & m;
                continue;
            }
            if (kind == 1) {
                p.a[r][s] = ext[rng() % 6];
                p.b[r][s] = ext[rng() % 6];
            } else {
                p.a[r][s] = (uint32_t) rng() & m;
                p.b[r][s] = (uint32_t) rng() & m;
            }
        }
        for (int c = 0; c < 16; ++c) {
            const uint32_t big[] = {0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x00000000u, 0x7FF00000u, 0x80100000u};
            if (kind == 3) {
                const int64_t lim = (rng() & 1) ? INT32_MAX : INT32_MIN;
                p.c[r][c] = (uint32_t) (int32_t) (lim - (lim > 0 ? 1 : -1) * (int64_t) (rng() % (uint64_t) (64 << (2 * w))));
            } else if (kind == 1) {
                p.c[r][c] = big[rng() % 6] + (uint32_t) ((int) (rng() % 4000001) - 2000000);
            } else if (kind == 2) {
                p.c[r][c] = (uint32_t) ((int) (rng() % 2001) - 1000);
            } else {
                p.c[r][c] = (uint32_t) rng();
            }
        }
    }
    return p;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 3: sparse forms. sparsemap: one-hot experiments; A row r holds 1 in stored slot j = r, B holds a distinct
// value in each of its 32 slots, so D[r][c] names the B slot that stored element j was multiplied with.

static bool s_bf16(int sf) { return sf == arith::S_BF16 || sf == arith::S_BF16H; }
static bool s_h(int sf) { return sf == arith::S_F16H || sf == arith::S_BF16H; }

static uint32_t sb_value(int sf, int t) {
    if (sf == arith::S_IU8) {
        return (uint32_t) t + 1;
    }
    if (sf == arith::S_FF || sf == arith::S_BF) {
        return (uint32_t) (((t >> 3) + 7) << 3 | (t & 7));  // E4M3 B
    }
    if (sf == arith::S_FB || sf == arith::S_BB) {
        return (uint32_t) (((t >> 2) + 15) << 2 | (t & 3));  // E5M2 B
    }
    return s_bf16(sf) ? 0x3F80u + (uint32_t) t : 0x3C00u + (uint32_t) t;
}
static uint32_t sa_one(int sf) {
    if (sf == arith::S_IU8) {
        return 1;
    }
    if (sf == arith::S_FF || sf == arith::S_FB) {
        return 0x38u;
    }
    if (sf == arith::S_BF || sf == arith::S_BB) {
        return 0x3Cu;
    }
    return s_bf16(sf) ? 0x3F80u : 0x3C00u;
}

// B slot named by a result word (-1: zero, -2: no single slot)
static int slot_of(int sf, uint32_t d) {
    if (d == 0) {
        return -1;
    }
    if (sf == arith::S_IU8) {
        return d >= 1 && d <= 32 ? (int) d - 1 : -2;
    }
    const am::Fmt fb = am::fmt_b(s_h(sf) ? (s_bf16(sf) ? am::F_BF16 : am::F_F16) : arith::sdense(sf));
    const double v = sf == arith::S_F16H ? am::value(am::F16, d) : sf == arith::S_BF16H ? am::value(am::BF16, d) : am::value(am::F32, d);
    for (int t = 0; t < 32; ++t) {
        if (am::value(fb, sb_value(sf, t)) == v) {
            return t;
        }
    }
    return -2;
}

// Runs index values V (same in every lane) through the one-hot problem; out[v][j] = B slot.
static std::vector<std::vector<int>> sparse_onehot(int sf, int key, const std::vector<uint32_t> & V) {
    arith::SBatch bt;
    bt.init(sf, key, (int) V.size());
    for (size_t w = 0; w < V.size(); ++w) {
        for (int r = 0; r < 16; ++r) {
            bt.set_a((int) w, r, r, sa_one(sf));
            for (int t = 0; t < 32; ++t) {
                bt.set_b((int) w, r, t, sb_value(sf, t));
            }
        }
        for (int l = 0; l < 32; ++l) {
            bt.idx[w * 32 + l] = V[w];
        }
    }
    arith::srun(bt);
    std::vector<std::vector<int>> out(V.size(), std::vector<int>(16));
    for (size_t w = 0; w < V.size(); ++w) {
        for (int j = 0; j < 16; ++j) {
            out[w][j] = slot_of(sf, bt.get_d((int) w, j, (j + 5) % 16));
        }
    }
    return out;
}

static bool mode_sparsemap(std::string & text) {
    std::mt19937_64 rng(4242);
    std::vector<uint32_t> V = {0u, 0xFFFFFFFFu, 0x44444444u, 0xEEEEEEEEu, 0xD8D8D8D8u};
    for (int i = 0; i < 59; ++i) {
        V.push_back((uint32_t) rng());
    }
    for (int sf = 0; sf < arith::NSFORMS; ++sf) {
        for (int key = 0; key < 2; ++key) {
            const auto out = sparse_onehot(sf, key, V);
            for (int j = 0; j < 16; ++j) {
                // which 2-bit field of V decides the slot of stored element j
                std::string fits;
                for (int p = 0; p < 16; ++p) {
                    int tab[4] = {-9, -9, -9, -9};
                    bool ok = true;
                    for (size_t w = 0; w < V.size() && ok; ++w) {
                        const int fv = (V[w] >> (2 * p)) & 3;
                        if (tab[fv] == -9) {
                            tab[fv] = out[w][j];
                        }
                        ok = tab[fv] == out[w][j];
                    }
                    if (ok) {
                        char buf[80];
                        std::snprintf(buf, sizeof buf, " field %d (bits %d-%d): %d %d %d %d", p, 2 * p, 2 * p + 1, tab[0], tab[1],
                                      tab[2], tab[3]);
                        fits += buf;
                    }
                }
                std::printf("RESULT sparsemap %-15s key %d stored %2d ->%s%s\n", arith::sform_name[sf], key, j,
                            fits.empty() ? " no single field fits; first slots:" : "", fits.c_str());
                if (fits.empty()) {
                    for (int w = 0; w < 6; ++w) {
                        std::printf("   V %08x -> %d\n", V[w], out[w][j]);
                    }
                }
            }
        }
    }
    // the rule (arith_model.h sparse_slot) against every one-hot result; --break uses the other index_key's fields
    bool ok = true;
    for (int sf = 0; sf < arith::NSFORMS; ++sf) {
        long bad = 0, n = 0;
        for (int key = 0; key < 2; ++key) {
            const auto out = sparse_onehot(sf, key, V);
            for (size_t w = 0; w < V.size(); ++w) {
                for (int j = 0; j < 16; ++j) {
                    bad += out[w][j] != am::sparse_slot(j, V[w], g_break ? key ^ 1 : key);
                    ++n;
                }
            }
        }
        ok = ok && bad == 0;
        add_text(text, "%s %ld/%ld one-hot slots as the rule", arith::sform_name[sf], n - bad, n);
    }
    return ok;
}

// Sparse problem: A stored slots, B's 32 slots per column, C, one index word per lane.
struct SProblem {
    ad::Problem p;       // a[r][0..15] stored A, c[r][c]; b[col][0..15] = B slots 0..15
    uint32_t b2[16][16];  // B slots 16..31
    uint32_t idx[32];
};

static int s_in_form(int sf) { return sf == arith::S_F16H ? am::F_F16 : sf == arith::S_BF16H ? am::F_BF16 : arith::sdense(sf); }

static std::vector<uint32_t> run_sparse(int sf, int key, const std::vector<SProblem> & ps) {
    std::vector<uint32_t> out(ps.size() * 256);
    const size_t CH = 8192;
    for (size_t base = 0; base < ps.size(); base += CH) {
        const int n = (int) std::min(CH, ps.size() - base);
        arith::SBatch bt;
        bt.init(sf, key, n);
        for (int w = 0; w < n; ++w) {
            const SProblem & q = ps[base + w];
            for (int r = 0; r < 16; ++r) {
                for (int j = 0; j < 16; ++j) {
                    bt.set_a(w, r, j, q.p.a[r][j]);
                    bt.set_b(w, r, j, q.p.b[r][j]);
                    bt.set_b(w, r, 16 + j, q.b2[r][j]);
                }
                for (int c = 0; c < 16; ++c) {
                    bt.set_c(w, r, c, q.p.c[r][c]);
                }
            }
            for (int l = 0; l < 32; ++l) {
                bt.idx[(size_t) w * 32 + l] = q.idx[l];
            }
        }
        arith::srun(bt);
        for (int w = 0; w < n; ++w) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    out[(base + w) * 256 + r * 16 + c] = bt.get_d(w, r, c);
                }
            }
        }
    }
    return out;
}

// The sparse model: select 16 B slots by the index, then the dense model. --break: the other index_key's fields.
static uint32_t sparse_model(int sf, int key, const SProblem & q, int r, int c) {
    uint32_t b32[32], bs[16];
    for (int j = 0; j < 16; ++j) {
        b32[j] = q.p.b[c][j];
        b32[16 + j] = q.b2[c][j];
    }
    am::sparse_select(b32, q.idx[r], q.idx[r + 16], g_break ? key ^ 1 : key, bs);
    const int form = s_in_form(sf);
    if (sf == arith::S_IU8) {
        return am::model_int(q.p.a[r], bs, 16, 8, true, true, q.p.c[r][c], 0);
    }
    if (s_h(sf)) {
        am::Params P = am::acc16_params(form, 3);
        P.rtz16 = sf == arith::S_BF16H ? 2 : 0;
        return am::model_acc16(form, q.p.a[r], bs, q.p.c[r][c], P);
    }
    return am::model_f32(form, q.p.a[r], bs, q.p.c[r][c], am::measured_params(form));
}

static SProblem make_sparse(int sf, int i, std::mt19937_64 & rng) {
    SProblem q;
    const int form = s_in_form(sf);
    if (sf == arith::S_IU8) {
        const IProblem ip = make_int(arith::IU8, i % 3, rng), ip2 = make_int(arith::IU8, (i + 1) % 3, rng);
        for (int r = 0; r < 16; ++r) {
            for (int j = 0; j < 16; ++j) {
                q.p.a[r][j] = ip.a[r][j];
                q.p.b[r][j] = ip.b[r][j];
                q.b2[r][j] = ip2.b[r][j];
            }
            for (int c = 0; c < 16; ++c) {
                q.p.c[r][c] = ip.c[r][c];
            }
        }
    } else if (s_h(sf)) {
        const int hf = sf == arith::S_F16H ? arith::F16H : arith::BF16H;
        q.p = make16(hf, i % 4, rng);
        const ad::Problem p2 = make16(hf, (i + 1) % 4, rng);
        std::memcpy(q.b2, p2.b, sizeof q.b2);
    } else {
        const int nd = form >= am::F_FF ? 5 : 4;
        q.p = ad::make(form, (ad::Dist) (i % nd), rng);
        const ad::Problem p2 = ad::make(form, (ad::Dist) ((i + 1) % nd), rng);
        std::memcpy(q.b2, p2.b, sizeof q.b2);
    }
    for (int l = 0; l < 32; ++l) {
        q.idx[l] = (uint32_t) rng();  // every value, including the ones the documents call invalid
    }
    return q;
}

static bool mode_sparse(long n, std::string & text) {
    bool ok = true;
    for (int sf = 0; sf < arith::NSFORMS; ++sf) {
        for (int key = 0; key < 2; ++key) {
            // random: model against card
            const long np = (n / 2 + 255) / 256;
            std::mt19937_64 rng(600 + sf * 2 + key);
            std::vector<SProblem> ps(np);
            for (long i = 0; i < np; ++i) {
                ps[i] = make_sparse(sf, (int) i, rng);
            }
            const std::vector<uint32_t> d = run_sparse(sf, key, ps);
            std::vector<long> bad(8, 0);
            std::vector<std::string> first(8);
            par_for(ps.size(), [&](size_t i, int t) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        const uint32_t want = sparse_model(sf, key, ps[i], r, c), got = d[i * 256 + r * 16 + c];
                        if (want != got) {
                            if (!bad[t]) {
                                char buf[120];
                                std::snprintf(buf, sizeof buf, "problem %zu (%d,%d): card %08x model %08x", i, r, c, got, want);
                                first[t] = buf;
                            }
                            ++bad[t];
                        }
                    }
                }
            });
            long nb = 0;
            std::string f1;
            for (int t = 0; t < 8; ++t) {
                nb += bad[t];
                if (f1.empty()) {
                    f1 = first[t];
                }
            }
            std::printf("RESULT sparse %-15s key %d random elements %ld model_mismatches %ld%s%s\n", arith::sform_name[sf], key,
                        np * 256, nb, nb ? " first: " : "", f1.c_str());
            ok = ok && nb == 0;
            add_text(text, "%s key %d random %ld/%ld", arith::sform_name[sf], key, np * 256 - nb, np * 256);
            // the same 16 products through the dense instruction and the sparse one (adversarial data; the
            // unselected B slots hold unrelated values): card against card
            const int dense = arith::sdense(sf);
            const long na = (n / 8 + 255) / 256;
            std::vector<ad::Problem> pd(na);
            std::vector<SProblem> pa(na);
            for (long i = 0; i < na; ++i) {
                ad::Problem p;
                if (sf == arith::S_IU8) {
                    const IProblem ip = make_int(arith::IU8, (int) (i % 4), rng);
                    for (int r = 0; r < 16; ++r) {
                        std::memcpy(p.a[r], ip.a[r], sizeof p.a[r]);
                        std::memcpy(p.b[r], ip.b[r], sizeof p.b[r]);
                        std::memcpy(p.c[r], ip.c[r], sizeof p.c[r]);
                    }
                } else if (s_h(sf)) {
                    const int hf = sf == arith::S_F16H ? arith::F16H : arith::BF16H;
                    if (i & 1) {
                        p = sf == arith::S_F16H ? ad::adversarial(am::F_F16, rng, 7, 7) : ad::adversarial(am::F_BF16, rng);
                        for (int r = 0; r < 16; ++r) {
                            for (int c = 0; c < 16; ++c) {
                                p.c[r][c] = conv16(hf, p.c[r][c]);
                            }
                        }
                    } else {
                        p = make16(hf, (int) (i % 4), rng);
                    }
                } else {
                    p = ad::adversarial(dense, rng);
                }
                pd[i] = p;
                SProblem & q = pa[i];
                q.p = p;
                // a valid index (the two stored elements of each pair name different slots) shared by all rows,
                // so every column can hold b[c][j] at the slot the index names
                uint32_t half[2] = {0, 0};
                for (int h = 0; h < 2; ++h) {
                    for (int pr = 0; pr < 4; ++pr) {
                        const uint32_t x = (uint32_t) (rng() % 4), y = (x + 1 + (uint32_t) (rng() % 3)) % 4;
                        half[h] |= (x | y << 2) << (4 * pr);
                    }
                }
                for (int l = 0; l < 32; ++l) {
                    q.idx[l] = (uint32_t) rng() & ~(0xFFFFu << (16 * key));
                    q.idx[l] |= half[l >> 4] << (16 * key);
                }
                for (int c = 0; c < 16; ++c) {
                    uint32_t b32[32];
                    for (int t = 0; t < 32; ++t) {
                        b32[t] = (uint32_t) rng() & (arith::elem_bits(dense) == 8 ? 0x77u : 0x3FFFu);  // unrelated
                    }
                    for (int j = 0; j < 16; ++j) {
                        b32[am::sparse_slot(j, j < 8 ? q.idx[0] : q.idx[16], key)] = p.b[c][j];
                    }
                    for (int j = 0; j < 16; ++j) {
                        q.p.b[c][j] = b32[j];
                        q.b2[c][j] = b32[16 + j];
                    }
                }
            }
            const std::vector<uint32_t> dd = run_problems(dense, pd, sf == arith::S_IU8 ? arith::M_SAB : 0);
            const std::vector<uint32_t> ds = run_sparse(sf, key, pa);
            const long used = na * 256, diff = count_diff(dd, ds, (size_t) used);
            std::printf("RESULT sparse %-15s key %d dense_vs_sparse elements %ld differ %ld\n",
                        arith::sform_name[sf], key, used, diff);
            ok = ok && diff == 0 && used > 0;
            add_text(text, "dense=sparse %ld/%ld", used - diff, used);
        }
    }
    return ok;
}

static bool int_signed_a(int mod) { return mod == arith::M_NEGA || mod == arith::M_SAB || mod == arith::M_SABC || mod == arith::M_SAC; }
static bool int_signed_b(int mod) { return mod == arith::M_NEGB || mod == arith::M_SAB || mod == arith::M_SABC || mod == arith::M_SBC; }
static bool int_clamp(int mod) { return mod >= arith::M_CLAMP; }

static Tally compare_int(int f, int mod, const std::vector<IProblem> & ps, const std::vector<uint32_t> & d, int clamp_mode,
                         int group) {
    std::vector<Tally> part(8);
    const int w = arith::elem_bits(f), k = int_k(f);
    const bool sa = int_signed_a(mod), sb = int_signed_b(mod);
    par_for(ps.size(), [&](size_t i, int t) {
        Tally & ta = part[t];
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                const uint32_t want = am::model_int(ps[i].a[r], ps[i].b[c], k, w, sa, sb, ps[i].c[r][c], clamp_mode, group);
                const uint32_t got = d[i * 256 + r * 16 + c];
                ++ta.n;
                if (want != got) {
                    if (!ta.bad) {
                        char buf[160];
                        int64_t ex = (int64_t) (int32_t) ps[i].c[r][c];
                        for (int s = 0; s < k; ++s) {
                            ex += am::int_val(ps[i].a[r][s], w, sa) * am::int_val(ps[i].b[c][s], w, sb);
                        }
                        std::snprintf(buf, sizeof buf, "problem %zu (%d,%d): C %08x exact %lld card %08x model %08x", i, r, c,
                                      ps[i].c[r][c], (long long) ex, got, want);
                        ta.first = buf;
                    }
                    ++ta.bad;
                }
            }
        }
    });
    Tally all;
    for (auto & t : part) {
        all.n += t.n;
        all.bad += t.bad;
        if (all.first.empty()) {
            all.first = t.first;
        }
    }
    return all;
}

// Measured: without the clamp bit an exact sum wrapped to 32 bits; with it, the running sum (C first) saturated to
// [INT32_MIN, INT32_MAX] after each step, a step being one lane half of A and B (8 slots for IU8 and IU4 with
// K = 16, 16 slots for IU4 with K = 32), also when both inputs are unsigned. --break: saturation only at the end.
static bool mode_int(long n, std::string & text) {
    bool ok = true;
    long ctl_final = 0, ctl_uns = 0, ctl_n = 0;
    for (int f : {arith::IU8, arith::IU4, arith::IU4K32}) {
        const int group = int_k(f) / 2;
        for (int mod : {arith::M_NONE, arith::M_NEGA, arith::M_NEGB, arith::M_SAB, arith::M_CLAMP, arith::M_SAC, arith::M_SBC,
                        arith::M_SABC}) {
            const long np = (n + 255) / 256;
            std::mt19937_64 rng(900 + f * 16 + mod);
            std::vector<IProblem> ps(np);
            for (long i = 0; i < np; ++i) {
                ps[i] = make_int(f, (int) (i % 4), rng);
            }
            const std::vector<uint32_t> d = run_int(f, mod, ps);
            const int cm = int_clamp(mod) ? (g_break ? 1 : 2) : 0;
            const Tally t = compare_int(f, mod, ps, d, cm, group);
            long sat = 0;
            for (uint32_t v : d) {
                sat += v == 0x7FFFFFFFu || v == 0x80000000u;
            }
            std::printf("RESULT int %-8s %-9s elements %ld model_mismatches %ld outputs_at_a_32bit_limit %ld%s%s\n",
                        arith::form_name[f], arith::mod_name[mod], t.n, t.bad, sat, t.bad ? " first: " : "", t.first.c_str());
            ok = ok && t.bad == 0;
            add_text(text, "%s %s %ld/%ld", arith::form_name[f], arith::mod_name[mod], t.n - t.bad, t.n);
            if (int_clamp(mod) && !g_break) {
                // controls: saturation once at the end; unsigned bounds for unsigned x unsigned
                const Tally t1 = compare_int(f, mod, ps, d, 1, group), t3 = compare_int(f, mod, ps, d, 3, group);
                std::printf("RESULT int %-8s %-9s control: saturated once at the end %ld, unsigned bounds %ld of %ld\n",
                            arith::form_name[f], arith::mod_name[mod], t1.bad, t3.bad, t1.n);
                ctl_final += t1.bad;
                ctl_uns += t3.bad;
                ctl_n += t1.n;
            } else if (!int_clamp(mod) && !g_break) {
                const Tally t1 = compare_int(f, mod, ps, d, 1, group);
                std::printf("RESULT int %-8s %-9s control: saturated (no clamp bit) %ld of %ld\n", arith::form_name[f],
                            arith::mod_name[mod], t1.bad, t1.n);
                ok = ok && t1.bad > 0;
            }
        }
    }
    if (!g_break) {
        std::printf("RESULT int controls over all clamp settings: saturated once at the end %ld, unsigned bounds %ld of %ld\n",
                    ctl_final, ctl_uns, ctl_n);
        ok = ok && ctl_final > 0 && ctl_uns > 0;
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 5: chains through C for the BF8 forms (the model chained against the card).
static bool mode_chain8(std::string & text) {
    bool ok = true;
    for (int form : {arith::FB, arith::BF, arith::BB}) {
        for (int steps : {4, 64, 256}) {
            const int waves = steps == 256 ? 32 : 128;
            std::mt19937_64 rng(steps * 11 + form);
            std::vector<ad::Problem> ps((size_t) waves * steps);
            for (size_t i = 0; i < ps.size(); ++i) {
                ps[i] = ad::make(form, (ad::Dist) ((i / steps + i) % 5 == 4 ? ad::ALLCODES : (i / steps + i) % 2), rng);
            }
            const std::vector<uint32_t> d = run_problems(form, ps, 0, steps);
            am::Params P = am::measured_params(form);
            if (g_break) {
                P.width += 1;
            }
            const Tally t = compare(ps, steps, d, [&](const ad::Problem * p, int st, int r, int c) {
                uint32_t acc = p[0].c[r][c];
                for (int s = 0; s < st; ++s) {
                    acc = am::model_f32(form, p[s].a[r], p[s].b[c], acc, P);
                }
                return acc;
            });
            std::printf("RESULT chain8 %-8s K %5d elements %ld model_mismatches %ld%s%s\n", arith::form_name[form], steps * 16, t.n,
                        t.bad, t.bad ? " first: " : "", t.first.c_str());
            ok = ok && t.bad == 0;
            add_text(text, "%s K%d %ld/%ld", arith::form_name[form], steps * 16, t.n - t.bad, t.n);
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Wide-range acceptance for the eight float forms: any finite bit pattern as input, a share of subnormal inputs, a
// subnormal or full-range accumulator, one to sixteen slots in use. The other random sets keep BF16 exponents
// within +-60, so no step ever works below the FP32 normal range; this set is what found the 2^-152 floor of the
// unit and the -0 of the BF16 accumulator form. --break: the model without that floor (and, for the 16-bit
// accumulator forms, with the wrong rounding).

static uint32_t any_finite(am::Fmt f, std::mt19937_64 & rng, int sub_per_64) {
    const am::FmtInfo fi = am::info(f);
    const int bits = fi.ebits + fi.mbits + 1;
    for (;;) {
        uint32_t v = (uint32_t) (rng() & ((1ull << bits) - 1));
        if ((int) (rng() & 63) < sub_per_64) {
            v &= ~(((1u << fi.ebits) - 1) << fi.mbits);
        }
        const am::Dec d = am::decode(f, v);
        if (!d.nan && !d.inf) {
            return v;
        }
    }
}

static ad::Problem make_wide(am::Fmt fa, am::Fmt fb, am::Fmt fc, int sub_per_64, std::mt19937_64 & rng) {
    static const int used[6] = {1, 2, 3, 5, 8, 16};
    ad::Problem p{};
    for (int r = 0; r < 16; ++r) {
        const int n = used[rng() % 6];
        for (int k = 0; k < n; ++k) {
            p.a[r][n == 16 ? k : (int) (rng() & 15)] = any_finite(fa, rng, sub_per_64);
        }
        for (int s = 0; s < 16; ++s) {
            p.b[r][s] = (rng() & 15) ? any_finite(fb, rng, sub_per_64) : 0u;
        }
        for (int c = 0; c < 16; ++c) {
            const int k = (int) (rng() % 10);
            p.c[r][c] = k < 2 ? 0u : any_finite(fc, rng, k < 6 ? 32 : 0);
        }
    }
    return p;
}

static bool mode_wide(long n, std::string & text) {
    bool ok = true;
    const long np = (n + 255) / 256;
    for (int f : {arith::F16, arith::BF16, arith::FF, arith::FB, arith::BF, arith::BB, arith::F16H, arith::BF16H}) {
        const bool h16 = f == arith::F16H || f == arith::BF16H;
        const int mform = h16 ? in_form(f) : f;
        const am::Fmt fc = !h16 ? am::F32 : f == arith::F16H ? am::F16 : am::BF16;
        ElemModel M;
        if (h16) {
            M = measured16(f);
        } else {
            am::Params P = am::measured_params(mform);
            if (g_break) {
                P.min_unit = -100000;
            }
            M = [=](const ad::Problem * p, int, int r, int c) { return am::model_f32(mform, p[0].a[r], p[0].b[c], p[0].c[r][c], P); };
        }
        for (int sub : {26, 0}) {
            std::mt19937_64 rng(4242 + 16 * f + sub);
            std::vector<ad::Problem> ps(np);
            for (long i = 0; i < np; ++i) {
                ps[i] = make_wide(am::fmt_a(mform), am::fmt_b(mform), fc, sub, rng);
            }
            const std::vector<uint32_t> d = run_problems(f, ps);
            const Tally t = compare(ps, 1, d, M);
            std::printf("RESULT wide %-9s subnormal inputs %2d/64 elements %ld model_mismatches %ld%s%s\n", arith::form_name[f], sub,
                        t.n, t.bad, t.bad ? " first: " : "", t.first.c_str());
            ok = ok && t.bad == 0;
            add_text(text, "%s sub%d %ld/%ld", arith::form_name[f], sub, t.n - t.bad, t.n);
            if (g_break || (f != arith::BF16 && f != arith::BF16H)) {
                continue;
            }
            // controls on the same card outputs: each rule this set found, changed by the smallest step, must
            // disagree with the card
            if (f == arith::BF16) {
                for (int mu : {-151, -153}) {
                    am::Params P = am::measured_params(mform);
                    P.min_unit = mu;
                    const Tally tc = compare(ps, 1, d, [=](const ad::Problem * p, int, int r, int c) {
                        return am::model_f32(mform, p[0].a[r], p[0].b[c], p[0].c[r][c], P);
                    });
                    std::printf("RESULT wide %-9s subnormal inputs %2d/64 control smallest unit 2^%d: mismatches %ld of %ld\n",
                                arith::form_name[f], sub, mu, tc.bad, tc.n);
                    ok = ok && tc.bad > 0;
                }
            } else {
                am::Params P = am::acc16_params(mform, 3);
                P.rtz16 = 2;
                P.neg_zero16 = false;
                const Tally tc = compare(ps, 1, d, [=](const ad::Problem * p, int, int r, int c) {
                    return am::model_acc16(mform, p[0].a[r], p[0].b[c], p[0].c[r][c], P);
                });
                std::printf("RESULT wide %-9s subnormal inputs %2d/64 control without the minus-zero rule: mismatches %ld of %ld\n",
                            arith::form_name[f], sub, tc.bad, tc.n);
                ok = ok && tc.bad > 0;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Edge cases quoted in measurements/matrix-arithmetic.md, run on the card so that the quoted bits are in a log:
// the review's counterexamples to the first exactness rule and error bound (subnormal FP16 inputs), the subnormal
// FP32 running sum, a product in the FP32 subnormal range, the -0 of the BF16 accumulator form; then a sweep of
// single BF16 products across the FP32 subnormal range, which fixes the smallest unit at 2^-152. --break: the
// expected bits of the first case are wrong.

static uint32_t bf16_of(int sign, int e, int m) { return ((uint32_t) sign << 15) | ((uint32_t) (e + 127) << 7) | (uint32_t) m; }

static bool mode_edge(std::string & text) {
    struct Case {
        const char * name;
        int form;
        uint32_t a[2], b[2], c, want;
        int slot;
    };
    const Case cases[] = {
        {"FP16 subnormal A, on the grid but not exact (exact sum 0x33000001)", arith::F16, {0x0001, 0x0001}, {0x3800, 0x0001}, 0, 0x33000000u, 0},
        {"FP16 subnormal A, error 2^-14 of the magnitudes (exact sum 0x3AFFFFF8)", arith::F16, {0x0001, 0x8001}, {0x7800, 0x2400}, 0, 0x3AFFFC00u, 0},
        {"BF16 2^-51 x 2^-51 with a negative subnormal FP32 C (exact sum 0x0C800000)", arith::BF16, {0x2600, 0}, {0x2600, 0}, 0x80400000u, 0x0C7FFFFFu, 0},
        {"BF16 2^-51 x 2^-51 with a positive subnormal FP32 C", arith::BF16, {0x2600, 0}, {0x2600, 0}, 0x00400000u, 0x0C800000u, 0},
        {"BF16 single product of -49.43 units of 2^-149 (nearest would be 0x80000031)", arith::BF16, {0xB7AB, 0}, {0x0025, 0}, 0, 0x80000032u, 0},
        {"BF16 into BF16, tiny negative product in the last step: -0", arith::BF16H, {0xB5AD, 0}, {0x006F, 0}, 0, 0x00008000u, 12},
    };
    bool ok = true;
    int ci = 0;
    for (const Case & cs : cases) {
        const bool h16 = cs.form == arith::BF16H;
        const int mform = h16 ? in_form(cs.form) : cs.form;
        std::vector<ad::Problem> ps(1);
        ps[0] = ad::Problem{};
        for (int k = 0; k < 2; ++k) {
            ps[0].a[0][cs.slot + k] = cs.a[k];
            ps[0].b[0][cs.slot + k] = cs.b[k];
        }
        ps[0].c[0][0] = cs.c;
        const std::vector<uint32_t> d = run_problems(cs.form, ps);
        const uint32_t model = h16 ? measured16(cs.form)(&ps[0], 1, 0, 0)
                                   : am::model_f32(mform, ps[0].a[0], ps[0].b[0], cs.c, am::measured_params(mform));
        const uint32_t want = (g_break && ci == 0) ? cs.want ^ 1u : cs.want;
        const bool good = d[0] == model && d[0] == want;
        std::printf("RESULT edge case %d %s: card 0x%08x model 0x%08x quoted 0x%08x %s\n", ci, cs.name, d[0], model, want,
                    good ? "ok" : "MISMATCH");
        ok = ok && good;
        ++ci;
    }
    // sweep: one BF16 product per element, (1 + ma/128) x (1 + mb/128) x 2^esum, both signs
    std::vector<ad::Problem> ps;
    for (int esum : {-140, -142, -144, -146, -148, -149, -150, -151, -152, -153, -156}) {
        for (int sign = 0; sign < 2; ++sign) {
            for (int blk = 0; blk < 8; ++blk) {
                ad::Problem p{};
                for (int r = 0; r < 16; ++r) {
                    p.a[r][0] = bf16_of(sign, -70, blk * 16 + r);
                    p.b[r][0] = bf16_of(0, esum + 70, (r * 8 + blk * 5) & 127);
                }
                ps.push_back(p);
            }
        }
    }
    const std::vector<uint32_t> d = run_problems(arith::BF16, ps);
    std::string line;
    for (int mu : {-152, -151, -153, -100000}) {
        am::Params P = am::measured_params(am::F_BF16);
        P.min_unit = mu;
        const Tally t = compare(ps, 1, d, [=](const ad::Problem * p, int, int r, int c) {
            return am::model_f32(am::F_BF16, p[0].a[r], p[0].b[c], p[0].c[r][c], P);
        });
        char buf[96];
        if (mu == -100000) {
            std::snprintf(buf, sizeof buf, " no smallest unit: %ld;", t.bad);
        } else {
            std::snprintf(buf, sizeof buf, " smallest unit 2^%d: %ld;", mu, t.bad);
        }
        line += buf;
        ok = ok && (mu == -152 ? t.bad == 0 : t.bad > 0);
        if (mu == -152) {
            add_text(text, "sweep %ld/%ld", t.n - t.bad, t.n);
        }
    }
    std::printf("RESULT edge sweep: %zu single BF16 products from 2^-156 to 2^-140, mismatches by model:%s\n", ps.size() * 256,
                line.c_str());
    add_text(text, "%d quoted cases %s", ci, ok ? "as quoted" : "NOT as quoted");
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 6: the forbidden sequence. For every wave the second instruction's result is matched against the model
// fed with candidate operand images: D1 as written (correct), the stale contents (INIT), and every mix of the four
// D1 registers between new and stale.

static uint32_t f16_rand(std::mt19937_64 & rng) { return ad::rand_val(am::F16, rng, -3, 3, 1, 1); }

static bool mode_hazard(std::string & text) {
    bool ok = true;
    const int waves = 64;
    std::mt19937_64 rng(77);
    std::vector<uint32_t> in4[5], c1((size_t) waves * 256);
    for (auto & v : in4) {
        v.resize((size_t) waves * 128);
    }
    for (int i = 0; i < 4; ++i) {
        for (auto & x : in4[i]) {
            x = f16_rand(rng) | f16_rand(rng) << 16;
        }
    }
    for (auto & x : in4[4]) {
        x = 0x3C003C00u ^ ((uint32_t) rng() & 0x01FF01FFu);  // stale contents: FP16 values near 1, all distinct-ish
    }
    for (auto & x : c1) {
        x = ad::rand_f32(rng, -4, 4);
    }
    // the model of the second instruction for a given operand image of D1 (4 words per lane)
    auto second = [&](int w, bool as_b, const uint32_t * img, int r, int c) {
        uint32_t a[16], b[16];
        for (int s = 0; s < 16; ++s) {
            const arith::Slot sa = arith::slot_ab(arith::F16, r, s), sb = arith::slot_ab(arith::F16, c, s);
            const uint32_t * x1 = &in4[3][(size_t) w * 128];
            a[s] = arith::get_bits(as_b ? x1 : img, sa, 4, 16);
            b[s] = arith::get_bits(as_b ? img : x1, sb, 4, 16);
        }
        const arith::Slot sc = arith::slot_cd(arith::F16, r, c);
        return am::model_f32(am::F_F16, a, b, c1[(size_t) w * 256 + sc.lane * 8 + sc.word], am::measured_params(am::F_F16));
    };
    for (int as_b = 0; as_b < 2; ++as_b) {
        std::vector<uint32_t> d1n, d2n, d1, d2;
        arith::hz_run(as_b * 2 + 1, waves, in4, c1, d1n, d2n);  // with the V_NOP
        arith::hz_run(as_b * 2, waves, in4, c1, d1, d2);        // nothing between
        long diff = 0, d1diff = 0, nop_model_bad = 0;
        std::vector<long> combo(17, 0);  // index: bit k set = register k read new; 16: no combination fits
        std::vector<std::vector<long>> lanefit(16, std::vector<long>(257, 0));
        for (int w = 0; w < waves; ++w) {
            for (size_t i = 0; i < 256; ++i) {
                diff += d2[(size_t) w * 256 + i] != d2n[(size_t) w * 256 + i];
            }
            for (size_t i = 0; i < 128; ++i) {
                d1diff += d1[(size_t) w * 128 + i] != d1n[(size_t) w * 128 + i];
            }
            const uint32_t * dnew = &d1n[(size_t) w * 128];
            const uint32_t * dold = &in4[4][(size_t) w * 128];
            auto fits = [&](int mask, const std::vector<uint32_t> & dd2) {
                uint32_t img[128];
                for (int l = 0; l < 32; ++l) {
                    for (int k = 0; k < 4; ++k) {
                        img[l * 4 + k] = (mask >> k & 1) ? dnew[l * 4 + k] : dold[l * 4 + k];
                    }
                }
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        const arith::Slot sc = arith::slot_cd(arith::F16, r, c);
                        if (second(w, as_b, img, r, c) != dd2[(size_t) w * 256 + sc.lane * 8 + sc.word]) {
                            return false;
                        }
                    }
                }
                return true;
            };
            nop_model_bad += !fits(15, d2n);
            // per row (A) or column (B) of the second instruction: which of its 8 registers (2 lanes x 4 VGPRs)
            // were read new; 256 = no mix of new and stale fits
            for (int q = 0; q < 16; ++q) {
                int fm = 256;
                for (int mask = 255; mask >= 0 && fm == 256; --mask) {
                    uint32_t img[128];
                    for (int l = 0; l < 32; ++l) {
                        for (int k = 0; k < 4; ++k) {
                            const bool mine = (l & 15) == q;
                            const bool nw = !mine || (mask >> ((l >> 4) * 4 + k) & 1);
                            img[l * 4 + k] = nw ? dnew[l * 4 + k] : dold[l * 4 + k];
                        }
                    }
                    bool all = true;
                    for (int o = 0; o < 16 && all; ++o) {
                        const int r = as_b ? o : q, c = as_b ? q : o;
                        const arith::Slot sc = arith::slot_cd(arith::F16, r, c);
                        all = second(w, as_b, img, r, c) == d2[(size_t) w * 256 + sc.lane * 8 + sc.word];
                    }
                    if (all) {
                        fm = mask;
                    }
                }
                ++lanefit[q][fm];
            }
            int found = 16;
            for (int mask = 15; mask >= 0 && found == 16; --mask) {
                if (fits(mask, d2)) {
                    found = mask;
                }
            }
            ++combo[found];
        }
        for (int q = 0; q < 16; ++q) {
            std::string one;
            for (int m = 0; m <= 256; ++m) {
                if (lanefit[q][m]) {
                    char buf[64];
                    if (m == 256) {
                        std::snprintf(buf, sizeof buf, "%sno mix fits x%ld", one.empty() ? "" : ", ", lanefit[q][m]);
                    } else {
                        std::snprintf(buf, sizeof buf, "%smask 0x%02x x%ld", one.empty() ? "" : ", ", m, lanefit[q][m]);
                    }
                    one += buf;
                }
            }
            std::printf("RESULT hazard D1 as %s, %s %2d (lanes %d and %d): registers read new (bit k: VGPR k of lane %d, "
                        "bit 4 + k: of lane %d; 0xff all new): %s\n", as_b ? "B" : "A",
                        as_b ? "column" : "row", q, q, q + 16, q, q + 16, one.c_str());
        }
        std::string hist;
        for (int m = 0; m <= 16; ++m) {
            if (combo[m]) {
                char buf[80];
                std::snprintf(buf, sizeof buf, " %s:%ld", m == 16 ? "no-mix-fits" : m == 15 ? "all-new" : m == 0 ? "all-stale" : std::to_string(m).c_str(),
                              combo[m]);
                hist += buf;
            }
        }
        std::printf("RESULT hazard D1 as %s: elements %d, D2 differs from the V_NOP version %ld, D1 differs %ld, model on the "
                    "V_NOP version wrong in %ld waves; waves by registers read new (bit k = VGPR k):%s\n",
                    as_b ? "B" : "A", waves * 256, diff, d1diff, nop_model_bad, hist.c_str());
        // PASS means the probe worked: the V_NOP version equals the model in every wave (the observation itself is
        // reported, not judged); --break expects the forbidden version to equal the V_NOP version
        ok = ok && nop_model_bad == 0 && (!g_break || diff == 0);
        add_text(text, "D1 as %s: %ld of %d differ from the V_NOP version; waves by whole-register mix of new and stale:%s", as_b ? "B" : "A", diff, waves * 256, hist.c_str());
    }
    return ok;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    long n = 1000000;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--break") {
            g_break = true;
        } else if (a == "--dump" && i + 1 < argc) {
            g_dump = std::atoi(argv[++i]);
        } else if (a == "--n" && i + 1 < argc) {
            n = std::atol(argv[++i]);
        }
    }
    require_arch();
    std::string text;
    bool ok;
    if (mode == "acc16") {
        ok = mode_acc16(n, text);
    } else if (mode == "hazard") {
        ok = mode_hazard(text);
    } else if (mode == "chain8") {
        ok = mode_chain8(text);
    } else if (mode == "int") {
        ok = mode_int(n, text);
    } else if (mode == "sparse") {
        ok = mode_sparse(n, text);
    } else if (mode == "sparsemap") {
        ok = mode_sparsemap(text);
    } else if (mode == "mods") {
        ok = mode_mods(n, text);
    } else if (mode == "ovfl") {
        ok = mode_ovfl(text);
    } else if (mode == "wide") {
        ok = mode_wide(n, text);
    } else if (mode == "edge") {
        ok = mode_edge(text);
    } else {
        std::fprintf(stderr, "usage: arith_more acc16|ovfl|mods|sparsemap|sparse|int|chain8|hazard|wide|edge [--n N] [--break] [--dump K]\n");
        return 2;
    }
    verdict("arith_" + mode, ok, text);
    return ok ? 0 : 1;
}
