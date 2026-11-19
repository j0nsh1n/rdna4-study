// Lane 5: acceptance of the CPU model (arith_model.h) against the card, bit for bit, plus the controls.
//
// Usage: arith_check MODE [--n N] [--break]
//   selftest  CPU only: the exact reference against hand-worked cases; the model against hand-worked structure
//             cases; the model with one stage changed must disagree with the measured model on random data.
//   random    N random elements per form (default 10^7 for f16, bf16, fp8_fp8; 10^6 for the BF8 forms),
//             distributions narrow/wide/cancel/bigsmall/allcodes; the model must agree on 100 %. Reports how often
//             the card equals the exact sum, and how often each one-stage-changed model disagrees (must be visible).
//   adv       adversarial sets: cancelling big terms in every slot pair and C, small terms at every gap, sign and
//             slot; 100 % agreement. The slot-pair loss fingerprint must change under a slot permutation.
//   special   NaN, infinity, zero, subnormal, overflow and FP8 NaN-code cases in random slots; 100 % agreement.
//   determ    3 launches identical; problems moved to other waves, rows permuted, after a launch of another form:
//             no output bit changes.
//   chain     chains through C (K = 64, 1024, 4096) on the card against the model chained; 100 % agreement.
//   mode      MODE register: the VALU control must change with each setting (proves the write); the matrix
//             outputs under each setting are compared with the default MODE (reported; PASS needs the control).
//   --break   perturbs the model (unit one bit wider) or the expectation; the verdict must turn to FAIL.
#include "arith_data.h"
#include "arith_gpu.h"

#include <atomic>
#include <functional>
#include <string>
#include <thread>

static bool g_break = false;

static am::Params params_for(int form) {
    am::Params p = am::measured_params(form);
    if (g_break) {
        p.width += 1;
    }
    return p;
}

// Runs problems on the card in chunks; returns D[problem][r][c].
static std::vector<uint32_t> run_problems(int form, const std::vector<ad::Problem> & ps, int mod = 0,
                                          bool use_mode = false, uint32_t mode = 0) {
    std::vector<uint32_t> out(ps.size() * 256);
    const size_t CH = 8192;
    for (size_t base = 0; base < ps.size(); base += CH) {
        const int n = (int) std::min(CH, ps.size() - base);
        arith::Batch bt;
        bt.mod = mod;
        bt.init(form, n, 1);
        for (int w = 0; w < n; ++w) {
            const ad::Problem & p = ps[base + w];
            for (int r = 0; r < 16; ++r) {
                for (int s = 0; s < 16; ++s) {
                    bt.set_a(w, 0, r, s, p.a[r][s]);
                    bt.set_b(w, 0, r, s, p.b[r][s]);
                }
                for (int c = 0; c < 16; ++c) {
                    bt.set_c(w, r, c, p.c[r][c]);
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

// Parallel loop over problems (8 threads at most, the machine rule for CPU work).
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

struct Tally {
    long n = 0, bad = 0, exact_eq = 0;
    std::string first;
};

// Compares card outputs with the model (and counts equality with the exact sum).
static Tally compare(int form, const std::vector<ad::Problem> & ps, const std::vector<uint32_t> & d,
                     const am::Params & P) {
    std::vector<Tally> part(8);
    par_for(ps.size(), [&](size_t i, int t) {
        Tally & ta = part[t];
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                const uint32_t got = d[i * 256 + r * 16 + c];
                const uint32_t want = am::model_f32(form, ps[i].a[r], ps[i].b[c], ps[i].c[r][c], P);
                const uint32_t ex = am::exact_round(form, ps[i].a[r], ps[i].b[c], ps[i].c[r][c], am::F32);
                ++ta.n;
                ta.exact_eq += got == ex;
                if (got != want) {
                    if (!ta.bad) {
                        char buf[200];
                        std::snprintf(buf, sizeof buf, "problem %zu (%d,%d): card %08x model %08x exact %08x", i, r, c,
                                      got, want, ex);
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
        all.exact_eq += t.exact_eq;
        if (all.first.empty()) {
            all.first = t.first;
        }
    }
    return all;
}

static const char * fname(int f) { return arith::form_name[f]; }

// ---------------------------------------------------------------------------------------------------------------

static bool mode_selftest(std::string & text) {
    long bad = 0;
    auto expect = [&](const char * what, uint32_t got, uint32_t want) {
        if (got != want) {
            std::printf("  selftest %s: got %08x want %08x\n", what, got, want);
            ++bad;
        }
    };
    // exact reference, hand-worked (values written out): 1 + 2^-24 ties to even; 1 + 2^-24 + 2^-60 rounds up;
    // 2^30 - 2^30 + 2^-40; subnormal 1.5 * 2^-149 ties to 2^-148; (1, -1) exact zero is +0
    auto ex2 = [](double x, double y, double z) {
        am::Exact e;
        for (double v : {x, y, z}) {
            if (v != 0) {
                int k;
                const double fr = std::frexp(std::fabs(v), &k);
                e.add(v < 0, (uint64_t) std::ldexp(fr, 53), k - 53);
            }
        }
        return e.round(am::F32);
    };
    expect("1 + 2^-24", ex2(1, 0x1p-24, 0), 0x3F800000u);
    expect("1 + 2^-24 + 2^-60", ex2(1, 0x1p-24, 0x1p-60), 0x3F800001u);
    expect("2^30 - 2^30 + 2^-40", ex2(0x1p30, -0x1p30, 0x1p-40), 0x2B800000u);
    expect("1.5 * 2^-149", ex2(0x1.8p-149, 0, 0), 0x00000002u);
    expect("1 - 1", ex2(1, -1, 0), 0x00000000u);
    expect("-(2^128) overflows", ex2(-0x1p127, -0x1p127, 0), 0xFF800000u);
    expect("E4M3 of 464 (tie, to 448 even)", am::round_int(am::E4M3, 464, 0), 0x7Eu);
    expect("E4M3 of 466 (above the tie: NaN)", am::round_int(am::E4M3, 466, 0), 0x7Fu);
    expect("E4M3 of 466 saturating", am::round_int(am::E4M3, 466, 0, false, true), 0x7Eu);
    expect("FP16 of 65520 (tie to inf)", am::round_int(am::F16, 65520, 0), 0x7C00u);
    // model, hand-worked structure cases (FP16, B = 1): values from the experiments in matrix-arithmetic.md 2.
    const am::Params P = params_for(am::F_F16);
    auto f16 = [](double v) { return (uint32_t) __builtin_bit_cast(uint16_t, (_Float16) v); };
    auto run1 = [&](std::initializer_list<std::pair<int, double>> t, uint32_t c) {
        uint32_t a[16] = {}, b[16];
        for (int s = 0; s < 16; ++s) {
            b[s] = 0x3C00;
        }
        for (auto [s, v] : t) {
            a[s] = f16(v);
        }
        return am::model_f32(am::F_F16, a, b, c, P);
    };
    // +2^15 x 2 at slot 0, -(2^15 x 2) at slot 1, -1.5 * 2^-24 * 2^16 at slot 2: floored to -2 units of 2^-8
    {
        uint32_t a[16] = {}, b[16];
        for (int s = 0; s < 16; ++s) {
            b[s] = 0x3C00;
        }
        a[0] = f16(32768);
        b[0] = f16(2);
        a[1] = f16(-32768);
        b[1] = f16(2);
        a[2] = f16(-0x1.8p-8);
        expect("in-step floor: -1.5 units -> -2 units", am::model_f32(am::F_F16, a, b, 0, P), 0xBC000000u);
    }
    expect("1 + (-1) = +0", run1({{0, 1}, {1, -1}}, 0), 0u);
    expect("pair summed exactly: 1024 - 1024, then 2^-15 + 2^-15 in one pair", run1({{0, 1024}, {1, -1024}, {2, 0x1p-15}, {3, 0x1p-15}}, 0),
           0x38800000u);
    expect("running sum below the unit vanishes", run1({{4, 1024}, {5, -1024}, {0, -0x1p-24}}, 0), 0u);
    // the measured order: slot 8 is added before slot 4, so +L@0 -L@8 leaves s@4 exact
    expect("step order {0-3},{8-11},{4-7}", run1({{0, 1024}, {8, -1024}, {4, 0x1.8p-14}}, 0), 0x38C00000u);
    // controls on random data: each one-stage change must disagree with the measured model somewhere visible
    std::mt19937_64 rng(5);
    std::vector<ad::Problem> ps;
    for (int i = 0; i < 64; ++i) {
        ps.push_back(ad::make(am::F_F16, (ad::Dist) (i % 4), rng));
    }
    struct V {
        const char * name;
        std::function<void(am::Params &)> f;
    };
    const std::vector<V> vars = {{"width+1", [](am::Params & p) { p.width += 1; }},
                                 {"products floored one by one", [](am::Params & p) { p.prod_each = true; }},
                                 {"steps of 4 exact", [](am::Params & p) { p.sub = 4; }},
                                 {"K-contiguous order", [](am::Params & p) { for (int i = 0; i < 16; ++i) p.order[i] = i; }},
                                 {"running sum floored when small", [](am::Params & p) { p.acc_floor_small = true; }}};
    for (const V & v : vars) {
        am::Params q = am::measured_params(am::F_F16);
        v.f(q);
        long diff = 0, n = 0;
        for (auto & p : ps) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    diff += am::model_f32(am::F_F16, p.a[r], p.b[c], p.c[r][c], q) !=
                            am::model_f32(am::F_F16, p.a[r], p.b[c], p.c[r][c], am::measured_params(am::F_F16));
                    ++n;
                }
            }
        }
        std::printf("RESULT selftest variant %-32s differs from the measured model on %ld of %ld\n", v.name, diff, n);
        if (diff == 0) {
            ++bad;
        }
    }
    text = std::to_string(bad) + " failures (exact reference, model hand cases, variant controls)";
    return bad == 0;
}

static bool mode_random(long n_main, std::string & text) {
    bool ok = true;
    const int forms[] = {arith::F16, arith::BF16, arith::FF, arith::FB, arith::BF, arith::BB};
    for (int form : forms) {
        const long n = form <= arith::FF ? n_main : std::max(256L, n_main / 10);
        const long np = (n + 255) / 256;
        std::mt19937_64 rng(1000 + form);
        std::vector<ad::Problem> ps(np);
        const int nd = form >= arith::FF ? 5 : 4;
        for (long i = 0; i < np; ++i) {
            ps[i] = ad::make(form, (ad::Dist) (i % nd), rng);
        }
        const std::vector<uint32_t> d = run_problems(form, ps);
        const Tally t = compare(form, ps, d, params_for(form));
        std::printf("RESULT random %-8s elements %ld model_mismatches %ld card_equals_exact %.4f%%%s%s\n", fname(form), t.n,
                    t.bad, 100.0 * t.exact_eq / t.n, t.bad ? " first: " : "", t.first.c_str());
        // one-stage-changed models on the first 4096 problems: must disagree visibly
        if (!g_break && form <= arith::FF) {
            const std::vector<ad::Problem> sub(ps.begin(), ps.begin() + std::min<long>(np, 4096));
            for (int v = 0; v < 3; ++v) {
                am::Params q = am::measured_params(form);
                const char * name = v == 0 ? "width+1" : v == 1 ? "products floored one by one" : "no group-local floor";
                if (v == 0) {
                    q.width += 1;
                } else if (v == 1) {
                    q.prod_each = true;
                } else {
                    q.local_floor = false;
                }
                const Tally tv = compare(form, sub, d, q);
                std::printf("RESULT random %-8s control %-28s mismatches %ld of %ld (%.3f%%)\n", fname(form), name, tv.bad,
                            tv.n, 100.0 * tv.bad / tv.n);
                if (v < 2 && tv.bad == 0) {
                    ok = false;
                }
            }
        }
        ok = ok && t.bad == 0;
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s%s %ld/%ld", text.empty() ? "" : "; ", fname(form), t.n - t.bad, t.n);
        text += buf;
    }
    return ok;
}

static bool mode_adv(long n, std::string & text) {
    bool ok = true;
    for (int form : {arith::F16, arith::BF16, arith::FF, arith::FB, arith::BF, arith::BB}) {
        std::mt19937_64 rng(77 + form);
        const long np = std::max(1L, (form <= arith::FF ? n : n / 4) / 256);
        std::vector<ad::Problem> ps(np);
        for (auto & p : ps) {
            p = ad::adversarial(form, rng);
        }
        const std::vector<uint32_t> d = run_problems(form, ps);
        const Tally t = compare(form, ps, d, params_for(form));
        std::printf("RESULT adv %-8s elements %ld model_mismatches %ld card_equals_exact %.4f%%%s%s\n", fname(form), t.n,
                    t.bad, 100.0 * t.exact_eq / t.n, t.bad ? " first: " : "", t.first.c_str());
        ok = ok && t.bad == 0;
        char buf[120];
        std::snprintf(buf, sizeof buf, "%s%s %ld/%ld", text.empty() ? "" : "; ", fname(form), t.n - t.bad, t.n);
        text += buf;
    }
    // Fingerprint: +L at slot x, -L at slot z, small at y (FP16, C = 0); the loss pattern over (x, z, y) must change
    // when the slots are permuted (swap slots 4 and 8), else the probe could not see the order.
    auto fingerprint = [&](const int * perm) {
        std::vector<ad::Problem> ps;
        ad::Problem p{};
        int k = 0;
        std::vector<std::array<int, 3>> idx;
        for (int x = 0; x < 16; ++x) {
            for (int z = 0; z < 16; ++z) {
                for (int y = 0; y < 16; ++y) {
                    if (x == z || y == x || y == z) {
                        continue;
                    }
                    const int r = k % 16;
                    for (int s = 0; s < 16; ++s) {
                        p.a[r][s] = 0;
                    }
                    p.a[r][perm[x]] = 0x6400;  // 1024
                    p.a[r][perm[z]] = 0xE400;
                    p.a[r][perm[y]] = 0x0600;  // 1.5 * 2^-14, gap 24
                    for (int c = 0; c < 16; ++c) {
                        for (int s = 0; s < 16; ++s) {
                            p.b[c][s] = 0x3C00;
                        }
                        p.c[r][c] = 0;
                    }
                    idx.push_back({x, z, y});
                    if (++k % 16 == 0) {
                        ps.push_back(p);
                    }
                }
            }
        }
        const std::vector<uint32_t> d = run_problems(arith::F16, ps);
        std::string fp;
        for (size_t i = 0; i < ps.size() * 16; ++i) {
            fp += d[(i / 16) * 256 + (i % 16) * 17] == 0x38C00000u ? 'E' : 'x';  // diagonal (r, r): exact 1.5*2^-14
        }
        return fp;
    };
    int id[16], sw[16];
    for (int i = 0; i < 16; ++i) {
        id[i] = sw[i] = i;
    }
    std::swap(sw[4], sw[8]);
    const std::string f0 = fingerprint(id), f1 = fingerprint(sw);
    long changed = 0, exact = 0;
    for (size_t i = 0; i < f0.size(); ++i) {
        changed += f0[i] != f1[i];
        exact += f0[i] == 'E';
    }
    std::printf("RESULT adv fingerprint f16 triples %zu exact_small_term %ld changed_by_swapping_slots_4_8 %ld\n", f0.size(),
                exact, changed);
    if (changed == 0) {
        ok = false;
    }
    text += "; fingerprint changes under a slot swap: " + std::to_string(changed);
    return ok;
}

// Special values in random slots on top of random finite data.
static bool mode_special(std::string & text) {
    bool ok = true;
    for (int form : {arith::F16, arith::BF16, arith::FF, arith::FB, arith::BF, arith::BB}) {
        const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
        auto specials = [](am::Fmt f) {
            const am::FmtInfo fi = am::info(f);
            const uint32_t sb = 1u << (fi.ebits + fi.mbits), emax = (1u << fi.ebits) - 1;
            std::vector<uint32_t> v = {0, sb, 1, sb | 1, (1u << fi.mbits) - 1};
            if (fi.e4m3) {
                v.insert(v.end(), {0x7F, 0xFF, 0x7E, 0xFE});
            } else {
                v.insert(v.end(), {emax << fi.mbits, sb | (emax << fi.mbits), (emax << fi.mbits) | 1,
                                   sb | (emax << fi.mbits) | (1u << (fi.mbits - 1)), ((emax - 1) << fi.mbits) | ((1u << fi.mbits) - 1),
                                   (emax << fi.mbits) | (1u << (fi.mbits - 1)) | 5});
            }
            return v;
        };
        const std::vector<uint32_t> sa = specials(fa), sbv = specials(fb);
        const std::vector<uint32_t> sc = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC12345u, 0x7F800001u, 0xFFC00001u,
                                          0x7F7FFFFFu, 0xFF7FFFFFu, 1u, 0x80000001u, 0x3F800000u, 0x00800000u};
        std::mt19937_64 rng(31 + form);
        std::vector<ad::Problem> ps(160);
        for (size_t i = 0; i < ps.size(); ++i) {
            ps[i] = ad::make(form, (ad::Dist) (i % 4), rng);
            for (int r = 0; r < 16; ++r) {
                const int k = (int) (rng() % 4);
                for (int j = 0; j < k; ++j) {
                    ps[i].a[r][rng() % 16] = sa[rng() % sa.size()];
                }
            }
            for (int c = 0; c < 16; ++c) {
                const int k = (int) (rng() % 3);
                for (int j = 0; j < k; ++j) {
                    ps[i].b[c][rng() % 16] = sbv[rng() % sbv.size()];
                }
                for (int r = 0; r < 16; ++r) {
                    if (rng() % 3 == 0) {
                        ps[i].c[r][c] = sc[rng() % sc.size()];
                    }
                }
            }
            // every 8th problem: all-zero products with signed zeros, so the zero-sign rule is hit often
            if (i % 8 == 7) {
                for (int r = 0; r < 16; ++r) {
                    for (int s = 0; s < 16; ++s) {
                        ps[i].a[r][s] = (rng() & 1) ? (1u << (am::info(fa).ebits + am::info(fa).mbits)) : 0;
                    }
                }
            }
        }
        const std::vector<uint32_t> d = run_problems(form, ps);
        const Tally t = compare(form, ps, d, params_for(form));
        long nan_in = 0, nan_inv = 0, neg0 = 0;
        for (uint32_t v : d) {
            nan_in += v == am::NAN_INPUT;
            nan_inv += v == am::NAN_INVALID;
            neg0 += v == 0x80000000u;
        }
        std::printf("RESULT special %-8s elements %ld model_mismatches %ld nan_input_results %ld nan_invalid_results %ld "
                    "negative_zero_results %ld%s%s\n",
                    fname(form), t.n, t.bad, nan_in, nan_inv, neg0, t.bad ? " first: " : "", t.first.c_str());
        ok = ok && t.bad == 0;
        char buf[120];
        std::snprintf(buf, sizeof buf, "%s%s %ld/%ld", text.empty() ? "" : "; ", fname(form), t.n - t.bad, t.n);
        text += buf;
    }
    return ok;
}

static bool mode_determ(std::string & text) {
    bool ok = true;
    for (int form : {arith::F16, arith::BF16, arith::FF}) {
        std::mt19937_64 rng(3 + form);
        std::vector<ad::Problem> ps(2048);
        for (size_t i = 0; i < ps.size(); ++i) {
            ps[i] = ad::make(form, (ad::Dist) (i % 4), rng);
        }
        const std::vector<uint32_t> d0 = run_problems(form, ps);
        long diff = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const std::vector<uint32_t> d1 = run_problems(form, ps);
            for (size_t i = 0; i < d0.size(); ++i) {
                diff += d0[i] != d1[i];
            }
        }
        // other waves: reversed problem order; other rows and columns: A rows and B columns reversed
        std::vector<ad::Problem> rv(ps.rbegin(), ps.rend());
        const std::vector<uint32_t> d2 = run_problems(form, rv);
        long moved = 0;
        for (size_t i = 0; i < ps.size(); ++i) {
            for (int k = 0; k < 256; ++k) {
                moved += d0[i * 256 + k] != d2[(ps.size() - 1 - i) * 256 + k];
            }
        }
        std::vector<ad::Problem> pr = ps;
        for (auto & p : pr) {
            ad::Problem q = p;
            for (int r = 0; r < 16; ++r) {
                for (int s = 0; s < 16; ++s) {
                    q.a[15 - r][s] = p.a[r][s];
                    q.b[15 - r][s] = p.b[r][s];
                }
                for (int c = 0; c < 16; ++c) {
                    q.c[15 - r][15 - c] = p.c[r][c];
                }
            }
            p = q;
        }
        // after a launch of another form with other data
        std::mt19937_64 rng2(9);
        std::vector<ad::Problem> other(64);
        for (auto & p : other) {
            p = ad::make(arith::BF16, ad::WIDE, rng2);
        }
        run_problems(form == arith::BF16 ? arith::F16 : arith::BF16, other);
        const std::vector<uint32_t> d3 = run_problems(form, pr);
        long rows = 0;
        for (size_t i = 0; i < ps.size(); ++i) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    rows += d0[i * 256 + r * 16 + c] != d3[i * 256 + (15 - r) * 16 + (15 - c)];
                }
            }
        }
        if (g_break) {
            diff += 1;  // the expectation broken on purpose
        }
        std::printf("RESULT determ %-8s elements %zu repeat_differences %ld other_wave_differences %ld "
                    "other_row_column_differences %ld\n",
                    fname(form), d0.size(), diff, moved, rows);
        ok = ok && diff == 0 && moved == 0 && rows == 0;
        char buf[120];
        std::snprintf(buf, sizeof buf, "%s%s %ld/%ld/%ld", text.empty() ? "" : "; ", fname(form), diff, moved, rows);
        text += buf;
    }
    text = "differences (repeat/wave/row) " + text;
    return ok;
}

static bool mode_chain(std::string & text) {
    bool ok = true;
    for (int form : {arith::F16, arith::BF16, arith::FF}) {
        for (int steps : {4, 64, 256}) {
            const int waves = steps == 256 ? 32 : 128;
            std::mt19937_64 rng(steps * 7 + form);
            arith::Batch bt;
            bt.init(form, waves, steps);
            std::vector<ad::Problem> ps((size_t) waves * steps);
            for (int w = 0; w < waves; ++w) {
                for (int s = 0; s < steps; ++s) {
                    ad::Problem & p = ps[(size_t) w * steps + s];
                    p = ad::make(form, (ad::Dist) ((w + s) % 2), rng);  // narrow or wide
                    for (int r = 0; r < 16; ++r) {
                        for (int k = 0; k < 16; ++k) {
                            bt.set_a(w, s, r, k, p.a[r][k]);
                            bt.set_b(w, s, r, k, p.b[r][k]);
                        }
                    }
                }
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        bt.set_c(w, r, c, ps[(size_t) w * steps].c[r][c]);
                    }
                }
            }
            arith::run(bt);
            const am::Params P = params_for(form);
            long bad = 0, n = 0;
            for (int w = 0; w < waves; ++w) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        uint32_t acc = ps[(size_t) w * steps].c[r][c];
                        for (int s = 0; s < steps; ++s) {
                            const ad::Problem & p = ps[(size_t) w * steps + s];
                            acc = am::model_f32(form, p.a[r], p.b[c], acc, P);
                        }
                        bad += acc != bt.get_d(w, r, c);
                        ++n;
                    }
                }
            }
            std::printf("RESULT chain %-8s K %5d elements %ld model_mismatches %ld\n", fname(form), steps * 16, n, bad);
            ok = ok && bad == 0;
            char buf[80];
            std::snprintf(buf, sizeof buf, "%s%s K%d %ld/%ld", text.empty() ? "" : "; ", fname(form), steps * 16, n - bad, n);
            text += buf;
        }
    }
    return ok;
}

// MODE: VALU control inputs: add_f32(1, 1.5 * 2^-24), mul_f32(-(1+2^-23), 1+2^-23), add_f16(1, 2^-12), mul_f32 of a
// denormal float by 1, add_f16 of an FP16 denormal and 0.
static bool mode_mode(std::string & text) {
    const uint32_t x[8] = {0x3F800000u, 0x33C00000u, 0xBF800001u, 0x3F800001u, 0x3C00u, 0x0C00u, 0x00020001u, 0x3F800000u};
    const uint32_t m0 = arith::run_valu(x, false, 0)[5];
    std::printf("RESULT mode default MODE register 0x%08x\n", m0);
    struct S {
        const char * name;
        uint32_t mode;
    };
    const uint32_t keep = m0 & arith::MODE_MASK;
    const std::vector<S> settings = {
        {"default (written back)", keep},
        {"round +inf (single and FP16)", (keep & ~0xFu) | 0x5u},
        {"round -inf (single and FP16)", (keep & ~0xFu) | 0xAu},
        {"round toward zero (single and FP16)", (keep & ~0xFu) | 0xFu},
        {"denormals flushed (single and FP16)", keep & ~0xF0u},
        {"single denormals flushed only", keep & ~0x30u},
        {"FP16_OVFL set", keep | 0x00800000u},
    };
    const std::vector<uint32_t> base = arith::run_valu(x, true, keep);
    bool control_ok = true;
    std::mt19937_64 rng(4242);
    std::vector<ad::Problem> ps(256);
    for (size_t i = 0; i < ps.size(); ++i) {
        ps[i] = ad::make(arith::F16, (ad::Dist) (i % 4), rng);
    }
    std::vector<ad::Problem> pb(ps.size()), pf(ps.size());
    std::mt19937_64 rngb(4243), rngf(4244);
    for (size_t i = 0; i < ps.size(); ++i) {
        pb[i] = ad::make(arith::BF16, (ad::Dist) (i % 4), rngb);
        pf[i] = ad::make(arith::FF, (ad::Dist) (i % 5), rngf);
    }
    // denormal-heavy problems: BF16 products near 2^-140 and FP16 subnormal inputs
    for (size_t i = 0; i < 32; ++i) {
        for (int r = 0; r < 16; ++r) {
            for (int s = 0; s < 16; ++s) {
                pb[i].a[r][s] = ad::rand_val(am::BF16, rngb, -75, -65);
                pb[i].b[r][s] = ad::rand_val(am::BF16, rngb, -75, -65);
                ps[i].a[r][s] = (uint32_t) (rng() & 0x83FF);  // FP16 subnormals and zeros
            }
            for (int c = 0; c < 16; ++c) {
                pb[i].c[r][c] = (uint32_t) (rng() & 0x807FFFFF);  // FP32 denormals
            }
        }
    }
    const std::vector<uint32_t> d16 = run_problems(arith::F16, ps), dbf = run_problems(arith::BF16, pb),
                                dff = run_problems(arith::FF, pf);
    for (const S & s : settings) {
        const std::vector<uint32_t> v = arith::run_valu(x, true, s.mode);
        long ch = 0;
        for (int i = 0; i < 5; ++i) {
            ch += v[i] != base[i];
        }
        const bool readback = (v[5] & arith::MODE_MASK) == (s.mode & arith::MODE_MASK);
        const std::vector<uint32_t> e16 = run_problems(arith::F16, ps, 0, true, s.mode),
                                    ebf = run_problems(arith::BF16, pb, 0, true, s.mode),
                                    eff = run_problems(arith::FF, pf, 0, true, s.mode);
        long c16 = 0, cbf = 0, cff = 0;
        for (size_t i = 0; i < d16.size(); ++i) {
            c16 += e16[i] != d16[i];
            cbf += ebf[i] != dbf[i];
            cff += eff[i] != dff[i];
        }
        std::printf("RESULT mode %-38s mode 0x%08x readback %s valu %08x %08x %04x %08x %04x (changed %ld of 5) "
                    "matrix outputs changed: f16 %ld bf16 %ld fp8 %ld of %zu each\n",
                    s.name, s.mode, readback ? "ok" : "WRONG", v[0], v[1], v[2], v[3], v[4], ch, c16, cbf, cff, d16.size());
        const bool is_default = s.mode == keep;
        const bool needs_change = !is_default && s.mode != (keep | 0x00800000u);
        if (!readback || (needs_change && ch == 0) || (is_default && ch != 0)) {
            control_ok = false;
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s%s: valu %ld changed, matrix %ld/%ld/%ld", text.empty() ? "" : "; ", s.name, ch,
                      c16, cbf, cff);
        text += buf;
    }
    if (g_break) {
        control_ok = false;  // the expectation broken on purpose
    }
    return control_ok;
}

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    long n = 10000000;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--break") {
            g_break = true;
        } else if (a == "--n" && i + 1 < argc) {
            n = std::atol(argv[++i]);
        }
    }
    std::string text;
    bool ok = false;
    if (mode == "selftest") {
        ok = mode_selftest(text);
    } else {
        require_arch();
        if (mode == "random") {
            ok = mode_random(n, text);
        } else if (mode == "adv") {
            ok = mode_adv(std::min(n, 2000000L), text);
        } else if (mode == "special") {
            ok = mode_special(text);
        } else if (mode == "determ") {
            ok = mode_determ(text);
        } else if (mode == "chain") {
            ok = mode_chain(text);
        } else if (mode == "mode") {
            ok = mode_mode(text);
        } else {
            std::fprintf(stderr, "usage: arith_check selftest|random|adv|special|determ|chain|mode [--n N] [--break]\n");
            return 2;
        }
    }
    if (g_break) {
        text += " (broken on purpose)";
    }
    verdict(("arith_" + mode).c_str(), ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
