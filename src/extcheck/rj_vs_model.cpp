// Check 2: the study's arithmetic model (src/study/arith_model.h) against AMD's rocjitsu rj_arith, bit for bit, on
// the same register images. CPU only. AMD's library is linked unmodified (src/extcheck/rj_build builds it from the
// download); gen_study.h (scripts/extcheck/c2_extract.py) carries the study's generators and register layouts,
// cut from the study's own files.
//
// Usage: rj_vs_model dense|sparse|special|edge|gpu DIR [--break]
//   dense   every dense gfx12 Wave32 form both cover: 6 FP32-accumulator float forms, FP16 and BF16 into 16-bit,
//           IU8, IU4 (K=16), IU4 (K=32) with every sign and clamp setting; >= 1,000,000 outputs per form.
//   sparse  SWMMAC: the 8 float forms and IU8, index_key 0 and 1, with valid 2:4 indices; plus random indices
//           (invalid ones are "unspecified" in AMD's header) as a separate line.
//   special NaN, infinity, maximum and zero codes dropped into random problems (not a study generator), float forms
//   edge    the six edge cases quoted in matrix-arithmetic.md (card values recorded in the repo's logs) and the
//           45,056-product BF16 subnormal sweep.
//   gpu DIR the card outputs recorded in the study's build/arith/explore (*.in + *.out), against both models.
//   --break the study's model is changed on purpose (each product truncated toward zero instead of the run sums
//           floored: Params.prod_each and prod_trunc; integer clamp only at the end; the other index_key for sparse): the harness must then report disagreements.
#include "gen_study.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <dirent.h>
#include <mutex>
#include <thread>

using Fn6 = int (*)(uint64_t, uint64_t, uint32_t * const *, const uint32_t * const *, const uint32_t * const *,
                    const uint32_t * const *);

static bool g_brk = false;
static uint64_t g_cpu = 0;
static const uint64_t SEM_EXACT = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
static const uint64_t SEM_LOOSE = RJ_ARITH_SEMANTICS_LOOSE;

struct Prob {
    uint32_t a[16][32], b[16][32], c[16][16];
};
struct Regs {
    uint32_t r[8][32];
};

static void put(Regs & g, const arith::Slot & s, uint32_t bits, int eb) {
    const uint32_t mask = (eb == 32 ? 0xFFFFFFFFu : ((1u << eb) - 1)) << s.bit;
    uint32_t & w = g.r[s.word][s.lane];
    w = (w & ~mask) | ((bits << s.bit) & mask);
}
static uint32_t get(const Regs & g, const arith::Slot & s, int eb) {
    const uint32_t v = g.r[s.word][s.lane] >> s.bit;
    return eb == 32 ? v : v & ((1u << eb) - 1);
}
static void ptrs(Regs & g, int n, uint32_t ** p) {
    for (int i = 0; i < n; ++i) p[i] = g.r[i];
}

// ------------------------------------------------------------------------------------------------ dense forms
struct Dense {
    const char * name;
    int f;       // arith::Form
    int mform;   // am::Form of the inputs (-1 for integer forms)
    Fn6 fn;
    int k;       // slots per row
};
#define D(nm, f, m, fn, k) {nm, arith::f, m, rj_arith_##fn, k}
static const Dense dense_forms[] = {
    D("f16", F16, am::F_F16, v_wmma_f32_16x16x16_f16_w32_gfx12, 16),
    D("bf16", BF16, am::F_BF16, v_wmma_f32_16x16x16_bf16_w32_gfx12, 16),
    D("fp8_fp8", FF, am::F_FF, v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12, 16),
    D("fp8_bf8", FB, am::F_FB, v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12, 16),
    D("bf8_fp8", BF, am::F_BF, v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12, 16),
    D("bf8_bf8", BB, am::F_BB, v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12, 16),
    D("f16_f16 (FP16 into FP16)", F16H, am::F_F16, v_wmma_f16_16x16x16_f16_w32_gfx12, 16),
    D("bf16_bf16 (BF16 into BF16)", BF16H, am::F_BF16, v_wmma_bf16_16x16x16_bf16_w32_gfx12, 16),
    D("iu8", IU8, -1, v_wmma_i32_16x16x16_iu8_w32_gfx12, 16),
    D("iu4 (K=16)", IU4, -1, v_wmma_i32_16x16x16_iu4_w32_gfx12, 16),
    D("iu4 (K=32)", IU4K32, -1, v_wmma_i32_16x16x32_iu4_w32_gfx12, 32),
};

struct Mod {
    const char * name;
    bool sa, sb, clamp;
};
static const Mod int_mods[] = {{"none", 0, 0, 0}, {"negA", 1, 0, 0}, {"negB", 0, 1, 0}, {"sAB", 1, 1, 0},
                               {"clamp", 0, 0, 1}, {"sAclamp", 1, 0, 1}, {"sBclamp", 0, 1, 1}, {"sABclamp", 1, 1, 1}};

static int a_regs(int f) { return arith::words_ab(f); }
static int cd_regs(int f) { return arith::acc16(f) ? 4 : 8; }
static int cd_bits(int f) { return arith::acc16(f) ? 16 : 32; }

static Prob to_prob(const ad::Problem & p) {
    Prob q{};
    for (int r = 0; r < 16; ++r) {
        std::memcpy(q.a[r], p.a[r], sizeof p.a[r]);
        std::memcpy(q.b[r], p.b[r], sizeof p.b[r]);
        std::memcpy(q.c[r], p.c[r], sizeof p.c[r]);
    }
    return q;
}
static Prob to_prob(const IProblem & p) {
    Prob q{};
    std::memcpy(q.a, p.a, sizeof q.a);
    std::memcpy(q.b, p.b, sizeof q.b);
    std::memcpy(q.c, p.c, sizeof q.c);
    return q;
}

// One call of AMD's instruction for a whole wave. Returns the library's status; out[r][c] gets D.
static int amd_dense(const Dense & F, const Prob & p, uint64_t sem, uint64_t iflags, uint32_t out[16][16]) {
    Regs A{}, B{}, C{}, Dd;
    std::memset(&Dd, 0xEE, sizeof Dd);
    const int eb = arith::elem_bits(F.f);
    for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < F.k; ++s) {
            put(A, arith::slot_ab(F.f, r, s), p.a[r][s], eb);
            put(B, arith::slot_ab(F.f, r, s), p.b[r][s], eb);
        }
        for (int c = 0; c < 16; ++c) put(C, arith::slot_cd(F.f, r, c), p.c[r][c], cd_bits(F.f));
    }
    uint32_t *dp[8], *ap[8], *bp[8], *cp[8];
    ptrs(Dd, 8, dp);
    ptrs(A, 8, ap);
    ptrs(B, 8, bp);
    ptrs(C, 8, cp);
    const int rc = F.fn(g_cpu | sem, iflags, dp, ap, bp, cp);
    if (rc == 0) {
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) out[r][c] = get(Dd, arith::slot_cd(F.f, r, c), cd_bits(F.f));
    }
    return rc;
}

static uint32_t study_dense(const Dense & F, const Prob & p, int r, int c, const Mod * md) {
    if (F.mform < 0) {
        const int w = arith::elem_bits(F.f);
        const int cm = md->clamp ? (g_brk ? 1 : 2) : 0;
        return am::model_int(p.a[r], p.b[c], F.k, w, md->sa, md->sb, p.c[r][c], cm, F.k / 2);
    }
    if (F.f == arith::F16H || F.f == arith::BF16H) {  // as measured16() in arith_more.hip
        am::Params P = am::acc16_params(F.mform, 3);
        P.rtz16 = F.f == arith::BF16H ? 2 : 0;
        P.prod_each = g_brk;
        P.prod_trunc = g_brk;
        return am::model_acc16(F.mform, p.a[r], p.b[c], p.c[r][c], P);
    }
    am::Params P = am::measured_params(F.mform);
    P.prod_each = g_brk;
    P.prod_trunc = g_brk;
    return am::model_f32(F.mform, p.a[r], p.b[c], p.c[r][c], P);
}

// ------------------------------------------------------------------------------------------------ bookkeeping
struct Miss {
    std::string cat;
    long prob;
    int r, c;
    uint32_t study, amd;
    std::string inputs;
    bool operator<(const Miss & o) const { return std::tie(cat, prob, r, c) < std::tie(o.cat, o.prob, o.r, o.c); }
};
struct Tally {
    long n = 0, bad = 0, loose_bad = 0, loose_n = 0, rc_fail = 0;
    std::vector<Miss> first;
};
static const size_t KEEP = 20;
static void keep(Tally & t, Miss && m) {
    t.first.push_back(std::move(m));
    std::sort(t.first.begin(), t.first.end());
    if (t.first.size() > KEEP) t.first.pop_back();
}
static void merge(Tally & into, const Tally & t) {
    into.n += t.n;
    into.bad += t.bad;
    into.loose_bad += t.loose_bad;
    into.loose_n += t.loose_n;
    into.rc_fail += t.rc_fail;
    for (const Miss & m : t.first) {
        into.first.push_back(m);
    }
    std::sort(into.first.begin(), into.first.end());
    if (into.first.size() > KEEP) into.first.resize(KEEP);
}
static std::string hexs(const uint32_t * v, int n, int bits) {
    std::string s;
    char b[16];
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, bits == 4 ? "%x " : bits == 8 ? "%02x " : bits == 16 ? "%04x " : "%08x ", v[i]);
        s += b;
    }
    return s;
}
static void par_for(size_t n, const std::function<void(size_t, int)> & f) {
    const int T = 8;
    std::vector<std::thread> th;
    std::atomic<size_t> next{0};
    for (int t = 0; t < T; ++t)
        th.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < n;) f(i, t);
        });
    for (auto & x : th) x.join();
}

// Compare AMD (bit-exact mode) with the study's model on a list of problems for one form and one modifier set.
static void run_probs(const Dense & F, const Mod * md, const char * cat, const std::vector<Prob> & ps, Tally & total,
                      bool want_loose) {
    std::vector<Tally> part(8);
    uint64_t iflags = 0;
    if (md) iflags = (md->sa ? RJ_ARITH_WMMA_SIGNED_A : 0) | (md->sb ? RJ_ARITH_WMMA_SIGNED_B : 0) | (md->clamp ? RJ_ARITH_WMMA_CLAMP : 0);
    par_for(ps.size(), [&](size_t i, int t) {
        Tally & ta = part[t];
        uint32_t out[16][16], lo[16][16];
        if (amd_dense(F, ps[i], SEM_EXACT, iflags, out) != 0) {
            ++ta.rc_fail;
            return;
        }
        const bool lok = want_loose && amd_dense(F, ps[i], SEM_LOOSE, iflags, lo) == 0;
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                const uint32_t m = study_dense(F, ps[i], r, c, md);
                ++ta.n;
                if (lok) {
                    ++ta.loose_n;
                    ta.loose_bad += lo[r][c] != m;
                }
                if (m != out[r][c]) {
                    ++ta.bad;
                    if (ta.first.size() < KEEP || true) {
                        Miss ms{std::string(cat) + (md ? std::string(" ") + md->name : ""), (long) i, r, c, m, out[r][c], ""};
                        ms.inputs = "A " + hexs(ps[i].a[r], F.k, arith::elem_bits(F.f)) + "| B " +
                                    hexs(ps[i].b[c], F.k, arith::elem_bits(F.f)) + "| C " + hexs(&ps[i].c[r][c], 1, cd_bits(F.f));
                        keep(ta, std::move(ms));
                    }
                }
            }
    });
    for (auto & t : part) merge(total, t);
}

static void report(const char * label, const std::string & what, const Tally & t) {
    std::printf("RESULT %-28s %-34s outputs %9ld  disagreements %9ld%s\n", label, what.c_str(), t.n, t.bad,
                t.rc_fail ? "  (AMD returned an error for some calls)" : "");
}
static void print_first(const char * label, const Tally & t) {
    int k = 0;
    for (const Miss & m : t.first) {
        std::printf("FIRST %-28s #%02d [%s] problem %ld (row %d, col %d) study 0x%08x amd 0x%08x inputs: %s\n", label, ++k,
                    m.cat.c_str(), m.prob, m.r, m.c, m.study, m.amd, m.inputs.c_str());
    }
}

// ------------------------------------------------------------------------------------------------ dense mode
static int mode_dense() {
    long total_bad = 0;
    for (const Dense & F : dense_forms) {
        Tally all;
        Tally loose_all;
        std::mt19937_64 rng(5000 + F.f);
        auto cats = [&](const char * cat, const std::vector<Prob> & ps, const Mod * md = nullptr) {
            Tally t;
            run_probs(F, md, cat, ps, t, true);
            report(F.name, std::string(cat) + (md ? std::string(" ") + md->name : ""), t);
            merge(all, t);
            all.loose_bad += 0;
        };
        if (F.mform < 0) {
            const int nprob = 125;  // 8 modifier sets x 4 kinds x 125 x 256 = 1,024,000 outputs
            for (const Mod & md : int_mods)
                for (int kind = 0; kind < 4; ++kind) {
                    std::vector<Prob> ps;
                    for (int i = 0; i < nprob; ++i) ps.push_back(to_prob(make_int(F.f, kind, rng)));
                    char cat[32];
                    std::snprintf(cat, sizeof cat, "kind%d", kind);
                    cats(cat, ps, &md);
                }
        } else {
            const bool h = F.f == arith::F16H || F.f == arith::BF16H;
            std::vector<const char *> names;
            const int ndist = (!h && F.mform >= am::F_FF) ? 5 : 4;
            const int ncat = ndist + 1;
            const int nprob = (1000000 + 256 * ncat - 1) / (256 * ncat);
            for (int d = 0; d < ndist; ++d) {
                std::vector<Prob> ps;
                for (int i = 0; i < nprob; ++i)
                    ps.push_back(to_prob(h ? make16(F.f, d, rng) : ad::make(F.mform, (ad::Dist) d, rng)));
                cats(h ? (d == 0 ? "narrow" : d == 1 ? "wide" : d == 2 ? "cancel" : "bigsmall") : ad::dist_name[d], ps);
            }
            std::vector<Prob> pa;
            for (int i = 0; i < nprob; ++i) {
                ad::Problem p = h ? (F.f == arith::F16H ? ad::adversarial(F.mform, rng, 7, 7) : ad::adversarial(F.mform, rng))
                                  : ad::adversarial(F.mform, rng);
                if (h)
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < 16; ++c) p.c[r][c] = conv16(F.f, p.c[r][c]);
                pa.push_back(to_prob(p));
            }
            cats("adversarial", pa);
        }
        report(F.name, "TOTAL (AMD bit-exact vs study)", all);
        print_first(F.name, all);
        total_bad += all.bad;
    }
    std::printf("DENSE_TOTAL_DISAGREEMENTS %ld\n", total_bad);
    return 0;
}

// ------------------------------------------------------------------------------------------------ special mode
// Not a study generator: NaN, infinity, maximum-finite and zero codes dropped at random into ad::make problems.
static const std::vector<uint32_t> & specials(am::Fmt f) {
    static const std::vector<uint32_t> h = {0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFD55, 0x7BFF, 0xFBFF, 0x0000, 0x8000, 0x0001};
    static const std::vector<uint32_t> bf = {0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFAA, 0x7F7F, 0xFF7F, 0x0000, 0x8000, 0x0001};
    static const std::vector<uint32_t> e4 = {0x7F, 0xFF, 0x7E, 0xFE, 0x00, 0x80, 0x01};
    static const std::vector<uint32_t> e5 = {0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0x7B, 0xFB, 0x00, 0x80, 0x01};
    return f == am::F16 ? h : f == am::BF16 ? bf : f == am::E4M3 ? e4 : e5;
}

static int mode_special() {
    long total_bad = 0;
    for (const Dense & F : dense_forms) {
        if (F.mform < 0) continue;
        const bool h = F.f == arith::F16H || F.f == arith::BF16H;
        std::mt19937_64 rng(9000 + F.f);
        const long nprob = 4000;
        std::vector<Prob> ps;
        for (long i = 0; i < nprob; ++i) {
            Prob q = to_prob(h ? make16(F.f, (int) (i % 4), rng) : ad::make(F.mform, (ad::Dist) (i % 4), rng));
            const am::Fmt fa = am::fmt_a(F.mform), fb = am::fmt_b(F.mform);
            for (int r = 0; r < 16; ++r)
                for (int s = 0; s < 16; ++s) {
                    if ((rng() & 15) == 0) q.a[r][s] = specials(fa)[rng() % specials(fa).size()];
                    if ((rng() & 15) == 0) q.b[r][s] = specials(fb)[rng() % specials(fb).size()];
                }
            static const uint32_t c32[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x80000000u, 0x7F7FFFFFu};
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c)
                    if ((rng() & 15) == 0)
                        q.c[r][c] = h ? specials(F.f == arith::F16H ? am::F16 : am::BF16)[rng() % 11] : c32[rng() % 7];
            ps.push_back(q);
        }
        Tally t;
        run_probs(F, nullptr, "special", ps, t, false);
        long nonfinite = 0, cnt = 0;  // how many AMD outputs are NaN or Inf: the pass must actually reach them
        for (size_t i = 0; i < ps.size(); i += 8) {
            uint32_t out[16][16];
            amd_dense(F, ps[i], SEM_EXACT, 0, out);
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c, ++cnt) {
                    const uint32_t v = out[r][c];
                    nonfinite += h ? ((v & 0x7C00u) == 0x7C00u && F.f == arith::F16H) || ((v & 0x7F80u) == 0x7F80u && F.f == arith::BF16H)
                                   : (v & 0x7F800000u) == 0x7F800000u;
                }
        }
        std::printf("NOTE %s: %ld of %ld sampled AMD outputs are NaN or Inf\n", F.name, nonfinite, cnt);
        report(F.name, "special values (NaN, Inf, max, zero)", t);
        print_first(F.name, t);
        total_bad += t.bad;
    }
    std::printf("SPECIAL_TOTAL_DISAGREEMENTS %ld\n", total_bad);
    return 0;
}

// ------------------------------------------------------------------------------------------------ sparse mode
struct Sparse {
    const char * name;
    int sf;
    Fn6 fn;
};
#define S(nm, sf, fn) {nm, arith::sf, rj_arith_##fn}
static const Sparse sparse_forms[] = {
    S("swmmac_f16", S_F16, v_swmmac_f32_16x16x32_f16_w32),   S("swmmac_bf16", S_BF16, v_swmmac_f32_16x16x32_bf16_w32),
    S("swmmac_fp8_fp8", S_FF, v_swmmac_f32_16x16x32_fp8_fp8_w32), S("swmmac_fp8_bf8", S_FB, v_swmmac_f32_16x16x32_fp8_bf8_w32),
    S("swmmac_bf8_fp8", S_BF, v_swmmac_f32_16x16x32_bf8_fp8_w32), S("swmmac_bf8_bf8", S_BB, v_swmmac_f32_16x16x32_bf8_bf8_w32),
    S("swmmac_f16_f16", S_F16H, v_swmmac_f16_16x16x32_f16_w32),  S("swmmac_bf16_bf16", S_BF16H, v_swmmac_bf16_16x16x32_bf16_w32),
    S("swmmac_iu8", S_IU8, v_swmmac_i32_16x16x32_iu8_w32),
};

static int amd_sparse(const Sparse & F, const SProblem & q, int key, uint32_t out[16][16]) {
    const int df = arith::sdense(F.sf), eb = arith::elem_bits(df);
    Regs A{}, B{}, I{}, Dd{};
    for (int r = 0; r < 16; ++r) {
        for (int j = 0; j < 16; ++j) {
            put(A, arith::slot_ab(df, r, j), q.p.a[r][j], eb);
            put(B, arith::sslot_b(F.sf, r, j), q.p.b[r][j], eb);
            put(B, arith::sslot_b(F.sf, r, 16 + j), q.b2[r][j], eb);
        }
        for (int c = 0; c < 16; ++c) put(Dd, arith::slot_cd(df, r, c), q.p.c[r][c], cd_bits(df));
    }
    for (int l = 0; l < 32; ++l) I.r[0][l] = q.idx[l];
    uint32_t *dp[8], *ap[8], *bp[8], *ip[8];
    ptrs(Dd, 8, dp);
    ptrs(A, 8, ap);
    ptrs(B, 8, bp);
    ptrs(I, 8, ip);
    uint64_t iflags = (key ? RJ_ARITH_SWMMAC_INDEX_KEY_1 : 0);
    if (F.sf == arith::S_IU8) iflags |= RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B;
    const int rc = F.fn(g_cpu | SEM_EXACT, iflags, dp, ap, bp, ip);
    if (rc == 0)
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) out[r][c] = get(Dd, arith::slot_cd(df, r, c), cd_bits(df));
    return rc;
}

// kind 0: valid (first < second position of each group of four); 1: equal positions; 2: decreasing positions.
static void make_index(SProblem & q, std::mt19937_64 & rng, int kind) {
    const uint32_t inc[] = {0x4, 0x8, 0xc, 0x9, 0xd, 0xe};  // nibble = first | second << 2
    const uint32_t eq[] = {0x0, 0x5, 0xa, 0xf};
    const uint32_t dec[] = {0x1, 0x2, 0x3, 0x6, 0x7, 0xb};
    for (int l = 0; l < 32; ++l) {
        uint32_t w = 0;
        for (int m = 0; m < 8; ++m) {
            const uint32_t nib = kind == 0 ? inc[rng() % 6] : kind == 1 ? eq[rng() % 4] : dec[rng() % 6];
            w |= nib << (4 * m);
        }
        q.idx[l] = w;
    }
}

static int mode_sparse() {
    long total_bad = 0;
    for (const Sparse & F : sparse_forms) {
        Tally all;
        for (int variant = 0; variant < 5; ++variant) {  // 0: valid, key 0; 1: valid, key 1; 2: random indices; 3: equal pairs; 4: decreasing pairs
            const int key = variant == 1;
            const long np = variant >= 2 ? 1000 : 2000;  // valid: 2 x 2000 x 256 = 1,024,000 outputs
            std::mt19937_64 rng(7000 + F.sf * 8 + variant);
            std::vector<SProblem> ps(np);
            for (long i = 0; i < np; ++i) {
                ps[i] = make_sparse(F.sf, (int) i, rng);
                if (variant != 2) make_index(ps[i], rng, variant < 2 ? 0 : variant - 2);
            }
            std::vector<Tally> part(8);
            const bool saved = g_break;
            g_break = g_brk;
            par_for(ps.size(), [&](size_t i, int t) {
                Tally & ta = part[t];
                uint32_t out[16][16];
                if (amd_sparse(F, ps[i], key, out) != 0) {
                    ++ta.rc_fail;
                    return;
                }
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 16; ++c) {
                        const uint32_t m = sparse_model(F.sf, key, ps[i], r, c);
                        ++ta.n;
                        if (m != out[r][c]) {
                            ++ta.bad;
                            const int df = arith::sdense(F.sf), eb = arith::elem_bits(df);
                            uint32_t b32[32];
                            for (int j = 0; j < 16; ++j) {
                                b32[j] = ps[i].p.b[c][j];
                                b32[16 + j] = ps[i].b2[c][j];
                            }
                            static const char * const vn[] = {"valid-index", "valid-index", "random-index", "equal-pair-index", "decreasing-pair-index"};
                            Miss ms{std::string(vn[variant]) + " key" + std::to_string(key), (long) i,
                                    r, c, m, out[r][c], ""};
                            ms.inputs = "A " + hexs(ps[i].p.a[r], 16, eb) + "| B(32) " + hexs(b32, 32, eb) + "| C " +
                                        hexs(&ps[i].p.c[r][c], 1, cd_bits(df)) + "| idx lanes r,r+16 " +
                                        hexs(&ps[i].idx[r], 1, 32) + hexs(&ps[i].idx[r + 16], 1, 32);
                            keep(ta, std::move(ms));
                        }
                    }
            });
            g_break = saved;
            Tally t;
            for (auto & x : part) merge(t, x);
            static const char * const vl[] = {"valid indices, key 0", "valid indices, key 1", "random indices, key 0 (invalid: AMD unspecified)",
                                              "equal-position pairs, key 0 (invalid)", "decreasing pairs, key 0 (invalid)"};
            report(F.name, vl[variant], t);
            if (variant < 2) {
                merge(all, t);
            } else {
                print_first(F.name, t);
            }
        }
        report(F.name, "TOTAL valid indices", all);
        print_first(F.name, all);
        total_bad += all.bad;
    }
    std::printf("SPARSE_TOTAL_DISAGREEMENTS_VALID %ld\n", total_bad);
    return 0;
}

// ------------------------------------------------------------------------------------------------ edge mode
static const Dense * dense_of(int f) {
    for (const Dense & d : dense_forms)
        if (d.f == f) return &d;
    return nullptr;
}

static int mode_edge() {
    int ci = 0, bad_card_s = 0, bad_card_a = 0;
    for (const EdgeCase & cs : edge_cases) {
        const Dense & F = *dense_of(cs.form);
        Prob p{};
        for (int k = 0; k < 2; ++k) {
            p.a[0][cs.slot + k] = cs.a[k];
            p.b[0][cs.slot + k] = cs.b[k];
        }
        p.c[0][0] = cs.c;
        uint32_t out[16][16];
        amd_dense(F, p, SEM_EXACT, 0, out);
        const uint32_t s = study_dense(F, p, 0, 0, nullptr), a = out[0][0];
        bad_card_s += s != cs.want;
        bad_card_a += a != cs.want;
        std::printf("EDGE case %d %s: recorded card 0x%08x study 0x%08x amd 0x%08x %s%s\n", ci++, cs.name, cs.want, s, a,
                    s == cs.want ? "study=card " : "study!=card ", a == cs.want ? "amd=card" : "amd!=card");
    }
    std::vector<Prob> ps;
    for (const ad::Problem & p : edge_sweep()) ps.push_back(to_prob(p));
    Tally t;
    run_probs(*dense_of(arith::BF16), nullptr, "sweep", ps, t, false);
    report("bf16 sweep 2^-156..2^-140", "AMD vs study (card log: study 45056/45056)", t);
    print_first("bf16 sweep", t);
    std::printf("EDGE_SUMMARY cases_study_ne_card %d cases_amd_ne_card %d sweep_disagreements %ld\n", bad_card_s, bad_card_a, t.bad);
    return 0;
}

// ------------------------------------------------------------------------------------------------ gpu mode
static int mode_gpu(const char * dir) {
    DIR * dp = opendir(dir);
    if (!dp) {
        std::perror(dir);
        return 2;
    }
    std::vector<std::string> names;
    while (dirent * e = readdir(dp)) {
        std::string n = e->d_name;
        if (n.size() > 3 && n.substr(n.size() - 3) == ".in") names.push_back(n.substr(0, n.size() - 3));
    }
    closedir(dp);
    std::sort(names.begin(), names.end());
    long tot = 0, bs = 0, ba = 0;
    for (const std::string & nm : names) {
        FILE * fi = std::fopen((std::string(dir) + "/" + nm + ".in").c_str(), "rb");
        FILE * fo = std::fopen((std::string(dir) + "/" + nm + ".out").c_str(), "rb");
        uint32_t h[7];
        if (!fi || !fo || std::fread(h, 4, 7, fi) != 7) continue;
        const int gform = (int) h[0], steps = (int) h[2], n = (int) h[5], shift = (int) h[6];
        if (h[1] != 0 || h[3] != 0 || steps != 1 || gform > 7) {
            std::printf("GPU %-12s skipped (form %d mod %u use_mode %u steps %d)\n", nm.c_str(), gform, h[1], h[3], steps);
            continue;
        }
        std::vector<uint32_t> el((size_t) n * 33), out(n);
        if (std::fread(el.data(), 4, el.size(), fi) != el.size() || std::fread(out.data(), 4, n, fo) != (size_t) n) continue;
        std::fclose(fi);
        std::fclose(fo);
        const Dense & F = *dense_of(gform);  // the runner's form numbers are arith::Form
        long bad_s = 0, bad_a = 0;
        std::string firstbad;
        for (int w0 = 0; w0 < n; w0 += 16) {
            Prob p{};
            int cols[16];
            for (int j = 0; j < 16 && w0 + j < n; ++j) {
                const uint32_t * e = &el[(size_t) (w0 + j) * 33];
                const int col = (j + shift) % 16;
                cols[j] = col;
                for (int s = 0; s < 16; ++s) {
                    p.a[j][s] = e[s];
                    p.b[col][s] = e[16 + s];
                }
                p.c[j][col] = e[32];
            }
            uint32_t d[16][16];
            amd_dense(F, p, SEM_EXACT, 0, d);
            for (int j = 0; j < 16 && w0 + j < n; ++j) {
                const uint32_t card = out[w0 + j], s = study_dense(F, p, j, cols[j], nullptr), a = d[j][cols[j]];
                const uint32_t mask = cd_bits(F.f) == 16 ? 0xFFFFu : 0xFFFFFFFFu;
                bad_s += (s & mask) != (card & mask);
                bad_a += (a & mask) != (card & mask);
                if ((s & mask) != (card & mask) || (a & mask) != (card & mask)) {
                    if (firstbad.empty()) {
                        char b[160];
                        std::snprintf(b, sizeof b, " first at element %d: card 0x%08x study 0x%08x amd 0x%08x", w0 + j, card, s, a);
                        firstbad = b;
                    }
                }
            }
        }
        std::printf("GPU %-12s form %-26s elements %7d study!=card %6ld amd!=card %6ld%s\n", nm.c_str(), F.name, n, bad_s, bad_a,
                    firstbad.c_str());
        tot += n;
        bs += bad_s;
        ba += bad_a;
    }
    std::printf("GPU_TOTAL elements %ld study_ne_card %ld amd_ne_card %ld\n", tot, bs, ba);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: rj_vs_model dense|sparse|edge|gpu DIR [--break]\n");
        return 2;
    }
    for (int i = 2; i < argc; ++i) g_brk = g_brk || std::string(argv[i]) == "--break";
    g_cpu = rj_arith_init_cpu_flags();
    if (const char * lv = std::getenv("EXTCHECK_CPU")) {  // baseline | v3 | v4 | zen4: AMD dispatches by CPU level
        const std::string l = lv;
        const uint64_t level = l == "baseline" ? RJ_ARITH_CPU_BASELINE : l == "v3" ? RJ_ARITH_CPU_X86_64_V3 : l == "v4" ? RJ_ARITH_CPU_X86_64_V4 : RJ_ARITH_CPU_X86_64_ZEN4;
        g_cpu = (g_cpu & ~RJ_ARITH_CPU_MASK) | level;
    }
    std::printf("CPU_FLAGS 0x%llx (%s)\n", (unsigned long long) g_cpu, rj_arith_cpu_name(g_cpu));
    const std::string m = argv[1];
    std::printf("MODE %s%s\n", m.c_str(), g_brk ? " --break" : "");
    if (m == "dense") return mode_dense();
    if (m == "sparse") return mode_sparse();
    if (m == "special") return mode_special();
    if (m == "edge") return mode_edge();
    if (m == "gpu" && argc >= 3) return mode_gpu(argv[2]);
    return 2;
}
