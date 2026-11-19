// Leftovers C, lane A, CD-3: combined modifier bits on v_dot2_f32_f16, and neg_hi on the integer dot forms.
//
// Usage: left_dotm [--break]
//   Every variant of left_dotm_variants.h runs on ROWS rows of random data (one instruction per row and lane) and is
//   compared bit for bit with the plain instruction run on data changed by a model, so no model of the adder is
//   needed (the method of conversions-and-dots 5a, which measured each bit alone):
//   - v_dot2_f32_f16 (64 combinations of neg_lo, neg_hi, op_sel and a cleared op_sel_hi on A and B, with the C bits):
//     composition of the single-bit effects: element k of source s reads half (k = lo ? op_sel : op_sel_hi) and is
//     negated when (k = lo ? neg_lo : neg_hi); C is |C| when neg_hi[2], then negated when neg_lo[2]. Verdict
//     dotm_compose: 0 mismatches in every combination.
//   - v_dot4_i32_iu8, v_dot8_i32_iu4 with neg_hi on A, B or C, beside every neg_lo setting (neg_lo = signed): two
//     candidates, neg_hi ignored, or neg_hi OR-ed into the sign setting. Verdict dotm_int_neghi: every case equals at
//     least one candidate on all rows (which one is INFO).
//   --break uses the composition model with op_sel ignored when a neg bit is set on that source: dotm_compose FAILs.
#include "../harness.h"
#include "left_dotm_variants.h"

#include <cstdarg>
#include <random>
#include <string>

namespace {

constexpr int ROWS = 32 * 4096;

#define DOTM_FK(ID, NL0, NL1, NL2, NH0, NH1, NH2, OS0, OS1, OSH0, OSH1, MODS)                                       \
    __global__ __launch_bounds__(32) void kf_##ID(const uint32_t * a, const uint32_t * b, const uint32_t * c,       \
                                                  uint32_t * d) {                                                  \
        for (int r = threadIdx.x; r < ROWS; r += 32) {                                                             \
            uint32_t o;                                                                                            \
            asm volatile("v_dot2_f32_f16 %0, %1, %2, %3 " MODS : "=v"(o) : "v"(a[r]), "v"(b[r]), "v"(c[r]));       \
            d[r] = o;                                                                                              \
        }                                                                                                          \
    }
DOTM_F16(DOTM_FK)

__global__ __launch_bounds__(32) void kf_plain(const uint32_t * a, const uint32_t * b, const uint32_t * c, uint32_t * d) {
    for (int r = threadIdx.x; r < ROWS; r += 32) {
        uint32_t o;
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(o) : "v"(a[r]), "v"(b[r]), "v"(c[r]));
        d[r] = o;
    }
}

#define DOTM_IK(ID, W4, NL0, NL1, NH0, NH1, NH2, TEXT)                                                              \
    __global__ __launch_bounds__(32) void ki_##ID(const uint32_t * a, const uint32_t * b, const uint32_t * c,       \
                                                  uint32_t * d) {                                                  \
        for (int r = threadIdx.x; r < ROWS; r += 32) {                                                             \
            uint32_t o;                                                                                            \
            asm volatile(TEXT : "=v"(o) : "v"(a[r]), "v"(b[r]), "v"(c[r]));                                        \
            d[r] = o;                                                                                              \
        }                                                                                                          \
    }
DOTM_INT(DOTM_IK)

using K = void (*)(const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *);
struct FV {
    int id, nl[3], nh[3], os[2], osh[2];
    const char * mods;
    K k;
};
#define DOTM_FROW(ID, NL0, NL1, NL2, NH0, NH1, NH2, OS0, OS1, OSH0, OSH1, MODS) \
    {ID, {NL0, NL1, NL2}, {NH0, NH1, NH2}, {OS0, OS1}, {OSH0, OSH1}, MODS, kf_##ID},
const FV fvars[] = {DOTM_F16(DOTM_FROW)};
struct IV {
    int id, w4, nl[2], nh[3];
    const char * text;
    K k;
};
#define DOTM_IROW(ID, W4, NL0, NL1, NH0, NH1, NH2, TEXT) {ID, W4, {NL0, NL1}, {NH0, NH1, NH2}, TEXT, ki_##ID},
const IV ivars[] = {DOTM_INT(DOTM_IROW)};

uint32_t *da, *db, *dc, *dd;

std::vector<uint32_t> run(K k, const std::vector<uint32_t> & a, const std::vector<uint32_t> & b,
                          const std::vector<uint32_t> & c) {
    CHECK(hipMemcpy(da, a.data(), ROWS * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), ROWS * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, c.data(), ROWS * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, da, db, dc, dd);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> d(ROWS);
    CHECK(hipMemcpy(d.data(), dd, ROWS * 4, hipMemcpyDeviceToHost));
    return d;
}

// one source word after the modifiers of source s: element lo reads half os, element hi reads half osh, each sign
// flipped by its neg bit
uint32_t xform(uint32_t w, int os, int osh, int nlo, int nhi) {
    uint32_t lo = os ? w >> 16 : w & 0xFFFF, hi = osh ? w >> 16 : w & 0xFFFF;
    lo ^= nlo ? 0x8000 : 0;
    hi ^= nhi ? 0x8000 : 0;
    return lo | hi << 16;
}

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    bool brk = false;
    for (int i = 1; i < argc; ++i) brk |= std::strcmp(argv[i], "--break") == 0;
    CHECK(hipMalloc(&da, ROWS * 4));
    CHECK(hipMalloc(&db, ROWS * 4));
    CHECK(hipMalloc(&dc, ROWS * 4));
    CHECK(hipMalloc(&dd, ROWS * 4));
    std::mt19937 rng(77);
    // FP16 halves: random sign and mantissa, exponent 5 to 25 (finite, products far from overflow); C: an FP32 with
    // exponent -20 to +20
    auto h16 = [&]() { return (uint32_t) ((rng() & 1) << 15 | (5 + rng() % 21) << 10 | (rng() & 0x3FF)); };
    std::vector<uint32_t> a(ROWS), b(ROWS), c(ROWS);
    for (int r = 0; r < ROWS; ++r) {
        a[r] = h16() | h16() << 16;
        b[r] = h16() | h16() << 16;
        c[r] = (rng() & 1) << 31 | (uint32_t) (127 - 20 + rng() % 41) << 23 | (rng() & 0x7FFFFF);
    }
    int bad_vars = 0;
    long long max_bad = 0, min_diff_plain = ROWS;
    for (const FV & v : fvars) {
        int os[2] = {v.os[0], v.os[1]}, osh[2] = {v.osh[0], v.osh[1]};
        if (brk) {
            for (int s = 0; s < 2; ++s) {
                if (v.nl[s] || v.nh[s]) {
                    os[s] = 0;
                    osh[s] = 1;
                }
            }
        }
        std::vector<uint32_t> a2(ROWS), b2(ROWS), c2(ROWS);
        for (int r = 0; r < ROWS; ++r) {
            a2[r] = xform(a[r], os[0], osh[0], v.nl[0], v.nh[0]);
            b2[r] = xform(b[r], os[1], osh[1], v.nl[1], v.nh[1]);
            uint32_t cc = c[r];
            if (v.nh[2]) cc &= 0x7FFFFFFFu;
            if (v.nl[2]) cc ^= 0x80000000u;
            c2[r] = cc;
        }
        const std::vector<uint32_t> got = run(v.k, a, b, c), want = run(kf_plain, a2, b2, c2),
                                    plain = run(kf_plain, a, b, c);
        long long bad = 0, diff = 0;
        for (int r = 0; r < ROWS; ++r) {
            bad += got[r] != want[r];
            diff += got[r] != plain[r];
        }
        bad_vars += bad != 0;
        max_bad = std::max(max_bad, bad);
        if (v.mods[0]) min_diff_plain = std::min(min_diff_plain, diff);  // an empty text is the plain instruction
        std::printf("RESULT dotm f16 %2d %-62s mismatches %lld of %d; differs from the plain instruction on %lld rows\n",
                    v.id, v.mods[0] ? v.mods : "(none: the plain instruction, a control)", bad, ROWS, diff);
    }
    check("dotm_compose", bad_vars == 0,
          "%zu combinations of v_dot2_f32_f16 modifier bits against the composition of the single-bit effects%s: %d "
          "combinations with mismatches (largest %lld of %d rows); each combination with a bit set differs from the "
          "plain instruction on at least %lld rows",
          sizeof fvars / sizeof fvars[0], brk ? " (deliberate break: op_sel ignored when a neg bit is set)" : "",
          bad_vars, max_bad, ROWS, min_diff_plain);
    // integer forms: references are the variants without neg_hi
    std::vector<uint32_t> ia(ROWS), ib(ROWS), ic(ROWS);
    for (int r = 0; r < ROWS; ++r) {
        ia[r] = rng();
        ib[r] = rng();
        ic[r] = rng() % 2000001 - 1000000;
    }
    int unexplained = 0;
    std::string classes;
    for (const IV & v : ivars) {
        if (!v.nh[0] && !v.nh[1] && !v.nh[2]) continue;
        auto ref = [&](int n0, int n1) -> K {
            for (const IV & u : ivars) {
                if (u.w4 == v.w4 && u.nl[0] == n0 && u.nl[1] == n1 && !u.nh[0] && !u.nh[1] && !u.nh[2]) return u.k;
            }
            return nullptr;
        };
        const std::vector<uint32_t> got = run(v.k, ia, ib, ic);
        const std::vector<uint32_t> ign = run(ref(v.nl[0], v.nl[1]), ia, ib, ic);
        const std::vector<uint32_t> orr = run(ref(v.nl[0] | v.nh[0], v.nl[1] | v.nh[1]), ia, ib, ic);
        long long bi = 0, bo = 0;
        for (int r = 0; r < ROWS; ++r) {
            bi += got[r] != ign[r];
            bo += got[r] != orr[r];
        }
        const char * cls = bi == 0 && bo == 0 ? "both (the bits change nothing here)" : bi == 0 ? "ignored" : bo == 0 ? "OR-ed into the sign setting" : "NEITHER";
        unexplained += bi != 0 && bo != 0;
        std::printf("RESULT dotm int %-58s vs ignored: %lld rows differ; vs OR-ed into neg_lo: %lld -> %s\n", v.text, bi, bo,
                    cls);
    }
    check("dotm_int_neghi", unexplained == 0,
          "neg_hi on v_dot4_i32_iu8 and v_dot8_i32_iu4 (A, B, C, with every neg_lo setting): %d cases match neither "
          "candidate (ignored, or OR-ed into the sign setting)",
          unexplained);
    return fails ? 1 : 0;
}
