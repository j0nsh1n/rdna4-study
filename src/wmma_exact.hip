// What a 16x16x16 matrix instruction computes, bit for bit: D = A * B + C with F32 accumulation.
// gfx1100: v_wmma_f32_16x16x16_f16 (gfx11 builtin). gfx1201: the F16, BF16 and FP8 forms (gfx12 builtins).
//
// The host packs A, B and C into each lane's register image with a layout table per (arch, element size), one
// wave runs one problem, and the host unpacks D with the result layout. Before anything else is trusted, a layout
// check multiplies by identity matrices (D must equal the other operand exactly); if it fails, no model fit is
// reported. Limitation: a K order that differs from the table in the same way for A and B passes that check, since
// the products pair up anyway; only the model fits below are sensitive to it. For gfx12 16-bit data AMD's guide
// table and AMD's GPUOpen samples (and LLVM, llama.cpp) describe different K orders inside a lane, so the fits are
// computed under both orders.
//
// CPU models (products are exact in every case; R = round to nearest even to F32):
//   exact     R(C + sum of all products), one rounding (exact arithmetic in a 640-bit accumulator)
//   fma_seq   FMA chain from C in K order 0..15;  fma_rev  the same, K order 15..0
//   g2, g4, g8   C, then each group of 2, 4 or 8 products in K order added exactly and rounded once
//   sum_then_c   R(R(sum of products) + C);  tree  pairwise sums with R at every node, C added last
// Data classes: known cases (two-term sums from rdna3-fp8/measurements/matrix-arithmetic.md 3.1, B = all ones),
// random, adversarial (cancellation, large and small terms mixed, large C, C cancelling the sum).
// Known answers on gfx1100 (measured with Vulkan on the RX 7900 XT): no model fits, and 1 + (-1) gives -2^-24.
//
// Usage: wmma_exact [run|selftest|emulate11|emulate12] [--break]
//   run       on the card (default)
//   selftest  CPU only: the exact accumulator against hand-worked cases, the layout tables for full coverage, and
//             a CPU emulation round trip through the register images.
//   emulate11, emulate12   CPU only: the whole run with an exact-sum stand-in for the instruction (tests the
//             packing, fitting and report code; "exact" must fit every output).
//   --break   perturb the expectation on purpose (the exact model gains one unit, the identity check expects one
//             unit more); the verdict must turn to FAIL.
#include "e4m3_ref.h"
#include "harness.h"

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));

enum Fmt { F16 = 0, BF16 = 1, FP8 = 2 };
static const char * fmt_name[3] = {"f16", "bf16", "fp8"};

// ---------------------------------------------------------------------------------------------------------------
// Device: one wave per problem. Register images: A and B NW words per lane, C and D 8 floats per lane.

template <int FMT>
__global__ __launch_bounds__(32) void mm_kernel(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                               const float * __restrict__ C, float * __restrict__ D) {
    const int p = blockIdx.x, lane = threadIdx.x;
    v8f c;
    for (int i = 0; i < 8; ++i) {
        c[i] = C[(p * 32 + lane) * 8 + i];
    }
#if defined(__GFX11__)
    if constexpr (FMT == F16) {
        uint32_t a[8], b[8];
        for (int i = 0; i < 8; ++i) {
            a[i] = A[(p * 32 + lane) * 8 + i];
            b[i] = B[(p * 32 + lane) * 8 + i];
        }
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(v16h, a), __builtin_bit_cast(v16h, b), c);
    }
#elif defined(__GFX12__)
    if constexpr (FMT == FP8) {
        const v2i a = {(int) A[(p * 32 + lane) * 2], (int) A[(p * 32 + lane) * 2 + 1]};
        const v2i b = {(int) B[(p * 32 + lane) * 2], (int) B[(p * 32 + lane) * 2 + 1]};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    } else {
        uint32_t a[4], b[4];
        for (int i = 0; i < 4; ++i) {
            a[i] = A[(p * 32 + lane) * 4 + i];
            b[i] = B[(p * 32 + lane) * 4 + i];
        }
        if constexpr (FMT == F16) {
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(v8h, a), __builtin_bit_cast(v8h, b), c);
        } else {
            c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8s, a), __builtin_bit_cast(v8s, b), c);
        }
    }
#endif
    for (int i = 0; i < 8; ++i) {
        D[(p * 32 + lane) * 8 + i] = c[i];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Layout tables. Element (r, k) of A, or (k, c) of B with c in place of r.

enum Arch { GFX11 = 0, GFX12 = 1 };

struct Slot {
    int lane, word, bit;
};

static int words_ab(Arch a, Fmt f) { return a == GFX11 ? 8 : f == FP8 ? 2 : 4; }

static int elem_bits(Fmt f) { return f == FP8 ? 8 : 16; }

// gfx11 (16-bit only): row r in lane r and again in lane r + 16 (replication), K in order, two per word.
// gfx12 16-bit (GPUOpen / LLVM / llama.cpp reading): lanes 0-15 hold K 0-7, lanes 16-31 K 8-15, two per word.
// gfx12 8-bit (guide table): lane {k[3], r[3:0]}, word k[2], byte k[1:0], which is the same K-contiguous order.
static std::vector<Slot> slots_ab(Arch a, Fmt f, int r, int k) {
    if (a == GFX11) {
        return {{r, k / 2, (k % 2) * 16}, {r + 16, k / 2, (k % 2) * 16}};
    }
    const int lane = r + 16 * (k >> 3), e = k & 7;
    if (f == FP8) {
        return {{lane, e >> 2, (e & 3) * 8}};
    }
    return {{lane, e >> 1, (e & 1) * 16}};
}

// Result and accumulator: gfx11 lane c + 16 * (r % 2), register r / 2; gfx12 lane 16 * (r / 8) + c, register r % 8.
static Slot slot_cd(Arch a, int r, int c) {
    return a == GFX11 ? Slot{c + 16 * (r & 1), r >> 1, 0} : Slot{16 * (r >> 3) + c, r & 7, 0};
}

// For gfx12 16-bit data: the K index AMD's guide table would give the element this program stores as K = k.
// Guide: lane half = k[2], register {k[3], k[1]}, half-word k[0].
static int guide_k_of_contig(int k) {
    const int h = k >> 3, e = k & 7;
    return ((e >> 2) << 3) | (h << 2) | (e & 3);
}

// ---------------------------------------------------------------------------------------------------------------
// Values. Each format element is kept as raw bits; its exact value as a double.

static float half_to_float(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

static double value_of(Fmt f, uint32_t bits) {
    if (f == F16) {
        return half_to_float((uint16_t) bits);
    }
    if (f == BF16) {
        return __builtin_bit_cast(float, bits << 16);
    }
    return e4m3_decode((uint8_t) bits);
}

static uint32_t bits_of_value(Fmt f, double v) {
    if (f == F16) {
        return __builtin_bit_cast(uint16_t, (_Float16) v);
    }
    if (f == BF16) {
        return __builtin_bit_cast(uint32_t, (float) v) >> 16;
    }
    return e4m3_encode(__builtin_bit_cast(uint32_t, (float) v), false);
}

struct Problem {
    uint32_t a[16][16], b[16][16];  // a[r][k], b[k][c] raw bits
    float c[16][16];
};

// ---------------------------------------------------------------------------------------------------------------
// Exact sums: products and C are exact doubles with at most 53 significant bits and exponents within
// [-300, 300]; positive and negative parts accumulate separately in 640-bit integers with the unit 2^-320.

struct Wide {
    static constexpr int L = 10, OFF = 320;
    uint64_t pos[L], neg[L];
    bool any_nonneg_zero;  // an exact zero result is -0 only if every term was -0
    void clear() {
        std::memset(pos, 0, sizeof pos);
        std::memset(neg, 0, sizeof neg);
        any_nonneg_zero = false;
    }
    static void add_mag(uint64_t * acc, uint64_t m, int sh) {
        const int w = sh >> 6, b = sh & 63;
        unsigned __int128 v = (unsigned __int128) m << b;
        for (int i = w; i < L && v; ++i) {
            const unsigned __int128 s = (unsigned __int128) acc[i] + (uint64_t) v;
            acc[i] = (uint64_t) s;
            v = (v >> 64) + (s >> 64);
        }
    }
    void add(double x) {
        if (x == 0) {
            any_nonneg_zero |= !std::signbit(x);
            return;
        }
        any_nonneg_zero = true;
        int e;
        const double fr = std::frexp(std::fabs(x), &e);  // x = fr * 2^e, fr in [0.5, 1)
        const uint64_t m = (uint64_t) std::ldexp(fr, 53);
        add_mag(x > 0 ? pos : neg, m, e - 53 + OFF);
    }
    // R32 of the exact value.
    float round() const {
        uint64_t d[L];
        bool negative = false;
        int cmp = 0;
        for (int i = L - 1; i >= 0 && !cmp; --i) {
            cmp = pos[i] > neg[i] ? 1 : pos[i] < neg[i] ? -1 : 0;
        }
        const uint64_t *big = cmp >= 0 ? pos : neg, *small = cmp >= 0 ? neg : pos;
        negative = cmp < 0;
        uint64_t borrow = 0;
        for (int i = 0; i < L; ++i) {
            const unsigned __int128 s = (unsigned __int128) big[i] - small[i] - borrow;
            d[i] = (uint64_t) s;
            borrow = (s >> 64) ? 1 : 0;
        }
        int p = -1;
        for (int i = L - 1; i >= 0; --i) {
            if (d[i]) {
                p = i * 64 + 63 - __builtin_clzll(d[i]);
                break;
            }
        }
        if (p < 0) {
            return any_nonneg_zero ? 0.0f : -0.0f;
        }
        auto bit = [&](int i) -> uint64_t { return i < 0 ? 0 : (d[i >> 6] >> (i & 63)) & 1; };
        auto any_below = [&](int i) {  // any bit with index < i
            if (i <= 0) {
                return false;
            }
            const int w = i >> 6;
            for (int j = 0; j < w; ++j) {
                if (d[j]) {
                    return true;
                }
            }
            return (i & 63) && (d[w] & ((1ull << (i & 63)) - 1));
        };
        const int lsb = std::max(p - 23, OFF - 149);
        uint64_t m = 0;
        for (int i = p; i >= lsb; --i) {
            m = (m << 1) | bit(i);
        }
        const bool rb = bit(lsb - 1), sticky = any_below(lsb - 1);
        if (rb && (sticky || (m & 1))) {
            ++m;
        }
        const float r = std::ldexp((float) m, lsb - OFF);
        return negative ? -r : r;
    }
};

static float exact_sum(const double * terms, int n) {
    Wide w;
    w.clear();
    for (int i = 0; i < n; ++i) {
        w.add(terms[i]);
    }
    return w.round();
}

// ---------------------------------------------------------------------------------------------------------------
// Models. prod[k] in the K order the hardware is assumed to use; c the accumulator input.

static bool g_break = false;

typedef float (*Model)(const double *, float);

static float m_exact(const double * p, float c) {
    double t[17];
    std::copy(p, p + 16, t);
    t[16] = c;
    const float r = exact_sum(t, 17);
    return g_break ? std::nextafter(r, INFINITY) : r;
}

static float r2(double a, double b) {
    const double t[2] = {a, b};
    return exact_sum(t, 2);
}

static float m_fma_seq(const double * p, float c) {
    float s = c;
    for (int k = 0; k < 16; ++k) {
        s = r2(p[k], s);
    }
    return s;
}

static float m_fma_rev(const double * p, float c) {
    float s = c;
    for (int k = 15; k >= 0; --k) {
        s = r2(p[k], s);
    }
    return s;
}

template <int G>
static float m_group(const double * p, float c) {
    float s = c;
    for (int g = 0; g < 16; g += G) {
        double t[G + 1];
        std::copy(p + g, p + g + G, t);
        t[G] = s;
        s = exact_sum(t, G + 1);
    }
    return s;
}

static float m_sum_then_c(const double * p, float c) {
    const float s = exact_sum(p, 16);
    return r2(s, c);
}

static float m_tree(const double * p, float c) {
    float v[16];
    for (int k = 0; k < 16; ++k) {
        v[k] = exact_sum(p + k, 1);
    }
    for (int n = 16; n > 1; n /= 2) {
        for (int i = 0; i < n / 2; ++i) {
            v[i] = r2(v[2 * i], v[2 * i + 1]);
        }
    }
    return r2(v[0], c);
}

static const std::vector<std::pair<const char *, Model>> models = {
    {"exact", m_exact}, {"fma_seq", m_fma_seq}, {"fma_rev", m_fma_rev}, {"g2", m_group<2>},
    {"g4", m_group<4>}, {"g8", m_group<8>},     {"sum_then_c", m_sum_then_c}, {"tree", m_tree}};

static bool same_bits(float a, float b) {
    return (std::isnan(a) && std::isnan(b)) || __builtin_bit_cast(uint32_t, a) == __builtin_bit_cast(uint32_t, b);
}

// ---------------------------------------------------------------------------------------------------------------
// Packing

struct Images {
    std::vector<uint32_t> a, b;
    std::vector<float> c, d;
};

static void put(std::vector<uint32_t> & img, size_t base, int nw, const Slot & s, uint32_t bits, int eb) {
    uint32_t & w = img[base + (size_t) s.lane * nw + s.word];
    const uint32_t mask = (eb == 32 ? 0xFFFFFFFFu : ((1u << eb) - 1)) << s.bit;
    w = (w & ~mask) | ((bits << s.bit) & mask);
}

static uint32_t get(const std::vector<uint32_t> & img, size_t base, int nw, const Slot & s, int eb) {
    return (img[base + (size_t) s.lane * nw + s.word] >> s.bit) & ((1u << eb) - 1);
}

static Images pack(Arch arch, Fmt f, const std::vector<Problem> & ps) {
    const int nw = words_ab(arch, f), eb = elem_bits(f);
    Images im;
    im.a.assign(ps.size() * 32 * nw, 0);
    im.b.assign(ps.size() * 32 * nw, 0);
    im.c.assign(ps.size() * 256, 0.0f);
    for (size_t p = 0; p < ps.size(); ++p) {
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < 16; ++k) {
                for (const Slot & s : slots_ab(arch, f, r, k)) {
                    put(im.a, p * 32 * nw, nw, s, ps[p].a[r][k], eb);
                    put(im.b, p * 32 * nw, nw, s, ps[p].b[k][r], eb);  // B column r, row k
                }
                const Slot s = slot_cd(arch, r, k);
                im.c[p * 256 + s.lane * 8 + s.word] = ps[p].c[r][k];
            }
        }
    }
    return im;
}

static float unpack_d(Arch arch, const std::vector<float> & d, size_t p, int r, int c) {
    const Slot s = slot_cd(arch, r, c);
    return d[p * 256 + s.lane * 8 + s.word];
}

// The CPU stand-in for the instruction, reading operands back out of the register images (selftest only).
static void emulate(Arch arch, Fmt f, size_t np, Images & im) {
    const int nw = words_ab(arch, f), eb = elem_bits(f);
    im.d.assign(np * 256, 0.0f);
    for (size_t p = 0; p < np; ++p) {
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                double prod[16];
                for (int k = 0; k < 16; ++k) {
                    const uint32_t a = get(im.a, p * 32 * nw, nw, slots_ab(arch, f, r, k)[0], eb);
                    const uint32_t b = get(im.b, p * 32 * nw, nw, slots_ab(arch, f, c, k)[0], eb);
                    prod[k] = value_of(f, a) * value_of(f, b);
                }
                const Slot s = slot_cd(arch, r, c);
                im.d[p * 256 + s.lane * 8 + s.word] = m_exact(prod, im.c[p * 256 + s.lane * 8 + s.word]);
            }
        }
    }
}

static void run_gpu(Arch arch, Fmt f, size_t np, Images & im) {
    uint32_t *da, *db;
    float *dc, *dd;
    CHECK(hipMalloc(&da, im.a.size() * 4));
    CHECK(hipMalloc(&db, im.b.size() * 4));
    CHECK(hipMalloc(&dc, im.c.size() * 4));
    CHECK(hipMalloc(&dd, im.c.size() * 4));
    CHECK(hipMemcpy(da, im.a.data(), im.a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, im.b.data(), im.b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, im.c.data(), im.c.size() * 4, hipMemcpyHostToDevice));
    if (f == F16) {
        mm_kernel<F16><<<np, 32>>>(da, db, dc, dd);
    } else if (f == BF16) {
        mm_kernel<BF16><<<np, 32>>>(da, db, dc, dd);
    } else {
        mm_kernel<FP8><<<np, 32>>>(da, db, dc, dd);
    }
    CHECK(hipDeviceSynchronize());
    im.d.resize(im.c.size());
    CHECK(hipMemcpy(im.d.data(), dd, im.d.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(da));
    CHECK(hipFree(db));
    CHECK(hipFree(dc));
    CHECK(hipFree(dd));
    (void) arch;
}

typedef std::function<void(Arch, Fmt, size_t, Images &)> Runner;

// ---------------------------------------------------------------------------------------------------------------
// Data

static uint32_t rand_bits(Fmt f, std::mt19937_64 & rng, int emin, int emax, bool positive = false) {
    const uint32_t sign = positive ? 0 : (uint32_t) (rng() & 1);
    const int e = emin + (int) (rng() % (uint64_t) (emax - emin + 1));
    if (f == F16) {
        return (sign << 15) | ((uint32_t) (e + 15) << 10) | (uint32_t) (rng() & 0x3FF);
    }
    if (f == BF16) {
        return (sign << 15) | ((uint32_t) (e + 127) << 7) | (uint32_t) (rng() & 0x7F);
    }
    uint32_t code;
    do {
        code = (uint32_t) (rng() & 0x7F);
    } while (code == 0x7F || (code >> 3) < (uint32_t) std::max(0, emin + 7) || (code >> 3) > (uint32_t) (emax + 7));
    return (sign << 7) | code;
}

static float rand_c(std::mt19937_64 & rng, int emin, int emax) {
    const int e = emin + (int) (rng() % (uint64_t) (emax - emin + 1));
    const uint32_t bits = (uint32_t) ((rng() & 1) << 31) | ((uint32_t) (e + 127) << 23) | (uint32_t) (rng() & 0x7FFFFF);
    return __builtin_bit_cast(float, bits);
}

static std::vector<Problem> random_problems(Fmt f, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Problem> ps(n);
    for (auto & p : ps) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                p.a[i][j] = rand_bits(f, rng, -4, 4);
                p.b[i][j] = rand_bits(f, rng, -4, 4);
                p.c[i][j] = (rng() & 7) == 0 ? 0.0f : rand_c(rng, -6, 6);
            }
        }
    }
    return ps;
}

// Kinds: 0 pairs that cancel exactly (x, -x) plus one small leftover; 1 terms from 2^-6 to 2^6 with mixed signs;
// 2 a large full-precision C (2^10 to 2^15) with small products; 3 C equal to minus the rounded product sum; 4 every term of one sign.
static std::vector<Problem> adversarial_problems(Fmt f, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Problem> ps(n);
    const uint32_t one = bits_of_value(f, 1.0);
    for (int pi = 0; pi < n; ++pi) {
        Problem & p = ps[pi];
        const int kind = pi % 5;
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                p.b[i][j] = one;
                p.c[i][j] = 0.0f;
                p.a[i][j] = rand_bits(f, rng, -4, 4);
            }
        }
        for (int r = 0; r < 16; ++r) {
            if (kind == 0) {
                for (int k = 0; k < 14; k += 2) {
                    p.a[r][k] = rand_bits(f, rng, -2, 4);
                    p.a[r][k + 1] = p.a[r][k] ^ (f == FP8 ? 0x80u : 0x8000u);
                }
                p.a[r][14] = rand_bits(f, rng, -6, -4);
                p.a[r][15] = 0;
            } else if (kind == 1) {
                for (int k = 0; k < 16; ++k) {
                    p.a[r][k] = rand_bits(f, rng, (k & 1) ? -6 : 3, (k & 1) ? -3 : 6);
                }
            } else if (kind == 2) {
                for (int k = 0; k < 16; ++k) {
                    p.a[r][k] = rand_bits(f, rng, -6, -2);
                }
                for (int c = 0; c < 16; ++c) {
                    p.c[r][c] = rand_c(rng, 10, 15);
                }
            } else if (kind == 3) {
                double s = 0;
                for (int k = 0; k < 16; ++k) {
                    s += value_of(f, p.a[r][k]);
                }
                for (int c = 0; c < 16; ++c) {
                    p.c[r][c] = (float) -s;
                }
            } else {
                for (int k = 0; k < 16; ++k) {
                    p.a[r][k] = rand_bits(f, rng, -4, 4, true) | ((pi & 8) ? (f == FP8 ? 0x80u : 0x8000u) : 0u);
                }
            }
        }
        if (kind != 2 && kind != 3) {
            // B with random positive entries keeps the cancellations but varies the products per column
            for (int i = 0; i < 16; ++i) {
                for (int j = 1; j < 16; ++j) {
                    p.b[i][j] = (kind == 0) ? one : rand_bits(f, rng, -1, 1, true);
                }
            }
        }
    }
    return ps;
}

struct KnownCase {
    const char * name;
    std::vector<std::pair<int, double>> terms;  // (K slot, value of A); B = 1
    double c;
    uint32_t gfx1100;  // F32 bits measured on the RX 7900 XT (matrix-arithmetic.md 3.1); 0xFFFFFFFF if not known
};

static const std::vector<KnownCase> known_cases = {
    {"3+5, C=0", {{0, 3}, {1, 5}}, 0, 0x41000000u},
    {"1+(-1), C=0", {{0, 1}, {1, -1}}, 0, 0xB3800000u},  // -2^-24
    {"8+(-1), C=0", {{0, 8}, {1, -1}}, 0, 0x40DFFFFFu},
    {"-8+1, C=0", {{0, -8}, {1, 1}}, 0, 0xC0E00001u},
    {"2+(-1), C=0", {{0, 2}, {1, -1}}, 0, 0x3F7FFFFEu},
    {"4+(-3), C=0", {{0, 4}, {1, -3}}, 0, 0x3F7FFFFCu},
    {"1+(-0.5), C=0", {{0, 1}, {1, -0.5}}, 0, 0x3EFFFFFEu},
    {"1@k0+(-0.5)@k8, C=0", {{0, 1}, {8, -0.5}}, 0, 0x3EFFFFFFu},
    {"1-1+1, C=0", {{0, 1}, {1, -1}, {2, 1}}, 0, 0x3F800000u},
    {"8, C=-1", {{0, 8}}, -1, 0x40E00002u},
    {"1, C=-0.5", {{0, 1}}, -0.5, 0x3F000002u},
    {"1, C=-2^-60", {{0, 1}}, -0x1p-60, 0x3F800001u},
    {"-1, C=-2^-30", {{0, -1}}, -0x1p-30, 0xBF7FFFFFu},
    {"32768, C=-1", {{0, 32768}}, -1, 0x46FFFE02u},
    {"1, C=-1", {{0, 1}}, -1, 0xFFFFFFFFu},
    {"0, C=-1", {{0, 0}}, -1, 0xBF800000u},
};

static std::vector<Problem> known_problem(Fmt f) {
    Problem p{};
    for (int r = 0; r < 16; ++r) {
        const KnownCase & kc = known_cases[r];
        for (int k = 0; k < 16; ++k) {
            p.a[r][k] = 0;
            for (int c = 0; c < 16; ++c) {
                p.b[k][c] = bits_of_value(f, 1.0);
            }
        }
        for (auto [k, v] : kc.terms) {
            // FP8 cannot hold 32768; that row then holds 256 and has no known answer
            p.a[r][k] = bits_of_value(f, f == FP8 && v > 448 ? 256.0 : v);
        }
        for (int c = 0; c < 16; ++c) {
            p.c[r][c] = (float) kc.c;
        }
    }
    return {p};
}

// ---------------------------------------------------------------------------------------------------------------

static void products(Fmt f, const Problem & p, int r, int c, const int * order, double * out) {
    for (int i = 0; i < 16; ++i) {
        const int k = order[i];
        out[i] = value_of(f, p.a[r][k]) * value_of(f, p.b[k][c]);
    }
}

// Identity checks: D must equal B for A = I, and A for B = I (C = 0, positive random entries).
static bool layout_check(Arch arch, Fmt f, const Runner & run, std::string & why) {
    std::mt19937_64 rng(7);
    const uint32_t one = bits_of_value(f, 1.0);
    std::vector<Problem> ps(4);
    for (int t = 0; t < 4; ++t) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                const uint32_t v = rand_bits(f, rng, -3, 3, true);
                ps[t].a[i][j] = t < 2 ? (i == j ? one : 0) : v;
                ps[t].b[i][j] = t < 2 ? v : (i == j ? one : 0);
                ps[t].c[i][j] = 0.0f;
            }
        }
    }
    Images im = pack(arch, f, ps);
    run(arch, f, ps.size(), im);
    long bad = 0;
    for (int t = 0; t < 4; ++t) {
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                double want = t < 2 ? value_of(f, ps[t].b[r][c]) : value_of(f, ps[t].a[r][c]);
                if (g_break) {
                    want = std::nextafter((float) want, INFINITY);
                }
                const float got = unpack_d(arch, im.d, t, r, c);
                if (!same_bits(got, (float) want)) {
                    if (bad < 4) {
                        std::printf("  layout mismatch %s %s at (%d,%d): got %a want %a\n", fmt_name[f],
                                    t < 2 ? "I*B" : "A*I", r, c, got, want);
                    }
                    ++bad;
                }
            }
        }
    }
    why = bad ? std::to_string(bad) + " of 1024 identity outputs wrong" : "identity checks exact";
    return bad == 0;
}

struct Fit {
    std::string best;
    double best_frac;
};

static Fit fit(Arch arch, Fmt f, const char * cls, const std::vector<Problem> & ps, const Images & im,
               const char * order_name, const int * order) {
    std::vector<long> match(models.size(), 0);
    long total = 0;
    for (size_t p = 0; p < ps.size(); ++p) {
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                double prod[16];
                products(f, ps[p], r, c, order, prod);
                const float got = unpack_d(arch, im.d, p, r, c);
                for (size_t m = 0; m < models.size(); ++m) {
                    match[m] += same_bits(got, models[m].second(prod, ps[p].c[r][c]));
                }
                ++total;
            }
        }
    }
    std::printf("RESULT fit %s %-11s order %-6s outputs %ld:", fmt_name[f], cls, order_name, total);
    Fit best{"", -1};
    for (size_t m = 0; m < models.size(); ++m) {
        const double frac = (double) match[m] / total;
        std::printf(" %s %.4f", models[m].first, frac);
        if (frac > best.best_frac) {
            best = {models[m].first, frac};
        }
    }
    std::printf("\n");
    return best;
}

static bool known(Arch arch, Fmt f, const Runner & run, std::string & text) {
    std::vector<Problem> ps = known_problem(f);
    Images im = pack(arch, f, ps);
    run(arch, f, 1, im);
    bool ok = true;
    int checked = 0;
    for (int r = 0; r < 16; ++r) {
        const KnownCase & kc = known_cases[r];
        const float got = unpack_d(arch, im.d, 0, r, 0);
        bool cols_same = true;
        for (int c = 1; c < 16; ++c) {
            cols_same = cols_same && same_bits(unpack_d(arch, im.d, 0, r, c), got);
        }
        double prod[16];
        const int order[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        products(f, ps[0], r, 0, order, prod);
        const float ex = m_exact(prod, ps[0].c[r][0]);
        const bool has_known = arch == GFX11 && f == F16 && kc.gfx1100 != 0xFFFFFFFFu;
        const bool match = !has_known || __builtin_bit_cast(uint32_t, got) == kc.gfx1100;
        std::printf("RESULT case %s %-22s gpu %a (0x%08x) exact %a%s%s%s\n", fmt_name[f], kc.name, got,
                    __builtin_bit_cast(uint32_t, got), ex, has_known ? (match ? "  = RX 7900 XT" : "  != RX 7900 XT") : "",
                    cols_same ? "" : "  columns differ", f == FP8 && r == 13 ? "  (256, not 32768)" : "");
        ok = ok && match;
        checked += has_known;
    }
    if (arch == GFX11 && f == F16) {
        const float g = unpack_d(arch, im.d, 0, 1, 0);
        text += std::string("; 1+(-1) gives ") + (g == -0x1p-24f ? "-2^-24 as on the RX 7900 XT" : "something else") +
                ", " + std::to_string(checked) + " known cases " + (ok ? "all match" : "DIFFER");
    }
    return ok;
}

static bool probe(Arch arch, const std::vector<Fmt> & fmts, const Runner & run, std::string & text) {
    bool ok = true;
    for (Fmt f : fmts) {
        std::string why;
        const bool lay = layout_check(arch, f, run, why);
        std::printf("RESULT layout %s %s: %s\n", fmt_name[f], lay ? "confirmed" : "REJECTED", why.c_str());
        text += std::string("; ") + fmt_name[f] + " layout " + (lay ? "ok" : "REJECTED");
        if (!lay) {
            ok = false;
            continue;
        }
        ok = known(arch, f, run, text) && ok;
        int contig[16], guide[16];
        for (int k = 0; k < 16; ++k) {
            contig[k] = k;
            guide[guide_k_of_contig(k)] = k;
        }
        const bool two_orders = arch == GFX12 && f != FP8;
        for (const char * cls : {"random", "adversarial"}) {
            const std::vector<Problem> ps = std::string(cls) == "random" ? random_problems(f, 1024, 11)
                                                                         : adversarial_problems(f, 1024, 12);
            Images im = pack(arch, f, ps);
            run(arch, f, ps.size(), im);
            const Fit a = fit(arch, f, cls, ps, im, "k", contig);
            char buf[160];
            std::snprintf(buf, sizeof buf, ", %s best %s %.3f", cls, a.best.c_str(), a.best_frac);
            text += buf;
            if (two_orders) {
                const Fit b = fit(arch, f, cls, ps, im, "guide", guide);
                std::snprintf(buf, sizeof buf, " (guide K order: %s %.3f)", b.best.c_str(), b.best_frac);
                text += buf;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// CPU self-test

static bool selftest(std::string & text) {
    long bad = 0;
    auto expect = [&](const char * what, float got, float want) {
        if (!same_bits(got, want)) {
            std::printf("  selftest %s: got %a want %a\n", what, got, want);
            ++bad;
        }
    };
    double z[16];
    std::fill(z, z + 16, -0.0);
    auto one_term = [&](double t, float c) {
        double p[16] = {};
        p[0] = t;
        return m_exact(p, c);
    };
    expect("1 + (-1) = +0", one_term(1, -1.0f), 0.0f);
    expect("2^24 + 1 ties to even", one_term(0x1p24, 1.0f), 0x1p24f);
    expect("2^24 + 3 rounds up", one_term(0x1p24, 3.0f), 0x1p24f + 4);
    expect("1 + 2^-60", one_term(0x1p-60, 1.0f), 1.0f);
    expect("2^30 - 2^30 + 2^-40", [&] {
        double p[16] = {0x1p30, -0x1p30, 0x1p-40};
        return m_exact(p, 0.0f);
    }(), 0x1p-40f);
    expect("1 + 2^-24 + 2^-24", [&] {
        double p[16] = {0x1p-24, 0x1p-24};
        return m_exact(p, 1.0f);
    }(), 1.0f + 0x1p-23f);
    expect("subnormal 2^-149 * 3 / 2 ties to even", one_term(0x1.8p-149, 0.0f), 0x1p-148f);
    expect("all -0 gives -0", m_exact(z, -0.0f), -0.0f);
    expect("fma_seq: 2^24, then +1 twice", [&] {
        double p[16] = {1, 1};
        return m_fma_seq(p, 0x1p24f);
    }(), 0x1p24f);
    expect("g2: 2^24 + (1 + 1)", [&] {
        double p[16] = {1, 1};
        return m_group<2>(p, 0x1p24f);
    }(), 0x1p24f + 2);
    // random cases where double is exact (few bits, narrow range) must agree with the wide accumulator
    std::mt19937_64 rng(3);
    for (int i = 0; i < 100000; ++i) {
        double p[16];
        double s = 0;
        for (int k = 0; k < 16; ++k) {
            p[k] = std::ldexp((double) ((int64_t) (rng() % 2001) - 1000), (int) (rng() % 20) - 10);
            s += p[k];
        }
        const float c = (float) std::ldexp((double) ((int64_t) (rng() % 2001) - 1000), (int) (rng() % 20) - 10);
        expect("double-exact random sum", m_exact(p, c), (float) (s + c));
        if (bad > 10) {
            break;
        }
    }
    // layout tables: every bit of every register image used exactly once; then a full round trip through the images with the CPU stand-in for the instruction
    for (Arch arch : {GFX11, GFX12}) {
        for (Fmt f : {F16, BF16, FP8}) {
            if (arch == GFX11 && f != F16) {
                continue;
            }
            const int nw = words_ab(arch, f), eb = elem_bits(f);
            std::vector<int> use(32 * nw * 32, 0), dus(32 * 8, 0);
            for (int r = 0; r < 16; ++r) {
                for (int k = 0; k < 16; ++k) {
                    for (const Slot & s : slots_ab(arch, f, r, k)) {
                        for (int b = 0; b < eb; ++b) {
                            ++use[(s.lane * nw + s.word) * 32 + s.bit + b];
                        }
                    }
                    const Slot s = slot_cd(arch, r, k);
                    ++dus[s.lane * 8 + s.word];
                }
            }
            // gfx11 replication puts the second copy in another lane, so every bit is still used once
            const int want_use = 1;
            long cover_bad = 0;
            for (int u : use) {
                cover_bad += u != want_use;
            }
            for (int u : dus) {
                cover_bad += u != 1;
            }
            if (cover_bad) {
                std::printf("  selftest layout %s %s: %ld register bits not covered as expected\n",
                            arch == GFX11 ? "gfx11" : "gfx12", fmt_name[f], cover_bad);
                bad += cover_bad;
            }
            std::vector<Problem> ps = random_problems(f, 8, 5);
            Images im = pack(arch, f, ps);
            emulate(arch, f, ps.size(), im);
            const int order[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            for (size_t p = 0; p < ps.size(); ++p) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        double prod[16];
                        products(f, ps[p], r, c, order, prod);
                        const double direct = [&] {
                            double t[17];
                            std::copy(prod, prod + 16, t);
                            t[16] = ps[p].c[r][c];
                            return (double) exact_sum(t, 17);
                        }();
                        expect("round trip through the register images", unpack_d(arch, im.d, p, r, c), (float) direct);
                    }
                }
            }
            std::string why;
            const Runner cpu = [](Arch a, Fmt fm, size_t np, Images & i) { emulate(a, fm, np, i); };
            if (!g_break && !layout_check(arch, f, cpu, why)) {
                std::printf("  selftest identity check on the CPU stand-in failed: %s\n", why.c_str());
                ++bad;
            }
        }
    }
    text = "CPU self-test: " + std::to_string(bad) + " failures" + (g_break ? " (expectation broken on purpose)" : "");
    return bad == 0;
}

int main(int argc, char ** argv) {
    std::string mode = "run";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--break") {
            g_break = true;
        } else {
            mode = a;
        }
    }
    std::string text;
    bool ok;
    if (mode == "selftest") {
        ok = selftest(text);
    } else if (mode == "emulate11" || mode == "emulate12") {
        // the full report path with the CPU stand-in (an exact-sum instruction): "exact" must fit 1.000
        const Arch arch = mode == "emulate11" ? GFX11 : GFX12;
        const std::vector<Fmt> fmts = arch == GFX11 ? std::vector<Fmt>{F16} : std::vector<Fmt>{F16, BF16, FP8};
        const Runner cpu = [](Arch a, Fmt f, size_t np, Images & im) { emulate(a, f, np, im); };
        text = std::string("CPU stand-in for ") + (arch == GFX11 ? "gfx1100" : "gfx1201");
        ok = probe(arch, fmts, cpu, text);
    } else {
        require_arch();
        const Arch arch = arch_is_gfx11() ? GFX11 : GFX12;
        const std::vector<Fmt> fmts = arch == GFX11 ? std::vector<Fmt>{F16} : std::vector<Fmt>{F16, BF16, FP8};
        const Runner gpu = [](Arch a, Fmt f, size_t np, Images & im) { run_gpu(a, f, np, im); };
        text = arch == GFX11 ? "gfx1100" : "gfx1201";
        ok = probe(arch, fmts, gpu, text);
        if (g_break) {
            text += " (expectation broken on purpose)";
        }
    }
    verdict(mode == "run" ? "wmma_exact" : ("wmma_exact_" + mode).c_str(), ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
