// FP8 (E4M3) hardware conversions on gfx1201, over all inputs, against the integer reference in e4m3_ref.h (whose
// own test is cvt_ref_check).
//
// decode1: __builtin_amdgcn_cvt_f32_fp8(word, sel), all 256 codes in each of the 4 byte positions, for each of the
//          4 selects. Reports which byte each select really reads (AMD's guide text for the byte select looks
//          reversed for selects 1 and 2).
// decode2: __builtin_amdgcn_cvt_pk_f32_fp8(word, word_sel), all 65536 byte pairs in each 16-bit half.
// encode:  __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, word_sel) over all 2^32 FP32 bit patterns (a = x, b = x with
//          its bits flipped by a constant, so both slots see every pattern), in three settings:
//          plain  - the MODE register as the kernel is launched; reference: OCP non-saturating (NaN for
//                   infinity and anything rounding above 448);
//          clamp  - x clamped to +-448 first (median of three, NaN passed through), as Mesa and HIP's header do;
//                   reference: saturating;
//          ovfl   - MODE.FP16_OVFL (bit 23) set, no clamp; reference: saturating, which is what AMD's guide says
//                   the bit does. Mesa's authors found infinity is not clamped this way; this shows what happens.
//          Mismatches are counted by class: NaN, infinity, above 448, ties, subnormal range, zero, other.
//          The untouched half of `old` must survive.
// Usage: cvt_fp8 [decode|encode|all]. Exits 1 if a decode check fails or the reference is contradicted outside the
// classes where the guide and Mesa disagree (infinity and above-448 in plain and ovfl settings are reported only).
#include "e4m3_ref.h"
#include "harness.h"

#include <cmath>
#include <string>

typedef float v2f __attribute__((ext_vector_type(2)));

static constexpr uint32_t OLD = 0x5A5A5A5Au;
static constexpr uint32_t FLIP = 0x9E3779B9u;

__host__ __device__ inline uint32_t fill_word(uint32_t c, int p) {
    uint32_t w = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t b = j == p ? c : (c * 7u + 0x35u * (j + 1)) & 0xFFu;
        w |= b << (8 * j);
    }
    return w;
}

__global__ void decode1(float * out) {
#if defined(__GFX12__)
    const int c = threadIdx.x, p = blockIdx.x;
    const int w = (int) fill_word(c, p);
    out[(0 * 4 + p) * 256 + c] = __builtin_amdgcn_cvt_f32_fp8(w, 0);
    out[(1 * 4 + p) * 256 + c] = __builtin_amdgcn_cvt_f32_fp8(w, 1);
    out[(2 * 4 + p) * 256 + c] = __builtin_amdgcn_cvt_f32_fp8(w, 2);
    out[(3 * 4 + p) * 256 + c] = __builtin_amdgcn_cvt_f32_fp8(w, 3);
#endif
}

__global__ void decode2(v2f * out) {
#if defined(__GFX12__)
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;  // 0..65535
    const uint32_t other = (pair * 2654435761u) >> 16;
    out[pair] = __builtin_amdgcn_cvt_pk_f32_fp8((int) (pair | (other << 16)), false);
    out[65536 + pair] = __builtin_amdgcn_cvt_pk_f32_fp8((int) (other | (pair << 16)), true);
#endif
}

enum Cls { C_NAN, C_INF, C_ABOVE, C_TIE, C_SUB, C_ZERO, C_OTHER, NCLS };
static const char * cls_name[NCLS] = {"nan", "inf", "above448", "tie", "subnormal", "zero", "other"};

__host__ __device__ inline int classify(uint32_t x) {
    const uint32_t a = x & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return C_NAN;
    }
    if (a == 0x7F800000u) {
        return C_INF;
    }
    if (a > 0x43E00000u) {  // 448
        return C_ABOVE;
    }
    if (a == 0) {
        return C_ZERO;
    }
    if (a < 0x3C800000u) {  // 2^-6, the smallest normal E4M3 value
        // a tie in the subnormal range is m24 * 2^(exp-14) with exactly half a step left over
        const int exp = (int) (a >> 23) - 127;
        const int s = 14 - exp;
        const uint32_t m24 = (a & 0x7FFFFFu) | 0x800000u;
        if (a >= 0x00800000u && s < 32 && (m24 & ((1u << s) - 1)) == (1u << (s - 1))) {
            return C_TIE;
        }
        return C_SUB;
    }
    return (a & 0xFFFFFu) == 0x80000u ? C_TIE : C_OTHER;  // half of one E4M3 step at this exponent
}

struct EncStats {
    unsigned long long mism[3][NCLS];
    unsigned long long count[NCLS];
    unsigned long long kept_bad;  // the half of `old` that must be preserved was changed
    unsigned int nex[3];
    uint32_t ex[3][8][3];  // per setting: input, hardware byte, reference byte
    uint32_t mode_launch, mode_ovfl;
};

template <int SETTING>
__global__ void encode(EncStats * st, uint64_t start, uint64_t count) {
#if defined(__GFX12__)
    if constexpr (SETTING == 2) {
        __builtin_amdgcn_s_setreg(1 | (23 << 6), 1);  // hwreg(HW_REG_MODE, 23, 1) = FP16_OVFL
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t mode = __builtin_amdgcn_s_getreg(1 | (31 << 11));  // hwreg(HW_REG_MODE, 0, 32)
        if constexpr (SETTING == 2) {
            st->mode_ovfl = mode;
        } else {
            st->mode_launch = mode;
        }
    }
    unsigned long long local[NCLS] = {};
    unsigned long long kept_bad = 0;
    for (uint64_t i = start + blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < start + count;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t xa = (uint32_t) i, xb = xa ^ FLIP;
        float fa = __uint_as_float(xa), fb = __uint_as_float(xb);
        if constexpr (SETTING == 1) {
            fa = isnan(fa) ? fa : __builtin_amdgcn_fmed3f(fa, -448.0f, 448.0f);
            fb = isnan(fb) ? fb : __builtin_amdgcn_fmed3f(fb, -448.0f, 448.0f);
        }
        const bool sel = (xa >> 7) & 1;  // both word selects, spread over all inputs
        const uint32_t w = sel ? (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(fa, fb, (int) OLD, true)
                               : (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(fa, fb, (int) OLD, false);
        const uint32_t ha = sel ? (w >> 16) & 0xFF : w & 0xFF, hb = sel ? w >> 24 : (w >> 8) & 0xFF;
        kept_bad += sel ? (w & 0xFFFF) != (OLD & 0xFFFF) : (w >> 16) != (OLD >> 16);
        const bool sat = SETTING != 0;
        const uint32_t ra = e4m3_encode(xa, sat), rb = e4m3_encode(xb, sat);
        if (ha != ra) {
            ++local[classify(xa)];
            const unsigned k = atomicAdd(&st->nex[SETTING], 1u);
            if (k < 8) {
                st->ex[SETTING][k][0] = xa;
                st->ex[SETTING][k][1] = ha;
                st->ex[SETTING][k][2] = ra;
            }
        }
        if (hb != rb) {
            ++local[classify(xb)];
        }
    }
    for (int c = 0; c < NCLS; ++c) {
        if (local[c]) {
            atomicAdd(&st->mism[SETTING][c], local[c]);
        }
    }
    if (kept_bad) {
        atomicAdd(&st->kept_bad, kept_bad);
    }
#endif
}

static bool same_float(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return (std::isnan(a) && std::isnan(b)) || x == y;
}

static bool run_decode(std::string & text) {
    float * d1;
    v2f * d2;
    CHECK(hipMalloc(&d1, 16 * 256 * 4));
    CHECK(hipMalloc(&d2, 2 * 65536 * sizeof(v2f)));
    decode1<<<4, 256>>>(d1);
    decode2<<<256, 256>>>(d2);
    CHECK(hipDeviceSynchronize());
    std::vector<float> h1(16 * 256);
    std::vector<v2f> h2(2 * 65536);
    CHECK(hipMemcpy(h1.data(), d1, h1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h2.data(), d2, h2.size() * sizeof(v2f), hipMemcpyDeviceToHost));
    bool ok = true;
    char buf[160];
    text += "select->byte:";
    bool identity = true;
    for (int s = 0; s < 4; ++s) {
        int found = -1, nfound = 0;
        for (int p = 0; p < 4; ++p) {
            int bad = 0;
            for (int c = 0; c < 256; ++c) {
                const float want = e4m3_decode((uint8_t) ((fill_word(c, p) >> (8 * p)) & 0xFF));
                bad += !same_float(h1[(s * 4 + p) * 256 + c], want);
            }
            std::printf("RESULT decode1 select %d byte %d mismatches %d of 256\n", s, p, bad);
            if (bad == 0) {
                found = p;
                ++nfound;
            }
        }
        std::snprintf(buf, sizeof buf, " %d->%s", s, nfound == 1 ? std::to_string(found).c_str() : "?");
        text += buf;
        ok = ok && nfound == 1;
        identity = identity && found == s;
    }
    text += identity ? " (as numbered)" : " (NOT as numbered)";
    long bad2[2][2] = {};  // [word_sel][order]: order 0 = x from the low byte
    for (int ws = 0; ws < 2; ++ws) {
        for (uint32_t pair = 0; pair < 65536; ++pair) {
            const v2f g = h2[ws * 65536 + pair];
            const float lo = e4m3_decode(pair & 0xFF), hi = e4m3_decode(pair >> 8);
            bad2[ws][0] += !(same_float(g.x, lo) && same_float(g.y, hi));
            bad2[ws][1] += !(same_float(g.x, hi) && same_float(g.y, lo));
        }
        std::printf("RESULT decode2 word_sel %d mismatches low-byte-first %ld high-byte-first %ld of 65536\n", ws,
                    bad2[ws][0], bad2[ws][1]);
    }
    const bool pk_ok = bad2[0][0] == 0 && bad2[1][0] == 0;
    std::snprintf(buf, sizeof buf, "; packed decode %s", pk_ok ? "exact, low byte first" : "MISMATCH");
    text += buf;
    CHECK(hipFree(d1));
    CHECK(hipFree(d2));
    return ok && pk_ok;
}

static bool run_encode(std::string & text) {
    EncStats * dst;
    CHECK(hipMalloc(&dst, sizeof(EncStats)));
    CHECK(hipMemset(dst, 0, sizeof(EncStats)));
    const uint64_t total = 1ull << 32, chunk = 1ull << 30;
    for (uint64_t s = 0; s < total; s += chunk) {
        encode<0><<<4096, 256>>>(dst, s, chunk);
        encode<1><<<4096, 256>>>(dst, s, chunk);
        encode<2><<<4096, 256>>>(dst, s, chunk);
        CHECK(hipDeviceSynchronize());
    }
    EncStats st;
    CHECK(hipMemcpy(&st, dst, sizeof st, hipMemcpyDeviceToHost));
    CHECK(hipFree(dst));
    std::printf("MODE register as launched 0x%08x, after setting FP16_OVFL 0x%08x\n", st.mode_launch, st.mode_ovfl);
    const char * names[3] = {"plain", "clamp", "ovfl"};
    bool ok = st.kept_bad == 0;
    char buf[200];
    for (int k = 0; k < 3; ++k) {
        std::printf("RESULT encode %-5s mismatches (both slots, 2^33 conversions):", names[k]);
        unsigned long long tot = 0;
        for (int c = 0; c < NCLS; ++c) {
            std::printf(" %s %llu", cls_name[c], st.mism[k][c]);
            tot += st.mism[k][c];
        }
        std::printf("\n");
        for (int e = 0; e < 8; ++e) {
            if (st.ex[k][e][0] | st.ex[k][e][1] | st.ex[k][e][2]) {
                std::printf("  example %s x=0x%08x (%g) hardware 0x%02x reference 0x%02x\n", names[k], st.ex[k][e][0],
                            (double) __builtin_bit_cast(float, st.ex[k][e][0]), st.ex[k][e][1], st.ex[k][e][2]);
            }
        }
        // infinity and above-448 are where the guide, Mesa and the OCP rule may disagree: reported, not judged,
        // except after the explicit clamp, where every class must match.
        unsigned long long judged = tot;
        if (k != 1) {
            judged -= st.mism[k][C_INF] + st.mism[k][C_ABOVE];
        }
        std::snprintf(buf, sizeof buf, "; %s: %llu mismatches (inf %llu, above448 %llu)", names[k], tot,
                      st.mism[k][C_INF], st.mism[k][C_ABOVE]);
        text += buf;
        ok = ok && judged == 0;
    }
    if (st.kept_bad) {
        std::snprintf(buf, sizeof buf, "; other half of old changed %llu times", st.kept_bad);
        text += buf;
    }
    return ok;
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string what = argc > 1 ? argv[1] : "all";
    std::string text;
    bool ok = true;
    if (what == "decode" || what == "all") {
        ok = run_decode(text) && ok;
    }
    if (what == "encode" || what == "all") {
        ok = run_encode(text) && ok;
    }
    verdict("cvt_fp8", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
