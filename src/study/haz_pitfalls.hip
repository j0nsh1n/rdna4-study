// Lane 8, item 4 (a, b, f): the RDNA 3 compiler pitfalls retested on gfx12.
//  (a) a 32-bit bit trick (E4M3 bytes to FP16 halves by shifts and masks) whose result feeds an FP16 matrix operand:
//      built per half, by __builtin_bit_cast of whole words, and per half with an empty asm barrier on the words.
//  (b) copies from 32-bit LDS words into an FP16 vector: memcpy through a word array, memcpy from LDS directly,
//      __builtin_bit_cast of a 4-word vector load.
//  (f) v_permlane16 and v_permlanex16 inside a per-lane branch, fetch-inactive bit on and off, five mask shapes:
//      what an executing lane receives when its partner lane is switched off.
// The instruction counts of each loop are checked by scripts/haz-expect/haz_pitfalls.txt; the card results here.
// Usage: haz_pitfalls [all|break]. "break" checks (a) against a reference with one mask changed; it must fail.
#include "haz_common.h"

using namespace haz;

typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef uint32_t v2u __attribute__((ext_vector_type(2)));

#define WMMA32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

__host__ __device__ inline uint32_t widen_even(uint32_t x) { return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3F803F80u); }
__host__ __device__ inline uint32_t widen_odd(uint32_t x) { return (x & 0x80008000u) | ((x >> 1) & 0x3F803F80u); }

__device__ inline _Float16 half_of(uint32_t w, int hi) { return __builtin_bit_cast(_Float16, (uint16_t) (hi ? w >> 16 : w)); }

// V = 0: halves set one by one; 1: bit_cast of the 4 words; 2: as 0 with an empty asm keeping each word whole.
template <int V>
__global__ void w_widen(const v2u * W, const v8h * B, v8f * D, int n) {
    const int l = threadIdx.x;
    const v8h b = B[l];
    v8f acc = {};
    for (int i = 0; i < n; ++i) {
        const v2u w = W[i * 32 + l];
        uint32_t e0 = widen_even(w.x), o0 = widen_odd(w.x), e1 = widen_even(w.y), o1 = widen_odd(w.y);
        v8h a;
        if (V == 1) {
            const v4u words = {e0, o0, e1, o1};
            a = __builtin_bit_cast(v8h, words);
        } else {
            if (V == 2) {
                asm("" : "+v"(e0), "+v"(o0), "+v"(e1), "+v"(o1));
            }
            a[0] = half_of(e0, 0);
            a[1] = half_of(e0, 1);
            a[2] = half_of(o0, 0);
            a[3] = half_of(o0, 1);
            a[4] = half_of(e1, 0);
            a[5] = half_of(e1, 1);
            a[6] = half_of(o1, 0);
            a[7] = half_of(o1, 1);
        }
        acc = WMMA32(a, b, acc);
    }
    D[l] = acc;
}

constexpr int LDS_FRAGS = 8;

// V = 0: memcpy into the vector from a 4-word array read element by element; 1: memcpy straight from LDS;
// 2: bit_cast of a 4-word vector load.
template <int V>
__global__ void w_lds(const uint32_t * W, const v8h * B, v8f * D, int n) {
    __shared__ uint32_t lds[LDS_FRAGS * 32 * 4];
    const int l = threadIdx.x;
    for (int i = l; i < LDS_FRAGS * 32 * 4; i += 32) {
        lds[i] = W[i];
    }
    __syncthreads();
    const v8h b = B[l];
    v8f acc = {};
    for (int i = 0; i < n; ++i) {
        const int base = ((i % LDS_FRAGS) * 32 + l) * 4;
        v8h a;
        if (V == 0) {
            uint32_t w[4];
            for (int j = 0; j < 4; ++j) {
                w[j] = lds[base + j];
            }
            __builtin_memcpy(&a, w, 16);
        } else if (V == 1) {
            __builtin_memcpy(&a, &lds[base], 16);
        } else {
            a = __builtin_bit_cast(v8h, *(const v4u *) &lds[base]);
        }
        acc = WMMA32(a, b, acc);
    }
    D[l] = acc;
}

// Lane permutes inside a per-lane branch. Every lane computes v before the branch. X16 = 0: v_permlane16 with
// selects that reverse each row of 16; X16 = 1: v_permlanex16 with identity selects (lane l reads lane l ^ 16).
// PIN = 1: an empty volatile asm on v before the branch keeps its computation in front of the branch (all lanes).
template <int X16, int FI, int PIN = 0>
__global__ void w_perm(const uint32_t * X, uint32_t * Y, int shape) {
    const int l = threadIdx.x;
    uint32_t v = X[l] * 3u + 1u;
    if (PIN) {
        asm volatile("" : "+v"(v));
    }
    const uint32_t key = X[32 + l];
    bool on;
    switch (shape) {
    case 0: on = l < 16; break;
    case 1: on = l >= 16; break;
    case 2: on = l & 1; break;
    case 3: on = (l & 3) == 0; break;
    default: on = key & 1; break;  // data-dependent, chosen on the host
    }
    uint32_t y = 0xDEAD0000u + l;
    if (on) {
        if (X16) {
            y = __builtin_amdgcn_permlanex16(y, v, 0x76543210u, 0xFEDCBA98u, FI, false);
        } else {
            y = __builtin_amdgcn_permlane16(y, v, 0x89ABCDEFu, 0x01234567u, FI, false);
        }
    }
    Y[l] = y;
}

namespace {

template <class T>
T * dalloc(size_t n) {
    T * p;
    CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

// D after n chained instructions; A of iteration i from the per-lane half images a[i].
std::vector<uint32_t> chain_ref(const std::vector<std::vector<uint16_t>> & a, const std::vector<uint16_t> & b) {
    Problem p;
    p.b = b;
    std::fill(p.c.begin(), p.c.end(), 0u);
    for (const auto & ai : a) {
        p.a = ai;
        p.c = reference(p);
    }
    return p.c;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    require_arch();
    Verdicts v;
    const int n = 6;
    const Problem pb = make_problem(9);
    v8h * B = (v8h *) dev_copy(pb.b);
    v8f * D = dalloc<v8f>(32);

    // (a) E4M3 words, NaN codes avoided (they widen to +-480 * 2^-8, a separate question).
    std::vector<uint32_t> words(n * 32 * 2);
    uint32_t s = 77;
    for (auto & w : words) {
        uint32_t x = 0;
        for (int k = 0; k < 4; ++k) {
            s = s * 1664525u + 1013904223u;
            uint32_t byte = (s >> 24) & 0xFF;
            if ((byte & 0x7F) == 0x7F) {
                byte ^= 1;
            }
            x |= byte << (8 * k);
        }
        w = x;
    }
    {
        std::vector<std::vector<uint16_t>> a(n, std::vector<uint16_t>(256));
        for (int i = 0; i < n; ++i) {
            for (int l = 0; l < 32; ++l) {
                const uint32_t x0 = words[(i * 32 + l) * 2], x1 = words[(i * 32 + l) * 2 + 1];
                uint32_t h[4] = {widen_even(x0), widen_odd(x0), widen_even(x1), widen_odd(x1)};
                if (mode == "break") {
                    h[1] = (x0 & 0x80008000u) | ((x0 >> 1) & 0x7F807F80u);  // the wrong odd mask from the RDNA 3 test
                }
                for (int k = 0; k < 4; ++k) {
                    a[i][l * 8 + 2 * k] = (uint16_t) h[k];
                    a[i][l * 8 + 2 * k + 1] = (uint16_t) (h[k] >> 16);
                }
            }
        }
        const auto ref = chain_ref(a, pb.b);
        v2u * dW = (v2u *) dev_copy(words);
        void (*ks[3])(const v2u *, const v8h *, v8f *, int) = {w_widen<0>, w_widen<1>, w_widen<2>};
        const char * names[3] = {"per_half", "bit_cast", "per_half_asm"};
        for (int k = 0; k < 3; ++k) {
            hipLaunchKernelGGL(ks[k], dim3(1), dim3(32), 0, 0, dW, B, D, n);
            CHECK(hipDeviceSynchronize());
            const auto d = host_copy((const uint32_t *) D, 256);
            int w = 0;
            for (int i = 0; i < 256; ++i) {
                w += d[i] != ref[i];
            }
            v.put(mode == "break" ? "pit_widen_break" : "pit_widen", w == 0,
                  fmt("%s: %d of 256 differ from the model (%d chained instructions)", names[k], w, n));
        }
        if (mode == "break") {
            return v.exit_code();
        }
    }
    // (b) LDS words: halves of small integers.
    {
        std::vector<uint16_t> h(LDS_FRAGS * 32 * 8);
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = h16((float) ((int) (i * 13 % 7) - 3));
        }
        std::vector<std::vector<uint16_t>> a(n, std::vector<uint16_t>(256));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 256; ++j) {
                a[i][j] = h[(i % LDS_FRAGS) * 256 + j];
            }
        }
        const auto ref = chain_ref(a, pb.b);
        uint32_t * dW = (uint32_t *) dev_copy(h);
        void (*ks[3])(const uint32_t *, const v8h *, v8f *, int) = {w_lds<0>, w_lds<1>, w_lds<2>};
        const char * names[3] = {"memcpy_words", "memcpy_lds", "bit_cast"};
        for (int k = 0; k < 3; ++k) {
            hipLaunchKernelGGL(ks[k], dim3(1), dim3(32), 0, 0, dW, B, D, n);
            CHECK(hipDeviceSynchronize());
            const auto d = host_copy((const uint32_t *) D, 256);
            int w = 0;
            for (int i = 0; i < 256; ++i) {
                w += d[i] != ref[i];
            }
            v.put("pit_lds", w == 0, fmt("%s: %d of 256 differ from the model", names[k], w));
        }
    }
    // (f) lane permutes. Classify what each executing lane got: the partner's v, its own old value, zero, other.
    {
        std::vector<uint32_t> X(64);
        for (int l = 0; l < 32; ++l) {
            X[l] = 1000 + l;
            X[32 + l] = (0x2C5A93E1u >> l) & 1;
        }
        uint32_t * dX = dev_copy(X);
        uint32_t * dY = dalloc<uint32_t>(32);
        void (*ks[6])(const uint32_t *, uint32_t *, int) = {w_perm<0, 0>, w_perm<0, 1>, w_perm<1, 0>,
                                                            w_perm<1, 1>, w_perm<0, 1, 1>, w_perm<1, 1, 1>};
        const char * names[6] = {"permlane16 fi=0", "permlane16 fi=1", "permlanex16 fi=0", "permlanex16 fi=1",
                                 "permlane16 fi=1 pinned", "permlanex16 fi=1 pinned"};
        const char * shapes[5] = {"lanes 0-15", "lanes 16-31", "odd lanes", "every 4th", "data-dependent"};
        for (int k = 0; k < 6; ++k) {
            for (int sh = 0; sh < 5; ++sh) {
                hipLaunchKernelGGL(ks[k], dim3(1), dim3(32), 0, 0, dX, dY, sh);
                CHECK(hipDeviceSynchronize());
                const auto y = host_copy(dY, 32);
                int on_n = 0, partner_on = 0, partner_off = 0, got_partner_on = 0, got_partner_off = 0,
                    off_old = 0, off_zero = 0, off_other = 0, off_untouched = 0, off_lanes = 0, off_partner_old = 0;
                for (int l = 0; l < 32; ++l) {
                    auto is_on = [&](int q) {
                        switch (sh) {
                        case 0: return q < 16;
                        case 1: return q >= 16;
                        case 2: return (q & 1) == 1;
                        case 3: return (q & 3) == 0;
                        default: return (X[32 + q] & 1) == 1;
                        }
                    };
                    if (!is_on(l)) {
                        ++off_lanes;
                        off_untouched += y[l] == 0xDEAD0000u + l;
                        continue;
                    }
                    ++on_n;
                    const int partner = (k < 2 || k == 4) ? (l & 16) + 15 - (l & 15) : l ^ 16;
                    const uint32_t want = X[partner] * 3u + 1u;
                    if (is_on(partner)) {
                        ++partner_on;
                        got_partner_on += y[l] == want;
                    } else {
                        ++partner_off;
                        got_partner_off += y[l] == want;
                        off_old += y[l] == 0xDEAD0000u + l;
                        off_zero += y[l] == 0;
                        off_partner_old += y[l] == 0xDEAD0000u + partner;
                        off_other += y[l] != want && y[l] != 0xDEAD0000u + l && y[l] != 0;
                    }
                }
                // Required either way: partners that are on are read; lanes that are off are untouched.
                v.put("pit_perm", got_partner_on == partner_on && off_untouched == off_lanes,
                      fmt("%s, %s: %d lanes on; partner on: %d of %d right; partner off: %d lanes got the partner's "
                          "value, %d kept their old value, %d got 0, %d other (of which %d the partner's old y); off "
                          "lanes untouched %d of %d",
                          names[k], shapes[sh], on_n, got_partner_on, partner_on, got_partner_off, off_old, off_zero,
                          off_other, off_partner_old, off_untouched, off_lanes));
                if (k >= 4 && partner_off > 0) {
                    v.put("pit_perm_pinned", got_partner_off == partner_off,
                          fmt("%s, %s: %d of %d lanes with a switched-off partner got its value", names[k], shapes[sh],
                              got_partner_off, partner_off));
                }
            }
        }
    }
    return v.exit_code();
}
