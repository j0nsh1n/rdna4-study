// Leftovers C, wave64 half of MA-2 and CH-6 (measurements/leftovers-c-b.md). Built with -mwavefrontsize64.
//
// Usage: left_mat64 MODE [--break]
//   mask    CH-6 in wave64: every dense and sparse matrix opcode under 7 64-bit EXEC masks, against the two dot
//           controls, and the chained time with a full and a one-lane EXEC (src/study/left_mask.h).
//   hazard  MA-2 in wave64: the forbidden back-to-back read (matrix-arithmetic section 13) for FP16 into FP16 feeding
//           FP16 into FP32 as A and as B, and the 8-bit integer form feeding itself as A and as B, against the same
//           pair with a V_NOP between; 64 waves. Prediction: section 13a's wave32 stale words (operand VGPRs 0 and 1
//           of lanes 0, 1, 8, 9 as A, of lanes 0-15 as B) mapped through the wave64 layout of compiler-and-hazards
//           section 8 (wave32 lane l, VGPR k of an operand of 2 n VGPRs sits in wave64 lane l + 32 (k / n), VGPR
//           k % n). The second instruction alone with exactly those words old is compared with the forbidden result
//           (INFO; the verdict judges the controls: D1 right, V_NOP version equal to the second instruction alone).
//   --break mask: expects the dot instructions to ignore EXEC too; hazard: claims no stale read in wave64.
#include "harness.h"

#define LM_WAVE 64
#include "left_mask.h"

#include <map>
#include <random>
#include <string>
#include <vector>

static bool g_break = false;

static void verdict(const std::string & probe, bool ok, const std::string & text) {
    std::printf("VERDICT %s: %s %s%s\n", probe.c_str(), ok ? "PASS" : "FAIL", text.c_str(),
                g_break ? " (broken on purpose)" : "");
}

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));

// Fixed registers: D1 v[0:3] (INIT = known old contents), first A v[8:9], first B v[16:17], first C v[24:27], the
// second instruction's other operand v[32:33], its C and D v[40:43].
#define H64_N0 ""
#define H64_N1 "v_nop\n"
#define H64_T0(N) "v_wmma_f16_16x16x16_f16 v[0:1], v[8:9], v[16:17], v[24:25]\n" N "v_wmma_f32_16x16x16_f16 v[40:43], v[0:1], v[32:33], v[40:43]"
#define H64_T1(N) "v_wmma_f16_16x16x16_f16 v[0:1], v[8:9], v[16:17], v[24:25]\n" N "v_wmma_f32_16x16x16_f16 v[40:43], v[32:33], v[0:1], v[40:43]"
#define H64_T2(N) "v_wmma_i32_16x16x16_iu8 v[0:3], v8, v16, v[24:27] neg_lo:[1,1,0]\n" N "v_wmma_i32_16x16x16_iu8 v[40:43], v0, v32, v[40:43] neg_lo:[1,1,0]"
#define H64_T3(N) "v_wmma_i32_16x16x16_iu8 v[0:3], v8, v16, v[24:27] neg_lo:[1,1,0]\n" N "v_wmma_i32_16x16x16_iu8 v[40:43], v32, v0, v[40:43] neg_lo:[1,1,0]"
// The second instruction alone (same registers).
#define H64_S0 "v_wmma_f32_16x16x16_f16 v[40:43], v[0:1], v[32:33], v[40:43]"
#define H64_S1 "v_wmma_f32_16x16x16_f16 v[40:43], v[32:33], v[0:1], v[40:43]"
#define H64_S2 "v_wmma_i32_16x16x16_iu8 v[40:43], v0, v32, v[40:43] neg_lo:[1,1,0]"
#define H64_S3 "v_wmma_i32_16x16x16_iu8 v[40:43], v32, v0, v[40:43] neg_lo:[1,1,0]"
enum { H64_NCASES = 4 };
struct H64Case {
    const char * name;
    int nv;  // D1 VGPRs the second instruction reads
    bool i8;
};
static const H64Case h64_cases[H64_NCASES] = {{"f16_f16 D1 as A of f16", 2, false},
                                              {"f16_f16 D1 as B of f16", 2, false},
                                              {"iu8 D1 VGPR 0 as A of iu8", 1, true},
                                              {"iu8 D1 VGPR 0 as B of iu8", 1, true}};

// in: per wave and lane 4 words each of A0, B0, C0, INIT, X1, C1 (24 words). V: 0 nothing between, 1 a V_NOP, 2 the
// second instruction alone with D1 = INIT given as the new value (the caller puts the mixed words there).
template <int CASE, int V>
__global__ __launch_bounds__(64) void h64(const uint32_t * __restrict__ in, uint32_t * __restrict__ out) {
    const size_t w = blockIdx.x, lane = threadIdx.x;
    const uint32_t * p = in + (w * 64 + lane) * 24;
    v4i a0, b0, c0, d1, x1, d2;
    __builtin_memcpy(&a0, p, 16);
    __builtin_memcpy(&b0, p + 4, 16);
    __builtin_memcpy(&c0, p + 8, 16);
    __builtin_memcpy(&d1, p + 12, 16);
    __builtin_memcpy(&x1, p + 16, 16);
    __builtin_memcpy(&d2, p + 20, 16);
#define H64_ASM(T)                                                                                       \
    asm volatile(T : "+{v[0:3]}"(d1), "+{v[40:43]}"(d2)                                                   \
                 : "{v[8:11]}"(a0), "{v[16:19]}"(b0), "{v[24:27]}"(c0), "{v[32:35]}"(x1))
#define H64_SEL(C)                     \
    if constexpr (CASE == C) {         \
        if constexpr (V == 0) {        \
            H64_ASM(H64_T##C(H64_N0)); \
        } else if constexpr (V == 1) { \
            H64_ASM(H64_T##C(H64_N1)); \
        } else {                       \
            H64_ASM(H64_S##C);         \
        }                              \
    }
    H64_SEL(0) H64_SEL(1) H64_SEL(2) H64_SEL(3)
    __builtin_memcpy(out + (w * 64 + lane) * 8, &d1, 16);
    __builtin_memcpy(out + (w * 64 + lane) * 8 + 4, &d2, 16);
}
typedef void (*H64K)(const uint32_t *, uint32_t *);
#define H64T(C) {h64<C, 0>, h64<C, 1>, h64<C, 2>}
static const H64K h64_kern[H64_NCASES][3] = {H64T(0), H64T(1), H64T(2), H64T(3)};

static std::vector<uint32_t> h64_run(int cs, int v, const std::vector<uint32_t> & in) {
    const size_t nw = in.size() / (64 * 24);
    uint32_t *di, *dout;
    CHECK(hipMalloc(&di, in.size() * 4));
    CHECK(hipMalloc(&dout, nw * 64 * 8 * 4));
    CHECK(hipMemcpy(di, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(h64_kern[cs][v], dim3((unsigned) nw), dim3(64), 0, 0, di, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(nw * 64 * 8);
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(di));
    CHECK(hipFree(dout));
    return out;
}

static bool mode_hazard(std::string & text) {
    bool ok = true;
    const int nhz = 64;
    for (int cs = 0; cs < H64_NCASES; ++cs) {
        const H64Case & H = h64_cases[cs];
        std::mt19937_64 rng(6400 + cs);
        std::vector<uint32_t> in((size_t) nhz * 64 * 24);
        for (size_t k = 0; k < in.size(); ++k) {
            const int slot = (int) (k % 24) / 4;
            if (H.i8) {
                in[k] = (uint32_t) rng();
            } else if (slot == 5) {
                in[k] = lm::rnd_word('f', rng);
            } else if (slot == 3) {
                in[k] = 0x3C003C00u ^ ((uint32_t) rng() & 0x01FF01FFu);
            } else {
                in[k] = (uint32_t) (0x3400 + (rng() % 0xC00)) | (uint32_t) (0x3400 + (rng() % 0xC00)) << 16 |
                        (uint32_t) (rng() & 1) << 15 | (uint32_t) (rng() & 1) << 31;
            }
        }
        const std::vector<uint32_t> o0 = h64_run(cs, 0, in), o1 = h64_run(cs, 1, in);
        long d1diff = 0, d2diff = 0;
        for (int w = 0; w < nhz; ++w) {
            for (int l = 0; l < 64; ++l) {
                for (int k = 0; k < 4; ++k) {
                    const size_t x = ((size_t) w * 64 + l) * 8 + k;
                    d1diff += o0[x] != o1[x];
                    d2diff += o0[x + 4] != o1[x + 4];
                }
            }
        }
        // Rows (as A) or columns (as B) of D2 where the forbidden result differs, through the wave64 D layout of
        // compiler-and-hazards section 8: D (r, c) in lane 16 (r >> 3) + c + 32 ((r >> 2) & 1), element r & 3.
        std::map<int, long> lines;
        for (int w = 0; w < nhz; ++w) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    const int l = 16 * (r >> 3) + c + 32 * ((r >> 2) & 1), k = r & 3;
                    const size_t x = ((size_t) w * 64 + l) * 8 + 4 + k;
                    if (o0[x] != o1[x]) {
                        ++lines[cs & 1 ? c : r];
                    }
                }
            }
        }
        std::string ls;
        for (const auto & kv : lines) {
            ls += " " + std::to_string(kv.first) + ":" + std::to_string(kv.second);
        }
        // Prediction: section 13a's wave32 stale words (operand VGPRs 0 and 1; lanes 0, 1, 8, 9 as A, lanes 0-15 as
        // B) mapped to wave64: wave32 lane l, VGPR k of an operand of n32 VGPRs sits in lane l + 32 (k / n64), VGPR
        // k % n64 with n64 = n32 / 2 (A and B of FP16: 4 -> 2; of the 8-bit form: 2 -> 1). --break uses lanes 16-31.
        // Second hypothesis for A, fitted after the first run showed rows 0, 4, 8, 12: the same words in lanes 0, 4,
        // 8, 12 (INFERRED from that run; checked here exactly).
        auto stale_words = [&](bool fitted) {
            std::vector<int> v;
            for (int l = 0; l < 32; ++l) {
                const bool sel = (cs & 1) ? l < 16 : fitted ? (l % 4 == 0 && l < 16) : (l == 0 || l == 1 || l == 8 || l == 9);
                for (int k = 0; k < 2 && sel; ++k) {
                    v.push_back((l + 32 * (k / H.nv)) * H.nv + k % H.nv);
                }
            }
            return v;
        };
        const std::vector<int> pred = stale_words(false), fit = stale_words(true);
        // The second instruction alone: variant 0 all new (must equal the V_NOP version), variant 1 the predicted
        // words old (must equal the forbidden version).
        std::vector<uint32_t> vin((size_t) nhz * 3 * 64 * 24);
        for (int w = 0; w < nhz; ++w) {
            for (int s2 = 0; s2 < 3; ++s2) {
                const size_t vw = (size_t) w * 3 + s2;
                for (int l = 0; l < 64; ++l) {
                    const uint32_t * q = &in[((size_t) w * 64 + l) * 24];
                    uint32_t * r = &vin[(vw * 64 + l) * 24];
                    std::memcpy(r, q, 24 * 4);
                    for (int k = 0; k < 4; ++k) {
                        r[12 + k] = o1[((size_t) w * 64 + l) * 8 + k];  // the new D1
                    }
                }
                if (s2 > 0) {
                    for (int j : s2 == 1 ? pred : fit) {
                        const int l = j / H.nv, k = j % H.nv;
                        vin[(vw * 64 + l) * 24 + 12 + k] = in[((size_t) w * 64 + l) * 24 + 12 + k];  // old word
                    }
                }
            }
        }
        const std::vector<uint32_t> vo = h64_run(cs, 2, vin);
        long nop_bad = 0, pred_bad = 0, fit_bad = 0;
        for (int w = 0; w < nhz; ++w) {
            for (int l = 0; l < 64; ++l) {
                for (int k = 0; k < 4; ++k) {
                    const size_t x = ((size_t) w * 64 + l) * 8 + 4 + k;
                    nop_bad += vo[(((size_t) w * 3) * 64 + l) * 8 + 4 + k] != o1[x];
                    pred_bad += vo[(((size_t) w * 3 + 1) * 64 + l) * 8 + 4 + k] != o0[x];
                    fit_bad += vo[(((size_t) w * 3 + 2) * 64 + l) * 8 + 4 + k] != o0[x];
                }
            }
        }
        auto words = [&](const std::vector<int> & v) {
            std::string t;
            for (int j : v) {
                t += " l" + std::to_string(j / H.nv) + "v" + std::to_string(j % H.nv);
            }
            return t;
        };
        std::printf("RESULT hazard64 %-28s waves %d D2 words differing from the V_NOP version %ld of %d, D1 differs %ld, "
                    "V_NOP version equal to the second instruction alone: %ld differ | %s where it differs (index:count):%s\n",
                    H.name, nhz, d2diff, nhz * 256, d1diff, nop_bad, cs & 1 ? "columns" : "rows", ls.c_str());
        std::printf("INFO hazard64 %-28s wave32 rule mapped to wave64 (stale words%s): %ld words differ from the forbidden "
                    "result -> %s\n",
                    H.name, words(pred).c_str(), pred_bad, pred_bad == 0 ? "exact" : "does not hold");
        if (!(cs & 1)) {
            std::printf("INFO hazard64 %-28s fitted: lanes 0, 4, 8, 12 (stale words%s): %ld words differ -> %s\n", H.name,
                        words(fit).c_str(), fit_bad, fit_bad == 0 ? "exact" : "does not hold");
        }
        // The verdict judges the controls: D1 written right, the V_NOP version equal to the second instruction alone,
        // and a stale read present (--break claims none in wave64). Which words are stale is INFO.
        ok = ok && d1diff == 0 && nop_bad == 0 && (g_break ? d2diff == 0 : d2diff > 0);
        char b[200];
        std::snprintf(b, sizeof b, "%s: %ld of %d words differ from the V_NOP version", H.name, d2diff, nhz * 256);
        text += (text.empty() ? "" : "; ") + std::string(b);
    }
    return ok;
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--break") {
            g_break = true;
        }
    }
    require_arch();
    std::string text;
    bool ok;
    if (mode == "mask") {
        ok = lm::mask_test(g_break, text);
    } else if (mode == "hazard") {
        ok = mode_hazard(text);
    } else {
        std::fprintf(stderr, "usage: left_mat64 mask|hazard [--break]\n");
        return 2;
    }
    verdict("left_mat64_" + mode, ok, text);
    return ok ? 0 : 1;
}
