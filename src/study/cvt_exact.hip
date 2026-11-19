// Lane 6, items 1 and 2: FP8 and BF8 conversions on gfx1201 over all inputs, judged in place by the integer
// reference in cvt_ref.h (tested on the CPU by cvt_refcheck).
//
// cvt_exact decode
//   v_cvt_f32_fp8 / v_cvt_f32_bf8 (all 256 codes at each of the 4 byte positions, each of the 4 byte selects; the
//   other bytes hold unrelated codes) and v_cvt_pk_f32_fp8 / _bf8 (all 65536 byte pairs in each 16-bit half, both
//   word selects). Reports which byte each select reads and the NaN bit patterns the hardware writes.
// cvt_exact encode <fp8|bf8> <setting> [--break]
//   v_cvt_pk_fp8_f32 / v_cvt_pk_bf8_f32 over all 2^32 FP32 inputs in the first slot and (input XOR a constant)
//   in the second, both word selects, under one MODE setting (cvt_gpu.h: plain, ovfl, rup, rdn, rtz, noden,
//   ovfl_rtz) with the vector-ALU control and MODE restored at the end. Reference: round to nearest even,
//   saturating when FP16_OVFL is set; either NaN code is accepted and counted separately. Counts every input class
//   (cvt_ref.h class8) and prints a histogram of the codes the hardware wrote per class (first slot).
//   Under FP16_OVFL the reference saturates finite values only: infinity is not clamped (first hour; cvt_ref.h).
//   --break: the reference rounds ties away from zero; the verdict must be FAIL.
#include "cvt_gpu.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

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

// out: [fmt][select][position][code]
__global__ void decode1(float * out) {
    const int c = threadIdx.x, p = blockIdx.x;
    const int w = (int) fill_word(c, p);
    float * o = out + p * 256 + c;
    o[0 * 1024] = __builtin_amdgcn_cvt_f32_fp8(w, 0);
    o[1 * 1024] = __builtin_amdgcn_cvt_f32_fp8(w, 1);
    o[2 * 1024] = __builtin_amdgcn_cvt_f32_fp8(w, 2);
    o[3 * 1024] = __builtin_amdgcn_cvt_f32_fp8(w, 3);
    o[4 * 1024] = __builtin_amdgcn_cvt_f32_bf8(w, 0);
    o[5 * 1024] = __builtin_amdgcn_cvt_f32_bf8(w, 1);
    o[6 * 1024] = __builtin_amdgcn_cvt_f32_bf8(w, 2);
    o[7 * 1024] = __builtin_amdgcn_cvt_f32_bf8(w, 3);
}

// out: [fmt][word_sel][pair]
__global__ void decode2(v2f * out) {
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t other = (pair * 2654435761u) >> 16;
    const int lo = (int) (pair | (other << 16)), hi = (int) (other | (pair << 16));
    out[pair] = __builtin_amdgcn_cvt_pk_f32_fp8(lo, false);
    out[65536 + pair] = __builtin_amdgcn_cvt_pk_f32_fp8(hi, true);
    out[2 * 65536 + pair] = __builtin_amdgcn_cvt_pk_f32_bf8(lo, false);
    out[3 * 65536 + pair] = __builtin_amdgcn_cvt_pk_f32_bf8(hi, true);
}

struct EncStats {
    unsigned long long count[cr::NCLS];
    unsigned long long mism[2][cr::NCLS];  // per slot, NaN-code differences not counted
    unsigned long long nanalt[cr::NCLS];   // both NaN, different code (accepted)
    unsigned long long mism_dir[cr::NCLS];  // first slot against the directed-rounding reference
    unsigned long long kept_bad;
    unsigned long long hist[cr::NCLS][256];
    unsigned int nex;
    uint32_t ex[16][4];
    uint32_t ctrl[cvt::NCTRL];
    uint32_t mode_launch;
};

template <int F>
__global__ __launch_bounds__(256) void encode(EncStats * st, uint64_t start, uint64_t count, int write, uint32_t mode,
                                              int sat, int dir_rnd, int ties_away) {
    __shared__ unsigned int lh[cr::NCLS * 256];
    for (int i = threadIdx.x; i < cr::NCLS * 256; i += blockDim.x) {
        lh[i] = 0;
    }
    __syncthreads();
    const uint32_t launch = cvt::mode_read();
    uint32_t old = 0;
    if (write) {
        old = cvt::mode_set(mode);
    }
    if (blockIdx.x == 0) {
        uint32_t c[cvt::NCTRL];
        cvt::control(c);
        if (threadIdx.x == 0) {
            for (int i = 0; i < cvt::NCTRL; ++i) {
                st->ctrl[i] = c[i];
            }
            st->mode_launch = launch;
        }
    }
    unsigned long long mism[2][cr::NCLS] = {}, nanalt = 0, mdir = 0, kept_bad = 0;
    for (uint64_t i = start + blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < start + count;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t xa = (uint32_t) i, xb = xa ^ FLIP;
        const bool sel = (xa >> 7) & 1;
        uint32_t w = OLD;
        if (F == cr::FP8) {
            if (sel) {
                asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2 op_sel:[0,0,1]" : "+v"(w) : "v"(xa), "v"(xb));
            } else {
                asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "+v"(w) : "v"(xa), "v"(xb));
            }
        } else {
            if (sel) {
                asm volatile("v_cvt_pk_bf8_f32 %0, %1, %2 op_sel:[0,0,1]" : "+v"(w) : "v"(xa), "v"(xb));
            } else {
                asm volatile("v_cvt_pk_bf8_f32 %0, %1, %2" : "+v"(w) : "v"(xa), "v"(xb));
            }
        }
        const uint32_t ha = sel ? (w >> 16) & 0xFF : w & 0xFF, hb = sel ? w >> 24 : (w >> 8) & 0xFF;
        kept_bad += sel ? (w & 0xFFFF) != (OLD & 0xFFFF) : (w >> 16) != (OLD >> 16);
        const uint32_t ra = cr::encode8(F, xa, sat, cr::RNE, ties_away, false),
                       rb = cr::encode8(F, xb, sat, cr::RNE, ties_away, false);
        const int ca = cr::class8(F, xa), cb = cr::class8(F, xb);
        atomicAdd(&lh[ca * 256 + ha], 1u);
        if (ha != ra) {
            if (cr::is_nan8(F, ha) && cr::is_nan8(F, ra)) {
                ++nanalt;
            } else {
                ++mism[0][ca];
                const unsigned k = atomicAdd(&st->nex, 1u);
                if (k < 16) {
                    st->ex[k][0] = xa;
                    st->ex[k][1] = ha;
                    st->ex[k][2] = ra;
                    st->ex[k][3] = ca;
                }
            }
        }
        if (hb != rb && !(cr::is_nan8(F, hb) && cr::is_nan8(F, rb))) {
            ++mism[1][cb];
        }
        if (dir_rnd >= 0 && ha != cr::encode8(F, xa, sat, dir_rnd, false, false)) {
            ++mdir;
        }
    }
    if (write) {
        cvt::mode_restore(old);
    }
    for (int s = 0; s < 2; ++s) {
        for (int c = 0; c < cr::NCLS; ++c) {
            if (mism[s][c]) {
                atomicAdd(&st->mism[s][c], mism[s][c]);
            }
        }
    }
    if (nanalt) {
        atomicAdd(&st->nanalt[cr::C_NAN], nanalt);
    }
    if (mdir) {
        atomicAdd(&st->mism_dir[0], mdir);
    }
    if (kept_bad) {
        atomicAdd(&st->kept_bad, kept_bad);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cr::NCLS * 256; i += blockDim.x) {
        if (lh[i]) {
            atomicAdd(&st->hist[i / 256][i % 256], (unsigned long long) lh[i]);
        }
    }
}

static bool same_float(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return (std::isnan(a) && std::isnan(b)) || x == y;
}
static uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static float float_of(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static int run_decode() {
    float * d1;
    v2f * d2;
    CHECK(hipMalloc(&d1, 8 * 1024 * 4));
    CHECK(hipMalloc(&d2, 4 * 65536 * sizeof(v2f)));
    decode1<<<4, 256>>>(d1);
    decode2<<<256, 256>>>(d2);
    CHECK(hipDeviceSynchronize());
    std::vector<float> h1(8 * 1024);
    std::vector<v2f> h2(4 * 65536);
    CHECK(hipMemcpy(h1.data(), d1, h1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(h2.data(), d2, h2.size() * sizeof(v2f), hipMemcpyDeviceToHost));
    CHECK(hipFree(d1));
    CHECK(hipFree(d2));
    bool ok = true;
    std::string text;
    char buf[200];
    for (int f = 0; f < 2; ++f) {
        const char * fn = f ? "bf8" : "fp8";
        std::map<uint32_t, int> nan_bits;
        std::string sel_map;
        for (int s = 0; s < 4; ++s) {
            int found = -1, nfound = 0;
            for (int p = 0; p < 4; ++p) {
                int bad = 0;
                for (int c = 0; c < 256; ++c) {
                    const float got = h1[(f * 4 + s) * 1024 + p * 256 + c];
                    const float want = float_of(cr::decode8(f, c));
                    bad += !same_float(got, want);
                    if (p == s && std::isnan(got)) {
                        ++nan_bits[bits_of(got)];
                    }
                }
                if (bad == 0) {
                    found = p;
                    ++nfound;
                }
                if (p == s) {
                    std::printf("RESULT decode %s select %d byte %d: %d mismatches of 256\n", fn, s, p, bad);
                }
            }
            sel_map += " " + std::to_string(s) + "->" + (nfound == 1 ? std::to_string(found) : std::string("?"));
            ok = ok && nfound == 1 && found == s;
        }
        long bad2[2] = {}, swapped[2] = {};
        for (int ws = 0; ws < 2; ++ws) {
            for (uint32_t pair = 0; pair < 65536; ++pair) {
                const v2f g = h2[(f * 2 + ws) * 65536 + pair];
                const float lo = float_of(cr::decode8(f, pair & 0xFF)), hi = float_of(cr::decode8(f, pair >> 8));
                bad2[ws] += !(same_float(g.x, lo) && same_float(g.y, hi));
                swapped[ws] += !(same_float(g.x, hi) && same_float(g.y, lo));
                if (std::isnan(g.x)) {
                    ++nan_bits[bits_of(g.x)];
                }
            }
        }
        std::printf("RESULT decode %s packed: word_sel 0 %ld, word_sel 1 %ld mismatches of 65536 (high-byte-first "
                    "order would give %ld, %ld)\n",
                    fn, bad2[0], bad2[1], swapped[0], swapped[1]);
        std::printf("RESULT decode %s NaN outputs:", fn);
        for (auto & kv : nan_bits) {
            std::printf(" 0x%08x x%d", kv.first, kv.second);
        }
        std::printf("\n");
        ok = ok && bad2[0] == 0 && bad2[1] == 0;
        std::snprintf(buf, sizeof buf, "%s%s: selects%s, packed low byte first %s; ", text.empty() ? "" : "",
                      fn, sel_map.c_str(), bad2[0] + bad2[1] == 0 ? "exact" : "MISMATCH");
        text += buf;
    }
    text += "(both formats, all codes, all positions)";
    verdict("cvt_exact_decode", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}

static int run_encode(int f, int si, bool brk) {
    const cvt::Setting & s = cvt::SETTINGS[si];
    const bool sat = s.write && (s.mode >> 23) & 1;
    const int rnd = s.write ? (int) (s.mode & 3) : 0;
    EncStats * dst;
    CHECK(hipMalloc(&dst, sizeof(EncStats)));
    CHECK(hipMemset(dst, 0, sizeof(EncStats)));
    const uint64_t total = 1ull << 32, chunk = 1ull << 30;
    for (uint64_t start = 0; start < total; start += chunk) {
        if (f == cr::FP8) {
            encode<cr::FP8><<<4096, 256>>>(dst, start, chunk, s.write, s.mode, sat, rnd ? rnd : -1, brk);
        } else {
            encode<cr::BF8><<<4096, 256>>>(dst, start, chunk, s.write, s.mode, sat, rnd ? rnd : -1, brk);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    EncStats st;
    CHECK(hipMemcpy(&st, dst, sizeof st, hipMemcpyDeviceToHost));
    CHECK(hipFree(dst));
    const char * fn = f ? "bf8" : "fp8";
    std::string why;
    const bool ctrl_ok = cvt::control_ok(s, st.ctrl, st.mode_launch, why);
    std::printf("CONTROL %s %s: %s %s\n", fn, s.name, ctrl_ok ? "ok" : "FAILED", why.c_str());
    std::printf("RESULT encode %s %s reference rne%s: class count / mismatches slot0 slot1\n", fn, s.name,
                sat ? " saturating" : "");
    unsigned long long judged = 0;
    for (int c = 0; c < cr::NCLS; ++c) {
        unsigned long long n = 0;
        for (int b = 0; b < 256; ++b) {
            n += st.hist[c][b];
        }
        const unsigned long long m = st.mism[0][c] + st.mism[1][c];
        judged += m;
        std::printf("  %-11s %11llu / %llu %llu\n", cr::cls_name[c], n, st.mism[0][c], st.mism[1][c]);
        // histogram of hardware codes, most frequent first
        std::vector<std::pair<unsigned long long, int>> h;
        for (int b = 0; b < 256; ++b) {
            if (st.hist[c][b]) {
                h.push_back({st.hist[c][b], b});
            }
        }
        std::sort(h.rbegin(), h.rend());
        std::printf("HIST %s %s %-11s %3zu codes:", fn, s.name, cr::cls_name[c], h.size());
        for (size_t i = 0; i < h.size() && i < 8; ++i) {
            std::printf(" 0x%02x:%llu", h[i].second, h[i].first);
        }
        std::printf("%s\n", h.size() > 8 ? " ..." : "");
    }
    std::printf("RESULT encode %s %s: NaN inputs given the other NaN code (accepted) %llu; untouched half changed "
                "%llu; slot0 mismatches against directed rounding (mode %d) %llu\n",
                fn, s.name, st.nanalt[cr::C_NAN], st.kept_bad, rnd, rnd ? st.mism_dir[0] : 0ull);
    for (unsigned k = 0; k < std::min(st.nex, 16u); ++k) {
        std::printf("  example x=0x%08x (%g) class %s hardware 0x%02x reference 0x%02x\n", st.ex[k][0],
                    (double) float_of(st.ex[k][0]), cr::cls_name[st.ex[k][3]], st.ex[k][1], st.ex[k][2]);
    }
    const bool ok = ctrl_ok && judged == 0 && st.kept_bad == 0;
    char buf[300];
    std::snprintf(buf, sizeof buf,
                  "%s %s: 2^33 conversions, %llu mismatches, other NaN code %llu, +inf->0x%02x, control %s%s",
                  fn, s.name, judged, st.nanalt[cr::C_NAN],
                  (int) (std::max_element(st.hist[cr::C_INF], st.hist[cr::C_INF] + 256) - st.hist[cr::C_INF]),
                  ctrl_ok ? "ok" : "FAILED", brk ? " (--break: ties away)" : "");
    verdict(("cvt_exact_encode_" + std::string(fn) + "_" + s.name).c_str(), ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string what = argc > 1 ? argv[1] : "decode";
    if (what == "decode") {
        return run_decode();
    }
    if (what == "encode" && argc >= 4) {
        const std::string fs = argv[2];
        const int f = fs == "fp8" ? cr::FP8 : (fs == "bf8" ? cr::BF8 : -1);
        const int si = cvt::setting_index(argv[3]);
        const bool brk = argc > 4 && std::strcmp(argv[4], "--break") == 0;
        if (f < 0 || si < 0) {
            std::fprintf(stderr, "usage: cvt_exact encode fp8|bf8 <setting> [--break]\n");
            return 2;
        }
        return run_encode(f, si, brk);
    }
    std::fprintf(stderr, "usage: cvt_exact decode | encode fp8|bf8 <setting> [--break]\n");
    return 2;
}
