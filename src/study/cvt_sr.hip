// Lane 6, item 3: stochastic rounding, v_cvt_sr_fp8_f32 and v_cvt_sr_bf8_f32 on gfx1201.
//
// cvt_sr rule <fp8|bf8>
//   about 56,500 inputs (56,514 for FP8, 56,502 for BF8: random values inside the finite range, values between the two largest codes and above, values in
//   the subnormal target range, exact codes, ties, mantissas close to all ones, zeros, infinities, NaNs), each
//   converted with every value of the operand's top bits (2^20 for FP8, 2^21 for BF8; the low bits random). Per
//   input: how many operand values round up (result differs from truncation), against the input's position
//   between its two codes; and per candidate rule in cvt_ref.h (sr8) the number of disagreeing conversions.
// cvt_sr sweep <fp8|bf8> <setting> <variant> [--break]
//   all 2^32 inputs, each with a pseudo-random operand, byte select cycling through 0-3, under one MODE setting
//   (cvt_gpu.h), judged against candidate rule <variant> (index into cr::sr_name). Also converts each input a
//   second time with the operand's low bits (the ones the rule ignores) flipped: the result must not change.
//   Counts by input class; the other three bytes of the destination must survive.
//   --break judges against the next rule in the list instead; the verdict must be FAIL.
#include "cvt_gpu.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t OLD = 0x5A5A5A5Au;

template <int F, int SEL>
__device__ __forceinline__ uint32_t sr_hw(uint32_t x, uint32_t r) {
    uint32_t w = OLD;
    if (F == cr::FP8) {
        asm volatile("v_cvt_sr_fp8_f32 %0, %1, %2 byte_sel:%3" : "+v"(w) : "v"(x), "v"(r), "i"(SEL));
    } else {
        asm volatile("v_cvt_sr_bf8_f32 %0, %1, %2 byte_sel:%3" : "+v"(w) : "v"(x), "v"(r), "i"(SEL));
    }
    return w;
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// One block per input. out[i * (2 + NSRVAR)]: ups, preserved-bytes failures, then mismatches per rule.
template <int F>
__global__ __launch_bounds__(256) void rule_kernel(const uint32_t * __restrict__ xs, uint32_t * __restrict__ out) {
    const int rb = F == cr::FP8 ? 20 : 21, sh = 32 - rb;
    const uint32_t x = xs[blockIdx.x];
    const uint32_t down = cr::encode8(F, x, false, cr::RTZ);
    unsigned ups = 0, kept = 0, mism[cr::NSRVAR] = {};
    for (uint32_t k = threadIdx.x; k < (1u << rb); k += blockDim.x) {
        const uint32_t r = (k << sh) | (hash32(k ^ x) >> rb);
        const uint32_t w = sr_hw<F, 0>(x, r);
        const uint32_t h = w & 0xFF;
        kept += (w & 0xFFFFFF00u) != (OLD & 0xFFFFFF00u);
        ups += h != down;
        for (int v = 0; v < cr::NSRVAR; ++v) {
            const uint32_t m = cr::sr8(F, x, r, v, false);
            mism[v] += h != m && !(cr::is_nan8(F, h) && cr::is_nan8(F, m));
        }
    }
    __shared__ unsigned acc[2 + cr::NSRVAR];
    if (threadIdx.x < 2 + cr::NSRVAR) {
        acc[threadIdx.x] = 0;
    }
    __syncthreads();
    atomicAdd(&acc[0], ups);
    atomicAdd(&acc[1], kept);
    for (int v = 0; v < cr::NSRVAR; ++v) {
        atomicAdd(&acc[2 + v], mism[v]);
    }
    __syncthreads();
    if (threadIdx.x < 2 + cr::NSRVAR) {
        out[blockIdx.x * (2 + cr::NSRVAR) + threadIdx.x] = acc[threadIdx.x];
    }
}

struct SweepStats {
    unsigned long long count[cr::NCLS], mism[cr::NCLS], lowdep[cr::NCLS], kept_bad, nanalt;
    unsigned int nex;
    uint32_t ex[16][5];
    uint32_t ctrl[cvt::NCTRL];
    uint32_t mode_launch;
};

template <int F>
__global__ __launch_bounds__(256) void sweep_kernel(SweepStats * st, uint64_t start, uint64_t count, int write,
                                                    uint32_t mode, int sat, int var) {
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
    const uint32_t lowmask = F == cr::FP8 ? 0xFFFu : 0x7FFu;
    unsigned long long cnt[cr::NCLS] = {}, mism[cr::NCLS] = {}, lowdep[cr::NCLS] = {}, kept = 0, nanalt = 0;
    for (uint64_t i = start + blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < start + count;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t x = (uint32_t) i, r = hash32(x * 0x9E3779B9u + 7u);
        const int sel = (x >> 9) & 3;
        uint32_t w, w2;
        switch (sel) {
        case 0: w = sr_hw<F, 0>(x, r); w2 = sr_hw<F, 0>(x, r ^ lowmask); break;
        case 1: w = sr_hw<F, 1>(x, r); w2 = sr_hw<F, 1>(x, r ^ lowmask); break;
        case 2: w = sr_hw<F, 2>(x, r); w2 = sr_hw<F, 2>(x, r ^ lowmask); break;
        default: w = sr_hw<F, 3>(x, r); w2 = sr_hw<F, 3>(x, r ^ lowmask); break;
        }
        const uint32_t m8 = 0xFFu << (8 * sel);
        kept += (w & ~m8) != (OLD & ~m8);
        const uint32_t h = (w >> (8 * sel)) & 0xFF;
        const uint32_t ref = cr::sr8(F, x, r, var, sat);
        const int c = cr::class8(F, x);
        ++cnt[c];
        if (h != ref) {
            if (cr::is_nan8(F, h) && cr::is_nan8(F, ref)) {
                ++nanalt;
            } else {
                ++mism[c];
                const unsigned k = atomicAdd(&st->nex, 1u);
                if (k < 16) {
                    st->ex[k][0] = x;
                    st->ex[k][1] = r;
                    st->ex[k][2] = h;
                    st->ex[k][3] = ref;
                    st->ex[k][4] = c;
                }
            }
        }
        lowdep[c] += w != w2;
    }
    if (write) {
        cvt::mode_restore(old);
    }
    for (int c = 0; c < cr::NCLS; ++c) {
        if (cnt[c]) {
            atomicAdd(&st->count[c], cnt[c]);
        }
        if (mism[c]) {
            atomicAdd(&st->mism[c], mism[c]);
        }
        if (lowdep[c]) {
            atomicAdd(&st->lowdep[c], lowdep[c]);
        }
    }
    if (kept) {
        atomicAdd(&st->kept_bad, kept);
    }
    if (nanalt) {
        atomicAdd(&st->nanalt, nanalt);
    }
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

// Input groups for the rule experiment, each with a name for the per-group summary.
struct Group {
    const char * name;
    int first, n;
};

static int run_rule(int f) {
    const cr::F8 fi = cr::f8info(f);
    const float maxv = float_of(cr::decode8(f, fi.maxcode));
    const float minnorm = std::ldexp(1.0f, fi.emin);
    std::mt19937_64 rng(777 + f);
    std::vector<uint32_t> xs;
    std::vector<Group> groups;
    auto add_group = [&](const char * name, int n, auto gen) {
        groups.push_back({name, (int) xs.size(), n});
        for (int i = 0; i < n; ++i) {
            xs.push_back(gen(i));
        }
    };
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    add_group("normal", 40000, [&](int) {
        const double lg = std::log2((double) minnorm) + u01(rng) * (std::log2((double) maxv) - std::log2((double) minnorm));
        uint32_t b = bits_of((float) std::exp2(lg));
        return (rng() & 1) ? b | 0x80000000u : b;
    });
    add_group("subnormal_target", 8000, [&](int) {
        uint32_t b = bits_of((float) (u01(rng) * minnorm));
        return (rng() & 1) ? b | 0x80000000u : b;
    });
    add_group("near_max", 4000, [&](int i) {
        // between the second largest code and twice the largest value
        const float lo = float_of(cr::decode8(f, fi.maxcode - 1));
        uint32_t b = bits_of(lo + (float) u01(rng) * (2 * maxv - lo));
        return (i & 1) ? b | 0x80000000u : b;
    });
    add_group("exact_codes", 2 * (int) fi.maxcode + 2, [&](int i) {
        return cr::decode8(f, (uint32_t) (i >> 1) | ((i & 1) ? 0x80u : 0u));
    });
    add_group("ties", 2 * (int) fi.maxcode, [&](int i) {
        const uint32_t c = (uint32_t) (i >> 1);
        const float a = float_of(cr::decode8(f, c)), b = float_of(cr::decode8(f, c + 1));
        const uint32_t m = bits_of(0.5f * (a + b));
        return (i & 1) ? m | 0x80000000u : m;
    });
    add_group("mantissa_near_ones", 4000, [&](int) {
        // the dropped bits all ones or nearly: tests whether the add carries into the exponent
        const int e = fi.emin + (int) (rng() % (uint64_t) (fi.bias + 1 - fi.emin));
        const uint32_t keep = (uint32_t) (rng() & ((1u << fi.M) - 1));
        const uint32_t low = ((1u << (23 - fi.M)) - 1) - (uint32_t) (rng() % 4096);
        uint32_t b = ((uint32_t) (e + 127) << 23) | (keep << (23 - fi.M)) | low;
        return (rng() & 1) ? b | 0x80000000u : b;
    });
    add_group("specials", 8, [&](int i) {
        const uint32_t s[8] = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x00000001u,
                               0x7F7FFFFFu};
        return s[i];
    });
    const int nx = (int) xs.size();
    uint32_t *dx, *dout;
    const int stride = 2 + cr::NSRVAR;
    CHECK(hipMalloc(&dx, nx * 4));
    CHECK(hipMalloc(&dout, (size_t) nx * stride * 4));
    CHECK(hipMemcpy(dx, xs.data(), nx * 4, hipMemcpyHostToDevice));
    if (f == cr::FP8) {
        rule_kernel<cr::FP8><<<nx, 256>>>(dx, dout);
    } else {
        rule_kernel<cr::BF8><<<nx, 256>>>(dx, dout);
    }
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out((size_t) nx * stride);
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(dx));
    CHECK(hipFree(dout));
    const char * fn = f ? "bf8" : "fp8";
    const int rb = f == cr::FP8 ? 20 : 21;
    std::printf("RESULT sr rule %s: %d inputs x 2^%d operand values; per group: inputs, kept-bytes failures, inputs "
                "whose round-up count != position, then mismatching conversions per candidate rule\n",
                fn, nx, rb);
    unsigned long long tot[cr::NSRVAR] = {};
    long kept_tot = 0, prob_tot = 0;
    for (const Group & g : groups) {
        unsigned long long per[cr::NSRVAR] = {};
        long kept = 0, prob_bad = 0, shown = 0;
        for (int i = g.first; i < g.first + g.n; ++i) {
            const uint32_t * o = &out[(size_t) i * stride];
            kept += o[1] != 0;
            for (int v = 0; v < cr::NSRVAR; ++v) {
                per[v] += o[2 + v];
            }
            // position in operand units: the top rb bits of the dropped fraction (normal and subnormal targets)
            const uint32_t x = xs[i], a = x & 0x7FFFFFFFu;
            if (a != 0 && a <= bits_of(maxv)) {
                uint32_t m24;
                int exp;
                const int sh = cr::dropped_bits(f, a, m24, exp);
                const uint32_t fr = sh >= 32 ? m24 : m24 & ((1u << sh) - 1);
                const uint32_t pos = sh - rb >= 32 ? 0u : fr >> (sh - rb);
                if (o[0] != pos) {
                    ++prob_bad;
                    if (shown++ < 3) {
                        std::printf("  %s x=0x%08x (%g): ups %u, position %u of %u\n", g.name, x, (double) float_of(x),
                                    o[0], pos, 1u << rb);
                    }
                }
            } else if (shown++ < 3) {
                std::printf("  %s x=0x%08x (%g): ups %u of %u\n", g.name, x, (double) float_of(x), o[0], 1u << rb);
            }
        }
        std::printf("  %-18s %6d %4ld %6ld |", g.name, g.n, kept, prob_bad);
        for (int v = 0; v < cr::NSRVAR; ++v) {
            std::printf(" %s %llu", cr::sr_name[v], per[v]);
            tot[v] += per[v];
        }
        std::printf("\n");
        kept_tot += kept;
        prob_tot += prob_bad;
    }
    int best = 0;
    for (int v = 1; v < cr::NSRVAR; ++v) {
        if (tot[v] < tot[best]) {
            best = v;
        }
    }
    char buf[300];
    std::snprintf(buf, sizeof buf,
                  "%s: best rule '%s' with %llu mismatches of %llu conversions; inputs in range whose round-up count "
                  "!= position: %ld; other bytes kept %s",
                  fn, cr::sr_name[best], tot[best], (unsigned long long) nx << rb, prob_tot, kept_tot ? "NO" : "yes");
    const bool ok = tot[best] == 0 && kept_tot == 0 && prob_tot == 0;
    verdict(("cvt_sr_rule_" + std::string(fn)).c_str(), ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

static int run_sweep(int f, int si, int var, bool brk) {
    const cvt::Setting & s = cvt::SETTINGS[si];
    const bool sat = s.write && (s.mode >> 23) & 1;
    const int judge = brk ? (var + 1) % cr::NSRVAR : var;
    SweepStats * dst;
    CHECK(hipMalloc(&dst, sizeof(SweepStats)));
    CHECK(hipMemset(dst, 0, sizeof(SweepStats)));
    const uint64_t total = 1ull << 32, chunk = 1ull << 30;
    for (uint64_t start = 0; start < total; start += chunk) {
        if (f == cr::FP8) {
            sweep_kernel<cr::FP8><<<4096, 256>>>(dst, start, chunk, s.write, s.mode, sat, judge);
        } else {
            sweep_kernel<cr::BF8><<<4096, 256>>>(dst, start, chunk, s.write, s.mode, sat, judge);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    SweepStats st;
    CHECK(hipMemcpy(&st, dst, sizeof st, hipMemcpyDeviceToHost));
    CHECK(hipFree(dst));
    const char * fn = f ? "bf8" : "fp8";
    std::string why;
    const bool ctrl_ok = cvt::control_ok(s, st.ctrl, st.mode_launch, why);
    std::printf("CONTROL sr %s %s: %s %s\n", fn, s.name, ctrl_ok ? "ok" : "FAILED", why.c_str());
    std::printf("RESULT sr sweep %s %s rule '%s'%s: class count / mismatches / results changed by the ignored low "
                "operand bits\n",
                fn, s.name, cr::sr_name[judge], sat ? " saturating" : "");
    unsigned long long m = 0, low = 0;
    for (int c = 0; c < cr::NCLS; ++c) {
        std::printf("  %-11s %11llu / %llu / %llu\n", cr::cls_name[c], st.count[c], st.mism[c], st.lowdep[c]);
        m += st.mism[c];
        low += st.lowdep[c];
    }
    for (unsigned k = 0; k < std::min(st.nex, 16u); ++k) {
        std::printf("  example x=0x%08x (%g) r=0x%08x class %s hardware 0x%02x rule 0x%02x\n", st.ex[k][0],
                    (double) float_of(st.ex[k][0]), st.ex[k][1], cr::cls_name[st.ex[k][4]], st.ex[k][2], st.ex[k][3]);
    }
    const bool ok = ctrl_ok && m == 0 && low == 0 && st.kept_bad == 0;
    char buf[300];
    std::snprintf(buf, sizeof buf,
                  "%s %s rule '%s': 2^32 inputs, %llu mismatches, other NaN code %llu, low operand bits changed %llu "
                  "results, other bytes changed %llu, control %s%s",
                  fn, s.name, cr::sr_name[judge], m, st.nanalt, low, st.kept_bad, ctrl_ok ? "ok" : "FAILED",
                  brk ? " (--break: wrong rule)" : "");
    verdict(("cvt_sr_sweep_" + std::string(fn) + "_" + s.name).c_str(), ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string what = argc > 2 ? argv[1] : "";
    const int f = argc > 2 ? (std::string(argv[2]) == "fp8" ? cr::FP8 : (std::string(argv[2]) == "bf8" ? cr::BF8 : -1))
                           : -1;
    if (what == "rule" && f >= 0) {
        return run_rule(f);
    }
    if (what == "sweep" && f >= 0 && argc >= 5) {
        const int si = cvt::setting_index(argv[3]);
        const int var = std::atoi(argv[4]);
        const bool brk = argc > 5 && std::strcmp(argv[5], "--break") == 0;
        if (si >= 0 && var >= 0 && var < cr::NSRVAR) {
            return run_sweep(f, si, var, brk);
        }
    }
    std::fprintf(stderr, "usage: cvt_sr rule fp8|bf8 | sweep fp8|bf8 <setting> <rule index> [--break]\n");
    return 2;
}
