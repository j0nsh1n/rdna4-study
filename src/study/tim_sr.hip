// Timing-lane leftovers: two more checks of the stochastic-rounding instructions (v_cvt_sr_fp8_f32, v_cvt_sr_bf8_f32).
//
// tim_sr select <fp8|bf8> [--break]
//   byte selects 1, 2 and 3 (cvt_sr's rule test only used select 0). Inputs: exact codes, exact ties (normal and
//   subnormal target range) and their FP32 neighbours, the overflow midpoint classes, ordinary values, both signs,
//   plus the specials. Each input is converted with every value of the operand's top 20 (FP8) / 21 (BF8) bits (the
//   bits below are 0, all ones or a hash, in turn). The selected byte must equal cr::sr8(SR_ALIGNED, no saturation)
//   and the three other destination bytes must keep the old value. --break judges against SR_CARRY_RNE instead.
// tim_sr bits <fp8|bf8> [--break]
//   how many operand bits the rounding uses. sr_aligned_bits() is cr::sr_aligned with the operand cut to its top B
//   bits; the hardware is compared against B-1, B and B+1 (B = 20 FP8, 21 BF8). B must give 0 mismatches and both
//   neighbours a nonzero count, else the input set cannot tell the bit counts apart. --break claims B+1 as the
//   accepted count.
#include "cvt_gpu.h"

#include <cmath>
#include <unistd.h>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t OLD = 0x5A5A5A5Au;

// Thermal rule (whole-card launches of a few tens of ms): start each launch below 60 C hotspot and stop the program at
// 95 C. The hotspot is read from the card record's HWMON (exported by the script); without it the gate is skipped.
static double hotspot_now() {
    const char * hw = std::getenv("HWMON");
    if (!hw) {
        return -1;
    }
    FILE * f = std::fopen((std::string(hw) + "/temp2_input").c_str(), "r");
    double v = -1000;
    if (f) {
        if (std::fscanf(f, "%lf", &v) != 1) {
            v = -1000;
        }
        std::fclose(f);
    }
    return v / 1000.0;
}
static void thermal_gate() {
    double t = hotspot_now();
    for (int i = 0; t >= 60.0 && i < 600; ++i) {
        usleep(100000);
        t = hotspot_now();
    }
    if (t >= 95.0) {
        std::printf("THERMAL hotspot %.0f C: stopping\n", t);
        std::exit(1);
    }
}

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

CR_FN uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Operand whose top rb bits are k; the bits below cycle through 0, all ones and a hash.
CR_FN uint32_t operand(uint32_t k, uint32_t x, int rb) {
    const int sh = 32 - rb;
    const uint32_t lowmask = (1u << sh) - 1;
    const uint32_t low = (k % 3 == 0) ? 0u : (k % 3 == 1) ? lowmask : (hash32(k ^ x) & lowmask);
    return (k << sh) | low;
}

// sr_aligned with the operand cut to its top B bits. Where the input has no bit as low as the operand's lowest
// kept bit (sh < B), the extra operand bits cannot carry and are dropped by the shift.
CR_FN uint32_t sr_aligned_bits(int f, uint32_t bits, uint32_t r, bool sat, int B) {
    const cr::F8 i = cr::f8info(f);
    const uint32_t a = bits & 0x7FFFFFFFu, s = (bits >> 31) ? 0x80u : 0u;
    if (a >= 0x7F800000u) {
        return cr::encode8(f, bits, sat, cr::RNE, false, false);
    }
    if (a == 0) {
        return s;
    }
    uint32_t m24;
    int exp;
    const int sh = cr::dropped_bits(f, a, m24, exp);
    const uint64_t R = r >> (32 - B);
    uint64_t q;
    if (sh >= B) {
        q = (sh - B >= 24) ? 0 : ((uint64_t) m24 + (R << (sh - B))) >> sh;
    } else {
        q = ((uint64_t) m24 + (R >> (B - sh))) >> sh;
    }
    const uint32_t code = exp < i.emin ? (uint32_t) q : ((uint32_t) (exp + i.bias - 1) << i.M) + (uint32_t) q;
    return s | (code > i.maxcode ? (sat ? i.maxcode : (f == cr::FP8 ? 0x7Fu : 0x7Cu)) : code);
}

// ---- select test: one block per input; out[i*3] mismatches, [1] other-byte failures, [2] other-NaN-code count
template <int F, int SEL>
__global__ __launch_bounds__(256) void select_kernel(const uint32_t * __restrict__ xs, uint32_t * __restrict__ out,
                                                     int var) {
    const int rb = F == cr::FP8 ? 20 : 21;
    const uint32_t x = xs[blockIdx.x];
    const uint32_t keep = ~(0xFFu << (8 * SEL));
    unsigned mism = 0, kept = 0, nanalt = 0;
    for (uint32_t k = threadIdx.x; k < (1u << rb); k += blockDim.x) {
        const uint32_t r = operand(k, x, rb);
        const uint32_t w = sr_hw<F, SEL>(x, r);
        const uint32_t h = (w >> (8 * SEL)) & 0xFF;
        const uint32_t ref = cr::sr8(F, x, r, var, false);
        kept += (w & keep) != (OLD & keep);
        if (h != ref) {
            if (cr::is_nan8(F, h) && cr::is_nan8(F, ref)) {
                ++nanalt;
            } else {
                ++mism;
            }
        }
    }
    __shared__ unsigned acc[3];
    if (threadIdx.x < 3) {
        acc[threadIdx.x] = 0;
    }
    __syncthreads();
    atomicAdd(&acc[0], mism);
    atomicAdd(&acc[1], kept);
    atomicAdd(&acc[2], nanalt);
    __syncthreads();
    if (threadIdx.x < 3) {
        out[blockIdx.x * 3 + threadIdx.x] = acc[threadIdx.x];
    }
}

// ---- bits test: out[i*4]: other-byte failures, then mismatches against B[0], B[1], B[2]
struct Bs {
    int b[3];
};

template <int F>
__global__ __launch_bounds__(256) void bits_kernel(const uint32_t * __restrict__ xs, uint32_t * __restrict__ out,
                                                   Bs bs) {
    const int rb = F == cr::FP8 ? 20 : 21;
    const uint32_t x = xs[blockIdx.x];
    unsigned kept = 0, mism[3] = {};
    for (uint32_t k = threadIdx.x; k < (1u << rb); k += blockDim.x) {
        const uint32_t r = operand(k, x, rb);
        const uint32_t w = sr_hw<F, 0>(x, r);
        const uint32_t h = w & 0xFF;
        kept += (w & 0xFFFFFF00u) != (OLD & 0xFFFFFF00u);
        for (int v = 0; v < 3; ++v) {
            const uint32_t m = sr_aligned_bits(F, x, r, false, bs.b[v]);
            mism[v] += h != m && !(cr::is_nan8(F, h) && cr::is_nan8(F, m));
        }
    }
    __shared__ unsigned acc[4];
    if (threadIdx.x < 4) {
        acc[threadIdx.x] = 0;
    }
    __syncthreads();
    atomicAdd(&acc[0], kept);
    for (int v = 0; v < 3; ++v) {
        atomicAdd(&acc[1 + v], mism[v]);
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        out[blockIdx.x * 4 + threadIdx.x] = acc[threadIdx.x];
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

struct Group {
    std::string name;
    int first, n;
};

struct InputSet {
    std::vector<uint32_t> xs;
    std::vector<Group> groups;
    template <class G>
    void add(const char * name, int n, G gen) {
        groups.push_back({name, (int) xs.size(), n});
        for (int i = 0; i < n; ++i) {
            xs.push_back(gen(i));
        }
    }
};

static uint32_t with_sign(uint32_t b, uint64_t coin) { return (coin & 1) ? b | 0x80000000u : b; }

// Groups shared by both tests: exact codes, ties, the overflow midpoint classes, the specials.
static void add_common(InputSet & s, int f, std::mt19937_64 & rng) {
    const cr::F8 fi = cr::f8info(f);
    const uint32_t maxbits = cr::decode8(f, fi.maxcode), mid = maxbits + (1u << (22 - fi.M));
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    s.add("exact_codes", 2 * (int) fi.maxcode + 2,
          [&](int i) { return cr::decode8(f, (uint32_t) (i >> 1) | ((i & 1) ? 0x80u : 0u)); });
    s.add("ties", 2 * (int) fi.maxcode, [&](int i) {
        const uint32_t c = (uint32_t) (i >> 1);
        const float a = float_of(cr::decode8(f, c)), b = float_of(cr::decode8(f, c + 1));
        return with_sign(bits_of(0.5f * (a + b)), (uint64_t) i);
    });
    // one FP32 step either side of every tie
    s.add("tie_neighbours", 4 * (int) fi.maxcode, [&](int i) {
        const uint32_t c = (uint32_t) (i >> 2);
        const float a = float_of(cr::decode8(f, c)), b = float_of(cr::decode8(f, c + 1));
        const uint32_t m = bits_of(0.5f * (a + b));
        return with_sign((i & 2) ? m + 1 : m - 1, (uint64_t) i);
    });
    s.add("at_mid", 2, [&](int i) { return with_sign(mid, (uint64_t) i); });
    s.add("max_to_mid", 64, [&](int i) {
        const uint32_t b = maxbits + 1 + (uint32_t) (rng() % (mid - maxbits - 1));
        return with_sign(b, (uint64_t) i);
    });
    s.add("above_mid", 64, [&](int i) {
        const uint32_t b = mid + 1 + (uint32_t) (rng() % (0x7F7FFFFFu - mid));
        return with_sign(b, (uint64_t) i);
    });
    s.add("specials", 8, [&](int i) {
        const uint32_t sp[8] = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x00000001u,
                                0x7F7FFFFFu};
        return sp[i];
    });
}

static void add_ordinary(InputSet & s, int f, std::mt19937_64 & rng, int n_norm, int n_sub, int n_deep) {
    const cr::F8 fi = cr::f8info(f);
    const float maxv = float_of(cr::decode8(f, fi.maxcode));
    const float minnorm = std::ldexp(1.0f, fi.emin);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    s.add("normal", n_norm, [&](int) {
        const double lg = std::log2((double) minnorm) + u01(rng) * (std::log2((double) maxv) - std::log2((double) minnorm));
        return with_sign(bits_of((float) std::exp2(lg)), rng());
    });
    s.add("subnormal_target", n_sub, [&](int) { return with_sign(bits_of((float) (u01(rng) * minnorm)), rng()); });
    // exponents emin-1 .. emin-5, random mantissa: the dropped part reaches weights 2^-21 .. 2^-25 of a step
    s.add("deep_sub", n_deep, [&](int) {
        const int e = fi.emin - 1 - (int) (rng() % 5);
        return with_sign(((uint32_t) (e + 127) << 23) | (uint32_t) (rng() & 0x7FFFFFu), rng());
    });
    s.add("mantissa_near_ones", n_norm / 4, [&](int) {
        const int e = fi.emin + (int) (rng() % (uint64_t) (fi.bias + 1 - fi.emin));
        const uint32_t keep = (uint32_t) (rng() & ((1u << fi.M) - 1));
        const uint32_t low = ((1u << (23 - fi.M)) - 1) - (uint32_t) (rng() % 4096);
        return with_sign(((uint32_t) (e + 127) << 23) | (keep << (23 - fi.M)) | low, rng());
    });
}

static int run_select(int f, bool brk) {
    std::mt19937_64 rng(4242 + f);
    InputSet s;
    add_ordinary(s, f, rng, 2000, 400, 400);
    add_common(s, f, rng);
    const int nx = (int) s.xs.size(), rb = f == cr::FP8 ? 20 : 21;
    const int var = brk ? cr::SR_CARRY_RNE : cr::SR_ALIGNED;
    const char * fn = f ? "bf8" : "fp8";
    uint32_t *dx, *dout;
    CHECK(hipMalloc(&dx, nx * 4));
    CHECK(hipMalloc(&dout, (size_t) nx * 12));
    CHECK(hipMemcpy(dx, s.xs.data(), nx * 4, hipMemcpyHostToDevice));
    std::printf("RESULT sr select %s: %d inputs x 2^%d operand values x 3 selects, judged against '%s'\n", fn, nx, rb,
                cr::sr_name[var]);
    unsigned long long tot_m = 0, tot_k = 0, tot_conv = 0;
    for (int sel = 1; sel <= 3; ++sel) {
        if (f == cr::FP8) {
            thermal_gate();
            if (sel == 1) select_kernel<cr::FP8, 1><<<nx, 256>>>(dx, dout, var);
            if (sel == 2) select_kernel<cr::FP8, 2><<<nx, 256>>>(dx, dout, var);
            if (sel == 3) select_kernel<cr::FP8, 3><<<nx, 256>>>(dx, dout, var);
        } else {
            thermal_gate();
            if (sel == 1) select_kernel<cr::BF8, 1><<<nx, 256>>>(dx, dout, var);
            if (sel == 2) select_kernel<cr::BF8, 2><<<nx, 256>>>(dx, dout, var);
            if (sel == 3) select_kernel<cr::BF8, 3><<<nx, 256>>>(dx, dout, var);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> out((size_t) nx * 3);
        CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
        unsigned long long cnt[cr::NCLS] = {}, mm[cr::NCLS] = {}, kp[cr::NCLS] = {}, na[cr::NCLS] = {};
        for (int i = 0; i < nx; ++i) {
            const int c = cr::class8(f, s.xs[i]);
            ++cnt[c];
            mm[c] += out[i * 3];
            kp[c] += out[i * 3 + 1];
            na[c] += out[i * 3 + 2];
        }
        std::printf("  select %d: class / inputs / mismatches / other-byte failures / other NaN code\n", sel);
        for (int c = 0; c < cr::NCLS; ++c) {
            if (cnt[c]) {
                std::printf("    %-11s %6llu / %llu / %llu / %llu\n", cr::cls_name[c], cnt[c], mm[c], kp[c], na[c]);
                tot_m += mm[c];
                tot_k += kp[c];
            }
        }
        tot_conv += (unsigned long long) nx << rb;
    }
    CHECK(hipFree(dx));
    CHECK(hipFree(dout));
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "%s selects 1-3: %d inputs, %llu conversions, %llu mismatches against '%s', other bytes changed %llu%s",
                  fn, nx, tot_conv, tot_m, cr::sr_name[var], tot_k, brk ? " (--break: wrong rule)" : "");
    const bool ok = tot_m == 0 && tot_k == 0;
    verdict(("tim_sr_select_" + std::string(fn)).c_str(), ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

static int run_bits(int f, bool brk) {
    std::mt19937_64 rng(9191 + f);
    InputSet s;
    add_ordinary(s, f, rng, 6000, 4000, 6000);
    add_common(s, f, rng);
    const int nx = (int) s.xs.size(), B = f == cr::FP8 ? 20 : 21, acc = brk ? B + 1 : B;
    const char * fn = f ? "bf8" : "fp8";

    // the parameterised reference must equal sr_aligned at the real B before anything is judged with it
    std::mt19937_64 rr(31337);
    unsigned long long refbad = 0, refn = 0;
    for (uint32_t x : s.xs) {
        for (int j = 0; j < 4096; ++j) {
            const uint32_t r = j == 0 ? 0u : j == 1 ? 0xFFFFFFFFu : j == 2 ? 0x80000000u : (uint32_t) rr();
            for (int sat = 0; sat < 2; ++sat) {
                refbad += sr_aligned_bits(f, x, r, sat, B) != cr::sr_aligned(f, x, r, sat);
                ++refn;
            }
        }
    }
    std::printf("CHECK sr_aligned_bits(B=%d) == cr::sr_aligned: %llu differences of %llu (CPU, %d inputs)\n", B, refbad,
                refn, nx);
    if (refbad) {
        verdict(("tim_sr_bits_" + std::string(fn)).c_str(), "FAIL", "parameterised reference does not reproduce sr_aligned");
        return 1;
    }

    uint32_t *dx, *dout;
    CHECK(hipMalloc(&dx, nx * 4));
    CHECK(hipMalloc(&dout, (size_t) nx * 16));
    CHECK(hipMemcpy(dx, s.xs.data(), nx * 4, hipMemcpyHostToDevice));
    const Bs bs = {{acc - 1, acc, acc + 1}};
    if (f == cr::FP8) {
        thermal_gate();
        bits_kernel<cr::FP8><<<nx, 256>>>(dx, dout, bs);
    } else {
        thermal_gate();
        bits_kernel<cr::BF8><<<nx, 256>>>(dx, dout, bs);
    }
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out((size_t) nx * 4);
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(dx));
    CHECK(hipFree(dout));
    std::printf("RESULT sr bits %s: %d inputs x 2^%d operand values; mismatching conversions per candidate bit count "
                "(B-1=%d, B=%d, B+1=%d, where B = %d is the claimed count)\n",
                fn, nx, B, bs.b[0], bs.b[1], bs.b[2], acc);
    unsigned long long tot[3] = {}, kept = 0;
    for (const Group & g : s.groups) {
        unsigned long long per[3] = {}, k = 0;
        for (int i = g.first; i < g.first + g.n; ++i) {
            k += out[i * 4];
            for (int v = 0; v < 3; ++v) {
                per[v] += out[i * 4 + 1 + v];
            }
        }
        std::printf("  %-19s %6d inputs  B-1: %llu  B: %llu  B+1: %llu  other-byte failures %llu\n", g.name.c_str(), g.n,
                    per[0], per[1], per[2], k);
        for (int v = 0; v < 3; ++v) {
            tot[v] += per[v];
        }
        kept += k;
    }
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "%s claimed B=%d: mismatches %llu; controls B-1 (%d) %llu, B+1 (%d) %llu; %d inputs x 2^%d operands; "
                  "other bytes changed %llu%s",
                  fn, acc, tot[1], bs.b[0], tot[0], bs.b[2], tot[2], nx, B, kept, brk ? " (--break: B+1 claimed)" : "");
    const bool ok = tot[1] == 0 && tot[0] > 0 && tot[2] > 0 && kept == 0;
    verdict(("tim_sr_bits_" + std::string(fn)).c_str(), ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string what = argc > 2 ? argv[1] : "";
    const int f = argc > 2 ? (std::string(argv[2]) == "fp8" ? cr::FP8 : (std::string(argv[2]) == "bf8" ? cr::BF8 : -1))
                           : -1;
    const bool brk = argc > 3 && std::strcmp(argv[3], "--break") == 0;
    if (what == "select" && f >= 0) {
        return run_select(f, brk);
    }
    if (what == "bits" && f >= 0) {
        return run_bits(f, brk);
    }
    std::fprintf(stderr, "usage: tim_sr select|bits fp8|bf8 [--break]\n");
    return 2;
}
