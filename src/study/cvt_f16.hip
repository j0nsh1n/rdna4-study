// FP16/BF16 conversion and packed-FP16 arithmetic probe for gfx1201 (RDNA 4, wave32).
// Every hardware result is checked against an integer-only reference written here; each reference is first
// checked on the CPU against an independent definition (x86 F16C, double arithmetic, fmaf, a nearest-value rule).
// Usage: cvt_f16 [f32f16|f16f32|bf16|pk|mix|all] [--break]
#include "harness.h"

#include <glob.h>
#include <immintrin.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>

typedef unsigned __int128 u128;

static std::atomic<int> g_nanex{0};  // limits the NaN-model counter-example prints
static bool g_break = false;  // --break: the RNE references round ties away / pk_fma reference rounds twice
static int g_max_temp = 0;
static double g_busy_s = 0;

// ---------------------------------------------------------------------------------------------------------------
// Thermal pacing: the card is shared, so every sweep waits for the hotspot to be cool and long jobs pause.
// ---------------------------------------------------------------------------------------------------------------
static int hotspot_mc() {
    glob_t g;
    int t = -1;
    if (glob("/sys/class/drm/card1/device/hwmon/hwmon*/temp2_input", 0, nullptr, &g) == 0) {
        FILE * f = std::fopen(g.gl_pathv[0], "r");
        if (f) {
            if (std::fscanf(f, "%d", &t) != 1) {
                t = -1;
            }
            std::fclose(f);
        }
    }
    globfree(&g);
    return t;
}

static void cool_wait() {
    for (int waited = 0;; ++waited) {
        int t = hotspot_mc();
        if (t > g_max_temp) {
            g_max_temp = t;
        }
        if (t >= 95000) {
            std::fprintf(stderr, "hotspot %d mC reached 95000: stopping\n", t);
            std::exit(2);
        }
        if (t < 65000 || waited > 240) {
            return;
        }
        sleep(1);
    }
}

static void pace(double s) {
    g_busy_s += s;
    if (g_busy_s > 8.0) {
        g_busy_s = 0;
        sleep(2);
    }
    cool_wait();
}

template <class F>
static void par_for(uint64_t n, F f) {
    const unsigned T = 8;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < T; ++t) {
        th.emplace_back([=, &f] { f(n * t / T, n * (t + 1) / T, t); });
    }
    for (auto & x : th) {
        x.join();
    }
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t u32() { return (uint32_t)next(); }
    uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};

// ---------------------------------------------------------------------------------------------------------------
// MODE register access (same pattern as arith_gpu.h): bits [7:0] and bit 23 are replaced, the rest is kept.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t MODE_MASK = 0x008000FFu;
constexpr uint32_t DEN_ON = 0xF0u;  // all denormals allowed (FP_DENORM = 3 in both fields)

__device__ __forceinline__ uint32_t mode_set(uint32_t mode) {
    uint32_t old, nw;
    asm volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_MODE)\n"
        "s_and_b32 %1, %0, %3\n"
        "s_or_b32 %1, %1, %2\n"
        "s_setreg_b32 hwreg(HW_REG_MODE), %1\n"
        "s_nop 7\ns_nop 7"
        : "=&s"(old), "=&s"(nw)
        : "s"(mode & MODE_MASK), "s"(~MODE_MASK)
        : "scc");
    return old;
}

__device__ __forceinline__ void mode_restore(uint32_t old) {
    // readfirstlane: after a divergent loop the compiler may keep `old` in a VGPR, which s_setreg cannot take.
    asm volatile("s_nop 7\ns_setreg_b32 hwreg(HW_REG_MODE), %0\ns_nop 7" ::"s"(__builtin_amdgcn_readfirstlane(old)));
}

__device__ __forceinline__ uint32_t mode_read() {
    uint32_t m;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE)" : "=s"(m));
    return m;
}

// ---------------------------------------------------------------------------------------------------------------
// FP32 -> FP16 integer reference (device + host). rm: 0 RNE, 1 +inf, 2 -inf, 3 toward zero, 4 ties away.
// ovfl: FP16_OVFL semantics (finite overflow clamps to max, true infinity stays infinity).
// ---------------------------------------------------------------------------------------------------------------
enum { RM_NE = 0, RM_PINF = 1, RM_NINF = 2, RM_ZERO = 3, RM_AWAY = 4 };

__host__ __device__ static inline uint32_t f32_to_f16_ref(uint32_t x, int rm, int ovfl, int * tie) {
    uint32_t s = (x >> 16) & 0x8000u;
    uint32_t e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
    *tie = 0;
    if (e == 255) {
        return m == 0 ? (s | 0x7C00u) : (s | 0x7E00u | (m >> 13));
    }
    uint64_t sig = e ? (m | 0x800000u) : m;
    if (sig == 0) {
        return s;
    }
    int E = e ? (int)e - 127 : -126;
    int r = 13 + (E < -14 ? -14 - E : 0);
    uint64_t q, rem, half;
    if (r >= 40) {
        q = 0;
        rem = 1;
        half = 1ull << 39;
    } else {
        q = sig >> r;
        rem = sig & ((1ull << r) - 1);
        half = 1ull << (r - 1);
    }
    bool neg = s != 0, inexact = rem != 0, up = false;
    switch (rm) {
        case RM_NE: up = rem > half || (rem == half && (q & 1)); break;
        case RM_PINF: up = !neg && inexact; break;
        case RM_NINF: up = neg && inexact; break;
        case RM_ZERO: up = false; break;
        default: up = rem >= half; break;
    }
    *tie = rem == half;
    q += up;
    uint32_t bits = E >= -14 ? (uint32_t)(((uint32_t)(E + 14) << 10) + q) : (uint32_t)q;
    if (bits >= 0x7C00u) {
        bool to_inf = rm == RM_NE || rm == RM_AWAY || (rm == RM_PINF && !neg) || (rm == RM_NINF && neg);
        bits = (to_inf && !ovfl) ? 0x7C00u : 0x7BFFu;
    }
    return s | bits;
}

// Input classes: 0 nan, 1 inf, 2 ovfl (finite, |x| >= 65520), 3 tie, 4 subnormal range, 5 zero, 6 other.
__host__ __device__ static inline int cls_of(uint32_t x, int tie) {
    uint32_t a = x & 0x7fffffffu;
    if (a > 0x7f800000u) return 0;
    if (a == 0x7f800000u) return 1;
    if (a == 0) return 5;
    if (a >= 0x477FF000u) return 2;
    if (tie) return 3;
    if (a < 0x38800000u) return 4;
    return 6;
}
static const char * const CLS[7] = {"nan", "inf", "ovfl", "tie", "subn", "zero", "other"};

struct Acc {
    uint32_t total[7], mism[7], ovh[2][4], nanh[5];
    uint32_t mism2, ctl_bad, mode_bad, nex;
    uint32_t ex[8][3];
};

struct Cnt {
    uint32_t total[7], mism[7], ovh[2][4], nanh[5], mism2;
};

__device__ static inline void judge(Acc * acc, Cnt & c, uint32_t x, uint32_t out, int rm, int ovfl, int rm2) {
    int tie;
    uint32_t ref = f32_to_f16_ref(x, rm, ovfl, &tie);
    int k = cls_of(x, tie);
    c.total[k]++;
    if (k == 0) {
        uint32_t s = ref & 0x8000u;
        int b;
        if ((out & 0x7fffu) > 0x7C00u) {
            b = out == ref ? 0 : out == (s | 0x7E00u) ? 1 : (out & 0x7fffu) == 0x7E00u ? 2 : 3;
        } else {
            b = 4;
        }
        c.nanh[b]++;
        return;
    }
    if (k == 1 || k == 2) {
        uint32_t m = out & 0x7fffu;
        int bk = m == 0x7BFFu ? 0 : m == 0x7C00u ? 1 : m > 0x7C00u ? 2 : 3;
        c.ovh[k - 1][bk]++;
    }
    if (out != ref) {
        c.mism[k]++;
        uint32_t i = atomicAdd(&acc->nex, 1u);
        if (i < 8) {
            acc->ex[i][0] = x;
            acc->ex[i][1] = out;
            acc->ex[i][2] = ref;
        }
    }
    if (rm2 >= 0) {
        int t2;
        if (out != f32_to_f16_ref(x, rm2, 0, &t2)) {
            c.mism2++;
        }
    }
}

__device__ static inline uint32_t wsum(uint32_t v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor(v, o);
    }
    return v;
}

typedef __fp16 half2_t __attribute__((ext_vector_type(2)));

// INSTR 0: v_cvt_f16_f32 (asm), 1: compiler cast (_Float16)x, 2: __builtin_amdgcn_cvt_pkrtz, 3: v_cvt_pk_rtz_f16_f32 (asm).
// For 2 and 3 the high half converts y = x * odd constant (a bijection), so both halves see all 2^32 inputs.
template <int INSTR>
__global__ __launch_bounds__(256) void f32f16_kernel(Acc * acc, uint64_t base, uint32_t count, uint32_t mode, int set_mode,
                                                     int rm, int ovfl, int rm2) {
    uint32_t old = 0;
    uint32_t bad_m = 0, bad_c = 0;
    if (set_mode) {
        old = mode_set(mode);
        uint32_t seen = mode_read();
        bad_m = ((seen ^ mode) & MODE_MASK) != 0;
        // VALU controls under the same MODE: 1 + 2^-24 and 1 + 2^-11 (half) only move under a directed round mode.
        uint32_t p32, n32, p16, n16;
        asm volatile("v_add_f32 %0, %1, %2" : "=v"(p32) : "v"(0x3F800000u), "v"(0x33800000u));
        asm volatile("v_add_f32 %0, %1, %2" : "=v"(n32) : "v"(0xBF800000u), "v"(0xB3800000u));
        asm volatile("v_add_f16 %0, %1, %2" : "=v"(p16) : "v"(0x3C00u), "v"(0x1000u));
        asm volatile("v_add_f16 %0, %1, %2" : "=v"(n16) : "v"(0xBC00u), "v"(0x9000u));
        uint32_t r1 = mode & 3, r2 = (mode >> 2) & 3;
        bad_c = (p32 != 0x3F800000u + (r1 == 1)) | (n32 != 0xBF800000u + (r1 == 2)) |
                ((p16 & 0xffffu) != 0x3C00u + (r2 == 1)) | ((n16 & 0xffffu) != 0xBC00u + (r2 == 2));
    }
    Cnt c;
    for (int i = 0; i < 7; ++i) { c.total[i] = 0; c.mism[i] = 0; }
    for (int i = 0; i < 8; ++i) { c.ovh[i >> 2][i & 3] = 0; }
    for (int i = 0; i < 5; ++i) { c.nanh[i] = 0; }
    c.mism2 = 0;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t k = tid; k < count; k += stride) {
        uint32_t x = (uint32_t)(base + k);
        if (INSTR == 0) {
            uint32_t r;
            asm volatile("v_cvt_f16_f32 %0, %1" : "=v"(r) : "v"(x));
            judge(acc, c, x, r & 0xffffu, rm, ovfl, rm2);
        } else if (INSTR == 1) {
            _Float16 h = (_Float16)__uint_as_float(x);
            judge(acc, c, x, (uint32_t)__builtin_bit_cast(uint16_t, h), rm, ovfl, rm2);
        } else {
            uint32_t y = x * 0x9E3779B1u;
            uint32_t r;
            if (INSTR == 2) {
                half2_t v = __builtin_amdgcn_cvt_pkrtz(__uint_as_float(x), __uint_as_float(y));
                r = __builtin_bit_cast(uint32_t, v);
            } else {
                asm volatile("v_cvt_pk_rtz_f16_f32 %0, %1, %2" : "=v"(r) : "v"(x), "v"(y));
            }
            judge(acc, c, x, r & 0xffffu, rm, ovfl, rm2);
            judge(acc, c, y, r >> 16, rm, ovfl, rm2);
        }
    }
    if (set_mode) {
        mode_restore(old);
    }
    bool lead = (threadIdx.x & 31) == 0;
    uint32_t v;
#define RED(dst, val) \
    v = wsum(val);    \
    if (lead && v) atomicAdd(&(dst), v)
    for (int i = 0; i < 7; ++i) {
        RED(acc->total[i], c.total[i]);
        RED(acc->mism[i], c.mism[i]);
    }
    for (int i = 0; i < 8; ++i) {
        RED(acc->ovh[i >> 2][i & 3], c.ovh[i >> 2][i & 3]);
    }
    for (int i = 0; i < 5; ++i) {
        RED(acc->nanh[i], c.nanh[i]);
    }
    RED(acc->mism2, c.mism2);
    RED(acc->ctl_bad, bad_c);
    RED(acc->mode_bad, bad_m);
#undef RED
}

static int eff_rne() { return g_break ? RM_AWAY : RM_NE; }

// ---------------------------------------------------------------------------------------------------------------
// CPU self-test of the FP32 -> FP16 reference against x86 F16C over all 2^32 inputs.
// ---------------------------------------------------------------------------------------------------------------
template <int IMM>
__attribute__((target("avx2,f16c"))) static uint64_t f16c_chunk(uint64_t lo, uint64_t hi, int rm, uint64_t * nan_bad) {
    uint64_t bad = 0, nb = 0;
    alignas(32) uint32_t in[8];
    alignas(16) uint16_t o[8];
    for (uint64_t b = lo; b < hi; b += 8) {
        for (int i = 0; i < 8; ++i) in[i] = (uint32_t)(b + i);
        __m256 v = _mm256_castsi256_ps(_mm256_load_si256((const __m256i *)in));
        _mm_store_si128((__m128i *)o, _mm256_cvtps_ph(v, IMM));
        for (int i = 0; i < 8; ++i) {
            int tie;
            uint32_t r = f32_to_f16_ref(in[i], rm, 0, &tie);
            if (r != o[i]) {
                bool isnan = (in[i] & 0x7fffffffu) > 0x7f800000u;
                (isnan ? nb : bad)++;
            }
        }
    }
    *nan_bad += nb;
    return bad;
}

static void selftest_f32f16(int rm, uint64_t * bad, uint64_t * nanbad) {
    // F16C immediates: 0 nearest, 1 down, 2 up, 3 truncate.
    std::vector<uint64_t> b(8, 0), nb(8, 0);
    par_for(1ull << 32, [&](uint64_t lo, uint64_t hi, unsigned t) {
        switch (rm) {
            case RM_NE:
            case RM_AWAY: b[t] = f16c_chunk<0>(lo, hi, rm, &nb[t]); break;
            case RM_PINF: b[t] = f16c_chunk<2>(lo, hi, rm, &nb[t]); break;
            case RM_NINF: b[t] = f16c_chunk<1>(lo, hi, rm, &nb[t]); break;
            default: b[t] = f16c_chunk<3>(lo, hi, rm, &nb[t]); break;
        }
    });
    *bad = *nanbad = 0;
    for (int t = 0; t < 8; ++t) {
        *bad += b[t];
        *nanbad += nb[t];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host side: sweeps of the 2^32 inputs.
// ---------------------------------------------------------------------------------------------------------------
static Acc * g_dacc = nullptr;

static Acc sweep(int instr, uint32_t mode, int set_mode, int rm, int ovfl, int rm2, double * ms_out) {
    Acc total;
    std::memset(&total, 0, sizeof total);
    cool_wait();
    CHECK(hipMemset(g_dacc, 0, sizeof(Acc)));
    auto t0 = std::chrono::steady_clock::now();
    const uint32_t count = 1u << 28;
    for (uint64_t chunk = 0; chunk < 16; ++chunk) {
        uint64_t base = chunk * count;
        dim3 grid(16384), blk(256);
        switch (instr) {
            case 0: f32f16_kernel<0><<<grid, blk>>>(g_dacc, base, count, mode, set_mode, rm, ovfl, rm2); break;
            case 1: f32f16_kernel<1><<<grid, blk>>>(g_dacc, base, count, mode, set_mode, rm, ovfl, rm2); break;
            case 2: f32f16_kernel<2><<<grid, blk>>>(g_dacc, base, count, mode, set_mode, rm, ovfl, rm2); break;
            default: f32f16_kernel<3><<<grid, blk>>>(g_dacc, base, count, mode, set_mode, rm, ovfl, rm2); break;
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(hipMemcpy(&total, g_dacc, sizeof total, hipMemcpyDeviceToHost));
    *ms_out = ms;
    pace(ms / 1000.0);
    return total;
}

static uint64_t sweep_bad(const Acc & a) {
    uint64_t b = a.nanh[4] + a.ctl_bad + a.mode_bad;
    for (int k = 1; k < 7; ++k) b += a.mism[k];
    return b;
}

static void print_acc(const char * label, const Acc & a, double ms, bool hist) {
    std::printf("  [%s] %.0f ms mism:", label, ms);
    for (int k = 1; k < 7; ++k) std::printf(" %s=%u/%u", CLS[k], a.mism[k], a.total[k]);
    std::printf(" | nan(%u): ref=%u qnan+sign=%u qnan=%u otherNaN=%u notNaN=%u", a.total[0], a.nanh[0], a.nanh[1],
                a.nanh[2], a.nanh[3], a.nanh[4]);
    if (a.mism2 || true) std::printf(" | vs2=%u", a.mism2);
    std::printf(" | ctl_bad=%u mode_bad=%u\n", a.ctl_bad, a.mode_bad);
    if (hist) {
        const char * nm[2] = {"inf", "ovfl"};
        for (int j = 0; j < 2; ++j) {
            std::printf("      %s-class outputs: max(0x7BFF)=%u inf(0x7C00)=%u nan=%u other=%u\n", nm[j], a.ovh[j][0],
                        a.ovh[j][1], a.ovh[j][2], a.ovh[j][3]);
        }
    }
    for (uint32_t i = 0; i < a.nex && i < 4; ++i) {
        std::printf("      e.g. x=0x%08x hw=0x%04x ref=0x%04x\n", a.ex[i][0], a.ex[i][1], a.ex[i][2]);
    }
}

static int run_f32f16() {
    std::printf("== f32f16: FP32 -> FP16, all 2^32 inputs ==\n");
    int fails = 0;
    char note[512];
    note[0] = 0;
    // 1. CPU self-test of the references against F16C.
    uint64_t st_bad[4] = {0, 0, 0, 0}, st_nan[4] = {0, 0, 0, 0};
    const int rms[4] = {eff_rne(), RM_PINF, RM_NINF, RM_ZERO};
    for (int i = 0; i < 4; ++i) {
        selftest_f32f16(rms[i], &st_bad[i], &st_nan[i]);
        std::printf("  selftest vs F16C rm=%d: non-NaN mismatches=%llu NaN mismatches=%llu\n", rms[i],
                    (unsigned long long)st_bad[i], (unsigned long long)st_nan[i]);
        if (st_bad[i] || st_nan[i]) ++fails;
    }
    int self_fail = fails;
    double ms;
    // 2. Baselines under RNE.
    Acc a = sweep(1, 0, 0, eff_rne(), 0, -1, &ms);
    print_acc("cast (_Float16)x, default MODE vs RNE", a, ms, true);
    fails += sweep_bad(a) != 0;
    uint32_t cast_bad = (uint32_t)sweep_bad(a);
    a = sweep(0, DEN_ON, 1, eff_rne(), 0, -1, &ms);
    print_acc("asm v_cvt_f16_f32, MODE=0xF0 vs RNE", a, ms, false);
    fails += sweep_bad(a) != 0;
    uint32_t asm_bad = (uint32_t)sweep_bad(a);
    // 3. Directed rounding, both fields: against the directed reference and against RNE (informational).
    const int dr[3] = {RM_PINF, RM_NINF, RM_ZERO};
    const int field_val[3] = {1, 2, 3};
    uint32_t dir_bad = 0, dir_vs_rne[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        uint32_t r = field_val[i];
        char lab[96];
        std::snprintf(lab, sizeof lab, "asm both fields MODE[3:0]=%u vs ref rm=%d (vs2 = RNE)", r | (r << 2), dr[i]);
        a = sweep(0, DEN_ON | r | (r << 2), 1, dr[i], 0, RM_NE, &ms);
        print_acc(lab, a, ms, false);
        dir_bad += (uint32_t)sweep_bad(a);
        dir_vs_rne[i] = a.mism2;
    }
    fails += dir_bad != 0;
    // 4. Which field controls the conversion: [1:0] only, [3:2] only.
    uint32_t lo_hits[3], hi_hits[3];
    for (int i = 0; i < 3; ++i) {
        uint32_t r = field_val[i];
        char lab[96];
        std::snprintf(lab, sizeof lab, "asm MODE[1:0]=%u only vs ref rm=%d (vs2 = RNE)", r, dr[i]);
        a = sweep(0, DEN_ON | r, 1, dr[i], 0, RM_NE, &ms);
        print_acc(lab, a, ms, false);
        lo_hits[i] = a.mism2;  // zero means it behaved like RNE
        fails += (a.ctl_bad || a.mode_bad) ? 1 : 0;
        std::snprintf(lab, sizeof lab, "asm MODE[3:2]=%u only vs ref rm=%d (vs2 = RNE)", r, dr[i]);
        a = sweep(0, DEN_ON | (r << 2), 1, dr[i], 0, RM_NE, &ms);
        print_acc(lab, a, ms, false);
        hi_hits[i] = a.mism2;
        fails += (a.ctl_bad || a.mode_bad) ? 1 : 0;
    }
    // 5. FP16_OVFL (bit 23) with each round mode of the FP16 field; reference models the clamp from the ISA text.
    uint32_t ovfl_bad = 0;
    for (int r = 0; r < 4; ++r) {
        char lab[96];
        std::snprintf(lab, sizeof lab, "asm FP16_OVFL=1 MODE[3:0]=%u vs ref rm=%d ovfl-clamp", r | (r << 2), r);
        a = sweep(0, DEN_ON | r | (r << 2) | (1u << 23), 1, r == 0 ? eff_rne() : r, 1, -1, &ms);
        print_acc(lab, a, ms, true);
        ovfl_bad += (uint32_t)sweep_bad(a);
    }
    fails += ovfl_bad != 0;
    // 6. v_cvt_pk_rtz_f16_f32: builtin at default MODE, asm under a directed MODE (it must ignore it).
    a = sweep(2, 0, 0, RM_ZERO, 0, -1, &ms);
    print_acc("pkrtz builtin, default MODE vs RTZ", a, ms, false);
    uint32_t pk_bad = (uint32_t)sweep_bad(a);
    for (int r = 1; r <= 2; ++r) {
        char lab[96];
        std::snprintf(lab, sizeof lab, "pkrtz asm, MODE[3:0]=%u vs RTZ", r | (r << 2));
        a = sweep(3, DEN_ON | r | (r << 2), 1, RM_ZERO, 0, -1, &ms);
        print_acc(lab, a, ms, false);
        pk_bad += (uint32_t)sweep_bad(a);
    }
    fails += pk_bad != 0;
    std::printf("  FINDING f32f16: FP16 round mode comes from MODE[3:2]: [1:0]-only matches RNE in %u/%u/%u mismatch-vs-RNE "
                "cases (r=1,2,3), [3:2]-only %u/%u/%u (0 = identical to RNE)\n",
                lo_hits[0], lo_hits[1], lo_hits[2], hi_hits[0], hi_hits[1], hi_hits[2]);
    std::snprintf(note, sizeof note,
                  "selftest_fail=%d cast_bad=%u asm_bad=%u directed_bad=%u (vs RNE %u/%u/%u) ovfl_bad=%u pkrtz_bad=%u",
                  self_fail, cast_bad, asm_bad, dir_bad, dir_vs_rne[0], dir_vs_rne[1], dir_vs_rne[2], ovfl_bad, pk_bad);
    verdict("cvt_f16_f32f16", fails ? "FAIL" : "PASS", note);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// FP16 -> FP32 (v_cvt_f32_f16), all 65536 codes.
// ---------------------------------------------------------------------------------------------------------------
static uint32_t f16_to_f32_ref(uint32_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ff;
    if (e == 31) return m == 0 ? (s | 0x7F800000u) : (s | 0x7FC00000u | (m << 13));
    if (e == 0) {
        if (m == 0) return s;
        int p = 31 - __builtin_clz(m);
        return s | ((uint32_t)(p - 24 + 127) << 23) | ((m << (23 - p)) & 0x7fffffu);
    }
    return s | ((e - 15 + 127) << 23) | (m << 13);
}

__global__ void f16f32_kernel(uint32_t * out, uint32_t * bad, uint32_t mode, int set_mode) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t old = 0;
    if (set_mode) {
        old = mode_set(mode);
        uint32_t seen = mode_read();
        if ((seen ^ mode) & MODE_MASK) atomicAdd(bad, 1u);
    }
    uint32_t r;
    asm volatile("v_cvt_f32_f16 %0, %1" : "=v"(r) : "v"(i));
    out[i] = r;
    if (set_mode) mode_restore(old);
}

__attribute__((target("f16c"))) static uint32_t f16c_widen(uint16_t h) {
    float f = _cvtsh_ss(h);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static int run_f16f32() {
    std::printf("== f16f32: FP16 -> FP32, all 65536 codes ==\n");
    int fails = 0;
    int self = 0;
    for (uint32_t h = 0; h < 65536; ++h) self += f16_to_f32_ref(h) != f16c_widen((uint16_t)h);
    std::printf("  selftest vs F16C _cvtsh_ss: mismatches=%d\n", self);
    uint32_t *d_out, *d_bad;
    CHECK(hipMalloc(&d_out, 65536 * 4));
    CHECK(hipMalloc(&d_bad, 4));
    std::vector<uint32_t> out(65536);
    struct Var {
        const char * name;
        uint32_t mode;
        int set;
    } vars[5] = {{"default MODE (no write)", 0, 0},
                 {"MODE=0xF0 (all denormals on)", 0xF0, 1},
                 {"MODE[7:4]=0 (flush FP32 + FP16/FP64 denormals)", 0x00, 1},
                 {"MODE[7:6]=0 (flush FP16/FP64 denormals only)", 0x30, 1},
                 {"MODE[5:4]=0 (flush FP32 denormals only)", 0xC0, 1}};
    uint32_t default_bad = 0;
    for (int v = 0; v < 5; ++v) {
        cool_wait();
        CHECK(hipMemset(d_bad, 0, 4));
        f16f32_kernel<<<256, 256>>>(d_out, d_bad, vars[v].mode, vars[v].set);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint32_t bad;
        CHECK(hipMemcpy(&bad, d_bad, 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(out.data(), d_out, 65536 * 4, hipMemcpyDeviceToHost));
        uint32_t mm[3] = {0, 0, 0};  // nan, subnormal, other
        uint32_t sub_flushed = 0, sub_total = 0, nan_total = 0, nan_exact = 0, nan_qbit = 0, nan_payload = 0, exs = 0;
        for (uint32_t h = 0; h < 65536; ++h) {
            uint32_t ref = f16_to_f32_ref(h), e = (h >> 10) & 31, m = h & 0x3ff;
            bool isnan = e == 31 && m;
            bool sub = e == 0 && m;
            if (isnan) {
                ++nan_total;
                nan_exact += out[h] == ref;
                nan_qbit += (out[h] >> 22) & 1;
                nan_payload += ((out[h] >> 13) & 0x3ff) == m;
            }
            if (sub) {
                ++sub_total;
                sub_flushed += out[h] == ((h & 0x8000u) << 16);
            }
            if (out[h] != ref) {
                ++mm[isnan ? 0 : sub ? 1 : 2];
                if (exs++ < 3) std::printf("      e.g. h=0x%04x hw=0x%08x ref=0x%08x\n", h, out[h], ref);
            }
        }
        std::printf("  [%s] mism nan=%u subn=%u other=%u | subnormal inputs flushed to zero: %u/%u | NaN: exact=%u/%u "
                    "quiet-bit-set=%u payload-kept=%u | mode_bad=%u\n",
                    vars[v].name, mm[0], mm[1], mm[2], sub_flushed, sub_total, nan_exact, nan_total, nan_qbit,
                    nan_payload, bad);
        if (v <= 1) default_bad += mm[0] + mm[1] + mm[2];
        if (bad) ++fails;
    }
    CHECK(hipFree(d_out));
    CHECK(hipFree(d_bad));
    fails += (self != 0) + (default_bad != 0);
    char note[256];
    std::snprintf(note, sizeof note, "selftest_mism=%d default_mism=%u (flush variants printed above)", self, default_bad);
    verdict("cvt_f16_f16f32", fails ? "FAIL" : "PASS", note);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// BF16: the compiler's cast sequences, checked over all 2^32 inputs (down) and all 65536 codes (up).
// ---------------------------------------------------------------------------------------------------------------
struct BfAcc {
    uint32_t mism, nanh[5], nan_total, nex, ex[8][3];
};

__global__ __launch_bounds__(256) void bf16_down_kernel(BfAcc * acc, uint64_t base, uint32_t count) {
    uint32_t nanh[5] = {0, 0, 0, 0, 0}, mism = 0, nant = 0;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t k = tid; k < count; k += stride) {
        uint32_t x = (uint32_t)(base + k);
        __bf16 y = (__bf16)__uint_as_float(x);
        uint32_t out = __builtin_bit_cast(uint16_t, y);
        if ((x & 0x7fffffffu) > 0x7f800000u) {
            uint32_t s = (x >> 16) & 0x8000u;
            int b;
            if ((out & 0x7fffu) > 0x7f80u) {
                b = out == (s | 0x7FC0u | ((x >> 16) & 0x7fu)) ? 0 : out == (s | 0x7FC0u) ? 1 : (out & 0x7fffu) == 0x7FC0u ? 2 : 3;
            } else {
                b = 4;
            }
            nanh[b]++;
            ++nant;
        } else {
            uint32_t ref = (x + 0x7fffu + ((x >> 16) & 1)) >> 16;
            if (out != ref) {
                ++mism;
                uint32_t i = atomicAdd(&acc->nex, 1u);
                if (i < 8) {
                    acc->ex[i][0] = x;
                    acc->ex[i][1] = out;
                    acc->ex[i][2] = ref;
                }
            }
        }
    }
    bool lead = (threadIdx.x & 31) == 0;
    uint32_t v;
    v = wsum(mism);
    if (lead && v) atomicAdd(&acc->mism, v);
    v = wsum(nant);
    if (lead && v) atomicAdd(&acc->nan_total, v);
    for (int i = 0; i < 5; ++i) {
        v = wsum(nanh[i]);
        if (lead && v) atomicAdd(&acc->nanh[i], v);
    }
}

__global__ void bf16_up_kernel(uint32_t * out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    __bf16 y = __builtin_bit_cast(__bf16, (uint16_t)i);
    out[i] = __float_as_uint((float)y);
}

// Independent nearest-value rule for the non-NaN bf16 reference: pick the closer of the two neighbouring bf16 values
// by exact double distance, ties to the even mantissa.
static uint32_t bf16_nearest(uint32_t x) {
    uint32_t sign = x & 0x80000000u, a = x & 0x7fffffffu;
    uint32_t lo = a & 0xffff0000u, hi = lo + 0x10000u;
    if (a == lo) return (x >> 16);
    float fa, flo;
    std::memcpy(&fa, &a, 4);
    std::memcpy(&flo, &lo, 4);
    double dlo = (double)fa - (double)flo;
    double dhi;
    if (hi >= 0x7f800000u) {
        dhi = 340282366920938463463374607431768211456.0 - (double)fa;  // distance to 2^128
    } else {
        float fhi;
        std::memcpy(&fhi, &hi, 4);
        dhi = (double)fhi - (double)fa;
    }
    uint32_t pick;
    if (dlo < dhi) pick = lo;
    else if (dhi < dlo) pick = hi;
    else pick = ((lo >> 16) & 1) ? hi : lo;
    return (sign | pick) >> 16;
}

static int run_bf16() {
    std::printf("== bf16: FP32 <-> BF16 ==\n");
    std::printf("  READ: no V_CVT_*_BF16 / V_CVT_PK_BF16 opcode appears in the RDNA4 ISA text (the only BF16 VALU ops are "
                "V_DOT2_*BF16 and WMMA/SWMMAC); BuiltinsAMDGPU.def has cvt builtins for bf16 only under scalef32/gfx950 "
                "features (fp8/bf8/fp6/fp4 with scale), none for plain FP32<->BF16 on gfx12\n");
    int fails = 0;
    // Self-test of the integer reference against the nearest-value rule on a sample plus specials.
    uint64_t st_bad = 0, st_n = 0;
    {
        Rng rng(12345);
        auto chk = [&](uint32_t x) {
            if ((x & 0x7fffffffu) > 0x7f800000u) return;
            ++st_n;
            uint32_t ref = (x + 0x7fffu + ((x >> 16) & 1)) >> 16;
            st_bad += ref != bf16_nearest(x);
        };
        for (uint32_t i = 0; i < (1u << 24); ++i) chk(rng.u32());
        for (uint32_t hi = 0; hi < 65536; ++hi) {
            chk(hi << 16);
            chk((hi << 16) | 0x8000u);
            chk((hi << 16) | 0x7fffu);
            chk((hi << 16) | 0x8001u);
            chk((hi << 16) | 0xffffu);
            chk((hi << 16) | 0x1u);
        }
    }
    std::printf("  selftest nearest-value rule vs integer RNE reference: mismatches=%llu of %llu\n",
                (unsigned long long)st_bad, (unsigned long long)st_n);
    fails += st_bad != 0;
    BfAcc *d_acc, acc;
    CHECK(hipMalloc(&d_acc, sizeof(BfAcc)));
    CHECK(hipMemset(d_acc, 0, sizeof(BfAcc)));
    cool_wait();
    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t chunk = 0; chunk < 16; ++chunk) {
        bf16_down_kernel<<<16384, 256>>>(d_acc, chunk << 28, 1u << 28);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pace(ms / 1000.0);
    CHECK(hipMemcpy(&acc, d_acc, sizeof acc, hipMemcpyDeviceToHost));
    std::printf("  [compiler (__bf16)x over 2^32] %.0f ms non-NaN mismatches vs RNE=%u | NaN inputs=%u: payload-kept-quiet=%u "
                "qnan+sign=%u qnan(0x7FC0/0xFFC0 wrong sign)=%u otherNaN=%u notNaN=%u\n",
                ms, acc.mism, acc.nan_total, acc.nanh[0], acc.nanh[1], acc.nanh[2], acc.nanh[3], acc.nanh[4]);
    for (uint32_t i = 0; i < acc.nex && i < 4; ++i) {
        std::printf("      e.g. x=0x%08x hw=0x%04x ref=0x%04x\n", acc.ex[i][0], acc.ex[i][1], acc.ex[i][2]);
    }
    fails += (acc.mism != 0) + (acc.nanh[4] != 0);
    CHECK(hipFree(d_acc));
    uint32_t * d_up;
    CHECK(hipMalloc(&d_up, 65536 * 4));
    bf16_up_kernel<<<256, 256>>>(d_up);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> up(65536);
    CHECK(hipMemcpy(up.data(), d_up, 65536 * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(d_up));
    uint32_t up_bad = 0;
    for (uint32_t c = 0; c < 65536; ++c) up_bad += up[c] != (c << 16);
    std::printf("  [compiler (float)bf16 over 65536 codes] mismatches vs code<<16 (NaN included) = %u\n", up_bad);
    fails += up_bad != 0;
    char note[256];
    std::snprintf(note, sizeof note, "no dedicated cvt opcode; down mism=%u nan_notNaN=%u up mism=%u selftest=%llu", acc.mism,
                  acc.nanh[4], up_bad, (unsigned long long)st_bad);
    verdict("cvt_f16_bf16", fails ? "FAIL" : "PASS", note);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Exact host arithmetic for FP16 / FP32 operands (integer, 128 bit), rounded once to the destination format.
// ---------------------------------------------------------------------------------------------------------------
struct Fmt {
    int mb, minE, infField, signBit;
};
static const Fmt F16F = {10, -14, 31, 15};
static const Fmt F32F = {23, -126, 255, 31};

static int bitlen(u128 v) {
    uint64_t hi = (uint64_t)(v >> 64), lo = (uint64_t)v;
    return hi ? 128 - __builtin_clzll(hi) : (lo ? 64 - __builtin_clzll(lo) : 0);
}

struct Rnd {
    uint32_t bits;
    bool tie, ovf;
};

// Rounds sign * mag * 2^exp (mag > 0) to the format; rm as in f32_to_f16_ref.
static Rnd round_exact(const Fmt & f, bool sign, u128 mag, int exp, int rm) {
    int L = bitlen(mag), E = L - 1 + exp;
    int qe = (E < f.minE ? f.minE : E) - f.mb;
    int r = qe - exp;
    u128 q, rem = 0, half = 1;
    bool tie = false;
    if (r <= 0) {
        q = mag << (-r);
    } else if (r > L) {
        q = 0;
        rem = 1;
        half = 2;
    } else {
        q = mag >> r;
        rem = mag & (((u128)1 << r) - 1);
        half = (u128)1 << (r - 1);
        tie = rem == half;
    }
    bool inexact = rem != 0, up;
    switch (rm) {
        case RM_NE: up = rem > half || (rem == half && (q & 1)); break;
        case RM_PINF: up = !sign && inexact; break;
        case RM_NINF: up = sign && inexact; break;
        case RM_ZERO: up = false; break;
        default: up = rem >= half; break;
    }
    q += up;
    uint64_t bits = E >= f.minE ? ((uint64_t)(E - f.minE) << f.mb) + (uint64_t)q : (uint64_t)q;
    bool ovf = bits >= ((uint64_t)f.infField << f.mb);
    if (ovf) {
        bool to_inf = rm == RM_NE || rm == RM_AWAY || (rm == RM_PINF && !sign) || (rm == RM_NINF && sign);
        bits = to_inf ? ((uint64_t)f.infField << f.mb) : (((uint64_t)f.infField << f.mb) - 1);
    }
    return {(uint32_t)(bits | ((uint64_t)sign << f.signBit)), tie, ovf};
}

struct Val {
    int cls;  // 0 finite, 1 inf, 2 nan
    int sign;
    uint64_t m;
    int e;  // value = m * 2^e
};

static Val dec16(uint32_t h) {
    int s = (h >> 15) & 1, ex = (h >> 10) & 31;
    uint32_t fr = h & 0x3ff;
    if (ex == 31) return {fr ? 2 : 1, s, 0, 0};
    if (ex == 0) return {0, s, fr, -24};
    return {0, s, fr | 0x400u, ex - 25};
}

static Val dec32(uint32_t w) {
    int s = (w >> 31) & 1, ex = (w >> 23) & 255;
    uint32_t fr = w & 0x7fffffu;
    if (ex == 255) return {fr ? 2 : 1, s, 0, 0};
    if (ex == 0) return {0, s, fr, -149};
    return {0, s, fr | 0x800000u, ex - 150};
}

struct Ex {
    int kind;  // 0 finite, 1 inf, 2 nan
    bool sign;
    u128 mag;
    int exp;
};

// a*b + c exactly (has_c = false: the product alone). Terms more than 64 bits apart act as a sticky bit only.
static Ex fma_exact(const Val & a, const Val & b, const Val & c, bool has_c) {
    if (a.cls == 2 || b.cls == 2 || (has_c && c.cls == 2)) return {2, false, 0, 0};
    bool az = a.cls == 0 && a.m == 0, bz = b.cls == 0 && b.m == 0;
    if ((a.cls == 1 && bz) || (b.cls == 1 && az)) return {2, false, 0, 0};
    bool psign = a.sign ^ b.sign;
    bool pinf = a.cls == 1 || b.cls == 1;
    if (has_c && pinf && c.cls == 1 && psign != (bool)c.sign) return {2, false, 0, 0};
    if (pinf) return {1, psign, 0, 0};
    if (has_c && c.cls == 1) return {1, (bool)c.sign, 0, 0};
    u128 mp = (u128)a.m * b.m;
    int ep = a.e + b.e;
    if (!has_c) return {0, psign, mp, ep};
    bool sc = c.sign;
    if (mp == 0 && c.m == 0) return {0, psign && sc, 0, 0};
    if (mp == 0) return {0, sc, c.m, c.e};
    if (c.m == 0) return {0, psign, mp, ep};
    int tp = bitlen(mp) - 1 + ep, tc = bitlen(c.m) - 1 + c.e;
    bool same = psign == sc;
    if (tp - tc > 64) return {0, psign, (mp << 40) + (same ? 1 : (u128)0) - (same ? 0 : 1), ep - 40};
    if (tc - tp > 64) return {0, sc, ((u128)c.m << 40) + (same ? 1 : (u128)0) - (same ? 0 : 1), c.e - 40};
    int emin = ep < c.e ? ep : c.e;
    u128 A = mp << (ep - emin), B = (u128)c.m << (c.e - emin);
    if (same) return {0, psign, A + B, emin};
    if (A == B) return {0, false, 0, 0};
    return A > B ? Ex{0, psign, A - B, emin} : Ex{0, sc, B - A, emin};
}

struct Res {
    uint32_t bits;
    bool nan, tie, ovf;
    bool zero;
};

static Res finish(const Fmt & f, const Ex & x, int rm) {
    uint32_t infb = (uint32_t)f.infField << f.mb;
    if (x.kind == 2) return {infb | (1u << (f.mb - 1)), true, false, false, false};
    if (x.kind == 1) return {infb | ((uint32_t)x.sign << f.signBit), false, false, false, false};
    if (x.mag == 0) return {(uint32_t)x.sign << f.signBit, false, false, false, true};
    Rnd r = round_exact(f, x.sign, x.mag, x.exp, rm);
    return {r.bits, false, r.tie, r.ovf, false};
}

// Round mag*2^exp to `prec` significant bits (RNE), keeping the value: models an intermediate FP32 rounding.
static Ex round_prec(Ex x, int prec) {
    if (x.kind != 0 || x.mag == 0) return x;
    int L = bitlen(x.mag);
    if (L <= prec) return x;
    int r = L - prec;
    u128 q = x.mag >> r, rem = x.mag & (((u128)1 << r) - 1), half = (u128)1 << (r - 1);
    if (rem > half || (rem == half && (q & 1))) ++q;
    return {0, x.sign, q << r, x.exp};
}

// Classes for packed results: 0 nan, 1 inf, 2 ovfl, 3 tie, 4 subn, 5 zero, 6 other.
static int res_cls(const Res & r, const Ex & x, uint32_t mask) {
    if (r.nan) return 0;
    if (x.kind == 1) return 1;
    if (r.ovf) return 2;
    if (r.zero) return 5;
    if (r.tie) return 3;
    uint32_t m = r.bits & mask;
    if (m != 0 && m <= (mask >> 5 ? 0x3ffu : 0u)) return 4;
    return 6;
}

static bool isnan16(uint32_t h) { return (h & 0x7fffu) > 0x7C00u; }
static bool isnan32(uint32_t w) { return (w & 0x7fffffffu) > 0x7f800000u; }

// ---------------------------------------------------------------------------------------------------------------
// Self-tests of the exact reference against independent definitions.
// ---------------------------------------------------------------------------------------------------------------
__attribute__((target("f16c"))) static uint16_t f16c_narrow(float f) { return _cvtss_sh(f, 0); }

// Independent: round a double (exactly representable sum) to FP16 with rint and a power-of-two grid.
static uint32_t double_to_f16_rne(double v) {
    if (v == 0) return std::signbit(v) ? 0x8000u : 0u;
    double av = std::fabs(v);
    int e = std::ilogb(av);
    int qe = (e < -14 ? -14 : e) - 10;
    double r = std::rint(std::ldexp(av, -qe));
    r = std::ldexp(r, qe);
    uint32_t sign = std::signbit(v) ? 0x8000u : 0u;
    if (r > 65504.0) return sign | 0x7C00u;
    return sign | f16c_narrow((float)r);
}

static uint32_t rand_half_any(Rng & rng);

static uint32_t gen_half(Rng & rng) {
    uint32_t r = rng.u32(), sg = (r & 1) << 15;
    switch ((r >> 1) % 16) {
        case 0: case 1: case 2: case 3: case 4: case 5: return rng.u32() & 0xffffu;
        case 6: return sg;
        case 7: return sg | (1 + rng.below(1023));
        case 8: {
            uint32_t e = 29 + rng.below(3), m = rng.u32() & 0x3ff;
            if (e == 31 && rng.below(4)) m = 0;
            return sg | (e << 10) | m;
        }
        case 9: return sg | 0x7C00u;
        case 10: return sg | 0x7C00u | (1 + rng.below(1023));
        case 11: return sg | ((1 + rng.below(30)) << 10) | (rng.below(4) << 8);
        case 12: return sg | (1u << 10) | (rng.u32() & 0x3ff);
        case 13: return sg | ((14 + rng.below(3)) << 10);
        case 14: return sg | ((1 + rng.below(30)) << 10) | (rng.u32() & 0x3ff);
        default: return rng.u32() & 0xffffu;
    }
}

// Two FP16 normals whose exact product is halfway between two FP16 values. Returns the unbiased product exponent.
static int gen_tie_pair(Rng & rng, uint32_t emax, uint32_t * ha, uint32_t * hb) {
    for (;;) {
        bool big = rng.below(2);
        int tot = big ? 10 : 9;
        int s = rng.below(tot + 1), t = tot - s;
        if (s > 10 || t > 10) continue;
        auto odd_in = [&](int sh) -> uint32_t {
            uint32_t lo = sh == 10 ? 1u : (1024u >> sh), hi = 2047u >> sh;
            uint32_t v = lo + rng.below(hi - lo + 1);
            return (v | 1u) > hi ? hi : (v | 1u);
        };
        uint32_t ma = odd_in(s) << s, mb = odd_in(t) << t;
        uint32_t P = ma * mb;
        if (big ? !(P >= (1u << 21) && (P & 0x7ff) == 0x400) : !(P < (1u << 21) && (P & 0x3ff) == 0x200)) continue;
        uint32_t ea = rng.below(emax + 1), eb = rng.below(emax + 1);
        *ha = (rng.below(2) << 15) | ((ea + 15) << 10) | (ma & 0x3ff);
        *hb = (rng.below(2) << 15) | ((eb + 15) << 10) | (mb & 0x3ff);
        return (int)(ea + eb);
    }
}

struct PkRef {
    Res r;
    Ex x;
};

// op 0 fma, 1 add (a + b), 2 mul (a * b).
static PkRef pk_ref(int op, uint32_t a, uint32_t b, uint32_t c, int rm, bool dbl) {
    Val va = dec16(a), vb = dec16(b), vc = dec16(c);
    static const Val ONE = {0, 0, 1, 0};
    Ex x;
    if (op == 0) x = fma_exact(va, vb, vc, true);
    else if (op == 1) x = fma_exact(va, ONE, vb, true);
    else x = fma_exact(va, vb, vc, false);
    if (dbl && op == 0) x = round_prec(x, 24);
    return {finish(F16F, x, rm), x};
}

static int selftest_ref16() {
    int bad = 0;
    // (a) generic rounder vs F16C on f32 inputs with every exponent.
    Rng rng(777);
    for (int i = 0; i < 4000000; ++i) {
        uint32_t w = rng.u32();
        if (i & 1) w = (w & 0x807fffffu) | ((95 + rng.below(70)) << 23);
        Val v = dec32(w);
        if (v.cls == 2) continue;
        Res r = v.cls == 1 ? Res{0x7C00u | ((uint32_t)v.sign << 15), false, false, false, false}
                           : finish(F16F, Ex{0, (bool)v.sign, v.m, v.e}, RM_NE);
        float f;
        std::memcpy(&f, &w, 4);
        bad += r.bits != f16c_narrow(f);
    }
    // (b) fma / add / mul vs the double-precision route where double is exact (small exponent spread).
    uint64_t n = 0;
    for (int i = 0; i < 6000000; ++i) {
        uint32_t a = gen_half(rng), b = gen_half(rng), c = gen_half(rng);
        if (isnan16(a) || isnan16(b) || isnan16(c)) continue;
        if (((a | b | c) & 0x7c00u) == 0x7c00u && ((a & 0x7c00) == 0x7c00 || (b & 0x7c00) == 0x7c00 || (c & 0x7c00) == 0x7c00)) continue;
        auto h2d = [](uint32_t h) { return (double)dec16(h).m * std::ldexp(1.0, dec16(h).e) * (((h >> 15) & 1) ? -1.0 : 1.0); };
        double da = h2d(a), db = h2d(b), dc = h2d(c);
        double pm = da * db;
        if (pm != 0 && dc != 0 && std::fabs(std::ilogb(pm) - std::ilogb(dc)) > 24) continue;
        ++n;
        double vf = std::fma(da, db, dc), va = da + db, vm = da * db;
        auto bz = [](double v, uint32_t ref) { return std::fabs(v) > 1e30 ? false : (double_to_f16_rne(v) != ref); };
        bad += bz(vf, pk_ref(0, a, b, c, RM_NE, false).r.bits);
        bad += bz(va, pk_ref(1, a, b, c, RM_NE, false).r.bits);
        bad += bz(vm, pk_ref(2, a, b, c, RM_NE, false).r.bits);
    }
    // (c) tie pairs: the product is exactly halfway, so it rounds to even; check against the double route too.
    for (int i = 0; i < 300000; ++i) {
        uint32_t a, b;
        gen_tie_pair(rng, 7, &a, &b);
        double da = dec16(a).m * std::ldexp(1.0, dec16(a).e) * (((a >> 15) & 1) ? -1.0 : 1.0);
        double db = dec16(b).m * std::ldexp(1.0, dec16(b).e) * (((b >> 15) & 1) ? -1.0 : 1.0);
        bad += double_to_f16_rne(da * db) != pk_ref(2, a, b, 0, RM_NE, false).r.bits;
        bad += !pk_ref(2, a, b, 0, RM_NE, false).r.tie;
    }
    // (d) FP32 fma incl. huge exponent spread (sticky path) vs fmaf.
    for (int i = 0; i < 4000000; ++i) {
        uint32_t w[3];
        for (int k = 0; k < 3; ++k) {
            w[k] = rng.u32();
            if (rng.below(4)) w[k] = (w[k] & 0x807fffffu) | ((100 + rng.below(60)) << 23);
        }
        if (rng.below(8) == 0) w[2] = (w[2] & 0x807fffffu) | (rng.below(120) << 23);
        float fa, fb, fc;
        std::memcpy(&fa, &w[0], 4);
        std::memcpy(&fb, &w[1], 4);
        std::memcpy(&fc, &w[2], 4);
        Val va = dec32(w[0]), vb = dec32(w[1]), vc = dec32(w[2]);
        if (va.cls == 2 || vb.cls == 2 || vc.cls == 2 || va.cls == 1 || vb.cls == 1 || vc.cls == 1) continue;
        float rf = std::fma(fa, fb, fc);
        uint32_t ru;
        std::memcpy(&ru, &rf, 4);
        Res r = finish(F32F, fma_exact(va, vb, vc, true), RM_NE);
        bad += r.bits != ru;
    }
    std::printf("  selftest of exact reference (F16C, double, fmaf; %llu double-route cases): mismatches=%d\n",
                (unsigned long long)n, bad);
    return bad;
}

// ---------------------------------------------------------------------------------------------------------------
// Packed FP16: v_pk_fma_f16, v_pk_add_f16, v_pk_mul_f16.
// ---------------------------------------------------------------------------------------------------------------
template <int OP>
__global__ void pk_kernel(const uint32_t * a, const uint32_t * b, const uint32_t * c, uint32_t * out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t r;
    if (OP == 0) asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a[i]), "v"(b[i]), "v"(c[i]));
    else if (OP == 1) asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(r) : "v"(a[i]), "v"(b[i]));
    else asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(r) : "v"(a[i]), "v"(b[i]));
    out[i] = r;
}

struct PkStat {
    uint64_t total[2][7] = {}, mism[2][7] = {}, nan_total = 0, nan_notnan = 0;
    uint64_t fused_vs_unfused = 0, hw_is_unfused_only = 0, hw_is_dbl_only = 0, dbl_vs_fused = 0;
    std::map<uint32_t, uint64_t> nanpat;
    uint64_t single = 0, single_ok = 0, invalid = 0, invalid_ok = 0, multi = 0, multi_first = 0, multi_last = 0;
    uint32_t ex[4][4];
    int nex = 0;
};

static void gen_pk_inputs(int op, uint64_t N, std::vector<uint32_t> & A, std::vector<uint32_t> & B, std::vector<uint32_t> & C) {
    A.resize(N);
    B.resize(N);
    C.resize(N);
    par_for(N, [&](uint64_t lo, uint64_t hi, unsigned t) {
        Rng rng(1000 + 77 * op + t * 1315423911ull);
        for (uint64_t i = lo; i < hi; ++i) {
            uint32_t w[3];
            for (int h = 0; h < 2; ++h) {
                uint32_t a, b, c;
                uint32_t sel = rng.below(8);
                if (sel == 0) {
                    // halfway products plus a tiny addend: separates fused from unfused and from FP32-then-FP16.
                    int Ep = gen_tie_pair(rng, 7, &a, &b);
                    int kc = Ep - 25 - (int)rng.below(10);
                    uint32_t sg = rng.below(2) << 15;
                    if (kc >= -14) c = sg | ((uint32_t)(kc + 15) << 10) | (rng.u32() & 0x3ff);
                    else c = sg | (1u << (kc < -24 ? 0 : kc + 24));
                } else if (sel == 1 && op == 1) {
                    // add: a half-ulp addend that makes an exact tie.
                    a = gen_half(rng) & 0xfbffu;
                    int ea = (a >> 10) & 31;
                    b = ea >= 12 ? (rng.below(2) << 15) | ((ea - 11) << 10) : gen_half(rng);
                    c = gen_half(rng);
                } else {
                    a = gen_half(rng);
                    b = gen_half(rng);
                    c = gen_half(rng);
                }
                w[0] = h ? (w[0] & 0xffffu) | (a << 16) : a;
                w[1] = h ? (w[1] & 0xffffu) | (b << 16) : b;
                w[2] = h ? (w[2] & 0xffffu) | (c << 16) : c;
            }
            A[i] = w[0];
            B[i] = w[1];
            C[i] = w[2];
        }
    });
}

static int run_pk() {
    std::printf("== pk: v_pk_fma_f16 / v_pk_add_f16 / v_pk_mul_f16 ==\n");
    int fails = 0;
    int self = selftest_ref16();
    fails += self != 0;
    const uint64_t N = 12000000;
    const char * opn[3] = {"pk_fma", "pk_add", "pk_mul"};
    uint32_t *dA, *dB, *dC, *dO;
    CHECK(hipMalloc(&dA, N * 4));
    CHECK(hipMalloc(&dB, N * 4));
    CHECK(hipMalloc(&dC, N * 4));
    CHECK(hipMalloc(&dO, N * 4));
    std::vector<uint32_t> A, B, C, O(N);
    char note[512];
    int len = 0;
    note[0] = 0;
    for (int op = 0; op < 3; ++op) {
        gen_pk_inputs(op, N, A, B, C);
        CHECK(hipMemcpy(dA, A.data(), N * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dB, B.data(), N * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dC, C.data(), N * 4, hipMemcpyHostToDevice));
        cool_wait();
        uint32_t blocks = (uint32_t)((N + 255) / 256);
        if (op == 0) pk_kernel<0><<<blocks, 256>>>(dA, dB, dC, dO, (uint32_t)N);
        else if (op == 1) pk_kernel<1><<<blocks, 256>>>(dA, dB, dC, dO, (uint32_t)N);
        else pk_kernel<2><<<blocks, 256>>>(dA, dB, dC, dO, (uint32_t)N);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(O.data(), dO, N * 4, hipMemcpyDeviceToHost));
        std::vector<PkStat> st(8);
        par_for(N, [&](uint64_t lo, uint64_t hi, unsigned t) {
            PkStat & s = st[t];
            for (uint64_t i = lo; i < hi; ++i) {
                for (int h = 0; h < 2; ++h) {
                    uint32_t a = (A[i] >> (16 * h)) & 0xffffu, b = (B[i] >> (16 * h)) & 0xffffu, c = (C[i] >> (16 * h)) & 0xffffu;
                    uint32_t hw = (O[i] >> (16 * h)) & 0xffffu;
                    PkRef pr = pk_ref(op, a, b, c, RM_NE, g_break && op == 0);
                    int k = res_cls(pr.r, pr.x, 0x7fffu);
                    s.total[h][k]++;
                    if (pr.r.nan) {
                        ++s.nan_total;
                        if (!isnan16(hw)) ++s.nan_notnan;
                        else s.nanpat[hw]++;
                        // NaN model: the first NaN operand (order a, b, c) quieted; an invalid operation gives 0xFE00.
                        uint32_t o3[3] = {a, b, c};
                        int no = op == 0 ? 3 : 2, first = -1, last = -1, cnt = 0;
                        for (int q = 0; q < no; ++q)
                            if (isnan16(o3[q])) {
                                if (first < 0) first = q;
                                last = q;
                                ++cnt;
                            }
                        bool ip = ((a & 0x7fffu) == 0x7C00u && (b & 0x7fffu) == 0) || ((b & 0x7fffu) == 0x7C00u && (a & 0x7fffu) == 0);
                        if (first < 0 || ip) {  // inf * 0 outranks a NaN addend
                            ++s.invalid;
                            s.invalid_ok += hw == 0xFE00u;
                        } else if (cnt == 1) {
                            ++s.single;
                            s.single_ok += hw == (o3[first] | 0x200u);
                            if (hw != (o3[first] | 0x200u) && g_nanex++ < 6)
                                std::printf("      NaN-model miss: op=%d a=0x%04x b=0x%04x c=0x%04x hw=0x%04x\n", op, a, b, c, hw);
                        } else {
                            ++s.multi;
                            s.multi_first += hw == (o3[first] | 0x200u);
                            s.multi_last += hw == (o3[last] | 0x200u);
                        }
                    } else if (hw != pr.r.bits) {
                        s.mism[h][k]++;
                        if (s.nex < 4) {
                            s.ex[s.nex][0] = a; s.ex[s.nex][1] = b; s.ex[s.nex][2] = c;
                            s.ex[s.nex][3] = hw | (pr.r.bits << 16);
                            ++s.nex;
                        }
                    }
                    if (op == 0 && !pr.r.nan) {
                        // evidence that the test can tell fused from unfused and from FP32-then-FP16
                        PkRef fu = pk_ref(0, a, b, c, RM_NE, false);
                        PkRef db = pk_ref(0, a, b, c, RM_NE, true);
                        Val va = dec16(a), vb = dec16(b), vc = dec16(c);
                        Ex p = fma_exact(va, vb, vc, false);
                        Res pr1 = finish(F16F, p, RM_NE);
                        Val vp = dec16(pr1.bits);
                        Res un = pr1.nan ? pr1 : finish(F16F, fma_exact(vp, Val{0, 0, 1, 0}, vc, true), RM_NE);
                        if (fu.r.bits != un.bits && !un.nan) {
                            ++s.fused_vs_unfused;
                            s.hw_is_unfused_only += hw == un.bits;
                        }
                        if (fu.r.bits != db.r.bits) {
                            ++s.dbl_vs_fused;
                            s.hw_is_dbl_only += hw == db.r.bits;
                        }
                    }
                }
            }
        });
        PkStat tot;
        for (auto & s : st) {
            for (int h = 0; h < 2; ++h)
                for (int k = 0; k < 7; ++k) {
                    tot.total[h][k] += s.total[h][k];
                    tot.mism[h][k] += s.mism[h][k];
                }
            tot.nan_total += s.nan_total;
            tot.nan_notnan += s.nan_notnan;
            tot.fused_vs_unfused += s.fused_vs_unfused;
            tot.hw_is_unfused_only += s.hw_is_unfused_only;
            tot.hw_is_dbl_only += s.hw_is_dbl_only;
            tot.dbl_vs_fused += s.dbl_vs_fused;
            tot.single += s.single; tot.single_ok += s.single_ok; tot.invalid += s.invalid; tot.invalid_ok += s.invalid_ok;
            tot.multi += s.multi; tot.multi_first += s.multi_first; tot.multi_last += s.multi_last;
            for (auto & kv : s.nanpat) tot.nanpat[kv.first] += kv.second;
            if (tot.nex < 4 && s.nex) {
                std::memcpy(tot.ex[tot.nex], s.ex[0], 16);
                ++tot.nex;
            }
        }
        uint64_t bad = tot.nan_notnan + (tot.single - tot.single_ok) + (tot.invalid - tot.invalid_ok) +
                       (tot.multi - tot.multi_first);
        std::printf("  %s over %llu words (%llu halves) mism[lo|hi] by class:", opn[op], (unsigned long long)N,
                    (unsigned long long)(2 * N));
        for (int k = 1; k < 7; ++k) {
            std::printf(" %s=%llu|%llu/%llu", CLS[k], (unsigned long long)tot.mism[0][k], (unsigned long long)tot.mism[1][k],
                        (unsigned long long)(tot.total[0][k] + tot.total[1][k]));
            bad += tot.mism[0][k] + tot.mism[1][k];
        }
        std::printf(" | nan results=%llu notNaN=%llu\n", (unsigned long long)tot.nan_total, (unsigned long long)tot.nan_notnan);
        std::printf("      NaN output patterns (hw):");
        std::vector<std::pair<uint64_t, uint32_t>> pv;
        for (auto & kv : tot.nanpat) pv.push_back({kv.second, kv.first});
        std::sort(pv.rbegin(), pv.rend());
        for (size_t i = 0; i < pv.size() && i < 6; ++i) std::printf(" 0x%04x:%llu", pv[i].second, (unsigned long long)pv[i].first);
        std::printf(" (distinct=%zu)\n", pv.size());
        std::printf("      NaN model (first NaN operand a,b,c quieted, sign kept; invalid op incl. inf*0 -> 0xFE00): one NaN operand %llu/%llu ok, "
                    "invalid op %llu/%llu ok, several NaN operands %llu: first-wins %llu last-wins %llu\n",
                    (unsigned long long)tot.single_ok, (unsigned long long)tot.single, (unsigned long long)tot.invalid_ok,
                    (unsigned long long)tot.invalid, (unsigned long long)tot.multi, (unsigned long long)tot.multi_first,
                    (unsigned long long)tot.multi_last);
        for (int i = 0; i < tot.nex; ++i)
            std::printf("      e.g. a=0x%04x b=0x%04x c=0x%04x hw=0x%04x ref=0x%04x\n", tot.ex[i][0], tot.ex[i][1], tot.ex[i][2],
                        tot.ex[i][3] & 0xffff, tot.ex[i][3] >> 16);
        if (op == 0)
            std::printf("      fused-vs-unfused: %llu cases where they differ, hw equals the unfused result in %llu of them; "
                        "FP32-then-FP16: %llu cases differ from fused, hw equals that in %llu\n",
                        (unsigned long long)tot.fused_vs_unfused, (unsigned long long)tot.hw_is_unfused_only,
                        (unsigned long long)tot.dbl_vs_fused, (unsigned long long)tot.hw_is_dbl_only);
        if (bad) ++fails;
        len += std::snprintf(note + len, sizeof note - len, "%s_bad=%llu ", opn[op], (unsigned long long)bad);
    }
    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dC));
    CHECK(hipFree(dO));
    std::snprintf(note + len, sizeof note - len, "selftest=%d", self);
    verdict("cvt_f16_pk", fails ? "FAIL" : "PASS", note);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// v_fma_mix_f32 / v_fma_mixlo_f16 / v_fma_mixhi_f16 with every combination of operand source
// (0 = FP32 word, 1 = FP16 low half, 2 = FP16 high half): pattern = s0 + 3*s1 + 9*s2.
// ---------------------------------------------------------------------------------------------------------------
#define MIX_SEL_f32 0, 0
#define MIX_SEL_lo 0, 1
#define MIX_SEL_hi 1, 1
#define MIXCASE_(OPN, P, O0, H0, O1, H1, O2, H2)                                                          \
    case P:                                                                                               \
        asm volatile(OPN " %0, %1, %2, %3 op_sel:[" #O0 "," #O1 "," #O2 "] op_sel_hi:[" #H0 "," #H1 "," #H2 \
                         "]"                                                                              \
                     : "+v"(d)                                                                            \
                     : "v"(A), "v"(B), "v"(C));                                                           \
        break;
#define MIXCASE_E(...) MIXCASE_(__VA_ARGS__)
#define MIXCASE(OPN, P, S0, S1, S2) MIXCASE_E(OPN, P, MIX_SEL_##S0, MIX_SEL_##S1, MIX_SEL_##S2)
#define LIST27(X, OPN) \
    X(OPN, 0, f32, f32, f32) \
    X(OPN, 1, lo, f32, f32) \
    X(OPN, 2, hi, f32, f32) \
    X(OPN, 3, f32, lo, f32) \
    X(OPN, 4, lo, lo, f32) \
    X(OPN, 5, hi, lo, f32) \
    X(OPN, 6, f32, hi, f32) \
    X(OPN, 7, lo, hi, f32) \
    X(OPN, 8, hi, hi, f32) \
    X(OPN, 9, f32, f32, lo) \
    X(OPN, 10, lo, f32, lo) \
    X(OPN, 11, hi, f32, lo) \
    X(OPN, 12, f32, lo, lo) \
    X(OPN, 13, lo, lo, lo) \
    X(OPN, 14, hi, lo, lo) \
    X(OPN, 15, f32, hi, lo) \
    X(OPN, 16, lo, hi, lo) \
    X(OPN, 17, hi, hi, lo) \
    X(OPN, 18, f32, f32, hi) \
    X(OPN, 19, lo, f32, hi) \
    X(OPN, 20, hi, f32, hi) \
    X(OPN, 21, f32, lo, hi) \
    X(OPN, 22, lo, lo, hi) \
    X(OPN, 23, hi, lo, hi) \
    X(OPN, 24, f32, hi, hi) \
    X(OPN, 25, lo, hi, hi) \
    X(OPN, 26, hi, hi, hi)

#define MIX_KERNEL(NAME, OPN)                                                                                         \
    __global__ void NAME(const uint32_t * a, const uint32_t * b, const uint32_t * c, const uint32_t * d0, uint32_t * out, \
                         uint32_t n, int pat) {                                                                       \
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;                                                           \
        if (i >= n) return;                                                                                           \
        uint32_t A = a[i], B = b[i], C = c[i], d = d0[i];                                                             \
        switch (pat) {                                                                                                \
            LIST27(MIXCASE, OPN)                                                                                      \
            default: break;                                                                                           \
        }                                                                                                             \
        out[i] = d;                                                                                                   \
    }
MIX_KERNEL(mix_f32_kernel, "v_fma_mix_f32")
MIX_KERNEL(mixlo_kernel, "v_fma_mixlo_f16")
MIX_KERNEL(mixhi_kernel, "v_fma_mixhi_f16")

static uint32_t gen_f32(Rng & rng) {
    uint32_t r = rng.u32();
    switch (rng.below(10)) {
        case 0: case 1: return r;
        case 2: case 3: case 4: return (r & 0x807fffffu) | ((100 + rng.below(60)) << 23);
        case 5: return (r & 0x80000000u) | (rng.below(4) == 0 ? 0 : (rng.below(0x800000)));  // zero or subnormal
        case 6: return (r & 0x80000000u) | 0x7f800000u | (rng.below(3) ? 0 : (rng.u32() & 0x7fffffu));
        case 7: return (r & 0x807fffffu) | ((rng.below(2) ? 253u + rng.below(2) : 1 + rng.below(3)) << 23);
        default: return f16_to_f32_ref(gen_half(rng));
    }
}

static uint32_t place(Rng & rng, int sel, uint32_t f32bits, bool f16ok, uint32_t h16) {
    if (sel == 0) return f16ok ? f16_to_f32_ref(h16) : f32bits;
    uint32_t decoy = rng.u32() & 0xffffu;
    return sel == 1 ? (decoy << 16) | h16 : (h16 << 16) | decoy;
}

struct MixStat {
    uint64_t total = 0, mism = 0, nan_total = 0, nan_notnan = 0, keep_bad = 0, alt_diff = 0, alt_hw = 0;
    uint64_t single = 0, single_ok = 0, invalid = 0, invalid_ok = 0, multi = 0, multi_first = 0, multi_last = 0;
    std::map<uint32_t, uint64_t> nanpat;
    uint32_t ex[3][6];
    int nex = 0;
};

static int run_mix() {
    std::printf("== mix: v_fma_mix_f32 / v_fma_mixlo_f16 / v_fma_mixhi_f16 ==\n");
    int fails = 0;
    const char * opn[3] = {"mix_f32", "mixlo_f16", "mixhi_f16"};
    const uint32_t N = 40000;
    uint32_t *dA, *dB, *dC, *dD, *dO;
    CHECK(hipMalloc(&dA, N * 4));
    CHECK(hipMalloc(&dB, N * 4));
    CHECK(hipMalloc(&dC, N * 4));
    CHECK(hipMalloc(&dD, N * 4));
    CHECK(hipMalloc(&dO, N * 4));
    char note[512];
    int len = 0;
    note[0] = 0;
    for (int op = 0; op < 3; ++op) {
        MixStat tot;
        uint64_t pat_bad[27] = {};
        for (int pat = 0; pat < 27; ++pat) {
            int sel[3] = {pat % 3, (pat / 3) % 3, pat / 9};
            std::vector<uint32_t> A(N), B(N), C(N), D(N), O(N);
            Rng rng(9000 + 31 * op + pat);
            for (uint32_t i = 0; i < N; ++i) {
                uint32_t v[3];
                uint32_t h[3];
                for (int k = 0; k < 3; ++k) {
                    h[k] = gen_half(rng);
                    v[k] = gen_f32(rng);
                }
                uint32_t mode = rng.below(4);
                if (sel[2] == 0 && mode == 0) {
                    // halfway FP16 product plus an addend far below FP32 resolution of the product
                    uint32_t ha, hb;
                    int Ep = gen_tie_pair(rng, 5, &ha, &hb);
                    h[0] = ha;
                    h[1] = hb;
                    int kc = Ep - 26 - (int)rng.below(14);
                    v[2] = (rng.below(2) << 31) | ((uint32_t)(kc + 127) << 23) | (rng.u32() & 0x7fffffu);
                    A[i] = place(rng, sel[0], 0, true, h[0]);
                    B[i] = place(rng, sel[1], 0, true, h[1]);
                    C[i] = v[2];
                } else {
                    A[i] = place(rng, sel[0], v[0], false, h[0]);
                    B[i] = place(rng, sel[1], v[1], false, h[1]);
                    C[i] = place(rng, sel[2], v[2], false, h[2]);
                    if (sel[0] == 0 && sel[1] == 0 && sel[2] == 0 && mode == 1) {
                        float fa, fb;
                        std::memcpy(&fa, &A[i], 4);
                        std::memcpy(&fb, &B[i], 4);
                        float p = fa * fb;
                        if (std::isfinite(p)) {
                            std::memcpy(&C[i], &p, 4);
                            C[i] ^= 0x80000000u;
                        }
                    }
                }
                D[i] = rng.u32();
            }
            CHECK(hipMemcpy(dA, A.data(), N * 4, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(dB, B.data(), N * 4, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(dC, C.data(), N * 4, hipMemcpyHostToDevice));
            CHECK(hipMemcpy(dD, D.data(), N * 4, hipMemcpyHostToDevice));
            cool_wait();
            if (op == 0) mix_f32_kernel<<<(N + 255) / 256, 256>>>(dA, dB, dC, dD, dO, N, pat);
            else if (op == 1) mixlo_kernel<<<(N + 255) / 256, 256>>>(dA, dB, dC, dD, dO, N, pat);
            else mixhi_kernel<<<(N + 255) / 256, 256>>>(dA, dB, dC, dD, dO, N, pat);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(O.data(), dO, N * 4, hipMemcpyDeviceToHost));
            auto val = [&](uint32_t w, int s) { return s == 0 ? dec32(w) : dec16(s == 1 ? (w & 0xffffu) : (w >> 16)); };
            MixStat st;
            auto opw = [&](uint32_t w, int sl) { return sl == 0 ? w : f16_to_f32_ref(sl == 1 ? (w & 0xffffu) : (w >> 16)); };
            auto to16 = [](uint32_t w) { return ((w >> 16) & 0x8000u) | 0x7C00u | 0x200u | ((w >> 13) & 0x3ffu); };
            for (uint32_t i = 0; i < N; ++i) {
                Val a = val(A[i], sel[0]), b = val(B[i], sel[1]), c = val(C[i], sel[2]);
                Ex x = fma_exact(a, b, c, true);
                uint32_t hw = O[i];
                st.total++;
                if (x.kind == 2) {
                    // NaN model: first NaN operand (a, b, c) widened and quieted; invalid operation -> default NaN.
                    uint32_t ow[3] = {opw(A[i], sel[0]), opw(B[i], sel[1]), opw(C[i], sel[2])};
                    int first = -1, last = -1, cnt = 0;
                    for (int q = 0; q < 3; ++q)
                        if (isnan32(ow[q])) {
                            if (first < 0) first = q;
                            last = q;
                            ++cnt;
                        }
                    uint32_t got = op == 0 ? hw : op == 1 ? (hw & 0xffffu) : (hw >> 16);
                    auto conv = [&](uint32_t w) { return op == 0 ? (w | 0x400000u) : to16(w); };
                    uint32_t dflt = op == 0 ? 0xFFC00000u : 0xFE00u;
                    auto isinf32 = [](uint32_t w) { return (w & 0x7fffffffu) == 0x7f800000u; };
                    auto iszero32 = [](uint32_t w) { return (w & 0x7fffffffu) == 0; };
                    bool ip = (isinf32(ow[0]) && iszero32(ow[1])) || (isinf32(ow[1]) && iszero32(ow[0]));
                    if (first < 0 || ip) {  // inf * 0 outranks a NaN addend
                        ++st.invalid;
                        st.invalid_ok += got == dflt;
                    } else if (cnt == 1) {
                        ++st.single;
                        st.single_ok += got == conv(ow[first]);
                        if (got != conv(ow[first]) && g_nanex++ < 6)
                            std::printf("      NaN-model miss: op=%d opnd=0x%08x 0x%08x 0x%08x hw=0x%08x\n", op, ow[0], ow[1], ow[2], hw);
                    } else {
                        ++st.multi;
                        st.multi_first += got == conv(ow[first]);
                        st.multi_last += got == conv(ow[last]);
                    }
                }
                if (op == 0) {
                    Res r = finish(F32F, x, RM_NE);
                    if (r.nan) {
                        ++st.nan_total;
                        if (!isnan32(hw)) ++st.nan_notnan;
                        else st.nanpat[hw]++;
                    } else if (hw != r.bits) {
                        ++st.mism;
                        if (st.nex < 3) {
                            uint32_t e6[6] = {A[i], B[i], C[i], hw, r.bits, 0};
                            std::memcpy(st.ex[st.nex++], e6, 24);
                        }
                    }
                    // unfused: product rounded to FP32 first
                    if (!r.nan && x.kind == 0) {
                        Ex p = fma_exact(a, b, c, false);
                        Res pr = finish(F32F, p, RM_NE);
                        if (!pr.nan && p.kind == 0) {
                            Val vp = dec32(pr.bits);
                            Res un = finish(F32F, fma_exact(vp, Val{0, 0, 1, 0}, c, true), RM_NE);
                            if (un.bits != r.bits && !un.nan) {
                                ++st.alt_diff;
                                st.alt_hw += hw == un.bits;
                            }
                        }
                    }
                } else {
                    Res r = finish(F16F, x, RM_NE);
                    uint32_t got = op == 1 ? (hw & 0xffffu) : (hw >> 16);
                    uint32_t kept = op == 1 ? (hw >> 16) : (hw & 0xffffu);
                    uint32_t kept_ref = op == 1 ? (D[i] >> 16) : (D[i] & 0xffffu);
                    st.keep_bad += kept != kept_ref;
                    if (r.nan) {
                        ++st.nan_total;
                        if (!isnan16(got)) ++st.nan_notnan;
                        else st.nanpat[got]++;
                    } else if (got != r.bits) {
                        ++st.mism;
                        if (st.nex < 3) {
                            uint32_t e6[6] = {A[i], B[i], C[i], hw, r.bits, 0};
                            std::memcpy(st.ex[st.nex++], e6, 24);
                        }
                    }
                    if (!r.nan && x.kind == 0) {
                        Res r32 = finish(F32F, x, RM_NE);
                        if (!r32.nan && r32.bits != 0 && (r32.bits & 0x7fffffffu) < 0x7f800000u) {
                            Val v32 = dec32(r32.bits);
                            Res dbl = finish(F16F, Ex{0, (bool)v32.sign, v32.m, v32.e}, RM_NE);
                            if (dbl.bits != r.bits) {
                                ++st.alt_diff;
                                st.alt_hw += got == dbl.bits;
                            }
                        }
                    }
                }
            }
            pat_bad[pat] = st.mism + st.nan_notnan + st.keep_bad + (st.single - st.single_ok) + (st.invalid - st.invalid_ok) +
                           (st.multi - st.multi_first);
            tot.single += st.single; tot.single_ok += st.single_ok; tot.invalid += st.invalid; tot.invalid_ok += st.invalid_ok;
            tot.multi += st.multi; tot.multi_first += st.multi_first; tot.multi_last += st.multi_last;
            tot.total += st.total;
            tot.mism += st.mism;
            tot.nan_total += st.nan_total;
            tot.nan_notnan += st.nan_notnan;
            tot.keep_bad += st.keep_bad;
            tot.alt_diff += st.alt_diff;
            tot.alt_hw += st.alt_hw;
            for (auto & kv : st.nanpat) tot.nanpat[kv.first] += kv.second;
            if (tot.nex < 3 && st.nex) std::memcpy(tot.ex[tot.nex++], st.ex[0], 24);
        }
        // targeted: mix_f32 is the only form with an FP32 destination; the alternative (double rounding) is meaningful for lo/hi.
        uint64_t bad = tot.mism + tot.nan_notnan + tot.keep_bad + (tot.single - tot.single_ok) + (tot.invalid - tot.invalid_ok) +
                       (tot.multi - tot.multi_first);
        std::printf("  %s: %llu cases over 27 operand-source patterns: mism=%llu, NaN results=%llu (notNaN=%llu), other-half "
                    "clobbered=%llu\n",
                    opn[op], (unsigned long long)tot.total, (unsigned long long)tot.mism, (unsigned long long)tot.nan_total,
                    (unsigned long long)tot.nan_notnan, (unsigned long long)tot.keep_bad);
        std::printf("      NaN model (first NaN operand quieted; invalid -> default NaN): one NaN operand %llu/%llu ok, invalid op %llu/%llu "
                    "ok, several NaN operands %llu: first-wins %llu last-wins %llu\n",
                    (unsigned long long)tot.single_ok, (unsigned long long)tot.single, (unsigned long long)tot.invalid_ok,
                    (unsigned long long)tot.invalid, (unsigned long long)tot.multi, (unsigned long long)tot.multi_first,
                    (unsigned long long)tot.multi_last);
        std::printf("      bad patterns:");
        int nb = 0;
        for (int p = 0; p < 27; ++p)
            if (pat_bad[p]) {
                std::printf(" %d(%llu)", p, (unsigned long long)pat_bad[p]);
                ++nb;
            }
        if (!nb) std::printf(" none");
        std::printf("\n      NaN output patterns:");
        std::vector<std::pair<uint64_t, uint32_t>> pv;
        for (auto & kv : tot.nanpat) pv.push_back({kv.second, kv.first});
        std::sort(pv.rbegin(), pv.rend());
        for (size_t i = 0; i < pv.size() && i < 5; ++i) std::printf(" 0x%x:%llu", pv[i].second, (unsigned long long)pv[i].first);
        std::printf(" (distinct=%zu)\n", pv.size());
        for (int i = 0; i < tot.nex; ++i)
            std::printf("      e.g. A=0x%08x B=0x%08x C=0x%08x hw=0x%08x ref=0x%08x\n", tot.ex[i][0], tot.ex[i][1], tot.ex[i][2],
                        tot.ex[i][3], tot.ex[i][4]);
        std::printf("      %s: %llu cases where %s differs from single rounding; hw equals the %s result in %llu of them\n",
                    op == 0 ? "fused-vs-unfused" : "single-vs-double-rounding", (unsigned long long)tot.alt_diff,
                    op == 0 ? "the product-rounded-first result" : "FP32-then-FP16", op == 0 ? "unfused" : "double-rounded",
                    (unsigned long long)tot.alt_hw);
        if (bad) ++fails;
        len += std::snprintf(note + len, sizeof note - len, "%s_bad=%llu ", opn[op], (unsigned long long)bad);
    }
    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dC));
    CHECK(hipFree(dD));
    CHECK(hipFree(dO));
    verdict("cvt_f16_mix", fails ? "FAIL" : "PASS", note);
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    const char * mode = "all";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--break")) g_break = true;
        else mode = argv[i];
    }
    hipDeviceProp_t prop = require_arch();
    std::printf("device: %s %s%s\n", prop.name, prop.gcnArchName, g_break ? "  [--break: reference deliberately wrong]" : "");
    CHECK(hipMalloc(&g_dacc, sizeof(Acc)));
    cool_wait();
    int rc = 0;
    bool all = !std::strcmp(mode, "all");
    if (all || !std::strcmp(mode, "f32f16")) rc |= run_f32f16();
    if (all || !std::strcmp(mode, "f16f32")) rc |= run_f16f32();
    if (all || !std::strcmp(mode, "bf16")) rc |= run_bf16();
    if (all || !std::strcmp(mode, "pk")) rc |= run_pk();
    if (all || !std::strcmp(mode, "mix")) rc |= run_mix();
    std::printf("hotspot max seen: %d mC\n", g_max_temp);
    return rc;
}
