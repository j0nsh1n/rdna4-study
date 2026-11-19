// Matrix-core probe, whole card, wall clock: how many WMMA instructions per second the 128 SIMDs sustain, and what the
// shader clock, power and hotspot do while they run.
//
// Usage: mat_card <reg:f16|reg:bf16|reg:fp8|reg:iu8|reg:iu4|reg:iu4k32|reg:sf16|reg:sfp8|reg:siu8|mem:f16|mem:fp8|fma|stream>
//                  [--break] [--short]
//   reg:F    register-only matrix load: 16 WMMA per loop iteration in one asm block over 4 independent accumulators,
//            A and B fixed in registers. 64 work-groups of 64*W threads (2W waves) with 40 KiB of unused dynamic
//            LDS each, so one group per CU; W = 1, 2, 4, 8, 16 waves per SIMD. --break claims half the expected
//            cycles per instruction, so the verdict must FAIL. iu4: 4-bit K = 16, iu4k32: 4-bit K = 32, sf16 / sfp8 /
//            siu8: sparse (SWMMAC) FP16, FP8 and 8-bit integer, K = 32, index register fixed.
//   mem:F    one WMMA per iteration into a single accumulator, A and B fragments loaded from memory every iteration;
//            source "cache" (each wave walks a small region of its own in a 32 MiB buffer) and "dram" (each wave
//            walks its share of 1 GiB). W = 1, 4, 16.
//   fma      16 independent FMA chains per lane, W = 8, for a clock comparison.
//   stream   1 GiB of 16-byte loads, grid-stride, for a clock comparison.
// --short shortens every timed launch (for a first look). One mode per run keeps the GPU busy for about 20 s at most.
//
// Leftovers C (MT-5c, MT-3): mat_card burst <form> --w W --ms L [--tag T]
//   One timed launch of about L ms (no calibration launches: the iteration count comes from the known rate, 128 SIMDs
//   x 3.2 GHz / cycles), after a tiny warm-up launch; started only below 55 C (else exit 7 without load), stopped
//   in-process at 95 C. forms: f16, bf16 (existing kernels), sbf16, sbf8, bf8, f16f16, bf16bf16, sf16f16 (new).
//   Prints one BURST line: rate from HIP events and from the waves' REALTIME span, the waves' counter clock, the
//   ratio to 128 x clock / cycles, placement, power and hotspot. scripts/leftc_bursts.py combines the bursts.
// Paths of the sysfs folders: env DRM_DEVICE and HWMON, else runs/latest/card.env.
#include "mem_common.h"

#include <atomic>
#include <cmath>
#include <functional>
#include <map>
#include <type_traits>

typedef int i2v __attribute__((ext_vector_type(2)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef _Float16 h8v __attribute__((ext_vector_type(8)));

struct Rec {
    uint64_t pl, cyc, rt0, rt1;  // placement (mem::placement), cycles over the timed loop, REALTIME at start and end
};

constexpr int NSIMD = 128;
constexpr int MAXREC = 8192;

// ---------------------------------------------------------------------------------------------------------------
// Register-only kernels. F: 0 f16, 1 bf16, 2 fp8, 3 iu8.
#define WM(OP, ACC, NEG) OP " %" #ACC ", %4, %5, %" #ACC NEG "\n"
#define WM4(OP, NEG) WM(OP, 0, NEG) WM(OP, 1, NEG) WM(OP, 2, NEG) WM(OP, 3, NEG)
#define WM16(OP, NEG) WM4(OP, NEG) WM4(OP, NEG) WM4(OP, NEG) WM4(OP, NEG)

template <int F>
using Frag = typename std::conditional<(F < 2), i4v, i2v>::type;

template <int F>
__device__ __forceinline__ void wmma16(i8v & c0, i8v & c1, i8v & c2, i8v & c3, const Frag<F> & a, const Frag<F> & b) {
    if constexpr (F == 0) {
        asm volatile(WM16("v_wmma_f32_16x16x16_f16", "") : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b));
    } else if constexpr (F == 1) {
        asm volatile(WM16("v_wmma_f32_16x16x16_bf16", "") : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b));
    } else if constexpr (F == 2) {
        asm volatile(WM16("v_wmma_f32_16x16x16_fp8_fp8", "") : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b));
    } else {
        asm volatile(WM16("v_wmma_i32_16x16x16_iu8", " neg_lo:[1,1,0]")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b));
    }
}

__device__ __forceinline__ int wave_index() { return blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); }

template <int F>
__global__ __launch_bounds__(1024) void reg_kernel(const unsigned * __restrict__ ab, unsigned * __restrict__ sink,
                                                   Rec * __restrict__ rec, int iters) {
    Frag<F> a, b;
    for (int k = 0; k < (int) (sizeof(Frag<F>) / 4); ++k) {
        a[k] = (int) ab[k];
        b[k] = (int) ab[4 + k];
    }
    asm volatile("" : "+v"(a), "+v"(b));  // A and B stay in registers, not rebuilt from the buffer
    i8v c0 = {}, c1 = {}, c2 = {}, c3 = {};
    const uint32_t pl = mem::placement();
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        wmma16<F>(c0, c1, c2, c3, a, b);
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    unsigned x = 0;
    for (int k = 0; k < 8; ++k) {
        x ^= (unsigned) (c0[k] ^ c1[k] ^ c2[k] ^ c3[k]);
    }
    if (x == 0x7fc0ffeeu) {  // impossible for the small inputs; keeps the accumulators alive
        sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Register-only kernels for the 4-bit and sparse forms. F: 4 iu4 K = 16, 5 iu4 K = 32, 6 sparse f16, 7 sparse fp8,
// 8 sparse iu8. Same loop as reg_kernel; sparse forms take D as C, so the accumulator appears once, plus one index
// register. The operand words come from ab: A at 0, B at 8, index at 16.
#define WS(OP, ACC, MOD) OP " %" #ACC ", %4, %5, %6" MOD "\n"
#define WS4(OP, MOD) WS(OP, 0, MOD) WS(OP, 1, MOD) WS(OP, 2, MOD) WS(OP, 3, MOD)
#define WS16(OP, MOD) WS4(OP, MOD) WS4(OP, MOD) WS4(OP, MOD) WS4(OP, MOD)

template <int N>
using Vec = typename std::conditional<N == 1, int,
            typename std::conditional<N == 2, i2v, typename std::conditional<N == 4, i4v, i8v>::type>::type>::type;

template <int F>
struct XOps {
    static constexpr int NA = F == 4 ? 1 : (F == 5 || F == 7 || F == 8) ? 2 : 4;
    static constexpr int NB = F == 4 ? 1 : F == 5 ? 2 : F == 6 ? 8 : 4;
    using A = Vec<NA>;
    using B = Vec<NB>;
};

template <int F>
__device__ __forceinline__ void wmma16x(i8v & c0, i8v & c1, i8v & c2, i8v & c3, const typename XOps<F>::A & a,
                                        const typename XOps<F>::B & b, int idx) {
    if constexpr (F == 4) {
        asm volatile(WM16("v_wmma_i32_16x16x16_iu4", " neg_lo:[1,1,0]")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b));
    } else if constexpr (F == 5) {
        asm volatile(WM16("v_wmma_i32_16x16x32_iu4", " neg_lo:[1,1,0]")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b));
    } else if constexpr (F == 6) {
        asm volatile(WS16("v_swmmac_f32_16x16x32_f16", " index_key:0")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b), "v"(idx));
    } else if constexpr (F == 7) {
        asm volatile(WS16("v_swmmac_f32_16x16x32_fp8_fp8", " index_key:0")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b), "v"(idx));
    } else {
        asm volatile(WS16("v_swmmac_i32_16x16x32_iu8", " index_key:0 neg_lo:[1,1,0]")
                     : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3)
                     : "v"(a), "v"(b), "v"(idx));
    }
}

template <int F>
__global__ __launch_bounds__(1024) void reg_kernel_x(const unsigned * __restrict__ ab, unsigned * __restrict__ sink,
                                                     Rec * __restrict__ rec, int iters) {
    using A = typename XOps<F>::A;
    using B = typename XOps<F>::B;
    unsigned wa[XOps<F>::NA], wb[XOps<F>::NB];
    for (int k = 0; k < XOps<F>::NA; ++k) {
        wa[k] = ab[k];
    }
    for (int k = 0; k < XOps<F>::NB; ++k) {
        wb[k] = ab[8 + k];
    }
    A a = __builtin_bit_cast(A, wa);
    B b = __builtin_bit_cast(B, wb);
    int idx = (int) ab[16];
    asm volatile("" : "+v"(a), "+v"(b), "+v"(idx));  // operands stay in registers, not rebuilt from the buffer
    i8v c0 = {}, c1 = {}, c2 = {}, c3 = {};
    const uint32_t pl = mem::placement();
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        wmma16x<F>(c0, c1, c2, c3, a, b, idx);
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    unsigned x = 0;
    for (int k = 0; k < 8; ++k) {
        x ^= (unsigned) (c0[k] ^ c1[k] ^ c2[k] ^ c3[k]);
    }
    if (x == 0x7fc0ffeeu) {  // impossible for the small inputs; keeps the accumulators alive
        sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// MT-3: the remaining forms. Y: 9 sparse bf16, 10 sparse bf8, 11 dense bf8, 12 f16 into f16, 13 bf16 into bf16,
// 14 sparse f16 into f16. Accumulator types differ (16-bit results: 4 VGPRs), so each form has its own asm.
#define WY(OP, ACC, TAIL) OP " %" #ACC ", %4, %5, " TAIL "\n"
#define WY4(OP, T0, T1, T2, T3) WY(OP, 0, T0) WY(OP, 1, T1) WY(OP, 2, T2) WY(OP, 3, T3)
#define WY16D(OP) WY4(OP, "%0", "%1", "%2", "%3") WY4(OP, "%0", "%1", "%2", "%3") WY4(OP, "%0", "%1", "%2", "%3") WY4(OP, "%0", "%1", "%2", "%3")
#define WY16S(OP) WY4(OP, "%6", "%6", "%6", "%6") WY4(OP, "%6", "%6", "%6", "%6") WY4(OP, "%6", "%6", "%6", "%6") WY4(OP, "%6", "%6", "%6", "%6")
template <int Y>
struct YOps {
    static constexpr bool sparse = Y == 9 || Y == 10 || Y == 14;
    static constexpr int NA = (Y == 10 || Y == 11) ? 2 : 4;
    static constexpr int NB = sparse ? (Y == 10 ? 4 : 8) : NA;
    static constexpr int NC = Y >= 12 ? 4 : 8;
    using A = Vec<NA>;
    using B = Vec<NB>;
    using C = Vec<NC>;
};
template <int Y>
__device__ __forceinline__ void wmma16y(typename YOps<Y>::C & c0, typename YOps<Y>::C & c1, typename YOps<Y>::C & c2,
                                        typename YOps<Y>::C & c3, const typename YOps<Y>::A & a,
                                        const typename YOps<Y>::B & b, int idx) {
#define YASM(TEXT) asm volatile(TEXT : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b), "v"(idx))
    if constexpr (Y == 9) YASM(WY16S("v_swmmac_f32_16x16x32_bf16"));
    else if constexpr (Y == 10) YASM(WY16S("v_swmmac_f32_16x16x32_bf8_bf8"));
    else if constexpr (Y == 11) YASM(WY16D("v_wmma_f32_16x16x16_bf8_bf8"));
    else if constexpr (Y == 12) YASM(WY16D("v_wmma_f16_16x16x16_f16"));
    else if constexpr (Y == 13) YASM(WY16D("v_wmma_bf16_16x16x16_bf16"));
    else YASM(WY16S("v_swmmac_f16_16x16x32_f16"));
#undef YASM
}
template <int Y>
__global__ __launch_bounds__(1024) void reg_kernel_y(const unsigned * __restrict__ ab, unsigned * __restrict__ sink,
                                                     Rec * __restrict__ rec, int iters) {
    using A = typename YOps<Y>::A;
    using B = typename YOps<Y>::B;
    using C = typename YOps<Y>::C;
    unsigned wa[YOps<Y>::NA], wb[YOps<Y>::NB];
    for (int k = 0; k < YOps<Y>::NA; ++k) {
        wa[k] = ab[k];
    }
    for (int k = 0; k < YOps<Y>::NB; ++k) {
        wb[k] = ab[8 + k];
    }
    A a = __builtin_bit_cast(A, wa);
    B b = __builtin_bit_cast(B, wb);
    int idx = (int) ab[16];
    asm volatile("" : "+v"(a), "+v"(b), "+v"(idx));
    C c0 = {}, c1 = {}, c2 = {}, c3 = {};
    const uint32_t pl = mem::placement();
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        wmma16y<Y>(c0, c1, c2, c3, a, b, idx);
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    unsigned x = 0;
    for (int k = 0; k < YOps<Y>::NC; ++k) {
        x ^= (unsigned) (c0[k] ^ c1[k] ^ c2[k] ^ c3[k]);
    }
    if (x == 0x7fc0ffeeu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Matrix load with operands from memory. F: 0 f16 (16 bytes per lane per fragment), 1 fp8 (8 bytes).
// A wave reads A at its running offset and B right behind it, the offset wraps inside the wave's region.
template <int F>
__global__ __launch_bounds__(1024) void mem_kernel(const char * __restrict__ buf, unsigned * __restrict__ sink,
                                                   Rec * __restrict__ rec, int iters, unsigned region_bytes) {
    constexpr unsigned LB = F == 0 ? 16 : 8;  // bytes per lane per fragment
    constexpr unsigned STEP = 2 * 32 * LB;    // A and B fragments of one iteration
    const int lane = threadIdx.x & 31;
    const char * base = buf + (size_t) wave_index() * region_bytes + lane * LB;
    const unsigned mask = region_bytes - 1;
    f8v c = {};
    const uint32_t pl = mem::placement();
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
#pragma unroll 4
    for (int it = 0; it < iters; ++it) {
        const unsigned off = ((unsigned) it * STEP) & mask;
        if constexpr (F == 0) {
            const uint4 va = *(const uint4 *) (base + off);
            const uint4 vb = *(const uint4 *) (base + off + 32 * LB);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(h8v, va),
                                                                 __builtin_bit_cast(h8v, vb), c);
        } else {
            const uint2 va = *(const uint2 *) (base + off);
            const uint2 vb = *(const uint2 *) (base + off + 32 * LB);
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(i2v, va),
                                                                     __builtin_bit_cast(i2v, vb), c);
        }
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    unsigned x = 0;
    for (int k = 0; k < 8; ++k) {
        x ^= __float_as_uint(c[k]);
    }
    if (x == 0x7fc0ffeeu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
    }
    if (lane == 0) {
        rec[wave_index()] = {pl, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// FMA load: 16 independent chains per lane, the two factors kept in registers. The accumulate form x += k0 * k1
// lets clang pair the FMAs as v_dual_fmac_f32 (two per lane per cycle); x = x * k0 + k1 compiled to unpaired
// v_fma_f32 and loaded the card half as much.
__global__ __launch_bounds__(1024) void fma_kernel(const float * __restrict__ k, float * __restrict__ sink,
                                                   Rec * __restrict__ rec, int iters) {
    float k0 = k[0], k1 = k[1];
    asm volatile("" : "+v"(k0), "+v"(k1));
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = 0.5f + (float) ((threadIdx.x + i) & 63) * 1e-3f;
    }
    const uint32_t pl = mem::placement();
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
#pragma unroll 4
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            x[i] = fmaf(k0, k1, x[i]);
        }
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (s == -12345.0f) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Streaming read: grid-stride 16-byte loads over the whole buffer, `passes` times.
__global__ __launch_bounds__(1024) void stream_kernel(const uint4 * __restrict__ src, size_t n,
                                                      unsigned * __restrict__ sink, Rec * __restrict__ rec,
                                                      int passes) {
    unsigned acc = 0;
    const uint64_t cy0 = wave_cycles();
    const uint64_t rt0 = mem::realtime();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
            const uint4 v = src[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    const uint64_t cy1 = wave_cycles();
    const uint64_t rt1 = mem::realtime();
    if (acc == 0xdeadbeefu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {0, cycle_diff(cy0, cy1), rt0, rt1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host side.

struct Series {
    double mn = 0, med = 0, mx = 0;
    bool ok = false;
};

static Series series_of(std::vector<double> v) {
    Series s;
    if (v.empty()) {
        return s;
    }
    const Stats st = stats_of(v);
    s.mn = st.min;
    s.med = st.median;
    s.mx = st.max;
    s.ok = true;
    return s;
}

static void load_card_env() {
    if (std::getenv("DRM_DEVICE") && std::getenv("HWMON")) {
        return;
    }
    std::ifstream f("runs/latest/card.env");
    std::string line;
    while (std::getline(f, line)) {
        for (const char * key : {"DRM_DEVICE", "HWMON"}) {
            const std::string k = std::string(key) + "=";
            if (line.compare(0, k.size(), k) == 0) {
                setenv(key, line.c_str() + k.size(), 0);
            }
        }
    }
    if (!std::getenv("DRM_DEVICE") || !std::getenv("HWMON")) {
        std::printf("WARN DRM_DEVICE/HWMON not found (env or runs/latest/card.env): clocks, power and hotspot not measured\n");
    }
}

// Hard stop at 95 C (exit 4); otherwise every configuration starts below 60 C: measured, a 2-second burst of
// whole-card matrix work lifts the hotspot by about 30 C, and it falls back within a second or two.
static void thermal_stop(double t) {
    mem::hottest() = std::max(mem::hottest(), t);
    std::printf("THERMAL hotspot %.0f C is at or above 95 C: stopping at once\n", t);
    mem::report_hottest();
    std::fflush(stdout);
    std::_Exit(4);
}

static void gate() {
    const double t = mem::hotspot_c();
    if (t >= 95.0) {
        thermal_stop(t);
    }
    const auto t0 = std::chrono::steady_clock::now();
    double now = t;
    while ((now = mem::hotspot_c()) >= 60.0 &&
           std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (t >= 60.0) {
        std::printf("THERMAL paused %.1f s: %.0f C -> %.0f C\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), t, now);
    }
}

// Polls pp_dpm_sclk (the '*' line), power1_average and temp2_input every 50 ms from a host thread.
class Sampler {
  public:
    void start() {
        run_ = true;
        th_ = std::thread([this] {
            const char * drm = std::getenv("DRM_DEVICE");
            const char * hw = std::getenv("HWMON");
            while (run_) {
                if (drm) {
                    const int m = std::atoi(mem::star_line(std::string(drm) + "/pp_dpm_sclk").c_str());
                    if (m > 0) {
                        sclk.push_back(m);
                    }
                }
                if (hw) {
                    const double p = mem::read_num(std::string(hw) + "/power1_average", 1e-6);
                    const double t = mem::read_num(std::string(hw) + "/temp2_input", 1e-3);
                    if (p >= 0) {
                        power.push_back(p);
                    }
                    if (t >= 0) {
                        temp.push_back(t);
                        if (t >= 95.0) {
                            thermal_stop(t);
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
    void stop() {
        run_ = false;
        th_.join();
        if (!temp.empty()) {
            mem::hottest() = std::max(mem::hottest(), *std::max_element(temp.begin(), temp.end()));
        }
    }
    std::vector<double> sclk, power, temp;

  private:
    std::atomic<bool> run_{false};
    std::thread th_;
};

struct Place {
    int distinct = 0, minw = 0, maxw = 0;
};

static Place place_of(const std::vector<Rec> & r) {
    std::map<uint32_t, int> m;
    for (const Rec & x : r) {
        ++m[mem::pl_simd((uint32_t) x.pl)];
    }
    Place p;
    p.distinct = (int) m.size();
    p.minw = 1 << 30;
    for (auto & kv : m) {
        p.minw = std::min(p.minw, kv.second);
        p.maxw = std::max(p.maxw, kv.second);
    }
    return p;
}

struct Meas {
    std::vector<double> ms;
    double ms_med = 0;
    double clk_mhz = 0;  // median over waves of cycles / (REALTIME ticks / 1e8)
    Place place;         // worst case over the timed launches
    Series sclk, power, temp;
};

static double time_one(const std::function<void()> & launch) {
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    CHECK(hipEventRecord(t0));
    launch();
    CHECK(hipEventRecord(t1));
    CHECK(hipEventSynchronize(t1));
    float m;
    CHECK(hipEventElapsedTime(&m, t0, t1));
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    return m;
}

// Scales `iters` so that one launch lasts about target_ms: short launches first (a launch of a few ms is too short
// to extrapolate from, so those grow tenfold), then one linear step.
static int calibrate(const std::function<void(int)> & launch, int iters0, double target_ms) {
    double it = iters0, ms = 0;
    for (int r = 0; r < 5; ++r) {
        ms = time_one([&] { launch((int) it); });
        if (ms >= 8.0) {
            break;
        }
        it *= std::min(100.0, 12.0 / std::max(ms, 0.01));
    }
    return (int) std::min(2e9, std::max(64.0, it * target_ms / ms));
}

// 1 warm launch, then 3 timed ones with the sampler running. Records come from every timed launch.
static Meas measure(const std::function<void()> & launch, int nwaves, Rec * d_rec) {
    Meas m;
    launch();
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    Sampler s;
    std::vector<double> clk;
    s.start();
    for (int rep = 0; rep < 3; ++rep) {
        m.ms.push_back(time_one(launch));
        std::vector<Rec> r(nwaves);
        CHECK(hipMemcpy(r.data(), d_rec, nwaves * sizeof(Rec), hipMemcpyDeviceToHost));
        for (const Rec & x : r) {
            if (x.rt1 > x.rt0) {
                clk.push_back((double) x.cyc / ((double) (x.rt1 - x.rt0) / 1e8) / 1e6);
            }
        }
        const Place p = place_of(r);
        if (rep == 0) {
            m.place = p;
        } else {
            m.place.distinct = std::min(m.place.distinct, p.distinct);
            m.place.minw = std::min(m.place.minw, p.minw);
            m.place.maxw = std::max(m.place.maxw, p.maxw);
        }
    }
    s.stop();
    m.ms_med = stats_of(m.ms).median;
    m.clk_mhz = clk.empty() ? 0 : stats_of(clk).median;
    m.sclk = series_of(s.sclk);
    m.power = series_of(s.power);
    m.temp = series_of(s.temp);
    return m;
}

static void print_env(const Meas & m) {
    std::printf(" clock_MHz=%.0f sclk_MHz(min/med/max)=%.0f/%.0f/%.0f power_W(min/max)=%.0f/%.0f hotspot_C(min/max)=%.0f/%.0f",
                m.clk_mhz, m.sclk.mn, m.sclk.med, m.sclk.mx, m.power.mn, m.power.mx, m.temp.mn, m.temp.mx);
}

static void print_place(const Meas & m) {
    std::printf(" simds=%d waves_per_simd=%d..%d", m.place.distinct, m.place.minw, m.place.maxw);
}

static int finish(const char * probe, bool pass, const std::string & text) {
    mem::report_hottest();
    verdict(probe, pass ? "PASS" : "FAIL", text.c_str());
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
static void launch_reg(int F, int W, int iters, const unsigned * ab, unsigned * sink, Rec * rec) {
    const dim3 g(64), b(64 * W);
    constexpr unsigned lds = 40 * 1024;  // unused; requested so that only one group fits on a CU
    switch (F) {
        case 0: hipLaunchKernelGGL(reg_kernel<0>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 1: hipLaunchKernelGGL(reg_kernel<1>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 2: hipLaunchKernelGGL(reg_kernel<2>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 4: hipLaunchKernelGGL(reg_kernel_x<4>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 5: hipLaunchKernelGGL(reg_kernel_x<5>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 6: hipLaunchKernelGGL(reg_kernel_x<6>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 7: hipLaunchKernelGGL(reg_kernel_x<7>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 8: hipLaunchKernelGGL(reg_kernel_x<8>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 9: hipLaunchKernelGGL(reg_kernel_y<9>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 10: hipLaunchKernelGGL(reg_kernel_y<10>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 11: hipLaunchKernelGGL(reg_kernel_y<11>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 12: hipLaunchKernelGGL(reg_kernel_y<12>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 13: hipLaunchKernelGGL(reg_kernel_y<13>, g, b, lds, 0, ab, sink, rec, iters); break;
        case 14: hipLaunchKernelGGL(reg_kernel_y<14>, g, b, lds, 0, ab, sink, rec, iters); break;
        default: hipLaunchKernelGGL(reg_kernel<3>, g, b, lds, 0, ab, sink, rec, iters); break;
    }
    CHECK(hipGetLastError());
}

static const char * opt_of(int argc, char ** argv, const char * f, const char * def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

// Leftovers C: one burst (see the usage text). Cycles per instruction from matrix-timing items 2 and 3.
static int run_burst(const std::string & form, int W, double target_ms, const std::string & tag) {
    struct FormDef {
        const char * name;
        int F;
        double cyc, ops;
        bool eight;
    };
    const FormDef defs[] = {{"f16", 0, 16, 8192, false},     {"bf16", 1, 16, 8192, false},     {"sbf16", 9, 16, 16384, false},
                            {"sbf8", 10, 8, 16384, true},    {"bf8", 11, 8, 8192, true},       {"f16f16", 12, 16, 8192, false},
                            {"bf16bf16", 13, 16, 8192, false}, {"sf16f16", 14, 16, 16384, false}};
    const FormDef * d = nullptr;
    for (const FormDef & x : defs) {
        if (form == x.name) {
            d = &x;
        }
    }
    if (!d || W < 1 || W > 16 || target_ms <= 0 || target_ms > 200 || (d->eight && target_ms > 100)) {
        std::fprintf(stderr, "burst: unknown form or a length beyond the thermal rule (0.2 s; 0.1 s for 8-bit forms)\n");
        return 2;
    }
    const double t0c = mem::hotspot_c();
    if (t0c >= 55.0) {
        std::printf("NOT STARTED: hotspot %.0f C at the start (limit 55 C)\n", t0c);
        return 7;
    }
    unsigned h[32];
    for (int k = 0; k < 32; ++k) {
        h[k] = d->eight ? 0x24242424u : 0x24002400u;
    }
    h[16] = 0x44444444u;
    unsigned *d_ab, *d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_ab, sizeof h));
    CHECK(hipMemcpy(d_ab, h, sizeof h, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_sink, 64 * 1024 * 4));
    CHECK(hipMalloc(&d_rec, MAXREC * sizeof(Rec)));
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    const int waves = 128 * W;
    // iterations for target_ms at 3.2 GHz: each wave issues 16 instructions per iteration, W waves share a SIMD
    const int iters = (int) std::max(16.0, target_ms * 1e-3 * 3.2e9 / (d->cyc * 16.0 * W));
    launch_reg(d->F, W, 16, d_ab, d_sink, d_rec);  // warm-up: code load, a few microseconds of work
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    Sampler smp;
    smp.start();
    const double ms = time_one([&] { launch_reg(d->F, W, iters, d_ab, d_sink, d_rec); });
    smp.stop();
    std::vector<Rec> r(waves);
    CHECK(hipMemcpy(r.data(), d_rec, waves * sizeof(Rec), hipMemcpyDeviceToHost));
    std::vector<double> clk;
    uint64_t rt_lo = ~0ull, rt_hi = 0;
    for (const Rec & x : r) {
        if (x.rt1 > x.rt0) {
            clk.push_back((double) x.cyc / ((double) (x.rt1 - x.rt0) / 1e8) / 1e6);
            rt_lo = std::min(rt_lo, x.rt0);
            rt_hi = std::max(rt_hi, x.rt1);
        }
    }
    const Place pl = place_of(r);
    const double clk_mhz = clk.empty() ? 0 : stats_of(clk).median;
    const double instr = (double) waves * iters * 16;
    const double rate_ev = instr / (ms * 1e-3), span_ms = (double) (rt_hi - rt_lo) / 1e5;
    const double rate_rt = span_ms > 0 ? instr / (span_ms * 1e-3) : 0;
    const double theory = NSIMD * clk_mhz * 1e6 / d->cyc;
    const Series sc = series_of(smp.sclk), pw = series_of(smp.power), tp = series_of(smp.temp);
    std::printf("BURST tag=%s form=%s W=%d target_ms=%.0f iters=%d event_ms=%.3f span_ms=%.3f Ginstr_s_event=%.3f "
                "Ginstr_s_span=%.3f clock_MHz=%.1f theory_Ginstr_s=%.3f ratio_event=%.4f ratio_span=%.4f TOPS_event=%.1f "
                "cycles=%.0f simds=%d waves_per_simd=%d..%d sclk_med=%.0f power_W_max=%.0f hotspot_C_start=%.0f "
                "hotspot_C_max=%.0f\n",
                tag.c_str(), d->name, W, target_ms, iters, ms, span_ms, rate_ev / 1e9, rate_rt / 1e9, clk_mhz, theory / 1e9,
                rate_ev / theory, rate_rt / theory, rate_ev * d->ops / 1e12, d->cyc, pl.distinct, pl.minw, pl.maxw, sc.med,
                pw.mx, t0c, tp.mx);
    mem::report_hottest();
    const bool placed = pl.distinct == NSIMD && pl.minw == W && pl.maxw == W;
    verdict(("mat_card_burst:" + form).c_str(), placed && clk_mhz > 0 ? "PASS" : "FAIL",
            placed ? "burst recorded, every SIMD with W waves" : "placement wrong or no clock");
    return placed && clk_mhz > 0 ? 0 : 1;
}

static int run_reg(int F, const char * fname, bool brk, double target_ms) {
    const double cyc_true = (F < 2 || F == 6) ? 16 : 8;
    const double ops = (F == 5 || F >= 6) ? 16384 : 8192;  // dense-equivalent operations per instruction (K = 32: twice)
    const double cyc = brk ? cyc_true / 2 : cyc_true;  // --break: claim half the cycles
    const std::string probe = std::string("mat_card_reg:") + fname;
    unsigned h[32];
    for (int k = 0; k < 32; ++k) {
        h[k] = (F < 2 || F == 6) ? 0x24002400u : 0x24242424u;
    }
    h[16] = 0x44444444u;  // sparse index: elements 0 and 1 of every four (read by the sparse kernels only)
    unsigned *d_ab, *d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_ab, sizeof h));
    CHECK(hipMemcpy(d_ab, h, sizeof h, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_sink, 64 * 1024 * 4));
    CHECK(hipMalloc(&d_rec, MAXREC * sizeof(Rec)));
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    const int Ws[] = {1, 2, 4, 8, 16};
    bool placement_ok = true;
    double best_rate = 0, best_ratio = 0;
    int best_W = 0;
    for (int W : Ws) {
        gate();
        const int waves = 128 * W;
        const int it0 = (int) (0.03 * 2.5e9 / (W * 16.0 * cyc_true));
        const int iters = calibrate([&](int n) { launch_reg(F, W, n, d_ab, d_sink, d_rec); }, it0, target_ms);
        const Meas m = measure([&] { launch_reg(F, W, iters, d_ab, d_sink, d_rec); }, waves, d_rec);
        const double rate = (double) waves * iters * 16 / (m.ms_med * 1e-3);
        const double theory = NSIMD * m.clk_mhz * 1e6 / cyc;
        const double ratio = rate / theory;
        std::printf("RESULT %s W=%d waves=%d iters=%d ms=%.1f Ginstr_s=%.1f TFLOPS=%.1f theory_Ginstr_s=%.1f ratio=%.3f", probe.c_str(),
                    W, waves, iters, m.ms_med, rate / 1e9, rate * ops / 1e12, theory / 1e9, ratio);
        print_place(m);
        print_env(m);
        std::printf("\n");
        std::fflush(stdout);
        if (W >= 2 && !(m.place.distinct == NSIMD && m.place.minw == W && m.place.maxw == W)) {
            placement_ok = false;
        }
        if (W == 1 && !(m.place.distinct == NSIMD && m.place.minw == 1 && m.place.maxw == 1)) {
            std::printf("NOTE %s W=1 with 64 groups of 64 threads did not give one wave per SIMD (simds=%d, waves per SIMD %d..%d)\n",
                        probe.c_str(), m.place.distinct, m.place.minw, m.place.maxw);
        }
        if (rate > best_rate) {
            best_rate = rate;
            best_ratio = ratio;
            best_W = W;
        }
    }
    std::printf("RESULT %s highest rate at W=%d: %.1f G instr/s, %.1f TFLOPS, ratio to theory %.3f (expected cycles %.0f)\n",
                probe.c_str(), best_W, best_rate / 1e9, best_rate * ops / 1e12, best_ratio, cyc);
    if (F >= 6) {
        std::printf("RESULT %s stored work (half of the dense-equivalent): %.1f T ops/s\n", probe.c_str(),
                    best_rate * ops / 2e12);
    }
    const bool ratio_ok = best_ratio >= 0.80 && best_ratio <= 1.02;
    char buf[300];
    std::snprintf(buf, sizeof buf, "placement W>=2 %s (128 SIMDs, W waves each); best W=%d ratio %.3f %s [0.80, 1.02]",
                  placement_ok ? "ok" : "BAD", best_W, best_ratio, ratio_ok ? "inside" : "outside");
    return finish(probe.c_str(), placement_ok && ratio_ok, buf);
}

static void launch_mem(int F, int W, int iters, const char * buf, unsigned * sink, Rec * rec, unsigned region) {
    const dim3 g(64), b(64 * W);
    if (F == 0) {
        hipLaunchKernelGGL(mem_kernel<0>, g, b, 40 * 1024, 0, buf, sink, rec, iters, region);
    } else {
        hipLaunchKernelGGL(mem_kernel<1>, g, b, 40 * 1024, 0, buf, sink, rec, iters, region);
    }
    CHECK(hipGetLastError());
}

static int run_mem(int F, const char * fname, double target_ms) {
    const double cyc = F == 0 ? 16 : 8;
    const double bytes_iter = F == 0 ? 1024 : 512;  // per wave per iteration: A and B fragments
    const std::string probe = std::string("mat_card_mem:") + fname;
    const size_t CACHE_BYTES = 32ull << 20, DRAM_BYTES = 1ull << 30;
    char *d_cache, *d_dram;
    unsigned* d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_cache, CACHE_BYTES));
    CHECK(hipMalloc(&d_dram, DRAM_BYTES));
    CHECK(hipMemset(d_cache, 0x24, CACHE_BYTES));  // small values, like the register runs
    CHECK(hipMemset(d_dram, 0x24, DRAM_BYTES));
    CHECK(hipMalloc(&d_sink, 64 * 1024 * 4));
    CHECK(hipMalloc(&d_rec, MAXREC * sizeof(Rec)));
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    const int Ws[] = {1, 4, 16};
    bool below = true;
    for (const char * src : {"cache", "dram"}) {
        const bool cache = src[0] == 'c';
        for (int W : Ws) {
            gate();
            const int waves = 128 * W;
            // cache: up to 64 KiB per wave, less when 32 MiB / waves is smaller; dram: the wave's share of 1 GiB
            const size_t total = cache ? CACHE_BYTES : DRAM_BYTES;
            const unsigned region = (unsigned) std::min<size_t>(cache ? 65536 : (size_t) 1 << 30, total / waves);
            const char * buf = cache ? d_cache : d_dram;
            const int iters = calibrate([&](int n) { launch_mem(F, W, n, buf, d_sink, d_rec, region); }, 100000, target_ms);
            const Meas m = measure([&] { launch_mem(F, W, iters, buf, d_sink, d_rec, region); }, waves, d_rec);
            const double s = m.ms_med * 1e-3;
            const double rate = (double) waves * iters / s;
            const double theory = NSIMD * m.clk_mhz * 1e6 / cyc;
            std::printf("RESULT %s src=%s W=%d waves=%d region_KiB=%u iters=%d ms=%.1f Ginstr_s=%.1f TFLOPS=%.1f reg_theory_Ginstr_s=%.1f ratio=%.3f GBps_read=%.1f",
                        probe.c_str(), src, W, waves, region >> 10, iters, m.ms_med, rate / 1e9, rate * 8192 / 1e12,
                        theory / 1e9, rate / theory, rate * bytes_iter / 1e9);
            print_place(m);
            print_env(m);
            std::printf("\n");
            std::fflush(stdout);
            if (!(rate < theory)) {
                below = false;
            }
        }
    }
    return finish(probe.c_str(), below,
                  below ? "every measured matrix rate is below the register-mode theoretical rate"
                        : "a measured matrix rate reached the register-mode theoretical rate");
}

static int run_fma(double target_ms) {
    const int W = 8, waves = 128 * W;
    float h[2] = {0.9999f, 1e-4f};
    float *d_k, *d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_k, sizeof h));
    CHECK(hipMemcpy(d_k, h, sizeof h, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_sink, 64 * 1024 * 4));
    CHECK(hipMalloc(&d_rec, MAXREC * sizeof(Rec)));
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    auto launch = [&](int n) {
        hipLaunchKernelGGL(fma_kernel, dim3(64), dim3(64 * W), 40 * 1024, 0, d_k, d_sink, d_rec, n);
        CHECK(hipGetLastError());
    };
    gate();
    const int iters = calibrate(launch, 1000000, target_ms);
    const Meas m = measure([&] { launch(iters); }, waves, d_rec);
    const double rate = (double) waves * 32 * 16 * iters / (m.ms_med * 1e-3);
    const double theory = NSIMD * 64.0 * m.clk_mhz * 1e6;
    std::printf("RESULT mat_card_fma W=%d waves=%d iters=%d ms=%.1f GFMA_s=%.1f theory_GFMA_s=%.1f ratio=%.3f", W, waves,
                iters, m.ms_med, rate / 1e9, theory / 1e9, rate / theory);
    print_place(m);
    print_env(m);
    std::printf("\n");
    const bool ok = m.place.distinct == NSIMD && m.place.minw == W && m.place.maxw == W;
    char buf[200];
    std::snprintf(buf, sizeof buf, "all launches completed; placement %d SIMDs, waves per SIMD %d..%d", m.place.distinct,
                  m.place.minw, m.place.maxw);
    return finish("mat_card_fma", ok, buf);
}

static int run_stream(double target_ms) {
    const size_t bytes = 1ull << 30, n = bytes / 16;
    const int blocks = 64 * 4, threads = 1024, waves = blocks * threads / 32;
    uint4 * d_src;
    unsigned * d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_src, bytes));
    CHECK(hipMemset(d_src, 0x24, bytes));
    CHECK(hipMalloc(&d_sink, (size_t) blocks * threads * 4));
    CHECK(hipMalloc(&d_rec, MAXREC * sizeof(Rec)));
    CHECK(hipMemset(d_rec, 0, MAXREC * sizeof(Rec)));
    auto launch = [&](int p) {
        hipLaunchKernelGGL(stream_kernel, dim3(blocks), dim3(threads), 0, 0, d_src, n, d_sink, d_rec, p);
        CHECK(hipGetLastError());
    };
    gate();
    const int passes = calibrate(launch, 4, target_ms);
    const Meas m = measure([&] { launch(passes); }, waves, d_rec);
    const double gbps = (double) passes * bytes / (m.ms_med * 1e-3) / 1e9;
    std::printf("RESULT mat_card_stream passes=%d ms=%.1f GBps=%.1f", passes, m.ms_med, gbps);
    print_env(m);
    std::printf("\n");
    char buf[200];
    std::snprintf(buf, sizeof buf, "%.1f GB/s against the 400..700 GB/s window", gbps);
    return finish("mat_card_stream", gbps >= 400 && gbps <= 700, buf);
}

int main(int argc, char ** argv) {
    require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: mat_card <reg:f16|reg:bf16|reg:fp8|reg:iu8|reg:iu4|reg:iu4k32|reg:sf16|reg:sfp8|reg:siu8|mem:f16|mem:fp8|fma|stream> [--break] [--short]\n");
        return 2;
    }
    load_card_env();
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    const bool shrt = mem::has_flag(argc, argv, "--short");
    std::printf("DATA mat_card mode=%s break=%d short=%d\n", mode.c_str(), brk, shrt);
    mem::clocks("start");
    if (mode == "burst" && argc >= 3) {
        return run_burst(argv[2], std::atoi(opt_of(argc, argv, "--w", "1")), std::atof(opt_of(argc, argv, "--ms", "100")),
                         opt_of(argc, argv, "--tag", "-"));
    }
    // 200 ms per timed launch keeps a whole-card configuration to about 1 s of load (thermal rule)
    if (mode == "reg:f16") return run_reg(0, "f16", brk, shrt ? 100 : 200);
    if (mode == "reg:bf16") return run_reg(1, "bf16", brk, shrt ? 100 : 200);
    if (mode == "reg:fp8") return run_reg(2, "fp8", brk, shrt ? 100 : 200);
    if (mode == "reg:iu8") return run_reg(3, "iu8", brk, shrt ? 100 : 200);
    if (mode == "reg:iu4") return run_reg(4, "iu4", brk, shrt ? 100 : 200);
    if (mode == "reg:iu4k32") return run_reg(5, "iu4k32", brk, shrt ? 100 : 200);
    if (mode == "reg:sf16") return run_reg(6, "sf16", brk, shrt ? 100 : 200);
    if (mode == "reg:sfp8") return run_reg(7, "sfp8", brk, shrt ? 100 : 200);
    if (mode == "reg:siu8") return run_reg(8, "siu8", brk, shrt ? 100 : 200);
    if (mode == "mem:f16") return run_mem(0, "f16", shrt ? 100 : 200);
    if (mode == "mem:fp8") return run_mem(1, "fp8", shrt ? 100 : 200);
    if (mode == "fma") return run_fma(shrt ? 100 : 200);
    if (mode == "stream") return run_stream(shrt ? 100 : 200);
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
