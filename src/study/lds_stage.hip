// Lane 3, item 3: matrix instructions fed from LDS against fed from global loads (L0 hits) against registers only.
//
// Usage: lds_stage <calib|feed> [--break]      (built twice: WGP mode, the HIP default, and -mcumode)
//   feed  a work-group of 1, 2, 4 or 8 waves; every wave runs a pipelined loop of 16 x 16 x 16 matrix instructions,
//         four operand sets in rotation: wait for the oldest set, multiply, reload the set used one instruction
//         earlier. Each instruction gets fresh A and B from:
//           reg      no loads (lane 4: 16 cycles FP16, 8 cycles FP8)
//           lds      ds_load_b128 A and B (FP16), ds_load_b64 A and B (FP8), lane-sequential, no bank conflicts
//           global   global_load_b128 A + global_load_tr_b128 B (FP16), global_load_b64 + global_load_tr_b64
//                    (FP8), from a 4 KiB region that stays in L0
//         Cycles per matrix instruction per wave, the waves' SIMDs and CUs. --break expects the FP16 register-only
//         loop at 32 cycles, so the calibration verdict must FAIL.
#include "lds_common.h"

#include <map>

#ifndef USE_CU_KEY
#define USE_CU_KEY 0
#endif

typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));

enum Var { REG16, LDS16, GLB16, REG8, LDS8, GLB8, NVAR };
static const char * var_name[NVAR] = {"f16_reg", "f16_lds_b128", "f16_global_b128+tr_b128", "fp8_reg", "fp8_lds_b64",
                                      "fp8_global_b64+tr_b64"};

#define WM16(c, a, b) "v_wmma_f32_16x16x16_f16 %" #c ", %" #a ", %" #b ", %" #c "\n\t"
#define WM8(c, a, b) "v_wmma_f32_16x16x16_fp8_fp8 %" #c ", %" #a ", %" #b ", %" #c "\n\t"
#define DSL(ins, r, off) ins " %" #r ", %12 offset:" #off "\n\t"
#define GLA(ins, r, off) ins " %" #r ", %12, %13 offset:" #off "\n\t"
#define GLB(ins, r, off) ins " %" #r ", %14, %13 offset:" #off "\n\t"

// One block of four stages. Stage s multiplies set s and reloads set (s + 3) % 4, whose last use was one matrix
// instruction earlier. Before the wait three sets are in flight (6 loads); waiting down to 4 completes set s.
#define LDS16_BLOCK                                                                                            \
    "s_wait_dscnt 4\n\t" WM16(0, 4, 8) DSL("ds_load_b128", 7, 3072) DSL("ds_load_b128", 11, 3584)              \
    "s_wait_dscnt 4\n\t" WM16(1, 5, 9) DSL("ds_load_b128", 4, 0) DSL("ds_load_b128", 8, 512)                   \
    "s_wait_dscnt 4\n\t" WM16(2, 6, 10) DSL("ds_load_b128", 5, 1024) DSL("ds_load_b128", 9, 1536)              \
    "s_wait_dscnt 4\n\t" WM16(3, 7, 11) DSL("ds_load_b128", 6, 2048) DSL("ds_load_b128", 10, 2560)
#define LDS8_BLOCK                                                                                             \
    "s_wait_dscnt 4\n\t" WM8(0, 4, 8) DSL("ds_load_b64", 7, 1536) DSL("ds_load_b64", 11, 1792)                 \
    "s_wait_dscnt 4\n\t" WM8(1, 5, 9) DSL("ds_load_b64", 4, 0) DSL("ds_load_b64", 8, 256)                      \
    "s_wait_dscnt 4\n\t" WM8(2, 6, 10) DSL("ds_load_b64", 5, 512) DSL("ds_load_b64", 9, 768)                   \
    "s_wait_dscnt 4\n\t" WM8(3, 7, 11) DSL("ds_load_b64", 6, 1024) DSL("ds_load_b64", 10, 1280)
#define GLB16_BLOCK                                                                                                \
    "s_wait_loadcnt 4\n\t" WM16(0, 4, 8) GLA("global_load_b128", 7, 3072) GLB("global_load_tr_b128", 11, 3584)     \
    "s_wait_loadcnt 4\n\t" WM16(1, 5, 9) GLA("global_load_b128", 4, 0) GLB("global_load_tr_b128", 8, 512)          \
    "s_wait_loadcnt 4\n\t" WM16(2, 6, 10) GLA("global_load_b128", 5, 1024) GLB("global_load_tr_b128", 9, 1536)     \
    "s_wait_loadcnt 4\n\t" WM16(3, 7, 11) GLA("global_load_b128", 6, 2048) GLB("global_load_tr_b128", 10, 2560)
#define GLB8_BLOCK                                                                                               \
    "s_wait_loadcnt 4\n\t" WM8(0, 4, 8) GLA("global_load_b64", 7, 1536) GLB("global_load_tr_b64", 11, 1792)      \
    "s_wait_loadcnt 4\n\t" WM8(1, 5, 9) GLA("global_load_b64", 4, 0) GLB("global_load_tr_b64", 8, 256)           \
    "s_wait_loadcnt 4\n\t" WM8(2, 6, 10) GLA("global_load_b64", 5, 512) GLB("global_load_tr_b64", 9, 768)        \
    "s_wait_loadcnt 4\n\t" WM8(3, 7, 11) GLA("global_load_b64", 6, 1024) GLB("global_load_tr_b64", 10, 1280)
#define REG16_BLOCK WM16(0, 4, 8) WM16(1, 5, 9) WM16(2, 6, 10) WM16(3, 7, 11)
#define REG8_BLOCK WM8(0, 4, 8) WM8(1, 5, 9) WM8(2, 6, 10) WM8(3, 7, 11)

#define OPS16 "+v"(c[0]), "+v"(c[1]), "+v"(c[2]), "+v"(c[3]), "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3]), \
              "+v"(b[0]), "+v"(b[1]), "+v"(b[2]), "+v"(b[3])
#define INS "v"(la), "s"(gbase), "v"(lt)

constexpr int ITERS = 256, UNROLL = 4, REPS = 6;

__device__ __forceinline__ uint32_t tr128_off(int s) { return (uint32_t) (((8 * (s / 16) + s % 8) * 16 + 8 * ((s / 8) % 2)) * 2); }
__device__ __forceinline__ uint32_t tr64_off(int s) { return (uint32_t) ((8 * (s / 16) + 4 * ((s / 8) % 2) + s % 4) * 16 + 8 * ((s / 4) % 2)); }

extern __shared__ uint32_t lds_mem[];

template <int V>
__global__ __launch_bounds__(512) void k_feed(const char * gbase, uint64_t * out) {
    for (uint32_t i = threadIdx.x; i < 4096 / 4; i += blockDim.x) {
        lds_mem[i] = 0x3c003c00u;  // FP16 1.0 pairs (FP8: 0x00, 0x3c: 0 and 1.0)
    }
    __syncthreads();
    constexpr bool f16 = V < REG8;
    using T = typename std::conditional<f16, u4, u2>::type;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t la = lane * (f16 ? 16 : 8), lt = f16 ? tr128_off(lane) : tr64_off(lane);
    v8f c[4];
    T a[4], b[4];
    for (int i = 0; i < 4; ++i) {
        c[i] = v8f{};
        a[i] = (T) 0x3c003c00u;
        b[i] = (T) 0x3c003c00u;
    }
    // prologue: sets 0, 1 and 2 in flight
    if constexpr (V == LDS16) {
        asm volatile(DSL("ds_load_b128", 4, 0) DSL("ds_load_b128", 8, 512) DSL("ds_load_b128", 5, 1024) DSL("ds_load_b128", 9, 1536)
                     DSL("ds_load_b128", 6, 2048) DSL("ds_load_b128", 10, 2560) : OPS16 : INS : "memory");
    } else if constexpr (V == LDS8) {
        asm volatile(DSL("ds_load_b64", 4, 0) DSL("ds_load_b64", 8, 256) DSL("ds_load_b64", 5, 512) DSL("ds_load_b64", 9, 768)
                     DSL("ds_load_b64", 6, 1024) DSL("ds_load_b64", 10, 1280) : OPS16 : INS : "memory");
    } else if constexpr (V == GLB16) {
        asm volatile(GLA("global_load_b128", 4, 0) GLB("global_load_tr_b128", 8, 512) GLA("global_load_b128", 5, 1024)
                     GLB("global_load_tr_b128", 9, 1536) GLA("global_load_b128", 6, 2048) GLB("global_load_tr_b128", 10, 2560)
                     : OPS16 : INS : "memory");
    } else if constexpr (V == GLB8) {
        asm volatile(GLA("global_load_b64", 4, 0) GLB("global_load_tr_b64", 8, 256) GLA("global_load_b64", 5, 512)
                     GLB("global_load_tr_b64", 9, 768) GLA("global_load_b64", 6, 1024) GLB("global_load_tr_b64", 10, 1280)
                     : OPS16 : INS : "memory");
    }
    // one untimed repetition (caches, instruction cache), then all waves start together and run REPS repetitions;
    // the span on the REALTIME clock across all waves gives the shared rate
    for (int rep = 0; rep <= REPS; ++rep) {
        if (rep == 1) {
            __syncthreads();
        }
        const uint64_t r0 = mem::realtime(), t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                if constexpr (V == REG16) asm volatile(REG16_BLOCK : OPS16 : INS : "memory");
                else if constexpr (V == LDS16) asm volatile(LDS16_BLOCK : OPS16 : INS : "memory");
                else if constexpr (V == GLB16) asm volatile(GLB16_BLOCK : OPS16 : INS : "memory");
                else if constexpr (V == REG8) asm volatile(REG8_BLOCK : OPS16 : INS : "memory");
                else if constexpr (V == LDS8) asm volatile(LDS8_BLOCK : OPS16 : INS : "memory");
                else asm volatile(GLB8_BLOCK : OPS16 : INS : "memory");
            }
        }
        const uint64_t t1 = wave_cycles(), r1 = mem::realtime();
        const uint32_t w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        if (rep >= 1 && lane == 0) {
            if (rep == 1) {
                out[w * 4 + 0] = 0;
                out[w * 4 + 1] = r0;
            }
            out[w * 4 + 0] += cycle_diff(t0, t1);
            out[w * 4 + 2] = r1;
        }
    }
    asm volatile("s_wait_dscnt 0\n\ts_wait_loadcnt 0" ::: "memory");
    float s = 0;
    for (int i = 0; i < 4; ++i) {
        s += c[i][0];
    }
    if (lane == 0) {
        const uint32_t w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        out[w * 4 + 3] = mem::placement() | ((uint64_t) (s == 12345.0f) << 40);
    }
}

using FK = void (*)(const char *, uint64_t *);
static FK kern(int v) {
    switch (v) {
        case REG16: return k_feed<REG16>;
        case LDS16: return k_feed<LDS16>;
        case GLB16: return k_feed<GLB16>;
        case REG8: return k_feed<REG8>;
        case LDS8: return k_feed<LDS8>;
        default: return k_feed<GLB8>;
    }
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "feed";
    const bool brk = core::flag(argc, argv, "--break");
    const char * tag = USE_CU_KEY ? "cu_mode" : "wgp_mode";
    mem::clocks("start");
    if (mode == "calib") {
        lds::calibrate(brk);
    } else if (mode == "feed") {
        char * g;
        uint64_t * out;
        CHECK(hipMalloc(&g, 1 << 20));
        CHECK(hipMalloc(&out, 64 * 4 * 8));
        std::vector<uint16_t> ones(1 << 19, 0x3c00);
        CHECK(hipMemcpy(g, ones.data(), 1 << 20, hipMemcpyHostToDevice));
        double reg16 = 0, reg8 = 0;
        for (int v = 0; v < NVAR; ++v) {
            for (uint32_t nw : {1u, 2u, 4u, 8u, 16u}) {
                for (int l = 0; l < 3; ++l) {
                    hipLaunchKernelGGL(kern(v), dim3(1), dim3(32 * nw), 4096, 0, g, out);
                    CHECK(hipGetLastError());
                    CHECK(hipDeviceSynchronize());
                    std::vector<uint64_t> o(nw * 4);
                    CHECK(hipMemcpy(o.data(), out, o.size() * 8, hipMemcpyDeviceToHost));
                    std::map<uint32_t, int> simds, cus;
                    std::vector<double> per, clk;
                    uint64_t rmin = ~0ull, rmax = 0;
                    const double mm = (double) REPS * ITERS * UNROLL * 4;  // per wave
                    for (uint32_t w = 0; w < nw; ++w) {
                        per.push_back((double) o[w * 4] / mm);
                        clk.push_back((double) o[w * 4] / (double) (o[w * 4 + 2] - o[w * 4 + 1]));  // cycles per 10 ns
                        rmin = std::min(rmin, o[w * 4 + 1]);
                        rmax = std::max(rmax, o[w * 4 + 2]);
                        simds[mem::pl_simd((uint32_t) o[w * 4 + 3])]++;
                        cus[mem::pl_cu((uint32_t) o[w * 4 + 3])]++;
                    }
                    int mx = 0;
                    for (auto & kv : simds) mx = std::max(mx, kv.second);
                    const Stats s = stats_of(per), ck = stats_of(clk);
                    const double span = (double) (rmax - rmin) * ck.median;  // cycles from first start to last end
                    const double simd_mm = span / (mm * mx);                // cycles per matrix instruction, busiest SIMD
                    const double opb = v < REG8 ? 1024 : 512;               // A and B bytes per matrix instruction
                    std::printf("RESULT stage %s %s waves %u launch %d cus %zu simds %zu max_waves_per_simd %d cycles_per_mm_per_wave "
                                "%.2f min %.2f max %.2f per_simd_cycles_per_mm %.2f operand_bytes_per_cycle_per_simd %.1f clock_GHz %.2f\n",
                                tag, var_name[v], nw, l, cus.size(), simds.size(), mx, s.median, s.min, s.max, simd_mm,
                                (v == REG16 || v == REG8) ? 0.0 : opb / simd_mm, ck.median / 10.0);
                    if (nw == 1 && v == REG16 && l == 2) reg16 = s.median;
                    if (nw == 1 && v == REG8 && l == 2) reg8 = s.median;
                }
            }
        }
        const double want16 = brk ? 32.0 : 16.0;
        lds::check("stage_calib", std::fabs(reg16 - want16) < 0.6 && std::fabs(reg8 - 8.0) < 0.4,
                   "register-only loop: FP16 %.2f cycles (expected %.0f; lane 4: 16.1), FP8 %.2f (expected 8; lane 4: 8.1)", reg16,
                   want16, reg8);
        CHECK(hipFree(g));
        CHECK(hipFree(out));
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::clocks("end");
    return lds::finish();
}
