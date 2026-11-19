// Lane 6: cycles per instruction for the FP8/BF8 conversions and the dot products, and cycles per weight for each
// way of widening FP8 (E4M3) weights, on gfx1201 (items 1-5 and 7 of the lane).
//
// Method: lane 1's (src/study/core_common.h): one wave alone on its SIMD, a timed loop of ITERS iterations of U
// copies of one inline-assembly block, U = 4, 6, 8; cost per pattern = (cycles(U=8) - cycles(U=4)) /
// (4 * ITERS * patterns per block), and U = 6 must sit midway (linearity). The blocks are generated with fixed
// registers by scripts/cvt_gen_timing.py (src/study/cvt_time_blocks.h has the text and the description of each).
// Instruction tests: 16 patterns per block; "c16" = 16 independent destinations (issue cost), "c1" = one dependent
// chain (latency). Linearity bound: 1.5 cycles per iteration or 5 % of the iteration, whichever is larger.
// Widening tests: 8 patterns per block, one pattern = one 32-bit word of 4 weights, scheduled
// step by step across the 8 words; "use_" tests add the instructions that consume the widened weights.
// Checks: the calibration (v_fma_f32: 1.00 independent, 5.00 chained, lane 1) and linearity of every test; the
// FP16 table of the LDS test must sit at LDS address 0 (the assembly addresses it from 0).
// Usage: cvt_time [test-name-substring]
#include "core_common.h"
#include "cvt_ref.h"
#include "cvt_time_blocks.h"

#include <algorithm>
#include <cmath>
#include <string>

// FP16 bits of an E4M3 code (exact: every E4M3 value is an FP16 value); NaN codes give an FP16 NaN.
__host__ __device__ inline uint16_t f16_of_e4m3(uint32_t c) {
    const uint32_t f = cr::decode8(cr::FP8, c);
    const uint32_t s = (f >> 16) & 0x8000u, a = f & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return (uint16_t) (s | 0x7E00u);
    }
    if (a == 0) {
        return (uint16_t) s;
    }
    return (uint16_t) (s | (((a >> 23) - 127 + 15) << 10) | ((a >> 13) & 0x3FFu));
}

#define CVT_KERNEL(NAME, P, W, DESC)                                                                              \
    template <int U>                                                                                              \
    __global__ __launch_bounds__(32) void k_##NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out, \
                                                   uint32_t * __restrict__ sink) {                                \
        __shared__ uint16_t tab[256];                                                                             \
        for (int i = threadIdx.x; i < 256; i += 32) {                                                             \
            tab[i] = f16_of_e4m3(i);                                                                              \
        }                                                                                                         \
        __syncthreads();                                                                                          \
        asm volatile(CVT_INIT ::"v"((uint32_t) in[0]), "v"((uint32_t) in[1]), "v"((uint32_t) in[2])              \
                     : CVT_CLOBBERS, "memory");                                                                   \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                              \
            const uint64_t t0 = wave_cycles();                                                                    \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                        \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                   \
                    asm volatile(CVT_BLOCK_##NAME ::: CVT_CLOBBERS, "vcc", "memory");                             \
                }                                                                                                 \
            }                                                                                                     \
            const uint64_t t1 = wave_cycles();                                                                    \
            if (threadIdx.x == 0) {                                                                               \
                out[rep] = cycle_diff(t0, t1);                                                                    \
            }                                                                                                     \
        }                                                                                                         \
        uint32_t s;                                                                                               \
        asm volatile("v_xor_b32 %0, v64, v128" : "=v"(s)::CVT_CLOBBERS);                                          \
        sink[threadIdx.x] = s;                                                                                    \
        if (threadIdx.x == 0) {                                                                                   \
            sink[64] = (uint32_t) (size_t) tab;                                                                   \
        }                                                                                                         \
    }
CVT_TESTS(CVT_KERNEL)

struct Test {
    const char * name;
    int patterns, wpp;
    const char * desc;
    core::Kern k4, k6, k8;
};
#define CVT_ENTRY(NAME, P, W, DESC) {#NAME, P, W, DESC, k_##NAME<4>, k_##NAME<6>, k_##NAME<8>},
static const Test TESTS[] = {CVT_TESTS(CVT_ENTRY)};

int main(int argc, char ** argv) {
    require_arch();
    const std::string filter = argc > 1 ? argv[1] : "";
    // FP8 word with four ordinary codes (1.0, 1.25, -0.5, 2.0 in E4M3); as FP32 or FP16 pairs it is a normal number
    core::set_init({0x40B03A38u, 0x3F800000u, 0x3C003C00u});
    bool ok = true, cal = true;
    int nlin = 0;
    std::printf("%-22s %8s %16s %8s %8s  %s\n", "test", "cyc/pat", "range", "cyc/wt", "U6-mid", "what");
    for (const Test & t : TESTS) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) {
            continue;
        }
        const core::Cost c = core::pattern_cost(t.k4, t.k6, t.k8, t.patterns);
        uint32_t lds_addr = 1;
        CHECK(hipMemcpy(&lds_addr, core::buffers().sink + 64, 4, hipMemcpyDeviceToHost));
        // long blocks (LDS waits, 80+ cycles) jitter by a few cycles per block, so the bound grows with the block
        const bool lin = std::fabs(c.mid) <= std::max(1.5, 0.05 * c.it_mid);
        nlin += !lin;
        char wt[32] = "-";
        if (t.wpp) {
            std::snprintf(wt, sizeof wt, "%.3f", c.per / t.wpp);
        }
        std::printf("RESULT %-22s %8.3f %7.3f-%-8.3f %8s %8.2f  %s%s%s [cycles/iteration U=4,6,8: %.1f %.1f %.1f]\n",
                    t.name, c.per, c.lo, c.hi, wt, c.mid, t.desc, lin ? "" : " NOT LINEAR",
                    lds_addr ? " LDS TABLE NOT AT 0" : "", c.it_lo, c.it_mid, c.it_hi);
        ok = ok && lin && lds_addr == 0;
        if (std::string(t.name) == "fma_f32_c16") {
            cal = cal && std::fabs(c.per - 1.0) < 0.05;
        }
        if (std::string(t.name) == "fma_f32_c1") {
            cal = cal && std::fabs(c.per - 5.0) < 0.1;
        }
    }
    ok = ok && cal;
    char buf[160];
    std::snprintf(buf, sizeof buf, "calibration v_fma_f32 1.00/5.00 %s; %d tests not linear", cal ? "reproduced" : "OFF",
                  nlin);
    verdict("cvt_time", ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}
