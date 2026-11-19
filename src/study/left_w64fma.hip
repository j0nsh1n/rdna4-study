// Leftovers C, lane A, fix round, test T1 (measurements/leftovers-c-a.md, SC-10): a copy of the FMA section of
// src/study/sched_wave.hip (scheduling-and-clocks item 4, where one wave64 wave issued v_fma_f32 at 0.81 per cycle),
// with the same kernel and the same host measurement, plus variants that pin the registers.
//
// Built once, wave64 in CU mode: -mcumode -DCUMODE=1 -mwavefrontsize64 -DW64=1 (as sched_wave64).
// Usage: left_w64fma [--break]
//   MODE 0  sched_wave's kernel unchanged (k_fma<0, U>): 16 chains and two constants, registers chosen by the
//           compiler (in that build: chains v0-v15, constants v16 and v17, so 8 of 16 FMAs read two registers of one
//           bank, bank = register number mod 4).
//   MODE 5  the same layout pinned by register constraints (chains v0-v15, constants v16, v17): must equal MODE 0.
//   MODE 2  chains in banks 2 and 3 only, constants v32 (bank 0) and v33 (bank 1): no FMA reads two registers of one
//           bank.
//   MODE 3  chains in banks 1 to 3, constants v24 and v28 (both bank 0): all 16 FMAs read two registers of one bank.
//   MODE 4  chains in bank 0 (v0, v4, ... v60), constants v64 (bank 0) and v65 (bank 1): all 16 read two registers of
//           bank 0 (the chain and one constant).
// The rate is lane-FMAs per cycle on the SIMD's own counter, from the slope of the span over two iteration counts and
// the difference of U = 8 and U = 4 blocks per iteration (sched_wave's method; 7 repetitions, median).
// Verdicts: w64fma_control: MODE 0 reproduces item 4 (51.85 lane-FMAs per cycle, +-3 %) and MODE 5 equals MODE 0
// within 2 %. w64fma_rates: INFO (the expectations are written in the write-up, T1). --break expects MODE 0 at 64.
// Listing: scripts/leftc_bankcheck.py counts, per kernel, the FMAs whose VGPR sources share a bank.
#include "sched_common.h"

#include <cstdarg>

#ifndef W64
#define W64 0
#endif

namespace {

constexpr int WS = W64 ? 64 : 32;  // lanes per wave in this build
constexpr int REC = 32;
constexpr int MAXW = 32;

#define F1(i) "v_fma_f32 %" #i ", %" #i ", %16, %17\n"
#define FMA16 F1(0) F1(1) F1(2) F1(3) F1(4) F1(5) F1(6) F1(7) F1(8) F1(9) F1(10) F1(11) F1(12) F1(13) F1(14) F1(15)
#define XOPS                                                                                                      \
    "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), "+v"(x[8]), \
        "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])
#define FMA_BLOCK() asm volatile(FMA16 : XOPS : "v"(k0), "v"(k1))

// pinned layouts: the 16 chain registers and the two constant registers of each mode
#define P16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)                                                       \
    "+{v" #a "}"(x[0]), "+{v" #b "}"(x[1]), "+{v" #c "}"(x[2]), "+{v" #d "}"(x[3]), "+{v" #e "}"(x[4]),           \
        "+{v" #f "}"(x[5]), "+{v" #g "}"(x[6]), "+{v" #h "}"(x[7]), "+{v" #i "}"(x[8]), "+{v" #j "}"(x[9]),       \
        "+{v" #k "}"(x[10]), "+{v" #l "}"(x[11]), "+{v" #m "}"(x[12]), "+{v" #n "}"(x[13]), "+{v" #o "}"(x[14]), \
        "+{v" #p "}"(x[15])
#define PIN_BLOCK(XO, K0, K1) asm volatile(FMA16 : XO : "{" K0 "}"(k0), "{" K1 "}"(k1))
#define X5 P16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#define X2 P16(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31)
#define X3 P16(1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21)
#define X4 P16(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)

__device__ __forceinline__ bool lane0() { return (threadIdx.x % WS) == 0; }

// sched_wave's k_fma (MODE 0 and 1 as there), with the pinned modes added. All waves of the group take a rank on their
// SIMD; waves on SIMD 0 with rank < nact run the loop, everyone else leaves.
template <int MODE, int U>
__global__ __launch_bounds__(1024) void k_fma(const float * in, uint64_t * out, float * sink, int nact, int n) {
    __shared__ int cnt[2];
    if (threadIdx.x < 2) {
        cnt[threadIdx.x] = 0;
    }
    __syncthreads();
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    const int simd = (hw >> 9) & 1;
    int rank = 0;
    if (lane0()) {
        rank = atomicAdd(&cnt[simd], 1);
    }
    rank = __builtin_amdgcn_readfirstlane(rank);
    const int w = threadIdx.x / WS;
    const uint32_t pl = sched::placement();
    if (lane0()) {
        out[w * REC] = pl;
        out[w * REC + 3] = rank;
        out[w * REC + 4] = simd;
    }
    __syncthreads();
    if (simd != 0 || rank >= nact) {
        return;
    }
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[i];
    }
    float k0 = in[16], k1 = in[17];
    asm volatile("" : "+v"(k0), "+v"(k1));
    const uint64_t s0 = sched::cyc();
#pragma unroll 1
    for (int it = 0; it < n; ++it) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (MODE == 0) {
                FMA_BLOCK();
            } else if (MODE == 5) {
                PIN_BLOCK(X5, "v16", "v17");
            } else if (MODE == 2) {
                PIN_BLOCK(X2, "v32", "v33");
            } else if (MODE == 3) {
                PIN_BLOCK(X3, "v24", "v28");
            } else {
                PIN_BLOCK(X4, "v64", "v65");
            }
        }
    }
    const uint64_t e0 = sched::cyc();
    if (lane0()) {
        out[w * REC + 1] = s0;
        out[w * REC + 2] = e0;
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    sink[threadIdx.x] = s;
}

uint64_t * d_out;
float * d_in;
float * d_sink;

std::vector<uint64_t> fetch(int nwaves) {
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> h((size_t) nwaves * REC);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * 8, hipMemcpyDeviceToHost));
    return h;
}
void clear_out() { CHECK(hipMemset(d_out, 0, (size_t) MAXW * REC * 8)); }

int fails = 0;
void put(const char * name, bool ok, const char * f, ...) __attribute__((format(printf, 3, 4)));
void put(const char * name, bool ok, const char * f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    fails += !ok;
    std::printf("VERDICT %s: %s %s\n", name, ok ? "PASS" : "FAIL", b);
    std::fflush(stdout);
}

using K = void (*)(const float *, uint64_t *, float *, int, int);
struct ModeK {
    int mode;
    const char * name;
    K k4, k8;
};
const ModeK kModes[] = {
    {0, "sched_wave kernel unchanged (compiler's registers)", k_fma<0, 4>, k_fma<0, 8>},
    {5, "pinned v0-v15, constants v16 v17 (as MODE 0)", k_fma<5, 4>, k_fma<5, 8>},
    {2, "chains in banks 2-3, constants v32 v33 (no shared bank)", k_fma<2, 4>, k_fma<2, 8>},
    {3, "chains in banks 1-3, constants v24 v28 (both bank 0)", k_fma<3, 4>, k_fma<3, 8>},
    {4, "chains in bank 0, constants v64 v65 (chain and v64 in bank 0)", k_fma<4, 4>, k_fma<4, 8>},
};

// sched_wave's measurement for one active wave: lane-FMAs per cycle per SIMD, one value per repetition
std::vector<double> rate(const ModeK & m) {
    const int n1 = 64, n2 = 192, nact = 1;
    std::vector<double> thr;
    for (int rep = 0; rep < 7; ++rep) {
        double per_iter[2];
        int got_min = 99;
        for (int ui = 0; ui < 2; ++ui) {
            double span[2];
            for (int k = 0; k < 2; ++k) {
                clear_out();
                hipLaunchKernelGGL(ui ? m.k8 : m.k4, dim3(1), dim3(1024), 0, 0, d_in, d_out, d_sink, nact, k ? n2 : n1);
                const auto h = fetch(1024 / WS);
                uint64_t mn = ~0ull, mx = 0;
                int got = 0;
                for (int w = 0; w < 1024 / WS; ++w) {
                    if (h[(size_t) w * REC + 2] == 0) {
                        continue;
                    }
                    ++got;
                    mn = std::min(mn, h[(size_t) w * REC + 1]);
                    mx = std::max(mx, h[(size_t) w * REC + 2]);
                }
                got_min = std::min(got_min, got);
                span[k] = (double) (mx - mn);
            }
            per_iter[ui] = (span[1] - span[0]) / (n2 - n1);
        }
        thr.push_back((double) got_min * WS * 4 * 16 / (per_iter[1] - per_iter[0]));
    }
    return thr;
}

}  // namespace

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const bool brk = sched::has_flag(argc, argv, "--break");
    std::printf("# left_w64fma: build wave%d, CU mode, device %s\n", WS, prop.gcnArchName);
    CHECK(hipMalloc(&d_out, (size_t) MAXW * REC * 8));
    CHECK(hipMalloc(&d_in, 64 * 4));
    CHECK(hipMalloc(&d_sink, 1024 * 4));
    {
        float h[64];
        for (int i = 0; i < 64; ++i) {
            h[i] = 1.0f + 0.01f * (float) (i & 15);
        }
        h[16] = 1.0f;  // k0 for the asm FMAs: x = x * 1 + 0 stays put
        h[17] = 0.0f;
        CHECK(hipMemcpy(d_in, h, sizeof h, hipMemcpyHostToDevice));
    }
    // three rounds over the modes, interleaved, 7 repetitions each per round
    std::vector<double> all[6];
    for (int round = 0; round < 3; ++round) {
        for (const ModeK & m : kModes) {
            const auto v = rate(m);
            all[m.mode].insert(all[m.mode].end(), v.begin(), v.end());
        }
    }
    double med[6] = {};
    for (const ModeK & m : kModes) {
        med[m.mode] = sched::pct(all[m.mode], 0.5);
        std::printf("RESULT w64fma mode %d %-62s lane_fma_per_cycle %s cycles_per_instr %.4f\n", m.mode, m.name,
                    sched::dist(all[m.mode], "%.2f").c_str(), WS / med[m.mode]);
    }
    const double want0 = brk ? 64.0 : 51.85;
    put("w64fma_control", std::fabs(med[0] / want0 - 1) <= 0.03 && std::fabs(med[5] / med[0] - 1) <= 0.02,
        "wave%d: sched_wave's kernel %.2f lane-FMAs per cycle (expect %.2f +-3 %%%s: scheduling-and-clocks item 4); "
        "the same layout pinned %.2f (expect within 2 %% of it)",
        WS, med[0], want0, brk ? ", deliberate break" : "", med[5]);
    std::printf("VERDICT w64fma_rates: INFO wave%d cycles per instruction: compiler's layout %.3f, pinned %.3f, no shared "
                "bank %.3f, constants in one bank (all 16) %.3f, chain and constant in one bank (all 16) %.3f\n",
                WS, WS / med[0], WS / med[5], WS / med[2], WS / med[3], WS / med[4]);
    return fails ? 1 : 0;
}
