// Lane 2, item 6 (occupancy against registers) and the several-waves half of item 5 (loads in flight with many
// waves), whole card.
//
// Usage: mem_occupancy <rule|bw> [--break]
//   rule  for each register count R (forced with an inline-assembly clobber of v[R-1]; the count is read back with
//         hipFuncGetAttributes and is in the .lto.s descriptor), 4096 one-wave work-groups each count themselves in
//         and out of a per-SIMD counter while spinning 200 us: the highest count seen on any SIMD is the number of
//         waves that fit on a SIMD. Compared with min(16, floor(1536 / (R rounded up to 24)))
//         (READ: research-5 and LLVM's 24-register granule on 1536-register targets).
//         --break compares with a 1024-register file instead: the verdict must FAIL.
//   bw    streaming read of 512 MiB with w waves per SIMD (128 x w one-wave work-groups, persistent; the placement of
//         every wave is recorded and the per-SIMD spread printed), N independent 16-byte loads per wave before each
//         wait (1, 4, 16), and the kernel padded to R registers. Answers: how many waves and loads in flight saturate
//         the bus, and whether the register count costs anything beyond the wave slots it takes.
#include "mem_common.h"

#include <map>

typedef unsigned u4 __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------------------------------------------
// rule
__device__ __forceinline__ void occ_body(uint32_t * cnt, uint32_t * maxc, uint64_t spin) {
    const uint32_t key = mem::pl_simd(mem::placement());
    uint32_t now = 0;
    if (threadIdx.x == 0) {
        now = atomicAdd(&cnt[key], 1u) + 1;
        atomicMax(&maxc[key], now);
    }
    const uint64_t t0 = mem::realtime();
    while (mem::realtime() - t0 < spin) {
        __builtin_amdgcn_s_sleep(2);
    }
    if (threadIdx.x == 0) {
        // a second look: waves that arrived after this one
        atomicMax(&maxc[key], atomicAdd(&cnt[key], 0u));
        atomicSub(&cnt[key], 1u);
    }
}

#define OCC_KERNEL(R, LAST)                                                                                        \
    __global__ __launch_bounds__(32) void k_occ_##R(uint32_t * cnt, uint32_t * maxc, uint64_t spin) {              \
        asm volatile("" ::: LAST);                                                                                 \
        occ_body(cnt, maxc, spin);                                                                                 \
    }
OCC_KERNEL(24, "v23")
OCC_KERNEL(25, "v24")
OCC_KERNEL(48, "v47")
OCC_KERNEL(49, "v48")
OCC_KERNEL(64, "v63")
OCC_KERNEL(72, "v71")
OCC_KERNEL(73, "v72")
OCC_KERNEL(96, "v95")
OCC_KERNEL(97, "v96")
OCC_KERNEL(120, "v119")
OCC_KERNEL(128, "v127")
OCC_KERNEL(144, "v143")
OCC_KERNEL(160, "v159")
OCC_KERNEL(168, "v167")
OCC_KERNEL(192, "v191")
OCC_KERNEL(193, "v192")
OCC_KERNEL(256, "v255")

struct OccK {
    int r;
    void (*k)(uint32_t *, uint32_t *, uint64_t);
};
static const OccK OCC[] = {{24, k_occ_24},   {25, k_occ_25},   {48, k_occ_48},   {49, k_occ_49},   {64, k_occ_64},
                           {72, k_occ_72},   {73, k_occ_73},   {96, k_occ_96},   {97, k_occ_97},   {120, k_occ_120},
                           {128, k_occ_128}, {144, k_occ_144}, {160, k_occ_160}, {168, k_occ_168}, {192, k_occ_192},
                           {193, k_occ_193}, {256, k_occ_256}};

static int rule_waves(int regs, int file) {
    const int alloc = (regs + 23) / 24 * 24;
    return std::min(16, file / alloc);
}

// ---------------------------------------------------------------------------------------------------------------
// bw: each wave takes batches b = wave, wave + waves, ...; a batch is N loads of 512 bytes (16 per lane).
template <int N>
__device__ __forceinline__ void bw_body(const char * base, uint32_t batches, unsigned * sink, uint32_t * place) {
    const uint32_t wave = __builtin_amdgcn_readfirstlane((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    const uint32_t waves = gridDim.x * blockDim.x / 32;
    const uint32_t lane_off = (threadIdx.x % 32) * 16;
    if (threadIdx.x % 32 == 0) {
        place[wave] = mem::placement();
    }
    unsigned acc = 0;
    for (uint32_t b = wave; b < batches; b += waves) {
        const char * q = base + (size_t) b * N * 512;
        u4 v[N];
#pragma unroll
        for (int k = 0; k < N; ++k) {
            asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(v[k]) : "v"(lane_off), "s"(q), "i"(k * 512) : "memory");
        }
        asm volatile("s_wait_loadcnt 0" ::: "memory");
#pragma unroll
        for (int k = 0; k < N; ++k) {
            asm volatile("" : "+v"(v[k]));
            acc += v[k].x ^ v[k].w;
        }
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#define BW_KERNEL(R, LAST)                                                                                         \
    template <int N>                                                                                               \
    __global__ __launch_bounds__(32) void k_bw_##R(const char * base, uint32_t batches, unsigned * sink,           \
                                                   uint32_t * place) {                                             \
        asm volatile("" ::: LAST);                                                                                 \
        bw_body<N>(base, batches, sink, place);                                                                    \
    }
BW_KERNEL(24, "v23")
BW_KERNEL(48, "v47")
BW_KERNEL(72, "v71")
BW_KERNEL(96, "v95")
BW_KERNEL(120, "v119")
BW_KERNEL(144, "v143")
BW_KERNEL(192, "v191")

using BwFn = void (*)(const char *, uint32_t, unsigned *, uint32_t *);
struct BwK {
    int r, n;
    BwFn k;
};
#define BW_ROW(R) {R, 1, k_bw_##R<1>}, {R, 4, k_bw_##R<4>}, {R, 16, k_bw_##R<16>}
static const BwK BW[] = {BW_ROW(24), BW_ROW(48), BW_ROW(72), BW_ROW(96), BW_ROW(120), BW_ROW(144), BW_ROW(192)};

// ---------------------------------------------------------------------------------------------------------------
struct Ctx {
    char * buf = nullptr;
    unsigned * sink = nullptr;
    uint32_t * place = nullptr;
    bool fail = false;
};

static int num_regs(const void * k) {
    hipFuncAttributes a;
    CHECK(hipFuncGetAttributes(&a, k));
    return a.numRegs;
}

static void mode_rule(Ctx & c, bool brk) {
    mem::clocks("rule-start");
    uint32_t *cnt, *maxc;
    CHECK(hipMalloc(&cnt, 2048 * 4));
    CHECK(hipMalloc(&maxc, 2048 * 4));
    const int file = brk ? 1024 : 1536;
    int agree = 0, total = 0;
    for (const OccK & o : OCC) {
        const int regs = num_regs((const void *) o.k);
        CHECK(hipMemset(cnt, 0, 2048 * 4));
        CHECK(hipMemset(maxc, 0, 2048 * 4));
        mem::cool_down();
        hipLaunchKernelGGL(o.k, dim3(4096), dim3(32), 0, 0, cnt, maxc, (uint64_t) 20000);  // 200 us of REALTIME
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> m(2048);
        CHECK(hipMemcpy(m.data(), maxc, 2048 * 4, hipMemcpyDeviceToHost));
        uint32_t hi = 0, simds = 0;
        std::vector<int> hist(17, 0);
        for (uint32_t v : m) {
            if (v) {
                ++simds;
                hi = std::max(hi, v);
                ++hist[std::min<uint32_t>(v, 16)];
            }
        }
        const int want = rule_waves(regs, file);
        ++total;
        agree += (int) hi == want;
        std::string h;
        for (int i = 1; i <= 16; ++i) {
            if (hist[i]) {
                h += " " + std::to_string(i) + ":" + std::to_string(hist[i]);
            }
        }
        std::printf("RESULT mem_occupancy rule pad %d vgprs %d waves_per_simd_max %u rule %d simds_used %u "
                    "histogram(waves:simds)%s\n",
                    o.r, regs, hi, want, simds, h.c_str());
        std::fflush(stdout);
    }
    char t[200];
    std::snprintf(t, sizeof t, "%d of %d register counts give the waves per SIMD of min(16, %d / (regs rounded up to 24))%s",
                  agree, total, file, brk ? " (deliberate break: wrong register file size)" : "");
    verdict("mem_occupancy_rule", agree == total ? "PASS" : "FAIL", t);
    c.fail |= agree != total;
    CHECK(hipFree(cnt));
    CHECK(hipFree(maxc));
    mem::clocks("rule-end");
}

static void mode_bw(Ctx & c) {
    mem::clocks("bw-start");
    const size_t bytes = 512ull << 20;
    for (const BwK & b : BW) {
        const int regs = num_regs((const void *) b.k);
        const int fit = rule_waves(regs, 1536);
        for (int w : {1, 2, 3, 4, 6, 8, 10, 12, 16}) {
            if (w > fit) {
                continue;
            }
            const uint32_t batches = (uint32_t) (bytes / (b.n * 512));
            const int waves = 128 * w;
            mem::cool_down();
            auto ms = time_ms(
                [&] {
                    hipLaunchKernelGGL(b.k, dim3(waves), dim3(32), 0, 0, c.buf, batches, c.sink, c.place);
                },
                5, 1);
            CHECK(hipGetLastError());
            std::vector<uint32_t> pl(waves);
            CHECK(hipMemcpy(pl.data(), c.place, waves * 4, hipMemcpyDeviceToHost));
            std::map<uint32_t, int> per;
            for (uint32_t p : pl) {
                ++per[mem::pl_simd(p)];
            }
            int lo = 1 << 30, hi = 0;
            for (auto & kv : per) {
                lo = std::min(lo, kv.second);
                hi = std::max(hi, kv.second);
            }
            char what[200];
            std::snprintf(what, sizeof what, "bw pad %d vgprs %d loads_in_flight %d waves_per_simd %d simds_used %zu "
                          "per_simd_min %d max %d", b.r, regs, b.n, w, per.size(), lo, hi);
            mem::report_bw("mem_occupancy", what, ms, (double) bytes);
        }
    }
    mem::clocks("bw-end");
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: mem_occupancy <rule|bw> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode %s%s\n", prop.name, prop.gcnArchName, mode.c_str(), brk ? " --break" : "");
    Ctx c;
    CHECK(hipMalloc(&c.buf, 512ull << 20));
    mem::fill(c.buf, 512ull << 20, mem::RANDOM);
    CHECK(hipMalloc(&c.sink, 128 * 16 * 32 * 4));
    CHECK(hipMalloc(&c.place, 128 * 16 * 4));
    if (mode == "rule") {
        mode_rule(c, brk);
    } else if (mode == "bw") {
        mode_bw(c);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::report_hottest();
    return c.fail ? 1 : 0;
}
