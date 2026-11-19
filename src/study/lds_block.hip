// Lane 3, item 6: block register loads and stores (GLOBAL_LOAD_BLOCK / GLOBAL_STORE_BLOCK) on gfx1201, wave32.
//
// Usage: lds_block <calib|layout|cost> [--break]
//   calib   the FMA calibration of lane 1 (16 independent v_fma_f32, differential U=8 minus U=4): 1.00 cycle/FMA.
//   layout  one wave. M0 is set in the same asm statement as the block instruction. The 32-register destination is
//           a 1024-bit "+v" operand pre-filled with a sentinel, so a VGPR the mask skips must keep its sentinel.
//           Rule under test (guide B 11.5): VGPR n gets the DWORD at address/4 + n when M0 bit n is set, and the
//           address advances 4 bytes per VGPR whether it was transferred or not. Loads, then stores; then an
//           address that is not 4-byte aligned and lanes only 16 bytes apart (reported, not judged).
//   cost    one wave, cycles for N back-to-back block loads (128 bytes per lane, 4 KiB per wave) then
//           s_wait_loadcnt 0, against the same bytes by 8 global_load_b128 per block (two address patterns), and
//           a block with mask 0x0000FFFF (16 registers). Hot (a 16 KiB region read before) and DRAM-cold (fresh
//           lines after a 512 MiB flush).
//           The VGPR pool is at most 128 registers (4 block tuples, or 32 b128 quads): with N > 4 blocks (N = 8)
//           the destinations are reused while earlier loads are in flight.
//   --break the layout check expects VGPR n to hold word addr/4 + n + 1, so VERDICT layout must FAIL.
#include "mem_common.h"
#include "core_common.h"

typedef unsigned u32x32 __attribute__((ext_vector_type(32)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------------------------------------------
// calibration (lane 1 method, core_common.h)
#define P_D_DKL(op, i) op " %" #i ", %" #i ", %16, %17\n"
CORE_KERNEL(calib_fma, float, float, "+v", "v", CORE_B16(P_D_DKL, "v_fma_f32"), "")

// ---------------------------------------------------------------------------------------------------------------
// block instructions. M0 and the instruction are one asm statement so the compiler cannot put anything between.
template <uint32_t MASK, int OFF>
__device__ __forceinline__ void blk_load(u32x32 & r, const char * base, uint32_t voff) {
    asm volatile("s_mov_b32 m0, %3\n\tglobal_load_block %0, %1, %2 offset:%4"
                 : "+v"(r)
                 : "v"(voff), "s"(base), "i"(MASK), "i"(OFF)
                 : "memory");
}

template <uint32_t MASK>
__device__ __forceinline__ void blk_store(const u32x32 & r, char * base, uint32_t voff) {
    asm volatile("s_mov_b32 m0, %3\n\tglobal_store_block %1, %0, %2"
                 :
                 : "v"(r), "v"(voff), "s"(base), "i"(MASK)
                 : "memory");
}

// ---------------------------------------------------------------------------------------------------------------
// layout kernels. out[lane*32 + n] = VGPR n of that lane after the load.
constexpr uint32_t SENT = 0xA5000000u;  // sentinel for a skipped register: SENT | lane << 8 | n

template <uint32_t MASK>
__global__ __launch_bounds__(32) void k_layout_load(const char * base, uint32_t stride, uint32_t extra, uint32_t * out) {
    const uint32_t lane = threadIdx.x;
    u32x32 r;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        r[n] = SENT | (lane << 8) | n;
    }
    blk_load<MASK, 0>(r, base, lane * stride + extra);
    asm volatile("s_wait_loadcnt 0" ::: "memory");
    asm volatile("" : "+v"(r));
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        out[lane * 32 + n] = r[n];
    }
}

// Source VGPR n of lane l holds STORE_VAL(l, n); skipped registers must leave memory untouched.
__host__ __device__ inline uint32_t store_val(uint32_t lane, uint32_t n) { return 0x10000000u + lane * 32 + n; }

template <uint32_t MASK>
__global__ __launch_bounds__(32) void k_layout_store(char * base, uint32_t stride, uint32_t extra) {
    const uint32_t lane = threadIdx.x;
    u32x32 r;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        r[n] = store_val(lane, n);
    }
    asm volatile("" : "+v"(r));
    blk_store<MASK>(r, base, lane * stride + extra);
    asm volatile("s_wait_storecnt 0" ::: "memory");
}

// ---------------------------------------------------------------------------------------------------------------
// cost kernel
enum Kind { BLK32 = 0, BLK16 = 1, B128 = 2, B128C = 3 };
constexpr uint32_t REGION = 16384;  // hot region (L0)
constexpr uint32_t BLOCK_BYTES = 4096;

template <int KIND>
constexpr const char * kind_name() {
    return KIND == BLK32 ? "blk32" : KIND == BLK16 ? "blk16" : KIND == B128 ? "b128_lane128" : "b128_coalesced";
}

// Block number i of an iteration, destination slot s: tuples t[0..3] for block kinds, quads q[(i*8+k) % 32] for b128.
template <int KIND, bool HOT, int I, int N, int NT, int NQ>
__device__ __forceinline__ void issue(u32x32 (&t)[NT], u4 (&q)[NQ], const char * base, uint32_t lane) {
    if constexpr (I < N) {
        constexpr uint32_t off = HOT ? (I * BLOCK_BYTES) & (REGION - 1) : I * BLOCK_BYTES;
        if constexpr (KIND == BLK32) {
            blk_load<0xFFFFFFFFu, (int) off>(t[I % NT], base, lane * 128);
        } else if constexpr (KIND == BLK16) {
            blk_load<0x0000FFFFu, (int) off>(t[I % NT], base, lane * 128);
        } else {
            const uint32_t vo = KIND == B128 ? lane * 128 : lane * 16;
            constexpr int KS = KIND == B128 ? 16 : 512;
#define LD(K)                                                                                                       \
    asm volatile("global_load_b128 %0, %1, %2 offset:%3"                                                           \
                 : "=v"(q[(I * 8 + K) % NQ])                                                                       \
                 : "v"(vo), "s"(base), "i"((int) off + K * KS)                                                     \
                 : "memory");
            LD(0) LD(1) LD(2) LD(3) LD(4) LD(5) LD(6) LD(7)
#undef LD
        }
        issue<KIND, HOT, I + 1, N, NT, NQ>(t, q, base, lane);
    }
}

// out: [0] timed cycles, [1] mismatches in the data of the last block, [2] checked words.
template <int KIND, bool HOT, int N>
__global__ __launch_bounds__(32) void k_cost(const char * base, uint32_t base_word, uint32_t iters, uint32_t warm,
                                             uint64_t * out) {
    const uint32_t lane = threadIdx.x;
    constexpr bool IS_BLK = KIND == BLK32 || KIND == BLK16;
    constexpr int NT = IS_BLK ? (N < 4 ? N : 4) : 1;
    constexpr int NQ = IS_BLK ? 1 : (N * 8 < 32 ? N * 8 : 32);
    u32x32 t[NT];
    u4 q[NQ];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        t[j] = u32x32{};
    }
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        q[j] = u4{};
    }
    auto step = [&](uint32_t it) {
        const uint32_t adv = HOT ? (it * N * BLOCK_BYTES) & (REGION - 1) : it * N * BLOCK_BYTES;
        issue<KIND, HOT, 0, N, NT, NQ>(t, q, base + adv, lane);
        asm volatile("s_wait_loadcnt 0" ::: "memory");
#pragma unroll
        for (int j = 0; j < (IS_BLK ? NT : 0); ++j) {
            asm volatile("" : "+v"(t[j]));
        }
#pragma unroll
        for (int j = 0; j < (IS_BLK ? 0 : NQ); ++j) {
            asm volatile("" : "+v"(q[j]));
        }
    };
#pragma unroll 1
    for (uint32_t it = 0; it < warm; ++it) {
        step(it);
    }
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t it = 0; it < iters; ++it) {
        step(it);
    }
    const uint64_t c1 = wave_cycles();
    // data check of the last iteration's last block (ramp: word i of the buffer holds i)
    const uint32_t last = iters - 1;
    const uint32_t adv = HOT ? (last * N * BLOCK_BYTES) & (REGION - 1) : last * N * BLOCK_BYTES;
    constexpr int LI = N - 1;
    const uint32_t boff = HOT ? (LI * BLOCK_BYTES) & (REGION - 1) : LI * BLOCK_BYTES;
    const uint32_t first_word = base_word + (adv + boff) / 4;
    uint32_t bad = 0, checked = 0;
    if constexpr (KIND == BLK32 || KIND == BLK16) {
        constexpr int NW = KIND == BLK32 ? 32 : 16;
#pragma unroll
        for (int n = 0; n < NW; ++n) {
            bad += t[LI % NT][n] != first_word + lane * 32 + n;
            ++checked;
        }
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const u4 v = q[(LI * 8 + k) % NQ];
            const uint32_t w = first_word + (KIND == B128 ? lane * 32 + k * 4 : lane * 4 + k * 128);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                bad += v[c] != w + c;
                ++checked;
            }
        }
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (lane == 0) {
        out[0] = c1 - c0;
        out[1] = bad;
        out[2] = checked * 32;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host
struct Ctx {
    char * buf = nullptr;    // 256 MiB ramp
    char * flush = nullptr;  // 512 MiB
    uint32_t * out = nullptr;
    uint64_t * cyc = nullptr;
    unsigned * sink = nullptr;
    bool fail = false;
};

__global__ void k_flush(const u4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        acc += p[i].x;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static void do_flush(Ctx & c) {
    hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const u4 *) c.flush, (512u << 20) / 16, c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

static uint32_t rd32(const std::vector<uint8_t> & m, int64_t a) {
    if (a < 0 || a + 4 > (int64_t) m.size()) {
        return 0xFFFFFFFFu;
    }
    uint32_t v;
    std::memcpy(&v, m.data() + a, 4);
    return v;
}

constexpr size_t RAMP_BYTES = 65536;

static std::vector<uint8_t> host_ramp() {
    std::vector<uint32_t> w(RAMP_BYTES / 4);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = (uint32_t) i;
    }
    std::vector<uint8_t> b(RAMP_BYTES);
    std::memcpy(b.data(), w.data(), RAMP_BYTES);
    return b;
}

template <uint32_t MASK>
static int layout_load(Ctx & c, char * dev_ramp, const std::vector<uint8_t> & ramp, uint32_t stride, uint32_t extra,
                       bool brk, const char * tag, bool judge) {
    hipLaunchKernelGGL(k_layout_load<MASK>, dim3(1), dim3(32), 0, 0, (const char *) dev_ramp, stride, extra, c.out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint32_t got[1024];
    CHECK(hipMemcpy(got, c.out, sizeof got, hipMemcpyDeviceToHost));
    int bad_rule = 0, bad_floor = 0, first_bad = -1;
    for (uint32_t l = 0; l < 32; ++l) {
        for (uint32_t n = 0; n < 32; ++n) {
            const int64_t a = (int64_t) l * stride + extra;
            const uint32_t sent = SENT | (l << 8) | n;
            const uint32_t want_exact = ((MASK >> n) & 1) ? rd32(ramp, a + 4 * (int64_t) n) + (brk ? 1u : 0u) : sent;
            const uint32_t want_floor = ((MASK >> n) & 1) ? rd32(ramp, (a & ~3ll) + 4 * (int64_t) n) : sent;
            const uint32_t v = got[l * 32 + n];
            if (v != want_exact) {
                if (first_bad < 0) {
                    first_bad = (int) (l * 32 + n);
                }
                ++bad_rule;
            }
            bad_floor += v != want_floor;
        }
    }
    std::printf("RESULT lds_block layout_load %s mask=0x%08X stride=%u extra=%u mismatches_vs_rule=%d mismatches_vs_floor=%d\n",
                tag, MASK, stride, extra, bad_rule, bad_floor);
    if (first_bad >= 0) {
        std::printf("RESULT lds_block layout_load %s first_mismatch lane=%d vgpr=%d got=0x%08X\n", tag, first_bad / 32,
                    first_bad % 32, got[first_bad]);
    }
    // a few readable cells: lane 0 and lane 1, VGPR 0..3 and 30..31
    std::printf("RESULT lds_block layout_load %s sample lane0_v0..3=%08X,%08X,%08X,%08X lane0_v31=%08X lane1_v0=%08X lane1_v1=%08X\n",
                tag, got[0], got[1], got[2], got[3], got[31], got[32], got[33]);
    std::fflush(stdout);
    if (judge && bad_rule != 0) {
        c.fail = true;
    }
    return bad_rule;
}

template <uint32_t MASK>
static int layout_store(Ctx & c, char * dev_mem, uint32_t stride, uint32_t extra, const char * tag, bool judge,
                        bool brk) {
    // memory starts as a sentinel ramp so an untouched word is recognisable
    std::vector<uint32_t> init(RAMP_BYTES / 4);
    for (size_t i = 0; i < init.size(); ++i) {
        init[i] = 0xEE000000u + (uint32_t) i;
    }
    CHECK(hipMemcpy(dev_mem, init.data(), RAMP_BYTES, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_layout_store<MASK>, dim3(1), dim3(32), 0, 0, dev_mem, stride, extra);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> got(RAMP_BYTES);
    CHECK(hipMemcpy(got.data(), dev_mem, RAMP_BYTES, hipMemcpyDeviceToHost));
    std::vector<uint8_t> base(RAMP_BYTES);
    std::memcpy(base.data(), init.data(), RAMP_BYTES);
    // models: exact bytes with lane 0 first (higher lanes win overlaps), the same with lane 31 first, and
    // rounded-down addresses
    auto model = [&](bool floor_addr, bool asc) {
        std::vector<uint8_t> m = base;
        for (int li = 0; li < 32; ++li) {
            const uint32_t l = asc ? li : 31 - li;
            for (uint32_t n = 0; n < 32; ++n) {
                if (!((MASK >> n) & 1)) {
                    continue;
                }
                int64_t a = (int64_t) l * stride + extra + 4 * (int64_t) n;
                if (floor_addr) {
                    a &= ~3ll;
                }
                const uint32_t v = store_val(l, n) + (brk ? 1u : 0u);
                std::memcpy(m.data() + a, &v, 4);
            }
        }
        return m;
    };
    auto diff = [&](const std::vector<uint8_t> & m) {
        int d = 0;
        for (size_t i = 0; i < RAMP_BYTES; i += 4) {
            d += rd32(m, (int64_t) i) != rd32(got, (int64_t) i);
        }
        return d;
    };
    const int d_exact = diff(model(false, true)), d_desc = diff(model(false, false)), d_floor = diff(model(true, true));
    int touched = 0;
    for (size_t i = 0; i < RAMP_BYTES; i += 4) {
        touched += rd32(got, (int64_t) i) != rd32(base, (int64_t) i);
    }
    std::printf("RESULT lds_block layout_store %s mask=0x%08X stride=%u extra=%u words_changed=%d diff_exact_lane0first=%d "
                "diff_exact_lane31first=%d diff_floor=%d\n",
                tag, MASK, stride, extra, touched, d_exact, d_desc, d_floor);
    std::fflush(stdout);
    if (judge && d_exact != 0) {
        c.fail = true;
    }
    return d_exact;
}

static void mode_layout(Ctx & c, bool brk) {
    const std::vector<uint8_t> ramp = host_ramp();
    char * dev_ramp;
    char * dev_mem;
    CHECK(hipMalloc(&dev_ramp, RAMP_BYTES));
    CHECK(hipMalloc(&dev_mem, RAMP_BYTES));
    CHECK(hipMemcpy(dev_ramp, ramp.data(), RAMP_BYTES, hipMemcpyHostToDevice));
    int bad = 0;
    bad += layout_load<0xFFFFFFFFu>(c, dev_ramp, ramp, 128, 0, brk, "aligned_full", true);
    bad += layout_load<0x0000FFFFu>(c, dev_ramp, ramp, 128, 0, brk, "aligned_low16", true);
    bad += layout_load<0xAAAAAAAAu>(c, dev_ramp, ramp, 128, 0, brk, "aligned_odd_regs", true);
    bad += layout_load<0x00000001u>(c, dev_ramp, ramp, 128, 0, brk, "aligned_one_reg", true);
    int sbad = 0;
    sbad += layout_store<0xFFFFFFFFu>(c, dev_mem, 128, 0, "aligned_full", true, brk);
    sbad += layout_store<0x0000FFFFu>(c, dev_mem, 128, 0, "aligned_low16", true, brk);
    sbad += layout_store<0xAAAAAAAAu>(c, dev_mem, 128, 0, "aligned_odd_regs", true, brk);
    sbad += layout_store<0x00000001u>(c, dev_mem, 128, 0, "aligned_one_reg", true, brk);
    if (bad == 0 && sbad == 0) {
        verdict("layout", "PASS", "VGPR n gets the DWORD at address/4+n when M0 bit n is set; skipped VGPRs keep their value; "
                                  "stores write only the masked DWORDs (4 masks, loads and stores)");
    } else {
        verdict("layout", "FAIL", "block load or store layout differs from the guide's rule");
        c.fail = true;
    }
    // odd cases: reported, not judged. A misaligned address or an overlap may behave in a way the rule does not cover.
    std::printf("INFO odd cases follow (reported only)\n");
    std::fflush(stdout);
    layout_load<0xFFFFFFFFu>(c, dev_ramp, ramp, 128, 2, false, "misaligned_by_2", false);
    layout_load<0xFFFFFFFFu>(c, dev_ramp, ramp, 128, 1, false, "misaligned_by_1", false);
    layout_load<0xFFFFFFFFu>(c, dev_ramp, ramp, 16, 0, false, "overlap_16B_lanes", false);
    layout_store<0xFFFFFFFFu>(c, dev_mem, 128, 2, "misaligned_by_2", false, false);
    layout_store<0xFFFFFFFFu>(c, dev_mem, 16, 0, "overlap_16B_lanes", false, false);
    CHECK(hipFree(dev_ramp));
    CHECK(hipFree(dev_mem));
}

template <int KIND, bool HOT, int N>
static void cost_one(Ctx & c) {
    const uint32_t iters = HOT ? 2048 : 2048 / N;  // cold: 8 MiB of fresh lines per repetition
    const uint32_t warm = HOT ? 16 : 0;
    std::vector<double> cyc;
    uint64_t bad = 0;
    const int reps = 6;
    for (int r = 0; r < reps; ++r) {
        size_t region_off = 0;
        if (!HOT) {
            do_flush(c);
            region_off = (size_t) r * (8u << 20);  // distinct 8 MiB parts of the 256 MiB region
        }
        hipLaunchKernelGGL((k_cost<KIND, HOT, N>), dim3(1), dim3(32), 0, 0, (const char *) (c.buf + region_off),
                           (uint32_t) (region_off / 4), iters, warm, c.cyc);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[3];
        CHECK(hipMemcpy(o, c.cyc, sizeof o, hipMemcpyDeviceToHost));
        if (r > 0) {
            cyc.push_back((double) o[0] / iters);
            bad += o[1];
        }
    }
    const Stats s = stats_of(cyc);
    // bytes moved per iteration: N blocks of 4 KiB (a 16-register block moves 2 KiB)
    const double bytes = (double) N * (KIND == BLK16 ? 2048 : 4096);
    std::printf("RESULT lds_block cost %s where=%s N=%d cyc_per_iter median=%.1f min=%.1f max=%.1f bytes_per_cycle median=%.2f "
                "min=%.2f max=%.2f last_block_data_mismatches=%llu\n",
                kind_name<KIND>(), HOT ? "hot" : "cold", N, s.median, s.min, s.max, bytes / s.median, bytes / s.max,
                bytes / s.min, (unsigned long long) bad);
    std::fflush(stdout);
    if (bad) {
        c.fail = true;
    }
}

template <int KIND, bool HOT>
static void cost_kind(Ctx & c) {
    cost_one<KIND, HOT, 1>(c);
    cost_one<KIND, HOT, 2>(c);
    cost_one<KIND, HOT, 4>(c);
    if constexpr (HOT) {
        cost_one<KIND, HOT, 8>(c);
    }
}

static void mode_cost(Ctx & c) {
    mem::clocks("start");
    cost_kind<BLK32, true>(c);
    cost_kind<BLK16, true>(c);
    cost_kind<B128, true>(c);
    cost_kind<B128C, true>(c);
    cost_kind<BLK32, false>(c);
    cost_kind<BLK16, false>(c);
    cost_kind<B128, false>(c);
    cost_kind<B128C, false>(c);
    mem::clocks("end");
    verdict("cost", c.fail ? "FAIL" : "PASS", c.fail ? "data of the last block did not match the ramp" : "last block's data matched the ramp in every configuration");
}

static void mode_calib(Ctx & c) {
    core::set_init({core::f32(1.0f), core::f32(1.0f), core::f32(0.0f)});
    const core::Cost k = core::pattern_cost(calib_fma<4>, calib_fma<6>, calib_fma<8>);
    std::printf("RESULT lds_block calib v_fma_f32 independent16 cycles_per_fma=%.3f lo=%.3f hi=%.3f loop_overhead_cyc_per_iter=%.2f "
                "linearity_dev_cyc_per_iter=%.2f\n",
                k.per, k.lo, k.hi, k.ovh, k.mid);
    const bool ok = k.per >= 0.95 && k.per <= 1.10;
    verdict("calib", ok ? "PASS" : "FAIL", "v_fma_f32, 16 independent chains: 1.00 cycle expected (0.95..1.10)");
    if (!ok) {
        c.fail = true;
    }
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: lds_block <calib|layout|cost> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode %s%s\n", prop.name, prop.gcnArchName, mode.c_str(), brk ? " --break" : "");
    std::fflush(stdout);
    Ctx c;
    CHECK(hipMalloc(&c.out, 4096));
    CHECK(hipMalloc(&c.cyc, 64));
    if (mode == "calib") {
        mode_calib(c);
    } else if (mode == "layout") {
        mode_layout(c, brk);
    } else if (mode == "cost") {
        CHECK(hipMalloc(&c.buf, 256u << 20));
        mem::fill(c.buf, 256u << 20, mem::RAMP);
        CHECK(hipMalloc(&c.flush, 512u << 20));
        mem::fill(c.flush, 512u << 20, mem::RANDOM);
        CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
        mode_cost(c);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    return c.fail ? 1 : 0;
}
