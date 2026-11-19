// Lane 3, items 1 and 2: LDS (shared memory) as a data store: cost of every access width, latency, accesses in
// flight, bank conflicts, store-to-load, several waves, size limits.
//
// Usage: lds_probe <calib|width|chain|inflight|pattern|waves|card|size> [--break]
//   calib     the FMA calibration (lds_common.h).
//   width     one wave: cost of each load and store form, 16 independent accesses per block (throughput) and each
//             access followed by s_wait_dscnt 0 (issue to data, the load-to-use latency). Lane-sequential addresses.
//   chain     one wave: dependent chains. LDS load whose data is the next address; load -> VALU add -> next address;
//             store then load of the same address (store-to-load) in the same wave.
//   inflight  one wave: N independent ds_load_b32 then one wait, N = 1..96 (the DS counter has 6 bits).
//   pattern   one wave, ds_load_b32/b64/b128: every lane the same address, sequential, strides, random; cost against
//             the bank model (32 banks of 4 bytes, bank = (byte address / 4) mod 32, one dword per bank per cycle,
//             same dword in one access counts once). --break replaces the colliding stride-32 pattern with stride 33
//             (no collision) while still expecting the collision cost: the verdict must FAIL.
//   waves     one work-group of 1 to 32 waves (16 per SIMD in CU mode is 32 waves): aggregate LDS read bytes per
//             cycle, b128 and b32, with each wave's SIMD recorded. Build with -mcumode for CU mode (lds_probe_cu).
//   card      whole card: every CU reading LDS with b128 (event timing, about 0.1 s).
//   size      LDS allocation limits: largest work-group allocation HIP accepts, work-groups per WGP against the
//             allocation (measured with spinning one-wave groups), and whether allocation size changes speed.
//
// Method: lane 1's differential cycle method (one wave alone; cost = (U=8 blocks - U=4 blocks) / (4 * ITERS * 16)),
// with the accesses written as inline assembly so the compiler adds nothing; the loop is checked in the listing.
// LDS is filled so every dword holds its own byte address (data = address), which makes a load usable as the next
// address and keeps values unchanged by any read order.
#include "lds_common.h"

#include <map>
#include <random>

// Built twice: default (WGP mode) and with -mcumode -DUSE_CU_KEY=1 (CU mode); the key decides how spinning groups
// are counted (per WGP or per CU).
#ifndef USE_CU_KEY
#define USE_CU_KEY 0
#endif

typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u3 __attribute__((ext_vector_type(3)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

enum Op {
    L_U8, L_U16, L_B32, L_B64, L_B96, L_B128, L_2A_B32, L_2A_B64,
    S_B8, S_B16, S_B32, S_B64, S_B96, S_B128, S_2A_B32, S_2A_B64,
    NOPS
};
static const char * op_name[NOPS] = {"load_u8", "load_u16", "load_b32", "load_b64", "load_b96", "load_b128", "load_2addr_b32",
                                     "load_2addr_b64", "store_b8", "store_b16", "store_b32", "store_b64", "store_b96",
                                     "store_b128", "store_2addr_b32", "store_2addr_b64"};
static const int op_bytes[NOPS] = {1, 2, 4, 8, 12, 16, 8, 16, 1, 2, 4, 8, 12, 16, 8, 16};  // per lane per access
__host__ __device__ constexpr uint32_t lane_stride_of(int op) {
    return (op == L_B64 || op == L_2A_B64 || op == S_B64 || op == S_2A_B64) ? 8
           : (op == L_B96 || op == L_B128 || op == S_B96 || op == S_B128) ? 16 : 4;
}

template <int OP>
struct Tr {
    using T = typename std::conditional<(OP == L_B64 || OP == L_2A_B32 || OP == S_B64 || OP == S_2A_B32), u2,
              typename std::conditional<(OP == L_B96 || OP == S_B96), u3,
              typename std::conditional<(OP == L_B128 || OP == L_2A_B64 || OP == S_B128 || OP == S_2A_B64), u4, unsigned>::type>::type>::type;
};

// One access. K selects a 512-byte row through the immediate offset (single-address forms) so the 16 accesses of a
// block touch different rows; the two-address forms read rows 0 and 1 of their unit (offset fields are 8 bits).
template <class T>
__device__ __forceinline__ uint32_t word0(const T & t) {
    if constexpr (sizeof(T) == 4) {
        return (uint32_t) t;
    } else {
        return (uint32_t) t[0];
    }
}

template <int OP, int K>
__device__ __forceinline__ void acc(typename Tr<OP>::T & d, uint32_t a) {
#define LDS_LD(ins) asm volatile(ins " %0, %1 offset:%2" : "+v"(d) : "v"(a), "i"(K * 512) : "memory")
#define LDS_ST(ins) asm volatile(ins " %0, %1 offset:%2" ::"v"(a), "v"(d), "i"(K * 512) : "memory")
    if constexpr (OP == L_U8) LDS_LD("ds_load_u8");
    else if constexpr (OP == L_U16) LDS_LD("ds_load_u16");
    else if constexpr (OP == L_B32) LDS_LD("ds_load_b32");
    else if constexpr (OP == L_B64) LDS_LD("ds_load_b64");
    else if constexpr (OP == L_B96) LDS_LD("ds_load_b96");
    else if constexpr (OP == L_B128) LDS_LD("ds_load_b128");
    else if constexpr (OP == L_2A_B32) asm volatile("ds_load_2addr_b32 %0, %1 offset0:0 offset1:32" : "+v"(d) : "v"(a) : "memory");
    else if constexpr (OP == L_2A_B64) asm volatile("ds_load_2addr_b64 %0, %1 offset0:0 offset1:32" : "+v"(d) : "v"(a) : "memory");
    else if constexpr (OP == S_B8) LDS_ST("ds_store_b8");
    else if constexpr (OP == S_B16) LDS_ST("ds_store_b16");
    else if constexpr (OP == S_B32) LDS_ST("ds_store_b32");
    else if constexpr (OP == S_B64) LDS_ST("ds_store_b64");
    else if constexpr (OP == S_B96) LDS_ST("ds_store_b96");
    else if constexpr (OP == S_B128) LDS_ST("ds_store_b128");
    else if constexpr (OP == S_2A_B32) asm volatile("ds_store_2addr_b32 %0, %1, %2 offset0:0 offset1:32" ::"v"(a), "v"(d.x), "v"(d.y) : "memory");
    else if constexpr (OP == S_2A_B64) asm volatile("ds_store_2addr_b64 %0, %1, %2 offset0:0 offset1:32" ::"v"(a), "v"(d.xy), "v"(d.zw) : "memory");
#undef LDS_LD
#undef LDS_ST
}

template <int OP, bool WAIT, int K>
__device__ __forceinline__ void block(typename Tr<OP>::T (&d)[16], uint32_t a) {
    if constexpr (K < 16) {
        acc<OP, K>(d[K], a);
        if constexpr (WAIT) {
            asm volatile("s_wait_dscnt 0" ::: "memory");
        }
        block<OP, WAIT, K + 1>(d, a);
    }
}

extern __shared__ uint32_t lds_mem[];

__device__ __forceinline__ void fill_lds(uint32_t bytes) {
    for (uint32_t i = threadIdx.x; i < bytes / 4; i += blockDim.x) {
        lds_mem[i] = i * 4;
    }
    __syncthreads();
}

// out[rep]: cycles of one repetition. addr: per-lane byte address.
template <int OP, bool WAIT, int U>
__global__ __launch_bounds__(32) void k_width(const uint32_t * __restrict__ addr, uint64_t * out, uint32_t * sink) {
    fill_lds(65536);
    using T = typename Tr<OP>::T;
    T d[16];
    for (int i = 0; i < 16; ++i) {
        d[i] = (T) (threadIdx.x * 4u);
    }
    const uint32_t a = addr[threadIdx.x];
    asm volatile("s_wait_dscnt 0" ::: "memory");
    for (int rep = 0; rep < core::REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < core::ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                block<OP, WAIT, 0>(d, a);
            }
        }
        asm volatile("s_wait_dscnt 0" ::: "memory");
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
        s ^= word0(d[i]);
    }
    sink[threadIdx.x] = s;
}

// ---------------------------------------------------------------------------------------------------------------
// Dependent chains (16 links per block, one register).
enum Chain { CH_LOAD, CH_VALU, CH_ST_LD, CH_VALU_ONLY, NCH };
static const char * ch_name[NCH] = {"load->address", "load->v_add->address", "store+load same address", "v_add alone"};

template <int CH>
__device__ __forceinline__ void link(uint32_t & x, uint32_t & t, uint32_t z) {
    if constexpr (CH == CH_LOAD) {
        asm volatile("ds_load_b32 %0, %0\n\ts_wait_dscnt 0" : "+v"(x)::"memory");
    } else if constexpr (CH == CH_VALU) {
        asm volatile("ds_load_b32 %1, %0\n\ts_wait_dscnt 0\n\tv_add_nc_u32 %0, %1, %2" : "+v"(x), "+v"(t) : "v"(z) : "memory");
    } else if constexpr (CH == CH_ST_LD) {
        // store the address, read it back, use it as the next address: store-to-load through LDS
        asm volatile("ds_store_b32 %0, %0\n\tds_load_b32 %0, %0\n\ts_wait_dscnt 0" : "+v"(x)::"memory");
    } else {
        asm volatile("v_add_nc_u32 %0, %0, %1" : "+v"(x) : "v"(z));
    }
}

template <int CH, int U>
__global__ __launch_bounds__(32) void k_chain(const uint32_t * __restrict__ addr, uint64_t * out, uint32_t * sink) {
    fill_lds(65536);
    uint32_t x = addr[threadIdx.x], t = 0, z = addr[64];  // addr[64] = 0, unknown to the compiler
    asm volatile("s_wait_dscnt 0" ::: "memory");
    for (int rep = 0; rep < core::REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < core::ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U * 16; ++u) {
                link<CH>(x, t, z);
            }
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    sink[threadIdx.x] = x ^ t;
}

// ---------------------------------------------------------------------------------------------------------------
// In flight: N independent ds_load_b32 then one wait, 64 batches.
template <int K, int N>
__device__ __forceinline__ void issue_b32(uint32_t (&r)[N], uint32_t a) {
    if constexpr (K < N) {
        asm volatile("ds_load_b32 %0, %1 offset:%2" : "=v"(r[K]) : "v"(a), "i"((K % 64) * 128) : "memory");
        issue_b32<K + 1, N>(r, a);
    }
}

template <int N>
__global__ __launch_bounds__(32) void k_inflight(uint32_t batches, uint64_t * out) {
    fill_lds(65536);
    uint32_t r[N];
    const uint32_t a = threadIdx.x * 4;
    uint32_t bad = 0;
    asm volatile("s_wait_dscnt 0" ::: "memory");
    const uint64_t t0 = wave_cycles();
#pragma unroll 1
    for (uint32_t b = 0; b < batches; ++b) {
        issue_b32<0, N>(r, a);
        asm volatile("s_wait_dscnt 0" ::: "memory");
#pragma unroll
        for (int k = 0; k < N; ++k) {
            asm volatile("" : "+v"(r[k]));
        }
    }
    const uint64_t t1 = wave_cycles();
#pragma unroll
    for (int k = 0; k < N; ++k) {
        bad += r[k] != a + (k % 64) * 128;
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (threadIdx.x == 0) {
        out[0] = cycle_diff(t0, t1);
        out[1] = bad;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Several waves: each wave runs `iters` blocks of 16 independent loads (no wait inside), then records cycles,
// REALTIME start and end, and its placement.
template <int OP>
__global__ __launch_bounds__(1024) void k_waves(uint32_t iters, uint32_t lds_bytes, uint64_t * out) {
    fill_lds(lds_bytes);
    using T = typename Tr<OP>::T;
    T d[16];
    for (int i = 0; i < 16; ++i) {
        d[i] = (T) 0u;
    }
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t a = lane * lane_stride_of(OP);
    asm volatile("s_wait_dscnt 0" ::: "memory");
    __syncthreads();
    const uint64_t r0 = mem::realtime();
    const uint64_t t0 = wave_cycles();
#pragma unroll 1
    for (uint32_t it = 0; it < iters; ++it) {
        block<OP, false, 0>(d, a);
    }
    asm volatile("s_wait_dscnt 0" ::: "memory");
    const uint64_t t1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
        s ^= word0(d[i]);
    }
    const uint32_t w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (lane == 0) {
        out[w * 4 + 0] = cycle_diff(t0, t1);
        out[w * 4 + 1] = r0;
        out[w * 4 + 2] = r1;
        out[w * 4 + 3] = mem::placement() | ((uint64_t) (s == 0x9e3779b9u) << 40);
    }
}

// Spinning one-wave groups: how many groups with a given LDS allocation sit on one WGP (or CU) at once.
__global__ __launch_bounds__(32) void k_spin(uint32_t * cur, uint32_t * mx, uint64_t spin_ticks) {
    fill_lds(1024);
    const uint32_t key = mem::pl_wgp(mem::placement()) | (USE_CU_KEY ? (mem::placement() & 512u) : 0u);
    if (threadIdx.x == 0) {
        const uint32_t now = atomicAdd(&cur[key], 1u) + 1;
        atomicMax(&mx[key], now);
        const uint64_t t0 = mem::realtime();
        while (mem::realtime() - t0 < spin_ticks) {
            __builtin_amdgcn_s_sleep(8);
        }
        atomicSub(&cur[key], 1u);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host

struct Bufs {
    uint32_t * addr = nullptr;
    uint64_t * out = nullptr;
    uint32_t * sink = nullptr;
};
static Bufs & bufs() {
    static Bufs b;
    if (!b.addr) {
        CHECK(hipMalloc(&b.addr, 128 * 4));
        CHECK(hipMalloc(&b.out, 1 << 20));
        CHECK(hipMalloc(&b.sink, 1024 * 4));
    }
    return b;
}
static void set_addr(const std::vector<uint32_t> & a) {
    std::vector<uint32_t> h(128, 0);
    std::copy(a.begin(), a.end(), h.begin());
    h[64] = 0;
    CHECK(hipMemcpy(bufs().addr, h.data(), h.size() * 4, hipMemcpyHostToDevice));
}

using WK = void (*)(const uint32_t *, uint64_t *, uint32_t *);
static std::vector<double> run_wave(WK k) {
    std::vector<double> v;
    for (int l = 0; l < core::LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 65536, 0, bufs().addr, bufs().out, bufs().sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t c[core::REPS];
        CHECK(hipMemcpy(c, bufs().out, sizeof c, hipMemcpyDeviceToHost));
        for (int r = 1; r < core::REPS; ++r) {
            v.push_back((double) c[r]);
        }
    }
    return v;
}
// Differential cost per access (or per chain link): accesses per block = 16.
static core::Cost diff_cost(WK lo, WK mid, WK hi) {
    const Stats a = stats_of(run_wave(lo)), m = stats_of(run_wave(mid)), b = stats_of(run_wave(hi));
    const double n = double(core::UHI - core::ULO) * core::ITERS * 16;
    const double per = (b.median - a.median) / n;
    return {per, (b.min - a.max) / n, (b.max - a.min) / n, 0, 0, (m.median - 0.5 * (a.median + b.median)) / core::ITERS, 0, 0, 0};
}

template <int OP, bool WAIT>
static core::Cost width_cost(const std::vector<uint32_t> & addr) {
    set_addr(addr);
    return diff_cost(k_width<OP, WAIT, core::ULO>, k_width<OP, WAIT, core::UMID>, k_width<OP, WAIT, core::UHI>);
}

static std::vector<uint32_t> seq(uint32_t stride) {
    std::vector<uint32_t> a(32);
    for (int l = 0; l < 32; ++l) a[l] = l * stride;
    return a;
}

static bool linear_ok(const core::Cost & c) { return std::fabs(c.mid) < 3.0; }

template <int OP>
static void width_one(int & nonlinear) {
    const auto a = seq(lane_stride_of(OP));
    const core::Cost t = width_cost<OP, false>(a), l = width_cost<OP, true>(a);
    nonlinear += !linear_ok(t) + !linear_ok(l);
    std::printf("RESULT lds width %s bytes_per_lane %d throughput_cycles %.2f lo %.2f hi %.2f bytes_per_cycle %.1f | "
                "issue_to_data_cycles %.2f lo %.2f hi %.2f | mid_dev %.2f %.2f\n",
                op_name[OP], op_bytes[OP], t.per, t.lo, t.hi, 32.0 * op_bytes[OP] / t.per, l.per, l.lo, l.hi, t.mid, l.mid);
}

static void mode_width() {
    int nl = 0;
    width_one<L_U8>(nl);
    width_one<L_U16>(nl);
    width_one<L_B32>(nl);
    width_one<L_B64>(nl);
    width_one<L_B96>(nl);
    width_one<L_B128>(nl);
    width_one<L_2A_B32>(nl);
    width_one<L_2A_B64>(nl);
    width_one<S_B8>(nl);
    width_one<S_B16>(nl);
    width_one<S_B32>(nl);
    width_one<S_B64>(nl);
    width_one<S_B96>(nl);
    width_one<S_B128>(nl);
    width_one<S_2A_B32>(nl);
    width_one<S_2A_B64>(nl);
    lds::check("lds_width_linear", nl == 0, "%d of 32 measurements not linear in the block count (U=6 off midpoint by 3+ cycles/iter)", nl);
}

template <int CH>
static core::Cost chain_cost(uint32_t & sink0) {
    const core::Cost c = diff_cost(k_chain<CH, core::ULO>, k_chain<CH, core::UMID>, k_chain<CH, core::UHI>);
    CHECK(hipMemcpy(&sink0, bufs().sink, 4, hipMemcpyDeviceToHost));
    return c;
}

static void mode_chain() {
    set_addr(seq(4));
    uint32_t s[NCH];
    core::Cost c[NCH] = {chain_cost<CH_LOAD>(s[0]), chain_cost<CH_VALU>(s[1]), chain_cost<CH_ST_LD>(s[2]), chain_cost<CH_VALU_ONLY>(s[3])};
    for (int i = 0; i < NCH; ++i) {
        std::printf("RESULT lds chain %s cycles_per_link %.2f lo %.2f hi %.2f mid_dev %.2f\n", ch_name[i], c[i].per, c[i].lo, c[i].hi, c[i].mid);
    }
    // lane 0: data = address = 0, so every chain must end where it started (load and store+load: x = 0; valu: x ^ t = 0)
    lds::check("lds_chain_values", s[0] == 0 && s[1] == 0 && s[2] == 0, "lane 0 chain end values %u %u %u (expected 0 0 0)", s[0], s[1], s[2]);
}

using IK = void (*)(uint32_t, uint64_t *);
template <int N>
static void inflight_one(int & bad) {
    std::vector<double> v;
    for (int rep = 0; rep < 6; ++rep) {
        hipLaunchKernelGGL(k_inflight<N>, dim3(1), dim3(32), 65536, 0, 256u, bufs().out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, bufs().out, 16, hipMemcpyDeviceToHost));
        bad += (int) o[1];
        if (rep) v.push_back(o[0] / 256.0);
    }
    const Stats s = stats_of(v);
    std::printf("RESULT lds inflight n %d cycles_per_batch %.1f min %.1f max %.1f per_load %.2f\n", N, s.median, s.min, s.max, s.median / N);
}

static void mode_inflight() {
    int bad = 0;
    inflight_one<1>(bad);
    inflight_one<2>(bad);
    inflight_one<4>(bad);
    inflight_one<8>(bad);
    inflight_one<16>(bad);
    inflight_one<24>(bad);
    inflight_one<32>(bad);
    inflight_one<48>(bad);
    inflight_one<56>(bad);
    inflight_one<62>(bad);
    inflight_one<63>(bad);
    inflight_one<64>(bad);
    inflight_one<65>(bad);
    inflight_one<72>(bad);
    inflight_one<96>(bad);
    lds::check("lds_inflight_values", bad == 0, "%d wrong values in the last batches (an early wait would show here)", bad);
}

// Bank model: 32 banks of one dword; an access of w bytes per lane touches w/4 dwords per lane; equal dwords merge.
// The lanes are served in groups of g lanes (g = 32: the whole wave at once); each group takes as many passes as
// its most used bank, and the groups add up.
static int bank_passes(const std::vector<uint32_t> & a, int w, int g = 32) {
    int total = 0;
    for (int l0 = 0; l0 < 32; l0 += g) {
        std::vector<std::vector<uint32_t>> per(32);
        for (int l = l0; l < l0 + g; ++l) {
            for (int j = 0; j < std::max(1, w / 4); ++j) {
                const uint32_t dw = a[l] / 4 + j;
                auto & v = per[dw % 32];
                if (std::find(v.begin(), v.end(), dw) == v.end()) v.push_back(dw);
            }
        }
        size_t m = 0;
        for (auto & v : per) m = std::max(m, v.size());
        total += (int) m;
    }
    return total;
}

struct Pat {
    std::string name;
    std::vector<uint32_t> a;    // the intended pattern (the model is computed from it)
    std::vector<uint32_t> run;  // the addresses actually used (differs only under --break)
};
static std::vector<Pat> patterns(int w, bool brk) {
    std::vector<Pat> p;
    std::mt19937 rng(17);
    auto add = [&](const std::string & n, const std::vector<uint32_t> & a) { p.push_back({n, a, a}); };
    add("same_address", std::vector<uint32_t>(32, 0));
    add("sequential", seq(w));
    for (int s : {2, 3, 4, 8, 16, 32, 33, 64}) {
        if (w == 4 || s <= 8 || s == 33) {
            add("stride_" + std::to_string(s), seq(s * w));
            if (brk && w == 4 && s == 32) p.back().run = seq(33 * 4);  // the break: the colliding pattern no longer collides
        }
    }
    {
        auto a = seq(w);
        a[1] = 128 * std::max(1, w / 4);  // lane 1 moved onto the banks of lane 0 (a different dword): one collision
        add("one_pair_collides", a);
    }
    {
        std::vector<uint32_t> a(32);
        for (int l = 0; l < 32; ++l) a[l] = (l % 16) * w;  // two half-waves read the same 16 elements
        add("halves_same", a);
    }
    for (int r = 0; r < 6; ++r) {
        std::vector<uint32_t> a(32);
        for (auto & x : a) x = (uint32_t) (rng() % (8192 / w)) * w;
        add("random_" + std::to_string(r), a);
    }
    return p;
}

template <int OP>
static int pattern_op(bool brk, std::vector<double> & ratios) {
    const int w = op_bytes[OP];
    int off_model = 0;
    for (const Pat & p : patterns(w, brk)) {
        const core::Cost t = width_cost<OP, false>(p.run), l = width_cost<OP, true>(p.run);
        const int m32 = bank_passes(p.a, w, 32), m16 = bank_passes(p.a, w, 16), m8 = bank_passes(p.a, w, 8);
        std::printf("RESULT lds pattern %s %s passes_wave %d passes_half %d passes_quarter %d throughput_cycles %.2f lo %.2f hi %.2f | "
                    "issue_to_data %.2f lo %.2f hi %.2f\n",
                    op_name[OP], p.name.c_str(), m32, m16, m8, t.per, t.lo, t.hi, l.per, l.lo, l.hi);
        if (OP == L_B64) {
            off_model += std::fabs(t.per - m16) > 0.1;  // 64-bit loads: each half-wave on its own
        }
        if (OP == L_B32) {
            off_model += std::fabs(t.per - m32) > 0.1;  // one cycle per pass for 32-bit loads
            if (p.name == "stride_32") ratios.push_back(t.per);
        }
    }
    return off_model;
}

static void mode_pattern(bool brk) {
    std::vector<double> r;
    const int off = pattern_op<L_B32>(brk, r);
    const int off64 = pattern_op<L_B64>(brk, r);
    pattern_op<L_B128>(brk, r);
    lds::check("lds_bank", off == 0 && off64 == 0, "ds_load_b32: %d of 18 patterns differ from the 32-bank model (one cycle per "
               "pass); ds_load_b64: %d of 15 differ from the same model applied to each half-wave; stride 32 dwords costs %.2f "
               "cycles (model 32)%s", off, off64, r.empty() ? -1.0 : r[0],
               brk ? " (break: the stride-32 pattern was run as stride 33)" : "");
}

template <int OP>
static void waves_one(uint32_t nw, uint32_t groups, uint32_t lds_bytes, const char * tag) {
    const uint32_t iters = 256;
    const uint32_t total = nw * groups;
    hipLaunchKernelGGL(k_waves<OP>, dim3(groups), dim3(32 * nw), lds_bytes, 0, iters, lds_bytes, bufs().out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> o(total * 4);
    CHECK(hipMemcpy(o.data(), bufs().out, o.size() * 8, hipMemcpyDeviceToHost));
    uint64_t cmax = 0, rmin = ~0ull, rmax = 0;
    std::vector<double> clk;
    std::map<uint32_t, int> simds, cus;
    for (uint32_t w = 0; w < total; ++w) {
        cmax = std::max(cmax, o[w * 4]);
        rmin = std::min(rmin, o[w * 4 + 1]);
        rmax = std::max(rmax, o[w * 4 + 2]);
        clk.push_back((double) o[w * 4] / ((o[w * 4 + 2] - o[w * 4 + 1]) * 10.0));  // GHz
        simds[mem::pl_simd((uint32_t) o[w * 4 + 3])]++;
        cus[mem::pl_cu((uint32_t) o[w * 4 + 3])]++;
    }
    const double bytes = (double) total * iters * 16 * 32 * op_bytes[OP];
    const double ghz = stats_of(clk).median;
    int mx = 0;
    for (auto & kv : simds) mx = std::max(mx, kv.second);
    std::printf("RESULT lds waves %s %s waves_per_group %u groups %u lds_bytes %u cus_used %zu simds_used %zu max_waves_per_simd %d "
                "bytes_per_cycle_slowest_wave %.1f bytes_per_cycle_wall %.1f clock_GHz %.2f GBps %.0f\n",
                tag, op_name[OP], nw, groups, lds_bytes, cus.size(), simds.size(), mx, bytes / cmax,
                bytes / ((rmax - rmin) * 10.0 * ghz), ghz, bytes / ((rmax - rmin) * 10.0));
}

static void mode_waves(const char * tag) {
    for (uint32_t nw : {1u, 2u, 4u, 8u, 16u, 32u}) {
        waves_one<L_B128>(nw, 1, 65536, tag);
    }
    for (uint32_t nw : {1u, 2u, 4u, 8u, 16u, 32u}) {
        waves_one<L_B32>(nw, 1, 65536, tag);
    }
    // allocation size does not matter for speed: one wave with 8 KiB against 64 KiB
    waves_one<L_B128>(1, 1, 8192, tag);
    waves_one<L_B128>(4, 1, 8192, tag);
}

static void mode_card(const char * tag) {
    if (!lds::thermal_gate("card")) {
        lds::check("lds_card_thermal", false, "thermal gate refused");
        return;
    }
    mem::clocks("card-before");
    // 2 groups of 16 waves per WGP-sized LDS budget: 32 KiB each, 512 groups
    waves_one<L_B128>(16, 512, 32768, tag);
    waves_one<L_B128>(16, 512, 32768, tag);
    waves_one<L_B32>(16, 512, 32768, tag);
    lds::after_full("card");
    mem::clocks("card-after");
}

static void mode_size(const char * tag) {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("RESULT lds size %s READ_runtime sharedMemPerBlock %zu maxSharedMemoryPerMultiProcessor %zu sharedMemPerBlockOptin %zu\n",
                tag, p.sharedMemPerBlock, p.maxSharedMemoryPerMultiProcessor, p.sharedMemPerBlockOptin);
    for (uint32_t b : {65536u, 65536u + 1024u, 131072u}) {
        hipLaunchKernelGGL(k_waves<L_B32>, dim3(1), dim3(32), b, 0, 1u, std::min(b, 65536u), bufs().out);
        const hipError_t e = hipGetLastError();
        const hipError_t e2 = hipDeviceSynchronize();
        std::printf("RESULT lds size %s launch_with_dynamic_lds %u -> %s / %s\n", tag, b, hipGetErrorString(e), hipGetErrorString(e2));
        (void) hipGetLastError();
    }
    uint32_t *cur, *mx;
    CHECK(hipMalloc(&cur, 2048 * 4));
    CHECK(hipMalloc(&mx, 2048 * 4));
    for (uint32_t b : {1024u, 16384u, 32768u, 49152u, 65536u}) {
        CHECK(hipMemset(cur, 0, 2048 * 4));
        CHECK(hipMemset(mx, 0, 2048 * 4));
        hipLaunchKernelGGL(k_spin, dim3(8192), dim3(32), b, 0, cur, mx, (uint64_t) 20000);  // 200 us each
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h(2048);
        CHECK(hipMemcpy(h.data(), mx, 2048 * 4, hipMemcpyDeviceToHost));
        std::map<uint32_t, int> hist;
        int units = 0;
        for (uint32_t v : h) {
            if (v) {
                hist[v]++;
                ++units;
            }
        }
        std::string s;
        for (auto & kv : hist) s += " " + std::to_string(kv.first) + "x" + std::to_string(kv.second);
        std::printf("RESULT lds size %s groups_at_once lds_bytes %u units %d (max one-wave groups at once : number of units)%s\n", tag, b, units, s.c_str());
    }
    CHECK(hipFree(cur));
    CHECK(hipFree(mx));
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "width";
    const bool brk = core::flag(argc, argv, "--break");
    const char * tag = USE_CU_KEY ? "cu_mode" : "wgp_mode";
    mem::clocks("start");
    if (mode == "calib") lds::calibrate(brk);
    else if (mode == "width") mode_width();
    else if (mode == "chain") mode_chain();
    else if (mode == "inflight") mode_inflight();
    else if (mode == "pattern") mode_pattern(brk);
    else if (mode == "waves") mode_waves(tag);
    else if (mode == "card") mode_card(tag);
    else if (mode == "size") mode_size(tag);
    else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::clocks("end");
    return lds::finish();
}
