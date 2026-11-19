// Lane 3, items 4 and 8: scalar memory loads and scratch (private memory) on gfx1201, one wave of 32 lanes.
//
// Usage: lds_smem <calib|chase|width|mixed|scratch|line|share|sbuf> [--break]
//   calib    16 independent v_fma_f32 per block, differential U=8 minus U=4 (lane 1's method): must be 1.00 cycle.
//   chase    scalar pointer chase (s_load_b64 + s_wait_kmcnt 0), one node per 64 or 128 bytes, against footprint.
//   width    independent scalar loads of each width from a hot buffer, N per batch then one wait; load-to-use latency.
//   mixed    N independent global_load_b128 that miss to memory, then a scalar load and its wait, then the vector wait.
//   scratch  scratch_store/scratch_load round trips and independent reloads (inline asm); two kernels that spill.
//   line     (leftovers C, LD-7a) the scalar cache's line size: load X (4 KiB aligned, fresh each trial), invalidate the
//            scalar cache, load X again (from the level behind it), then time a load of X + d: a hit means X + d came
//            in with X's line. d = 4 .. 512, from X and from X + 64. Calibration: the reload of X is a hit, the load
//            after the invalidation is not. --break expects the load after the invalidation to be a hit.
//   share    (leftovers C, LD-7b; build with -mcumode) 64 single-wave groups, one per CU (40 KiB of LDS each). Every
//            group chases its own fresh region once (touch); after a barrier each group chases, timed, the region of
//            a partner in a given relation (other CU of the same WGP, other WGP of the same shader array, other
//            array of the same shader engine, other engine), then a fresh region of its own (cold) and its own
//            touched region again (hot). Region sizes 4 KiB and 64 KiB. Which partners' lines come back faster than
//            a cold load shows which level the scalar path shares. --break expects the cold chase to be a hit.
//   sbuf     (leftovers C, LD-7c) s_buffer_load_b32/b64/b128/b256 through a buffer descriptor against s_load of the
//            same width, N = 1, 8, 16 per batch from the hot 4 KiB (the width mode's method); the values are checked.
//            Calibration: s_load_b32 and b128 reproduce item 4 (46.1 at N = 1; 69.1 and 75.1 at N = 16). --break
//            expects twice item 4's figures.
// --break  the calibration is told to expect 2.00 cycles per FMA, so VERDICT calib must FAIL (the leftovers modes
//          line, share and sbuf break their own check instead).
// Every timed region is inline assembly; the compiler cannot move uses across the waits because each destination
// is re-defined by an empty "+v"/"+s" statement after the wait (pattern of mem_inflight.hip).
#include "mem_common.h"

#include <map>
#include <set>

typedef unsigned su3 __attribute__((ext_vector_type(3)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u16v __attribute__((ext_vector_type(16)));

namespace {

constexpr int REPS = 8;
constexpr int ITERS = 128;

bool g_fail = false;
void vd(const char * name, bool ok, const char * text) {
    verdict(name, ok ? "PASS" : "FAIL", text);
    g_fail |= !ok;
}

// ---------------------------------------------------------------------------------------------------------------
// calib
#define FM(i) "v_fma_f32 %" #i ", %" #i ", %16, %17\n\t"
#define FMA16 FM(0) FM(1) FM(2) FM(3) FM(4) FM(5) FM(6) FM(7) FM(8) FM(9) FM(10) FM(11) FM(12) FM(13) FM(14) FM(15)
#define XO(i) "+v"(x[i])
template <int U>
__global__ __launch_bounds__(32) void k_calib(const uint32_t * in, uint64_t * out, float * sink) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = __uint_as_float(in[0]) + (float) i;
    }
    float k0 = __uint_as_float(in[1]), k1 = __uint_as_float(in[2]);
    asm volatile("" : "+v"(k0), "+v"(k1));
    for (int rep = 0; rep < REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                asm volatile(FMA16 : XO(0), XO(1), XO(2), XO(3), XO(4), XO(5), XO(6), XO(7), XO(8), XO(9), XO(10),
                             XO(11), XO(12), XO(13), XO(14), XO(15) : "v"(k0), "v"(k1));
            }
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    sink[threadIdx.x] = s;
}

template <int U>
std::vector<uint64_t> run_calib(uint32_t * in, uint64_t * out, float * sink) {
    std::vector<uint64_t> all;
    for (int l = 0; l < 3; ++l) {
        hipLaunchKernelGGL(k_calib<U>, dim3(1), dim3(32), 0, 0, in, out, sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[REPS];
        CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
        for (int r = 1; r < REPS; ++r) {
            all.push_back(o[r]);
        }
    }
    return all;
}

double calibrate(bool brk) {
    uint32_t *in;
    uint64_t * out;
    float * sink;
    CHECK(hipMalloc(&in, 64));
    CHECK(hipMalloc(&out, 64));
    CHECK(hipMalloc(&sink, 256));
    const float f[3] = {1.0f, 0.5f, 1.0f};  // x = x * 0.5 + 1 stays finite
    CHECK(hipMemcpy(in, f, sizeof f, hipMemcpyHostToDevice));
    const auto a = run_calib<4>(in, out, sink), b = run_calib<8>(in, out, sink);
    std::vector<double> per;
    for (size_t i = 0; i < a.size(); ++i) {
        per.push_back(((double) b[i] - (double) a[i]) / (4.0 * ITERS * 16));
    }
    const Stats s = stats_of(per);
    std::printf("RESULT lds_smem calib cycles_per_fma %.3f min %.3f max %.3f samples %zu\n", s.median, s.min, s.max,
                per.size());
    const double lo = brk ? 1.90 : 0.95, hi = brk ? 2.10 : 1.10;
    char t[160];
    std::snprintf(t, sizeof t, "v_fma_f32 x16 independent: %.3f cycles per FMA, accepted %.2f..%.2f%s", s.median, lo, hi,
                  brk ? " (deliberate break: expects 2.00)" : "");
    vd("calib", s.median >= lo && s.median <= hi, t);
    CHECK(hipFree(in));
    CHECK(hipFree(out));
    CHECK(hipFree(sink));
    return s.median;
}

// ---------------------------------------------------------------------------------------------------------------
// shared host context
struct Ctx {
    char * ramp = nullptr;   // ramp of 32-bit words: word i holds i
    char * flush = nullptr;  // read to push everything out of L2 and the memory-side cache
    uint64_t * out = nullptr;
    unsigned * sink = nullptr;
};

__global__ void k_flush(const u4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        acc += p[i].x;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
constexpr uint32_t FLUSH_BYTES = 256u << 20;
void do_flush(Ctx & c) {
    hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const u4 *) c.flush, FLUSH_BYTES / 16, c.sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------------------------------------------
// chase
#define CHASE1 asm volatile("s_load_b64 %0, %0, 0x0\n\ts_wait_kmcnt 0" : "+s"(p)::"memory");
__global__ __launch_bounds__(32) void k_chase(uint64_t start, uint32_t warm, uint32_t steps, uint32_t reps,
                                              uint32_t inval, uint64_t * out) {
    uint64_t p = start;
    if (inval) {
        asm volatile("s_dcache_inv\n\ts_wait_kmcnt 0" ::: "memory");
    }
#pragma unroll 1
    for (uint32_t w = 0; w < warm; w += 4) {
        CHASE1 CHASE1 CHASE1 CHASE1
    }
    for (uint32_t r = 0; r < reps; ++r) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (uint32_t s = 0; s < steps; s += 4) {
            CHASE1 CHASE1 CHASE1 CHASE1
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[r] = cycle_diff(t0, t1);
        }
    }
    if (threadIdx.x == 0) {
        out[16] = p;
    }
}

struct ChaseRow {
    size_t footprint;
    size_t stride;
    double med;
};

ChaseRow chase_one(Ctx & c, char * buf, size_t footprint, size_t stride, bool cold, bool & ok) {
    const size_t nodes = footprint / stride;
    std::vector<uint32_t> order;
    const uint64_t start = mem::build_chase(buf, nodes, stride, true, 7, &order);
    const uint32_t S = cold ? 4096 : 8192, reps = cold ? 1 : 6;
    const uint32_t warm = cold ? 0 : (uint32_t) ((nodes + 3) / 4 * 4);
    if (cold) {
        do_flush(c);
    }
    hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, warm, S, reps, cold ? 1u : 0u, c.out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint64_t o[17];
    CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
    const uint64_t total = (uint64_t) warm + (uint64_t) reps * S;
    const uint64_t want = (uint64_t) buf + (uint64_t) order[total % nodes] * stride;
    const bool good = o[16] == want;
    ok &= good;
    std::vector<double> v;
    for (uint32_t r = cold ? 0 : 1; r < reps; ++r) {
        v.push_back((double) o[r] / S);
    }
    const Stats s = stats_of(v);
    std::printf("RESULT lds_smem chase mode %s stride %zu footprint_KiB %.0f nodes %zu cycles_per_load %.1f min %.1f "
                "max %.1f final_pointer %s\n",
                cold ? "cold" : "warm", stride, footprint / 1024.0, nodes, s.median, s.min, s.max, good ? "ok" : "WRONG");
    std::fflush(stdout);
    return {footprint, stride, s.median};
}

void mode_chase(Ctx & c) {
    const size_t MAXB = 64ull << 20;
    char * buf;
    CHECK(hipMalloc(&buf, MAXB));
    const size_t sizes[] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 1024, 4096, 16384, 65536};
    bool ok = true;
    for (size_t stride : {32, 64, 128, 256}) {
        std::vector<ChaseRow> rows;
        for (size_t kb : sizes) {
            rows.push_back(chase_one(c, buf, kb * 1024, stride, false, ok));
        }
        const double low = rows[0].med;
        size_t last_low = rows[0].footprint, first_high = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].med > 1.3 * low) {
                first_high = rows[i].footprint;
                break;
            }
            last_low = rows[i].footprint;
        }
        std::printf("RESULT lds_smem chase_step stride %zu base_cycles %.1f last_footprint_KiB_below_1.3x %.0f "
                    "first_footprint_KiB_above_1.3x %.0f lines_at_last_low %zu\n",
                    stride, low, last_low / 1024.0, first_high / 1024.0, last_low / stride);
    }
    for (int rep = 0; rep < 3; ++rep) {
        chase_one(c, buf, 256u << 10, 64, true, ok);
        chase_one(c, buf, 64ull << 20, 64, true, ok);
    }
    vd("chase", ok, "final pointer of every chase equals the host's prediction (order[total % nodes])");
    CHECK(hipFree(buf));
}

// ---------------------------------------------------------------------------------------------------------------
// width
enum { B32, B64, B96, B128, B256, B512, U8, U16 };
template <int T>
struct ST;
template <> struct ST<B32> { using type = unsigned; static constexpr int dw = 1; static constexpr const char * name = "b32"; };
template <> struct ST<B64> { using type = uint64_t; static constexpr int dw = 2; static constexpr const char * name = "b64"; };
template <> struct ST<B96> { using type = su3; static constexpr int dw = 3; static constexpr const char * name = "b96"; };
template <> struct ST<B128> { using type = u4; static constexpr int dw = 4; static constexpr const char * name = "b128"; };
template <> struct ST<B256> { using type = u8v; static constexpr int dw = 8; static constexpr const char * name = "b256"; };
template <> struct ST<B512> { using type = u16v; static constexpr int dw = 16; static constexpr const char * name = "b512"; };
template <> struct ST<U8> { using type = unsigned; static constexpr int dw = 1; static constexpr const char * name = "u8"; };
template <> struct ST<U16> { using type = unsigned; static constexpr int dw = 1; static constexpr const char * name = "u16"; };

// Each load reads its own 64-byte line of the hot 4 KiB buffer: offset K * 64.
template <int T, int K>
__device__ __forceinline__ void sld(typename ST<T>::type & r, const char * base) {
#define SL(OP) asm volatile(OP " %0, %1, %2" : "=s"(r) : "s"(base), "i"(K * 64) : "memory")
    if constexpr (T == B32) SL("s_load_b32");
    else if constexpr (T == B64) SL("s_load_b64");
    else if constexpr (T == B96) SL("s_load_b96");
    else if constexpr (T == B128) SL("s_load_b128");
    else if constexpr (T == B256) SL("s_load_b256");
    else if constexpr (T == B512) SL("s_load_b512");
    else if constexpr (T == U8) SL("s_load_u8");
    else SL("s_load_u16");
#undef SL
}

template <int T, int K, int N>
__device__ __forceinline__ void sissue(typename ST<T>::type (&r)[N], const char * base) {
    if constexpr (K < N) {
        sld<T, K>(r[K], base);
        sissue<T, K + 1, N>(r, base);
    }
}

template <class V>
__device__ __forceinline__ uint32_t dword(const V & v, int d) {
    uint32_t t[16];
    __builtin_memcpy(t, &v, sizeof(V) < 64 ? sizeof(V) : 64);
    return t[d];
}

// out: [0] cycles for `batches` batches, [1] mismatching words in the last batch.
template <int T, int N>
__global__ __launch_bounds__(32) void k_width(const char * base, uint32_t batches, uint64_t * out) {
    using V = typename ST<T>::type;
    V r[N > 0 ? N : 1];
    // untimed batch so the 4 KiB is in the scalar cache and the instruction cache holds the loop
    if constexpr (N > 0) {
        sissue<T, 0, N>(r, base);
    }
    asm volatile("s_wait_kmcnt 0" ::: "memory");
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" : "+s"(r[k]));
    }
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t b = 0; b < batches; ++b) {
        if constexpr (N > 0) {
            sissue<T, 0, N>(r, base);
        }
        asm volatile("s_wait_kmcnt 0" ::: "memory");
#pragma unroll
        for (int k = 0; k < N; ++k) {
            asm volatile("" : "+s"(r[k]));
        }
    }
    const uint64_t c1 = wave_cycles();
    uint32_t bad = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        const uint32_t w0 = k * 16;  // word index of byte offset k * 64
        if constexpr (T == U8) {
            bad += (dword(r[k], 0) != (w0 & 0xffu));
        } else if constexpr (T == U16) {
            bad += (dword(r[k], 0) != (w0 & 0xffffu));
        } else {
#pragma unroll
            for (int d = 0; d < ST<T>::dw; ++d) {
                bad += (dword(r[k], d) != w0 + d);
            }
        }
    }
    if (threadIdx.x == 0) {
        out[0] = cycle_diff(c0, c1);
        out[1] = bad;
    }
}

struct WRow {
    double med;
    bool ok;
};

template <int T, int N>
WRow width_one(Ctx & c, double empty, bool print) {
    const uint32_t batches = 256;
    std::vector<double> cyc;
    uint64_t bad = 0;
    for (int t = 0; t < 6; ++t) {
        hipLaunchKernelGGL((k_width<T, N>), dim3(1), dim3(32), 0, 0, (const char *) c.ramp, batches, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        if (t > 0) {
            cyc.push_back((double) o[0] / batches);
        }
        bad += o[1];
    }
    const Stats s = stats_of(cyc);
    if (print) {
        const double net = s.median - empty;
        const int bytes = T == U8 ? 1 : T == U16 ? 2 : ST<T>::dw * 4;
        std::printf("RESULT lds_smem width type %s N %d cycles_per_batch %.1f min %.1f max %.1f net_of_empty_loop %.1f "
                    "cycles_per_load %.2f bytes_per_cycle %.2f wrong_words %llu\n",
                    ST<T>::name, N, s.median, s.min, s.max, net, N ? net / N : 0.0, N ? N * bytes / net : 0.0,
                    (unsigned long long) bad);
        std::fflush(stdout);
    }
    return {s.median, bad == 0};
}

template <int T, int... Ns>
bool width_list(Ctx & c, double empty) {
    bool ok = true;
    ((ok &= width_one<T, Ns>(c, empty, true).ok), ...);
    return ok;
}

// Load-to-use. MODE 0: load + wait. 1: + v_mov from the SGPR + dependent v_add. 2: loop-carried round trip: the
// loaded value goes through v_mov and v_readfirstlane and becomes the SGPR offset of the next load. 3: as 2 plus a
// v_add of zero between.
template <int MODE>
__global__ __launch_bounds__(32) void k_lat(const char * base, uint64_t * out) {
    unsigned s = 0, v = 1;
    for (int rep = 0; rep < REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if constexpr (MODE == 0) {
                    asm volatile("s_load_b32 %0, %1, 0x40\n\ts_wait_kmcnt 0" : "=s"(s) : "s"(base) : "memory");
                } else if constexpr (MODE == 1) {
                    asm volatile("s_load_b32 %0, %2, 0x40\n\ts_wait_kmcnt 0\n\tv_mov_b32 %1, %0\n\tv_add_nc_u32 %1, %1, %0"
                                 : "=s"(s), "+v"(v) : "s"(base) : "memory");
                } else if constexpr (MODE == 2) {
                    asm volatile("s_load_b32 %0, %2, %0\n\ts_wait_kmcnt 0\n\tv_mov_b32 %1, %0\n\tv_readfirstlane_b32 %0, %1"
                                 : "+s"(s), "=&v"(v) : "s"(base) : "memory");
                } else {
                    asm volatile("s_load_b32 %0, %2, %0\n\ts_wait_kmcnt 0\n\tv_mov_b32 %1, %0\n\tv_add_nc_u32 %1, 0, %1\n\t"
                                 "v_readfirstlane_b32 %0, %1" : "+s"(s), "=&v"(v) : "s"(base) : "memory");
                }
            }
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    if (threadIdx.x == 0) {
        out[16] = s;
        out[17] = v;
    }
}

template <int MODE>
double lat_one(Ctx & c, const char * what, bool & ok) {
    std::vector<double> all;
    uint64_t o[18] = {};
    for (int l = 0; l < 3; ++l) {
        hipLaunchKernelGGL(k_lat<MODE>, dim3(1), dim3(32), 0, 0, (const char *) c.ramp, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        for (int r = 1; r < REPS; ++r) {
            all.push_back((double) o[r] / (ITERS * 4));
        }
    }
    const uint64_t want = MODE < 2 ? 16 : 0;  // word 16 is at byte offset 0x40; word 0 holds 0
    ok &= o[16] == want;
    const Stats s = stats_of(all);
    std::printf("RESULT lds_smem lat %s cycles_per_step %.2f min %.2f max %.2f final_s %llu (want %llu)\n", what, s.median,
                s.min, s.max, (unsigned long long) o[16], (unsigned long long) want);
    std::fflush(stdout);
    return s.median;
}

void mode_width(Ctx & c) {
    mem::fill(c.ramp, 1u << 20, mem::RAMP);
    const double empty = width_one<B32, 0>(c, 0, false).med;
    std::printf("RESULT lds_smem width empty_batch_cycles %.1f (one s_wait_kmcnt 0 with nothing outstanding + loop)\n",
                empty);
    bool ok = true;
    ok &= width_list<B32, 1, 2, 4, 8, 16, 31>(c, empty);
    ok &= width_list<B64, 1, 2, 4, 8, 16, 31>(c, empty);
    ok &= width_list<B96, 1, 2, 4, 8, 16>(c, empty);
    ok &= width_list<B128, 1, 2, 4, 8, 16>(c, empty);
    ok &= width_list<B256, 1, 2, 4, 8>(c, empty);
    ok &= width_list<B512, 1, 2, 4>(c, empty);
    ok &= width_list<U8, 1, 2, 4, 8, 16, 31>(c, empty);
    ok &= width_list<U16, 1, 2, 4, 8, 16, 31>(c, empty);
    vd("width", ok, "every loaded value of every width equals the ramp data at its offset (b96 N>16, b128 N>16, b256 N>8, b512 N>4 not run: SGPR budget)");
    bool lok = true;
    const double a = lat_one<0>(c, "load_wait", lok);
    const double b = lat_one<1>(c, "load_wait_vmov_vadd", lok);
    const double d = lat_one<2>(c, "load_wait_vmov_readfirstlane_to_soffset", lok);
    const double e = lat_one<3>(c, "load_wait_vmov_vadd_readfirstlane_to_soffset", lok);
    std::printf("RESULT lds_smem lat_summary load_wait %.2f plus_vmov_vadd_issue %.2f round_trip_vmov_rfl %.2f "
                "round_trip_vmov_vadd_rfl %.2f extra_cost_of_vector_round_trip %.2f\n", a, b, d, e, d - a);
    vd("lat", lok, "scalar value carried through the vector unit and back is intact");
}

// ---------------------------------------------------------------------------------------------------------------
// mixed
template <int K>
__device__ __forceinline__ void vld(u4 & r, const char * base, uint32_t lane_off) {
    asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 512) : "memory");
}
template <int K, int N>
__device__ __forceinline__ void vissue(u4 (&r)[N], const char * base, uint32_t lane_off) {
    if constexpr (K < N) {
        vld<K>(r[K], base, lane_off);
        vissue<K + 1, N>(r, base, lane_off);
    }
}

constexpr int MREPS = 9;  // the first is a warm-up and is dropped on the host
// out[rep * 4 + j]: j0 issue time of the vector loads, j1 scalar load plus wait, j2 rest of the vector wait, j3 =
// wrong values; out[MREPS * 4] total wrong.
template <int N, int COLD>
__global__ __launch_bounds__(32) void k_mixed(const char * vbase, const char * sbase, uint32_t vword0, uint32_t sword0,
                                               uint64_t * out) {
    u4 r[N > 0 ? N : 1];
    uint32_t d0[MREPS], d1[MREPS], d2[MREPS];
    uint32_t bad = 0;
    const uint32_t lane_off = (threadIdx.x % 32) * 16;
#pragma unroll
    for (int rep = 0; rep < MREPS; ++rep) {
        const char * vp = vbase + (size_t) rep * (N > 0 ? N : 1) * 512;
        const char * sp = sbase + (COLD ? (size_t) rep * 4096 : (size_t) 0);
        asm volatile("" : "+s"(sp), "+s"(vp));
        unsigned s;
        if constexpr (!COLD) {
            asm volatile("s_load_b32 %0, %1, 0x0\n\ts_wait_kmcnt 0" : "=s"(s) : "s"(sp) : "memory");
        }
        const uint64_t t0 = wave_cycles();
        if constexpr (N > 0) {
            vissue<0, N>(r, vp, lane_off);
        }
        const uint64_t t1 = wave_cycles();
        asm volatile("s_load_b32 %0, %1, 0x0\n\ts_wait_kmcnt 0" : "=s"(s) : "s"(sp) : "memory");
        const uint64_t t2 = wave_cycles();
        asm volatile("s_wait_loadcnt 0" ::: "memory");
        const uint64_t t3 = wave_cycles();
        if constexpr (N > 0) {
#pragma unroll
            for (int k = 0; k < N; ++k) {
                asm volatile("" : "+v"(r[k]));
            }
            bad += (r[N - 1].x != vword0 + rep * N * 128 + (N - 1) * 128 + (threadIdx.x % 32) * 4) +
                   (r[0].w != vword0 + rep * N * 128 + (threadIdx.x % 32) * 4 + 3);
        }
        bad += s != sword0 + (COLD ? rep * 1024 : 0);
        d0[rep] = (uint32_t) (t1 - t0);
        d1[rep] = (uint32_t) (t2 - t1);
        d2[rep] = (uint32_t) (t3 - t2);
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int rep = 0; rep < MREPS; ++rep) {
            out[rep * 4 + 0] = d0[rep];
            out[rep * 4 + 1] = d1[rep];
            out[rep * 4 + 2] = d2[rep];
        }
        out[MREPS * 4] = bad;
    }
}

struct MRes {
    double issue, scalar, rest;
};

template <int N, int COLD>
MRes mixed_one(Ctx & c, size_t & vcur, size_t & scur, bool & ok) {
    std::vector<double> a, b, d;
    uint64_t o[MREPS * 4 + 1];
    for (int l = 0; l < 5; ++l) {
        do_flush(c);
        const size_t vbytes = (size_t) MREPS * (N ? N : 1) * 512;
        hipLaunchKernelGGL((k_mixed<N, COLD>), dim3(1), dim3(32), 0, 0, (const char *) (c.ramp + vcur),
                           (const char *) (c.ramp + (64u << 20) + (COLD ? scur : 0)), (uint32_t) (vcur / 4),
                           (uint32_t) ((((size_t) 64 << 20) + (COLD ? scur : 0)) / 4), c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        vcur += vbytes;
        scur += (size_t) MREPS * 4096;
        ok &= o[MREPS * 4] == 0;
        for (int r = 1; r < MREPS; ++r) {
            a.push_back((double) o[r * 4]);
            b.push_back((double) o[r * 4 + 1]);
            d.push_back((double) o[r * 4 + 2]);
        }
    }
    const Stats sa = stats_of(a), sb = stats_of(b), sd = stats_of(d);
    std::printf("RESULT lds_smem mixed vector_loads %d scalar %s issue_cycles %.0f scalar_load_wait_cycles %.0f "
                "(min %.0f max %.0f) rest_of_vector_wait_cycles %.0f (min %.0f max %.0f) wrong %s\n",
                N, COLD ? "cold" : "hot", sa.median, sb.median, sb.min, sb.max, sd.median, sd.min, sd.max,
                ok ? "0" : "YES");
    std::fflush(stdout);
    return {sa.median, sb.median, sd.median};
}

void mode_mixed(Ctx & c) {
    mem::fill(c.ramp, 128u << 20, mem::RAMP);
    size_t vcur = 4096, scur = 0;  // never reused: every run reads lines no earlier run touched
    bool ok = true;
    const MRes h0 = mixed_one<0, 0>(c, vcur, scur, ok);
    const MRes h16 = mixed_one<16, 0>(c, vcur, scur, ok);
    const MRes h32 = mixed_one<32, 0>(c, vcur, scur, ok);
    const MRes c0 = mixed_one<0, 1>(c, vcur, scur, ok);
    const MRes c16 = mixed_one<16, 1>(c, vcur, scur, ok);
    const MRes c32 = mixed_one<32, 1>(c, vcur, scur, ok);
    std::printf("RESULT lds_smem mixed_summary hot_scalar_alone %.0f with16 %.0f with32 %.0f | cold_scalar_alone %.0f "
                "with16 %.0f with32 %.0f | vector_wait_after_scalar16 %.0f vector_wait_after_scalar32 %.0f\n",
                h0.scalar, h16.scalar, h32.scalar, c0.scalar, c16.scalar, c32.scalar, h16.rest, h32.rest);
    char t[300];
    // The vector loads must really have been in flight (their wait took long) and the data must be right.
    const bool real = h16.issue + h16.scalar + h16.rest > 300 && h32.issue + h32.scalar + h32.rest > 300;
    std::snprintf(t, sizeof t, "values correct; vector batch took %.0f (N=16) and %.0f (N=32) cycles from first issue to last data",
                  h16.issue + h16.scalar + h16.rest, h32.issue + h32.scalar + h32.rest);
    vd("mixed", ok && real, t);
}

// ---------------------------------------------------------------------------------------------------------------
// scratch
// MODE 0 b32 store+reload chain; 1 same with s_wait_storecnt 0 between; 2 b128 chain; 3 baseline: only the dependent
// add; 4 / 5: 16 independent reloads of b32 / b128 then one wait; 6 baseline: one s_wait_loadcnt 0 per batch.
// out: [rep] cycles; [16] result word checks (mismatches).
constexpr int SITERS = 256;
#define SB32 "scratch_store_b32 off, %0, off offset:512\n\tscratch_load_b32 %0, off, off offset:512\n\ts_wait_loadcnt 0\n\t"
#define SB32W "scratch_store_b32 off, %0, off offset:512\n\ts_wait_storecnt 0\n\tscratch_load_b32 %0, off, off offset:512\n\ts_wait_loadcnt 0\n\t"
#define SB128 "scratch_store_b128 off, %0, off offset:512\n\tscratch_load_b128 %0, off, off offset:512\n\ts_wait_loadcnt 0\n\t"
template <int MODE>
__global__ __launch_bounds__(32) void k_scr(uint32_t flag, uint64_t * out, uint32_t * res) {
    // Never executed (flag is 0): its only job is to make the compiler allocate a private segment, so the kernel
    // is launched with scratch and its flat-scratch base is set up. The inline-assembly accesses below use
    // offsets 0..1023 of the lane's own region and run only when flag is 0, so they cannot clash with it.
    volatile uint32_t keep[256];
    if (flag) {
        keep[threadIdx.x] = flag;
        res[1] = keep[flag & 255];
    }
    const uint32_t lane = threadIdx.x % 32;
    uint32_t bad = 0;
    unsigned x = lane * 3 + 1;
    u4 xv = {lane * 3 + 1, lane * 3 + 2, lane * 3 + 3, lane * 3 + 4};
    if constexpr (MODE == 0 || MODE == 1) {
        asm volatile("scratch_store_b32 off, %0, off offset:512\n\ts_wait_storecnt 0" ::"v"(x) : "memory");
    } else if constexpr (MODE == 2) {
        asm volatile("scratch_store_b128 off, %0, off offset:512\n\ts_wait_storecnt 0" ::"v"(xv) : "memory");
    }
    uint32_t total_adds = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < SITERS / 4; ++it) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if constexpr (MODE == 0) {
                    asm volatile(SB32 "v_add_nc_u32 %0, %0, 1" : "+v"(x)::"memory");
                } else if constexpr (MODE == 1) {
                    asm volatile(SB32W "v_add_nc_u32 %0, %0, 1" : "+v"(x)::"memory");
                } else if constexpr (MODE == 2) {
                    asm volatile(SB128 "" : "+v"(xv)::"memory");
                    xv.x += 1;
                } else if constexpr (MODE == 3) {
                    asm volatile("v_add_nc_u32 %0, %0, 1" : "+v"(x));
                }
            }
        }
        const uint64_t t1 = wave_cycles();
        total_adds += SITERS;
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    if constexpr (MODE == 0 || MODE == 1 || MODE == 3) {
        bad += x != lane * 3 + 1 + total_adds;
    } else if constexpr (MODE == 2) {
        bad += (xv.x != lane * 3 + 1 + total_adds) + (xv.w != lane * 3 + 4);
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (threadIdx.x == 0) {
        res[0] = bad;
        out[16] = x;
    }
}

// Throughput of 16 independent reloads. B128: each load is 16 bytes at offset K * 16, else 4 bytes at K * 4.
template <int B128, int LOADS>
__global__ __launch_bounds__(32) void k_scr_tp(uint32_t flag, uint64_t * out, uint32_t * res) {
    volatile uint32_t keep[256];
    if (flag) {
        keep[threadIdx.x] = flag;
        res[1] = keep[flag & 255];
    }
    const uint32_t lane = threadIdx.x % 32;
    uint32_t v[16];
    u4 w[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        v[k] = lane * 100 + k;
        w[k] = {lane * 100 + k * 4, lane * 100 + k * 4 + 1, lane * 100 + k * 4 + 2, lane * 100 + k * 4 + 3};
    }
    // offsets: K = 16 * j for b128 and 4 * j for b32, j = 0..15
    if constexpr (B128) {
        asm volatile("scratch_store_b128 off, %0, off offset:0\n\tscratch_store_b128 off, %1, off offset:16\n\t"
                     "scratch_store_b128 off, %2, off offset:32\n\tscratch_store_b128 off, %3, off offset:48\n\t"
                     "scratch_store_b128 off, %4, off offset:64\n\tscratch_store_b128 off, %5, off offset:80\n\t"
                     "scratch_store_b128 off, %6, off offset:96\n\tscratch_store_b128 off, %7, off offset:112\n\t"
                     "scratch_store_b128 off, %8, off offset:128\n\tscratch_store_b128 off, %9, off offset:144\n\t"
                     "scratch_store_b128 off, %10, off offset:160\n\tscratch_store_b128 off, %11, off offset:176\n\t"
                     "scratch_store_b128 off, %12, off offset:192\n\tscratch_store_b128 off, %13, off offset:208\n\t"
                     "scratch_store_b128 off, %14, off offset:224\n\tscratch_store_b128 off, %15, off offset:240\n\t"
                     "s_wait_storecnt 0" ::"v"(w[0]), "v"(w[1]), "v"(w[2]), "v"(w[3]), "v"(w[4]), "v"(w[5]), "v"(w[6]),
                     "v"(w[7]), "v"(w[8]), "v"(w[9]), "v"(w[10]), "v"(w[11]), "v"(w[12]), "v"(w[13]), "v"(w[14]),
                     "v"(w[15]) : "memory");
    } else {
        asm volatile("scratch_store_b32 off, %0, off offset:0\n\tscratch_store_b32 off, %1, off offset:4\n\t"
                     "scratch_store_b32 off, %2, off offset:8\n\tscratch_store_b32 off, %3, off offset:12\n\t"
                     "scratch_store_b32 off, %4, off offset:16\n\tscratch_store_b32 off, %5, off offset:20\n\t"
                     "scratch_store_b32 off, %6, off offset:24\n\tscratch_store_b32 off, %7, off offset:28\n\t"
                     "scratch_store_b32 off, %8, off offset:32\n\tscratch_store_b32 off, %9, off offset:36\n\t"
                     "scratch_store_b32 off, %10, off offset:40\n\tscratch_store_b32 off, %11, off offset:44\n\t"
                     "scratch_store_b32 off, %12, off offset:48\n\tscratch_store_b32 off, %13, off offset:52\n\t"
                     "scratch_store_b32 off, %14, off offset:56\n\tscratch_store_b32 off, %15, off offset:60\n\t"
                     "s_wait_storecnt 0" ::"v"(v[0]), "v"(v[1]), "v"(v[2]), "v"(v[3]), "v"(v[4]), "v"(v[5]), "v"(v[6]),
                     "v"(v[7]), "v"(v[8]), "v"(v[9]), "v"(v[10]), "v"(v[11]), "v"(v[12]), "v"(v[13]), "v"(v[14]),
                     "v"(v[15]) : "memory");
    }
    uint32_t bad = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        u4 r[16];
        unsigned q[16];
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < 64; ++it) {
            if constexpr (LOADS) {
                if constexpr (B128) {
#define L(K) asm volatile("scratch_load_b128 %0, off, off offset:" #K : "=v"(r[K / 16]) : : "memory");
                    L(0) L(16) L(32) L(48) L(64) L(80) L(96) L(112) L(128) L(144) L(160) L(176) L(192) L(208) L(224) L(240)
#undef L
                } else {
#define L(K) asm volatile("scratch_load_b32 %0, off, off offset:" #K : "=v"(q[K / 4]) : : "memory");
                    L(0) L(4) L(8) L(12) L(16) L(20) L(24) L(28) L(32) L(36) L(40) L(44) L(48) L(52) L(56) L(60)
#undef L
                }
            }
            asm volatile("s_wait_loadcnt 0" ::: "memory");
            if constexpr (LOADS) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    if constexpr (B128) {
                        asm volatile("" : "+v"(r[k]));
                    } else {
                        asm volatile("" : "+v"(q[k]));
                    }
                }
            }
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
        if constexpr (LOADS) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if constexpr (B128) {
                    bad += (r[k].x != w[k].x) + (r[k].w != w[k].w);
                } else {
                    bad += q[k] != v[k];
                }
            }
        }
    }
    for (int m = 16; m >= 1; m /= 2) {
        bad += __shfl_xor(bad, m);
    }
    if (threadIdx.x == 0) {
        res[0] = bad;
    }
}

template <class K>
double scr_run(K kern, const char * what, double div, Ctx & c, uint32_t * res, bool & ok) {
    std::vector<double> all;
    uint64_t o[17] = {};
    uint32_t h[2] = {0, 0};
    for (int l = 0; l < 3; ++l) {
        hipLaunchKernelGGL(kern, dim3(1), dim3(32), 0, 0, 0u, c.out, res);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(h, res, sizeof h, hipMemcpyDeviceToHost));
        ok &= h[0] == 0;
        for (int r = 1; r < REPS; ++r) {
            all.push_back((double) o[r] / div);
        }
    }
    const Stats s = stats_of(all);
    std::printf("RESULT lds_smem scratch %s cycles %.2f min %.2f max %.2f mismatches %u\n", what, s.median, s.min, s.max, h[0]);
    std::fflush(stdout);
    return s.median;
}

// Spill probes: only compiled and run; the listing shows which scratch instructions the compiler emits.
__global__ __launch_bounds__(32) void k_spill_array(const float * in, const int * idx, float * out) {
    float a[300];
    for (int i = 0; i < 300; ++i) {
        a[i] = in[i] + (float) threadIdx.x;
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) {
        s += a[idx[j] % 300];
    }
    out[threadIdx.x] = s;
}

__global__ __launch_bounds__(32) void k_spill_regs(const float * in, float * out) {
    // fully unrolled so the 300 values are SSA registers (more than the 256 VGPRs): a true register spill
    float v[300];
#pragma unroll
    for (int i = 0; i < 300; ++i) {
        v[i] = in[i] + (float) threadIdx.x;
        asm volatile("" : "+v"(v[i]));
    }
    float s = 0;
#pragma unroll
    for (int i = 299; i >= 0; --i) {
        s += v[i] * (float) (i + 1);
    }
    out[threadIdx.x] = s;
}

void mode_scratch(Ctx & c) {
    uint32_t * res;
    CHECK(hipMalloc(&res, 16));
    bool ok = true;
    const double add = scr_run(k_scr<3>, "baseline_dependent_add_per_step", SITERS, c, res, ok);
    const double w = scr_run(k_scr<0>, "b32_store_reload_chain_per_round_trip", SITERS, c, res, ok);
    const double ws = scr_run(k_scr<1>, "b32_store_wait_storecnt_reload_chain_per_round_trip", SITERS, c, res, ok);
    const double w128 = scr_run(k_scr<2>, "b128_store_reload_chain_per_round_trip", SITERS, c, res, ok);
    const double e = scr_run(k_scr_tp<0, 0>, "empty_batch_one_wait_per_batch", 64, c, res, ok);
    const double t32 = scr_run(k_scr_tp<0, 1>, "b32_16_independent_reloads_per_batch", 64, c, res, ok);
    const double t128 = scr_run(k_scr_tp<1, 1>, "b128_16_independent_reloads_per_batch", 64, c, res, ok);
    std::printf("RESULT lds_smem scratch_summary chain_b32 %.1f chain_b32_net_of_add %.1f chain_b32_with_storecnt_wait "
                "%.1f chain_b128 %.1f | 16_reloads_b32 batch %.1f net %.1f per_load %.2f | 16_reloads_b128 batch %.1f "
                "net %.1f per_load %.2f bytes_per_cycle_per_wave %.1f\n",
                w, w - add, ws, w128, t32, t32 - e, (t32 - e) / 16, t128, t128 - e, (t128 - e) / 16, 16 * 16 * 32 / (t128 - e));
    vd("scratch", ok, "reloaded values equal what was stored (chains and independent reloads, b32 and b128)");

    float *in, *fo;
    int * idx;
    CHECK(hipMalloc(&in, 304 * 4));
    CHECK(hipMalloc(&idx, 8 * 4));
    CHECK(hipMalloc(&fo, 32 * 4));
    std::vector<float> hin(304);
    for (int i = 0; i < 304; ++i) {
        hin[i] = (float) (i % 13) * 0.25f;
    }
    int hidx[8] = {3, 299, 150, 17, 42, 288, 77, 5};
    CHECK(hipMemcpy(in, hin.data(), 304 * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(idx, hidx, sizeof hidx, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_spill_array, dim3(1), dim3(32), 0, 0, in, idx, fo);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> ho(32);
    CHECK(hipMemcpy(ho.data(), fo, 32 * 4, hipMemcpyDeviceToHost));
    bool sok = true;
    for (int t = 0; t < 32; ++t) {
        float s = 0;
        for (int j = 0; j < 8; ++j) {
            s += hin[hidx[j] % 300] + (float) t;
        }
        sok &= std::fabs(s - ho[t]) < 1e-3f;
    }
    hipLaunchKernelGGL(k_spill_regs, dim3(1), dim3(32), 0, 0, in, fo);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(ho.data(), fo, 32 * 4, hipMemcpyDeviceToHost));
    bool rok = true;
    for (int t = 0; t < 32; ++t) {
        double s = 0;
        for (int i = 0; i < 300; ++i) {
            s += ((double) hin[i] + t) * (i + 1);
        }
        rok &= std::fabs(s - ho[t]) < 1e-3 * std::fabs(s);
    }
    std::printf("RESULT lds_smem spill_kernels array_alloca_result %s register_spill_result %s (instructions: see listing)\n",
                sok ? "ok" : "WRONG", rok ? "ok" : "WRONG");
    vd("spill", sok && rok, "both spill kernels give the host's result");
    CHECK(hipFree(res));
    CHECK(hipFree(in));
    CHECK(hipFree(idx));
    CHECK(hipFree(fo));
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// line (LD-7a)
// A wave-uniform 64-bit value forced into SGPRs (the compiler loads kernel data next to "memory" asm with vector loads).
__device__ __forceinline__ uint64_t sgpr64(uint64_t v) {
    const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v), hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
    return ((uint64_t) hi << 32) | lo;
}

__global__ __launch_bounds__(32) void k_line(const uint64_t * pa, uint32_t n, uint32_t * out) {
#pragma unroll 1
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t p1 = sgpr64(pa[2 * i]), p2 = sgpr64(pa[2 * i + 1]);
        uint32_t x, y, t0, t1, t2, t3;
        asm volatile("s_load_b32 %[x], %[p1], 0x0\n\ts_wait_kmcnt 0\n\t"
                     "s_dcache_inv\n\ts_wait_kmcnt 0\n\t"
                     "s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n\t"
                     "s_load_b32 %[x], %[p1], 0x0\n\ts_wait_kmcnt 0\n\t"
                     "s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n\t"
                     "s_load_b32 %[y], %[p2], 0x0\n\ts_wait_kmcnt 0\n\t"
                     "s_getreg_b32 %[t2], hwreg(HW_REG_SHADER_CYCLES_LO)\n\t"
                     "s_load_b32 %[x], %[p1], 0x0\n\ts_wait_kmcnt 0\n\t"
                     "s_getreg_b32 %[t3], hwreg(HW_REG_SHADER_CYCLES_LO)"
                     : [x] "=&s"(x), [y] "=&s"(y), [t0] "=&s"(t0), [t1] "=&s"(t1), [t2] "=&s"(t2), [t3] "=&s"(t3)
                     : [p1] "s"(p1), [p2] "s"(p2)
                     : "memory");
        if (threadIdx.x == 0) {
            out[4 * i] = t1 - t0;
            out[4 * i + 1] = t2 - t1;
            out[4 * i + 2] = t3 - t2;
            out[4 * i + 3] = x ^ y;
        }
    }
}

void mode_line(bool brk) {
    const size_t BUF = 64ull << 20;
    char * buf;
    uint64_t * pa;
    uint32_t * out;
    const std::vector<int> ds = {4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512};
    const int T = 24;
    const int n = 2 * (int) ds.size() * T;
    CHECK(hipMalloc(&buf, BUF + 4096));
    CHECK(hipMalloc(&pa, (size_t) n * 16));
    CHECK(hipMalloc(&out, (size_t) n * 16));
    mem::fill(buf, BUF, mem::RAMP);
    // distinct 4 KiB-aligned X for every trial, in random order
    std::vector<uint32_t> pages(BUF / 4096);
    std::iota(pages.begin(), pages.end(), 0u);
    std::mt19937_64 rng(4711);
    std::shuffle(pages.begin(), pages.end(), rng);
    std::vector<uint64_t> h((size_t) n * 2);
    std::vector<int> dof(n), oof(n);
    int k = 0;
    for (int o : {0, 64}) {
        for (int d : ds) {
            for (int t = 0; t < T; ++t, ++k) {
                const uint64_t x = (uint64_t) buf + (uint64_t) pages[k] * 4096 + o;
                h[2 * k] = x;
                h[2 * k + 1] = x + d;
                dof[k] = d;
                oof[k] = o;
            }
        }
    }
    // trials in random order, so a slow stretch of the run does not fall on one d
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<uint64_t> hp((size_t) n * 2);
    for (int i = 0; i < n; ++i) {
        hp[2 * i] = h[2 * perm[i]];
        hp[2 * i + 1] = h[2 * perm[i] + 1];
    }
    CHECK(hipMemcpy(pa, hp.data(), hp.size() * 8, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_line, dim3(1), dim3(32), 0, 0, pa, (uint32_t) n, out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> r((size_t) n * 4);
    CHECK(hipMemcpy(r.data(), out, r.size() * 4, hipMemcpyDeviceToHost));
    std::vector<double> cold, hot;
    std::map<std::pair<int, int>, std::vector<double>> byd;
    for (int i = 0; i < n; ++i) {
        const int j = perm[i];
        cold.push_back(r[4 * i]);
        hot.push_back(r[4 * i + 2]);
        byd[{oof[j], dof[j]}].push_back(r[4 * i + 1]);
    }
    const Stats sc = stats_of(cold), sh = stats_of(hot);
    std::printf("RESULT lds_smem line X_after_invalidation cycles median %.0f min %.0f max %.0f | X_reloaded median %.0f min %.0f "
                "max %.0f (%d trials)\n",
                sc.median, sc.min, sc.max, sh.median, sh.min, sh.max, n);
    const double mid = 0.5 * (sc.median + sh.median);
    for (int o : {0, 64}) {
        int first_miss = -1;
        std::string line;
        for (int d : ds) {
            const Stats st = stats_of(byd[{o, d}]);
            char b[80];
            std::snprintf(b, sizeof b, " d%d:%.0f(%.0f-%.0f)", d, st.median, st.min, st.max);
            line += b;
            if (first_miss < 0 && st.median > mid) {
                first_miss = d;
            }
        }
        std::printf("RESULT lds_smem line from X+%d: X+d cycles median(min-max):%s\n", o, line.c_str());
        std::printf("INFO lds_smem line from X+%d: first d that misses (median above %.0f): %d\n", o, mid, first_miss);
    }
    // calibration: the reload is a hit (item 4: 36.5 cycles per dependent load with the loop, about 26 without), the
    // load after the invalidation is not (item 4: 140 to 220 past the scalar cache)
    const bool cold_hit = sc.median <= 45;
    const bool ok = sh.median >= 15 && sh.median <= 45 && (brk ? cold_hit : sc.median >= 100);
    char t[200];
    std::snprintf(t, sizeof t, "reload of X %.0f cycles (hit expected, 15..45), load after invalidation %.0f (%s)%s", sh.median,
                  sc.median, brk ? "claimed a hit: <= 45" : "miss expected, >= 100", brk ? " (deliberate break)" : "");
    vd("line_calib", ok, t);
    CHECK(hipFree(buf));
    CHECK(hipFree(pa));
    CHECK(hipFree(out));
}

// ---------------------------------------------------------------------------------------------------------------
// share (LD-7b): see the usage text. Barrier: an atomic counter with a REALTIME-bounded spin (50 ms).
__device__ bool grid_barrier(unsigned * ctr, unsigned target) {
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    const uint64_t r0 = mem::realtime();
    for (;;) {
        const unsigned v = __hip_atomic_load(ctr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        if (__builtin_amdgcn_readfirstlane(v) >= target) {
            return true;
        }
        if (mem::realtime() - r0 > 5000000) {
            return false;
        }
    }
}

__device__ __forceinline__ bool related(int rel, uint32_t a, uint32_t b) {
    switch (rel) {
        case 0: return mem::pl_wgp(a) == mem::pl_wgp(b) && mem::pl_cu(a) != mem::pl_cu(b);
        case 1: return mem::pl_sa(a) == mem::pl_sa(b) && mem::pl_wgp(a) != mem::pl_wgp(b);
        case 2: return mem::pl_se(a) == mem::pl_se(b) && mem::pl_sa(a) != mem::pl_sa(b);
        default: return mem::pl_se(a) != mem::pl_se(b);
    }
}

#define SCH asm volatile("s_load_b64 %0, %0, 0x0\n\ts_wait_kmcnt 0" : "+s"(p)::"memory");
__device__ __forceinline__ uint64_t chase_n(uint64_t p, uint32_t nodes) {
    p = sgpr64(p);
#pragma unroll 1
    for (uint32_t s = 0; s < nodes; s += 4) {
        SCH SCH SCH SCH
    }
    return p;
}

// out per group (8 words): partner, cycles partner region, cycles own fresh region, cycles own region at the end, ok
// flag, final pointers' check (5 bits), cycles own region right after touching it. The fresh regions were just
// written by the chase builder, so they come from L2 or the memory-side cache, not from DRAM.
__global__ __launch_bounds__(32) void k_share(const uint64_t * ts, const uint64_t * us, uint32_t nodes, int rel,
                                              uint32_t * pl, unsigned * bar, uint32_t * out) {
    extern __shared__ char lds_hold[];
    const int G = gridDim.x, g = blockIdx.x;
    const uint32_t me = mem::placement();
    if (threadIdx.x == 0) {
        pl[g] = me;
        lds_hold[0] = 0;
    }
    __threadfence();
    bool ok = grid_barrier(bar, G);
    int cnt = 0;
    for (int h = 0; h < G; ++h) {
        cnt += h != g && related(rel, me, __hip_atomic_load(&pl[h], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
    cnt = __builtin_amdgcn_readfirstlane(cnt);
    int partner = -1, want = cnt ? g % cnt : 0;
    for (int h = 0; h < G && partner < 0; ++h) {
        if (h != g && related(rel, me, __hip_atomic_load(&pl[h], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))) {
            if (want-- == 0) {
                partner = h;
            }
        }
    }
    partner = __builtin_amdgcn_readfirstlane(partner);
    // touch the own region (one full pass)
    const uint64_t own = sgpr64(ts[g]);
    const uint64_t e0 = chase_n(own, nodes);
    // hot calibration: the own region again at once (4 KiB per group, two groups per WGP: fits a 16 KiB cache)
    const uint64_t th0 = wave_cycles();
    const uint64_t e4 = chase_n(own, nodes);
    const uint64_t th1 = wave_cycles();
    __threadfence();
    ok = grid_barrier(bar + 1, G) && ok;
    uint64_t t0 = wave_cycles();
    const uint64_t pt = partner >= 0 ? sgpr64(ts[partner]) : 0, uf = sgpr64(us[g]);
    const uint64_t e1 = partner >= 0 ? chase_n(pt, nodes) : 0;
    uint64_t t1 = wave_cycles();
    const uint64_t e2 = chase_n(uf, nodes);
    uint64_t t2 = wave_cycles();
    const uint64_t e3 = chase_n(own, nodes);
    uint64_t t3 = wave_cycles();
    if (threadIdx.x == 0) {
        out[g * 8 + 0] = (uint32_t) partner;
        out[g * 8 + 1] = (uint32_t) cycle_diff(t0, t1);
        out[g * 8 + 2] = (uint32_t) cycle_diff(t1, t2);
        out[g * 8 + 3] = (uint32_t) cycle_diff(t2, t3);
        out[g * 8 + 4] = ok;
        out[g * 8 + 5] = (uint32_t) ((e0 == own) | ((partner < 0 || e1 == pt) << 1) | ((e2 == uf) << 2) | ((e3 == own) << 3) |
                                     ((e4 == own) << 4));
        out[g * 8 + 6] = (uint32_t) cycle_diff(th0, th1);
    }
}

void mode_share(bool brk) {
    const int G = 64, LAUNCH_REPS = 3;
    const size_t sizes[2] = {4096, 65536};
    const size_t stride = 64;
    const size_t per_launch = 2 * G * sizes[1];
    const size_t BUF = per_launch * 4 * 2 * LAUNCH_REPS;  // fresh regions for every launch
    char * buf;
    uint64_t *ts, *us;
    uint32_t *pl, *out;
    unsigned * bar;
    CHECK(hipMalloc(&buf, BUF));
    CHECK(hipMalloc(&ts, G * 8));
    CHECK(hipMalloc(&us, G * 8));
    CHECK(hipMalloc(&pl, G * 4));
    CHECK(hipMalloc(&out, G * 32));
    CHECK(hipMalloc(&bar, 16));
    const char * rname[4] = {"other CU, same WGP", "other WGP, same shader array", "other array, same engine", "other engine"};
    size_t at = 0;
    std::vector<double> hot_all, cold_all;
    bool all_ok = true;
    for (size_t sz : sizes) {
        const uint32_t nodes = (uint32_t) (sz / stride);
        for (int rel = 0; rel < 4; ++rel) {
            std::vector<double> part, cold, hot, hot0;
            int pairs = 0, bad = 0;
            for (int lr = 0; lr < LAUNCH_REPS; ++lr) {
                std::vector<uint64_t> hts(G), hus(G);
                for (int g = 0; g < G; ++g) {
                    hts[g] = mem::build_chase(buf + at, nodes, stride, true, 100 + g + 1000 * lr);
                    at += sz;
                    hus[g] = mem::build_chase(buf + at, nodes, stride, true, 200 + g + 1000 * lr);
                    at += sz;
                }
                CHECK(hipMemcpy(ts, hts.data(), G * 8, hipMemcpyHostToDevice));
                CHECK(hipMemcpy(us, hus.data(), G * 8, hipMemcpyHostToDevice));
                CHECK(hipMemset(bar, 0, 16));
                CHECK(hipMemset(out, 0, G * 32));
                hipLaunchKernelGGL(k_share, dim3(G), dim3(32), 40 * 1024, 0, ts, us, nodes, rel, pl, bar, out);
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                std::vector<uint32_t> o(G * 8), hpl(G);
                CHECK(hipMemcpy(o.data(), out, G * 32, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(hpl.data(), pl, G * 4, hipMemcpyDeviceToHost));
                std::set<uint32_t> cus;
                for (int g = 0; g < G; ++g) {
                    cus.insert(mem::pl_cu(hpl[g]));
                    const bool good = o[g * 8 + 4] == 1 && o[g * 8 + 5] == 31;
                    bad += !good;
                    if ((int) o[g * 8] >= 0 && (int) o[g * 8] < G) {
                        ++pairs;
                        part.push_back((double) o[g * 8 + 1] / nodes);
                    }
                    cold.push_back((double) o[g * 8 + 2] / nodes);
                    hot.push_back((double) o[g * 8 + 3] / nodes);
                    hot0.push_back((double) o[g * 8 + 6] / nodes);
                }
                if (cus.size() != (size_t) G) {
                    ++bad;
                    std::printf("RESULT lds_smem share: only %zu distinct CUs in the launch\n", cus.size());
                }
            }
            const Stats sp = stats_of(part), scd = stats_of(cold), sh = stats_of(hot), s0 = stats_of(hot0);
            std::printf("RESULT lds_smem share region_KiB %zu partner %-30s pairs %d | own region right after touching %.1f "
                        "(%.1f-%.1f) | partner region cycles_per_load %.1f (min %.1f max %.1f) | own fresh region %.1f "
                        "(%.1f-%.1f) | own region at the end %.1f (%.1f-%.1f) | barrier or pointer failures %d\n",
                        sz / 1024, rname[rel], pairs, s0.median, s0.min, s0.max, sp.median, sp.min, sp.max, scd.median, scd.min,
                        scd.max, sh.median, sh.min, sh.max, bad);
            all_ok = all_ok && bad == 0 && pairs > 0;
            cold_all.insert(cold_all.end(), cold.begin(), cold.end());
            if (sz == sizes[0]) {
                hot_all.insert(hot_all.end(), hot0.begin(), hot0.end());
            }
        }
    }
    const Stats sh = stats_of(hot_all), sc = stats_of(cold_all);
    const bool cal = sh.median <= 45 && (brk ? sc.median <= 45 : sc.median >= 150);
    char t[260];
    std::snprintf(t, sizeof t, "own 4 KiB region right after touching %.1f cycles per load (hit expected, <= 45; item 4: 36.5); "
                  "fresh region %.1f (%s); every group on its own CU, barriers and final pointers right: %s%s", sh.median,
                  sc.median, brk ? "claimed a hit" : "not a hit expected, >= 150", all_ok ? "yes" : "NO",
                  brk ? " (deliberate break)" : "");
    vd("share_calib", cal && all_ok, t);
    for (auto * q : {(void *) buf, (void *) ts, (void *) us, (void *) pl, (void *) out, (void *) bar}) {
        CHECK(hipFree(q));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// sbuf (LD-7c): s_buffer_load against s_load. Descriptor (guide B, SMEM, READ): base 47:0, stride 61:48 = 0,
// num_records 95:64 in bytes (stride 0); the other fields are ignored by scalar loads.
template <int T, int K>
__device__ __forceinline__ void sbld(typename ST<T>::type & r, u4 desc) {
#define SBL(OP) asm volatile(OP " %0, %1, %2" : "=s"(r) : "s"(desc), "i"(K * 64) : "memory")
    if constexpr (T == B32) SBL("s_buffer_load_b32");
    else if constexpr (T == B64) SBL("s_buffer_load_b64");
    else if constexpr (T == B128) SBL("s_buffer_load_b128");
    else SBL("s_buffer_load_b256");
#undef SBL
}
template <int T, int K, int N>
__device__ __forceinline__ void sbissue(typename ST<T>::type (&r)[N], u4 desc) {
    if constexpr (K < N) {
        sbld<T, K>(r[K], desc);
        sbissue<T, K + 1, N>(r, desc);
    }
}
template <int T, int N>
__global__ __launch_bounds__(32) void k_bwidth(u4 desc, uint32_t batches, uint64_t * out) {
    using V = typename ST<T>::type;
    V r[N];
    sbissue<T, 0, N>(r, desc);
    asm volatile("s_wait_kmcnt 0" ::: "memory");
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" : "+s"(r[k]));
    }
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t b = 0; b < batches; ++b) {
        sbissue<T, 0, N>(r, desc);
        asm volatile("s_wait_kmcnt 0" ::: "memory");
#pragma unroll
        for (int k = 0; k < N; ++k) {
            asm volatile("" : "+s"(r[k]));
        }
    }
    const uint64_t c1 = wave_cycles();
    uint32_t bad = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
#pragma unroll
        for (int d = 0; d < ST<T>::dw; ++d) {
            bad += (dword(r[k], d) != (uint32_t) (k * 16 + d));
        }
    }
    if (threadIdx.x == 0) {
        out[0] = cycle_diff(c0, c1);
        out[1] = bad;
    }
}
template <int T, int N>
WRow bwidth_one(Ctx & c, u4 desc, double empty) {
    const uint32_t batches = 256;
    std::vector<double> cyc;
    uint64_t bad = 0;
    for (int t = 0; t < 6; ++t) {
        hipLaunchKernelGGL((k_bwidth<T, N>), dim3(1), dim3(32), 0, 0, desc, batches, c.out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
        if (t > 0) {
            cyc.push_back((double) o[0] / batches);
        }
        bad += o[1];
    }
    const Stats s = stats_of(cyc);
    std::printf("RESULT lds_smem sbuf type buffer_%s N %d cycles_per_batch %.1f min %.1f max %.1f net_of_empty_loop %.1f "
                "wrong_words %llu\n",
                ST<T>::name, N, s.median, s.min, s.max, s.median - empty, (unsigned long long) bad);
    return {s.median, bad == 0};
}

void mode_sbuf(Ctx & c, bool brk) {
    mem::fill(c.ramp, 1u << 20, mem::RAMP);
    const uint64_t base = (uint64_t) c.ramp;
    const u4 desc = {(uint32_t) base, (uint32_t) (base >> 32) & 0xFFFFu, 1u << 20, 0u};
    const double empty = width_one<B32, 0>(c, 0, false).med;
    std::printf("RESULT lds_smem sbuf empty_batch_cycles %.1f\n", empty);
    bool ok = true;
    struct P {
        const char * w;
        int n;
        double s, b;
    };
    std::vector<P> rows;
    auto pair = [&](const char * w, int n, WRow a, WRow b) {
        ok = ok && a.ok && b.ok;
        rows.push_back({w, n, a.med, b.med});
    };
    pair("b32", 1, width_one<B32, 1>(c, empty, true), bwidth_one<B32, 1>(c, desc, empty));
    pair("b32", 8, width_one<B32, 8>(c, empty, true), bwidth_one<B32, 8>(c, desc, empty));
    pair("b32", 16, width_one<B32, 16>(c, empty, true), bwidth_one<B32, 16>(c, desc, empty));
    pair("b64", 1, width_one<B64, 1>(c, empty, true), bwidth_one<B64, 1>(c, desc, empty));
    pair("b64", 8, width_one<B64, 8>(c, empty, true), bwidth_one<B64, 8>(c, desc, empty));
    pair("b64", 16, width_one<B64, 16>(c, empty, true), bwidth_one<B64, 16>(c, desc, empty));
    pair("b128", 1, width_one<B128, 1>(c, empty, true), bwidth_one<B128, 1>(c, desc, empty));
    pair("b128", 8, width_one<B128, 8>(c, empty, true), bwidth_one<B128, 8>(c, desc, empty));
    pair("b128", 16, width_one<B128, 16>(c, empty, true), bwidth_one<B128, 16>(c, desc, empty));
    pair("b256", 1, width_one<B256, 1>(c, empty, true), bwidth_one<B256, 1>(c, desc, empty));
    pair("b256", 8, width_one<B256, 8>(c, empty, true), bwidth_one<B256, 8>(c, desc, empty));
    for (const P & r : rows) {
        std::printf("INFO lds_smem sbuf %s N %d s_load %.1f s_buffer_load %.1f cycles per batch, ratio %.3f\n", r.w, r.n, r.s,
                    r.b, r.b / r.s);
    }
    // calibration against item 4: b32 N = 1 46.1, N = 16 69.1; b128 N = 16 75.1 (within 10 %); --break doubles them
    const double f = brk ? 2.0 : 1.0;
    auto near = [&](double got, double want) { return std::fabs(got - f * want) <= 0.10 * f * want; };
    const bool cal = near(rows[0].s, 46.1) && near(rows[2].s, 69.1) && near(rows[8].s, 75.1);
    char t[220];
    std::snprintf(t, sizeof t, "s_load b32 N=1 %.1f, N=16 %.1f, b128 N=16 %.1f cycles per batch (item 4: 46.1, 69.1, 75.1%s); every "
                  "value of both load kinds right: %s", rows[0].s, rows[2].s, rows[8].s, brk ? ", doubled on purpose" : "",
                  ok ? "yes" : "NO");
    vd("sbuf_calib", cal && ok, t);
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: lds_smem <calib|chase|width|mixed|scratch> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode %s%s\n", prop.name, prop.gcnArchName, mode.c_str(), brk ? " --break" : "");
    mem::clocks("start");
    const bool own_break = mode == "line" || mode == "share" || mode == "sbuf";
    calibrate(brk && !own_break);
    Ctx c;
    CHECK(hipMalloc(&c.out, 1024));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    if (mode == "chase" || mode == "mixed") {
        CHECK(hipMalloc(&c.flush, FLUSH_BYTES));
        mem::fill(c.flush, FLUSH_BYTES, mem::RANDOM);
    }
    if (mode == "width" || mode == "sbuf") {
        CHECK(hipMalloc(&c.ramp, 1u << 20));
    }
    if (mode == "mixed") {
        CHECK(hipMalloc(&c.ramp, 256u << 20));
    }
    if (mode == "calib") {
    } else if (mode == "chase") {
        mode_chase(c);
    } else if (mode == "width") {
        mode_width(c);
    } else if (mode == "mixed") {
        mode_mixed(c);
    } else if (mode == "scratch") {
        mode_scratch(c);
    } else if (mode == "line") {
        mode_line(brk);
    } else if (mode == "share") {
        mode_share(brk);
    } else if (mode == "sbuf") {
        mode_sbuf(c, brk);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::clocks("end");
    return g_fail ? 1 : 0;
}
