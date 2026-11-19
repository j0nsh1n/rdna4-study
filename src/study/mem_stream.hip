// Lane 2, wall-clock streaming probes (whole card): items 1 (calibration and data content), 2a (bandwidth against
// buffer size, split / shared / private), 3 (the cache trap), 4 (load width, alignment, stride, random, addressing)
// and the streaming part of 8 (cache hints).
//
// Usage: mem_stream <calib|size:pass1|size:win|size:split|size:shared|size:private|trap|shape|hints> [--break]
//   calib   the first-hour read kernel (src/bandwidth.hip, same launch) on 1 GiB, zero data first (known answer
//           634 GB/s, taken on the earlier overclocked profile), then byte 0x55, a ramp and random data; also a
//           32 MiB buffer (fits the 64 MB memory-side cache) read repeatedly with each pattern.
//           --break times a read kernel whose result is unused, so the compiler deletes the loads: the verdict must
//           FAIL (bandwidth above the bus limit).
//   size:P  bandwidth against buffer size, one part per job: pass1 (one 16-byte load per thread, one pass per launch,
//           launches back to back), win (every wave reads one shared window or its own, fewest instructions),
//           split (the whole grid reads one buffer of S bytes, each thread the same addresses every pass), shared
//           (every work-group reads all S bytes), private (every wave reads its own S bytes). split/shared/private
//           repeat the passes inside the kernel.
//   trap    the read kernel of calib launched repeatedly over one buffer of S bytes (what a naive benchmark does).
//   shape   inline-assembly loads: width 4/8/12/16 bytes per lane, misalignment, lane stride, random lanes or
//           random rows, SGPR base + 32-bit VGPR offset against a 64-bit VGPR address.
//   hints   16-byte streaming loads over 1 GiB with each temporal hint (TH) and scope.
// Every timed configuration first waits for the hotspot to cool (mem::cool_down), and prints the clocks.
#include "mem_common.h"

constexpr double BUS_GBPS = 640.0;  // 256 bits x 20 Gbps at the reported 1258 MHz memory clock (READ, research-5)

// ---------------------------------------------------------------------------------------------------------------
// The first-hour read kernel, unchanged (src/bandwidth.hip).
__global__ void read_ref(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) {
        *sink = acc;
    }
}

// Deliberate break: nothing depends on the loads, so they are dead code.
__global__ void read_dead(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    (void) acc;
    (void) sink;
}

// ---------------------------------------------------------------------------------------------------------------
// One element per thread, grid sized to the buffer: the single-pass sweep.
__global__ __launch_bounds__(256) void read_one(const uint4 * __restrict__ base, unsigned * __restrict__ sink) {
    const uint4 v = base[blockIdx.x * (size_t) blockDim.x + threadIdx.x];
    if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9e3779b9u) {
        sink[threadIdx.x] = v.x;
    }
}

__global__ void empty_kernel(unsigned * sink) {
    if (sink == nullptr) {
        sink[0] = 0;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Inline-assembly loads for shape and hints. W bytes per lane; ADDR 0: SGPR base + 32-bit VGPR offset ("saddr"),
// ADDR 1: 64-bit VGPR address. H: temporal hint / scope suffix (W = 16, ADDR = 0 only).
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u3 __attribute__((ext_vector_type(3)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));

// Register type of a W-byte load.
template <int W> struct VecT;
template <> struct VecT<4> { using T = unsigned; };
template <> struct VecT<8> { using T = u2; };
template <> struct VecT<12> { using T = u3; };
template <> struct VecT<16> { using T = u4; };
template <int W> using Vec = typename VecT<W>::T;

template <int W>
__device__ __forceinline__ unsigned fold(const Vec<W> & v) {
    if constexpr (W == 4) {
        return v;
    } else if constexpr (W == 8) {
        return v.x ^ v.y;
    } else if constexpr (W == 12) {
        return v.x ^ v.y ^ v.z;
    } else {
        return v.x ^ v.y ^ v.z ^ v.w;
    }
}

// The loads are inline assembly, so the compiler neither waits for them nor knows when their registers are written:
// the wait names all eight destinations as in/out operands, so no use of them can be moved above it and none of the
// destinations is reused while its load is in flight.
template <class V>
__device__ __forceinline__ void wait8(V (&v)[8]) {
    asm volatile("s_wait_loadcnt 0"
                 : "+v"(v[0]), "+v"(v[1]), "+v"(v[2]), "+v"(v[3]), "+v"(v[4]), "+v"(v[5]), "+v"(v[6]), "+v"(v[7])
                 :
                 : "memory");
}
template <class V>
__device__ __forceinline__ void wait1(V & v) {
    asm volatile("s_wait_loadcnt 0" : "+v"(v) : : "memory");
}

template <int W, int ADDR, int H>
__device__ __forceinline__ Vec<W> ld(const char * base, uint32_t off) {
    if constexpr (W == 16 && ADDR == 0) {
        u4 v;
#define X(i, n, s) \
    if constexpr (H == i) asm volatile("global_load_b128 %0, %1, %2" s : "=v"(v) : "v"(off), "s"(base) : "memory");
        MEM_HINTS(X)
#undef X
        return v;
    } else if constexpr (ADDR == 0) {
        if constexpr (W == 4) {
            unsigned v;
            asm volatile("global_load_b32 %0, %1, %2" : "=v"(v) : "v"(off), "s"(base) : "memory");
            return v;
        } else if constexpr (W == 8) {
            u2 v;
            asm volatile("global_load_b64 %0, %1, %2" : "=v"(v) : "v"(off), "s"(base) : "memory");
            return v;
        } else {
            u3 v;
            asm volatile("global_load_b96 %0, %1, %2" : "=v"(v) : "v"(off), "s"(base) : "memory");
            return v;
        }
    } else {
        const char * a = base + off;
        if constexpr (W == 4) {
            unsigned v;
            asm volatile("global_load_b32 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
            return v;
        } else if constexpr (W == 8) {
            u2 v;
            asm volatile("global_load_b64 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
            return v;
        } else if constexpr (W == 12) {
            u3 v;
            asm volatile("global_load_b96 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
            return v;
        } else {
            u4 v;
            asm volatile("global_load_b128 %0, %1, off" : "=v"(v) : "v"(a) : "memory");
            return v;
        }
    }
}

// Geometry: a "row" is one load instruction of a wave (32 lanes). Lane address within row r:
//   RANDOM = 0: r * 32 * lane_stride + lane * lane_stride + misalign            (sequential rows)
//   RANDOM = 1: (hash(r * 32 + lane) mod lines) * 128 + misalign                 (every lane a random line)
//   RANDOM = 2: (hash(r) mod (bytes / 4096)) * 4096 + lane * lane_stride + misalign (each row at a random place)
// Each wave takes rows wave, wave + waves, ...; 8 loads are issued, then one wait, then the values are folded.
constexpr int BATCH = 8;  // wait8 is written for 8

template <int W, int ADDR, int H, int RANDOM>
__global__ __launch_bounds__(256) void shape_kernel(const char * __restrict__ base, uint32_t rows, uint32_t lane_stride,
                                                    uint32_t misalign, uint32_t lines, uint32_t pages,
                                                    unsigned * __restrict__ sink) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t waves = gridDim.x * blockDim.x / 32;
    unsigned acc = 0;
    for (uint32_t r0 = wave; r0 < rows; r0 += BATCH * waves) {
        Vec<W> v[BATCH];
#pragma unroll
        for (int k = 0; k < BATCH; ++k) {
            uint32_t r = r0 + k * waves;
            r = r < rows ? r : r0;
            uint32_t off;
            if (RANDOM == 0) {
                off = r * 32u * lane_stride + lane * lane_stride + misalign;
            } else if (RANDOM == 1) {
                off = (mem::hash32(r * 32u + lane) % lines) * 128u + misalign;
            } else {
                off = (mem::hash32(r) % pages) * 4096u + lane * lane_stride + misalign;
            }
            v[k] = ld<W, ADDR, H>(base, off);
        }
        wait8(v);
#pragma unroll
        for (int k = 0; k < BATCH; ++k) {
            acc += fold<W>(v[k]);
        }
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Repeated-pass read. MODE 0 split (the grid reads the buffer, each thread the same addresses every pass), 1 shared
// (every work-group reads the whole buffer, starting at a rotated row so groups are not in lockstep), 2 private
// (every wave its own n elements). n = elements of 16 bytes per buffer (per wave for private), below 2^32.
// The thread's m elements are read `passes` times as one stream: 8 loads are issued per iteration even when m < 8
// (measured: a loop that restarted per pass fell into a one-load tail for small buffers and timed latency instead).
// Loads are inline assembly with an SGPR base and a 32-bit byte offset (buffers up to 4 GiB), followed by one explicit
// wait: measured, the compiler's 64-bit addressing cost 12 vector instructions per load, enough to cap cache-level
// bandwidth. acc += keeps every load live.
enum { SPLIT = 0, SHARED = 1, PRIVATE = 2 };

template <int MODE>
__global__ __launch_bounds__(256) void read_rep(const uint4 * __restrict__ base, uint32_t n, int passes,
                                                unsigned * __restrict__ sink) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const char * p = (const char *) base;
    uint32_t start, stride;
    if (MODE == SPLIT) {
        start = gid;
        stride = gridDim.x * blockDim.x;
    } else if (MODE == SHARED) {
        start = threadIdx.x;
        stride = blockDim.x;
    } else {
        p = (const char *) base + (size_t) (gid / 32) * n * 16;
        start = threadIdx.x % 32;
        stride = 32;
    }
    unsigned acc = 0;
    if (start < n) {
        const uint32_t m = (n - start + stride - 1) / stride;
        uint32_t c = MODE == SHARED ? (blockIdx.x * 7u) % m : 0;
        const uint32_t start_b = start * 16, stride_b = stride * 16;
        uint32_t off = start_b + c * stride_b;
        const uint64_t total = (uint64_t) m * passes;
        uint64_t t = 0;
        for (; t + 8 <= total; t += 8) {
            u4 v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                v[k] = ld<16, 0, 0>(p, off);
                ++c;
                off += stride_b;
                if (c == m) {
                    c = 0;
                    off = start_b;
                }
            }
            wait8(v);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc += fold<16>(v[k]);
            }
        }
        for (; t < total; ++t) {
            u4 a = ld<16, 0, 0>(p, off);
            wait1(a);
            acc += fold<16>(a);
            ++c;
            off += stride_b;
            if (c == m) {
                c = 0;
                off = start_b;
            }
        }
    }
    sink[gid] = acc;
}

// Window read with the fewest instructions: every wave reads a window of `win` bytes (a power of two, at least
// 4 KiB) in 4 KiB steps, 8 loads of 512 bytes per step with immediate offsets from a wave-uniform base (scalar
// arithmetic), one wait, and only one add per load. PRIV 0: all waves share one window (rotated start per wave);
// PRIV 1: every wave has its own window.
template <int PRIV>
__global__ __launch_bounds__(256) void read_win(const char * __restrict__ base, uint32_t win, int steps,
                                                unsigned * __restrict__ sink) {
    const uint32_t wave = __builtin_amdgcn_readfirstlane((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    const char * b = PRIV ? base + (size_t) wave * win : base;
    const uint32_t lane_off = (threadIdx.x % 32) * 16;
    uint32_t pos = PRIV ? 0 : (wave * 4096u) & (win - 1);
    unsigned acc = 0;
    for (int st = 0; st < steps; ++st) {
        const char * q = b + pos;
        u4 v[8];
#define L(k) asm volatile("global_load_b128 %0, %1, %2 offset:" #k "*512" : "=v"(v[k]) : "v"(lane_off), "s"(q) : "memory");
        L(0) L(1) L(2) L(3) L(4) L(5) L(6) L(7)
#undef L
        wait8(v);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc += v[k].x;
        }
        pos = (pos + 4096u) & (win - 1);
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------------------------
struct Ctx {
    hipDeviceProp_t prop;
    char * buf = nullptr;
    size_t bytes = 0;
    unsigned * sink = nullptr;
    bool fail = false;
};

static void alloc(Ctx & c, size_t bytes, int pattern) {
    if (c.buf) {
        CHECK(hipFree(c.buf));
    }
    c.bytes = bytes;
    CHECK(hipMalloc(&c.buf, bytes));
    mem::fill(c.buf, bytes, pattern);
}

static const int REF_GRID_MUL = 32, REF_BLOCK = 256;

static double time_ref(Ctx & c, size_t bytes, bool dead, const std::string & what, int reps = 20) {
    mem::cool_down();
    const int grid = c.prop.multiProcessorCount * REF_GRID_MUL;
    const size_t n = bytes / 16;
    auto ms = time_ms(
        [&] {
            if (dead) {
                hipLaunchKernelGGL(read_dead, dim3(grid), dim3(REF_BLOCK), 0, 0, (const uint4 *) c.buf, n, c.sink);
            } else {
                hipLaunchKernelGGL(read_ref, dim3(grid), dim3(REF_BLOCK), 0, 0, (const uint4 *) c.buf, n, c.sink);
            }
        },
        reps, 3);
    CHECK(hipGetLastError());
    return mem::report_bw("mem_stream", what, ms, (double) bytes).median;
}

static void mode_calib(Ctx & c, bool brk) {
    const size_t GiB = 1ull << 30;
    alloc(c, GiB, mem::ZERO);
    mem::clocks("calib-start");
    const double zero = time_ref(c, GiB, brk, brk ? "calib read_dead(BREAK) 1GiB zero" : "calib read_ref 1GiB zero");
    char t[256];
    const bool near = std::fabs(zero - 634.0) <= 0.05 * 634.0, bus = zero <= BUS_GBPS * 1.01;
    std::snprintf(t, sizeof t,
                  "1 GiB zero-filled read %.1f GB/s; known answer 634 GB/s (first hour, overclocked profile) +-5%%: %s; "
                  "at or below the %.0f GB/s bus limit: %s%s",
                  zero, near ? "yes" : "no", BUS_GBPS, bus ? "yes" : "no",
                  brk ? " (deliberate break: loads deleted by the compiler)" : "");
    verdict("mem_stream_calib", near && bus ? "PASS" : "FAIL", t);
    c.fail |= !(near && bus);
    if (brk) {
        return;
    }
    // Data content: four patterns, 1 GiB (memory) and 32 MiB (memory-side cache) read by repeated launches.
    double big[4], small[4];
    for (int p = 0; p < 4; ++p) {
        mem::fill(c.buf, GiB, p);
        big[p] = time_ref(c, GiB, false, std::string("calib read_ref 1GiB ") + mem::pattern_name(p));
        small[p] = time_ref(c, 32ull << 20, false, std::string("calib read_ref 32MiB ") + mem::pattern_name(p), 50);
    }
    std::snprintf(t, sizeof t, "1 GiB: zero %.1f, byte55 %.1f, ramp %.1f, random %.1f GB/s; 32 MiB: %.1f, %.1f, %.1f, %.1f",
                  big[0], big[1], big[2], big[3], small[0], small[1], small[2], small[3]);
    verdict("mem_stream_content", "INFO", t);
    mem::clocks("calib-end");
}

template <int MODE>
static void sweep(Ctx & c, const char * name, int grid, const std::vector<double> & sizes_kib) {
    const int block = 256;
    const size_t waves = (size_t) grid * block / 32;
    for (double kib : sizes_kib) {
        const size_t s = (size_t) (kib * 1024) / 16 * 16;
        const uint32_t n = (uint32_t) (s / 16);
        const double per_pass = MODE == SPLIT ? (double) s : MODE == SHARED ? (double) s * grid : (double) s * waves;
        const int passes = (int) std::max(1.0, std::min(4096.0, std::floor(2.0 * (1ull << 30) / per_pass)));
        const size_t need = MODE == PRIVATE ? s * waves : s;
        if (need > c.bytes) {
            continue;
        }
        mem::cool_down();
        auto ms = time_ms(
            [&] {
                hipLaunchKernelGGL(read_rep<MODE>, dim3(grid), dim3(block), 0, 0, (const uint4 *) c.buf, n, passes,
                                   c.sink);
            },
            5, 2);
        CHECK(hipGetLastError());
        char w[160];
        std::snprintf(w, sizeof w, "size %s S_KiB %.1f footprint_KiB %.1f passes %d", name, kib,
                      (double) need / 1024.0, passes);
        mem::report_bw("mem_stream", w, ms, per_pass * passes);
    }
}

// One pass per launch, many launches back to back between two events (one 16-byte load per thread, grid = S / 4 KiB;
// every launch sweeps the buffer in the same order, so the cache sees a plain cyclic sweep). The time of an empty
// launch measured the same way is subtracted ("net"); the raw figure is printed too.
static void sweep1(Ctx & c, const std::vector<double> & sizes_kib) {
    const int grid = 512, block = 256;
    const int r_empty = 200;
    mem::cool_down();
    auto e = time_ms(
        [&] {
            for (int r = 0; r < r_empty; ++r) {
                hipLaunchKernelGGL(empty_kernel, dim3(grid), dim3(block), 0, 0, c.sink);
            }
        },
        5, 2);
    const double empty_us = stats_of(e).median * 1e3 / r_empty;
    std::printf("RESULT mem_stream size pass1 empty_launch_us %.3f\n", empty_us);
    for (double kib : sizes_kib) {
        const size_t s = (size_t) (kib * 1024) / 16 * 16;
        if (s > c.bytes) {
            continue;
        }
        const int reps = (int) std::max(4.0, std::min(200.0, std::floor(8.0 * (1ull << 30) / (double) s)));
        mem::cool_down();
        auto ms = time_ms(
            [&] {
                for (int r = 0; r < reps; ++r) {
                    hipLaunchKernelGGL(read_one, dim3((uint32_t) (s / 16 / block)), dim3(block), 0, 0,
                                       (const uint4 *) c.buf, c.sink);
                }
            },
            5, 1);
        CHECK(hipGetLastError());
        std::vector<double> net;
        for (double m : ms) {
            net.push_back(m - empty_us * 1e-3 * reps);
        }
        char w[160];
        std::snprintf(w, sizeof w, "size pass1_raw S_KiB %.1f launches %d", kib, reps);
        mem::report_bw("mem_stream", w, ms, (double) s * reps);
        std::snprintf(w, sizeof w, "size pass1_net S_KiB %.1f launches %d", kib, reps);
        mem::report_bw("mem_stream", w, net, (double) s * reps);
    }
}

template <int PRIV>
static void sweep_win(Ctx & c, const std::vector<double> & sizes_kib) {
    const int grid = 256, block = 256;  // 2048 waves: 16 per SIMD, all resident at once
    const size_t waves = (size_t) grid * block / 32;
    for (double kib : sizes_kib) {
        const uint32_t win = (uint32_t) (kib * 1024);
        if ((PRIV ? (size_t) win * waves : (size_t) win) > c.bytes) {
            continue;
        }
        const int steps = (int) std::max(1.0, std::min(8192.0, (PRIV ? 4.0 : 1.0) * win / 4096.0));
        const int st = std::max(steps, 256);
        mem::cool_down();
        auto ms = time_ms(
            [&] {
                hipLaunchKernelGGL(read_win<PRIV>, dim3(grid), dim3(block), 0, 0, c.buf, win, st, c.sink);
            },
            5, 2);
        CHECK(hipGetLastError());
        char wn[160];
        std::snprintf(wn, sizeof wn, "size %s S_KiB %.1f footprint_KiB %.1f steps %d", PRIV ? "win_private" : "win_shared",
                      kib, (PRIV ? (double) win * waves : (double) win) / 1024.0, st);
        mem::report_bw("mem_stream", wn, ms, (double) waves * st * 4096.0);
    }
}

// Each part is a separate GPU job (thermal rule: full-card jobs of about 20 s at most).
static void mode_size(Ctx & c, const std::string & part) {
    alloc(c, 4ull << 30, mem::RANDOM);
    mem::clocks(("size-" + part + "-start").c_str());
    if (part == "pass1") {
        sweep1(c, {1024,  2048,  4096,   6144,   7168,   8192,   9216,   10240,  12288,  14336,
                   16384, 20480, 24576,  32768,  49152,  57344,  65536,  73728,  81920,  98304,
                   114688, 131072, 163840, 196608, 262144, 524288, 1048576, 4194304});
    } else if (part == "win") {
        // every wave reads the same window, or its own (footprint 2048 x S)
        sweep_win<0>(c, {4, 8, 16, 24, 32, 48, 64, 128, 192, 256, 320, 384, 512, 1024, 2048, 4096, 8192});
        sweep_win<1>(c, {4, 8, 16, 32, 64, 128, 256, 512});
    } else if (part == "split") {
        // 512 groups of 256 = 4096 waves (two rounds of 16 per SIMD)
        sweep<SPLIT>(c, "split", 512,
                     {256,   512,   1024,  1536,  2048,  2560,  3072,  4096,   6144,   7168,   8192,  9216,
                      10240, 12288, 16384, 18432, 20480, 22528, 24576, 32768,  40960,  49152,  57344, 65536,
                      73728, 81920, 98304, 131072, 163840, 196608, 262144, 393216, 524288, 1048576, 4194304});
    } else if (part == "shared") {
        // 128 groups, each reads the whole buffer
        sweep<SHARED>(c, "shared", 128, {4,   8,   16,  24,  28,  32,  36,  40,  48,   64,   96,   128, 192,
                                         224, 256, 288, 320, 352, 384, 448, 512, 768, 1024, 2048, 4096, 8192});
    } else if (part == "private") {
        // 128 groups = 1024 waves, footprint 1024 x S
        sweep<PRIVATE>(c, "private", 128, {1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 64, 128, 256, 1024});
    } else {
        std::fprintf(stderr, "unknown size part %s\n", part.c_str());
        std::exit(2);
    }
    mem::clocks(("size-" + part + "-end").c_str());
}

static void mode_trap(Ctx & c) {
    alloc(c, 2ull << 30, mem::RANDOM);
    mem::clocks("trap-start");
    for (int p : {mem::RANDOM, mem::ZERO}) {
        mem::fill(c.buf, c.bytes, p);
        for (size_t mib : {32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512, 1024, 2048}) {
            time_ref(c, mib << 20, false, std::string("trap read_ref ") + mem::pattern_name(p) + " MiB " +
                                              std::to_string(mib));
        }
    }
    mem::clocks("trap-end");
}

template <int W, int ADDR, int H, int RANDOM>
static double shape_run(Ctx & c, const char * label, uint32_t lane_stride, uint32_t misalign, int reps = 10) {
    const int grid = 512, block = 256;
    const uint32_t row_bytes = 32 * lane_stride;
    const size_t span = RANDOM == 1 ? c.bytes : c.bytes - 4096;
    uint32_t rows = (uint32_t) (span / row_bytes);
    if (RANDOM != 0) {
        rows = (uint32_t) ((256ull << 20) / (32 * W));  // a fixed amount of useful data: 256 MiB
    }
    const uint32_t lines = (uint32_t) ((c.bytes - 256) / 128), pages = (uint32_t) ((c.bytes - 8192) / 4096);
    mem::cool_down();
    auto ms = time_ms(
        [&] {
            hipLaunchKernelGGL((shape_kernel<W, ADDR, H, RANDOM>), dim3(grid), dim3(block), 0, 0, c.buf, rows,
                               lane_stride, misalign, lines, pages, c.sink);
        },
        reps, 2);
    CHECK(hipGetLastError());
    const double useful = (double) rows * 32 * W;
    // Lines touched per row: sequential rows touch the span they cover (shared with neighbouring rows when lanes are
    // dense); random lanes touch 32 lines.
    double touched;
    if (RANDOM == 1) {
        touched = (double) rows * 32 * 128;
    } else {
        const double span_row = std::max<double>(row_bytes, 32.0 * W);
        touched = (double) rows * (lane_stride <= 128 ? span_row : 32.0 * 128);
    }
    char w[200];
    std::snprintf(w, sizeof w, "shape %s W %d addr %s stride %u misalign %u random %d", label, W,
                  ADDR ? "vaddr64" : "saddr", lane_stride, misalign, RANDOM);
    const double gb = mem::report_bw("mem_stream", w, ms, useful).median;
    const Stats s = stats_of(ms);
    std::printf("RESULT mem_stream %s touched_GBps %.1f loads_per_ns %.3f\n", w, touched / (s.median * 1e6),
                (double) rows / (s.median * 1e6));
    (void) label;
    return gb;
}

static void mode_shape(Ctx & c) {
    alloc(c, 1ull << 30, mem::RANDOM);
    mem::clocks("shape-start");
    // width, dense, aligned, both address forms
    shape_run<4, 0, 0, 0>(c, "width", 4, 0);
    shape_run<8, 0, 0, 0>(c, "width", 8, 0);
    shape_run<12, 0, 0, 0>(c, "width", 12, 0);
    const double w16 = shape_run<16, 0, 0, 0>(c, "width", 16, 0);
    shape_run<4, 1, 0, 0>(c, "addr", 4, 0);
    shape_run<8, 1, 0, 0>(c, "addr", 8, 0);
    shape_run<12, 1, 0, 0>(c, "addr", 12, 0);
    shape_run<16, 1, 0, 0>(c, "addr", 16, 0);
    // alignment of the row start (rows are 512 bytes = 4 lines when aligned)
    for (uint32_t m : {0u, 4u, 8u, 16u, 32u, 64u, 96u}) {
        shape_run<16, 0, 0, 0>(c, "align", 16, m);
    }
    shape_run<4, 0, 0, 0>(c, "align", 4, 2);
    shape_run<4, 0, 0, 0>(c, "align", 4, 64);
    // lane stride: lanes spread over more lines
    for (uint32_t s : {16u, 32u, 64u, 128u, 256u, 512u}) {
        shape_run<16, 0, 0, 0>(c, "stride", s, 0);
    }
    for (uint32_t s : {4u, 8u, 16u, 32u, 64u, 128u}) {
        shape_run<4, 0, 0, 0>(c, "stride", s, 0);
    }
    // random: each lane a random line, or each row (512 B) at a random 4 KiB-aligned place
    shape_run<16, 0, 0, 1>(c, "random_lane", 16, 0, 5);
    shape_run<4, 0, 0, 1>(c, "random_lane", 4, 0, 5);
    shape_run<16, 1, 0, 1>(c, "random_lane", 16, 0, 5);
    shape_run<16, 0, 0, 2>(c, "random_row", 16, 0, 5);
    shape_run<4, 0, 0, 2>(c, "random_row", 4, 0, 5);
    char t[200];
    const bool ok = w16 > 0.85 * 634.0 && w16 <= BUS_GBPS * 1.01;
    std::snprintf(t, sizeof t, "inline-asm 16-byte dense loads %.1f GB/s; must be within 15%% of 634 and below the bus",
                  w16);
    verdict("mem_stream_shape", ok ? "PASS" : "FAIL", t);
    c.fail |= !ok;
    mem::clocks("shape-end");
}

template <int H>
static void hint_one(Ctx & c) {
    shape_run<16, 0, H, 0>(c, mem::hint_name(H), 16, 0);
    if constexpr (H + 1 < mem::NHINTS) {
        hint_one<H + 1>(c);
    }
}

static void mode_hints(Ctx & c) {
    alloc(c, 1ull << 30, mem::RANDOM);
    mem::clocks("hints-start");
    hint_one<0>(c);
    mem::clocks("hints-end");
}

int main(int argc, char ** argv) {
    Ctx c;
    c.prop = require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: mem_stream <calib|size:PART|trap|shape|hints> [--break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool brk = mem::has_flag(argc, argv, "--break");
    CHECK(hipMalloc(&c.sink, 1 << 22));
    std::printf("device %s (%s), %d CUs, mode %s%s\n", c.prop.name, c.prop.gcnArchName, c.prop.multiProcessorCount,
                mode.c_str(), brk ? " --break" : "");
    if (mode == "calib") {
        mode_calib(c, brk);
    } else if (mode.rfind("size:", 0) == 0) {
        mode_size(c, mode.substr(5));
    } else if (mode == "trap") {
        mode_trap(c);
    } else if (mode == "shape") {
        mode_shape(c);
    } else if (mode == "hints") {
        mode_hints(c);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::report_hottest();
    if (c.buf) {
        CHECK(hipFree(c.buf));
    }
    CHECK(hipFree(c.sink));
    return c.fail ? 1 : 0;
}
