// Store and copy bandwidth, write combining, and bandwidth against compute-unit count (lane 2 of notes/STUDY-PLAN.md).
// Usage: mem_store <store|copy|cus> [--break]
//   store: write 1 GiB with the cache-policy variants of a b128 store, then b32 / half-line / split-line / one-line-per-lane
//          patterns to see whether partial-line writes are combined.
//   copy:  copy, nontemporal copy, in-place read-modify-write and a read-only reference.
//   cus:   read bandwidth against the number of work-groups (so compute units) and against the work-group size.
// --break moves the expected copy bandwidth to 1100 GB/s so the copy verdict must FAIL (self-test of the verdict).
#include "mem_common.h"

#include <cmath>
#include <cstdarg>

typedef unsigned v4u __attribute__((ext_vector_type(4)));

constexpr size_t kBytes = size_t(1) << 30;
constexpr int kGrid = 64 * 32;
constexpr int kBlock = 256;

// ---------------------------------------------------------------------------------------------------------------
// Device side

__device__ __forceinline__ size_t gtid() { return blockIdx.x * (size_t) blockDim.x + threadIdx.x; }
__device__ __forceinline__ size_t gstride() { return (size_t) gridDim.x * blockDim.x; }

// A value that depends on the index, so no store can be dropped or hoisted out of the loop.
__device__ __forceinline__ v4u make_val(size_t i) {
    const unsigned a = (unsigned) i * 0x9E3779B1u;
    return v4u{a, a ^ 0x5bd1e995u, a + 0x27d4eb2fu, ~a};
}

// v4u is used instead of the uint4 struct for plain stores so the compiler cannot split them into scalar stores.
__global__ void st_rt(v4u * dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        dst[i] = make_val(i);
    }
}

__global__ void st_nt(v4u * dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        __builtin_nontemporal_store(make_val(i), dst + i);
    }
}

// The cache policy is chosen by hand with inline asm, so the compiler cannot change it.
template <int K>
__global__ void st_asm(v4u * dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        const v4u v = make_val(i);
        v4u * p = dst + i;
        if constexpr (K == 0) {
            asm volatile("global_store_b128 %0, %1, off th:TH_STORE_HT" ::"v"(p), "v"(v) : "memory");
        } else if constexpr (K == 1) {
            asm volatile("global_store_b128 %0, %1, off th:TH_STORE_WB" ::"v"(p), "v"(v) : "memory");
        } else {
            asm volatile("global_store_b128 %0, %1, off scope:SCOPE_SYS" ::"v"(p), "v"(v) : "memory");
        }
    }
}

// Consecutive lanes write consecutive words: one store instruction fills exactly one 128-byte line.
__global__ void st_b32(unsigned * dst, size_t nwords) {
    for (size_t i = gtid(); i < nwords; i += gstride()) {
        dst[i] = (unsigned) i * 0x9E3779B1u;
    }
}

// Every other word: each line is touched once with half of it written.
__global__ void st_half(unsigned * dst, size_t niter) {
    for (size_t i = gtid(); i < niter; i += gstride()) {
        dst[2 * i] = (unsigned) i * 0x9E3779B1u;
    }
}

// Asm keeps the two half-line stores as two separate instructions in program order (the compiler would merge them into
// one b64 store per lane).
__global__ void st_evenodd(unsigned * dst, size_t regions) {
    const size_t g = gtid();
    const size_t wave = g >> 5;
    const unsigned lane = (unsigned) (g & 31);
    const size_t nwaves = gstride() >> 5;
    for (size_t r = wave; r < regions; r += nwaves) {
        unsigned * p = dst + r * 64 + 2 * lane;
        const unsigned v = (unsigned) (r * 32 + lane) * 0x9E3779B1u;
        const unsigned w = ~v;
        asm volatile("global_store_b32 %0, %1, off" ::"v"(p), "v"(v) : "memory");
        asm volatile("global_store_b32 %0, %1, off offset:4" ::"v"(p), "v"(w) : "memory");
    }
}

// One line per lane: 16 of the 128 bytes of every line are written.
__global__ void st_strided(char * dst, size_t regions) {
    const size_t g = gtid();
    const size_t wave = g >> 5;
    const unsigned lane = (unsigned) (g & 31);
    const size_t nwaves = gstride() >> 5;
    for (size_t r = wave; r < regions; r += nwaves) {
        *(v4u *) (dst + r * 4096 + (size_t) lane * 128) = make_val(r * 32 + lane);
    }
}

__global__ void cp_rt(const v4u * __restrict__ src, v4u * __restrict__ dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        dst[i] = src[i];
    }
}

__global__ void cp_nt(const v4u * __restrict__ src, v4u * __restrict__ dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        __builtin_nontemporal_store(__builtin_nontemporal_load(src + i), dst + i);
    }
}

__global__ void cp_ntload(const v4u * __restrict__ src, v4u * __restrict__ dst, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        dst[i] = __builtin_nontemporal_load(src + i);
    }
}

__global__ void rmw(v4u * buf, size_t n) {
    for (size_t i = gtid(); i < n; i += gstride()) {
        v4u x = buf[i];
        x += 1u;
        buf[i] = x;
    }
}

// The accumulator goes to a sink array unconditionally, so the loads cannot be removed.
__global__ void rd(const v4u * __restrict__ src, size_t n, unsigned * __restrict__ sink) {
    unsigned acc = 0;
    for (size_t i = gtid(); i < n; i += gstride()) {
        const v4u v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    sink[gtid()] = acc;
}

// Each work-group walks its own contiguous slice, the block striding through it, four independent 16-byte loads per
// lane per iteration. `slice` is a multiple of 4 * blockDim.x, so there is no tail.
__global__ void __launch_bounds__(1024) rd_cus(const v4u * __restrict__ src, size_t slice, unsigned * __restrict__ sink,
                                               unsigned * __restrict__ pl) {
    if (threadIdx.x == 0) {
        pl[blockIdx.x] = mem::placement();
    }
    const v4u * p = src + blockIdx.x * slice;
    const size_t bd = blockDim.x;
    unsigned acc = 0;
#pragma unroll 1
    for (size_t i = threadIdx.x; i < slice; i += 4 * bd) {
        const v4u a = p[i];
        const v4u b = p[i + bd];
        const v4u c = p[i + 2 * bd];
        const v4u d = p[i + 3 * bd];
        const v4u x = a ^ b ^ c ^ d;
        acc ^= x.x ^ x.y ^ x.z ^ x.w;
    }
    sink[blockIdx.x * bd + threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------------------------
// Host side

static const double kLines = (double) kBytes / 128.0;

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// One configuration: cool down, time, print the bandwidth line and the line-rate line. `lines` is the number of
// 128-byte lines the kernel touches.
template <class F>
static Stats run_store(const char * name, F launch, double bytes, double lines) {
    mem::cool_down();
    const std::vector<double> ms = time_ms(launch, 20, 3);
    CHECK(hipGetLastError());
    const std::string what = std::string("store ") + name;
    const Stats s = mem::report_bw("mem_store", what, ms, bytes);
    const double med = stats_of(ms).median;
    std::printf("RESULT mem_store %s Glines/s %.3f lines %.0f bytes_written %.0f\n", what.c_str(),
                lines / (med * 1e6), lines, bytes);
    std::fflush(stdout);
    return s;
}

static int mode_store() {
    void * d;
    CHECK(hipMalloc(&d, kBytes));
    mem::fill(d, kBytes, mem::RANDOM);
    v4u * d16 = (v4u *) d;
    unsigned * d4 = (unsigned *) d;
    const size_t n16 = kBytes / 16, n4 = kBytes / 4;
    const double B = (double) kBytes;
    int fails = 0;

    const Stats rt = run_store("st_rt", [&] { st_rt<<<kGrid, kBlock>>>(d16, n16); }, B, kLines);
    run_store("st_nt", [&] { st_nt<<<kGrid, kBlock>>>(d16, n16); }, B, kLines);
    run_store("st_ht", [&] { st_asm<0><<<kGrid, kBlock>>>(d16, n16); }, B, kLines);
    run_store("st_wb", [&] { st_asm<1><<<kGrid, kBlock>>>(d16, n16); }, B, kLines);
    run_store("st_sys", [&] { st_asm<2><<<kGrid, kBlock>>>(d16, n16); }, B, kLines);
    run_store("st_b32", [&] { st_b32<<<kGrid, kBlock>>>(d4, n4); }, B, kLines);
    run_store("st_half", [&] { st_half<<<kGrid, kBlock>>>(d4, n4 / 2); }, B / 2, kLines);
    run_store("st_evenodd", [&] { st_evenodd<<<kGrid, kBlock>>>(d4, kBytes / 256); }, B, kLines);
    run_store("st_strided", [&] { st_strided<<<kGrid, kBlock>>>((char *) d, kBytes / 4096); }, B / 8, kLines);

    const bool ok = rt.median >= 300.0 && rt.median <= 720.0;
    verdict("mem_store_store", ok ? "PASS" : "FAIL",
            fmt("st_rt median %.1f GB/s, accepted 300 to 720 GB/s (below the bus limit)", rt.median).c_str());
    fails += ok ? 0 : 1;
    CHECK(hipFree(d));
    return fails;
}

static int mode_copy(bool brk) {
    void *src, *dst;
    unsigned * sink;
    CHECK(hipMalloc(&src, kBytes));
    CHECK(hipMalloc(&dst, kBytes));
    CHECK(hipMalloc(&sink, (size_t) kGrid * kBlock * sizeof(unsigned)));
    mem::fill(src, kBytes, mem::RANDOM, 1);
    mem::fill(dst, kBytes, mem::RANDOM, 2);
    const v4u * s16 = (const v4u *) src;
    v4u * d16 = (v4u *) dst;
    const size_t n16 = kBytes / 16;
    const double B = (double) kBytes;

    auto run = [&](const char * name, auto launch, double bytes) {
        mem::cool_down();
        const std::vector<double> ms = time_ms(launch, 20, 3);
        CHECK(hipGetLastError());
        return mem::report_bw("mem_store", std::string("copy ") + name, ms, bytes);
    };
    const Stats rt = run("cp_rt", [&] { cp_rt<<<kGrid, kBlock>>>(s16, d16, n16); }, 2 * B);
    run("cp_nt", [&] { cp_nt<<<kGrid, kBlock>>>(s16, d16, n16); }, 2 * B);
    run("cp_ntload", [&] { cp_ntload<<<kGrid, kBlock>>>(s16, d16, n16); }, 2 * B);
    run("rmw", [&] { rmw<<<kGrid, kBlock>>>(d16, n16); }, 2 * B);
    run("rd", [&] { rd<<<kGrid, kBlock>>>(s16, n16, sink); }, B);

    const double expect = brk ? 1100.0 : 550.0;
    if (brk) {
        std::printf("NOTE --break: expected copy bandwidth moved from 550 to 1100 GB/s on purpose; the verdict must FAIL\n");
    }
    const bool ok = std::fabs(rt.median - expect) <= 0.10 * expect;
    verdict("mem_store_copy", ok ? "PASS" : "FAIL",
            fmt("cp_rt median %.1f GB/s, expected %.0f GB/s within 10 percent (%.1f to %.1f)", rt.median, expect,
                0.9 * expect, 1.1 * expect)
                .c_str());
    CHECK(hipFree(src));
    CHECK(hipFree(dst));
    CHECK(hipFree(sink));
    return ok ? 0 : 1;
}

static int mode_cus() {
    void * src;
    unsigned *sink, *pl;
    CHECK(hipMalloc(&src, kBytes));
    CHECK(hipMalloc(&sink, (size_t) 1 << 22));
    CHECK(hipMalloc(&pl, 4096 * sizeof(unsigned)));
    mem::fill(src, kBytes, mem::RANDOM);
    const v4u * s16 = (const v4u *) src;

    // Returns the median GB/s.
    auto run = [&](const char * sweep, int k, int block) {
        const bool small = (size_t) k * block < 8192;
        const size_t nvec = (small ? (size_t) 256 << 20 : kBytes) / 16;
        size_t slice = nvec / k;
        slice -= slice % (4 * (size_t) block);
        const double bytes = (double) k * slice * 16;
        mem::cool_down();
        const std::vector<double> ms =
            time_ms([&] { rd_cus<<<k, block>>>(s16, slice, sink, pl); }, small ? 5 : 20, 3);
        CHECK(hipGetLastError());
        std::vector<unsigned> h(k);
        CHECK(hipMemcpy(h.data(), pl, k * sizeof(unsigned), hipMemcpyDeviceToHost));
        std::vector<unsigned> cu, wgp, sa, se;
        for (unsigned p : h) {
            cu.push_back(mem::pl_cu(p));
            wgp.push_back(mem::pl_wgp(p));
            sa.push_back(mem::pl_sa(p));
            se.push_back(mem::pl_se(p));
        }
        auto distinct = [](std::vector<unsigned> & v) {
            std::sort(v.begin(), v.end());
            return (int) (std::unique(v.begin(), v.end()) - v.begin());
        };
        // placement is recorded by thread 0 only: in WGP mode a large group also uses the WGP's other CU, so "cus"
        // can undercount; the WGP, shader-array and shader-engine counts are exact
        const std::string what = fmt("cus %s k=%d block=%d MiB_read=%.0f cus=%d wgps=%d sas=%d ses=%d", sweep, k, block,
                                     bytes / (1 << 20), distinct(cu), distinct(wgp), distinct(sa), distinct(se));
        return mem::report_bw("mem_store", what, ms, bytes).median;
    };

    // A: 1, 2, 4 groups must not get slower (more compute units help until the bus is full), and the whole card must
    // reach the calibrated read speed (634 GB/s, mem_stream calib) within 10 %. Measured: one group of 1024 already
    // reads about a third of the bus, so a fixed ratio between 64 groups and 1 is not a requirement.
    double a1 = 0, a2 = 0, a4 = 0, a64 = 0;
    for (int k : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}) {
        const double g = run("A", k, 1024);
        if (k == 1) a1 = g;
        if (k == 2) a2 = g;
        if (k == 4) a4 = g;
        if (k == 64) a64 = g;
    }
    for (int k : {1, 2, 4, 8, 16, 32, 64, 96, 128, 256}) {
        run("B", k, 256);
    }
    for (int block : {32, 64, 128, 256, 512, 1024}) {
        run("C", 65536 / block, block);
    }

    const bool rising = a2 >= a1 && a4 >= a2;
    const bool full = std::fabs(a64 - 634.0) <= 0.10 * 634.0;
    const bool ok = rising && full;
    verdict("mem_store_cus", ok ? "PASS" : "FAIL",
            fmt("sweep A (groups of 1024): 1, 2, 4, 64 groups give %.1f, %.1f, %.1f, %.1f GB/s; rising from 1 to 4: %s; "
                "64 groups within 10%% of 634 GB/s: %s",
                a1, a2, a4, a64, rising ? "yes" : "no", full ? "yes" : "no")
                .c_str());
    CHECK(hipFree(src));
    CHECK(hipFree(sink));
    CHECK(hipFree(pl));
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    require_arch();
    const char * mode = argc > 1 ? argv[1] : "";
    const bool brk = mem::has_flag(argc, argv, "--break");
    const bool is_store = std::strcmp(mode, "store") == 0, is_copy = std::strcmp(mode, "copy") == 0,
               is_cus = std::strcmp(mode, "cus") == 0;
    if (!is_store && !is_copy && !is_cus) {
        std::fprintf(stderr, "usage: mem_store <store|copy|cus> [--break]\n");
        return 2;
    }
    mem::clocks("start");
    int fails = 0;
    if (is_store) {
        fails = mode_store();
    } else if (is_copy) {
        fails = mode_copy(brk);
    } else {
        fails = mode_cus();
    }
    mem::clocks("end");
    mem::report_hottest();
    return fails ? 1 : 0;
}
