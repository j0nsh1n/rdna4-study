// Shared parts of the memory-system probes (lane 2 of notes/STUDY-PLAN.md).
//
// Device side: the REALTIME counter (100 MHz, chip-wide), the wave's placement (shader engine, shader array, WGP, CU,
// SIMD, wave slot), and fill kernels for the data patterns. Host side: a clock/power/hotspot snapshot read from sysfs
// (paths from the card record, exported by scripts/study-memory.sh), a pause that waits for the hotspot to cool,
// bandwidth reporting, and buffers of random 64-bit pointer cycles for the latency probes.
#pragma once
#include "../harness.h"

#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace mem {

// ---------------------------------------------------------------------------------------------------------------
// Device helpers

__device__ __forceinline__ uint64_t realtime() {
    uint64_t t;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0" : "=s"(t)::"memory");
    return t;
}

// Packed placement: bits 3:0 SE, 4 SA, 8:5 WGP, 9 CU within the WGP, 10 SIMD within the CU, 15:11 wave slot.
// HW_ID1 (guide: WAVE_ID 4:0, SIMD_ID 9:8 with bit 0 = CU in WGP and bit 1 = SIMD in CU, WGP_ID 13:10, SA_ID 16);
// RDNA 4 dropped SE_ID from HW_ID1, so the shader engine comes from the GET_SE_AID_ID message (bits 3:0).
__device__ __forceinline__ uint32_t placement() {
    uint32_t id1, se;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(id1));
    asm volatile("s_sendmsg_rtn_b32 %0, sendmsg(MSG_RTN_GET_SE_AID_ID)\n\ts_wait_kmcnt 0" : "=s"(se));
    const uint32_t slot = id1 & 31u, simd = (id1 >> 8) & 3u, wgp = (id1 >> 10) & 15u, sa = (id1 >> 16) & 1u;
    return (se & 15u) | (sa << 4) | (wgp << 5) | ((simd & 1u) << 9) | ((simd >> 1) << 10) | (slot << 11);
}

__host__ __device__ inline uint32_t pl_se(uint32_t p) { return p & 15u; }
__host__ __device__ inline uint32_t pl_sa(uint32_t p) { return p & 31u; }      // SE + SA: one shader array
__host__ __device__ inline uint32_t pl_wgp(uint32_t p) { return p & 511u; }    // + WGP
__host__ __device__ inline uint32_t pl_cu(uint32_t p) { return p & 1023u; }    // + CU
__host__ __device__ inline uint32_t pl_simd(uint32_t p) { return p & 2047u; }  // + SIMD
__host__ __device__ inline uint32_t pl_slot(uint32_t p) { return p >> 11; }

__host__ __device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Load temporal hints (TH) and scopes in the assembler's spelling (guide table 13, research-10 section 4).
#define MEM_HINTS(X)                                                                                                \
    X(0, "rt", "") X(1, "nt", " th:TH_LOAD_NT") X(2, "ht", " th:TH_LOAD_HT") X(3, "lu", " th:TH_LOAD_LU")        \
        X(4, "nt_rt", " th:TH_LOAD_NT_RT") X(5, "rt_nt", " th:TH_LOAD_RT_NT") X(6, "nt_ht", " th:TH_LOAD_NT_HT") \
            X(7, "rt_se", " scope:SCOPE_SE") X(8, "rt_dev", " scope:SCOPE_DEV") X(9, "rt_sys", " scope:SCOPE_SYS") \
                X(10, "nt_sys", " th:TH_LOAD_NT scope:SCOPE_SYS")
constexpr int NHINTS = 11;
inline const char * hint_name(int h) {
#define X(i, n, s) \
    if (h == i) return n;
    MEM_HINTS(X)
#undef X
    return "?";
}

enum Pattern { ZERO = 0, BYTE55 = 1, RAMP = 2, RANDOM = 3 };

__global__ void fill_kernel(uint32_t * p, size_t n, int pattern, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t v = 0;
        switch (pattern) {
            case BYTE55: v = 0x55555555u; break;
            case RAMP: v = (uint32_t) i; break;
            case RANDOM: v = hash32((uint32_t) i * 0x9E3779B9u ^ hash32(seed + (uint32_t) (i >> 32))); break;
            default: v = 0;
        }
        p[i] = v;
    }
}

__global__ void scatter_kernel(char * dst, size_t stride, const uint64_t * next, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        *(uint64_t *) (dst + i * stride) = next[i];
    }
}

inline void fill(void * p, size_t bytes, int pattern, uint32_t seed = 1) {
    hipLaunchKernelGGL(fill_kernel, dim3(4096), dim3(256), 0, 0, (uint32_t *) p, bytes / 4, pattern, seed);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

inline const char * pattern_name(int p) {
    switch (p) {
        case ZERO: return "zero";
        case BYTE55: return "byte55";
        case RAMP: return "ramp";
        default: return "random";
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host helpers

inline std::string read_file(const std::string & path) {
    std::ifstream f(path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

inline std::string star_line(const std::string & path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find('*') != std::string::npos) {
            const size_t c = line.find(':');
            std::string v = c == std::string::npos ? line : line.substr(c + 1);
            v.erase(std::remove(v.begin(), v.end(), '*'), v.end());
            v.erase(std::remove(v.begin(), v.end(), ' '), v.end());
            return v;
        }
    }
    return "?";
}

inline double read_num(const std::string & path, double scale) {
    std::ifstream f(path);
    double v;
    if (f >> v) {
        return v * scale;
    }
    return -1;
}

// Hotspot (junction) temperature in C, or -1 when the card record's paths are not exported.
inline double hotspot_c() {
    const char * hw = std::getenv("HWMON");
    return hw ? read_num(std::string(hw) + "/temp2_input", 1e-3) : -1;
}

// One line with the current core clock, memory clock, reported power and hotspot. The power reading is the driver's
// (scaled down on this card's main rail by the owner's regulator mod: not true board power).
inline void clocks(const char * tag) {
    const char * drm = std::getenv("DRM_DEVICE");
    const char * hw = std::getenv("HWMON");
    if (!drm || !hw) {
        std::printf("CLOCKS %s unavailable (DRM_DEVICE/HWMON not set)\n", tag);
        return;
    }
    std::printf("CLOCKS %s sclk=%s mclk=%s power=%.0fW cap=%.0fW hotspot=%.0fC\n", tag,
                star_line(std::string(drm) + "/pp_dpm_sclk").c_str(), star_line(std::string(drm) + "/pp_dpm_mclk").c_str(),
                read_num(std::string(hw) + "/power1_average", 1e-6), read_num(std::string(hw) + "/power1_cap", 1e-6),
                hotspot_c());
    std::fflush(stdout);
}

// Thermal rule (owner): a full-card job must start below 70 C. Between the configurations of one job this waits,
// idle, while the hotspot is above `limit`; the highest reading is kept for the summary.
inline double & hottest() {
    static double h = 0;
    return h;
}
inline void cool_down(double limit = 80.0) {
    double t = hotspot_c();
    hottest() = std::max(hottest(), t);
    if (t < 0 || t < limit) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while ((t = hotspot_c()) >= 70.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(120)) {
            std::printf("THERMAL still %.0f C after 120 s idle; continuing\n", t);
            break;
        }
    }
    std::printf("THERMAL paused %.1f s at %.0f C -> %.0f C\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), limit, t);
}
inline void report_hottest() {
    hottest() = std::max(hottest(), hotspot_c());
    std::printf("THERMAL highest hotspot seen by this program: %.0f C\n", hottest());
}

// Bandwidth line: medians and range over the timed repetitions.
inline Stats report_bw(const char * probe, const std::string & what, const std::vector<double> & ms, double bytes) {
    const Stats s = stats_of(ms);
    std::printf("RESULT %s %s GBps %.1f min %.1f max %.1f us %.2f\n", probe, what.c_str(), bytes / (s.median * 1e6),
                bytes / (s.max * 1e6), bytes / (s.min * 1e6), s.median * 1e3);
    std::fflush(stdout);
    return {bytes / (s.median * 1e6), bytes / (s.max * 1e6), bytes / (s.min * 1e6)};
}

inline bool has_flag(int argc, char ** argv, const char * f) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return true;
        }
    }
    return false;
}

// Pointer-chase buffer: `nodes` nodes, one per `stride` bytes; node i holds the 64-bit device address of the next
// node in a single cycle through all nodes. random=true: a random cyclic order (Sattolo), or with block > 0 random
// within consecutive blocks of `block` nodes; false: increasing order.
// The device buffer must hold nodes * stride bytes. Returns the address of the first node; order_out (optional)
// receives the visiting order, so the node reached after k steps is order[k % nodes].
inline uint64_t build_chase(void * dev, size_t nodes, size_t stride, bool random, uint32_t seed = 7,
                            std::vector<uint32_t> * order_out = nullptr, size_t block = 0) {
    std::vector<uint32_t> order(nodes);
    std::iota(order.begin(), order.end(), 0u);
    if (random && block) {
        // random order inside each block of `block` nodes, blocks visited in increasing order: the same lines as a
        // fully random walk, but few changes of page
        std::mt19937_64 rng(seed);
        for (size_t b = 0; b < nodes; b += block) {
            std::shuffle(order.begin() + b, order.begin() + std::min(nodes, b + block), rng);
        }
    } else if (random) {
        std::mt19937_64 rng(seed);
        for (size_t i = nodes - 1; i > 0; --i) {  // Sattolo: one cycle through all nodes
            std::uniform_int_distribution<size_t> d(0, i - 1);
            std::swap(order[i], order[d(rng)]);
        }
    }
    // next[order[k]] = order[k+1], copied packed and scattered to the strided nodes on the device.
    std::vector<uint64_t> next(nodes);
    const uint64_t base = (uint64_t) dev;
    for (size_t k = 0; k < nodes; ++k) {
        next[order[k]] = base + (uint64_t) order[(k + 1) % nodes] * stride;
    }
    uint64_t * tmp;
    CHECK(hipMalloc(&tmp, nodes * 8));
    CHECK(hipMemcpy(tmp, next.data(), nodes * 8, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(scatter_kernel, dim3(2048), dim3(256), 0, 0, (char *) dev, stride, (const uint64_t *) tmp, nodes);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipFree(tmp));
    const uint64_t start = base + (uint64_t) order[0] * stride;
    if (order_out) {
        *order_out = std::move(order);
    }
    return start;
}

}  // namespace mem
