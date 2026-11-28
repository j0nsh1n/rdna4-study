// Blind Spot #3: Store-hint MALL residency/eviction and cross-wave 32 B write combining on gfx1201.
// Tests all 9 valid gfx1201 global_store_b128 cache-hint/scope encodings (verified with llvm-mc):
//   rt (default), nt, ht, wb, nt_rt, rt_nt, nt_ht, rt_sys, nt_sys.
//
// Usage: store_mall [--break]
//   Part 3A: Producer writes 4 MiB / 32 MiB chase buffer with hint H -> separate consumer chases it
//            (and streams 16 MiB) -> measures whether store hint H allocates in the 64 MB MALL.
//   Part 3B: Hot 32 MiB chase buffer in MALL -> 256 MiB streaming write with hint H -> re-chase
//            -> measures whether store hint H evicts hot activations from MALL.
//   Part 3C: 512 MiB write-combining bandwidth across full_b128, full_b32, same_wave_4x32B,
//            xwave_same_wg_32B (4-wave 32 B row-piece pattern), xwave_diff_wg_32B, quarter_32B_only, eighth_16B_only.
//   --break: corrupts expected pointer and write-combining checksum; must exit non-zero (FAIL).
#include "../study/mem_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned u4 __attribute__((ext_vector_type(4)));

#define STORE_HINTS(X)                                                                           \
    X(0, "rt", "")                                                                               \
    X(1, "nt", " th:TH_STORE_NT")                                                                \
    X(2, "ht", " th:TH_STORE_HT")                                                                \
    X(3, "wb", " th:TH_STORE_WB")                                                                \
    X(4, "nt_rt", " th:TH_STORE_NT_RT")                                                          \
    X(5, "rt_nt", " th:TH_STORE_RT_NT")                                                          \
    X(6, "nt_ht", " th:TH_STORE_NT_HT")                                                          \
    X(7, "rt_sys", " scope:SCOPE_SYS")                                                           \
    X(8, "nt_sys", " th:TH_STORE_NT scope:SCOPE_SYS")

constexpr int NSHINTS = 9;

static const char * shint_name(int h) {
    switch (h) {
#define X(i, n, s) case i: return n;
        STORE_HINTS(X)
#undef X
        default: return "?";
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Part 3A/3B kernels: flush MALL, warm MALL, hinted store copy (NT load so source never pollutes MALL),
// hinted stream store, single-wave pointer chase, and consumer 16 MiB streaming read.

__global__ __launch_bounds__(256) void k_read_rt(const u4 * __restrict__ src, size_t n16, unsigned * sink) {
    unsigned acc = 0;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += step) {
        u4 v;
        const u4 * p = src + i;
        asm volatile("global_load_b128 %0, %1, off" : "=v"(v) : "v"(p) : "memory");
        asm volatile("s_wait_loadcnt 0" : "+v"(v) :: "memory");
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) *sink = acc;
}

template <int H>
__global__ __launch_bounds__(256) void k_store_copy(const u4 * __restrict__ src, u4 * __restrict__ dst, size_t n16) {
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += step) {
        u4 v;
        const u4 * sp = src + i;
        u4 * dp = dst + i;
        // NT load so reading src does not allocate in MALL
        asm volatile("global_load_b128 %0, %1, off th:TH_LOAD_NT" : "=v"(v) : "v"(sp) : "memory");
        asm volatile("s_wait_loadcnt 0" : "+v"(v) :: "memory");
#define X(idx, nm, mod) \
        if constexpr (H == idx) asm volatile("global_store_b128 %0, %1, off" mod :: "v"(dp), "v"(v) : "memory");
        STORE_HINTS(X)
#undef X
    }
}

template <int H>
__global__ __launch_bounds__(256) void k_store_stream(u4 * __restrict__ dst, size_t n16, unsigned seed) {
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += step) {
        unsigned a = (unsigned) i * 0x9E3779B1u ^ seed;
        u4 v = {a, a ^ 0x5bd1e995u, a + 0x27d4eb2fu, ~a};
        u4 * dp = dst + i;
#define X(idx, nm, mod) \
        if constexpr (H == idx) asm volatile("global_store_b128 %0, %1, off" mod :: "v"(dp), "v"(v) : "memory");
        STORE_HINTS(X)
#undef X
    }
}

__device__ __forceinline__ uint64_t shader_cycles() {
    uint32_t hi1 = 0, lo = 0, hi2 = 0;
    asm volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)\n"
        "s_getreg_b32 %1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
        "s_getreg_b32 %2, hwreg(HW_REG_SHADER_CYCLES_HI)\n"
        : "=s"(hi1), "=s"(lo), "=s"(hi2));
    uint32_t hi = (hi1 == hi2) ? hi1 : ((lo & 0x80000000u) ? hi1 : hi2);
    return ((uint64_t) hi << 32) | lo;
}

__global__ __launch_bounds__(32) void k_chase(uint64_t start, uint32_t steps, uint64_t * out) {
    if (threadIdx.x != 0) return;
    uint64_t p = start;
    const uint64_t c0 = shader_cycles();
    const uint64_t r0 = mem::realtime();
#pragma unroll 1
    for (uint32_t i = 0; i < steps; i++) {
        uint64_t nxt;
        asm volatile("global_load_b64 %0, %1, off\ns_wait_loadcnt 0" : "=v"(nxt) : "v"(p) : "memory");
        p = nxt;
    }
    const uint64_t c1 = shader_cycles();
    const uint64_t r1 = mem::realtime();
    out[0] = c1 - c0;
    out[1] = r1 - r0;
    out[2] = p;
}

__global__ __launch_bounds__(256) void k_read_bw(const u4 * __restrict__ src, size_t n16, unsigned * sink) {
    unsigned acc = 0;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += step) {
        const u4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) *sink = acc;
}

// ---------------------------------------------------------------------------------------------------------------
// Part 3C kernels: cross-wave 32 B piece write combining (512 MiB = 4M x 128 B lines = 128M uint32 words)

__device__ __forceinline__ unsigned pat_val(size_t word_idx, unsigned seed) {
    return ((unsigned) word_idx * 0x9E3779B1u) ^ seed;
}

// 1. Full 128 B lines via b128 (4 contiguous lines per wave per iter)
__global__ __launch_bounds__(256) void wc_full_b128(u4 * __restrict__ dst, size_t n16, unsigned seed) {
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += step) {
        size_t w = i * 4;
        u4 v = {pat_val(w, seed), pat_val(w + 1, seed), pat_val(w + 2, seed), pat_val(w + 3, seed)};
        u4 * p = dst + i;
        asm volatile("global_store_b128 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// 2. Full 128 B lines via b32 (1 contiguous line per wave per iter)
__global__ __launch_bounds__(256) void wc_full_b32(unsigned * __restrict__ dst, size_t nwords, unsigned seed) {
    const size_t step = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < nwords; i += step) {
        unsigned v = pat_val(i, seed);
        unsigned * p = dst + i;
        asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// 3. Same wave writes all 4 x 32 B quarters of 4 lines (4 separate b32 instructions per wave)
__global__ __launch_bounds__(256) void wc_same_wave_4x32B(unsigned * __restrict__ dst, size_t nlines, unsigned seed) {
    const size_t g = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t wave = g >> 5;
    const size_t nwaves = ((size_t) gridDim.x * blockDim.x) >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned line_sub = lane >> 3;   // 0..3 -> which of 4 lines
    const unsigned word_in_q = lane & 7u;  // 0..7 -> 32 B sector
    for (size_t base_line = wave * 4; base_line < nlines; base_line += nwaves * 4) {
        const size_t l = base_line + line_sub;
        for (unsigned q = 0; q < 4; q++) {
            const size_t wi = l * 32 + q * 8 + word_in_q;
            unsigned v = pat_val(wi, seed);
            unsigned * p = dst + wi;
            asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
        }
    }
}

// 4. Cross-wave SAME workgroup (same CU in -mcumode): 4 waves of a quad each write one 32 B quarter
//    of the SAME 4 cache lines (cross-wave 32 B row-piece pattern, 0 wasted lanes).
__global__ __launch_bounds__(256) void wc_xwave_same_wg_32B(unsigned * __restrict__ dst, size_t nlines, unsigned seed) {
    const size_t g = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t wave = g >> 5;
    const size_t quad = wave >> 2;
    const unsigned q = (unsigned) (wave & 3u); // which 32 B quarter (0..3) this wave writes
    const size_t nquads = ((size_t) gridDim.x * blockDim.x) >> 7;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned line_sub = lane >> 3;
    const unsigned word_in_q = lane & 7u;
    for (size_t base_line = quad * 4; base_line < nlines; base_line += nquads * 4) {
        const size_t wi = (base_line + line_sub) * 32 + q * 8 + word_in_q;
        unsigned v = pat_val(wi, seed);
        unsigned * p = dst + wi;
        asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// 5. Cross-wave DIFFERENT workgroups (spread across CUs): WG (b & 3) writes quarter q of shared 4-line tiles.
__global__ __launch_bounds__(256) void wc_xwave_diff_wg_32B(unsigned * __restrict__ dst, size_t nlines, unsigned seed) {
    const unsigned q = blockIdx.x & 3u;
    const size_t wg_grp = blockIdx.x >> 2;
    const size_t nwg_grp = gridDim.x >> 2;
    const unsigned w_in_wg = threadIdx.x >> 5; // 0..7
    const size_t tile = wg_grp * 8 + w_in_wg;
    const size_t ntiles = nwg_grp * 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned line_sub = lane >> 3;
    const unsigned word_in_q = lane & 7u;
    for (size_t base_line = tile * 4; base_line < nlines; base_line += ntiles * 4) {
        const size_t wi = (base_line + line_sub) * 32 + q * 8 + word_in_q;
        unsigned v = pat_val(wi, seed);
        unsigned * p = dst + wi;
        asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// 6. Quarter-line only: writes ONLY 32 B (words 0..7) of every 128 B line (128 MiB written over 512 MiB span)
__global__ __launch_bounds__(256) void wc_quarter_32B_only(unsigned * __restrict__ dst, size_t nlines, unsigned seed) {
    const size_t g = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t wave = g >> 5;
    const size_t nwaves = ((size_t) gridDim.x * blockDim.x) >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned line_sub = lane >> 3;
    const unsigned word_in_q = lane & 7u;
    for (size_t base_line = wave * 4; base_line < nlines; base_line += nwaves * 4) {
        const size_t wi = (base_line + line_sub) * 32 + word_in_q;
        unsigned v = pat_val(wi, seed);
        unsigned * p = dst + wi;
        asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// 7. Eighth-line only: writes ONLY 16 B (words 0..3) of every 128 B line (64 MiB written over 512 MiB span -> sub-32B RMW!)
__global__ __launch_bounds__(256) void wc_eighth_16B_only(unsigned * __restrict__ dst, size_t nlines, unsigned seed) {
    const size_t g = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t wave = g >> 5;
    const size_t nwaves = ((size_t) gridDim.x * blockDim.x) >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned line_sub = lane >> 2;   // 0..7 -> 8 lines per wave
    const unsigned word_in_e = lane & 3u;  // 0..3 -> 16 B
    for (size_t base_line = wave * 8; base_line < nlines; base_line += nwaves * 8) {
        const size_t wi = (base_line + line_sub) * 32 + word_in_e;
        unsigned v = pat_val(wi, seed);
        unsigned * p = dst + wi;
        asm volatile("global_store_b32 %0, %1, off" :: "v"(p), "v"(v) : "memory");
    }
}

// Verify written buffer words on GPU (returns count of mismatches)
__global__ __launch_bounds__(256) void k_verify_words(const unsigned * __restrict__ buf, size_t nlines,
                                                      unsigned seed, unsigned words_per_line, unsigned * bad_cnt) {
    const size_t step = (size_t) gridDim.x * blockDim.x;
    unsigned local_bad = 0;
    for (size_t l = blockIdx.x * (size_t) blockDim.x + threadIdx.x; l < nlines; l += step) {
        for (unsigned w = 0; w < words_per_line; w++) {
            size_t wi = l * 32 + w;
            if (buf[wi] != pat_val(wi, seed)) local_bad++;
        }
    }
    if (local_bad) atomicAdd(bad_cnt, local_bad);
}

// ---------------------------------------------------------------------------------------------------------------
// Host helpers

static double median_v(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct ChaseRes { double cyc, ns, ghz; bool ok; };

static ChaseRes run_chase(uint64_t start, uint64_t expect, uint32_t steps, uint64_t * d_out) {
    hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, steps, d_out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint64_t o[3] = {0, 0, 0};
    CHECK(hipMemcpy(o, d_out, sizeof o, hipMemcpyDeviceToHost));
    double cyc = (double) o[0] / steps;
    double ns = ((double) o[1] * 10.0) / steps;
    double ghz = (o[1] > 0) ? ((double) o[0] / ((double) o[1] * 10.0)) : 0.0;
    return {cyc, ns, ghz, o[2] == expect};
}

static void flush_mall(const u4 * d_flush, size_t n16, unsigned * d_sink) {
    hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, d_flush, n16, d_sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

template <int H>
static void launch_store_copy(const u4 * src, u4 * dst, size_t n16) {
    hipLaunchKernelGGL(k_store_copy<H>, dim3(512), dim3(256), 0, 0, src, dst, n16);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

template <int H>
static void launch_store_stream(u4 * dst, size_t n16, unsigned seed) {
    hipLaunchKernelGGL(k_store_stream<H>, dim3(1024), dim3(256), 0, 0, dst, n16, seed);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

static double time_read_bw(const u4 * src, size_t bytes, unsigned * d_sink, hipEvent_t e0, hipEvent_t e1) {
    CHECK(hipEventRecord(e0, 0));
    hipLaunchKernelGGL(k_read_bw, dim3(512), dim3(256), 0, 0, src, bytes / 16, d_sink);
    CHECK(hipEventRecord(e1, 0));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double) bytes / ((double) ms * 1e6);
}

template <int H>
static void run_hint_row(size_t kib, const u4 * d_src, u4 * d_dst, const u4 * d_flush, size_t flush_n16,
                         u4 * d_stream, size_t stream_n16, uint64_t start, uint64_t expect, uint32_t steps,
                         uint64_t * d_out, unsigned * d_sink, hipEvent_t e0, hipEvent_t e1, int & fail) {
    const size_t bytes = kib * 1024;
    const size_t n16 = bytes / 16;

    // 3A: flush -> hinted store copy -> consumer chase (median of 3) + consumer read BW (median of 3)
    std::vector<double> c_alloc, ns_alloc, bw_alloc;
    bool ok_a = true;
    for (int t = 0; t < 3; t++) {
        flush_mall(d_flush, flush_n16, d_sink);
        launch_store_copy<H>(d_src, d_dst, n16);
        ChaseRes r = run_chase(start, expect, steps, d_out);
        c_alloc.push_back(r.cyc);
        ns_alloc.push_back(r.ns);
        ok_a &= r.ok;
    }
    for (int t = 0; t < 3; t++) {
        flush_mall(d_flush, flush_n16, d_sink);
        launch_store_copy<H>(d_src, d_dst, n16);
        bw_alloc.push_back(time_read_bw(d_dst, bytes, d_sink, e0, e1));
    }

    // 3B: flush -> warm d_dst twice in MALL -> 256 MiB hinted store stream -> re-chase d_dst (median of 3)
    std::vector<double> c_evict, ns_evict;
    bool ok_b = true;
    for (int t = 0; t < 3; t++) {
        flush_mall(d_flush, flush_n16, d_sink);
        hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, d_dst, n16, d_sink);
        hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, d_dst, n16, d_sink);
        CHECK(hipDeviceSynchronize());
        launch_store_stream<H>(d_stream, stream_n16, (unsigned) (0x1234 + t + H * 17));
        ChaseRes r = run_chase(start, expect, steps, d_out);
        c_evict.push_back(r.cyc);
        ns_evict.push_back(r.ns);
        ok_b &= r.ok;
    }

    std::printf("RESULT store_mall KiB %zu hint %-6s alloc_cyc %.1f alloc_ns %.1f cons_bw %.1f GB/s evict_cyc %.1f evict_ns %.1f ptr %s\n",
                kib, shint_name(H), median_v(c_alloc), median_v(ns_alloc), median_v(bw_alloc),
                median_v(c_evict), median_v(ns_evict), (ok_a && ok_b) ? "ok" : "WRONG");
    std::fflush(stdout);
    if (!(ok_a && ok_b)) fail++;

    if constexpr (H + 1 < NSHINTS) {
        run_hint_row<H + 1>(kib, d_src, d_dst, d_flush, flush_n16, d_stream, stream_n16,
                            start, expect, steps, d_out, d_sink, e0, e1, fail);
    }
}

int main(int argc, char ** argv) {
    bool do_break = false;
    for (int i = 1; i < argc; i++) if (!std::strcmp(argv[i], "--break")) do_break = true;

    mem::clocks("store_mall-start");
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));

    const size_t kMaxChase = 32u << 20;
    const size_t kFlush = 256u << 20;
    const size_t kWC = 512u << 20;

    char * d_src_c = nullptr, * d_dst_c = nullptr;
    u4 * d_flush = nullptr, * d_stream = nullptr;
    unsigned * d_sink = nullptr, * d_bad = nullptr;
    uint64_t * d_out = nullptr;
    CHECK(hipMalloc(&d_src_c, kMaxChase));
    CHECK(hipMalloc(&d_dst_c, kMaxChase));
    CHECK(hipMalloc(&d_flush, kFlush));
    CHECK(hipMalloc(&d_stream, kFlush));
    CHECK(hipMalloc(&d_sink, 64));
    CHECK(hipMalloc(&d_bad, sizeof(unsigned)));
    CHECK(hipMalloc(&d_out, 32));
    CHECK(hipMemset(d_flush, 0x5a, kFlush));
    CHECK(hipMemset(d_stream, 0xa5, kFlush));

    int fail = 0;
    const uint32_t steps = 8192;

    for (size_t kib : {size_t(4096), size_t(16384), size_t(32768)}) {
        mem::cool_down();
        const size_t bytes = kib * 1024;
        const size_t nodes = bytes / 128;
        std::vector<uint32_t> order;
        // Build chase with pointers targeting d_dst_c!
        const uint64_t start = mem::build_chase(d_dst_c, nodes, 128, true, 29, &order);
        // Copy built chase table into d_src_c so producer copies d_src_c -> d_dst_c with hint H
        CHECK(hipMemcpy(d_src_c, d_dst_c, bytes, hipMemcpyDeviceToDevice));
        const uint64_t expect = (uint64_t) d_dst_c + (uint64_t) order[steps % nodes] * 128 + (do_break ? 128 : 0);

        // References: cold (flushed) and warm (read twice with TH_LOAD_RT)
        std::vector<double> c_cold, c_warm, bw_cold, bw_warm;
        bool ok_r = true;
        for (int t = 0; t < 3; t++) {
            flush_mall(d_flush, kFlush / 16, d_sink);
            ChaseRes rc = run_chase(start, expect, steps, d_out);
            c_cold.push_back(rc.cyc); ok_r &= rc.ok;
            flush_mall(d_flush, kFlush / 16, d_sink);
            bw_cold.push_back(time_read_bw((const u4 *) d_dst_c, bytes, d_sink, e0, e1));

            flush_mall(d_flush, kFlush / 16, d_sink);
            hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, (const u4 *) d_dst_c, bytes / 16, d_sink);
            hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, (const u4 *) d_dst_c, bytes / 16, d_sink);
            CHECK(hipDeviceSynchronize());
            ChaseRes rw = run_chase(start, expect, steps, d_out);
            c_warm.push_back(rw.cyc); ok_r &= rw.ok;

            flush_mall(d_flush, kFlush / 16, d_sink);
            hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, (const u4 *) d_dst_c, bytes / 16, d_sink);
            hipLaunchKernelGGL(k_read_rt, dim3(512), dim3(256), 0, 0, (const u4 *) d_dst_c, bytes / 16, d_sink);
            CHECK(hipDeviceSynchronize());
            bw_warm.push_back(time_read_bw((const u4 *) d_dst_c, bytes, d_sink, e0, e1));
        }
        double cold_m = median_v(c_cold), warm_m = median_v(c_warm);
        std::printf("RESULT store_mall KiB %zu ref cold_cyc %.1f warm_cyc %.1f delta %.1f cold_bw %.1f warm_bw %.1f GB/s ptr %s\n",
                    kib, cold_m, warm_m, cold_m - warm_m, median_v(bw_cold), median_v(bw_warm), ok_r ? "ok" : "WRONG");
        if (!ok_r) fail++;
        if (kib >= 16384 && (cold_m - warm_m) < 80.0) {
            std::printf("CONTROL store_mall_ref_%zu: FAIL cold-warm delta %.1f < 80 cyc\n", kib, cold_m - warm_m);
            fail++;
        }
        run_hint_row<0>(kib, (const u4 *) d_src_c, (u4 *) d_dst_c, d_flush, kFlush / 16,
                        d_stream, kFlush / 16, start, expect, steps, d_out, d_sink, e0, e1, fail);
    }

    // Free chase/flush buffers before allocating 512 MiB write-combining buffer
    CHECK(hipFree(d_src_c));
    CHECK(hipFree(d_dst_c));
    CHECK(hipFree(d_flush));
    CHECK(hipFree(d_stream));

    // ---- Part 3C: Cross-wave 32 B write combining ----
    mem::cool_down();
    unsigned * d_wc = nullptr;
    CHECK(hipMalloc(&d_wc, kWC));
    const size_t nlines = kWC / 128;
    const size_t nwords = kWC / 4;
    const int grid = 1024; // 16 WGs/CU, multiple of 4

    auto bench_wc = [&](const char * name, size_t logical_bytes, unsigned check_wpl, auto fn) {
        std::vector<double> gbps;
        unsigned seed = 0x5151u;
        // 1 warm + 5 timed
        for (int t = 0; t < 6; t++) {
            seed += 0x101u;
            CHECK(hipEventRecord(e0, 0));
            fn(seed);
            CHECK(hipEventRecord(e1, 0));
            CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            if (t > 0) gbps.push_back((double) logical_bytes / ((double) ms * 1e6));
        }
        CHECK(hipMemset(d_bad, 0, sizeof(unsigned)));
        unsigned verify_seed = do_break ? (seed ^ 1u) : seed;
        hipLaunchKernelGGL(k_verify_words, dim3(512), dim3(256), 0, 0, d_wc, nlines, verify_seed, check_wpl, d_bad);
        CHECK(hipDeviceSynchronize());
        unsigned bad = 0;
        CHECK(hipMemcpy(&bad, d_bad, sizeof(unsigned), hipMemcpyDeviceToHost));
        double med = median_v(gbps);
        double lo = *std::min_element(gbps.begin(), gbps.end());
        double hi = *std::max_element(gbps.begin(), gbps.end());
        double glines = (double) nlines / ((double) logical_bytes / (med * 1e9)) / 1e9;
        std::printf("RESULT write_combine %-20s logical_MiB %3zu bw %.1f (%.1f-%.1f) GB/s lines %.2f G/s verify %s (bad=%u)\n",
                    name, logical_bytes >> 20, med, lo, hi, glines, bad == 0 ? "ok" : "FAIL", bad);
        std::fflush(stdout);
        if (bad != 0) fail++;
    };

    bench_wc("full_b128", kWC, 32, [&](unsigned s) {
        hipLaunchKernelGGL(wc_full_b128, dim3(grid), dim3(256), 0, 0, (u4 *) d_wc, kWC / 16, s);
    });
    bench_wc("full_b32", kWC, 32, [&](unsigned s) {
        hipLaunchKernelGGL(wc_full_b32, dim3(grid), dim3(256), 0, 0, d_wc, nwords, s);
    });
    bench_wc("same_wave_4x32B", kWC, 32, [&](unsigned s) {
        hipLaunchKernelGGL(wc_same_wave_4x32B, dim3(grid), dim3(256), 0, 0, d_wc, nlines, s);
    });
    bench_wc("xwave_same_wg_32B", kWC, 32, [&](unsigned s) {
        hipLaunchKernelGGL(wc_xwave_same_wg_32B, dim3(grid), dim3(256), 0, 0, d_wc, nlines, s);
    });
    bench_wc("xwave_diff_wg_32B", kWC, 32, [&](unsigned s) {
        hipLaunchKernelGGL(wc_xwave_diff_wg_32B, dim3(grid), dim3(256), 0, 0, d_wc, nlines, s);
    });
    bench_wc("quarter_32B_only", kWC / 4, 8, [&](unsigned s) {
        hipLaunchKernelGGL(wc_quarter_32B_only, dim3(grid), dim3(256), 0, 0, d_wc, nlines, s);
    });
    bench_wc("eighth_16B_only", kWC / 8, 4, [&](unsigned s) {
        hipLaunchKernelGGL(wc_eighth_16B_only, dim3(grid), dim3(256), 0, 0, d_wc, nlines, s);
    });

    CHECK(hipFree(d_wc));
    CHECK(hipFree(d_sink));
    CHECK(hipFree(d_bad));
    CHECK(hipFree(d_out));
    CHECK(hipEventDestroy(e0));
    CHECK(hipEventDestroy(e1));
    mem::clocks("store_mall-end");

    std::printf("VERDICT store_mall: %s (fail=%d%s)\n", fail == 0 ? "PASS" : "FAIL", fail,
                do_break ? ", deliberate break" : "");
    return fail == 0 ? 0 : 1;
}
