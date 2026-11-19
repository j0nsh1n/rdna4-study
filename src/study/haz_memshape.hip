// haz_memshape: which global load instructions clang emits for a per-lane read of 8/16/32/64 bytes written in the
// usual HIP ways, whether the data comes out right, and what the split forms cost. One kernel per shape (k_<shape><bytes>);
// the instruction counts are checked from the -save-temps listing by scripts/mem_loopcheck.py (haz-expect/haz_memshape.txt).
// Run: haz_memshape [break]   ("break" compares against a deliberately shifted reference and must report FAIL)
#include "harness.h"

#include <cstdio>
#include <cstring>

typedef uint32_t v2u __attribute__((ext_vector_type(2)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));
typedef uint32_t v16u __attribute__((ext_vector_type(16)));

template <int B> struct VecOf;
template <> struct VecOf<8> { typedef v2u T; };
template <> struct VecOf<16> { typedef v4u T; };
template <> struct VecOf<32> { typedef v8u T; };
template <> struct VecOf<64> { typedef v16u T; };

// Every reader gets a base pointer that is only known to the compiler as a plain pointer, plus an item index; the
// run-time byte offset is folded into the base by the kernel, so alignment cannot be proven from the call site.
template <int B> struct RdStruct {  // shape 1
    static constexpr int W = B / 4;
    struct S { uint32_t x[B / 4]; };
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const S s = reinterpret_cast<const S *>(base)[i];
        for (int k = 0; k < W; ++k) r[k] = s.x[k];
    }
};
template <int B> struct RdMemcpy {  // shape 2
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        std::memcpy(r, base + i * B, B);
    }
};
template <int B> struct RdVec {  // shape 3
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const typename VecOf<B>::T v = reinterpret_cast<const typename VecOf<B>::T *>(base)[i];
        for (int k = 0; k < W; ++k) r[k] = v[k];
    }
};
template <int B> struct RdAssume {  // shape 6
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const char * p = static_cast<const char *>(__builtin_assume_aligned(base + i * B, B < 16 ? B : 16));
        std::memcpy(r, p, B);
    }
};
template <int B> struct RdMask {  // shape 7: clear the low address bits up to the natural alignment
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        constexpr uintptr_t m = (B < 16 ? B : 16) - 1;
        const char * p = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(base + i * B) & ~m);
        std::memcpy(r, p, B);
    }
};
template <int B> struct RdLoop {  // shape 8: element by element through a uint32_t pointer
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const uint32_t * q = reinterpret_cast<const uint32_t *>(base) + i * W;
        for (int k = 0; k < W; ++k) r[k] = q[k];
    }
};
struct RdUint4 {  // shape 4
    static constexpr int W = 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const uint4 v = reinterpret_cast<const uint4 *>(base)[i];
        r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
    }
};
struct RdFloat4 {  // shape 5
    static constexpr int W = 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const float4 v = reinterpret_cast<const float4 *>(base)[i];
        r[0] = __float_as_uint(v.x); r[1] = __float_as_uint(v.y); r[2] = __float_as_uint(v.z); r[3] = __float_as_uint(v.w);
    }
};
struct RdNtV4 {  // shape 9a: nontemporal load on the native vector
    static constexpr int W = 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const v4u v = __builtin_nontemporal_load(reinterpret_cast<const v4u *>(base) + i);
        for (int k = 0; k < 4; ++k) r[k] = v[k];
    }
};
// Shape 9b: the builtin rejects HIP's uint4 itself ("must be a pointer to integer, float, pointer, or a vector"), so the
// uint4 memory is read through a v4u pointer and the result is put into a uint4.
struct RdNtU4 {
    static constexpr int W = 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const v4u v = __builtin_nontemporal_load(reinterpret_cast<const v4u *>(base) + i);
        const uint4 u = make_uint4(v[0], v[1], v[2], v[3]);
        r[0] = u.x; r[1] = u.y; r[2] = u.z; r[3] = u.w;
    }
};

// Control, not a natural spelling: volatile word loads cannot be merged, so this is the forced 4 x global_load_b32 form
// that the timing comparison needs because clang never split any of the natural spellings.
template <int B> struct RdVol {
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        typedef __attribute__((address_space(1))) const volatile uint32_t gvu32;  // volatile blocks address-space inference
        gvu32 * q = (gvu32 *)(base) + i * W;
        for (int k = 0; k < W; ++k) r[k] = q[k];
    }
};

// Second split control without volatile's forced waits: the word stride is a run-time value (always 1), so the four
// adjacent loads cannot be proven adjacent and stay four independent global_load_b32.
__device__ uint32_t g_stride = 1;
template <int B> struct RdSplit {
    static constexpr int W = B / 4;
    static __device__ __forceinline__ void rd(const char * base, size_t i, uint32_t * r) {
        const uint32_t * q = reinterpret_cast<const uint32_t *>(base) + i * W;
        const uint32_t st = g_stride;
        for (int k = 0; k < W; ++k) r[k] = q[k * st];
    }
};

// X(shape, bytes, reader)
#define ALLCASES(X)                                                                                              \
    X(struct, 8, RdStruct<8>) X(struct, 16, RdStruct<16>) X(struct, 32, RdStruct<32>) X(struct, 64, RdStruct<64>) \
    X(memcpy, 8, RdMemcpy<8>) X(memcpy, 16, RdMemcpy<16>) X(memcpy, 32, RdMemcpy<32>) X(memcpy, 64, RdMemcpy<64>) \
    X(vec, 8, RdVec<8>) X(vec, 16, RdVec<16>) X(vec, 32, RdVec<32>) X(vec, 64, RdVec<64>)                         \
    X(mask, 8, RdMask<8>) X(mask, 16, RdMask<16>) X(mask, 32, RdMask<32>) X(mask, 64, RdMask<64>)                 \
    X(assume, 8, RdAssume<8>) X(assume, 16, RdAssume<16>) X(assume, 32, RdAssume<32>) X(assume, 64, RdAssume<64>) \
    X(loop, 8, RdLoop<8>) X(loop, 16, RdLoop<16>) X(loop, 32, RdLoop<32>) X(loop, 64, RdLoop<64>)                 \
    X(uint4, 16, RdUint4) X(float4, 16, RdFloat4) X(ntv4u, 16, RdNtV4) X(ntuint4, 16, RdNtU4) X(vol, 16, RdVol<16>) X(split, 16, RdSplit<16>)

// The 16-byte shapes that are also timed.
#define TIMED(X)                                                                                                \
    X(struct, 16, RdStruct<16>) X(memcpy, 16, RdMemcpy<16>) X(vec, 16, RdVec<16>) X(mask, 16, RdMask<16>)       \
    X(assume, 16, RdAssume<16>) X(loop, 16, RdLoop<16>) X(uint4, 16, RdUint4) X(float4, 16, RdFloat4)           \
    X(ntv4u, 16, RdNtV4) X(vol, 16, RdVol<16>) X(split, 16, RdSplit<16>)

// Correctness kernels: every lane reads one B-byte item and stores its words one by one.
#define DEF_COPY(sh, B, RD)                                                                                      \
    __global__ __launch_bounds__(256) void k_##sh##B(const char * in, uint32_t * out, size_t off, int n) { \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;                                                     \
        if (i < n) {                                                                                             \
            uint32_t r[RD::W];                                                                                   \
            RD::rd(in + off, i, r);                                                                              \
            for (int k = 0; k < RD::W; ++k) out[(size_t)i * RD::W + k] = r[k];                                    \
        }                                                                                                        \
    }
ALLCASES(DEF_COPY)

// Bandwidth kernels: 4 independent items per thread, xor of everything read; the store happens only on a value the
// compiler cannot know, so the loads cannot be removed and no output traffic is added.
#define DEF_BW(sh, B, RD)                                                                                        \
    __global__ __launch_bounds__(256) void kb_##sh##B(const char * in, uint32_t * out, size_t off,     \
                                                                 size_t q, uint32_t sentinel) {                  \
        const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                                          \
        uint32_t acc = 0;                                                                                        \
        for (int j = 0; j < 4; ++j) {                                                                            \
            uint32_t r[RD::W];                                                                                   \
            RD::rd(in + off, i + j * q, r);                                                                      \
            for (int k = 0; k < RD::W; ++k) acc ^= r[k];                                                         \
        }                                                                                                        \
        if (acc == sentinel) out[0] = acc;                                                                       \
    }
TIMED(DEF_BW)

// Single-wave kernels: one wave of 32 lanes issues 64 independent 16-byte loads per pass (512 contiguous bytes each),
// then consumes them; two passes over the same bytes (the second through a run-time base, so it is not merged with the
// first). Cycles of each pass go to res[0], res[1].
#define DEF_CYC(sh, B, RD)                                                                                       \
    __global__ __launch_bounds__(32) void kc_##sh##B(const char * in, uint64_t * res, size_t off,     \
                                                                size_t off2, uint32_t * sink) {                  \
        const int lane = threadIdx.x;                                                                            \
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;                                                                 \
        uint64_t cy[2];                                                                                          \
        for (int pass = 0; pass < 2; ++pass) {                                                                   \
            const char * b = in + (pass ? off2 : off);                                                           \
            asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));                                           \
            const uint64_t t0 = wave_cycles();                                                                   \
            _Pragma("unroll") for (int j = 0; j < 64; ++j) {                                                     \
                uint32_t r[RD::W];                                                                               \
                RD::rd(b, (size_t)j * 32 + lane, r);                                                             \
                a0 ^= r[0]; a1 ^= r[1]; a2 ^= r[2]; a3 ^= r[3];                                                  \
            }                                                                                                    \
            asm volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3));                                           \
            cy[pass] = cycle_diff(t0, wave_cycles());                                                            \
        }                                                                                                        \
        if (lane == 0) { res[0] = cy[0]; res[1] = cy[1]; }                                                       \
        sink[lane] = a0 ^ a1 ^ a2 ^ a3;                                                                          \
    }
TIMED(DEF_CYC)

// Flush kernel: streams a large buffer so the single-wave first pass starts with cold caches.
__global__ __launch_bounds__(256) void k_flush(const uint4 * p, uint32_t * out, size_t n, uint32_t sentinel) {
    uint32_t acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == sentinel) out[0] = acc;
}

struct Case {
    const char * kernel;
    const char * name;
    int bytes;
    void (*fn)(const char *, uint32_t *, size_t, int);
};
#define ENT(sh, B, RD) {"k_" #sh #B, "haz_memshape_" #sh #B, B, k_##sh##B},
static const Case kCases[] = {ALLCASES(ENT)};

struct Bw {
    const char * name;
    void (*fn)(const char *, uint32_t *, size_t, size_t, uint32_t);
    void (*cyc)(const char *, uint64_t *, size_t, size_t, uint32_t *);
};
#define ENTB(sh, B, RD) {#sh #B, kb_##sh##B, kc_##sh##B},
static const Bw kBw[] = {TIMED(ENTB)};

static uint32_t lcg(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return s ^ (s >> 15);
}

int main(int argc, char ** argv) {
    const bool brk = argc > 1 && std::strcmp(argv[1], "break") == 0;
    hipDeviceProp_t prop = require_arch();
    std::printf("device 0: %s (%s)%s\n", prop.name, prop.gcnArchName, brk ? "  [break mode: reference is shifted on purpose]" : "");

    // ---- correctness: input of random words, read at a run-time offset of 64 bytes (a multiple of every width) ----
    const int n = 1 << 16;
    const size_t off = 64;
    const size_t in_words = (size_t)n * 16 + 64;
    std::vector<uint32_t> h_in(in_words);
    uint32_t seed = 12345u;
    for (auto & w : h_in) w = lcg(seed);
    char * d_in;
    uint32_t * d_out;
    CHECK(hipMalloc(&d_in, in_words * 4));
    CHECK(hipMalloc(&d_out, (size_t)n * 16 * 4));
    CHECK(hipMemcpy(d_in, h_in.data(), in_words * 4, hipMemcpyHostToDevice));
    int fails = 0;
    for (const Case & c : kCases) {
        const int W = c.bytes / 4;
        CHECK(hipMemset(d_out, 0, (size_t)n * W * 4));
        const char * a0 = d_in;
        uint32_t * a1 = d_out;
        size_t a2 = off;
        int a3 = n;
        void * args[] = {&a0, &a1, &a2, &a3};
        CHECK(hipLaunchKernel((const void *)c.fn, dim3(n / 256), dim3(256), args, 0, nullptr));
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h_out((size_t)n * W);
        CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * 4, hipMemcpyDeviceToHost));
        size_t bad = 0, first = 0;
        for (size_t k = 0; k < h_out.size(); ++k) {
            const uint32_t ref = h_in[off / 4 + k + (brk ? 1 : 0)];
            if (h_out[k] != ref) {
                if (!bad) first = k;
                ++bad;
            }
        }
        char txt[160];
        if (bad) {
            std::snprintf(txt, sizeof txt, "%zu of %zu words differ (first at %zu)", bad, h_out.size(), first);
            ++fails;
        } else {
            std::snprintf(txt, sizeof txt, "%zu words equal the input", h_out.size());
        }
        verdict(c.name, bad ? "FAIL" : "PASS", txt);
    }
    CHECK(hipFree(d_out));

    // ---- cost: whole card, 64 MB and 1 GiB buffers; 4 items of 16 bytes per thread ----
    const size_t sizes[2] = {64u << 20, 1u << 30};
    char * d_big;
    CHECK(hipMalloc(&d_big, (1u << 30) + 64));
    uint32_t * d_sink;
    CHECK(hipMalloc(&d_sink, 4096));
    {
        std::vector<uint32_t> fill((1u << 30) / 4);
        for (auto & w : fill) w = lcg(seed);
        CHECK(hipMemcpy(d_big, fill.data(), 1u << 30, hipMemcpyHostToDevice));
    }
    std::printf("\nMEASURED whole-card read, 16 bytes per lane, 4 items per thread (median of runs, GB/s = 1e9 B/s; min-max in brackets)\n");
    for (size_t sz : sizes) {
        const size_t items = sz / 16, q = items / 4;
        const int blocks = (int)(q / 256);
        std::printf("buffer %zu MiB:\n", sz >> 20);
        for (const Bw & b : kBw) {
            const char * a0 = d_big;
            uint32_t * a1 = d_sink;
            size_t a2 = 0, a3 = q;
            uint32_t a4 = 0xdeadbeefu;
            void * args[] = {&a0, &a1, &a2, &a3, &a4};
            const int reps = sz > (256u << 20) ? 20 : 100;
            auto ms = time_ms([&] { hipLaunchKernel((const void *)b.fn, dim3(blocks), dim3(256), args, 0, nullptr); }, reps);
            const Stats s = stats_of(ms);
            std::printf("  %-8s %7.1f GB/s  (%.1f-%.1f)  median %.4f ms\n", b.name, sz / (s.median * 1e6),
                        sz / (s.max * 1e6), sz / (s.min * 1e6), s.median);
        }
    }

    // ---- single wave: 64 independent loads, cycles for the whole group; first pass cold-ish, second warm ----
    std::printf("\nMEASURED single wave (32 lanes), 64 independent 16-byte loads then consume; cycles for the pass, per load in brackets\n");
    uint64_t * d_res;
    CHECK(hipMalloc(&d_res, 16));
    char * d_fl = d_big + (64u << 20);  // flush region: 512 MiB away from the probed 32 KiB
    for (const Bw & b : kBw) {
        std::vector<uint64_t> p1, p2;
        for (int rep = 0; rep < 15; ++rep) {
            const uint4 * fp = reinterpret_cast<const uint4 *>(d_fl);
            size_t fn_ = (512u << 20) / 16;
            uint32_t fs = 0xdeadbeefu;
            uint32_t * fo = d_sink;
            void * fargs[] = {&fp, &fo, &fn_, &fs};
            CHECK(hipLaunchKernel((const void *)k_flush, dim3(4096), dim3(256), fargs, 0, nullptr));
            const char * a0 = d_big;
            size_t a2 = 0, a3 = 0;
            uint32_t * a4 = d_sink + 64;
            void * args[] = {&a0, &d_res, &a2, &a3, &a4};
            CHECK(hipLaunchKernel((const void *)b.cyc, dim3(1), dim3(32), args, 0, nullptr));
            uint64_t h[2];
            CHECK(hipMemcpy(h, d_res, 16, hipMemcpyDeviceToHost));
            p1.push_back(h[0]);
            p2.push_back(h[1]);
        }
        std::sort(p1.begin(), p1.end());
        std::sort(p2.begin(), p2.end());
        std::printf("  %-8s cold pass %6llu (%.1f/load, min %llu)   warm pass %6llu (%.1f/load, min %llu)\n", b.name,
                    (unsigned long long)p1[7], p1[7] / 64.0, (unsigned long long)p1[0], (unsigned long long)p2[7],
                    p2[7] / 64.0, (unsigned long long)p2[0]);
    }
    CHECK(hipFree(d_big));
    CHECK(hipFree(d_in));
    CHECK(hipFree(d_sink));
    CHECK(hipFree(d_res));
    std::printf("\n%s\n", fails ? "RESULT: FAIL" : "RESULT: PASS");
    return fails ? 1 : 0;
}
