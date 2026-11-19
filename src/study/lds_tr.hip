// Lane 3, item 5: the transposing loads global_load_tr_b128 (16-bit elements) and global_load_tr_b64 (8-bit).
//
// Usage: lds_tr <calib|layout|wmma|inflight|inflight2|mask|bw> [--break]
//   calib     the FMA calibration (lds_common.h).
//   layout    marker data at random 16-byte (8-byte) aligned per-lane addresses; every 16-bit (8-bit) element each
//             lane receives is compared with the model below. Then misaligned lane addresses: which address the
//             instruction really uses. --break swaps lanes 3 and 5 in the model: the verdict must FAIL.
//   wmma      FP16 and FP8 products D = A * B with A row-major (plain load) and B row-major (transposing load, no
//             shuffle), against a CPU reference; exact with small integer data. Control: B through a plain load
//             must give a wrong product (proves the check can fail).
//   inflight  one wave, N loads of whole 16 x 16 tiles then one wait; plain b128 against tr_b128, plain b64 against
//             tr_b64; data in L0, L2 and memory. N = 1 gives the latency (lane 2: plain b128 135 / 314 / 772).
//   inflight2 (leftovers C, LD-6a) the same past 24 in flight: N = 24, 32, 40, 48, 56 for the b128 forms (4 VGPRs
//             per load, so 56 is near the register limit) and N = 24, 32, 48, 63, 64, 72 for the b64 forms; data in L2
//             and memory. Calibration and data check as inflight; the per-load cost past 24 and the step at the load
//             counter's limit (63, lane 2 item 5) are INFO.
//   mask      (leftovers C, LD-6b) EXEC partial around one transposing load (7 masks); destination preset to a
//             sentinel; every lane's address valid. For each output element: the model's value (from the source
//             lane's address), the sentinel (not written), zero, or other; INFO. The full mask must equal the model
//             (--break swaps lanes 3 and 5 in the model: the verdict must FAIL).
//   bw        whole card, every wave reads tiles of a window (4 in flight): L0, L2, memory-side cache, DRAM.
//   bw2 KIND SHARE WINDOW_MIB (leftovers C, LD-9): one burst of about 0.1 s (started below 55 C, stopped at 95 C).
//             KIND b128 or tr_b128; SHARE "shared" = bw's walk (wave w reads 4-tile group w + s x waves at step s,
//             so with a 32 MiB window each wave alternates between the same two groups, 4 KiB in all) or "unique"
//             (wave w walks its own run of groups w x steps + s, so no wave ever rereads its own data; rereads of
//             the window come from other waves, long after). Prints one TRBW line.
//
// Model (MEASURED in development, checked by `layout`), lanes in groups of 8 for 16-bit and 16 for 8-bit:
//   tr_b128: lane l, element i (0..7, two per VGPR) = element (l % 8) of the 16 bytes at the address of lane
//            8 * (l / 8) + i. An 8 x 8 transpose of 16-bit elements inside each group of 8 lanes.
//   tr_b64:  lane l, byte i (0..7, four per VGPR) = byte (l % 8) of the 8 bytes at the address of lane
//            16 * (l / 16) + 4 * ((l / 8) % 2) + (i % 4) + 8 * (i / 4).
// Addresses that give the matrix operand (WMMA B layout: lane c + 16 * (k / 8) holds K = 8 * (k / 8) .. +7 of
// column c) from a row-major K x N matrix with leading dimension ld:
//   tr_b128: lane s points at row 8 * (s / 16) + s % 8, column 8 * ((s / 8) % 2).
//   tr_b64:  lane s points at row 8 * (s / 16) + 4 * ((s / 8) % 2) + s % 4, column 8 * ((s / 4) % 2).
#include "lds_common.h"

#include <atomic>
#include <map>
#include <thread>
#include <numeric>
#include <random>

typedef short v8s __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));
#define GPTR(T, p) ((T __attribute__((address_space(1))) *) (p))

static int tr128_src_lane(int l, int i) { return 8 * (l / 8) + i; }
static int tr64_src_lane(int l, int i) { return 16 * (l / 16) + 4 * ((l / 8) % 2) + (i % 4) + 8 * (i / 4); }
__host__ __device__ inline uint32_t tr128_b_off(int s, int ld) {  // bytes, 16-bit elements
    return (uint32_t) (((8 * (s / 16) + s % 8) * ld + 8 * ((s / 8) % 2)) * 2);
}
__host__ __device__ inline uint32_t tr64_b_off(int s, int ld) {  // bytes, 8-bit elements
    return (uint32_t) ((8 * (s / 16) + 4 * ((s / 8) % 2) + s % 4) * ld + 8 * ((s / 4) % 2));
}

// ---------------------------------------------------------------------------------------------------------------
// layout

__global__ __launch_bounds__(32) void k_layout(const char * base, const uint32_t * off, uint32_t * out) {
    const int l = threadIdx.x;
    const v8s a = __builtin_amdgcn_global_load_tr_b128_v8i16(GPTR(v8s, base + off[l]));
    const v2i b = __builtin_amdgcn_global_load_tr_b64_v2i32(GPTR(v2i, base + off[32 + l]));
    for (int i = 0; i < 8; ++i) {
        out[l * 8 + i] = (uint16_t) a[i];
    }
    out[256 + l * 2] = (uint32_t) b[0];
    out[256 + l * 2 + 1] = (uint32_t) b[1];
}

struct Dev {
    char * buf = nullptr;
    uint32_t * off = nullptr;
    uint32_t * out = nullptr;
};

static void run_layout(Dev & d, const std::vector<uint8_t> & mem, const std::vector<uint32_t> & off, std::vector<uint32_t> & res) {
    CHECK(hipMemcpy(d.buf, mem.data(), mem.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d.off, off.data(), 64 * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_layout, dim3(1), dim3(32), 0, 0, d.buf, d.off, d.out);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    res.resize(320);
    CHECK(hipMemcpy(res.data(), d.out, 320 * 4, hipMemcpyDeviceToHost));
}

static uint16_t h16(const std::vector<uint8_t> & m, size_t byte) { return (uint16_t) (m[byte] | (m[byte + 1] << 8)); }
static uint16_t got128(const std::vector<uint32_t> & r, int l, int i) { return (uint16_t) r[l * 8 + i]; }
static uint8_t got64(const std::vector<uint32_t> & r, int l, int i) { return (uint8_t) (r[256 + l * 2 + i / 4] >> (8 * (i % 4))); }

static void mode_layout(bool brk) {
    Dev d;
    const size_t MEM = 65536;
    CHECK(hipMalloc(&d.buf, MEM + 64));
    CHECK(hipMalloc(&d.off, 64 * 4));
    CHECK(hipMalloc(&d.out, 320 * 4));
    std::mt19937 rng(11);
    int bad128 = 0, bad64 = 0, trials = 0;
    for (int t = 0; t < 8; ++t, ++trials) {
        // random distinct slots: 16-byte slots for tr_b128, 8-byte slots in the upper half for tr_b64
        std::vector<uint32_t> slots(2048);
        std::iota(slots.begin(), slots.end(), 0u);
        std::shuffle(slots.begin(), slots.end(), rng);
        std::vector<uint32_t> off(64);
        std::vector<uint8_t> mem(MEM + 64);
        for (auto & b : mem) {
            b = (uint8_t) (0x80 | (rng() & 0x7f));  // filler bytes >= 0x80 never equal a marker
        }
        for (int l = 0; l < 32; ++l) {
            off[l] = slots[l] * 16;
            off[32 + l] = 32768 + slots[32 + l] * 8;
            for (int e = 0; e < 8; ++e) {  // 16-bit marker: lane * 8 + element (< 0x80 in each byte)
                const uint16_t v = (uint16_t) (l * 8 + e);
                mem[off[l] + 2 * e] = (uint8_t) (v & 0x7f);
                mem[off[l] + 2 * e + 1] = (uint8_t) (v >> 7);
                mem[off[32 + l] + e] = (uint8_t) (l * 8 + e);  // 8-bit marker: lane * 8 + byte
            }
        }
        std::vector<uint32_t> r;
        run_layout(d, mem, off, r);
        for (int l = 0; l < 32; ++l) {
            for (int i = 0; i < 8; ++i) {
                int s = tr128_src_lane(l, i);
                if (brk) {
                    s = s == 3 ? 5 : s == 5 ? 3 : s;
                }
                bad128 += got128(r, l, i) != h16(mem, off[s] + 2 * (l % 8));
                int s8 = tr64_src_lane(l, i);
                if (brk) {
                    s8 = s8 == 3 ? 5 : s8 == 5 ? 3 : s8;
                }
                bad64 += got64(r, l, i) != mem[off[32 + s8] + (l % 8)];
            }
        }
    }
    lds::check("tr_layout", bad128 == 0 && bad64 == 0,
               "%d trials of random per-lane addresses: tr_b128 %d of %d elements differ from the model, tr_b64 %d of %d%s",
               trials, bad128, trials * 256, bad64, trials * 256, brk ? " (break: model with lanes 3 and 5 swapped)" : "");

    // Misaligned lane addresses: marker = byte address (16-bit words hold their own word index, bytes are unique
    // inside the 256-byte window used for tr_b64).
    std::vector<uint8_t> mem(MEM + 64);
    for (size_t i = 0; i + 1 < mem.size(); i += 2) {
        mem[i] = (uint8_t) ((i / 2) & 255);
        mem[i + 1] = (uint8_t) ((i / 2) >> 8);
    }
    for (int i = 0; i < 512; ++i) {
        mem[32768 + i] = (uint8_t) (i * 37 + 11);  // unique within any 256 consecutive bytes
    }
    for (int mis : {0, 2, 4, 8}) {
        std::vector<uint32_t> off(64);
        for (int l = 0; l < 32; ++l) {
            off[l] = 1024 + tr128_b_off(l, 16) + mis;
            off[32 + l] = 32768 + 64 + tr64_b_off(l, 16) + (mis == 8 ? 4 : mis / 2);
        }
        std::vector<uint32_t> r;
        run_layout(d, mem, off, r);
        const int mis64 = mis == 8 ? 4 : mis / 2;
        // hypotheses: exact byte address, or the address rounded down to the access size
        int ex128 = 0, dn128 = 0, ex64 = 0, dn64 = 0, dn64_4 = 0;
        for (int l = 0; l < 32; ++l) {
            for (int i = 0; i < 8; ++i) {
                const size_t a = off[tr128_src_lane(l, i)] + 2 * (l % 8);
                ex128 += got128(r, l, i) == h16(mem, a);
                dn128 += got128(r, l, i) == h16(mem, (off[tr128_src_lane(l, i)] & ~15u) + 2 * (l % 8));
                const size_t b = off[32 + tr64_src_lane(l, i)];
                ex64 += got64(r, l, i) == mem[b + (l % 8)];
                dn64 += got64(r, l, i) == mem[(b & ~7u) + (l % 8)];
                dn64_4 += got64(r, l, i) == mem[(b & ~3u) + (l % 8)];
            }
        }
        std::printf("RESULT tr misalign b128_bytes %d exact %d/256 rounded16 %d/256 | b64_bytes %d exact %d/256 rounded8 %d/256 "
                    "rounded4 %d/256\n",
                    mis, ex128, dn128, mis64, ex64, dn64, dn64_4);
    }
    CHECK(hipFree(d.buf));
    CHECK(hipFree(d.off));
    CHECK(hipFree(d.out));
}

// ---------------------------------------------------------------------------------------------------------------
// wmma: one wave, one 16 x 16 x 16 product. A row-major (ld 16 + pad), B row-major with leading dimension ldb.
// tr = 1: B through the transposing load; tr = 0 (control): B through a plain load at the A-style address.

__global__ __launch_bounds__(32) void k_mm_f16(const _Float16 * A, const _Float16 * B, int ldb, int tr, float * D) {
    const int l = threadIdx.x;
    const v8h a = *(const v8h *) (A + (l % 16) * 16 + 8 * (l / 16));
    v8h b;
    if (tr) {
        b = __builtin_bit_cast(v8h, __builtin_amdgcn_global_load_tr_b128_v8i16(GPTR(v8s, (const char *) B + tr128_b_off(l, ldb))));
    } else {
        b = *(const v8h *) (B + (l % 16) * ldb + 8 * (l / 16));
    }
    const v8f c = {};
    const v8f d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) {
        D[l * 8 + i] = d[i];
    }
}

__global__ __launch_bounds__(32) void k_mm_fp8(const uint8_t * A, const uint8_t * B, int ldb, int tr, float * D) {
    const int l = threadIdx.x;
    const v2i a = *(const v2i *) (A + (l % 16) * 16 + 8 * (l / 16));
    v2i b;
    if (tr) {
        b = __builtin_amdgcn_global_load_tr_b64_v2i32(GPTR(v2i, B + tr64_b_off(l, ldb)));
    } else {
        b = *(const v2i *) (B + (l % 16) * ldb + 8 * (l / 16));
    }
    const v8f c = {};
    const v8f d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) {
        D[l * 8 + i] = d[i];
    }
}

static void mode_wmma(bool brk) {
    std::mt19937 rng(5);
    // E4M3 codes of -2, -1, -0.5, 0, 0.5, 1, 2 (exact in FP16 too)
    const float vals[7] = {-2, -1, -0.5f, 0, 0.5f, 1, 2};
    const uint8_t e4m3[7] = {0xC0, 0xB8, 0xB0, 0x00, 0x30, 0x38, 0x40};
    void *dA, *dB;
    float * dD;
    CHECK(hipMalloc(&dA, 4096));
    CHECK(hipMalloc(&dB, 16 * 128 * 2));
    CHECK(hipMalloc(&dD, 256 * 4));
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        for (int ldb : {16, 64}) {
            std::vector<int> ia(256), ib(16 * ldb);
            for (auto & x : ia) x = (int) (rng() % 7);
            for (auto & x : ib) x = (int) (rng() % 7);
            std::vector<_Float16> hA(256), hB(16 * ldb);
            std::vector<uint8_t> qA(256), qB(16 * ldb);
            for (int i = 0; i < 256; ++i) {
                hA[i] = (_Float16) vals[ia[i]];
                qA[i] = e4m3[ia[i]];
            }
            for (int i = 0; i < 16 * ldb; ++i) {
                hB[i] = (_Float16) vals[ib[i]];
                qB[i] = e4m3[ib[i]];
            }
            if (fp8) {
                CHECK(hipMemcpy(dA, qA.data(), 256, hipMemcpyHostToDevice));
                CHECK(hipMemcpy(dB, qB.data(), qB.size(), hipMemcpyHostToDevice));
            } else {
                CHECK(hipMemcpy(dA, hA.data(), 512, hipMemcpyHostToDevice));
                CHECK(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
            }
            int wrong[2] = {0, 0};
            for (int tr = 0; tr < 2; ++tr) {
                if (fp8) {
                    hipLaunchKernelGGL(k_mm_fp8, dim3(1), dim3(32), 0, 0, (const uint8_t *) dA, (const uint8_t *) dB, ldb, tr, dD);
                } else {
                    hipLaunchKernelGGL(k_mm_f16, dim3(1), dim3(32), 0, 0, (const _Float16 *) dA, (const _Float16 *) dB, ldb, tr, dD);
                }
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                std::vector<float> D(256);
                CHECK(hipMemcpy(D.data(), dD, 1024, hipMemcpyDeviceToHost));
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        double ref = 0;
                        for (int k = 0; k < 16; ++k) {
                            ref += (double) vals[ia[r * 16 + k]] * vals[ib[k * ldb + c]];
                        }
                        if (brk && r == 0 && c == 0) {
                            ref += 1;  // the break: an expectation one unit off
                        }
                        const float got = D[(16 * (r / 8) + c) * 8 + r % 8];
                        wrong[tr] += got != (float) ref;
                    }
                }
            }
            char name[32];
            std::snprintf(name, sizeof name, "tr_wmma_%s_ld%d", fp8 ? "fp8" : "f16", ldb);
            lds::check(name, wrong[1] == 0 && wrong[0] > 0,
                       "%s A row-major x B row-major (ld %d) via %s: %d of 256 outputs differ from the CPU; control with "
                       "a plain load of B: %d differ (must be > 0)",
                       fp8 ? "FP8" : "FP16", ldb, fp8 ? "global_load_tr_b64" : "global_load_tr_b128", wrong[1], wrong[0]);
        }
    }
    CHECK(hipFree(dA));
    CHECK(hipFree(dB));
    CHECK(hipFree(dD));
}

// ---------------------------------------------------------------------------------------------------------------
// inflight: KIND 0 plain b128, 1 tr_b128 (tiles of 512 bytes); 2 plain b64, 3 tr_b64 (tiles of 256 bytes).

template <int KIND>
struct Ld {
    static constexpr int TILE = KIND < 2 ? 512 : 256;
    using T = typename std::conditional<KIND < 2, u4, u2>::type;
};

template <int KIND, int K>
__device__ __forceinline__ void ldk(typename Ld<KIND>::T & r, const char * base, uint32_t lane_off) {
    if constexpr (KIND == 0) {
        asm volatile("global_load_b128 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 512) : "memory");
    } else if constexpr (KIND == 1) {
        asm volatile("global_load_tr_b128 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 512) : "memory");
    } else if constexpr (KIND == 2) {
        asm volatile("global_load_b64 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 256) : "memory");
    } else {
        asm volatile("global_load_tr_b64 %0, %1, %2 offset:%3" : "=v"(r) : "v"(lane_off), "s"(base), "i"(K * 256) : "memory");
    }
}

template <int KIND, int K, int N>
__device__ __forceinline__ void issue(typename Ld<KIND>::T (&r)[N], const char * base, uint32_t lane_off) {
    if constexpr (K < N) {
        ldk<KIND, K>(r[K], base, lane_off);
        issue<KIND, K + 1, N>(r, base, lane_off);
    }
}

template <int KIND, int N>
__device__ __forceinline__ void wait_all(typename Ld<KIND>::T (&r)[N]) {
    asm volatile("s_wait_loadcnt 0" ::: "memory");
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" : "+v"(r[k]));
    }
}

__device__ __forceinline__ uint32_t lane_offset(int kind, int l) {
    switch (kind) {
        case 0: return l * 16;
        case 1: return tr128_b_off(l, 16);
        case 2: return l * 8;
        default: return tr64_b_off(l, 16);
    }
}

__device__ __forceinline__ uint32_t bytefold(uint32_t x) { return (x ^ (x >> 8) ^ (x >> 16) ^ (x >> 24)) & 255u; }

// out: [0] cycles, [1] xor of every byte received in the last batch (permutation-invariant check)
template <int KIND, int N>
__global__ __launch_bounds__(32) void k_inflight(const char * base, uint32_t batches, uint32_t wrap, uint32_t warm, uint64_t * out) {
    using T = typename Ld<KIND>::T;
    constexpr uint32_t BATCH = N * Ld<KIND>::TILE;
    const uint32_t lo = lane_offset(KIND, threadIdx.x);
    T r[N];
#pragma unroll 1
    for (uint32_t b = 0; b < warm; ++b) {
        issue<KIND, 0, N>(r, base + (size_t) (b % wrap) * BATCH, lo);
        wait_all<KIND, N>(r);
    }
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t b = 0; b < batches; ++b) {
        issue<KIND, 0, N>(r, base + (size_t) (b % wrap) * BATCH, lo);
        wait_all<KIND, N>(r);
    }
    const uint64_t c1 = wave_cycles();
    uint32_t x = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        if constexpr (KIND < 2) {
            x ^= bytefold(r[k].x ^ r[k].y ^ r[k].z ^ r[k].w);
        } else {
            x ^= bytefold(r[k].x ^ r[k].y);
        }
    }
    for (int m = 16; m >= 1; m /= 2) {
        x ^= __shfl_xor(x, m);
    }
    if (threadIdx.x == 0) {
        out[0] = cycle_diff(c0, c1);
        out[1] = x;
    }
}

using InflightK = void (*)(const char *, uint32_t, uint32_t, uint32_t, uint64_t *);

template <int KIND>
static InflightK pick(int n) {
    switch (n) {
        case 1: return k_inflight<KIND, 1>;
        case 2: return k_inflight<KIND, 2>;
        case 4: return k_inflight<KIND, 4>;
        case 8: return k_inflight<KIND, 8>;
        case 16: return k_inflight<KIND, 16>;
        default: return k_inflight<KIND, 24>;
    }
}
// LD-6a: larger batches (b128 kinds up to 56 loads = 224 VGPRs; b64 kinds up to 72 = 144 VGPRs).
template <int KIND>
static InflightK pick2(int n) {
    if constexpr (KIND < 2) {
        switch (n) {
            case 1: return k_inflight<KIND, 1>;
            case 24: return k_inflight<KIND, 24>;
            case 32: return k_inflight<KIND, 32>;
            case 40: return k_inflight<KIND, 40>;
            case 48: return k_inflight<KIND, 48>;
            default: return k_inflight<KIND, 56>;
        }
    } else {
        switch (n) {
            case 1: return k_inflight<KIND, 1>;
            case 24: return k_inflight<KIND, 24>;
            case 32: return k_inflight<KIND, 32>;
            case 48: return k_inflight<KIND, 48>;
            case 63: return k_inflight<KIND, 63>;
            case 64: return k_inflight<KIND, 64>;
            default: return k_inflight<KIND, 72>;
        }
    }
}

static void mode_inflight2(bool brk) {
    const size_t BIG = 768ull << 20;
    char * buf;
    uint64_t * out;
    CHECK(hipMalloc(&buf, BIG));
    CHECK(hipMalloc(&out, 64));
    mem::fill(buf, BIG, mem::RANDOM, 3);
    std::vector<uint8_t> host_tail(4 << 20);
    const size_t cached_at = BIG - host_tail.size();
    CHECK(hipMemcpy(host_tail.data(), buf + cached_at, host_tail.size(), hipMemcpyDeviceToHost));
    size_t fresh = 0;
    const char * kname[4] = {"b128", "tr_b128", "b64", "tr_b64"};
    int badx = 0, runs = 0;
    double lat[3] = {};
    std::map<std::string, double> med;
    for (int level = 1; level < 3; ++level) {
        for (int kind = 0; kind < 4; ++kind) {
            const std::vector<int> ns = kind < 2 ? std::vector<int>{1, 24, 32, 40, 48, 56} : std::vector<int>{1, 24, 32, 48, 63, 64, 72};
            for (int n : ns) {
                const uint32_t tile = kind < 2 ? 512 : 256, batch = n * tile;
                std::vector<double> cyc;
                for (int rep = 0; rep < 6; ++rep) {
                    uint32_t batches, wrap, warm;
                    const char * base;
                    if (level == 1) {
                        wrap = std::max(1u, (1u << 20) / batch);
                        batches = 256, warm = wrap, base = buf + cached_at;
                    } else {
                        batches = wrap = 32, warm = 0, base = buf + fresh;
                        fresh += (size_t) batches * batch;
                    }
                    InflightK k = kind == 0 ? pick2<0>(n) : kind == 1 ? pick2<1>(n) : kind == 2 ? pick2<2>(n) : pick2<3>(n);
                    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, base, batches, wrap, warm, out);
                    CHECK(hipGetLastError());
                    CHECK(hipDeviceSynchronize());
                    uint64_t o[2];
                    CHECK(hipMemcpy(o, out, 16, hipMemcpyDeviceToHost));
                    if (rep > 0) {
                        cyc.push_back((double) o[0] / batches);
                    }
                    if (level == 1) {
                        const size_t at = (size_t) ((batches - 1) % wrap) * batch;
                        uint32_t x = 0;
                        for (size_t i = 0; i < batch; ++i) {
                            x ^= host_tail[at + i];
                        }
                        if (brk) {
                            x ^= 1;
                        }
                        badx += x != o[1];
                        ++runs;
                    }
                }
                const Stats st = stats_of(cyc);
                if (n == 1 && kind == 0) {
                    lat[level] = st.median;
                }
                med[std::string(level == 1 ? "L2 " : "memory ") + kname[kind] + " " + std::to_string(n)] = st.median;
                std::printf("RESULT tr inflight2 level %s kind %s n %d cycles_per_batch %.1f min %.1f max %.1f per_load %.2f "
                            "bytes_per_cycle %.1f\n",
                            level == 1 ? "L2" : "memory", kname[kind], n, st.median, st.min, st.max, st.median / n,
                            n * tile / st.median);
            }
        }
    }
    if (fresh > BIG) {
        lds::check("tr_inflight2_space", false, "memory runs used %zu bytes, more than the buffer", fresh);
    }
    // INFO: the transposing form against the plain form at each N, and the step at the load counter's limit
    for (const char * lv : {"L2", "memory"}) {
        for (int kind : {1, 3}) {
            const std::vector<int> ns = kind < 2 ? std::vector<int>{24, 32, 40, 48, 56} : std::vector<int>{24, 32, 48, 63, 64, 72};
            std::string line;
            for (int n : ns) {
                const std::string k1 = std::string(lv) + " " + kname[kind] + " " + std::to_string(n);
                const std::string k0 = std::string(lv) + " " + kname[kind - 1] + " " + std::to_string(n);
                char b[64];
                std::snprintf(b, sizeof b, " N%d %.3f", n, med[k1] / med[k0]);
                line += b;
            }
            std::printf("INFO tr inflight2 %s %s cycles_per_batch relative to the plain form:%s\n", lv, kname[kind], line.c_str());
        }
        for (int kind : {2, 3}) {
            auto m = [&](int n) { return med[std::string(lv) + " " + kname[kind] + " " + std::to_string(n)]; };
            std::printf("INFO tr inflight2 %s %s per extra load: 48->63 %.2f cycles, 63->64 %.2f cycles, 64->72 %.2f cycles\n",
                        lv, kname[kind], (m(63) - m(48)) / 15, m(64) - m(63), (m(72) - m(64)) / 8);
        }
    }
    lds::check("tr_inflight2_data", badx == 0, "%d of %d L2 runs returned bytes whose xor differs from the data%s", badx, runs,
               brk ? " (break: expected xor off by one)" : "");
    // known answer: plain b128 N = 1 as mode inflight (lane 2: L2 314, memory 772; this lane's own: 285, 804)
    const bool ok = std::fabs(lat[1] - 314) <= 0.3 * 314 && std::fabs(lat[2] - 772) <= 0.35 * 772;
    lds::check("tr_inflight2_calib", ok, "plain b128 N=1: L2 %.0f, memory %.0f cycles (lane 2: 314, 772)", lat[1], lat[2]);
    CHECK(hipFree(buf));
    CHECK(hipFree(out));
}

// ---------------------------------------------------------------------------------------------------------------
// mask (LD-6b): EXEC partial around one transposing load.
__global__ __launch_bounds__(32) void k_trmask(const char * base, const uint32_t * off, uint32_t mask, uint32_t * out) {
    const int l = threadIdx.x;
    u4 a = {0xDEAD0000u | (uint32_t) l, 0xDEAD0100u | (uint32_t) l, 0xDEAD0200u | (uint32_t) l, 0xDEAD0300u | (uint32_t) l};
    u2 b = {0xBEEF0000u | (uint32_t) l, 0xBEEF0100u | (uint32_t) l};
    const char * pa = base + off[l];
    const char * pb = base + off[32 + l];
    asm volatile("s_mov_b32 s96, exec_lo\n\ts_mov_b32 exec_lo, %2\n\tglobal_load_tr_b128 %0, %3, off\n\t"
                 "global_load_tr_b64 %1, %4, off\n\ts_wait_loadcnt 0\n\ts_mov_b32 exec_lo, s96"
                 : "+v"(a), "+v"(b) : "s"(mask), "v"(pa), "v"(pb) : "s96", "memory");
    for (int i = 0; i < 4; ++i) {
        out[l * 4 + i] = a[i];
    }
    out[128 + l * 2] = b[0];
    out[128 + l * 2 + 1] = b[1];
}

static void mode_mask(bool brk) {
    const size_t MEM = 65536;
    char * buf;
    uint32_t *doff, *dout;
    CHECK(hipMalloc(&buf, MEM + 64));
    CHECK(hipMalloc(&doff, 64 * 4));
    CHECK(hipMalloc(&dout, 192 * 4));
    std::mt19937 rng(23);
    std::vector<uint32_t> slots(2048);
    std::iota(slots.begin(), slots.end(), 0u);
    std::shuffle(slots.begin(), slots.end(), rng);
    std::vector<uint32_t> off(64);
    std::vector<uint8_t> mem(MEM + 64);
    for (auto & b : mem) {
        b = (uint8_t) (0x80 | (rng() & 0x7f));
    }
    for (int l = 0; l < 32; ++l) {
        off[l] = slots[l] * 16;
        off[32 + l] = 32768 + slots[32 + l] * 8;
        for (int e = 0; e < 8; ++e) {
            const uint16_t v = (uint16_t) (l * 8 + e + 1);  // never zero
            mem[off[l] + 2 * e] = (uint8_t) (v & 0x7f);
            mem[off[l] + 2 * e + 1] = (uint8_t) (v >> 7);
            mem[off[32 + l] + e] = (uint8_t) (l * 8 + e + 1);
        }
    }
    CHECK(hipMemcpy(buf, mem.data(), mem.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(doff, off.data(), 64 * 4, hipMemcpyHostToDevice));
    auto ml = [&](int l) { return brk ? (l == 3 ? 5 : l == 5 ? 3 : l) : l; };  // the break swaps lanes 3 and 5
    const uint32_t masks[] = {0xFFFFFFFFu, 0x0000FFFFu, 0xFFFF0000u, 0x55555555u, 0x11111111u, 0x00000001u, 0u};
    bool full_ok = false;
    for (uint32_t m : masks) {
        CHECK(hipMemset(dout, 0, 192 * 4));
        hipLaunchKernelGGL(k_trmask, dim3(1), dim3(32), 0, 0, buf, doff, m, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> r(192);
        CHECK(hipMemcpy(r.data(), dout, 192 * 4, hipMemcpyDeviceToHost));
        // per element of each form: on or off lane x {model, sentinel, zero, other}; for on lanes also whether the
        // source lane was on
        long c128[2][4] = {}, c64[2][4] = {}, src_off_model = 0, src_off_n = 0;
        for (int l = 0; l < 32; ++l) {
            const int on = (m >> l) & 1;
            for (int i = 0; i < 8; ++i) {
                const uint16_t got = (uint16_t) (r[l * 4 + i / 2] >> (16 * (i % 2)));
                const int src = ml(tr128_src_lane(l, i));
                const uint16_t want = h16(mem, off[src] + 2 * (l % 8));
                const uint16_t sent = (uint16_t) ((0xDEAD0000u | ((uint32_t) (i / 2) << 8) | (uint32_t) l) >> (16 * (i % 2)));
                const int k = got == want ? 0 : got == sent ? 1 : got == 0 ? 2 : 3;
                ++c128[on][k];
                if (on && !((m >> tr128_src_lane(l, i)) & 1)) {
                    ++src_off_n;
                    src_off_model += k == 0;
                }
            }
            for (int i = 0; i < 8; ++i) {
                const uint8_t got = (uint8_t) (r[128 + l * 2 + i / 4] >> (8 * (i % 4)));
                const int src = ml(tr64_src_lane(l, i));
                const uint8_t want = mem[off[32 + src] + (l % 8)];
                const uint8_t sent = (uint8_t) ((0xBEEF0000u | ((uint32_t) (i / 4) << 8) | (uint32_t) l) >> (8 * (i % 4)));
                const int k = got == want ? 0 : got == sent ? 1 : got == 0 ? 2 : 3;
                ++c64[on][k];
            }
        }
        std::printf("RESULT tr mask EXEC %08x | tr_b128 on-lane elements: model %ld sentinel %ld zero %ld other %ld; off-lane: "
                    "model %ld sentinel %ld zero %ld other %ld; on-lane elements whose source lane is off: %ld, of them model "
                    "%ld | tr_b64 on: model %ld sentinel %ld zero %ld other %ld; off: model %ld sentinel %ld zero %ld other %ld\n",
                    m, c128[1][0], c128[1][1], c128[1][2], c128[1][3], c128[0][0], c128[0][1], c128[0][2], c128[0][3],
                    src_off_n, src_off_model, c64[1][0], c64[1][1], c64[1][2], c64[1][3], c64[0][0], c64[0][1], c64[0][2],
                    c64[0][3]);
        if (m == 0xFFFFFFFFu) {
            full_ok = c128[1][0] == 256 && c64[1][0] == 256;
        }
    }
    lds::check("tr_mask_full", full_ok, "full EXEC: every element of both transposing loads equals the layout model%s",
               brk ? " (break: lanes 3 and 5 swapped in the model)" : "");
    CHECK(hipFree(buf));
    CHECK(hipFree(doff));
    CHECK(hipFree(dout));
}

static void mode_inflight(bool brk) {
    const size_t BIG = 768ull << 20;
    char * buf;
    uint64_t * out;
    CHECK(hipMalloc(&buf, BIG));
    CHECK(hipMalloc(&out, 64));
    mem::fill(buf, BIG, mem::RANDOM, 3);
    std::vector<uint8_t> host_tail(4 << 20);
    const size_t cached_at = BIG - host_tail.size();  // L0 and L2 regions live in the last 4 MiB
    CHECK(hipMemcpy(host_tail.data(), buf + cached_at, host_tail.size(), hipMemcpyDeviceToHost));
    size_t fresh = 0;  // next unused byte for memory runs (the start of the buffer: out of the 64 MB cache by now)
    const char * kname[4] = {"b128", "tr_b128", "b64", "tr_b64"};
    int badx = 0, runs = 0;
    double lat[4][3] = {};
    for (int level = 0; level < 3; ++level) {
        for (int kind = 0; kind < 4; ++kind) {
            for (int n : {1, 2, 4, 8, 16, 24}) {
                const uint32_t tile = kind < 2 ? 512 : 256, batch = n * tile;
                std::vector<double> cyc;
                for (int rep = 0; rep < 6; ++rep) {
                    uint32_t batches, wrap, warm;
                    const char * base;
                    if (level == 0) {
                        wrap = std::max(1u, 16384u / batch);
                        batches = 512, warm = wrap, base = buf + cached_at;
                    } else if (level == 1) {
                        wrap = (1u << 20) / batch;
                        batches = 512, warm = wrap, base = buf + cached_at;
                    } else {
                        batches = wrap = 64, warm = 0, base = buf + fresh;
                        fresh += (size_t) batches * batch;
                    }
                    InflightK k = kind == 0 ? pick<0>(n) : kind == 1 ? pick<1>(n) : kind == 2 ? pick<2>(n) : pick<3>(n);
                    hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, base, batches, wrap, warm, out);
                    CHECK(hipGetLastError());
                    CHECK(hipDeviceSynchronize());
                    uint64_t o[2];
                    CHECK(hipMemcpy(o, out, 16, hipMemcpyDeviceToHost));
                    if (rep > 0) {
                        cyc.push_back((double) o[0] / batches);
                    }
                    if (level < 2) {  // check the bytes of the last batch
                        const size_t at = (size_t) ((batches - 1) % wrap) * batch;
                        uint32_t x = 0;
                        for (size_t i = 0; i < batch; ++i) {
                            x ^= host_tail[at + i];
                        }
                        if (brk) {
                            x ^= 1;
                        }
                        badx += x != o[1];
                        ++runs;
                    }
                }
                const Stats s = stats_of(cyc);
                if (n == 1) {
                    lat[kind][level] = s.median;
                }
                std::printf("RESULT tr inflight level %s kind %s n %d cycles_per_batch %.1f min %.1f max %.1f per_load %.2f "
                            "bytes_per_cycle %.1f\n",
                            level == 0 ? "L0" : level == 1 ? "L2" : "memory", kname[kind], n, s.median, s.min, s.max,
                            s.median / n, n * tile / s.median);
            }
        }
    }
    lds::check("tr_inflight_data", badx == 0, "%d of %d cached runs returned bytes whose xor differs from the data%s", badx,
               runs, brk ? " (break: expected xor off by one)" : "");
    // known answer: lane 2 measured one plain b128 batch at 135 (L0), 314 (L2), 772 (memory) cycles
    const double ref[3] = {135, 314, 772}, tol[3] = {0.1, 0.3, 0.35};
    bool ok = true;
    for (int lv = 0; lv < 3; ++lv) {
        ok = ok && std::fabs(lat[0][lv] - ref[lv]) <= tol[lv] * ref[lv];
    }
    lds::check("tr_inflight_calib", ok, "plain b128 N=1: L0 %.0f, L2 %.0f, memory %.0f cycles (lane 2: 135, 314, 772)", lat[0][0],
               lat[0][1], lat[0][2]);
    CHECK(hipFree(buf));
    CHECK(hipFree(out));
}

// ---------------------------------------------------------------------------------------------------------------
// bw: whole card. Each wave reads 4 tiles per step from a window, starting at its own tile, stepping by the
// number of waves, `steps` times.

template <int KIND>
__global__ __launch_bounds__(256) void k_bw(const char * base, uint32_t ntiles, uint32_t steps, uint32_t * sink) {
    using T = typename Ld<KIND>::T;
    constexpr uint32_t TILE = Ld<KIND>::TILE;
    const uint32_t wave = blockIdx.x * 8 + threadIdx.x / 32, nw = gridDim.x * 8;
    const uint32_t lo = lane_offset(KIND, threadIdx.x % 32);
    uint32_t t = (wave * 4) % ntiles, x = 0;
    T r[4];
#pragma unroll 1
    for (uint32_t s = 0; s < steps; ++s) {
        // 4 consecutive tiles: one contiguous 2 KiB (1 KiB) piece per wave and step
        issue<KIND, 0, 4>(r, base + (size_t) t * TILE, lo);
        wait_all<KIND, 4>(r);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x ^= r[k].x;
        }
        t = (t + nw * 4) % ntiles;  // ntiles is a multiple of 4, so a 4-tile step never runs past the window
    }
    if (x == 0x12345679u) {
        sink[0] = x;
    }
}

// LD-9: SHARE 0 the bw walk, 1 the unique walk.
template <int KIND, int SHARE>
__global__ __launch_bounds__(256) void k_bw2(const char * base, uint32_t ngroups, uint32_t steps, uint32_t * sink) {
    using T = typename Ld<KIND>::T;
    constexpr uint32_t TILE = Ld<KIND>::TILE;
    const uint32_t wave = blockIdx.x * 8 + threadIdx.x / 32, nw = gridDim.x * 8;
    const uint32_t lo = lane_offset(KIND, threadIdx.x % 32);
    uint32_t x = 0;
    T r[4];
#pragma unroll 1
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t g = SHARE ? (uint32_t) (((uint64_t) wave * steps + s) % ngroups) : (uint32_t) (((uint64_t) wave + (uint64_t) s * nw) % ngroups);
        issue<KIND, 0, 4>(r, base + (size_t) g * 4 * TILE, lo);
        wait_all<KIND, 4>(r);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x ^= r[k].x;
        }
    }
    if (x == 0x12345679u) {
        sink[0] = x;
    }
}

static int mode_bw2(int argc, char ** argv, bool brk) {
    (void) brk;
    if (argc < 5) {
        std::fprintf(stderr, "usage: lds_tr bw2 b128|tr_b128 shared|unique WINDOW_MIB\n");
        return 2;
    }
    const std::string kind = argv[2], share = argv[3];
    const size_t win = (size_t) std::atoi(argv[4]) << 20;
    const bool tr = kind == "tr_b128", uniq = share == "unique";
    if ((kind != "b128" && !tr) || (share != "shared" && !uniq) || win < (1u << 20) || win > (512ull << 20)) {
        std::fprintf(stderr, "bw2: bad arguments\n");
        return 2;
    }
    const double t0c = mem::hotspot_c();
    if (t0c >= 55.0) {
        std::printf("NOT STARTED: hotspot %.0f C at the start (limit 55 C)\n", t0c);
        return 7;
    }
    char * buf;
    uint32_t * sink;
    CHECK(hipMalloc(&buf, win + 4096));
    CHECK(hipMalloc(&sink, 64));
    mem::fill(buf, win + 4096, mem::RANDOM, 9);
    const uint32_t blocks = 1024, TILE = 512;
    const uint32_t ngroups = (uint32_t) (win / (4 * TILE));
    // about 0.1 s at the expected rate (section 5: 5.3 TB/s for the bw walk at 32 MiB; unique and DRAM slower)
    const double est = win >= (256ull << 20) ? 0.63e12 : uniq ? 2.5e12 : 5.3e12;
    const uint32_t steps = (uint32_t) std::max(1.0, 0.1 * est / ((double) blocks * 8 * 4 * TILE));
    const double bytes = (double) steps * blocks * 8 * 4 * TILE;
    auto launch = [&](uint32_t st) {
        if (tr && uniq) hipLaunchKernelGGL((k_bw2<1, 1>), dim3(blocks), dim3(256), 0, 0, buf, ngroups, st, sink);
        if (tr && !uniq) hipLaunchKernelGGL((k_bw2<1, 0>), dim3(blocks), dim3(256), 0, 0, buf, ngroups, st, sink);
        if (!tr && uniq) hipLaunchKernelGGL((k_bw2<0, 1>), dim3(blocks), dim3(256), 0, 0, buf, ngroups, st, sink);
        if (!tr && !uniq) hipLaunchKernelGGL((k_bw2<0, 0>), dim3(blocks), dim3(256), 0, 0, buf, ngroups, st, sink);
    };
    launch(1);  // warm-up: code load, one step
    CHECK(hipDeviceSynchronize());
    std::atomic<bool> run{true};
    std::atomic<double> hot{t0c};
    std::thread th([&] {
        while (run) {
            const double t = mem::hotspot_c();
            hot = std::max(hot.load(), t);
            if (t >= 95.0) {
                std::printf("THERMAL %.0f C: stopping\n", t);
                std::fflush(stdout);
                std::_Exit(4);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    const std::vector<double> ms = time_ms([&] { launch(steps); }, 1, 0);
    run = false;
    th.join();
    CHECK(hipGetLastError());
    std::printf("TRBW tag=ld9 kind=%s share=%s window_MiB=%zu steps=%u bytes=%.0f ms=%.3f GBs=%.1f hotspot_C_start=%.0f "
                "hotspot_C_max=%.0f\n",
                kind.c_str(), share.c_str(), win >> 20, steps, bytes, ms[0], bytes / (ms[0] * 1e-3) / 1e9, t0c, hot.load());
    mem::hottest() = std::max(mem::hottest(), hot.load());
    lds::check("tr_bw2_burst", ms[0] > 0, "burst recorded");
    CHECK(hipFree(buf));
    CHECK(hipFree(sink));
    return 0;
}

static void mode_bw() {
    const size_t BIG = 512ull << 20;
    char * buf;
    uint32_t * sink;
    CHECK(hipMalloc(&buf, BIG + 4096));
    CHECK(hipMalloc(&sink, 64));
    mem::fill(buf, BIG + 4096, mem::RANDOM, 9);
    const uint32_t blocks = 1024;  // 8192 waves, 4 per SIMD
    const char * kname[4] = {"b128", "tr_b128", "b64", "tr_b64"};
    const struct {
        const char * name;
        size_t window;
        double bytes;  // per launch
    } win[4] = {{"L0_16KiB", 16384, 2e9}, {"L2_2MiB", 2u << 20, 2e9}, {"MALL_32MiB", 32u << 20, 1e9}, {"DRAM_512MiB", BIG, (double) BIG}};
    mem::clocks("bw-before");
    for (auto & w : win) {
        for (int kind = 0; kind < 4; ++kind) {
            if (!lds::thermal_gate("bw")) {
                lds::check("tr_bw_thermal", false, "thermal gate refused");
                return;
            }
            const uint32_t tile = kind < 2 ? 512 : 256;
            const uint32_t ntiles = (uint32_t) (w.window / tile);
            const uint32_t steps = (uint32_t) std::max(1.0, w.bytes / ((double) blocks * 8 * 4 * tile));
            const double bytes = (double) steps * blocks * 8 * 4 * tile;
            auto launch = [&]() {
                if (kind == 0) hipLaunchKernelGGL(k_bw<0>, dim3(blocks), dim3(256), 0, 0, buf, ntiles, steps, sink);
                if (kind == 1) hipLaunchKernelGGL(k_bw<1>, dim3(blocks), dim3(256), 0, 0, buf, ntiles, steps, sink);
                if (kind == 2) hipLaunchKernelGGL(k_bw<2>, dim3(blocks), dim3(256), 0, 0, buf, ntiles, steps, sink);
                if (kind == 3) hipLaunchKernelGGL(k_bw<3>, dim3(blocks), dim3(256), 0, 0, buf, ntiles, steps, sink);
            };
            const std::vector<double> ms = time_ms(launch, 7, 2);
            CHECK(hipGetLastError());
            mem::report_bw("tr_bw", std::string(w.name) + " " + kname[kind], ms, bytes);
            lds::after_full("bw");
        }
    }
    mem::clocks("bw-after");
    CHECK(hipFree(buf));
    CHECK(hipFree(sink));
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "layout";
    const bool brk = core::flag(argc, argv, "--break");
    mem::clocks("start");
    if (mode == "calib") {
        lds::calibrate(brk);
    } else if (mode == "layout") {
        mode_layout(brk);
    } else if (mode == "wmma") {
        mode_wmma(brk);
    } else if (mode == "inflight") {
        mode_inflight(brk);
    } else if (mode == "inflight2") {
        mode_inflight2(brk);
    } else if (mode == "mask") {
        mode_mask(brk);
    } else if (mode == "bw") {
        mode_bw();
    } else if (mode == "bw2") {
        if (mode_bw2(argc, argv, brk) != 0) {
            return 2;
        }
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    mem::clocks("end");
    return lds::finish();
}
