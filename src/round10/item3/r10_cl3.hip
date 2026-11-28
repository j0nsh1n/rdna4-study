// Round 10, item 3: why does Vulkan read untouched memory more slowly than HIP? The HIP half of the two-way bisect.
// Round 9's r9_cl2 cold and fused cells with one factor changed at a time, toward Vulkan's side (vkr10 is the other half;
// the keys and the output lines are the same on both sides so scripts/analysis can pair them).
//
// Usage: r10_cl3 bis <spec>...     one cell per spec
//        r10_cl3 reach <spec>      N 4 KiB pages walked twice (the translation reach test)
//        r10_cl3 info              memory kinds this build can allocate (no kernel launch except none)
// bis keys (defaults are round 9's HIP cold cell, r9_cl2 ci == 1):
//   name=  walk=random|page  order=cold|fused|noevict|fresh  reps=20  steps=1024  (steps is rounded down to a multiple of 16)
//   form=p64|m64|b32|u32v   p64: 64-bit absolute addresses, one global_load_b64 and one wait per step (r9_cl2);
//        m64: p64 plus a mask and a shift-and-sum on the chain (H1: Vulkan's arithmetic, same 8-byte load);
//        b32: 4-byte load, 32-bit byte offsets from an SGPR base (H2: Vulkan's load width, no arithmetic);
//        u32v: Vulkan's loop (4-byte load, `p = v & mask`, `acc += v >> 15`, index arithmetic, `lane & zero`): H1 + H2 together
//   alloc=one|sep|sub       one: all regions in ONE hipMalloc (r9_cl2: 8 MiB); sep: one hipMalloc per region (H4, Vulkan's round 9
//        layout); sub: one 512 MiB hipMalloc, regions at 2 MiB-aligned offsets inside it
//   mem=dev|devunc|host|hostnc   dev: hipMalloc (VRAM); devunc: hipExtMallocWithFlags uncached; host: pinned system memory mapped
//        into the device (hipHostMalloc mapped); hostnc: the same, non-coherent flag
//   evict=512|256|0         MiB read by the evict kernel before every repetition (r9_cl2: 512)
//   evform=asm|plain        asm: mem_latency.hip's k_hread (inline asm, each load waited); plain: ordinary uint4 loads
//   scope=rt|dev|sys|nt|se  load modifier on the chase loads (rt = none, as r9_cl2): parity with Vulkan's `coherent`
//   stride=<bytes>          (form p64|m64) one cyclic chain of `steps` nodes, one node per stride (page order)
//   loop=<slots>            the walk wraps inside this many 128-byte lines (power of two); the footprint control
//   warm=<ms>               optional pre-cell core-clock warm-up spin in ms (0..2000, default 0; parity with vkr10's warm=150)
// Build flavours: R10_W64 (hipcc -mwavefrontsize64 -DR10_W64) launches 64 lanes per block and prints w=64 on every line (H3).
// Output: CELL/ROW/RESULT lines with the same keys as vkr10 (api=hip), memory clock read before and after every repetition.
#include <time.h>
#include <string.h>
#define main r9_mem_latency_main
#include "../../study/mem_latency.hip"
#undef main

#ifdef R10_W64
#define R10_BLK 64
#define R10_WAVE 64
#else
#define R10_BLK 32
#define R10_WAVE 32
#endif

constexpr uint32_t NSLOT = 32768, LCG_A = 1664525u;
constexpr size_t REG_BYTES = (size_t) NSLOT * 128;
constexpr size_t MIB = 1u << 20;

enum Form { P64 = 0, M64 = 1, B32 = 2, U32V = 3 };
static const char * const FORM_N[] = {"p64", "m64", "b32", "u32v"};
enum Order { O_COLD = 0, O_FUSED = 1, O_NOEVICT = 2, O_FRESH = 3 };
static const char * const ORD_N[] = {"cold", "fused", "noevict", "fresh"};
static const char * const ALLOC_N[] = {"one", "sep", "sub"};
static const char * const MEM_N[] = {"dev", "devunc", "host", "hostnc"};
static const char * const SC_N[] = {"rt", "dev", "sys", "nt", "se"};
static const char * const EVF_N[] = {"asm", "plain"};

// Load modifiers (guide table 13, as mem_common.h's MEM_HINTS); index = the scope= key.
#define R10_SCOPES(X) X(0, "") X(1, " scope:SCOPE_DEV") X(2, " scope:SCOPE_SYS") X(3, " th:TH_LOAD_NT") X(4, " scope:SCOPE_SE")

__host__ __device__ static inline uint32_t stamp_of(uint32_t salt) { return (salt >> 15) | 1u; }

// ---------------------------------------------------------------------------------------------------------------
// the chase, one template per loop form. State: P64/M64 the absolute address, B32 the byte offset from base, U32V the slot.

template <int FORM, int SC>
__device__ __forceinline__ uint64_t chase_f(uint64_t st, uint32_t steps, const char * base, uint64_t m, uint32_t mask, uint32_t zero,
                                            uint32_t & acc) {
    if constexpr (FORM == P64 && SC == 0) {
        return chase(st, steps);  // mem_latency.hip's own loop, as r9_cl2 uses it
    } else if constexpr (FORM == P64 || FORM == M64) {
        uint64_t p = st;
#pragma unroll 1
        for (uint32_t i = 0; i < steps / 16; ++i) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#define X(k, s) \
    if constexpr (SC == k) asm volatile("global_load_b64 %0, %0, off" s "\n\ts_wait_loadcnt 0" : "+v"(p)::"memory");
                R10_SCOPES(X)
#undef X
                if constexpr (FORM == M64) {
                    p &= m;
                    acc += (uint32_t) p >> 15;
                }
            }
        }
        return p;
    } else if constexpr (FORM == B32) {
        uint32_t off = (uint32_t) st;
#pragma unroll 1
        for (uint32_t i = 0; i < steps / 16; ++i) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#define X(k, s) \
    if constexpr (SC == k) asm volatile("global_load_b32 %0, %0, %1" s "\n\ts_wait_loadcnt 0" : "+v"(off) : "s"(base) : "memory");
                R10_SCOPES(X)
#undef X
            }
        }
        return off;
    } else {
        uint32_t p = (uint32_t) st;
#pragma unroll 1
        for (uint32_t i = 0; i < steps / 16; ++i) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t off = (p << 7) + ((threadIdx.x & zero) << 2);
                uint32_t v;
#define X(k, s) \
    if constexpr (SC == k) asm volatile("global_load_b32 %0, %1, %2" s "\n\ts_wait_loadcnt 0" : "=v"(v) : "v"(off), "s"(base) : "memory");
                R10_SCOPES(X)
#undef X
                p = v & mask;
                acc += v >> 15;
            }
        }
        return p;
    }
}

// out: [0] cycles, [1] REALTIME ticks, [2] final state, [3] stamp sum
template <int FORM, int SC>
__global__ __launch_bounds__(R10_BLK) void k_chase_f(uint64_t start, const char * base, uint64_t m, uint32_t mask, uint32_t zero,
                                                     uint32_t steps, uint64_t * out) {
    uint32_t acc = 0;
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
    const uint64_t p = chase_f<FORM, SC>(start, steps, base, m, mask, zero, acc);
    const uint64_t c1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = p;
        out[3] = acc;
    }
}

// Optional core-clock warm-up spin (parity with vkr10's 150 ms warm-up): spins for `ticks` of the 100 MHz realtime clock.
__global__ __launch_bounds__(256) void k_warm_spin(uint64_t ticks, unsigned * sink) {
    const uint64_t t0 = mem::realtime();
    uint32_t acc = threadIdx.x + blockIdx.x * 256u;
    while (mem::realtime() - t0 < ticks) {
        acc = acc * 1664525u + 1013904223u;
    }
    if (threadIdx.x == 0 && acc == 0xffffffffu) sink[blockIdx.x] = acc;
}

// The reach test: the chain of `n` nodes walked twice, each pass timed. out: [0] cycles pass 1, [1] pass 2, [2] final, [3] REALTIME.
__global__ __launch_bounds__(R10_BLK) void k_reach(uint64_t start, uint32_t n, uint64_t * out) {
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
    uint64_t p = chase(start, n);
    const uint64_t c1 = wave_cycles();
    p = chase(p, n);
    const uint64_t c2 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = c2 - c1;
        out[2] = p;
        out[3] = r1 - r0;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// producers: line s of a region (128 bytes), form-specific content of dword 0 (and 1)

template <int FORM>
__device__ __forceinline__ void write_line_f(char * base, uint32_t slot, uint32_t lane, uint32_t mul, uint32_t c, uint32_t salt, uint32_t mask) {
    const uint32_t nxt = (mul * slot + c) & mask;
    uint32_t * line = (uint32_t *) (base + (size_t) slot * 128);
    if constexpr (FORM == P64 || FORM == M64) {
        const uint64_t a = (uint64_t) base + (uint64_t) nxt * 128;
        line[lane] = lane == 0 ? (uint32_t) a : lane == 1 ? (uint32_t) (a >> 32) : (slot ^ salt) + lane;
    } else if constexpr (FORM == B32) {
        line[lane] = lane == 0 ? nxt * 128u : (slot ^ salt) + lane;
    } else {
        line[lane] = lane == 0 ? (nxt | (stamp_of(salt) << 15)) : (slot ^ salt) + lane;
    }
}

template <int FORM>
__global__ __launch_bounds__(256) void k_prod_f(char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask) {
    const uint32_t t = blockIdx.x * 256 + threadIdx.x;
    write_line_f<FORM>(base, t >> 5, t & 31, a, c, salt, mask);
}

// node i at base + i * stride holds the absolute address of node (i + 1) % count
__global__ __launch_bounds__(256) void k_chain(char * base, uint32_t stride, uint32_t count) {
    const uint32_t i = blockIdx.x * 256 + threadIdx.x;
    if (i >= count) return;
    const uint32_t nx = i + 1 == count ? 0 : i + 1;
    *(uint64_t *) (base + (size_t) i * stride) = (uint64_t) base + (uint64_t) nx * stride;
}

// One block of 256: writes everything, then wave 0 chases (same outputs as k_chase_f).
template <int FORM, int SC>
__global__ __launch_bounds__(256) void k_fused_f(char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask, uint64_t start,
                                                 uint64_t m, uint32_t zero, uint32_t steps, uint64_t * out) {
    for (uint32_t slot = threadIdx.x >> 5; slot <= mask; slot += 8) write_line_f<FORM>(base, slot, threadIdx.x & 31, a, c, salt, mask);
    __threadfence_block();
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t acc = 0;
        const uint64_t r0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
        const uint64_t p = chase_f<FORM, SC>(start, steps, base, m, mask, zero, acc);
        const uint64_t c1 = wave_cycles();
        const uint64_t r1 = mem::realtime();
        if (threadIdx.x == 0) {
            out[0] = c1 - c0;
            out[1] = r1 - r0;
            out[2] = p;
            out[3] = acc;
        }
    }
}

// the plain-loads eviction: the same grid-stride read as k_hread<0> without the inline asm and the wait after every load
__global__ __launch_bounds__(256) void k_evict_plain(const uint4 * __restrict__ p, uint32_t n16, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------------------------
// host

static int uclk_now() {
    const char * d = std::getenv("DRM_DEVICE");
    if (!d) return -1;
    std::string p = std::string(d) + "/gpu_metrics";
    FILE * f = std::fopen(p.c_str(), "rb");
    if (!f) return -1;
    unsigned char b[64];
    const size_t n = std::fread(b, 1, sizeof b, f);
    std::fclose(f);
    return n >= 60 ? (int) (b[58] | (b[59] << 8)) : -1;
}

static std::string kvs(const char * spec, const char * key, const char * def) {
    const std::string k = std::string(key) + "=";
    const char * s = spec;
    while (s && *s) {
        if (!std::strncmp(s, k.c_str(), k.size())) {
            const char * v = s + k.size();
            return std::string(v, std::strcspn(v, ","));
        }
        const char * c = std::strchr(s, ',');
        s = c ? c + 1 : nullptr;
    }
    return def;
}
static double kvd(const char * spec, const char * key, double def) {
    const std::string v = kvs(spec, key, "");
    return v.empty() ? def : std::atof(v.c_str());
}
static int idx_of(const std::string & s, const char * const * names, int n) {
    for (int i = 0; i < n; ++i)
        if (s == names[i]) return i;
    return -1;
}

static uint32_t lcg_final(uint32_t a, uint32_t start, uint32_t steps, uint32_t c, uint32_t mask) {
    uint32_t p = start;
    for (uint32_t i = 0; i < steps; ++i) p = (a * p + c) & mask;
    return p;
}

// Waits (read only) until the hotspot is below 58 C; false after 120 s.
static bool wait_cool(const std::string & name) {
    for (int w = 0; mem::hotspot_c() >= 58.0; ++w) {
        if (w > 240) {
            std::printf("NOT RUN %s: hotspot %.0f C (a cell starts below 58 C)\n", name.c_str(), mem::hotspot_c());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return true;
}

struct Mem {
    char * p = nullptr;      // device pointer
    void * host = nullptr;   // host allocation (host kinds), freed with hipHostFree
};

static bool mem_alloc(Mem & m, size_t bytes, int kind) {
    m = Mem();
    hipError_t e = hipSuccess;
    if (kind == 0) e = hipMalloc((void **) &m.p, bytes);
    else if (kind == 1) e = hipExtMallocWithFlags((void **) &m.p, bytes, hipDeviceMallocUncached);
    else {
        e = hipHostMalloc(&m.host, bytes, hipHostMallocMapped | (kind == 3 ? hipHostMallocNonCoherent : 0));
        if (e == hipSuccess) e = hipHostGetDevicePointer((void **) &m.p, m.host, 0);
    }
    if (e != hipSuccess) {
        std::printf("INFO allocation of %zu MiB with mem=%s failed: %s\n", bytes >> 20, MEM_N[kind], hipGetErrorString(e));
        (void) hipGetLastError();
        m = Mem();
        return false;
    }
    return true;
}
static void mem_free(Mem & m) {
    if (m.host) (void) hipHostFree(m.host);
    else if (m.p) (void) hipFree(m.p);
    m = Mem();
}

// n regions of `sz` bytes in the chosen layout (region 0 is round 9's ra)
struct Regions {
    std::vector<Mem> own;
    std::vector<char *> r;
    bool make(int alloc, int kind, int n, size_t sz) {
        r.assign(n, nullptr);
        if (alloc == 1) {  // sep
            for (int i = 0; i < n; ++i) {
                own.emplace_back();
                if (!mem_alloc(own.back(), sz, kind)) return false;
                r[i] = own.back().p;
            }
            return true;
        }
        const size_t step = (sz + 2 * MIB - 1) / (2 * MIB) * (2 * MIB);
        const size_t first = alloc == 2 ? 128 * MIB : 0;
        size_t total = first + step * n + (alloc == 2 ? 128 * MIB : 0);
        if (alloc == 2 && total < 512 * MIB) total = 512 * MIB;
        if (alloc == 0) total = sz * n;  // one hipMalloc of exactly the regions (r9_cl2: 8 MiB for two)
        own.emplace_back();
        if (!mem_alloc(own.back(), total, kind)) return false;
        for (int i = 0; i < n; ++i) r[i] = own.back().p + (alloc == 0 ? sz * i : first + step * i);
        return true;
    }
    void release() {
        for (auto & m : own) mem_free(m);
        own.clear();
    }
};

template <int FORM>
static void launch_prod(char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask) {
    hipLaunchKernelGGL(k_prod_f<FORM>, dim3(NSLOT / 8), dim3(256), 0, 0, base, a, c, salt, mask);
}
static void produce(int form, char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask) {
    if (form == P64 || form == M64) launch_prod<P64>(base, a, c, salt, mask);
    else if (form == B32) launch_prod<B32>(base, a, c, salt, mask);
    else launch_prod<U32V>(base, a, c, salt, mask);
}

template <int FORM, int SC>
static void launch_chase(uint64_t start, const char * base, uint32_t mask, uint32_t steps, uint64_t * out) {
    hipLaunchKernelGGL((k_chase_f<FORM, SC>), dim3(1), dim3(R10_BLK), 0, 0, start, base, ~0ull, mask, 0u, steps, out);
}
template <int FORM, int SC>
static void launch_fused(char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask, uint64_t start, uint32_t steps, uint64_t * out) {
    hipLaunchKernelGGL((k_fused_f<FORM, SC>), dim3(1), dim3(256), 0, 0, base, a, c, salt, mask, start, ~0ull, 0u, steps, out);
}
template <int FORM>
static void chase_sc(int sc, uint64_t start, const char * base, uint32_t mask, uint32_t steps, uint64_t * out) {
    switch (sc) {
        case 0: launch_chase<FORM, 0>(start, base, mask, steps, out); break;
        case 1: launch_chase<FORM, 1>(start, base, mask, steps, out); break;
        case 2: launch_chase<FORM, 2>(start, base, mask, steps, out); break;
        case 3: launch_chase<FORM, 3>(start, base, mask, steps, out); break;
        default: launch_chase<FORM, 4>(start, base, mask, steps, out); break;
    }
}
static void do_chase(int form, int sc, uint64_t start, const char * base, uint32_t mask, uint32_t steps, uint64_t * out) {
    if (form == P64) chase_sc<P64>(sc, start, base, mask, steps, out);
    else if (form == M64) chase_sc<M64>(sc, start, base, mask, steps, out);
    else if (form == B32) chase_sc<B32>(sc, start, base, mask, steps, out);
    else chase_sc<U32V>(sc, start, base, mask, steps, out);
}
template <int FORM>
static void fused_sc(int sc, char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask, uint64_t start, uint32_t steps, uint64_t * out) {
    switch (sc) {
        case 0: launch_fused<FORM, 0>(base, a, c, salt, mask, start, steps, out); break;
        case 1: launch_fused<FORM, 1>(base, a, c, salt, mask, start, steps, out); break;
        case 2: launch_fused<FORM, 2>(base, a, c, salt, mask, start, steps, out); break;
        case 3: launch_fused<FORM, 3>(base, a, c, salt, mask, start, steps, out); break;
        default: launch_fused<FORM, 4>(base, a, c, salt, mask, start, steps, out); break;
    }
}
static void do_fused(int form, int sc, char * base, uint32_t a, uint32_t c, uint32_t salt, uint32_t mask, uint64_t start, uint32_t steps, uint64_t * out) {
    if (form == P64) fused_sc<P64>(sc, base, a, c, salt, mask, start, steps, out);
    else if (form == M64) fused_sc<M64>(sc, base, a, c, salt, mask, start, steps, out);
    else if (form == B32) fused_sc<B32>(sc, base, a, c, salt, mask, start, steps, out);
    else fused_sc<U32V>(sc, base, a, c, salt, mask, start, steps, out);
}

static void evict(const Mem & ev, size_t bytes, int evform, unsigned * sink) {
    if (evform == 0) hipLaunchKernelGGL(k_hread<0>, dim3(512), dim3(256), 0, 0, ev.p, (uint32_t) (bytes / 16), sink);
    else hipLaunchKernelGGL(k_evict_plain, dim3(512), dim3(256), 0, 0, (const uint4 *) ev.p, (uint32_t) (bytes / 16), sink);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
}

static int run_bis(const char * spec) {
    const std::string name = kvs(spec, "name", "bis"), walk = kvs(spec, "walk", "random");
    const int form = idx_of(kvs(spec, "form", "p64"), FORM_N, 4), order = idx_of(kvs(spec, "order", "cold"), ORD_N, 4),
              alloc = idx_of(kvs(spec, "alloc", "one"), ALLOC_N, 3), kind = idx_of(kvs(spec, "mem", "dev"), MEM_N, 4),
              sc = idx_of(kvs(spec, "scope", "rt"), SC_N, 5), evform = idx_of(kvs(spec, "evform", "asm"), EVF_N, 2);
    const int reps = (int) kvd(spec, "reps", 20), ev_mib = (int) kvd(spec, "evict", 512), warm_ms = (int) kvd(spec, "warm", 0);
    const uint32_t steps = (uint32_t) kvd(spec, "steps", 1024) / 16 * 16, stride = (uint32_t) kvd(spec, "stride", 0),
                   loop = (uint32_t) kvd(spec, "loop", NSLOT);
    const bool page = walk == "page";
    if (form < 0 || order < 0 || alloc < 0 || kind < 0 || sc < 0 || evform < 0 || (!page && walk != "random") || reps < 2 || reps > 400 ||
        warm_ms < 0 || warm_ms > 2000 ||
        steps < 16 || steps > NSLOT || (ev_mib != 0 && ev_mib != 256 && ev_mib != 512) || loop < 8 || loop > NSLOT || (loop & (loop - 1)) ||
        (stride && ((form != P64 && form != M64) || stride % 128 || stride > 2 * MIB || order == O_FUSED)) ||
        (order == O_NOEVICT && ev_mib)) {
        std::printf("VERDICT r10_cl3 bis_spec: FAIL bad cell %s\n", spec);
        return 1;
    }
    if (!wait_cool(name)) return 1;
    const bool fused = order == O_FUSED, fresh = order == O_FRESH;
    const int nreg = fresh ? reps : 1;
    const uint32_t mask = loop - 1, count = stride ? steps : 0;
    const size_t regsize = stride ? std::max(REG_BYTES, (size_t) stride * count) : REG_BYTES;
    Regions R;
    if (!R.make(alloc, kind, nreg + 1, regsize)) {
        std::printf("NOT RUN %s: allocation failed (alloc=%s mem=%s)\n", name.c_str(), ALLOC_N[alloc], MEM_N[kind]);
        R.release();
        return 1;
    }
    Mem ev, sinkm, outm;
    if (ev_mib) {
        CHECK(hipMalloc((void **) &ev.p, (size_t) ev_mib * MIB));
        mem::fill(ev.p, (size_t) ev_mib * MIB, mem::RANDOM);
    }
    CHECK(hipMalloc((void **) &sinkm.p, 512 * 256 * 4));
    CHECK(hipMalloc((void **) &outm.p, 64));
    uint64_t * out = (uint64_t *) outm.p;
    unsigned * sink = (unsigned *) sinkm.p;
    const uint32_t cold_a = page ? 1u : LCG_A, cold_c = page ? 1u : 1013904223u, cold_salt = 0x1234567u;
    if (!fused) {
        for (int i = 1; i <= nreg; ++i) {
            if (stride) hipLaunchKernelGGL(k_chain, dim3((count + 255) / 256), dim3(256), 0, 0, R.r[i], stride, count);
            else produce(form, R.r[i], cold_a, cold_c, cold_salt, mask);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        if (fresh && ev_mib) evict(ev, (size_t) ev_mib * MIB, evform, sink);
    }
    if (warm_ms > 0) {
        hipLaunchKernelGGL(k_warm_spin, dim3(64), dim3(256), 0, 0, (uint64_t) warm_ms * 100000ull, sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    const int uc0 = uclk_now();
    std::printf("CELL %s api=hip q=hip form=%s order=%s walk=%s alloc=%s mem=%s evict=%d evform=%s scope=%s w=%d stride=%u loop=%u reps=%d steps=%u "
                "uclk_start=%d hotspot=%.0f\n",
                name.c_str(), FORM_N[form], ORD_N[order], page ? "page" : "random", ALLOC_N[alloc], MEM_N[kind], ev_mib,
                evform ? "plain" : "asm", SC_N[sc], R10_WAVE, stride, loop, reps, steps, uc0, mem::hotspot_c());
    std::vector<double> cps, nps, clk;
    int okn = 0;
    for (int r = 0; r < reps; ++r) {
        char * rg = fused ? R.r[0] : R.r[fresh ? r + 1 : 1];
        const uint32_t salt = 0x5bd1e995u * (uint32_t) (r + 1);
        const uint32_t a = page ? 1u : LCG_A, cc = page ? 1u : 1013904223u + 2u * (uint32_t) r;
        const uint32_t start = stride ? (uint32_t) ((uint64_t) r * 37u % count) : ((uint32_t) r * 7919u + 1u) & mask;
        const uint32_t fin_slot = stride ? (start + steps) % count : lcg_final(fused ? a : cold_a, start, steps, fused ? cc : cold_c, mask);
        const uint64_t unit = stride ? stride : 128u;
        uint64_t st, exp;  // start state and expected final state of the form
        if (form == P64 || form == M64) st = (uint64_t) rg + (uint64_t) start * unit, exp = (uint64_t) rg + (uint64_t) fin_slot * unit;
        else if (form == B32) st = (uint64_t) start * 128, exp = (uint64_t) fin_slot * 128;
        else st = start, exp = fin_slot;
        const uint32_t expacc = form == M64 ? 0u : form == U32V ? steps * stamp_of(fused ? salt : cold_salt) : 0u;
        const int ua = uclk_now();
        if (ev_mib && !fresh) evict(ev, (size_t) ev_mib * MIB, evform, sink);
        if (fused) do_fused(form, sc, rg, a, cc, salt, mask, st, steps, out);
        else do_chase(form, sc, st, rg, mask, steps, out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[4];
        CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
        const int ub = uclk_now();
        int ok = o[2] == exp;
        if (form == U32V) ok = ok && (uint32_t) o[3] == expacc;
        const double mhz = o[1] ? (double) o[0] / (double) o[1] * 100.0 : 0.0;
        std::printf("ROW r10 api=hip cell=%s q=hip form=%s order=%s walk=%s alloc=%s mem=%s evict=%d evform=%s scope=%s w=%d stride=%u loop=%u rep=%d "
                    "cycles=%llu rt=%llu ok=%d steps=%u clk_mhz=%.0f uclk0=%d uclk1=%d\n",
                    name.c_str(), FORM_N[form], ORD_N[order], page ? "page" : "random", ALLOC_N[alloc], MEM_N[kind], ev_mib,
                    evform ? "plain" : "asm", SC_N[sc], R10_WAVE, stride, loop, r, (unsigned long long) o[0], (unsigned long long) o[1], ok, steps,
                    mhz, ua, ub);
        if (r > 0) {
            cps.push_back((double) o[0] / steps);
            nps.push_back((double) o[1] * 10.0 / steps);
            clk.push_back(mhz);
            okn += ok;
        }
    }
    const Stats cl = stats_of(clk);
    std::printf("RESULT r10 api=hip cell=%s q=hip form=%s order=%s walk=%s alloc=%s mem=%s evict=%d evform=%s scope=%s w=%d stride=%u loop=%u reps=%d "
                "counted=%zu cyc_per_step_med=%.1f ns_per_step_med=%.1f pointer_ok=%d/%zu clk_mhz_med=%.0f min=%.0f max=%.0f uclk_start=%d uclk_end=%d\n",
                name.c_str(), FORM_N[form], ORD_N[order], page ? "page" : "random", ALLOC_N[alloc], MEM_N[kind], ev_mib, evform ? "plain" : "asm",
                SC_N[sc], R10_WAVE, stride, loop, reps, cps.size(), stats_of(cps).median, stats_of(nps).median, okn, cps.size(), cl.median, cl.min,
                cl.max, uc0, uclk_now());
    // A printed FAIL must not still exit 0: the wrapper and the analyser treat 0 as success.
    const bool clock_ok = (cl.max - cl.min) <= 0.10 * cl.median;
    std::printf("VERDICT r10_cl3 cell_clock: %s %s: core clock of the counted repetitions %.0f-%.0f MHz, median %.0f (the spread must stay within 10 %% of the median)\n",
                clock_ok ? "PASS" : "FAIL", name.c_str(), cl.min, cl.max, cl.median);
    std::fflush(stdout);
    R.release();
    mem_free(ev);
    mem_free(sinkm);
    mem_free(outm);
    return clock_ok ? 0 : 1;
}

static int run_reach(const char * spec) {
    const std::string name = kvs(spec, "name", "reach"), list = kvs(spec, "n", "64,256,1024,4096,8192");
    const int reps = (int) kvd(spec, "reps", 6), ev_mib = (int) kvd(spec, "evict", 512), kind = idx_of(kvs(spec, "mem", "dev"), MEM_N, 4);
    if (reps < 2 || reps > 100 || (ev_mib != 0 && ev_mib != 256 && ev_mib != 512) || kind < 0) {
        std::printf("VERDICT r10_cl3 reach_spec: FAIL bad spec %s\n", spec);
        return 1;
    }
    if (!wait_cool(name)) return 1;
    Mem ev, sinkm, outm;
    if (ev_mib) {
        CHECK(hipMalloc((void **) &ev.p, (size_t) ev_mib * MIB));
        mem::fill(ev.p, (size_t) ev_mib * MIB, mem::RANDOM);
    }
    CHECK(hipMalloc((void **) &sinkm.p, 512 * 256 * 4));
    CHECK(hipMalloc((void **) &outm.p, 64));
    for (const char * p = list.c_str(); *p;) {
        const uint32_t n = (uint32_t) std::atoi(p);
        const char * cm = std::strchr(p, ',');
        p = cm ? cm + 1 : p + std::strlen(p);
        if (n < 16 || n > 8192 || n % 16) {
            std::printf("VERDICT r10_cl3 reach_spec: FAIL n=%u not a multiple of 16 in 16..8192\n", n);
            continue;
        }
        Mem reg;
        if (!mem_alloc(reg, 32 * MIB, kind)) return 1;
        hipLaunchKernelGGL(k_chain, dim3((n + 255) / 256), dim3(256), 0, 0, reg.p, 4096u, n);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::printf("CELL %s api=hip q=hip n=%u reps=%d mem=%s evict=%d w=%d uclk_start=%d hotspot=%.0f\n", name.c_str(), n, reps, MEM_N[kind], ev_mib,
                    R10_WAVE, uclk_now(), mem::hotspot_c());
        for (int r = 0; r < reps; ++r) {
            const uint32_t s0 = (uint32_t) ((uint64_t) r * 37u % n);
            const int ua = uclk_now();
            if (ev_mib) evict(ev, (size_t) ev_mib * MIB, 0, (unsigned *) sinkm.p);
            hipLaunchKernelGGL(k_reach, dim3(1), dim3(R10_BLK), 0, 0, (uint64_t) reg.p + (uint64_t) s0 * 4096u, n, (uint64_t *) outm.p);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            uint64_t o[4];
            CHECK(hipMemcpy(o, outm.p, sizeof o, hipMemcpyDeviceToHost));
            const uint64_t exp = (uint64_t) reg.p + (uint64_t) ((s0 + 2ull * n) % n) * 4096u;
            std::printf("ROW r10 api=hip cell=%s mode=reach q=hip n=%u rep=%d cycles1=%llu cycles2=%llu rt=%llu ok=%d steps=%u uclk0=%d uclk1=%d\n",
                        name.c_str(), n, r, (unsigned long long) o[0], (unsigned long long) o[1], (unsigned long long) o[3], o[2] == exp, n, ua,
                        uclk_now());
        }
        mem_free(reg);
    }
    mem_free(ev);
    mem_free(sinkm);
    mem_free(outm);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: r10_cl3 info | bis <spec>... | reach <spec>\n");
        return 2;
    }
    const hipDeviceProp_t prop = require_arch();
    std::printf("device %s (%s), r10_cl3 %s wave=%d\n", prop.name, prop.gcnArchName, argv[1], R10_WAVE);
    const std::string mode = argv[1];
    int rc = 2;
    if (mode == "info") {
        // each kind of host/device memory this build can allocate (4 MiB, freed again); no kernel is launched
        rc = 0;
        for (int k = 0; k < 4; ++k) {
            Mem m;
            const bool ok = mem_alloc(m, 4 * MIB, k);
            std::printf("INFO mem=%s allocation %s\n", MEM_N[k], ok ? "ok" : "FAILED");
            mem_free(m);
        }
    } else if (mode == "bis") {
        rc = 0;
        for (int i = 2; i < argc; ++i) rc |= run_bis(argv[i]);
    } else if (mode == "reach") {
        rc = 0;
        for (int i = 2; i < argc; ++i) rc |= run_reach(argv[i]);
    }
    mem::report_hottest();
    return rc;
}
