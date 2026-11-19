// Leftovers C, lane A, MS-6: cache hints through the buffer-load builtins' cache-policy operand.
//
// Usage: left_hint [--break]
//   The builtin __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, soffset, aux) is instantiated for aux = 0..31; the
//   script checks in the listing that each kernel's buffer_load_b128 carries th = aux & 7 and scope = aux >> 3 (READ:
//   IntrinsicsAMDGPU.td, "gfx12+: bits [0-2] = th, bits [3-4] = scope", research-10 section 4).
//   On the card, memory-system item 8's test, here for RT, NT, HT and LU through the builtin and through inline
//   assembly (global_load_b128 th:...) side by side: flush (512 MiB read), a hinted read of a 4 MiB region by the whole
//   card, then a fresh one-wave chase over that region (nodes 256 bytes apart, random order, 4,096 steps, a new launch
//   so L2 is empty): about 400-480 cycles means the hinted read left the data in the memory-side cache, about 630 or
//   more means it did not. A checksum of the hinted read equals a plain read's (the buffer descriptor works).
//   Verdicts: hint_control (the inline-assembly NT and LU chases at least 1.3 x the RT chase: the effect exists in
//   this run, as item 8 found: 637.5 and 704.9 against 405.1), hint_builtin (each builtin hint within 15 % of the same
//   hint through inline assembly, and every checksum right).
//   --break expects the builtin NT to behave as RT (within 15 % of the RT chase): FAIL.
#include "mem_common.h"

#include <cstdarg>
#include <map>

namespace {

typedef unsigned u4 __attribute__((ext_vector_type(4)));
constexpr size_t H = 4u << 20;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc_of(const void * p, uint32_t bytes) {
    // dword 3 for gfx11 and gfx12 raw buffers as composable_kernel sets it (CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000)
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), (short) 0, (int) bytes, 0x31004000);
}

// the hinted read: every thread reads 16 bytes of the region per step; out: checksum per thread
template <int AUX>
__global__ __launch_bounds__(256) void k_bl(const char * buf, uint32_t * out) {
    const __amdgpu_buffer_rsrc_t r = rsrc_of(buf, (uint32_t) H);
    uint32_t s = 0;
    for (uint32_t off = (blockIdx.x * 256 + threadIdx.x) * 16; off < H; off += gridDim.x * 256 * 16) {
        const u4 v = __builtin_bit_cast(u4, __builtin_amdgcn_raw_buffer_load_b128(r, (int) off, 0, AUX));
        s ^= v.x + v.y + v.z + v.w;
    }
    out[blockIdx.x * 256 + threadIdx.x] = s;
}
#define AUX_LIST(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) \
    X(18) X(19) X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)
#define AUX_INST(A) template __global__ void k_bl<A>(const char *, uint32_t *);
AUX_LIST(AUX_INST)

// the same read through inline assembly with the hint in the instruction text
#define ASM_K(NAME, HINT)                                                                                            \
    __global__ __launch_bounds__(256) void k_asm_##NAME(const char * buf, uint32_t * out) {                         \
        uint32_t s = 0;                                                                                              \
        for (uint32_t off = (blockIdx.x * 256 + threadIdx.x) * 16; off < H; off += gridDim.x * 256 * 16) {           \
            u4 v;                                                                                                    \
            asm volatile("global_load_b128 %0, %1, off" HINT "\n s_wait_loadcnt 0" : "=v"(v) : "v"(buf + off)       \
                         : "memory");                                                                                \
            s ^= v.x + v.y + v.z + v.w;                                                                              \
        }                                                                                                            \
        out[blockIdx.x * 256 + threadIdx.x] = s;                                                                     \
    }
ASM_K(rt, "")
ASM_K(nt, " th:TH_LOAD_NT")
ASM_K(ht, " th:TH_LOAD_HT")
ASM_K(lu, " th:TH_LOAD_LU")

#define CH1 "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
#define CH16 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1
__global__ __launch_bounds__(32) void k_chase(uint64_t start, uint32_t steps, uint64_t * out) {
    uint64_t p = start;
    const uint64_t c0 = wave_cycles();
#pragma unroll 1
    for (uint32_t i = 0; i < steps / 16; ++i) {
        asm volatile(CH16 : "+v"(p)::"memory");
    }
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = p;
    }
}

__global__ void k_flush(const u4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += p[i].x;
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1200];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = mem::has_flag(argc, argv, "--break");
    mem::clocks("start");
    char *buf, *fl;
    uint32_t *out, *sink;
    uint64_t * cout;
    CHECK(hipMalloc(&buf, H));
    CHECK(hipMalloc(&fl, 512u << 20));
    CHECK(hipMalloc(&out, 256 * 256 * 4));
    CHECK(hipMalloc(&sink, 512 * 256 * 4));
    CHECK(hipMalloc(&cout, 16));
    mem::fill(fl, 512u << 20, mem::RANDOM);
    std::vector<uint32_t> order;
    const size_t nodes = H / 256;
    const uint64_t start = mem::build_chase(buf, nodes, 256, true, 3, &order);
    const uint64_t expect = (uint64_t) buf + (uint64_t) order[4096 % nodes] * 256;
    using K = void (*)(const char *, uint32_t *);
    auto sum_of = [&](K k) {
        hipLaunchKernelGGL(k, dim3(256), dim3(256), 0, 0, buf, out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h(256 * 256);
        CHECK(hipMemcpy(h.data(), out, h.size() * 4, hipMemcpyDeviceToHost));
        uint64_t s = 0;
        for (uint32_t v : h) s = s * 31 + v;
        return s;
    };
    const uint64_t ref = sum_of(k_asm_rt);
    // one trial: flush, hinted read, chase in a new launch
    auto trial = [&](K k, bool & sum_ok, bool & ptr_ok) {
        hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const u4 *) fl, (uint32_t) ((512u << 20) / 16), sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        sum_ok &= sum_of(k) == ref;
        hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, 4096u, cout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, cout, 16, hipMemcpyDeviceToHost));
        ptr_ok &= o[1] == expect;
        return (double) o[0] / 4096;
    };
    struct R {
        const char * name;
        K asm_k, bl_k;
    } rs[] = {{"rt", k_asm_rt, k_bl<0>}, {"nt", k_asm_nt, k_bl<1>}, {"ht", k_asm_ht, k_bl<2>}, {"lu", k_asm_lu, k_bl<3>}};
    std::map<std::string, double> as, bl;
    bool sum_ok = true, ptr_ok = true;
    for (const R & r : rs) {
        std::vector<double> a, b;
        for (int t = 0; t < 5; ++t) {  // interleaved: the two paths alternate
            a.push_back(trial(r.asm_k, sum_ok, ptr_ok));
            b.push_back(trial(r.bl_k, sum_ok, ptr_ok));
        }
        as[r.name] = median(a);
        bl[r.name] = median(b);
        std::printf("RESULT hint %s chase after a hinted 4 MiB read: inline asm %.1f (%.1f-%.1f), builtin aux %d %.1f "
                    "(%.1f-%.1f) cycles per load\n",
                    r.name, as[r.name], *std::min_element(a.begin(), a.end()), *std::max_element(a.begin(), a.end()),
                    (int) (&r - rs), bl[r.name], *std::min_element(b.begin(), b.end()), *std::max_element(b.begin(), b.end()));
    }
    // also the other aux values with a scope: does the scope field change where the data is left? (INFO)
    for (int aux : {8, 16, 24, 9, 25}) {
        K k = nullptr;
#define PICK(A) if (aux == A) k = k_bl<A>;
        AUX_LIST(PICK)
#undef PICK
        std::vector<double> b;
        for (int t = 0; t < 3; ++t) b.push_back(trial(k, sum_ok, ptr_ok));
        std::printf("RESULT hint builtin aux %2d (th %d, scope %d): chase %.1f cycles per load\n", aux, aux & 7, aux >> 3,
                    median(b));
    }
    check("hint_control", as["nt"] >= 1.3 * as["rt"] && as["lu"] >= 1.3 * as["rt"] && ptr_ok,
          "inline assembly: RT %.1f, NT %.1f, HT %.1f, LU %.1f cycles per chase load (item 8: 405.1, 637.5, 460.7, 704.9); "
          "NT and LU at least 1.3 x RT; chase pointers %s",
          as["rt"], as["nt"], as["ht"], as["lu"], ptr_ok ? "right" : "WRONG");
    bool within = sum_ok;
    std::string txt;
    for (const R & r : rs) {
        const double want = brk && std::string(r.name) == "nt" ? as["rt"] : as[r.name];
        within &= std::fabs(bl[r.name] / want - 1) <= 0.15;
        char b[96];
        std::snprintf(b, sizeof b, " %s %.1f/%.1f", r.name, bl[r.name], want);
        txt += b;
    }
    check("hint_builtin", within,
          "builtin against inline assembly (cycles per chase load):%s, each within 15 %%%s; checksums of every hinted "
          "read %s",
          txt.c_str(), brk ? " (deliberate break: builtin NT expected to behave as RT)" : "", sum_ok ? "right" : "WRONG");
    mem::clocks("end");
    return fails ? 1 : 0;
}
