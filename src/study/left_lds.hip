// Leftovers C, lane A: LDS items LD-2, LD-1a, LD-1b, LD-1d, LD-3a, LD-3b (measurements/leftovers-c-a.md).
//
// Usage: left_lds <banks|inflight2|misalign|atomic|swz> [--break]
//   banks      LD-3a, LD-3b: cost per access of ds_load_b32/b64/b96/b128 and ds_store_b32/b64/b96/b128 for the
//              patterns of lds item 2 (same address, sequential, strides, random), against the 32-bank model applied
//              to the whole wave, half-waves and quarter-waves (and thirds for 96-bit); after the stores, which lane's
//              value is left where lanes store to one address. Calibration: lds item 2 (b32 stride 32 = 32.00,
//              stride 2 = 2.00, sequential 1.00; b128 sequential 6.00). --break expects stride 33 to collide like
//              stride 32: the calibration verdict must FAIL.
//              (stride 8 elements: 32 cycles each, lds item 2) against conflict-free (sequential, 6 cycles each).
//              The guide's 6-bit DS counter (63, READ) predicts that the 64th load cannot issue before the first
//              returns when loads are slow; whether the wave stops issuing earlier, and where, is the finding (INFO).
//              Verdict lds_inflight_calib: the data time grows by the per-access cost of lds item 2 (32 and 6 cycles
//              per load between N = 32 and 96). --break expects 6 cycles per load for stride 8: FAIL.
//   misalign   LD-1a: ds_load_b32/b64/b96/b128 at +1, +2, +3, +4, +8 bytes from the aligned lane address: which bytes
//              come back (byte-exact unaligned, or the address masked to 4, 8 or 16 bytes, guide 3.3.5.1) and the
//              cost. --break checks the aligned control against the masked-to-16 rule for b32 at +4: FAIL.
//   atomic     LD-1b: ds_add_u32, ds_add_rtn_u32, ds_max_u32, ds_cmpstore_b32 under the patterns: results against the
//              requirement (32 lanes adding 1 to one word give +32; returned values distinct), and cost. --break
//              expects +31: FAIL.
//   swz        LD-1d: ds_swizzle_b32 offsets (bit-mask and quad-permute modes against the guide's formulas, rotate and
//              FFT modes reported) and cost; ds_bpermute_b32 cost under index patterns. --break uses a model with one
//              and-mask bit wrong: FAIL.
// Method: one wave alone; each timed region is one inline-assembly statement with the counter reads, a loop of NIT
// iterations of R copies of a 16-instruction block, and s_wait_dscnt 0 before the second read; the per-lane address
// is in v1 and the data in v2..v5 (fixed registers). Cost per access = (R = 6 - R = 2) / (4 x 16 x NIT); R = 4 must
// sit midway. The addresses are data, so one kernel per instruction serves every pattern.
#include "tim_common.h"

#include <map>
#include <random>
#include <set>

namespace {

constexpr int REPS = 6;
constexpr int LAUNCHES = 3;
constexpr uint32_t NIT = 8;
constexpr uint32_t LDS_WORDS = 16384;  // 64 KiB

__device__ __forceinline__ void fill_lds(uint32_t * lds, uint32_t kind) {
    for (uint32_t i = threadIdx.x; i < LDS_WORDS; i += 32) {
        uint32_t v;
        if (kind == 0) {
            v = i * 4;  // data = byte address of the dword
        } else if (kind == 1) {
            v = 0;
        } else {
            // a byte pattern: byte b holds (b * 29 + 7) & 255
            const uint32_t b = i * 4;
            v = ((b * 29 + 7) & 255) | (((b + 1) * 29 + 7) & 255) << 8 | (((b + 2) * 29 + 7) & 255) << 16 |
                (((b + 3) * 29 + 7) & 255) << 24;
        }
        lds[i] = v;
    }
    __syncthreads();
}

#define LCLOB "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory"
#define LIN [a] "{v1}"(a), [d0] "{v2}"(d0), [d1] "{v3}"(d1), [d2] "{v4}"(d2), [d3] "{v5}"(d3)

// out: per repetition [t0, t1]
#define LDS_K(NAME, TEXT, R)                                                                                         \
    __global__ __launch_bounds__(32) void k_##NAME##_##R(const uint32_t * in, uint32_t * out) {                     \
        __shared__ uint32_t lds[LDS_WORDS];                                                                          \
        fill_lds(lds, 0);                                                                                            \
        const uint32_t l = threadIdx.x;                                                                              \
        const uint32_t a = in[l], d0 = in[32 + l], d1 = in[64 + l], d2 = in[96 + l], d3 = in[128 + l];              \
        for (int rep = 0; rep < REPS; ++rep) {                                                                       \
            uint32_t t0, t1, cnt;                                                                                    \
            asm volatile("s_wait_dscnt 0\n s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n" \
                         " 2:\n .rept " #R "\n" TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT  \
                         TEXT TEXT ".endr\n s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n"              \
                         " s_cbranch_scc1 2b\n s_wait_dscnt 0\n s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n" \
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt)                                         \
                         : [n] "s"(NIT), LIN                                                                         \
                         : LCLOB);                                                                                   \
            if (l == 0) {                                                                                            \
                out[2 * rep] = t0;                                                                                   \
                out[2 * rep + 1] = t1;                                                                               \
            }                                                                                                        \
        }                                                                                                            \
        /* a read the compiler sees keeps the fill and the 64 KiB allocation (without it both were removed) */      \
        out[64 + l] = lds[(l * 997u) % LDS_WORDS];                                                                   \
    }

// name, instruction text (one access), kind for the model: load or store width in bytes
#define LDS_OPS(X)                                                                                                   \
    X(ld32, "ds_load_b32 v16, v1\n")                                                                                 \
    X(ld64, "ds_load_b64 v[16:17], v1\n")                                                                            \
    X(ld96, "ds_load_b96 v[16:18], v1\n")                                                                            \
    X(ld128, "ds_load_b128 v[16:19], v1\n")                                                                          \
    X(st32, "ds_store_b32 v1, v2\n")                                                                                 \
    X(st64, "ds_store_b64 v1, v[2:3]\n")                                                                             \
    X(st96, "ds_store_b96 v1, v[2:4]\n")                                                                             \
    X(st128, "ds_store_b128 v1, v[2:5]\n")                                                                           \
    X(add, "ds_add_u32 v1, v2\n")                                                                                    \
    X(addrtn, "ds_add_rtn_u32 v16, v1, v2\n")                                                                        \
    X(max, "ds_max_u32 v1, v2\n")                                                                                    \
    X(cmpst, "ds_cmpstore_b32 v1, v2, v3\n")                                                                         \
    X(bperm, "ds_bpermute_b32 v16, v1, v2\n")                                                                        \
    X(bperm_w, "ds_bpermute_b32 v16, v1, v2\ns_wait_dscnt 0\n")                                                      \
    X(ld32_w, "ds_load_b32 v16, v1\ns_wait_dscnt 0\n")                                                               \
    X(addrtn_w, "ds_add_rtn_u32 v16, v1, v2\ns_wait_dscnt 0\n")                                                      \
    X(sw_x1, "ds_swizzle_b32 v16, v2 offset:0x041f\n")                                                               \
    X(sw_x1_w, "ds_swizzle_b32 v16, v2 offset:0x041f\ns_wait_dscnt 0\n")                                             \
    X(sw_rev, "ds_swizzle_b32 v16, v2 offset:0x7c1f\n")                                                              \
    X(sw_bc5, "ds_swizzle_b32 v16, v2 offset:0x00a0\n")                                                              \
    X(sw_mix, "ds_swizzle_b32 v16, v2 offset:0x083c\n")                                                              \
    X(sw_x16, "ds_swizzle_b32 v16, v2 offset:0x401f\n")                                                              \
    X(sw_qp1, "ds_swizzle_b32 v16, v2 offset:0x80b1\n")                                                              \
    X(sw_qp2, "ds_swizzle_b32 v16, v2 offset:0x804f\n")                                                              \
    X(sw_rot, "ds_swizzle_b32 v16, v2 offset:0xc07f\n")                                                              \
    X(sw_fft, "ds_swizzle_b32 v16, v2 offset:0xe005\n")                                                              \
    X(sw_rot5, "ds_swizzle_b32 v16, v2 offset:0xc0a1\n")

#define LDS_KS(NAME, TEXT) LDS_K(NAME, TEXT, 2) LDS_K(NAME, TEXT, 4) LDS_K(NAME, TEXT, 6)
LDS_OPS(LDS_KS)

using Kern = void (*)(const uint32_t *, uint32_t *);
struct OpK {
    const char * name;
    Kern k[3];
};
#define LDS_ROW(NAME, TEXT) {#NAME, {k_##NAME##_2, k_##NAME##_4, k_##NAME##_6}},
const OpK ops[] = {LDS_OPS(LDS_ROW)};
const OpK & op(const std::string & n) {
    for (const OpK & o : ops) {
        if (n == o.name) return o;
    }
    std::fprintf(stderr, "no op %s\n", n.c_str());
    std::exit(2);
}

// LD-2: N loads issued back to back; out per repetition [t_issued - t0, t_data - t0]. W32: ds_load_b32 instead of b128
// (is the issue limit a count of instructions or of bytes?)
template <int N, bool W32 = false>
__global__ __launch_bounds__(32) void k_infl(const uint32_t * in, uint32_t * out) {
    __shared__ uint32_t lds[LDS_WORDS];
    fill_lds(lds, 0);
    const uint32_t l = threadIdx.x;
    const uint32_t a = in[l], d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        uint32_t t0, t1, t2;
        if (W32) {
            asm volatile("s_wait_dscnt 0\n s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n .rept %c[n]\n"
                         " ds_load_b32 v16, v1\n .endr\n s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                         " s_wait_dscnt 0\n s_getreg_b32 %[t2], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [t2] "=&s"(t2)
                         : [n] "i"(N), LIN
                         : LCLOB);
        } else {
            asm volatile("s_wait_dscnt 0\n s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n .rept %c[n]\n"
                         " ds_load_b128 v[16:19], v1\n .endr\n s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                         " s_wait_dscnt 0\n s_getreg_b32 %[t2], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                         : [t0] "=&s"(t0), [t1] "=&s"(t1), [t2] "=&s"(t2)
                         : [n] "i"(N), LIN
                         : LCLOB);
        }
        if (l == 0) {
            out[2 * rep] = t1 - t0;
            out[2 * rep + 1] = t2 - t0;
        }
    }
    out[64 + l] = lds[(l * 997u) % LDS_WORDS];
}
#define INFL_NS(X) X(1) X(2) X(4) X(8) X(12) X(14) X(15) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(24) X(32) X(40) X(48) X(56) X(60) X(61) X(62) X(63) X(64) X(65) X(66) X(67) X(68) X(72) X(80) X(96)

// Correctness: one instruction of kind `opi` with the lanes' addresses and data; out: [lane*4 + 0..3] the returned
// registers (v16..v19), then the first 4,096 dwords of LDS after the instruction.
__global__ __launch_bounds__(32) void k_check(int opi, uint32_t init, const uint32_t * in, uint32_t * out) {
    __shared__ uint32_t lds[LDS_WORDS];
    fill_lds(lds, init);
    const uint32_t l = threadIdx.x;
    const uint32_t a = in[l], d0 = in[32 + l], d1 = in[64 + l], d2 = in[96 + l], d3 = in[128 + l];
    uint32_t r0 = 0xDEADBEEF, r1 = 0xDEADBEEF, r2 = 0xDEADBEEF, r3 = 0xDEADBEEF;
#define CK(I, TEXT)                                                                                                  \
    case I:                                                                                                          \
        asm volatile("v_mov_b32 v16, 0xdeadbeef\n v_mov_b32 v17, 0xdeadbeef\n v_mov_b32 v18, 0xdeadbeef\n"          \
                     " v_mov_b32 v19, 0xdeadbeef\n" TEXT " s_wait_dscnt 0\n v_mov_b32 %[r0], v16\n"                  \
                     " v_mov_b32 %[r1], v17\n v_mov_b32 %[r2], v18\n v_mov_b32 %[r3], v19\n"                          \
                     : [r0] "=v"(r0), [r1] "=v"(r1), [r2] "=v"(r2), [r3] "=v"(r3)                                    \
                     : LIN                                                                                           \
                     : LCLOB);                                                                                       \
        break;
    switch (opi) {
        CK(0, "ds_load_b32 v16, v1\n")
        CK(1, "ds_load_b64 v[16:17], v1\n")
        CK(2, "ds_load_b96 v[16:18], v1\n")
        CK(3, "ds_load_b128 v[16:19], v1\n")
        CK(4, "ds_store_b32 v1, v2\n")
        CK(5, "ds_store_b64 v1, v[2:3]\n")
        CK(6, "ds_store_b96 v1, v[2:4]\n")
        CK(7, "ds_store_b128 v1, v[2:5]\n")
        CK(8, "ds_add_u32 v1, v2\n")
        CK(9, "ds_add_rtn_u32 v16, v1, v2\n")
        CK(10, "ds_max_u32 v1, v2\n")
        CK(11, "ds_cmpstore_b32 v1, v2, v3\n")
        CK(12, "ds_cmpstore_rtn_b32 v16, v1, v2, v3\n")
        CK(13, "ds_bpermute_b32 v16, v1, v2\n")
        CK(14, "ds_swizzle_b32 v16, v2 offset:0x041f\n")
        CK(15, "ds_swizzle_b32 v16, v2 offset:0x7c1f\n")
        CK(16, "ds_swizzle_b32 v16, v2 offset:0x00a0\n")
        CK(17, "ds_swizzle_b32 v16, v2 offset:0x083c\n")
        CK(18, "ds_swizzle_b32 v16, v2 offset:0x401f\n")
        CK(19, "ds_swizzle_b32 v16, v2 offset:0x80b1\n")
        CK(20, "ds_swizzle_b32 v16, v2 offset:0x804f\n")
        CK(21, "ds_swizzle_b32 v16, v2 offset:0xc07f\n")
        CK(22, "ds_swizzle_b32 v16, v2 offset:0xe005\n")
        CK(23, "ds_swizzle_b32 v16, v2 offset:0xc0a1\n")
        default: break;
    }
#undef CK
    out[l * 4 + 0] = r0;
    out[l * 4 + 1] = r1;
    out[l * 4 + 2] = r2;
    out[l * 4 + 3] = r3;
    __syncthreads();
    for (uint32_t i = l; i < 4096; i += 32) out[128 + i] = lds[i];
}

uint32_t *g_in, *g_out;

// lane inputs: address, then d0..d3
struct Lanes {
    uint32_t a[32], d[4][32];
};
void upload(const Lanes & L) {
    std::vector<uint32_t> h(160);
    for (int i = 0; i < 32; ++i) {
        h[i] = L.a[i];
        for (int k = 0; k < 4; ++k) h[32 * (k + 1) + i] = L.d[k][i];
    }
    CHECK(hipMemcpy(g_in, h.data(), h.size() * 4, hipMemcpyHostToDevice));
}

std::vector<double> samples(Kern k) {
    std::vector<double> v;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL(k, dim3(1), dim3(32), 0, 0, g_in, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 2 * REPS);
        for (int r = 1; r < REPS; ++r) v.push_back((double) (h[2 * r + 1] - h[2 * r]));
    }
    return v;
}
struct Cost {
    double per, mid, lo, hi;
};
Cost cost(const OpK & o) {
    tim::Clean m[3];
    for (int i = 0; i < 3; ++i) m[i] = tim::clean_of(samples(o.k[i]));
    const double d = 4.0 * 16 * NIT;
    return {(m[2].med - m[0].med) / d, (m[1].med - 0.5 * (m[0].med + m[2].med)) / NIT, (m[2].lo - m[0].hi) / d,
            (m[2].hi - m[0].lo) / d};
}

// Address patterns in units of the element size W (bytes); "ofs" adds a byte offset to every lane.
std::vector<uint32_t> pattern(const std::string & p, uint32_t W, uint32_t ofs = 0) {
    std::vector<uint32_t> a(32);
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t e = i;
        if (p == "same") e = 0;
        else if (p == "seq") e = i;
        else if (p.rfind("s", 0) == 0 && p.size() > 1 && std::isdigit((unsigned char) p[1])) e = i * std::stoul(p.substr(1));
        else if (p.rfind("rand", 0) == 0) {
            std::mt19937 rng(100 + std::stoul(p.substr(4)));
            std::vector<uint32_t> pool(512);
            for (uint32_t k = 0; k < 512; ++k) pool[k] = k;
            std::shuffle(pool.begin(), pool.end(), rng);
            e = pool[i];
        }
        a[i] = e * W + ofs;
    }
    return a;
}

// Bank model: groups of g lanes in turn; in each, the passes are the most distinct dwords any bank must deliver.
int model(const std::vector<uint32_t> & a, uint32_t bytes, int g) {
    int total = 0;
    const int n = (int) a.size();
    for (int s = 0; s < n; s += g) {
        std::map<int, std::set<uint32_t>> bank;
        for (int i = s; i < std::min(n, s + g); ++i) {
            for (uint32_t w = a[i] / 4; w < (a[i] + bytes + 3) / 4; ++w) bank[w % 32].insert(w);
        }
        size_t mx = 0;
        for (auto & kv : bank) mx = std::max(mx, kv.second.size());
        total += (int) mx;
    }
    return total;
}

const char * PATS[] = {"same", "seq", "s2", "s3", "s4", "s8", "s16", "s32", "s33", "rand0", "rand1", "rand2"};

int mode_banks(bool brk) {
    std::map<std::string, double> got;
    int nonlin = 0, st_nonlin = 0;
    struct W {
        const char * op;
        uint32_t bytes, elem;
    } ws[] = {{"ld32", 4, 4}, {"ld64", 8, 8}, {"ld96", 12, 16}, {"ld128", 16, 16},
              {"st32", 4, 4}, {"st64", 8, 8}, {"st96", 12, 16}, {"st128", 16, 16}};
    std::printf("RESULT banks columns: op pattern cycles_per_access range model_passes wave half quarter third(b96: 11 "
                "lanes, then 11, 10)\n");
    for (const W & w : ws) {
        std::vector<std::string> pats(std::begin(PATS), std::end(PATS));
        if (w.bytes == 12) pats.push_back("packed12");
        for (const std::string & p : pats) {
            std::vector<uint32_t> a = p == "packed12" ? pattern("seq", 12) : pattern(p, w.elem);
            Lanes L{};
            for (int i = 0; i < 32; ++i) {
                L.a[i] = a[i];
                for (int k = 0; k < 4; ++k) L.d[k][i] = 0x1000 * (k + 1) + i;
            }
            upload(L);
            const Cost c = cost(op(w.op));
            got[std::string(w.op) + "/" + p] = c.per;
            if (w.op[0] == 'l') {
                nonlin += std::fabs(c.mid) > 1.5;
            } else {
                st_nonlin += std::fabs(c.mid) > 1.5;
            }
            // thirds: 11, 11, 10 lanes
            int third = 0;
            for (int s : {0, 11, 22}) {
                third += model(std::vector<uint32_t>(a.begin() + s, a.begin() + std::min(32, s + 11)), w.bytes, 32);
            }
            std::printf("RESULT banks %-6s %-8s %6.2f (%.2f-%.2f) model %2d %2d %2d %2d mid %.2f\n", w.op, p.c_str(), c.per,
                        c.lo, c.hi, model(a, w.bytes, 32), model(a, w.bytes, 16), model(a, w.bytes, 8), third, c.mid);
            std::fflush(stdout);
        }
    }
    const double s33 = got["ld32/s33"], s32 = got["ld32/s32"], s2 = got["ld32/s2"], sq = got["ld32/seq"],
                 q = got["ld128/seq"];
    const bool ok = std::fabs(s32 - 32) <= 0.1 && std::fabs(s2 - 2) <= 0.05 && std::fabs(sq - 1) <= 0.05 &&
                    std::fabs(q - 6) <= 0.1 && (brk ? std::fabs(s33 - 32) <= 0.1 : std::fabs(s33 - 1) <= 0.05) &&
                    nonlin == 0;
    tim::check("lds_banks_calib", ok,
               "ds_load_b32 stride 32 %.2f (lds item 2: 32.00), stride 2 %.2f (2.00), sequential %.2f (1.00), stride 33 "
               "%.2f (%s); ds_load_b128 sequential %.2f (6.00); %d load points off the midpoint (stores, INFO: %d)",
               s32, s2, sq, s33, brk ? "deliberate break: expected 32" : "1.00", q, nonlin, st_nonlin);
    // who wins when every lane stores to one address
    for (int opi : {4, 5, 7}) {
        Lanes L{};
        for (int i = 0; i < 32; ++i) {
            L.a[i] = 256;
            for (int k = 0; k < 4; ++k) L.d[k][i] = 0x100 * (k + 1) + i;
        }
        upload(L);
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, opi, 1u, g_in, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 128 + 4096);
        std::printf("RESULT banks store_same_address %s words at 256: %#x %#x %#x %#x (lane = low byte)\n",
                    opi == 4 ? "b32" : opi == 5 ? "b64" : "b128", h[128 + 64], h[128 + 65], h[128 + 66], h[128 + 67]);
    }
    return 0;
}

// expected value of a misaligned load: semantics 0 = byte-exact at the address, m = address masked to m bytes
uint32_t byte_at(uint32_t b) { return (b * 29 + 7) & 255; }
uint32_t word_from(uint32_t addr, int k) {
    return byte_at(addr + 4 * k) | byte_at(addr + 4 * k + 1) << 8 | byte_at(addr + 4 * k + 2) << 16 |
           byte_at(addr + 4 * k + 3) << 24;
}

int mode_misalign(bool brk) {
    struct C {
        int opi;
        const char * op;
        uint32_t bytes, elem;
    } cs[] = {{0, "ld32", 4, 4}, {1, "ld64", 8, 8}, {2, "ld96", 12, 16}, {3, "ld128", 16, 16}};
    bool all_one = true;
    for (const C & c : cs) {
        for (uint32_t off : {0u, 1u, 2u, 3u, 4u, 8u}) {
            const std::vector<uint32_t> a = pattern("seq", c.elem, off);
            Lanes L{};
            for (int i = 0; i < 32; ++i) L.a[i] = a[i];
            upload(L);
            hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, c.opi, 2u, g_in, g_out);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const std::vector<uint32_t> h = tim::host_u32(g_out, 128);
            // candidates: byte-exact; masked to 4, 8, 16 bytes
            const uint32_t masks[4] = {0, 4, 8, 16};
            int match[4] = {0, 0, 0, 0};
            for (int i = 0; i < 32; ++i) {
                for (int m = 0; m < 4; ++m) {
                    const uint32_t base = masks[m] ? (a[i] & ~(masks[m] - 1)) : a[i];
                    bool eq = true;
                    for (int k = 0; k < (int) c.bytes / 4; ++k) eq &= h[i * 4 + k] == word_from(base, k);
                    match[m] += eq;
                }
            }
            std::string which;
            const char * mn[4] = {"byte-exact", "masked-to-4", "masked-to-8", "masked-to-16"};
            for (int m = 0; m < 4; ++m) {
                if (match[m] == 32) which += std::string(which.empty() ? "" : "=") + mn[m];
            }
            if (which.empty()) {
                which = "NONE";
                all_one = false;
            }
            // control for the break: the aligned b32 at +4 judged against masked-to-16 must fail
            if (brk && c.opi == 0 && off == 4) all_one &= match[3] == 32;
            // cost
            upload(L);
            const Cost t = cost(op(c.op));
            std::printf("RESULT misalign %-5s +%u bytes: lanes matching byte-exact %d, masked-4 %d, masked-8 %d, "
                        "masked-16 %d -> %s; cycles per access %.2f (%.2f-%.2f); model passes (byte span) %d\n",
                        c.op, off, match[0], match[1], match[2], match[3], which.c_str(), t.per, t.lo, t.hi,
                        model(a, c.bytes, 32));
            std::fflush(stdout);
        }
    }
    tim::check("lds_misalign", all_one,
               "every width and offset returned one of the candidate byte sets in all 32 lanes%s",
               brk ? " (deliberate break: the aligned b32 at +4 judged by the masked-to-16 rule)" : "");
    return 0;
}

int mode_atomic(bool brk) {
    bool ok = true;
    struct P {
        const char * name;
        uint32_t stride_dw;  // 0 = all lanes one word
    } ps[] = {{"same", 0}, {"seq", 1}, {"s2", 2}, {"s4", 4}, {"s8", 8}, {"s16", 16}, {"s32", 32}};
    for (const P & p : ps) {
        Lanes L{};
        for (int i = 0; i < 32; ++i) {
            L.a[i] = 1024 + 4 * p.stride_dw * i;
            L.d[0][i] = 1;            // add operand
            L.d[1][i] = 1000 + i;     // max operand / cmpstore: the new value
        }
        // ds_add_u32: one instruction, LDS zero
        upload(L);
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, 8, 1u, g_in, g_out);
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> h = tim::host_u32(g_out, 128 + 4096);
        std::map<uint32_t, uint32_t> want;
        for (int i = 0; i < 32; ++i) want[L.a[i] / 4] += 1;
        uint32_t add_bad = 0, sum = 0;
        for (auto & kv : want) {
            const uint32_t w = brk && p.stride_dw == 0 ? kv.second - 1 : kv.second;
            add_bad += h[128 + kv.first] != w;
            sum += h[128 + kv.first];
        }
        // ds_add_rtn_u32: returned values per word must be distinct, 0..k-1
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, 9, 1u, g_in, g_out);
        CHECK(hipDeviceSynchronize());
        h = tim::host_u32(g_out, 128 + 4096);
        std::map<uint32_t, std::set<uint32_t>> ret;
        std::string order;
        for (int i = 0; i < 32; ++i) {
            ret[L.a[i] / 4].insert(h[i * 4]);
            if (p.stride_dw == 0) order += std::to_string(h[i * 4]) + (i < 31 ? "," : "");
        }
        uint32_t rtn_bad = 0;
        for (auto & kv : ret) {
            rtn_bad += kv.second.size() != want[kv.first] || *kv.second.rbegin() != want[kv.first] - 1;
        }
        // ds_max_u32 with lane values 1000 + lane, then ds_cmpstore_b32 (new = 2000 + lane, compare = 1000 + lane)
        for (int i = 0; i < 32; ++i) L.d[0][i] = 1000 + i;
        upload(L);
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, 10, 1u, g_in, g_out);
        CHECK(hipDeviceSynchronize());
        h = tim::host_u32(g_out, 128 + 4096);
        std::map<uint32_t, uint32_t> mx;
        for (int i = 0; i < 32; ++i) mx[L.a[i] / 4] = std::max(mx[L.a[i] / 4], 1000u + i);
        uint32_t max_bad = 0;
        for (auto & kv : mx) max_bad += h[128 + kv.first] != kv.second;
        // cmpstore on LDS that holds the byte address (init 0): mem == cmp only for the lane whose cmp is the address
        for (int i = 0; i < 32; ++i) {
            L.d[0][i] = 5000 + i;  // the value to store
            L.d[1][i] = L.a[i];    // compare value: equal to the word's content (data = byte address)
        }
        upload(L);
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, 12, 0u, g_in, g_out);
        CHECK(hipDeviceSynchronize());
        h = tim::host_u32(g_out, 128 + 4096);
        uint32_t cmp_bad = 0;
        std::string cmp_note;
        for (auto & kv : want) {
            const uint32_t v = h[128 + kv.first];
            // one lane wins (its value 5000 + lane); the others see a changed word and do nothing
            cmp_bad += !(v >= 5000 && v < 5032 && L.a[v - 5000] / 4 == kv.first);
            if (p.stride_dw == 0) cmp_note = " cmpstore winner lane " + std::to_string(v - 5000);
        }
        const bool pok = add_bad == 0 && rtn_bad == 0 && max_bad == 0 && cmp_bad == 0;
        ok &= pok;
        // costs
        Lanes T{};
        for (int i = 0; i < 32; ++i) {
            T.a[i] = 1024 + 4 * p.stride_dw * i;
            T.d[0][i] = 1;
            T.d[1][i] = 0;
        }
        upload(T);
        const Cost ca = cost(op("add")), cr = cost(op("addrtn")), cm = cost(op("max")), cc = cost(op("cmpst")),
                   cw = cost(op("addrtn_w")), cl = cost(op("ld32"));
        std::vector<uint32_t> av(T.a, T.a + 32);
        int same_count = 0;
        for (auto & kv : want) same_count = std::max<int>(same_count, (int) kv.second);
        std::printf("RESULT atomic %-4s add %s rtn %s max %s cmpstore %s%s | cycles per access: ds_add %.2f ds_add_rtn %.2f "
                    "ds_max %.2f ds_cmpstore %.2f ds_load_b32 %.2f; add_rtn issue to data %.2f | load model passes %d, "
                    "most lanes on one word %d%s\n",
                    p.name, add_bad ? "WRONG" : "ok", rtn_bad ? "WRONG" : "ok", max_bad ? "WRONG" : "ok",
                    cmp_bad ? "WRONG" : "ok", cmp_note.c_str(), ca.per, cr.per, cm.per, cc.per, cl.per, cw.per,
                    model(av, 4, 32), same_count, p.stride_dw == 0 ? (" rtn order " + order).c_str() : "");
        std::fflush(stdout);
        (void) sum;
    }
    tim::check("lds_atomic", ok,
               "ds_add_u32 sums, ds_add_rtn_u32 returned sets, ds_max_u32 and ds_cmpstore_rtn_b32 results right for all 7 "
               "patterns%s",
               brk ? " (deliberate break: 32 lanes on one word expected to add 31)" : "");
    return 0;
}

int mode_swz(bool brk) {
    // data: lane value 100 + lane in v2 (d0)
    Lanes L{};
    for (int i = 0; i < 32; ++i) {
        L.a[i] = 4 * i;
        L.d[0][i] = 100 + i;
    }
    struct S {
        int opi;
        const char * op;
        uint32_t off;
    } ss[] = {{14, "sw_x1", 0x041f}, {15, "sw_rev", 0x7c1f}, {16, "sw_bc5", 0x00a0}, {17, "sw_mix", 0x083c},
              {18, "sw_x16", 0x401f}, {19, "sw_qp1", 0x80b1}, {20, "sw_qp2", 0x804f}, {21, "sw_rot", 0xc07f},
              {22, "sw_fft", 0xe005}, {23, "sw_rot5", 0xc0a1}};
    int mism = 0, modeled = 0;
    for (const S & s : ss) {
        upload(L);
        hipLaunchKernelGGL(k_check, dim3(1), dim3(32), 0, 0, s.opi, 0u, g_in, g_out);
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 128);
        std::string got;
        int bad = 0;
        const bool has_model = s.off < 0xc000;
        for (int i = 0; i < 32; ++i) {
            int j;
            if (s.off & 0x8000) {
                j = (i & ~3) | ((s.off >> (2 * (i & 3))) & 3);
            } else {
                uint32_t andm = s.off & 31, orm = (s.off >> 5) & 31, xorm = (s.off >> 10) & 31;
                if (brk) andm ^= 1;  // the model with one and-mask bit wrong
                j = (int) (((i & andm) | orm) ^ xorm);
            }
            const uint32_t v = h[i * 4];
            got += std::to_string((int) v - 100) + (i < 31 ? "," : "");
            if (has_model) bad += v != 100u + j;
        }
        if (has_model) {
            ++modeled;
            mism += bad;
        }
        upload(L);
        const Cost c = cost(op(s.op));
        std::printf("RESULT swz %-6s offset %#06x source lanes [%s] %s; cycles per access %.2f\n", s.op, s.off, got.c_str(),
                    has_model ? (bad ? "MISMATCH" : "= model") : "(no model: reported)", c.per);
        std::fflush(stdout);
    }
    upload(L);
    const Cost w1 = cost(op("sw_x1_w"));
    std::printf("RESULT swz issue_to_data sw_x1 %.2f cycles\n", w1.per);
    tim::check("lds_swizzle", mism == 0,
               "%d bit-mask and quad-permute offsets against the guide's formulas: %d lane mismatches%s", modeled, mism,
               brk ? " (deliberate break: and-mask bit 0 flipped in the model)" : "");
    // bpermute: index patterns (address = 4 x source lane)
    struct B {
        const char * name;
        int k;
    } bs[] = {{"identity", -1}, {"all_lane0", 0}, {"stride2", 2}, {"stride4", 4}, {"stride8", 8}, {"stride16", 16},
              {"stride32", 32}, {"reverse", -2}, {"random", -3}};
    double id_issue = 0, id_lat = 0;
    std::mt19937 rng(5);
    for (const B & b : bs) {
        Lanes P{};
        for (int i = 0; i < 32; ++i) {
            int src = i;
            if (b.k == 0) src = 0;
            else if (b.k > 0) src = (i * b.k) % 32;
            else if (b.k == -2) src = 31 - i;
            else if (b.k == -3) src = (int) (rng() % 32);
            P.a[i] = 4 * src;
            P.d[0][i] = 100 + i;
        }
        upload(P);
        const Cost is = cost(op("bperm")), lt = cost(op("bperm_w"));
        if (b.k == -1) {
            id_issue = is.per;
            id_lat = lt.per;
        }
        std::printf("RESULT bperm %-9s cycles per access %.2f, issue to data %.2f\n", b.name, is.per, lt.per);
        std::fflush(stdout);
    }
    tim::check("lds_bperm_calib", std::fabs(id_lat - 36) <= 4,
               "ds_bpermute_b32 identity: issue to data %.2f cycles (execution-core item 2: 36), independent %.2f per "
               "access (16 in flight: 5.06)",
               id_lat, id_issue);
    return 0;
}

template <int N, bool W32 = false>
std::pair<double, double> infl_one() {
    std::vector<double> is, dt;
    for (int l = 0; l < LAUNCHES; ++l) {
        hipLaunchKernelGGL((k_infl<N, W32>), dim3(1), dim3(32), 0, 0, g_in, g_out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const std::vector<uint32_t> h = tim::host_u32(g_out, 2 * REPS);
        for (int r = 1; r < REPS; ++r) {
            is.push_back(h[2 * r]);
            dt.push_back(h[2 * r + 1]);
        }
    }
    return {tim::clean_of(is).med, tim::clean_of(dt).med};
}

int mode_inflight2(bool brk) {
    std::map<std::string, std::map<int, std::pair<double, double>>> got;
    for (const char * p : {"s8", "seq", "b32_s32"}) {
        Lanes L{};
        const std::vector<uint32_t> a = std::string(p) == "b32_s32" ? pattern("s32", 4) : pattern(p, 16);
        for (int i = 0; i < 32; ++i) L.a[i] = a[i];
        upload(L);
        if (std::string(p) == "b32_s32") {
#define INFL_RUN(N) got[p][N] = infl_one<N, true>();
            INFL_NS(INFL_RUN)
#undef INFL_RUN
        } else {
#define INFL_RUN(N) got[p][N] = infl_one<N>();
            INFL_NS(INFL_RUN)
#undef INFL_RUN
        }
        for (auto & kv : got[p]) {
            std::printf("RESULT inflight2 %-4s N %3d issue_cycles %7.1f data_cycles %7.1f\n", p, kv.first, kv.second.first,
                        kv.second.second);
        }
        std::fflush(stdout);
    }
    const double s8 = (got["s8"][96].second - got["s8"][32].second) / 64.0;
    const double sq = (got["seq"][96].second - got["seq"][32].second) / 64.0;
    const double w8 = brk ? 6.0 : 32.0;
    tim::check("lds_inflight_calib", std::fabs(s8 - w8) <= 0.5 && std::fabs(sq - 6.0) <= 0.2,
               "data time per extra ds_load_b128 between N = 32 and 96: stride 8 %.2f cycles (expect %.0f%s), sequential "
               "%.2f (expect 6; lds item 2)",
               s8, w8, brk ? ": deliberate break" : "", sq);
    // the finding: up to which N issue goes at about one load per cycle, and the step at 63/64 of the counter
    for (const char * p : {"s8", "seq", "b32_s32"}) {
        int last_fast = 0;
        for (auto & kv : got[p]) {
            if (kv.first > 1 && kv.second.first - 8.0 <= 1.5 * kv.first) last_fast = kv.first;
        }
        std::printf("VERDICT lds_issue_queue: INFO %s: issue at about one load per cycle up to N = %d; issue time at "
                    "N = 62, 63, 64, 65: %.0f %.0f %.0f %.0f\n",
                    p, last_fast, got[p][62].first, got[p][63].first, got[p][64].first, got[p][65].first);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = tim::flag(argc, argv, "--break");
    CHECK(hipMalloc(&g_in, 160 * 4));
    CHECK(hipMalloc(&g_out, (128 + 4096) * 4));  // timing kernels use [0, 2 x REPS) and [64, 96)
    mem::clocks("start");
    if (mode == "banks") {
        mode_banks(brk);
    } else if (mode == "inflight2") {
        mode_inflight2(brk);
    } else if (mode == "misalign") {
        mode_misalign(brk);
    } else if (mode == "atomic") {
        mode_atomic(brk);
    } else if (mode == "swz") {
        mode_swz(brk);
    } else {
        std::fprintf(stderr, "usage: left_lds <banks|inflight2|misalign|atomic|swz> [--break]\n");
        return 2;
    }
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
