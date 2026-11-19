// Leftovers pass, matrix unit (notes/STUDY-PLAN.md, "leftovers"): the items of the "Not measured" lists of
// measurements/matrix-arithmetic.md and compiler-and-hazards.md that concern the matrix instructions.
//
// Usage: left_mat MODE [--n N] [--break]
//   s4map   one-hot experiments for the 4-bit integer forms: which B register position each A position meets in the
//           dense IU4 forms (K = 16, 32) and in the sparse ones (v_swmmac_i32_16x16x32_iu4, index_key 0 and 1;
//           v_swmmac_i32_16x16x64_iu4), and which index field decides it; against the rule below.
//   s4      the two 4-bit sparse forms on random data, random index words, every sign setting with and without
//           clamp: the gather rule, then the exact integer model; controls: other clamp grouping, the other key.
//           Also the same products through the dense K = 32 form and the sparse K = 64 form, card against card.
//   mask    EXEC set around one instruction (7 masks, 0 included; D preset to a sentinel, read back under full EXEC)
//           for several matrix forms and two dot instructions (the controls that obey the mask).
//   mods    modifiers not covered before: NEG and ABS on C of the BF8 forms, NEG and NEG_HI on A and B of the sparse
//           float forms, NEG_HI on the integer forms (raw encodings: the assembler refuses it).
//   hazard  the forbidden back-to-back read for BF16 into BF16, the 8-bit integer form, a sparse form, and with a
//           second wave on the same SIMD.
//   hazard2 (leftovers C, MA-2) the same read with the second instruction of another form than the first (FP16 into
//           FP32 then FP16 into FP16; FP8 then FP16; 8-bit integer then BF16; FP16 into FP16 then FP8; each as A and
//           as B), alone and beside a busy wave, and every pair (old and new) under whole-card load: 128 groups of 8
//           waves, 4 running the pair and 4 a long FP16 matrix chain on the same SIMDs, about 80 us. The prediction
//           (section 13a's rule: the second instruction reads operand VGPRs 0 and 1 of lanes 0-15 old, which shows in
//           rows 0, 1, 8, 9 as A and in every column as B) must reproduce the forbidden result exactly. --card runs the
//           whole-card setting only.
//   mods2   (leftovers C, MA-1 and MA-3) NEG_HI on B of the 4-bit and unsigned 8-bit forms, NEG_HI on the unsigned
//           4-bit K = 32 form, and the C bits on the other sparse float forms (raw words, as mods).
//   mask2   (leftovers C, CH-6) every dense and sparse matrix opcode under the 7 masks, and the chained time with a
//           full and a one-lane EXEC (src/study/left_mask.h; the wave64 build is left_mat64 mask).
//   --break perturbs the rule or the expectation; the verdict must turn to FAIL.
#include "arith_data.h"
#include "arith_gpu.h"

#include <atomic>
#include <cstdarg>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>

#define LM_WAVE 32
#include "left_mask.h"

using arith::v2i;
using arith::v4i;
using arith::v8i;

static bool g_break = false;
static bool g_card = false;  // hazard2 --card: the whole-card setting (a separate job: it waits for the thermal gate)

static void verdict(const std::string & probe, bool ok, const std::string & text) {
    std::printf("VERDICT %s: %s %s%s\n", probe.c_str(), ok ? "PASS" : "FAIL", text.c_str(),
                g_break ? " (broken on purpose)" : "");
}

static void add_text(std::string & text, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
static void add_text(std::string & text, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!text.empty()) {
        text += "; ";
    }
    text += buf;
}

static void par_for(size_t n, const std::function<void(size_t, int)> & f) {
    const int T = 8;
    std::vector<std::thread> th;
    std::atomic<size_t> next{0};
    for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < n;) {
                f(i, t);
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Generic one-instruction kernels. An op is a struct with the operand types and one asm statement; %0 is C and D
// (in and out), %1 A, %2 B, %3 the index VGPR (ignored by dense forms). Every wave loads its own operands.
#define LOP(NAME, TA, TB, TC, TEXT)                                                                \
    struct NAME {                                                                                  \
        typedef TA ta;                                                                             \
        typedef TB tb;                                                                             \
        typedef TC tc;                                                                             \
        static __device__ __forceinline__ void go(TA a, TB b, TC & c, int i) {                     \
            asm volatile(TEXT : "+v"(c) : "v"(a), "v"(b), "v"(i));                                 \
        }                                                                                          \
    };

// 4-bit integer forms; settings: neg_lo bits make A / B signed, clamp saturates.
#define S4SET_0 ""
#define S4SET_1 " neg_lo:[1,0,0]"
#define S4SET_2 " neg_lo:[0,1,0]"
#define S4SET_3 " neg_lo:[1,1,0]"
#define S4SET_4 " clamp"
#define S4SET_5 " neg_lo:[1,0,0] clamp"
#define S4SET_6 " neg_lo:[0,1,0] clamp"
#define S4SET_7 " neg_lo:[1,1,0] clamp"
static const char * const s4set_name[8] = {"uu", "sA", "sB", "sAB", "uu clamp", "sA clamp", "sB clamp", "sAB clamp"};
static bool set_sa(int set) { return set & 1; }
static bool set_sb(int set) { return set & 2; }
#define S4OPS(S)                                                                                                   \
    LOP(S32k0_##S, int, v2i, v8i, "v_swmmac_i32_16x16x32_iu4 %0, %1, %2, %3 index_key:0" S4SET_##S)                \
    LOP(S32k1_##S, int, v2i, v8i, "v_swmmac_i32_16x16x32_iu4 %0, %1, %2, %3 index_key:1" S4SET_##S)                \
    LOP(S64_##S, v2i, v4i, v8i, "v_swmmac_i32_16x16x64_iu4 %0, %1, %2, %3" S4SET_##S)                              \
    LOP(D16_##S, int, int, v8i, "v_wmma_i32_16x16x16_iu4 %0, %1, %2, %0" S4SET_##S)                                \
    LOP(D32_##S, v2i, v2i, v8i, "v_wmma_i32_16x16x32_iu4 %0, %1, %2, %0" S4SET_##S)
S4OPS(0) S4OPS(1) S4OPS(2) S4OPS(3) S4OPS(4) S4OPS(5) S4OPS(6) S4OPS(7)

// Modifier ops (item 3). BF8 forms with C bits; sparse float forms with A/B bits.
LOP(FB_none, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_bf8 %0, %1, %2, %0")
LOP(FB_negC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_bf8 %0, %1, %2, %0 neg_lo:[0,0,1]")
LOP(FB_absC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_bf8 %0, %1, %2, %0 neg_hi:[0,0,1]")
LOP(FB_nabsC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_bf8 %0, %1, %2, %0 neg_lo:[0,0,1] neg_hi:[0,0,1]")
LOP(BF_none, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_fp8 %0, %1, %2, %0")
LOP(BF_negC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_fp8 %0, %1, %2, %0 neg_lo:[0,0,1]")
LOP(BF_absC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_fp8 %0, %1, %2, %0 neg_hi:[0,0,1]")
LOP(BF_nabsC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_fp8 %0, %1, %2, %0 neg_lo:[0,0,1] neg_hi:[0,0,1]")
LOP(BB_none, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_bf8 %0, %1, %2, %0")
LOP(BB_negC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_bf8 %0, %1, %2, %0 neg_lo:[0,0,1]")
LOP(BB_absC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_bf8 %0, %1, %2, %0 neg_hi:[0,0,1]")
LOP(BB_nabsC, v2i, v2i, v8i, "v_wmma_f32_16x16x16_bf8_bf8 %0, %1, %2, %0 neg_lo:[0,0,1] neg_hi:[0,0,1]")
LOP(FF_none, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_fp8 %0, %1, %2, %0")
#define SPMODS(P, OPC, TC)                                                                       \
    LOP(P##_none, v4i, v8i, TC, OPC " %0, %1, %2, %3")                                           \
    LOP(P##_negA, v4i, v8i, TC, OPC " %0, %1, %2, %3 neg_lo:[1,0,0]")                            \
    LOP(P##_neghiA, v4i, v8i, TC, OPC " %0, %1, %2, %3 neg_hi:[1,0,0]")                          \
    LOP(P##_negB, v4i, v8i, TC, OPC " %0, %1, %2, %3 neg_lo:[0,1,0]")                            \
    LOP(P##_neghiB, v4i, v8i, TC, OPC " %0, %1, %2, %3 neg_hi:[0,1,0]")
SPMODS(SF16, "v_swmmac_f32_16x16x32_f16", v8i)
SPMODS(SBF16, "v_swmmac_f32_16x16x32_bf16", v8i)
SPMODS(SF16H, "v_swmmac_f16_16x16x32_f16", v4i)
SPMODS(SBF16H, "v_swmmac_bf16_16x16x32_bf16", v4i)
LOP(SFF_none, v2i, v4i, v8i, "v_swmmac_f32_16x16x32_fp8_fp8 %0, %1, %2, %3")
LOP(SIU8_none, v2i, v4i, v8i, "v_swmmac_i32_16x16x32_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]")
LOP(IU8_sAB, v2i, v2i, v8i, "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]")
LOP(IU8_uu, v2i, v2i, v8i, "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0")
LOP(IU8_sA, v2i, v2i, v8i, "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[1,0,0]")
LOP(IU8_sB, v2i, v2i, v8i, "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %0 neg_lo:[0,1,0]")

// Raw encodings with fixed registers: NEG_HI on the integer forms (the assembler rejects it) and C modifiers on a
// sparse float form (the assembler rejects them, C being D there). Base words from llvm-mc for
// v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[10:11], v[0:7] neg_lo:[1,1,0] (and the iu4 forms, the sparse f16 form);
// neg_hi[i] is bit 8 + i of the first word, neg_lo[i] bit 29 + i of the second.
#define RAWOP(NAME, TA, TB, RA, RB, W0, W1)                                                                    \
    struct NAME {                                                                                              \
        typedef TA ta;                                                                                         \
        typedef TB tb;                                                                                         \
        typedef v8i tc;                                                                                        \
        static __device__ __forceinline__ void go(TA a, TB b, v8i & c, int) {                                  \
            asm volatile(".long " #W0 "\n.long " #W1 : "+{v[0:7]}"(c) : RA(a), RB(b));                         \
        }                                                                                                      \
    };
RAWOP(IU8_sAB_raw, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc444000, 0x7c021508)
RAWOP(IU8_sAB_nhA, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc444100, 0x7c021508)
RAWOP(IU8_sAB_nhB, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc444200, 0x7c021508)
RAWOP(IU8_uu_nhA, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc444100, 0x1c021508)
RAWOP(IU4_sAB_nhA, int, int, "{v8}", "{v9}", 0xcc454100, 0x7c021308)
RAWOP(IU4_uu_nhA, int, int, "{v8}", "{v9}", 0xcc454100, 0x1c021308)
RAWOP(IU4K32_sAB_nhA, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc4a4100, 0x7c021508)
#define RAWSOP(NAME, W0, W1)                                                                                    \
    struct NAME {                                                                                              \
        typedef v4i ta;                                                                                        \
        typedef v8i tb;                                                                                        \
        typedef v8i tc;                                                                                        \
        static __device__ __forceinline__ void go(v4i a, v8i b, v8i & c, int i) {                              \
            asm volatile(".long " #W0 "\n.long " #W1 : "+{v[0:7]}"(c) : "{v[8:11]}"(a), "{v[12:19]}"(b), "{v20}"(i)); \
        }                                                                                                      \
    };
RAWSOP(SF16_raw, 0xcc504000, 0x1c521908)
RAWSOP(SF16_negC_raw, 0xcc504000, 0x9c521908)
RAWSOP(SF16_absC_raw, 0xcc504400, 0x1c521908)
// Leftovers C (MA-1, MA-3): the same fields on more forms. Base words from llvm-mc for the plain instruction with
// these registers (v_swmmac_f32_16x16x32_bf16 0xcc514000 0x1c521908, _f16_16x16x32_f16 0xcc524000,
// _bf16_16x16x32_bf16 0xcc534000, _f32_16x16x32_fp8_fp8 0xcc574000, _bf8_bf8 0xcc5a4000, all with second word
// 0x1c521908 for v[0:x], v[8:..], v[12:..], v20; v_wmma_i32_16x16x16_iu4 v[0:7], v8, v9 0xcc454000 0x1c021308;
// v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[10:11] 0xcc4a4000 0x1c021508, signed 0x7c021508); NEG C is bit 63
// (bit 31 of the second word), ABS C bit 10, NEG_HI on A bit 8 and on B bit 9 of the first word, as in 9a.
RAWOP(IU8_uu_nhB, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc444200, 0x1c021508)
RAWOP(IU4_uu_raw, int, int, "{v8}", "{v9}", 0xcc454000, 0x1c021308)
RAWOP(IU4_sAB_nhB, int, int, "{v8}", "{v9}", 0xcc454200, 0x7c021308)
RAWOP(IU4_uu_nhB, int, int, "{v8}", "{v9}", 0xcc454200, 0x1c021308)
RAWOP(IU4K32_uu_raw, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc4a4000, 0x1c021508)
RAWOP(IU4K32_sAB_nhB, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc4a4200, 0x7c021508)
RAWOP(IU4K32_uu_nhA, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc4a4100, 0x1c021508)
RAWOP(IU4K32_uu_nhB, v2i, v2i, "{v[8:9]}", "{v[10:11]}", 0xcc4a4200, 0x1c021508)
#define RAWSOPX(NAME, TA, TC, RA, RC, W0, W1)                                                                   \
    struct NAME {                                                                                              \
        typedef TA ta;                                                                                         \
        typedef v8i tb;                                                                                        \
        typedef TC tc;                                                                                         \
        static __device__ __forceinline__ void go(TA a, v8i b, TC & c, int i) {                                \
            asm volatile(".long " #W0 "\n.long " #W1 : "+" RC(c) : RA(a), "{v[12:19]}"(b), "{v20}"(i));       \
        }                                                                                                      \
    };
RAWSOP(SBF16_raw, 0xcc514000, 0x1c521908)
RAWSOP(SBF16_negC_raw, 0xcc514000, 0x9c521908)
RAWSOP(SBF16_absC_raw, 0xcc514400, 0x1c521908)
RAWSOPX(SF16H_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc524000, 0x1c521908)
RAWSOPX(SF16H_negC_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc524000, 0x9c521908)
RAWSOPX(SF16H_absC_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc524400, 0x1c521908)
RAWSOPX(SBF16H_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc534000, 0x1c521908)
RAWSOPX(SBF16H_negC_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc534000, 0x9c521908)
RAWSOPX(SBF16H_absC_raw, v4i, v4i, "{v[8:11]}", "{v[0:3]}", 0xcc534400, 0x1c521908)
// FP8 sparse: B is 4 VGPRs; the 8 bound to v[12:19] cover them (the instruction reads v[12:15]).
RAWSOPX(SFF_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc574000, 0x1c521908)
RAWSOPX(SFF_negC_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc574000, 0x9c521908)
RAWSOPX(SFF_absC_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc574400, 0x1c521908)
RAWSOPX(SBB_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc5a4000, 0x1c521908)
RAWSOPX(SBB_negC_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc5a4000, 0x9c521908)
RAWSOPX(SBB_absC_raw, v2i, v8i, "{v[8:9]}", "{v[0:7]}", 0xcc5a4400, 0x1c521908)
LOP(SBB_none, v2i, v4i, v8i, "v_swmmac_f32_16x16x32_bf8_bf8 %0, %1, %2, %3")


template <class OP>
__global__ __launch_bounds__(32) void gk(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                        const uint32_t * __restrict__ C, const uint32_t * __restrict__ I,
                                        uint32_t * __restrict__ D) {
    typedef typename OP::ta TA;
    typedef typename OP::tb TB;
    typedef typename OP::tc TC;
    constexpr int NA = sizeof(TA) / 4, NB = sizeof(TB) / 4, NC = sizeof(TC) / 4;
    const size_t w = blockIdx.x, lane = threadIdx.x;
    TA a;
    TB b;
    TC c;
    __builtin_memcpy(&a, A + (w * 32 + lane) * NA, sizeof a);
    __builtin_memcpy(&b, B + (w * 32 + lane) * NB, sizeof b);
    __builtin_memcpy(&c, C + (w * 32 + lane) * NC, sizeof c);
    const int idx = (int) I[w * 32 + lane];
    OP::go(a, b, c, idx);
    __builtin_memcpy(D + (w * 32 + lane) * NC, &c, sizeof c);
}

typedef void (*GK)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *);
struct OpInfo {
    GK k;
    int na, nb, nc;
};
static std::map<std::string, OpInfo> g_ops;
static void register_hz();
#define REG(NAME) g_ops[#NAME] = OpInfo{gk<NAME>, (int) sizeof(NAME::ta) / 4, (int) sizeof(NAME::tb) / 4, (int) sizeof(NAME::tc) / 4};
#define REGS4(S) REG(S32k0_##S) REG(S32k1_##S) REG(S64_##S) REG(D16_##S) REG(D32_##S)
static void register_ops() {
    REGS4(0) REGS4(1) REGS4(2) REGS4(3) REGS4(4) REGS4(5) REGS4(6) REGS4(7)
    REG(FB_none) REG(FB_negC) REG(FB_absC) REG(FB_nabsC) REG(BF_none) REG(BF_negC) REG(BF_absC) REG(BF_nabsC)
    REG(BB_none) REG(BB_negC) REG(BB_absC) REG(BB_nabsC) REG(FF_none)
    REG(SF16_none) REG(SF16_negA) REG(SF16_neghiA) REG(SF16_negB) REG(SF16_neghiB)
    REG(SBF16_none) REG(SBF16_negA) REG(SBF16_neghiA) REG(SBF16_negB) REG(SBF16_neghiB)
    REG(SF16H_none) REG(SF16H_negA) REG(SF16H_neghiA) REG(SF16H_negB) REG(SF16H_neghiB)
    REG(SBF16H_none) REG(SBF16H_negA) REG(SBF16H_neghiA) REG(SBF16H_negB) REG(SBF16H_neghiB)
    REG(SFF_none) REG(SIU8_none) REG(IU8_sAB) REG(IU8_uu) REG(IU8_sA) REG(IU8_sB)
    REG(IU8_sAB_raw) REG(IU8_sAB_nhA) REG(IU8_sAB_nhB) REG(IU8_uu_nhA) REG(IU4_sAB_nhA) REG(IU4_uu_nhA) REG(IU4K32_sAB_nhA)
    REG(SF16_raw) REG(SF16_negC_raw) REG(SF16_absC_raw)
    REG(IU8_uu_nhB) REG(IU4_uu_raw) REG(IU4_sAB_nhB) REG(IU4_uu_nhB) REG(IU4K32_uu_raw) REG(IU4K32_sAB_nhB)
    REG(IU4K32_uu_nhA) REG(IU4K32_uu_nhB) REG(SBF16_raw) REG(SBF16_negC_raw) REG(SBF16_absC_raw) REG(SF16H_raw)
    REG(SF16H_negC_raw) REG(SF16H_absC_raw) REG(SBF16H_raw) REG(SBF16H_negC_raw) REG(SBF16H_absC_raw) REG(SFF_raw)
    REG(SFF_negC_raw) REG(SFF_absC_raw) REG(SBB_raw) REG(SBB_negC_raw) REG(SBB_absC_raw) REG(SBB_none)
    register_hz();
}

// Register images of one launch: per wave and lane, na A words, nb B words, nc C/D words, one index word.
struct Img {
    std::string op;
    int waves = 0, na = 0, nb = 0, nc = 0;
    std::vector<uint32_t> a, b, c, i, d;
    void init(const std::string & name, int n) {
        const auto it = g_ops.find(name);
        if (it == g_ops.end()) {
            std::fprintf(stderr, "no op %s\n", name.c_str());
            std::exit(2);
        }
        op = name;
        waves = n;
        na = it->second.na;
        nb = it->second.nb;
        nc = it->second.nc;
        a.assign((size_t) n * 32 * na, 0);
        b.assign((size_t) n * 32 * nb, 0);
        c.assign((size_t) n * 32 * nc, 0);
        i.assign((size_t) n * 32, 0);
    }
    uint32_t & aw(int w, int lane, int k) { return a[((size_t) w * 32 + lane) * na + k]; }
    uint32_t & bw(int w, int lane, int k) { return b[((size_t) w * 32 + lane) * nb + k]; }
    uint32_t & cw(int w, int lane, int k) { return c[((size_t) w * 32 + lane) * nc + k]; }
    uint32_t dw(int w, int lane, int k) const { return d[((size_t) w * 32 + lane) * nc + k]; }
};

static void launch(Img & m) {
    uint32_t *da, *db, *dc, *di, *dd;
    CHECK(hipMalloc(&da, m.a.size() * 4));
    CHECK(hipMalloc(&db, m.b.size() * 4));
    CHECK(hipMalloc(&dc, m.c.size() * 4));
    CHECK(hipMalloc(&di, m.i.size() * 4));
    CHECK(hipMalloc(&dd, m.c.size() * 4));
    CHECK(hipMemcpy(da, m.a.data(), m.a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, m.b.data(), m.b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, m.c.data(), m.c.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(di, m.i.data(), m.i.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dd, 0xEE, m.c.size() * 4));
    hipLaunchKernelGGL(g_ops[m.op].k, dim3(m.waves), dim3(32), 0, 0, da, db, dc, di, dd);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    m.d.resize(m.c.size());
    CHECK(hipMemcpy(m.d.data(), dd, m.d.size() * 4, hipMemcpyDeviceToHost));
    for (auto * p : {da, db, dc, di, dd}) {
        CHECK(hipFree(p));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Item 1: 4-bit forms. A register position of a nibble operand with nw VGPRs per lane, for row (column) r:
// position p = h * 8 nw + v * 8 + n means lane r + 16 h, VGPR v, nibble n ("natural order").
struct NPos {
    int lane, word, bit;
};
static NPos npos(int nw, int r, int p) {
    const int per = 8 * nw;
    return {r + 16 * (p / per), (p % per) / 8, (p % 8) * 4};
}
static void put_nib(std::vector<uint32_t> & img, int nw, int w, int r, int p, uint32_t v) {
    const NPos q = npos(nw, r, p);
    uint32_t & x = img[((size_t) w * 32 + q.lane) * nw + q.word];
    x = (x & ~(0xFu << q.bit)) | ((v & 0xF) << q.bit);
}
static uint32_t get_nib(const std::vector<uint32_t> & img, int nw, int w, int r, int p) {
    const NPos q = npos(nw, r, p);
    return (img[((size_t) w * 32 + q.lane) * nw + q.word] >> q.bit) & 0xF;
}
// C / D element (r, c): lane 16 (r >> 3) + c, VGPR r & 7 (as the dense forms).
static int cd_lane(int r, int c) { return 16 * (r >> 3) + c; }

// The 4-bit kinds: sparse K = 32 (key 0, 1), sparse K = 64, dense K = 16, dense K = 32.
enum S4Kind { K_S32 = 0, K_S64, K_D16, K_D32 };
static std::string s4op(int kind, int key, int set) {
    const char * base[] = {"S32k", "S64_", "D16_", "D32_"};
    std::string s = base[kind];
    if (kind == K_S32) {
        s += std::to_string(key) + "_";
    }
    return s + std::to_string(set);
}
static int s4_nstored(int kind) { return kind == K_S32 ? 16 : kind == K_S64 ? 32 : kind == K_D16 ? 16 : 32; }
static int s4_nslots(int kind) { return kind == K_S32 ? 32 : kind == K_S64 ? 64 : kind == K_D16 ? 16 : 32; }

// The rule (found by s4map, held by s4): stored A position j (natural order) meets B position (natural order)
//   K = 32 sparse: 4 g + field, g = ((j & 7) >> 1) + 4 (j >> 3), field = bits 2 ((j & 7) + 8 key) of the index word
//                  of lane r + 16 (j >> 3) (the 16-bit sparse rule, arith_model.h sparse_slot);
//   K = 64 sparse: 4 g + field, g = ((j & 15) >> 1) + 8 (j >> 4), field = bits 2 (j & 15) of the index word of
//                  lane r + 16 (j >> 4);
//   dense: B position j.
// --break uses the other half of the index word (K = 32: the other key; K = 64: the high and low halves swapped).
static int s4_rule(int kind, int key, int j, uint32_t idx_lo, uint32_t idx_hi, bool brk) {
    if (kind == K_D16 || kind == K_D32) {
        return j;
    }
    if (kind == K_S32) {
        return am::sparse_slot(j, j < 8 ? idx_lo : idx_hi, brk ? key ^ 1 : key);
    }
    const int g = ((j & 15) >> 1) + 8 * (j >> 4);
    const uint32_t w = j < 16 ? idx_lo : idx_hi;
    const int f = (j & 15) ^ (brk ? 8 : 0);
    return 4 * g + (int) ((w >> (2 * f)) & 3);
}

// One-hot experiment: A row r holds 1 at A position q(r) = r + qoff, all else 0; every B column holds digit
// encodings of its positions (wave 2s: (p & 7) + 1, wave 2s + 1: (p >> 3) + 1), unsigned; index words per lane
// given. Returns for each index set s and row r the B position met (-1: zero, -2: inconsistent over columns).
static std::vector<std::vector<int>> s4_onehot(int kind, int key, int qoff, const std::vector<std::vector<uint32_t>> & idx) {
    const int nsets = (int) idx.size();
    Img m;
    m.init(s4op(kind, key, 0), nsets * 2);
    const int nwa = m.na, nwb = m.nb;
    for (int s = 0; s < nsets; ++s) {
        for (int dg = 0; dg < 2; ++dg) {
            const int w = s * 2 + dg;
            for (int r = 0; r < 16; ++r) {
                put_nib(m.a, nwa, w, r, r + qoff, 1);
                for (int p = 0; p < s4_nslots(kind); ++p) {
                    put_nib(m.b, nwb, w, r, p, (uint32_t) (dg ? (p >> 3) : (p & 7)) + 1);
                }
            }
            for (int l = 0; l < 32; ++l) {
                m.i[(size_t) w * 32 + l] = idx[s][l];
            }
        }
    }
    launch(m);
    std::vector<std::vector<int>> out(nsets, std::vector<int>(16));
    for (int s = 0; s < nsets; ++s) {
        for (int r = 0; r < 16; ++r) {
            int res = -9;
            for (int c = 0; c < 16 && res != -2; ++c) {
                const uint32_t lo = m.dw(s * 2, cd_lane(r, c), r & 7), hi = m.dw(s * 2 + 1, cd_lane(r, c), r & 7);
                const int v = (lo == 0 && hi == 0) ? -1 : (lo >= 1 && lo <= 8 && hi >= 1 && hi <= 8) ? (int) (hi - 1) * 8 + (int) (lo - 1) : -2;
                res = (res == -9 || res == v) ? v : -2;
            }
            out[s][r] = res;
        }
    }
    return out;
}

static bool mode_s4map(std::string & text) {
    bool ok = true;
    std::mt19937_64 rng(4040);
    std::vector<std::vector<uint32_t>> idx;
    for (uint32_t u : {0u, 0xFFFFFFFFu, 0x55555555u, 0xAAAAAAAAu, 0xE4E4E4E4u, 0x1B1B1B1Bu}) {
        idx.push_back(std::vector<uint32_t>(32, u));
    }
    for (int s = 0; s < 58; ++s) {
        std::vector<uint32_t> v(32);
        for (auto & x : v) {
            x = (uint32_t) rng();  // different in every lane, so the lane that supplies the field shows
        }
        idx.push_back(v);
    }
    struct K {
        int kind, key;
        const char * name;
    };
    const K ks[] = {{K_D16, 0, "dense iu4 K=16"}, {K_D32, 0, "dense iu4 K=32"}, {K_S32, 0, "swmmac iu4 K=32 key 0"},
                    {K_S32, 1, "swmmac iu4 K=32 key 1"}, {K_S64, 0, "swmmac iu4 K=64"}};
    for (const K & k : ks) {
        const int nst = s4_nstored(k.kind);
        long n = 0, bad = 0, zero = 0, incons = 0;
        for (int qoff = 0; qoff < nst; qoff += 16) {
            const auto out = s4_onehot(k.kind, k.key, qoff, idx);
            for (int r = 0; r < 16; ++r) {
                const int j = r + qoff;
                // fit: which (lane, 2-bit field) decides the B position of A position j, and its table
                std::string fits;
                for (int ln = 0; ln < 2; ++ln) {
                    for (int f = 0; f < 16; ++f) {
                        int tab[4] = {-9, -9, -9, -9};
                        bool good = true;
                        for (size_t s = 0; s < idx.size() && good; ++s) {
                            const int fv = (int) ((idx[s][r + 16 * ln] >> (2 * f)) & 3);
                            if (tab[fv] == -9) {
                                tab[fv] = out[s][r];
                            }
                            good = tab[fv] == out[s][r] && out[s][r] >= 0;
                        }
                        const bool flat = tab[0] == tab[1] && tab[1] == tab[2] && tab[2] == tab[3];
                        if (good && flat) {
                            // no index field matters (dense forms): one entry says so
                            char buf[96];
                            std::snprintf(buf, sizeof buf, " %d for every index value", tab[0]);
                            fits = buf;
                            ln = 2;
                            break;
                        }
                        if (good) {
                            char buf[96];
                            std::snprintf(buf, sizeof buf, " [lane r+%d bits %d-%d: %d %d %d %d]", 16 * ln, 2 * f, 2 * f + 1,
                                          tab[0], tab[1], tab[2], tab[3]);
                            fits += buf;
                        }
                    }
                }
                std::printf("RESULT s4map %-22s A position %2d (lane r+%d VGPR %d nibble %d) -> B position%s%s\n", k.name, j,
                            npos(k.kind == K_S32 || k.kind == K_D16 ? 1 : 2, 0, j).lane,
                            npos(k.kind == K_S32 || k.kind == K_D16 ? 1 : 2, 0, j).word, j % 8,
                            fits.empty() ? " (no single field fits) first:" : "", fits.c_str());
                if (fits.empty()) {
                    for (int s = 0; s < 4; ++s) {
                        std::printf("   index set %d -> %d\n", s, out[s][r]);
                    }
                }
                for (size_t s = 0; s < idx.size(); ++s) {
                    const int want = s4_rule(k.kind, k.key, j, idx[s][r], idx[s][r + 16], g_break);
                    ++n;
                    bad += out[s][r] != want;
                    zero += out[s][r] == -1;
                    incons += out[s][r] == -2;
                }
            }
        }
        std::printf("RESULT s4map %-22s observations %ld as the rule %ld (zero %ld, inconsistent over columns %ld)\n", k.name,
                    n, n - bad, zero, incons);
        ok = ok && bad == 0;
        add_text(text, "%s %ld/%ld", k.name, n - bad, n);
    }
    return ok;
}

// Random 4-bit problems in register images; the model reads them back through the same natural positions.
struct S4Run {
    std::vector<uint32_t> d;
};

static uint32_t s4_model(int kind, int key, int set, const Img & m, int w, int r, int c, int clamp_mode, int group, int keyuse) {
    const int nst = s4_nstored(kind);
    uint32_t a[32], b[32];
    for (int j = 0; j < nst; ++j) {
        a[j] = get_nib(m.a, m.na, w, r, j);
        const int p = s4_rule(kind, keyuse, j, m.i[(size_t) w * 32 + r], m.i[(size_t) w * 32 + r + 16], false);
        b[j] = get_nib(m.b, m.nb, w, c, p);
    }
    (void) key;
    return am::model_int(a, b, nst, 4, set_sa(set), set_sb(set), m.c[((size_t) w * 32 + cd_lane(r, c)) * 8 + (r & 7)], clamp_mode, group);
}

// kind of data: 0 uniform nibbles and C; 1 extremes (C near the 32-bit limits); 2 C near a limit and long runs of
// large products of one sign (separates saturation per step from saturation at the end).
static void s4_fill(Img & m, int kind, int w, int dk, std::mt19937_64 & rng) {
    const int nst = s4_nstored(kind), nsl = s4_nslots(kind);
    for (int r = 0; r < 16; ++r) {
        const int s0 = (int) (rng() % nst), len = 1 + (int) (rng() % nst);
        for (int j = 0; j < nst; ++j) {
            uint32_t v = (uint32_t) rng() & 0xF;
            if (dk == 1) {
                const uint32_t ext[] = {7, 8, 15, 0, 1};
                v = ext[rng() % 5];
            } else if (dk == 2) {
                v = ((j - s0 + nst) % nst) < len ? 7 : 8;
            }
            put_nib(m.a, m.na, w, r, j, v);
        }
        for (int p = 0; p < nsl; ++p) {
            uint32_t v = (uint32_t) rng() & 0xF;
            if (dk == 1) {
                const uint32_t ext[] = {7, 8, 15, 0, 1};
                v = ext[rng() % 5];
            } else if (dk == 2) {
                v = (rng() & 3) ? 7 : v;
            }
            put_nib(m.b, m.nb, w, r, p, v);
        }
    }
    for (int l = 0; l < 32; ++l) {
        m.i[(size_t) w * 32 + l] = (uint32_t) rng();
        for (int k = 0; k < 8; ++k) {
            uint32_t cv = (uint32_t) rng();
            if (dk == 1) {
                const uint32_t big[] = {0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x00000000u};
                cv = big[rng() % 4] + (uint32_t) ((int) (rng() % 4001) - 2000);
            } else if (dk == 2) {
                const int64_t lim = (rng() & 1) ? INT32_MAX : INT32_MIN;
                cv = (uint32_t) (int32_t) (lim - (lim > 0 ? 1 : -1) * (int64_t) (rng() % 4096));
            }
            m.cw(w, l, k) = cv;
        }
    }
}

static bool mode_s4(long n, std::string & text) {
    bool ok = true;
    long ctl_end = 0, ctl_key = 0, ctl_grp = 0, ctl_n = 0;
    struct K {
        int kind, key;
        const char * name;
    };
    const K ks[] = {{K_S32, 0, "swmmac_iu4_k32 key 0"}, {K_S32, 1, "swmmac_iu4_k32 key 1"}, {K_S64, 0, "swmmac_iu4_k64"}};
    for (const K & k : ks) {
        const int nst = s4_nstored(k.kind);
        for (int set = 0; set < 8; ++set) {
            const bool clamp = set >= 4;
            const int nw = (int) ((n + 255) / 256);
            std::mt19937_64 rng(5000 + k.kind * 64 + k.key * 16 + set);
            Img m;
            m.init(s4op(k.kind, k.key, set), nw);
            for (int w = 0; w < nw; ++w) {
                s4_fill(m, k.kind, w, w % 3, rng);
            }
            launch(m);
            // measured: clamp saturates the running sum after each half of the stored elements (one lane half)
            const int cm = clamp ? (g_break ? 1 : 2) : 0, grp = nst / 2;
            std::vector<long> bad(8, 0), cend(8, 0), ckey(8, 0), cgrp(8, 0);
            std::vector<std::string> first(8);
            par_for((size_t) nw, [&](size_t w, int t) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        const uint32_t got = m.dw((int) w, cd_lane(r, c), r & 7);
                        const uint32_t want = s4_model(k.kind, k.key, set, m, (int) w, r, c, cm, grp, k.key);
                        if (got != want) {
                            if (!bad[t]) {
                                char buf[128];
                                std::snprintf(buf, sizeof buf, "wave %zu (%d,%d): card %08x model %08x", w, r, c, got, want);
                                first[t] = buf;
                            }
                            ++bad[t];
                        }
                        if (!g_break) {
                            cend[t] += clamp && got != s4_model(k.kind, k.key, set, m, (int) w, r, c, 1, grp, k.key);
                            cgrp[t] += clamp && got != s4_model(k.kind, k.key, set, m, (int) w, r, c, 2, grp / 2, k.key);
                            if (k.kind == K_S32) {
                                ckey[t] += got != s4_model(k.kind, k.key, set, m, (int) w, r, c, cm, grp, k.key ^ 1);
                            } else {
                                // K = 64 control: the index halves swapped
                                const int nsl = nst;
                                uint32_t a[32], b[32];
                                for (int j = 0; j < nsl; ++j) {
                                    a[j] = get_nib(m.a, m.na, (int) w, r, j);
                                    b[j] = get_nib(m.b, m.nb, (int) w, c,
                                                   s4_rule(k.kind, 0, j, m.i[w * 32 + r], m.i[w * 32 + r + 16], true));
                                }
                                ckey[t] += got != am::model_int(a, b, nsl, 4, set_sa(set), set_sb(set),
                                                                m.c[((size_t) w * 32 + cd_lane(r, c)) * 8 + (r & 7)], cm, grp);
                            }
                        }
                    }
                }
            });
            long nb = 0, ne = 0, nk = 0, ng = 0;
            std::string f1;
            for (int t = 0; t < 8; ++t) {
                nb += bad[t];
                ne += cend[t];
                nk += ckey[t];
                ng += cgrp[t];
                if (f1.empty()) {
                    f1 = first[t];
                }
            }
            long sat = 0;
            for (uint32_t v : m.d) {
                sat += v == 0x7FFFFFFFu || v == 0x80000000u;
            }
            const long ne_ = (long) nw * 256;
            std::printf("RESULT s4 %-22s %-9s elements %ld model_mismatches %ld outputs_at_a_32bit_limit %ld | controls: "
                        "saturated once at the end %ld, saturated per quarter %ld, other index half %ld%s%s\n",
                        k.name, s4set_name[set], ne_, nb, sat, ne, ng, nk, nb ? " first: " : "", f1.c_str());
            ok = ok && nb == 0;
            if (!g_break) {
                ok = ok && nk > 0;
                ctl_key += nk;
            }
            if (clamp) {
                ctl_end += ne;
                ctl_grp += ng;
                ctl_n += ne_;
            }
            add_text(text, "%s %s %ld/%ld", k.name, s4set_name[set], ne_ - nb, ne_);
        }
    }
    if (!g_break) {
        std::printf("RESULT s4 controls over all clamp settings: saturated once at the end %ld, per quarter %ld of %ld\n", ctl_end,
                    ctl_grp, ctl_n);
        ok = ok && ctl_end > 0 && ctl_grp > 0;
    }
    // the same 32 products through the dense K = 32 form and the sparse K = 64 form (valid index, the unselected B
    // nibbles unrelated): card against card, signed and clamped
    {
        const int nw = (int) ((n / 4 + 255) / 256);
        std::mt19937_64 rng(5999);
        Img sp, de;
        sp.init(s4op(K_S64, 0, 7), nw);
        de.init(s4op(K_D32, 0, 7), nw);
        for (int w = 0; w < nw; ++w) {
            s4_fill(sp, K_S64, w, w % 3, rng);
            // a valid index shared by all rows: the two stored elements of each pair name different positions
            uint32_t half[2];
            for (int h = 0; h < 2; ++h) {
                half[h] = 0;
                for (int pr = 0; pr < 8; ++pr) {
                    const uint32_t x = (uint32_t) (rng() % 4), y = (x + 1 + (uint32_t) (rng() % 3)) % 4;
                    half[h] |= (x | y << 2) << (4 * pr);
                }
            }
            for (int l = 0; l < 32; ++l) {
                sp.i[(size_t) w * 32 + l] = half[l >> 4];
            }
            for (int r = 0; r < 16; ++r) {
                for (int j = 0; j < 32; ++j) {
                    put_nib(de.a, de.na, w, r, j, get_nib(sp.a, sp.na, w, r, j));
                    put_nib(de.b, de.nb, w, r, j, get_nib(sp.b, sp.nb, w, r, s4_rule(K_S64, 0, j, half[0], half[1], false)));
                }
            }
            for (int l = 0; l < 32; ++l) {
                for (int kk = 0; kk < 8; ++kk) {
                    de.cw(w, l, kk) = sp.cw(w, l, kk);
                }
            }
        }
        launch(sp);
        launch(de);
        long diff = 0;
        for (size_t x = 0; x < sp.d.size(); ++x) {
            diff += sp.d[x] != de.d[x];
        }
        std::printf("RESULT s4 dense_k32_vs_sparse_k64 (sAB clamp, valid index) elements %ld differ %ld\n", (long) nw * 256, diff);
        ok = ok && diff == 0;
        add_text(text, "dense K=32 = sparse K=64 %ld/%ld", (long) nw * 256 - diff, (long) nw * 256);
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 2: the lane mask. EXEC is set to `mask` around one instruction; D starts as a sentinel image (sparse forms:
// D is C, so C itself is the sentinel); everything is stored under full EXEC afterwards.
#define MOP(NAME, TA, TB, TC, TEXT)                                                                               \
    struct NAME {                                                                                                 \
        typedef TA ta;                                                                                            \
        typedef TB tb;                                                                                            \
        typedef TC tc;                                                                                            \
        static __device__ __forceinline__ void go(TA a, TB b, TC c, TC & d, int i, uint32_t m) {                   \
            asm volatile("s_mov_b32 s96, exec_lo\n\ts_mov_b32 exec_lo, %5\n\t" TEXT "\n\ts_mov_b32 exec_lo, s96"    \
                         : "+v"(d) : "v"(a), "v"(b), "v"(c), "v"(i), "s"(m) : "s96");                              \
        }                                                                                                         \
    };
MOP(M_f16, v4i, v4i, v8i, "v_wmma_f32_16x16x16_f16 %0, %1, %2, %3")
MOP(M_fp8, v2i, v2i, v8i, "v_wmma_f32_16x16x16_fp8_fp8 %0, %1, %2, %3")
MOP(M_iu8, v2i, v2i, v8i, "v_wmma_i32_16x16x16_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]")
MOP(M_f16h, v4i, v4i, v4i, "v_wmma_f16_16x16x16_f16 %0, %1, %2, %3")
MOP(M_sf16, v4i, v8i, v8i, "v_swmmac_f32_16x16x32_f16 %0, %1, %2, %4")
MOP(M_dot4, int, int, int, "v_dot4_f32_fp8_fp8 %0, %1, %2, %3")
MOP(M_dot2, int, int, int, "v_dot2_f32_f16 %0, %1, %2, %3")

template <class OP>
__global__ __launch_bounds__(32) void mk(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                        const uint32_t * __restrict__ C, const uint32_t * __restrict__ I,
                                        uint32_t * __restrict__ D, uint32_t mask) {
    typedef typename OP::ta TA;
    typedef typename OP::tb TB;
    typedef typename OP::tc TC;
    const size_t lane = threadIdx.x;
    TA a;
    TB b;
    TC c, d;
    __builtin_memcpy(&a, A + lane * (sizeof(TA) / 4), sizeof a);
    __builtin_memcpy(&b, B + lane * (sizeof(TB) / 4), sizeof b);
    __builtin_memcpy(&c, C + lane * (sizeof(TC) / 4), sizeof c);
    __builtin_memcpy(&d, D + lane * (sizeof(TC) / 4), sizeof d);
    OP::go(a, b, c, d, (int) I[lane], mask);
    __builtin_memcpy(D + lane * (sizeof(TC) / 4), &d, sizeof d);
}
typedef void (*MK)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *, uint32_t);

struct MaskOp {
    const char * name;
    MK k;
    int na, nb, nc;
    bool sparse, dot;
};
#define MREG(NAME, SP, DOT) MaskOp{#NAME, mk<NAME>, (int) sizeof(NAME::ta) / 4, (int) sizeof(NAME::tb) / 4, (int) sizeof(NAME::tc) / 4, SP, DOT}

static std::vector<uint32_t> run_mask(const MaskOp & o, const std::vector<uint32_t> & a, const std::vector<uint32_t> & b,
                                      const std::vector<uint32_t> & c, const std::vector<uint32_t> & idx,
                                      const std::vector<uint32_t> & d0, uint32_t mask) {
    uint32_t *da, *db, *dc, *di, *dd;
    CHECK(hipMalloc(&da, a.size() * 4));
    CHECK(hipMalloc(&db, b.size() * 4));
    CHECK(hipMalloc(&dc, c.size() * 4));
    CHECK(hipMalloc(&di, idx.size() * 4));
    CHECK(hipMalloc(&dd, d0.size() * 4));
    CHECK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, c.data(), c.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(di, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dd, d0.data(), d0.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(o.k, dim3(1), dim3(32), 0, 0, da, db, dc, di, dd, mask);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(d0.size());
    CHECK(hipMemcpy(out.data(), dd, out.size() * 4, hipMemcpyDeviceToHost));
    for (auto * p : {da, db, dc, di, dd}) {
        CHECK(hipFree(p));
    }
    return out;
}

static bool mode_mask(std::string & text) {
    bool ok = true;
    const MaskOp ops[] = {MREG(M_f16, false, false), MREG(M_fp8, false, false), MREG(M_iu8, false, false),
                          MREG(M_f16h, false, false), MREG(M_sf16, true, false), MREG(M_dot4, false, true),
                          MREG(M_dot2, false, true)};
    const uint32_t masks[] = {0xFFFFFFFFu, 0x0000FFFFu, 0xFFFF0000u, 0x55555555u, 0x11111111u, 0x00000001u, 0u};
    std::mt19937_64 rng(2222);
    for (const MaskOp & o : ops) {
        const bool f8 = std::strstr(o.name, "fp8") || std::strstr(o.name, "dot4");
        const bool i8 = std::strstr(o.name, "iu8");
        auto rnd = [&]() -> uint32_t {
            if (i8) {
                return (uint32_t) rng() & 0x3F3F3F3Fu;  // small signed bytes, nonzero sums
            }
            if (f8) {
                uint32_t v = 0;
                for (int k = 0; k < 4; ++k) {
                    v |= (uint32_t) (0x30 + (rng() % 16)) << (8 * k);  // E4M3 values 0.5 .. 1.9
                }
                return v;
            }
            return (uint32_t) (0x3800 + (rng() % 0x800)) | (uint32_t) (0x3800 + (rng() % 0x800)) << 16;  // FP16 0.5 .. 2
        };
        std::vector<uint32_t> a(32 * o.na), b(32 * o.nb), c(32 * o.nc), idx(32), d0(32 * o.nc);
        for (auto & x : a) x = rnd();
        for (auto & x : b) x = rnd();
        for (auto & x : idx) x = (uint32_t) rng();
        for (size_t k = 0; k < c.size(); ++k) {
            // C small (0.25 .. 0.5, or small integers; FP16 pairs for the 16-bit accumulator)
            c[k] = i8 ? (uint32_t) (rng() % 100) : std::strstr(o.name, "f16h") ? 0x34003400u : 0x3E800000u + (uint32_t) (rng() & 0x7FFFFF);
            d0[k] = 0xDEAD0000u | (uint32_t) k;  // sentinel
        }
        if (o.sparse) {
            d0 = c;  // C is D: the untouched C is the sentinel
        }
        const std::vector<uint32_t> ref = run_mask(o, a, b, c, idx, d0, 0xFFFFFFFFu);
        long ref_unwritten = 0;
        for (size_t k = 0; k < ref.size(); ++k) {
            ref_unwritten += ref[k] == d0[k];
        }
        for (uint32_t m : masks) {
            const std::vector<uint32_t> out = run_mask(o, a, b, c, idx, d0, m);
            const std::vector<uint32_t> out2 = run_mask(o, a, b, c, idx, d0, m);
            long eq_ref = 0, off = 0, off_written = 0, on = 0, on_right = 0, lost = 0;
            for (int l = 0; l < 32; ++l) {
                const bool en = (m >> l) & 1;
                for (int k = 0; k < o.nc; ++k) {
                    const size_t x = (size_t) l * o.nc + k;
                    eq_ref += out[x] == ref[x];
                    lost += out[x] == 0xEEEEEEEEu;
                    if (en) {
                        ++on;
                        on_right += out[x] == ref[x];
                    } else {
                        ++off;
                        off_written += out[x] != d0[x];
                    }
                }
            }
            const bool repeat = out == out2;
            std::printf("RESULT mask %-7s EXEC %08x words %zu equal_to_full_EXEC %ld | enabled lanes' words %ld right %ld | "
                        "disabled lanes' words %ld written %ld | repeat_identical %d | full-EXEC words equal to the sentinel %ld\n",
                        o.name, m, out.size(), eq_ref, on, on_right, off, off_written, (int) repeat, ref_unwritten);
            // matrix forms (the claim under test): every word equals the full-EXEC result; dot instructions (the
            // control): enabled lanes right, disabled lanes untouched. --break expects the dot to ignore EXEC too.
            const bool ignore = !o.dot || g_break;
            const bool good = ignore ? (eq_ref == (long) out.size() && off_written == off) : (on_right == on && off_written == 0);
            ok = ok && good && repeat && lost == 0 && ref_unwritten == 0;
        }
        add_text(text, "%s %s", o.name, o.dot ? "obeys EXEC" : "ignores EXEC");
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 3: modifiers. A transform changes sign bits of one operand in memory; the card with the modifier on the
// original data is compared with the card without it on the transformed data (and on the original data, to see
// whether the bit is ignored).
enum Xf { X_NONE = 0, X_A_EVEN, X_A_ODD, X_B_EVEN, X_B_ODD, X_C_NEG, X_C_ABS, X_C_NEGABS, X_B_ALL, X_B_V03, X_B_V47,
          X_B_LANELO, X_B_LANEHI, X_B_EVENPAIR, X_B_ODDPAIR, NXF };
static const char * const xf_name[NXF] = {"nothing changed",           "A sign flipped, even slots", "A sign flipped, odd slots",
                                          "B sign flipped, even slots", "B sign flipped, odd slots",  "C sign flipped",
                                          "C sign cleared",             "C sign set",                 "B sign flipped, all slots",
                                          "B flipped, VGPRs 0-3",       "B flipped, VGPRs 4-7",       "B flipped, slots 0-15",
                                          "B flipped, slots 16-31",     "B flipped, slots 4g, 4g+1",  "B flipped, slots 4g+2, 4g+3"};

// One problem for a dense or sparse float form: A 16 stored slots per row, B up to 32 slots per column, C bits.
struct FP {
    uint32_t a[16][16], b[16][32], c[16][16], idx[32];
};

static void apply(FP & p, int x, int ebits, int cbits, int nb) {
    const uint32_t sa = 1u << (ebits - 1), sc = 1u << (cbits - 1);
    for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 16; ++s) {
            if ((x == X_A_EVEN && !(s & 1)) || (x == X_A_ODD && (s & 1))) {
                p.a[r][s] ^= sa;
            }
        }
        for (int t = 0; t < nb; ++t) {
            const bool v03 = (t & 15) < 8;
            if ((x == X_B_EVEN && !(t & 1)) || (x == X_B_ODD && (t & 1)) || x == X_B_ALL || (x == X_B_V03 && v03) ||
                (x == X_B_V47 && !v03) || (x == X_B_LANELO && t < 16) || (x == X_B_LANEHI && t >= 16) ||
                (x == X_B_EVENPAIR && !(t & 2)) || (x == X_B_ODDPAIR && (t & 2))) {
                p.b[r][t] ^= sa;
            }
        }
        for (int c = 0; c < 16; ++c) {
            if (x == X_C_NEG) {
                p.c[r][c] ^= sc;
            } else if (x == X_C_ABS) {
                p.c[r][c] &= ~sc;
            } else if (x == X_C_NEGABS) {
                p.c[r][c] |= sc;
            }
        }
    }
}

// Special values of a format: zeros, smallest subnormals, NaNs with payloads, infinities (E4M3: its two NaN codes).
static std::vector<uint32_t> specials(am::Fmt f) {
    const am::FmtInfo fi = am::info(f);
    const uint32_t sb = 1u << (fi.ebits + fi.mbits), emax = (1u << fi.ebits) - 1;
    std::vector<uint32_t> v = {0, sb, 1, sb | 1};
    if (fi.e4m3) {
        v.insert(v.end(), {0x7Fu, 0xFFu});
    } else {
        v.insert(v.end(), {emax << fi.mbits, sb | (emax << fi.mbits), (emax << fi.mbits) | 3, sb | (emax << fi.mbits) | (1u << (fi.mbits - 1))});
    }
    return v;
}

// kind: the dense form (am::Form numbering) of the inputs; out16: 0 FP32 accumulator, else the 16-bit format.
static FP make_fp(int form, am::Fmt out16, bool special, std::mt19937_64 & rng) {
    FP p;
    const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
    int lo, hi;
    ad::wide_range(fa, lo, hi);
    const int l2 = std::max(lo, -8), h2 = std::min(hi, 8);
    const std::vector<uint32_t> sa = specials(fa), sb = specials(fb);
    const std::vector<uint32_t> sc32 = {0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC12345u, 0xFF800001u, 1u, 0x80000001u};
    const std::vector<uint32_t> sc16 = out16 == am::F16 ? std::vector<uint32_t>{0u, 0x8000u, 0x7C00u, 0xFC00u, 0x7E12u, 0xFC01u, 1u, 0x8001u}
                                                         : std::vector<uint32_t>{0u, 0x8000u, 0x7F80u, 0xFF80u, 0x7FC5u, 0xFF81u, 1u, 0x8001u};
    for (int r = 0; r < 16; ++r) {
        for (int s = 0; s < 16; ++s) {
            p.a[r][s] = ad::rand_val(fa, rng, l2, h2);
        }
        for (int t = 0; t < 32; ++t) {
            p.b[r][t] = ad::rand_val(fb, rng, l2, h2);
        }
        for (int c = 0; c < 16; ++c) {
            p.c[r][c] = out16 == am::F32 ? ad::rand_f32(rng, -4, 8) : ad::rand_val(out16, rng, -4, 8);
        }
        if (special) {
            p.a[r][rng() % 16] = sa[rng() % sa.size()];
            p.b[r][rng() % 32] = sb[rng() % sb.size()];
            for (int c = 0; c < 16; ++c) {
                if (rng() % 4 == 0) {
                    p.c[r][c] = out16 == am::F32 ? sc32[rng() % sc32.size()] : sc16[rng() % sc16.size()];
                }
            }
        }
    }
    for (auto & x : p.idx) {
        x = (uint32_t) rng();
    }
    return p;
}

// Runs problems through op; dense: the arith_gpu.h slot layout of `dform` (arith::Form); sparse: A as the dense
// A image, B slot t at lane c + 16 (t >> 4), position t & 15 of the dense order. acc16: 16-bit C and D.
static std::vector<uint32_t> run_fp(const std::string & op, int dform, bool sparse, const std::vector<FP> & ps) {
    std::vector<uint32_t> out(ps.size() * 256);
    const int eb = arith::elem_bits(dform), cb = arith::acc16(dform) ? 16 : 32;
    const size_t CH = 8192;
    for (size_t base = 0; base < ps.size(); base += CH) {
        const int n = (int) std::min(CH, ps.size() - base);
        Img m;
        m.init(op, n);
        for (int w = 0; w < n; ++w) {
            const FP & p = ps[base + w];
            for (int r = 0; r < 16; ++r) {
                for (int s = 0; s < 16; ++s) {
                    arith::put_bits(&m.a[(size_t) w * 32 * m.na], arith::slot_ab(dform, r, s), m.na, p.a[r][s], eb);
                    if (!sparse) {
                        arith::put_bits(&m.b[(size_t) w * 32 * m.nb], arith::slot_ab(dform, r, s), m.nb, p.b[r][s], eb);
                    }
                }
                if (sparse) {
                    for (int t = 0; t < 32; ++t) {
                        const int e = t & 15;
                        const arith::Slot sl = eb == 8 ? arith::Slot{r + 16 * (t >> 4), e >> 2, (e & 3) * 8}
                                                       : arith::Slot{r + 16 * (t >> 4), e >> 1, (e & 1) * 16};
                        arith::put_bits(&m.b[(size_t) w * 32 * m.nb], sl, m.nb, p.b[r][t], eb);
                    }
                }
                for (int c = 0; c < 16; ++c) {
                    arith::put_bits(&m.c[(size_t) w * 32 * m.nc], arith::slot_cd(dform, r, c), m.nc, p.c[r][c], cb);
                }
            }
            for (int l = 0; l < 32; ++l) {
                m.i[(size_t) w * 32 + l] = p.idx[l];
            }
        }
        launch(m);
        for (int w = 0; w < n; ++w) {
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    out[(base + w) * 256 + r * 16 + c] =
                        arith::get_bits(&m.d[(size_t) w * 32 * m.nc], arith::slot_cd(dform, r, c), m.nc, cb);
                }
            }
        }
    }
    return out;
}

static long ndiff(const std::vector<uint32_t> & x, const std::vector<uint32_t> & y) {
    long d = 0;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
        d += x[i] != y[i];
    }
    return d;
}

struct ModCase {
    const char * op;    // with the modifier
    const char * base;  // the same instruction without it
    int expect;         // transform that should make the two equal (X_NONE: the bit is ignored)
    int ctl1, ctl2;     // transforms that must not
};

static bool run_modcases(const char * fname, int dform, bool sparse, am::Fmt out16, const std::vector<ModCase> & cs, long n,
                         std::string & text) {
    bool ok = true;
    const long np = (n + 255) / 256;
    std::mt19937_64 rng(7000 + dform * 3 + sparse);
    std::vector<FP> ps(np);
    for (long i = 0; i < np; ++i) {
        ps[i] = make_fp(arith::acc16(dform) ? (dform == arith::F16H ? am::F_F16 : am::F_BF16) : dform, out16, i & 1, rng);
    }
    const int eb = arith::elem_bits(dform), cb = out16 == am::F32 ? 32 : 16, nb = sparse ? 32 : 16;
    for (const ModCase & c : cs) {
        const std::vector<uint32_t> dm = run_fp(c.op, dform, sparse, ps);
        const int xs[4] = {g_break ? c.ctl1 : c.expect, c.ctl1, c.ctl2, X_NONE};
        long d[4];
        for (int k = 0; k < 4; ++k) {
            const size_t m = k == 0 ? ps.size() : std::min<size_t>(ps.size(), 1024);
            std::vector<FP> px(ps.begin(), ps.begin() + m);
            for (auto & p : px) {
                apply(p, xs[k], eb, cb, nb);
            }
            const std::vector<uint32_t> db = run_fp(c.base, dform, sparse, px);
            std::vector<uint32_t> dmm(dm.begin(), dm.begin() + m * 256);
            d[k] = ndiff(dmm, db);
            std::printf("RESULT mods %-14s %-16s vs %-12s with %-28s elements %zu differ %ld%s\n", fname, c.op, c.base, xf_name[xs[k]],
                        m * 256, d[k], k == 0 ? "" : k == 3 ? " (no change: is the bit ignored?)" : " (control)");
        }
        ok = ok && d[0] == 0 && d[1] > 0 && d[2] > 0;
        add_text(text, "%s %s = %s: %ld differ of %ld", fname, c.op, xf_name[xs[0]], d[0], np * 256);
    }
    return ok;
}

// NEG_HI on the integer forms (raw encodings): the card with the bit against the same instruction without it and
// against the other sign settings, on random integer data; plus the raw encoding without the bit against the
// assembled instruction (the encoding is right). Reported, not judged, except the encoding check and a control
// that the comparison sees a sign-setting change.
static bool mode_inthi(long n, std::string & text) {
    bool ok = true;
    const long np = (n + 255) / 256;
    struct C {
        const char * op;
        int dform;
        std::vector<const char *> cmp;
    };
    const std::vector<C> cs = {{"IU8_sAB_raw", arith::IU8, {"IU8_sAB", "IU8_uu"}},
                               {"IU8_sAB_nhA", arith::IU8, {"IU8_sAB", "IU8_uu", "IU8_sA", "IU8_sB"}},
                               {"IU8_sAB_nhB", arith::IU8, {"IU8_sAB", "IU8_uu", "IU8_sA", "IU8_sB"}},
                               {"IU8_uu_nhA", arith::IU8, {"IU8_uu", "IU8_sAB", "IU8_sA", "IU8_sB"}},
                               {"IU4_sAB_nhA", arith::IU4, {"D16_3", "D16_0", "D16_1", "D16_2"}},
                               {"IU4_uu_nhA", arith::IU4, {"D16_0", "D16_3", "D16_1", "D16_2"}},
                               {"IU4K32_sAB_nhA", arith::IU4K32, {"D32_3", "D32_0", "D32_1", "D32_2"}}};
    for (const C & c : cs) {
        std::mt19937_64 rng(8100 + c.dform);
        Img m;
        m.init(c.op, (int) np);
        for (auto & x : m.a) x = (uint32_t) rng();
        for (auto & x : m.b) x = (uint32_t) rng();
        for (auto & x : m.c) x = (uint32_t) rng();
        launch(m);
        std::string line;
        long first = -1, second = -1;
        for (size_t k = 0; k < c.cmp.size(); ++k) {
            Img q = m;
            q.op = c.cmp[k];
            launch(q);
            const long d = ndiff(m.d, q.d);
            if (k == 0) first = d;
            if (k == 1) second = d;
            char buf[96];
            std::snprintf(buf, sizeof buf, " vs %s: %ld differ;", c.cmp[k], d);
            line += buf;
        }
        std::printf("RESULT mods inthi %-15s elements %ld%s\n", c.op, np * 256, line.c_str());
        if (std::string(c.op) == "IU8_sAB_raw") {
            ok = ok && first == (g_break ? 1 : 0) && second > 0;
        }
        add_text(text, "%s vs %s %ld differ", c.op, c.cmp[0], first);
    }
    return ok;
}

static bool mode_mods(long n, std::string & text) {
    bool ok = true;
    // BF8 forms: the C bits (the assembler accepts them; on FP8 x FP8 they were ignored, matrix-arithmetic.md 9)
    for (int f : {arith::FB, arith::BF, arith::BB}) {
        const char * p = f == arith::FB ? "FB" : f == arith::BF ? "BF" : "BB";
        const std::string none = std::string(p) + "_none", ng = std::string(p) + "_negC", ab = std::string(p) + "_absC",
                          na = std::string(p) + "_nabsC";
        static std::vector<std::string> keep;
        keep.insert(keep.end(), {none, ng, ab, na});
        const size_t k0 = keep.size() - 4;
        const std::vector<ModCase> cs = {{keep[k0 + 1].c_str(), keep[k0].c_str(), X_NONE, X_C_NEG, X_C_ABS},
                                         {keep[k0 + 2].c_str(), keep[k0].c_str(), X_NONE, X_C_ABS, X_C_NEG},
                                         {keep[k0 + 3].c_str(), keep[k0].c_str(), X_NONE, X_C_NEGABS, X_C_NEG}};
        ok = run_modcases(arith::form_name[f], f, false, am::F32, cs, n, text) && ok;
    }
    // sparse float forms: NEG and NEG_HI on A and B
    struct S {
        const char * pre;
        int dform;
        am::Fmt out16;
    };
    const S ss[] = {{"SF16", arith::F16, am::F32}, {"SBF16", arith::BF16, am::F32}, {"SF16H", arith::F16H, am::F16},
                    {"SBF16H", arith::BF16H, am::BF16}};
    static std::vector<std::string> keep2;
    for (const S & s : ss) {
        const std::string pre = s.pre;
        keep2.insert(keep2.end(), {pre + "_none", pre + "_negA", pre + "_neghiA", pre + "_negB", pre + "_neghiB"});
        const size_t k0 = keep2.size() - 5;
        const char * b = keep2[k0].c_str();
        // measured: NEG / NEG_HI on B act on the B values after the gather, by the parity of the stored slot they
        // meet, which for the output is the same as on A; the B register halves (the dense reading) are the controls
        const std::vector<ModCase> cs = {{keep2[k0 + 1].c_str(), b, X_A_EVEN, X_A_ODD, X_C_NEG},
                                         {keep2[k0 + 2].c_str(), b, X_A_ODD, X_A_EVEN, X_C_NEG},
                                         {keep2[k0 + 3].c_str(), b, X_A_EVEN, X_B_EVEN, X_A_ODD},
                                         {keep2[k0 + 4].c_str(), b, X_A_ODD, X_B_ODD, X_A_EVEN}};
        ok = run_modcases(s.pre, s.dform, true, s.out16, cs, n, text) && ok;
    }
    // what NEG / NEG_HI on B do in the sparse forms: every candidate transform, on the FP16 form (exploration)
    {
        const long np = 1024;
        std::mt19937_64 rng(7333);
        std::vector<FP> ps(np);
        for (long i = 0; i < np; ++i) {
            ps[i] = make_fp(am::F_F16, am::F32, i & 1, rng);
        }
        for (const char * op : {"SF16_negB", "SF16_neghiB"}) {
            const std::vector<uint32_t> dm = run_fp(op, arith::F16, true, ps);
            for (int x = 0; x < NXF; ++x) {
                std::vector<FP> px = ps;
                for (auto & p : px) {
                    apply(p, x, 16, 32, 32);
                }
                std::printf("RESULT mods scan %-12s vs SF16_none with %-28s elements %ld differ %ld\n", op, xf_name[x], np * 256,
                            ndiff(dm, run_fp("SF16_none", arith::F16, true, px)));
            }
        }
    }
    // sparse FP16 form, C bits by raw encoding (the assembler rejects them, C being D): the raw word without the
    // bits must equal the assembled instruction; with them, measured: NEG flips and ABS clears the sign of C
    {
        const std::vector<ModCase> cs = {{"SF16_raw", "SF16_none", X_NONE, X_C_NEG, X_C_ABS},
                                         {"SF16_negC_raw", "SF16_none", X_C_NEG, X_C_ABS, X_NONE},
                                         {"SF16_absC_raw", "SF16_none", X_C_ABS, X_C_NEG, X_NONE}};
        ok = run_modcases("SF16 raw", arith::F16, true, am::F32, cs, n / 4, text) && ok;
    }
    ok = mode_inthi(n / 4, text) && ok;
    return ok;
}

// Leftovers C, MA-1 and MA-3 (mode mods2). Integer forms: NEG_HI on B for the 4-bit forms and unsigned 8-bit (MA-1),
// NEG_HI on A and B of the unsigned 4-bit K = 32 form (MA-3); expected ignored, as NEG_HI on A was (9a, MEASURED).
// The raw words without the bit must equal the assembled instruction (the encoding is right). Sparse float forms:
// the C bits (MA-3). Expected for the 16-bit-input forms: NEG flips and ABS clears C's sign, as on the FP16 into
// FP32 sparse form and on every 16-bit dense form (9 and 9a, MEASURED); for FP8 and BF8 sparse: unknown (the dense
// FP8 and BF8 forms ignore the bits), so those rows are INFO with "ignored" as the hypothesis tested.
static bool mode_mods2(long n, std::string & text) {
    bool ok = true;
    const long np = (n + 255) / 256;
    struct C {
        const char * op;
        int dform;
        std::vector<const char *> cmp;  // cmp[0]: expected equal (0 differ); the rest: controls (must differ)
    };
    const std::vector<C> cs = {{"IU4_uu_raw", arith::IU4, {"D16_0", "D16_3"}},
                               {"IU4K32_uu_raw", arith::IU4K32, {"D32_0", "D32_3"}},
                               {"IU8_uu_nhB", arith::IU8, {"IU8_uu", "IU8_sAB", "IU8_sA", "IU8_sB"}},
                               {"IU4_sAB_nhB", arith::IU4, {"D16_3", "D16_0", "D16_1", "D16_2"}},
                               {"IU4_uu_nhB", arith::IU4, {"D16_0", "D16_3", "D16_1", "D16_2"}},
                               {"IU4K32_sAB_nhB", arith::IU4K32, {"D32_3", "D32_0", "D32_1", "D32_2"}},
                               {"IU4K32_uu_nhB", arith::IU4K32, {"D32_0", "D32_3", "D32_1", "D32_2"}},
                               {"IU4K32_uu_nhA", arith::IU4K32, {"D32_0", "D32_3", "D32_1", "D32_2"}}};
    for (const C & c : cs) {
        std::mt19937_64 rng(8200 + c.dform);
        Img m;
        m.init(c.op, (int) np);
        for (auto & x : m.a) x = (uint32_t) rng();
        for (auto & x : m.b) x = (uint32_t) rng();
        for (auto & x : m.c) x = (uint32_t) rng();
        launch(m);
        std::string line;
        bool good = true;
        for (size_t k = 0; k < c.cmp.size(); ++k) {
            Img q = m;
            q.op = c.cmp[k];
            launch(q);
            const long d = ndiff(m.d, q.d);
            // --break expects NEG_HI to act (the instruction with it to differ from the one without)
            const bool raw = std::strstr(c.op, "_raw") != nullptr;
            good = good && (k == 0 ? ((g_break && !raw) ? d > 0 : d == 0) : d > 0);
            char buf[96];
            std::snprintf(buf, sizeof buf, " vs %s: %ld differ%s;", c.cmp[k], d, k == 0 ? " (expected 0)" : " (control)");
            line += buf;
        }
        std::printf("RESULT mods2 int %-15s elements %ld%s\n", c.op, np * 256, line.c_str());
        ok = ok && good;
        add_text(text, "%s %s", c.op, good ? "as expected" : "NOT as expected");
    }
    struct S {
        const char * pre;
        const char * none;
        int dform;
        am::Fmt out16;
        bool predicted;
    };
    const S ss[] = {{"SBF16", "SBF16_none", arith::BF16, am::F32, true},
                    {"SF16H", "SF16H_none", arith::F16H, am::F16, true},
                    {"SBF16H", "SBF16H_none", arith::BF16H, am::BF16, true},
                    {"SFF", "SFF_none", arith::FF, am::F32, false},
                    {"SBB", "SBB_none", arith::BB, am::F32, false}};
    static std::vector<std::string> keep;
    for (const S & q : ss) {
        const std::string pre = q.pre;
        keep.insert(keep.end(), {pre + "_raw", pre + "_negC_raw", pre + "_absC_raw"});
        const size_t k0 = keep.size() - 3;
        // the raw word without bits must equal the assembled instruction (asserted for every form)
        const std::vector<ModCase> c0 = {{keep[k0].c_str(), q.none, X_NONE, X_C_NEG, X_C_ABS}};
        const bool enc = run_modcases((pre + " raw").c_str(), q.dform, true, q.out16, c0, n / 4, text);
        ok = ok && enc;
        // C bits: predicted forms flip / clear; FP8 and BF8: hypothesis "ignored", INFO
        const std::vector<ModCase> c1 =
            q.predicted ? std::vector<ModCase>{{keep[k0 + 1].c_str(), q.none, X_C_NEG, X_C_ABS, X_NONE},
                                               {keep[k0 + 2].c_str(), q.none, X_C_ABS, X_C_NEG, X_NONE}}
                        : std::vector<ModCase>{{keep[k0 + 1].c_str(), q.none, X_NONE, X_C_NEG, X_C_ABS},
                                               {keep[k0 + 2].c_str(), q.none, X_NONE, X_C_ABS, X_C_NEG}};
        std::string t2;
        const bool r = run_modcases((pre + " C").c_str(), q.dform, true, q.out16, c1, n / 4, t2);
        if (q.predicted) {
            ok = ok && r;
            add_text(text, "%s", t2.c_str());
        } else {
            std::printf("INFO mods2 %s: C bits %s (hypothesis \"ignored\": %s)\n", q.pre, r ? "ignored" : "not ignored",
                        t2.c_str());
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Item 4: the forbidden back-to-back read (matrix-arithmetic.md section 13) for more forms. Fixed registers:
// D1 v[0:7] (starts as INIT: known old contents; for the sparse case C0, since D is C there), first A v[8:15],
// first B v[16:23], first C v[24:31], the second instruction's other operand v[32:39], its C and D v[40:47],
// index words v48, v49. V = 1 puts a v_nop between the two instructions.
#define HZ_N0 ""
#define HZ_N1 "v_nop\n"
#define HZ_T0(N) "v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_f32_16x16x16_f16 v[40:47], v[0:3], v[32:35], v[40:47]"
#define HZ_T1(N) "v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_f32_16x16x16_f16 v[40:47], v[32:35], v[0:3], v[40:47]"
#define HZ_T2(N) "v_wmma_bf16_16x16x16_bf16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_bf16_16x16x16_bf16 v[40:43], v[0:3], v[32:35], v[40:43]"
#define HZ_T3(N) "v_wmma_bf16_16x16x16_bf16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_bf16_16x16x16_bf16 v[40:43], v[32:35], v[0:3], v[40:43]"
#define HZ_T4(N) "v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]\n" N "v_wmma_i32_16x16x16_iu8 v[40:47], v[0:1], v[32:33], v[40:47] neg_lo:[1,1,0]"
#define HZ_T5(N) "v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]\n" N "v_wmma_i32_16x16x16_iu8 v[40:47], v[6:7], v[32:33], v[40:47] neg_lo:[1,1,0]"
#define HZ_T6(N) "v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]\n" N "v_wmma_i32_16x16x16_iu8 v[40:47], v[32:33], v[0:1], v[40:47] neg_lo:[1,1,0]"
#define HZ_T7(N) "v_swmmac_f16_16x16x32_f16 v[0:3], v[8:11], v[16:23], v48\n" N "v_swmmac_f32_16x16x32_f16 v[40:47], v[0:3], v[32:39], v49"
// MA-2: the second instruction of another form than the first.
#define HZ_T8(N) "v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[24:31]\n" N "v_wmma_f16_16x16x16_f16 v[40:43], v[0:3], v[32:35], v[40:43]"
#define HZ_T9(N) "v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[24:31]\n" N "v_wmma_f16_16x16x16_f16 v[40:43], v[32:35], v[0:3], v[40:43]"
#define HZ_T10(N) "v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[16:17], v[24:31]\n" N "v_wmma_f32_16x16x16_f16 v[40:47], v[0:3], v[32:35], v[40:47]"
#define HZ_T11(N) "v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[16:17], v[24:31]\n" N "v_wmma_f32_16x16x16_f16 v[40:47], v[32:35], v[0:3], v[40:47]"
#define HZ_T12(N) "v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]\n" N "v_wmma_bf16_16x16x16_bf16 v[40:43], v[0:3], v[32:35], v[40:43]"
#define HZ_T13(N) "v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]\n" N "v_wmma_bf16_16x16x16_bf16 v[40:43], v[32:35], v[0:3], v[40:43]"
#define HZ_T14(N) "v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[0:1], v[32:33], v[40:47]"
#define HZ_T15(N) "v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]\n" N "v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[32:33], v[0:1], v[40:47]"
enum { HZ_NCASES = 16, HZ_NOLD = 8 };
struct HzCase {
    const char * name;
    const char * second;  // the second instruction alone, as a generic op (A, B, C/D, index)
    bool as_b;
    int nv, off, nc;      // VGPRs of D1 read as the operand, from D1 VGPR off; words of C/D of the second
    bool sparse_first;
};
static const HzCase hz_cases[HZ_NCASES] = {
    {"f16_f16 D1 as A of f16 (lane 5 case)", "H2_f16", false, 4, 0, 8, false},
    {"f16_f16 D1 as B of f16 (lane 5 case)", "H2_f16", true, 4, 0, 8, false},
    {"bf16_bf16 D1 as A of bf16_bf16", "H2_bf16h", false, 4, 0, 4, false},
    {"bf16_bf16 D1 as B of bf16_bf16", "H2_bf16h", true, 4, 0, 4, false},
    {"iu8 D1 VGPRs 0-1 as A of iu8", "IU8_sAB", false, 2, 0, 8, false},
    {"iu8 D1 VGPRs 6-7 as A of iu8", "IU8_sAB", false, 2, 6, 8, false},
    {"iu8 D1 VGPRs 0-1 as B of iu8", "IU8_sAB", true, 2, 0, 8, false},
    {"swmmac_f16_f16 D1 as A of swmmac_f16", "SF16_none", false, 4, 0, 8, true},
    {"f32_f16 D1 VGPRs 0-3 as A of f16_f16", "H2_f16h", false, 4, 0, 4, false},
    {"f32_f16 D1 VGPRs 0-3 as B of f16_f16", "H2_f16h", true, 4, 0, 4, false},
    {"fp8 D1 VGPRs 0-3 as A of f32_f16", "H2_f16", false, 4, 0, 8, false},
    {"fp8 D1 VGPRs 0-3 as B of f32_f16", "H2_f16", true, 4, 0, 8, false},
    {"iu8 D1 VGPRs 0-3 as A of bf16_bf16", "H2_bf16h", false, 4, 0, 4, false},
    {"iu8 D1 VGPRs 0-3 as B of bf16_bf16", "H2_bf16h", true, 4, 0, 4, false},
    {"f16_f16 D1 VGPRs 0-1 as A of fp8", "FF_none", false, 2, 0, 8, false},
    {"f16_f16 D1 VGPRs 0-1 as B of fp8", "FF_none", true, 2, 0, 8, false}};
// Input formats of the new cases (h FP16 pairs, b BF16 pairs, f FP32, 8 FP8 bytes, i random bits): first A and B,
// first C, the second's other operand, the second's C.
static const char * const hz_fmt[HZ_NCASES] = {"", "", "", "", "", "", "", "", "hfhh", "hfhh", "8fhf", "8fhf",
                                               "iibb", "iibb", "hh8f", "hh8f"};
LOP(H2_f16, v4i, v4i, v8i, "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0")
LOP(H2_bf16h, v4i, v4i, v4i, "v_wmma_bf16_16x16x16_bf16 %0, %1, %2, %0")
LOP(H2_f16h, v4i, v4i, v4i, "v_wmma_f16_16x16x16_f16 %0, %1, %2, %0")

// in: per wave and lane 8 words each of A0, B0, C0, INIT, X1, C1, then I0, I1 (8 words: 0, 1 used).
// out: D1 (8 words), D2 (8 words) per lane; info: HW_ID1 per wave. busy: the upper half of the block's waves run a
// matrix chain (waves are dealt to the SIMDs in turn, so wave k and k + 4 of a group share one; checked by HW_ID1).
// busy 2 (whole card): the chain is 500 times longer (about 80 us) and the pair waves first sleep about 25 us, so
// the pair runs while every SIMD's matrix unit is busy.
template <int CASE, int V>
__global__ __launch_bounds__(256) void hzk(const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                                         uint32_t * __restrict__ info, int busy) {
    const int wib = threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t w = (size_t) blockIdx.x * (blockDim.x / 32) + wib;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    if (lane == 0) {
        info[w] = hw;
    }
    const uint32_t * p = in + (w * 32 + lane) * 56;
    v8i a0, b0, c0, d1, x1, d2, ii;
    __builtin_memcpy(&a0, p, 32);
    __builtin_memcpy(&b0, p + 8, 32);
    __builtin_memcpy(&c0, p + 16, 32);
    __builtin_memcpy(&d1, p + 24, 32);
    __builtin_memcpy(&x1, p + 32, 32);
    __builtin_memcpy(&d2, p + 40, 32);
    __builtin_memcpy(&ii, p + 48, 32);
    const int i0 = ii[0], i1 = ii[1];
    __syncthreads();
    if (busy && wib >= (int) (blockDim.x / 64)) {
        // a partner on the same SIMD keeps the matrix unit busy while the even waves run the sequence
        const int reps = busy == 2 ? 2000 : 4;
        for (int k = 0; k < reps; ++k) {
            asm volatile(
                "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\nv_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n"
                "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\nv_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n"
                "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\nv_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n"
                "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\nv_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
                : "+v"(d2) : "v"(a0.lo), "v"(b0.lo));
        }
    } else {
        if (busy == 2) {
            for (int k = 0; k < 10; ++k) {
                asm volatile("s_sleep 127");
            }
        }
#define HZ_ASM(T)                                                                                       \
    asm volatile(T : "+{v[0:7]}"(d1), "+{v[40:47]}"(d2)                                                  \
                 : "{v[8:15]}"(a0), "{v[16:23]}"(b0), "{v[24:31]}"(c0), "{v[32:39]}"(x1), "{v48}"(i0), "{v49}"(i1))
#define HZ_SEL(C)                                \
    if constexpr (CASE == C) {                   \
        if constexpr (V == 0) {                  \
            HZ_ASM(HZ_T##C(HZ_N0));              \
        } else {                                 \
            HZ_ASM(HZ_T##C(HZ_N1));              \
        }                                        \
    }
        HZ_SEL(0) HZ_SEL(1) HZ_SEL(2) HZ_SEL(3) HZ_SEL(4) HZ_SEL(5) HZ_SEL(6) HZ_SEL(7)
        HZ_SEL(8) HZ_SEL(9) HZ_SEL(10) HZ_SEL(11) HZ_SEL(12) HZ_SEL(13) HZ_SEL(14) HZ_SEL(15)
    }
    __builtin_memcpy(out + (w * 32 + lane) * 16, &d1, 32);
    __builtin_memcpy(out + (w * 32 + lane) * 16 + 8, &d2, 32);
}
typedef void (*HZK)(const uint32_t *, uint32_t *, uint32_t *, int);
#define HZPAIR(C) {hzk<C, 0>, hzk<C, 1>}
static const HZK hz_kern[HZ_NCASES][2] = {HZPAIR(0), HZPAIR(1), HZPAIR(2),  HZPAIR(3),  HZPAIR(4),  HZPAIR(5),
                                          HZPAIR(6), HZPAIR(7), HZPAIR(8),  HZPAIR(9),  HZPAIR(10), HZPAIR(11),
                                          HZPAIR(12), HZPAIR(13), HZPAIR(14), HZPAIR(15)};

static void hz_launch(int cs, int v, int blocks, int wpb, int busy, const std::vector<uint32_t> & in, std::vector<uint32_t> & out,
                      std::vector<uint32_t> & info) {
    const size_t nw = (size_t) blocks * wpb;
    uint32_t *di, *dout, *dinfo;
    CHECK(hipMalloc(&di, in.size() * 4));
    CHECK(hipMalloc(&dout, nw * 32 * 16 * 4));
    CHECK(hipMalloc(&dinfo, nw * 4));
    CHECK(hipMemcpy(di, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(hz_kern[cs][v], dim3(blocks), dim3(32 * wpb), 0, 0, di, dout, dinfo, busy);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    out.resize(nw * 32 * 16);
    info.resize(nw);
    CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(info.data(), dinfo, info.size() * 4, hipMemcpyDeviceToHost));
    for (auto * q : {di, dout, dinfo}) {
        CHECK(hipFree(q));
    }
}

// Input word for slot 0 A0, 1 B0, 2 C0, 4 X1, 5 C1: random bytes for the integer cases; else 16-bit pairs, except
// the FP32 accumulator of the second instruction (cases 0, 1, 7).
static uint32_t hz_rand_fmt(char f, std::mt19937_64 & rng) {
    switch (f) {
    case 'h':
        return ad::rand_val(am::F16, rng, -3, 3, 1, 1) | ad::rand_val(am::F16, rng, -3, 3, 1, 1) << 16;
    case 'b':
        return ad::rand_val(am::BF16, rng, -3, 3, 1, 1) | ad::rand_val(am::BF16, rng, -3, 3, 1, 1) << 16;
    case 'f':
        return ad::rand_f32(rng, -4, 4);
    case '8':
        return ad::rand_code8(am::E4M3, rng) | ad::rand_code8(am::E4M3, rng) << 8 | ad::rand_code8(am::E4M3, rng) << 16 |
               ad::rand_code8(am::E4M3, rng) << 24;
    default:
        return (uint32_t) rng();
    }
}

static uint32_t hz_rand(int cs, int slot, std::mt19937_64 & rng) {
    if (cs >= HZ_NOLD) {
        const char * f = hz_fmt[cs];
        return hz_rand_fmt(slot <= 1 ? f[0] : slot == 2 ? f[1] : slot == 4 ? f[2] : f[3], rng);
    }
    if (cs >= 4 && cs <= 6) {
        return (uint32_t) rng();
    }
    if (slot == 5 && cs != 2 && cs != 3) {
        return ad::rand_f32(rng, -4, 4);
    }
    const am::Fmt f = (cs == 2 || cs == 3) ? am::BF16 : am::F16;
    return ad::rand_val(f, rng, -3, 3, 1, 1) | ad::rand_val(f, rng, -3, 3, 1, 1) << 16;
}

// mode hazard: the old cases alone and beside one busy wave (section 13a). mode hazard2 (MA-2): the new cases alone
// and beside one busy wave, and every case under whole-card load (busy 2).
static bool mode_hazard(std::string & text, bool v2) {
    bool ok = true, pred_ok = true;
    const int nhz = 64;
    for (int cs = 0; cs < HZ_NCASES; ++cs) {
        const HzCase & H = hz_cases[cs];
        if (!v2 && cs >= HZ_NOLD) {
            break;
        }
        for (int busy = 0; busy < 3; ++busy) {
            // hazard2 runs cases 0 and 1 alone as its calibration; without --card the new cases alone and beside one
            // busy wave; with --card every case under whole-card load
            const bool cal = cs < 2 && busy == 0;
            if ((!v2 && busy == 2) || (v2 && !cal && (g_card ? busy != 2 : (busy == 2 || cs < HZ_NOLD)))) {
                continue;
            }
            // alone: one wave per work-group; busy: work-groups of 8 waves, waves 4-7 running a matrix chain;
            // busy 2: 128 such groups (4 per WGP), so every SIMD of the card runs a chain
            const int wpb = busy ? 8 : 1, blocks = busy == 2 ? 128 : busy ? nhz * 2 / wpb : nhz;
            const int nw = blocks * wpb;
            std::mt19937_64 rng(9100 + cs);
            std::vector<uint32_t> in((size_t) nw * 32 * 56);
            for (int w = 0; w < nw; ++w) {
                for (int l = 0; l < 32; ++l) {
                    uint32_t * q = &in[((size_t) w * 32 + l) * 56];
                    for (int s = 0; s < 6; ++s) {
                        for (int k = 0; k < 8; ++k) {
                            q[s * 8 + k] = s == 3 ? (cs >= 4 && cs <= 6 ? (uint32_t) rng() : 0x3C003C00u ^ ((uint32_t) rng() & 0x01FF01FFu))
                                                  : hz_rand(cs, s, rng);
                        }
                    }
                    if (H.sparse_first) {
                        for (int k = 0; k < 8; ++k) {
                            q[24 + k] = q[16 + k];  // D1 is C0 in place
                        }
                    }
                    q[48] = (uint32_t) rng();
                    q[49] = (uint32_t) rng();
                }
            }
            std::vector<uint32_t> o0, o1, info0, info1;
            hz_launch(cs, 1, blocks, wpb, busy, in, o1, info1);
            hz_launch(cs, 0, blocks, wpb, busy, in, o0, info0);
            // hazard waves: alone every wave; busy waves 0-3 of each group. Every one counts the differing results;
            // the stale-word fit uses at most 64 of them.
            std::vector<int> hw, hw_all;
            std::set<uint32_t> simds;
            long shared = 0;
            for (int w = 0; w < nw; ++w) {
                if (busy && (w % wpb) >= wpb / 2) {
                    continue;
                }
                hw_all.push_back(w);
                simds.insert((info0[w] >> 8) & 0xD3Fu);  // SIMD, WGP, shader array, shader engine fields of HW_ID1
                if (hw.size() < (size_t) nhz) {
                    hw.push_back(w);
                }
                if (busy) {
                    const int b0 = w - w % wpb;
                    bool sh = false;
                    for (int u = b0; u < b0 + wpb; ++u) {
                        sh = sh || ((u % wpb) >= wpb / 2 && ((info0[u] >> 8) & 3) == ((info0[w] >> 8) & 3));
                    }
                    shared += sh;
                }
            }
            long d2diff = 0, d1diff = 0;
            for (int w : hw_all) {
                for (int l = 0; l < 32; ++l) {
                    for (int k = 0; k < 8; ++k) {
                        const size_t x = ((size_t) w * 32 + l) * 16 + k;
                        d1diff += o0[x] != o1[x];
                        if (k < H.nc) {
                            // 16-bit results: two elements per word
                            const uint32_t dx = o0[x + 8] ^ o1[x + 8];
                            d2diff += H.nc == 8 ? dx != 0 : ((dx & 0xFFFFu) != 0) + ((dx >> 16) != 0);
                        }
                    }
                }
            }
            // which D1 words were read stale: the second instruction alone on the card, with every mix of new and
            // old words of one row's (column's) two lanes x nv VGPRs, the same mix in every row
            const int nmask = 1 << (2 * H.nv);
            Img m;
            m.init(H.second, (int) hw.size() * nmask);
            for (size_t h = 0; h < hw.size(); ++h) {
                const int w = hw[h];
                for (int mk = 0; mk < nmask; ++mk) {
                    const int wv = (int) h * nmask + mk;
                    for (int l = 0; l < 32; ++l) {
                        const uint32_t * q = &in[((size_t) w * 32 + l) * 56];
                        const uint32_t * d1n = &o1[((size_t) w * 32 + l) * 16];
                        for (int k = 0; k < H.nv; ++k) {
                            const bool old = (mk >> ((l >> 4) * H.nv + k)) & 1;
                            const uint32_t val = old ? q[24 + H.off + k] : d1n[H.off + k];
                            if (H.as_b) {
                                m.bw(wv, l, k) = val;
                            } else {
                                m.aw(wv, l, k) = val;
                            }
                        }
                        for (int k = 0; k < (H.as_b ? m.na : m.nb); ++k) {
                            (H.as_b ? m.aw(wv, l, k) : m.bw(wv, l, k)) = q[32 + k];
                        }
                        for (int k = 0; k < m.nc; ++k) {
                            m.cw(wv, l, k) = q[40 + k];
                        }
                        m.i[(size_t) wv * 32 + l] = q[49];
                    }
                }
            }
            launch(m);
            std::map<int, long> hist;  // per row: smallest mask of stale words that reproduces the forbidden result
            long nop_bad = 0;
            for (size_t h = 0; h < hw.size(); ++h) {
                const int w = hw[h];
                for (int qrow = 0; qrow < 16; ++qrow) {
                    int best = -1;
                    for (int pc = 0; pc <= 2 * H.nv && best < 0; ++pc) {
                        for (int mk = 0; mk < nmask && best < 0; ++mk) {
                            if (__builtin_popcount(mk) != pc) {
                                continue;
                            }
                            bool all = true;
                            for (int o = 0; o < 16 && all; ++o) {
                                const int r = H.as_b ? o : qrow, c = H.as_b ? qrow : o;
                                const int ln = cd_lane(r, c);
                                // FP32 / integer D: VGPR r & 7; 16-bit D: VGPR (r & 7) >> 1, compare the half
                                const int k = H.nc == 8 ? (r & 7) : (r & 7) >> 1;
                                const uint32_t sh = H.nc == 8 ? 0 : 16 * (r & 1), msk = H.nc == 8 ? ~0u : 0xFFFFu;
                                const uint32_t got = (o0[((size_t) w * 32 + ln) * 16 + 8 + k] >> sh) & msk;
                                const uint32_t mod = (m.dw((int) h * nmask + mk, ln, k) >> sh) & msk;
                                all = got == mod;
                            }
                            if (all) {
                                best = mk;
                            }
                        }
                    }
                    ++hist[qrow * 1000 + (best < 0 ? 999 : best)];
                    // the V_NOP version must equal the second instruction alone on the new D1 (mask 0)
                    for (int o = 0; o < 16; ++o) {
                        const int r = H.as_b ? o : qrow, c = H.as_b ? qrow : o;
                        const int ln = cd_lane(r, c), k = H.nc == 8 ? (r & 7) : (r & 7) >> 1;
                        const uint32_t sh = H.nc == 8 ? 0 : 16 * (r & 1), msk = H.nc == 8 ? ~0u : 0xFFFFu;
                        nop_bad += ((o1[((size_t) w * 32 + ln) * 16 + 8 + k] >> sh) & msk) != ((m.dw((int) h * nmask, ln, k) >> sh) & msk);
                    }
                }
            }
            std::string rows;
            for (const auto & kv : hist) {
                const int qrow = kv.first / 1000, mk = kv.first % 1000;
                if (mk == 0) {
                    continue;
                }
                char buf[80];
                if (mk == 999) {
                    std::snprintf(buf, sizeof buf, " %d:no-mix-fits x%ld", qrow, kv.second);
                } else {
                    std::snprintf(buf, sizeof buf, " %d:0x%02x x%ld", qrow, mk, kv.second);
                }
                rows += buf;
            }
            const char * bname = busy == 2 ? "card-busy" : busy ? "busy-partner" : "alone";
            std::printf("RESULT hazard %-40s %s waves %zu (fit on %zu; sharing a SIMD with a busy wave: %ld; distinct SIMDs %zu) D2 "
                        "elements differing from the V_NOP version %ld of %zu, D1 differs %ld, V_NOP version wrong %ld | %s with "
                        "stale words (bit h*%d+k: lane r+16h, operand VGPR k):%s\n",
                        H.name, bname, hw_all.size(), hw.size(), shared, simds.size(), d2diff, hw_all.size() * 256, d1diff,
                        nop_bad, H.as_b ? "columns" : "rows", H.nv, rows.empty() ? " none" : rows.c_str());
            if (v2) {
                // MA-2: section 13a's rule as an exact prediction. The second instruction alone, fed operand VGPRs 0
                // and 1 old in lanes 0-15 (mix 0x03) and everything else new, must give the forbidden result in rows
                // 0, 1, 8, 9 (as A) or in every column (as B); every other row must equal the all-new result (mix 0).
                // --break predicts the B pattern (every row stale) for the A cases. How many results differ depends on
                // the data (an old word can give the same product), so the count is INFO.
                long pred_bad = 0;
                for (size_t h = 0; h < hw.size(); ++h) {
                    const int w = hw[h];
                    for (int qrow = 0; qrow < 16; ++qrow) {
                        const bool stale = H.as_b || g_break || qrow == 0 || qrow == 1 || qrow == 8 || qrow == 9;
                        const int mk = stale ? 0x03 : 0;
                        for (int o = 0; o < 16; ++o) {
                            const int r = H.as_b ? o : qrow, c = H.as_b ? qrow : o;
                            const int ln = cd_lane(r, c), k = H.nc == 8 ? (r & 7) : (r & 7) >> 1;
                            const uint32_t sh = H.nc == 8 ? 0 : 16 * (r & 1), msk = H.nc == 8 ? ~0u : 0xFFFFu;
                            pred_bad += ((o0[((size_t) w * 32 + ln) * 16 + 8 + k] >> sh) & msk) !=
                                        ((m.dw((int) h * nmask + mk, ln, k) >> sh) & msk);
                        }
                    }
                }
                const long upper = (long) hw_all.size() * (H.as_b ? 256 : 64);
                // Whole-card load (busy 2): every one of the pair waves (not only the fitted ones) is classified: its
                // result equals the rule's prediction (stale), the all-new result (= the V_NOP version), or neither.
                long w_stale = 0, w_new = 0, w_mixed = 0;
                if (busy == 2) {
                    Img m2;
                    m2.init(H.second, (int) hw_all.size() * 2);
                    for (size_t h = 0; h < hw_all.size(); ++h) {
                        const int w = hw_all[h];
                        for (int v = 0; v < 2; ++v) {
                            const int wv = (int) h * 2 + v;
                            for (int l = 0; l < 32; ++l) {
                                const uint32_t * q = &in[((size_t) w * 32 + l) * 56];
                                const uint32_t * d1n = &o1[((size_t) w * 32 + l) * 16];
                                for (int k = 0; k < H.nv; ++k) {
                                    const bool old = v == 1 && l < 16 && k < 2;
                                    (H.as_b ? m2.bw(wv, l, k) : m2.aw(wv, l, k)) = old ? q[24 + H.off + k] : d1n[H.off + k];
                                }
                                for (int k = 0; k < (H.as_b ? m2.na : m2.nb); ++k) {
                                    (H.as_b ? m2.aw(wv, l, k) : m2.bw(wv, l, k)) = q[32 + k];
                                }
                                for (int k = 0; k < m2.nc; ++k) {
                                    m2.cw(wv, l, k) = q[40 + k];
                                }
                                m2.i[(size_t) wv * 32 + l] = q[49];
                            }
                        }
                    }
                    launch(m2);
                    for (size_t h = 0; h < hw_all.size(); ++h) {
                        const int w = hw_all[h];
                        bool st = true, nw = true;
                        for (int r = 0; r < 16; ++r) {
                            const bool stale_row = H.as_b || r == 0 || r == 1 || r == 8 || r == 9;
                            for (int c = 0; c < 16; ++c) {
                                const int rr = H.as_b ? c : r, cc = H.as_b ? r : c;  // as B the stale lines are columns
                                const int ln = cd_lane(rr, cc), k = H.nc == 8 ? (rr & 7) : (rr & 7) >> 1;
                                const uint32_t sh = H.nc == 8 ? 0 : 16 * (rr & 1), msk = H.nc == 8 ? ~0u : 0xFFFFu;
                                const uint32_t got = (o0[((size_t) w * 32 + ln) * 16 + 8 + k] >> sh) & msk;
                                st = st && got == ((m2.dw((int) h * 2 + (stale_row ? 1 : 0), ln, k) >> sh) & msk);
                                nw = nw && got == ((m2.dw((int) h * 2, ln, k) >> sh) & msk);
                            }
                        }
                        w_stale += st && !nw;
                        w_new += nw;
                        w_mixed += !st && !nw;
                    }
                    std::printf("RESULT hazard2 %-40s %s waves: %ld read old words exactly by the rule, %ld read only new words "
                                "(as with a V_NOP), %ld neither\n",
                                H.name, bname, w_stale, w_new, w_mixed);
                }
                std::printf("RESULT hazard2 %-40s %s results differing from the predicted forbidden result %ld of %zu (fit "
                            "waves); differing from the V_NOP version %ld, at most %ld possible under the rule%s\n",
                            H.name, bname, pred_bad, hw.size() * 256, d2diff, upper, cal ? " (calibration: section 13)" : "");
                // alone and beside one busy wave every fitted wave must match the rule exactly; under whole-card load
                // every wave must match either the rule or the all-new result (never a mix), with at least one stale
                pred_ok = pred_ok && d2diff <= upper && d2diff > 0 &&
                          (busy == 2 ? (w_mixed == 0 && w_stale > 0 && !g_break) : pred_bad == 0);
                ok = ok && nop_bad == 0 && d1diff == 0 && (!busy || shared == (long) hw_all.size());
                if (busy == 2) {
                    ok = ok && simds.size() == 128;
                }
                add_text(text, "%s %s: %ld of %zu differ", H.name, bname, d2diff, hw_all.size() * 256);
                continue;
            }
            ok = ok && nop_bad == 0 && d1diff == 0 && (!g_break || d2diff == 0);
            if (!g_break) {
                // pinned after the review: every affected row or column is explained by stale operand VGPRs 0 and 1
                // of the lower lanes (mask 0x03) and by nothing else; the forbidden read is wrong in every case; in
                // the busy cases every tested wave shares its SIMD with the partner
                bool only_03 = true;
                for (const auto & kv : hist) {
                    const int mk = kv.first % 1000;
                    only_03 = only_03 && (mk == 0 || mk == 0x03);
                }
                ok = ok && only_03 && d2diff > 0 && (!busy || shared == (long) hw.size());
            }
            if (cs == 0 && !busy && !g_break) {
                ok = ok && d2diff == 4096;  // lane 5's result for this sequence (section 13)
            }
            if (cs == 1 && !busy && !g_break) {
                ok = ok && d2diff == 16384;
            }
            add_text(text, "%s %s: %ld of %zu differ", H.name, busy ? "busy" : "alone", d2diff, hw.size() * 256);
        }
    }
    if (v2) {
        verdict("left_mat_hazard2_prediction", pred_ok,
                "every forbidden result equals section 13a's rule (operand VGPRs 0, 1 of lanes 0-15 read old; rows 0, 1, 8, 9 "
                "as A, every column as B) in every fitted wave; under whole-card load every wave either by the rule or "
                "all new");
        ok = ok && pred_ok;
    }
    return ok;
}

#define REG2(NAME) g_ops[#NAME] = OpInfo{gk<NAME>, (int) sizeof(NAME::ta) / 4, (int) sizeof(NAME::tb) / 4, (int) sizeof(NAME::tc) / 4};
static void register_hz() { REG2(H2_f16) REG2(H2_bf16h) REG2(H2_f16h) }

// ---------------------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    long n = 500000;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--break") {
            g_break = true;
        } else if (a == "--card") {
            g_card = true;
        } else if (a == "--n" && i + 1 < argc) {
            n = std::atol(argv[++i]);
        }
    }
    require_arch();
    register_ops();
    std::string text;
    bool ok;
    if (mode == "s4map") {
        ok = mode_s4map(text);
    } else if (mode == "s4") {
        ok = mode_s4(n, text);
    } else if (mode == "mask") {
        ok = mode_mask(text);
    } else if (mode == "hazard") {
        ok = mode_hazard(text, false);
    } else if (mode == "mods2") {
        ok = mode_mods2(n, text);
    } else if (mode == "mask2") {
        ok = lm::mask_test(g_break, text);
    } else if (mode == "hazard2") {
        ok = mode_hazard(text, true);
    } else if (mode == "mods") {
        ok = mode_mods(n, text);
    } else {
        std::fprintf(stderr, "usage: left_mat s4map|s4|mask|mask2|mods|mods2|hazard|hazard2 [--n N] [--break]\n");
        return 2;
    }
    verdict("left_mat_" + mode, ok, text);
    return ok ? 0 : 1;
}
