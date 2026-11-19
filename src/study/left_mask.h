// Leftovers C, CH-6: every matrix opcode under a lane mask, in wave32 (left_mat mask2) and wave64 (left_mat64 mask),
// with the time of a chained instruction under a full and a one-lane EXEC. Included by left_mat.hip (wave32 build)
// and left_mat64.hip (built with -mwavefrontsize64); LM_WAVE is 32 or 64.
//
// Test (as compiler-and-hazards section 1 and left_mat mask): EXEC set around one instruction in inline assembly (7
// masks, 0 included), D preset to a sentinel (sparse forms: D is C, so the untouched C is the sentinel), everything
// stored under full EXEC afterwards, each launch run twice. A matrix form passes when every result word equals the
// full-EXEC result (so every lane's operands were read and every lane's D written); the two dot instructions are the
// controls that obey the mask. Timing: one wave, the chain D = op(A, B, D) of 16 and of 48 instructions between two
// reads of the cycle counter, EXEC set before the first read; cycles per instruction = (best t48 - best t16) / 32 over
// 7 repetitions each.
// Known chain costs (matrix-timing sections 2 and 3): 16 for the 16-bit-input forms, 8 for the 8-bit and 4-bit ones.
#pragma once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef LM_WAVE
#error "define LM_WAVE 32 or 64"
#endif

namespace lm {

typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));
template <int N> struct VT;
template <> struct VT<1> { typedef int t; };
template <> struct VT<2> { typedef v2i_t t; };
template <> struct VT<4> { typedef v4i_t t; };
template <> struct VT<8> { typedef v8i_t t; };
// VGPRs per lane of an operand that takes n VGPRs in wave32
constexpr int W(int n) { return LM_WAVE == 64 ? (n > 1 ? n / 2 : 1) : n; }

#if LM_WAVE == 64
typedef uint64_t mask_t;
#define LM_SAVE "s_mov_b64 s[96:97], exec\n\ts_mov_b64 exec, %[m]\n\t"
#define LM_REST "\n\ts_mov_b64 exec, s[96:97]"
#define LM_CLOB "s96", "s97"
#define LM_MC "s"
#else
typedef uint32_t mask_t;
#define LM_SAVE "s_mov_b32 s96, exec_lo\n\ts_mov_b32 exec_lo, %[m]\n\t"
#define LM_REST "\n\ts_mov_b32 exec_lo, s96"
#define LM_CLOB "s96"
#define LM_MC "s"
#endif
#define LM_R8(x) x x x x x x x x
#define LM_R16(x) LM_R8(x) LM_R8(x)
#define LM_R48(x) LM_R16(x) LM_R16(x) LM_R16(x)
#define LM_CYC(r) "s_getreg_b32 " r ", hwreg(HW_REG_SHADER_CYCLES_LO)\n\t"

// kind: 0 dense (C separate in the mask test, D = C chain in the timing), 1 sparse (C is D, index in VGPR), 2 dot.
// fa, fb, fc: data kinds (h FP16, b BF16, 8 E4M3, e E5M2, i small 8-bit integers, 4 small nibbles, f FP32,
// H FP16 accumulator pairs, B BF16 accumulator pairs, I int32 accumulator).
#define LMOP(NAME, NA, NB, NC, KIND, FA, FB, FC, CYC, OPC, MODS)                                                         \
    struct NAME {                                                                                                \
        typedef VT<W(NA)>::t ta;                                                                                 \
        typedef VT<W(NB)>::t tb;                                                                                 \
        typedef VT<W(NC)>::t tc;                                                                                 \
        static constexpr int kind = KIND, cyc = CYC;                                                             \
        static constexpr const char * fmt = FA FB FC;                                                            \
        static __device__ __forceinline__ void go(ta a, tb b, tc c, tc & d, int i, mask_t m) {                  \
            if constexpr (KIND == 1) {                                                                           \
                asm volatile(LM_SAVE OPC " %[d], %[a], %[b], %[i]" MODS LM_REST                                       \
                             : [d] "+v"(d) : [a] "v"(a), [b] "v"(b), [i] "v"(i), [m] LM_MC(m) : LM_CLOB);       \
            } else {                                                                                             \
                asm volatile(LM_SAVE OPC " %[d], %[a], %[b], %[c]" MODS LM_REST                                       \
                             : [d] "+v"(d) : [a] "v"(a), [b] "v"(b), [c] "v"(c), [m] LM_MC(m) : LM_CLOB);       \
            }                                                                                                    \
        }                                                                                                        \
        static __device__ __forceinline__ uint32_t time(ta a, tb b, tc & d, int i, mask_t m, bool longer) {     \
            uint32_t t0, t1;                                                                                     \
            if (longer) {                                                                                        \
                asm volatile(LM_SAVE LM_CYC("%[t0]") LM_R48(OPC " %[d], %[a], %[b], " TAIL##KIND MODS "\n\t")         \
                                 LM_CYC("%[t1]") "s_nop 0" LM_REST                                               \
                             : [d] "+v"(d), [t0] "=&s"(t0), [t1] "=&s"(t1)                                       \
                             : [a] "v"(a), [b] "v"(b), [i] "v"(i), [m] LM_MC(m) : LM_CLOB);                     \
            } else {                                                                                             \
                asm volatile(LM_SAVE LM_CYC("%[t0]") LM_R16(OPC " %[d], %[a], %[b], " TAIL##KIND MODS "\n\t")         \
                                 LM_CYC("%[t1]") "s_nop 0" LM_REST                                               \
                             : [d] "+v"(d), [t0] "=&s"(t0), [t1] "=&s"(t1)                                       \
                             : [a] "v"(a), [b] "v"(b), [i] "v"(i), [m] LM_MC(m) : LM_CLOB);                     \
            }                                                                                                    \
            return t1 - t0;                                                                                      \
        }                                                                                                        \
    };
#define TAIL0 "%[d]"
#define TAIL1 "%[i]"
#define TAIL2 "%[d]"

LMOP(L_f16, 4, 4, 8, 0, "h", "h", "f", 16, "v_wmma_f32_16x16x16_f16", "")
LMOP(L_bf16, 4, 4, 8, 0, "b", "b", "f", 16, "v_wmma_f32_16x16x16_bf16", "")
LMOP(L_f16h, 4, 4, 4, 0, "h", "h", "H", 16, "v_wmma_f16_16x16x16_f16", "")
LMOP(L_bf16h, 4, 4, 4, 0, "b", "b", "B", 16, "v_wmma_bf16_16x16x16_bf16", "")
LMOP(L_fp8, 2, 2, 8, 0, "8", "8", "f", 8, "v_wmma_f32_16x16x16_fp8_fp8", "")
LMOP(L_fp8bf8, 2, 2, 8, 0, "8", "e", "f", 8, "v_wmma_f32_16x16x16_fp8_bf8", "")
LMOP(L_bf8fp8, 2, 2, 8, 0, "e", "8", "f", 8, "v_wmma_f32_16x16x16_bf8_fp8", "")
LMOP(L_bf8, 2, 2, 8, 0, "e", "e", "f", 8, "v_wmma_f32_16x16x16_bf8_bf8", "")
LMOP(L_iu8, 2, 2, 8, 0, "i", "i", "I", 8, "v_wmma_i32_16x16x16_iu8", " neg_lo:[1,1,0]")
LMOP(L_iu4, 1, 1, 8, 0, "4", "4", "I", 8, "v_wmma_i32_16x16x16_iu4", "")
LMOP(L_iu4k32, 2, 2, 8, 0, "4", "4", "I", 8, "v_wmma_i32_16x16x32_iu4", "")
LMOP(L_sf16, 4, 8, 8, 1, "h", "h", "f", 16, "v_swmmac_f32_16x16x32_f16", "")
LMOP(L_sbf16, 4, 8, 8, 1, "b", "b", "f", 16, "v_swmmac_f32_16x16x32_bf16", "")
LMOP(L_sf16h, 4, 8, 4, 1, "h", "h", "H", 16, "v_swmmac_f16_16x16x32_f16", "")
LMOP(L_sbf16h, 4, 8, 4, 1, "b", "b", "B", 16, "v_swmmac_bf16_16x16x32_bf16", "")
LMOP(L_sfp8, 2, 4, 8, 1, "8", "8", "f", 8, "v_swmmac_f32_16x16x32_fp8_fp8", "")
LMOP(L_sbf8, 2, 4, 8, 1, "e", "e", "f", 8, "v_swmmac_f32_16x16x32_bf8_bf8", "")
LMOP(L_siu8, 2, 4, 8, 1, "i", "i", "I", 8, "v_swmmac_i32_16x16x32_iu8", "")
LMOP(L_siu4, 1, 2, 8, 1, "4", "4", "I", 8, "v_swmmac_i32_16x16x32_iu4", "")
LMOP(L_siu4k64, 2, 4, 8, 1, "4", "4", "I", 8, "v_swmmac_i32_16x16x64_iu4", "")
LMOP(L_dot4, 1, 1, 1, 2, "8", "8", "f", 0, "v_dot4_f32_fp8_fp8", "")
LMOP(L_dot2, 1, 1, 1, 2, "h", "h", "f", 0, "v_dot2_f32_f16", "")

template <class OP>
__global__ __launch_bounds__(LM_WAVE) void mk(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                             const uint32_t * __restrict__ C, const uint32_t * __restrict__ I,
                                             uint32_t * __restrict__ D, mask_t mask) {
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

// Timing: 7 repetitions each of the 16- and 48-instruction chains, alternating; out[0..6] short, out[7..13] long.
template <class OP>
__global__ __launch_bounds__(LM_WAVE) void tk(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                             const uint32_t * __restrict__ C, const uint32_t * __restrict__ I,
                                             uint32_t * __restrict__ D, uint32_t * __restrict__ out, mask_t mask) {
    typedef typename OP::ta TA;
    typedef typename OP::tb TB;
    typedef typename OP::tc TC;
    const size_t lane = threadIdx.x;
    TA a;
    TB b;
    TC d;
    __builtin_memcpy(&a, A + lane * (sizeof(TA) / 4), sizeof a);
    __builtin_memcpy(&b, B + lane * (sizeof(TB) / 4), sizeof b);
    __builtin_memcpy(&d, C + lane * (sizeof(TC) / 4), sizeof d);
    const int i = (int) I[lane];
    for (int r = 0; r < 7; ++r) {
        const uint32_t s = OP::time(a, b, d, i, mask, false);
        const uint32_t l = OP::time(a, b, d, i, mask, true);
        if (lane == 0) {
            out[r] = s;
            out[7 + r] = l;
        }
    }
    __builtin_memcpy(D + lane * (sizeof(TC) / 4), &d, sizeof d);
}

typedef void (*MK)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *, mask_t);
typedef void (*TK)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *, uint32_t *, mask_t);
struct Op {
    const char * name;
    MK k;
    TK t;
    int na, nb, nc, kind, cyc;
    const char * fmt;
};
#define LMREG(NAME) \
    Op { #NAME, mk<NAME>, tk<NAME>, (int) sizeof(NAME::ta) / 4, (int) sizeof(NAME::tb) / 4, (int) sizeof(NAME::tc) / 4, NAME::kind, NAME::cyc, NAME::fmt }

inline std::vector<Op> ops() {
    return {LMREG(L_f16),   LMREG(L_bf16),  LMREG(L_f16h),  LMREG(L_bf16h),   LMREG(L_fp8),   LMREG(L_fp8bf8),
            LMREG(L_bf8fp8), LMREG(L_bf8),  LMREG(L_iu8),   LMREG(L_iu4),     LMREG(L_iu4k32), LMREG(L_sf16),
            LMREG(L_sbf16), LMREG(L_sf16h), LMREG(L_sbf16h), LMREG(L_sfp8),   LMREG(L_sbf8),  LMREG(L_siu8),
            LMREG(L_siu4),  LMREG(L_siu4k64), LMREG(L_dot4), LMREG(L_dot2)};
}

inline uint32_t rnd_word(char f, std::mt19937_64 & rng) {
    uint32_t v = 0;
    switch (f) {
    case 'h':  // FP16 0.5 .. 2
        return (uint32_t) (0x3800 + (rng() % 0x800)) | (uint32_t) (0x3800 + (rng() % 0x800)) << 16;
    case 'b':  // BF16 0.5 .. 2
        return (uint32_t) (0x3F00 + (rng() % 0x100)) | (uint32_t) (0x3F00 + (rng() % 0x100)) << 16;
    case '8':  // E4M3 0.5 .. 1.9
        for (int k = 0; k < 4; ++k) v |= (uint32_t) (0x30 + (rng() % 16)) << (8 * k);
        return v;
    case 'e':  // E5M2 0.5 .. 1.75
        for (int k = 0; k < 4; ++k) v |= (uint32_t) (0x38 + (rng() % 8)) << (8 * k);
        return v;
    case 'i':  // small signed bytes, nonzero sums
        return (uint32_t) rng() & 0x3F3F3F3Fu;
    case '4':  // nibbles 0 .. 3
        return (uint32_t) rng() & 0x33333333u;
    case 'f':  // FP32 0.25 .. 0.5
        return 0x3E800000u + (uint32_t) (rng() & 0x7FFFFF);
    case 'H':
        return 0x34003400u;
    case 'B':
        return 0x3E803E80u;
    default:  // 'I'
        return (uint32_t) (rng() % 100);
    }
}

inline std::vector<uint32_t> run_mask(const Op & o, const std::vector<uint32_t> & a, const std::vector<uint32_t> & b,
                                      const std::vector<uint32_t> & c, const std::vector<uint32_t> & idx,
                                      const std::vector<uint32_t> & d0, mask_t mask) {
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
    hipLaunchKernelGGL(o.k, dim3(1), dim3(LM_WAVE), 0, 0, da, db, dc, di, dd, mask);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(d0.size());
    CHECK(hipMemcpy(out.data(), dd, out.size() * 4, hipMemcpyDeviceToHost));
    for (auto * p : {da, db, dc, di, dd}) {
        CHECK(hipFree(p));
    }
    return out;
}

// Cycles per chained instruction ((best t48 - best t16) / 32) under `mask`.
inline double run_time(const Op & o, const std::vector<uint32_t> & a, const std::vector<uint32_t> & b,
                       const std::vector<uint32_t> & c, const std::vector<uint32_t> & idx, mask_t mask) {
    uint32_t *da, *db, *dc, *di, *dd, *dout;
    CHECK(hipMalloc(&da, a.size() * 4));
    CHECK(hipMalloc(&db, b.size() * 4));
    CHECK(hipMalloc(&dc, c.size() * 4));
    CHECK(hipMalloc(&di, idx.size() * 4));
    CHECK(hipMalloc(&dd, c.size() * 4));
    CHECK(hipMalloc(&dout, 14 * 4));
    CHECK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, c.data(), c.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(di, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(o.t, dim3(1), dim3(LM_WAVE), 0, 0, da, db, dc, di, dd, dout, mask);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint32_t t[14];
    CHECK(hipMemcpy(t, dout, sizeof t, hipMemcpyDeviceToHost));
    for (auto * p : {da, db, dc, di, dd, dout}) {
        CHECK(hipFree(p));
    }
    // Disturbances only add cycles (the desktop's waves, the card-wide stall), so the best sample of each length is
    // used (STUDY-PLAN, disturbed timings: "the best sample for a minimum").
    const uint32_t s = *std::min_element(t, t + 7), l = *std::min_element(t + 7, t + 14);
    return ((double) l - (double) s) / 32.0;
}

// Returns true when every check holds. text receives a one-line summary. brk: the dot instructions are expected to
// ignore EXEC as well (the deliberate break).
inline bool mask_test(bool brk, std::string & text) {
    bool ok = true;
    const mask_t ones = ~(mask_t) 0;
    std::vector<mask_t> masks = {ones, (mask_t) (ones >> (LM_WAVE / 2)), (mask_t) (ones << (LM_WAVE / 2)),
                                 (mask_t) 0x5555555555555555ull, (mask_t) 0x1111111111111111ull, (mask_t) 1, (mask_t) 0};
    std::mt19937_64 rng(2222 + LM_WAVE);
    int n_ignore = 0, n_obey = 0, n_time_ok = 0, n_timed = 0;
    for (const Op & o : ops()) {
        const char fa = o.fmt[0], fb = o.fmt[1], fc = o.fmt[2];
        std::vector<uint32_t> a((size_t) LM_WAVE * o.na), b((size_t) LM_WAVE * o.nb), c((size_t) LM_WAVE * o.nc),
            idx(LM_WAVE), d0((size_t) LM_WAVE * o.nc);
        for (auto & x : a) x = rnd_word(fa, rng);
        for (auto & x : b) x = rnd_word(fb, rng);
        for (auto & x : idx) x = (uint32_t) rng();
        for (size_t k = 0; k < c.size(); ++k) {
            c[k] = rnd_word(fc, rng);
            d0[k] = 0xDEAD0000u | (uint32_t) k;  // sentinel
        }
        if (o.kind == 1) {
            d0 = c;  // C is D: the untouched C is the sentinel
        }
        const std::vector<uint32_t> ref = run_mask(o, a, b, c, idx, d0, ones);
        long ref_unwritten = 0;
        for (size_t k = 0; k < ref.size(); ++k) {
            ref_unwritten += ref[k] == d0[k];
        }
        bool op_ok = ref_unwritten == 0;
        for (mask_t m : masks) {
            const std::vector<uint32_t> out = run_mask(o, a, b, c, idx, d0, m);
            const std::vector<uint32_t> out2 = run_mask(o, a, b, c, idx, d0, m);
            long eq_ref = 0, off = 0, off_written = 0, on = 0, on_right = 0;
            for (int l = 0; l < LM_WAVE; ++l) {
                const bool en = (m >> l) & 1;
                for (int k = 0; k < o.nc; ++k) {
                    const size_t x = (size_t) l * o.nc + k;
                    eq_ref += out[x] == ref[x];
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
            std::printf("RESULT mask w%d %-10s EXEC %016llx words %zu equal_to_full_EXEC %ld | enabled lanes' words %ld right %ld | "
                        "disabled lanes' words %ld written %ld | repeat_identical %d | full-EXEC words equal to the sentinel %ld\n",
                        LM_WAVE, o.name, (unsigned long long) m, out.size(), eq_ref, on, on_right, off, off_written,
                        (int) repeat, ref_unwritten);
            const bool ignore = o.kind != 2 || brk;
            const bool good = ignore ? (eq_ref == (long) out.size() && off_written == off) : (on_right == on && off_written == 0);
            op_ok = op_ok && good && repeat;
        }
        if (o.kind == 2) {
            n_obey += op_ok;
        } else {
            n_ignore += op_ok;
            // timing: full EXEC against lane 0 only; the full-EXEC chain must match the known cost
            const double tf = run_time(o, a, b, c, idx, ones), t1 = run_time(o, a, b, c, idx, (mask_t) 1);
            const bool known = std::fabs(tf - o.cyc) <= 0.3, same = std::fabs(tf - t1) <= 0.05;
            std::printf("RESULT mask_time w%d %-10s cycles_per_chained_instruction full_EXEC %.3f one_lane %.3f known %d "
                        "(within 0.3: %d; equal within 0.05: %d)\n",
                        LM_WAVE, o.name, tf, t1, o.cyc, (int) known, (int) same);
            ++n_timed;
            n_time_ok += known && same;
            op_ok = op_ok && known && same;
        }
        ok = ok && op_ok;
    }
    char buf[300];
    std::snprintf(buf, sizeof buf,
                  "wave%d: %d of 20 matrix forms read and write every lane under all 7 masks, %d of 2 dot controls obey "
                  "the mask; %d of %d chained times equal the known cost and are the same with one lane",
                  LM_WAVE, n_ignore, n_obey, n_time_ok, n_timed);
    text = buf;
    return ok;
}

}  // namespace lm
