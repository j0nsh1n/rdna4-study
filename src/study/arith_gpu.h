// Lane 5 (matrix arithmetic): the GPU side shared by the arith_* probes. One wave runs one 16x16x16 problem (or a
// chain of them through C). Every matrix instruction is written in inline assembly so the modifier bits can be set
// and so a MODE write before it cannot be reordered around it (volatile asm statements keep their order; a builtin
// could move).
//
// Register images (gfx1201, wave32; layout from src/wmma_exact.hip, confirmed there with identity matrices):
//   slot s = 0..15 of an A row r / B column c: lane r + 16 * (s >> 3);
//     16-bit data: VGPR (s & 7) >> 1, half (s & 1);   8-bit data: VGPR (s & 7) >> 2, byte (s & 3);
//     4-bit data (IU4, K = 16): VGPR 0, nibble (s & 7).
//   C and D, 32-bit: element (r, c) in lane 16 * (r >> 3) + c, VGPR r & 7. 16-bit accumulators: the same lane,
//     VGPR (r & 7) >> 1, half (r & 1) (checked by arith_layout: C passes through unchanged with A = 0).
// "Slot" is where a value sits; which K index it stands for is a labelling question (see arith_model.h).
#pragma once
#include "../harness.h"

#include <string>

namespace arith {

enum Form { F16 = 0, BF16, FF, FB, BF, BB, F16H, BF16H, IU8, IU4, IU4K32, NFORMS };
static const char * const form_name[NFORMS] = {"f16",   "bf16",   "fp8_fp8", "fp8_bf8", "bf8_fp8",
                                               "bf8_bf8", "f16_f16", "bf16_bf16", "iu8",   "iu4", "iu4_k32"};

// Modifier variants (index into the assembly suffix table). 16-bit float forms: 0 none, 1 NEG A, 2 NEG_HI A,
// 3 NEG B, 4 NEG_HI B, 5 NEG C, 6 ABS C (NEG_HI[2]), 7 NEG and ABS C. Integer forms: the NEG bits are signedness
// flags: 0 unsigned x unsigned, 1 signed A, 3 signed B, 8 signed both, then +clamp: 9 (none signed), 10 (both),
// 11 (signed A), 12 (signed B).
enum Mod { M_NONE = 0, M_NEGA, M_NEGHIA, M_NEGB, M_NEGHIB, M_NEGC, M_ABSC, M_NEGABSC, M_SAB, M_CLAMP, M_SABC, M_SAC, M_SBC,
           NMODS };
static const char * const mod_name[NMODS] = {"none",   "negA",  "neghiA",  "negB", "neghiB", "negC",
                                             "absC",   "negabsC", "sAB",   "clamp", "sABclamp", "sAclamp", "sBclamp"};

#define ARITH_MODSTR_0 ""
#define ARITH_MODSTR_1 " neg_lo:[1,0,0]"
#define ARITH_MODSTR_2 " neg_hi:[1,0,0]"
#define ARITH_MODSTR_3 " neg_lo:[0,1,0]"
#define ARITH_MODSTR_4 " neg_hi:[0,1,0]"
#define ARITH_MODSTR_5 " neg_lo:[0,0,1]"
#define ARITH_MODSTR_6 " neg_hi:[0,0,1]"
#define ARITH_MODSTR_7 " neg_lo:[0,0,1] neg_hi:[0,0,1]"
#define ARITH_MODSTR_8 " neg_lo:[1,1,0]"
#define ARITH_MODSTR_9 " clamp"
#define ARITH_MODSTR_10 " neg_lo:[1,1,0] clamp"
#define ARITH_MODSTR_11 " neg_lo:[1,0,0] clamp"
#define ARITH_MODSTR_12 " neg_lo:[0,1,0] clamp"

inline int words_ab(int f) { return f == IU4 ? 1 : (f == FF || f == FB || f == BF || f == BB || f == IU8 || f == IU4K32) ? 2 : 4; }
inline int elem_bits(int f) { return (f == IU4 || f == IU4K32) ? 4 : words_ab(f) == 2 ? 8 : 16; }
inline bool acc16(int f) { return f == F16H || f == BF16H; }

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));

template <int F>
struct Types;
template <> struct Types<F16> { typedef v4i ab; typedef v8i cd; };
template <> struct Types<BF16> { typedef v4i ab; typedef v8i cd; };
template <> struct Types<F16H> { typedef v4i ab; typedef v4i cd; };
template <> struct Types<BF16H> { typedef v4i ab; typedef v4i cd; };
template <> struct Types<FF> { typedef v2i ab; typedef v8i cd; };
template <> struct Types<FB> { typedef v2i ab; typedef v8i cd; };
template <> struct Types<BF> { typedef v2i ab; typedef v8i cd; };
template <> struct Types<BB> { typedef v2i ab; typedef v8i cd; };
template <> struct Types<IU8> { typedef v2i ab; typedef v8i cd; };
template <> struct Types<IU4> { typedef int ab; typedef v8i cd; };
template <> struct Types<IU4K32> { typedef v2i ab; typedef v8i cd; };

// Instruction text per form. A string literal is needed inside asm(), so the form and modifier select a
// specialisation through a switch of string-literal macros.
#define ARITH_INSTR_F16 "v_wmma_f32_16x16x16_f16"
#define ARITH_INSTR_BF16 "v_wmma_f32_16x16x16_bf16"
#define ARITH_INSTR_FF "v_wmma_f32_16x16x16_fp8_fp8"
#define ARITH_INSTR_FB "v_wmma_f32_16x16x16_fp8_bf8"
#define ARITH_INSTR_BF "v_wmma_f32_16x16x16_bf8_fp8"
#define ARITH_INSTR_BB "v_wmma_f32_16x16x16_bf8_bf8"
#define ARITH_INSTR_F16H "v_wmma_f16_16x16x16_f16"
#define ARITH_INSTR_BF16H "v_wmma_bf16_16x16x16_bf16"
#define ARITH_INSTR_IU8 "v_wmma_i32_16x16x16_iu8"
#define ARITH_INSTR_IU4 "v_wmma_i32_16x16x16_iu4"
#define ARITH_INSTR_IU4K32 "v_wmma_i32_16x16x32_iu4"

#define ARITH_CASE(F, M)                                                                                         \
    if constexpr (FORM == F && MOD == M) {                                                                       \
        asm volatile(ARITH_INSTR_##F " %0, %1, %2, %3" ARITH_MODSTR_##M : "=&v"(d) : "v"(a), "v"(b), "v"(c));    \
    }
#define ARITH_FLOAT16_MODS(F) \
    ARITH_CASE(F, 0) ARITH_CASE(F, 1) ARITH_CASE(F, 2) ARITH_CASE(F, 3) ARITH_CASE(F, 4) ARITH_CASE(F, 5) \
    ARITH_CASE(F, 6) ARITH_CASE(F, 7)
#define ARITH_INT_MODS(F) ARITH_CASE(F, 0) ARITH_CASE(F, 1) ARITH_CASE(F, 3) ARITH_CASE(F, 8) ARITH_CASE(F, 9) ARITH_CASE(F, 10) \
    ARITH_CASE(F, 11) ARITH_CASE(F, 12)

template <int FORM, int MOD>
__device__ __forceinline__ typename Types<FORM>::cd mm(typename Types<FORM>::ab a, typename Types<FORM>::ab b,
                                                       typename Types<FORM>::cd c) {
    typename Types<FORM>::cd d;
    ARITH_FLOAT16_MODS(F16)
    ARITH_FLOAT16_MODS(BF16)
    ARITH_FLOAT16_MODS(F16H)
    ARITH_FLOAT16_MODS(BF16H)
    // FP8/BF8: the guide allows no modifier on A and B; NEG and ABS on C are tried (5, 6, 7).
    ARITH_CASE(FF, 0) ARITH_CASE(FF, 5) ARITH_CASE(FF, 6) ARITH_CASE(FF, 7)
    ARITH_CASE(FB, 0) ARITH_CASE(BF, 0) ARITH_CASE(BB, 0)
    ARITH_INT_MODS(IU8)
    ARITH_INT_MODS(IU4)
    ARITH_INT_MODS(IU4K32)
    return d;
}

// MODE: bits [7:0] (FP_ROUND, FP_DENORM) and bit 23 (FP16_OVFL) are replaced by `mode`; the rest is kept. The old
// value is returned for the restore. The s_nops give the write time to reach the VALU before the next instruction.
constexpr uint32_t MODE_MASK = 0x008000FFu;

__device__ __forceinline__ uint32_t mode_set(uint32_t mode) {
    uint32_t old, nw;
    asm volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_MODE)\n"
        "s_and_b32 %1, %0, %3\n"
        "s_or_b32 %1, %1, %2\n"
        "s_setreg_b32 hwreg(HW_REG_MODE), %1\n"
        "s_nop 7\ns_nop 7"
        : "=&s"(old), "=&s"(nw)
        : "s"(mode & MODE_MASK), "s"(~MODE_MASK)
        : "scc");
    return old;
}

__device__ __forceinline__ void mode_restore(uint32_t old) {
    asm volatile("s_nop 7\ns_setreg_b32 hwreg(HW_REG_MODE), %0\ns_nop 7" ::"s"(old));
}

__device__ __forceinline__ uint32_t mode_read() {
    uint32_t m;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE)" : "=s"(m));
    return m;
}

// One wave per problem; `steps` instructions chained through C. A, B: [wave][step][lane][words];
// C, D: [wave][lane][8] (16-bit accumulators use the first 4 words). use_mode: write MODE around the chain.
// mode_seen[wave] receives the MODE value read back after the write (0xFFFFFFFF when MODE is not touched).
template <int FORM, int MOD>
__global__ __launch_bounds__(32) void mm_kernel(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                               const uint32_t * __restrict__ C, uint32_t * __restrict__ D, int steps,
                                               int use_mode, uint32_t mode, uint32_t * __restrict__ mode_seen) {
    typedef typename Types<FORM>::ab TAB;
    typedef typename Types<FORM>::cd TCD;
    constexpr int NW = sizeof(TAB) / 4, NC = sizeof(TCD) / 4;
    const int w = blockIdx.x, lane = threadIdx.x;
    TCD c;
    __builtin_memcpy(&c, C + ((size_t) w * 32 + lane) * 8, sizeof c);
    uint32_t old = 0;
    if (use_mode) {
        old = mode_set(mode);
        if (lane == 0) {
            mode_seen[w] = mode_read();
        }
    }
    for (int s = 0; s < steps; ++s) {
        TAB a, b;
        const size_t o = (((size_t) w * steps + s) * 32 + lane) * NW;
        __builtin_memcpy(&a, A + o, sizeof a);
        __builtin_memcpy(&b, B + o, sizeof b);
        c = mm<FORM, MOD>(a, b, c);
    }
    if (use_mode) {
        mode_restore(old);
    }
    uint32_t out[8] = {};
    __builtin_memcpy(out, &c, sizeof c);
    for (int i = 0; i < NC; ++i) {
        D[((size_t) w * 32 + lane) * 8 + i] = out[i];
    }
}

// The MODE control: VALU operations on given inputs under the same MODE write. x[0..7] floats; out:
// 0 v_add_f32(x0, x1), 1 v_mul_f32(x2, x3), 2 v_add_f16(lo16 x4, lo16 x5), 3 v_mul_f32(x6, x7),
// 4 v_add_f16(lo16 x6, lo16 x7), 5 MODE as read back.
__global__ __launch_bounds__(32) void valu_kernel(const uint32_t * __restrict__ x, uint32_t * __restrict__ out,
                                                 int use_mode, uint32_t mode) {
    uint32_t r0, r1, r2, r3, r4;
    const uint32_t x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
    uint32_t old = 0;
    if (use_mode) {
        old = mode_set(mode);
    }
    const uint32_t seen = mode_read();
    asm volatile(
        "v_add_f32 %0, %5, %6\n"
        "v_mul_f32 %1, %7, %8\n"
        "v_add_f16 %2, %9, %10\n"
        "v_mul_f32 %3, %11, %12\n"
        "v_add_f16 %4, %11, %12\n"
        : "=&v"(r0), "=&v"(r1), "=&v"(r2), "=&v"(r3), "=&v"(r4)
        : "v"(x0), "v"(x1), "v"(x2), "v"(x3), "v"(x4), "v"(x5), "v"(x6), "v"(x7));
    if (use_mode) {
        mode_restore(old);
    }
    if (threadIdx.x == 0) {
        out[0] = r0;
        out[1] = r1;
        out[2] = r2 & 0xFFFF;
        out[3] = r3;
        out[4] = r4 & 0xFFFF;
        out[5] = seen;
    }
}

typedef void (*Kern)(const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *, int, int, uint32_t, uint32_t *);

template <int F>
inline Kern kern_for_mod(int m) {
    switch (m) {
#define ARITH_KM(M) \
    case M:         \
        return mm_kernel<F, M>;
        ARITH_KM(0) ARITH_KM(1) ARITH_KM(2) ARITH_KM(3) ARITH_KM(4) ARITH_KM(5) ARITH_KM(6) ARITH_KM(7) ARITH_KM(8)
        ARITH_KM(9) ARITH_KM(10) ARITH_KM(11) ARITH_KM(12)
#undef ARITH_KM
    }
    return nullptr;
}

inline Kern kern_for(int f, int m) {
    switch (f) {
    case F16: return kern_for_mod<F16>(m);
    case BF16: return kern_for_mod<BF16>(m);
    case FF: return kern_for_mod<FF>(m);
    case FB: return kern_for_mod<FB>(m);
    case BF: return kern_for_mod<BF>(m);
    case BB: return kern_for_mod<BB>(m);
    case F16H: return kern_for_mod<F16H>(m);
    case BF16H: return kern_for_mod<BF16H>(m);
    case IU8: return kern_for_mod<IU8>(m);
    case IU4: return kern_for_mod<IU4>(m);
    case IU4K32: return kern_for_mod<IU4K32>(m);
    }
    return nullptr;
}

// Which (form, modifier) pairs exist as kernels with a real instruction (others assemble to nothing).
inline bool mod_valid(int f, int m) {
    if (f == F16 || f == BF16 || f == F16H || f == BF16H) {
        return m <= M_NEGABSC;
    }
    if (f == FF) {
        return m == M_NONE || m == M_NEGC || m == M_ABSC || m == M_NEGABSC;
    }
    if (f == FB || f == BF || f == BB) {
        return m == M_NONE;
    }
    return m == M_NONE || m == M_NEGA || m == M_NEGB || m == M_SAB || m == M_CLAMP || m == M_SABC || m == M_SAC ||
           m == M_SBC;
}

// ---------------------------------------------------------------------------------------------------------------
// Host: images and launches.

struct Slot {
    int lane, word, bit;
};

// Slot s of A row r (or B column r). IU4 with K = 32: slots 0..31, lane r + 16 * (s >> 4), VGPR (s & 15) >> 3,
// nibble s & 7 (an assumption the integer sums cannot tell from another layout that pairs A and B alike).
inline Slot slot_ab(int f, int r, int s) {
    if (f == IU4K32) {
        return {r + 16 * (s >> 4), (s & 15) >> 3, (s & 7) * 4};
    }
    const int lane = r + 16 * (s >> 3), e = s & 7;
    if (f == IU4) {
        return {lane, 0, e * 4};
    }
    if (elem_bits(f) == 8) {
        return {lane, e >> 2, (e & 3) * 8};
    }
    return {lane, e >> 1, (e & 1) * 16};
}

inline Slot slot_cd(int f, int r, int c) {
    const int lane = 16 * (r >> 3) + c, e = r & 7;
    if (acc16(f)) {
        return {lane, e >> 1, (e & 1) * 16};
    }
    return {lane, e, 0};
}

inline void put_bits(uint32_t * img, const Slot & s, int nw, uint32_t bits, int eb) {
    uint32_t & w = img[s.lane * nw + s.word];
    const uint32_t mask = (eb == 32 ? 0xFFFFFFFFu : ((1u << eb) - 1)) << s.bit;
    w = (w & ~mask) | ((bits << s.bit) & mask);
}

inline uint32_t get_bits(const uint32_t * img, const Slot & s, int nw, int eb) {
    const uint32_t v = img[s.lane * nw + s.word] >> s.bit;
    return eb == 32 ? v : v & ((1u << eb) - 1);
}

// Problems for one launch: `waves` waves of `steps` instructions each.
struct Batch {
    int form = F16, mod = M_NONE, waves = 0, steps = 1;
    std::vector<uint32_t> a, b, c, d;
    std::vector<uint32_t> mode_seen;
    void init(int f, int nwaves, int nsteps) {
        form = f;
        waves = nwaves;
        steps = nsteps;
        const int nw = words_ab(f);
        a.assign((size_t) waves * steps * 32 * nw, 0);
        b.assign(a.size(), 0);
        c.assign((size_t) waves * 256, 0);
        d.assign(c.size(), 0);
    }
    uint32_t * a_img(int w, int s) { return &a[(((size_t) w * steps + s) * 32) * words_ab(form)]; }
    uint32_t * b_img(int w, int s) { return &b[(((size_t) w * steps + s) * 32) * words_ab(form)]; }
    void set_a(int w, int s, int r, int slot, uint32_t bits) {
        put_bits(a_img(w, s), slot_ab(form, r, slot), words_ab(form), bits, elem_bits(form));
    }
    void set_b(int w, int s, int col, int slot, uint32_t bits) {
        put_bits(b_img(w, s), slot_ab(form, col, slot), words_ab(form), bits, elem_bits(form));
    }
    int cbits() const { return acc16(form) ? 16 : 32; }
    void set_c(int w, int r, int col, uint32_t bits) {
        put_bits(&c[(size_t) w * 256], slot_cd(form, r, col), 8, bits, cbits());
    }
    uint32_t get_d(int w, int r, int col) const { return get_bits(&d[(size_t) w * 256], slot_cd(form, r, col), 8, cbits()); }
};

// Runs a batch on the card. use_mode/mode: see mm_kernel.
inline void run(Batch & bt, bool use_mode = false, uint32_t mode = 0) {
    uint32_t *da, *db, *dc, *dd, *dm;
    CHECK(hipMalloc(&da, bt.a.size() * 4));
    CHECK(hipMalloc(&db, bt.b.size() * 4));
    CHECK(hipMalloc(&dc, bt.c.size() * 4));
    CHECK(hipMalloc(&dd, bt.c.size() * 4));
    CHECK(hipMalloc(&dm, (size_t) bt.waves * 4));
    CHECK(hipMemcpy(da, bt.a.data(), bt.a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, bt.b.data(), bt.b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, bt.c.data(), bt.c.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dd, 0xEE, bt.c.size() * 4));
    CHECK(hipMemset(dm, 0xFF, (size_t) bt.waves * 4));
    Kern k = kern_for(bt.form, bt.mod);
    if (!k || !mod_valid(bt.form, bt.mod)) {
        std::fprintf(stderr, "no kernel for form %d modifier %d\n", bt.form, bt.mod);
        std::exit(2);
    }
    hipLaunchKernelGGL(k, dim3(bt.waves), dim3(32), 0, 0, da, db, dc, dd, bt.steps, use_mode ? 1 : 0, mode, dm);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    bt.d.resize(bt.c.size());
    CHECK(hipMemcpy(bt.d.data(), dd, bt.d.size() * 4, hipMemcpyDeviceToHost));
    bt.mode_seen.resize(bt.waves);
    CHECK(hipMemcpy(bt.mode_seen.data(), dm, (size_t) bt.waves * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(da));
    CHECK(hipFree(db));
    CHECK(hipFree(dc));
    CHECK(hipFree(dd));
    CHECK(hipFree(dm));
}

// Runs the VALU control: returns the 6 output words of valu_kernel.
inline std::vector<uint32_t> run_valu(const uint32_t x[8], bool use_mode, uint32_t mode) {
    uint32_t *dx, *dout;
    CHECK(hipMalloc(&dx, 32));
    CHECK(hipMalloc(&dout, 32));
    CHECK(hipMemcpy(dx, x, 32, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(valu_kernel, dim3(1), dim3(32), 0, 0, dx, dout, use_mode ? 1 : 0, mode);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> o(8);
    CHECK(hipMemcpy(o.data(), dout, 32, hipMemcpyDeviceToHost));
    CHECK(hipFree(dx));
    CHECK(hipFree(dout));
    o.resize(6);
    return o;
}

}  // namespace arith

// ---------------------------------------------------------------------------------------------------------------
// Sparse forms (SWMMAC, lane 5 item 3). A holds 16 stored elements per row (the dense A image of the same input
// width); B is K = 32 dense (16-bit: 8 VGPRs, 8-bit: 4 VGPRs); C is also D; the index is one VGPR per lane and
// index_key selects which part of it is used.
namespace arith {

enum SForm { S_F16 = 0, S_BF16, S_FF, S_F16H, S_BF16H, S_FB, S_BF, S_BB, S_IU8, NSFORMS };
static const char * const sform_name[NSFORMS] = {"swmmac_f16",     "swmmac_bf16",    "swmmac_fp8_fp8", "swmmac_f16_f16",
                                                 "swmmac_bf16_bf16", "swmmac_fp8_bf8", "swmmac_bf8_fp8", "swmmac_bf8_bf8",
                                                 "swmmac_iu8"};
inline int sdense(int sf) {
    static const int t[] = {F16, BF16, FF, F16H, BF16H, FB, BF, BB, IU8};
    return t[sf];
}
inline int swords_b(int sf) { return elem_bits(sdense(sf)) == 8 ? 4 : 8; }

template <int SF> struct STypes;
template <> struct STypes<S_F16> { typedef v4i a; typedef v8i b; typedef v8i cd; };
template <> struct STypes<S_BF16> { typedef v4i a; typedef v8i b; typedef v8i cd; };
template <> struct STypes<S_FF> { typedef v2i a; typedef v4i b; typedef v8i cd; };
template <> struct STypes<S_F16H> { typedef v4i a; typedef v8i b; typedef v4i cd; };
template <> struct STypes<S_BF16H> { typedef v4i a; typedef v8i b; typedef v4i cd; };
template <> struct STypes<S_FB> { typedef v2i a; typedef v4i b; typedef v8i cd; };
template <> struct STypes<S_BF> { typedef v2i a; typedef v4i b; typedef v8i cd; };
template <> struct STypes<S_BB> { typedef v2i a; typedef v4i b; typedef v8i cd; };
template <> struct STypes<S_IU8> { typedef v2i a; typedef v4i b; typedef v8i cd; };

#define ARITH_SINSTR_S_F16 "v_swmmac_f32_16x16x32_f16"
#define ARITH_SINSTR_S_BF16 "v_swmmac_f32_16x16x32_bf16"
#define ARITH_SINSTR_S_FF "v_swmmac_f32_16x16x32_fp8_fp8"
#define ARITH_SINSTR_S_F16H "v_swmmac_f16_16x16x32_f16"
#define ARITH_SINSTR_S_BF16H "v_swmmac_bf16_16x16x32_bf16"
#define ARITH_SINSTR_S_FB "v_swmmac_f32_16x16x32_fp8_bf8"
#define ARITH_SINSTR_S_BF "v_swmmac_f32_16x16x32_bf8_fp8"
#define ARITH_SINSTR_S_BB "v_swmmac_f32_16x16x32_bf8_bf8"
#define ARITH_SINSTR_S_IU8 "v_swmmac_i32_16x16x32_iu8"
// the 8-bit integer sparse form runs signed x signed
#define ARITH_SMOD_S_F16 ""
#define ARITH_SMOD_S_BF16 ""
#define ARITH_SMOD_S_FF ""
#define ARITH_SMOD_S_F16H ""
#define ARITH_SMOD_S_BF16H ""
#define ARITH_SMOD_S_FB ""
#define ARITH_SMOD_S_BF ""
#define ARITH_SMOD_S_BB ""
#define ARITH_SMOD_S_IU8 " neg_lo:[1,1,0]"
#define ARITH_SCASE(F, K)                                                                                          \
    if constexpr (SF == F && KEY == K) {                                                                           \
        asm volatile(ARITH_SINSTR_##F " %0, %1, %2, %3 index_key:" #K ARITH_SMOD_##F : "+v"(c) : "v"(a), "v"(b),  \
                     "v"(idx));                                                                                    \
    }

template <int SF, int KEY>
__device__ __forceinline__ void smm(typename STypes<SF>::a a, typename STypes<SF>::b b, typename STypes<SF>::cd & c,
                                   int idx) {
    ARITH_SCASE(S_F16, 0) ARITH_SCASE(S_F16, 1) ARITH_SCASE(S_BF16, 0) ARITH_SCASE(S_BF16, 1)
    ARITH_SCASE(S_FF, 0) ARITH_SCASE(S_FF, 1) ARITH_SCASE(S_F16H, 0) ARITH_SCASE(S_F16H, 1)
    ARITH_SCASE(S_BF16H, 0) ARITH_SCASE(S_BF16H, 1) ARITH_SCASE(S_FB, 0) ARITH_SCASE(S_FB, 1)
    ARITH_SCASE(S_BF, 0) ARITH_SCASE(S_BF, 1) ARITH_SCASE(S_BB, 0) ARITH_SCASE(S_BB, 1)
    ARITH_SCASE(S_IU8, 0) ARITH_SCASE(S_IU8, 1)
}

// One wave per problem. A: [wave][lane][wa], B: [wave][lane][wb], C, D: [wave][lane][8], I: [wave][lane].
template <int SF, int KEY>
__global__ __launch_bounds__(32) void smm_kernel(const uint32_t * __restrict__ A, const uint32_t * __restrict__ B,
                                                const uint32_t * __restrict__ C, const uint32_t * __restrict__ I,
                                                uint32_t * __restrict__ D) {
    typedef typename STypes<SF>::a TA;
    typedef typename STypes<SF>::b TB;
    typedef typename STypes<SF>::cd TCD;
    const int w = blockIdx.x, lane = threadIdx.x;
    TA a;
    TB b;
    TCD c;
    __builtin_memcpy(&a, A + ((size_t) w * 32 + lane) * (sizeof(TA) / 4), sizeof a);
    __builtin_memcpy(&b, B + ((size_t) w * 32 + lane) * (sizeof(TB) / 4), sizeof b);
    __builtin_memcpy(&c, C + ((size_t) w * 32 + lane) * 8, sizeof c);
    const int idx = (int) I[(size_t) w * 32 + lane];
    smm<SF, KEY>(a, b, c, idx);
    uint32_t out[8] = {};
    __builtin_memcpy(out, &c, sizeof c);
    for (int i = 0; i < 8; ++i) {
        D[((size_t) w * 32 + lane) * 8 + i] = out[i];
    }
}

typedef void (*SKern)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t *);
inline SKern skern_for(int sf, int key) {
    switch (sf * 2 + key) {
    case 0: return smm_kernel<S_F16, 0>;
    case 1: return smm_kernel<S_F16, 1>;
    case 2: return smm_kernel<S_BF16, 0>;
    case 3: return smm_kernel<S_BF16, 1>;
    case 4: return smm_kernel<S_FF, 0>;
    case 5: return smm_kernel<S_FF, 1>;
    case 6: return smm_kernel<S_F16H, 0>;
    case 7: return smm_kernel<S_F16H, 1>;
    case 8: return smm_kernel<S_BF16H, 0>;
    case 9: return smm_kernel<S_BF16H, 1>;
    case 10: return smm_kernel<S_FB, 0>;
    case 11: return smm_kernel<S_FB, 1>;
    case 12: return smm_kernel<S_BF, 0>;
    case 13: return smm_kernel<S_BF, 1>;
    case 14: return smm_kernel<S_BB, 0>;
    case 15: return smm_kernel<S_BB, 1>;
    case 16: return smm_kernel<S_IU8, 0>;
    case 17: return smm_kernel<S_IU8, 1>;
    }
    return nullptr;
}

// B slot t = 0..31 of column col: lane col + 16 * (t >> 4); 16-bit: VGPR (t & 15) >> 1, half t & 1;
// 8-bit: VGPR (t & 15) >> 2, byte t & 3 (the dense layout, extended to K = 32).
inline Slot sslot_b(int sf, int col, int t) {
    const int lane = col + 16 * (t >> 4), e = t & 15;
    if (elem_bits(sdense(sf)) == 8) {
        return {lane, e >> 2, (e & 3) * 8};
    }
    return {lane, e >> 1, (e & 1) * 16};
}

struct SBatch {
    int sf = S_F16, key = 0, waves = 0;
    std::vector<uint32_t> a, b, c, d, idx;
    int wa() const { return words_ab(sdense(sf)); }
    void init(int f, int k, int n) {
        sf = f;
        key = k;
        waves = n;
        a.assign((size_t) n * 32 * wa(), 0);
        b.assign((size_t) n * 32 * swords_b(f), 0);
        c.assign((size_t) n * 256, 0);
        idx.assign((size_t) n * 32, 0);
    }
    void set_a(int w, int r, int j, uint32_t bits) {
        put_bits(&a[(size_t) w * 32 * wa()], slot_ab(sdense(sf), r, j), wa(), bits, elem_bits(sdense(sf)));
    }
    void set_b(int w, int col, int t, uint32_t bits) {
        put_bits(&b[(size_t) w * 32 * swords_b(sf)], sslot_b(sf, col, t), swords_b(sf), bits, elem_bits(sdense(sf)));
    }
    void set_c(int w, int r, int col, uint32_t bits) {
        put_bits(&c[(size_t) w * 256], slot_cd(sdense(sf), r, col), 8, bits, acc16(sdense(sf)) ? 16 : 32);
    }
    uint32_t get_d(int w, int r, int col) const {
        return get_bits(&d[(size_t) w * 256], slot_cd(sdense(sf), r, col), 8, acc16(sdense(sf)) ? 16 : 32);
    }
};

inline void srun(SBatch & bt) {
    uint32_t *da, *db, *dc, *di, *dd;
    CHECK(hipMalloc(&da, bt.a.size() * 4));
    CHECK(hipMalloc(&db, bt.b.size() * 4));
    CHECK(hipMalloc(&dc, bt.c.size() * 4));
    CHECK(hipMalloc(&di, bt.idx.size() * 4));
    CHECK(hipMalloc(&dd, bt.c.size() * 4));
    CHECK(hipMemcpy(da, bt.a.data(), bt.a.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, bt.b.data(), bt.b.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc, bt.c.data(), bt.c.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(di, bt.idx.data(), bt.idx.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dd, 0xEE, bt.c.size() * 4));
    hipLaunchKernelGGL(skern_for(bt.sf, bt.key), dim3(bt.waves), dim3(32), 0, 0, da, db, dc, di, dd);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    bt.d.resize(bt.c.size());
    CHECK(hipMemcpy(bt.d.data(), dd, bt.d.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(da));
    CHECK(hipFree(db));
    CHECK(hipFree(dc));
    CHECK(hipFree(di));
    CHECK(hipFree(dd));
}

}  // namespace arith

// ---------------------------------------------------------------------------------------------------------------
// The documented hazard (lane 5 item 6): an FP16-into-FP16 result D1 used as A (V 0, 1) or B (V 2, 3) of the next
// FP16-into-FP32 instruction, with nothing between (V 0, 2; forbidden by AMD's guide) or with a V_NOP (V 1, 3).
// D1's registers start with a known pattern (INIT), so a stale read can be recognised. Each wave writes D1 and D2.
namespace arith {

#define ARITH_HZ_FIRST "v_wmma_f16_16x16x16_f16 %0, %2, %3, %4\n"
template <int V>
__global__ __launch_bounds__(32) void hz_kernel(const uint32_t * __restrict__ A0, const uint32_t * __restrict__ B0,
                                               const uint32_t * __restrict__ C0, const uint32_t * __restrict__ X1,
                                               const uint32_t * __restrict__ C1, const uint32_t * __restrict__ INIT,
                                               uint32_t * __restrict__ D1out, uint32_t * __restrict__ D2out) {
    const int w = blockIdx.x, lane = threadIdx.x;
    const size_t o4 = ((size_t) w * 32 + lane) * 4, o8 = ((size_t) w * 32 + lane) * 8;
    v4i a0, b0, c0, x1, d1;
    v8i c1, d2;
    __builtin_memcpy(&a0, A0 + o4, 16);
    __builtin_memcpy(&b0, B0 + o4, 16);
    __builtin_memcpy(&c0, C0 + o4, 16);
    __builtin_memcpy(&x1, X1 + o4, 16);
    __builtin_memcpy(&d1, INIT + o4, 16);
    __builtin_memcpy(&c1, C1 + o8, 32);
    if constexpr (V == 0) {
        asm volatile(ARITH_HZ_FIRST "v_wmma_f32_16x16x16_f16 %1, %0, %5, %6\n"
                     : "+&v"(d1), "=&v"(d2) : "v"(a0), "v"(b0), "v"(c0), "v"(x1), "v"(c1));
    } else if constexpr (V == 1) {
        asm volatile(ARITH_HZ_FIRST "v_nop\nv_wmma_f32_16x16x16_f16 %1, %0, %5, %6\n"
                     : "+&v"(d1), "=&v"(d2) : "v"(a0), "v"(b0), "v"(c0), "v"(x1), "v"(c1));
    } else if constexpr (V == 2) {
        asm volatile(ARITH_HZ_FIRST "v_wmma_f32_16x16x16_f16 %1, %5, %0, %6\n"
                     : "+&v"(d1), "=&v"(d2) : "v"(a0), "v"(b0), "v"(c0), "v"(x1), "v"(c1));
    } else {
        asm volatile(ARITH_HZ_FIRST "v_nop\nv_wmma_f32_16x16x16_f16 %1, %5, %0, %6\n"
                     : "+&v"(d1), "=&v"(d2) : "v"(a0), "v"(b0), "v"(c0), "v"(x1), "v"(c1));
    }
    __builtin_memcpy(D1out + o4, &d1, 16);
    __builtin_memcpy(D2out + o8, &d2, 32);
}

typedef void (*HzKern)(const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *, const uint32_t *,
                       const uint32_t *, uint32_t *, uint32_t *);
inline HzKern hz_for(int v) {
    return v == 0 ? hz_kernel<0> : v == 1 ? hz_kernel<1> : v == 2 ? hz_kernel<2> : hz_kernel<3>;
}

// in: a0, b0, c0, x1, init (4 words per lane), c1 (8 words per lane), all [wave][lane][...]; out: d1, d2.
inline void hz_run(int v, int waves, const std::vector<uint32_t> * in4, const std::vector<uint32_t> & c1,
                   std::vector<uint32_t> & d1, std::vector<uint32_t> & d2) {
    uint32_t * dev[8];
    const size_t n4 = (size_t) waves * 32 * 4 * 4, n8 = n4 * 2;
    for (int i = 0; i < 5; ++i) {
        CHECK(hipMalloc(&dev[i], n4));
        CHECK(hipMemcpy(dev[i], in4[i].data(), n4, hipMemcpyHostToDevice));
    }
    CHECK(hipMalloc(&dev[5], n8));
    CHECK(hipMemcpy(dev[5], c1.data(), n8, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&dev[6], n4));
    CHECK(hipMalloc(&dev[7], n8));
    hipLaunchKernelGGL(hz_for(v), dim3(waves), dim3(32), 0, 0, dev[0], dev[1], dev[2], dev[3], dev[5], dev[4], dev[6], dev[7]);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    d1.resize(n4 / 4);
    d2.resize(n8 / 4);
    CHECK(hipMemcpy(d1.data(), dev[6], n4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(d2.data(), dev[7], n8, hipMemcpyDeviceToHost));
    for (auto * p : dev) {
        CHECK(hipFree(p));
    }
}

}  // namespace arith
