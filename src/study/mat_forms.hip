// Lane 4, items 1, 2, 3, 6 and 7: cycles per matrix instruction for every dense (WMMA) and sparse (SWMMAC) form,
// chained into one accumulator and over 4 independent ones; the sparse index content; the accumulator as an inline
// constant and the negate/absolute modifiers; data content. Method and registers: mat_common.h.
//
// Usage: mat_forms forms|index|mods|data [--break]
//   forms  calibration (16 independent v_fma_f32: 1.00 cycles known from lane 1) and every form, chain and x4.
//          PASS needs the calibration and the first-hour known answers (FP16, BF16 16; FP8, 8-bit integer 8 cycles,
//          within 3 %) and every loop linear. Other forms are compared with AMD's calculator and reported, not judged.
//          --break claims FP16 takes 8 cycles: must FAIL.
//   index  sparse forms with different index-register contents; PASS when no content moves the time by over 2 %.
//   mods   C as an inline constant against a register; negate and absolute modifiers; PASS when the inline constants
//          and the A/B modifiers (chain and x4) and the C modifiers with 4 accumulators are within 2 % of the plain
//          form. The C modifiers in a dependent chain are reported, not judged (measured: they add 2 cycles).
//   data   zeros, ones, random, denormals, NaN, infinity, large cancellation for FP16, BF16, FP16-accumulator, FP8,
//          8-bit integer forms, chain and 8 accumulators; PASS when no content moves the time by over 2 %.
#include "mat_common.h"

#include <random>

// Calibration: 16 independent FMAs over the 8 X registers (each register every 8th instruction, beyond the
// 5-cycle latency), sources v138/v139.
#define FMA8                                                                                                  \
    "v_fma_f32 v130, v130, v138, v139\nv_fma_f32 v131, v131, v138, v139\nv_fma_f32 v132, v132, v138, v139\n" \
    "v_fma_f32 v133, v133, v138, v139\nv_fma_f32 v134, v134, v138, v139\nv_fma_f32 v135, v135, v138, v139\n" \
    "v_fma_f32 v136, v136, v138, v139\nv_fma_f32 v137, v137, v138, v139\n"
MAT_TEST(calib_fma, FMA8 FMA8)

#define FORM(F)                       \
    MAT_TEST(F##_c1, MAT_C1(MF_##F)) \
    MAT_TEST(F##_x4, MAT_C4(MF_##F))
#define SFORM(F)                       \
    MAT_TEST(s##F##_c1, MAT_C1(MS_##F)) \
    MAT_TEST(s##F##_x4, MAT_C4(MS_##F))
FORM(F16)
FORM(BF16)
FORM(F16H)
FORM(BF16H)
FORM(FF)
FORM(FB)
FORM(BF)
FORM(BB)
FORM(IU8)
FORM(IU8_SA)
FORM(IU8_SB)
FORM(IU8_S)
FORM(IU8_C)
FORM(IU8_SAC)
FORM(IU8_SBC)
FORM(IU8_SC)
FORM(IU4)
FORM(IU4K32)
SFORM(F16)
SFORM(BF16)
SFORM(F16H)
SFORM(BF16H)
SFORM(IU8)
SFORM(IU4K32)
SFORM(IU4K64)
SFORM(FF)
SFORM(FB)
SFORM(BF)
SFORM(BB)

// Sparse with the second index set (index_key:1 reads the high 16 bits of the index register).
#define MS_F16K1(n, m) "v_swmmac_f32_16x16x32_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B8(m) ", v128 index_key:1\n"
#define MS_IU8K1(n, m) "v_swmmac_i32_16x16x32_iu8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B4(m) ", v128 index_key:1 neg_lo:[1,1,0]\n"
SFORM(F16K1)
SFORM(IU8K1)

// Item 6. C as inline constant: every instruction writes its accumulator from A x B + 0 (or + 1.0); no chain.
#define MF_F16_C0(n, m) "v_wmma_f32_16x16x16_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", 0\n"
#define MF_F16_C1(n, m) "v_wmma_f32_16x16x16_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", 1.0\n"
#define MF_FF_C0(n, m) "v_wmma_f32_16x16x16_fp8_fp8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B2(m) ", 0\n"
#define MF_IU8_C0(n, m) "v_wmma_i32_16x16x16_iu8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B2(m) ", 0 neg_lo:[1,1,0]\n"
// C from a register that no instruction writes (D7), D rotating or fixed: the register counterpart of the above.
#define MF_F16_CR(n, m) "v_wmma_f32_16x16x16_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", v[72:79]\n"
#define MF_FF_CR(n, m) "v_wmma_f32_16x16x16_fp8_fp8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B2(m) ", v[72:79]\n"
#define MF_IU8_CR(n, m) "v_wmma_i32_16x16x16_iu8 " MAT_D(n) ", " MAT_A2(m) ", " MAT_B2(m) ", v[72:79] neg_lo:[1,1,0]\n"
// 7 accumulators (D7 is the constant C source) rotating.
#define MAT_C7(F) MAT_R2(F(0, 0) F(1, 0) F(2, 0) F(3, 0) F(4, 0) F(5, 0) F(6, 0) F(0, 0))
MAT_TEST(f16_cin0_one, MAT_C1(MF_F16_C0))
MAT_TEST(f16_cin0_rot, MAT_C7(MF_F16_C0))
MAT_TEST(f16_cin1_rot, MAT_C7(MF_F16_C1))
MAT_TEST(f16_creg_one, MAT_C1(MF_F16_CR))
MAT_TEST(f16_creg_rot, MAT_C7(MF_F16_CR))
MAT_TEST(ff_cin0_rot, MAT_C7(MF_FF_C0))
MAT_TEST(ff_creg_rot, MAT_C7(MF_FF_CR))
MAT_TEST(iu8_cin0_rot, MAT_C7(MF_IU8_C0))
MAT_TEST(iu8_creg_rot, MAT_C7(MF_IU8_CR))
// Modifiers on the FP16 form (chain): NEG on A, NEG_HI on A, NEG on B, NEG on C, ABS on C (NEG_HI[2]), all.
#define MF_F16_M(n, m, MOD) "v_wmma_f32_16x16x16_f16 " MAT_D(n) ", " MAT_A4(m) ", " MAT_B4(m) ", " MAT_D(n) " " MOD "\n"
#define MF_F16_NA(n, m) MF_F16_M(n, m, "neg_lo:[1,0,0]")
#define MF_F16_NHA(n, m) MF_F16_M(n, m, "neg_hi:[1,0,0]")
#define MF_F16_NB(n, m) MF_F16_M(n, m, "neg_lo:[0,1,0]")
#define MF_F16_NC(n, m) MF_F16_M(n, m, "neg_lo:[0,0,1]")
#define MF_F16_AC(n, m) MF_F16_M(n, m, "neg_hi:[0,0,1]")
#define MF_F16_ALL(n, m) MF_F16_M(n, m, "neg_lo:[1,1,1] neg_hi:[1,1,1]")
MAT_TEST(f16_negA, MAT_C1(MF_F16_NA))
MAT_TEST(f16_neghiA, MAT_C1(MF_F16_NHA))
MAT_TEST(f16_negB, MAT_C1(MF_F16_NB))
MAT_TEST(f16_negC, MAT_C1(MF_F16_NC))
MAT_TEST(f16_absC, MAT_C1(MF_F16_AC))
MAT_TEST(f16_allmods, MAT_C1(MF_F16_ALL))
// The same with 4 accumulators, where a modifier cost could not hide behind the chain.
MAT_TEST(f16_negA_x4, MAT_C4(MF_F16_NA))
MAT_TEST(f16_absC_x4, MAT_C4(MF_F16_AC))
MAT_TEST(f16_allmods_x4, MAT_C4(MF_F16_ALL))

// Item 7 uses the chain and an 8-accumulator build of the plain forms.
#define DFORM(F) MAT_TEST(F##_x8, MAT_C8(MF_##F))
DFORM(F16)
DFORM(BF16)
DFORM(F16H)
DFORM(FF)
DFORM(BB)
DFORM(IU8_S)
DFORM(IU8_SC)

// AMD's calculator (sources/matrix_calc.py, rdna4 entries): cycles per instruction.
struct Form {
    const char * name;
    mat::Test c1, x4;
    double calc;
};
#define F_ENTRY(F, CALC) Form{#F, MAT_ENTRY(F##_c1), MAT_ENTRY(F##_x4), CALC}
#define S_ENTRY(F, CALC) Form{"s" #F, MAT_ENTRY(s##F##_c1), MAT_ENTRY(s##F##_x4), CALC}

static std::vector<Form> forms() {
    return {F_ENTRY(F16, 16),    F_ENTRY(BF16, 16),  F_ENTRY(F16H, 16),     F_ENTRY(BF16H, 16),   F_ENTRY(FF, 8),
            F_ENTRY(FB, 8),      F_ENTRY(BF, 8),     F_ENTRY(BB, 8),        F_ENTRY(IU8, 8),      F_ENTRY(IU8_SA, 8),
            F_ENTRY(IU8_SB, 8),  F_ENTRY(IU8_S, 8),  F_ENTRY(IU8_C, 8),     F_ENTRY(IU8_SAC, 8),  F_ENTRY(IU8_SBC, 8),
            F_ENTRY(IU8_SC, 8),  F_ENTRY(IU4, 8),    F_ENTRY(IU4K32, 8),    S_ENTRY(F16, 16),     S_ENTRY(BF16, 16),
            S_ENTRY(F16H, 16),   S_ENTRY(BF16H, 16), S_ENTRY(IU8, 8),       S_ENTRY(IU4K32, 8),   S_ENTRY(IU4K64, 8),
            S_ENTRY(FF, 8),      S_ENTRY(FB, 8),     S_ENTRY(BF, 8),        S_ENTRY(BB, 8)};
}

// Multiply-adds per instruction (M x N x K; for sparse forms the K of the dense equivalent).
static double macs_of(const std::string & n) {
    if (n == "IU4K32" || n[0] == 's') {
        return n == "sIU4K64" ? 16 * 16 * 64 : 16 * 16 * 32;
    }
    return 16 * 16 * 16;
}

static int mode_forms(bool brk) {
    mat::set_input(mat::default_input());
    mat::Checks ck;
    const auto cal = mat::measure("mat_forms", {MAT_ENTRY(calib_fma)}, 16, "fma");
    ck.add(mat::near(cal.at("calib_fma").per, 1.0, 0.05) && cal.at("calib_fma").lin,
           "calibration %.3f cycles per independent FMA (lane 1: 1.00)", cal.at("calib_fma").per);
    std::map<std::string, double> known = {{"F16", brk ? 8.0 : 16.0}, {"BF16", 16.0}, {"FF", 8.0}, {"IU8_S", 8.0}};
    int nonlin = 0, differs = 0;
    for (const Form & f : forms()) {
        const auto m = mat::measure("mat_forms", {f.c1, f.x4}, 16);
        const mat::Measured c1 = m.at(f.c1.name), x4 = m.at(f.x4.name);
        nonlin += !c1.lin + !x4.lin;
        const bool calc_ok = mat::near(c1.per, f.calc, 0.03 * f.calc);
        differs += !calc_ok;
        std::printf("RESULT mat_forms form %-8s chain %7.3f [%.3f-%.3f] x4 %7.3f [%.3f-%.3f] calculator %2.0f %s "
                    "macs_per_cycle_chain %.0f macs_per_cycle_x4 %.0f\n",
                    f.name, c1.per, c1.lo, c1.hi, x4.per, x4.lo, x4.hi, f.calc, calc_ok ? "same" : "DIFFERS",
                    macs_of(f.name) / c1.per, macs_of(f.name) / x4.per);
        if (known.count(f.name)) {
            ck.add(mat::near(c1.per, known[f.name], 0.03 * known[f.name]),
                   "%s chain %.3f cycles, expected %.0f (first hour)", f.name, c1.per, known[f.name]);
        }
    }
    ck.add(nonlin == 0, "%d loops non-linear", nonlin);
    std::printf("RESULT mat_forms forms_differing_from_calculator %d\n", differs);
    return ck.finish("mat_forms_forms");
}

// Sparse index contents (per lane): each 4-bit group holds two 2-bit element positions for one group of four.
static int mode_index() {
    struct Idx {
        const char * name;
        uint32_t v;
        bool random, valid;
    };
    const std::vector<Idx> idxs = {{"0x44444444_pos01", 0x44444444u, false, true},
                                   {"0xEEEEEEEE_pos23", 0xEEEEEEEEu, false, true},
                                   {"0xD8D8D8D8_mixed", 0xD8D8D8D8u, false, true},
                                   {"random_valid", 0, true, true},
                                   {"0x00000000_pos00", 0x00000000u, false, false},
                                   {"0xFFFFFFFF_pos33", 0xFFFFFFFFu, false, false},
                                   {"random_any", 0, true, false}};
    const std::vector<mat::Test> tests = {MAT_ENTRY(sF16_c1), MAT_ENTRY(sF16_x4), MAT_ENTRY(sIU8_c1),
                                          MAT_ENTRY(sIU8_x4), MAT_ENTRY(sFF_c1),  MAT_ENTRY(sF16K1_c1),
                                          MAT_ENTRY(sIU8K1_c1), MAT_ENTRY(sIU4K64_c1)};
    std::mt19937 rng(7);
    // valid pairs (lo < hi) as 4-bit codes: positions (i, j) -> i | j << 2
    const uint32_t valid[6] = {0x4, 0x8, 0xC, 0x9, 0xD, 0xE};
    std::map<std::string, double> base;
    mat::Checks ck;
    double worst = 0;
    std::string worst_at;
    for (const Idx & ix : idxs) {
        mat::Input in = mat::default_input();
        for (int l = 0; l < 32; ++l) {
            uint32_t v = ix.v;
            if (ix.random) {
                v = 0;
                for (int g = 0; g < 8; ++g) {
                    v |= (ix.valid ? valid[rng() % 6] : (rng() & 15u)) << (4 * g);
                }
            }
            in.w[mat::IN_IDX + l] = v;
        }
        mat::set_input(in);
        std::printf("RESULT mat_forms index_content %s\n", ix.name);
        const auto m = mat::measure("mat_forms", tests, 16);
        for (const auto & [n, v] : m) {
            if (!base.count(n)) {
                base[n] = v.per;
            }
            const double rel = v.per / base[n] - 1;
            if (std::fabs(rel) > std::fabs(worst)) {
                worst = rel;
                worst_at = std::string(n) + " " + ix.name;
            }
        }
    }
    ck.add(std::fabs(worst) <= 0.02, "largest change from the first index content %.2f %% (%s)", 100 * worst,
           worst_at.empty() ? "-" : worst_at.c_str());
    return ck.finish("mat_forms_index");
}

static int mode_mods() {
    mat::set_input(mat::default_input());
    const auto m = mat::measure(
        "mat_forms",
        {MAT_ENTRY(F16_c1), MAT_ENTRY(F16_x4), MAT_ENTRY(f16_cin0_one), MAT_ENTRY(f16_cin0_rot), MAT_ENTRY(f16_cin1_rot),
         MAT_ENTRY(f16_creg_one), MAT_ENTRY(f16_creg_rot), MAT_ENTRY(FF_x4), MAT_ENTRY(ff_cin0_rot),
         MAT_ENTRY(ff_creg_rot), MAT_ENTRY(IU8_S_x4), MAT_ENTRY(iu8_cin0_rot), MAT_ENTRY(iu8_creg_rot),
         MAT_ENTRY(f16_negA), MAT_ENTRY(f16_neghiA), MAT_ENTRY(f16_negB), MAT_ENTRY(f16_negC), MAT_ENTRY(f16_absC),
         MAT_ENTRY(f16_allmods), MAT_ENTRY(f16_negA_x4), MAT_ENTRY(f16_absC_x4), MAT_ENTRY(f16_allmods_x4)},
        16);
    mat::Checks ck;
    const std::vector<std::pair<const char *, const char *>> pairs = {
        {"f16_cin0_one", "F16_x4"},   {"f16_cin0_rot", "F16_x4"},  {"f16_cin1_rot", "F16_x4"},
        {"f16_creg_one", "F16_x4"},   {"f16_creg_rot", "F16_x4"},  {"ff_cin0_rot", "FF_x4"},
        {"ff_creg_rot", "FF_x4"},     {"iu8_cin0_rot", "IU8_S_x4"}, {"iu8_creg_rot", "IU8_S_x4"},
        {"f16_negA", "F16_c1"},       {"f16_neghiA", "F16_c1"},    {"f16_negB", "F16_c1"},
        {"f16_negC", "F16_c1"},       {"f16_absC", "F16_c1"},      {"f16_allmods", "F16_c1"},
        {"f16_negA_x4", "F16_x4"},    {"f16_absC_x4", "F16_x4"},   {"f16_allmods_x4", "F16_x4"}};
    for (const auto & [a, b] : pairs) {
        const double rel = m.at(a).per / m.at(b).per - 1;
        std::printf("RESULT mat_forms mods %-16s %7.3f against %-8s %7.3f change %+6.2f %%\n", a, m.at(a).per, b,
                    m.at(b).per, 100 * rel);
        // No document gives a cost for the C modifiers in a dependent chain; they are reported, not judged.
        const bool judged = !(std::string(a) == "f16_negC" || std::string(a) == "f16_absC" ||
                              std::string(a) == "f16_allmods");
        if (judged) {
            ck.add(std::fabs(rel) <= 0.02, "%s %+.2f %% against %s", a, 100 * rel, b);
        } else {
            std::printf("RESULT mat_forms mods_not_judged %s adds %.3f cycles per instruction in a chain\n", a,
                        m.at(a).per - m.at(b).per);
        }
    }
    return ck.finish("mat_forms_mods");
}

// Item 7. Encodings.
static uint16_t f16_bits(float f) {
    const _Float16 h = (_Float16) f;
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
static uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
static uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

enum Enc { E_F16, E_BF16, E_FP8, E_BF8, E_I8 };

// One data kind: generator of A elements, B elements (as raw element codes) and the C word.
struct Kind {
    const char * name;
    // returns the code of one element for operand 0 = A, 1 = B; i = element index within the lane's word list
    uint32_t (*elem)(Enc, int op, int i, std::mt19937 &);
    uint32_t (*cword)(Enc, std::mt19937 &);
};

static uint32_t one_code(Enc e) {
    switch (e) {
        case E_F16: return f16_bits(1.0f);
        case E_BF16: return bf16_bits(1.0f);
        case E_FP8: return 0x38;
        case E_BF8: return 0x3C;
        default: return 1;
    }
}
static uint32_t c_zero(Enc, std::mt19937 &) { return 0; }
static uint32_t c_rand(Enc e, std::mt19937 & r) {
    if (e == E_I8) {
        return r() & 0xFFFFu;
    }
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    return f32_bits(u(r));
}
static uint32_t k_zero(Enc, int, int, std::mt19937 &) { return 0; }
static uint32_t k_one(Enc e, int, int, std::mt19937 &) { return one_code(e); }
static uint32_t k_rand(Enc e, int, int, std::mt19937 & r) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    switch (e) {
        case E_F16: return f16_bits(u(r));
        case E_BF16: return bf16_bits(u(r));
        case E_FP8: {  // any finite E4M3 code (0x7F/0xFF are NaN)
            uint32_t c;
            do {
                c = r() & 0xFF;
            } while ((c & 0x7F) == 0x7F);
            return c;
        }
        case E_BF8: {  // any finite E5M2 code (exponent 31 is inf/NaN)
            uint32_t c;
            do {
                c = r() & 0xFF;
            } while ((c & 0x7C) == 0x7C);
            return c;
        }
        default: return r() & 0xFF;
    }
}
// Denormal inputs in their own format (integers: the largest magnitudes instead, which overflow the sum).
static uint32_t k_denorm(Enc e, int, int, std::mt19937 & r) {
    switch (e) {
        case E_F16: return 1 + r() % 0x3FF;
        case E_BF16: return 1 + r() % 0x7F;
        case E_FP8: return 1 + r() % 7;
        case E_BF8: return 1 + r() % 3;
        default: return (r() & 1) ? 0x7F : 0x80;
    }
}
// Normal inputs whose products are denormal in the accumulator (f32 for FP16/BF16/FP8 forms; FP16 for the
// FP16-accumulator form, chosen by the caller through Enc E_F16 with a 16-bit accumulator). FP16 and FP8 products
// cannot reach f32 denormals; those forms use their smallest normal value instead.
static uint32_t k_tinyprod(Enc e, int, int, std::mt19937 &) {
    switch (e) {
        case E_F16: return f16_bits(0x1p-10f);  // product 2^-20: denormal in FP16, normal in FP32
        case E_BF16: return bf16_bits(0x1p-70f);  // product 2^-140: denormal in FP32
        case E_FP8: return 0x08;                 // 2^-6, smallest normal
        case E_BF8: return 0x04;                 // 2^-14
        default: return 1;
    }
}
static uint32_t k_nan(Enc e, int op, int, std::mt19937 &) {
    if (op == 1) {
        return one_code(e);
    }
    switch (e) {
        case E_F16: return 0x7E00;
        case E_BF16: return 0x7FC0;
        case E_FP8: return 0x7F;
        case E_BF8: return 0x7E;
        default: return 0;
    }
}
static uint32_t k_inf(Enc e, int op, int, std::mt19937 &) {
    if (op == 1) {
        return one_code(e);
    }
    switch (e) {
        case E_F16: return 0x7C00;
        case E_BF16: return 0x7F80;
        case E_FP8: return 0x7E;  // E4M3 has no infinity: largest finite (448)
        case E_BF8: return 0x7C;
        default: return 0;
    }
}
// Infinity times zero: NaN products.
static uint32_t k_infzero(Enc e, int op, int i, std::mt19937 & r) {
    if (op == 1) {
        return 0;
    }
    return k_inf(e, 0, i, r);
}
// Large cancellation: A alternates +big and -big within every row, B is +big everywhere; every row of A has as
// many +big as -big elements, so the exact sum of products is 0 while each product is huge.
static uint32_t k_cancel(Enc e, int op, int i, std::mt19937 &) {
    const bool neg = op == 0 && (i & 1);
    switch (e) {
        case E_F16: return f16_bits(neg ? -32768.0f : 32768.0f);
        case E_BF16: return bf16_bits(neg ? -0x1p60f : 0x1p60f);
        case E_FP8: return neg ? 0xFE : 0x7E;  // +-448
        case E_BF8: return neg ? 0xFB : 0x7B;  // +-57344
        default: return neg ? 0x81 : 0x7F;     // -127 / +127 (signed)
    }
}
static uint32_t c_one(Enc e, std::mt19937 &) { return e == E_I8 ? 1 : f32_bits(1.0f); }

static const std::vector<Kind> & kinds() {
    static const std::vector<Kind> k = {
        {"zero", k_zero, c_zero},       {"one", k_one, c_zero},          {"random", k_rand, c_rand},
        {"denormal_in", k_denorm, c_zero}, {"tiny_products", k_tinyprod, c_zero}, {"nan", k_nan, c_zero},
        {"inf", k_inf, c_zero},         {"inf_times_zero", k_infzero, c_zero}, {"cancel", k_cancel, c_one}};
    return k;
}

static void fill_kind(mat::Input & in, const Kind & k, Enc e, bool acc16, std::mt19937 & r) {
    const int bits = (e == E_F16 || e == E_BF16) ? 16 : 8;
    const int per_word = 32 / bits;
    for (int op = 0; op < 2; ++op) {
        const int base = op == 0 ? mat::IN_A : mat::IN_B;
        for (int l = 0; l < 32; ++l) {
            for (int w = 0; w < 8; ++w) {
                uint32_t word = 0;
                for (int j = 0; j < per_word; ++j) {
                    word |= (k.elem(e, op, w * per_word + j, r) & ((1u << bits) - 1)) << (bits * j);
                }
                in.w[base + l * 8 + w] = word;
            }
        }
    }
    for (int l = 0; l < 32; ++l) {
        for (int w = 0; w < 8; ++w) {
            uint32_t c = k.cword(e == E_I8 ? E_I8 : E_F16, r);
            if (acc16) {  // 16-bit accumulator: two halves per word
                float f;
                std::memcpy(&f, &c, 4);
                c = f16_bits(f) | ((uint32_t) f16_bits(f) << 16);
            }
            in.w[mat::IN_D + l * 8 + w] = c;
        }
    }
}

static int mode_data() {
    struct DF {
        const char * name;
        Enc enc;
        bool acc16;
        mat::Test c1, x8;
    };
    const std::vector<DF> dfs = {
        {"F16", E_F16, false, MAT_ENTRY(F16_c1), MAT_ENTRY(F16_x8)},
        {"BF16", E_BF16, false, MAT_ENTRY(BF16_c1), MAT_ENTRY(BF16_x8)},
        {"F16H", E_F16, true, MAT_ENTRY(F16H_c1), MAT_ENTRY(F16H_x8)},
        {"FF", E_FP8, false, MAT_ENTRY(FF_c1), MAT_ENTRY(FF_x8)},
        {"BB", E_BF8, false, MAT_ENTRY(BB_c1), MAT_ENTRY(BB_x8)},
        {"IU8_S", E_I8, false, MAT_ENTRY(IU8_S_c1), MAT_ENTRY(IU8_S_x8)},
        {"IU8_SC", E_I8, false, MAT_ENTRY(IU8_SC_c1), MAT_ENTRY(IU8_SC_x8)},
    };
    mat::Checks ck;
    std::mt19937 rng(11);
    for (const DF & d : dfs) {
        double base_c1 = 0, base_x8 = 0, worst = 0;
        std::string worst_at;
        for (const Kind & k : kinds()) {
            mat::Input in = mat::default_input();
            fill_kind(in, k, d.enc, d.acc16, rng);
            mat::set_input(in);
            char label[96];
            std::snprintf(label, sizeof label, "mat_forms data %s %s", d.name, k.name);
            const auto m = mat::measure(label, {d.c1, d.x8}, 16);
            const double c1 = m.at(d.c1.name).per, x8 = m.at(d.x8.name).per;
            if (base_c1 == 0) {
                base_c1 = c1;
                base_x8 = x8;
            }
            for (double rel : {c1 / base_c1 - 1, x8 / base_x8 - 1}) {
                if (std::fabs(rel) > std::fabs(worst)) {
                    worst = rel;
                    worst_at = k.name;
                }
            }
        }
        ck.add(std::fabs(worst) <= 0.02, "%s largest change from zeros %+.2f %% (%s)", d.name, 100 * worst,
               worst_at.empty() ? "-" : worst_at.c_str());
    }
    return ck.finish("mat_forms_data");
}

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "forms";
    const bool brk = mat::flag(argc, argv, "--break");
    if (mode == "forms") {
        return mode_forms(brk);
    }
    if (mode == "index") {
        return mode_index();
    }
    if (mode == "mods") {
        return mode_mods();
    }
    if (mode == "data") {
        return mode_data();
    }
    std::fprintf(stderr, "usage: mat_forms forms|index|mods|data [--break]\n");
    return 2;
}
