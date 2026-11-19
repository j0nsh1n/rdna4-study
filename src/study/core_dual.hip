// Lane 1, item 3: dual issue (VOPD) and VGPR bank effects, one wave alone on its SIMD.
//
// Registers are fixed so the bank rules are under control: the 16 chain registers are v40..v55 (operands %0..%15),
// the constants sit in v1, v2, v3, v4, v5, v6, v9, v13, v17, v21 (banks = register mod 4: 1, 2, 3, 0, 1, 2, 1, 1,
// 1, 1).
// A dual (VOPD) pattern is one v_dual_X :: v_dual_Y instruction whose X writes an even chain and Y the next odd chain
// (the destination parity rule); X reads v1, v2 and Y reads v3, v4 (src0 banks 1/3, src1 banks 2/0: legal). The 16
// patterns of a block cycle through 8 chain pairs, so each chain is rewritten every 8 instructions, more than the
// 5-cycle latency: the stream is independent. The "split" pattern is the same two operations as two ordinary VOP2
// instructions. If the hardware dual-issues, a dual pattern costs 1 cycle and a split pattern 2.
// The "dep" form rewrites the same chain pair in every pattern (each instruction waits for the one before), which
// --break presents as an independent stream so the verdict must FAIL.
// The compiler-paired form ("c_fmac") is plain C (x = fmaf(k0, k1, x) over 16 chains) built with dual issue on (the
// default); the script checks its loop for v_dual_ instructions.
#include "core_common.h"

#include <map>

#define DUAL_X16                                                                                             \
    "+{v40}"(x[0]), "+{v41}"(x[1]), "+{v42}"(x[2]), "+{v43}"(x[3]), "+{v44}"(x[4]), "+{v45}"(x[5]),          \
        "+{v46}"(x[6]), "+{v47}"(x[7]), "+{v48}"(x[8]), "+{v49}"(x[9]), "+{v50}"(x[10]), "+{v51}"(x[11]),    \
        "+{v52}"(x[12]), "+{v53}"(x[13]), "+{v54}"(x[14]), "+{v55}"(x[15])
#define DUAL_K10                                                                                                \
    "{v1}"(k[0]), "{v2}"(k[1]), "{v3}"(k[2]), "{v4}"(k[3]), "{v5}"(k[4]), "{v6}"(k[5]), "{v9}"(k[6]), "{v13}"(k[7]), \
        "{v17}"(k[8]), "{v21}"(k[9])
#define DUAL_K10_PIN                                                                                            \
    "+{v1}"(k[0]), "+{v2}"(k[1]), "+{v3}"(k[2]), "+{v4}"(k[3]), "+{v5}"(k[4]), "+{v6}"(k[5]), "+{v9}"(k[6]),     \
        "+{v13}"(k[7]), "+{v17}"(k[8]), "+{v21}"(k[9])

#define DUAL_KERNEL(NAME, BODY)                                                                                 \
    template <int U>                                                                                            \
    __global__ __launch_bounds__(32) void NAME(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,    \
                                               uint32_t * __restrict__ sink) {                                  \
        float x[16], k[10];                                                                                     \
        for (int i = 0; i < 16; ++i) {                                                                          \
            x[i] = core::from_bits<float>(in[0]);                                                               \
        }                                                                                                       \
        for (int i = 0; i < 10; ++i) {                                                                          \
            k[i] = core::from_bits<float>(in[1 + (i & 1)]);                                                     \
        }                                                                                                       \
        /* pin the constants in their registers so the loop does not re-materialise them */                     \
        asm volatile("" : DUAL_K10_PIN);                                                                        \
        for (int rep = 0; rep < core::REPS; ++rep) {                                                            \
            const uint64_t t0 = wave_cycles();                                                                  \
            _Pragma("unroll 1") for (int it = 0; it < core::ITERS; ++it) {                                      \
                _Pragma("unroll") for (int u = 0; u < U; ++u) {                                                 \
                    asm volatile(BODY : DUAL_X16 : DUAL_K10 : "vcc");                                            \
                }                                                                                               \
            }                                                                                                   \
            const uint64_t t1 = wave_cycles();                                                                  \
            if (threadIdx.x == 0) {                                                                             \
                out[rep] = cycle_diff(t0, t1);                                                                  \
            }                                                                                                   \
        }                                                                                                       \
        uint32_t s = 0;                                                                                         \
        for (int i = 0; i < 16; ++i) {                                                                          \
            s ^= core::fold(x[i]);                                                                              \
        }                                                                                                       \
        sink[threadIdx.x] = s;                                                                                  \
    }

#define PAIRS16(P) P(0, 1) P(2, 3) P(4, 5) P(6, 7) P(8, 9) P(10, 11) P(12, 13) P(14, 15) \
                   P(0, 1) P(2, 3) P(4, 5) P(6, 7) P(8, 9) P(10, 11) P(12, 13) P(14, 15)
#define PAIRS_DEP(P) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) \
                     P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1) P(0, 1)

#define A(a) " %" #a
#define LIT "0x3f800000"
// Dual forms: X in the first slot, Y in the second.
#define DU_FMAC(a, b) "v_dual_fmac_f32" A(a) ", v1, v2 :: v_dual_fmac_f32" A(b) ", v3, v4\n"
#define DU_FMAAK(a, b) "v_dual_fmaak_f32" A(a) ", v1, v2, " LIT " :: v_dual_fmaak_f32" A(b) ", v3, v4, " LIT "\n"
// FMAMK reads its second input through the SRC2 port: one must be even, one odd (v2, v5).
#define DU_FMAMK(a, b) "v_dual_fmamk_f32" A(a) ", v1, " LIT ", v2 :: v_dual_fmamk_f32" A(b) ", v4, " LIT ", v5\n"
#define DU_MUL(a, b) "v_dual_mul_f32" A(a) ", v1, v2 :: v_dual_mul_f32" A(b) ", v3, v4\n"
#define DU_ADD(a, b) "v_dual_add_f32" A(a) ", v1, v2 :: v_dual_add_f32" A(b) ", v3, v4\n"
#define DU_SUB(a, b) "v_dual_sub_f32" A(a) ", v1, v2 :: v_dual_sub_f32" A(b) ", v3, v4\n"
#define DU_SUBREV(a, b) "v_dual_subrev_f32" A(a) ", v1, v2 :: v_dual_subrev_f32" A(b) ", v3, v4\n"
#define DU_MULDX9(a, b) "v_dual_mul_dx9_zero_f32" A(a) ", v1, v2 :: v_dual_mul_dx9_zero_f32" A(b) ", v3, v4\n"
#define DU_MOV(a, b) "v_dual_mov_b32" A(a) ", v1 :: v_dual_mov_b32" A(b) ", v3\n"
#define DU_CND(a, b) "v_dual_cndmask_b32" A(a) ", v1, v2 :: v_dual_cndmask_b32" A(b) ", v3, v4\n"
#define DU_MAX(a, b) "v_dual_max_num_f32" A(a) ", v1, v2 :: v_dual_max_num_f32" A(b) ", v3, v4\n"
#define DU_MIN(a, b) "v_dual_min_num_f32" A(a) ", v1, v2 :: v_dual_min_num_f32" A(b) ", v3, v4\n"
#define DU_DOTF16(a, b) "v_dual_dot2acc_f32_f16" A(a) ", v1, v2 :: v_dual_dot2acc_f32_f16" A(b) ", v3, v4\n"
#define DU_DOTBF16(a, b) "v_dual_dot2acc_f32_bf16" A(a) ", v1, v2 :: v_dual_dot2acc_f32_bf16" A(b) ", v3, v4\n"
#define DU_ADDNC(a, b) "v_dual_add_f32" A(a) ", v1, v2 :: v_dual_add_nc_u32" A(b) ", v3, v4\n"
#define DU_LSHL(a, b) "v_dual_add_f32" A(a) ", v1, v2 :: v_dual_lshlrev_b32" A(b) ", v3, v4\n"
#define DU_AND(a, b) "v_dual_add_f32" A(a) ", v1, v2 :: v_dual_and_b32" A(b) ", v3, v4\n"
#define DU_FMAC_ADDNC(a, b) "v_dual_fmac_f32" A(a) ", v1, v2 :: v_dual_add_nc_u32" A(b) ", v3, v4\n"
// Same VGPR as src0 of both halves: breaks "different banks" but gfx12 allows it (LLVM AllowSameVGPR); the
// assembler accepts it. Compare with DU_MUL (src0 v1 and v3).
#define DU_MUL_SAMESRC0(a, b) "v_dual_mul_f32" A(a) ", v1, v2 :: v_dual_mul_f32" A(b) ", v1, v4\n"
// Split forms: the same two operations as ordinary instructions.
#define SP_FMAC(a, b) "v_fmac_f32" A(a) ", v1, v2\nv_fmac_f32" A(b) ", v3, v4\n"
#define SP_MUL(a, b) "v_mul_f32" A(a) ", v1, v2\nv_mul_f32" A(b) ", v3, v4\n"
#define SP_ADDNC(a, b) "v_add_f32" A(a) ", v1, v2\nv_add_nc_u32" A(b) ", v3, v4\n"
#define SP_DOTF16(a, b) "v_dot2_f32_f16" A(a) ", v1, v2" A(a) "\nv_dot2_f32_f16" A(b) ", v3, v4" A(b) "\n"
// Instruction size: literal-carrying forms are 64 bits (VOP2 + literal) or 96 bits (VOPD or VOP3 + literal).
#define SP_FMAAK(a, b) "v_fmaak_f32" A(a) ", v1, v2, " LIT "\nv_fmaak_f32" A(b) ", v3, v4, " LIT "\n"
#define SP_FMA3LIT(a, b) "v_fma_f32" A(a) ", v1, v2, " LIT "\nv_fma_f32" A(b) ", v3, v4, " LIT "\n"
#define SP_FMA3NOLIT(a, b) "v_fma_f32" A(a) ", v1, v2, 1.0\nv_fma_f32" A(b) ", v3, v4, 1.0\n"
// Transcendental beside ordinary vector work: v_exp_f32 takes 4 issue cycles alone (core_valu). If FMAs from the same
// wave issue while the transcendental unit works, exp + 3 FMAs costs 4 cycles; if not, 7. (The FMAs share one
// destination register. Versions with four or more back-to-back writes to one register were not linear in the block
// count and were dropped; see the measurement file.)
#define TR_EXP2(a, b) "v_exp_f32" A(a) ", v1\nv_exp_f32" A(b) ", v3\n"
#define TR_EXP_FMA(a, b) "v_exp_f32" A(a) ", v1\nv_fma_f32" A(b) ", v1, v2, v3\n"
#define TR_EXP_FMA3(a, b) "v_exp_f32" A(a) ", v1\nv_fma_f32" A(b) ", v1, v2, v3\nv_fma_f32" A(b) ", v5, v6, v4\nv_fma_f32" A(b) ", v1, v2, v3\n"
// Pairs the guide does not allow: there is no dual encoding, so they can only run as two instructions.
#define SP_MIX_ADD(a, b) "v_fma_mix_f32" A(a) ", v1, v2" A(a) " op_sel_hi:[1,1,0]\nv_add_f32" A(b) ", v3, v4\n"
#define SP_FMA3_ADD(a, b) "v_fma_f32" A(a) ", v1, v2" A(a) "\nv_add_f32" A(b) ", v3, v4\n"
#define SP_MULLO_ADD(a, b) "v_mul_lo_u32" A(a) ", v1, v2\nv_add_f32" A(b) ", v3, v4\n"

// Bank tests on ordinary VOP3 FMAs: two instructions per pattern (destination chains a and b), the sources are the
// constants; bank = register mod 4. Bank 1 holds v1, v5, v9, v13, v17, v21.
#define BK(a, b, r1, r2) "v_fma_f32" A(a) ", " r1 "\nv_fma_f32" A(b) ", " r2 "\n"
#define BK_DIFF(a, b) BK(a, b, "v1, v2, v3", "v5, v6, v4")              // three banks in each
#define BK_SAME2(a, b) BK(a, b, "v1, v5, v2", "v9, v13, v6")            // two sources in bank 1, changing
#define BK_SAME2_FIXED(a, b) BK(a, b, "v1, v5, v2", "v1, v5, v2")       // two sources in bank 1, repeated
#define BK_REPEAT(a, b) BK(a, b, "v1, v1, v5", "v9, v9, v13")           // three bank-1 operands, two registers
#define BK_SAMEREG(a, b) BK(a, b, "v1, v1, v1", "v5, v5, v5")           // one register read three times
#define BK_SAME3_FIXED(a, b) BK(a, b, "v1, v5, v9", "v1, v5, v9")       // three bank-1 registers, the same triple
#define BK_SAME3_AFTER(a, b) BK(a, b, "v2, v6, v3", "v1, v5, v9")       // bank-1 triple after an instruction on 2/3
#define BK_SAME3_MIXED(a, b) BK(a, b, "v1, v5, v9", "v13, v17, v2")     // bank-1 triple, then two bank-1 + bank 2
#define BK_SAME3_NOREUSE(a, b) BK(a, b, "v1, v5, v9", "v13, v17, v21")  // two different bank-1 triples

#define DEF(id, P) DUAL_KERNEL(id, PAIRS16(P))
DEF(du_fmac, DU_FMAC)
DEF(du_fmaak, DU_FMAAK)
DEF(du_fmamk, DU_FMAMK)
DEF(du_mul, DU_MUL)
DEF(du_add, DU_ADD)
DEF(du_sub, DU_SUB)
DEF(du_subrev, DU_SUBREV)
DEF(du_muldx9, DU_MULDX9)
DEF(du_mov, DU_MOV)
DEF(du_cnd, DU_CND)
DEF(du_max, DU_MAX)
DEF(du_min, DU_MIN)
DEF(du_dotf16, DU_DOTF16)
DEF(du_dotbf16, DU_DOTBF16)
DEF(du_addnc, DU_ADDNC)
DEF(du_lshl, DU_LSHL)
DEF(du_and, DU_AND)
DEF(du_fmac_addnc, DU_FMAC_ADDNC)
DEF(du_mul_samesrc0, DU_MUL_SAMESRC0)
DEF(sp_fmac, SP_FMAC)
DEF(sp_mul, SP_MUL)
DEF(sp_addnc, SP_ADDNC)
DEF(sp_dotf16, SP_DOTF16)
DEF(sp_fmaak, SP_FMAAK)
DEF(sp_fma3lit, SP_FMA3LIT)
DEF(sp_fma3nolit, SP_FMA3NOLIT)
DEF(tr_exp2, TR_EXP2)
DEF(tr_exp_fma, TR_EXP_FMA)
DEF(tr_exp_fma3, TR_EXP_FMA3)
DEF(sp_mix_add, SP_MIX_ADD)
DEF(sp_fma3_add, SP_FMA3_ADD)
DEF(sp_mullo_add, SP_MULLO_ADD)
DEF(bk_diff, BK_DIFF)
DEF(bk_same2, BK_SAME2)
DEF(bk_same2_fixed, BK_SAME2_FIXED)
DEF(bk_repeat, BK_REPEAT)
DEF(bk_samereg, BK_SAMEREG)
DEF(bk_same3_fixed, BK_SAME3_FIXED)
DEF(bk_same3_after, BK_SAME3_AFTER)
DEF(bk_same3_mixed, BK_SAME3_MIXED)
DEF(bk_same3_noreuse, BK_SAME3_NOREUSE)
DUAL_KERNEL(du_fmac_dep, PAIRS_DEP(DU_FMAC))

// Compiler-paired FMAC stream: plain C; a marker asm keeps the loop recognisable to the loop checker.
template <int U>
__global__ __launch_bounds__(32) void c_fmac(const uint64_t * __restrict__ in, uint64_t * __restrict__ out,
                                             uint32_t * __restrict__ sink) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = core::from_bits<float>(in[0]);
    }
    float k0 = core::from_bits<float>(in[1]), k1 = core::from_bits<float>(in[2]);
    asm volatile("" : "+v"(k0), "+v"(k1));
    for (int rep = 0; rep < core::REPS; ++rep) {
        const uint64_t t0 = wave_cycles();
#pragma unroll 1
        for (int it = 0; it < core::ITERS; ++it) {
            asm volatile("; c_fmac block" ::);
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    x[i] = __builtin_fmaf(k0, k1, x[i]);
                }
            }
        }
        const uint64_t t1 = wave_cycles();
        if (threadIdx.x == 0) {
            out[rep] = cycle_diff(t0, t1);
        }
    }
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
        s ^= core::fold(x[i]);
    }
    sink[threadIdx.x] = s;
}

struct T {
    const char * id;
    const char * what;
    core::Kern k[3];
    double expect;  // cycles per pattern if the claim in "what" holds; 0 = information only
};

#define K3(id) \
    { id<core::ULO>, id<core::UMID>, id<core::UHI> }

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = core::flag(argc, argv, "--break");
    // Dual patterns: 1 cycle each if both halves issue together (literal-carrying and VCC-reading pairs are reported
    // without an expectation: measured slower, see the measurement file). Split and not-allowed pairs: 2 cycles (two
    // instructions; mul_lo is quarter rate so 5). Bank tests: 2 instructions per pattern, 2 cycles if banks are free.
    std::vector<T> ts = {
        {"du_fmac", "v_dual_fmac_f32 x2", K3(du_fmac), 1},
        {"du_fmaak", "v_dual_fmaak_f32 x2 (shared literal, 96-bit instruction)", K3(du_fmaak), 0},
        {"du_fmamk", "v_dual_fmamk_f32 x2 (shared literal, 96-bit instruction)", K3(du_fmamk), 0},
        {"du_mul", "v_dual_mul_f32 x2", K3(du_mul), 1},
        {"du_add", "v_dual_add_f32 x2", K3(du_add), 1},
        {"du_sub", "v_dual_sub_f32 x2", K3(du_sub), 1},
        {"du_subrev", "v_dual_subrev_f32 x2", K3(du_subrev), 1},
        {"du_muldx9", "v_dual_mul_dx9_zero_f32 x2", K3(du_muldx9), 1},
        {"du_mov", "v_dual_mov_b32 x2", K3(du_mov), 1},
        {"du_cnd", "v_dual_cndmask_b32 x2 (VCC)", K3(du_cnd), 0},
        {"du_max", "v_dual_max_num_f32 x2", K3(du_max), 1},
        {"du_min", "v_dual_min_num_f32 x2", K3(du_min), 1},
        {"du_dotf16", "v_dual_dot2acc_f32_f16 x2", K3(du_dotf16), 1},
        {"du_dotbf16", "v_dual_dot2acc_f32_bf16 x2", K3(du_dotbf16), 1},
        {"du_addnc", "v_dual_add_f32 :: v_dual_add_nc_u32", K3(du_addnc), 1},
        {"du_lshl", "v_dual_add_f32 :: v_dual_lshlrev_b32", K3(du_lshl), 1},
        {"du_and", "v_dual_add_f32 :: v_dual_and_b32", K3(du_and), 1},
        {"du_fmac_addnc", "v_dual_fmac_f32 :: v_dual_add_nc_u32", K3(du_fmac_addnc), 1},
        {"du_mul_samesrc0", "v_dual_mul_f32 x2, both src0 = v1 (gfx12 same-VGPR exception)", K3(du_mul_samesrc0), 1},
        {"sp_fmac", "v_fmac_f32 + v_fmac_f32 as two instructions", K3(sp_fmac), 2},
        {"sp_mul", "v_mul_f32 + v_mul_f32 as two instructions", K3(sp_mul), 2},
        {"sp_addnc", "v_add_f32 + v_add_nc_u32 as two instructions", K3(sp_addnc), 2},
        {"sp_dotf16", "v_dot2_f32_f16 x2 as two VOP3P instructions", K3(sp_dotf16), 2},
        {"sp_fmaak", "2 x v_fmaak_f32 (VOP2 + literal, 64-bit instructions)", K3(sp_fmaak), 0},
        {"sp_fma3lit", "2 x v_fma_f32 with a literal (VOP3 + literal, 96-bit instructions)", K3(sp_fma3lit), 0},
        {"sp_fma3nolit", "2 x v_fma_f32 with an inline constant (64-bit instructions)", K3(sp_fma3nolit), 2},
        {"tr_exp2", "2 x v_exp_f32 (known: 4 cycles each)", K3(tr_exp2), 8},
        {"tr_exp_fma", "v_exp_f32 + 1 v_fma_f32 (4 if they overlap, 5 if not)", K3(tr_exp_fma), 0},
        {"tr_exp_fma3", "v_exp_f32 + 3 v_fma_f32 (4 if they overlap, 7 if not)", K3(tr_exp_fma3), 0},
        {"sp_mix_add", "v_fma_mix_f32 + v_add_f32 (fma_mix has no dual form)", K3(sp_mix_add), 2},
        {"sp_fma3_add", "v_fma_f32 (VOP3) + v_add_f32", K3(sp_fma3_add), 2},
        {"sp_mullo_add", "v_mul_lo_u32 + v_add_f32", K3(sp_mullo_add), 0},
        {"bk_diff", "2 x v_fma_f32, sources in three banks", K3(bk_diff), 2},
        {"bk_same2", "2 x v_fma_f32, two sources in bank 1 (changing registers)", K3(bk_same2), 2},
        {"bk_same2_fixed", "2 x v_fma_f32, two sources in bank 1 (same registers)", K3(bk_same2_fixed), 2},
        {"bk_repeat", "2 x v_fma_f32, (v1,v1,v5) then (v9,v9,v13)", K3(bk_repeat), 2},
        {"bk_samereg", "2 x v_fma_f32, one register read three times", K3(bk_samereg), 2},
        {"bk_same3_fixed", "2 x v_fma_f32, three bank-1 registers, same triple in both", K3(bk_same3_fixed), 0},
        {"bk_same3_after", "v_fma_f32 on banks 2/3, then one on three bank-1 registers", K3(bk_same3_after), 0},
        {"bk_same3_mixed", "(v1,v5,v9) then (v13,v17,v2)", K3(bk_same3_mixed), 0},
        {"bk_same3_noreuse", "(v1,v5,v9) then (v13,v17,v21): two different bank-1 triples", K3(bk_same3_noreuse), 0},
        {"c_fmac", "compiler-built: 16 chains x = fmaf(k0, k1, x), dual issue on (pattern = one FMA)", K3(c_fmac), 0.5},
        {"du_fmac_dep", "v_dual_fmac_f32 x2, every pattern on the same chain pair (dependent)", K3(du_fmac_dep), 0},
    };
    if (brk) {
        // Present the dependent stream as the independent one: the verdict must fail.
        ts = {{"du_fmac", "BROKEN: dependent stream presented as independent", K3(du_fmac_dep), 1}};
    }
    core::set_init(core::Init{core::f32(1.0f), core::f32(1.0f), core::f32(0.0f)});
    bool ok = true;
    int checked = 0, failed = 0;
    std::string fails;
    for (const T & t : ts) {
        // One re-measure on a mismatch: another process using the GPU can disturb a single measurement. Both are
        // printed; the second decides.
        core::Cost r;
        bool lin = false, pass = false;
        for (int attempt = 0; attempt < 2 && !pass; ++attempt) {
            if (attempt) {
                std::printf("RESULT dual %-16s first measurement %.3f (mid %.1f), measuring again\n", t.id, r.per, r.mid);
            }
            r = core::pattern_cost(t.k[0], t.k[1], t.k[2]);
            lin = std::fabs(r.mid) <= 1.5;
            pass = lin && (t.expect <= 0 || std::fabs(r.per - t.expect) <= 0.05 * t.expect);
        }
        if (t.expect > 0) {
            ++checked;
        }
        if (!pass) {
            ++failed;
            fails += std::string(" ") + t.id;
        }
        std::printf("RESULT dual %-16s cycles_per_pattern %.3f [%.3f-%.3f] loop %.0f per_iteration U4/6/8 %.1f %.1f %.1f%s%s"
                    "  (%s)\n", t.id, r.per, r.lo, r.hi, r.ovh, r.it_lo, r.it_mid, r.it_hi,
                    t.expect > 0 ? (pass ? " ok" : " MISMATCH") : "", lin ? "" : " NONLINEAR", t.what);
    }
    ok = failed == 0;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%d patterns with an expected rate, %d mismatches or non-linear loops:%s", checked,
                  failed, failed ? fails.c_str() : " none");
    verdict("core_dual", ok ? "PASS" : "FAIL", buf);
    return ok ? 0 : 1;
}
