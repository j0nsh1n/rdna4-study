#!/usr/bin/env python3
# Writes src/study/left_core_blocks.h: the instruction blocks timed by src/study/left_core.hip (leftovers C, lane A:
# SC-10 wave64 issue, EC-5 register-bank stall, EC-6 runs of 12-byte instructions, EC-8 SGPR-writing VALU).
# Usage: scripts/leftc_gen_core.py <output header>   (scripts/study-leftovers-c-a.sh checks the committed header
# matches). Registers are fixed here so the register banks (register number mod 4) are part of each block.
# Each block is X(name, n, "text"): n instructions per block (a whole number of periods of its pattern).
import sys

r16 = list(range(16, 32))
W = {}   # SC-10: built for wave32 and wave64
E = {}   # EC items: wave32 only (32-bit SGPR destinations)


def rep(period, n_min=16):
    """A pattern of `period` instruction texts repeated to a block of at least n_min instructions."""
    k = -(-n_min // len(period))
    return period * k


# ---- SC-10 (wave64 against wave32) ----
W["w_fma_ic"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r) for r in r16]
W["w_dep"] = ["v_fma_f32 v16, v16, 0.5, 0.5"] * 16
W["w_fma_v1"] = ["v_fma_f32 v%d, v%d, v32, 0.5" % (r, r) for r in r16]
W["w_fma_vv"] = ["v_fma_f32 v%d, v%d, v32, v33" % (r, r) for r in r16]
# two constants in the two banks after the chain's own: at most one register per bank
W["w_fma_vv_spread"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (r, r, 32 + (r + 1) % 4, 32 + (r + 2) % 4) for r in r16]
W["w_fmac"] = ["v_fmac_f32 v%d, v32, v33" % r for r in r16]
W["w_fmac_spread"] = ["v_fmac_f32 v%d, v%d, v%d" % (r, 32 + (r + 1) % 4, 32 + (r + 2) % 4) for r in r16]
W["w_fmac_e64"] = ["v_fmac_f32_e64 v%d, v32, v33" % r for r in r16]
W["w_fmac_ic"] = ["v_fmac_f32 v%d, 0.5, v%d" % (r, 32 + (r + 1) % 4) for r in r16]
W["w_add_e32"] = ["v_add_f32 v%d, 0.5, v%d" % (r, r) for r in r16]
W["w_add_e64"] = ["v_add_f32_e64 v%d, 0.5, v%d" % (r, r) for r in r16]
W["w_add_vv"] = ["v_add_f32 v%d, v%d, v%d" % (r, 32 + (r + 1) % 4, r) for r in r16]
W["w_mul_e32"] = ["v_mul_f32 v%d, 0.5, v%d" % (r, r) for r in r16]
W["w_mov"] = ["v_mov_b32 v%d, v%d" % (r, 32 + (r + 1) % 4) for r in r16]
W["w_mov_ic"] = ["v_mov_b32 v%d, 0.5" % r for r in r16]
W["w_addu"] = ["v_add_nc_u32 v%d, 1, v%d" % (r, r) for r in r16]
W["w_fmaak"] = ["v_fmaak_f32 v%d, v%d, v%d, 0x3f400000" % (r, r, 32 + (r + 1) % 4) for r in r16]
W["w_pkfma"] = ["v_pk_fma_f16 v%d, v%d, v32, v33" % (r, r) for r in r16]
W["w_cvt"] = ["v_cvt_f32_i32 v%d, v%d" % (r, r) for r in r16]
W["w_exp"] = ["v_exp_f32 v%d, v%d" % (r, r) for r in r16]
W["w_fma_f64"] = ["v_fma_f64 v[%d:%d], v[%d:%d], 0.5, 0.5" % (r, r + 1, r, r + 1) for r in range(16, 32, 2)] * 2
# three different registers of bank 1, the same triple in every instruction (wave32: 2.968 per pair)
W["w_bank3_same"] = ["v_fma_f32 v%d, v33, v37, v41" % r for r in r16]
# two different bank-1 triples alternating (wave32: 9.999 per pair)
W["w_bank3_alt"] = [("v_fma_f32 v%d, v33, v37, v41" if i % 2 == 0 else "v_fma_f32 v%d, v45, v49, v53") % r
                    for i, r in enumerate(r16)]
# known answer in both wave sizes: chained 16x16x16 FP16 matrix instruction, 16.000 cycles (matrix-timing, sched item 4)
# (the operand widths differ between the wave sizes, so each build gets its own block)
W32 = {"w_wmma": ["v_wmma_f32_16x16x16_f16 v[40:47], v[48:51], v[52:55], v[40:47]"] * 16}
W64 = {"w_wmma": ["v_wmma_f32_16x16x16_f16 v[40:43], v[48:49], v[52:53], v[40:43]"] * 16}

# ---- EC-5: the bank stall. X reads three different registers of one bank (bank 1); "fresh" uses a different
# triple each time (execution-core 8g: 5.0 cycles each), "same" one triple (item 3: 1.48 each). Between two X: k
# ordinary FMAs (inline constants, one VGPR read) or k FMAs with two SGPR sources in different SGPR banks.
TRIPLES = [(57, 61, 65), (69, 73, 77), (81, 85, 89), (93, 97, 101)]
ORD = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r) for r in (16, 18, 20, 22, 24, 26, 28, 30)]
SG = ["v_fma_f32 v%d, v%d, s40, s42" % (r, r) for r in (16, 18, 20, 22, 24, 26, 28, 30)]
for kind, fill in (("o", ORD), ("s", SG)):
    for k in (0, 1, 3, 7, 15):
        if kind == "s" and k not in (1, 3):
            continue
        for tri in ("fresh", "same"):
            seq = []
            for j in range(max(1, 16 // (k + 1))):
                t = TRIPLES[j % 4] if tri == "fresh" else TRIPLES[0]
                seq.append("v_fma_f32 v%d, v%d, v%d, v%d" % (17 + 2 * (j % 4), t[0], t[1], t[2]))
                seq += [fill[(j * k + i) % 8] for i in range(k)]
            E["bk_%s_%s_k%d" % (tri, kind, k)] = seq
# the two known pair costs (execution-core item 3) and the one-bank FMA of 8g
E["bk_cal_same"] = ["v_fma_f32 v%d, v33, v37, v41" % r for r in r16]
E["bk_cal_alt"] = [("v_fma_f32 v%d, v33, v37, v41" if i % 2 == 0 else "v_fma_f32 v%d, v45, v49, v53") % r
                   for i, r in enumerate(r16)]
E["bk_cal_mixed"] = [("v_fma_f32 v%d, v33, v37, v41" if i % 2 == 0 else "v_fma_f32 v%d, v45, v49, v34") % r
                     for i, r in enumerate(r16)]

# ---- EC-6: runs of L 12-byte instructions (VOP3 with a real literal) between G 8-byte ones (inline constants)
L12 = "v_fma_f32 v%d, v%d, 0x3f400000, 0.5"
L8 = "v_fma_f32 v%d, v%d, 0.5, 0.5"
for L in (1, 2, 3, 4, 6, 8, 12, 16):
    for G in (0, 1, 2, 3, 4, 8, 16):
        if (G == 0 and L != 16) or (G == 3 and L != 1):
            continue
        per = ["12"] * L + ["8"] * G
        seq = rep(per)
        E["run_L%d_G%d" % (L, G)] = [(L12 if s == "12" else L8) % (16 + i % 16, 16 + i % 16) for i, s in enumerate(seq)]

# ---- EC-8: n consecutive SGPR-writing VALU instructions, then m FMAs (inline constants)
SW = {
    "rfl": "v_readfirstlane_b32 s%d, v%d",
    "cmp": "v_cmp_gt_f32_e64 s%d, v%d, 0.5",
    "sexp": "v_s_exp_f32 s%d, s40",
}
for nm, t in SW.items():
    for n in (1, 2, 3, 4, 8, 16):
        for m in (0, 1, 2, 4):
            if n == 16 and m != 0:
                continue
            per = []
            for i in range(n):
                r = 16 + i
                per.append(t % (r, r) if "%d, v%d" in t else t % r)
            per += ["FMA"] * m
            # the FMAs rotate over 16 registers through the whole block, so none depends on a recent one
            blk, f = [], 0
            for x in rep(per):
                if x == "FMA":
                    blk.append("v_fma_f32 v%d, v%d, 0.5, 0.5" % (32 + f % 16, 32 + f % 16))
                    f += 1
                else:
                    blk.append(x)
            E["sw_%s_n%d_m%d" % (nm, n, m)] = blk
    # destination patterns for n = 16: one register for all, even registers (pair-aligned), two alternating
    one = t % (16, 16) if "%d, v%d" in t else t % 16
    E["sw_%s_same" % nm] = [one] * 16
    E["sw_%s_even" % nm] = [(t % (16 + 2 * (i % 8), 16 + i) if "%d, v%d" in t else t % (16 + 2 * (i % 8))) for i in range(16)]
    E["sw_%s_alt2" % nm] = [(t % (16 + (i % 2), 16 + i) if "%d, v%d" in t else t % (16 + (i % 2))) for i in range(16)]

# ---- EC-1, EC-2a to EC-2e: instructions not timed by execution-core (wave32). _c16: 16 independent results, _c1: one
# dependent chain (latency), _c8 for FP64 (8 register pairs).
E["ins_permlane64_c16"] = ["v_permlane64_b32 v%d, v%d" % (r, r) for r in r16]
E["ins_permlane64_c1"] = ["v_permlane64_b32 v16, v16"] * 16
for f in ("rcp", "rsq", "sqrt", "log", "exp"):
    for t in ("f16", "f32"):
        E["ins_s_%s_%s_c16" % (f, t)] = ["v_s_%s_%s s%d, s40" % (f, t, r) for r in r16]
        # the chain waits for the SGPR write before the next VALU reads it, as the compiler emits it (core_valu PSEUDO)
        E["ins_s_%s_%s_c1" % (f, t)] = ["v_s_%s_%s s16, s16\ns_wait_alu depctr_va_sdst(0)" % (f, t)] * 16
for f in ("sqrt", "rsq", "rcp"):
    E["ins_%s_f64_c8" % f] = ["v_%s_f64 v[%d:%d], v[%d:%d]" % (f, 16 + 2 * (i % 8), 17 + 2 * (i % 8), 16 + 2 * (i % 8), 17 + 2 * (i % 8)) for i in range(16)]
    E["ins_%s_f64_c1" % f] = ["v_%s_f64 v[16:17], v[16:17]" % f] * 16
for nm, op in (("dot2f16f16", "v_dot2_f16_f16"), ("dot2bf16bf16", "v_dot2_bf16_bf16"), ("dot2f32f16", "v_dot2_f32_f16")):
    E["ins_%s_c16" % nm] = ["%s v%d, v32, v33, v%d" % (op, r, r) for r in r16]
    E["ins_%s_c1" % nm] = ["%s v16, v32, v33, v16" % op] * 16
# 64-bit integer add as the pair the compiler emits: carry through VCC, or through an SGPR (one per pair)
E["ins_add64_vcc_c8"] = sum([["v_add_co_u32 v%d, vcc_lo, v%d, v32" % (16 + 2 * i, 16 + 2 * i),
                              "v_add_co_ci_u32 v%d, vcc_lo, v%d, v33, vcc_lo" % (17 + 2 * i, 17 + 2 * i)] for i in range(8)], [])
E["ins_add64_sgpr_c8"] = sum([["v_add_co_u32 v%d, s%d, v%d, v32" % (16 + 2 * i, 40 + i, 16 + 2 * i),
                               "v_add_co_ci_u32_e64 v%d, null, v%d, 0, s%d" % (17 + 2 * i, 17 + 2 * i, 40 + i)] for i in range(8)], [])
E["ins_add64_vcc_c1"] = ["v_add_co_u32 v16, vcc_lo, v16, v32", "v_add_co_ci_u32 v17, vcc_lo, v17, v33, vcc_lo"] * 8
E["ins_divpair_c8"] = sum([["v_div_scale_f32 v%d, vcc_lo, v32, v33, v32" % (16 + 2 * i),
                            "v_div_fmas_f32 v%d, v32, v33, v34" % (17 + 2 * i)] for i in range(8)], [])
E["ins_divscale_c16"] = ["v_div_scale_f32 v%d, vcc_lo, v32, v33, v32" % r for r in r16]
E["ins_divfmas_c16"] = ["v_div_fmas_f32 v%d, v32, v33, v34" % r for r in r16]

# ---- T1 (fix round, src/study/left_bank.hip, both wave sizes): register-bank blocks. Kept out of LEFT_W_BLOCKS so the
# left_core objects (and the code addresses that EC-7 depends on) stay as they were. Calibration copies of three W
# blocks: independent FMA (1.000 in both sizes), dependent FMA (5.000), and the 8-of-16 block (wave64 1.250, wave32
# 1.000; run 20261117T201152Z).
T = {}
T["t_fma_ic"] = W["w_fma_ic"]
T["t_dep"] = W["w_dep"]
T["t_fma_vv"] = W["w_fma_vv"]
# T1 (fix round): every instruction of the block has two VGPR sources in one bank. Three VGPR sources: the two
# constants in one bank (v32, v36) and the chains in banks 1-3; the chain and one constant in bank 0; v_fmac with its
# accumulator in bank 0. Two VGPR sources: the chain and v32 in bank 0 (FMA with an inline constant, VOP2 add). And one
# register read twice (src0 = src1).
B0 = [r for r in range(16, 84, 4) if r != 32][:16]          # bank 0, v32 left out
NB0 = [r for r in range(17, 60) if r % 4 and r not in (33, 37)][:16]  # banks 1-3, the constants left out
T["t_fma_kk_same"] = ["v_fma_f32 v%d, v%d, v32, v36" % (r, r) for r in NB0]
T["t_fma_vv_all"] = ["v_fma_f32 v%d, v%d, v32, v33" % (r, r) for r in B0]
T["t_fmac_all"] = ["v_fmac_f32 v%d, v32, v33" % r for r in B0]
T["t_fma_v1_all"] = ["v_fma_f32 v%d, v%d, v32, 0.5" % (r, r) for r in B0]
T["t_add_vv_all"] = ["v_add_f32 v%d, v32, v%d" % (r, r) for r in B0]
T["t_fma_xx"] = ["v_fma_f32 v%d, v%d, v%d, 0.5" % (r, r, r) for r in r16]
# after a first run (all-bank-0 chains cost 1.343 in both wave sizes): bank-0 writes only (sources in banks 1-3), bank-0
# reads only (destinations in banks 1-3), and all 16 instructions with two sources in one bank but the banks rotating:
# the chain and the first constant share bank r mod 4 (cc_rot), or the two constants share bank (r + 1) mod 4 (kk_rot)
T["t_wr_b0"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (d, s) for d, s in zip(B0, NB0)]
T["t_rd_b0"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (d, s) for d, s in zip(NB0, B0)]
T["t_fma_cc_rot"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (r, r, 84 + r % 4, 84 + (r + 1) % 4) for r in r16]
T["t_fma_kk_rot"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (r, r, 88 + (r + 1) % 4, 92 + (r + 1) % 4) for r in r16]

# ---- T3 (fix round): the FP64 and pseudo-scalar rows of EC-2 again, in kernels that set every input register to a
# chosen value first (left_bank mode insdata, wave32). dfx_*: FP64 with a fixed input pair v[48:49] (the input never changes);
# the ins_* copies read and write the same registers, so a chain's value moves unless it is a fixed point.
D = {}
for f in ("sqrt", "rsq", "rcp"):
    D["dat_dfx_%s_f64_c8" % f] = ["v_%s_f64 v[%d:%d], v[48:49]" % (f, 16 + 2 * (i % 8), 17 + 2 * (i % 8)) for i in range(16)]
    for c in ("c8", "c1"):
        D["dat_ins_%s_f64_%s" % (f, c)] = E["ins_%s_f64_%s" % (f, c)]
for f in ("rcp", "rsq", "sqrt", "log", "exp"):
    for t in ("f16", "f32"):
        for c in ("c16", "c1"):
            D["dat_ins_s_%s_%s_%s" % (f, t, c)] = E["ins_s_%s_%s_%s" % (f, t, c)]

out = ["// Generated by scripts/leftc_gen_core.py: the blocks of left_core.hip. Do not edit by hand.", "#pragma once"]
for macro, blocks in (("LEFT_W_BLOCKS", W), ("LEFT_W32_BLOCKS", W32), ("LEFT_W64_BLOCKS", W64), ("LEFT_E_BLOCKS", E),
                      ("LEFT_T_BLOCKS", T), ("LEFT_D_BLOCKS", D)):
    out.append("#define %s(X) \\" % macro)
    for k, v in blocks.items():
        assert 16 <= len(v) <= 32, (k, len(v))
        s = "\\n".join(v).replace("\n", "\\n") + "\\n"
        out.append('    X(%s, %d, "%s") \\' % (k, len(v), s))
    out.append("")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(len(W), "wave-size blocks,", len(E), "wave32 blocks,", len(T), "bank blocks,", len(D), "data blocks")
