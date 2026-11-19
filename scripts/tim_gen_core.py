#!/usr/bin/env python3
# Writes src/study/tim_core_blocks.h: the 16-instruction blocks timed by tim_core.hip (timing leftovers, item 3).
# Usage: scripts/tim_gen_core.py <output header>   (scripts/study-leftovers-timing.sh checks the committed header
# matches). Registers are fixed here so the register banks (register number mod 4) are part of each block.
import sys
def V(r): return "v%d" % r
B = {}
r16 = list(range(16, 32))
B["fma_ic"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r) for r in r16]
B["fma_vv"] = ["v_fma_f32 v%d, v%d, v32, v33" % (r, r) for r in r16]
B["fma_v3bank"] = ["v_fma_f32 v%d, v%d, %s" % (r, r, "v34, v35" if r % 4 < 2 else "v32, v33") for r in r16]
B["fma_ss"] = ["v_fma_f32 v%d, v%d, s40, s41" % (r, r) for r in r16]
B["fma_ss2"] = ["v_fma_f32 v%d, v%d, s40, s42" % (r, r) for r in r16]
B["fma_v1"] = ["v_fma_f32 v%d, v%d, v32, 0.5" % (r, r) for r in r16]
B["fma_v1b2"] = ["v_fma_f32 v%d, v%d, v%d, 0.5" % (r, r, 34 if r % 4 < 2 else 32) for r in r16]
B["add_vop2"] = ["v_add_f32 v%d, 0.5, v%d" % (r, r) for r in r16]
B["fma_lit"] = ["v_fma_f32 v%d, v%d, 0x3f400000, 0.5" % (r, r) for r in r16]
pairs = [(16 + 2 * i, 17 + 2 * i) for i in range(8)] * 2
B["du_fmac"] = ["v_dual_fmac_f32 v%d, v32, v33 :: v_dual_fmac_f32 v%d, v34, v35" % p for p in pairs]
B["du_fmaak"] = ["v_dual_fmaak_f32 v%d, v32, v33, 0x3f800000 :: v_dual_fmaak_f32 v%d, v34, v35, 0x3f800000" % p for p in pairs]
B["du_fmaak_mov"] = ["v_dual_fmaak_f32 v%d, v32, v33, 0x3f800000 :: v_dual_mov_b32 v%d, v35" % p for p in pairs]
B["du_fmaak_samek"] = B["du_fmaak"]
B["du_mix"] = [B["du_fmaak"][i] if i % 2 == 0 else B["du_fmac"][i] for i in range(16)]
B["du_fmaak_nop"] = [x + "\ns_nop 0" for x in B["du_fmaak"]]
B["du_fmac_nop"] = [x + "\ns_nop 0" for x in B["du_fmac"]]
B["fma_lit_nop"] = [x + "\ns_nop 0" for x in B["fma_lit"]]
B["du_fmaak_ic"] = ["v_dual_fmaak_f32 v%d, 0.5, v33, 0x3f800000 :: v_dual_fmaak_f32 v%d, 0.5, v35, 0x3f800000" % p for p in pairs]
B["du_mov_lit"] = ["v_dual_mov_b32 v%d, 0x3f800000 :: v_dual_mov_b32 v%d, v35" % p for p in pairs]
B["du_mov"] = ["v_dual_mov_b32 v%d, v34 :: v_dual_mov_b32 v%d, v35" % p for p in pairs]
B["mov_lit"] = ["v_mov_b32 v%d, 0x3f400000" % r for r in r16]
# literal placement: 0x3f400000 (0.75) is a real 32-bit literal; 0x3f800000 (1.0) is encoded as an inline constant
B["du_mov_reallit"] = ["v_dual_mov_b32 v%d, 0x3f400000 :: v_dual_mov_b32 v%d, v35" % p for p in pairs]
B["add_lit"] = ["v_add_f32 v%d, 0x3f400000, v%d" % (r, r) for r in r16]
B["fmaak_vop2"] = ["v_fmaak_f32 v%d, v%d, v32, 0x3f400000" % (r, r) for r in r16]
B["fma_lit_2v"] = ["v_fma_f32 v%d, v%d, 0x3f400000, v32" % (r, r) for r in r16]
B["fma_lit_e64mov"] = ["v_mov_b32_e64 v%d, 0x3f400000" % r for r in r16]
B["add_e64"] = ["v_add_f32_e64 v%d, 0.5, v%d" % (r, r) for r in r16]
B["fma_lit_1in4"] = [("v_fma_f32 v%d, v%d, 0x3f400000, 0.5" if i % 4 == 0 else "v_fma_f32 v%d, v%d, 0.5, 0.5") % (16 + i, 16 + i) for i in range(16)]
B["s_add_sreg"] = ["s_add_co_i32 s%d, s%d, s40" % (r, r) for r in r16]
B["s_add_2sreg"] = ["s_add_co_i32 s%d, s41, s40" % r for r in r16]
# SGPR read banks: two scalar sources at different distances (bank model: 64-bit pairs, see tim_core single)
for a, b in [(40, 41), (40, 42), (40, 44), (40, 48), (40, 56), (40, 40), (41, 43)]:
    B["s2_%d_%d" % (a, b)] = ["s_add_co_i32 s%d, s%d, s%d" % (r, a, b) for r in r16]
B["s_add_s42"] = ["s_add_co_i32 s%d, s%d, s42" % (r, r) for r in r16]
B["s_add_s44"] = ["s_add_co_i32 s%d, s%d, s44" % (r, r) for r in r16]
# runs of 12-byte instructions between 8-byte ones (run length L of each)
L12 = "v_fma_f32 v%d, v%d, 0x3f400000, 0.5"
L8 = "v_fma_f32 v%d, v%d, 0.5, 0.5"
for L in (2, 4, 8):
    B["lit_run%d" % L] = [((L12 if (i // L) % 2 == 0 else L8) % (16 + i, 16 + i)) for i in range(16)]
B["lit_3of4"] = [((L8 if i % 4 == 3 else L12) % (16 + i, 16 + i)) for i in range(16)]
B["s_and_sreg"] = ["s_and_b32 s%d, s%d, s40" % (r, r) for r in r16]
# scalar
s16 = list(range(16, 32))
B["s_add"] = ["s_add_co_i32 s%d, s%d, 1" % (r, r) for r in s16]
B["s_add_lit"] = ["s_add_co_i32 s%d, s%d, 0x12345" % (r, r) for r in s16]
B["s_mov"] = ["s_mov_b32 s%d, 1" % r for r in s16]
B["s_mov_lit"] = ["s_mov_b32 s%d, 0x12345" % r for r in s16]
B["s_add4"] = ["s_add_co_i32 s%d, s%d, 1" % (16 + i % 4, 16 + i % 4) for i in range(16)]
B["s_cvt"] = ["s_cvt_f32_i32 s%d, s%d" % (r, r) for r in s16]
B["s_nop"] = ["s_nop 0"] * 16
B["s_add_fma"] = ["s_add_co_i32 s%d, s%d, 1\nv_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r, r, r) for r in s16]
B["s_add_x2"] = ["s_add_co_i32 s%d, s%d, 1\ns_add_co_i32 s%d, s%d, 1" % (r, r, r + 16, r + 16) for r in s16]
# VALU writing an SGPR
B["ps_exp"] = ["v_s_exp_f32 s%d, s40" % r for r in s16]
B["ps_exp_ic"] = ["v_s_exp_f32 s%d, 0.5" % r for r in s16]
B["ps_exp_fma"] = ["v_s_exp_f32 s%d, s40\nv_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r, r) for r in s16]
B["ps_exp_snop"] = ["v_s_exp_f32 s%d, s40\ns_nop 0" % r for r in s16]
B["ps_exp2"] = ["v_s_exp_f32 s%d, s40" % (16 + 2 * (i % 8)) for i in range(16)]
B["ps_exp_sadd"] = ["v_s_exp_f32 s%d, s40\ns_add_co_i32 s%d, s%d, 1" % (r, r + 16, r + 16) for r in s16]
B["v_exp"] = ["v_exp_f32 v%d, v%d" % (r, r) for r in r16]
B["cmp_sgpr"] = ["v_cmp_gt_f32_e64 s%d, v%d, 0.5" % (r, r) for r in s16]
B["cmp_vcc"] = ["v_cmp_gt_f32_e32 vcc_lo, 0.5, v%d" % r for r in r16]
B["readlane"] = ["v_readfirstlane_b32 s%d, v%d" % (r, r) for r in s16]
# write-after-write on one register, sources independent
B["waw1"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r) for r in r16]
B["waw2"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (16 + 2 * (i // 2), 16 + i) for i in range(16)]
B["waw4"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (16 + 4 * (i // 4), 16 + i) for i in range(16)]
B["waw8"] = ["v_fma_f32 v%d, v%d, 0.5, 0.5" % (16 + 8 * (i // 8), 16 + i) for i in range(16)]
B["waw16"] = ["v_fma_f32 v16, v%d, 0.5, 0.5" % (17 + i) for i in range(16)]
B["waw4_add"] = ["v_add_f32 v%d, 0.5, v%d" % (16 + 4 * (i // 4), 16 + i) for i in range(16)]
# multi-wave: dependent chains and s_delay_alu
B["dep"] = ["v_fma_f32 v16, v16, 0.5, 0.5"] * 16
B["dep_da"] = ["s_delay_alu instid0(VALU_DEP_1)\nv_fma_f32 v16, v16, 0.5, 0.5"] * 16
B["dep2_da"] = ["s_delay_alu instid0(VALU_DEP_2)\nv_fma_f32 v%d, v%d, 0.5, 0.5" % (16 + i % 2, 16 + i % 2) for i in range(16)]
B["ind_da"] = ["s_delay_alu instid0(VALU_DEP_1)\nv_fma_f32 v%d, v%d, 0.5, 0.5" % (r, r) for r in r16]
B["dep_add_da"] = ["s_delay_alu instid0(VALU_DEP_1)\nv_add_f32 v16, 0.5, v16"] * 16
B["dep_da_nop"] = ["s_delay_alu instid0(VALU_DEP_1)\nv_fma_f32 v16, v16, 0.5, 0.5\ns_nop 0"] * 16
# per-lane sources: acc = fma(a, b, acc), 8 accumulators
B["pl_good"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (16 + (i % 8), 25 + (i % 8), 34 + (i % 8), 16 + (i % 8)) for i in range(16)]
B["pl_twobank"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (16 + (i % 8), 32 + (i % 8), 41 + (i % 8), 16 + (i % 8)) for i in range(16)]
B["pl_onebank"] = ["v_fma_f32 v%d, v%d, v%d, v%d" % (16 + 4 * (i % 4), 32 + 4 * (i % 4), 48 + 4 * (i % 4), 16 + 4 * (i % 4)) for i in range(16)]
# VOPD pairs of FMACs with per-lane sources (the shape clang gives compiler-and-hazards' acc8var loop): "bad" reads
# three different registers of one bank across the pair (a, a+16, a+32), "good" at most two per bank
accs = [16 + 2 * i for i in range(8)] * 2
B["du_pl_bad"] = ["v_dual_fmac_f32 v%d, v%d, v%d :: v_dual_fmac_f32 v%d, v%d, v%d" % (a, a + 16, a + 35, a + 1, a + 17, a + 32) for a in accs]
B["du_pl_good"] = ["v_dual_fmac_f32 v%d, v%d, v%d :: v_dual_fmac_f32 v%d, v%d, v%d" % (a, a + 17, a + 34, a + 1, a + 19, a + 32) for a in accs]
B["pl_fmac"] = ["v_fmac_f32 v%d, v%d, v%d" % (16 + (i % 8), 25 + (i % 8), 34 + (i % 8)) for i in range(16)]
del B["du_fmaak_samek"]
out = ["// Generated by scripts/tim_gen_core.py: the 16-instruction blocks of tim_core.hip. Do not edit by hand.",
       "#pragma once", "#define TIM_CORE_BLOCKS(X) \\"]
for k, v in B.items():
    assert len(v) == 16, k
    s = "\\n".join(v).replace("\n", "\\n") + "\\n"
    out.append('    X(%s, "%s") \\' % (k, s))
out.append("")
# the code-size sweep (tim_core size): instruction lengths 4, 8 and 12 bytes
out.append("#define TIM_SIZE_BLOCKS(X) \\")
for k in ["fma_ic", "fma_lit", "du_fmac", "du_fmaak", "add_vop2", "s_nop"]:
    out.append('    X(%s, "%s") \\' % (k, "\\n".join(B[k]).replace("\n", "\\n") + "\\n"))
out.append("")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(len(B), "blocks")
