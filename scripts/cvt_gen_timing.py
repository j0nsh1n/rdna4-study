"""Generate src/study/cvt_time_blocks.h: the inline-assembly blocks timed by src/study/cvt_time.hip.

Usage: python3 scripts/cvt_gen_timing.py > src/study/cvt_time_blocks.h

Each test is one block of P patterns with fixed registers (v64-v137, clobbered by the kernel), so the compiler cannot
change what is timed. Instruction tests: 16 patterns, one instruction each, either 16 independent destinations
("c16", the issue cost) or one dependent chain ("c1", the latency). Widening tests: 8 patterns, one 32-bit word
(4 FP8 weights) each, emitted step by step across the 8 words (as an unrolled, scheduled loop would), so that no
instruction waits for the one just before it.
Registers: v64-v95 destinations and temporaries, v96-v111 the weight or source words, v104-v127 temporaries of the
widening tests, v128-v135 the accumulators of the "use" tests, v136 and v137 constant operands (activations).
"""

TESTS = []  # (name, patterns, weights per pattern, description, block text)


def add(name, lines, patterns, wpp, desc):
    TESTS.append((name, patterns, wpp, desc, "\n".join(lines) + "\n"))


def inst(name, fmt_c16, fmt_c1, desc):
    add(name + "_c16", [fmt_c16.format(d=64 + i, s=96 + i, p=64 + 2 * i, q=65 + 2 * i) for i in range(16)], 16, 0,
        desc + ", 16 independent")
    add(name + "_c1", [fmt_c1] * 16, 16, 0, desc + ", one dependent chain")


inst("fma_f32", "v_fma_f32 v{d}, v{s}, v136, v{d}", "v_fma_f32 v64, v96, v136, v64", "v_fma_f32 (calibration)")
inst("cvt_f32_fp8", "v_cvt_f32_fp8 v{d}, v{s}", "v_cvt_f32_fp8 v64, v64", "v_cvt_f32_fp8 (VOP1, byte 0)")
inst("cvt_f32_fp8_sel", "v_cvt_f32_fp8_e64 v{d}, v{s} byte_sel:2", "v_cvt_f32_fp8_e64 v64, v64 byte_sel:2",
     "v_cvt_f32_fp8 (VOP3, byte_sel 2)")
inst("cvt_f32_bf8", "v_cvt_f32_bf8 v{d}, v{s}", "v_cvt_f32_bf8 v64, v64", "v_cvt_f32_bf8")
add("cvt_pk_f32_fp8_c16", ["v_cvt_pk_f32_fp8 v[{}:{}], v{}".format(64 + 2 * i, 65 + 2 * i, 96 + (i % 8))
                           for i in range(16)], 16, 0, "v_cvt_pk_f32_fp8, 16 independent")
add("cvt_pk_f32_fp8_c1", ["v_cvt_pk_f32_fp8 v[64:65], v64"] * 16, 16, 0, "v_cvt_pk_f32_fp8, one dependent chain")
add("cvt_pk_f32_bf8_c16", ["v_cvt_pk_f32_bf8 v[{}:{}], v{}".format(64 + 2 * i, 65 + 2 * i, 96 + (i % 8))
                           for i in range(16)], 16, 0, "v_cvt_pk_f32_bf8, 16 independent")
add("cvt_pk_f32_bf8_c1", ["v_cvt_pk_f32_bf8 v[64:65], v64"] * 16, 16, 0, "v_cvt_pk_f32_bf8, one dependent chain")
inst("cvt_pk_fp8_f32", "v_cvt_pk_fp8_f32 v{d}, v{s}, v136", "v_cvt_pk_fp8_f32 v64, v64, v136", "v_cvt_pk_fp8_f32")
inst("cvt_pk_bf8_f32", "v_cvt_pk_bf8_f32 v{d}, v{s}, v136", "v_cvt_pk_bf8_f32 v64, v64, v136", "v_cvt_pk_bf8_f32")
inst("cvt_sr_fp8_f32", "v_cvt_sr_fp8_f32 v{d}, v{s}, v136", "v_cvt_sr_fp8_f32 v64, v64, v136", "v_cvt_sr_fp8_f32")
inst("cvt_sr_bf8_f32", "v_cvt_sr_bf8_f32 v{d}, v{s}, v136", "v_cvt_sr_bf8_f32 v64, v64, v136", "v_cvt_sr_bf8_f32")
inst("cvt_pkrtz_f16_f32", "v_cvt_pk_rtz_f16_f32 v{d}, v{s}, v136", "v_cvt_pk_rtz_f16_f32 v64, v64, v136",
     "v_cvt_pk_rtz_f16_f32 (reference, lane 1)")
for a, b in (("fp8", "fp8"), ("fp8", "bf8"), ("bf8", "fp8"), ("bf8", "bf8")):
    inst("dot4_%s_%s" % (a, b), "v_dot4_f32_%s_%s v{d}, v{s}, v136, v{d}" % (a, b),
         "v_dot4_f32_%s_%s v64, v96, v136, v64" % (a, b), "v_dot4_f32_%s_%s (chain through the accumulator)" % (a, b))
inst("dot2_f32_f16", "v_dot2_f32_f16 v{d}, v{s}, v136, v{d}", "v_dot2_f32_f16 v64, v96, v136, v64",
     "v_dot2_f32_f16 (reference, lane 1)")
inst("dot2_f32_bf16", "v_dot2_f32_bf16 v{d}, v{s}, v136, v{d}", "v_dot2_f32_bf16 v64, v96, v136, v64",
     "v_dot2_f32_bf16 (reference, lane 1)")
inst("dot2_f16_f16", "v_dot2_f16_f16 v{d}, v{s}, v136, v{d}", "v_dot2_f16_f16 v64, v96, v136, v64", "v_dot2_f16_f16")
inst("dot2_bf16_bf16", "v_dot2_bf16_bf16 v{d}, v{s}, v136, v{d}", "v_dot2_bf16_bf16 v64, v96, v136, v64",
     "v_dot2_bf16_bf16")
inst("dot4_i32_iu8", "v_dot4_i32_iu8 v{d}, v{s}, v136, v{d} neg_lo:[1,1,0]",
     "v_dot4_i32_iu8 v64, v96, v136, v64 neg_lo:[1,1,0]", "v_dot4_i32_iu8 signed x signed")
inst("dot8_i32_iu4", "v_dot8_i32_iu4 v{d}, v{s}, v136, v{d} neg_lo:[1,1,0]",
     "v_dot8_i32_iu4 v64, v96, v136, v64 neg_lo:[1,1,0]", "v_dot8_i32_iu4 signed x signed")


def widen(name, steps, desc, consume=None):
    """steps: list of format strings over w (word reg), and per-word temporaries t0..t3 (v64+4j..) and u0..u2
    (v104+3j..). Emitted step-major over 8 words."""
    lines = []
    for st in steps + (consume or []):
        for j in range(8):
            lines.append(st.format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j, t3=67 + 4 * j,
                                   u0=104 + 3 * j, u1=105 + 3 * j, u2=106 + 3 * j, acc=64 + j))
    add(name, lines, 8, 4, desc)


# (a) four single-byte conversions to FP32
A = ["v_cvt_f32_fp8_e64 v{t0}, v{w} byte_sel:0", "v_cvt_f32_fp8_e64 v{t1}, v{w} byte_sel:1",
     "v_cvt_f32_fp8_e64 v{t2}, v{w} byte_sel:2", "v_cvt_f32_fp8_e64 v{t3}, v{w} byte_sel:3"]
# (b) two packed conversions to FP32 (word select 0 and 1)
B = ["v_cvt_pk_f32_fp8_e64 v[{t0}:{t1}], v{w}", "v_cvt_pk_f32_fp8_e64 v[{t2}:{t3}], v{w} op_sel:[1,0]"]
# (c) RDNA 3 shift widening to two FP16 pairs times 2^-8: even = (b0, b2), odd = (b1, b3)
C = ["v_lshlrev_b32 v{t0}, 8, v{w}", "v_lshlrev_b32 v{t1}, 7, v{w}", "v_lshrrev_b32 v{t2}, 1, v{w}",
     "v_and_b32 v{t1}, 0x3f803f80, v{t1}", "v_and_b32 v{t2}, 0x3f803f80, v{t2}",
     "v_and_or_b32 v{t0}, v{t0}, 0x80008000, v{t1}", "v_and_or_b32 v{t2}, v{w}, 0x80008000, v{t2}"]
# (d) two packed conversions to FP32, then two FP32-to-FP16 packs: (b0, b1) and (b2, b3)
D = B + ["v_cvt_pk_rtz_f16_f32 v{u0}, v{t0}, v{t1}", "v_cvt_pk_rtz_f16_f32 v{u1}, v{t2}, v{t3}"]
# (e) FP16 table in LDS (byte address 2 * code at LDS 0): pairs (b0, b1) in t2, (b2, b3) in t3
E = ["v_lshlrev_b32 v{t0}, 1, v{w}", "v_lshrrev_b32 v{t1}, 7, v{w}", "v_lshrrev_b32 v{u0}, 15, v{w}",
     "v_lshrrev_b32 v{u1}, 23, v{w}",
     "v_and_b32 v{t0}, 0x1fe, v{t0}", "v_and_b32 v{t1}, 0x1fe, v{t1}", "v_and_b32 v{u0}, 0x1fe, v{u0}",
     "v_and_b32 v{u1}, 0x1fe, v{u1}",
     "ds_load_u16_d16 v{t2}, v{t0}", "ds_load_u16_d16_hi v{t2}, v{t1}", "ds_load_u16_d16 v{t3}, v{u0}",
     "ds_load_u16_d16_hi v{t3}, v{u1}"]
widen("widen_a_cvt4", A, "(a) 4 x v_cvt_f32_fp8 -> 4 FP32")
widen("widen_b_pk2", B, "(b) 2 x v_cvt_pk_f32_fp8 -> 4 FP32")
widen("widen_c_shift", C, "(c) shifts and masks -> 2 FP16 pairs (x 2^-8)")
widen("widen_d_pk_rtz", D, "(d) 2 x v_cvt_pk_f32_fp8 + 2 x v_cvt_pk_rtz_f16_f32 -> 2 FP16 pairs")
lines = []
for st in E:
    for j in range(8):
        lines.append(st.format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j, t3=67 + 4 * j,
                               u0=104 + 3 * j, u1=105 + 3 * j, u2=106 + 3 * j))
lines.append("s_wait_dscnt 0")
add("widen_e_lds", lines, 8, 4, "(e) LDS FP16 table, 4 x ds_load_u16_d16(_hi) -> 2 FP16 pairs")
# consumers: FP32 weights into v_fmac_f32 against activation v136; FP16 pairs into v_dot2_f32_f16 against v137;
# one accumulator per word in v128-v135.
FA = ["v_fmac_f32 v{a}, v{{t0}}, v136", "v_fmac_f32 v{a}, v{{t1}}, v136", "v_fmac_f32 v{a}, v{{t2}}, v136",
      "v_fmac_f32 v{a}, v{{t3}}, v136"]


def consume(name, steps, cons, desc):
    lines = []
    for st in steps:
        for j in range(8):
            lines.append(st.format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j, t3=67 + 4 * j,
                                   u0=104 + 3 * j, u1=105 + 3 * j, u2=106 + 3 * j))
    if name.endswith("_lds"):
        lines.append("s_wait_dscnt 0")
    for st in cons:
        for j in range(8):
            lines.append(st.format(a=128 + j).format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j,
                                                     t3=67 + 4 * j, u0=104 + 3 * j, u1=105 + 3 * j))
    add(name, lines, 8, 4, desc)


DOT_EVEN = ["v_dot2_f32_f16 v{a}, v{{t0}}, v137, v{a}", "v_dot2_f32_f16 v{a}, v{{t2}}, v137, v{a}"]
DOT_D = ["v_dot2_f32_f16 v{a}, v{{u0}}, v137, v{a}", "v_dot2_f32_f16 v{a}, v{{u1}}, v137, v{a}"]
DOT_E = ["v_dot2_f32_f16 v{a}, v{{t2}}, v137, v{a}", "v_dot2_f32_f16 v{a}, v{{t3}}, v137, v{a}"]
consume("use_a_cvt4_fma", A, FA, "(a) + 4 x v_fmac_f32")
consume("use_b_pk2_fma", B, FA, "(b) + 4 x v_fmac_f32")
consume("use_c_shift_dot2", C, DOT_EVEN, "(c) + 2 x v_dot2_f32_f16")
consume("use_d_pk_rtz_dot2", D, DOT_D, "(d) + 2 x v_dot2_f32_f16")
consume("use_e_lds_dot2", E, DOT_E, "(e) + 2 x v_dot2_f32_f16")
consume("use_f_dot4", [], ["v_dot4_f32_fp8_fp8 v{a}, v{{w}}, v137, v{a}"], "(f) 1 x v_dot4_f32_fp8_fp8, no widening")

# ---- leftovers C (lane A), CD-1: the FP32 consumers as dual-issue pairs. Per word two accumulators v(128+2j) and
# v(129+2j) (even and odd, different banks); X takes the bank-0 and bank-1 weights with activation v136, Y the bank-2
# and bank-3 weights with activation v137, so every VOPD bank and parity rule holds (checked by the assembler).
DUAL = ["v_dual_fmac_f32 v{a0}, v{{t0}}, v136 :: v_dual_fmac_f32 v{a1}, v{{t2}}, v137",
        "v_dual_fmac_f32 v{a0}, v{{t1}}, v136 :: v_dual_fmac_f32 v{a1}, v{{t3}}, v137"]


def consume2(name, steps, cons, desc):
    lines = []
    for st in steps:
        for j in range(8):
            lines.append(st.format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j, t3=67 + 4 * j,
                                   u0=104 + 3 * j, u1=105 + 3 * j, u2=106 + 3 * j))
    for st in cons:
        for j in range(8):
            lines.append(st.format(a0=128 + 2 * j, a1=129 + 2 * j).format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j,
                                                                          t2=66 + 4 * j, t3=67 + 4 * j))
    add(name, lines, 8, 4, desc)


consume2("use_a_cvt4_dual", A, DUAL, "(a) + 2 x v_dual_fmac_f32 pairs (CD-1)")
consume2("use_b_pk2_dual", B, DUAL, "(b) + 2 x v_dual_fmac_f32 pairs (CD-1)")
# CD-2: the LDS table method with the loads issued ahead of their use. A block is two halves of 4 words; each half
# computes its addresses, issues its 16 loads, waits only until the other half's loads are back (s_wait_dscnt 16:
# at most this half's 16 outstanding) and consumes those, so a half's loads complete while the other half works.
lines = []
for half in (0, 1):
    me = range(4 * half, 4 * half + 4)
    other = range(4 * (1 - half), 4 * (1 - half) + 4)
    for st in E:
        for j in me:
            lines.append(st.format(w=96 + j, t0=64 + 4 * j, t1=65 + 4 * j, t2=66 + 4 * j, t3=67 + 4 * j,
                                   u0=104 + 3 * j, u1=105 + 3 * j, u2=106 + 3 * j))
    lines.append("s_wait_dscnt 16")
    for st in ["v_dot2_f32_f16 v{a}, v{t2}, v137, v{a}", "v_dot2_f32_f16 v{a}, v{t3}, v137, v{a}"]:
        for j in other:
            lines.append(st.format(a=128 + j, t2=66 + 4 * j, t3=67 + 4 * j))
add("use_e_lds_ahead", lines, 8, 4, "(e) with the loads issued a half-block ahead + 2 x v_dot2_f32_f16 (CD-2)")
# CD-4: stochastic rounding with a random-number generator in the loop, 16 independent streams: value v64+i (bank i),
# state v112 + (i+1) mod 16 (bank i+1, so the conversion's two sources sit in different banks), temporary v96+i,
# result v80+i. One pattern = one conversion. The _b0 forms keep value and state in one bank (state v112+i).
GEN = {
    # xorshift32: x ^= x << 13; x ^= x >> 17; x ^= x << 5 (6 VALU)
    "xorshift": ["v_lshlrev_b32 v{t}, 13, v{s}", "v_xor_b32 v{s}, v{s}, v{t}", "v_lshrrev_b32 v{t}, 17, v{s}",
                 "v_xor_b32 v{s}, v{s}, v{t}", "v_lshlrev_b32 v{t}, 5, v{s}", "v_xor_b32 v{s}, v{s}, v{t}"],
    # 24-bit multiply-add LCG (2 VALU)
    "lcg24": ["v_mul_u32_u24 v{s}, 0x19660d, v{s}", "v_add_nc_u32 v{s}, 0x3c6ef35f, v{s}"],
    # full 32-bit LCG (v_mul_lo_u32 is quarter rate, execution-core item 2: 4 cycles)
    "lcg32": ["v_mul_lo_u32 v{s}, 0x19660d, v{s}", "v_add_nc_u32 v{s}, 0x3c6ef35f, v{s}"],
    # one random word for four conversions: a shift of the previous word per conversion (1 VALU)
    "shift": ["v_lshrrev_b32 v{s}, 8, v{s}"],
}
for gname, gen in GEN.items():
    for suf, sh in (("", 1), ("_b0", 0)):
        lines = []
        for st in gen + ["v_cvt_sr_fp8_f32 v{d}, v{v}, v{s}"]:
            for i in range(16):
                lines.append(st.format(s=112 + (i + sh) % 16, t=96 + i, d=80 + i, v=64 + i))
        add("sr_" + gname + suf, lines, 16, 1, "v_cvt_sr_fp8_f32 + %s generator, 16 streams (CD-4)" % gname)


def cstr(text):
    return "\n".join('    "%s\\n"' % ln for ln in text.strip("\n").split("\n"))


print("// Generated by scripts/cvt_gen_timing.py; do not edit by hand.")
print("#pragma once")
for name, p, w, desc, text in TESTS:
    print("#define CVT_BLOCK_%s \\\n%s" % (name, " \\\n".join(cstr(text).split("\n"))))
init = (["v_mov_b32 v%d, %%0" % r for r in range(64, 136)] + ["v_mov_b32 v136, %1", "v_mov_b32 v137, %2"] +
        ["v_mov_b32 v%d, %%0" % r for r in range(138, 144)])
print("#define CVT_INIT \\\n%s" % " \\\n".join(cstr("\n".join(init)).split("\n")))
print("#define CVT_CLOBBERS %s" % ", ".join('"v%d"' % r for r in range(64, 144)))
print("#define CVT_TESTS(X) \\")
print(" \\\n".join('    X(%s, %d, %d, "%s")' % (n, p, w, d) for n, p, w, d, _ in TESTS))
