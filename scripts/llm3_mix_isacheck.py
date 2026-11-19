#!/usr/bin/env python3
"""ISA check of the three loops of src/power_mix.hip (round 3, Q1c): the instruction mix is what the probe claims, per weight.
Usage: llm3_mix_isacheck.py <fp8.s> <q8.s> <read.s> [<fp8row.s> <q8row.s>] [--break]   (the kernels disassembled by llvm-objdump --disassemble-symbols)
Row variants (whole-kernel counts, they have a block reduction): fp8row has 16 v_fma_mix_f32, one global_load_b128 of weights plus 4 of activations, a barrier and
LDS accesses (ds_), no loop needed; q8row has a loop with two v_dot4_i32_iu8, a barrier and LDS accesses.
Expected (from the real kernels' ISA, run q1/q1-isa-counts.txt): fp8 loop: 16 v_fma_mix_f32 and 30 to 45 integer bit ops per 16 weights, one 128-bit
non-temporal load of weights; q8 loop: two v_dot4_i32_iu8 per 8 weights, no v_fma_mix with an F16 pair from shifts (fp8's), at most 10 integer bit ops;
read loop: one 128-bit load, no FMA, no dot. --break expects THREE dots in the q8 loop (the code has two): must FAIL (exit 1)."""
import importlib.util
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("isaclass", os.path.join(here, "llm3_isaclass.py"))
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)


def loop_ops(path):
    ins = ic.parse(path)
    lp = ic.loops(ins)
    if not lp:
        return None
    i, j = lp[0]
    return [o for _, o, _ in ins[i : j + 1]]


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    brk = "--break" in sys.argv
    if len(a) not in (3, 5):
        print(__doc__)
        return 2
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += not ok

    f, q, r = (loop_ops(p) for p in a[:3])
    if None in (f, q, r):
        print("FAIL a loop was not found")
        return 1
    bit = lambda ops: sum(1 for o in ops if o.startswith(("v_and", "v_or", "v_xor", "v_lshl", "v_lshr", "v_ashr", "v_not", "v_perm", "v_bfe", "v_alignbit", "v_add_nc_u32_e32")) or o in ("v_and_or_b32", "v_or3_b32"))
    check("fp8 loop: 16 v_fma_mix_f32 (found %d)" % f.count("v_fma_mix_f32"), f.count("v_fma_mix_f32") == 16)
    check("fp8 loop: 30 to 60 integer bit ops (found %d)" % bit(f), 30 <= bit(f) <= 60)
    check("fp8 loop: exactly one global_load_b128 of weights plus 4 of activations (found %d)" % f.count("global_load_b128"), f.count("global_load_b128") == 5)
    want_dots = 3 if brk else 2
    check("q8 loop: %d v_dot4_i32_iu8 (found %d)%s" % (want_dots, q.count("v_dot4_i32_iu8"), " (deliberately wrong expectation)" if brk else ""), q.count("v_dot4_i32_iu8") == want_dots)
    check("q8 loop: no shift-widening FMAs (v_fma_mix count %d at most 2)" % q.count("v_fma_mix_f32"), q.count("v_fma_mix_f32") <= 2)
    check("q8 loop: at most 10 integer bit ops (found %d)" % bit(q), bit(q) <= 10)
    check("read loop: one global_load_b128, no FMA, no dot", r.count("global_load_b128") == 1 and not any(("fma" in o or "dot" in o) for o in r))
    if len(a) == 5:
        fr = [o for _, o, _ in ic.parse(a[3])]
        qr = [o for _, o, _ in ic.parse(a[4])]
        check("fp8row: 16 v_fma_mix_f32 (found %d)" % fr.count("v_fma_mix_f32"), fr.count("v_fma_mix_f32") == 16)
        check("fp8row: 1 + 4 global_load_b128 (found %d)" % fr.count("global_load_b128"), fr.count("global_load_b128") == 5)
        check("fp8row: a barrier and LDS accesses (barrier %d, ds_ %d)" % (sum(o.startswith("s_barrier") for o in fr), sum(o.startswith("ds_") for o in fr)), any(o.startswith("s_barrier") for o in fr) and any(o.startswith("ds_") for o in fr))
        want_dots = 3 if brk else 2
        check("q8row: %d v_dot4_i32_iu8 in the loop (found %d)%s" % (want_dots, qr.count("v_dot4_i32_iu8"), " (deliberately wrong expectation)" if brk else ""), qr.count("v_dot4_i32_iu8") == want_dots)
        check("q8row: a barrier and LDS accesses (barrier %d, ds_ %d)" % (sum(o.startswith("s_barrier") for o in qr), sum(o.startswith("ds_") for o in qr)), any(o.startswith("s_barrier") for o in qr) and any(o.startswith("ds_") for o in qr))
        check("q8row: a loop exists", bool(ic.loops(ic.parse(a[4]))))
    print("VERDICT power_mix-isa " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
