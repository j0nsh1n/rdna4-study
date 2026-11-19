r"""Check the timed loops of the memory probes in an AMDGPU .lto.s file (from hipcc -save-temps).

Usage: python3 mem_loopcheck.py <file.s> <expect-file>
Reuses the kernel and innermost-loop finder of core_loopcheck.py. For every kernel prints its innermost loops as an
opcode histogram, its VGPR count (.amdhsa_next_free_vgpr) and its scratch size, then checks the expectation lines:

    <kernel regex> loop|kernel <instr regex> <op> <n>     count of instructions in the innermost loops (summed) or
                                                           in the whole kernel whose text matches the regex from its
                                                           start (no spaces in the regex: use \s), op >= <= ==
    <kernel regex> vgpr <op> <n>                          the kernel's VGPR count
    <kernel regex> scratch == 0                           no private segment

A kernel regex must match at least one kernel. Exit status 1 if any expectation fails.
The memory probes' loops are compiler-built (with inline-assembly loads in some), so the core check of "no added
instructions" does not apply; what matters is which memory instructions, how many, and no spills.
"""
import collections
import operator
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_loopcheck import kernels, loops  # noqa: E402

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def ops_in(lines):
    out = []
    for ln in lines:
        t = ln.strip()
        if not t or t.startswith((";", ".")) or t.endswith(":"):
            continue
        out.append(t)
    return out


def descriptors(text):
    vg, sc = {}, {}
    for m in re.finditer(r"\.amdhsa_kernel (\S+)(.*?)\.end_amdhsa_kernel", text, re.S):
        v = re.search(r"\.amdhsa_next_free_vgpr (\d+)", m.group(2))
        s = re.search(r"\.amdhsa_private_segment_fixed_size (\d+)", m.group(2))
        vg[m.group(1)] = int(v.group(1)) if v else -1
        sc[m.group(1)] = int(s.group(1)) if s else -1
    return vg, sc


def main():
    path, expect = sys.argv[1], sys.argv[2]
    text = open(path).read()
    vg, sc = descriptors(text)
    info = {}
    for name, body in kernels(text.splitlines()):
        inner = loops(body)
        loop_ops = []
        for s, e in inner:
            loop_ops.extend(ops_in(body[s:e + 1]))
        info[name] = (ops_in(body), loop_ops, len(inner))
        hist = collections.Counter(o.split()[0] for o in loop_ops)
        mem = " ".join(f"{k}x{v}" for k, v in sorted(hist.items()) if re.match(r"(global|buffer|scratch|s_load|s_prefetch|s_wait_loadcnt|s_wait_storecnt)", k))
        print(f"{name}: vgpr {vg.get(name, -1)} scratch {sc.get(name, -1)} inner_loops {len(inner)} "
              f"loop_instructions {len(loop_ops)} memory/waits in loops: [{mem}]")
    bad = 0
    for raw in open(expect):
        ln = raw.split("#")[0].strip()
        if not ln:
            continue
        f = ln.split()
        rx = re.compile(f[0])
        names = [n for n in info if rx.search(n)]
        if not names:
            print(f"FAIL no kernel matches {f[0]}")
            bad += 1
            continue
        for n in names:
            if f[1] == "vgpr":
                got, op, want = vg.get(n, -1), f[2], int(f[3])
            elif f[1] == "scratch":
                got, op, want = sc.get(n, -1), f[2], int(f[3])
            else:
                allops, loop_ops, _ = info[n]
                pool = loop_ops if f[1] == "loop" else allops
                orx = re.compile(f[2])
                got, op, want = sum(1 for o in pool if orx.match(o)), f[3], int(f[4])
            ok = OPS[op](got, want)
            if not ok:
                bad += 1
            print(f"{'ok' if ok else 'FAIL'} {n}: {' '.join(f[1:])} (got {got})")
    print(f"mem_loopcheck: {len(info)} kernels, {bad} expectations failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
