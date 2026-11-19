#!/usr/bin/env python3
"""Round 3, Q1a: count the instructions of the hot loop of a kernel by class, from llvm-objdump output of one kernel.
Usage: llm3_isaclass.py <disassembly.s> [--weights-per-iter N] [--loop K] [--list]
A loop is a backward branch (the 16-bit offset is signed). Without --loop the loop holding the most vector-memory loads of 128 bits or the
longest one is printed; --list prints every loop. Classes: global/buffer loads, scalar loads, LDS (ds_), convert (v_cvt*), fma (v_fma*, v_fmac*,
v_pk_fma*), mad, dot (v_dot*), integer bit ops (and/or/xor/shifts/perm/bfe/alignbit/lshl_or/and_or/bitop), integer add/mul, float other (mul/add/max/min/exp...),
VALU other, SALU, waits, branch. Exit 1 if no loop is found."""
import collections
import re
import sys


def parse(path):
    ins = []
    for ln in open(path):
        m = re.match(r"^\s+([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):", ln)
        if m:
            ins.append((int(m.group(3), 16), m.group(1), m.group(2)))
    return ins


def loops(ins):
    out = []
    index = {a: i for i, (a, _, _) in enumerate(ins)}
    for i, (a, op, args) in enumerate(ins):
        if op.startswith("s_cbranch") or op == "s_branch":
            m = re.search(r"(\d+)$", args)
            if not m:
                continue
            off = int(m.group(1))
            if off >= 0x8000:
                off -= 0x10000
            if off < 0:
                tgt = a + 4 + 4 * off
                j = index.get(tgt)
                if j is None:
                    j = next((k for k, (aa, _, _) in enumerate(ins) if aa >= tgt), None)
                out.append((j, i))
    return out


BITOPS = re.compile(r"^v_(and|or|xor|not|lshl|lshr|ashr|lshlrev|lshrrev|ashrrev|perm|bfe|bfi|alignbit|alignbyte|bitop3|lshl_or|and_or|or3|xor3|pack|cndmask|bfrev|cvt_pk_u8|permlane|mov|movrel|dual_mov|sad|byte)")


def classify(op):
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "load (global/buffer)"
    if op.startswith(("global_store", "buffer_store", "flat_store", "global_atomic")):
        return "store/atomic"
    if op.startswith("s_load") or op.startswith("s_buffer_load"):
        return "scalar load"
    if op.startswith("ds_"):
        return "LDS"
    if op.startswith("v_cvt") or op.startswith("v_pk_cvt"):
        return "convert (v_cvt)"
    if "dot" in op and op.startswith("v_"):
        return "dot (v_dot)"
    if op.startswith(("v_fma", "v_fmac", "v_pk_fma", "v_mad", "v_pk_mad", "v_wmma")):
        return "fma/mad"
    if op.startswith("v_dual"):
        return "dual-issue VALU"
    if op.startswith("v_") and BITOPS.match(op):
        return "integer bit/move"
    if op.startswith("v_") and re.match(r"^v_(add|sub|mul|mad|min|max|cmp|sad|mbcnt|readlane|writelane|readfirstlane|lshl_add|add_lshl|cndmask)", op):
        return "VALU add/mul/compare/other"
    if op.startswith("v_"):
        return "VALU other"
    if op.startswith("s_wait") or op in ("s_nop", "s_delay_alu", "s_clause"):
        return "wait/delay/clause"
    if op.startswith("s_cbranch") or op == "s_branch":
        return "branch"
    if op.startswith("s_"):
        return "SALU"
    return "other"


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("-"):
        print(__doc__)
        return 2
    ins = parse(a[0])
    wpi = float(a[a.index("--weights-per-iter") + 1]) if "--weights-per-iter" in a else None
    lp = loops(ins)
    if not lp:
        print("FAIL no loop (backward branch) found")
        return 1
    print(f"{a[0]}: {len(ins)} instructions, {len(lp)} loops (instruction index ranges: {[(i, j) for i, j in lp]})")
    pick = int(a[a.index("--loop") + 1]) if "--loop" in a else None
    if "--list" in a or pick is None:
        for n, (i, j) in enumerate(lp):
            body = ins[i : j + 1]
            c = collections.Counter(classify(o) for _, o, _ in body)
            loads = sum(1 for _, o, _ in body if o.startswith(("global_load", "buffer_load")))
            print(f"  loop {n}: instructions {i}..{j} ({len(body)}), vector loads {loads}, " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
        if pick is None:
            return 0
    i, j = lp[pick]
    body = ins[i : j + 1]
    c = collections.Counter(classify(o) for _, o, _ in body)
    print(f"loop {pick}: {len(body)} instructions per iteration")
    valu = sum(v for k, v in c.items() if k in ("convert (v_cvt)", "dot (v_dot)", "fma/mad", "dual-issue VALU", "integer bit/move", "VALU add/mul/compare/other", "VALU other"))
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v:4d}" + (f"  {v / wpi:7.3f} per weight" if wpi else ""))
    print(f"  {'VALU in all (not counting waits)':30s} {valu:4d}" + (f"  {valu / wpi:7.3f} per weight" if wpi else ""))
    ops = collections.Counter(o for _, o, _ in body)
    print("  opcodes: " + ", ".join(f"{k} x{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
