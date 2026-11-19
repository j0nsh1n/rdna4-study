r"""Find matrix-instruction pairs where the second reads the first's result as A or B with no VALU between.

Usage: python3 haz_hazscan.py <file.s> [<expect-file>]

The rule (AMD's guide, LLVM fixWMMAHazards): a V_NOP or another VALU instruction must separate a matrix instruction
from a following one whose A or B overlaps its result. C may overlap (the hardware waits). The scan walks forward
from every matrix instruction through its block and on into successor blocks (any path), and stops at the first
VALU instruction. It reports:
  HAZARD   a matrix instruction reached that way whose A or B overlaps the first one's result
  RETURN   a function return (s_setpc_b64) reached that way: the caller's next instruction is unknown here
  CALL     a matrix instruction reached from a call site (s_swappc_b64) before any VALU: the callee's last
           instruction is unknown here
Output: "<KIND> <function> line <n>: ..." lines, then "COUNT <function> hazard=<h> return=<r> call=<c>".
Expect file lines: "<function regex> hazard|return|call == <n>"; exit status 1 if any fails.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz_maskscan import blocks, demangle, functions  # noqa: E402

MATRIX = re.compile(r"^v_(wmma|swmmac)_")
VREG = re.compile(r"^v(\d+)$|^v\[(\d+):(\d+)\]$")


def vregs(op):
    m = VREG.match(op)
    if not m:
        return set()
    if m.group(1):
        return {int(m.group(1))}
    return set(range(int(m.group(2)), int(m.group(3)) + 1))


def is_valu(opc):
    return opc.startswith("v_") and not MATRIX.match(opc)


def scan(body):
    bl = blocks(body)
    index = {lab: i for i, (lab, _) in enumerate(bl) if lab}

    def succ(i):
        insts = bl[i][1]
        last = insts[-1] if insts else None
        s = []
        if last and (last[1] == "s_branch" or last[1].startswith("s_cbranch")) and last[2] and last[2][0] in index:
            s.append(index[last[2][0]])
        if not last or last[1] not in ("s_branch", "s_endpgm", "s_setpc_b64"):
            if i + 1 < len(bl):
                s.append(i + 1)
        return s

    found = []

    def walk(i, pos, dest, origin, seen):
        insts = bl[i][1]
        for no, opc, ops in insts[pos:]:
            if is_valu(opc):
                return
            if MATRIX.match(opc):
                ab = vregs(ops[1]) | vregs(ops[2]) if len(ops) > 2 else set()
                if dest is not None and ab & dest:
                    found.append(("HAZARD", no, f"{opc} {', '.join(ops)} (after line {origin})"))
                if dest is None:
                    found.append(("CALL", no, f"{opc} {', '.join(ops)} (after call at line {origin})"))
                return
            if opc == "s_setpc_b64" and dest is not None:
                found.append(("RETURN", no, f"return with the result of line {origin} pending"))
                return
            if opc == "s_endpgm":
                return
        for j in succ(i):
            if (j, origin) not in seen:
                seen.add((j, origin))
                walk(j, 0, dest, origin, seen)

    for i, (_, insts) in enumerate(bl):
        for p, (no, opc, ops) in enumerate(insts):
            if MATRIX.match(opc) and ops:
                walk(i, p + 1, vregs(ops[0]), no, set())
            if opc == "s_swappc_b64":
                walk(i, p + 1, None, no, set())
    return sorted(set(found), key=lambda f: f[1])


def main():
    lines = open(sys.argv[1]).read().split("\n")
    funcs = list(functions(lines))
    names = demangle([f for f, _ in funcs])
    counts = {}
    for f, body in funcs:
        found = scan(body)
        nm = names.get(f, f)
        c = {k: sum(1 for x in found if x[0] == k.upper()) for k in ("hazard", "return", "call")}
        counts[nm] = c
        for kind, no, text in found[:6]:
            print(f"{kind} {nm} line {no}: {text}")
        print(f"COUNT {nm} hazard={c['hazard']} return={c['return']} call={c['call']}")
    if len(sys.argv) < 3:
        return 0
    bad = 0
    for ln in open(sys.argv[2]):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        rx, what, op, n = ln.split()
        hit = [k for k in counts if re.search(rx, k)]
        if not hit:
            print(f"EXPECT FAIL: no function matches {rx}")
            bad += 1
            continue
        got = sum(counts[k][what] for k in hit)
        ok = got == int(n) if op == "==" else got >= int(n) if op == ">=" else got <= int(n)
        print(f"EXPECT {'PASS' if ok else 'FAIL'}: {rx} {what} {op} {n} (got {got})")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
