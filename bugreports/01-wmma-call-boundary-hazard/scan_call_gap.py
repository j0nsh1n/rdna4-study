#!/usr/bin/env python3
"""Report whether an AMDGPU (gfx11/gfx12) assembly listing has a WMMA hazard gap across a call boundary.

Usage: python3 -I scan_call_gap.py <file.s>

Rule being checked (AMD RDNA 3/4 ISA guide; LLVM GCNHazardRecognizer::fixWMMAHazards): when a WMMA/SWMMAC
instruction writes a VGPR range that the next WMMA/SWMMAC reads as A or B, at least one other VALU instruction
(V_NOP is enough) must separate them.

The scanner is a heuristic over the textual listing. It does not know which registers are read:

  CALLEE    function ends on a matrix instruction: walking back from `s_setpc_b64` over non-VALU instructions,
            the first `v_` instruction is v_wmma_*/v_swmmac_*. gap=yes means no VALU/V_NOP before the return.
  CALLSITE  after `s_swappc_b64`, walking forward over non-VALU instructions, the first `v_` instruction is a
            matrix instruction. gap=yes means no VALU/V_NOP between the return and the next matrix instruction.
  STRAIGHT  inside one function without calls: matrix instruction followed by another matrix instruction with
            no VALU between (bare) and matrix, VALU, matrix (covered).

Final line: VERDICT bug_present=yes|no   (yes = some callee ends on a matrix instruction AND some call site is
followed directly by a matrix instruction; the plain-C++ reproducer produces both).
"""
import re
import sys

sys.dont_write_bytecode = True

MATRIX = ("v_wmma_", "v_swmmac_")


def kind(mn):
    if mn.startswith(MATRIX):
        return "matrix"
    if mn.startswith("v_"):
        return "valu"
    return "other"


def functions(text):
    cur, body = None, []
    for i, ln in enumerate(text.split("\n"), 1):
        m = re.match(r"^([A-Za-z_][\w.$]*):\s*(;.*@.*)?$", ln)
        if m and "@" in (m.group(2) or ""):
            cur, body = m.group(1), []
            continue
        if cur and re.match(r"^\s*\.Lfunc_end", ln):
            yield cur, body
            cur = None
            continue
        if cur:
            t = ln.split(";")[0].strip()
            if not t or t.startswith(".") or t.endswith(":"):
                continue
            parts = t.split(None, 1)
            body.append((i, parts[0], parts[1] if len(parts) > 1 else ""))


def short(name):
    m = re.match(r"^_Z(\d+)", name)
    if m:
        n = int(m.group(1))
        return name[2 + len(m.group(1)):2 + len(m.group(1)) + n]
    return name


def main(path):
    text = open(path).read()
    callee_gap = call_gap = False
    for name, body in functions(text):
        sn = short(name)
        for idx, (ln, mn, ops) in enumerate(body):
            if mn == "s_setpc_b64":
                prev = next(((l, m) for l, m, _ in reversed(body[:idx]) if kind(m) != "other"), None)
                gap = bool(prev and kind(prev[1]) == "matrix")
                callee_gap |= gap
                print(f"CALLEE {sn} line {ln}: last_vector_instr={prev[1] if prev else 'none'} gap={'yes' if gap else 'no'}")
            elif mn == "s_swappc_b64":
                nxt = next(((l, m) for l, m, _ in body[idx + 1:] if kind(m) != "other"), None)
                gap = bool(nxt and kind(nxt[1]) == "matrix")
                call_gap |= gap
                print(f"CALLSITE {sn} line {ln}: next_vector_instr={nxt[1] if nxt else 'none'} gap={'yes' if gap else 'no'}")
        bare = covered = 0
        for idx, (ln, mn, ops) in enumerate(body):
            if kind(mn) != "matrix":
                continue
            seen_valu = False
            for _, m2, _ in body[idx + 1:]:
                if m2 in ("s_swappc_b64", "s_setpc_b64", "s_endpgm") or m2.startswith(("s_cbranch", "s_branch")):
                    break
                k = kind(m2)
                if k == "matrix":
                    bare, covered = (bare, covered + 1) if seen_valu else (bare + 1, covered)
                    break
                if k == "valu":
                    seen_valu = True
        if bare or covered:
            print(f"STRAIGHT {sn}: matrix_pairs_bare={bare} matrix_pairs_with_valu={covered}")
    print(f"VERDICT bug_present={'yes' if callee_gap and call_gap else 'no'} callee_gap={'yes' if callee_gap else 'no'} "
          f"callsite_gap={'yes' if call_gap else 'no'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
