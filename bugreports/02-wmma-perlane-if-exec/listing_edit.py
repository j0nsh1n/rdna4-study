#!/usr/bin/env python3
"""Edit a copy of an AMDGPU listing for the deliberate breaks of verify_cpu.sh.

Usage: python3 -I listing_edit.py hoist <in.s> <out.s> <function>
         Move the first v_wmma/v_swmmac of <function> to just before the first EXEC-narrowing instruction
         (s_*_saveexec_* or v_cmpx_*) that precedes it. The matrix instruction is then outside the mask.
       python3 -I listing_edit.py inject <in.s> <out.s> <function>
         Copy the first v_wmma/v_swmmac of <function> to just after the first EXEC-narrowing instruction
         (or, if the function has none, insert `s_mov_b32 exec_lo, 1` and the copy before the first store).
Exit status 2 on a usage error or when the pattern is not found.
"""
import re
import sys

sys.dont_write_bytecode = True

MATRIX = re.compile(r"^\s*v_(wmma|swmmac)_")
NARROW = re.compile(r"^\s*(s_[a-z0-9_]*saveexec_b\d+|v_cmpx_\S+)\s")


def span(lines, fn):
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^(_Z\d+{re.escape(fn)}\w*|{re.escape(fn)}):", ln)), None)
    if start is None:
        return None
    end = next((i for i in range(start, len(lines)) if lines[i].lstrip().startswith(".Lfunc_end")), len(lines))
    return start, end


def main(argv):
    if len(argv) != 4 or argv[0] not in ("hoist", "inject"):
        print(__doc__)
        return 2
    mode, src, dst, fn = argv
    lines = open(src).read().split("\n")
    sp = span(lines, fn)
    if sp is None:
        print(f"function {fn} not found")
        return 2
    lo, hi = sp
    m = next((i for i in range(lo, hi) if MATRIX.match(lines[i])), None)
    if m is None:
        print("no matrix instruction in function")
        return 2
    if mode == "hoist":
        n = next((i for i in range(m, lo, -1) if NARROW.match(lines[i])), None)
        if n is None:
            print("no EXEC-narrowing instruction before the matrix instruction")
            return 2
        line = lines.pop(m)
        lines.insert(n, line)
    else:
        n = next((i for i in range(lo, hi) if NARROW.match(lines[i])), None)
        if n is None:
            st = next((i for i in range(m, hi) if re.match(r"^\s*(global|flat)_store", lines[i])), None)
            if st is None:
                print("no insertion point")
                return 2
            lines[st:st] = ["\ts_mov_b32 exec_lo, 1", lines[m]]
        else:
            lines.insert(n + 1, lines[m])
    open(dst, "w").write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
