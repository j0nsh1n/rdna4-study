#!/usr/bin/env python3
"""Edit a copy of an AMDGPU listing for the deliberate breaks of verify_cpu.sh.

Usage: python3 -I listing_edit.py hoist <in.s> <out.s> <function>
         Move the straight-line instructions that sit between the first EXEC-narrowing instruction and the first
         permute of <function> (the sunk definition of the permute source) to just before the narrowing
         instruction. Labels and branches stay where they are. The text is edited only; the edited listing is
         not meant to be assembled.
       python3 -I listing_edit.py inject <in.s> <out.s> <function>
         Insert `v_mov_b32 <src>, 0` for the permute's source register right after the first EXEC-narrowing
         instruction, i.e. a definition of the source inside the narrowed region.
Exit status 2 on a usage error or when the pattern is not found.
"""
import re
import sys

sys.dont_write_bytecode = True

NARROW = re.compile(r"^\s*(s_[a-z0-9_]*saveexec_b\d+|v_cmpx_\S+)\s")
PERM = re.compile(r"^\s*v_permlanex?16_b32\s+(\S+),\s*([^,\s]+)")
MOVABLE = re.compile(r"^\s*(global_load|flat_load|v_|s_mov_b32|s_wait_loadcnt|s_wait_kmcnt|s_delay_alu|s_wait_alu)")


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
    p = next((i for i in range(lo, hi) if PERM.match(lines[i])), None)
    n = next((i for i in range(lo, hi) if NARROW.match(lines[i])), None)
    if p is None or n is None:
        print("no permute or no EXEC-narrowing instruction in function")
        return 2
    if mode == "hoist":
        if n > p:
            print("narrowing instruction is after the permute")
            return 2
        moved = [i for i in range(n + 1, p) if MOVABLE.match(lines[i])]
        if not any(lines[i].lstrip().startswith(("global_load", "flat_load")) or lines[i].lstrip().startswith("v_") for i in moved):
            print("no sunk definition between the narrowing instruction and the permute")
            return 2
        block = [lines[i] for i in moved]
        keep = [ln for i, ln in enumerate(lines) if i not in set(moved)]
        # n is unchanged by removing lines after it
        keep[n:n] = block
        lines = keep
    else:
        reg = PERM.match(lines[p]).group(2)
        lines.insert(n + 1, f"\tv_mov_b32_e32 {reg}, 0")
    open(dst, "w").write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
