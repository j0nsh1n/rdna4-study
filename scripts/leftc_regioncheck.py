#!/usr/bin/env python3
"""Listing check for compiler-built timed regions (leftovers C, lane A).

Usage: python3 leftc_regioncheck.py <disassembly> <kernel regex> <timed-op regex> <allowed regex> [--min N]
In every kernel whose name matches, a timed region runs from one read of HW_REG_SHADER_CYCLES_LO to the next. A region
that holds at least one instruction matching <timed-op regex> (for example the loads being timed) may hold only
instructions matching <timed-op regex> or <allowed regex> (waits, the HI/LO counter reads, address moves); anything
else (a spill, a store, a branch) is printed and the check fails. Prints one line per distinct region shape and the
number of timed regions; fails when fewer than --min (default 1) timed regions were found.
"""
import collections
import re
import sys


def main():
    dis, kre, tre, are = sys.argv[1:5]
    mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1
    kre, tre, are = re.compile(kre), re.compile(tre), re.compile(are)
    cur, region, shapes, bad, timed = None, None, collections.Counter(), 0, 0
    for ln in open(dis):
        m = re.match(r"^[0-9a-f]+ <(\S+)>:", ln)
        if m:
            cur = m.group(1) if kre.search(m.group(1)) else None
            region = None
            continue
        if cur is None or not ln.startswith("\t"):
            continue
        t = ln.partition("//")[0].strip()
        if not t:
            continue
        op = t.split()[0]
        if op == "s_getreg_b32" and "HW_REG_SHADER_CYCLES_LO" in t:
            if region is not None and any(tre.search(o) for o in region):
                timed += 1
                extra = [o for o in region if not tre.search(o) and not are.search(o)]
                shape = " ".join(f"{k}x{v}" for k, v in sorted(collections.Counter(region).items()))
                shapes[(cur, shape)] += 1
                if extra:
                    bad += 1
                    print(f"EXTRA in {cur}: {' '.join(extra)}")
            region = []
            continue
        if region is not None:
            region.append(op)
    for (k, s), n in sorted(shapes.items()):
        print(f"region x{n} {k}: {s}")
    print(f"regioncheck: {timed} timed regions, {bad} with instructions outside the allowed set")
    sys.exit(1 if bad or timed < mn else 0)


if __name__ == "__main__":
    main()
