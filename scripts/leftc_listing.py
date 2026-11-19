#!/usr/bin/env python3
"""Leftovers C, lane B: listing checks on the disassembly of a built code object (llvm-objdump -d output).

Usage:
  leftc_listing.py pairs DISASM 'FIRST::SECOND' ...
      every pair of instructions (text up to the comment, spaces normalised) must appear back to back somewhere, and
      also with exactly one v_nop between (the control kernel); nothing else may sit between them.
  leftc_listing.py runs DISASM MNEMONIC:LEN ...
      every mnemonic must appear as a run of exactly LEN consecutive instructions somewhere (a timed chain that the
      compiler left whole).
  leftc_listing.py count DISASM MNEMONIC:N ...
      the mnemonic must appear exactly N times.
Prints one VERDICT line; exit 1 when a check fails.
"""
import re
import sys


def instructions(path):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            s = line.split("//")[0].strip()
            if not s or s.endswith(":") or s.startswith(("Disassembly", ".", "/")) or "file format" in s:
                continue
            out.append(re.sub(r"\s+", " ", s.replace(", ", ",")))
    return out


def norm(t):
    return re.sub(r"\s+", " ", t.strip().replace(", ", ","))


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    cmd, path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    ins = instructions(path)
    bad = []
    if cmd == "pairs":
        for a in args:
            first, second = (norm(x) for x in a.split("::"))
            adj = sum(1 for i in range(len(ins) - 1) if ins[i] == first and ins[i + 1] == second)
            nop = sum(1 for i in range(len(ins) - 2)
                      if ins[i] == first and ins[i + 1].startswith("v_nop") and ins[i + 2] == second)
            print(f"LISTING pair {first} :: {second}: back to back {adj}, with one v_nop {nop}")
            if adj < 1 or nop < 1:
                bad.append(a)
    elif cmd == "runs":
        mn = [x.split(" ")[0] for x in ins]
        for a in args:
            m, n = a.rsplit(":", 1)
            n = int(n)
            found = 0
            i = 0
            while i < len(mn):
                j = i
                while j < len(mn) and mn[j] == m:
                    j += 1
                if j - i == n:
                    found += 1
                i = max(j, i + 1)
            print(f"LISTING run {m} x{n}: {found}")
            if found < 1:
                bad.append(a)
    elif cmd == "count":
        mn = [x.split(" ")[0] for x in ins]
        for a in args:
            m, n = a.rsplit(":", 1)
            c = mn.count(m)
            print(f"LISTING count {m}: {c} (expected {n})")
            if c != int(n):
                bad.append(a)
    else:
        print(__doc__)
        return 2
    ok = not bad and bool(ins)
    print(f"VERDICT listing_{cmd}: {'PASS' if ok else 'FAIL'} {len(args)} checks on {path.rsplit('/', 1)[-1]}"
          + (f"; failed: {', '.join(bad)}" if bad else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
