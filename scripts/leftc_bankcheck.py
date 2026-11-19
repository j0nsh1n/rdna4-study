#!/usr/bin/env python3
"""Listing check for src/study/left_w64fma.hip (leftovers C, lane A, fix round, T1).

Usage: python3 leftc_bankcheck.py <.lto.s listing>
For every kernel k_fma<MODE, U>, finds the timed loop (the loop whose body holds the v_fma_f32 block) and checks:
the body holds U x 16 v_fma_f32; the other instructions of the body (loop control, waits, and in the pinned modes
2-4 the compiler's v_mov copies into the pinned registers) are the same for U = 4 and U = 8, so the U = 8 minus U = 4
difference that the probe reports cancels them (lane 1's loop rule); and the number of FMAs whose VGPR sources share a
register bank (register number mod 4) is the mode's: MODE 0 and 5 half, MODE 2 none, MODE 3 and 4 all. Prints one
line per kernel; exit 1 on any mismatch.
"""
import collections
import re
import sys

WANT = {0: 0.5, 5: 0.5, 2: 0.0, 3: 1.0, 4: 1.0}


def main():
    lines = open(sys.argv[1]).read().splitlines()
    kern = None
    bodies = {}
    label = None
    body = []
    for ln in lines:
        m = re.match(r"^(_Z\S*k_fmaILi(\d+)ELi(\d+)E\S*):", ln)
        if m:
            kern = (int(m.group(2)), int(m.group(3)))
            label = None
            continue
        if kern is None:
            continue
        m = re.match(r"^(\.LBB\S+):", ln)
        if m:
            label, body = m.group(1), []
            continue
        t = ln.split(";")[0].strip()
        if not t or t.startswith("."):
            continue
        if label:
            body.append(t)
            if t.startswith("s_cbranch") and t.endswith(label):
                if any(x.startswith("v_fma_f32") for x in body):
                    bodies[kern] = body
                label = None
    bad = 0
    for (mode, u), body in sorted(bodies.items()):
        fma = [x for x in body if x.startswith("v_fma_f32")]
        other = [x.split()[0] for x in body if not x.startswith("v_fma_f32")]
        other4 = [x.split()[0] for x in bodies.get((mode, 4), []) if not x.startswith("v_fma_f32")]
        same = collections.Counter(other) == collections.Counter(other4)
        shared = 0
        for x in fma:
            regs = [int(r) for r in re.findall(r"\bv(\d+)\b", x)[1:]]
            banks = [r % 4 for r in regs]
            shared += len(set(banks)) < len(banks)
        frac = shared / len(fma) if fma else -1
        ok = len(fma) == 16 * u and same and abs(frac - WANT.get(mode, -9)) < 1e-9
        bad += not ok
        print(f"{'ok' if ok else 'MISMATCH'} k_fma<{mode},{u}>: {len(fma)} FMAs in the loop, {shared} with two VGPR "
              f"sources in one bank (expect {WANT.get(mode)} of them), other instructions "
              f"{dict(collections.Counter(other))} (the same as at U = 4: {'yes' if same else 'NO'})")
    print(f"bankcheck: {len(bodies)} timed loops, {bad} mismatched")
    sys.exit(1 if bad or len(bodies) != 2 * len(WANT) else 0)


if __name__ == "__main__":
    main()
