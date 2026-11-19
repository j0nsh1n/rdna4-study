#!/usr/bin/env python3
"""Listing check for left_hint.hip (leftovers C, lane A, MS-6): every k_bl<AUX> kernel's buffer_load_b128 must carry
th = AUX & 7 and scope = AUX >> 3 (IntrinsicsAMDGPU.td, gfx12: bits [0-2] = th, bits [3-4] = scope; names from
SIDefines.h CPol, research-10 section 4: TH_BYPASS = 3 is printed only with scope 3).

Usage: python3 leftc_hintcheck.py <disassembly>. Exit 1 on any mismatch or a missing aux value.
"""
import re
import sys

TH = {"": 0, "TH_LOAD_NT": 1, "TH_LOAD_HT": 2, "TH_LOAD_LU": 3, "TH_LOAD_BYPASS": 3, "TH_LOAD_NT_RT": 4,
      "TH_LOAD_RT_NT": 5, "TH_LOAD_NT_HT": 6, "0x7": 7}
SCOPE = {"": 0, "SCOPE_SE": 1, "SCOPE_DEV": 2, "SCOPE_SYS": 3}
seen, bad, cur = {}, 0, None
for ln in open(sys.argv[1]):
    m = re.match(r"^[0-9a-f]+ <\S*k_blILi(\d+)E", ln)
    if m:
        cur = int(m.group(1))
        continue
    if cur is not None and "buffer_load_b128" in ln:
        t = ln.split("//")[0]
        th = re.search(r"th:(\S+)", t)
        sc = re.search(r"scope:(\S+)", t)
        got = (TH.get(th.group(1) if th else "", -1), SCOPE.get(sc.group(1) if sc else "", -1))
        want = (cur & 7, cur >> 3)
        seen[cur] = got
        if got != want:
            bad += 1
            print(f"MISMATCH aux {cur}: listing th {got[0]} scope {got[1]}, expected {want}")
        cur = None
missing = [a for a in range(32) if a not in seen]
print(f"regioncheck: {len(seen)} builtin aux values, {bad} with th/scope different from aux & 7 / aux >> 3, missing {missing}")
sys.exit(1 if bad or missing else 0)
