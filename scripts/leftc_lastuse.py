#!/usr/bin/env python3
"""MS-7 (leftovers C, lane A): compile src/study/left_lastuse.ll for gfx1201 with clang -x ir and check the hints.

Usage: python3 leftc_lastuse.py <repo> [--break]
The three global loads are told apart by their offsets (0, 4 x 4096 = 16384, 4 x 8192 = 32768 bytes from the lane's
address). Verdict lastuse_listing: the plain load has no th field, the !nontemporal one th:TH_LOAD_NT (the control,
memory-system item 8) and the !amdgpu.last.use one th:TH_LOAD_LU. --break expects TH_LOAD_LU on the plain load: FAIL.
"""
import os
import re
import subprocess
import sys

repo = sys.argv[1]
brk = "--break" in sys.argv
clang = os.path.expanduser("~/.local/opt/rocm/llvm/bin/clang")
src = os.path.join(repo, "src/study/left_lastuse.ll")
r = subprocess.run([clang, "-x", "ir", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-O3", "-S", "-o", "-", src],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr)
    print("VERDICT lastuse_listing: FAIL clang did not compile the IR")
    sys.exit(1)
loads = {}
for ln in r.stdout.splitlines():
    t = ln.strip()
    if t.startswith("global_load"):
        m = re.search(r"offset:(\d+)", t)
        off = int(m.group(1)) if m else 0
        th = re.search(r"th:(\S+)", t)
        loads[off] = th.group(1) if th else "none"
        print("RESULT lastuse", t)
want = {0: "TH_LOAD_LU" if brk else "none", 16384: "TH_LOAD_LU", 32768: "TH_LOAD_NT"}
ok = all(loads.get(k) == v for k, v in want.items())
print("VERDICT lastuse_listing: %s plain load %s, !amdgpu.last.use %s, !nontemporal %s (expect %s, TH_LOAD_LU, "
      "TH_LOAD_NT%s)" % ("PASS" if ok else "FAIL", loads.get(0), loads.get(16384), loads.get(32768), want[0],
                         "; deliberate break" if brk else ""))
sys.exit(0 if ok else 1)
