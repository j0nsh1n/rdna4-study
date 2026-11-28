#!/usr/bin/env python3
"""Report, for every v_permlane16/v_permlanex16 in an AMDGPU listing, whether its source VGPR is last written
inside or before the region where EXEC is narrowed.

Usage: python3 -I permsrc_scan.py <file.s>

Why it matters: with fetch-inactive set, the permute reads the source register of lanes that are switched off.
If the source is written only inside the narrowed region, the switched-off lanes never write it, so an active
lane reads a stale value from its partner.

Method (listing order, which is enough for the straight-line regions this bug produces):
  - the region starts at s_*_saveexec_* / v_cmpx_* / `s_and_b32|s_and_b64 exec...` and ends at
    `s_or_b32|s_or_b64 exec_lo|exec, ...` (restore);
  - every instruction that writes a VGPR (v_*, global/flat/buffer/scratch/ds loads) records whether the write
    happened inside the region (a self-copy `v_mov_b32 vN, vN` is not a write: ROCm 10.2 -O1 emits one inside the
    region for the pinned source, and switched-off lanes keep their value);
  - at a permute (v_permlane16_b32 vdst, vsrc, ...), the source is operand 2 (vsrc).
Output: `PERMUTE <fn> line N: <mnemonic> src=<vreg> src_def=inside_mask|before_mask|none` and a final
`SUMMARY <fn> inside=<n> before=<n>` per function.
"""
import re
import sys

sys.dont_write_bytecode = True

VREG = re.compile(r"^v(\d+)$|^v\[(\d+):(\d+)\]$")
LOAD = re.compile(r"^(global|flat|buffer|scratch|ds)_load")
PERM = re.compile(r"^v_permlane(x?)16_b32")
NARROW = re.compile(r"^(s_[a-z0-9_]*saveexec_b\d+|v_cmpx_\S+)$")


def vregs(op):
    m = VREG.match(op.strip())
    if not m:
        return set()
    if m.group(1):
        return {int(m.group(1))}
    return set(range(int(m.group(2)), int(m.group(3)) + 1))


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
            ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
            body.append((i, parts[0], ops))


def short(name):
    m = re.match(r"^_Z(\d+)", name)
    if m:
        k = 2 + len(m.group(1))
        return name[k:k + int(m.group(1))]
    return name


def main(path):
    text = open(path).read()
    for name, body in functions(text):
        masked = False
        last_def = {}  # vgpr number -> True if the last write was inside the region
        inside = before = 0
        sn = short(name)
        for no, mn, ops in body:
            if PERM.match(mn):
                src = vregs(ops[1]) if len(ops) > 1 else set()
                states = {last_def.get(r) for r in src}
                if True in states:
                    state = "inside_mask"
                    inside += 1
                elif False in states:
                    state = "before_mask"
                    before += 1
                else:
                    state = "none"
                print(f"PERMUTE {sn} line {no}: {mn} src={ops[1] if len(ops) > 1 else '?'} src_def={state}")
            if NARROW.match(mn) or (re.match(r"^s_and_b(32|64)$", mn) and ops and ops[0] in ("exec", "exec_lo")):
                masked = True
            elif re.match(r"^s_or_b(32|64)$", mn) and ops and ops[0] in ("exec", "exec_lo"):
                masked = False
            if (mn.startswith("v_") and not mn.startswith(("v_cmp", "v_readfirstlane", "v_readlane"))) or LOAD.match(mn):
                self_copy = re.match(r"^v_mov_b32(_e32)?$", mn) and len(ops) > 1 and ops[0] == ops[1]
                if ops and not self_copy:
                    for r in vregs(ops[0]):
                        last_def[r] = masked
        print(f"SUMMARY {sn} inside={inside} before={before}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
