#!/usr/bin/env python3
"""List the WMMA/SWMMAC instructions of an AMDGPU listing that may execute under a partial EXEC mask.

Usage: python3 -I exec_mask_scan.py <file.s>

Method: a forward data-flow over the basic blocks of each function (self-contained, no other files needed).
State per program point: `partial` (EXEC may differ from the full wave mask) and `fulls` (scalar registers known to
hold a copy of the full wave mask).
  - the function entry is full (the wave comes from a launch whose block size is a multiple of the wave size);
  - EXEC becomes partial at s_*_saveexec_*, at v_cmpx_*, and at a write to exec/exec_lo/exec_hi from anything that
    is not a full copy;
  - EXEC is full again at `s_or_b32 exec_lo, exec_lo, sN` with sN in `fulls`, at `s_mov_* exec, -1` or from a full
    copy, and at `s_or_saveexec_* sN, -1`;
  - a path merge is partial if any incoming path is partial; `fulls` merges by intersection.
Output, per function: `MASKED <fn> line N: <instr>` (first 4 sites) and `COUNT <fn> wmma=N`.
The scan follows one function at a time: a WMMA in a non-inlined callee that is CALLED under a partial mask is not
reported.
"""
import re
import sys

sys.dont_write_bytecode = True

MATRIX = re.compile(r"^v_(wmma|swmmac)_")
SREG = re.compile(r"^s(\d+)$|^s\[(\d+):(\d+)\]$")
NO_DEST = re.compile(r"^s_(cmp|bitcmp|cbranch|branch|wait|nop|endpgm|setprio|sleep|barrier|delay_alu|sendmsg|setpc|"
                     r"trap|icache|clause|code_end|dcache|prefetch|set_inst_prefetch|ttrace|singleuse|version|waitcnt)")
EXEC = ("exec", "exec_lo", "exec_hi")


def regs(op):
    m = SREG.match(op)
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
            body.append((i, ln))


def blocks(body):
    """[(label|None, [(lineno, mnemonic, [operands])])]"""
    out, cur, insts = [], None, []
    for no, ln in body:
        t = ln.split(";")[0].strip()
        if not t:
            continue
        if t.endswith(":") and not t.startswith("s_"):
            out.append((cur, insts))
            cur, insts = t[:-1], []
            continue
        if t.startswith("."):
            continue
        parts = t.split(None, 1)
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        insts.append((no, parts[0], ops))
        if parts[0] in ("s_branch", "s_endpgm", "s_setpc_b64") or parts[0].startswith("s_cbranch"):
            out.append((cur, insts))
            cur, insts = f"__after{no}", []
    out.append((cur, insts))
    return [(lab, ins) for lab, ins in out if ins or (lab and not lab.startswith("__after"))]


def step(state, opc, ops):
    partial, full = state
    full = set(full)
    dest = ops[0] if ops else ""
    if opc.startswith("v_cmpx"):
        return True, full
    if not opc.startswith("s_") or NO_DEST.match(opc):
        return partial, full
    if "saveexec" in opc:
        full -= regs(dest)
        if not partial:
            full |= regs(dest)
        if opc.startswith("s_or_saveexec") and len(ops) > 1 and ops[1] == "-1":
            return False, full
        return True, full
    if dest in EXEC:
        srcs = ops[1:]
        if opc.startswith("s_or_b") and any(regs(s) and regs(s) <= full for s in srcs):
            return False, full
        if opc.startswith("s_mov_b") and srcs and (srcs[0] == "-1" or (regs(srcs[0]) and regs(srcs[0]) <= full)):
            return False, full
        return True, full
    if opc.startswith("s_mov_b") and len(ops) > 1 and ops[1] in EXEC and not partial:
        return partial, full | regs(dest)
    return partial, full - regs(dest)


def scan(body):
    bl = blocks(body)
    index = {lab: i for i, (lab, _) in enumerate(bl) if lab}
    succ = []
    for i, (_, insts) in enumerate(bl):
        s = []
        last = insts[-1] if insts else None
        if last and (last[1] == "s_branch" or last[1].startswith("s_cbranch")) and last[2] and last[2][0] in index:
            s.append(index[last[2][0]])
        if not last or last[1] not in ("s_branch", "s_endpgm", "s_setpc_b64"):
            if i + 1 < len(bl):
                s.append(i + 1)
        succ.append(s)
    ins = [None] * len(bl)
    ins[0] = (False, frozenset())
    work = [0]
    while work:
        i = work.pop()
        st = (ins[i][0], set(ins[i][1]))
        for _, opc, ops in bl[i][1]:
            st = step(st, opc, ops)
        out = (st[0], frozenset(st[1]))
        for j in succ[i]:
            new = out if ins[j] is None else (ins[j][0] or out[0], ins[j][1] & out[1])
            if new != ins[j]:
                ins[j] = new
                work.append(j)
    sites = []
    for i, (_, insts) in enumerate(bl):
        if ins[i] is None:
            continue
        st = (ins[i][0], set(ins[i][1]))
        for no, opc, ops in insts:
            if st[0] and MATRIX.match(opc):
                sites.append((no, opc, ", ".join(ops)))
            st = step(st, opc, ops)
    return sites


def short(name):
    m = re.match(r"^_Z(\d+)", name)
    if m:
        k = 2 + len(m.group(1))
        return name[k:k + int(m.group(1))]
    return name


def main(path):
    text = open(path).read()
    for name, body in functions(text):
        sites = scan(body)
        sn = short(name)
        for no, opc, ops in sites[:4]:
            print(f"MASKED {sn} line {no}: {opc} {ops}")
        print(f"COUNT {sn} wmma={len(sites)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
