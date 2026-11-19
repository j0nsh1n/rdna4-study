r"""Find every matrix instruction (and lane permute) that may run with a partial EXEC, in an AMDGPU listing.

Usage: python3 haz_maskscan.py <file.s> [<expect-file>]

On gfx1201 a matrix instruction ignores EXEC (measured, src/study/haz_mask.hip): it reads every lane's operands and
writes every lane's result. Under a partial EXEC the compiler believes the switched-off lanes are untouched, so their
operand registers may hold stale values and their result registers get overwritten. This scan finds the sites.

Method: per function, a forward data-flow over the basic blocks of the listing. State: whether EXEC may be partial,
and which scalar registers hold a copy of a full EXEC. Entry: full (a wave of a launch whose block size is a multiple
of 32). EXEC becomes partial at s_and_saveexec / s_and_not1_saveexec / s_xor_saveexec, at an s_and / s_and_not1 /
s_andn2 / s_xor / s_mov / s_cselect writing exec_lo from anything that is not a full copy, and at v_cmpx. It is full
again at s_or exec with a full copy, s_mov exec from a full copy or -1, and s_or_saveexec with -1. Paths merge as
"partial if any path is partial"; the set of full copies merges by intersection. Any other write to a scalar register
removes it from the set.

Output: one "MASKED <function> <line> <instruction>" line per site, then "COUNT <function> <wmma> <permlane>" per
function. With an expect file (lines "<function regex> wmma|permlane == <n>", '#' comments), each line must match at
least one function and the summed count must agree; exit status 1 otherwise.
"""
import re
import subprocess
import sys

MATRIX = re.compile(r"^v_(wmma|swmmac)_")
PERMUTE = re.compile(r"^v_permlane")
SREG = re.compile(r"^s(\d+)$|^s\[(\d+):(\d+)\]$")
NO_DEST = re.compile(r"^s_(cmp|bitcmp|cbranch|branch|wait|nop|endpgm|setprio|sleep|barrier|delay_alu|sendmsg|"
                     r"setpc|trap|icache|clause|code_end|dcache|prefetch|set_inst_prefetch|ttrace|singleuse|"
                     r"version|waitcnt)")


def regs(op):
    m = SREG.match(op)
    if not m:
        return set()
    if m.group(1):
        return {int(m.group(1))}
    return set(range(int(m.group(2)), int(m.group(3)) + 1))


def functions(lines):
    cur, body, start = None, [], 0
    for i, ln in enumerate(lines):
        m = re.match(r"^([A-Za-z_][\w.$]*):\s*(;.*@.*)?$", ln)
        if m and not ln.startswith(".") and "@" in (m.group(2) or ""):
            cur, body = m.group(1), []
            continue
        if cur and re.match(r"^\s*\.Lfunc_end", ln):
            yield cur, body
            cur = None
            continue
        if cur:
            body.append((i + 1, ln))


def demangle(names):
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
        return dict(zip(names, out))
    except OSError:
        return {n: n for n in names}


def blocks(body):
    """[(label, [(lineno, opcode, operands)])]; label None for the entry block."""
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
    if opc.startswith("s_") and not NO_DEST.match(opc):
        if dest in ("exec_lo", "exec"):
            srcs = ops[1:]
            if opc.startswith("s_or_b") and any(regs(s) and regs(s) <= full for s in srcs):
                return False, full
            if opc.startswith("s_mov_b") and srcs and (srcs[0] == "-1" or (regs(srcs[0]) and regs(srcs[0]) <= full)):
                return False, full
            return True, full
        if "saveexec" in opc:
            was_partial = partial
            full -= regs(dest)
            if not was_partial:
                full |= regs(dest)
            if opc.startswith("s_or_saveexec") and len(ops) > 1 and ops[1] == "-1":
                return False, full
            return True, full
        if opc.startswith("s_mov_b") and len(ops) > 1 and ops[1] in ("exec_lo", "exec") and not partial:
            full |= regs(dest)
            return partial, full
        full -= regs(dest)
    return partial, full


def scan(body):
    bl = blocks(body)
    index = {lab: i for i, (lab, _) in enumerate(bl) if lab}
    succ = []
    for i, (_, insts) in enumerate(bl):
        s = []
        last = insts[-1] if insts else None
        if last and (last[1] == "s_branch" or last[1].startswith("s_cbranch")):
            if last[2] and last[2][0] in index:
                s.append(index[last[2][0]])
        if not last or not (last[1] in ("s_branch", "s_endpgm", "s_setpc_b64")):
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
            if st[0] and (MATRIX.match(opc) or PERMUTE.match(opc)):
                sites.append((no, opc, ", ".join(ops)))
            st = step(st, opc, ops)
    return sites


def main():
    lines = open(sys.argv[1]).read().split("\n")
    funcs = list(functions(lines))
    names = demangle([f for f, _ in funcs])
    counts = {}
    for f, body in funcs:
        sites = scan(body)
        nw = sum(1 for s in sites if MATRIX.match(s[1]))
        npl = sum(1 for s in sites if PERMUTE.match(s[1]))
        nm = names.get(f, f)
        counts[nm] = (nw, npl)
        for no, opc, ops in sites[:4]:
            print(f"MASKED {nm} line {no}: {opc} {ops}")
        if len(sites) > 4:
            print(f"MASKED {nm} ... {len(sites) - 4} more")
        print(f"COUNT {nm} wmma={nw} permlane={npl}")
    if len(sys.argv) < 3:
        return 0
    bad = 0
    for ln in open(sys.argv[2]):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        rx, what, op, n = ln.split()
        hit = [k for k in counts if re.search(rx, k)]
        if not hit:
            print(f"EXPECT FAIL: no function matches {rx}")
            bad += 1
            continue
        got = sum(counts[k][0 if what == "wmma" else 1] for k in hit)
        ok = got == int(n) if op == "==" else got >= int(n) if op == ">=" else got <= int(n)
        print(f"EXPECT {'PASS' if ok else 'FAIL'}: {rx} {what} {op} {n} (got {got})")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
