"""Check the timed loops of the execution-core probes in an AMDGPU .s file (from hipcc -save-temps).

Usage: python3 core_loopcheck.py <file.s> [--allow REGEX]
A loop whose only inline assembly is a marker comment is a compiler-built loop: its opcodes are printed as C LOOP and
it is not checked.
For every kernel, finds each innermost backward-branch loop that contains inline assembly (;;#ASMSTART) and lists
the instructions in it that the compiler added outside the assembly blocks. Loop control (s_add/s_sub/s_cmp/
s_cbranch/s_wait_alu/s_delay_alu on SGPRs) is expected; anything else (a v_mov, a wait counter, a spill) is printed as
EXTRA and the kernel is reported as NOT CLEAN. Exit status 1 if any kernel is not clean.
Also counts the instructions inside the assembly blocks of the loop by opcode, so the loop body can be compared with
what the source meant. The probes build each loop with U = 4, 6 and 8 assembly blocks (template argument); the
instructions the compiler adds must be the same in all three, otherwise they grow with the block count and do not
cancel in the differential: such a kernel is reported as GROWS. (The compiler does interleave its loop control
between the blocks; that is fine as long as it appears once per iteration.)
"""
import collections
import re
import sys

CONTROL = re.compile(r"^(s_add_co_i32|s_add_co_u32|s_sub_co_i32|s_cmp_\w+|s_cbranch_\w+|s_branch|s_wait_alu|s_delay_alu|"
                     r"s_add_nc_u64|s_mov_b32|s_cselect_b32)$")


def kernels(lines):
    cur, body = None, []
    for ln in lines:
        m = re.match(r"^(_Z\S+):\s*(;.*)?$", ln)
        if m:
            if cur:
                yield cur, body
            cur, body = m.group(1), []
            continue
        if cur and re.match(r"^\s*\.Lfunc_end", ln):
            yield cur, body
            cur, body = None, []
            continue
        if cur:
            body.append(ln)


def loops(body):
    labels = {}
    for i, ln in enumerate(body):
        m = re.match(r"^(\.LBB\S+):", ln)
        if m:
            labels[m.group(1)] = i
    out = []
    for i, ln in enumerate(body):
        m = re.match(r"^\s*(s_cbranch_\w+|s_branch)\s+(\.LBB\S+)", ln)
        if m and m.group(2) in labels and labels[m.group(2)] < i:
            out.append((labels[m.group(2)], i))
    # innermost: no other loop strictly inside
    inner = [l for l in out if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in out)]
    return inner


def scan(body, start, end):
    in_asm = False
    added, asm_ops = [], collections.Counter()
    has_asm = False
    for ln in body[start:end + 1]:
        t = ln.strip()
        if t.startswith(";;#ASMSTART"):
            in_asm, has_asm = True, True
            continue
        if t.startswith(";;#ASMEND"):
            in_asm = False
            continue
        if not t or t.startswith((";", ".")) or t.endswith(":"):
            continue
        op = t.split()[0]
        if in_asm:
            asm_ops[op] += 1
        else:
            added.append(t)
    return has_asm, added, asm_ops


def main():
    path = sys.argv[1]
    allow = None
    if "--allow" in sys.argv:
        allow = re.compile(sys.argv[sys.argv.index("--allow") + 1])
    lines = open(path).read().splitlines()
    bad = 0
    total = 0
    by_base = collections.defaultdict(dict)  # kernel name without the U argument -> {U: sorted added opcodes}
    for name, body in kernels(lines):
        for s, e in loops(body):
            has_asm, added, ops = scan(body, s, e)
            if not has_asm:
                continue
            if not ops:
                # only a marker comment: a compiler-built loop, reported for reading, not checked
                hist = collections.Counter(x.split()[0] for x in added)
                print(f"C LOOP {name}: " + " ".join(f"{k}x{v}" for k, v in sorted(hist.items())))
                continue
            total += 1
            extra = [x for x in added if not CONTROL.match(x.split()[0])]
            if allow:
                extra = [x for x in extra if not allow.match(x.split()[0])]
            m = re.match(r"^(.*)ILi(\d+)E(.*)$", name)
            if m:
                by_base[m.group(1) + m.group(3)][int(m.group(2))] = sorted(x.split()[0] for x in added)
            hist = " ".join(f"{k}x{v}" for k, v in sorted(ops.items()))
            if extra:
                bad += 1
                print(f"NOT CLEAN {name}: asm [{hist}] EXTRA: " + " | ".join(extra))
            else:
                print(f"clean {name}: asm [{hist}] loop control: " + " ".join(x.split()[0] for x in added))
    grows = 0
    for base, per_u in sorted(by_base.items()):
        if len(per_u) > 1 and len({tuple(v) for v in per_u.values()}) > 1:
            grows += 1
            print(f"GROWS {base}: " + "; ".join(f"U={u}: {' '.join(v)}" for u, v in sorted(per_u.items())))
    print(f"loopcheck: {total} timed loops, {bad} not clean, {grows} kernels whose added instructions change with U")
    sys.exit(1 if bad or grows or total == 0 else 0)


if __name__ == "__main__":
    main()
