#!/usr/bin/env python3
"""Listing check for the one-statement timed regions of left_core.hip (leftovers C, lane A).

Usage: python3 leftc_asmcheck.py <disassembly from llvm-objdump -d> <blocks header> [--extra OP ...]
Every kernel k_<block>_<R> holds timed regions between two `s_getreg_b32 ... HW_REG_SHADER_CYCLES_LO`. Each region
must hold exactly R copies of the block's instructions as written in the header (same opcodes, same counts) plus the
loop control (s_mov_b32, s_add_co_i32, s_cmp_lg_u32, s_cbranch_scc1) and one of each --extra opcode (left_lds: the
s_wait_dscnt before the second counter read), nothing else. The encoded size of every
instruction is read from the disassembly; the bytes per instruction of each block are printed (the run12 rows of
EC-6 depend on them: 12 bytes with a literal, 8 without). Exit 1 on any mismatch.
"""
import collections
import re
import sys

CONTROL = collections.Counter({"s_mov_b32": 1, "s_add_co_i32": 1, "s_cmp_lg_u32": 1, "s_cbranch_scc1": 1})


def norm(op):
    """The disassembler adds the encoding suffix the source may leave out."""
    return re.sub(r"_e(32|64)$", "", op)


def blocks(header):
    out = {}
    for m in re.finditer(r'X\((\w+), (\d+), "((?:[^"\\]|\\.)*)"\)', open(header).read()):
        text = m.group(3).replace("\\n", "\n")
        ops = collections.Counter(norm(ln.split()[0]) for ln in text.splitlines() if ln.strip())
        out[m.group(1)] = (int(m.group(2)), ops)
    return out


def main():
    dis, header = sys.argv[1], sys.argv[2]
    extra = sys.argv[sys.argv.index("--extra") + 1:] if "--extra" in sys.argv else []
    control = CONTROL + collections.Counter(extra)
    B = blocks(header)
    cur = None
    regions = collections.defaultdict(list)  # kernel -> list of [instructions (op, bytes)]
    inside = None
    for ln in open(dis):
        m = re.match(r"^[0-9a-f]+ <(\S+)>:", ln)
        if m:
            cur = m.group(1)
            inside = None
            continue
        if cur is None or not ln.startswith("\t"):
            continue
        body, _, enc = ln.partition("//")
        t = body.strip()
        if not t:
            continue
        op = norm(t.split()[0])
        words = enc.split(":", 1)[1].split() if ":" in enc else []
        if op == "s_getreg_b32" and "HW_REG_SHADER_CYCLES_LO" in t:
            if inside is None:
                inside = []
            else:
                regions[cur].append(inside)
                inside = None
            continue
        if inside is not None:
            inside.append((op, 4 * len(words)))
    bad = checked = 0
    for kern, regs in sorted(regions.items()):
        m = re.search(r"\d+k_(\w+)_(\d+)E", kern)
        if not m or m.group(1) not in B:
            continue
        name, R = m.group(1), int(m.group(2))
        n, ops = B[name]
        want = collections.Counter({k: v * R for k, v in ops.items()}) + control
        for reg in regs:
            checked += 1
            got = collections.Counter(op for op, _ in reg)
            if got != want:
                bad += 1
                print(f"MISMATCH {name} R={R}: got {dict(got - want)} missing {dict(want - got)}")
        if R == 2:
            sizes = collections.Counter(b for op, b in regs[0] if op in ops)
            print(f"ok {name}: {n} instructions per block, bytes {dict(sorted(sizes.items()))}")
    names = {re.search(r"\d+k_(\w+)_(\d+)E", k).group(1) for k in regions if re.search(r"\d+k_(\w+)_(\d+)E", k)}
    missing = [b for b in B if b not in names]
    print(f"asmcheck: {checked} timed regions checked, {bad} mismatched; blocks in the header but not in this object: "
          f"{len(missing)}")
    sys.exit(1 if bad or checked == 0 else 0)


if __name__ == "__main__":
    main()
