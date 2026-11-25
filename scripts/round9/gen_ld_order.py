#!/usr/bin/env python3
"""Writes src/round9/ld_order_blocks.h: the patterns of ld_order.hip (round 9, item B), from the table below.

Usage: python3 -I gen_ld_order.py <output header>        (standard library only)

One pattern is a list of tokens. The header holds, per pattern, the inline-assembly text (the single source of truth:
the listing check compares the code object against this text), the kind of every destination register, and the claims
that the analysis judges. Tokens:
  M     global_load_b32 of a line far away (GDDR6: fresh, never read since the cache flush); the m-th M of a pattern is
        16 KiB further on than the one before
  H     global_load_b32 of the near line (an L0 hit: the prelude loaded it just before the timed region)
  L     ds_load_b32 (LDS)
  S     s_load_b32 of a near line (scalar-cache hit: the prelude loaded it just before)
  F     s_load_b32 of a far line (fresh)
  WL n  s_wait_loadcnt n      WD n  s_wait_dscnt n      WK n  s_wait_kmcnt n
  T     read the cycle counter, then copy every destination register (v_mov, after the read, so the copy is not timed)
        into a snapshot register. A pattern has one or two T.
Destination registers: the g-th global load writes v(40+g) (at most 5), LDS v45, the scalar load s48. Every one is set to
a sentinel before the timed region, so a register read before its data arrived shows the sentinel.

Claims (what the instruction set says must hold at a snapshot, under in-order retirement of each counter):
  must   registers that must hold their data at snapshot i
  stale  registers that are not expected to hold data yet (the wait did not cover them and they are far away): an
         informational check that the probe can see "not yet" at all
A pattern named brk_* is a deliberate break: its `must` claim is false on purpose (the wrong counter was waited), and the
analysis requires the probe to SEE the early read (wrong words in at least 90 % of the lane checks).
"""
import sys

SENTINEL = "0xdeadbeef"
SLOTS = ["g0", "g1", "g2", "g3", "g4", "lds", "sca"]  # snapshot order: v64..v70 and v72..v78

# name, tokens, comment
PATTERNS = [
    ("empty", "T", "baseline: two counter reads back to back"),
    ("pure_hit", "H WL0 T", "one L0 hit, waited"),
    ("pure_miss", "M WL0 T", "one GDDR6 miss, waited"),
    ("mh_w1", "M H WL1 T", "miss then hit, wait 1: in order costs the miss; out of order costs the hit"),
    ("hm_w1", "H M WL1 T", "hit then miss, wait 1: costs the hit"),
    ("mh_w0", "M H WL0 T", "both, wait 0"),
    ("hm_w0", "H M WL0 T", "both, wait 0"),
    ("mm_w1", "M M WL1 T", "two misses, wait 1"),
    ("mm_w0", "M M WL0 T", "two misses, wait 0"),
]
for seq, tag in (("M H H H H", "mhhhh"), ("H H H H M", "hhhhm"), ("H M H M H", "hmhmh")):
    for n in (1, 2, 3, 4):
        PATTERNS.append((f"{tag}_w{n}", f"{seq} WL{n} T", f"five loads {seq}, wait {n}"))
PATTERNS += [
    ("gl_dswait", "M L WD0 T WL0 T", "far global + LDS: dscnt first (LDS back, global not), then loadcnt"),
    ("brk_wrongcnt", "M L WD0 T", "BREAK: waits the LDS counter, claims the global register is valid"),
    ("lds_only", "L WD0 T", "LDS load alone"),
    ("s_hit", "S WK0 T", "scalar load, scalar-cache hit"),
    ("s_far", "F WK0 T", "scalar load, far"),
    ("m_s_kmwait", "M S WK0 T WL0 T", "far global + scalar hit: kmcnt first (scalar back, global not), then loadcnt"),
    ("s_m_lwait", "S M WL0 T", "scalar hit then far global, wait loadcnt only (scalar must have arrived by then)"),
    ("brk_wrongkm", "M S WK0 T", "BREAK: waits the scalar counter, claims the global register is valid"),
]


def build(tokens):
    """Returns (asm lines, kinds per slot, snapshot claims)."""
    lines = []
    kinds = ["-"] * 7
    g = m = 0
    snaps = []
    cnt_slots = {"loadcnt": [], "dscnt": [], "kmcnt": []}  # per counter: [(slot, kind)] in issue order
    covered = set()
    for tok in tokens.split():
        if tok == "M":
            lines.append(f"global_load_b32 v{40 + g}, v60, %[fbase] offset:{16384 * m}")
            kinds[g] = "M"
            cnt_slots["loadcnt"].append((g, "M"))
            g += 1
            m += 1
        elif tok == "H":
            lines.append(f"global_load_b32 v{40 + g}, v62, %[nbase]")
            kinds[g] = "H"
            cnt_slots["loadcnt"].append((g, "H"))
            g += 1
        elif tok == "L":
            lines.append("ds_load_b32 v45, v63")
            kinds[5] = "L"
            cnt_slots["dscnt"].append((5, "L"))
        elif tok in ("S", "F"):
            lines.append(f"s_load_b32 s48, %[{'sn' if tok == 'S' else 'sf'}], 0x0")
            kinds[6] = tok
            cnt_slots["kmcnt"].append((6, tok))
        elif tok[:2] in ("WL", "WD", "WK"):
            cnt = {"WL": "loadcnt", "WD": "dscnt", "WK": "kmcnt"}[tok[:2]]
            n = int(tok[2:])
            lines.append(f"s_wait_{cnt} {n}")
            issued = cnt_slots[cnt]
            # loads of one counter retire in order: after `wait n`, all but the newest n of them are done
            covered |= {slot for slot, _ in issued[: max(0, len(issued) - n)]}
        elif tok == "T":
            t = len(snaps) + 1
            lines.append(f"s_getreg_b32 %[t{t}], hwreg(HW_REG_SHADER_CYCLES_LO)")
            base = 64 if t == 1 else 72
            for i in range(7):
                lines.append(f"v_mov_b32 v{base + i}, " + (f"v{40 + i}" if i < 6 else "s48"))
            allloads = [x for v in cnt_slots.values() for x in v]
            far_covered = any(k in ("M", "F") and sl in covered for sl, k in allloads)
            stale = set() if far_covered else {sl for sl, k in allloads if k in ("M", "F") and sl not in covered}
            snaps.append((set(covered), stale))
    return lines, kinds, snaps


def mask(s):
    return sum(1 << i for i in s)


def main():
    out = sys.argv[1]
    rows = []
    for name, tokens, comment in PATTERNS:
        lines, kinds, snaps = build(tokens)
        brk = name.startswith("brk_")
        claims = []
        for must, stale in snaps:
            if brk:
                # the false claim: the global register g0 is valid although only another counter was waited
                must = must | {0}
                stale = stale - {0}
            claims.append((mask(must), mask(stale)))
        while len(claims) < 2:
            claims.append((0, 0))
        nt = len(snaps)
        rows.append((name, tokens, comment, lines, "".join(kinds), nt, claims))
    with open(out, "w") as f:
        f.write("// Generated by scripts/round9/gen_ld_order.py: do not edit. Patterns of ld_order.hip (round 9, item B).\n")
        f.write("// X(name, nsnap, kinds, must1, stale1, must2, stale2, text)\n")
        f.write("// kinds: one character per slot g0 g1 g2 g3 g4 lds sca: M far global, H near global, L LDS, S near scalar,\n")
        f.write("// F far scalar, - unused. must/stale: bit masks over the same slots (see the generator's docstring).\n")
        f.write("#define LD_ORDER_PATTERNS(X) \\\n")
        for name, tokens, comment, lines, kinds, nt, cl in rows:
            f.write(f"    /* {tokens}: {comment} */ \\\n")
            text = "".join(f'"{ln}\\n" ' for ln in lines)
            f.write(f"    X({name}, {nt}, \"{kinds}\", {cl[0][0]}, {cl[0][1]}, {cl[1][0]}, {cl[1][1]}, {text}) \\\n")
        f.write("\n")


if __name__ == "__main__":
    main()
