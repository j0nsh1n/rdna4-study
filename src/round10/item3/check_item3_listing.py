#!/usr/bin/env python3
"""Round 10 item 3: checks that the HIP bisect kernels compiled to the loads they are meant to be, from the disassembly of the
binary that will run (llvm-objdump of the code object taken out of bin/r10_cl3 or bin/r10_cl3_w64). CPU only.

Usage: check_item3_listing.py <dis.txt> <notes.txt> <32|64>     the listing check (exit 1 if any claim fails)
       check_item3_listing.py selftest <dis.txt> <notes.txt> <32|64>   deliberate breaks: every one must be caught

Claims, per k_chase_f<FORM, SCOPE> (FORM 0 p64, 1 m64, 2 b32, 3 u32v; SCOPE 0 rt, 1 dev, 2 sys, 3 nt, 4 se):
  - exactly 16 global loads in the unrolled body, all of the form's width (b64 for 0 and 1, b32 for 2 and 3);
  - every load carries the scope's modifier and no other (SCOPE 0: none at all);
  - b64 loads use `off` (a per-lane 64-bit address), b32 loads the SGPR-base form `v, v, s[..]`;
  - every load is followed by `s_wait_loadcnt 0x0` before the next load (each step is waited, as the chain needs);
  - FORM 0 and 2: no vector ALU instruction between loads; FORM 1: at least 2 v_and_b32 per step (the mask on the 64-bit
    address); FORM 3: at least 1 v_and_b32 and 1 v_lshl_or_b32 (or v_lshl_add_u32) per step (mask and index arithmetic);
  - the SCOPE 0, FORM 0 kernel has the same load text as mem_latency.hip's own k_chase (what r9_cl2 runs);
  - the code object's .wavefront_size is the build's (32 or 64) for every k_chase_f kernel.
"""
import re
import sys

MODS = ["", "scope:SCOPE_DEV", "scope:SCOPE_SYS", "th:TH_LOAD_NT", "scope:SCOPE_SE"]
FORM_N = ["p64", "m64", "b32", "u32v"]
SC_N = ["rt", "dev", "sys", "nt", "se"]


def kernels(dis):
    """name -> list of instruction texts (comments removed)."""
    out, cur = {}, None
    for line in dis.splitlines():
        m = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur and line.startswith("\t"):
            out[cur].append(re.sub(r"\s*//.*$", "", line).strip())
    return out


def check_kernel(form, sc, ins):
    errs = []
    loads = [i for i, t in enumerate(ins) if t.startswith("global_load_")]
    width = "b64" if form in (0, 1) else "b32"
    if len(loads) != 16:
        errs.append(f"{len(loads)} loads, expected 16")
    for i in loads:
        t = ins[i]
        if not t.startswith(f"global_load_{width} "):
            errs.append(f"load is not {width}: {t}")
        rest = t.split(None, 1)[1]
        mod = rest.split("off", 1)[1].strip() if width == "b64" and " off" in " " + rest else (
            re.sub(r"^.*?s\[\d+:\d+\]", "", rest).strip() if width == "b32" else "?")
        if mod != MODS[sc]:
            errs.append(f"modifier '{mod}' on {t}, expected '{MODS[sc]}'")
        if width == "b64" and ", off" not in t:
            errs.append(f"b64 load without `off`: {t}")
        if width == "b32" and not re.search(r"s\[\d+:\d+\]", t):
            errs.append(f"b32 load without an SGPR base: {t}")
    for a, b in zip(loads, loads[1:]):
        between = ins[a + 1 : b]
        if not any(t.startswith("s_wait_loadcnt") for t in between):
            errs.append(f"no s_wait_loadcnt between loads at {a} and {b}")
        valu = [t for t in between if t.startswith("v_")]
        if form in (0, 2) and valu:
            errs.append(f"vector ALU between loads of a pure chain: {valu[0]}")
    n_and = sum(t.startswith("v_and_b32") for t in ins)
    n_lsh = sum(t.startswith(("v_lshl_or_b32", "v_lshl_add_u32")) for t in ins)
    if form == 1 and n_and < 32:
        errs.append(f"{n_and} v_and_b32, expected >= 32 (mask on both halves of 16 steps)")
    if form == 3 and (n_and < 16 or n_lsh < 16):
        errs.append(f"{n_and} v_and_b32 / {n_lsh} v_lshl_or_b32 or v_lshl_add_u32, expected >= 16 each")
    return errs


def check(dis, notes, wave):
    ks = kernels(dis)
    errs, seen = [], 0
    for name, ins in ks.items():
        m = re.match(r"_Z9k_chase_fILi(\d)ELi(\d)E", name)
        if not m:
            continue
        form, sc = int(m.group(1)), int(m.group(2))
        seen += 1
        for e in check_kernel(form, sc, ins):
            errs.append(f"k_chase_f<{FORM_N[form]},{SC_N[sc]}>: {e}")
    if seen != 20:
        errs.append(f"{seen} k_chase_f kernels in the listing, expected 20")
    base = [t for t in ks.get("_Z9k_chase_fILi0ELi0EEvmPKcmjjjPm", []) if t.startswith("global_load_")]
    r9 = [t for t in ks.get("_Z7k_chasemjjPm", []) if t.startswith("global_load_")]
    if not base or not r9 or set(base) != set(r9):
        errs.append(f"k_chase_f<p64,rt> load text {sorted(set(base))} differs from mem_latency.hip k_chase {sorted(set(r9))}")
    # the notes list every kernel with its .name and .wavefront_size; take the ones of the chase kernels
    blocks = re.split(r"\n(?=\s*- )", notes)
    ws = [int(re.search(r"\.wavefront_size:\s*(\d+)", b).group(1)) for b in blocks
          if "k_chase_fILi" in b and re.search(r"\.wavefront_size:\s*(\d+)", b)]
    if len(ws) != 20 or any(w != wave for w in ws):
        errs.append(f"wavefront_size of the chase kernels {sorted(set(ws))} (n={len(ws)}), expected 20 x {wave}")
    return errs


def selftest(dis, notes, wave):
    base = check(dis, notes, wave)
    if base:
        print("SELFTEST FAIL: the unmodified listing does not pass:", *base[:3], sep="\n  ")
        return 1
    ks = kernels(dis)
    victim = {f: next(n for n in ks if n.startswith(f"_Z9k_chase_fILi{f}ELi0E")) for f in range(4)}
    breaks = []

    def in_kernel(name, fn):
        def go(d):
            head, _, tail = d.partition(f"<{name}>:")
            body, sep, rest = tail.partition("\n\n")
            return head + f"<{name}>:" + fn(body) + sep + rest
        return go

    sc1 = next(n for n in ks if n.startswith("_Z9k_chase_fILi0ELi1E"))
    breaks.append(("scope modifier removed from the dev kernel", lambda d: in_kernel(sc1, lambda b: b.replace(" scope:SCOPE_DEV", ""))(d), None))
    breaks.append(("a b32 load in the b64 chain", lambda d: in_kernel(victim[0], lambda b: b.replace("global_load_b64", "global_load_b32", 1))(d), None))
    breaks.append(("a wait removed from the b32 chain", lambda d: in_kernel(victim[2], lambda b: b.replace("s_wait_loadcnt 0x0", "s_nop 0", 1))(d), None))
    breaks.append(("an ALU op slipped into the pure p64 chain", lambda d: in_kernel(victim[0], lambda b: b.replace("s_wait_loadcnt 0x0", "s_wait_loadcnt 0x0\n\tv_add_nc_u32_e32 v9, 1, v9", 1))(d), None))
    breaks.append(("the mask removed from the m64 chain", lambda d: in_kernel(victim[1], lambda b: b.replace("v_and_b32_e32", "v_or_b32_e32"))(d), None))
    breaks.append(("a different load text in k_chase", lambda d: in_kernel("_Z7k_chasemjjPm", lambda b: b.replace("global_load_b64", "global_load_b64 th:TH_LOAD_NT", 1) if False else b.replace(" off", " off scope:SCOPE_SE"))(d), None))
    breaks.append(("wrong wavefront size in the notes", None, lambda n: n.replace(f".wavefront_size: {wave}", f".wavefront_size: {96 - wave}")))
    bad = 0
    for label, dfn, nfn in breaks:
        d = dfn(dis) if dfn else dis
        n = nfn(notes) if nfn else notes
        if d == dis and n == notes:
            print(f"SELFTEST FAIL: break '{label}' changed nothing")
            bad += 1
            continue
        e = check(d, n, wave)
        print(f"  break '{label}': {'caught' if e else 'NOT CAUGHT'}" + (f" ({e[0][:90]})" if e else ""))
        bad += not e
    print("SELFTEST", "PASS" if not bad else "FAIL", f"({len(breaks)} breaks, {bad} not caught)")
    return 1 if bad else 0


def main(argv):
    if len(argv) == 5 and argv[1] == "selftest":
        return selftest(open(argv[2]).read(), open(argv[3]).read(), int(argv[4]))
    if len(argv) == 4:
        e = check(open(argv[1]).read(), open(argv[2]).read(), int(argv[3]))
        print("LISTING", "FAIL" if e else "PASS", *e[:10], sep="\n  " if e else " ")
        if not e:
            print("  20 k_chase_f kernels: widths, modifiers, waits, ALU between loads and wavefront size as designed")
        return 1 if e else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
