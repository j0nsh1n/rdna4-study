r"""Check 1 table: run the study's hazard and mask scans over every listing in a directory.

Usage: python3 c1_scan.py <dir with *.s> [--selftest]
Columns: compiler, target, opt, then for the haz listing: callee return (first_mm_tied return==1 means the callee ends
in a matrix instruction with no VALU before s_setpc_b64), caller call (h_call3 / h_call4 call==1 means a matrix
instruction follows the call with no VALU), the straight-line control (hazard), the inline-asm break (hazard); for the
mask listing: matrix instructions that may run under a partial EXEC in k_ifload / k_early / k_loop.
--selftest: edit a copy of a listing (remove the V_NOP from h_asm_nop, which must raise hazard 0 -> 1; add a V_NOP to
first_mm_tied before its return, which must drop return 1 -> 0) and report PASS/FAIL.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import haz_hazscan as hs  # noqa: E402
import haz_maskscan as ms  # noqa: E402


def hazard_counts(path):
    lines = open(path).read().split("\n")
    funcs = list(ms.functions(lines))
    names = ms.demangle([f for f, _ in funcs])
    out = {}
    for f, body in funcs:
        found = hs.scan(body)
        out[names.get(f, f)] = {k: sum(1 for x in found if x[0] == k.upper()) for k in ("hazard", "return", "call")}
    return out


def mask_counts(path):
    lines = open(path).read().split("\n")
    funcs = list(ms.functions(lines))
    names = ms.demangle([f for f, _ in funcs])
    out = {}
    for f, body in funcs:
        sites = ms.scan(body)
        out[names.get(f, f)] = sum(1 for x in sites if ms.MATRIX.match(x[1]))
    return out


def pick(counts, rx, kind):
    hit = [k for k in counts if re.search(rx, k)]
    if not hit:
        return "n/a"
    return str(sum(counts[k][kind] for k in hit))


def main():
    d = sys.argv[1]
    if "--selftest" in sys.argv:
        return selftest(d)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.s"))):
        m = re.match(r"(.+?)-(gfx\d+)-(O\d)-(\w+)\.s$", os.path.basename(p))
        rows.append((m.groups(), p))
    haz = {}
    msk = {}
    for (comp, tgt, opt, src), p in rows:
        if src in ("haz_hazard", "haz_gfx11"):
            haz[(comp, tgt, opt)] = hazard_counts(p)
        if src in ("haz_mask", "haz_gfx11"):
            msk[(comp, tgt, opt)] = mask_counts(p)
    print("| compiler | target | opt | callee returns on its matrix instr (return) | caller: matrix instr right after call, h_call3 / h_call4 (call) | straight-line control (hazard) | asm break (hazard) | masked WMMA k_ifload / k_early / k_loop |")
    print("|---|---|---|---|---|---|---|---|")
    keys = sorted(set(haz) | set(msk))
    for k in keys:
        h = haz.get(k)
        m = msk.get(k)
        if h:
            ret = pick(h, r"^first_mm_tied", "return")
            c3 = pick(h, r"^h_call3", "call")
            c4 = pick(h, r"^h_call4", "call")
            st = pick(h, r"^h_straight", "hazard")
            asm = pick(h, r"^h_asm_raw", "hazard")
        else:
            ret = c3 = c4 = st = asm = "n/a"
        if m:
            def mm(rx):
                hit = [n for n in m if re.search(rx, n)]
                return str(sum(m[n] for n in hit)) if hit else "n/a"
            mk = " / ".join(mm(r) for r in (r"^k_ifload", r"^k_early", r"^k_loop"))
        else:
            mk = "n/a"
        print(f"| {k[0]} | {k[1]} | {k[2]} | {ret} | {c3} / {c4} | {st} | {asm} | {mk} |")


def selftest(d):
    import shutil
    import tempfile
    src = os.path.join(d, "r101-gfx1201-O3-haz_hazard.s")
    text = open(src).read().split("\n")
    work = os.environ.get("EXTCHECK_SCRATCH", tempfile.mkdtemp())
    os.makedirs(work, exist_ok=True)

    def func_range(name_rx):
        start = None
        for i, ln in enumerate(text):
            m = re.match(r"^([A-Za-z_][\w.$]*):\s*;.*@", ln)
            if m and re.search(name_rx, ms.demangle([m.group(1)]).get(m.group(1), m.group(1))):
                start = i
            if start is not None and re.match(r"^\s*\.Lfunc_end", ln):
                return start, i
        raise SystemExit(f"function {name_rx} not found")

    ok = True
    # 1. remove the v_nop from h_asm_nop: hazard 0 -> 1
    a, b = func_range(r"^h_asm_nop")
    broken = text[:a] + [ln for ln in text[a:b] if ln.strip() != "v_nop"] + text[b:]
    p = os.path.join(work, "break-nonop.s")
    open(p, "w").write("\n".join(broken))
    h0 = {k: v for k, v in hazard_counts(src).items() if k.startswith("h_asm_nop")}
    h1 = {k: v for k, v in hazard_counts(p).items() if k.startswith("h_asm_nop")}
    n0 = sum(v["hazard"] for v in h0.values())
    n1 = sum(v["hazard"] for v in h1.values())
    res = n0 == 0 and n1 == 1
    print(f"SELFTEST break (v_nop removed from h_asm_nop): hazard {n0} -> {n1}: {'PASS' if res else 'FAIL'}")
    ok &= res
    # 2. add a v_nop before s_setpc_b64 in first_mm_tied: return 1 -> 0
    a, b = func_range(r"^first_mm_tied")
    fixed = list(text)
    for i in range(a, b):
        if fixed[i].strip().startswith("s_setpc_b64"):
            fixed.insert(i, "\tv_nop")
            break
    p = os.path.join(work, "fixed-callee.s")
    open(p, "w").write("\n".join(fixed))
    r0 = sum(v["return"] for k, v in hazard_counts(src).items() if k.startswith("first_mm_tied"))
    r1 = sum(v["return"] for k, v in hazard_counts(p).items() if k.startswith("first_mm_tied"))
    res = r0 == 1 and r1 == 0
    print(f"SELFTEST fix (v_nop added before the return in first_mm_tied): return {r0} -> {r1}: {'PASS' if res else 'FAIL'}")
    ok &= res
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
