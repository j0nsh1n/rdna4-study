#!/usr/bin/env python3
"""Analysis of ld_mat_k (round 9, item A, load-ratio variant). Standard library only; run with python3 -I.

Usage: ana_ld_mat_k.py <log> [<log>...]     the logs of the jobs together (any order)
       ana_ld_mat_k.py --selftest          synthetic input in three worlds and with deliberate breaks, all must come out as planned
Prints "CONTROL <name>: PASS|FAIL ...", a table (INFO), "VERDICT <level> ...", and "SUMMARY ...". Exit 1 if a control fails.

Question: one load group per G matrix instructions (G = 1, 4, 8), a wait in front of the first instruction of the group
(variant s) or in front of every instruction (variant w). With e1 = cost(k1s) - matrix only (the 26-cycle charge of one
landing load per instruction, measured in the same run), the extra over matrix only at G is
    per landing load:                 e1 / G   for s and for w       (fraction f = 1 / G)
    per instruction with a wait:      e1 / G   for s, e1 for w       (f = 1 / G for s, f = 1 for w)
    whenever loads are in flight:     e1       for s and for w       (f = 1 for both)
The verdict names the world the numbers sit in; "no world" when they sit in none (the fractions are printed either way).

Controls (expect-ld_mat_k.md has the numbers, written before the run):
  a_ref        matrix only, one wave: 7.9-8.1 cycles per matrix instruction
  b_k1         k1s_d54 reproduces the ld_mat D = 60 cell: L0 33.6 +- 3, L2 35.7 +- 3          NOT COUNTED (see below)
  c_latency    k8s_d0 and k4s_d0: the cost of one group is the cost seen in ld_mat at D = 0: L0 104.8 +- 20 %, L2 308.2 +- 20 % per group   NOT COUNTED
  d_break      loads aimed at L0 under the L2 label (k8s_d0) move the cost by at least 70 % of the L2 - L0 difference
  e_cover      the two lookaheads of one (G, variant) agree within 2.0 cycles at one level (the latency is covered)
  placed       every burst had its one wave placed

Not counted (round 2 decision of the owner's review): b_k1 (2 controls) and c_latency (4) compare with the first ld_mat run, which
is superseded (ring and address faults, round9.md Item A). They are still printed, with the value they were compared with, as
"CONTROL <name>: NOT COUNTED"; they never fail and are left out of the control count. Against the rerun of ld_mat they would not
all agree (k1s_d54 33.07 against 39.31 at L0). The verdict does not use them: it rests on ratios inside this probe.
"""
import statistics
import sys

REF = (7.9, 8.1)
K1_WANT = {"L0": 33.6, "L2": 35.7}
K1_TOL = 3.0
GROUP_WANT = {"L0": 104.8, "L2": 308.2}
GROUP_TOL = 0.20
COVER_TOL = 2.0
FOLLOW = {1: ("k1s_d24", "k1s_d54"), 4: ("k4s_d12", "k4s_d24"), 8: ("k8s_d6", "k8s_d12")}
FOLLOW_W = {4: ("k4w_d12", "k4w_d24"), 8: ("k8w_d6", "k8w_d12")}


def parse(text):
    bursts = {}
    for ln in text.splitlines():
        if not ln.startswith("RESULT ld_mat_k burst "):
            continue
        t = ln.split()[3:]
        d = dict(zip(t[0::2], t[1::2]))
        bursts.setdefault((d["level"], d["aimed"], d["cell"]), []).append(d)
    return bursts


def cell_value(bs):
    ok = [float(b["cyc_per_mm_med"]) for b in bs if b["placed"] == "1"]
    return statistics.median(ok) if ok else float("nan")


def near(cost, pred, ref, e1):
    """within 1.5 cycles + 10 % of the extra the prediction carries"""
    return abs(cost - pred) <= 1.5 + 0.1 * (pred - ref)


def judge(bursts):
    res, verdicts = [], {}

    def ctl(name, ok, text):
        res.append((name, bool(ok), text))

    def ctl_nc(name, ok, text):
        """A control against the superseded first ld_mat run: printed with its would-be result, never counted."""
        res.append((name, None, "(would %s) %s" % ("pass" if ok else "fail", text)))

    val = {k: cell_value(v) for k, v in bursts.items()}
    refs = [v for k, v in val.items() if k[2] == "ref"]
    ref = statistics.median(refs) if refs else float("nan")
    ctl("a_ref", bool(refs) and REF[0] <= ref <= REF[1], "one wave, matrix only: %.3f cycles per matrix instruction (8.0; gate %.1f-%.1f)" % (ref, *REF))

    def v(level, cell, aimed=None):
        return val.get((level, aimed or level, cell))

    for lv in ("L0", "L2"):
        x = v(lv, "k1s_d54")
        if x is None or x != x:
            ctl_nc("b_k1_" + lv, False, "no k1s_d54 cell at %s" % lv)
        else:
            ctl_nc("b_k1_" + lv, abs(x - K1_WANT[lv]) <= K1_TOL, "%s k1s_d54: %.2f (first ld_mat run D = 60 gave %.2f; gate +-%.1f)" % (lv, x, K1_WANT[lv], K1_TOL))
    for lv in ("L0", "L2"):
        for g in (4, 8):
            x = v(lv, "k%ds_d0" % g)
            if x is None or x != x:
                ctl_nc("c_latency_%s_%d" % (lv, g), False, "no k%ds_d0 cell at %s" % (g, lv))
                continue
            grp = (x - ref) * g
            want = GROUP_WANT[lv]
            ctl_nc("c_latency_%s_%d" % (lv, g), abs(grp - want) <= GROUP_TOL * want,
                   "%s k%ds_d0: %.2f per instruction = %.1f extra cycles per group (first ld_mat run D = 0 gave %.1f; gate %.0f %%)" % (lv, g, x, grp, want, 100 * GROUP_TOL))
    brk = [(k, x) for k, x in val.items() if k[0] != k[1] and k[2] == "k8s_d0"]
    if brk:
        for k, x in brk:
            lab, aim = k[0], k[1]
            tl, ta = v(lab, "k8s_d0"), v(aim, "k8s_d0")
            if tl is None or ta is None:
                ctl("d_break_" + lab, False, "break %s aimed at %s: the normal cells are missing" % (lab, aim))
                continue
            diff, moved = tl - ta, tl - x
            ctl("d_break_" + lab, diff > 0 and moved >= 0.7 * diff,
                "k8s_d0 labelled %s aimed at %s: %.2f (normal %s %.2f, %s %.2f): moved %.2f of %.2f (gate 70 %%)" % (lab, aim, x, lab, tl, aim, ta, moved, diff))
    else:
        ctl("d_break", False, "no deliberate-break cell (label differs from aim) in the logs")
    for lv in ("L0", "L2"):
        for name, (a, b) in [("k%d%s" % (g, "s"), FOLLOW[g]) for g in FOLLOW] + [("k%dw" % g, FOLLOW_W[g]) for g in FOLLOW_W]:
            xa, xb = v(lv, a), v(lv, b)
            if xa is None or xb is None or xa != xa or xb != xb:
                ctl("e_cover_%s_%s" % (lv, name), False, "%s: %s or %s missing" % (lv, a, b))
            else:
                ctl("e_cover_%s_%s" % (lv, name), abs(xa - xb) <= COVER_TOL, "%s %s %.2f against %s %.2f (gate %.1f)" % (lv, a, xa, b, xb, COVER_TOL))
    bad = [(k, [b["placed"] for b in vs]) for k, vs in bursts.items() if any(b["placed"] != "1" for b in vs)]
    n = sum(len(vs) for vs in bursts.values())
    ctl("placed", n > 0 and not bad, "%d bursts, %d cells with a burst whose wave was not placed" % (n, len(bad)))

    for lv in ("L0", "L2"):
        k1 = v(lv, "k1s_d54")
        if k1 is None or k1 != k1 or ref != ref:
            verdicts[lv] = "NO DATA"
            continue
        e1 = k1 - ref
        sw = {}
        for g in (4, 8):
            for var, tab in (("s", FOLLOW), ("w", FOLLOW_W)):
                x = v(lv, tab[g][1])
                if x is None or x != x:
                    sw[(g, var)] = None
                    continue
                sw[(g, var)] = (x, (x - ref) / e1)
        if any(t is None for t in sw.values()):
            verdicts[lv] = "NO DATA"
            continue

        def is_(g, var, f):
            x = sw[(g, var)][0]
            return near(x, ref + e1 * f, ref, e1)

        s_land = all(is_(g, "s", 1.0 / g) for g in (4, 8))
        w_land = all(is_(g, "w", 1.0 / g) for g in (4, 8))
        s_all = all(is_(g, "s", 1.0) for g in (4, 8))
        w_all = all(is_(g, "w", 1.0) for g in (4, 8))
        if s_land and w_land:
            verdicts[lv] = "PER LANDING LOAD"
        elif s_land and w_all:
            verdicts[lv] = "PER INSTRUCTION WITH A WAIT IN FRONT"
        elif s_all and w_all:
            verdicts[lv] = "WHENEVER LOADS ARE IN FLIGHT (not per landing, not per wait)"
        else:
            verdicts[lv] = "NO WORLD FITS"
        verdicts[lv] += " (e1 %.1f; fractions of e1: " % e1 + ", ".join("G%d%s %.2f" % (g, var, sw[(g, var)][1]) for g in (4, 8) for var in ("s", "w")) + "; per landing 0.25 / 0.125, per wait or in flight 1.00)"
    return res, val, ref, verdicts


def tables(val, ref):
    out = ["INFO one wave: cycles per matrix instruction (extra over matrix only %.2f in brackets)" % ref]
    names = sorted({k[2] for k in val if k[2] != "ref"}, key=lambda n: (int(n[1:n.index("s") if "s" in n else n.index("w")]), n))
    out.append("cell        " + " ".join("%16s" % lv for lv in ("L0", "L2")))
    for n in names:
        row = []
        for lv in ("L0", "L2"):
            x = val.get((lv, lv, n))
            row.append("%16s" % ("-" if x is None else "%.2f (%+.2f)" % (x, x - ref)))
        out.append("%-11s " % n + " ".join(row))
    return out


def report(text):
    bursts = parse(text)
    res, val, ref, verdicts = judge(bursts)
    bad = 0
    for n, ok, t in res:
        print("CONTROL %s: %s %s" % (n, "NOT COUNTED" if ok is None else ("PASS" if ok else "FAIL"), t))
        bad += ok is False
    for ln in tables(val, ref):
        print(ln)
    for lv, t in verdicts.items():
        print("VERDICT %s: %s" % (lv, t))
    counted = [r for r in res if r[1] is not None]
    print("SUMMARY %d controls counted, %d failed, %d not counted (compared with the superseded first ld_mat run)" % (
        len(counted), bad, len(res) - len(counted)))
    return bad, verdicts


# ---------------------------------------------------------------------------------------------------------------
# synthetic input
E1 = {"L0": 25.5, "L2": 27.7}
GRP = {"L0": 104.8, "L2": 308.2}
NAMES = ["k1s_d24", "k1s_d54", "k4s_d0", "k4s_d12", "k4s_d24", "k4w_d12", "k4w_d24", "k8s_d0", "k8s_d6", "k8s_d12", "k8w_d6", "k8w_d12"]


def synth(world="land", ref=8.08, break_moves=True, k1_shift=0.0, placed=1):
    L = []

    def line(level, aimed, cell, cyc):
        g = 0 if cell == "ref" else int(cell[1])
        var = "-" if cell == "ref" else cell[2]
        d = -1 if cell == "ref" else int(cell.split("_d")[1])
        for rep in range(3):
            L.append("RESULT ld_mat_k burst level %s aimed %s cell %s g %d v %s dval %d rep %d iters 1000 ms 20.00 cyc_per_mm_med %.3f "
                     "clock_ghz 3.000 placed %d simds 1 hot_start 40 hot_max 60" % (level, aimed, cell, g, var, d, rep, cyc, placed))

    line("-", "-", "ref", ref)
    for lv in ("L0", "L2"):
        for n in NAMES:
            g, var, d = int(n[1]), n[2], int(n.split("_d")[1])
            if d == 0:
                cost = ref + GRP[lv] / g
            elif g == 1:
                cost = ref + E1[lv] + k1_shift
            elif world == "land":
                cost = ref + E1[lv] / g
            elif world == "wait":
                cost = ref + (E1[lv] / g if var == "s" else E1[lv])
            else:
                cost = ref + E1[lv]
            line(lv, lv, n, cost)
    line("L2", "L0", "k8s_d0", ref + (GRP["L0"] if break_moves else GRP["L2"]) / 8)
    return "\n".join(L) + "\n"


def selftest():
    import contextlib
    import io

    ok = True

    def run(label, text, expect_fail, expect_verdict=None):
        nonlocal ok
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bad, verdicts = report(text)
        good = (bad > 0) == expect_fail
        if expect_verdict:
            good = good and all(t.startswith(expect_verdict) for t in verdicts.values())
        print("selftest %s: %s (controls failed: %d; verdicts: %s)" % (label, "OK" if good else "WRONG", bad, "; ".join(t.split(" (")[0] for t in verdicts.values())))
        if not good:
            sys.stdout.write(buf.getvalue())
        ok &= good

    run("world: charge per landing load, controls pass, verdict must say per landing", synth("land"), False, "PER LANDING LOAD")
    run("world: charge per instruction with a wait, verdict must say so", synth("wait"), False, "PER INSTRUCTION WITH A WAIT")
    run("world: charge whenever loads are in flight, verdict must say so", synth("state"), False, "WHENEVER LOADS ARE IN FLIGHT")
    run("deliberate break: matrix-only reads 9.5", synth("land", ref=9.5), True)
    run("deliberate break: aimed loads do not move the cost", synth("land", break_moves=False), True)
    # the six controls against the superseded run: printed, never counted, so a k1s_d54 6 cycles off no longer fails the run ...
    run("k1s_d54 reads 6 cycles off the first ld_mat run: its control is NOT COUNTED, no failure", synth("land", k1_shift=6.0), False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(synth("land", k1_shift=6.0))
    out = buf.getvalue()
    good = out.count("NOT COUNTED") == 6 and "CONTROL b_k1_L0: NOT COUNTED (would fail)" in out and "13 controls counted, 0 failed, 6 not counted" in out
    print("selftest six controls print NOT COUNTED and the summary counts 13 + 6: %s" % ("OK" if good else "WRONG"))
    ok &= good
    # ... and the counted controls still catch faults
    run("deliberate break: bursts not placed", synth("land", placed=0), True)
    # a log with one cell removed must fail a control, not pass silently
    text = "\n".join(l for l in synth("land").splitlines() if " cell k4w_d24 " not in l) + "\n"
    run("deliberate break: a cell missing", text, True)
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    if a[0] == "--selftest":
        return selftest()
    text = "".join(open(p).read() + "\n" for p in a)
    return 1 if report(text)[0] else 0


if __name__ == "__main__":
    sys.exit(main())
