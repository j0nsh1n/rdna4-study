#!/usr/bin/env python3
"""Analysis of ld_mat (round 9, item A). Standard library only; run with python3 -I.

Usage: ana_ld_mat.py [--hit CYCLES] <log> [<log>...]   the logs of the card and one-wave jobs together (any order)
       ana_ld_mat.py --selftest                         synthetic input, with deliberate breaks that must FAIL
Prints "CONTROL <name>: PASS|FAIL ...", then tables (INFO) of cycles per matrix instruction and the extra over the
matrix-only reference, and "SUMMARY ...". Exit 1 if any control fails.
--hit: the pure L0 hit cost of the ld_order run of the same session (cycles); control (b) then expects the extra per
group of the one-wave L0 D = 0 cell to be that plus the 18-cycle wake charge of matrix-timing.md 8a, within 15 %.

Controls (expect-ld_mat.md has the numbers, written before the run):
  a_ref        matrix-only reference: 8.0 cycles per matrix instruction one wave (7.9-8.1), whole card 7.9-8.3 per register count
  b_8a         one wave, L0, D = 0: extra per group equals the load-wait cost of 8a plus the matrix wake charge
  c_back       one wave, D = all, L0: data is back before use, no extra over (a) within 1 %
  d_break      loads aimed one level lower than labelled move the D = 0 cost down by at least 70 % of the latency difference
  placed       every whole-card burst had W waves on each of the 128 SIMDs
(e, the listing check of every code object, is check_listing.py in build_round9.sh.)

Burst guard (round 2 review, Neon finding 1): before any control, a burst is rejected, and left out of every cell value, when its
clock_ghz lies outside 0.5-4.0 or its per-SIMD min or max differs from its median by more than 20 %. The count and the bursts
are printed ("GUARD ..."); a rejected burst is a data fault, not a control, so the control count does not change. A separate data check
(CHECK cells_survive, not one of the controls) fails any cell left with no burst after the guard, whichever controls read that cell. The cause of the
one archived rejection (a 58 GHz clock) is open: the code's unsigned subtraction survives a plain 32-bit wrap.
"""
import statistics
import sys

REF_ONE = (7.9, 8.1)
REF_CARD = (7.9, 8.3)
LEVELS = ["L0", "L2", "MALL", "GDDR"]
CHARGE = 18.0
CLOCK_GHZ = (0.5, 4.0)
SPREAD_MAX = 0.20


def bad_burst(d):
    """Why a burst cannot be used, or None: an impossible clock, or a min/max too far from the median of the same burst."""
    try:
        clk, med, lo, hi = float(d["clock_ghz"]), float(d["cyc_per_mm_med"]), float(d["min"]), float(d["max"])
    except (KeyError, ValueError):
        return "unreadable burst line"
    if not CLOCK_GHZ[0] <= clk <= CLOCK_GHZ[1]:
        return "clock_ghz %.3f outside %.1f-%.1f" % (clk, *CLOCK_GHZ)
    if med <= 0 or max(abs(lo - med), abs(hi - med)) > SPREAD_MAX * med:
        return "min %.3f / max %.3f more than %.0f %% from the median %.3f" % (lo, hi, 100 * SPREAD_MAX, med)
    return None


def parse(text, guard=True):
    """-> (bursts by cell key, rejected bursts as (key, rep, reason)). guard=False keeps every burst (the self-test's break)."""
    bursts, rejected = {}, []
    for ln in text.splitlines():
        if not ln.startswith("RESULT ld_mat burst "):
            continue
        t = ln.split()[3:]
        d = dict(zip(t[0::2], t[1::2]))
        key = (d["mode"], int(d["R"]), d["level"], d["aimed"], d["D"])
        why = bad_burst(d) if guard else None
        bursts.setdefault(key, [])  # a cell whose every burst was rejected stays in the table as nan, and its control fails
        if why:
            rejected.append((key, d.get("rep", "?"), why, d.get("placed")))
            continue
        bursts[key].append(d)
    return bursts, rejected


def cell_value(bs):
    ok = [float(b["cyc_per_mm_med"]) for b in bs if b["placed"] == "1"]
    return statistics.median(ok) if ok else float("nan")


def judge(bursts, hit=None, rejected=()):
    res = []

    def ctl(name, ok, text):
        res.append((name, bool(ok), text))

    val = {k: cell_value(v) for k, v in bursts.items()}
    one_ref = [v for k, v in val.items() if k[0] == "one" and k[4] == "ref"]
    if one_ref:
        r = statistics.median(one_ref)
        ctl("a_ref_one", REF_ONE[0] <= r <= REF_ONE[1], "one wave, matrix only: %.3f cycles per matrix instruction (8.0; gate %.1f-%.1f)" % (r, *REF_ONE))
    else:
        ctl("a_ref_one", False, "no one-wave matrix-only cell in the logs")
    refs = {k[1]: v for k, v in val.items() if k[0] == "card" and k[4] == "ref"}
    if refs:
        ctl("a_ref_card", all(REF_CARD[0] <= v <= REF_CARD[1] for v in refs.values()),
            "whole card, matrix only: " + ", ".join("R %d %.3f" % (r, v) for r, v in sorted(refs.items())) + " (8.034 at W = 8; gate %.1f-%.1f)" % REF_CARD)
    else:
        ctl("a_ref_card", False, "no whole-card matrix-only cell in the logs")
    ref1 = statistics.median(one_ref) if one_ref else 8.0

    def one(level, d, aimed=None):
        return val.get(("one", 209, level, aimed or level, d))

    l0 = one("L0", "0")
    if l0 is not None and l0 == l0:
        extra = l0 - ref1
        if hit is not None:
            want = hit + CHARGE
            ctl("b_8a", abs(extra - want) <= 0.15 * want, "one wave, L0, D = 0: extra %.1f cycles per group; ld_order hit %.1f + wake charge %.0f = %.1f (gate 15 %%)" % (extra, hit, CHARGE, want))
        else:
            a8 = 77.0 + CHARGE
            sa = 130.0 + CHARGE
            ok = abs(extra - a8) <= 0.2 * a8 or abs(extra - sa) <= 0.15 * sa
            ctl("b_8a", ok, "one wave, L0, D = 0: extra %.1f cycles per group; 8a form 77 + 18 = %.0f, saddr form ~130 + 18 = %.0f (pass if near either; give --hit to be exact)" % (extra, a8, sa))
    else:
        ctl("b_8a", False, "no one-wave L0 D = 0 cell (R 209)")
    lall = one("L0", "all")
    if lall is not None and lall == lall:
        ctl("c_back", abs(lall - ref1) <= 0.01 * ref1, "one wave, L0, D = all (60): %.3f against matrix only %.3f (%.2f %%; gate 1 %%). Judged after run 1: matrix-timing.md 8a (load data arriving during a matrix chain: +26.0) predicts %.2f, so the gate premise was wrong, see round9.md" % (lall, ref1, 100 * (lall / ref1 - 1), ref1 + 26.0))
    else:
        ctl("c_back", False, "no one-wave L0 D = all cell")
    brk = [(k, v) for k, v in val.items() if k[0] == "one" and k[2] != k[3] and k[4] == "0"]
    if brk:
        for k, v in brk:
            lab, aim = k[2], k[3]
            true_lab, true_aim = one(lab, "0"), one(aim, "0")
            if true_lab is None or true_aim is None:
                ctl("d_break_" + lab, False, "break %s aimed at %s: the normal cells are missing" % (lab, aim))
                continue
            diff = true_lab - true_aim
            moved = true_lab - v
            ctl("d_break_" + lab, diff > 0 and moved >= 0.7 * diff,
                "loads labelled %s but aimed at %s: D = 0 cost %.1f (normal %s %.1f, %s %.1f): moved %.1f of %.1f (gate 70 %%)" % (lab, aim, v, lab, true_lab, aim, true_aim, moved, diff))
    else:
        ctl("d_break", False, "no deliberate-break cell (label differs from aim) in the logs")
    bad = [(k, [b["placed"] for b in v]) for k, v in bursts.items() if k[0] == "card" and any(b["placed"] != "1" for b in v)]
    bad += [(k, [pl]) for k, _, _, pl in rejected if k[0] == "card" and pl != "1"]  # a burst the guard rejected was still placed or not
    n = sum(len(v) for k, v in bursts.items() if k[0] == "card") + sum(1 for k, *_ in rejected if k[0] == "card")
    ctl("placed", n > 0 and not bad, "%d whole-card bursts, %d cells with a burst not placed as W waves on 128 SIMDs" % (n, len(bad)))
    return res, val, ref1


def tables(val, ref1):
    out = []
    refc = {k[1]: v for k, v in val.items() if k[0] == "card" and k[4] == "ref"}
    out.append("INFO one wave (R 209): cycles per matrix instruction (extra over %.2f), by level and lookahead" % ref1)
    ds = ["0", "1", "2", "4", "8", "all"]
    out.append("level   " + " ".join("%8s" % ("D=" + d) for d in ds))
    for lv in LEVELS:
        row = []
        for d in ds:
            v = val.get(("one", 209, lv, lv, d))
            row.append("%8s" % ("-" if v is None else "%.2f" % v))
        out.append("%-7s " % lv + " ".join(row))
    out.append("INFO model for one wave: extra(D) = max(0, extra(0) - 8 D); measured / model extra")
    for lv in LEVELS:
        e0 = val.get(("one", 209, lv, lv, "0"))
        if e0 is None:
            continue
        e0 -= ref1
        row = []
        for d, dv in zip(ds, [0, 1, 2, 4, 8, 60]):
            v = val.get(("one", 209, lv, lv, d))
            if v is None:
                row.append("%12s" % "-")
            else:
                m = max(0.0, e0 - 8 * dv)
                row.append("%12s" % ("%.1f/%.1f" % (v - ref1, m)))
        out.append("%-7s " % lv + " ".join(row))
    for R in (51, 90, 167, 209):
        out.append("INFO whole card R %d: cycles per matrix instruction per SIMD (matrix only %s)" % (R, "%.3f" % refc[R] if R in refc else "-"))
        out.append("level   " + " ".join("%8s" % ("D=" + d) for d in ds))
        for lv in LEVELS:
            row = []
            for d in ds:
                v = val.get(("card", R, lv, lv, d))
                row.append("%8s" % ("-" if v is None else "%.2f" % v))
            out.append("%-7s " % lv + " ".join(row))
        gains = []
        for lv in LEVELS:
            a, b = val.get(("card", R, lv, lv, "0")), val.get(("card", R, lv, lv, "4"))
            if a is not None and b is not None and a == a and b == b:
                gains.append("%s %.1f %%" % (lv, 100 * (a - b) / a))
        if gains:
            out.append("INFO gain from D = 0 to D = 4 at R %d: %s" % (R, ", ".join(gains)))
    return out


def report(text, hit=None, guard=True):
    bursts, rejected = parse(text, guard)
    n_all = sum(len(v) for v in bursts.values()) + len(rejected)
    print("GUARD %d of %d bursts rejected (clock_ghz outside %.1f-%.1f, or min/max more than %.0f %% from the median)%s" % (
        len(rejected), n_all, *CLOCK_GHZ, 100 * SPREAD_MAX, "" if not rejected else ": " + "; ".join(
            "%s R %d %s D %s rep %s: %s" % (k[0], k[1], k[2], k[4], rep, why) for k, rep, why, _ in rejected)))
    res, val, ref1 = judge(bursts, hit, rejected)
    empty = sorted(k for k, v in bursts.items() if not v)
    check_bad = 1 if empty else 0
    print("CHECK cells_survive: %s %d cells with no burst left after the guard%s" % (
        "FAIL" if empty else "PASS", len(empty), "" if not empty else ": " + "; ".join(
            "%s R %d %s aimed %s D %s" % k for k in empty)))
    bad = 0
    for n, ok, t in res:
        print("CONTROL %s: %s %s" % (n, "PASS" if ok else "FAIL", t))
        bad += not ok
    for ln in tables(val, ref1):
        print(ln)
    print("SUMMARY %d controls, %d failed, %d bursts rejected by the guard; data check cells_survive %s" % (
        len(res), bad, len(rejected), "FAIL" if check_bad else "PASS"))
    return bad + check_bad, len(rejected)


# ---------------------------------------------------------------------------------------------------------------
LAT = {"L0": 130.0, "L2": 290.0, "MALL": 460.0, "GDDR": 650.0}
DALL = {51: 10, 90: 30, 167: 60, 209: 60}
W = {51: 16, 90: 16, 167: 9, 209: 7}


def synth(ref=8.0, break_moves=True, back_extra=0.0, placed=1, ref_card=8.034, corrupt=0, corrupt_how="clock", target=("card", 51, "-", "ref")):
    """corrupt: how many of the 3 bursts of the cell target (default the whole-card R 51 matrix-only cell) are corrupted (corrupt_how "clock": a 58 GHz clock
    and a 312-cycle median as in the archived burst; "spread": a sane clock and a min 30 % below the median)."""
    L = []

    def line(mode, R, level, aimed, D, cyc, pl=1):
        dval = -1 if D == "ref" else (DALL[R] if D == "all" else int(D))
        for rep in range(3):
            c, lo, hi, clk, plb = cyc, cyc, cyc, 3.0, pl
            if (mode, R, level, D) == target and rep < corrupt:
                if corrupt_how in ("clock", "clock_unplaced"):
                    c, lo, hi, clk = 312.036, 14.157, 466.754, 58.076
                    if corrupt_how == "clock_unplaced":
                        plb = 0
                else:
                    lo = 0.7 * cyc
            L.append("RESULT ld_mat burst mode %s R %d W %d level %s aimed %s D %s dval %d kernel k rep %d iters 1000 ms 50.00 cyc_per_mm_med %.3f min %.3f max %.3f clock_ghz %.3f placed %d simds %d hot_start 40 hot_max 80"
                     % (mode, R, W[R] if mode == "card" else 1, level, aimed, D, dval, rep, c, lo, hi, clk, plb, 128 if mode == "card" else 1))

    line("one", 209, "-", "-", "ref", ref)
    for R in (51, 90, 167, 209):
        line("card", R, "-", "-", "ref", ref_card)
    for lv in LEVELS:
        for d in ("0", "1", "2", "4", "8", "all"):
            dv = DALL[209] if d == "all" else int(d)
            extra = max(0.0, LAT[lv] + CHARGE - 8 * dv)
            line("one", 209, lv, lv, d, ref + extra + (back_extra if (lv == "L0" and d == "all") else 0.0))
            for R in (51, 90, 167, 209):
                dvr = DALL[R] if d == "all" else int(d)
                ex = max(0.0, LAT[lv] + CHARGE - 8 * dvr) / W[R]
                line("card", R, lv, lv, d, ref_card + ex, placed)
    # the break: labelled L2, loads aimed at L0
    mv = LAT["L0"] if break_moves else LAT["L2"]
    line("one", 209, "L2", "L0", "0", ref + mv + CHARGE)
    return "\n".join(L) + "\n"


def selftest():
    ok = True

    def run(label, text, expect_fail, hit=None, rejected=0, guard=True):
        nonlocal ok
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bad, rej = report(text, hit, guard)
        good = (bad > 0) == expect_fail and rej == rejected
        print("selftest %s: %s (controls failed: %d, bursts rejected: %d, expected %d)" % (label, "OK" if good else "WRONG", bad, rej, rejected))
        if not good:
            sys.stdout.write(buf.getvalue())
        ok &= good

    run("synthetic card, every control passes", synth(), False)
    run("synthetic card with --hit 130", synth(), False, hit=130.0)
    run("deliberate break: matrix-only reference reads 9.5", synth(ref=9.5), True)
    run("deliberate break: aimed loads do not move the cost", synth(break_moves=False), True)
    run("deliberate break: L0 D = all shows 5 extra cycles", synth(back_extra=5.0), True)
    run("deliberate break: bursts not placed as 128 x W", synth(placed=0), True)
    run("deliberate break: --hit 200 against a 130-cycle hit", synth(), True, hit=200.0)
    run("guard: one burst with a 58 GHz clock is rejected and every control still passes", synth(corrupt=1), False, rejected=1)
    run("guard: two of three bursts of the reference cell corrupted: the clean one is used, controls pass", synth(corrupt=2), False, rejected=2)
    run("guard: a burst whose min is 30 % below its median (sane clock) is rejected", synth(corrupt=1, corrupt_how="spread"), False, rejected=1)
    run("deliberate break: the same two corrupted bursts WITHOUT the guard fail a_ref_card", synth(corrupt=2), True, guard=False)
    run("deliberate break: a rejected burst that was also not placed still fails the placed control", synth(corrupt=1, corrupt_how="clock_unplaced"), True, rejected=1)
    run("deliberate break: all three bursts of the reference cell corrupted (nothing left to judge) fails", synth(corrupt=3), True, rejected=3)
    other = ("card", 90, "L2", "4")  # a cell no control reads
    run("guard: one of three bursts of a cell no control reads is rejected: passes", synth(corrupt=1, target=other), False, rejected=1)
    run("deliberate break: all three bursts of a cell no control reads rejected: only cells_survive fails", synth(corrupt=3, target=other), True, rejected=3)
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    if a[0] == "--selftest":
        return selftest()
    hit = None
    if a[0] == "--hit":
        hit = float(a[1])
        a = a[2:]
    text = "".join(open(p).read() + "\n" for p in a)
    return 1 if report(text, hit)[0] else 0


if __name__ == "__main__":
    sys.exit(main())
