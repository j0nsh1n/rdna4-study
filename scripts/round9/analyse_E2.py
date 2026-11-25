#!/usr/bin/env python3
"""Item E2 analysis: single-wave latency at L2 (1 MiB), the memory-side cache (32 MiB) and GDDR6 (1 GiB), in random and page
order, run beside a light background streaming read (r9_bgread), each chase labelled by the memory clock the 5 ms sampler read
DURING it (the samples inside the chase's own CLOCK_MONOTONIC window; at least 3 samples and at least 90 % in one level). A window
with fewer than 3 samples is "too short" (nothing known), one with enough samples and less than 90 % in one level is "mixed", and
a chase with no sampler file is "unchecked"; only a clock level is a clean label.

  python3 -I analyse_E2.py LABEL,LOG,SAMPLERCSV...   LABEL names the job (e2-idle-*, e2-bg<blocks>-*); LOG is the gated job's log
  python3 -I analyse_E2.py [--expect N] LABEL,LOG,SAMPLERCSV...   (fewer than N jobs, or a job whose chases are not exactly the 7
                                                    in EXPECT_POINTS, once each, fails "complete")
  python3 -I analyse_E2.py --selftest
Controls (expectations in expect/expect-E2.md): every pointer ok; a clean chase at the 1124 label and one at the 1408 label (else
the background did not hold the clock up: the item stays open); the background reads at most 10 % of 700 GB/s; the L2 chase in
215-270 cycles and within 5 % of the idle jobs'; the 32 MiB page-order chase below 470 cycles (not pushed out to GDDR6); two jobs at
one label with background rates 2x apart agree within 5 % at 1 GiB (SKIP when there is no such pair)."""
import importlib.util
import os
import random
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r9lib", os.path.join(HERE, "r9lib.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
_spec2 = importlib.util.spec_from_file_location("analyse_hier", os.path.join(HERE, "analyse_hier.py"))
H = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(H)

BG_MAX_GBPS = 70.0
L2_BAND = (215.0, 270.0)
L2_SAME = 0.05
MALL_PAGE_MAX = 470.0
GDDR_AGREE = 0.05
MIN_SAMPLES = 3
SHARE = 0.90
# the chases every job must hold, once each: (walk, KiB). From scripts/round9/e2_inner.sh.
EXPECT_POINTS = sorted([("random", 1024), ("page", 1024), ("random", 32768), ("page", 32768), ("random", 1048576),
                        ("random_fresh", 1048576), ("page", 1048576)])
# MEASURED, round9.md item E (random walk, cold-start labels), for the printout only
REF = [("L2 1 MiB random", "245.9 cycles at 456"), ("32 MiB random", "552 at 456, 454 at 772, 578 at 875 (bg)"),
       ("1 GiB random fresh", "467-483 at 456, 772, 875"), ("1 GiB random after a long walk", "852-879")]


def label_of(samp, t0, t1):
    """(label, share, n samples, histogram) from the sampler's rows inside [t0, t1] (r9lib.window_label)."""
    return L.window_label(samp, t0, t1, MIN_SAMPLES, SHARE)


def parse_job(label, text, samp):
    pts, bg, blocks = [], None, 0
    for line in text.splitlines():
        if line.startswith("INFO e2 blocks="):
            m = re.match(r"INFO e2 blocks=(\d+)", line)
            blocks = int(m.group(1)) if m else 0
        elif line.startswith("RESULT r9_bgread "):
            d = L.kvs(line)
            try:
                bg = float(d["gbps"])
            except (KeyError, ValueError):
                pass
        elif line.startswith("RESULT r9_lat "):
            d = L.kvs(line)
            try:
                p = dict(job=label, KiB=float(d["KiB"]), walk=d["walk"], cycles=float(d["cycles"]), ns=float(d["ns"]),
                         ok=d["pointer"] == "ok", t0=float(d["t0_ns"]), t1=float(d["t1_ns"]))
            except (KeyError, ValueError):
                continue
            if samp:
                p["label"], p["share"], p["nsamp"], p["hist"] = label_of(samp, p["t0"], p["t1"])
            else:
                p["label"], p["share"], p["nsamp"], p["hist"] = "unchecked", float("nan"), 0, {}
            pts.append(p)
    for p in pts:
        p["blocks"] = blocks
        p["bg"] = bg
    return dict(label=label, blocks=blocks, bg=bg, points=pts)


def analyse(jobs, out=print, expect=None):
    v = L.Verdicts("E2")
    if expect:
        empty = [j["label"] for j in jobs if not j["points"]]
        wrong = [j["label"] for j in jobs if j["points"] and sorted((p["walk"], p["KiB"]) for p in j["points"]) != EXPECT_POINTS]
        v.add("complete", len(jobs) == expect and not empty and not wrong, "%d of %d jobs given%s%s" % (
            len(jobs), expect, "" if not empty else "; no chases in " + ",".join(empty[:4]),
            "" if not wrong else "; chases are not exactly the %d expected (a missing, duplicated or extra one) in %s" % (len(EXPECT_POINTS), ",".join(wrong[:4]))))
    pts = [p for j in jobs for p in j["points"]]
    if not pts:
        v.add("data", False, "no RESULT r9_lat lines found")
        return v
    wrong = [p for p in pts if not p["ok"]]
    v.add("pointers", not wrong, "%d of %d chase points have the right final pointer" % (len(pts) - len(wrong), len(pts)))
    for lab in (1124, 1408):
        got = [p for p in pts if p["label"] == lab]
        v.add("label_%d" % lab, bool(got), "%d chases carry a clean %d MHz label (the sampler's own readings during the chase)%s" % (
            len(got), lab, "" if got else "; the background did not hold the clock there: this part of the item stays open"))
    bgj = [j for j in jobs if j["blocks"] > 0]
    heavy = [j for j in bgj if j["bg"] is None or j["bg"] > BG_MAX_GBPS]
    v.add("bg_light", bool(bgj) and not heavy, "background reads %s GB/s (limit %.0f, 10 %% of 700): %s" % (
        ", ".join("%s %s" % (j["label"], "?" if j["bg"] is None else "%.1f" % j["bg"]) for j in bgj), BG_MAX_GBPS,
        "light" if not heavy else "NOT light: " + ", ".join(j["label"] for j in heavy)))
    l2 = [p for p in pts if p["KiB"] == 1024 and p["walk"] == "random"]
    bad = [p for p in l2 if not L2_BAND[0] <= p["cycles"] <= L2_BAND[1]]
    v.add("l2_band", bool(l2) and not bad, "L2 (1 MiB, random) in %.0f-%.0f cycles in %d of %d chases%s" % (
        *L2_BAND, len(l2) - len(bad), len(l2), "" if not bad else "; outside: " + ", ".join("%s %.0f" % (p["job"], p["cycles"]) for p in bad[:4])))
    idle = [p["cycles"] for p in l2 if p["blocks"] == 0]
    busy = [p["cycles"] for p in l2 if p["blocks"] > 0]
    if idle and busy:
        a, b = L.median(idle), L.median(busy)
        same = abs(b / a - 1) <= L2_SAME
        v.add("l2_unchanged", same, "L2 1 MiB: median %.1f beside the background against %.1f idle (%.1f %%; limit %.0f %%): %s" % (
            b, a, 100 * (b / a - 1), 100 * L2_SAME,
            "within the limit" if same else "outside the limit; the jobs also differ in clocks, so no cause is separated"))
    else:
        v.add("l2_unchanged", None, "no idle job or no background job with a 1 MiB chase: control open")
    mall = [p for p in pts if p["KiB"] == 32768 and p["walk"] == "page"]
    bad = [p for p in mall if p["cycles"] >= MALL_PAGE_MAX]
    v.add("mall_level", bool(mall) and not bad, "32 MiB page-order chase below %.0f cycles (not pushed out to GDDR6) in %d of %d chases%s" % (
        MALL_PAGE_MAX, len(mall) - len(bad), len(mall), "" if not bad else "; at or above: " + ", ".join("%s %.0f" % (p["job"], p["cycles"]) for p in bad[:4])))
    # the same label at two background rates
    g = [p for p in pts if p["KiB"] == 1048576 and p["walk"] == "page" and isinstance(p["label"], int) and p["bg"]]
    pair = None
    for lab in sorted({p["label"] for p in g}):
        x = sorted([p for p in g if p["label"] == lab], key=lambda p: p["bg"])
        if len(x) >= 2 and x[-1]["bg"] >= 2 * x[0]["bg"]:
            pair = (lab, x[0], x[-1])
            break
    if pair:
        lab, lo, hi = pair
        r = hi["cycles"] / lo["cycles"]
        v.add("bg_independence", abs(r - 1) <= GDDR_AGREE, "1 GiB page order at the %d label: %.1f cycles at %.1f GB/s background and %.1f at %.1f GB/s (ratio %.3f; limit %.0f %%)" % (
            lab, lo["cycles"], lo["bg"], hi["cycles"], hi["bg"], r, 100 * GDDR_AGREE))
    else:
        v.add("bg_independence", None, "no two jobs at one label with background rates 2x apart: control open")
    # findings
    out("")
    out("chases by size, walk and clean memory-clock label (median over jobs; cycles are core cycles, ns from REALTIME):")
    keys = sorted({(p["KiB"], p["walk"], str(p["label"])) for p in pts}, key=lambda k: (k[0], k[1], k[2]))
    for k in keys:
        sel = [p for p in pts if (p["KiB"], p["walk"], str(p["label"])) == k]
        out("  %9.0f KiB %-13s label %-6s n=%-3d %7.1f cycles %8.2f ns   background %s GB/s" % (
            k[0], k[1], k[2], len(sel), L.median([p["cycles"] for p in sel]), L.median([p["ns"] for p in sel]),
            ",".join(sorted({"%.0f" % p["bg"] if p["bg"] else "0" for p in sel}))))
    out("")
    for kib in (1024, 32768, 1048576):
        for walk in ("random", "page", "random_fresh"):
            by = {}
            for p in pts:
                if p["KiB"] == kib and p["walk"] == walk and isinstance(p["label"], int):
                    by.setdefault(p["label"], []).append(p)
            if len(by) >= 2:
                cyc = {k: L.median([p["cycles"] for p in x]) for k, x in by.items()}
                ns = {k: L.median([p["ns"] for p in x]) for k, x in by.items()}
                out("RESULT E2 latency_vs_memory_clock KiB=%d walk=%s levels=%s cycles=%s ns=%s ratio_cycles=%.3f ratio_ns=%.3f %s" % (
                    kib, walk, ",".join(str(k) for k in sorted(by)), ",".join("%.0f" % cyc[k] for k in sorted(by)),
                    ",".join("%.1f" % ns[k] for k in sorted(by)), max(cyc.values()) / min(cyc.values()), max(ns.values()) / min(ns.values()),
                    "(above 1.10: varies across the observed labels)" if max(max(cyc.values()) / min(cyc.values()), max(ns.values()) / min(ns.values())) > 1.10
                    else "(within 1.10 across the observed labels)"))
            elif by:
                out("RESULT E2 latency_vs_memory_clock KiB=%d walk=%s: only the label %s seen, no comparison" % (kib, walk, ",".join(str(k) for k in by)))
    out("NOTE: the background traffic and the memory-clock label change together in these jobs, so this analysis cannot separate a "
        "memory-clock contribution from a traffic contribution; a ratio above is an association. The one control that holds a label "
        "and varies the traffic is bg_independence.")
    out("reference (round9.md, cold-start labels, random walk): " + "; ".join("%s: %s" % r for r in REF))
    return v


def synth(rng, bg_gbps=(20.0, 45.0), l2=246.0, ptr_ok=True, l2_bg_shift=1.0, mall=372.0, labels=(1124, 1408), bg_effect=1.0):
    """Three jobs: idle (456), bg blocks 2 (labels[0]), bg blocks 8 (labels[1]); each with 6 chases and a sampler."""
    jobs = []
    t = 1e12
    plan = [("e2-idle-r1", 0, 456, None), ("e2-bg2-r1", 2, labels[0], bg_gbps[0]), ("e2-bg8-r1", 8, labels[1], bg_gbps[1])]
    for name, blocks, lev, gb in plan:
        samp = dict(mono_ns=[t + i * 5e6 for i in range(4000)], uclk_cur_MHz=[float(lev)] * 4000)
        lines = ["INFO e2 blocks=%d background settled, uclk=%d" % (blocks, lev)]
        tt = t + 10e6
        for kib, walk, cyc in ((1024, "random", l2 * (l2_bg_shift if blocks else 1.0)), (1024, "page", 250.0), (32768, "page", mall), (32768, "random", 450.0),
                               (1048576, "page", 515.0 * (bg_effect if blocks == 8 else 1.0)), (1048576, "random_fresh", 470.0), (1048576, "random", 860.0)):
            dur = 60e6 if kib == 1024 else 200e6
            lines.append("RESULT r9_lat mode=lat walk=%s KiB=%d cycles=%.1f ns=%.2f pointer=%s uclk_before=%d uclk_after=%d uclk_process_start=456 trials=40 t0_ns=%d t1_ns=%d" % (
                walk, kib, cyc * rng.uniform(0.995, 1.005), cyc / 3.0, "ok" if ptr_ok else "WRONG", lev, lev, tt, tt + dur))
            tt += dur + 5e6
        if gb is not None:
            lines.append("RESULT r9_bgread blocks=%d MiB=1024 iters=2048 launches=300 seconds=10.00 gbps=%.1f hotspot_C=50" % (blocks, gb))
        jobs.append(parse_job(name, "\n".join(lines), samp))
    return jobs


def selftest():
    def case(**kw):
        return lambda: analyse(synth(random.Random(3), **kw), out=lambda *_: None)

    cases = [
        ("base", True, case()),
        ("break labels not reached (everything at 456)", False, case(labels=(456, 456)), "label_"),
        ("break background too heavy (120 GB/s)", False, case(bg_gbps=(60.0, 120.0)), "bg_light"),
        ("break L2 moved by the background (+15 %)", False, case(l2_bg_shift=1.15), "l2_"),
        ("break wrong pointer", False, case(ptr_ok=False), "pointers"),
        ("break 32 MiB pushed out of the memory-side cache (510)", False, case(mall=510.0), "mall_level"),
        ("break 1 GiB changes with the background rate at one label (+12 %)", False, case(labels=(1408, 1408), bg_effect=1.12), "bg_independence"),
        ("break no data", False, lambda: analyse([], out=lambda *_: None), "data"),
    ]
    ok = L.run_selftest("E2", cases)
    import contextlib
    import io

    def lines(**kw):
        buf = []
        with contextlib.redirect_stdout(io.StringIO()):
            analyse(synth(random.Random(3), **kw), out=buf.append)
        return buf

    base = lines()
    tags = [x for x in base if x.startswith("RESULT E2 latency_vs_memory_clock") and "ratio_cycles" in x]
    g1 = bool(tags) and all(("above 1.10: varies across the observed labels" in x) == (max(float(x.split("ratio_cycles=")[1].split()[0]), float(x.split("ratio_ns=")[1].split()[0])) > 1.10) for x in tags)
    g2 = any("within 1.10 across the observed labels" in x for x in tags) and not any("depends" in x for x in base)
    g3 = any("cannot separate a memory-clock contribution" in x for x in base)
    jobs3 = synth(random.Random(3))
    dup = [dict(jobs3[0], points=jobs3[0]["points"][:1] + jobs3[0]["points"][:1] + jobs3[0]["points"][2:])] + jobs3[1:]
    first = [dict(j, points=j["points"][:1]) for j in jobs3]
    for nm, jb, want in (("every expected job present: complete passes", jobs3, True), ("break: a job missing: complete fails", jobs3[:2], False),
                         ("break: a job without chases: complete fails", [dict(jobs3[0], points=[])] + jobs3[1:], False),
                         ("break: a duplicate chase replaces another: complete fails", dup, False),
                         ("break: every job cut to its first chase: complete fails", first, False)):
        with contextlib.redirect_stdout(io.StringIO()):
            vv = analyse(jb, out=lambda *_: None, expect=3)
        gg = any(r[0] == "complete" and bool(r[1]) == want for r in vv.rows)
        print("SELFTEST E2 %s: %s" % (nm, "ok" if gg else "WRONG"))
        ok &= gg
    for nm, shift, want in (("l2_unchanged PASS line says within the limit", 1.0, True), ("break: l2_unchanged FAIL line does not carry the pass wording", 1.15, False)):
        with contextlib.redirect_stdout(io.StringIO()) as cap:
            analyse(synth(random.Random(3), l2_bg_shift=shift), out=lambda *_: None)
        row = [x for x in cap.getvalue().splitlines() if " l2_unchanged: " in x]
        gg = len(row) == 1 and (("PASS" in row[0] and "within the limit" in row[0]) if want else ("FAIL" in row[0] and "does not disturb" not in row[0] and "outside the limit" in row[0]))
        print("SELFTEST E2 %s: %s" % (nm, "ok" if gg else "WRONG"))
        ok &= gg
    for nm, g in (("the tag follows the ratio (above 1.10 only when it is above)", g1), ("a ratio within 1.10 says so and nothing says depends", g2),
                  ("the output says it cannot separate a memory-clock contribution", g3)):
        print("SELFTEST E2 tag: %s: %s" % (nm, "ok" if g else "WRONG"))
        ok &= g
    # the label function itself: a chase window with only 2 samples is "mixed"; a window at 85 % in one level is "mixed"; 95 % is clean
    mono = [i * 5e6 for i in range(100)]
    s = dict(mono_ns=mono, uclk_cur_MHz=[1124.0] * 100)
    a = label_of(s, 0, 9e6)[0] == "too short"
    b = label_of(s, 0, 4e8)[0] == 1124
    s2 = dict(mono_ns=mono, uclk_cur_MHz=[1124.0] * 85 + [1408.0] * 15)
    c = label_of(s2, 0, 5e8)[0] == "mixed"
    s3 = dict(mono_ns=mono, uclk_cur_MHz=[1124.0] * 96 + [1408.0] * 4)
    d = label_of(s3, 0, 5e8)[0] == 1124
    for nm, g in (("window with 2 samples is too short, not mixed", a), ("window with 80 clean samples is 1124", b), ("85 % in one level is mixed", c), ("96 % in one level is clean", d)):
        print("SELFTEST E2 label: %s: %s" % (nm, "ok" if g else "WRONG"))
        ok &= g
    return ok


def main(argv):
    if argv and argv[0] == "--selftest":
        return 0 if selftest() else 1
    expect = None
    if argv[:1] == ["--expect"] and len(argv) > 1:
        expect = int(argv[1])
        argv = argv[2:]
    if not argv:
        print(__doc__)
        return 2
    jobs = []
    for a in argv:
        parts = a.split(",")
        try:
            with open(parts[1], errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""  # a missing log: the job has no chases, and `complete` says so
        samp = None
        if len(parts) > 2 and os.path.exists(parts[2]):
            with open(parts[2], errors="replace") as f:
                samp = H.read_sampler(f.read())
        jobs.append(parse_job(parts[0], text, samp))
    v = analyse(jobs, expect=expect)
    return 0 if v.summary() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
