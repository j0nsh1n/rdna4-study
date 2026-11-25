#!/usr/bin/env python3
"""Item E analysis: latency and bandwidth of L0, L2, the memory-side cache and GDDR6, each point labelled by the memory clock
(gpu_metrics current_uclk read right before and after it), and the dependence of the cache levels on that clock.

  python3 -I analyse_hier.py LABEL,LOG[,SAMPLERCSV]...     LABEL starts e-lat, e-bw, e-cold or e-bg; LOG is the gated job's
                                                           stdout (RESULT r9_lat / r9_bw / mem_stream lines, JOB_MONO_* lines)
  python3 -I analyse_hier.py [--expect e-lat=3,e-bw=3,e-cold=12,e-bg=3] LABEL,LOG[,SAMPLERCSV]...   (a missing or empty job fails "complete")
  python3 -I analyse_hier.py --selftest
An ENDPOINT label is the clock read right before and right after a point (equal, else "mixed"). The 90 % rule: with a sampler
file, a point (its own t0_ns/t1_ns window) keeps its endpoint label only if at least 3 sampler rows fall inside the window and at
least 90 % of them read that clock. Every point gets one of five states and a point is clean only in the first:
  clean              the endpoints agree and the window holds at least 3 sampler rows, at least 90 % of them at that clock
  too short          the endpoints agree but the window holds fewer than 3 rows of the 5 ms sampler: nothing is known about the
                     clock inside it (the first version called these "mixed")
  endpoints differ   the reading before the point differs from the reading after it: not evaluated against a window
  mixed              the endpoints agree, the window holds at least 3 rows and less than 90 % of them at one clock
  unchecked          no sampler file or no t0_ns/t1_ns for the point: the endpoint label alone is never taken as clean
Only clean points enter a label comparison or a label control; the others are kept apart under their state. A
background job (e-bg) counts only if the driver's checks printed "INFO bg alive after" for the start, r9_lat and r9_bw and no
"INFO bg DEAD" (the background loop was running through the probes); otherwise all its points are rejected and the verdict
bg_running fails. Latency is compared across labels per job state (steady, cold, bg) and never pooled across states; the
comparison says "varies across the observed labels" and shows no cause. The job's own level histogram comes from the 5 ms
sampler inside the job's window. Controls: L0 75-90 cycles; L2 (1-4 MiB) 215-270; GDDR6 bandwidth at the 1408 label
700.3 +-1.5 % (SKIP when no such job); label against bandwidth within 3 % of 0.5031 GB/s per MHz, and the label-shift break
(every label moved one level) must be caught; every pointer ok. With --expect the `complete` control also needs every job to hold
exactly the points its driver asks for (EXPECT_POINTS: kind, walk and size, once each): a duplicate, a missing or an extra point
fails it."""
import importlib.util
import os
import random
import re
import sys

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("r9lib", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r9lib.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

LEVELS = [456, 772, 875, 1124, 1258, 1408]
L0_BAND = (75.0, 90.0)
L2_BAND = (215.0, 270.0)
BW_1408 = (700.3, 0.015)
GBPS_PER_MHZ = 0.5031
LABEL_TOL = 0.03
# the points each family of jobs must hold, once each: (kind, walk or "", KiB). From scripts/round9/run_E.sh and e_bg_inner.sh.
_LAT = [("lat", "random", k) for k in (16, 1024, 2048, 4096, 32768, 524288)] + [("lat", "random_fresh", 524288),
                                                                              ("lat", "random", 1048576), ("lat", "random_fresh", 1048576)]
EXPECT_POINTS = {
    "e-lat": sorted(_LAT),
    "e-bw": sorted(("bw", "", k) for k in (16, 1024, 2048, 4096, 32768, 524288, 1048576)),
    "e-cold": sorted(("lat", "random", k) for k in (16, 4096, 32768, 524288)),
    "e-bg": sorted([x for x in _LAT if x[2] < 1048576] + [("bw", "", 32768), ("bw", "", 1048576)]),
}


def parse_job(label, text, samp):
    """-> dict(label, points=[...], hist=..., share=...). A point: dict(kind lat|bw, KiB, cycles, ns, gbps, ok, before, after, tag)."""
    pts, last_bw = [], None
    t0 = t1 = None
    for line in text.splitlines():
        if line.startswith("JOB_MONO_START"):
            t0 = float(line.split()[1])
        elif line.startswith("JOB_MONO_END"):
            t1 = float(line.split()[1])
        elif line.startswith("RESULT r9_lat "):
            d = L.kvs(line)
            try:
                pts.append(dict(kind="lat", walk=d["walk"], mode=d["mode"], KiB=float(d["KiB"]), cycles=float(d["cycles"]), ns=float(d["ns"]),
                                ok=d["pointer"] == "ok", before=int(d["uclk_before"]), after=int(d["uclk_after"]),
                                t0=float(d["t0_ns"]) if "t0_ns" in d else None, t1=float(d["t1_ns"]) if "t1_ns" in d else None))
            except (KeyError, ValueError):
                pass
        elif line.startswith("RESULT mem_stream size split"):
            m = re.search(r"S_KiB ([\d.]+) .* GBps ([\d.]+)", line)
            if m:
                last_bw = (float(m.group(1)), float(m.group(2)))
        elif line.startswith("RESULT r9_bw ") and last_bw:
            d = L.kvs(line)
            pts.append(dict(kind="bw", KiB=last_bw[0], gbps=last_bw[1], ok=True, before=int(d["uclk_before"]), after=int(d["uclk_after"]),
                            t0=float(d["t0_ns"]) if "t0_ns" in d else None, t1=float(d["t1_ns"]) if "t1_ns" in d else None))
            last_bw = None
    hist, share = {}, float("nan")
    if samp and t0 is not None and t1 is not None:
        for t, u in zip(samp["mono_ns"], samp["uclk_cur_MHz"]):
            if t0 <= t <= t1 and u > 0:
                hist[int(u)] = hist.get(int(u), 0) + 1
        n = sum(hist.values())
        share = max(hist.values()) / n if n else float("nan")
    state = "bg" if "-bg" in label else "cold" if "-cold" in label else "steady"
    rejected = None
    if state == "bg":
        info = [ln for ln in text.splitlines() if ln.startswith("INFO bg ") or ln.startswith("BACKGROUND LOOP NOT")]
        steps = set()
        for ln in info:
            m = re.match(r"INFO bg alive after (\S+) launches=(\d+)", ln)
            if m:
                steps.add(m.group(1))
        dead = [ln for ln in info if "DEAD" in ln or ln.startswith("BACKGROUND LOOP NOT")]
        if dead:
            rejected = dead[0]
        elif steps != {"start", "r9_lat", "r9_bw"}:
            rejected = "no proof that the background loop ran through the probes (alive checks seen: %s)" % (",".join(sorted(steps)) or "none")
    for p in pts:
        p["tag"] = label
        p["state"] = state
        p["windowed"] = p.get("t0") is not None and samp is not None
        if not (p["before"] == p["after"] and p["before"] > 0):
            lev, why = "endpoints differ", "endpoints differ"
        elif not p["windowed"]:
            lev, why = "unchecked", "unchecked"
        else:  # the 90 % rule on the point's own window
            w = L.window_label(samp, p["t0"], p["t1"])[0]
            if w == p["before"]:
                lev, why = w, "clean"
            elif w == "too short":
                lev, why = "too short", "too short"
            else:
                lev, why = "mixed", "mixed"
        p["level"] = lev
        p["why"] = why
    return dict(label=label, points=pts, hist=hist, share=share, state=state, rejected=rejected)


def clean(p):
    """A point with a clean label: an int level that the window corroborated."""
    return isinstance(p["level"], int)


def one_level_up(level):
    if level not in LEVELS:
        return level
    i = LEVELS.index(level)
    return LEVELS[i + 1] if i + 1 < len(LEVELS) else LEVELS[i - 1]


def label_checks(v, pts, shifted=False):
    """Bandwidth over label: every 1 GiB read point with a clean label of 1124 or above. Returns (n eligible, n failing)."""
    elig = [p for p in pts if p["kind"] == "bw" and p["KiB"] >= 1048576 and clean(p) and p["level"] >= 1124]
    bad = []
    for p in elig:
        lab = one_level_up(p["level"]) if shifted else p["level"]
        r = p["gbps"] / lab / GBPS_PER_MHZ
        if abs(r - 1) > LABEL_TOL:
            bad.append((p["tag"], lab, p["gbps"], r))
    return len(elig), bad


def point_ids(j):
    """The identities of a job's points, sorted: (kind, walk, KiB)."""
    return sorted((p["kind"], p.get("walk", ""), p["KiB"]) for p in j["points"])


def complete(v, jobs, expect):
    """expect: family prefix -> number of jobs (e-lat 3, e-bw 3, e-cold 12, e-bg 3). A missing job, a job with no points, or a job
    whose points are not exactly EXPECT_POINTS[family] (once each; a duplicate replacing another point counts as both) is a FAIL."""
    miss = []
    for fam, n in expect.items():
        have = [j for j in jobs if j["label"].startswith(fam + "-")]
        empty = [j["label"] for j in have if not j["points"]]
        if len(have) != n:
            miss.append("%s: %d of %d jobs given" % (fam, len(have), n))
        if empty:
            miss.append("%s: no points in %s" % (fam, ",".join(empty[:4])))
        want = EXPECT_POINTS.get(fam)
        if want is not None:
            for j in have:
                got = point_ids(j)
                if j["points"] and got != want:
                    lost = [x for x in want if got.count(x) == 0]
                    dup = sorted({x for x in got if got.count(x) > 1})
                    extra = sorted({x for x in got if x not in want})
                    miss.append("%s: points are not the expected %d (missing %s, duplicated %s, unexpected %s)" % (
                        j["label"], len(want), lost[:2] or "none", dup[:2] or "none", extra[:2] or "none"))
    v.add("complete", not miss, "%s: %s" % (", ".join("%s x%d" % kv for kv in expect.items()), "all jobs present with exactly their points" if not miss else "; ".join(miss[:6])))


def analyse(jobs, out=print, expect=None):
    v = L.Verdicts("E")
    if expect:
        complete(v, jobs, expect)
    bgj = [j for j in jobs if j["state"] == "bg"]
    rej = [j for j in bgj if j.get("rejected")]
    if bgj:
        v.add("bg_running", not rej, "%d of %d background jobs have the driver's proof that the FP16 loop ran through the probes%s" % (
            len(bgj) - len(rej), len(bgj), "" if not rej else "; REJECTED (no point of these is used): " + "; ".join("%s: %s" % (j["label"], j["rejected"][:90]) for j in rej)))
    else:
        v.add("bg_running", None, "no background job in the input: control open")
    jobs = [j for j in jobs if not j.get("rejected")]
    pts = [p for j in jobs for p in j["points"]]
    if not pts:
        v.add("data", False, "no RESULT r9_lat / r9_bw lines found")
        return v
    lat = [p for p in pts if p["kind"] == "lat"]
    l0 = [p for p in lat if p["KiB"] == 16 and p["walk"] == "random"]
    bad = [p for p in l0 if not L0_BAND[0] <= p["cycles"] <= L0_BAND[1]]
    v.add("l0_chase", bool(l0) and not bad, "L0 (16 KiB) chase between %.0f and %.0f cycles in %d of %d points%s" % (
        *L0_BAND, len(l0) - len(bad), len(l0), "" if not bad else "; outside: " + ", ".join("%s %.0f" % (p["tag"], p["cycles"]) for p in bad[:4])))
    l2 = [p for p in lat if p["KiB"] in (1024, 2048, 4096) and p["walk"] == "random"]
    bad = [p for p in l2 if not L2_BAND[0] <= p["cycles"] <= L2_BAND[1]]
    v.add("l2_chase", bool(l2) and not bad, "L2 (1-4 MiB) chase between %.0f and %.0f cycles in %d of %d points%s" % (
        *L2_BAND, len(l2) - len(bad), len(l2), "" if not bad else "; outside: " + ", ".join("%s %.0fKiB %.0f" % (p["tag"], p["KiB"], p["cycles"]) for p in bad[:4])))
    wrong = [p for p in lat if not p["ok"]]
    v.add("pointers", not wrong, "%d of %d chase points have the right final pointer" % (len(lat) - len(wrong), len(lat)))
    gd = [p for p in pts if p["kind"] == "bw" and p["KiB"] >= 1048576 and p["level"] == 1408]
    if gd:
        bad = [p for p in gd if abs(p["gbps"] / BW_1408[0] - 1) > BW_1408[1]]
        v.add("gddr6_bw_1408", not bad, "GDDR6 read at the 1408 label: %s GB/s (needs %.1f +-%.1f %%)" % (
            ", ".join("%.1f" % p["gbps"] for p in gd), BW_1408[0], 100 * BW_1408[1]))
    else:
        v.add("gddr6_bw_1408", None, "no 1 GiB bandwidth job carries the 1408 label: control open")
    n, bad = label_checks(v, pts)
    if n:
        v.add("label_vs_bandwidth", not bad, "%d 1 GiB points at 1124 MHz or above: bandwidth over label within %.0f %% of %.4f GB/s per MHz%s" % (
            n, 100 * LABEL_TOL, GBPS_PER_MHZ, "" if not bad else "; off: " + ", ".join("%s label %d %.1f GB/s (ratio %.3f)" % b for b in bad[:4])))
        n2, bad2 = label_checks(v, pts, shifted=True)
        v.add("label_shift_break_caught", len(bad2) == n2, "break: every label moved one level: %d of %d points now fail the check (must be all)" % (len(bad2), n2))
    else:
        v.add("label_vs_bandwidth", None, "no 1 GiB point with a clean label of 1124 MHz or above: control open")
        v.add("label_shift_break_caught", None, "same: break not exercised")
    # findings
    cnt = {k: sum(1 for p in pts if p["why"] == k) for k in ("clean", "too short", "endpoints differ", "mixed", "unchecked")}
    ev = cnt["clean"] + cnt["too short"] + cnt["mixed"]
    out("")
    out("90 %% rule: %d points. %d were evaluated against the sampler rows inside their own window (clean %d, too short %d, mixed %d); "
        "%d not evaluated because the endpoint readings differ; %d unchecked (no sampler file or no t0_ns/t1_ns in the log). "
        "'too short' is a window with fewer than 3 rows of the 5 ms sampler: it says nothing about the clock inside it." % (
            len(pts), ev, cnt["clean"], cnt["too short"], cnt["mixed"], cnt["endpoints differ"], cnt["unchecked"]))
    out("job levels (5 ms sampler inside each job's window):")
    for j in jobs:
        if j["hist"]:
            out("  %-12s %s  modal share %.2f" % (j["label"], " ".join("%d:%d" % kv for kv in sorted(j["hist"].items())), j["share"]))
    out("")
    out("latency, median over points, by job state and label: a clean clock level, or the state that kept the point from having one (cycles / ns):")
    keys = sorted({(p["state"], p["walk"], p["KiB"], p["level"]) for p in lat}, key=lambda k: (k[0], k[1], k[2], str(k[3])))
    for k in keys:
        sel = [p for p in lat if (p["state"], p["walk"], p["KiB"], p["level"]) == k]
        out("  %-6s %-13s %9.0f KiB  label %-16s n=%-3d %7.1f cycles %8.2f ns" % (
            k[0], k[1], k[2], k[3], len(sel), L.median([p["cycles"] for p in sel]), L.median([p["ns"] for p in sel])))
    out("")
    out("bandwidth ('split' read), median GB/s by job state and label:")
    bkeys = sorted({(p["state"], p["KiB"], p["level"]) for p in pts if p["kind"] == "bw"}, key=lambda k: (k[0], k[1], str(k[2])))
    for k in bkeys:
        sel = [p for p in pts if p["kind"] == "bw" and (p["state"], p["KiB"], p["level"]) == k]
        out("  %-6s %9.0f KiB  label %-16s n=%-3d %7.1f GB/s" % (k[0], k[1], k[2], len(sel), L.median([p["gbps"] for p in sel])))
    out("")
    out("latency across the ENDPOINT labels, per job state (never pooled across states; the label, the job state and other "
        "conditions change together, so a ratio is an association and shows no cause):")
    for state in ("steady", "cold", "bg"):
        for kib in (32768, 4096, 1024, 524288, 1048576):
            by = {}
            for p in lat:
                if p["state"] == state and p["KiB"] == kib and p["walk"] == "random" and clean(p):
                    by.setdefault(p["level"], []).append(p)
            if len(by) >= 2:
                cyc = {k: L.median([p["cycles"] for p in x]) for k, x in by.items()}
                ns = {k: L.median([p["ns"] for p in x]) for k, x in by.items()}
                rc, rn = max(cyc.values()) / min(cyc.values()), max(ns.values()) / min(ns.values())
                out("RESULT E latency_vs_endpoint_label state=%s KiB=%d labels=%s cycles=%s ns=%s ratio_cycles=%.3f ratio_ns=%.3f %s" % (
                    state, kib, ",".join(str(k) for k in sorted(by)), ",".join("%.0f" % cyc[k] for k in sorted(by)),
                    ",".join("%.1f" % ns[k] for k in sorted(by)), rc, rn,
                    "(above 1.10: varies across the observed labels)" if max(rc, rn) > 1.10 else "(within 1.10 across the observed labels)"))
            elif by:
                out("RESULT E latency_vs_endpoint_label state=%s KiB=%d labels=%s: only %d endpoint label seen, no comparison" % (
                    state, kib, ",".join(str(k) for k in by), len(by)))
    return v


def synth(rng, l0=81.0, l2=240.0, gddr_gbps=700.3, ptr_ok=True, bw_label=1408, with_1408=True, n_cold=6, bg="alive", bg_l2=240.0,
          steady_label=1408, win_ms=60.0, sampler=True, flip_endpoints=False, drop_point=None, dup_point=None):
    """Synthetic jobs with exactly the points of EXPECT_POINTS, each with its own t0_ns/t1_ns window of win_ms and a 5 ms sampler
    that reads the point's label inside the window (sampler=False: no sampler file at all; flip_endpoints: every point's reading
    after differs from the one before; drop_point / dup_point: (family, index) the break removes, or replaces the next point by)."""
    jobs = []
    base = 1e12

    def build(label, fam, specs, default):
        """specs: list of (kind, walk, KiB, lab) in EXPECT_POINTS order; returns the parsed job."""
        lines = ["JOB_MONO_START %d" % base]
        wins = []
        specs = list(specs)
        if drop_point and drop_point[0] == fam and label.endswith(("-r1", "-1")):
            del specs[drop_point[1]]
        if dup_point and dup_point[0] == fam and label.endswith(("-r1", "-1")):
            specs[dup_point[1] + 1] = specs[dup_point[1]]
        for i, (kind, walk, kib, lab) in enumerate(specs):
            t0 = base + (i + 1) * 1e8
            t1 = t0 + win_ms * 1e6
            wins.append((t0, t1, lab))
            after = lab + 1 if flip_endpoints else lab
            if kind == "lat":
                cyc = l0 if kib == 16 else (l2 + {1024: 0, 2048: 5, 4096: 10}.get(kib, 0) if kib <= 4096 else (430.0 if kib == 32768 else 760.0))
                if label.startswith("e-bg") and kib in (1024, 2048, 4096):
                    cyc = bg_l2 + {1024: 0, 2048: 5, 4096: 10}[kib]
                if walk == "random_fresh":
                    cyc = 480.0
                lines.append("RESULT r9_lat mode=lat walk=%s KiB=%d cycles=%.1f ns=%.2f pointer=%s uclk_before=%d uclk_after=%d uclk_process_start=456 t0_ns=%d t1_ns=%d" % (
                    walk, kib, cyc * rng.uniform(0.99, 1.01), cyc / 3.0, "ok" if ptr_ok else "WRONG", lab, after, t0, t1))
            else:
                g = 1500.0 if kib == 32768 else (gddr_gbps * rng.uniform(0.997, 1.003) if kib >= 1048576 else 5000.0)
                lines.append("RESULT mem_stream size split S_KiB %.1f footprint_KiB %.1f passes 2 GBps %.1f min %.1f max %.1f us 1.00" % (kib, kib, g, g, g))
                lines.append("RESULT r9_bw KiB=%d uclk_before=%d uclk_after=%d uclk_process_start=456 t0_ns=%d t1_ns=%d" % (kib, lab, after, t0, t1))
        end = base + (len(specs) + 2) * 1e8
        lines.append("JOB_MONO_END %d rc=0" % end)
        samp = None
        if sampler:
            ts = [base + i * 5e6 for i in range(int((end - base) / 5e6))]
            us = []
            for t in ts:
                lab = default
                for t0, t1, l_ in wins:
                    if t0 <= t <= t1:
                        lab = l_
                us.append(float(lab))
            samp = dict(mono_ns=ts, uclk_cur_MHz=us)
        return lines, samp

    def point_specs(fam, lab_of):
        return [(k, w, kib, lab_of(k, kib)) for k, w, kib in EXPECT_POINTS[fam]]

    for r in (1, 2, 3):
        lines, samp = build("e-lat-r%d" % r, "e-lat", point_specs("e-lat", lambda k, kib: steady_label), steady_label)
        jobs.append(parse_job("e-lat-r%d" % r, "\n".join(lines), samp))
        lines, samp = build("e-bw-r%d" % r, "e-bw", point_specs("e-bw", lambda k, kib: bw_label if kib >= 1048576 else 1408), 1408)
        jobs.append(parse_job("e-bw-r%d" % r, "\n".join(lines), samp))
    for n in range(n_cold):
        lab = (456, 772, 1124)[n % 3]
        lines, samp = build("e-cold-%d" % (n + 1), "e-cold", point_specs("e-cold", lambda k, kib, lab=lab: lab), lab)
        jobs.append(parse_job("e-cold-%d" % (n + 1), "\n".join(lines), samp))
    if bg:  # three background jobs at the 875 label; bg: "alive" (the driver's proof), "dead" (a DEAD line), "missing" (no proof lines)
        for r in (1, 2, 3):
            lines, samp = build("e-bg-r%d" % r, "e-bg", point_specs("e-bg", lambda k, kib: 875), 875)
            extra = []
            if bg == "alive":
                extra = ["INFO bg alive after start launches=100", "INFO bg alive after r9_lat launches=200", "INFO bg alive after r9_bw launches=300"]
            elif bg == "dead":
                extra = ["INFO bg DEAD after r9_lat alive=0 launches=100 previous=100"]
            lines = lines[:1] + extra + lines[1:]
            jobs.append(parse_job("e-bg-r%d" % r, "\n".join(lines), samp))
    if not with_1408:
        for j in jobs:
            for p in j["points"]:
                if p["level"] == 1408:
                    p["level"] = "mixed"
    return jobs


def selftest():
    def case(**kw):
        return lambda: analyse(synth(random.Random(5), **kw), out=lambda *_: None)

    cases = [
        ("base", True, case()),
        ("no 1408 job: control open, not a failure", True, case(with_1408=False)),
        ("break l0_reads_120", False, case(l0=120.0), "l0_chase"),
        ("break l2_reads_330", False, case(l2=330.0), "l2_chase"),
        ("break gddr6_650", False, case(gddr_gbps=650.0), "gddr6_bw_1408"),
        ("break label_one_level_too_low", False, case(gddr_gbps=565.0), "label_vs_bandwidth"),
        ("break wrong_pointer", False, case(ptr_ok=False), "pointers"),
        ("break no_data", False, lambda: analyse([], out=lambda *_: None), "data"),
        ("break background loop dead (a DEAD line)", False, case(bg="dead"), "bg_running"),
        ("break background loop with no proof lines (the first version's output)", False, case(bg="missing"), "bg_running"),
    ]
    ok = L.run_selftest("E", cases)

    def lines_of(**kw):
        buf = []
        analyse(synth(random.Random(5), **kw), out=buf.append)
        return buf

    import contextlib
    import io

    def quiet(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()

    def check(name, good):
        nonlocal ok
        print("SELFTEST E %s: %s" % (name, "ok" if good else "WRONG"))
        ok &= bool(good)

    # no pooling across job states: steady jobs have one label (456), the background jobs another (875) and 25 % higher L2;
    # the first version pooled both by label and printed a ratio of 1.25 with "depends on the memory clock"
    out = quiet(lambda: lines_of(steady_label=456, bg_l2=300.0))
    cmp_lines = [x for x in out if x.startswith("RESULT E latency_vs") and ("state=steady" in x or "state=bg" in x)]
    check("steady 456 against bg 875 is not pooled: no ratio line for either state, only single-label lines",
          cmp_lines and not any("ratio_cycles" in x for x in cmp_lines) and any("state=bg" in x and "only 1 endpoint label" in x for x in cmp_lines))
    # the tag is conditional: two labels within 5 % say "within 1.10"; two labels 30 % apart say "varies across the observed labels"
    rng = random.Random(2)
    jobs = synth(rng, steady_label=456, bg=None)
    for j in synth(random.Random(3), steady_label=1408, bg=None):
        j["label"] += "-b"
        for p in j["points"]:
            p["tag"] = j["label"]
            if p["kind"] == "lat":
                p["cycles"] *= 1.05
        jobs.append(j)
    out = quiet(lambda: (lambda b: (analyse(jobs, out=b.append), b)[1])([]))
    t1 = [x for x in out if x.startswith("RESULT E latency_vs_endpoint_label state=steady KiB=1024")]
    check("two labels 5 % apart: tag says within 1.10, not the old unconditional claim",
          t1 and "ratio_cycles" in t1[0] and "within 1.10" in t1[0] and "depends" not in t1[0] and "above 1.10" not in t1[0])
    for j in jobs:
        if j["label"].endswith("-b"):
            for p in j["points"]:
                if p["kind"] == "lat":
                    p["cycles"] *= 1.25
    out = quiet(lambda: (lambda b: (analyse(jobs, out=b.append), b)[1])([]))
    t2 = [x for x in out if x.startswith("RESULT E latency_vs_endpoint_label state=steady KiB=1024")]
    check("two labels 31 % apart: tag says varies across the observed labels", t2 and "above 1.10: varies across the observed labels" in t2[0])
    full = synth(random.Random(5))
    expect = {"e-lat": 3, "e-bw": 3, "e-cold": 6, "e-bg": 3}
    miss = [j for j in full if j["label"] != "e-bw-r2"]
    empty = [dict(j, points=[]) if j["label"] == "e-cold-3" else j for j in full]
    for nm, jb, want in (("every expected job present: complete passes", full, True), ("break: one job (e-bw-r2) missing: complete fails", miss, False),
                         ("break: one job (e-cold-3) with no points: complete fails", empty, False)):
        vv = quiet(lambda jb=jb: analyse(jb, out=lambda *_: None, expect=expect))
        check(nm, any(r[0] == "complete" and bool(r[1]) == want for r in vv.rows))
    # the 90 % rule: a point labelled 1408 at both ends whose own window reads 1408 in only half of its sampler rows is mixed
    mono = [1e12 + i * 5e6 for i in range(200)]
    txt = "JOB_MONO_START 1000000000000\nRESULT r9_lat mode=lat walk=random KiB=1024 cycles=240.0 ns=80 pointer=ok uclk_before=1408 uclk_after=1408 t0_ns=%d t1_ns=%d\n" % (mono[0], mono[199])
    half = dict(mono_ns=mono, uclk_cur_MHz=[1408.0] * 100 + [1124.0] * 100)
    full = dict(mono_ns=mono, uclk_cur_MHz=[1408.0] * 200)
    check("a window with half its samples at 1408 is mixed, a clean one keeps 1408",
          parse_job("e-lat-x", txt, half)["points"][0]["level"] == "mixed" and parse_job("e-lat-x", txt, full)["points"][0]["level"] == 1408)

    def whys(**kw):
        c = {}
        for j in synth(random.Random(5), **kw):
            for p in j["points"]:
                c[p["why"]] = c.get(p["why"], 0) + 1
        return c

    base = whys()
    npts = sum(base.values())
    check("base: all %d points have a clean label (windows of 60 ms hold 12 sampler rows)" % npts, set(base) == {"clean"})
    shortw = whys(win_ms=4.0)
    check("windows of 4 ms (at most 1 row of the 5 ms sampler) are 'too short', not 'mixed' and not clean",
          set(shortw) == {"too short"})
    check("break: a window that really changes clock inside is 'mixed', not 'too short'",
          parse_job("e-lat-x", txt, half)["points"][0]["why"] == "mixed")
    check("endpoints that differ are 'endpoints differ' and are not evaluated against the window", set(whys(flip_endpoints=True)) == {"endpoints differ"})
    check("no sampler file: every point is 'unchecked' (the endpoint label alone is never taken as clean)", set(whys(sampler=False)) == {"unchecked"})
    vv = quiet(lambda: analyse(synth(random.Random(5), sampler=False), out=lambda *_: None))
    check("no sampler file: the 1408 bandwidth control stays open (SKIP), it is not passed on endpoint labels",
          any(r[0] == "gddr6_bw_1408" and r[1] is None for r in vv.rows))
    vv = quiet(lambda: analyse(synth(random.Random(5), win_ms=4.0), out=lambda *_: None))
    check("windows too short for any label: the 1408 bandwidth control stays open as well",
          any(r[0] == "gddr6_bw_1408" and r[1] is None for r in vv.rows))
    lines = []
    quiet(lambda: analyse(synth(random.Random(5), win_ms=4.0), out=lines.append))
    check("the report counts the states and does not call short windows mixed", any(
        x.startswith("90 % rule: ") and ("clean 0, too short %d, mixed 0" % npts) in x for x in lines))
    for nm, kw in (("a duplicate replaces another point (e-lat-r1 point 2 twice)", dict(dup_point=("e-lat", 2))),
                   ("a point is missing from one job (e-cold-1 loses its second)", dict(drop_point=("e-cold", 1)))):
        vv = quiet(lambda kw=kw: analyse(synth(random.Random(5), **kw), out=lambda *_: None, expect={"e-lat": 3, "e-bw": 3, "e-cold": 6, "e-bg": 3}))
        check("break: " + nm + ": complete fails", any(r[0] == "complete" and r[1] is False for r in vv.rows))
    reduced = [dict(j, points=j["points"][:1]) for j in synth(random.Random(5))]
    vv = quiet(lambda: analyse(reduced, out=lambda *_: None, expect={"e-lat": 3, "e-bw": 3, "e-cold": 6, "e-bg": 3}))
    check("break: every job reduced to its first point (Codex round 2): complete fails", any(r[0] == "complete" and r[1] is False for r in vv.rows))
    return ok


def read_sampler(text):
    cols, data = None, {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if cols is None:
            cols = f
            data = {c: [] for c in cols}
            continue
        if len(f) != len(cols):
            continue
        for c, x in zip(cols, f):
            try:
                data[c].append(float(x) if not x.startswith("0x") else float(int(x, 16)))
            except ValueError:
                data[c].append(float("nan"))
    return data


def main(argv):
    if argv and argv[0] == "--selftest":
        return 0 if selftest() else 1
    expect = None
    if argv[:1] == ["--expect"] and len(argv) > 1:
        try:
            expect = {k: int(n) for k, n in (kv.split("=") for kv in argv[1].split(","))}
        except ValueError:
            print(__doc__)
            return 2
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
            text = ""  # a missing log: the job has no points, and `complete` says so
        samp = None
        if len(parts) > 2 and os.path.exists(parts[2]):
            with open(parts[2], errors="replace") as f:
                samp = read_sampler(f.read())
        jobs.append(parse_job(parts[0], text, samp))
    v = analyse(jobs, expect=expect)
    return 0 if v.summary() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
