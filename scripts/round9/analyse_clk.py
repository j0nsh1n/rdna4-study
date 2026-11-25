#!/usr/bin/env python3
"""Item D analysis: the core clock the shader sees at dispatches of a fixed length issued at a fixed cadence and waited on.
Reads the per-dispatch CSV of vkr9 clk / r9_clk and the sysfs sampler CSV beside it (r9_sampler samples.csv), prints the clock
per cell, the spread between the runs of one cell, and PASS / FAIL per control.

  python3 -I analyse_clk.py LABEL,CLKCSV[,SAMPLERCSV]...
  python3 -I analyse_clk.py --selftest
The clock of a dispatch is (cycles / REALTIME ticks) x 100 MHz from wave 0 of group 0, over the whole dispatch, and over its
first and last quarter. Controls: calibration (cycles over REALTIME against cycles over the queue's or the event's own
timer, 3 x 30 ms dispatches, 1.5 %); waves agree (1.03); dispatch length (3 %); cadence (1 %); at least 90 % of dispatches valid;
the probe's own verdicts (its cadence verdict: submissions late p95 under 200 us and the largest under 1000 us, read from the probe's
output file beside the CSV; a FAIL there fails the cell, and so does a CSV with no such file or no cadence verdict in it). The
validity control `valid_<cell>` also needs the call records to be exactly 0..n-1 (no repetition lost, doubled or replaced) and,
for a cell label of the form d-vk-<ms>-r<N> or d-hip-<ms>-r<N>, the CSV's own api and planned length to be the label's. The realtime-shift check (REALTIME x 1.05 must move the clock by 1/1.05)
only tests this script's own arithmetic (the pipeline), not the probe or the clock, so it is a self-test and not a control of a run. A cell label ending in -r<N> belongs
to the group without it; groups of two or more runs get a run-to-run spread."""
import bisect
import importlib.util
import os
import random
import re
import sys

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("r9lib", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r9lib.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

CALIB_TOL = 0.015
WAVE_TOL = 1.03
LEN_TOL = 0.03
CAD_TOL = 0.01
VALID_MIN = 0.90
SHIFT_TOL = 0.005
LABEL_RE = re.compile(r"^d-(vk|hip)-([0-9.]+)-r\d+$")


def identity_problem(label, meta, calls):
    """Why the records cannot be the cell the label names, or None: call ids not exactly 0..n-1, or (for a D label) another api or length."""
    ids = [int(r.get("idx", -1)) for r in calls]
    if ids != list(range(len(ids))):
        dup = len(ids) - len(set(ids))
        return "call ids are not 0..%d (%d duplicated, %d distinct)" % (len(ids) - 1, dup, len(set(ids)))
    m = LABEL_RE.match(label)
    if m:
        api = {"vk": "vulkan", "hip": "hip"}[m.group(1)]
        if meta.get("api") != api or abs(float(meta.get("ms", "nan")) - float(m.group(2))) > 1e-6:
            return "the CSV says api %s, ms %s; the label says %s, %s ms" % (meta.get("api"), meta.get("ms"), api, m.group(2))
    return None


def read_clk_csv(text):
    """-> (meta dict, list of row dicts with numeric fields)."""
    meta, rows, head = {}, [], None
    for line in text.splitlines():
        if line.startswith("#"):
            meta.update(L.kvs(line[1:]))
        elif line.startswith("kind,"):
            head = line.split(",")
        elif line.strip() and head:
            f = line.split(",")
            if len(f) != len(head):
                continue
            d = {"kind": f[0]}
            for k, x in zip(head[1:], f[1:]):
                d[k] = float(x)
            rows.append(d)
    return meta, rows


def probe_verdicts(csv_path):
    """The VERDICT lines of the probe's own output file beside a call CSV (vkr9: <tag>-clk-cell.csv -> <tag>.out; r9_clk: <tag>.csv
    -> <tag>.out), or None when there is no such file."""
    p = csv_path[:-len("-clk-cell.csv")] + ".out" if csv_path.endswith("-clk-cell.csv") else (csv_path[:-4] + ".out" if csv_path.endswith(".csv") else "")
    try:
        with open(p, errors="replace") as f:
            return [ln.strip() for ln in f if ln.startswith("VERDICT ")]
    except OSError:
        return None


def read_sampler(text):
    """-> dict of lists keyed by column name (mono_ns, gfxclk_cur_MHz, ...). Also accepts vkgame's *-sysfs.csv."""
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
    if "t_mono_ns" in data:  # vkgame sampler's names
        data["mono_ns"] = data["t_mono_ns"]
        data["gfxclk_cur_MHz"] = data.get("cur_gfxclk_MHz", [])
        data["sclk_star_MHz"] = data.get("sclk_MHz", [])
        data["indep_throttle_status"] = data.get("indep_throttle_status", [])
        data["hotspot_C"] = data.get("hotspot_C", [])
    return data


def pipeline(rows, rt_scale=1.0):
    """Per call row: clocks in MHz (whole, first quarter, last quarter); None where the record is unusable.
    REALTIME ticks are multiplied by rt_scale (1.0 = as recorded; the break uses 1.05)."""
    out = []
    for r in rows:
        if r["kind"] != "call" or r["nrec"] < 2 or r["rt1"] <= r["rt0"]:
            out.append(None)
            continue
        whole = (r["cyc1"] - r["cyc0"]) / ((r["rt1"] - r["rt0"]) * rt_scale) * 100.0
        q1 = (r["cycq1"] - r["cyc0"]) / ((r["rtq1"] - r["rt0"]) * rt_scale) * 100.0 if r["rtq1"] > r["rt0"] else float("nan")
        q4 = (r["cyc1"] - r["cycq3"]) / ((r["rt1"] - r["rtq3"]) * rt_scale) * 100.0 if r["rt1"] > r["rtq3"] else float("nan")
        out.append((whole, q1, q4))
    return out


def pipeline_ignores_scale(rows, rt_scale=1.0):
    """The break: a pipeline that does not respond to the realtime shift."""
    return pipeline(rows, 1.0)


def analyse_cell(label, meta, rows, samp, v, pipe=pipeline, out=print, probe=None):
    ms, cad = float(meta.get("ms", 0)), float(meta.get("cad", 0))
    tsp = float(meta.get("tsp_ns", 1.0))
    calls = [r for r in rows if r["kind"] == "call"]
    cal = [r for r in rows if r["kind"] == "calib"]
    clk = pipe(calls)
    good = [(i, c) for i, c in enumerate(clk) if c is not None]
    ident = identity_problem(label, meta, calls)
    v.add("valid_" + label, len(calls) >= 50 and len(good) >= VALID_MIN * len(calls) and ident is None,
          "%s: %d dispatches, %d with usable records (need 90 %%, at least 50 dispatches); records %s" % (
              label, len(calls), len(good), "are the cell's own, ids 0..%d" % (len(calls) - 1) if ident is None else "NOT the cell's own: " + ident))
    if not good:
        return None
    whole = [c[0] for _, c in good]
    q1 = [c[1] for _, c in good if c[1] == c[1]]
    q4 = [c[2] for _, c in good if c[2] == c[2]]
    # calibration
    if cal:
        last = cal[-1]
        rtc = (last["cyc1"] - last["cyc0"]) / (last["rt1"] - last["rt0"]) * 100.0
        ns = (last["ts_bot"] - last["ts_top"]) * tsp
        tsc = (last["cyc1"] - last["cyc0"]) / ns * 1000.0 if ns > 0 else float("nan")
        v.add("calib_" + label, abs(rtc / tsc - 1) <= CALIB_TOL,
              "%s: 30 ms dispatch: %.0f MHz from REALTIME, %.0f MHz from the %s timer (%.2f %% apart, limit %.1f %%)" % (
                  label, rtc, tsc, "HIP event" if tsp == 1.0 else "queue timestamp", 100 * abs(rtc / tsc - 1), 100 * CALIB_TOL))
    else:
        v.add("calib_" + label, None, "%s: no calibration dispatches in the file" % label)
    ratios = [r["wave_mhz_max"] / r["wave_mhz_min"] for r in calls if r["wave_mhz_min"] > 0]
    wr = L.median(ratios) if ratios else float("nan")
    v.add("waves_" + label, wr <= WAVE_TOL, "%s: highest over lowest wave clock, median over dispatches %.4f (limit %.2f)" % (
        label, wr, WAVE_TOL))
    lens = [(r["rt1"] - r["rt0"]) / 1e5 for r in calls if r["rt1"] > r["rt0"]]
    ml = L.median(lens)
    v.add("length_" + label, abs(ml - ms) <= max(LEN_TOL * ms, 0.03),
          "%s: median dispatch length %.3f ms against planned %.3f ms" % (label, ml, ms))
    ts = [r["t_submit_mono_ns"] for r in calls]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    mg = L.median(gaps) / 1e6 if gaps else float("nan")
    v.add("cadence_" + label, abs(mg / cad - 1) <= CAD_TOL, "%s: median submission spacing %.3f ms against %.3f ms" % (label, mg, cad))
    # the probe's own verdicts (cadence, thermal), from its output file; a FAIL there is a FAIL of the cell
    if probe is not None:
        cad_lines = [x for x in probe if " cadence:" in x]
        bad = [x for x in probe if ": FAIL" in x]
        v.add("probe_" + label, bool(cad_lines) and not bad,
              "%s: the probe's own verdicts: %s" % (label, "; ".join(x.replace("VERDICT ", "", 1)[:150] for x in (bad or cad_lines)) or "no cadence verdict found in the probe output"))
    else:
        v.add("probe_" + label, False, "%s: no probe output file next to the CSV: the probe's cadence verdict was not read (a usable CSV without it fails)" % label)
    # sysfs beside it
    sysf = ""
    if samp and samp.get("mono_ns"):
        t = samp["mono_ns"]
        cur, star, thr = [], [], 0
        for r in calls:
            mid = 0.5 * (r["t_submit_mono_ns"] + r["t_done_mono_ns"])
            i = bisect.bisect_left(t, mid)
            i = min(max(i, 0), len(t) - 1)
            if i > 0 and abs(t[i - 1] - mid) < abs(t[i] - mid):
                i -= 1
            if abs(t[i] - mid) < 20e6:
                cur.append(samp["gfxclk_cur_MHz"][i])
                star.append(samp["sclk_star_MHz"][i])
        thrs = samp.get("indep_throttle_status") or []
        thr = sum(1 for x in thrs if x == x and x != 0)
        hot = max([x for x in samp.get("hotspot_C", []) if x == x] or [float("nan")])
        if cur:
            mc = L.median(cur)
            sysf = " sysfs_gfxclk_med=%.0f sysfs_sclk_star_med=%.0f in_kernel_over_sysfs=%.3f throttle_samples_nonzero=%d hotspot_max=%.0f" % (
                mc, L.median(star), L.median(whole) / mc if mc > 0 else float("nan"), thr, hot)
    stepfrac = sum(1 for a, b in zip(q1, q4) if abs(b - a) / L.median(whole) > 0.05) / max(1, min(len(q1), len(q4)))
    out("RESULT D %s n=%d clock_MHz median=%.0f p5=%.0f p95=%.0f min=%.0f max=%.0f spread_p95_p5=%.1f%% first=%.0f "
        "first_quarter_med=%.0f last_quarter_med=%.0f step_within_dispatch_frac=%.2f%s" % (
            label, len(whole), L.median(whole), L.pct(whole, 5), L.pct(whole, 95), min(whole), max(whole),
            100 * (L.pct(whole, 95) - L.pct(whole, 5)) / L.median(whole), whole[0], L.median(q1), L.median(q4), stepfrac, sysf))
    return L.median(whole)


def analyse(cells, pipe=pipeline, out=print):
    """cells: label -> (meta, rows, sampler[, probe verdict lines])."""
    v = L.Verdicts("D")
    meds = {}
    for label, cell in cells.items():
        meta, rows, samp = cell[:3]
        if not rows:  # the file was missing or empty: the driver lists every cell it ran, so this is a failed cell
            v.add("complete_" + label, False, "%s: no call records (missing or empty CSV)" % label)
            continue
        meds[label] = analyse_cell(label, meta, rows, samp, v, pipe, out, cell[3] if len(cell) > 3 else None)
    groups = {}
    for label, m in meds.items():
        if m is not None:
            groups.setdefault(re.sub(r"-r\d+$", "", label), []).append(m)
    for g, ms in sorted(groups.items()):
        if len(ms) >= 2:
            sp = L.spread(ms)
            out("RESULT D group %s runs=%d run_medians_MHz=%s spread=%.1f%% (max - min over median of the run medians; a clock spread, "
                "not comparable with a timing spread)" % (g, len(ms), " ".join("%.0f" % x for x in ms), 100 * sp))
    return v


def synth_probe(cad_late_us=0.3, max_late_us=5.0):
    return ["VERDICT vkgame cadence: %s cell: submissions late against the cadence p95 %.1f us (limit 200), largest %.1f us (limit 1000)" % (
        "FAIL" if cad_late_us > 200 or max_late_us > 1000 else "PASS", cad_late_us, max_late_us)]


def synth_cell(rng, ms=1.0, cad=16.7, n=300, clk=2400.0, calib_ts_off=1.0, wave_ratio=1.01, len_mult=1.0, cad_mult=1.0,
               drop=0.0, api="vulkan"):
    tsp = 10.0 if api == "vulkan" else 1.0
    meta = {"api": api, "ms": str(ms), "cad": str(cad), "tsp_ns": str(tsp)}
    rows = []
    for i in range(n):
        c = clk * rng.uniform(0.985, 1.015) * (0.8 if i == 0 else 1.0)
        dt = int(ms * 1e5 * len_mult)
        rt0 = 5_000_000_000 + int(i * cad * 1e5)
        bad = rng.random() < drop
        rows.append(dict(kind="call", idx=i, t_submit_mono_ns=1e12 + i * cad * 1e6 * cad_mult, t_done_mono_ns=1e12 + i * cad * 1e6 * cad_mult + 1.5e6,
                         ts_top=0.0, ts_bot=0.0, rt0=rt0, cyc0=1e9, rt1=rt0 + (0 if bad else dt), cyc1=1e9 + c * dt / 100.0,
                         rtq1=rt0 + dt // 4, cycq1=1e9 + c * (dt // 4) / 100.0, rtq3=rt0 + 3 * dt // 4,
                         cycq3=1e9 + c * (3 * dt // 4) / 100.0, nrec=0 if bad else 120, wave_mhz_min=c / wave_ratio, wave_mhz_max=c))
    for i in range(3):
        dt = 3_000_000
        c = clk
        rows.append(dict(kind="calib", idx=i, t_submit_mono_ns=2e12, t_done_mono_ns=2e12 + 3e7, ts_top=100.0,
                         ts_bot=100.0 + dt * 10.0 / tsp * calib_ts_off,
                         rt0=9_000_000_000, cyc0=1e9, rt1=9_000_000_000 + dt, cyc1=1e9 + c * dt / 100.0, rtq1=0, cycq1=0, rtq3=0,
                         cycq3=0, nrec=300, wave_mhz_min=c, wave_mhz_max=c))
    return meta, rows


def synth_sampler(rng, clk=2300.0):
    t = [1e12 + i * 5e6 for i in range(0, 4000)]
    return {"mono_ns": t, "gfxclk_cur_MHz": [clk * rng.uniform(0.98, 1.02) for _ in t], "sclk_star_MHz": [1794.0] * len(t),
            "indep_throttle_status": [0.0] * len(t), "hotspot_C": [45.0] * len(t)}


def selftest():
    rng = random.Random(3)

    def case(pipe=pipeline, probe=None, **kw):
        def run():
            cells = {"c-r%d" % k: synth_cell(rng, clk=2400 * (1 + 0.01 * k), **kw) + (synth_sampler(rng), probe if probe is not None else synth_probe())
                     for k in (1, 2)}
            return analyse(cells, pipe=pipe, out=lambda *_: None)
        return run

    cases = [
        ("base", True, case()),
        ("break probe_max_lateness_1169us (the probe's own cadence FAIL)", False, case(probe=synth_probe(0.2, 1169.4)), "probe_"),
        ("break probe_output_without_a_cadence_verdict", False, case(probe=[]), "probe_"),
        ("break calibration_timer_5pct_off", False, case(calib_ts_off=1.05), "calib_"),
        ("break waves_disagree_10pct", False, case(wave_ratio=1.10), "waves_"),
        ("break dispatch_1.5x_long", False, case(len_mult=1.5), "length_"),
        ("break cadence_8pct_slow", False, case(cad_mult=1.08), "cadence_"),
        ("break half_the_records_lost", False, case(drop=0.5), "valid_"),
        ("hip layout", True, case(api="hip")),
        ("break a cell missing (no CSV)", False, lambda: analyse({"c-r1": ({}, [], None, None)}, out=lambda *_: None), "complete_"),
    ]
    # identity and sidecar breaks, on cells labelled as the driver labels them
    def named(label, mutate=None, probe="ok", **kw):
        def run():
            meta, rows = synth_cell(random.Random(4), **kw)
            if mutate:
                meta, rows = mutate(meta, rows)
            return analyse({label: (meta, rows, synth_sampler(random.Random(4)), synth_probe() if probe == "ok" else None)}, out=lambda *_: None)
        return run

    def dup_ids(meta, rows):
        calls = [r for r in rows if r["kind"] == "call"]
        calls[10] = dict(calls[9])  # call 10 replaced by a second copy of call 9
        return meta, calls + [r for r in rows if r["kind"] != "call"]

    cases2 = [
        ("a labelled cell with its own api and length passes", True, named("d-vk-1.0-r1", ms=1.0)),
        ("break: the label says 0.3 ms, the CSV 1.0 ms", False, named("d-vk-0.3-r1", ms=1.0), "valid_"),
        ("break: the label says hip, the CSV vulkan", False, named("d-hip-1.0-r1", ms=1.0), "valid_"),
        ("break: call 10 replaced by a copy of call 9 (same count, a repetition lost)", False, named("d-vk-1.0-r1", mutate=dup_ids, ms=1.0), "valid_"),
        ("break: a usable CSV with no probe output file", False, named("d-vk-1.0-r1", probe=None, ms=1.0), "probe_"),
    ]
    ok = L.run_selftest("D", cases) and L.run_selftest("D identity", cases2)
    # the realtime-shift check: only this script's arithmetic, so a self-test and not a control of a run
    meta, rows = synth_cell(random.Random(3))
    calls = [r for r in rows if r["kind"] == "call"]
    for nm, pf, want in (("pipeline follows REALTIME x 1.05", pipeline, True), ("break: a pipeline that ignores the scale", pipeline_ignores_scale, False)):
        base = L.median([c[0] for c in pf(calls) if c is not None])
        shifted = L.median([c[0] for c in pf(calls, 1.05) if c is not None])
        got = abs(shifted / base - 1 / 1.05) <= SHIFT_TOL
        print("SELFTEST D rt_shift: %s: %s" % (nm, "ok" if got == want else "WRONG"))
        ok &= got == want
    return ok


def main(argv):
    if argv and argv[0] == "--selftest":
        return 0 if selftest() else 1
    if not argv:
        print(__doc__)
        return 2
    cells = {}
    for a in argv:
        parts = a.split(",")
        label, path = parts[0], parts[1]
        try:
            with open(path, errors="replace") as f:
                meta, rows = read_clk_csv(f.read())
        except OSError:
            meta, rows = {}, []
        samp = None
        if len(parts) > 2 and os.path.exists(parts[2]):
            with open(parts[2], errors="replace") as f:
                samp = read_sampler(f.read())
        cells[label] = (meta, rows, samp, probe_verdicts(path))
    v = analyse(cells)
    return 0 if v.summary() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
