#!/usr/bin/env python3
"""Item G1 analysis: is the in-kernel core clock steady under profile_standard, and how does it compare with the same runs
under auto? Each run is one CSV of the 1.0 ms call-then-wait cell (vkr9 clk); a run's value is the median clock over its
dispatches after the first 20 (the ramp from idle).

  python3 -I analyse_stable.py --before auto --after auto --in-force profile_standard --standard CSV... --auto CSV...
  python3 -I analyse_stable.py --selftest
Controls: the level read after equals the level read before and is auto; the level read inside the job was profile_standard;
each run has at least 90 % usable dispatches and its call records are exactly 0..n-1 of the 1.0 ms / 16.7 ms cell (a lost or
doubled call, or another cell's file, fails `valid_`); the probe's own cadence verdict (next to each CSV) exists and is not FAIL
(a run with call records and no such verdict fails; a run that never started has no records and fails `valid_` alone); the standard runs' medians lie within +-0.3 % (half of max - min over median)."""
import importlib.util
import os
import random
import sys

sys.dont_write_bytecode = True
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("r9lib", os.path.join(_here, "r9lib.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
_spec2 = importlib.util.spec_from_file_location("analyse_clk", os.path.join(_here, "analyse_clk.py"))
CLK = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(CLK)

STEADY = 0.003
SKIP_FIRST = 20


def run_value(meta, rows):
    """-> (median clock MHz after the first 20 dispatches, usable fraction, dispatches)."""
    calls = [r for r in rows if r["kind"] == "call"]
    clk = CLK.pipeline(calls)
    use = [c[0] for c in clk[SKIP_FIRST:] if c is not None]
    frac = sum(1 for c in clk if c is not None) / len(calls) if calls else 0.0
    return (L.median(use) if use else float("nan")), frac, len(calls)


def analyse(before, after, in_force, std, auto, out=print):
    """std, auto: lists of (meta, rows[, probe verdict lines]) per run."""
    v = L.Verdicts("G1")
    v.add("level_restored", before == after and after == "auto",
          "level before the script %r, after %r (both must be auto)" % (before, after))
    v.add("level_in_force", in_force == "profile_standard",
          "level read inside the job after the write: %r (must be profile_standard)" % in_force)
    series = {}
    for name, runs in (("standard", std), ("auto", auto)):
        vals = []
        for i, run in enumerate(runs):
            meta, rows = run[:2]
            probe = run[2] if len(run) > 2 else None
            if probe is not None:
                bad = [x for x in probe if ": FAIL" in x]
                v.add("probe_%s_%d" % (name, i + 1), bool([x for x in probe if " cadence:" in x]) and not bad,
                      "%s run %d: the probe's own verdicts: %s" % (name, i + 1, "; ".join(x.replace("VERDICT ", "", 1)[:150] for x in (bad or probe)) or "no cadence verdict found in the probe output"))
            elif rows:
                v.add("probe_%s_%d" % (name, i + 1), False, "%s run %d: call records but no probe output file: the probe's cadence verdict was not read" % (name, i + 1))
            m, frac, n = run_value(meta, rows)
            vals.append(m)
            calls = [r for r in rows if r["kind"] == "call"]
            ids_ok = [int(r.get("idx", -1)) for r in calls] == list(range(len(calls)))
            cell_ok = not rows or (meta.get("ms") == "1.000" and meta.get("cad") == "16.700")
            v.add("valid_%s_%d" % (name, i + 1), frac >= 0.9 and n >= 50 and ids_ok and cell_ok,
                  "%s run %d: %d dispatches, %.0f %% usable%s%s" % (name, i + 1, n, 100 * frac, "" if ids_ok else "; call ids are not 0..n-1",
                                                                    "" if cell_ok else "; not the 1.0 ms / 16.7 ms cell (CSV says ms %s, cad %s)" % (meta.get("ms"), meta.get("cad"))))
        series[name] = vals
    s = [x for x in series["standard"] if x == x]
    a = [x for x in series["auto"] if x == x]
    if len(s) >= 2:
        half = (max(s) - min(s)) / 2 / L.median(s)
        v.add("steady_standard", half <= STEADY, "profile_standard: %d run medians %s MHz, +-%.2f %% (limit +-%.1f %%)" % (
            len(s), " ".join("%.0f" % x for x in s), 100 * half, 100 * STEADY))
    else:
        v.add("steady_standard", False, "fewer than two usable standard runs")
    if len(a) >= 2:
        half = (max(a) - min(a)) / 2 / L.median(a)
        out("RESULT G1 auto: %d run medians %s MHz, +-%.2f %%, spread (max-min)/median %.1f %%" % (
            len(a), " ".join("%.0f" % x for x in a), 100 * half, 100 * L.spread(a)))
    if s and a:
        out("RESULT G1 standard median %.0f MHz, auto median %.0f MHz, ratio %.3f; standard +-%.2f %% against auto +-%.2f %%" % (
            L.median(s), L.median(a), L.median(s) / L.median(a), 100 * (max(s) - min(s)) / 2 / L.median(s),
            100 * (max(a) - min(a)) / 2 / L.median(a)))
    return v


def synth_run(rng, clk, n=250):
    meta = {"ms": "1.000", "cad": "16.700", "tsp_ns": "10.0"}
    rows = []
    for i in range(n):
        c = clk * rng.uniform(0.995, 1.005) * (0.8 if i < 3 else 1.0)
        dt = 100000
        rt0 = 5_000_000_000 + i * 1670000
        rows.append(dict(kind="call", idx=i, t_submit_mono_ns=1e12 + i * 16.7e6, t_done_mono_ns=1e12 + i * 16.7e6 + 1.5e6, ts_top=0.0, ts_bot=0.0,
                         rt0=rt0, cyc0=1e9, rt1=rt0 + dt, cyc1=1e9 + c * dt / 100.0, rtq1=rt0 + dt // 4, cycq1=1e9 + c * (dt // 4) / 100.0,
                         rtq3=rt0 + 3 * dt // 4, cycq3=1e9 + c * (3 * dt // 4) / 100.0, nrec=100, wave_mhz_min=c, wave_mhz_max=c))
    return meta, rows


def selftest():
    def case(swing=0.0015, before="auto", after="auto", in_force="profile_standard", std_clk=1650.0, late=5.0):
        def run():
            rng = random.Random(11)
            pr = CLK.synth_probe(0.2, late)
            std = [synth_run(rng, std_clk * (1 + swing * rng.uniform(-1, 1))) + (pr,) for _ in range(10)]
            auto = [synth_run(rng, 2300.0 * (1 + 0.08 * rng.uniform(-1, 1))) + (pr,) for _ in range(10)]
            return analyse(before, after, in_force, std, auto, out=lambda *_: None)
        return run

    cases = [
        ("base", True, case()),
        ("break ten_percent_swing_must_fail_steadiness", False, case(swing=0.10), "steady_standard"),
        ("break level_not_restored", False, case(after="profile_standard"), "level_restored"),
        ("break level_not_in_force", False, case(in_force="auto"), "level_in_force"),
        ("break probe_max_lateness_1526us (std-1 of the first run)", False, case(late=1526.4), "probe_"),
    ]

    def raw(mutate, probe=True):
        def run():
            rng = random.Random(11)
            pr = CLK.synth_probe(0.2, 5.0)
            std = [list(synth_run(rng, 1650.0)) + [pr] for _ in range(10)]
            auto = [list(synth_run(rng, 2300.0)) + [pr] for _ in range(10)]
            mutate(std, auto)
            return analyse("auto", "auto", "profile_standard", [tuple(x) for x in std], [tuple(x) for x in auto], out=lambda *_: None)
        return run

    def dup_call(std, auto):
        rows = std[2][1]
        i = next(k for k, r in enumerate(rows) if r["kind"] == "call" and r["idx"] == 11)
        rows[i] = dict(rows[i - 1])  # call 11 replaced by a copy of call 10

    def other_cell(std, auto):
        std[4][0]["ms"] = "0.600"

    def no_sidecar(std, auto):
        std[5][2] = None

    def never_ran(std, auto):
        std[9] = ({}, [], None)

    cases += [
        ("break: call 11 of one run replaced by a copy of call 10", False, raw(dup_call), "valid_"),
        ("break: one run's CSV is the 0.6 ms cell", False, raw(other_cell), "valid_"),
        ("break: a run with call records and no probe output file", False, raw(no_sidecar), "probe_"),
        ("a run that never started (no records, no sidecar) fails valid_ only, as in the archived run", False, raw(never_ran), "valid_"),
    ]
    ok = L.run_selftest("G1", cases)
    # the never-started run must not add a probe failure (the archived counts stay 34 of 39)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        vv = raw(never_ran)()
    only = [r[0] for r in vv.fails] == ["valid_standard_10"]
    print("SELFTEST G1 a never-started run adds exactly one FAIL (valid_): %s" % ("ok" if only else "WRONG"))
    return ok and only


def main(argv):
    if argv and argv[0] == "--selftest":
        return 0 if selftest() else 1
    opts = {"--before": "", "--after": "", "--in-force": ""}
    lists = {"--standard": [], "--auto": []}
    cur = None
    it = iter(argv)
    for a in it:
        if a in opts:
            opts[a] = next(it)
            cur = None
        elif a in lists:
            cur = a
        elif cur:
            lists[cur].append(a)
        else:
            print(__doc__)
            return 2

    def load(paths):
        out = []
        for p in paths:
            try:
                with open(p, errors="replace") as f:
                    meta, rows = CLK.read_clk_csv(f.read())
                out.append((meta, rows, CLK.probe_verdicts(p)))
            except OSError:
                out.append(({}, [], None))
        return out

    v = analyse(opts["--before"], opts["--after"], opts["--in-force"], load(lists["--standard"]), load(lists["--auto"]))
    return 0 if v.summary() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
