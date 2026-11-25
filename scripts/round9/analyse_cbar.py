#!/usr/bin/env python3
"""Item C analysis: does the data one dispatch wrote sit in L2 for the next dispatch? Reads the ROW c lines of vkr9 cbar and
r9_cl2 (from the gate's .stdout logs or the program output), prints a table and PASS / FAIL per control.

  python3 -I analyse_cbar.py LOG...        one or more logs; a "-r<N>" in the file name is the process repeat
  python3 -I analyse_cbar.py --page LOG... item C2: the logs of the page-order (increasing) walk; bands of memory-system.md's
                                           page-order rows: 485-545 and 360-390
  python3 -I analyse_cbar.py [--page] --runs N --reps R --cells VK,HIP LOG...
                                           completeness: N process repeats per api, R repetitions per cell and run, VK / HIP distinct
                                           cells per run; anything missing is a FAIL of the control "complete" (the drivers pass these).
                                           The cells must be exactly the names of CELL_NAMES (CELL_NAMES_PAGE with --page), and the
                                           repetitions of every cell and run exactly 0..R-1, each once: a duplicate replacing another
                                           repetition, or a renamed cell, fails it although the counts are right
  python3 -I analyse_cbar.py --selftest    synthetic logs: the base case must PASS, every break must FAIL
Controls: cold in 440-600 cycles; fused at most 330 and under 0.7 x cold; HIP two launches in 330-540; kind b2 pointer check
all ok; race break detected in at least 90 % of the repetitions; repeats of control cells within 10 %."""
import importlib.util
import os
import random
import re
import sys

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("r9lib", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r9lib.py"))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

COLD_BAND = (440.0, 600.0)
FUSED_MAX = 330.0
TWOLAUNCH_BAND = (330.0, 540.0)
COLD_BAND_PAGE = (470.0, 560.0)          # GDDR6 in page order: 485-545 (memory-system.md 2b), with a margin of about 15
MALL_PAGE = (350.0, 400.0)               # memory-side cache in page order: 360-390
TWOLAUNCH_BANDS_PAGE = (MALL_PAGE, COLD_BAND_PAGE)   # either one passes the control; which one it is is the finding
_KINDS = ("none", "exec", "b2", "legacy", "vv7", "full", "event")
CELL_NAMES = {"vulkan": {"n-%s-%s" % (q, k) for q in "gc" for k in _KINDS} | {"%s-%s" % (o, q) for o in ("cold", "fused", "race") for q in "gc"},
              "hip": {"twolaunch", "cold", "fused", "race"}}
CELL_NAMES_PAGE = {"vulkan": {"n-g-b2", "n-c-b2"} | {"%s-%s" % (o, q) for o in ("cold", "fused", "race") for q in "gc"},
                   "hip": {"twolaunch", "cold", "fused", "race"}}
RACE_DETECT = 0.90
REPEAT_TOL = 0.10
CLK_SPREAD = 0.10


def band_name(m, page):
    """Which level a median sits at. Page order: the bands of memory-system.md; else only the same-kernel level is named."""
    if m <= FUSED_MAX:
        return "same-kernel level"
    if page:
        if MALL_PAGE[0] <= m <= MALL_PAGE[1]:
            return "page-order band %.0f-%.0f" % MALL_PAGE
        if COLD_BAND_PAGE[0] <= m <= COLD_BAND_PAGE[1]:
            return "page-order band %.0f-%.0f" % COLD_BAND_PAGE
    return "between the bands" if page else "above the same-kernel level"


def parse(lines_by_run):
    """lines_by_run: {run id: [lines]} -> {(api, queue, kind, order, cell): {run: [(rep, cycles, rt, steps, ok)]}}"""
    cells = {}
    for run, lines in lines_by_run.items():
        for line in lines:
            if not line.startswith("ROW c "):
                continue
            d = L.kvs(line)
            try:
                key = (d["api"], d["queue"], d["kind"], d["order"], d["cell"])
                rec = (int(d["rep"]), float(d["cycles"]), float(d["rt"]), float(d["steps"]), int(d["ok"]))
            except (KeyError, ValueError):
                continue
            cells.setdefault(key, {}).setdefault(run, []).append(rec)
    return cells


def stats(runs):
    """Per run: median cycles per step over repetitions >= 1, median ns per step, ok count, counted."""
    out = {}
    for run, recs in runs.items():
        use = [r for r in recs if r[0] > 0]
        if not use:
            continue
        clk = [r[1] / r[2] * 100.0 for r in use if r[2] > 0]
        out[run] = dict(cps=L.median([r[1] / r[3] for r in use]), nps=L.median([r[2] * 10.0 / r[3] for r in use]),
                        ok=sum(r[4] for r in use), n=len(use), clk=L.median(clk), clk_min=min(clk) if clk else float("nan"),
                        clk_max=max(clk) if clk else float("nan"))
    return out


def complete(v, cells, expect):
    """expect = dict(runs=N, reps=R, cells={"vulkan": n, "hip": n}): every api has N process repeats, every cell is in every one of them
    with R repetitions, and each api has the expected number of distinct cells. One missing job or cell is a FAIL, not a smaller average."""
    missing = []
    for api, ncell in expect["cells"].items():
        keys = sorted(k for k in cells if k[0] == api)
        if len(keys) != ncell:
            missing.append("%s: %d distinct cells found, %d expected" % (api, len(keys), ncell))
        names = expect.get("names", {}).get(api)
        if names is not None:
            got = {k[4] for k in keys}
            if got != names:
                missing.append("%s: cells are not the expected names (missing %s, unexpected %s)" % (
                    api, ",".join(sorted(names - got)[:3]) or "none", ",".join(sorted(got - names)[:3]) or "none"))
        prefixes = {r.split(":")[0] for k in keys for r in cells[k]}
        if len(prefixes) > 1:
            missing.append("%s: logs of different drivers mixed (%s)" % (api, ",".join(sorted(prefixes))))
        for k in keys:
            have = {r.split(":")[1] for r in cells[k]}
            want = {str(i) for i in range(1, expect["runs"] + 1)}
            if have != want:
                missing.append("%s/%s %s: runs %s of %s" % (k[0], k[1], k[4], ",".join(sorted(have)) or "none", ",".join(sorted(want))))
            for r, recs in cells[k].items():
                if len(recs) != expect["reps"]:
                    missing.append("%s/%s %s run %s: %d repetitions of %d" % (k[0], k[1], k[4], r, len(recs), expect["reps"]))
                elif sorted(x[0] for x in recs) != list(range(expect["reps"])):
                    missing.append("%s/%s %s run %s: repetition ids are not 0..%d once each (%d distinct)" % (
                        k[0], k[1], k[4], r, expect["reps"] - 1, len({x[0] for x in recs})))
    v.add("complete", not missing, "%d runs x %d repetitions x %s cells: %s" % (
        expect["runs"], expect["reps"], "+".join(str(n) for n in expect["cells"].values()), "all present" if not missing else "; ".join(missing[:6])))


def analyse(cells, out=print, page=False, expect=None):
    v = L.Verdicts("C2" if page else "C")
    if expect:
        complete(v, cells, expect)
    cold_band = COLD_BAND_PAGE if page else COLD_BAND
    st = {k: stats(r) for k, r in cells.items()}
    med = {k: L.median([s["cps"] for s in runs.values()]) for k, runs in st.items() if runs}

    def pick(api, queue, order, kind=None):
        for k in st:
            if k[0] == api and k[1] == queue and k[3] == order and (kind is None or k[2] == kind) and st[k]:
                return k
        return None

    groups = sorted({(k[0], k[1]) for k in st})
    if not groups:
        v.add("data", False, "no ROW c lines found")
        return v
    cold_ref = {}
    for api, queue in groups:
        tag = "%s/%s" % (api, queue)
        kc = pick(api, queue, "cold")
        if kc is None:
            v.add("cold_" + tag, False, "%s: no cold cell in the logs" % tag)
        else:
            m = med[kc]
            cold_ref[(api, queue)] = m
            v.add("cold_" + tag, cold_band[0] <= m <= cold_band[1],
                  "%s: evicted region, loaded links, reads %.0f cycles per step (band %.0f-%.0f%s)" % (
                      tag, m, *cold_band, ", page order" if page else ""))
        kf = pick(api, queue, "fused")
        if kf is None:
            v.add("fused_" + tag, False, "%s: no fused cell in the logs" % tag)
        else:
            m = med[kf]
            c = cold_ref.get((api, queue))
            v.add("fused_" + tag, m <= FUSED_MAX and (c is None or m < 0.7 * c),
                  "%s: same-kernel read %.0f cycles per step (at most %.0f and under 0.7 x cold %s)" % (
                      tag, m, FUSED_MAX, "n/a" if c is None else "%.0f" % c))
        kt = pick(api, queue, "twolaunch")
        if api == "hip":
            if kt is None:
                v.add("twolaunch_hip", False, "hip: no two-launch cell in the logs")
            else:
                m = med[kt]
                if page:
                    which = band_name(m, True)
                    v.add("twolaunch_hip", which.startswith("page-order band"),
                          "hip: producer launch then consumer launch reads %.0f cycles per step (%s; the page-order bands of "
                          "memory-system.md are %.0f-%.0f and %.0f-%.0f, the control needs one of the two; no cache level is named: "
                          "the bands are the earlier page-order chases' own rows, not a calibration of this probe)" % (
                              m, which, *MALL_PAGE, *COLD_BAND_PAGE))
                else:
                    v.add("twolaunch_hip", TWOLAUNCH_BAND[0] <= m <= TWOLAUNCH_BAND[1],
                          "hip: producer launch then consumer launch reads %.0f cycles per step (band %.0f-%.0f of memory-system.md 2c; "
                          "the cache level is not named)" % (m, *TWOLAUNCH_BAND))
        kr = pick(api, queue, "race")
        if kr is None:
            v.add("race_" + tag, False, "%s: no race cell in the logs" % tag)
        else:
            tot = sum(s["n"] for s in st[kr].values())
            bad = sum(s["n"] - s["ok"] for s in st[kr].values())
            v.add("race_" + tag, tot > 0 and bad / tot >= RACE_DETECT,
                  "%s: break (consumer before producer) fails the pointer check in %d of %d repetitions (needs %.0f %%); "
                  "else the check cannot see stale data" % (tag, bad, tot, 100 * RACE_DETECT))
        if api == "vulkan":
            kb = pick(api, queue, "normal", "b2")
            if kb is None:
                v.add("b2_pointer_" + tag, False, "%s: no b2 cell in the logs" % tag)
            else:
                tot = sum(s["n"] for s in st[kb].values())
                ok = sum(s["ok"] for s in st[kb].values())
                v.add("b2_pointer_" + tag, tot > 0 and ok == tot,
                      "%s: kind b2, producer then consumer: pointer check ok in %d of %d repetitions" % (tag, ok, tot))
    # the core clock of every cell (cycles over REALTIME, from the chase itself): steady within the cell, and the cells of one job
    # alike. The first version's first Vulkan cell ran 14 repetitions at 1.19 GHz, then at 3.2 GHz.
    for api, queue in groups:
        bad, tot = [], 0
        for k, runs in sorted(st.items()):
            if (k[0], k[1]) != (api, queue):
                continue
            for run, s_ in runs.items():
                tot += 1
                if not (s_["clk"] == s_["clk"] and (s_["clk_max"] - s_["clk_min"]) <= CLK_SPREAD * s_["clk"]):
                    bad.append("%s run %s %.0f-%.0f MHz (median %.0f)" % (k[4], run.split(":")[-1], s_["clk_min"], s_["clk_max"], s_["clk"]))
            jobmed = {}
        for run in sorted({r for k, rr in st.items() if (k[0], k[1]) == (api, queue) for r in rr}):
            cm = {k[4]: rr[run]["clk"] for k, rr in st.items() if (k[0], k[1]) == (api, queue) and run in rr}
            m = L.median(list(cm.values()))
            bad += ["%s run %s median %.0f MHz against the job's %.0f" % (c, run.split(":")[-1], x, m) for c, x in cm.items() if abs(x / m - 1) > CLK_SPREAD]
        v.add("clock_%s/%s" % (api, queue), tot > 0 and not bad,
              "%s/%s: the core clock inside every cell moves by at most %.0f %% of its median, and every cell of a job is within %.0f %% of the job's median clock (%d cell runs)%s" % (
                  api, queue, 100 * CLK_SPREAD, 100 * CLK_SPREAD, tot, "" if not bad else "; outside: " + "; ".join(bad[:4])))
    # repeats of the control cells
    for k, runs in st.items():
        if k[3] in ("cold", "fused", "twolaunch") and len(runs) >= 2:
            vals = [s["cps"] for s in runs.values()]
            sp = L.spread(vals)
            v.add("repeat_%s_%s_%s" % (k[0], k[1], k[3]), sp <= REPEAT_TOL,
                  "%s/%s %s: run medians %s, spread %.1f %% (limit %.0f %%)" % (
                      k[0], k[1], k[3], " ".join("%.0f" % x for x in vals), 100 * sp, 100 * REPEAT_TOL))
    # findings: where each barrier kind's consumer read from
    refs = {}
    for api, queue in groups:
        if (api, queue) in cold_ref:
            refs[(api, queue)] = cold_ref[(api, queue)]
    fz = [med[k] for k in med if k[3] == "fused" and k[0] == "vulkan"]
    tl = [med[k] for k in med if k[3] == "twolaunch"]
    cd = [med[k] for k in med if k[3] == "cold" and k[0] == "vulkan"]
    ref = []
    if fz:
        ref.append(("same-kernel (fused) read", L.median(fz)))
    if tl:
        ref.append(("HIP two-launch read", L.median(tl)))
    if cd:
        ref.append(("evicted cold read", L.median(cd)))
    out("")
    out("%-10s %-8s %-7s %-8s %9s %9s %7s  %s" % ("api", "queue", "kind", "order", "cyc/step", "ns/step", "ptr ok", "nearest control"))
    for k in sorted(med, key=lambda k: (k[0], k[1], k[3] != "normal", k[2])):
        if k[3] != "normal":
            continue
        runs = st[k]
        nps = L.median([s["nps"] for s in runs.values()])
        tot = sum(s["n"] for s in runs.values())
        ok = sum(s["ok"] for s in runs.values())
        near = min(ref, key=lambda r: abs(__import__("math").log(med[k] / r[1])))[0] if ref else "n/a"
        out("%-10s %-8s %-7s %-8s %9.0f %9.1f %3d/%-3d  %s" % (k[0], k[1], k[2], k[3], med[k], nps, ok, tot, near))
    out("controls' reference reads used for 'nearest' (measured cells of this run, not cache levels): " + ", ".join("%s %.0f" % r for r in ref))
    if page:
        out("")
        out("BAND page-order bands of memory-system.md: %.0f-%.0f and %.0f-%.0f (the random-order rows there are 700-820 and 420-500); "
            "no cache level is named here" % (*MALL_PAGE, *COLD_BAND_PAGE))
        for k in sorted(med, key=lambda k: (k[0], k[1], k[3], k[2])):
            if k[3] in ("race",):
                continue
            out("BAND %-7s %-8s %-9s %-8s %5.0f cycles per step: %s" % (k[0], k[1], k[3], k[2], med[k], band_name(med[k], True)))
    return v


def synth(rng, race_ok=False, cold_cyc=515.0, twolaunch_cyc=410.0, b2_bad=False, drift_fused_run3=1.0, ramp=False):
    """Synthetic logs, three process repeats of a Vulkan job and a HIP job. Returns {run name: [lines]}."""
    logs = {}
    for r in (1, 2, 3):
        vk, hip = [], []

        def rows(dst, api, queue, kind, order, cell, cyc, okp=1.0, reps=20):
            for rep in range(reps):
                c = cyc * rng.uniform(0.97, 1.03) * 1024
                ok = 1 if rng.random() < okp else 0
                mhz = 3000.0
                if ramp and cell == "n-g-b2" and rep < 14:  # the first Vulkan cell of a job at the idle core clock
                    mhz = 1190.0
                dst.append("ROW c api=%s cell=%s queue=%s kind=%s order=%s rep=%d cycles=%d rt=%d final=1 expect=1 ok=%d steps=1024" %
                           (api, cell, queue, kind, order, rep, c, c / mhz * 100.0, ok))

        for q in ("graphics", "compute"):
            for kind in ("none", "exec", "b2", "legacy", "vv7", "full", "event"):
                cyc = 250.0 if (q == "graphics" and kind == "full") else 415.0
                okp = 1.0
                if b2_bad and kind == "b2" and q == "compute":
                    okp = 0.6
                if kind in ("none", "exec"):
                    okp = 0.5
                rows(vk, "vulkan", q, kind, "normal", "n-%s-%s" % (q[0], kind), cyc, okp)
            rows(vk, "vulkan", q, "b2", "cold", "cold-" + q[0], cold_cyc)
            rows(vk, "vulkan", q, "b2", "fused", "fused-" + q[0], 190.0 * (drift_fused_run3 if r == 3 else 1.0))
            rows(vk, "vulkan", q, "b2", "race", "race-" + q[0], 300.0, 1.0 if race_ok else 0.02)
        rows(hip, "hip", "hip", "launch", "twolaunch", "twolaunch", twolaunch_cyc)
        rows(hip, "hip", "hip", "launch", "cold", "cold", cold_cyc)
        rows(hip, "hip", "hip", "launch", "fused", "fused", 195.0)
        rows(hip, "hip", "hip", "launch", "race", "race", 330.0, 1.0 if race_ok else 0.0)
        logs["c-vk:%d" % r] = vk
        logs["c-hip:%d" % r] = hip
    return logs


def walk_is_full_cycle(a, c, n=32768):
    """The page walks of vkr9 and r9_cl2 (next = (a * slot + c) mod n, a = 1, c = 1 or n - 1) visit every slot once."""
    p, seen = 0, set()
    for _ in range(n):
        seen.add(p)
        p = (a * p + c) % n
    return len(seen) == n and p == 0


def selftest():
    EXP = dict(runs=3, reps=20, cells={"vulkan": 20, "hip": 4}, names=CELL_NAMES)

    def case(page=False, drop_run=None, drop_cell=None, short=None, dup_rep=None, rename=None, **kw):
        def run():
            logs = synth(random.Random(7), **kw)
            if drop_run:
                del logs[drop_run]
            if drop_cell:
                logs = {k: [x for x in v_ if "cell=%s " % drop_cell not in x] for k, v_ in logs.items()}
            if short:
                logs = {k: [x for x in v_ if not ("cell=n-g-b2 " in x and "rep=19 " in x)] if k == short else v_ for k, v_ in logs.items()}
            if dup_rep:  # repetition 10 of the cell replaced by a second copy of repetition 9: the count stays 20
                key, run_ = dup_rep
                out_ = []
                for x in logs[run_]:
                    if "cell=%s " % key in x and " rep=10 " in x:
                        x = x.replace(" rep=10 ", " rep=9 ")
                    out_.append(x)
                logs[run_] = out_
            if rename:  # one cell name changed in every run, the count of distinct cells stays 20
                old, new = rename
                logs = {k: [x.replace("cell=%s " % old, "cell=%s " % new) for x in v_] for k, v_ in logs.items()}
            return analyse(parse(logs), out=lambda *_: None, page=page, expect=EXP)
        return run

    def walk(a, c, steps=1024, start=7919, n=32768):
        def run():
            v = L.Verdicts("C2walk")
            p = start
            for _ in range(steps):
                p = (a * p + c) % n
            want = (start + steps * (1 if c == 1 else -1)) % n
            v.add("walk_a%d_c%d" % (a, c), walk_is_full_cycle(a, c) and p == want,
                  "full cycle over %d slots and the walk of %d steps ends at %d (want %d)" % (n, steps, p, want))
            return v
        return run

    cases = [
        ("base", True, case()),
        ("break race_not_detected", False, case(race_ok=True), "race_"),
        ("break core clock ramps inside the first cell (1.19 GHz for 14 repetitions, then 3 GHz)", False, case(ramp=True), "clock_"),
        ("break cold_reads_cache", False, case(cold_cyc=260.0), "cold_"),
        ("break hip_two_launch_reads_l0", False, case(twolaunch_cyc=150.0), "twolaunch_hip"),
        ("break b2_pointer_fails", False, case(b2_bad=True), "b2_pointer_"),
        ("break repeat_drift", False, case(drift_fused_run3=1.5), "repeat_"),
        ("break no_data", False, lambda: analyse({}, out=lambda *_: None), "data"),
        ("break one Vulkan job lost (c-vk-r2): the old analysis averaged the two left", False, case(drop_run="c-vk:2"), "complete"),
        ("break one HIP job lost (c-hip-r3)", False, case(drop_run="c-hip:3"), "complete"),
        ("break one cell lost in every run (race-c)", False, case(drop_cell="race-c"), "complete"),
        ("break one run short of repetitions (n-g-b2 of c-vk-r1 has 19 of 20)", False, case(short="c-vk:1"), "complete"),
        ("break: repetition 10 of n-g-b2 in c-vk-r1 replaced by a copy of repetition 9 (the count is still 20)", False,
         case(dup_rep=("n-g-b2", "c-vk:1")), "complete"),
        ("break: a cell renamed in every run (n-g-vv7 -> n-g-vv8; 20 distinct cells still)", False, case(rename=("n-g-vv7", "n-g-vv8")), "complete"),
        # page order: cold 515 (GDDR6 485-545), two launches 375 (memory-side cache 360-390) or 515 (GDDR6) both pass
        ("page base, two launches in the memory-side cache band", True, case(page=True, cold_cyc=515.0, twolaunch_cyc=375.0)),
        ("page base, two launches in the GDDR6 band", True, case(page=True, cold_cyc=515.0, twolaunch_cyc=515.0)),
        ("page break: random-order numbers (cold 770, two launches 711) judged by page bands", False,
         case(page=True, cold_cyc=770.0, twolaunch_cyc=711.0), "cold_"),
        ("page break: two launches between the bands (440)", False, case(page=True, cold_cyc=515.0, twolaunch_cyc=440.0), "twolaunch_hip"),
        ("page break: race not detected", False, case(page=True, cold_cyc=515.0, twolaunch_cyc=375.0, race_ok=True), "race_"),
        ("page walk a=1 c=1 is a full cycle and ends at start + steps", True, walk(1, 1)),
        ("page walk a=1 c=n-1 (the race cell's odd repetitions) is a full cycle and ends at start - steps", True, walk(1, 32767)),
        ("break: a walk with an even step (a=1 c=2) is not a full cycle", False, walk(1, 2), "walk_"),
    ]
    return L.run_selftest("C", cases)


def main(argv):
    if argv and argv[0] == "--selftest":
        return 0 if selftest() else 1
    page = False
    expect = None
    opts = {}
    while argv and argv[0].startswith("--"):
        if argv[0] == "--page":
            page = True
            argv = argv[1:]
        elif argv[0] in ("--runs", "--reps", "--cells") and len(argv) > 1:
            opts[argv[0]] = argv[1]
            argv = argv[2:]
        else:
            print(__doc__)
            return 2
    if opts:
        try:
            vk, hip = (int(x) for x in opts["--cells"].split(","))
            expect = dict(runs=int(opts["--runs"]), reps=int(opts["--reps"]), cells={"vulkan": vk, "hip": hip},
                          names=CELL_NAMES_PAGE if page else CELL_NAMES)
        except (KeyError, ValueError):
            print(__doc__)
            return 2
    if not argv:
        print(__doc__)
        return 2
    by_run = {}
    for path in argv:
        m = re.search(r"-r(\d+)", os.path.basename(path))
        run = "%s:%s" % (re.sub(r"-r\d+.*", "", os.path.basename(path)), m.group(1) if m else "0")
        with open(path, errors="replace") as f:
            by_run[run] = f.read().splitlines()
    v = analyse(parse(by_run), page=page, expect=expect)
    return 0 if v.summary() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
