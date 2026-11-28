#!/usr/bin/env python3
"""Round 10 item 3 analysis: reads the ROW and CELL lines of vkr10 and r10_cl3 and judges the controls, then the gap closure.

Usage: analyse_item3.py [--unpinned] <log>...     (jobwrap .out files or gate logs; every ROW line is used)
       analyse_item3.py --selftest                synthetic data: the real judgement passes it, and every deliberate break must fail

Cell names are `<api>-<id>-<walk>`, api vk or hip, walk r (random) or p (page); the run script writes them. A cell's number is the
median of its counted repetitions (repetition 0 is not counted; repetitions with ok=0 are not counted and make `pointer_ok` fail).
Gap = vk-base - hip-base in the same walk. A change from Vulkan toward HIP CLOSES a share of it, (vk-base - v) / gap; the matching change
from HIP toward Vulkan REOPENS a share, (h - hip-base) / gap. An answer needs both at 80 % or more, in cycles and in ns, in both walks.
Controls are judged first; a failed control stays failed (nothing is edited to pass) and the item says which part it spoils.
"""
import collections
import re
import statistics
import sys

sys.dont_write_bytecode = True

# (factor, Vulkan change, matching HIP change); the all-factors pair is the combination
PAIRS = [("form", "v1", "h1h2"), ("alloc", "v3", "h4"), ("evict", "v4", "h5"), ("all", "v5", "h6")]
BANDS = {  # (api, walk) -> cold cycles per step band; source rows: measurements/round9.md, items C (random) and C2 (page)
    ("hip", "r"): (690, 720, "HIP cold random 700 (C: 699 707 700), PLAN-r10 band 690-720"),
    ("hip", "p"): (470, 560, "HIP cold page 488 (C2: 489 488 488), GDDR6 page-order band 470-560"),
    ("vk", "r"): (740, 900, "Vulkan cold random 812-842 (C cell medians), 10 % either side"),
    ("vk", "p"): (720, 880, "Vulkan cold page 799-802 (C2 cell medians), 10 % either side"),
}
FUSED_BANDS = {"hip": (200, 270, "HIP fused 231 (C: 231 234 231, C2 231 235 231)"),
               "vk": (260, 340, "Vulkan fused 294-302 (C 302 302, C2 294 295)")}


def parse(lines):
    rows, cells, notrun, verdict_fail = [], [], [], []
    for ln in lines:
        # A binary can print VERDICT FAIL and still have been treated as success. Any failed VERDICT fails the item.
        if re.search(r"^VERDICT\b.*\bFAIL\b", ln):
            verdict_fail.append(ln.strip())
        m = re.match(r"^(ROW|CELL|NOT RUN)\b", ln)
        if not m:
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", ln))
        if m.group(1) == "ROW":
            if "cycles" in kv or "cycles1" in kv:
                rows.append(kv)
        elif m.group(1) == "CELL":
            cells.append(kv)
        else:
            notrun.append(ln.strip())
    return rows, cells, notrun, verdict_fail


def key_of(cell):
    m = re.match(r"^(vk|hip)-(.+)-([rp])$", cell)
    return m.groups() if m else None


def cell_stats(rows):
    """cell name -> list of per-process dicts of cycles/ns per step of the counted repetitions."""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("mode") == "reach":
            continue
        by[(r["cell"], r.get("_job", ""))].append(r)
    out = collections.defaultdict(list)
    for (cell, job), rs in by.items():
        counted = [r for r in rs if int(r["rep"]) > 0]
        good = [r for r in counted if r["ok"] == "1"]
        if not counted:
            continue
        cyc = [int(r["cycles"]) / int(r["steps"]) for r in good]
        ns = [int(r["rt"]) * 10.0 / int(r["steps"]) for r in good]
        late = [int(r["cycles"]) / int(r["steps"]) for r in good if int(r["rep"]) >= 12]
        lab = collections.defaultdict(list)
        for r in good:
            if r.get("uclk0") == r.get("uclk1") and r.get("uclk0", "-1") != "-1":
                lab[r["uclk0"]].append((int(r["cycles"]) / int(r["steps"]), int(r["rt"]) * 10.0 / int(r["steps"])))
        out[cell].append(dict(job=job, n=len(counted), ok=len(good), cyc=statistics.median(cyc) if cyc else float("nan"),
                              ns=statistics.median(ns) if ns else float("nan"),
                              late=statistics.median(late) if late else float("nan"), lab=lab))
    return out


def med(st, cell, what="cyc"):
    v = [p[what] for p in st.get(cell, []) if p[what] == p[what]]
    return statistics.median(v) if v else None


def judge(rows, cells, notrun, unpinned=True, verdict_fail=None):
    verdict_fail = list(verdict_fail or [])
    st = cell_stats(rows)
    res = []  # (name, ok, text)

    def v(name, ok, text):
        res.append((name, bool(ok), text))

    v("verdict_line", not verdict_fail,
      "every VERDICT line passes" if not verdict_fail else "a VERDICT line failed: " + "; ".join(verdict_fail[:4]))

    def get(api, i, w, what="cyc"):
        return med(st, f"{api}-{i}-{w}", what)

    # 58 C start rule and the cells that did not run
    hot = [(c["cell"] if "cell" in c else c.get("name", "?"), float(c["hotspot"])) for c in cells if "hotspot" in c]
    v("start_temp", all(h <= 58.0 for _, h in hot) and not notrun,
      f"{len(hot)} cells started at or below 58 C (highest {max([h for _, h in hot], default=0):.0f} C); {len(notrun)} NOT RUN")
    # pointer check on every cell that is not a deliberate break
    bad = []
    for cell, ps in st.items():
        k = key_of(cell)
        for p in ps:
            if p["ok"] != p["n"]:
                bad.append(f"{cell} {p['ok']}/{p['n']}")
    v("pointer_ok", not bad, "every counted repetition ends on the expected final address (and stamp sum where there is one)"
      if not bad else "wrong final pointer: " + ", ".join(bad[:6]))
    # repeats within 5 % (process repeats of one cell)
    spread = []
    for cell, ps in st.items():
        if len(ps) >= 2:
            xs = [p["cyc"] for p in ps]
            sp = (max(xs) - min(xs)) / statistics.median(xs)
            if sp > 0.05:
                spread.append(f"{cell} {sp * 100:.1f} %")
    v("repeats", not spread, f"process repeats of every cell agree within 5 %" if not spread else "spread above 5 %: " + ", ".join(spread[:6]))
    # the bands of the two baselines and their fused cells
    for (api, w), (lo, hi, src) in BANDS.items():
        x = get(api, "base", w)
        v(f"{api}_base_cold_{w}", x is not None and lo <= x <= hi, f"{api}-base-{w} = {x} cycles/step, band {lo}-{hi} ({src})")
    for api, (lo, hi, src) in FUSED_BANDS.items():
        for w in "rp":
            x = get(api, "fbase", w)
            if x is not None:
                v(f"{api}_fused_{w}", lo <= x <= hi, f"{api}-fbase-{w} = {x} (cache-hit band {lo}-{hi}; {src})")
    # deliberate breaks: they must read as caches / system memory, or the probe cannot tell
    for api in ("vk", "hip"):
        for w in "rp":
            b = get(api, "base", w)
            ne, lp = get(api, "noev", w), get(api, "loop8", w)
            if b and ne is not None:
                v(f"{api}_noevict_break_{w}", ne <= 0.6 * b, f"{api}-noev-{w} = {ne:.0f} vs cold {b:.0f}: with no eviction it must read at a cache level (at most 60 % of cold)")
            if b and lp is not None:
                v(f"{api}_loop8_break_{w}", lp <= 0.5 * b, f"{api}-loop8-{w} = {lp:.0f} vs cold {b:.0f}: an 8-line footprint must read from L0/L2 (at most 50 % of cold)")
    gtt = {"vk": ("gtt", "vk-gtt-r"), "hip": ("host", "hip-host-r")}
    for api, (nm, cell) in gtt.items():
        x, b = med(st, cell), get(api, "base", "r")
        if x is not None and b:
            v(f"{api}_placement_break", x >= 2.5 * b, f"{cell} = {x:.0f} vs {b:.0f}: system memory must read at least 2.5x slower than VRAM or placement is invisible")
    # strides: 128 B equals the page-order cell of the same form; 2 MiB at least as slow as 4 KiB
    for api, ref in (("hip", get("hip", "base", "p")), ("vk", get("vk", "v1", "p"))):
        s128, s4k, s2m = (med(st, f"{api}-st{x}-p") for x in ("128", "4k", "2m"))
        if s128 is not None and ref:
            v(f"{api}_stride128", abs(s128 - ref) <= 0.05 * ref, f"{api}-st128-p = {s128:.0f} vs {ref:.0f} (same form, page order): within 5 %")
        if s4k is not None and s2m is not None:
            v(f"{api}_stride_order", s2m >= 0.95 * s4k, f"{api}: 2 MiB stride {s2m:.0f} vs 4 KiB {s4k:.0f}: must be at least as slow (one translation miss per step in both)")
    # gap and closure
    report = []
    answered = {}
    for w in "rp":
        vb, hb = get("vk", "base", w), get("hip", "base", w)
        vbn, hbn = get("vk", "base", w, "ns"), get("hip", "base", w, "ns")
        if not (vb and hb):
            continue
        gap, gapn = vb - hb, (vbn - hbn) if vbn and hbn else None
        report.append(f"walk {w}: vk-base {vb:.0f} cyc / {vbn or 0:.0f} ns, hip-base {hb:.0f} cyc / {hbn or 0:.0f} ns, gap {gap:.0f} cyc / {gapn or 0:.0f} ns")
        for fac, vid, hid in PAIRS:
            x, y = get("vk", vid, w), get("hip", hid, w)
            xn, yn = get("vk", vid, w, "ns"), get("hip", hid, w, "ns")
            if x is None or y is None or gap <= 0:
                continue
            close, reopen = (vb - x) / gap, (y - hb) / gap
            closen = (vbn - xn) / gapn if xn and gapn and gapn > 0 else None
            reopenn = (yn - hbn) / gapn if yn and gapn and gapn > 0 else None
            ok = close >= 0.8 and reopen >= 0.8 and closen is not None and reopenn is not None and closen >= 0.8 and reopenn >= 0.8
            answered.setdefault(fac, []).append(ok)
            report.append(f"  {fac:5s} walk {w}: {vid} closes {close * 100:.0f} % ({'-' if closen is None else f'{closen * 100:.0f}'} % in ns), "
                          f"{hid} reopens {reopen * 100:.0f} % ({'-' if reopenn is None else f'{reopenn * 100:.0f}'} % in ns)  -> {'both >= 80 %' if ok else 'not an answer'}")
    failed_ctrl = [n for n, ok, _ in res if not ok]
    names = [f for f, oks in answered.items() if f != "all" and len(oks) == 2 and all(oks)]
    combo = len(answered.get("all", [])) == 2 and all(answered["all"])
    if failed_ctrl:
        summary = "CONTROLS FAILED (" + ", ".join(failed_ctrl[:8]) + "): no part of the closure below is interpreted until the control is explained"
    elif names:
        summary = "ANSWER: " + ", ".join(names) + " closes >= 80 % of the gap (cycles and ns, both walks) and its HIP reverse reopens >= 80 %"
    else:
        summary = ("NO SINGLE CHANGE closes and reopens 80 %" + ("; the combination of all the factors does" if combo else "")
                   + ": report the split by factor (step 7), name no one cause")
    return res, report, summary, st


def matched_clock(st):
    lines = []
    for w in "rp":
        for cell in (f"vk-base-{w}", f"hip-base-{w}"):
            for p in st.get(cell, []):
                for lab, xs in sorted(p["lab"].items(), key=lambda t: -len(t[1])):
                    if len(xs) >= 3:
                        lines.append(f"  {cell} job {p['job']}: label uclk {lab} MHz, {len(xs)} reps, median {statistics.median(x[0] for x in xs):.0f} cyc / "
                                     f"{statistics.median(x[1] for x in xs):.0f} ns")
    return lines


def reach(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("mode") == "reach" and int(r["rep"]) > 0 and r["ok"] == "1":
            n = int(r["n"])
            by[r["api"]][n].append((int(r["cycles1"]) / n, int(r["cycles2"]) / n))
    out = []
    for api, d in by.items():
        ns = sorted(d)
        p2 = {n: statistics.median(x[1] for x in d[n]) for n in ns}
        knee = next((n for n in ns if p2[n] > 1.5 * p2[ns[0]]), None)
        out.append(f"  {api}: second pass per step " + ", ".join(f"N={n}: {p2[n]:.0f}" for n in ns) + f"; first N above 1.5x the N={ns[0]} value: {knee}")
    return out


def run(files, unpinned=True):
    lines = []
    for f in files:
        job = re.sub(r"\.(out|log|stdout)$", "", f.split("/")[-1])
        for ln in open(f, errors="replace"):
            lines.append(ln if not ln.startswith("ROW") else ln.rstrip("\n") + f" _job={job}\n")
    rows, cells, notrun, verdict_fail = parse(lines)
    res, report, summary, st = judge(rows, cells, notrun, unpinned, verdict_fail)
    print(f"# item 3 analysis ({'UNPINNED: step 1 not run, memory clock as the card chose it; read the matched-clock lines' if unpinned else 'pinned'})")
    for n, ok, t in res:
        print(f"VERDICT item3 {n}: {'PASS' if ok else 'FAIL'} {t}")
    print("## gap and closure (cycles per step and ns per step; Vulkan runs at about 2.0 GHz, HIP at about 3.0 GHz, so ns is the fair column)")
    print(*report, sep="\n")
    mc = matched_clock(st)
    if mc:
        print("## cells compared only on repetitions whose memory clock read the same before and after")
        print(*mc, sep="\n")
    rc = reach(rows)
    if rc:
        print("## reach (N 4 KiB pages walked twice; cycles per step)")
        print(*rc, sep="\n")
    print(summary)
    return 1 if any(not ok for _, ok, _ in res) else 0


# ----------------------------------------------------------------------------------------------------------------------------------
def _synth(over=None):
    """Synthetic ROW lines: HIP cold 700/488 and Vulkan 830/800 (round 9), the form factor explains all of the gap."""
    over = over or {}
    base = {  # (api, id, walk) -> cycles per step
        ("hip", "base", "r"): 700, ("hip", "base", "p"): 488, ("vk", "base", "r"): 830, ("vk", "base", "p"): 800,
        ("hip", "fbase", "r"): 231, ("hip", "fbase", "p"): 231, ("vk", "fbase", "r"): 302, ("vk", "fbase", "p"): 295,
        ("vk", "v1", "r"): 705, ("vk", "v1", "p"): 492, ("hip", "h1h2", "r"): 825, ("hip", "h1h2", "p"): 795,
        ("vk", "v3", "r"): 828, ("vk", "v3", "p"): 799, ("hip", "h4", "r"): 702, ("hip", "h4", "p"): 490,
        ("vk", "v4", "r"): 826, ("vk", "v4", "p"): 797, ("hip", "h5", "r"): 703, ("hip", "h5", "p"): 489,
        ("vk", "v5", "r"): 706, ("vk", "v5", "p"): 493, ("hip", "h6", "r"): 822, ("hip", "h6", "p"): 790,
        ("vk", "noev", "r"): 300, ("vk", "noev", "p"): 290, ("hip", "noev", "r"): 280, ("hip", "noev", "p"): 270,
        ("vk", "loop8", "r"): 200, ("hip", "loop8", "r"): 150, ("vk", "gtt", "r"): 2500, ("hip", "host", "r"): 2400,
        ("hip", "st128", "p"): 489, ("hip", "st4k", "p"): 560, ("hip", "st2m", "p"): 700,
        ("vk", "st128", "p"): 492, ("vk", "st4k", "p"): 570, ("vk", "st2m", "p"): 710,
    }
    base.update(over)
    ghz = {"hip": 3.0, "vk": 3.0}
    lines = []
    for (api, i, w), c in base.items():
        cell = f"{api}-{i}-{w}"
        lines.append(f"CELL {cell} api={api} hotspot=40")
        for rep in range(20):
            rt = int(c * 1024 / (ghz[api] * 1000.0) * 100 / 1.0)  # 10 ns ticks for 1024 steps at ghz
            lines.append(f"ROW r10 api={api} cell={cell} rep={rep} cycles={int(c * 1024)} rt={rt} ok=1 steps=1024 uclk0=1124 uclk1=1124 _job=j1\n")
    for rep in range(1, 5):
        for n, p1, p2 in ((64, 300, 120), (8192, 500, 400)):
            lines.append(f"ROW r10 api=hip cell=reach mode=reach n={n} rep={rep} cycles1={p1 * n} cycles2={p2 * n} rt=1 ok=1 steps={n} uclk0=1 uclk1=1")
    return lines


def selftest():
    bad = 0

    def case(label, over=None, mutate=None, expect_fail=None, expect_summary=None):
        nonlocal bad
        lines = _synth(over)
        if mutate:
            lines = mutate(lines)
        rows, cells, notrun, verdict_fail = parse(lines)
        res, report, summary, _ = judge(rows, cells, notrun, verdict_fail=verdict_fail)
        failed = {n for n, ok, _ in res if not ok}
        status = 1 if failed else 0
        good = True
        if expect_fail is None:
            good = status == 0 and not failed and summary.startswith(expect_summary or "ANSWER")
        else:
            good = status == 1 and any(f.startswith(expect_fail) for f in failed)
        print(f"  {label}: {'ok' if good else 'WRONG'} (failed controls {sorted(failed)[:4]}; {summary[:70]})")
        bad += not good

    print("self-test of analyse_item3.py")
    case("clean data: every control passes and the form pair is the answer", expect_summary="ANSWER: form")
    case("deliberate break: no-evict cell reads cold", {("vk", "noev", "r"): 830}, expect_fail="vk_noevict_break_r")
    case("deliberate break: the footprint control reads cold", {("hip", "loop8", "r"): 700}, expect_fail="hip_loop8_break_r")
    case("deliberate break: system memory as fast as VRAM", {("hip", "host", "r"): 700}, expect_fail="hip_placement_break")
    case("deliberate break: 2 MiB stride faster than 4 KiB", {("hip", "st2m", "p"): 400}, expect_fail="hip_stride_order")
    case("deliberate break: HIP baseline outside its band", {("hip", "base", "p"): 600}, expect_fail="hip_base_cold_p")
    case("deliberate break: a wrong final pointer", mutate=lambda ls: [l.replace("ok=1", "ok=0") if "cell=vk-v3-r" in l and "rep=3 " in l else l for l in ls],
         expect_fail="pointer_ok")
    case("deliberate break: process repeats differ by 20 %", mutate=lambda ls: ls + [l.replace("_job=j1", "_job=j2").replace(f"cycles={830 * 1024}", f"cycles={1000 * 1024}") for l in ls if "cell=vk-base-r" in l],
         expect_fail="repeats")
    case("no single change: nothing closes the gap", {("vk", "v1", "r"): 828, ("vk", "v1", "p"): 799, ("vk", "st128", "p"): 799, ("hip", "h1h2", "r"): 702, ("hip", "h1h2", "p"): 490},
         expect_summary="NO SINGLE CHANGE")
    case("closure without the reverse is not an answer", {("hip", "h1h2", "r"): 720, ("hip", "h1h2", "p"): 500}, expect_summary="NO SINGLE CHANGE")
    case("a cell starting at 65 C", mutate=lambda ls: [l.replace("hotspot=40", "hotspot=65") if l.startswith("CELL vk-base-r") else l for l in ls], expect_fail="start_temp")
    case("deliberate break: a failed cell_clock VERDICT",
         mutate=lambda ls: ls + ["VERDICT r10_cl3 cell_clock: FAIL spread 12% of median"],
         expect_fail="verdict_line")
    print("SELFTEST", "PASS" if not bad else "FAIL", f"({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    unp = "--pinned" not in a
    files = [x for x in a if not x.startswith("--")]
    if not files:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(files, unp))
