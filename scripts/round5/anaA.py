#!/usr/bin/env python3 -I
# Item A summary: per job, the verdict, whether every RESULT line is identical across repetitions, and core clock/power range per state.
import glob, re, hashlib, csv, os, sys, collections
R = os.environ["R_RUN"]
rows = collections.defaultdict(dict)   # (state, job) -> rep -> (verdict_pass, hash)
clk = collections.defaultdict(list)
for log in sorted(glob.glob(f"{R}/A-[LHP]*-*.log")):
    n = os.path.basename(log)[:-4]
    m = re.match(r"A-([LHP])(\d+)-(.+)$", n)
    if not m: continue
    st, rep, job = m.groups()
    t = open(log).read()
    res = [l for l in t.splitlines() if l.startswith("RESULT")]
    ver = [l for l in t.splitlines() if l.startswith("VERDICT")]
    ok = bool(ver) and all(": PASS" in v for v in ver)
    h = hashlib.md5("\n".join(res).encode()).hexdigest()[:8]
    rows[(st, job)][int(rep)] = (ok, h, len(res))
    sp = f"{R}/watch/{n}/samples.csv"
    if os.path.exists(sp):
        with open(sp) as f:
            next(f)
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    if float(r["gfx_act_pct"]) >= 30 or float(r["sclk_MHz"]) > 2000:
                        clk[(st, n)].append((float(r["sclk_MHz"]), float(r["power_W"]), float(r["hotspot_C"])))
                except Exception: pass
print("job | state | reps | all PASS | RESULT lines identical across reps")
bad = 0
for (st, job), d in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    allp = all(v[0] for v in d.values()); same = len({v[1] for v in d.values()}) == 1
    bad += (not allp)
    print(f"{job:28s} {st} reps={sorted(d)} pass={allp} identical={same} hashes={sorted({v[1] for v in d.values()})}")
for st in "LHP":
    xs = [c for (s, n), v in clk.items() if s == st for c in v]
    if xs:
        cl = sorted(x[0] for x in xs); pw = sorted(x[1] for x in xs)
        print(f"state {st}: awake/busy samples {len(xs)}; core MHz min {cl[0]:.0f} p50 {cl[len(cl)//2]:.0f} max {cl[-1]:.0f}; reported W max {pw[-1]:.0f}; hotspot max {max(x[2] for x in xs):.0f}")
print("jobs with a non-PASS verdict:", bad)
