#!/usr/bin/env python3 -I
"""Item B: the engine sampler's busy ns per frame against the probe's own per-burst GPU time, arm by arm.
Usage: bana.py <engines.csv> <bursts prefix, e.g. B/B-g-1-dense-> <engine> (--pid PID | --name NAME)
The counter of a process is read at plateaus (>= 3 equal samples between arms); the busy time of arm k+1 is plateau k+1 minus plateau k
(arm 1 is not used: the first sample comes after it started). The probe's time per frame is (bottom - top) timestamps of the same arm.
Limit: 0.1 ms per frame."""
import csv, glob, sys, statistics as st
csvp, pref, eng = sys.argv[1:4]
pid = name = None
if "--pid" in sys.argv: pid = sys.argv[sys.argv.index("--pid") + 1]
if "--name" in sys.argv: name = sys.argv[sys.argv.index("--name") + 1]
rows = [r for r in csv.DictReader(open(csvp)) if r["engine"] == eng and ((pid and r["pid"] == pid) or (name and r["name"] == name))]
rows.sort(key=lambda r: float(r["t_s"]))
vals = [int(r["busy_ns"]) for r in rows]
plat = []; i = 0
while i < len(vals):
    j = i
    while j + 1 < len(vals) and vals[j + 1] == vals[i]: j += 1
    if j - i + 1 >= 3: plat.append(vals[i])
    i = j + 1
arms = sorted(glob.glob(pref + "*-bursts.csv"))
bad = 0; allp = []; alls = []
for k in range(1, len(plat)):
    if k >= len(arms): break
    dur = [(int(r["bottom_tick"]) - int(r["top_tick"])) * 10.0 for r in csv.DictReader(open(arms[k]))]
    s = (plat[k] - plat[k - 1]) / len(dur); p = st.mean(dur)
    allp.append(p); alls.append(s)
    d = (s - p) / 1e6
    print(f"arm {k+1} ({arms[k].split('-bursts')[0].split('-')[-1]}): sampler {s/1e6:.4f} ms/frame, probe {p/1e6:.4f} ms/frame (n {len(dur)}), difference {d:+.4f} ms")
    bad += abs(d) > 0.1
if not alls:
    print(f"VERDICT enginesampler_{eng}: FAIL no complete arm (plateaus {len(plat)})"); sys.exit(1)
print(f"VERDICT enginesampler_{eng}: {'PASS' if not bad else 'FAIL'} {len(alls)} arms, mean difference {(st.mean(alls)-st.mean(allp))/1e6:+.4f} ms per frame (limit 0.1 each)")
sys.exit(0 if not bad else 1)
