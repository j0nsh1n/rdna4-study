#!/usr/bin/env python3 -I
import csv, glob, statistics as st, sys
import os
R = os.environ.get("R_RUN", ".")
def load(job):
    lines = open(f"{R}/watch/{job}/samples.csv").read().splitlines()[1:]
    rd = [r for r in csv.DictReader(l for l in lines if not l.startswith("#")) if r["busy_pct"]]
    busy = [r for r in rd if float(r["busy_pct"]) >= 90 and float(r["power_W"]) > 120]
    busy = busy[15:]   # drop the first 0.3 s: the card's power average is still rising
    if len(busy) < 15: return None
    pw = [float(r["power_W"]) for r in busy]; cl = [float(r["sclk_MHz"]) for r in busy]
    thr = sum(1 for r in busy if int(r["indep_throttle_status"], 16) & ~(1 << 36 if False else 0) != 0)
    return dict(n=len(busy), W=st.mean(pw), MHz=st.mean(cl), thr=thr, hot=max(float(r["hotspot_C"]) for r in busy))
for pre, label in (("Fp", "patched tree (real FP8 decode kernel, mmvf)"), ("F", "unmodified tree (FP8 dequantise path)")):
    jobs = sorted(set(j.split("/")[2] for j in glob.glob(f"{R}/watch/{pre}-[0-9][0-9]-*/")))
    diffs = []; print(f"== {label}")
    for i in range(0, len(jobs), 2):
        a, b = jobs[i], jobs[i + 1]
        da, db = load(a), load(b)
        if not da or not db: print("skip", a, b, da, db); continue
        f, q = (da, db) if "f8" in a else (db, da)
        diffs.append(f["W"] - q["W"])
        print(f"pair {a[-9:]}/{b[-9:]}: FP8 {f['W']:.1f} W {f['MHz']:.0f} MHz (n {f['n']}, flagged {f['thr']}); Q8_0 {q['W']:.1f} W {q['MHz']:.0f} MHz (n {q['n']}, flagged {q['thr']}); FP8 - Q8_0 = {f['W']-q['W']:+.1f} W")
    if diffs: print(f"mean {st.mean(diffs):+.1f} W, min {min(diffs):+.1f}, max {max(diffs):+.1f} over {len(diffs)} pairs")
