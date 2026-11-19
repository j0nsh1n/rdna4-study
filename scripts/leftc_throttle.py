#!/usr/bin/env python3
"""Leftovers C, lane B, SC-3: what goes with throttle bit 36 (TEMP_HOTSPOT) in the kept sched_trace samples.

Usage: leftc_throttle.py [--break] TRACE_SYSFS.csv...   (sched_trace's *-sysfs.csv: 50 ms or faster samples with
       hotspot, power, sclk and indep_throttle_status; read only)

For every sample: bit 36 set or not, hotspot, its rise over the last 200 ms, reported power, sclk, time since the load
started. For each variable the separation of samples with and without the bit is the AUC (probability that a random
sample with the bit has a higher value than one without; 0.5 = no association). Control with a known answer: PPT0
(bit 0) against reported power (scheduling item 5: set at the 317 W cap) must give AUC >= 0.9, and the same with the
PPT0 labels shuffled must give 0.4 .. 0.6. --break uses the shuffled labels for the first check, so it must FAIL.
"""
import csv
import random
import sys


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        try:
            t = float(r["t_from_load_start_s"])
            hot = float(r["hotspot_C"])
            pw = float(r["power_W_reported"])
            sclk = float(r["sclk_MHz"])
            th = int(r["indep_throttle_status"], 16)
        except (KeyError, ValueError):
            continue
        out.append({"t": t, "hot": hot, "power": pw, "sclk": sclk, "b36": (th >> 36) & 1, "ppt0": th & 1, "file": path})
    # rise of the hotspot over the last 200 ms (by time, not by sample count: samplers differ)
    for i, r in enumerate(out):
        j = i
        while j > 0 and r["t"] - out[j - 1]["t"] <= 0.2 + 1e-9:
            j -= 1
        dt = r["t"] - out[j]["t"]
        r["dhot"] = (r["hot"] - out[j]["hot"]) / dt if dt > 0 else 0.0
    return out


def auc(xs, labels):
    pos = [x for x, l in zip(xs, labels) if l]
    neg = [x for x, l in zip(xs, labels) if not l]
    if not pos or not neg:
        return float("nan")
    allv = sorted((x, l) for x, l in zip(xs, labels))
    # rank-sum with ties averaged
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks.setdefault(k, (i + j + 1) / 2)
        i = j
    rp = sum(ranks[k] for k, (x, l) in enumerate(allv) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    args = [a for a in sys.argv[1:] if a != "--break"]
    brk = "--break" in sys.argv
    if not args:
        print(__doc__)
        return 2
    rows = []
    for p in args:
        rows += load(p)
    loaded = [r for r in rows if r["t"] >= 0]  # samples during or after a load (t from the load start)
    n36 = sum(r["b36"] for r in loaded)
    print(f"RESULT throttle samples {len(rows)} (after load start {len(loaded)}) with bit 36: {n36}, with PPT0: "
          f"{sum(r['ppt0'] for r in loaded)}")
    for name in sorted({r['file'] for r in rows}):
        rr = [r for r in loaded if r["file"] == name]
        b = [r for r in rr if r["b36"]]
        if b:
            print(f"RESULT throttle {name.rsplit('/', 1)[-1]}: bit 36 in {len(b)} of {len(rr)} samples, at t "
                  f"{min(r['t'] for r in b):.2f}-{max(r['t'] for r in b):.2f} s, hotspot {min(r['hot'] for r in b):.0f}-"
                  f"{max(r['hot'] for r in b):.0f} C, rise {min(r['dhot'] for r in b):.0f}-{max(r['dhot'] for r in b):.0f} C/s, "
                  f"power {min(r['power'] for r in b):.0f}-{max(r['power'] for r in b):.0f} W")
        else:
            print(f"RESULT throttle {name.rsplit('/', 1)[-1]}: bit 36 in 0 of {len(rr)} samples")
    lab = [r["b36"] for r in loaded]
    for var in ("hot", "dhot", "power", "sclk", "t"):
        print(f"RESULT throttle bit 36 AUC by {var}: {auc([r[var] for r in loaded], lab):.3f}")
    # share of samples with the bit by hotspot band and by rise band
    for var, edges in (("hot", [0, 60, 70, 80, 85, 90, 200]), ("dhot", [-1e9, 0, 20, 50, 100, 1e9]), ("t", [0, 0.1, 0.25, 0.5, 1, 2, 1e9])):
        parts = []
        for lo, hi in zip(edges, edges[1:]):
            s = [r for r in loaded if lo <= r[var] < hi]
            if s:
                parts.append(f"[{lo:g},{hi:g}): {sum(r['b36'] for r in s)}/{len(s)}")
        print(f"RESULT throttle bit 36 share by {var}: " + "; ".join(parts))
    # control
    rng = random.Random(7)
    shuf = [r["ppt0"] for r in loaded]
    rng.shuffle(shuf)
    a_ppt = auc([r["power"] for r in loaded], shuf if brk else [r["ppt0"] for r in loaded])
    a_shuf = auc([r["power"] for r in loaded], shuf)
    ok = a_ppt >= 0.9 and 0.4 <= a_shuf <= 0.6
    print(f"VERDICT throttle_control: {'PASS' if ok else 'FAIL'} PPT0 by reported power AUC {a_ppt:.3f} (>= 0.9 expected"
          f"{'; break: shuffled labels' if brk else ''}), with shuffled labels {a_shuf:.3f} (0.4 .. 0.6 expected)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
