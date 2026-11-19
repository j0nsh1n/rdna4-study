#!/usr/bin/env python3
"""Tables of measurements/gaming-probes.md from one run folder of scripts/study-gaming-probes.sh.

Usage: python3 -I scripts/gprobe_tables.py runs/<UTC time>-gprobe

Reads the DIST, RESULT, PIECES, STATE, BURST, BIN, RAMP and INFO lines of the job logs and prints markdown tables.
Nothing is computed that the logs do not print, except the selection and the rounding.
"""
import glob
import os
import re
import sys

DIST = re.compile(r"^DIST (\S+) (\S+) n=(\d+)(?: min=(\S+) p50=(\S+) p95=(\S+) p99=(\S+) max=(\S+) max_at=(\S+))?")


def logs(run):
    out = {}
    for p in sorted(glob.glob(os.path.join(run, "*.log"))):
        with open(p, errors="replace") as f:
            out[os.path.basename(p)[:-4]] = f.read().splitlines()
    return out


def dists(lines):
    d = {}
    for ln in lines:
        m = DIST.match(ln)
        if m and m.group(4):
            d[(m.group(1), m.group(2))] = dict(n=int(m.group(3)), min=float(m.group(4)), p50=float(m.group(5)),
                                              p95=float(m.group(6)), p99=float(m.group(7)), max=float(m.group(8)),
                                              at=int(m.group(9)))
    return d


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def kv(line, key):
    m = re.search(rf"\b{re.escape(key)}=(\S+)", line)
    return m.group(1) if m else None


def p1(L):
    print("\n### P1 conditions (one row per condition; us unless stated; gap = next dispatch's first entry minus "
          "this dispatch's last exit)\n")
    print("| job | condition | spec | clock MHz p50 | dispatch us p50 | gap p50 / p95 / p99 / max (n) | timestamps per "
          "dispatch p50 | list wall p50 / p99 |")
    print("|---|---|---|---|---|---|---|---|")
    for job in sorted(j for j in L if j.startswith("p1-") and "dump" not in j):
        D = dists(L[job])
        for ln in L[job]:
            if not ln.startswith("COND "):
                continue
            name = ln.split()[1]
            spec = (kv(ln, "spec") or "").replace(f"name={name},", "")
            g = D.get((name, "gap_to_next_us(next_first_entry_minus_last_exit)"))
            c = D.get((name, "clock_MHz(cycles/REALTIME_per_group)"))
            du = D.get((name, "dispatch_dur_us(first_entry_to_last_exit)"))
            t = D.get((name, "list_ts_per_dispatch_us"))
            w = D.get((name, "list_wall_submit_to_fence_us"))
            gs = f"{f(g['p50'])} / {f(g['p95'])} / {f(g['p99'])} / {f(g['max'])} ({g['n']})" if g else "-"
            print(f"| {job} | {name} | `{spec}` | {f(c and c['p50'], 0)} | {f(du and du['p50'])} | {gs} | "
                  f"{f(t and t['p50'], 3)} | {f(w and w['p50'], 1)} / {f(w and w['p99'], 1)} |")
    for job in sorted(j for j in L if j.startswith("p1-")):
        for ln in L[job]:
            if ln.startswith(("FIT ", "INFO cal", "RESULT dispatch back-to-back")):
                print(f"\n    {job}: {ln}")


def p2(L):
    print("\n### P2 one-group probes (wall = submit to fence seen by the host, us)\n")
    print("| job | arm and queue | n | wall p50 / p95 / p99 / max | share above 60 us | GPU top-bottom p50 |")
    print("|---|---|---|---|---|---|")
    for job in sorted(j for j in L if j.startswith("p2-") and j != "p2-lone-procs"):
        D = dists(L[job])
        for ln in L[job]:
            if ln.startswith("RESULT ") and "share_above_60us" in ln:
                tag = ln.split()[1]
                w = D.get((tag, "wall_us(submit_to_fence)"))
                gp = D.get((tag, "gpu_us(top_to_bottom_ts)"))
                if not w:
                    continue
                print(f"| {job} | {tag} | {w['n']} | {f(w['p50'],1)} / {f(w['p95'],1)} / {f(w['p99'],1)} / "
                      f"{f(w['max'],1)} | {kv(ln, 'share_above_60us')} | {f(gp and gp['p50'])} |")
    print("\nWhere the time goes (median us: submit -> first timestamp -> shader entry -> shader exit -> end-of-pipe "
          "timestamp -> host wakes; the probe uses at most the first 4096 samples of each class here):\n")
    for job in sorted(j for j in L if j.startswith("p2-")):
        for ln in L[job]:
            if ln.startswith("PIECES "):
                print(f"    {job}: {ln[7:]}")
    print("\nFrames and the in-frame list:\n")
    for job in sorted(j for j in L if j.startswith("p2-")):
        D = dists(L[job])
        for (tag, what), v in sorted(D.items()):
            if what.startswith(("frame_", "inframe_")):
                print(f"    {job}: {tag} {what}: n {v['n']} p50 {v['p50']:.2f} p95 {v['p95']:.2f} max {v['max']:.2f}")
        for ln in L[job]:
            if ln.startswith(("INFO draw", "STATE d", "STATE in", "STATE b2b", "RESULT d", "RESULT in")) or \
                    (ln.startswith("INFO ") and "pass GPU time" in ln):
                print(f"    {job}: {ln}")
    if "p2-lone-procs" in L:
        print("\nSix short processes (300 submissions per queue each):\n")
        for ln in L["p2-lone-procs"]:
            if ln.startswith(("PROCESS", "RESULT lone-alone", "PIECES lone-alone-compute")):
                print(f"    {ln}")


def p3(L):
    print("\n### P3 preemption pairs\n")
    for job in sorted(j for j in L if j.startswith("p3-")):
        for ln in L[job]:
            if ln.startswith(("RESULT preempt A_shape", "RESULT preempt summary", "RESULT preempt T_A",
                              "VERDICT preempt", "INFO preempt")):
                print(f"    {job}: {ln}")


def p4(L):
    print("\n### P4 bursts\n")
    print("| job | first burst median MHz | steady bursts: median clock p50 (p5 bin..p95 bin) | burst GPU ms p50 | "
          "submit to start us p50 | hotspot max | reported power max |")
    print("|---|---|---|---|---|---|---|")
    for job in sorted(j for j in L if j.startswith("p4-cad")):
        D = dists(L[job])
        name = next((k[0] for k in D), None)
        fb = next((ln for ln in L[job] if ln.startswith("INFO") and "first burst" in ln), "")
        st = next((ln for ln in L[job] if ln.startswith("STATE ") and "whole-job" not in ln), "")
        c = D.get((name, "steady_burst_median_clock_MHz"))
        g = D.get((name, "steady_burst_gpu_ms(bursts_10+)"))
        s = D.get((name, "steady_submit_to_start_us"))
        hot = re.search(r"hotspot_C\(min/max\)=\S+/(\S+)", st)
        pw = re.search(r"power_W_reported\(min/max\)=\S+/(\S+)", st)
        m = re.search(r"median clock (\d+) MHz", fb)
        print(f"| {job} | {m.group(1) if m else '-'} | {f(c and c['p50'], 0)} ({f(c and c['min'], 0)}..{f(c and c['max'], 0)}) "
              f"| {f(g and g['p50'], 3)} | {f(s and s['p50'], 1)} | {hot.group(1) if hot else '-'} | "
              f"{pw.group(1) if pw else '-'} |")
    for job in sorted(j for j in L if j.startswith("p4-")):
        for ln in L[job]:
            if ln.startswith(("RESULT", "RAMP", "BIN", "INFO")) or (ln.startswith("BURST") and " i=0 " in ln):
                print(f"    {job}: {ln}")


def verdicts(L):
    print("\n### Verdicts\n")
    for job in sorted(L):
        for ln in L[job]:
            if ln.startswith("VERDICT"):
                print(f"    {job}: {ln[:300]}")


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print(__doc__)
        return 2
    L = logs(sys.argv[1])
    print(f"# Tables from {os.path.basename(os.path.normpath(sys.argv[1]))}")
    p1(L)
    p2(L)
    p3(L)
    p4(L)
    verdicts(L)
    return 0


if __name__ == "__main__":
    sys.exit(main())
