#!/usr/bin/env python3
"""Lane 7: prints the markdown tables of measurements/scheduling-and-clocks.md from one run folder's logs.

Usage: sched_tables.py <run folder> [<another run folder> ...]   (later folders fill tables the first lacks)
Only reads the logs; every number in a table is copied from a RESULT, WINDOW, COOL or EVENT line.
"""
import os
import re
import sys


def lines(folders, name):
    for d in folders:
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                return f.read().splitlines()
    return []


def kv(s):
    return dict(re.findall(r"(\w[\w./%()-]*)=([^\s|]+)", s))


def dist(s, key):
    """'key median a p10 b p90 c min d max e n f' -> dict."""
    m = re.search(re.escape(key) + r" median (\S+) p10 (\S+) p90 (\S+) min (\S+) max (\S+) n (\d+)", s)
    if not m:
        return None
    return dict(zip(["med", "p10", "p90", "min", "max", "n"], m.groups()))


def fmt(d, keys=("med", "p10", "p90", "max")):
    if not d:
        return "-"
    return " / ".join(d[k] for k in keys)


def item1(F):
    L = lines(F, "launch.log")
    print("### Item 1: launch + hipDeviceSynchronize, empty kernel (wall us median / p10 / p90 / max; GPU event us median / p10 / p90 / max)\n")
    print("| Grid (groups) | Block | Wall | GPU (events) | n |")
    print("|---|---|---|---|---|")
    for s in L:
        if s.startswith("RESULT launch empty grid="):
            g = re.search(r"grid=(\d+) block=(\d+)", s)
            w, e = dist(s, "wall_us"), dist(s, "gpu_us")
            print(f"| {g.group(1)} | {g.group(2)} | {fmt(w)} | {fmt(e)} | {w['n']} |")
    print()
    for s in L:
        if s.startswith("RESULT launch") and "grid=" not in s:
            print("    " + s)
    print()
    I = lines(F, "launch-idle.log")
    print("### Item 1: the same one-wave launch after idle\n")
    print("| Idle before | Wall us (median / p10 / p90 / max) | GPU us (median / p10 / p90 / max) | Clock in the kernel MHz (median / p10 / p90 / max) | n |")
    print("|---|---|---|---|---|")
    for s in I:
        m = re.match(r"RESULT idle (\S+) onewave", s)
        if m:
            w, e, c = dist(s, "wall_us"), dist(s, "gpu_us"), dist(s, "body_clock_MHz")
            print(f"| {m.group(1).replace('_', ' ')} | {fmt(w)} | {fmt(e)} | {fmt(c)} | {w['n']} |")
    print()
    print("### Item 1: clock ramp seen from inside (cycle counter / REALTIME per window)\n")
    for name in ("ramp-onewave-idle10.log", "ramp-onewave-idle100.log", "ramp-onewave-idle1000.log", "ramp-card-idle1000.log"):
        R = lines(F, name)
        for s in R:
            if s.startswith("RESULT ramp"):
                print("    " + s)
    print()
    print("| t from kernel start (ms) | one wave, idle 10 ms | one wave, idle 100 ms | one wave, idle 1000 ms | whole card, idle 1000 ms |")
    print("|---|---|---|---|---|")
    cols = []
    for name in ("ramp-onewave-idle10.log", "ramp-onewave-idle100.log", "ramp-onewave-idle1000.log", "ramp-card-idle1000.log"):
        d = {}
        for s in lines(F, name):
            m = re.search(r"t_ms=([\d.]+) clock_MHz=(\d+)", s)
            if s.startswith("RAMPWIN") and m:
                d[float(m.group(1))] = m.group(2)
        cols.append(d)
    for t in (0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 30, 50, 100, 145):
        print(f"| {t} | " + " | ".join(c.get(float(t), "-") for c in cols) + " |")
    print()
    for name in ("ramp-card-idle1000.log",):
        print("    sysfs during the whole-card ramp (5 ms samples, every 3rd):")
        k = 0
        for s in lines(F, name):
            if s.startswith("SYSFS"):
                if k % 3 == 0:
                    print("    " + s)
                k += 1
    print()


def item2(F):
    print("### Item 2: placement (from every wave's own record)\n")
    print("| Mode | Grid | Block | SEs | WGPs | CUs | SIMDs | max waves per SIMD | max groups per CU | max groups per WGP | SIMDs per group | first 64 groups started within (us) | span (us) | slot gap us (median / p90 / n) | same CU as launch 0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("place-wgp.log", "place-cu.log"):
        for s in lines(F, name):
            if s.startswith("RESULT place"):
                p = [x.strip() for x in s.split("|")]
                head = p[0].split()
                g = re.search(r"grid=(\d+) block=(\d+)", p[0])
                a = p[1].split()
                b = p[2].split()
                c = p[3].split()
                d = p[4].split()
                print(f"| {head[2]} | {g.group(1)} | {g.group(2)} | {' | '.join(a)} | {' | '.join(b)} | {c[0]} | {c[1]} | {d[0]} / {d[1]} / {d[2]} | {p[5]} |")
    print()
    for name in ("place-wgp.log", "place-cu.log"):
        for s in lines(F, name):
            if s.startswith("ORDER"):
                print("    " + s[:600])
    print()
    print("### Item 2: registers against waves per SIMD (256-thread groups, grid 4096)\n")
    print("| Mode | live floats | VGPRs (numRegs) | rounded to 24 | rule min(16, 1536 / rounded) | step (whole groups) | expected | measured |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ("regs-wgp.log", "regs-cu.log"):
        for s in lines(F, name):
            if s.startswith("RESULT regs"):
                k = kv(s)
                mode = s.split()[2]
                print(f"| {mode} | {k['live_floats']} | {k['numRegs']} | {k['rounded']} | {k['rule_waves_per_SIMD']} | {k['group_step']} | {k['expected']} | {k['measured_max_waves_per_SIMD']} |")
    print()
    print("### Item 2: throughput against compute units (one group per CU, CU mode)\n")
    print("| Groups = CUs | FMA TFMA/s | FMA per CU | Stream GB/s | Stream per CU |")
    print("|---|---|---|---|---|")
    fma, st = {}, {}
    for s in lines(F, "scale-cu.log"):
        k = kv(s)
        if " fma " in s:
            fma[k["groups"]] = k
        if " stream " in s:
            st[k["groups"]] = k
    for g in fma:
        print(f"| {g} | {fma[g]['TFMA_s']} | {fma[g]['per_CU']} | {st.get(g, {}).get('GB_s', '-')} | {st.get(g, {}).get('per_CU', '-')} |")
    for s in lines(F, "scale-cu.log"):
        if "sysfs" in s or s.startswith("VERDICT"):
            print("    " + s)
    print()
    print("### Item 8: hardware ID fields over a launch that covers the card\n")
    for s in lines(F, "ids.log"):
        if s.startswith("RESULT ids") or s.startswith("VERDICT"):
            print("    " + s)
    print()


def item5(F):
    print("### Item 5: traces\n")
    for m in ("idle", "light", "stream", "fma", "f16", "fp8"):
        L = lines(F, f"trace-{m}.log")
        if not L:
            continue
        print(f"#### {m}\n")
        for s in L:
            if s.startswith(("RESULT trace", "WINDOW", "COOL", "VERDICT", "THERMAL", "START")):
                print("    " + s)
        print()


def item7(F):
    print("### Item 7a: one wave against the desktop\n")
    for s in lines(F, "spin-desktop.log"):
        if s.startswith(("RESULT", "VERDICT", "HIST")) and not s.endswith(": 0"):
            print("    " + s)
    print()
    for s in [x for x in lines(F, "spin-desktop.log") if x.startswith("EVENT")][:12]:
        print("    " + s)
    print()
    print("### Item 7b: two processes\n")
    for name in ("pair-big-alone.log", "pair-big-big.log", "pair-part-part.log", "pair-small-big.log", "pair-small-small.log"):
        print(f"#### {name}\n")
        for s in lines(F, name):
            if s.startswith(("RESULT", "VERDICT", "RUNS")):
                print("    " + s[:700])
        print()


def helpers(F):
    for name in ("barrier-wgp.log", "barrier-cu.log", "wave32.log", "wave64.log", "vksched.log"):
        L = lines(F, name)
        if not L:
            continue
        print(f"### {name}\n")
        for s in L:
            if s.startswith(("RESULT", "VERDICT")) and "HIST" not in s:
                print("    " + s[:400])
        print()


def main():
    F = sys.argv[1:]
    if not F:
        print(__doc__)
        return 2
    item1(F)
    item2(F)
    item5(F)
    item7(F)
    helpers(F)
    return 0


if __name__ == "__main__":
    sys.exit(main())
