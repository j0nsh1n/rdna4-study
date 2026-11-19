#!/usr/bin/env python3
"""Round 3, Q3: the interleaved runs of three widen_cache binaries (old = round 2's, new = the fix round's, nocal = new without the 1 GiB known-answer read).
  llm3_q3.py table <run folder>     reads <run>/r3q3-<binary>-r<k>.log
  llm3_q3.py selftest
Per binary and row (widen_dst_reused, widen_dst_rotating, read_rotating, write_reused, write_rotating): median, min and max of the repetitions' GB/s; per repetition the ratio
new / old and nocal / old (paired by repetition number, so a drifting session cancels); verdict lines say whether the ranges of old and new overlap.
The round-2 binary prints S_MiB as 3 S for a widen row and has no VERDICT lines; the parser only needs the row name and the GBps field. Exit 1 if a binary has fewer than 6 usable repetitions."""
import glob
import os
import re
import statistics
import sys

ROWS = ("widen_dst_reused", "widen_dst_rotating", "read_rotating", "write_reused", "write_rotating")


def parse(path):
    out = {}
    for ln in open(path):
        m = re.match(r"RESULT (\w+) .*? GBps ([0-9.]+) ", ln + " ")
        if m and m.group(1) in ROWS:
            out[m.group(1)] = float(m.group(2))
    return out


def table(run, min_reps=6):
    data = {}
    for p in sorted(glob.glob(os.path.join(run, "r3q3-*-r*.log"))):
        m = re.match(r".*r3q3-(\w+)-r(\d+)\.log$", p)
        d = parse(p)
        if m and len(d) == len(ROWS):
            data.setdefault(m.group(1), {})[int(m.group(2))] = d
    bad = 0
    for b, reps in sorted(data.items()):
        if len(reps) < min_reps:
            print(f"FAIL {b}: {len(reps)} usable repetitions, at least {min_reps} needed")
            bad = 1
    if not data or bad:
        return 1
    for row in ROWS:
        print(f"-- {row} (GB/s)")
        for b, reps in sorted(data.items()):
            v = [r[row] for r in reps.values()]
            print(f"RESULT {row} {b} reps {len(v)} median {statistics.median(v):.1f} min {min(v):.1f} max {max(v):.1f} spread_pct {100 * (max(v) - min(v)) / statistics.median(v):.1f}")
        if "old" in data:
            for b in ("new", "nocal"):
                if b in data:
                    ks = sorted(set(data[b]) & set(data["old"]))
                    ratio = [100 * (data[b][k][row] / data["old"][k][row] - 1) for k in ks]
                    lo_o, hi_o = min(r[row] for r in data["old"].values()), max(r[row] for r in data["old"].values())
                    lo_b, hi_b = min(r[row] for r in data[b].values()), max(r[row] for r in data[b].values())
                    sep = "ranges separate" if (hi_b < lo_o or hi_o < lo_b) else "ranges overlap"
                    print(f"RESULT {row} {b}-against-old paired_diff_pct median {statistics.median(ratio):.1f} min {min(ratio):.1f} max {max(ratio):.1f} ({len(ks)} pairs; {sep})")
    return 0


def selftest():
    import tempfile
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", ".")) as td:
        for b, base in (("old", 660.0), ("new", 633.6), ("nocal", 660.0)):
            for k in range(1, 7):
                with open(os.path.join(td, f"r3q3-{b}-r{k}.log"), "w") as f:
                    for row in ROWS:
                        f.write(f"RESULT {row} S_MiB 75 moved_MiB 75 median_ms 0.1 GBps {base + k * 0.1:.1f} best_GBps 700.0 worst_GBps 600.0\n")
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = table(td)
        o = buf.getvalue()
        ok = rc == 0 and "new-against-old paired_diff_pct median -4.0" in o and "ranges separate" in o and "nocal-against-old paired_diff_pct median 0.0" in o
        print(("PASS" if ok else "FAIL") + " known answer: a planted 4 % gap is found (-4.0, ranges separate), nocal equals old (0.0)")
        fails += not ok
        os.remove(os.path.join(td, "r3q3-new-r6.log"))
        with contextlib.redirect_stdout(io.StringIO()):
            rc = table(td)
        ok = rc == 1
        print(("PASS" if ok else "FAIL") + " break: five repetitions of a binary are refused (six needed)")
        fails += not ok
    print("VERDICT llm3_q3 selftest " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        sys.exit(selftest())
    if a[:1] == ["table"] and len(a) == 2:
        sys.exit(table(a[1]))
    print(__doc__)
    sys.exit(2)
