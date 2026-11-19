#!/usr/bin/env python3
"""Round 3, Q1b: per-kernel medians of the counters that do work (SQ_WAVES, SQ_BUSY_CYCLES, SIMD_UTILIZATION, GRBM_*) from rocprofv3 counter-collection CSVs.
  llm3_cnt.py table <label> <csv> [<csv> ...] --kernel REGEX      one row per (counter): dispatches, median value, median duration (ns), median value per us
  llm3_cnt.py selftest
Several CSVs (one per counter set, same run) are merged by counter name. Exit 1 if no dispatch matches the kernel regex (a check that cannot pass on nothing)."""
import csv
import re
import statistics
import sys


def read(paths, kernel):
    rx = re.compile(kernel)
    rows = {}
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if rx.search(r["Kernel_Name"]):
                    dur = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
                    rows.setdefault(r["Counter_Name"], []).append((float(r["Counter_Value"]), dur))
    return rows


def table(label, paths, kernel):
    rows = read(paths, kernel)
    if not rows:
        print(f"FAIL {label}: no dispatch matches /{kernel}/")
        return 1
    for c, v in sorted(rows.items()):
        vals = [x for x, _ in v]
        durs = [d for _, d in v]
        med, dmed = statistics.median(vals), statistics.median(durs)
        print(f"RESULT {label} {c} dispatches {len(v)} median {med:.6g} min {min(vals):.6g} max {max(vals):.6g} duration_ns_median {dmed:.0f} value_per_us {med / (dmed / 1000):.6g}")
    return 0


def selftest():
    import os
    import tempfile
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", ".")) as td:
        p = os.path.join(td, "c.csv")
        with open(p, "w") as f:
            f.write('"Kernel_Name","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"\n')
            for i in range(5):
                f.write(f'"mul_mat_vec_f<fp8>",SQ_WAVES,{100 + i},{1000 * i},{1000 * i + 500}\n')
                f.write(f'"other",SQ_WAVES,7,{1000 * i},{1000 * i + 100}\n')
        rows = read([p], "mul_mat_vec_f")
        ok = list(rows) == ["SQ_WAVES"] and statistics.median(v for v, _ in rows["SQ_WAVES"]) == 102 and all(d == 500 for _, d in rows["SQ_WAVES"])
        print(("PASS" if ok else "FAIL") + " known answer: median 102 over the five matching dispatches, the other kernel excluded")
        fails += not ok
        rc = table("selftest", [p], "no_such_kernel")
        ok = rc == 1
        print(("PASS" if ok else "FAIL") + " break: a regex matching nothing fails")
        fails += not ok
    print("VERDICT llm3_cnt selftest " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        sys.exit(selftest())
    if a[:1] == ["table"] and "--kernel" in a:
        k = a[a.index("--kernel") + 1]
        paths = [x for x in a[2:] if x != "--kernel" and x != k]
        sys.exit(table(a[1], paths, k))
    print(__doc__)
    sys.exit(2)
