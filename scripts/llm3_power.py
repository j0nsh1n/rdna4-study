#!/usr/bin/env python3
"""Round 3, Q1c: power of the three streaming mixes of src/power_mix.hip from the 20 ms sampler, windowed by the probe's own loop markers.
  llm3_power.py table <run folder>     reads <run>/r3m-<variant>-r<k>.log (LOOP_START / LOOP_END mono_s, RESULT line) and <run>/r3m/<variant>-r<k>/samples.csv
  llm3_power.py selftest
Per job: the samples inside [LOOP_START + 1 s, LOOP_END] (the first second lets the clock settle), median reported watts, median core clock, share of samples with a power-limit
throttle bit (indep_throttle_status non-zero), median GPU and memory busy. Per variant the median over its repetitions, with min and max. Differences use the repetitions
in pairs (same repetition number, taken in order fp8, q8, read), so a drifting card state cancels. Reported watts are scaled down by the power-limit mod: differences only.
A job whose window has fewer than 100 samples or whose bandwidth line is missing is refused (exit 1)."""
import csv
import glob
import os
import re
import statistics
import sys


def job(logp, sampp):
    txt = open(logp).read()
    m0 = re.search(r"LOOP_START mono_s ([0-9.]+)", txt)
    m1 = re.search(r"LOOP_END mono_s ([0-9.]+)", txt)
    mr = re.search(r"RESULT (\w+) launches \d+ loop_s [0-9.]+ GBps_median ([0-9.]+)", txt)
    if not (m0 and m1 and mr):
        raise ValueError(f"{logp}: no loop markers or no RESULT line")
    t0, t1 = float(m0.group(1)) + 1.0, float(m1.group(1))
    p, clk, thr, gb, mb = [], [], 0, [], []
    n = 0
    with open(sampp) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(hdr)}
    for r in rows:
        t = int(r[ix["mono_ns"]]) / 1e9
        if t0 <= t <= t1:
            n += 1
            p.append(float(r[ix["power_W"]]))
            clk.append(float(r[ix["sclk_MHz"]]))
            gb.append(float(r[ix["busy_pct"]]))
            mb.append(float(r[ix["mem_busy_pct"]]))
            thr += int(r[ix["indep_throttle_status"]], 16) != 0
    if n < 100:
        raise ValueError(f"{sampp}: only {n} samples inside the loop window")
    return {"variant": mr.group(1), "gbps": float(mr.group(2)), "w": statistics.median(p), "mhz": statistics.median(clk), "thr": thr / n, "busy": statistics.median(gb),
            "membusy": statistics.median(mb), "n": n}


def table(run):
    jobs = {}
    for logp in sorted(glob.glob(os.path.join(run, "r3m-*-r*.log"))):
        m = re.match(r".*r3m-(\w+)-r(\d+)\.log$", logp)
        if not m:
            continue
        v, k = m.group(1), int(m.group(2))
        sampp = os.path.join(run, "r3m", f"{v}-r{k}", "samples.csv")
        if not os.path.exists(sampp):
            continue
        try:
            jobs.setdefault(v, {})[k] = job(logp, sampp)
        except ValueError as e:
            print(f"FAIL {e}")
            return 1
    if not jobs:
        print("FAIL no r3m jobs found")
        return 1
    for v, d in sorted(jobs.items()):
        r = list(d.values())
        f = lambda key: (statistics.median(x[key] for x in r), min(x[key] for x in r), max(x[key] for x in r))
        w, g, c, t = f("w"), f("gbps"), f("mhz"), f("thr")
        print(f"RESULT variant {v} reps {len(r)} GBps_median {g[0]:.1f} [{g[1]:.1f} to {g[2]:.1f}] power_W_median {w[0]:.1f} [{w[1]:.1f} to {w[2]:.1f}] core_MHz_median {c[0]:.0f} [{c[1]:.0f} to {c[2]:.0f}] "
              f"power_limit_share {t[0]:.2f} [{t[1]:.2f} to {t[2]:.2f}] gpu_busy_median {statistics.median(x['busy'] for x in r):.0f} mem_busy_median {statistics.median(x['membusy'] for x in r):.0f}")
    for a, b in (("fp8", "q8"), ("fp8", "read"), ("q8", "read"), ("fp8row", "q8row"), ("fp8row", "read"), ("q8row", "read"), ("fp8row", "fp8"), ("q8row", "q8"),
                 ("fp8rowR", "q8rowR"), ("fp8rowR", "readR"), ("q8rowR", "readR"), ("fp8rowR", "fp8row"), ("q8rowR", "q8row"), ("readR", "read")):
        if a in jobs and b in jobs:
            ks = sorted(set(jobs[a]) & set(jobs[b]))
            dw = [jobs[a][k]["w"] - jobs[b][k]["w"] for k in ks]
            dg = [100 * (jobs[a][k]["gbps"] / jobs[b][k]["gbps"] - 1) for k in ks]
            dc = [jobs[a][k]["mhz"] - jobs[b][k]["mhz"] for k in ks]
            if dw:
                print(f"RESULT pair {a}-{b} reps {len(ks)} power_diff_W median {statistics.median(dw):.1f} [{min(dw):.1f} to {max(dw):.1f}] bandwidth_diff_pct median {statistics.median(dg):.1f} [{min(dg):.1f} to {max(dg):.1f}] core_MHz_diff median {statistics.median(dc):.0f}")
    return 0


def selftest():
    import tempfile
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", ".")) as td:
        def make(v, k, watts, gbps, nsamp=300, markers=True):
            base = 1000.0
            with open(os.path.join(td, f"r3m-{v}-r{k}.log"), "w") as f:
                if markers:
                    f.write(f"LOOP_START mono_s {base}\nLOOP_END mono_s {base + 10}\n")
                f.write(f"RESULT {v} launches 100 loop_s 10.0 GBps_median {gbps} min 1 max 2 groups 10\n")
            d = os.path.join(td, "r3m", f"{v}-r{k}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "samples.csv"), "w") as f:
                f.write("t_s,mono_ns,sclk_MHz,power_W,busy_pct,mem_busy_pct,indep_throttle_status\n")
                for i in range(nsamp):
                    f.write(f"{i * 0.03},{int((base + 0.5 + i * 0.03) * 1e9)},3300,{watts},99,80,0x0\n")
        for k in (1, 2, 3):
            make("fp8", k, 300, 600)
            make("q8", k, 250, 600)
        r = table(td)
        print(("PASS" if r == 0 else "FAIL") + " clean data accepted")
        fails += r != 0
        make("read", 1, 200, 700, nsamp=30)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r2 = table(td)
        out = buf.getvalue()
        print(("PASS" if r2 == 1 else "FAIL") + " break: a job with 30 samples inside the window is refused")
        fails += r2 != 1
        os.remove(os.path.join(td, "r3m-read-r1.log"))
        make("read", 1, 200, 700, markers=False)
        with contextlib.redirect_stdout(io.StringIO()):
            r3 = table(td)
        print(("PASS" if r3 == 1 else "FAIL") + " break: a log without loop markers is refused")
        fails += r3 != 1
        import shutil
        shutil.rmtree(os.path.join(td, "r3m", "read-r1"))
        os.remove(os.path.join(td, "r3m-read-r1.log"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            table(td)
        ok = "power_diff_W median 50.0" in buf.getvalue()
        print(("PASS" if ok else "FAIL") + " known answer: 300 W against 250 W gives +50.0 W in the paired difference")
        fails += not ok
    print("VERDICT llm3_power selftest " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        sys.exit(selftest())
    if a[:1] == ["table"] and len(a) == 2:
        sys.exit(table(a[1]))
    print(__doc__)
    sys.exit(2)
