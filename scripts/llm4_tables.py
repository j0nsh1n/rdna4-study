#!/usr/bin/env python3
"""Round 4: tables for llm-runtimes.md, built from the raw run files only (no number is typed by hand). python3 -I.
  llm4_tables.py tables <run folder> [--uclk 1258]    markdown tables on stdout
  llm4_tables.py selftest
A job counts for a table only if its .meta says the memory clock was --uclk before and after (jobs at another clock are listed apart)."""
import glob, json, os, re, statistics, sys


def meta(run, name):
    p = os.path.join(run, name + ".meta")
    d = {}
    if os.path.exists(p):
        for l in open(p):
            if "=" in l:
                k, v = l.rstrip("\n").split("=", 1)
                d[k] = v
    return d


def good(run, name, uclk):
    m = meta(run, name)
    return m.get("exit") == "0" and m.get("uclk_before") == str(uclk) and m.get("uclk_after") == str(uclk)


def bench_rows(log):
    out = {}
    for l in open(log):
        if l.startswith("{"):
            d = json.loads(l)
            key = f"pp{d['n_prompt']}" if d["n_gen"] == 0 else f"tg{d['n_gen']}"
            out[key] = (statistics.median(d["samples_ts"]), min(d["samples_ts"]), max(d["samples_ts"]))
    return out


def batched(log):
    rows = {}
    for l in open(log):
        c = [x.strip() for x in l.split("|")]
        if len(c) > 9 and c[1].isdigit() and c[3].isdigit():
            rows[int(c[3])] = (float(c[6]), float(c[8]))  # S_PP, S_TG
    return rows


def result(log):
    for l in open(log):
        if l.startswith("RESULT "):
            return json.loads(l[7:])
    return None


def tbl(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]) + "\n"


def kld(log):
    t = open(log).read()
    g = lambda rx: (re.search(rx, t) or [None, None])[1]
    return {"ppl": g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), "mean": g(r"Mean\s+KLD:\s*([\d.]+)"), "max": g(r"Maximum KLD:\s*([\d.]+)"),
            "p99": g(r"99\.0%\s+KLD:\s*([\d.]+)"), "top": g(r"Same top p:\s*([\d.]+)")}


def ppl_llama(log):
    m = re.findall(r"Final estimate: PPL = ([\d.]+)", open(log).read())
    return m[-1] if m else None


def tables(run, uclk):
    L = []
    names = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(run, "*.meta")))
    other = [n for n in names if meta(run, n).get("uclk_before") not in (None, str(uclk)) or meta(run, n).get("uclk_after") not in (None, str(uclk))]
    L.append(f"Memory clock {uclk} MHz before and after every job in the tables below; {len(other)} jobs of the run ran at another clock and are not used here.\n")
    # llama.cpp speed
    rows = []
    for n in names:
        if n.startswith("r4k-bench-") and good(run, n, uclk):
            lab = n[len("r4k-bench-"):]
            b = bench_rows(os.path.join(run, n + ".log"))
            p = os.path.join(run, "r4k-pp2048-" + lab + ".log")
            b2 = bench_rows(p) if os.path.exists(p) and good(run, "r4k-pp2048-" + lab, uclk) else {}
            f = lambda d, k: f"{d[k][0]:.0f}" if k in d else "-"
            rows.append([lab, f(b, "pp128"), f(b, "pp512"), f(b2, "pp2048"), f"{b['tg128'][0]:.1f}" if "tg128" in b else "-"])
    L.append("### llama.cpp (tokens/s, median of 5)\n\n" + tbl(["model/build", "pp128", "pp512", "pp2048 (ub 2048)", "tg128"], rows))
    rows = []
    for n in names:
        if n.startswith("r4k-batched-") and good(run, n, uclk):
            b = batched(os.path.join(run, n + ".log"))
            rows.append([n[len("r4k-batched-"):]] + [f"{b[k][1]:.0f}" if k in b else "-" for k in range(1, 9)])
    L.append("### llama.cpp parallel streams (S_TG tokens/s, 128 prompt + 128 generated per stream)\n\n" + tbl(["model/build"] + [str(k) for k in range(1, 9)], rows))
    # torch and vllm speed
    rows = []
    for n in names:
        if re.match(r"r4(t-pp|t-tg|v)-", n) and good(run, n, uclk):
            r = result(os.path.join(run, n + ".log"))
            if not r:
                continue
            pf = {k: f"{v['tok_s_median']:.0f}" for k, v in r.get("prefill", {}).items()}
            dc = {k: f"{v['tok_s_median']:.1f}" for k, v in r.get("decode", {}).items()}
            rows.append([n, json.dumps(pf) if pf else "-", " ".join(f"{k}:{v}" for k, v in dc.items()) or "-"])
    L.append("### PyTorch and vLLM (tokens/s, medians)\n\n" + tbl(["job", "prefill tokens", "decode streams:tokens/s"], rows))
    # gemm
    for n in names:
        if n.startswith("r4t-gemm-") and good(run, n, uclk):
            r = result(os.path.join(run, n + ".log"))
            rows = [[k, f"{v['fp8']['tflops']:.1f}" if "fp8" in v else "-", f"{v['bf16']['tflops']:.1f}"] for k, v in r["shapes"].items()]
            L.append(f"### GEMM rates, {n} (TFLOPS, median of 50)\n\n" + tbl(["shape", "FP8 _scaled_mm", "bf16 matmul"], rows))
    # quality
    rows = []
    for n in names:
        if n.startswith("r4q-ppl4-") and good(run, n, uclk):
            rows.append([n, ppl_llama(os.path.join(run, n + ".log"))])
        if re.match(r"r4q-[tv]-ppl4-", n) and good(run, n, uclk):
            r = result(os.path.join(run, n + ".log"))
            if r:
                rows.append([n, f"{r['ppl']:.4f}"])
    L.append("### perplexity, 4 chunks of 512 (second half of each chunk scored)\n\n" + tbl(["job", "ppl"], rows))
    rows = []
    for n in names:
        if n.startswith("r4q-kld-") and good(run, n, uclk):
            k = kld(os.path.join(run, n + ".log"))
            rows.append([n, k["ppl"], k["mean"], k["p99"], k["max"], k["top"]])
        if re.match(r"r4q-[tv]-kl-", n) and good(run, n, uclk):
            r = result(os.path.join(run, n + ".log"))
            if r:
                rows.append([n, f"{r['ppl']:.4f}",  (f"{r['kl_mean']:.6f}" if 'kl_mean' in r else "- (no full distributions)"), "-", (f"{r['kl_max']:.4f}" if 'kl_max' in r else "-"),
                             (f"{r['same_top_pct']:.3f}" if 'same_top_pct' in r else "-") + (f" (mean |dlogp target| {r['mean_abs_dlogp_target']:.5f})" if 'mean_abs_dlogp_target' in r else "")])
    L.append("### distance from the true (F32) weights\n\n" + tbl(["job", "ppl", "mean KLD", "99 % KLD", "max KLD", "same top token %"], rows))
    # traces
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "r4tr-*.analysis.json"))):
        a = json.load(open(f))
        g = a["groups"]
        top = ", ".join(f"{k} {100 * v['share_of_kernel_time']:.0f} %" for k, v in list(g.items())[:4])
        rows.append([os.path.basename(f)[:-14], f"{a['wall_ms_per_pass']:.2f}", f"{a['kernels_per_pass']:.0f}", f"{100 * a['gap_share']:.1f}", f"{a['median_gap_us']:.2f}", top])
    L.append("### kernel traces (per pass or per token)\n\n" + tbl(["trace", "wall ms", "kernels", "gaps %", "median gap us", "largest groups (share of kernel time)"], rows))
    # sampler
    rows = []
    for d in sorted(glob.glob(os.path.join(run, "beh", "r4b-*"))):
        sp = os.path.join(d, "summary.txt")
        if not os.path.isdir(d) or not os.path.exists(sp):
            continue
        t = open(sp).read()
        g = lambda rx: (re.search(rx, t) or [None, "?"])[1]
        m = re.search(r"SCLK_MHz awake \(time-weighted\) p10 (\d+) p50 (\d+) p90 (\d+) p99 (\d+) max (\d+)", t)
        pw = re.search(r"POWER_W reported[^\n]* p10 (\d+) p50 (\d+) p90 (\d+) p99 (\d+) max (\d+)", t)
        th = re.search(r"THROTTLE indep_throttle_status: ([^\n]*)", t)
        rows.append([os.path.basename(d), g(r"SAMPLES (\d+)"), f"{m.group(2)} / {m.group(5)}" if m else "?", f"{pw.group(2)} / {pw.group(5)}" if pw else "?",
                     g(r"HOTSPOT_C max (\d+)"), th.group(1) if th else "?", g(r"UCLK_MHz[^\n]* p50 (\d+)")])
    L.append("### clocks, power, hotspot (20 ms sampler; core MHz awake median / max; reported W median / max)\n\n" +
             tbl(["recording", "samples", "core MHz", "power W", "hotspot max C", "throttle bits", "memory MHz"], rows))
    return "\n".join(L)


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "r4k-bench-x.meta"), "w").write("uclk_before=1258\nuclk_after=1258\nexit=0\n")
    open(os.path.join(d, "r4k-bench-x.log"), "w").write(json.dumps({"n_prompt": 512, "n_gen": 0, "samples_ts": [10, 30, 20]}) + "\n" +
                                                         json.dumps({"n_prompt": 0, "n_gen": 128, "samples_ts": [5, 7, 6]}) + "\n")
    t = tables(d, 1258)
    assert "| x | - | 20 | - | 6.0 |" in t, t
    # a job at another memory clock must not appear in the tables (the break)
    open(os.path.join(d, "r4k-bench-x.meta"), "w").write("uclk_before=1408\nuclk_after=1408\nexit=0\n")
    t = tables(d, 1258)
    assert "| x |" not in t and "1 jobs of the run ran at another clock" in t, t
    # a failed job must not appear either
    open(os.path.join(d, "r4k-bench-x.meta"), "w").write("uclk_before=1258\nuclk_after=1258\nexit=8\n")
    assert "| x |" not in tables(d, 1258)
    assert kld.__name__ == "kld"
    print("VERDICT llm4_tables selftest PASS")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        print(tables(sys.argv[2], int(sys.argv[sys.argv.index("--uclk") + 1]) if "--uclk" in sys.argv else 1258))
