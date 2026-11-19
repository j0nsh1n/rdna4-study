#!/usr/bin/env python3
"""Parsers, checks and table builders for the LLM data dive (stdlib only).

Usage:
  llm_tables.py tables <run folder>     writes the tables of measurements/llm-workload.md to stdout (Markdown)
  llm_tables.py streamed <model.gguf>   one `streamed` line for models.txt: tensor bytes, token_embd bytes, bytes decode streams
  llm_tables.py check-self <kld log>    the same-run-again KL divergence against its noise floor (exit 1 if above it)
  llm_tables.py selftest [scratch dir]  the helper's own checks and deliberate breaks (each must FAIL)

Rules built in (study rules): medians, never means; min and max next to them; a row names the log it comes from;
rows taken at different memory clocks are never merged (every builder calls one_clock, which raises on a mix); a row
whose model type does not match its label is refused (a Q8_0 result under an FP8 label).
"""
import glob
import json
import math
import os
import re
import statistics
import sys

TYPE_OF = {"fp8": "F8_E4M3", "q8_0": "Q8_0"}


class TableError(Exception):
    pass


def read_meta(log):
    meta = {}
    p = re.sub(r"\.log$", ".meta", log)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    meta[k] = v
    return meta


def parse_bench(path):
    """llama-bench jsonl rows of one job log."""
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            n_p, n_g, depth = d["n_prompt"], d["n_gen"], d.get("n_depth", 0)
            if n_p and not n_g:
                test = f"pp{n_p}"
            elif n_g and not n_p:
                test = f"tg{n_g}"
            else:
                test = f"pp{n_p}+tg{n_g}"
            if depth:
                test += f"@d{depth}"
            s = d["samples_ts"]
            rows.append(dict(test=test, n_prompt=n_p, n_gen=n_g, depth=depth, median=statistics.median(s), lo=min(s),
                             hi=max(s), n=len(s), model_type=d["model_type"], model_size=d["model_size"],
                             log=os.path.basename(path)))
    return rows


def check_label(rows, label):
    """label like '3b-fp8-unmod' or '8b-q8_0': every row must carry the model type of that label."""
    want = TYPE_OF["q8_0" if "q8_0" in label else "fp8"]
    bad = [r for r in rows if want not in r["model_type"]]
    if bad:
        raise TableError(f"{label}: log {bad[0]['log']} holds a '{bad[0]['model_type']}' model, label wants {want}")


def merge_same_clock(items, what="table"):
    """items: (name, uclk string, row). All rows of one table must share the memory clock."""
    clocks = {u for _, u, _ in items}
    if len(clocks) > 1:
        by = {}
        for n, u, _ in items:
            by.setdefault(u, []).append(n)
        raise TableError(f"{what}: rows at different memory clocks would be merged: "
                         + "; ".join(f"{u} MHz: {', '.join(v[:4])}" for u, v in sorted(by.items())))
    return [r for _, _, r in items]


def stream_bytes(path):
    """Bytes of one decoded token's weight reads, READ from the GGUF tensor table: all tensor bytes minus
    token_embd.weight (the input embedding stays on the CPU; a token reads one row of it). The output head
    (output.weight) is a separate tensor in these files and is streamed, so a file without one is refused."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gguf_rows
    f, _, tensors, data_start = gguf_rows.read_gguf(path)
    size = os.fstat(f.fileno()).st_size
    f.close()
    order = sorted(tensors, key=lambda t: t["offset"])
    sizes = {}
    for a, b in zip(order, order[1:]):
        sizes[a["name"]] = b["offset"] - a["offset"]
    if order:
        sizes[order[-1]["name"]] = size - data_start - order[-1]["offset"]
    if any(v <= 0 for v in sizes.values()) or not sizes:
        raise TableError(f"{path}: a tensor has no bytes (offsets out of order or past the end of the file)")
    for need in ("token_embd.weight", "output.weight"):
        if need not in sizes:
            raise TableError(f"{path}: no tensor {need} (tied embeddings: decode would read token_embd as the head)")
    total = sum(sizes.values())
    return dict(tensors=total, token_embd=sizes["token_embd.weight"], streamed=total - sizes["token_embd.weight"])


KLD_RE = {
    "mean_kld": re.compile(r"Mean\s+KLD:\s*([0-9.eE+-]+)"),
    "max_kld": re.compile(r"Maximum KLD:\s*([0-9.eE+-]+)"),
    "median_kld": re.compile(r"Median\s+KLD:\s*([0-9.eE+-]+)"),
    "p99_kld": re.compile(r"99\.0%\s+KLD:\s*([0-9.eE+-]+)"),
    "same_top": re.compile(r"Same top p:\s*([0-9.]+)"),
    "mean_ppl_q": re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
    "mean_ppl_base": re.compile(r"Mean PPL\(base\)\s*:\s*([0-9.]+)"),
}


def parse_kld(text):
    out = {}
    for k, rx in KLD_RE.items():
        m = rx.search(text)
        if m:
            out[k] = float(m.group(1))
    return out


# The saved logits are 16-bit compressed, so the same run against its own file is not exactly 0: the 99th percentile
# sat at 3.4e-5, the maximum at 6.3e-5 and the top token agreed in 99.992 to 100 % in the first run. The mean prints
# with 6 decimals, so "0.000000" means below 5e-7. The floor below is that noise, with some room on the maximum.
SELF_MEAN_MAX = 1e-6
SELF_MAX_MAX = 2e-4
SELF_TOP_MIN = 99.9


def check_self_kld(k):
    """The same run against its own saved logits must be at the noise floor of the 16-bit file."""
    if not k or "mean_kld" not in k:
        raise TableError("no KL divergence statistics found")
    if k["mean_kld"] >= SELF_MEAN_MAX or k.get("max_kld", 0.0) >= SELF_MAX_MAX or k.get("same_top", 100.0) < SELF_TOP_MIN:
        raise TableError(f"self-comparison is above the noise floor (mean < {SELF_MEAN_MAX:g}, max < {SELF_MAX_MAX:g}, "
                         f"same top >= {SELF_TOP_MIN} %): mean {k['mean_kld']}, max {k.get('max_kld')}, "
                         f"same top p {k.get('same_top')} %")


def check_tiny(k, limit=1e-4):
    if not k or "mean_kld" not in k:
        raise TableError("no KL divergence statistics found")
    if k["mean_kld"] >= limit:
        raise TableError(f"mean KLD {k['mean_kld']:.3e} is not below {limit:.0e}")


def parse_batched(path):
    """llama-batched-bench table rows: {B: S_TG}. Columns: PP TG B N_KV T_PP S_PP T_TG S_TG T S."""
    res = {}
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 10 and c[0].isdigit():
                res[int(c[2])] = float(c[7])
    return res


# ---------------------------------------------------------------------------------------------------------------
# Table builders (Markdown). Every table names the logs its numbers come from.
CONFIG_ORDER = ["fp8-unmod", "fp8-patched", "q8_0"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def uclk_of(run, name):
    """Memory clock label of one job: 'before/after' from its .meta."""
    m = read_meta(os.path.join(run, name + ".log"))
    return m.get("uclk_before", "?") + ("" if m.get("uclk_before") == m.get("uclk_after") else "/" + m.get("uclk_after", "?"))


def one_clock(run, names, what):
    """The memory clock label shared by the existing jobs `names` (before/after each job); raises when they differ,
    so no table merges rows taken at different memory clocks. A job missing from the run is skipped."""
    items = [(n, uclk_of(run, n), None) for n in names if os.path.exists(os.path.join(run, n + ".log"))]
    merge_same_clock(items, what)
    return items[0][1] if items else "?"


def fmt(x, d=1):
    return f"{x:.{d}f}"


def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bench_rows(run, name, label):
    path = os.path.join(run, name + ".log")
    if not os.path.exists(path):
        return []
    rows = parse_bench(path)
    check_label(rows, label)
    return rows


def sec_known(run):
    out = ["### Known answers (3B, 3 repetitions; first-hour numbers were at 1258 MHz memory)", ""]
    one_clock(run, [f"known-bench-3b-{c}" for c in CONFIG_ORDER], "known answers")
    rows = []
    first = {"fp8-unmod": (8536, 76.8), "fp8-patched": (9051, 146.0), "q8_0": (12123, 134.7)}
    for c in CONFIG_ORDER:
        r = bench_rows(run, f"known-bench-3b-{c}", f"3b-{c}")
        d = {x["test"]: x for x in r}
        if not d:
            continue
        pp, tg = d["pp512"], d["tg128"]
        rows.append([c, fmt(pp["median"], 0), f"{pp['lo']:.0f}-{pp['hi']:.0f}", first[c][0],
                     fmt(100 * (pp["median"] / first[c][0] - 1), 1) + " %", fmt(tg["median"], 1),
                     f"{tg['lo']:.1f}-{tg['hi']:.1f}", first[c][1], fmt(100 * (tg["median"] / first[c][1] - 1), 1) + " %",
                     uclk_of(run, f"known-bench-3b-{c}")])
    out.append(tbl(["config", "pp512 t/s", "min-max", "first hour", "diff", "tg128 t/s", "min-max", "first hour", "diff", "uclk"], rows))
    tb = []
    for k in ("f8-unmod", "f8-patched", "q8-unmod", "f16-unmod"):
        p = os.path.join(run, f"known-tbo-{k}.log")
        if os.path.exists(p):
            m = re.search(r"(\d+/\d+) tests passed", ANSI.sub("", open(p, errors="replace").read()))
            tb.append(f"known-tbo-{k}: {m.group(1) if m else 'no result line'} passed")
    out += ["", "test-backend-ops (ROCm0): " + "; ".join(tb)]
    return "\n".join(out)


def sec_probe(run):
    out = ["### Profiler known answer (bandwidth probe, 1 GiB read and 2 GiB copy, 50 reps)", ""]
    one_clock(run, ["probe-bw-plain", "probe-bw-trace"], "profiler known answer")
    plain = open(os.path.join(run, "probe-bw-plain.log"), errors="replace").read()
    res = {m.group(1): float(m.group(2)) for m in re.finditer(r"RESULT (read|copy) MiB \d+ median_ms ([0-9.]+)", plain)}
    import csv
    kt = os.path.join(run, "rocp-bw", "bw_kernel_trace.csv")
    d = {}
    with open(kt) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["Kernel_Name"].split("(")[0], []).append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6)
    rows = []
    for k in ("read", "copy"):
        tr = statistics.median(d[k + "_kernel"])
        rows.append([k, fmt(res[k], 4), fmt(tr, 4), fmt(100 * (tr / res[k] - 1), 2) + " %"])
    out.append(tbl(["kernel", "probe's own median ms (events)", "rocprofv3 median ms", "difference"], rows))
    return "\n".join(out)


def sec_a1(run):
    out = ["### A1: where one pp512 pass goes (kernel trace, median of 3 passes; wall = first kernel start to last kernel end)", ""]
    rows = []
    one_clock(run, [f"{k}-pp512-{m}-{c}" for k in ("ctl", "trace") for m in ("3b", "8b") for c in CONFIG_ORDER], "A1")
    for model in ("3b", "8b"):
        for c in CONFIG_ORDER:
            f = os.path.join(run, f"trace-pp512-{model}-{c}.analysis.json")
            if not os.path.exists(f):
                continue
            a = json.load(open(f))
            g = a["groups_ms"]
            ctl = a.get("control_ms_per_pass", float("nan"))
            gemm = g["gemm_fp16"] + g["mmq"]
            rows.append([f"{model} {c}", fmt(ctl, 2), fmt(a["wall_ms"], 2), fmt(gemm, 2), fmt(g["widen"], 2), fmt(g["act_convert"], 2),
                         fmt(g["scale_mul"], 2), fmt(g["quantize_act"], 2), fmt(g["attention"], 2), fmt(g["elementwise"], 2),
                         fmt(g["matvec_fp8"] + g["matvec_q"], 2), fmt(a["gap_ms"], 2), fmt(a.get("profiled_over_control", float("nan")), 3)])
    out.append(tbl(["config", "unprofiled ms/pass", "traced wall ms", "GEMM or MMQ", "widen", "act convert", "scale mul", "act quantize",
                    "attention", "norm/rope/etc", "matvec", "gaps", "traced / unprofiled"], rows))
    return "\n".join(out)


def sec_a2(run):
    out = ["### A2: matrix rate llama.cpp gets per weight shape, prompt pass (TFLOPS or int8 TOPS, median over layers and 3 passes)", ""]
    rows = []
    one_clock(run, [f"trace-pp512-{m}-{c}" for m in ("3b", "8b") for c in CONFIG_ORDER], "A2")
    for model in ("3b", "8b"):
        cols = {}
        for c in CONFIG_ORDER:
            f = os.path.join(run, f"trace-pp512-{model}-{c}.analysis.json")
            if os.path.exists(f):
                for sh in json.load(open(f)).get("shapes", []):
                    cols.setdefault((sh["pos"], sh["K"], sh["N"]), {})[c] = sh["tflops"]
        for (pos, k, n), v in cols.items():
            rows.append([model, pos, k, n] + [fmt(v.get(c, float("nan")), 1) for c in CONFIG_ORDER])
    out.append(tbl(["model", "matrix", "K", "N"] + CONFIG_ORDER, rows))
    return "\n".join(out)


def sec_a3(run):
    out = ["### A3: test-backend-ops perf, MUL_MAT m=4096, k=14336 (8B down projection), median of 6 runs (3 per build)", ""]
    pat = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=4096,n=(\d+),k=14336[^:]*:\s+\d+ runs -\s+([0-9.]+) us/run")
    data = {}
    n_logs = 0
    one_clock(run, [f"shapes-{b}-r{r}" for b in ("unmod", "patched") for r in (1, 2, 3)], "A3")
    for b in ("unmod", "patched"):
        for r in (1, 2, 3):
            p = os.path.join(run, f"shapes-{b}-r{r}.log")
            if not os.path.exists(p):
                continue
            n_logs += 1
            for m in pat.finditer(ANSI.sub("", open(p, errors="replace").read())):
                data.setdefault((m.group(1), int(m.group(2))), {}).setdefault(b, []).append(float(m.group(3)))
    rows = []
    flops = lambda n: 2.0 * 4096 * n * 14336
    for n in sorted({k[1] for k in data}):
        row = [n]
        for t in ("f16", "q8_0", "f8_e4m3"):
            for b in ("unmod", "patched"):
                v = data.get((t, n), {}).get(b)
                row.append(fmt(statistics.median(v), 1) if v else "-")
        rows.append(row)
    out.append(tbl(["n", "f16 unmod us", "f16 patched us", "q8_0 unmod us", "q8_0 patched us", "f8 unmod us", "f8 patched us"], rows))
    out.append("")
    v = {t: statistics.median(data[(t, 512)]["unmod"]) for t in ("f16", "q8_0", "f8_e4m3") if (t, 512) in data and "unmod" in data[(t, 512)]}
    if len(v) == 3:
        out.append(f"At n = 512 (unmodified build): f16 {v['f16']:.1f} us ({flops(512) / v['f16'] / 1e6:.1f} TFLOPS), q8_0 {v['q8_0']:.1f} us "
                   f"({flops(512) / v['q8_0'] / 1e6:.1f}), f8_e4m3 {v['f8_e4m3']:.1f} us ({flops(512) / v['f8_e4m3'] / 1e6:.1f}); "
                   f"f8 minus f16 = {v['f8_e4m3'] - v['f16']:.1f} us, the widen cost (58.7 M weights, 3 bytes each = "
                   f"{58.7e6 * 3 / (v['f8_e4m3'] - v['f16']) / 1e3:.0f} GB/s). Logs: shapes-*-r*.log.")
    return "\n".join(out)


def sec_b1(run, model):
    out = [f"### B1: {model} prompt and generation speed, tokens/s (median of 5 repetitions; 3 for pp4096)", ""]
    tests = ["pp16", "pp32", "pp64", "pp128", "pp256", "pp512", "pp1024", "pp2048", "pp4096", "tg128"]
    rows = []
    one_clock(run, [f"bench-{model}-{c}-{x}" for c in CONFIG_ORDER for x in ("prompts", "pp2048", "pp4096")], f"B1 {model}")
    for c in CONFIG_ORDER:
        label = f"{model}-{c}"
        d = {}
        clocks = set()
        for suffix in ("prompts", "pp2048", "pp4096"):
            for r in bench_rows(run, f"bench-{label}-{suffix}", label):
                d[r["test"]] = r
            if os.path.exists(os.path.join(run, f"bench-{label}-{suffix}.log")):
                clocks.add(uclk_of(run, f"bench-{label}-{suffix}"))
        if not d:
            continue
        rows.append([c] + [fmt(d[t]["median"], 1 if t == "tg128" else 0) if t in d else "-" for t in tests] + [",".join(sorted(clocks))])
    out.append(tbl(["config"] + tests + ["uclk"], rows))
    # spread line
    sp = []
    for c in CONFIG_ORDER:
        for r in bench_rows(run, f"bench-{model}-{c}-prompts", f"{model}-{c}"):
            if r["test"] in ("pp512", "tg128"):
                sp.append(f"{c} {r['test']}: {r['lo']:.1f}-{r['hi']:.1f}")
    out += ["", "Min-max of the repetitions: " + "; ".join(sp)]
    return "\n".join(out)


def sec_b4(run):
    out = ["### B4: generation at context depth, tokens/s (tg128 after a prompt of the given depth; median of 5)", ""]
    rows = []
    one_clock(run, [f"bench-{m}-{c}-depth" for m in ("3b", "8b") for c in CONFIG_ORDER], "B4")
    for model in ("3b", "8b"):
        for c in CONFIG_ORDER:
            r = bench_rows(run, f"bench-{model}-{c}-depth", f"{model}-{c}")
            d = {x["depth"]: x for x in r}
            if not d:
                continue
            base = d[0]["median"]
            rows.append([f"{model} {c}"] + [f"{d[k]['median']:.1f} ({100 * (d[k]['median'] / base - 1):+.1f} %)" if k in d else "-" for k in (0, 512, 4096)])
    out.append(tbl(["config", "depth 0", "depth 512", "depth 4096"], rows))
    return "\n".join(out)


def sec_b2(run):
    out = ["### B2: parallel streams, S_TG tokens/s (llama-batched-bench, 128 generated tokens per stream)", ""]
    one_clock(run, [f"batched-{m}-{c}-npp{n}" for m in ("3b", "8b") for c in CONFIG_ORDER for n in (128, 512)], "B2")
    for npp in (128, 512):
        rows = []
        for model in ("3b", "8b"):
            for c in CONFIG_ORDER:
                p = os.path.join(run, f"batched-{model}-{c}-npp{npp}.log")
                if os.path.exists(p):
                    d = parse_batched(p)
                    rows.append([f"{model} {c}"] + [fmt(d[b], 1) if b in d else "-" for b in range(1, 9)])
        out += [f"Prompt {npp} tokens per stream:", "", tbl(["config"] + [f"{b}" for b in range(1, 9)], rows), ""]
    return "\n".join(out)


def sec_b3(run):
    out = ["### B3: quality", "", "Perplexity at 4 chunks of 512 (final estimate):", ""]
    rows = []
    for pre, nm in (("llama32-3b", "3b"), ("llama31-8b", "8b")):
        for v in ("fp8-unmod-default", "fp8-unmod-b8", "fp8-patched", "fp8-patched-b8", "q8_0"):
            p = os.path.join(run, f"ppl4-{pre}-{v}.log")
            if os.path.exists(p):
                m = re.search(r"Final estimate: PPL = ([0-9.]+)", open(p, errors="replace").read())
                rows.append([nm, v, m.group(1) if m else "no final estimate (NaN?)"])
    out.append(tbl(["model", "run", "PPL"], rows))
    out += ["", "KL divergence of each run against the saved logits of the unmodified FP8 default-batch run (50 chunks of 512):", ""]
    rows = []
    for pre, nm in (("llama32-3b", "3b"), ("llama31-8b", "8b")):
        for v, desc in (("a-same", "(a) same run again"), ("b-b8", "(b) FP8 batch 8"), ("c-patched", "(c) FP8 patched build"), ("d-q8_0", "(d) Q8_0")):
            p = os.path.join(run, f"kld-{pre}-{v}.log")
            if os.path.exists(p):
                k = parse_kld(ANSI.sub("", open(p, errors="replace").read()))
                rows.append([nm, desc, k.get("mean_kld", "?"), k.get("median_kld", "?"), k.get("p99_kld", "?"), k.get("max_kld", "?"), k.get("same_top", "?")])
    out.append(tbl(["model", "comparison", "mean KLD", "median KLD", "99 % KLD", "max KLD", "same top token %"], rows))
    return "\n".join(out)


def sample_stats(d):
    """From a watch_card summary.txt: the lines the write-up quotes."""
    p = os.path.join(d, "summary.txt")
    if not os.path.exists(p):
        return None
    t = open(p).read()
    g = lambda rx: (re.search(rx, t) or [None, "?"])[1]
    o = {}
    m = re.search(r"SCLK_MHz awake \(time-weighted\) p10 (\d+) p50 (\d+) p90 (\d+) p99 (\d+) max (\d+)", t)
    o["sclk"] = f"{m.group(2)} / {m.group(3)} / {m.group(5)}" if m else "?"
    m = re.search(r"POWER_W reported[^\n]* p10 (\d+) p50 (\d+) p90 (\d+) p99 (\d+) max (\d+)", t)
    o["power"] = f"{m.group(2)} / {m.group(3)} / {m.group(5)}" if m else "?"
    o["hot"] = g(r"HOTSPOT_C max (\d+)")
    m = re.search(r"UCLK_MHz[^\n]* p10 (\d+) p50 (\d+) p90 (\d+) p99 (\d+) max (\d+)", t)
    o["uclk"] = f"{m.group(1)}-{m.group(5)}" if m else "?"
    o["samples"] = g(r"SAMPLES (\d+)")
    m = re.search(r"CLOCK_BANDS[^\n]*", t)
    o["bands"] = m.group(0) if m else ""
    m = re.search(r"THROTTLE indep_throttle_status: ([^\n]*)", t)
    o["throttle"] = m.group(1) if m else "?"
    return o


def sec_c1(run):
    out = ["### C1: clocks, power and temperature around 20 s loads (sampler every 20 ms; core clock while awake: median / p90 / max MHz; reported power W: median / p90 / max; profile Testing, 317 W cap)", ""]
    rows = []
    names = sorted(glob.glob(os.path.join(run, "beh", "*")))
    for d in names:
        if not os.path.isdir(d):
            continue
        s = sample_stats(d)
        if s:
            rows.append([os.path.basename(d), s["samples"], s["sclk"], s["power"], s["hot"], s["uclk"], s["throttle"]])
    out.append(tbl(["recording", "samples", "core MHz awake", "power W", "hotspot max C", "memory clock MHz min-max", "throttle bits seen"], rows))
    return "\n".join(out)


def sec_c2(run):
    out = ["### C2 and C3: bandwidth reached in generation, and gaps between kernels (tg128 traces; unprofiled rates from ctl-tg128)", ""]
    rows, grows = [], []
    one_clock(run, [f"{k}-tg128-{m}-{c}" for k in ("ctl", "trace") for m in ("3b", "8b") for c in CONFIG_ORDER], "C2 and C3")
    for model in ("3b", "8b"):
        for c in CONFIG_ORDER:
            f = os.path.join(run, f"trace-tg128-{model}-{c}.analysis.json")
            if not os.path.exists(f):
                continue
            a = json.load(open(f))
            g = a["groups_ms"]
            mv = g["matvec_fp8"] + g["matvec_q"]
            shapes = "; ".join(f"{s['pos']} {s['gbps']:.0f}" for s in a.get("shapes", []))
            rows.append([f"{model} {c}", fmt(a["wall_ms"], 3), fmt(mv, 3), fmt(100 * mv / a["wall_ms"], 1) + " %", shapes])
            gu = a["gap_us"]
            grows.append([f"{model} {c}", a["kernels_per_pass"], fmt(gu["median"], 2), fmt(gu["p95"], 2), fmt(gu["max"], 1),
                          fmt(100 * a["gap_ms"] / a["wall_ms"], 1) + " %", fmt(a.get("implied_gap_us_unprofiled", float("nan")), 2),
                          fmt(a.get("profiled_over_control", float("nan")), 3)])
    out += [tbl(["config", "traced ms/token", "matvec kernels ms", "share of token", "GB/s per matrix (name, from K x N x bytes / kernel time)"], rows), "",
            tbl(["config", "kernels/token", "median gap us", "p95 us", "max us", "gaps share of token", "implied unprofiled gap us", "traced / unprofiled"], grows)]
    return "\n".join(out)


def sec_c4(run):
    out = ["### C4: first prompt after idle (pp512 t/s of consecutive tests; idle2s = 2 s idle before each test, loop = back to back)", ""]
    one_clock(run, [f"beh-{k}-{c}" for k in ("idle2s-pp", "loop-pp", "idle2s-tg") for c in ("3b-fp8-patched", "3b-q8_0")], "C4")
    rows = []
    for c in ("3b-fp8-patched", "3b-q8_0"):
        for kind in ("idle2s-pp", "loop-pp"):
            p = os.path.join(run, f"beh-{kind}-{c}.log")
            if not os.path.exists(p):
                continue
            ts = []
            with open(p, errors="replace") as f:
                for line in f:
                    if line.startswith("{"):
                        ts += json.loads(line)["samples_ts"]
            rows.append([c, kind, " ".join(f"{x:.0f}" for x in ts)])
    out.append(tbl(["config", "series", "t/s in order"], rows))
    rows = []
    for c in ("3b-fp8-patched", "3b-q8_0"):
        p = os.path.join(run, f"beh-idle2s-tg-{c}.log")
        if os.path.exists(p):
            ts = []
            for line in open(p, errors="replace"):
                if line.startswith("{"):
                    ts += json.loads(line)["samples_ts"]
            rows.append([c, "idle2s-tg128", " ".join(f"{x:.1f}" for x in ts)])
    if rows:
        out += ["", tbl(["config", "series", "t/s in order"], rows)]
    return "\n".join(out)


def sec_clocks(run):
    out = ["### Memory clock of every GPU job (current_uclk before/after; the run's first reading is the reference)", ""]
    seen = {}
    for m in sorted(glob.glob(os.path.join(run, "*.meta"))):
        meta = read_meta(m.replace(".meta", ".log"))
        key = meta.get("uclk_before", "?") + "/" + meta.get("uclk_after", "?")
        seen.setdefault(key, []).append(os.path.basename(m)[:-5])
    for k, v in seen.items():
        out.append(f"- {k} MHz: {len(v)} jobs" + ("" if k in ("1408/1408",) else " (" + ", ".join(v[:12]) + ")"))
    return "\n".join(out)


def sec_a5(run):
    out = ["### A5: Q8_0 through dequantize + hipBLAS (-DGGML_CUDA_FORCE_CUBLAS=ON) against the normal Q8_0 and FP8", ""]
    rows = []
    tests = ["pp16", "pp32", "pp64", "pp128", "pp256", "pp512", "pp1024", "tg128"]
    one_clock(run, [n for m in ("3b", "8b") for n in (f"bench-{m}-q8_0-prompts", f"a5-bench-{m}-q8_0-cublas", f"bench-{m}-fp8-unmod-prompts")], "A5")
    for model in ("3b", "8b"):
        for lab, name in (("q8_0 normal build (MMQ)", f"bench-{model}-q8_0-prompts"), ("q8_0 FORCE_CUBLAS build", f"a5-bench-{model}-q8_0-cublas"),
                          ("fp8 unmod", f"bench-{model}-fp8-unmod-prompts")):
            p = os.path.join(run, name + ".log")
            if os.path.exists(p):
                d = {r["test"]: r for r in parse_bench(p)}
                rows.append([model, lab] + [fmt(d[t]["median"], 0 if t != "tg128" else 1) if t in d else "-" for t in tests] + [uclk_of(run, name)])
    out.append(tbl(["model", "config"] + tests + ["uclk"], rows))
    for model in ("3b", "8b"):
        f = os.path.join(run, f"a5-trace-pp512-{model}-q8_0-cublas.report.txt")
        if os.path.exists(f):
            out += ["", f"A5 {model} trace report ({os.path.basename(f)}):", "", "```", open(f).read().split("VERDICT")[0].strip().replace(os.path.abspath(run), "runs/" + os.path.basename(os.path.abspath(run))), "```"]
    return "\n".join(out)


def model_sizes(run):
    sizes = {}
    p = os.path.join(run, "models.txt")
    for line in open(p):
        f = line.split()
        if len(f) == 3 and f[1].isdigit():
            sizes[f[0]] = int(f[1])
    return sizes


def model_streamed(run):
    """models.txt lines `streamed <model> <tensor bytes> <token_embd bytes> <streamed bytes>` (written by setup)."""
    out = {}
    for line in open(os.path.join(run, "models.txt")):
        f = line.split()
        if len(f) == 5 and f[0] == "streamed" and all(x.isdigit() for x in f[2:]):
            out[f[1]] = dict(tensors=int(f[2]), token_embd=int(f[3]), streamed=int(f[4]))
    if not out:
        raise TableError("models.txt has no `streamed` lines: the generation share of the bus needs the bytes decode "
                         "reads, not the file size (run setup again)")
    return out


def probe_bus_gbps(run):
    plain = open(os.path.join(run, "probe-bw-plain.log"), errors="replace").read()
    m = re.search(r"RESULT read MiB 1024 median_ms ([0-9.]+)", plain)
    return 1024 * 1048576 / (float(m.group(1)) * 1e6)


def kv_bytes(model, depth):
    """F16 K and V cache bytes one token reads at a context depth (READ from the models' layer count and KV width)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import llm_kernels
    m = llm_kernels.MODELS[model]
    return m["layers"] * 2 * m["kv"] * depth * 2


def sec_bw(run):
    bus = probe_bus_gbps(run)
    sz = model_sizes(run)
    st = model_streamed(run)
    clk = one_clock(run, ["probe-bw-plain"] + [f"bench-{m}-{c}-depth" for m in ("3b", "8b") for c in CONFIG_ORDER],
                    "generation against the bus")
    out = [f"### Generation against the bus: tokens/s x streamed bytes / {bus:.0f} GB/s (the bandwidth probe's read rate in this run, "
           f"memory clock {clk} MHz)", "",
           "Streamed bytes = all tensor bytes of the GGUF minus `token_embd.weight`, which decode does not stream (READ from the "
           "GGUF tensor table with `llm_tables.py streamed`, recorded in models.txt). The last three columns add the F16 KV cache at "
           "depth 4096 to the streamed bytes and compare the predicted slowdown (INFERRED: speed falls with the extra bytes) with the "
           "measured one.", ""]
    rows = []
    for model, pre in (("3b", "llama32-3b"), ("8b", "llama31-8b")):
        for c in CONFIG_ORDER:
            r = bench_rows(run, f"bench-{model}-{c}-depth", f"{model}-{c}")
            d = {x["depth"]: x for x in r}
            if 0 not in d:
                continue
            name = f"{pre}-fp8-outq8" if "fp8" in c else f"{pre}-q8_0"
            b = st[name]["streamed"]
            g = d[0]["median"] * b / 1e9
            row = [f"{model} {c}", fmt(d[0]["median"], 1), fmt(sz[name] / 1e9, 3), fmt(st[name]["token_embd"] / 1e9, 3), fmt(b / 1e9, 3),
                   fmt(g, 0), fmt(100 * g / bus, 1) + " %"]
            if 4096 in d:
                kv = kv_bytes(model, 4096)
                row += [fmt(kv / 1e9, 3), f"{100 * (b / (b + kv) - 1):+.1f} %", f"{100 * (d[4096]['median'] / d[0]['median'] - 1):+.1f} %"]
            else:
                row += ["-", "-", "-"]
            rows.append(row)
    out.append(tbl(["config", "tg128 t/s (depth 0)", "file GB", "token_embd GB", "streamed GB", "GB/s", "share of the read rate",
                    "KV GB at 4096", "predicted speed change", "measured"], rows))
    return "\n".join(out)


def sec_det(run):
    out = ["### Determinism: the logits file of the same perplexity run written twice", ""]
    for line in open(os.path.join(run, "summary.md")):
        if line.startswith("- determinism"):
            out.append(line.rstrip())
    return "\n".join(out)


def tables(run):
    parts = [sec_clocks(run), sec_probe(run), sec_known(run), sec_a1(run), sec_a2(run), sec_a3(run), sec_a5(run), sec_b1(run, "3b"), sec_b1(run, "8b"), sec_bw(run),
             sec_b4(run), sec_b2(run), sec_b3(run), sec_det(run), sec_c2(run), sec_c1(run), sec_c4(run)]
    return "\n\n".join(parts)

# ---------------------------------------------------------------------------------------------------------------
# Round 2, item 1: quality against the exact-weights reference. Logs: r2q-<model>-ref (writes the logits), then
# r2q-<model>-fp8-unmod, -fp8-patched, -q8_0 (each compared against them).
R2Q_FILE = (("fp8-unmod", "-auto.gguf", "llama.cpp-latest/"), ("fp8-patched", "-auto.gguf", "llama.cpp-latest-patched/"),
            ("q8_0", "-q8_0.gguf", "llama.cpp-latest/"), ("f16ref", "-f16.gguf", "llama.cpp-latest/"))
R2Q_REF = ("-exact-f32.gguf", "-f16.gguf")


def r2q_header(text):
    m = re.search(r"^# \S+: (.*)$", text, re.M)
    return m.group(1) if m else ""


def r2q_rows(run, pre, chunks):
    """Returns (reference PPL, reference chunk count, rows) for one model; raises TableError on a wrong model, build, chunk count,
    reference or memory clock. Every run must say it compared against the reference of this model (its PPL(base) equals the reference's)."""
    ref_log = os.path.join(run, f"r2q-{pre}-ref.log")
    rt = ANSI.sub("", open(ref_log, errors="replace").read())
    refname = [x for x in R2Q_REF if f"{pre}{x}" in r2q_header(rt).split(" -m ")[-1].split()[0]]
    if not refname:
        raise TableError(f"{ref_log}: the reference run did not load {pre}{R2Q_REF[0]} or {pre}{R2Q_REF[1]}")
    refname = refname[0]
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", rt)
    nref = re.search(r"calculating perplexity over (\d+) chunks", rt)
    if not m or not nref:
        raise TableError(f"{ref_log}: no final estimate or chunk count (the reference run failed)")
    if int(nref.group(1)) != chunks:
        raise TableError(f"{ref_log}: {nref.group(1)} chunks, {chunks} wanted")
    uclks = {read_meta(ref_log).get("uclk_before"), read_meta(ref_log).get("uclk_after")}
    rows = []
    for v, fname, build in R2Q_FILE:
        p = os.path.join(run, f"r2q-{pre}-{v}.log")
        if v == "f16ref" and (not os.path.exists(p) or refname == "-f16.gguf"):
            continue
        t = ANSI.sub("", open(p, errors="replace").read())
        hd = r2q_header(t)
        if f"{pre}{fname}" not in hd:
            raise TableError(f"{p}: label {v} but the command loads something else: {hd[:160]}")
        if build not in hd:
            raise TableError(f"{p}: label {v} wants the build folder {build}: {hd[:160]}")
        n = re.search(r"computing over (\d+) chunks", t)
        if not n or int(n.group(1)) != chunks:
            raise TableError(f"{p}: chunk count {n.group(1) if n else 'missing'}, {chunks} wanted")
        k = parse_kld(t)
        if "mean_kld" not in k or "mean_ppl_q" not in k:
            raise TableError(f"{p}: no KL divergence statistics")
        # the reference run's "Final estimate" and the comparison's "Mean PPL(base)" are computed differently and differ by about 2e-3
        if abs(k["mean_ppl_base"] - float(m.group(1))) > 5e-3:
            raise TableError(f"{p}: PPL(base) {k['mean_ppl_base']} is not the reference's {m.group(1)}: another reference was used")
        meta = read_meta(p)
        uclks |= {meta.get("uclk_before"), meta.get("uclk_after")}
        rows.append((v, k))
    bases = {k["mean_ppl_base"] for _, k in rows}
    if max(bases) - min(bases) > 1e-4:
        raise TableError(f"{pre}: the runs report different PPL(base) values {sorted(bases)}: they did not share one reference")
    if len(uclks - {None}) > 1:
        raise TableError(f"{pre}: memory clocks differ between the jobs: {sorted(uclks - {None})}")
    return float(m.group(1)), rows, sorted(uclks - {None}), refname


def sec_r2q(run, chunks=50):
    out = ["### Item 1: perplexity and KL divergence against the reference (3B: exact F32; 8B: F16, see the text)", ""]
    for pre, nm in (("llama32-3b", "3B"), ("llama31-8b", "8B")):
        if not os.path.exists(os.path.join(run, f"r2q-{pre}-ref.log")):
            continue
        ppl, rows, uclk, refname = r2q_rows(run, pre, chunks)
        out += [f"{nm} ({chunks} chunks of 512; reference {pre}{refname}, perplexity {ppl}; memory clock {', '.join(uclk)} MHz):", ""]
        out.append(tbl(["run", "PPL", "PPL - reference", "mean KLD", "median KLD", "99 % KLD", "max KLD", "same top token %"],
                       [[v, k["mean_ppl_q"], f"{k['mean_ppl_q'] - k['mean_ppl_base']:+.4f}", k["mean_kld"], k.get("median_kld", "?"), k.get("p99_kld", "?"),
                         k.get("max_kld", "?"), k.get("same_top", "?")] for v, k in rows]))
        out.append("")
    return "\n".join(out)

# ---------------------------------------------------------------------------------------------------------------
# Round 2, item 6: the clock inside each test of a llama-bench run that has a 20 ms sampler recording. The tests run back to
# back (each a warm-up pass plus its repetitions), so the window of each test is rebuilt backwards from the last sample at
# which the graphics engine was busy, using the tests' own durations (avg_ns x (reps + 1)).
def read_samples(path):
    rows = []
    with open(path) as f:
        hdr = None
        for line in f:
            if line.startswith("#"):
                continue
            if hdr is None:
                hdr = line.strip().split(",")
                continue
            v = line.strip().split(",")
            if len(v) == len(hdr):
                d = dict(zip(hdr, v))
                rows.append((float(d["t_s"]), float(d["gfxclk_cur_MHz"]), float(d["gfx_act_pct"])))
    return rows


def test_windows(tests, end_s):
    """tests: [(name, duration_s)] in run order; returns [(name, start_s, end_s)] with the last test ending at end_s."""
    out, t = [], end_s
    for name, dur in reversed(tests):
        out.append((name, t - dur, t))
        t -= dur
    return list(reversed(out))


def window_clock(samples, start, end):
    """mean and minimum graphics clock of the samples inside [start, end), and the clock of the first sample of it."""
    v = [c for t, c, _ in samples if start <= t < end]
    if not v:
        raise TableError(f"no sample inside the window {start:.3f} to {end:.3f} s")
    return statistics.mean(v), min(v), v[0]


def pearson(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def r2clock_job(log, samp):
    rows = []
    with open(log, errors="replace") as f:
        for line in f:
            if line.startswith("{"):
                d = json.loads(line)
                rows.append((f"pp{d['n_prompt']}", d["avg_ns"] * (len(d["samples_ns"]) + 1) / 1e9, statistics.median(d["samples_ts"])))
    smp = read_samples(samp)
    busy = [t for t, _, a in smp if a >= 20]
    if not rows or not busy:
        raise TableError(f"{log}: no test rows or no busy samples")
    wins = test_windows([(n, d) for n, d, _ in rows], busy[-1])
    out = {}
    for (n, a, b), (_, _, ts) in zip(wins, rows):
        mean, lo, first = window_clock(smp, a, b)
        act = statistics.mean(x for t, _, x in smp if a <= t < b)
        out[n] = dict(ts=ts, mean=mean, min=lo, first=first, dur=b - a, act=act)
    return out


def sec_r2clock(run):
    import glob
    per = {}
    for log in sorted(glob.glob(os.path.join(run, "r2clock-*.log"))):
        name = os.path.basename(log)[len("r2clock-"):-4]
        label, i = name.rsplit("-", 1)
        d = os.path.join(run, "r2clock", name)
        if not os.path.exists(os.path.join(d, "samples.csv")):
            continue
        meta = read_meta(log)
        if meta.get("uclk_before") != meta.get("uclk_after"):
            raise TableError(f"{name}: memory clock changed inside the job")
        per.setdefault(label, []).append((int(i), r2clock_job(log, os.path.join(d, "samples.csv"))))
    out = ["### Item 6: the 8B prompt rows against the graphics clock inside each test", ""]
    for label, runs in sorted(per.items()):
        rows = []
        for t in ("pp16", "pp32", "pp64", "pp128", "pp256", "pp512", "pp1024"):
            ts = [r[t]["ts"] for _, r in runs if t in r]
            mean = [r[t]["mean"] for _, r in runs if t in r]
            first = [r[t]["first"] for _, r in runs if t in r]
            act = [r[t]["act"] for _, r in runs if t in r]
            rows.append([t, len(ts), fmt(statistics.median(ts), 0), f"{min(ts):.0f} to {max(ts):.0f}", f"{max(ts) / min(ts):.2f}",
                         fmt(statistics.median(mean), 0), f"{min(mean):.0f} to {max(mean):.0f}", fmt(pearson(ts, mean), 2), fmt(pearson(ts, first), 2), f"{min(act):.0f} to {max(act):.0f}", fmt(pearson(ts, act), 2)])
        out += [f"{label} ({len(runs)} runs, memory clock held):", "",
                tbl(["row", "runs", "median tok/s", "min to max", "max/min", "median clock in window MHz", "clock min to max", "r(speed, mean clock)", "r(speed, clock at start)", "GPU busy % in window", "r(speed, busy %)"], rows), ""]
    return "\n".join(out)

# ---------------------------------------------------------------------------------------------------------------
# Round 2, item 7a: streams, repeated and interleaved. Logs r2streams-<label>-r<k> (label like 3b-fp8-patched).
R2S_FILE = {"fp8-unmod": ("-fp8-outq8.gguf", "llama.cpp-latest/"), "fp8-patched": ("-fp8-outq8.gguf", "llama.cpp-latest-patched/"), "q8_0": ("-q8_0.gguf", "llama.cpp-latest/")}


def r2streams_data(run, reps=6):
    """{label: {B: [S_TG per repetition]}}; raises on a wrong model or build in a log, a missing repetition or a memory clock that moved."""
    data, uclks = {}, set()
    for model, pre in (("3b", "llama32-3b"), ("8b", "llama31-8b")):
        for c in CONFIG_ORDER:
            label = f"{model}-{c}"
            fname, build = R2S_FILE[c]
            for r in range(1, reps + 1):
                p = os.path.join(run, f"r2streams-{label}-r{r}.log")
                if not os.path.exists(p):
                    raise TableError(f"{label}: repetition {r} of {reps} is missing")
                hd = r2q_header(ANSI.sub("", open(p, errors="replace").read()))
                if f"{pre}{fname}" not in hd or build not in hd:
                    raise TableError(f"{p}: label {label} but the command is {hd[:150]}")
                d = parse_batched(p)
                if sorted(d) != list(range(1, 9)):
                    raise TableError(f"{p}: streams {sorted(d)}, 1 to 8 wanted")
                m = read_meta(p)
                uclks |= {m.get("uclk_before"), m.get("uclk_after")}
                for b, v in d.items():
                    data.setdefault(label, {}).setdefault(b, []).append(v)
    if len(uclks - {None}) > 1:
        raise TableError(f"memory clocks differ between the jobs: {sorted(uclks - {None})}")
    return data, sorted(uclks - {None})


def sec_r2streams(run):
    data, uclk = r2streams_data(run)
    out = ["### Item 7a: streams 1 to 8, S_TG tokens/s, median of 6 interleaved repetitions (min to max in brackets; spread above 6 % marked *)", "",
           f"Memory clock {', '.join(uclk)} MHz in every job.", ""]
    for model in ("3b", "8b"):
        rows = []
        for c in CONFIG_ORDER:
            v = data[f"{model}-{c}"]
            rows.append([f"{model} {c}"] + [f"{statistics.median(v[b]):.1f} [{min(v[b]):.0f}-{max(v[b]):.0f}]" + ("*" if (max(v[b]) - min(v[b])) / statistics.median(v[b]) > 0.06 else "") for b in range(1, 9)])
        out += [tbl(["config"] + [str(b) for b in range(1, 9)], rows), ""]
        rows = []
        for b in range(1, 9):
            p, q = data[f"{model}-fp8-patched"][b], data[f"{model}-q8_0"][b]
            mp, mq = statistics.median(p), statistics.median(q)
            sep = "separate" if (min(p) > max(q) or max(p) < min(q)) else "ranges overlap"
            rows.append([b, fmt(mp, 1), fmt(mq, 1), f"{100 * (mp / mq - 1):+.1f} %", sep])
        out += [f"{model}: patched FP8 against Q8_0 per stream count:", "", tbl(["streams", "patched", "Q8_0", "patched vs Q8_0", "the six repetitions of each"], rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Round 2, item 7b: test-backend-ops MUL_MAT perf lines at the real weight shapes.
PERF_RE = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),.*?\):\s+\d+ runs -\s+([0-9.]+) us/run")
SHAPES_3B = ((3072, 3072, "q, o"), (1024, 3072, "k, v"), (8192, 3072, "gate, up"), (3072, 8192, "down"), (128256, 3072, "output"))
SHAPES_8B = ((4096, 4096, "q, o"), (1024, 4096, "k, v"), (14336, 4096, "gate, up"), (4096, 14336, "down"), (128256, 4096, "output"))


def r2shapes_data(run, reps=3):
    """{(type, m, n, k): [us per repetition]} from r2shapes-r<k> logs; duplicates inside one log (the tool's own m=4096 k=14336
    cases are in its list too) are averaged by their median."""
    data = {}
    for r in range(1, reps + 1):
        p = os.path.join(run, f"r2shapes-r{r}.log")
        if not os.path.exists(p):
            raise TableError(f"repetition {r} of {reps} is missing")
        m = read_meta(p)
        if m.get("uclk_before") != m.get("uclk_after"):
            raise TableError(f"{p}: memory clock changed inside the job")
        one = {}
        for line in ANSI.sub("", open(p, errors="replace").read()).splitlines():
            x = PERF_RE.search(line)
            if x:
                one.setdefault((x.group(1), int(x.group(2)), int(x.group(3)), int(x.group(4))), []).append(float(x.group(5)))
        for k, v in one.items():
            data.setdefault(k, []).append(statistics.median(v))
    return data


def sec_r2shapes(run, widen_rate_ref=None):
    data = r2shapes_data(run)
    med = {k: statistics.median(v) for k, v in data.items()}
    out = ["### Item 7b: MUL_MAT at the real shapes (test-backend-ops perf, median of 3 repetitions, microseconds per call)", ""]
    for label, shapes in (("3B", SHAPES_3B), ("8B", SHAPES_8B)):
        rows = []
        for m, k, what in shapes:
            for n in (1, 8, 32, 512, 2048):
                try:
                    f8, f16, q8 = med[("f8_e4m3", m, n, k)], med[("f16", m, n, k)], med[("q8_0", m, n, k)]
                except KeyError:
                    raise TableError(f"{label} shape m={m} n={n} k={k}: a type is missing")
                flop = 2.0 * m * n * k
                widen = f8 - f16
                implied = (3.0 * m * k / (widen * 1e3)) if n >= 32 and widen > 0 else float("nan")
                rows.append([f"{m} x {k} ({what})", n, fmt(f8, 1), fmt(f16, 1), fmt(q8, 1), fmt(flop / (f16 * 1e6), 1), fmt(flop / (q8 * 1e6), 1),
                             fmt(widen, 1), fmt(implied, 0) if implied == implied else "-"])
        out += [f"{label}:", "", tbl(["m x k", "n", "f8_e4m3 us", "f16 us", "q8_0 us", "f16 TFLOPS", "q8_0 TFLOPS", "f8 - f16 us", "implied widen GB/s (3 m k / diff)"], rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
CANNED_KLD = """====== Perplexity statistics ======
Mean PPL(Q)                   :   9.704900 +/-   0.814000
Mean PPL(base)                :   9.704900 +/-   0.814000
====== KL divergence statistics ======
Mean    KLD:   {mean} +/-   0.000000
Maximum KLD:   {mx}
99.0%   KLD:   0.000000
Median  KLD:   0.000000
====== Token probability statistics ======
Same top p: {top} +/- 0.000 %
"""


def canned_gguf(path, with_embd=True, with_output=True):
    """A tiny GGUF v3 file: output.weight 320 bytes, token_embd.weight 640, one layer matrix 1280 (32-byte aligned)."""
    import struct
    names = []
    off = 0
    for name, size in (("output.weight", 320), ("token_embd.weight", 640), ("blk.0.ffn_down.weight", 1280)):
        if (name == "token_embd.weight" and not with_embd) or (name == "output.weight" and not with_output):
            continue
        names.append((name, size, off))
        off += size
    head = b"GGUF" + struct.pack("<IQQ", 3, len(names), 0)
    for name, size, o in names:
        head += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<IQIQ", 1, size, 0, o)
    head += b"\0" * (-len(head) % 32)
    with open(path, "wb") as f:
        f.write(head + bytes(off))


def canned_run(scratch, name, uclk_of_job=None):
    """A small run folder: the bandwidth probe (640 GB/s), the 3B depth jobs of three configs, models.txt with `streamed`
    lines. uclk_of_job: {job name: 'before/after' override}."""
    run = os.path.join(scratch, name)
    os.makedirs(run, exist_ok=True)
    uclk_of_job = uclk_of_job or {}

    def meta(job):
        u = uclk_of_job.get(job, "1408/1408").split("/")
        with open(os.path.join(run, job + ".meta"), "w") as f:
            f.write(f"uclk_before={u[0]}\nuclk_after={u[-1]}\nexit=0\n")

    with open(os.path.join(run, "probe-bw-plain.log"), "w") as f:
        f.write("RESULT read MiB 1024 median_ms 1.6777216\n")
    meta("probe-bw-plain")
    for c, typ, ts0, ts4 in (("fp8-unmod", "F8_E4M3", 50.0, 48.0), ("fp8-patched", "F8_E4M3", 100.0, 90.0), ("q8_0", "Q8_0", 100.0, 90.0)):
        job = f"bench-3b-{c}-depth"
        with open(os.path.join(run, job + ".log"), "w") as f:
            for d, ts in ((0, ts0), (512, ts0), (4096, ts4)):
                f.write(json.dumps(dict(n_prompt=0, n_gen=128, n_depth=d, samples_ts=[ts, ts, ts], model_type=f"llama 3B {typ}",
                                        model_size=1)) + "\n")
        meta(job)
    with open(os.path.join(run, "models.txt"), "w") as f:
        f.write("llama32-3b-fp8-outq8 4000000000 abc\nllama32-3b-q8_0 3800000000 def\n")
        f.write("streamed llama32-3b-fp8-outq8 4000000000 800000000 3200000000\n")
        f.write("streamed llama32-3b-q8_0 3800000000 400000000 3400000000\n")
    return run


def selftest(scratch=None):
    bad = 0
    if scratch is None:
        scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "selftest-tables")
    os.makedirs(scratch, exist_ok=True)

    def expect(name, want_fail, fn):
        nonlocal bad
        try:
            fn()
            got_fail, why = False, ""
        except TableError as e:
            got_fail, why = True, str(e)
        status = "ok" if got_fail == want_fail else "WRONG"
        if got_fail != want_fail:
            bad += 1
        print(f"  {status:5s} {name}: {'FAIL' if got_fail else 'PASS'} seen (wanted {'FAIL' if want_fail else 'PASS'})"
              f"{': ' + why if why else ''}")

    def check(name, ok, note=""):
        nonlocal bad
        if not ok:
            bad += 1
        print(f"  {'ok' if ok else 'WRONG':5s} {name}{': ' + note if note else ''}")

    fp8 = dict(test="pp512", model_type="llama 3B F8_E4M3", model_size=4025906976, log="a.log")
    q8 = dict(test="pp512", model_type="llama 3B Q8_0", model_size=3832688896, log="b.log")
    expect("control: FP8 row under an FP8 label", False, lambda: check_label([fp8], "3b-fp8-unmod"))
    expect("break 7: Q8_0 numbers under the FP8 label", True, lambda: check_label([q8], "3b-fp8-unmod"))
    expect("control: Q8_0 row under a Q8_0 label", False, lambda: check_label([q8], "3b-q8_0"))
    same = CANNED_KLD.format(mean="0.000000", mx="0.000063", top="99.992")
    expect("control: self-KLD at the noise floor (as measured: mean 0, max 6.3e-5, top 99.992 %)", False,
           lambda: check_self_kld(parse_kld(same)))
    off = CANNED_KLD.format(mean="0.000123", mx="0.000063", top="99.992")
    expect("break 8a: mean KLD 0.000123 with the maximum and top-token agreement inside the floor", True, lambda: check_self_kld(parse_kld(off)))
    expect("break 8a2: mean 0 but the maximum KLD 0.002 (above the floor)", True,
           lambda: check_self_kld(parse_kld(CANNED_KLD.format(mean="0.000000", mx="0.002000", top="99.992"))))
    expect("break 8a3: mean 0 but only 99.5 % of the top tokens agree", True,
           lambda: check_self_kld(parse_kld(CANNED_KLD.format(mean="0.000000", mx="0.000063", top="99.500"))))
    huge = CANNED_KLD.format(mean="5.208000", mx="19.0", top="3.100")
    expect("break 8b: 3B against 8B logits is not 'tiny'", True, lambda: check_tiny(parse_kld(huge)))
    small = CANNED_KLD.format(mean="0.000012", mx="0.001", top="99.990")
    expect("control: a tiny KLD passes the tiny classifier", False, lambda: check_tiny(parse_kld(small)))
    expect("break: no statistics in the log", True, lambda: check_tiny(parse_kld("nothing here")))
    expect("break: no statistics in the log (self check)", True, lambda: check_self_kld(parse_kld("nothing here")))
    expect("control: rows at one memory clock merge", False,
           lambda: merge_same_clock([("a", "1408", 1), ("b", "1408", 2)]))
    expect("break 9: a row at 1258 MHz inside a 1408 MHz table", True,
           lambda: merge_same_clock([("a", "1408", 1), ("b", "1258", 2), ("c", "1408", 3)]))

    # streamed bytes: a canned GGUF whose tensor sizes are known (320 + 640 + 1280; token_embd 640)
    g = os.path.join(scratch, "canned.gguf")
    canned_gguf(g)
    sb = stream_bytes(g)
    check("known answer: streamed bytes of the canned GGUF are 2240 - 640 = 1600, not the file size",
          sb == dict(tensors=2240, token_embd=640, streamed=1600) and os.path.getsize(g) > 2240, str(sb))
    g2 = os.path.join(scratch, "canned-no-embd.gguf")
    canned_gguf(g2, with_embd=False)
    expect("break: a GGUF without token_embd.weight is refused", True, lambda: stream_bytes(g2))
    g3 = os.path.join(scratch, "canned-no-output.gguf")
    canned_gguf(g3, with_output=False)
    expect("break: a GGUF without output.weight (tied embeddings) is refused", True, lambda: stream_bytes(g3))

    # the real builders on a canned run folder: the share uses streamed bytes; a 1258 MHz job inside a 1408 MHz table raises
    run = canned_run(scratch, "canned-good")
    t = sec_bw(run)
    # 3B fp8 patched: 100 t/s x 3.2 GB = 320 GB/s of the 640 GB/s probe = 50.0 % (a file-size share would be 62.5 %)
    row = [ln for ln in t.splitlines() if ln.startswith("| 3b fp8-patched")]
    check("known answer: the bus share of 100 t/s on 3.2 GB streamed against 640 GB/s is 50.0 %",
          bool(row) and "| 320 | 50.0 % |" in row[0] and "62.5" not in row[0], row[0] if row else "no row")
    check("known answer: KV bytes at 4096 (28 layers, 2 x 1024, F16) are 0.470 GB and the predicted change is -12.8 %",
          bool(row) and "| 0.470 | -12.8 % | -10.0 % |" in row[0], row[0] if row else "no row")
    bad_run = canned_run(scratch, "canned-b4", {"bench-3b-fp8-patched-depth": "1258/1258"})
    expect("control: B4 on a run with one memory clock", False, lambda: sec_b4(run))
    expect("break 9: B4 with one job at 1258 MHz among 1408 MHz jobs", True, lambda: sec_b4(bad_run))
    expect("break 9b: the bus share with the probe at 1258 MHz and the jobs at 1408 MHz", True,
           lambda: sec_bw(canned_run(scratch, "canned-probe", {"probe-bw-plain": "1258/1258"})))
    expect("break 9c: a job whose memory clock changed inside it (1408/1258)", True,
           lambda: sec_b4(canned_run(scratch, "canned-inside", {"bench-3b-q8_0-depth": "1408/1258"})))
    run = canned_run(scratch, "canned-nostream")
    open(os.path.join(run, "models.txt"), "w").write("llama32-3b-fp8-outq8 4000000000 abc\nllama32-3b-q8_0 3800000000 def\n")
    expect("break: models.txt without streamed lines is refused (no silent file-size share)", True, lambda: sec_bw(run))
    # round 2: the item-1 table builder on a canned run folder
    def canned_r2q(name, mut=None):
        d = os.path.join(scratch, name)
        os.makedirs(d, exist_ok=True)
        mut = mut or {}
        bins = {"fp8-unmod": "/b/llama.cpp-latest/build/bin", "fp8-patched": "/b/llama.cpp-latest-patched/build/bin", "q8_0": "/b/llama.cpp-latest/build/bin"}
        files = {"fp8-unmod": "llama32-3b-auto.gguf", "fp8-patched": "llama32-3b-auto.gguf", "q8_0": "llama32-3b-q8_0.gguf"}
        open(os.path.join(d, "r2q-llama32-3b-ref.log"), "w").write(
            f"# r2q-llama32-3b-ref: /b/llama.cpp-latest/build/bin/llama-perplexity -m /r/refs/{mut.get('ref_file', 'llama32-3b-exact-f32.gguf')}\\n"
            f"perplexity: calculating perplexity over {mut.get('ref_chunks', 50)} chunks, n_ctx=512\nFinal estimate: PPL = 9.4601 +/- 0.2\n")
        open(os.path.join(d, "r2q-llama32-3b-ref.meta"), "w").write("uclk_before=1408\nuclk_after=1408\n")
        for v in files:
            base = mut.get("base", 9.4601)
            f = mut.get("file", {}).get(v, files[v])
            b = mut.get("bin", {}).get(v, bins[v])
            body = CANNED_KLD.format(mean="0.000020", mx="0.0001", top="99.9").replace("9.704900 +/-   0.814000", "9.4620 +/- 0.2", 1)
            body = body.replace("Mean PPL(base)                :   9.704900", f"Mean PPL(base)                :   {base}")
            open(os.path.join(d, f"r2q-llama32-3b-{v}.log"), "w").write(
                f"# r2q-llama32-3b-{v}: {b}/llama-perplexity -m /m/{f}\nkl_divergence: computing over {mut.get('chunks', 50)} chunks, n_ctx=512\n" + body)
            open(os.path.join(d, f"r2q-llama32-3b-{v}.meta"), "w").write(f"uclk_before=1408\nuclk_after={mut.get('uclk', 1408) if v == 'q8_0' else 1408}\n")
        return d
    expect("control: r2q table on a clean canned run", False, lambda: sec_r2q(canned_r2q("r2q-ok")))
    expect("break: the Q8_0 label loads the FP8 file", True, lambda: sec_r2q(canned_r2q("r2q-b1", dict(file=dict(q8_0="llama32-3b-auto.gguf")))))
    expect("break: the patched label runs the unmodified build", True, lambda: sec_r2q(canned_r2q("r2q-b2", dict(bin=dict(**{"fp8-patched": "/b/llama.cpp-latest/build/bin"})))))
    expect("break: 4 chunks where 50 are claimed", True, lambda: sec_r2q(canned_r2q("r2q-b3", dict(chunks=4))))
    expect("break: the reference run has 4 chunks", True, lambda: sec_r2q(canned_r2q("r2q-b4", dict(ref_chunks=4))))
    expect("break: the comparisons used another reference (PPL(base) 9.70 against 9.4601)", True, lambda: sec_r2q(canned_r2q("r2q-b5", dict(base=9.7049))))
    expect("break: the reference run loaded the Q8_0 file", True, lambda: sec_r2q(canned_r2q("r2q-b7", dict(ref_file="llama32-3b-q8_0.gguf"))))
    expect("break: one job at another memory clock", True, lambda: sec_r2q(canned_r2q("r2q-b6", dict(uclk=1258))))
    print(f"llm_tables selftest: {'FAILED' if bad else 'every break FAILED as intended, every control PASSED'}")
    # round 2, item 6: window arithmetic and the clock lookup on a canned sampler recording
    smp = [(i * 0.02, 1000.0 if i * 0.02 < 1.0 else 3000.0, 90.0) for i in range(100)]
    wins = test_windows([("pp16", 0.5), ("pp32", 0.5), ("pp64", 1.0)], 2.0)
    check("windows of 3 back-to-back tests end at the last busy sample", wins == [("pp16", 0.0, 0.5), ("pp32", 0.5, 1.0), ("pp64", 1.0, 2.0)], str(wins))
    check("control: the clock inside the first window is 1000 MHz and inside the last 3000", window_clock(smp, 0.0, 0.5)[0] == 1000.0 and window_clock(smp, 1.0, 2.0)[0] == 3000.0)
    check("break: a window shifted by one test would read 3000 MHz for the first test", window_clock(smp, 1.0, 1.5)[0] != window_clock(smp, 0.0, 0.5)[0])
    expect("break: a window outside the recording is refused", True, lambda: window_clock(smp, 5.0, 6.0))
    check("control: Pearson of a perfect line is 1", abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1) < 1e-12)
    check("break: Pearson of a perfect negative line is -1 (a wrong sign would show)", abs(pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1) < 1e-12)
    # round 2, items 7a and 7b: canned run folders
    def canned_streams(name, mut=None):
        d = os.path.join(scratch, name)
        os.makedirs(d, exist_ok=True)
        mut = mut or {}
        for model, pre in (("3b", "llama32-3b"), ("8b", "llama31-8b")):
            for c in CONFIG_ORDER:
                fname, build = R2S_FILE[c]
                for r in range(1, 7):
                    if mut.get("missing") == (f"{model}-{c}", r):
                        continue
                    f = mut.get("file", {}).get(f"{model}-{c}", fname)
                    rows = "\n".join(f"|   128 |    128 |    {b} |   {256 * b} |    0.1 |  1000.0 |    1.0 |    {100.0 * b + r:.2f} |    1.0 |   1.0 |" for b in range(1, 9))
                    open(os.path.join(d, f"r2streams-{model}-{c}-r{r}.log"), "w").write(f"# x: /b/{build}bin/llama-batched-bench -m /m/{pre}{f}\n{rows}\n")
                    open(os.path.join(d, f"r2streams-{model}-{c}-r{r}.meta"), "w").write(f"uclk_before=1408\nuclk_after={1258 if mut.get('uclk') == (model, c, r) else 1408}\n")
        return d
    expect("control: streams table on a clean canned run", False, lambda: sec_r2streams(canned_streams("r2s-ok")))
    expect("break: a repetition is missing", True, lambda: sec_r2streams(canned_streams("r2s-b1", dict(missing=("3b-q8_0", 4)))))
    expect("break: the Q8_0 label loads the FP8 file", True, lambda: sec_r2streams(canned_streams("r2s-b2", dict(file={"8b-q8_0": "-fp8-outq8.gguf"}))))
    expect("break: one repetition at another memory clock", True, lambda: sec_r2streams(canned_streams("r2s-b3", dict(uclk=("3b", "fp8-patched", 2)))))

    def canned_shapes(name, drop=None, reps=3):
        d = os.path.join(scratch, name)
        os.makedirs(d, exist_ok=True)
        for r in range(1, reps + 1):
            lines = []
            for shapes in (SHAPES_3B, SHAPES_8B):
                for m, k, _ in shapes:
                    for n in (1, 8, 32, 512, 2048):
                        for t, us in (("f8_e4m3", 300.0), ("f16", 100.0), ("q8_0", 80.0)):
                            if drop != (t, m, n, k):
                                lines.append(f"  MUL_MAT(type_a={t},type_b=f32,m={m},n={n},k={k},bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0,offset_a=0):   5964 runs -   {us + r:.2f} us/run - 117.44 MFLOP/run -  1 GFLOPS")
            open(os.path.join(d, f"r2shapes-r{r}.log"), "w").write("\n".join(lines) + "\n")
            open(os.path.join(d, f"r2shapes-r{r}.meta"), "w").write("uclk_before=1408\nuclk_after=1408\n")
        return d
    expect("control: shapes table on a clean canned run", False, lambda: sec_r2shapes(canned_shapes("r2sh-ok")))
    expect("break: one type missing for one shape", True, lambda: sec_r2shapes(canned_shapes("r2sh-b1", drop=("f16", 8192, 512, 3072))))
    expect("break: only two of three repetitions", True, lambda: sec_r2shapes(canned_shapes("r2sh-b2", reps=2)))
    ok = sec_r2shapes(canned_shapes("r2sh-v"))
    check("known answer: canned medians f8 302 us, f16 102 us (difference 200 us), 3B gate/up 8192 x 3072 at n=512 implies 3 m k / 200 us = 377 GB/s", "| 8192 x 3072 (gate, up) | 512 | 302.0 | 102.0 | 82.0 |" in ok and "| 377 |" in ok)
    return 1 if bad else 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(selftest(sys.argv[2] if len(sys.argv) > 2 else None))
    if len(sys.argv) == 3 and sys.argv[1] == "tables":
        try:
            print(tables(sys.argv[2]))
        except TableError as e:
            sys.exit(f"FAIL {e}")
        return
    if len(sys.argv) == 3 and sys.argv[1] in ("r2streams", "r2shapes"):
        try:
            print({"r2streams": sec_r2streams, "r2shapes": sec_r2shapes}[sys.argv[1]](sys.argv[2]))
        except TableError as e:
            sys.exit(f"FAIL {e}")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "r2clock":
        try:
            print(sec_r2clock(sys.argv[2]))
        except TableError as e:
            sys.exit(f"FAIL {e}")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "r2q":
        try:
            print(sec_r2q(sys.argv[2]))
        except TableError as e:
            sys.exit(f"FAIL {e}")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "streamed":
        try:
            b = stream_bytes(sys.argv[2])
        except TableError as e:
            sys.exit(f"FAIL {e}")
        print(f"streamed {os.path.basename(sys.argv[2])[:-5]} {b['tensors']} {b['token_embd']} {b['streamed']}")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "check-self":
        try:
            k = parse_kld(ANSI.sub("", open(sys.argv[2], errors="replace").read()))
            check_self_kld(k)
        except (TableError, OSError) as e:
            sys.exit(f"FAIL {e}")
        print(f"PASS self-comparison at the noise floor: mean {k['mean_kld']}, max {k.get('max_kld')}, same top {k.get('same_top')} %")
        return
    sys.exit("usage: llm_tables.py selftest [dir] | tables <run folder> | streamed <model.gguf> | check-self <kld log>")


if __name__ == "__main__":
    main()
