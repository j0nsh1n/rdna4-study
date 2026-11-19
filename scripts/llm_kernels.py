#!/usr/bin/env python3
"""Reads a rocprofv3 kernel-trace CSV of a llama.cpp run and says where the time went (stdlib only).

Usage:
  llm_kernels.py report <kernel_trace.csv> --kind pp|tg [--reps N] [--control-ts T --tokens N] [--model 3b|8b]
                 [--weight-bytes B] [--json OUT]
      Splits the trace into passes (pp: one llama_decode of a prompt batch; tg: one generated token), keeps the
      passes of the commonest kernel sequence, and prints per-group time, gaps between kernels and checks.
      Prints VERDICT PASS/FAIL lines; exit 1 if any check fails.
  llm_kernels.py groups <kernel_trace.csv> --expect GROUP[,GROUP] [--min-share S]
      Whole-file shares for single-operation runs; exit 1 if the named groups together hold no more than S.
  llm_kernels.py selftest [scratch folder]
      the helper's own checks and deliberate breaks (each must FAIL, and trip the check it names)

Groups (names frozen after the step-2 and step-5 traces of this study; a name that matches none lands in "other"):
  widen (FP8 or Q8_0 weights widened to FP16), act_convert (activations to FP16), scale_mul (the per-matrix scale
  multiply of the FP8 route), gemm_fp16 (hipBLASLt/Tensile and rocBLAS GEMMs), mmq (int8 matrix kernel),
  matvec_fp8, matvec_q (decode kernels), quantize_act (activations to int8), attention, elementwise, runtime_copy.
Labels in the output follow the study: everything printed here is MEASURED from the named trace file.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

GROUPS = [
    ("widen", r"convert_unary_cont_vec4<ggml_fp8_e4m3_t, __half>|convert_fp8_scaled_f16|dequantize_block"),
    ("act_convert", r"convert_unary"),
    ("scale_mul", r"k_bin_bcast<&\(op_mul"),
    ("gemm_fp16", r"^Cijk_|^gemm|rocblas|hipblas"),
    ("mmq", r"mul_mat_q<"),
    ("matvec_fp8", r"mul_mat_vec_f<ggml_fp8"),
    ("matvec_q", r"mul_mat_vec_q<"),
    ("quantize_act", r"quantize_q8_1|quantize_mmq_q8_1"),
    ("attention", r"flash_attn|fattn"),
    ("elementwise", r"rms_norm|rope_norm|k_bin_bcast|unary_gated|k_set_rows|k_get_rows|scale_f32|soft_max|"
                    r"k_sum|unary_op|clamp|k_repeat|cpy_|concat|argsort|k_argmax"),
    ("runtime_copy", r"^__amd_rocclr"),
]
GROUP_NAMES = [g for g, _ in GROUPS] + ["other"]
COMPILED = [(g, re.compile(r)) for g, r in GROUPS]

# Weight shapes (K inputs, N outputs) of a dense Llama layer, in the order llama.cpp issues them.
MODELS = {
    "3b": dict(layers=28, embd=3072, ff=8192, kv=1024),
    "8b": dict(layers=32, embd=4096, ff=14336, kv=1024),
}


def classify(name):
    for g, rx in COMPILED:
        if rx.search(name):
            return g
    return "other"


def load(path):
    """Kernel rows sorted by start time: (name, start_ns, end_ns). A missing file gives no rows."""
    rows = []
    try:
        f = open(path, newline="")
    except OSError:
        return rows
    with f:
        for r in csv.DictReader(f):
            rows.append((r["Kernel_Name"], int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    rows.sort(key=lambda x: x[1])
    return rows


def control_ts(log):
    """Median tokens/s of the first result line (llama-bench jsonl) in a job log."""
    with open(log) as f:
        for line in f:
            if line.startswith("{"):
                return statistics.median(json.loads(line)["samples_ts"])
    raise SystemExit(f"no llama-bench result line in {log}")


def segment(rows, gap_ns):
    segs, cur = [], []
    for r in rows:
        if cur and r[1] - cur[-1][2] > gap_ns:
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    return segs


def pct(vals, p):
    v = sorted(vals)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def analyse(rows, kind, reps=None, expect_passes=None, min_passes=None):
    """Returns a dict of results and a list of (verdict name, ok, text)."""
    checks = []
    model = [r for r in rows if classify(r[0]) != "runtime_copy"]
    checks.append(("kernel-rows-found", len(model) > 0, f"{len(model)} model kernel rows"))
    if not model:
        return {}, checks
    gap_ns = 1_000_000 if kind == "pp" else 150_000
    segs = segment(model, gap_ns)
    sizes = [len(s) for s in segs]
    # the commonest size among the big segments (start-up kernels of the first pass form small ones)
    big = [x for x in sizes if x >= 0.5 * max(sizes)]
    modal = max(set(big), key=lambda x: (big.count(x), x))
    sel = [s for s in segs if len(s) == modal]
    if reps is not None:
        sel = sel[-reps:] if len(sel) > reps else sel
    seqs = {tuple(k[0] for k in s) for s in sel}
    checks.append(("passes-identical", len(seqs) == 1,
                   f"{len(sel)} passes of {modal} kernels, {len(seqs)} different kernel sequences"))
    if expect_passes is not None:
        checks.append(("total-kernel-count", len(model) == expect_passes * modal,
                       f"{len(model)} model kernel rows, {expect_passes} passes x {modal} = {expect_passes * modal} expected"))
    if reps is not None:
        need = reps if min_passes is None else min_passes
        checks.append(("passes-count", len(sel) >= need, f"{len(sel)} passes kept, at least {need} wanted"))
    walls, busys, gapsums, ngaps = [], [], [], []
    gap_all = []
    group_time = {g: [] for g in GROUP_NAMES}
    overlap = 0
    for s in sel:
        wall = s[-1][2] - s[0][1]
        busy = sum(e - b for _, b, e in s)
        gaps = [s[i + 1][1] - s[i][2] for i in range(len(s) - 1)]
        overlap += sum(1 for g in gaps if g < 0)
        walls.append(wall)
        busys.append(busy)
        gapsum = sum(g for g in gaps)
        gapsums.append(gapsum)
        gap_all.extend(g / 1000.0 for g in gaps)
        gt = {g: 0 for g in GROUP_NAMES}
        for n, b, e in s:
            gt[classify(n)] += e - b
        for g in GROUP_NAMES:
            group_time[g].append(gt[g])
    med = statistics.median
    res = dict(
        kind=kind, passes=len(sel), kernels_per_pass=modal,
        wall_ms=med(walls) / 1e6, busy_ms=med(busys) / 1e6, gap_ms=med(gapsums) / 1e6,
        wall_ms_min=min(walls) / 1e6, wall_ms_max=max(walls) / 1e6,
        groups_ms={g: med(group_time[g]) / 1e6 for g in GROUP_NAMES},
        gap_us=dict(median=med(gap_all), p95=pct(gap_all, 95), p99=pct(gap_all, 99), max=max(gap_all),
                    min=min(gap_all)),
        gap_hist={lab: sum(1 for g in gap_all if lo <= g < hi)
                  for lab, lo, hi in (("<2us", -1e9, 2), ("2-3us", 2, 3), ("3-4us", 3, 4), ("4-6us", 4, 6),
                                      ("6-10us", 6, 10), ("10-50us", 10, 50), (">=50us", 50, 1e12))},
        n_gaps=len(gap_all),
    )
    res["other_share"] = res["groups_ms"]["other"] / max(res["busy_ms"], 1e-9)
    checks.append(("other-below-5pct", res["other_share"] < 0.05,
                   f"group 'other' holds {100 * res['other_share']:.2f} % of kernel time"))
    checks.append(("no-overlap", overlap == 0, f"{overlap} kernel pairs overlap in time"))
    chk = abs(res["busy_ms"] + res["gap_ms"] - res["wall_ms"]) / max(res["wall_ms"], 1e-9)
    checks.append(("kernels-plus-gaps-equal-wall", chk < 0.01,
                   f"kernel sum {res['busy_ms']:.3f} + gaps {res['gap_ms']:.3f} against wall {res['wall_ms']:.3f} ms"))
    # per-kernel-in-pass detail kept for the shape tables
    res["_sel"] = sel
    return res, checks


def gemm_shapes(res, model, kind_w):
    """Time and TFLOPS (or GB/s) per weight position of one pass. pp: the 7 matmuls of a layer, in issue order
    (the last layer's three full-size ones first; the later ones are matrix-vector). tg: the 6 per layer."""
    m = MODELS[model]
    e, f, kv = m["embd"], m["ff"], m["kv"]
    pp_shapes = [("q", e, e), ("k", e, kv), ("v", e, kv), ("o", e, e), ("gate", e, f), ("up", e, f), ("down", f, e)]
    tg_shapes = [("q", e, e), ("k", e, kv), ("v", e, kv), ("o", e, e), ("gate+up", e, 2 * f), ("down", f, e)]
    out = []
    sel = res["_sel"]
    big = ("gemm_fp16", "mmq")
    if res["kind"] == "pp":
        per_pos = {i: [] for i in range(7)}
        for s in sel:
            ks = [(n, e2 - b2) for n, b2, e2 in s if classify(n) in big]
            n_full = (m["layers"] - 1) * 7 + 3
            for j, (_, d) in enumerate(ks[:n_full]):
                per_pos[j % 7].append(d)
        for i, (nm, k, n) in enumerate(pp_shapes):
            if per_pos[i]:
                t = statistics.median(per_pos[i]) / 1e9
                out.append(dict(pos=nm, K=k, N=n, calls=len(per_pos[i]), median_ms=t * 1e3,
                                tflops=2.0 * 512 * k * n / t / 1e12))
    else:
        per_pos = {i: [] for i in range(6)}
        for s in sel:
            ks = [(n, e2 - b2) for n, b2, e2 in s if classify(n) in ("matvec_fp8", "matvec_q")]
            n_layer = m["layers"] * 6
            for j, (_, d) in enumerate(ks[:n_layer]):
                per_pos[j % 6].append(d)
        for i, (nm, k, n) in enumerate(tg_shapes):
            if per_pos[i]:
                t = statistics.median(per_pos[i]) / 1e9
                out.append(dict(pos=nm, K=k, N=n, calls=len(per_pos[i]), median_us=t * 1e6,
                                gbps=k * n * kind_w / t / 1e9))
    return out


def render(res, checks, path):
    lines = [f"trace file: {path}",
             f"MEASURED {res['kind']}: {res['passes']} passes of {res['kernels_per_pass']} kernels (median per pass)",
             f"  wall {res['wall_ms']:.3f} ms (min {res['wall_ms_min']:.3f}, max {res['wall_ms_max']:.3f}); "
             f"kernels {res['busy_ms']:.3f} ms; gaps {res['gap_ms']:.3f} ms ({100 * res['gap_ms'] / res['wall_ms']:.1f} %)"]
    for g in GROUP_NAMES:
        t = res["groups_ms"][g]
        if t > 0:
            lines.append(f"  {g:13s} {t:9.3f} ms  {100 * t / res['busy_ms']:5.1f} % of kernel time")
    gu = res["gap_us"]
    lines.append(f"  gaps between kernels: median {gu['median']:.2f} us, p95 {gu['p95']:.2f}, p99 {gu['p99']:.2f}, "
                 f"min {gu['min']:.2f}, max {gu['max']:.1f} (n {res['n_gaps']})")
    lines.append("  gap histogram: " + ", ".join(f"{k} {v}" for k, v in res["gap_hist"].items()))
    for name, ok, text in checks:
        lines.append(f"VERDICT {'PASS' if ok else 'FAIL'} {name}: {text}")
    return "\n".join(lines)


OVERHEAD_LIMIT = 1.15


def overhead_check(wall_ms, ctl_ms):
    """The profiler's own cost: the profiled pass against the unprofiled control. Returns (ratio, check)."""
    ratio = wall_ms / ctl_ms
    return ratio, ("profiler-overhead-below-15pct", ratio <= OVERHEAD_LIMIT,
                   f"profiled pass {wall_ms:.3f} ms against unprofiled {ctl_ms:.3f} ms: x{ratio:.3f}"
                   + ("; absolute times of this trace are not quoted, shares only" if ratio > OVERHEAD_LIMIT else ""))


def cmd_report(a):
    rows = load(a.csv)
    res, checks = analyse(rows, a.kind, a.reps, a.expect_passes, a.min_passes)
    if not res:
        print("\n".join(f"VERDICT {'PASS' if ok else 'FAIL'} {n}: {t}" for n, ok, t in checks))
        return 1
    if a.control_log:
        a.control_ts = control_ts(a.control_log)
        a.tokens = a.tokens or (512 if a.kind == "pp" else 1)
    if a.control_ts and a.tokens:
        ctl_ms = 1000.0 * a.tokens / a.control_ts
        ratio, chk = overhead_check(res["wall_ms"], ctl_ms)
        res["control_ms_per_pass"] = ctl_ms
        # the unprofiled wall minus the profiled kernel sum, per gap: what the gaps cost when nothing profiles
        res["implied_gap_us_unprofiled"] = (ctl_ms - res["busy_ms"]) * 1000.0 / max(res["kernels_per_pass"] - 1, 1)
        res["profiled_over_control"] = ratio
        res["shares_only"] = ratio > OVERHEAD_LIMIT
        checks.append(chk)
    if a.model:
        w = a.weight_bytes
        res["shapes"] = gemm_shapes(res, a.model, w)
    print(render(res, checks, a.csv))
    if a.model:
        for s in res["shapes"]:
            if "tflops" in s:
                print(f"  shape {s['pos']:5s} K={s['K']} N={s['N']}: median {s['median_ms']:.3f} ms, {s['tflops']:.1f} TFLOPS-equivalent")
            else:
                print(f"  shape {s['pos']:7s} K={s['K']} N={s['N']}: median {s['median_us']:.1f} us, {s['gbps']:.0f} GB/s")
    if a.json:
        out = {k: v for k, v in res.items() if not k.startswith("_")}
        out["checks"] = [dict(name=n, ok=ok, text=t) for n, ok, t in checks]
        out["trace"] = a.csv
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if all(ok for _, ok, _ in checks) else 1


def cmd_groups(a):
    """Whole-file shares (no pass structure), for single-operation runs such as test-backend-ops."""
    rows = [r for r in load(a.csv) if classify(r[0]) != "runtime_copy"]
    if not rows:
        print("VERDICT FAIL kernel-rows-found: 0 model kernel rows")
        return 1
    t = {g: 0 for g in GROUP_NAMES}
    for n, b, e in rows:
        t[classify(n)] += e - b
    tot = sum(t.values())
    print(f"trace file: {a.csv}: {len(rows)} kernels, {tot / 1e6:.3f} ms")
    for g in GROUP_NAMES:
        if t[g]:
            print(f"  {g:13s} {t[g] / 1e6:10.3f} ms {100 * t[g] / tot:5.1f} %")
    ok = True
    if a.expect:
        names = a.expect.split(",")
        unknown = [n for n in names if n not in t]
        if unknown:
            print(f"VERDICT FAIL group-name: unknown group {unknown}")
            return 1
        share = sum(t[n] for n in names) / tot
        ok = share > a.min_share
        print(f"VERDICT {'PASS' if ok else 'FAIL'} group-{'+'.join(names)}-above-{int(100 * a.min_share)}pct: {100 * share:.1f} %")
    return 0 if ok else 1

# ---------------------------------------------------------------------------------------------------------------
# Round 2: overlap evidence (item 3) and per-group comparison of two traces (item 5)
def load_full(path):
    """Kernel rows with ids, sorted by start: (name, start, end, queue, stream, dispatch)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["Kernel_Name"], int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Queue_Id"], r["Stream_Id"],
                         int(r["Dispatch_Id"])))
    rows.sort(key=lambda x: x[1])
    return rows


def find_overlaps(rows, gap_ns=150_000):
    """Pairs where a kernel starts before an earlier-started one has ended (a running maximum of the ends, so a kernel
    that overlaps a long one is found too). Each hit: both kernels, queue and stream ids, overlap in ns, dispatch id
    difference, and the position of the second kernel inside its pass (passes split at gaps above gap_ns)."""
    model = [r for r in rows if classify(r[0]) != "runtime_copy"]
    pos = {}
    for seg in segment([(r[0], r[1], r[2], r) for r in model], gap_ns):
        for i, x in enumerate(seg):
            pos[x[3][5]] = (i, len(seg))
    hits = []
    top = None
    for r in model:
        if top is not None and r[1] < top[2]:
            hits.append(dict(first=top[0], second=r[0], first_group=classify(top[0]), second_group=classify(r[0]),
                             same_queue=top[3] == r[3], same_stream=top[4] == r[4], overlap_ns=top[2] - r[1],
                             dispatch_gap=r[5] - top[5], first_ns=top[2] - top[1], second_ns=r[2] - r[1],
                             pos_in_pass=pos.get(r[5], (None, None))[0], pass_len=pos.get(r[5], (None, None))[1],
                             data_dependent=None))
        if top is None or r[2] > top[2]:
            top = r
    return dict(rows=len(model), queues=sorted({r[3] for r in model}), pairs=hits)


def cmd_overlap(a):
    allpos = []
    for path in a.csv:
        res = find_overlaps(load_full(path))
        h = res["pairs"]
        print(f"{path}: {res['rows']} kernel rows, queues {res['queues']}, {len(h)} overlapping pairs")
        for x in h:
            print(f"  {x['overlap_ns']:5d} ns  {x['first_group']}/{x['first'][:34]} ({x['first_ns']} ns) -> {x['second_group']}/{x['second'][:34]}"
                  f" ({x['second_ns']} ns)  same queue {x['same_queue']}, same stream {x['same_stream']}, dispatch +{x['dispatch_gap']}, position {x['pos_in_pass']}/{x['pass_len']}")
        allpos.append({(x["first_group"], x["second_group"], x["pos_in_pass"]) for x in h})
    if len(allpos) > 1:
        from collections import Counter
        c = Counter(k for s in allpos for k in s)
        rec = {k: v for k, v in c.items() if v >= 2}
        print(f"overlap kinds (first group, second group, position in pass) seen in at least 2 of {len(allpos)} traces: {len(rec)}")
        for k, v in sorted(rec.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {v} traces: {k}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({p: find_overlaps(load_full(p)) for p in a.csv}, f, indent=1)
    return 0

def cmd_topk(a):
    """The kernels of a trace by total time: name, count, median duration (us), share. For naming the kernels a library call picks."""
    rows = load(a.csv)
    if not rows:
        print("FAIL no kernel rows")
        return 1
    agg = {}
    for n, b, e in rows:
        agg.setdefault(n, []).append(e - b)
    tot = sum(sum(v) for v in agg.values())
    print(f"{a.csv}: {len(rows)} kernels, {tot / 1e6:.3f} ms in all")
    for n, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:a.n]:
        print(f"  {100 * sum(v) / tot:5.1f} %  x{len(v):5d}  median {statistics.median(v) / 1000:9.2f} us  {n[:a.width]}")
    return 0

PMC_MEMORY_COUNTERS = re.compile(r"GL2C|TCP|FETCH|L2Cache|MemUnit|SQ_INSTS|SQ_WAVE32|SQ_WAVE64|SQ_INST_CYCLES|SQ_WAVE_CYCLES|VALU")


def cmd_pmc(a):
    """Per kernel and counter: the median of a counter-collection CSV. A counter that reads 0 in every dispatch of a kernel that is
    known to move data (--moves-data names it) is reported as NONFUNCTIONAL: it was listed and accepted, but it counts nothing."""
    rows = {}
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            if a.kernel in r["Kernel_Name"]:
                rows.setdefault((r["Kernel_Name"][:40], r["Counter_Name"]), []).append(float(r["Counter_Value"]))
    if not rows:
        print(f"FAIL {a.csv}: no rows for a kernel named *{a.kernel}*")
        return 1
    bad = 0
    for (k, c), v in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        med = statistics.median(v)
        dead = (max(v) == 0.0) and a.moves_data
        bad += dead
        print(f"{'NONFUNCTIONAL' if dead else 'ok':13s} {c:26s} {k:42s} n={len(v):3d} median {med:.6g}")
    return 1 if bad else 0


def pass_groups(rows, kind, reps=None):
    """Median per-group time (ms), pass wall time and kernel count of the commonest pass: the `analyse` numbers, without the checks."""
    res, _ = analyse(rows, kind, reps=reps)
    return res


def cmd_groupcmp(a):
    ra = pass_groups(load(a.csv_a), a.kind)
    rb = pass_groups(load(a.csv_b), a.kind)
    if not ra or not rb:
        print("FAIL no kernel rows")
        return 1
    na, nb = a.tokens_a, a.tokens_b
    print(f"A {a.csv_a}: {ra['passes']} passes of {ra['kernels_per_pass']} kernels, {na} tokens each; B {a.csv_b}: {rb['passes']} passes of {rb['kernels_per_pass']} kernels, {nb} tokens each")
    print(f"per {a.per} tokens (ms): A wall {ra['wall_ms'] * a.per / na:.3f}, B wall {rb['wall_ms'] * a.per / nb:.3f}; kernel sum A {ra['busy_ms'] * a.per / na:.3f}, B {rb['busy_ms'] * a.per / nb:.3f}; gaps A {ra['gap_ms'] * a.per / na:.3f}, B {rb['gap_ms'] * a.per / nb:.3f}")
    print(f"{'group':14s} {'A ms':>9s} {'B ms':>9s} {'B-A ms':>9s}")
    for g in GROUP_NAMES:
        x, y = ra["groups_ms"][g] * a.per / na, rb["groups_ms"][g] * a.per / nb
        if x or y:
            print(f"{g:14s} {x:9.3f} {y:9.3f} {y - x:9.3f}")
    if a.model:
        m = MODELS[a.model]
        flop_tok = 2 * m["layers"] * (2 * m["embd"] * m["embd"] + 2 * m["embd"] * m["kv"] + 3 * m["embd"] * m["ff"])
        for g in ("gemm_fp16", "mmq"):
            x, y = ra["groups_ms"][g], rb["groups_ms"][g]
            if x and y:
                print(f"{g} rate (layer matrices only, {flop_tok / 1e9:.2f} GFLOP per token): A {flop_tok * na / (x * 1e9):.1f} T(FLOP|OP)/s, B {flop_tok * nb / (y * 1e9):.1f}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# Selftest on canned traces: three prompt passes of a made-up FP8 route with known durations. The breaks that concern
# the report's own checks go through cmd_report, the code the pipeline runs, on a canned CSV file.
def canned(passes=4, drop=None, rename=None, rename_all=False, extra_init=True):
    rows, t = [], 1_000_000
    if extra_init:
        rows.append(("__amd_rocclr_copyBuffer", t, t + 5000))
        t += 400_000_000
    for p in range(passes):
        kern = []
        for layer in range(2):
            kern += [("void convert_unary_cont_vec4<ggml_fp8_e4m3_t, __half>(void const*, __half*, long)", 100_000),
                     ("void convert_unary_cont_vec4<float, __half>(void const*, __half*, long)", 20_000),
                     ("Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64", 300_000),
                     ("void k_bin_bcast<&(op_mul(float, float)), float, float, float, float const*>(x)", 30_000),
                     ("void flash_attn_ext_f16<128, 128, 16, 4, false, false, false>(char const*)", 150_000)]
        for i, (n, d) in enumerate(kern):
            if rename and i == rename[0] and (rename_all or p == 1):
                n = rename[1]
            if drop is not None and i == drop and p == 1:
                continue
            rows.append((n, t, t + d))
            t += d + 3000
        t += 3_000_000
    return rows


def write_trace(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Kernel_Name", "Start_Timestamp", "End_Timestamp"])
        for n, b, e in rows:
            w.writerow([n, b, e])


def run_report(rows, scratch, control_ms=None):
    """cmd_report on a canned trace file, as the pipeline calls it. Returns (exit status, printed text)."""
    import argparse as ap
    import contextlib
    import io
    path = os.path.join(scratch, "canned_kernel_trace.csv")
    write_trace(rows, path)
    a = ap.Namespace(csv=path, kind="pp", reps=3, expect_passes=4, min_passes=None, control_log=None,
                     control_ts=(512 * 1000.0 / control_ms) if control_ms else None, tokens=512 if control_ms else None,
                     model=None, weight_bytes=1.0, json=None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_report(a)
    return rc, buf.getvalue()


def selftest(scratch=None):
    bad = 0
    if scratch is None:
        scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "selftest-kernels")
    os.makedirs(scratch, exist_ok=True)

    def expect(name, want_fail, checks, must=None):
        """must: the named check has to be among the failures (a break that trips another check proves nothing)."""
        nonlocal bad
        failed = [n for n, ok, _ in checks if not ok]
        got = bool(failed)
        right = got == want_fail and (must is None or not want_fail or must in failed)
        if not right:
            bad += 1
        print(f"  {'ok' if right else 'WRONG':5s} {name}: {'FAIL' if got else 'PASS'} seen"
              f"{' (' + ', '.join(failed) + ')' if failed else ''} (wanted {'FAIL' if want_fail else 'PASS'}"
              f"{' of ' + must if must and want_fail else ''})")

    def report_checks(rc, text):
        return [(ln.split()[2].rstrip(":"), ln.split()[1] == "PASS", ln) for ln in text.splitlines() if ln.startswith("VERDICT")]

    res, checks = analyse(canned(), "pp", reps=3, expect_passes=4)
    expect("control: canned clean trace", False, checks)
    g = res["groups_ms"]
    known = {"widen": 0.2, "act_convert": 0.04, "gemm_fp16": 0.6, "scale_mul": 0.06, "attention": 0.3}
    ok = all(abs(g[k] - v) < 1e-6 for k, v in known.items())
    expect("known answer: group sums of the canned trace", False, [("group-sums", ok, str(g))])
    expect("break: one kernel row deleted from one pass", True, analyse(canned(drop=2), "pp", reps=3, expect_passes=4)[1],
           must="total-kernel-count")
    expect("break: a kernel renamed in one pass only", True,
           analyse(canned(rename=(2, "unknown_kernel_xyz")), "pp", reps=3, expect_passes=4)[1], must="passes-identical")
    wall = res["wall_ms"]
    # the next four go through cmd_report: the pipeline's own code, on a canned CSV
    expect("control: report on the clean trace, profiled 2 % slower than the control", False,
           report_checks(*run_report(canned(), scratch, control_ms=wall / 1.02)))
    expect("break: a kernel name the groups do not know in every pass (other above 5 %)", True,
           report_checks(*run_report(canned(rename=(2, "unknown_kernel_xyz"), rename_all=True), scratch)),
           must="other-below-5pct")
    expect("break: profiled run 50 % slower than the unprofiled control", True,
           report_checks(*run_report(canned(), scratch, control_ms=wall / 1.5)), must="profiler-overhead-below-15pct")
    rc, text = run_report([], scratch)
    expect("break: no kernel rows at all (a regex that matches nothing)", True, report_checks(rc, text),
           must="kernel-rows-found")
    if rc != 1:
        print(f"  WRONG report on an empty trace returned {rc}, wanted 1")
        bad += 1
    # grouping known answer, as `groups` is run on the f16 test-backend-ops trace
    import argparse as ap
    import contextlib
    import io

    def run_groups(rows, expect_groups, min_share):
        path = os.path.join(scratch, "canned_groups.csv")
        write_trace(rows, path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc2 = cmd_groups(ap.Namespace(csv=path, expect=expect_groups, min_share=min_share))
        return rc2, buf.getvalue()

    f16 = [("Cijk_Alik_Bljk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x64", 0, 884_000),
           ("void convert_unary_cont_vec4<float, __half>(void const*, __half*, long)", 900_000, 1_000_000)]
    rc2, _ = run_groups(f16, "gemm_fp16,act_convert", 0.95)
    expect("control: GEMM plus activation convert hold 100 % of a canned f16 run", False,
           [("groups", rc2 == 0, "")])
    rc2, _ = run_groups(f16, "gemm_fp16", 0.95)
    expect("break: the GEMM group alone is 88.4 %, below 95 %", True, [("groups", rc2 == 0, "")], must="groups")
    other = f16 + [("some_unknown_kernel", 1_000_000, 1_400_000)]
    rc2, _ = run_groups(other, "gemm_fp16,act_convert", 0.95)
    expect("break: an unknown kernel takes 28 % of the run", True, [("groups", rc2 == 0, "")], must="groups")
    # round 2: overlap finder. rows are (name, start, end, queue, stream, dispatch)
    def mk(over, q2="1"):
        return [("k_set_rows<float>", 1000, 2000, "1", "1", 1), ("flash_attn_tile<128>", 2000 - over, 3000, q2, q2, 2),
                ("rms_norm_f32", 3100, 3500, "1", "1", 3)]
    expect("control: back-to-back kernels have no overlap", False, [("overlap", not find_overlaps(mk(-50))["pairs"], "")])
    h = find_overlaps(mk(300))["pairs"]
    expect("control: a planted 300 ns overlap on one queue is found, with the right size and ids", False,
           [("overlap", len(h) == 1 and h[0]["overlap_ns"] == 300 and h[0]["same_queue"] and h[0]["dispatch_gap"] == 1, str(h))])
    expect("break: a clean trace must not give a hit (a finder that always finds one fails here)", True,
           [("overlap", len(find_overlaps(mk(-50))["pairs"]) == 1, "")], must="overlap")
    h2 = find_overlaps(mk(300, q2="2"))["pairs"]
    expect("control: the same overlap on another queue is reported as a different queue", False,
           [("overlap", len(h2) == 1 and not h2[0]["same_queue"], str(h2))])
    expect("break: the other-queue overlap must not be reported as the same queue", True,
           [("overlap", len(h2) == 1 and h2[0]["same_queue"], str(h2))], must="overlap")
    long_k = [("big", 0, 100_000, "1", "1", 1), ("a", 10_000, 20_000, "1", "1", 2), ("b", 50_000, 60_000, "1", "1", 3)]
    expect("control: two kernels inside one long kernel are both found (running maximum, not only neighbours)", False,
           [("overlap", len(find_overlaps(long_k)["pairs"]) == 2, "")])
    expect("break: only neighbours counted would find one, not two", True,
           [("overlap", len(find_overlaps(long_k)["pairs"]) == 1, "")], must="overlap")
    # round 2: the counter summary. A canned counter-collection file: one working counter and one that reads 0 for a kernel that moves data
    import argparse as _ap
    import contextlib as _cl
    import io as _io
    def pmc_rc(vals):
        path = os.path.join(scratch, "canned_pmc.csv")
        with open(path, "w", newline="") as f:
            f.write('"Kernel_Name","Counter_Name","Counter_Value"\n')
            for c, v in vals:
                f.write(f'"read_kernel(x)","{c}",{v}\n')
        with _cl.redirect_stdout(_io.StringIO()):
            return cmd_pmc(_ap.Namespace(csv=path, kernel="read_kernel", moves_data=True))
    expect("control: SQ_WAVES counts 8192 on a kernel that moves data", False, [("pmc", pmc_rc([("SQ_WAVES", 8192)]) == 0, "")])
    expect("break: GL2C_EA_RDREQ_sum reads 0 on a kernel that reads 256 MiB is reported NONFUNCTIONAL", True,
           [("pmc", pmc_rc([("SQ_WAVES", 8192), ("GL2C_EA_RDREQ_sum", 0)]) == 0, "")], must="pmc")
    expect("break: no rows for the named kernel", True, [("pmc", cmd_pmc(_ap.Namespace(csv=os.path.join(scratch, "canned_pmc.csv"), kernel="nonexistent_kernel", moves_data=True)) == 0, "")], must="pmc")
    print(f"llm_kernels selftest: {'FAILED' if bad else 'every break FAILED as intended, every control PASSED'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("report")
    r.add_argument("csv")
    r.add_argument("--kind", choices=["pp", "tg"], required=True)
    r.add_argument("--reps", type=int)
    r.add_argument("--expect-passes", type=int, help="passes the run made in all (warm-up included): the kernel rows must be that many times one pass")
    r.add_argument("--min-passes", type=int, help="fewest passes of the commonest sequence to accept (default --reps); a few tokens of a long run split at a host-side pause")
    r.add_argument("--control-log", help="llama-bench jsonl log of the unprofiled control (sets --control-ts and --tokens)")
    r.add_argument("--control-ts", type=float, help="tokens/s of the unprofiled control")
    r.add_argument("--tokens", type=int, help="tokens per pass in the control (512 for pp, 1 for tg)")
    r.add_argument("--model", choices=["3b", "8b"])
    r.add_argument("--weight-bytes", type=float, default=1.0, help="bytes per weight (1 for FP8, 1.0625 for Q8_0)")
    r.add_argument("--json")
    g = sp.add_parser("groups")
    g.add_argument("csv")
    g.add_argument("--expect", help="group (or groups joined by commas) that together must hold more than --min-share of the kernel time")
    g.add_argument("--min-share", type=float, default=0.9)
    o = sp.add_parser("overlap")
    o.add_argument("csv", nargs="+")
    o.add_argument("--json")
    c = sp.add_parser("groupcmp")
    c.add_argument("csv_a")
    c.add_argument("csv_b")
    c.add_argument("--kind", choices=["pp", "tg"], default="pp")
    c.add_argument("--tokens-a", type=int, required=True)
    c.add_argument("--tokens-b", type=int, required=True)
    c.add_argument("--per", type=int, default=512, help="tokens to scale both traces to")
    c.add_argument("--model", choices=["3b", "8b"], help="adds the matrix-kernel rate from the layer matrices' FLOPs")
    t = sp.add_parser("topk")
    t.add_argument("csv")
    t.add_argument("-n", type=int, default=6)
    t.add_argument("--width", type=int, default=150)
    pm = sp.add_parser("pmc")
    pm.add_argument("csv")
    pm.add_argument("--kernel", default="read_kernel")
    pm.add_argument("--moves-data", action="store_true", help="the named kernel certainly reads or writes memory: an all-zero memory or instruction counter is then NONFUNCTIONAL")
    st = sp.add_parser("selftest")
    st.add_argument("scratch", nargs="?", help="folder for the canned trace files (default runs/selftest-kernels)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest(a.scratch))
    if a.cmd == "groups":
        sys.exit(cmd_groups(a))
    if a.cmd == "pmc":
        sys.exit(cmd_pmc(a))
    if a.cmd == "topk":
        sys.exit(cmd_topk(a))
    if a.cmd == "overlap":
        sys.exit(cmd_overlap(a))
    if a.cmd == "groupcmp":
        sys.exit(cmd_groupcmp(a))
    sys.exit(cmd_report(a))


if __name__ == "__main__":
    main()
