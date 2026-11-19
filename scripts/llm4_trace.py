#!/usr/bin/env python3
"""Round 4: analyse a rocprofv3 kernel trace of a PyTorch or vLLM run. Pure Python (no GPU), run with python3 -I.
  llm4_trace.py report <t_kernel_trace.csv> --reps N [--mark-a GRID --mark-b GRID] [--json out.json]
  llm4_trace.py selftest
The harnesses launch a marker kernel (torch.ones of 1,234,567 floats, grid 308,736) before and another (2,345,678 floats, grid 586,496) after the
measured passes. The region between them is split into groups by kernel name; gaps are the idle times between one kernel's end and the next's start
on the one queue the work runs on (the timeline is sorted by start time; a kernel that starts before the previous one ended counts as no gap).
"""
import csv, json, re, statistics, sys

MARK_A, MARK_B = 308736, 586496
# first match wins; names are the profiler's Kernel_Name
GROUPS = [
    ("gemm_fp8", r"Cijk_.*F8|f8f8|fp8.*gemm|gemm.*fp8|e4m3.*gemm|scaled_mm|_fp8_.*matmul|wvSplitK.*fp8|wvSplitKQ|fp8_gemm"),
    ("gemm_16bit", r"Cijk_|gemm|Gemm|GEMM|wvSplitK|skinny|matmul"),
    ("fused_triton", r"^triton_"),
    ("attention", r"_fwd_kernel|flash|attn|attention|sdpa|fmha|paged|softmax|mha|unified_attention"),
    ("kv_cache_write", r"reshape_and_cache|kv_cache|cache_kernel|concat_and_cache"),
    ("rms_norm", r"rms_norm|rmsnorm|layer_norm|RMSNorm|fused_add_rms"),
    ("rope", r"rotary|rope"),
    ("act_mul", r"silu_and_mul|act_and_mul|SiLU|swiglu"),
    ("fp8_quantize", r"ConvertToFloat8|launch_clamp_scalar|scaled_fp8_quant|static_scaled|dynamic_per_token|per_tensor_quant|quant_kernel"),
    ("embedding_gather", r"index_select|embedding|IndexKernel|gather"),
    ("copy_fill", r"copy|Copy|memcpy|rocclr|fillBuffer|Fill"),
    ("elementwise_other", r"elementwise|Elementwise|reduce_kernel|triu|mul_|add_|cat"),
]
_RES = [(g, re.compile(p)) for g, p in GROUPS]


def classify(name):
    for g, r in _RES:
        if r.search(name):
            return g
    return "other"


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r["Grid_Size_X"])))
    rows.sort()
    return rows


def region(rows, ga, gb):
    ia = [i for i, r in enumerate(rows) if r[3] == ga and "elementwise" in r[2]]
    ib = [i for i, r in enumerate(rows) if r[3] == gb and "elementwise" in r[2]]
    if len(ia) != 1 or len(ib) != 1 or ia[0] >= ib[0]:
        raise SystemExit(f"FAIL markers: grid {ga} found {len(ia)}x, grid {gb} found {len(ib)}x (need exactly one each, A before B)")
    return rows[ia[0] + 1:ib[0]]


def union_gaps(reg):
    """Merge the kernels' intervals (several queues or copy engines can overlap): busy = time with at least one kernel running, gaps = the idle
    stretches between the merged intervals."""
    merged = []
    for st, en, _, _ in reg:
        if merged and st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    busy = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return busy, gaps


def report(rows, reps, ga=MARK_A, gb=MARK_B):
    reg = region(rows, ga, gb)
    if not reg:
        raise SystemExit("FAIL no kernels between the markers")
    wall = max(r[1] for r in reg) - reg[0][0]
    ktime = sum(e - s for s, e, _, _ in reg)
    busy, gaps = union_gaps(reg)
    overlap = sum(1 for i in range(len(reg) - 1) if reg[i + 1][0] < reg[i][1])
    groups = {}
    names = {}
    for s, e, n, _ in reg:
        g = classify(n)
        groups.setdefault(g, [0, 0])
        groups[g][0] += e - s
        groups[g][1] += 1
        names.setdefault(n, [0, 0, g])
        names[n][0] += e - s
        names[n][1] += 1
    out = {"reps": reps, "kernels_total": len(reg), "kernels_per_pass": len(reg) / reps, "wall_ms_per_pass": wall / reps / 1e6,
           "kernel_ms_per_pass": ktime / reps / 1e6, "busy_ms_per_pass": busy / reps / 1e6, "gap_ms_per_pass": sum(gaps) / reps / 1e6,
           "gap_share": sum(gaps) / wall, "busy_share": busy / wall, "kernel_sum_over_wall": ktime / wall,
           "median_gap_us": statistics.median(gaps) / 1e3 if gaps else 0.0,
           "p95_gap_us": sorted(gaps)[int(0.95 * (len(gaps) - 1))] / 1e3 if gaps else 0.0, "overlapping_pairs": overlap,
           "groups": {g: {"ms_per_pass": v[0] / reps / 1e6, "share_of_kernel_time": v[0] / ktime, "kernels_per_pass": v[1] / reps} for g, v in sorted(groups.items(), key=lambda kv: -kv[1][0])},
           "top_kernels": [{"name": n, "group": v[2], "ms_per_pass": v[0] / reps / 1e6, "calls_per_pass": v[1] / reps}
                           for n, v in sorted(names.items(), key=lambda kv: -kv[1][0])[:14]]}
    return out


def text(out):
    L = [f"passes {out['reps']}: {out['kernels_per_pass']:.0f} kernels per pass, wall {out['wall_ms_per_pass']:.3f} ms per pass, sum of kernel times {out['kernel_ms_per_pass']:.3f} ms "
         f"({100 * out['kernel_sum_over_wall']:.1f} % of wall), busy (union) {100 * out['busy_share']:.1f} %, gaps {out['gap_ms_per_pass']:.3f} ms ({100 * out['gap_share']:.1f} %), median gap {out['median_gap_us']:.2f} us, p95 {out['p95_gap_us']:.2f} us, overlapping pairs {out['overlapping_pairs']}"]
    for g, v in out["groups"].items():
        L.append(f"  {g:18s} {v['ms_per_pass']:9.3f} ms  {100 * v['share_of_kernel_time']:5.1f} %  {v['kernels_per_pass']:7.1f} kernels")
    L.append("  top kernels:")
    for k in out["top_kernels"]:
        L.append(f"    {k['ms_per_pass']:8.3f} ms x{k['calls_per_pass']:.0f} [{k['group']}] {k['name'][:150]}")
    return "\n".join(L)


def selftest():
    def mk(fn, kernels):
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp", "Grid_Size_X"])
            for n, s, e, g in kernels:
                w.writerow(["KERNEL_DISPATCH", n, s, e, g])
    import tempfile, os
    d = tempfile.mkdtemp()
    ew = "void at::native::vectorized_elementwise_kernel<4, x>"
    ks = [("warm", 0, 10, 1), (ew, 20, 30, MARK_A),
          ("Cijk_Alik_Bljk_F8BS_BH", 100, 200, 16384), ("Cijk_Alik_Bljk_BBS_BH", 210, 250, 4096), (ew, 260, 270, 999),
          ("Cijk_Alik_Bljk_F8BS_BH", 300, 400, 16384), ("Cijk_Alik_Bljk_BBS_BH", 410, 450, 4096), (ew, 460, 470, 999),
          (ew, 600, 610, MARK_B)]
    fn = os.path.join(d, "a.csv")
    mk(fn, [(n, s, e, g) for n, s, e, g in ks])
    out = report(read_rows(fn), 2)
    g = out["groups"]
    assert abs(g["gemm_fp8"]["ms_per_pass"] - 100 / 1e6) < 1e-12 and abs(g["gemm_16bit"]["ms_per_pass"] - 40 / 1e6) < 1e-12, g
    assert out["kernels_total"] == 6 and out["kernels_per_pass"] == 3
    # gaps: 10 (200->210), 10 (250->260), 30 (270->300), 10, 10 ; wall 100..470
    assert abs(out["gap_ms_per_pass"] - (10 + 10 + 30 + 10 + 10) / 2 / 1e6) < 1e-12, out["gap_ms_per_pass"]
    assert out["overlapping_pairs"] == 0 and abs(out["busy_share"] + out["gap_share"] - 1.0) < 1e-9
    # overlapping kernels on two queues: union busy time, not the sum
    ov = [(ew, 20, 30, MARK_A), ("k1", 100, 200, 1), ("k2", 150, 250, 1), (ew, 300, 310, MARK_B)]
    mk(fn, ov)
    o2 = report(read_rows(fn), 1)
    assert abs(o2["busy_ms_per_pass"] - 150 / 1e6) < 1e-12 and abs(o2["kernel_ms_per_pass"] - 200 / 1e6) < 1e-12 and o2["gap_ms_per_pass"] == 0.0, o2
    # breaks: a missing marker, a duplicated marker and a reversed order must FAIL
    for label, kk, ma, mb in (("missing B", ks[:-1], MARK_A, MARK_B), ("two A", ks + [(ew, 700, 710, MARK_A)], MARK_A, MARK_B),
                              ("B before A", ks, MARK_B, MARK_A)):
        mk(fn, kk)
        try:
            report(read_rows(fn), 2, ma, mb)
        except SystemExit as e:
            assert str(e).startswith("FAIL"), e
        else:
            raise AssertionError("break not detected: " + label)
    assert classify("Cijk_Alik_Bljk_F8BS_BH_Bias") == "gemm_fp8" and classify("Cijk_Alik_Bljk_BBS_BH") == "gemm_16bit"
    assert classify("void at::native::vectorized_elementwise_kernel<4>") == "elementwise_other"
    # vLLM's FP8 decode kernel carries Float8 in its template arguments, not "fp8"; its bf16 twin and the LM head are 16-bit
    assert classify("void wvSplitKQ_hf_sml_<__hip_bfloat16, c10::Float8_e4m3fn, 32, 2, 16, 16, 2, 1>(int, int, c10::Float8_e4m3fn const*)") == "gemm_fp8"
    assert classify("void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2, 16, 16, 2, 1>(int, int, __hip_bfloat16 const*)") == "gemm_16bit"
    assert classify("void at::native::ConvertToFloat8Kernel<float>") == "fp8_quantize"
    print("VERDICT llm4_trace selftest PASS")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "report":
        import argparse, glob
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("csv"); ap.add_argument("--reps", type=int, required=True)
        ap.add_argument("--mark-a", type=int, default=MARK_A); ap.add_argument("--mark-b", type=int, default=MARK_B); ap.add_argument("--json")
        a = ap.parse_args()
        o = report(read_rows(a.csv), a.reps, a.mark_a, a.mark_b)
        print(text(o))
        if a.json:
            json.dump(o, open(a.json, "w"), indent=1)
        print("VERDICT PASS markers found, region analysed")
