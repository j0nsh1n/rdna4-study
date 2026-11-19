#!/usr/bin/env python3
"""Round 3, Q5: join the in-kernel REALTIME stamps of src/pair_overlap.hip with a rocprofv3 kernel trace and classify every neighbouring pair.

  llm3_pairs.py join <dump.csv> <kernel_trace.csv> [--label TEXT]
  llm3_pairs.py selftest

Per pair (previous kernel, next kernel) two gaps, each on its own clock, so no alignment is needed for the verdict:
  profiler gap  = Start_Timestamp(next) - End_Timestamp(prev)                        (ns)
  in-kernel gap = 10 ns * (first REALTIME read of the earliest wave of next - last REALTIME read of the latest wave of prev)
A gap below 0 is an overlap on that clock; an in-kernel gap of -1, 0 or +1 tick is AMBIGUOUS (resolution 10 ns).
Classes: both (real overlap: both clocks), profiler-only (an artifact of the profiler's interval stamps, the waves did not overlap),
in-kernel-only, ambiguous, neither. The kernels of the dump are matched to the LAST rows of the trace with the same names (warm-ups come first);
the kind sequence must agree. The two clocks are also put on one axis: an offset c (t_ns = 10 * tick + c) is feasible when every kernel's
first stamp is not before its dispatch start and its last stamp not after its dispatch end; the feasible interval and the drift over the run are printed.
Exit 1 on a matching failure (counts or kinds do not agree, no feasible offset by more than a profiler timestamp's own jitter is reported, not failed)."""
import csv
import sys

NAMES = {"A": "pair_a", "B": "pair_b", "P": "spin_k", "Q": "spin_k"}


def load_dump(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["kind"], int(r["first_min"]), int(r["first_max"]), int(r["last_min"]), int(r["last_max"])))
    return rows


def load_trace(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r["Dispatch_Id"])))
    rows.sort()
    return rows


def match(dump, trace):
    """Pair each dump kernel with a trace row: the last len(dump) rows of the wanted names, in start order."""
    want = {NAMES[k] for k, *_ in dump}
    cand = [t for t in trace if any(n in t[2] for n in want)]
    if len(cand) < len(dump):
        return None, f"the trace holds {len(cand)} rows of {sorted(want)}, the dump {len(dump)} kernels"
    cand = cand[-len(dump):]
    for i, ((k, *_), t) in enumerate(zip(dump, cand)):
        if NAMES[k] not in t[2]:
            return None, f"kernel {i}: the dump says {k} ({NAMES[k]}), the trace row is {t[2][:40]}"
    return cand, ""


def classify(dump, rows, pair_filter):
    """pair_filter(i) -> label or None for the pair (i, i+1). Returns dict label -> list of (prof_gap_ns, inkernel_gap_ns)."""
    out = {}
    for i in range(len(dump) - 1):
        lab = pair_filter(i)
        if lab is None:
            continue
        pg = rows[i + 1][0] - rows[i][1]
        ig = 10 * (dump[i + 1][1] - dump[i][4])
        out.setdefault(lab, []).append((pg, ig))
    return out


def summarize(label, gaps):
    n = len(gaps)
    prof = [g for g in gaps if g[0] < 0]
    both = [g for g in prof if g[1] < -10]
    amb = [g for g in prof if -10 <= g[1] <= 10]
    art = [g for g in prof if g[1] > 10]
    ik_only = [g for g in gaps if g[0] >= 0 and g[1] < -10]
    ik_amb = [g for g in gaps if g[0] >= 0 and -10 <= g[1] <= 10]
    ik_all = [g for g in gaps if g[1] < -10]
    sizes = sorted(-g[0] for g in prof)
    print(f"RESULT {label} pairs {n} profiler_overlap {len(prof)} inkernel_overlap {len(ik_all)} both {len(both)} profiler_only {len(art)} "
          f"profiler_overlap_inkernel_ambiguous {len(amb)} inkernel_only {len(ik_only)} inkernel_ambiguous_without_profiler_overlap {len(ik_amb)}")
    if sizes:
        print(f"RESULT {label} profiler_overlap_size_ns min {sizes[0]} median {sizes[len(sizes) // 2]} max {sizes[-1]}")
    return {"n": n, "prof": len(prof), "ik": len(ik_all), "both": len(both), "art": len(art), "amb": len(amb), "ik_only": len(ik_only)}


def offset_interval(dump, rows):
    lo = max(r[0] - 10 * d[1] for d, r in zip(dump, rows))
    hi = min(r[1] - 10 * d[4] for d, r in zip(dump, rows))
    span_ik = 10 * (dump[-1][4] - dump[0][1])
    span_pr = rows[-1][1] - rows[0][0]
    return lo, hi, span_ik, span_pr


def cmd_join(dump_path, trace_path, label):
    dump, trace = load_dump(dump_path), load_trace(trace_path)
    rows, why = match(dump, trace)
    if rows is None:
        print(f"FAIL {label}: {why}")
        return 1
    lo, hi, sik, spr = offset_interval(dump, rows)
    print(f"RESULT {label} clock_alignment feasible_offset_interval_ns {lo} to {hi} width {hi - lo} {'FEASIBLE' if lo <= hi else 'NO-OFFSET-FITS'}; "
          f"run span in-kernel {sik} ns, profiler {spr} ns, ratio {sik / spr:.6f}")
    # with the rate difference taken out (profiler ns per in-kernel ns = spr / sik over the whole run) a single offset may fit
    a = spr / sik
    lo2 = max(r[0] - a * 10 * d[1] for d, r in zip(dump, rows))
    hi2 = min(r[1] - a * 10 * d[4] for d, r in zip(dump, rows))
    print(f"RESULT {label} clock_alignment_rate_corrected scale {a:.7f} feasible_offset_interval_ns {lo2:.0f} to {hi2:.0f} width {hi2 - lo2:.0f} {'FEASIBLE' if lo2 <= hi2 else 'NO-OFFSET-FITS'}")
    kinds = "".join(k for k, *_ in dump)
    if set(kinds) == {"P", "Q"}:
        res = classify(dump, rows, lambda i: "P_to_Q" if kinds[i] == "P" else None)
    else:
        res = classify(dump, rows, lambda i: "dependent_A_to_B" if kinds[i] == "A" else "neighbour_B_to_nextA")
    for lab, gaps in res.items():
        summarize(f"{label} {lab}", gaps)
        marg = sorted(ig - pg for pg, ig in gaps)
        print(f"RESULT {label} {lab} margin_ns (in-kernel gap minus profiler gap) median {marg[len(marg) // 2]} min {marg[0]} max {marg[-1]}")
        for g in [g for g in gaps if g[0] < 0][:20]:
            print(f"DETAIL {label} {lab} profiler_gap_ns {g[0]} inkernel_gap_ns {g[1]}")
    return 0


def canned(artifact=True, real=False, kinds_bad=False, n=40):
    """dump (kind, first_min, first_max, last_min, last_max in ticks) and trace rows with every pair 1 us apart on both clocks,
    plus one planted pair at index 10: profiler overlap of 50 ns with an in-kernel gap of +200 ns (artifact) or -300 ns (real)."""
    dump, rows = [], []
    t = 1000
    for i in range(2 * n):
        k = "AB"[i % 2]
        dur = 300
        first, last = t, t + dur
        dump.append((k, first, first, last, last))
        pad = 15  # profiler interval is wider than the waves' span
        rows.append((10 * first - pad, 10 * last + pad, NAMES[k] if not (kinds_bad and i == 5) else "other_kernel", i))
        t = last + 100
    if artifact or real:
        i = 10  # between kernel 10 and 11
        gap_ticks = 20 if artifact else -30
        shift = (dump[10][4] + gap_ticks) - dump[11][1]
        for j in range(11, 2 * n):
            k, a, b, c, d = dump[j]
            dump[j] = (k, a + shift, b + shift, c + shift, d + shift)
            r = rows[j]
            rows[j] = (r[0] + 10 * shift, r[1] + 10 * shift, r[2], r[3])
        if artifact:
            # widen the profiler's interval of kernel 10 so its end passes the start of kernel 11 (stamps jitter), the waves are untouched
            r = rows[10]
            rows[10] = (r[0], rows[11][0] + 50, r[2], r[3])
    return dump, rows


def selftest():
    import os
    import tempfile
    fails = 0

    def expect(name, got, want):
        nonlocal fails
        ok = got == want
        print(("PASS" if ok else "FAIL (selftest broken)") + f" {name}: got {got}, want {want}")
        fails += not ok

    def run(dump, rows):
        pf = lambda i: "dep" if i % 2 == 0 else "nei"
        res = classify(dump, rows, pf)
        return {k: summarize("selftest " + k, v) for k, v in res.items()}

    d, r = canned(artifact=False, real=False)
    s = run(d, r)
    expect("control: clean pairs, no overlap on either clock", (s["dep"]["prof"], s["dep"]["ik"], s["nei"]["prof"], s["nei"]["ik"]), (0, 0, 0, 0))
    d, r = canned(artifact=True)
    s = run(d, r)
    expect("planted artifact (profiler intervals overlap by 50 ns, waves 200 ns apart): profiler_only 1", (s["nei"]["prof"] + s["dep"]["prof"], s["nei"]["art"] + s["dep"]["art"], s["nei"]["both"] + s["dep"]["both"]), (1, 1, 0))
    d, r = canned(real=True, artifact=False)
    s = run(d, r)
    # the planted real overlap: waves overlap by 300 ns; the profiler intervals (padded 15 ns each side) overlap too
    expect("planted real overlap: seen on both clocks", s["dep"]["both"] + s["nei"]["both"], 1)
    expect("planted real overlap: in-kernel count 1", s["dep"]["ik"] + s["nei"]["ik"], 1)
    # a profiler-only overlap must not be counted as real by a checker that only looks at the profiler: the two columns differ
    expect("artifact: in-kernel overlap count is 0 although the profiler shows one", run(*canned(artifact=True))["dep"]["ik"] + run(*canned(artifact=True))["nei"]["ik"], 0)
    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", ".")) as td:
        def write(dump, rows, tag):
            dp, tp = os.path.join(td, tag + ".dump.csv"), os.path.join(td, tag + ".trace.csv")
            with open(dp, "w") as f:
                f.write("idx,kind,first_min,first_max,last_min,last_max\n")
                for i, (k, a, b, c, e) in enumerate(dump):
                    f.write(f"{i},{k},{a},{b},{c},{e}\n")
            with open(tp, "w") as f:
                f.write('"Start_Timestamp","End_Timestamp","Kernel_Name","Dispatch_Id"\n')
                for a, b, nme, di in rows:
                    f.write(f'{a},{b},"{nme}",{di}\n')
            return dp, tp

        d, r = canned(artifact=False, real=False)
        expect("join: clean data matches", cmd_join(*write(d, r, "ok"), "selftest"), 0)
        d, r = canned(kinds_bad=True)
        expect("break: a kernel of another name inside the matched rows is refused", cmd_join(*write(d, r, "kinds"), "selftest"), 1)
        d, r = canned(artifact=False)
        expect("break: a trace with fewer rows than the dump is refused", cmd_join(*write(d, r[:-6], "short"), "selftest"), 1)
    print("VERDICT llm3_pairs selftest " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


def main():
    a = sys.argv[1:]
    if a[:1] == ["join"] and len(a) >= 3:
        label = a[a.index("--label") + 1] if "--label" in a else "pairs"
        return cmd_join(a[1], a[2], label)
    if a[:1] == ["selftest"]:
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
