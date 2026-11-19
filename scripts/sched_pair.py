#!/usr/bin/env python3
"""Lane 7 item 7b: merges the two files of one two-process job (sched_share pair) on REALTIME.

Usage: sched_pair.py <A.csv> <B.csv>
       sched_pair.py --preempt <file.csv>... [--calib-us X]   (leftovers C, SC-11: sched_share preempt files)
                     [--expect wait|overlap|any|gap] [--shift-b-ms X]
                                                              (gaming round P3: wait = SC-11's rule; overlap = the
                                                              positive control, B must start beside A in >= 90 % of
                                                              the trials; any = exploratory, no verdict on the outcome;
                                                              --shift-b-ms moves B's start and end later in the analysis
                                                              only, a deliberate break of the overlap detection)
       sched_pair.py --shift <A.csv> <B.csv> [--break]        (leftovers C, SC-4: shifted-trace control of a small+small
                                                              pair: B's events moved by +-5 to +-100 ms; every shifted
                                                              coincidence must be at most 3 %, the unshifted one within
                                                              1 point of item 7's 98.8 %; --break judges the unshifted
                                                              data by the chance limit)
big + big:   per-process launch intervals (first group start to last group end); share of the busy time in which
             both processes had a launch on the card; runs of consecutive launches of one process (time slices) and
             the gap at each switch.
small + big: the spin wave's cycles per step and its disturbed steps inside and outside the other process's launches.
small + small: each wave's disturbed fraction.
Prints RESULT lines; exits 1 if a file is missing or empty.
"""
import csv
import sys


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def launches(rows):
    by = {}
    for r in rows:
        k = int(r["launch"])
        a, b = int(r["r0"]), int(r["r1"])
        lo, hi = by.get(k, (a, b))
        by[k] = (min(lo, a), max(hi, b))
    return [by[k] for k in sorted(by)]


def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, round(p * (len(v) - 1)))]


def dist(v, unit):
    if not v:
        return "n 0"
    return (f"median {q(v, .5):.3f} p10 {q(v, .1):.3f} p90 {q(v, .9):.3f} min {min(v):.3f} max {max(v):.3f} "
            f"n {len(v)} {unit}")


def overlap_len(iv_a, iv_b):
    """Total length of the intersection of two sorted interval lists (ticks)."""
    i = j = 0
    tot = 0
    while i < len(iv_a) and j < len(iv_b):
        lo = max(iv_a[i][0], iv_b[j][0])
        hi = min(iv_a[i][1], iv_b[j][1])
        if hi > lo:
            tot += hi - lo
        if iv_a[i][1] < iv_b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def big_big(ra, rb):
    la, lb = launches(ra), launches(rb)
    ua, ub = union(la), union(lb)
    both = overlap_len(ua, ub)
    anyt = sum(b - a for a, b in union(la + lb))
    print(f"RESULT pair big+big launches A={len(la)} B={len(lb)} span_ms={anyt / 1e5:.1f} "
          f"both_on_card_fraction={both / max(anyt, 1):.3f}")
    print(f"RESULT pair big+big launch_ms A {dist([(b - a) / 1e5 for a, b in la], 'ms')}")
    print(f"RESULT pair big+big launch_ms B {dist([(b - a) / 1e5 for a, b in lb], 'ms')}")
    # Runs: launches ordered by start, consecutive launches of one process form a run (a time slice).
    ev = sorted([(a, b, "A") for a, b in la] + [(a, b, "B") for a, b in lb])
    runs, gaps = [], []
    cur, s0, e0 = None, 0, 0
    for a, b, who in ev:
        if who != cur:
            if cur is not None:
                runs.append((cur, (e0 - s0) / 1e5))
                gaps.append((a - e0) / 1e5)
            cur, s0, e0 = who, a, b
        else:
            e0 = max(e0, b)
    if cur is not None:
        runs.append((cur, (e0 - s0) / 1e5))
    print(f"RESULT pair big+big runs={len(runs)} run_ms {dist([r for _, r in runs], 'ms')}")
    print(f"RESULT pair big+big switch_gap_ms {dist(gaps, 'ms')} (negative = the next process started before the "
          f"previous run ended)")
    print("RUNS " + " ".join(f"{w}:{r:.2f}" for w, r in runs[:40]))
    # Per CU: were groups of both processes resident on one CU at the same time?
    by_cu = {}
    for rows, who in ((ra, "A"), (rb, "B")):
        for r in rows:
            by_cu.setdefault(int(r["cu"]), {}).setdefault(who, []).append((int(r["r0"]), int(r["r1"])))
    same_cu = sum(overlap_len(union(v.get("A", [])), union(v.get("B", []))) for v in by_cu.values())
    tot_cu = sum(sum(b - a for a, b in union(v.get("A", []) + v.get("B", []))) for v in by_cu.values())
    print(f"RESULT pair big+big per_CU_both_resident_fraction={same_cu / max(tot_cu, 1):.3f}")


def unwrap(vals):
    out, base, prev = [], 0, None
    for v in vals:
        if prev is not None and v < prev - (1 << 31):
            base += 1 << 32
        out.append(v + base)
        prev = v
    return out


def small_big(rs, rb):
    lb = union(launches(rb))
    # Map the big process's REALTIME to low 32 bits relative to the small trace's first sample.
    rt = unwrap([int(r["rt_lo"]) for r in rs])
    first_full = None
    lo0 = int(rs[0]["rt_lo"])
    hi_guess = lb[0][0] if lb else 0
    # The high bits of the small trace equal those of the big launches near it (both within seconds).
    first_full = (hi_guess & ~0xFFFFFFFF) | lo0
    if first_full - hi_guess > (1 << 31):
        first_full -= 1 << 32
    elif hi_guess - first_full > (1 << 31):
        first_full += 1 << 32
    t = [first_full + (x - rt[0]) for x in rt]
    inside = []
    for x in t:
        k = 0
        while k < len(lb) and lb[k][1] < x:
            k += 1
        inside.append(k < len(lb) and lb[k][0] <= x <= lb[k][1])
    steps = [int(r["step"]) for r in rs]
    cyc = [int(r["cycles"]) for r in rs]
    dis = [r["disturbed"] == "1" for r in rs]
    # Cycles per step between consecutive regular samples (every 1000th step), inside and outside B's launches.
    per_in, per_out, n_dis_in, n_dis_out = [], [], 0, 0
    for i in range(1, len(rs)):
        ds = steps[i] - steps[i - 1]
        if ds == 1000 and not dis[i] and not dis[i - 1]:
            (per_in if inside[i] and inside[i - 1] else per_out).append((cyc[i] - cyc[i - 1]) / ds)
    for i in range(len(rs)):
        if dis[i]:
            if inside[i]:
                n_dis_in += 1
            else:
                n_dis_out += 1
    tin = sum(1 for x in inside if x)
    print(f"RESULT pair small+big samples_inside_B_launches={tin} of {len(rs)}")
    print(f"RESULT pair small+big cycles_per_step inside_B {dist(per_in, 'cycles')}")
    print(f"RESULT pair small+big cycles_per_step outside_B {dist(per_out, 'cycles')}")
    print(f"RESULT pair small+big disturbed_steps inside_B={n_dis_in} outside_B={n_dis_out}")


def events(rows):
    """Start REALTIME (low 32 bits, unwrapped) of each run of disturbed rows (rows within 20 ticks merge)."""
    out, prev = [], None
    rt = unwrap([int(r["rt_lo"]) for r in rows])
    for r, t in zip(rows, rt):
        if r["disturbed"] == "1":
            if prev is None or t - prev > 20:
                out.append(t)
            prev = t
    return out


def small_small(ra, rb):
    for rows, who in ((ra, "A"), (rb, "B")):
        d = sum(1 for r in rows if r["disturbed"] == "1")
        print(f"RESULT pair small+small {who} disturbed_rows={d} rows={len(rows)}")
    # Do the two waves (two processes, different SIMDs) lose time at the same moments?
    a, b = events(ra), events(rb)
    if not a or not b:
        print("RESULT pair small+small coincidence: no events")
        return
    import bisect
    near = []
    for t in a:
        i = bisect.bisect_left(b, t)
        near.append(min(abs(t - b[j]) for j in (i - 1, i) if 0 <= j < len(b)))
    within = sum(1 for x in near if x <= 100)
    print(f"RESULT pair small+small events A={len(a)} B={len(b)} A_events_within_1us_of_a_B_event={within} "
          f"({within / len(a):.3f}) nearest_ticks {dist(near, 'ticks of 10 ns')}")
    on_simd = {r["hw"] for r in ra} | {r["hw"] for r in rb}
    print(f"RESULT pair small+small distinct HW_ID1 values over both waves: {len(on_simd)} "
          f"({', '.join(sorted(hex(int(x)) for x in on_simd))})")


def shift_control(ra, rb, brk):
    """SC-4: coincidence of A's events with B's events moved by a fixed time (no common cause can survive a shift)."""
    import bisect
    a, b = events(ra), events(rb)
    if not a or not b:
        print("VERDICT shift_control: FAIL no events")
        return 1

    def coinc(shift_ticks):
        bs = [t + shift_ticks for t in b]
        hit = 0
        for t in a:
            i = bisect.bisect_left(bs, t - 100)
            hit += i < len(bs) and bs[i] <= t + 100
        return hit / len(a)

    base = coinc(0)
    print(f"RESULT shift events A={len(a)} B={len(b)} unshifted coincidence within 1 us {base:.4f}")
    worst = 0.0
    for ms in (5, 10, 20, 50, 100):
        for sgn in (1, -1):
            c = coinc(sgn * ms * 100000)
            worst = max(worst, c)
            print(f"RESULT shift B moved {sgn * ms:+d} ms: coincidence {c:.4f}")
    # chance estimate for comparison: A's events against B's event density, a window of 2 us
    span = (max(a[-1], b[-1]) - min(a[0], b[0])) or 1
    chance = len(b) * 200 / span
    print(f"INFO shift chance of a coincidence for independent events: {chance:.4f} (2 us window x B's event rate)")
    limit = 0.03
    if brk:
        ok = base <= limit
        text = f"break: the unshifted data judged by the chance limit: {base:.4f} <= {limit}"
    else:
        ok = worst <= limit and abs(base - 0.988) <= 0.01
        text = (f"unshifted {base:.4f} (item 7: 0.988 +-0.01); largest shifted coincidence {worst:.4f} (limit {limit}) over "
                f"10 shifts of 5 to 100 ms")
    print(f"VERDICT shift_control: {'PASS' if ok else 'FAIL'} {text}")
    return 0 if ok else 1


def preempt(paths, calib_us, expect="wait", shift_b_ms=0.0):
    """SC-11. Rows: one per trial and kernel (A: the long holder, B: the 1 ms kernel launched 1 ms after A)."""
    rows = []
    for p in paths:
        rows += load(p)
    if shift_b_ms:
        for r in rows:
            if r["kernel"] == "B" and r["control"] != "1":
                for k in ("first_start_rt", "last_start_rt", "last_end_rt"):
                    r[k] = str(int(r[k]) + int(shift_b_ms * 100000))
    by = {}
    for r in rows:
        by.setdefault(int(r["trial"]), {})[r["kernel"]] = r
    ctrl_delay, bad_calib = [], 0
    per_ta = {}
    contradictions, invalid = 0, 0
    for k in sorted(by):
        t = by[k]
        if "B" not in t:
            continue
        b = t["B"]
        b0, b1, bl = int(b["first_start_rt"]), int(b["last_end_rt"]), int(b["launch_rt"])
        if b["control"] == "1":
            ctrl_delay.append((b0 - bl) / 100.0)
            continue
        if "A" not in t:
            invalid += 1
            continue
        a = t["A"]
        a0, a1, gap = int(a["first_start_rt"]), int(a["last_end_rt"]), int(a["max_gap_ticks"])
        ta = int(a["ta_ms"])
        d = per_ta.setdefault(ta, {"waited": 0, "preempted": 0, "contra": 0, "start_minus_end_us": [],
                                   "b_delay_us": [], "a_gap_us": [], "invalid": 0})
        if a0 >= bl:  # A had not started when B was launched: not a test of preemption
            d["invalid"] += 1
            invalid += 1
            continue
        d["start_minus_end_us"].append((b0 - a1) / 100.0)
        d.setdefault("last_start_minus_end_us", []).append((int(b["last_start_rt"]) - a1) / 100.0)
        d["b_delay_us"].append((b0 - bl) / 100.0)
        d["a_gap_us"].append(gap / 100.0)
        if b0 >= a1 - 1000:
            d["waited"] += 1
        elif gap >= 50000:
            d["preempted"] += 1
        else:
            d["contra"] += 1
            contradictions += 1
    for ta in sorted(per_ta):
        d = per_ta[ta]
        print(f"RESULT preempt T_A={ta}ms trials={d['waited'] + d['preempted'] + d['contra']} waited_for_A={d['waited']} "
              f"started_while_A_saved={d['preempted']} started_while_A_ran_without_gap={d['contra']} "
              f"A_not_started_yet={d['invalid']}")
        print(f"RESULT preempt T_A={ta}ms B_start_minus_A_end_us {dist(d['start_minus_end_us'], 'us')}")
        print(f"RESULT preempt T_A={ta}ms B_launch_to_start_us {dist(d['b_delay_us'], 'us')}")
        print(f"RESULT preempt T_A={ta}ms A_longest_wave_gap_us {dist(d['a_gap_us'], 'us')}")
        print(f"RESULT preempt T_A={ta}ms B_last_group_start_minus_A_end_us "
              f"{dist(d.get('last_start_minus_end_us', []), 'us')}")
    print(f"RESULT preempt control (B alone) launch_to_start_us {dist(ctrl_delay, 'us')}")
    # Limits from scheduling item 1 (launch + synchronize after 1 to 1000 ms idle: medians 55 to 71 us, largest
    # 371.9 us), plus 10 us for the host-to-REALTIME mapping. calib_us = (median limit, largest limit).
    med_lim, max_lim = calib_us
    ok_c = (len(ctrl_delay) >= 4 and min(ctrl_delay) >= -10 and q(ctrl_delay, .5) <= med_lim
            and max(ctrl_delay) <= max_lim)
    print(f"VERDICT preempt_calib: {'PASS' if ok_c else 'FAIL'} B alone, launch to first wave: median "
          f"{q(ctrl_delay, .5):.1f} us (limit {med_lim:g}), largest {max(ctrl_delay or [0]):.1f} us (limit {max_lim:g}), "
          f"{len(ctrl_delay)} control trials (scheduling item 1 after idle: medians 55 to 71 us, tails to 372 us; "
          f"10 us allowed for the host-to-REALTIME mapping)")
    n = sum(d["waited"] + d["preempted"] + d["contra"] for d in per_ta.values())
    beside = sum(d["contra"] for d in per_ta.values())
    print(f"RESULT preempt summary trials={n} B_waited_for_A_end={sum(d['waited'] for d in per_ta.values())} "
          f"B_started_beside_running_A_no_gap={beside} B_started_while_A_saved={sum(d['preempted'] for d in per_ta.values())} "
          f"A_not_started_yet={invalid}")
    if expect == "gap":
        # E8: the gap detector's known answer. A's wave 0 of group 0 sleeps K steps without reading REALTIME once
        # past half of T_A; every A trial must then show a longest wave gap of 0.5 ms or more.
        gaps = [g for d in per_ta.values() for g in d["a_gap_us"]]
        fired = sum(1 for g in gaps if g >= 500)
        ok_g = len(gaps) > 0 and fired == len(gaps)
        print(f"VERDICT preempt_gap_detector: {'PASS' if ok_g else 'FAIL'} A's longest wave gap at least 0.5 ms in {fired} "
              f"of {len(gaps)} trials (an injected sleep without REALTIME reads; expected every trial); gaps "
              f"{dist(gaps, 'us')}")
        return 0 if ok_c and ok_g else 1
    if expect != "wait":
        if expect == "overlap":
            ok_d = n > 0 and beside >= 0.9 * n and invalid <= n // 5
            print(f"VERDICT preempt_detect: {'PASS' if ok_d else 'FAIL'} positive control{' (DELIBERATE BREAK: B moved '
                  f'{shift_b_ms:g} ms later in the analysis)' if shift_b_ms else ''}: B started beside A, A's waves "
                  f"without a gap, in {beside} of {n} trials (expected at least 90 %)")
        else:
            ok_d = True
            print(f"INFO preempt exploratory (no verdict on the outcome): B started beside A in {beside} of {n} trials")
        return 0 if ok_c and ok_d else 1
    ok_s = contradictions == 0 and n > 0 and invalid <= n // 5
    print(f"VERDICT preempt_consistent: {'PASS' if ok_s else 'FAIL'} {n} overlap trials: every B either started after A "
          f"ended or while A's waves show a gap of 0.5 ms or more (contradictions {contradictions}, trials where A had not "
          f"started at B's launch {invalid})")
    waited = sum(d["waited"] for d in per_ta.values())
    print(f"INFO preempt expectation (READ priority disabled; MEASURED whole launches alternate): B waits for A's end "
          f"in {waited} of {n} trials -> {'as expected' if waited == n else 'NOT as expected: preemption seen'}")
    return 0 if ok_c and ok_s else 1


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--shift":
        try:
            return shift_control(load(sys.argv[2]), load(sys.argv[3]), "--break" in sys.argv)
        except OSError as e:
            print(f"VERDICT shift_control: FAIL {e}")
            return 1
    if len(sys.argv) >= 2 and sys.argv[1] == "--preempt":
        args = sys.argv[2:]
        calib = (81.0, 382.0)
        if "--calib-us" in args:
            i = args.index("--calib-us")
            calib = (float(args[i + 1]), float(args[i + 1]))
            del args[i:i + 2]
        expect, shift = "wait", 0.0
        if "--expect" in args:
            i = args.index("--expect")
            expect = args[i + 1]
            del args[i:i + 2]
        if "--shift-b-ms" in args:
            i = args.index("--shift-b-ms")
            shift = float(args[i + 1])
            del args[i:i + 2]
        try:
            return preempt(args, calib, expect, shift)
        except (OSError, KeyError, ValueError) as e:
            print(f"VERDICT preempt_calib: FAIL {e}")
            return 1
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        ra, rb = load(sys.argv[1]), load(sys.argv[2])
    except OSError as e:
        print(f"VERDICT sched_pair: FAIL {e}")
        return 1
    if not ra or not rb:
        print("VERDICT sched_pair: FAIL empty file")
        return 1
    ka, kb = ra[0]["kind"], rb[0]["kind"]
    if ka == "big" and kb == "big":
        big_big(ra, rb)
    elif ka == "small" and kb == "big":
        small_big(ra, rb)
    elif ka == "small" and kb == "small":
        small_small(ra, rb)
    else:
        print(f"VERDICT sched_pair: FAIL unsupported pair {ka}+{kb}")
        return 1
    print("VERDICT sched_pair: PASS merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
