#!/usr/bin/env python3
"""Gaming round, E1-E3: clocks of the dense FMA load (Vulkan gdense.comp and the HIP twin dense_twin.hip) from the
raw files, with one set of definitions for both.

  gprobe_dense.py job <run folder> <job name> [--break-windows] [--expect-stop|--expect-no-stop]
                                                                   one job's arms: ramps, bursts, density, checks
  gprobe_dense.py summary <run folder>                             tables over every job of the run (rounds pooled)
  gprobe_dense.py listing <vk asm file> <hip asm file>             the two loops hold the same instructions
  gprobe_dense.py f2 <run folder>                                  F2: target ramps by predecessor, with the pauses

Definitions. Clock of a window = cycle-counter increase over REALTIME increase x 100 MHz, from one recording wave's
(REALTIME, cycles) records; windows are 0.25 ms in the first 10 ms of a dispatch and 1 ms after, measured from the
earliest group start of the dispatch. "final" = mean of the windows in the last 20 % of the dispatch; t95 = the
start of the first window from which every later window is at least 95 % of final (the study's sched_launch rule).
"step" = the start of the first window at or after 1 ms whose clock is more than 5 % above the median of the 1-5 ms
windows. Fill = FMA instructions per SIMD per cycle over every wave of a dispatch: the waves' chunk counts (128
v_fma_f32 each) over 128 SIMDs x the cycles from the dispatch's first group start to its last group end (cycles = the
dispatch's counter clock x that time).
--break-windows (deliberate break): each window's cycles are divided by twice its REALTIME span.
--expect-stop (F5): the sampler's lowered limit must have stopped the load (verdict sampler_stop); --expect-no-stop
(F5's control): the stop word must never be set and every group must run its full length (verdict no_stop).
"""
import csv
import glob
import os
import re
import statistics as st
import sys

RT_MHZ = 100.0


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def q(v, p):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, max(0, round(p * (len(v) - 1))))]


def dist(v, fmt="%.0f"):
    v = [x for x in v if x == x]
    if not v:
        return "n=0"
    return (f"n={len(v)} min={fmt % min(v)} p50={fmt % q(v, .5)} p95={fmt % q(v, .95)} max={fmt % max(v)}")


class Arm:
    def __init__(self, run, job, name, log_arm):
        self.run, self.job, self.name = run, job, name
        base = os.path.join(run, f"{job}-dense-{name}")
        self.info = log_arm
        self.groups = rows(base + "-groups.csv")
        self.rec = rows(base + "-rec.csv") if os.path.exists(base + "-rec.csv") else []
        self.bursts = rows(base + "-bursts.csv") if os.path.exists(base + "-bursts.csv") else []
        m = rows(base + "-map.csv")[0]
        self.hip = "host_off0_ns" in m
        self.load_rt_ns = (None, None)
        if self.hip:
            self.load_rt_ns = (int(m["launch_host_ns"]), int(m["sync_host_ns"]))
            self.off = int(m["host_off0_ns"])
            self.drift = int(m["host_off1_ns"]) - self.off
            sy = rows(base + "-sysfs.csv")
            self.sys = [(self.host_to_rt(int(r["t_host_ns"])), r) for r in sy]
        else:
            d0, m0, d1, m1 = int(m["d0"]), float(m["m0"]), int(m["d1"]), float(m["m1"])
            self.rate = (d1 - d0) / (m1 - m0) if m1 > m0 else 0.1
            self.d0, self.m0 = d0, m0
            sy = rows(os.path.join(run, f"{job}-sysfs.csv"))
            self.sys = [(self.mono_to_rt(float(r["t_mono_ns"])), r) for r in sy]

    def host_to_rt(self, ns):
        return (ns - self.off) / 10.0

    def mono_to_rt(self, ns):
        return self.d0 + (ns - self.m0) * self.rate

    def sys_between(self, a, b):
        return [r for t, r in self.sys if a <= t <= b]

    def sys_near(self, t):
        best = min(self.sys, key=lambda x: abs(x[0] - t), default=None)
        return best[1] if best else None


def windows(recs, t0, t_end, brk=False):
    """recs: [(rt, cyc)] of one wave in time order -> [(start_ms, end_ms, MHz)] relative to t0."""
    out = []
    if len(recs) < 2:
        return out
    ms = 0.0
    j = 0
    n = len(recs)
    while t0 + ms * 1e5 < t_end:
        step = 0.25 if ms < 10 else 1.0
        a, b = t0 + ms * 1e5, t0 + (ms + step) * 1e5
        while j < n and recs[j][0] < a:
            j += 1
        k = j
        while k < n and recs[k][0] < b:
            k += 1
        if k - 1 > j:
            dr = recs[k - 1][0] - recs[j][0]
            dc = recs[k - 1][1] - recs[j][1]
            if dr > 0:
                out.append((ms, ms + step, dc / ((2 * dr) if brk else dr) * RT_MHZ))
        ms += step
    return out


def ramp_metrics(win):
    """win: [(start_ms, end_ms, MHz)] of one dispatch (the median over recording waves)."""
    if not win:
        return {}
    dur = win[-1][1]
    fin_w = [c for s, e, c in win if s >= 0.8 * dur]
    final = st.mean(fin_w) if fin_w else float("nan")
    t95 = float("nan")
    for i in range(len(win)):
        if all(c >= 0.95 * final for _, _, c in win[i:]):
            t95 = win[i][0]
            break
    below = [s for s, e, c in win if c < 1750]
    plate = [c for s, e, c in win if 1 <= s < 10]
    p15 = [c for s, e, c in win if 1 <= s < 5]
    step = float("nan")
    if p15:
        ref = q(p15, .5)
        step = next((s for s, e, c in win if s >= 1 and c > 1.05 * ref), float("nan"))
    return {"first": win[0][2], "plateau_1_10ms": q(plate, .5) if plate else float("nan"), "final": final,
            "t95": t95, "last_below_1750": max(below) if below else float("nan"), "dur": dur, "step": step}


def median_windows(per_wave):
    """[[(s, e, MHz)]] -> [(s, e, median MHz, min MHz, max MHz, n)] over the waves that have each window."""
    by = {}
    for w in per_wave:
        for s, e, c in w:
            by.setdefault((s, e), []).append(c)
    return [(s, e, q(v, .5), min(v), max(v), len(v)) for (s, e), v in sorted(by.items())]


def read_log_arms(log):
    arms = {}
    if not os.path.exists(log):
        return arms
    for line in open(log, errors="replace"):
        m = re.match(r"ARM (\S+) (.*)", line)
        if m:
            kv = dict(re.findall(r"(\w+)=([^\s]+)", m.group(2)))
            arms[m.group(1)] = kv
    return arms


def steady_check(arm, win, t0, brk_label):
    """counter clock against pp_dpm_sclk over windows where the counter is steady; returns verdict pieces."""
    res = []
    if not win:
        return res
    plate = [c for s, e, c, *_ in win if 1 <= s < 5]
    if plate:
        pm = q(plate, .5)
        rise = next((s for s, e, c, *_ in win if s >= 1 and c > 1.03 * pm), win[-1][1])
        a, b = 2.0, rise - 2.0
        if b - a >= 8:
            res.append(("plateau", a, b))
    dur = win[-1][1]
    tail = [(s, e, c) for s, e, c, *_ in win if s >= dur - 15]
    if tail and dur >= 40:
        cs = [c for _, _, c in tail]
        if max(cs) <= 1.01 * min(cs):
            res.append(("final", dur - 15, dur))
    out = []
    for label, a, b in res:
        cnt = q([c for s, e, c, *_ in win if s >= a and e <= b], .5)
        sy = arm.sys_between(t0 + a * 1e5, t0 + b * 1e5)
        sc = q([float(r["sclk_MHz"]) for r in sy], .5)
        out.append((label, a, b, cnt, sc, len(sy)))
    return out


def arm_stop(text, name):
    """(stop word at the end, by whom) from the arm's INFO line (Vulkan and HIP spellings)."""
    m = re.search(rf"INFO {re.escape(name)} .*?stop word at the end:? (\d) \(([^)]*)\)", text)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def burst_fill(wave_rows, groups, clk_mhz):
    """FMA per SIMD per cycle of one dispatch: its waves' chunks over 128 SIMDs x its cycles."""
    st0 = [int(g["start_rt"]) for g in groups if int(g["end_rt"])]
    en = [int(g["end_rt"]) for g in groups if int(g["end_rt"])]
    if not st0 or not wave_rows or not clk_mhz == clk_mhz:
        return float("nan")
    cyc = clk_mhz * 1e6 * (max(en) - min(st0)) / 1e8
    return sum(wave_rows) / cyc if cyc > 0 else float("nan")


def analyse_job(run, job, brk=False, expect=None):
    log = os.path.join(run, f"{job}.log")
    log_arms = read_log_arms(log)
    names = sorted({re.sub(rf"^{re.escape(job)}-dense-(.*)-groups\.csv$", r"\1", os.path.basename(p))
                    for p in glob.glob(os.path.join(run, f"{job}-dense-*-groups.csv"))})
    fails = 0
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    if not names:
        # no data is a failure, never a pass (a job that did not start, or a break run on absent files)
        print(f"VERDICT dense arms: FAIL {job}: no dense-load files in {run}")
        return 1
    for name in names:
        info = log_arms.get(name, {})
        arm = Arm(run, job, name, info)
        burst_ms = float(info.get("burst_ms", info.get("ms", "nan")))
        cadence = float(info.get("cadence_ms", "0") or 0)
        min_fill = float(info.get("min_fill", "0") or 0)
        path = "hip" if arm.hip else ("vk-" + info.get("queue", "?"))
        nb = max(int(g["burst"]) for g in arm.groups) + 1
        if not arm.hip:
            m = re.search(rf"INFO {re.escape(name)} load_realtime_ns start=(\d+) end=(\d+)", text)
            if m:
                arm.load_rt_ns = (int(m.group(1)), int(m.group(2)))
        recs = {}
        for r in arm.rec:
            recs.setdefault((int(r["burst"]), int(r["recorder"])), []).append((int(r["rt"]), int(r["cyc"])))
        gb = {}
        for g in arm.groups:
            gb.setdefault(int(g["burst"]), []).append(g)
        wv = os.path.join(run, f"{job}-dense-{name}-waves.csv")
        wchunks = {}
        if os.path.exists(wv):
            for r in rows(wv):
                wchunks.setdefault(int(r["burst"]), []).append(int(r["chunks"]))
        # density: every group ran for the whole planned length
        min_run, max_run, spreads, missing = [], [], [], 0
        for b in range(nb):
            gs = gb.get(b, [])
            st0 = [int(g["start_rt"]) for g in gs if int(g["end_rt"])]
            missing += sum(1 for g in gs if not int(g["end_rt"]))
            if not st0:
                continue
            spreads.append((max(st0) - min(st0)) / 100.0)
            rr = [(int(g["end_rt"]) - int(g["start_rt"])) / 1e5 for g in gs if int(g["end_rt"])]
            min_run.append(min(rr))
            max_run.append(max(rr))
        stopw, by = arm_stop(text, name)
        by_sampler = stopw == 1 and ("tripwire" in by)
        trip = re.search(rf"INFO {re.escape(name)} stop word set by the host ([0-9.]+) ms", text)
        runs = [(int(g["end_rt"]) - int(g["start_rt"])) / 1e5 for g in arm.groups if int(g["end_rt"])]
        if arm.load_rt_ns[0]:
            print(f"LOADTIME {name} path={path} start_realtime_ns={arm.load_rt_ns[0]} end_realtime_ns={arm.load_rt_ns[1]}")
        if expect == "stop":
            # F5: the sampler thread itself must have set the stop word, and the load must have ended because of it
            m = re.search(r"THERMAL hotspot ([0-9.]+) C: stopping the load \(sampler's tripwire, limit ([0-9.]+) C\) at "
                          r"(mono|host)_ns=(\d+)", text)
            cross = float("nan")
            if m:
                ns = int(m.group(4))
                cross = arm.host_to_rt(ns) if m.group(3) == "host" else arm.mono_to_rt(ns)
            ends = [int(g["end_rt"]) for g in arm.groups if int(g["end_rt"])]
            after = [(e - cross) / 1e5 for e in ends] if cross == cross else []
            ok = (bool(m) and by_sampler and missing == 0 and runs and max(runs) < 0.9 * burst_ms and after
                  and min(after) >= -0.5 and max(after) <= 6.0)
            fails += not ok
            print(f"VERDICT dense sampler_stop: {'PASS' if ok else 'FAIL'} {name} ({path}): "
                  f"{'the sampler logged ' + m.group(1) + ' C against a limit of ' + m.group(2) + ' C' if m else 'no sampler stop line'}; "
                  f"stop word at the end {stopw} ({by or 'no INFO line'}); every group's run {dist(runs, '%.3f')} ms of a "
                  f"planned {burst_ms:g} ms (limit: all below {0.9 * burst_ms:.1f}); group ends after the crossing sample "
                  f"{dist(after, '%.3f')} ms (limit -0.5 to 6.0); groups without an end {missing}")
        elif expect == "nostop":
            ok = stopw == 0 and missing == 0 and runs and min(runs) >= 0.95 * burst_ms
            fails += not ok
            print(f"VERDICT dense no_stop: {'PASS' if ok else 'FAIL'} {name} ({path}): stop word at the end {stopw} "
                  f"({by or 'no INFO line'}); every group's run {dist(runs, '%.3f')} ms of a planned {burst_ms:g} ms "
                  f"(limit: all at least {0.95 * burst_ms:.1f}); groups without an end {missing}")
        elif trip:
            x = float(trip.group(1))
            ok = missing == 0 and runs and max(runs) <= x + 1.0 and min(runs) >= x - 3.0 and max(runs) < 0.9 * burst_ms
            fails += not ok
            print(f"VERDICT dense tripwire: {'PASS' if ok else 'FAIL'} {name} ({path}): the host set the stop word {x:.3f} ms "
                  f"after the submission of a {burst_ms:g} ms load; every wave 0 ended after {dist(runs, '%.3f')} ms of "
                  f"work (limit {x - 3:.1f} to {x + 1:.1f} ms); groups without an end {missing}")
        elif burst_ms == burst_ms and min_run:
            # Start spread limit 15 % of the burst + 20 us. It was 5 % + 20 us until development run dev3-000239
            # showed 2,048 waves (64 groups of 1,024) taking 75 to 235 us to launch; widened after seeing that.
            lim_us = 0.15 * burst_ms * 1000 + 20
            ok = missing == 0 and min(min_run) >= 0.95 * burst_ms and q(spreads, 1) <= lim_us
            fails += not ok
            print(f"VERDICT dense density: {'PASS' if ok else 'FAIL'} {name} ({path}): every group's run "
                  f"(end - start of its wave 0) at least 95 % of {burst_ms:g} ms in every dispatch: shortest "
                  f"{min(min_run):.3f} ms; group start spread {dist(spreads, '%.1f')} us (limit {lim_us:.0f}); "
                  f"groups without an end {missing}{'; the stop word was set (' + by + ')' if stopw else ''}")
        # per dispatch: window clocks over the recording waves
        if trip:
            continue
        if nb == 1:
            gs = gb.get(0, [])
            t0 = min(int(g["start_rt"]) for g in gs)
            t_end = max(int(g["end_rt"]) for g in gs)
            per = [windows(recs[k], t0, t_end, brk) for k in sorted(recs) if k[0] == 0]
            mw = median_windows(per)
            mt = ramp_metrics([(s, e, c) for s, e, c, *_ in mw])
            if not mt:
                print(f"RAMP {name} no windows")
                continue
            # density over every wave: FMA instructions per SIMD per cycle (128 SIMDs; one wave32 VOP3 FMA per cycle at most)
            util = float("nan")
            ch = wchunks.get(0, [])
            if ch:
                tot = sum(e - s for s, e, *_ in mw)
                clk = sum(c * (e - s) for s, e, c, *_ in mw) / tot if tot else float("nan")
                util = burst_fill(ch, gs, clk)
                print(f"WAVES {name} waves={len(ch)} chunks per wave {dist(ch)} total {sum(ch)}; mean counter clock "
                      f"{clk:.0f} MHz over {(t_end - t0) / 1e5:.2f} ms -> {util:.3f} FMA instructions per SIMD per cycle "
                      f"(128 SIMDs; at most 1)")
            if min_fill > 0:
                ok = util == util and util >= min_fill
                fails += not ok
                print(f"VERDICT dense fill: {'PASS' if ok else 'FAIL'} {name} ({path}): {util:.3f} FMA per SIMD per cycle "
                      f"(limit at least {min_fill:.2f})")
            print(f"RAMPSUM {name} path={path} waves={info.get('waves', '?')} fma_per_simd_cycle={util:.3f} recorders={len(per)} first_window_MHz="
                  f"{mt['first']:.0f} plateau_1_10ms_MHz={mt['plateau_1_10ms']:.0f} last_window_below_1750_ms="
                  f"{mt['last_below_1750']:.2f} t95_ms={mt['t95']:.2f} final_MHz={mt['final']:.0f} dur_ms={mt['dur']:.1f} "
                  f"step_ms={mt['step']:.2f} start_realtime_ns={arm.load_rt_ns[0] or 0} end_realtime_ns={arm.load_rt_ns[1] or 0}")
            for s, e, c, lo, hi, n in mw:
                if s in (0, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 35, 40, 50, 60, 80, 99) or s % 5 == 0:
                    sy = arm.sys_near(t0 + s * 1e5)
                    print(f"RAMP {name} t_ms={s:.2f} clock_MHz={c:.0f} waves_min_max={lo:.0f}/{hi:.0f} (n {n}) | sysfs "
                          f"sclk={sy['sclk_MHz'] if sy else '-'} busy={sy['busy_pct'] if sy else '-'} "
                          f"power={sy['power_W_reported'] if sy else '-'} hotspot={sy['hotspot_C'] if sy else '-'} "
                          f"throttle={sy['indep_throttle_status'] if sy else '-'} profile={sy.get('power_profile', '-') if sy else '-'}")
            if expect is None:
                chk = steady_check(arm, mw, t0, brk)
                judged = [c for c in chk if c[5] >= 3]
                ok = bool(judged) and all(abs(c[3] / c[4] - 1) <= 0.02 for c in judged if c[4] == c[4] and c[4] > 0)
                txt = "; ".join(f"{lab} {a:.0f}-{b:.0f} ms: counter {cnt:.0f} MHz, pp_dpm_sclk {sc:.0f} MHz ({n} samples)"
                                for lab, a, b, cnt, sc, n in chk) or "no steady window of 8 ms or more"
                fails += not ok
                print(f"VERDICT dense counter_vs_sclk: {'PASS' if ok else 'FAIL'} {name}{' (DELIBERATE BREAK: REALTIME span doubled)' if brk else ''}: "
                      f"{txt} (limit 2 %, at least 3 samples; windows where the counter clock is steady)")
            sy = arm.sys_between(t0 - 2e5, t_end + 2e5)
            if sy:
                print(f"SYSFS {name} during the dispatch: hotspot max {max(float(r['hotspot_C']) for r in sy):.0f} C, "
                      f"power {min(float(r['power_W_reported']) for r in sy):.0f}-{max(float(r['power_W_reported']) for r in sy):.0f} W "
                      f"reported, throttle bits OR 0x{eval_or(sy):x}, profiles {sorted({r.get('power_profile', '-') for r in sy})}")
        else:
            # bursts: one clock per burst (median over recording waves of first-to-last record), and by time into burst
            bc, starts, bins, fills = [], [], {}, []
            tops = {int(b["burst"]): (int(b["top_tick"]), int(b["bottom_tick"])) for b in arm.bursts}
            t_arm = min(int(g["start_rt"]) for g in arm.groups if int(g["end_rt"]))
            for b in range(nb):
                cl = []
                for (bb, r), v in recs.items():
                    if bb != b or len(v) < 2:
                        continue
                    dr = v[-1][0] - v[0][0]
                    if dr > 0:
                        cl.append((v[-1][1] - v[0][1]) / ((2 * dr) if brk else dr) * RT_MHZ)
                    gs = gb.get(b, [])
                    t0 = min(int(g["start_rt"]) for g in gs) if gs else v[0][0]
                    for k in range(1, len(v)):
                        into = (v[k - 1][0] - t0) / 1e5
                        dr2 = v[k][0] - v[k - 1][0]
                        if dr2 > 0:
                            bins.setdefault(int(into / 0.25), []).append((v[k][1] - v[k - 1][1]) / dr2 * RT_MHZ)
                c = q(cl, .5) if cl else float("nan")
                bc.append(c)
                gs = gb.get(b, [])
                starts.append((min(int(g["start_rt"]) for g in gs) - t_arm) / 1e5 if gs else float("nan"))
                fills.append(burst_fill(wchunks.get(b, []), gs, c) if b in wchunks else float("nan"))
            busy = sum(tops[b][1] - tops[b][0] for b in tops)
            span = max(v[1] for v in tops.values()) - min(v[0] for v in tops.values())
            share = busy / span if span else float("nan")
            high = [i for i, c in enumerate(bc) if c > 2000]
            first_step = high[0] if high else None
            lows = [c for c in bc if c <= 2000]
            gpu_ms = [(tops[b][1] - tops[b][0]) / 1e5 for b in sorted(tops)]
            fk = [f for f in fills if f == f]
            print(f"BURSTSUM {name} path={path} waves={info.get('waves', '?')} burst_ms={burst_ms:g} cadence_ms={cadence:g} "
                  f"bursts={nb} measured_busy_share={share:.3f} first_burst_MHz={bc[0]:.0f} "
                  f"bursts_above_2000MHz={len(high)} first_above_2000_at_burst={first_step} "
                  f"at_ms={starts[first_step] if first_step is not None else float('nan'):.1f} "
                  f"fill_min={min(fk) if fk else float('nan'):.3f} fill_p50={q(fk, .5):.3f} "
                  f"fill_max={max(fk) if fk else float('nan'):.3f} fill_bursts={len(fk)} "
                  f"gpu_ms_p50={q(gpu_ms, .5):.3f} gpu_ms_max={max(gpu_ms) if gpu_ms else float('nan'):.3f} "
                  f"group_run_ms_max={max(max_run) if max_run else float('nan'):.3f} "
                  f"start_spread_us_p50={q(spreads, .5):.1f} start_spread_us_max={max(spreads) if spreads else float('nan'):.1f} "
                  f"low_bursts_MHz {dist(lows)} all_bursts_MHz {dist(bc)}")
            if min_fill > 0:
                ok = len(fk) == nb and min(fk) >= min_fill
                fails += not ok
                print(f"VERDICT dense fill: {'PASS' if ok else 'FAIL'} {name} ({path}): fill per burst {dist(fk, '%.3f')} "
                      f"FMA per SIMD per cycle over {len(fk)} of {nb} bursts (limit: every burst at least {min_fill:.2f})")
            for i, c in enumerate(bc):
                if i < 12 or i % 10 == 0 or i == nb - 1 or (first_step is not None and abs(i - first_step) <= 2):
                    t = tops.get(i)
                    sy = arm.sys_near(t[0]) if t else None
                    print(f"BURST {name} i={i} t_ms={starts[i]:.2f} clock_MHz={c:.0f} fill={fills[i]:.3f} gpu_ms="
                          f"{(t[1] - t[0]) / 1e5 if t else float('nan'):.3f} | sysfs sclk={sy['sclk_MHz'] if sy else '-'} "
                          f"power={sy['power_W_reported'] if sy else '-'} hotspot={sy['hotspot_C'] if sy else '-'} "
                          f"busy={sy['busy_pct'] if sy else '-'} throttle={sy['indep_throttle_status'] if sy else '-'} "
                          f"profile={sy.get('power_profile', '-') if sy else '-'}")
            for k in sorted(bins):
                v = bins[k]
                print(f"BIN {name} into_ms={k * 0.25:.2f}-{(k + 1) * 0.25:.2f} samples={len(v)} clock_MHz p5={q(v, .05):.0f} "
                      f"p50={q(v, .5):.0f} p95={q(v, .95):.0f}")
            if tops:
                a = min(v[0] for v in tops.values())
                b = max(v[1] for v in tops.values())
                sy = arm.sys_between(a, b)
                if sy:
                    print(f"SYSFS {name} during the arm: hotspot max {max(float(r['hotspot_C']) for r in sy):.0f} C, "
                          f"power {min(float(r['power_W_reported']) for r in sy):.0f}-"
                          f"{max(float(r['power_W_reported']) for r in sy):.0f} W reported, sclk "
                          f"{dist([float(r['sclk_MHz']) for r in sy])}, throttle bits OR 0x{eval_or(sy):x}, "
                          f"profiles {sorted({r.get('power_profile', '-') for r in sy})}")
    return fails


def eval_or(sy):
    v = 0
    for r in sy:
        try:
            v |= int(r["indep_throttle_status"], 16)
        except (KeyError, ValueError):
            pass
    return v


def summary(run):
    """Pools the arms of every job by arm name without its round suffix (-rN)."""
    ramps, bursts = {}, {}
    for log in sorted(glob.glob(os.path.join(run, "*.dense.txt"))):
        for line in open(log, errors="replace"):
            m = re.match(r"RAMPSUM (\S+) (.*)", line)
            if m:
                kv = dict(re.findall(r"(\w+)=([^\s]+)", m.group(2)))
                ramps.setdefault(re.sub(r"-r\d+$", "", m.group(1)), []).append(kv)
            m = re.match(r"BURSTSUM (\S+) (.*)", line)
            if m:
                kv = dict(re.findall(r"(\w+)=([^\s]+)", m.group(2)))
                bursts.setdefault(re.sub(r"-r\d+$", "", m.group(1)), []).append(kv)
    print("| arm | path | waves | runs | first window MHz | plateau 1-10 ms MHz | step ms | last window below 1750 ms | t95 ms | final MHz | fill |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(ramps.items()):
        f = lambda key, fmt="%.0f": " / ".join(fmt % float(x.get(key, "nan")) for x in v)
        print(f"| {k} | {v[0]['path']} | {v[0]['waves']} | {len(v)} | {f('first_window_MHz')} | {f('plateau_1_10ms_MHz')} | "
              f"{f('step_ms', '%.1f')} | {f('last_window_below_1750_ms', '%.1f')} | {f('t95_ms', '%.1f')} | {f('final_MHz')} | "
              f"{f('fma_per_simd_cycle', '%.3f')} |")
    print()
    print("| arm | path | waves | burst ms | cadence ms | runs | busy share | bursts above 2 GHz (of) | first above 2 GHz at ms | first burst MHz | fill min-max (bursts with records) | burst GPU ms max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(bursts.items()):
        print(f"| {k} | {v[0]['path']} | {v[0]['waves']} | {v[0]['burst_ms']} | {v[0]['cadence_ms']} | {len(v)} | "
              f"{' / '.join(x['measured_busy_share'] for x in v)} | "
              f"{' / '.join(x['bursts_above_2000MHz'] + ' (' + x['bursts'] + ')' for x in v)} | "
              f"{' / '.join(x['at_ms'] for x in v)} | {' / '.join(x['first_burst_MHz'] for x in v)} | "
              f"{' / '.join(x.get('fill_min', '?') + '-' + x.get('fill_max', '?') + ' (' + x.get('fill_bursts', '?') + ')' for x in v)} | "
              f"{' / '.join(x.get('gpu_ms_max', '?') for x in v)} |")


def load_ends(run):
    """Every load end (CLOCK_REALTIME ns) in the run folder: Vulkan arms' INFO lines and the HIP twin's map files."""
    ends = []
    for log in glob.glob(os.path.join(run, "*.log")):
        for line in open(log, errors="replace"):
            m = re.match(r"INFO (\S+) load_realtime_ns start=(\d+) end=(\d+)", line)
            if m:
                ends.append((int(m.group(3)), os.path.basename(log)[:-4] + ":" + m.group(1)))
    for mp in glob.glob(os.path.join(run, "*-dense-*-map.csv")):
        r = rows(mp)
        if r and "sync_host_ns" in r[0]:
            ends.append((int(r[0]["sync_host_ns"]), os.path.basename(mp)[:-8]))
    return sorted(ends)


def f2(run):
    """F2: each target's ramp with its predecessor and the time from the latest earlier load end to its load start."""
    ends = load_ends(run)
    table = {}
    print("| job | order | target | predecessor | pause set s | since last load end s (which) | first window MHz | plateau 1-10 ms MHz | step ms | t95 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    jobs = []
    for log in glob.glob(os.path.join(run, "f2-*.log")):
        job = os.path.basename(log)[:-4]
        txt = open(log, errors="replace").read()
        m = re.search(r"SEQ order=(\d+) target=(\S+) pred=(\S+) pause_s=(\S+)", txt)
        d = os.path.join(run, job + ".dense.txt")
        if not m or not os.path.exists(d):
            continue
        rs = [ln for ln in open(d, errors="replace") if ln.startswith("RAMPSUM ")]
        if not rs:
            continue
        kv = dict(re.findall(r"(\w+)=([^\s]+)", rs[-1]))
        jobs.append((int(m.group(1)), job, m.group(2), m.group(3), m.group(4), kv))
    for order, job, tgt, pred, pause, kv in sorted(jobs):
        st = int(kv.get("start_realtime_ns", "0"))
        prev = [e for e in ends if e[0] < st and not e[1].startswith(job + ":target")]
        since = (st - prev[-1][0]) / 1e9 if prev and st else float("nan")
        which = prev[-1][1] if prev else "-"
        key = (tgt, pred, pause)
        table.setdefault(key, []).append((float(kv["first_window_MHz"]), float(kv["plateau_1_10ms_MHz"]),
                                          float(kv["step_ms"]), float(kv["t95_ms"]), since, order))
        print(f"| {job} | {order} | {tgt} | {pred} | {pause} | {since:.2f} ({which}) | {kv['first_window_MHz']} | "
              f"{kv['plateau_1_10ms_MHz']} | {kv['step_ms']} | {kv['t95_ms']} |")
    print()
    print("| target | predecessor | pause set s | runs | since last load end s (min-max) | first window MHz (each run) | plateau MHz (min-max) | step ms (each run, in run order) | steps before 20 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (tgt, pred, pause), v in sorted(table.items()):
        v = sorted(v, key=lambda x: x[5])
        st = [x[2] for x in v]
        print(f"| {tgt} | {pred} | {pause} | {len(v)} | {min(x[4] for x in v):.1f}-{max(x[4] for x in v):.1f} | "
              f"{' '.join('%.0f' % x[0] for x in v)} | {min(x[1] for x in v):.0f}-{max(x[1] for x in v):.0f} | "
              f"{' '.join('%.2f' % x for x in st)} | {sum(1 for x in st if x == x and x < 20)} of {len(st)} |")


def loop_ops(path, marker):
    """Instruction counts of the loop that holds the 128 FMAs: from the label before the first v_fma_f32 to the
    branch that closes it, plus the scalar block that follows up to the next label that jumps back."""
    lines = [ln.split(";")[0].strip() for ln in open(path, errors="replace")]
    if marker:
        try:
            i0 = next(i for i, ln in enumerate(lines) if marker in ln)
            lines = lines[i0:]
        except StopIteration:
            pass
    idx = [i for i, ln in enumerate(lines) if ln.startswith("v_fma_f32")]
    if not idx:
        return {}, 0
    a, b = idx[0], idx[-1]
    body = [ln.split()[0] for ln in lines[a:b + 1] if ln and not ln.endswith(":")]
    c = {}
    for op in body:
        c[op] = c.get(op, 0) + 1
    return c, len(idx)


def listing(vk, hip):
    cv, nv = loop_ops(vk, "")
    ch, nh = loop_ops(hip, "k_dense")
    fv = {k: v for k, v in cv.items() if k.startswith("v_")}
    fh = {k: v for k, v in ch.items() if k.startswith("v_")}
    print(f"LISTING vk loop body (first to last v_fma_f32): {sorted(cv.items())}")
    print(f"LISTING hip loop body (first to last v_fma_f32): {sorted(ch.items())}")
    ok = nv == 128 and nh == 128 and fv == fh == {"v_fma_f32": 128}
    print(f"VERDICT dense same_loop: {'PASS' if ok else 'FAIL'} both loops hold 128 v_fma_f32 in one unbroken block with "
          f"no other vector instruction between them (Vulkan {nv}, HIP {nh}; vector ops vk {fv}, hip {fh})")
    return 0 if ok else 1


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "job":
        expect = "stop" if "--expect-stop" in sys.argv else "nostop" if "--expect-no-stop" in sys.argv else None
        return 1 if analyse_job(sys.argv[2], sys.argv[3], "--break-windows" in sys.argv, expect) else 0
    if len(sys.argv) >= 3 and sys.argv[1] == "f2":
        f2(sys.argv[2])
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2])
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "listing":
        return listing(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
