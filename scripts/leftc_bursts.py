#!/usr/bin/env python3
"""Leftovers C, lane B: combines the single-burst whole-card jobs (one burst per locked job, so that no cool-down
wait ever holds the GPU lock) into the items' tables and VERDICT lines.

Usage:
  leftc_bursts.py mt5c RUN_DIR [--break]   mat_card bursts tagged mt5c: FP16 at W = 1 and 16, launch lengths
                                           50, 100, 200 ms; loss = 1 - ratio(W=16) / ratio(W=1) per length.
                                           Calibration (matrix-timing item 11, 0.17 s launches): W = 1 at 200 ms
                                           ratio 1.000 +-0.005, W = 16 at 200 ms 0.977 +-0.006. The classification
                                           (fixed per-launch cost: loss(50) / loss(200) >= 2.5; per instruction:
                                           0.5 .. 1.3; below 0.5 the loss grows with the launch length) is INFO. --break claims W = 16 equals W = 1 within 0.2 %.
  leftc_bursts.py mt3 RUN_DIR [--break]    mat_card bursts tagged mt3: every form's W = 1 ratio to 128 SIMDs x
                                           clock / cycles must lie in 0.99 .. 1.01 (items 11, 11a). --break halves
                                           the expected cycles.
  leftc_bursts.py ld9 RUN_DIR [--break]    lds_tr bw2 bursts: see that mode's usage text.
Reads every *.log in RUN_DIR with BURST (or TRBW) lines of the given tag. Exit 1 on FAIL or when bursts are missing.
"""
import glob
import os
import statistics
import sys


def bursts(run_dir, tag, key="BURST"):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*.log"))):
        with open(p, errors="replace") as f:
            for line in f:
                if line.startswith(key + " ") and f"tag={tag}" in line:
                    d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                    d["_file"] = os.path.basename(p)
                    out.append(d)
    return out


def med(v):
    return statistics.median(v) if v else float("nan")


def mt5c(run_dir, brk):
    b = bursts(run_dir, "mt5c")
    cell = {}
    for d in b:
        cell.setdefault((int(d["W"]), int(float(d["target_ms"]))), []).append(d)
    rows, ok = {}, True
    for (w, ms), ds in sorted(cell.items()):
        re_ = [float(d["ratio_event"]) for d in ds]
        rs = [float(d["ratio_span"]) for d in ds]
        rows[(w, ms)] = (med(re_), med(rs))
        print(f"RESULT mt5c W={w} launch_ms={ms} bursts={len(ds)} ratio_event median {med(re_):.4f} (min {min(re_):.4f} max "
              f"{max(re_):.4f}) ratio_span median {med(rs):.4f} clock_MHz {med([float(d['clock_MHz']) for d in ds]):.0f} "
              f"event_ms {med([float(d['event_ms']) for d in ds]):.2f} hotspot_max {max(float(d['hotspot_C_max']) for d in ds):.0f}")
    loss = {}
    for ms in (50, 100, 200):
        if (1, ms) in rows and (16, ms) in rows:
            for k, name in ((0, "event"), (1, "span")):
                loss[(ms, name)] = 1 - rows[(16, ms)][k] / rows[(1, ms)][k]
            print(f"RESULT mt5c launch_ms={ms} loss_event {100 * loss[(ms, 'event')]:.2f} % loss_span "
                  f"{100 * loss[(ms, 'span')]:.2f} %")
    if (50, "event") in loss and (200, "event") in loss and loss[(200, "event")] > 0:
        q = loss[(50, "event")] / loss[(200, "event")]
        kind = ("fixed per launch" if q >= 2.5 else "grows with the launch length (neither rule)" if q < 0.5
                else "per instruction" if q <= 1.3 else "neither rule")
        print(f"INFO mt5c loss(50 ms) / loss(200 ms) = {q:.2f} -> {kind}")
    c1, c16 = rows.get((1, 200)), rows.get((16, 200))
    if not c1 or not c16 or len(cell) < 6:
        print("VERDICT mt5c_calib: FAIL bursts missing (each burst needs a hotspot below 55 C to start)")
        return 1
    if brk:
        ok = abs(c16[0] - c1[0]) <= 0.002
        text = f"break: claims W=16 ({c16[0]:.4f}) equals W=1 ({c1[0]:.4f}) within 0.2 %"
    else:
        ok = abs(c1[0] - 1.000) <= 0.005 and abs(c16[0] - 0.977) <= 0.006
        text = f"W=1 at 200 ms ratio {c1[0]:.4f} (1.000 +-0.005), W=16 {c16[0]:.4f} (item 11: 0.977 +-0.006)"
    print(f"VERDICT mt5c_calib: {'PASS' if ok else 'FAIL'} {text}")
    return 0 if ok else 1


def mt3(run_dir, brk):
    b = bursts(run_dir, "mt3")
    forms = {}
    for d in b:
        forms.setdefault(d["form"], {}).setdefault(int(d["W"]), []).append(d)
    ok, n = True, 0
    for f, ws in sorted(forms.items()):
        for w, ds in sorted(ws.items()):
            f_ = 2.0 if brk else 1.0
            r = med([float(x["ratio_event"]) for x in ds]) * f_
            tops = med([float(x["TOPS_event"]) for x in ds])
            print(f"RESULT mt3 form={f} W={w} bursts={len(ds)} ratio {r:.4f} TOPS {tops:.1f} cycles_claimed "
                  f"{float(ds[0]['cycles']) / f_:.0f} clock_MHz {med([float(x['clock_MHz']) for x in ds]):.0f} "
                  f"hotspot_max {max(float(x['hotspot_C_max']) for x in ds):.0f}")
            if w == 1:
                n += 1
                ok = ok and 0.99 <= r <= 1.01
    ok = ok and n == 6
    print(f"VERDICT mt3_ratio: {'PASS' if ok else 'FAIL'} {n} of 6 forms measured at W=1, every ratio in 0.99..1.01"
          + (" (break: half the cycles claimed)" if brk else ""))
    return 0 if ok else 1


def ld9(run_dir, brk):
    b = bursts(run_dir, "ld9", key="TRBW")
    cell = {}
    for d in b:
        cell.setdefault((d["kind"], d["share"], int(d["window_MiB"])), []).append(float(d["GBs"]))
    for k in sorted(cell):
        v = cell[k]
        print(f"RESULT ld9 kind={k[0]} tiles={k[1]} window_MiB={k[2]} bursts={len(v)} GB/s median {med(v):.0f} (min {min(v):.0f} "
              f"max {max(v):.0f})")
    shared32 = cell.get(("tr_b128", "shared", 32))
    dram = cell.get(("tr_b128", "shared", 512))
    uniq32 = cell.get(("tr_b128", "unique", 32))
    if not shared32 or not dram or not uniq32:
        print("VERDICT ld9_calib: FAIL bursts missing (each burst needs a hotspot below 55 C to start)")
        return 1
    # calibration: lds section 5 (5,346 GB/s at 32 MiB shared, +-10 %; DRAM 629); --break claims twice DRAM
    want_dram = 2 * 629 if brk else 629
    ok = abs(med(shared32) - 5346) <= 0.10 * 5346 and abs(med(dram) - want_dram) <= 0.05 * want_dram
    print(f"VERDICT ld9_calib: {'PASS' if ok else 'FAIL'} tr_b128 shared 32 MiB {med(shared32):.0f} GB/s (section 5: 5346 "
          f"+-10 %), DRAM {med(dram):.0f} (expected {want_dram} +-5 %)")
    print(f"INFO ld9 unique tiles at 32 MiB: {med(uniq32):.0f} GB/s against the sharing explanation's bound 2400 "
          f"(lane 2's memory-side cache 2.0 TB/s x 1.2) -> {'consistent' if med(uniq32) <= 2400 else 'NOT consistent'}")
    return 0 if ok else 1


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("mt5c", "mt3", "ld9"):
        print(__doc__)
        return 2
    brk = "--break" in sys.argv
    return {"mt5c": mt5c, "mt3": mt3, "ld9": ld9}[sys.argv[1]](sys.argv[2], brk)


if __name__ == "__main__":
    sys.exit(main())
