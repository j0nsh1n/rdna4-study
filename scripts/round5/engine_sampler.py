"""Read-only: every PERIOD s, per process holding the RX 9070 XT, the summed drm-engine-* busy ns (fdinfo).
Usage: python3 engine_sampler.py OUT.csv SECONDS [PERIOD]"""
import os, sys, time
PDEV = "0000:03:00.0"
out, secs = sys.argv[1], float(sys.argv[2])
period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

def sample():
    rows = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        d = f"/proc/{pid}/fdinfo"
        try:
            fds = os.listdir(d)
        except OSError:
            continue
        seen = {}
        for fd in fds:
            try:
                with open(f"{d}/{fd}") as f:
                    t = f.read()
            except OSError:
                continue
            if f"drm-pdev:\t{PDEV}" not in t:
                continue
            kv = dict(l.split(":", 1) for l in t.splitlines() if ":" in l)
            cid = kv.get("drm-client-id", "").strip()
            if cid in seen:
                continue
            seen[cid] = {k[len("drm-engine-"):]: int(v.split()[0]) for k, v in kv.items() if k.startswith("drm-engine-")}
        if seen:
            try:
                name = open(f"/proc/{pid}/comm").read().strip()
            except OSError:
                name = "?"
            tot = {}
            for e in seen.values():
                for k, v in e.items():
                    tot[k] = tot.get(k, 0) + v
            rows[pid] = (name, tot)
    return rows

with open(out, "w") as f:
    f.write("t_s,pid,name,engine,busy_ns\n")
    t0 = time.monotonic()
    while (now := time.monotonic() - t0) < secs:
        for pid, (name, tot) in sample().items():
            for k, v in tot.items():
                f.write(f"{now:.3f},{pid},{name},{k},{v}\n")
        f.flush()
        time.sleep(max(0.0, period - (time.monotonic() - t0 - now)))
