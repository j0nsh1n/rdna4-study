#!/usr/bin/env bash
# Item E, inside one gated job: a background FP16 matrix loop (r9_bg, the compute-only load that held the memory clock near
# 772 MHz in round 7) runs while the latency and bandwidth probes run beside it. The loop is throttled (it pauses at a
# hotspot cap and resumes 4 C lower), so it is still running when the probes are; this script checks that after the
# start and after each probe (the process is alive and its launch count moved since the last check), prints one
# "INFO bg alive after <step> launches=<n>" or "INFO bg DEAD after <step> ..." line per check, and exits 13 if any check
# failed or the loop ended on its tripwire: the analysis then takes none of this job's points.
# The loop ends by itself after 40 s and is stopped by its PID when the probes finish. Usage: e_bg_inner.sh <out dir> <tag>
# Environment: R9_BIN (the binaries), R9_BG_CAP (hotspot cap in C, default 76), DRM_DEVICE.
set -u
out=$1 tag=$2
here=$(cd "$(dirname "$0")" && pwd)
bin=${R9_BIN:-$(cd "$here/../.." && pwd)/build/round9/bin}
mkdir -p "$out"
rm -f "$out/$tag.bg.ready" "$out/$tag.bg.ready.count"
"$bin/r9_bg" 40 "$out/$tag.bg.ready" 128 20000 "${R9_BG_CAP:-76}" >"$out/$tag.bg.out" 2>&1 &
bp=$!
for _ in $(seq 1 100); do [[ -e $out/$tag.bg.ready ]] && break; kill -0 "$bp" 2>/dev/null || break; sleep 0.1; done
[[ -e $out/$tag.bg.ready ]] || { echo "BACKGROUND LOOP NOT READY: $(head -c 200 "$out/$tag.bg.out")"; exit 12; }
sleep 1.5
bad=0 last=0
bg_check() { # bg_check <step>
    local alive=1 n
    kill -0 "$bp" 2>/dev/null || alive=0
    n=$(cat "$out/$tag.bg.ready.count" 2>/dev/null || echo 0)
    if ((alive && n > last)); then
        echo "INFO bg alive after $1 launches=$n"
    else
        echo "INFO bg DEAD after $1 alive=$alive launches=$n previous=$last"
        bad=1
    fi
    last=$n
}
bg_check "start"
echo "INFO background loop running, uclk=$(od -An -tu2 -j58 -N2 "$DRM_DEVICE/gpu_metrics" | tr -d ' ')"
"$bin/r9_lat" lat 16,1024,2048,4096,32768,524288 3
rc1=$?
bg_check "r9_lat"
"$bin/r9_bw" 32768,1048576
rc2=$?
bg_check "r9_bw"
kill -TERM "$bp" 2>/dev/null
wait "$bp" 2>/dev/null
brc=$?
cat "$out/$tag.bg.out"
((brc == 5)) && { echo "INFO bg DEAD: the loop ended on its 88 C tripwire"; bad=1; }
((bad)) && { echo "BACKGROUND LOOP NOT RUNNING THROUGH THE PROBES: this job's points are rejected"; exit 13; }
exit $((rc1 || rc2))
