#!/usr/bin/env bash
# Item E2, inside one gated job: a background streaming read (r9_bgread, a second process, its own 1 GiB buffer, a few
# work-groups) holds the memory clock up while single-wave chases run beside it (r9_lat, L2 1 MiB, memory-side cache 32 MiB,
# GDDR6 1 GiB, random and page order). The background ends by itself after <seconds> and is stopped by its PID when the chases are
# done. Usage: e2_inner.sh <out dir> <tag> <blocks|0> [settle seconds=4]      blocks 0: no background (the idle control)
set -u
out=$1 tag=$2 blocks=$3 settle=${4:-4}
here=$(cd "$(dirname "$0")" && pwd)
bin=${R9_BIN:-$(cd "$here/../.." && pwd)/build/round9/bin}
sizes=${R9_E2_SIZES:-1024,32768,1048576}
mkdir -p "$out"
uclk() { od -An -tu2 -j58 -N2 "$DRM_DEVICE/gpu_metrics" | tr -d ' '; }
bp=""
if [[ $blocks -gt 0 ]]; then
    rm -f "$out/$tag.bg.ready"
    "$bin/r9_bgread" 55 "$out/$tag.bg.ready" "$blocks" >"$out/$tag.bg.out" 2>&1 &
    bp=$!
    for _ in $(seq 1 200); do [[ -e $out/$tag.bg.ready ]] && break; kill -0 "$bp" 2>/dev/null || break; sleep 0.1; done
    [[ -e $out/$tag.bg.ready ]] || { echo "BACKGROUND READ NOT READY: $(head -c 200 "$out/$tag.bg.out")"; exit 12; }
    sleep "$settle"
fi
echo "INFO e2 blocks=$blocks background settled, uclk=$(uclk)"
"$bin/r9_lat" lat "$sizes" 3 both 60
rc=$?
if [[ -n $bp ]]; then
    kill -TERM "$bp" 2>/dev/null
    wait "$bp" 2>/dev/null
    cat "$out/$tag.bg.out"
fi
echo "INFO e2 blocks=$blocks chases done, uclk=$(uclk)"
exit $rc
