#!/usr/bin/env bash
# Runs as the command of one gated GPU job (scripts/study-llm.sh r4dev ...): a read-only sampler (r9_sampler, 5 ms, sysfs
# only) starts first, then the load, then a 1 s tail, then the sampler is stopped by its PID. The load's output goes to
# <out>/<tag>.out and its result lines are repeated on stdout (so the gate's log carries them).
# Usage: jobwrap.sh <out dir> <tag> <command...>        Environment: DRM_DEVICE and HWMON (the gate exports them from the card record).
set -u
out=$1 tag=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
bin=${R9_BIN:-$(cd "$here/../.." && pwd)/build/round9/bin}
mkdir -p "$out/$tag.watch"
mono() { python3 -I -c 'import time; print(time.monotonic_ns())'; }
"$bin/r9_sampler" "$out/$tag.watch" 5 >"$out/$tag.sampler.log" 2>&1 &
wp=$!
for _ in $(seq 1 50); do [[ -e $out/$tag.watch/ready ]] && break; kill -0 "$wp" 2>/dev/null || break; sleep 0.1; done
[[ -e $out/$tag.watch/ready ]] || { echo "SAMPLER NOT READY: $(head -c 200 "$out/$tag.sampler.log")"; exit 11; }
sleep 0.5
echo "JOB_MONO_START $(mono)"
VKG_OUT=$out VKG_TAG=$tag "$@" >"$out/$tag.out" 2>&1
rc=$?
echo "JOB_MONO_END $(mono) rc=$rc"
grep -E '^(CELL|RESULT|ROW|VERDICT|THERMAL|STATE|INFO|NOT RUN|device|CLOCKS)' "$out/$tag.out"
sleep 1
kill -TERM "$wp" 2>/dev/null
wait "$wp" 2>/dev/null
exit "$rc"
