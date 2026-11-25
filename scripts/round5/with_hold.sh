#!/usr/bin/env bash
# usage: with_hold.sh <cmd...>: an idle HIP process with a compute queue open beside the command (own PID only)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
$R/bin/hip_hold 30 >$R/hold-last.txt 2>&1 &
hp=$!
sleep 3
echo "HOLD pid $hp"; cat /proc/$hp/status | grep -c . >/dev/null
"$@"; rc=$?
kill -TERM "$hp" 2>/dev/null; wait "$hp" 2>/dev/null
exit $rc
