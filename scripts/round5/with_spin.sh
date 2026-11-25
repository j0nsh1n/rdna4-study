#!/usr/bin/env bash
# usage: with_spin.sh <spin seconds> <cmd...>: runs a one-wave spinner beside the command, stops only its own spinner by PID
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
s=$1; shift
$R/bin/spin "$s" 1 >"$R/spin-last.txt" 2>&1 &
sp=$!
sleep 2
"$@"
rc=$?
kill -TERM "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
exit "$rc"
