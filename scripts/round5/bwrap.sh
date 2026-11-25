#!/usr/bin/env bash
# usage: bwrap.sh <engines csv> <sampler seconds> <cmd...>: the unchanged engine sampler beside one command (own PIDs only)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
csv=$1; s=$2; shift 2
python3 -I "$HERE/engine_sampler.py" "$csv" "$s" 0.2 &
sp=$!
sleep 1
export VKG_OUT=$R/B VKG_SPV=$R/b/spv
echo "BWRAP command pid $$ start $(date -u +%T.%N)"
"$@"; rc=$?
echo "BWRAP command exit $rc end $(date -u +%T.%N)"
wait "$sp"
exit $rc
