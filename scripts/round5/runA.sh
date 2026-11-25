#!/usr/bin/env bash
# usage: runA.sh <state L|H> <rep> : all 16 GPU jobs of the arithmetic lane, each gated and watched
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
cd "$REPO" || exit 2
st=$1; rep=$2
while read -r prog arg; do
  n=A-$st$rep-$prog-$arg
  if [[ $st == H ]]; then
    $HERE/g.sh $n 70 200 $HERE/with_spin.sh 30 build/arith/bin/$prog $arg >/dev/null 2>&1
  else
    $HERE/g.sh $n 70 200 build/arith/bin/$prog $arg >/dev/null 2>&1
  fi
  rc=$?
  v=$(grep -h '^VERDICT' $R/$n.log | cut -c1-60)
  echo "$n rc=$rc $v"
  ((rc==0)) || { echo "STOP: $n rc=$rc"; exit $rc; }
done < $R/arith_jobs.txt
