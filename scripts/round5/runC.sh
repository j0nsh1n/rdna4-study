#!/usr/bin/env bash
# usage: runC.sh <state L|H> <rep> : the GPU jobs of the conversions lane (cvt_time is timing only and is left out)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
R=${R_RUN:?set R_RUN to the run folder (it holds bin/, watch/ and the job logs)}
cd "$REPO" || exit 2
st=$1; rep=$2
while read -r prog args; do
  tag=$(echo "$args" | tr ' ' '_')
  n=A-$st$rep-$prog-$tag
  if [[ $st == H ]]; then
    $HERE/g.sh $n 60 300 $HERE/with_spin.sh 60 build/bin/$prog $args >/dev/null 2>&1
  else
    $HERE/g.sh $n 60 300 build/bin/$prog $args >/dev/null 2>&1
  fi
  rc=$?
  v=$(grep -h '^VERDICT' $R/$n.log | cut -c1-60 | head -1)
  echo "$n rc=$rc $v"
  ((rc==0)) || { echo "STOP: $n rc=$rc hotspot=$(sed -n 's/^hotspot_max=//p' $R/$n.meta)"; exit $rc; }
done < $R/conv_jobs.txt
