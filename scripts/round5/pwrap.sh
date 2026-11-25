#!/usr/bin/env bash
# usage: pwrap.sh <n bursts> <burst cmd...> -- <check cmd...>
# Runs the bit-check in the background and, beside it, <n> heavy bursts, each started only when the hotspot is below 55 C
# (sensor polled with no load). Stops only its own processes by PID.
HOT=${HOTSPOT_INPUT:?set HOTSPOT_INPUT to the hwmon temp2_input file of the card}
n=$1; shift
burst=(); while [[ $1 != -- ]]; do burst+=("$1"); shift; done; shift
"$@" &
cp=$!
sleep 2
for ((i = 1; i <= n; i++)); do
  w=0
  while (( $(cat $HOT) / 1000 >= 55 )); do sleep 0.5; w=$((w+1)); (( w > 240 )) && { echo "PWRAP: hotspot stayed >= 55 C for 120 s, burst $i skipped"; break 2; }; done
  echo "PWRAP burst $i at $(date -u +%T.%N) hotspot $(( $(cat $HOT) / 1000 )) C"
  "${burst[@]}"; echo "PWRAP burst $i exit $?"
  sleep 3
done
wait "$cp"; rc=$?
echo "PWRAP check exit $rc"
exit $rc
