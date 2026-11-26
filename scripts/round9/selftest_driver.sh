#!/usr/bin/env bash
# CPU-only self-test of what every round 9 driver shares (common.sh): (1) finish() ends with the analysis's failure status, or 1 when a
# gated job failed, and only 0 when both are fine (the old `analysis | tee` pattern returns tee's 0: shown below as the break);
# (2) no committed script or source names a home directory: the lock and hold defaults live under runs/ of the checkout and move with it.
# Usage: scripts/round9/selftest_driver.sh        Exit 0 only if every check passes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
t=$repo/build/round9/tmp-selftest-driver
rm -rf "$t" && mkdir -p "$t"
rc=0
run_finish() { # run_finish <analysis exit status> <gated job failures> -> the driver's exit status
    (
        export R9_REPO=$repo CARD_ENV=/nonexistent
        . "$here/common.sh"
        RUN=$t
        GATE_FAILS=$2
        finish bash -c "echo analysis output; exit $1" >/dev/null
    )
    echo $?
}
check() { # check <name> <got> <want>
    if [[ $2 == "$3" ]]; then echo "PASS $1 (exit $2)"; else echo "FAIL $1: exit $2, wanted $3"; rc=1; fi
}
check "analysis passes, no job failed" "$(run_finish 0 0)" 0
check "a control fails (analysis exit 1), jobs fine" "$(run_finish 1 0)" 1
check "analysis passes but a gated job failed" "$(run_finish 0 2)" 1
check "analysis exit 3 is kept" "$(run_finish 3 0)" 3
old=$(bash -c 'bash -c "exit 1" | tee /dev/null >/dev/null; echo $?')
check "break: the first drivers' pattern 'analysis | tee' (tee's status, hides the failure)" "$old" 0
n=$(grep -rn "/home/\|$(id -un)" "$repo/scripts/round9" "$repo/src/round9" --include='*' 2>/dev/null | grep -v "selftest_driver.sh" | wc -l)
check "no committed round 9 script or source names /home/ or a user name" "$n" 0
d=$(bash -c 'unset STUDY_GPU_LOCK R9_HOLD; export R9_REPO='"$repo"' CARD_ENV=/nonexistent; . '"$here"'/common.sh; echo "$STUDY_GPU_LOCK $HOLD"')
[[ $d == "$repo/runs/gpu.lock $repo/runs/HOLD-FSR" ]] && echo "PASS lock and hold defaults follow the checkout" || { echo "FAIL defaults: $d"; rc=1; }
# (3) gate() refuses while RESTORE-FAILED exists (round 10 P0), and starts the job once it is gone (the control). The repository is a
# fake with a stub scripts/study-llm.sh, so no real job can start.
fk=$t/fakerepo
mkdir -p "$fk/scripts" "$t/gaterun"
printf '#!/bin/sh\necho ran >>"%s/job-ran"\n' "$t" >"$fk/scripts/study-llm.sh"
chmod +x "$fk/scripts/study-llm.sh"
run_gate() { # run_gate <RESTORE-FAILED path> -> the gate's exit status
    (
        export R9_REPO=$fk CARD_ENV=/nonexistent R9_RESTORE_FAILED=$1 R9_HOLD=$t/nohold R9_PAUSE=0
        . "$here/common.sh"
        RUN=$t/gaterun
        gate probe 60 5 true >"$t/gate.out" 2>&1
    )
    echo $?
}
: >"$t/RESTORE-FAILED"
g=$(run_gate "$t/RESTORE-FAILED")
[[ $g != 0 && ! -e $t/job-ran ]] && grep -q "REFUSED probe" "$t/gate.out" && echo "PASS gate() refuses while RESTORE-FAILED exists (exit $g, job not started)" || { echo "FAIL gate with RESTORE-FAILED: exit $g"; rc=1; }
g=$(run_gate "$t/absent")
[[ $g == 0 && -e $t/job-ran ]] && echo "PASS control: without the file the same gate() call runs its job" || { echo "FAIL gate control: exit $g"; rc=1; }
exit $rc
