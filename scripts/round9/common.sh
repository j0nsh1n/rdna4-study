# shellcheck shell=bash
# Sourced by the round 9 drivers: paths, the gate call, the hold file, the wait for an idle memory clock.
# STUDY_GPU_LOCK and R9_HOLD default to paths under runs/ of the checkout (set them in the environment when the layout differs).
# R9_REPO (default: this checkout) is where scripts/study-llm.sh and runs/latest/card.env live; after the branch is merged
# it is the main repository. R9_RUN names the run folder (default runs/<UTC>-r9<item>).
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${R9_REPO:-$(cd "$here/../.." && pwd)}
BIN=$REPO/build/round9/bin
cd "$REPO" || exit 2
export STUDY_GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
export LLM_EXPECT_PROFILE=${LLM_EXPECT_PROFILE:-Default}
export R9_BIN=$BIN VKG_SPV=$REPO/build/round9/spv
HOLD=${R9_HOLD:-$REPO/runs/HOLD-FSR}
# RESTORE-FAILED is written by scripts/round10/pin_mclk.sh when it could not put the card's power level and memory-clock mask back;
# gate() refuses to start any job while it exists (remove it by hand after checking both sysfs files).
RESTORE_FAILED=${R9_RESTORE_FAILED:-$REPO/runs/RESTORE-FAILED}
CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
if [[ -r $CARD_ENV ]]; then
    # shellcheck disable=SC1090
    . "$CARD_ENV"
    export DRM_DEVICE HWMON
fi
uclk() { od -An -tu2 -j58 -N2 "${DRM_DEVICE:-/nonexistent}/gpu_metrics" 2>/dev/null | tr -d ' '; }

# start_run <item>: creates the run folder, copies the expectations, builds if a binary is missing.
start_run() {
    local item=$1
    RUN=${R9_RUN:-$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r9$item}
    export LLM_RUN=$RUN
    mkdir -p "$RUN"
    cp "$here/expect/expect-$item.md" "$RUN/expect.md"
    [[ -x $BIN/vkr9 && -x $BIN/r9_lat ]] || "$here/build.sh" >"$RUN/build.log" 2>&1 || { echo "build failed, see $RUN/build.log"; exit 2; }
    echo "run folder $RUN"
}

# gate_refuse_if_restore_failed <name>: non-zero (and counts as a failed gate) while the RESTORE-FAILED file exists.
gate_refuse_if_restore_failed() {
    [[ -e $RESTORE_FAILED ]] || return 0
    echo "$(date -u +%T) REFUSED $1: $RESTORE_FAILED exists (a memory-clock restore failed; check the card's power_dpm_force_performance_level and pp_dpm_mclk by hand, then remove the file)"
    GATE_FAILS=$((GATE_FAILS + 1))
    return 1
}

# gate <name> <hot limit C> <timeout s> <command...>: one gated GPU job. Waits while the hold file exists.
gate() {
    local name=$1 hot=$2 tmo=$3
    shift 3
    gate_refuse_if_restore_failed "$name" || return 9
    while [[ -e $HOLD ]]; do sleep 5; done
    gate_refuse_if_restore_failed "$name" || return 9  # it can appear during the hold wait
    echo "$(date -u +%T) start $name"
    "$REPO/scripts/study-llm.sh" r4dev "r9-$name" "$hot" "$tmo" "$@" >"$RUN/$name.stdout" 2>&1
    local rc=$?
    echo "$(date -u +%T) end $name exit=$rc"
    ((rc != 0)) && GATE_FAILS=$((GATE_FAILS + 1))
    sleep "${R9_PAUSE:-10}"
    return $rc
}
GATE_FAILS=0

# finish <analysis command...>: runs the analysis (its output on the screen and in $RUN/analysis.txt) and ends the driver with
# a status that cannot hide a failure: the analysis's own status if it failed (a failed control), else 1 if any gated job failed.
# (A plain `analysis | tee` returns tee's status, which is how the first drivers logged rc=0 beside failed controls.)
finish() {
    "$@" | tee "$RUN/analysis.txt"
    local a=${PIPESTATUS[0]}
    echo "DRIVER analysis exit=$a, gated jobs that failed: $GATE_FAILS" | tee -a "$RUN/analysis.txt"
    ((a != 0)) && exit "$a"
    ((GATE_FAILS != 0)) && exit 1
    exit 0
}

# wait_idle_uclk: until the memory clock reads 456, at most 90 s (read only); logs what it saw.
wait_idle_uclk() {
    local w=0
    while [[ $(uclk) != 456 && $w -lt 90 ]]; do sleep 1; w=$((w + 1)); done
    echo "$(date -u +%FT%T.%3NZ) before $1: uclk=$(uclk) extra_wait_s=$w" >>"$RUN/between-jobs.log"
}
