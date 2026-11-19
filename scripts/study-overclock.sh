#!/usr/bin/env bash
# Overclock comparison (owner decision 2026-11-17): the same short measurements on the LACT profile "Testing" (stock
# clocks, 317 W cap), then on "Default" (the owner's overclock: 374 W cap, +150 MHz, -50 mV, memory request
# 1409 MHz), then on "Testing" again. Short bursts only: the whole-card probes run with --short, each burst about
# 0.1 to 0.2 s with the in-process 95 C stop of mat_card; no sustained load on the overclock profile.
#
# Usage: scripts/study-overclock.sh            (needs build/study/bin/mat_card and build/gfx1201-rocm10/bandwidth)
#        ROUNDS=stock scripts/study-overclock.sh   two rounds on "Testing" only, for the spread of the stock numbers;
#                                                  this form never calls `lact cli profile set`: if the profile is
#                                                  not "Testing" when a round starts, it stops instead.
# Results: runs/<UTC time>-overclock/ (or the folder named in OVERCLOCK_RUN_DIR): one log per profile round and
# job, profile-<round>.txt, summary.md.
#
# This script changes a card setting: it switches the LACT profile with `lact cli profile set`. It switches back to
# "Testing" on every exit the shell can act on (normal end, an error, INT, TERM, HUP), tries twice, checks that the
# switch took, and exits 5 if it did not (the summary then says RESTORE FAILED; the shared lock is given up when the
# script ends, even though the card may still be on the overclock profile). Signals are ignored while it switches
# back: one that arrives then is lost, and the comparison form goes on to its last stock round. It cannot switch
# back after SIGKILL, and it acts on INT, TERM or HUP only once the running step has ended (a job: at most 260 s;
# a wait for the lock: 60 s; a lact call: 20 s). It runs nothing, and calls no `profile set`, if the profile at the
# start is not "Testing".
# Every GPU job takes the shared lock (waiting at most 60 s; a longer wait ends the run), runs under nice with a time
# limit, and starts only with at least 8 GB of memory available and a hotspot below 60 C, both checked again once
# the lock is held. For the overclock round the lock is taken before the switch and kept until the card is back on
# "Testing", so no other job that uses the lock can run on the overclock profile. A hotspot of 95 C or more after
# any job ends the run. A sensor or memory figure that cannot be read counts as a failed check.
# scripts/study-overclock-selftest.sh exercises these paths with a stand-in for lact.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH HIP_VISIBLE_DEVICES=0 CARD_ARCH=gfx1201

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -r $CARD_ENV ]] || { echo "study-overclock: no card record at $CARD_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CARD_ENV"
export DRM_DEVICE HWMON
[[ $HIP_INDEX == 0 ]] || { echo "study-overclock: device index is $HIP_INDEX, expected 0" >&2; exit 2; }

MAT=$REPO/build/study/bin/mat_card
BW=$REPO/build/gfx1201-rocm10/bandwidth
[[ -x $MAT && -x $BW ]] || { echo "study-overclock: build mat_card and bandwidth first" >&2; exit 2; }

MODE=${ROUNDS:-compare}
[[ $MODE == compare || $MODE == stock ]] || { echo "study-overclock: ROUNDS is 'stock' or unset" >&2; exit 2; }
RUN_DIR=${OVERCLOCK_RUN_DIR:-$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-overclock}
mkdir -p "$RUN_DIR"
SUM=$RUN_DIR/summary.md
sum() { echo "$*" | tee -a "$SUM"; }

# Each reader prints a whole number or fails, so that a gate can never pass on an unreadable value.
read_int() {
    local v
    v=$(cat "$1" 2>/dev/null) || return 1
    [[ $v =~ ^[0-9]+$ ]] || return 1
    echo "$v"
}
hot() {
    local v
    v=$(read_int "$HWMON/temp2_input") || return 1
    echo $((v / 1000))
}
cap_w() {
    local v
    v=$(read_int "$HWMON/power1_cap") || return 1
    echo $((v / 1000000))
}
mem_gb() {
    local v
    v=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null)
    [[ $v =~ ^[0-9]+$ ]] || return 1
    echo "$v"
}
profile_now() { timeout 20 lact cli profile get 2>/dev/null | tr -d '[:space:]'; }
card_state() {
    echo "profile $(profile_now); power1_cap $(cap_w || echo unreadable) W;" \
        "od: $(tr '\n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage" | tr -s ' ');" \
        "mclk levels: $(tr '\n' ' ' <"$DRM_DEVICE/pp_dpm_mclk" | tr -s ' '); hotspot $(hot || echo unreadable) C"
}

# Switch the profile and wait until the driver shows the expected power cap (the daemon applies settings itself).
set_profile() {
    local name=$1 want_cap=$2 t=0
    timeout 20 lact cli profile set "$name" >>"$RUN_DIR/lact.log" 2>&1 || return 1
    while [[ $(cap_w) != "$want_cap" ]]; do
        ((t >= 30)) && return 1
        sleep 1
        t=$((t + 1))
    done
    sleep 6
    [[ $(profile_now) == "$name" ]]
}

signals_on() {
    trap 'exit 130' INT TERM
    trap 'exit 129' HUP
}
# 1 = nothing to switch back. It becomes 0 only in the comparison form, once the start check has passed.
RESTORED=1
restore() {
    ((RESTORED)) && return
    # A signal must not cut the switch back short; lact and sleep inherit the ignore.
    trap '' INT TERM HUP
    RESTORED=1
    if set_profile Testing 317 || set_profile Testing 317; then
        sum "- restored: $(card_state)"
    else
        sum "- RESTORE FAILED: the card is NOT confirmed back on Testing: $(card_state)"
        exit 5
    fi
    signals_on
}
trap restore EXIT
signals_on

# The shared lock on fd 9. hold_lock keeps it across several jobs; job() takes it for one job when it is not held.
HELD=0
take_lock() {
    [[ -r $GPU_LOCK ]] || return 1
    exec 9<"$GPU_LOCK" || return 1
    flock -w 60 9 && return 0
    exec 9<&-
    return 1
}
drop_lock() {
    flock -u 9
    exec 9<&-
}
hold_lock() { take_lock && HELD=1; }
release_hold() {
    ((HELD)) || return 0
    HELD=0
    drop_lock
}

STOP=0
stop_run() {
    STOP=1
    sum "- STOPPED: $*"
}
# True when memory and hotspot allow a start. Unreadable values fail.
ready() {
    local m t
    m=$(mem_gb) && t=$(hot) && ((m >= 8 && t < 60))
}
# job <log> <command...>
job() {
    local log=$1 w=0 rc t m try
    shift
    ((STOP)) && { echo "SKIPPED (run stopped): $*" >>"$log"; return 9; }
    for try in 1 2 3; do
        m=$(mem_gb) || { stop_run "available memory unreadable before: $*"; return 5; }
        if ((m < 8)); then
            echo "NOT STARTED: below 8 GB available memory: $*" >>"$log"
            stop_run "below 8 GB available memory before: $*"
            return 5
        fi
        while :; do
            t=$(hot) || { stop_run "hotspot unreadable before: $*"; return 7; }
            ((t < 60)) && break
            if ((w >= 180)); then
                echo "NOT STARTED: hotspot still $t C after 3 minutes: $*" >>"$log"
                stop_run "hotspot still $t C after 3 minutes before: $*"
                return 7
            fi
            sleep 2
            w=$((w + 2))
        done
        # With the lock already held for the round, the checks above were made under it.
        ((HELD)) && break
        if ! take_lock; then
            echo "NOT STARTED: the shared lock was not free within 60 s: $*" >>"$log"
            stop_run "the shared lock was not free within 60 s (or not readable) before: $*"
            return 6
        fi
        ready && break
        # Memory or hotspot changed while waiting for the lock: give the lock back and check again.
        drop_lock
        if ((try == 3)); then
            echo "NOT STARTED: memory or hotspot not in range with the lock held, three tries: $*" >>"$log"
            stop_run "memory or hotspot not in range with the lock held before: $*"
            return 8
        fi
    done
    {
        echo "## $(date -u +%FT%TZ) $*"
        echo "before: $(card_state)"
    } >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 240 "$@" >>"$log" 2>&1 9<&-
    rc=$?
    ((HELD)) || drop_lock
    if t=$(hot); then
        echo "after: exit $rc; hotspot $t C" >>"$log"
        ((t >= 95)) && stop_run "hotspot $t C after: $*"
    else
        echo "after: exit $rc; hotspot unreadable" >>"$log"
        stop_run "hotspot unreadable after: $*"
    fi
    return "$rc"
}

round() {
    local tag=$1 name=$2 cap=$3 log mode
    if [[ $(profile_now) != "$name" ]]; then
        if [[ $MODE == stock ]]; then
            stop_run "$tag: the profile is '$(profile_now)', not $name; the stock-only form does not switch profiles"
            return 1
        fi
        set_profile "$name" "$cap" || { stop_run "$tag: could not switch to $name"; return 1; }
    fi
    card_state >"$RUN_DIR/profile-$tag.txt"
    sum "- $tag: $(cat "$RUN_DIR/profile-$tag.txt")"
    for mode in reg:f16 reg:fp8 fma stream; do
        log=$RUN_DIR/$tag-mat_card-${mode/:/_}.log
        job "$log" "$MAT" "$mode" --short
        sum "  - $tag mat_card $mode: $(grep -h -E '^RESULT .*(highest rate|W=1 |GFMA_s|GBps)' "$log" 2>/dev/null | head -2 | cut -c1-330 | tr '\n' ' ')"
    done
    log=$RUN_DIR/$tag-bandwidth.log
    for _ in 1 2 3; do
        job "$log" "$BW" 1024 50
    done
    sum "  - $tag bandwidth (GB/s read, 3 rounds): $(awk '$1 == "RESULT" && $2 == "read" {printf "%s ", $8}' "$log" 2>/dev/null)"
}

sum "# Overclock comparison, $(date -u +%FT%TZ); commit $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (with uncommitted changes)')"
if [[ $(profile_now) != Testing ]]; then
    sum "- the profile at the start is $(profile_now), not Testing; nothing run"
    exit 3
fi
if [[ $MODE == stock ]]; then
    round S1-testing Testing 317
    ((STOP)) || round S2-testing Testing 317
else
    RESTORED=0
    round A-testing Testing 317
    if ! ((STOP)); then
        if hold_lock; then
            round B-overclock Default 374
        else
            stop_run "the shared lock was not free within 60 s (or not readable) before the switch to Default"
        fi
    fi
    restore
    release_hold
    ((STOP)) || round C-testing Testing 317
fi
sum "- end: $(card_state)"
[[ $(profile_now) == Testing && $(cap_w) == 317 ]] || exit 4
((STOP)) && exit 1
exit 0
