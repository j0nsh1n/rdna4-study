#!/usr/bin/env bash
# Lane 2 of notes/STUDY-PLAN.md: build and run the memory-system probes (src/study/mem_*.hip), one command.
#
# Usage: scripts/study-memory.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles every probe for gfx1201 into build/study/bin/<probe> (-save-temps output in
#           build/study/tmp-<probe>/) and checks its loops against scripts/mem-expect/<probe>.txt
#           (scripts/mem_loopcheck.py: which loads and waits, how many, no spills)
#   run     runs every probe mode as its own GPU job
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env, written by scripts/first-hour.sh info).
# Results: runs/<UTC time>-memory/: one log per job, summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, the card record's device index, not started below 8 GB available memory. Before
# each job: pp_od_clk_voltage and power1_cap are logged and compared with the values at the start of the run (a
# change stops all timed work); sclk, mclk, reported power and hotspot are logged before and after. Thermal rule for
# whole-card jobs: start only below 70 C hotspot and at least 20 s after the previous whole-card job ended (each
# whole-card job is kept to about 20 s; the programs also pause between configurations above 80 C).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-memory.sh"
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1

snap() {
    local drm=$1 hw=$2
    echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')" \
        " power: $(($(cat "$hw/power1_average") / 1000000)) W  hotspot: $(($(cat "$hw/temp2_input") / 1000)) C"
}

# Locked runner: re-entered under flock, so the clocks are read while this job holds the GPU.
if [[ ${1:-} == __locked ]]; then
    log=$2
    drm=$3
    hw=$4
    shift 4
    # Memory and hotspot were checked before waiting for the lock; after a long wait they may have changed, so
    # both are checked again here and the job is not run if either is still bad. The card configuration is
    # compared before the lock wait only; it is logged again in the "clocks" lines.
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    if [[ ${FULL_JOB:-0} == 1 ]]; then
        w=0
        while (($(cat "$hw/temp2_input") >= 70000)); do
            if ((w >= 120)); then
                echo "NOT STARTED: hotspot still $(($(cat "$hw/temp2_input") / 1000)) C two minutes after taking the lock" >>"$log"
                exit 7
            fi
            sleep 2
            w=$((w + 2))
        done
    fi
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    # A probe that hangs must not hold the card and the shared lock; no job here needs more than a few minutes.
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-600}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-600} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-memory: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
export DRM_DEVICE HWMON

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-memory"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/study/bin
FAILED=0
HOTTEST=0
LAST_FULL_END=0
STOPPED=0

sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

hotspot_c() {
    echo $(($(cat "$HWMON/temp2_input") / 1000))
}

card_config() {
    echo "power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}
CONFIG0=$(card_config)

# name | source | extra compiler flags. mem_inflight is built in CU mode so a work-group sits on one CU.
PROBES=(
    "mem_stream|mem_stream.hip|"
    "mem_latency|mem_latency.hip|"
    "mem_inflight|mem_inflight.hip|-mcumode"
    "mem_occupancy|mem_occupancy.hip|"
    "mem_store|mem_store.hip|"
)
# probe | mode | 1 = whole-card job (thermal rule), 0 = one wave or one CU
JOBS=(
    "mem_stream|calib|1"
    "mem_stream|size:pass1|1"
    "mem_stream|size:win|1"
    "mem_stream|size:split|1"
    "mem_stream|size:shared|1"
    "mem_stream|size:private|1"
    "mem_stream|trap|1"
    "mem_stream|shape|1"
    "mem_stream|hints|1"
    "mem_latency|chase|0"
    "mem_latency|scope|0"
    "mem_latency|persist|0"
    "mem_latency|hints|1"
    "mem_latency|hints_l2|0"
    "mem_latency|prefetch|0"
    "mem_inflight|inflight|0"
    "mem_inflight|waits|0"
    "mem_occupancy|rule|1"
    "mem_occupancy|bw|1"
    "mem_store|store|1"
    "mem_store|copy|1"
    "mem_store|cus|1"
)
BREAKS=(
    "mem_stream|calib|1"
    "mem_latency|chase|0"
    "mem_inflight|waits|0"
    "mem_occupancy|rule|1"
    "mem_store|copy|1"
)

build_one() {
    local name=$1 src=$2 flags=$3 dir log
    dir=$REPO/build/study/tmp-$name
    log=$RUN_DIR/build-$name.log
    mkdir -p "$dir" "$BIN"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags -save-temps \
        "$REPO/src/study/$src" -o "$BIN/$name" -lamdhip64) >"$log" 2>&1
}

phase_build() {
    local p name src flags pids=() rc lc expect
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'), $(command -v hipcc))"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        if [[ ! -f $REPO/src/study/$src ]]; then
            sum "- $name: source src/study/$src missing; FAILED"
            FAILED=1
            continue
        fi
        build_one "$name" "$src" "$flags" &
        pids+=($!)
    done
    wait
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        [[ -f $REPO/src/study/$src ]] || continue
        if [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
            continue
        fi
        expect=$REPO/scripts/mem-expect/$name.txt
        if [[ ! -f $expect ]]; then
            sum "- $name: built; no loop expectations (scripts/mem-expect/$name.txt missing); FAILED"
            FAILED=1
            continue
        fi
        python3 "$REPO/scripts/mem_loopcheck.py" "$REPO/build/study/tmp-$name"/*.lto.s "$expect" \
            >"$RUN_DIR/loopcheck-$name.txt" 2>&1
        rc=$?
        lc=$(tail -1 "$RUN_DIR/loopcheck-$name.txt")
        if ((rc == 0)); then
            sum "- $name: built; $lc"
        else
            sum "- $name: built; $lc; LOOP CHECK FAILED (see loopcheck-$name.txt)"
            FAILED=1
        fi
    done
}

# gpu_job <log> <full> <command...>
gpu_job() {
    local log=$1 full=$2 now cfg t
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((STOPPED)); then
        echo "NOT STARTED: the card configuration changed earlier in this run" >>"$log"
        return 6
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if [[ $HIP_INDEX != 0 ]]; then
        echo "NOT STARTED: device index is $HIP_INDEX; this study only runs on device 0 (the discrete card)" >>"$log"
        return 8
    fi
    cfg=$(card_config)
    echo "card configuration: $cfg" >>"$log"
    if [[ $cfg != "$CONFIG0" ]]; then
        echo "NOT STARTED: card configuration changed since the start of the run (was: $CONFIG0)" >>"$log"
        sum "- STOPPED: the card configuration changed before $*; no further timed jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
    if ((full)); then
        now=$(date +%s)
        if ((now - LAST_FULL_END < 20)); then
            sleep $((20 - (now - LAST_FULL_END)))
        fi
        local waited=0
        while (($(hotspot_c) >= 70)); do
            if ((waited >= 600)); then
                echo "NOT STARTED: hotspot still $(hotspot_c) C after waiting 10 minutes" >>"$log"
                return 7
            fi
            sleep 2
            waited=$((waited + 2))
        done
    fi
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    FULL_JOB=$full HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
    local rc=$?
    t=$(grep -h "^THERMAL highest" "$log" | tail -1 | awk '{print $(NF-1)}')
    [[ $t =~ ^[0-9]+$ ]] && ((t > HOTTEST)) && HOTTEST=$t
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    ((full)) && LAST_FULL_END=$(date +%s)
    return "$rc"
}

phase_run() {
    local j name mode full log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration at start: $CONFIG0)"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name mode full <<<"$j"
        log=$RUN_DIR/$name-${mode//:/_}.log
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name $mode: not built"
            FAILED=1
            continue
        fi
        gpu_job "$log" "$full" "$BIN/$name" "$mode"
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $name $mode: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        else
            sum "- $name $mode: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j name mode full log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS='|' read -r name mode full <<<"$j"
        log=$RUN_DIR/$name-${mode//:/_}-break.log
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name $mode --break: not built"
            FAILED=1
            continue
        fi
        gpu_job "$log" "$full" "$BIN/$name" "$mode" --break
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [a-z_]*: FAIL" "$log"; then
            sum "- $name $mode --break: FAIL as intended. $v"
        else
            sum "- $name $mode --break: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

sum "# Memory-system study run, $(date -u +%FT%TZ)"
sum "- commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (with uncommitted changes)'); card $ARCH, PCI $PCI_BUS; kernel $(uname -r)"
case ${1:-all} in
    build) phase_build ;;
    run) phase_run ;;
    breaks) phase_breaks ;;
    all)
        phase_build
        phase_run
        phase_breaks
        ;;
    *) die "unknown phase ${1:-}; use build, run, breaks or all" ;;
esac
sum "- highest hotspot seen: $HOTTEST C"
echo "study-memory: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
