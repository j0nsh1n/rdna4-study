#!/usr/bin/env bash
# Lane 4 of notes/STUDY-PLAN.md: build and run the matrix-timing probes (src/study/mat_*.hip and the first-hour
# src/wmma_time.hip), one command.
#
# Usage: scripts/study-matrix-timing.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles every probe for gfx1201 into build/study/bin/<probe> (-save-temps output in
#           build/study/tmp-<probe>/), checks that its timed loops hold only the assembly blocks and the loop control
#           (scripts/core_loopcheck.py, where the loops are compiler-built around assembly blocks) and that they hold
#           the intended instructions (scripts/mem_loopcheck.py with scripts/mat-expect/<probe>.txt)
#   run     runs every probe mode as its own GPU job
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env, written by scripts/first-hour.sh info).
# Results: runs/<UTC time>-matrix/: one log per job, clock-log.csv (once a second: sclk, mclk, reported power,
# hotspot), summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, device index 0 from the card record, not started below 8 GB
# available memory. Before each job the LACT profile (must be "Testing"), pp_od_clk_voltage and power1_cap are
# logged and compared with the start of the run (a change stops all timed work); sclk, mclk, reported power and
# hotspot are logged before and after. Thermal rule for whole-card jobs: start only below 70 C hotspot and at least
# 20 s after the previous whole-card job ended; each is kept to about 20 s. A hotspot of 95 C or more after any job
# stops all further GPU work.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-matrix-timing.sh"
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
LACT_CONFIG=/etc/lact/config.yaml
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
    # Memory and hotspot were checked before waiting for the lock; after a long wait they may have changed.
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
    # A probe that hangs must not hold the card and the shared lock.
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-300}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-300} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-matrix-timing: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
export DRM_DEVICE HWMON

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-matrix"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/study/bin
FAILED=0
HOTTEST=0
LAST_FULL_END=0
STOPPED=0
SAMPLER=

sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

hotspot_c() {
    echo $(($(cat "$HWMON/temp2_input") / 1000))
}

lact_profile() {
    awk '/^current_profile:/ {print $2}' "$LACT_CONFIG" 2>/dev/null
}

card_config() {
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}
CONFIG0=$(card_config)

# name | source under src/ | extra compiler flags | 1 = loops are compiler-built around assembly blocks
# (core_loopcheck applies), 0 = the timed loop is inside one assembly statement or compiler-built from builtins
PROBES=(
    "wmma_time|wmma_time.hip|-mllvm -amdgpu-enable-vopd=0|0"
    "mat_forms|study/mat_forms.hip||1"
    "mat_pipe|study/mat_pipe.hip||1"
    "mat_share_cu|study/mat_share.hip|-DCUMODE=1 -mcumode|0"
    "mat_share_wgp|study/mat_share.hip|-DCUMODE=0|0"
    "mat_card|study/mat_card.hip|-mcumode|0"
)
# probe | arguments | 1 = whole-card job (thermal rule), 0 = one wave or one CU
JOBS=(
    "wmma_time||0"
    "mat_forms|forms|0"
    "mat_forms|index|0"
    "mat_forms|mods|0"
    "mat_forms|data|0"
    "mat_pipe|lat|0"
    "mat_pipe|hazard|0"
    "mat_pipe|overlap|0"
    "mat_share_cu||0"
    "mat_share_wgp||0"
    "mat_card|reg:f16|1"
    "mat_card|reg:bf16|1"
    "mat_card|reg:fp8|1"
    "mat_card|reg:iu8|1"
    "mat_card|reg:iu4|1"
    "mat_card|reg:iu4k32|1"
    "mat_card|reg:sf16|1"
    "mat_card|reg:sfp8|1"
    "mat_card|reg:siu8|1"
    "mat_card|fma|1"
    "mat_card|stream|1"
    "mat_card|mem:f16|1"
    "mat_card|mem:fp8|1"
)
BREAKS=(
    "wmma_time|--expect f16=8|0"
    "mat_forms|forms --break|0"
    "mat_pipe|lat --break|0"
    "mat_share_cu|--break|0"
    "mat_card|reg:f16 --break|1"
)

build_one() {
    local name=$1 src=$2 flags=$3 dir log
    dir=$REPO/build/study/tmp-$name
    log=$RUN_DIR/build-$name.log
    mkdir -p "$dir" "$BIN"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags -save-temps \
        "$REPO/src/$src" -o "$BIN/$name" -lamdhip64) >"$log" 2>&1
}

phase_build() {
    local p name src flags core rc lc expect running
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'), $(command -v hipcc))"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags core <<<"$p"
        if [[ ! -f $REPO/src/$src ]]; then
            sum "- $name: source src/$src missing; FAILED"
            FAILED=1
            continue
        fi
        # at most 8 cores: the six builds run 4 at a time, each compiler single-threaded
        while running=$(jobs -rp | wc -l) && ((running >= 4)); do
            wait -n
        done
        build_one "$name" "$src" "$flags" &
    done
    wait
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags core <<<"$p"
        [[ -f $REPO/src/$src ]] || continue
        if [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
            continue
        fi
        lc=""
        if ((core)); then
            python3 "$REPO/scripts/core_loopcheck.py" "$REPO/build/study/tmp-$name"/*.lto.s \
                >"$RUN_DIR/corecheck-$name.txt" 2>&1
            rc=$?
            lc="$(tail -1 "$RUN_DIR/corecheck-$name.txt"); "
            if ((rc != 0)); then
                sum "- $name: built; ${lc}NOT CLEAN (see corecheck-$name.txt)"
                FAILED=1
                continue
            fi
        fi
        expect=$REPO/scripts/mat-expect/$name.txt
        [[ $name == mat_share_* ]] && expect=$REPO/scripts/mat-expect/mat_share.txt
        if [[ ! -f $expect ]]; then
            sum "- $name: built; no loop expectations ($expect missing); FAILED"
            FAILED=1
            continue
        fi
        python3 "$REPO/scripts/mem_loopcheck.py" "$REPO/build/study/tmp-$name"/*.lto.s "$expect" \
            >"$RUN_DIR/loopcheck-$name.txt" 2>&1
        rc=$?
        lc="$lc$(tail -1 "$RUN_DIR/loopcheck-$name.txt")"
        if ((rc == 0)); then
            sum "- $name: built; $lc"
        else
            sum "- $name: built; $lc; LOOP CHECK FAILED (see loopcheck-$name.txt)"
            FAILED=1
        fi
    done
}

# Once-a-second clock log for the whole run (read-only sysfs).
start_sampler() {
    local csv=$RUN_DIR/clock-log.csv
    echo "utc,sclk_mhz,mclk_mhz,power_w,hotspot_c,edge_c" >"$csv"
    (
        while :; do
            echo "$(date -u +%FT%TZ),$(grep '\*' "$DRM_DEVICE/pp_dpm_sclk" | grep -o '[0-9]*Mhz' | tr -d Mhz)," \
                "$(grep '\*' "$DRM_DEVICE/pp_dpm_mclk" | grep -o '[0-9]*Mhz' | tr -d Mhz),$(($(cat "$HWMON/power1_average") / 1000000))," \
                "$(($(cat "$HWMON/temp2_input") / 1000)),$(($(cat "$HWMON/temp1_input") / 1000))" | tr -d ' ' >>"$csv"
            sleep 1
        done
    ) &
    SAMPLER=$!
}

stop_sampler() {
    # only the sampler this script started, by its PID
    if [[ -n $SAMPLER ]]; then
        kill "$SAMPLER" 2>/dev/null
        wait "$SAMPLER" 2>/dev/null
        SAMPLER=
    fi
}
trap stop_sampler EXIT

# gpu_job <log> <full> <command...>
gpu_job() {
    local log=$1 full=$2 now cfg t
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((STOPPED)); then
        echo "NOT STARTED: timed work was stopped earlier in this run" >>"$log"
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
    if [[ $(lact_profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)', not Testing" >>"$log"
        sum "- STOPPED: the LACT profile is '$(lact_profile)', not Testing, before $*; no further timed jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
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
    t=$(grep -h "^THERMAL highest" "$log" | tail -1 | grep -o '[0-9]* C' | tr -d ' C')
    [[ $t =~ ^[0-9]+$ ]] && ((t > HOTTEST)) && HOTTEST=$t
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    if ((HOTTEST >= 95)); then
        sum "- STOPPED: hotspot reached $HOTTEST C during or after $*; no further GPU jobs"
        STOPPED=1
        FAILED=1
    fi
    ((full)) && LAST_FULL_END=$(date +%s)
    return "$rc"
}

log_name() {
    local name=$1 args=$2
    args=${args//:/_}
    args=${args//=/_}
    args=${args// /_}
    args=${args//--/}
    echo "$RUN_DIR/$name${args:+-$args}.log"
}

phase_run() {
    local j name args full log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration at start: $CONFIG0)"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name args full <<<"$j"
        log=$(log_name "$name" "$args")
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name $args: not built"
            FAILED=1
            continue
        fi
        # shellcheck disable=SC2086 # args is a word list on purpose
        gpu_job "$log" "$full" "$BIN/$name" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $name $args: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        else
            sum "- $name $args: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j name args full log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS='|' read -r name args full <<<"$j"
        log=$(log_name "$name" "$args-break")
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name $args: not built"
            FAILED=1
            continue
        fi
        # shellcheck disable=SC2086 # args is a word list on purpose
        gpu_job "$log" "$full" "$BIN/$name" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [^ ]*: FAIL" "$log"; then
            sum "- $name $args: FAIL as intended. $v"
        else
            sum "- $name $args: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

sum "# Matrix-timing study run, $(date -u +%FT%TZ)"
sum "- commit $(git -C "$REPO" rev-parse --short HEAD)$([[ -z $(git -C "$REPO" status --porcelain -- src scripts) ]] || echo ' (with uncommitted changes)'); card $ARCH, PCI $PCI_BUS; kernel $(uname -r)"
sum "- LACT profile $(lact_profile); power1_cap $(($(cat "$HWMON/power1_cap") / 1000000)) W (a 1.2x regulator-level power-limit mod is in the card: reported power is scaled down)"
case ${1:-all} in
    build) phase_build ;;
    run | breaks | all)
        [[ ${1:-all} == all ]] && phase_build
        if [[ $(lact_profile) != Testing ]]; then
            sum "- NOT RUN: LACT profile is '$(lact_profile)', not Testing"
            FAILED=1
        else
            start_sampler
            [[ ${1:-all} != breaks ]] && phase_run
            [[ ${1:-all} != run ]] && phase_breaks
            stop_sampler
        fi
        ;;
    *) die "unknown phase ${1:-}; use build, run, breaks or all" ;;
esac
sum "- highest hotspot seen: $HOTTEST C"
echo "study-matrix-timing: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
