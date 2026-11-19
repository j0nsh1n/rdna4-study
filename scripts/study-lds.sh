#!/usr/bin/env bash
# Lane 3 of notes/STUDY-PLAN.md: build and run the LDS, scalar-load and special-load probes (src/study/lds_*.hip).
#
# Usage: scripts/study-lds.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles every probe for gfx1201 into build/lds/<probe>/ (-save-temps listing there) and checks its loops
#           against scripts/lds-expect/<source>.txt with scripts/mem_loopcheck.py (taken from STUDY_REPO)
#   run     runs every probe mode as its own GPU job
#   breaks  runs each deliberate break; each must report FAIL
# Environment: STUDY_REPO (default: this script's root) supplies src/harness.h, src/study/{core,mem}_common.h,
# scripts/{core,mem}_loopcheck.py and runs/latest/card.env. ROCM_PATH (default ~/.local/opt/rocm).
# Results: runs/<UTC time>-lds/: one log per job, the configuration record, summary.md. Exit 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, device 0 only, not started below 8 GB available memory. Before each
# job the LACT profile, power cap and clock tables are recorded; if the profile is not "Testing" the wall-clock
# (whole-card) jobs are skipped. Thermal rule: every job starts below 65 C hotspot; a reading of 95 C stops the run.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-lds.sh"
STUDY_REPO=${STUDY_REPO:-$ROOT}
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
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

# Locked runner: re-entered under flock so the clocks are read while this job holds the GPU.
if [[ ${1:-} == __locked ]]; then
    log=$2
    drm=$3
    hw=$4
    shift 4
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    w=0
    while (($(cat "$hw/temp2_input") >= 65000)); do
        if ((w >= 120)); then
            echo "NOT STARTED: hotspot still $(($(cat "$hw/temp2_input") / 1000)) C two minutes after taking the lock" >>"$log"
            exit 7
        fi
        sleep 2
        w=$((w + 2))
    done
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-180}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-180} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-lds: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$STUDY_REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV (STUDY_REPO=$STUDY_REPO)"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
[[ ${HIP_INDEX:-} == 0 ]] || die "card record says HIP_INDEX=${HIP_INDEX:-unset}; this study runs on device 0 only"
export DRM_DEVICE HWMON

RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-lds"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BUILD=$ROOT/build/lds
FAILED=0
HOTTEST=0
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

profile() {
    awk '/^current_profile:/ {print $2}' /etc/lact/config.yaml 2>/dev/null | tr -d '"'
}

card_config() {
    echo "profile=$(profile) power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}
CONFIG0=$(card_config)

# probe binary | source | extra compiler flags | expectation file
PROBES=(
    "lds_probe|lds_probe.hip||lds_probe"
    "lds_probe_cu|lds_probe.hip|-mcumode -DUSE_CU_KEY=1|lds_probe"
    "lds_tr|lds_tr.hip||lds_tr"
    "lds_stage|lds_stage.hip||lds_stage"
    "lds_stage_cu|lds_stage.hip|-mcumode -DUSE_CU_KEY=1|lds_stage"
    "lds_smem|lds_smem.hip||lds_smem"
    "lds_block|lds_block.hip||lds_block"
    "lds_dvgpr|lds_dvgpr.hip||lds_dvgpr"
)
# probe | mode | 1 = whole-card wall-clock job, 0 = one wave or one work-group (cycle counter)
JOBS=(
    "lds_probe|calib|0"
    "lds_probe|width|0"
    "lds_probe|chain|0"
    "lds_probe|inflight|0"
    "lds_probe|pattern|0"
    "lds_probe|waves|0"
    "lds_probe_cu|waves|0"
    "lds_probe|size|0"
    "lds_probe_cu|size|0"
    "lds_probe|card|1"
    "lds_probe_cu|card|1"
    "lds_tr|layout|0"
    "lds_tr|wmma|0"
    "lds_tr|inflight|0"
    "lds_tr|bw|1"
    "lds_stage|feed|0"
    "lds_stage_cu|feed|0"
    "lds_smem|calib|0"
    "lds_smem|chase|0"
    "lds_smem|width|0"
    "lds_smem|mixed|0"
    "lds_smem|scratch|0"
    "lds_block|calib|0"
    "lds_block|layout|0"
    "lds_block|cost|0"
    "lds_dvgpr|probe|0"
)
# The three breaks the brief requires come first: a wrong expected calibration, a transposing-load layout with two
# lanes swapped, an LDS pattern that should collide but does not.
BREAKS=(
    "lds_probe|calib|0"
    "lds_tr|layout|0"
    "lds_probe|pattern|0"
    "lds_tr|wmma|0"
    "lds_stage|feed|0"
    "lds_smem|calib|0"
    "lds_block|layout|0"
    "lds_dvgpr|probe|0"
)

build_one() {
    local name=$1 src=$2 flags=$3 dir=$BUILD/$1
    mkdir -p "$dir"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags -save-temps \
        -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -I"$ROOT/src/study" "$ROOT/src/study/$src" -o "$dir/$name" -lamdhip64) \
        >"$RUN_DIR/build-$name.log" 2>&1
}

phase_build() {
    local p name src flags exp rc lc n=0
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'), $(command -v hipcc)); STUDY_REPO=$STUDY_REPO"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags exp <<<"$p"
        [[ -f $ROOT/src/study/$src ]] || continue
        build_one "$name" "$src" "$flags" &
        n=$((n + 1))
        # at most 4 compiles at once (8-core limit for CPU work)
        ((n % 4 == 0)) && wait
    done
    wait
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags exp <<<"$p"
        if [[ ! -f $ROOT/src/study/$src ]]; then
            sum "- $name: source src/study/$src missing; FAILED"
            FAILED=1
            continue
        fi
        if [[ ! -x $BUILD/$name/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
            continue
        fi
        python3 "$STUDY_REPO/scripts/mem_loopcheck.py" "$BUILD/$name"/*.lto.s "$ROOT/scripts/lds-expect/$exp.txt" \
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
    local log=$1 full=$2 cfg t waited=0
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((STOPPED)); then
        echo "NOT STARTED: the run was stopped earlier (configuration change or 95 C)" >>"$log"
        return 6
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    cfg=$(card_config)
    {
        echo "card configuration: $cfg"
        echo "pp_dpm_sclk: $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_dpm_sclk")"
        echo "pp_dpm_mclk: $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_dpm_mclk")"
    } >>"$log"
    if [[ $cfg != "$CONFIG0" ]]; then
        echo "NOT STARTED: card configuration changed since the start of the run (was: $CONFIG0)" >>"$log"
        sum "- STOPPED: the card configuration changed before $*; no further jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
    if ((full)) && [[ $(profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(profile)', not 'Testing'; wall-clock jobs are skipped" >>"$log"
        return 9
    fi
    while (($(hotspot_c) >= 65)); do
        if ((waited >= 600)); then
            echo "NOT STARTED: hotspot still $(hotspot_c) C after waiting 10 minutes" >>"$log"
            return 7
        fi
        sleep 2
        waited=$((waited + 2))
    done
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    FULL_JOB=$full HIP_VISIBLE_DEVICES=0 CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
    local rc=$?
    t=$(grep -h "^THERMAL highest" "$log" | tail -1 | awk '{print $(NF-1)}')
    [[ $t =~ ^[0-9]+$ ]] && ((t > HOTTEST)) && HOTTEST=$t
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    if ((HOTTEST >= 95)); then
        sum "- STOPPED: hotspot reached $HOTTEST C"
        STOPPED=1
        FAILED=1
    fi
    return "$rc"
}

run_list() {  # run_list <break 0|1> <entries...>
    local brk=$1 j name mode full log rc v
    shift
    for j in "$@"; do
        IFS='|' read -r name mode full <<<"$j"
        if ((brk)); then
            log=$RUN_DIR/$name-$mode-break.log
        else
            log=$RUN_DIR/$name-$mode.log
        fi
        if [[ ! -x $BUILD/$name/$name ]]; then
            sum "- $name $mode: not built"
            FAILED=1
            continue
        fi
        if ((brk)); then
            gpu_job "$log" "$full" "$BUILD/$name/$name" "$mode" --break
        else
            gpu_job "$log" "$full" "$BUILD/$name/$name" "$mode"
        fi
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((brk)); then
            if ((rc == 1)) && grep -q "^VERDICT [a-z0-9_]*: FAIL" "$log"; then
                sum "- $name $mode --break: FAIL as intended. $v"
            else
                sum "- $name $mode --break: did NOT fail (exit $rc). ${v:-no verdict}"
                FAILED=1
            fi
        elif ((rc == 0)); then
            sum "- $name $mode: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        elif ((rc == 9)); then
            sum "- $name $mode: skipped (profile not Testing)"
        else
            sum "- $name $mode: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

sum "# LDS and special-load study run (lane 3), $(date -u +%FT%TZ)"
sum "- card $ARCH, PCI $PCI_BUS; kernel $(uname -r); configuration at start: $CONFIG0"
sum "- profile 'Testing' = stock clocks, 317 W cap; a regulator-level power-limit mod (1.2x) is in the card, so the reported power is scaled down"
case ${1:-all} in
    build) phase_build ;;
    run)
        sum "## run"
        run_list 0 "${JOBS[@]}"
        ;;
    breaks)
        sum "## deliberate breaks (each must FAIL)"
        run_list 1 "${BREAKS[@]}"
        ;;
    all)
        phase_build
        sum "## run"
        run_list 0 "${JOBS[@]}"
        sum "## deliberate breaks (each must FAIL)"
        run_list 1 "${BREAKS[@]}"
        ;;
    *) die "unknown phase ${1:-}; use build, run, breaks or all" ;;
esac
sum "- highest hotspot seen: $HOTTEST C"
echo "study-lds: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
