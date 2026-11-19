#!/usr/bin/env bash
# Leftovers pass, matrix unit (notes/STUDY-PLAN.md): build and run src/study/left_mat.hip, left_prio.hip, left_rot.hip,
# left_dot.hip.
#
# Usage: scripts/study-leftovers-matrix.sh [build|run|breaks|all] [MODE...]   (default all = build, run, breaks)
#   build   compiles left_mat for gfx1201 into build/left/bin (with -save-temps into build/left/tmp, for the listing),
#           and left_prio (CU mode) with its two code objects: left_prio.hsaco and left_prio_rr.hsaco, the second
#           assembled from the same listing with the round-robin bit set in every kernel descriptor (checked);
#           left_rot (single-wave cycle counts, with -save-temps into build/left/rot) and left_dot (dot instructions)
#   run     every mode as its own GPU job (single waves or one work-group); with MODE arguments, only those modes
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env, written by scripts/first-hour.sh info).
# Results: runs/<UTC time>-leftmat/: one log per job and summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, device index 0 from the card record, not started below 8 GB
# available memory; the LACT profile and the clocks are logged before and after each job. Results here are bits,
# not times, so a profile other than "Testing" is recorded, not fatal. CPU jobs use at most 8 threads, under nice.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-leftovers-matrix.sh"
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
LACT_CONFIG=/etc/lact/config.yaml
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH

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
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-600}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-600} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-leftovers-matrix: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-leftmat"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/left/bin
FAILED=0


sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

lact_profile() {
    awk '/^current_profile:/ {print $2}' "$LACT_CONFIG" 2>/dev/null
}

card_config() {
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}



sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

lact_profile() {
    awk '/^current_profile:/ {print $2}' "$LACT_CONFIG" 2>/dev/null
}

card_config() {
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}

# program | arguments, one job each (the first word of the arguments is the mode name for MODE selection)
JOBS=(
    "left_mat|s4map"
    "left_mat|s4"
    "left_mat|mask"
    "left_mat|mods --n 1000000"
    "left_mat|hazard"
    "left_prio|prio build/left/bin/left_prio.hsaco"
    "left_prio|prio build/left/bin/left_prio_rr.hsaco --rr"
    "left_rot|rot"
    "left_dot|nan"
    "left_dot|int"
    "left_dot|mods"
)
BREAKS=(
    "left_mat|s4map --break"
    "left_mat|s4 --n 50000 --break"
    "left_mat|mask --break"
    "left_mat|mods --n 100000 --break"
    "left_mat|hazard --break"
    "left_prio|prio build/left/bin/left_prio.hsaco --break"
    "left_prio|prio build/left/bin/left_prio_rr.hsaco --rr --break"
    "left_rot|rot --break"
    "left_dot|nan --break"
    "left_dot|int --break"
    "left_dot|mods --break"
)

phase_build() {
    local log=$RUN_DIR/build-left_mat.log v n llvm=$ROCM_PATH/llvm/bin pdir=$REPO/build/left/prio
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN" "$REPO/build/left/tmp" "$pdir"
    if (cd "$REPO/build/left/tmp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -save-temps \
        -DHARNESS_ARCH='"gfx1201"' "$REPO/src/study/left_mat.hip" -o "$BIN/left_mat" -lamdhip64) >"$log" 2>&1 &&
        [[ -x $BIN/left_mat ]]; then
        sum "- left_mat: built"
    else
        sum "- left_mat: build FAILED (see build-left_mat.log)"
        FAILED=1
    fi
    log=$RUN_DIR/build-left_prio.log
    if (cd "$pdir" &&
        nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -DHARNESS_ARCH='"gfx1201"' \
            "$REPO/src/study/left_prio.hip" -o "$BIN/left_prio" -lamdhip64 &&
        nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -DHARNESS_ARCH='"gfx1201"' \
            -DLEFT_PRIO_DEVICE_ONLY --cuda-device-only -S "$REPO/src/study/left_prio.hip" -o left_prio.s &&
        sed 's/\.amdhsa_round_robin_scheduling 0/.amdhsa_round_robin_scheduling 1/' left_prio.s >left_prio_rr.s &&
        for v in left_prio left_prio_rr; do
            "$llvm/clang" --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -c "$v.s" -o "$v.o" &&
                "$llvm/ld.lld" -shared "$v.o" -o "$BIN/$v.hsaco" || exit 1
        done) >"$log" 2>&1; then
        # the descriptors as the disassembler reads them: 5 kernels, the bit clear in one object and set in the other
        n=$("$llvm/llvm-objdump" -D --mcpu=gfx1201 -j .rodata "$BIN/left_prio_rr.hsaco" | grep -c 'round_robin_scheduling 1')
        v=$("$llvm/llvm-objdump" -D --mcpu=gfx1201 -j .rodata "$BIN/left_prio.hsaco" | grep -c 'round_robin_scheduling 0')
        if [[ $n == 5 && $v == 5 ]]; then
            sum "- left_prio: built; round-robin bit set in 5 of 5 descriptors of left_prio_rr.hsaco, clear in 5 of 5 of left_prio.hsaco"
        else
            sum "- left_prio: descriptor check FAILED (set $n, clear $v of 5)"
            FAILED=1
        fi
    else
        sum "- left_prio: build FAILED (see build-left_prio.log)"
        FAILED=1
    fi
    log=$RUN_DIR/build-left_rot.log
    mkdir -p "$REPO/build/left/rot"
    if (cd "$REPO/build/left/rot" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -save-temps \
        -DHARNESS_ARCH='"gfx1201"' "$REPO/src/study/left_rot.hip" -o "$BIN/left_rot" -lamdhip64) >"$log" 2>&1; then
        # 32 variants x (32 + 64) matrix instructions, nothing else added by the compiler inside the timed loops
        n=$(grep -c 'v_wmma' "$REPO"/build/left/rot/*.lto.s)
        if [[ $n == 3072 ]]; then
            sum "- left_rot: built; 3072 matrix instructions in the listing as written"
        else
            sum "- left_rot: listing check FAILED ($n matrix instructions, 3072 expected)"
            FAILED=1
        fi
    else
        sum "- left_rot: build FAILED (see build-left_rot.log)"
        FAILED=1
    fi
    log=$RUN_DIR/build-left_dot.log
    mkdir -p "$REPO/build/left/dot"
    if (cd "$REPO/build/left/dot" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -save-temps -pthread \
        -DHARNESS_ARCH='"gfx1201"' "$REPO/src/study/left_dot.hip" -o "$BIN/left_dot" -lamdhip64) >"$log" 2>&1; then
        sum "- left_dot: built"
    else
        sum "- left_dot: build FAILED (see build-left_dot.log)"
        FAILED=1
    fi
}

# job <log> <command...>: a GPU job under the shared lock
job() {
    local log=$1
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if [[ ${HIP_INDEX:-unset} != 0 ]]; then
        echo "NOT STARTED: device index is ${HIP_INDEX:-unset}; this study only runs on device 0 (the discrete card)" >>"$log"
        return 8
    fi
    echo "card configuration: $(card_config)" >>"$log"
    [[ $(lact_profile) == Testing ]] || echo "NOTE: LACT profile is '$(lact_profile)', not Testing (bits, not times)" >>"$log"
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
}

log_name() {
    local args="$1-$2"
    args=${args// /_}
    args=${args//--/}
    args=${args//build\/left\/bin\//}
    echo "$RUN_DIR/$args.log"
}

selected() {
    local m=$1 s
    ((${#ONLY[@]} == 0)) && return 0
    for s in "${ONLY[@]}"; do
        [[ $s == "${m%% *}" ]] && return 0
    done
    return 1
}

phase_run() {
    local j prog args log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration: $(card_config))"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r prog args <<<"$j"
        selected "$args" || continue
        log=$(log_name "$prog" "$args")
        [[ $prog == left_prio ]] && args=${args#prio }
        [[ $prog == left_rot ]] && args=${args#rot}
        # shellcheck disable=SC2086 # args is a word list on purpose
        job "$log" "$BIN/$prog" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-600 | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $prog $args: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        else
            sum "- $prog $args: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j prog args log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS='|' read -r prog args <<<"$j"
        selected "$args" || continue
        log=$(log_name "$prog" "$args")
        [[ $prog == left_prio ]] && args=${args#prio }
        [[ $prog == left_rot ]] && args=${args#rot}
        # shellcheck disable=SC2086 # args is a word list on purpose
        job "$log" "$BIN/$prog" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-300 | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [^ ]*: FAIL" "$log"; then
            sum "- $prog $args: FAIL as intended. $v"
        else
            sum "- $prog $args: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

PHASE=${1:-all}
shift || true
ONLY=("$@")
sum "# Matrix leftovers run, $(date -u +%FT%TZ)"
sum "- commit $(git -C "$REPO" rev-parse --short HEAD)$([[ -z $(git -C "$REPO" status --porcelain -- src scripts) ]] || echo ' (with uncommitted changes)'); card $ARCH, PCI $PCI_BUS; kernel $(uname -r)"
sum "- LACT profile $(lact_profile)${ONLY[*]:+; modes: ${ONLY[*]}}"
case $PHASE in
    build) phase_build ;;
    run) phase_run ;;
    breaks) phase_breaks ;;
    all)
        phase_build
        phase_run
        phase_breaks
        ;;
    *) die "unknown phase $PHASE; use build, run, breaks or all" ;;
esac
echo "study-leftovers-matrix: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
