#!/usr/bin/env bash
# Lane 5 of notes/STUDY-PLAN.md: build and run the matrix-arithmetic probes (src/study/arith_*), one command.
#
# Usage: scripts/study-matrix-arith.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles the GPU probes for gfx1201 (arith_check, arith_more, arith_run, arith_dump) and the CPU programs
#           (arith_stats, arith_qworst, arith_realw, arith_fit, arith_eval) into build/arith/bin, and extracts real
#           E4M3 weight rows (scripts/gguf_rows.py) for arith_realw when the model file is present
#   run     every mode of arith_check and arith_more (GPU, single waves) and arith_stats (CPU) as its own job
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env, written by scripts/first-hour.sh info);
#   REALW_MODEL=<gguf> (default the FP8 Llama 3.2 3B file of the rdna3-fp8 project, read only).
# Results: runs/<UTC time>-arith/: one log per job and summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, device index 0 from the card record, not started below 8 GB
# available memory; the LACT profile and the clocks are logged before and after each job. Results here are bits,
# not times, so a profile other than "Testing" is recorded, not fatal. CPU jobs use at most 8 threads, under nice.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-matrix-arith.sh"
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
    echo "study-matrix-arith: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-arith"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/arith/bin
REALW_MODEL=${REALW_MODEL:-${LLM_MODELS:-$HOME/Projects/rdna3-fp8/models}/llama32-3b-fp8-outq8.gguf}
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

# name | source under src/study | gpu (1) or cpu (0)
PROBES=(
    "arith_check|arith_check.hip|1"
    "arith_more|arith_more.hip|1"
    "arith_run|arith_run.hip|1"
    "arith_dump|arith_dump.hip|1"
    "arith_stats|arith_stats.cpp|0"
    "arith_qworst|arith_qworst.cpp|0"
    "arith_realw|arith_realw.cpp|0"
    "arith_fit|arith_fit.cpp|0"
    "arith_eval|arith_eval.cpp|0"
)
# program | arguments | gpu (1) or cpu (0)
JOBS=(
    "arith_check|selftest|0"
    "arith_check|random|1"
    "arith_check|adv|1"
    "arith_check|special|1"
    "arith_check|determ|1"
    "arith_check|chain|1"
    "arith_check|mode|1"
    "arith_more|acc16|1"
    "arith_more|ovfl|1"
    "arith_more|mods|1"
    "arith_more|sparsemap|1"
    "arith_more|sparse|1"
    "arith_more|int|1"
    "arith_more|chain8|1"
    "arith_more|hazard|1"
    "arith_more|wide|1"
    "arith_more|edge|1"
    "arith_stats|exact|0"
    "arith_stats|diff|0"
    "arith_stats|quant|0"
    "arith_stats|worst|0"
    "arith_qworst||0"
    "arith_realw|build/arith/realw/rows.bin|0"
)
BREAKS=(
    "arith_check|selftest --break|0"
    "arith_check|random --n 200000 --break|1"
    "arith_check|special --break|1"
    "arith_check|determ --break|1"
    "arith_check|mode --break|1"
    "arith_more|acc16 --n 100000 --break|1"
    "arith_more|ovfl --break|1"
    "arith_more|mods --n 100000 --break|1"
    "arith_more|sparsemap --break|1"
    "arith_more|sparse --n 50000 --break|1"
    "arith_more|int --n 50000 --break|1"
    "arith_more|chain8 --break|1"
    "arith_more|hazard --break|1"
    "arith_more|wide --n 200000 --break|1"
    "arith_more|edge --break|1"
    "arith_stats|exact --break|0"
    "arith_stats|quant --break|0"
    "arith_qworst|--break|0"
    "arith_realw|build/arith/realw/rows.bin --break|0"
)

phase_build() {
    local p name src gpu log
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN" "$REPO/build/arith/tmp"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src gpu <<<"$p"
        log=$RUN_DIR/build-$name.log
        if ((gpu)); then
            (cd "$REPO/build/arith/tmp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 \
                -DHARNESS_ARCH='"gfx1201"' "$REPO/src/study/$src" -o "$BIN/$name" -lamdhip64) >"$log" 2>&1
        else
            nice -n 10 g++ -O2 -std=c++17 -Wall -pthread "$REPO/src/study/$src" -o "$BIN/$name" >"$log" 2>&1
        fi
        if (($? != 0)) || [[ ! -x $BIN/$name ]]; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        else
            sum "- $name: built"
        fi
    done
    log=$RUN_DIR/build-realw-rows.log
    if [[ -f $REALW_MODEL ]]; then
        mkdir -p "$REPO/build/arith/realw"
        if nice -n 10 python3 "$REPO/scripts/gguf_rows.py" "$REALW_MODEL" extract blk.10.ffn_down.weight \
            "$REPO/build/arith/realw/rows.bin" 3072 >"$log" 2>&1; then
            sum "- real weight rows: $(tail -1 "$log")"
        else
            sum "- real weight rows: extraction FAILED (see build-realw-rows.log)"
            FAILED=1
        fi
    else
        sum "- real weight rows: $REALW_MODEL not found; arith_realw will fail"
    fi
}

# job <log> <gpu> <command...>
job() {
    local log=$1 gpu=$2
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if ((!gpu)); then
        nice -n 10 timeout 900 "$@" >>"$log" 2>&1
        return
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
    local name=$1 args=$2
    args=${args// /_}
    args=${args//--/}
    args=${args//\//_}
    echo "$RUN_DIR/$name${args:+-$args}.log"
}

phase_run() {
    local j name args gpu log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration: $(card_config))"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name args gpu <<<"$j"
        log=$(log_name "$name" "$args")
        # shellcheck disable=SC2086 # args is a word list on purpose
        job "$log" "$gpu" "$BIN/$name" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-400 | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $name $args: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        else
            sum "- $name $args: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j name args gpu log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS='|' read -r name args gpu <<<"$j"
        log=$(log_name "$name" "$args")
        # shellcheck disable=SC2086 # args is a word list on purpose
        job "$log" "$gpu" "$BIN/$name" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-300 | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [^ ]*: FAIL" "$log"; then
            sum "- $name $args: FAIL as intended. $v"
        else
            sum "- $name $args: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

sum "# Matrix-arithmetic study run, $(date -u +%FT%TZ)"
sum "- commit $(git -C "$REPO" rev-parse --short HEAD)$([[ -z $(git -C "$REPO" status --porcelain -- src scripts) ]] || echo ' (with uncommitted changes)'); card $ARCH, PCI $PCI_BUS; kernel $(uname -r)"
sum "- LACT profile $(lact_profile)"
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
echo "study-matrix-arith: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
