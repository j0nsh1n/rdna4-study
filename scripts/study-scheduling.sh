#!/usr/bin/env bash
# Lane 7 of notes/STUDY-PLAN.md: scheduling, launch, clocks, power and sharing. Builds and runs the sched_* probes
# (src/study/), the Vulkan queue probe (src/vk/vksched.c), and writes measurements' raw material to a run folder.
#
# Usage: scripts/study-scheduling.sh [build|run|traces|breaks|all]   (default all = build, run, traces, breaks)
#   build   compiles every probe for gfx1201 into build/ (-save-temps output in build/tmp-<probe>/) and the Vulkan
#           probe into build/vk/
#   run     every short probe as its own GPU job (launch, placement, barriers, wave sizes, sharing, Vulkan queues)
#   traces  item 5: one clock/power/temperature trace per load type (idle, light, stream, fma, f16, fp8), each until
#           95 C or 20 s, at least 60 s and a hotspot below 55 C between them, 30 s of cool-down sampled after each
#   breaks  each deliberate break must print FAIL and exit non-zero (a check that cannot fail is useless)
# Environment: SCHED_RUN_DIR (reuse one run folder across phases); SCHED_TRACES (load types to trace, default
# "idle light stream fma f16 fp8"); STUDY_REPO (default: this folder's parent) supplies src/harness.h, src/study/*.h and the card record
# runs/latest/card.env; CARD_ENV overrides the card record.
# Results: runs/<UTC time>-sched/ under this folder: one log per job, config.txt, summary.md. Exit status 1 if
# anything failed.
# GPU jobs: the shared lock, nice, a time limit, HIP device 0, not started below 8 GB available memory. Before each
# job the LACT profile must be "Testing" (else all wall-clock work stops); clocks, reported power and hotspot are
# logged before and after. Whole-card jobs wait for the hotspot below 60 C (the probes check it too) and stop their
# load at 95 C from an in-process 50 ms sampler. A hotspot of 95 C or more after any job stops all further GPU work.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-scheduling.sh"
STUDY_REPO=${STUDY_REPO:-$ROOT}
cd "$ROOT" || exit 2
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
LACT_CONFIG=/etc/lact/config.yaml
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=0 CARD_ARCH=gfx1201

snap() {
    local drm=$1 hw=$2
    echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')" \
        " power(reported, scaled down by the power-limit mod): $(($(cat "$hw/power1_average") / 1000000)) W" \
        " hotspot: $(($(cat "$hw/temp2_input") / 1000)) C  busy: $(cat "$drm/gpu_busy_percent") %"
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
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-300}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-300} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

# Two processes of the pair test, started together inside one locked job (item 7b). Arguments: kind of A, kind of
# B, seconds, output prefix. Both wait for the same wall-clock start time.
if [[ ${1:-} == __pair ]]; then
    ka=$2
    kb=$3
    secs=$4
    out=$5
    "$ROOT/build/sched_share" gate || exit 7
    start=$(($(date +%s%3N) + 2500))
    "$ROOT/build/sched_share" pair --role A --kind "$ka" --start-ms "$start" --seconds "$secs" --out "$out" >"$out-A.log" 2>&1 &
    pa=$!
    "$ROOT/build/sched_share" pair --role B --kind "$kb" --start-ms "$start" --seconds "$secs" --out "$out" >"$out-B.log" 2>&1 &
    pb=$!
    wait "$pa"
    ra=$?
    wait "$pb"
    rb=$?
    cat "$out-A.log" "$out-B.log"
    python3 "$ROOT/scripts/sched_pair.py" "$out-A.csv" "$out-B.csv"
    rp=$?
    ((ra == 0 && rb == 0 && rp == 0)) || exit 1
    exit 0
fi

die() {
    echo "study-scheduling: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$STUDY_REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV (set STUDY_REPO or CARD_ENV)"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
export DRM_DEVICE HWMON

RUN_DIR=${SCHED_RUN_DIR:-"$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-sched"}
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$ROOT/build
FAILED=0
STOPPED=0
HOTTEST=0
LAST_TRACE_END=0

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
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") perf_level=$(cat "$DRM_DEVICE/power_dpm_force_performance_level") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null)"
}

record_config() {
    local f=$RUN_DIR/config.txt
    {
        echo "# Configuration record (read-only), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "STUDY_REPO=$STUDY_REPO CARD_ENV=$CARD_ENV"
        echo "lact current_profile: $(lact_profile)"
        echo "power1_cap: $(cat "$HWMON/power1_cap")  power1_cap_default: $(cat "$HWMON/power1_cap_default" 2>/dev/null)  power1_cap_max: $(cat "$HWMON/power1_cap_max" 2>/dev/null)"
        echo "Note: a regulator-level power-limit mod (1.2x) is in the card; power1_average is scaled down on the main rail and is not true board power."
        echo "power_dpm_force_performance_level: $(cat "$DRM_DEVICE/power_dpm_force_performance_level")"
        echo "--- pp_dpm_sclk"
        cat "$DRM_DEVICE/pp_dpm_sclk"
        echo "--- pp_dpm_mclk"
        cat "$DRM_DEVICE/pp_dpm_mclk"
        echo "--- pp_od_clk_voltage"
        cat "$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null
        echo "--- pp_power_profile_mode"
        cat "$DRM_DEVICE/pp_power_profile_mode" 2>/dev/null
        echo "--- hwmon"
        for f2 in "$HWMON"/{temp1_label,temp2_label,temp3_label,temp1_crit,temp2_crit,temp3_crit,temp1_emergency,temp2_emergency,temp3_emergency,fan1_min,fan1_max,fan1_enable,power1_label,freq1_label,freq2_label}; do
            [[ -f $f2 ]] && echo "$(basename "$f2"): $(cat "$f2")"
        done
        echo "--- idle snapshot: $(snap "$DRM_DEVICE" "$HWMON")"
        echo "--- kfd topology (node $KFD_NODE)"
        grep -E "simd_count|cu_per_simd_array|simd_arrays_per_engine|array_count|max_waves_per_simd|wave_front_size|lds_size|num_xcc|max_engine_clk_fcompute|gfx_target_version" \
            "/sys/class/kfd/kfd/topology/nodes/$KFD_NODE/properties" 2>/dev/null
        echo "--- versions"
        hipcc --version 2>/dev/null | head -2
        uname -r
        echo "--- /etc/lact/config.yaml (profiles section names)"
        grep -nE "^  [A-Za-z].*:$|current_profile" "$LACT_CONFIG" 2>/dev/null | head -20
    } >"$f" 2>&1
}

CONFIG0=$(card_config)

# Runs one GPU job. Arguments: job name, timeout seconds, command...
gpu_job() {
    local name=$1 tmo=$2 log h
    shift 2
    log=$RUN_DIR/$name.log
    if ((STOPPED)); then
        sum "- $name: NOT RUN (GPU work stopped earlier)"
        return 1
    fi
    if [[ $(lact_profile) != Testing ]]; then
        sum "- $name: NOT RUN: LACT profile is '$(lact_profile)', not 'Testing'; wall-clock work stopped"
        STOPPED=1
        FAILED=1
        return 1
    fi
    if [[ $(card_config) != "$CONFIG0" ]]; then
        sum "- $name: NOT RUN: card configuration changed since the start of the run; work stopped"
        echo "now: $(card_config)" >>"$RUN_DIR/config.txt"
        STOPPED=1
        FAILED=1
        return 1
    fi
    if (($(avail_gb) < 8)); then
        sum "- $name: NOT RUN: $(avail_gb) GB available (floor 8)"
        FAILED=1
        return 1
    fi
    {
        echo "# $name: $*"
        echo "# start $(date -u +%Y-%m-%dT%H:%M:%SZ) profile=$(lact_profile)"
    } >"$log"
    JOB_TIMEOUT=$tmo flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
    local rc=$?
    echo "# end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >>"$log"
    h=$(hotspot_c)
    ((h > HOTTEST)) && HOTTEST=$h
    local v
    v=$(grep -h "^VERDICT" "$log" | sed 's/^/  - /')
    if ((${IN_BREAK:-0})); then
        sum "- $name: exit $rc (deliberate break; judged below)"
    elif ((rc == 0)); then
        sum "- $name: ok"
    else
        sum "- $name: FAILED (exit $rc)"
        FAILED=1
    fi
    [[ -n $v ]] && sum "$v"
    if grep -q "HOTTEST" "$log"; then
        local hh
        hh=$(grep -h "^HOTTEST" "$log" | awk '{print $2}' | sort -n | tail -1)
        ((hh > HOTTEST)) && HOTTEST=$hh
    fi
    if ((h >= 95)); then
        sum "- hotspot $h C after $name: all further GPU work stopped"
        STOPPED=1
        FAILED=1
    fi
    return "$rc"
}

PROBES=(
    "sched_launch|sched_launch.hip||"
    "sched_place_wgp|sched_place.hip||"
    "sched_place_cu|sched_place.hip|-mcumode -DCUMODE=1|"
    "sched_trace|sched_trace.hip|-mcumode|"
    "sched_share|sched_share.hip|-mcumode|"
    "sched_barrier|sched_barrier.hip||"
    "sched_barrier_cu|sched_barrier.hip|-mcumode -DCUMODE=1|"
    "sched_wave32|sched_wave.hip|-mcumode -DCUMODE=1|"
    "sched_wave64|sched_wave.hip|-mcumode -DCUMODE=1 -mwavefrontsize64 -DW64=1|"
    "sched_wave32_wgp|sched_wave.hip||"
)

build_one() {
    local name=$1 src=$2 flags=$3 dir
    dir=$BIN/tmp-$name
    mkdir -p "$dir"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags \
        -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -I"$ROOT/src/study" -save-temps "$ROOT/src/study/$src" \
        -o "$BIN/$name" -lamdhip64) >"$RUN_DIR/build-$name.log" 2>&1
}

phase_build() {
    local p name src flags rest running
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN/vk"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags rest <<<"$p"
        [[ -f $ROOT/src/study/$src ]] || continue
        # at most 8 cores: 4 builds at a time
        while running=$(jobs -rp | wc -l) && ((running >= 4)); do
            wait -n
        done
        build_one "$name" "$src" "$flags" &
    done
    wait
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags rest <<<"$p"
        if [[ ! -f $ROOT/src/study/$src ]]; then
            sum "- $name: source src/study/$src missing; FAILED"
            FAILED=1
        elif [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        else
            sum "- $name: built"
        fi
    done
    # The dual-issue FMA loops must hold v_dual_fmac (the load the trace and scaling rows assume).
    if grep -q "v_dual_fmac" "$BIN"/tmp-sched_trace/*gfx1201*.s 2>/dev/null; then
        sum "- sched_trace: fma loop holds v_dual_fmac_f32"
    else
        sum "- sched_trace: fma loop has no v_dual_fmac_f32; FAILED"
        FAILED=1
    fi
    if [[ -f $ROOT/src/vk/vksched.c ]]; then
        if nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp "$ROOT/src/vk/shaders/vksched_one.comp" \
            -o "$BIN/vk/vksched_one.spv" >"$RUN_DIR/build-vksched.log" 2>&1 &&
            spirv-val --target-env vulkan1.3 "$BIN/vk/vksched_one.spv" >>"$RUN_DIR/build-vksched.log" 2>&1 &&
            nice -n 10 cc -O2 -std=c11 -Wall -Wextra -o "$BIN/vk/vksched" "$ROOT/src/vk/vksched.c" -lvulkan -lm \
                >>"$RUN_DIR/build-vksched.log" 2>&1; then
            sum "- vksched: built"
        else
            sum "- vksched: build FAILED (see build-vksched.log)"
            FAILED=1
        fi
    else
        sum "- vksched: source src/vk/vksched.c missing; FAILED"
        FAILED=1
    fi
}

phase_run() {
    sum ""
    sum "## run"
    gpu_job launch-calib 60 "$BIN/sched_launch" calib
    gpu_job launch 120 "$BIN/sched_launch" launch
    gpu_job launch-idle 120 "$BIN/sched_launch" idle
    local i
    for i in 10 100 1000; do
        gpu_job "ramp-onewave-idle$i" 60 "$BIN/sched_launch" ramp --idle-ms "$i" --ms 150
    done
    gpu_job ramp-card-idle1000 240 "$BIN/sched_launch" ramp --idle-ms 1000 --ms 150 --card
    gpu_job place-wgp 120 "$BIN/sched_place_wgp" place
    gpu_job place-cu 120 "$BIN/sched_place_cu" place
    gpu_job regs-wgp 60 "$BIN/sched_place_wgp" regs
    gpu_job regs-cu 60 "$BIN/sched_place_cu" regs
    gpu_job ids 60 "$BIN/sched_place_wgp" ids
    gpu_job scale-cu 240 "$BIN/sched_place_cu" scale
    gpu_job detector-selftest 30 "$BIN/sched_share" selftest
    gpu_job spin-desktop 60 "$BIN/sched_share" spin --seconds 4 --out "$RUN_DIR/spin"
    gpu_job spin-multi 60 "$BIN/sched_share" multi --seconds 0.5
    gpu_job pair-big-alone 240 "$BIN/sched_share" pair --role A --kind big --start-ms 0 --seconds 0.6 --out "$RUN_DIR/alone"
    gpu_job pair-big-big 240 "$SELF" __pair big big 0.6 "$RUN_DIR/pair-big-big"
    gpu_job pair-part-part 240 "$SELF" __pair part part 0.6 "$RUN_DIR/pair-part-part"
    gpu_job pair-small-big 240 "$SELF" __pair small big 0.6 "$RUN_DIR/pair-small-big"
    gpu_job pair-small-small 120 "$SELF" __pair small small 1.0 "$RUN_DIR/pair-small-small"
    run_helper_jobs
}

# Items 3, 4 and 6 (probes written by helpers, rebuilt and rerun here).
run_helper_jobs() {
    gpu_job barrier-wgp 120 "$BIN/sched_barrier"
    gpu_job barrier-cu 120 "$BIN/sched_barrier_cu"
    gpu_job wave32 120 "$BIN/sched_wave32"
    gpu_job wave64 120 "$BIN/sched_wave64"
    mkdir -p "$RUN_DIR/vk"
    gpu_job vksched 200 "$BIN/vk/vksched" --spv "$BIN/vk/vksched_one.spv" --out "$RUN_DIR/vk"
}

phase_traces() {
    sum ""
    sum "## traces (item 5; each until 95 C or 20 s; 30 s of cool-down sampled)"
    local m now w
    # SCHED_TRACES narrows the list: a load whose trace reached 95 C in under 3 s is not traced again (brief).
    for m in ${SCHED_TRACES:-idle light stream fma f16 fp8}; do
        ((STOPPED)) && break
        if [[ $m != idle && $m != light ]]; then
            # at least 60 s after the previous whole-card trace and a hotspot below 55 C, outside the lock
            now=$(date +%s)
            w=$((LAST_TRACE_END + 60 - now))
            ((w > 0)) && sleep "$w"
            local waited=0
            while (($(hotspot_c) >= 55)); do
                if ((waited >= 300)); then
                    sum "- trace-$m: NOT RUN: hotspot still $(hotspot_c) C after 5 minutes"
                    FAILED=1
                    continue 2
                fi
                sleep 5
                waited=$((waited + 5))
            done
        fi
        if [[ $m == idle ]]; then
            gpu_job "trace-$m" 120 "$BIN/sched_trace" idle --max-s 10 --pre-s 1 --post-s 0 --out "$RUN_DIR/trace-$m"
        else
            gpu_job "trace-$m" 200 "$BIN/sched_trace" "$m" --max-s 20 --pre-s 1 --post-s 30 --out "$RUN_DIR/trace-$m"
        fi
        LAST_TRACE_END=$(date +%s)
    done
}

# name | command (words) ; every one must exit non-zero and print a FAIL verdict. break-wave-rows-wgp-build runs the
# wave probe built without -mcumode: its multi-wave FMA rows then read 0 and the rows check must fail.
phase_breaks() {
    sum ""
    sum "## deliberate breaks (each must FAIL)"
    local b name rc
    local -a BREAKS=(
        "break-calib|$BIN/sched_launch calib --expect-fma 2"
        "break-placement|$BIN/sched_place_wgp place --break"
        "break-detector-clean|$BIN/sched_share selftest --expect-clean 1"
        "break-detector-injected|$BIN/sched_share selftest --expect-injected 0"
        "break-barrier-calib|$BIN/sched_barrier --break"
        "break-wave-wmma|$BIN/sched_wave32 --break"
        "break-wave-rows-wgp-build|$BIN/sched_wave32_wgp"
        "break-vksched|$BIN/vk/vksched --spv $BIN/vk/vksched_one.spv --out $RUN_DIR/vk-break --break"
    )
    mkdir -p "$RUN_DIR/vk-break"
    for b in "${BREAKS[@]}"; do
        name=${b%%|*}
        # shellcheck disable=SC2086 # the command is a word list on purpose
        if [[ $b == *selftest* ]]; then
            ${b#*|} >"$RUN_DIR/$name.log" 2>&1
            rc=$?
        else
            local fsave=$FAILED
            # shellcheck disable=SC2086
            IN_BREAK=1 gpu_job "$name" 200 ${b#*|}
            rc=$?
            FAILED=$fsave
        fi
        if ((rc != 0)) && grep -q "VERDICT.*FAIL" "$RUN_DIR/$name.log"; then
            sum "- $name: FAILED as intended: $(grep -h 'VERDICT.*FAIL' "$RUN_DIR/$name.log" | head -1)"
        else
            sum "- $name: did NOT fail (exit $rc); the check cannot fail: FAILED"
            FAILED=1
        fi
    done
}

phase=${1:-all}
{
    echo "# Lane 7 run $(date -u +%Y-%m-%dT%H:%M:%SZ), phase $phase"
    echo "Profile: $(lact_profile) (stock clocks, 317 W cap); power-limit mod present (reported power scaled down); desktop in use."
    echo "STUDY_REPO=$STUDY_REPO"
    echo
} >>"$SUMMARY"
record_config
case $phase in
build) phase_build ;;
run) phase_run ;;
traces) phase_traces ;;
breaks) phase_breaks ;;
all)
    phase_build
    phase_run
    phase_traces
    phase_breaks
    ;;
*) die "unknown phase $phase (build|run|traces|breaks|all)" ;;
esac
sum ""
sum "Highest hotspot seen: $HOTTEST C. Run folder: $RUN_DIR"
((FAILED)) && sum "RESULT: something FAILED (see above)" || sum "RESULT: all checks passed"
cat "$SUMMARY"
exit "$FAILED"
