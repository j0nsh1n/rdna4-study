#!/usr/bin/env bash
# Lane 9 of notes/STUDY-PLAN.md: the RX 9070 XT through Vulkan compute (RADV, ACO). Builds the shaders and the host
# program (src/vk/), then runs every probe as its own GPU job, then the deliberate breaks.
#
# Usage: scripts/study-vulkan.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles every shader variant with glslc into build/spv/ and the host program into build/vkstudy
#   run     runs the machine-code checks (mode compile: listings in <run>/asm/) and every measurement as its own
#           GPU job
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Results: runs/<UTC time>-vulkan/: one log per job, clock-log.csv (once a second: sclk, mclk, reported power,
# hotspot), summary.md, asm/ (ACO machine code of every pipeline), coopmat-f16-check-*.txt (item 6 bits).
# Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, not started below 8 GB available memory. Before each job the LACT
# profile (must be "Testing"), pp_od_clk_voltage and power1_cap are logged and compared with the start of the run (a
# change stops all timed work); sclk, mclk, reported power and hotspot are logged before and after. Thermal rule:
# each whole-card repetition is at most about 0.25 s of load (0.05 to 0.1 s in practice) and starts only below 60 C hotspot (the program waits; it
# samples clock, power and hotspot every 50 ms and stops at 95 C). A hotspot of 95 C stops all further GPU work.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-vulkan.sh"
cd "$ROOT" || exit 2
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
LACT_CONFIG=/etc/lact/config.yaml
SPV=$ROOT/build/spv
BIN=$ROOT/build/vkstudy

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
    # Memory was checked before waiting for the lock; after a long wait it may have changed.
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    # A probe that hangs must not hold the card and the shared lock.
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-600}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-600} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-vulkan: $*" >&2
    exit 2
}

# The discrete card: vendor 0x1002, device 0x7550.
DRM_DEVICE=
for d in /sys/class/drm/card[0-9]*/device; do
    [[ $(cat "$d/vendor" 2>/dev/null) == 0x1002 && $(cat "$d/device" 2>/dev/null) == 0x7550 ]] || continue
    DRM_DEVICE=$(readlink -f "$d")
    break
done
[[ -n $DRM_DEVICE ]] || die "no RX 9070 XT (1002:7550) under /sys/class/drm"
HWMON=$(echo "$DRM_DEVICE"/hwmon/hwmon*)
[[ -f $HWMON/temp2_input ]] || die "no hotspot sensor under $DRM_DEVICE/hwmon"

RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-vulkan"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
FAILED=0
HOTTEST=0
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

# spv name | shader source | glslc defines
SHADERS=(
    "mat_f16|mat.comp|-DT_F16 -DLOADAB"
    "mat_bf16|mat.comp|-DT_BF16 -DLOADAB"
    "mat_fp8|mat.comp|-DT_FP8 -DLOADAB"
    "mat_i8|mat.comp|-DT_I8 -DLOADAB"
    "mat_f16_splat|mat.comp|-DT_F16"
    "mat_bf16_splat|mat.comp|-DT_BF16"
    "mat_fp8_splat|mat.comp|-DT_FP8"
    "mat_i8_splat|mat.comp|-DT_I8"
    "mat_f16_unused|mat.comp|-DT_F16 -DLOADAB -DUNUSED"
    "fma|fma.comp|"
    "bw|bw.comp|"
    "wave|wave.comp|"
    "wave_coop|wave.comp|-DCOOP"
    "empty|empty.comp|"
    "one|one.comp|"
    "check|check.comp|"
    "check_big_f16|check_big.comp|"
    "check_big_bf16|check_big.comp|-DT_BF16"
    "check_big_fp8|check_big.comp|-DT_FP8"
    "dot|dot.comp|"
    "load_f16_row|load.comp|"
    "load_f16_col|load.comp|-DCOLMAJOR"
    "load_fp8_row|load.comp|-DT_FP8"
    "load_fp8_col|load.comp|-DT_FP8 -DCOLMAJOR"
)
# arguments | 1 = whole-card job (documentation only: the program applies the thermal rule to every repetition)
JOBS=(
    "info|0"
    "compile|0"
    "wave|0"
    "dispatch|0"
    "check|0"
    "bw|1"
    "fma --sg 32|1"
    "fma --sg 64|1"
    "mat f16|1"
    "mat f16 --sg 32|1"
    "mat bf16|1"
    "mat bf16 --sg 32|1"
    "mat fp8|1"
    "mat fp8 --sg 32|1"
    "mat i8|1"
    "mat i8 --sg 32|1"
    "mat f16 --w 1|1"
    "mat f16 --sg 32 --w 1|1"
    "mat fp8 --sg 32 --w 1|1"
    "mat f16 --sg 32 --w 16|1"
    "mat f16 --splat|1"
    "mat f16 --sg 32 --splat|1"
    "mat i8 --sg 32 --splat|1"
    "load f16 row tight l2|1"
    "load f16 row tight mall|1"
    "load f16 row tight dram|1"
    "load f16 col tight l2|1"
    "load f16 col tight dram|1"
    "load f16 row wide l2|1"
    "load f16 row wide dram|1"
    "load f16 col wide dram|1"
    "load f16 row tight l2 --sg 32|1"
    "load f16 row tight dram --sg 32|1"
    "load fp8 row tight l2|1"
    "load fp8 row tight mall|1"
    "load fp8 row tight dram|1"
    "load fp8 col tight dram|1"
    "load fp8 row wide dram|1"
)
BREAKS=(
    "bw --expect 1266|1"
    "mat f16 --unused|1"
    "load f16 row tight l2 --swap-check|1"
)

phase_build() {
    local s name src defs log=$RUN_DIR/build.log rc=0
    mkdir -p "$SPV"
    sum "## build ($(glslc --version 2>&1 | head -1 | tr -d "\n"); $(cc --version | head -1))"
    for s in "${SHADERS[@]}"; do
        IFS='|' read -r name src defs <<<"$s"
        # shellcheck disable=SC2086 # defs is a word list on purpose
        if ! nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp $defs "$ROOT/src/vk/shaders/$src" \
            -o "$SPV/$name.spv" >>"$log" 2>&1; then
            sum "- shader $name: glslc FAILED (see build.log)"
            rc=1
            continue
        fi
        if ! spirv-val --target-env vulkan1.3 "$SPV/$name.spv" >>"$log" 2>&1; then
            sum "- shader $name: spirv-val FAILED (see build.log)"
            rc=1
        fi
    done
    if nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -o "$BIN" "$ROOT/src/vk/vkstudy.c" \
        "$ROOT/src/vk/vkh.c" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- host program built: $BIN ($(grep -c warning "$log") compiler warnings)"
    else
        sum "- host program: build FAILED (see build.log)"
        rc=1
    fi
    ((rc == 0)) && sum "- ${#SHADERS[@]} shader variants compiled and validated"
    ((rc)) && FAILED=1
    return "$rc"
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

# gpu_job <log> <command...>
gpu_job() {
    local log=$1 cfg t
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((STOPPED)); then
        echo "NOT STARTED: timed work was stopped earlier in this run" >>"$log"
        return 6
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
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
    t=$(hotspot_c)
    ((t > HOTTEST)) && HOTTEST=$t
    VKS_OUT=$RUN_DIR VKS_SPV=$SPV flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
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
    return "$rc"
}

log_name() {
    local args=$1
    args=${args//--/}
    args=${args// /_}
    echo "$RUN_DIR/$args.log"
}

phase_run() {
    local j args log rc v
    sum "## run (sysfs $DRM_DEVICE; card configuration at start: $CONFIG0)"
    [[ -x $BIN ]] || {
        sum "- host program not built"
        FAILED=1
        return
    }
    for j in "${JOBS[@]}"; do
        IFS="|" read -r args _ <<<"$j"
        log=$(log_name "$args")
        # shellcheck disable=SC2086 # args is a word list on purpose
        gpu_job "$log" "$BIN" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $args: ok. $(grep -h '^RESULT' "$log" | head -3 | cut -c1-400 | tr '\n' ' ') ${v}"
        else
            sum "- $args: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j args log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS="|" read -r args _ <<<"$j"
        log=$(log_name "$args-break")
        # shellcheck disable=SC2086 # args is a word list on purpose
        gpu_job "$log" "$BIN" $args
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [^ ]*: FAIL" "$log"; then
            sum "- $args: FAIL as intended. $v"
        else
            sum "- $args: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

sum "# Vulkan view study run, $(date -u +%FT%TZ)"
sum "- card sysfs $DRM_DEVICE; kernel $(uname -r); $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | tr -s ' ')"
sum "- LACT profile $(lact_profile); power1_cap $(($(cat "$HWMON/power1_cap") / 1000000)) W (a 1.2x regulator-level power-limit mod is in the card: reported power is scaled down)"
case ${1:-all} in
    build) phase_build ;;
    run | breaks | all)
        if [[ ${1:-all} == all ]]; then
            phase_build || die "build failed; see $RUN_DIR"
        fi
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
echo "study-vulkan: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
