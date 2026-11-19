#!/usr/bin/env bash
# Leftovers pass, timing oddities and the remaining non-matrix items (notes/STUDY-PLAN.md): builds and runs
# src/study/tim_*.hip (and tim_bf8search.cpp, a CPU check), and a few Vulkan load runs with build/tim/vkstudy.
#
# Usage: scripts/study-leftovers-timing.sh [build|run|breaks|all] [MODE...]   (default all = build, run, breaks)
#   build   compiles every probe for gfx1201 into build/tim/bin (with -save-temps into build/tim/tmp), checks that
#           src/study/tim_core_blocks.h is what scripts/tim_gen_core.py writes, and builds the Vulkan program and the
#           two load shaders into build/tim (the same commands as scripts/study-vulkan.sh)
#   run     every mode as its own job; with MODE arguments only the jobs whose first argument matches
#   breaks  each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env).
# Results: runs/<UTC time>-timing/: one log per job and summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, device index 0, not started below 8 GB available memory; the LACT
# profile is logged with every job and timed jobs are not started unless it is "Testing". Thermal rule: the probes
# here are one wave or one work-group, with three exceptions that use the whole card in short launches and gate
# themselves (start below 60 C, stop at 95 C): the Vulkan load runs (bursts of about 0.15 s, vkstudy), tim_sr (launches
# of a few tens of ms) and tim_slow (a 512 MiB read of about 1 ms before every repetition).
# The CPU check uses at most 8 threads, under nice.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-leftovers-timing.sh"
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
    echo "study-leftovers-timing: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
export DRM_DEVICE HWMON

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-timing"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/tim/bin
VK=$REPO/build/tim/vkstudy
SPV=$REPO/build/tim/spv
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

# probe | extra hipcc flags (CU mode where a work-group must stay on one CU)
PROBES=(
    "tim_disturb|"
    "tim_core|-mcumode"
    "tim_wait|"
    "tim_mat|-mcumode"
    "tim_sr|"
    "tim_l2|"
    "tim_slow|-mcumode"
)

# program | arguments (one job each; the first word of the arguments is the mode for MODE selection)
JOBS=(
    "tim_disturb|stats"
    "tim_disturb|ramp"
    "tim_core|single"
    "tim_core|waves"
    "tim_core|size fma_ic=8 fma_lit=12 du_fmac=8 du_fmaak=12 add_vop2=4 s_nop=4"
    "tim_wait|run"
    "tim_mat|run"
    "tim_sr|select fp8"
    "tim_sr|select bf8"
    "tim_sr|bits fp8"
    "tim_sr|bits bf8"
    "tim_bf8search|"
    "tim_l2|chase"
    "tim_slow|waits"
    "vkstudy|load fp8 row tight dram --w 4"
    "vkstudy|load fp8 row tight dram --w 8"
    "vkstudy|load fp8 row tight dram --w 16"
    "vkstudy|load f16 row tight dram --w 8"
    "vkstudy|load fp8 row tight dram --sg 32 --w 8"
)
BREAKS=(
    "tim_disturb|stats --break"
    "tim_core|single --break"
    "tim_wait|run --break"
    "tim_mat|run --break"
    "tim_sr|select fp8 --break"
    "tim_sr|bits fp8 --break"
    "tim_bf8search|--break"
    "tim_l2|chase --break"
    "tim_slow|waits --break"
)

phase_build() {
    local p name flags log n
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN" "$REPO/build/tim/tmp" "$SPV"
    if python3 "$REPO/scripts/tim_gen_core.py" "$REPO/build/tim/tim_core_blocks.h" >/dev/null &&
        cmp -s "$REPO/build/tim/tim_core_blocks.h" "$REPO/src/study/tim_core_blocks.h"; then
        sum "- tim_core_blocks.h: matches scripts/tim_gen_core.py"
    else
        sum "- tim_core_blocks.h: does NOT match scripts/tim_gen_core.py"
        FAILED=1
    fi
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name flags <<<"$p"
        [[ -f $REPO/src/study/$name.hip ]] || {
            sum "- $name: no source, skipped"
            continue
        }
        log=$RUN_DIR/build-$name.log
        # shellcheck disable=SC2086 # flags is a word list on purpose
        if (cd "$REPO/build/tim/tmp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 $flags -save-temps \
            -DHARNESS_ARCH='"gfx1201"' -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/study/$name.hip" -o "$BIN/$name" \
            -lamdhip64) >"$log" 2>&1; then
            sum "- $name: built${flags:+ ($flags)}"
        else
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        fi
    done
    # Every timed region is one inline-assembly statement, so the compiler cannot add anything inside it. The matrix
    # probe's listing is checked for the count of matrix instructions it must hold.
    n=$("$ROCM_PATH/llvm/bin/llvm-objdump" -d --mcpu=gfx1201 \
        "$REPO"/build/tim/tmp/tim_mat.hip-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.gfx1201.img.lto.o 2>/dev/null |
        grep -c 'v_wmma_f32_16x16x16')
    if [[ $n == 12288 ]]; then
        sum "- tim_mat object: 12288 matrix instructions (12 kernels x 4 repetitions x 256), as written"
    else
        sum "- tim_mat object check FAILED: ${n:-0} matrix instructions, 12288 expected"
        FAILED=1
    fi
    if [[ -f $REPO/src/study/tim_bf8search.cpp ]]; then
        if nice -n 10 g++ -O2 -std=c++17 -pthread -I"$REPO/src/study" "$REPO/src/study/tim_bf8search.cpp" \
            -o "$BIN/tim_bf8search" >"$RUN_DIR/build-tim_bf8search.log" 2>&1; then
            sum "- tim_bf8search: built"
        else
            sum "- tim_bf8search: build FAILED"
            FAILED=1
        fi
    fi
    log=$RUN_DIR/build-vkstudy.log
    if nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp "$REPO/src/vk/shaders/load.comp" \
        -o "$SPV/load_f16_row.spv" >"$log" 2>&1 &&
        nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp -DT_FP8 "$REPO/src/vk/shaders/load.comp" \
            -o "$SPV/load_fp8_row.spv" >>"$log" 2>&1 &&
        nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -o "$VK" "$REPO/src/vk/vkstudy.c" \
            "$REPO/src/vk/vkh.c" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- vkstudy and the two load shaders: built"
    else
        sum "- vkstudy: build FAILED (see build-vkstudy.log)"
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
    if [[ $(lact_profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)', not Testing" >>"$log"
        return 6
    fi
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH VKS_OUT=$RUN_DIR VKS_SPV=$SPV VKS_TARGET_S=0.15 \
        flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
}

# cpu_job <log> <command...>: no GPU, at most 8 threads (the program's own limit), under nice
cpu_job() {
    local log=$1
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    nice -n 10 timeout 900 "$@" >>"$log" 2>&1
}

log_name() {
    local args="$1-$2"
    args=${args// /_}
    args=${args//--/}
    args=${args//=/}
    echo "$RUN_DIR/${args:0:80}.log"
}

selected() {
    local m=$1 s
    ((${#ONLY[@]} == 0)) && return 0
    for s in "${ONLY[@]}"; do
        [[ $s == "${m%% *}" ]] && return 0
    done
    return 1
}

run_one() {
    local prog=$1 args=$2 log=$3 exe
    if [[ $prog == vkstudy ]]; then
        exe=$VK
    else
        exe=$BIN/$prog
    fi
    if [[ ! -x $exe ]]; then
        echo "NOT BUILT: $exe" >>"$log"
        return 9
    fi
    # shellcheck disable=SC2086 # args is a word list on purpose
    if [[ $prog == tim_bf8search ]]; then
        cpu_job "$log" "$exe" $args
    else
        job "$log" "$exe" $args
    fi
}

phase_run() {
    local j prog args log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration: $(card_config))"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r prog args <<<"$j"
        selected "${args:-$prog}" || selected "$prog" || continue
        log=$(log_name "$prog" "$args")
        run_one "$prog" "$args" "$log"
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-500 | tr '\n' ' ')
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
        selected "${args:-$prog}" || selected "$prog" || continue
        log=$(log_name "$prog" "$args")
        run_one "$prog" "$args" "$log"
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
sum "# Timing leftovers run, $(date -u +%FT%TZ)"
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
echo "study-leftovers-timing: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
