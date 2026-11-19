#!/usr/bin/env bash
# Lane 6 of notes/STUDY-PLAN.md: conversions, dot products and small formats on gfx1201 (src/study/cvt_*).
#
# Usage: scripts/study-conversions.sh [build|cpu|run|breaks|all]   (default all)
#   build   compiles every probe into build/bin/ (-save-temps in build/tmp-<probe>/), checks that
#           src/study/cvt_time_blocks.h is what scripts/cvt_gen_timing.py generates, and checks every timed loop of
#           cvt_time with core_loopcheck.py
#   cpu     the CPU self-tests of the references (cvt_refcheck, cvt_dot selftest)
#   run     every GPU job, one at a time, each under the shared lock; jobs that load the whole card wait for the
#           hotspot to fall below 65 C first and the script stops at 95 C
#   breaks  the deliberate breaks: each must report FAIL
# Environment: STUDY_REPO (default: this folder's parent) supplies src/harness.h, src/study/core_common.h,
#   src/study/arith_model.h, src/e4m3_ref.h and scripts/core_loopcheck.py; CARD_ENV (default
#   $STUDY_REPO/runs/latest/card.env); LLAMA_DIR (llama.cpp tree with ggml/include, for cvt_refcheck) and LLAMA_LIB
#   (the folder holding its libggml-base.so; default: the first build*/bin under LLAMA_DIR that has it).
# Results: runs/<UTC time>-conv/: one log per job, summary.md (configuration record, verdicts). Exit status 1 if
# anything failed.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-conversions.sh"
STUDY_REPO=${STUDY_REPO:-$ROOT}
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH
LLAMA_DIR=${LLAMA_DIR:-${LLM_BUILD_ROOT:-$HOME/Projects/rdna3-fp8/build}/llama.cpp-latest}
if [[ -z ${LLAMA_LIB:-} ]]; then
    for d in "$LLAMA_DIR"/build*/bin; do
        if [[ -f $d/libggml-base.so ]]; then
            LLAMA_LIB=$d
            break
        fi
    done
fi
LLAMA_LIB=${LLAMA_LIB:-$LLAMA_DIR/build-gfx1201/bin}
HOT_WAIT=65000
HOT_STOP=95000

# Locked runner: re-entered under flock, so the clocks are read while this job holds the GPU.
if [[ ${1:-} == __locked ]]; then
    log=$2
    drm=$3
    shift 3
    snap() {
        echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')"
    }
    echo "clocks before: $(snap)" >>"$log"
    nice -n 10 "$@" >>"$log" 2>&1
    rc=$?
    echo "clocks after: $(snap)" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-conversions: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$STUDY_REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
sensors=("$DRM_CARD"/device/hwmon/hwmon*/temp2_input)
HWMON_TEMP=${sensors[0]}
[[ -r $HWMON_TEMP ]] || die "no hotspot sensor under $DRM_CARD/device/hwmon"

RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-conv"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$ROOT/build/bin
FAILED=0
HOT_MAX=0

sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

# Reads the hotspot (millidegrees) into HOT and keeps the highest value seen (no subshell, so HOT_MAX survives).
HOT=0
hotspot() {
    HOT=$(<"$HWMON_TEMP")
    if ((HOT > HOT_MAX)); then
        HOT_MAX=$HOT
    fi
}

# name | source | extra flags
PROBES=(
    "cvt_exact|cvt_exact.hip|"
    "cvt_sr|cvt_sr.hip|"
    "cvt_widen|cvt_widen.hip|"
    "cvt_dot|cvt_dot.hip|-pthread"
    "cvt_time|cvt_time.hip|-mllvm -amdgpu-enable-vopd=0"
    "cvt_f16|cvt_f16.hip|"
)

build_one() {
    local name=$1 src=$2 flags=$3 dir log
    dir=$ROOT/build/tmp-$name
    log=$RUN_DIR/build-$name.log
    mkdir -p "$dir" "$BIN"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags -save-temps \
        -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -I"$ROOT/src/study" "$ROOT/src/study/$src" -o "$BIN/$name" \
        -lamdhip64) >"$log" 2>&1
}

phase_build() {
    local p name src flags pids=() rc
    sum "## build ($(hipcc --version 2>/dev/null | grep -m1 -o 'clang version [0-9.]*'), $(command -v hipcc))"
    if python3 "$ROOT/scripts/cvt_gen_timing.py" | cmp -s - "$ROOT/src/study/cvt_time_blocks.h"; then
        sum "- cvt_time_blocks.h matches scripts/cvt_gen_timing.py"
    else
        sum "- cvt_time_blocks.h differs from scripts/cvt_gen_timing.py output; FAILED"
        FAILED=1
    fi
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        if [[ ! -f $ROOT/src/study/$src ]]; then
            sum "- $name: source src/study/$src missing; FAILED"
            FAILED=1
            continue
        fi
        build_one "$name" "$src" "$flags" &
        pids+=($!)
    done
    # CPU reference test (links llama.cpp's ggml-base for the independent E4M3 decoder)
    (nice -n 10 g++ -O2 -std=c++17 -Wall -I"$STUDY_REPO/src" -I"$LLAMA_DIR/ggml/include" \
        "$ROOT/src/study/cvt_refcheck.cpp" -L"$LLAMA_LIB" -lggml-base -Wl,-rpath,"$LLAMA_LIB" -pthread \
        -o "$BIN/cvt_refcheck") >"$RUN_DIR/build-cvt_refcheck.log" 2>&1 &
    pids+=($!)
    wait "${pids[@]}"
    for p in "${PROBES[@]}" "cvt_refcheck||"; do
        IFS='|' read -r name src flags <<<"$p"
        if [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        else
            sum "- $name: built"
        fi
    done
    python3 "$STUDY_REPO/scripts/core_loopcheck.py" "$ROOT/build/tmp-cvt_time"/*img.lto.s >"$RUN_DIR/loopcheck-cvt_time.txt" 2>&1
    rc=$?
    sum "- cvt_time loops: $(tail -1 "$RUN_DIR/loopcheck-cvt_time.txt")$( ((rc == 0)) || echo '; LOOP CHECK FAILED')"
    ((rc == 0)) || FAILED=1
}

# gpu_job <heavy 0|1> <log> <command...>
gpu_job() {
    local heavy=$1 log=$2
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if ((heavy)); then
        hotspot
        while ((HOT >= HOT_WAIT)); do
            if ((HOT >= HOT_STOP)); then
                echo "STOPPED: hotspot $HOT" >>"$log"
                sum "- STOPPED: hotspot reached $((HOT / 1000)) C"
                echo "study-conversions: hotspot $HOT, stopping" >&2
                exit 3
            fi
            sleep 5
            hotspot
        done
        echo "hotspot before: $((HOT / 1000)) C" >>"$log"
    fi
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$@"
    local rc=$?
    hotspot
    echo "hotspot after: $((HOT / 1000)) C" >>"$log"
    if ((HOT >= HOT_STOP)); then
        sum "- STOPPED: hotspot reached $((HOT / 1000)) C after $*"
        exit 3
    fi
    return "$rc"
}

# run_step <heavy> <label> <log name> <command...>: one job, its verdicts into the summary
run_step() {
    local heavy=$1 label=$2 name=$3 rc v
    shift 3
    if [[ ! -x $1 ]]; then
        sum "- $label: not built"
        FAILED=1
        return
    fi
    gpu_job "$heavy" "$RUN_DIR/$name.log" "$@"
    rc=$?
    v=$(grep -h "^VERDICT" "$RUN_DIR/$name.log" | tr '\n' ' ')
    if ((rc == 0)); then
        sum "- $label: ok. $v"
    else
        sum "- $label: FAILED (exit $rc). ${v:-no verdict} (see $name.log)"
        FAILED=1
    fi
}

cpu_step() {
    local label=$1 name=$2 rc v
    shift 2
    nice -n 10 "$@" >"$RUN_DIR/$name.log" 2>&1
    rc=$?
    v=$(grep -h "^VERDICT" "$RUN_DIR/$name.log" | tr '\n' ' ')
    if ((rc == 0)); then
        sum "- $label: ok. $v"
    else
        sum "- $label: FAILED (exit $rc). ${v:-no verdict} (see $name.log)"
        FAILED=1
    fi
}

phase_cpu() {
    sum "## CPU self-tests of the references"
    cpu_step "cvt_refcheck (E4M3/E5M2 reference vs llama.cpp, FP16, nearest-value search)" refcheck "$BIN/cvt_refcheck"
    cpu_step "cvt_dot selftest (dot rules vs fmaf and double)" dot-selftest env HIP_VISIBLE_DEVICES=-1 "$BIN/cvt_dot" selftest
}

phase_run() {
    local f s
    sum "## run (device index $HIP_INDEX from $CARD_ENV)"
    run_step 0 "cvt_exact decode (FP8/BF8 to FP32, all codes, selects, pairs)" exact-decode "$BIN/cvt_exact" decode
    for f in fp8 bf8; do
        for s in plain ovfl rup rdn rtz noden ovfl_rtz; do
            run_step 1 "cvt_exact encode $f $s (all 2^32)" "exact-encode-$f-$s" "$BIN/cvt_exact" encode "$f" "$s"
        done
    done
    for f in fp8 bf8; do
        run_step 1 "cvt_sr rule $f" "sr-rule-$f" "$BIN/cvt_sr" rule "$f"
        for s in plain ovfl rtz noden; do
            run_step 1 "cvt_sr sweep $f $s (all 2^32)" "sr-sweep-$f-$s" "$BIN/cvt_sr" sweep "$f" "$s" 4
        done
    done
    run_step 1 "cvt_widen (all 2^32 words, methods a-e)" widen "$BIN/cvt_widen"
    run_step 0 "cvt_dot run (targeted + random, seed 1000)" dot-run "$BIN/cvt_dot" run
    run_step 0 "cvt_dot run, fresh sample (seed 777)" dot-run-777 "$BIN/cvt_dot" run --seed 777
    for s in rtz rup rdn noden ovfl; do
        run_step 0 "cvt_dot mode $s" "dot-mode-$s" "$BIN/cvt_dot" mode "$s"
    done
    run_step 0 "cvt_time (cycles, one wave)" time "$BIN/cvt_time"
    for s in f32f16 f16f32 bf16 pk mix; do
        run_step 1 "cvt_f16 $s (FP16/BF16 conversions, packed math, fma_mix)" "f16-$s" "$BIN/cvt_f16" "$s"
    done
}

# break_step <heavy> <label> <name> <command...>: must exit 1 with a FAIL verdict; heavy -1 runs on the CPU only
break_step() {
    local heavy=$1 label=$2 name=$3 rc v log
    shift 3
    log=$RUN_DIR/$name.log
    if [[ ! -x $1 ]]; then
        sum "- $label: not built"
        FAILED=1
        return
    fi
    if ((heavy < 0)); then
        # CPU only: no GPU is made visible
        HIP_VISIBLE_DEVICES=-1 nice -n 10 "$@" >"$log" 2>&1
    else
        gpu_job "$heavy" "$log" "$@"
    fi
    rc=$?
    v=$(grep -h "^VERDICT.*FAIL" "$log" | head -3 | tr '\n' ' ')
    if ((rc == 1)) && [[ -n $v ]]; then
        sum "- $label: FAIL as intended. $v"
    else
        sum "- $label: did NOT fail (exit $rc). $(grep -h '^VERDICT' "$log" | head -3 | tr '\n' ' ')"
        FAILED=1
    fi
}

phase_breaks() {
    sum "## deliberate breaks (each must FAIL)"
    break_step -1 "cvt_refcheck --break (reference rounds ties away)" break-refcheck "$BIN/cvt_refcheck" --break
    break_step 1 "cvt_exact encode fp8 plain --break (GPU reference rounds ties away)" break-exact \
        "$BIN/cvt_exact" encode fp8 plain --break
    break_step 1 "cvt_widen --break (one LDS table entry wrong)" break-widen "$BIN/cvt_widen" --break
    break_step 0 "cvt_dot run --break (dot model rounded once more: products split over two adder steps)" break-dot \
        "$BIN/cvt_dot" run --break
    break_step -1 "cvt_dot selftest --break (sequential rule with the additions reversed)" break-dot-selftest \
        "$BIN/cvt_dot" selftest --break
    break_step 1 "cvt_sr sweep fp8 plain --break (judged against the guide's rule)" break-sr \
        "$BIN/cvt_sr" sweep fp8 plain 4 --break
    break_step 1 "cvt_f16 --break" break-f16 "$BIN/cvt_f16" all --break
}

sum "# Conversions and dot products (lane 6), $(date -u +%FT%TZ)"
sum "- card $ARCH, PCI $PCI_BUS; kernel $(uname -r); LACT $(grep -m1 'current_profile' /etc/lact/config.yaml 2>/dev/null | tr -s ' ')"
sum "- probes from $ROOT; shared headers and checkers from $STUDY_REPO; llama.cpp decoder from $LLAMA_LIB"
case ${1:-all} in
    build) phase_build ;;
    cpu) phase_cpu ;;
    run) phase_run ;;
    breaks) phase_breaks ;;
    all)
        phase_build
        phase_cpu
        phase_run
        phase_breaks
        ;;
    *) die "unknown phase ${1:-}; use build, cpu, run, breaks or all" ;;
esac
sum "- highest hotspot read by this script: $((HOT_MAX / 1000)) C; LACT at the end $(grep -m1 'current_profile' /etc/lact/config.yaml 2>/dev/null | tr -s ' ')"
echo "study-conversions: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
