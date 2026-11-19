#!/usr/bin/env bash
# Lane 1 of notes/STUDY-PLAN.md: build and run the execution-core probes (src/study/core_*.hip), one command.
#
# Usage: scripts/study-core.sh [build|run|breaks|all]   (default all = build, then run, then breaks)
#   build   compiles every probe for gfx1201 into build/study/bin/<probe> (-save-temps output in
#           build/study/tmp-<probe>/), checks every timed loop
#           (scripts/core_loopcheck.py) and records which dual-issue (VOPD) rules the assembler enforces
#   run     runs every probe once, each as its own short GPU job
#   breaks  runs the probes that have a --break switch; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env, written by scripts/first-hour.sh info).
# Results: runs/<UTC time>-core/: one log per step, summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, the card record's device index, not started below 8 GB available memory;
# pp_dpm_sclk and pp_dpm_mclk are logged before and after each job.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-core.sh"
# core_counter reads runs/latest/card.env relative to the working directory
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH

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
    echo "study-core: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-core"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=$REPO/build/study/bin
FAILED=0

sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

# name | source | extra compiler flags. core_share is built twice: CU mode and WGP mode.
PROBES=(
    "core_valu|core_valu.hip|-mllvm -amdgpu-enable-vopd=0"
    "core_dual|core_dual.hip|"
    "core_scalar|core_scalar.hip|-mllvm -amdgpu-enable-vopd=0"
    "core_share_cu|core_share.hip|-DCUMODE=1 -mcumode"
    "core_share_wgp|core_share.hip|-DCUMODE=0"
    "core_flow|core_flow.hip|-mllvm -amdgpu-enable-vopd=0"
    "core_counter|core_counter.hip|"
)
# Probes whose timed loops are inline-assembly blocks built at U = 4, 6 and 8 (checked by core_loopcheck.py).
LOOPCHECKED=" core_valu core_dual core_scalar core_share_cu core_share_wgp core_flow "
# Probes with a --break switch.
BREAKABLE=(core_valu core_dual core_share_cu core_flow core_counter)

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
    local p name src flags pids=() rc lc
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
        # at most 7 compilers at once (machine rule: builds use at most 8 cores)
        if ((${#pids[@]} >= 7)); then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
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
        if [[ $LOOPCHECKED == *" $name "* ]]; then
            python3 "$REPO/scripts/core_loopcheck.py" "$REPO/build/study/tmp-$name"/*.lto.s >"$RUN_DIR/loopcheck-$name.txt" 2>&1
            rc=$?
            lc=$(tail -1 "$RUN_DIR/loopcheck-$name.txt")
            if ((rc == 0)); then
                sum "- $name: built; $lc"
            else
                sum "- $name: built; $lc; LOOP CHECK FAILED (see loopcheck-$name.txt)"
                FAILED=1
            fi
        else
            sum "- $name: built (no inline-assembly timed loops to check; read its .lto.s in build/study/tmp-$name)"
        fi
    done
    vopd_rules
}

# Which VOPD rules the assembler itself enforces. Each line: expected outcome, then the instruction.
vopd_rules() {
    local log=$RUN_DIR/vopd-assembler.txt line want text out got
    [[ -x $ROCM_PATH/llvm/bin/llvm-mc ]] || {
        sum "- VOPD assembler rules: llvm-mc not found in $ROCM_PATH/llvm/bin; FAILED"
        FAILED=1
        return
    }
    local cases=(
        "accept|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v3, v4"
        "reject|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v5, v4"
        "reject|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v3, v6"
        "accept|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v1, v4"
        "reject|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v42, v3, v4"
        "reject|v_dual_fmamk_f32 v40, v1, 0x3f800000, v2 :: v_dual_fmamk_f32 v41, v3, 0x3f800000, v4"
        "accept|v_dual_fmamk_f32 v40, v1, 0x3f800000, v2 :: v_dual_fmamk_f32 v41, v4, 0x3f800000, v5"
        "reject|v_dual_fmaak_f32 v40, v1, v2, 0x3f800000 :: v_dual_fmaak_f32 v41, v3, v4, 0x40000000"
        "reject|v_dual_add_nc_u32 v40, v1, v2 :: v_dual_add_f32 v41, v3, v4"
        "reject|v_dual_fma_mix_f32 v40, v1, v2, v40 :: v_dual_add_f32 v41, v3, v4"
        "unknown|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v40, v4"
        "unknown|v_dual_mul_f32 v40, v1, v2 :: v_dual_mul_f32 v41, v3, v40"
    )
    : >"$log"
    local agree=0 total=0
    for line in "${cases[@]}"; do
        want=${line%%|*}
        text=${line#*|}
        if out=$(echo "$text" | "$ROCM_PATH/llvm/bin/llvm-mc" -arch=amdgcn -mcpu=gfx1201 -show-encoding 2>&1); then
            got=accept
        else
            got=reject
        fi
        echo "$got (expected $want): $text" >>"$log"
        grep "error" <<<"$out" | head -1 | sed 's/^/    /' >>"$log"
        if [[ $want != unknown ]]; then
            total=$((total + 1))
            [[ $want == "$got" ]] && agree=$((agree + 1))
        fi
    done
    sum "- VOPD assembler rules: $agree of $total cases as expected (vopd-assembler.txt)"
    ((agree == total)) || FAILED=1
}

# gpu_job <log> <command...>
gpu_job() {
    local log=$1
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$@"
}

phase_run() {
    local p name src flags log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV)"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        log=$RUN_DIR/$name.log
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name: not built"
            FAILED=1
            continue
        fi
        gpu_job "$log" "$BIN/$name"
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 0)); then
            sum "- $name: ok. ${v}"
        else
            sum "- $name: FAILED (exit $rc). ${v:-no verdict} (see $name.log)"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local name log rc v
    sum "## deliberate breaks (each must FAIL)"
    for name in "${BREAKABLE[@]}"; do
        log=$RUN_DIR/$name-break.log
        if [[ ! -x $BIN/$name ]]; then
            sum "- $name --break: not built"
            FAILED=1
            continue
        fi
        gpu_job "$log" "$BIN/$name" --break
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | tr '\n' ' ')
        if ((rc == 1)) && grep -q "^VERDICT [a-z_]*: FAIL" "$log"; then
            sum "- $name --break: FAIL as intended. $v"
        else
            sum "- $name --break: did NOT fail (exit $rc). ${v:-no verdict}"
            FAILED=1
        fi
    done
}

sum "# Execution-core study run, $(date -u +%FT%TZ)"
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
echo "study-core: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
