#!/usr/bin/env bash
# Leftovers C, lane B (notes/STUDY-PLAN.md; measurements/leftovers-c-b.md): matrix-unit rules, scheduling and
# preemption, transposing and scalar loads, the Vulkan view, and the CPU analyses of the arithmetic files.
#
# Usage: scripts/study-leftovers-c-b.sh [build|run|breaks|all] [ITEM...]   (default all = build, run, breaks)
#   build   compiles the probes for gfx1201 into build/lcb/ (-save-temps output in build/lcb/tmp-<probe>/), the
#           Vulkan program and shaders into build/lcb/vk, the CPU programs into build/lcb/cpu, and checks the listing
#           of every timed region named in LISTING_CHECKS
#   run     every job of the selected items (item IDs as in the write-up: SC-11, MA-2, ...), each its own job
#   breaks  each deliberate break of the selected items; each must exit non-zero with a FAIL verdict
# Environment: STUDY_REPO (default: this folder's parent) supplies src/harness.h and the shared headers, and the
# unchanged probe sources when this folder lacks them; CARD_ENV (default $STUDY_REPO/runs/latest/card.env);
# SC_RUNS (default $STUDY_REPO/runs) holds the kept runs SC-3 and SC-4 analyse (read only);
# LCB_BIN (default build/lcb) is the folder for binaries and listings;
# LCB_GATE_S sets how long a whole-card job waits for the 55 C gate (default 300 s);
# LCB_RUN_DIR reuses one run folder across phases; LCB_NOCARD=1 skips the whole-card jobs (reported NOT RUN), so
# the others need not wait for the thermal gate.
# Results: runs/<UTC time>-leftc-b/: one log per job, summary.md with every VERDICT line. Exit status 1 if anything
# failed.
# GPU jobs: the shared lock, nice -n 10, a time limit, HIP device 0, not started below 8 GB available memory, not
# started unless `lact cli profile get` reads "Testing"; clocks, reported power and hotspot logged before and after.
# Whole-card jobs (kind "card") also wait, outside the lock, for a hotspot below 55 C (500 ms polls, give up after
# 300 s: the job is then NOT RUN) and the probes stop their load at 95 C (90 C for SC-5) from an in-process sampler.
# A hotspot of 95 C or more after any job stops all further GPU work. CPU jobs: nice, at most 8 threads.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-leftovers-c-b.sh"
STUDY_REPO=${STUDY_REPO:-$ROOT}
cd "$ROOT" || exit 2
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=$HOME/.local/opt/rocm
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=0 CARD_ARCH=gfx1201

snap() {
    local drm=$1 hw=$2
    echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')" \
        " power(reported, scaled down by the power-limit mod): $(($(cat "$hw/power1_average") / 1000000)) W" \
        " hotspot: $(($(cat "$hw/temp2_input") / 1000)) C  fan: $(cat "$hw/fan1_input" 2>/dev/null) rpm"
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
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-120}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-120} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

# SC-11: two processes started together inside one locked job (as study-scheduling.sh's pair mode). Arguments:
# B's stream priority (normal|high), output prefix, extra arguments for both.
if [[ ${1:-} == __preempt_pair ]]; then
    prio=$2
    out=$3
    shift 3
    start=$(($(date +%s%3N) + 3000))
    "${LCB_BIN:-$ROOT/build/lcb}/sched_share" preempt --role A --start-ms "$start" --out "$out" "$@" >"$out-A.log" 2>&1 &
    pa=$!
    "${LCB_BIN:-$ROOT/build/lcb}/sched_share" preempt --role B --prio "$prio" --start-ms "$start" --out "$out" "$@" >"$out-B.log" 2>&1 &
    pb=$!
    wait "$pa"
    ra=$?
    wait "$pb"
    rb=$?
    cat "$out-A.log" "$out-B.log"
    python3 "$ROOT/scripts/sched_pair.py" --preempt "$out-A.csv" "$out-B.csv"
    rp=$?
    ((ra == 0 && rb == 0 && rp == 0)) || exit 1
    exit 0
fi

# SC-11, one process: A on the lowest- and B on the highest-priority stream. Argument: output prefix.
if [[ ${1:-} == __preempt_same ]]; then
    out=$2
    "${LCB_BIN:-$ROOT/build/lcb}/sched_share" preempt --role AB --out "$out" || exit 1
    python3 "$ROOT/scripts/sched_pair.py" --preempt "$out-AB.csv" || exit 1
    exit 0
fi

die() {
    echo "study-leftovers-c-b: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$STUDY_REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV (set STUDY_REPO or CARD_ENV)"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
[[ ${HIP_INDEX:-} == 0 ]] || die "card record says HIP_INDEX=${HIP_INDEX:-unset}; this study runs on device 0 only"
export DRM_DEVICE HWMON

SC_RUNS=${SC_RUNS:-$STUDY_REPO/runs}
RUN_DIR=${LCB_RUN_DIR:-"$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-leftc-b"}
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
BIN=${LCB_BIN:-$ROOT/build/lcb}
FAILED=0
STOPPED=0
HOTTEST=0
CARD_BLOCKED=0

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
    timeout 20 lact cli profile get 2>/dev/null | tr -d '[:space:]'
}

# Source of a probe: this folder's copy if it has one, else the study repository's.
src_of() {
    if [[ -f $ROOT/$1 ]]; then
        echo "$ROOT/$1"
    else
        echo "$STUDY_REPO/$1"
    fi
}

# Waits outside the lock for a hotspot below 55 C; returns 1 after LCB_GATE_S seconds (default 300).
cool_gate() {
    local waited=0
    while (($(hotspot_c) >= 55)); do
        ((waited >= 2 * ${LCB_GATE_S:-300})) && return 1
        sleep 0.5
        waited=$((waited + 1))
    done
    return 0
}

# Runs one GPU job. Arguments: log name, kind (gpu|card), timeout seconds, command...
gpu_job() {
    local name=$1 kind=$2 tmo=$3 log h prof
    shift 3
    log=$RUN_DIR/$name.log
    if ((STOPPED)); then
        echo "NOT RUN: GPU work stopped earlier" >"$log"
        return 10
    fi
    prof=$(lact_profile)
    if [[ $prof != Testing ]]; then
        echo "NOT RUN: LACT profile is '$prof', not 'Testing'; GPU work stopped" >"$log"
        STOPPED=1
        return 6
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT RUN: $(avail_gb) GB available (floor 8)" >"$log"
        return 5
    fi
    {
        echo "# $name ($kind): $*"
        echo "# start $(date -u +%Y-%m-%dT%H:%M:%SZ) profile=$prof"
    } >"$log"
    if [[ $kind == card ]]; then
        if [[ -n ${LCB_NOCARD:-} ]]; then
            echo "NOT RUN: whole-card job deferred (LCB_NOCARD set)" >>"$log"
            return 11
        fi
        if ((CARD_BLOCKED)); then
            echo "NOT RUN: the 55 C gate did not open for an earlier whole-card job of this run" >>"$log"
            return 7
        fi
        if ! cool_gate; then
            echo "NOT RUN: hotspot still $(hotspot_c) C after ${LCB_GATE_S:-300} s (start limit 55 C)" >>"$log"
            CARD_BLOCKED=1
            return 7
        fi
        echo "# hotspot $(hotspot_c) C at the gate" >>"$log"
    fi
    JOB_TIMEOUT=$tmo flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
    local rc=$?
    echo "# end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >>"$log"
    h=$(hotspot_c)
    ((h > HOTTEST)) && HOTTEST=$h
    # sched_trace prints "HOTTEST <n> C"; mat_card, lds_tr and vkstudy print "THERMAL highest ... <n> C"
    if grep -qE "^(HOTTEST|THERMAL highest)" "$log"; then
        local hh
        hh=$(grep -hE "^(HOTTEST|THERMAL highest)" "$log" | awk '{printf "%d\n", $(NF-1)}' | sort -n | tail -1)
        ((hh > HOTTEST)) && HOTTEST=$hh
    fi
    if ((h >= 95)); then
        sum "- hotspot $h C after $name: all further GPU work stopped"
        STOPPED=1
        FAILED=1
    fi
    return "$rc"
}

cpu_job() {
    local name=$1 tmo=$2 log
    shift 2
    log=$RUN_DIR/$name.log
    echo "# $name (cpu): $*" >"$log"
    echo "# start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 "$tmo" "$@" >>"$log" 2>&1
    local rc=$?
    echo "# end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >>"$log"
    return "$rc"
}

# probe name | source (relative) | extra hipcc flags
PROBES=(
    "sched_share|src/study/sched_share.hip|-mcumode"
    "left_mat|src/study/left_mat.hip|"
    "left_mat64|src/study/left_mat64.hip|-mwavefrontsize64"
    "lds_tr|src/study/lds_tr.hip|"
    "lds_smem|src/study/lds_smem.hip|"
    "lds_smem_cu|src/study/lds_smem.hip|-mcumode"
    "tim_mat|src/study/tim_mat.hip|-mcumode"
    "mat_card|src/study/mat_card.hip|-mcumode"
    "sched_trace|src/study/sched_trace.hip|-mcumode"
    "left_rot|src/study/left_rot.hip|"
)

# CPU programs (g++, as scripts/study-matrix-arith.sh builds them)
CPU_PROBES=(arith_realw arith_stats arith_qworst)
# EQ-1a: real E4M3 tensors (read only) of four layers, three projections each
REALW_MODEL=${REALW_MODEL:-${LLM_MODELS:-$HOME/Projects/rdna3-fp8/models}/llama32-3b-fp8-outq8.gguf}
REALW_TENSORS=(blk.0.attn_q blk.0.ffn_gate blk.0.ffn_down blk.10.attn_q blk.10.ffn_gate blk.10.ffn_down
    blk.20.attn_q blk.20.ffn_gate blk.20.ffn_down blk.27.attn_q blk.27.ffn_gate blk.27.ffn_down)

build_cpu() {
    local name log t
    mkdir -p "$BIN/cpu" "$BIN/realw"
    for name in "${CPU_PROBES[@]}"; do
        selected_probe "$name" || continue
        log=$RUN_DIR/build-$name.log
        if nice -n 10 g++ -O2 -std=c++17 -Wall -pthread -I"$ROOT/src/study" -I"$STUDY_REPO/src/study" \
            "$(src_of "src/study/$name.cpp")" -o "$BIN/cpu/$name" >"$log" 2>&1; then
            sum "- $name: built (CPU)"
        else
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        fi
    done
    selected_probe arith_realw || return 0
    if [[ ! -f $REALW_MODEL ]]; then
        sum "- real weight rows: $REALW_MODEL not found; EQ-1a cannot run"
        FAILED=1
        return 0
    fi
    for t in "${REALW_TENSORS[@]}"; do
        log=$RUN_DIR/build-realw-$t.log
        if nice -n 10 python3 -I "$(src_of scripts/gguf_rows.py)" "$REALW_MODEL" extract "$t.weight" \
            "$BIN/realw/$t.bin" 3072 >"$log" 2>&1; then
            sum "- real weight rows $t: $(tail -1 "$log")"
        else
            sum "- real weight rows $t: extraction FAILED (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

# MT-7: left_prio's kernels as two code objects from one listing, the second with the round-robin bit set in every
# kernel descriptor (the method of scripts/study-leftovers-matrix.sh).
build_prio() {
    local log=$RUN_DIR/build-left_prio.log llvm=$ROCM_PATH/llvm/bin pdir=$BIN/prio n v k
    selected_probe left_prio || return 0
    mkdir -p "$pdir"
    if (cd "$pdir" &&
        nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -DHARNESS_ARCH='"gfx1201"' \
            -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" "$(src_of src/study/left_prio.hip)" -o "$BIN/left_prio" -lamdhip64 &&
        nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -DHARNESS_ARCH='"gfx1201"' \
            -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -DLEFT_PRIO_DEVICE_ONLY --cuda-device-only -S \
            "$(src_of src/study/left_prio.hip)" -o left_prio.s &&
        sed 's/\.amdhsa_round_robin_scheduling 0/.amdhsa_round_robin_scheduling 1/' left_prio.s >left_prio_rr.s &&
        for v in left_prio left_prio_rr; do
            "$llvm/clang" --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -c "$v.s" -o "$v.o" &&
                "$llvm/ld.lld" -shared "$v.o" -o "$BIN/$v.hsaco" || exit 1
        done) >"$log" 2>&1; then
        k=$(grep -c '^\s*\.amdhsa_kernel ' "$pdir/left_prio.s")
        n=$("$llvm/llvm-objdump" -D --mcpu=gfx1201 -j .rodata "$BIN/left_prio_rr.hsaco" | grep -c 'round_robin_scheduling 1')
        v=$("$llvm/llvm-objdump" -D --mcpu=gfx1201 -j .rodata "$BIN/left_prio.hsaco" | grep -c 'round_robin_scheduling 0')
        if [[ $n == "$k" && $v == "$k" && $k -ge 21 ]]; then
            sum "- left_prio: built; round-robin bit set in $n of $k descriptors of left_prio_rr.hsaco, clear in $v of $k of left_prio.hsaco"
        else
            sum "- left_prio: descriptor check FAILED (set $n, clear $v of $k)"
            FAILED=1
        fi
    else
        sum "- left_prio: build FAILED (see build-left_prio.log)"
        FAILED=1
    fi
}

# Vulkan: shader variants (spv name | source | glslc defines) and the host program, as scripts/study-vulkan.sh builds
# them; vkh.c and the shaders this lane did not change come from the study repository.
VK_SHADERS=(
    "mat_f16|mat.comp|-DT_F16 -DLOADAB"
    "mat_bf16|mat.comp|-DT_BF16 -DLOADAB"
    "mat_fp8|mat.comp|-DT_FP8 -DLOADAB"
    "mat_fp8bf8|mat.comp|-DT_FP8BF8 -DLOADAB"
    "mat_bf8fp8|mat.comp|-DT_BF8FP8 -DLOADAB"
    "mat_bf8|mat.comp|-DT_BF8 -DLOADAB"
    "mat_i8sat|mat.comp|-DT_I8SAT -DLOADAB"
    "load_f16_row|load.comp|"
    "load_fp8_row|load.comp|-DT_FP8"
    "load_f16_row_pipe|load.comp|-DPIPE"
    "load_fp8_row_pipe|load.comp|-DT_FP8 -DPIPE"
    "load_f16_row_pipebrk|load.comp|-DPIPE -DPIPE_BREAK"
    "load_f16_row_extra8|load.comp|-DEXTRA=8"
    "check_big_f16|check_big.comp|"
    "check_big_bf16|check_big.comp|-DT_BF16"
    "check_big_fp8|check_big.comp|-DT_FP8"
)
build_vk() {
    local v name src defs log=$RUN_DIR/build-vk.log rc=0
    selected_probe vkstudy || return 0
    mkdir -p "$BIN/spv"
    for v in "${VK_SHADERS[@]}"; do
        IFS='|' read -r name src defs <<<"$v"
        # shellcheck disable=SC2086 # defs is a word list on purpose
        if ! nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp -I"$STUDY_REPO/src/vk/shaders" $defs \
            "$(src_of "src/vk/shaders/$src")" -o "$BIN/spv/$name.spv" >>"$log" 2>&1 ||
            ! spirv-val --target-env vulkan1.3 "$BIN/spv/$name.spv" >>"$log" 2>&1; then
            sum "- shader $name: FAILED (see build-vk.log)"
            rc=1
        fi
    done
    if nice -n 10 glslc --target-env=vulkan1.3 -O -fshader-stage=comp "$(src_of src/vk/shaders/vksched_one.comp)" \
        -o "$BIN/spv/vksched_one.spv" >>"$log" 2>&1 && nice -n 10 cc -O2 -std=c11 -Wall -Wextra -o "$BIN/vksched" \
        "$(src_of src/vk/vksched.c)" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- vksched: built"
    else
        sum "- vksched: build FAILED (see build-vk.log)"
        rc=1
    fi
    if nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$STUDY_REPO/src/vk" -o "$BIN/vkstudy" \
        "$(src_of src/vk/vkstudy.c)" "$(src_of src/vk/vkh.c)" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- vkstudy: built ($(grep -c warning "$log") compiler warnings); ${#VK_SHADERS[@]} shader variants"
    else
        sum "- vkstudy: build FAILED (see build-vk.log)"
        rc=1
    fi
    ((rc)) && FAILED=1
    return 0
}

build_one() {
    local name=$1 src=$2 flags=$3 dir
    dir=$BIN/tmp-$name
    mkdir -p "$dir"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' $flags \
        -I"$ROOT/src/study" -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -save-temps "$(src_of "$src")" \
        -o "$BIN/$name" -lamdhip64) >"$RUN_DIR/build-$name.log" 2>&1
}

phase_build() {
    local p name src flags running
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN"
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        selected_probe "$name" || continue
        # at most 8 cores: 4 builds at a time
        while running=$(jobs -rp | wc -l) && ((running >= 4)); do
            wait -n
        done
        build_one "$name" "$src" "$flags" &
    done
    wait
    build_cpu
    build_prio
    build_vk
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags <<<"$p"
        selected_probe "$name" || continue
        if [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
        else
            sum "- $name: built ($src${flags:+, $flags})"
        fi
    done
    listing_checks
}

# Listing checks on the final code object (llvm-objdump): probe | extra objdump attributes | leftc_listing.py command |
# its arguments separated by |. Every timed region and every back-to-back pair is one inline-assembly statement; these
# confirm that the code object holds them whole.
LISTING_CHECKS=(
    "left_mat||pairs|v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[24:31]::v_wmma_f16_16x16x16_f16 v[40:43], v[0:3], v[32:35], v[40:43]|v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[24:31]::v_wmma_f16_16x16x16_f16 v[40:43], v[32:35], v[0:3], v[40:43]|v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[16:17], v[24:31]::v_wmma_f32_16x16x16_f16 v[40:47], v[0:3], v[32:35], v[40:47]|v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[16:17], v[24:31]::v_wmma_f32_16x16x16_f16 v[40:47], v[32:35], v[0:3], v[40:47]|v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]::v_wmma_bf16_16x16x16_bf16 v[40:43], v[0:3], v[32:35], v[40:43]|v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[16:17], v[24:31] neg_lo:[1,1,0]::v_wmma_bf16_16x16x16_bf16 v[40:43], v[32:35], v[0:3], v[40:43]|v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]::v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[0:1], v[32:33], v[40:47]|v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]::v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[32:33], v[0:1], v[40:47]"
    "left_mat||runs|v_wmma_f32_16x16x16_f16:48|v_wmma_f32_16x16x16_f16:16|v_wmma_f32_16x16x16_bf16:48|v_wmma_f32_16x16x16_bf16:16|v_wmma_f16_16x16x16_f16:48|v_wmma_f16_16x16x16_f16:16|v_wmma_bf16_16x16x16_bf16:48|v_wmma_bf16_16x16x16_bf16:16|v_wmma_f32_16x16x16_fp8_fp8:48|v_wmma_f32_16x16x16_fp8_fp8:16|v_wmma_f32_16x16x16_fp8_bf8:48|v_wmma_f32_16x16x16_fp8_bf8:16|v_wmma_f32_16x16x16_bf8_fp8:48|v_wmma_f32_16x16x16_bf8_fp8:16|v_wmma_f32_16x16x16_bf8_bf8:48|v_wmma_f32_16x16x16_bf8_bf8:16|v_wmma_i32_16x16x16_iu8:48|v_wmma_i32_16x16x16_iu8:16|v_wmma_i32_16x16x16_iu4:48|v_wmma_i32_16x16x16_iu4:16|v_wmma_i32_16x16x32_iu4:48|v_wmma_i32_16x16x32_iu4:16|v_swmmac_f32_16x16x32_f16:48|v_swmmac_f32_16x16x32_f16:16|v_swmmac_f32_16x16x32_bf16:48|v_swmmac_f32_16x16x32_bf16:16|v_swmmac_f16_16x16x32_f16:48|v_swmmac_f16_16x16x32_f16:16|v_swmmac_bf16_16x16x32_bf16:48|v_swmmac_bf16_16x16x32_bf16:16|v_swmmac_f32_16x16x32_fp8_fp8:48|v_swmmac_f32_16x16x32_fp8_fp8:16|v_swmmac_f32_16x16x32_bf8_bf8:48|v_swmmac_f32_16x16x32_bf8_bf8:16|v_swmmac_i32_16x16x32_iu8:48|v_swmmac_i32_16x16x32_iu8:16|v_swmmac_i32_16x16x32_iu4:48|v_swmmac_i32_16x16x32_iu4:16|v_swmmac_i32_16x16x64_iu4:48|v_swmmac_i32_16x16x64_iu4:16"
    "left_mat64|+wavefrontsize64|pairs|v_wmma_f16_16x16x16_f16 v[0:1], v[8:9], v[16:17], v[24:25]::v_wmma_f32_16x16x16_f16 v[40:43], v[0:1], v[32:33], v[40:43]|v_wmma_f16_16x16x16_f16 v[0:1], v[8:9], v[16:17], v[24:25]::v_wmma_f32_16x16x16_f16 v[40:43], v[32:33], v[0:1], v[40:43]|v_wmma_i32_16x16x16_iu8 v[0:3], v8, v16, v[24:27] neg_lo:[1,1,0]::v_wmma_i32_16x16x16_iu8 v[40:43], v0, v32, v[40:43] neg_lo:[1,1,0]|v_wmma_i32_16x16x16_iu8 v[0:3], v8, v16, v[24:27] neg_lo:[1,1,0]::v_wmma_i32_16x16x16_iu8 v[40:43], v32, v0, v[40:43] neg_lo:[1,1,0]"
    "lds_tr||runs|global_load_tr_b128:56|global_load_b128:56|global_load_tr_b64:72|global_load_tr_b64:64|global_load_tr_b64:63|global_load_b64:64|global_load_b64:63"
    "tim_mat||count|v_swmmac_f32_16x16x32_f16:1024|v_wmma_f32_16x16x16_bf16:1024|v_wmma_i32_16x16x16_iu8:1024|v_wmma_f16_16x16x16_f16:1024|v_wmma_f32_16x16x16_bf8_bf8:1024"
    "left_rot||count|v_wmma_f32_16x16x16_f16:3072|v_wmma_f32_16x16x16_fp8_fp8:480"
    "mat_card||runs|v_swmmac_f32_16x16x32_bf16:16|v_swmmac_f32_16x16x32_bf8_bf8:16|v_wmma_f32_16x16x16_bf8_bf8:16|v_wmma_f16_16x16x16_f16:16|v_wmma_bf16_16x16x16_bf16:16|v_swmmac_f16_16x16x32_f16:16|v_wmma_f32_16x16x16_f16:16"
    "left_mat64|+wavefrontsize64|runs|v_wmma_f32_16x16x16_f16:48|v_wmma_f32_16x16x16_f16:16|v_wmma_f32_16x16x16_bf16:48|v_wmma_f32_16x16x16_bf16:16|v_wmma_f16_16x16x16_f16:48|v_wmma_f16_16x16x16_f16:16|v_wmma_bf16_16x16x16_bf16:48|v_wmma_bf16_16x16x16_bf16:16|v_wmma_f32_16x16x16_fp8_fp8:48|v_wmma_f32_16x16x16_fp8_fp8:16|v_wmma_f32_16x16x16_fp8_bf8:48|v_wmma_f32_16x16x16_fp8_bf8:16|v_wmma_f32_16x16x16_bf8_fp8:48|v_wmma_f32_16x16x16_bf8_fp8:16|v_wmma_f32_16x16x16_bf8_bf8:48|v_wmma_f32_16x16x16_bf8_bf8:16|v_wmma_i32_16x16x16_iu8:48|v_wmma_i32_16x16x16_iu8:16|v_wmma_i32_16x16x16_iu4:48|v_wmma_i32_16x16x16_iu4:16|v_wmma_i32_16x16x32_iu4:48|v_wmma_i32_16x16x32_iu4:16|v_swmmac_f32_16x16x32_f16:48|v_swmmac_f32_16x16x32_f16:16|v_swmmac_f32_16x16x32_bf16:48|v_swmmac_f32_16x16x32_bf16:16|v_swmmac_f16_16x16x32_f16:48|v_swmmac_f16_16x16x32_f16:16|v_swmmac_bf16_16x16x32_bf16:48|v_swmmac_bf16_16x16x32_bf16:16|v_swmmac_f32_16x16x32_fp8_fp8:48|v_swmmac_f32_16x16x32_fp8_fp8:16|v_swmmac_f32_16x16x32_bf8_bf8:48|v_swmmac_f32_16x16x32_bf8_bf8:16|v_swmmac_i32_16x16x32_iu8:48|v_swmmac_i32_16x16x32_iu8:16|v_swmmac_i32_16x16x32_iu4:48|v_swmmac_i32_16x16x32_iu4:16|v_swmmac_i32_16x16x64_iu4:48|v_swmmac_i32_16x16x64_iu4:16"
)

listing_checks() {
    local row probe attr cmd rest img dis
    local -a args
    for row in "${LISTING_CHECKS[@]}"; do
        IFS='|' read -r probe attr cmd rest <<<"$row"
        selected_probe "$probe" || continue
        img=$(find "$BIN/tmp-$probe" -maxdepth 1 -name '*gfx1201.img' | head -1)
        dis=$RUN_DIR/listing-$probe.dis
        "$ROCM_PATH/llvm/bin/llvm-objdump" -d --mcpu=gfx1201 ${attr:+--mattr=$attr} "$img" >"$dis" 2>&1
        IFS='|' read -r -a args <<<"$rest"
        if python3 "$ROOT/scripts/leftc_listing.py" "$cmd" "$dis" "${args[@]}" >"$RUN_DIR/listing-$probe-$cmd.log" 2>&1; then
            sum "- listing $probe $cmd: $(grep -h '^VERDICT' "$RUN_DIR/listing-$probe-$cmd.log")"
        else
            sum "- listing $probe $cmd: FAILED: $(grep -h '^VERDICT' "$RUN_DIR/listing-$probe-$cmd.log")"
            FAILED=1
        fi
    done
}

# item | job name | kind (gpu|card|cpu|pair) | timeout s | command (words; @RUN@ = run folder, @BIN@ = binaries)
JOBS=(
    "SC-11|sc11-pair-normal|pair|120|__preempt_pair normal @RUN@/sc11-normal"
    "SC-11|sc11-pair-high|pair|120|__preempt_pair high @RUN@/sc11-high"
    "SC-11|sc11-same-process|pair|120|__preempt_same @RUN@/sc11-same"
)
JOBS+=(
    "MA-2|ma2-hazard2|gpu|300|@BIN@/left_mat hazard2"
    "MA-2|ma2-hazard2-card|card|300|@BIN@/left_mat hazard2 --card"
    "MA-2|ma2-hazard64|gpu|300|@BIN@/left_mat64 hazard"
    "MA-1|ma1-ma3-mods2|gpu|300|@BIN@/left_mat mods2"
    "CH-6|ch6-mask-w32|gpu|300|@BIN@/left_mat mask2"
    "CH-6|ch6-mask-w64|gpu|300|@BIN@/left_mat64 mask"
    "LD-6a|ld6a-calib|gpu|120|@BIN@/lds_tr calib"
    "LD-6a|ld6a-inflight2|gpu|300|@BIN@/lds_tr inflight2"
    "LD-6b|ld6b-mask|gpu|120|@BIN@/lds_tr mask"
    "LD-7a|ld7a-line|gpu|120|@BIN@/lds_smem line"
    "LD-7b|ld7b-share|gpu|180|@BIN@/lds_smem_cu share"
    "LD-7c|ld7c-sbuf|gpu|120|@BIN@/lds_smem sbuf"
    "MT-7|mt7-levels|gpu|300|@BIN@/left_prio @BIN@/left_prio.hsaco --levels"
    "MT-7|mt7-levels-rr|gpu|300|@BIN@/left_prio @BIN@/left_prio_rr.hsaco --levels --rr"
    "MT-5b|mt5b-handover|gpu|300|@BIN@/tim_mat handover"
    "MT-6|mt6-rot|gpu|300|@BIN@/left_rot"
)
# whole-card bursts: one burst per job, so the 55 C wait happens outside the lock (items 11 and 11a's figures; the
# analysis jobs combine the bursts)
JOBS+=(
    "MT-5c|mt5c-w1-50ms-b1|card|120|@BIN@/mat_card burst f16 --w 1 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w16-50ms-b1|card|120|@BIN@/mat_card burst f16 --w 16 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w1-100ms-b1|card|120|@BIN@/mat_card burst f16 --w 1 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w16-100ms-b1|card|120|@BIN@/mat_card burst f16 --w 16 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w1-200ms-b1|card|120|@BIN@/mat_card burst f16 --w 1 --ms 200 --tag mt5c"
    "MT-5c|mt5c-w16-200ms-b1|card|120|@BIN@/mat_card burst f16 --w 16 --ms 200 --tag mt5c"
    "MT-5c|mt5c-w1-50ms-b2|card|120|@BIN@/mat_card burst f16 --w 1 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w16-50ms-b2|card|120|@BIN@/mat_card burst f16 --w 16 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w1-100ms-b2|card|120|@BIN@/mat_card burst f16 --w 1 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w16-100ms-b2|card|120|@BIN@/mat_card burst f16 --w 16 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w1-200ms-b2|card|120|@BIN@/mat_card burst f16 --w 1 --ms 200 --tag mt5c"
    "MT-5c|mt5c-w16-200ms-b2|card|120|@BIN@/mat_card burst f16 --w 16 --ms 200 --tag mt5c"
    "MT-5c|mt5c-w1-50ms-b3|card|120|@BIN@/mat_card burst f16 --w 1 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w16-50ms-b3|card|120|@BIN@/mat_card burst f16 --w 16 --ms 50 --tag mt5c"
    "MT-5c|mt5c-w1-100ms-b3|card|120|@BIN@/mat_card burst f16 --w 1 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w16-100ms-b3|card|120|@BIN@/mat_card burst f16 --w 16 --ms 100 --tag mt5c"
    "MT-5c|mt5c-w1-200ms-b3|card|120|@BIN@/mat_card burst f16 --w 1 --ms 200 --tag mt5c"
    "MT-5c|mt5c-w16-200ms-b3|card|120|@BIN@/mat_card burst f16 --w 16 --ms 200 --tag mt5c"
    "MT-3|mt3-sbf16-w1-b1|card|120|@BIN@/mat_card burst sbf16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sbf16-w1-b2|card|120|@BIN@/mat_card burst sbf16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sbf16-w16|card|120|@BIN@/mat_card burst sbf16 --w 16 --ms 100 --tag mt3"
    "MT-3|mt3-sbf8-w1-b1|card|120|@BIN@/mat_card burst sbf8 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sbf8-w1-b2|card|120|@BIN@/mat_card burst sbf8 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sbf8-w16|card|120|@BIN@/mat_card burst sbf8 --w 16 --ms 100 --tag mt3"
    "MT-3|mt3-bf8-w1-b1|card|120|@BIN@/mat_card burst bf8 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-bf8-w1-b2|card|120|@BIN@/mat_card burst bf8 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-bf8-w16|card|120|@BIN@/mat_card burst bf8 --w 16 --ms 100 --tag mt3"
    "MT-3|mt3-f16f16-w1-b1|card|120|@BIN@/mat_card burst f16f16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-f16f16-w1-b2|card|120|@BIN@/mat_card burst f16f16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-f16f16-w16|card|120|@BIN@/mat_card burst f16f16 --w 16 --ms 100 --tag mt3"
    "MT-3|mt3-bf16bf16-w1-b1|card|120|@BIN@/mat_card burst bf16bf16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-bf16bf16-w1-b2|card|120|@BIN@/mat_card burst bf16bf16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-bf16bf16-w16|card|120|@BIN@/mat_card burst bf16bf16 --w 16 --ms 100 --tag mt3"
    "MT-3|mt3-sf16f16-w1-b1|card|120|@BIN@/mat_card burst sf16f16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sf16f16-w1-b2|card|120|@BIN@/mat_card burst sf16f16 --w 1 --ms 100 --tag mt3"
    "MT-3|mt3-sf16f16-w16|card|120|@BIN@/mat_card burst sf16f16 --w 16 --ms 100 --tag mt3"
    "LD-9|ld9-tr_b128-shared-512-0|card|120|@BIN@/lds_tr bw2 tr_b128 shared 512"
    "LD-9|ld9-tr_b128-shared-32-1|card|120|@BIN@/lds_tr bw2 tr_b128 shared 32"
    "LD-9|ld9-tr_b128-unique-32-2|card|120|@BIN@/lds_tr bw2 tr_b128 unique 32"
    "LD-9|ld9-b128-shared-32-3|card|120|@BIN@/lds_tr bw2 b128 shared 32"
    "LD-9|ld9-b128-unique-32-4|card|120|@BIN@/lds_tr bw2 b128 unique 32"
    "LD-9|ld9-tr_b128-shared-4-5|card|120|@BIN@/lds_tr bw2 tr_b128 shared 4"
    "LD-9|ld9-tr_b128-unique-4-6|card|120|@BIN@/lds_tr bw2 tr_b128 unique 4"
    "LD-9|ld9-tr_b128-shared-16-7|card|120|@BIN@/lds_tr bw2 tr_b128 shared 16"
    "LD-9|ld9-tr_b128-unique-16-8|card|120|@BIN@/lds_tr bw2 tr_b128 unique 16"
    "LD-9|ld9-tr_b128-shared-64-9|card|120|@BIN@/lds_tr bw2 tr_b128 shared 64"
    "LD-9|ld9-tr_b128-unique-64-10|card|120|@BIN@/lds_tr bw2 tr_b128 unique 64"
    "LD-9|ld9-tr_b128-shared-32-11|card|120|@BIN@/lds_tr bw2 tr_b128 shared 32"
    "LD-9|ld9-tr_b128-unique-32-12|card|120|@BIN@/lds_tr bw2 tr_b128 unique 32"
    "LD-9|ld9-tr_b128-shared-512-13|card|120|@BIN@/lds_tr bw2 tr_b128 shared 512"
    "SC-5|sc5-idle|gpu|60|@BIN@/sched_trace idle --burst-ms 100 --sample-ms 5 --out @RUN@/sc5-idle"
    "SC-5|sc5-fma-b1|card|120|@BIN@/sched_trace fma --burst-ms 100 --sample-ms 5 --out @RUN@/sc5-fma-b1"
    "SC-5|sc5-fma-b2|card|120|@BIN@/sched_trace fma --burst-ms 100 --sample-ms 5 --out @RUN@/sc5-fma-b2"
    "SC-5|sc5-fp8|card|120|@BIN@/sched_trace fp8 --burst-ms 100 --sample-ms 5 --out @RUN@/sc5-fp8"
    "SC-5|sc5-f16|card|120|@BIN@/sched_trace f16 --burst-ms 200 --sample-ms 5 --out @RUN@/sc5-f16"
    "VK-7a|vk7a-fp8-b1|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat fp8 --w 8 --reps 1"
    "VK-7a|vk7a-fp8bf8-b1|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat fp8bf8 --w 8 --reps 1"
    "VK-7a|vk7a-bf8fp8-b1|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat bf8fp8 --w 8 --reps 1"
    "VK-7a|vk7a-bf8-b1|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat bf8 --w 8 --reps 1"
    "VK-7a|vk7a-i8sat-b1|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat i8sat --w 8 --reps 1"
    "VK-7a|vk7a-fp8-b2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat fp8 --w 8 --reps 1"
    "VK-7a|vk7a-fp8bf8-b2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat fp8bf8 --w 8 --reps 1"
    "VK-7a|vk7a-bf8fp8-b2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat bf8fp8 --w 8 --reps 1"
    "VK-7a|vk7a-bf8-b2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat bf8 --w 8 --reps 1"
    "VK-7a|vk7a-i8sat-b2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat i8sat --w 8 --reps 1"
    "VK-3|vk3-f16-l2-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1"
    "VK-3|vk3-f16-l2-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1 --variant pipe"
    "VK-3|vk3-f16-mall-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight mall --w 4 --reps 1"
    "VK-3|vk3-f16-mall-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight mall --w 4 --reps 1 --variant pipe"
    "VK-3|vk3-fp8-l2-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load fp8 row tight l2 --w 4 --reps 1"
    "VK-3|vk3-fp8-l2-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load fp8 row tight l2 --w 4 --reps 1 --variant pipe"
    "VK-3|vk3-fp8-mall-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load fp8 row tight mall --w 4 --reps 1"
    "VK-3|vk3-fp8-mall-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load fp8 row tight mall --w 4 --reps 1 --variant pipe"
    "VK-3|vk3-f16-l2-sg32-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --sg 32 --reps 1"
    "VK-3|vk3-f16-l2-sg32-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --sg 32 --reps 1 --variant pipe"
    "VK-4|vk4-f16-l2-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1"
    "VK-4|vk4-f16-l2-extra8|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1 --variant extra8"
    "VK-4|vk4-f16-l2-sg32-naive|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1 --sg 32"
    "VK-4|vk4-f16-l2-sg32-extra8|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1 --sg 32 --variant extra8"
    "VK-2|vk2-f16|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.15 @BIN@/vkstudy mat f16 --w 8 --reps 1"
    "VK-2|vk2-bf16|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.15 @BIN@/vkstudy mat bf16 --w 8 --reps 1"
    "SC-12|sc12-keepalive|gpu|120|@BIN@/vksched --spv @BIN@/spv/vksched_one.spv --keepalive"
    "VK-5|vk5-check2|gpu|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ @BIN@/vkstudy check2"
    "MT-5c|mt5c-analysis|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py mt5c @RUN@"
    "MT-3|mt3-analysis|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py mt3 @RUN@"
    "LD-9|ld9-analysis|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py ld9 @RUN@"
)
T3=@SCRUNS@/20261117T164432Z-sched
JOBS+=(
    "SC-4|sc4-shift|cpu|120|python3 -I @ROOT@/scripts/sched_pair.py --shift @SCRUNS@/20261117T165519Z-sched/pair-small-small-A.csv @SCRUNS@/20261117T165519Z-sched/pair-small-small-B.csv"
    "SC-3|sc3-throttle|cpu|120|python3 -I @ROOT@/scripts/leftc_throttle.py $T3/trace-f16-sysfs.csv $T3/trace-fp8-sysfs.csv $T3/trace-stream-sysfs.csv $T3/trace-light-sysfs.csv $T3/trace-idle-sysfs.csv"
)
JOBS+=(
    "EQ-1a|eq1a-standin|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.ffn_down.bin"
    "EQ-1a|eq1a-blk.0.attn_q|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.attn_q.bin --real-only"
    "EQ-1a|eq1a-blk.0.attn_q-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.attn_q.bin --real-only --scale"
    "EQ-1a|eq1a-blk.0.ffn_gate|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.ffn_gate.bin --real-only"
    "EQ-1a|eq1a-blk.0.ffn_gate-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.ffn_gate.bin --real-only --scale"
    "EQ-1a|eq1a-blk.0.ffn_down|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.ffn_down.bin --real-only"
    "EQ-1a|eq1a-blk.0.ffn_down-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.ffn_down.bin --real-only --scale"
    "EQ-1a|eq1a-blk.10.attn_q|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.attn_q.bin --real-only"
    "EQ-1a|eq1a-blk.10.attn_q-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.attn_q.bin --real-only --scale"
    "EQ-1a|eq1a-blk.10.ffn_gate|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.ffn_gate.bin --real-only"
    "EQ-1a|eq1a-blk.10.ffn_gate-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.ffn_gate.bin --real-only --scale"
    "EQ-1a|eq1a-blk.10.ffn_down|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.ffn_down.bin --real-only"
    "EQ-1a|eq1a-blk.10.ffn_down-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.10.ffn_down.bin --real-only --scale"
    "EQ-1a|eq1a-blk.20.attn_q|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.attn_q.bin --real-only"
    "EQ-1a|eq1a-blk.20.attn_q-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.attn_q.bin --real-only --scale"
    "EQ-1a|eq1a-blk.20.ffn_gate|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.ffn_gate.bin --real-only"
    "EQ-1a|eq1a-blk.20.ffn_gate-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.ffn_gate.bin --real-only --scale"
    "EQ-1a|eq1a-blk.20.ffn_down|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.ffn_down.bin --real-only"
    "EQ-1a|eq1a-blk.20.ffn_down-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.20.ffn_down.bin --real-only --scale"
    "EQ-1a|eq1a-blk.27.attn_q|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.attn_q.bin --real-only"
    "EQ-1a|eq1a-blk.27.attn_q-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.attn_q.bin --real-only --scale"
    "EQ-1a|eq1a-blk.27.ffn_gate|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.ffn_gate.bin --real-only"
    "EQ-1a|eq1a-blk.27.ffn_gate-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.ffn_gate.bin --real-only --scale"
    "EQ-1a|eq1a-blk.27.ffn_down|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.ffn_down.bin --real-only"
    "EQ-1a|eq1a-blk.27.ffn_down-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.27.ffn_down.bin --real-only --scale"
    "EQ-3|eq3-quantbf16|cpu|1200|@BIN@/cpu/arith_stats quantbf16"
    "EQ-2|eq2-nosub|cpu|1800|@BIN@/cpu/arith_qworst --nosub"
    "EQ-2|eq2-k1024|cpu|1800|@BIN@/cpu/arith_qworst --k1024"
    "EQ-2|eq2-nonfinite|cpu|1200|@BIN@/cpu/arith_qworst --nonfinite"
)
BREAKS=(
    "SC-11|break-sc11-calib|cpu|60|python3 @ROOT@/scripts/sched_pair.py --preempt @RUN@/sc11-normal-A.csv @RUN@/sc11-normal-B.csv --calib-us 1"
    "MA-2|break-ma2-hazard2|gpu|300|@BIN@/left_mat hazard2 --break"
    "MA-2|break-ma2-hazard64|gpu|300|@BIN@/left_mat64 hazard --break"
    "MA-1|break-ma1-ma3-mods2|gpu|300|@BIN@/left_mat mods2 --break"
    "CH-6|break-ch6-mask-w32|gpu|300|@BIN@/left_mat mask2 --break"
    "CH-6|break-ch6-mask-w64|gpu|300|@BIN@/left_mat64 mask --break"
    "LD-6a|break-ld6a-inflight2|gpu|300|@BIN@/lds_tr inflight2 --break"
    "LD-6b|break-ld6b-mask|gpu|120|@BIN@/lds_tr mask --break"
    "LD-7a|break-ld7a-line|gpu|120|@BIN@/lds_smem line --break"
    "LD-7b|break-ld7b-share|gpu|180|@BIN@/lds_smem_cu share --break"
    "LD-7c|break-ld7c-sbuf|gpu|120|@BIN@/lds_smem sbuf --break"
    "MT-7|break-mt7-levels|gpu|300|@BIN@/left_prio @BIN@/left_prio.hsaco --levels --break"
    "MT-5b|break-mt5b-handover|gpu|300|@BIN@/tim_mat handover --break"
    "MT-6|break-mt6-rot|gpu|300|@BIN@/left_rot --break"
    "MT-5c|break-mt5c|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py mt5c @RUN@ --break"
    "MT-3|break-mt3|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py mt3 @RUN@ --break"
    "LD-9|break-ld9|cpu|60|python3 @ROOT@/scripts/leftc_bursts.py ld9 @RUN@ --break"
    "SC-5|break-sc5-idle|gpu|60|@BIN@/sched_trace idle --burst-ms 100 --sample-ms 5 --out @RUN@/sc5-idle-break --break"
    "VK-7a|break-vk7a|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy mat fp8bf8 --w 8 --reps 1 --expect-half"
    "VK-3|break-vk3-pipe|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.075 @BIN@/vkstudy load f16 row tight l2 --w 4 --reps 1 --variant pipebrk"
    "VK-2|break-vk2|card|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ VKS_TARGET_S=0.15 @BIN@/vkstudy mat f16 --w 8 --reps 1 --expect-half"
    "SC-12|break-sc12-keepalive|gpu|120|@BIN@/vksched --spv @BIN@/spv/vksched_one.spv --keepalive --break"
    "VK-5|break-vk5-check2|gpu|120|env VKS_SPV=@BIN@/spv VKS_OUT=@RUN@ @BIN@/vkstudy check2 --break"
    "SC-4|break-sc4-shift|cpu|120|python3 -I @ROOT@/scripts/sched_pair.py --shift @SCRUNS@/20261117T165519Z-sched/pair-small-small-A.csv @SCRUNS@/20261117T165519Z-sched/pair-small-small-B.csv --break"
    "SC-3|break-sc3-throttle|cpu|120|python3 -I @ROOT@/scripts/leftc_throttle.py --break $T3/trace-f16-sysfs.csv $T3/trace-fp8-sysfs.csv"
    "EQ-1a|break-eq1a-scaled|cpu|1200|@BIN@/cpu/arith_realw @BIN@/realw/blk.0.attn_q.bin --real-only --scale --break"
    "EQ-3|break-eq3-quantbf16|cpu|1200|@BIN@/cpu/arith_stats quantbf16 --break"
    "EQ-2|break-eq2-nosub|cpu|1800|@BIN@/cpu/arith_qworst --nosub --break"
    "EQ-2|break-eq2-nonfinite|cpu|1200|@BIN@/cpu/arith_qworst --nonfinite --break"
)

selected() {
    local it=$1 s
    ((${#ONLY[@]} == 0)) && return 0
    for s in "${ONLY[@]}"; do
        [[ $s == "$it" ]] && return 0
    done
    return 1
}

# A probe is built when any selected item's job uses it (all probes when nothing is selected).
selected_probe() {
    local p=$1 j it
    ((${#ONLY[@]} == 0)) && return 0
    for j in "${JOBS[@]}" "${BREAKS[@]}"; do
        it=${j%%|*}
        selected "$it" && [[ $j == *"@BIN@/$p "* || $j == *"@BIN@/$p" || $j == *"@BIN@/cpu/$p "* ||
            ($p == sched_share && $j == *__preempt_*) || ($p == vkstudy && $j == *"@BIN@/vk"*) ]] && return 0
    done
    return 1
}

expand() {
    local c=$1
    c=${c//@RUN@/$RUN_DIR}
    c=${c//@BIN@/$BIN}
    c=${c//@ROOT@/$ROOT}
    c=${c//@SCRUNS@/$SC_RUNS}
    echo "$c"
}

# Runs one table row; prints the exit status.
run_row() {
    local row=$1 it name kind tmo cmd
    IFS='|' read -r it name kind tmo cmd <<<"$row"
    cmd=$(expand "$cmd")
    local -a words
    read -r -a words <<<"$cmd"
    case $kind in
    cpu) cpu_job "$name" "$tmo" "${words[@]}" ;;
    # SC-11's kernels hold the CUs with sleeping waves (near-idle power): gated like few-wave jobs
    pair) gpu_job "$name" gpu "$tmo" "$SELF" "${words[@]}" ;;
    *) gpu_job "$name" "$kind" "$tmo" "${words[@]}" ;;
    esac
}

phase_run() {
    local row it name rc v
    sum ""
    sum "## run"
    for row in "${JOBS[@]}"; do
        IFS='|' read -r it name _ <<<"$row"
        selected "$it" || continue
        run_row "$row"
        rc=$?
        v=$(grep -h "^VERDICT" "$RUN_DIR/$name.log" 2>/dev/null | sed 's/^/  - /')
        if ((rc == 0)); then
            sum "- $it $name: ok"
        else
            sum "- $it $name: FAILED (exit $rc)$(grep -h '^NOT RUN' "$RUN_DIR/$name.log" 2>/dev/null | sed 's/^/; /')"
            FAILED=1
        fi
        [[ -n $v ]] && sum "$v"
    done
}

phase_breaks() {
    local row it name rc
    sum ""
    sum "## deliberate breaks (each must FAIL)"
    for row in "${BREAKS[@]}"; do
        IFS='|' read -r it name _ <<<"$row"
        selected "$it" || continue
        run_row "$row"
        rc=$?
        if ((rc != 0)) && grep -q "^VERDICT.*FAIL" "$RUN_DIR/$name.log"; then
            sum "- $it $name: FAILED as intended: $(grep -h '^VERDICT.*FAIL' "$RUN_DIR/$name.log" | head -1)"
        else
            sum "- $it $name: did NOT fail (exit $rc); the check cannot fail: FAILED"
            FAILED=1
        fi
    done
}

PHASE=${1:-all}
shift || true
ONLY=("$@")
{
    echo "# Leftovers C lane B, $(date -u +%Y-%m-%dT%H:%M:%SZ), phase $PHASE${ONLY[*]:+, items: ${ONLY[*]}}"
    echo "- STUDY_REPO=$STUDY_REPO CARD_ENV=$CARD_ENV; LACT profile $(lact_profile); kernel $(uname -r)"
    echo "- idle snapshot: $(snap "$DRM_DEVICE" "$HWMON")"
} >>"$SUMMARY"
case $PHASE in
build) phase_build ;;
run) phase_run ;;
breaks) phase_breaks ;;
all)
    phase_build
    phase_run
    phase_breaks
    ;;
*) die "unknown phase $PHASE (build|run|breaks|all)" ;;
esac
sum ""
sum "Highest hotspot seen: $HOTTEST C. Run folder: $RUN_DIR"
if ((FAILED)); then
    sum "RESULT: something FAILED (see above)"
else
    sum "RESULT: all checks passed"
fi
cat "$SUMMARY"
exit "$FAILED"
