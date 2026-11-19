#!/usr/bin/env bash
# Gaming round, probe lane: how the RX 9070 XT and RADV handle game-like work. Plain Vulkan (src/vk/vkgame*.c) for
# P1 (short passes in one command buffer), P2 (the compute queue's timing; its draw arms are refused after review) and
# the experiments of the review round E1-E8 (the dense FMA load src/vk/shaders/gdense.comp and its HIP twin
# src/study/dense_twin.hip, analysed by scripts/gprobe_dense.py); HIP (src/study/sched_share.hip, scripts/sched_pair.py)
# for P3 (preemption, with a positive control) and E8 (the gap detector's known answer). Second fix round: F1 (dense
# whole-card bursts), F2 (a HIP, Vulkan or no predecessor before a Vulkan or HIP ramp, randomised), F3 (E3's gaps with
# dense loads), F4 (the 100-dispatch list at the high and the low clock), F5 (the in-load stop, end to end).
#
# Usage: scripts/study-gaming-probes.sh [build|run|breaks|all] [p1 p2 p3 p4 e1 ... e8 f1 ... f5]
#   build   shaders (glslc, spirv-val) into build/spv/, the Vulkan probe build/vkgame, the HIP probe build/sched_share,
#           and the study's own calibration programs (vkstudy, vksched, sched_launch) from STUDY_REPO
#   run     every probe of the named items (default all) as its own GPU job, calibrations first
#   breaks  each deliberate break must FAIL (GPU breaks of the named items, and the CPU re-judging breaks)
#   all     build, run, breaks
# Environment: GPROBE_RUN_DIR (reuse one run folder across phases); GPROBE_ONLY (a regular expression: run only the
# matching jobs, for development runs); GPROBE_REF_RUN (the run folder whose logs the CPU-only breaks of p1-p4, e1 and
# e8 re-judge; default the run folder itself); GPROBE_F2_SEED (F2's shuffle seed, default 20261118); STUDY_REPO (default: the folder above scripts/,
# or its repo-master/ snapshot when that exists) supplies src/harness.h, src/study/*.h, src/vk/vksched.c,
# src/vk/vkstudy.c, src/vk/vkh.c, src/study/sched_launch.hip and runs/latest/card.env (CARD_ENV overrides the card record).
# Results: runs/<UTC time>-gprobe/: one log per job, CSV files of raw samples, asm/ (machine code of every pipeline),
# clock-log.csv (once a second), summary.md (every VERDICT line). Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a time limit, at least 8 GB available, the LACT profile "Testing" (read with
# `lact cli profile get`) and the card configuration unchanged since the start; each job waits outside the lock for a
# hotspot below 58 C. The probes watch the hotspot every 5 ms during the load and end it themselves at 88 C (the dense
# load through a stop word read by every wave); a job whose in-run maximum is above 85 C stops all further GPU work.
# FMA-class loads: at most 0.1 s continuous; duty-cycled loads above 0.40 busy (measured) at most 0.1 s, at or below
# 0.40 at most 1 s. No full-screen pixel-shader FMA load. No window, no swapchain, no display connection, no sound.
# A whole-card job whose 5 ms sampler shows a gap above 50 ms is set aside (superseded/), and repeated after a minute,
# at most twice. Any reported power sample above 317 W is written to the summary.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$ROOT/scripts/study-gaming-probes.sh"
if [[ -z ${STUDY_REPO:-} ]]; then
    STUDY_REPO=$ROOT
    [[ -f $ROOT/repo-master/src/harness.h ]] && STUDY_REPO=$ROOT/repo-master
fi
cd "$ROOT" || exit 2
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES=0
BIN=$ROOT/build
SPV=$BIN/spv

snap() {
    local drm=$1 hw=$2
    echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')" \
        " power(reported, scaled down by the power-limit mod): $(($(cat "$hw/power1_average") / 1000000)) W" \
        " hotspot: $(($(cat "$hw/temp2_input") / 1000)) C  busy: $(cat "$drm/gpu_busy_percent") %"
}

# Locked runner: re-entered under flock, so the clocks are read while this job holds the GPU.
if [[ ${1:-} == __locked ]]; then
    log=$2 drm=$3 hw=$4 limit=$5
    shift 5
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 "$limit" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after $limit s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    echo "after the job: LACT profile $(timeout 20 lact cli profile get 2>/dev/null | tr -d '[:space:]'), power1_cap $(($(cat "$hw/power1_cap") / 1000000)) W" >>"$log"
    exit "$rc"
fi

# P3: two processes started together inside one locked job. Arguments: output prefix, expectation for the analysis
# (wait | overlap | any), A's shape options (one word list), options for both (one word list).
if [[ ${1:-} == __pair ]]; then
    out=$2 expect=$3 aopts=$4 common=$5
    "$BIN/sched_share" gate || exit 7
    start=$(($(date +%s%3N) + 2500))
    # shellcheck disable=SC2086 # option lists on purpose
    "$BIN/sched_share" preempt --role A --start-ms "$start" --out "$out" $aopts $common >"$out-A.log" 2>&1 &
    pa=$!
    # shellcheck disable=SC2086
    "$BIN/sched_share" preempt --role B --start-ms "$start" --out "$out" $common >"$out-B.log" 2>&1 &
    pb=$!
    wait "$pa"
    ra=$?
    wait "$pb"
    rb=$?
    cat "$out-A.log" "$out-B.log"
    python3 -I "$ROOT/scripts/sched_pair.py" --preempt "$out-A.csv" "$out-B.csv" --expect "$expect"
    rp=$?
    ((ra == 0 && rb == 0 && rp == 0)) || exit 1
    exit 0
fi

# P2: six short processes of the one-group probe in one locked job (is the compute queue's slow path a state of the
# process?).
if [[ ${1:-} == __procs ]]; then
    for i in 1 2 3 4 5 6; do
        echo "PROCESS $i"
        VKG_LONE_N=300 VKG_TAG=p2-lone-procs-$i "$BIN/vkgame" lone || exit 1
    done
    exit 0
fi

# F2: a predecessor, a pause, then the target process, inside one locked job (the lock is held through the pause:
# it is the experiment's variable, and no other lock-respecting job may run in it). Arguments: order, target, pred
# (hip | vk | none), pause in s. VKG_OUT and VKG_TAG come from gpu_job.
if [[ ${1:-} == __seq ]]; then
    order=$2 target=$3 pred=$4 pause=$5
    one="groups=1,local=32,every=64,recstride=1" cu="groups=64,local=32,every=64,recstride=8"
    echo "SEQ order=$order target=$target pred=$pred pause_s=$pause"
    case $pred in
        hip) "$BIN/dense_twin" --out "$VKG_OUT/$VKG_TAG-pred" --idle-ms 1000 pred-hip,1,32,64,60,1 || exit 1 ;;
        vk) VKG_TAG=$VKG_TAG-pred "$BIN/vkgame" burst "name=pred-vk,q=g,$one,cont=60,idle=1" || exit 1 ;;
        none) ;;
        *) exit 2 ;;
    esac
    echo "SEQ predecessor ended $(date -u +%FT%T.%NZ); pause $pause s"
    sleep "$pause"
    case $target in
        vkg1) "$BIN/vkgame" burst "name=target-vkg1,q=g,$one,cont=60,idle=1" ;;
        vkgcu) "$BIN/vkgame" burst "name=target-vkgcu,q=g,$cu,cont=60,idle=1" ;;
        vkc1) "$BIN/vkgame" burst "name=target-vkc1,q=c,$one,cont=60,idle=1" ;;
        vkccu) "$BIN/vkgame" burst "name=target-vkccu,q=c,$cu,cont=60,idle=1" ;;
        hip1) "$BIN/dense_twin" --out "$VKG_OUT/$VKG_TAG" --idle-ms 1000 target-hip1,1,32,64,60,1 ;;
        hipcu) "$BIN/dense_twin" --out "$VKG_OUT/$VKG_TAG" --idle-ms 1000 target-hipcu,64,32,64,60,8 ;;
        *) exit 2 ;;
    esac
    exit $?
fi

die() {
    echo "study-gaming-probes: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$STUDY_REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || CARD_ENV=$ROOT/runs/latest/card.env
[[ -f $CARD_ENV ]] || die "no card record (set STUDY_REPO or CARD_ENV)"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 && -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record: not gfx1201 or no sysfs paths"
[[ $(cat "$DRM_DEVICE/vendor") == 0x1002 && $(cat "$DRM_DEVICE/device") == 0x7550 ]] || die "sysfs is not the RX 9070 XT"
export DRM_DEVICE HWMON

RUN_DIR=${GPROBE_RUN_DIR:-"$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-gprobe"}
# home/ takes Mesa's shader cache for the command-stream dump jobs, which run with HOME pointed here; vksched/ the
# study vksched's sample file (it does not create the folder).
mkdir -p "$RUN_DIR/home" "$RUN_DIR/vksched" "$RUN_DIR/vkstudy" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
REF=${GPROBE_REF_RUN:-$RUN_DIR}
FAILED=0
STOPPED=0
HOTTEST=0
SAMPLER=

sum() { echo "$*" >>"$SUMMARY"; }
avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
hotspot_c() { echo $(($(cat "$HWMON/temp2_input") / 1000)); }
lact_profile() { timeout 20 lact cli profile get 2>/dev/null | tr -d '[:space:]'; }
card_config() {
    echo "power1_cap=$(cat "$HWMON/power1_cap") perf_level=$(cat "$DRM_DEVICE/power_dpm_force_performance_level") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null)"
}
CONFIG0=$(card_config)

# Once-a-second clock log for the whole run (read-only sysfs).
start_sampler() {
    local csv=$RUN_DIR/clock-log.csv
    [[ -f $csv ]] || echo "utc,sclk_mhz,mclk_mhz,power_w_reported,hotspot_c,edge_c,busy_pct" >"$csv"
    (
        while :; do
            echo "$(date -u +%FT%TZ),$(grep '\*' "$DRM_DEVICE/pp_dpm_sclk" | grep -o '[0-9]*Mhz' | tr -d Mhz)," \
                "$(grep '\*' "$DRM_DEVICE/pp_dpm_mclk" | grep -o '[0-9]*Mhz' | tr -d Mhz),$(($(cat "$HWMON/power1_average") / 1000000))," \
                "$(($(cat "$HWMON/temp2_input") / 1000)),$(($(cat "$HWMON/temp1_input") / 1000)),$(cat "$DRM_DEVICE/gpu_busy_percent")" |
                tr -d ' ' >>"$csv"
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

# Waits outside the lock until the hotspot is below 58 C (at most 300 s).
cool_gate() {
    local i
    for ((i = 0; i < 600; i++)); do
        (($(hotspot_c) < 58)) && return 0
        sleep 0.5
    done
    return 1
}

# Waits outside the lock until the 1-minute load average is below 8 (at most 900 s): other agents' work on this host
# stretched the 5 ms sampler to 30-50 ms per reading (development run dev2-232334).
# Also the share of idle CPU time over 2 s must be at least 40 % (the load average lags).
load1() { cut -d' ' -f1 /proc/loadavg; }
idle_share() {
    local a b
    a=$(awk '/^cpu / {print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9}' /proc/stat)
    sleep 2
    b=$(awk '/^cpu / {print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9}' /proc/stat)
    awk -v a="$a" -v b="$b" 'BEGIN {split(a, x, " "); split(b, y, " "); printf "%.2f", (y[1] - x[1]) / (y[2] - x[2])}'
}
quiet_gate() {
    local i
    for ((i = 0; i < 300; i++)); do
        awk -v l="$(load1)" -v s="$(idle_share)" 'BEGIN {exit !(l < 8 && s >= 0.40)}' && return 0
        sleep 1
    done
    return 1
}

# A game running (a Windows executable under Proton/Wine, or a known game name), from /proc/*/comm only (comm is cut
# at 15 characters, so "Cyberpunk2077.e" counts too). Prints the first match.
game_running() {
    local f c
    for f in /proc/[0-9]*/comm; do
        c=$(cat "$f" 2>/dev/null) || continue
        if [[ $c == *.exe || ($c == *.* && ${#c} -ge 15 && $c =~ \.[A-Za-z]{0,2}$) || $c == Cyberpunk* ]]; then
            echo "${f%/comm} $c"
            return 0
        fi
    done
    return 1
}
# The conditions every GPU job needs right before it takes the lock: profile Testing, the card configuration of the
# run's start (power cap 317 W among it), no game. Prints the reason when one fails.
gpu_preconditions() {
    local p g
    p=$(lact_profile)
    [[ $p == Testing ]] || { echo "LACT profile is '$p', not Testing"; return 1; }
    [[ $(card_config) == "$CONFIG0" ]] || { echo "card configuration changed since the start of the run"; return 1; }
    g=$(game_running) && { echo "a game process is running ($g)"; return 1; }
    return 0
}

# gpu_job <name> <time limit s> <command...>; the command runs with VKG_* set for this job.
gpu_job() {
    local name=$1 limit=$2 log cfg t rc
    shift 2
    log=$RUN_DIR/$name.log
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((STOPPED)); then
        echo "NOT STARTED: GPU work was stopped earlier in this run" >>"$log"
        return 6
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if [[ $(lact_profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)', not Testing" >>"$log"
        sum "- STOPPED: the LACT profile is not Testing before $name; no further GPU jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
    cfg=$(card_config)
    echo "card configuration: $cfg" >>"$log"
    if [[ $cfg != "$CONFIG0" ]]; then
        echo "NOT STARTED: card configuration changed since the start of the run (was: $CONFIG0)" >>"$log"
        sum "- STOPPED: the card configuration changed before $name; no further GPU jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
    if ! cool_gate; then
        echo "NOT STARTED: hotspot still $(hotspot_c) C after 300 s" >>"$log"
        return 7
    fi
    if ! quiet_gate; then
        echo "NOT STARTED: host load average still $(load1) after 900 s (the probes' 5 ms sampler and host timing need a quiet host)" >>"$log"
        return 7
    fi
    echo "host load average (1 min) at start: $(load1)" >>"$log"
    echo "hotspot at start: $(hotspot_c) C" >>"$log"
    # again right before the lock: the gates above can wait minutes (job f2-o016 of 20261118T012647Z-gprobe-f passed
    # the profile check at 01:32:25 and ran at 01:34:44, around the owner's switch to another profile)
    local why
    if ! why=$(gpu_preconditions); then
        echo "NOT STARTED: $why (checked right before the lock)" >>"$log"
        sum "- STOPPED: $why before $name; no further GPU jobs"
        STOPPED=1
        FAILED=1
        return 6
    fi
    echo "right before the lock: LACT profile $(lact_profile), $(card_config | cut -d' ' -f1), no game process" >>"$log"
    VKG_OUT=$RUN_DIR VKG_SPV=$SPV VKG_TAG=$name flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" \
        "$limit" "$@"
    rc=$?
    # in-run maxima: vkgame "THERMAL highest N", the HIP programs "HOTTEST N C", the dense twin "THERMAL <arm> highest N"
    t=$(grep -hoE '^(THERMAL highest|HOTTEST|THERMAL [^ ]+ highest) [0-9]+' "$log" | grep -oE '[0-9]+$' | sort -n | tail -1)
    for x in $t $(hotspot_c); do
        ((x > HOTTEST)) && HOTTEST=$x
    done
    echo "in-run hotspot maximum: ${t:-none} C" >>"$log"
    if [[ -n $t ]] && ((t > 85)); then
        sum "- STOPPED: $name reached $t C in the run (above 85 C); no further GPU jobs"
        STOPPED=1
        FAILED=1
    fi
    return "$rc"
}

# A job's verdict lines into the summary; a non-zero exit or any FAIL marks the run failed.
record() {
    local name=$1 rc=$2 log=$RUN_DIR/$1.log v
    v=$(grep -h "^VERDICT" "$log" | cut -c1-400)
    if ((rc == 0)) && ! grep -q "^VERDICT .*: FAIL" "$log"; then
        sum "- $name: ok"
    else
        sum "- $name: FAILED (exit $rc; see $name.log)"
        FAILED=1
    fi
    [[ -n $v ]] && awk '{print "  - " $0}' <<<"$v" >>"$SUMMARY"
}

# A deliberate break: the job must exit non-zero with at least one FAIL verdict.
record_break() {
    local name=$1 rc=$2 log=$RUN_DIR/$1.log v
    v=$(grep -h "^VERDICT .*FAIL" "$log" | cut -c1-400 | head -3)
    if ((rc != 0)) && [[ -n $v ]]; then
        sum "- $name: FAIL as intended"
    else
        sum "- $name: did NOT fail (exit $rc)"
        FAILED=1
    fi
    [[ -n $v ]] && awk '{print "  - " $0}' <<<"$v" >>"$SUMMARY"
}

# Calibration of P2: the study's vksched, "alone" mode. Known answer (READ, scheduling item 6 and leftovers SC-12):
# compute queue 25 to 42 % of submissions above twice the median (7 runs), graphics queue about 1 %.
# judge_vksched <log> [swap]
judge_vksched() {
    local log=$1 swap=${2:-} c g
    c=$(awk '/^RESULT alone compute slow_frac=/ {sub(/.*slow_frac=/, ""); print $1}' "$log")
    g=$(awk '/^RESULT alone graphics slow_frac=/ {sub(/.*slow_frac=/, ""); print $1}' "$log")
    [[ -n $swap ]] && { local x=$c; c=$g; g=$x; }
    if [[ -n $c && -n $g ]] && awk -v c="$c" -v g="$g" 'BEGIN {exit !(c >= 0.20 && c <= 0.50 && g <= 0.03)}'; then
        echo "VERDICT gprobe vksched_calibration: PASS ${swap:+DELIBERATE BREAK (labels swapped) }compute alone slow share $c (study 0.25-0.42; limit 0.20..0.50), graphics $g (limit 0.03)"
        return 0
    fi
    echo "VERDICT gprobe vksched_calibration: FAIL ${swap:+DELIBERATE BREAK (labels swapped) }compute alone slow share ${c:-none} (study 0.25-0.42; limit 0.20..0.50), graphics ${g:-none} (limit 0.03)"
    return 1
}

# Calibration of P1: the study's vkstudy, mode dispatch (READ, vulkan-view item 7: 100 one-group dispatches with
# barriers in one command buffer, 1.13 us per dispatch, 1.12-1.14 over 5 buffers). judge_vkstudy <log> [low high]
judge_vkstudy() {
    local log=$1 lo=${2:-0.9} hi=${3:-1.4} m
    m=$(grep -oE 'back-to-back .*median [0-9.]+ us per dispatch' "$log" | grep -oE 'median [0-9.]+' | cut -d' ' -f2)
    if [[ -n $m ]] && awk -v m="$m" -v lo="$lo" -v hi="$hi" 'BEGIN {exit !(m >= lo && m <= hi)}'; then
        echo "VERDICT gprobe vkstudy_calibration: PASS $m us per dispatch (vulkan-view item 7: 1.13; limit $lo..$hi)"
        return 0
    fi
    echo "VERDICT gprobe vkstudy_calibration: FAIL ${m:-none} us per dispatch (vulkan-view item 7: 1.13; limit $lo..$hi)"
    return 1
}

# Calibration of P4: the study's HIP ramp (scheduling item 1: whole card after 1 s idle, first window 2688 MHz, 95 %
# of the final clock at 14 to 15 ms). judge_hipramp <log> [low high]
judge_hipramp() {
    local log=$1 lo=${2:-2600} hi=${3:-2850} f t
    f=$(grep -oE 'first_window_MHz=[0-9.]+' "$log" | tail -1 | cut -d= -f2)
    t=$(grep -oE 'time_to_95pct_ms=[0-9.]+' "$log" | tail -1 | cut -d= -f2)
    if [[ -n $f && -n $t ]] && awk -v f="$f" -v t="$t" -v lo="$lo" -v hi="$hi" 'BEGIN {exit !(f >= lo && f <= hi && t >= 12 && t <= 17)}'; then
        echo "VERDICT gprobe hip_ramp_calibration: PASS first window $f MHz (limit $lo..$hi; study 2688), 95 % at $t ms (limit 12..17; study 14-15)"
        return 0
    fi
    echo "VERDICT gprobe hip_ramp_calibration: FAIL first window ${f:-none} MHz (limit $lo..$hi; study 2688), 95 % at ${t:-none} ms (limit 12..17; study 14-15)"
    return 1
}

phase_build() {
    local s rc=0 log=$RUN_DIR/build.log name
    mkdir -p "$SPV" "$BIN/vksched" "$BIN/vkstudy/spv" "$BIN/tmp-sched_share" "$BIN/tmp-sched_launch"
    sum "## build ($(glslc --version 2>&1 | head -1 | tr -d '\n'); $(cc --version | head -1); $(hipcc --version 2>/dev/null | grep -m1 -i 'clang version' | tr -d '\n'))"
    for s in gchain.comp gtiny.comp gburst.comp gdense.comp gdraw.vert gdraw.frag; do
        name=${s%.*}
        [[ $s == *.vert ]] && name=gdraw_vert
        [[ $s == *.frag ]] && name=gdraw_frag
        if ! nice -n 10 glslc --target-env=vulkan1.3 -O "$ROOT/src/vk/shaders/$s" -o "$SPV/$name.spv" >>"$log" 2>&1 ||
            ! spirv-val --target-env vulkan1.3 "$SPV/$name.spv" >>"$log" 2>&1; then
            sum "- shader $s: FAILED (see build.log)"
            rc=1
        fi
    done
    if nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$ROOT/src/vk" -o "$BIN/vkgame" \
        "$ROOT/src/vk/vkgame.c" "$ROOT/src/vk/vkgame_chain.c" "$ROOT/src/vk/vkgame_share.c" \
        "$ROOT/src/vk/vkgame_burst.c" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- vkgame built ($(grep -c warning "$log") compiler warnings in build.log)"
    else
        sum "- vkgame: build FAILED (see build.log)"
        rc=1
    fi
    if (cd "$BIN/tmp-sched_share" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' \
        -mcumode -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -I"$ROOT/src/study" -save-temps \
        "$ROOT/src/study/sched_share.hip" -o "$BIN/sched_share" -lamdhip64) >>"$log" 2>&1; then
        sum "- sched_share built (CU mode; -save-temps in build/tmp-sched_share)"
    else
        sum "- sched_share: build FAILED (see build.log)"
        rc=1
    fi
    mkdir -p "$BIN/tmp-dense"
    if (cd "$BIN/tmp-dense" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' \
        -mcumode -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -save-temps "$ROOT/src/study/dense_twin.hip" \
        -o "$BIN/dense_twin" -lamdhip64) >>"$log" 2>&1; then
        mkdir -p "$RUN_DIR/asm"
        cp "$BIN/tmp-dense/dense_twin.hip-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.gfx1201.img.lto.s" "$RUN_DIR/asm/dense_twin.s"
        sum "- dense_twin built (wave32, CU mode; listing in asm/dense_twin.s)"
    else
        sum "- dense_twin: build FAILED (see build.log)"
        rc=1
    fi
    # The study's own programs, unchanged, for the calibrations of P2 and P4.
    if nice -n 10 glslc --target-env=vulkan1.3 -O "$STUDY_REPO/src/vk/shaders/vksched_one.comp" \
        -o "$BIN/vksched/vksched_one.spv" >>"$log" 2>&1 &&
        nice -n 10 cc -O2 -std=c11 -Wall -Wextra -o "$BIN/vksched/vksched" "$STUDY_REPO/src/vk/vksched.c" -lvulkan -lm \
            >>"$log" 2>&1 &&
        (cd "$BIN/tmp-sched_launch" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 \
            -DHARNESS_ARCH='"gfx1201"' -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" \
            "$STUDY_REPO/src/study/sched_launch.hip" -o "$BIN/sched_launch" -lamdhip64) >>"$log" 2>&1 &&
        nice -n 10 glslc --target-env=vulkan1.3 -O "$STUDY_REPO/src/vk/shaders/empty.comp" -o "$BIN/vkstudy/spv/empty.spv" \
            >>"$log" 2>&1 &&
        nice -n 10 glslc --target-env=vulkan1.3 -O "$STUDY_REPO/src/vk/shaders/one.comp" -o "$BIN/vkstudy/spv/one.spv" \
            >>"$log" 2>&1 && cp "$BIN/vkstudy/spv/one.spv" "$SPV/one.spv" &&
        nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -o "$BIN/vkstudy/vkstudy" \
            "$STUDY_REPO/src/vk/vkstudy.c" "$STUDY_REPO/src/vk/vkh.c" -lvulkan -lpthread -lm >>"$log" 2>&1; then
        sum "- study programs built from $STUDY_REPO: vksched, sched_launch, vkstudy"
    else
        sum "- study programs: build FAILED (see build.log)"
        rc=1
    fi
    ((rc)) && FAILED=1
    return "$rc"
}

V="$BIN/vkgame"
P1_BAR="bar=none bar=b2 bar=legacy bar=vv7 bar=exec bar=event bar=full"

# One job per line: name | time limit | command (word list; @RUN@ is the run folder). The jobs_* functions are
# called by name from phase_run.
# shellcheck disable=SC2329
jobs_p1() {
    local q b s g n
    echo "p1-info|60|$V info"
    # what RADV emits for each barrier kind (command-stream dump to the log; tiny lists)
    s=""
    for q in g c; do
        for b in $P1_BAR; do s+=" name=ib_${q}_${b#bar=},q=$q,$b,n=3,reps=1,grid=8"; done
        s+=" name=ib_${q}_b2_ts0,q=$q,bar=b2,ts=0,n=3,reps=1,grid=8"
    done
    echo "p1-dump-chain|60|env HOME=@RUN@/home RADV_DEBUG=dumpibs $V chain$s"
    echo "p1-dump-tiny-top|60|env HOME=@RUN@/home RADV_DEBUG=dumpibs VKG_LONE_N=2 VKG_TINY_TS=top $V lone"
    echo "p1-dump-tiny-bottom|60|env HOME=@RUN@/home RADV_DEBUG=dumpibs VKG_LONE_N=2 $V lone"
    # known answer: vulkan-view item 7 (100 one-group dispatches, legacy barrier, BOTTOM timestamp after each)
    echo "p1-cal-vkstudy|60|env VKS_SPV=$BIN/vkstudy/spv VKS_OUT=@RUN@/vkstudy $BIN/vkstudy/vkstudy dispatch"
    echo "p1-cal|60|$V chain name=cal_vv7,q=c,bar=vv7,grid=1,body=0,n=100,reps=5,ts=1,stride=0,cal=1 name=cal_vv7_gfx,q=g,bar=vv7,grid=1,body=0,n=100,reps=5,ts=1,stride=0"
    for q in g c; do
        s=""
        for b in $P1_BAR; do s+=" name=bar_${q}_${b#bar=},q=$q,$b"; done
        echo "p1-bar-$q|120|$V chain$s name=chk_${q}_b2,q=$q,bar=b2,check=1,reps=50"
    done
    for q in g c; do
        s=""
        for g in 1 8 64 256 1024 4096; do
            s+=" name=grid_${q}_b2_$g,q=$q,bar=b2,grid=$g"
            [[ $q == g ]] && s+=" name=grid_${q}_none_$g,q=$q,bar=none,grid=$g"
        done
        echo "p1-grid-$q|120|$V chain$s"
    done
    s=""
    for b in b2 none; do
        for n in 0 3000 15000 60000; do s+=" name=body_${b}_$n,q=g,bar=$b,body=$n"; done
        s+=" name=card_${b}_15000,q=g,bar=$b,grid=4096,body=15000,reps=100"
    done
    echo "p1-body|120|$V chain$s"
    s=""
    for b in b2 none; do
        for n in 1 5 20 40; do s+=" name=n_g_${b}_$n,q=g,bar=$b,n=$n,fit=g_$b"; done
    done
    for n in 1 5 20 40; do s+=" name=n_c_b2_$n,q=c,bar=b2,n=$n,fit=c_b2"; done
    echo "p1-n|120|$V chain$s"
    echo "p1-ts|120|$V chain name=ts0_g,q=g,ts=0 name=ts1_g,q=g,ts=1 name=ts2_g,q=g,ts=2 name=ts0_c,q=c,ts=0 name=ts1_c,q=c,ts=1 name=stride_auto_4096,q=g,grid=4096,reps=50 name=stride_all_4096,q=g,grid=4096,stride=1,reps=50"
    echo "p1-subs|120|$V chain name=subs0_g,q=g name=subs1_g,q=g,subs=1 name=subs2_g,q=g,subs=2,reps=50 name=subs0_c,q=c name=subs1_c,q=c,subs=1 name=subs2_c,q=c,subs=2,reps=50"
    echo "p1-gap|120|$V chain name=gap0_g,q=g name=gap1ms_g,q=g,gap=1000,reps=100 name=gap10ms_g,q=g,gap=10000,reps=50 name=gap100ms_g,q=g,gap=100000,reps=10 name=gap10ms_c,q=c,gap=10000,reps=50"
    # "about 70 us at any size": per-group work of 120,000 cycles (about 70 us at 1.7 GHz), grids 8 to 2048
    s=""
    for g in 8 64 256 1024 2048; do s+=" name=size_${g},q=g,bar=b2,grid=$g,body=120000,n=5,reps=20"; done
    echo "p1-size|120|$V chain$s"
    echo "p1-order|120|$V chain name=alt_1_4096,q=g,grid=1,grid2=4096 name=alt_4096_1,q=g,grid=4096,grid2=1 name=alt_none_1_4096,q=g,bar=none,grid=1,grid2=4096 name=sg32,q=g,sg=32 name=local256,q=g,local=256,grid=16"
}
# shellcheck disable=SC2329
jobs_p2() {
    local q f
    echo "p2-cal-vksched|120|$BIN/vksched/vksched --spv $BIN/vksched/vksched_one.spv --out @RUN@/vksched"
    echo "p2-lone|60|$V lone"
    echo "p2-lone-top|60|env VKG_TINY_TS=top $V lone"
    # the slow path's share per process: six short processes in one job
    echo "p2-lone-procs|120|$SELF __procs"
    echo "p2-nodraw|60|$V share name=nodraw_c,draw=off,cq=c,dur=1 name=nodraw_g,draw=off,cq=g,dur=1"
    # No draw arm: after the heat incident a full-screen pixel-shader FMA load is not run at all (the probe refuses
    # it too). The draw arms' results are those of runs/20261117T223829Z-gprobe.
}
# shellcheck disable=SC2329
jobs_p3() {
    local c="--trials 10 --ta-ms 4,16,64"
    echo "p3-room|120|$SELF __pair @RUN@/p3-room overlap --a-lds-kib=0|$c"
    echo "p3-few|120|$SELF __pair @RUN@/p3-few overlap --a-groups=16|$c --dump 5"
    echo "p3-fma|120|$SELF __pair @RUN@/p3-fma any --a-hold=fma|$c"
    echo "p3-sleep|120|$SELF __pair @RUN@/p3-sleep wait --a-hold=sleep|$c"
}
# shellcheck disable=SC2329
jobs_p4() {
    # Nothing to run. p4-cal-hip (the study's sched_launch ramp) has no in-load stop: it cannot end its whole-card
    # FMA load at 88 C, so after the second review it is not run again; its earlier runs are cited. The bursts and
    # ramps of P4 moved to E1-E3 with the dense load.
    :
}
# Dense-load arms (E1-E3). ONEWAVE / PERCU / CARD: 1 group of 32, 64 groups of 32 (one wave per CU if the groups are
# spread; placement is not recorded), 64 groups of 1024 (2,048 waves, the study's --card shape). every: chunks between
# checks (the study's HIP ramp checks every 8 chunks of 128 FMAs; one wave alone uses 64 so its records are about 3 to
# 5 us apart).
VK_ONE="groups=1,local=32,every=64,recstride=1"
VK_CU="groups=64,local=32,every=64,recstride=8"
VK_CARD="groups=64,local=1024,every=8,recstride=8"
# E2's whole-card bursts check the time bound after every chunk: with 2,048 waves the SIMDs' arbitration starves most
# waves, and a starved wave ran 0.6 to 1.8 ms past a 2 ms bound before its next check at every=64 (development run
# dev2-232334); records only at every 64th check (the instruction stream is the same; only the intervals differ)
VK_CARD_B="groups=64,local=1024,every=1,recevery=64,recstride=8"
# shellcheck disable=SC2329
jobs_e1() {
    local r q hip vk
    echo "e1-listing|60|$V listing"
    echo "e1-trip-vk|60|$V burst name=trip_vk_g,q=g,$VK_CU,cont=40,idle=0.2,stopat=10 name=trip_vk_c,q=c,$VK_ONE,cont=40,idle=0.2,stopat=10"
    echo "e1-trip-hip|60|$BIN/dense_twin --out @RUN@/e1-trip-hip --idle-ms 200 --stop-after-ms 10 trip_hip,64,32,64,40,8"
    for r in 1 2 3 4 5; do
        hip="e1-hip-r$r|60|$BIN/dense_twin --out @RUN@/e1-hip-r$r --idle-ms 1000 hip-onewave-r$r,1,32,64,60,1 hip-percu-r$r,64,32,64,60,8 hip-card-r$r,64,1024,8,60,8"
        vk=""
        for q in g c; do
            vk+="e1-vk$q-r$r|60|$V burst name=vk$q-onewave-r$r,q=$q,$VK_ONE,cont=60,idle=1 name=vk$q-percu-r$r,q=$q,$VK_CU,cont=60,idle=1 name=vk$q-card-r$r,q=$q,$VK_CARD,cont=60,idle=1"$'\n'
        done
        if ((r % 2)); then echo "$hip"; printf '%s' "$vk"; else printf '%s' "$vk"; echo "$hip"; fi
    done
}
# shellcheck disable=SC2329
jobs_e2() {
    local r i k arms=("2:16.7" "2:11.1" "2:8.3" "2:6" "4:16.7" "8:16.7") s a b c
    for r in 1 2 3 4 5; do
        s=""
        for ((k = 0; k < 6; k++)); do
            i=$(((k + r) % 6))
            a=${arms[$i]}
            b=${a%%:*} c=${a##*:}
            s+=" name=cu-b${b}-c${c}-r$r,q=g,$VK_CU,blen=$b,cad=$c,dur=1,idle=1"
        done
        echo "e2-percu-r$r|90|$V burst$s"
        for ((k = 0; k < 6; k++)); do
            i=$(((k + r) % 6))
            a=${arms[$i]}
            b=${a%%:*} c=${a##*:}
            echo "e2-card-b${b}-c${c}-r$r|60|$V burst name=card-b${b}-c${c}-r$r,q=g,$VK_CARD_B,blen=$b,cad=$c,dur=1,idle=1"
        done
    done
}
# shellcheck disable=SC2329
jobs_e3() {
    local r k i g q gaps=(0.5 1 2 3 5 8) s
    for r in 1 2 3 4 5; do
        s=""
        for ((k = 0; k < 12; k++)); do
            i=$(((k + 2 * r) % 12))
            g=${gaps[$((i / 2))]}
            q=g
            ((i % 2)) && q=c
            s+=" name=gap$g-$q-r$r,q=$q,$VK_CU,blen=2,gap=$g,dur=1,idle=1"
        done
        echo "e3-r$r|90|$V burst$s"
    done
}
# shellcheck disable=SC2329
jobs_e4() {
    local r c base="bar=vv7,grid=1,body=0,n=100,reps=20,ts=1,stride=0"
    for r in 1 2 3 4 5; do
        c="name=base-r$r,q=c,$base name=ts0bot-r$r,q=c,$base,ts0=bot name=bar0-r$r,q=c,$base,bar0=1"
        c+=" name=ts0bot_bar0-r$r,q=c,$base,ts0=bot,bar0=1 name=asvkstudy-r$r,q=c,$base,ts0=bot,bar0=1,spv=one.spv,push=0"
        # the clock as the one change: about 60 ms of lists that keep the card busy, then the same list, then a
        # stamped list with a 2,000-cycle spin that only reads the clock (stride 1)
        c+=" name=warm-r$r,q=c,bar=b2,grid=64,body=60000,n=20,reps=80 name=basewarm-r$r,q=c,$base"
        c+=" name=clockwarm-r$r,q=c,bar=vv7,grid=1,body=2000,n=100,reps=5,ts=1,stride=1"
        if ((r % 2)); then
            echo "e4-both-r$r|60|$V chain $c"
            echo "e4-conly-r$r|60|env VKG_FAMILIES=c $V chain $c"
        else
            echo "e4-conly-r$r|60|env VKG_FAMILIES=c $V chain $c"
            echo "e4-both-r$r|60|$V chain $c"
        fi
        echo "e4-vkstudy-r$r|60|env VKS_SPV=$BIN/vkstudy/spv VKS_OUT=@RUN@/vkstudy $BIN/vkstudy/vkstudy dispatch"
    done
}
# shellcheck disable=SC2329
jobs_e5() {
    local r q k i kinds=(none exec b2 legacy vv7 full event) s
    for r in 1 2 3 4 5; do
        for q in g c; do
            s=""
            for ((k = 0; k < 7; k++)); do
                i=$(((k + r) % 7))
                s+=" name=e5_${q}_${kinds[$i]}_r$r,q=$q,bar=${kinds[$i]},check=1,explore=1,reps=20"
            done
            echo "e5-$q-r$r|60|$V chain$s"
        done
    done
}
# shellcheck disable=SC2329
jobs_e6() {
    local p
    for p in 1 2 3; do
        echo "e6-bottom-p$p|60|$V slow name=bottom_p$p,n=300,rounds=6,cpu=3"
        echo "e6-top-p$p|60|env VKG_TINY_TS=top $V slow name=top_p$p,n=300,rounds=6,cpu=3"
    done
}
# shellcheck disable=SC2329
jobs_e7() {
    local p
    for p in 1 2; do echo "e7-rt-p$p|60|$V rt name=rt_p$p,rounds=5,passes=20"; done
}
# shellcheck disable=SC2329
jobs_e8() {
    echo "e8-inject|120|$SELF __pair @RUN@/e8-inject gap --a-inject-sleeps=250|--trials 10 --ta-ms 4,16"
}

# Second fix round. Dense shapes, checked every 8 chunks (E1's whole-card setting): the whole card (2,048 waves) and
# 64 groups of 128 threads (256 waves). minfill: the fill every dispatch must reach (FMA per SIMD per cycle).
VK_CARD_D="groups=64,local=1024,every=8,recevery=8,recstride=8,minfill=0.8"
VK_D64="groups=64,local=128,every=8,recevery=8,recstride=8,minfill=0.6"
# shellcheck disable=SC2329
jobs_f1() {
    local r i k arms=("2:16.7" "2:11.1" "2:8.3" "2:6" "4:16.7") a b c
    for r in 1 2 3 4 5; do
        for ((k = 0; k < 5; k++)); do
            i=$(((k + r) % 5))
            a=${arms[$i]}
            b=${a%%:*} c=${a##*:}
            echo "f1-card-b${b}-c${c}-r$r|60|$V burst name=f1card-b${b}-c${c}-r$r,q=g,$VK_CARD_D,blen=$b,cad=$c,dur=1,idle=1"
        done
    done
}
# F2: every (target, predecessor) combination 6 times, in one shuffled order (seed recorded in f2-order.txt).
# shellcheck disable=SC2329
jobs_f2() {
    local seed=${GPROBE_F2_SEED:-20261118}
    python3 -I - "$seed" <<'PY' | tee "$RUN_DIR/f2-order.txt" | while IFS=' ' read -r o t p z; do
import random, sys
seed = int(sys.argv[1])
combos = [(t, p, z) for t in ("vkg1", "vkgcu", "vkc1", "vkccu")
          for p, z in (("hip", 1), ("hip", 5), ("hip", 20), ("vk", 1), ("none", 20))]
combos += [(t, p, z) for t in ("hip1", "hipcu") for p, z in (("vk", 1), ("none", 20))]
jobs = [c for c in combos for _ in range(6)]
random.Random(seed).shuffle(jobs)
for i, (t, p, z) in enumerate(jobs, 1):
    print(f"{i:03d} {t} {p} {z}")
PY
        echo "f2-o$o-$t-$p${z}s|90|$SELF __seq $o $t $p $z"
    done
}
# shellcheck disable=SC2329
jobs_f3() {
    local r k i g sh q gaps=(0.5 1 2 3) shapes=(d64 card) spec
    for r in 1 2 3 4 5; do
        for ((k = 0; k < 16; k++)); do
            i=$(((k + 3 * r) % 16))
            g=${gaps[$((i / 4))]}
            sh=${shapes[$(((i / 2) % 2))]}
            q=g
            ((i % 2)) && q=c
            spec=$VK_D64
            [[ $sh == card ]] && spec=$VK_CARD_D
            echo "f3-$sh-$q-gap$g-r$r|60|$V burst name=f3$sh-$q-gap$g-r$r,q=$q,$spec,blen=2.8,gap=$g,dur=1,idle=1"
        done
    done
}
# F4: the E4 base list (100 one-group dispatches, write-to-write barrier, compute-only queue) right after a warm-up
# of 80 million cycles in the same command buffer, against the same list without it; clock probes before and after.
# shellcheck disable=SC2329
jobs_f4() {
    local r lo hi base="bar=vv7,grid=1,body=0,n=100,reps=4,ts=1,stride=0,probe=1"
    for r in 1 2 3 4 5; do
        lo="name=low-r$r,q=c,$base"
        hi="name=high-r$r,q=c,$base,pre=80000000"
        if ((r % 2)); then echo "f4-r$r|60|$V chain $lo $hi"; else echo "f4-r$r|60|$V chain $hi $lo"; fi
        echo "f4-vkstudy-r$r|60|env VKS_SPV=$BIN/vkstudy/spv VKS_OUT=@RUN@/vkstudy $BIN/vkstudy/vkstudy dispatch"
    done
}
# F5: the stop limit lowered to 2 C above the hotspot at the load's start (the sampler must stop the load), and a
# control with the normal 88 C limit, on both Vulkan queues and in the HIP twin.
F5_VK="groups=64,local=32,every=64,recstride=8,cont=100"
F5_HIP="64,32,64,100,8"
# shellcheck disable=SC2329
jobs_f5() {
    local r q
    for r in 1 2; do
        for q in g c; do
            echo "f5-vk$q-trip-r$r|60|env VKG_HOT_STOP_C=+2 $V burst name=f5vk$q-trip-r$r,q=$q,$F5_VK,idle=1"
            echo "f5-vk$q-ctl-r$r|60|$V burst name=f5vk$q-ctl-r$r,q=$q,$F5_VK,idle=1"
        done
        echo "f5-hip-trip-r$r|60|$BIN/dense_twin --out @RUN@/f5-hip-trip-r$r --idle-ms 1000 --hot-stop-c +2 f5hip-trip-r$r,$F5_HIP"
        echo "f5-hip-ctl-r$r|60|$BIN/dense_twin --out @RUN@/f5-hip-ctl-r$r --idle-ms 1000 f5hip-ctl-r$r,$F5_HIP"
    done
}

# Largest 5 ms sampler interval of a job (ms), from the probes' sampler_interval verdicts.
sampler_largest() {
    grep -hoE 'largest [0-9.]+ ms \(limit 20\)' "$RUN_DIR/$1.log" | grep -oE '[0-9.]+' | sort -n | tail -1
}
# Highest reported power in a job's sysfs files (W), from the column power_W_reported.
power_max() {
    local f m=0 x
    for f in "$RUN_DIR/$1"-sysfs.csv "$RUN_DIR/$1"-dense-*-sysfs.csv "$RUN_DIR/$1"-pred-*sysfs.csv; do
        [[ -f $f ]] || continue
        x=$(awk -F, 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "power_W_reported") c = i; next}
            c && $c + 0 > m {m = $c + 0} END {print m + 0}' "$f")
        awk -v a="$x" -v b="$m" 'BEGIN {exit !(a > b)}' && m=$x
    done
    echo "$m"
}
# Moves a job's files aside before it is repeated.
set_aside() {
    local name=$1 n=$2 f
    mkdir -p "$RUN_DIR/superseded"
    for f in "$RUN_DIR/$name.log" "$RUN_DIR/$name.dense.txt" "$RUN_DIR/$name"-*; do
        [[ -e $f ]] && mv "$f" "$RUN_DIR/superseded/attempt$n-$(basename "$f")"
    done
}

# Runs one job line; __pair lines carry their two option lists after a '|'.
run_line() {
    local line=$1 name limit cmd extra rc attempt gap pw
    IFS='|' read -r name limit cmd extra <<<"$line"
    cmd=${cmd//@RUN@/$RUN_DIR}
    for attempt in 1 2 3; do
        if [[ $cmd == *" __pair "* ]]; then
            local self out expect aopts
            read -r self _ out expect aopts <<<"$cmd"
            gpu_job "$name" "$limit" "$self" __pair "$out" "$expect" "${aopts//=/ }" "$extra"
        else
            # shellcheck disable=SC2086 # cmd is a word list on purpose
            gpu_job "$name" "$limit" $cmd
        fi
        rc=$?
        pw=$(power_max "$name")
        awk -v p="$pw" 'BEGIN {exit !(p > 317)}' && sum "- POWER: $name reported $pw W in a sample (above 317 W)"
        # the thermal rule: a whole-card job must not run with a blind watch (sampler gap above 50 ms)
        [[ $name == *card* ]] || break
        gap=$(sampler_largest "$name")
        if [[ -n $gap ]] && awk -v g="$gap" 'BEGIN {exit !(g > 50)}' && ((attempt < 3)); then
            sum "- $name: sampler gap $gap ms (above 50 ms) in a whole-card job; set aside, repeated after 60 s"
            set_aside "$name" "$attempt"
            sleep 60
            continue
        fi
        break
    done
    local judged=
    case $name in
        p1-cal-vkstudy) judged=$(judge_vkstudy "$RUN_DIR/$name.log") || rc=1 ;;
        p2-cal-vksched) judged=$(judge_vksched "$RUN_DIR/$name.log") || rc=1 ;;
        p4-cal-hip) judged=$(judge_hipramp "$RUN_DIR/$name.log") || rc=1 ;;
    esac
    [[ -n $judged ]] && echo "$judged" >>"$RUN_DIR/$name.log"
    dense_analysis "$name" || rc=1
    record "$name" "$rc"
}

# The dense-load jobs' clocks and checks come from scripts/gprobe_dense.py (CPU only, after the job).
dense_analysis() {
    local name=$1 out=$RUN_DIR/$1.dense.txt rc=0
    case $name in
        e1-hip-* | e1-vk* | e1-trip-* | e2-* | e3-* | break-e2-* | f1-* | f2-* | f3-* | break-f1-*)
            python3 -I "$ROOT/scripts/gprobe_dense.py" job "$RUN_DIR" "$name" "${@:2}" >"$out" 2>&1 || rc=1
            cat "$out" >>"$RUN_DIR/$name.log"
            ;;
        f5-*-trip-*)
            python3 -I "$ROOT/scripts/gprobe_dense.py" job "$RUN_DIR" "$name" --expect-stop >"$out" 2>&1 || rc=1
            cat "$out" >>"$RUN_DIR/$name.log"
            ;;
        f5-*-ctl-*)
            python3 -I "$ROOT/scripts/gprobe_dense.py" job "$RUN_DIR" "$name" --expect-no-stop >"$out" 2>&1 || rc=1
            cat "$out" >>"$RUN_DIR/$name.log"
            ;;
        e1-listing)
            python3 -I "$ROOT/scripts/gprobe_dense.py" listing "$RUN_DIR/asm/gdense_l32_w32.s" "$RUN_DIR/asm/dense_twin.s" \
                >"$out" 2>&1 || rc=1
            python3 -I "$ROOT/scripts/gprobe_dense.py" listing "$RUN_DIR/asm/gdense_l1024_w32.s" "$RUN_DIR/asm/dense_twin.s" \
                >>"$out" 2>&1 || rc=1
            cat "$out" >>"$RUN_DIR/$name.log"
            ;;
    esac
    return "$rc"
}

phase_run() {
    local item line
    sum "## run (sysfs $DRM_DEVICE; card configuration at start: $CONFIG0; LACT profile $(lact_profile))"
    [[ -x $V && -x $BIN/sched_share && -x $BIN/vksched/vksched && -x $BIN/sched_launch && -x $BIN/vkstudy/vkstudy &&
        -x $BIN/dense_twin ]] || {
        sum "- programs not built"
        FAILED=1
        return
    }
    for item in "${ITEMS[@]}"; do
        sum "### $item"
        while IFS= read -r line; do
            # GPROBE_ONLY (development runs): only the jobs whose names match this extended regular expression
            [[ -n ${GPROBE_ONLY:-} && ! ${line%%|*} =~ ${GPROBE_ONLY} ]] && continue
            run_line "$line"
        done < <("jobs_$item")
    done
}

phase_breaks() {
    local item rc log
    sum "## deliberate breaks (each must FAIL)"
    for item in "${ITEMS[@]}"; do
        case $item in
            p1)
                gpu_job break-p1-dependency 60 "$V" chain name=brk_dep_none,q=g,bar=none,check=1,reps=50
                record_break break-p1-dependency $?
                log=$RUN_DIR/break-p1-vkstudy.log
                if [[ -f $REF/p1-cal-vkstudy.log ]]; then
                    judge_vkstudy "$REF/p1-cal-vkstudy.log" 2.0 3.0 >"$log"
                    rc=$?
                    sed -i 's/^VERDICT gprobe vkstudy_calibration:/VERDICT gprobe vkstudy_calibration (DELIBERATE BREAK: limits 2.0..3.0):/' "$log"
                    record_break break-p1-vkstudy "$rc"
                else
                    sum "- break-p1-vkstudy: not run (no p1-cal-vkstudy.log in this run folder)"
                    FAILED=1
                fi
                ;;
            p2)
                gpu_job break-p2-swap 60 "$V" lone --swap-labels
                record_break break-p2-swap $?
                log=$RUN_DIR/break-p2-vksched-swap.log
                if [[ -f $REF/p2-cal-vksched.log ]]; then
                    judge_vksched "$REF/p2-cal-vksched.log" swap >"$log"
                    record_break break-p2-vksched-swap $?
                else
                    sum "- break-p2-vksched-swap: not run (no p2-cal-vksched.log in this run folder)"
                    FAILED=1
                fi
                ;;
            p3)
                log=$RUN_DIR/break-p3-shift.log
                if [[ -f $REF/p3-room-A.csv ]]; then
                    python3 -I "$ROOT/scripts/sched_pair.py" --preempt "$REF/p3-room-A.csv" "$REF/p3-room-B.csv" \
                        --expect overlap --shift-b-ms 300 >"$log" 2>&1
                    record_break break-p3-shift $?
                else
                    sum "- break-p3-shift: not run (no p3-room files in this run folder)"
                    FAILED=1
                fi
                ;;
            p4)
                log=$RUN_DIR/break-p4-hipramp.log
                if [[ -f $REF/p4-cal-hip.log ]]; then
                    judge_hipramp "$REF/p4-cal-hip.log" 1700 1900 >"$log"
                    rc=$?
                    sed -i 's/^VERDICT gprobe hip_ramp_calibration:/VERDICT gprobe hip_ramp_calibration (DELIBERATE BREAK: limits of the Vulkan plateau):/' "$log"
                    record_break break-p4-hipramp "$rc"
                else
                    sum "- break-p4-hipramp: not run (no p4-cal-hip.log in this run folder)"
                    FAILED=1
                fi
                ;;
            e1)
                log=$RUN_DIR/break-e1-windows.log
                if [[ -f $REF/e1-vkg-r1-dense-vkg-card-r1-groups.csv ]]; then
                    python3 -I "$ROOT/scripts/gprobe_dense.py" job "$REF" e1-vkg-r1 --break-windows >"$log" 2>&1
                    record_break break-e1-windows $?
                else
                    sum "- break-e1-windows: not run (no e1-vkg-r1 files in this run folder)"
                    FAILED=1
                fi
                log=$RUN_DIR/break-e1-listing.log
                if [[ -f $REF/asm/gburst_l64.s ]]; then
                    # the old burst shader's listing in place of the dense one: its loop is not 128 v_fma_f32
                    python3 -I "$ROOT/scripts/gprobe_dense.py" listing "$REF/asm/gburst_l64.s" "$REF/asm/dense_twin.s" \
                        >"$log" 2>&1
                    record_break break-e1-listing $?
                else
                    sum "- break-e1-listing: not run (no listing in this run folder)"
                    FAILED=1
                fi
                ;;
            e2)
                # odd groups stop after a tenth of the burst: the density check must fail
                gpu_job break-e2-skew 60 "$V" burst "name=skew,q=g,$VK_CU,blen=2,cad=16.7,dur=0.3,idle=0.2,skew=1"
                rc=$?
                dense_analysis break-e2-skew || rc=1
                record_break break-e2-skew "$rc"
                ;;
            f1)
                # E2's whole-card setting (a time check after every chunk) judged by the dense fill limit: must fail
                gpu_job break-f1-fill 60 "$V" burst "name=brkfill,q=g,groups=64,local=1024,every=1,recevery=64,recstride=8,minfill=0.8,blen=2,cad=16.7,dur=0.2,idle=1"
                rc=$?
                dense_analysis break-f1-fill || rc=1
                record_break break-f1-fill "$rc"
                ;;
            f4)
                # a warm-up far too short to step the clock: the warm_clock check must fail
                gpu_job break-f4-nowarm 60 "$V" chain "name=nowarm,q=c,bar=vv7,grid=1,body=0,n=100,reps=3,ts=1,stride=0,probe=1,pre=1000"
                record_break break-f4-nowarm $?
                ;;
            f5)
                # the verdicts asked the wrong way round of the first round's files: both must fail
                for x in "vkg-ctl-r1 --expect-stop ctl-as-trip" "vkg-trip-r1 --expect-no-stop trip-as-ctl"; do
                    read -r j o b <<<"$x"
                    log=$RUN_DIR/break-f5-$b.log
                    if compgen -G "$RUN_DIR/f5-$j-dense-*-groups.csv" >/dev/null; then
                        python3 -I "$ROOT/scripts/gprobe_dense.py" job "$RUN_DIR" "f5-$j" "$o" >"$log" 2>&1
                        record_break "break-f5-$b" $?
                    else
                        sum "- break-f5-$b: not run (no f5-$j files in this run folder)"
                        FAILED=1
                    fi
                done
                ;;
            e8)
                log=$RUN_DIR/break-e8-nogap.log
                if [[ -f $REF/p3-sleep-A.csv ]]; then
                    # the detector's verdict asked of A without an injected gap
                    python3 -I "$ROOT/scripts/sched_pair.py" --preempt "$REF/p3-sleep-A.csv" "$REF/p3-sleep-B.csv" \
                        --expect gap >"$log" 2>&1
                    record_break break-e8-nogap $?
                else
                    sum "- break-e8-nogap: not run (no p3-sleep files in this run folder)"
                    FAILED=1
                fi
                ;;
        esac
    done
}

PHASE=${1:-all}
shift || true
ITEMS=("$@")
((${#ITEMS[@]})) || ITEMS=(p1 p2 p3 p4 e1 e2 e3 e4 e5 e6 e7 e8 f1 f2 f3 f4 f5)
for i in "${ITEMS[@]}"; do [[ $i =~ ^(p[1-4]|e[1-8]|f[1-5])$ ]] || die "unknown item $i; use p1 p2 p3 p4 e1 ... e8 f1 ... f5"; done

sum "# Gaming probes run, $(date -u +%FT%TZ), phase $PHASE, items ${ITEMS[*]}"
sum "- card sysfs $DRM_DEVICE; kernel $(uname -r); $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | tr -s ' ')"
sum "- STUDY_REPO $STUDY_REPO; LACT profile $(lact_profile); power1_cap $(($(cat "$HWMON/power1_cap") / 1000000)) W (a 1.2x regulator-level power-limit mod is in the card: reported power is scaled down)"
case $PHASE in
    build) phase_build ;;
    run | breaks | all)
        if [[ $PHASE == all ]]; then phase_build || die "build failed; see $RUN_DIR"; fi
        if [[ $(lact_profile) != Testing ]]; then
            sum "- NOT RUN: LACT profile is '$(lact_profile)', not Testing"
            FAILED=1
        else
            start_sampler
            [[ $PHASE != breaks ]] && phase_run
            [[ $PHASE != run ]] && phase_breaks
            stop_sampler
        fi
        ;;
    *) die "unknown phase $PHASE; use build, run, breaks or all" ;;
esac
sum "- highest hotspot seen: $HOTTEST C; LACT profile at the end: $(lact_profile)"
echo "study-gaming-probes: results in $RUN_DIR"
exit "$FAILED"
