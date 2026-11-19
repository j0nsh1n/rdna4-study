#!/usr/bin/env bash
# LLM data dive of the RDNA 4 study: what llama.cpp shows about the RX 9070 XT (gfx1201) and its driver stack.
# Measurement only: no llama.cpp kernel code is changed, nothing here tries to make llama.cpp faster.
#
# Usage: scripts/study-llm.sh <mode>
#   setup      no GPU: card record, build record (git hash, binary times, linked libraries, graph switch), model sizes
#              and checksums, memory, LACT profile, memory clock
#   selftest   no GPU: the gate's deliberate breaks (each must FAIL), plus the Python helpers' own breaks
#   probe      step 2: the profiler against a known answer (the study's bandwidth probe), counters, graph tracing
#   known      step 3: known answers on the 3B (llama-bench, test-backend-ops) on both builds
#   shapes     step 4 (A3): test-backend-ops perf at the models' matrix shapes, f8_e4m3 against f16 against q8_0
#   profile    step 5 (A1, A2, C2, C3): per-kernel traces of pp512 and tg128, profiled and unprofiled control
#   grid       steps 6 and 7 (B1, B2, B4): prompt sizes, depth, parallel streams
#   quality    step 8 (B3): perplexity and KL divergence against a reference-logits file
#   behaviour  step 9 (C1 to C4): clocks, power, temperature and memory clock under prompt and generation loops
#   builds     (CPU only) the A5 build: Q8_0 through hipBLAS, a compile-time switch, in a new build folder
#   a5         the A5 build against the normal Q8_0 build (prompt speed and a kernel trace)
#   determinism two identical 50-chunk runs per model; the logits files are compared with cmp and the evidence kept
#   r2q        round 2, item 1: perplexity and KL divergence against the exact-weights F32 reference (refs in LLM_RUN)
#   r2mallbuild (CPU) build src/widen_cache.hip and check its ISA; r2mallcheck its known answer and deliberate break; r2mall the table
#   r3sys, r3counters   round 3 Q4: kernel/driver settings (read only); the 13 counter sets with ROCm 10.1 and with ROCm 10.0's rocprofv3
#   r3build, r3check    round 3 (CPU) build the probes pair_overlap, power_mix and widen_cache_nocal, ISA checks and breaks; (GPU) their breaks on the card
#   r3fix               round 3 fix round (GPU, after r3build): power_mix known answer on a dirty result buffer, before and after the fix
#   r3q3, r3pairs, r3power, r3cnt   round 3: Q3 interleaved binaries; Q5 kernel pairs on two clocks; Q1c matched-bandwidth power; Q1b counters on the decode kernels
#   reanalyse  (no GPU) the trace analysis again over the traces already in LLM_RUN
#   all        setup, selftest, probe, known, shapes, profile, grid, quality, behaviour (not builds, a5, determinism)
#   dev        <job name> <hot limit C> <timeout s> <command...>: one gated job, for development
#   grid3b, profile3b: the grid or the profile for the 3B only
# Environment: LLM_RUN (continue in an existing run folder), LLM_BUILD_ROOT (folder with the llama.cpp builds, local.env
# and ppl-text.txt; default under $HOME/Projects), LLM_MODELS (the GGUF folder; default LLM_BUILD_ROOT/../models), CARD_ENV (card record; default runs/latest/card.env).
# Results: runs/<UTC>-llm/ (git-ignored): one log and one .meta (clocks, temperatures, exit) per GPU job, config.txt,
# summary.md. Exit 1 if anything failed, 2 usage or setup error.
# GPU jobs: the shared lock, nice, a time limit, the HIP device index from the card record (the machine has an
# integrated GPU too), at least 8 GB available memory, a hotspot below the start limit (60 C for loops over 10 s, 70 C
# for short benches), stopped at 95 C by the sampler's own guard (`watch_card guard`, hotspot read every 20 ms, TERM to the job's
# process group and to its systemd scope, KILL 0.5 s later; an unreadable or non-numeric hotspot is a stop too; also stopped at 100 C of the memory
# temperature (temp3; an unreadable memory sensor is only a WARNING); a guard that dies mid-job stops the job at once: GUARD LOST, exit 10), no game running (a game found: wait up to 10 min, then skip the
# job, exit 7). Before every job the LACT profile must be the expected one (LLM_EXPECT_PROFILE; "Testing" for the old modes, "Default" for the r4* modes of round 4, which also checks the
# overclock settings: 374 W cap, +150 MHz offset, -50 mV) and the card configuration unchanged, else all GPU
# work stops. current_uclk (memory clock) is logged before and after every job; a job whose memory clock differs from
# the run's first reading is labelled with it and never merged with the others. The script never changes LACT, sysfs
# power or clock files, or the performance level.
# Test hooks (read only with LLM_SELFTEST=1): LLM_RUN_DIR LLM_LOCK LLM_MEMINFO LLM_PROC LLM_LACT_CONFIG LLM_HOT_FILE
# LLM_UCLK_FILE LLM_LOCK_WAIT LLM_GAME_WAIT LLM_GUARD_C LLM_GUARD_MS LLM_HOT_WAIT LLM_MEM_CAP LLM_BIG_CAP LLM_CARDCFG_FILE LLM_MEM_FILE
# LLM_GUARD_MEM_C.
# Expected LACT profile: LLM_EXPECT_PROFILE (a real setting, not a test hook). Default: "Default" for modes starting with r4, else "Testing". LACT's
# profile "Default" is the daemon's top-level setting: current_profile is null in /etc/lact/config.yaml (read only; `lact cli profile get` agrees).
# Memory cap: one owner-approved exception: the job named r8x-ref-* (the exact 8B reference run, round 4 follow-up) runs under MemoryMax=16G
# (BIG_CAP); every other job keeps the cap below. Memory cap: every GPU job runs in its own systemd user scope with MemoryMax=10G and MemorySwapMax=0 (the machine rule
# for a job; LLM_MEM_CAP changes it in a selftest only). The kernel kills a job that goes past it (exit 137) instead of the
# machine going into swap; no scope available means the job is not started (exit 9). Round 2's 8B F32 attempts ran
# BEFORE this cap existed.
# Exit codes of a GPU job: 0 ok, 3 profile or configuration changed (all GPU work stopped), 4 hotspot too high to
# start, 5 below 8 GB available, 6 lock wait timed out, 7 a game is running, 8 stopped at the hotspot limit, 9 no memory
# cap available (job not started), 10 the thermal guard is missing, stale or failed to start (job not started or stopped), 137 killed before its time limit (the memory cap). The checks
# run twice: before the lock wait (which may wait for a game or a cool-down) and again after the lock is acquired
# (no waiting there: the job is not started and the lock is released).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF=$ROOT/scripts/study-llm.sh
cd "$ROOT" || exit 2

SELFTEST=0
[[ ${LLM_SELFTEST:-} == 1 ]] && SELFTEST=1
TEST_HOOKS=(LLM_RUN_DIR LLM_LOCK LLM_MEMINFO LLM_PROC LLM_LACT_CONFIG LLM_HOT_FILE LLM_UCLK_FILE LLM_LOCK_WAIT
    LLM_GAME_WAIT LLM_GUARD_C LLM_GUARD_MS LLM_HOT_WAIT LLM_MEM_CAP LLM_BIG_CAP LLM_CARDCFG_FILE LLM_MEM_FILE LLM_GUARD_MEM_C)
hook() {
    if ((SELFTEST)) && [[ -n ${!1:-} ]]; then printf '%s\n' "${!1}"; else printf '%s\n' "$2"; fi
}
if ((SELFTEST)); then
    echo "!!! SELF-TEST MODE (LLM_SELFTEST=1): test hooks are read. Never set LLM_SELFTEST for a real run. !!!" >&2
else
    ignored=""
    for v in "${TEST_HOOKS[@]}"; do [[ -n ${!v:-} ]] && ignored+=" $v"; done
    [[ -n $ignored ]] && echo "study-llm: ignored (test hooks, read only with LLM_SELFTEST=1):$ignored" >&2
fi

ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1

# shellcheck source=/dev/null
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
BUILD_ROOT=${LLM_BUILD_ROOT:-$HOME/Projects/rdna3-fp8/build}
LOCAL_ENV=$BUILD_ROOT/local.env
GPU_LOCK=""
# shellcheck source=/dev/null
[[ -r $LOCAL_ENV ]] && GPU_LOCK=$(. "$LOCAL_ENV" >/dev/null 2>&1; echo "${GPU_LOCK:-}")
GPU_LOCK=${STUDY_GPU_LOCK:-${GPU_LOCK:-$ROOT/runs/gpu.lock}}
GPU_LOCK=$(hook LLM_LOCK "$GPU_LOCK")
LACT_CONFIG=$(hook LLM_LACT_CONFIG /etc/lact/config.yaml)
MEMINFO=$(hook LLM_MEMINFO /proc/meminfo)
PROC=$(hook LLM_PROC /proc)
LOCK_WAIT=$(hook LLM_LOCK_WAIT 900)
GAME_WAIT=$(hook LLM_GAME_WAIT 600)
GUARD_C=$(hook LLM_GUARD_C 95)
GUARD_MS=$(hook LLM_GUARD_MS 20)
GUARD_MEM_C=$(hook LLM_GUARD_MEM_C 100)   # memory (VRAM) temperature stop limit; the card reports 108 C as critical
HOT_WAIT=$(hook LLM_HOT_WAIT 300)
MEM_CAP=$(hook LLM_MEM_CAP 10G)
BIG_CAP=$(hook LLM_BIG_CAP 16G)   # only for the one job named r8x-ref-*, see gpu_job

UNMOD=$BUILD_ROOT/llama.cpp-latest/build-gfx1201-rocm10/bin
PATCHED=$BUILD_ROOT/llama.cpp-latest-patched/build-gfx1201-rocm10/bin
MODELS=${LLM_MODELS:-$BUILD_ROOT/../models}
TEXT=$BUILD_ROOT/ppl-text.txt
WATCH=$ROOT/build/study/bin/watch_card
# The thermal guard is part of watch_card: build it when it is missing or older than its source (same flags as study-home.sh).
ensure_watch() {
    [[ -x $WATCH && $WATCH -nt $ROOT/src/study/watch_card.cpp && $WATCH -nt $ROOT/src/study/watch_card_sysfs.h ]] && return 0
    mkdir -p "$(dirname "$WATCH")"
    nice -n 10 g++ -std=c++20 -O2 -Wall -Wextra -Werror -o "$WATCH" "$ROOT/src/study/watch_card.cpp"
}
BANDWIDTH=$ROOT/build/gfx1201-rocm10/bandwidth
PY=(python3 -I)

CARD_ENV=${CARD_ENV:-$ROOT/runs/latest/card.env}
[[ -r $CARD_ENV ]] || { echo "study-llm: no card record at $CARD_ENV" >&2; exit 2; }
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 && -d ${DRM_DEVICE:-} && -d ${HWMON:-} ]] || { echo "study-llm: bad card record $CARD_ENV" >&2; exit 2; }
[[ ${HIP_INDEX:-} =~ ^[0-9]+$ ]] || { echo "study-llm: the card record has no HIP_INDEX" >&2; exit 2; }
CARD_ID="$(cat "$DRM_DEVICE/vendor" 2>/dev/null):$(cat "$DRM_DEVICE/device" 2>/dev/null)"
[[ $CARD_ID == 0x1002:0x7550 ]] || { echo "study-llm: $DRM_DEVICE is PCI $CARD_ID, not the RX 9070 XT" >&2; exit 2; }
export DRM_DEVICE HWMON HIP_VISIBLE_DEVICES=$HIP_INDEX

RUN_DIR=${LLM_RUN:-}
[[ -n $RUN_DIR ]] || RUN_DIR=$(hook LLM_RUN_DIR "$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-llm")
mkdir -p "$RUN_DIR" || { echo "study-llm: cannot create $RUN_DIR" >&2; exit 2; }
export LLM_RUN=$RUN_DIR
SUMMARY=$RUN_DIR/summary.md
FAILED=0
STOPPED=0
HOTTEST=0

sum() { echo "$*" >>"$SUMMARY"; }
die() { echo "study-llm: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------------------------------------------
# Card readings (read-only)
avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' "$MEMINFO"; }
hotspot_c() {
    local f
    f=$(hook LLM_HOT_FILE "$HWMON/temp2_input")
    echo $(($(cat "$f" 2>/dev/null || echo 999000) / 1000))
}
uclk_mhz() {
    local f
    f=$(hook LLM_UCLK_FILE "")
    if [[ -n $f ]]; then cat "$f"; else od -An -tu2 -j58 -N2 "$DRM_DEVICE/gpu_metrics" 2>/dev/null | tr -d ' '; fi
}
lact_profile() {
    local v
    [[ -r $LACT_CONFIG ]] || { echo unreadable; return; }
    v=$(awk '/^current_profile:/ {print $2}' "$LACT_CONFIG" 2>/dev/null)
    case $v in null | "" | "~") echo Default ;; *) echo "$v" ;; esac
}
EXPECT_PROFILE=${LLM_EXPECT_PROFILE:-}
if [[ -z $EXPECT_PROFILE ]]; then
    if [[ ${1:-} == r4* || ${1:-} == r8x* ]]; then EXPECT_PROFILE=Default; else EXPECT_PROFILE=Testing; fi
fi
export LLM_EXPECT_PROFILE=$EXPECT_PROFILE
# the settings that belong to the profile "Default" (the overclock); empty output means they match
profile_settings_wrong() {
    local c
    c=$(card_config)
    [[ $EXPECT_PROFILE == Default ]] || return 0
    [[ $c == *"power1_cap=374000000"* ]] || echo "power cap is not 374 W"
    [[ $c == *"150Mhz"* ]] || echo "core offset is not +150 MHz"
    [[ $c == *"-50mV"* ]] || echo "voltage offset is not -50 mV"
    return 0
}
card_config() {
    local f
    f=$(hook LLM_CARDCFG_FILE "")
    if [[ -n $f ]]; then cat "$f"; return; fi
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") perf_level=$(cat "$DRM_DEVICE/power_dpm_force_performance_level") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null)"
}
snap() {
    echo "uclk=$(uclk_mhz) sclk=$(grep '\*' "$DRM_DEVICE/pp_dpm_sclk" | tr -s ' ' | tr -d '\n') hotspot=$(hotspot_c)C" \
        "power_reported=$(($(cat "$HWMON/power1_average") / 1000000))W perf=$(cat "$DRM_DEVICE/power_dpm_force_performance_level")"
}
CONFIG0=$(card_config)
RUN_UCLK=$(uclk_mhz)

# ---------------------------------------------------------------------------------------------------------------
# Is a game running? /proc only, nothing is signalled (copied from study-home.sh; that file is too large to source).
GAME_PATH_RE='steamapps[/\\][c]ommon[/\\]|compatibilitytools[.]d[/\\]'
game_signs() {
    local pid name d found=1
    while read -r pid name; do
        [[ $pid =~ ^[0-9]+$ ]] || continue
        if [[ $name == GameThread ]]; then
            echo "card client $pid is named GameThread"
            found=0
        elif grep -qaE "$GAME_PATH_RE" "$PROC/$pid/cmdline" 2>/dev/null; then
            echo "card client $pid ($name) runs from a game or Proton folder"
            found=0
        fi
    done < <(WATCH_PROC=$PROC "$WATCH" clients --pids 2>/dev/null)
    while read -r d; do
        [[ -n $d ]] || continue
        d=${d%/cmdline}
        echo "process ${d##*/} ($(cat "$d/comm" 2>/dev/null)) has a game or Proton path in its command line"
        found=0
    done < <(grep -laE "$GAME_PATH_RE" "$PROC"/[0-9]*/cmdline 2>/dev/null | head -5)
    return "$found"
}

# ---------------------------------------------------------------------------------------------------------------
# Locked runner: re-entered under flock, so clocks are read while this job holds the GPU. The job runs in the
# background under nice and timeout (timeout leads its own process group); `watch_card guard` reads the hotspot every GUARD_MS
# (20 ms) and ends that group at the hotspot limit or when the sensor cannot be read.
if [[ ${1:-} == __locked ]]; then
    log=$2
    meta=$3
    tmo=$4
    hlim=$5
    shift 5
    # every gate again, now that this job holds the lock: the wait may have been long (another project's job)
    if [[ $(lact_profile) != "$EXPECT_PROFILE" ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)', not '$EXPECT_PROFILE', after the lock wait" >>"$log"
        exit 3
    fi
    if [[ -n $(profile_settings_wrong) ]]; then
        echo "NOT STARTED: the card settings do not match '$EXPECT_PROFILE' ($(profile_settings_wrong | tr '\n' ';')) after the lock wait" >>"$log"
        exit 3
    fi
    if [[ $(card_config) != "${LLM_CONFIG0:-$CONFIG0}" ]]; then
        echo "NOT STARTED: card configuration changed since the start of the run, seen after the lock wait" >>"$log"
        exit 3
    fi
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait ($(avail_gb) GB)" >>"$log"
        exit 5
    fi
    if signs=$(game_signs); then
        echo "NOT STARTED: a game is running after the lock wait ($(echo "$signs" | head -2 | tr '\n' ';'))" >>"$log"
        exit 7
    fi
    if (($(hotspot_c) >= hlim)); then
        echo "NOT STARTED: hotspot $(hotspot_c) C is at or above the start limit $hlim C after the lock wait" >>"$log"
        exit 4
    fi
    if [[ ! -x $WATCH || $WATCH -ot $ROOT/src/study/watch_card.cpp ]]; then
        echo "NOT STARTED: the thermal guard ($WATCH) is missing or older than src/study/watch_card.cpp; a GPU job never runs without it" >>"$log"
        exit 10
    fi
    MEM_CAP=${LLM_CAP_THIS_JOB:-$MEM_CAP}
    CAPRUN=(systemd-run --user --scope -q -p "MemoryMax=$MEM_CAP" -p MemorySwapMax=0)
    if ! "${CAPRUN[@]}" true >/dev/null 2>&1; then
        echo "NOT STARTED: no memory cap available (systemd-run --user --scope failed); a GPU job never runs uncapped" >>"$log"
        exit 9
    fi
    {
        echo "uclk_before=$(uclk_mhz)"
        echo "snap_before=$(snap)"
        echo "mem_cap=$MEM_CAP swap_cap=0"
    } >"$meta"
    echo "clocks before: $(snap)" >>"$log"
    t0=$(date +%s.%N)
    # the job's own scope has a name the guard knows, so a stop reaches every process in it, setsid children included
    SCOPE="llmjob-$$-$(date +%s%N).scope"
    nice -n 10 timeout --signal=TERM --kill-after=20 "$tmo" "${CAPRUN[@]}" --unit="${SCOPE%.scope}" "$@" >>"$log" 2>&1 &
    jp=$!
    "$WATCH" guard --pid "$jp" --hot-file "$(hook LLM_HOT_FILE "$HWMON/temp2_input")" --stop-hot "$GUARD_C" --period-ms "$GUARD_MS" \
        --mem-file "$(hook LLM_MEM_FILE "$HWMON/temp3_input")" --stop-mem "$GUARD_MEM_C" --scope "$SCOPE" \
        --kill-after-ms 500 --log "$log" --csv "${meta%.meta}.guard.csv" 2>/dev/null &
    gp=$!
    sleep 0.3
    grc=-1
    if ! kill -0 "$gp" 2>/dev/null; then
        # the guard ends by itself only when the job has ended (0), at a stop (8), or because it could not start (2)
        wait "$gp"
        grc=$?
        if ((grc != 0 && grc != 8)); then
            echo "THERMAL GUARD FAILED TO START (exit $grc): the job was stopped (PID $jp)" >>"$log"
            kill -TERM "$jp" 2>/dev/null
            wait "$jp" 2>/dev/null
            exit 10
        fi
    fi
    hot_stop=0
    guard_lost=0
    if ((grc < 0)); then
        # watch the guard while the job runs: the first of the two to end. A guard that ends with anything but 0 (the job
        # ended) or 8 (it stopped the job) while the job still runs has died or been killed: the job is stopped at once.
        wait -n -p who "$jp" "$gp"
        wrc=$?
        if [[ ${who:-} == "$gp" ]]; then
            grc=$wrc
            if ((grc != 0 && grc != 8)) && kill -0 "$jp" 2>/dev/null; then
                guard_lost=1
                echo "GUARD LOST: the thermal guard ended with status $grc while the job ran; the job is being stopped (PID $jp, scope $SCOPE)" >>"$log"
                kill -TERM -- "-$jp" 2>/dev/null || kill -TERM "$jp" 2>/dev/null
                systemctl --user kill --signal=TERM --kill-whom=all "$SCOPE" >/dev/null 2>&1 &
                for _ in 1 2 3 4 5 6 7 8 9 10; do
                    kill -0 "$jp" 2>/dev/null || break
                    sleep 0.05
                done
                if kill -0 "$jp" 2>/dev/null; then
                    kill -KILL -- "-$jp" 2>/dev/null || kill -KILL "$jp" 2>/dev/null
                fi
                systemctl --user kill --signal=KILL --kill-whom=all "$SCOPE" >/dev/null 2>&1
            fi
            wait "$jp"
            rc=$?
        else
            rc=$wrc
            wait "$gp"
            grc=$?
        fi
    else
        wait "$jp"
        rc=$?
    fi
    ((grc == 8)) && hot_stop=1
    hmax=$(sed -n 's/^guard: .*hotspot max \([0-9][0-9]*\) C.*/\1/p' "$log" | tail -1)
    hmax=${hmax:-unreadable}
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f", b - a}')
    if ((rc == 137)) && ((!hot_stop)) && awk -v w="$wall" -v t="$tmo" 'BEGIN {exit !(w < t)}'; then
        echo "MEMORY CAP: job killed (exit 137) after $wall s, before its time limit of $tmo s; the cap is $MEM_CAP with no swap" >>"$log"
    elif ((rc == 124 || rc == 137)); then
        echo "TIMEOUT: job stopped after $tmo s" >>"$log"
    fi
    echo "clocks after: $(snap)" >>"$log"
    {
        echo "uclk_after=$(uclk_mhz)"
        echo "snap_after=$(snap)"
        echo "hotspot_max=$hmax"
        echo "thermal_stop=$hot_stop"
        echo "guard_ms=$GUARD_MS guard_exit=$grc guard_lost=$guard_lost"
        echo "wall_s=$wall"
        echo "exit=$rc"
    } >>"$meta"
    if ((grc != 0 && grc != 8)); then
        echo "THERMAL GUARD FAILED: watch_card guard exited $grc during the job" >>"$log"
        exit 10
    fi
    ((hot_stop)) && exit 8
    exit "$rc"
fi

# gpu_job <name> <hot-start-limit C> <timeout s> <command...>: one gated GPU job. Output: $RUN_DIR/<name>.log and
# .meta. Returns the job's status (codes in the header). Pre-job checks never wait forever.
gpu_job() {
    local name=$1 hlim=$2 tmo=$3 log meta waited h rc signs
    shift 3
    log=$RUN_DIR/$name.log
    meta=$RUN_DIR/$name.meta
    if ((STOPPED)); then
        sum "- $name: NOT RUN (GPU work stopped earlier)"
        return 3
    fi
    if ! ensure_watch; then
        sum "- $name: NOT RUN: the thermal guard (watch_card) could not be built; a GPU job never runs without it"
        FAILED=1
        return 10
    fi
    if [[ $(lact_profile) != "$EXPECT_PROFILE" ]]; then
        sum "- $name: NOT RUN: LACT profile is '$(lact_profile)', not '$EXPECT_PROFILE'; all GPU work stopped"
        STOPPED=1
        FAILED=1
        return 3
    fi
    if [[ -n $(profile_settings_wrong) ]]; then
        sum "- $name: NOT RUN: the card settings do not match '$EXPECT_PROFILE' ($(profile_settings_wrong | tr '\n' ';')); all GPU work stopped"
        STOPPED=1
        FAILED=1
        return 3
    fi
    if [[ $(card_config) != "$CONFIG0" ]]; then
        sum "- $name: NOT RUN: card configuration changed since the start of the run; all GPU work stopped"
        STOPPED=1
        FAILED=1
        return 3
    fi
    if (($(avail_gb) < 8)); then
        sum "- $name: NOT RUN: $(avail_gb) GB available (floor 8)"
        FAILED=1
        return 5
    fi
    waited=0
    while signs=$(game_signs); do
        if ((waited >= GAME_WAIT)); then
            sum "- $name: SKIPPED, a game was running for $GAME_WAIT s ($(echo "$signs" | head -2 | tr '\n' ';'))"
            FAILED=1
            return 7
        fi
        sleep 10
        waited=$((waited + 10))
    done
    waited=0
    while (($(hotspot_c) >= hlim)); do
        if ((waited >= HOT_WAIT)); then
            sum "- $name: NOT RUN: hotspot still $(hotspot_c) C (start limit $hlim) after $HOT_WAIT s"
            FAILED=1
            return 4
        fi
        sleep 5
        waited=$((waited + 5))
    done
    {
        echo "# $name: $*"
        echo "# start $(date -u +%Y-%m-%dT%H:%M:%SZ) profile=$(lact_profile) hotspot start limit $hlim C"
    } >"$log"
    local cap=$MEM_CAP
    [[ $name == r8x-ref-* ]] && cap=$BIG_CAP
    LLM_CAP_THIS_JOB="$cap" LLM_CONFIG0="$CONFIG0" flock -w "$LOCK_WAIT" -E 6 "$GPU_LOCK" "$SELF" __locked "$log" "$meta" "$tmo" "$hlim" "$@"
    rc=$?
    echo "# end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit $rc" >>"$log"
    if ((rc == 6)); then
        sum "- $name: NOT RUN: the GPU lock was held for $LOCK_WAIT s"
        FAILED=1
        return 6
    fi
    if ((rc >= 3 && rc <= 10)) && grep -q '^NOT STARTED' "$log"; then
        sum "- $name: NOT RUN after the lock wait: $(grep -m1 '^NOT STARTED' "$log" | cut -d: -f2-)"
        FAILED=1
        ((rc == 3)) && STOPPED=1
        return "$rc"
    fi
    h=$(hotspot_c)
    ((h > HOTTEST)) && HOTTEST=$h
    local ub ua label=""
    ub=$(sed -n 's/^uclk_before=//p' "$meta" 2>/dev/null)
    ua=$(sed -n 's/^uclk_after=//p' "$meta" 2>/dev/null)
    if [[ $ub != "$RUN_UCLK" || $ua != "$RUN_UCLK" ]]; then
        label="uclk=${ub}/${ua}"
        echo "uclk_label=$label" >>"$meta"
        sum "- $name: MEMORY CLOCK $label differs from the run's first reading ($RUN_UCLK): its bandwidth rows are quoted separately, never merged"
    fi
    if ((rc == 0)); then
        sum "- $name: ok (memory clock $ub MHz, hotspot max $(sed -n 's/^hotspot_max=//p' "$meta") C)"
    elif ((rc == 8)); then
        sum "- $name: STOPPED at the hotspot limit ($GUARD_C C)"
        FAILED=1
    elif grep -q '^MEMORY CAP' "$log"; then
        sum "- $name: KILLED by the memory cap ($MEM_CAP, no swap), exit $rc"
        FAILED=1
    else
        sum "- $name: FAILED (exit $rc)"
        FAILED=1
    fi
    if ((h >= GUARD_C)); then
        sum "- hotspot $h C after $name: all further GPU work stopped"
        STOPPED=1
        FAILED=1
    fi
    return "$rc"
}

# Wrapper run as the job command: starts the read-only sampler (20 ms), runs the command, then stops the sampler by
# its PID. Arguments: output folder, label, then the command.
if [[ ${1:-} == __watch ]]; then
    out=$2
    label=$3
    shift 3
    mkdir -p "$out"
    "$WATCH" record --out "$out" --label "$label" --period-ms 20 --seconds 0 >"$out.rec.log" 2>&1 &
    wp=$!
    for _ in $(seq 1 100); do [[ -e $out/ready ]] && break; sleep 0.1; done
    "$@" &
    cp=$!
    trap 'kill -TERM "$cp" 2>/dev/null' TERM INT
    wait "$cp"
    rc=$?
    wait "$cp" 2>/dev/null
    rc=$?
    sleep 1
    kill -TERM "$wp" 2>/dev/null
    wait "$wp" 2>/dev/null
    exit "$rc"
fi

# Hidden mode for the selftest: one gated job through the real code, exit = its status.
if [[ ${1:-} == __gatejob ]]; then
    shift
    gpu_job "$@"
    rc=$?
    echo "RC1=$rc STOPPED=$STOPPED"
    gpu_job "$1-next" "$2" 5 true
    echo "RC2=$? STOPPED=$STOPPED"
    cat "$SUMMARY"
    exit "$rc"
fi

# ---------------------------------------------------------------------------------------------------------------
# setup: no GPU
phase_setup() {
    local f=$RUN_DIR/config.txt d name
    sum ""
    sum "## setup (no GPU)"
    {
        echo "# Configuration record (read-only), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "CARD_ENV=$CARD_ENV HIP_INDEX=$HIP_INDEX (HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES)"
        echo "lact current_profile: $(lact_profile)"
        echo "card_config: $CONFIG0"
        echo "power1_cap_default: $(cat "$HWMON/power1_cap_default" 2>/dev/null)"
        echo "Note: a regulator-level power-limit mod (1.2x) is in the card; reported power is scaled down on the main rail."
        echo "current_uclk at start (gpu_metrics offset 58): $RUN_UCLK MHz"
        echo "snapshot: $(snap)"
        echo "--- pp_dpm_sclk"
        cat "$DRM_DEVICE/pp_dpm_sclk"
        echo "--- pp_dpm_mclk"
        cat "$DRM_DEVICE/pp_dpm_mclk"
        echo "available memory: $(avail_gb) GB; cores: $(nproc)"
        echo "kernel: $(uname -r); $(hipcc --version 2>/dev/null | head -1)"
        echo "gpu lock: $GPU_LOCK"
        echo "--- rocminfo agents (HIP order: the CPU is not a HIP device)"
        rocminfo 2>/dev/null | grep -E "^  Name:|Marketing Name" | paste - - | sed 's/  */ /g'
    } >"$f" 2>&1
    sum "- card record written to config.txt (LACT profile $(lact_profile), memory clock $RUN_UCLK MHz, hotspot $(hotspot_c) C)"
    [[ $(lact_profile) == "$EXPECT_PROFILE" && -z $(profile_settings_wrong) ]] || { sum "- LACT profile is not $EXPECT_PROFILE (or its settings differ): stop (FAILED)"; FAILED=1; }
    echo "lact cli profile get (read only): $(lact cli profile get 2>&1 | head -1)" >>"$f"
    {
        echo "# Build record, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for d in "$UNMOD" "$PATCHED"; do
            echo "== $d"
            local src
            src=$(cd "$d/../.." && pwd)
            echo "source tree HEAD: $(git -C "$src" rev-parse HEAD 2>/dev/null) $(git -C "$src" log -1 --format=%cs 2>/dev/null)"
            echo "git status lines: $(git -C "$src" status --short 2>/dev/null | wc -l); diff: $(git -C "$src" diff --shortstat 2>/dev/null)"
            for b in llama-bench llama-batched-bench llama-perplexity test-backend-ops libggml-hip.so.1.0.0; do
                echo "binary $b: $(stat -c '%y %s' "$d/$b" 2>/dev/null)"
            done
            grep -E "^(CMAKE_BUILD_TYPE|GGML_HIP_GRAPHS|GGML_HIP_NO_VMM|GGML_HIP_MMQ_MFMA|GGML_CUDA_FORCE_CUBLAS|GGML_CUDA_FORCE_MMQ|AMDGPU_TARGETS|GPU_TARGETS|CMAKE_HIP_COMPILER)[:=]" \
                "$d/../CMakeCache.txt" 2>/dev/null
            ldd "$d/llama-bench" 2>/dev/null | grep -E "hipblas|rocblas|amdhip" | sed 's/(0x.*//'
        done
    } >"$RUN_DIR/builds.txt" 2>&1
    sum "- build record written to builds.txt"
    {
        echo "# Models (size in bytes, sha256), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for name in llama32-3b-fp8-outq8 llama32-3b-q8_0 llama32-3b-auto llama31-8b-fp8-outq8 llama31-8b-q8_0 llama31-8b-auto; do
            echo "$name $(stat -c %s "$MODELS/$name.gguf" 2>/dev/null) $(sha256sum "$MODELS/$name.gguf" 2>/dev/null | cut -d' ' -f1)"
        done
        echo "ppl-text.txt $(stat -c %s "$TEXT") $(sha256sum "$TEXT" | cut -d' ' -f1)"
        echo "# bytes decode streams per token (READ from the GGUF tensor table): all tensor bytes minus token_embd.weight"
        echo "# streamed <model> <tensor bytes> <token_embd bytes> <streamed bytes>"
    } >"$RUN_DIR/models.txt" 2>&1
    sum "- model sizes and checksums written to models.txt"
    for name in llama32-3b-fp8-outq8 llama32-3b-q8_0 llama31-8b-fp8-outq8 llama31-8b-q8_0; do
        if ! "${PY[@]}" "$ROOT/scripts/llm_tables.py" streamed "$MODELS/$name.gguf" >>"$RUN_DIR/models.txt" 2>&1; then
            sum "- streamed bytes of $name could not be read (FAILED): $(tail -1 "$RUN_DIR/models.txt")"
            FAILED=1
        fi
    done
    sum "- streamed bytes per model (token_embd left out) written to models.txt"
    for f in "$UNMOD/llama-bench" "$PATCHED/llama-bench" "$UNMOD/test-backend-ops" "$PATCHED/test-backend-ops" "$WATCH" "$BANDWIDTH"; do
        [[ -x $f ]] || { sum "- missing tool: $f (FAILED)"; FAILED=1; }
    done
    [[ -n $GPU_LOCK ]] || { sum "- no GPU_LOCK found in $LOCAL_ENV (FAILED)"; FAILED=1; }
}

# ---------------------------------------------------------------------------------------------------------------
# selftest: the gate's deliberate breaks. Each case runs the real __gatejob code with fake /proc, meminfo, lock,
# LACT config, hotspot and memory-clock files, and checks the exit status and the summary line. A control case with
# nothing wrong must pass, so a check that cannot pass is also caught.
ST_FAIL=0
st_expect() { # name, want, got, note
    if [[ $2 == "$3" ]]; then
        echo "  ok    $1: $4"
        sum "- selftest $1: $4"
    else
        echo "  WRONG $1: wanted $2, got $3 ($4)"
        sum "- selftest $1: WRONG, wanted $2 got $3 ($4)"
        ST_FAIL=1
    fi
}
st_job() { # case dir, then gatejob args; env set by the caller through ST_ENV
    local d=$1
    shift
    rm -rf "$d"
    mkdir -p "$d"
    env LLM_SELFTEST=1 LLM_RUN="$d" LLM_RUN_DIR="$d" LLM_LOCK="$ST/lock" LLM_MEMINFO="$ST/meminfo" LLM_PROC="$ST/proc" \
        LLM_LACT_CONFIG="$ST/lact.yaml" LLM_HOT_FILE="$ST/hot" LLM_UCLK_FILE="$ST/uclk" LLM_MEM_FILE="$ST/mem" LLM_LOCK_WAIT=3 LLM_GAME_WAIT=2 \
        LLM_HOT_WAIT=2 WATCH_PDEV=0000:99:00.0 "${ST_ENV[@]}" "$SELF" __gatejob "$@" >"$d/out.txt" 2>"$d/err.txt"
}
st_reset() {
    ST_ENV=()
    printf 'MemAvailable:   20971520 kB\n' >"$ST/meminfo"
    printf 'current_profile: Testing\n' >"$ST/lact.yaml"
    echo 40000 >"$ST/hot"
    echo 50000 >"$ST/mem"
    echo 1408 >"$ST/uclk"
    rm -rf "$ST/proc"
    mkdir -p "$ST/proc"
    : >"$ST/lock"
}
# The fast thermal guard, one real runner path per case (fake hotspot file, dummy jobs). The fake file is replaced atomically, as
# sysfs is read whole. Writer timestamps and the dummy's TERM trap use the same wall clock, so the latency is
# (job got TERM) - (hotspot file changed), with a few ms for the `date` call of the trap. Also run alone: `selftest-guard`.
st_hot() { printf '%s\n' "$1" >"$ST/hot.new" && mv -f "$ST/hot.new" "$ST/hot"; }
st_mem() { printf '%s\n' "$1" >"$ST/mem.new" && mv -f "$ST/mem.new" "$ST/mem"; }
# the pid of the watch_card guard somewhere below process $1, found by following parents (never by name)
st_find_guard() {
    local c g
    for c in $(pgrep -P "$1" 2>/dev/null); do
        if [[ $(cat "/proc/$c/comm" 2>/dev/null) == watch_card ]]; then
            echo "$c"
            return 0
        fi
        g=$(st_find_guard "$c") && [[ -n $g ]] && { echo "$g"; return 0; }
    done
    return 1
}
st_guard_cases() {
    local rc lat bystander writer child dir=$ST/guard
    ensure_watch || { echo "selftest: the thermal guard (watch_card) does not build"; ST_FAIL=1; return 1; }
    mkdir -p "$ST" "$dir"
    # the dummy job: a shell with a child; on TERM it writes the time and ends (the child gets the group's TERM too)
    cat >"$dir/dummy.sh" <<'DUMMY'
#!/bin/sh
trap 'date +%s%N > "$DEATH"; exit 0' TERM
sleep 41 &
echo $! > "$CHILD"
wait
DUMMY
    chmod +x "$dir/dummy.sh"
    # break 5d: the hotspot crosses 95 C at four different moments: each time the job's whole group ends within 100 ms
    local trial worst=0 d_ok=ok rc_ok=8 pre
    sleep 60 &
    bystander=$!
    for trial in 1 2 3 4; do
        st_reset
        echo 60000 >"$ST/hot"
        rm -f "$dir/death" "$dir/tw" "$dir/child"
        ST_ENV=(DEATH="$dir/death" CHILD="$dir/child")
        pre=0.$((trial * 53 % 20 + 10))3
        ( sleep 3; st_hot 80000; sleep 0.137; st_hot 90000; sleep "$pre"; date +%s%N >"$dir/tw"; st_hot 96000 ) &
        writer=$!
        st_job "$ST/b5d-fast" job-fast 70 60 sh "$dir/dummy.sh"
        rc=$?
        wait "$writer"
        [[ $rc == 8 ]] || rc_ok=$rc
        lat=$(( ($(cat "$dir/death" 2>/dev/null || echo 0) - $(cat "$dir/tw" 2>/dev/null || echo 0)) / 1000000 ))
        echo "  measured (trial $trial): the hotspot file changed to 96 C, the job received TERM $lat ms later (guard period ${GUARD_MS} ms)"
        sum "- measured stop latency, trial $trial (hotspot written to TERM received by the job): $lat ms"
        ((lat >= 0 && lat < 100)) || d_ok=slow
        ((lat > worst)) && worst=$lat
    done
    st_expect "break5d-stop" 8 "$rc_ok" "the hotspot crossed 96 C (limit 95) in four trials: the dummy job was stopped each time, exit 8"
    st_expect "break5d-latency" ok "$d_ok" "TERM reached the job at most $worst ms after the hotspot file changed (limit 100 ms, 4 trials)"
    child=$(cat "$dir/child" 2>/dev/null || echo 0)
    kill -0 "$child" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5d-group" gone "$rc" "the job's child process ended too (the whole process group got the signal)"
    kill -0 "$bystander" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5d-bystander" alive "$rc" "a process in the runner's own group was not touched (only the job's group is stopped)"
    kill "$bystander" 2>/dev/null
    grep -q "^THERMAL STOP: hotspot 96 C at t=.*TERM sent to the job's process group" "$ST/b5d-fast/job-fast.log" && rc=ok || rc=missing
    st_expect "break5d-log" ok "$rc" "the job log says THERMAL STOP with the reading, the time and the group"
    grep -q "^thermal_stop=1" "$ST/b5d-fast/job-fast.meta" && grep -q "^hotspot_max=96" "$ST/b5d-fast/job-fast.meta" && rc=ok || rc=missing
    st_expect "break5d-meta" ok "$rc" "the .meta says thermal_stop=1, hotspot_max=96"
    awk -F, '/^[0-9]/ {n++; if (p != "") {g = $1 - p; if (g > mx) mx = g}; p = $1} END {exit !(n > 100 && mx < 0.06)}' "$ST/b5d-fast/job-fast.guard.csv" && rc=ok || rc=missing
    st_expect "break5d-rate" ok "$rc" "the guard file holds the 20 ms samples (over 100 rows, no gap above 60 ms)"
    # break 5g: the guard alone against a job group with no timeout(1) in front (timeout would pass the signal on by itself): the
    # child of the job must end too, and the leader's own TERM trap must not be the only thing that ends it
    st_reset
    echo 60000 >"$ST/hot"
    rm -f "$dir/death" "$dir/child" "$dir/leader" "$dir/g.log"
    ( set -m; DEATH="$dir/death" CHILD="$dir/child" sh "$dir/dummy.sh" & echo $! >"$dir/leader"; wait ) &
    writer=$!
    sleep 0.5
    "$WATCH" guard --pid "$(cat "$dir/leader")" --hot-file "$ST/hot" --stop-hot 95 --kill-after-ms 30000 --log "$dir/g.log" --csv "$dir/g.csv" 2>/dev/null &
    local gp=$! n
    sleep 0.5
    st_hot 97000
    for n in $(seq 1 30); do [[ -s $dir/death ]] && break; sleep 0.1; done
    sleep 0.3
    child=$(cat "$dir/child" 2>/dev/null || echo 0)
    # KILL is 30 s away here, so only the TERM can have ended the child
    kill -0 "$child" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5g-direct-group" gone "$rc" "the job's child ended 0.3 s after the leader got TERM (the signal went to the process group, not only to the leader)"
    grep -q "TERM sent to the job's process group" "$dir/g.log" && rc=ok || rc=missing
    st_expect "break5g-direct-log" ok "$rc" "the guard log says it signalled the process group"
    kill "$gp" "$child" 2>/dev/null
    wait "$gp" "$writer" 2>/dev/null
    # break 5e: an unreadable sensor is a stop too (the file vanishes, holds garbage, or holds the 999 C sentinel)
    local how
    for how in vanish garbage nan negnan inf partial zero sentinel; do
        st_reset
        echo 60000 >"$ST/hot"
        rm -f "$dir/death" "$dir/child"
        ST_ENV=(DEATH="$dir/death" CHILD="$dir/child")
        case $how in
        vanish) ( sleep 3; rm -f "$ST/hot" ) & ;;
        garbage) ( sleep 3; st_hot "n/a" ) & ;;
        nan) ( sleep 3; st_hot "nan" ) & ;;
        negnan) ( sleep 3; st_hot "-nan" ) & ;;
        inf) ( sleep 3; st_hot "inf" ) & ;;
        partial) ( sleep 3; st_hot "96abc" ) & ;;
        zero) ( sleep 3; st_hot "0" ) & ;;
        sentinel) ( sleep 3; st_hot 999000 ) & ;;
        esac
        writer=$!
        st_job "$ST/b5e-$how" job-$how 70 60 sh "$dir/dummy.sh"
        rc=$?
        wait "$writer"
        st_expect "break5e-$how-stop" 8 "$rc" "sensor $how: the dummy job was stopped, exit 8"
        if [[ $how == sentinel ]]; then
            grep -q "^THERMAL STOP: hotspot 999 C at" "$ST/b5e-$how/job-$how.log" && rc=ok || rc=missing
            st_expect "break5e-$how-log" ok "$rc" "the log names the reading (hotspot 999 C)"
        else
            grep -q "^THERMAL STOP: sensor unreadable" "$ST/b5e-$how/job-$how.log" && rc=ok || rc=missing
            st_expect "break5e-$how-log" ok "$rc" "the log says: sensor unreadable"
        fi
        [[ -s $dir/death ]] && rc=ok || rc=missing
        st_expect "break5e-$how-ended" ok "$rc" "the job received TERM"
    done
    # break 5h: the memory (VRAM) temperature has its own stop limit (100 C); an unreadable memory sensor is a warning only
    st_reset
    echo 60000 >"$ST/hot"
    rm -f "$dir/death" "$dir/child" "$dir/tw"
    ST_ENV=(DEATH="$dir/death" CHILD="$dir/child")
    ( sleep 3; st_mem 99000; sleep 0.5; date +%s%N >"$dir/tw"; st_mem 101000 ) &
    writer=$!
    st_job "$ST/b5h-mem" job-mem 70 60 sh "$dir/dummy.sh"
    rc=$?
    wait "$writer"
    st_expect "break5h-mem-stop" 8 "$rc" "memory 99 C then 101 C (limit 100), hotspot 60 C: the dummy job was stopped, exit 8"
    lat=$(( ($(cat "$dir/death" 2>/dev/null || echo 0) - $(cat "$dir/tw" 2>/dev/null || echo 0)) / 1000000 ))
    echo "  measured: the memory temperature file changed to 101 C, the job received TERM $lat ms later"
    sum "- measured stop latency at the memory limit (memory file written to TERM received by the job): $lat ms"
    st_expect "break5h-mem-latency" ok "$( ((lat >= 0 && lat < 100)) && echo ok || echo slow)" "TERM reached the job within 100 ms of the memory temperature change ($lat ms)"
    grep -q "^THERMAL STOP: memory temperature 101 C" "$ST/b5h-mem/job-mem.log" && rc=ok || rc=missing
    st_expect "break5h-mem-log" ok "$rc" "the log names the memory temperature reading (101 C)"
    awk -F, '/^[0-9]/ && $3 > 98 {f = 1} END {exit !f}' "$ST/b5h-mem/job-mem.guard.csv" && rc=ok || rc=missing
    st_expect "break5h-mem-csv" ok "$rc" "the guard file holds the memory readings (third column)"
    st_reset
    echo 60000 >"$ST/hot"
    ( sleep 1; st_mem 99000; sleep 1; rm -f "$ST/mem"; sleep 1; st_mem "n/a" ) &
    writer=$!
    st_job "$ST/b5h-memok" job-memok 70 60 sleep 5
    rc=$?
    wait "$writer"
    st_expect "break5h-mem-control" 0 "$rc" "memory 99 C (below its limit), then the memory sensor missing and garbled, hotspot fine: the job ran to its end, exit 0"
    grep -q "^WARNING: memory temperature unreadable" "$ST/b5h-memok/job-memok.log" && ! grep -q "THERMAL STOP" "$ST/b5h-memok/job-memok.log" && rc=ok || rc=missing
    st_expect "break5h-mem-warning" ok "$rc" "an unreadable memory sensor is logged as a WARNING and is not a stop"
    # break 5i: a job's children that leave the process group with setsid are stopped too (the guard signals the job's whole systemd scope)
    cat >"$dir/dummy_s.sh" <<'DUMMY'
#!/bin/sh
trap 'date +%s%N > "$SDEATH"; exit 0' TERM
echo $$ > "$SPID"
sleep 41 &
wait
DUMMY
    cat >"$dir/dummy_k.sh" <<'DUMMY'
#!/bin/sh
trap '' TERM
echo $$ > "$KPID"
exec sleep 41
DUMMY
    cat >"$dir/dummy_setsid.sh" <<'DUMMY'
#!/bin/sh
trap 'date +%s%N > "$DEATH"; exit 0' TERM
sleep 41 &
echo $! > "$CHILD"
setsid sh "$DIR/dummy_s.sh" &
setsid sh "$DIR/dummy_k.sh" &
wait
DUMMY
    chmod +x "$dir"/dummy_*.sh
    st_reset
    echo 60000 >"$ST/hot"
    rm -f "$dir/death" "$dir/child" "$dir/sdeath" "$dir/spid" "$dir/kpid" "$dir/tw"
    ST_ENV=(DEATH="$dir/death" CHILD="$dir/child" SDEATH="$dir/sdeath" SPID="$dir/spid" KPID="$dir/kpid" DIR="$dir")
    ( sleep 3; date +%s%N >"$dir/tw"; st_hot 96000 ) &
    writer=$!
    st_job "$ST/b5i-setsid" job-setsid 70 60 sh "$dir/dummy_setsid.sh"
    rc=$?
    wait "$writer"
    st_expect "break5i-stop" 8 "$rc" "a job with children in other sessions (setsid) was stopped at 96 C, exit 8"
    local sp kp
    sp=$(cat "$dir/spid" 2>/dev/null || echo 0)
    kp=$(cat "$dir/kpid" 2>/dev/null || echo 0)
    kill -0 "$sp" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5i-setsid-term" gone "$rc" "the setsid child that handles TERM ended (the scope got TERM)"
    kill -0 "$kp" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5i-setsid-kill" gone "$rc" "the setsid child that ignores TERM ended too (the scope got KILL)"
    grep -q "KILL sent to the job's process group .* and to its scope llmjob-" "$ST/b5i-setsid/job-setsid.log" && rc=ok || rc=missing
    st_expect "break5i-log" ok "$rc" "the log says TERM and KILL went to the scope as well"
    lat=$(( ($(cat "$dir/sdeath" 2>/dev/null || echo 0) - $(cat "$dir/tw" 2>/dev/null || echo 0)) / 1000000 ))
    echo "  measured: the hotspot file changed to 96 C, the setsid child received TERM $lat ms later"
    sum "- measured stop latency of a setsid child (hotspot written to TERM received): $lat ms"
    st_expect "break5i-latency" ok "$( ((lat > 0 && lat < 150)) && echo ok || echo slow)" "the setsid child got TERM within 150 ms of the hotspot change ($lat ms)"
    # break 5j: the guard itself is killed in the middle of a job: the runner stops the job at once and marks GUARD LOST
    st_reset
    echo 60000 >"$ST/hot"
    rm -f "$dir/death" "$dir/child"
    ST_ENV=(DEATH="$dir/death" CHILD="$dir/child")
    st_job "$ST/b5j-lost" job-lost 70 60 sh "$dir/dummy.sh" &
    local jb=$! gpid="" tk
    for n in $(seq 1 50); do
        gpid=$(st_find_guard "$jb") && [[ -n $gpid ]] && break
        sleep 0.1
    done
    sleep 2
    tk=$(date +%s%N)
    [[ -n $gpid ]] && kill -KILL "$gpid" 2>/dev/null
    wait "$jb"
    rc=$?
    lat=$(( ($(cat "$dir/death" 2>/dev/null || echo 0) - tk) / 1000000 ))
    echo "  measured: the guard was killed, the job received TERM $lat ms later"
    sum "- measured stop latency after the guard was killed mid-job (kill to TERM received by the job): $lat ms"
    st_expect "break5j-exit" 10 "$rc" "the guard killed 2 s into a job: the runner exit status is 10"
    st_expect "break5j-latency" ok "$( ((lat >= 0 && lat < 100)) && echo ok || echo slow)" "the job received TERM within 100 ms of the guard's death ($lat ms)"
    grep -q "^GUARD LOST" "$ST/b5j-lost/job-lost.log" && rc=ok || rc=missing
    st_expect "break5j-log" ok "$rc" "the log says GUARD LOST"
    child=$(cat "$dir/child" 2>/dev/null || echo 0)
    kill -0 "$child" 2>/dev/null && rc=alive || rc=gone
    st_expect "break5j-group" gone "$rc" "the job's child ended too"
    awk -F= '/^wall_s=/ {exit !($2 < 8)}' "$ST/b5j-lost/job-lost.meta" && grep -q "guard_lost=1" "$ST/b5j-lost/job-lost.meta" && rc=ok || rc=missing
    st_expect "break5j-meta" ok "$rc" "the .meta says guard_lost=1 and the job ended in under 8 s, not after its 41 s"
    # control: a stream that stays below the limit (94 C) does not stop a job that ends by itself
    st_reset
    echo 60000 >"$ST/hot"
    ( sleep 2; for v in 80 90 94 90 80; do st_hot ${v}000; sleep 0.4; done ) &
    writer=$!
    st_job "$ST/b5f-control" job-below 70 60 sleep 5
    rc=$?
    wait "$writer"
    st_expect "guard-control" 0 "$rc" "hotspot up to 94 C (limit 95): the job ran to its end, exit 0"
    grep -q "THERMAL STOP" "$ST/b5f-control/job-below.log" && rc=found || rc=none
    st_expect "guard-control-log" none "$rc" "no THERMAL STOP line in the log"
    grep -q "^thermal_stop=0" "$ST/b5f-control/job-below.meta" && grep -q "^hotspot_max=94" "$ST/b5f-control/job-below.meta" && rc=ok || rc=missing
    st_expect "guard-control-meta" ok "$rc" "the .meta says thermal_stop=0, hotspot_max=94 (from the 20 ms guard)"
    grep -q "^guard: .* samples over" "$ST/b5f-control/job-below.log" && rc=ok || rc=missing
    st_expect "guard-control-line" ok "$rc" "the log ends with the guard's own summary line"
}
phase_selftest() {
    sum ""
    sum "## selftest (no GPU; every deliberate break must FAIL as intended)"
    ST=$RUN_DIR/selftest
    mkdir -p "$ST"
    ensure_watch || { echo "selftest: the thermal guard (watch_card) does not build"; FAILED=1; return 1; }
    echo "thermal guard build tested: $WATCH ($(stat -c '%y' "$WATCH" | cut -c1-19), sha256 $(sha256sum "$WATCH" | cut -c1-16))"
    local out rc rc2 t0 t1 holder line
    echo "gate selftest:"

    st_reset
    st_job "$ST/c0-control" job-ok 70 20 true
    rc=$?
    st_expect "control-ok" 0 "$rc" "no fault: the job runs, exit 0"
    grep -q "^exit=0" "$ST/c0-control/job-ok.meta" || { echo "  WRONG control-ok: no meta"; ST_FAIL=1; }
    grep -q "memory clock 1408" "$ST/c0-control/summary.md" || { echo "  WRONG control-ok: summary lacks the memory clock"; ST_FAIL=1; }

    # break 1: a fake /proc with a card client named GameThread
    st_reset
    mkdir -p "$ST/proc/5001/fdinfo"
    echo GameThread >"$ST/proc/5001/comm"
    printf 'Z:\\game.exe\0' >"$ST/proc/5001/cmdline"
    printf 'drm-driver:\tamdgpu\ndrm-pdev:\t0000:99:00.0\ndrm-client-id:\t5001\n' >"$ST/proc/5001/fdinfo/7"
    st_job "$ST/b1-game" job-game 70 20 true
    st_expect "break1-game-client" 7 "$?" "a card client named GameThread: job skipped, exit 7"
    # the same, found only by a Steam path in a command line
    st_reset
    mkdir -p "$ST/proc/5002"
    echo steam.exe >"$ST/proc/5002/comm"
    printf '/x/SteamLibrary/steamapps/common/Game/launcher.exe\0' >"$ST/proc/5002/cmdline"
    st_job "$ST/b1b-game" job-game2 70 20 true
    st_expect "break1b-game-path" 7 "$?" "a process with a steamapps/common path: job skipped, exit 7"

    # break 2: meminfo at 4 GB
    st_reset
    printf 'MemAvailable:   4194304 kB\n' >"$ST/meminfo"
    st_job "$ST/b2-mem" job-mem 70 20 true
    st_expect "break2-memory" 5 "$?" "4 GB available: NOT RUN, exit 5"

    # break 3: the lock is held by another process longer than the wait
    st_reset
    flock "$ST/lock" sleep 8 &
    holder=$!
    sleep 0.5
    st_job "$ST/b3-lock" job-lock 70 20 true
    st_expect "break3-lock" 6 "$?" "lock held 8 s, wait 3 s: NOT RUN, exit 6"
    wait "$holder"

    # break 4: the LACT profile reads Default: this job and the next one are not run
    st_reset
    printf 'current_profile: Testing\n' >"$ST/lact.yaml"
    # the profile is changed after the run's start record, through a job that edits the fake file
    st_job "$ST/b4-lact" job-lact 70 20 sh -c "printf 'current_profile: Default\n' > '$ST/lact.yaml'"
    rc=$?
    st_expect "break4-lact-first" 0 "$rc" "the job that changed the fake profile itself ran"
    grep -q "RC2=3 STOPPED=1" "$ST/b4-lact/out.txt" && rc=ok || rc=missing
    st_expect "break4-lact-stop" ok "$rc" "profile Default: the next job is NOT RUN (exit 3), all GPU work stopped"

    # break 4b (round 4): the expected profile is a setting. "Default" is LACT's top-level setting, current_profile null in the file.
    local goodcfg='lact_profile=Default power1_cap=374000000 perf_level=auto OD_SCLK_OFFSET: 150Mhz OD_MCLK: 0: 97Mhz 1: 1409MHz OD_VDDGFX_OFFSET: -50mV'
    st_reset
    printf 'current_profile: null\n' >"$ST/lact.yaml"
    printf '%s\n' "$goodcfg" >"$ST/cardcfg"
    ST_ENV=(LLM_EXPECT_PROFILE=Default LLM_CARDCFG_FILE="$ST/cardcfg")
    st_job "$ST/b4b-ctl" job-def 70 20 true
    st_expect "default-control" 0 "$?" "expected Default, profile null (LACT's Default), 374 W and the overclock offsets: the job runs, exit 0"
    st_reset
    printf 'current_profile: Testing\n' >"$ST/lact.yaml"
    printf '%s\n' "$goodcfg" >"$ST/cardcfg"
    ST_ENV=(LLM_EXPECT_PROFILE=Default LLM_CARDCFG_FILE="$ST/cardcfg")
    st_job "$ST/b4b-testing" job-def2 70 20 true
    st_expect "break4b-testing-when-default-expected" 3 "$?" "expected Default, profile Testing: NOT RUN, exit 3"
    st_reset
    printf 'current_profile: null\n' >"$ST/lact.yaml"
    st_job "$ST/b4b-oldmode" job-def3 70 20 true
    st_expect "break4b-default-when-testing-expected" 3 "$?" "expected Testing (old modes), profile Default: NOT RUN, exit 3"
    st_reset
    printf 'current_profile: null\n' >"$ST/lact.yaml"
    printf '%s\n' "${goodcfg/374000000/317000000}" >"$ST/cardcfg"
    ST_ENV=(LLM_EXPECT_PROFILE=Default LLM_CARDCFG_FILE="$ST/cardcfg")
    st_job "$ST/b4b-cap" job-def4 70 20 true
    st_expect "break4b-power-cap" 3 "$?" "profile Default but the power cap reads 317 W: NOT RUN, exit 3"
    st_reset
    printf 'current_profile: null\n' >"$ST/lact.yaml"
    printf '%s\n' "${goodcfg/-50mV/0mV}" >"$ST/cardcfg"
    ST_ENV=(LLM_EXPECT_PROFILE=Default LLM_CARDCFG_FILE="$ST/cardcfg")
    st_job "$ST/b4b-volt" job-def5 70 20 true
    st_expect "break4b-voltage" 3 "$?" "profile Default but the voltage offset is not -50 mV: NOT RUN, exit 3"
    st_reset
    printf 'current_profile: null\n' >"$ST/lact.yaml"
    printf '%s\n' "$goodcfg" >"$ST/cardcfg"
    ST_ENV=(LLM_EXPECT_PROFILE=Default LLM_CARDCFG_FILE="$ST/cardcfg")
    st_job "$ST/b4b-switch" job-def6 70 20 sh -c "printf 'current_profile: Testing\n' > '$ST/lact.yaml'"
    grep -q "RC2=3 STOPPED=1" "$ST/b4b-switch/out.txt" && rc=ok || rc=missing
    st_expect "break4b-switch-during-run" ok "$rc" "profile switched to Testing during a Default run: the next job is NOT RUN, all GPU work stopped"

    # break 5a: the poll stops a dummy process at a fake low limit
    st_reset
    echo 60000 >"$ST/hot"
    ST_ENV=(LLM_GUARD_C=50)
    t0=$(date +%s)
    st_job "$ST/b5a-guard" job-guard 70 60 sleep 40
    rc=$?
    t1=$(date +%s)
    st_expect "break5a-guard" 8 "$rc" "hotspot 60 C over a fake limit of 50 C: the dummy job was stopped, exit 8"
    ((t1 - t0 < 15)) && rc=fast || rc=slow
    st_expect "break5a-guard-time" fast "$rc" "stopped within $((t1 - t0)) s, not after its 40 s"
    st_guard_cases
    # break 5b: the hotspot reads 96 C after a job: further GPU work is stopped
    st_reset
    st_job "$ST/b5b-hot" job-hot 70 20 sh -c "echo 96000 > '$ST/hot'"
    rc=$?
    grep -q "RC2=3 STOPPED=1" "$ST/b5b-hot/out.txt" && rc=ok || rc=missing
    st_expect "break5b-after-96C" ok "$rc" "96 C after a job: the next job is NOT RUN, all GPU work stopped"
    # a hot card at the start: not started
    st_reset
    echo 75000 >"$ST/hot"
    st_job "$ST/b5c-start" job-start 70 20 true
    st_expect "break5c-hot-start" 4 "$?" "hotspot 75 C against a start limit of 70 C: NOT RUN, exit 4"

    # break 10: another job holds the lock; while it does, the machine changes. Each gate must be read again after the
    # lock is acquired (the pre-lock checks passed: all four started from a clean state).
    st_lockhold() { # change command run by the lock holder after 2 s; then wait until the holder owns the lock
        local script=$ST/holder.sh
        printf '#!/bin/sh\nsleep 2\n%s\nsleep 0.3\n' "$1" >"$script"
        chmod +x "$script"
        flock "$ST/lock" "$script" &
        holder=$!
        sleep 0.5
    }
    st_reset
    st_lockhold ":"
    st_job "$ST/b10-control" job-lockctl 70 20 true
    st_expect "lockgate-control" 0 "$?" "lock held 2 s by another job, nothing changed: the job runs, exit 0"
    wait "$holder"
    st_reset
    st_lockhold "echo 80000 > '$ST/hot'"
    st_job "$ST/b10a-hot" job-lockhot 70 20 true
    st_expect "lockgatea-hot-after-lock" 4 "$?" "hotspot 80 C appeared while waiting for the lock (start limit 70 C): NOT RUN, exit 4"
    grep -q "NOT RUN after the lock wait" "$ST/b10a-hot/summary.md" && rc=ok || rc=missing
    st_expect "lockgatea-seen-after-lock" ok "$rc" "the summary says it was seen after the lock wait, not before"
    wait "$holder"
    st_reset
    st_lockhold "mkdir -p '$ST/proc/5003/fdinfo'; echo GameThread > '$ST/proc/5003/comm'; printf 'Z:\\\\game.exe\\000' > '$ST/proc/5003/cmdline'; printf 'drm-driver:\\tamdgpu\\ndrm-pdev:\\t0000:99:00.0\\ndrm-client-id:\\t5003\\n' > '$ST/proc/5003/fdinfo/7'"
    st_job "$ST/b10b-game" job-lockgame 70 20 true
    st_expect "lockgateb-game-after-lock" 7 "$?" "a game started while waiting for the lock: NOT RUN, exit 7"
    grep -q "NOT RUN after the lock wait" "$ST/b10b-game/summary.md" && rc=ok || rc=missing
    st_expect "lockgateb-seen-after-lock" ok "$rc" "the summary says it was seen after the lock wait, not before"
    wait "$holder"
    st_reset
    st_lockhold "printf 'current_profile: Default\\n' > '$ST/lact.yaml'"
    st_job "$ST/b10c-lact" job-locklact 70 20 true
    st_expect "lockgatec-lact-after-lock" 3 "$?" "the LACT profile changed to Default while waiting for the lock: NOT RUN, exit 3"
    grep -q "NOT RUN after the lock wait" "$ST/b10c-lact/summary.md" && rc=ok || rc=missing
    st_expect "lockgatec-seen-after-lock" ok "$rc" "the summary says it was seen after the lock wait, not before"
    grep -q "RC1=3 STOPPED=1" "$ST/b10c-lact/out.txt" && rc=ok || rc=missing
    st_expect "lockgatec-lact-stop" ok "$rc" "and all GPU work is stopped"
    wait "$holder"
    st_reset
    st_lockhold "printf 'MemAvailable:   4194304 kB\\n' > '$ST/meminfo'"
    st_job "$ST/b10d-mem" job-lockmem 70 20 true
    st_expect "lockgated-memory-after-lock" 5 "$?" "memory fell to 4 GB while waiting for the lock: NOT RUN, exit 5"
    grep -q "NOT RUN after the lock wait" "$ST/b10d-mem/summary.md" && rc=ok || rc=missing
    st_expect "lockgated-seen-after-lock" ok "$rc" "the summary says it was seen after the lock wait, not before"
    wait "$holder"

    # break 11: the memory cap. A dummy job allocates and touches 800 MB under a fake cap of 300M: the kernel must kill it
    # (exit 137, "MEMORY CAP" in its log). Controls: 100 MB under the same cap passes; the same 800 MB under a 2G cap passes, so
    # the kill above comes from the cap and not from the dummy.
    local alloc='import sys; b = bytes([1]) * (int(sys.argv[1]) * 2**20); print("touched", len(b))'
    st_reset
    ST_ENV=(LLM_MEM_CAP=300M)
    st_job "$ST/b11-cap" job-cap 70 60 python3 -I -c "$alloc" 800
    rc=$?
    st_expect "break11-memory-cap" 137 "$rc" "800 MB allocated under a 300M cap: the dummy job was killed, exit 137"
    grep -q "^MEMORY CAP" "$ST/b11-cap/job-cap.log" && grep -q "KILLED by the memory cap" "$ST/b11-cap/summary.md" && rc=ok || rc=missing
    st_expect "break11-cap-recorded" ok "$rc" "the log and the summary say the memory cap killed it"
    st_reset
    ST_ENV=(LLM_MEM_CAP=300M)
    st_job "$ST/b11c-small" job-capsmall 70 60 python3 -I -c "$alloc" 100
    st_expect "capcontrol-small" 0 "$?" "100 MB under the 300M cap: the job runs, exit 0"
    st_reset
    ST_ENV=(LLM_MEM_CAP=2G)
    st_job "$ST/b11d-big" job-capbig 70 60 python3 -I -c "$alloc" 800
    st_expect "capcontrol-big-under-2G" 0 "$?" "the same 800 MB under a 2G cap: the job runs, exit 0"
    grep -q "^mem_cap=2G swap_cap=0" "$ST/b11d-big/job-capbig.meta" && rc=ok || rc=missing
    st_expect "capcontrol-meta" ok "$rc" "the .meta records the cap and swap_cap=0"

    # the one 16G exception belongs to the name r8x-ref-* only: another name keeps the normal cap
    st_reset
    ST_ENV=(LLM_MEM_CAP=2G LLM_BIG_CAP=3G)
    st_job "$ST/b11e-ref" r8x-ref-captest 70 60 true
    grep -q "^mem_cap=3G swap_cap=0" "$ST/b11e-ref/r8x-ref-captest.meta" && rc=ok || rc=missing
    st_expect "bigcap-ref-job" ok "$rc" "a job named r8x-ref-* runs under the big cap (3G in this test), recorded in its .meta"
    st_reset
    ST_ENV=(LLM_MEM_CAP=2G LLM_BIG_CAP=3G)
    st_job "$ST/b11f-other" r8x-kld-captest 70 60 true
    grep -q "^mem_cap=2G swap_cap=0" "$ST/b11f-other/r8x-kld-captest.meta" && rc=ok || rc=missing
    st_expect "bigcap-other-job" ok "$rc" "any other job name keeps the normal cap (2G in this test)"
    # break 9: the memory clock changes inside a job: the row is labelled, never merged
    st_reset
    st_job "$ST/b9-uclk" job-uclk 70 20 sh -c "echo 1258 > '$ST/uclk'"
    grep -q "MEMORY CLOCK uclk=1408/1258" "$ST/b9-uclk/summary.md" && rc=ok || rc=missing
    st_expect "break9-uclk" ok "$rc" "memory clock 1408 then 1258 in one job: labelled uclk=1408/1258"

    # determinism evidence: the production det_finish on canned logits files
    local dd=$ST/det
    rm -rf "$dd"
    mkdir -p "$dd"
    head -c 100000 /dev/urandom >"$dd/a.bin"
    cp "$dd/a.bin" "$dd/b.bin"
    line=$(det_finish selftest "$dd/a.bin" "$dd/b.bin" "$dd/ev-same.log")
    rc=$?
    st_expect "det-control-identical" 0 "$rc" "two identical files: cmp exit 0"
    grep -q "byte-identical" <<<"$line" && rc=ok || rc=missing
    st_expect "det-control-line" ok "$rc" "the summary line says byte-identical"
    [[ $(sed -n 's/^sha256_a: //p' "$dd/ev-same.log") == "$(sed -n 's/^sha256_b: //p' "$dd/ev-same.log")" && -n $(sed -n 's/^sha256_a: //p' "$dd/ev-same.log") ]] && grep -q "^cmp_exit: 0" "$dd/ev-same.log" && grep -q "^size_a: 100000" "$dd/ev-same.log" && rc=ok || rc=missing
    st_expect "det-control-evidence" ok "$rc" "the evidence log has cmp_exit 0, the size and equal sha256 of both files"
    [[ ! -e $dd/a.bin && ! -e $dd/b.bin ]] && rc=ok || rc=missing
    st_expect "det-control-deleted" ok "$rc" "the logits files are deleted after the evidence is written"
    head -c 100000 /dev/urandom >"$dd/a.bin"
    cp "$dd/a.bin" "$dd/b.bin"
    printf 'X' | dd of="$dd/b.bin" bs=1 seek=5000 conv=notrunc 2>/dev/null
    line=$(det_finish selftest "$dd/a.bin" "$dd/b.bin" "$dd/ev-diff.log")
    rc=$?
    st_expect "det-break-differ" 1 "$rc" "one byte changed in the second file: cmp exit 1, reported as DIFFER"
    grep -q "DIFFER" <<<"$line" && grep -q "^cmp_exit: 1" "$dd/ev-diff.log" && [[ $(sed -n 's/^sha256_a: //p' "$dd/ev-diff.log") != "$(sed -n 's/^sha256_b: //p' "$dd/ev-diff.log")" ]] && rc=ok || rc=missing
    st_expect "det-break-evidence" ok "$rc" "the evidence log has cmp_exit 1, the cmp output and two different sha256"
    head -c 1000 /dev/urandom >"$dd/a.bin"
    det_finish selftest "$dd/a.bin" "$dd/missing.bin" "$dd/ev-missing.log" >/dev/null
    st_expect "det-break-missing" 2 "$?" "a logits file missing: FAILED, not 'identical'"
    cp "$dd/a.bin" "$dd/b.bin"
    det_finish selftest "$dd/a.bin" "$dd/b.bin" "$dd/no-such-folder/ev.log" >/dev/null
    rc=$?
    [[ -e $dd/a.bin && -e $dd/b.bin ]] && rc2=kept || rc2=deleted
    st_expect "det-break-no-evidence" "2 kept" "$rc $rc2" "the evidence cannot be written: FAILED and the logits files are kept"
    # the same-run-again gate of the quality phase (llm_tables.py check-self): a log at the noise floor passes, a shifted one fails
    printf 'Mean    KLD:   0.000000 +/-   0.000000\nMaximum KLD:   0.000063\nSame top p: 99.992 +/- 0.001 %%\n' >"$dd/kld-ok.log"
    printf 'Mean    KLD:   0.000123 +/-   0.000010\nMaximum KLD:   0.000063\nSame top p: 99.992 +/- 0.001 %%\n' >"$dd/kld-shift.log"
    "${PY[@]}" "$ROOT/scripts/llm_tables.py" check-self "$dd/kld-ok.log" >/dev/null 2>&1
    st_expect "selfkld-control" 0 "$?" "the same run at the noise floor passes the quality gate"
    "${PY[@]}" "$ROOT/scripts/llm_tables.py" check-self "$dd/kld-shift.log" >/dev/null 2>&1
    st_expect "selfkld-break" 1 "$?" "a shifted run (mean KLD 0.000123) fails the quality gate"
    echo "python helpers:"
    local pyrc=0
    for f in llm_kernels.py llm_tables.py; do
        if [[ -f $ROOT/scripts/$f ]]; then
            "${PY[@]}" "$ROOT/scripts/$f" selftest "$ST/$f.d" >"$ST/$f.out" 2>&1 || pyrc=1
            sed 's/^/  /' "$ST/$f.out"
            sum "- selftest $f: $(tail -1 "$ST/$f.out")"
        else
            echo "  missing scripts/$f"
            pyrc=1
        fi
    done
    # the reference writer and checker (round 2) need the convert venv's gguf, numpy and torch
    if [[ -x $ROOT/toolchains/convert-venv/bin/python ]]; then
        "$ROOT/toolchains/convert-venv/bin/python" -W ignore -I "$ROOT/scripts/llm_ref.py" selftest >"$ST/llm_ref.py.out" 2>&1 || pyrc=1
        sed 's/^/  /' "$ST/llm_ref.py.out"
        sum "- selftest llm_ref.py: $(tail -1 "$ST/llm_ref.py.out")"
    else
        echo "  no convert venv: llm_ref.py selftest not run"
        sum "- selftest llm_ref.py: NOT RUN (no convert venv)"
        pyrc=1
    fi
    # round 3 helpers (no GPU): each must end in its own "VERDICT ... PASS" line
    for f in llm3_pairs.py llm3_cnt.py llm3_power.py llm3_q3.py; do
        LLM_RUN=$RUN_DIR "${PY[@]}" "$ROOT/scripts/$f" selftest >"$ST/$f.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/$f.out" || pyrc=1
        sed 's/^/  /' "$ST/$f.out" | grep -v '^  RESULT'
        sum "- selftest $f: $(tail -1 "$ST/$f.out")"
    done
    if [[ -x $ROOT/toolchains/convert-venv/bin/python ]]; then
        "$ROOT/toolchains/convert-venv/bin/python" -W ignore -I "$ROOT/scripts/llm3_mixed_ref.py" selftest >"$ST/llm3_mixed_ref.py.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/llm3_mixed_ref.py.out" || pyrc=1
        sed 's/^/  /' "$ST/llm3_mixed_ref.py.out"
        sum "- selftest llm3_mixed_ref.py: $(tail -1 "$ST/llm3_mixed_ref.py.out")"
    else
        sum "- selftest llm3_mixed_ref.py: NOT RUN (no convert venv)"
        pyrc=1
    fi
    # round 4 helpers (no GPU): the trace analysis, the origin comparison, the PyTorch harness's own checks
    "${PY[@]}" "$ROOT/scripts/llm4_trace.py" selftest >"$ST/llm4_trace.py.out" 2>&1 || pyrc=1
    grep -q '^VERDICT .* selftest PASS' "$ST/llm4_trace.py.out" || pyrc=1
    sum "- selftest llm4_trace.py: $(tail -1 "$ST/llm4_trace.py.out")"
    for f in llm4_isa.py llm4_tables.py; do
        "${PY[@]}" "$ROOT/scripts/$f" selftest >"$ST/$f.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/$f.out" || pyrc=1
        sum "- selftest $f: $(tail -1 "$ST/$f.out")"
    done
    if [[ -x $ROOT/toolchains/convert-venv/bin/python ]]; then
        "$ROOT/toolchains/convert-venv/bin/python" -W ignore -I "$ROOT/scripts/llm4_origin.py" selftest >"$ST/llm4_origin.py.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/llm4_origin.py.out" || pyrc=1
        sum "- selftest llm4_origin.py: $(tail -1 "$ST/llm4_origin.py.out")"
    fi
    if [[ -x $ROOT/toolchains/torch-rocm10-venv/bin/python ]]; then
        "$ROOT/toolchains/torch-rocm10-venv/bin/python" -I "$ROOT/scripts/llm4_torch.py" selftest >"$ST/llm4_torch.py.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/llm4_torch.py.out" || pyrc=1
        sum "- selftest llm4_torch.py: $(tail -1 "$ST/llm4_torch.py.out")"
        "$ROOT/toolchains/vllm-rocm101-venv/bin/python" -I "$ROOT/scripts/llm4_vllm.py" selftest >"$ST/llm4_vllm.py.out" 2>&1 || pyrc=1
        grep -q '^VERDICT .* selftest PASS' "$ST/llm4_vllm.py.out" || pyrc=1
        sum "- selftest llm4_vllm.py: $(tail -1 "$ST/llm4_vllm.py.out")"
    else
        sum "- selftest llm4_torch.py: NOT RUN (no torch venv)"
        pyrc=1
    fi
    ((pyrc)) && ST_FAIL=1
    if ((ST_FAIL)); then
        sum "- selftest RESULT: FAILED"
        FAILED=1
    else
        sum "- selftest RESULT: every deliberate break failed as intended, every control passed"
    fi
}

# ---------------------------------------------------------------------------------------------------------------
# probe (step 2): the profiler against a known answer, counters and graph tracing
phase_probe() {
    sum ""
    sum "## probe (step 2: rocprofv3 known answer, counters, HIP graphs)"
    mkdir -p "$RUN_DIR/rocp-avail"
    gpu_job probe-avail 70 120 rocprofv3-avail list
    gpu_job probe-bw-plain 70 120 "$BANDWIDTH" 1024 50
    gpu_job probe-bw-trace 70 240 rocprofv3 --kernel-trace --stats -f csv -d "$RUN_DIR/rocp-bw" -o bw -- "$BANDWIDTH" 1024 50
    gpu_job probe-pmc 70 120 rocprofv3 --pmc GRBM_COUNT -f csv -d "$RUN_DIR/rocp-pmc" -o pmc -- "$BANDWIDTH" 256 5
    local g=$MODELS/llama32-3b-q8_0.gguf
    gpu_job probe-graph-kernel 70 300 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/rocp-graph-k" -o gk -- \
        "$UNMOD/llama-bench" -m "$g" -ngl 99 -p 0 -n 16 -r 1 -o jsonl
    gpu_job probe-graph-rt 70 300 rocprofv3 --kernel-trace --hip-graph-trace --runtime-trace -f csv json -d "$RUN_DIR/rocp-graph-rt" -o gr -- \
        "$UNMOD/llama-bench" -m "$g" -ngl 99 -p 0 -n 16 -r 1 -o jsonl
}

# ---------------------------------------------------------------------------------------------------------------
# configurations: label | build bin | model file name (without .gguf)
CONFIGS_3B=("3b-fp8-unmod|$UNMOD|llama32-3b-fp8-outq8" "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8" "3b-q8_0|$UNMOD|llama32-3b-q8_0")
CONFIGS_8B=("8b-fp8-unmod|$UNMOD|llama31-8b-fp8-outq8" "8b-fp8-patched|$PATCHED|llama31-8b-fp8-outq8" "8b-q8_0|$UNMOD|llama31-8b-q8_0")

# analyse_trace <name> <kind pp|tg> <model 3b|8b> <weight bytes> <control log> <csv> <extra args...>: the Python helper's
# report; verdict FAILs fail the run, except the profiler-overhead one, which is a warning (shares only).
analyse_trace() {
    local name=$1 kind=$2 model=$3 wb=$4 ctl=$5 csv=$6 hard
    shift 6
    "${PY[@]}" "$ROOT/scripts/llm_kernels.py" report "$csv" --kind "$kind" --model "$model" --weight-bytes "$wb" \
        --control-log "$ctl" --json "$RUN_DIR/$name.analysis.json" "$@" >"$RUN_DIR/$name.report.txt" 2>&1
    hard=$(grep -c "^VERDICT FAIL" "$RUN_DIR/$name.report.txt")
    if grep -q "^VERDICT FAIL profiler-overhead" "$RUN_DIR/$name.report.txt"; then
        sum "  - $name: WARNING profiled run more than 15 % slower than the unprofiled control: shares only, no absolute times"
        hard=$((hard - 1))
    fi
    if ((hard > 0)); then
        sum "  - $name: analysis FAILED ($(grep '^VERDICT FAIL' "$RUN_DIR/$name.report.txt" | head -2 | tr '\n' ';'))"
        FAILED=1
    else
        sum "  - $name: analysed (report in $name.report.txt)"
    fi
}

# The grouping known answer, for `profile` and `reanalyse` alike. The plan guessed "GEMM group above 90 %"; the first
# trace gave 88.4 %, the other 11.6 % being the test's own B-matrix convert to FP16 (group act_convert). The check is
# now GEMM plus that convert above 95 %: the f16 route is exactly those two kernels, so what is left is unknown kernels.
grouping_known_answer() {
    local rep=$RUN_DIR/trace-bops-f16.report.txt
    if "${PY[@]}" "$ROOT/scripts/llm_kernels.py" groups "$RUN_DIR/trace-bops-f16.d/t_kernel_trace.csv" --expect gemm_fp16,act_convert --min-share 0.95 >"$rep" 2>&1; then
        sum "  - grouping known answer (f16 MUL_MAT: GEMM plus activation convert above 95 %): PASS ($(grep VERDICT "$rep"); GEMM alone $(grep -E '^  gemm_fp16' "$rep" | awk '{print $4}') %)"
    else
        sum "  - grouping known answer (f16 MUL_MAT: GEMM plus activation convert above 95 %): FAILED ($(tail -2 "$rep" | tr '\n' ';'))"
        FAILED=1
    fi
}

# profile (step 5): per-kernel traces of one prompt pass (pp512) and of a generation run (tg128), each next to an
# unprofiled control of the same command (the profiler's own cost). Analysis: scripts/llm_kernels.py.
phase_profile() {
    local c label bin mdl m size wb
    sum ""
    sum "## profile (step 5: rocprofv3 kernel traces, with unprofiled controls)"
    for c in "${PROFILE_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        m=$MODELS/$mdl.gguf
        size=${label%%-*}
        wb=1.0
        [[ $label == *q8_0* ]] && wb=1.0625
        gpu_job "ctl-pp512-$label" 70 600 "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 5 -o jsonl
        gpu_job "trace-pp512-$label" 70 600 rocprofv3 --kernel-trace --stats -f csv -d "$RUN_DIR/trace-pp512-$label.d" -o t -- \
            "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 3 -o jsonl
        analyse_trace "trace-pp512-$label" pp "$size" "$wb" "$RUN_DIR/ctl-pp512-$label.log" "$RUN_DIR/trace-pp512-$label.d/t_kernel_trace.csv" --reps 3 --expect-passes 4
        gpu_job "ctl-tg128-$label" 70 600 "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128 -r 3 -o jsonl
        gpu_job "trace-tg128-$label" 70 600 rocprofv3 --kernel-trace --stats -f csv -d "$RUN_DIR/trace-tg128-$label.d" -o t -- \
            "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
        analyse_trace "trace-tg128-$label" tg "$size" "$wb" "$RUN_DIR/ctl-tg128-$label.log" "$RUN_DIR/trace-tg128-$label.d/t_kernel_trace.csv" --reps 128 --min-passes 115 --expect-passes 129
    done
    # known answer for the grouping: a single f16 MUL_MAT run
    gpu_job trace-bops-f16 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/trace-bops-f16.d" -o t -- \
        "$UNMOD/test-backend-ops" perf -b ROCm0 -o MUL_MAT -p 'type_a=f16,type_b=f32,m=4096,n=512,k=14336'
    grouping_known_answer
    # deliberate break 10: a profiled command that launches no kernel returns no rows (the profiler's own kernel regex
    # option did not filter in step 2, so this is the way to get an empty trace); the analysis must FAIL
    gpu_job trace-nomatch 70 300 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/trace-nomatch.d" -o t -- \
        "$UNMOD/llama-bench" --list-devices
    if "${PY[@]}" "$ROOT/scripts/llm_kernels.py" report "$RUN_DIR/trace-nomatch.d/t_kernel_trace.csv" --kind tg >"$RUN_DIR/trace-nomatch.report.txt" 2>&1; then
        sum "  - break10 (profiler regex that matches nothing): did NOT fail: the check cannot fail: FAILED"
        FAILED=1
    else
        sum "  - break10 (profiler regex that matches nothing): FAIL seen as intended ($(grep -m1 VERDICT "$RUN_DIR/trace-nomatch.report.txt"))"
    fi
}

bench_cmd() { # bin model extra...
    local b=$1 m=$2
    shift 2
    echo "$b/llama-bench" -m "$m" -ngl 99 -o jsonl "$@"
}
jsonl_ts() { # log: median tokens/s of the first result line
    grep -h '^{' "$1" | head -1 | "${PY[@]}" -c 'import json,sys,statistics;d=json.loads(sys.stdin.readline());print(statistics.median(d["samples_ts"]))'
}

# known (step 3): known answers on the 3B on both builds
phase_known() {
    local c label bin mdl
    sum ""
    sum "## known (step 3: llama-bench and test-backend-ops on the 3B)"
    for c in "${CONFIGS_3B[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        gpu_job "known-bench-$label" 70 600 "$bin/llama-bench" -m "$MODELS/$mdl.gguf" -ngl 99 -p 512 -n 128 -r 3 -o jsonl
    done
    gpu_job known-tbo-f8-unmod 70 900 "$UNMOD/test-backend-ops" test -b ROCm0 -p f8_e4m3
    gpu_job known-tbo-f8-patched 70 900 "$PATCHED/test-backend-ops" test -b ROCm0 -p f8_e4m3
    gpu_job known-tbo-q8-unmod 70 900 "$UNMOD/test-backend-ops" test -b ROCm0 -o MUL_MAT -p "type_a=q8_0"
    gpu_job known-tbo-f16-unmod 70 900 "$UNMOD/test-backend-ops" test -b ROCm0 -o MUL_MAT -p "type_a=f16"
    local k
    for k in known-tbo-f8-unmod known-tbo-f8-patched known-tbo-q8-unmod known-tbo-f16-unmod; do
        [[ -f $RUN_DIR/$k.log ]] && sum "  - $k: $(grep -E 'tests passed|Backend ROCm0:' "$RUN_DIR/$k.log" | tail -2 | tr '\n' ' ' | sed 's/\x1b\[[0-9;]*m//g')"
    done
}

# shapes (step 4, A3): test-backend-ops perf at the one shape family the tool offers (m = 4096, k = 14336, the 8B's
# down projection), n = 1 to 8 and 512, f8_e4m3 against f16 against q8_0, three repeats, both builds
phase_shapes() {
    local r b
    sum ""
    sum "## shapes (step 4: test-backend-ops perf MUL_MAT, m=4096 k=14336)"
    for b in unmod patched; do
        for r in 1 2 3; do
            gpu_job "shapes-$b-r$r" 70 600 "$([[ $b == unmod ]] && echo "$UNMOD" || echo "$PATCHED")/test-backend-ops" perf -b ROCm0 -o MUL_MAT \
                -p 'type_a=(f8_e4m3|f16|q8_0),type_b=f32,m=4096,n=[0-9]+,k=14336'
        done
    done
}

# grid (steps 6 and 7): prompt sizes, depth, parallel streams
phase_grid() {
    local c label bin mdl m
    sum ""
    sum "## grid (B1 prompt sizes, B4 depth, B2 parallel streams)"
    for c in "${GRID_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        m=$MODELS/$mdl.gguf
        gpu_job "bench-$label-prompts" 70 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 16,32,64,128,256,512,1024 -n 128 -r 5 -o jsonl
        gpu_job "bench-$label-pp2048" 70 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 2048 -n 0 -b 2048 -ub 2048 -r 5 -o jsonl
        gpu_job "bench-$label-pp4096" 70 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 4096 -n 0 -b 4096 -ub 4096 -r 3 -o jsonl
        gpu_job "bench-$label-depth" 70 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128 -d 0,512,4096 -r 5 -o jsonl
        gpu_job "batched-$label-npp128" 70 900 "$bin/llama-batched-bench" -m "$m" -ngl 99 -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,3,4,5,6,7,8
        gpu_job "batched-$label-npp512" 70 900 "$bin/llama-batched-bench" -m "$m" -ngl 99 -c 8192 -b 512 -ub 512 -npp 512 -ntg 128 -npl 1,2,4,8
    done
}

# quality (step 8, B3): perplexity at 4 chunks (known answers), then 50 chunks with KL divergence against the
# unmodified FP8 default-batch run
phase_quality() {
    local pre mdl chunks base v bin file extra name
    sum ""
    sum "## quality (B3: perplexity and KL divergence)"
    mkdir -p "$RUN_DIR/kld"
    for pre in llama32-3b llama31-8b; do
        # 4 chunks, the first-hour and llama-more settings
        gpu_job "ppl4-$pre-fp8-unmod-default" 70 900 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99
        gpu_job "ppl4-$pre-fp8-unmod-b8" 70 900 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99 -b 8 -ub 8
        gpu_job "ppl4-$pre-fp8-patched" 70 900 "$PATCHED/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99
        gpu_job "ppl4-$pre-fp8-patched-b8" 70 900 "$PATCHED/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99 -b 8 -ub 8
        gpu_job "ppl4-$pre-q8_0" 70 900 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-q8_0.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99
        # 50 chunks: reference logits, then the comparisons
        base=$RUN_DIR/kld/$pre-base.bin
        gpu_job "kld-$pre-base" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$base"
        [[ -s $base ]] || { sum "- kld $pre: no reference logits file (FAILED)"; FAILED=1; continue; }
        gpu_job "kld-$pre-a-same" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$base" --kl-divergence
        # the saved logits are 16-bit compressed, so "the same run again" sits at a noise floor, not at exactly 0
        if ! "${PY[@]}" "$ROOT/scripts/llm_tables.py" check-self "$RUN_DIR/kld-$pre-a-same.log" >"$RUN_DIR/kld-$pre-a-same.check.txt" 2>&1; then
            sum "- kld $pre: PARKED: the same run again is above the noise floor ($(cat "$RUN_DIR/kld-$pre-a-same.check.txt")); the comparisons of this model stop here, the finding is reported"
            rm -f "$base"
            continue
        fi
        sum "- kld $pre: $(cat "$RUN_DIR/kld-$pre-a-same.check.txt")"
        gpu_job "kld-$pre-b-b8" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 -b 8 -ub 8 --kl-divergence-base "$base" --kl-divergence
        gpu_job "kld-$pre-c-patched" 70 1500 "$PATCHED/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$base" --kl-divergence
        gpu_job "kld-$pre-d-q8_0" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-q8_0.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$base" --kl-divergence
        rm -f "$base"
    done
}

# behaviour (step 9, C1 to C4): the sampler records clocks, power, temperatures at 20 ms around each load
cooldown() { # seconds
    sleep "$1"
}
phase_behaviour() {
    local label bin mdl m reps_pp reps_tg c
    sum ""
    sum "## behaviour (C1 to C4: 20 ms sampler around loops of about 20 s, 60 s cool-down between)"
    mkdir -p "$RUN_DIR/beh"
    gpu_job beh-idle 60 60 "$SELF" __watch "$RUN_DIR/beh/idle" idle sleep 10
    for c in "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8|350|25" "3b-q8_0|$UNMOD|llama32-3b-q8_0|450|23" \
        "8b-fp8-patched|$PATCHED|llama31-8b-fp8-outq8|170|13" "8b-q8_0|$UNMOD|llama31-8b-q8_0|200|12"; do
        IFS='|' read -r label bin mdl reps_pp reps_tg <<<"$c"
        m=$MODELS/$mdl.gguf
        gpu_job "beh-load-$label" 70 300 "$SELF" __watch "$RUN_DIR/beh/load-$label" "load-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 1 -r 1 -o jsonl
        cooldown 20
        gpu_job "beh-pp-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/pp-$label" "pp-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r "$reps_pp" -o jsonl
        cooldown 60
        gpu_job "beh-tg-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/tg-$label" "tg-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128 -r "$reps_tg" -o jsonl
        cooldown 60
    done
    # C4: prompts and generation 2 s after idle, against the same in a loop
    for c in "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8" "3b-q8_0|$UNMOD|llama32-3b-q8_0"; do
        IFS='|' read -r label bin mdl <<<"$c"
        m=$MODELS/$mdl.gguf
        gpu_job "beh-idle2s-pp-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/idle2s-pp-$label" "idle2s-pp-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 512,512,512,512,512,512 -n 0 -r 1 --no-warmup --delay 2 -o jsonl
        gpu_job "beh-loop-pp-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/loop-pp-$label" "loop-pp-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 12 --no-warmup -o jsonl
        gpu_job "beh-idle2s-tg-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/idle2s-tg-$label" "idle2s-tg-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128,128,128,128 -r 1 --no-warmup --delay 2 -o jsonl
        cooldown 30
    done
}

# builds (CPU only, A5): Q8_0 through dequantize + hipBLAS, by a compile-time switch (no kernel code touched), in a NEW
# folder next to the existing builds. nice, -j8, memory capped at 10 GB.
FORCECUBLAS_DIR=$BUILD_ROOT/llm-dive-forcecublas
phase_builds() {
    local src=$BUILD_ROOT/llama.cpp-latest
    sum ""
    sum "## builds (A5: -DGGML_CUDA_FORCE_CUBLAS=ON, unmodified source tree, new build folder)"
    if (($(avail_gb) < 8)); then
        sum "- build NOT RUN: $(avail_gb) GB available (floor 8)"
        FAILED=1
        return
    fi
    mkdir -p "$FORCECUBLAS_DIR"
    if ! HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++" HIP_PATH="$ROCM_PATH" nice -n 10 cmake -S "$src" -B "$FORCECUBLAS_DIR" \
        -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
        -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DLLAMA_CURL=OFF -DGGML_CUDA_FORCE_CUBLAS=ON >"$RUN_DIR/build-forcecublas.cmake.log" 2>&1; then
        sum "- configure FAILED (see build-forcecublas.cmake.log)"
        FAILED=1
        return
    fi
    if systemd-run --user --scope -q -p MemoryMax=10G nice -n 10 cmake --build "$FORCECUBLAS_DIR" -j8 \
        --target llama-bench test-backend-ops >"$RUN_DIR/build-forcecublas.build.log" 2>&1; then
        sum "- built $FORCECUBLAS_DIR (source HEAD $(git -C "$src" rev-parse --short HEAD))"
    else
        sum "- build FAILED: $(grep -m2 -E 'error' "$RUN_DIR/build-forcecublas.build.log")"
        FAILED=1
    fi
    grep -E "^(GGML_CUDA_FORCE_CUBLAS|GGML_HIP_GRAPHS|CMAKE_BUILD_TYPE)[:=]" "$FORCECUBLAS_DIR/CMakeCache.txt" >"$RUN_DIR/builds-forcecublas.txt" 2>&1
    stat -c '%y %n' "$FORCECUBLAS_DIR/bin/llama-bench" >>"$RUN_DIR/builds-forcecublas.txt" 2>&1
}

# a5: the A5 build against the unmodified build, Q8_0, prompt speed and a kernel trace
phase_a5() {
    local pre bin=$FORCECUBLAS_DIR/bin label m
    [[ -x $bin/llama-bench ]] || { sum ""; sum "## a5: NOT RUN, no build at $FORCECUBLAS_DIR"; FAILED=1; return; }
    sum ""
    sum "## a5 (Q8_0 through hipBLAS: -DGGML_CUDA_FORCE_CUBLAS=ON)"
    for pre in 3b:llama32-3b 8b:llama31-8b; do
        label=${pre%%:*}
        m=$MODELS/${pre#*:}-q8_0.gguf
        gpu_job "a5-bench-$label-q8_0-cublas" 70 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 16,32,64,128,256,512,1024 -n 128 -r 5 -o jsonl
        gpu_job "a5-ctl-pp512-$label-q8_0-cublas" 70 600 "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 5 -o jsonl
        gpu_job "a5-trace-pp512-$label-q8_0-cublas" 70 600 rocprofv3 --kernel-trace --stats -f csv -d "$RUN_DIR/a5-trace-pp512-$label-q8_0-cublas.d" -o t -- \
            "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 3 -o jsonl
    done
}

# reanalyse: no GPU. Runs the analysis again over the traces already in the run folder (after a change to the analysis).
phase_reanalyse() {
    local c label bin mdl size wb
    sum ""
    sum "## reanalyse (no GPU: the trace analysis over the saved traces)"
    for c in "${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        size=${label%%-*}
        wb=1.0
        [[ $label == *q8_0* ]] && wb=1.0625
        analyse_trace "trace-pp512-$label" pp "$size" "$wb" "$RUN_DIR/ctl-pp512-$label.log" "$RUN_DIR/trace-pp512-$label.d/t_kernel_trace.csv" --reps 3 --expect-passes 4
        analyse_trace "trace-tg128-$label" tg "$size" "$wb" "$RUN_DIR/ctl-tg128-$label.log" "$RUN_DIR/trace-tg128-$label.d/t_kernel_trace.csv" --reps 128 --min-passes 115 --expect-passes 129
    done
    for c in "$RUN_DIR"/a5-trace-pp512-*.d; do
        [[ -d $c ]] || continue
        label=${c##*/a5-trace-pp512-}
        label=${label%.d}
        analyse_trace "a5-trace-pp512-$label" pp "${label%%-*}" 1.0625 "$RUN_DIR/a5-ctl-pp512-$label.log" "$c/t_kernel_trace.csv" --reps 3 --expect-passes 4
    done
    grouping_known_answer
    if ! "${PY[@]}" "$ROOT/scripts/llm_kernels.py" report "$RUN_DIR/trace-nomatch.d/t_kernel_trace.csv" --kind tg >"$RUN_DIR/trace-nomatch.report.txt" 2>&1; then
        sum "  - break10 (a profiled command with no kernels): FAIL seen as intended ($(grep -m1 VERDICT "$RUN_DIR/trace-nomatch.report.txt"))"
    fi
}

# det_compare <a> <b> <evidence file>: compares two logits files and writes the evidence (size and sha256 of each, the
# cmp exit status and its output) BEFORE the caller may delete a file. Returns cmp's status: 0 identical, 1 differ, 2 trouble.
det_compare() {
    local a=$1 b=$2 ev=$3 out rc
    {
        echo "# determinism evidence, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "file_a: ${a##*/}"
        echo "size_a: $(stat -c %s "$a" 2>&1)"
        echo "sha256_a: $(sha256sum "$a" 2>&1 | cut -d' ' -f1)"
        echo "file_b: ${b##*/}"
        echo "size_b: $(stat -c %s "$b" 2>&1)"
        echo "sha256_b: $(sha256sum "$b" 2>&1 | cut -d' ' -f1)"
    } 2>/dev/null >"$ev" || return 2
    out=$(cmp "$a" "$b" 2>&1)
    rc=$?
    {
        echo "cmp_exit: $rc"
        echo "cmp_output: ${out:-<empty: no difference>}"
        ((rc == 1)) && echo "differing_bytes: $(cmp -l "$a" "$b" 2>/dev/null | wc -l)"
    } >>"$ev"
    return "$rc"
}

# det_finish <label> <a> <b> <evidence file>: det_compare, then one summary line on stdout, then the two logits files
# are deleted only if the evidence file was written. Returns 0 identical, 1 differ (a finding), 2 failed.
det_finish() {
    local pre=$1 a=$2 b=$3 ev=$4 rc
    if [[ ! -s $a || ! -s $b ]]; then
        echo "- determinism $pre: a logits file is missing (FAILED)"
        return 2
    fi
    det_compare "$a" "$b" "$ev"
    rc=$?
    if [[ ! -s $ev ]] || ! grep -q '^cmp_exit:' "$ev"; then
        echo "- determinism $pre: the evidence could not be written to ${ev##*/}; the logits files were kept (FAILED)"
        return 2
    fi
    case $rc in
    0) echo "- determinism $pre: the two logits files are byte-identical (cmp exit 0; $(sed -n 's/^size_a: //p' "$ev") bytes each, sha256 $(sed -n 's/^sha256_a: //p' "$ev" | cut -c1-16)...; evidence in ${ev##*/})" ;;
    1) echo "- determinism $pre: the two logits files DIFFER ($(sed -n 's/^differing_bytes: //p' "$ev") bytes; evidence in ${ev##*/})" ;;
    *) echo "- determinism $pre: cmp failed with exit $rc (FAILED; evidence in ${ev##*/})"; return 2 ;;
    esac
    rm -f "$a" "$b"
    return "$rc"
}

# determinism: the reference logits of the unmodified FP8 default-batch run written twice, compared byte for byte
phase_determinism() {
    local pre a b line rc
    sum ""
    sum "## determinism (two identical perplexity runs, logits files compared with cmp, evidence kept in det-<model>.compare.log)"
    mkdir -p "$RUN_DIR/kld"
    for pre in llama32-3b llama31-8b; do
        a=$RUN_DIR/kld/$pre-det1.bin
        b=$RUN_DIR/kld/$pre-det2.bin
        gpu_job "det-$pre-1" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$a"
        gpu_job "det-$pre-2" 70 1500 "$UNMOD/llama-perplexity" -m "$MODELS/$pre-auto.gguf" -f "$TEXT" -c 512 --chunks 50 -ngl 99 --kl-divergence-base "$b"
        line=$(det_finish "$pre" "$a" "$b" "$RUN_DIR/det-$pre.compare.log")
        rc=$?
        sum "$line"
        ((rc == 2)) && FAILED=1
    done
}

# ---------------------------------------------------------------------------------------------------------------
# Round 2 (llm-workload-2.md). Items run in the run folder LLM_RUN, after the round-1 phases' gates.
# r2q (item 1): quality against an exact-weights reference (F32 = FP8 x scale, built by scripts/llm_ref.py into
# $RUN_DIR/refs). The 3B reference fits the card (-lm none: reading the file into memory; with mmap the driver pins the
# mapped pages and the upload took minutes). The 8B reference (32 GB) does not: -ngl layers on the card, the rest read
# from the memory-mapped file by the CPU (--no-op-offload: without it the loader copied each CPU layer to the card from mapped pages
# on every pass and a pass did not finish in 8 minutes; see llm-workload-2.md).
R2_8B_NGL=${R2_8B_NGL:-27}   # layers of the 16 GB F16 reference on the card
R2_CHUNKS=${R2_CHUNKS:-50}
ref_evidence() { # <label> <logits file> <evidence file>: size and sha256 BEFORE the file is deleted; returns 1 if it could not be written
    local label=$1 f=$2 ev=$3
    [[ -s $f ]] || { echo "- $label: no logits file (FAILED)"; return 1; }
    { echo "file: ${f##*/}"; echo "size: $(stat -c %s "$f")"; echo "sha256: $(sha256sum "$f" | cut -d' ' -f1)"; } >"$ev" 2>/dev/null
    [[ -s $ev ]] && grep -q '^sha256: [0-9a-f]\{64\}$' "$ev" || { echo "- $label: evidence not written, logits kept (FAILED)"; return 1; }
    rm -f "$f"
    echo "- $label: reference logits $(sed -n 's/^size: //p' "$ev") bytes, sha256 $(sed -n 's/^sha256: //p' "$ev" | cut -c1-16)... logged in ${ev##*/}, file deleted"
}
phase_r2q() {
    local spec pre refname lm ngl extra ref base v bin file
    sum ""
    sum "## r2q (item 1: perplexity and KL divergence against the reference, $R2_CHUNKS chunks of 512)"
    mkdir -p "$RUN_DIR/kld2"
    # pre | reference file name | load mode | -ngl | extra arguments. The 8B exact F32 file (32 GB) could not be run on this machine (see
    # llm-workload-2.md); its reference here is the F16 file (16 GB), whose own error is measured on the 3B (compared as "f16ref").
    for spec in "llama32-3b|exact-f32|none|99|" "llama31-8b|f16|mmap|$R2_8B_NGL|--no-op-offload"; do
        IFS='|' read -r pre refname lm ngl extra <<<"$spec"
        [[ -n ${R2_ONLY:-} && $pre != "$R2_ONLY" ]] && continue
        ref=$RUN_DIR/refs/$pre-$refname.gguf
        [[ -s $ref ]] || { sum "- r2q $pre: no reference file $ref (FAILED)"; FAILED=1; continue; }
        base=$RUN_DIR/kld2/$pre-$refname-base.bin
        # shellcheck disable=SC2086
        gpu_job "r2q-$pre-ref" 60 "${R2_REF_TIMEOUT:-3600}" "$UNMOD/llama-perplexity" -m "$ref" -f "$TEXT" -c 512 --chunks "$R2_CHUNKS" -ngl "$ngl" -lm "$lm" $extra --kl-divergence-base "$base" || \
            { sum "- r2q $pre: the reference run did not finish (a partial logits file is not used): FAILED"; rm -f "$base"; FAILED=1; continue; }
        [[ -s $base ]] || { sum "- r2q $pre: the reference run wrote no logits file (FAILED)"; FAILED=1; continue; }
        for v in "fp8-unmod|$UNMOD|$MODELS/$pre-auto.gguf" "fp8-patched|$PATCHED|$MODELS/$pre-auto.gguf" "q8_0|$UNMOD|$MODELS/$pre-q8_0.gguf" "f16ref|$UNMOD|$RUN_DIR/refs/$pre-f16.gguf"; do
            IFS='|' read -r v bin file <<<"$v"
            [[ $v == f16ref && $refname == f16 ]] && continue
            [[ -s $file ]] || continue
            gpu_job "r2q-$pre-$v" 70 1500 "$bin/llama-perplexity" -m "$file" -f "$TEXT" -c 512 --chunks "$R2_CHUNKS" -ngl 99 -lm none --kl-divergence-base "$base" --kl-divergence
        done
        sum "$(ref_evidence "r2q $pre" "$base" "$RUN_DIR/r2q-$pre.base.txt")"
    done
}

# r2libs (item 2): the first-hour library probe (hipBLASLt FP8 and hipBLAS FP16 GEMM) under a kernel trace, at the first-hour shape
# and at llama.cpp's shapes (M = weight rows, N = tokens, K = inputs). rocBLAS's own call log (ROCBLAS_LAYER=2) names the compute type.
PROBE_LT=$ROOT/build/gfx1201/hipblaslt_fp8
phase_r2libs() {
    local spec tag m n k
    sum ""
    sum "## r2libs (item 2: the first-hour probe under rocprofv3, first-hour shape against llama.cpp's shapes)"
    [[ -x $PROBE_LT ]] || { sum "- no probe binary $PROBE_LT (FAILED)"; FAILED=1; return; }
    ldd "$PROBE_LT" 2>&1 | grep -E "hipblas|rocblas|amdhip|hsa" | sed 's/(0x.*//' >"$RUN_DIR/r2libs-ldd.txt"
    for spec in "orig|4096|4096|512" "n512k4096|4096|512|4096" "n512k14336|4096|512|14336" "n512k8192|3072|512|8192" "cube|4096|4096|4096" "n512k512|4096|512|512"; do
        IFS='|' read -r tag m n k <<<"$spec"
        gpu_job "r2libs-$tag" 70 300 env ROCBLAS_LAYER=2 rocprofv3 --kernel-trace --hip-trace -f csv -d "$RUN_DIR/r2libs-$tag.d" -o t -- "$PROBE_LT" "$m" "$n" "$k" 50
    done
    # control: the same llama.cpp-shaped call with rocBLAS's hipBLASLt backend switched off (a different library path, if the default is hipBLASLt)
    gpu_job r2libs-n512k4096-notensile 70 300 env ROCBLAS_LAYER=2 ROCBLAS_USE_HIPBLASLT=0 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2libs-n512k4096-notensile.d" -o t -- "$PROBE_LT" 4096 512 4096 50
    # llama.cpp's own f16 route at the same shape as a reference point
    gpu_job r2libs-bops-f16 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2libs-bops-f16.d" -o t -- \
        "$UNMOD/test-backend-ops" perf -b ROCm0 -o MUL_MAT -p 'type_a=f16,type_b=f32,m=4096,n=512,k=14336'
}

# r2overlap (item 3): the tg128 trace of the 8B patched build repeated (the rerun's no-overlap FAIL), the 3B and Q8_0 for contrast,
# graphs switched off, and graph concurrency switched on (GGML_CUDA_GRAPH_OPT=1: forked streams, a positive control for the finder)
phase_r2overlap() {
    local i
    sum ""
    sum "## r2overlap (item 3: where kernels overlap in tg128 traces, repeated)"
    local m8=$MODELS/llama31-8b-fp8-outq8.gguf m3=$MODELS/llama32-3b-fp8-outq8.gguf q8=$MODELS/llama31-8b-q8_0.gguf
    for i in 1 2 3 4 5; do
        gpu_job "r2ov-8b-patched-$i" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ov-8b-patched-$i.d" -o t -- "$PATCHED/llama-bench" -m "$m8" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
    done
    for i in 1 2; do
        gpu_job "r2ov-3b-patched-$i" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ov-3b-patched-$i.d" -o t -- "$PATCHED/llama-bench" -m "$m3" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
        gpu_job "r2ov-8b-q8-$i" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ov-8b-q8-$i.d" -o t -- "$UNMOD/llama-bench" -m "$q8" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
        gpu_job "r2ov-8b-nograph-$i" 70 600 env GGML_CUDA_DISABLE_GRAPHS=1 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ov-8b-nograph-$i.d" -o t -- "$PATCHED/llama-bench" -m "$m8" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
        gpu_job "r2ov-8b-graphopt-$i" 70 600 env GGML_CUDA_GRAPH_OPT=1 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ov-8b-graphopt-$i.d" -o t -- "$PATCHED/llama-bench" -m "$m8" -ngl 99 -p 0 -n 128 -r 1 -o jsonl
    done
    local d
    for d in 8b-patched 3b-patched 8b-q8 8b-nograph 8b-graphopt; do
        "${PY[@]}" "$ROOT/scripts/llm_kernels.py" overlap "$RUN_DIR"/r2ov-$d-[0-9].d/t_kernel_trace.csv --json "$RUN_DIR/r2ov-$d.overlap.json" >"$RUN_DIR/r2ov-$d.overlap.txt" 2>&1
        sum "- overlap $d: $(head -1 "$RUN_DIR/r2ov-$d.overlap.txt" | cut -c1-200) (full list in r2ov-$d.overlap.txt)"
    done
}

# r2ub (item 5): pp at micro-batch 2048 against 4 x 512, traced, 3B patched FP8 and Q8_0 (same command shape as round 1)
phase_r2ub() {
    local c label bin mdl m
    sum ""
    sum "## r2ub (item 5: the 3B prompt pass at micro-batch 2048 against 512, per-group kernel times)"
    for c in "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8" "3b-q8_0|$UNMOD|llama32-3b-q8_0"; do
        IFS='|' read -r label bin mdl <<<"$c"
        m=$MODELS/$mdl.gguf
        gpu_job "r2ub-ctl-512-$label" 70 600 "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 5 -o jsonl
        gpu_job "r2ub-ctl-2048-$label" 70 600 "$bin/llama-bench" -m "$m" -ngl 99 -p 2048 -n 0 -b 2048 -ub 2048 -r 5 -o jsonl
        gpu_job "r2ub-trace-512-$label" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ub-trace-512-$label.d" -o t -- \
            "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r 3 -o jsonl
        gpu_job "r2ub-trace-2048-$label" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2ub-trace-2048-$label.d" -o t -- \
            "$bin/llama-bench" -m "$m" -ngl 99 -p 2048 -n 0 -b 2048 -ub 2048 -r 3 -o jsonl
        "${PY[@]}" "$ROOT/scripts/llm_kernels.py" groupcmp "$RUN_DIR/r2ub-trace-512-$label.d/t_kernel_trace.csv" "$RUN_DIR/r2ub-trace-2048-$label.d/t_kernel_trace.csv" \
            --kind pp --tokens-a 512 --tokens-b 2048 --per 2048 --model 3b >"$RUN_DIR/r2ub-$label.groupcmp.txt" 2>&1
        sum "- groupcmp $label: $(sed -n 2p "$RUN_DIR/r2ub-$label.groupcmp.txt" | cut -c1-220) (r2ub-$label.groupcmp.txt)"
    done
}

# r2clock (item 6): the round-1 prompt-size command of the 8B repeated under the 20 ms sampler, to relate the clock at the start of
# each short test to its speed
phase_r2clock() {
    local i c label bin mdl
    sum ""
    sum "## r2clock (item 6: 8B prompt rows pp16 to pp1024 repeated under the sampler)"
    mkdir -p "$RUN_DIR/r2clock"
    for i in 1 2 3 4 5 6; do
        for c in "8b-fp8-patched|$PATCHED|llama31-8b-fp8-outq8" "8b-q8_0|$UNMOD|llama31-8b-q8_0"; do
            IFS='|' read -r label bin mdl <<<"$c"
            gpu_job "r2clock-$label-$i" 60 600 "$SELF" __watch "$RUN_DIR/r2clock/$label-$i" "r2clock-$label-$i" \
                "$bin/llama-bench" -m "$MODELS/$mdl.gguf" -ngl 99 -p 16,32,64,128,256,512,1024 -n 0 -r 5 -o jsonl
            cooldown $((i * 7 % 20 + 5))
        done
    done
}

# r2streams (item 7a): llama-batched-bench streams 1 to 8, repeated in an interleaved order
phase_r2streams() {
    local rep c label bin mdl
    sum ""
    sum "## r2streams (item 7a: streams 1 to 8, interleaved repetitions)"
    for rep in 1 2 3 4 5 6; do
        for c in "${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}"; do
            IFS='|' read -r label bin mdl <<<"$c"
            gpu_job "r2streams-$label-r$rep" 70 900 "$bin/llama-batched-bench" -m "$MODELS/$mdl.gguf" -ngl 99 -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,3,4,5,6,7,8
        done
    done
}

# r2counters (item 4): which hardware counters of gfx1201 count anything. A counter is run against the bandwidth probe (its read and
# copy kernels move 256 MiB); one that reads 0 in every dispatch of a kernel that certainly moves data counts nothing.
phase_r2counters() {
    local set n
    sum ""
    sum "## r2counters (item 4: which counters count on gfx1201, against a kernel that reads 256 MiB)"
    gpu_job r2c-list 70 120 rocprofv3-avail list
    for set in "GRBM_COUNT GRBM_GUI_ACTIVE" "SQ_WAVES SQ_BUSY_CYCLES" "SIMD_UTILIZATION" "SQ_INSTS_VALU SQ_INSTS_WAVE32_VALU SQ_INSTS_FLAT" "SQ_WAVE32_INSTS SQ_INSTS_LDS SQ_INST_CYCLES_VMEM" \
        "SQ_WAVE_CYCLES SQ_INST_CYCLES_VALU SQ_INSTS_TEX_STORE" "GL2C_EA_RDREQ_sum GL2C_EA_WRREQ_sum" "GL2C_HIT_sum GL2C_MISS_sum" "FETCH_SIZE" "TCP_REQ TCP_REQ_MISS" \
        "L2CacheHit" "GL2C_EA_RDREQ_128B_sum GL2C_EA_WRREQ_64B_sum" "MemUnitBusy VALUBusy"; do
        n=$(echo "$set" | tr ' ' '-' | cut -c1-40)
        # shellcheck disable=SC2086
        gpu_job "r2c-$n" 70 300 rocprofv3 --pmc $set -f csv -d "$RUN_DIR/r2c-$n.d" -o p -- "$BANDWIDTH" 256 3
        "${PY[@]}" "$ROOT/scripts/llm_kernels.py" pmc "$RUN_DIR/r2c-$n.d/p_counter_collection.csv" --kernel read_kernel --moves-data >"$RUN_DIR/r2c-$n.pmc.txt" 2>&1
        sum "  - $set: $(awk '{printf "%s %s; ", $1, $NF}' "$RUN_DIR/r2c-$n.pmc.txt" | cut -c1-200)"
    done
}

# r2mall (item 4, the 64 MB memory-side cache): the widen access shape (read S, write 2 S) with the scratch destination reused or
# rotating, at three sizes, plus read-only and write-only kernels. Counters cannot answer this (r2counters).
phase_r2mall() {
    local sz rep
    sum ""
    sum "## r2mall (item 4: does the write half of a widen stay in the last-level cache? src/widen_cache.hip)"
    [[ -x $ROOT/build/gfx1201-rocm10/widen_cache ]] || { sum "- no build/gfx1201-rocm10/widen_cache (build it with scripts/study-llm.sh r2mallbuild) (FAILED)"; FAILED=1; return; }
    for rep in $(seq 1 "${R2MALL_REPS:-3}"); do
        for sz in 8 25 50 100; do
            gpu_job "r2mall-$sz-r$rep" 70 300 "$ROOT/build/gfx1201-rocm10/widen_cache" "$sz" 200
            grep -q '^VERDICT widen_cache PASS' "$RUN_DIR/r2mall-$sz-r$rep.log" 2>/dev/null || { sum "- r2mall-$sz-r$rep: no 'VERDICT widen_cache PASS' in its log (FAILED)"; FAILED=1; }
        done
    done
}

# r2mallbuild (CPU only): build src/widen_cache.hip with -save-temps and check the ISA of its three loops against
# scripts/mem-expect/widen_cache.txt; a deliberately wrong expectation must FAIL.
phase_r2mallbuild() {
    local d=$RUN_DIR/widen_cache-isa exp=$ROOT/scripts/mem-expect/widen_cache.txt rc
    sum ""
    sum "## r2mallbuild (src/widen_cache.hip: build and ISA check of the timed loops, CPU only)"
    mkdir -p "$d" "$ROOT/build/gfx1201-rocm10"
    if ! (cd "$d" && systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 \
        -DHARNESS_ARCH='"gfx1201"' -save-temps "$ROOT/src/widen_cache.hip" -o "$ROOT/build/gfx1201-rocm10/widen_cache" -lamdhip64) >"$RUN_DIR/r2mallbuild.log" 2>&1; then
        sum "- build FAILED (r2mallbuild.log)"; FAILED=1; return
    fi
    sum "- built build/gfx1201-rocm10/widen_cache"
    "${PY[@]}" "$ROOT/scripts/mem_loopcheck.py" "$d"/*.lto.s "$exp" >"$RUN_DIR/r2mallbuild-loopcheck.txt" 2>&1
    rc=$?
    sum "- ISA check of the timed loops: $(tail -1 "$RUN_DIR/r2mallbuild-loopcheck.txt") (r2mallbuild-loopcheck.txt)"
    ((rc == 0)) || FAILED=1
    sed 's/^widen_kernel loop global_store_b128\\s == 2$/widen_kernel loop global_store_b128\\s == 3/' "$exp" >"$d/expect-broken.txt"
    cmp -s "$exp" "$d/expect-broken.txt" && { sum "- break not applied (FAILED)"; FAILED=1; return; }
    "${PY[@]}" "$ROOT/scripts/mem_loopcheck.py" "$d"/*.lto.s "$d/expect-broken.txt" >"$RUN_DIR/r2mallbuild-loopcheck-break.txt" 2>&1
    if (($? == 1)) && grep -q '^FAIL .*widen_kernel' "$RUN_DIR/r2mallbuild-loopcheck-break.txt"; then
        sum "- deliberate break (expects 3 stores in the widen loop, it has 2): the ISA check FAILED as intended"
    else
        sum "- deliberate break was NOT caught by the ISA check (FAILED)"; FAILED=1
    fi
}

# r2mallcheck: the probe's known answer on the card once, and its deliberate break (8 MiB read instead of 1 GiB) which must FAIL.
phase_r2mallcheck() {
    local f0=$FAILED rc
    sum ""
    sum "## r2mallcheck (src/widen_cache.hip: known answer, and its deliberate break must FAIL)"
    [[ -x $ROOT/build/gfx1201-rocm10/widen_cache ]] || { sum "- no build/gfx1201-rocm10/widen_cache (r2mallbuild first) (FAILED)"; FAILED=1; return; }
    gpu_job r2mallcheck-ok 70 120 "$ROOT/build/gfx1201-rocm10/widen_cache" 25 50
    grep -q '^VERDICT known-answer PASS' "$RUN_DIR/r2mallcheck-ok.log" 2>/dev/null || { sum "- r2mallcheck-ok: no 'VERDICT known-answer PASS' (FAILED)"; FAILED=1; }
    f0=$FAILED
    gpu_job r2mallcheck-break 70 120 "$ROOT/build/gfx1201-rocm10/widen_cache" 25 50 break
    rc=$?
    if ((rc == 1)) && grep -q '^VERDICT known-answer FAIL' "$RUN_DIR/r2mallcheck-break.log"; then
        FAILED=$f0
        sum "- r2mallcheck-break: the deliberate break FAILED as intended ($(grep -m1 '^RESULT known_answer_read' "$RUN_DIR/r2mallcheck-break.log"))"
    else
        sum "- r2mallcheck-break: the deliberate break did NOT fail (exit $rc) (FAILED)"; FAILED=1
    fi
}

# r2build (item 7b, CPU only): test-backend-ops from a SEPARATE copy of the unmodified llama.cpp tree with extra MUL_MAT perf cases
# (scripts/llm2-test-shapes.patch, tests/ only), built into the copy's own folder. Proves with a diff that nothing under ggml/
# or any other source folder differs from the original tree.
SHAPES_SRC=$BUILD_ROOT/llama.cpp-llm2-shapes
SHAPES_BIN=$SHAPES_SRC/build-gfx1201-rocm10/bin
phase_r2build() {
    local orig=$BUILD_ROOT/llama.cpp-latest d=$RUN_DIR/r2build-diff.txt
    sum ""
    sum "## r2build (item 7b: test-backend-ops with extra MUL_MAT shapes, a separate copy of the tree)"
    if [[ ! -d $SHAPES_SRC/.git ]]; then
        git clone -q "$orig" "$SHAPES_SRC" || { sum "- clone FAILED"; FAILED=1; return; }
        git -C "$SHAPES_SRC" apply "$ROOT/scripts/llm2-test-shapes.patch" || { sum "- the patch does not apply (FAILED)"; FAILED=1; return; }
    fi
    {
        echo "# original HEAD $(git -C "$orig" rev-parse HEAD), copy HEAD $(git -C "$SHAPES_SRC" rev-parse HEAD)"
        echo "# files changed in the copy against its HEAD:"
        git -C "$SHAPES_SRC" status --short
        echo "# every file the ORIGINAL tree tracks (git ls-files), compared byte for byte with the copy, except tests/test-backend-ops.cpp:"
        local f n=0 bad=0
        while IFS= read -r f; do
            [[ $f == tests/test-backend-ops.cpp ]] && continue
            n=$((n + 1))
            cmp -s "$orig/$f" "$SHAPES_SRC/$f" || { echo "DIFFERS: $f"; bad=$((bad + 1)); }
        done < <(git -C "$orig" ls-files)
        echo "compared $n files, $bad differ"
        echo "# files under ggml/ that differ: $(git -C "$SHAPES_SRC" diff --name-only -- ggml | wc -l)"
        echo "# the patch's own diffstat:"
        git -C "$SHAPES_SRC" diff --stat
        echo "# end"
    } >"$d" 2>&1
    if grep -q '^DIFFERS' "$d" || ! grep -q ', 0 differ$' "$d" || [[ $(git -C "$SHAPES_SRC" status --short | grep -v ' tests/test-backend-ops.cpp$' | grep -vc '^??') -gt 0 ]]; then
        sum "- the copy differs from the original outside tests/test-backend-ops.cpp: see $d (FAILED)"
        FAILED=1
        return
    fi
    sum "- diff against the original tree: only tests/test-backend-ops.cpp differs (proof in r2build-diff.txt)"
    if (($(avail_gb) < 8)); then sum "- build NOT RUN: $(avail_gb) GB available (floor 8)"; FAILED=1; return; fi
    mkdir -p "$SHAPES_SRC/build-gfx1201-rocm10"
    if ! HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++" HIP_PATH="$ROCM_PATH" nice -n 10 cmake -S "$SHAPES_SRC" -B "$SHAPES_SRC/build-gfx1201-rocm10" \
        -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
        -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DLLAMA_CURL=OFF >"$RUN_DIR/r2build.cmake.log" 2>&1; then
        sum "- configure FAILED (r2build.cmake.log)"; FAILED=1; return
    fi
    if systemd-run --user --scope -q -p MemoryMax=10G nice -n 10 cmake --build "$SHAPES_SRC/build-gfx1201-rocm10" -j8 --target test-backend-ops >"$RUN_DIR/r2build.build.log" 2>&1; then
        sum "- built $SHAPES_BIN/test-backend-ops ($(stat -c '%y' "$SHAPES_BIN/test-backend-ops"))"
    else
        sum "- build FAILED: $(grep -m2 -E 'error' "$RUN_DIR/r2build.build.log")"; FAILED=1
    fi
}

# r2shapes (item 7b): the new perf cases, three repetitions
phase_r2shapes() {
    local r
    sum ""
    sum "## r2shapes (item 7b: MUL_MAT perf at the 3B and 8B weight shapes, f8_e4m3 / f16 / q8_0, n = 1 8 32 512 2048)"
    [[ -x $SHAPES_BIN/test-backend-ops ]] || { sum "- no $SHAPES_BIN/test-backend-ops (run r2build) (FAILED)"; FAILED=1; return; }
    for r in 1 2 3; do
        gpu_job "r2shapes-r$r" 70 1800 "$SHAPES_BIN/test-backend-ops" perf -b ROCm0 -o MUL_MAT \
            -p 'type_a=(f8_e4m3|f16|q8_0),type_b=f32,m=(3072|1024|8192|128256|4096|14336),n=(1|8|32|512|2048),k=(3072|8192|4096|14336),bs=\[1,1\]'
    done
}

# r2shapes-trace (item 7b follow-up): at n = 2048 f8 - f16 was far below the widen time; the kernels of one shape (3B gate/up,
# m 8192 k 3072) at n = 512 and 2048, f8_e4m3 and f16, each in its own trace
phase_r2shapes_trace() {
    local t n
    sum ""
    sum "## r2shapes-trace (item 7b: kernels of m=8192 k=3072 at n=512 and n=2048)"
    [[ -x $SHAPES_BIN/test-backend-ops ]] || { sum "- no $SHAPES_BIN/test-backend-ops (FAILED)"; FAILED=1; return; }
    for n in 512 2048; do
        for t in f8_e4m3 f16; do
            gpu_job "r2sht-$t-n$n" 70 600 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r2sht-$t-n$n.d" -o t -- \
                "$SHAPES_BIN/test-backend-ops" perf -b ROCm0 -o MUL_MAT -p "type_a=$t,type_b=f32,m=8192,n=$n,k=3072,bs=\\[1,1\\]"
        done
    done
}

# ---------------------------------------------------------------------------------------------------------------
# Round 3. Q4: the same counter attempts with ROCm 10.0's rocprofv3 (pip packages in toolchains/torch-rocm10-venv, nothing installed)
R3_VENV=$ROOT/toolchains/torch-rocm10-venv
R3_VLIB=$R3_VENV/lib/python3.14/site-packages/_rocm_sdk_core/lib
R3_COUNTER_SETS=("GRBM_COUNT GRBM_GUI_ACTIVE" "SQ_WAVES SQ_BUSY_CYCLES" "SIMD_UTILIZATION" "SQ_INSTS_VALU SQ_INSTS_WAVE32_VALU SQ_INSTS_FLAT" "SQ_WAVE32_INSTS SQ_INSTS_LDS SQ_INST_CYCLES_VMEM"
    "SQ_WAVE_CYCLES SQ_INST_CYCLES_VALU SQ_INSTS_TEX_STORE" "GL2C_EA_RDREQ_sum GL2C_EA_WRREQ_sum" "GL2C_HIT_sum GL2C_MISS_sum" "FETCH_SIZE" "TCP_REQ TCP_REQ_MISS"
    "L2CacheHit" "GL2C_EA_RDREQ_128B_sum GL2C_EA_WRREQ_64B_sum" "MemUnitBusy VALUBusy")

# r3sys: read-only record of the kernel and the driver settings a counter collection could depend on (no root, nothing changed)
phase_r3sys() {
    local f=$RUN_DIR/r3sys.txt d
    sum ""
    sum "## r3sys (Q4: kernel version and amdgpu perf-counter related settings, read only)"
    {
        echo "# uname -r: $(uname -r)"
        echo "# /proc/sys/kernel/perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>&1)"
        echo "# /proc/sys/kernel/kptr_restrict: $(cat /proc/sys/kernel/kptr_restrict 2>&1)"
        echo "# power_dpm_force_performance_level: $(cat "$DRM_DEVICE/power_dpm_force_performance_level" 2>&1)"
        echo "# amdgpu module parameters (readable ones):"
        for d in /sys/module/amdgpu/parameters/*; do printf '%s=%s\n' "${d##*/}" "$(cat "$d" 2>/dev/null | tr '\n' ' ')"; done
        echo "# amdkfd module parameters:"
        for d in /sys/module/amdkfd/parameters/*; do printf '%s=%s\n' "${d##*/}" "$(cat "$d" 2>/dev/null | tr '\n' ' ')"; done
        echo "# kfd topology nodes (name, gfx_target_version, capability, debug_prop, num_cp_queues):"
        for d in /sys/class/kfd/kfd/topology/nodes/*; do
            echo "node ${d##*/}: $(grep -hE '^(gfx_target_version|capability|debug_prop|num_cp_queues|simd_count) ' "$d/properties" 2>/dev/null | tr '\n' ' ')"
        done
        echo "# card $DRM_DEVICE: perf-counter related files present:"
        ls "$DRM_DEVICE" | grep -iE 'perf|counter|pmu|spm|thread_trace|pm_info|gpu_metrics' | tr '\n' ' '
        echo
        echo "# perf events registered for amdgpu (PMU):"
        ls /sys/bus/event_source/devices 2>/dev/null | grep -iE 'amd|drm' | tr '\n' ' '
        echo
        echo "# rocprofv3 10.1: $(rocprofv3 --version 2>&1 | head -2 | tr '\n' ' ')"
        echo "# rocprofv3 10.0: $("$R3_VENV/bin/rocprofv3" --version 2>&1 | head -2 | tr '\n' ' ')"
        echo "# 10.0 package has librocprofv3-list-avail.so.1 but its rocprofv3-avail looks for librocprofv3-list-avail.so: $(ls "$R3_VLIB/rocprofiler-sdk/" | grep -c 'list-avail.so$') file(s) named exactly so"
    } >"$f" 2>&1
    sum "- written to r3sys.txt"
}

# r3counters (Q4): lists, then the 13 counter sets of round 2 against the same 256 MiB read kernel with the 10.1 profiler (control, same session) and the
# 10.0 profiler. 10.0 is run twice: "as is" (the app keeps its RUNPATH, the 10.1 runtime libraries) and "pure" (LD_LIBRARY_PATH on the 10.0 libraries).
# A hung profiler (it aborts and its signal handler waits) is stopped by the job's time limit and counts as a result.
r3_run_set() { # tag, profiler binary, env prefix words (or "-"), set
    local tag=$1 prof=$2 envp=$3 set=$4 n d
    R3_F0=$FAILED
    n=$(echo "$set" | tr ' ' '-' | cut -c1-40)
    d=$RUN_DIR/r3c-$tag-$n.d
    # shellcheck disable=SC2086
    if [[ $envp == - ]]; then
        gpu_job "r3c-$tag-$n" 70 25 "$prof" --pmc $set -f csv -d "$d" -o p -- "$BANDWIDTH" 256 3
    else
        gpu_job "r3c-$tag-$n" 70 25 env $envp "$prof" --pmc $set -f csv -d "$d" -o p -- "$BANDWIDTH" 256 3
    fi
    local rc=$?
    # the 10.0 profiler is EXPECTED to fail here (that is the finding): its job exit does not make the run FAILED
    if [[ $tag == r100* && $rc -ne 0 ]]; then FAILED=$R3_F0; fi
    if [[ -s $d/p_counter_collection.csv ]]; then
        "${PY[@]}" "$ROOT/scripts/llm_kernels.py" pmc "$d/p_counter_collection.csv" --kernel read_kernel --moves-data >"$RUN_DIR/r3c-$tag-$n.pmc.txt" 2>&1
        sum "  - [$tag] $set: $(awk '{printf "%s %s; ", $1, $NF}' "$RUN_DIR/r3c-$tag-$n.pmc.txt" | cut -c1-200)"
    else
        sum "  - [$tag] $set: NO counter file (job exit $rc; $(grep -m1 -E 'aqlprofile API table load failed|Error|abort' "$RUN_DIR/r3c-$tag-$n.log" | cut -c1-120))"
    fi
}
phase_r3counters() {
    local set
    sum ""
    sum "## r3counters (Q4: the 13 counter sets with ROCm 10.1 (control) and ROCm 10.0 rocprofv3)"
    gpu_job r3c-r101-listavail 70 120 rocprofv3 --list-avail
    gpu_job r3c-r101-avail-list 70 120 rocprofv3-avail list
    gpu_job r3c-r100pure-listavail 70 120 env LD_LIBRARY_PATH="$R3_VLIB" "$R3_VENV/bin/rocprofv3" --list-avail
    gpu_job r3c-r100pure-avail-list 70 120 env LD_LIBRARY_PATH="$R3_VLIB" "$R3_VENV/bin/rocprofv3-avail" list
    # the 10.0 profiler without counters: does a kernel trace work at all (so a counter failure is about counters)
    gpu_job r3c-r100pure-kerneltrace 70 90 env LD_LIBRARY_PATH="$R3_VLIB" "$R3_VENV/bin/rocprofv3" --kernel-trace -f csv -d "$RUN_DIR/r3c-r100pure-kerneltrace.d" -o t -- "$BANDWIDTH" 256 3
    gpu_job r3c-r101-kerneltrace 70 90 rocprofv3 --kernel-trace -f csv -d "$RUN_DIR/r3c-r101-kerneltrace.d" -o t -- "$BANDWIDTH" 256 3
    for set in "${R3_COUNTER_SETS[@]}"; do
        r3_run_set r101 rocprofv3 - "$set"
        r3_run_set r100pure "$R3_VENV/bin/rocprofv3" "LD_LIBRARY_PATH=$R3_VLIB" "$set"
    done
    # the 10.0 profiler on the 10.1 runtime libraries (the app's own RUNPATH): three sets only, each hangs until the time limit
    for set in "${R3_COUNTER_SETS[0]}" "${R3_COUNTER_SETS[1]}" "${R3_COUNTER_SETS[6]}"; do
        r3_run_set r100asis "$R3_VENV/bin/rocprofv3" - "$set"
    done
}

# ---------------------------------------------------------------------------------------------------------------
# Round 3 (CPU): build the new probes, check their ISA, and make every check fail once on purpose
R3_BUILD=$ROOT/build/gfx1201-rocm10
r3_hipcc() { # <source> <output binary> <save-temps dir> <log>
    mkdir -p "$3"
    (cd "$3" && systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' -I"$ROOT/src" \
        -save-temps "$1" -o "$2" -lamdhip64) >"$4" 2>&1
}
phase_r3build() {
    local d=$RUN_DIR/r3build rc o k
    sum ""
    sum "## r3build (probes of round 3: build, ISA checks, deliberate breaks; CPU only)"
    mkdir -p "$d" "$R3_BUILD"
    r3_hipcc "$ROOT/src/pair_overlap.hip" "$R3_BUILD/pair_overlap" "$d/pair" "$d/pair.build.log" || { sum "- pair_overlap build FAILED (r3build/pair.build.log)"; FAILED=1; return; }
    r3_hipcc "$ROOT/src/power_mix.hip" "$R3_BUILD/power_mix" "$d/mix" "$d/mix.build.log" || { sum "- power_mix build FAILED (r3build/mix.build.log)"; FAILED=1; return; }
    r3_hipcc "$ROOT/src/widen_cache.hip" "$R3_BUILD/widen_cache_nocal" "$d/nocal" "$d/nocal.build.log" || { sum "- widen_cache_nocal build FAILED"; FAILED=1; return; }
    sum "- built pair_overlap, power_mix, widen_cache_nocal in build/gfx1201-rocm10"
    "${PY[@]}" "$ROOT/scripts/llm3_pair_isacheck.py" "$d"/pair/*.lto.s >"$d/pair-isa.txt" 2>&1; rc=$?
    sum "- pair_overlap ISA check: $(tail -1 "$d/pair-isa.txt")"; ((rc == 0)) || FAILED=1
    for k in order count; do
        "${PY[@]}" "$ROOT/scripts/llm3_pair_isacheck.py" "$d"/pair/*.lto.s --break $k >"$d/pair-isa-break-$k.txt" 2>&1
        if (($? == 1)) && grep -q '^FAIL' "$d/pair-isa-break-$k.txt"; then sum "- pair_overlap ISA break '$k': FAILED as intended"; else sum "- pair_overlap ISA break '$k' NOT caught (FAILED)"; FAILED=1; fi
    done
    o=$(ls "$d"/mix/*.img.lto.o)
    for k in mix_fp8 mix_q8 mix_read mix_fp8row mix_q8row; do
        "$ROCM_PATH/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx1201 --disassemble-symbols="$("$ROCM_PATH/lib/llvm/bin/llvm-nm" --defined-only "$o" | awk -v k="$k" '$2 == "T" && $3 ~ ("^_Z[0-9]+" k "[A-Z]") {print $3}')" "$o" >"$d/$k.s" 2>&1
    done
    "${PY[@]}" "$ROOT/scripts/llm3_mix_isacheck.py" "$d/mix_fp8.s" "$d/mix_q8.s" "$d/mix_read.s" "$d/mix_fp8row.s" "$d/mix_q8row.s" >"$d/mix-isa.txt" 2>&1; rc=$?
    sum "- power_mix ISA check: $(tail -1 "$d/mix-isa.txt")"; ((rc == 0)) || FAILED=1
    "${PY[@]}" "$ROOT/scripts/llm3_mix_isacheck.py" "$d/mix_fp8.s" "$d/mix_q8.s" "$d/mix_read.s" "$d/mix_fp8row.s" "$d/mix_q8row.s" --break >"$d/mix-isa-break.txt" 2>&1
    if (($? == 1)) && grep -q '^FAIL' "$d/mix-isa-break.txt"; then sum "- power_mix ISA break (3 dots expected, 2 present): FAILED as intended"; else sum "- power_mix ISA break NOT caught (FAILED)"; FAILED=1; fi
    "${PY[@]}" "$ROOT/scripts/mem_loopcheck.py" "$d"/nocal/*.lto.s "$ROOT/scripts/mem-expect/widen_cache.txt" >"$d/nocal-loopcheck.txt" 2>&1
    sum "- widen_cache_nocal ISA check of the timed loops: $(tail -1 "$d/nocal-loopcheck.txt")"
}

# r3check (GPU): the deliberate breaks of the probes themselves must FAIL on the card (exit 1 and their VERDICT line)
phase_r3check() {
    local rc
    sum ""
    sum "## r3check (probe breaks on the card)"
    gpu_job r3check-pair-clock 70 60 "$R3_BUILD/pair_overlap" clock 0 0
    grep -q '^VERDICT clock-known-answer PASS' "$RUN_DIR/r3check-pair-clock.log" || { sum "- pair_overlap clock known answer not PASS (FAILED)"; FAILED=1; }
    gpu_job r3check-pair-ok 70 120 "$R3_BUILD/pair_overlap" plain 256 500
    grep -q '^VERDICT pair_overlap PASS' "$RUN_DIR/r3check-pair-ok.log" || { sum "- pair_overlap plain 256 500 not PASS (FAILED)"; FAILED=1; }
    local f0=$FAILED
    gpu_job r3check-pair-break 70 120 "$R3_BUILD/pair_overlap" plain 256 500 - break
    rc=$?
    if ((rc == 1)) && grep -q '^VERDICT data-check FAIL' "$RUN_DIR/r3check-pair-break.log"; then FAILED=$f0; sum "- pair_overlap break (B expects the wrong tag): data check FAILED as intended"; else sum "- pair_overlap break NOT caught (exit $rc) (FAILED)"; FAILED=1; fi
    for v in fp8 q8 fp8row q8row read; do
        gpu_job "r3check-mix-ok-$v" 70 120 "$R3_BUILD/power_mix" $v 2
        grep -q '^VERDICT power_mix PASS' "$RUN_DIR/r3check-mix-ok-$v.log" || { sum "- power_mix $v known answer not PASS (FAILED)"; FAILED=1; }
    done
    for v in fp8row q8row; do
        gpu_job "r3check-mix-ok-${v}R" 70 120 "$R3_BUILD/power_mix" $v 2 rand
        grep -q '^VERDICT result-known-answer PASS' "$RUN_DIR/r3check-mix-ok-${v}R.log" || { sum "- power_mix $v rand: CPU known answer not PASS (FAILED)"; FAILED=1; }
    done
    # the known answer must not depend on fresh VRAM being zero: the result buffer starts full of 1.0f
    for v in fp8row q8row; do
        gpu_job "r3check-mix-dirty-$v" 70 120 "$R3_BUILD/power_mix" $v 2 dirty
        grep -q '^VERDICT power_mix PASS' "$RUN_DIR/r3check-mix-dirty-$v.log" || { sum "- power_mix $v on a dirty result buffer not PASS (FAILED)"; FAILED=1; }
    done
    f0=$FAILED
    for v in fp8 q8row fp8rowR q8rowR; do
        if [[ $v == *R ]]; then gpu_job "r3check-mix-break-$v" 70 120 "$R3_BUILD/power_mix" "${v%R}" 2 break rand; else gpu_job "r3check-mix-break-$v" 70 120 "$R3_BUILD/power_mix" $v 2 break; fi
        rc=$?
        if ((rc == 1)) && grep -q '^VERDICT result-known-answer FAIL' "$RUN_DIR/r3check-mix-break-$v.log"; then FAILED=$f0; sum "- power_mix $v break (wrong expectation): known answer FAILED as intended"; else sum "- power_mix $v break NOT caught (exit $rc) (FAILED)"; FAILED=1; fi
    done
}

# r3fix (GPU, needs r3build): the dirty-buffer break. R3_FIX_BEFORE names a power_mix built from the source before the fix
# plus the dirty argument only; its q8row known answer must FAIL, and the fixed binary must PASS on the same dirty buffer
phase_r3fix() {
    local rc v f0
    sum ""
    sum "## r3fix (result buffer starts dirty)"
    if [[ -x ${R3_FIX_BEFORE:-} ]]; then
        f0=$FAILED
        gpu_job r3fix-before-q8row 70 120 "$R3_FIX_BEFORE" q8row 2 dirty
        rc=$?
        if ((rc == 1)) && grep -q '^VERDICT result-known-answer FAIL' "$RUN_DIR/r3fix-before-q8row.log"; then FAILED=$f0; sum "- power_mix before the fix, q8row on a dirty buffer: known answer FAILED as intended"; else sum "- the dirty-buffer break was NOT caught by the old code (exit $rc) (FAILED)"; FAILED=1; fi
    fi
    for v in fp8 q8 read fp8row q8row; do
        gpu_job "r3fix-after-$v" 70 120 "$R3_BUILD/power_mix" $v 2
        grep -q '^VERDICT power_mix PASS' "$RUN_DIR/r3fix-after-$v.log" || { sum "- power_mix $v known answer not PASS (FAILED)"; FAILED=1; }
    done
    for v in fp8row q8row; do
        gpu_job "r3fix-after-dirty-$v" 70 120 "$R3_BUILD/power_mix" $v 2 dirty
        grep -q '^VERDICT power_mix PASS' "$RUN_DIR/r3fix-after-dirty-$v.log" || { sum "- power_mix $v on a dirty buffer not PASS (FAILED)"; FAILED=1; }
    done
}

# r3q3 (Q3): old (round 2), new (fix round) and nocal widen_cache binaries, interleaved, order rotated per repetition, S = 25 MiB, 200 calls, through the gated runner
phase_r3q3() {
    local k i v order
    local -A bin=([old]="$ROOT/runs/20261118T165826Z-llm2fix/widen_cache-round2-binary" [new]="$R3_BUILD/widen_cache" [nocal]="$R3_BUILD/widen_cache_nocal")
    local -a names=(old new nocal)
    sum ""
    sum "## r3q3 (Q3: old, new and nocal widen_cache binaries interleaved, ${R3Q3_REPS:-8} repetitions each)"
    for v in old new nocal; do [[ -x ${bin[$v]} ]] || { sum "- no binary for $v: ${bin[$v]} (FAILED)"; FAILED=1; return; }; done
    for k in $(seq 1 "${R3Q3_REPS:-8}"); do
        for i in 0 1 2; do
            v=${names[$(((i + k - 1) % 3))]}
            if [[ $v == nocal ]]; then
                gpu_job "r3q3-$v-r$k" 70 120 "${bin[$v]}" 25 200 nocal
            else
                gpu_job "r3q3-$v-r$k" 70 120 "${bin[$v]}" 25 200
            fi
        done
    done
    "${PY[@]}" "$ROOT/scripts/llm3_q3.py" table "$RUN_DIR" >"$RUN_DIR/r3q3-table.txt" 2>&1 || { sum "- r3q3 table FAILED (r3q3-table.txt)"; FAILED=1; }
    sum "- table in r3q3-table.txt"
}

# r3pairs (Q5): in-kernel REALTIME stamps of A-B pairs against the profiler's stamps. Without the profiler the probe's own in-kernel numbers; with it a join.
phase_r3pairs() {
    local mode kib n k name prof d="$RUN_DIR/r3p"
    sum ""
    sum "## r3pairs (Q5: kernel pairs on two clocks)"
    mkdir -p "$d"
    [[ -x $R3_BUILD/pair_overlap ]] || { sum "- no $R3_BUILD/pair_overlap (r3build first) (FAILED)"; FAILED=1; return; }
    : >"$RUN_DIR/r3p-join.txt"
    # controls first: negative (host sync between kernels) and positive (two streams), both clocks
    for prof in noprof prof; do
        for mode in "hostsync 256 2000" "twostreams 256 1000"; do
            read -r m kib n <<<"$mode"
            name="r3p-$m-$kib-$prof-r1"
            r3_pair_job "$name" "$prof" "$m" "$kib" "$n"
        done
    done
    # the pairs: sizes of a few to tens of microseconds, plain launches and a graph, with and without the profiler; the profiler runs three times
    for k in 1 2 3; do
        for mode in plain graph; do
            for spec in "256 20000" "2048 5000" "8192 2000"; do
                read -r kib n <<<"$spec"
                for prof in prof noprof; do
                    [[ $prof == noprof && $k -gt 1 ]] && continue
                    r3_pair_job "r3p-$mode-$kib-$prof-r$k" "$prof" "$mode" "$kib" "$n"
                done
            done
        done
    done
}
r3_pair_job() { # name, prof|noprof, mode, KiB, pairs
    local name=$1 prof=$2 mode=$3 kib=$4 n=$5 d="$RUN_DIR/r3p"
    if [[ $prof == prof ]]; then
        gpu_job "$name" 70 600 rocprofv3 --kernel-trace -f csv -d "$d/$name.d" -o t -- "$R3_BUILD/pair_overlap" "$mode" "$kib" "$n" "$d/$name.dump.csv"
        if [[ -s $d/$name.dump.csv && -s $d/$name.d/t_kernel_trace.csv ]]; then
            "${PY[@]}" "$ROOT/scripts/llm3_pairs.py" join "$d/$name.dump.csv" "$d/$name.d/t_kernel_trace.csv" --label "$name" >>"$RUN_DIR/r3p-join.txt" 2>&1 || { sum "- $name: join FAILED (r3p-join.txt)"; FAILED=1; }
        else
            sum "- $name: no dump or no trace (FAILED)"; FAILED=1
        fi
    else
        gpu_job "$name" 70 600 "$R3_BUILD/pair_overlap" "$mode" "$kib" "$n" "$d/$name.dump.csv"
    fi
    grep -q '^VERDICT pair_overlap PASS' "$RUN_DIR/$name.log" 2>/dev/null || { sum "- $name: no 'VERDICT pair_overlap PASS' in its log (FAILED)"; FAILED=1; }
}

# r3power (Q1c): fp8, q8, read mixes, interleaved, 10 s each, with the 20 ms sampler; then scripts/llm3_power.py table
phase_r3power() {
    local k v
    sum ""
    sum "## r3power (Q1c: instruction mix at matched bandwidth, ${R3P_REPS:-6} repetitions each of ${R3P_VARIANTS:-fp8 q8 read fp8row q8row}, 10 s loops)"
    [[ -x $R3_BUILD/power_mix ]] || { sum "- no $R3_BUILD/power_mix (r3build first) (FAILED)"; FAILED=1; return; }
    mkdir -p "$RUN_DIR/r3m"
    gpu_job r3m-idle 60 60 "$SELF" __watch "$RUN_DIR/r3m/idle" idle sleep 8
    for k in $(seq 1 "${R3P_REPS:-6}"); do
        for v in ${R3P_VARIANTS:-fp8 q8 read fp8row q8row}; do
            # a trailing R in the variant name means random data (power_mix ... rand)
            if [[ $v == *R ]]; then
                gpu_job "r3m-$v-r$k" 60 300 "$SELF" __watch "$RUN_DIR/r3m/$v-r$k" "$v-r$k" "$R3_BUILD/power_mix" "${v%R}" 10 rand
            else
                gpu_job "r3m-$v-r$k" 60 300 "$SELF" __watch "$RUN_DIR/r3m/$v-r$k" "$v-r$k" "$R3_BUILD/power_mix" "$v" 10
            fi
            grep -q '^VERDICT power_mix PASS' "$RUN_DIR/r3m-$v-r$k.log" 2>/dev/null || { sum "- r3m-$v-r$k: no 'VERDICT power_mix PASS' (FAILED)"; FAILED=1; }
            cooldown 8
        done
    done
    "${PY[@]}" "$ROOT/scripts/llm3_power.py" table "$RUN_DIR" >"$RUN_DIR/r3m-table.txt" 2>&1 || { sum "- r3m table FAILED (r3m-table.txt)"; FAILED=1; }
    sum "- table in r3m-table.txt"
}

# r3cnt (Q1b): the counters that work, per decode kernel of the two real builds (8B, 16 generated tokens), kernel trace alongside
phase_r3cnt() {
    local c v bin mdl set n m
    sum ""
    sum "## r3cnt (Q1b: SQ_WAVES, SQ_BUSY_CYCLES, SIMD_UTILIZATION, GRBM on the real decode kernels)"
    for c in "fp8|$PATCHED|llama31-8b-fp8-outq8" "q8|$UNMOD|llama31-8b-q8_0"; do
        IFS='|' read -r v bin mdl <<<"$c"
        m=$MODELS/$mdl.gguf
        for set in "SQ_WAVES SQ_BUSY_CYCLES" "SIMD_UTILIZATION" "GRBM_COUNT GRBM_GUI_ACTIVE"; do
            n=$(echo "$set" | tr ' ' '-' | cut -c1-30)
            # shellcheck disable=SC2086
            gpu_job "r3k-$v-$n" 70 900 rocprofv3 --pmc $set -f csv -d "$RUN_DIR/r3k-$v-$n.d" -o p -- "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 16 -r 1 --no-warmup -o jsonl
        done
        # per kernel family: the fp8 matvec (patched), Q8_0 matvec with and without fusion
        if [[ $v == fp8 ]]; then
            "${PY[@]}" "$ROOT/scripts/llm3_cnt.py" table "$v-matvec_f" "$RUN_DIR"/r3k-$v-*.d/p_counter_collection.csv --kernel 'mul_mat_vec_f<ggml_fp8_e4m3_t, float, 1, 256, true' >>"$RUN_DIR/r3k-table.txt" 2>&1 || { sum "- r3k table fp8 FAILED"; FAILED=1; }
        else
            "${PY[@]}" "$ROOT/scripts/llm3_cnt.py" table "$v-matvec_q-fused" "$RUN_DIR"/r3k-$v-*.d/p_counter_collection.csv --kernel 'mul_mat_vec_q<\(ggml_type\)8, 1, true' >>"$RUN_DIR/r3k-table.txt" 2>&1 || { sum "- r3k table q8 fused FAILED"; FAILED=1; }
            "${PY[@]}" "$ROOT/scripts/llm3_cnt.py" table "$v-matvec_q-plain" "$RUN_DIR"/r3k-$v-*.d/p_counter_collection.csv --kernel 'mul_mat_vec_q<\(ggml_type\)8, 1, false' >>"$RUN_DIR/r3k-table.txt" 2>&1 || { sum "- r3k table q8 plain FAILED"; FAILED=1; }
        fi
    done
    sum "- table in r3k-table.txt"
}

# ---------------------------------------------------------------------------------------------------------------
# Round 4 (llm-runtimes.md): PyTorch with real FP8 matrix maths, vLLM, and llama.cpp on the LACT profile "Default" (the owner's overclock).
# These modes expect the profile "Default" (LLM_EXPECT_PROFILE, set by the mode name) and check its settings in every gate.
TVENV=${LLM4_TORCH_VENV:-$ROOT/toolchains/torch-rocm10-venv}   # the GPU PyTorch venv (ROCm 10.0); LLM4_TORCH_VENV=toolchains/vllm-rocm101-venv runs the same harness on torch 2.13 + ROCm 10.1
VVENV=$ROOT/toolchains/vllm-rocm101-venv
TPY=("$TVENV/bin/python" -I "$ROOT/scripts/llm4_torch.py")
VPY=(env "PYTHONPATH=$VVENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi" "$VVENV/bin/python" "$ROOT/scripts/llm4_vllm.py")
RH3=$MODELS/llama32-3b-redhat-fp8
META3=$MODELS/llama32-3b-instruct-16bit
RH8=$MODELS/Llama-3.1-8B-Instruct-FP8
TRUE_F32=$MODELS/llama32-3b-true-f32.gguf
TRUE_BF16=$MODELS/llama32-3b-true-bf16.gguf
R4_CONFIGS=("3b-fp8-unmod|$UNMOD|llama32-3b-fp8-outq8" "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8" "3b-q8_0|$UNMOD|llama32-3b-q8_0"
    "8b-fp8-unmod|$UNMOD|llama31-8b-fp8-outq8" "8b-fp8-patched|$PATCHED|llama31-8b-fp8-outq8" "8b-q8_0|$UNMOD|llama31-8b-q8_0" "3b-bf16-true|$UNMOD|llama32-3b-true-bf16")
R4_ONLY=${R4_ONLY:-}   # a substring: only jobs whose label contains it (for reruns)
r4_want() { [[ -z $R4_ONLY || $1 == *"$R4_ONLY"* ]]; }

phase_r4setup() {
    sum ""
    sum "## r4setup (no GPU: versions, models and the origin check)"
    {
        echo "# Round 4 record, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "expected profile: $EXPECT_PROFILE; lact current: $(lact_profile); $(lact cli profile get 2>&1 | head -1)"
        echo "card_config: $(card_config)"
        echo "snapshot: $(snap)"
        "$TVENV/bin/python" -I -c "import torch,transformers,safetensors;print('torch venv: torch',torch.__version__,'transformers',transformers.__version__,'safetensors',safetensors.__version__)" 2>&1 | tail -1
        "$VVENV/bin/python" -I -c "import torch;print('vllm venv: torch',torch.__version__)" 2>&1 | tail -1
        PYTHONPATH="$VVENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi" "$VVENV/bin/python" -c "import vllm;print('vllm',vllm.__version__)" 2>&1 | tail -1
        echo "rocprofv3 (torch venv, ROCm 10.0): $("$TVENV/bin/rocprofv3" --version 2>&1 | head -2 | tr '\n' ' ')"
        echo "rocprofv3 (vllm venv, ROCm 10.1): $("$VVENV/bin/rocprofv3" --version 2>&1 | head -2 | tr '\n' ' ')"
        echo "rocprofv3 (system ROCm 10.1): $(rocprofv3 --version 2>&1 | head -2 | tr '\n' ' ')"
        for f in "$RH3"/model.safetensors "$META3"/model-0000*.safetensors "$RH8"/model-0000*.safetensors "$TRUE_F32" "$TRUE_BF16"; do
            echo "$(stat -c %s "$f") $(sha256sum "$f" | cut -d' ' -f1) ${f##*/}"
        done
    } >"$RUN_DIR/r4-config.txt" 2>&1
    sum "- record in r4-config.txt (profile $(lact_profile), memory clock $RUN_UCLK MHz)"
    [[ $(lact_profile) == "$EXPECT_PROFILE" && -z $(profile_settings_wrong) ]] || { sum "- profile or settings differ from $EXPECT_PROFILE: stop (FAILED)"; FAILED=1; }
    if "$ROOT/toolchains/convert-venv/bin/python" -I "$ROOT/scripts/llm4_origin.py" compare "$RH3" "$MODELS/llama32-3b-fp8-outq8.gguf" >"$RUN_DIR/r4-origin-3b.txt" 2>&1; then
        sum "- origin: $(tail -1 "$RUN_DIR/r4-origin-3b.txt") (r4-origin-3b.txt)"
    else
        sum "- origin check: $(tail -1 "$RUN_DIR/r4-origin-3b.txt")"
    fi
}

# r4known: llama.cpp on this profile: pp128/512 + tg128, pp2048, parallel streams (both builds for FP8, Q8_0, and the 3B 16-bit true weights)
phase_r4known() {
    local c label bin mdl m
    sum ""
    sum "## r4known (llama.cpp on the profile $EXPECT_PROFILE: prompts, generation, streams)"
    for c in "${R4_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        r4_want "$label" || continue
        m=$MODELS/$mdl.gguf
        gpu_job "r4k-bench-$label" 60 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 128,512 -n 128 -r 5 -o jsonl
        gpu_job "r4k-pp2048-$label" 60 900 "$bin/llama-bench" -m "$m" -ngl 99 -p 2048 -n 0 -b 2048 -ub 2048 -r 5 -o jsonl
        if [[ $label != *fp8-unmod ]]; then
            gpu_job "r4k-batched-$label" 60 900 "$bin/llama-batched-bench" -m "$m" -ngl 99 -c 4096 -b 512 -ub 512 -npp 128 -ntg 128 -npl 1,2,3,4,5,6,7,8
        fi
        cooldown 15
    done
}

# r4torch: GEMM rates at the models' shapes, then prompt and generation speed through PyTorch (3B FP8, 3B bf16 of Meta's weights, 8B FP8)
phase_r4torch() {
    local spec label model flag
    sum ""
    sum "## r4torch (PyTorch eager: GEMMs, prompt, generation)"
    mkdir -p "$RUN_DIR/beh"
    r4_want gemm && for spec in "3b|$RH3" "8b|$RH8"; do
        IFS='|' read -r label model <<<"$spec"
        gpu_job "r4t-gemm-$label" 60 600 "$SELF" __watch "$RUN_DIR/beh/r4t-gemm-$label" "r4t-gemm-$label" "${TPY[@]}" gemm --model "$model" --reps 50 --ns 16 128 512 2048
        cooldown 20
    done
    for spec in "3b-fp8|$RH3|--fp8" "3b-bf16|$META3|--bf16" "8b-fp8|$RH8|--fp8"; do
        IFS='|' read -r label model flag <<<"$spec"
        r4_want "$label" || continue
        gpu_job "r4t-pp-$label" 60 900 "${TPY[@]}" bench --model "$model" "$flag" --text "$TEXT" --prefill 128 512 --streams --reps 7
        cooldown 20
        gpu_job "r4t-tg-$label-a" 60 900 "${TPY[@]}" bench --model "$model" "$flag" --text "$TEXT" --prefill --streams 1 2 3 4 --reps 3 --steps 128
        cooldown 25
        gpu_job "r4t-tg-$label-b" 60 900 "${TPY[@]}" bench --model "$model" "$flag" --text "$TEXT" --prefill --streams 5 6 7 8 --reps 3 --steps 128
        cooldown 25
    done
    # 2048-token prompts last and apart: the first torch prefill-2048 run ended in a GPU hang (see llm-runtimes.md)
    if [[ ${R4_PP2048:-} == 1 ]]; then
        for spec in "3b-fp8|$RH3|--fp8" "3b-bf16|$META3|--bf16" "8b-fp8|$RH8|--fp8"; do
            IFS='|' read -r label model flag <<<"$spec"
            r4_want "$label" || continue
            gpu_job "r4t-pp2048-$label" 60 900 "${TPY[@]}" bench --model "$model" "$flag" --text "$TEXT" --prefill 2048 --streams --reps 5
            cooldown 30
        done
    fi
}

# r4vllm: the same through vLLM (graphs, fused kernels), one engine start per model
phase_r4vllm() {
    local spec label model
    sum ""
    sum "## r4vllm (vLLM: prompt and generation)"
    for spec in "3b-fp8|$RH3" "3b-bf16|$META3" "8b-fp8|$RH8"; do
        IFS='|' read -r label model <<<"$spec"
        r4_want "$label" || continue
        local extra=()
        [[ $label == 8b-* ]] && extra=(--gpu-util 0.93 --max-len 2560)   # the 8B's weights leave no KV cache at 0.80 of the 16 GB
        gpu_job "r4v-$label" 60 1500 "${VPY[@]}" bench --model "$model" --text "$TEXT" --prefill 128 512 2048 --streams 1 2 3 4 5 6 7 8 --reps 3 --steps 128 "${extra[@]}"
        cooldown 30
    done
}

# r4q: quality. The reference is Meta's 16-bit 3B converted to F32 and run by llama.cpp (50 chunks, saved logits); every runtime is compared with it.
R4_KLD=$RUN_DIR/kld4
R4_KL_CHUNKS=${R4_KL_CHUNKS:-25}   # 25 chunks of 512: the 50-chunk F32 reference run hit the 95 C stop after 14 s (F32 GEMMs run hot)
phase_r4q() {
    local c label bin mdl spec model flag base
    sum ""
    sum "## r4q (perplexity and distance from the true weights)"
    mkdir -p "$R4_KLD"
    base=$R4_KLD/true-f32.bin
    for c in "${R4_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        r4_want "$label" || continue
        [[ ${R4_KL_ONLY:-} == 1 ]] && continue
        gpu_job "r4q-ppl4-$label" 60 900 "$bin/llama-perplexity" -m "$MODELS/$mdl.gguf" -f "$TEXT" -c 512 --chunks 4 -ngl 99
    done
    if [[ ${R4_KL_ONLY:-} != 1 ]] && r4_want true-f32; then
        gpu_job r4q-ppl4-3b-true-f32 60 1500 "$UNMOD/llama-perplexity" -m "$TRUE_F32" -f "$TEXT" -c 512 --chunks 4 -ngl 99 -lm none
    fi
    if [[ ! -s $base ]]; then
        gpu_job r4q-base-3b-true-f32 60 2400 "$UNMOD/llama-perplexity" -m "$TRUE_F32" -f "$TEXT" -c 512 --chunks "$R4_KL_CHUNKS" -ngl 99 -lm none --kl-divergence-base "$base"
    fi
    [[ -s $base ]] || { sum "- r4q: no reference logits file (FAILED)"; FAILED=1; return; }
    for c in "${R4_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        [[ $label == 3b-* ]] || continue
        r4_want "$label" || continue
        gpu_job "r4q-kld-$label" 60 1500 "$bin/llama-perplexity" -m "$MODELS/$mdl.gguf" -f "$TEXT" -c 512 --chunks "$R4_KL_CHUNKS" -ngl 99 --kl-divergence-base "$base" --kl-divergence
    done
    for spec in "3b-fp8|$RH3|--fp8" "3b-bf16|$META3|--bf16" "8b-fp8|$RH8|--fp8"; do
        IFS='|' read -r label model flag <<<"$spec"
        r4_want "$label" || continue
        # R4_SKIP_TORCH_FP8=1 leaves the PyTorch FP8 jobs for last: that path hung the GPU in decode and 2048-token prompts (llm-runtimes.md)
        if [[ $flag == --bf16 || ${R4_SKIP_TORCH_FP8:-} != 1 ]]; then
            [[ ${R4_KL_ONLY:-} == 1 ]] || gpu_job "r4q-t-ppl4-$label" 60 900 "${TPY[@]}" ppl --model "$model" "$flag" --text "$TEXT" --chunks 4
            [[ $label == 3b-* ]] && gpu_job "r4q-t-kl-$label" 60 1500 "${TPY[@]}" ppl --model "$model" "$flag" --text "$TEXT" --chunks "$R4_KL_CHUNKS" --ref "$base"
        fi
        [[ ${R4_KL_ONLY:-} == 1 ]] || gpu_job "r4q-v-ppl4-$label" 60 1500 "${VPY[@]}" ppl --model "$model" --text "$TEXT" --chunks 4
        [[ $label == 3b-* ]] && gpu_job "r4q-v-kl-$label" 60 1800 "${VPY[@]}" ppl --model "$model" --text "$TEXT" --chunks "$R4_KL_CHUNKS" --ref "$base"
    done
}
# r8x: the exact 8B comparison (round 4 follow-up, one owner-approved job under a 16 GB cap). The reference is the mixed F16/F32 file made from the
# 8B FP8 GGUF with scripts/llm3_mixed_ref.py (fp8 x scale, exact); the same text and 8 chunks of 512 as the 3B true-reference rows of round 4.
R8_SRC=$MODELS/llama31-8b-auto.gguf
R8_DIR=$RUN_DIR/r8x
R8_REF=$R8_DIR/llama31-8b-mixed-exact.gguf
R8_BASE=$R8_DIR/exact8b.bin
R8_CHUNKS=${R8_CHUNKS:-8}
R8_CONFIGS=("8b-fp8-unmod|$UNMOD|llama31-8b-fp8-outq8" "8b-fp8-patched|$PATCHED|llama31-8b-fp8-outq8" "8b-q8_0|$UNMOD|llama31-8b-q8_0")
phase_r8xbuild() { # (no GPU) build and bit-check the reference under the 10 GB cap, log size and sha256
    local py=$ROOT/toolchains/convert-venv/bin/python cap=(systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0) t0
    sum ""
    sum "## r8xbuild (no GPU: the mixed F16/F32 exact reference of the 8B)"
    mkdir -p "$R8_DIR"
    "${cap[@]}" true 2>/dev/null || { sum "- no memory cap available (FAILED)"; FAILED=1; return; }
    t0=$(date +%s)
    if ! nice -n 10 "${cap[@]}" "$py" -W ignore -I "$ROOT/scripts/llm3_mixed_ref.py" build "$R8_SRC" "$R8_REF" >"$R8_DIR/build.log" 2>&1; then
        sum "- build FAILED: $(tail -1 "$R8_DIR/build.log")"
        FAILED=1
        return
    fi
    sum "- build: $(tail -1 "$R8_DIR/build.log") in $(($(date +%s) - t0)) s (memory cap 10G)"
    if ! nice -n 10 "${cap[@]}" "$py" -W ignore -I "$ROOT/scripts/llm3_mixed_ref.py" compare "$R8_SRC" "$R8_REF" 16 >"$R8_DIR/compare.log" 2>&1; then
        sum "- bit check FAILED: $(tail -1 "$R8_DIR/compare.log")"
        FAILED=1
        return
    fi
    sum "- bit check: $(tail -1 "$R8_DIR/compare.log")"
    { echo "file: ${R8_REF##*/}"; echo "size: $(stat -c %s "$R8_REF")"; echo "sha256: $(sha256sum "$R8_REF" | cut -d' ' -f1)"; } >"$R8_DIR/ref-gguf.txt"
    sum "- reference file $(sed -n 's/^size: //p' "$R8_DIR/ref-gguf.txt") bytes, sha256 $(sed -n 's/^sha256: //p' "$R8_DIR/ref-gguf.txt")"
}
phase_r8xref() { # the one GPU job under the 16 GB cap; R8_NGL (layers on the card) must be given
    [[ ${R8_NGL:-} =~ ^[0-9]+$ ]] || { sum "- r8xref: set R8_NGL (the number of layers on the card) (FAILED)"; FAILED=1; return; }
    [[ -s $R8_REF ]] || { sum "- r8xref: no reference file $R8_REF (FAILED)"; FAILED=1; return; }
    sum ""
    sum "## r8xref (reference perplexity with saved logits, -ngl $R8_NGL, memory cap $BIG_CAP, $R8_CHUNKS chunks of 512)"
    gpu_job r8x-ref-8b-exact 60 "${R8_TIMEOUT:-3000}" "$UNMOD/llama-perplexity" -m "$R8_REF" -f "$TEXT" -c 512 --chunks "$R8_CHUNKS" -ngl "$R8_NGL" -lm mmap ${R8_EXTRA:-} --kl-divergence-base "$R8_BASE"
}
phase_r8xkld() { # the three comparisons at the normal 10 GB cap
    local c label bin mdl
    [[ -s $R8_BASE ]] || { sum "- r8xkld: no reference logits $R8_BASE (FAILED)"; FAILED=1; return; }
    sum ""
    sum "## r8xkld (KL divergence against the exact 8B reference)"
    for c in "${R8_CONFIGS[@]}"; do
        IFS='|' read -r label bin mdl <<<"$c"
        gpu_job "r8x-kld-$label" 60 1500 "$bin/llama-perplexity" -m "$MODELS/$mdl.gguf" -f "$TEXT" -c 512 --chunks "$R8_CHUNKS" -ngl 99 --kl-divergence-base "$R8_BASE" --kl-divergence
    done
}
phase_r8xfinish() { # sizes and sha256 first, then delete the reference GGUF and the logits
    sum ""
    sum "## r8xfinish"
    sum "$(ref_evidence "r8x exact 8B" "$R8_BASE" "$RUN_DIR/r8x-exact8b.base.txt")"
    if [[ -s $R8_DIR/ref-gguf.txt ]]; then
        rm -f "$R8_REF"
        sum "- reference GGUF deleted (size and sha256 in r8x/ref-gguf.txt)"
    fi
}
r4_finish_ref() { # evidence for the reference logits file, then delete it (run when every comparison is done)
    sum "$(ref_evidence "r4q 3b-true-f32" "$R4_KLD/true-f32.bin" "$RUN_DIR/r4q-true-f32.base.txt")"
}

# r4trace: kernel traces of PyTorch and vLLM (a marked region between two marker kernels), and of llama.cpp's 3B for the same profile
r4_report() { # name reps
    local csv
    csv=$(find "$RUN_DIR/$1.d" -name t_kernel_trace.csv 2>/dev/null | head -1)
    if [[ -z $csv ]] || ! "${PY[@]}" "$ROOT/scripts/llm4_trace.py" report "$csv" --reps "$2" --json "$RUN_DIR/$1.analysis.json" >"$RUN_DIR/$1.report.txt" 2>&1; then
        sum "  - $1: analysis FAILED ($(tail -1 "$RUN_DIR/$1.report.txt" 2>/dev/null))"
        FAILED=1
    else
        sum "  - $1: $(head -1 "$RUN_DIR/$1.report.txt" | cut -c1-200)"
    fi
}
phase_r4trace() {
    local spec label model flag kind n reps
    sum ""
    sum "## r4trace (rocprofv3 kernel traces between marker kernels)"
    for spec in "3b-fp8|$RH3|--fp8" "3b-bf16|$META3|--bf16" "8b-fp8|$RH8|--fp8"; do
        IFS='|' read -r label model flag <<<"$spec"
        r4_want "$label" || continue
        for kind in prefill decode; do
            if [[ $kind == prefill ]]; then reps=3; else reps=64; fi
            gpu_job "r4tr-t-$kind-$label" 60 900 env LLM4_NO_TRITON=1 "$TVENV/bin/rocprofv3" --kernel-trace -f csv -d "$RUN_DIR/r4tr-t-$kind-$label.d" -o t -- \
                "${TPY[@]}" trace --model "$model" "$flag" --text "$TEXT" --kind "$kind" --tokens 512 --reps "$reps"
            r4_report "r4tr-t-$kind-$label" "$reps"
            gpu_job "r4tr-v-$kind-$label" 60 1500 env "PYTHONPATH=$VVENV/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi" "$VVENV/bin/rocprofv3" --kernel-trace -f csv \
                -d "$RUN_DIR/r4tr-v-$kind-$label.d" -o t -- "$VVENV/bin/python" "$ROOT/scripts/llm4_vllm.py" trace --model "$model" --text "$TEXT" --kind "$kind" --tokens 512 --reps "$reps" --streams 1
            r4_report "r4tr-v-$kind-$label" "$reps"
            cooldown 20
        done
    done
}
# llama.cpp on this profile with the round 1 trace pipeline (3B only; the traces of round 1 were on the profile Testing)
# r4reanalyse: (no GPU) the trace analysis again over the traces already in the run folder
phase_r4reanalyse() {
    local d n reps
    sum ""
    sum "## r4reanalyse (no GPU)"
    for d in "$RUN_DIR"/r4tr-*.d; do
        n=${d##*/}
        n=${n%.d}
        if [[ $n == *prefill* ]]; then reps=3; else reps=64; fi
        r4_report "$n" "$reps"
    done
}
phase_r4llamatrace() {
    PROFILE_CONFIGS=("${CONFIGS_3B[@]}")
    phase_profile
}

# r4beh: clocks, power and hotspot (20 ms sampler) around loops of about 20 s for every runtime
phase_r4beh() {
    local spec label model flag kind c bin mdl m
    sum ""
    sum "## r4beh (sampler around loops of about 20 s; 60 s cool-down between)"
    mkdir -p "$RUN_DIR/beh"
    gpu_job r4b-idle 60 60 "$SELF" __watch "$RUN_DIR/beh/r4b-idle" r4b-idle sleep 10
    for kind in prefill decode; do
        # the PyTorch FP8 loops ended in a GPU hang (r4b-t-prefill-3b-fp8: busy 100 %, 3.58 GHz, 56 C, 108 W); R4_SKIP_TORCH_FP8=1 leaves them out
        [[ ${R4_SKIP_TORCH_FP8:-} == 1 ]] || gpu_job "r4b-t-$kind-3b-fp8" 60 300 "$SELF" __watch "$RUN_DIR/beh/r4b-t-$kind-3b-fp8" "r4b-t-$kind-3b-fp8" "${TPY[@]}" loop --model "$RH3" --fp8 --text "$TEXT" --kind "$kind" --tokens 512 --seconds 20
        cooldown 60
        gpu_job "r4b-v-$kind-3b-fp8" 60 600 "$SELF" __watch "$RUN_DIR/beh/r4b-v-$kind-3b-fp8" "r4b-v-$kind-3b-fp8" "${VPY[@]}" loop --model "$RH3" --text "$TEXT" --kind "$kind" --tokens 512 --seconds 20
        cooldown 60
    done
    for c in "3b-fp8-patched|$PATCHED|llama32-3b-fp8-outq8|350|25" "3b-q8_0|$UNMOD|llama32-3b-q8_0|450|23"; do
        IFS='|' read -r label bin mdl reps_pp reps_tg <<<"$c"
        m=$MODELS/$mdl.gguf
        gpu_job "r4b-l-pp-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/r4b-l-pp-$label" "r4b-l-pp-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 512 -n 0 -r "$reps_pp" -o jsonl
        cooldown 60
        gpu_job "r4b-l-tg-$label" 60 300 "$SELF" __watch "$RUN_DIR/beh/r4b-l-tg-$label" "r4b-l-tg-$label" "$bin/llama-bench" -m "$m" -ngl 99 -p 0 -n 128 -r "$reps_tg" -o jsonl
        cooldown 60
    done
}

phase=${1:-}
case $phase in
setup) phase_setup ;;
all)
    phase_setup
    phase_selftest
    ((FAILED)) && { sum "- selftest failed: no GPU work is started"; } || {
        phase_probe
        phase_known
        phase_shapes
        PROFILE_CONFIGS=("${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}")
        phase_profile
        GRID_CONFIGS=("${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}")
        phase_grid
        phase_quality
        phase_behaviour
    }
    ;;
selftest) phase_selftest ;;
selftest-guard) # only the thermal guard cases
    ST=$RUN_DIR/selftest
    mkdir -p "$ST"
    ST_FAIL=0
    st_guard_cases
    sum "- selftest-guard RESULT: $( ((ST_FAIL)) && echo FAILED || echo 'every guard check passed')"
    FAILED=$ST_FAIL ;;
probe) phase_probe ;;
profile) PROFILE_CONFIGS=("${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}"); phase_profile ;;
known) phase_known ;;
shapes) phase_shapes ;;
grid) GRID_CONFIGS=("${CONFIGS_3B[@]}" "${CONFIGS_8B[@]}"); phase_grid ;;
grid3b) GRID_CONFIGS=("${CONFIGS_3B[@]}"); phase_grid ;;
quality) phase_quality ;;
behaviour) phase_behaviour ;;
reanalyse) phase_reanalyse ;;
determinism) phase_determinism ;;
r2q) phase_r2q ;;
r3sys) phase_r3sys ;;
r3counters) phase_r3counters ;;
r3build) phase_r3build ;;
r3check) phase_r3check ;;
r3fix) phase_r3fix ;;
r3q3) phase_r3q3 ;;
r3pairs) phase_r3pairs ;;
r3power) phase_r3power ;;
r3cnt) phase_r3cnt ;;
r2build) phase_r2build ;;
r2shapes) phase_r2shapes ;;
r2shapes-trace) phase_r2shapes_trace ;;
r2counters) phase_r2counters ;;
r2mall) phase_r2mall ;;
r2mallbuild) phase_r2mallbuild ;;
r2mallcheck) phase_r2mallcheck ;;
r2libs) phase_r2libs ;;
r2overlap) phase_r2overlap ;;
r2ub) phase_r2ub ;;
r2clock) phase_r2clock ;;
r2streams) phase_r2streams ;;
builds) phase_builds ;;
a5) phase_a5 ;;
dev) shift; gpu_job "$@" ;;
r4dev) shift; gpu_job "$@" ;;
r4setup) phase_r4setup ;;
r4known) phase_r4known ;;
r4torch) phase_r4torch ;;
r4vllm) phase_r4vllm ;;
r4q) phase_r4q ;;
r4finishref) r4_finish_ref ;;
r8xbuild) phase_r8xbuild ;;
r8xref) phase_r8xref ;;
r8xkld) phase_r8xkld ;;
r8xfinish) phase_r8xfinish ;;
r4trace) phase_r4trace ;;
r4llamatrace) phase_r4llamatrace ;;
r4reanalyse) phase_r4reanalyse ;;
r4beh) phase_r4beh ;;
profile3b) PROFILE_CONFIGS=("${CONFIGS_3B[@]}"); phase_profile ;;
*) die "unknown mode '$phase' (see the usage text at the top)" ;;
esac
sum ""
sum "Run folder: $RUN_DIR"
((FAILED)) && sum "RESULT: something FAILED (see above)" || sum "RESULT: all checks passed"
cat "$SUMMARY"
exit "$FAILED"
