#!/usr/bin/env bash
# Home session: the steps of notes/home-session.md, one sub-command per step. Every step records the card with
# build/study/bin/watch_card (read-only: sysfs and /proc, it never opens the GPU) and writes into
# runs/<UTC time>-home/<step>/, adding one line per result to that folder's summary.md.
#
# Usage: scripts/study-home.sh <step> [options]
#   build                                   compile the recorder (g++ only)
#   check                                   read-only: card settings, LACT profiles, whether a step or a game is
#                                           running, who holds the card, memory, hotspot
#   config                                  read-only: the card, lock files, programs and folders this script uses
#   game <label> [--seconds 120] [--period-ms 2] [--upscaler [--log PATH]]
#                                           A: record while the owner plays; no GPU lock, no GPU load
#   upscaler <step folder> [--log PATH]     A: read the upscaler's timing log for one game recording (read-only)
#   short-tests [label]                     B: the short tests on the current settings (control)
#   power-mode [--mode NAME] [--profile NAME] --owner-ok=power-mode
#                                           B: the short tests, the driver's power mode changed (default
#                                           3D_FULL_SCREEN in the profile Home-Power), then back
#   perf-high --owner-ok=perf-high          B: the same with the performance level pinned high (profile Home-High)
#   heat normal | heat fan100 --owner-ok=fan100
#                                           C: 20 s matrix traces and 3 x 20 s light and stream loads
#   wall-meter                              C: idle, stream, FP16 matrix for 20 s each; the owner reads a wall meter
#   memclock --owner-ok=memclock            D: the memory clock request set by hand: read the level, stream test
#   overclock --owner-ok=overclock          D: scripts/study-overclock.sh end to end (it switches the profile)
#   oc-traces --owner-ok=oc-traces          D: 20 s traces on the overclock profile, stopped at 88 C
#
# Settings: this script never writes sysfs and never changes a setting by itself. A step that needs a setting changed
# stops and says what the owner must do, unless started with --owner-ok=<that step>. Then it checks that the LACT
# profile exists, switches with `lact cli profile set`, reads the intended setting back, and compares every other
# setting with those read at the start: a profile that differs in anything else is switched back at once and no load
# runs on it. "Testing" is put back on every exit the shell can act on (normal end, error, INT, TERM, HUP; signals
# wait while it switches back), tried twice, compared with the start settings, and the step exits 5 with
# "RESTORE FAILED" and the differing fields if they differ. It cannot switch back after SIGKILL (the running load and
# the recorder then get TERM from the kernel as their parent ends). The overclock step hands all of this to
# scripts/study-overclock.sh, unchanged.
# One step at a time: every step but check, config, build and upscaler holds a lock file of this checkout
# (runs/.study-home.lock) and a second copy refuses to start. A step that starts a GPU load refuses while a game
# runs (exit 7). The game is found from /proc only, never by its name (under Proton no process is named like the
# game): a card client named GameThread, a card client whose command line runs from steamapps/common, or any process
# whose command line holds steamapps/common or a Proton compatibilitytools.d path (either slash).
# GPU loads: the shared GPU lock (at most 60 s wait), nice, a time limit, at least 8 GB available memory, a start gate
# on the hotspot (below 60 C; 55 C before a trace; at most 3 minutes of waiting), and the recorder as temperature
# guard: it reads the hotspot and memory temperature every 2 ms and at the stop (hotspot 95 C, 88 C on the overclock
# profile; memory 90 C; or an unreadable temperature) sends TERM to the load and KILL to its process group 1 s later
# if it still runs. Any temperature stop, or a hotspot of 95 C read after a load, ends the step's remaining loads
# (the switch back still runs). The probes' own stops come on top (notes/home-session.md). The game step starts no
# load.
# Exit: 0 done, 1 a step stopped or a check failed, 2 usage or setup (wrong card, missing program), 3 the card or LACT
# is not as the step needs at the start (not on Testing, a profile missing), 5 RESTORE FAILED, 6 another step of
# the kit is running, 7 a game is running (load steps), 10 the step needs the owner's flag.
# Options read from the environment: CARD_ENV (the card record; default runs/latest/card.env; the card it resolves
# to is printed and must be PCI 1002:7550), HOME_VKGAME and HOME_VKSPV (the Vulkan probes).
# Test hooks, read only with HOME_SELFTEST=1 (scripts/study-home-selftest.sh), which prints a warning: HOME_RUN_DIR,
# HOME_GPU_LOCK, HOME_KIT_LOCK, HOME_MEMINFO, HOME_BIN_LAUNCH, HOME_BIN_TRACE, HOME_BIN_BW, HOME_STEAM_ROOT,
# HOME_SETTLE_S, HOME_SWITCH_WAIT_S, HOME_SPACING_S, HOME_UPS_TAIL_S, HOME_PROC, HOME_LOCAL_ENV. Without HOME_SELFTEST=1 a set hook is
# ignored and named.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 2

SELFTEST=0
[[ ${HOME_SELFTEST:-} == 1 ]] && SELFTEST=1
TEST_HOOKS=(HOME_RUN_DIR HOME_GPU_LOCK HOME_KIT_LOCK HOME_MEMINFO HOME_BIN_LAUNCH HOME_BIN_TRACE HOME_BIN_BW
    HOME_STEAM_ROOT HOME_SETTLE_S HOME_SWITCH_WAIT_S HOME_SPACING_S HOME_UPS_TAIL_S HOME_PROC HOME_LOCAL_ENV)
# hook <name> <default>: the test hook's value in self-test mode, else the default.
hook() {
    if ((SELFTEST)) && [[ -n ${!1:-} ]]; then
        printf '%s\n' "${!1}"
    else
        printf '%s\n' "$2"
    fi
}
if ((SELFTEST)); then
    echo "!!! SELF-TEST MODE (HOME_SELFTEST=1): test hooks are read. Never set HOME_SELFTEST for a real run. !!!" >&2
else
    ignored=""
    for v in "${TEST_HOOKS[@]}" HOME_PROFILE_POWER HOME_PROFILE_HIGH HOME_PROFILE_FAN HOME_PROFILE_MEM HOME_POWER_MODE; do
        [[ -n ${!v:-} ]] && ignored+=" $v"
    done
    [[ -n $ignored ]] && echo "study-home: ignored (test hooks, read only with HOME_SELFTEST=1, or no longer read):$ignored" >&2
fi

LOCAL_ENV=$(hook HOME_LOCAL_ENV "$REPO/local.env")
[[ -r $LOCAL_ENV ]] && . "$LOCAL_ENV"   # machine settings, git-ignored; see local.env.example
# What the upscaler log reader matches (local.env.example, under STUDY_UPSCALER_LOG): the stage names that tell the two
# models apart, the names printed for them, the settings key that starts a session, and the self-check tag.
UPS_STAGES_1=${STUDY_UPSCALER_STAGES_1:-exposure, input, network, output}
UPS_STAGES_2=${STUDY_UPSCALER_STAGES_2:-stage1, stage2, stage3, stage4}
UPS_LABEL_1=${STUDY_UPSCALER_LABEL_1:-1}
UPS_LABEL_2=${STUDY_UPSCALER_LABEL_2:-2}
UPS_MODEL_KEY=${STUDY_UPSCALER_MODEL_KEY:-model}
UPS_CHECK_TAG=${STUDY_UPSCALER_CHECK_TAG:-selfcheck}
GPU_LOCK=$(hook HOME_GPU_LOCK "${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}")
KIT_LOCK=$(hook HOME_KIT_LOCK "$REPO/runs/.study-home.lock")
MEMINFO=$(hook HOME_MEMINFO /proc/meminfo)
PROC=$(hook HOME_PROC /proc)
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH HIP_VISIBLE_DEVICES=0

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -r $CARD_ENV ]] || { echo "study-home: no card record at $CARD_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CARD_ENV"
[[ -n ${DRM_DEVICE:-} && -n ${HWMON:-} ]] || { echo "study-home: the card record lacks DRM_DEVICE or HWMON" >&2; exit 2; }
# HIP_VISIBLE_DEVICES=0 above selects the probes' device; the record must say that device is this card.
[[ ${HIP_INDEX:-} == 0 ]] || { echo "study-home: the card record gives HIP device index '${HIP_INDEX:-}', expected 0" >&2; exit 2; }
CARD_ID="$(cat "$DRM_DEVICE/vendor" 2>/dev/null):$(cat "$DRM_DEVICE/device" 2>/dev/null)"
if [[ $CARD_ID != 0x1002:0x7550 ]]; then
    echo "study-home: $DRM_DEVICE is PCI '$CARD_ID', not the RX 9070 XT (0x1002:0x7550); check the card record $CARD_ENV" >&2
    exit 2
fi
[[ $HWMON == "$DRM_DEVICE"/hwmon/* ]] || { echo "study-home: HWMON ($HWMON) is not a sensor folder of $DRM_DEVICE" >&2; exit 2; }
export DRM_DEVICE HWMON
echo "study-home: card ${DRM_DEVICE##*/} (PCI 1002:7550) from $CARD_ENV"

WATCH=$REPO/build/study/bin/watch_card
BIN_LAUNCH=$(hook HOME_BIN_LAUNCH "$REPO/build/sched_launch")
BIN_TRACE=$(hook HOME_BIN_TRACE "$REPO/build/sched_trace")
BIN_BW=$(hook HOME_BIN_BW "$REPO/build/gfx1201-rocm10/bandwidth")
SETTLE_S=$(hook HOME_SETTLE_S 3)
SWITCH_WAIT_S=$(hook HOME_SWITCH_WAIT_S 30)
SPACING_S=$(hook HOME_SPACING_S 60)
UPS_TAIL_S=$(hook HOME_UPS_TAIL_S 5)
# The Vulkan probes of the gaming round (dense FMA load, E1-E3). Not in this repository yet: give their paths.
VKGAME=${HOME_VKGAME:-}
VKSPV=${HOME_VKSPV:-}
VK_CU="groups=64,local=32,every=64,recstride=8"

# Profiles the owner makes in LACT, each a copy of "Testing" with one setting changed (notes/home-session.md).
PROFILE_POWER=Home-Power
POWER_MODE=3D_FULL_SCREEN
PROFILE_HIGH=Home-High
PROFILE_FAN=Home-Fan100
PROFILE_MEM=Home-Mem
OC_STOP_C=88
MEM_STOP_C=90

STEP=${1:-}
[[ -n $STEP ]] || { sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
shift
OWNER_OK=""
ARGS=()
for a in "$@"; do
    case $a in
    --owner-ok=*) OWNER_OK=${a#--owner-ok=} ;;
    *) ARGS+=("$a") ;;
    esac
done

if [[ $STEP == config ]]; then
    echo "self-test mode: $( ((SELFTEST)) && echo YES || echo no)"
    echo "card record: $CARD_ENV"
    echo "card: $DRM_DEVICE (PCI $CARD_ID), sensors $HWMON"
    echo "GPU lock: $GPU_LOCK"
    echo "kit lock: $KIT_LOCK"
    echo "memory figures: $MEMINFO"
    echo "processes: $PROC"
    echo "run folder: $(hook HOME_RUN_DIR "$REPO/runs/<UTC time>-home")"
    echo "probes: $BIN_LAUNCH $BIN_TRACE $BIN_BW; Vulkan: ${VKGAME:-not given} ${VKSPV:-}"
    echo "profiles: $PROFILE_POWER ($POWER_MODE by default), $PROFILE_HIGH, $PROFILE_FAN, $PROFILE_MEM, Default (overclock)"
    exit 0
fi

# ---------------------------------------------------------------------------------------------------------------
# One step at a time in this checkout: a non-blocking lock on fd 8, kept until the script ends. Children that may
# outlive the script get 8>&- so that they do not keep it.

kit_busy() { ! (exec 7>>"$KIT_LOCK" && flock -n 7) 2>/dev/null; }
case $STEP in
check | build | upscaler) ;;
*)
    exec 8>>"$KIT_LOCK" || { echo "study-home: cannot open the kit lock $KIT_LOCK" >&2; exit 2; }
    if ! flock -n 8; then
        echo "study-home: another step of this kit is running in this checkout (it holds $KIT_LOCK)." >&2
        echo "Let it finish (or end it in its own terminal with Ctrl-C), then start this step again." >&2
        exit 6
    fi
    ;;
esac

# ---------------------------------------------------------------------------------------------------------------
# Readers. Each prints a value or fails, so that no gate passes on an unreadable value.

read_int() {
    local v
    v=$(cat "$1" 2>/dev/null) || return 1
    [[ $v =~ ^[0-9]+$ ]] || return 1
    echo "$v"
}
hot() {
    local v
    v=$(read_int "$HWMON/temp2_input") || return 1
    echo $((v / 1000))
}
cap_w() {
    local v
    v=$(read_int "$HWMON/power1_cap") || return 1
    echo $((v / 1000000))
}
mem_gb() {
    local v
    v=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' "$MEMINFO" 2>/dev/null)
    [[ $v =~ ^[0-9]+$ ]] || return 1
    echo "$v"
}
profile_now() { timeout 20 lact cli profile get 2>/dev/null | tr -d '[:space:]'; }
profile_exists() { timeout 20 lact cli profile list 2>/dev/null | tr -d '\r' | sed 's/^ *//; s/ *$//' | grep -qxF -- "$1"; }
perf_level() { tr -d '[:space:]' <"$DRM_DEVICE/power_dpm_force_performance_level" 2>/dev/null; }
# The active power mode's name (the line whose name carries '*').
power_mode() { awk '/^ *[0-9]+ +[A-Z0-9_]+ *\*/ {gsub(/[*:]/, "", $2); print $2}' "$DRM_DEVICE/pp_power_profile_mode" 2>/dev/null; }
power_modes_offered() { awk '/^ *[0-9]+ +[A-Z0-9_]+ *\*? *:/ {gsub(/[*:]/, "", $2); printf "%s ", $2}' "$DRM_DEVICE/pp_power_profile_mode" 2>/dev/null; }
od_value() { awk -v k="$1" '$0 ~ "^"k":" {getline; gsub(/[^0-9-]/, ""); print; exit}' "$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null; }
od_mclk_max() { awk '/^OD_MCLK:/ {f=1; next} f && /^1:/ {gsub(/[^0-9]/, "", $2); print $2; exit}' "$DRM_DEVICE/pp_od_clk_voltage" 2>/dev/null; }
active_mclk() { awk '/\*/ {gsub(/[^0-9]/, "", $2); print $2}' "$DRM_DEVICE/pp_dpm_mclk" 2>/dev/null; }
# The memory clock that really runs, from gpu_metrics (current_uclk, offset 58). pp_dpm_mclk marks only its listed
# levels and marks none at an overclocked 1408 MHz (seen 2026-11-18).
uclk_now() {
    od -An -tu2 -j58 -N2 "$DRM_DEVICE/gpu_metrics" 2>/dev/null | tr -d ' '
}
# Fan settings. On this card (firmware fan control) LACT's fixed speed ("Static") writes a flat curve to
# gpu_od/fan_ctrl/fan_curve and turns zero RPM off; on cards without it LACT uses hwmon pwm1_enable=1 and pwm1. This
# card has no pwm1_enable (pwm1 is read-only), but both ways are read so that neither can stay changed unseen.
fan_curve() { awk '/^[0-9]+:/ {printf "%s%s", (n++ ? "," : ""), $2 $3}' "$DRM_DEVICE/gpu_od/fan_ctrl/fan_curve" 2>/dev/null; }
fan_od() { awk 'NR == 2 {print $1}' "$DRM_DEVICE/gpu_od/fan_ctrl/$1" 2>/dev/null; }
pwm_enable() {
    if [[ -e $HWMON/pwm1_enable ]]; then
        read_int "$HWMON/pwm1_enable" || echo unreadable
    else
        echo absent
    fi
}

# Every setting a step may change, as name=value fields on one line; compared before and after a switch.
settings() {
    local pe
    pe=$(pwm_enable)
    echo "profile=$(profile_now) cap_W=$(cap_w || echo ?) perf=$(perf_level) mode=$(power_mode)" \
        "sclk_offset=$(od_value OD_SCLK_OFFSET) vddgfx_offset=$(od_value OD_VDDGFX_OFFSET) mclk_request=$(od_mclk_max)" \
        "fan_curve=$(fan_curve) fan_zero_rpm=$(fan_od fan_zero_rpm_enable) fan_zero_rpm_stop_C=$(fan_od fan_zero_rpm_stop_temperature)" \
        "fan_target_C=$(fan_od fan_target_temperature) fan_min_pwm=$(fan_od fan_minimum_pwm)" \
        "fan_acoustic_limit=$(fan_od acoustic_limit_rpm_threshold) fan_acoustic_target=$(fan_od acoustic_target_rpm_threshold)" \
        "pwm1_enable=$pe$([[ $pe == 1 ]] && echo " pwm1=$(read_int "$HWMON/pwm1" || echo ?)")"
}
# diff_fields <settings A> <settings B>: the names of the fields that differ (a field on one side only differs too).
diff_fields() {
    awk -v a="$1" -v b="$2" '
        function parse(s, M,    n, i, f, k) {
            n = split(s, f, " ")
            for (i = 1; i <= n; i++) {
                k = f[i]; sub(/=.*/, "", k); M[k] = substr(f[i], length(k) + 2)
                if (!(k in seen)) { seen[k] = 1; order[++no] = k }
            }
        }
        BEGIN {
            parse(a, A); parse(b, B)
            for (i = 1; i <= no; i++) { k = order[i]; if (!(k in A) || !(k in B) || A[k] != B[k]) out = out (out == "" ? "" : " ") k }
            print out
        }'
}
# The "Testing" profile as notes/STUDY-PLAN.md describes it (stock: 317 W, offsets 0, memory request 1259 MHz, auto).
is_testing() {
    [[ $(profile_now) == Testing && $(cap_w) == 317 && $(perf_level) == auto && $(power_mode) == BOOTUP_DEFAULT &&
        $(od_value OD_SCLK_OFFSET) == 0 && $(od_value OD_VDDGFX_OFFSET) == 0 && $(od_mclk_max) == 1259 ]]
}

# ---------------------------------------------------------------------------------------------------------------
# Run folders

session_dir() {
    local d
    d=$(hook HOME_RUN_DIR "")
    [[ -n $d ]] && { echo "$d"; return; }
    # Reuse a home session folder started in the last 12 hours, so one sitting's steps share one summary.
    d=$(find "$REPO/runs" -maxdepth 1 -name '*-home' -type d -mmin -720 2>/dev/null | sort | tail -1)
    [[ -n $d ]] && { echo "$d"; return; }
    echo "$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-home"
}
SESSION=$(session_dir)
mkdir -p "$SESSION" || { echo "study-home: cannot create $SESSION" >&2; exit 2; }
SUM=$SESSION/summary.md
# The settings read on the stock Testing profile in this sitting (by check or a step's start): memclock compares with it.
TESTING_REC=$SESSION/testing-settings.txt
sum() { echo "$*" | tee -a "$SUM"; }
say() { echo "$*"; }
# step_dir <name>: a new folder for this step (name, name-2, name-3, ...).
step_dir() {
    local d=$SESSION/$1 k=2
    while [[ -e $d ]]; do
        d=$SESSION/$1-$k
        k=$((k + 1))
    done
    mkdir -p "$d" && echo "$d"
}

build_watch() {
    [[ -x $WATCH && $WATCH -nt $REPO/src/study/watch_card.cpp && $WATCH -nt $REPO/src/study/watch_card_sysfs.h ]] && return 0
    mkdir -p "$(dirname "$WATCH")"
    nice -n 10 g++ -std=c++20 -O2 -Wall -Wextra -Werror -o "$WATCH" "$REPO/src/study/watch_card.cpp"
}
# need_bins <program...>: every program the step will start must exist before any setting is touched.
need_bins() {
    local b
    for b in "$@"; do
        if [[ ! -x $b ]]; then
            echo "study-home: the program $b is missing or not executable; build it first. Nothing was changed." >&2
            sum "- $STEP: not run, $b is missing"
            exit 2
        fi
    done
}
# The Vulkan probes, if given: both paths must be usable.
need_vk() {
    [[ -z $VKGAME && -z $VKSPV ]] && return 0
    need_bins "$VKGAME"
    [[ -d $VKSPV ]] || { echo "study-home: HOME_VKSPV ($VKSPV) is not a folder" >&2; exit 2; }
}

# ---------------------------------------------------------------------------------------------------------------
# The recorder: rec_start <folder> <label> <period ms> [recorder options...]; rec_stop. Only this script's own child is
# signalled.

REC_PID=""
rec_start() {
    local dir=$1 label=$2 per=$3 n=0
    shift 3
    mkdir -p "$dir" && rm -f "$dir/ready"
    "$WATCH" record --out "$dir" --label "$label" --period-ms "$per" --die-with-parent "$@" >"$dir/recorder.out" 2>&1 8>&- 9<&- &
    REC_PID=$!
    # The recorder's start-up (rpm, lact and a scan of /proc) can take seconds: wait for its first sample on disk.
    until [[ -e $dir/ready ]]; do
        if ! kill -0 "$REC_PID" 2>/dev/null || ((n >= 600)); then
            return 1
        fi
        sleep 0.05
        n=$((n + 1))
    done
}
rec_stop() {
    [[ -n $REC_PID ]] || return 0
    kill -TERM "$REC_PID" 2>/dev/null
    wait "$REC_PID" 2>/dev/null
    REC_PID=""
}
# The lines of a recording's summary that answer most questions.
rec_brief() {
    grep -h -E '^(CLOCK_BANDS|AWAKE_SHARES|HIGH_RUNS|LOW_RUNS|HOTSPOT_C|MEM_C|THROTTLE indep|TIMING)' "$1/summary.txt" 2>/dev/null |
        sed 's/^/    /'
}

# ---------------------------------------------------------------------------------------------------------------
# Restore: put "Testing" back and compare the settings with those read at the start.

BASE_SETTINGS=""
RESTORED=1
# set_profile <name> <check function>: switch, then wait up to 30 s for the check, settle, and check again. With the
# lact call (at most 20 s) and the reads, one try can take about a minute; restore makes two.
set_profile() {
    local name=$1 check=$2 t=0
    timeout 20 lact cli profile set "$name" >>"$SESSION/lact.log" 2>&1 || return 1
    until "$check"; do
        ((t >= SWITCH_WAIT_S)) && return 1
        sleep 1
        t=$((t + 1))
    done
    sleep "$SETTLE_S"
    [[ $(profile_now) == "$name" ]] && "$check"
}
# shellcheck disable=SC2329  # called by name (set_profile)
back_to_base() { [[ $(settings) == "$BASE_SETTINGS" ]]; }
# unexpected_fields <allowed field...>: the fields that differ from the start settings and are not allowed to (the
# profile name always may).
unexpected_fields() {
    local f out="" allowed=" profile $* "
    for f in $(diff_fields "$BASE_SETTINGS" "$(settings)"); do
        [[ $allowed == *" $f "* ]] || out+="$f "
    done
    echo "${out% }"
}
# switch_to <profile> <check function> <allowed field...>: the switch, the intended setting, and nothing else changed.
# On failure it stops the run; the caller then switches back.
switch_to() {
    local prof=$1 check=$2 extra
    shift 2
    if ! set_profile "$prof" "$check"; then
        stop_run "$STEP: the switch to $prof did not show the intended setting; switching back, no load on it. Now: $(settings)"
        return 1
    fi
    extra=$(unexpected_fields "$@")
    if [[ -n $extra ]]; then
        stop_run "$STEP: $prof differs from the start settings in more than intended: $extra. Switching back, no load on it. Now: $(settings)"
        return 1
    fi
}
signals_on() {
    trap 'exit 130' INT TERM
    trap 'exit 129' HUP
}
restore() {
    ((RESTORED)) && return
    # A signal must not cut the switch back short.
    trap '' INT TERM HUP
    RESTORED=1
    rec_stop
    if set_profile Testing back_to_base || set_profile Testing back_to_base; then
        sum "- restored: $(settings)"
    else
        local now
        now=$(settings)
        sum "- RESTORE FAILED: the card is NOT confirmed back on its starting settings. Differs: $(diff_fields "$BASE_SETTINGS" "$now")"
        sum "  now:   $now"
        sum "  start: $BASE_SETTINGS"
        sum "  1. In LACT, choose the profile Testing FIRST. LACT remembers the chosen profile and applies it again at boot,"
        sum "     so a reboot while another profile is chosen brings that profile back (Default: +150 MHz, -50 mV, 374 W)."
        sum "  2. Check by hand: power limit 317 W, performance level Automatic, power profile mode BOOTUP_DEFAULT, no clock"
        sum "     or voltage offset, memory clock 1259 MHz (no maximum set), fan: your usual curve with Zero RPM on."
        sum "  3. Only then, if the driver still shows a difference, reboot. Afterwards run: scripts/study-home.sh check"
        exit 5
    fi
    signals_on
}
JOB_PID=""
# end_job <pid>: TERM (timeout passes it on to the probe), up to 1 s to end, then KILL to its process group (timeout
# leads its own group, so this reaches the probe and anything it started).
end_job() {
    local p=$1 i
    kill -TERM "$p" 2>/dev/null
    for ((i = 0; i < 20; i++)); do
        kill -0 "$p" 2>/dev/null || break
        sleep 0.05
    done
    if kill -0 "$p" 2>/dev/null; then
        kill -KILL -- "-$p" 2>/dev/null
        kill -KILL "$p" 2>/dev/null
    fi
    wait "$p" 2>/dev/null
}
# On any exit: end this script's own running load first, so that no load runs while the profile is switched back;
# then the recorder; then the switch back.
# shellcheck disable=SC2329  # called by the trap
on_exit() {
    if [[ -n $JOB_PID ]]; then
        end_job "$JOB_PID"
        JOB_PID=""
    fi
    rec_stop
    restore
}
trap on_exit EXIT
signals_on

# ---------------------------------------------------------------------------------------------------------------
# GPU jobs: the shared lock on fd 9 (hold_lock keeps it across several jobs).

HELD=0
take_lock() {
    [[ -r $GPU_LOCK ]] || return 1
    exec 9<"$GPU_LOCK" || return 1
    flock -w 60 9 && return 0
    exec 9<&-
    return 1
}
drop_lock() {
    flock -u 9 2>/dev/null
    exec 9<&-
}
hold_lock() { take_lock && HELD=1; }
release_hold() {
    ((HELD)) || return 0
    HELD=0
    drop_lock
}

STOP=0
stop_run() {
    STOP=1
    sum "- STOPPED: $*"
}
# job <folder> <name> <start below C> <stop at C> <command...>: one GPU load with the recorder running beside it as
# its temperature guard. JOB_LIMIT_S (default 240) is the load's time limit.
job() {
    local dir=$1 name=$2 start_c=$3 stop_c=$4 log rc t m w=0 try jp ended told=0 limit=${JOB_LIMIT_S:-240}
    shift 4
    log=$dir/$name.log
    ((STOP)) && { echo "SKIPPED (run stopped): $*" >>"$log"; return 9; }
    for try in 1 2 3; do
        m=$(mem_gb) || { stop_run "available memory unreadable before $name"; return 5; }
        if ((m < 8)); then
            echo "NOT STARTED: below 8 GB available memory" >>"$log"
            stop_run "below 8 GB available memory before $name"
            return 5
        fi
        while :; do
            t=$(hot) || { stop_run "hotspot unreadable before $name"; return 7; }
            ((t < start_c)) && break
            if ((!told)); then
                say "waiting for the hotspot to fall below $start_c C before $name (now $t C; at most 3 minutes)"
                told=1
            fi
            if ((w >= 180)); then
                echo "NOT STARTED: hotspot still $t C after 3 minutes" >>"$log"
                stop_run "the hotspot stayed at $t C or more for 3 minutes before $name (it starts below $start_c C)." \
                    "The step ends here and runs no further load. Let the card cool, check that the fan turns, then start the step again."
                return 7
            fi
            sleep 2
            w=$((w + 2))
        done
        ((HELD)) && break
        if ! take_lock; then
            echo "NOT STARTED: the shared lock was not free within 60 s" >>"$log"
            stop_run "the shared lock was not free within 60 s before $name"
            return 6
        fi
        m=$(mem_gb) && t=$(hot) && ((m >= 8 && t < start_c)) && break
        drop_lock
        if ((try == 3)); then
            stop_run "memory or hotspot not in range with the lock held before $name, three tries"
            return 8
        fi
    done
    {
        echo "## $(date -u +%FT%TZ) $name: $*"
        echo "before: $(settings); hotspot $(hot) C"
    } >>"$log"
    rm -f "$dir/$name.pid"
    if ! rec_start "$dir/rec-$name" "$name" 2 --guard-pidfile "$dir/$name.pid" --stop-hot "$stop_c" \
        --stop-mem "$MEM_STOP_C" --guard-log "$log"; then
        rec_stop
        ((HELD)) || drop_lock
        stop_run "the recorder (the temperature guard) did not start before $name; no load run"
        return 7
    fi
    # setpriv: a TERM to the load when this script ends without its traps (SIGKILL), so that no load outlives it.
    HOME_JOB=$name HOME_REC_DIR=$dir/rec-$name nice -n 10 setpriv --pdeathsig TERM \
        timeout --signal=TERM --kill-after=20 "$limit" "$@" >>"$log" 2>&1 8>&- 9<&- &
    jp=$!
    JOB_PID=$jp
    echo "$jp" >"$dir/$name.pid.tmp" && mv -f "$dir/$name.pid.tmp" "$dir/$name.pid"
    # A prompt that must appear only once the load really runs (the wall-meter readings).
    [[ -n ${HOME_PROMPT:-} ]] && say "$HOME_PROMPT"
    ended=""
    wait -n -p ended "$jp" "$REC_PID"
    rc=$?
    if [[ $ended != "$jp" ]]; then
        # The recorder ended first: no temperature guard is left, so the load ends too.
        REC_PID=""
        end_job "$jp"
        rc=1
        stop_run "the recorder (the temperature guard) ended during $name; the load was ended"
    fi
    JOB_PID=""
    # Anything left in the load's process group (a child that outlived its wrapper or ignored TERM) is ended before
    # the script goes on, so that no load runs while a profile is switched.
    kill -KILL -- "-$jp" 2>/dev/null
    sleep 0.5
    rec_stop
    ((HELD)) || drop_lock
    if t=$(hot); then
        echo "after: exit $rc; hotspot $t C" >>"$log"
        ((t >= 95)) && stop_run "hotspot $t C after $name"
    else
        echo "after: exit $rc; hotspot unreadable" >>"$log"
        stop_run "hotspot unreadable after $name"
    fi
    if grep -q '^THERMAL STOP' "$log"; then
        stop_run "$name: $(grep -m1 '^THERMAL STOP' "$log" | cut -c15-). No further load in this step."
        grep -q '^THERMAL STOP: the load was still running' "$log" && sum "  - $name: it was still running 1 s after TERM; KILL was sent"
    fi
    return "$rc"
}
# One line per job for the summary: the probe's own result lines and the recorder's clock lines.
job_line() {
    local dir=$1 name=$2
    sum "  - $name: $(grep -h -E '^(RAMP|RESULT|VERDICT|WINDOW)' "$dir/$name.log" 2>/dev/null | head -4 | cut -c1-200 | tr '\n' ' ')"
    rec_brief "$dir/rec-$name" | tee -a "$SUM" >/dev/null
}

# ---------------------------------------------------------------------------------------------------------------
# The owner's flag

# needs_owner <step> <what to do, one instruction per argument>: prints and exits 10 unless the flag names this step.
needs_owner() {
    local s=$1 line
    shift
    [[ $OWNER_OK == "$s" ]] && return 0
    say "This step changes a card setting, so it does not run without you."
    say "What to do:"
    for line in "$@"; do
        say "  - $line"
    done
    say "Then start it with:  scripts/study-home.sh $STEP ${ARGS[*]:+${ARGS[*]} }--owner-ok=$s"
    sum "- $STEP: not run, it needs the owner's flag --owner-ok=$s"
    exit 10
}
# need_profile <name>: the LACT profile must exist before anything runs.
need_profile() {
    profile_exists "$1" && return 0
    sum "- $STEP: not run: LACT has no profile named '$1' (lact cli profile list: $(timeout 20 lact cli profile list 2>/dev/null | tr '\n' ' ')). Make it first (notes/home-session.md)."
    exit 3
}
# The start check of every step that changes a setting: the card must be on "Testing" as the study describes it.
start_on_testing() {
    if ! is_testing; then
        sum "- $STEP: the card is not on the stock Testing settings at the start; nothing run: $(settings)"
        exit 3
    fi
    BASE_SETTINGS=$(settings)
    echo "$BASE_SETTINGS" >"$1/settings-start.txt"
    echo "$BASE_SETTINGS" >"$TESTING_REC"
}

# ---------------------------------------------------------------------------------------------------------------
# B: the short tests

short_tests() {
    local dir=$1 tag=$2
    job "$dir" "$tag-hip-ramp" 60 95 "$BIN_LAUNCH" ramp --idle-ms 1000 --ms 100 --card
    job_line "$dir" "$tag-hip-ramp"
    if [[ -n $VKGAME ]]; then
        VKG_OUT=$dir VKG_SPV=$VKSPV VKG_TAG=$tag-vk-ramp job "$dir" "$tag-vk-ramp" 60 95 \
            "$VKGAME" burst "name=$tag-vk-ramp,q=g,$VK_CU,cont=60,idle=1"
        job_line "$dir" "$tag-vk-ramp"
        VKG_OUT=$dir VKG_SPV=$VKSPV VKG_TAG=$tag-vk-burst60 job "$dir" "$tag-vk-burst60" 60 95 \
            "$VKGAME" burst "name=$tag-vk-burst60,q=g,$VK_CU,blen=2,cad=16.7,dur=1,idle=1"
        job_line "$dir" "$tag-vk-burst60"
    else
        sum "  - $tag: the Vulkan burst and ramp tests were skipped: they are not in the repository yet. Set HOME_VKGAME to"
        sum "    the vkgame program and HOME_VKSPV to its shader folder (see notes/home-session.md) to include them."
    fi
}

# A setting step of part B: control on Testing, the changed setting, control on Testing again.
# b_setting <step> <profile> <check function> <what the owner sets> <field the profile may change...>
b_setting() {
    local s=$1 prof=$2 check=$3 what=$4 dir
    shift 4
    needs_owner "$s" "In LACT, make a new profile named $prof as a copy of Testing, and change only this: $what." \
        "Choose the Testing profile again in LACT. The script switches to $prof with: lact cli profile set $prof" \
        "and switches back to Testing at the end."
    need_bins "$BIN_LAUNCH"
    need_vk
    need_profile "$prof"
    dir=$(step_dir "$s")
    start_on_testing "$dir"
    sum "- $s ($(date -u +%FT%TZ)): control on Testing, then $prof ($what), then Testing again"
    short_tests "$dir" "A-testing"
    ((STOP)) && return 1
    hold_lock || { stop_run "the shared lock was not free within 60 s before the switch to $prof"; return 1; }
    RESTORED=0
    if switch_to "$prof" "$check" "$@"; then
        settings >"$dir/settings-$prof.txt"
        sum "  - switched: $(settings)"
        short_tests "$dir" "B-$prof"
    fi
    restore
    release_hold
    ((STOP)) || short_tests "$dir" "C-testing"
    ((STOP)) && return 1
    return 0
}
# LACT needs the performance level "manual" to set a power mode, so either level is accepted here.
# shellcheck disable=SC2329  # called by name (set_profile)
check_power_mode() {
    local p
    p=$(perf_level)
    [[ $(power_mode) == "$POWER_MODE" && ($p == auto || $p == manual) ]]
}
# shellcheck disable=SC2329  # called by name (set_profile)
check_perf_high() { [[ $(perf_level) == high ]]; }

# ---------------------------------------------------------------------------------------------------------------
# C: heat, power, long loads

LAST_TRACE=0
# Loads start at least 60 s after the previous one ends (the study's trace rule).
space_loads() {
    local wait_s=$((LAST_TRACE + SPACING_S - SECONDS))
    ((LAST_TRACE > 0 && wait_s > 0)) && { say "waiting $wait_s s between loads"; sleep "$wait_s"; }
}
# trace <dir> <name> <load> <seconds> <stop C> [cool-down s, default 10]: one sched_trace run, below 55 C.
trace() {
    local dir=$1 name=$2 load=$3 secs=$4 stop_c=$5 post=${6:-10}
    ((STOP)) && return 9
    space_loads
    job "$dir" "$name" 55 "$stop_c" "$BIN_TRACE" "$load" --max-s "$secs" --pre-s 1 --post-s "$post" --out "$dir/$name"
    LAST_TRACE=$SECONDS
    job_line "$dir" "$name"
}
# long_load <dir> <name> <load> <stop C>: three 20 s sched_trace runs one after the other (the tool stops a load at
# 20 s), in one job and one recording. Between the runs each new process starts the device and runs a short
# calibration load; the stream load also waits until the hotspot is below 60 C (sched_trace's own start gate, at
# most 180 s each, hence the longer time limit). The part start and end times go to the log and the summary.
long_load() {
    local dir=$1 name=$2 load=$3 stop_c=$4 k parts waits
    ((STOP)) && return 9
    space_loads
    # shellcheck disable=SC2016  # expanded by the inner bash
    JOB_LIMIT_S=720 job "$dir" "$name" 55 "$stop_c" env LC_ALL=C bash -c '
        for k in 1 2 3; do
            echo "PART $k start $EPOCHREALTIME"
            "$1" "$2" --max-s 20 --pre-s 0 --post-s 0 --out "$3-$k"
            rc=$?
            echo "PART $k end $EPOCHREALTIME exit $rc"
            ((rc == 0)) || exit "$rc"
        done' _ "$BIN_TRACE" "$load" "$dir/$name"
    LAST_TRACE=$SECONDS
    job_line "$dir" "$name"
    for k in 1 2 3; do
        [[ -f $dir/$name-$k-sysfs.csv ]] || sum "  - $name: part $k did not run"
    done
    # Each part's length beyond its 20 s of load is the time the card was not under that load between the parts.
    parts=$(awk '$1 == "PART" && $3 == "start" {s[$2] = $4} $1 == "PART" && $3 == "end" {d = d sprintf("%s%.1f s", (d == "" ? "" : ", "), $4 - s[$2])}
        END {print "the parts took " (d == "" ? "none" : d) " (20 s of load each; the rest is start-up, calibration and any wait)"}' "$dir/$name.log")
    waits=$(grep -h '^THERMAL gate: waited' "$dir/$name.log" 2>/dev/null | tr '\n' ' ')
    sum "  - $name: $parts; sched_trace's wait for the hotspot: ${waits:-none}"
}
# shellcheck disable=SC2329  # called by name (set_profile)
check_fan100() {
    local c
    c=$(fan_curve)
    [[ $c =~ ^[0-9]+C100%(,[0-9]+C100%)*$ || $(fan_od fan_minimum_pwm) == 100 ||
        ($(pwm_enable) == 1 && $(read_int "$HWMON/pwm1") == 255) ]]
}
heat_loads() {
    local dir=$1 tag=$2
    trace "$dir" "$tag-f16" f16 20 95
    trace "$dir" "$tag-fp8" fp8 20 95
    long_load "$dir" "$tag-light60" light 95
    long_load "$dir" "$tag-stream60" stream 95
}

# ---------------------------------------------------------------------------------------------------------------
# The upscaler's timing log (part A, second half): read-only, one file.

steam_log_path() {
    local root lib vdf inst roots
    roots=$(hook HOME_STEAM_ROOT "$HOME/.local/share/Steam $HOME/.steam/steam")
    for root in $roots; do
        vdf=$root/steamapps/libraryfolders.vdf
        [[ -r $vdf ]] || continue
        # The library whose "apps" list holds Cyberpunk 2077 (app 1091500).
        lib=$(awk '/"path"/ {gsub(/"/, "", $2); p=$2} /"1091500"/ {print p; exit}' "$vdf")
        [[ -n $lib ]] || continue
        inst=$(awk -F'"' '/"installdir"/ {print $4; exit}' "$lib/steamapps/appmanifest_1091500.acf" 2>/dev/null)
        [[ -n $inst ]] || continue
        [[ -n ${STUDY_UPSCALER_LOG:-} ]] || return 1
        echo "$lib/steamapps/common/$inst/$STUDY_UPSCALER_LOG"
        return 0
    done
    return 1
}
log_size() { stat -c %s "$1" 2>/dev/null || echo -1; }

# upscaler_read <step folder> [log path]. Returns 0 with a result, 1 without one (no timing line in the recording),
# 2 for a missing folder.
upscaler_read() {
    local dir=$1 log=${2:-} s0="" s1="" t0 t1 out rc
    [[ -d $dir ]] || { echo "study-home: no folder $dir" >&2; return 2; }
    if [[ -z $log && -r $dir/upscaler-log.txt ]]; then
        log=$(sed -n 's/^path: //p' "$dir/upscaler-log.txt")
    fi
    [[ -z $log ]] && log=$(steam_log_path)
    [[ -n $log && -r $log ]] || { sum "- upscaler: the log file was not found (${log:-no path}); give it with --log PATH"; return 1; }
    if [[ -r $dir/upscaler-log.txt ]]; then
        s0=$(sed -n 's/^size_start: //p' "$dir/upscaler-log.txt")
        s1=$(sed -n 's/^size_end: //p' "$dir/upscaler-log.txt")
    fi
    # Local time of day of the recording, for logs whose lines carry [HH:MM:SS.mmm]; a line written up to 5 s after
    # the end still counts (a line covers the last 300 frames, about 5 s at 60 frames per second).
    t0=$(sed -n 's/^start_utc: //p' "$dir/header.txt" 2>/dev/null)
    t1=$(sed -n 's/^end_utc: //p' "$dir/header.txt" 2>/dev/null)
    [[ -n $t0 ]] && t0=$(date -d "$t0" +%H:%M:%S)
    [[ -n $t1 ]] && t1=$(date -d "$t1 5 seconds" +%H:%M:%S)
    out=$dir/upscaler-read.txt
    LC_ALL=C gawk -v s0="${s0:--1}" -v s1="${s1:--1}" -v t0="${t0:-}" -v t1="${t1:-}" \
        -v st1="$UPS_STAGES_1" -v st2="$UPS_STAGES_2" -v lab1="$UPS_LABEL_1" -v lab2="$UPS_LABEL_2" \
        -v mkey="$UPS_MODEL_KEY" -v ctag="$UPS_CHECK_TAG" '
        function tod(l,    mm) { return match(l, /^\[([0-9][0-9]:[0-9][0-9]:[0-9][0-9])/, mm) ? mm[1] : "" }
        function newsession() { ns++; smx[ns] = ""; nt[ns] = 0; cnt[ns] = 0 }
        # The model from the stage names of a timing line (the two models print different stage names: settings
        # STUDY_UPSCALER_STAGES_1 and _2). Returns 1, 2 or the text "unknown (stages ...)".
        function model_of(st,    n) {
            n = st; gsub(/ [0-9.]+/, "", n)
            return n == st1 ? 1 : n == st2 ? 2 : "unknown (stages " n ")"
        }
        function label(m) { return m == 1 ? lab1 : m == 2 ? lab2 : m }
        # Whether the self-check tag stands in the line as a word of its own (followed by a space or the line end).
        function has_tag(l, t,    p, c) { p = index(l, t); if (p == 0) return 0; c = substr(l, p + length(t), 1); return (c == " " || c == "") }
        function pair(l, re,    q) { return match(l, re, q) ? q[1] " (max " q[2] ")" : "n/a" }
        BEGIN { ns = 0; newsession(); off = 0 }
        {
            start = off; off += length($0) + 1   # byte offset of this line (the log is ASCII; the CR is counted)
            sub(/\r$/, "")
            if (cnt[ns] > 0 && ($0 ~ /==== session start/ || (index($0, "settings: " mkey "=") > 0 && nt[ns] > 0))) newsession()
            cnt[ns]++
            if ($0 ~ /network kernels:/) {
                if (has_tag($0, ctag)) smx[ns] = ctag " (self-check passed)"
                else if (index($0, ctag "(unchecked)")) smx[ns] = ctag "(unchecked): the self-check did not run"
                else smx[ns] = "no " ctag " in the network kernels line"
            }
            if (match($0, /timing, last ([0-9]+) dispatches: network GPU ([0-9.]+) \(max ([0-9.]+)\) ms \[([^]]*)\] \(([0-9]+) frames\)/, m)) {
                k = ++nt[ns]
                T[ns, k] = tod($0); M[ns, k] = m[2]; X[ns, k] = m[3]; S[ns, k] = m[4]; F[ns, k] = m[5]
                MOD[ns, k] = model_of(m[4])
                TOT[ns, k] = pair($0, "TOTAL\\(t3-t0\\) ([0-9.]+) \\(max ([0-9.]+)\\) ms")
                RES[ns, k] = pair($0, "input pass end -> resume ([0-9.]+) \\(max ([0-9.]+)\\) ms")
                if (match($0, /> 0\.1 ms in ([0-9]+) of ([0-9]+) frames/, q)) RES[ns, k] = RES[ns, k] ", > 0.1 ms in " q[1] " of " q[2] " frames"
                CPU[ns, k] = pair($0, "CPU dispatch ([0-9.]+) \\(max ([0-9.]+)\\) ms")
                inwin[ns, k] = (s0 >= 0 && s1 > s0 && start >= s0 && start < s1)
                intime[ns, k] = (t0 != "" && T[ns, k] != "" && (t0 <= t1 ? (T[ns, k] >= t0 && T[ns, k] <= t1) : (T[ns, k] >= t0 || T[ns, k] <= t1)))
            }
        }
        END {
            # Which lines: appended to the log during the recording; else written during it by time of day. Lines of
            # any other time are never taken as the result of this recording.
            how = ""
            for (s = ns; s >= 1 && how == ""; s--) for (k = 1; k <= nt[s]; k++) if (inwin[s, k]) { how = "appended to the log during the recording"; pick = s; key = "w" }
            for (s = ns; s >= 1 && how == ""; s--) for (k = 1; k <= nt[s]; k++) if (intime[s, k]) { how = "written during the recording by time of day (the log has no date: a line of another day at the same time would match too)"; pick = s; key = "t" }
            if (how == "") {
                last = ""
                for (s = ns; s >= 1 && last == ""; s--) if (nt[s] > 0) last = T[s, nt[s]] " (session " s " of " ns ")"
                print "UPSCALER NO_LINES no timing lines in this recording" (last == "" ? " (none in the whole log: is the timing log switched on in the launch options?)" : "; the newest timing line in the log is from " last ", outside the recording, and is not used")
                exit 3
            }
            print "UPSCALER session " pick " of " ns "; lines " how
            n = 0
            for (k = 1; k <= nt[pick]; k++) {
                if (key == "w" && !inwin[pick, k]) continue
                if (key == "t" && !intime[pick, k]) continue
                L[++n] = k
            }
            print "UPSCALER dropped the first timing line of the recording (" T[pick, L[1]] ", pass " M[pick, L[1]] " ms): its 300 frames can include some from before the situation was set up"
            if (n < 2) { print "UPSCALER NO_RESULT only one timing line fell inside the recording, and it was dropped"; exit 3 }
            nk = 0; nm = 0; nu = 0; sum = 0; c = 0
            for (i = 2; i <= n; i++) {
                k = L[i]; mo = MOD[pick, k]
                if (mo == 1) nk++; else if (mo == 2) nm++; else nu++
                c++; sum += M[pick, k]; v[c] = M[pick, k] + 0
                printf "UPSCALER_LINE %s model %s pass_ms %s (max %s) frames %s stages [%s] total_ms %s resume_ms %s cpu_dispatch_ms %s\n", T[pick, k], label(mo), M[pick, k], X[pick, k], F[pick, k], S[pick, k], TOT[pick, k], RES[pick, k], CPU[pick, k]
            }
            if (nk == c) mod = lab1 " (stage names " st1 " in all " c " lines)"
            else if (nm == c) mod = lab2 " (stage names " st2 " in all " c " lines)"
            else mod = "MIXED: " lab1 " in " nk ", " lab2 " in " nm ", other in " nu " of " c " lines"
            print "UPSCALER model " mod "; " (smx[pick] == "" ? "no network kernels line" : smx[pick])
            asort(v)
            printf "UPSCALER pass_ms over %d lines (the first dropped): mean of means %.3f, median %.3f, lowest %.3f, highest %.3f\n", c, sum / c, v[int((c + 1) / 2)], v[1], v[c]
        }' "$log" >"$out"
    rc=$?
    {
        echo "UPSCALER log $log"
        cat "$out"
        echo "RECORDER (same recording):"
        rec_brief "$dir"
        echo "EARLIER (offline, gaming round): Model $UPS_LABEL_1's pass takes about 1.61 ms at the high clock (3.1 to 3.2 GHz) and"
        echo "  about 2.8 to 3.6 ms at the low clock (1.6 to 1.8 GHz)."
    } >"$out.tmp" && mv "$out.tmp" "$out"
    sum "- upscaler $(basename "$dir"): $(grep -E '^UPSCALER (session|NO_LINES|NO_RESULT|dropped|model|pass_ms)' "$out" | sed 's/^UPSCALER //' | tr '\n' '|' | sed 's/|$//; s/|/ | /g')"
    sum "  clock in the same recording: $(grep -h -E '^AWAKE_SHARES' "$dir/summary.txt" 2>/dev/null)"
    cat "$out"
    ((rc == 0)) && return 0
    return 1
}

# ---------------------------------------------------------------------------------------------------------------
# Is a game running? Read from /proc only (nothing is signalled). Under Proton no process carries the game's name:
# the game shows among the card's clients as "GameThread", its command line is a Windows path (S:\steamapps\common\..),
# and steam.exe, wineserver and xalia.exe carry steamapps/common or compatibilitytools.d paths. The patterns are
# written so that they do not match the command line of the grep that searches for them.
GAME_PATH_RE='steamapps[/\\][c]ommon[/\\]|compatibilitytools[.]d[/\\]'

# game_signs: prints what shows a running game (one reason per line); false when nothing does.
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
# no_game: a step that starts a GPU load does not start beside a game.
no_game() {
    local signs
    signs=$(game_signs) || return 0
    echo "study-home: a game seems to be running, so this step (it starts GPU loads) does not start:" >&2
    printf '  %s\n' "$signs" | head -6 >&2
    echo "Close the game (and wait until Steam shows it stopped), then start the step again." >&2
    sum "- $STEP: not run, a game seems to be running ($(echo "$signs" | head -2 | tr '\n' ';'))"
    exit 7
}

# ---------------------------------------------------------------------------------------------------------------
# Steps

build_watch || { echo "study-home: the recorder did not build" >&2; exit 2; }
case $STEP in
short-tests | power-mode | perf-high | heat | wall-meter | memclock | overclock | oc-traces) no_game ;;
esac

case $STEP in
build)
    say "recorder built: $WATCH"
    ;;
check)
    d=$(step_dir check)
    {
        echo "settings: $(settings)"
        if is_testing; then
            echo "stock Testing settings: yes"
            settings >"$TESTING_REC"
        else
            echo "stock Testing settings: NO"
        fi
        if kit_busy; then
            echo "another step of the kit is running now: YES (wait for it before starting a step)"
        else
            echo "another step of the kit is running now: no"
        fi
        if signs=$(game_signs); then
            echo "a game is running: YES (load steps will not start): $(echo "$signs" | head -3 | tr '\n' ';')"
        else
            echo "a game is running: no"
        fi
        echo "LACT profiles: $(timeout 20 lact cli profile list 2>/dev/null | tr '\n' ' ')"
        echo "power modes offered: $(power_modes_offered)"
        echo "active memory clock: $(active_mclk) MHz; hotspot $(hot) C; available memory $(mem_gb) GB"
        echo "programs that hold the card (close these for the 'apps closed' test; the desktop itself stays):"
        "$WATCH" clients | sed 's/^/  /'
    } | tee "$d/check.txt"
    sum "- check: $(head -3 "$d/check.txt" | tr '\n' ' ')"
    ;;
game)
    label=${ARGS[0]:-}
    usage_game="usage: study-home.sh game <label> [--seconds 120] [--period-ms 2] [--upscaler [--log PATH]]"
    [[ $label =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "$usage_game" >&2; exit 2; }
    secs=120 per=2 ups=0 log=""
    set -- "${ARGS[@]:1}"
    while (($#)); do
        case $1 in
        --seconds | --period-ms | --log)
            [[ $# -ge 2 && -n $2 ]] || { echo "study-home: $1 needs a value; $usage_game" >&2; exit 2; }
            case $1 in
            --seconds) secs=$2 ;;
            --period-ms) per=$2 ;;
            --log) log=$2 ;;
            esac
            shift
            ;;
        --upscaler) ups=1 ;;
        *) echo "study-home: unknown option $1" >&2; exit 2 ;;
        esac
        shift
    done
    [[ $secs =~ ^[0-9]+([.][0-9]+)?$ && $secs != 0 ]] || { echo "study-home: --seconds takes a positive number" >&2; exit 2; }
    [[ $per =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "study-home: --period-ms takes a number (0.5 or more)" >&2; exit 2; }
    d=$(step_dir "game-$label")
    if ((ups)); then
        [[ -z $log ]] && log=$(steam_log_path)
        [[ -n $log && -r $log ]] || { sum "- game $label: the upscaler log was not found (${log:-no path}); give it with --log PATH"; exit 1; }
        printf 'path: %s\nsize_start: %s\n' "$log" "$(log_size "$log")" >"$d/upscaler-log.txt"
    fi
    game_signs >/dev/null || say "Note: no running game found (no GameThread card client, no Steam game or Proton process). Start the game first if you meant to record it."
    say "Recording for $secs s ($label). Play now; hold the situation steady until it says done (Ctrl-C ends it early)."
    # Ctrl-C and a closed terminal go to the recorder, which then finishes its files; this script carries on to the
    # summary either way.
    trap '' INT HUP
    "$WATCH" record --out "$d" --label "game-$label" --period-ms "$per" --seconds "$secs" --die-with-parent >"$d/recorder.out" 2>&1 8>&-
    rc=$?
    if ((ups)); then
        say "Recording done. Keep playing $UPS_TAIL_S more seconds while the upscaler's log catches up."
        sleep "$UPS_TAIL_S"
        echo "size_end: $(log_size "$log")" >>"$d/upscaler-log.txt"
    fi
    say "done."
    sum "- game $label ($(date -u +%FT%TZ), ${secs} s every $per ms, recorder exit $rc):"
    rec_brief "$d" | tee -a "$SUM"
    ((ups)) && upscaler_read "$d" "$log"
    ((rc == 0)) || exit 1
    ;;
upscaler)
    [[ -n ${ARGS[0]:-} ]] || { echo "usage: study-home.sh upscaler <step folder> [--log PATH]" >&2; exit 2; }
    log=""
    [[ ${ARGS[1]:-} == --log ]] && log=${ARGS[2]:-}
    upscaler_read "${ARGS[0]}" "$log"
    exit $?
    ;;
short-tests)
    label=${ARGS[0]:-control}
    [[ $label =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "usage: study-home.sh short-tests [label]" >&2; exit 2; }
    need_bins "$BIN_LAUNCH"
    need_vk
    d=$(step_dir "short-$label")
    sum "- short-tests $label ($(date -u +%FT%TZ)): $(settings)"
    "$WATCH" clients >"$d/clients.txt"
    short_tests "$d" "$label"
    ((STOP)) && exit 1
    ;;
power-mode)
    set -- "${ARGS[@]}"
    while (($#)); do
        case $1 in
        --mode) POWER_MODE=${2:-}; shift ;;
        --profile) PROFILE_POWER=${2:-}; shift ;;
        *) echo "usage: study-home.sh power-mode [--mode NAME] [--profile NAME] --owner-ok=power-mode" >&2; exit 2 ;;
        esac
        shift
    done
    [[ $PROFILE_POWER =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "study-home: --profile needs a profile name" >&2; exit 2; }
    [[ " $(power_modes_offered)" == *" $POWER_MODE "* ]] || { echo "study-home: the driver offers no power mode '$POWER_MODE' (it offers: $(power_modes_offered))" >&2; exit 2; }
    b_setting power-mode "$PROFILE_POWER" check_power_mode "the power profile mode set to $POWER_MODE (LACT shows it only with the performance level 'Manual'; that is fine)" mode perf || exit 1
    ;;
perf-high)
    b_setting perf-high "$PROFILE_HIGH" check_perf_high "the performance level set to 'Highest Clocks' (high)" perf || exit 1
    ;;
heat)
    kind=${ARGS[0]:-}
    case $kind in
    normal)
        need_bins "$BIN_TRACE"
        d=$(step_dir heat-normal)
        start_on_testing "$d"
        sum "- heat normal ($(date -u +%FT%TZ)): the usual fan curve; $(settings)"
        heat_loads "$d" normal
        ;;
    fan100)
        needs_owner fan100 "In LACT, make a new profile named $PROFILE_FAN as a copy of Testing, and change only the fan: Static, 100 %." \
            "Choose the Testing profile again in LACT. The script switches with: lact cli profile set $PROFILE_FAN, and back to Testing at the end." \
            "Stay at the machine: the fan will be loud for about 10 minutes."
        need_bins "$BIN_TRACE"
        need_profile "$PROFILE_FAN"
        d=$(step_dir heat-fan100)
        start_on_testing "$d"
        hold_lock || { stop_run "the shared lock was not free within 60 s before the switch to $PROFILE_FAN"; exit 1; }
        RESTORED=0
        if switch_to "$PROFILE_FAN" check_fan100 fan_curve fan_zero_rpm fan_min_pwm pwm1_enable pwm1; then
            sum "- heat fan100 ($(date -u +%FT%TZ)): $(settings)"
            heat_loads "$d" fan100
        fi
        restore
        release_hold
        ;;
    *) echo "usage: study-home.sh heat normal | heat fan100 --owner-ok=fan100" >&2; exit 2 ;;
    esac
    ((STOP)) && exit 1
    ;;
wall-meter)
    need_bins "$BIN_TRACE"
    d=$(step_dir wall-meter)
    start_on_testing "$d"
    sum "- wall-meter ($(date -u +%FT%TZ)): idle, stream, FP16 matrix, 20 s each"
    declare -A METER DRIVER
    for load in idle stream f16; do
        say ""
        if [[ $load == idle ]]; then
            say ">>> $load: read the wall meter while this runs (20 s) and remember the number."
            rd=$d/rec-idle
            mkdir -p "$rd"
            "$WATCH" record --out "$rd" --label idle --seconds 20 --die-with-parent >"$rd/recorder.out" 2>&1 8>&- ||
                { stop_run "the idle recording failed"; exit 1; }
        else
            # No cool-down in the recording, so its median power is the load's. The prompt is printed only when
            # the load starts, after the spacing and the temperature gate.
            say ">>> $load: wait for the line 'NOW: read the wall meter' (the load may first wait to start)."
            HOME_PROMPT=">>> NOW: read the wall meter for $load while this runs (20 s) and remember the number."
            trace "$d" "$load" "$load" 20 95 0
            HOME_PROMPT=""
            rd=$d/rec-$load
        fi
        ((STOP)) && exit 1
        DRIVER[$load]=$(awk '/^POWER_W/ {for (i = 1; i <= NF; i++) if ($i == "p50") print $(i + 1)}' "$rd/summary.txt")
        v=""
        until [[ $v =~ ^[0-9]+([.][0-9]+)?$ ]]; do
            read -r -p "Meter reading during $load, in watts: " v || { stop_run "no meter reading typed"; exit 1; }
        done
        METER[$load]=$v
        sum "  - $load: meter ${METER[$load]} W, driver's reading (median) ${DRIVER[$load]} W"
    done
    for load in stream f16; do
        sum "  - $load minus idle: meter $(awk -v a="${METER[$load]}" -v b="${METER[idle]}" 'BEGIN {print a - b}') W," \
            "driver $(awk -v a="${DRIVER[$load]}" -v b="${DRIVER[idle]}" 'BEGIN {print a - b}') W, ratio" \
            "$(awk -v a="${METER[$load]}" -v b="${METER[idle]}" -v c="${DRIVER[$load]}" -v e="${DRIVER[idle]}" 'BEGIN {if (c - e > 0) printf "%.2f", (a - b) / (c - e); else print "n/a"}')" \
            "(wall watts per reported watt; includes the power supply's loss and any change in the rest of the PC)"
    done
    ;;
memclock)
    needs_owner memclock "Run scripts/study-home.sh check on Testing first (it records the Testing settings for this step)." \
        "In LACT, make a new profile named $PROFILE_MEM as a copy of Testing. In it, set only the maximum memory clock to 1409 MHz and press Apply, by hand, the way you remember it working." \
        "Leave LACT on $PROFILE_MEM. The script checks that nothing else differs from Testing, reads what the card does, runs a 10 s memory test and a short one, and then switches back with: lact cli profile set Testing"
    need_bins "$BIN_TRACE" "$BIN_BW"
    d=$(step_dir memclock)
    if [[ ! -s $TESTING_REC ]]; then
        sum "- memclock: not run: no record of the Testing settings in this sitting; choose Testing in LACT, run scripts/study-home.sh check, then choose $PROFILE_MEM and start this step again"
        exit 3
    fi
    if [[ $(profile_now) != "$PROFILE_MEM" || $(od_mclk_max) != 1409 ]]; then
        sum "- memclock: not run: the profile is '$(profile_now)' and the memory request reads $(od_mclk_max) MHz; expected $PROFILE_MEM and 1409"
        exit 3
    fi
    # The settings to come back to: those read on Testing earlier in this sitting.
    BASE_SETTINGS=$(cat "$TESTING_REC")
    RESTORED=0
    extra=$(unexpected_fields mclk_request)
    if [[ -n $extra ]]; then
        stop_run "memclock: $PROFILE_MEM differs from Testing in more than the memory request: $extra. Switching back to Testing, no load. Now: $(settings)"
        restore
        exit 1
    fi
    {
        echo "settings: $(settings)"
        echo "pp_dpm_mclk: $(tr '\n' ' ' <"$DRM_DEVICE/pp_dpm_mclk")"
    } >"$d/memclock-state.txt"
    pp_mark=$(active_mclk)
    sum "- memclock ($(date -u +%FT%TZ)): request $(od_mclk_max) MHz; running memory clock $(uclk_now) MHz (gpu_metrics); pp_dpm_mclk marks ${pp_mark:-no level}; levels: $(tr '\n' ' ' <"$DRM_DEVICE/pp_dpm_mclk")"
    if [[ $(uclk_now) == 1258 ]]; then
        say "The memory still runs at 1258 MHz. A new memory maximum takes effect only when the displays blank: switch"
        say "your monitors off for about 15 s and on again (or log out and in), then start this step again. (Found 2026-11-18.)"
    fi
    hold_lock || { stop_run "the shared lock was not free within 60 s"; exit 1; }
    trace "$d" stream10 stream 10 95
    for k in 1 2 3; do job "$d" "bandwidth-$k" 60 95 "$BIN_BW" 1024 50; done
    sum "  - bandwidth read GB/s (study on stock: 632.6 to 632.9): $(awk '$1 == "RESULT" && $2 == "read" {printf "%s ", $8}' "$d"/bandwidth-*.log 2>/dev/null)"
    sum "  - memory clock seen by the recorder: $(grep -h '^UCLK_MHz\|^MCLK_MHz' "$d/rec-stream10/summary.txt" 2>/dev/null | tr '\n' ' ')"
    restore
    release_hold
    ((STOP)) && exit 1
    ;;
overclock)
    needs_owner overclock "Stay at the machine. This runs the study's overclock comparison end to end: it switches LACT to Default (your overclock) and back to Testing itself." \
        "It is the first real run of that script since it was reworked; its own checks stop it if anything is off."
    d=$(step_dir overclock)
    rec_start "$d/rec" overclock 5 || { rec_stop; stop_run "the recorder did not start"; exit 1; }
    OVERCLOCK_RUN_DIR=$d/run "$REPO/scripts/study-overclock.sh" 2>&1 | tee "$d/overclock.out"
    rc=${PIPESTATUS[0]}
    rec_stop
    sum "- overclock ($(date -u +%FT%TZ)): scripts/study-overclock.sh exit $rc (0 done, 1 stopped, 3 not on Testing at the start, 4 not Testing at the end, 5 RESTORE FAILED); its summary: $d/run/summary.md"
    ((rc == 0)) || exit "$rc"
    ;;
oc-traces)
    needs_owner oc-traces "Stay at the machine and watch the hotspot: the overclock profile reached 102 C under a long load before the fan curve was changed." \
        "The script switches LACT to Default with: lact cli profile set Default, runs four 20 s loads, each stopped at $OC_STOP_C C, and switches back to Testing."
    need_bins "$BIN_TRACE"
    need_profile Default
    d=$(step_dir oc-traces)
    start_on_testing "$d"
    hold_lock || { stop_run "the shared lock was not free within 60 s before the switch to Default"; exit 1; }
    RESTORED=0
    # shellcheck disable=SC2329  # called by name (set_profile)
    check_default() { [[ $(cap_w) == 374 && $(od_value OD_SCLK_OFFSET) == 150 && $(od_value OD_VDDGFX_OFFSET) == -50 ]]; }
    if switch_to Default check_default cap_W sclk_offset vddgfx_offset mclk_request; then
        sum "- oc-traces ($(date -u +%FT%TZ)): $(settings)"
        for load in light stream f16 fp8; do
            trace "$d" "oc-$load" "$load" 20 "$OC_STOP_C"
            ((STOP)) && break
        done
    fi
    restore
    release_hold
    ((STOP)) && exit 1
    ;;
*)
    echo "study-home: unknown step '$STEP' (run without arguments for the list)" >&2
    exit 2
    ;;
esac
exit 0
