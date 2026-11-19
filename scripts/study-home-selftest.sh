#!/usr/bin/env bash
# Self-test of the home-session kit: the recorder (build/study/bin/watch_card) against a fake sensor tree and
# constructed recordings, scripts/study-home.sh in self-test mode (HOME_SELFTEST=1) with a stand-in for `lact` first in
# PATH (it records every call and never talks to the daemon) and fake probes in place of every GPU program, and the
# upscaler-log reader against made-up logs in the real log's line format. Nothing here touches the card's settings,
# starts a GPU load or takes the shared GPU lock or the kit's own lock: the tests use lock files of their own.
#
# Usage: scripts/study-home-selftest.sh                       all tests, about 6 minutes
#        scripts/study-home-selftest.sh r01 k03               named tests only
#        BREAK=<name> scripts/study-home-selftest.sh <test>   deliberate break, must FAIL:
#          expected values: wrongexp (r01, r03 expect a wrong value), upscaler (u01 expects a wrong mean)
#          properties: noflag (k01: the owner's flag is not checked), norestore (k04: the switch back is not
#          compared), gamelock (k05: the game step takes the GPU lock), nokitlock (k19: a second copy is not refused),
#          nofanfields (k17: zero RPM is not among the compared settings), thermalstop (k20: a temperature stop does
#          not end the step's loads), hooks (k24: test hooks are read without HOME_SELFTEST=1), extrafield (k22: a
#          profile's other fields are not compared), nokill (k21 and r10: the guard never sends KILL), nopdeath (k31:
#          a load outlives a script killed with SIGKILL), nogameguard (k32: load steps start beside a game)
# Results: runs/<UTC time>-home-selftest/: results.txt (one line per test, then VERDICT) and one folder per test with
# result.txt, the script's output, the stand-in's call record, the fake probes' record and the run folder.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 2
OUT=$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-home-selftest
mkdir -p "$OUT"
RES=$OUT/results.txt
SCRIPT=$REPO/scripts/study-home.sh
WATCH=$REPO/build/study/bin/watch_card
LOCK=$OUT/test-gpu.lock
: >"$LOCK"
BREAK=${BREAK:-none}
FAILS=0
CXX=(g++ -std=c++20 -O2 -Wall -Wextra -Werror)

mkdir -p "$(dirname "$WATCH")"
"${CXX[@]}" -o "$WATCH" "$REPO/src/study/watch_card.cpp" || { echo "selftest: build failed" | tee -a "$RES"; exit 2; }

BROKEN=$REPO/scripts/.study-home-broken.sh
BROKEN_CPP=$REPO/build/home-selftest-broken/watch_card.cpp
# shellcheck disable=SC2329  # called by the trap
cleanup() {
    rm -f "$BROKEN"
    # A broken recorder must never stay where the kit finds it.
    [[ $BREAK == nokill ]] && "${CXX[@]}" -o "$WATCH" "$REPO/src/study/watch_card.cpp"
}
trap cleanup EXIT
case $BREAK in
none | wrongexp | upscaler) ;;
noflag)
    # shellcheck disable=SC2016
    sed 's/^    \[\[ \$OWNER_OK == "\$s" \]\] && return 0$/    return 0/' "$SCRIPT" >"$BROKEN"
    ;;
norestore)
    sed 's/^back_to_base() { .*$/back_to_base() { true; }/' "$SCRIPT" >"$BROKEN"
    ;;
gamelock)
    sed "s/^    trap '' INT HUP\$/    hold_lock; trap '' INT HUP/" "$SCRIPT" >"$BROKEN"
    ;;
nokitlock)
    sed 's/^    if ! flock -n 8; then$/    if false; then/' "$SCRIPT" >"$BROKEN"
    ;;
nofanfields)
    # shellcheck disable=SC2016
    sed 's/ fan_zero_rpm=\$(fan_od fan_zero_rpm_enable)//' "$SCRIPT" >"$BROKEN"
    ;;
thermalstop)
    # shellcheck disable=SC2016
    sed '/^    if grep -q .\^THERMAL STOP. "\$log"; then$/,/^    fi$/d' "$SCRIPT" >"$BROKEN"
    ;;
hooks)
    sed 's/^    if ((SELFTEST)) && \[\[ -n \${!1:-} \]\]; then$/    if [[ -n ${!1:-} ]]; then/' "$SCRIPT" >"$BROKEN"
    ;;
extrafield)
    # shellcheck disable=SC2016
    sed 's/^        \[\[ \$allowed == \*" \$f "\* \]\] || out+="\$f "$/        :/' "$SCRIPT" >"$BROKEN"
    ;;
nogameguard)
    sed 's/^short-tests | power-mode | perf-high | heat | wall-meter | memclock | overclock | oc-traces) no_game ;;$/short-tests) ;;/' "$SCRIPT" >"$BROKEN"
    ;;
nopdeath)
    sed 's/ nice -n 10 setpriv --pdeathsig TERM \\$/ nice -n 10 \\/' "$SCRIPT" >"$BROKEN"
    ;;
nokill)
    mkdir -p "$(dirname "$BROKEN_CPP")"
    sed '/::kill(-g.pid, SIGKILL);/d; /::kill(g.pid, SIGKILL);/d' "$REPO/src/study/watch_card.cpp" >"$BROKEN_CPP"
    cmp -s "$REPO/src/study/watch_card.cpp" "$BROKEN_CPP" && { echo "selftest: the break did not change the recorder" | tee -a "$RES"; exit 2; }
    "${CXX[@]}" -I "$REPO/src/study" -o "$WATCH" "$BROKEN_CPP" || { echo "selftest: broken build failed" | tee -a "$RES"; exit 2; }
    ;;
*)
    echo "selftest: BREAK is one of wrongexp upscaler noflag norestore gamelock nokitlock nofanfields thermalstop hooks extrafield nokill nopdeath nogameguard" >&2
    exit 2
    ;;
esac
if [[ -f $BROKEN ]]; then
    cmp -s "$SCRIPT" "$BROKEN" && { echo "selftest: the break did not change the script" | tee -a "$RES"; exit 2; }
    chmod +x "$BROKEN"
    SCRIPT=$BROKEN
fi
[[ $BREAK != none ]] && echo "DELIBERATE BREAK: $BREAK" | tee -a "$RES"

# ---------------------------------------------------------------------------------------------------------------
# The fake card. gpu_metrics is built from the kernel's struct gpu_metrics_v1_3 (format 1, content 3, 120 bytes):
# 0 structure_size, 2 format, 3 content, 16 average_gfx_activity, 22 average_socket_power, 40 average_gfxclk,
# 54 current_gfxclk, 58 current_uclk, 68 throttle_status (32 bits), 96 firmware_timestamp (all ones, as on this card),
# 106 voltage_gfx, 112 indep_throttle_status (64 bits).

le() {
    local v=$1 n=$2 i out=""
    for ((i = 0; i < n; i++)); do out+=$(printf '\\x%02x' $(((v >> (8 * i)) & 255))); done
    printf '%s' "$out"
}
zeros() { le 0 "$1"; }
# mk_metrics <file> <current gfxclk> <average gfxclk> <throttle_status> <indep_throttle_status> <vgfx mV> <activity> <socket W>
#            [<current uclk MHz, default 1258>]
mk_metrics() {
    local fmt
    fmt="$(le 120 2)\\x01\\x03$(zeros 12)$(le "$7" 2)$(zeros 4)$(le "$8" 2)$(zeros 16)$(le "$3" 2)$(zeros 12)$(le "$2" 2)"
    fmt+="$(zeros 2)$(le "${9:-1258}" 2)$(zeros 8)$(le "$4" 4)$(zeros 24)$(le -1 8)$(zeros 2)$(le "$6" 2)$(zeros 4)$(le "$5" 8)"
    # shellcheck disable=SC2059
    printf "$fmt" >"$1.tmp" && mv "$1.tmp" "$1"
}
# mk_modes <file> <active mode name>: pp_power_profile_mode as this card prints it.
mk_modes() {
    local i=0 m star
    {
        echo "PROFILE_INDEX(NAME) CLOCK_TYPE(NAME) FPS MinActiveFreqType MinActiveFreq BoosterFreqType BoosterFreq PD_Data_limit_c PD_Data_error_coeff PD_Data_error_rate_coeff"
        for m in BOOTUP_DEFAULT 3D_FULL_SCREEN POWER_SAVING VIDEO VR COMPUTE CUSTOM WINDOW_3D; do
            star=" "
            [[ $m == "$2" ]] && star="*"
            printf '%2d %14s%s:\n' "$i" "$m" "$star"
            echo "                    0(       GFXCLK)       0       1       0       4     800 4587520  -65536       0"
            echo "                    1(         FCLK)       0       3       0       1       0 5898240   -6553   -6553"
            i=$((i + 1))
        done
    } >"$1.tmp" && mv "$1.tmp" "$1"
}
mk_od() {
    printf 'OD_SCLK_OFFSET:\n%sMhz\nOD_MCLK:\n0: 97Mhz\n1: %sMHz\nOD_RANGE:\nSCLK_OFFSET:    -500Mhz       1000Mhz\nMCLK:      97Mhz       1500Mhz\nVDDGFX_OFFSET:    -200mv          0mv\nOD_VDDGFX_OFFSET:\n%smV\n' \
        "$2" "$3" "$4" >"$1.tmp" && mv "$1.tmp" "$1"
}
# The fan files as this card prints them (gpu_od/fan_ctrl, read on the machine 2026-11-18).
mk_fan() {
    if [[ $2 == normal ]]; then
        printf 'OD_FAN_CURVE:\n0: 55C 30%%\n1: 58C 58%%\n2: 62C 73%%\n3: 66C 94%%\n4: 73C 100%%\nOD_RANGE:\nFAN_CURVE(hotspot temp): 25C 100C\nFAN_CURVE(fan speed): 30%% 100%%\n' >"$1.tmp"
    else
        printf 'OD_FAN_CURVE:\n0: 55C 100%%\n1: 58C 100%%\n2: 62C 100%%\n3: 66C 100%%\n4: 73C 100%%\nOD_RANGE:\nFAN_CURVE(hotspot temp): 25C 100C\nFAN_CURVE(fan speed): 30%% 100%%\n' >"$1.tmp"
    fi
    mv "$1.tmp" "$1"
}
mk_zero() { printf 'FAN_ZERO_RPM_ENABLE:\n%s\nOD_RANGE:\nZERO_RPM_ENABLE: 0 1\n' "$2" >"$1.tmp" && mv "$1.tmp" "$1"; }
put() { printf '%s\n' "$2" >"$1.tmp" && mv "$1.tmp" "$1"; }

# setup <test>: a fresh folder T with the fake card on the stock Testing settings, the stand-in lact, the fake probes,
# a fake /proc, and the environment that points the kit at all of it. The sensor folder sits under the device folder
# as on the machine; $T/hwmon is a link to it.
setup() {
    T=$OUT/$1
    local F=$T/drm/gpu_od/fan_ctrl
    mkdir -p "$T/bin" "$F" "$T/drm/hwmon/hwmon9" "$T/proc/sys/kernel" "$T/proc/4242/fdinfo" "$T/proc/4243/fdinfo" "$T/run"
    ln -sfn drm/hwmon/hwmon9 "$T/hwmon"
    echo Testing >"$T/profile"
    printf 'Default\nTesting\nHome-Power\nHome-High\nHome-Fan100\nHome-Mem\n' >"$T/profiles"
    put "$T/drm/vendor" 0x1002
    put "$T/drm/device" 0x7550
    put "$T/drm/pp_dpm_sclk" $'S: 91Mhz\n0: 500Mhz\n1: 2460Mhz *'
    put "$T/drm/pp_dpm_mclk" $'0: 96Mhz\n1: 456Mhz\n5: 1258Mhz *'
    mk_metrics "$T/drm/gpu_metrics" 2455 2440 2 $(((1 << 36) | 1)) 1056 37 123
    put "$T/drm/gpu_busy_percent" 37
    put "$T/drm/mem_busy_percent" 12
    put "$T/drm/mem_info_vram_used" 1073741824
    put "$T/drm/power_dpm_force_performance_level" auto
    mk_modes "$T/drm/pp_power_profile_mode" BOOTUP_DEFAULT
    mk_od "$T/drm/pp_od_clk_voltage" 0 1259 0
    mk_fan "$F/fan_curve" normal
    mk_zero "$F/fan_zero_rpm_enable" 1
    put "$F/fan_minimum_pwm" $'FAN_MINIMUM_PWM:\n30\nOD_RANGE:\nMINIMUM_PWM: 30 100'
    put "$F/fan_zero_rpm_stop_temperature" $'FAN_ZERO_RPM_STOP_TEMPERATURE:\n55\nOD_RANGE:\nZERO_RPM_STOP_TEMPERATURE: 30 80'
    put "$F/fan_target_temperature" $'FAN_TARGET_TEMPERATURE:\n76\nOD_RANGE:\nTARGET_TEMPERATURE: 25 105'
    put "$F/acoustic_limit_rpm_threshold" $'OD_ACOUSTIC_LIMIT:\n6000\nOD_RANGE:\nACOUSTIC_LIMIT: 500 6000'
    put "$F/acoustic_target_rpm_threshold" $'OD_ACOUSTIC_TARGET:\n1000\nOD_RANGE:\nACOUSTIC_TARGET: 500 6000'
    put "$T/hwmon/power1_average" 123000000
    put "$T/hwmon/power1_cap" 317000000
    put "$T/hwmon/power1_cap_default" 317000000
    put "$T/hwmon/temp1_input" 45000
    put "$T/hwmon/temp2_input" 50000
    put "$T/hwmon/temp3_input" 52000
    put "$T/hwmon/fan1_input" 1234
    put "$T/hwmon/pwm1" 0
    put "$T/proc/sys/kernel/osrelease" 9.9.9-fake
    put "$T/proc/4242/comm" fakegame
    printf 'pos:\t0\nflags:\t02\ndrm-driver:\tamdgpu\ndrm-pdev:\t0000:99:00.0\ndrm-client-id:\t7\n' >"$T/proc/4242/fdinfo/5"
    printf 'pos:\t0\nflags:\t02\ndrm-driver:\tamdgpu\ndrm-pdev:\t0000:99:00.0\ndrm-client-id:\t7\n' >"$T/proc/4242/fdinfo/6"
    put "$T/proc/4243/comm" otherdevice
    printf 'drm-driver:\tamdgpu\ndrm-pdev:\t0000:7c:00.0\ndrm-client-id:\t3\n' >"$T/proc/4243/fdinfo/5"
    put "$T/meminfo" "MemAvailable:   16777216 kB"
    printf 'ARCH=gfx1201\nHIP_INDEX=0\nDRM_DEVICE=%s\nHWMON=%s\n' "$T/drm" "$T/drm/hwmon/hwmon9" >"$T/card.env"
    # The stand-in for lact. A "switch" rewrites the fake card's files the way LACT would apply that profile; the fixed
    # fan speed follows LACT on this card (a flat curve and zero RPM off), or hwmon pwm1 with SHIM_FAN_HWMON=1.
    cat >"$T/bin/lact" <<'EOF'
#!/usr/bin/env bash
# Stand-in for lact: never talks to the daemon. Records every call. SHIM_SET_FAIL=all: every `profile set` fails.
# SHIM_NOAPPLY=1: a set changes the profile name only. SHIM_TESTING_KEEPS_MODE=1: "Testing" leaves the power mode.
# SHIM_TESTING_KEEPS_FAN=1: "Testing" leaves every fan setting. SHIM_TESTING_KEEPS_ZERO=1: "Testing" rewrites the
# curve but leaves zero RPM off. SHIM_EXTRA_MCLK=1: Home-Power also sets the memory request to 1409 MHz.
source "$SHIM_DIR/fake.sh"
echo "$*" >>"$SHIM_DIR/calls.log"
case "$1 $2 $3" in
"cli profile get") cat "$SHIM_DIR/profile" ;;
"cli profile list") cat "$SHIM_DIR/profiles" ;;
"cli profile set")
    [[ ${SHIM_SET_FAIL:-} == all ]] && exit 1
    grep -qxF -- "$4" "$SHIM_DIR/profiles" || { echo "Error: profile $4 not found" >&2; exit 1; }
    echo "$4" >"$SHIM_DIR/profile"
    [[ ${SHIM_NOAPPLY:-} == 1 ]] && exit 0
    D=$SHIM_DIR/drm H=$SHIM_DIR/hwmon F=$SHIM_DIR/drm/gpu_od/fan_ctrl
    case $4 in
    Testing)
        put "$H/power1_cap" 317000000; mk_od "$D/pp_od_clk_voltage" 0 1259 0
        if [[ ${SHIM_TESTING_KEEPS_FAN:-} != 1 ]]; then
            mk_fan "$F/fan_curve" normal
            [[ ${SHIM_TESTING_KEEPS_ZERO:-} == 1 ]] || mk_zero "$F/fan_zero_rpm_enable" 1
            [[ -e $H/pwm1_enable ]] && { put "$H/pwm1_enable" 2; put "$H/pwm1" 0; }
        fi
        if [[ ${SHIM_TESTING_KEEPS_MODE:-} != 1 ]]; then put "$D/power_dpm_force_performance_level" auto; mk_modes "$D/pp_power_profile_mode" BOOTUP_DEFAULT; fi ;;
    Home-Power)
        put "$D/power_dpm_force_performance_level" manual; mk_modes "$D/pp_power_profile_mode" 3D_FULL_SCREEN
        [[ ${SHIM_EXTRA_MCLK:-} == 1 ]] && mk_od "$D/pp_od_clk_voltage" 0 1409 0 ;;
    Home-High) put "$D/power_dpm_force_performance_level" high ;;
    Home-Fan100)
        if [[ ${SHIM_FAN_HWMON:-} == 1 ]]; then put "$H/pwm1_enable" 1; put "$H/pwm1" 255
        else mk_fan "$F/fan_curve" fan100; mk_zero "$F/fan_zero_rpm_enable" 0; fi ;;
    Home-Mem) mk_od "$D/pp_od_clk_voltage" 0 1409 0 ;;
    Default) put "$H/power1_cap" 374000000; mk_od "$D/pp_od_clk_voltage" 150 1409 -50 ;;
    esac ;;
esac
exit 0
EOF
    declare -f le zeros mk_metrics mk_modes mk_od mk_fan mk_zero put >"$T/fake.sh"
    # One fake probe for every GPU program: it records its call, whether the GPU lock was held while it ran, the fake
    # card's power mode and performance level at that moment, and how many samples the job's recording held at its
    # start; sleeps FAKE_PROBE_S (ignoring TERM with FAKE_IGNORE_TERM=1); writes <out>-sysfs.csv.
    cat >"$T/bin/fakeprobe" <<'EOF'
#!/usr/bin/env bash
[[ ${FAKE_IGNORE_TERM:-} == 1 ]] && trap '' TERM
lk=held
flock -n "$HOME_GPU_LOCK" true && lk=free
mode=$(awk '/^ *[0-9]+ +[A-Z0-9_]+\*/ {gsub(/[*:]/, "", $2); print $2}' "$SHIM_DIR/drm/pp_power_profile_mode")
rows=$(awk '!/^#/ {n++} END {print (n > 0 ? n - 1 : 0)}' "${HOME_REC_DIR:-/nonexistent}/samples.csv" 2>/dev/null)
echo "$$ $(basename "$0") [${HOME_JOB:-}] $* | lock=$lk mode=$mode perf=$(cat "$SHIM_DIR/drm/power_dpm_force_performance_level") rec_rows=${rows:-none}" >>"$SHIM_DIR/probe-calls.log"
echo "RESULT fakeprobe lock=$lk mode=$mode"
# FAKE_POWER_MAP="stream=200 f16=300": the fake reported power while a load of that name runs, then 50 W again.
for kv in ${FAKE_POWER_MAP:-}; do
    [[ $1 == "${kv%%=*}" ]] && echo "$((${kv#*=} * 1000000))" >"$SHIM_DIR/hwmon/power1_average"
done
prev=""
for a in "$@"; do [[ $prev == --out ]] && echo "t_s" >"$a-sysfs.csv"; prev=$a; done
sleep "${FAKE_PROBE_S:-0.2}"
[[ -n ${FAKE_POWER_MAP:-} ]] && echo 50000000 >"$SHIM_DIR/hwmon/power1_average"
exit 0
EOF
    chmod +x "$T/bin/lact" "$T/bin/fakeprobe"
    : >"$T/calls.log"
    : >"$T/probe-calls.log"
}
# The self-test environment of the kit (KENV).
kit_env() {
    KENV=(SHIM_DIR="$T" PATH="$T/bin:$PATH" HOME_SELFTEST=1 CARD_ENV="$T/card.env" HOME_RUN_DIR="$T/run"
        HOME_GPU_LOCK="$LOCK" HOME_KIT_LOCK="$T/kit.lock" HOME_MEMINFO="$T/meminfo" HOME_BIN_LAUNCH="$T/bin/fakeprobe"
        HOME_BIN_TRACE="$T/bin/fakeprobe" HOME_BIN_BW="$T/bin/fakeprobe" HOME_SETTLE_S=0 HOME_SWITCH_WAIT_S=3
        HOME_UPS_TAIL_S=0 HOME_PROC="$T/proc" HOME_LOCAL_ENV=/dev/null WATCH_PDEV=0000:99:00.0 WATCH_PROC="$T/proc")
}
# kit <env assignments...> -- <script arguments...>: runs study-home.sh against the fakes; output in out.txt (or KIT_OUT).
kit() {
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    kit_env
    env "${KENV[@]}" "${envs[@]}" "$SCRIPT" "$@" >>"${KIT_OUT:-$T/out.txt}" 2>&1 </dev/null
}
# kit_bg: the same in the background (its PID in BG_PID), with INT and HUP set back to their default (a background
# job of a script starts with INT ignored, and an ignored signal cannot be trapped).
kit_bg() {
    local envs=()
    while [[ $1 != -- ]]; do envs+=("$1"); shift; done
    shift
    kit_env
    env --default-signal=INT,HUP "${KENV[@]}" "${envs[@]}" "$SCRIPT" "$@" >>"${KIT_OUT:-$T/out.txt}" 2>&1 </dev/null &
    BG_PID=$!
}
count() {
    local pat=$1 n
    shift
    n=$(cat "$@" 2>/dev/null | grep -c -- "$pat")
    echo "${n:-0}"
}
sets() { count "profile set${1:+ $1}" "$T/calls.log"; }
probes() { count '' "$T/probe-calls.log"; }
record() {
    local name=$1 want=$2 ok=$3 seen=$4 st=PASS
    ((ok)) || { st=FAIL; FAILS=$((FAILS + 1)); }
    echo "$st $name | expected: $want | seen: $seen" | tee -a "$RES" >"$T/result.txt"
    cat "$T/result.txt"
}
want() { [[ $# -eq 0 ]] && return 0; local t; for t in "$@"; do [[ $t == "$CUR" ]] && return 0; done; return 1; }
# csv_col <file> <column>: the distinct values of one column of samples.csv.
csv_col() { awk -F, -v c="$2" '/^#/ {next} !h {for (i = 1; i <= NF; i++) if ($i == c) k = i; h = 1; next} {print $k}' "$1" | sort -u | tr '\n' ' '; }
# rec <seconds> <period> [more options]: the recorder alone against the fake card of this test.
rec() {
    local s=$1 p=$2
    shift 2
    env SHIM_DIR="$T" PATH="$T/bin:$PATH" DRM_DEVICE="$T/drm" HWMON="$T/hwmon" WATCH_PDEV=0000:99:00.0 WATCH_PROC="$T/proc" \
        "$WATCH" record --out "$T/rec" --label "$CUR" --seconds "$s" --period-ms "$p" "$@" >"$T/out.txt" 2>&1
}
# wait_for <seconds> <command...>: polls every 0.1 s until the command succeeds; false on timeout. Every wait of a
# test is bounded, so that a test of broken code fails instead of hanging.
wait_for() {
    local n=$(($1 * 10))
    shift
    until "$@"; do
        ((n-- > 0)) || return 1
        sleep 0.1
    done
}
has_line() { grep -q -- "$1" "${@:2}" 2>/dev/null; }

# ---------------------------------------------------------------------------------------------------------------
# Recorder

r01() {
    setup r01
    rec 1 2
    local rc=$? csv=$T/rec/samples.csv s n want_sclk=2460 ok=1 seen
    [[ $BREAK == wrongexp ]] && want_sclk=2461
    s=$T/rec/summary.txt
    n=$(awk '/^SAMPLES/ {print $2}' "$s")
    # Every column against the fake card's contents (power 123 W, edge 45, hotspot 50, memory 52 C, fan 1234 rpm,
    # busy 37 %, memory busy 12 %, 1024 MiB used; gpu_metrics current 2455 MHz, average 2440, activity 37 %,
    # socket 123 W, 1056 mV, throttle_status 0x2, indep 0x1000000001).
    seen="sclk $(csv_col "$csv" sclk_MHz)lvl $(csv_col "$csv" sclk_lvl)mclk $(csv_col "$csv" mclk_MHz)cur $(csv_col "$csv" gfxclk_cur_MHz)avg $(csv_col "$csv" gfxclk_avg_MHz)"
    seen+="act $(csv_col "$csv" gfx_act_pct)sock $(csv_col "$csv" socket_power_W)mV $(csv_col "$csv" vgfx_mV)thr $(csv_col "$csv" throttle_status)ithr $(csv_col "$csv" indep_throttle_status)"
    seen+="W $(csv_col "$csv" power_W)edge $(csv_col "$csv" edge_C)hot $(csv_col "$csv" hotspot_C)mem $(csv_col "$csv" mem_C)fan $(csv_col "$csv" fan_rpm)busy $(csv_col "$csv" busy_pct)mbusy $(csv_col "$csv" mem_busy_pct)vram $(csv_col "$csv" vram_used_MiB)"
    [[ $seen == "sclk $want_sclk lvl 1 mclk 1258 cur 2455 avg 2440 act 37 sock 123 mV 1056 thr 0x2 ithr 0x1000000001 W 123.0 edge 45.0 hot 50.0 mem 52.0 fan 1234 busy 37 mbusy 12 vram 1024 " ]] || ok=0
    grep -q "^CLOCK_BANDS.* 1.9-2.9 100.0%" "$s" || ok=0
    grep -q "^THROTTLE indep_throttle_status: $n samples with a bit; PPT0(0) $n; TEMP_HOTSPOT(36) $n\$" "$s" || ok=0
    grep -q "^THROTTLE throttle_status (legacy word, bits not named): $n samples with a bit; bit(1) $n\$" "$s" || ok=0
    grep -q "^METRICS_REFRESH distinct tables 1;" "$s" || ok=0
    grep -q "the average column equals it in 0 of $n samples" "$s" || ok=0
    grep -qx "fakegame: 1 client" "$T/rec/clients-start.txt" || ok=0
    grep -q otherdevice "$T/rec/clients-start.txt" "$T/rec/clients-end.txt" && ok=0
    grep -qx "start lact_profile: Testing" "$T/rec/header.txt" || ok=0
    grep -qx "start power_profile_mode_active: 0:BOOTUP_DEFAULT" "$T/rec/header.txt" || ok=0
    grep -qx "start fan_zero_rpm_enable: FAN_ZERO_RPM_ENABLE: | 1 | OD_RANGE: | ZERO_RPM_ENABLE: 0 1" "$T/rec/header.txt" || ok=0
    grep -qx "kernel: 9.9.9-fake" "$T/rec/header.txt" || ok=0
    [[ -e $T/rec/ready ]] || ok=0
    ((rc == 0 && n >= 400 && n <= 501)) || ok=0
    record r01 "fake card read back exactly; sclk $want_sclk, band 1.9-2.9 100 %, PPT0 and TEMP_HOTSPOT and legacy bit 1 in every sample, one metrics table, average column 0 matches, fakegame listed, other device not; ready file; 400-501 samples in 1 s" \
        "$ok" "exit $rc, samples $n; $seen; $(grep -h '^CLOCK_BANDS' "$s" | cut -c1-140)"
}
r02() {
    setup r02
    local s ok=1 hi lo dt
    put "$T/drm/pp_dpm_sclk" $'S: 91Mhz *\n0: 500Mhz\n1: 2460Mhz'
    mk_metrics "$T/drm/gpu_metrics" 91 91 0 0 750 0 40
    # The clock file and the metrics table change on a known schedule: 0.5 s at 3200 MHz, 0.5 s at 1703, again,
    # then asleep; six different metrics tables in all. The schedule starts once the first sample is on disk.
    (
        wait_for 20 test -e "$T/rec/ready" || exit
        sleep 0.4
        for v in 3200 1703 3200 1703; do
            put "$T/drm/pp_dpm_sclk" "S: 91Mhz"$'\n'"0: 500Mhz"$'\n'"1: ${v}Mhz *"
            mk_metrics "$T/drm/gpu_metrics" "$v" "$v" 0 0 "$((800 + v / 100))" 50 100
            sleep 0.5
        done
        put "$T/drm/pp_dpm_sclk" $'S: 92Mhz *\n0: 500Mhz\n1: 2460Mhz'
        mk_metrics "$T/drm/gpu_metrics" 92 92 0 0 751 0 40
    ) &
    local wp=$!
    rec 3 2
    wait "$wp"
    s=$T/rec/summary.txt
    hi=$(grep '^HIGH_RUNS' "$s")
    lo=$(grep '^LOW_RUNS' "$s")
    [[ $hi =~ :\ 2\ times.*longest\ ([0-9.]+)\ ms ]] && dt=${BASH_REMATCH[1]} || ok=0
    awk -v d="${dt:-0}" 'BEGIN {exit !(d >= 430 && d <= 570)}' || ok=0
    [[ $lo =~ :\ 2\ times.*longest\ ([0-9.]+)\ ms ]] && dt=${BASH_REMATCH[1]} || ok=0
    awk -v d="${dt:-0}" 'BEGIN {exit !(d >= 430 && d <= 570)}' || ok=0
    grep -q '^METRICS_REFRESH distinct tables 6;' "$s" || ok=0
    record r02 "a known schedule (2 x 0.5 s above 2.9 GHz, 2 x 0.5 s at 1.7 GHz, six metrics tables): HIGH_RUNS 2 and LOW_RUNS 2, longest 430-570 ms each, 6 distinct tables" \
        "$ok" "$hi | $lo | $(grep '^METRICS_REFRESH' "$s")"
}
r03() {
    setup r03
    local f=$T/made.csv s ok=1 m
    # A constructed recording of 100 samples, 10 ms apart, with one late sample (50 ms gap after row 95):
    # rows 0-19 asleep (91 MHz), 20-49 at 1703, 50-59 at 3200, 60-79 at 1820, 80-89 at 3200, 90-99 at 2460.
    # Hotspot 85 C in rows 50-59, 92 C in 80-84, 81 C in 85-89, else 50. indep 0x1000000000 in rows 50-56,
    # 0x1000000002 in 80-82; legacy 0x2 and 0x8002 in the same rows. Power 47 W asleep, 100 W, 300 W above 2.9 GHz,
    # 350 W in row 85.
    awk 'BEGIN {
        print "# label=made period_ms=10"
        print "t_s,sclk_lvl,sclk_MHz,throttle_status,indep_throttle_status,power_W,hotspot_C"
        t = 0
        for (i = 0; i < 100; i++) {
            s = i < 20 ? 91 : i < 50 ? 1703 : i < 60 ? 3200 : i < 80 ? 1820 : i < 90 ? 3200 : 2460
            h = (i >= 50 && i < 60) ? 85 : (i >= 80 && i < 85) ? 92 : (i >= 85 && i < 90) ? 81 : 50
            it = (i >= 50 && i <= 56) ? "0x1000000000" : (i >= 80 && i <= 82) ? "0x1000000002" : "0x0"
            lt = (i >= 50 && i <= 56) ? "0x2" : (i >= 80 && i <= 82) ? "0x8002" : "0x0"
            p = i < 20 ? 47 : s > 2900 ? 300 : 100
            if (i == 85) p = 350
            printf "%.3f,%s,%d,%s,%s,%d,%d\n", t, (i < 20 ? "S" : "1"), s, lt, it, p, h
            t += (i == 95) ? 0.050 : 0.010
        }
        print "# samples=100 missed_slots=4 stopped_by=time"
    }' >"$f"
    "$WATCH" summarize "$f" >"$T/summary.txt" 2>&1
    s=$T/summary.txt
    # Time shares: the late sample (row 95, 2460 MHz) stands for 50 ms, so the 2460 band is 9x10+50 = 140 ms of
    # 1040 ms; the bands below are therefore shares of 1040 ms, worked out by hand.
    m="asleep <1.0 19.2% 1.0-1.65 0.0% 1.65-1.76 28.8% 1.76-1.9 19.2% 1.9-2.9 13.5% >2.9 19.2%"
    [[ $BREAK == wrongexp ]] && m="asleep <1.0 19.2% 1.0-1.65 0.0% 1.65-1.76 28.8% 1.76-1.9 19.2% 1.9-2.9 13.5% >2.9 25.0%"
    grep -qF "CLOCK_BANDS GHz (share of time, pp_dpm_sclk): $m; unreadable 0.0%" "$s" || ok=0
    grep -qx "SAMPLES 100 over 1.040 s; requested period 10 ms" "$s" || ok=0
    grep -qx "TIMING period_ms median 10.000 p99 10.000 longest_gap 50.000; missed slots 4" "$s" || ok=0
    grep -qx "HIGH_RUNS above 2.9 GHz: 2 times, total 0.200 s, median 100.0 ms, longest 100.0 ms" "$s" || ok=0
    grep -qx "LOW_RUNS awake below 1.9 GHz: 2 times, total 0.500 s, median 300.0 ms, longest 300.0 ms" "$s" || ok=0
    grep -qx "HOTSPOT_C max 92; time above 80 C 0.200 s, above 90 C 0.050 s" "$s" || ok=0
    grep -qx "THROTTLE indep_throttle_status: 10 samples with a bit; PPT1(1) 3; TEMP_HOTSPOT(36) 10" "$s" || ok=0
    grep -qx "THROTTLE throttle_status (legacy word, bits not named): 10 samples with a bit; bit(1) 10; bit(15) 3" "$s" || ok=0
    grep -q "^POWER_W reported .* max 350 (n 100)" "$s" || ok=0
    record r03 "summary of a constructed recording equals the hand-worked values (bands, timing, runs, hotspot times, throttle counts, power max)" \
        "$ok" "$(grep -E '^(CLOCK_BANDS|TIMING|HIGH_RUNS|LOW_RUNS|HOTSPOT_C|THROTTLE indep)' "$s" | tr '\n' ' ' | cut -c1-600)"
}
r04() {
    setup r04
    env DRM_DEVICE="$T/drm" HWMON="$T/hwmon" WATCH_PDEV=0000:99:00.0 WATCH_PROC="$T/proc" PATH="$T/bin:$PATH" SHIM_DIR="$T" \
        "$WATCH" record --out "$T/rec" --label r04 >"$T/out.txt" 2>&1 </dev/null &
    local p=$! rc fds ok=1
    sleep 1
    # The recorder's open files while it runs: only its own output files, nothing under /dev.
    fds=$(for f in /proc/"$p"/fd/*; do readlink "$f"; done 2>/dev/null | grep -v -E "^$T/rec/|^$T/out.txt$|^/dev/null$" | tr '\n' ' ')
    kill -TERM "$p"
    wait "$p"
    rc=$?
    [[ -z $fds ]] || ok=0
    ((rc == 0)) || ok=0
    grep -qx "stopped_by: signal" "$T/rec/header.txt" || ok=0
    grep -q "^SAMPLES" "$T/rec/summary.txt" || ok=0
    [[ -s $T/rec/clients-end.txt ]] || ok=0
    record r04 "no time limit, TERM after 1 s: exit 0, files finished (summary, end conditions, end clients), no device or other file open" \
        "$ok" "exit $rc, other open files: '${fds}', $(grep -h '^stopped_by' "$T/rec/header.txt"), $(grep -h '^SAMPLES' "$T/rec/summary.txt")"
}
r05() {
    setup r05
    local f=$T/made.csv s ok=1
    # Nine samples at 3200 MHz 10 ms apart, then 1703 MHz standing for 1 s (the next sample comes 1 s later), one more
    # at 1703 and one at 3200: time at 1703 1010 ms, at 3200 9 x 10 + 10 (the last sample gets the median gap) =
    # 100 ms, of 1110 ms. Time-weighted: p10, p50 and p90 are 1703, p99 (1098.9 ms > 1010) is 3200. Weighting by
    # samples (10 of 12 at 3200) would give 3200 from p50 up.
    awk 'BEGIN {
        print "# label=weights period_ms=10"
        print "t_s,sclk_MHz"
        for (i = 0; i < 9; i++) printf "%.3f,3200\n", i * 0.01
        print "0.090,1703"; print "1.090,1703"; print "1.100,3200"
    }' >"$f"
    "$WATCH" summarize "$f" >"$T/summary.txt" 2>&1
    s=$T/summary.txt
    grep -qx "SCLK_MHz all (time-weighted) p10 1703 p50 1703 p90 1703 p99 3200 max 3200 (n 12)" "$s" || ok=0
    grep -qx "SCLK_MHz awake (time-weighted) p10 1703 p50 1703 p90 1703 p99 3200 max 3200 (n 12)" "$s" || ok=0
    grep -q "^AWAKE_SHARES (share of awake time): 1.65-1.9 GHz 91.0%, >2.9 GHz 9.0%; awake 100.0% of the time" "$s" || ok=0
    record r05 "clock percentiles weighted by time: 1703 MHz for 1010 of 1110 ms gives p10-p90 1703 and p99 3200 (n 12), shares 91.0 / 9.0 %" \
        "$ok" "$(grep -E '^(SCLK_MHz|AWAKE_SHARES)' "$s" | tr '\n' ' ')"
}
r06() {
    setup r06
    local f=$T/made.csv s ok=1
    # indep_throttle_status 0x0200000000000001 (bit 57 FIT and bit 0 PPT0) in 3 of 4 samples. As a double the word
    # loses bit 0 (2^57 + 1 has 58 significant bits); kept as an integer both bits count.
    awk 'BEGIN {
        print "# label=bits period_ms=10"
        print "t_s,sclk_MHz,indep_throttle_status,metrics_id"
        print "0.00,2460,0x0200000000000001,0x0200000000000001"
        print "0.01,2460,0x0200000000000001,0x0200000000000000"
        print "0.02,2460,0x0200000000000001,0x0200000000000001"
        print "0.03,2460,0x0,0x0200000000000000"
    }' >"$f"
    "$WATCH" summarize "$f" >"$T/summary.txt" 2>&1
    s=$T/summary.txt
    grep -qx "THROTTLE indep_throttle_status: 3 samples with a bit; PPT0(0) 3; FIT(57) 3" "$s" || ok=0
    # Hashes that differ only in bit 0 are four distinct tables in a row.
    grep -q "^METRICS_REFRESH distinct tables 4;" "$s" || ok=0
    record r06 "a 64-bit throttle word with bit 57 set stays exact: PPT0(0) 3 and FIT(57) 3; metrics hashes differing in bit 0 count as 4 tables" \
        "$ok" "$(grep -E '^(THROTTLE indep|METRICS_REFRESH)' "$s" | tr '\n' ' ')"
}
r07() {
    setup r07
    env DRM_DEVICE="$T/drm" HWMON="$T/hwmon" WATCH_PDEV=0000:99:00.0 WATCH_PROC="$T/proc" PATH="$T/bin:$PATH" SHIM_DIR="$T" \
        "$WATCH" record --out "$T/rec" --label r07 >"$T/out.txt" 2>&1 </dev/null &
    local p=$! rc ok=1
    wait_for 10 test -e "$T/rec/ready"
    sleep 0.3
    kill -HUP "$p"
    wait "$p"
    rc=$?
    ((rc == 0)) || ok=0
    grep -qx "stopped_by: signal" "$T/rec/header.txt" || ok=0
    grep -q "^SAMPLES" "$T/rec/summary.txt" || ok=0
    record r07 "HUP (the terminal closed): exit 0, stopped by the signal, summary written" \
        "$ok" "exit $rc, $(grep -h '^stopped_by' "$T/rec/header.txt" 2>/dev/null), $(grep -h '^SAMPLES' "$T/rec/summary.txt" 2>/dev/null)"
}
r08() {
    setup r08
    local args rc all=1 seen="" k=0
    for args in "--seconds" "--seconds 1 --period-ms" "--period-ms nan" "--seconds abc" "--period-ms 2x" "--seconds -1" \
        "--seconds inf" "--period-ms 0.1" "--seconds --period-ms"; do
        k=$((k + 1))
        # shellcheck disable=SC2086  # the option list is split on purpose
        # (A recorder that ignored a bad option would record on: the time limit keeps the test bounded.)
        timeout 5 env DRM_DEVICE="$T/drm" HWMON="$T/hwmon" "$WATCH" record --out "$T/rec$k" $args >>"$T/out.txt" 2>&1 </dev/null
        rc=$?
        seen+="'$args': exit $rc; "
        ((rc == 2)) && [[ ! -e $T/rec$k ]] || all=0
    done
    record r08 "an option with a missing or non-numeric value (nan, inf, 2x, abc, -1, below 0.5 ms): exit 2, no recording made" \
        "$all" "$seen"
}
r09() {
    setup r09
    local csv=$T/rec/samples.csv s ok=1 vals
    # After the first sample, the active line of pp_dpm_sclk turns into garbage: it must read as missing (-1), not 0.
    (
        wait_for 20 test -e "$T/rec/ready" || exit
        put "$T/drm/pp_dpm_sclk" $'S: 91Mhz\n0: 500Mhz\n1: xyzMhz *'
    ) &
    local wp=$!
    rec 1 2
    wait "$wp"
    s=$T/rec/summary.txt
    vals=$(csv_col "$csv" sclk_MHz)
    [[ $vals == "-1 2460 " ]] || ok=0
    grep -q "^CLOCK_BANDS GHz (share of time, pp_dpm_sclk): asleep <1.0 0.0% " "$s" || ok=0
    grep -q "^AWAKE_SHARES" "$s" || ok=0
    awk '/^CLOCK_BANDS/ {for (i = 1; i <= NF; i++) if ($i == "unreadable") {v = $(i + 1); sub(/%/, "", v); exit !(v > 50)}}' "$s" || ok=0
    record r09 "garbage in the active pp_dpm_sclk line after the start: the clock reads as missing (-1), never 0 MHz; asleep share 0.0 %, unreadable above 50 %" \
        "$ok" "sclk values: $vals; $(grep -h '^CLOCK_BANDS' "$s")"
}
r10() {
    setup r10
    local tp t0 dt ok=1 alive=no
    # A load that ignores TERM, led by timeout(1) as in the kit; the memory temperature then crosses 90 C.
    timeout 60 bash -c 'trap "" TERM; sleep 30' &
    tp=$!
    env SHIM_DIR="$T" PATH="$T/bin:$PATH" DRM_DEVICE="$T/drm" HWMON="$T/hwmon" WATCH_PDEV=0000:99:00.0 WATCH_PROC="$T/proc" \
        "$WATCH" record --out "$T/rec" --label r10 --guard-pidfile "$T/load.pid" --stop-hot 95 --stop-mem 90 \
        --guard-log "$T/guard.log" >"$T/out.txt" 2>&1 </dev/null &
    local rp=$!
    wait_for 10 test -e "$T/rec/ready"
    echo "$tp" >"$T/load.pid"
    sleep 0.3
    t0=$EPOCHREALTIME
    put "$T/hwmon/temp3_input" 91000
    wait "$tp"
    dt=$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN {printf "%.1f", b - a}')
    kill -0 "$tp" 2>/dev/null && alive=yes
    kill -TERM "$rp"
    wait "$rp"
    has_line '^THERMAL STOP: memory 91 C (stops at hotspot 95 C, memory 90 C); TERM sent to the load' "$T/guard.log" || ok=0
    has_line '^THERMAL STOP: the load was still running 1 s after TERM; KILL sent to its process group' "$T/guard.log" || ok=0
    awk -v d="$dt" 'BEGIN {exit !(d >= 0.9 && d <= 3)}' || ok=0
    [[ $alive == no ]] || ok=0
    grep -q '^guard: pid file' "$T/rec/header.txt" || ok=0
    record r10 "recorder as guard: memory 91 C (limit 90) gives TERM; a load that ignores TERM gets KILL to its group 1 s later and ends within 0.9-3 s" \
        "$ok" "ended after $dt s, still alive: $alive; guard log: $(tr '\n' ' ' <"$T/guard.log" 2>/dev/null | cut -c1-300)"
}

# ---------------------------------------------------------------------------------------------------------------
# Kit

k01() {
    setup k01
    kit -- power-mode
    local rc=$?
    record k01 "power-mode without the owner's flag: exit 10, no profile set, no probe, instructions printed" \
        $((rc == 10 && $(sets) == 0 && $(probes) == 0 && $(count 'lact cli profile set Home-Power' "$T/out.txt") == 1)) \
        "exit $rc, profile set calls $(sets), probe calls $(probes)"
}
k02() {
    setup k02
    local s rc all=1 seen=""
    for s in perf-high "heat fan100" memclock overclock oc-traces; do
        # shellcheck disable=SC2086
        kit -- $s
        rc=$?
        ((rc == 10)) || all=0
        seen+="$s: exit $rc; "
    done
    ((all && $(sets) == 0 && $(probes) == 0)) || all=0
    record k02 "every other step that changes a setting, without its flag: exit 10, no profile set, no probe" \
        "$all" "${seen}profile set calls $(sets), probe calls $(probes)"
}
k03() {
    setup k03
    kit -- power-mode --owner-ok=power-mode
    local rc=$? a b c ok=1
    a=$(count 'A-testing-hip-ramp.*lock=held mode=BOOTUP_DEFAULT perf=auto' "$T/probe-calls.log")
    b=$(count 'B-Home-Power-hip-ramp.*lock=held mode=3D_FULL_SCREEN perf=manual' "$T/probe-calls.log")
    c=$(count 'C-testing-hip-ramp.*lock=held mode=BOOTUP_DEFAULT perf=auto' "$T/probe-calls.log")
    ((rc == 0 && a == 1 && b == 1 && c == 1 && $(probes) == 3 && $(sets Home-Power) == 1 && $(sets Testing) == 1)) || ok=0
    grep -q '^- restored: profile=Testing cap_W=317 perf=auto mode=BOOTUP_DEFAULT' "$T/out.txt" || ok=0
    grep -q 'Vulkan burst and ramp tests were skipped' "$T/out.txt" || ok=0
    grep -q 'SELF-TEST MODE' "$T/out.txt" || ok=0
    record k03 "power-mode with the flag, the stand-in switches: control, changed (probe sees 3D_FULL_SCREEN), control; every probe under the lock; one set each way; restored; Vulkan tests politely skipped; self-test warning printed" \
        "$ok" "exit $rc, probes A/B/C $a/$b/$c of $(probes), set Home-Power $(sets Home-Power), set Testing $(sets Testing)"
}
k04() {
    setup k04
    kit SHIM_TESTING_KEEPS_MODE=1 -- power-mode --owner-ok=power-mode
    local rc=$?
    record k04 "the switch back leaves the power mode changed: exit 5, RESTORE FAILED naming the fields, two tries, no control round after" \
        $((rc == 5 && $(count 'RESTORE FAILED.*Differs: perf mode$' "$T/out.txt") == 1 && $(sets Testing) == 2 && $(count 'C-testing' "$T/probe-calls.log") == 0)) \
        "exit $rc, RESTORE FAILED lines $(count 'RESTORE FAILED' "$T/out.txt") ($(grep -h 'RESTORE FAILED' "$T/out.txt" | cut -c1-140)), set Testing calls $(sets Testing), C-round probes $(count 'C-testing' "$T/probe-calls.log")"
}
k05() {
    setup k05
    flock "$LOCK" sleep 12 &
    local hp=$! t0 rc dt
    sleep 0.5
    t0=$SECONDS
    kit -- game uncapped --seconds 2
    rc=$?
    dt=$((SECONDS - t0))
    wait "$hp"
    record k05 "game step while another process holds the GPU lock: done in under 8 s (no lock wait), exit 0, no probe, no profile set, recording summarized" \
        $((rc == 0 && dt < 8 && $(probes) == 0 && $(sets) == 0 && $(count '^SAMPLES' "$T"/run/game-uncapped/summary.txt) == 1)) \
        "exit $rc after $dt s, probes $(probes), profile set calls $(sets)"
}
k06() {
    setup k06
    echo Default >"$T/profile"
    kit -- power-mode --owner-ok=power-mode
    local rc=$?
    record k06 "power-mode with the flag but the card not on Testing at the start: exit 3, no set, no probe" \
        $((rc == 3 && $(sets) == 0 && $(probes) == 0)) "exit $rc, profile set calls $(sets), probes $(probes)"
}
k07() {
    setup k07
    kit SHIM_NOAPPLY=1 -- perf-high --owner-ok=perf-high
    local rc=$?
    record k07 "perf-high, the switch does not take (setting unchanged): stop, restored, exit 1, only the first control ran" \
        $((rc == 1 && $(probes) == 1 && $(count 'B-Home-High' "$T/probe-calls.log") == 0 && $(count '^- restored' "$T/out.txt") == 1 && $(sets Home-High) == 1)) \
        "exit $rc, probes $(probes), set Home-High $(sets Home-High), restored lines $(count '^- restored' "$T/out.txt")"
}
# k_signal <test> <signal> <expected exit>: the signal arrives during the changed round's load.
k_signal() {
    setup "$1"
    local sig=$2 want_rc=$3 rc alive=no pid
    kit_bg FAKE_PROBE_S=6 -- power-mode --owner-ok=power-mode
    wait_for 40 has_line 'B-Home-Power' "$T/probe-calls.log"
    sleep 0.5
    kill "-$sig" "$BG_PID"
    wait "$BG_PID"
    rc=$?
    pid=$(awk '/B-Home-Power/ {print $1}' "$T/probe-calls.log")
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && alive=yes
    record "$1" "$sig during the changed round's load: the load is ended, exit $want_rc, restored, no further round" \
        $((rc == want_rc && $(count '^- restored' "$T/out.txt") == 1 && $(count 'C-testing' "$T/probe-calls.log") == 0)) \
        "exit $rc, probe still running: $alive, restored lines $(count '^- restored' "$T/out.txt"), set Testing $(sets Testing)"
    [[ $alive == no ]] || { FAILS=$((FAILS + 1)); echo "FAIL ${1}b | the probe outlived the script" | tee -a "$RES"; }
}
k08() { k_signal k08 TERM 130; }
k09() {
    setup k09
    put "$T/meminfo" "MemAvailable:    4194304 kB"
    kit -- short-tests lowmem
    local rc=$?
    record k09 "short-tests with 4 GB available: not started, exit 1, no probe" \
        $((rc == 1 && $(probes) == 0 && $(count 'below 8 GB' "$T/out.txt") >= 1)) "exit $rc, probes $(probes)"
}
k10() {
    setup k10
    # 1 s after the probe has started, the fake hotspot jumps to 96 C.
    (
        wait_for 60 test -s "$T/probe-calls.log" || exit
        sleep 1
        put "$T/hwmon/temp2_input" 96000
    ) &
    local hp=$! t0=$SECONDS rc dt
    kit FAKE_PROBE_S=8 -- short-tests hotstop
    rc=$?
    dt=$((SECONDS - t0))
    wait "$hp"
    record k10 "hotspot reaches 96 C during a load: THERMAL STOP from the recorder, the load ends early, no further load, exit 1" \
        $((rc == 1 && dt < 7 && $(probes) == 1 && $(count '^THERMAL STOP: hotspot 96 C (stops at hotspot 95 C, memory 90 C); TERM' "$T"/run/short-hotstop/*.log) == 1 && $(count 'STOPPED: hotspot 96 C' "$T/out.txt") == 1)) \
        "exit $rc after $dt s, probes $(probes), thermal stop lines $(count '^THERMAL STOP' "$T"/run/short-hotstop/*.log)"
}
k11() {
    setup k11
    mkdir -p "$T/spv"
    kit HOME_VKGAME="$T/bin/fakeprobe" HOME_VKSPV="$T/spv" -- short-tests vk
    local rc=$?
    record k11 "short-tests with the Vulkan probes given: HIP ramp, Vulkan ramp, Vulkan 60/s bursts, each under the lock" \
        $((rc == 0 && $(probes) == 3 && $(count 'burst name=vk-vk-ramp,q=g,groups=64,local=32,every=64,recstride=8,cont=60,idle=1 | lock=held' "$T/probe-calls.log") == 1 && $(count 'blen=2,cad=16.7,dur=1,idle=1 | lock=held' "$T/probe-calls.log") == 1)) \
        "exit $rc, probes $(probes)"
}
k12() {
    setup k12
    kit HOME_SPACING_S=1 -- heat normal
    local rc=$? parts
    parts=$(find "$T/run/heat-normal" -name '*-sysfs.csv' | wc -l)
    record k12 "heat normal: FP16 and FP8 traces, then light and stream as three 20 s parts each, part lengths reported; all 8 probe runs under the lock; no profile set" \
        $((rc == 0 && $(probes) == 8 && $(count 'lock=held' "$T/probe-calls.log") == 8 && parts == 8 && $(sets) == 0 && $(count 'f16 --max-s 20 --pre-s 1 --post-s 10' "$T/probe-calls.log") == 1 && $(count 'light --max-s 20 --pre-s 0 --post-s 0' "$T/probe-calls.log") == 3 && $(count 'stream60: the parts took [0-9.]* s, [0-9.]* s, [0-9.]* s (20 s of load each' "$T/run/summary.md") == 1)) \
        "exit $rc, probes $(probes), under the lock $(count 'lock=held' "$T/probe-calls.log"), part files $parts, profile set calls $(sets); $(grep -h 'stream60: the parts' "$T/run/summary.md" | cut -c1-160)"
}
k13() {
    setup k13
    kit HOME_SPACING_S=1 -- heat fan100 --owner-ok=fan100
    local rc=$?
    record k13 "heat fan100 with the flag: switch to Home-Fan100 (flat 100 % curve, zero RPM off), 8 probe runs, switch back, restored with zero RPM on" \
        $((rc == 0 && $(probes) == 8 && $(sets Home-Fan100) == 1 && $(sets Testing) == 1 && $(count '^- restored: .*fan_zero_rpm=1 ' "$T/out.txt") == 1 && $(count 'fan_curve=55C100%,58C100%,62C100%,66C100%,73C100% fan_zero_rpm=0 ' "$T/run/summary.md") >= 1)) \
        "exit $rc, probes $(probes), set Home-Fan100 $(sets Home-Fan100), set Testing $(sets Testing), restored $(count '^- restored' "$T/out.txt")"
}
k14() {
    setup k14
    put "$T/hwmon/power1_average" 50000000
    # Meter readings typed: idle 100, stream 300, FP16 450 W. Driver: 50, 200, 300 W (fake).
    # Expected ratios: (300 - 100) / (200 - 50) = 1.33 and (450 - 100) / (300 - 50) = 1.40.
    kit_env
    env "${KENV[@]}" HOME_SPACING_S=1 FAKE_PROBE_S=3 FAKE_POWER_MAP="stream=200 f16=300" \
        "$SCRIPT" wall-meter >"$T/out.txt" 2>&1 < <(printf '100\n300\n450\n')
    local rc=$?
    record k14 "wall-meter with typed readings: driver medians 50, 200, 300 W; ratios 1.33 (stream) and 1.40 (FP16)" \
        $((rc == 0 && $(count 'idle: meter 100 W, driver.s reading (median) 50 W' "$T/out.txt") == 1 && $(count 'stream minus idle: meter 200 W, driver 150 W, ratio 1.33' "$T/out.txt") == 1 && $(count 'f16 minus idle: meter 350 W, driver 250 W, ratio 1.40' "$T/out.txt") == 1)) \
        "exit $rc; $(grep -E 'minus idle' "$T/out.txt" | cut -c1-80 | tr '\n' ' ')"
}
k15() {
    setup k15
    kit HOME_SPACING_S=1 -- oc-traces --owner-ok=oc-traces
    local rc=$?
    record k15 "oc-traces with the flag: switch to Default (374 W, +150, -50 mV read back), light, stream, FP16, FP8 each 20 s under the held lock, back to Testing, restored" \
        $((rc == 0 && $(probes) == 4 && $(count 'lock=held' "$T/probe-calls.log") == 4 && $(sets Default) == 1 && $(sets Testing) == 1 && $(count '^- restored' "$T/out.txt") == 1 && $(count 'cap_W=374 .*sclk_offset=150 vddgfx_offset=-50 mclk_request=1409' "$T/run/summary.md") >= 1)) \
        "exit $rc, probes $(probes), set Default $(sets Default), set Testing $(sets Testing), restored $(count '^- restored' "$T/out.txt")"
}
k16() {
    setup k16
    # check on Testing records the settings; then the owner chooses Home-Mem by hand (request 1409 MHz).
    kit -- check
    echo Home-Mem >"$T/profile"
    mk_od "$T/drm/pp_od_clk_voltage" 0 1409 0
    kit -- memclock --owner-ok=memclock
    local rc=$?
    record k16 "memclock with the flag on Home-Mem (request 1409) after check on Testing: running level read (1258), 10 s stream and 3 read tests under the lock, switch back to Testing (request 1259 again), restored" \
        $((rc == 0 && $(probes) == 4 && $(count 'lock=held' "$T/probe-calls.log") == 4 && $(sets Testing) == 1 && $(count 'request 1409 MHz; running memory clock 1258 MHz (gpu_metrics)' "$T/run/summary.md") == 1 && $(count '^- restored: .*mclk_request=1259' "$T/out.txt") == 1)) \
        "exit $rc, probes $(probes), set Testing $(sets Testing), restored $(count '^- restored' "$T/out.txt")"
}
# k_fan_left <test> <what>: heat fan100 with a stand-in whose switch back leaves a fan setting static.
k17() {
    setup k17
    kit HOME_SPACING_S=1 SHIM_TESTING_KEEPS_ZERO=1 -- heat fan100 --owner-ok=fan100
    local rc=$?
    record k17 "fan 100 % the way LACT does it here (flat curve, zero RPM off); the switch back rewrites the curve but leaves zero RPM off: exit 5, RESTORE FAILED naming fan_zero_rpm, two tries" \
        $((rc == 5 && $(count 'RESTORE FAILED.*Differs: fan_zero_rpm$' "$T/out.txt") == 1 && $(sets Testing) == 2)) \
        "exit $rc, $(grep -h 'RESTORE FAILED' "$T/out.txt" | cut -c1-160), set Testing $(sets Testing)"
}
k18() {
    setup k18
    put "$T/hwmon/pwm1_enable" 2
    kit HOME_SPACING_S=1 SHIM_FAN_HWMON=1 SHIM_TESTING_KEEPS_FAN=1 -- heat fan100 --owner-ok=fan100
    local rc=$? ok=1
    ((rc == 5 && $(count 'RESTORE FAILED.*Differs: pwm1_enable pwm1$' "$T/out.txt") == 1 && $(probes) == 8)) || ok=0
    record k18 "fan 100 % through hwmon (pwm1_enable 1, pwm1 255, on a card that has pwm1_enable), left static after the switch back: the loads ran, then exit 5, RESTORE FAILED naming pwm1_enable and pwm1" \
        "$ok" "exit $rc, probes $(probes), $(grep -h 'RESTORE FAILED' "$T/out.txt" | cut -c1-160)"
}
k19() {
    setup k19
    local rc1 rc2 rc3 ok=1
    # A first copy runs short-tests with a 4 s load; while it runs, a second short-tests and a game step are started,
    # and check reports on it.
    kit_bg FAKE_PROBE_S=4 -- short-tests first
    wait_for 20 test -s "$T/probe-calls.log"
    KIT_OUT=$T/out-second.txt kit -- short-tests second
    rc2=$?
    KIT_OUT=$T/out-game.txt kit -- game during --seconds 1
    rc3=$?
    KIT_OUT=$T/out-check.txt kit -- check
    wait "$BG_PID"
    rc1=$?
    ((rc1 == 0 && rc2 == 6 && rc3 == 6 && $(probes) == 1)) || ok=0
    grep -q 'another step of this kit is running' "$T/out-second.txt" || ok=0
    grep -q 'another step of the kit is running now: YES' "$T/out-check.txt" || ok=0
    [[ -e $T/run/short-second || -e $T/run/game-during ]] && ok=0
    record k19 "a second copy while a step runs: short-tests and game refused (exit 6, nothing run), check says a step is running; the first copy ends normally" \
        "$ok" "first exit $rc1, second exit $rc2, game exit $rc3, probes $(probes), check: $(grep -h 'running now' "$T/out-check.txt")"
}
k20() {
    setup k20
    local rc hp ok=1
    # The overclock traces stop at 88 C. During the first load the fake hotspot reads 89 C; once the stop is logged
    # the card "cools" to 50 C, so that only the stop itself can end the step's remaining loads.
    (
        wait_for 60 test -s "$T/probe-calls.log" || exit
        sleep 0.5
        put "$T/hwmon/temp2_input" 89000
        wait_for 60 eval "grep -qs '^THERMAL STOP' '$T'/run/oc-traces/*.log" || exit
        put "$T/hwmon/temp2_input" 50000
    ) &
    hp=$!
    kit HOME_SPACING_S=1 FAKE_PROBE_S=4 -- oc-traces --owner-ok=oc-traces
    rc=$?
    kill "$hp" 2>/dev/null
    wait "$hp" 2>/dev/null
    ((rc == 1 && $(probes) == 1 && $(sets Default) == 1 && $(sets Testing) == 1 && $(count '^- restored' "$T/out.txt") == 1)) || ok=0
    grep -q 'STOPPED: oc-light: hotspot 89 C (stops at hotspot 88 C, memory 90 C); TERM sent' "$T/out.txt" || ok=0
    record k20 "THERMAL STOP at 88 C in the first overclock trace: the step's remaining three loads do not run, the switch back is done, exit 1" \
        "$ok" "exit $rc, probes $(probes), set Default $(sets Default), set Testing $(sets Testing), restored $(count '^- restored' "$T/out.txt"); $(grep -h 'STOPPED' "$T/out.txt" | head -2 | cut -c1-140 | tr '\n' ' ')"
}
k21() {
    setup k21
    local rc t0 dt hp ok=1
    # A probe that ignores TERM; 1 s after it starts the hotspot reads 96 C.
    (
        wait_for 60 test -s "$T/probe-calls.log" || exit
        sleep 1
        put "$T/hwmon/temp2_input" 96000
    ) &
    hp=$!
    t0=$SECONDS
    kit FAKE_PROBE_S=40 FAKE_IGNORE_TERM=1 -- short-tests kill
    rc=$?
    dt=$((SECONDS - t0))
    wait "$hp"
    ((rc == 1 && dt <= 10 && $(probes) == 1)) || ok=0
    has_line '^THERMAL STOP: the load was still running 1 s after TERM; KILL sent' "$T"/run/short-kill/*.log || ok=0
    grep -q 'KILL was sent' "$T/run/summary.md" || ok=0
    record k21 "a load that ignores TERM at a THERMAL STOP: KILL to its process group 1 s later, the step ends within 10 s (not the 20 s kill-after of timeout), exit 1" \
        "$ok" "exit $rc after $dt s, probes $(probes), $(grep -h 'KILL' "$T"/run/short-kill/*.log | cut -c1-120)"
}
k22() {
    setup k22
    kit SHIM_EXTRA_MCLK=1 -- power-mode --owner-ok=power-mode
    local rc=$? ok=1
    ((rc == 1 && $(probes) == 1 && $(count 'B-Home-Power' "$T/probe-calls.log") == 0 && $(sets Testing) == 1 && $(count '^- restored' "$T/out.txt") == 1)) || ok=0
    grep -q 'Home-Power differs from the start settings in more than intended: mclk_request\.' "$T/out.txt" || ok=0
    record k22 "Home-Power also changes the memory request (a field the power-mode check does not look at): stop naming mclk_request, switched back and restored, no load on it, exit 1" \
        "$ok" "exit $rc, probes $(probes), set Testing $(sets Testing), $(grep -h 'more than intended' "$T/out.txt" | cut -c1-140)"
}
k23() {
    setup k23
    printf 'Default\nTesting\n' >"$T/profiles"
    kit -- heat fan100 --owner-ok=fan100
    local rc=$?
    record k23 "heat fan100 with the flag but no profile Home-Fan100 in LACT: exit 3, no set, no probe, the missing name said" \
        $((rc == 3 && $(sets) == 0 && $(probes) == 0 && $(count "no profile named 'Home-Fan100'" "$T/out.txt") == 1)) \
        "exit $rc, profile set calls $(sets), probes $(probes)"
}
k24() {
    setup k24
    local ok=1 o1=$T/out-real.txt o2=$T/out-selftest.txt rc
    # Stale test hooks in the environment without HOME_SELFTEST: config (read-only, starts nothing, makes no folder)
    # shows what a real run would use.
    env SHIM_DIR="$T" PATH="$T/bin:$PATH" CARD_ENV="$T/card.env" HOME_GPU_LOCK="$T/stale.lock" HOME_MEMINFO="$T/meminfo" \
        HOME_RUN_DIR="$T/run" HOME_BIN_TRACE="$T/bin/fakeprobe" HOME_PROFILE_POWER=Stale "$SCRIPT" config >"$o1" 2>&1 </dev/null
    local want_lock=$REPO/runs/gpu.lock
    [[ -r $REPO/local.env ]] && want_lock=$(. "$REPO/local.env" >/dev/null 2>&1; echo "${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}")
    grep -qx "GPU lock: $want_lock" "$o1" || ok=0
    grep -qx "memory figures: /proc/meminfo" "$o1" || ok=0
    grep -qx "run folder: $REPO/runs/<UTC time>-home" "$o1" || ok=0
    grep -q "^probes: $REPO/build/sched_launch $REPO/build/sched_trace " "$o1" || ok=0
    grep -q "ignored (test hooks.*: HOME_RUN_DIR HOME_GPU_LOCK HOME_MEMINFO HOME_BIN_TRACE HOME_PROFILE_POWER$" "$o1" || ok=0
    grep -qx "self-test mode: no" "$o1" || ok=0
    grep -q "card drm (PCI 1002:7550) from $T/card.env" "$o1" || ok=0
    kit HOME_GPU_LOCK="$T/stale.lock" -- config
    cp "$T/out.txt" "$o2"
    grep -qx "GPU lock: $T/stale.lock" "$o2" || ok=0
    grep -q "SELF-TEST MODE" "$o2" || ok=0
    # A card record that points at another card: refused before anything runs.
    put "$T/drm/device" 0x744c
    KIT_OUT=$T/out-wrongcard.txt kit -- check
    rc=$?
    ((rc == 2)) || ok=0
    grep -q "PCI '0x1002:0x744c', not the RX 9070 XT" "$T/out-wrongcard.txt" || ok=0
    record k24 "stale test hooks without HOME_SELFTEST=1 are ignored and named (real lock, /proc/meminfo, real run folder and probes); with it they are read and a warning printed; a record pointing at PCI 1002:744c is refused (exit 2)" \
        "$ok" "$(grep -E '^(GPU lock|memory figures|self-test)' "$o1" | tr '\n' ' '); wrong card exit $rc"
}
k25() { k_signal k25 INT 130; }
k26() { k_signal k26 HUP 129; }
k27() {
    setup k27
    # The recorder's start-up takes 2 s here (its lact call): the load must still start after the first sample.
    kit WATCH_LACT="sleep 2; lact cli profile get" -- short-tests ready
    local rc=$? rows
    rows=$(sed -n 's/.*rec_rows=\([0-9a-z]*\).*/\1/p' "$T/probe-calls.log")
    record k27 "a recorder slow to start (2 s): the load starts only after the first sample is on disk (rec_rows >= 1 at the probe's start)" \
        $((rc == 0 && $(probes) == 1)) "exit $rc, samples in the recording when the probe started: $rows"
    if ! [[ $rows =~ ^[0-9]+$ ]] || ((rows < 1)); then
        FAILS=$((FAILS + 1))
        echo "FAIL k27b | the probe started before the first sample ($rows)" | tee -a "$RES"
    fi
}
k28() {
    setup k28
    kit HOME_BIN_TRACE="$T/bin/nothere" -- heat fan100 --owner-ok=fan100
    local rc=$?
    record k28 "heat fan100 with the flag but the trace program missing: exit 2 before any switch (no set, no probe)" \
        $((rc == 2 && $(sets) == 0 && $(probes) == 0 && $(count 'is missing' "$T/out.txt") >= 1)) "exit $rc, profile set calls $(sets), probes $(probes)"
}
k29() {
    setup k29
    local rc1 rc2 ok=1
    # memclock without a record of Testing in this sitting: refused. Then with the record, a Home-Mem that also raises
    # the power cap: switched back, no load.
    echo Home-Mem >"$T/profile"
    mk_od "$T/drm/pp_od_clk_voltage" 0 1409 0
    kit -- memclock --owner-ok=memclock
    rc1=$?
    echo Testing >"$T/profile"
    mk_od "$T/drm/pp_od_clk_voltage" 0 1259 0
    kit -- check
    echo Home-Mem >"$T/profile"
    mk_od "$T/drm/pp_od_clk_voltage" 0 1409 0
    put "$T/hwmon/power1_cap" 374000000
    kit -- memclock --owner-ok=memclock
    rc2=$?
    ((rc1 == 3 && rc2 == 1 && $(probes) == 0 && $(sets Testing) == 1 && $(count '^- restored' "$T/out.txt") == 1)) || ok=0
    grep -q 'Home-Mem differs from Testing in more than the memory request: cap_W\.' "$T/out.txt" || ok=0
    record k29 "memclock without a Testing record: exit 3; with one, a Home-Mem that also sets 374 W: switched back (restored), no load, exit 1" \
        "$ok" "exits $rc1 and $rc2, probes $(probes), set Testing $(sets Testing), $(grep -h 'more than the memory' "$T/out.txt" | cut -c1-120)"
}
k30() {
    setup k30
    local p rc ok=1
    # The terminal closes during a game recording: HUP to the whole process group (script and recorder).
    kit_env
    setsid env "${KENV[@]}" "$SCRIPT" game hup --seconds 30 >"$T/out.txt" 2>&1 </dev/null &
    p=$!
    wait_for 20 test -e "$T/run/game-hup/ready"
    sleep 0.5
    kill -HUP -- "-$p"
    wait "$p"
    rc=$?
    ((rc == 0)) || ok=0
    grep -qx "stopped_by: signal" "$T/run/game-hup/header.txt" || ok=0
    grep -q "^SAMPLES" "$T/run/game-hup/summary.txt" || ok=0
    grep -q "^- game hup (.*recorder exit 0)" "$T/run/summary.md" || ok=0
    record k30 "HUP to the game step's process group (terminal closed): the recorder finishes its files and the session summary still gets the game line" \
        "$ok" "exit $rc; $(grep -h '^stopped_by' "$T/run/game-hup/header.txt" 2>/dev/null); $(grep -h '^- game hup' "$T/run/summary.md" 2>/dev/null)"
}
k31() {
    setup k31
    local rc alive=yes pid ok=1
    # SIGKILL to the script during the changed round's load: no trap can run, so the switch back cannot happen (the
    # sheet says so), but the load and the recorder must not outlive the script.
    kit_bg FAKE_PROBE_S=30 -- power-mode --owner-ok=power-mode
    wait_for 40 has_line 'B-Home-Power' "$T/probe-calls.log"
    sleep 0.5
    kill -KILL "$BG_PID"
    wait "$BG_PID"
    rc=$?
    pid=$(awk '/B-Home-Power/ {print $1}' "$T/probe-calls.log")
    wait_for 3 eval "! kill -0 '$pid' 2>/dev/null" && alive=no
    # Code that fails this test leaves the probe running: end its process group (timeout leads it) here.
    if [[ $alive == yes ]]; then
        local pg
        pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
        [[ $pg =~ ^[0-9]+$ ]] && ((pg > 1)) && kill -KILL -- "-$pg" 2>/dev/null
    fi
    ((rc == 137)) || ok=0
    [[ $alive == no ]] || ok=0
    [[ $(cat "$T/profile") == Home-Power && $(count '^- restored' "$T/out.txt") == 0 ]] || ok=0
    # The recorder finishes its files once it gets the TERM.
    wait_for 5 grep -qx 'stopped_by: signal' "$T/run/power-mode/rec-B-Home-Power-hip-ramp/header.txt" || ok=0
    record k31 "SIGKILL to the script during the changed round's load: the load ends within 3 s (parent-death TERM), the recorder ends too; the card stays on Home-Power and nothing claims a restore (documented limit)" \
        "$ok" "exit $rc, probe still running: $alive, profile now $(cat "$T/profile"), restored lines $(count '^- restored' "$T/out.txt")"
}
# fake_proc <pid> <comm> <client: yes|no> <command line words...>: one process in the fake /proc (command line words
# NUL-separated as in /proc; a client has an fdinfo entry for the card).
fake_proc() {
    local pid=$1 comm=$2 client=$3
    shift 3
    mkdir -p "$T/proc/$pid/fdinfo"
    put "$T/proc/$pid/comm" "$comm"
    printf '%s\0' "$@" >"$T/proc/$pid/cmdline"
    [[ $client == yes ]] && printf 'drm-driver:\tamdgpu\ndrm-pdev:\t0000:99:00.0\ndrm-client-id:\t%s\n' "$pid" >"$T/proc/$pid/fdinfo/7"
    return 0
}
# The Wine side of a Proton game as seen on the machine (2026-11-18): no process named like the game.
wine_helpers() {
    fake_proc 5003 services.exe no 'C:\windows\system32\services.exe'
    fake_proc 5004 winedevice.exe no 'C:\windows\system32\winedevice.exe'
    fake_proc 5005 svchost.exe no 'C:\windows\system32\svchost.exe' -k LocalServiceNetworkRestricted
    fake_proc 5006 tabtip.exe no 'C:\windows\system32\tabtip.exe'
    fake_proc 5007 xalia.exe yes 'C:\windows\system32\xalia.exe'
}
k32() {
    local case rc seen="" ok=1 want_rc
    # Each case: a fresh fake /proc, short-tests (must not start beside a game) and check (must say so).
    for case in lead gamethread backslash proton none; do
        setup "k32-$case"
        wine_helpers
        case $case in
        lead)
            # The lead's observation: the game is the card client GameThread; "Cyberpunk 2077" is only in steam.exe's
            # command line.
            fake_proc 5001 GameThread yes 'Z:\game.exe'
            fake_proc 5002 steam.exe no 'c:\windows\system32\steam.exe' '/var/mnt/games/SteamLibrary/steamapps/common/Cyberpunk 2077/REDprelauncher.exe' ;;
        gamethread) fake_proc 5001 GameThread yes 'Z:\game.exe' ;;
        backslash) fake_proc 5001 MainThrd yes 'S:\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe' ;;
        proton) fake_proc 5008 wineserver no '/home/u/.local/share/Steam/compatibilitytools.d/GE-Proton11-11/files/bin/wineserver' ;;
        none) ;;
        esac
        kit -- short-tests "$case"
        rc=$?
        KIT_OUT=$T/out-check.txt kit -- check
        want_rc=7
        [[ $case == none ]] && want_rc=0
        if [[ $case == none ]]; then
            ((rc == 0 && $(probes) == 1)) && grep -q '^a game is running: no$' "$T/out-check.txt" || ok=0
        else
            ((rc == 7 && $(probes) == 0)) && grep -q '^a game is running: YES' "$T/out-check.txt" || ok=0
        fi
        seen+="$case: exit $rc (want $want_rc), probes $(probes), $(grep -h '^a game is running' "$T/out-check.txt" | cut -c1-90); "
    done
    grep -q 'card client 5001 is named GameThread' "$OUT/k32-lead/out.txt" || ok=0
    grep -q 'process 5002 (steam.exe) has a game or Proton path' "$OUT/k32-lead/out.txt" || ok=0
    grep -q 'card client 5001 (MainThrd) runs from a game or Proton folder' "$OUT/k32-backslash/out.txt" || ok=0
    grep -q 'process 5008 (wineserver) has a game or Proton path' "$OUT/k32-proton/out.txt" || ok=0
    T=$OUT/k32-lead
    record k32 "a Proton game with no process named like it: short-tests refused (exit 7, no load) and check says YES for the lead's layout (GameThread client, the path only in steam.exe), a GameThread client alone, a client running from S:\\steamapps\\common, a Proton wineserver alone; Wine helpers alone count as no game (the step runs)" \
        "$ok" "$seen"
}

# ---------------------------------------------------------------------------------------------------------------
# Upscaler log reader: made-up logs in the line format of the real log (CRLF endings, the [time thread] prefix, the
# timing line's fields as the upscaler project describes them; numbers and names made up).

tline() {
    # tline <time> <pid> <pass> <max> <stages> <frames> <total> <resume> <cpu>
    printf '[%s %s] [amd_fidelityfx_dx12]   U1 upscaler info: timing, last 300 dispatches: network GPU %s (max %s) ms [%s] (%s frames); TOTAL(t3-t0) %s (max 9.000) ms; D3D12 input pass 0.000 (max 0.000) ms, output pass 0.000 (max 0.000) ms; input pass end -> resume %s (max 9.100) ms, > 0.1 ms in %s of %s frames, of it not network 0.004 (max 0.005) ms; CPU dispatch %s (max 0.900) ms (split n/a, network enqueue 0.100 (max 0.800))\r\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$6" "$6" "$9"
}
K_ST() { printf 'exposure 0.010, input %s, network %s, output %s' "$1" "$2" "$3"; }
M_ST() { printf 'stage1 0.300, stage2 %s, stage3 %s, stage4 0.200' "$1" "$2"; }
mk_log() {
    local f=$1
    {
        # Session 1: model 2. There is no "network model" line for model 2; the host line is printed for model 1 too.
        printf '[20:00:00.000 100] [amd_fidelityfx_dx12] ==== session start: Cyberpunk2077.exe (process 100) ====\r\n'
        printf '[20:00:01.000 100] [amd_fidelityfx_dx12] settings: model=2 (ini), sr_fallback_to_amd=1 (default)\r\n'
        printf '[20:00:01.500 100] [amd_fidelityfx_dx12] #3 the host chose model 2: settings: model=2 (menu)\r\n'
        printf '[20:00:02.100 100] [amd_fidelityfx_dx12]   U1 upscaler info: network kernels: io=direct_textures model=2 selfcheck(unchecked) approx=a1\r\n'
        tline 20:00:55.000 100 2.400 2.800 "$(M_ST 1.500 0.400)" 300 2.410 2.405 0.050
        tline 20:01:00.000 100 2.500 2.900 "$(M_ST 1.600 0.400)" 300 2.510 2.505 0.050
        tline 20:01:05.000 100 2.600 3.000 "$(M_ST 1.680 0.420)" 300 2.610 2.605 0.050
        # Session 2: model 1.
        printf '[21:00:00.000 200] [amd_fidelityfx_dx12] ==== session start: Cyberpunk2077.exe (process 200) ====\r\n'
        printf '[21:00:01.000 200] [amd_fidelityfx_dx12] settings: model=2 (ini), sr_fallback_to_amd=1 (default)\r\n'
        printf '[21:00:01.500 200] [amd_fidelityfx_dx12] #3 the host chose model 2: settings: model=2 (menu)\r\n'
        printf '[21:00:02.000 200] [amd_fidelityfx_dx12]   U1 upscaler info: network model 1 (runtime option model=1), HDR colour\r\n'
        printf '[21:00:02.100 200] [amd_fidelityfx_dx12]   U1 upscaler info: network kernels: io=direct_textures model=1 network=fused selfcheck hidden=fused\r\n'
        tline 21:01:00.000 200 1.610 1.700 "$(K_ST 0.310 0.750 0.540)" 300 1.620 1.615 0.040
    } >"$f"
    # Offset where the recording's window starts (after the first model 1 timing line).
    stat -c %s "$f"
    {
        tline 21:01:05.000 200 2.900 3.400 "$(K_ST 0.560 1.360 0.968)" 298 2.910 2.905 0.041
        tline 21:01:10.000 200 3.100 3.600 "$(K_ST 0.600 1.450 1.038)" 300 3.110 3.105 0.042
        tline 21:01:15.000 200 2.700 3.200 "$(K_ST 0.520 1.250 0.918)" 300 2.710 2.705 0.043
    } >>"$f"
    stat -c %s "$f"
    tline 21:01:20.000 200 1.590 1.650 "$(K_ST 0.310 0.740 0.530)" 300 1.600 1.595 0.040 >>"$f"
}
# mk_game_dir <dir> <local start HH:MM:SS> <local end HH:MM:SS>: a made-up game recording folder.
mk_game_dir() {
    mkdir -p "$1"
    # The recorder writes UTC; the log's times are local, so the made-up times are given in local time here.
    printf 'label: game-test\nstart_utc: %s\nend_utc: %s\n' "$(date -u -d "@$(date -d "today $2" +%s)" +%FT%T.000Z)" \
        "$(date -u -d "@$(date -d "today $3" +%s)" +%FT%T.000Z)" >"$1/header.txt"
    printf 'AWAKE_SHARES (share of awake time): 1.65-1.9 GHz 80.0%%, >2.9 GHz 5.0%%; awake 99.0%% of the time\nCLOCK_BANDS GHz (share of time, pp_dpm_sclk): made up\n' >"$1/summary.txt"
}
u01() {
    setup u01
    local o a b mean="2.900" ok=1
    [[ $BREAK == upscaler ]] && mean="2.000"
    { read -r a; read -r b; } < <(mk_log "$T/game.log")
    mk_game_dir "$T/run/game-x" 21:01:03 21:01:16
    printf 'path: %s\nsize_start: %s\nsize_end: %s\n' "$T/game.log" "$a" "$b" >"$T/run/game-x/upscaler-log.txt"
    kit -- upscaler "$T/run/game-x"
    local rc=$?
    o=$T/run/game-x/upscaler-read.txt
    # The window holds the lines of 21:01:05, :10 and :15; the first is dropped; 3.100 and 2.700 remain (mean 2.900,
    # median of two by the reader's rule: the lower, 2.700).
    grep -q '^UPSCALER session 2 of 2; lines appended to the log during the recording' "$o" || ok=0
    grep -q '^UPSCALER dropped the first timing line of the recording (21:01:05, pass 2.900 ms)' "$o" || ok=0
    grep -qx 'UPSCALER model 1 (stage names exposure, input, network, output in all 2 lines); selfcheck (self-check passed)' "$o" || ok=0
    [[ $(count '^UPSCALER_LINE' "$o") == 2 ]] || ok=0
    grep -qx 'UPSCALER_LINE 21:01:10 model 1 pass_ms 3.100 (max 3.600) frames 300 stages \[exposure 0.010, input 0.600, network 1.450, output 1.038\] total_ms 3.110 (max 9.000) resume_ms 3.105 (max 9.100), > 0.1 ms in 300 of 300 frames cpu_dispatch_ms 0.042 (max 0.900)' "$o" || ok=0
    grep -qx "UPSCALER pass_ms over 2 lines (the first dropped): mean of means $mean, median 2.700, lowest 2.700, highest 3.100" "$o" || ok=0
    grep -q '^ *AWAKE_SHARES (share of awake time): 1.65-1.9 GHz 80.0%' "$o" || ok=0
    grep -q "upscaler game-x: session 2 of 2; lines appended .* | dropped the first .* | model 1 .* | pass_ms over 2 lines (the first dropped): mean of means $mean" "$T/run/summary.md" || ok=0
    record u01 "log window from the recording's byte offsets: session 2, first line dropped and said, model 1 from the stage names, per line pass/max/stages/TOTAL/resume/CPU dispatch, mean $mean over the 2 remaining; the summary line carries session, drop, model and result" \
        "$((ok && rc == 0))" "exit $rc; $(grep -E '^UPSCALER (session|model|pass_ms)' "$o" | tr '\n' ' ')"
}
u02() {
    setup u02
    local o ok=1
    mk_log "$T/game.log" >/dev/null
    mk_game_dir "$T/run/game-y" 20:00:58 20:01:02
    kit -- upscaler "$T/run/game-y" --log "$T/game.log"
    local rc=$?
    o=$T/run/game-y/upscaler-read.txt
    # Window 20:00:58 to 20:01:02 plus 5 s: the model 2 lines of 20:01:00 (dropped) and 20:01:05.
    grep -q '^UPSCALER session 1 of 2; lines written during the recording by time of day (the log has no date' "$o" || ok=0
    grep -qx 'UPSCALER model 2 (stage names stage1, stage2, stage3, stage4 in all 1 lines); selfcheck(unchecked): the self-check did not run' "$o" || ok=0
    grep -q '^UPSCALER dropped the first timing line of the recording (20:01:00' "$o" || ok=0
    grep -qx 'UPSCALER pass_ms over 1 lines (the first dropped): mean of means 2.600, median 2.600, lowest 2.600, highest 2.600' "$o" || ok=0
    grep -q 'stages \[stage1 0.300, stage2 1.680, stage3 0.420, stage4 0.200\]' "$o" || ok=0
    grep -q 'the log has no date' "$T/run/summary.md" || ok=0
    record u02 "no byte offsets: lines by time of day, said to ignore the date: session 1, model 2 recognised from the stage names (the session has no network model line), selfcheck(unchecked) not taken as passed, first line dropped, 2.600" \
        "$((ok && rc == 0))" "exit $rc; $(grep -E '^UPSCALER (session|model|pass_ms)' "$o" | tr '\n' ' ' | cut -c1-400)"
}
u03() {
    setup u03
    local o ok=1
    mk_log "$T/game.log" >/dev/null
    mk_game_dir "$T/run/game-z" 03:00:00 03:01:00
    kit -- upscaler "$T/run/game-z" --log "$T/game.log"
    local rc=$?
    o=$T/run/game-z/upscaler-read.txt
    grep -q '^UPSCALER NO_LINES no timing lines in this recording; the newest timing line in the log is from 21:01:20 (session 2 of 2), outside the recording, and is not used' "$o" || ok=0
    grep -q 'pass_ms\|UPSCALER_LINE' "$o" && ok=0
    grep -q '^- upscaler game-z: NO_LINES no timing lines in this recording' "$T/run/summary.md" || ok=0
    grep -q 'pass_ms' "$T/run/summary.md" && ok=0
    record u03 "no line inside the recording: 'no timing lines in this recording' in the result and the summary, no numbers of another session, exit 1" \
        "$((ok && rc == 1))" "exit $rc; $(grep -E '^UPSCALER' "$o" | tr '\n' ' ' | cut -c1-300)"
}
u04() {
    setup u04
    local o ok=1
    mk_log "$T/game.log" >/dev/null
    mk_game_dir "$T/run/game-w" 20:00:53 20:00:54
    kit -- upscaler "$T/run/game-w" --log "$T/game.log"
    local rc=$?
    o=$T/run/game-w/upscaler-read.txt
    # Window 20:00:53 to 20:00:59: only the line of 20:00:55, which is dropped.
    grep -q '^UPSCALER NO_RESULT only one timing line fell inside the recording, and it was dropped' "$o" || ok=0
    grep -q 'pass_ms over' "$o" && ok=0
    record u04 "only one timing line inside the recording: it is dropped and no result is given, exit 1" \
        "$((ok && rc == 1))" "exit $rc; $(grep -E '^UPSCALER' "$o" | tr '\n' ' ' | cut -c1-300)"
}

# The reader's literals come from settings (local.env): a made-up project "ProjX" with models A and B, its own stage
# names, settings key and self-check tag, written to a made-up local.env and handed over with HOME_LOCAL_ENV.
mk_local_x() {
    cat >"$1" <<'EOF'
STUDY_UPSCALER_STAGES_1="alpha, bravo, charlie"
STUDY_UPSCALER_STAGES_2="delta, echo, foxtrot, golf"
STUDY_UPSCALER_LABEL_1=A
STUDY_UPSCALER_LABEL_2=B
STUDY_UPSCALER_MODEL_KEY=projx_model
STUDY_UPSCALER_CHECK_TAG=chk=ok
EOF
}
mk_log_x() {
    local f=$1
    {
        printf '[20:00:00.000 100] [amd_fidelityfx_dx12] ==== session start: Game.exe (process 100) ====\r\n'
        printf '[20:00:01.000 100] [amd_fidelityfx_dx12] settings: projx_model=A (ini)\r\n'
        printf '[20:00:01.500 100] [amd_fidelityfx_dx12] #3 the host chose ProjX Model A: settings: projx_model=A (menu)\r\n'
        printf '[20:00:02.000 100] [amd_fidelityfx_dx12]   U1 upscaler info: network kernels: io=direct_textures model=A chk=ok\r\n'
        tline 20:00:55.000 100 1.600 1.700 'alpha 0.310, bravo 0.750, charlie 0.540' 300 1.610 1.605 0.040
        tline 20:01:00.000 100 1.610 1.700 'alpha 0.310, bravo 0.760, charlie 0.540' 300 1.620 1.615 0.040
        tline 20:01:05.000 100 1.650 1.750 'alpha 0.320, bravo 0.780, charlie 0.550' 300 1.660 1.655 0.040
        # No session start line: only the settings key opens the next session, as the reader's setting says.
        printf '[20:01:10.000 100] [amd_fidelityfx_dx12] settings: projx_model=B (menu)\r\n'
        printf '[20:01:10.500 100] [amd_fidelityfx_dx12]   U1 upscaler info: network kernels: io=direct_textures model=B chk=ok(unchecked)\r\n'
        tline 20:01:15.000 100 2.000 2.100 'delta 0.300, echo 1.000, foxtrot 0.500, golf 0.200' 300 2.010 2.005 0.050
        tline 20:01:20.000 100 2.200 2.300 'delta 0.300, echo 1.100, foxtrot 0.600, golf 0.200' 300 2.210 2.205 0.050
        tline 20:01:25.000 100 2.400 2.500 'delta 0.300, echo 1.200, foxtrot 0.700, golf 0.200' 300 2.410 2.405 0.050
    } >"$f"
}
u05() {
    setup u05
    local oa ob ok=1 rca rcb
    mk_local_x "$T/local.env"
    mk_log_x "$T/game.log"
    mk_game_dir "$T/run/game-a" 20:00:58 20:01:02
    mk_game_dir "$T/run/game-b" 20:01:13 20:01:17
    kit HOME_LOCAL_ENV="$T/local.env" -- upscaler "$T/run/game-a" --log "$T/game.log"
    rca=$?
    kit HOME_LOCAL_ENV="$T/local.env" -- upscaler "$T/run/game-b" --log "$T/game.log"
    rcb=$?
    oa=$T/run/game-a/upscaler-read.txt
    ob=$T/run/game-b/upscaler-read.txt
    # Window A: the lines of 20:01:00 (dropped) and 20:01:05, session 1. Window B: 20:01:15 (dropped) and 20:01:20, session 2.
    grep -q '^UPSCALER session 1 of 2; lines written during the recording by time of day' "$oa" || ok=0
    grep -qx 'UPSCALER model A (stage names alpha, bravo, charlie in all 1 lines); chk=ok (self-check passed)' "$oa" || ok=0
    grep -q '^UPSCALER_LINE 20:01:05 model A pass_ms 1.650 ' "$oa" || ok=0
    grep -q '^UPSCALER session 2 of 2; lines written during the recording by time of day' "$ob" || ok=0
    grep -qx 'UPSCALER model B (stage names delta, echo, foxtrot, golf in all 1 lines); chk=ok(unchecked): the self-check did not run' "$ob" || ok=0
    grep -q '^UPSCALER_LINE 20:01:20 model B pass_ms 2.200 ' "$ob" || ok=0
    grep -qx "UPSCALER pass_ms over 1 lines (the first dropped): mean of means 2.200, median 2.200, lowest 2.200, highest 2.200" "$ob" || ok=0
    grep -q "Model A's pass takes" "$oa" || ok=0
    record u05 "reader literals from settings: a made-up project's stage names, model names A and B, settings key and self-check tag, given through a local.env, are recognised (A in session 1, B in session 2 opened by the settings key alone, the unchecked tag not taken as passed)" \
        "$((ok && rca == 0 && rcb == 0))" "exit $rca/$rcb; $(grep -h -E '^UPSCALER (session|model|pass_ms)' "$oa" "$ob" | tr '\n' ' ' | cut -c1-400)"
}
u06() {
    setup u06
    local o ok=1 rc
    mk_log_x "$T/game.log"
    mk_game_dir "$T/run/game-c" 20:00:58 20:01:02
    kit -- upscaler "$T/run/game-c" --log "$T/game.log"
    rc=$?
    o=$T/run/game-c/upscaler-read.txt
    # The same log without the settings: the made-up stage names are not the defaults, so the model is not recognised.
    grep -q '^UPSCALER model MIXED: 1 in 0, 2 in 0, other in 1 of 1 lines' "$o" || ok=0
    grep -q 'unknown (stages alpha, bravo, charlie)' "$o" || ok=0
    grep -q '^UPSCALER_LINE 20:01:05 model unknown (stages alpha, bravo, charlie) ' "$o" || ok=0
    record u06 "the same made-up log read with the default settings: its stage names are not recognised and the line says so (the literals live in the settings, not in the reader)" \
        "$((ok && rc == 0))" "exit $rc; $(grep -E '^UPSCALER (session|model)' "$o" | tr '\n' ' ' | cut -c1-300)"
}

RAN=0
for CUR in r01 r02 r03 r04 r05 r06 r07 r08 r09 r10 k01 k02 k03 k04 k05 k06 k07 k08 k09 k10 k11 k12 k13 k14 k15 k16 k17 \
    k18 k19 k20 k21 k22 k23 k24 k25 k26 k27 k28 k29 k30 k31 k32 u01 u02 u03 u04 u05 u06; do
    want "$@" && { "$CUR"; RAN=$((RAN + 1)); }
done
if ((RAN == 0)); then
    echo "selftest: no test of that name ran" | tee -a "$RES"
    FAILS=$((FAILS + 1))
fi
if ((FAILS)); then
    echo "VERDICT home_selftest: FAIL ($FAILS failed of $RAN)" | tee -a "$RES"
    exit 1
fi
echo "VERDICT home_selftest: PASS ($RAN tests)" | tee -a "$RES"
