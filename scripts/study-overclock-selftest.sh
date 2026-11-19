#!/usr/bin/env bash
# Self-test of scripts/study-overclock.sh: runs its failure paths with a stand-in for `lact` first in PATH, so no
# card setting changes, and stores the evidence (exit code, duration, jobs started, every lact call) per test.
#
# Usage: scripts/study-overclock-selftest.sh                 all tests, about 8 minutes
#        scripts/study-overclock-selftest.sh t01 t08         named tests only
#        BREAK=stockswitch scripts/study-overclock-selftest.sh t02    deliberate break: must FAIL
#        BREAK=nohold scripts/study-overclock-selftest.sh t12         deliberate break: must FAIL
#        BREAK=earlyrelease scripts/study-overclock-selftest.sh t14   deliberate break: must FAIL
# Results: runs/<UTC time>-overclock-selftest/: results.txt (one line per test, then VERDICT), and per test a folder
# with the script's run folder, out.txt, calls.log (the stand-in's record) and the stand-in's state.
#
# The card must be on the "Testing" profile (read with the real `lact cli profile get`, nothing is set). Tests t11
# to t14 start real GPU jobs on that profile: the same short bursts as the stock form of the script (t11 one job,
# t12 one round, t13 two rounds, t14 three rounds). t13 uses the real lact (stock form: it only reads the profile).
# In t14 the stand-in's "switch" changes only a file that stands for the power cap, so the script runs its whole
# comparison form, overclock round included, while the card stays on "Testing".
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${LOCK%/*}"
OUT=$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-overclock-selftest
mkdir -p "$OUT"
RES=$OUT/results.txt
SCRIPT=$REPO/scripts/study-overclock.sh
BREAK=${BREAK:-none}

[[ $(lact cli profile get 2>/dev/null | tr -d '[:space:]') == Testing ]] || {
    echo "selftest: the card is not on the Testing profile; nothing run" | tee -a "$RES"
    exit 3
}

# Deliberate breaks: a changed copy of the script beside the real one (so that its paths resolve), removed at exit.
BROKEN=$REPO/scripts/.study-overclock-broken.sh
cleanup() { rm -f "$BROKEN"; }
trap cleanup EXIT
case $BREAK in
none) ;;
stockswitch)
    # The stock form may switch profiles again.
    # shellcheck disable=SC2016
    sed 's/^        if \[\[ \$MODE == stock \]\]; then$/        if false; then/' "$SCRIPT" >"$BROKEN"
    ;;
nohold)
    # The overclock round no longer keeps the lock.
    sed 's/^        if hold_lock; then$/        if true; then/' "$SCRIPT" >"$BROKEN"
    ;;
earlyrelease)
    # The lock is given up before the switch back.
    sed 's/^    restore$/    release_hold; restore/' "$SCRIPT" >"$BROKEN"
    ;;
*)
    echo "selftest: BREAK is stockswitch, nohold or earlyrelease" >&2
    exit 2
    ;;
esac
if [[ $BREAK != none ]]; then
    cmp -s "$SCRIPT" "$BROKEN" && { echo "selftest: the break did not change the script" | tee -a "$RES"; exit 2; }
    chmod +x "$BROKEN"
    SCRIPT=$BROKEN
    echo "DELIBERATE BREAK: $BREAK" | tee -a "$RES"
fi

FAILS=0
# setup <test>: a fresh folder with the stand-in and its state. Sets T (folder) and RD (the script's run folder).
setup() {
    T=$OUT/$1
    RD=$T/run
    mkdir -p "$T/bin" "$T/hwmon"
    echo Testing >"$T/profile"
    echo 317000000 >"$T/hwmon/power1_cap"
    sed "s|^HWMON=.*|HWMON=$T/hwmon|" "$REPO/runs/latest/card.env" >"$T/card-fake.env"
    cat >"$T/bin/lact" <<'EOF'
#!/usr/bin/env bash
# Stand-in for lact: never talks to the daemon. Records every call; keeps the "profile" in a file.
# SHIM_FLIP_AFTER=n: `profile get` answers Default from call n+1 on. SHIM_SET_FAIL_FIRST=n: the first n
# `profile set` calls fail (n=all: every one). SHIM_CAP_FILE: see below.
echo "$*" >>"$SHIM_DIR/calls.log"
case "$1 $2 $3" in
"cli profile get")
    n=$(($(cat "$SHIM_DIR/gets" 2>/dev/null || echo 0) + 1))
    echo "$n" >"$SHIM_DIR/gets"
    if [[ -n ${SHIM_FLIP_AFTER:-} ]] && ((n > SHIM_FLIP_AFTER)); then echo Default; else cat "$SHIM_DIR/profile"; fi
    ;;
"cli profile set")
    k=$(($(cat "$SHIM_DIR/sets" 2>/dev/null || echo 0) + 1))
    echo "$k" >"$SHIM_DIR/sets"
    f=${SHIM_SET_FAIL_FIRST:-0}
    [[ $f == all ]] && exit 1
    ((k <= f)) && exit 1
    echo "$4" >"$SHIM_DIR/profile"
    # SHIM_CAP_FILE: a file that stands for power1_cap; the "switch" is then seen to take.
    if [[ -n ${SHIM_CAP_FILE:-} ]]; then
        if [[ $4 == Default ]]; then echo 374000000 >"$SHIM_CAP_FILE"; else echo 317000000 >"$SHIM_CAP_FILE"; fi
    fi
    ;;
esac
EOF
    chmod +x "$T/bin/lact"
}
# shim <env assignments...> -- <script arguments>: runs the script with the stand-in; prints nothing; returns its exit.
run_shim() {
    env SHIM_DIR="$T" PATH="$T/bin:$PATH" OVERCLOCK_RUN_DIR="$RD" "$@" "$SCRIPT" >"$T/out.txt" 2>&1
}
# start_shim <env assignments...>: the comparison form in the background. The subshell execs, so SP is the script's own PID and a
# signal sent to it reaches the script.
start_shim() {
    (exec env SHIM_DIR="$T" PATH="$T/bin:$PATH" OVERCLOCK_RUN_DIR="$RD" "$@" "$SCRIPT" >"$T/out.txt" 2>&1) &
    SP=$!
}
# count <pattern> <file...>: the number of matching lines, 0 when a file is missing.
count() {
    local pat=$1 n
    shift
    n=$(cat "$@" 2>/dev/null | grep -c -- "$pat")
    echo "${n:-0}"
}
calls() { count "$1" "$T/calls.log"; }
jobs_started() { count '^## 20' "$RD"/*.log; }
# check <test> <what was expected> <condition result 0/1> <what was seen>
record() {
    local name=$1 want=$2 ok=$3 seen=$4 st=PASS
    ((ok)) || { st=FAIL; FAILS=$((FAILS + 1)); }
    echo "$st $name | expected: $want | seen: $seen" | tee -a "$RES"
}
want() { [[ $# -eq 0 ]] && return 0; local t; for t in "$@"; do [[ $t == "$CUR" ]] && return 0; done; return 1; }

t01() {
    setup t01
    run_shim CARD_ENV="$T/card-fake.env" ROUNDS=stock
    local rc=$? j s
    j=$(jobs_started); s=$(calls 'profile set')
    record t01 "stock form, hotspot unreadable: exit 1, no job, no profile set" \
        $((rc == 1 && j == 0 && s == 0 && $(count 'STOPPED: hotspot unreadable' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, profile set calls $s, $(grep -m1 -o 'STOPPED: hotspot unreadable' "$T/out.txt")"
}
t02() {
    setup t02
    run_shim SHIM_FLIP_AFTER=1 ROUNDS=stock
    local rc=$? j s
    j=$(jobs_started); s=$(calls 'profile set')
    record t02 "stock form, profile reads Default after the start check: exit 4, no job, no profile set" \
        $((rc == 4 && j == 0 && s == 0)) "exit $rc, jobs $j, profile set calls $s"
}
t03() {
    setup t03
    echo Default >"$T/profile"
    run_shim
    local rc=$? j s
    j=$(jobs_started); s=$(calls 'profile set')
    record t03 "comparison form, profile is Default at the start: exit 3, no job, no profile set" \
        $((rc == 3 && j == 0 && s == 0)) "exit $rc, jobs $j, profile set calls $s"
}
t04() {
    setup t04
    run_shim CARD_ENV="$T/card-fake.env" SHIM_SET_FAIL_FIRST=all
    local rc=$? j st sd
    j=$(jobs_started); st=$(calls 'profile set Testing'); sd=$(calls 'profile set Default')
    record t04 "comparison form, sensors unreadable, switch back always fails: exit 5, RESTORE FAILED, 2 tries, no job" \
        $((rc == 5 && j == 0 && st == 2 && sd == 0 && $(count 'RESTORE FAILED' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, set Testing calls $st, set Default calls $sd, RESTORE FAILED lines $(count 'RESTORE FAILED' "$T/out.txt")"
}
t05() {
    setup t05
    run_shim CARD_ENV="$T/card-fake.env"
    local rc=$? j st sd
    j=$(jobs_started); st=$(calls 'profile set Testing'); sd=$(calls 'profile set Default')
    record t05 "comparison form, sensors unreadable, switch back works: exit 1, restored, 1 try, no job" \
        $((rc == 1 && j == 0 && st == 1 && sd == 0 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, set Testing calls $st, set Default calls $sd, restored lines $(count '^- restored' "$T/out.txt")"
}
t06() {
    setup t06
    run_shim CARD_ENV="$T/card-fake.env" SHIM_SET_FAIL_FIRST=1
    local rc=$? j st
    j=$(jobs_started); st=$(calls 'profile set Testing')
    record t06 "comparison form, first switch back fails, second works: exit 1, restored, 2 tries, no job" \
        $((rc == 1 && j == 0 && st == 2 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, set Testing calls $st, restored lines $(count '^- restored' "$T/out.txt")"
}
t07() {
    setup t07
    echo 70000 >"$T/hwmon/temp2_input"
    local t0=$SECONDS rc j
    run_shim CARD_ENV="$T/card-fake.env" ROUNDS=stock
    rc=$?
    local dt=$((SECONDS - t0))
    j=$(jobs_started)
    record t07 "stock form, hotspot stays at 70 C: gives up after 3 minutes, exit 1, no job" \
        $((rc == 1 && j == 0 && dt >= 178 && dt <= 200 && $(count 'hotspot still 70 C after 3 minutes' "$T/out.txt") >= 1)) \
        "exit $rc after $dt s, jobs $j, $(grep -m1 -o 'STOPPED: hotspot still 70 C after 3 minutes' "$T/out.txt")"
}
t08() {
    setup t08
    flock "$LOCK" sleep 75 &
    local hp=$! t0 rc j dt
    sleep 1
    t0=$SECONDS
    run_shim ROUNDS=stock
    rc=$?
    dt=$((SECONDS - t0))
    j=$(jobs_started)
    wait "$hp"
    record t08 "stock form, lock held by another process for 75 s: gives up after 60 s, exit 1, no job" \
        $((rc == 1 && j == 0 && dt >= 59 && dt <= 66)) "exit $rc after $dt s, jobs $j"
}
# signal_in_lock_wait <test> <signal> <expected exit>
signal_in_lock_wait() {
    setup "$1"
    flock "$LOCK" sleep 40 &
    local hp=$! t0 rc j dt st
    sleep 1
    t0=$SECONDS
    start_shim
    sleep 5
    kill "-$2" "$SP"
    wait "$SP"
    rc=$?
    dt=$((SECONDS - t0))
    j=$(jobs_started); st=$(calls 'profile set Testing')
    wait "$hp"
    record "$1" "comparison form, $2 while it waits for the lock (held 40 s): exit $3 once the wait ends, restored, no job" \
        $((rc == $3 && j == 0 && st == 1 && dt >= 38 && dt <= 60 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc after $dt s, jobs $j, set Testing calls $st, restored lines $(count '^- restored' "$T/out.txt")"
}
t09() { signal_in_lock_wait t09 TERM 130; }
t10() { signal_in_lock_wait t10 HUP 129; }
t11() {
    setup t11
    start_shim
    local rc j n=0 log=$RD/A-testing-mat_card-reg_f16.log v
    until grep -q '^## 20' "$log" 2>/dev/null; do
        sleep 0.2
        n=$((n + 1))
        ((n > 300)) && break
    done
    sleep 0.3
    kill -TERM "$SP"
    wait "$SP"
    rc=$?
    j=$(jobs_started); v=$(count '^VERDICT' "$log")
    record t11 "comparison form, TERM while the first GPU job runs: the job finishes, then exit 130, restored, no further job" \
        $((rc == 130 && j == 1 && v == 1 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, VERDICT lines in the job's log $v, restored lines $(count '^- restored' "$T/out.txt")"
}
t12() {
    setup t12
    start_shim
    local rc n=0 held=unknown lk=0 sd st bc
    # Round A is done when its streaming log has three "after:" lines; the lock probe comes 8 s later, inside the
    # 30 s the script then waits for a power cap that the stand-in never changes.
    until (($(count '^after:' "$RD/A-testing-bandwidth.log") == 3)); do
        sleep 0.5
        n=$((n + 1))
        ((n > 600)) && break
    done
    sleep 8
    if flock -n "$LOCK" true; then held=free; else held=held lk=1; fi
    wait "$SP"
    rc=$?
    sd=$(calls 'profile set Default'); st=$(calls 'profile set Testing')
    bc=$(find "$RD" -name 'B-*' -o -name 'C-*' | wc -l)
    record t12 "comparison form, switch to Default does not take: round A runs, lock held during the switch, stop, restored, exit 1, no round B or C" \
        $((lk == 1 && rc == 1 && sd == 1 && st == 1 && bc == 0 && $(jobs_started) == 7 && $(count 'could not switch to Default' "$T/out.txt") == 1 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc, jobs $(jobs_started), lock during the switch: $held, set Default calls $sd, set Testing calls $st, B or C files $bc"
}
t13() {
    T=$OUT/t13
    RD=$T/run
    mkdir -p "$T"
    env OVERCLOCK_RUN_DIR="$RD" ROUNDS=stock "$SCRIPT" >"$T/out.txt" 2>&1
    local rc=$? j ok
    j=$(jobs_started); ok=$(count '^after: exit 0;' "$RD"/*.log)
    record t13 "stock form with the real lact, normal: exit 0, 14 jobs, each exit 0" \
        $((rc == 0 && j == 14 && ok == 14)) "exit $rc, jobs $j, jobs with exit 0: $ok"
}

# wait_lines <n> <pattern> <file>: waits (at most 5 minutes) until the file has n matching lines.
wait_lines() {
    local n=0
    until (($(count "$2" "$3") >= $1)); do
        sleep 0.5
        n=$((n + 1))
        ((n > 600)) && return 1
    done
}
lock_state() { if flock -n "$LOCK" true; then echo free; else echo held; fi; }
t14() {
    setup t14
    # A sensor folder that is the real one except for the power cap, which the stand-in's "switch" rewrites.
    local real f rc sw rs ed j ok sd st lk=0
    real=$(sed -n 's/^HWMON=//p' "$REPO/runs/latest/card.env")
    for f in "$real"/*; do
        [[ $(basename "$f") == power1_cap ]] || ln -s "$f" "$T/hwmon/$(basename "$f")"
    done
    start_shim CARD_ENV="$T/card-fake.env" SHIM_CAP_FILE="$T/hwmon/power1_cap"
    # 2 s after round A's last job the script is inside the switch to "Default" (it waits 6 s after the cap changes);
    # 2 s after round B's last job it is inside the switch back. Both must find the lock held.
    wait_lines 3 '^after:' "$RD/A-testing-bandwidth.log"
    sleep 2
    sw=$(lock_state)
    wait_lines 3 '^after:' "$RD/B-overclock-bandwidth.log"
    sleep 2
    rs=$(lock_state)
    wait "$SP"
    rc=$?
    ed=$(lock_state)
    j=$(jobs_started); ok=$(count '^after: exit 0;' "$RD"/*.log)
    sd=$(calls 'profile set Default'); st=$(calls 'profile set Testing')
    [[ $sw == held && $rs == held && $ed == free ]] && lk=1
    record t14 "comparison form, whole flow with the stand-in (the card stays on Testing): exit 0, 21 jobs each exit 0, lock held during the switch and during the switch back, free at the end, one set Default and one set Testing" \
        $((lk == 1 && rc == 0 && j == 21 && ok == 21 && sd == 1 && st == 1 && $(count '^- restored' "$T/out.txt") == 1)) \
        "exit $rc, jobs $j, jobs with exit 0: $ok, lock during the switch: $sw, during the switch back: $rs, at the end: $ed, set Default calls $sd, set Testing calls $st"
}

RAN=0
for CUR in t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 t11 t12 t13 t14; do
    want "$@" && { "$CUR"; RAN=$((RAN + 1)); }
done
if ((RAN == 0)); then
    echo "selftest: no test of that name ran" | tee -a "$RES"
    FAILS=$((FAILS + 1))
fi
p=$(lact cli profile get 2>/dev/null | tr -d '[:space:]')
real=$(sed -n 's/^HWMON=//p' "$REPO/runs/latest/card.env")
dev=$(sed -n 's/^DRM_DEVICE=//p' "$REPO/runs/latest/card.env")
echo "card after the tests (real lact and sysfs): profile $p; power1_cap $(cat "$real/power1_cap"); $(grep -A1 -E 'OD_SCLK_OFFSET|OD_VDDGFX_OFFSET' "$dev/pp_od_clk_voltage" | tr '\n' ' ')" | tee -a "$RES"
[[ $p == Testing && $(cat "$real/power1_cap") == 317000000 ]] || FAILS=$((FAILS + 1))
if ((FAILS)); then
    echo "VERDICT overclock_selftest: FAIL ($FAILS failed)" | tee -a "$RES"
    exit 1
fi
echo "VERDICT overclock_selftest: PASS" | tee -a "$RES"
