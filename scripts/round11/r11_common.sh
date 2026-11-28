# shellcheck shell=bash
# Sourced by the round 11 runners (run_r11_pcie.sh, run_r11_video.sh, run_r11_rt.sh). The gate every GPU job goes through, the card lookup by
# PCI device id, the locked job runner (flock, nice -n 10, timeout, hotspot guard, sampler, kernel-log check), and the gate's own CPU
# self-test. Nothing here writes sysfs, uses sudo, or touches git.
#
# Exit codes (the same table as round 10): 0 ok; 2 usage or refusal before any job; 3 wrong card; 4 too hot to start; 5 memory below 8 GiB;
# 6 lock wait expired; 8 a job hit the 95 C guard; 10 a job failed or timed out; 12 the kernel log cannot be read; 13 a GPU fault or reset
# in the kernel log; 14 a probe's own control failed.
# Test hooks (read only when R11_SELFTEST=1): R11_MEMINFO R11_HOT_FILE R11_DEVID_FILE R11_JOURNAL_FILE R11_HOLD R11_RESTORE_FAILED.
# Environment: STUDY_GPU_LOCK (default <repo>/runs/gpu.lock), R11_LOCK_WAIT (s, default 900),
#   R11_REPO (this checkout by default), R11_BIN (binaries, default <repo>/build/round11/bin), R11_SPV (default <repo>/build/round11/spv),
#   R11_RUN (run folder, default <repo>/runs/<UTC>-r11<item>), CARD_ENV (not needed: the card is found by device id).

R11_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R11_REPO=${R11_REPO:-$(cd "$R11_COMMON_DIR/../.." && pwd)}
R11_BIN=${R11_BIN:-$R11_REPO/build/round11/bin}
R11_SPV=${R11_SPV:-$R11_REPO/build/round11/spv}
STUDY_GPU_LOCK=${STUDY_GPU_LOCK:-$R11_REPO/runs/gpu.lock}
R11_LOCK_WAIT=${R11_LOCK_WAIT:-900}
R11_DEVID=0x7550
R11_MIN_KB=$((8 * 1024 * 1024))
R11_START_C=58
R11_GUARD_C=95
R11_FAULT_RE='(amdgpu|\[drm|gfxhub|mmhub).*(page fault|GPU reset|ring [A-Za-z0-9_.]+ timeout|VM_L2_PROTECTION|hang|\*ERROR\*)|amdgpu.*(timeout|fault)'
R11_PAUSE=${R11_PAUSE:-10}

if [[ ${R11_SELFTEST:-0} == 1 ]]; then
    R11_MEMINFO=${R11_MEMINFO:-/proc/meminfo}
    R11_HOLD=${R11_HOLD:-$R11_REPO/runs/HOLD-FSR}
    R11_RESTORE_FAILED=${R11_RESTORE_FAILED:-$R11_REPO/runs/RESTORE-FAILED}
    R11_JOURNAL_FILE=${R11_JOURNAL_FILE:-}
else
    for v in R11_MEMINFO R11_HOT_FILE R11_DEVID_FILE R11_JOURNAL_FILE R11_HOLD R11_RESTORE_FAILED; do
        [[ -n ${!v:-} ]] && echo "r11_common: ignored $v (test hook, read only with R11_SELFTEST=1)" >&2
    done
    R11_MEMINFO=/proc/meminfo
    R11_HOT_FILE=""
    R11_DEVID_FILE=""
    R11_JOURNAL_FILE=""
    R11_HOLD=$R11_REPO/runs/HOLD-FSR
    R11_RESTORE_FAILED=$R11_REPO/runs/RESTORE-FAILED
fi

r11_log() { echo "$(date -u +%T) $*"; }
r11_die() { local c=$1; shift; echo "$(date -u +%T) REFUSED/FAILED (exit $c): $*" >&2; exit "$c"; }

# The card: the amdgpu device with vendor 0x1002 and device id 0x7550, found in /sys/class/drm (the machine has an iGPU too, so no index is trusted).
# Sets R11_PCI R11_DRM (sysfs device folder) R11_HWMON R11_RENDER (/dev/dri/renderD*). Returns 3 when absent or not unique.
r11_find_card() {
    R11_PCI="" R11_DRM="" R11_HWMON="" R11_RENDER=""
    local c d n=0 id
    for c in /sys/class/drm/card[0-9]*; do
        [[ -e $c/device/device ]] || continue
        [[ $c == *-* ]] && continue
        id=$(cat "${R11_DEVID_FILE:-$c/device/device}" 2>/dev/null)
        [[ $id == "$R11_DEVID" && $(cat "$c/device/vendor" 2>/dev/null) == 0x1002 ]] || continue
        d=$(readlink -f "$c/device")
        R11_DRM=$d
        R11_PCI=$(basename "$d")
        n=$((n + 1))
    done
    ((n == 1)) || { echo "r11: found $n card(s) with device id $R11_DEVID (need exactly 1)" >&2; return 3; }
    R11_HWMON=$(ls -d "$R11_DRM"/hwmon/hwmon* 2>/dev/null | head -1)
    local r
    for r in "$R11_DRM"/drm/renderD*; do [[ -e $r ]] && R11_RENDER=/dev/dri/$(basename "$r"); done
    [[ -n $R11_HWMON && -n $R11_RENDER ]] || { echo "r11: card $R11_PCI has no hwmon or render node" >&2; return 3; }
    return 0
}

r11_hot_c() {  # whole degrees C of the hotspot, empty when unreadable
    local f=${R11_HOT_FILE:-$R11_HWMON/temp2_input} v
    v=$(cat "$f" 2>/dev/null)
    [[ $v =~ ^[0-9]+$ ]] || return 1
    echo $((v / 1000))
}
r11_avail_kb() { awk '/^MemAvailable:/ {print $2}' "$R11_MEMINFO"; }

# r11_gate_static: the checks that never wait. Returns the exit code to refuse with (0 = pass).
r11_gate_static() {
    r11_find_card || return 3
    local kb
    kb=$(r11_avail_kb)
    [[ $kb =~ ^[0-9]+$ ]] || { echo "r11: cannot read MemAvailable from $R11_MEMINFO" >&2; return 5; }
    if ((kb < R11_MIN_KB)); then echo "r11: only $((kb / 1024)) MiB available, the floor is $((R11_MIN_KB / 1024)) MiB" >&2; return 5; fi
    if [[ -e $R11_HOLD ]]; then echo "r11: hold file $R11_HOLD exists (the owner's stop sign)" >&2; return 2; fi
    if [[ -e $R11_RESTORE_FAILED ]]; then echo "r11: $R11_RESTORE_FAILED exists (a memory-clock restore failed; check by hand)" >&2; return 2; fi
    return 0
}

# r11_wait_cool: waits up to 300 s for the hotspot to be at or below R11_START_C; returns 4 otherwise (also when the sensor cannot be read).
r11_wait_cool() {
    local w=0 t
    while :; do
        t=$(r11_hot_c) || { echo "r11: hotspot unreadable" >&2; return 4; }
        ((t <= R11_START_C)) && return 0
        ((w >= 300)) && { echo "r11: hotspot still ${t} C after 300 s (limit $R11_START_C C to start)" >&2; return 4; }
        sleep 5
        w=$((w + 5))
    done
}

# r11_journal_check <epoch seconds>: 0 clean, 12 unreadable, 13 a fault line (printed).
r11_journal_check() {
    local since=$1 out
    if [[ -n ${R11_JOURNAL_FILE:-} ]]; then
        out=$(cat "$R11_JOURNAL_FILE" 2>/dev/null) || return 12
    else
        out=$(journalctl -k --since "@$since" --no-pager -q 2>&1) || { echo "r11: journalctl -k not readable: $(echo "$out" | head -1)" >&2; return 12; }
    fi
    local hit
    hit=$(echo "$out" | grep -Ei "$R11_FAULT_RE" | head -5)
    if [[ -n $hit ]]; then echo "r11: kernel log shows a GPU fault or reset:"; echo "$hit"; return 13; fi
    return 0
}

# Link and clock record of the card (sysfs reads only), one line.
r11_card_state() {
    local d=$R11_DRM
    echo "link $(cat "$d/current_link_speed" 2>/dev/null) x$(cat "$d/current_link_width" 2>/dev/null) (max $(cat "$d/max_link_speed" 2>/dev/null) x$(cat "$d/max_link_width" 2>/dev/null)); hotspot $(r11_hot_c) C; power $(awk '{printf "%.0f", $1/1e6}' "$R11_HWMON/power1_average" 2>/dev/null) W; sclk $(grep '\*' "$d/pp_dpm_sclk" 2>/dev/null | tr -s ' ' | tr '\n' ' '); mclk $(grep '\*' "$d/pp_dpm_mclk" 2>/dev/null | tr -s ' ' | tr '\n' ' ')"
}

# r11_pg_alive <pgid>: 0 while any process of that process group is not a zombie.
r11_pg_alive() {
    ps -eo pgid=,stat= 2>/dev/null | awk -v g="$1" '$1 == g && $2 !~ /^Z/ { f = 1 } END { exit !f }'
}

# r11_job_cleanup: ends everything r11_locked_job started (payload process group, guard, sampler) and waits for all of it, so no GPU
# payload outlives the lock. Idempotent; the lock descriptor is closed by the caller only after this returns.
R11_JOB_PGID="" R11_JOB_JPID="" R11_JOB_GPID="" R11_JOB_SPID="" R11_JOB_ACTIVE=0
r11_job_cleanup() {
    local i
    if [[ -n $R11_JOB_GPID ]]; then kill "$R11_JOB_GPID" 2>/dev/null; wait "$R11_JOB_GPID" 2>/dev/null; fi
    if [[ -n $R11_JOB_PGID ]]; then
        kill -TERM -- "-$R11_JOB_PGID" 2>/dev/null
        for ((i = 0; i < 80; i++)); do r11_pg_alive "$R11_JOB_PGID" || break; sleep 0.1; done
        if r11_pg_alive "$R11_JOB_PGID"; then
            r11_log "payload group $R11_JOB_PGID ignored TERM for 8 s: KILL"
            kill -KILL -- "-$R11_JOB_PGID" 2>/dev/null
            for ((i = 0; i < 50; i++)); do r11_pg_alive "$R11_JOB_PGID" || break; sleep 0.1; done
        fi
    fi
    [[ -n $R11_JOB_JPID && -z $R11_JOB_PGID ]] && kill -TERM "$R11_JOB_JPID" 2>/dev/null
    [[ -n $R11_JOB_JPID ]] && wait "$R11_JOB_JPID" 2>/dev/null
    if [[ -n $R11_JOB_SPID ]]; then kill -TERM "$R11_JOB_SPID" 2>/dev/null; wait "$R11_JOB_SPID" 2>/dev/null; fi
    R11_JOB_PGID="" R11_JOB_JPID="" R11_JOB_GPID="" R11_JOB_SPID=""
}
# r11_job_abort <signal number>: the trap body; cleanup first, then the lock descriptor, then leave the runner.
r11_job_abort() {
    local sig=$1
    trap '' INT TERM HUP
    r11_log "signal $sig: stopping the payload group, guard and sampler before releasing the lock"
    r11_job_cleanup
    R11_JOB_ACTIVE=0
    flock -u 9 2>/dev/null
    exec 9>&-
    trap - EXIT INT TERM HUP
    if [[ $sig == EXIT ]]; then return 0; fi
    case $sig in INT) exit 130 ;; HUP) exit 129 ;; *) exit 143 ;; esac
}

# r11_locked_job <name> <timeout s> <command...>  (run it from a runner that set R11_RUN)
# One gated GPU job: the static gate, the lock (waits R11_LOCK_WAIT s, exit 6), the gate again under the lock, a cool card, the sampler beside the job,
# nice -n 10 timeout, the 95 C guard, clocks before and after, the kernel-log check. stdout and stderr go to $R11_RUN/<name>.log.
# Returns the exit code table above. R11_NO_LOCK=1 (selftest only) uses STUDY_GPU_LOCK as given and does not refuse the real card's absence.
r11_locked_job() {
    local name=$1 tmo=$2
    shift 2
    local log=$R11_RUN/$name.log rc t0 jpid spid gpid pgid
    mkdir -p "$R11_RUN"
    r11_gate_static || { rc=$?; r11_log "gate refused $name (exit $rc)"; return "$rc"; }
    mkdir -p "$(dirname "$STUDY_GPU_LOCK")"
    exec 9>>"$STUDY_GPU_LOCK" || return 2
    r11_log "waiting for the lock $name"
    if ! flock -w "$R11_LOCK_WAIT" 9; then r11_log "lock wait of $R11_LOCK_WAIT s expired for $name"; exec 9>&-; return 6; fi
    # Everything below holds the lock. The traps go in before any child starts: whatever ends the runner, the payload group, guard and
    # sampler are stopped and waited for first, and only then is descriptor 9 (the lock) closed.
    R11_JOB_ACTIVE=1
    trap 'r11_job_abort INT' INT
    trap 'r11_job_abort TERM' TERM
    trap 'r11_job_abort HUP' HUP
    trap '((R11_JOB_ACTIVE)) && r11_job_abort EXIT' EXIT
    r11_gate_static || { rc=$?; r11_log "gate refused $name after the lock (exit $rc)"; r11_job_abort EXIT; return "$rc"; }
    r11_wait_cool || { rc=$?; r11_job_abort EXIT; return "$rc"; }
    r11_log "start $name: $(r11_card_state)" | tee -a "$R11_RUN/card-state.log"
    t0=$(date +%s)
    local sdir=$R11_RUN/$name.sampler
    mkdir -p "$sdir"
    spid=""
    if [[ -x $R11_BIN/r11_sampler ]]; then
        DRM_DEVICE=$R11_DRM HWMON=$R11_HWMON "$R11_BIN/r11_sampler" "$sdir" 100 >/dev/null 2>"$sdir/stderr" &
        spid=$!
        R11_JOB_SPID=$spid
        local w=0
        while [[ ! -e $sdir/ready && $w -lt 50 ]]; do sleep 0.1; w=$((w + 1)); done
    else
        r11_log "WARNING no r11_sampler in $R11_BIN: no clock record for $name"
    fi
    # setsid: the payload (and timeout and whatever it starts) is its own session and process group, so one kill reaches all of it.
    setsid nice -n 10 timeout -k 5 "$tmo" "$@" >"$log" 2>&1 &
    jpid=$!
    R11_JOB_JPID=$jpid
    # The group is trusted only once setsid has really made the payload its own group leader (pgid == pid): before that the child still
    # shares the runner's group, and a group kill would hit the runner itself.
    local k
    for ((k = 0; k < 100; k++)); do
        pgid=$(ps -o pgid= -p "$jpid" 2>/dev/null | tr -d ' ')
        [[ $pgid == "$jpid" ]] && { R11_JOB_PGID=$jpid; break; }
        [[ -z $pgid ]] && break
        sleep 0.05
    done
    if [[ -z $R11_JOB_PGID ]] && kill -0 "$jpid" 2>/dev/null; then r11_log "payload $jpid did not become its own process group: stopping it"; r11_job_abort EXIT; return 10; fi
    # Guard: hotspot read every 200 ms; at the limit (or an unreadable sensor) TERM the job's process by PID, KILL 2 s later.
    (
        while kill -0 "$jpid" 2>/dev/null; do
            t=$(r11_hot_c) || { echo "GUARD: hotspot unreadable, stopping job $jpid" >>"$R11_RUN/guard.log"; kill -TERM "$jpid" 2>/dev/null; sleep 2; kill -KILL "$jpid" 2>/dev/null; exit 0; }
            if ((t >= R11_GUARD_C)); then
                echo "GUARD: hotspot ${t} C at the ${R11_GUARD_C} C limit, stopping job $jpid" >>"$R11_RUN/guard.log"
                kill -TERM "$jpid" 2>/dev/null; sleep 2; kill -KILL "$jpid" 2>/dev/null
                exit 0
            fi
            sleep 0.2
        done
    ) &
    gpid=$!
    R11_JOB_GPID=$gpid
    wait "$jpid"
    rc=$?
    R11_JOB_JPID=""
    r11_job_cleanup
    r11_log "end $name exit=$rc: $(r11_card_state)" | tee -a "$R11_RUN/card-state.log"
    if grep -q '^GUARD:' "$R11_RUN/guard.log" 2>/dev/null && [[ $rc -ne 0 ]]; then rc=8; fi
    local jr=0
    r11_journal_check "$t0" >"$R11_RUN/$name.journal" 2>&1 || jr=$?
    cat "$R11_RUN/$name.journal"
    R11_JOB_ACTIVE=0
    trap - EXIT INT TERM HUP
    flock -u 9
    exec 9>&-
    sleep "$R11_PAUSE"
    ((jr != 0)) && return "$jr"
    if ((rc == 1)); then return 14; fi
    ((rc != 0 && rc != 8)) && rc=10
    return "$rc"
}

# ---------------------------------------------------------------------------------------------------------------------------------
# CPU self-test of the gate (stand-in data only; the real lock file is never opened). Prints PASS/FAIL per case; returns the number of failures.
# Every refusal is a deliberate break of the gate: the gate must refuse (exit code as listed).
r11_gate_selftest() {
    local d=$1 bad=0 rc
    mkdir -p "$d"
    expect() { # name expected got
        if [[ $2 == "$3" ]]; then echo "gate-selftest PASS $1 (exit $3)"; else echo "gate-selftest FAIL $1: expected exit $2, got $3"; bad=$((bad + 1)); fi
    }
    export R11_SELFTEST=1
    # stand-in meminfo and hotspot
    printf 'MemTotal: 32000000 kB\nMemAvailable: 16000000 kB\n' >"$d/meminfo.ok"
    printf 'MemTotal: 32000000 kB\nMemAvailable: 6000000 kB\n' >"$d/meminfo.low"
    echo 40000 >"$d/hot.ok"
    echo 80000 >"$d/hot.hot"
    echo garbage >"$d/hot.bad"
    echo 0x7550 >"$d/dev.ok"
    echo 0x13c0 >"$d/dev.wrong"
    : >"$d/journal.clean"
    echo '[  12.3] amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1' >"$d/journal.fault"
    if [[ -e /sys/class/drm/card1/device/device ]] || ls /sys/class/drm/card[0-9]*/device/device >/dev/null 2>&1; then
        (R11_MEMINFO=$d/meminfo.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none; r11_gate_static >/dev/null 2>&1); rc=$?
        # With a card present and everything fine the gate passes; with no 0x7550 on this machine it must say exit 3.
        if ls /sys/class/drm/card[0-9]*/device/device >/dev/null 2>&1 && grep -qx 0x7550 /sys/class/drm/card[0-9]*/device/device 2>/dev/null; then expect "static gate, right card, enough memory" 0 "$rc"; fi
    fi
    (R11_DEVID_FILE=$d/dev.wrong; r11_find_card >/dev/null 2>&1); rc=$?
    expect "break: card with another device id" 3 "$rc"
    (R11_MEMINFO=$d/meminfo.low R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none; r11_gate_static >/dev/null 2>&1); rc=$?
    expect "break: 6 GB available (floor 8 GiB)" 5 "$rc"
    : >"$d/hold"
    (R11_MEMINFO=$d/meminfo.ok R11_HOLD=$d/hold R11_RESTORE_FAILED=$d/none; r11_gate_static >/dev/null 2>&1); rc=$?
    expect "break: hold file present" 2 "$rc"
    (R11_MEMINFO=$d/meminfo.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/hold; r11_gate_static >/dev/null 2>&1); rc=$?
    expect "break: RESTORE-FAILED file present" 2 "$rc"
    (R11_HOT_FILE=$d/hot.hot; R11_START_C=58; r11_hot_c >/dev/null; t=$(r11_hot_c); [[ $t -gt 58 ]]); rc=$?
    expect "hotspot 80 C reads above the 58 C start limit" 0 "$rc"
    (R11_HOT_FILE=$d/hot.bad; r11_hot_c >/dev/null 2>&1); rc=$?
    expect "break: unreadable hotspot is an error" 1 "$rc"
    (R11_JOURNAL_FILE=$d/journal.clean r11_journal_check 0 >/dev/null 2>&1); rc=$?
    expect "clean kernel log" 0 "$rc"
    (R11_JOURNAL_FILE=$d/journal.fault r11_journal_check 0 >/dev/null 2>&1); rc=$?
    expect "break: ring timeout in the kernel log" 13 "$rc"
    (R11_JOURNAL_FILE=$d/journal.missing r11_journal_check 0 >/dev/null 2>&1); rc=$?
    expect "break: kernel log unreadable" 12 "$rc"
    # lock wait: a holder keeps a stand-in lock; the job must give up with exit 6 and never run its command
    (
        export STUDY_GPU_LOCK=$d/lock R11_LOCK_WAIT=1 R11_RUN=$d/run R11_MEMINFO=$d/meminfo.ok R11_HOT_FILE=$d/hot.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none R11_PAUSE=0
        r11_gate_static >/dev/null 2>&1 || exit 77  # no card on this machine: the case cannot run, reported below
        exec 8>>"$d/lock"
        flock 8
        r11_locked_job selftest-held 5 touch "$d/should-not-exist" >/dev/null 2>&1
    ); rc=$?
    if ((rc == 77)); then echo "gate-selftest SKIP lock-wait case (no 0x7550 card on this machine)"; else expect "break: lock held by someone else" 6 "$rc"; [[ ! -e $d/should-not-exist ]] || { echo "gate-selftest FAIL held lock still ran the job"; bad=$((bad + 1)); }; fi
    # a clean locked job over a stand-in lock: the command runs under flock, with nice 10
    (
        export STUDY_GPU_LOCK=$d/lock2 R11_LOCK_WAIT=5 R11_RUN=$d/run2 R11_MEMINFO=$d/meminfo.ok R11_HOT_FILE=$d/hot.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none R11_PAUSE=0 R11_JOURNAL_FILE=$d/journal.clean
        r11_gate_static >/dev/null 2>&1 || exit 77
        r11_locked_job selftest-ok 20 bash -c 'echo "nice=$(nice) parent_nice='"$(nice)"'"; sleep 0.3' >/dev/null 2>&1
    ); rc=$?
    if ((rc == 77)); then echo "gate-selftest SKIP locked-job case (no 0x7550 card on this machine)"; else
        expect "clean locked job" 0 "$rc"
        local jn pn
        jn=$(sed -n 's/^nice=\(-\?[0-9]\+\) .*/\1/p' "$d/run2/selftest-ok.log" 2>/dev/null)
        pn=$(sed -n 's/.*parent_nice=\(-\?[0-9]\+\).*/\1/p' "$d/run2/selftest-ok.log" 2>/dev/null)
        if [[ $jn =~ ^-?[0-9]+$ && $pn =~ ^-?[0-9]+$ ]] && { ((jn >= pn + 10)) || ((jn == 19)); }; then echo "gate-selftest PASS the job ran 10 nice levels below its parent (parent $pn, job $jn)"; else echo "gate-selftest FAIL job nice $jn is not parent $pn + 10: $(cat "$d/run2/selftest-ok.log" 2>/dev/null)"; bad=$((bad + 1)); fi
    fi
    # a fault line in the kernel log after a job: exit 13
    (
        export STUDY_GPU_LOCK=$d/lock3 R11_LOCK_WAIT=5 R11_RUN=$d/run3 R11_MEMINFO=$d/meminfo.ok R11_HOT_FILE=$d/hot.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none R11_PAUSE=0 R11_JOURNAL_FILE=$d/journal.fault
        r11_gate_static >/dev/null 2>&1 || exit 77
        r11_locked_job selftest-fault 20 true >/dev/null 2>&1
    ); rc=$?
    if ((rc == 77)); then echo "gate-selftest SKIP fault case"; else expect "break: GPU fault in the kernel log after the job" 13 "$rc"; fi
    # a job that times out: timeout exits 124 -> 10
    (
        export STUDY_GPU_LOCK=$d/lock4 R11_LOCK_WAIT=5 R11_RUN=$d/run4 R11_MEMINFO=$d/meminfo.ok R11_HOT_FILE=$d/hot.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none R11_PAUSE=0 R11_JOURNAL_FILE=$d/journal.clean
        r11_gate_static >/dev/null 2>&1 || exit 77
        r11_locked_job selftest-timeout 1 sleep 30 >/dev/null 2>&1
    ); rc=$?
    if ((rc == 77)); then echo "gate-selftest SKIP timeout case"; else expect "break: job exceeds its time limit" 10 "$rc"; fi
    # the runner is killed (SIGTERM) while its payload runs: the payload must be dead before the lock can be taken, and the lock free after.
    # A private lock and a harmless sleep; the real lock and the GPU are never touched.
    (
        export STUDY_GPU_LOCK=$d/lock5 R11_LOCK_WAIT=5 R11_RUN=$d/run5 R11_MEMINFO=$d/meminfo.ok R11_HOT_FILE=$d/hot.ok R11_HOLD=$d/none R11_RESTORE_FAILED=$d/none R11_PAUSE=0 R11_JOURNAL_FILE=$d/journal.clean
        r11_gate_static >/dev/null 2>&1 || exit 77
        rm -f "$d/payload.pid" "$d/payload.child.pid"
        r11_locked_job selftest-kill 120 bash -c 'echo $$ >"'"$d"'/payload.pid"; sleep 300 & echo $! >"'"$d"'/payload.child.pid"; wait' >/dev/null 2>&1 &
        rpid=$!
        for ((i = 0; i < 300; i++)); do [[ -s $d/payload.child.pid ]] && break; sleep 0.1; done
        [[ -s $d/payload.child.pid ]] || { kill -KILL "$rpid" 2>/dev/null; exit 78; }
        kill -TERM "$rpid"
        # take the lock the moment it is free; the payload and its child must already be gone then
        exec 7>>"$d/lock5"
        flock -w 20 7 || { echo "lock-never-free"; pkill_pid=$(<"$d/payload.child.pid"); kill -KILL "$pkill_pid" "$(<"$d/payload.pid")" 2>/dev/null; exit 79; }
        alive=0
        for f in payload.pid payload.child.pid; do kill -0 "$(<"$d/$f")" 2>/dev/null && alive=1; done
        wait "$rpid" 2>/dev/null
        exit $((alive * 80))
    ) >"$d/kill-case.txt" 2>&1; rc=$?
    case $rc in
        77) echo "gate-selftest SKIP kill case (no 0x7550 card on this machine)" ;;
        0) echo "gate-selftest PASS killed runner: payload and its child were dead before the lock was free, and the lock was then free (exit 0)" ;;
        *) echo "gate-selftest FAIL killed runner: exit $rc (78 payload never started, 79 lock not free within 20 s, 80 payload still alive when the lock was taken)"; bad=$((bad + 1)) ;;
    esac
    unset R11_SELFTEST
    return "$bad"
}
