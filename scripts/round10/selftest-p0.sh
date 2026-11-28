#!/usr/bin/env bash
# Round 10 P0 self-test: dry run of pin_mclk.sh. No sudo (a stub sudo in PATH records any call: there must be none), no real sysfs
# (plain files in the scratch folder), a scratch lock, no GPU. Everything lands in runs/<UTC>-r10p0/ (untracked).
# A second process tries `flock -n` on the scratch lock every 20 ms and logs when it first succeeds.
# Usage: scripts/round10/selftest-p0.sh        Exit 0 only if every case passes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
DRIVER=${R10_SELFTEST_DRIVER:-$here/pin_mclk.sh}  # override only to show that a new case fails on an older driver
T=$repo/runs/$(date -u +%Y%m%dT%H%M%S)-r10p0
mkdir -p "$T/bin"
rc=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; rc=1; }
# stub sudo: any call is a failure of the test (the dry run must never reach sudo)
printf '#!/bin/sh\necho "sudo $*" >>"%s/sudo-calls.log"\nexit 1\n' "$T" >"$T/bin/sudo"
chmod +x "$T/bin/sudo"
export PATH=$T/bin:$PATH
export R10_DRY_WRITE_DELAY=0.15

# mk <case>: fresh scratch card, lock, run folder. Sets C (case dir), SYS, LOCKF, RUNF, RF (RESTORE-FAILED path).
mk() {
    C=$T/$1
    rm -rf "$C" && mkdir -p "$C/sys"
    SYS=$C/sys LOCKF=$C/lock RUNF=$C/run RF=$C/RESTORE-FAILED
    printf 'auto' >"$SYS/power_dpm_force_performance_level"
    printf '0: 96Mhz\n1: 456Mhz *\n2: 772Mhz\n3: 875Mhz\n4: 1124Mhz\n5: 1258Mhz\n' >"$SYS/pp_dpm_mclk"
    printf '0x7550' >"$SYS/device"
}
# drv <extra env...> -- args: runs the driver with the case's scratch paths
drv() {
    env R10_SYSFS="$SYS" R10_LOCK="$LOCKF" R10_RUN="$RUNF" R10_RESTORE_FAILED="$RF" "$@"
}
poller() { # poller <log>: logs the time of the first successful flock -n, then ends
    (while :; do
        if flock -n "$LOCKF" true; then date +%s.%N >"$1"; break; fi
        sleep 0.02
    done) &
    POLLER=$!
}
waitfor() { local w=0; while [[ ! -e $1 ]] && ((w < 300)); do sleep 0.05; w=$((w + 1)); done; [[ -e $1 ]]; }
ts_of() { grep -m1 "$1" "$RUNF/events.log" | cut -d' ' -f1; }  # time of the first events.log line matching
later() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 > b+0)}'; }  # later <a> <b>: a > b
lvl() { cat "$SYS/power_dpm_force_performance_level"; }

# ---- pass ----
mk pass
out=$(drv "$DRIVER" --dry-run --level 3 -- bash -c 'sleep 0.3; echo payload-ran')
r=$?
grep -q "RESTORE-OK" "$RUNF/events.log" && [[ $r == 0 && $(lvl) == auto && ! -e $RF && $(<"$RUNF/payload.out") == payload-ran ]] \
    && [[ $(cat "$SYS/.mclk-mask") == "0 1 2 3 4 5" && -e $RUNF/changed-by-us && -e $RUNF/pinned ]] \
    && grep -q "write 'manual'" "$RUNF/events.log" && grep -q "write '3' to pp_dpm_mclk" "$RUNF/events.log" \
    && ok "pass: normal run exit 0, pinned level 3 then restored; level reads auto, mask '0 1 2 3 4 5', no RESTORE-FAILED" \
    || { bad "pass: exit $r level $(lvl)"; cat "$RUNF/events.log"; }

# ---- break 1: SIGKILL the locked job; the lock must outlive it until the watchdog restored ----
b1() { # b1 <case> <env assignment or ''>: sets rs (watchdog restore time) and ac (lock acquire time); runs in this shell
    mk "$1"
    local envs=()
    [[ -n $2 ]] && envs=("$2")
    drv "${envs[@]}" "$DRIVER" --dry-run -- sleep 30 >/dev/null 2>&1 &
    local outer=$!
    waitfor "$RUNF/pinned" || { rs="" ac=none; return; }
    poller "$C/acquired"
    kill -KILL "$(cat "$RUNF/job.pid")"
    local w=0
    while [[ ! -e $C/acquired ]] && ((w < 200)); do sleep 0.05; w=$((w + 1)); done
    wait "$outer" 2>/dev/null
    waitfor "$RUNF/watchdog.rc"
    sleep 0.3
    kill "$POLLER" 2>/dev/null
    rs=$(grep -m1 'watchdog RESTORE-OK' "$RUNF/events.log" | cut -d' ' -f1)
    ac=$(cat "$C/acquired" 2>/dev/null || echo none)
}
b1 break1 ""
if [[ -n $rs && $ac != none ]] && later "$ac" "$rs" && [[ $(lvl) == auto ]] && ! kill -0 "$(cat "$RUNF/payload.pid")" 2>/dev/null; then
    ok "break 1: after SIGKILL of the locked job the lock was acquired at $ac, after the watchdog's restore line at $rs; level auto, payload gone"
else
    bad "break 1: restore '$rs', acquire '$ac', level $(lvl)"
fi
cp "$RUNF/events.log" "$C/events.excerpt"
b1 break1-nofd "R10_BREAK_WD_NOFD=1"
if [[ -z $rs || $ac == none ]]; then
    bad "break 1b: timestamps missing (restore '$rs', acquire '$ac'): the case did not run"
elif later "$ac" "$rs"; then
    bad "break 1b: the check did NOT notice a watchdog without the lock descriptor (acquire $ac after restore $rs)"
else
    ok "break 1b (the break inside the break): watchdog without the lock descriptor: lock acquired at $ac, BEFORE the restore line at $rs, so the same check FAILS as it must"
fi

# ---- break 2: pre-state is not auto ----
mk break2a
printf 'manual' >"$SYS/power_dpm_force_performance_level"
out=$(drv "$DRIVER" --dry-run -- sleep 0.2)
r=$?
if [[ $r == 20 && $out == *"NOT CHANGED: level was 'manual'"* && $(lvl) == manual && ! -e $RUNF/changed-by-us && ! -e $SYS/.mclk-mask ]] && ! grep -q write "$RUNF/events.log"; then
    ok "break 2a: level was manual: exit 20, 'NOT CHANGED', no write, no restore, level still manual"
else bad "break 2a: exit $r, out: $out"; fi
mk break2b
printf '1' >"$SYS/.mclk-mask"
out=$(drv "$DRIVER" --dry-run -- sleep 0.2)
r=$?
if [[ $r == 20 && $out == *"limited"* && $(lvl) == auto && ! -e $RUNF/changed-by-us && $(cat "$SYS/.mclk-mask") == 1 ]]; then
    ok "break 2b: level auto but the mask already limited: exit 20, nothing written"
else bad "break 2b: exit $r, out: $out"; fi

# ---- break 3: injected restore-write failure (normal path) ----
mk break3
out=$(drv R10_INJECT=restore-write "$DRIVER" --dry-run -- sleep 0.2)
r=$?
gate_refuses() { # gate_refuses: calls gate() from common.sh (fake repo, stub job script) with this case's RESTORE-FAILED path
    local fake=$C/fakerepo
    mkdir -p "$fake/scripts" "$fake/build/round9/bin" "$C/gaterun"
    printf '#!/bin/sh\necho STUB-JOB-RAN >>"%s/gate-job-ran"\n' "$C" >"$fake/scripts/study-llm.sh"
    chmod +x "$fake/scripts/study-llm.sh"
    (
        export R9_REPO=$fake CARD_ENV=/nonexistent R9_RESTORE_FAILED=$RF R9_HOLD=$C/nohold R9_PAUSE=0
        . "$repo/scripts/round9/common.sh"
        RUN=$C/gaterun
        gate probe 60 5 true >"$C/gate.out" 2>&1
    )
    echo $?
}
g=$(gate_refuses)
if [[ $r == 21 && -e $RF && $g != 0 && ! -e $C/gate-job-ran && $(lvl) == manual ]] && grep -q "RESTORE-FAILED" "$RUNF/events.log"; then
    ok "break 3: restore write failed: exit 21, RESTORE-FAILED exists, gate() refused (exit $g, job not started); level still manual (as designed: by-hand restore)"
else bad "break 3: exit $r RF=$([[ -e $RF ]] && echo y || echo n) gate=$g"; fi
rm -f "$RF"
g=$(gate_refuses)
[[ $g == 0 && -e $C/gate-job-ran ]] && ok "break 3 control: with RESTORE-FAILED removed the same gate() call starts its job" || bad "break 3 control: gate exit $g"

# ---- break 4: the watchdog has no credential ----
mk break4
drv R10_INJECT=wd-cred "$DRIVER" --dry-run -- sleep 30 >/dev/null 2>&1 &
outer=$!
waitfor "$RUNF/pinned"
kill -KILL "$(cat "$RUNF/job.pid")"
wait "$outer" 2>/dev/null
waitfor "$RUNF/watchdog.rc"
g=$(gate_refuses)
if [[ $(cat "$RUNF/watchdog.rc") != 0 && -e $RF && $g != 0 && $(lvl) == manual ]] && grep -q "watchdog RESTORE-FAILED" "$RUNF/events.log"; then
    ok "break 4: watchdog without credential: its exit status $(cat "$RUNF/watchdog.rc"), RESTORE-FAILED exists, gate() refuses; level left manual"
else bad "break 4: wd rc $(cat "$RUNF/watchdog.rc" 2>/dev/null) RF=$([[ -e $RF ]] && echo y || echo n) gate=$g"; fi

# ---- break 5: a second pinned job waits for the first ----
mk break5
drv R10_RUN="$C/runA" "$DRIVER" --dry-run -- sleep 1.2 >/dev/null 2>&1 &
A=$!
waitfor "$C/runA/pinned"
drv R10_RUN="$C/runB" "$DRIVER" --dry-run -- sleep 0.2 >"$C/B.out" 2>&1 &
B=$!
wait "$A"; ra=$?
wait "$B"; rb=$?
ta=$(grep -m1 'RESTORE-OK' "$C/runA/events.log" | cut -d' ' -f1)
tb=$(grep -m1 'locked job start' "$C/runB/events.log" | cut -d' ' -f1)
if [[ $ra == 0 && $rb == 0 && -n $ta && -n $tb ]] && later "$tb" "$ta" && [[ $(<"$C/runB/before.level") == auto && $(lvl) == auto ]]; then
    ok "break 5: second job's locked start ($tb) came after the first job's restore ($ta); it saw auto and both exited 0"
else bad "break 5: A=$ra B=$rb restore-A='$ta' start-B='$tb'"; cat "$C/B.out"; fi

# ---- cap: the watchdog kills a payload that runs too long and restores ----
mk cap
out=$(drv "$DRIVER" --dry-run --cap-s 1 -- sleep 30)
r=$?
if [[ $r == 22 && $(lvl) == auto && -e $RUNF/capped ]] && grep -q 'watchdog RESTORE-OK' "$RUNF/events.log"; then
    ok "cap: payload over 1 s was killed by the watchdog, which restored; exit 22, level auto"
else bad "cap: exit $r level $(lvl)"; cat "$RUNF/events.log"; fi

# ---- guards (extra) ----
mk guard
out=$(drv "$DRIVER" -- true 2>&1); r=$?
[[ $r == 2 && $out == *R10_ALLOW_REAL_SYSFS_WRITES* && ! -e $RUNF/locked ]] && ok "guard: a real run without the switch is refused (exit 2), nothing started" || bad "guard switch: exit $r"
printf '0x744c' >"$SYS/device"
out=$(drv "$DRIVER" --dry-run -- true 2>&1); r=$?
[[ $r == 2 && $out == *"not 0x7550"* && ! -e $RUNF/locked && $(lvl) == auto ]] && ok "guard: a sysfs path whose device reads 0x744c is refused (exit 2), nothing started" || bad "guard device: exit $r"
# the guards must be able to fail: copies with the guard lines removed must not refuse
# (the copies also lose the inner guards, which would otherwise refuse the same way)
sed -e '/R10-SWITCH-GUARD/,+3d' -e '/R10-INNER-GUARD/d' "$DRIVER" >"$T/bin/pin_nosw.sh"
sed -e '/R10-DEVICE-GUARD/,+3d' -e '/R10-INNER-GUARD/d' "$DRIVER" >"$T/bin/pin_nodev.sh"
chmod +x "$T/bin/pin_nosw.sh" "$T/bin/pin_nodev.sh"
mk guard2
printf '0x744c' >"$SYS/device"
out=$(drv "$T/bin/pin_nodev.sh" --dry-run -- true 2>&1); r=$?
[[ $r == 0 ]] && ok "guard break: a copy without the device guard accepts device 0x744c (the test can see the gap)" || bad "guard break (device): exit $r: $out"
mk guard3
out=$(drv "$T/bin/pin_nosw.sh" -- true 2>&1); r=$?
# without the switch guard (outer and inner) the real path starts the locked job (then stops at its missing password file: no sudo, no write)
[[ -e $RUNF/locked ]] && ok "guard break: a copy without the switch guard gets as far as starting the locked job (it stops at the missing password file, exit $r)" || bad "guard break (switch): exit $r: $out"

# ---- fix round (Neon review of 6eb2018) ----
sudo_log=$T/sudo-calls.log
# F1: the internal entry points must apply the same guards. R10_DRY unset (= real), no switch, a stub sudo in PATH, a secret file.
mk f1
printf 'stub-secret-not-real' >"$C/secret"
before_calls=$(cat "$sudo_log" 2>/dev/null | wc -l)
# timeout: an unguarded old driver's __watchdog 1 would wait for pid 1 forever
out=$(drv R10_SECRET_FILE="$C/secret" timeout 10 "$DRIVER" __locked true 2>&1); r=$?
out2=$(drv timeout 5 "$DRIVER" __watchdog 1 2>&1); r2=$?
after_calls=$(cat "$sudo_log" 2>/dev/null | wc -l)
if [[ $r == 2 && $r2 == 2 && $after_calls == "$before_calls" && -e $C/secret && ! -e $RUNF/watchdog.ready && $(lvl) == auto ]]; then
    ok "fix 1: __locked and __watchdog without the switch exit 2; the stub sudo was not called, the secret file is still there, no watchdog started"
else
    bad "fix 1: __locked exit $r, __watchdog exit $r2, sudo calls $((after_calls - before_calls)), secret $([[ -e $C/secret ]] && echo kept || echo unlinked)"
    # the old driver's stub-sudo call is expected there; keep the final sudo check honest for the new driver only
    [[ -n ${R10_SELFTEST_DRIVER:-} ]] && : >"$T/sudo-calls.log"
fi
# same call with the switch set but a wrong device: still refused
mk f1b
printf '0x744c' >"$SYS/device"
printf 'stub-secret-not-real' >"$C/secret"
out=$(drv R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved R10_SECRET_FILE="$C/secret" timeout 10 "$DRIVER" __locked true 2>&1); r=$?
[[ $r == 2 && -e $C/secret && $(lvl) == auto ]] && ok "fix 1b: __locked with the switch but device 0x744c is refused (exit 2)" || bad "fix 1b: exit $r"

# kill_at <case> <delay who> <wait for file> <kill who> <payload...>: runs the driver with a delay hook, waits for the file, kills.
# F2a: SIGKILL the job inside do_restore (after the claim): the watchdog takes over.
mk f2a
poller_start() { poller "$C/acquired"; }
drv R10_TEST_RESTORE_DELAY=3 R10_TEST_RESTORE_DELAY_WHO=job "$DRIVER" --dry-run -- sleep 0.4 >/dev/null 2>&1 &
outer=$!
if waitfor "$RUNF/restore-claim"; then
    poller "$C/acquired"
    kill -KILL "$(cat "$RUNF/job.pid")"
    wait "$outer" 2>/dev/null
    w=0; while [[ ! -e $C/acquired ]] && ((w < 400)); do sleep 0.05; w=$((w + 1)); done
    sleep 0.3; kill "$POLLER" 2>/dev/null
    ac=$(cat "$C/acquired" 2>/dev/null || echo none)
    rs=$(grep -m1 'RESTORE-OK' "$RUNF/events.log" | cut -d' ' -f1)
    if [[ $(lvl) == auto || -e $RF ]] && [[ $ac != none ]] && { [[ -e $RF ]] || { [[ -n $rs ]] && later "$ac" "$rs"; }; }; then
        ok "fix 2a: job SIGKILLed after taking the restore claim: lock freed at $ac only after a restore ('$rs'); level $(lvl), RESTORE-FAILED $([[ -e $RF ]] && echo yes || echo no)"
    else bad "fix 2a: level $(lvl) RF=$([[ -e $RF ]] && echo y || echo n) acquire '$ac' restore '$rs'"; fi
else
    kill -KILL "$(cat "$RUNF/job.pid" 2>/dev/null)" 2>/dev/null; wait "$outer" 2>/dev/null
    bad "fix 2a: no restore claim appeared"
fi
# F2b: the watchdog dies after claiming during a cap kill: the job takes over.
mk f2b
drv R10_TEST_RESTORE_DELAY=3 R10_TEST_RESTORE_DELAY_WHO=watchdog "$DRIVER" --dry-run --cap-s 1 -- sleep 30 >/dev/null 2>&1 &
outer=$!
if waitfor "$RUNF/restore-claim"; then
    poller "$C/acquired"
    jp=$(cat "$RUNF/job.pid")
    wdp=""
    for d in /proc/[0-9]*; do
        c=""
        { c=$(tr '\0' ' ' <"$d/cmdline"); } 2>/dev/null || continue
        [[ $c == *"__watchdog $jp"* ]] && wdp=${d#/proc/}
    done
    [[ -n $wdp ]] && kill -KILL "$wdp"
    wait "$outer" 2>/dev/null; r=$?
    # the killed watchdog's orphaned sleep may still hold the lock descriptor for a moment; wait for the lock to come free
    w=0; while [[ ! -e $C/acquired ]] && ((w < 400)); do sleep 0.05; w=$((w + 1)); done
    sleep 0.3; kill "$POLLER" 2>/dev/null
    ac=$(cat "$C/acquired" 2>/dev/null || echo none)
    rs=$(grep -m1 'RESTORE-OK' "$RUNF/events.log" | cut -d' ' -f1)
    if [[ -n $wdp && $ac != none ]] && { [[ -e $RF ]] || { [[ $(lvl) == auto && -n $rs ]] && later "$ac" "$rs"; }; }; then
        ok "fix 2b: watchdog SIGKILLed after claiming during a cap kill: the job took over; lock freed at $ac after the restore ('$rs'); level $(lvl), exit $r"
    else bad "fix 2b: wd pid '$wdp' level $(lvl) RF=$([[ -e $RF ]] && echo y || echo n) acquire '$ac' restore '$rs'"; fi
else
    bad "fix 2b: no restore claim appeared"; kill -KILL "$(cat "$RUNF/job.pid" 2>/dev/null)" 2>/dev/null; wait "$outer" 2>/dev/null
fi
# F3: SIGTERM during the normal restore must not re-enter do_restore.
mk f3
drv R10_TEST_RESTORE_DELAY=2 R10_TEST_RESTORE_DELAY_WHO=job "$DRIVER" --dry-run -- sleep 0.4 >"$C/f3.out" 2>&1 &
outer=$!
if waitfor "$RUNF/restore-claim"; then
    t0=$SECONDS
    kill -TERM "$(cat "$RUNF/job.pid")"
    wait "$outer"; r=$?
    el=$((SECONDS - t0))
    nok=$(grep -c 'RESTORE-OK' "$RUNF/events.log")
    if [[ $r == 0 && $(cat "$RUNF/restore.rc" 2>/dev/null) == 0 && $(lvl) == auto && ! -e $RF && $nok == 1 ]] && ((el < 20)); then
        ok "fix 3: SIGTERM during the normal restore: restore completed (restore.rc 0, level auto, one RESTORE-OK), exit $r, no deadlock (${el} s)"
    else bad "fix 3: exit $r restore.rc '$(cat "$RUNF/restore.rc" 2>/dev/null)' level $(lvl) RF=$([[ -e $RF ]] && echo y || echo n) restore-ok lines $nok after ${el} s"; fi
else bad "fix 3: no restore claim appeared"; kill -KILL "$(cat "$RUNF/job.pid" 2>/dev/null)" 2>/dev/null; wait "$outer" 2>/dev/null; fi
# F4: the watchdog inherits the credential descriptor (dry run, stub secret) and reads it from offset 0 twice; the payload does not get it.
mk f4
printf 'stub-secret-not-real' >"$C/secret"
drv R10_TEST_SECRET_FILE="$C/secret" R10_TEST_CRED_PROBE=1 "$DRIVER" --dry-run -- bash -c 'echo "var=${R10_CRED_FD:-unset}" >payload-env.txt; readlink /proc/self/fd/* 2>/dev/null | grep -c secret >>payload-env.txt; true' >/dev/null 2>&1
r=$?
jfd=$(grep -o 'credential descriptor [0-9]*' "$RUNF/events.log" | grep -o '[0-9]*$')
probe=$(cat "$RUNF/cred-probe" 2>/dev/null)
exp="fd=$jfd len1=20 len2=20 same=yes"
penv=$(cat "$RUNF/payload-env.txt" "payload-env.txt" 2>/dev/null | tr '\n' ' ')
rm -f payload-env.txt
if [[ $r == 0 && -n $jfd && $probe == "$exp" && ! -e $C/secret && $penv == "var=unset 0 " ]]; then
    ok "fix 4: stub secret unlinked at once; the watchdog saw the same descriptor number ($jfd) and read it from offset 0 twice (20 bytes each, equal); the payload has neither the variable nor the descriptor"
else bad "fix 4: exit $r job fd '$jfd' probe '$probe' secret $([[ -e $C/secret ]] && echo kept || echo unlinked) payload '$penv'"; fi
# F6: the driver itself refuses while RESTORE-FAILED exists (outer check and the locked job's own check)
mk f6
printf 'x' >"$RF"
out=$(drv "$DRIVER" --dry-run -- true 2>&1); r=$?
mkdir -p "$RUNF"
exec 9>>"$LOCKF" || { bad "fix 6: cannot open fd 9"; exec 9>&-; }
if flock -n 9; then
    out2=$(drv R10_DRY=1 "$DRIVER" __locked true 2>&1); r2=$?
else
    bad "fix 6: could not hold the scratch lock on fd 9"
    out2="" r2=99
fi
exec 9>&-
# R10_DRY=1 gets __locked past guard_exit, so the exit 2 below must come from the locked job's own RESTORE-FAILED check
if [[ $r == 2 && $r2 == 2 && $out2 == *"NOT CHANGED:"* && ! -e $RUNF/pinned && $(lvl) == auto ]] && ! grep -q "write" "$RUNF/events.log" 2>/dev/null; then
    ok "fix 6: with RESTORE-FAILED present the driver refuses (exit 2) and so does a direct __locked; nothing written"
else bad "fix 6: exit $r / $r2, out2: $out2"; fi

# ---- follow-ups from the Neon confirm review (CONFIRM-P0.md) ----
# N1: a dry run whose R10_LOCK is the hardcoded checkout default lock is refused BEFORE flock, even when STUDY_GPU_LOCK points elsewhere.
# The driver is copied into a scratch repo tree so its <repo>/runs/gpu.lock is a scratch file: the real lock is never touched.
mk n1
FR=$C/fakerepo
mkdir -p "$FR/scripts/round10" "$FR/runs"
cp "$DRIVER" "$FR/scripts/round10/pin_mclk.sh"
chmod +x "$FR/scripts/round10/pin_mclk.sh"
FLOCK=$FR/runs/gpu.lock
: >"$FLOCK"
n1run() { drv STUDY_GPU_LOCK="$C/other-lock" R10_LOCK="$FLOCK" R10_LOCK_WAIT=2 timeout 20 "$1" --dry-run -- true 2>&1; }
out=$(n1run "$FR/scripts/round10/pin_mclk.sh"); r=$?
if [[ $r == 2 && $out == *"a dry run needs R10_LOCK"* && ! -e $RUNF ]]; then
    ok "follow-up 1: dry run with R10_LOCK = the checkout default lock (STUDY_GPU_LOCK elsewhere) is refused before flock (exit 2, no run folder)"
else bad "follow-up 1: exit $r, out: $out"; fi
sed 's/ && \$(readlink -f "\$R10_LOCK") != "\$(readlink -f "\$REPO\/runs\/gpu\.lock")" \]\]/ ]]/' "$FR/scripts/round10/pin_mclk.sh" >"$FR/scripts/round10/pin_nolockchk.sh"
chmod +x "$FR/scripts/round10/pin_nolockchk.sh"
if cmp -s "$FR/scripts/round10/pin_mclk.sh" "$FR/scripts/round10/pin_nolockchk.sh"; then bad "follow-up 1 break: the sed did not change the copy"; else
    rm -rf "$RUNF"
    out=$(n1run "$FR/scripts/round10/pin_nolockchk.sh"); r=$?
    [[ -e $RUNF ]] && ok "follow-up 1 break: a copy without the default-lock check goes on to make the run folder and take the lock (exit $r), so the check above can fail" || bad "follow-up 1 break: copy still refused (exit $r): $out"
fi
# N2: __locked refuses at its very top: with no switch it must not even write job.pid, events.log or locked into the run folder.
mk n2
mkdir -p "$RUNF"
out=$(drv timeout 10 "$DRIVER" __locked true 2>&1); r=$?
if [[ $r == 2 && ! -e $RUNF/job.pid && ! -e $RUNF/events.log && ! -e $RUNF/locked ]]; then
    ok "follow-up 2: __locked refused by the guard writes nothing into the run folder (no job.pid, events.log, locked)"
else bad "follow-up 2: exit $r, run folder holds: $(ls "$RUNF" 2>/dev/null | tr '\n' ' ')"; fi
# N3a: restore.rc is written by rename, never by redirect. The link is planted after the claim and before put_rc
# (the run folder must start empty: a reused folder is refused). A redirect would write through the link.
mk n3a
printf 'untouched' >"$C/rc-target"
drv R10_TEST_RESTORE_DELAY=2 R10_TEST_RESTORE_DELAY_WHO=job "$DRIVER" --dry-run -- true >/dev/null 2>&1 &
outer=$!
if waitfor "$RUNF/restore-claim"; then
    ln -sfn "$C/rc-target" "$RUNF/restore.rc"
    wait "$outer"; r=$?
else
    r=99
    kill -KILL "$outer" 2>/dev/null; wait "$outer" 2>/dev/null
fi
if [[ $r == 0 && ! -L $RUNF/restore.rc && $(<"$RUNF/restore.rc") == 0 && $(<"$C/rc-target") == untouched ]]; then
    ok "follow-up 3a: restore.rc was replaced by rename (regular file holding 0); the symlink target was not written through"
else bad "follow-up 3a: exit $r, restore.rc $([[ -L $RUNF/restore.rc ]] && echo 'still a symlink' || echo regular), target '$(<"$C/rc-target")'"; fi
# N3b: alive()/claim_alive() parse /proc/<pid>/stat correctly when comm holds spaces and ')'. The functions are cut out of the driver.
mk n3b
fn_src=$(sed -n '/^pstat() {/,/^}/p;/^alive() {/,/^}/p;/^claim_alive() {/,/^}/p' "$DRIVER")
mkfifo "$C/fifo"
bash -c 'printf "a b) Z c" >/proc/self/comm; read -t 25 <>"$1"' _ "$C/fifo" &
LIVE=$!
bash -c 'bash -c "printf \"a b) c\" >/proc/self/comm" & echo $! >"$1"; exec sleep 6' _ "$C/zpid" &
ZPAR=$!
waitfor "$C/zpid"; sleep 0.3
ZOMB=$(<"$C/zpid")
want_start=$(sed 's/^.*) //' "/proc/$LIVE/stat" | awk '{print $20}')
res=$(bash -c "$fn_src"'
    st=$(pstat "$1") && [[ ${st#* } == "$3" ]] && echo live-start-ok
    alive "$1" && echo live-alive
    claim_alive "$1" "$3" && echo claim-match
    claim_alive "$1" 1 || echo claim-reused-pid-dead
    alive "$2" || echo zombie-not-alive
    alive 99999999 || echo gone-not-alive' _ "$LIVE" "$ZOMB" "$want_start" 2>&1 | tr '\n' ' ')
exp="live-start-ok live-alive claim-match claim-reused-pid-dead zombie-not-alive gone-not-alive "
if [[ $(<"/proc/$LIVE/comm") == "a b) Z c" && $(<"/proc/$ZOMB/comm") == "a b) c" && $res == "$exp" ]]; then
    ok "follow-up 3b: alive()/claim_alive() right for comm 'a b) Z c' (live, start time matches), a zombie with comm 'a b) c', a reused-pid start time and a missing pid"
else bad "follow-up 3b: got '$res' want '$exp' (comm live '$(<"/proc/$LIVE/comm" 2>/dev/null)' zombie '$(<"/proc/$ZOMB/comm" 2>/dev/null)')"; fi
kill "$LIVE" "$ZPAR" 2>/dev/null; wait "$LIVE" "$ZPAR" 2>/dev/null
# N3c: a restore claim whose pid is alive but whose recorded start time is not that process's (pid reuse) is taken over at once, not after 120 s.
n3c() { # n3c <driver>: after the payload starts, plants a claim naming a LIVE process with a wrong start time. Sets EL and RR.
    mk n3c
    drv timeout 20 "$1" --dry-run -- sleep 2 >/dev/null 2>&1 &
    local outer=$! lp="" jp="" d cmd t0
    EL=99
    RR=99
    if waitfor "$RUNF/payload.pid"; then
        mkdir -p "$RUNF/restore-claim"
        sleep 60 &
        lp=$!
        echo "$lp 1" >"$RUNF/restore-claim/pid"
        t0=$SECONDS
        wait "$outer"
        RR=$?
        EL=$((SECONDS - t0))
    else
        kill -KILL "$outer" 2>/dev/null || true
        wait "$outer" 2>/dev/null || true
    fi
    jp=$(cat "$RUNF/job.pid" 2>/dev/null || true)
    if [[ -n $jp ]]; then
        for d in /proc/[0-9]*; do
            { cmd=$(tr '\0' ' ' <"$d/cmdline"); } 2>/dev/null || continue
            [[ $cmd == *"__watchdog $jp"* ]] && kill -KILL "${d#/proc/}" 2>/dev/null || true
        done
    fi
    [[ -n $lp ]] && kill -KILL "$lp" 2>/dev/null || true
    [[ -n $lp ]] && wait "$lp" 2>/dev/null || true
}
n3c "$DRIVER"; el=$EL r=$RR
if [[ $r == 0 ]] && ((el < 12)) && grep -q 'died before restore.rc; taking over' "$RUNF/events.log" && [[ $(lvl) == auto && ! -e $RF ]]; then
    ok "follow-up 3c: a claim naming a live process with a different start time was taken over at once (${el} s, exit 0, level auto)"
else bad "follow-up 3c: exit $r after ${el} s, level $(lvl)"; fi
sed '/R10-STARTTIME/d' "$DRIVER" >"$T/bin/pin_nostart.sh"
chmod +x "$T/bin/pin_nostart.sh"
n3c "$T/bin/pin_nostart.sh"; el=$EL r=$RR
if [[ $r != 0 ]] && ! grep -q 'taking over' "$RUNF/events.log" 2>/dev/null; then
    ok "follow-up 3c break: a copy without the start-time check keeps waiting on the reused pid (killed by timeout after ${el} s, exit $r), so the case above can fail"
else bad "follow-up 3c break: the copy still took over (exit $r after ${el} s)"; fi
# N4: an R10_CRED_FD inherited from the caller is ignored on a dry run without a secret file, and is never put into the bash -c string.
mk n4
PWN=$C/pwned
out=$(drv R10_CRED_FD="1>&1; touch $PWN #" timeout 20 "$DRIVER" --dry-run -- bash -c 'echo "var=${R10_CRED_FD:-unset}" >"$1"' _ "$C/n4-env" 2>&1); r=$?
if [[ $r == 0 && ! -e $PWN && $(<"$C/n4-env") == var=unset ]] && ! grep -q 'credential descriptor' "$RUNF/events.log"; then
    ok "follow-up 4: an inherited R10_CRED_FD (with a command in it) is ignored on a dry run without a secret file: no command ran, payload runs, variable unset, no 'credential descriptor' logged"
else bad "follow-up 4: exit $r, pwned $([[ -e $PWN ]] && echo yes || echo no), env '$(cat "$C/n4-env" 2>/dev/null)'"; fi
sed '/unset R10_CRED_FD # only cred_open/d' "$DRIVER" >"$T/bin/pin_nounset.sh"
sed -i 's/\[\[ \${R10_CRED_FD:-} =~ \^\[0-9\]+\$ \]\] && extra/[[ -n ${R10_CRED_FD:-} ]] \&\& extra/' "$T/bin/pin_nounset.sh"
chmod +x "$T/bin/pin_nounset.sh"
mk n4b
if cmp -s "$DRIVER" "$T/bin/pin_nounset.sh"; then bad "follow-up 4 break: the sed did not change the copy"; else
    drv R10_CRED_FD="1>&1; touch $PWN #" timeout 20 "$T/bin/pin_nounset.sh" --dry-run -- true >/dev/null 2>&1
    [[ -e $PWN ]] && ok "follow-up 4 break: a copy with neither the unset nor the digit check runs the injected command (the test can see it)" || bad "follow-up 4 break: no injection even without both fixes"
fi

# Kimi #5: a second run in the same folder is refused before the lock, so stale done/restore.rc cannot skip a restore.
mk k5
out1=$(drv "$DRIVER" --dry-run -- true 2>&1); r1=$?
sz=$(wc -c <"$RUNF/events.log")
out2=$(drv "$DRIVER" --dry-run -- true 2>&1); r2=$?
sz2=$(wc -c <"$RUNF/events.log")
flock -n "$LOCKF" true; fl=$?
if [[ $r1 == 0 && $r2 == 2 && $out2 == *"not empty"* && $sz2 == "$sz" && $(lvl) == auto && $fl == 0 ]]; then
    ok "kimi 5: a second run in the same folder is refused before the lock (exit 2, events unchanged, level auto, lock free)"
else bad "kimi 5: first $r1 second $r2 flock $fl level $(lvl) events $sz->$sz2: $out2"; fi
awk '
    /^# R10-RUN-DIR$/ { print "mkdir -p \"$R10_RUN\" || exit 2"; skip=1; next }
    /^# R10-RUN-DIR-END$/ { skip=0; next }
    skip { next }
    { print }
' "$DRIVER" >"$T/bin/pin_reusedir.sh"
chmod +x "$T/bin/pin_reusedir.sh"
if cmp -s "$DRIVER" "$T/bin/pin_reusedir.sh"; then bad "kimi 5 break: the copy is unchanged"; else
    mk k5red
    drv timeout 30 "$T/bin/pin_reusedir.sh" --dry-run -- true >/dev/null 2>&1; rA=$?
    drv timeout 30 "$T/bin/pin_reusedir.sh" --dry-run -- true >/dev/null 2>&1; rB=$?
    for d in /proc/[0-9]*; do
        { cmd=$(tr '\0' ' ' <"$d/cmdline"); } 2>/dev/null || continue
        [[ $cmd == *"pin_reusedir.sh"* && $cmd == *__watchdog* ]] && kill -KILL "${d#/proc/}" 2>/dev/null || true
    done
    if [[ $rA == 0 && $(lvl) != auto ]]; then
        ok "kimi 5 break: a copy that reuses the folder leaves level '$(lvl)' after the second run (exit $rB)"
    else bad "kimi 5 break: first $rA second $rB level $(lvl) (the second run should leave a non-auto level)"; fi
fi
# Kimi #6: no sidecar. auto and at least two listed levels is the only known-full state.
mk k6auto
if [[ -e $SYS/.mclk-mask ]]; then bad "kimi 6 setup: a sidecar is present"; else
    out=$(drv "$DRIVER" --dry-run -- true 2>&1); r=$?
    if [[ $r == 0 && $(lvl) == auto && -e $RUNF/changed-by-us ]] && grep -q RESTORE-OK "$RUNF/events.log" && [[ $out != *"NOT CHANGED:"* ]]; then
        ok "kimi 6: real format, level auto, six levels, no sidecar: accepted and restored"
    else bad "kimi 6 auto: exit $r level $(lvl): $out"; fi
fi
mk k6man
printf 'manual' >"$SYS/power_dpm_force_performance_level"
rm -f "$SYS/.mclk-mask"
out=$(drv "$DRIVER" --dry-run -- true 2>&1); r=$?
if [[ $r == 20 && $out == *"NOT CHANGED: level was 'manual'"* && $out != *"mask 'all'"* && $(lvl) == manual ]] && ! grep -q "write '" "$RUNF/events.log"; then
    ok "kimi 6: real format, level manual: fail closed (exit 20), not reported as mask all"
else bad "kimi 6 manual: exit $r: $out"; fi
mk k6un
chmod 000 "$SYS/power_dpm_force_performance_level"
rm -f "$SYS/.mclk-mask"
out=$(drv "$DRIVER" --dry-run -- true 2>&1); r=$?
chmod 644 "$SYS/power_dpm_force_performance_level" 2>/dev/null || true
if [[ $r == 20 && $out == *"unreadable"* && $out != *"mask 'all'"* ]] && ! grep -q "write '" "$RUNF/events.log" 2>/dev/null; then
    ok "kimi 6: an unreadable performance level fails closed"
else bad "kimi 6 unreadable: exit $r: $out"; fi
awk '
    /^[[:space:]]*# R10-REAL-MASK$/ {
        print "    n=$(all_indices \"$R10_SYSFS/$MCLK_F\" | wc -w)"
        print "    n=${n//[[:space:]]/}"
        print "    if ((n >= 2)); then echo all; else echo \"limited: $n\"; fi"
        skip=1; next
    }
    /^[[:space:]]*# R10-REAL-MASK-END$/ { skip=0; next }
    skip { next }
    { print }
' "$DRIVER" >"$T/bin/pin_maskcount.sh"
chmod +x "$T/bin/pin_maskcount.sh"
if cmp -s "$DRIVER" "$T/bin/pin_maskcount.sh"; then bad "kimi 6 break: the copy is unchanged"; else
    mk k6red
    printf 'manual' >"$SYS/power_dpm_force_performance_level"
    rm -f "$SYS/.mclk-mask"
    # The job also refuses a non-auto level, so the old bug is in mask_state itself: six listed levels became "all".
    {
        sed -n '/^LEVEL_F=/p;/^MCLK_F=/p;/^dry() /p;/^level_now() /p;/^all_indices() /p' "$DRIVER"
        sed -n '/^mask_state() {/,/^}/p' "$T/bin/pin_maskcount.sh"
        echo 'mask_state'
    } >"$C/runmask.sh"
    got=$(R10_DRY=1 R10_SYSFS="$SYS" R10_RUN="$RUNF" bash "$C/runmask.sh")
    if [[ $got == all ]]; then
        ok "kimi 6 break: the old mask_state reports 'all' for a manual level with six listed levels"
    else bad "kimi 6 break: old mask_state said '$got'"; fi
fi
# Kimi #7: __locked with the owner switch set, and fd 9 not the lock, writes nothing and does not call sudo.
mk k7
printf 'stub-secret-not-real' >"$C/secret"
before=0
[[ -f $sudo_log ]] && before=$(wc -l <"$sudo_log")
out=$(drv R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved R10_SECRET_FILE="$C/secret" timeout 10 "$DRIVER" __locked true 2>&1); r=$?
after=0
[[ -f $sudo_log ]] && after=$(wc -l <"$sudo_log")
if [[ $r == 2 && $out == *"fd 9"* && ! -e $RUNF/job.pid && -e $C/secret && $after == "$before" && $(lvl) == auto ]]; then
    ok "kimi 7: __locked with the owner switch but without fd 9 exits 2 before any run file; stub sudo not called"
else bad "kimi 7: exit $r sudo lines $before->$after job.pid $([[ -e $RUNF/job.pid ]] && echo present || echo absent): $out"; fi
sed '/R10-LOCK-FD/d' "$DRIVER" >"$T/bin/pin_nolockfd.sh"
chmod +x "$T/bin/pin_nolockfd.sh"
if cmp -s "$DRIVER" "$T/bin/pin_nolockfd.sh"; then bad "kimi 7 break: the copy is unchanged"; else
    mk k7red
    mkdir -p "$RUNF"
    printf 'stub-secret-not-real' >"$C/secret"
    mkdir -p "$C/bin"
    printf '#!/bin/sh\necho "sudo $*" >>"%s/private-sudo.log"\nexit 1\n' "$C" >"$C/bin/sudo"
    chmod +x "$C/bin/sudo"
    oldpath=$PATH
    PATH="$C/bin:$PATH"
    out=$(drv R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved R10_SECRET_FILE="$C/secret" timeout 10 "$T/bin/pin_nolockfd.sh" __locked true 2>&1); r=$?
    PATH=$oldpath
    jp=$(cat "$RUNF/job.pid" 2>/dev/null || true)
    if [[ -n $jp ]]; then
        for d in /proc/[0-9]*; do
            { cmd=$(tr '\0' ' ' <"$d/cmdline"); } 2>/dev/null || continue
            [[ $cmd == *"__watchdog $jp"* ]] && kill -KILL "${d#/proc/}" 2>/dev/null || true
        done
    fi
    if [[ -e $RUNF/job.pid || -s $C/private-sudo.log ]]; then
        ok "kimi 7 break: without the fd 9 check, __locked writes a run file or calls sudo (exit $r)"
    else bad "kimi 7 break: still refused with no run file and no sudo call (exit $r): $out"; fi
    # the private stub must be the one that ran, never the suite stub
    [[ -e $T/sudo-calls.log ]] && bad "kimi 7 break: the suite sudo stub was called"
fi

# Kimi R2 #1: fd 9 open on the right file is not proof of the lock. __locked must also see an exclusive flock held by that very descriptor.
# k7run <case> <driver> <holder 0|1>: fd 9 opened on the lock WITHOUT flock; holder=1 means another process holds the flock. Sets r, out, rj (job.pid present), rs (private sudo called).
k7run() {
    mk "$1"
    mkdir -p "$RUNF" "$C/bin"
    printf 'stub-secret-not-real' >"$C/secret"
    printf '#!/bin/sh\necho "sudo $*" >>"%s/private-sudo.log"\nexit 1\n' "$C" >"$C/bin/sudo"
    chmod +x "$C/bin/sudo"
    local hp="" oldpath=$PATH
    if [[ $3 == 1 ]]; then
        flock "$LOCKF" sleep 20 &
        hp=$!
        sleep 0.4
    fi
    PATH="$C/bin:$PATH"
    out=$(
        exec 9>>"$LOCKF"
        drv R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved R10_SECRET_FILE="$C/secret" timeout 10 "$2" __locked true 2>&1
    ); r=$?
    PATH=$oldpath
    [[ -n $hp ]] && { kill "$hp" 2>/dev/null; wait "$hp" 2>/dev/null; }
    rj=0 rs=0
    [[ -e $RUNF/job.pid ]] && rj=1
    [[ -s $C/private-sudo.log ]] && rs=1
    local jp
    jp=$(cat "$RUNF/job.pid" 2>/dev/null || true)
    if [[ -n $jp ]]; then
        for d in /proc/[0-9]*; do
            { cmd=$(tr '\0' ' ' <"$d/cmdline"); } 2>/dev/null || continue
            [[ $cmd == *"__watchdog $jp"* ]] && kill -KILL "${d#/proc/}" 2>/dev/null || true
        done
    fi
}
k7run k7unlocked "$DRIVER" 0
if [[ $r == 2 && $out == *"fd 9"* && $rj == 0 && $rs == 0 && -e $C/secret && $(lvl) == auto ]]; then
    ok "kimi R2 1a: fd 9 open on the lock but never flocked: __locked exits 2 before any run file or sudo call"
else bad "kimi R2 1a: exit $r job.pid $rj sudo $rs: $out"; fi
k7run k7held "$DRIVER" 1
if [[ $r == 2 && $out == *"fd 9"* && $rj == 0 && $rs == 0 && -e $C/secret && $(lvl) == auto ]]; then
    ok "kimi R2 1b: another process holds the flock and this fd 9 does not: __locked exits 2 before any run file or sudo call"
else bad "kimi R2 1b: exit $r job.pid $rj sudo $rs: $out"; fi
# red: an inode-only check (the old behaviour) lets both through
sed '/^    \[\[ -n \$want && \$want == "\$got" \]\] || return 1$/a\    return 0' "$DRIVER" >"$T/bin/pin_inodeonly.sh"
chmod +x "$T/bin/pin_inodeonly.sh"
if cmp -s "$DRIVER" "$T/bin/pin_inodeonly.sh"; then bad "kimi R2 1 break: the copy is unchanged"; else
    k7run k7redA "$T/bin/pin_inodeonly.sh" 0
    ra=$((rj + rs))
    k7run k7redB "$T/bin/pin_inodeonly.sh" 1
    rb=$((rj + rs))
    if ((ra > 0 && rb > 0)); then
        ok "kimi R2 1 break: with an inode-only fd 9 check, both the unlocked and the competing-holder case get past the guard"
    else bad "kimi R2 1 break: red copy still refused (unlocked $ra, competing $rb)"; fi
fi

# Real-hardware regression (2026-11-26): amdgpu appends a trailing space to every inactive pp_dpm_mclk line
# ("1: 456Mhz ") and no trailing space after "*" on the active line ("1: 456Mhz *"). If the active level moves
# between before and after restore (e.g. 1 -> 3), listing() must strip trailing spaces on all lines so restore
# does not falsely trigger RESTORE-FAILED while level=auto and mask=all.
mk_amdgpu_mclk() { # mk_amdgpu_mclk <file> <active_idx>
    local f=$1 act=$2 i
    local vals=("0: 96Mhz" "1: 456Mhz" "2: 772Mhz" "3: 875Mhz" "4: 1124Mhz" "5: 1258Mhz")
    : >"$f"
    for i in "${!vals[@]}"; do
        if [[ $i == "$act" ]]; then
            printf '%s *\n' "${vals[$i]}" >>"$f"
        else
            printf '%s \n' "${vals[$i]}" >>"$f"
        fi
    done
}
mk hwstar
mk_amdgpu_mclk "$SYS/pp_dpm_mclk" 1
out=$(drv "$DRIVER" --dry-run --level 5 -- bash -c "printf '0: 96Mhz \n1: 456Mhz \n2: 772Mhz \n3: 875Mhz *\n4: 1124Mhz \n5: 1258Mhz \n' >'$SYS/pp_dpm_mclk'" 2>&1); r=$?
if [[ $r == 0 && $(lvl) == auto && ! -e $RF ]] && grep -q 'RESTORE-OK level=auto mask=all' "$RUNF/events.log"; then
    ok "hw-star-move: active '*' moving 1->3 with amdgpu trailing spaces still reports RESTORE-OK (no false RESTORE-FAILED)"
else
    bad "hw-star-move: exit $r lvl=$(lvl) rf=$([[ -e $RF ]] && echo present || echo absent): $out"
fi
python3 -I - "$DRIVER" "$T/bin/pin_oldlisting.sh" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").readlines()
out = []
for ln in lines:
    if "# R10-LISTING-NORM" in ln:
        out.append('listing() { sed \'s/[[:space:]]*\\*[[:space:]]*$//\' "$R10_SYSFS/$MCLK_F" 2>/dev/null; }\n')
    else:
        out.append(ln)
open(dst, "w", encoding="utf-8").writelines(out)
PY
chmod +x "$T/bin/pin_oldlisting.sh"
if cmp -s "$DRIVER" "$T/bin/pin_oldlisting.sh"; then bad "hw-star-move break: copy unchanged"; else
    mk hwstarred
    mk_amdgpu_mclk "$SYS/pp_dpm_mclk" 1
    out=$(drv "$T/bin/pin_oldlisting.sh" --dry-run --level 5 -- bash -c "printf '0: 96Mhz \n1: 456Mhz \n2: 772Mhz \n3: 875Mhz *\n4: 1124Mhz \n5: 1258Mhz \n' >'$SYS/pp_dpm_mclk'" 2>&1); r=$?
    if [[ $r == 21 && -e $RF ]]; then
        ok "hw-star-move break: old listing() sed falsely exits 21 (RESTORE-FAILED) when active '*' moves 1->3"
    else
        bad "hw-star-move break: expected exit 21 + RESTORE-FAILED, got exit $r rf=$([[ -e $RF ]] && echo present || echo absent): $out"
    fi
fi

# Kimi follow-up (2026-11-26): narrow the SIGKILL window between writing `manual` and creating `changed-by-us`.
# With `pin-attempt` written before the first write, killing the locked job right after `manual` still causes
# the watchdog to restore `auto` + full mask before releasing the lock.
mk pinwin
drv R10_TEST_PAUSE_BEFORE_MARK=2 "$DRIVER" --dry-run --level 3 -- sleep 5 >"$C/out" 2>&1 &
p=$!
w=0; while [[ $(lvl) != manual ]] && ((w < 100)); do sleep 0.05; w=$((w + 1)); done
jp=$(cat "$RUNF/job.pid" 2>/dev/null || true)
[[ -n $jp ]] && kill -KILL "$jp" 2>/dev/null
wait "$p" 2>/dev/null
# wait for lock free (watchdog done restoring)
if ! flock -w 10 "$LOCKF" true; then bad "pin-attempt-window: lock not freed within 10 s"; fi
if [[ $(lvl) == auto && ! -e $RUNF/changed-by-us && -e $RUNF/pin-attempt && ! -e $RF ]] && grep -q 'RESTORE-OK level=auto mask=all' "$RUNF/events.log"; then
    ok "pin-attempt-window: SIGKILL between manual write and changed-by-us still restores to auto via pin-attempt"
else
    bad "pin-attempt-window: lvl=$(lvl) cbu=$([[ -e $RUNF/changed-by-us ]] && echo y || echo n) pa=$([[ -e $RUNF/pin-attempt ]] && echo y || echo n) rf=$([[ -e $RF ]] && echo y || echo n)"
fi
sed '/R10-PIN-ATTEMPT$/d' "$DRIVER" >"$T/bin/pin_noattempt.sh"
chmod +x "$T/bin/pin_noattempt.sh"
if cmp -s "$DRIVER" "$T/bin/pin_noattempt.sh"; then bad "pin-attempt-window break: copy unchanged"; else
    mk pinwinred
    drv R10_TEST_PAUSE_BEFORE_MARK=2 "$T/bin/pin_noattempt.sh" --dry-run --level 3 -- sleep 5 >"$C/out" 2>&1 &
    p=$!
    w=0; while [[ $(lvl) != manual ]] && ((w < 100)); do sleep 0.05; w=$((w + 1)); done
    jp=$(cat "$RUNF/job.pid" 2>/dev/null || true)
    [[ -n $jp ]] && kill -KILL "$jp" 2>/dev/null
    wait "$p" 2>/dev/null
    if ! flock -w 10 "$LOCKF" true; then bad "pin-attempt-window break: lock not freed within 10 s"; fi
    if [[ $(lvl) == manual ]] && grep -q 'RESTORE-SKIPPED nothing was changed by us' "$RUNF/events.log"; then
        ok "pin-attempt-window break: without pin-attempt, SIGKILL in that window skips restore and leaves level manual"
    else
        bad "pin-attempt-window break: expected manual + RESTORE-SKIPPED, got lvl=$(lvl)"
    fi
fi

# Dry-run coverage for the job-side skip branch: pin-attempt exists, first write failed cleanly (state still auto/all).
mk pinskip
out=$(drv R10_INJECT=pin-write "$DRIVER" --dry-run --level 3 -- true 2>&1); r=$?
if [[ $r == 23 && $(lvl) == auto && -e $RUNF/pin-attempt && ! -e $RUNF/changed-by-us && ! -e $RF ]] \
    && grep -q 'RESTORE-SKIPPED pin attempted but state still auto/all' "$RUNF/events.log"; then
    ok "pin-attempt-skip: failed first write leaves auto/all untouched and logs RESTORE-SKIPPED (exit 23)"
else
    bad "pin-attempt-skip: exit $r lvl=$(lvl) rf=$([[ -e $RF ]] && echo present || echo absent): $out"
fi

[[ -e $T/sudo-calls.log ]] && bad "the stub sudo was called: $(cat "$T/sudo-calls.log")" || ok "sudo was never called in any case"
echo "scratch: $T"
exit $rc
