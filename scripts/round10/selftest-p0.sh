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
        [[ $(tr '\0' ' ' <"$d/cmdline" 2>/dev/null) == *"__watchdog $jp"* ]] && wdp=${d#/proc/}
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
out2=$(drv "$DRIVER" __locked true 2>&1); r2=$?
if [[ $r == 2 && $r2 == 2 && ! -e $RUNF/pinned && $(lvl) == auto ]] && ! grep -q "write" "$RUNF/events.log" 2>/dev/null; then
    ok "fix 6: with RESTORE-FAILED present the driver refuses (exit 2) and so does a direct __locked; nothing written"
else bad "fix 6: exit $r / $r2"; fi

[[ -e $T/sudo-calls.log ]] && bad "the stub sudo was called: $(cat "$T/sudo-calls.log")" || ok "sudo was never called in any case"
echo "scratch: $T"
exit $rc
