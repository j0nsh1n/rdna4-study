#!/usr/bin/env bash
# Round 10 P0: pin the memory clock (power_dpm_force_performance_level=manual + a level index in pp_dpm_mclk) for ONE payload,
# and restore `auto` plus the full mask under the same shared lock, whatever happens to this script.
# Redesign of round 9's stable_clocks.sh (see measurements/round10-p0.md). Differences that matter:
#   - Nothing touches sysfs outside the locked job. The outer part only checks guards, makes the run folder and takes the lock.
#   - Pre-state rule: nothing is written unless the level reads `auto` and the mclk mask allows all levels. Otherwise exit 20
#     ("NOT CHANGED: level was X"). The file `changed-by-us` appears only after our own successful write, and only that file
#     allows a restore.
#   - A watchdog (own session, inherits the lock descriptor) outlives a SIGKILL of the locked job; the lock is held until the
#     watchdog has restored. It also enforces the runtime cap.
#   - A failed restore writes RESTORE-FAILED (fixed place, see R10_RESTORE_FAILED); gate() in scripts/round9/common.sh refuses
#     to start any job while that file exists.
#
# Usage: pin_mclk.sh [--dry-run] [--level N] [--cap-s S] -- <payload command...>
# Environment (all of these are exported to the locked job):
#   R10_SYSFS            REQUIRED. The card's device directory (holds power_dpm_force_performance_level, pp_dpm_mclk, `device`).
#   R10_LOCK             The shared lock file. Real runs default to $STUDY_GPU_LOCK or <repo>/runs/gpu.lock; a dry run REQUIRES it
#                        to be set and refuses the real lock path.
#   R10_RUN              Run folder (default <repo>/runs/<UTC>-r10p0).
#   R10_RESTORE_FAILED   Where RESTORE-FAILED is written (default <repo>/runs/RESTORE-FAILED,
#                        the same place gate() looks; R9_RESTORE_FAILED is the same switch on the gate side, both default alike).
#   R10_CAP_S            Hard time limit of the pinned payload (default 120 s); the watchdog kills it and restores.
#   R10_SECRET_FILE      Real runs only: the password file (private prompt, delivered as a file). Opened as a descriptor and
#                        unlinked at once by cred_open; never an argument.
#   R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved   THE SWITCH. A real (non --dry-run) run refuses without it. Document it, never
#                        set it from a script or test; the owner sets it by hand for the first real run.
# Exit: 0 payload ok; 2 guard or usage refusal (nothing touched); 6 lock wait expired; 20 NOT CHANGED (pre-state not auto);
#       21 restore failed (RESTORE-FAILED written); 22 payload hit the cap; 23 the pin write failed (restored); else the payload's.
# Dry run only: R10_INJECT=restore-write (writes of auto fail), wd-cred (the watchdog has no credential); R10_DRY_WRITE_DELAY (s per
# simulated write, stands in for sudo latency); R10_BREAK_WD_NOFD=1 (start the watchdog WITHOUT the lock descriptor: the break).
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
REPO=$(cd "$here/../.." && pwd)
DEFAULT_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
RESTORE_FAILED=${R10_RESTORE_FAILED:-${R9_RESTORE_FAILED:-$REPO/runs/RESTORE-FAILED}}
FULLMASK_FALLBACK="0 1 2 3 4 5"
LEVEL_F=power_dpm_force_performance_level
MCLK_F=pp_dpm_mclk
PROG=pin_mclk.sh

# ---- shared helpers -------------------------------------------------------------------------------------------------
now() { date +%s.%N; }
log() { printf '%s %s %s\n' "$(now)" "${R10_WHO:-outer}" "$*" >>"$R10_RUN/events.log"; }
alive() { # alive <pid>: running and not a zombie
    [[ -r /proc/$1/stat ]] && [[ $(awk '{print $3}' "/proc/$1/stat" 2>/dev/null) != Z ]]
}
dry() { [[ ${R10_DRY:-0} == 1 ]]; }
injected() { [[ ,${R10_INJECT:-}, == *,"$1",* ]]; }

level_now() { cat "$R10_SYSFS/$LEVEL_F" 2>/dev/null; }
listing() { sed 's/[[:space:]]*\*[[:space:]]*$//' "$R10_SYSFS/$MCLK_F" 2>/dev/null; }
all_indices() { awk -F: '/^[0-9]+:/{printf "%s%s", (n++?" ":""), $1}' "$1"; }
# mask_state: "all" or "limited: <text>". Dry runs model the driver's soft limit in a sidecar file. A real card's pp_dpm_mclk
# marks the current clock with `*`, not the mask, so the real branch can only check that the levels are listed (NOT MEASURED).
mask_state() {
    if dry; then
        local m
        m=$(cat "$R10_SYSFS/.mclk-mask" 2>/dev/null)
        [[ -z $m || $m == "$(all_indices "$R10_RUN/before.mclk")" ]] && echo all || echo "limited: $m"
    else
        (($(all_indices "$R10_SYSFS/$MCLK_F" | wc -w) >= 2)) && echo all || echo "limited: fewer than 2 levels listed"
    fi
}

# ---- the real-run guards, shared by every path that can write ------------------------------------------------------------
# guard: 0 only if this process may touch the card: the device reads 0x7550 and, for a real run, the owner's switch is set; for a
# dry run the card is not under /sys and the lock is not the default lock. The internal entry points
# (__locked, __watchdog) are reachable without the outer checks, so cred_open, credential_step and sysfs_write call this too.
guard() {
    [[ -n ${R10_SYSFS:-} && -d $R10_SYSFS ]] || return 1
    [[ $(cat "$R10_SYSFS/device" 2>/dev/null) == 0x7550 ]] || return 1
    if dry; then
        [[ -n ${R10_LOCK:-} ]] || return 1
        local l
        l=$(readlink -f "$R10_LOCK")
        [[ $l != "$(readlink -f "$DEFAULT_LOCK")" ]] || return 1
        [[ $(readlink -f "$R10_SYSFS") != /sys/* ]]
    else
        [[ ${R10_ALLOW_REAL_SYSFS_WRITES:-} == yes-owner-approved ]]
    fi
}
# guard_exit: refuse and stop (entry points)
guard_exit() {
    guard || { echo "$PROG: internal entry refused (switch/device/dry-run guard). Nothing changed." >&2; exit 2; }
}

# ---- credentials: ONE function the self-test stubs ------------------------------------------------------------------------
# credential_step: 0 only if a write may be attempted now. Dry run: a stub (fails for the watchdog under R10_INJECT=wd-cred).
# Real: `sudo -S -p '' -v` reading /proc/self/fd/N, a fresh open of the unlinked password file, so the offset is 0. The path is
# never an argument. NOT RUN BY ANY TEST.
credential_step() {
    if dry; then
        [[ ${R10_WHO:-} == watchdog ]] && injected wd-cred && return 1
        return 0
    fi
    guard || return 1 # R10-INNER-GUARD
    [[ -n ${R10_CRED_FD:-} ]] || return 1
    sudo -S -p '' -v <"/proc/self/fd/$R10_CRED_FD" 2>/dev/null
}
# cred_open: real runs only, in the locked job: open the password file as a descriptor, unlink it at once, drop the variable.
cred_open() {
    guard || return 1 # R10-INNER-GUARD
    [[ -n ${R10_SECRET_FILE:-} && -r $R10_SECRET_FILE ]] || return 1
    exec {R10_CRED_FD}<"$R10_SECRET_FILE" || return 1
    export R10_CRED_FD # the watchdog (own session, same descriptor table copy) needs the number
    rm -f -- "$R10_SECRET_FILE"
    unset R10_SECRET_FILE
}

# sysfs_write <file name> <value>: the only place that writes sysfs.
sysfs_write() {
    local f=$1 v=$2
    guard || { log "write '$v' to $f: guard refused"; return 1; } # R10-INNER-GUARD
    if dry; then
        credential_step || { log "write '$v' to $f: no credential"; return 1; }
        sleep "${R10_DRY_WRITE_DELAY:-0}"
        if [[ ${R10_RESTORING:-0} == 1 && $f == "$LEVEL_F" ]] && injected restore-write; then
            log "write '$v' to $f: FAILED (injected)"
            return 1
        fi
        if [[ $f == "$MCLK_F" ]]; then printf '%s' "$v" >"$R10_SYSFS/.mclk-mask"; else printf '%s' "$v" >"$R10_SYSFS/$f"; fi
        log "write '$v' to $f"
        return 0
    fi
    credential_step || { log "write '$v' to $f: no credential"; return 1; }
    sudo -n bash -c 'printf "%s" "$2" > "$1"' _ "$R10_SYSFS/$f" "$v" </dev/null 2>/dev/null
    local rc=$?
    log "write '$v' to $f: status $rc"
    return $rc
}

# ---- restore: under the lock, claimed once (the parent and the watchdog never both write) ------------------------------------
# do_restore: 0 only if both files read back as before. Result in restore.rc; whoever loses the claim waits for it.
# The claim is a directory that appears already holding its owner's pid (claim_try). A waiter whose claimer died before restore.rc
# appeared takes over, so a killed job or watchdog never leaves the card in manual with the lock released and no RESTORE-FAILED.
claim_try() {
    local t=$R10_RUN/restore-claim.tmp.$$
    rm -rf "$t"
    mkdir "$t" && echo $$ >"$t/pid" || return 1
    mv -T "$t" "$R10_RUN/restore-claim" 2>/dev/null && return 0
    rm -rf "$t"
    return 1
}
do_restore() {
    # after a cap kill the watchdog restores (the design says so); the job only waits for its result, unless the watchdog is gone
    local only_wait=0 w=0 cp
    [[ ${R10_WHO:-} == job && -e $R10_RUN/capped ]] && only_wait=1
    while :; do
        if ((!only_wait)) || { [[ ! -d $R10_RUN/restore-claim ]] && ! alive "${WD:-0}"; }; then
            claim_try && break
        fi
        [[ -e $R10_RUN/restore.rc ]] && return "$(cat "$R10_RUN/restore.rc")"
        cp=$(cat "$R10_RUN/restore-claim/pid" 2>/dev/null)
        if [[ -n $cp ]] && ! alive "$cp" && [[ ! -e $R10_RUN/restore.rc ]]; then
            log "restore claimer $cp died before restore.rc; taking over"
            mv -T "$R10_RUN/restore-claim" "$R10_RUN/restore-claim.dead.$cp.$$" 2>/dev/null
            only_wait=0
            continue
        fi
        w=$((w + 1))
        if ((w > 2400)); then
            log "RESTORE-FAILED claimer $cp still running after 120 s"
            echo "RESTORE-FAILED $(date -u +%FT%TZ) run $R10_RUN: restore claimer $cp did not finish; check $R10_SYSFS/$LEVEL_F by hand" >"$RESTORE_FAILED"
            return 1
        fi
        sleep 0.05
    done
    dry && [[ -n ${R10_TEST_RESTORE_DELAY_WHO:-} && ${R10_TEST_RESTORE_DELAY_WHO} == "${R10_WHO:-}" ]] && sleep "${R10_TEST_RESTORE_DELAY:-0}"
    local rc=0 mask
    if [[ ! -e $R10_RUN/changed-by-us ]]; then
        log "RESTORE-SKIPPED nothing was changed by us"
        echo 0 >"$R10_RUN/restore.rc"
        return 0
    fi
    R10_RESTORING=1
    mask=$(all_indices "$R10_RUN/before.mclk")
    [[ -n $mask ]] || mask=$FULLMASK_FALLBACK
    sysfs_write "$MCLK_F" "$mask" || rc=1
    sysfs_write "$LEVEL_F" auto || rc=1
    local lv ms
    lv=$(level_now)
    ms=$(mask_state)
    if ((rc == 0)) && [[ $lv == auto && $ms == all ]] && [[ $(listing) == "$(cat "$R10_RUN/before.mclk.norm")" ]]; then
        log "RESTORE-OK level=$lv mask=$ms"
    else
        rc=1
        log "RESTORE-FAILED level now '$lv', mask '$ms'. Restore by hand: echo auto | sudo tee $R10_SYSFS/$LEVEL_F"
        {
            echo "RESTORE-FAILED $(date -u +%FT%TZ) run $R10_RUN"
            echo "level now '$lv', mask '$ms'; by hand: echo auto | sudo tee $R10_SYSFS/$LEVEL_F, then write '$mask' to $R10_SYSFS/$MCLK_F"
            echo "Remove this file only after checking both files; gate() refuses to start jobs while it exists."
        } >"$RESTORE_FAILED"
    fi
    echo "$rc" >"$R10_RUN/restore.rc"
    return $rc
}

# ---- the watchdog ---------------------------------------------------------------------------------------------------------
watchdog() {
    local ppid=$1 t0=0 started=0
    guard_exit # R10-INNER-GUARD
    trap '' INT TERM HUP # a restore must not be cut short by a stray signal
    R10_WHO=watchdog
    if dry && [[ ${R10_TEST_CRED_PROBE:-0} == 1 && -n ${R10_CRED_FD:-} ]]; then # self-test: is the descriptor inherited and re-readable?
        local r1 r2
        r1=$(<"/proc/self/fd/$R10_CRED_FD") r2=$(<"/proc/self/fd/$R10_CRED_FD")
        printf 'fd=%s len1=%s len2=%s same=%s\n' "$R10_CRED_FD" "${#r1}" "${#r2}" "$([[ -n $r1 && $r1 == "$r2" ]] && echo yes || echo no)" >"$R10_RUN/cred-probe"
    fi
    : >"$R10_RUN/watchdog.ready"
    log "started for parent $ppid"
    while :; do
        [[ -e $R10_RUN/done ]] && { log "done token seen; exit without restoring"; exit 0; }
        alive "$ppid" || break
        if [[ -e $R10_RUN/payload.pid ]]; then
            ((started)) || { started=1; t0=$SECONDS; }
            if ((SECONDS - t0 >= R10_CAP_S)); then
                : >"$R10_RUN/capped"
                log "runtime cap ${R10_CAP_S}s reached"
                break
            fi
        fi
        sleep 0.05
    done
    [[ -e $R10_RUN/done ]] && exit 0
    log "restoring (parent alive: $(alive "$ppid" && echo yes || echo no))"
    kill_payload
    do_restore
    local rc=$?
    echo "$rc" >"$R10_RUN/watchdog.rc"
    # a live parent (cap case) finishes on its own; wait for its `done` so the lock stays held until then
    local w=0
    while alive "$ppid" && [[ ! -e $R10_RUN/done ]] && ((w < 1200)); do sleep 0.05; w=$((w + 1)); done
    exit "$rc"
}
kill_payload() {
    local p
    p=$(cat "$R10_RUN/payload.pid" 2>/dev/null) || return 0
    [[ -n $p ]] || return 0
    kill -TERM -- "-$p" 2>/dev/null
    local w=0
    while alive "$p" && ((w < 40)); do sleep 0.05; w=$((w + 1)); done
    alive "$p" && kill -KILL -- "-$p" 2>/dev/null
    return 0
}

# lock_fd: the descriptor number of the lock file in this process ("" if none)
lock_fd() {
    local f
    for f in /proc/$$/fd/*; do
        [[ $(readlink -f "$f" 2>/dev/null) == "$(readlink -f "$R10_LOCK")" ]] && { echo "${f##*/}"; return; }
    done
}

# ---- the locked job ---------------------------------------------------------------------------------------------------------
locked_job() {
    R10_WHO=job
    echo $$ >"$R10_RUN/job.pid"
    log "locked job start pid $$ (lock descriptor $(lock_fd))"
    : >"$R10_RUN/locked"
    guard_exit # R10-INNER-GUARD
    if [[ -e $RESTORE_FAILED ]]; then log "RESTORE-FAILED exists; refusing"; echo "NOT CHANGED: $RESTORE_FAILED exists" >&2; exit 2; fi
    if ! dry; then
        cred_open || { log "no credential file"; echo "NOT CHANGED: no credential" >&2; exit 2; }
    elif [[ -n ${R10_TEST_SECRET_FILE:-} ]]; then # self-test only: a stub secret file stands in for the password file
        R10_SECRET_FILE=$R10_TEST_SECRET_FILE cred_open || { log "no stub credential file"; exit 2; }
    fi
    [[ -n ${R10_CRED_FD:-} ]] && log "credential descriptor $R10_CRED_FD"
    local before
    before=$(level_now)
    cp "$R10_SYSFS/$MCLK_F" "$R10_RUN/before.mclk" 2>/dev/null
    listing >"$R10_RUN/before.mclk.norm"
    printf '%s' "$before" >"$R10_RUN/before.level"
    local ms
    ms=$(mask_state)
    if [[ $before != auto || $ms != all ]]; then
        local why="level was '$before', mask '$ms'"
        log "NOT CHANGED: $why"
        echo "NOT CHANGED: $why (someone else's)"
        echo "$why" >"$R10_RUN/not-changed"
        exit 20
    fi
    # the watchdog: own session, inherits the lock descriptor (unless the break removes it)
    local lfd
    lfd=$(lock_fd)
    if dry && [[ ${R10_BREAK_WD_NOFD:-0} == 1 && -n $lfd ]]; then
        setsid bash -c "exec $lfd>&-; exec bash \"\$0\" __watchdog \"\$1\"" "$SELF" $$ </dev/null >>"$R10_RUN/watchdog.out" 2>&1 &
    else
        setsid bash "$SELF" __watchdog $$ </dev/null >>"$R10_RUN/watchdog.out" 2>&1 &
    fi
    WD=$!
    local w=0
    while [[ ! -e $R10_RUN/watchdog.ready ]] && ((w < 200)); do sleep 0.05; w=$((w + 1)); done
    [[ -e $R10_RUN/watchdog.ready ]] || { log "watchdog did not start; nothing written"; echo "NOT CHANGED: no watchdog"; exit 2; }
    trap 'on_signal TERM' TERM
    trap 'on_signal INT' INT
    trap 'on_signal HUP' HUP
    local rc=0
    # pin
    if sysfs_write "$LEVEL_F" manual; then
        : >"$R10_RUN/changed-by-us"
    else
        [[ $(level_now) != auto ]] && : >"$R10_RUN/changed-by-us"
        rc=23
    fi
    if ((rc == 0)); then
        sysfs_write "$MCLK_F" "$R10_LEVEL" || rc=23
    fi
    if ((rc == 0)); then
        : >"$R10_RUN/pinned"
        log "pinned: level manual, mclk index $R10_LEVEL"
        # the payload runs in its own session, without the lock and credential descriptors (only the job and the watchdog hold them)
        local cfd="" extra=""
        cfd=$lfd
        [[ -n $cfd ]] && extra="exec $cfd>&-;"
        [[ -n ${R10_CRED_FD:-} ]] && extra="$extra exec ${R10_CRED_FD}<&-; unset R10_CRED_FD;"
        setsid bash -c "$extra exec \"\$@\"" _ "${PAYLOAD[@]}" >"$R10_RUN/payload.out" 2>&1 &
        local pp=$!
        echo "$pp" >"$R10_RUN/payload.pid"
        wait "$pp"
        rc=$?
        log "payload ended status $rc"
    fi
    finish_restore "$rc"
}
on_signal() {
    log "signal $1 received"
    trap '' INT TERM HUP
    finish_restore 130
}
finish_restore() {
    trap '' INT TERM HUP # a signal now must not re-enter do_restore
    local rc=$1
    kill_payload
    do_restore
    local rr=$?
    : >"$R10_RUN/done"
    local w=0
    while alive "$WD" && ((w < 400)); do sleep 0.05; w=$((w + 1)); done
    log "job exit (payload/pin status $rc, restore status $rr)"
    ((rr != 0)) && exit 21
    [[ -e $R10_RUN/capped ]] && exit 22
    exit "$rc"
}

# ---- entry points -----------------------------------------------------------------------------------------------------------
case ${1:-} in
    __locked)
        shift
        PAYLOAD=("$@")
        locked_job
        ;;
    __watchdog)
        watchdog "$2"
        ;;
esac

DRY=0 LEVEL=1 CAP=${R10_CAP_S:-120} PAYLOAD=()
while (($#)); do
    case $1 in
        --dry-run) DRY=1 ;;
        --level) LEVEL=$2; shift ;;
        --cap-s) CAP=$2; shift ;;
        --) shift; PAYLOAD=("$@"); break ;;
        *) echo "$PROG: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
((${#PAYLOAD[@]})) || { echo "usage: $PROG [--dry-run] [--level N] [--cap-s S] -- <payload command...>" >&2; exit 2; }
[[ $LEVEL =~ ^[0-9]+$ && $CAP =~ ^[0-9]+$ ]] || { echo "$PROG: --level and --cap-s are whole numbers" >&2; exit 2; }
SYSFS=${R10_SYSFS:-}
[[ -n $SYSFS && -d $SYSFS ]] || { echo "$PROG: R10_SYSFS must name the card's device directory. Nothing changed." >&2; exit 2; }
if ((!DRY)) && [[ ${R10_ALLOW_REAL_SYSFS_WRITES:-} != yes-owner-approved ]]; then  # R10-SWITCH-GUARD
    echo "$PROG refuses a real run: set R10_ALLOW_REAL_SYSFS_WRITES=yes-owner-approved (the owner does this by hand; --dry-run needs nothing). Nothing changed." >&2
    exit 2
fi
if [[ $(cat "$SYSFS/device" 2>/dev/null) != 0x7550 ]]; then  # R10-DEVICE-GUARD
    echo "$PROG refuses: $SYSFS/device reads '$(cat "$SYSFS/device" 2>/dev/null)', not 0x7550 (RX 9070 XT). Nothing changed." >&2
    exit 2
fi
[[ ! -e $RESTORE_FAILED ]] || { echo "$PROG refuses: $RESTORE_FAILED exists. Check the card, then remove it. Nothing changed." >&2; exit 2; }
if ((DRY)); then
    [[ -n ${R10_LOCK:-} && $(readlink -f "$R10_LOCK") != "$(readlink -f "$DEFAULT_LOCK")" ]] || { echo "$PROG: a dry run needs R10_LOCK set to a scratch lock, never the real one. Nothing changed." >&2; exit 2; }
    [[ $(readlink -f "$SYSFS") != /sys/* ]] || { echo "$PROG: a dry run never points at /sys. Nothing changed." >&2; exit 2; }
fi
R10_LOCK=${R10_LOCK:-$DEFAULT_LOCK}
R10_RUN=${R10_RUN:-$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r10p0}
mkdir -p "$R10_RUN" || exit 2
export R10_DRY=$DRY R10_LEVEL=$LEVEL R10_CAP_S=$CAP R10_SYSFS=$SYSFS R10_LOCK R10_RUN R10_RESTORE_FAILED=$RESTORE_FAILED
echo "run folder $R10_RUN"
flock -w "${R10_LOCK_WAIT:-600}" -E 6 "$R10_LOCK" bash "$SELF" __locked "${PAYLOAD[@]}"
rc=$?
((rc == 6)) && echo "$PROG: the lock was held for ${R10_LOCK_WAIT:-600} s. Nothing changed."
exit $rc
