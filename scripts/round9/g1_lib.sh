# shellcheck shell=bash
# Sourced by stable_clocks.sh and g1_inner.sh: the one place that writes to sysfs. Needs: LEVEL (the sysfs file), DRY (0/1).
# Real runs also need G1_SECRET_FILE in the environment (the password file; it is never an argument of any command, so no
# command line, process list or gate log shows it, and no process other than sudo's own authentication reads it).
# Dry runs keep the simulated level in a file ($DRY_LEVEL_FILE) so that the outer script and the gated script see the same level.
#
# sudo is used in two steps: `sudo -S -p '' -v` reads the password file once to authenticate (it runs no command, so no child gets
# the file as standard input), then every write is `sudo -n ... </dev/null` (no password prompt, no inherited descriptor).
sudo_auth() {
    [[ -n ${G1_SECRET_FILE:-} && -r $G1_SECRET_FILE ]] || return 1
    sudo -S -p '' -v <"$G1_SECRET_FILE" 2>/dev/null
}

level_now() {
    if [[ ${DRY:-0} == 1 ]]; then cat "$DRY_LEVEL_FILE" 2>/dev/null || echo auto; else cat "$LEVEL"; fi
}

# set_level <value>: 0 only if the write itself succeeded
set_level() {
    if [[ ${DRY:-0} == 1 ]]; then
        if [[ $1 == auto && ${DRY_FAIL_AUTO:-0} == 1 ]]; then
            echo "DRYRUN write 'auto' to $LEVEL: FAILED (injected, the level is left unchanged)"
            return 1
        fi
        echo "DRYRUN write '$1' to $LEVEL"
        printf '%s' "$1" >"$DRY_LEVEL_FILE"
        return 0
    fi
    local try
    for try in 1 2; do
        sudo -n bash -c 'printf "%s" "$1" > "$2"' _ "$1" "$LEVEL" </dev/null 2>/dev/null && return 0
        ((try == 1)) && sudo_auth  # the ticket may not reach this process (other session or parent): authenticate once more, retry once
    done
    return 1
}

# restore_auto: writes `auto` unless the level already reads auto, then reads the level back. Returns 0 only if it reads auto.
restore_auto() {
    local now rc
    now=$(level_now)
    if [[ $now == auto ]]; then
        echo "RESTORE nothing to do; level reads: auto"
        return 0
    fi
    set_level auto
    rc=$?
    now=$(level_now)
    if ((rc == 0)) && [[ $now == auto ]]; then
        echo "RESTORE wrote auto; level now: auto"
        return 0
    fi
    echo "RESTORE FAILED: the level could not be restored to auto (write status $rc, level now: $now). Restore it by hand: echo auto | sudo tee $LEVEL"
    return 1
}

# on_exit_restore: the EXIT trap of both G1 scripts: restores, and a failed restore makes the script's exit status non-zero. Signals
# are ignored while it restores (a second INT or TERM used to run the signal trap, which exits, in the middle of restore_auto).
on_exit_restore() {
    local rc=$?
    trap - EXIT
    trap '' INT TERM HUP QUIT  # ignore signals while restoring: a second one must not abort the write of auto half way
    restore_auto || { ((rc == 0)) && rc=4; }
    exit "$rc"
}

# trap_all: restore on EXIT and on every signal that would end the script (the signal handlers exit, which runs the EXIT trap).
trap_all() {
    trap on_exit_restore EXIT
    local s
    for s in INT TERM HUP QUIT; do
        trap "echo 'signal $s received'; exit \$((128 + \$(kill -l $s)))" "$s"
    done
}
