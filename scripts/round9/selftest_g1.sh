#!/usr/bin/env bash
# CPU-only self-test of the G1 sudo path (no real sudo, no sysfs write, no GPU job): the dry run of stable_clocks.sh, and the real
# code path of g1_lib.sh / g1_inner.sh against a STUB `sudo` placed first in PATH (it records how it was called and "writes" to a file in
# the scratch folder). Checks:
#   1. the dry run's gate command line does not carry the password file's path (a deliberately broken copy that does carry it must be caught);
#   2. a failed write of `auto` makes the dry run exit non-zero and say the level could not be restored (and the first version's
#      restore_auto, run against the same stub, wrongly claims success: the break);
#   3. every write goes through `sudo -n` with standard input /dev/null (no descriptor of the password file reaches a root child);
#   4. the traps restore `auto` on INT, TERM, HUP and QUIT and the exit status is the signal's (128 + n);
#   5. a second signal during the restore does not abort it (a copy without the ignore line must show the abort);
#   6. a cell started by g1_inner.sh does not get G1_SECRET_FILE in its environment (a copy without `env -u` must leak it);
#   7. stable_clocks.sh refuses a real run without G1_REDESIGNED=1 (Codex round 2 finding 1) and the dry run still works
#      (a copy without the guard must NOT refuse). Every real-run attempt here points R9_REPO at a folder that does not exist, so
#      even a broken guard stops at the first `cd`: no sudo, no sysfs write, no GPU job.
# Usage: scripts/round9/selftest_g1.sh        Exit 0 only if every check passes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
t=$repo/build/round9/tmp-selftest-g1
rm -rf "$t" && mkdir -p "$t/bin" "$t/stub" "$t/run"
rc=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; rc=1; }
SECRET=$t/SECRETMARK-password-file
printf 'pw\n' >"$SECRET"
for f in stable_clocks.sh g1_lib.sh g1_inner.sh common.sh analyse_stable.py analyse_clk.py r9lib.py jobwrap.sh; do ln -s "$here/$f" "$t/bin/$f"; done

# 1. the dry run
export DRY_LEVEL_FILE=$t/dry-level
out=$(G1_SECRET_FILE=$SECRET bash "$here/stable_clocks.sh" --dry-run --runs 1 2>&1)
r=$?
line=$(grep "^DRYRUN gate g1" <<<"$out")
[[ $r == 0 && -n $line ]] && ok "dry run exits 0 and prints the gate command line" || bad "dry run exit $r"
grep -q SECRETMARK <<<"$out" && bad "the password file's path appears in the dry run's output" || ok "the password file's path is not in the dry run's gate command line or anywhere else in its output"
# the break: a copy that passes the secret as an argument must be caught by the same grep
sed 's|gate_cmd=(bash "$here/g1_inner.sh" "$LEVEL" "$RUN" "$runs" "${flags\[@\]}")|gate_cmd=(bash "$here/g1_inner.sh" "$SECRET" "$LEVEL" "$RUN" "$runs" "${flags[@]}")|' "$here/stable_clocks.sh" >"$t/bin/stable_clocks_broken.sh"
out=$(G1_SECRET_FILE=$SECRET bash "$t/bin/stable_clocks_broken.sh" --dry-run --runs 1 2>&1)
grep -q "^DRYRUN gate g1.*SECRETMARK" <<<"$out" && ok "break caught: a copy that passes the path as an argument shows it in the gate command line" || bad "the break (secret as an argument) was not caught"

# 2. the injected failure of the write of auto
out=$(bash "$here/stable_clocks.sh" --dry-run --inject-restore-fail --runs 1 2>&1)
r=$?
[[ $r != 0 ]] && grep -q "could not be restored" <<<"$out" && ok "dry run with a failing write of auto: exit $r and 'the level could not be restored'" || bad "injected restore failure: exit $r"
out=$(bash "$here/stable_clocks.sh" --dry-run --inject-fail --runs 1 2>&1)
r=$?
[[ $r != 0 ]] && grep -q "RESTORE wrote auto; level now: auto" <<<"$out" && ok "dry run with a failure after the first write: the trap restores auto (exit $r, still non-zero)" || bad "injected failure: exit $r"

# the stub sudo
cat >"$t/stub/sudo" <<'STUB'
#!/usr/bin/env bash
# stub: -S -v reads the password from stdin and starts a ticket; `-n bash -c ...` needs the ticket, records its stdin, runs the command.
log=$STUB_DIR/calls.log
args=("$@")
stdin=$(readlink /proc/$$/fd/0)
echo "sudo ${args[*]:0:5} | stdin=$stdin" >>"$log"
has() { local a; for a in "${args[@]}"; do [[ $a == "$1" ]] && return 0; done; return 1; }
if has -S; then
    IFS= read -r pw
    [[ $pw == pw ]] || exit 1
    : >"$STUB_DIR/ticket"
    has -v && exit 0
fi
if has -n && [[ ! -e $STUB_DIR/ticket ]]; then exit 1; fi
[[ -n ${STUB_FAIL_AUTO:-} && ${args[*]} == *" _ auto "* ]] && exit 1
while [[ ${args[0]} != bash ]]; do args=("${args[@]:1}"); done
exec "${args[@]}"
STUB
chmod +x "$t/stub/sudo"
export STUB_DIR=$t/stub PATH=$t/stub:$PATH
[[ $(command -v sudo) == "$t/stub/sudo" ]] || { echo "FAIL the stub sudo is not first in PATH: stopping before any real sudo can run"; exit 2; }
run_inner() { # run_inner <extra env...>: g1_inner.sh against the stub, 0 runs (no GPU job), returns its exit status
    echo profile_standard >"$t/level"
    printf auto >"$t/level"
    : >"$STUB_DIR/calls.log"
    rm -f "$STUB_DIR/ticket"
    env G1_SECRET_FILE="$SECRET" "$@" bash "$here/g1_inner.sh" "$t/level" "$t/run" 0 >"$t/inner.out" 2>&1
}
run_inner
r=$?
[[ $r == 0 && $(cat "$t/level") == auto ]] && ok "real path against the stub: exit 0, level back to auto" || bad "stub run: exit $r, level $(cat "$t/level")"
grep -q "LEVEL_IN_FORCE profile_standard" "$t/inner.out" && ok "the level read back after the write was profile_standard before any run" || bad "no readback of profile_standard"
if grep -q "^sudo -n" "$STUB_DIR/calls.log" && ! grep "^sudo -n" "$STUB_DIR/calls.log" | grep -qv "stdin=/dev/null"; then ok "every write is sudo -n with standard input /dev/null"; else bad "a write did not have /dev/null as standard input: $(grep '^sudo -n' "$STUB_DIR/calls.log" | head -2)"; fi
# (the stub's own record of `sudo -S -v` shows its standard input, which is the file: that is the one authentication, no command runs)
if grep "^sudo -n" "$STUB_DIR/calls.log" | grep -q SECRETMARK || grep -q SECRETMARK "$t/inner.out"; then bad "the password file's path appears in a write's call record or in the script's output"; else ok "the password file's path is in no write's call record and not in the script's output"; fi
[[ $(grep -c "^sudo -S" "$STUB_DIR/calls.log") == 1 ]] && ok "the password file is read once (one sudo -S -v), before the first write" || bad "sudo -S was called $(grep -c '^sudo -S' "$STUB_DIR/calls.log") times"
run_inner STUB_FAIL_AUTO=1
r=$?
[[ $r != 0 ]] && grep -q "could not be restored" "$t/inner.out" && [[ $(cat "$t/level") == profile_standard ]] && ok "stub whose write of auto fails: exit $r, 'could not be restored', level left at profile_standard" || bad "stub failing auto: exit $r"
# the first version's restore against the same stub: it prints success although the write failed
git -C "$repo" show 11ef028:scripts/round9/g1_lib.sh >"$t/old_g1_lib.sh" 2>/dev/null
if [[ -s $t/old_g1_lib.sh ]]; then
    echo profile_standard >"$t/level"
    out=$(STUB_FAIL_AUTO=1 SECRET=$SECRET LEVEL=$t/level DRY=0 bash -c ". '$t/old_g1_lib.sh'; restore_auto; echo \"old restore_auto returned \$?\"" 2>&1)
    grep -q "RESTORE wrote auto" <<<"$out" && grep -q "returned 0" <<<"$out" && ok "break shown: the first version's restore_auto claims 'wrote auto' and returns 0 while the level still reads $(cat "$t/level")" || bad "could not show the old restore_auto (output: $out)"
fi

# 4. signals: the traps restore auto and the exit status is 128 + n
for sig in INT TERM HUP QUIT; do
    echo profile_standard >"$t/level"
    : >"$STUB_DIR/calls.log"
    : >"$STUB_DIR/ticket"
    (
        export G1_SECRET_FILE=$SECRET LEVEL=$t/level DRY=0
        . "$here/g1_lib.sh"
        trap_all
        sleep 30 &
        echo $! >"$t/sleep.pid"
        wait
    ) >"$t/sig.out" 2>&1 &
    pid=$!
    for _ in $(seq 1 50); do [[ -s $t/sleep.pid ]] && break; sleep 0.1; done
    sleep 0.3
    kill -"$sig" "$pid"
    wait "$pid"
    r=$?
    kill "$(cat "$t/sleep.pid")" 2>/dev/null
    rm -f "$t/sleep.pid"
    want=$((128 + $(kill -l "$sig")))
    [[ $(cat "$t/level") == auto && $r == "$want" ]] && ok "SIG$sig: level restored to auto, exit $r" || bad "SIG$sig: level $(cat "$t/level"), exit $r (wanted $want)"
done

# 5. a second signal during the restore: a slow write of auto (the real restore_auto against a slow set_level) must still finish
double_signal() { # double_signal <lib file>: prints the level after two TERMs, the second sent while the write of auto is under way
    echo profile_standard >"$t/level"
    rm -f "$t/sleep.pid"
    (
        export G1_SECRET_FILE=$SECRET LEVEL=$t/level DRY=0
        . "$1"
        set_level() { sleep 1; printf auto >"$LEVEL"; return 0; }
        trap_all
        sleep 30 &
        echo $! >"$t/sleep.pid"
        wait
    ) >"$t/dbl.out" 2>&1 &
    local pid=$!
    for _ in $(seq 1 50); do [[ -s $t/sleep.pid ]] && break; sleep 0.1; done
    sleep 0.3
    kill -TERM "$pid"
    sleep 0.4
    kill -TERM "$pid" 2>/dev/null
    wait "$pid"
    DBL_RC=$?
    kill "$(cat "$t/sleep.pid")" 2>/dev/null
    DBL_LEVEL=$(cat "$t/level")
}
double_signal "$here/g1_lib.sh"
[[ $DBL_LEVEL == auto && $DBL_RC == 143 ]] && ok "a second TERM during the restore does not abort it: level auto, exit $DBL_RC" || bad "second TERM: level $DBL_LEVEL, exit $DBL_RC"
sed '/# ignore signals while restoring/d' "$here/g1_lib.sh" >"$t/bin/g1_lib_noignore.sh"
grep -q "trap '' INT" "$t/bin/g1_lib_noignore.sh" && bad "the break copy still has the ignore line"
double_signal "$t/bin/g1_lib_noignore.sh"
[[ $DBL_LEVEL != auto ]] && ok "break caught: without the ignore line the second TERM aborts the restore (level left at $DBL_LEVEL)" || bad "the break (no ignore line) did not abort the restore"

# 6. the environment of a cell: stub BIN with a sampler that only says ready and a load that dumps its environment
mkdir -p "$t/bin2"
cat >"$t/bin2/r9_sampler" <<'STUB'
#!/usr/bin/env bash
mkdir -p "$1" && : >"$1/ready"
exec sleep 30
STUB
cat >"$t/bin2/vkr9" <<'STUB'
#!/usr/bin/env bash
env >"$VKG_OUT/envdump.$VKG_TAG"
STUB
chmod +x "$t/bin2/r9_sampler" "$t/bin2/vkr9"
cell_env() { # cell_env <g1_inner copy>: runs one standard cell and one auto cell against the stubs, prints nothing, leaves envdump files
    rm -rf "$t/run6" && mkdir -p "$t/run6"
    printf auto >"$t/level"
    : >"$STUB_DIR/calls.log"
    rm -f "$STUB_DIR/ticket"
    env G1_SECRET_FILE="$SECRET" R9_BIN="$t/bin2" bash "$1" "$t/level" "$t/run6" 1 >"$t/cell.out" 2>&1
}
cell_env "$here/g1_inner.sh"
if [[ -s $t/run6/loads/envdump.std-1 && -s $t/run6/loads/envdump.auto-1 ]] && ! grep -q "G1_SECRET_FILE\|SECRETMARK" "$t/run6/loads/envdump.std-1" "$t/run6/loads/envdump.auto-1"; then
    ok "the load of a cell has no G1_SECRET_FILE and no password path in its environment (2 cells)"
else
    bad "a cell saw the secret or did not run: $(ls "$t/run6/loads" 2>&1 | head -3)"
fi
sed 's|env -u G1_SECRET_FILE bash|bash|' "$here/g1_inner.sh" >"$t/bin/g1_inner_leaky.sh"
cell_env "$t/bin/g1_inner_leaky.sh"
grep -q "G1_SECRET_FILE" "$t/run6/loads/envdump.std-1" 2>/dev/null && ok "break caught: without env -u the cell's environment holds G1_SECRET_FILE" || bad "the break (no env -u) was not caught"

# 7. the refusal
export R9_REPO=$t/no-such-repo
out=$(G1_SECRET_FILE=$SECRET bash "$here/stable_clocks.sh" 2>&1)
r=$?
[[ $r == 2 ]] && grep -q "Codex round 2 finding 1" <<<"$out" && grep -q "G1_REDESIGNED=1" <<<"$out" && ok "without G1_REDESIGNED the script refuses (exit $r) and names Codex round 2 finding 1" || bad "no refusal: exit $r, $out"
out=$(G1_REDESIGNED=0 G1_SECRET_FILE=$SECRET bash "$here/stable_clocks.sh" 2>&1)
[[ $? == 2 ]] && grep -q "Codex round 2 finding 1" <<<"$out" && ok "G1_REDESIGNED=0 refuses as well" || bad "G1_REDESIGNED=0 did not refuse"
out=$(G1_REDESIGNED=1 G1_SECRET_FILE=$SECRET bash "$here/stable_clocks.sh" 2>&1)
grep -q "Codex round 2 finding 1" <<<"$out" && bad "G1_REDESIGNED=1 still printed the refusal" || ok "with G1_REDESIGNED=1 the refusal is not printed (this run then stops at the missing R9_REPO, nothing else runs)"
sed 's|^if ((!DRY)) \&\& \[\[ ${G1_REDESIGNED:-} != 1 \]\]; then  # G1-GUARD|if false; then|' "$here/stable_clocks.sh" >"$t/bin/stable_clocks_unguarded.sh"
grep -q "^if false; then" "$t/bin/stable_clocks_unguarded.sh" || bad "the break copy still has the guard"
out=$(G1_SECRET_FILE=$SECRET bash "$t/bin/stable_clocks_unguarded.sh" 2>&1)
grep -q "Codex round 2 finding 1" <<<"$out" && bad "the break (no guard) still refused" || ok "break caught: a copy without the guard does not refuse (it stops at the missing R9_REPO)"
unset R9_REPO
out=$(G1_SECRET_FILE=$SECRET bash "$here/stable_clocks.sh" --dry-run --runs 1 2>&1)
[[ $? == 0 ]] && ! grep -q "refuses" <<<"$out" && ok "the dry run is not refused (exit 0)" || bad "the dry run was refused or failed"
exit $rc
