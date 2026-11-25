#!/usr/bin/env bash
# Item G1, the part that runs inside the gated job (the shared lock is held for all of it): profile_standard on, 10 runs of
# the 1.0 ms call-then-wait cell, `auto` again at once, the same 10 runs under auto. A trap restores `auto` on EXIT and on
# INT, TERM, HUP and QUIT (not on SIGKILL: stable_clocks.sh restores right after the gate returns, which covers that).
# A restore that fails (the write or the readback) makes this script exit non-zero and say so.
# G1_SECRET_FILE stays set in this script (a retried write may need one more sudo authentication) but is removed from the
# environment of every cell it starts (`env -u`), so the load, the sampler and their children never see it.
# Usage: g1_inner.sh <level file> <run dir> <runs> [--dry-run] [--inject-fail] [--inject-restore-fail]
# The password file is NOT an argument (the gate logs its command line): real runs read it from G1_SECRET_FILE.
# --dry-run: prints the sysfs writes and the commands only (no sudo, no GPU). --inject-fail: fails after the first write.
# --inject-restore-fail: the dry run's write of `auto` fails (the level stays profile_standard).
set -u
here=$(cd "$(dirname "$0")" && pwd)
LEVEL=$1 RUN=$2 runs=$3
shift 3
DRY=0 inject=0
for a in "$@"; do
    case $a in
        --dry-run) DRY=1 ;;
        --inject-fail) inject=1 ;;
        --inject-restore-fail) DRY_FAIL_AUTO=1 ;;
    esac
done
export DRY_FAIL_AUTO=${DRY_FAIL_AUTO:-0}
DRY_LEVEL_FILE=${DRY_LEVEL_FILE:-$(cd "$here/../.." && pwd)/build/round9/tmp-g1-dry/level}
BIN=${R9_BIN:-$(cd "$here/../.." && pwd)/build/round9/bin}
. "$here/g1_lib.sh"
trap_all

cell() { # cell <tag>: one run of the 1.0 ms call-then-wait cell
    local tag=$1
    if ((DRY)); then
        echo "DRYRUN gpu run $tag: env -u G1_SECRET_FILE jobwrap.sh $RUN/loads $tag $BIN/vkr9 clk name=cell,q=g,ms=1.0,cad=16.7,dur=4,calib=0,idle=0.5"
        return 0
    fi
    env -u G1_SECRET_FILE bash "$here/jobwrap.sh" "$RUN/loads" "$tag" "$BIN/vkr9" clk "name=cell,q=g,ms=1.0,cad=16.7,dur=4,calib=0,idle=0.5"
}

echo "LEVEL_BEFORE_INNER $(level_now)"
((DRY)) || sudo_auth || { echo "sudo refused the password; nothing changed"; exit 2; }
set_level profile_standard || { echo "could not set profile_standard"; exit 3; }
echo "LEVEL_IN_FORCE $(level_now)"
[[ $(level_now) == profile_standard ]] || { echo "the level reads $(level_now) after the write, not profile_standard: no GPU run"; exit 3; }
if ((inject)); then
    echo "INJECTED FAILURE after the first write"
    exit 1
fi
for i in $(seq 1 "$runs"); do cell "std-$i" || echo "CELL std-$i returned $? (a cell verdict failed; kept going, the analysis judges it)"; done
set_level auto || exit 3
echo "LEVEL_AFTER_STANDARD_RUNS $(level_now)"
[[ $(level_now) == auto ]] || { echo "the level reads $(level_now) after the write of auto: no GPU run"; exit 3; }
for i in $(seq 1 "$runs"); do cell "auto-$i" || echo "CELL auto-$i returned $? (a cell verdict failed; kept going, the analysis judges it)"; done
exit 0
