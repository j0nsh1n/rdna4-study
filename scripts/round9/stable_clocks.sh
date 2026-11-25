#!/usr/bin/env bash
# Item G1 (owner-approved): is `power_dpm_force_performance_level = profile_standard` steady in the in-kernel core clock?
# One gated job (g1_inner.sh; the shared lock is held for the whole sequence, so no other job sees the changed level):
#   read the level (before); set profile_standard; 10 runs of the 1.0 ms call-then-wait cell of item D (Vulkan graphics queue,
#   4 s each); set `auto` at once; the same 10 runs under auto; read the level (after).
# `auto` is restored on every exit that a shell can see (success, failure, INT, TERM, HUP, QUIT; the traps are here and in
# g1_inner.sh), and again right after the gate returns, before any analysis, which also covers a job that was killed with
# SIGKILL. A restore counts only if the write succeeded and the level reads `auto` afterwards; if not, this script prints
# "the level could not be restored", leaves a non-zero exit status and says how to restore by hand.
# The only sysfs writes are to power_dpm_force_performance_level, through sudo (g1_lib.sh): one authentication from the password
# file (`sudo -S -v`), then `sudo -n </dev/null` for every write. The password file's PATH is never an argument of a command the
# gate logs, and the file is never printed, copied or logged: it reaches the scripts through the environment variable
# G1_SECRET_FILE (set by this script from its argument or from the caller's environment).
# Usage: G1_SECRET_FILE=<password file> scripts/round9/stable_clocks.sh [--runs N]
#        scripts/round9/stable_clocks.sh <password file> [--runs N]          (the same; the path is then in this script's own argv)
#        scripts/round9/stable_clocks.sh --dry-run [--inject-fail] [--inject-restore-fail] [--runs N]
#          no sudo, no write, no GPU job: prints each simulated sysfs write, the exact command line the gate would log, and each
#          command it would run. --inject-fail fails after the first write (the trap must restore); --inject-restore-fail makes the
#          write of `auto` fail (the script must say the level could not be restored and exit non-zero).
# REFUSES a real run unless G1_REDESIGNED=1 is set (Codex round 2 finding 1): the recovery write after a killed job runs outside the
# GPU lock, so a second gated job could start under profile_standard or have its own level changed by the late write. Do not set
# G1_REDESIGNED=1 until the recovery runs under the same lock as the mutation. The dry run is not affected.
# GPU time: 2 x 10 x 4.5 s = 1.5 minutes. Environment: R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh).
set -u
here=$(cd "$(dirname "$0")" && pwd)
DRY=0 inject=0 runs=10 DRY_FAIL_AUTO=0 flags=()
SECRET=${G1_SECRET_FILE:-}
while (($#)); do
    case $1 in
        --dry-run) DRY=1 ;;
        --inject-fail) inject=1 ;;
        --inject-restore-fail) DRY_FAIL_AUTO=1 ;;
        --runs) runs=$2; shift ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) SECRET=$1 ;;
    esac
    shift
done
export DRY_FAIL_AUTO
if ((!DRY)) && [[ ${G1_REDESIGNED:-} != 1 ]]; then  # G1-GUARD
    echo "stable_clocks.sh refuses to run: Codex round 2 finding 1 (the recovery write after a killed job runs outside the GPU lock, so another gated job can start under profile_standard or have its own level changed). It runs again only after the recovery runs under the same lock, with G1_REDESIGNED=1. Nothing was changed. --dry-run still works." >&2
    exit 2
fi
if ((DRY)); then
    LEVEL=${LEVEL:-/sys/class/drm/cardX/device/power_dpm_force_performance_level}
    RUN=${R9_RUN:-/nonexistent-dry-run}
    DRY_LEVEL_FILE=${DRY_LEVEL_FILE:-$(cd "$here/../.." && pwd)/build/round9/tmp-g1-dry/level}
    mkdir -p "$(dirname "$DRY_LEVEL_FILE")" && printf 'auto' >"$DRY_LEVEL_FILE"
    export DRY_LEVEL_FILE
    . "$here/g1_lib.sh"
    trap_all
    flags=(--dry-run)
    ((inject)) && flags+=(--inject-fail)
    ((DRY_FAIL_AUTO)) && flags+=(--inject-restore-fail)
    echo "LEVEL_BEFORE $(level_now)"
    gate_cmd=(bash "$here/g1_inner.sh" "$LEVEL" "$RUN" "$runs" "${flags[@]}")
    echo "DRYRUN gate g1: scripts/study-llm.sh r4dev r9-g1 60 400 ${gate_cmd[*]}"
    "${gate_cmd[@]}"
    rc=$?
    echo "INNER exit $rc"
    echo "LEVEL_AFTER_INNER $(level_now)"
    restore_auto || { echo "DRYRUN: restore after the gate failed"; ((rc == 0)) && rc=4; }
    exit $rc
fi
[[ -n $SECRET && -r $SECRET ]] || { echo "usage: G1_SECRET_FILE=<password file> stable_clocks.sh [--runs N] | --dry-run [--inject-fail] [--inject-restore-fail]" >&2; exit 2; }
export G1_SECRET_FILE=$SECRET
. "$here/common.sh"
python3 -I "$here/analyse_stable.py" --selftest || { echo "analysis self-test failed; nothing changed"; exit 2; }
start_run G1
LEVEL=$DRM_DEVICE/power_dpm_force_performance_level
. "$here/g1_lib.sh"
trap_all
before=$(level_now)
echo "LEVEL_BEFORE $before" | tee "$RUN/levels.txt"
sudo_auth || { echo "sudo refused the password; nothing changed"; exit 2; }
gate_cmd=(bash "$here/g1_inner.sh" "$LEVEL" "$RUN" "$runs")
gate g1 60 400 "${gate_cmd[@]}"
rc=$?
restore_auto | tee -a "$RUN/levels.txt"  # at once, before the analysis: a job killed with SIGKILL cannot run its own trap
rr=${PIPESTATUS[0]}
after=$(level_now)
echo "LEVEL_AFTER_JOB $after" | tee -a "$RUN/levels.txt"
args_std=() args_auto=()
for i in $(seq 1 "$runs"); do
    args_std+=("$RUN/loads/std-$i-clk-cell.csv")
    args_auto+=("$RUN/loads/auto-$i-clk-cell.csv")
done
in_force=$(grep -h "^LEVEL_IN_FORCE" "$RUN/g1.stdout" "$RUN/r9-g1.log" 2>/dev/null | tail -1 | awk "{print \$2}")
python3 -I "$here/analyse_stable.py" --before "$before" --after "$after" --in-force "$in_force" --standard "${args_std[@]}" --auto "${args_auto[@]}" |
    tee "$RUN/analysis.txt"
ra=${PIPESTATUS[0]}
((rr != 0)) && echo "G1: the level could not be restored to auto (see RESTORE FAILED above): exit non-zero"
((rc != 0 || rr != 0 || ra != 0)) && exit $((rc != 0 ? rc : (rr != 0 ? 4 : ra)))
exit 0
