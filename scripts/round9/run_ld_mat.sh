#!/usr/bin/env bash
# Round 9 item A: every cell of ld_mat through the study gate (lock, LACT profile check, 8 GB floor, memory cap, nice,
# thermal guard at 95 C, hotspot start limit). The LEAD runs this; a worker never does.
# Usage: scripts/round9/run_ld_mat.sh [run folder, relative to the repo root; default runs/<UTC>-r9-ldmat]
#   env HIT=<cycles>   the pure L0 hit cost of the ld_order run of the same session, handed to the analysis (control b)
#   env ONLY=<one|card|R>   run only the one-wave jobs, only the whole-card jobs, or only the whole-card jobs of one register count
# Needs: scripts/round9/build_round9.sh ld_mat done. Jobs, in this order:
#   r9-ldm-one       one wave (R 209), 4 levels x (ref, D 0 1 2 4 8 all), 3 launches of ~20 ms each, no cool-down needed
#   r9-ldm-break     the deliberate break: L2 label, loads aimed at L0, D = 0 (control d)
#   r9-ldm-R<R>-<L>  whole card, register count R in 51 90 167 209, level L in L0 L2 MALL GDDR: (ref first, in the L0 job)
#                    D 0 1 2 4 8 all, 3 bursts of ~50 ms each, every recorded burst started below 58 C (the program idles until it is; the
#                    calibration launches before the bursts have no such check, the gate's start limit is 60 C and its stop 95 C)
# GPU load: about 350 bursts of at most 0.1 s plus calibration launches: about 1.5 minutes of real load; wall time about
# 40 minutes (the hotspot recovery between bursts dominates). Stops at the first failed job.
# Gate arguments, as in runs/20261124T193633Z-r8fsr/grid.sh: study-llm.sh r4dev <job> <hotspot start limit C> <timeout s> <cmd>.
set -u
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$H/../.." && pwd)
cd "$ROOT" || exit 2
RUN=${1:-runs/$(date -u +%Y%m%dT%H%M%SZ)-r9-ldmat}
export STUDY_GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
export LLM_EXPECT_PROFILE=${LLM_EXPECT_PROFILE:-Default}
export LLM_RUN=$RUN
BIN=build/round9/bin/ld_mat
ONLY=${ONLY:-all}
mkdir -p "$RUN"

python3 -I "$H/ana_ld_mat.py" --selftest || { echo "analysis self-test failed: no GPU job"; exit 2; }
[[ -x $BIN ]] || { echo "no $BIN: run scripts/round9/build_round9.sh ld_mat first"; exit 2; }
(cd "$ROOT" && sha256sum -c --quiet build/round9/ext/ld_mat/sha256.txt 2>&1 | grep -q 'bin/ld_mat: FAILED') && { echo "binary differs from the one listed: rebuild"; exit 2; }
{ echo "<!-- copied $(date -u +%FT%TZ) before the first GPU job; git $(git rev-parse --short HEAD 2>/dev/null) -->"; cat "$H/expect-ld_mat.md"; } >"$RUN/expect-ld_mat.md"
cp build/round9/check-ld_mat.txt "$RUN/" 2>/dev/null

job() {  # job <name> <hotspot start limit> <timeout> <args...>
    local name=$1 hot=$2 tmo=$3
    shift 3
    echo "$(date -u +%T) start $name"
    scripts/study-llm.sh r4dev "$name" "$hot" "$tmo" "$BIN" "$@" >"$RUN/gate-$name.out" 2>&1
    local r=$?
    echo "$(date -u +%T) end $name exit $r"
    return $r
}

echo "STARTS $(date -u +%FT%TZ) ld_mat (ONLY=$ONLY), GPU lock $STUDY_GPU_LOCK"
rc=0
if [[ $ONLY == all || $ONLY == one ]]; then
    job r9-ldm-one 60 300 one --r 209 --levels L0,L2,MALL,GDDR --ds ref,0,1,2,4,8,all --bursts 3 --ms 20 || rc=$?
    [[ $rc -eq 0 ]] && { job r9-ldm-break 60 120 one --r 209 --levels L2 --ds 0 --aim L0 --bursts 3 --ms 20 || rc=$?; }
fi
if [[ $rc -eq 0 && ( $ONLY == all || $ONLY == card || $ONLY =~ ^[0-9]+$ ) ]]; then
    for R in 51 90 167 209; do
        [[ $ONLY =~ ^[0-9]+$ && $ONLY != "$R" ]] && continue
        first=1
        for L in L0 L2 MALL GDDR; do
            ds=0,1,2,4,8,all
            [[ $first -eq 1 ]] && ds=ref,$ds
            first=0
            job "r9-ldm-R$R-$L" 60 900 card --r "$R" --levels "$L" --ds "$ds" --bursts 3 --ms 50 || { rc=$?; break 2; }
            sleep 5
        done
    done
fi
echo "ENDS $(date -u +%FT%TZ) ld_mat exit $rc"
[[ $rc -eq 0 ]] || exit $rc
hit=()
[[ -n ${HIT:-} ]] && hit=(--hit "$HIT")
python3 -I "$H/ana_ld_mat.py" "${hit[@]}" "$RUN"/r9-ldm-*.log | tee "$RUN/ana-ld_mat.txt"
exit "${PIPESTATUS[0]}"
