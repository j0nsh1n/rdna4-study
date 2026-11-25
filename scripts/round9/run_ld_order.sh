#!/usr/bin/env bash
# Round 9 item B: every cell of ld_order through the study gate (lock, LACT profile check, 8 GB floor, memory cap, nice,
# thermal guard at 95 C, hotspot start limit). The LEAD runs this; a worker never does.
# Usage: scripts/round9/run_ld_order.sh [run folder, relative to the repo root; default runs/<UTC>-r9-ldorder]
# Needs: scripts/round9/build_round9.sh ld_order done (the binary and its listing check). Two jobs of a few seconds of
# GPU time each (29 patterns x 3 launches of one wave, a 512 MiB flush before each launch); about 2 minutes of wall time.
# Gate arguments, as in runs/20261124T193633Z-r8fsr/grid.sh: study-llm.sh r4dev <job> <hotspot start limit C> <timeout s> <cmd>.
set -u
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$H/../.." && pwd)
cd "$ROOT" || exit 2
RUN=${1:-runs/$(date -u +%Y%m%dT%H%M%SZ)-r9-ldorder}
export STUDY_GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
export LLM_EXPECT_PROFILE=${LLM_EXPECT_PROFILE:-Default}
export LLM_RUN=$RUN
BIN=build/round9/bin/ld_order
mkdir -p "$RUN"

# 1. the analysis must pass its own deliberate breaks before any job
python3 -I "$H/ana_ld_order.py" --selftest || { echo "analysis self-test failed: no GPU job"; exit 2; }
# 2. the binary is the one the listing check looked at
[[ -x $BIN ]] || { echo "no $BIN: run scripts/round9/build_round9.sh ld_order first"; exit 2; }
(cd "$ROOT" && sha256sum -c --quiet build/round9/ext/ld_order/sha256.txt 2>&1 | grep -q 'bin/ld_order: FAILED') && { echo "binary differs from the one listed: rebuild"; exit 2; }
# 3. the expectations are written into the run folder before the first job
{ echo "<!-- copied $(date -u +%FT%TZ) before the first GPU job; git $(git rev-parse --short HEAD 2>/dev/null) -->"; cat "$H/expect-ld_order.md"; } >"$RUN/expect-ld_order.md"
cp build/round9/check-ld_order.txt "$RUN/" 2>/dev/null

echo "STARTS $(date -u +%FT%TZ) ld_order, GPU lock $STUDY_GPU_LOCK"
rc=0
for n in 1 2; do
    scripts/study-llm.sh r4dev "r9-ldo-$n" 60 120 "$BIN" run >"$RUN/gate-r9-ldo-$n.out" 2>&1
    r=$?
    echo "$(date -u +%FT%TZ) r9-ldo-$n exit $r"
    if [[ $r -ne 0 ]]; then rc=$r; break; fi
    sleep 5
done
echo "ENDS $(date -u +%FT%TZ) ld_order exit $rc"
[[ $rc -eq 0 ]] || exit $rc
python3 -I "$H/ana_ld_order.py" "$RUN"/r9-ldo-1.log "$RUN"/r9-ldo-2.log | tee "$RUN/ana-ld_order.txt"
exit "${PIPESTATUS[0]}"
