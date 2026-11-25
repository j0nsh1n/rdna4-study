#!/usr/bin/env bash
# Round 9 item A, load-ratio variant: every cell of ld_mat_k through the study gate (lock, LACT profile check, 8 GB floor,
# memory cap, nice, thermal guard at 95 C, hotspot start limit). The LEAD runs this; a worker never does.
# Usage: scripts/round9/run_ld_mat_k.sh [run folder, relative to the repo root; default runs/<UTC>-r9-ldmatk]
# Needs: scripts/round9/build_round9.sh ld_mat_k done (listing checks passed). Jobs, in this order:
#   r9-ldmk-r1, r9-ldmk-r2   one wave (209 registers): matrix only, then 12 cells (G 1, 4, 8; variants s, w; two lookaheads and
#                            D = 0) at L0 and L2, 3 bursts of ~20 ms per cell; two process repeats
#   r9-ldmk-break            the deliberate break: k8s_d0 labelled L2, loads aimed at L0
# GPU time: about 50 bursts of ~20 ms per job: about 3 s of real load per job, 10 s in all; wall time about 3 minutes (the gate
# and the cool-down checks dominate). Stops at the first failed job.
# Gate arguments as in run_ld_mat.sh: study-llm.sh r4dev <job> <hotspot start limit C> <timeout s> <cmd>.
set -u
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$H/../.." && pwd)
cd "$ROOT" || exit 2
RUN=${1:-runs/$(date -u +%Y%m%dT%H%M%SZ)-r9-ldmatk}
export STUDY_GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
export LLM_EXPECT_PROFILE=${LLM_EXPECT_PROFILE:-Default}
export LLM_RUN=$RUN
BIN=build/round9/bin/ld_mat_k
CELLS=k1s_d24,k1s_d54,k4s_d0,k4s_d12,k4s_d24,k4w_d12,k4w_d24,k8s_d0,k8s_d6,k8s_d12,k8w_d6,k8w_d12
mkdir -p "$RUN"

python3 -I "$H/ana_ld_mat_k.py" --selftest | tee "$RUN/selftest-ana.txt"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "analysis self-test failed: no GPU job"; exit 2; }
[[ -x $BIN ]] || { echo "no $BIN: run scripts/round9/build_round9.sh ld_mat_k first"; exit 2; }
(sha256sum -c --quiet build/round9/ext/ld_mat_k/sha256.txt 2>&1 | grep -q 'bin/ld_mat_k: FAILED') && { echo "binary differs from the one listed: rebuild"; exit 2; }
{ echo "<!-- copied $(date -u +%FT%TZ) before the first GPU job; git $(git rev-parse --short HEAD 2>/dev/null) -->"; cat "$H/expect-ld_mat_k.md"; } >"$RUN/expect-ld_mat_k.md"
cp build/round9/check-ld_mat_k.txt "$RUN/" 2>/dev/null

job() {  # job <name> <hotspot start limit> <timeout> <args...>
    local name=$1 hot=$2 tmo=$3
    shift 3
    echo "$(date -u +%T) start $name"
    scripts/study-llm.sh r4dev "$name" "$hot" "$tmo" "$BIN" "$@" >"$RUN/gate-$name.out" 2>&1
    local r=$?
    echo "$(date -u +%T) end $name exit $r"
    return $r
}

echo "STARTS $(date -u +%FT%TZ) ld_mat_k, GPU lock $STUDY_GPU_LOCK"
rc=0
for r in 1 2; do
    [[ $rc -eq 0 ]] && { job "r9-ldmk-r$r" 60 300 --levels L0,L2 --cells "ref,$CELLS" --bursts 3 --ms 20 || rc=$?; }
done
[[ $rc -eq 0 ]] && { job r9-ldmk-break 60 120 --levels L2 --cells k8s_d0 --aim L0 --bursts 3 --ms 20 || rc=$?; }
echo "ENDS $(date -u +%FT%TZ) ld_mat_k exit $rc"
[[ $rc -eq 0 ]] || exit $rc
python3 -I "$H/ana_ld_mat_k.py" "$RUN"/r9-ldmk-*.log | tee "$RUN/ana-ld_mat_k.txt"
exit "${PIPESTATUS[0]}"
