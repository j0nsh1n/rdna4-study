#!/usr/bin/env bash
# Item C2 driver: item C again with the walk in page order (increasing, next = slot + 1) instead of a random permutation:
# the cold, fused, two-launch and b2 cells (and the race break) on the compute and graphics queue and through HIP. Three process
# repeats of every cell, each as one gated job (shared lock, game check, profile and card-configuration checks, 10 GB cap, 95 C
# guard). The analysis's own self-test (with the page-order breaks) runs first (no GPU).
# GPU time: about 0.5 minute of load (6 jobs of ~5 s); wall time about 5 minutes (gate overhead and 10 s pauses).
# Usage: scripts/round9/run_C2.sh        Environment: R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh)
set -u
. "$(dirname "$0")/common.sh"
python3 -I "$here/analyse_cbar.py" --selftest || { echo "analysis self-test failed; no GPU job started"; exit 2; }
grep -aq "warm_ms" "$BIN/vkr9" 2>/dev/null || { echo "vkr9 in $BIN is the first version (no warm-up, stamps or clock check): run scripts/round9/build.sh"; exit 2; }
start_run C2
cells=()
for q in g c; do
    cells+=("name=n-$q-b2,q=$q,kind=b2,order=normal,walk=page,reps=20" "name=cold-$q,q=$q,kind=b2,order=cold,walk=page,reps=20"
        "name=fused-$q,q=$q,kind=b2,order=fused,walk=page,reps=20" "name=race-$q,q=$q,kind=b2,order=race,walk=page,reps=20")
done
for r in 1 2 3; do
    gate "c2-vk-r$r" 60 120 bash "$here/jobwrap.sh" "$RUN/loads" "c2-vk-r$r" "$BIN/vkr9" cbar "${cells[@]}"
    gate "c2-hip-r$r" 60 120 bash "$here/jobwrap.sh" "$RUN/loads" "c2-hip-r$r" "$BIN/r9_cl2" 20 1024 page
done
finish python3 -I "$here/analyse_cbar.py" --page --runs 3 --reps 20 --cells 8,4 "$RUN"/r9-c2-*.log
