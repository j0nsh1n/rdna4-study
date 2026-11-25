#!/usr/bin/env bash
# Item C driver: does the data one dispatch wrote sit in L2 for the next dispatch, per Vulkan barrier kind (compute and
# graphics queue) and through HIP? Three process repeats of every cell, each as one gated job (shared lock, game check,
# profile and card-configuration checks, 10 GB cap, 95 C guard). The analysis's own self-test runs first (no GPU).
# GPU time: about 0.5 minute of load; the gate overhead dominates the wall time.
# Usage: scripts/round9/run_C.sh        Environment: R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh)
set -u
. "$(dirname "$0")/common.sh"
python3 -I "$here/analyse_cbar.py" --selftest || { echo "analysis self-test failed; no GPU job started"; exit 2; }
grep -aq "warm_ms" "$BIN/vkr9" 2>/dev/null || { echo "vkr9 in $BIN is the first version (no warm-up, stamps or clock check): run scripts/round9/build.sh"; exit 2; }
start_run C
cells=()
for q in g c; do
    for k in none exec b2 legacy vv7 full event; do cells+=("name=n-$q-$k,q=$q,kind=$k,order=normal,reps=20"); done
    cells+=("name=cold-$q,q=$q,kind=b2,order=cold,reps=20" "name=fused-$q,q=$q,kind=b2,order=fused,reps=20"
        "name=race-$q,q=$q,kind=b2,order=race,reps=20")
done
for r in 1 2 3; do
    gate "c-vk-r$r" 60 120 bash "$here/jobwrap.sh" "$RUN/loads" "c-vk-r$r" "$BIN/vkr9" cbar "${cells[@]}"
    gate "c-hip-r$r" 60 120 bash "$here/jobwrap.sh" "$RUN/loads" "c-hip-r$r" "$BIN/r9_cl2" 20 1024
done
# 3 process repeats of 20 Vulkan cells (2 queues x 10) and 4 HIP cells, 20 repetitions each: anything missing fails the analysis
finish python3 -I "$here/analyse_cbar.py" --runs 3 --reps 20 --cells 20,4 "$RUN"/r9-c-*.log
