#!/usr/bin/env bash
# Item D driver: call, then wait, at a 16.7 ms cadence for 20 s; dispatches of 0.3, 0.6, 1.0 and 1.7 ms; Vulkan (graphics
# queue) and HIP; the 1.0 ms Vulkan cell repeated 3 more times (spread between runs); a sysfs sampler beside every job.
# Each cell is one gated job. No profile switch (G2 is not approved). GPU time: about 4 minutes of load, 11 cells of 20 s.
# Usage: scripts/round9/run_D.sh        Environment: R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh)
set -u
. "$(dirname "$0")/common.sh"
python3 -I "$here/analyse_clk.py" --selftest || { echo "analysis self-test failed; no GPU job started"; exit 2; }
start_run D
for ms in 0.3 0.6 1.0 1.7; do
    gate "d-vk-$ms-r1" 60 90 bash "$here/jobwrap.sh" "$RUN/loads" "d-vk-$ms-r1" "$BIN/vkr9" clk "name=cell,q=g,ms=$ms,cad=16.7,dur=20"
done
for r in 2 3 4; do
    gate "d-vk-1.0-r$r" 60 90 bash "$here/jobwrap.sh" "$RUN/loads" "d-vk-1.0-r$r" "$BIN/vkr9" clk "name=cell,q=g,ms=1.0,cad=16.7,dur=20"
done
for ms in 0.3 0.6 1.0 1.7; do
    gate "d-hip-$ms-r1" 60 90 bash "$here/jobwrap.sh" "$RUN/loads" "d-hip-$ms-r1" "$BIN/r9_clk" "$RUN/loads/d-hip-$ms-r1.csv" "$ms" 16.7 20
done
# the analysis gets every cell this driver should have run, whether or not its file exists: a missing cell fails it (complete_<cell>)
args=()
for t in d-vk-0.3-r1 d-vk-0.6-r1 d-vk-1.0-r1 d-vk-1.7-r1 d-vk-1.0-r2 d-vk-1.0-r3 d-vk-1.0-r4 d-hip-0.3-r1 d-hip-0.6-r1 d-hip-1.0-r1 d-hip-1.7-r1; do
    f=$RUN/loads/$t-clk-cell.csv
    [[ $t == d-hip-* ]] && f=$RUN/loads/$t.csv
    args+=("$t,$f,$RUN/loads/$t.watch/samples.csv")
done
finish python3 -I "$here/analyse_clk.py" "${args[@]}"
