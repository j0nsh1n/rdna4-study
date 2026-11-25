#!/usr/bin/env bash
# Item E driver: the memory hierarchy at the memory clock the card now reaches, each job labelled by its memory clock.
#   steady latency   r9_lat lat   L0 16 KiB, L2 1/2/4 MiB, memory-side cache 32 MiB, GDDR6 512 MiB and 1 GiB, 3 repeats
#   bandwidth        r9_bw        the same sizes ("split" read), 3 repeats
#   cold chases      r9_lat cold  12 processes, half started after the memory clock fell back to 456, half straight after
#                                 the previous job; each chase labelled by the reading just before it
#   background load  e_bg_inner.sh latency and bandwidth beside an FP16 matrix loop (the 772 MHz state of round 7), 3 repeats
# Every job is gated and has r9_sampler (5 ms) beside it. GPU time about 2 minutes of load.
# Usage: scripts/round9/run_E.sh        Environment: R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh)
set -u
. "$(dirname "$0")/common.sh"
python3 -I "$here/analyse_hier.py" --selftest || { echo "analysis self-test failed; no GPU job started"; exit 2; }
bash "$here/selftest_e_bg.sh" || { echo "background-loop self-test failed; no GPU job started"; exit 2; }
grep -aq "cap C" "$BIN/r9_bg" 2>/dev/null || { echo "r9_bg in $BIN is the first version (no throttle) or missing: run scripts/round9/build.sh"; exit 2; }
start_run E
lat=16,1024,2048,4096,32768,524288,1048576
for r in 1 2 3; do
    wait_idle_uclk "e-lat-r$r"
    gate "e-lat-r$r" 60 300 bash "$here/jobwrap.sh" "$RUN/loads" "e-lat-r$r" "$BIN/r9_lat" lat "$lat" 5
    wait_idle_uclk "e-bw-r$r"
    gate "e-bw-r$r" 70 200 bash "$here/jobwrap.sh" "$RUN/loads" "e-bw-r$r" "$BIN/r9_bw" "$lat"
done
for n in $(seq 1 12); do
    ((n % 2)) && wait_idle_uclk "e-cold-$n"
    gate "e-cold-$n" 60 200 bash "$here/jobwrap.sh" "$RUN/loads" "e-cold-$n" "$BIN/r9_lat" cold 16,4096,32768,524288
done
for r in 1 2 3; do
    wait_idle_uclk "e-bg-r$r"
    gate "e-bg-r$r" 60 300 bash "$here/jobwrap.sh" "$RUN/loads" "e-bg-r$r" bash "$here/e_bg_inner.sh" "$RUN/loads" "e-bg-r$r"
done
# every job this driver should have run, whether or not its log exists; --expect names the families and how many jobs each has
args=()
for t in e-lat-r1 e-lat-r2 e-lat-r3 e-bw-r1 e-bw-r2 e-bw-r3 $(seq -f "e-cold-%g" 1 12) e-bg-r1 e-bg-r2 e-bg-r3; do
    args+=("$t,$RUN/r9-$t.log,$RUN/loads/$t.watch/samples.csv")
done
finish python3 -I "$here/analyse_hier.py" --expect e-lat=3,e-bw=3,e-cold=12,e-bg=3 "${args[@]}"
