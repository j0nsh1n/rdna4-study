#!/usr/bin/env bash
# Item E2 driver: single-wave latency at L2 (1 MiB), the memory-side cache (32 MiB, random and page order) and GDDR6 (1 GiB, random,
# fresh and page order), run beside a light background streaming read (r9_bgread: a second process, its own 1 GiB buffer, a few work-
# groups) that holds the memory clock up; each chase labelled by the memory clock the 5 ms sampler read during it. Two idle jobs (no
# background) give the comparison at the idle clock. Every job is gated and has r9_sampler (5 ms) beside it.
# Jobs: e2-idle-r1, e2-idle-r2, then for each blocks value in R9_E2_BLOCKS (default "2 8 32") e2-bg<blocks>-r1 and -r2 (8 jobs).
# GPU time: about 4 minutes of light load (each job: a 4 s settle, then one-wave chases for 10-25 s; the background is a few
# work-groups); wall time 10-15 minutes (host time to build the 1 GiB chains, a wait for the idle memory clock before every job, the
# 10 s pause between jobs).
# Usage: scripts/round9/run_E2.sh        Environment: R9_E2_BLOCKS, R9_REPO, R9_RUN, STUDY_GPU_LOCK, LLM_EXPECT_PROFILE (see common.sh)
set -u
. "$(dirname "$0")/common.sh"
python3 -I "$here/analyse_E2.py" --selftest || { echo "analysis self-test failed; no GPU job started"; exit 2; }
start_run E2
[[ -x $BIN/r9_bgread ]] || { echo "no $BIN/r9_bgread: run scripts/round9/build.sh"; exit 2; }
blocks_list=${R9_E2_BLOCKS:-"2 8 32"}
for r in 1 2; do
    wait_idle_uclk "e2-idle-r$r"
    gate "e2-idle-r$r" 60 300 bash "$here/jobwrap.sh" "$RUN/loads" "e2-idle-r$r" bash "$here/e2_inner.sh" "$RUN/loads" "e2-idle-r$r" 0
    for b in $blocks_list; do
        wait_idle_uclk "e2-bg$b-r$r"
        gate "e2-bg$b-r$r" 60 300 bash "$here/jobwrap.sh" "$RUN/loads" "e2-bg$b-r$r" bash "$here/e2_inner.sh" "$RUN/loads" "e2-bg$b-r$r" "$b"
    done
done
args=()
n=0
for r in 1 2; do
    for t in e2-idle-r$r $(for b in $blocks_list; do echo "e2-bg$b-r$r"; done); do
        args+=("$t,$RUN/r9-$t.log,$RUN/loads/$t.watch/samples.csv")
        n=$((n + 1))
    done
done
finish python3 -I "$here/analyse_E2.py" --expect "$n" "${args[@]}"
