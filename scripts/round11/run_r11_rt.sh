#!/usr/bin/env bash
# Round 11 item 1: the ray tracing units through Vulkan ray queries (src/round11/r11_rt.c, shaders src/round11/shaders/rt_*.comp).
# Timing, shader-clock cycles and the compiler listing only: the probe never reads the bytes of an acceleration structure (open question 3, default no).
#
# Usage: run_r11_rt.sh [--plan] [--cpu-checks] [--run] [--smoke] [--only TEXT] [--big] [--strict-clock] [--out DIR]
#   --plan         print the cells, controls, breaks, guards and the estimated card time; no lock, no GPU, no Vulkan
#   --cpu-checks   no GPU: bash -n, the SPIR-V files, the probe's cell list, the CPU reference test and its six deliberate breaks, the shader/CPU hash constants,
#                  the sampler's self-test, the gate's self-test, no $HOME path in the sources
#   --run          one locked job: controls first (analytic one-triangle count, moved-triangle break, extra-instance break), then every scene's 4096-ray CPU check
#                  and its cells, then the chain cells; the machine-code listing of the three pipelines goes to <run>/rt/asm; sampler beside the job
#   --smoke        with --run: only the controls and the one-triangle cells (3 repeats), to prove the chain before the full run
#   --only TEXT    with --run: only cells whose name contains TEXT (the controls still run first)
#   --big          add the 2M-triangle scene (CPU check about 20 s longer)
# Environment: see r11_common.sh; R11_RT_REPS (repeats per rate cell, default 30).
# Nothing here writes sysfs, uses sudo, or changes git.
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
# shellcheck source=r11_common.sh
. "$here/r11_common.sh"
PLAN=0 CPU=0 RUN=0 SMOKE=0 BIG=0 STRICT=0 ONLY="" OUT=${R11_RUN:-}
while (($#)); do
    case $1 in
        --plan) PLAN=1 ;; --cpu-checks) CPU=1 ;; --run) RUN=1 ;; --smoke) SMOKE=1 ;; --big) BIG=1 ;; --strict-clock) STRICT=1 ;; --only) ONLY=${2:-}; shift ;; --out) OUT=${2:-}; shift ;;
        *) echo "unknown option $1 (see the header of $SELF)"; exit 2 ;;
    esac
    shift
done
REPS=${R11_RT_REPS:-30}

do_plan() {
    local extra=""
    ((BIG)) && extra="--big"
    echo "PLAN r11 ray tracing (item 1): one locked job, probe src/round11/r11_rt.c (Vulkan, VK_KHR_ray_query + acceleration_structure + shader_clock), RADV on the RX 9070 XT only"
    "$R11_BIN/r11_rt" --plan $extra 2>/dev/null | grep -v '^PLANCELL' || echo "PLAN (build the probe to see its cell count: scripts/round11/build_round11.sh)"
    echo "PLAN scenes: S0 one triangle; S1 N triangles in a grid, N = 8..262144 (x8); S2 sliver triangles, diagonal (loose boxes) and axis-aligned (tight boxes); S3 1..4096 instances of one triangle, plain and rotated 180 degrees; an empty built top level"
    echo "PLAN rate cells: closest vs first hit, coherent vs shuffled ray order, tMax 1.01 vs 1e4, plus floor cells (tMax 0.5: every ray misses at the root) and 3 occupancy cells; $REPS repeats of 50 ms+ each"
    echo "PLAN latency cells: one lane, a chain of 4096 rays whose origin depends on the last hit through a run-time zero; against the same rays independent on a whole wave"
    echo "PLAN controls: CPU brute force over a 4096-ray sample of every scene (hit/miss exact, distance 1e-4 relative); S0 hand count (64 x 65 / 2 = 2080); rate-cell hit counts equal the CPU's exact lattice count; chain >= 3x the wave; core clock flat within 3 %"
    echo "PLAN breaks: (1) a triangle moved on the card only: the CPU check must report mismatches; (2) the chain with all lanes active: the chain check must FAIL; (3) one extra instance: the hand count must FAIL"
    echo "PLAN not done: an unbuilt top level (undefined use, could hang the card; the empty BUILT one replaces it), the ray-pipeline arm, acceleration-structure byte reads (default no), the 2M scene without --big"
    echo "PLAN guards: lock, nice -n 10, timeout 3600 s, 0x7550 lookup, 8 GiB floor, hotspot <= ${R11_START_C} C to start, the probe waits below 60 C before every scene (up to 120 s each), ${R11_GUARD_C} C guard, journalctl -k check, r11_sampler"
}

analyse() {
    local f=$R11_RUN/rt.log bad=0
    [[ -s $f ]] || { echo "no log"; return 14; }
    grep -E '^(DEVICE|LIST|CONTROL|BREAK|CHECK|CLOCKS|SUMMARY|VERDICT|THERMAL)' "$f"
    echo "-- cells (median Mrays/s, or chain ns per ray):"
    grep -E '^CELL ' "$f" | awk '{ n = $2; m = ""; for (i = 3; i <= NF; i++) if ($i ~ /^(median_Mrays_s|chain_ns_per_ray|chain_over_wave)=/) m = m " " $i; print n m }'
    echo "-- flagged:"
    grep -E 'CTRLFAIL|hits_exact=FAIL|clock_flat=NO|chain_check=FAIL' "$f" || echo "(none)"
    grep -q '^VERDICT r11_rt: PASS' "$f" || bad=1
    grep -E 'CTRLFAIL|hits_exact=FAIL|chain_check=FAIL' "$f" >/dev/null && bad=1
    # the clock control (expect-1.md control 5): count the cells, check the probe's own count, and say plainly when clocks drifted
    local no sm n m
    no=$(grep -c 'clock_flat=NO' "$f")
    sm=$(sed -n 's/^SUMMARY .*clock_flat_failed=\([0-9][0-9]*\/[0-9][0-9]*\).*/\1/p' "$f" | tail -1)
    if [[ -z $sm ]]; then echo "no clock_flat_failed=N/M in the SUMMARY line: FAIL"; bad=1
    else
        n=${sm%/*}; m=${sm#*/}
        echo "CLOCK clock_flat_failed=$n/$m cells (core clock moved more than 3% between first and last repeat)"
        ((n == no)) || { echo "CLOCK the SUMMARY says $n drifting cells but the log has $no clock_flat=NO lines: FAIL"; bad=1; }
        if ((n > 0)); then
            if grep -q '^SUMMARY .*strict_clock=1' "$f"; then echo "CLOCK --strict-clock: drifting cells fail the run"; bad=1
            else echo "CLOCK WARNING: unpinned clock ramp in $n/$m cells; the verdict is not a clean flat-clock pass"; fi
        fi
    fi
    return $bad
}

if ((PLAN)); then do_plan; exit 0; fi

if ((CPU)); then
    bad=0
    ok() { echo "ok   $*"; }
    nok() { echo "FAIL $*"; bad=$((bad + 1)); }
    T=$(mktemp -d "${R11_CPUCHECK_DIR:-$R11_REPO/runs}/r11-rt-cpucheck.XXXXXX") || exit 2
    bash -n "$SELF" && ok "bash -n run_r11_rt.sh" || nok "bash -n run_r11_rt.sh"
    bash -n "$here/r11_common.sh" && ok "bash -n r11_common.sh" || nok "bash -n r11_common.sh"
    for c in flock nice timeout journalctl awk glslc spirv-val; do command -v "$c" >/dev/null && ok "tool $c" || nok "tool $c missing"; done
    for b in r11_rt r11_rt_selftest r11_sampler; do [[ -x $R11_BIN/$b ]] && ok "binary $b" || nok "binary $b missing in $R11_BIN (build_round11.sh)"; done
    if [[ -x $R11_BIN/r11_rt ]]; then
        R11_SPV=$R11_SPV "$R11_BIN/r11_rt" --cpu-checks >"$T/rtcpu.txt" 2>&1 && ok "probe --cpu-checks: $(tail -1 "$T/rtcpu.txt")" || { nok "probe --cpu-checks"; cat "$T/rtcpu.txt"; }
        do_plan >"$T/plan.txt" 2>&1 && ok "plan prints ($(grep -c . "$T/plan.txt") lines)" || nok "plan"
    fi
    for s in rt_rate rt_check rt_chain; do
        [[ -s $R11_SPV/$s.spv ]] && spirv-val --target-env vulkan1.3 "$R11_SPV/$s.spv" 2>/dev/null && ok "spirv-val $s.spv" || nok "$s.spv missing or invalid in $R11_SPV"
    done
    # the GLSL and C hash must agree: the constants and the three shift/multiply steps are the same text
    g=$(grep -E '0x7feb352d|0x846ca68b' "$R11_REPO/src/round11/shaders/rt_common.glsl" | grep -o '0x[0-9a-f]*' | tr '\n' ' ')
    h=$(grep -E '0x7feb352d|0x846ca68b' "$R11_REPO/src/round11/r11_rt_ref.h" | grep -o '0x[0-9a-f]*' | tr '\n' ' ')
    [[ -n $g && $g == "$h" ]] && ok "hash constants identical in rt_common.glsl and r11_rt_ref.h ($g)" || nok "hash constants differ: glsl [$g] c [$h]"
    if [[ -x $R11_BIN/r11_rt_selftest ]]; then
        "$R11_BIN/r11_rt_selftest" >"$T/rtself.txt" 2>&1 && ok "CPU reference test passes: $(tail -1 "$T/rtself.txt")" || { nok "CPU reference test"; cat "$T/rtself.txt"; }
        for n in 1 2 3 4 5 6; do "$R11_BIN/r11_rt_selftest" --break "$n" >/dev/null 2>&1 && nok "reference-test break $n stayed green" || ok "reference-test break $n went red (as required)"; done
    fi
    # analyse() on synthetic logs: flat clocks pass; a drifting clock passes with a warning; --strict-clock fails; a wrong or missing count fails
    synth() { # name cell-line summary-line
        mkdir -p "$T/an_$1"
        { echo "CELL c1 median_Mrays_s=10 clock_flat=$2"; echo "$3"; echo "VERDICT r11_rt: PASS x"; } >"$T/an_$1/rt.log"
    }
    synth flat yes "SUMMARY controls_or_breaks_failed=0 clock_flat_failed=0/1 strict_clock=0"
    (R11_RUN=$T/an_flat; analyse >"$T/an_flat.out" 2>&1) && ok "analyse(): a flat-clock synthetic run passes" || nok "analyse(): flat-clock run failed"
    synth ramp NO "SUMMARY controls_or_breaks_failed=0 clock_flat_failed=1/1 strict_clock=0"
    (R11_RUN=$T/an_ramp; analyse >"$T/an_ramp.out" 2>&1) && grep -q 'not a clean flat-clock pass' "$T/an_ramp.out" && ok "unpinned ramp: passes and says it is not a clean flat-clock pass" || nok "unpinned ramp mishandled"
    synth strict NO "SUMMARY controls_or_breaks_failed=0 clock_flat_failed=1/1 strict_clock=1"
    (R11_RUN=$T/an_strict; analyse >"$T/an_strict.out" 2>&1) && nok "BREAK --strict-clock with a drifting cell was accepted" || ok "BREAK --strict-clock with a drifting cell: analyse() fails as required"
    synth lie NO "SUMMARY controls_or_breaks_failed=0 clock_flat_failed=0/1 strict_clock=0"
    (R11_RUN=$T/an_lie; analyse >"$T/an_lie.out" 2>&1) && nok "BREAK a SUMMARY that hides a clock_flat=NO cell was accepted" || ok "BREAK a SUMMARY that hides a drifting cell: analyse() fails as required"
    synth nocnt yes "SUMMARY controls_or_breaks_failed=0"
    (R11_RUN=$T/an_nocnt; analyse >"$T/an_nocnt.out" 2>&1) && nok "BREAK a SUMMARY without the clock count was accepted" || ok "BREAK SUMMARY without the clock count: analyse() fails as required"
    [[ -x $R11_BIN/r11_sampler ]] && "$R11_BIN/r11_sampler" --selftest >/dev/null && ok "sampler self-test" || nok "sampler self-test"
    (R11_SELFTEST=1 r11_gate_selftest "$T/gate") >"$T/gate.txt" 2>&1; g=$(grep -c 'gate-selftest FAIL' "$T/gate.txt")
    ((g == 0)) && ok "gate self-test ($(grep -c 'gate-selftest PASS' "$T/gate.txt") cases pass, $(grep -c 'gate-selftest SKIP' "$T/gate.txt") skipped)" || { nok "gate self-test: $g failed"; grep FAIL "$T/gate.txt"; }
    r11_find_card && ok "card found by device id: $R11_PCI" || nok "card lookup"
    grep -rn "$HOME" "$R11_REPO/src/round11" "$R11_REPO/scripts/round11" 2>/dev/null | grep -q . && nok "a path under \$HOME is written in src/round11 or scripts/round11" || ok "no \$HOME path in src/round11 or scripts/round11"
    grep -rn "AccelerationStructure.*map\|vkMapMemory.*as\.\|vkCmdCopyAccelerationStructure" "$R11_REPO/src/round11/r11_rt.c" | grep -v '^.*//' | grep -q . && nok "the probe seems to read acceleration-structure memory (open question 3 says no)" || ok "no read of acceleration-structure bytes in r11_rt.c"
    rm -rf "$T"
    echo "cpu-checks: $bad problem(s)"
    exit $((bad ? 1 : 0))
fi

if ((RUN)); then
    r11_find_card || exit 3
    R11_RUN=${OUT:-$R11_REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r11rt}
    export R11_RUN
    mkdir -p "$R11_RUN"
    [[ -s $R11_REPO/scripts/round11/expect/expect-1.md ]] && cp "$R11_REPO/scripts/round11/expect/expect-1.md" "$R11_RUN/expect.md" || r11_die 2 "expect-1.md missing: expectations come first"
    [[ -x $R11_BIN/r11_rt && -s $R11_SPV/rt_rate.spv ]] || r11_die 2 "no probe or shaders in $R11_BIN, $R11_SPV: run scripts/round11/build_round11.sh"
    args=(--run --out "$R11_RUN/rt" --reps "$REPS")
    ((SMOKE)) && args=(--run --out "$R11_RUN/rt" --reps 3 --quick --only S0_n64)
    [[ -n $ONLY ]] && args+=(--only "$ONLY")
    ((BIG)) && args+=(--big)
    ((STRICT)) && args+=(--strict-clock)
    r11_locked_job rt 3600 env R11_SPV="$R11_SPV" "$R11_BIN/r11_rt" "${args[@]}"
    rc=$?
    cp "$R11_RUN/rt.log" "$R11_RUN/rt.stdout" 2>/dev/null
    analyse | tee "$R11_RUN/analysis.txt"
    a=${PIPESTATUS[0]}
    ((a != 0 && rc == 0)) && rc=14
    echo "DRIVER exit=$rc"
    exit "$rc"
fi
echo "give a mode: --plan, --cpu-checks or --run (see the header of $SELF)"
exit 2
