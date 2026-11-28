#!/usr/bin/env bash
# Round 11 item 4b: PCIe copy speed, latency and resizable BAR (HIP probe src/round11/r11_pcie.hip).
#
# Usage: run_r11_pcie.sh [--plan] [--cpu-checks] [--run] [--smoke] [--out DIR]
#   --plan         print the jobs, the controls, the guards and the estimated card time; no lock, no GPU
#   --cpu-checks   no GPU: bash -n, tools, the probe's CPU logic test and its deliberate breaks, the sampler's self-test, the gate's self-test, the card's
#                  link and BAR facts read from sysfs (read only), the SDMA switch name in the ROCm runtime, no $HOME path in the sources
#   --run          two locked jobs: the probe with the default copy engine, then with HSA_ENABLE_SDMA=0 (blit-kernel copies); sampler beside both
#   --smoke        with --run: a short probe (copies up to 16 MiB, 5 repeats, 200 latency iterations, 500 ping-pongs) to prove the whole chain first
# Environment: see r11_common.sh; R11_PCIE_MAX_MIB (largest copy, default 1024), R11_PCIE_REPS (default 30).
# Nothing here writes sysfs, uses sudo, or changes git. The probe is the only GPU program; it needs the card's PCI address (found by device id).
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
# shellcheck source=r11_common.sh
. "$here/r11_common.sh"
PLAN=0 CPU=0 RUN=0 SMOKE=0 OUT=${R11_RUN:-}
while (($#)); do
    case $1 in
        --plan) PLAN=1 ;; --cpu-checks) CPU=1 ;; --run) RUN=1 ;; --smoke) SMOKE=1 ;; --out) OUT=${2:-}; shift ;;
        *) echo "unknown option $1 (see the header of $SELF)"; exit 2 ;;
    esac
    shift
done
MAXMIB=${R11_PCIE_MAX_MIB:-1024}
REPS=${R11_PCIE_REPS:-30}

# link_gts / link_lanes: the card's maximum link as sysfs states it ("32.0 GT/s PCIe", "16"); the ceiling is for the maximum, the current state is recorded.
link_gts() { awk '{print $1}' "$R11_DRM/max_link_speed" 2>/dev/null; }
link_lanes() { cat "$R11_DRM/max_link_width" 2>/dev/null; }
# bar_facts: read-only. BAR0 size from the resource file (hex start/end), and the CPU-visible and total card memory from the amdgpu counters.
bar_facts() {
    local s e vis tot
    read -r s e _ < <(sed -n 1p "$R11_DRM/resource")
    vis=$(cat "$R11_DRM/mem_info_vis_vram_total" 2>/dev/null)
    tot=$(cat "$R11_DRM/mem_info_vram_total" 2>/dev/null)
    echo "BAR0 size $(( ($(printf '%d' "$e") - $(printf '%d' "$s") + 1) / 1048576 )) MiB; CPU-visible card memory ${vis:-?} bytes of ${tot:-?}"
    if [[ -n $vis && -n $tot && $vis == "$tot" ]]; then echo "resizable BAR: ON (the CPU sees all of the card's memory)"; else echo "resizable BAR: NOT ON, or not readable (visible $vis of $tot)"; fi
}

do_plan() {
    echo "PLAN r11 PCIe (item 4b): probe src/round11/r11_pcie.hip, two locked jobs (default copy engine; HSA_ENABLE_SDMA=0), about 3-4 min of card time in all"
    echo "PLAN copies: host-to-device, device-to-host, both at once; pinned, pageable, pageable off by one byte; 4 KiB .. ${MAXMIB} MiB in steps of 4x; $REPS repeats (15 above 256 MiB); median, min, max"
    echo "PLAN every cell: the destination is zeroed first, an exact 64-bit hash computed on the card (or on the host after a device-to-host copy) must equal the host's"
    echo "PLAN latency: 4 B and 64 B copy-and-wait, pinned and pageable, 2000 iterations (median, p95, min); flag ping-pong through coherent pinned memory, 5000 round trips, card-timed and CPU-timed"
    echo "PLAN controls: a rate above the PCIe ceiling (32 GT/s x16 x 128/130 / 8 = 63.0 GB/s per direction, 126.0 for both at once) rejects its cell; card-to-card 256 MiB copy must read 0.70-1.05 x 633 GB/s (read plus write)"
    echo "PLAN breaks: (1) one byte flipped on the card after a verified copy: the hash must FAIL; (2) the copy skipped: the rate must exceed the ceiling and be flagged; (3) ping-pong with no responder: the card's wait must time out"
    echo "PLAN also read (sysfs, no GPU): link speed and width before and after, BAR0 size, CPU-visible against total card memory"
    echo "PLAN host needs: pinned and pageable 1 GiB buffers, up to 2 GiB per kind (floor: 8 GiB available, checked); disk under 100 MB"
    echo "PLAN guards: lock, nice -n 10, timeout 420 s per job, 0x7550 card lookup, 8 GiB floor, hotspot <= ${R11_START_C} C to start, ${R11_GUARD_C} C guard, journalctl -k fault check, r11_sampler 100 ms"
}

# expected_cells <max MiB>: the sweep is 4 KiB .. max in steps of 4x, 3 copy directions for pinned and pageable and 2 (no bidir) for the misaligned buffer
expected_cells() {
    local n=0 b=4096 lim=$(($1 * 1048576))
    while ((b <= lim)); do n=$((n + 1)); b=$((b * 4)); done
    echo $((n * 8))
}

analyse() { # analysis of a run folder: the table lines of the 1 GiB cells and every control line, then the verdict of each job
    local f bad=0 want got skips mib=$MAXMIB
    [[ -s $R11_RUN/max-mib.txt ]] && mib=$(<"$R11_RUN/max-mib.txt")
    want=$(expected_cells "$mib")
    for f in "$R11_RUN"/pcie-*.log; do
        [[ -e $f ]] || { echo "no pcie-*.log in $R11_RUN"; return 14; }
        echo "== $(basename "$f" .log)"
        grep -E '^(DEVICE|CONTROL|BREAK|BREAKPREP|INFO|LAT|SUMMARY|VERDICT)' "$f"
        grep -E '^CELL .*bytes=(4096|1048576|67108864|1073741824) ' "$f"
        grep -E '^(CELLFAIL|CELLSKIP)' "$f"
        grep -q '^VERDICT r11_pcie: PASS' "$f" || { echo "JOB FAILED its controls: $(basename "$f")"; bad=1; }
        skips=$(grep -c '^CELLSKIP' "$f")
        got=$(grep -c '^CELL dir=' "$f")
        ((skips == 0)) || { echo "SWEEP SHORT: $skips CELLSKIP line(s) in $(basename "$f")"; bad=1; }
        ((got == want)) || { echo "SWEEP SHORT: $got of the $want planned cells ran in $(basename "$f") (up to $mib MiB)"; bad=1; }
        ((skips == 0 && got == want)) && echo "SWEEP COMPLETE: $got of $want planned cells ran in $(basename "$f"), no CELLSKIP"
    done
    return $bad
}

if ((PLAN)); then do_plan; exit 0; fi

if ((CPU)); then
    bad=0
    ok() { echo "ok   $*"; }
    nok() { echo "FAIL $*"; bad=$((bad + 1)); }
    T=$(mktemp -d "${R11_CPUCHECK_DIR:-$R11_REPO/runs}/r11-pcie-cpucheck.XXXXXX") || exit 2
    bash -n "$SELF" && ok "bash -n run_r11_pcie.sh" || nok "bash -n run_r11_pcie.sh"
    bash -n "$here/r11_common.sh" && ok "bash -n r11_common.sh" || nok "bash -n r11_common.sh"
    for c in flock nice timeout journalctl awk; do command -v "$c" >/dev/null && ok "tool $c" || nok "tool $c missing"; done
    do_plan >"$T/plan.txt" && ok "plan prints" || nok "plan"
    [[ $(expected_cells 1024) == 80 && $(expected_cells 16) == 56 ]] && ok "planned cell count: 80 up to 1024 MiB, 56 up to 16 MiB" || nok "planned cell count wrong ($(expected_cells 1024), $(expected_cells 16))"
    # analyse() on synthetic logs: a complete sweep passes; a CELLSKIP, a short sweep and a passing verdict over a short sweep all fail
    synth() { # name cells skipped
        mkdir -p "$T/an_$1"
        { for ((i = 0; i < $2; i++)); do echo "CELL dir=h2d mem=pinned bytes=4096 reps=1"; done; for ((i = 0; i < $3; i++)); do echo "CELLSKIP mem=pinned bytes=1073741824: allocation failed"; done
          echo "SUMMARY controls_failed=0"; echo "VERDICT r11_pcie: PASS x"; } >"$T/an_$1/pcie-default.log"
        echo 1024 >"$T/an_$1/max-mib.txt"
    }
    synth full 80 0
    (R11_RUN=$T/an_full; analyse >"$T/an_full.out" 2>&1) && grep -q 'SWEEP COMPLETE: 80 of 80' "$T/an_full.out" && ok "analyse(): a complete sweep (80 of 80, no CELLSKIP) passes" || nok "analyse(): the complete sweep failed"
    synth skip 80 1
    (R11_RUN=$T/an_skip; analyse >"$T/an_skip.out" 2>&1) && nok "BREAK a CELLSKIP line under a PASS verdict was accepted" || ok "BREAK CELLSKIP under a PASS verdict: analyse() fails as required"
    synth short 72 0
    (R11_RUN=$T/an_short; analyse >"$T/an_short.out" 2>&1) && nok "BREAK a short sweep (72 of 80 cells) under a PASS verdict was accepted" || ok "BREAK short sweep under a PASS verdict: analyse() fails as required"
    for b in r11_pcie r11_sampler r11_pcie_logic_test; do [[ -x $R11_BIN/$b ]] && ok "binary $R11_BIN/$b" || nok "binary $b missing in $R11_BIN (build_round11.sh)"; done
    if [[ -x $R11_BIN/r11_pcie_logic_test ]]; then
        "$R11_BIN/r11_pcie_logic_test" >"$T/logic.txt" 2>&1 && ok "probe logic test passes (ceilings, rate rejection, hash, breaks)" || { nok "probe logic test"; cat "$T/logic.txt"; }
    fi
    if [[ -x $R11_BIN/r11_sampler ]]; then
        "$R11_BIN/r11_sampler" --selftest >/dev/null && ok "sampler self-test passes" || nok "sampler self-test"
        "$R11_BIN/r11_sampler" --selftest --break >/dev/null && nok "sampler deliberate break stayed green" || ok "sampler deliberate break FAILED as required"
    fi
    (R11_SELFTEST=1 r11_gate_selftest "$T/gate") >"$T/gate.txt" 2>&1; g=$(grep -c 'gate-selftest FAIL' "$T/gate.txt")
    ((g == 0)) && ok "gate self-test ($(grep -c 'gate-selftest PASS' "$T/gate.txt") cases pass, $(grep -c 'gate-selftest SKIP' "$T/gate.txt") skipped)" || { nok "gate self-test: $g failed"; grep FAIL "$T/gate.txt"; }
    if r11_find_card; then
        ok "card found by device id: $R11_PCI ($R11_RENDER)"
        echo "     link now: $(r11_card_state)"
        g=$(link_gts); l=$(link_lanes)
        awk -v g="$g" -v l="$l" 'BEGIN { exit !(g+0 > 0 && l+0 > 0) }' && ok "max link $g GT/s x$l: ceiling $(awk -v g="$g" -v l="$l" 'BEGIN { printf "%.3f", g * l * 128 / 130 / 8 }') GB/s per direction" || nok "max link not readable"
        bar_facts | sed 's/^/     /'
        [[ -r $R11_DRM/pp_dpm_pcie ]] && echo "     pp_dpm_pcie: $(tr '\n' ';' <"$R11_DRM/pp_dpm_pcie")" || echo "     note: pp_dpm_pcie not readable"
    else nok "card lookup"; fi
    lib=$(ls "${ROCM_PATH:-$HOME/.local/opt/rocm}"/lib/libhsa-runtime64.so* 2>/dev/null | head -1)
    if [[ -n $lib ]] && strings "$lib" | grep -qx HSA_ENABLE_SDMA; then ok "HSA_ENABLE_SDMA is a switch name in $(basename "$lib")"; else nok "HSA_ENABLE_SDMA not found in the HSA runtime library (the second job would test nothing)"; fi
    grep -rn "$HOME" "$R11_REPO/src/round11" "$R11_REPO/scripts/round11" 2>/dev/null | grep -q . && nok "a path under \$HOME is written in src/round11 or scripts/round11" || ok "no \$HOME path in src/round11 or scripts/round11"
    rm -rf "$T"
    echo "cpu-checks: $bad problem(s)"
    exit $((bad ? 1 : 0))
fi

if ((RUN)); then
    r11_find_card || exit 3
    R11_RUN=${OUT:-$R11_REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r11pcie}
    export R11_RUN
    mkdir -p "$R11_RUN"
    [[ -s $R11_REPO/scripts/round11/expect/expect-4b.md ]] && cp "$R11_REPO/scripts/round11/expect/expect-4b.md" "$R11_RUN/expect.md" || r11_die 2 "expect-4b.md missing: expectations come first"
    [[ -x $R11_BIN/r11_pcie ]] || r11_die 2 "no probe in $R11_BIN: run scripts/round11/build_round11.sh"
    hip=0
    if [[ -r $R11_REPO/runs/latest/card.env ]]; then hip=$(sed -n 's/^HIP_INDEX=//p' "$R11_REPO/runs/latest/card.env"); hip=${hip:-0}; fi
    { echo "BAR and link facts (sysfs, read only):"; bar_facts; echo "card: $(r11_card_state)"; } | tee "$R11_RUN/bar.txt"
    args=(--pci "$R11_PCI" --gts "$(link_gts)" --lanes "$(link_lanes)" --max-mib "$MAXMIB" --reps "$REPS")
    ((SMOKE)) && args=(--pci "$R11_PCI" --gts "$(link_gts)" --lanes "$(link_lanes)" --max-mib 16 --reps 5 --lat-iters 200 --pp-iters 500)
    ((SMOKE)) && echo 16 >"$R11_RUN/max-mib.txt" || echo "$MAXMIB" >"$R11_RUN/max-mib.txt"
    rc=0
    r11_locked_job pcie-default 420 env HIP_VISIBLE_DEVICES="$hip" "$R11_BIN/r11_pcie" "${args[@]}"; r=$?; ((r != 0)) && rc=$r
    ((rc == 0 || rc == 14)) && { r11_locked_job pcie-sdma0 420 env HIP_VISIBLE_DEVICES="$hip" HSA_ENABLE_SDMA=0 "$R11_BIN/r11_pcie" "${args[@]}"; r=$?; ((r != 0 && rc == 0)) && rc=$r; }
    analyse | tee "$R11_RUN/analysis.txt"
    a=${PIPESTATUS[0]}
    ((a != 0 && rc == 0)) && rc=14
    echo "DRIVER exit=$rc"
    exit "$rc"
fi
echo "give a mode: --plan, --cpu-checks or --run (see the header of $SELF)"
exit 2
