#!/usr/bin/env bash
# Round 11 item 3: the graphics fixed blocks (pixel output / ROPs, texture filtering, rasteriser, surface compression) through Vulkan
# (src/round11/r11_gfx.c, shaders src/round11/gfx/gfx_*). Headless: no window, no swapchain.
#
# Usage: run_r11_gfx.sh [--plan] [--cpu-checks] [--run] [--smoke] [--quick] [--strict-spread] [--only TEXT[,TEXT]] [--out DIR]
#   --plan         print the cells, controls, breaks, guards and the estimated card time; no lock, no GPU, no Vulkan
#   --cpu-checks   no GPU: bash -n, the SPIR-V files, the probe's cell list, the CPU reference test and its nine deliberate breaks, the shader/CPU
#                  hash constants, the sampler's self-test, the gate's self-test, no $HOME path and no __pycache__ in the sources
#   --run          one locked job: controls and deliberate breaks first (fill count and read-back, scissored and skipped last draw, texture gradient,
#                  raster exact count, compression read-back), then every cell; machine-code listings go to <run>/gfx/asm; sampler beside the job
#   --smoke        with --run: the controls and breaks plus one cell of each block (about 20 s of card load), to prove the chain before the full run
#   --strict-spread  with --run: a cell whose repeats differ by more than 10% fails the run (default: counted, verdict PASS WITH SPREAD WARNINGS)
#   --quick        with --run: 2 repeats, half-length bursts, no 8192 x 8192 RGBA16F/RGBA32F textures
#   --only TEXT    with --run: only cells whose name contains TEXT (comma separated: any piece); the controls and breaks still run first
# Environment: see r11_common.sh; R11_GFX_REPS (repeats per cell, default 5); R11_GFX_BURST_MS (burst length, default 100, at most 200); R11_GFX_COOL_C (per-cell hotspot ceiling, default 72, 50 to 85);
#   R11_SAMPLER (path of r11_sampler when it is not in R11_BIN; the sampler is searched in $R11_SAMPLER, $R11_BIN, then <repo>/build/round11/bin).
#   --cpu-checks skips (and says so) the sampler self-test when no sampler is found; --run refuses without one (R11_NO_SAMPLER=1 overrides: no clock record).
# The desktop compositor shares the graphics queue: tell the owner before the window. Bursts are at most 0.25 s each.
# Nothing here writes sysfs, uses sudo, or changes git.
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
# shellcheck source=r11_common.sh
. "$here/r11_common.sh"
PLAN=0 CPU=0 RUN=0 SMOKE=0 QUICK=0 STRICT=0 ONLY="" OUT=${R11_RUN:-}
while (($#)); do
    case $1 in
        --plan) PLAN=1 ;; --cpu-checks) CPU=1 ;; --run) RUN=1 ;; --smoke) SMOKE=1 ;; --quick) QUICK=1 ;; --strict-spread) STRICT=1 ;; --only) ONLY=${2:-}; shift ;; --out) OUT=${2:-}; shift ;;
        *) echo "unknown option $1 (see the header of $SELF)"; exit 2 ;;
    esac
    shift
done
REPS=${R11_GFX_REPS:-5}
BURST=${R11_GFX_BURST_MS:-100}
COOL=${R11_GFX_COOL_C:-72}
SMOKE_ONLY="fill_RGBA8_1MiB,raster_leg256_colormask0,tex_RGBA8_256_bilinear,comp_32MiB_random_disabled"

# build_args <out dir>: the probe's argument list in ARGS. The base (smoke or full) is chosen first, then every modifier is appended,
# so no modifier is lost to a later reassignment (--smoke used to drop --strict-spread and --cool-c).
build_args() {
    if ((SMOKE)); then ARGS=(--run --out "$1" --reps 2 --burst-ms 50 --quick --only "$SMOKE_ONLY")
    else
        ARGS=(--run --out "$1" --reps "$REPS" --burst-ms "$BURST")
        ((QUICK)) && ARGS+=(--quick)
    fi
    ARGS+=(--cool-c "$COOL")
    ((STRICT)) && ARGS+=(--strict-spread)
    [[ -n $ONLY ]] && ARGS+=(--only "$ONLY")
    return 0
}

find_sampler() { # prints the path of r11_sampler, empty when none
    local c
    for c in "${R11_SAMPLER:-}" "$R11_BIN/r11_sampler" "$R11_REPO/build/round11/bin/r11_sampler"; do
        [[ -n $c && -x $c ]] && { echo "$c"; return 0; }
    done
    return 1
}

do_plan() {
    echo "PLAN r11 graphics fixed blocks (item 3): one locked job, probe src/round11/r11_gfx.c (Vulkan 1.3 dynamic rendering, headless), RADV on the RX 9070 XT only"
    if [[ -x $R11_BIN/r11_gfx ]]; then
        local a=""
        ((QUICK)) && a="--quick"
        "$R11_BIN/r11_gfx" --plan $a 2>/dev/null | grep -v '^PLANCELL' || echo "PLAN (the probe's plan failed)"
    else
        echo "PLAN (build the probe to see its cell count: src/round11/gfx/build_gfx.sh)"
    fi
    echo "PLAN pixel output: formats R8 RG8 RGBA8 RGB10A2 R16F RGBA16F R32F RGBA32F and D32 (depth only) at 1, 4 (inside the 8 MB L2), 32 (inside the 64 MB cache) and 512 MiB (GDDR6); blend on/off for RGBA8 and RGBA16F; MSAA 4"
    echo "PLAN raster: right triangles of 0.125, 0.5, 2, 8, 32, 128 and 512 pixel^2 in a 2048 x 2048 grid, colour writes off with an idle depth attachment, depth-only, and culled (control)"
    echo "PLAN texture: compute shader (no ROP), 256 x 256 (cache) and 8192 x 8192 (memory), R8 RGBA8 RGBA16F RGBA32F BC1 BC7; texelFetch, nearest, bilinear, trilinear, aniso 4x and 16x"
    echo "PLAN compression: RGBA8 at 32 and 512 MiB, solid colour against per-pixel random bytes, default against VK_IMAGE_COMPRESSION_DISABLED_EXT, writing and reading back; RADV debug DCC variables are not used (this card is GFX12)"
    echo "PLAN controls: occlusion count == passes x W x H x samples; read-back of two pixel regions against the CPU colour; raster count == triangles x exact covered pixels (culled: 0); a constant texture returns its constant through every filter; 512 MiB random write <= the bus"
    echo "PLAN breaks: (1) last draw scissored to half; (2) last draw skipped; (3) compression read-back of an ordinary image checked as 'disabled'; (4) nearest sampler against the bilinear analytic values; (5) culled pipeline against the visible expectation"
    echo "PLAN not done: MSAA 8, resolve, 3D and array textures, integer render targets, a DCC switch through RADV_DEBUG (wrong generation)"
    echo "PLAN guards: lock, nice -n 10, timeout 3600 s, 0x7550 lookup, 8 GiB floor, hotspot <= ${R11_START_C} C to start, the probe waits below 60 C before every group and pauses at 72 C until 66 C before every cell and between bursts (--cool-c, up to 120 s each), ${R11_GUARD_C} C guard, journalctl -k check, r11_sampler"
    echo "PLAN timing: bursts of ${BURST} ms (limit 250 ms), ${REPS} repeats, GPU timestamps; desktop shares the graphics queue (tell the owner first)"
}

analyse() {
    local f=$R11_RUN/gfx.log bad=0
    [[ -s $f ]] || { echo "no log"; return 14; }
    grep -E '^(DEVICE|COMPRESSION_CONTROL|LIST|CONTROL|BREAK|CC |COMPSWITCH|CLOCKS|BURST|SUMMARY|VERDICT|THERMAL|SKIP)' "$f"
    echo "-- cells (median rate):"
    grep -E '^CELL ' "$f" | awk '{ n = $2; m = ""; for (i = 3; i <= NF; i++) if ($i ~ /^median_(GPix_s|GB_s|GTri_s|GSample_s)=/) m = m " " $i; print n m }'
    echo "-- flagged:"
    grep -E 'CTRLFAIL|BURSTOVER|SPREAD>10|NOT_APPLIED|setting_applied=0|PROBE INVALID|UNVERIFIED|NO SWITCH AVAILABLE' "$f" || echo "(none)"
    grep -q '^VERDICT r11_gfx: PASS' "$f" || bad=1
    grep -E 'CTRLFAIL|PROBE INVALID' "$f" >/dev/null && bad=1
    # a burst over the 250 ms limit is a failed control (expect-3.md control 6), whatever the probe's own verdict said
    local over spr
    over=$(grep -c ' BURSTOVER' "$f"); grep -qE '^SUMMARY .*burst_over=[1-9]' "$f" && over=$((over + 1))
    ((over > 0)) && { echo "BURSTOVER: a burst ran over the 250 ms safety limit: FAIL"; bad=1; }
    grep -q '^SUMMARY .*burst_over=' "$f" || { echo "no burst_over count in the SUMMARY line: FAIL"; bad=1; }
    spr=$(sed -n 's/^SUMMARY .*spread_flagged=\([0-9][0-9]*\).*/\1/p' "$f" | tail -1)
    [[ -n $spr ]] || { echo "no spread_flagged count in the SUMMARY line: FAIL"; bad=1; spr=0; }
    if ((spr > 0)); then
        if grep -q '^SUMMARY .*strict_spread=1' "$f"; then echo "SPREAD: $spr cell(s) with repeats more than 10% apart (--strict-spread: failed)"; bad=1
        else echo "SPREAD WARNING: $spr cell(s) with repeats more than 10% apart (unpinned auto-boost clocks); this is NOT an unqualified clean pass"; fi
    fi
    return $bad
}

if ((PLAN)); then do_plan; ((CPU)) || exit 0; echo; fi

if ((CPU)); then
    bad=0
    ok() { echo "ok   $*"; }
    nok() { echo "FAIL $*"; bad=$((bad + 1)); }
    mkdir -p "${R11_CPUCHECK_DIR:-$R11_REPO/runs}"
    T=$(mktemp -d "${R11_CPUCHECK_DIR:-$R11_REPO/runs}/r11-gfx-cpucheck.XXXXXX") || exit 2
    bash -n "$SELF" && ok "bash -n run_r11_gfx.sh" || nok "bash -n run_r11_gfx.sh"
    bash -n "$here/r11_common.sh" && ok "bash -n r11_common.sh" || nok "bash -n r11_common.sh"
    for c in flock nice timeout journalctl awk glslc spirv-val; do command -v "$c" >/dev/null && ok "tool $c" || nok "tool $c missing"; done
    for b in r11_gfx r11_gfx_selftest; do [[ -x $R11_BIN/$b ]] && ok "binary $b" || nok "binary $b missing in $R11_BIN (src/round11/gfx/build_gfx.sh)"; done
    if [[ -x $R11_BIN/r11_gfx ]]; then
        R11_SPV=$R11_SPV "$R11_BIN/r11_gfx" --cpu-checks >"$T/gfxcpu.txt" 2>&1 && ok "probe --cpu-checks: $(tail -1 "$T/gfxcpu.txt")" || { nok "probe --cpu-checks"; cat "$T/gfxcpu.txt"; }
        do_plan >"$T/plan.txt" 2>&1 && ok "plan prints ($(grep -c . "$T/plan.txt") lines)" || nok "plan"
        "$R11_BIN/r11_gfx" --plan 2>/dev/null | grep -q '^PLANCELL comp_512MiB_random_disabled_write' && ok "plan lists the compression cells" || nok "plan lacks the compression cells"
    fi
    for s in gfx_fill_vert gfx_fill_frag gfx_depth_frag gfx_tri_vert gfx_tri_frag gfx_tex_comp gfx_texctl_comp gfx_readsum_comp; do
        [[ -s $R11_SPV/$s.spv ]] && spirv-val --target-env vulkan1.3 "$R11_SPV/$s.spv" 2>/dev/null && ok "spirv-val $s.spv" || nok "$s.spv missing or invalid in $R11_SPV"
    done
    # the GLSL and C hash must agree: the constants and the shift steps are the same text
    g=$(grep -E '0x7feb352d|0x846ca68b' "$R11_REPO/src/round11/gfx/gfx_common.glsl" | grep -o '0x[0-9a-f]*' | tr '\n' ' ')
    h=$(grep -E '0x7feb352d|0x846ca68b' "$R11_REPO/src/round11/r11_gfx_ref.h" | grep -o '0x[0-9a-f]*' | tr '\n' ' ')
    [[ -n $g && $g == "$h" ]] && ok "hash constants identical in gfx_common.glsl and r11_gfx_ref.h ($g)" || nok "hash constants differ: glsl [$g] c [$h]"
    # the per-pixel colour of the incompressible arm: same index packing in both
    grep -q '((y << 14) | x) ^ seed' "$R11_REPO/src/round11/gfx/gfx_common.glsl" && grep -q '((y << 14) | x) ^ seed' "$R11_REPO/src/round11/r11_gfx_ref.h" && ok "px_hash packing identical in the shader and the reference" || nok "px_hash packing differs"
    if [[ -x $R11_BIN/r11_gfx_selftest ]]; then
        "$R11_BIN/r11_gfx_selftest" >"$T/gfxself.txt" 2>&1 && ok "CPU reference test passes: $(tail -1 "$T/gfxself.txt")" || { nok "CPU reference test"; cat "$T/gfxself.txt"; }
        for n in 1 2 3 4 5 6 7 8 9; do "$R11_BIN/r11_gfx_selftest" --break "$n" >/dev/null 2>&1 && nok "reference-test break $n stayed green" || ok "reference-test break $n went red (as required)"; done
    fi
    if smp=$(find_sampler); then
        "$smp" --selftest >/dev/null && ok "sampler self-test ($smp)" || nok "sampler self-test"
    else
        echo "skip sampler self-test: no r11_sampler in \$R11_SAMPLER, \$R11_BIN or <repo>/build/round11/bin (build_round11.sh builds it; it is not part of this item)"
    fi
    # analyse() on synthetic logs: a clean run passes, a BURSTOVER run and a failed-spread run fail, a spread-only run passes but says so
    synth() { # name verdict-line summary-line [extra cell line]
        mkdir -p "$T/an_$1"
        { echo "CELL fill_x kind=fill median_ms=1.0 max_ms=1.1${4:-}"; echo "$3"; echo "$2"; } >"$T/an_$1/gfx.log"
    }
    synth clean "VERDICT r11_gfx: PASS all controls passed" "SUMMARY controls_or_breaks_failed=0 burst_over=0 spread_flagged=0 strict_spread=0"
    (R11_RUN=$T/an_clean; analyse >"$T/an_clean.out" 2>&1) && ok "analyse(): a clean synthetic run passes" || nok "analyse(): a clean synthetic run failed"
    synth over "VERDICT r11_gfx: PASS all controls passed" "SUMMARY controls_or_breaks_failed=0 burst_over=1 spread_flagged=0 strict_spread=0" " BURSTOVER"
    (R11_RUN=$T/an_over; analyse >"$T/an_over.out" 2>&1) && nok "BREAK BURSTOVER log (even with a PASS verdict line) was accepted" || ok "BREAK BURSTOVER log: analyse() fails as required"
    synth spr "VERDICT r11_gfx: PASS WITH SPREAD WARNINGS (2 cells)" "SUMMARY controls_or_breaks_failed=0 burst_over=0 spread_flagged=2 strict_spread=0" " SPREAD>10%"
    (R11_RUN=$T/an_spr; analyse >"$T/an_spr.out" 2>&1) && grep -q 'NOT an unqualified clean pass' "$T/an_spr.out" && ok "spread-only run: passes and says it is not an unqualified clean pass" || nok "spread-only run mishandled"
    synth sprs "VERDICT r11_gfx: FAIL see CTRLFAIL" "SUMMARY controls_or_breaks_failed=1 burst_over=0 spread_flagged=2 strict_spread=1" " SPREAD>10%"
    (R11_RUN=$T/an_sprs; analyse >"$T/an_sprs.out" 2>&1) && nok "BREAK --strict-spread failure was accepted" || ok "BREAK --strict-spread failure: analyse() fails as required"
    synth nocnt "VERDICT r11_gfx: PASS all controls passed" "SUMMARY controls_or_breaks_failed=0"
    (R11_RUN=$T/an_nocnt; analyse >"$T/an_nocnt.out" 2>&1) && nok "BREAK a SUMMARY without burst_over/spread_flagged counts was accepted" || ok "BREAK SUMMARY without the counts: analyse() fails as required"
    # build_args: every modifier survives --smoke (the old order dropped --strict-spread and --cool-c)
    (SMOKE=1 STRICT=1 QUICK=0 ONLY="" COOL=70; build_args X; a=" ${ARGS[*]} "; [[ $a == *" --strict-spread "* && $a == *" --cool-c 70 "* && $a == *" --reps 2 "* && $a == *" --only $SMOKE_ONLY "* ]]) && ok "build_args: --smoke --strict-spread keeps --strict-spread and --cool-c" || nok "build_args: --smoke lost a modifier"
    (SMOKE=0 STRICT=1 QUICK=1 ONLY="a,b" COOL=70 REPS=5 BURST=100; build_args X; a=" ${ARGS[*]} "; [[ $a == *" --quick "* && $a == *" --strict-spread "* && $a == *" --only a,b "* && $a == *" --reps 5 "* ]]) && ok "build_args: full run keeps --quick --strict-spread --only" || nok "build_args: full run lost a modifier"
    (SMOKE=1 STRICT=0 QUICK=0 ONLY="" COOL=70; build_args X; a=" ${ARGS[*]} "; [[ $a != *"--strict-spread"* ]]) && ok "build_args: no --strict-spread unless asked" || nok "build_args: --strict-spread added unasked"
    # break: the old assignment order loses --strict-spread under --smoke, so the check above can fail
    (SMOKE=1 STRICT=1; ARGS=(--run --out X --reps 5 --burst-ms 100 --cool-c 70); ((STRICT)) && ARGS+=(--strict-spread); ((SMOKE)) && ARGS=(--run --out X --reps 2 --burst-ms 50 --quick --only "$SMOKE_ONLY"); a=" ${ARGS[*]} "; [[ $a != *" --strict-spread "* ]]) && ok "break: the old order drops --strict-spread under --smoke (as the Round 2 review found)" || nok "break: the old order did not lose --strict-spread"
    (R11_SELFTEST=1 r11_gate_selftest "$T/gate") >"$T/gate.txt" 2>&1; g=$(grep -c 'gate-selftest FAIL' "$T/gate.txt")
    ((g == 0)) && ok "gate self-test ($(grep -c 'gate-selftest PASS' "$T/gate.txt") cases pass, $(grep -c 'gate-selftest SKIP' "$T/gate.txt") skipped)" || { nok "gate self-test: $g failed"; grep FAIL "$T/gate.txt"; }
    r11_find_card && ok "card found by device id: $R11_PCI" || nok "card lookup"
    grep -rn "$HOME" "$R11_REPO/src/round11" "$R11_REPO/scripts/round11" 2>/dev/null | grep -q . && nok "a path under \$HOME is written in src/round11 or scripts/round11" || ok "no \$HOME path in src/round11 or scripts/round11"
    find "$R11_REPO/src/round11" "$R11_REPO/scripts/round11" -name __pycache__ 2>/dev/null | grep -q . && nok "a __pycache__ folder is in src/round11 or scripts/round11" || ok "no __pycache__ in src/round11 or scripts/round11"
    if grep -nE 'setenv|putenv' "$R11_REPO/src/round11/r11_gfx.c" >/dev/null || grep -nE '^[[:space:]]*(export[[:space:]]+)?RADV_[A-Z_]+=' "$SELF" >/dev/null; then nok "the item sets a RADV variable (the DCC arm uses VK_EXT_image_compression_control)"; else ok "no RADV variable is set by the probe or the runner"; fi
    rm -rf "$T"
    echo "cpu-checks: $bad problem(s)"
    exit $((bad ? 1 : 0))
fi

if ((RUN)); then
    r11_find_card || exit 3
    R11_RUN=${OUT:-$R11_REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r11gfx}
    export R11_RUN
    mkdir -p "$R11_RUN"
    [[ -s $R11_REPO/scripts/round11/expect/expect-3.md ]] && cp "$R11_REPO/scripts/round11/expect/expect-3.md" "$R11_RUN/expect.md" || r11_die 2 "expect-3.md missing: expectations come first"
    [[ -x $R11_BIN/r11_gfx && -s $R11_SPV/gfx_fill_frag.spv ]] || r11_die 2 "no probe or shaders in $R11_BIN, $R11_SPV: run src/round11/gfx/build_gfx.sh"
    # r11_locked_job starts $R11_BIN/r11_sampler: when the sampler lives elsewhere, stage a folder with links to both
    if ! smp=$(find_sampler); then
        [[ ${R11_NO_SAMPLER:-0} == 1 ]] || r11_die 2 "no r11_sampler found (R11_SAMPLER, R11_BIN, <repo>/build/round11/bin): no clock record; R11_NO_SAMPLER=1 overrides"
    elif [[ $smp != "$R11_BIN/r11_sampler" ]]; then
        mkdir -p "$R11_RUN/bin"
        ln -sf "$R11_BIN/r11_gfx" "$R11_RUN/bin/r11_gfx"
        ln -sf "$smp" "$R11_RUN/bin/r11_sampler"
        R11_BIN=$R11_RUN/bin
    fi
    build_args "$R11_RUN/gfx"
    r11_locked_job gfx 3600 env R11_SPV="$R11_SPV" "$R11_BIN/r11_gfx" "${ARGS[@]}"
    rc=$?
    cp "$R11_RUN/gfx.log" "$R11_RUN/gfx.stdout" 2>/dev/null
    analyse | tee "$R11_RUN/analysis.txt"
    a=${PIPESTATUS[0]}
    ((a != 0 && rc == 0)) && rc=14
    echo "DRIVER exit=$rc"
    exit "$rc"
fi
echo "give a mode: --plan, --cpu-checks or --run (see the header of $SELF)"
exit 2
