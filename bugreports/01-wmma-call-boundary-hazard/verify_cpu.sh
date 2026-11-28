#!/usr/bin/env bash
# CPU-only check for bundle 01 (WMMA hazard across a call boundary). No GPU is opened or needed.
#
# Compiles repro.hip (gfx1201, gfx1200) and repro_gfx11.hip (gfx1151, gfx1150) with every compiler found,
# scans each listing with scan_call_gap.py, compiles test_wmma_call_hazard.ll, runs FileCheck on it when
# FileCheck is installed, prints the compiler matrix, then runs deliberate breaks (a V_NOP inserted into a copy
# of the listing must turn the detector RED; the untouched listing must be GREEN again).
#
# Usage: bash verify_cpu.sh [--out DIR]       (default DIR: a new mktemp directory)
# Environment overrides: ROCM_101, ROCM_102 (ROCm install roots), FEDORA_CXX, FEDORA_CLANG.
# Exit 0: every expectation holds. Exit 1: a check failed. Exit 2: nothing could be compiled / bad usage.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=""
while [ $# -gt 0 ]; do
    case $1 in
    --out) OUT=${2:?--out needs a directory}; shift 2 ;;
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$OUT" ] || OUT=$(mktemp -d "${TMPDIR:-/tmp}/bugreport-01.XXXXXX")
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)
SCAN="python3 -I $HERE/scan_call_gap.py"

R101=${ROCM_101:-$HOME/.local/opt/rocm-10.1.0}
R102=${ROCM_102:-$HOME/.local/opt/rocm-10.2.0}
FEDORA_CXX=${FEDORA_CXX:-/usr/bin/clang++}
FEDORA_CLANG=${FEDORA_CLANG:-/usr/bin/clang}
FROCM=$R101
[ -d "$FROCM" ] || FROCM=$R102

# tag | label | hip compiler | plain clang (for .ll) | opt levels
declare -A HIPCC LLCLANG LABEL OPTS
HIPCC[r101]="$R101/bin/hipcc";  LLCLANG[r101]="$R101/lib/llvm/bin/clang"; OPTS[r101]="O3"
HIPCC[r102]="$R102/bin/hipcc";  LLCLANG[r102]="$R102/lib/llvm/bin/clang"; OPTS[r102]="O3 O1"
HIPCC[fed]="$FEDORA_CXX";       LLCLANG[fed]="$FEDORA_CLANG";             OPTS[fed]="O3"
TAGS="r101 r102 fed"

ver() { "$1" --version 2>/dev/null | head -1; }
for t in $TAGS; do
    c=${LLCLANG[$t]}
    if [ -x "$c" ]; then LABEL[$t]=$(ver "$c"); else LABEL[$t]=""; fi
done

FAIL=0
fail() { echo "FAIL: $*"; FAIL=$((FAIL + 1)); }
MATRIX=$OUT/matrix.md
: >"$MATRIX"
say() { echo "$*" | tee -a "$MATRIX"; }

build_hip() { # tag opt target src outfile
    local tag=$1 opt=$2 tgt=$3 src=$4 out=$5
    if [ "$tag" = fed ]; then
        "${HIPCC[$tag]}" --rocm-path="$FROCM" -nogpulib -Wno-unused-command-line-argument -x hip --cuda-device-only \
            -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out" >"$out.log" 2>&1
    else
        (cd "$OUT" && "${HIPCC[$tag]}" -x hip --cuda-device-only -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out") >"$out.log" 2>&1
    fi
}

say "# Bundle 01: WMMA hazard across a call boundary (CPU only)"
say ""
say "Compilers found:"
FOUND=0
for t in $TAGS; do
    if [ -n "${LABEL[$t]}" ] && [ -x "${HIPCC[$t]}" ]; then
        say "- $t: ${LABEL[$t]}"
        FOUND=$((FOUND + 1))
    else
        say "- $t: NOT FOUND (skipped)"
    fi
done
[ "$FOUND" -gt 0 ] || { echo "no compiler available" >&2; exit 2; }
say ""
say "| Compiler | Opt | Target | Source | callee ends on WMMA | call site followed by WMMA | asm V_NOP workaround leaves a gap | straight-line control (bare/with VALU) | Bug reproduced |"
say "|---|---|---|---|---|---|---|---|---|"

FIRST_LISTING=""
for src in repro.hip:"gfx1201 gfx1200" repro_gfx11.hip:"gfx1151 gfx1150"; do
    file=${src%%:*}
    targets=${src#*:}
    for t in $TAGS; do
        { [ -n "${LABEL[$t]}" ] && [ -x "${HIPCC[$t]}" ]; } || continue
        for opt in ${OPTS[$t]}; do
            for tgt in $targets; do
                s=$OUT/${file%.hip}-$t-$opt-$tgt.s
                if ! build_hip "$t" "$opt" "$tgt" "$HERE/$file" "$s"; then
                    fail "compile $file $t $opt $tgt (see $s.log)"
                    say "| $t | $opt | $tgt | $file | compile failed | | | | FAIL |"
                    continue
                fi
                res=$($SCAN "$s")
                echo "$res" >"$s.scan"
                callee=$(echo "$res" | grep -c '^CALLEE .*gap=yes')
                site=$(echo "$res" | grep -c '^CALLSITE caller_plain .*gap=yes')
                wk=$(echo "$res" | grep -c '^CALLSITE workaround_asm_nop .*gap=no')
                ctl=$(echo "$res" | sed -n 's/^STRAIGHT control_inline: matrix_pairs_bare=\([0-9]*\) matrix_pairs_with_valu=\([0-9]*\)/\1\/\2/p')
                verdict=PASS
                [ "$callee" -ge 1 ] && [ "$site" -ge 1 ] || verdict=FAIL
                [ "$ctl" = "0/1" ] && [ "$wk" -ge 1 ] || verdict=FAIL
                [ "$verdict" = PASS ] || fail "$file $t $opt $tgt: callee_gap=$callee callsite_gap=$site control=$ctl workaround_closed=$wk"
                say "| $t | $opt | $tgt | $file | $([ "$callee" -ge 1 ] && echo yes || echo no) | $([ "$site" -ge 1 ] && echo yes || echo no) | $([ "$wk" -ge 1 ] && echo no || echo YES) | ${ctl:-none} | $([ "$verdict" = PASS ] && echo YES || echo NO) |"
                if [ -z "$FIRST_LISTING" ] && [ "$file" = repro.hip ] && [ "$tgt" = gfx1201 ]; then FIRST_LISTING=$s; fi
            done
        done
    done
done

# LLVM IR reproducer. llc is not shipped in every ROCm tree, so the clang driver (same backend) compiles the IR.
say ""
say "## LLVM IR reproducer (test_wmma_call_hazard.ll, gfx1201 and gfx1200, -O2)"
say ""
FILECHECK=""
for c in "$R101/lib/llvm/bin/FileCheck" "$R102/lib/llvm/bin/FileCheck" "$(command -v FileCheck 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FILECHECK=$c; break; fi
done
say "| Compiler | Target | scanner: callee gap / call-site gap | FileCheck CURRENT (must pass) | FileCheck FIXED (must fail today) |"
say "|---|---|---|---|---|"
LL_FIRST=""
for t in $TAGS; do
    c=${LLCLANG[$t]}
    [ -n "${LABEL[$t]}" ] && [ -x "$c" ] || continue
    extra=""
    [ "$t" = fed ] && extra="-nogpulib"
    for tgt in gfx1201 gfx1200; do
        s=$OUT/ir-$t-$tgt.s
        # shellcheck disable=SC2086
        if ! "$c" $extra --target=amdgcn-amd-amdhsa "-mcpu=$tgt" -x ir -S -O2 -Wno-override-module \
            "$HERE/test_wmma_call_hazard.ll" -o "$s" >"$s.log" 2>&1; then
            fail "compile .ll $t $tgt (see $s.log)"
            continue
        fi
        res=$($SCAN "$s")
        callee=$(echo "$res" | grep -c '^CALLEE first_mm .*gap=yes')
        site=$(echo "$res" | grep -c '^CALLSITE caller .*gap=yes')
        cur="n/a"
        fix="n/a"
        if [ -n "$FILECHECK" ]; then
            if "$FILECHECK" --check-prefix=CURRENT "$HERE/test_wmma_call_hazard.ll" <"$s" >/dev/null 2>&1; then cur=pass; else cur=FAIL; fail ".ll CURRENT $t $tgt"; fi
            if "$FILECHECK" --check-prefix=FIXED "$HERE/test_wmma_call_hazard.ll" <"$s" >/dev/null 2>&1; then fix="passes (unexpected)"; fail ".ll FIXED unexpectedly passes $t $tgt"; else fix="fails (gap present)"; fi
        fi
        [ "$callee" -ge 1 ] && [ "$site" -ge 1 ] || fail ".ll $t $tgt: callee_gap=$callee callsite_gap=$site"
        say "| $t | $tgt | $([ "$callee" -ge 1 ] && echo yes || echo no) / $([ "$site" -ge 1 ] && echo yes || echo no) | $cur | $fix |"
        if [ "$tgt" = gfx1201 ] && [ -z "$LL_FIRST" ]; then LL_FIRST=$s; fi
    done
done
[ -n "$FILECHECK" ] || say "(FileCheck not found: only the scanner was applied to the .ll output)"

# Deliberate breaks: close the gap in a copy of the listing, expect RED; the original must be GREEN.
say ""
say "## Deliberate breaks"
say ""
if [ -z "$FIRST_LISTING" ]; then
    fail "no gfx1201 listing available for the deliberate break"
else
    verdict_of() { $SCAN "$1" | sed -n 's/^VERDICT bug_present=\([a-z]*\).*/\1/p'; }
    b1=$OUT/break-callee-vnop.s
    b2=$OUT/break-callsite-vnop.s
    sed 's/^\([[:space:]]*\)s_setpc_b64/\1v_nop\n\1s_setpc_b64/' "$FIRST_LISTING" >"$b1"
    sed 's/^\([[:space:]]*\)s_swappc_b64\(.*\)$/\1s_swappc_b64\2\n\1v_nop/' "$FIRST_LISTING" >"$b2"
    v0=$(verdict_of "$FIRST_LISTING")
    v1=$(verdict_of "$b1")
    v2=$(verdict_of "$b2")
    say "- original listing: bug_present=$v0 (GREEN means the bug is detected)"
    say "- break 1, V_NOP inserted before s_setpc_b64 in the callee: bug_present=$v1 (RED expected)"
    say "- break 2, V_NOP inserted after s_swappc_b64 in the caller: bug_present=$v2 (RED expected)"
    [ "$v0" = yes ] || fail "break control: original listing not detected"
    [ "$v1" = no ] || fail "break 1 did not turn the detector RED"
    [ "$v2" = no ] || fail "break 2 did not turn the detector RED"
    v3=$(verdict_of "$FIRST_LISTING")
    say "- restored original: bug_present=$v3 (GREEN expected)"
    [ "$v3" = yes ] || fail "restored original is not GREEN"
    if [ -n "$FILECHECK" ] && [ -n "$LL_FIRST" ]; then
        b3=$OUT/break-ir-callee-vnop.s
        sed 's/^\([[:space:]]*\)s_setpc_b64/\1v_nop\n\1s_setpc_b64/' "$LL_FIRST" >"$b3"
        if "$FILECHECK" --check-prefix=FIXED "$HERE/test_wmma_call_hazard.ll" <"$b3" >/dev/null 2>&1; then
            say "- IR output with a V_NOP inserted: FileCheck FIXED passes (as expected after a fix)"
        else
            fail "IR break: FIXED did not pass with a V_NOP inserted"
        fi
        if "$FILECHECK" --check-prefix=CURRENT "$HERE/test_wmma_call_hazard.ll" <"$b3" >/dev/null 2>&1; then
            fail "IR break: CURRENT still passes with a V_NOP inserted"
        else
            say "- IR output with a V_NOP inserted: FileCheck CURRENT fails (RED, as expected)"
        fi
    fi
fi

say ""
if [ "$FAIL" -eq 0 ]; then
    say "RESULT bundle 01: PASS (bug reproduced in every listing; every break went RED; restore went GREEN)"
    echo "output directory: $OUT"
    exit 0
fi
say "RESULT bundle 01: FAIL ($FAIL problem(s))"
echo "output directory: $OUT"
exit 1
