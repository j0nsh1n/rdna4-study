#!/usr/bin/env bash
# CPU-only check for bundle 02 (WMMA inside lane-divergent control flow). No GPU is opened or needed.
#
# Compiles repro.hip (gfx1201, gfx1200) with every compiler found, scans each listing with exec_mask_scan.py
# (a self-contained EXEC-mask tracker), compiles test_wmma_partial_exec.ll, runs FileCheck when installed, prints
# the compiler matrix, then runs deliberate breaks (WMMA hoisted out of the mask in a copy of bad_if: detector
# RED; WMMA injected into a mask in a copy of good_select: detector flags it; originals: GREEN).
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
[ -n "$OUT" ] || OUT=$(mktemp -d "${TMPDIR:-/tmp}/bugreport-02.XXXXXX")
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)
SCAN="python3 -I $HERE/exec_mask_scan.py"
EDIT="python3 -I $HERE/listing_edit.py"

R101=${ROCM_101:-$HOME/.local/opt/rocm-10.1.0}
R102=${ROCM_102:-$HOME/.local/opt/rocm-10.2.0}
FEDORA_CXX=${FEDORA_CXX:-/usr/bin/clang++}
FEDORA_CLANG=${FEDORA_CLANG:-/usr/bin/clang}
FROCM=$R101
[ -d "$FROCM" ] || FROCM=$R102

declare -A HIPCC LLCLANG LABEL OPTS
HIPCC[r101]="$R101/bin/hipcc";  LLCLANG[r101]="$R101/lib/llvm/bin/clang"; OPTS[r101]="O3"
HIPCC[r102]="$R102/bin/hipcc";  LLCLANG[r102]="$R102/lib/llvm/bin/clang"; OPTS[r102]="O3 O1"
HIPCC[fed]="$FEDORA_CXX";       LLCLANG[fed]="$FEDORA_CLANG";             OPTS[fed]="O3"
TAGS="r101 r102 fed"

for t in $TAGS; do
    c=${LLCLANG[$t]}
    if [ -x "$c" ]; then LABEL[$t]=$("$c" --version 2>/dev/null | head -1); else LABEL[$t]=""; fi
done

FAIL=0
fail() { echo "FAIL: $*"; FAIL=$((FAIL + 1)); }
MATRIX=$OUT/matrix.md
: >"$MATRIX"
say() { echo "$*" | tee -a "$MATRIX"; }
count_of() { sed -n "s/^COUNT $2 wmma=\([0-9]*\)\$/\1/p" "$1"; }  # scan-output-file function

build_hip() { # tag opt target src outfile
    local tag=$1 opt=$2 tgt=$3 src=$4 out=$5
    if [ "$tag" = fed ]; then
        "${HIPCC[$tag]}" --rocm-path="$FROCM" -nogpulib -Wno-unused-command-line-argument -x hip --cuda-device-only \
            -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out" >"$out.log" 2>&1
    else
        (cd "$OUT" && "${HIPCC[$tag]}" -x hip --cuda-device-only -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out") >"$out.log" 2>&1
    fi
}

say "# Bundle 02: WMMA under a per-lane condition (CPU only)"
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
say "Cells: number of WMMA instructions the scan finds under a possibly partial EXEC (bad kernels must show 1, the control 0)."
say ""
say "| Compiler | Opt | Target | bad_if | bad_early_return | bad_perlane_loop | good_select | Result |"
say "|---|---|---|---|---|---|---|---|"

FIRST=""
for t in $TAGS; do
    { [ -n "${LABEL[$t]}" ] && [ -x "${HIPCC[$t]}" ]; } || continue
    for opt in ${OPTS[$t]}; do
        for tgt in gfx1201 gfx1200; do
            s=$OUT/repro-$t-$opt-$tgt.s
            if ! build_hip "$t" "$opt" "$tgt" "$HERE/repro.hip" "$s"; then
                fail "compile repro.hip $t $opt $tgt (see $s.log)"
                say "| $t | $opt | $tgt | compile failed | | | | FAIL |"
                continue
            fi
            $SCAN "$s" >"$s.scan"
            a=$(count_of "$s.scan" bad_if); b=$(count_of "$s.scan" bad_early_return)
            c=$(count_of "$s.scan" bad_perlane_loop); g=$(count_of "$s.scan" good_select)
            r=PASS
            [ "${a:-x}" = 1 ] && [ "${b:-x}" = 1 ] && [ "${c:-x}" = 1 ] && [ "${g:-x}" = 0 ] || r=FAIL
            [ "$r" = PASS ] || fail "repro.hip $t $opt $tgt: bad_if=${a:-?} early=${b:-?} loop=${c:-?} select=${g:-?}"
            say "| $t | $opt | $tgt | ${a:-?} | ${b:-?} | ${c:-?} | ${g:-?} | $r |"
            if [ -z "$FIRST" ] && [ "$tgt" = gfx1201 ]; then FIRST=$s; fi
        done
    done
done

say ""
say "## LLVM IR reproducer (test_wmma_partial_exec.ll, -O2)"
say ""
FILECHECK=""
for c in "$R101/lib/llvm/bin/FileCheck" "$R102/lib/llvm/bin/FileCheck" "$(command -v FileCheck 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FILECHECK=$c; break; fi
done
say "| Compiler | Target | scan bad_if / good_select | FileCheck BAD (must pass) | FileCheck GOOD (must pass) |"
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
            "$HERE/test_wmma_partial_exec.ll" -o "$s" >"$s.log" 2>&1; then
            fail "compile .ll $t $tgt (see $s.log)"
            continue
        fi
        $SCAN "$s" >"$s.scan"
        a=$(count_of "$s.scan" bad_if); g=$(count_of "$s.scan" good_select)
        cb="n/a"; cg="n/a"
        if [ -n "$FILECHECK" ]; then
            if "$FILECHECK" --check-prefix=BAD "$HERE/test_wmma_partial_exec.ll" <"$s" >/dev/null 2>&1; then cb=pass; else cb=FAIL; fail ".ll BAD $t $tgt"; fi
            if "$FILECHECK" --check-prefix=GOOD "$HERE/test_wmma_partial_exec.ll" <"$s" >/dev/null 2>&1; then cg=pass; else cg=FAIL; fail ".ll GOOD $t $tgt"; fi
        fi
        [ "${a:-x}" = 1 ] && [ "${g:-x}" = 0 ] || fail ".ll $t $tgt: bad_if=${a:-?} good_select=${g:-?}"
        say "| $t | $tgt | ${a:-?} / ${g:-?} | $cb | $cg |"
        if [ "$tgt" = gfx1201 ] && [ -z "$LL_FIRST" ]; then LL_FIRST=$s; fi
    done
done
[ -n "$FILECHECK" ] || say "(FileCheck not found: only the scanner was applied to the .ll output)"

say ""
say "## Deliberate breaks"
say ""
if [ -z "$FIRST" ]; then
    fail "no gfx1201 listing available for the deliberate breaks"
else
    c0=$(count_of "$FIRST.scan" bad_if)
    g0=$(count_of "$FIRST.scan" good_select)
    say "- original: bad_if=$c0 (GREEN means detected), good_select=$g0 (GREEN means clean)"
    [ "$c0" = 1 ] && [ "$g0" = 0 ] || fail "originals are not GREEN"
    if $EDIT hoist "$FIRST" "$OUT/break-hoisted.s" bad_if; then
        $SCAN "$OUT/break-hoisted.s" >"$OUT/break-hoisted.scan"
        h=$(count_of "$OUT/break-hoisted.scan" bad_if)
        say "- break 1, WMMA of bad_if moved before the mask write: bad_if=$h (RED expected: 0)"
        [ "$h" = 0 ] || fail "break 1 did not turn the detector RED (bad_if=$h)"
    else
        fail "break 1: could not edit the listing"
    fi
    if $EDIT inject "$FIRST" "$OUT/break-injected.s" good_select; then
        $SCAN "$OUT/break-injected.s" >"$OUT/break-injected.scan"
        j=$(count_of "$OUT/break-injected.scan" good_select)
        say "- break 2, WMMA injected after the mask write in good_select: good_select=$j (flagged expected: 1)"
        [ "$j" = 1 ] || fail "break 2 was not flagged (good_select=$j)"
    else
        fail "break 2: could not edit the listing"
    fi
    $SCAN "$FIRST" >"$OUT/restored.scan"
    c1=$(count_of "$OUT/restored.scan" bad_if); g1=$(count_of "$OUT/restored.scan" good_select)
    say "- restored original: bad_if=$c1, good_select=$g1 (GREEN expected: 1 and 0)"
    [ "$c1" = 1 ] && [ "$g1" = 0 ] || fail "restored original is not GREEN"
    if [ -n "$FILECHECK" ] && [ -n "$LL_FIRST" ]; then
        if $EDIT hoist "$LL_FIRST" "$OUT/break-ir-hoisted.s" bad_if; then
            if "$FILECHECK" --check-prefix=BAD "$HERE/test_wmma_partial_exec.ll" <"$OUT/break-ir-hoisted.s" >/dev/null 2>&1; then
                fail "IR break: BAD still passes with the WMMA hoisted out of the mask"
            else
                say "- IR output with the WMMA hoisted out of the mask: FileCheck BAD fails (RED, as expected)"
            fi
        else
            fail "IR break: could not edit the IR listing"
        fi
    fi
fi

say ""
if [ "$FAIL" -eq 0 ]; then
    say "RESULT bundle 02: PASS (WMMA found under a partial mask in all 3 bad kernels, none in the control; breaks RED; restore GREEN)"
    echo "output directory: $OUT"
    exit 0
fi
say "RESULT bundle 02: FAIL ($FAIL problem(s))"
echo "output directory: $OUT"
exit 1
