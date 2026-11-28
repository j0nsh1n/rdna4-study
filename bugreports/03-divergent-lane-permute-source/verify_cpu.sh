#!/usr/bin/env bash
# CPU-only check for bundle 03 (permlane16/permlanex16 source defined inside the divergent region). No GPU is
# opened or needed.
#
# Compiles repro.hip (gfx1201, gfx1200) with every compiler found, scans each listing with permsrc_scan.py,
# compiles test_permlane_fetch_inactive.ll and runs FileCheck when installed, checks at IR level (opt) that the
# source computation is sunk into the branch before code generation, prints the compiler matrix, then runs
# deliberate breaks (source definition hoisted before the mask in a copy of bad_sunk: detector RED; source
# redefined inside the mask in a copy of good_pinned: detector flags it; originals: GREEN).
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
[ -n "$OUT" ] || OUT=$(mktemp -d "${TMPDIR:-/tmp}/bugreport-03.XXXXXX")
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)
SCAN="python3 -I $HERE/permsrc_scan.py"
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
# src_def of the permute in function $2 from scan-output file $1: inside_mask | before_mask | none | (empty)
src_def() { sed -n "s/^PERMUTE $2 line [0-9]*: .* src_def=\([a-z_]*\)\$/\1/p" "$1" | head -1; }

build_hip() { # tag opt target src outfile
    local tag=$1 opt=$2 tgt=$3 src=$4 out=$5
    if [ "$tag" = fed ]; then
        "${HIPCC[$tag]}" --rocm-path="$FROCM" -nogpulib -Wno-unused-command-line-argument -x hip --cuda-device-only \
            -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out" >"$out.log" 2>&1
    else
        (cd "$OUT" && "${HIPCC[$tag]}" -x hip --cuda-device-only -S "-$opt" "--offload-arch=$tgt" "$src" -o "$out") >"$out.log" 2>&1
    fi
}

say "# Bundle 03: permlane16/permlanex16 source defined inside the divergent region (CPU only)"
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
say "Cells: where the permute's source VGPR is last written (inside_mask = bug, before_mask = good)."
say ""
say "| Compiler | Opt | Target | bad_sunk (permlane16) | good_pinned | bad_sunk_x (permlanex16) | good_pinned_x | Result |"
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
            a=$(src_def "$s.scan" bad_sunk); b=$(src_def "$s.scan" good_pinned)
            c=$(src_def "$s.scan" bad_sunk_x); d=$(src_def "$s.scan" good_pinned_x)
            r=PASS
            [ "$a" = inside_mask ] && [ "$b" = before_mask ] && [ "$c" = inside_mask ] && [ "$d" = before_mask ] || r=FAIL
            [ "$r" = PASS ] || fail "repro.hip $t $opt $tgt: ${a:-?} ${b:-?} ${c:-?} ${d:-?}"
            say "| $t | $opt | $tgt | ${a:-?} | ${b:-?} | ${c:-?} | ${d:-?} | $r |"
            if [ -z "$FIRST" ] && [ "$tgt" = gfx1201 ]; then FIRST=$s; fi
        done
    done
done

say ""
say "## LLVM IR reproducer (test_permlane_fetch_inactive.ll, gfx1201, -O2)"
say ""
FILECHECK=""
OPT=""
for c in "$R101/lib/llvm/bin/FileCheck" "$R102/lib/llvm/bin/FileCheck" "$(command -v FileCheck 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FILECHECK=$c; break; fi
done
for c in "$R101/lib/llvm/bin/opt" "$R102/lib/llvm/bin/opt" "$(command -v opt 2>/dev/null)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then OPT=$c; break; fi
done
say "| Compiler | scan bad_sunk / good_pinned | FileCheck SUNK (must pass) | FileCheck PINNED (must pass) |"
say "|---|---|---|---|"
LL_FIRST=""
for t in $TAGS; do
    c=${LLCLANG[$t]}
    [ -n "${LABEL[$t]}" ] && [ -x "$c" ] || continue
    extra=""
    [ "$t" = fed ] && extra="-nogpulib"
    s=$OUT/ir-$t-gfx1201.s
    # shellcheck disable=SC2086
    if ! "$c" $extra --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -x ir -S -O2 -Wno-override-module \
        "$HERE/test_permlane_fetch_inactive.ll" -o "$s" >"$s.log" 2>&1; then
        fail "compile .ll $t (see $s.log)"
        continue
    fi
    $SCAN "$s" >"$s.scan"
    a=$(src_def "$s.scan" bad_sunk); b=$(src_def "$s.scan" good_pinned)
    cs="n/a"; cp="n/a"
    if [ -n "$FILECHECK" ]; then
        if "$FILECHECK" --check-prefix=SUNK "$HERE/test_permlane_fetch_inactive.ll" <"$s" >/dev/null 2>&1; then cs=pass; else cs=FAIL; fail ".ll SUNK $t"; fi
        if "$FILECHECK" --check-prefix=PINNED "$HERE/test_permlane_fetch_inactive.ll" <"$s" >/dev/null 2>&1; then cp=pass; else cp=FAIL; fail ".ll PINNED $t"; fi
    fi
    [ "$a" = inside_mask ] && [ "$b" = before_mask ] || fail ".ll $t: bad_sunk=${a:-?} good_pinned=${b:-?}"
    say "| $t | ${a:-?} / ${b:-?} | $cs | $cp |"
    [ -n "$LL_FIRST" ] || LL_FIRST=$s
done
[ -n "$FILECHECK" ] || say "(FileCheck not found: only the scanner was applied to the .ll output)"

# Where the sinking happens: in the IR passes, before code generation.
say ""
say "IR-level check: block that holds the load of x after each opt pass (bad_sunk starts with the load in \`entry\`)."
say ""
if [ -n "$OPT" ]; then
    say "| opt pass | block of the load in bad_sunk | block of the load in good_pinned |"
    say "|---|---|---|"
    block_of() { # file function -> block label containing the first load
        awk -v fn="$2" '$0 ~ "^define .*@" fn "\\(" {inf=1} inf && /^[A-Za-z0-9_.]+:/ {blk=$1; sub(":","",blk)} inf && /= load / {print blk; exit} inf && /^}/ {exit}' "$1"
    }
    for pass in none sink instcombine; do
        f=$OUT/ir-$pass.ll
        if [ "$pass" = none ]; then cp "$HERE/test_permlane_fetch_inactive.ll" "$f"; else "$OPT" -S -passes="$pass" "$HERE/test_permlane_fetch_inactive.ll" -o "$f" 2>"$f.log" || { fail "opt -passes=$pass"; continue; }; fi
        bb=$(block_of "$f" bad_sunk); gb=$(block_of "$f" good_pinned)
        say "| $pass | ${bb:-?} | ${gb:-?} |"
        if [ "$pass" = none ]; then [ "$bb" = entry ] && [ "$gb" = entry ] || fail "IR input: loads are not in entry"; fi
        if [ "$pass" != none ]; then [ "$bb" = "then" ] && [ "$gb" = entry ] || fail "opt $pass: bad_sunk load in ${bb:-?}, good_pinned load in ${gb:-?}"; fi
    done
else
    say "(opt not found: IR-level check skipped)"
fi

# The backend's MachineSink is not the cause: with it disabled the result is the same.
say ""
for t in $TAGS; do
    c=${LLCLANG[$t]}
    [ -n "${LABEL[$t]}" ] && [ -x "$c" ] || continue
    extra=""
    [ "$t" = fed ] && extra="-nogpulib"
    s=$OUT/ir-nomachinesink-$t.s
    # shellcheck disable=SC2086
    if "$c" $extra --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -x ir -S -O2 -Wno-override-module -mllvm -disable-machine-sink \
        "$HERE/test_permlane_fetch_inactive.ll" -o "$s" >"$s.log" 2>&1; then
        $SCAN "$s" >"$s.scan"
        a=$(src_def "$s.scan" bad_sunk)
        say "- $t with -mllvm -disable-machine-sink: bad_sunk=${a:-?} (unchanged: the sinking happens in the IR passes)"
        [ "$a" = inside_mask ] || fail "$t with machine sink disabled: bad_sunk=${a:-?}"
    else
        fail "compile .ll $t with machine sink disabled (see $s.log)"
    fi
done

say ""
say "## Deliberate breaks"
say ""
if [ -z "$FIRST" ]; then
    fail "no gfx1201 listing available for the deliberate breaks"
else
    a0=$(src_def "$FIRST.scan" bad_sunk); b0=$(src_def "$FIRST.scan" good_pinned)
    say "- original: bad_sunk=$a0 (GREEN means the bug is detected), good_pinned=$b0 (GREEN means clean)"
    [ "$a0" = inside_mask ] && [ "$b0" = before_mask ] || fail "originals are not GREEN"
    if $EDIT hoist "$FIRST" "$OUT/break-hoisted.s" bad_sunk; then
        $SCAN "$OUT/break-hoisted.s" >"$OUT/break-hoisted.scan"
        h=$(src_def "$OUT/break-hoisted.scan" bad_sunk)
        say "- break 1, source definition moved before the mask write in bad_sunk: bad_sunk=$h (RED expected: before_mask)"
        [ "$h" = before_mask ] || fail "break 1 did not turn the detector RED (bad_sunk=$h)"
    else
        fail "break 1: could not edit the listing"
    fi
    if $EDIT inject "$FIRST" "$OUT/break-injected.s" good_pinned; then
        $SCAN "$OUT/break-injected.s" >"$OUT/break-injected.scan"
        j=$(src_def "$OUT/break-injected.scan" good_pinned)
        say "- break 2, source redefined inside the mask in good_pinned: good_pinned=$j (flagged expected: inside_mask)"
        [ "$j" = inside_mask ] || fail "break 2 was not flagged (good_pinned=$j)"
    else
        fail "break 2: could not edit the listing"
    fi
    $SCAN "$FIRST" >"$OUT/restored.scan"
    a1=$(src_def "$OUT/restored.scan" bad_sunk); b1=$(src_def "$OUT/restored.scan" good_pinned)
    say "- restored original: bad_sunk=$a1, good_pinned=$b1 (GREEN expected: inside_mask and before_mask)"
    [ "$a1" = inside_mask ] && [ "$b1" = before_mask ] || fail "restored original is not GREEN"
    if [ -n "$FILECHECK" ] && [ -n "$LL_FIRST" ]; then
        if $EDIT hoist "$LL_FIRST" "$OUT/break-ir-hoisted.s" bad_sunk; then
            if "$FILECHECK" --check-prefix=SUNK "$HERE/test_permlane_fetch_inactive.ll" <"$OUT/break-ir-hoisted.s" >/dev/null 2>&1; then
                fail "IR break: SUNK still passes with the source hoisted before the mask"
            else
                say "- IR output with the source hoisted before the mask: FileCheck SUNK fails (RED, as expected)"
            fi
        else
            fail "IR break: could not edit the IR listing"
        fi
    fi
fi

say ""
if [ "$FAIL" -eq 0 ]; then
    say "RESULT bundle 03: PASS (bad_sunk defines the source inside the mask, good_pinned before it; breaks RED; restore GREEN)"
    echo "output directory: $OUT"
    exit 0
fi
say "RESULT bundle 03: FAIL ($FAIL problem(s))"
echo "output directory: $OUT"
exit 1
