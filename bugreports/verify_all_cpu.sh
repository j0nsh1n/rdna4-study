#!/usr/bin/env bash
# Runs every bundle's verify_cpu.sh (CPU only, no GPU is opened), then scans bugreports/ for hygiene problems.
#
# Usage: bash bugreports/verify_all_cpu.sh [--out DIR]
#   --out DIR   where compiler output and matrices go (default: a new mktemp directory). Nothing is written
#               inside bugreports/.
#   --skip-bundles   run only the hygiene scan and its self-test (fast).
# Exit 0: all bundles pass and the hygiene scan is clean. Exit 1: any failure. Exit 2: usage error.
#
# Hygiene scan over every file under bugreports/: no home-directory paths, no user or account names, no
# __pycache__ or .pyc files, no emoji, no stray build output (.s, .o, .bc, .hsaco, .log, .out).
set -u
export PYTHONDONTWRITEBYTECODE=1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=""
SKIP=0
while [ $# -gt 0 ]; do
    case $1 in
    --out) OUT=${2:?--out needs a directory}; shift 2 ;;
    --skip-bundles) SKIP=1; shift ;;
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$OUT" ] || OUT=$(mktemp -d "${TMPDIR:-/tmp}/bugreports-verify.XXXXXX")
mkdir -p "$OUT" || exit 2
OUT=$(cd "$OUT" && pwd)

FAIL=0
declare -A STATUS
for b in 01-wmma-call-boundary-hazard 02-wmma-perlane-if-exec 03-divergent-lane-permute-source; do
    if [ "$SKIP" -eq 1 ]; then STATUS[$b]=SKIPPED; continue; fi
    echo "=== $b"
    if bash "$HERE/$b/verify_cpu.sh" --out "$OUT/$b" >"$OUT/$b.stdout" 2>&1; then
        STATUS[$b]=PASS
    else
        STATUS[$b]=FAIL
        FAIL=$((FAIL + 1))
        tail -n 20 "$OUT/$b.stdout"
    fi
    grep -E '^RESULT bundle' "$OUT/$b.stdout"
done

# Patterns are assembled from pieces so that this file does not match itself.
P_HOME="/ho""me/"
P_USERS="/Us""ers/"
ACCOUNT=$(id -un 2>/dev/null || true)
P_EXTRA=${HYGIENE_EXTRA_PATTERNS:-}

# hygiene_scan DIR: prints one line per problem, returns the number of problems in $HYG.
hygiene_scan() {
    local dir=$1 f
    HYG=0
    hyg() { echo "HYGIENE FAIL: $*"; HYG=$((HYG + 1)); }
    while IFS= read -r -d '' f; do
        if grep -I -n -E -e "$P_HOME" -e "$P_USERS" "$f" >/dev/null 2>&1; then hyg "home path in $f"; fi
        if [ -n "$P_EXTRA" ] && grep -I -n -E -e "$P_EXTRA" "$f" >/dev/null 2>&1; then hyg "extra hygiene pattern in $f"; fi
        if [ -n "$ACCOUNT" ] && grep -I -n -w -F -e "$ACCOUNT" "$f" >/dev/null 2>&1; then hyg "login name in $f"; fi
        if LC_ALL=C.UTF-8 grep -I -n -P '[\x{1F000}-\x{1FFFF}\x{2600}-\x{27BF}\x{FE0F}]' "$f" >/dev/null 2>&1; then hyg "emoji in $f"; fi
        case $f in
        */__pycache__/* | *.pyc | *.s | *.o | *.bc | *.hsaco | *.log | *.out) hyg "stray build output $f" ;;
        esac
    done < <(find "$dir" -type f -print0)
    if find "$dir" -name '__pycache__' -print -quit | grep -q .; then hyg "__pycache__ directory present"; fi
}

echo "=== hygiene scan of $HERE"
hygiene_scan "$HERE"
HYG_REAL=$HYG
if [ "$HYG_REAL" -eq 0 ]; then echo "hygiene: clean"; else FAIL=$((FAIL + HYG_REAL)); fi

# Self-test of the scan: deliberate breaks on a copy must be caught (RED), the clean copy must pass (GREEN).
echo "=== hygiene self-test (deliberate breaks on a copy)"
ST=$OUT/hygiene-selftest
rm -rf "$ST"
mkdir -p "$ST"
ST_FAIL=0
st_case() { # name, mutation command (run inside the copy)
    local name=$1
    rm -rf "$ST/copy"
    cp -r "$HERE" "$ST/copy"
    (cd "$ST/copy" && eval "$2")
    hygiene_scan "$ST/copy" >/dev/null
    if [ "$HYG" -ge 1 ]; then echo "  break '$name': RED as required ($HYG finding(s))"; else echo "  break '$name': NOT detected"; ST_FAIL=$((ST_FAIL + 1)); fi
}
st_case "home path" "printf '%s\n' \"see ${P_HOME}someone/file\" >> README.md"
st_case "mac path" "printf '%s\n' \"see ${P_USERS}someone/file\" >> README.md"
st_case "login name" "printf '%s\n' \"built by ${ACCOUNT:-nobody-user}\" >> README.md"
st_case "pycache" "mkdir -p __pycache__ && : > __pycache__/x.cpython-312.pyc"
st_case "stray listing" ": > stray.s"
st_case "emoji" "printf '\\360\\237\\231\\202\\n' >> README.md"
rm -rf "$ST/copy"
cp -r "$HERE" "$ST/copy"
hygiene_scan "$ST/copy" >/dev/null
if [ "$HYG" -eq 0 ]; then echo "  clean copy: GREEN as required"; else echo "  clean copy: unexpected findings"; ST_FAIL=$((ST_FAIL + 1)); fi
rm -rf "$ST"
FAIL=$((FAIL + ST_FAIL))
HYG=$HYG_REAL

echo
echo "| Bundle | Result |"
echo "|---|---|"
for b in 01-wmma-call-boundary-hazard 02-wmma-perlane-if-exec 03-divergent-lane-permute-source; do
    echo "| $b | ${STATUS[$b]} |"
done
echo "| hygiene scan | $([ "$HYG_REAL" -eq 0 ] && echo PASS || echo FAIL) |"
echo "| hygiene self-test | $([ "$ST_FAIL" -eq 0 ] && echo PASS || echo FAIL) |"
echo "output directory: $OUT"
if [ "$FAIL" -eq 0 ]; then echo "ALL CHECKS PASS"; exit 0; fi
echo "FAILURES: $FAIL"
exit 1
