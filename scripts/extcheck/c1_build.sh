#!/usr/bin/env bash
# Check 1: build haz_hazard.hip / haz_mask.hip (and the gfx11 port) to assembly, listing only (no GPU).
# Usage: c1_build.sh <out dir>.
# Compilers (override with env vars): ROCM_101, ROCM_102, FEDORA_CXX (device side only).
# Exit 0 only when all 24 builds succeed and produce non-empty .s listings; exit 1 otherwise.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?out dir}
mkdir -p "$OUT"
R1=${ROCM_101:-$HOME/.local/opt/rocm-10.1.0}
R2=${ROCM_102:-$HOME/.local/opt/rocm-10.2.0}
FCXX=${FEDORA_CXX:-/usr/bin/clang++}
INC=(-I"$ROOT/src" -I"$ROOT/src/study")
N_OK=0
N_FAIL=0
N_EXP=0
build() { # tag target opt src compiler [extra flags...]
    local tag=$1 tgt=$2 opt=$3 src=$4 base
    shift 4
    base=$(basename "$src" .hip)
    local f=$OUT/$tag-$tgt-$opt-$base
    N_EXP=$((N_EXP + 1))
    if nice -n 10 "$@" -x hip --cuda-device-only -S "-$opt" -std=c++17 --offload-arch="$tgt" -DHARNESS_ARCH="\"$tgt\"" "${INC[@]}" \
        "$src" -o "$f.s" >"$f.log" 2>&1 && [[ -s "$f.s" ]]; then
        echo "ok   $tag $tgt $opt $base"
        N_OK=$((N_OK + 1))
    else
        echo "FAIL $tag $tgt $opt $base"
        N_FAIL=$((N_FAIL + 1))
    fi
}
for src in "$ROOT/src/study/haz_hazard.hip" "$ROOT/src/study/haz_mask.hip"; do
    for t in gfx1201 gfx1200; do
        build r101 "$t" O3 "$src" "$R1/bin/hipcc"
        build r102 "$t" O3 "$src" "$R2/bin/hipcc"
        build r102 "$t" O1 "$src" "$R2/bin/hipcc"
        build fedora22 "$t" O3 "$src" "$FCXX" "--rocm-path=$R1" -nogpulib -Wno-unused-command-line-argument
    done
done
for t in gfx1151 gfx1150; do
    src=$ROOT/src/extcheck/haz_gfx11.hip
    build r101 "$t" O3 "$src" "$R1/bin/hipcc"
    build r102 "$t" O3 "$src" "$R2/bin/hipcc"
    build r102 "$t" O1 "$src" "$R2/bin/hipcc"
    build fedora22 "$t" O3 "$src" "$FCXX" "--rocm-path=$R1" -nogpulib -Wno-unused-command-line-argument
done
echo "c1_build summary: ok=$N_OK fail=$N_FAIL expected=$N_EXP"
((N_FAIL == 0 && N_OK == N_EXP)) || exit 1
