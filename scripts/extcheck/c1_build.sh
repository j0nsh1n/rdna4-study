#!/usr/bin/env bash
# Check 1: build haz_hazard.hip / haz_mask.hip (and the gfx11 port) to assembly, listing only (no GPU).
# Usage: c1_build.sh <out dir>.  Compilers: ROCm 10.1, ROCm 10.2, Fedora clang 22.1.8 (device side only).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?out dir}
mkdir -p "$OUT"
R1=$HOME/.local/opt/rocm-10.1.0
R2=$HOME/.local/opt/rocm-10.2.0
INC="-I$ROOT/src -I$ROOT/src/study"
build() { # tag compiler-and-flags target opt src
    local tag=$1 cc=$2 tgt=$3 opt=$4 src=$5 base
    base=$(basename "$src" .hip)
    local f=$OUT/$tag-$tgt-$opt-$base
    if nice -n 10 $cc -x hip --cuda-device-only -S -$opt -std=c++17 --offload-arch=$tgt -DHARNESS_ARCH="\"$tgt\"" $INC \
        "$src" -o "$f.s" >"$f.log" 2>&1; then echo "ok   $tag $tgt $opt $base"; else echo "FAIL $tag $tgt $opt $base"; fi
}
for src in "$ROOT/src/study/haz_hazard.hip" "$ROOT/src/study/haz_mask.hip"; do
    for t in gfx1201 gfx1200; do
        build r101 "$R1/bin/hipcc" $t O3 "$src"
        build r102 "$R2/bin/hipcc" $t O3 "$src"
        build r102 "$R2/bin/hipcc" $t O1 "$src"
        build fedora22 "/usr/bin/clang++ --rocm-path=$R1 -nogpulib -Wno-unused-command-line-argument" $t O3 "$src"
    done
done
for t in gfx1151 gfx1150; do
    src=$ROOT/src/extcheck/haz_gfx11.hip
    build r101 "$R1/bin/hipcc" $t O3 "$src"
    build r102 "$R2/bin/hipcc" $t O3 "$src"
    build r102 "$R2/bin/hipcc" $t O1 "$src"
    build fedora22 "/usr/bin/clang++ --rocm-path=$R1 -nogpulib -Wno-unused-command-line-argument" $t O3 "$src"
done
