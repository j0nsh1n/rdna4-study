#!/usr/bin/env bash
# Round 11 item 3: compile the graphics probe, its CPU reference test and its shaders (no GPU is used; nothing is run except with --test).
#   shaders -> spv/gfx_*.spv (glslc --target-env=vulkan1.3, spirv-val); bin/r11_gfx (Vulkan, C11) and bin/r11_gfx_selftest (C11)
# Usage: src/round11/gfx/build_gfx.sh [--test]     Output folder: R11_OUT_ROOT (default <repo>/build/round11). Exit 1 if anything fails.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ROOT=${R11_REPO:-$(cd "$here/../../.." && pwd)}
OUT=${R11_OUT_ROOT:-$ROOT/build/round11}
S=$ROOT/src/round11
mkdir -p "$OUT/spv" "$OUT/bin" "$OUT/tmp"
rc=0
step() { # name command...
    local n=$1
    shift
    if "$@" >"$OUT/tmp/$n.log" 2>&1; then
        if grep -qiE 'warning|error' "$OUT/tmp/$n.log"; then echo "$n ok WITH WARNINGS (see $OUT/tmp/$n.log)"; else echo "$n ok"; fi
    else
        echo "$n FAILED (see $OUT/tmp/$n.log)"
        rc=1
    fi
}
for f in gfx_fill.vert gfx_fill.frag gfx_depth.frag gfx_tri.vert gfx_tri.frag gfx_tex.comp gfx_texctl.comp gfx_readsum.comp; do
    n=${f%.*}_${f##*.}
    step "shader-$n" bash -c "nice -n 10 glslc --target-env=vulkan1.3 -O -I '$here' '$here/$f' -o '$OUT/spv/$n.spv' && spirv-val --target-env vulkan1.3 '$OUT/spv/$n.spv'"
done
step r11_gfx_selftest nice -n 10 cc -O2 -std=c11 -Wall -Wextra -I"$S" -o "$OUT/bin/r11_gfx_selftest" "$S/r11_gfx_selftest.c" -lm
step r11_gfx nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$S" -o "$OUT/bin/r11_gfx" "$S/r11_gfx.c" -lvulkan -lm
if [[ ${1:-} == --test ]]; then
    "$OUT/bin/r11_gfx_selftest" || rc=1
    for n in 1 2 3 4 5 6 7 8 9; do
        "$OUT/bin/r11_gfx_selftest" --break $n >/dev/null && { echo "gfx selftest: break $n stayed green"; rc=1; } || echo "gfx selftest break $n went red (as required)"
    done
fi
exit $rc
