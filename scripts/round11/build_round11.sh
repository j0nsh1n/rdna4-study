#!/usr/bin/env bash
# Round 11: compile every probe into <out>/bin and <out>/spv (no GPU is used; nothing is run except the CPU self-tests with --test).
#   shaders -> spv/rt_*.spv (glslc, spirv-val); r11_sampler, r11_pcie_logic_test, r11_rt_selftest, r11_rt (C/C++); r11_pcie (HIP, gfx1201)
# Usage: scripts/round11/build_round11.sh [--test]     Output folder: R11_OUT_ROOT (default <repo>/build/round11). Exit 1 if anything fails.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ROOT=${R11_REPO:-$(cd "$here/../.." && pwd)}
OUT=${R11_OUT_ROOT:-$ROOT/build/round11}
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH PATH
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
for s in rt_rate rt_check rt_chain; do
    step "shader-$s" bash -c "nice -n 10 glslc --target-env=vulkan1.3 -O -I '$S/shaders' '$S/shaders/$s.comp' -o '$OUT/spv/$s.spv' && spirv-val --target-env vulkan1.3 '$OUT/spv/$s.spv'"
done
step r11_sampler nice -n 10 g++ -O2 -std=c++17 -Wall -Wextra -o "$OUT/bin/r11_sampler" "$S/r11_sampler.cpp"
step r11_pcie_logic_test nice -n 10 g++ -O2 -std=c++17 -Wall -Wextra -o "$OUT/bin/r11_pcie_logic_test" "$S/r11_pcie_logic_test.cpp"
step r11_rt_selftest nice -n 10 cc -O2 -std=c11 -Wall -Wextra -o "$OUT/bin/r11_rt_selftest" "$S/r11_rt_selftest.c" -lm
step r11_rt nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$S" -o "$OUT/bin/r11_rt" "$S/r11_rt.c" -lvulkan -lm
step r11_pcie bash -c "cd '$OUT/tmp' && nice -n 10 hipcc -O3 -std=c++17 -Wall -Wextra --offload-arch=gfx1201 -DHARNESS_ARCH='\"gfx1201\"' -I'$ROOT/src' -I'$S' '$S/r11_pcie.hip' -o '$OUT/bin/r11_pcie' -lamdhip64"
if [[ ${1:-} == --test ]]; then
    # CPU self-tests, each with its deliberate breaks: a break that stays green is a failure of the build step.
    "$OUT/bin/r11_sampler" --selftest || rc=1
    "$OUT/bin/r11_sampler" --selftest --break >/dev/null && { echo "sampler selftest: break stayed green"; rc=1; } || echo "sampler selftest break went red (as required)"
    "$OUT/bin/r11_pcie_logic_test" || rc=1
    for b in R11_BREAK_CEILING R11_BREAK_HASH; do
        g++ -O2 -std=c++17 -D$b -o "$OUT/tmp/pcie_logic_$b" "$S/r11_pcie_logic_test.cpp" && "$OUT/tmp/pcie_logic_$b" >/dev/null && { echo "pcie logic test: $b stayed green"; rc=1; } || echo "pcie logic test with $b went red (as required)"
    done
    "$OUT/bin/r11_rt_selftest" || rc=1
    for n in 1 2 3 4 5 6; do
        "$OUT/bin/r11_rt_selftest" --break $n >/dev/null && { echo "rt selftest: break $n stayed green"; rc=1; } || echo "rt selftest break $n went red (as required)"
    done
fi
exit $rc
