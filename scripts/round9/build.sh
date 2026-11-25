#!/usr/bin/env bash
# Round 9 clocks branch: compile the probes of items C, D, E and G1 into build/round9/ (no GPU is used).
#   shaders -> build/round9/spv/*.spv (glslc, spirv-val); vkr9 (Vulkan), r9_cl2 r9_clk r9_lat r9_bw r9_bg r9_bgread (HIP), r9_sampler (C++)
# Usage: scripts/round9/build.sh [--quiet]     Exit 1 if anything fails to build.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ROOT=${R9_REPO:-$(cd "$here/../.." && pwd)}
OUT=$ROOT/build/round9
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH PATH
mkdir -p "$OUT/spv" "$OUT/bin" "$OUT/tmp"
rc=0
for s in r9_prod r9_chase r9_fused r9_evict r9_spin; do
    if nice -n 10 glslc --target-env=vulkan1.3 -O -I "$ROOT/src/vk/shaders" "$ROOT/src/round9/shaders/$s.comp" -o "$OUT/spv/$s.spv" &&
        spirv-val --target-env vulkan1.3 "$OUT/spv/$s.spv"; then
        echo "shader $s ok"
    else
        echo "shader $s FAILED"
        rc=1
    fi
done
if nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$ROOT/src/vk" -o "$OUT/bin/vkr9" \
    "$ROOT/src/round9/vkr9.c" "$ROOT/src/vk/vkgame_chain.c" "$ROOT/src/vk/vkgame_share.c" "$ROOT/src/vk/vkgame_burst.c" \
    -lvulkan -lpthread -lm; then
    echo "vkr9 ok"
else
    echo "vkr9 FAILED"
    rc=1
fi
if nice -n 10 g++ -O2 -std=c++17 -Wall -Wextra -o "$OUT/bin/r9_sampler" "$ROOT/src/round9/r9_sampler.cpp"; then
    echo "r9_sampler ok"
else
    echo "r9_sampler FAILED"
    rc=1
fi
for p in r9_cl2 r9_clk r9_lat r9_bw r9_bg r9_bgread; do
    if (cd "$OUT/tmp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' \
        "$ROOT/src/round9/$p.hip" -o "$OUT/bin/$p" -lamdhip64 2>&1 | grep -E 'error' -A4; exit "${PIPESTATUS[0]}"); then
        echo "$p ok"
    else
        echo "$p FAILED"
        rc=1
    fi
done
exit $rc
