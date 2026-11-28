#!/usr/bin/env bash
# Round 10 item 3 build (CPU only; nothing here opens the GPU):
#   shaders -> <out>/spv/*.spv (glslc, spirv-val) : round 9's r9_chase/prod/fused/evict/spin unchanged, plus this item's r10_* shaders
#   <out>/bin/vkr10 (Vulkan, -Wall -Wextra), r9_sampler (round 9's sysfs sampler), r10_cl3 (HIP, wave32) and r10_cl3_w64 (-mwavefrontsize64)
#   <out>/asm/<bin>/ : the code object taken OUT OF THE BINARY THAT WILL RUN (llvm-objdump --offloading), its disassembly and notes
#   the listing checks (check_item3_listing.py: loads, modifiers, waits, wavefront size; with its deliberate breaks) and the SPIR-V
#   check that only r10_chase64c carries the Coherent decoration
# Usage: src/round10/item3/build.sh [--quiet]     Environment: R10_ITEM3_OUT (default <repo>/build/round10/item3), ROCM_PATH
# Exit 1 if anything fails to build or a check fails.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${R10_ITEM3_OUT:-$REPO/build/round10/item3}
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
export ROCM_PATH HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
LLVM=$ROCM_PATH/llvm/bin
PY=(python3 -I)
mkdir -p "$OUT/spv" "$OUT/bin" "$OUT/asm" "$OUT/tmp" "$OUT/logs"
rc=0
say() { [[ ${1:-} == --quiet ]] || echo "$@"; }
fail() { echo "FAILED: $*"; rc=1; }

GL=(nice -n 10 glslc --target-env=vulkan1.3 -O -I "$REPO/src/vk/shaders")
shader() {  # shader <source> <out name> [glslc -D options...]
    local src=$1 name=$2
    shift 2
    if "${GL[@]}" "$@" "$src" -o "$OUT/spv/$name.spv" && spirv-val --target-env vulkan1.3 "$OUT/spv/$name.spv" && [[ -s $OUT/spv/$name.spv ]]; then
        say "shader $name ok ($(stat -c %s "$OUT/spv/$name.spv") bytes)"
    else
        fail "shader $name"
    fi
}
for s in r9_prod r9_chase r9_fused r9_evict r9_spin; do shader "$REPO/src/round9/shaders/$s.comp" "$s"; done
S=$HERE/shaders
shader "$S/r10_chase64.comp" r10_chase64
shader "$S/r10_chase64.comp" r10_chase64v -DLANEV
shader "$S/r10_chase64.comp" r10_chase64c -DCOH
shader "$S/r10_chase64.comp" r10_reach -DPASS2 -DLANEV
shader "$S/r10_prod64.comp" r10_prod64
shader "$S/r10_chain.comp" r10_chain
shader "$S/r10_fused64.comp" r10_fused64

# only the coherent variant may carry the Coherent decoration (it is the cache-policy parity test against HIP's scope:SCOPE_DEV)
for n in r10_chase64 r10_chase64v r10_chase64c r10_reach r10_fused64; do
    c=$(spirv-dis "$OUT/spv/$n.spv" 2>/dev/null | grep -c 'Coherent\|Volatile')
    if { [[ $n == r10_chase64c ]] && ((c == 1)); } || { [[ $n != r10_chase64c ]] && ((c == 0)); }; then say "spirv $n: Coherent/Volatile decorations = $c as designed"; else fail "spirv $n has $c Coherent/Volatile decorations"; fi
done

if nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers -I"$REPO/src/vk" -o "$OUT/bin/vkr10" \
    "$HERE/vkr10.c" "$REPO/src/vk/vkgame_chain.c" "$REPO/src/vk/vkgame_share.c" "$REPO/src/vk/vkgame_burst.c" -lvulkan -lpthread -lm 2>"$OUT/logs/build-vkr10.log"; then
    say "vkr10 ok ($(wc -l <"$OUT/logs/build-vkr10.log") warning lines)"
else
    fail "vkr10 (see $OUT/logs/build-vkr10.log)"
fi
nice -n 10 g++ -O2 -std=c++17 -Wall -Wextra -o "$OUT/bin/r9_sampler" "$REPO/src/round9/r9_sampler.cpp" 2>"$OUT/logs/build-r9_sampler.log" && say "r9_sampler ok" || fail "r9_sampler"

hip() {  # hip <binary> <extra flags...>
    local b=$1
    shift
    (cd "$OUT/tmp" && nice -n 10 hipcc -O3 -std=c++17 -Wall -Wextra --offload-arch=gfx1201 "$@" -DHARNESS_ARCH='"gfx1201"' -I"$REPO/src" -I"$REPO/src/study" \
        "$HERE/r10_cl3.hip" -o "$OUT/bin/$b" -lamdhip64) >"$OUT/logs/build-$b.log" 2>&1
    if [[ -x $OUT/bin/$b ]] && ! grep -q "error:" "$OUT/logs/build-$b.log"; then say "$b ok ($(grep -c "warning:" "$OUT/logs/build-$b.log") warnings)"; else fail "$b (see $OUT/logs/build-$b.log)"; return 1; fi
    local ext=$OUT/asm/$b
    rm -rf "$ext" && mkdir -p "$ext"
    # llvm-objdump --offloading writes the code objects next to the binary; keep the device one, drop the host stub
    (cd "$OUT/bin" && "$LLVM/llvm-objdump" --offloading "$b" >/dev/null 2>&1 && mv "$b.0.hip-amdgcn-amd-amdhsa--gfx1201" "$ext/co.elf" && rm -f "$b".0.host-*) || { fail "offload extraction of $b"; return 1; }
    "$LLVM/llvm-objdump" -d --mcpu=gfx1201 "$ext/co.elf" >"$ext/dis.txt" 2>&1
    "$LLVM/llvm-readelf" --notes "$ext/co.elf" >"$ext/notes.txt" 2>&1
    sha256sum "$OUT/bin/$b" "$ext/co.elf" >"$ext/sha256.txt"
}
check_listing() {  # check_listing <binary> <wave>
    local ext=$OUT/asm/$1
    "${PY[@]}" "$HERE/check_item3_listing.py" "$ext/dis.txt" "$ext/notes.txt" "$2" >"$OUT/logs/check-$1.txt" 2>&1 || fail "listing check of $1 (see $OUT/logs/check-$1.txt)"
    "${PY[@]}" "$HERE/check_item3_listing.py" selftest "$ext/dis.txt" "$ext/notes.txt" "$2" >>"$OUT/logs/check-$1.txt" 2>&1 || fail "listing self-test of $1 (a deliberate break was not caught)"
    say "listing $1: $(grep -m1 '^LISTING' "$OUT/logs/check-$1.txt"), $(grep '^SELFTEST' "$OUT/logs/check-$1.txt")"
}
hip r10_cl3 && check_listing r10_cl3 32
hip r10_cl3_w64 -mwavefrontsize64 -DR10_W64 && check_listing r10_cl3_w64 64
"${PY[@]}" "$HERE/analyse_item3.py" --selftest >"$OUT/logs/analyse-selftest.txt" 2>&1 && say "analyse_item3 self-test: $(tail -1 "$OUT/logs/analyse-selftest.txt")" || fail "analyse_item3 self-test"
exit $rc
