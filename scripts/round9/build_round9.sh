#!/usr/bin/env bash
# Round 9 loads: build the probes and run the listing checks (CPU only, no GPU).
# Usage: scripts/round9/build_round9.sh <ld_order|ld_mat|ld_mat_k|all>
# Toolchain and flags of scripts/study-leftovers-c-a.sh and round 7's r7_build.sh: ROCm under $HOME/.local/opt/rocm,
# hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode. Output: build/round9/bin/<probe>, build/round9/ext/<probe>/
# (code object, disassembly, notes, sha256), build/round9/check-<probe>.txt. The listing is taken OUT OF THE BINARY THAT
# WILL RUN (llvm-objdump --offloading), not from a side build. Exit 1 if a build or a check fails, or if a deliberate
# break does not fail.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
export ROCM_PATH HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
LLVM=$ROCM_PATH/llvm/bin
OUT=$REPO/build/round9
PY=(python3 -I)
FAILED=0
mkdir -p "$OUT/bin" "$OUT/ext"

gen() {
    "${PY[@]}" "$HERE/gen_ld_order.py" "$REPO/src/round9/ld_order_blocks.h" || return 1
    "${PY[@]}" "$HERE/gen_ld_mat.py" "$REPO/src/round9/ld_mat_blocks.h" || return 1
    "${PY[@]}" "$HERE/gen_ld_mat_k.py" "$REPO/src/round9/ld_mat_k_blocks.h" || return 1
}

build_one() {  # build_one <probe>
    local p=$1 dir=$OUT/tmp-$1 ext=$OUT/ext/$1
    mkdir -p "$dir"
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcumode -DHARNESS_ARCH='"gfx1201"' \
        -I"$REPO/src" -I"$REPO/src/study" -I"$REPO/src/round9" "$REPO/src/round9/$p.hip" -o "$OUT/bin/$p" -lamdhip64) \
        >"$OUT/build-$p.log" 2>&1
    if [[ ! -x $OUT/bin/$p ]] || grep -q "error:" "$OUT/build-$p.log"; then
        echo "build FAILED: $p (see $OUT/build-$p.log)"
        FAILED=1
        return 1
    fi
    rm -rf "$ext" && mkdir -p "$ext"
    # llvm-objdump --offloading writes the code objects next to the binary; move the device one, delete the host stub
    "$LLVM/llvm-objdump" --offloading "$OUT/bin/$p" >/dev/null 2>&1
    mv "$OUT/bin/$p".0.hip-amdgcn-amd-amdhsa--gfx1201 "$ext/co.elf" 2>/dev/null
    rm -f "$OUT/bin/$p".0.host-*
    "$LLVM/llvm-objdump" -d --mcpu=gfx1201 "$ext/co.elf" >"$ext/dis.txt" 2>&1
    "$LLVM/llvm-readelf" --notes "$ext/co.elf" >"$ext/notes.txt" 2>&1
    sha256sum "$OUT/bin/$p" "$ext/co.elf" >"$ext/sha256.txt"
    echo "built $p: $(sha256sum "$OUT/bin/$p" | cut -c1-16), listing $ext/dis.txt"
}

check_one() {  # check_one <probe>
    local p=$1 ext=$OUT/ext/$1
    {
        echo "## listing of bin/$p ($(cut -c1-16 <"$ext/sha256.txt" | head -1)), header src/round9/${p}_blocks.h"
        "${PY[@]}" "$HERE/check_listing.py" "$p" "$ext/dis.txt" "$ext/notes.txt" "$REPO/src/round9/${p}_blocks.h"
        echo "## deliberate breaks (every one must fail)"
        "${PY[@]}" "$HERE/check_listing.py" selftest "$p" "$ext/dis.txt" "$ext/notes.txt" "$REPO/src/round9/${p}_blocks.h"
    } >"$OUT/check-$p.txt" 2>&1
    local rc=$?
    cat "$OUT/check-$p.txt"
    if ! "${PY[@]}" "$HERE/check_listing.py" "$p" "$ext/dis.txt" "$ext/notes.txt" "$REPO/src/round9/${p}_blocks.h" >/dev/null 2>&1; then
        echo "listing check FAILED for $p"
        FAILED=1
    fi
    if ! "${PY[@]}" "$HERE/check_listing.py" selftest "$p" "$ext/dis.txt" "$ext/notes.txt" "$REPO/src/round9/${p}_blocks.h" >/dev/null 2>&1; then
        echo "listing self-test FAILED for $p (a deliberate break did not fail, or the real header did)"
        FAILED=1
    fi
    rm -f "$REPO/src/round9/${p}_blocks.h.bad"
    return $rc
}

what=${1:-all}
gen || exit 2
for p in ld_order ld_mat ld_mat_k; do
    [[ $what == all || $what == "$p" ]] || continue
    [[ -f $REPO/src/round9/$p.hip ]] || { echo "no source for $p yet"; continue; }
    build_one "$p" && check_one "$p"
done
exit $FAILED
