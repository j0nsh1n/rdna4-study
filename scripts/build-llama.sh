#!/bin/bash
# Builds llama.cpp (unmodified and patched trees of the RDNA 3 project) for one arch with the active ROCm.
# Usage: scripts/build-llama.sh [arch]   (default gfx1201). Output: <tree>/build-<arch>-<tag>, tag from ROCM_PATH's version.
set -uo pipefail
ARCH=${1:-gfx1201}
RDNA3=${RDNA3:-$HOME/Projects/rdna3-fp8}
: "${ROCM_PATH:?set ROCM_PATH to the ROCm to build with}"
case $(cat "$ROCM_PATH/.info/version" 2>/dev/null) in
    10.*) TAG=rocm10 ;;
    *) echo "build-llama: ROCM_PATH has no 10.x version file; refusing to guess a tag" >&2; exit 2 ;;
esac
avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if ((avail < 8)); then
    echo "build-llama: only $avail GB available, need 8" >&2
    exit 2
fi
rc=0
for tree in llama.cpp-latest llama.cpp-latest-patched; do
    src=$RDNA3/build/$tree
    out=$src/build-$ARCH-$TAG
    if ! HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++" HIP_PATH="$ROCM_PATH" cmake -S "$src" -B "$out" \
        -DGGML_HIP=ON -DGPU_TARGETS="$ARCH" -DAMDGPU_TARGETS="$ARCH" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF >"$out.cmake.log" 2>&1; then
        echo "$tree: configure FAILED (see $out.cmake.log)"
        rc=1
        continue
    fi
    if systemd-run --user --scope -q -p MemoryMax=10G nice -n 10 cmake --build "$out" -j8 \
        --target llama-perplexity llama-bench llama-batched-bench test-backend-ops llama-completion >"$out.build.log" 2>&1; then
        echo "$tree: built in $out"
    else
        echo "$tree: build FAILED (see $out.build.log): $(grep -m2 -E 'error' "$out.build.log")"
        rc=1
    fi
done
exit $rc
