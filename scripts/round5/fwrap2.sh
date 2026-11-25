#!/usr/bin/env bash
# usage: fwrap2.sh <type_a>: one perf case of the PATCHED tree's test-backend-ops (real FP8 decode kernel), 8B down-proj shape, n = 1
exec "${LLM_BUILD_ROOT:?set LLM_BUILD_ROOT}"/llama.cpp-latest-patched/build-gfx1201-rocm10/bin/test-backend-ops perf -b ROCm0 -o MUL_MAT -p "type_a=$1,type_b=f32,m=4096,n=1,k=14336,bs=\[1,1\]"
