#!/usr/bin/env bash
# usage: fwrap.sh <type_a>: one perf case of the round-2 shapes build, 8B gate/up shape at n = 1 (one case = 1 to 1.5 s of loop)
exec "${LLM_BUILD_ROOT:?set LLM_BUILD_ROOT}"/llama.cpp-llm2-shapes/build-gfx1201-rocm10/bin/test-backend-ops perf -b ROCm0 -o MUL_MAT -p "type_a=$1,type_b=f32,m=14336,n=1,k=4096,bs=\[1,1\]"
