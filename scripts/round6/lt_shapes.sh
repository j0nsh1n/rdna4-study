#!/usr/bin/env bash
# Round 6's own copy. usage: lt_shapes.sh <rocm version dir name, e.g. 10.1.0> <fp8|bf16> <M>
v=$1; t=$2; M=$3
# LT_BIN_DIR holds lt_bench_<version> (one build per ROCm version); ROCM_PARENT holds rocm-<version> trees.
B=${LT_BIN_DIR:?set LT_BIN_DIR}/lt_bench_$v
export LD_LIBRARY_PATH=${ROCM_PARENT:?set ROCM_PARENT}/rocm-$v/lib
for s in qkv:5120:3072 o:3072:3072 gateup:16384:3072 down:3072:8192; do
  IFS=: read -r nm N K <<<"$s"
  echo "== $nm M=$M N=$N K=$K $t"
  $B $t $M $N $K || echo "FAILED $nm"
done
