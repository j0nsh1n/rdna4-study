#!/usr/bin/env bash
# Check 2: build AMD's rj_arith (unmodified, from the download) and the comparison harness. CPU only.
# Usage: c2_build.sh <run dir>      (run dir: runs/<stamp>-extcheck)
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?run dir}
# RJ_ARITH_SRC: a copy of emulation/rocjitsu/lib/arith from github.com/ROCm/rocm-systems (MIT; the study used commit 09ba9dce).
RJ=${RJ_ARITH_SRC:?set RJ_ARITH_SRC to a copy of rocm-systems emulation/rocjitsu/lib/arith}
mkdir -p "$RUN/c2"
[ -f "$RUN/rjbuild/rj_arith/src/librj_arith.a" ] || {
    cmake -S "$ROOT/src/extcheck/rj_build" -B "$RUN/rjbuild" -G Ninja -DRJ_ARITH_SRC="$RJ" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CXX_COMPILER_LAUNCHER=
    nice -n 10 cmake --build "$RUN/rjbuild" --parallel 6
}
python3 -I "$ROOT/scripts/extcheck/c2_extract.py" "$ROOT/src/study" "$RUN/c2/gen_study.h"
nice -n 10 g++ -O2 -std=c++20 -pthread -I"$RUN/c2" -I"$ROOT/src/study" -I"$RJ/include" \
    "$ROOT/src/extcheck/rj_vs_model.cpp" "$RUN/rjbuild/rj_arith/src/librj_arith.a" -o "$RUN/c2/rj_vs_model"
