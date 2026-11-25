#!/usr/bin/env bash
# ctr.sh STATE_LABEL   (auto | profile_standard)
# Round 6 (the round's own copy: private paths replaced by environment variables): profiler counters on the real one-token decode kernels (llama.cpp 3B Q8_0, FP8 patched, 'auto' (FP8 layers + bf16 head; no true 16-bit GGUF exists on disk); vLLM 3B FP8 and bf16
# with --enforce-eager). Per workload two token counts (16 and 32); analyse.py subtracts them so prompt and start-up cancel.
# Only the counters round 5's audit found exact: GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU SQ_WAVE32_INSTS FETCH_SIZE TCP_REQ
# TCP_REQ_MISS, plus SQ_WAVES as a control. Each --pmc group is one app pass (one full rerun of the program).
# Every GPU job goes through scripts/study-llm.sh dev (LLM_EXPECT_PROFILE=Default: lock, 20 ms thermal guard, 10 GB scope, hotspot < 60 C).
# Reads the performance level and the LACT profile before and after every job (out/jobs-<state>.tsv) and refuses if the level is not
# STATE_LABEL. Changes no setting. Output: out/<state>/<job>/pass_N/*.csv and table-<state>.md in this folder.
# Environment: R6_ONLY=substring (only jobs whose name contains it), R6_NO_VLLM=1 (skip vLLM jobs).
# Required: LLM_BUILD_ROOT (llama.cpp builds and ppl-text.txt), LLM_MODELS (the GGUF and vLLM model folders), VLLM_VENV (a venv with vLLM
# on ROCm 10.1). ROCM_DIR (the ROCm 10.1 tree). Optional: CARD_ENV (card record).
# Copy ctr.sh, analyse.py and verify.py into a new folder runs/<name>/ of the repository and run ctr.sh there: ROOT (the repository) is two
# levels above the script, and out/ and watch/ are written beside it. verify.py reads out/ of the folder it is run from.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
STATE=${1:-}
case $STATE in auto | profile_standard) ;; *) echo "usage: ctr.sh auto|profile_standard" >&2; exit 2 ;; esac
CARD_ENV=${CARD_ENV:-$ROOT/runs/latest/card.env}
# shellcheck source=/dev/null
source "$CARD_ENV" || exit 2
[[ -d ${DRM_DEVICE:-} ]] || { echo "bad card record $CARD_ENV" >&2; exit 2; }
LEVEL_FILE=$DRM_DEVICE/power_dpm_force_performance_level
level() { tr -d '[:space:]' <"$LEVEL_FILE"; }
lact_prof() { lact cli profile get 2>&1 | tr '\n' ' ' | sed 's/ *$//'; }
[[ $(level) == "$STATE" ]] || { echo "REFUSED: performance level is '$(level)', STATE_LABEL says '$STATE'" >&2; exit 3; }

export LLM_EXPECT_PROFILE=Default LLM_RUN=$HERE
S=$ROOT/scripts/study-llm.sh
V=${ROCM_DIR:?set ROCM_DIR to the ROCm 10.1 tree}
BR=${LLM_BUILD_ROOT:?set LLM_BUILD_ROOT to the folder of the llama.cpp builds}
UNMOD=$BR/llama.cpp-latest/build-gfx1201-rocm10/bin
PATCHED=$BR/llama.cpp-latest-patched/build-gfx1201-rocm10/bin
M=${LLM_MODELS:?set LLM_MODELS to the model folder}
TEXT=$BR/ppl-text.txt
VVENV=${VLLM_VENV:?set VLLM_VENV to the vLLM venv}
VSITE=$(echo "$VVENV"/lib/python3*/site-packages)
VPYPATH=$VSITE/_rocm_sdk_core/share/amd_smi
OUT=$HERE/out/$STATE
mkdir -p "$HERE/watch" "$OUT"
# counter groups; one pass each. Override with R6_SETS="a b|c d" (groups separated by |).
IFS='|' read -r -a SETS <<<"${R6_SETS:-GRBM_COUNT GRBM_GUI_ACTIVE|SQ_WAVES|SQ_INSTS_VALU|SQ_WAVE32_INSTS|FETCH_SIZE|TCP_REQ TCP_REQ_MISS}"
pm=(); for s in "${SETS[@]}"; do pm+=(--pmc "$s"); done
LOG=$HERE/out/jobs-$STATE.tsv
[[ -s $LOG ]] || echo -e "time\tjob\tlevel_before\tlact_before\tlevel_after\tlact_after\texit" >>"$LOG"
STOP=0
# llama.cpp jobs: the ROCm 10.1 tree under $ROCM_DIR/... vLLM jobs: the venv's own ROCm 10.1 SDK and its rocprofv3, with no LD_LIBRARY_PATH
# (round 6 dry run: the tree's libraries next to the venv's gave "Option 'spirv-expand-step' registered more than once", two LLVM copies).
PROF=$V/bin/rocprofv3; PENV=(env "LD_LIBRARY_PATH=$V/lib" "PATH=$V/bin:$PATH" "ROCM_PATH=$V")
VSDK=$VSITE/_rocm_sdk_core
use_vllm_prof() { PROF=$VSDK/bin/rocprofv3; PENV=(env "PATH=$VSDK/bin:$PATH"); }
use_llama_prof() { PROF=$V/bin/rocprofv3; PENV=(env "LD_LIBRARY_PATH=$V/lib" "PATH=$V/bin:$PATH" "ROCM_PATH=$V"); }

# gjob NAME TIMEOUT_S -- program args...   (one gated rocprofv3 job with level/profile checks around it)
gjob() {
    local name=$1 tmo=$2; shift 2
    local lb la lv_b lv_a rc
    lv_b=$(level); lb=$(lact_prof)
    if [[ $lv_b != "$STATE" ]]; then echo "REFUSED $name: level '$lv_b' != '$STATE'" | tee -a "$LOG" >&2; STOP=1; return 3; fi
    "$S" dev "$name" 60 "$tmo" "$S" __watch "$HERE/watch/$name" "$name" \
        "${PENV[@]}" "$PROF" "${pm[@]}" -f csv -d "$OUT/$name" -o p -- "$@" >/dev/null 2>&1
    rc=$?
    lv_a=$(level); la=$(lact_prof)
    echo -e "$(date -u +%FT%TZ)\t$name\t$lv_b\t$lb\t$lv_a\t$la\t$rc" >>"$LOG"
    [[ $lv_a == "$STATE" ]] || { echo "level changed during $name: '$lv_a'" >&2; STOP=1; }
    ((rc == 3)) && STOP=1
    return "$rc"
}
want() { [[ -z ${R6_ONLY:-} || $1 == *"$R6_ONLY"* ]]; }

# llama.cpp: 128 tokens of context (prefill), then N generated tokens, one repetition, no warm-up
LL=("q8|$UNMOD|llama32-3b-q8_0" "fp8|$PATCHED|llama32-3b-fp8-outq8" "f16|$UNMOD|llama32-3b-auto")
for c in "${LL[@]}"; do
    IFS='|' read -r label bin mdl <<<"$c"
    for N in 16 32; do
        ((STOP)) && break
        n=ll-$label-n$N; want "$n" || continue
        rm -rf "$OUT/$n"
        gjob "$n" 600 "$bin/llama-bench" -m "$M/$mdl.gguf" -ngl 99 -d 128 -p 0 -n "$N" -r 1 --no-warmup -o jsonl
        echo "$STATE $n exit $?"
    done
done
# vLLM, eager, one engine process per job: warm-up pass, then a second pass; each decodes N+1 tokens of a 128-token prompt
if [[ -z ${R6_NO_VLLM:-} ]]; then
    use_vllm_prof
    VL=("fp8|$M/llama32-3b-redhat-fp8" "bf16|$M/llama32-3b-instruct-16bit")
    for c in "${VL[@]}"; do
        IFS='|' read -r label model <<<"$c"
        for N in 16 32; do
            ((STOP)) && break
            n=vl-$label-n$N; want "$n" || continue
            rm -rf "$OUT/$n"
            gjob "$n" 1500 env "PYTHONPATH=$VPYPATH" "$VVENV/bin/python" "$ROOT/scripts/llm4_vllm.py" trace --model "$model" --text "$TEXT" \
                --eager --kind decode --streams 1 --reps "$N" --max-len 1024 --gpu-util 0.6
            echo "$STATE $n exit $?"
        done
    done
fi
python3 -I "$HERE/analyse.py" "$HERE" "$STATE" | tee "$HERE/table-$STATE.md"
exit "$STOP"
