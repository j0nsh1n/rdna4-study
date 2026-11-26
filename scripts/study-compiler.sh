#!/usr/bin/env bash
# Lane 8 of notes/STUDY-PLAN.md (compiler and hazards on gfx12): build every probe with listings, check the listings,
# run every probe on the card, run the deliberate breaks. One command.
#
# Usage: scripts/study-compiler.sh [build|run|breaks|llama|all]   (default all = build, run, breaks)
#   build   compiles src/study/haz_*.hip for gfx1201 into build/<probe> (-save-temps output in build/tmp-<probe>/);
#           haz_issue is built four ways (default, s_delay_alu off, dual issue off, expert scheduling mode) and
#           haz_w64 with -mwavefrontsize64; then the listing checks: scripts/haz_maskscan.py,
#           scripts/haz_hazscan.py and the study's scripts/mem_loopcheck.py with scripts/haz-expect/*.txt
#   run     runs every probe as its own GPU job
#   breaks  runs each deliberate break; each must report FAIL (a check that cannot fail is useless)
#   llama   compiles two of llama.cpp's matrix-multiply instance files for gfx1201 (device code only, about two
#           minutes, several GB of memory) and runs the mask scan on them; not part of "all"
# Environment: STUDY_REPO (default: this folder's parent) supplies src/harness.h, src/study/arith_model.h and
# scripts/mem_loopcheck.py; LLAMA_SRC (a llama.cpp checkout; see the script for its default) for "llama".
# Results: runs/<UTC time>-haz/: one log per job and per check, card.txt (profile, clocks, hotspot), summary.txt.
# Exit status 1 if anything failed.
# GPU jobs: the shared lock, nice, a two-minute limit, HIP device 0 (the discrete card), not started below 8 GB
# available memory or before the hotspot is under 65 C. Every probe is single waves or one work-group.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDY_REPO=${STUDY_REPO:-$ROOT}
LLAMA_SRC=${LLAMA_SRC:-$HOME/Projects/rdna3-fp8/build/llama.cpp-latest}
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
LACT_CONFIG=/etc/lact/config.yaml
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH PYTHONDONTWRITEBYTECODE=1
cd "$ROOT" || exit 2

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$ROOT/runs/$STAMP-haz
mkdir -p "$RUN" "$ROOT/build"
SUMMARY=$RUN/summary.txt
FAILS=0
DRM=$(ls -d /sys/class/drm/card*/device 2>/dev/null | while read -r d; do
    [[ -e $d/pp_dpm_sclk ]] && echo "$d" && break
done)
HW=$(ls -d "$DRM"/hwmon/hwmon* 2>/dev/null | head -1)

note() { echo "$*" | tee -a "$SUMMARY"; }
avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
hotspot() { echo $(($(cat "$HW/temp2_input") / 1000)); }
card_line() {
    echo "$(date -u +%H:%M:%S) profile: $(grep -m1 current_profile "$LACT_CONFIG" | awk '{print $2}')" \
        " sclk: $(grep '\*' "$DRM/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$DRM/pp_dpm_mclk" | tr -s ' ')" \
        " hotspot: $(hotspot) C"
}

# build <name> <extra flags...>: binary build/<name>, listing build/tmp-<name>/*.img.lto.s from src/study/<src>.hip
build_one() {
    local name=$1 src=$2
    shift 2
    local tmp=$ROOT/build/tmp-$name
    mkdir -p "$tmp"
    if (cd "$tmp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 -DHARNESS_ARCH='"gfx1201"' "$@" \
        -I"$STUDY_REPO/src" -I"$STUDY_REPO/src/study" -I"$ROOT/src/study" -save-temps \
        "$ROOT/src/study/$src.hip" -o "$ROOT/build/$name" -lamdhip64) >"$RUN/build-$name.log" 2>&1; then
        note "BUILD $name: ok"
    else
        note "BUILD $name: FAILED (see build-$name.log)"
        FAILS=$((FAILS + 1))
    fi
}

listing() { ls "$ROOT/build/tmp-$1"/*.img.lto.s 2>/dev/null | head -1; }

# check <label> <command...>: a listing check that must pass
check() {
    local label=$1
    shift
    if "$@" >"$RUN/check-$label.log" 2>&1; then
        note "CHECK $label: PASS"
    else
        note "CHECK $label: FAIL (see check-$label.log)"
        FAILS=$((FAILS + 1))
    fi
}

do_build() {
    build_one haz_mask haz_mask
    build_one haz_hazard haz_hazard
    build_one haz_pitfalls haz_pitfalls
    build_one haz_issue-default haz_issue -DHAZ_VARIANT='"default"'
    build_one haz_issue-nodelay haz_issue -DHAZ_VARIANT='"nodelay"' -mllvm --amdgpu-enable-delay-alu=0
    build_one haz_issue-novopd haz_issue -DHAZ_VARIANT='"novopd"' -mllvm --amdgpu-enable-vopd=0
    build_one haz_issue-expert haz_issue -DHAZ_VARIANT='"expert"' -mllvm --amdgpu-expert-scheduling-mode
    build_one haz_w64 haz_w64 -mwavefrontsize64
    build_one haz_slow haz_slow
    build_one haz_memshape haz_memshape
    build_one haz_regs haz_regs
    local E=$ROOT/scripts/haz-expect LC=$STUDY_REPO/scripts/mem_loopcheck.py
    check maskscan-haz_mask python3 "$ROOT/scripts/haz_maskscan.py" "$(listing haz_mask)" "$E/maskscan-haz_mask.txt"
    check hazscan-haz_hazard python3 "$ROOT/scripts/haz_hazscan.py" "$(listing haz_hazard)" "$E/hazscan-haz_hazard.txt"
    check loops-haz_pitfalls python3 "$LC" "$(listing haz_pitfalls)" "$E/haz_pitfalls.txt"
    for v in default nodelay novopd expert; do
        check "loops-haz_issue-$v" python3 "$LC" "$(listing "haz_issue-$v")" "$E/haz_issue-$v.txt"
    done
    for p in haz_slow haz_memshape haz_regs; do
        check "loops-$p" python3 "$LC" "$(listing "$p")" "$E/$p.txt"
    done
}

# gpu <label> <expect: pass|fail> <command...>: one GPU job under the lock
gpu() {
    local label=$1 want=$2
    shift 2
    local log=$RUN/$label.log
    if (($(avail_gb) < 8)); then
        note "JOB $label: NOT STARTED (below 8 GB available)"
        FAILS=$((FAILS + 1))
        return
    fi
    local w=0
    while (($(hotspot) >= 65)); do
        if ((w >= 120)); then
            note "JOB $label: NOT STARTED (hotspot $(hotspot) C after two minutes)"
            FAILS=$((FAILS + 1))
            return
        fi
        sleep 5
        w=$((w + 5))
    done
    {
        echo "# $label: $*"
        echo "# before: $(card_line)"
    } >"$log"
    HIP_VISIBLE_DEVICES=0 CARD_ARCH=gfx1201 timeout 120 flock "$GPU_LOCK" nice -n 10 "$@" >>"$log" 2>&1
    local rc=$?
    echo "# after: $(card_line)  exit $rc" >>"$log"
    if [[ $want == pass && $rc == 0 ]] || [[ $want == fail && $rc == 1 ]]; then
        note "JOB $label: as expected ($want, exit $rc)"
    else
        note "JOB $label: UNEXPECTED (wanted $want, exit $rc; see $label.log)"
        FAILS=$((FAILS + 1))
    fi
}

do_run() {
    card_line >"$RUN/card.txt"
    gpu run-haz_mask pass "$ROOT/build/haz_mask" all
    gpu run-haz_hazard pass "$ROOT/build/haz_hazard" all
    gpu run-haz_pitfalls pass "$ROOT/build/haz_pitfalls" all
    for v in default nodelay novopd expert; do
        gpu "run-haz_issue-$v" pass "$ROOT/build/haz_issue-$v" all
    done
    gpu run-haz_w64 pass "$ROOT/build/haz_w64" all
    gpu run-haz_slow pass "$ROOT/build/haz_slow"
    gpu run-haz_memshape pass "$ROOT/build/haz_memshape"
    gpu run-haz_regs pass "$ROOT/build/haz_regs"
    card_line >>"$RUN/card.txt"
}

# A listing check given a wrong expectation must fail.
break_listing() {
    local label=$1 sedexpr=$2 expect=$3
    shift 3
    sed -E "$sedexpr" "$expect" >"$RUN/break-$label.expect"
    if cmp -s "$expect" "$RUN/break-$label.expect"; then
        note "BREAK $label: NOT APPLIED (the edit changed nothing)"
        FAILS=$((FAILS + 1))
        return
    fi
    if "$@" "$RUN/break-$label.expect" >"$RUN/break-$label.log" 2>&1; then
        note "BREAK $label: UNEXPECTED PASS"
        FAILS=$((FAILS + 1))
    else
        note "BREAK $label: failed as it must"
    fi
}

do_breaks() {
    local E=$ROOT/scripts/haz-expect LC=$STUDY_REPO/scripts/mem_loopcheck.py
    # mask scan: claim the per-lane-if kernel is clean, and claim the clean kernel is masked
    break_listing maskscan-masked-called-clean 's/^\^k_ifload wmma == 1/^k_ifload wmma == 0/' \
        "$E/maskscan-haz_mask.txt" python3 "$ROOT/scripts/haz_maskscan.py" "$(listing haz_mask)"
    break_listing maskscan-clean-called-masked 's/^\^k_sink wmma == 0/^k_sink wmma == 1/' \
        "$E/maskscan-haz_mask.txt" python3 "$ROOT/scripts/haz_maskscan.py" "$(listing haz_mask)"
    # hazard scan: claim the no-V_NOP inline assembly is clean
    break_listing hazscan-raw-called-clean 's/^\^h_asm_raw hazard == 1/^h_asm_raw hazard == 0/' \
        "$E/hazscan-haz_hazard.txt" python3 "$ROOT/scripts/haz_hazscan.py" "$(listing haz_hazard)"
    # loop counts: one wrong count
    break_listing loops-wrong-count 's/^(w_widenILi0E loop .*_b32 ==) 14$/\1 13/' \
        "$E/haz_pitfalls.txt" python3 "$LC" "$(listing haz_pitfalls)"
    # card checks with the fix removed or the expectation inverted
    gpu break-haz_mask fail "$ROOT/build/haz_mask" break
    gpu break-haz_hazard fail "$ROOT/build/haz_hazard" break
    gpu break-haz_pitfalls fail "$ROOT/build/haz_pitfalls" break
    gpu break-haz_issue fail "$ROOT/build/haz_issue-default" break
    gpu break-haz_w64 fail "$ROOT/build/haz_w64" break
    gpu break-haz_slow fail "$ROOT/build/haz_slow" break
    gpu break-haz_memshape fail "$ROOT/build/haz_memshape" break
    gpu break-haz_regs fail "$ROOT/build/haz_regs" break
}

do_llama() {
    local G=$LLAMA_SRC/ggml tmp=$ROOT/build/tmp-haz_llama
    if (($(avail_gb) < 8)); then
        note "LLAMA: NOT STARTED (below 8 GB available)"
        FAILS=$((FAILS + 1))
        return
    fi
    mkdir -p "$tmp"
    for inst in q8_0 q4_0; do
        if (cd "$tmp" && nice -n 10 hipcc -x hip --cuda-device-only -O3 -std=c++17 --offload-arch=gfx1201 -save-temps \
            -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__ -DGGML_HIP_NO_MMQ_MFMA -DGGML_CUDA_NO_FA -I"$G/include" -I"$G/src" \
            -I"$G/src/ggml-cuda" -c "$G/src/ggml-cuda/template-instances/mmq-instance-$inst.cu" \
            -o "$tmp/mmq-$inst.o") >"$RUN/llama-$inst.log" 2>&1; then
            local s
            s=$(ls -t "$tmp"/mmq-instance-"$inst"*gfx1201*.s 2>/dev/null | head -1)
            python3 "$ROOT/scripts/haz_maskscan.py" "$s" >"$RUN/llama-maskscan-$inst.txt"
            python3 "$ROOT/scripts/haz_hazscan.py" "$s" >"$RUN/llama-hazscan-$inst.txt"
            note "LLAMA $inst: built; mask scan: $(grep -c '^MASKED' "$RUN/llama-maskscan-$inst.txt") listed sites;" \
                "hazard scan: $(grep -c '^HAZARD' "$RUN/llama-hazscan-$inst.txt") hazards"
        else
            note "LLAMA $inst: build FAILED (see llama-$inst.log)"
            FAILS=$((FAILS + 1))
        fi
    done
}

note "lane 8 run $STAMP, root $ROOT, study repo $STUDY_REPO, $(hipcc --version 2>/dev/null | grep -m1 -i 'clang version')"
case ${1:-all} in
build) do_build ;;
run) do_run ;;
breaks) do_breaks ;;
llama) do_llama ;;
all)
    do_build
    do_run
    do_breaks
    ;;
*)
    echo "usage: $0 [build|run|breaks|llama|all]" >&2
    exit 2
    ;;
esac
note "failures: $FAILS"
((FAILS == 0))
