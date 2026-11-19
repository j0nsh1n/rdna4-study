#!/usr/bin/env bash
# Leftovers C, lane A (measurements/leftovers-c-a.md): builds and runs the lane's probes, one job per item mode.
#
# Usage: scripts/study-leftovers-c-a.sh [build|run|breaks|all] [MODE...]   (default all = build, run, breaks)
#   build   compiles every probe for gfx1201 into build/leftc-a/bin (with -save-temps into build/leftc-a/tmp-<name>),
#           checks that src/study/left_core_blocks.h is what scripts/leftc_gen_core.py writes, and checks the listing
#           of every timed region (scripts/leftc_asmcheck.py and the per-probe checks below)
#   run     every job; with MODE arguments only the jobs whose mode (first argument, or the program name) matches
#   breaks  each deliberate break; each must report FAIL (a check that cannot fail is useless)
# Environment: CARD_ENV=<card.env> (default runs/latest/card.env).
# Results: runs/<UTC time>-leftc-a/: one log per job and summary.md. Exit status 1 if anything failed.
# GPU jobs: the shared lock (taken per job, never across a build or a cool-down), nice, a time limit, device index 0,
# not started below 8 GB available memory, not started unless `lact cli profile get` reads "Testing" (read-only);
# clocks, power and hotspot are logged before and after each job. Thermal rule: the probes are one wave or a few
# waves, except the whole-card bursts of l0card (0.1 s) and misal (about 35 ms each): before such a job the script
# waits outside the GPU lock until the hotspot is below 55 C (at most 120 s; if it is still at 55 C or more the job is
# NOT STARTED), and inside the program a burst starts only below 55 C and no burst follows one that ended at 95 C or
# more. l0card and misal bw run one burst per job, three jobs per configuration (l0card W = 16: four with its break).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/study-leftovers-c-a.sh"
cd "$REPO" || exit 2
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
GPU_LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
mkdir -p "${GPU_LOCK%/*}"
ROCM_PATH=$HOME/.local/opt/rocm
HIP_PATH=$ROCM_PATH
PATH=$ROCM_PATH/bin:$PATH
export ROCM_PATH HIP_PATH PATH

snap() {
    local drm=$1 hw=$2
    echo "sclk: $(grep '\*' "$drm/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$drm/pp_dpm_mclk" | tr -s ' ')" \
        " power: $(($(cat "$hw/power1_average") / 1000000)) W  hotspot: $(($(cat "$hw/temp2_input") / 1000)) C"
}

lact_profile() {
    lact cli profile get 2>/dev/null | head -1 | tr -d '[:space:]'
}

# Locked runner: re-entered under flock, so the clocks are read while this job holds the GPU.
if [[ ${1:-} == __locked ]]; then
    log=$2
    drm=$3
    hw=$4
    shift 4
    if (($(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) < 8)); then
        echo "NOT STARTED: below 8 GB available memory after the lock wait" >>"$log"
        exit 5
    fi
    if [[ $(lact_profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)' after the lock wait, not Testing" >>"$log"
        exit 6
    fi
    echo "clocks before: $(snap "$drm" "$hw")" >>"$log"
    nice -n 10 timeout --signal=TERM --kill-after=20 "${JOB_TIMEOUT:-120}" "$@" >>"$log" 2>&1
    rc=$?
    ((rc == 124 || rc == 137)) && echo "TIMEOUT: job stopped after ${JOB_TIMEOUT:-120} s" >>"$log"
    echo "clocks after: $(snap "$drm" "$hw")" >>"$log"
    exit "$rc"
fi

die() {
    echo "study-leftovers-c-a: $*" >&2
    exit 2
}

CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
[[ -f $CARD_ENV ]] || die "no card record at $CARD_ENV; run scripts/first-hour.sh info first"
# shellcheck source=/dev/null
source "$CARD_ENV"
[[ ${ARCH:-} == gfx1201 ]] || die "card record says ARCH=${ARCH:-unset}; these probes are for gfx1201"
[[ -d ${HWMON:-} && -d ${DRM_DEVICE:-} ]] || die "card record lacks HWMON or DRM_DEVICE"
export DRM_DEVICE HWMON

RUN_DIR="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-leftc-a"
mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"
SUMMARY=$RUN_DIR/summary.md
OUT=$REPO/build/leftc-a
BIN=$OUT/bin
FAILED=0

sum() {
    echo "$*" >>"$SUMMARY"
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

card_config() {
    echo "lact_profile=$(lact_profile) power1_cap=$(cat "$HWMON/power1_cap") $(tr -s ' \n' ' ' <"$DRM_DEVICE/pp_od_clk_voltage")"
}

# binary | source | extra hipcc flags | listing check: asm:<header> (leftc_asmcheck.py), or region:<name> (a
# leftc_regioncheck.py call in region_check below), or none
PROBES=(
    "left_core|left_core.hip|-mcumode|asm:left_core_blocks.h"
    "left_core64|left_core.hip|-mcumode -mwavefrontsize64 -DW64=1|asm:left_core_blocks.h"
    "left_core_wgp|left_core.hip|-DWGP_BUILD=1|asm:left_core_blocks.h"
    "left_mem|left_mem.hip|-mcumode|region:left_mem"
    "left_vopd|left_vopd.hip||region:left_vopd"
    "left_spin|left_spin.hip||region:left_spin"
    "left_lds|left_lds.hip||region:left_lds"
    "left_comm|left_comm.hip|-mcumode|none"
    "left_comm_wgp|left_comm.hip||none"
    "left_cvt|left_cvt.hip||region:left_cvt"
    "left_dotm|left_dotm.hip||none"
    "left_hint|left_hint.hip||region:left_hint"
    "haz_issue_place|haz_issue.hip|-DHAZ_PLACEMENT -DHAZ_VARIANT='\"default\"'|region:haz_issue_place"
    "left_misal|left_misal.hip||region:left_misal"
    "left_haz|left_haz.hip||region:left_haz"
    "left_bank|left_bank.hip|-mcumode|asm:left_core_blocks.h"
    "left_bank64|left_bank.hip|-mcumode -mwavefrontsize64 -DW64=1|asm:left_core_blocks.h"
    "left_w64fma|left_w64fma.hip|-mcumode -DCUMODE=1 -mwavefrontsize64 -DW64=1|region:left_w64fma"
    "left_hazard_O0|left_hazard.hip|-O0|region:left_hazard_O0"
    "left_hazard_O3|left_hazard.hip|-O3|region:left_hazard_O3"
)
# CH-5b: the hazard and mask probes of compiler-and-hazards at other optimisation levels (-O3 the control); the
# later -O flag overrides build_one's -O3
# (haz_hazard does not build at -O0: "illegal VGPR to SGPR copy" in s_hazard_wait's inline assembly, then the
# linker aborts: its "+s" operand is a per-lane stack value at -O0; left out here. src/study/left_hazard.hip is the
# copy that builds at -O0, test T5)
for lvl in O0 O1 O2 Os O3; do
    for p in haz_mask haz_hazard haz_pitfalls haz_memshape; do
        [[ $p == haz_hazard && $lvl == O0 ]] && continue
        PROBES+=("${p}_$lvl|$p.hip|-$lvl|none")
    done
done

# region_check <name> <disassembly>: the compiler-built timed regions of one probe hold only the timed instructions,
# waits and counter reads
region_check() {
    local rc=$REPO/scripts/leftc_regioncheck.py
    case $1 in
        left_mem)
            python3 "$rc" "$2" 'k_ns' '^global_load_b(32|128)$' \
                '^s_getreg_b32$|^s_wait_loadcnt$|^s_wait_alu$|^s_delay_alu$|^s_cmp_eq_u32$|^s_cselect_b32$' --min 5
            ;;
        left_vopd)
            # the seven pairs as written: each forbidden or allowed pair once per loop iteration, in the text order
            local want="v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23|v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v23, v10|v_dual_add_f32 v10, v21, v22 :: v_dual_mul_f32 v11, v10, v23|v_dual_fmac_f32 v10, v21, v22 :: v_dual_fmac_f32 v11, v10, v23|v_dual_mov_b32 v10, v21 :: v_dual_mov_b32 v11, v10|v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23|v_dual_add_f32 v10, v21, v22 :: v_dual_mov_b32 v21, v23"
            local got
            got=$(grep -o 'v_dual_[^/]*' "$2" | sed 's/ *$//' | paste -sd'|')
            echo "VOPD pairs in the object: $got"
            if [[ $got == "$want" ]]; then
                echo "regioncheck: the 7 VOPD pairs are in the object as written, in order"
            else
                echo "regioncheck: VOPD pairs differ from the source" && return 1
            fi
            ;;
        left_spin)
            # each step: 16 blocks of 16 FMAs and the counter reads; R adds the REALTIME message. The loop bodies are
            # compiler-built; check that the FMAs are all there (256 per step body) in every k_spin kernel
            # and that no spin kernel reads memory after its start (kernel arguments: s_load at offsets below 0x40)
            awk '/^[0-9a-f]+ <.*k_spin.*>:/ {k=$2; n[k]=0; l[k]=0} /v_fma_f32/ && k {n[k]++}
                 /(global|buffer|scratch)_load|s_load/ && k && !/s\[0:1\]|s\[4:5\], 0x0/ {l[k]++}
                 END {for (x in n) print x, n[x], l[x]}' "$2" |
                awk '{print} $2 % 256 != 0 || $2 == 0 || $3 != 0 {bad=1} END {print "regioncheck: " NR " spin kernels, FMA count a multiple of 256 and no memory read in the loop: " (bad ? "NO" : "yes"); exit bad}'
            ;;
        left_lds)
            # every timed block as the source's LDS_OPS list writes it (expectation file), plus the final wait; and
            # each k_infl<N> holding 6 x N ds_load_b128 (six repetitions, unrolled)
            python3 "$REPO/scripts/leftc_asmcheck.py" "$2" "$REPO/scripts/leftc-expect/left_lds.txt" --extra s_wait_dscnt ||
                return 1
            # every kernel must keep its 64 KiB of LDS (a kernel whose LDS the compiler drops reads out of range: zeros,
            # at no bank cost)
            local s=("${2%/dis.txt}"/*.img.lto.s)
            awk '/\.amdhsa_kernel _ZN12_GLOBAL__N_1/ {k=$2} /\.amdhsa_kernel _ZN3mem/ {k=""}
                 /\.amdhsa_group_segment_fixed_size/ && k != "" {print k, $2}' "${s[@]}" |
                awk '{n++} $2 < 65536 {bad++; print "LDS " $2 " bytes in " $1}
                     END {print "regioncheck: " n " kernels, " bad + 0 " with less than 64 KiB of LDS"; exit bad > 0}' || return 1
            awk '/^[0-9a-f]+ <.*k_inflILi[0-9]+E.*>:/ {match($2, /ILi[0-9]+ELb[01]E/); k=substr($2, RSTART+3, RLENGTH-3); n[k]=0}
                 /ds_load_b128 v\[16:19\], v1|ds_load_b32 v16, v1/ && k != "" {n[k]++} END {for (x in n) print x, n[x]}' "$2" |
                awk '{print "k_infl N=" $1 ": " $2 " loads"} $2 != 6 * ($1 + 0) {bad=1}
                     END {print "regioncheck: " NR " k_infl kernels, 6 x N loads in each: " (bad ? "NO" : "yes"); exit bad}'
            ;;
        left_cvt)
            # the timed loops are compiler-built around one asm block each: core_loopcheck.py (lane 1's check) on the
            # listing; and the generated header must match its generator
            python3 "$REPO/scripts/cvt_gen_timing.py" | cmp -s - "$REPO/src/study/cvt_time_blocks.h" ||
                { echo "cvt_time_blocks.h differs from scripts/cvt_gen_timing.py" && return 1; }
            python3 "$REPO/scripts/leftc_gen_cd2.py" | cmp -s - "$REPO/src/study/left_cd2_blocks.h" ||
                { echo "left_cd2_blocks.h differs from scripts/leftc_gen_cd2.py" && return 1; }
            python3 "$REPO/scripts/core_loopcheck.py" "${2%/dis.txt}"/*.lto.s --allow '^s_wait_alu$' | tail -1
            ;;
        left_w64fma) python3 "$REPO/scripts/leftc_bankcheck.py" "${2%/dis.txt}"/*.img.lto.s ;;
        left_hazard_O3)
            # the control: the same hazard-scan expectations as haz_hazard at -O3 (compiler-and-hazards item 2)
            python3 "$REPO/scripts/haz_hazscan.py" "${2%/dis.txt}"/*.img.lto.s "$REPO/scripts/haz-expect/hazscan-haz_hazard.txt" |
                tail -14 && echo "regioncheck: hazard scan of the -O3 build matches haz_hazard's expectations"
            ;;
        left_hazard_O0)
            # INFO: the -O0 listing against the -O3 expectations (printed; the card run judges the results)
            python3 "$REPO/scripts/haz_hazscan.py" "${2%/dis.txt}"/*.img.lto.s "$REPO/scripts/haz-expect/hazscan-haz_hazard.txt" |
                grep "^EXPECT"
            echo "regioncheck: -O0 hazard scan printed above (INFO; the -O3 expectations count -O3 code)"
            ;;
        left_hint) python3 "$REPO/scripts/leftc_hintcheck.py" "$2" ;;
        left_misal)
            # the wide load is emitted in every kernel, never split: k_copy 1, k_bw 4, k_cyc 64 (a loop of two passes)
            awk '/^[0-9a-f]+ </ {k=""} /^[0-9a-f]+ <.*k_(copy|bw|cyc)/ {k=$2; n[k]=0; s[k]=0} /global_load_b128/ && k {n[k]++}
                 /global_load_b(32|64|96) / && k {s[k]++} END {for (x in n) print x, n[x], s[x]}' "$2" |
                awk '{print} ($1 ~ /copy/ && $2 != 1) || ($1 ~ /bw/ && $2 != 4) || ($1 ~ /cyc/ && $2 != 64) || $3 != 0 {bad=1}
                     END {print "regioncheck: " NR " kernels, wide loads as written and none split: " (bad ? "NO" : "yes"); exit bad}'
            ;;
        left_haz)
            # the sequences are inline assembly: check that the cmpx kernels hold v_cmpx directly followed by the
            # permute (no compiler instruction between), and the gap variants their v_mov or s_nop
            awk '/^[0-9a-f]+ </ {k=$2} /v_cmpx_gt_u32/ {c=k; getline n; sub(/\/\/.*/, "", n); gsub(/^[ \t]+|[ \t]+$/, "", n); split(n, w, " "); print c, w[1]}' "$2" |
                awk '{print} ($1 ~ /_none/ && $2 !~ /permlane/) || ($1 ~ /_mov/ && $2 !~ /^v_mov_b32/) || ($1 ~ /_nop/ && $2 != "s_nop") {bad=1}
                     END {print "regioncheck: " NR " cmpx kernels, the instruction after v_cmpx as written: " (bad ? "NO" : "yes"); exit bad || NR != 12}'
            ;;
        haz_issue_place)
            # the placement build keeps the timed loops of the default build (its expectation file)
            python3 "$REPO/scripts/mem_loopcheck.py" "${2%/dis.txt}"/*.img.lto.s "$REPO/scripts/haz-expect/haz_issue-default.txt" |
                tail -1 | sed 's/^/regioncheck: /'
            ;;
        *) echo "no region check for $1" && return 1 ;;
    esac
}

# program | arguments | time limit in s [| expect-fail] (one job each; the first word of the arguments is the mode for
# selection). "expect-fail" marks a job whose FAIL verdict is a finding recorded in the write-up: it must keep
# failing, and passing counts as a failure of the run (the finding changed). left_hazard_O0 all: at -O0 the call-gap
# miss of -O3 does not occur (hazard_call_miss FAILs) and the v_cmpx/permute test loses the switched-off partners'
# values (hazard_cmpx_permlane FAILs); measurements/leftovers-c-a.md, CH-5b, T5.
JOBS=(
    "left_core|w64|120"
    "left_core64|w64|120"
    "left_mem|ns|120"
    "left_core|insn|120"
    "left_vopd|vopd|60"
    "left_spin|spin|120"
    "left_lds|banks|120"
    "left_lds|inflight2|60"
    "left_lds|misalign|60"
    "left_lds|atomic|120"
    "left_lds|swz|60"
    "left_mem|tlb|120"
    "left_mem|multi|120"
    "left_comm|lds|60"
    "left_comm_wgp|lds|60"
    "left_comm|global|60"
    "left_core|bank|120"
    "left_core|run12|120"
    "left_core|sgprw|120"
    "left_core|fsize|120"
    "left_core_wgp|fsize|120"
    "left_cvt|cvt|120"
    "left_dotm|dotm|120"
    "left_hint|hint|120"
    "leftc_lastuse|lastuse|60"
    "haz_issue_place|place|60"
    "left_misal|misal|120"
    "left_misal|misal bw 0|30"
    "left_misal|misal bw 0 b2|30"
    "left_misal|misal bw 0 b3|30"
    "left_misal|misal bw 4|30"
    "left_misal|misal bw 4 b2|30"
    "left_misal|misal bw 4 b3|30"
    "left_misal|misal bw 8|30"
    "left_misal|misal bw 8 b2|30"
    "left_misal|misal bw 8 b3|30"
    "left_misal|misal bw 12|30"
    "left_misal|misal bw 12 b2|30"
    "left_misal|misal bw 12 b3|30"
    "left_haz|cmpx|60"
    "left_haz|sgpr|60"
    "left_w64fma|w64fma|60"
    "left_bank|bank|120"
    "left_bank64|bank|120"
    "left_bank|insdata|120"
    "left_cvt|cd2|120"
    "left_cvt|cd2 b2|120"
    "left_cvt|cd2 b3|120"
    "left_hazard_O0|all|60|expect-fail"
    "left_hazard_O3|all|60"
    "leftc_olevel|olevel|120"
    "left_mem|l0card 0|30"
    "left_mem|l0card 1|30"
    "left_mem|l0card 1 b2|30"
    "left_mem|l0card 1 b3|30"
    "left_mem|l0card 2|30"
    "left_mem|l0card 2 b2|30"
    "left_mem|l0card 2 b3|30"
    "left_mem|l0card 4|30"
    "left_mem|l0card 4 b2|30"
    "left_mem|l0card 4 b3|30"
    "left_mem|l0card 8|30"
    "left_mem|l0card 8 b2|30"
    "left_mem|l0card 8 b3|30"
    "left_mem|l0card 16|30"
    "left_mem|l0card 16 b2|30"
    "left_mem|l0card 16 b3|30"
)
BREAKS=(
    "left_core|w64 --break|120"
    "left_core64|w64 --break|120"
    "left_mem|ns --break|120"
    "left_core|insn --break|120"
    "left_vopd|vopd --break|60"
    "left_spin|spin --break|120"
    "left_lds|banks --break|120"
    "left_lds|inflight2 --break|60"
    "left_lds|misalign --break|60"
    "left_lds|atomic --break|120"
    "left_lds|swz --break|60"
    "left_mem|tlb --break|60"
    "left_mem|multi --break|60"
    "left_comm|lds --break|60"
    "left_comm|global --break|60"
    "left_core|bank --break|120"
    "left_core|run12 --break|120"
    "left_core|sgprw --break|120"
    "left_core|fsize --break|120"
    "left_cvt|cvt --break|120"
    "left_dotm|dotm --break|120"
    "left_mem|l0card 16 --break|30"
    "left_hint|hint --break|120"
    "leftc_lastuse|lastuse --break|60"
    "haz_issue_place|place --break|60"
    "left_misal|misal --break|60"
    "left_haz|cmpx --break|60"
    "left_haz|sgpr --break|60"
    "left_w64fma|w64fma --break|60"
    "left_bank|bank --break|120"
    "left_bank64|bank --break|120"
    "left_bank|insdata --break|120"
    "left_cvt|cd2 --break|120"
    "left_hazard_O0|break|60"
    "left_hazard_O3|break|60"
)
for lvl in O0 O1 O2 Os O3; do
    if [[ $lvl == O0 ]]; then
        # CH-5b finding: at -O0 the pinned permute source of pitfall (f) is not pinned (pit_perm_pinned FAILs)
        JOBS+=("haz_mask_$lvl|all|60" "haz_pitfalls_$lvl|all|60|expect-fail")
    else
        JOBS+=("haz_mask_$lvl|all|60" "haz_pitfalls_$lvl|all|60")
    fi
    [[ $lvl == O0 ]] && continue
    JOBS+=("haz_hazard_$lvl|all|60")
    BREAKS+=("haz_hazard_$lvl|break|60")
done

build_one() {
    local name=$1 src=$2 flags=$3 dir=$OUT/tmp-$1
    mkdir -p "$dir" "$BIN"
    # shellcheck disable=SC2086 # flags is a word list on purpose
    (cd "$dir" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1201 $flags -save-temps \
        -DHARNESS_ARCH='"gfx1201"' -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/study/$src" -o "$BIN/$name" \
        -lamdhip64) >"$RUN_DIR/build-$name.log" 2>&1
}

phase_build() {
    local p name src flags check pids=() rc
    sum "## build ($(hipcc --version 2>/dev/null | head -1 | tr -d '\n'))"
    mkdir -p "$BIN"
    if python3 "$REPO/scripts/leftc_gen_core.py" "$OUT/left_core_blocks.h" >/dev/null &&
        cmp -s "$OUT/left_core_blocks.h" "$REPO/src/study/left_core_blocks.h"; then
        sum "- left_core_blocks.h: matches scripts/leftc_gen_core.py"
    else
        sum "- left_core_blocks.h: does NOT match scripts/leftc_gen_core.py"
        FAILED=1
    fi
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags check <<<"$p"
        build_one "$name" "$src" "$flags" &
        pids+=($!)
        # at most 4 compilers at once (machine rule: builds use at most 8 cores; hipcc runs two jobs per file)
        if ((${#pids[@]} >= 4)); then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    wait
    for p in "${PROBES[@]}"; do
        IFS='|' read -r name src flags check <<<"$p"
        if [[ ! -x $BIN/$name ]] || grep -q "error:" "$RUN_DIR/build-$name.log"; then
            sum "- $name: build FAILED (see build-$name.log)"
            FAILED=1
            continue
        fi
        "$ROCM_PATH/llvm/bin/llvm-objdump" -d --mcpu=gfx1201 "$OUT/tmp-$name"/*.img.lto.o >"$OUT/tmp-$name/dis.txt" 2>&1
        case $check in
            asm:*)
                python3 "$REPO/scripts/leftc_asmcheck.py" "$OUT/tmp-$name/dis.txt" "$REPO/src/study/${check#asm:}" \
                    >"$RUN_DIR/asmcheck-$name.txt" 2>&1
                rc=$?
                if ((rc == 0)); then
                    sum "- $name: built${flags:+ ($flags)}; $(tail -1 "$RUN_DIR/asmcheck-$name.txt")"
                else
                    sum "- $name: built; LISTING CHECK FAILED (see asmcheck-$name.txt)"
                    FAILED=1
                fi
                ;;
            region:*)
                region_check "${check#region:}" "$OUT/tmp-$name/dis.txt" >"$RUN_DIR/regioncheck-$name.txt" 2>&1
                rc=$?
                if ((rc == 0)); then
                    sum "- $name: built${flags:+ ($flags)}; $(tail -1 "$RUN_DIR/regioncheck-$name.txt")"
                else
                    sum "- $name: built; LISTING CHECK FAILED (see regioncheck-$name.txt)"
                    FAILED=1
                fi
                ;;
            *) sum "- $name: built${flags:+ ($flags)} (listing: $OUT/tmp-$name/dis.txt)" ;;
        esac
    done
}

# job <log> <time limit> <command...>: a GPU job under the shared lock
job() {
    local log=$1 limit=$2
    shift 2
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 5
    fi
    if [[ ${HIP_INDEX:-unset} != 0 ]]; then
        echo "NOT STARTED: device index is ${HIP_INDEX:-unset}; this study only runs on device 0" >>"$log"
        return 8
    fi
    echo "card configuration: $(card_config)" >>"$log"
    # whole-card jobs (l0card, misal): wait outside the GPU lock until the hotspot is below 55 C (at most 120 s);
    # a job whose wait timed out is not started
    if [[ $* == *" l0card "* || $* == *" misal"* ]]; then
        local waited=0
        while (($(cat "$HWMON/temp2_input") >= 55000 && waited < 240)); do
            sleep 0.5
            waited=$((waited + 1))
        done
        echo "cool-down before the lock: $((waited / 2)) s, hotspot $(($(cat "$HWMON/temp2_input") / 1000)) C" >>"$log"
        if (($(cat "$HWMON/temp2_input") >= 55000)); then
            echo "NOT STARTED: hotspot still at 55 C or more after 120 s of cool-down" >>"$log"
            return 7
        fi
    fi
    if [[ $(lact_profile) != Testing ]]; then
        echo "NOT STARTED: LACT profile is '$(lact_profile)', not Testing" >>"$log"
        return 6
    fi
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH LEFTC_RUN=$RUN_DIR JOB_TIMEOUT=$limit \
        flock "$GPU_LOCK" "$SELF" __locked "$log" "$DRM_DEVICE" "$HWMON" "$@"
}

log_name() {
    local args="$1-$2"
    args=${args// /_}
    args=${args//--/}
    args=${args//=/}
    echo "$RUN_DIR/${args:0:80}.log"
}

selected() {
    local m=$1 s
    ((${#ONLY[@]} == 0)) && return 0
    for s in "${ONLY[@]}"; do
        [[ $s == "${m%% *}" ]] && return 0
    done
    return 1
}

# CH-5b: the listing scans of compiler-and-hazards on every level, against the -O3 expectation files. The -O3
# control must pass them all (verdict olevel_control); the other levels' results are INFO (the expectation files count
# -O3 code).
olevel_scans() {
    local E=$REPO/scripts/haz-expect lvl l1 l2 l3 l4 r1 r2 r3 r4 ctl=1
    for lvl in O0 O1 O2 Os O3; do
        l1=$(find "$OUT/tmp-haz_mask_$lvl" -name "*.img.lto.s" 2>/dev/null | head -1)
        l2=$(find "$OUT/tmp-haz_hazard_$lvl" -name "*.img.lto.s" 2>/dev/null | head -1)
        [[ -n $l2 ]] || l2=/dev/null
        l3=$(find "$OUT/tmp-haz_pitfalls_$lvl" -name "*.img.lto.s" 2>/dev/null | head -1)
        l4=$(find "$OUT/tmp-haz_memshape_$lvl" -name "*.img.lto.s" 2>/dev/null | head -1)
        python3 "$REPO/scripts/haz_maskscan.py" "$l1" "$E/maskscan-haz_mask.txt" >"$RUN_DIR/olevel-mask-$lvl.txt" 2>&1
        r1=$?
        python3 "$REPO/scripts/haz_hazscan.py" "$l2" "$E/hazscan-haz_hazard.txt" >"$RUN_DIR/olevel-hazard-$lvl.txt" 2>&1
        r2=$?
        python3 "$REPO/scripts/mem_loopcheck.py" "$l3" "$E/haz_pitfalls.txt" >"$RUN_DIR/olevel-pitfalls-$lvl.txt" 2>&1
        r3=$?
        python3 "$REPO/scripts/mem_loopcheck.py" "$l4" "$E/haz_memshape.txt" >"$RUN_DIR/olevel-memshape-$lvl.txt" 2>&1
        r4=$?
        echo "RESULT olevel -$lvl: mask scan $( ((r1 == 0)) && echo matches || echo differs) ($(grep -c '^MASKED\|^COUNT' "$RUN_DIR/olevel-mask-$lvl.txt") lines)," \
            "hazard scan $( ((r2 == 0)) && echo matches || echo differs) ($(grep -c '^HAZARD' "$RUN_DIR/olevel-hazard-$lvl.txt") HAZARD sites)," \
            "pitfall loops $( ((r3 == 0)) && echo match || echo differ), memshape loads $( ((r4 == 0)) && echo match || echo differ)"
        grep -h "FAIL\|HAZARD\|CALL\|RETURN" "$RUN_DIR/olevel-hazard-$lvl.txt" | head -12 | sed "s/^/INFO -$lvl hazard: /"
        if [[ $lvl == O3 ]] && ((r1 || r2 || r3 || r4)); then
            ctl=0
        fi
    done
    if ((ctl)); then
        echo "VERDICT olevel_control: PASS the -O3 control builds match every expectation file of compiler-and-hazards"
    else
        echo "VERDICT olevel_control: FAIL an -O3 control build does not match its expectation file"
        return 1
    fi
}

run_one() {
    local prog=$1 args=$2 limit=$3 log=$4 exe=$BIN/$1
    # CPU-only checks: no GPU lock
    if [[ $prog == leftc_olevel ]]; then
        echo "## $(date -u +%FT%TZ) $prog (CPU only)" >>"$log"
        olevel_scans >>"$log" 2>&1
        return
    fi
    if [[ $prog == leftc_lastuse ]]; then
        echo "## $(date -u +%FT%TZ) $prog $args (CPU only)" >>"$log"
        # shellcheck disable=SC2086 # args is a word list on purpose
        nice -n 10 timeout "$limit" python3 "$REPO/scripts/leftc_lastuse.py" "$REPO" ${args#lastuse} >>"$log" 2>&1
        return
    fi
    if [[ ! -x $exe ]]; then
        echo "NOT BUILT: $exe" >>"$log"
        return 9
    fi
    # shellcheck disable=SC2086 # args is a word list on purpose
    job "$log" "$limit" "$exe" $args
}

phase_run() {
    local j prog args limit flag log rc v
    sum "## run (device index $HIP_INDEX from $CARD_ENV; card configuration: $(card_config))"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r prog args limit flag <<<"$j"
        selected "$args" || selected "$prog" || continue
        log=$(log_name "$prog" "$args")
        run_one "$prog" "$args" "$limit" "$log"
        rc=$?
        v=$(grep -h "^VERDICT" "$log" | cut -c1-600 | tr '\n' ' ')
        if [[ ${flag:-} == expect-fail ]]; then
            if ((rc == 1)) && grep -q "^VERDICT [^ ]*: FAIL" "$log"; then
                sum "- $prog $args: FAILS as found (a finding, see the write-up). $(grep -h "^VERDICT [^ ]*: FAIL" "$log" | head -2 | cut -c1-300 | tr '\n' ' ')"
            else
                sum "- $prog $args: expected to FAIL as found, but exit $rc (the finding changed). ${v:-no verdict}"
                FAILED=1
            fi
        elif ((rc == 0)); then
            sum "- $prog $args: ok, $(grep -c '^RESULT' "$log") results. ${v}"
        else
            sum "- $prog $args: FAILED (exit $rc). ${v:-no verdict} (see $(basename "$log"))"
            FAILED=1
        fi
    done
}

phase_breaks() {
    local j prog args limit log rc v
    sum "## deliberate breaks (each must FAIL)"
    for j in "${BREAKS[@]}"; do
        IFS='|' read -r prog args limit <<<"$j"
        selected "$args" || selected "$prog" || continue
        log=$(log_name "$prog" "$args")
        run_one "$prog" "$args" "$limit" "$log"
        rc=$?
        v=$(grep -h "^VERDICT [^ ]*: FAIL" "$log" | cut -c1-400 | tr '\n' ' ')
        if ((rc == 1)) && [[ -n $v ]]; then
            sum "- $prog $args: FAIL as intended. $v"
        else
            sum "- $prog $args: did NOT fail (exit $rc). ${v:-no FAIL verdict}"
            FAILED=1
        fi
    done
}

PHASE=${1:-all}
shift || true
ONLY=("$@")
sum "# Leftovers C lane A run, $(date -u +%FT%TZ)"
sum "- commit $(git -C "$REPO" rev-parse --short HEAD)$([[ -z $(git -C "$REPO" status --porcelain -- src scripts) ]] || echo ' (with uncommitted changes)'); card $ARCH, PCI $PCI_BUS; kernel $(uname -r)"
sum "- LACT profile $(lact_profile)${ONLY[*]:+; modes: ${ONLY[*]}}"
case $PHASE in
    build) phase_build ;;
    run) phase_run ;;
    breaks) phase_breaks ;;
    all)
        phase_build
        phase_run
        phase_breaks
        ;;
    *) die "unknown phase $PHASE; use build, run, breaks or all" ;;
esac
echo "study-leftovers-c-a: results in $RUN_DIR"
cat "$SUMMARY"
exit "$FAILED"
