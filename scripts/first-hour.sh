#!/usr/bin/env bash
# First checks on a new card, the same way on any card: one command, one run folder, one summary.
#
# Usage: scripts/first-hour.sh <phase...|all>
#   info       read-only (sysfs, rocminfo, vulkaninfo, versions); writes the card record card.env. No GPU load.
#   build      compiles src/ for the card's arch into build/<arch>/ (ARCH=gfx1201 or gfx1100 overrides, for a card
#              that is not installed: the compile gate)
#   bandwidth  read and copy bandwidth, 3 rounds, buffers above the cache trap
#   probes     E4M3 reference test (CPU), matrix instruction timing and exactness, FP8 conversions (gfx1201)
#   decode     FP8 decode prototype: correctness modes and timing at 1, 4 and 8 columns
#   llama      llama.cpp unmodified and patched: kernel tests, perplexity (NaN check), llama-bench
#   libs       hipBLASLt FP8 GEMM and torch._scaled_mm (gfx1201 only)
#   all        every phase above, in that order
# Environment:
#   ARCH=gfx1100|gfx1201   build phase only: compile for this arch without reading the card
#   RUN=<run folder>       add phases to this run folder (default: a new one for info and all, else runs/latest)
#   MODELS="llama32-3b llama31-8b"   models for the llama phase (default llama32-3b)
#   DRY_RUN=1              write GPU commands to the logs instead of running them (tests the script itself)
# Results: runs/<UTC time>-<arch>/: card.env, one log per phase, summary.md. Exit status 1 if anything failed.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/scripts/first-hour.sh"
[[ -r $REPO/local.env ]] && . "$REPO/local.env"   # machine settings, git-ignored; see local.env.example
RDNA3=${RDNA3_DIR:-$HOME/Projects/rdna3-fp8}
LLAMA_UNMOD=$RDNA3/build/llama.cpp-latest
LLAMA_PATCHED=$RDNA3/build/llama.cpp-latest-patched
# llama.cpp is built once per ROCm; the build folder carries the toolchain's tag so results never mix two ROCm versions.
if [[ -z ${LLAMA_TAG+x} ]]; then
    case $(cat "${ROCM_PATH:-/nonexistent}/.info/version" 2>/dev/null) in
        10.*) LLAMA_TAG=rocm10 ;;
        *) LLAMA_TAG="" ;;
    esac
fi
MODEL_DIR=$RDNA3/models
PPL_TEXT=$RDNA3/build/ppl-text.txt
# PyTorch 2.15.0+rocm10.0 with its own bundled ROCm 10.0 (toolchains/ is not in git; package list in
# toolchains/torch-rocm10-freeze.txt). The earlier 2.13.0+rocm7.1 environment in rdna3-fp8/build was deleted.
TORCH_PY=${TORCH_PY:-$REPO/toolchains/torch-rocm10-venv/bin/python}
LOOP_ISA=$RDNA3/examples/tools/loop_isa.py
KFD=/sys/class/kfd/kfd/topology/nodes
MODELS=${MODELS:-llama32-3b}
DRY_RUN=${DRY_RUN:-0}

GPU_LOCK=${STUDY_GPU_LOCK:-$(sed -n 's/^GPU_LOCK=//p' "$RDNA3/build/local.env" 2>/dev/null | head -1)}
GPU_LOCK=${GPU_LOCK:-$REPO/runs/gpu.lock}

FAILED=0

die() {
    echo "first-hour: $*" >&2
    exit 2
}

avail_gb() {
    awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

# ---------------------------------------------------------------------------------------------------------------
# Locked runner: re-entered under flock, so clocks are read while this job holds the GPU.

if [[ ${1:-} == __locked ]]; then
    log=$2
    shift 2
    # shellcheck source=/dev/null
    source "$RUN_DIR/card.env"
    snap() {
        echo "sclk: $(grep '\*' "$DRM_DEVICE/pp_dpm_sclk" | tr -s ' ')  mclk: $(grep '\*' "$DRM_DEVICE/pp_dpm_mclk" | tr -s ' ')  power: $(cat "$HWMON/power1_average" 2>/dev/null) uW  edge: $(cat "$HWMON/temp1_input" 2>/dev/null) mC"
    }
    {
        echo "clocks before: $(snap)"
        echo "pp_dpm_sclk:"
        cat "$DRM_DEVICE/pp_dpm_sclk"
        echo "pp_dpm_mclk:"
        cat "$DRM_DEVICE/pp_dpm_mclk"
    } >>"$log"
    nice -n 10 "$@" >>"$log" 2>&1
    rc=$?
    {
        echo "clocks after: $(snap)"
        echo "pp_dpm_sclk:"
        cat "$DRM_DEVICE/pp_dpm_sclk"
        echo "pp_dpm_mclk:"
        cat "$DRM_DEVICE/pp_dpm_mclk"
    } >>"$log"
    exit "$rc"
fi

# ---------------------------------------------------------------------------------------------------------------
# Card detection, from the KFD topology only (no GPU load)

# Prints "<node> <arch>" for the discrete card, or stops. The integrated GPU (gfx_target_version 100306) and CPU
# nodes are ignored.
find_card() {
    local n v simd cand="" other=""
    for n in "$KFD"/*; do
        v=$(awk '$1 == "gfx_target_version" {print $2}' "$n/properties")
        simd=$(awk '$1 == "simd_count" {print $2}' "$n/properties")
        [[ ${simd:-0} -eq 0 || $v == 100306 ]] && continue
        case $v in
            110000) cand+="${n##*/} gfx1100;" ;;
            120001) cand+="${n##*/} gfx1201;" ;;
            *) other+=" node ${n##*/} gfx_target_version $v" ;;
        esac
    done
    if [[ -z $cand ]]; then
        die "no gfx1100 or gfx1201 card found in $KFD (other GPUs:${other:- none}); this script knows only those two"
    fi
    if [[ $cand == *";"*";"* ]]; then
        echo "first-hour: more than one candidate card ($cand); using the first" >&2
    fi
    echo "${cand%%;*}"
}

prop() {
    awk -v k="$2" '$1 == k {print $2}' "$KFD/$1/properties"
}

# Largest cache entry of a level, in KB (the driver lists one entry per cache instance).
cache_kb() {
    local best=0 c l s
    for c in "$KFD/$1"/caches/*; do
        l=$(awk '$1 == "level" {print $2}' "$c/properties")
        s=$(awk '$1 == "size" {print $2}' "$c/properties")
        [[ $l == "$2" && $s -gt $best ]] && best=$s
    done
    echo "$best"
}

write_card_env() {
    local node=$1 arch=$2 out=$3 loc pci minor drmdev card hwmon simd spc l2 l3 trap idx
    loc=$(prop "$node" location_id)
    pci=$(printf '%04x:%02x:%02x.%x' "$(prop "$node" domain)" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7)))
    minor=$(prop "$node" drm_render_minor)
    drmdev=$(readlink -f "/sys/class/drm/renderD$minor/device")
    card=""
    for c in /sys/class/drm/card[0-9]*; do
        [[ $c == *-* ]] && continue
        [[ $(readlink -f "$c/device") == "$drmdev" ]] && card=$c
    done
    hwmon=$(find "$drmdev/hwmon" -mindepth 1 -maxdepth 1 -name 'hwmon*' 2>/dev/null | head -1)
    simd=$(prop "$node" simd_count)
    spc=$(prop "$node" simd_per_cu)
    l2=$(cache_kb "$node" 2)
    l3=$(cache_kb "$node" 3)
    trap=$(((l3 > 0 ? l3 : l2) * 4 / 1024))
    # HIP numbers the GPU agents in ROCr's order, which rocminfo prints; checked against the HIP runtime itself
    # (src/hip_devices.cpp) before the first GPU job.
    idx=$(rocminfo 2>/dev/null | awk -v loc="$loc" '
        /^  Device Type:/ {gpu = ($3 == "GPU")}
        /^  BDFID:/ {if (gpu) {if ($2 == loc) {print i; exit} i++}}')
    cat >"$out" <<EOF
# Card record, written by scripts/first-hour.sh info on $(date -u +%FT%TZ). Read by every later phase.
ARCH=$arch
KFD_NODE=$node
PCI_BUS=$pci
DEVICE_ID=$(printf '0x%04x' "$(prop "$node" device_id)")
HIP_INDEX=${idx:-0}
HIP_INDEX_SOURCE=rocminfo-order
DRM_CARD=$card
DRM_DEVICE=$drmdev
HWMON=$hwmon
SIMD_COUNT=$simd
CU_COUNT=$((simd / spc))
WAVES_PER_SIMD=$(prop "$node" max_waves_per_simd)
L2_KB=$l2
L3_KB=$l3
MEM_WIDTH_BITS=$(awk '$1 == "width" {print $2}' "$KFD/$node"/mem_banks/0/properties)
MEM_CLK_MAX_MHZ=$(awk '$1 == "mem_clk_max" {print $2}' "$KFD/$node"/mem_banks/0/properties)
CACHE_TRAP_MB=$trap
EOF
}

# ---------------------------------------------------------------------------------------------------------------
# Run folder and summary

new_run_dir() {
    local arch=$1 d
    d="$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-$arch"
    mkdir -p "$d" || die "cannot create $d"
    ln -sfn "${d##*/}" "$REPO/runs/latest"
    echo "$d"
}

sum() {
    echo "$*" >>"$RUN_DIR/summary.md"
}

load_card() {
    [[ -f $RUN_DIR/card.env ]] || die "no card record in $RUN_DIR; run the info phase first"
    # shellcheck source=/dev/null
    source "$RUN_DIR/card.env"
    export RUN_DIR CARD_ARCH=$ARCH
}

# Confirms HIP_INDEX with the HIP runtime (device list only, no kernel) before the first GPU job of this call.
HIP_CHECKED=0
check_hip_index() {
    ((HIP_CHECKED)) && return 0
    local lister=$REPO/build/cpu/hip_devices line idx
    [[ -x $lister ]] || {
        sum "- HIP device index: hip_devices not built; using $HIP_INDEX from rocminfo order"
        HIP_CHECKED=1
        return 0
    }
    if ((DRY_RUN)); then
        HIP_CHECKED=1
        return 0
    fi
    line=$(env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES "$lister" | grep -i "pci $PCI_BUS ")
    idx=$(awk '{print $2}' <<<"$line")
    if [[ -z $idx ]]; then
        sum "- HIP device index: HIP runtime does not list $PCI_BUS; FAILED"
        return 1
    fi
    if [[ $idx != "$HIP_INDEX" ]]; then
        sum "- HIP device index: rocminfo order gave $HIP_INDEX, HIP runtime says $idx; using $idx"
        HIP_INDEX=$idx
        sed -i "s/^HIP_INDEX=.*/HIP_INDEX=$idx/" "$RUN_DIR/card.env"
    fi
    local note
    note=$(printf '%s' "$line" | tr -d "'" | sed 's/[&|\\]/\\&/g')
    sed -i "s|^HIP_INDEX_SOURCE=.*|HIP_INDEX_SOURCE='hip-runtime ($note)'|" "$RUN_DIR/card.env"
    bash -n "$RUN_DIR/card.env" || { sum "- card.env is no longer valid shell after the device check; FAILED"; return 1; }
    HIP_CHECKED=1
    return 0
}

# Child processes of a PID, recursively (for stopping a job we started).
descendants() {
    local c
    for c in $(cat /proc/"$1"/task/*/children 2>/dev/null); do
        descendants "$c"
        echo "$c"
    done
}

# gpu_job <log> <command...>: one GPU job, under the shared lock, nice, with the device index of the card record.
# Not started below 8 GB available memory; stopped if available memory falls below 3 GB while it runs.
gpu_job() {
    local log=$1 pid wpid rc
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    if ((DRY_RUN)); then
        echo "DRY_RUN: not run" >>"$log"
        return 4
    fi
    check_hip_index || return 1
    if (($(avail_gb) < 8)); then
        echo "NOT STARTED: $(avail_gb) GB available, below 8 GB" >>"$log"
        return 1
    fi
    HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=$ARCH flock "$GPU_LOCK" "$SELF" __locked "$log" "$@" &
    pid=$!
    (
        while kill -0 "$pid" 2>/dev/null; do
            if (($(avail_gb) < 3)); then
                echo "STOPPED: available memory fell below 3 GB" >>"$log"
                # shellcheck disable=SC2046
                kill -TERM $(descendants "$pid") "$pid" 2>/dev/null
                exit 0
            fi
            sleep 2
        done
    ) &
    wpid=$!
    wait "$pid"
    rc=$?
    kill "$wpid" 2>/dev/null
    wait "$wpid" 2>/dev/null
    echo "exit status $rc" >>"$log"
    return "$rc"
}

# cpu_job <log> <command...>: a CPU-only step (no lock), nice.
cpu_job() {
    local log=$1
    shift
    echo "## $(date -u +%FT%TZ) $*" >>"$log"
    nice -n 10 "$@" >>"$log" 2>&1
    local rc=$?
    echo "exit status $rc" >>"$log"
    return "$rc"
}

# Records a step's outcome; a failed step marks the phase failed but the phase goes on.
PHASE_FAILED=0
step() {
    local rc=$1 what=$2 log=$3
    if ((rc == 0)); then
        sum "- $what: ok"
    elif ((rc == 4)); then
        sum "- $what: not run (DRY_RUN)"
    elif ((rc == 3)); then
        sum "- $what: FAILED, program refused the card (see $log)"
        PHASE_FAILED=1
    else
        sum "- $what: FAILED (exit $rc, see $log)"
        PHASE_FAILED=1
    fi
}

last_verdict() {
    grep -h "^VERDICT $1:" "$2" 2>/dev/null | tail -1 | sed "s/^VERDICT $1: //"
}

median_spread() {
    sort -n | awk '{v[NR] = $1} END {if (NR == 0) {print "no result"; exit} printf "%.1f (min %.1f, max %.1f, n %d)", v[int((NR + 1) / 2)], v[1], v[NR], NR}'
}

# ---------------------------------------------------------------------------------------------------------------
# Phases

phase_info() {
    local log=$1 d n
    d=$DRM_DEVICE
    n=$KFD_NODE
    {
        echo "# info, $(date -u +%FT%TZ); read-only, no GPU load"
        echo "## card record"
        cat "$RUN_DIR/card.env"
        echo "## KFD node $n properties"
        cat "$KFD/$n/properties"
        echo "## caches: count by (level, size KB, line size, type)"
        for c in "$KFD/$n"/caches/*; do
            awk '$1 == "level" || $1 == "size" || $1 == "cache_line_size" || $1 == "type" {printf "%s %s ", $1, $2} END {print ""}' "$c/properties"
        done | sort | uniq -c
        echo "## mem_banks"
        for b in "$KFD/$n"/mem_banks/*; do
            echo "${b##*/}: $(tr '\n' ' ' <"$b/properties")"
        done
        echo "## clocks and power ($d)"
        for f in pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_dpm_socclk pp_dpm_pcie power_dpm_force_performance_level pp_power_profile_mode; do
            echo "### $f"
            cat "$d/$f" 2>/dev/null || echo "(not present)"
        done
        echo "### hwmon ($HWMON)"
        for f in power1_cap power1_cap_default power1_cap_max power1_average freq1_input freq2_input temp1_input temp2_input temp3_input; do
            echo "$f $(cat "$HWMON/$f" 2>/dev/null || echo -)"
        done
        echo "## firmware versions reported by the driver"
        for f in "$d"/fw_version/*; do
            echo "${f##*/} $(cat "$f" 2>/dev/null)"
        done
        echo "vbios_version $(cat "$d/vbios_version" 2>/dev/null)"
        echo "## versions"
        echo "kernel $(uname -r)"
        echo "hipcc $(command -v hipcc)"
        echo "ROCM_PATH ${ROCM_PATH:-unset} (version file: $(cat "${ROCM_PATH:-/nonexistent}/.info/version" 2>/dev/null || echo none))"
        hipcc --version 2>/dev/null | head -3
        rpm -q rocm-core hipblaslt rocblas hipblas rocminfo 2>/dev/null
        echo "## rocminfo (agents)"
        rocminfo 2>/dev/null | grep -E '^  (Name|Marketing Name|Device Type|BDFID|Compute Unit|SIMDs per CU|Max Clock Freq|Wavefront Size|Node):'
    } >"$log" 2>&1
    sum "- card: $ARCH, KFD node $n, PCI $PCI_BUS, device $DEVICE_ID, HIP index $HIP_INDEX (from rocminfo order; checked against the HIP runtime before the first GPU job)"
    sum "- $SIMD_COUNT SIMDs = $CU_COUNT CUs, $WAVES_PER_SIMD waves per SIMD, L2 $L2_KB KB, L3 $L3_KB KB, memory $MEM_WIDTH_BITS bits at up to $MEM_CLK_MAX_MHZ MHz; cache trap: keep buffers above $CACHE_TRAP_MB MB"
    sum "- performance level $(cat "$d/power_dpm_force_performance_level" 2>/dev/null), profile $(grep -o '[A-Z_0-9]*\*' "$d/pp_power_profile_mode" 2>/dev/null | tr -d '*' | head -1), power cap $(($(cat "$HWMON/power1_cap" 2>/dev/null || echo 0) / 1000000)) W, top mclk $(tail -1 "$d/pp_dpm_mclk" | tr -s ' ')"
    sum "- kernel $(uname -r); toolchain: $(hipcc --version 2>/dev/null | head -1) at $(command -v hipcc); llama.cpp builds: build-$ARCH${LLAMA_TAG:+-$LLAMA_TAG}"
    if command -v vulkaninfo >/dev/null; then
        local vlog=$RUN_DIR/vulkaninfo.txt sec
        timeout 120 vulkaninfo >"$vlog" 2>/dev/null
        # the GPU section whose deviceID matches the card
        sec=$(awk -v id="$DEVICE_ID" '/^GPU[0-9]+:/ {g = $1} $1 == "deviceID" && tolower($3) == tolower(id) && !s {s = g} END {print s}' "$vlog")
        {
            echo "## vulkaninfo, section $sec"
            awk -v s="$sec" '/^GPU[0-9]+:/ {on = ($1 == s)} on' "$vlog" | grep -E 'deviceName|driverInfo|apiVersion|cooperative_matrix|shader_float8|shader_bfloat16|CooperativeMatrix|Float8|BFloat16'
            echo "(vulkaninfo does not print the cooperative-matrix type and size list; a Vulkan program is needed for that)"
        } >>"$log"
        sum "- Vulkan (${sec%:}): $(awk -v s="$sec" '/^GPU[0-9]+:/ {on = ($1 == s)} on' "$vlog" | grep -E 'driverInfo|shaderFloat8CooperativeMatrix|shaderBFloat16CooperativeMatrix' | tr -s ' \t' ' ' | sort -u | tr '\n' ';')"
    else
        sum "- vulkaninfo not installed"
    fi
    return 0
}

# Compiles one program: build_one <arch> <name> <source> [extra flags...]; assembly goes to build/<arch>/asm.
# Prints the device assembly listing of a program built by build_one. ROCm 10 (clang 24) writes it as
# <name>.hip-...img.lto.s; older toolchains as <name>-hip-...-<arch>.s. build_one removes old listings first, so
# whichever exists belongs to the build just made.
device_asm() {
    local arch=$1 name=$2 f
    for f in "$REPO/build/$arch/asm/$name.hip-hip-amdgcn-amd-amdhsa.hipfb.amdgcn.$arch.img.lto.s" \
        "$REPO/build/$arch/asm/$name-hip-amdgcn-amd-amdhsa-$arch.s"; do
        if [[ -f $f ]]; then
            echo "$f"
            return 0
        fi
    done
    return 1
}

build_one() {
    local arch=$1 name=$2 src=$3 log=$4 out
    shift 4
    out=$REPO/build/$arch
    rm -f "$out/asm/$name"-hip-* "$out/asm/$name".hip-* "$out/asm/$name"-host-*
    echo "## $(date -u +%FT%TZ) $name" >>"$log"
    (cd "$out/asm" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch="$arch" -DHARNESS_ARCH="\"$arch\"" -save-temps \
        "$REPO/src/$src" -o "$out/$name" "$@" -lamdhip64) >>"$log" 2>&1
}

# Every chain kernel's loop must hold 16 matrix instructions per accumulator and 16 FMAs per FMA-per-iteration
# (the loop is unrolled 16 times), and the cycle counter must be read through SHADER_CYCLES. On gfx1100 the
# gfx12-only kinds (bf16, fp8, iu8) are compiled empty and never run, so they are not checked.
check_wmma_time_asm() {
    local arch=$1 log=$2 s bad=0 total=0
    s=$(device_asm "$arch" wmma_time) || {
        echo "asm check: no device assembly listing for wmma_time" >>"$log"
        return 1
    }
    echo "asm check: reading $s" >>"$log"
    grep -q 'SHADER_CYCLES' "$s" || {
        echo "asm check: no SHADER_CYCLES read in $s" >>"$log"
        return 1
    }
    while read -r k f a ok; do
        total=$((total + 1))
        [[ $ok == yes ]] || {
            bad=$((bad + 1))
            echo "asm check: chain<$k, $f, $a> loop not as intended" >>"$log"
        }
    done < <(python3 -I "$LOOP_ISA" "$s" chain | awk -v g11="$([[ $arch == gfx1100 ]] && echo 1 || echo 0)" '
        /^void chain</ && g11 {match($0, /<[0-9]+,/); if (substr($0, RSTART + 1, RLENGTH - 2) + 0 > 1) {if (name) print name, ((good && (lines > 0 || wantw + wantf == 0)) ? "yes" : "no"); name = ""; skip = 1; next} skip = 0}
        skip {next}
        /^void chain</ {if (name) print name, ((good && (lines > 0 || wantw + wantf == 0)) ? "yes" : "no"); match($0, /<[0-9]+, [0-9]+, [0-9]+>/); split(substr($0, RSTART + 1, RLENGTH - 2), t, ", "); name = t[1] " " t[2] " " t[3]; good = 1; lines = 0; wantw = (t[1] > 0) ? 16 * t[3] : 0; wantf = 16 * t[2]; next}
        /^    / {lines++; w = 0; f = 0; n = split($0, x, ", "); for (i = 1; i <= n; i++) {split(x[i], y, " "); if (y[1] ~ /^v_wmma/) w += y[2]; if (y[1] == "v_fma_f32" || y[1] == "v_fmac_f32") f += y[2]} if (w != wantw || f != wantf) good = 0}
        END {if (name) print name, ((good && (lines > 0 || wantw + wantf == 0)) ? "yes" : "no")}')
    echo "asm check wmma_time ($arch): $((total - bad)) of $total kernels have the intended loop" >>"$log"
    ((total > 0 && bad == 0))
}

phase_build() {
    local log=$1 arch=$2 out rc tp=""
    out=$REPO/build/$arch
    mkdir -p "$out/asm" "$REPO/build/cpu"
    if (($(avail_gb) < 8)); then
        sum "- build not started: $(avail_gb) GB available, below 8 GB"
        return 1
    fi
    build_one "$arch" bandwidth bandwidth.hip "$log"
    step $? "build bandwidth ($arch)" "$log"
    build_one "$arch" fp8_gemv fp8_gemv.hip "$log"
    step $? "build fp8_gemv ($arch)" "$log"
    # dual issue off, so one FMA takes one issue slot and calibrates the counter
    build_one "$arch" wmma_time wmma_time.hip "$log" -mllvm -amdgpu-enable-vopd=0
    rc=$?
    step $rc "build wmma_time ($arch)" "$log"
    if ((rc == 0)); then
        check_wmma_time_asm "$arch" "$log"
        step $? "assembly of wmma_time: 16 matrix instructions and 16k FMAs per loop, SHADER_CYCLES read" "$log"
    fi
    build_one "$arch" wmma_exact wmma_exact.hip "$log"
    step $? "build wmma_exact ($arch)" "$log"
    if [[ $arch == gfx1201 ]]; then
        build_one "$arch" cvt_fp8 cvt_fp8.hip "$log"
        rc=$?
        step $rc "build cvt_fp8 ($arch)" "$log"
        if ((rc == 0)); then
            grep -c -E 'v_cvt_f32_fp8|v_cvt_pk_f32_fp8|v_cvt_pk_fp8_f32' "$(device_asm "$arch" cvt_fp8)" >/dev/null
            step $? "assembly of cvt_fp8 contains the FP8 conversion instructions" "$log"
        fi
        if [[ -f ${ROCM_PATH:-/nonexistent}/include/hipblaslt/hipblaslt.h || -f /usr/include/hipblaslt/hipblaslt.h ]]; then
            tp=""
        elif [[ -f $REPO/build/third_party/hipblaslt-7.1.1/hipblaslt/hipblaslt.h ]]; then
            tp=$REPO/build/third_party/hipblaslt-7.1.1
        else
            sum "- build hipblaslt_fp8: skipped, no hipBLASLt headers (install hipblaslt-devel)"
            tp=none
        fi
        if [[ $tp != none ]]; then
            build_one "$arch" hipblaslt_fp8 hipblaslt_fp8.hip "$log" ${tp:+-I"$tp"} -l:libhipblaslt.so.1 -lhipblas
            step $? "build hipblaslt_fp8 ($arch)${tp:+, headers from $tp}" "$log"
        fi
    else
        sum "- cvt_fp8, hipblaslt_fp8: gfx1201 only, not built for $arch"
    fi
    # CPU programs, the same for every arch
    echo "## $(date -u +%FT%TZ) cvt_ref_check" >>"$log"
    nice -n 10 g++ -O2 -std=c++17 -I"$LLAMA_UNMOD/ggml/include" "$REPO/src/cvt_ref_check.cpp" \
        -L"$LLAMA_UNMOD/build-gfx1201/bin" -lggml-base -Wl,-rpath,"$LLAMA_UNMOD/build-gfx1201/bin" \
        -o "$REPO/build/cpu/cvt_ref_check" >>"$log" 2>&1
    step $? "build cvt_ref_check (CPU, links llama.cpp's ggml-base)" "$log"
    # A ROCm outside /usr (ROCM_PATH) is not on g++'s default paths, so name its headers and libraries.
    local rocm_flags=()
    if [[ -d ${ROCM_PATH:-/nonexistent}/include/hip ]]; then
        rocm_flags=("-I$ROCM_PATH/include" "-L$ROCM_PATH/lib" "-Wl,-rpath,$ROCM_PATH/lib")
    fi
    nice -n 10 g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ "${rocm_flags[@]}" "$REPO/src/hip_devices.cpp" \
        -o "$REPO/build/cpu/hip_devices" -lamdhip64 >>"$log" 2>&1
    step $? "build hip_devices (CPU)" "$log"
    ((PHASE_FAILED == 0))
}

phase_bandwidth() {
    local log=$1 mib r rd cp
    mib=$((CACHE_TRAP_MB * 2 > 1024 ? CACHE_TRAP_MB * 2 : 1024))
    for r in 1 2 3; do
        gpu_job "$log" "$REPO/build/$ARCH/bandwidth" "$mib" 50
        step $? "bandwidth round $r ($mib MiB per buffer)" "$log"
    done
    rd=$(awk '$1 == "RESULT" && $2 == "read" {print $8}' "$log" | median_spread)
    cp=$(awk '$1 == "RESULT" && $2 == "copy" {print $8}' "$log" | median_spread)
    sum "- read GB/s: $rd"
    sum "- copy GB/s: $cp"
    sum "- memory clock at the end: $(grep '\*' "$DRM_DEVICE/pp_dpm_mclk" | tr -s ' ') (top level $(tail -1 "$DRM_DEVICE/pp_dpm_mclk" | tr -s ' '))"
    ((PHASE_FAILED == 0))
}

phase_probes() {
    local log=$1 b
    b=$REPO/build/$ARCH
    cpu_job "$log" "$REPO/build/cpu/cvt_ref_check"
    step $? "E4M3 reference against llama.cpp and the nearest-value rule (CPU): $(last_verdict cvt_ref "$log")" "$log"
    cpu_job "$log" env HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$b/wmma_exact" selftest
    step $? "wmma_exact CPU self-test: $(last_verdict wmma_exact_selftest "$log")" "$log"
    gpu_job "$log" "$b/wmma_time"
    step $? "wmma_time: $(last_verdict wmma_time "$log")" "$log"
    gpu_job "$log" "$b/wmma_exact"
    step $? "wmma_exact: $(last_verdict wmma_exact "$log")" "$log"
    if [[ $ARCH == gfx1201 ]]; then
        gpu_job "$log" "$b/cvt_fp8" all
        step $? "cvt_fp8: $(last_verdict cvt_fp8 "$log")" "$log"
    else
        sum "- cvt_fp8: skipped, needs gfx1201 (no FP8 conversions on $ARCH)"
    fi
    ((PHASE_FAILED == 0))
}

phase_decode() {
    local log=$1 g v m n r
    g=$REPO/build/$ARCH/fp8_gemv
    for v in a b; do
        for m in plain nan ovf; do
            gpu_job "$log" "$g" "$v" 2 4 16384 14336 20 "$m"
            step $? "decode variant $v mode $m: $(last_verdict decode "$log")" "$log"
        done
    done
    sum ""
    sum "| variant | R | N | weight read GB/s, median of 3 rounds (min, max) |"
    sum "|---|---|---|---|"
    for v in a b; do
        for n in 1 4 8; do
            for r in 1 2 4; do
                local tl=$RUN_DIR/decode-time-$v-$r-$n.log
                for _ in 1 2 3; do
                    gpu_job "$tl" "$g" "$v" "$r" "$n" 16384 14336 200 time
                    (($? == 0 || $? == 4)) || PHASE_FAILED=1
                done
                sum "| $v | $r | $n | $(awk '$1 == "RESULT" {print $NF}' "$tl" | median_spread) |"
            done
        done
    done
    sum ""
    ((PHASE_FAILED == 0))
}

llama_bin() {
    if [[ -n $LLAMA_TAG ]]; then
        echo "$1/build-$ARCH-$LLAMA_TAG/bin"
    elif [[ $ARCH == gfx1100 ]]; then
        echo "$1/build-hip/bin"
    else
        echo "$1/build-$ARCH/bin"
    fi
}

phase_llama() {
    local log=$1 label root bin mdl l res f
    for f in "$(llama_bin "$LLAMA_UNMOD")"/{test-backend-ops,llama-perplexity,llama-bench} \
        "$(llama_bin "$LLAMA_PATCHED")"/{test-backend-ops,llama-perplexity,llama-bench} "$PPL_TEXT"; do
        [[ -e $f ]] || {
            sum "- missing $f"
            return 1
        }
    done
    for mdl in $MODELS; do
        for f in auto fp8-outq8 q8_0; do
            [[ -e $MODEL_DIR/$mdl-$f.gguf ]] || {
                sum "- missing model $MODEL_DIR/$mdl-$f.gguf"
                return 1
            }
        done
    done
    for label in unmodified patched; do
        root=$LLAMA_UNMOD
        [[ $label == patched ]] && root=$LLAMA_PATCHED
        bin=$(llama_bin "$root")
        l=$RUN_DIR/llama-$label-tbo.log
        gpu_job "$l" "$bin/test-backend-ops" test -b ROCm0 -p f8_e4m3
        step $? "$label test-backend-ops f8_e4m3: $(grep -E 'tests passed' "$l" | tail -1 | tr -s ' ')" "$l"
        for mdl in $MODELS; do
            for cfg in "fp8-default auto" "fp8-b8 auto -b 8 -ub 8" "q8_0-default q8_0"; do
                local name file extra
                read -r name file extra <<<"$cfg"
                l=$RUN_DIR/llama-$label-ppl-$mdl-$name.log
                # shellcheck disable=SC2086
                gpu_job "$l" "$bin/llama-perplexity" -m "$MODEL_DIR/$mdl-$file.gguf" -f "$PPL_TEXT" -c 512 --chunks 4 -ngl 99 $extra
                local prc=$?
                res=$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$l" | tail -1)
                # llama-perplexity exits 0 when the result is NaN, so a missing number must fail the step itself.
                if [[ -z $res ]]; then
                    res="no final estimate (NaN or no result); chunks: $(grep -o -E '\[[0-9]+\][^ ,]*' "$l" | tail -4 | tr '\n' ' ')"
                    prc=1
                fi
                step "$prc" "$label perplexity $mdl $name: $res" "$l"
            done
            for file in fp8-outq8 q8_0; do
                l=$RUN_DIR/llama-$label-bench-$mdl-$file.log
                gpu_job "$l" "$bin/llama-bench" -m "$MODEL_DIR/$mdl-$file.gguf" -ngl 99 -p 512 -n 128 -r 3 -o md
                local brc=$?
                step "$brc" "$label llama-bench $mdl $file: $(grep -E 'pp512|tg128' "$l" | awk -F'|' '{gsub(/ +/, " "); printf "%s %s; ", $(NF-2), $(NF-1)}')" "$l"
            done
        done
    done
    ((PHASE_FAILED == 0))
}

phase_libs() {
    local log=$1
    if [[ $ARCH != gfx1201 ]]; then
        sum "- skipped: needs gfx1201 (hipBLASLt ships no FP8 kernels for $ARCH)"
        return 0
    fi
    if [[ -x $REPO/build/$ARCH/hipblaslt_fp8 ]]; then
        gpu_job "$log" "$REPO/build/$ARCH/hipblaslt_fp8" 4096 4096 512 50
        step $? "hipBLASLt FP8: $(last_verdict hipblaslt_fp8 "$log")" "$log"
    else
        sum "- hipBLASLt FP8: not built (see the build log; hipblaslt-devel may be missing)"
        PHASE_FAILED=1
    fi
    if [[ -x $TORCH_PY ]]; then
        gpu_job "$log" "$TORCH_PY" -I "$REPO/src/torch_scaled_mm.py"
        step $? "torch._scaled_mm: $(last_verdict torch_scaled_mm "$log")" "$log"
    else
        sum "- torch._scaled_mm: no PyTorch venv at $TORCH_PY"
        PHASE_FAILED=1
    fi
    ((PHASE_FAILED == 0))
}

# ---------------------------------------------------------------------------------------------------------------

run_phase() {
    local name=$1 log
    log=$RUN_DIR/$1.log
    PHASE_FAILED=0
    sum ""
    sum "## $name ($(date -u +%FT%TZ))"
    if [[ $name == build ]]; then
        "phase_$name" "$log" "$ARCH"
    else
        "phase_$name" "$log"
    fi
    local rc=$?
    if ((rc != 0)); then
        sum "- phase $name: FAILED (log $log)"
        FAILED=1
    fi
    echo "first-hour: $name $([[ $rc == 0 ]] && echo "done" || echo "FAILED"), log $log"
}

main() {
    (($# > 0)) || die "usage: scripts/first-hour.sh <info|build|bandwidth|probes|decode|llama|libs|all>..."
    local phases=() p
    for p in "$@"; do
        case $p in
            all) phases+=(info build bandwidth probes decode llama libs) ;;
            info | build | bandwidth | probes | decode | llama | libs) phases+=("$p") ;;
            *) die "unknown phase $p" ;;
        esac
    done

    # Build-only call with ARCH set: the compile gate, no card needed, no run folder.
    if [[ -n ${ARCH:-} && ${phases[*]} == build ]]; then
        [[ $ARCH == gfx1100 || $ARCH == gfx1201 ]] || die "ARCH must be gfx1100 or gfx1201"
        RUN_DIR=$REPO/build/$ARCH
        mkdir -p "$RUN_DIR"
        : >"$RUN_DIR/summary.md"
        : >"$RUN_DIR/build.log"
        run_phase build
        cat "$RUN_DIR/summary.md"
        exit "$FAILED"
    fi
    [[ -z ${ARCH:-} ]] || die "ARCH only applies to a build-only call; GPU phases use the card record"

    local node arch
    read -r node arch <<<"$(find_card)"
    if [[ " ${phases[*]} " == *" info "* ]]; then
        RUN_DIR=${RUN:-$(new_run_dir "$arch")}
        mkdir -p "$RUN_DIR"
        write_card_env "$node" "$arch" "$RUN_DIR/card.env"
        {
            echo "# First hour: $(date -u +%FT%TZ), $arch"
            echo "Run folder: $RUN_DIR"
        } >>"$RUN_DIR/summary.md"
    else
        RUN_DIR=${RUN:-$(readlink -f "$REPO/runs/latest")}
        [[ -d $RUN_DIR ]] || die "no run folder; start with the info phase"
    fi
    load_card
    [[ $ARCH == "$arch" ]] || die "card record in $RUN_DIR says $ARCH but the installed card is $arch; run info again"
    for p in "${phases[@]}"; do
        run_phase "$p"
    done
    echo "first-hour: summary in $RUN_DIR/summary.md"
    exit "$FAILED"
}

main "$@"
