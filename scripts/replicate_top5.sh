#!/usr/bin/env bash
# Independent operator top-5 GPU replication kit for rdna4-fp8.
#
# Usage:
#   scripts/replicate_top5.sh --plan
#   scripts/replicate_top5.sh --cpu-checks [--out DIR]
#   scripts/replicate_top5.sh --run --out DIR [--probes p1,p2,p3,p4,p5] [--device 0x7550|any-amd] [--lock PATH]
#
# Probes:
#   p1  Non-inlined WMMA call-boundary compiler failure (src/study/haz_hazard.hip, scripts/haz_hazscan.py)
#   p2  LDS CU vs WGP bandwidth difference (src/study/lds_probe.hip, scripts/lds-expect/lds_probe.txt)
#   p3  Neighboring-SIMD L0/memory interference (src/study/mem_inflight.hip, scripts/mem-expect/mem_inflight.txt)
#   p4  Load beside FP8 WMMA (src/round10/ld_mat2.hip, scripts/round10/run_ld_mat2_nopin.sh)
#   p5  HIP vs Vulkan clock & cold-read behavior (src/round10/item3/, scripts/round10/run_item3_bisect.sh)
#
# Exit codes:
#   0  All requested checks/probes passed
#   1  One or more probes, checks, or deliberate breaks failed
#   2  Usage error or pre-flight refusal (missing file, RESTORE-FAILED, missing card)
#   4  Thermal limit exceeded (hotspot > 58 C at start, or >= 95 C guard)
#   5  Available RAM below 8 GiB
#   6  Lock wait expired
#  11  GPU fault or reset found in kernel log
#  12  Kernel log unreadable (journalctl -k)
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
SELF="$HERE/replicate_top5.sh"
EXPECT="$HERE/replicate_top5_expect.md"
ENV_MANIFEST="$HERE/env_manifest.py"
ARCHIVER="$HERE/archive_runs.py"

[[ -r $REPO/local.env ]] && . "$REPO/local.env"

HOT_LIMIT_C=58
HOT_STOP_C=95
MIN_AVAIL_KB=$((8 * 1024 * 1024))
LOCK_WAIT_S=${REPLICATE_LOCK_WAIT_S:-900}
HOT_WAIT_S=${REPLICATE_HOT_WAIT_S:-300}
FAULT_RE='(amdgpu|\[drm|gfxhub|mmhub).*(page fault|GPU reset|ring [A-Za-z0-9_.]+ timeout|VM_L2_PROTECTION|hang|\*ERROR\*)|amdgpu.*(timeout|fault)'

MODE=""
OUT_DIR=""
PROBES_ARG="p1,p2,p3,p4,p5"
DEVICE_ARG="any-amd"
LOCK_ARG=""

usage() {
    sed -n '2,28p' "$SELF"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)
            MODE="plan"
            shift
            ;;
        --cpu-checks)
            MODE="cpu-checks"
            shift
            ;;
        --run)
            MODE="run"
            shift
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "replicate_top5: --out requires a directory" >&2; exit 2; }
            OUT_DIR="$2"
            shift 2
            ;;
        --probes)
            [[ $# -ge 2 ]] || { echo "replicate_top5: --probes requires a comma-separated list" >&2; exit 2; }
            PROBES_ARG="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || { echo "replicate_top5: --device requires 0xXXXX or any-amd" >&2; exit 2; }
            DEVICE_ARG="$2"
            shift 2
            ;;
        --lock)
            [[ $# -ge 2 ]] || { echo "replicate_top5: --lock requires a path" >&2; exit 2; }
            LOCK_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "replicate_top5: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

[[ -n $MODE ]] || { usage >&2; exit 2; }

say() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"
}

# ---- Pure gate helpers (argument-driven so --cpu-checks can test them on fake files) ----

gate_pci_dir() {
    # gate_pci_dir <drm_device_dir> <want: any-amd | 0xXXXX>
    local dev_dir="$1" want="${2:-any-amd}" v d
    { v=$(<"$dev_dir/vendor"); } 2>/dev/null || return 1
    { d=$(<"$dev_dir/device"); } 2>/dev/null || return 1
    v=${v,,}
    d=${d,,}
    [[ $v == "0x1002" || $v == "1002" ]] || return 1
    if [[ $want == "any-amd" ]]; then
        [[ $d =~ ^0x[0-9a-f]+$ ]] || return 1
        return 0
    fi
    want=${want,,}
    [[ $d == "$want" ]]
}

gate_ram_file() {
    local meminfo="$1" kb
    kb=$(awk '/^MemAvailable:/ {print $2}' "$meminfo" 2>/dev/null)
    [[ $kb =~ ^[0-9]+$ ]] && ((kb >= MIN_AVAIL_KB))
}

hot_milli_dir() {
    local hwmon="$1" t
    { t=$(<"$hwmon/temp2_input"); } 2>/dev/null || return 1
    [[ $t =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$t"
}

gate_hot_wait() {
    # gate_hot_wait <hwmon_dir> <limit_c> <max_wait_s> [poll_s]
    local hwmon="$1" limit_c="$2" max_wait="$3" poll="${4:-1}"
    local waited=0 t=0
    while :; do
        t=$(hot_milli_dir "$hwmon") || return 1
        ((t <= limit_c * 1000)) && return 0
        ((waited >= max_wait)) && return 1
        sleep "$poll"
        waited=$((waited + poll))
    done
}

gate_no_restore_failed() {
    local path="$1"
    [[ -z $path || ! -e $path ]]
}

gate_fault_scan() {
    # Returns 0 if clean, 1 if an amdgpu/drm GPU fault or reset line is present.
    ! grep -E -i -q "$FAULT_RE" "$1"
}

gate_lock_acquire() {
    # gate_lock_acquire <lock_file> <wait_s>: opens fd 9 and acquires flock; 0 on success, 6 on timeout, 2 on open err.
    mkdir -p "$(dirname "$1")" 2>/dev/null || return 2
    exec 9>"$1" || return 2
    flock -w "$2" 9 || return 6
}

check_no_home_leaks() {
    # Checks file paths for raw home-directory path prefixes.
    python3 -I - "$@" <<'PY'
import re, sys
sys.dont_write_bytecode = True
pat = re.compile("/" + "(?:home|Users)/")
bad = 0
for p in sys.argv[1:]:
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for idx, ln in enumerate(fh, 1):
            if pat.search(ln):
                sys.stderr.write(f"HOME-LEAK {p}:{idx}\n")
                bad = 1
sys.exit(bad)
PY
}

# ---- Probe catalog ----

probe_ref_files() {
    case "$1" in
        p1)
            printf '%s\n' \
                "src/study/haz_hazard.hip" \
                "scripts/haz_hazscan.py" \
                "scripts/haz-expect/hazscan-haz_hazard.txt" \
                "scripts/study-compiler.sh"
            ;;
        p2)
            printf '%s\n' \
                "src/study/lds_probe.hip" \
                "scripts/lds-expect/lds_probe.txt" \
                "scripts/mem_loopcheck.py" \
                "scripts/study-lds.sh"
            ;;
        p3)
            printf '%s\n' \
                "src/study/mem_inflight.hip" \
                "scripts/mem-expect/mem_inflight.txt" \
                "scripts/mem_loopcheck.py" \
                "scripts/study-memory.sh"
            ;;
        p4)
            printf '%s\n' \
                "src/round10/ld_mat2.hip" \
                "scripts/round10/check_ld_mat2.py" \
                "scripts/round10/run_ld_mat2_nopin.sh"
            ;;
        p5)
            printf '%s\n' \
                "src/round10/item3/build.sh" \
                "src/round10/item3/vkr10.c" \
                "src/round10/item3/r10_cl3.hip" \
                "src/round10/item3/check_item3_listing.py" \
                "src/round10/item3/analyse_item3.py" \
                "scripts/round10/run_item3_bisect.sh"
            ;;
        *)
            return 1
            ;;
    esac
}

parse_probes() {
    local IFS=',' p seen=""
    SELECTED_PROBES=()
    for p in $PROBES_ARG; do
        case "$p" in
            p1|p2|p3|p4|p5)
                if [[ " $seen " != *" $p "* ]]; then
                    SELECTED_PROBES+=("$p")
                    seen="$seen $p"
                fi
                ;;
            *)
                echo "replicate_top5: unknown probe ID '$p' (expected p1..p5)" >&2
                exit 2
                ;;
        esac
    done
    [[ ${#SELECTED_PROBES[@]} -gt 0 ]] || { echo "replicate_top5: empty --probes list" >&2; exit 2; }
}

print_plan() {
    cat <<EOF
Top-5 GPU Replication Plan (independent clean-checkout kit)
Repository:   $REPO
Expectations: $EXPECT
Env manifest: $ENV_MANIFEST
Archiver:     $ARCHIVER
Device arg:   $DEVICE_ARG (default any-amd picks 0x7550 if present, else first 0x1002 DRM card)
Lock rule:    --lock PATH, else \$STUDY_GPU_LOCK, else \$OUT/.replicate.lock

Shared Guards (enforced on every probe in --run):
  - Git checkout status check (records commit & dirty flag via env_manifest.py, warns if dirty)
  - env_manifest.py before and after run
  - RAM floor: MemAvailable >= 8 GiB
  - No RESTORE-FAILED file present
  - Thermal start <= ${HOT_LIMIT_C} C (wait up to ${HOT_WAIT_S} s) before every probe; post-probe check >= ${HOT_STOP_C} C (plus active ${HOT_STOP_C} C guard inside p4 and p5 drivers)
  - Per-probe journalctl -k scan for amdgpu/drm faults, resets, or ring timeouts
  - Probes p4 and p5 require PCI 0x7550 / gfx1201 and are recorded as SKIPPED on other cards

Probes:
  1. [p1] Non-inlined WMMA call-boundary compiler failure (~1 min GPU)
     Sources: src/study/haz_hazard.hip, scripts/haz_hazscan.py, scripts/haz-expect/hazscan-haz_hazard.txt
     Ref:     scripts/study-compiler.sh
     Checks:  Disassembly return==1 / call==1; card 64/256 wrong (bit-identical to no-V_NOP asm);
              0/256 with V_NOP; deliberate break exits 1.

  2. [p2] LDS CU vs WGP bandwidth difference (~2 min GPU)
     Sources: src/study/lds_probe.hip (WGP + -mcumode CU build), scripts/lds-expect/lds_probe.txt
     Checks:  mem_loopcheck.py; FMA calib 1.00 +- 6%; WGP 128 B/cyc/WGP (~64-65/CU) vs
              CU 32-bit 128 B/cyc/CU (~1.9x WGP); deliberate calib & pattern breaks exit 1.

  3. [p3] Neighboring-SIMD L0/memory interference (~2 min GPU)
     Sources: src/study/mem_inflight.hip (-mcumode, waits mode), scripts/mem-expect/mem_inflight.txt
     Checks:  mem_loopcheck.py; <=1.20x with 1 slow wave on same SIMD; >=2.0x slowdown at >=10 slow
              waves on other SIMD (~4x L0, ~2x L2); deliberate break exits 1.

  4. [p4] Load beside FP8 WMMA (ld_mat2, no-pin L0/L2) (~15-25 min GPU)
     Sources: src/round10/ld_mat2.hip, scripts/round10/check_ld_mat2.py, scripts/round10/run_ld_mat2_nopin.sh
     Checks:  check_ld_mat2.py + selftest; ref gate 7.9-8.1 (1-wave) / 7.9-8.3 (card);
              1-wave ~25 cyc/mm L0/L2 plateau; c6_aim_break >= 90% gap closure; c7/c8/c9/c10 controls.

  5. [p5] HIP vs Vulkan clock & cold-read behavior (~6 min GPU, --no-pin default)
     Sources: src/round10/item3/{build.sh,vkr10.c,r10_cl3.hip,check_item3_listing.py,analyse_item3.py}
     Driver:  scripts/round10/run_item3_bisect.sh (--no-pin --steps 2-6)
     Checks:  SPIR-V & HIP listing checks; matched-clock (uclk0==uclk1) cold latency comparison;
              no-evict / loop8 / placement breaks; repeats <= 5%; start <= 58 C.

Estimated total GPU time (all 5 probes): ~26-36 minutes (mostly 58 C thermal cool-downs in p4/p5).
EOF
}

# ---- --cpu-checks mode (100% CPU-only, zero GPU binaries, zero sysfs, zero real lock) ----

run_cpu_checks() {
    local dir="$OUT_DIR"
    if [[ -z $dir ]]; then
        dir="$REPO/build/replicate-top5-cpu-checks-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mkdir -p "$dir" || { echo "replicate_top5: cannot create $dir" >&2; exit 2; }
    local T="$dir/fake"
    rm -rf "$T"
    mkdir -p "$T"

    local fails=0
    pass() { printf 'PASS %s\n' "$1"; }
    fail() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }
    t_ok() {
        local msg="$1"
        shift
        if "$@" >/dev/null 2>&1; then pass "$msg"; else fail "$msg"; fi
    }
    t_no() {
        local msg="$1"
        shift
        if "$@" >/dev/null 2>&1; then fail "$msg (deliberate break was NOT caught)"; else pass "$msg"; fi
    }

    echo "## Syntax and static hygiene"
    if bash -n "$SELF"; then pass "bash -n replicate_top5.sh"; else fail "bash -n replicate_top5.sh"; fi
    if command -v shellcheck >/dev/null 2>&1; then
        if shellcheck -S warning "$SELF"; then pass "shellcheck -S warning replicate_top5.sh"; else fail "shellcheck -S warning replicate_top5.sh"; fi
    else
        pass "shellcheck not installed (skipped)"
    fi
    if check_no_home_leaks "$SELF" "$EXPECT" "$ARCHIVER"; then
        pass "zero home-path leaks in replicate_top5 files and archive_runs.py"
    else
        fail "home-path leak detected in scripts"
    fi

    echo "## Expectations coverage and referenced repo files"
    if [[ -r $EXPECT ]]; then
        pass "expectations file exists: replicate_top5_expect.md"
        local pid
        for pid in p1 p2 p3 p4 p5; do
            if grep -q "\`$pid\`" "$EXPECT"; then
                pass "expectations covers probe ID $pid"
            else
                fail "expectations missing probe ID $pid"
            fi
        done
    else
        fail "expectations file missing: $EXPECT"
    fi

    local pid rf
    for pid in p1 p2 p3 p4 p5; do
        while IFS= read -r rf; do
            if [[ -e $REPO/$rf ]]; then
                pass "$pid file exists: $rf"
            else
                fail "$pid file missing: $rf"
            fi
        done < <(probe_ref_files "$pid")
    done
    [[ -r $ENV_MANIFEST ]] && pass "env_manifest.py exists" || fail "env_manifest.py missing"
    [[ -r $ARCHIVER ]] && pass "archive_runs.py exists" || fail "archive_runs.py missing"

    echo "## Safety gates against fake files (with deliberate breaks)"
    mkdir -p "$T/gpu_7550" "$T/gpu_744c" "$T/gpu_intel"
    printf '0x1002\n' >"$T/gpu_7550/vendor"; printf '0x7550\n' >"$T/gpu_7550/device"
    printf '0x1002\n' >"$T/gpu_744c/vendor"; printf '0x744c\n' >"$T/gpu_744c/device"
    printf '0x8086\n' >"$T/gpu_intel/vendor"; printf '0x7550\n' >"$T/gpu_intel/device"
    t_ok "PCI gate accepts 0x1002:0x7550 when requesting 0x7550" gate_pci_dir "$T/gpu_7550" 0x7550
    t_ok "PCI gate accepts 0x1002:0x744c under any-amd" gate_pci_dir "$T/gpu_744c" any-amd
    t_no "PCI break: refuses 0x744c when 0x7550 requested" gate_pci_dir "$T/gpu_744c" 0x7550
    t_no "PCI break: refuses non-AMD vendor 0x8086" gate_pci_dir "$T/gpu_intel" any-amd
    t_no "PCI break: refuses missing device dir" gate_pci_dir "$T/missing_gpu" any-amd

    printf 'MemAvailable:   16777216 kB\n' >"$T/mem_16g"
    printf 'MemAvailable:    8388608 kB\n' >"$T/mem_8g"
    printf 'MemAvailable:    8388607 kB\n' >"$T/mem_under"
    printf 'MemAvailable:    4194304 kB\n' >"$T/mem_4g"
    printf 'corrupt\n' >"$T/mem_junk"
    t_ok "RAM gate accepts 16 GiB" gate_ram_file "$T/mem_16g"
    t_ok "RAM gate accepts exact 8 GiB floor" gate_ram_file "$T/mem_8g"
    t_no "RAM break: refuses 8 GiB - 1 KiB" gate_ram_file "$T/mem_under"
    t_no "RAM break: refuses 4 GiB" gate_ram_file "$T/mem_4g"
    t_no "RAM break: refuses corrupt meminfo" gate_ram_file "$T/mem_junk"

    mkdir -p "$T/hw"
    printf '48000\n' >"$T/hw/temp2_input"
    t_ok "thermal gate accepts 48 C" gate_hot_wait "$T/hw" "$HOT_LIMIT_C" 1 1
    printf '58000\n' >"$T/hw/temp2_input"
    t_ok "thermal gate accepts exact 58 C limit" gate_hot_wait "$T/hw" "$HOT_LIMIT_C" 1 1
    printf '58001\n' >"$T/hw/temp2_input"
    t_no "thermal break: refuses 58.001 C" gate_hot_wait "$T/hw" "$HOT_LIMIT_C" 1 1
    printf '72000\n' >"$T/hw/temp2_input"
    t_no "thermal break: refuses 72 C" gate_hot_wait "$T/hw" "$HOT_LIMIT_C" 1 1
    printf 'bad\n' >"$T/hw/temp2_input"
    t_no "thermal break: refuses non-numeric sensor" gate_hot_wait "$T/hw" "$HOT_LIMIT_C" 1 1
    t_no "thermal break: refuses missing sensor dir" gate_hot_wait "$T/hw_missing" "$HOT_LIMIT_C" 1 1

    : >"$T/restore_failed_sentinel"
    t_ok "RESTORE-FAILED gate passes when absent" gate_no_restore_failed "$T/no_such_restore_failed"
    t_no "RESTORE-FAILED break: refuses when sentinel file exists" gate_no_restore_failed "$T/restore_failed_sentinel"

    printf 'Nov 26 03:00:00 host kernel: usb 8-1: cannot submit urb 0, error -19: no device\n' >"$T/kern_usb"
    printf 'Nov 26 03:00:01 host kernel: amdgpu 0000:03:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32777)\n' >"$T/kern_pf"
    printf 'Nov 26 03:00:02 host kernel: amdgpu 0000:03:00.0: amdgpu: GPU reset begin!\n' >"$T/kern_reset"
    printf 'Nov 26 03:00:03 host kernel: amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1\n' >"$T/kern_ring"
    t_ok "fault scan passes on unrelated USB line" gate_fault_scan "$T/kern_usb"
    t_no "fault scan break: catches amdgpu page fault" gate_fault_scan "$T/kern_pf"
    t_no "fault scan break: catches amdgpu GPU reset" gate_fault_scan "$T/kern_reset"
    t_no "fault scan break: catches amdgpu ring timeout" gate_fault_scan "$T/kern_ring"

    echo "## Private lock contention test (zero shared lock touched)"
    : >"$T/fake.lock"
    (
        exec 8>"$T/fake.lock"
        flock -n 8 || exit 1
        sleep 10
    ) &
    local holder=$!
    sleep 0.3
    local lock_rc=0
    bash -c "$(declare -f gate_lock_acquire); gate_lock_acquire '$T/fake.lock' 1" >/dev/null 2>&1 || lock_rc=$?
    if ((lock_rc == 6)); then
        pass "lock gate returns exit 6 when fake lock is held"
    else
        fail "lock gate returned $lock_rc instead of 6 while fake lock was held"
    fi
    kill "$holder" 2>/dev/null || true
    wait "$holder" 2>/dev/null || true
    if bash -c "$(declare -f gate_lock_acquire); gate_lock_acquire '$T/fake.lock' 1" >/dev/null 2>&1; then
        pass "lock gate acquires fake lock once released"
    else
        fail "lock gate failed after fake lock was released"
    fi

    echo "## End-to-end metadata + archive handoff on fake run dir"
    local fake_run="$T/fake_run"
    local fake_arc="$T/fake_archive"
    mkdir -p "$fake_run/p1"
    cp "$EXPECT" "$fake_run/replicate_top5_expect.md"
    printf 'VERDICT haz_hazard: PASS (fake cpu-check fixture)\n' >"$fake_run/p1/haz_hazard.log"
    printf 'PROBE p1 status=PASS\n' >"$fake_run/summary.txt"
    if python3 -I "$ARCHIVER" pack --run-dir "$fake_run" --out-dir "$fake_arc" --repo "$REPO" --name fake-top5 --no-live-manifest >"$T/pack.log" 2>&1; then
        pass "archive_runs.py pack (--no-live-manifest) succeeded"
    else
        fail "archive_runs.py pack failed (see $T/pack.log)"
    fi
    if python3 -I "$ARCHIVER" verify --out-dir "$fake_arc" --name fake-top5 >"$T/verify.log" 2>&1; then
        pass "archive_runs.py verify passed on packed bundle"
    else
        fail "archive_runs.py verify failed (see $T/verify.log)"
    fi
    if check_no_home_leaks "$fake_arc/fake-top5.manifest.json" "$fake_arc/fake-top5.SHA256SUMS"; then
        pass "packed bundle manifest and checksums contain zero home-path leaks"
    else
        fail "home-path leak found in packed bundle metadata"
    fi

    echo "cpu-checks: $fails failed (artifacts in $dir)"
    ((fails == 0))
}

# ---- --run mode helpers (GPU execution; never invoked by --plan or --cpu-checks) ----

select_amd_drm_card() {
    # Sets RESOLVED_DRM, RESOLVED_HWMON, RESOLVED_DEVID_HEX, RESOLVED_ARCH, RESOLVED_HIP_INDEX.
    # Matches the chosen DRM card to its exact KFD node by drm_render_minor so arch and HIP_VISIBLE_DEVICES
    # always refer to the selected card. Under any-amd, prefers 0x7550, else a single discrete AMD GPU
    # (simd_count > 8), refusing ambiguous multi-dGPU setups unless --device 0xXXXX is specified.
    local want="$1" d vid did render_node render_minor="" chosen_dev="" chosen_did="" chosen_minor=""
    local dgpu_devs=() dgpu_dids=() dgpu_minors=()
    local all_devs=() all_dids=() all_minors=()

    kfd_info_for_minor() {
        # kfd_info_for_minor <minor> -> prints "<simd_count> <gfx_ver> <hip_index>" or returns 1
        local want_m="$1" n simd ver m hip_idx=0
        for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/[0-9]* 2>/dev/null | sort -t/ -k8,8n); do
            [[ -r $n/properties ]] || continue
            simd=$(awk '$1 == "simd_count" {print $2}' "$n/properties")
            [[ ${simd:-0} -gt 0 ]] || continue
            ver=$(awk '$1 == "gfx_target_version" {print $2}' "$n/properties")
            m=$(awk '$1 == "drm_render_minor" {print $2}' "$n/properties")
            if [[ $m == "$want_m" ]]; then
                printf '%s %s %s\n' "$simd" "$ver" "$hip_idx"
                return 0
            fi
            hip_idx=$((hip_idx + 1))
        done
        return 1
    }

    for d in /sys/class/drm/card[0-9]*/device; do
        [[ -f $d/vendor && -f $d/device ]] || continue
        vid=$(<"$d/vendor")
        did=$(<"$d/device")
        vid=${vid,,}
        did=${did,,}
        [[ $vid == "0x1002" || $vid == "1002" ]] || continue
        render_node=""
        local rn
        for rn in "$d"/drm/renderD[0-9]*; do
            [[ -e $rn ]] || continue
            render_node=$(basename "$rn")
            break
        done
        render_minor=${render_node#renderD}
        if [[ $want != "any-amd" ]]; then
            if [[ $did == "${want,,}" ]]; then
                chosen_dev=$(readlink -f "$d")
                chosen_did="$did"
                chosen_minor="$render_minor"
                break
            fi
            continue
        fi
        if [[ $did == "0x7550" ]]; then
            chosen_dev=$(readlink -f "$d")
            chosen_did="$did"
            chosen_minor="$render_minor"
            break
        fi
        all_devs+=("$(readlink -f "$d")")
        all_dids+=("$did")
        all_minors+=("$render_minor")
        local kinfo ksimd=0
        if [[ -n $render_minor ]] && kinfo=$(kfd_info_for_minor "$render_minor"); then
            ksimd=${kinfo%% *}
            if ((ksimd > 8)); then
                dgpu_devs+=("$(readlink -f "$d")")
                dgpu_dids+=("$did")
                dgpu_minors+=("$render_minor")
            fi
        fi
    done

    if [[ -z $chosen_dev && $want == "any-amd" ]]; then
        if ((${#dgpu_devs[@]} == 1)); then
            chosen_dev="${dgpu_devs[0]}"
            chosen_did="${dgpu_dids[0]}"
            chosen_minor="${dgpu_minors[0]}"
        elif ((${#dgpu_devs[@]} > 1)); then
            echo "replicate_top5: multiple discrete AMD GPUs detected (${dgpu_dids[*]}); pass --device 0xXXXX explicitly" >&2
            return 1
        elif ((${#all_devs[@]} == 1)); then
            chosen_dev="${all_devs[0]}"
            chosen_did="${all_dids[0]}"
            chosen_minor="${all_minors[0]}"
        fi
    fi

    [[ -n $chosen_dev ]] || return 1
    RESOLVED_DRM="$chosen_dev"
    RESOLVED_DEVID_HEX="$chosen_did"
    RESOLVED_HWMON=$(ls -d "$RESOLVED_DRM"/hwmon/hwmon* 2>/dev/null | head -1)
    [[ -d $RESOLVED_HWMON ]] || return 1

    RESOLVED_ARCH=""
    RESOLVED_HIP_INDEX=""
    local kinfo ksimd kver khip
    if [[ -n $chosen_minor ]] && kinfo=$(kfd_info_for_minor "$chosen_minor"); then
        read -r ksimd kver khip <<<"$kinfo"
        RESOLVED_HIP_INDEX="$khip"
        case "$kver" in
            120001) RESOLVED_ARCH="gfx1201" ;;
            120000) RESOLVED_ARCH="gfx1200" ;;
            110501) RESOLVED_ARCH="gfx1151" ;;
            110500) RESOLVED_ARCH="gfx1150" ;;
            110000) RESOLVED_ARCH="gfx1100" ;;
            110001) RESOLVED_ARCH="gfx1101" ;;
            110002) RESOLVED_ARCH="gfx1102" ;;
        esac
    fi
    if [[ -z $RESOLVED_ARCH && -r $REPO/runs/latest/card.env ]]; then
        local cdev carch chi
        cdev=$(sed -n 's/^DRM_DEVICE=//p' "$REPO/runs/latest/card.env" | head -1)
        if [[ -n $cdev && $(readlink -f "$cdev" 2>/dev/null) == "$RESOLVED_DRM" ]]; then
            carch=$(sed -n 's/^ARCH=//p' "$REPO/runs/latest/card.env" | head -1)
            chi=$(sed -n 's/^HIP_INDEX=//p' "$REPO/runs/latest/card.env" | head -1)
            [[ -n $carch ]] && RESOLVED_ARCH="$carch"
            [[ -z $RESOLVED_HIP_INDEX && $chi =~ ^[0-9]+$ ]] && RESOLVED_HIP_INDEX="$chi"
        fi
    fi
    if [[ -z $RESOLVED_ARCH || -z $RESOLVED_HIP_INDEX ]]; then
        echo "replicate_top5: could not resolve KFD arch/HIP_INDEX for $RESOLVED_DRM ($RESOLVED_DEVID_HEX, renderD${chosen_minor:-?})" >&2
        return 1
    fi
    return 0
}

pre_probe_gates() {
    local label="$1" restore_sentinel="$2"
    gate_no_restore_failed "$restore_sentinel" || { say "REFUSED $label: RESTORE-FAILED exists ($restore_sentinel)"; return 2; }
    gate_pci_dir "$RESOLVED_DRM" "$RESOLVED_DEVID_HEX" || { say "REFUSED $label: PCI device check failed on $RESOLVED_DRM"; return 2; }
    gate_ram_file /proc/meminfo || { say "REFUSED $label: available RAM below 8 GiB"; return 5; }
    gate_hot_wait "$RESOLVED_HWMON" "$HOT_LIMIT_C" "$HOT_WAIT_S" 2 || { say "REFUSED $label: hotspot did not reach <= ${HOT_LIMIT_C} C within ${HOT_WAIT_S} s"; return 4; }
    return 0
}

post_probe_fault_check() {
    local label="$1" t0="$2" pdir="$3"
    local kl="$pdir/kernlog.txt"
    if ! journalctl -k --since "@$t0" --no-pager -q >"$kl" 2>"$pdir/journal.err"; then
        say "REFUSED $label: kernel log unreadable after job"
        return 12
    fi
    if ! gate_fault_scan "$kl"; then
        grep -E -i "$FAULT_RE" "$kl" >"$pdir/faults.txt" || true
        say "FAULT $label: GPU fault found in kernel log (see $pdir/faults.txt)"
        return 11
    fi
    local hot_now
    hot_now=$(hot_milli_dir "$RESOLVED_HWMON" 2>/dev/null || echo 0)
    if ((hot_now >= HOT_STOP_C * 1000)); then
        say "THERMAL STOP $label: hotspot reached $((hot_now / 1000)) C (>= ${HOT_STOP_C} C)"
        return 4
    fi
    return 0
}

run_gpu_suite() {
    [[ -n $OUT_DIR ]] || { echo "replicate_top5: --run requires --out DIR" >&2; exit 2; }
    parse_probes

    local rocm_path="${ROCM_PATH:-}"
    if [[ -z $rocm_path ]] && command -v hipcc >/dev/null 2>&1; then
        rocm_path=$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)
    fi
    if [[ -n $rocm_path ]]; then
        export ROCM_PATH="$rocm_path"
        export HIP_PATH="$rocm_path"
        export PATH="$rocm_path/bin:$PATH"
    fi
    local restore_sentinel="${R10_RESTORE_FAILED:-${R9_RESTORE_FAILED:-}}"

    gate_no_restore_failed "$restore_sentinel" || { echo "replicate_top5: RESTORE-FAILED exists ($restore_sentinel)" >&2; exit 2; }
    journalctl -k -n 1 --no-pager -q >/dev/null 2>&1 || { echo "replicate_top5: journalctl -k unreadable" >&2; exit 12; }
    select_amd_drm_card "$DEVICE_ARG" || { echo "replicate_top5: no matching AMD DRM card for --device $DEVICE_ARG" >&2; exit 2; }
    gate_ram_file /proc/meminfo || { echo "replicate_top5: available RAM below 8 GiB" >&2; exit 5; }

    mkdir -p "$OUT_DIR/probes" "$OUT_DIR/build" "$OUT_DIR/archive" || { echo "replicate_top5: cannot create $OUT_DIR" >&2; exit 2; }
    OUT_DIR=$(cd "$OUT_DIR" && pwd)

    local lock_path="${LOCK_ARG:-${STUDY_GPU_LOCK:-$OUT_DIR/.replicate.lock}}"
    local summary="$OUT_DIR/summary.txt"
    : >"$summary"

    {
        echo "<!-- copied $(date -u +%FT%TZ) before probes; git $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) -->"
        cat "$EXPECT"
    } >"$OUT_DIR/replicate_top5_expect.md"

    cat >"$OUT_DIR/card.env" <<EOF
ARCH=$RESOLVED_ARCH
DRM_DEVICE=$RESOLVED_DRM
HWMON=$RESOLVED_HWMON
HIP_INDEX=$RESOLVED_HIP_INDEX
EOF
    export CARD_ENV="$OUT_DIR/card.env"

    python3 -I "$ENV_MANIFEST" --repo "$REPO" --device "$RESOLVED_DEVID_HEX" --out "$OUT_DIR/env-manifest-before.json" || {
        echo "replicate_top5: env_manifest.py before run failed" >&2
        exit 2
    }

    local git_dirty
    git_dirty=$(git -C "$REPO" status --porcelain 2>/dev/null || true)
    if [[ -n $git_dirty ]]; then
        say "WARN: git checkout has uncommitted changes (recorded as dirty in env-manifest-before.json)" | tee -a "$summary"
    else
        say "git checkout clean ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))" | tee -a "$summary"
    fi

    say "Starting top-5 replication (device $RESOLVED_DEVID_HEX arch $RESOLVED_ARCH hip_index $RESOLVED_HIP_INDEX)" | tee -a "$summary"

    local overall_rc=0 pid pdir t0 rc build_dir
    build_dir="$OUT_DIR/build"
    for pid in "${SELECTED_PROBES[@]}"; do
        pdir="$OUT_DIR/probes/$pid"
        mkdir -p "$pdir"
        if [[ ($pid == "p4" || $pid == "p5") && ($RESOLVED_DEVID_HEX != "0x7550" || $RESOLVED_ARCH != "gfx1201") ]]; then
            say "PROBE $pid SKIPPED (requires PCI 0x7550 / gfx1201; detected $RESOLVED_DEVID_HEX / $RESOLVED_ARCH)" | tee -a "$summary"
            continue
        fi
        pre_probe_gates "$pid" "$restore_sentinel"
        rc=$?
        if ((rc != 0)); then
            say "PROBE $pid NOT STARTED (gate exit $rc)" | tee -a "$summary"
            overall_rc=$rc
            break
        fi

        t0=$(date +%s)
        rc=0
        case "$pid" in
            p1)
                (
                    mkdir -p "$build_dir/haz_hazard_tmp" &&
                    cd "$build_dir/haz_hazard_tmp" &&
                    nice -n 10 hipcc -O3 -std=c++17 --offload-arch="$RESOLVED_ARCH" -DHARNESS_ARCH="\"$RESOLVED_ARCH\"" \
                        -I"$REPO/src" -I"$REPO/src/study" -save-temps \
                        "$REPO/src/study/haz_hazard.hip" -o "$build_dir/haz_hazard" -lamdhip64
                ) >"$pdir/build.log" 2>&1 || rc=$?
                if ((rc == 0)); then
                    local haz_s
                    haz_s=$(ls "$build_dir/haz_hazard_tmp"/*.img.lto.s 2>/dev/null | head -1)
                    if [[ -n $haz_s && -f $haz_s ]]; then
                        cp "$haz_s" "$pdir/haz_hazard.s"
                        python3 -I "$REPO/scripts/haz_hazscan.py" "$haz_s" "$REPO/scripts/haz-expect/hazscan-haz_hazard.txt" >"$pdir/hazscan.log" 2>&1 || rc=$?
                    else
                        echo "replicate_top5 p1: missing *.img.lto.s in $build_dir/haz_hazard_tmp" >"$pdir/hazscan.log"
                        rc=1
                    fi
                fi
                if ((rc == 0)); then
                    (
                        gate_lock_acquire "$lock_path" "$LOCK_WAIT_S" || exit $?
                        date +%s >"$pdir/t0"
                        HIP_VISIBLE_DEVICES="$RESOLVED_HIP_INDEX" CARD_ARCH="$RESOLVED_ARCH" \
                            timeout -k 10 120 nice -n 10 "$build_dir/haz_hazard" all >"$pdir/run.log" 2>&1 || exit 1
                        local brk_rc=0
                        HIP_VISIBLE_DEVICES="$RESOLVED_HIP_INDEX" CARD_ARCH="$RESOLVED_ARCH" \
                            timeout -k 10 120 nice -n 10 "$build_dir/haz_hazard" break >"$pdir/break.log" 2>&1 || brk_rc=$?
                        ((brk_rc == 1)) || exit 1
                    ) || rc=$?
                fi
                ;;
            p2)
                (
                    mkdir -p "$build_dir/lds_wgp" "$build_dir/lds_cu" &&
                    (cd "$build_dir/lds_wgp" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch="$RESOLVED_ARCH" -DHARNESS_ARCH="\"$RESOLVED_ARCH\"" \
                        -save-temps -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/study/lds_probe.hip" -o "$build_dir/lds_probe" -lamdhip64) &&
                    (cd "$build_dir/lds_cu" && nice -n 10 hipcc -O3 -std=c++17 --offload-arch="$RESOLVED_ARCH" -DHARNESS_ARCH="\"$RESOLVED_ARCH\"" \
                        -mcumode -DUSE_CU_KEY=1 -save-temps -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/study/lds_probe.hip" -o "$build_dir/lds_probe_cu" -lamdhip64)
                ) >"$pdir/build.log" 2>&1 || rc=$?
                if ((rc == 0)); then
                    local wgp_s cu_s
                    wgp_s=$(ls "$build_dir/lds_wgp"/*.lto.s 2>/dev/null | head -1)
                    cu_s=$(ls "$build_dir/lds_cu"/*.lto.s 2>/dev/null | head -1)
                    [[ -n $wgp_s && -f $wgp_s ]] && cp "$wgp_s" "$pdir/lds_probe_wgp.s"
                    [[ -n $cu_s && -f $cu_s ]] && cp "$cu_s" "$pdir/lds_probe_cu.s"
                    python3 -I "$REPO/scripts/mem_loopcheck.py" "$wgp_s" "$REPO/scripts/lds-expect/lds_probe.txt" >"$pdir/loopcheck-wgp.log" 2>&1 || rc=$?
                    python3 -I "$REPO/scripts/mem_loopcheck.py" "$cu_s" "$REPO/scripts/lds-expect/lds_probe.txt" >"$pdir/loopcheck-cu.log" 2>&1 || rc=$?
                fi
                if ((rc == 0)); then
                    (
                        gate_lock_acquire "$lock_path" "$LOCK_WAIT_S" || exit $?
                        date +%s >"$pdir/t0"
                        export DRM_DEVICE="$RESOLVED_DRM" HWMON="$RESOLVED_HWMON" HIP_VISIBLE_DEVICES="$RESOLVED_HIP_INDEX" CARD_ARCH="$RESOLVED_ARCH"
                        timeout -k 10 180 nice -n 10 "$build_dir/lds_probe" calib >"$pdir/wgp-calib.log" 2>&1 || exit 1
                        timeout -k 10 180 nice -n 10 "$build_dir/lds_probe" waves >"$pdir/wgp-waves.log" 2>&1 || exit 1
                        timeout -k 10 180 nice -n 10 "$build_dir/lds_probe_cu" waves >"$pdir/cu-waves.log" 2>&1 || exit 1
                        timeout -k 10 180 nice -n 10 "$build_dir/lds_probe" card >"$pdir/wgp-card.log" 2>&1 || exit 1
                        timeout -k 10 180 nice -n 10 "$build_dir/lds_probe_cu" card >"$pdir/cu-card.log" 2>&1 || exit 1
                        local b1=0 b2=0
                        timeout -k 10 120 nice -n 10 "$build_dir/lds_probe" calib --break >"$pdir/break-calib.log" 2>&1 || b1=$?
                        timeout -k 10 120 nice -n 10 "$build_dir/lds_probe" pattern --break >"$pdir/break-pattern.log" 2>&1 || b2=$?
                        ((b1 == 1 && b2 == 1)) || exit 1
                    ) || rc=$?
                fi
                ;;
            p3)
                (
                    mkdir -p "$build_dir/mem_inflight" &&
                    cd "$build_dir/mem_inflight" &&
                    nice -n 10 hipcc -O3 -std=c++17 --offload-arch="$RESOLVED_ARCH" -DHARNESS_ARCH="\"$RESOLVED_ARCH\"" \
                        -mcumode -save-temps -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/study/mem_inflight.hip" -o "$build_dir/mem_inflight" -lamdhip64
                ) >"$pdir/build.log" 2>&1 || rc=$?
                if ((rc == 0)); then
                    local mem_s
                    mem_s=$(ls "$build_dir/mem_inflight"/*.lto.s 2>/dev/null | head -1)
                    [[ -n $mem_s && -f $mem_s ]] && cp "$mem_s" "$pdir/mem_inflight.s"
                    python3 -I "$REPO/scripts/mem_loopcheck.py" "$mem_s" "$REPO/scripts/mem-expect/mem_inflight.txt" >"$pdir/loopcheck.log" 2>&1 || rc=$?
                fi
                if ((rc == 0)); then
                    (
                        gate_lock_acquire "$lock_path" "$LOCK_WAIT_S" || exit $?
                        date +%s >"$pdir/t0"
                        export DRM_DEVICE="$RESOLVED_DRM" HWMON="$RESOLVED_HWMON" HIP_VISIBLE_DEVICES="$RESOLVED_HIP_INDEX" CARD_ARCH="$RESOLVED_ARCH"
                        timeout -k 10 240 nice -n 10 "$build_dir/mem_inflight" waits >"$pdir/waits.log" 2>&1 || exit 1
                        local b1=0
                        timeout -k 10 240 nice -n 10 "$build_dir/mem_inflight" waits --break >"$pdir/waits-break.log" 2>&1 || b1=$?
                        ((b1 == 1)) || exit 1
                    ) || rc=$?
                fi
                ;;
            p4)
                local ldm2_bin="$build_dir/ld_mat2"
                (
                    nice -n 10 hipcc -O3 -std=c++17 -mcumode --offload-arch="$RESOLVED_ARCH" -DHARNESS_ARCH="\"$RESOLVED_ARCH\"" \
                        -I"$REPO/src" -I"$REPO/src/study" "$REPO/src/round10/ld_mat2.hip" -o "$ldm2_bin" -lamdhip64
                ) >"$pdir/build.log" 2>&1 || rc=$?
                if ((rc == 0)); then
                    STUDY_GPU_LOCK="$lock_path" CARD_ENV="$OUT_DIR/card.env" LDM2_BIN="$ldm2_bin" LDM2_EXPECT="$EXPECT" \
                        bash "$REPO/scripts/round10/run_ld_mat2_nopin.sh" "$pdir/run" >"$pdir/driver.log" 2>&1 || rc=$?
                fi
                ;;
            p5)
                local i3_out="$build_dir/item3"
                R10_ITEM3_OUT="$i3_out" bash "$REPO/src/round10/item3/build.sh" >"$pdir/build.log" 2>&1 || rc=$?
                if ((rc == 0)); then
                    STUDY_GPU_LOCK="$lock_path" CARD_ENV="$OUT_DIR/card.env" R10_ITEM3_BIN="$i3_out" R10_RUN="$pdir/run" \
                        bash "$REPO/scripts/round10/run_item3_bisect.sh" --no-pin --steps 2-6 --out "$pdir/run" >"$pdir/driver.log" 2>&1 || rc=$?
                fi
                ;;
        esac

        if [[ -r $pdir/t0 ]]; then
            t0=$(<"$pdir/t0")
        fi
        post_probe_fault_check "$pid" "$t0" "$pdir"
        local frc=$?
        if ((frc != 0)); then
            say "PROBE $pid STOPPED by post-probe check (exit $frc)" | tee -a "$summary"
            overall_rc=$frc
            break
        fi
        if ((rc != 0)); then
            say "PROBE $pid FAILED (exit $rc; see probes/$pid)" | tee -a "$summary"
            case "$rc" in
                2|4|5|6|11|12)
                    overall_rc=$rc
                    break
                    ;;
                13)
                    overall_rc=11
                    break
                    ;;
                *)
                    ((overall_rc == 0)) && overall_rc=1
                    ;;
            esac
        else
            say "PROBE $pid PASS" | tee -a "$summary"
        fi
    done

    python3 -I "$ENV_MANIFEST" --repo "$REPO" --device "$RESOLVED_DEVID_HEX" --out "$OUT_DIR/env-manifest-after.json" || {
        say "WARN: env_manifest.py after run failed" | tee -a "$summary"
        ((overall_rc == 0)) && overall_rc=1
    }

    local bundle_name
    bundle_name="replicate-top5-$(basename "$OUT_DIR")"
    if python3 -I "$ARCHIVER" pack --run-dir "$OUT_DIR" --out-dir "$OUT_DIR/archive" --repo "$REPO" --name "$bundle_name" --exclude build --no-live-manifest >"$OUT_DIR/archive/pack.log" 2>&1 &&
       python3 -I "$ARCHIVER" verify --out-dir "$OUT_DIR/archive" --name "$bundle_name" >"$OUT_DIR/archive/verify.log" 2>&1; then
        say "ARCHIVE PASS: archive/$bundle_name.tar.gz" | tee -a "$summary"
    else
        say "ARCHIVE FAILED (see archive/)" | tee -a "$summary"
        ((overall_rc == 0)) && overall_rc=1
    fi

    return "$overall_rc"
}

case "$MODE" in
    plan) print_plan ;;
    cpu-checks) run_cpu_checks ;;
    run) run_gpu_suite ;;
esac
