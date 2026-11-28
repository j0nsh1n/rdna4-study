#!/usr/bin/env bash
# Round 10 item 1: ld_mat2 at L0 and L2, no pinned memory clock. The MALL and GDDR6 cells are NOT run here: the binary holds
# them (and refuses them without --pinned) and they wait for item 2a's pinned job, which reuses this exact binary.
# Usage: run_ld_mat2_nopin.sh [run folder]        a real run (GPU): shared lock, then the jobs below, in this order
#        run_ld_mat2_nopin.sh --plan              print the jobs, limits and the time estimate; touches nothing
#        run_ld_mat2_nopin.sh --cpu-checks [dir]  CPU only: self-tests of every gate with deliberate breaks, the listing check
#                                                 of the binary that would run, the plan against the listing; no lock, no GPU
# Jobs (each is one process; the reference runs first inside every job and nothing is interpreted if it does not read 8.0):
#   one-L0L2    one wave (209 registers), L0 and L2: K = 1 D 0,1,2,4,8,24,54; K = 4 D 0,2,6,12; K = 8 D 0,2,6,12; 3 bursts of 20 ms
#   card209-L0  whole card, 7 waves per SIMD, L0: K = 1 D 0,2,8,max;           3 bursts of 50 ms
#   card209-L2  whole card, 7 waves per SIMD, L2: K = 1 D 0,2,8,max and K = 8 D 0,2,8,max
#   card51-L0   whole card, 16 waves per SIMD, L0: K = 1 D 0,2,6 (6 is the largest 51 registers hold)
#   card51-L2   whole card, 16 waves per SIMD, L2: K = 1 D 0,2,6 and K = 8 D 0,2,6
#   break-one, break-card   the deliberate break: label L2, loads aimed at L0, K = 1 D 0; must move >= 90 % of the L2 - L0 gap
# Every job: before it, the card is checked (PCI device 0x7550, no RESTORE-FAILED file), at least 8 GiB of memory is available and the
# hotspot is waited down to 58 C (or the job is not started); it runs under nice -n 10 and a time limit; the binary itself waits
# for 58 C before EVERY launch (calibration launches included) and stops at 95 C; after it the kernel log is scanned for GPU faults.
# Environment: STUDY_GPU_LOCK (or local.env; without a lock there is no real run), LDM2_BIN (default build/round10/bin/ld_mat2),
#   LDM2_EXPECT (the expectation file, copied into the run folder before the first job; default <this folder>/expect-ld_mat2.md),
#   CARD_ENV (default runs/latest/card.env), ROCM_PATH, R10_BUDGET_S (no new job starts after this many seconds; default 2700),
#   R10_LOCK_WAIT_S (default 900), R10_HOT_WAIT_S (default 300).
# Exit: 0 all jobs ran and every control passed; 1 a job failed or a control failed; 2 refused before any GPU work (usage, card, lock
#   not configured, listing check, expectations missing); 6 the lock wait expired; 5 below 8 GiB; 4 too hot to start; 8 budget used
#   up (remaining jobs not started); 11 GPU fault found in the kernel log; 12 kernel log unreadable.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$H/../.." && pwd)
# shellcheck source=/dev/null
[[ -r $ROOT/local.env ]] && . "$ROOT/local.env"
ROCM_PATH=${ROCM_PATH:-$HOME/.local/opt/rocm}
LLVM=$ROCM_PATH/llvm/bin
BIN=${LDM2_BIN:-$ROOT/build/round10/bin/ld_mat2}
EXPECT=${LDM2_EXPECT:-$H/expect-ld_mat2.md}
CARD_ENV=${CARD_ENV:-$ROOT/runs/latest/card.env}
LOCK=${STUDY_GPU_LOCK:-}
BUDGET=${R10_BUDGET_S:-2700}
LOCK_WAIT=${R10_LOCK_WAIT_S:-900}
HOT_WAIT=${R10_HOT_WAIT_S:-300}
RESTORE_FAILED=${R10_RESTORE_FAILED:-${R9_RESTORE_FAILED:-$ROOT/runs/RESTORE-FAILED}}
HOT_LIMIT=58
MIN_AVAIL_KB=$((8 * 1024 * 1024))
FAULT_RE='(amdgpu|\[drm|gfxhub|mmhub).*(page fault|GPU reset|ring [A-Za-z0-9_.]+ timeout|VM_L2_PROTECTION|hang|\*ERROR\*)|amdgpu.*(timeout|fault)'

ONE_CELLS="ref,1:0,1:1,1:2,1:4,1:8,1:24,1:54,4:0,4:2,4:6,4:12,8:0,8:2,8:6,8:12"
# name|timeout s|arguments after the binary
JOBS=(
    "one-L0L2|420|--mode one --r 209 --levels L0,L2 --cells $ONE_CELLS --bursts 3 --ms 20"
    "card209-L0|600|--mode card --r 209 --levels L0 --cells ref,1:0,1:2,1:8,1:max --bursts 3 --ms 50"
    "card209-L2|900|--mode card --r 209 --levels L2 --cells ref,1:0,1:2,1:8,1:max,8:0,8:2,8:8,8:max --bursts 3 --ms 50"
    "card51-L0|450|--mode card --r 51 --levels L0 --cells ref,1:0,1:2,1:6 --bursts 3 --ms 50"
    "card51-L2|600|--mode card --r 51 --levels L2 --cells ref,1:0,1:2,1:6,8:0,8:2,8:6 --bursts 3 --ms 50"
    "break-one|240|--mode one --r 209 --levels L2 --aim L0 --cells 1:0 --bursts 3 --ms 20"
    "break-card|300|--mode card --r 209 --levels L2 --aim L0 --cells 1:0 --bursts 3 --ms 50"
)

# ---- gates (each takes its inputs as arguments, so --cpu-checks can feed them fake files) --------------------------------
say() { printf '%s %s\n' "$(date -u +%T)" "$*"; }

# gate_card <drm device dir>: 0 only if vendor 0x1002 and device 0x7550 (the RX 9070 XT)
gate_card() {
    local v d
    { v=$(<"$1/vendor"); } 2>/dev/null || return 1
    { d=$(<"$1/device"); } 2>/dev/null || return 1
    [[ $v == 0x1002 && $d == 0x7550 ]]
}
# gate_ram <meminfo file>: 0 only if MemAvailable is at least 8 GiB
gate_ram() {
    local kb
    kb=$(awk '/^MemAvailable:/ {print $2}' "$1" 2>/dev/null)
    [[ $kb =~ ^[0-9]+$ ]] && ((kb >= MIN_AVAIL_KB))
}
# hot_milli <hwmon dir>: the hotspot in millidegrees on stdout; fails if unreadable or not a number
hot_milli() {
    local t
    { t=$(<"$1/temp2_input"); } 2>/dev/null || return 1
    [[ $t =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$t"
}
# wait_cool <hwmon dir> <limit C> <max wait s> [poll s]: 0 when the hotspot reads at or below the limit; 1 on timeout or unreadable
wait_cool() {
    local t=0 waited=0 poll=${4:-1}
    while :; do
        t=$(hot_milli "$1") || return 1
        ((t <= $2 * 1000)) && return 0
        ((waited >= $3)) && return 1
        sleep "$poll"
        waited=$((waited + poll))
    done
}
# fault_scan <text file>: 0 if no GPU fault line, 1 if one is found (the lines are printed)
fault_scan() {
    ! grep -E -i "$FAULT_RE" "$1"
}
# lock_acquire <lock file> <wait s>: opens it on descriptor 9 of the calling shell and takes it; 0 or 6
lock_acquire() {
    exec 9>"$1" || return 2
    flock -w "$2" 9 || return 6
}
# break_fraction <L0 value> <L2 value> <break value>: how much of the L2 - L0 gap the aimed-at-L0 break closed; fails if below 0.9
break_fraction() {
    awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN { if (b - a <= 0) { print "gap<=0"; exit 1 } f = (b - c) / (b - a); printf "%.3f\n", f; exit !(f >= 0.9) }'
}
# cell_med <log> <mode> <R> <level> <aimed> <K> <D>: the median-of-bursts of one cell line, or nothing
cell_med() {
    awk -v mode="$2" -v r="$3" -v lv="$4" -v am="$5" -v k="$6" -v d="$7" '
        $1 == "RESULT" && $2 == "ld_mat2" && $3 == "cell" {
            for (i = 4; i < NF; i += 2) v[$i] = $(i + 1)
            if (v["mode"] == mode && v["R"] == r && v["level"] == lv && v["aimed"] == am && v["K"] == k && v["D"] == d) { print v["cyc_per_mm_median_of_bursts"]; exit }
        }' "$1"
}
# card_env_load: sets DRM_DEVICE, HWMON, HIP_INDEX from the card record; 0 if usable
card_env_load() {
    [[ -r $CARD_ENV ]] || return 1
    # shellcheck source=/dev/null
    . "$CARD_ENV"
    [[ ${ARCH:-} == gfx1201 && -d ${DRM_DEVICE:-} && -d ${HWMON:-} && ${HIP_INDEX:-} =~ ^[0-9]+$ ]]
}
mclk_label() { awk '/\*/ {gsub(/[ *]/, ""); print; exit}' "$DRM_DEVICE/pp_dpm_mclk" 2>/dev/null; }

# the listing of the binary that will run, and its check (CPU only). out: <dir>/ld_mat2.s, notes, check output
listing_check() {
    local dir=$1 tmp
    tmp=$dir/extract
    rm -rf "$tmp" && mkdir -p "$tmp" || return 1
    cp "$BIN" "$tmp/ld_mat2" || return 1
    (cd "$tmp" && "$LLVM/llvm-objdump" --offloading ld_mat2 >/dev/null 2>&1) || return 1
    local co
    co=$(ls "$tmp"/ld_mat2.0.hip-amdgcn-amd-amdhsa--gfx1201 2>/dev/null) || return 1
    "$LLVM/llvm-objdump" -d --mcpu=gfx1201 "$co" >"$dir/ld_mat2.s" 2>&1 || return 1
    "$LLVM/llvm-readelf" --notes "$co" >"$dir/notes.txt" 2>&1 || return 1
    sha256sum "$BIN" >"$dir/sha256.txt"
    python3 -I "$H/check_ld_mat2.py" check "$dir/ld_mat2.s" --notes "$dir/notes.txt" >"$dir/check-ld_mat2.txt" 2>&1
}

# cell_name <K> <D> <R> <level>: the block name the binary picks (find_cell in ld_mat2.hip): the fixed variant when the window
# (NMM/K)*256 is at least the L0 span of 8192 bytes; NMM is 512 for K = 32, else 256
cell_name() {
    local nmm=256 nm="k_k$1_d$2_r$3"
    (($1 == 32)) && nmm=512
    if [[ $4 == L0 ]] && (((nmm / $1) * 256 >= 8192)); then nm+="_f"; fi
    printf '%s\n' "$nm"
}

# plan_check <listing .s>: every K:D of every job names a block the binary holds (the fixed variant at L0 when the window is >= 8 KiB)
plan_check() {
    local dis=$1 bad=0 j name tmo args mode r levels cells lv spec K D nm
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name tmo args <<<"$j"
        r=209
        [[ $args == *"--r 51"* ]] && r=51
        levels=$(sed -E 's/.*--levels ([^ ]+).*/\1/' <<<"$args")
        cells=$(sed -E 's/.*--cells ([^ ]+).*/\1/' <<<"$args")
        for lv in ${levels//,/ }; do
            [[ $args == *"--aim "* ]] && lv=$(sed -E 's/.*--aim ([^ ]+).*/\1/' <<<"$args")
            for spec in ${cells//,/ }; do
                [[ $spec == ref ]] && { grep -q "k_ref_r${r}E" "$dis" || { echo "plan: no k_ref_r$r in the listing"; bad=1; }; continue; }
                K=${spec%%:*}
                D=${spec##*:}
                if [[ $D == max ]]; then
                    D=$(grep -o "k_k${K}_d[0-9]*_r${r}E" "$dis" | sed -E "s/k_k${K}_d([0-9]+)_r.*/\1/" | sort -n | tail -1)
                fi
                nm=$(cell_name "$K" "$D" "$r" "$lv")
                grep -q "${nm}E" "$dis" || { echo "plan: job $name cell $spec level $lv needs $nm: not in the listing"; bad=1; }
            done
        done
    done
    return $bad
}

print_plan() {
    local j name tmo args n=0 tot=0
    echo "ld_mat2, round 10 item 1, no-pin cells: L0 and L2 only (MALL and GDDR6 wait for the pinned item 2a)"
    echo "binary    $BIN"
    echo "lock      ${LOCK:-(not configured: a real run refuses; set STUDY_GPU_LOCK or local.env)}"
    echo "expect    $EXPECT"
    echo "budget    no new job after ${BUDGET} s; each job under nice -n 10 and its own time limit; the binary waits for <= ${HOT_LIMIT} C before every launch"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name tmo args <<<"$j"
        n=$((n + 1))
        tot=$((tot + tmo))
        printf '  %d. %-11s limit %4d s   %s\n' "$n" "$name" "$tmo" "$args"
    done
    echo "sum of the time limits: $tot s (upper bound; the budget stops new jobs after $BUDGET s, so the hard worst case is $BUDGET + the last limit <= 3600 s)"
    echo "launches: one-wave job 31 cells x 6 launches of 20 ms (a few seconds of load); whole card 27 cells x (3 timed bursts of 50 ms + 3 short calibration launches)"
    echo "estimate: about 15 to 25 minutes (81 whole-card timed bursts, about 8 s each with the 58 C wait, plus 7 gates of about 15 s); GPU load itself under 10 s"
    echo "not run here: MALL and GDDR6 cells, K = 4 and K = 32 on the whole card, R 90 and 167 (round 9: same shape as R 51 and 209)"
}

# ---- modes --------------------------------------------------------------------------------------------------------------
cpu_checks() {
    local dir=${1:-$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-r10-ldmat2-cpu} fails=0 T
    mkdir -p "$dir" && T=$dir/fake && rm -rf "$T" && mkdir -p "$T" || { echo "cannot make $dir"; return 2; }
    ok() { echo "PASS $*"; }
    bad() { echo "FAIL $*"; fails=$((fails + 1)); }
    t_ok() { local m=$1; shift; if "$@" >/dev/null 2>&1; then ok "$m"; else bad "$m"; fi; }
    t_no() { local m=$1; shift; if "$@" >/dev/null 2>&1; then bad "$m (the deliberate break was NOT caught)"; else ok "$m"; fi; }
    echo "## tools"
    for c in flock nice timeout python3 awk journalctl od sha256sum g++; do command -v "$c" >/dev/null && ok "tool $c" || bad "tool $c missing"; done
    bash -n "$H/run_ld_mat2_nopin.sh" && ok "bash -n" || bad "bash -n"
    echo "## the card, read only (the real sysfs files are read, nothing is written)"
    if card_env_load; then
        t_ok "card record $CARD_ENV: $DRM_DEVICE reads vendor/device 0x1002/0x7550" gate_card "$DRM_DEVICE"
        t_ok "hotspot readable ($(hot_milli "$HWMON" 2>/dev/null) m C now)" hot_milli "$HWMON"
        echo "     memory clock label now: $(mclk_label)"
    else
        bad "card record $CARD_ENV unusable"
    fi
    [[ -e $RESTORE_FAILED ]] && bad "RESTORE-FAILED exists: $RESTORE_FAILED" || ok "no RESTORE-FAILED file"
    echo "## gates against fake files, with deliberate breaks"
    mkdir -p "$T/gpu_ok" "$T/gpu_bad"
    printf '0x1002\n' >"$T/gpu_ok/vendor"; printf '0x7550\n' >"$T/gpu_ok/device"
    printf '0x1002\n' >"$T/gpu_bad/vendor"; printf '0x744c\n' >"$T/gpu_bad/device"
    t_ok "card gate accepts 0x7550" gate_card "$T/gpu_ok"
    t_no "card gate refuses 0x744c" gate_card "$T/gpu_bad"
    t_no "card gate refuses a missing device" gate_card "$T/none"
    printf 'MemAvailable:   16777216 kB\n' >"$T/mem_ok"; printf 'MemAvailable:    6291456 kB\n' >"$T/mem_low"
    printf 'MemAvailable:    8388608 kB\n' >"$T/mem_edge"; printf 'MemAvailable:    8388607 kB\n' >"$T/mem_under"; printf 'junk\n' >"$T/mem_junk"
    t_ok "RAM gate accepts 16 GiB" gate_ram "$T/mem_ok"
    t_ok "RAM gate accepts exactly 8 GiB" gate_ram "$T/mem_edge"
    t_no "RAM gate refuses 8 GiB - 1 kB" gate_ram "$T/mem_under"
    t_no "RAM gate refuses 6 GiB" gate_ram "$T/mem_low"
    t_no "RAM gate refuses an unreadable meminfo" gate_ram "$T/mem_junk"
    mkdir -p "$T/hw"
    printf '50000\n' >"$T/hw/temp2_input"; t_ok "thermal wait accepts 50 C at once" wait_cool "$T/hw" $HOT_LIMIT 2 1
    printf '58000\n' >"$T/hw/temp2_input"; t_ok "thermal wait accepts exactly 58 C" wait_cool "$T/hw" $HOT_LIMIT 2 1
    printf '58001\n' >"$T/hw/temp2_input"; t_no "thermal wait refuses 58.001 C after its wait" wait_cool "$T/hw" $HOT_LIMIT 2 1
    printf '70000\n' >"$T/hw/temp2_input"; t_no "thermal wait refuses 70 C after its wait" wait_cool "$T/hw" $HOT_LIMIT 2 1
    printf 'x\n' >"$T/hw/temp2_input"; t_no "thermal wait refuses an unreadable value" wait_cool "$T/hw" $HOT_LIMIT 2 1
    t_no "thermal wait refuses a missing sensor" wait_cool "$T/none" $HOT_LIMIT 2 1
    printf '70000\n' >"$T/hw/temp2_input"
    (sleep 2; printf '50000\n' >"$T/hw/temp2_input") &
    local cp=$! t0 t1
    t0=$SECONDS
    if wait_cool "$T/hw" $HOT_LIMIT 10 1; then t1=$((SECONDS - t0)); ((t1 >= 1)) && ok "thermal wait really waits (cooled after ${t1} s)" || bad "thermal wait returned without waiting"; else bad "thermal wait did not see the sensor cool"; fi
    wait "$cp"
    printf 'Nov 26 03:00:00 h kernel: usb 8-1: cannot submit urb 0, error -19: no device\n' >"$T/log_clean"
    printf 'Nov 26 03:00:01 h kernel: amdgpu 0000:03:00.0: amdgpu: GPU reset begin!\n' >"$T/log_reset"
    printf 'Nov 26 03:00:02 h kernel: amdgpu 0000:03:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32777)\n' >"$T/log_pf"
    printf 'Nov 26 03:00:03 h kernel: amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1\n' >"$T/log_ring"
    t_ok "fault scan: a USB line is not a GPU fault" fault_scan "$T/log_clean"
    t_no "fault scan catches a GPU reset" fault_scan "$T/log_reset"
    t_no "fault scan catches a page fault" fault_scan "$T/log_pf"
    t_no "fault scan catches a ring timeout" fault_scan "$T/log_ring"
    if journalctl -k -n 1 --no-pager >/dev/null 2>&1; then ok "kernel log readable with journalctl -k"; else bad "journalctl -k not readable (a real run would exit 12)"; fi
    echo "## lock (a private temp lock; the shared lock is not touched)"
    : >"$T/lock"
    (exec 8>"$T/lock"; flock -n 8; sleep 30) &
    local hp=$!
    sleep 0.5
    t_no "lock wait times out while another holder has the lock" bash -c "$(declare -f lock_acquire); lock_acquire '$T/lock' 1"
    kill "$hp" 2>/dev/null; wait "$hp" 2>/dev/null
    t_ok "lock is taken once the holder is gone" bash -c "$(declare -f lock_acquire); lock_acquire '$T/lock' 1"
    [[ -n $LOCK ]] && ok "shared lock configured: $LOCK" || bad "no shared lock configured (STUDY_GPU_LOCK or local.env): a real run would refuse"
    echo "## break fraction (control 6)"
    t_ok "break closes 96 % of the gap (L0 25, L2 100, break 28)" break_fraction 25 100 28
    t_no "break closing 27 % of the gap is refused (break 80)" break_fraction 25 100 80
    t_no "break with L2 not above L0 is refused" break_fraction 100 25 30
    echo "## expectations"
    [[ -r $EXPECT ]] && ok "expectation file $EXPECT" || bad "expectation file $EXPECT missing"
    echo "## the binary that would run"
    if [[ -x $BIN ]]; then
        ok "binary $BIN ($(sha256sum "$BIN" | cut -c1-16))"
        if listing_check "$dir"; then ok "listing check of the binary's own code object: $(tail -1 "$dir/check-ld_mat2.txt")"; else bad "listing check failed, see $dir/check-ld_mat2.txt"; fi
        if python3 -I "$H/check_ld_mat2.py" selftest "$dir/ld_mat2.s" "$dir/breaks" --notes "$dir/notes.txt" >"$dir/selftest-ld_mat2.txt" 2>&1; then ok "listing check self-test: $(tail -1 "$dir/selftest-ld_mat2.txt")"; else bad "listing self-test failed, see $dir/selftest-ld_mat2.txt"; fi
        if plan_check "$dir/ld_mat2.s"; then ok "every cell of every job is a block of the binary"; else bad "a job names a cell the binary does not hold"; fi
        sed -E 's/(k_k8_d[0-9]+_r[0-9]+)_f(E)/\1\2/' "$dir/ld_mat2.s" >"$dir/ld_mat2.nof8.s"
        t_no "deliberate break: plan_check refuses a listing whose K=8 fixed (_f) blocks are removed" plan_check "$dir/ld_mat2.nof8.s"
    else
        bad "no executable at $BIN"
    fi
    echo "## block-name rule (window == span boundary)"
    [[ $(cell_name 8 0 209 L0) == k_k8_d0_r209_f ]] && ok "K=8 at L0 (window == span 8192) resolves to the fixed block" || bad "K=8 at L0 must resolve to k_k8_d0_r209_f"
    [[ $(cell_name 8 0 209 L2) == k_k8_d0_r209 ]] && ok "K=8 at L2 resolves to the walking block" || bad "K=8 at L2 must be the walking block"
    [[ $(cell_name 1 2 51 L0) == k_k1_d2_r51_f ]] && ok "K=1 at L0 resolves to the fixed block" || bad "K=1 at L0 must be fixed"
    [[ $(cell_name 32 0 209 L0) == k_k32_d0_r209 ]] && ok "K=32 at L0 (window 4096 < span) resolves to the walking block" || bad "K=32 at L0 must be the walking block"
    echo "## analysis self-test (plain C++, no GPU)"
    if g++ -std=c++17 -O1 -DLDM2_CPU_ONLY -x c++ "$ROOT/src/round10/ld_mat2.hip" -o "$dir/analysis_selftest" 2>"$dir/analysis_selftest.build"; then
        "$dir/analysis_selftest" >"$dir/analysis_selftest.txt" 2>&1 && ok "analysis: $(tail -1 "$dir/analysis_selftest.txt")" || bad "analysis self-test failed, see $dir/analysis_selftest.txt"
    else
        bad "analysis self-test did not build, see $dir/analysis_selftest.build"
    fi
    echo "## the plan"
    print_plan | sed 's/^/     /'
    echo "cpu-checks: $((fails)) failed"
    ((fails == 0))
}

real_run() {
    local RUN=${1:-$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-r10-ldmat2}
    [[ -n $LOCK ]] || { echo "no shared GPU lock configured (STUDY_GPU_LOCK or local.env): refused"; return 2; }
    [[ -x $BIN ]] || { echo "no binary at $BIN: refused"; return 2; }
    [[ -r $EXPECT ]] || { echo "no expectation file at $EXPECT: write the expectations before the first GPU job"; return 2; }
    card_env_load || { echo "card record $CARD_ENV unusable: refused"; return 2; }
    gate_card "$DRM_DEVICE" || { echo "$DRM_DEVICE is not the RX 9070 XT (0x1002:0x7550): refused"; return 2; }
    [[ ! -e $RESTORE_FAILED ]] || { echo "$RESTORE_FAILED exists: a memory-clock restore failed; check by hand first"; return 2; }
    mkdir -p "$RUN" || return 2
    export DRM_DEVICE HWMON HIP_VISIBLE_DEVICES=$HIP_INDEX
    { echo "<!-- copied $(date -u +%FT%TZ) before the first GPU job; git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) -->"; cat "$EXPECT"; } >"$RUN/expect-ld_mat2.md"
    listing_check "$RUN" || { echo "listing check of the binary FAILED (see $RUN/check-ld_mat2.txt): no GPU job"; return 2; }
    tail -1 "$RUN/check-ld_mat2.txt"
    python3 -I "$H/check_ld_mat2.py" selftest "$RUN/ld_mat2.s" "$RUN/breaks" --notes "$RUN/notes.txt" >"$RUN/selftest-ld_mat2.txt" 2>&1 || { echo "listing self-test FAILED: no GPU job"; return 2; }
    plan_check "$RUN/ld_mat2.s" || { echo "plan names a cell the binary does not hold: no GPU job"; return 2; }
    print_plan >"$RUN/plan.txt"
    # the shared lock: held for the whole run
    lock_acquire "$LOCK" "$LOCK_WAIT" || { echo "could not take $LOCK within $LOCK_WAIT s"; return 6; }
    say "STARTS ld_mat2 nopin, lock $LOCK, run folder $RUN"
    local start=$SECONDS rc=0 j name tmo args jr t0 kl
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name tmo args <<<"$j"
        if ((SECONDS - start > BUDGET)); then say "budget of $BUDGET s used: $name and the rest not started"; rc=8; break; fi
        [[ ! -e $RESTORE_FAILED ]] || { say "RESTORE-FAILED appeared: stopping"; rc=2; break; }
        gate_card "$DRM_DEVICE" || { say "card check failed before $name"; rc=2; break; }
        gate_ram /proc/meminfo || { say "below 8 GiB available before $name"; rc=5; break; }
        wait_cool "$HWMON" $HOT_LIMIT "$HOT_WAIT" 2 || { say "hotspot did not reach <= $HOT_LIMIT C within $HOT_WAIT s before $name (or unreadable): not started"; rc=4; break; }
        t0=$(date +%s)
        say "start $name (limit $tmo s) mclk $(mclk_label)"
        # shellcheck disable=SC2086
        timeout -k 10 "$tmo" nice -n 10 "$BIN" $args >"$RUN/$name.log" 2>&1
        jr=$?
        say "end $name exit $jr, mclk $(mclk_label), hotspot $(hot_milli "$HWMON") m C"
        echo "job $name exit $jr started $(date -u -d @"$t0" +%T) args: $args" >>"$RUN/jobs.txt"
        kl=$RUN/$name.kernlog
        if journalctl -k --since "@$t0" --no-pager >"$kl" 2>&1; then
            if ! fault_scan "$kl" >"$RUN/$name.faults"; then say "GPU FAULT in the kernel log during $name (see $RUN/$name.faults)"; rc=11; break; fi
        else
            say "kernel log unreadable after $name"; rc=12; break
        fi
        ((jr != 0)) && { say "$name failed (exit $jr); stopping"; rc=1; break; }
    done
    # controls over the logs: every CONTROL line, and the break fractions
    grep -h '^CONTROL' "$RUN"/*.log >"$RUN/controls.txt" 2>/dev/null
    local m0 m2 mb f t mode l0log l2log blog r
    for t in "one one-L0L2 one-L0L2 break-one 209" "card card209-L0 card209-L2 break-card 209"; do
        read -r mode l0log l2log blog r <<<"$t"
        m0=$(cell_med "$RUN/$l0log.log" "$mode" "$r" L0 L0 1 0); m2=$(cell_med "$RUN/$l2log.log" "$mode" "$r" L2 L2 1 0); mb=$(cell_med "$RUN/$blog.log" "$mode" "$r" L2 L0 1 0)
        if [[ -n $m0 && -n $m2 && -n $mb ]]; then
            if f=$(break_fraction "$m0" "$m2" "$mb"); then
                echo "CONTROL c6_aim_break_$mode PASS the break moved $f of the L2 - L0 gap (L0 $m0, L2 $m2, break $mb)" | tee -a "$RUN/controls.txt"
            else
                echo "CONTROL c6_aim_break_$mode FAIL the break moved $f of the gap (L0 $m0, L2 $m2, break $mb)" | tee -a "$RUN/controls.txt"
            fi
        else
            echo "CONTROL c6_aim_break_$mode FAIL missing cell (L0 '$m0' L2 '$m2' break '$mb'; a job did not run)" | tee -a "$RUN/controls.txt"
        fi
    done
    grep -q ' FAIL' "$RUN/controls.txt" && ((rc == 0)) && rc=1
    echo "controls: $(grep -c ' PASS' "$RUN/controls.txt") PASS, $(grep -c ' FAIL' "$RUN/controls.txt") FAIL (all lines in $RUN/controls.txt)"
    say "ENDS ld_mat2 nopin exit $rc after $((SECONDS - start)) s"
    return $rc
}

case "${1:-}" in
    --plan) print_plan ;;
    --cpu-checks) cpu_checks "${2:-}" ;;
    -h | --help) sed -n '2,32p' "$0" ;;
    --*) echo "unknown option $1"; exit 2 ;;
    *) real_run "${1:-}" ;;
esac
