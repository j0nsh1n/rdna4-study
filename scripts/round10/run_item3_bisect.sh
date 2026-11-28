#!/usr/bin/env bash
# Round 10 item 3: Vulkan against HIP cold reads, the two-way bisect (steps 2-6 of measurements plan item 3; step 1 only with --pin).
#
#   Step 1  pinned memory clock rerun of the two baselines (needs --pin <level> and the owner's sudo approval for pin_mclk.sh; with
#           --no-pin, the default, it is skipped and said so on the first line of the run log and of the analysis)
#   Step 2  read, no kernels: heaps and memory types, the type chosen for every placement, the machine code of every Vulkan pipeline of the item
#           (vkr10 info creates the pipelines and dispatches nothing), HIP allocation kinds (r10_cl3 info)
#   Step 3  one factor at a time in both directions: form (loop and load width), allocation layout, eviction size and form, scope modifiers,
#           wave size (HIP wave64 build), 3 process repeats, random and page walks, cold and fused cells
#   Step 4  placement: Vulkan device-local / BAR-visible / system memory; HIP device / uncached / pinned host (the deliberate break)
#   Step 5  strides 128 B, 4 KiB, 64 KiB, 2 MiB over a 2 GiB region and the reach test (N 4 KiB pages walked twice)
#   Step 6  eviction: no eviction (the deliberate break) and a fresh never-read region for every repetition
#   Steps 2-6 run as ONE locked window; the shared lock is held from the first check to the journal check, nothing else touches the card.
#
# Usage: run_item3_bisect.sh [--plan] [--cpu-checks] [--no-pin | --pin LEVEL] [--steps 2-6|1|all] [--out DIR]
#   --plan         print the jobs, cells, guards and the estimated GPU time; no lock, no GPU
#   --cpu-checks   no GPU: bash -n, the plan, cell-spec validation, the analysis self-test, the gate's refusals on stand-in data (wrong card, low
#                  memory, hot card, held lock, a kernel-log fault) and, when built, the listing checks
# Environment: STUDY_GPU_LOCK (default <repo>/runs/gpu.lock), R10_ITEM3_BIN (build folder, default
#   <repo>/build/round10/item3; build it with src/round10/item3/build.sh), R10_RUN (run folder, default <repo>/runs/<UTC>-r10i3),
#   R10_LOCK_WAIT (s, default 900), R10_JOB_TIMEOUT (s per job, default 150), R10_HOLD (hold file, default <repo>/runs/HOLD-FSR),
#   R10_RESTORE_FAILED (default <repo>/runs/RESTORE-FAILED), CARD_ENV (card record, default <repo>/runs/latest/card.env).
# Gate of every job: the lock, nice -n 10, a time limit, the card found by PCI device id 0x7550 (never by index alone), at least 8 GiB available memory,
#   a 10 GB / no swap systemd scope, the hotspot at or below 58 C (waits up to 5 min, else exit 4), a 95 C guard (TERM then KILL of the job process group when that pid is the leader, and of its systemd user scope), no HOLD or
#   RESTORE-FAILED file, no game (checked by the hold file only: the owner places it), and afterwards `journalctl -k` is searched for GPU faults and resets.
# Exit: 0 ok; 2 usage or refusal before any job; 3 wrong card; 4 too hot to start; 5 memory below 8 GiB; 6 lock wait expired; 8 a job hit the 95 C guard;
#   9 no memory scope; 10 a job failed or timed out; 12 the kernel log cannot be read; 13 a GPU fault or reset in the kernel log; 14 the analysis failed a control.
# Nothing here writes sysfs, uses sudo or changes git. Step 1 calls scripts/round10/pin_mclk.sh, which refuses unless the owner sets its own switch by hand.
set -u
SELF=$(readlink -f "${BASH_SOURCE[0]}")
here=$(dirname "$SELF")
REPO=$(cd "$here/../.." && pwd)
SRC=$REPO/src/round10/item3
BIN=${R10_ITEM3_BIN:-$REPO/build/round10/item3}
LOCK=${STUDY_GPU_LOCK:-$REPO/runs/gpu.lock}
REAL_LOCK_DEFAULT=$REPO/runs/gpu.lock
HOLD=${R10_HOLD:-$REPO/runs/HOLD-FSR}
RESTORE_FAILED=${R10_RESTORE_FAILED:-${R9_RESTORE_FAILED:-$REPO/runs/RESTORE-FAILED}}
LOCK_WAIT=${R10_LOCK_WAIT:-900}
JOB_TMO=${R10_JOB_TIMEOUT:-150}
CARD_ENV=${CARD_ENV:-$REPO/runs/latest/card.env}
DEVID=0x7550
MIN_KB=$((8 * 1024 * 1024))
START_C=58
GUARD_C=95
FAULT_RE='amdgpu.*(fault|reset|timeout|hang|ring .* timed out)|VM_L2_PROTECTION|\*ERROR\*|gpu page fault'
PLAN=0 CPU=0 PIN="" STEPS=2-6 OUT=${R10_RUN:-} STEP1_PAYLOAD=0 SELFTEST_GATE=0
while (($#)); do
    case $1 in
        --plan) PLAN=1 ;;
        --cpu-checks) CPU=1 ;;
        --no-pin) PIN="" ;;
        --pin) PIN=${2:-}; shift ;;
        --steps) STEPS=${2:-}; shift ;;
        --out) OUT=${2:-}; shift ;;
        --step1-payload) STEP1_PAYLOAD=1 ;;
        --selftest-gate) SELFTEST_GATE=1 ;;
        *) echo "unknown option $1 (see the header of $SELF)"; exit 2 ;;
    esac
    shift
done
[[ $STEPS == 2-6 || $STEPS == 1 || $STEPS == all ]] || { echo "--steps takes 2-6, 1 or all"; exit 2; }
[[ -z $PIN || $PIN =~ ^[0-9]+$ ]] || { echo "--pin takes a pp_dpm_mclk level index"; exit 2; }
[[ $STEPS == 2-6 || -n $PIN ]] || { echo "step 1 needs --pin <level> (pinned memory clock; the owner's sudo approval comes first)"; exit 2; }
SELFTEST=${R10_SELFTEST:-0}

# ---------------------------------------------------------------------------------------------------------------------------------
# cells. Every cell name is <api>-<id>-<walk r|p>, which scripts the analysis (src/round10/item3/analyse_item3.py) reads.
VK_KEYS="name q form walk order reps steps alloc mem evict sg coh stride loop warm n"
HIP_KEYS="name walk order reps steps form alloc mem evict evform scope stride loop warm n"
vk_cells_base() {  # the two baselines, both walks (also step 1's whole payload)
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=vk-base-$w,walk=$W"
    done
}
hip_cells_base() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=hip-base-$w,walk=$W"
    done
}
vk_cells_3() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=vk-base-$w,walk=$W"
        echo "name=vk-baseg-$w,walk=$W,q=g"
        echo "name=vk-fbase-$w,walk=$W,order=fused"
        echo "name=vk-v1-$w,walk=$W,form=p64v"
        echo "name=vk-v1n-$w,walk=$W,form=p64"
        echo "name=vk-v1c-$w,walk=$W,form=p64,coh=1"
        echo "name=vk-vf1-$w,walk=$W,form=p64v,order=fused"
        echo "name=vk-v2-$w,walk=$W,sg=64"
        echo "name=vk-v3-$w,walk=$W,alloc=one"
        echo "name=vk-v3s-$w,walk=$W,alloc=sub"
        echo "name=vk-v3d-$w,walk=$W,alloc=ded"
        echo "name=vk-v4-$w,walk=$W,evict=512"
        echo "name=vk-v5-$w,walk=$W,form=p64v,alloc=one,evict=512"
    done
    echo "name=vk-loop8-r,walk=random,loop=8"
}
hip_cells_3() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=hip-base-$w,walk=$W"
        echo "name=hip-fbase-$w,walk=$W,order=fused"
        echo "name=hip-h1-$w,walk=$W,form=m64"
        echo "name=hip-h2-$w,walk=$W,form=b32"
        echo "name=hip-h1h2-$w,walk=$W,form=u32v"
        echo "name=hip-hf1-$w,walk=$W,form=u32v,order=fused"
        echo "name=hip-h4-$w,walk=$W,alloc=sep"
        echo "name=hip-h7-$w,walk=$W,alloc=sub"
        echo "name=hip-h5-$w,walk=$W,evict=256"
        echo "name=hip-h5p-$w,walk=$W,evict=256,evform=plain"
        echo "name=hip-h6-$w,walk=$W,form=u32v,alloc=sep,evict=256"
        echo "name=hip-hsd-$w,walk=$W,scope=dev"
        echo "name=hip-hss-$w,walk=$W,scope=sys"
    done
    echo "name=hip-loop8-r,walk=random,loop=8"
}
hip_cells_w64() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=hip-h3-$w,walk=$W"
        echo "name=hip-h6w-$w,walk=$W,form=u32v,alloc=sep,evict=256"
    done
}
vk_cells_4() {
    echo "name=vk-base-r,walk=random"
    echo "name=vk-bar-r,walk=random,mem=bar"
    echo "name=vk-gtt-r,walk=random,mem=gtt"
}
hip_cells_4() {
    echo "name=hip-base-r,walk=random"
    echo "name=hip-devunc-r,walk=random,mem=devunc"
    echo "name=hip-host-r,walk=random,mem=host"
    echo "name=hip-hostnc-r,walk=random,mem=hostnc"
}
STRIDES="st128:128 st4k:4096 st64k:65536 st2m:2097152"
vk_cells_5() {
    local s
    for s in $STRIDES; do echo "name=vk-${s%%:*}-p,walk=page,form=p64v,stride=${s##*:}"; done
}
hip_cells_5() {
    local s
    for s in $STRIDES; do echo "name=hip-${s%%:*}-p,walk=page,form=p64,stride=${s##*:}"; done
}
vk_cells_6() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=vk-noev-$w,walk=$W,order=noevict,evict=0"
    done
    echo "name=vk-fresh-r,walk=random,order=fresh"
}
hip_cells_6() {
    local W w
    for W in random page; do
        w=${W:0:1}
        echo "name=hip-noev-$w,walk=$W,order=noevict,evict=0"
    done
    echo "name=hip-fresh-r,walk=random,order=fresh"
}
# jobs: <name>|<binary>|<mode>|<cell function>|<step>|<process repeats>
JOBS=(
    "s2-vk-info|vkr10|info||2|1"
    "s2-hip-info|r10_cl3|info||2|1"
    "s3-vk|vkr10|bis|vk_cells_3|3|3"
    "s3-hip|r10_cl3|bis|hip_cells_3|3|3"
    "s3-hipw64|r10_cl3_w64|bis|hip_cells_w64|3|2"
    "s4-vk|vkr10|bis|vk_cells_4|4|1"
    "s4-hip|r10_cl3|bis|hip_cells_4|4|1"
    "s5-vk|vkr10|bis|vk_cells_5|5|2"
    "s5-hip|r10_cl3|bis|hip_cells_5|5|2"
    "s5-vk-reach|vkr10|reach|vk_reach|5|2"
    "s5-hip-reach|r10_cl3|reach|hip_reach|5|2"
    "s6-vk|vkr10|bis|vk_cells_6|6|2"
    "s6-hip|r10_cl3|bis|hip_cells_6|6|2"
)
vk_reach() { echo "name=vk-reach,reps=6"; }
hip_reach() { echo "name=hip-reach,reps=6"; }
# seconds of GPU load per job and repeat, measured against round 9 (a Vulkan cell is the 150 ms warm-up plus 20 x about 1 ms; a HIP cell 20 evicts of 512 MiB)
declare -A EST=([s2-vk-info]=1 [s2-hip-info]=1 [s3-vk]=14 [s3-hip]=12 [s3-hipw64]=3 [s4-vk]=2 [s4-hip]=3 [s5-vk]=6 [s5-hip]=6 [s5-vk-reach]=3 [s5-hip-reach]=3 [s6-vk]=3 [s6-hip]=3)

in_steps() {  # in_steps <step>
    case $STEPS in
        2-6) [[ $1 -ge 2 ]] ;;
        1) [[ $1 -eq 1 ]] ;;
        all) true ;;
    esac
}

cell_args() {  # cell_args <function>  -> the cells, one per line
    "$1"
}

validate_cells() {  # every key of every cell is one the binary parses, names are unique and well formed
    local bad=0 j name bin mode fn step rep line k keys names=""
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name bin mode fn step rep <<<"$j"
        [[ -n $fn ]] || continue
        keys=$VK_KEYS
        [[ $bin == r10_cl3* ]] && keys=$HIP_KEYS
        while IFS= read -r line; do
            [[ $line =~ ^name=(vk|hip)-[a-z0-9]+-[rp](,|$) ]] || [[ $mode == reach && $line =~ ^name= ]] || { echo "bad cell name: $line"; bad=1; }
            [[ $bin == vkr10 && $line == name=hip-* ]] && { echo "Vulkan job with a HIP cell: $line"; bad=1; }
            [[ $bin == r10_cl3* && $line == name=vk-* ]] && { echo "HIP job with a Vulkan cell: $line"; bad=1; }
            IFS=, read -ra kvs <<<"$line"
            for k in "${kvs[@]}"; do
                [[ $k == *=* ]] || { echo "not key=value: $k in $line"; bad=1; continue; }
                [[ " $keys " == *" ${k%%=*} "* ]] || { echo "key ${k%%=*} is not parsed by $bin: $line"; bad=1; }
            done
            if [[ $mode == bis ]]; then  # a name appears once per job
                [[ $names == *"|$name:${line%%,*}|"* ]] && { echo "duplicate cell in job $name: ${line%%,*}"; bad=1; }
                names="$names|$name:${line%%,*}|"
            fi
        done < <("$fn")
    done
    return $bad
}

print_plan() {
    local j name bin mode fn step rep total=0 n
    echo "PLAN item 3 (steps $STEPS; $([[ -n $PIN ]] && echo "step 1 pinned at mclk level $PIN via scripts/round10/pin_mclk.sh" || echo "step 1 SKIPPED: --no-pin, the memory clock is the card's own; per-repetition uclk labels and r9_sampler record it"))"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r name bin mode fn step rep <<<"$j"
        in_steps "$step" || continue
        n=0
        [[ -n $fn ]] && n=$("$fn" | wc -l)
        echo "  job $name x$rep: $bin $mode, $n cells, about ${EST[$name]} s of load per repeat (step $step)"
        total=$((total + ${EST[$name]} * rep))
        if [[ -n $fn && ${VERBOSE:-0} == 1 ]]; then "$fn" | sed 's/^/      /'; fi
    done
    echo "  estimated GPU load $total s (+ about 2 s startup and a cool-down check per job; the locked window is about $((total * 2 + 60)) s; 3-4 minutes of load is the plan's figure)"
    echo "  guards: lock $LOCK (wait $LOCK_WAIT s), nice -n 10, timeout $JOB_TMO s per job, card $DEVID, >= 8 GiB available, scope MemoryMax=10G, start <= ${START_C} C, stop at ${GUARD_C} C, kernel log searched after every job"
}

# ---------------------------------------------------------------------------------------------------------------------------------
# gate pieces (each reads sysfs or /proc only; the *_FILE hooks exist for the self-test and are honoured only when R10_SELFTEST=1)
card_find() {  # sets DRM_DEVICE, HWMON, HIP_INDEX; exit 3 if no card has device id 0x7550
    local d id
    DRM_DEVICE="" HWMON=""
    for d in /sys/class/drm/card[0-9]*/device; do
        [[ -f $d/device ]] || continue
        id=$(<"$d/device")
        [[ $SELFTEST == 1 && -n ${R10_TEST_DEVID:-} ]] && id=$R10_TEST_DEVID
        if [[ $id == "$DEVID" ]]; then
            DRM_DEVICE=$(readlink -f "$d")
            HWMON=$(ls -d "$DRM_DEVICE"/hwmon/hwmon* 2>/dev/null | head -1)
            break
        fi
    done
    [[ -n $DRM_DEVICE && -n $HWMON ]] || return 3
    HIP_INDEX=0
    if [[ -r $CARD_ENV ]]; then
        # shellcheck disable=SC1090
        local hi
        hi=$(sed -n 's/^HIP_INDEX=//p' "$CARD_ENV" | head -1)
        [[ $hi =~ ^[0-9]+$ ]] && HIP_INDEX=$hi
    fi
    return 0
}
hotspot_c() {
    local f=$HWMON/temp2_input
    [[ $SELFTEST == 1 && -n ${R10_TEST_HOT_FILE:-} ]] && f=$R10_TEST_HOT_FILE
    local m
    m=$(cat "$f" 2>/dev/null) || { echo 999; return; }
    [[ $m =~ ^[0-9]+$ ]] || { echo 999; return; }
    echo $((m / 1000))
}
mem_avail_kb() {
    local f=/proc/meminfo
    [[ $SELFTEST == 1 && -n ${R10_TEST_MEMINFO:-} ]] && f=$R10_TEST_MEMINFO
    awk '/^MemAvailable:/ {print $2}' "$f"
}
wait_cool() {  # wait_cool <what>: up to 5 min below START_C; exit 4 otherwise
    local w=0 t
    while t=$(hotspot_c); ((t > START_C)); do
        ((w >= ${R10_COOL_WAIT:-300})) && { echo "REFUSED $1: hotspot $t C after ${w}s (start limit $START_C C)"; return 4; }
        sleep 1
        w=$((w + 1))
    done
    ((w > 0)) && echo "$(date -u +%T) cooled $w s before $1 (now $t C)"
    return 0
}
kernel_faults() {  # kernel_faults <epoch>: lines of the kernel log since then that look like a GPU fault or reset. 12 if the log cannot be read.
    local out rc
    if [[ $SELFTEST == 1 && -n ${R10_TEST_JOURNAL:-} ]]; then
        if [[ ${R10_TEST_JOURNAL_FAIL:-0} == 1 ]]; then
            echo "journal read failed" >&2
            return 12
        fi
        grep -Ei "$FAULT_RE" "$R10_TEST_JOURNAL" || true
        return 0
    fi
    out=$(journalctl -k --since "@$1" --no-pager -q 2>"$RUN/journal.stderr")
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "journal read failed (rc=$rc)" >&2
        return 12
    fi
    grep -Ei "$FAULT_RE" <<< "$out" || true
    return 0
}
journal_readable() {
    [[ $SELFTEST == 1 && -n ${R10_TEST_JOURNAL:-} ]] && return 0
    journalctl -k -n 1 --no-pager -q >/dev/null 2>&1
}
check_start() {  # the checks that need no lock and no waiting
    card_find || { echo "REFUSED: no card with PCI device id $DEVID under /sys/class/drm (wrong GPU installed?)"; return 3; }
    local kb
    kb=$(mem_avail_kb)
    [[ $kb =~ ^[0-9]+$ ]] && ((kb >= MIN_KB)) || { echo "REFUSED: ${kb:-?} kB available, need $MIN_KB kB (8 GiB)"; return 5; }
    [[ ! -e $RESTORE_FAILED ]] || { echo "REFUSED: $RESTORE_FAILED exists (a memory-clock restore failed; check power_dpm_force_performance_level and pp_dpm_mclk by hand, then remove it)"; return 2; }
    journal_readable || { echo "REFUSED: the kernel log cannot be read (journalctl -k): the fault check after the run would be blind (add the user to systemd-journal or run where it works)"; return 12; }
    return 0
}

take_lock() {
    mkdir -p "$(dirname "$LOCK")" 2>/dev/null
    exec 9>>"$LOCK" || { echo "cannot open the lock $LOCK"; return 2; }
    flock -w "$LOCK_WAIT" 9 || { echo "REFUSED: lock $LOCK held by another job for more than $LOCK_WAIT s"; return 6; }
    return 0
}

# Stop a thermal-limit job by its process group (only when this pid is the leader) and, if a user scope was used, by that unit.
stop_payload() { # stop_payload <pid> <unit or empty> <TERM|KILL>
    local pid=$1 unit=$2 sig=$3 pgid rest pp d
    if [[ -n $unit ]]; then
        systemctl --user kill "--signal=$sig" --kill-whom=all "${unit}.scope" 2>/dev/null || true
    fi
    pgid=$(sed 's/^.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $3}')
    if [[ -n $pgid && $pgid == "$pid" ]]; then
        kill "-$sig" -- "-$pid" 2>/dev/null || true
    fi
    for d in /proc/[0-9]*; do
        rest=$(sed 's/^.*) //' "$d/stat" 2>/dev/null) || continue
        pp=${rest#* }
        pp=${pp%% *}
        [[ $pp == "$pid" ]] && kill "-$sig" "${d#/proc/}" 2>/dev/null || true
    done
    kill "-$sig" "$pid" 2>/dev/null || true
}

JOBS_FAILED=0
run_job() {  # run_job <name> <binary> <args...>: one gated, guarded job; output in $RUN/<name>.out
    local name=$1 bin=$2
    shift 2
    local rc t0 jobpid guardpid wp scope=() unit=""
    while [[ -e $HOLD ]]; do echo "$(date -u +%T) HOLD file $HOLD present, waiting"; sleep 5; done
    [[ ! -e $RESTORE_FAILED ]] || { echo "REFUSED $name: $RESTORE_FAILED exists"; return 2; }
    card_find || { echo "REFUSED $name: card $DEVID not found"; return 3; }
    local kb
    kb=$(mem_avail_kb)
    [[ $kb =~ ^[0-9]+$ ]] && ((kb >= MIN_KB)) || { echo "REFUSED $name: ${kb:-?} kB available (need 8 GiB)"; return 5; }
    wait_cool "$name" || return 4
    if [[ $SELFTEST != 1 ]]; then
        systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 true 2>/dev/null || { echo "REFUSED $name: no systemd user scope for the 10 GB cap"; return 9; }
        unit="r10i3-$$-${name//[^A-Za-z0-9]/_}"
        scope=(systemd-run --user --scope -q --unit="$unit" -p MemoryMax=10G -p MemorySwapMax=0)
    fi
    mkdir -p "$RUN/$name.watch"
    local sampler=("$BIN/bin/r9_sampler" "$RUN/$name.watch" 5)
    [[ $SELFTEST == 1 ]] && sampler=(bash -c 'mkdir -p "$1"; : >"$1/ready"; exec sleep 600' x "$RUN/$name.watch")
    DRM_DEVICE=$DRM_DEVICE HWMON=$HWMON "${sampler[@]}" >"$RUN/$name.sampler.log" 2>&1 9>&- &
    wp=$!
    for _ in $(seq 1 50); do [[ -e $RUN/$name.watch/ready ]] && break; kill -0 "$wp" 2>/dev/null || break; sleep 0.1; done
    [[ -e $RUN/$name.watch/ready ]] || { echo "REFUSED $name: sampler not ready"; kill -TERM "$wp" 2>/dev/null; return 10; }
    t0=$(date +%s)
    echo "$(date -u +%T) start $name: $bin $*"
    # the job: spv folder and out folder for the Vulkan harness, card record for the HIP harness; nice, a time limit, the memory scope
    DRM_DEVICE=$DRM_DEVICE HWMON=$HWMON HIP_VISIBLE_DEVICES=$HIP_INDEX CARD_ARCH=gfx1201 VKG_OUT=$RUN VKG_SPV=$BIN/spv VKG_TAG=$name \
        setsid "${scope[@]}" nice -n 10 timeout -k 5 "$JOB_TMO" "$BIN/bin/$bin" "$@" >"$RUN/$name.out" 2>&1 9>&- &
    jobpid=$!
    # thermal guard: hotspot every 250 ms. At GUARD_C stop the whole payload (process group and systemd scope), not only the wrapper.
    (
        while kill -0 "$jobpid" 2>/dev/null; do
            if (($(hotspot_c) >= GUARD_C)); then
                : >"$RUN/$name.guard-fired"
                stop_payload "$jobpid" "$unit" TERM
                sleep 1
                stop_payload "$jobpid" "$unit" KILL
                break
            fi
            sleep 0.25
        done
    ) 9>&- &
    guardpid=$!
    wait "$jobpid"
    rc=$?
    kill "$guardpid" 2>/dev/null
    wait "$guardpid" 2>/dev/null
    sleep 1
    kill -TERM "$wp" 2>/dev/null
    wait "$wp" 2>/dev/null
    grep -E '^(CELL|RESULT|VERDICT|THERMAL|STATE|INFO|NOT RUN|device|CLOCKS|DEVICE|PIPE)' "$RUN/$name.out" | cut -c1-260 >>"$RUN/summary-lines.txt"
    echo "$(date -u +%T) end $name exit=$rc"
    if [[ -e $RUN/$name.guard-fired ]]; then echo "JOB $name stopped by the ${GUARD_C} C guard"; return 8; fi
    local faults frc
    faults=$(kernel_faults "$t0")
    frc=$?
    if [[ $frc -ne 0 ]]; then
        echo "REFUSED $name: kernel log read failed after the job (status $frc)"
        return 12
    fi
    if [[ -n $faults ]]; then
        printf '%s\n' "$faults" >"$RUN/$name.faults"
        echo "GPU FAULT in the kernel log during $name (saved in $RUN/$name.faults); stopping all GPU work:"
        printf '  %s\n' "$faults" | head -5
        return 13
    fi
    [[ $rc -eq 0 ]] || { echo "JOB $name failed (exit $rc)"; JOBS_FAILED=$((JOBS_FAILED + 1)); return 10; }
    return 0
}

run_jobs() {  # run_jobs <step list...>: the planned jobs of those steps in order, process repeats in rounds
    local r maxr=3 j name bin mode fn step rep a args rc
    for ((r = 1; r <= maxr; r++)); do
        for j in "${JOBS[@]}"; do
            IFS='|' read -r name bin mode fn step rep <<<"$j"
            in_steps "$step" || continue
            ((r <= rep)) || continue
            args=("$mode")
            if [[ -n $fn ]]; then while IFS= read -r a; do args+=("$a"); done < <("$fn"); fi
            run_job "$name-r$r" "$bin" "${args[@]}"
            rc=$?
            case $rc in 0 | 10) ;; *) return $rc ;; esac
            sleep "${R10_PAUSE:-5}"
        done
    done
    return 0
}

run_listing() { # run_listing <dis> <notes> <wave>: print two lines, return the checker's status (not head's)
    local out rc
    out=$(python3 -I "$SRC/check_item3_listing.py" "$1" "$2" "$3" 2>&1)
    rc=$?
    printf '%s\n' "$out" | head -2
    return "$rc"
}

analyse() {
    {
        if [[ -z $PIN ]]; then echo "# STEP 1 NOT RUN (--no-pin): the memory clock was the card's own; see the matched-clock lines for repetitions whose clock read the same before and after"; fi
        python3 -I "$SRC/analyse_item3.py" "$RUN"/s[2-6]-*.out
    } | tee "$RUN/analysis.txt"
    return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------------------------------------------------------------------------
cpu_checks() {
    local bad=0 t
    echo "== bash -n"
    bash -n "$SELF" && echo "syntax ok" || bad=1
    echo "== cell specs"
    validate_cells && echo "every key is one the binaries parse; names unique and well formed" || bad=1
    echo "== plan"
    VERBOSE=0 print_plan
    echo "== analysis self-test"
    python3 -I "$SRC/analyse_item3.py" --selftest | tail -2 || bad=1
    [[ ${PIPESTATUS[0]} -eq 0 ]] || bad=1
    if [[ -f $BIN/asm/r10_cl3/dis.txt ]]; then
        echo "== listing checks of the built binaries ($BIN)"
        run_listing "$BIN/asm/r10_cl3/dis.txt" "$BIN/asm/r10_cl3/notes.txt" 32 || bad=1
        run_listing "$BIN/asm/r10_cl3_w64/dis.txt" "$BIN/asm/r10_cl3_w64/notes.txt" 64 || bad=1
    else
        echo "== listing checks skipped: $BIN not built (src/round10/item3/build.sh)"
    fi
    echo "== gate refusals on stand-in data (no GPU binary is started; the sampler and the payload are stubs)"
    t=${R10_CPU_SCRATCH:-$BIN/selftest}
    rm -rf "$t"
    mkdir -p "$t/bin"
    printf '#!/usr/bin/env bash\necho "ROW r10 api=stub cell=stub rep=1 cycles=1 rt=1 ok=1 steps=1 uclk0=1 uclk1=1"\necho "CELL stub api=stub hotspot=40"\nexit "${STUB_EXIT:-0}"\n' >"$t/bin/stub"
    chmod +x "$t/bin/stub"
    printf '40000\n' >"$t/hot-ok"
    printf '71000\n' >"$t/hot-bad"
    printf 'MemAvailable:   20000000 kB\n' >"$t/mem-ok"
    printf 'MemAvailable:    4000000 kB\n' >"$t/mem-low"
    : >"$t/journal-clean"
    printf 'kernel: amdgpu 0000:03:00.0: amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:32769)\n' >"$t/journal-fault"
    printf 'not a listing\n' >"$t/bad.dis"
    printf 'not notes\n' >"$t/bad.notes"
    bash -c 'python3 -I "$1" "$2" "$3" "$4" | head -2 >/dev/null' _ "$SRC/check_item3_listing.py" "$t/bad.dis" "$t/bad.notes" 32
    local red=$? det
    run_listing "$t/bad.dis" "$t/bad.notes" 32 >/dev/null
    det=$?
    if [[ $red -eq 0 && $det -ne 0 ]]; then
        echo "  listing status: pipe-to-head exits $red; captured helper exits $det"
    else
        echo "  listing status: pipe $red helper $det (wanted 0 and nonzero)"
        bad=1
    fi
    local card
    card=$(for d in /sys/class/drm/card[0-9]*/device; do [[ -f $d/device && $(<"$d/device") == "$DEVID" ]] && echo yes; done | head -1)
    if [[ $card != yes ]]; then
        echo "  (no card with id $DEVID in /sys/class/drm: the wrong-card test below is the only gate test that can run here)"
    fi
    selftest_case() {  # selftest_case <label> <expected exit> <env...>
        local label=$1 want=$2 got
        shift 2
        env R10_SELFTEST=1 R10_ITEM3_BIN="$t" R10_RUN="$t/run-$RANDOM" STUDY_GPU_LOCK="$t/lock" R10_LOCK_WAIT=1 R10_COOL_WAIT=1 R10_PAUSE=0 \
            R10_TEST_HOT_FILE="$t/hot-ok" R10_TEST_MEMINFO="$t/mem-ok" R10_TEST_JOURNAL="$t/journal-clean" R10_HOLD="$t/no-hold" R10_RESTORE_FAILED="$t/no-restore" \
            "$@" bash "$SELF" --selftest-gate >"$t/last.txt" 2>&1
        got=$?
        if [[ $got -eq $want ]]; then echo "  $label: exit $got as expected"; else echo "  $label: exit $got, EXPECTED $want"; sed 's/^/      /' "$t/last.txt" | tail -4; bad=1; fi
    }
    if [[ $card == yes ]]; then
        selftest_case "clean pass through one stub job" 0
        selftest_case "BREAK wrong card (device id 0x1234)" 3 R10_TEST_DEVID=0x1234
        selftest_case "BREAK memory below 8 GiB" 5 R10_TEST_MEMINFO="$t/mem-low"
        selftest_case "BREAK hotspot 71 C does not cool" 4 R10_TEST_HOT_FILE="$t/hot-bad"
        selftest_case "BREAK a GPU page fault in the kernel log" 13 R10_TEST_JOURNAL="$t/journal-fault"
        selftest_case "BREAK the job exits non-zero (counted as a failed job)" 10 STUB_EXIT=3
        selftest_case "BREAK RESTORE-FAILED file present" 2 R10_RESTORE_FAILED="$t/hot-ok"
        selftest_case "BREAK journal read fails after the job" 12 R10_TEST_JOURNAL_FAIL=1
        redj=$(bash -c 'f() { echo "journal read failed" >&2; return 0; }; faults=$(f); printf "rc=%s faults=%s\n" "$?" "${faults:-empty}"')
        if [[ $redj == "rc=0 faults=empty" ]]; then
            echo "  red journal: a helper that returns 0 after a failed read reports $redj"
        else
            echo "  red journal control: $redj"
            bad=1
        fi
        # Killing only the wrapper leaves its grandchild. The guard must not do that.
        bash -c 'sleep 300 & echo $! >"'"$t"'/red-child.pid"; sleep 30' &
        local rpid=$! rcid
        sleep 0.3
        rcid=$(cat "$t/red-child.pid" 2>/dev/null || true)
        kill -TERM "$rpid" 2>/dev/null || true
        sleep 0.2
        if [[ -n $rcid ]] && kill -0 "$rcid" 2>/dev/null; then
            echo "  red thermal: TERM of wrapper $rpid left grandchild $rcid"
        else
            echo "  red thermal: grandchild did not survive the wrapper (${rcid:-missing})"
            bad=1
        fi
        kill -KILL "$rpid" ${rcid:-} 2>/dev/null || true
        local d rest pp
        for d in /proc/[0-9]*; do
            rest=$(sed 's/^.*) //' "$d/stat" 2>/dev/null) || continue
            pp=${rest#* }; pp=${pp%% *}
            [[ $pp == "$rpid" ]] && kill -KILL "${d#/proc/}" 2>/dev/null || true
        done
        wait "$rpid" 2>/dev/null || true
        cp "$t/bin/stub" "$t/bin/stub.orig"
        printf '%s\n' '#!/usr/bin/env bash' 'sleep 300 &' "echo \$! >'$t/grandchild.pid'" "printf '96000\n' >'$t/hot-ok'" 'wait' >"$t/bin/stub"
        chmod +x "$t/bin/stub"
        selftest_case "thermal guard kills the payload process group" 8
        local gpid
        gpid=$(cat "$t/grandchild.pid" 2>/dev/null || true)
        if [[ -n $gpid ]] && ! kill -0 "$gpid" 2>/dev/null; then
            echo "  thermal grandchild $gpid is gone"
        else
            echo "  thermal grandchild still alive (${gpid:-missing})"
            bad=1
            [[ -n $gpid ]] && kill -KILL "$gpid" 2>/dev/null || true
        fi
        mv "$t/bin/stub.orig" "$t/bin/stub"
        printf '40000\n' >"$t/hot-ok"
        (
            exec 8>>"$t/lock"
            flock -n 8 || exit 1
            sleep 6
        ) &
        local holder=$!
        sleep 0.5
        selftest_case "BREAK the lock is held by another job" 6
        kill "$holder" 2>/dev/null
        wait "$holder" 2>/dev/null
    else
        selftest_case "BREAK wrong card (no 0x7550 here)" 3
    fi
    rm -rf "$t"
    ((bad)) && echo "CPU CHECKS FAILED" || echo "CPU CHECKS PASSED"
    return $bad
}

# the gate on stand-in data: lock, checks, one stub job through run_job, kernel log check; exit codes as the real run
selftest_gate() {
    [[ $SELFTEST == 1 ]] || { echo "selftest only"; return 2; }
    [[ $LOCK != "$REAL_LOCK_DEFAULT" ]] || { echo "the self-test refuses the real lock"; return 2; }
    RUN=$R10_RUN
    mkdir -p "$RUN"
    check_start || return $?
    take_lock || return $?
    run_job stub stub bis "name=vk-base-r"
    local rc=$?
    [[ ${STUB_EXIT:-0} -ne 0 && $JOBS_FAILED -eq 1 ]] || [[ ${STUB_EXIT:-0} -eq 0 && $JOBS_FAILED -eq 0 ]] || return 1
    return $rc
}

# ---------------------------------------------------------------------------------------------------------------------------------
if ((CPU)); then
    cpu_checks
    exit $?
fi
if ((PLAN)); then
    VERBOSE=1 print_plan
    exit 0
fi
((SELFTEST_GATE)) && { selftest_gate; exit $?; }

# ---- a real run
[[ -x $BIN/bin/vkr10 && -x $BIN/bin/r10_cl3 && -x $BIN/bin/r10_cl3_w64 && -x $BIN/bin/r9_sampler && -d $BIN/spv ]] || { echo "REFUSED: $BIN is not built (src/round10/item3/build.sh)"; exit 2; }
RUN=${OUT:-$REPO/runs/$(date -u +%Y%m%dT%H%M%SZ)-r10i3}
mkdir -p "$RUN"
check_start || exit $?
if ((STEP1_PAYLOAD)); then  # inside pin_mclk.sh: the lock is already held by it
    echo "STEP 1 payload (memory clock pinned by pin_mclk.sh, lock held by it)"
    mkdir -p "$RUN/pinned"
    RUN=$RUN/pinned
    for r in 1 2 3; do
        mapfile -t a < <(vk_cells_base)
        run_job "s1-vk-r$r" vkr10 bis "${a[@]}" || exit $?
        mapfile -t a < <(hip_cells_base)
        run_job "s1-hip-r$r" r10_cl3 bis "${a[@]}" || exit $?
    done
    exit 0
fi
{
    echo "run folder $RUN; steps $STEPS; $([[ -n $PIN ]] && echo "step 1 pinned at level $PIN" || echo "STEP 1 SKIPPED (--no-pin): the pinned memory clock rerun waits for the owner's sudo approval")"
    echo "binaries: $(sha256sum "$BIN/bin/vkr10" | cut -c1-16) vkr10, $(sha256sum "$BIN/bin/r10_cl3" | cut -c1-16) r10_cl3 (build folder $BIN)"
} | tee "$RUN/run.log"
if in_steps 1 && [[ -n $PIN ]]; then
    echo "STEP 1: pinning mclk level $PIN through scripts/round10/pin_mclk.sh (it takes the shared lock itself; it refuses unless the owner has set its switch by hand)"
    R10_SYSFS=$DRM_DEVICE "$REPO/scripts/round10/pin_mclk.sh" --level "$PIN" -- bash "$SELF" --step1-payload --out "$RUN" | tee -a "$RUN/run.log"
    rc=${PIPESTATUS[0]}
    ((rc != 0)) && { echo "step 1 failed (exit $rc); steps 2-6 are not started"; exit "$rc"; }
fi
if [[ $STEPS == 1 ]]; then exit 0; fi
take_lock || exit $?
echo "lock $LOCK held" | tee -a "$RUN/run.log"
start=$(date +%s)
run_jobs > >(tee -a "$RUN/run.log") 2>&1
rc=$?
sleep 1
echo "jobs finished in $(($(date +%s) - start)) s, exit $rc, failed jobs $JOBS_FAILED" | tee -a "$RUN/run.log"
((rc != 0)) && exit "$rc"
analyse
arc=$?
((arc != 0)) && exit 14
((JOBS_FAILED != 0)) && exit 10
exit 0
