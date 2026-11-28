#!/usr/bin/env bash
# Heavy FP8 WMMA HIP async-overlap runner beside 2560x1440 RGBA16F Vulkan graphics
# (SAT-100% in_flight=2 and PACE-60 16.6 ms cadence).
# Tests default ROCm stream priority (DEBUG_HIP_IGNORE_STREAM_PRIORITY=1), true HIGH
# KFD priority (DEBUG_HIP_IGNORE_STREAM_PRIORITY=0), WGP-unit CU masks (cu32, cu32_alt,
# cu16), queue/dispatch flags, and HIP Graph (chain=80), plus deliberate break and
# graphics wall-time + frame-completion-interval analysis.
#
# Usage:
#   scripts/round12/run_async_overlap.sh [--out DIR] [--gfx-spv DIR]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/local.env" ] && source "$ROOT/local.env"

OUT_BIN="${ROUND12_BUILD_DIR:-$ROOT/build/round12}"
LOCK="${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}"
RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-async-overlap"
GFX_SPV="$OUT_BIN/spv"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      [ $# -ge 2 ] || { echo "ERROR: --out requires an argument" >&2; exit 2; }
      RUN_DIR="$2"; shift 2 ;;
    --gfx-spv)
      [ $# -ge 2 ] || { echo "ERROR: --gfx-spv requires an argument" >&2; exit 2; }
      GFX_SPV="$2"; shift 2 ;;
    *) echo "Usage: $0 [--out DIR] [--gfx-spv DIR]" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOCK")" "$RUN_DIR"

DRM_SYSFS_ROOT="${STUDY_DRM_ROOT:-/sys/class/drm}"
DRM_DEV=""
shopt -s nullglob
for d in "$DRM_SYSFS_ROOT"/card[0-9]*/device; do
  if [[ -r "$d/vendor" && -r "$d/device" ]] &&
     [[ "$(cat "$d/vendor" 2>/dev/null)" == "0x1002" ]] &&
     [[ "$(cat "$d/device" 2>/dev/null)" == "0x7550" ]]; then
    DRM_DEV=$(readlink -f "$d")
    break
  fi
done
[[ -n "$DRM_DEV" ]] || { echo "ABORT: No RX 9070 XT found under $DRM_SYSFS_ROOT" >&2; exit 2; }

HOTSPOT_CANDIDATE=""
for h in "$DRM_DEV"/hwmon/hwmon*; do
  if [[ -r "$h/temp2_input" ]]; then
    HOTSPOT_CANDIDATE="$h/temp2_input"
    break
  fi
done
shopt -u nullglob
if [[ -z "$HOTSPOT_CANDIDATE" ]]; then
  echo "ABORT: THERMAL STOP no readable temp2_input under $DRM_DEV/hwmon" >&2
  exit 2
fi
export HOTSPOT_PATH="$HOTSPOT_CANDIDATE"
export METRICS_PATH="$DRM_DEV/gpu_metrics"

HOT_STOP_MC=92000
read_hotspot_mc() {
  local raw mc
  if [[ -z "${HOTSPOT_PATH:-}" || ! -r "$HOTSPOT_PATH" ]]; then
    echo "ABORT: THERMAL STOP hotspot sensor unreadable (${HOTSPOT_PATH:-unset})" >&2
    return 5
  fi
  if ! raw=$(cat "$HOTSPOT_PATH" 2>/dev/null); then
    echo "ABORT: THERMAL STOP failed reading hotspot sensor ($HOTSPOT_PATH)" >&2
    return 5
  fi
  mc="${raw//[[:space:]]/}"
  if [[ ! "$mc" =~ ^[0-9]+$ ]] || (( mc <= 0 )); then
    echo "ABORT: THERMAL STOP invalid hotspot reading '${raw}' from $HOTSPOT_PATH" >&2
    return 5
  fi
  if (( mc >= HOT_STOP_MC )); then
    echo "ABORT: THERMAL STOP hotspot $((mc / 1000))C >= $((HOT_STOP_MC / 1000))C ($HOTSPOT_PATH)" >&2
    return 5
  fi
  echo "$mc"
}

check_log_clean() {
  local label="$1" rc="$2" log="$3"
  if grep -q "THERMAL STOP" "$log"; then
    echo "ERROR: $label hit THERMAL STOP (rc=$rc; see $log)" >&2
    exit 5
  fi
  if grep -q "exact=FAIL" "$log"; then
    echo "ERROR: $label reported exact=FAIL (rc=$rc; see $log)" >&2
    exit 3
  fi
  if (( rc != 0 )); then
    echo "ERROR: $label failed with exit status $rc (see $log)" >&2
    exit "$rc"
  fi
}

check_prio2_diag() {
  local label="$1" rc="$2" out_log="$3" err_log="$4"
  if grep -q "THERMAL STOP" "$out_log" "$err_log"; then
    echo "ERROR: $label hit THERMAL STOP (rc=$rc)" >&2
    exit 5
  fi
  if grep -q "exact=FAIL" "$out_log"; then
    echo "ERROR: $label reported exact=FAIL (rc=$rc)" >&2
    exit 3
  fi
  if (( rc != 0 )); then
    echo "ERROR: $label failed with exit status $rc (see $err_log)" >&2
    exit "$rc"
  fi
  if ! grep -Eq "Created SWq=.*priority 2\b" "$err_log"; then
    echo "ERROR: $label did not confirm hardware queue 'priority 2' in $err_log" >&2
    exit 3
  fi
  grep -E "Created SWq|allocated hardware queues|CU mask" "$err_log" | tee -a "$RUN_DIR/meta.txt"
}

(
  flock -x 9
  ACTIVE_GSTOP=""
  ACTIVE_GPID=""
  ACTIVE_HPID=""
  cleanup_bg() {
    if [[ -n "$ACTIVE_GSTOP" ]]; then
      touch "$ACTIVE_GSTOP" 2>/dev/null || true
    fi
    if [[ -n "$ACTIVE_GPID" ]] && kill -0 "$ACTIVE_GPID" 2>/dev/null; then
      wait "$ACTIVE_GPID" 2>/dev/null || true
    fi
    if [[ -n "$ACTIVE_HPID" ]] && kill -0 "$ACTIVE_HPID" 2>/dev/null; then
      kill "$ACTIVE_HPID" 2>/dev/null || true
      wait "$ACTIVE_HPID" 2>/dev/null || true
    fi
  }
  trap cleanup_bg EXIT INT TERM

  if ps -eo args | grep -v grep | grep -qiE 'Cyberpunk2077\.exe|helldivers2\.exe|Client-Win64-Shipping\.exe|FidelityFX_FSR\.exe|Neverness|HT-Win64|HTGame\.exe'; then
    echo "ABORT: active game detected" >&2
    exit 9
  fi

  HOT_INIT_MC=$(read_hotspot_mc)
  T_START="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "T_START=$T_START DRM_DEV=$DRM_DEV HOTSPOT_PATH=$HOTSPOT_PATH HOT_INIT_C=$((HOT_INIT_MC / 1000))" | tee "$RUN_DIR/meta.txt"

  "$OUT_BIN/bin/hip_wmma_prio" info | tee -a "$RUN_DIR/meta.txt"
  set +e
  "$OUT_BIN/bin/hip_wmma_prio" break > "$RUN_DIR/hip_break.log" 2>&1
  brc=$?
  set -e
  echo "BREAK hip_wmma_prio rc=$brc line=$(cat "$RUN_DIR/hip_break.log")" | tee -a "$RUN_DIR/meta.txt"
  if grep -q "THERMAL STOP" "$RUN_DIR/hip_break.log"; then
    echo "ERROR: hip_wmma_prio break hit THERMAL STOP (rc=$brc)" >&2
    exit 5
  fi
  if [ "$brc" -eq 0 ] || ! grep -q "VERDICT hip_wmma_prio: FAIL" "$RUN_DIR/hip_break.log"; then
    echo "ERROR: hip_wmma_prio break did not fail" >&2
    exit 3
  fi

  # Verify AMD_LOG_LEVEL=3 confirms hardware queue priority 2 when DEBUG_HIP_IGNORE_STREAM_PRIORITY=0
  set +e
  AMD_LOG_LEVEL=3 DEBUG_HIP_IGNORE_STREAM_PRIORITY=0 "$OUT_BIN/bin/hip_wmma_prio" run diag_prio2 high 2 16 1 0 0 - - >"$RUN_DIR/diag_prio2.out" 2>"$RUN_DIR/diag_prio2.err"
  drc=$?
  set -e
  check_prio2_diag "diag_prio2" "$drc" "$RUN_DIR/diag_prio2.out" "$RUN_DIR/diag_prio2.err"

  set +e
  AMD_LOG_LEVEL=3 DEBUG_HIP_IGNORE_STREAM_PRIORITY=0 HSA_CU_MASK=0:0-31 "$OUT_BIN/bin/hip_wmma_prio" run diag_cu32_p2 high 2 16 1 0 0 - - >"$RUN_DIR/diag_cu32_p2.out" 2>"$RUN_DIR/diag_cu32_p2.err"
  drc=$?
  set -e
  check_prio2_diag "diag_cu32_p2" "$drc" "$RUN_DIR/diag_cu32_p2.out" "$RUN_DIR/diag_cu32_p2.err"

  # Standalone GFX baselines in the exact serve regimes (in_flight=2 SAT and 16.6 ms PACE, zero HIP)
  rm -f "$RUN_DIR/base_sat.ready" "$RUN_DIR/base_sat.stop" "$RUN_DIR/base_pace.ready" "$RUN_DIR/base_pace.stop"
  set +e
  "$OUT_BIN/bin/gfx_load" "$GFX_SPV" serve 6450 "$RUN_DIR/base_sat.ready" "$RUN_DIR/base_sat.stop" 40 2 0 > "$RUN_DIR/gfx_sat_alone.log" 2>&1
  grc=$?
  set -e
  check_log_clean "gfx_sat_alone" "$grc" "$RUN_DIR/gfx_sat_alone.log"

  set +e
  "$OUT_BIN/bin/gfx_load" "$GFX_SPV" serve 10791 "$RUN_DIR/base_pace.ready" "$RUN_DIR/base_pace.stop" 15 1 16.6 > "$RUN_DIR/gfx_pace_alone.log" 2>&1
  grc=$?
  set -e
  check_log_clean "gfx_pace_alone" "$grc" "$RUN_DIR/gfx_pace_alone.log"

  run_arm() {
    local name="$1" mode="$2" chain="$3" iters="$4" env_str="$5"
    local claim_p2=0
    if [[ "$name" == *p2* ]]; then
      claim_p2=1
      if [[ "$env_str" != *"DEBUG_HIP_IGNORE_STREAM_PRIORITY=0"* ]]; then
        echo "ERROR: arm '$name' claims priority 2 without DEBUG_HIP_IGNORE_STREAM_PRIORITY=0" >&2
        exit 3
      fi
    fi

    echo "=== Arm: $name (mode=$mode chain=$chain iters=$iters env='$env_str') ==="
    local alone_log="$RUN_DIR/hip_${name}_alone.log"
    local hrc=0
    set +e
    env $env_str "$OUT_BIN/bin/hip_wmma_prio" run "${name}_alone" "$mode" "$chain" "$iters" 5 5 80 - - > "$alone_log" 2>&1
    hrc=$?
    set -e
    check_log_clean "hip ${name}_alone" "$hrc" "$alone_log"
    if (( claim_p2 == 1 )) && ! grep -Eq "^STREAM_INFO .*prio=-1\b" "$alone_log"; then
      echo "ERROR: arm '$name' alone did not report high stream priority (prio=-1) in $alone_log" >&2
      exit 3
    fi
    grep -E "^STREAM_INFO|^HIP_SUMMARY" "$alone_log"

    for reg in sat pace; do
      local g_iters g_infl g_pace g_maxf h_trials h_gap cool_c
      if [[ "$reg" == "sat" ]]; then
        g_iters=6450; g_infl=2; g_pace=0; g_maxf=500; h_gap=2; cool_c=65
        if [[ "$chain" -eq 1 ]]; then h_trials=6; else h_trials=3; fi
      else
        g_iters=10791; g_infl=1; g_pace=16.6; g_maxf=80; h_gap=2; h_trials=5; cool_c=70
      fi
      local hc_mc hc cooled=0
      for _ in $(seq 1 50); do
        hc_mc=$(read_hotspot_mc)
        hc=$((hc_mc / 1000))
        if (( hc < cool_c )); then
          cooled=1
          break
        fi
        sleep 0.15
      done
      if (( cooled == 0 )); then
        hc_mc=$(read_hotspot_mc)
      fi

      local h_ready="$RUN_DIR/${name}_${reg}.hready"
      local g_ready="$RUN_DIR/${name}_${reg}.gready"
      local g_stop="$RUN_DIR/${name}_${reg}.gstop"
      local hip_log="$RUN_DIR/hip_${name}_${reg}.log"
      local gfx_log="$RUN_DIR/gfx_${name}_${reg}.log"
      rm -f "$h_ready" "$g_ready" "$g_stop"

      ACTIVE_GSTOP="$g_stop"
      env $env_str "$OUT_BIN/bin/hip_wmma_prio" run "${name}_${reg}" "$mode" "$chain" "$iters" "$h_trials" "$h_gap" 0 "$h_ready" "$g_ready" > "$hip_log" 2>&1 &
      local hpid=$!
      ACTIVE_HPID="$hpid"
      for _ in $(seq 1 200); do
        [[ -f "$h_ready" ]] && break
        kill -0 "$hpid" 2>/dev/null || break
        sleep 0.02
      done

      local grc=0
      if [[ ! -f "$h_ready" ]] && ! kill -0 "$hpid" 2>/dev/null; then
        set +e
        wait "$hpid"
        hrc=$?
        set -e
        ACTIVE_HPID=""
        ACTIVE_GSTOP=""
        touch "$g_stop"
        echo "ERROR: hip_wmma_prio ${name}_${reg} exited before ready (rc=$hrc; see $hip_log)" >&2
        check_log_clean "hip ${name}_${reg}" "$hrc" "$hip_log"
        exit "${hrc:-1}"
      fi

      "$OUT_BIN/bin/gfx_load" "$GFX_SPV" serve "$g_iters" "$g_ready" "$g_stop" "$g_maxf" "$g_infl" "$g_pace" > "$gfx_log" 2>&1 &
      local gpid=$!
      ACTIVE_GPID="$gpid"
      set +e
      wait "$hpid"
      hrc=$?
      touch "$g_stop"
      wait "$gpid"
      grc=$?
      set -e
      ACTIVE_HPID=""
      ACTIVE_GPID=""
      ACTIVE_GSTOP=""

      check_log_clean "hip ${name}_${reg}" "$hrc" "$hip_log"
      check_log_clean "gfx ${name}_${reg}" "$grc" "$gfx_log"
      if (( claim_p2 == 1 )) && ! grep -Eq "^STREAM_INFO .*prio=-1\b" "$hip_log"; then
        echo "ERROR: arm '${name}_${reg}' did not report high stream priority (prio=-1) in $hip_log" >&2
        exit 3
      fi
    done
  }

  # 1. Baseline priority=1 (default ROCm DEBUG_HIP_IGNORE_STREAM_PRIORITY=1) vs true priority=2 (ABBA order)
  run_arm "base_p1_c80"    "norm"       80 369   ""
  run_arm "true_p2_c80"    "high"       80 369   "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0"
  run_arm "true_p2_c80_r2" "high"       80 369   "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0"
  run_arm "base_p1_c80_r2" "norm"       80 369   ""
  # 2. True KFD priority=2 + monolithic chain=1
  run_arm "true_p2_mono"   "high"       1  58838 "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0"
  # 3. CU masks (WGP units: 16 WGPs = 32 CUs, 16 alt WGPs, 8 WGPs = 16 CUs)
  run_arm "cu32_p1_c80"    "cu32"       80 369   ""
  run_arm "cu32alt_p1_c80" "cu32_alt"   80 369   ""
  run_arm "cu16_p1_c80"    "cu16"       80 369   ""
  # 4. Half-CU mask + TRUE priority=2 (via HSA_CU_MASK=0:0-31 + DEBUG_HIP_IGNORE_STREAM_PRIORITY=0)
  run_arm "cu32_p2_c80"    "high"       80 369   "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0 HSA_CU_MASK=0:0-31"
  # 5. Indirect dispatch + polling + TRUE priority=2
  run_arm "dir0_p2_c80"    "high"       80 369   "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0 AMD_DIRECT_DISPATCH=0 HSA_ENABLE_INTERRUPT=0"
  # 6. HIP Graph (chain=80) + TRUE priority=2
  run_arm "graph_p2_c80"   "graph_high" 80 369   "DEBUG_HIP_IGNORE_STREAM_PRIORITY=0"

  echo "=== Kernel amdgpu fault check ==="
  journalctl -k --since "$T_START" --no-pager > "$RUN_DIR/kern_full.log"
  grep -Ei 'amdgpu.*(fault|reset|hang|ring.*timeout|gpu_recover|page fault)' "$RUN_DIR/kern_full.log" > "$RUN_DIR/faults.log" || true
  if [[ -s "$RUN_DIR/faults.log" ]]; then
    echo "AMDGPU_FAULTS_FOUND" | tee -a "$RUN_DIR/meta.txt"
    cat "$RUN_DIR/faults.log"
    exit 4
  fi
  echo "ZERO_AMDGPU_FAULTS" | tee -a "$RUN_DIR/meta.txt"
) 9>"$LOCK"

python3 - "$RUN_DIR" <<'PY'
import re, statistics, sys
from pathlib import Path

d = Path(sys.argv[1])
arms = ["base_p1_c80", "true_p2_c80", "true_p2_c80_r2", "base_p1_c80_r2", "true_p2_mono", "cu32_p1_c80", "cu32alt_p1_c80", "cu16_p1_c80", "cu32_p2_c80", "dir0_p2_c80", "graph_p2_c80"]

def parse_hip(p):
    rows = []
    therm = False
    for ln in p.read_text().splitlines():
        if "THERMAL STOP" in ln: therm = True
        m = re.search(r"t_start_ms=([\d.]+)\s+t_end_ms=([\d.]+)\s+wall_ms=([\d.]+)\s+gpu_ms=([\d.]+)\s+sclk=(\d+)\s+hot_c=([\d.]+)\s+exact=(\w+)", ln)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)), float(m.group(6)), m.group(7)))
    return rows, therm

def parse_gfx(p):
    rows = []
    therm = False
    for ln in p.read_text().splitlines():
        if "THERMAL STOP" in ln: therm = True
        m = re.search(r"t_start_ms=([\d.]+)\s+t_end_ms=([\d.]+)\s+gpu_ms=([\d.]+)\s+wall_ms=([\d.]+)\s+hot_c=([\d.]+)", ln)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return rows, therm

def pct(v, q):
    if not v: return float("nan")
    s = sorted(v)
    return s[min(len(s)-1, max(0, int(round(q*(len(s)-1)))))]

print(f"{'ARM':15s} {'REG':4s} {'G_dtAvg':>7s} {'G_dt50':>7s} {'G_dt95':>7s} {'G_wal50':>7s} {'G_gpu50':>7s} {'H_full':>6s} {'H_p50':>7s} {'H_p95':>7s} {'H_max':>7s} {'hot':>4s} {'therm':>5s}")
for reg in ("sat", "pace"):
    gp = d / f"gfx_{reg}_alone.log"
    if gp.exists():
        gr, g_th = parse_gfx(gp)
        gr_s = gr[3:] if len(gr) > 5 else gr
        dts = [gr_s[i][1] - gr_s[i-1][1] for i in range(1, len(gr_s))]
        dt_avg = sum(dts) / len(dts) if dts else float("nan")
        gg = [r[2] for r in gr_s]
        gw = [r[3] for r in gr_s]
        hot = max((r[4] for r in gr), default=0)
        print(f"{'gfx_alone':15s} {reg:4s} {dt_avg:7.2f} {pct(dts,0.5):7.2f} {pct(dts,0.95):7.2f} {pct(gw,0.5):7.2f} {pct(gg,0.5):7.2f} {'-':>6s} {'-':>7s} {'-':>7s} {'-':>7s} {hot:4.0f} {'Y' if g_th else 'N':>5s}")

for arm in arms:
    al_p = d / f"hip_{arm}_alone.log"
    if not al_p.exists(): continue
    al, al_th = parse_hip(al_p)
    al_g = [r[3] for r in al]
    print(f"{arm:15s} {'aln':4s} {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s} {f'{len(al)}/{len(al)}':>6s} {pct(al_g,0.5):7.2f} {pct(al_g,0.95):7.2f} {max(al_g) if al_g else 0:7.2f} {max((r[5] for r in al), default=0):4.0f} {'Y' if al_th else 'N':>5s}")
    for reg in ("sat", "pace"):
        hp = d / f"hip_{arm}_{reg}.log"
        gp = d / f"gfx_{arm}_{reg}.log"
        if not hp.exists() or not gp.exists(): continue
        hr, h_th = parse_hip(hp)
        gr, g_th = parse_gfx(gp)
        if not gr or not hr: continue
        g_lo, g_hi = gr[0][0], gr[-1][1]
        # Strictly fully overlapped HIP trials (started >= g_lo and ended <= g_hi)
        h_full = [r for r in hr if r[0] >= g_lo and r[1] <= g_hi]
        h_use = h_full if h_full else [r for r in hr if r[1] >= g_lo and r[0] <= g_hi]
        # Frames overlapping at least one active HIP trial
        gov_idx = [i for i, f in enumerate(gr) if any(f[1] >= h[0] and f[0] <= h[1] for h in h_use)]
        dts = [gr[i][1] - gr[i-1][1] for i in gov_idx if i > 0]
        dt_avg = sum(dts) / len(dts) if dts else float("nan")
        gg = [gr[i][2] for i in gov_idx]
        gw = [gr[i][3] for i in gov_idx]
        hg = [r[3] for r in h_use]
        hot = max([r[5] for r in hr] + [r[4] for r in gr])
        print(f"{arm:15s} {reg:4s} {dt_avg:7.2f} {pct(dts,0.5):7.2f} {pct(dts,0.95):7.2f} {pct(gw,0.5):7.2f} {pct(gg,0.5):7.2f} {f'{len(h_full)}/{len(hr)}':>6s} {pct(hg,0.5):7.2f} {pct(hg,0.95):7.2f} {max(hg) if hg else 0:7.2f} {hot:4.0f} {'Y' if (h_th or g_th) else 'N':>5s}")
PY
