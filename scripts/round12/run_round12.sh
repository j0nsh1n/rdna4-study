#!/usr/bin/env bash
# Run Round 12 no-sudo probes (vk_wgp_probe, store_mall, load_cluster_w32/w64, vk_ld_mat2)
# plus deliberate breaks under STUDY_GPU_LOCK, with real-game, 95 C thermal, and amdgpu fault guards.
#
# Usage:
#   scripts/round12/run_round12.sh [--out DIR] [--only wgp|store|cluster|vk|all]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/local.env" ] && source "$ROOT/local.env"

OUT_BIN="${ROUND12_BUILD_DIR:-$ROOT/build/round12}"
LOCK="${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}"
RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-round12"
ONLY="all"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      [ $# -ge 2 ] || { echo "ERROR: --out requires an argument" >&2; exit 2; }
      RUN_DIR="$2"; shift 2 ;;
    --only)
      [ $# -ge 2 ] || { echo "ERROR: --only requires an argument" >&2; exit 2; }
      ONLY="$2"; shift 2 ;;
    *) echo "Usage: $0 [--out DIR] [--only wgp|store|cluster|vk|all]" >&2; exit 2 ;;
  esac
done

case "$ONLY" in
  all|wgp|store|cluster|vk) ;;
  *)
    echo "ERROR: invalid --only value '$ONLY' (expected: wgp|store|cluster|vk|all)" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$LOCK")" "$RUN_DIR/vk_asm" "$RUN_DIR/wgp_asm"

DRM_SYSFS_ROOT="${STUDY_DRM_ROOT:-/sys/class/drm}"

check_temp() {
  local d h f raw mc sensor=""
  shopt -s nullglob
  for d in "$DRM_SYSFS_ROOT"/card[0-9]*/device; do
    if [[ -r "$d/vendor" && -r "$d/device" ]] &&
       [[ "$(cat "$d/vendor" 2>/dev/null)" == "0x1002" ]] &&
       [[ "$(cat "$d/device" 2>/dev/null)" == "0x7550" ]]; then
      for h in "$d"/hwmon/hwmon*; do
        if [[ -r "$h/temp2_input" ]]; then
          sensor="$h/temp2_input"
          break 2
        fi
      done
    fi
  done
  shopt -u nullglob

  if [[ -z "$sensor" ]]; then
    echo "ABORT: THERMAL STOP no readable RX 9070 XT hotspot sensor under $DRM_SYSFS_ROOT" >&2
    exit 2
  fi
  if ! raw=$(cat "$sensor" 2>/dev/null); then
    echo "ABORT: THERMAL STOP failed reading hotspot sensor ($sensor)" >&2
    exit 2
  fi
  mc="${raw//[[:space:]]/}"
  if [[ ! "$mc" =~ ^[0-9]+$ ]] || (( mc <= 0 )); then
    echo "ABORT: THERMAL STOP invalid hotspot reading '${raw}' from $sensor" >&2
    exit 2
  fi
  echo "temp($sensor)=$((mc / 1000))C"
  if (( mc >= 95000 )); then
    echo "ABORT: THERMAL STOP hotspot $((mc / 1000))C >= 95C ($sensor)" >&2
    exit 2
  fi
}

run_break() {
  local label="$1" expect_pat="$2"; shift 2
  local log="$RUN_DIR/${label}_break.log"
  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e
  local hits
  hits=$(grep -cE "$expect_pat" "$log" || true)
  echo "BREAK $label rc=$rc fail_matches=$hits" | tee -a "$RUN_DIR/breaks.log"
  if [ "$rc" -eq 0 ] || [ "$hits" -eq 0 ]; then
    echo "ERROR: $label --break did not fail with expected pattern '$expect_pat' (rc=$rc, matches=$hits)" >&2
    exit 3
  fi
}

(
  flock -x 9
  if ps -eo args | grep -v grep | grep -qiE 'Cyberpunk2077\.exe|helldivers2\.exe|Client-Win64-Shipping\.exe|FidelityFX_FSR\.exe|Neverness|HT-Win64|HTGame\.exe'; then
    echo "ABORT: active game process detected" >&2
    exit 2
  fi
  T_START="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "T_START=$T_START ONLY=$ONLY" | tee "$RUN_DIR/meta.txt"
  check_temp | tee -a "$RUN_DIR/meta.txt"
  : > "$RUN_DIR/breaks.log"

  if [ "$ONLY" = "all" ] || [ "$ONLY" = "wgp" ]; then
    echo "=== vk_wgp_probe ==="
    VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/wgp_asm" "$OUT_BIN/bin/vk_wgp_probe" | tee "$RUN_DIR/vk_wgp_probe.log"
    run_break vk_wgp_probe "VERDICT vk_wgp_probe: FAIL" env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/wgp_asm" "$OUT_BIN/bin/vk_wgp_probe" --break
    check_temp | tee -a "$RUN_DIR/meta.txt"
  fi

  if [ "$ONLY" = "all" ] || [ "$ONLY" = "store" ]; then
    echo "=== store_mall ==="
    "$OUT_BIN/bin/store_mall" | tee "$RUN_DIR/store_mall.log"
    run_break store_mall "ptr WRONG|verify FAIL" "$OUT_BIN/bin/store_mall" --break
    check_temp | tee -a "$RUN_DIR/meta.txt"
  fi

  if [ "$ONLY" = "all" ] || [ "$ONLY" = "cluster" ]; then
    echo "=== load_cluster_w32 ==="
    "$OUT_BIN/bin/load_cluster_w32" | tee "$RUN_DIR/load_cluster_w32.log"
    run_break load_cluster_w32 "exact=FAIL" "$OUT_BIN/bin/load_cluster_w32" --break
    check_temp | tee -a "$RUN_DIR/meta.txt"

    echo "=== load_cluster_w64 ==="
    "$OUT_BIN/bin/load_cluster_w64" | tee "$RUN_DIR/load_cluster_w64.log"
    run_break load_cluster_w64 "exact=FAIL" "$OUT_BIN/bin/load_cluster_w64" --break
    check_temp | tee -a "$RUN_DIR/meta.txt"
  fi

  if [ "$ONLY" = "all" ] || [ "$ONLY" = "vk" ]; then
    echo "=== vk_ld_mat2 ==="
    VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/vk_asm" "$OUT_BIN/bin/vk_ld_mat2" | tee "$RUN_DIR/vk_ld_mat2.log"
    run_break vk_ld_mat2 "exact=FAIL" env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/vk_asm" "$OUT_BIN/bin/vk_ld_mat2" --break
    check_temp | tee -a "$RUN_DIR/meta.txt"
  fi

  echo "=== Kernel amdgpu fault check ==="
  journalctl -k --since "$T_START" --no-pager > "$RUN_DIR/kern_full.log"
  grep -Ei 'amdgpu.*(fault|reset|hang|ring.*timeout|gpu_recover|page fault)' "$RUN_DIR/kern_full.log" > "$RUN_DIR/faults.log" || true
  if [ -s "$RUN_DIR/faults.log" ]; then
    echo "AMDGPU_FAULTS_FOUND" | tee -a "$RUN_DIR/meta.txt"
    cat "$RUN_DIR/faults.log"
    exit 4
  fi
  echo "ZERO_AMDGPU_FAULTS" | tee -a "$RUN_DIR/meta.txt"
  echo "VERDICT run_round12: PASS"
) 9>"$LOCK"
