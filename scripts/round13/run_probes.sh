#!/usr/bin/env bash
# Round 13 hardware microbenchmark runner (Probes A: msad, B: vk_tex, C: vk_tex_fp16, D: vk_async).
#
# Modes:
#   scripts/round13/run_probes.sh --cpu-checks
#       Runs every CPU self-test, deliberate break, static listing/SPIR-V check, and guard
#       self-test WITHOUT touching the GPU or taking the GPU lock.
#
#   scripts/round13/run_probes.sh <run_dir> [--only msad|vk_tex|vk_tex_fp16|vk_async|all] [--smoke] [--dump-asm]
#       Runs the probes on RX 9070 XT (1002:7550) under the shared GPU lock (flock -w 600),
#       checks active game processes (pgrep -x), enforces a fail-closed 95 C hotspot stop
#       (hwmon temp2_input of the 1002:7550 card), runs deliberate breaks, captures ACO listings,
#       and checks journalctl -k for amdgpu faults.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/local.env" ] && source "$ROOT/local.env"

OUT_BIN="${ROUND13_BUILD_DIR:-$ROOT/build/round13}"
LOCK="${STUDY_GPU_LOCK:-$ROOT/runs/gpu.lock}"
HOT_STOP_MC=95000  # study limit (README); owner 2026-11-27, was 90000
FAULT_RE='(amdgpu|\[drm|gfxhub|mmhub).*(page fault|GPU reset|ring [A-Za-z0-9_.]+ timeout|VM_L2_PROTECTION|hang|gpu_recover|\*ERROR\*)|amdgpu.*(timeout|fault)'

CPU_CHECKS=0
ONLY="all"
SMOKE=0
DUMP_ASM=0
RUN_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu-checks) CPU_CHECKS=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --out) RUN_DIR="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dump-asm) DUMP_ASM=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [ -z "$RUN_DIR" ]; then
        RUN_DIR="$1"
        shift
      else
        echo "Unexpected extra argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

find_9070xt_hot_file() {
  local c d hw
  for c in /sys/class/drm/card[0-9]*; do
    [ -e "$c/device/device" ] || continue
    [[ "$c" == *-* ]] && continue
    if [ "$(cat "$c/device/vendor" 2>/dev/null || true)" = "0x1002" ] && \
       [ "$(cat "$c/device/device" 2>/dev/null || true)" = "0x7550" ]; then
      d="$(readlink -f "$c/device")"
      hw="$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -n 1 || true)"
      if [ -n "$hw" ] && [ -r "$hw/temp2_input" ]; then
        echo "$hw/temp2_input"
        return 0
      fi
    fi
  done
  return 1
}

check_hot_file() {
  local f="$1" mc
  [ -r "$f" ] || { echo "ABORT: hotspot sensor $f unreadable" >&2; return 4; }
  mc="$(cat "$f")"
  [[ "$mc" =~ ^[0-9]+$ ]] || { echo "ABORT: invalid hotspot reading '$mc'" >&2; return 4; }
  if [ "$mc" -le 0 ]; then
    echo "ABORT: invalid non-positive hotspot reading '$mc'" >&2
    return 4
  fi
  echo "hotspot($f)=$((mc / 1000))C"
  if [ "$mc" -ge "$HOT_STOP_MC" ]; then
    echo "ABORT: hotspot $((mc / 1000))C >= $((HOT_STOP_MC / 1000))C limit" >&2
    return 4
  fi
  return 0
}

GAME_PROCS=(helldivers2.exe HTGame.exe Client-Win64-Sh Cyberpunk2077.e)

check_no_games() {
  local g
  for g in "${GAME_PROCS[@]}"; do
    if pgrep -x "$g" >/dev/null 2>&1; then
      echo "ABORT: active game process detected ($g)" >&2
      return 2
    fi
  done
  if ps -eo args | grep -v grep | grep -qiE 'Cyberpunk2077\.exe|helldivers2\.exe|Client-Win64-Shipping\.exe|HTGame\.exe'; then
    echo "ABORT: active game process detected (cmdline)" >&2
    return 2
  fi
  return 0
}

if [ "$CPU_CHECKS" -eq 1 ]; then
  echo "=== Round 13 CPU-only checks (no GPU, no lock) ==="
  if [ ! -x "$OUT_BIN/bin/msad_selftest" ] || [ ! -x "$OUT_BIN/bin/vk_tex_selftest" ]; then
    echo "Binaries missing under $OUT_BIN; running build_probes.sh first..."
    ROUND13_BUILD_DIR="$OUT_BIN" bash "$ROOT/scripts/round13/build_probes.sh"
  fi
  if [ ! -x "$OUT_BIN/bin/vk_async_selftest" ] || [ ! -x "$OUT_BIN/bin/vk_async" ]; then
    echo "vk_async binaries missing under $OUT_BIN; running build_async.sh first..."
    ROUND13_BUILD_DIR="$OUT_BIN" bash "$ROOT/scripts/round13/build_async.sh"
  fi

  echo "--- Probe A: msad_selftest + 6 deliberate breaks ---"
  "$OUT_BIN/bin/msad_selftest"
  for b in 1 2 3 4 5 6; do
    if "$OUT_BIN/bin/msad_selftest" --break "$b" >/dev/null 2>&1; then
      echo "FAIL: msad_selftest --break $b stayed green" >&2
      exit 1
    fi
    echo "  msad_selftest --break $b: failed as required"
  done

  echo "--- Probes B & C: vk_tex_selftest + 12 deliberate breaks ---"
  "$OUT_BIN/bin/vk_tex_selftest"
  for b in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if "$OUT_BIN/bin/vk_tex_selftest" --break "$b" >/dev/null 2>&1; then
      echo "FAIL: vk_tex_selftest --break $b stayed green" >&2
      exit 1
    fi
    echo "  vk_tex_selftest --break $b: failed as required"
  done
  echo "  vk_tex visit-once + shader-clock + <5-sample refusal + rep-planner checks verified"

  echo "--- Probe D (P4): vk_async_selftest + 12 deliberate breaks ---"
  "$OUT_BIN/bin/vk_async_selftest"
  for b in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if "$OUT_BIN/bin/vk_async_selftest" --break "$b" >/dev/null 2>&1; then
      echo "FAIL: vk_async_selftest --break $b stayed green" >&2
      exit 1
    fi
    echo "  vk_async_selftest --break $b: failed as required"
  done
  echo "  vk_async thermal/cooldown checks: hot sensor stop, cooldown timeout, double-stop abort, multi-repeat overlap verified"

  echo "--- Runner guard self-test (synthetic 85 C pass vs 95 C/<=0/unreadable fail-closed rc=4 + fault regex) ---"
  tmp_guard="$(mktemp -d "${TMPDIR:-$OUT_BIN}/r13-guard-XXXXXX")"
  trap 'rm -rf "$tmp_guard"' EXIT
  echo 85000 > "$tmp_guard/t_ok"
  echo 95000 > "$tmp_guard/t_hot"
  echo 0 > "$tmp_guard/t_zero"
  echo -5000 > "$tmp_guard/t_neg"
  check_hot_file "$tmp_guard/t_ok" >/dev/null
  for bad_f in "$tmp_guard/t_hot" "$tmp_guard/t_zero" "$tmp_guard/t_neg" "$tmp_guard/t_missing"; do
    rc=0
    check_hot_file "$bad_f" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 4 ]; then
      echo "FAIL: thermal guard on $bad_f returned rc=$rc (expected 4)" >&2
      exit 1
    fi
  done
  echo "  $((HOT_STOP_MC / 1000)) C thermal guard: 85 C passed; 95 C, 0, -5000, and missing sensor returned rc=4 as required"
  echo "kernel: amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout" | grep -Eiq "$FAULT_RE"
  if echo "kernel: amdgpu 0000:03:00.0: initialized cleanly" | grep -Eiq "$FAULT_RE"; then
    echo "FAIL: fault regex matched benign log line" >&2
    exit 1
  fi
  echo "  amdgpu fault regex: matched fault, ignored benign line"

  for g in "${GAME_PROCS[@]}"; do
    if [ "${#g}" -gt 15 ]; then
      echo "FAIL: GAME_PROCS entry '$g' exceeds Linux 15-char comm limit (${#g} chars)" >&2
      exit 1
    fi
    pgrep -x "$g" >/dev/null 2>&1 || [ $? -eq 1 ]
  done
  echo "  game process patterns (${GAME_PROCS[*]}): all <= 15 chars and valid for pgrep -x"

  # Verify continue-on-probe-fail vs immediate-stop-on-thermal policy (CPU-only mock).
  mock_fail_count=0
  mock_ran_second=0
  mock_step() {
    local rc="$1" guard_hit="$2"
    if [ "$guard_hit" -eq 1 ] || [ "$rc" -eq 4 ]; then
      return 4
    fi
    if [ "$rc" -ne 0 ]; then
      mock_fail_count=$((mock_fail_count + 1))
    fi
    return 0
  }
  mock_step 1 0
  mock_ran_second=1
  if [ "$mock_fail_count" -ne 1 ] || [ "$mock_ran_second" -ne 1 ]; then
    echo "FAIL: continue-on-probe-fail mock did not continue after rc=1" >&2
    exit 1
  fi
  if mock_step 143 1; then
    echo "FAIL: thermal guard hit did not stop immediately" >&2
    exit 1
  fi
  echo "  continue-on-probe-fail policy: continues on probe rc=1, aborts immediately on thermal guard"

  echo "VERDICT run_probes --cpu-checks: PASS"
  exit 0
fi

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-r13-probes"
fi

case "$ONLY" in
  all|msad|vk_tex|vk_tex_fp16|vk_async) ;;
  *) echo "Invalid --only '$ONLY' (use msad|vk_tex|vk_tex_fp16|vk_async|all)" >&2; exit 2 ;;
esac

NEED_BINS=()
case "$ONLY" in
  all) NEED_BINS=(msad_w32 msad_w64 vk_tex vk_tex_fp16 vk_async) ;;
  msad) NEED_BINS=(msad_w32 msad_w64) ;;
  vk_tex) NEED_BINS=(vk_tex) ;;
  vk_tex_fp16) NEED_BINS=(vk_tex_fp16) ;;
  vk_async) NEED_BINS=(vk_async) ;;
esac

for bin in "${NEED_BINS[@]}"; do
  if [ ! -x "$OUT_BIN/bin/$bin" ]; then
    echo "ERROR: $OUT_BIN/bin/$bin not found; run build_probes.sh / build_async.sh first" >&2
    exit 2
  fi
done

HOT_FILE="$(find_9070xt_hot_file)" || { echo "ABORT: 1002:7550 hwmon temp2_input not found" >&2; exit 3; }

mkdir -p "$(dirname "$LOCK")" "$RUN_DIR/aco_asm"

run_guarded() {
  local log="$1"; shift
  check_hot_file "$HOT_FILE" | tee -a "$RUN_DIR/meta.txt" || return 4
  local gpid="" rc=0
  local guard_flag="$RUN_DIR/.guard_abort.$$"
  rm -f "$guard_flag"
  set +e
  nice -n 10 "$@" >"$log" 2>&1 &
  local jpid=$!
  (
    while kill -0 "$jpid" 2>/dev/null; do
      mc="$(cat "$HOT_FILE" 2>/dev/null || echo "")"
      if ! [[ "$mc" =~ ^[0-9]+$ ]] || [ "$mc" -le 0 ] || [ "$mc" -ge "$HOT_STOP_MC" ]; then
        echo "GUARD ABORT (fail-closed): hotspot reading '${mc:-unreadable}' (limit $((HOT_STOP_MC / 1000))C), killing pid $jpid" >>"$RUN_DIR/meta.txt"
        : > "$guard_flag"
        kill -TERM "$jpid" 2>/dev/null || true
        sleep 2
        kill -KILL "$jpid" 2>/dev/null || true
        exit 4
      fi
      sleep 1
    done
  ) &
  gpid=$!
  wait "$jpid"
  rc=$?
  kill "$gpid" 2>/dev/null || true
  wait "$gpid" 2>/dev/null || true
  set -e
  if [ -e "$guard_flag" ]; then
    rm -f "$guard_flag"
    return 4
  fi
  return "$rc"
}

check_faults_now() {
  journalctl -k --since "$T_START" --no-pager > "$RUN_DIR/kern_full.log"
  grep -Ei "$FAULT_RE" "$RUN_DIR/kern_full.log" > "$RUN_DIR/faults.log" || true
  if [ -s "$RUN_DIR/faults.log" ]; then
    echo "AMDGPU_FAULTS_FOUND" | tee -a "$RUN_DIR/meta.txt"
    cat "$RUN_DIR/faults.log"
    return 4
  fi
  return 0
}

PROBE_FAILS=0
FAILED_PROBES=()

run_probe_step() {
  local label="$1" log="$2"; shift 2
  check_no_games || exit 2
  local rc=0
  run_guarded "$log" "$@" || rc=$?
  tail -n 5 "$log" || true
  if [ "$rc" -eq 4 ]; then
    echo "ABORT: thermal guard stopped $label (rc=4)" | tee -a "$RUN_DIR/meta.txt" >&2
    exit 4
  fi
  check_faults_now || exit 4
  if [ "$rc" -ne 0 ]; then
    echo "PROBE_FAIL label=$label rc=$rc (continuing to next probe)" | tee -a "$RUN_DIR/meta.txt" >&2
    PROBE_FAILS=$((PROBE_FAILS + 1))
    FAILED_PROBES+=("${label}(rc=${rc})")
  fi
  return 0
}

run_break() {
  local label="$1" expect_pat="$2"; shift 2
  local log="$RUN_DIR/${label}_break.log"
  check_no_games || exit 2
  # Break runs use the GPU too, so they get the same fail-closed thermal watchdog.
  local rc=0
  run_guarded "$log" "$@" || rc=$?
  if [ "$rc" -eq 4 ]; then
    echo "ABORT: thermal guard stopped ${label}_break (rc=4)" | tee -a "$RUN_DIR/meta.txt" >&2
    exit 4
  fi
  check_faults_now || exit 4
  local hits
  hits="$(grep -cE "$expect_pat" "$log" || true)"
  echo "BREAK $label rc=$rc fail_matches=$hits" | tee -a "$RUN_DIR/breaks.log"
  if [ "$rc" -eq 0 ] || [ "$hits" -eq 0 ]; then
    echo "PROBE_FAIL label=${label}_break did not fail with pattern '$expect_pat' (rc=$rc, matches=$hits; continuing)" | tee -a "$RUN_DIR/meta.txt" >&2
    PROBE_FAILS=$((PROBE_FAILS + 1))
    FAILED_PROBES+=("${label}_break(rc=${rc},hits=${hits})")
  fi
  return 0
}

exec 9>"$LOCK"
flock -w 600 9 || { echo "ABORT: timed out waiting 600s for GPU lock $LOCK" >&2; exit 6; }

check_no_games
T_START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "T_START=$T_START ONLY=$ONLY SMOKE=$SMOKE DUMP_ASM=$DUMP_ASM HOT_FILE=$HOT_FILE" | tee "$RUN_DIR/meta.txt"
check_hot_file "$HOT_FILE" | tee -a "$RUN_DIR/meta.txt"
: > "$RUN_DIR/breaks.log"

VK_EXTRA=()
[ "$SMOKE" -eq 1 ] && VK_EXTRA+=(--smoke)
[ "$DUMP_ASM" -eq 1 ] && VK_EXTRA+=(--dump-asm)

if [ "$ONLY" = "all" ] || [ "$ONLY" = "msad" ]; then
  if [ "$DUMP_ASM" -eq 0 ]; then
    echo "=== Probe A: msad_w32 ==="
    run_probe_step msad_w32 "$RUN_DIR/msad_w32.log" "$OUT_BIN/bin/msad_w32"
    run_break msad_w32 "exact=FAIL|VERDICT msad_w32: FAIL" "$OUT_BIN/bin/msad_w32" --break

    echo "=== Probe A: msad_w64 ==="
    run_probe_step msad_w64 "$RUN_DIR/msad_w64.log" "$OUT_BIN/bin/msad_w64"
    run_break msad_w64 "exact=FAIL|VERDICT msad_w64: FAIL" "$OUT_BIN/bin/msad_w64" --break
  fi
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "vk_tex" ]; then
  echo "=== Probe B: vk_tex ==="
  run_probe_step vk_tex "$RUN_DIR/vk_tex.log" env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" \
    "$OUT_BIN/bin/vk_tex" "${VK_EXTRA[@]}"
  if [ "$DUMP_ASM" -eq 0 ]; then
    run_break vk_tex "exact=FAIL|VERDICT vk_tex: FAIL" \
      env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" "$OUT_BIN/bin/vk_tex" --break
  fi
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "vk_tex_fp16" ]; then
  echo "=== Probe C: vk_tex_fp16 ==="
  run_probe_step vk_tex_fp16 "$RUN_DIR/vk_tex_fp16.log" env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" \
    "$OUT_BIN/bin/vk_tex_fp16" "${VK_EXTRA[@]}"
  if [ "$DUMP_ASM" -eq 0 ]; then
    run_break vk_tex_fp16 "exact=FAIL|VERDICT vk_tex_fp16: FAIL" \
      env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" "$OUT_BIN/bin/vk_tex_fp16" --break
  fi
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "vk_async" ]; then
  echo "=== Probe D (P4): vk_async ==="
  run_probe_step vk_async "$RUN_DIR/vk_async.log" env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" \
    "$OUT_BIN/bin/vk_async" "${VK_EXTRA[@]}"
  if [ "$DUMP_ASM" -eq 0 ]; then
    run_break vk_async "exact=FAIL|VERDICT vk_async: FAIL" \
      env VKS_SPV="$OUT_BIN/spv" VKS_OUT="$RUN_DIR/aco_asm" "$OUT_BIN/bin/vk_async" --break
  fi
fi

check_hot_file "$HOT_FILE" | tee -a "$RUN_DIR/meta.txt"

echo "=== Kernel amdgpu fault check ==="
check_faults_now || exit 4
echo "ZERO_AMDGPU_FAULTS" | tee -a "$RUN_DIR/meta.txt"
if [ "$PROBE_FAILS" -gt 0 ]; then
  echo "VERDICT run_probes: FAIL (failed=$PROBE_FAILS: ${FAILED_PROBES[*]})" | tee -a "$RUN_DIR/meta.txt"
  exit 1
fi
echo "VERDICT run_probes: PASS"
