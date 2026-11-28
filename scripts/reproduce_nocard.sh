#!/usr/bin/env bash
# CPU-only checks for a clean checkout of this study.
# Does not start a GPU job and does not take gpu-bench.lock.
#
#   scripts/reproduce_nocard.sh --out report.json [--skip-p0] [--prove-fail]
#
# Exit 0 when every recorded check passed or was skipped. Exit 1 when any
# check failed, or when the report still contains a /home or /Users path.
# Exit 2 on usage.
# --skip-p0 records round 10's pin-mclk dry run as skipped (another writer
# may be editing that script). --prove-fail appends one check that runs
# `false`. The default command does not include that check.
#
# Each included item is a CPU self-test (python3 -I or bash) that does not
# open a device:
#   scripts/env_manifest.py --self-test
#   scripts/archive_runs.py --self-test
#   scripts/replicate_top5.sh --cpu-checks
#   bugreports/verify_all_cpu.sh --skip-bundles, when bugreports/ is in the tree
#   scripts/round10/selftest-p0.sh, unless --skip-p0
#   scripts/extcheck/c1_scan.py --selftest, when the hazard listing is in the tree
#   scripts/extcheck/c3_chance.py, when a pair-small-small run is in the tree
#   the round 9 analysis --selftest scripts and the three round 9 shell self-tests
#   the stdlib LLM selftests, including llm4_vllm.py (that selftest never imports torch)
#   scripts/study-home-selftest.sh (fake sensors and a stub lact; about 6 minutes)
#
# Left out after reading them, because they are not a clean-checkout CPU self-test:
#   extcheck/c1_build.sh compiles device assembly and is not a self-test
#   extcheck/c2_build.sh needs an arithmetic tree that is not in this repo
#   extcheck/c2_extract.py writes a header and has no self-test
#   round9/check_listing.py selftest needs a real disassembly
#   llm3_pair_isacheck.py and llm3_mix_isacheck.py need listings
#   llm_ref.py, llm3_mixed_ref.py and llm4_origin.py import a convert environment
#   llm4_torch.py imports torch while the module loads
#   study-overclock-selftest.sh talks to the card
#   the study-*.sh drivers, first-hour.sh and round9/run_*.sh start GPU work
set -u
set -o pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
cd "$REPO" || exit 2

OUT=""
SKIP_P0=0
PROVE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --out)
            [[ $# -ge 2 ]] || { echo "reproduce_nocard: --out needs a path" >&2; exit 2; }
            OUT=$2
            shift 2
            ;;
        --skip-p0) SKIP_P0=1; shift ;;
        --prove-fail) PROVE=1; shift ;;
        *) echo "reproduce_nocard: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$OUT" ]] || { echo "reproduce_nocard: --out is required" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")"
OUTDIR=$(cd "$(dirname "$OUT")" && pwd)
OUTBASE=$(basename "$OUT")
OUT="$OUTDIR/$OUTBASE"
SCRATCH="$OUTDIR/extcheck-scratch"
LOGDIR="$OUTDIR/nocard-logs"
RECORDS="$OUTDIR/nocard-records.jsonl"
rm -rf "$LOGDIR" "$SCRATCH"
mkdir -p "$LOGDIR" "$SCRATCH"
: >"$RECORDS"
export EXTCHECK_SCRATCH="$SCRATCH"

record() {
    python3 -I - "$RECORDS" "$@" <<'PY'
import json, sys
sys.dont_write_bytecode = True
path = sys.argv[1]
name, command, status, exit_code, runtime, detail = sys.argv[2:8]
log = sys.argv[8] if len(sys.argv) > 8 else ""
row = {
    "name": name,
    "command": command,
    "status": status,
    "exit_code": None if exit_code == "null" else int(exit_code),
    "runtime_s": None if runtime == "null" else float(runtime),
    "detail": None if detail == "-" else detail,
}
if log:
    row["log"] = log
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

run_check() {
    local name=$1 display=$2
    shift 2
    local log="$LOGDIR/${name}.log" start end rc rt rel
    start=$(date +%s.%N)
    "$@" >"$log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    rt=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    if [[ $rc -eq 0 ]]; then
        rm -f "$log"
        record "$name" "$display" "pass" "$rc" "$rt" "-"
        return 0
    fi
    # Keep the failure log. Strip home directories so the saved text and the report stay free of them.
    python3 -I - "$log" <<'PY'
import os, re, sys
sys.dont_write_bytecode = True
p = sys.argv[1]
home = os.environ.get("HOME", "")
with open(p, encoding="utf-8", errors="replace") as fh:
    text = fh.read()
if home:
    text = text.replace(home, "[home]")
text = re.sub(r"/(?:home|Users)/[^/\s\"']+", "[home]", text)
with open(p, "w", encoding="utf-8") as fh:
    fh.write(text)
PY
    rel=${log#"$OUTDIR"/}
    [[ $rel == "$log" || $rel == /* ]] && rel="nocard-logs/${name}.log"
    record "$name" "$display" "fail" "$rc" "$rt" "$rel" "$rel"
}

skip_check() {
    record "$1" "$2" "skipped" "null" "null" "$3"
}

run_check env-manifest "python3 -I scripts/env_manifest.py --self-test" \
    python3 -I scripts/env_manifest.py --self-test
run_check archive-runs "python3 -I scripts/archive_runs.py --self-test" \
    python3 -I scripts/archive_runs.py --self-test
run_check replicate-top5 "bash scripts/replicate_top5.sh --cpu-checks --out <scratch>" \
    bash scripts/replicate_top5.sh --cpu-checks --out "$SCRATCH/replicate-top5"
if [[ -f "$REPO/bugreports/verify_all_cpu.sh" ]]; then
    run_check bugreports-hygiene "bash bugreports/verify_all_cpu.sh --skip-bundles --out <scratch>" \
        bash bugreports/verify_all_cpu.sh --skip-bundles --out "$SCRATCH/bugreports-hygiene"
else
    skip_check bugreports-hygiene "bash bugreports/verify_all_cpu.sh --skip-bundles --out <scratch>" \
        "bugreports/ is not in this checkout"
fi

if [[ $SKIP_P0 -eq 1 ]]; then
    skip_check round10-p0 "bash scripts/round10/selftest-p0.sh" "skipped by --skip-p0"
else
    run_check round10-p0 "bash scripts/round10/selftest-p0.sh" \
        bash scripts/round10/selftest-p0.sh
fi

c1=$(find "$REPO" -name 'r101-gfx1201-O3-haz_hazard.s' -print -quit 2>/dev/null || true)
if [[ -n "$c1" && -f "$c1" ]]; then
    c1_dir=${c1#"$REPO"/}
    c1_dir=$(dirname "$c1_dir")
    run_check c1-scan "python3 -I scripts/extcheck/c1_scan.py ${c1_dir} --selftest" \
        python3 -I scripts/extcheck/c1_scan.py "$c1_dir" --selftest
else
    skip_check c1-scan "python3 -I scripts/extcheck/c1_scan.py <listings> --selftest" \
        "hazard listing is not in this checkout"
fi

pair=""
if [[ -d "$REPO/runs" ]]; then
    pair=$(find "$REPO/runs" -name 'pair-small-small-A.csv' -print -quit 2>/dev/null || true)
fi
if [[ -n "$pair" && -f "${pair%-A.csv}-B.csv" ]]; then
    pair_dir=${pair#"$REPO"/}
    pair_dir=$(dirname "$pair_dir")
    run_check c3-chance "python3 -I scripts/extcheck/c3_chance.py . ${pair_dir}" \
        python3 -I scripts/extcheck/c3_chance.py . "$pair_dir"
else
    skip_check c3-chance "python3 -I scripts/extcheck/c3_chance.py . <pair run>" \
        "pair-small-small csv is not in this checkout"
fi

run_check analyse-stable "python3 -I scripts/round9/analyse_stable.py --selftest" \
    python3 -I scripts/round9/analyse_stable.py --selftest
run_check analyse-hier "python3 -I scripts/round9/analyse_hier.py --selftest" \
    python3 -I scripts/round9/analyse_hier.py --selftest
run_check analyse-clk "python3 -I scripts/round9/analyse_clk.py --selftest" \
    python3 -I scripts/round9/analyse_clk.py --selftest
run_check analyse-cbar "python3 -I scripts/round9/analyse_cbar.py --selftest" \
    python3 -I scripts/round9/analyse_cbar.py --selftest
run_check ana-ld-order "python3 -I scripts/round9/ana_ld_order.py --selftest" \
    python3 -I scripts/round9/ana_ld_order.py --selftest
run_check ana-ld-mat "python3 -I scripts/round9/ana_ld_mat.py --selftest" \
    python3 -I scripts/round9/ana_ld_mat.py --selftest
run_check ana-ld-mat-k "python3 -I scripts/round9/ana_ld_mat_k.py --selftest" \
    python3 -I scripts/round9/ana_ld_mat_k.py --selftest
run_check analyse-e2 "python3 -I scripts/round9/analyse_E2.py --selftest" \
    python3 -I scripts/round9/analyse_E2.py --selftest

run_check round9-driver "bash scripts/round9/selftest_driver.sh" \
    bash scripts/round9/selftest_driver.sh
# Non-interactive shells started by editors or daemons often inherit SIGHUP=SIG_IGN,
# which POSIX bash cannot un-ignore; restore default signal dispositions before exec.
run_check round9-g1 "bash scripts/round9/selftest_g1.sh" \
    python3 -I -c 'import os, signal; [signal.signal(s, signal.SIG_DFL) for s in (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM)]; os.execvp("bash", ["bash", "scripts/round9/selftest_g1.sh"])'
run_check round9-e-bg "bash scripts/round9/selftest_e_bg.sh" \
    bash scripts/round9/selftest_e_bg.sh

run_check llm3-cnt "python3 -I scripts/llm3_cnt.py selftest" \
    python3 -I scripts/llm3_cnt.py selftest
run_check llm3-q3 "python3 -I scripts/llm3_q3.py selftest" \
    python3 -I scripts/llm3_q3.py selftest
run_check llm3-power "python3 -I scripts/llm3_power.py selftest" \
    python3 -I scripts/llm3_power.py selftest
run_check llm3-pairs "python3 -I scripts/llm3_pairs.py selftest" \
    python3 -I scripts/llm3_pairs.py selftest
run_check llm4-trace "python3 -I scripts/llm4_trace.py selftest" \
    python3 -I scripts/llm4_trace.py selftest
run_check llm4-tables "python3 -I scripts/llm4_tables.py selftest" \
    python3 -I scripts/llm4_tables.py selftest
run_check llm4-isa "python3 -I scripts/llm4_isa.py selftest" \
    python3 -I scripts/llm4_isa.py selftest
run_check llm4-vllm "python3 -I scripts/llm4_vllm.py selftest" \
    python3 -I scripts/llm4_vllm.py selftest
run_check llm-tables "python3 -I scripts/llm_tables.py selftest <scratch>" \
    python3 -I scripts/llm_tables.py selftest "$SCRATCH/llm-tables"
run_check llm-kernels "python3 -I scripts/llm_kernels.py selftest <scratch>" \
    python3 -I scripts/llm_kernels.py selftest "$SCRATCH/llm-kernels"

# r04 fails if the recorder still has a descriptor open that is not its own
# output. A shell started by an editor inherits that editor's files, and the
# recorder then fails the check on those. A clean shell has 0, 1 and 2 only.
run_check study-home "bash scripts/study-home-selftest.sh" \
    bash -c 'for n in $(ls /proc/self/fd); do if [[ $n =~ ^[0-9]+$ ]] && (( n > 2 )); then eval "exec ${n}>&-"; fi; done; exec bash scripts/study-home-selftest.sh'

if [[ $PROVE -eq 1 ]]; then
    run_check deliberate-fail "false" false
fi

rmdir "$LOGDIR" 2>/dev/null || true
python3 -I - "$RECORDS" "$OUT" "$SKIP_P0" "$PROVE" <<'PY'
import datetime, json, re, sys
records, out_path, skip_p0, prove = sys.argv[1:5]
checks = []
with open(records, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            checks.append(json.loads(line))
doc = {
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "skip_p0": skip_p0 == "1",
    "prove_fail": prove == "1",
    "checks": checks,
    "n_pass": sum(1 for c in checks if c["status"] == "pass"),
    "n_fail": sum(1 for c in checks if c["status"] == "fail"),
    "n_skip": sum(1 for c in checks if c["status"] == "skipped"),
}
blob = json.dumps(doc, indent=1, sort_keys=True) + "\n"
if re.search(r"/home/|/Users/", blob):
    sys.stderr.write("reproduce_nocard: refused to write a report that still contains a home path\n")
    sys.exit(1)
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(blob)
print("n_pass=%s n_fail=%s n_skip=%s" % (doc["n_pass"], doc["n_fail"], doc["n_skip"]))
sys.exit(1 if doc["n_fail"] else 0)
PY
rc=$?
rm -f "$RECORDS"
exit "$rc"
