#!/usr/bin/env bash
# Build all Round 12 (blind-spot follow-up) probes cleanly into build/round12/.
# No GPU access required; verifies distinct bar_lds*.spv hashes, all 8 explicit
# gfx1201 store-hint forms, and zero in-flight global_load FIFO hazards + true
# clumped scheduling across all 22 load_cluster kernels (w32 & w64).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${ROUND12_BUILD_DIR:-$ROOT/build/round12}"
mkdir -p "$OUT/bin" "$OUT/spv"

GLSLANG="$(command -v glslangValidator || true)"
GLSLC="$(command -v glslc || true)"
compile_spv() {
  local src="$1" out="$2"; shift 2
  if [ -n "$GLSLANG" ]; then
    "$GLSLANG" -V --target-env vulkan1.3 "$@" "$src" -o "$out"
  elif [ -n "$GLSLC" ]; then
    "$GLSLC" --target-env=vulkan1.3 -O "$@" "$src" -o "$out"
  else
    echo "ERROR: need glslangValidator or glslc in PATH" >&2
    exit 1
  fi
}

echo "[1/5] Building vk_wgp_probe (RADV CU-vs-WGP mode)"
for spec in "bar_lds0:0" "bar_lds16k:16384" "bar_lds28k:28672" "bar_lds32k:32768" "bar_lds48k:49152"; do
  name="${spec%%:*}"; bytes="${spec##*:}"
  uints=$((bytes / 4))
  compile_spv "$ROOT/src/round12/vk_wgp_bar.comp" "$OUT/spv/${name}.spv" -DLDS_UINTS="$uints"
done
cc -O2 -Wall -Wextra -Wno-missing-field-initializers \
  "$ROOT/src/round12/vk_wgp_probe.c" "$ROOT/src/vk/vkh.c" \
  -o "$OUT/bin/vk_wgp_probe" -lvulkan -lpthread -lm

echo "[2/5] Building async_overlap (gfx_load + gdraw shaders + hip_wmma_prio)"
compile_spv "$ROOT/src/vk/shaders/gdraw.vert" "$OUT/spv/gdraw_vert.spv"
compile_spv "$ROOT/src/vk/shaders/gdraw.frag" "$OUT/spv/gdraw_frag.spv"
cc -O2 -Wall -Wextra -Wno-missing-field-initializers \
  "$ROOT/src/round12/gfx_load.c" -o "$OUT/bin/gfx_load" -lvulkan -lm
hipcc -O3 --offload-arch=gfx1201 -mcumode \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/hip_wmma_prio.hip" -o "$OUT/bin/hip_wmma_prio"

echo "[3/5] Building store_mall (Blind Spot #3)"
hipcc -O3 -S --cuda-device-only --offload-arch=gfx1201 -mcumode \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/store_mall.hip" -o "$OUT/store_mall.s"
hipcc -O3 --offload-arch=gfx1201 -mcumode \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/store_mall.hip" -o "$OUT/bin/store_mall"

echo "[4/5] Building load_cluster_w32 & load_cluster_w64 (Blind Spot #4)"
hipcc -O3 -S --cuda-device-only --offload-arch=gfx1201 -mcumode -mno-wavefrontsize64 \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/load_cluster.hip" -o "$OUT/load_cluster_w32.s"
hipcc -O3 -S --cuda-device-only --offload-arch=gfx1201 -mcumode -mwavefrontsize64 -DW64=1 \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/load_cluster.hip" -o "$OUT/load_cluster_w64.s"
hipcc -O3 --offload-arch=gfx1201 -mcumode -mno-wavefrontsize64 \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/load_cluster.hip" -o "$OUT/bin/load_cluster_w32"
hipcc -O3 --offload-arch=gfx1201 -mcumode -mwavefrontsize64 -DW64=1 \
  -DHARNESS_ARCH=\"gfx1201\" \
  "$ROOT/src/round12/load_cluster.hip" -o "$OUT/bin/load_cluster_w64"

echo "[5/5] Building vk_ld_mat2 (Blind Spot #10)"
for spec in "ref:0" "k1:1" "k8:8"; do
  name="${spec%%:*}"; mode="${spec##*:}"
  compile_spv "$ROOT/src/round12/vk_ld_mat2.comp" "$OUT/spv/vk_lm_${name}.spv" -DMODE="$mode"
done
cc -O2 -Wall -Wextra -Wno-missing-field-initializers \
  "$ROOT/src/round12/vk_ld_mat2.c" "$ROOT/src/vk/vkh.c" \
  -o "$OUT/bin/vk_ld_mat2" -lvulkan -lpthread -lm

echo "=== Static checks (SPV distinctness, store_mall hints, load_cluster FIFO + clump order) ==="
python3 - "$OUT" <<'PY'
import hashlib, re, sys
from pathlib import Path

out = Path(sys.argv[1])

# 1. Verify all 5 bar_lds*.spv have distinct hashes (catches -DLDS_BYTES vs -DLDS_UINTS bugs!)
spv_names = ["bar_lds0.spv", "bar_lds16k.spv", "bar_lds28k.spv", "bar_lds32k.spv", "bar_lds48k.spv"]
hashes = {}
for name in spv_names:
    h = hashlib.sha256((out / "spv" / name).read_bytes()).hexdigest()[:16]
    hashes[name] = h
if len(set(hashes.values())) != len(spv_names):
    print(f"FAIL: bar_lds*.spv hashes are not all distinct: {hashes}", file=sys.stderr)
    sys.exit(1)
print(f"bar_lds*.spv distinctness check: PASS ({len(spv_names)} distinct SPIR-V binaries)")

# 2. Verify all 8 explicit non-default gfx1201 store-hint combinations appear in store_mall.s
sm = (out / "store_mall.s").read_text()
expected_patterns = [
    ("nt",     r"global_store_b128[^\n]*\bth:TH_STORE_NT(?!_)(?![^\n]*scope:SCOPE_SYS)"),
    ("ht",     r"global_store_b128[^\n]*\bth:TH_STORE_HT\b"),
    ("wb",     r"global_store_b128[^\n]*\bth:TH_STORE_WB\b"),
    ("nt_rt",  r"global_store_b128[^\n]*\bth:TH_STORE_NT_RT\b"),
    ("rt_nt",  r"global_store_b128[^\n]*\bth:TH_STORE_RT_NT\b"),
    ("nt_ht",  r"global_store_b128[^\n]*\bth:TH_STORE_NT_HT\b"),
    ("rt_sys", r"global_store_b128(?![^\n]*th:TH_STORE_)[^\n]*\bscope:SCOPE_SYS\b"),
    ("nt_sys", r"global_store_b128[^\n]*\bth:TH_STORE_NT\b[^\n]*\bscope:SCOPE_SYS\b"),
]
for label, pat in expected_patterns:
    if not re.search(pat, sm):
        print(f"FAIL: store_mall.s missing hint form '{label}' ({pat})", file=sys.stderr)
        sys.exit(1)
print(f"store_mall.s hint check: PASS ({len(expected_patterns)} explicit non-default forms + default rt)")

# 3. Verify zero in-flight global_load FIFO hazards (including PAT=5 s_cbranch_scc1!)
#    and verify true clumped scheduling (no global_load interleaved inside 4-WMMA groups) for PAT=2,4,10.
def parse_vregs(tok):
    m = re.match(r'^v\[(\d+):(\d+)\]$', tok)
    if m: return set(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.match(r'^v(\d+)$', tok)
    if m: return {int(m.group(1))}
    return set()

bad = 0
verified = 0
for fname in ("load_cluster_w32.s", "load_cluster_w64.s"):
    txt = (out / fname).read_text()
    for fn in re.split(r'(?=^_Z9k_clusterILi\d+EE)', txt, flags=re.M):
        m = re.match(r'^_Z9k_clusterILi(\d+)EE', fn)
        if not m: continue
        pat = int(m.group(1))
        verified += 1
        lines = fn.splitlines()
        # Find the loop back-edge (any s_cbranch_* targeting an earlier .LBB label!)
        label_pos = {}
        loop_start = loop_end = None
        for idx, ln in enumerate(lines):
            ml = re.match(r'^(\.LBB\d+_\d+):', ln.strip())
            if ml:
                label_pos[ml.group(1)] = idx
            mb = re.search(r's_cbranch_\w+\s+(\.LBB\d+_\d+)', ln)
            if mb and mb.group(1) in label_pos:
                loop_start = label_pos[mb.group(1)]
                loop_end = idx
        if loop_start is None:
            print(f"FAIL {fname} PAT={pat}: could not locate loop back-edge", file=sys.stderr)
            bad += 1
            continue
        loop_body = lines[loop_start:loop_end + 1]
        trace = lines[:loop_start] + loop_body * 2
        fifo, errs = [], []
        for ln in trace:
            s = ln.strip()
            if s.startswith('s_wait_loadcnt'):
                k = int(s.split()[1], 0)
                if k == 0: fifo.clear()
                elif len(fifo) > k: fifo = fifo[-k:]
            elif s.startswith('global_load_'):
                parts = [p.strip() for p in s.split(None, 1)[1].split(',')]
                dst, addr = parse_vregs(parts[0]), parse_vregs(parts[1])
                for fd, fln in fifo:
                    if addr & fd: errs.append(f"ADDR-OVERLAP: {s} vs {fln}")
                    if dst & fd: errs.append(f"DST-OVERLAP: {s} vs {fln}")
                if addr & dst: errs.append(f"SELF-OVERLAP: {s}")
                fifo.append((dst, s))
            elif s.startswith('v_wmma_'):
                parts = [p.strip() for p in s.split(None, 1)[1].split(',')]
                srcs = parse_vregs(parts[1]) | parse_vregs(parts[2])
                for fd, fln in fifo:
                    if srcs & fd: errs.append(f"WMMA-RACE: {s} vs {fln}")
            elif s.startswith('v_') and not s.startswith('v_wmma_'):
                used = set()
                for t in re.findall(r'v\[\d+:\d+\]|v\d+', s): used |= parse_vregs(t)
                for fd, fln in fifo:
                    if used & fd: errs.append(f"VALU-RACE: {s} vs {fln}")

        # Verify true clumped order in PAT 2, 4, 10: must be L* W M M M M L* W M M M M
        if pat in (2, 4, 10):
            seq = []
            for ln in loop_body:
                s = ln.strip()
                if s.startswith('global_load_'): seq.append('L')
                elif s.startswith('s_wait_loadcnt'): seq.append('W')
                elif s.startswith('v_wmma_'): seq.append('M')
            n_l = 8 if pat == 4 else 4
            expect_seq = (['L'] * n_l + ['W'] + ['M'] * 4) * 2
            if seq != expect_seq:
                errs.append(f"CLUMP-INTERLEAVE: got {''.join(seq)} expected {''.join(expect_seq)}")

        if errs:
            bad += 1
            print(f"FAIL {fname} PAT={pat}: {errs[0]}", file=sys.stderr)
if verified != 22 or bad:
    print(f"FAIL: verified={verified} (expected 22), bad={bad}", file=sys.stderr)
    sys.exit(1)
print(f"load_cluster FIFO + clump check: PASS ({verified}/22 kernels clean)")
PY

echo "VERDICT build_round12: PASS"
