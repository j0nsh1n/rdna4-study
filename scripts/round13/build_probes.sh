#!/usr/bin/env bash
# Round 13 hardware microbenchmark builder (CPU only — zero GPU access).
# Builds Probes A (msad_w32, msad_w64), B (vk_tex), and C (vk_tex_fp16), plus CPU self-tests,
# into build/round13/ (override with ROUND13_BUILD_DIR).
# Statically verifies:
#   1. All 93+93 HIP timed loops in msad_w32.s and msad_w64.s contain ONLY the intended SAD/MSAD
#      and FMA instructions at exact 16*U counts, zero spills/v_mov, and correct wavefront_size32.
#   2. All 23 Vulkan compute SPIR-V shaders pass spirv-val (--target-env vulkan1.3), have distinct
#      SHA-256 hashes, and contain exact OpImage* and OpExtInst Fma (%v2half / %float) counts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${ROUND13_BUILD_DIR:-$ROOT/build/round13}"
ROCM_PATH="${ROCM_PATH:-$HOME/.local/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"

mkdir -p "$OUT/bin" "$OUT/spv" "$OUT/asm"

GLSLC="$(command -v glslc || true)"
SPIRV_VAL="$(command -v spirv-val || true)"
SPIRV_DIS="$(command -v spirv-dis || true)"
HIPCC="$(command -v hipcc || true)"

for req in "$GLSLC" "$SPIRV_VAL" "$SPIRV_DIS" "$HIPCC"; do
  if [ -z "$req" ]; then
    echo "ERROR: missing required toolchain binary (need glslc, spirv-val, spirv-dis, hipcc)" >&2
    exit 1
  fi
done

echo "[1/4] Building CPU self-tests (msad_selftest, vk_tex_selftest)"
nice -n 10 g++ -O2 -std=c++17 -Wall -Wextra -Werror \
  "$ROOT/src/round13/msad_selftest.cpp" -o "$OUT/bin/msad_selftest"
nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Werror \
  "$ROOT/src/round13/vk_tex_selftest.c" -o "$OUT/bin/vk_tex_selftest" -lm

echo "[2/4] Building Probe A: msad_w32 & msad_w64 (HIP gfx1201, -mcumode, VOPD off)"
HIP_COMMON=(-O3 -std=c++17 -Wall -Wextra --offload-arch=gfx1201 -mcumode
            -mllvm -amdgpu-enable-vopd=0 -DHARNESS_ARCH=\"gfx1201\")
nice -n 10 "$HIPCC" "${HIP_COMMON[@]}" -S --cuda-device-only -mno-wavefrontsize64 -DW64=0 \
  "$ROOT/src/round13/msad.hip" -o "$OUT/asm/msad_w32.s"
nice -n 10 "$HIPCC" "${HIP_COMMON[@]}" -S --cuda-device-only -mwavefrontsize64 -DW64=1 \
  "$ROOT/src/round13/msad.hip" -o "$OUT/asm/msad_w64.s"
nice -n 10 "$HIPCC" "${HIP_COMMON[@]}" -mno-wavefrontsize64 -DW64=0 \
  "$ROOT/src/round13/msad.hip" -o "$OUT/bin/msad_w32"
nice -n 10 "$HIPCC" "${HIP_COMMON[@]}" -mwavefrontsize64 -DW64=1 \
  "$ROOT/src/round13/msad.hip" -o "$OUT/bin/msad_w64"

echo "[3/4] Building Probe B: vk_tex (C + 3 GLSL compute shaders)"
for spec in "fetch:0" "lod:1" "gather:2"; do
  name="${spec%%:*}"; mode="${spec##*:}"
  nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O -DOP_MODE="$mode" \
    "$ROOT/src/round13/shaders/vk_tex.comp" -o "$OUT/spv/vk_tex_${name}.spv"
  "$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_tex_${name}.spv"
done
nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "$ROOT/src/round13/vk_tex.c" "$ROOT/src/round13/vk_tex_common.c" \
  -o "$OUT/bin/vk_tex" -lvulkan -lpthread -lm

echo "[4/4] Building Probe C: vk_tex_fp16 (C + 20 GLSL compute shaders)"
for pr in "fp16:1" "fp32:0"; do
  pname="${pr%%:*}"; use16="${pr##*:}"
  for n in 0 2 4 8 16 32 64; do
    spv="$OUT/spv/vk_mix_${pname}_n${n}.spv"
    nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O -DUSE_FP16="$use16" -DDO_TEX=1 -DFMA_PER_TEX="$n" \
      "$ROOT/src/round13/shaders/vk_tex_fp16.comp" -o "$spv"
    "$SPIRV_VAL" --target-env vulkan1.3 "$spv"
  done
  for n in 4 16 64; do
    spv="$OUT/spv/vk_alu_${pname}_n${n}.spv"
    nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O -DUSE_FP16="$use16" -DDO_TEX=0 -DFMA_PER_TEX="$n" \
      "$ROOT/src/round13/shaders/vk_tex_fp16.comp" -o "$spv"
    "$SPIRV_VAL" --target-env vulkan1.3 "$spv"
  done
done
nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "$ROOT/src/round13/vk_tex_fp16.c" "$ROOT/src/round13/vk_tex_common.c" \
  -o "$OUT/bin/vk_tex_fp16" -lvulkan -lpthread -lm

echo "=== Static listing & SPIR-V verification ==="
python3 - "$OUT" "$SPIRV_DIS" "$ROOT" <<'PY'
import collections, hashlib, re, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1])
spirv_dis = sys.argv[2]
root = Path(sys.argv[3])

# 1. Strict HIP assembly verification for msad_w32.s and msad_w64.s
op_map = {0: "v_msad_u8", 1: "v_sad_u8", 2: "v_sad_u16", 3: "v_sad_hi_u8"}
for fname, want_wf in (("msad_w32.s", "1"), ("msad_w64.s", "0")):
    txt = (out / "asm" / fname).read_text()
    wfs = re.findall(r"\.amdhsa_wavefront_size32\s+(\d+)", txt)
    if not wfs or any(w != want_wf for w in wfs):
        print(f"FAIL {fname}: expected wavefront_size32={want_wf}, got {set(wfs)}", file=sys.stderr)
        sys.exit(1)
    checked = 0
    for fn in re.split(r"(?=^_Z\S+:)", txt, flags=re.M):
        m = re.match(r"^(_Z\S+):", fn)
        if not m or "k_sad_exact" in m.group(1):
            continue
        name = m.group(1)
        lines = fn.splitlines()
        labels, loops = {}, []
        for i, ln in enumerate(lines):
            ml = re.match(r"^(\.L\S+):", ln.strip())
            if ml:
                labels[ml.group(1)] = i
            mb = re.match(r"^\s*(s_cbranch_\w+|s_branch)\s+(\.L\S+)", ln)
            if mb and mb.group(2) in labels and labels[mb.group(2)] < i:
                loops.append((labels[mb.group(2)], i))
        inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
        if not inner:
            print(f"FAIL {fname} {name}: no inner loop found", file=sys.stderr)
            sys.exit(1)
        best = max(inner, key=lambda se: sum(1 for ln in lines[se[0]:se[1] + 1] if ln.strip().startswith("v_")))
        vops = collections.Counter()
        vseq = []
        for ln in lines[best[0]:best[1] + 1]:
            t = ln.strip()
            if not t or t.startswith((";", ".")) or t.endswith(":"):
                continue
            op = t.split()[0]
            if op.startswith(("v_", "global_", "scratch_", "ds_")):
                vops[op] += 1
                vseq.append((op, t))
        m_acc = re.search(r"k_sad_accILi(\d+)ELi(\d+)ELi(\d+)E", name)
        m_s0  = re.search(r"k_sad_src0ILi(\d+)ELi(\d+)E", name)
        m_fma = re.search(r"k_fma_baseILi(\d+)ELi(\d+)E", name)
        m_mix = re.search(r"k_sad_fma_mixILi(\d+)ELi(\d+)ELi(\d+)E", name)
        if m_acc:
            want = {op_map[int(m_acc.group(1))]: 16 * int(m_acc.group(3))}
        elif m_s0:
            want = {op_map[int(m_s0.group(1))]: 16 * int(m_s0.group(2))}
        elif m_fma:
            want = {"v_fmac_f32_e32": 16 * int(m_fma.group(2))}
        elif m_mix:
            op, nfma, u = int(m_mix.group(1)), int(m_mix.group(2)), int(m_mix.group(3))
            want = {op_map[op]: 16 * u, "v_fmac_f32_e32": 16 * u * nfma}
        else:
            print(f"FAIL {fname}: unexpected kernel {name}", file=sys.stderr)
            sys.exit(1)
        if dict(vops) != want:
            print(f"FAIL {fname} {name}: loop vector ops {dict(vops)} != expected {want}", file=sys.stderr)
            sys.exit(1)
        # Verify zero same-bank VGPR read conflicts in k_fma_base and k_sad_fma_mix,
        # and exact 1:NFMA instruction interleave in k_sad_fma_mix.
        if m_fma or m_mix:
            for op, inst in vseq:
                regs = [int(r) for r in re.findall(r"\bv(\d+)\b", inst)]
                src_regs = regs[:3] if op == "v_fmac_f32_e32" else regs[1:4]
                if len(src_regs) != 3 or len({r % 4 for r in src_regs}) != 3:
                    print(f"FAIL {fname} {name}: VGPR bank conflict on '{inst}' (banks {[r % 4 for r in src_regs]})", file=sys.stderr)
                    sys.exit(1)
        if m_mix:
            sad_op = op_map[int(m_mix.group(1))]
            nfma = int(m_mix.group(2))
            u = int(m_mix.group(3))
            want_seq = ([sad_op] + ["v_fmac_f32_e32"] * nfma) * (16 * u)
            got_seq = [op for op, _ in vseq]
            if got_seq != want_seq:
                print(f"FAIL {fname} {name}: mix interleave broken (first 12 ops {got_seq[:12]})", file=sys.stderr)
                sys.exit(1)
        checked += 1
    if checked != 93:
        print(f"FAIL {fname}: expected 93 timed kernels, found {checked}", file=sys.stderr)
        sys.exit(1)
    print(f"CHECK {fname}: PASS ({checked}/93 timed loops exact, wavefront_size32={want_wf}, zero spills/v_mov/bank-conflicts, exact mix interleave)")

# 2. Verify all 23 SPIR-V binaries have distinct SHA-256 hashes and exact SPIR-V op counts.
spv_dir = out / "spv"
expected_spvs = []
for name, op in (("vk_tex_fetch.spv", "OpImageFetch"),
                 ("vk_tex_lod.spv", "OpImageSampleExplicitLod"),
                 ("vk_tex_gather.spv", "OpImageGather")):
    expected_spvs.append((name, op, 16, None, 0))
for pname, ftype in (("fp16", "%v2half"), ("fp32", "%float")):
    for n in (0, 2, 4, 8, 16, 32, 64):
        expected_spvs.append((f"vk_mix_{pname}_n{n}.spv", "OpImageSampleExplicitLod", 4, ftype, 4 * n))
    for n in (4, 16, 64):
        expected_spvs.append((f"vk_alu_{pname}_n{n}.spv", "OpImageSampleExplicitLod", 0, ftype, 4 * n))

hashes = {}
for name, img_op, want_img, ftype, want_fma in expected_spvs:
    p = spv_dir / name
    if not p.is_file():
        print(f"FAIL: missing SPIR-V {p}", file=sys.stderr)
        sys.exit(1)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    hashes[name] = h
    dis = subprocess.check_output([spirv_dis, str(p)], text=True)
    dis_lines = dis.splitlines()
    got_img = sum(1 for ln in dis_lines if img_op in ln)
    if got_img != want_img:
        print(f"FAIL {name}: {img_op} count {got_img} != {want_img}", file=sys.stderr)
        sys.exit(1)
    # Verify clock ordering in SPIR-V: Coherent buffer, 2 OpReadClockKHR before OpLoopMerge,
    # and strictly after the loop merge label: OpStore -> OpMemoryBarrier -> OpLoad ->
    # OpBranchConditional -> 2 OpReadClockKHR.
    if not any("OpDecorate" in ln and "Coherent" in ln for ln in dis_lines):
        print(f"FAIL {name}: missing Coherent decoration on OutBuf", file=sys.stderr)
        sys.exit(1)
    loop_merge_idx = None
    loop_merge_lbl = None
    loop_end_idx = None
    for idx, ln in enumerate(dis_lines):
        mlm = re.search(r"OpLoopMerge\s+(%\S+)", ln)
        if mlm and loop_merge_idx is None:
            loop_merge_idx = idx
            loop_merge_lbl = mlm.group(1)
        elif loop_merge_lbl is not None and loop_end_idx is None:
            if re.match(rf"\s*{re.escape(loop_merge_lbl)}\s*=\s*OpLabel\b", ln):
                loop_end_idx = idx
    if loop_merge_idx is None or loop_end_idx is None:
        print(f"FAIL {name}: could not locate OpLoopMerge and merge label in SPIR-V", file=sys.stderr)
        sys.exit(1)
    clk_indices = [idx for idx, ln in enumerate(dis_lines) if "OpReadClockKHR" in ln]
    pre_clks = [idx for idx in clk_indices if idx < loop_merge_idx]
    post_clks = [idx for idx in clk_indices if idx > loop_end_idx]
    if len(pre_clks) < 2 or len(post_clks) < 2:
        print(f"FAIL {name}: OpReadClockKHR order broken (pre={len(pre_clks)}, post={len(post_clks)})", file=sys.stderr)
        sys.exit(1)
    first_post_clk = post_clks[0]
    post_slice = dis_lines[loop_end_idx:first_post_clk]
    mb_rel = [i for i, ln in enumerate(post_slice) if "OpMemoryBarrier" in ln]
    if not mb_rel:
        print(f"FAIL {name}: missing OpMemoryBarrier between loop end and end OpReadClockKHR", file=sys.stderr)
        sys.exit(1)
    mb_pos = mb_rel[-1]
    has_store_before_mb = any("OpStore " in ln for ln in post_slice[:mb_pos])
    after_mb = post_slice[mb_pos + 1:]
    load_rel = [i for i, ln in enumerate(after_mb) if "OpLoad %v4float" in ln]
    br_rel = [i for i, ln in enumerate(after_mb) if "OpBranchConditional" in ln]
    if not (has_store_before_mb and load_rel and br_rel and load_rel[0] < br_rel[-1]):
        print(f"FAIL {name}: end OpReadClockKHR not ordered after store->OpMemoryBarrier->reload->branch", file=sys.stderr)
        sys.exit(1)
    if ftype is not None:
        pat = re.compile(rf"OpExtInst\s+{re.escape(ftype)}\s+\S+\s+Fma\b")
        got_fma = sum(1 for ln in dis_lines if pat.search(ln))
        # Shaders with FMA_PER_TEX > 0 add 5 post-loop parity FMAs (for even iters) outside the loop.
        want_total_fma = want_fma + 5 if want_fma > 0 else 0
        if got_fma != want_total_fma:
            print(f"FAIL {name}: {ftype} total Fma count {got_fma} != {want_total_fma}", file=sys.stderr)
            sys.exit(1)
        # Also verify the loop body itself contains exact want_fma instructions.
        loop_fma = sum(1 for ln in dis_lines[loop_merge_idx:loop_end_idx] if pat.search(ln))
        if want_fma > 0 and loop_fma != want_fma:
            print(f"FAIL {name}: {ftype} loop-body Fma count {loop_fma} != {want_fma}", file=sys.stderr)
            sys.exit(1)

if len(set(hashes.values())) != len(expected_spvs):
    print(f"FAIL: duplicate SPIR-V hashes detected: {hashes}", file=sys.stderr)
    sys.exit(1)
print(f"CHECK SPIR-V: PASS ({len(expected_spvs)} distinct valid shaders; exact OpImage*, Fma, and ordered Coherent store->barrier->reload->OpReadClockKHR)")

# 3. Verify vkt_dispatch timestamps use vkCmdWriteTimestamp2 TOP_OF_PIPE/COMPUTE_SHADER before dispatch and BOTTOM_OF_PIPE after.
common_c = (root / "src/round13/vk_tex_common.c").read_text()
m_disp = re.search(r"double\s+vkt_dispatch\([^)]*\)\s*\{.*?\n\}", common_c, flags=re.S)
if not m_disp:
    print("FAIL: could not find vkt_dispatch in vk_tex_common.c", file=sys.stderr)
    sys.exit(1)
body = m_disp.group(0)
if not re.search(
    r"vkCmdWriteTimestamp2\([^,]+,\s*VK_PIPELINE_STAGE_2_(?:TOP_OF_PIPE|COMPUTE_SHADER)_BIT,\s*VT\.qp,\s*0u?\s*\);\s*"
    r"vkCmdDispatch\([^;]+;\s*"
    r"vkCmdWriteTimestamp2\([^,]+,\s*VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,\s*VT\.qp,\s*1u?\s*\);",
    body,
):
    print("FAIL: vkt_dispatch timestamp stages must be TOP_OF_PIPE/COMPUTE_SHADER (0) -> vkCmdDispatch -> BOTTOM_OF_PIPE (1)", file=sys.stderr)
    sys.exit(1)
print("CHECK vk_tex_common timestamps: PASS (vkCmdWriteTimestamp2 TOP_OF_PIPE/COMPUTE_SHADER -> dispatch -> BOTTOM_OF_PIPE)")
print("NOTE ACO ISA: runtime RADV/ACO listings (image_load/image_sample/image_gather4/v_pk_fma_f16) are")
print("              captured and checked automatically via VK_KHR_pipeline_executable_properties on run (--dump-asm or full run).")
PY

echo "VERDICT build_probes: PASS"
