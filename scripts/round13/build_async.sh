#!/usr/bin/env bash
# Round 13 Probe P4 (vk_async) builder (CPU only — zero GPU access).
# Builds vk_async, vk_async_selftest, and all required SPIR-V shaders (gdraw, fp16, sad, tex, stream)
# into build/round13/ (override with ROUND13_BUILD_DIR).
# Statically validates every SPIR-V shader with spirv-val (--target-env vulkan1.3), saves
# spirv-dis listings to build/round13/asm/*.spvasm, and verifies distinct SHA-256 hashes and
# exact intended SPIR-V instruction counts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${ROUND13_BUILD_DIR:-$ROOT/build/round13}"

mkdir -p "$OUT/bin" "$OUT/spv" "$OUT/asm"

GLSLC="$(command -v glslc || true)"
SPIRV_VAL="$(command -v spirv-val || true)"
SPIRV_DIS="$(command -v spirv-dis || true)"

for req in "$GLSLC" "$SPIRV_VAL" "$SPIRV_DIS"; do
  if [ -z "$req" ]; then
    echo "ERROR: missing required toolchain binary (need glslc, spirv-val, spirv-dis)" >&2
    exit 1
  fi
done

echo "[1/3] Building vk_async_selftest (CPU only)"
nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Werror \
  "$ROOT/src/round13/vk_async_selftest.c" -o "$OUT/bin/vk_async_selftest" -lm

echo "[2/3] Compiling and validating SPIR-V shaders for vk_async"
# Graphics load shaders (reused from Round 12 src/vk/shaders/gdraw.{vert,frag})
nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O \
  "$ROOT/src/vk/shaders/gdraw.vert" -o "$OUT/spv/gdraw_vert.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/gdraw_vert.spv"

nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O \
  "$ROOT/src/vk/shaders/gdraw.frag" -o "$OUT/spv/gdraw_frag.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/gdraw_frag.spv"

# Compute job (a) FP16 packed FMA ALU (reused from P2 vk_tex_fp16.comp)
nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O -DUSE_FP16=1 -DDO_TEX=0 -DFMA_PER_TEX=64 \
  "$ROOT/src/round13/shaders/vk_tex_fp16.comp" -o "$OUT/spv/vk_alu_fp16_n64.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_alu_fp16_n64.spv"

# Compute job (b) Integer SAD-like ALU (no msad in GLSL)
nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O \
  "$ROOT/src/round13/shaders/vk_async_sad.comp" -o "$OUT/spv/vk_async_sad.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_async_sad.spv"

# Compute job (c) Texture-sampling heavy (reused from P2 vk_tex.comp OP_MODE=1)
nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O -DOP_MODE=1 \
  "$ROOT/src/round13/shaders/vk_tex.comp" -o "$OUT/spv/vk_tex_lod.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_tex_lod.spv"

# Compute job (d) 128 MiB -> 128 MiB DRAM streaming + deterministic initializer
nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O \
  "$ROOT/src/round13/shaders/vk_async_stream_init.comp" -o "$OUT/spv/vk_async_stream_init.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_async_stream_init.spv"

nice -n 10 "$GLSLC" --target-env=vulkan1.3 -O \
  "$ROOT/src/round13/shaders/vk_async_stream.comp" -o "$OUT/spv/vk_async_stream.spv"
"$SPIRV_VAL" --target-env vulkan1.3 "$OUT/spv/vk_async_stream.spv"

echo "[3/3] Building vk_async binary (C11 + vk_tex_common)"
nice -n 10 cc -O2 -std=c11 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "$ROOT/src/round13/vk_async.c" "$ROOT/src/round13/vk_tex_common.c" \
  -o "$OUT/bin/vk_async" -lvulkan -lpthread -lm

echo "=== Static SPIR-V listing & instruction verification ==="
python3 - "$OUT" "$SPIRV_DIS" <<'PY'
import hashlib, re, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1])
spirv_dis = sys.argv[2]
spv_dir = out / "spv"
asm_dir = out / "asm"

shaders = [
    "gdraw_vert.spv",
    "gdraw_frag.spv",
    "vk_alu_fp16_n64.spv",
    "vk_async_sad.spv",
    "vk_tex_lod.spv",
    "vk_async_stream_init.spv",
    "vk_async_stream.spv",
]

hashes = {}
listings = {}
for name in shaders:
    p = spv_dir / name
    if not p.is_file():
        print(f"FAIL: missing SPIR-V {p}", file=sys.stderr)
        sys.exit(1)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    hashes[name] = h
    dis = subprocess.check_output([spirv_dis, str(p)], text=True)
    listings[name] = dis
    (asm_dir / name.replace(".spv", ".spvasm")).write_text(dis)

if len(set(hashes.values())) != len(shaders):
    print(f"FAIL: duplicate SPIR-V hashes: {hashes}", file=sys.stderr)
    sys.exit(1)

def count_re(txt, pat):
    return sum(1 for ln in txt.splitlines() if re.search(pat, ln))

# 1. Graphics fragment load: >= 2 FP32 Fma
n_gfma = count_re(listings["gdraw_frag.spv"], r"OpExtInst\s+%float\s+\S+\s+Fma\b")
if n_gfma < 2:
    print(f"FAIL gdraw_frag.spv: expected >=2 FP32 Fma, got {n_gfma}", file=sys.stderr)
    sys.exit(1)
print(f"CHECK gdraw_frag.spv: PASS (sha256={hashes['gdraw_frag.spv']}, FP32_Fma={n_gfma})")

# 2. Job (a) FP16: exact 256 v2half Fma inside loop (+5 post-loop even-iters parity = 261 total), 0 image samples
h_lines = listings["vk_alu_fp16_n64.spv"].splitlines()
n_hfma = count_re(listings["vk_alu_fp16_n64.spv"], r"OpExtInst\s+%v2half\s+\S+\s+Fma\b")
n_himg = count_re(listings["vk_alu_fp16_n64.spv"], r"OpImageSampleExplicitLod\b")
loop_merge = None
in_loop = False
n_hfma_loop = 0
for ln in h_lines:
    mlm = re.search(r"OpLoopMerge\s+(%\S+)", ln)
    if mlm and loop_merge is None:
        loop_merge = mlm.group(1)
        in_loop = True
        continue
    if in_loop:
        if re.match(rf"\s*{re.escape(loop_merge)}\s*=\s*OpLabel\b", ln):
            in_loop = False
        elif re.search(r"OpExtInst\s+%v2half\s+\S+\s+Fma\b", ln):
            n_hfma_loop += 1
if n_hfma != 261 or n_hfma_loop != 256 or n_himg != 0:
    print(f"FAIL vk_alu_fp16_n64.spv: v2half_Fma_total={n_hfma} (want 261), loop={n_hfma_loop} (want 256), img={n_himg} (want 0)", file=sys.stderr)
    sys.exit(1)
print(f"CHECK vk_alu_fp16_n64.spv: PASS (sha256={hashes['vk_alu_fp16_n64.spv']}, v2half_Fma_loop={n_hfma_loop}, total={n_hfma})")

# 3. Job (b) Integer SAD-like ALU (no msad in GLSL): >= 64 UMax, >= 64 UMin, >= 64 OpISub, 0 float Fma
sad_dis = listings["vk_async_sad.spv"]
n_umax = count_re(sad_dis, r"OpExtInst\s+%uint\s+\S+\s+UMax\b")
n_umin = count_re(sad_dis, r"OpExtInst\s+%uint\s+\S+\s+UMin\b")
n_isub = count_re(sad_dis, r"OpISub\s+%uint\b")
n_iadd = count_re(sad_dis, r"OpIAdd\s+%uint\b")
n_ffma = count_re(sad_dis, r"OpExtInst\s+%(?:float|v2half)\s+\S+\s+Fma\b")
if n_umax < 64 or n_umin < 64 or n_isub < 64 or n_ffma != 0:
    print(f"FAIL vk_async_sad.spv: UMax={n_umax} UMin={n_umin} ISub={n_isub} float_Fma={n_ffma}", file=sys.stderr)
    sys.exit(1)
print(f"CHECK vk_async_sad.spv: PASS (sha256={hashes['vk_async_sad.spv']}, UMax={n_umax}, UMin={n_umin}, ISub={n_isub}, IAdd={n_iadd}, float_Fma=0)")

# 4. Job (c) Texture sampling: exact 16 OpImageSampleExplicitLod
n_tex = count_re(listings["vk_tex_lod.spv"], r"OpImageSampleExplicitLod\b")
if n_tex != 16:
    print(f"FAIL vk_tex_lod.spv: OpImageSampleExplicitLod={n_tex} (want 16)", file=sys.stderr)
    sys.exit(1)
print(f"CHECK vk_tex_lod.spv: PASS (sha256={hashes['vk_tex_lod.spv']}, OpImageSampleExplicitLod={n_tex})")

# 5. Job (d) Memory streaming: >= 4 uvec4 loads, >= 4 uvec4 XORs, >= 4 OpStore + OpLoopMerge
st_dis = listings["vk_async_stream.spv"]
n_ld4 = count_re(st_dis, r"OpLoad\s+%v4uint\b")
n_xor4 = count_re(st_dis, r"OpBitwiseXor\s+%v4uint\b")
n_st4 = count_re(st_dis, r"OpStore\s+%\S+\s+%\S+")
n_loop = count_re(st_dis, r"OpLoopMerge\b")
if n_ld4 < 4 or n_xor4 < 4 or n_st4 < 4 or n_loop < 2:
    print(f"FAIL vk_async_stream.spv: ld_v4uint={n_ld4} xor_v4uint={n_xor4} stores={n_st4} loops={n_loop}", file=sys.stderr)
    sys.exit(1)
print(f"CHECK vk_async_stream.spv: PASS (sha256={hashes['vk_async_stream.spv']}, ld_v4uint={n_ld4}, xor_v4uint={n_xor4}, stores={n_st4}, loops={n_loop})")

# 6. Stream init shader
si_dis = listings["vk_async_stream_init.spv"]
n_si_st = count_re(si_dis, r"OpStore\s+%\S+\s+%\S+")
if n_si_st < 1:
    print(f"FAIL vk_async_stream_init.spv: stores={n_si_st}", file=sys.stderr)
    sys.exit(1)
print(f"CHECK vk_async_stream_init.spv: PASS (sha256={hashes['vk_async_stream_init.spv']}, stores={n_si_st})")
print("NOTE ACO ISA: runtime RADV/ACO listings are captured and checked via VK_KHR_pipeline_executable_properties on run (--dump-asm or full run).")
PY

echo "VERDICT build_async: PASS"
