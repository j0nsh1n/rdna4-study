#!/usr/bin/env bash
# Round 13 (P1): Compile AMD FidelityFX SDK v1.1.4 (MIT) Vulkan GLSL shaders with glslc,
# validate every SPIR-V with spirv-val, build vk_pipestats, run CPU self-tests, and
# optionally collect RADV/ACO pipeline stats + disassembly under the shared GPU lock
# (pipeline creation only — zero dispatches).
#
# Usage:
#   scripts/round13/build_mit_stats.sh --ffx-src DIR --out DIR [--lock FILE] [--no-gpu | --run-gpu] [--group1-only]
#
# Environment defaults (if flags omitted):
#   FFX_SRC_DIR   : path to FidelityFX-SDK v1.1.4 sparse checkout (required)
#   MIT_STATS_OUT : scratch output directory (default: $ROOT/build/round13-mit-stats)
#   STUDY_GPU_LOCK: shared GPU lock path (required for --run-gpu)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/local.env" ] && source "$ROOT/local.env"

FFX_SRC="${FFX_SRC_DIR:-}"
OUT="${MIT_STATS_OUT:-$ROOT/build/round13-mit-stats}"
LOCK="${STUDY_GPU_LOCK:-}"
RUN_GPU=0
GROUP1_ONLY=0
JOBS="${JOBS:-8}"

while [ $# -gt 0 ]; do
  case "$1" in
    --ffx-src) FFX_SRC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --lock) LOCK="$2"; shift 2 ;;
    --run-gpu) RUN_GPU=1; shift ;;
    --no-gpu) RUN_GPU=0; shift ;;
    --group1-only) GROUP1_ONLY=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Usage: $0 --ffx-src DIR --out DIR [--lock FILE] [--no-gpu|--run-gpu] [--group1-only]" >&2; exit 2 ;;
  esac
done

if [ -z "$FFX_SRC" ] || [ ! -d "$FFX_SRC/sdk/include/FidelityFX/gpu" ]; then
  echo "ERROR: --ffx-src / FFX_SRC_DIR must point to FidelityFX-SDK v1.1.4 checkout" >&2
  exit 2
fi

for tool in glslc spirv-val cc python3; do
  command -v "$tool" >/dev/null || { echo "ERROR: missing required tool: $tool" >&2; exit 2; }
done

mkdir -p "$OUT/bin" "$OUT/spv" "$OUT/asm" "$OUT/logs"

echo "[1/5] Building vk_pipestats and running CPU-only checks"
cc -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers \
  "$ROOT/src/round13/vk_pipestats.c" -o "$OUT/bin/vk_pipestats" -lvulkan

if grep -q "vkCmdDispatch" "$ROOT/src/round13/vk_pipestats.c"; then
  echo "ERROR: vk_pipestats.c must not call vkCmdDispatch" >&2
  exit 1
fi
echo "STATIC CHECK no-dispatch in vk_pipestats.c: PASS"

python3 "$ROOT/scripts/round13/isa_classify.py" --self-test

echo "[2/5] Generating FidelityFX SDK v1.1.4 shader compilation & permutation plan"
# Every define below is copied verbatim from FidelityFX-SDK v1.1.4 CMakeCompile*Shaders.txt
# and sdk/src/backends/vk/CMakeShaders*.txt (commit c6efa6bf7f2027b3ec94f28578bb5965eabb9e55):
#   - Global Vulkan base args (sdk/src/backends/vk/CMakeShaders*.txt:24-25):
#       --target-env=vulkan1.2 -Os -DFFX_GLSL=1 -DFFX_GPU=1
#   - Wave64 in Vulkan (sdk/src/backends/vk/ffx_vk.cpp:3625-3633):
#       selected at VkComputePipelineCreateInfo via requiredSubgroupSize=64 (same SPIR-V).
#   - FP16 permutations (sdk/include/FidelityFX/gpu/CMakeCompileShaders.txt:92,102,112,122):
#       -DFFX_HALF=0 vs -DFFX_HALF=1
#   - FSR 3.1 Upscaler (sdk/include/FidelityFX/gpu/fsr3upscaler/CMakeCompileFSR3UpscalerShaders.txt:23-42):
#       UPSAMPLE_SAMPLERS_USE_DATA_HALF=0, ACCUMULATE_SAMPLERS_USE_DATA_HALF=0,
#       REPROJECT_SAMPLERS_USE_DATA_HALF=1, POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0,
#       UPSAMPLE_USE_LANCZOS_TYPE=2; options: REPROJECT_USE_LANCZOS_TYPE={0,1}, HDR_COLOR_INPUT={0,1},
#       LOW_RESOLUTION_MOTION_VECTORS={0,1}, JITTERED_MOTION_VECTORS={0,1},
#       INVERTED_DEPTH={0,1}, APPLY_SHARPENING={0,1}.
#       Canonical game config: REPROJECT_USE_LANCZOS_TYPE=0, HDR=1, LOW_RES_MV=1, JITTERED_MV=0,
#       INVERTED_DEPTH=1, APPLY_SHARPENING=0 (plus APPLY_SHARPENING=1 for accumulate_sharpen).
#   - FSR 3.1 Optical Flow (sdk/include/FidelityFX/gpu/opticalflow/CMakeCompileOpticalflowShaders.txt:23-28):
#       FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT={0,1} (canonical: 1; also test prepare_luma with 0).
#   - FSR 3.1 Frame Interpolation (sdk/include/FidelityFX/gpu/frameinterpolation/CMakeCompileFrameinterpolationShaders.txt:23-35):
#       UPSAMPLE_SAMPLERS_USE_DATA_HALF=0, ACCUMULATE_SAMPLERS_USE_DATA_HALF=0,
#       REPROJECT_SAMPLERS_USE_DATA_HALF=1, POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0,
#       UPSAMPLE_USE_LANCZOS_TYPE=2; options: LOW_RES_MOTION_VECTORS=1, JITTER_MOTION_VECTORS=0, INVERTED_DEPTH=1.
#   - FSR 1 (sdk/include/FidelityFX/gpu/fsr1/CMakeCompileFSR1Shaders.txt:23-30):
#       APPLY_RCAS=1, RCAS_PASSTHROUGH_ALPHA=0, SRGB_CONVERSIONS=0 (plus EASU APPLY_RCAS=0).
#   - CAS (sdk/include/FidelityFX/gpu/cas/CMakeCompileCASShaders.txt:23-28):
#       SHARPEN_ONLY={0,1}, COLOR_SPACE_CONVERSION=0 (linear) and 4 (sRGB in/out).
#   - SPD (sdk/include/FidelityFX/gpu/spd/CMakeCompileSPDShaders.txt:23-29):
#       LINEAR_SAMPLE=0, WAVE_INTEROP_LDS={0,1}, DOWNSAMPLE_FILTER=0 (mean).
#   - Blur (sdk/include/FidelityFX/gpu/blur/CMakeCompileBlurShaders.txt:23-28):
#       KERNEL_DIMENSION={3,15}, KERNEL_PERMUTATION=0.
#   - LPM (sdk/include/FidelityFX/gpu/lpm/CMakeCompileLPMShaders.txt:23-25): no extra options.
#   - Denoiser (sdk/include/FidelityFX/gpu/denoiser/CMakeCompileDenoiserShaders.txt:23-27):
#       INVERTED_DEPTH=1.
#   - CACAO (sdk/include/FidelityFX/gpu/cacao/CMakeCompileCACAOShaders.txt:23-28):
#       APPLY_SMART=1.
#   - SSSR (sdk/include/FidelityFX/gpu/sssr/CMakeCompileSSSRShaders.txt:23-27):
#       INVERTED_DEPTH=1.
#   - Parallel Sort (sdk/include/FidelityFX/gpu/parallelsort/CMakeCompileParallelSortShaders.txt:23-27):
#       HAS_PAYLOAD=1 (plus scatter/reduce HAS_PAYLOAD=0).

python3 - "$FFX_SRC" "$OUT" "$GROUP1_ONLY" <<'PY'
import json, sys
from pathlib import Path

ffx_src = Path(sys.argv[1])
out = Path(sys.argv[2])
group1_only = int(sys.argv[3])

gpu_inc = ffx_src / "sdk/include/FidelityFX/gpu"
vk_shd = ffx_src / "sdk/src/backends/vk/shaders"

# Each entry: (group, component, pass_label, rel_glsl, sdk_selects_fp16, extra_defines, cite)
SPECS = []

# --- Group 1: FSR 3.1 Upscaler (10 passes + accumulate_sharpen) ---
# sdk/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp:85-341
fsr3_base = [
    "-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
    "-DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2",
    "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0",
    "-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1",
    "-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1",
    "-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0",
    "-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=1",
]
fsr3_passes = [
    ("prepare_inputs",         "fsr3upscaler/ffx_fsr3upscaler_prepare_inputs_pass.glsl",         False, 0),
    ("luma_pyramid",           "fsr3upscaler/ffx_fsr3upscaler_luma_pyramid_pass.glsl",           False, 0),
    ("shading_change_pyramid", "fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid_pass.glsl", False, 0),
    ("shading_change",         "fsr3upscaler/ffx_fsr3upscaler_shading_change_pass.glsl",         True,  0),
    ("prepare_reactivity",     "fsr3upscaler/ffx_fsr3upscaler_prepare_reactivity_pass.glsl",     True,  0),
    ("luma_instability",       "fsr3upscaler/ffx_fsr3upscaler_luma_instability_pass.glsl",       False, 0),
    ("accumulate",             "fsr3upscaler/ffx_fsr3upscaler_accumulate_pass.glsl",             True,  0),
    ("accumulate_sharpen",     "fsr3upscaler/ffx_fsr3upscaler_accumulate_pass.glsl",             True,  1),
    ("rcas",                   "fsr3upscaler/ffx_fsr3upscaler_rcas_pass.glsl",                   False, 0),
    ("autogen_reactive",       "fsr3upscaler/ffx_fsr3upscaler_autogen_reactive_pass.glsl",       True,  0),
    ("debug_view",             "fsr3upscaler/ffx_fsr3upscaler_debug_view_pass.glsl",             False, 0),
]
for name, rel, uses16, sharp in fsr3_passes:
    SPECS.append((
        1, "fsr3upscaler", name, rel, uses16,
        fsr3_base + [f"-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={sharp}"],
        "CMakeCompileFSR3UpscalerShaders.txt:23-42; ffx_fsr3upscaler_shaderblobs.cpp:85-341",
    ))

# --- Group 1: FSR 3.1 Optical Flow (7 passes) ---
# sdk/src/backends/shared/blob_accessors/ffx_opticalflow_shaderblobs.cpp:72-196 (all 7 ignore is16bit)
of_passes = [
    ("prepare_luma_hdr",      "opticalflow/ffx_opticalflow_prepare_luma_pass.glsl",                       1),
    ("prepare_luma_sdr",      "opticalflow/ffx_opticalflow_prepare_luma_pass.glsl",                       0),
    ("luminance_pyramid",     "opticalflow/ffx_opticalflow_compute_luminance_pyramid_pass.glsl",          1),
    ("scd_histogram",         "opticalflow/ffx_opticalflow_generate_scd_histogram_pass.glsl",             1),
    ("scd_divergence",        "opticalflow/ffx_opticalflow_compute_scd_divergence_pass.glsl",             1),
    ("optical_flow_v5",       "opticalflow/ffx_opticalflow_compute_optical_flow_advanced_pass_v5.glsl",   1),
    ("filter_optical_flow",   "opticalflow/ffx_opticalflow_filter_optical_flow_pass_v5.glsl",             1),
    ("scale_optical_flow",    "opticalflow/ffx_opticalflow_scale_optical_flow_advanced_pass_v5.glsl",     1),
]
for name, rel, hdr in of_passes:
    SPECS.append((
        1, "opticalflow", name, rel, False,
        [f"-DFFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT={hdr}"],
        "CMakeCompileOpticalflowShaders.txt:23-28; ffx_opticalflow_shaderblobs.cpp:72-196",
    ))

# --- Group 1: FSR 3.1 Frame Interpolation (11 passes) ---
# sdk/src/backends/shared/blob_accessors/ffx_frameinterpolation_shaderblobs.cpp:87-283 (all 11 ignore is16bit)
fi_base = [
    "-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
    "-DFFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2",
    "-DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=1",
    "-DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0",
    "-DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=1",
]
fi_passes = [
    ("reconstruct_and_dilate",            "frameinterpolation/ffx_frameinterpolation_reconstruct_and_dilate_pass.glsl"),
    ("setup",                             "frameinterpolation/ffx_frameinterpolation_setup_pass.glsl"),
    ("reconstruct_previous_depth",        "frameinterpolation/ffx_frameinterpolation_reconstruct_previous_depth_pass.glsl"),
    ("game_motion_vector_field",          "frameinterpolation/ffx_frameinterpolation_game_motion_vector_field_pass.glsl"),
    ("game_mv_inpainting_pyramid",        "frameinterpolation/ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid_pass.glsl"),
    ("optical_flow_vector_field",         "frameinterpolation/ffx_frameinterpolation_optical_flow_vector_field_pass.glsl"),
    ("disocclusion_mask",                 "frameinterpolation/ffx_frameinterpolation_disocclusion_mask_pass.glsl"),
    ("interpolation",                     "frameinterpolation/ffx_frameinterpolation_pass.glsl"),
    ("inpainting_pyramid",                "frameinterpolation/ffx_frameinterpolation_compute_inpainting_pyramid_pass.glsl"),
    ("inpainting",                        "frameinterpolation/ffx_frameinterpolation_inpainting_pass.glsl"),
    ("debug_view",                        "frameinterpolation/ffx_frameinterpolation_debug_view_pass.glsl"),
]
for name, rel in fi_passes:
    SPECS.append((
        1, "frameinterpolation", name, rel, False, fi_base,
        "CMakeCompileFrameinterpolationShaders.txt:23-35; ffx_frameinterpolation_shaderblobs.cpp:87-283",
    ))

# --- Group 1: FSR 1 (EASU + RCAS) ---
SPECS.append((
    1, "fsr1", "easu", "fsr1/ffx_fsr1_easu_pass.glsl", True,
    ["-DFFX_FSR1_OPTION_APPLY_RCAS=0", "-DFFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA=0", "-DFFX_FSR1_OPTION_SRGB_CONVERSIONS=0"],
    "CMakeCompileFSR1Shaders.txt:23-30; ffx_fsr1_shaderblobs.cpp:50-111",
))
SPECS.append((
    1, "fsr1", "rcas", "fsr1/ffx_fsr1_rcas_pass.glsl", True,
    ["-DFFX_FSR1_OPTION_APPLY_RCAS=1", "-DFFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA=0", "-DFFX_FSR1_OPTION_SRGB_CONVERSIONS=0"],
    "CMakeCompileFSR1Shaders.txt:23-30; ffx_fsr1_shaderblobs.cpp:50-111",
))

# --- Group 1: CAS ---
for name, so, cs in [
    ("scale_sharpen_linear", 0, 0),
    ("sharpen_only_linear",  1, 0),
    ("sharpen_only_srgb",    1, 4),
]:
    SPECS.append((
        1, "cas", name, "cas/ffx_cas_sharpen_pass.glsl", True,
        [f"-DFFX_CAS_OPTION_SHARPEN_ONLY={so}", f"-DFFX_CAS_COLOR_SPACE_CONVERSION={cs}"],
        "CMakeCompileCASShaders.txt:23-28; ffx_cas_shaderblobs.cpp:54-85",
    ))

# --- Group 1: SPD ---
for name, lin, lds, flt in [
    ("downsample_wave_mean", 0, 0, 0),
    ("downsample_lds_mean",  0, 1, 0),
    ("downsample_wave_max",  0, 0, 2),
]:
    SPECS.append((
        1, "spd", name, "spd/ffx_spd_downsample_pass.glsl", True,
        [f"-DFFX_SPD_OPTION_LINEAR_SAMPLE={lin}", f"-DFFX_SPD_OPTION_WAVE_INTEROP_LDS={lds}", f"-DFFX_SPD_OPTION_DOWNSAMPLE_FILTER={flt}"],
        "CMakeCompileSPDShaders.txt:23-29; ffx_spd_shaderblobs.cpp:43-85",
    ))

if not group1_only:
    # --- Group 2: Blur ---
    for dim in (3, 15):
        SPECS.append((
            2, "blur", f"blur_k{dim}", "blur/ffx_blur_pass.glsl", True,
            [f"-DFFX_BLUR_OPTION_KERNEL_DIMENSION={dim}", "-DFFX_BLUR_OPTION_KERNEL_PERMUTATION=0"],
            "CMakeCompileBlurShaders.txt:23-28; ffx_blur_shaderblobs.cpp:72-91",
        ))
    # --- Group 2: LPM ---
    SPECS.append((
        2, "lpm", "filter", "lpm/ffx_lpm_filter_pass.glsl", True, [],
        "CMakeCompileLPMShaders.txt:23-25; ffx_lpm_shaderblobs.cpp:39-65",
    ))
    # --- Group 2: Denoiser (8 passes) ---
    for name in [
        "prepare_shadow_mask",
        "shadows_tile_classification",
        "filter_soft_shadows_0",
        "filter_soft_shadows_1",
        "filter_soft_shadows_2",
        "reproject_reflections",
        "prefilter_reflections",
        "resolve_temporal_reflections",
    ]:
        SPECS.append((
            2, "denoiser", name, f"denoiser/ffx_denoiser_{name}_pass.glsl", True,
            ["-DFFX_DENOISER_OPTION_INVERTED_DEPTH=1"],
            "CMakeCompileDenoiserShaders.txt:23-27; ffx_denoiser_shaderblobs.cpp:80-290",
        ))
    # --- Group 2: CACAO (all 31 passes; representative + full) ---
    cacao_files = sorted((vk_shd / "cacao").glob("*.glsl"))
    for cf in cacao_files:
        short = cf.stem.replace("ffx_cacao_", "").replace("_pass", "")
        SPECS.append((
            2, "cacao", short, f"cacao/{cf.name}", True,
            ["-DFFX_CACAO_OPTION_APPLY_SMART=1"],
            "CMakeCompileCACAOShaders.txt:23-28; ffx_cacao_shaderblobs.cpp:165-720",
        ))
    # --- Group 2: SSSR (5 passes) ---
    for name in [
        "classify_tiles",
        "depth_downsample",
        "intersect",
        "prepare_blue_noise_texture",
        "prepare_indirect_args",
    ]:
        SPECS.append((
            2, "sssr", name, f"sssr/ffx_sssr_{name}_pass.glsl", True,
            ["-DFFX_SSSR_OPTION_INVERTED_DEPTH=1"],
            "CMakeCompileSSSRShaders.txt:23-27; ffx_sssr_shaderblobs.cpp:61-210",
        ))
    # --- Group 2: Parallel Sort (6 passes; ignores is16bit in shaderblobs.cpp:51-160) ---
    for name in [
        "setup_indirect_args",
        "sum",
        "reduce",
        "scan",
        "scan_add",
        "scatter",
    ]:
        SPECS.append((
            2, "parallelsort", name, f"parallelsort/ffx_parallelsort_{name}_pass.glsl", False,
            ["-DFFX_PARALLELSORT_OPTION_HAS_PAYLOAD=1"],
            "CMakeCompileParallelSortShaders.txt:23-27; ffx_parallelsort_shaderblobs.cpp:51-160",
        ))

compile_jobs = []
manifest_rows = []
metadata = {}

for grp, comp, pname, rel, uses16, extra, cite in SPECS:
    src = vk_shd / rel
    if not src.exists():
        raise SystemExit(f"Missing shader source: {src}")
    inc_comp = gpu_inc / comp
    for half in (0, 1):
        prec = "fp16" if half else "fp32"
        spv_id = f"{comp}__{pname}__{prec}"
        spv_path = out / "spv" / f"{spv_id}.spv"
        cmd = [
            "glslc",
            "-fshader-stage=comp",
            "--target-env=vulkan1.2",
            "-Os",
            "-DFFX_GLSL=1",
            "-DFFX_GPU=1",
            f"-DFFX_HALF={half}",
            f"-I{gpu_inc}",
            f"-I{inc_comp}",
            *extra,
            str(src),
            "-o",
            str(spv_path),
        ]
        compile_jobs.append({"spv_id": spv_id, "spv_path": str(spv_path), "cmd": cmd})
        for sg in (32, 64):
            pipe_id = f"{comp}__{pname}__{prec}__w{sg}"
            asm_path = out / "asm" / f"{pipe_id}.asm"
            manifest_rows.append(f"{pipe_id}\t{spv_path}\tmain\t{sg}\t{asm_path}")
            metadata[pipe_id] = {
                "group": grp,
                "component": comp,
                "pass": pname,
                "source": rel,
                "fp16_define": half,
                "sdk_selects_fp16": uses16,
                "sdk_dispatched": (half == 0) or uses16,
                "req_sg": sg,
                "defines": extra,
                "cite": cite,
            }

(out / "compile_jobs.json").write_text(json.dumps(compile_jobs, indent=2) + "\n")
(out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(out / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n")
print(f"Plan generated: {len(SPECS)} passes, {len(compile_jobs)} SPIR-V builds, {len(manifest_rows)} pipelines")
PY

echo "[3/5] Compiling all shaders with glslc and validating every SPIR-V with spirv-val (jobs=$JOBS)"
python3 - "$OUT/compile_jobs.json" "$JOBS" <<'PY'
import concurrent.futures, json, subprocess, sys

jobs = json.loads(open(sys.argv[1]).read())
workers = int(sys.argv[2])

def run_one(j):
    r1 = subprocess.run(j["cmd"], capture_output=True, text=True)
    if r1.returncode != 0:
        return (j["spv_id"], "glslc", r1.stderr.strip() or r1.stdout.strip())
    r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", j["spv_path"]], capture_output=True, text=True)
    if r2.returncode != 0:
        return (j["spv_id"], "spirv-val", r2.stderr.strip() or r2.stdout.strip())
    return (j["spv_id"], "ok", "")

fails = []
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    for sid, stage, msg in ex.map(run_one, jobs):
        if stage != "ok":
            fails.append((sid, stage, msg))

print(f"Compiled + spirv-val validated: {len(jobs) - len(fails)}/{len(jobs)} PASS")
if fails:
    for sid, stage, msg in fails[:20]:
        print(f"FAIL [{sid}] ({stage}): {msg}", file=sys.stderr)
    sys.exit(1)
PY

echo "[4/5] Running CPU-only SPIR-V reflection self-test + deliberate break"
FIRST_SPV="$OUT/spv/fsr3upscaler__accumulate__fp16.spv"
"$OUT/bin/vk_pipestats" --reflect-only "$FIRST_SPV" >/dev/null
if "$OUT/bin/vk_pipestats" --reflect-only "$FIRST_SPV" --entry nonexistent_ep >/dev/null 2>&1; then
  echo "ERROR: deliberate reflect break did not fail" >&2
  exit 1
fi
echo "REFLECT self-test + deliberate break: PASS"

if [ "$RUN_GPU" -eq 0 ]; then
  echo "[5/5] Skipping GPU pipeline creation (--no-gpu). Run with --run-gpu --lock <LOCK> to collect RADV/ACO stats."
  echo "VERDICT build_mit_stats (CPU-only): PASS"
  exit 0
fi

if [ -z "$LOCK" ]; then
  echo "ERROR: --run-gpu requires --lock FILE or STUDY_GPU_LOCK" >&2
  exit 2
fi

echo "[5/5] Collecting RADV/ACO pipeline stats under shared GPU lock (pipeline creation only, zero dispatches)"
mkdir -p "$(dirname "$LOCK")"

run_locked() {
  # Linux comm names are at most 15 characters, so pgrep -x needs the 15-character forms.
  for proc in helldivers2.exe HTGame.exe Client-Win64-Sh Cyberpunk2077.e; do
    if pgrep -x "$proc" >/dev/null 2>&1; then
      echo "ABORT: game process '$proc' is running" >&2
      exit 2
    fi
  done

  # Hotspot check (< 90 C) on the 1002:7550 card; fail closed when no readable, valid sensor is found.
  hot_checked=0
  for dev in /sys/class/drm/card*/device; do
    if [ -r "$dev/vendor" ] && [ -r "$dev/device" ]; then
      v=$(cat "$dev/vendor"); d=$(cat "$dev/device")
      if [ "$v" = "0x1002" ] && [ "$d" = "0x7550" ]; then
        for t in "$dev"/hwmon/hwmon*/temp2_input; do
          if [ -r "$t" ]; then
            mc=$(cat "$t" 2>/dev/null || echo "")
            if ! [[ "$mc" =~ ^[0-9]+$ ]] || [ "$mc" -le 0 ]; then
              echo "ABORT: hotspot reading '${mc:-unreadable}' invalid ($t)" >&2
              exit 2
            fi
            echo "HOTSPOT_PRE=$((mc / 1000))C ($t)"
            if [ "$mc" -ge 90000 ]; then
              echo "ABORT: hotspot >= 90C" >&2
              exit 2
            fi
            hot_checked=1
          fi
        done
      fi
    fi
  done
  if [ "$hot_checked" -ne 1 ]; then
    echo "ABORT: no readable hotspot sensor for the 1002:7550 card" >&2
    exit 2
  fi

  T_START="$(date '+%Y-%m-%d %H:%M:%S')"
  "$OUT/bin/vk_pipestats" --batch "$OUT/manifest.tsv" --jsonl-out "$OUT/pipestats.jsonl" \
    | tee "$OUT/logs/vk_pipestats.log"

  # Deliberate break on GPU pipeline creation: bad entry point must fail cleanly
  if "$OUT/bin/vk_pipestats" "$FIRST_SPV" --entry nonexistent_ep >"$OUT/logs/vk_break.log" 2>&1; then
    echo "ERROR: deliberate GPU pipeline break unexpectedly succeeded" >&2
    exit 3
  fi
  echo "DELIBERATE-BREAK GPU bad entry point: FAIL as expected (caught)"

  # Kernel amdgpu fault check
  journalctl -k --since "$T_START" --no-pager > "$OUT/logs/kern.log" 2>/dev/null || true
  if grep -Eqi 'amdgpu.*(fault|reset|hang|ring.*timeout|gpu_recover|page fault)' "$OUT/logs/kern.log"; then
    echo "ABORT: amdgpu kernel fault detected" >&2
    grep -Ei 'amdgpu.*(fault|reset|hang|ring.*timeout|gpu_recover|page fault)' "$OUT/logs/kern.log" >&2
    exit 4
  fi
  echo "ZERO_AMDGPU_FAULTS"
}

export -f run_locked
export OUT FIRST_SPV
flock -w 600 "$LOCK" bash -c 'set -euo pipefail; run_locked'

echo "Classifying all captured ACO disassemblies (STATIC instruction counts)"
python3 "$ROOT/scripts/round13/isa_classify.py" --dir "$OUT/asm" --json-out "$OUT/isa_classes.json"

python3 - "$OUT/isa_classes.json" "$OUT/pipestats.jsonl" "$OUT/metadata.json" "$OUT/summary.json" <<'PY'
import json, sys
from pathlib import Path

cls = json.loads(Path(sys.argv[1]).read_text())
meta = json.loads(Path(sys.argv[3]).read_text())
pipe = {}
for ln in Path(sys.argv[2]).read_text().splitlines():
    if ln.strip():
        row = json.loads(ln)
        pipe[row["id"]] = row

unknown_all = {}
merged = []
for pid, m in meta.items():
    p = pipe[pid]
    c = cls[pid]
    for op, cnt in c["unknown_ops"].items():
        unknown_all[op] = unknown_all.get(op, 0) + cnt
    merged.append({
        "id": pid,
        **m,
        "sg_actual": p["sg_actual"],
        "bindings": p["bindings"],
        "driver": p["driver"],
        "stats": p["stats"],
        "isa": c,
    })

Path(sys.argv[4]).write_text(json.dumps(merged, indent=2) + "\n")
print(f"Merged {len(merged)} pipelines into {sys.argv[4]}; total unknown_ops={unknown_all}")
if unknown_all:
    print(f"WARN: unrecognized opcodes found: {unknown_all}", file=sys.stderr)
PY

echo "VERDICT build_mit_stats: PASS"
