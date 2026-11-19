// Lane 9 probes: the RX 9070 XT through Vulkan compute (RADV, ACO). One mode per item of the lane brief.
// Usage: vkstudy <mode> [options]; modes: info, compile, wave, bw, fma, mat, load, check, dispatch, check2.
// check2 (leftovers C, VK-5): the large-accumulator sets of check with odd C values; every result word against the
// step model of matrix-arithmetic (products summed in steps of 4 for 16-bit inputs, of 8 for 8-bit, the running sum
// rounded to FP32 nearest even after each step). --break uses round toward zero in the model.
// Every mode prints VERDICT lines and exits 1 if one of them is FAIL.
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkh.h"

static int g_fail;
static double g_target_s = 0.25; // seconds of whole-card load per timed repetition (thermal rule: 0.2 to 1 s)
static int g_reps = 5;

static void verdict(const char *name, int pass, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#include <stdarg.h>
static void verdict(const char *name, int pass, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("VERDICT %s: %s ", name, pass ? "PASS" : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!pass) g_fail = 1;
}

static const char *arg_val(int argc, char **argv, const char *key) {
    for (int i = 0; i + 1 < argc; i++)
        if (!strcmp(argv[i], key)) return argv[i + 1];
    return NULL;
}

static int arg_has(int argc, char **argv, const char *key) {
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], key)) return 1;
    return 0;
}

static void print_samples(const char *label, SampleStats s) {
    printf("SAMPLES %s: %d samples; sclk %.0f / %.0f / %.0f MHz (min/median/max); reported power %.0f-%.0f W; "
           "hotspot %.0f-%.0f C\n",
           label, s.n, s.sclk_min, s.sclk_med, s.sclk_max, s.pw_min, s.pw_max, s.t_min, s.t_max);
}

// ---------------- fp16 helpers ----------------

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (e == 0) {
        if (!m) {
            bits = s;
        } else {
            float v = ldexpf((float)m, -24);
            return (h >> 15) ? -v : v;
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// ---------------- info (item 1) ----------------

static const char *comp_name(VkComponentTypeKHR t) {
    switch ((int)t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
    case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: return "fp8_e4m3";
    case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: return "fp8_e5m2";
    default: return "other";
    }
}

static void stage_names(VkShaderStageFlags f, char *out, size_t cap) {
    out[0] = 0;
    const struct {
        VkShaderStageFlags b;
        const char *n;
    } t[] = {{VK_SHADER_STAGE_VERTEX_BIT, "vertex"},     {VK_SHADER_STAGE_FRAGMENT_BIT, "fragment"},
             {VK_SHADER_STAGE_COMPUTE_BIT, "compute"},   {VK_SHADER_STAGE_GEOMETRY_BIT, "geometry"},
             {VK_SHADER_STAGE_TASK_BIT_EXT, "task"},     {VK_SHADER_STAGE_MESH_BIT_EXT, "mesh"},
             {VK_SHADER_STAGE_RAYGEN_BIT_KHR, "raygen"}, {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, "tess-control"},
             {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, "tess-eval"}};
    for (size_t i = 0; i < sizeof t / sizeof *t; i++)
        if (f & t[i].b) {
            strncat(out, out[0] ? "," : "", cap - strlen(out) - 1);
            strncat(out, t[i].n, cap - strlen(out) - 1);
        }
    if (!out[0]) snprintf(out, cap, "none");
}

static int mode_info(void) {
    VkPhysicalDeviceSubgroupSizeControlProperties ssc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sgp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &ssc};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cmp = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR, &sgp};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &cmp};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(G.pd, &p2);
    VkPhysicalDeviceLimits *L = &p2.properties.limits;
    char st[200];
    printf("INFO driver: %s, %s; driverID %d; conformance %u.%u.%u.%u; API %u.%u.%u; driverVersion 0x%x\n",
           drv.driverName, drv.driverInfo, (int)drv.driverID, drv.conformanceVersion.major,
           drv.conformanceVersion.minor, drv.conformanceVersion.subminor, drv.conformanceVersion.patch,
           VK_API_VERSION_MAJOR(p2.properties.apiVersion), VK_API_VERSION_MINOR(p2.properties.apiVersion),
           VK_API_VERSION_PATCH(p2.properties.apiVersion), p2.properties.driverVersion);
    stage_names(sgp.supportedStages, st, sizeof st);
    printf("INFO subgroup: default size %u; stages with subgroup ops %s; operations mask 0x%x; quad ops in all "
           "stages %u\n",
           sgp.subgroupSize, st, sgp.supportedOperations, sgp.quadOperationsInAllStages);
    stage_names(ssc.requiredSubgroupSizeStages, st, sizeof st);
    printf("INFO subgroup size control: min %u max %u; maxComputeWorkgroupSubgroups %u; required size allowed in: "
           "%s\n",
           ssc.minSubgroupSize, ssc.maxSubgroupSize, ssc.maxComputeWorkgroupSubgroups, st);
    printf("INFO compute limits: work-group count max %u x %u x %u; work-group size max %u x %u x %u; invocations max "
           "%u; shared memory %u bytes; push constants %u bytes; storage buffer range %u\n",
           L->maxComputeWorkGroupCount[0], L->maxComputeWorkGroupCount[1], L->maxComputeWorkGroupCount[2],
           L->maxComputeWorkGroupSize[0], L->maxComputeWorkGroupSize[1], L->maxComputeWorkGroupSize[2],
           L->maxComputeWorkGroupInvocations, L->maxComputeSharedMemorySize, L->maxPushConstantsSize,
           L->maxStorageBufferRange);
    printf("INFO timestamps: period %.3f ns; compute and graphics %u; queue family used %u\n", L->timestampPeriod,
           L->timestampComputeAndGraphics, G.qf);
    stage_names(cmp.cooperativeMatrixSupportedStages, st, sizeof st);
    printf("INFO cooperative matrix stages: %s\n", st);

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, &fexe};
    VkPhysicalDeviceShaderBfloat16FeaturesKHR fbf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                                                      &fclk};
    VkPhysicalDeviceShaderFloat8FeaturesEXT ff8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &fbf};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                                                        &ff8};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fcm};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vkGetPhysicalDeviceFeatures2(G.pd, &f2);
    printf("INFO features: cooperativeMatrix %u robustBufferAccess %u; shaderFloat8 %u shaderFloat8CooperativeMatrix "
           "%u; shaderBFloat16Type %u DotProduct %u CooperativeMatrix %u; shaderFloat16 %u shaderInt8 %u; "
           "subgroupSizeControl %u computeFullSubgroups %u; shaderSubgroupClock %u shaderDeviceClock %u; "
           "pipelineExecutableInfo %u\n",
           fcm.cooperativeMatrix, fcm.cooperativeMatrixRobustBufferAccess, ff8.shaderFloat8,
           ff8.shaderFloat8CooperativeMatrix, fbf.shaderBFloat16Type, fbf.shaderBFloat16DotProduct,
           fbf.shaderBFloat16CooperativeMatrix, f12.shaderFloat16, f12.shaderInt8, f13.subgroupSizeControl,
           f13.computeFullSubgroups, fclk.shaderSubgroupClock, fclk.shaderDeviceClock, fexe.pipelineExecutableInfo);
    for (uint32_t i = 0; i < G.memprops.memoryHeapCount; i++)
        printf("INFO memory heap %u: %.2f GiB%s\n", i, G.memprops.memoryHeaps[i].size / 1073741824.0,
               (G.memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : "");

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            G.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t c = 0;
    get(G.pd, &c, NULL);
    VkCooperativeMatrixPropertiesKHR *m = calloc(c, sizeof *m);
    for (uint32_t i = 0; i < c; i++) m[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    get(G.pd, &c, m);
    int all16 = 1, f16 = 0, bf16 = 0, fp8 = 0, i8 = 0;
    for (uint32_t i = 0; i < c; i++) {
        printf("ENTRY %2u: %ux%ux%u  %s x %s, %s -> %s  sat=%u scope=%s\n", i, m[i].MSize, m[i].NSize, m[i].KSize,
               comp_name(m[i].AType), comp_name(m[i].BType), comp_name(m[i].CType), comp_name(m[i].ResultType),
               m[i].saturatingAccumulation, m[i].scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : "other");
        all16 &= m[i].MSize == 16 && m[i].NSize == 16 && m[i].KSize == 16 && m[i].scope == VK_SCOPE_SUBGROUP_KHR;
        if (m[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
            f16 |= m[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR;
            bf16 |= m[i].AType == VK_COMPONENT_TYPE_BFLOAT16_KHR;
            fp8 |= m[i].AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && m[i].BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT;
        }
        i8 |= m[i].AType == VK_COMPONENT_TYPE_SINT8_KHR && m[i].BType == VK_COMPONENT_TYPE_SINT8_KHR &&
              m[i].ResultType == VK_COMPONENT_TYPE_SINT32_KHR;
    }
    free(m);
    // Known answer: the 20-entry table read on 2026-11-17 (coopmat-radv-9070xt.txt).
    verdict("info", c == 20 && all16 && f16 && bf16 && fp8 && i8,
            "%u entries (earlier read: 20), all 16x16x16 subgroup %s; f16/bf16/fp8->f32 and s8->s32 present %d%d%d%d",
            c, all16 ? "yes" : "no", f16, bf16, fp8, i8);
    return 0;
}

// ---------------- compile (machine-code checks, items 2, 3, 4, 5, 8) ----------------

// leftovers C (VK-7a): the mixed E4M3/E5M2 forms and saturating signed 8-bit after the four of item 4
static const char *MAT_TYPES[] = {"f16", "bf16", "fp8", "i8", "fp8bf8", "bf8fp8", "bf8", "i8sat"};
#define N_MAT_TYPES 8

static void grep_lines(const Pipe *p, const char *const *pats, int npat, int maxl) {
    if (!p->asm_text) return;
    char *dup = strdup(p->asm_text);
    int shown = 0;
    for (char *s = strtok(dup, "\n"); s && shown < maxl; s = strtok(NULL, "\n")) {
        for (int i = 0; i < npat; i++)
            if (strstr(s, pats[i]) && s[0] != '#') {
                printf("  %s: %s\n", p->name, s);
                shown++;
                break;
            }
    }
    free(dup);
}

static void print_stats(const Pipe *p) {
    if (!p->asm_text) return;
    char *dup = strdup(p->asm_text);
    printf("STATS %s: subgroup %u;", p->name, p->sg_size);
    for (char *s = strtok(dup, "\n"); s; s = strtok(NULL, "\n")) {
        if (!strncmp(s, "# stat ", 7) &&
            (strstr(s, "VGPRs") || strstr(s, "SGPRs") || strstr(s, "Spill") || strstr(s, "Instructions") ||
             strstr(s, "Code size") || strstr(s, "Latency") || strstr(s, "Waves")))
            printf(" %s;", s + 7);
    }
    printf("\n");
    free(dup);
}

static int mode_compile(void) {
    char spv[64], lab[96];
    const char *clock_pats[] = {"s_getreg", "s_memtime", "s_sendmsg_rtn", "s_memrealtime"};
    for (int sgi = 0; sgi < 3; sgi++) {
        uint32_t sg = sgi == 0 ? 0 : sgi == 1 ? 32 : 64;
        for (int t = 0; t < 8; t++) {
            snprintf(spv, sizeof spv, "mat_%s%s.spv", MAT_TYPES[t % 4], t >= 4 ? "_splat" : "");
            snprintf(lab, sizeof lab, "mat_%s%s_sg%u", MAT_TYPES[t % 4], t >= 4 ? "_splat" : "", sg);
            Pipe p = pipe_create(spv, lab, 64, sg, 0);
            int n = pipe_loop_report(&p, "v_wmma", t == 0 || t == 4);
            print_stats(&p);
            verdict(lab, n > 0 && n % 16 == 0 && loop_count("v_wmma") == n && !loop_count("scratch") &&
                             !loop_count("global_load") && !loop_count("buffer_load"),
                    "inner loop holds %d matrix instructions (16 per source iteration expected), no loads", n);
            if (t == 0 && sgi == 0) grep_lines(&p, clock_pats, 4, 8);
            pipe_destroy(&p);
        }
    }
    {
        Pipe p = pipe_create("mat_f16_unused.spv", "mat_f16_unused", 64, 0, 0);
        int n = pipe_loop_report(&p, "v_wmma", 1);
        printf("INFO mat_f16_unused: %d matrix instructions in the hottest loop (the deliberate break: the compiler "
               "is expected to delete them)\n",
               n);
        pipe_destroy(&p);
    }
    for (int sgi = 0; sgi < 3; sgi++) {
        uint32_t sg = sgi == 0 ? 0 : sgi == 1 ? 32 : 64;
        snprintf(lab, sizeof lab, "fma_sg%u", sg);
        Pipe p = pipe_create("fma.spv", lab, 256, sg, 0);
        pipe_loop_report(&p, "fma", 1);
        int dual = loop_count("v_dual_");
        int single = loop_count("v_fma_f32") + loop_count("v_fmac_f32") + loop_count("v_fmamk") + loop_count("v_fmaak");
        print_stats(&p);
        verdict(lab, (2 * dual + single) % 16 == 0 && 2 * dual + single > 0,
                "inner loop: %d dual-issue instructions, %d single FMAs (%d FMAs)", dual, single, 2 * dual + single);
        pipe_destroy(&p);
    }
    {
        Pipe p = pipe_create("bw.spv", "bw", 256, 0, 0);
        pipe_loop_report(&p, "load", 1);
        print_stats(&p);
        pipe_destroy(&p);
    }
    const char *lt[] = {"f16_row", "f16_col", "fp8_row", "fp8_col"};
    for (int i = 0; i < 4; i++) {
        for (int sgi = 0; sgi < 3; sgi++) {
            uint32_t sg = sgi == 0 ? 0 : sgi == 1 ? 32 : 64;
            snprintf(spv, sizeof spv, "load_%s.spv", lt[i]);
            snprintf(lab, sizeof lab, "load_%s_sg%u", lt[i], sg);
            Pipe p = pipe_create(spv, lab, 64, sg, 0);
            int n = pipe_loop_report(&p, "v_wmma", 1);
            print_stats(&p);
            printf("INFO %s: loads in loop: global_load_tr %d, global_load %d (incl. tr), buffer_load %d, "
                   "ds_ %d, v_perm/permlane %d, v_mov %d\n",
                   lab, loop_count("global_load_tr"), loop_count("global_load"), loop_count("buffer_load"),
                   loop_count("ds_"), loop_count("v_perm"), loop_count("v_mov"));
            verdict(lab, n >= 1, "inner loop holds %d matrix instructions", n);
            pipe_destroy(&p);
        }
    }
    {
        Pipe p = pipe_create("check.spv", "check_sg0", 64, 0, 0);
        const char *pats[] = {"v_wmma", "global_load", "global_store", "buffer_load", "buffer_store"};
        grep_lines(&p, pats, 5, 40);
        pipe_destroy(&p);
    }
    {
        Pipe p = pipe_create("dot.spv", "dot", 64, 0, 0);
        const char *pats[] = {"v_dot", "v_cvt", "v_fma_mix", "v_pk_", "v_mul_f16", "v_fma_f16", "v_minimum",
                              "v_maximum", "v_med3", "v_min", "v_max", "v_perm", "v_lshl", "v_and"};
        grep_lines(&p, pats, 14, 80);
        pipe_destroy(&p);
    }
    return 0;
}

// ---------------- wave (item 2) ----------------

static int mode_wave(void) {
    struct {
        const char *spv;
        uint32_t ls, req;
    } cfg[] = {{"wave.spv", 32, 0},       {"wave.spv", 64, 0},       {"wave.spv", 128, 0},
               {"wave.spv", 256, 0},      {"wave.spv", 1024, 0},     {"wave.spv", 64, 32},
               {"wave.spv", 256, 32},     {"wave.spv", 64, 64},      {"wave.spv", 256, 64},
               {"wave_coop.spv", 32, 0},  {"wave_coop.spv", 64, 0},  {"wave_coop.spv", 128, 0},
               {"wave_coop.spv", 256, 0}, {"wave_coop.spv", 64, 32}, {"wave_coop.spv", 64, 64},
               {"wave_coop.spv", 128, 64}};
    Buf out = buf_create(4096, 0), m = buf_create(1 << 20, 0);
    bind_buffers(&out, NULL, &m, NULL);
    int ok = 1;
    for (size_t i = 0; i < sizeof cfg / sizeof *cfg; i++) {
        char lab[96];
        snprintf(lab, sizeof lab, "%s_ls%u_req%u", strncmp(cfg[i].spv, "wave_coop", 9) ? "plain" : "coop", cfg[i].ls,
                 cfg[i].req);
        Pipe p = pipe_create(cfg[i].spv, lab, cfg[i].ls, cfg[i].req, 0);
        Fill f = {&out, 0};
        run_dispatch(&p, 2, NULL, 0, &f, 1, NULL);
        uint32_t r[8];
        buf_download(&out, r, sizeof r);
        char wm[200] = "";
        if (p.asm_text) {
            const char *w = strstr(p.asm_text, "v_wmma");
            if (w) {
                const char *e = strchr(w, '\n');
                size_t n = e ? (size_t)(e - w) : strlen(w);
                if (n > 150) n = 150;
                memcpy(wm, w, n);
                wm[n] = 0;
            }
        }
        int good = r[0] == p.sg_size && r[4] == r[0] && (!cfg[i].req || r[0] == cfg[i].req) &&
                   r[1] * r[0] >= cfg[i].ls;
        ok &= good;
        printf("WAVE %-22s local %4u required %2u: driver says subgroup %u; shader sees gl_SubgroupSize %u, "
               "gl_NumSubgroups %u%s%s %s\n",
               lab, cfg[i].ls, cfg[i].req, p.sg_size, r[0], r[1], wm[0] ? "; " : "", wm, good ? "" : "MISMATCH");
        pipe_destroy(&p);
    }
    verdict("wave", ok, "gl_SubgroupSize matches the pipeline executable's subgroup size and every required size");
    buf_destroy(&out);
    buf_destroy(&m);
    return 0;
}

// ---------------- bandwidth (item 3a) ----------------

static int mode_bw(int argc, char **argv) {
    double expect = arg_val(argc, argv, "--expect") ? atof(arg_val(argc, argv, "--expect")) : 633.0;
    const uint64_t bytes = 1ull << 30;
    const uint32_t n_vec4 = (uint32_t)(bytes / 16), groups = 2048;
    uint32_t *h = malloc(bytes);
    uint64_t seed = 0x1a2b3c4d;
    uint32_t sum = 0;
    for (uint64_t i = 0; i < bytes / 8; i++) {
        uint64_t v = rng_next(&seed);
        memcpy(&h[2 * i], &v, 8);
        sum += (uint32_t)v + (uint32_t)(v >> 32);
    }
    Buf in = buf_create(bytes, 0), part = buf_create(4096, 0);
    buf_upload(&in, h, bytes);
    free(h);
    bind_buffers(&in, NULL, &part, NULL);
    Pipe p = pipe_create("bw.spv", "bw", 256, 0, 0);
    int reps = 25, bad = 0;
    double gbs[25];
    Fill f = {&part, 0};
    thermal_gate();
    sampler_start();
    for (int r = 0; r < reps; r++) {
        double ns = run_dispatch(&p, groups, &n_vec4, 4, &f, 1, NULL);
        uint32_t pr[1024], s = 0;
        buf_download(&part, pr, sizeof pr);
        for (int i = 0; i < 1024; i++) s += pr[i];
        if (s != sum) bad++;
        gbs[r] = bytes / ns;
    }
    SampleStats st = sampler_stop();
    double med = median(gbs + 1, reps - 1);
    printf("RESULT bw: 1 GiB random data, %u work-groups of 256, 16-byte loads, 4 in flight: median %.1f GB/s "
           "(min %.1f, max %.1f) over %d dispatches after one warm-up; sum checks failed: %d\n",
           groups, med, vmin(gbs + 1, reps - 1), vmax(gbs + 1, reps - 1), reps - 1, bad);
    print_samples("bw", st);
    verdict("bw", !bad && fabs(med / expect - 1) <= 0.10, "median %.1f GB/s against expected %.1f GB/s (+-10 %%); "
            "sum check %s", med, expect, bad ? "FAILED" : "exact");
    pipe_destroy(&p);
    buf_destroy(&in);
    buf_destroy(&part);
    return 0;
}

// ---------------- timed whole-card job with work stealing (items 3b, 4, 5) ----------------

typedef struct {
    double ns;
    double clk_mhz;    // median over subgroups of shader-clock cycles / realtime seconds
    double rt_ratio;   // max realtime ticks x 10 ns / dispatch time (sanity of the 100 MHz assumption)
    uint32_t *done;    // per subgroup
} RunOut;

// push layout shared by mat and fma: iters, chunks, a, b
// warm: push constants of a warm-up dispatch run first in the same submission (NULL = none). The chunk count is
// the second push word in every timed shader (load: third), so callers build it.
static RunOut timed_run(const Pipe *p, uint32_t groups, uint32_t nsub, const void *warm, void *push, size_t pb,
                        Buf *out, Buf *info, uint32_t out_fill) {
    RunOut r = {0};
    Fill f[2] = {{out, out_fill}, {info, 0}};
    r.ns = run_dispatch_warm(p, groups, warm, push, pb, f, 2, NULL);
    size_t ib = 16 + 16ull * nsub;
    uint32_t *iv = malloc(ib);
    buf_download(info, iv, ib);
    double *clk = malloc(sizeof(double) * nsub);
    int nc = 0;
    double rtmax = 0;
    r.done = malloc(4 * nsub);
    for (uint32_t s = 0; s < nsub; s++) {
        uint32_t *q = iv + 4 + 4 * s;
        r.done[s] = q[0];
        double cyc = (double)q[1] + 4294967296.0 * q[2], rt = q[3];
        if (rt > rtmax) rtmax = rt;
        if (rt > 0 && q[0] > 0) clk[nc++] = cyc / (rt * 10.0) * 1e3; // MHz if realtime runs at 100 MHz
    }
    r.clk_mhz = nc ? median(clk, nc) : NAN;
    r.rt_ratio = rtmax * 10.0 / r.ns;
    free(clk);
    free(iv);
    return r;
}

static uint32_t req_sg(int argc, char **argv) {
    const char *v = arg_val(argc, argv, "--sg");
    return v ? (uint32_t)atoi(v) : 0;
}

// sum over j < n of (j & 3): the per-chunk start offsets the timed shaders add
static uint64_t s3(uint64_t n) { return 6 * (n / 4) + (n % 4) * ((n % 4) - 1) / 2; }

// ---------------- fma (item 3b) ----------------

static int mode_fma(int argc, char **argv) {
    uint32_t sg = req_sg(argc, argv);
    char lab[64];
    snprintf(lab, sizeof lab, "fma_sg%u", sg);
    Pipe p = pipe_create("fma.spv", lab, 256, sg, 0);
    pipe_loop_report(&p, "fma", 0);
    int dual = loop_count("v_dual_"), single = loop_count("v_fma_f32") + loop_count("v_fmac_f32") +
                                               loop_count("v_fmamk") + loop_count("v_fmaak");
    int fmas = 2 * dual + single, issue = dual + single;
    uint32_t sgs = p.sg_size;
    const uint32_t nsub = 2048, groups = nsub * sgs / 256;
    Buf out = buf_create(4ull * nsub * sgs, 0), info = buf_create(16 + 16ull * nsub, 0), ab = buf_create(512, 0);
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    buf_upload(&ab, ones, sizeof ones);
    bind_buffers(&out, &info, &ab, NULL);
    struct {
        uint32_t iters, chunks;
        float a, b;
    } pc = {1024, nsub, 1.0f, 1.0f};
    // calibration: one chunk per subgroup
    thermal_gate();
    // calibration with a warm-up of the same size first, so the clock has ramped
    pc.chunks = 4 * nsub;
    RunOut c = timed_run(&p, groups, nsub, &pc, &pc, sizeof pc, &out, &info, 0xffffffffu);
    free(c.done);
    double per_chunk = c.ns / pc.chunks;
    uint32_t chunks = (uint32_t)(g_target_s * 1e9 / per_chunk);
    pc.chunks = chunks;
    __typeof__(pc) warm = pc;
    warm.chunks = chunks / 3 > nsub ? chunks / 3 : nsub;
    double tf[16], clk[16], ratio[16];
    int bad = 0, reps = g_reps;
    uint32_t *res = malloc(4ull * nsub * sgs);
    // Machine code: per SIMD per cycle, wave32 issues one instruction (a dual-issue one carries 2 FMAs);
    // INFERRED for wave64: one pass per instruction (RDNA wave64 FP32 runs both halves in one cycle).
    double fma_per_simd_cycle = (double)fmas / issue * (sgs == 64 ? 64 : 32);
    for (int r = 0; r < reps; r++) {
        thermal_gate();
        sampler_start();
        RunOut o = timed_run(&p, groups, nsub, &warm, &pc, sizeof pc, &out, &info, 0xffffffffu);
        SampleStats st = sampler_stop();
        buf_download(&out, res, 4ull * nsub * sgs);
        uint64_t tot = 0;
        int rb = 0;
        for (uint32_t s = 0; s < nsub; s++) {
            tot += o.done[s];
            for (uint32_t l = 0; l < sgs; l++) {
                uint64_t d = o.done[s];
                uint32_t want = (uint32_t)(d * (16 * (l & 3) + 480 + 16ull * pc.iters) + 16 * s3(d));
                if (res[s * sgs + l] != want) rb++;
            }
        }
        if (tot != chunks) rb++;
        bad += rb != 0;
        double fma = (double)chunks * pc.iters * 16.0 * sgs;
        tf[r] = 2 * fma / o.ns / 1e3;
        clk[r] = o.clk_mhz;
        ratio[r] = fma / (o.ns * 1e-9) / (128.0 * fma_per_simd_cycle * o.clk_mhz * 1e6);
        printf("REP fma %d: %.3f s, %.2f TFLOPS (%.2f T FMA/s), counter clock %.0f MHz, ratio to machine-code rate "
               "%.3f, realtime/timestamp %.3f, result errors %d\n",
               r, o.ns * 1e-9, tf[r], tf[r] / 2, o.clk_mhz, ratio[r], o.rt_ratio, rb);
        print_samples("fma", st);
        free(o.done);
    }
    double mr = median(ratio, reps);
    printf("RESULT fma sg%u: %d FMAs per loop in %d instructions (%d dual-issue): %.1f FMA per SIMD per cycle "
           "implied; median %.2f TFLOPS (min %.2f max %.2f), clock %.0f MHz, ratio %.3f (min %.3f max %.3f); "
           "%u subgroups of %u, %u chunks of %u iterations\n",
           sg, fmas, issue, dual, fma_per_simd_cycle, median(tf, reps), vmin(tf, reps), vmax(tf, reps),
           median(clk, reps), mr, vmin(ratio, reps), vmax(ratio, reps), nsub, sgs, chunks, pc.iters);
    verdict(lab, !bad && mr >= 0.90 && mr <= 1.02,
            "rate / (128 SIMDs x %.0f FMA per cycle x counter clock) = %.3f, expected 0.90-1.02; results %s",
            fma_per_simd_cycle, mr, bad ? "WRONG" : "exact");
    free(res);
    pipe_destroy(&p);
    buf_destroy(&out);
    buf_destroy(&info);
    buf_destroy(&ab);
    return 0;
}

// ---------------- mat (item 4) ----------------

static int mode_mat(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "mat <f16|bf16|fp8|i8|fp8bf8|bf8fp8|bf8|i8sat> [--sg 32|64] [--w N] [--unused] [--expect-half]\n");
        return 2;
    }
    const char *ty = argv[2];
    int t = -1;
    for (int i = 0; i < N_MAT_TYPES; i++)
        if (!strcmp(ty, MAT_TYPES[i])) t = i;
    if (t < 0) return 2;
    int unused = arg_has(argc, argv, "--unused"), splat = arg_has(argc, argv, "--splat");
    uint32_t sg = req_sg(argc, argv);
    uint32_t w = arg_val(argc, argv, "--w") ? (uint32_t)atoi(arg_val(argc, argv, "--w")) : 8;
    char spv[64], lab[96];
    snprintf(spv, sizeof spv, "mat_%s%s%s.spv", ty, splat ? "_splat" : "", unused ? "_unused" : "");
    snprintf(lab, sizeof lab, "mat_%s%s_sg%u_w%u%s", ty, splat ? "_splat" : "", sg, w, unused ? "_unused" : "");
    Pipe p = pipe_create(spv, lab, 64, sg, 0);
    int nw = pipe_loop_report(&p, "v_wmma", 0);
    uint32_t sgs = p.sg_size, nsub = 128 * w, groups = nsub * sgs / 64;
    const int itype = t == 3 || t == 7;
    // --expect-half (leftovers C, the deliberate break of VK-7a): claims half the HIP rate
    const double cyc = t < 2 ? 16.0 : 8.0, hip = (t < 2 ? 211.0 : 420.0) * (arg_has(argc, argv, "--expect-half") ? 0.5 : 1.0);
    Buf out = buf_create(4096ull * nsub, 0), info = buf_create(16 + 16ull * nsub, 0), ab = buf_create(4096, 0);
    // A and B for the loaded form: every element 0.25 (1 for 8-bit integers), the same values as the splat form.
    {
        uint8_t h[1024];
        for (int i = 0; i < 512; i++) {
            const uint16_t v16 = t == 0 ? 0x3400 : 0x3E80; // 0.25 in FP16, BF16
            // 8-bit: A in bytes 0-255, B in 256-511; 0.25 is 0x28 in E4M3 and 0x34 in E5M2; 1 in int8
            const int bpart = i >= 256, e5 = (t == 4 && bpart) || (t == 5 && !bpart) || t == 6;
            if (t < 2) memcpy(h + 2 * i, &v16, 2);
            else h[i] = itype ? 0x01 : e5 ? 0x34 : 0x28;
        }
        buf_upload(&ab, h, sizeof h);
    }
    bind_buffers(&out, &info, &ab, &ab); // B is read from binding 3 (the same buffer; leftovers C)
    struct {
        uint32_t iters, chunks;
        float a, b;
    } pc = {256, nsub, itype ? 1.0f : 0.25f, itype ? 1.0f : 0.25f};
    const double inc = itype ? 16.0 : 1.0; // per multiply-add, every element
    thermal_gate();
    pc.chunks = 16 * nsub;
    RunOut c = timed_run(&p, groups, nsub, &pc, &pc, sizeof pc, &out, &info, 0xffffffffu);
    free(c.done);
    double per_chunk = c.ns / pc.chunks;
    uint32_t chunks = (uint32_t)(g_target_s * 1e9 / per_chunk);
    if (unused) chunks = nsub * 64; // no calibration possible when the work is gone; a fixed count
    pc.chunks = chunks;
    __typeof__(pc) warm = pc;
    warm.chunks = chunks / 3 > nsub ? chunks / 3 : nsub;
    double tf[16], clk[16], rth[16];
    int bad = 0, reps = g_reps;
    uint32_t *res = malloc(4096ull * nsub);
    for (int r = 0; r < reps; r++) {
        thermal_gate();
        sampler_start();
        RunOut o = timed_run(&p, groups, nsub, &warm, &pc, sizeof pc, &out, &info, 0xffffffffu);
        SampleStats st = sampler_stop();
        buf_download(&out, res, 4096ull * nsub);
        uint64_t tot = 0;
        long rb = 0;
        for (uint32_t s = 0; s < nsub; s++) {
            tot += o.done[s];
            uint64_t d = o.done[s];
            uint32_t want = (uint32_t)(d * (6 + (uint64_t)(16.0 * pc.iters * inc)) + 4 * s3(d));
            for (int e = 0; e < 256; e++) {
                if (res[s * 1024 + e] != want) {
                    if (!rb)
                        printf("MISMATCH subgroup %u entry %d: got %u expected %u (chunks done %u)\n", s, e,
                               res[s * 1024 + e], want, o.done[s]);
                    rb++;
                }
            }
        }
        if (tot != chunks) rb++;
        bad += rb != 0;
        double mma = (double)chunks * pc.iters * 16.0;
        tf[r] = mma * 8192.0 / o.ns / 1e3;
        clk[r] = o.clk_mhz;
        rth[r] = (mma / (o.ns * 1e-9)) / (128.0 * o.clk_mhz * 1e6 / cyc);
        double *dd = malloc(sizeof(double) * nsub);
        int zero = 0;
        for (uint32_t q = 0; q < nsub; q++) {
            dd[q] = o.done[q];
            zero += o.done[q] == 0;
        }
        printf("REP %s %d: %.3f s, %.1f TFLOPS, %.2f G instr/s, counter clock %.0f MHz, ratio to 128 x clock / %.0f "
               "= %.3f, realtime/timestamp %.3f, wrong elements %ld; chunks per subgroup min %.0f median %.0f max %.0f, "
               "%d subgroups got none\n",
               lab, r, o.ns * 1e-9, tf[r], mma / o.ns, o.clk_mhz, cyc, rth[r], o.rt_ratio, rb, vmin(dd, nsub),
               median(dd, nsub), vmax(dd, nsub), zero);
        free(dd);
        print_samples(lab, st);
        free(o.done);
    }
    double m = median(tf, reps);
    printf("RESULT %s: subgroup %u, %u subgroups (%u per SIMD on average), %d matrix instructions in the loop; "
           "median %.1f TFLOPS (min %.1f max %.1f) against HIP %.0f: %.3f; counter clock %.0f MHz; ratio to theoretical "
           "%.3f (min %.3f max %.3f)\n",
           lab, sgs, nsub, w, nw, m, vmin(tf, reps), vmax(tf, reps), hip, m / hip, median(clk, reps),
           median(rth, reps), vmin(rth, reps), vmax(rth, reps));
    if (t >= 4) {
        // leftovers C (VK-7a): judged at the clock the bursts ran at (they are short, so part of them falls in the
        // clock ramp): the rate against 128 SIMDs x counter clock / 8 cycles (matrix-timing item 2) within 3 %;
        // --expect-half claims 4 cycles, so the expected ratio is 2
        const double want = arg_has(argc, argv, "--expect-half") ? 2.0 : 1.0, rr = median(rth, reps);
        verdict(lab, !bad && fabs(rr / want - 1) <= 0.03,
                "median ratio to 128 x counter clock / 8 = %.3f (expected %.1f +-3 %%; %.1f TFLOPS at %.0f MHz); results %s", rr,
                want, m, median(clk, reps), bad ? "WRONG (work deleted or miscounted)" : "exact");
    } else {
        verdict(lab, !bad && fabs(m / hip - 1) <= 0.10,
                "median %.1f TFLOPS against the HIP figure %.0f (+-10 %%): %.3f; results %s", m, hip, m / hip,
                bad ? "WRONG (work deleted or miscounted)" : "exact");
    }
    free(res);
    pipe_destroy(&p);
    buf_destroy(&out);
    buf_destroy(&info);
    buf_destroy(&ab);
    return 0;
}

// ---------------- load (item 5) ----------------

static int mode_load(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "load <f16|fp8> <row|col> <tight|wide> <l2|mall|dram> [--sg N] [--w N] [--swap-check]\n");
        return 2;
    }
    int fp8 = !strcmp(argv[2], "fp8"), col = !strcmp(argv[3], "col"), wide = !strcmp(argv[4], "wide");
    const char *src = argv[5];
    int swap = arg_has(argc, argv, "--swap-check");
    // leftovers C: --variant pipe | pipebrk | extra8 selects load_<t>_<layout>_<variant>.spv (VK-3, VK-4)
    const char *variant = arg_val(argc, argv, "--variant");
    uint32_t sg = req_sg(argc, argv);
    uint32_t w = arg_val(argc, argv, "--w") ? (uint32_t)atoi(arg_val(argc, argv, "--w")) : 4;
    const uint32_t es = fp8 ? 1 : 2;
    const uint64_t half = 512ull << 20; // bytes per operand buffer
    uint64_t win_bytes = !strcmp(src, "l2") ? (2ull << 20) : !strcmp(src, "mall") ? (16ull << 20) : half;
    const uint32_t stride = wide ? 4096 : 16, tpr = stride / 16;
    const uint32_t window = (uint32_t)(win_bytes / (256 * es)), T = 64;
    char spv[64], lab[128];
    snprintf(spv, sizeof spv, "load_%s_%s%s%s.spv", fp8 ? "fp8" : "f16", col ? "col" : "row", variant ? "_" : "",
             variant ? variant : "");
    snprintf(lab, sizeof lab, "load_%s_%s_%s_%s_sg%u_w%u%s%s%s", fp8 ? "fp8" : "f16", col ? "col" : "row",
             wide ? "wide" : "tight", src, sg, w, swap ? "_swapcheck" : "", variant ? "_" : "", variant ? variant : "");
    Pipe p = pipe_create(spv, lab, 64, sg, 0);
    pipe_loop_report(&p, "v_wmma", variant != NULL); // the leftovers variants print their loop body (listing check)
    int ntr = loop_count("global_load_tr"), nld = loop_count("global_load") + loop_count("buffer_load");
    uint32_t sgs = p.sg_size, nsub = 128 * w, groups = nsub * sgs / 64;

    // Data: random -1, 0, 1 so every product and sum is an exact integer in FP32.
    const uint64_t n = half / es;
    uint8_t *ha = malloc(half), *hb = malloc(half);
    uint64_t seed = 0x5eed0001;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t r = rng_next(&seed);
        int va = (int)(r % 3) - 1, vb = (int)((r >> 8) % 3) - 1;
        if (fp8) {
            ha[i] = va == 0 ? 0 : va > 0 ? 0x38 : 0xB8;
            hb[i] = vb == 0 ? 0 : vb > 0 ? 0x38 : 0xB8;
        } else {
            uint16_t xa = va == 0 ? 0 : va > 0 ? 0x3C00 : 0xBC00, xb = vb == 0 ? 0 : vb > 0 ? 0x3C00 : 0xBC00;
            memcpy(ha + 2 * i, &xa, 2);
            memcpy(hb + 2 * i, &xb, 2);
        }
    }
    // Per tile of the window: sum over all C elements of A_t x B_t = sum_k colsumA_k x rowsumB_k.
    int64_t wsum = 0;
    int a_col = swap ? !col : col; // --swap-check: the host reads the layout the wrong way round on purpose
    for (uint32_t g = 0; g < window; g++) {
        uint64_t off = (uint64_t)(g / tpr) * 16 * stride + (uint64_t)(g % tpr) * 16;
        int ca[16] = {0}, rb[16] = {0};
        for (int r = 0; r < 16; r++)
            for (int cc = 0; cc < 16; cc++) {
                uint64_t idx = off + (uint64_t)r * stride + cc;
                int va, vb;
                if (fp8) {
                    va = ha[idx] == 0 ? 0 : ha[idx] == 0x38 ? 1 : -1;
                    vb = hb[idx] == 0 ? 0 : hb[idx] == 0x38 ? 1 : -1;
                } else {
                    uint16_t xa, xb;
                    memcpy(&xa, ha + 2 * idx, 2);
                    memcpy(&xb, hb + 2 * idx, 2);
                    va = xa == 0 ? 0 : xa == 0x3C00 ? 1 : -1;
                    vb = xb == 0 ? 0 : xb == 0x3C00 ? 1 : -1;
                }
                // memory block M[r][cc]; row-major A(i,k)=M[i][k] so colsumA_k sums over r at cc=k;
                // column-major A(i,k)=M[k][i] so colsumA_k sums over cc at r=k. B is the mirror image.
                if (!a_col) ca[cc] += va; else ca[r] += va;
                if (!col) rb[r] += vb; else rb[cc] += vb;
            }
        for (int k = 0; k < 16; k++) wsum += (int64_t)ca[k] * rb[k];
    }
    Buf A = buf_create(half, 0), B = buf_create(half, 0), out = buf_create(1024ull * nsub, 0),
        info = buf_create(16 + 16ull * nsub, 0);
    buf_upload(&A, ha, half);
    buf_upload(&B, hb, half);
    free(ha);
    free(hb);
    bind_buffers(&out, &info, &A, &B);
    struct {
        uint32_t window, tiles, chunks, stride, tpr, step;
    } pc = {window, T, window / T, stride, tpr, tpr == 1 ? 16 * stride : 16};
    const uint32_t per_pass = window / T;
    uint32_t chunks;
    if (!strcmp(src, "dram")) {
        chunks = 4 * per_pass; // four passes over 1 GiB: each pass is far larger than every cache
    } else {
        thermal_gate();
        pc.chunks = per_pass > nsub ? per_pass : nsub;
        RunOut c = timed_run(&p, groups, nsub, &pc, &pc, sizeof pc, &out, &info, 0xffffffffu);
        free(c.done);
        double per_chunk = c.ns / pc.chunks;
        uint32_t passes = (uint32_t)(g_target_s * 1e9 / per_chunk / per_pass);
        // per element |sum| grows at most 16 per tile; keep under 2^24 per subgroup with margin
        uint64_t max_tiles = (uint64_t)passes * per_pass * T / nsub;
        while (passes > 1 && max_tiles * 16 > 12000000ull) {
            passes--;
            max_tiles = (uint64_t)passes * per_pass * T / nsub;
        }
        chunks = (passes ? passes : 1) * per_pass;
    }
    pc.chunks = chunks;
    __typeof__(pc) warm = pc;
    warm.chunks = per_pass * (chunks / per_pass / 3 ? chunks / per_pass / 3 : 1);
    const int64_t want = (int64_t)(chunks / per_pass) * wsum;
    double tf[16], gbs[16], lclk[16], tpg[16];
    int bad = 0, reps = g_reps;
    float *res = malloc(1024ull * nsub);
    for (int r = 0; r < reps; r++) {
        thermal_gate();
        sampler_start();
        RunOut o = timed_run(&p, groups, nsub, &warm, &pc, sizeof pc, &out, &info, 0xffffffffu);
        SampleStats st = sampler_stop();
        buf_download(&out, res, 1024ull * nsub);
        uint64_t tot = 0;
        double got = 0;
        int nonint = 0;
        for (uint32_t s = 0; s < nsub; s++) {
            tot += o.done[s];
            for (int e = 0; e < 256; e++) {
                float v = res[s * 256 + e];
                if (v != floorf(v)) nonint++;
                got += v;
            }
        }
        int rb = tot != chunks || nonint || got != (double)want;
        bad += rb;
        double tiles = (double)chunks * T;
        tf[r] = tiles * 8192.0 / o.ns / 1e3;
        gbs[r] = tiles * 512.0 * es / o.ns;
        lclk[r] = o.clk_mhz;
        tpg[r] = tf[r] / (o.clk_mhz * 1e-3); // leftovers C: TFLOPS per GHz of the counter clock
        printf("REP %s %d: %.4f s, %.2f TFLOPS, %.0f GB/s of operands, counter clock %.0f MHz, %.3f TFLOPS per GHz, sum %.0f "
               "expected %lld%s\n",
               lab, r, o.ns * 1e-9, tf[r], gbs[r], o.clk_mhz, tpg[r], got, (long long)want, rb ? " WRONG" : "");
        print_samples(lab, st);
        free(o.done);
    }
    printf("RESULT %s: subgroup %u, %u subgroups; loop: %d transposing loads, %d loads in all; window %.0f MiB, "
           "%u chunks of %u tiles; median %.2f TFLOPS (min %.2f max %.2f), %.0f GB/s (min %.0f max %.0f)\n",
           lab, sgs, nsub, ntr, nld, win_bytes * 2 / 1048576.0, chunks, T, median(tf, reps), vmin(tf, reps),
           vmax(tf, reps), median(gbs, reps), vmin(gbs, reps), vmax(gbs, reps));
    printf("RESULT %s clock: counter %.0f MHz, %.3f TFLOPS per GHz (min %.3f max %.3f)\n", lab, median(lclk, reps),
           median(tpg, reps), vmin(tpg, reps), vmax(tpg, reps));
    // Ceilings from lane 2 (memory-system.md): the DRAM bus is 640-644 GB/s (1 % allowed for timing), and the
    // highest cache rate seen for the whole card is 9.5 TB/s (L2). A median above them would mean loads were
    // skipped. The 32 MiB window is judged against the L2 ceiling too: part of it may be served from L2.
    const double limit = !strcmp(src, "dram") ? 650.0 : 9500.0, mg = median(gbs, reps);
    verdict(lab, !bad && mg <= limit,
            "checksum over every output element %s (%u tiles); %.0f GB/s of operands against the ceiling "
            "%.0f GB/s (%s)",
            bad ? "WRONG" : "exact", chunks * T, mg, limit, !strcmp(src, "dram") ? "DRAM bus + 1 %" : "highest L2 rate");
    free(res);
    pipe_destroy(&p);
    buf_destroy(&A);
    buf_destroy(&B);
    buf_destroy(&out);
    buf_destroy(&info);
    return 0;
}

// ---------------- check (item 6) ----------------

#define NSETS 10

static int mode_check(void) {
    uint16_t a[NSETS * 256], b[NSETS * 256];
    float c[NSETS * 256];
    uint64_t seed = 0xc0ffee;
    for (int i = 0; i < NSETS * 256; i++) {
        // normal FP16 values with random mantissa in +-[2^-4, 4)
        uint64_t r = rng_next(&seed);
        a[i] = (uint16_t)(((r & 1) << 15) | ((uint32_t)(11 + (r >> 1) % 6) << 10) | ((r >> 8) & 1023));
        b[i] = (uint16_t)((((r >> 20) & 1) << 15) | ((uint32_t)(11 + (r >> 21) % 6) << 10) | ((r >> 28) & 1023));
        c[i] = (float)(((int64_t)((r >> 40) & 0xffff) - 32768) / 8192.0);
    }
    // Set 0: row 0 of A = (1, 0, ...), column 0 of B = (1, 0, ...), C[0][0] = -1, so D[0][0] = 1 x 1 - 1 = 0.
    for (int k = 0; k < 16; k++) {
        a[0 * 16 + k] = k == 0 ? 0x3C00 : 0;
        b[k * 16 + 0] = k == 0 ? 0x3C00 : 0;
    }
    c[0] = -1.0f;
    // Sets 8 and 9: every A and B element 0.25, so each D element is C + 16 x 1/16 = C + 1 exactly; C = 2^22
    // (where the timed matrix loop stopped counting) and 2^21.
    for (int i = 0; i < 256; i++) {
        a[8 * 256 + i] = a[9 * 256 + i] = b[8 * 256 + i] = b[9 * 256 + i] = 0x3400;
        c[8 * 256 + i] = 4194304.0f;
        c[9 * 256 + i] = 2097152.0f;
    }
    Buf A = buf_create(sizeof a, 0), B = buf_create(sizeof b, 0), C = buf_create(sizeof c, 0),
        D = buf_create(sizeof c, 0);
    buf_upload(&A, a, sizeof a);
    buf_upload(&B, b, sizeof b);
    buf_upload(&C, c, sizeof c);
    bind_buffers(&D, &C, &A, &B);
    uint32_t sgs[3] = {0, 32, 64};
    float d[3][NSETS * 256];
    uint32_t cks[3];
    int ok = 1;
    for (int v = 0; v < 3; v++) {
        char lab[64];
        snprintf(lab, sizeof lab, "check_sg%u", sgs[v]);
        Pipe p = pipe_create("check.spv", lab, sgs[v] ? sgs[v] : 64, sgs[v], 0);
        uint32_t ls = p.sg_size;
        pipe_destroy(&p);
        p = pipe_create("check.spv", lab, ls, sgs[v], 0); // one subgroup per work-group
        Fill f = {&D, 0x7fc00001u};
        run_dispatch(&p, NSETS, NULL, 0, &f, 1, NULL);
        buf_download(&D, d[v], sizeof d[v]);
        uint32_t h = 2166136261u;
        for (int i = 0; i < NSETS * 256; i++) {
            uint32_t u = f2u(d[v][i]);
            for (int k = 0; k < 4; k++) h = (h ^ ((u >> (8 * k)) & 255)) * 16777619u;
        }
        cks[v] = h;
        // reference: exact products in double, one rounding at the end; tolerance from the magnitudes
        double maxrel = 0, maxrel_rand = 0; // all sets; the random sets 0-7 only (8 and 9 lose products by design)
        int out_tol = 0;
        for (int s = 0; s < NSETS; s++)
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++) {
                    double acc = c[s * 256 + i * 16 + j], mag = fabs(acc);
                    for (int k = 0; k < 16; k++) {
                        double pr = (double)h2f(a[s * 256 + i * 16 + k]) * h2f(b[s * 256 + k * 16 + j]);
                        acc += pr;
                        mag += fabs(pr);
                    }
                    double err = fabs(d[v][s * 256 + i * 16 + j] - acc) / mag;
                    if (err > maxrel) maxrel = err;
                    if (s < 8 && err > maxrel_rand) maxrel_rand = err;
                    if (err > 1e-6) out_tol++;
                }
        float d00 = d[v][0];
        printf("RESULT %s: C = 2^22 plus 16 products of 1/16 gives %.1f (exact: 4194305); C = 2^21 gives %.1f "
               "(exact: 2097153)\n",
               lab, d[v][8 * 256], d[v][9 * 256]);
        printf("RESULT %s: subgroup %u; D[0][0] of set 0 = %g (bits 0x%08x); max |error| / sum of |terms| %.3g in "
               "sets 0-7 (random), %.3g over all sets; checksum (FNV-1a over %d result words) 0x%08x\n",
               lab, ls, d00, f2u(d00), maxrel_rand, maxrel, NSETS * 256, h);
        ok &= d00 == 0.0f && out_tol == 0;
        if (G.out_dir[0]) {
            char path[700];
            snprintf(path, sizeof path, "%s/coopmat-f16-check-%s.txt", G.out_dir, lab);
            FILE *fo = fopen(path, "w");
            if (fo) {
                fprintf(fo, "# Vulkan cooperative-matrix D = A x B + C, A and B FP16 (16x16, row-major: element "
                            "(row, col) at row*16+col), C and D FP32 (row-major). RADV %s, subgroup %u.\n"
                            "# One block per set: 16 lines each of A (hex FP16), B (hex FP16), C (hex FP32), D (hex "
                            "FP32). Checksum FNV-1a over D bytes (little endian, sets in order): 0x%08x\n",
                        "Mesa 27.0.0-devel", ls, h);
                for (int s = 0; s < NSETS; s++) {
                    fprintf(fo, "set %d\n", s);
                    const char *nm[4] = {"A", "B", "C", "D"};
                    for (int q = 0; q < 4; q++)
                        for (int i = 0; i < 16; i++) {
                            fprintf(fo, "%s%2d", nm[q], i);
                            for (int j = 0; j < 16; j++) {
                                int x = s * 256 + i * 16 + j;
                                if (q == 0) fprintf(fo, " %04x", a[x]);
                                if (q == 1) fprintf(fo, " %04x", b[x]);
                                if (q == 2) fprintf(fo, " %08x", f2u(c[x]));
                                if (q == 3) fprintf(fo, " %08x", f2u(d[v][x]));
                            }
                            fprintf(fo, "\n");
                        }
                }
                fclose(fo);
            }
        }
        pipe_destroy(&p);
    }
    // Large accumulators, per input type (an observation for the arithmetic lane; no verdict).
    {
        const double cs[] = {2097152.0, 4194304.0, 4194306.0, 8388608.0, 8388612.0, 16777200.0};
        enum { NB = sizeof cs / sizeof *cs };
        float cb[NB * 256];
        for (int q = 0; q < NB; q++)
            for (int i = 0; i < 256; i++) cb[q * 256 + i] = (float)cs[q];
        Buf CB = buf_create(sizeof cb, 0), DB = buf_create(sizeof cb, 0), AB = buf_create(1024, 0);
        buf_upload(&CB, cb, sizeof cb);
        const char *tn[3] = {"f16", "bf16", "fp8"};
        for (int ty = 0; ty < 3; ty++) {
            uint8_t h[1024];
            for (int i = 0; i < 512; i++) {
                const uint16_t v16 = ty == 0 ? 0x3400 : 0x3E80;
                if (ty < 2) memcpy(h + 2 * i, &v16, 2);
                else h[i] = 0x28;
            }
            buf_upload(&AB, h, sizeof h);
            bind_buffers(&DB, &CB, &AB, NULL);
            char spv[64];
            snprintf(spv, sizeof spv, "check_big_%s.spv", tn[ty]);
            Pipe p = pipe_create(spv, spv, 64, 64, 0);
            Fill f = {&DB, 0x7fc00001u};
            run_dispatch(&p, NB, NULL, 0, &f, 1, NULL);
            float db[NB * 256];
            buf_download(&DB, db, sizeof db);
            printf("RESULT big-accumulator %s: C + 16 x (0.25 x 0.25), exact C + 1:", tn[ty]);
            for (int q = 0; q < NB; q++) {
                int uni = 1;
                for (int i = 1; i < 256; i++) uni &= db[q * 256 + i] == db[q * 256];
                printf(" C=%.0f -> %.1f%s%s", cs[q], db[q * 256], db[q * 256] == (float)(cs[q] + 1) ? " (exact)" : " (LOST)",
                       uni ? "" : " (not uniform)");
            }
            printf("\n");
            pipe_destroy(&p);
        }
        buf_destroy(&CB);
        buf_destroy(&DB);
        buf_destroy(&AB);
    }
    int same = !memcmp(d[0], d[1], sizeof d[0]) && !memcmp(d[1], d[2], sizeof d[0]);
    printf("RESULT check: wave sizes default/32/64 give %s result bits (checksums 0x%08x 0x%08x 0x%08x)\n",
           same ? "identical" : "DIFFERENT", cks[0], cks[1], cks[2]);
    verdict("check", ok, "D[0][0] = 1 x 1 + (-1) is zero and every element is within 1e-6 of the exact sum");
    buf_destroy(&A);
    buf_destroy(&B);
    buf_destroy(&C);
    buf_destroy(&D);
    return 0;
}

// ---------------- check2 (leftovers C, VK-5) ----------------

// The step model for these sets: every product is 1/16; nsteps steps of `per` products each.
static float step_model(float c, int nsteps, double step, int rtz) {
    float acc = c;
    for (int s = 0; s < nsteps; s++) {
        double x = (double)acc + step; // exact in double for these magnitudes
        float r = (float)x;            // round to nearest even
        if (rtz && fabs((double)r) > fabs(x)) r = nextafterf(r, 0.0f);
        acc = r;
    }
    return acc;
}

static int mode_check2(int argc, char **argv) {
    const int rtz = arg_has(argc, argv, "--break");
    // old sets (vulkan-view item 6; even C: the calibration) and the new odd ones
    const double cs[] = {2097152.0, 4194304.0, 8388608.0,                 // old: 2^21, 2^22, 2^23
                         2097152.25, 4194304.5, 4194305.5, 4194306.5,    // 2^21 + 1/4, 2^22 + 1/2, + 3/2, + 5/2
                         8388609.0, 8388611.0, 16777218.0, 16777222.0}; // 2^23 + 1, + 3, 2^24 + 2, + 6
    enum { NB = sizeof cs / sizeof *cs };
    float cb[NB * 256];
    for (int q = 0; q < NB; q++)
        for (int i = 0; i < 256; i++) cb[q * 256 + i] = (float)cs[q];
    Buf CB = buf_create(sizeof cb, 0), DB = buf_create(sizeof cb, 0), AB = buf_create(1024, 0);
    buf_upload(&CB, cb, sizeof cb);
    const char *tn[3] = {"f16", "bf16", "fp8"};
    long bad_all = 0, words = 0, cal_bad = 0;
    for (int ty = 0; ty < 3; ty++) {
        uint8_t h[1024];
        for (int i = 0; i < 512; i++) {
            const uint16_t v16 = ty == 0 ? 0x3400 : 0x3E80;
            if (ty < 2) memcpy(h + 2 * i, &v16, 2);
            else h[i] = 0x28;
        }
        buf_upload(&AB, h, sizeof h);
        bind_buffers(&DB, &CB, &AB, NULL);
        char spv[64];
        snprintf(spv, sizeof spv, "check_big_%s.spv", tn[ty]);
        Pipe p = pipe_create(spv, spv, 64, 64, 0);
        Fill f = {&DB, 0x7fc00001u};
        run_dispatch(&p, NB, NULL, 0, &f, 1, NULL);
        float db[NB * 256];
        buf_download(&DB, db, sizeof db);
        // 16-bit inputs: 4 steps of 4 products (4 x 1/16 = 0.25 each); 8-bit: 2 steps of 8 (0.5 each)
        const int nsteps = ty < 2 ? 4 : 2;
        const double step = ty < 2 ? 0.25 : 0.5;
        printf("RESULT check2 %s: C -> card (model, exact C + 1):", tn[ty]);
        for (int q = 0; q < NB; q++) {
            const float m = step_model((float)cs[q], nsteps, step, rtz);
            long bad = 0;
            for (int i = 0; i < 256; i++) bad += f2u(db[q * 256 + i]) != f2u(m);
            bad_all += bad;
            words += 256;
            if (q < 3) cal_bad += bad;
            printf(" %.2f -> %.2f (%.2f, %.2f)%s", cs[q], db[q * 256], m, cs[q] + 1, bad ? " MISMATCH" : "");
        }
        printf("\n");
        pipe_destroy(&p);
    }
    verdict("check2", bad_all == 0 && words == 3 * NB * 256,
            "%ld of %ld result words differ from the step model (%s); of them in the old even-C sets: %ld", bad_all, words,
            rtz ? "break: round toward zero" : "round to nearest even after each step", cal_bad);
    buf_destroy(&CB);
    buf_destroy(&DB);
    buf_destroy(&AB);
    return 0;
}

// ---------------- dispatch (item 7) ----------------

static void pct(const char *label, double *v, int n) {
    double *c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && c[j - 1] > c[j]; j--) {
            double t = c[j];
            c[j] = c[j - 1];
            c[j - 1] = t;
        }
    printf("RESULT dispatch %s: median %.2f us, p10 %.2f, p90 %.2f, min %.2f, max %.2f (n = %d)\n", label,
           median(v, n) / 1e3, c[n / 10] / 1e3, c[n * 9 / 10] / 1e3, c[0] / 1e3, c[n - 1] / 1e3, n);
    free(c);
}

static int mode_dispatch(void) {
    enum { N = 200 };
    double wall[N], gpu[N];
    Buf out = buf_create(4096, 0);
    bind_buffers(&out, NULL, NULL, NULL);
    Pipe pe = pipe_create("empty.spv", "empty", 64, 0, 0), po = pipe_create("one.spv", "one", 64, 0, 0);
    // (a) empty command buffer: submit to fence
    for (int i = 0; i < N + 10; i++) {
        VK_CHECK(vkResetCommandBuffer(G.cb, 0));
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(G.cb, &bi));
        VK_CHECK(vkEndCommandBuffer(G.cb));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &G.cb;
        double t0 = now_ns();
        VK_CHECK(vkQueueSubmit(G.q, 1, &si, G.fence));
        VK_CHECK(vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, UINT64_MAX));
        double t1 = now_ns();
        VK_CHECK(vkResetFences(G.dev, 1, &G.fence));
        if (i >= 10) wall[i - 10] = t1 - t0;
    }
    pct("empty command buffer, submit to fence (wall)", wall, N);
    int sane = 1;
    const struct {
        const Pipe *p;
        uint32_t groups;
        const char *name;
    } cases[] = {{&pe, 0, "zero work-groups"}, {&pe, 1, "empty shader, 1 work-group of 64"},
                 {&po, 1, "one store, 1 work-group of 64"}, {&pe, 1024, "empty shader, 1024 work-groups"}};
    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
        for (int i = 0; i < N + 10; i++) {
            double w, g = run_dispatch(cases[c].p, cases[c].groups, NULL, 0, NULL, 0, &w);
            if (i >= 10) {
                wall[i - 10] = w;
                gpu[i - 10] = g;
                if (g < 0 || g > w) sane = 0;
            }
        }
        char l[160];
        snprintf(l, sizeof l, "%s, submit to fence (wall)", cases[c].name);
        pct(l, wall, N);
        snprintf(l, sizeof l, "%s, GPU time between timestamps", cases[c].name);
        pct(l, gpu, N);
    }
    // (b) 100 dispatches of one work-group in one command buffer, a barrier and a timestamp between each
    enum { K = 100 };
    double per[5];
    for (int r = 0; r < 5; r++) {
        VK_CHECK(vkResetCommandBuffer(G.cb, 0));
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(G.cb, &bi));
        vkCmdResetQueryPool(G.cb, G.qp, 0, K + 1);
        vkCmdBindPipeline(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, po.pipe);
        vkCmdBindDescriptorSets(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pl, 0, 1, &G.ds, 0, NULL);
        vkCmdWriteTimestamp(G.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.qp, 0);
        for (int i = 0; i < K; i++) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(G.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, NULL, 0, NULL);
            vkCmdDispatch(G.cb, 1, 1, 1);
            vkCmdWriteTimestamp(G.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.qp, (uint32_t)i + 1);
        }
        VK_CHECK(vkEndCommandBuffer(G.cb));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &G.cb;
        VK_CHECK(vkQueueSubmit(G.q, 1, &si, G.fence));
        VK_CHECK(vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(G.dev, 1, &G.fence));
        uint64_t ts[K + 1];
        VK_CHECK(vkGetQueryPoolResults(G.dev, G.qp, 0, K + 1, sizeof ts, ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        for (int i = 0; i < K; i++)
            if (ts[i + 1] < ts[i]) sane = 0;
        per[r] = (ts[K] - ts[0]) * G.ts_period_ns / K;
    }
    printf("RESULT dispatch back-to-back (100 one-work-group dispatches with barriers in one command buffer): "
           "median %.2f us per dispatch (min %.2f max %.2f over 5 command buffers)\n",
           median(per, 5) / 1e3, vmin(per, 5) / 1e3, vmax(per, 5) / 1e3);
    uint32_t one = 0;
    buf_download(&out, &one, 4);
    verdict("dispatch", sane && one == 1,
            "timestamps ordered and GPU time below wall time in every sample; the store landed (%u)", one);
    pipe_destroy(&pe);
    pipe_destroy(&po);
    buf_destroy(&out);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vkstudy <info|compile|wave|bw|fma|mat|load|check|dispatch> [options]\n");
        return 2;
    }
    const char *mode = argv[1];
    // Measured on this card: a 0.1 s dual-issue FMA burst lifts the hotspot from 57 to 91 C (reported 384 W), a
    // matrix burst far less; so FMA repetitions are kept shorter than the rest.
    if (!strcmp(mode, "fma")) g_target_s = 0.1;
    if (!strcmp(mode, "mat")) g_target_s = 0.2;
    if (getenv("VKS_TARGET_S")) g_target_s = atof(getenv("VKS_TARGET_S"));
    if (g_target_s > 1.0) g_target_s = 1.0; // thermal rule: at most about 1 s of whole-card load per repetition
    if (arg_val(argc, argv, "--reps")) g_reps = atoi(arg_val(argc, argv, "--reps"));
    if (g_reps < 1) g_reps = 1;
    if (g_reps > 16) g_reps = 16;
    vkh_init(0);
    print_clocks("start");
    int rc = 2;
    if (!strcmp(mode, "info")) rc = mode_info();
    else if (!strcmp(mode, "compile")) rc = mode_compile();
    else if (!strcmp(mode, "wave")) rc = mode_wave();
    else if (!strcmp(mode, "bw")) rc = mode_bw(argc, argv);
    else if (!strcmp(mode, "fma")) rc = mode_fma(argc, argv);
    else if (!strcmp(mode, "mat")) rc = mode_mat(argc, argv);
    else if (!strcmp(mode, "load")) rc = mode_load(argc, argv);
    else if (!strcmp(mode, "check")) rc = mode_check();
    else if (!strcmp(mode, "dispatch")) rc = mode_dispatch();
    else if (!strcmp(mode, "check2")) rc = mode_check2(argc, argv);
    else fprintf(stderr, "unknown mode %s\n", mode);
    print_clocks("end");
    printf("THERMAL highest %.0f C\n", highest_hotspot());
    vkDeviceWaitIdle(G.dev);
    if (rc) return rc;
    return g_fail ? 1 : 0;
}
