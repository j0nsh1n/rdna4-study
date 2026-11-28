// Round 13 Probe P4 (vk_async): All-Vulkan async compute beside graphics on RX 9070 XT (RADV/ACO).
//
// Architecture:
//   - One process, one active VkDevice at a time, two queues:
//     * Graphics queue: 2560x1440 RGBA16F fullscreen fragment load reused verbatim from
//       src/round12/gfx_load.c (lines 16-18, 167-220, 243-264, 333-397) in the two Round 12
//       regimes (scripts/round12/run_async_overlap.sh:83-88):
//         - SAT-100%: in_flight=2, pace_ms=0.0, iters=6450 (~3.8 ms/frame)
//         - PACE-60 : in_flight=1, pace_ms=16.6, iters=10791 (~5.7 ms/frame draw + gap)
//     * Compute queue: separate compute-only queue family if RADV exposes one (all queue families
//       reported at startup), tested across VK_KHR_global_priority / VK_EXT_global_priority levels
//       LOW / MEDIUM / HIGH / REALTIME without root (refused levels recorded as VK_ERROR_NOT_PERMITTED,
//       never retried with privileges), scheduled in ABBA palindrome order across allowed priorities.
//   - Compute jobs (each a chain of N dispatches: N=1 vs N=80 in BOTH PACE-60 and SAT-100,
//     with equal total work between N=1 and N=80, separated by compute barriers):
//     (a) fp16   : packed FP16 FMA ALU (P2's vk_tex_fp16.comp, USE_FP16=1 DO_TEX=0 FMA_PER_TEX=64)
//     (b) sad    : integer SAD-like ALU (vk_async_sad.comp; GLSL has no msad4 / v_msad_u8)
//     (c) tex    : texture-sampling heavy (P2's vk_tex.comp OP_MODE=1, 32 MiB RGBA16F MALL)
//     (d) stream : 128 MiB -> 128 MiB (256 MiB > 64 MiB MALL) coalesced uvec4 DRAM streaming (5120 MiB = 5.0 GiB)
//   - Controls & guards:
//     * Exact bit-level CPU verification of every compute job (N=1, N=80)
//     * Deliberate break (--break) corrupts GPU inputs/coords on all 4 jobs and must FAIL 4/4
//     * Fail-closed 95 C hotspot stop (VKA_THERMAL_STOP_C) (unreadable sensor = immediate abort) in every submission loop
//     * Bounded < 60 C cooldown wait before every arm & every repeat, PACE-60 before SAT-100,
//       bounded moderate fan pre-warm before SAT-100, continue after 1st 95 C stop (non-zero exit),
//       abort whole run on 2nd 95 C stop
//     * ACO disassembly & instruction verification via VK_KHR_pipeline_executable_properties
//
// Usage: vk_async [--dump-asm] [--break] [--smoke]
#define _GNU_SOURCE
#include "vk_async_ref.h"
#include "vk_tex_common.h"
#include "vk_tex_ref.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Graphics parameters reused from src/round12/gfx_load.c:16-18 and run_async_overlap.sh:83-88.
#define GFX_DW 2560u
#define GFX_DH 1440u
#define GFX_RING 4
#define GFX_SAT_ITERS 6450u
#define GFX_PACE_ITERS 10791u
#define GFX_PACE_MS 16.6
#define GFX_WARMUP_FRAMES 3
#define GFX_SAT_REPEAT_MAX_FRAMES 36
#define GFX_PACE_REPEAT_MAX_FRAMES 20
#define FAN_PREWARM_TARGET_RPM 800
#define FAN_PREWARM_SOFT_CAP_C 78.0
#define FAN_PREWARM_MAX_MS 4000.0
#define FAN_PREWARM_SMOKE_MS 2000.0

#define CHECK_WGS 4u
#define STREAM_CHK_BLOCKS 8u
#define STREAM_CHK_BLOCK_VEC4 256u
#define STREAM_CHK_TOTAL_VEC4 (STREAM_CHK_BLOCKS * STREAM_CHK_BLOCK_VEC4)

#define VKA_CHECK(x)                                                                                 \
    do {                                                                                             \
        VkResult r_ = (x);                                                                           \
        if (r_ != VK_SUCCESS) {                                                                      \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);      \
            exit(3);                                                                                 \
        }                                                                                            \
    } while (0)

enum {
    JOB_FP16   = VKA_JOB_FP16,
    JOB_SAD    = VKA_JOB_SAD,
    JOB_TEX    = VKA_JOB_TEX,
    JOB_STREAM = VKA_JOB_STREAM,
    JOB_COUNT  = VKA_JOB_COUNT
};

static const char * const k_job_names[JOB_COUNT] = {"fp16", "sad", "tex", "stream"};

enum {
    REG_SAT = 0,
    REG_PACE = 1,
    REG_COUNT = 2
};

static const char * const k_reg_names[REG_COUNT] = {"SAT-100", "PACE-60"};

static const VkQueueGlobalPriority k_all_prios[4] = {
    VK_QUEUE_GLOBAL_PRIORITY_LOW,
    VK_QUEUE_GLOBAL_PRIORITY_MEDIUM,
    VK_QUEUE_GLOBAL_PRIORITY_HIGH,
    VK_QUEUE_GLOBAL_PRIORITY_REALTIME,
};
static const char * const k_prio_names[4] = {"LOW", "MEDIUM", "HIGH", "REALTIME"};

typedef struct {
    uint32_t w, h, mask_x, mask_y;
    uint32_t strips_x_mask, strips_x_shift, tiles_y_mask, y_tile_step;
    uint32_t iters, break_dx, out_stride_wg, strip_tiles;
    float inv_w, inv_h, du, dv;
} TexPC;

typedef struct {
    uint32_t iters;
    uint32_t pass_idx;
    uint32_t out_stride_wg;
    uint32_t break_mask;
} SadPC;

typedef struct {
    uint32_t total_vec4;
} StreamInitPC;

typedef struct {
    uint32_t base_vec4;
    uint32_t chunks;
    uint32_t iters;
    uint32_t salt;
    uint32_t break_mask;
} StreamPC;

// Graphics resources matching src/round12/gfx_load.c:103-116, 164-241.
typedef struct {
    VkCommandPool pool;
    VkImage img;
    VkDeviceMemory imem;
    VkImageView iview;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VktPipe aco_info;
    VkCommandBuffer cbs[GFX_RING];
    VkQueryPool qps[GFX_RING];
    VkFence fns[GFX_RING];
} GfxRes;

typedef struct {
    VktPipe p_fp16;
    VktPipe p_sad;
    VktPipe p_tex;
    VktPipe p_stream;
    VktPipe p_stream_init;
    VktImg tex_img;
    VktBuf fp16_out;
    VktBuf fp16_kbuf;
    VktBuf sad_out;
    VktBuf sad_in;
    VktBuf tex_out;
    VktBuf stream_src;
    VktBuf stream_dst;
    VktBuf stream_chk;
    VkDescriptorSet ds_fp16;
    VkDescriptorSet ds_sad;
    VkDescriptorSet ds_tex;
    VkDescriptorSet ds_stream;
    VkDescriptorSet ds_stream_init;
    VkCommandBuffer copy_cb;
    VkFence copy_fence;
} CompRes;

static pthread_mutex_t g_shared_q_mu = PTHREAD_MUTEX_INITIALIZER;

static double mono_ms(void) {
    return vkt_now_ns() * 1e-6;
}

static void thermal_guard_or_abort(const char * where) {
    double hot = -1.0;
    if (!vkt_check_hotspot_fail_closed(&hot)) {
        fprintf(stderr, "THERMAL ABORT (fail-closed) in %s: hot=%.1f C (limit=%.0f C, sensor=%s)\n",
                where, hot, VKA_THERMAL_STOP_C, VT.hwmon_hot_path[0] ? VT.hwmon_hot_path : "(missing)");
        fflush(stderr);
        exit(4);
    }
}

static VkResult submit_queue_safe(VkQueue q, const VkSubmitInfo * si, VkFence fence) {
    if (VT.same_queue_handle) pthread_mutex_lock(&g_shared_q_mu);
    VkResult r = vkQueueSubmit(q, 1u, si, fence);
    if (VT.same_queue_handle) pthread_mutex_unlock(&g_shared_q_mu);
    return r;
}

static uint32_t ctz_u32(uint32_t v) {
    uint32_t s = 0;
    while (v > 1u) { v >>= 1; ++s; }
    return s;
}

static char * slurp_file(const char * path, size_t * out_sz) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = (char *)malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        exit(2);
    }
    fclose(f);
    *out_sz = (size_t)n;
    return b;
}

// ---- Graphics setup (reuses src/round12/gfx_load.c:164-264) ----
static void gfx_init_res(GfxRes * g, int capture_asm) {
    memset(g, 0, sizeof *g);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = VT.gfx_qf;
    VKA_CHECK(vkCreateCommandPool(VT.dev, &cpi, NULL, &g->pool));

    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ii.extent = (VkExtent3D){GFX_DW, GFX_DH, 1u};
    ii.mipLevels = 1u;
    ii.arrayLayers = 1u;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKA_CHECK(vkCreateImage(VT.dev, &ii, NULL, &g->img));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(VT.dev, g->img, &mr);
    uint32_t mtype = UINT32_MAX;
    for (uint32_t i = 0; i < VT.memprops.memoryTypeCount; ++i) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (VT.memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mtype = i;
            break;
        }
    }
    if (mtype == UINT32_MAX) {
        fprintf(stderr, "no device-local memory for graphics render target\n");
        exit(3);
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mtype};
    VKA_CHECK(vkAllocateMemory(VT.dev, &ai, NULL, &g->imem));
    VKA_CHECK(vkBindImageMemory(VT.dev, g->img, g->imem, 0));

    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = g->img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    VKA_CHECK(vkCreateImageView(VT.dev, &vi, NULL, &g->iview));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0u, 4u};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1u;
    pli.pPushConstantRanges = &pcr;
    VKA_CHECK(vkCreatePipelineLayout(VT.dev, &pli, NULL, &g->pl));

    char vp[700], fp[700];
    snprintf(vp, sizeof vp, "%s/gdraw_vert.spv", VT.spv_dir);
    snprintf(fp, sizeof fp, "%s/gdraw_frag.spv", VT.spv_dir);
    size_t vsz = 0, fsz = 0;
    char * vcode = slurp_file(vp, &vsz);
    char * fcode = slurp_file(fp, &fsz);
    VkShaderModule vs, fs;
    VkShaderModuleCreateInfo smv = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, vsz, (const uint32_t *)vcode};
    VkShaderModuleCreateInfo smf = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, fsz, (const uint32_t *)fcode};
    VKA_CHECK(vkCreateShaderModule(VT.dev, &smv, NULL, &vs));
    VKA_CHECK(vkCreateShaderModule(VT.dev, &smf, NULL, &fs));
    free(vcode);
    free(fcode);

    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL},
    };
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vpt = {0.0f, 0.0f, (float)GFX_DW, (float)GFX_DH, 0.0f, 1.0f};
    VkRect2D sc = {{0, 0}, {GFX_DW, GFX_DH}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1u;
    vps.pViewports = &vpt;
    vps.scissorCount = 1u;
    vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = 0xfu;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1u;
    cb.pAttachments = &cba;
    VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1u;
    rci.pColorAttachmentFormats = &fmt;

    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
    if (capture_asm && VT.has_exec_props) {
        gi.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR |
                   VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    gi.stageCount = 2u;
    gi.pStages = st;
    gi.pVertexInputState = &vin;
    gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vps;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pColorBlendState = &cb;
    gi.layout = g->pl;
    gi.basePipelineIndex = -1;
    VKA_CHECK(vkCreateGraphicsPipelines(VT.dev, VK_NULL_HANDLE, 1u, &gi, NULL, &g->pipe));
    vkDestroyShaderModule(VT.dev, vs, NULL);
    vkDestroyShaderModule(VT.dev, fs, NULL);

    if (capture_asm && VT.has_exec_props) {
        vkt_capture_pipe_aco(g->pipe, "vk_async_gdraw", &g->aco_info);
    }

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g->pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = GFX_RING;
    VKA_CHECK(vkAllocateCommandBuffers(VT.dev, &cai, g->cbs));
    for (int i = 0; i < GFX_RING; ++i) {
        VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2u;
        VKA_CHECK(vkCreateQueryPool(VT.dev, &qpi, NULL, &g->qps[i]));
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKA_CHECK(vkCreateFence(VT.dev, &fci, NULL, &g->fns[i]));
    }

    // One-time transition UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL (gfx_load.c:222-240).
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKA_CHECK(vkBeginCommandBuffer(g->cbs[0], &bi));
    VkImageMemoryBarrier2 ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    ib.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ib.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = g->img;
    ib.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1u;
    di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(g->cbs[0], &di);
    VKA_CHECK(vkEndCommandBuffer(g->cbs[0]));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1u;
    si.pCommandBuffers = &g->cbs[0];
    VKA_CHECK(submit_queue_safe(VT.gfx_q, &si, g->fns[0]));
    VKA_CHECK(vkWaitForFences(VT.dev, 1u, &g->fns[0], VK_TRUE, UINT64_MAX));
    VKA_CHECK(vkResetFences(VT.dev, 1u, &g->fns[0]));
}

static void gfx_destroy_res(GfxRes * g) {
    if (!g->pool) return;
    free(g->aco_info.asm_text);
    for (int i = 0; i < GFX_RING; ++i) {
        if (g->fns[i]) vkDestroyFence(VT.dev, g->fns[i], NULL);
        if (g->qps[i]) vkDestroyQueryPool(VT.dev, g->qps[i], NULL);
    }
    if (g->pipe) vkDestroyPipeline(VT.dev, g->pipe, NULL);
    if (g->pl) vkDestroyPipelineLayout(VT.dev, g->pl, NULL);
    if (g->iview) vkDestroyImageView(VT.dev, g->iview, NULL);
    if (g->img) vkDestroyImage(VT.dev, g->img, NULL);
    if (g->imem) vkFreeMemory(VT.dev, g->imem, NULL);
    vkDestroyCommandPool(VT.dev, g->pool, NULL);
    memset(g, 0, sizeof *g);
}

// Reuses src/round12/gfx_load.c:243-264 verbatim.
static void record_gfx_frame(const GfxRes * g, VkCommandBuffer cb, VkQueryPool qp, uint32_t iters) {
    VKA_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKA_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0u, 2u);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1u;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, qp, 0u);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = g->iview;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = (VkRect2D){{0, 0}, {GFX_DW, GFX_DH}};
    ri.layerCount = 1u;
    ri.colorAttachmentCount = 1u;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g->pipe);
    vkCmdPushConstants(cb, g->pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0u, 4u, &iters);
    vkCmdDraw(cb, 3u, 1u, 0u, 0u);
    vkCmdEndRendering(cb);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, qp, 1u);
    VKA_CHECK(vkEndCommandBuffer(cb));
}

// ---- Background graphics serve thread (reuses src/round12/gfx_load.c:333-397) ----
typedef struct {
    const GfxRes * gfx;
    uint32_t iters;
    int in_flight;
    double pace_ms;
    int warmup_frames;
    int max_frames;
    atomic_int stop_req;
    atomic_int thermal_status;
    atomic_int thermal_hot_mc;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int ready;
    VkaGfxFrame * frames;
    int frame_cap;
    int frame_count;
} GfxWorker;

static void * gfx_worker_thread(void * arg) {
    GfxWorker * w = (GfxWorker *)arg;
    const GfxRes * g = w->gfx;
    const int warm_need = (w->warmup_frames > 0) ? w->warmup_frames : GFX_WARMUP_FRAMES;
    for (int i = 0; i < GFX_RING; ++i) {
        record_gfx_frame(g, g->cbs[i], g->qps[i], w->iters);
    }
    double tsub[GFX_RING] = {0};
    int submitted = 0, completed = 0;
    double next_sub_ms = mono_ms();

    for (;;) {
        int stop_now = atomic_load(&w->stop_req) || (submitted >= w->max_frames);
        double hot = -1.0;
        const int hs = vkt_read_hotspot_status(&hot);
        if (hs != VKA_HOT_OK) {
            fprintf(stderr, "THERMAL STOP (fail-closed) in gfx_worker_thread: status=%s hot=%.1f C (limit=%.0f C)\n",
                    hs == VKA_HOT_SENSOR_ERR ? "SENSOR_ERR" : "OVER_LIMIT", hot, VKA_THERMAL_STOP_C);
            fflush(stderr);
            const int prev = atomic_load(&w->thermal_status);
            atomic_store(&w->thermal_status, vka_thermal_merge_status(prev, hs));
            atomic_store(&w->thermal_hot_mc, hot > 0.0 ? (int)llround(hot * 1000.0) : -1);
            stop_now = 1;
        }
        if (!stop_now && (submitted - completed) < w->in_flight) {
            if (w->pace_ms > 0.0 && submitted >= warm_need) {
                double now = mono_ms();
                if (now < next_sub_ms) {
                    usleep((useconds_t)((next_sub_ms - now) * 1000.0));
                }
                next_sub_ms = mono_ms() + w->pace_ms;
            }
            const int slot = submitted % GFX_RING;
            VKA_CHECK(vkResetFences(VT.dev, 1u, &g->fns[slot]));
            tsub[slot] = mono_ms();
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1u;
            si.pCommandBuffers = &g->cbs[slot];
            VKA_CHECK(submit_queue_safe(VT.gfx_q, &si, g->fns[slot]));
            ++submitted;
        }
        if (completed < submitted && ((submitted - completed) >= w->in_flight || stop_now)) {
            const int slot = completed % GFX_RING;
            VKA_CHECK(vkWaitForFences(VT.dev, 1u, &g->fns[slot], VK_TRUE, UINT64_MAX));
            const double td = mono_ms();
            uint64_t qv[2] = {0, 0};
            VKA_CHECK(vkGetQueryPoolResults(VT.dev, g->qps[slot], 0u, 2u, sizeof qv, qv, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double gpu_ms = (double)(qv[1] - qv[0]) * VT.ts_period_ns * 1e-6;
            double fhot = -1.0;
            const int fhs = vkt_read_hotspot_status(&fhot);
            if (fhs != VKA_HOT_OK) {
                const int prev = atomic_load(&w->thermal_status);
                if (prev == VKA_HOT_OK) {
                    fprintf(stderr, "THERMAL STOP (fail-closed) in gfx_worker_thread post-frame: status=%s hot=%.1f C (limit=%.0f C)\n",
                            fhs == VKA_HOT_SENSOR_ERR ? "SENSOR_ERR" : "OVER_LIMIT", fhot, VKA_THERMAL_STOP_C);
                    fflush(stderr);
                }
                atomic_store(&w->thermal_status, vka_thermal_merge_status(prev, fhs));
                atomic_store(&w->thermal_hot_mc, fhot > 0.0 ? (int)llround(fhot * 1000.0) : -1);
                stop_now = 1;
            }

            pthread_mutex_lock(&w->mu);
            if (w->frame_count < w->frame_cap) {
                w->frames[w->frame_count++] = (VkaGfxFrame){tsub[slot], td, gpu_ms, td - tsub[slot], fhot};
            }
            ++completed;
            if (!w->ready && completed >= warm_need) {
                w->ready = 1;
                pthread_cond_broadcast(&w->cv);
            }
            pthread_mutex_unlock(&w->mu);
        }
        if (stop_now && completed >= submitted) break;
    }

    pthread_mutex_lock(&w->mu);
    if (!w->ready) {
        w->ready = 1;
        pthread_cond_broadcast(&w->cv);
    }
    pthread_mutex_unlock(&w->mu);
    return NULL;
}

// ---- Compute setup & descriptor allocation (no vkDeviceWaitIdle during overlap!) ----
static VkDescriptorSet alloc_and_write_ds(const VktImg * im, const VktBuf * out_buf, const VktBuf * in_buf) {
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = VT.dp;
    dai.descriptorSetCount = 1u;
    dai.pSetLayouts = &VT.dsl;
    VKA_CHECK(vkAllocateDescriptorSets(VT.dev, &dai, &ds));

    VkDescriptorImageInfo di = {im->sampler, im->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo db1 = {(out_buf ? out_buf : &VT.dummy_buf)->buf, 0u, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo db2 = {(in_buf ? in_buf : &VT.dummy_buf)->buf, 0u, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[3];
    memset(w, 0, sizeof w);
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = ds;
    w[0].dstBinding = 0u;
    w[0].descriptorCount = 1u;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &di;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = ds;
    w[1].dstBinding = 1u;
    w[1].descriptorCount = 1u;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &db1;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = ds;
    w[2].dstBinding = 2u;
    w[2].descriptorCount = 1u;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &db2;
    vkUpdateDescriptorSets(VT.dev, 3u, w, 0u, NULL);
    return ds;
}

static void comp_init_res(CompRes * c, int init_data, int break_fp16) {
    memset(c, 0, sizeof *c);
    c->p_fp16 = vkt_pipe_create("vk_alu_fp16_n64.spv", "vk_async_fp16_w32", 256u, 32u);
    c->p_sad = vkt_pipe_create("vk_async_sad.spv", "vk_async_sad_w32", 256u, 32u);
    c->p_tex = vkt_pipe_create("vk_tex_lod.spv", "vk_async_tex_w32", 256u, 32u);
    c->p_stream = vkt_pipe_create("vk_async_stream.spv", "vk_async_stream_w32", 256u, 32u);
    c->p_stream_init = vkt_pipe_create("vk_async_stream_init.spv", "vk_async_stream_init_w32", 256u, 32u);

    if (!init_data) return;

    uint32_t tw = 0, th = 0;
    vkt_level_dims(VKT_FMT_RGBA16_SFLOAT, VKT_LVL_MALL, &tw, &th);  // 2048x2048 = 32 MiB
    c->tex_img = vkt_img_create(VK_FORMAT_R16G16B16A16_SFLOAT, tw, th, VK_FILTER_LINEAR);
    vkt_img_fill_ref(&c->tex_img, VKT_FMT_RGBA16_SFLOAT, 1);

    c->fp16_out = vkt_buf_create(CHECK_WGS * 256u * 2u * 4u * sizeof(float), 1);
    c->fp16_kbuf = vkt_buf_create(256u * 4u * sizeof(float), 1);
    float * kptr = (float *)c->fp16_kbuf.map;
    for (uint32_t lid = 0; lid < 256u; ++lid) {
        float ak_lo = 0.0f, ak_hi = 0.0f;
        vkt_osc_lane_k(lid, &ak_lo, &ak_hi);
        kptr[lid * 4u + 0u] = -1.0f;
        kptr[lid * 4u + 1u] = -1.0f;
        kptr[lid * 4u + 2u] = ak_lo + ((break_fp16 && lid == 0u) ? 0.25f : 0.0f);
        kptr[lid * 4u + 3u] = ak_hi;
    }

    c->sad_out = vkt_buf_create(CHECK_WGS * 256u * sizeof(uint32_t), 1);
    c->sad_in = vkt_buf_create(256u * 4u * sizeof(uint32_t), 1);
    uint32_t * sptr = (uint32_t *)c->sad_in.map;
    for (uint32_t lid = 0; lid < 256u; ++lid) {
        vka_sad_lane_input(lid, sptr + lid * 4u);
    }

    c->tex_out = vkt_buf_create(CHECK_WGS * 256u * 4u * sizeof(float), 1);

    const VkDeviceSize stream_bytes = (VkDeviceSize)VKA_STREAM_TOTAL_VEC4 * 16ull;
    c->stream_src = vkt_buf_create(stream_bytes, 0);
    c->stream_dst = vkt_buf_create(stream_bytes, 0);
    c->stream_chk = vkt_buf_create((VkDeviceSize)STREAM_CHK_TOTAL_VEC4 * 16ull, 1);

    c->ds_fp16 = alloc_and_write_ds(&c->tex_img, &c->fp16_out, &c->fp16_kbuf);
    c->ds_sad = alloc_and_write_ds(&c->tex_img, &c->sad_out, &c->sad_in);
    c->ds_tex = alloc_and_write_ds(&c->tex_img, &c->tex_out, NULL);
    c->ds_stream = alloc_and_write_ds(&c->tex_img, &c->stream_dst, &c->stream_src);
    c->ds_stream_init = alloc_and_write_ds(&c->tex_img, &c->stream_src, NULL);

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = VT.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1u;
    VKA_CHECK(vkAllocateCommandBuffers(VT.dev, &cai, &c->copy_cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKA_CHECK(vkCreateFence(VT.dev, &fci, NULL, &c->copy_fence));

    // Initialize 128 MiB stream_src on GPU with deterministic hash.
    thermal_guard_or_abort("stream_init");
    VKA_CHECK(vkResetCommandBuffer(c->copy_cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKA_CHECK(vkBeginCommandBuffer(c->copy_cb, &bi));
    vkCmdBindPipeline(c->copy_cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->p_stream_init.pipe);
    vkCmdBindDescriptorSets(c->copy_cb, VK_PIPELINE_BIND_POINT_COMPUTE, VT.pl, 0u, 1u, &c->ds_stream_init, 0u, NULL);
    StreamInitPC ipc = {VKA_STREAM_TOTAL_VEC4};
    vkCmdPushConstants(c->copy_cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof ipc, &ipc);
    vkCmdDispatch(c->copy_cb, VKA_STREAM_GROUPS, 1u, 1u);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1u;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(c->copy_cb, &di);
    VKA_CHECK(vkEndCommandBuffer(c->copy_cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1u;
    si.pCommandBuffers = &c->copy_cb;
    VKA_CHECK(submit_queue_safe(VT.q, &si, c->copy_fence));
    VKA_CHECK(vkWaitForFences(VT.dev, 1u, &c->copy_fence, VK_TRUE, UINT64_MAX));
    VKA_CHECK(vkResetFences(VT.dev, 1u, &c->copy_fence));
}

static void comp_destroy_res(CompRes * c) {
    if (c->copy_fence) vkDestroyFence(VT.dev, c->copy_fence, NULL);
    vkt_buf_destroy(&c->stream_chk);
    vkt_buf_destroy(&c->stream_dst);
    vkt_buf_destroy(&c->stream_src);
    vkt_buf_destroy(&c->tex_out);
    vkt_buf_destroy(&c->sad_in);
    vkt_buf_destroy(&c->sad_out);
    vkt_buf_destroy(&c->fp16_kbuf);
    vkt_buf_destroy(&c->fp16_out);
    vkt_img_destroy(&c->tex_img);
    vkt_pipe_destroy(&c->p_stream_init);
    vkt_pipe_destroy(&c->p_stream);
    vkt_pipe_destroy(&c->p_tex);
    vkt_pipe_destroy(&c->p_sad);
    vkt_pipe_destroy(&c->p_fp16);
    memset(c, 0, sizeof *c);
}

static TexPC make_tex_pc(uint32_t w, uint32_t h, uint32_t iters, uint32_t break_dx, uint32_t strip_taps) {
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (strip_taps < tiles_x) ? strip_taps : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    TexPC pc = {0};
    pc.w = w;
    pc.h = h;
    pc.mask_x = w - 1u;
    pc.mask_y = h - 1u;
    pc.strips_x_mask = strips_x - 1u;
    pc.strips_x_shift = ctz_u32(strips_x);
    pc.tiles_y_mask = tiles_y - 1u;
    pc.y_tile_step = 1u;
    pc.iters = iters;
    pc.break_dx = break_dx;
    pc.out_stride_wg = CHECK_WGS;
    pc.strip_tiles = strip;
    pc.inv_w = 1.0f / (float)w;
    pc.inv_h = 1.0f / (float)h;
    pc.du = 16.0f / (float)w;
    pc.dv = 16.0f / (float)h;
    return pc;
}

// Record a compute chain of `chain_n` dispatches (1 or 80) with compute-to-compute barriers
// between dispatches and start/end compute-queue timestamps.
static void record_comp_chain(const CompRes * c, int job, uint32_t chain_n, uint32_t iters, int do_break) {
    VKA_CHECK(vkResetCommandBuffer(VT.cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKA_CHECK(vkBeginCommandBuffer(VT.cb, &bi));
    vkCmdResetQueryPool(VT.cb, VT.qp, 0u, 2u);

    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    uint32_t groups = 1024u;
    if (job == JOB_FP16) {
        pipe = c->p_fp16.pipe; ds = c->ds_fp16; groups = VKA_FP16_GROUPS;
    } else if (job == JOB_SAD) {
        pipe = c->p_sad.pipe; ds = c->ds_sad; groups = VKA_SAD_GROUPS;
    } else if (job == JOB_TEX) {
        pipe = c->p_tex.pipe; ds = c->ds_tex; groups = VKA_TEX_GROUPS;
    } else {
        pipe = c->p_stream.pipe; ds = c->ds_stream; groups = VKA_STREAM_GROUPS;
    }

    vkCmdBindPipeline(VT.cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(VT.cb, VK_PIPELINE_BIND_POINT_COMPUTE, VT.pl, 0u, 1u, &ds, 0u, NULL);
    vkCmdWriteTimestamp2(VT.cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VT.qp, 0u);

    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1u;
    di.pMemoryBarriers = &mb;

    for (uint32_t p = 0u; p < chain_n; ++p) {
        if (job == JOB_FP16) {
            TexPC pc = make_tex_pc(c->tex_img.w, c->tex_img.h, iters, 0u, 4u);
            vkCmdPushConstants(VT.cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof pc, &pc);
        } else if (job == JOB_SAD) {
            SadPC pc = {iters, p, CHECK_WGS, do_break ? 1u : 0u};
            vkCmdPushConstants(VT.cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof pc, &pc);
        } else if (job == JOB_TEX) {
            TexPC pc = make_tex_pc(c->tex_img.w, c->tex_img.h, iters, do_break ? 1u : 0u, 16u);
            vkCmdPushConstants(VT.cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof pc, &pc);
        } else {
            StreamPC pc;
            if (chain_n <= 1u) {
                pc.base_vec4 = 0u;
                pc.chunks = 4u;
                pc.iters = iters;
                pc.salt = VKA_STREAM_BASE_SALT;
            } else {
                pc.base_vec4 = (p & 3u) * VKA_STREAM_QUARTER_VEC4;
                pc.chunks = 1u;
                pc.iters = iters;
                pc.salt = VKA_STREAM_BASE_SALT ^ p;
            }
            pc.break_mask = do_break ? 1u : 0u;
            vkCmdPushConstants(VT.cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof pc, &pc);
        }
        vkCmdDispatch(VT.cb, groups, 1u, 1u);
        if (p + 1u < chain_n) {
            vkCmdPipelineBarrier2(VT.cb, &di);
        }
    }

    vkCmdWriteTimestamp2(VT.cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VT.qp, 1u);
    VKA_CHECK(vkEndCommandBuffer(VT.cb));
}

static VkaCompTrial run_comp_chain_once(int * out_hot_status, double * out_hot_c) {
    double hot = -1.0;
    int hs = vkt_read_hotspot_status(&hot);
    if (hs != VKA_HOT_OK) {
        if (out_hot_status) *out_hot_status = hs;
        if (out_hot_c) *out_hot_c = hot;
        return (VkaCompTrial){0.0, 0.0, 0.0, 0.0};
    }
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1u;
    si.pCommandBuffers = &VT.cb;
    const double t0 = mono_ms();
    VKA_CHECK(submit_queue_safe(VT.q, &si, VT.fence));
    VKA_CHECK(vkWaitForFences(VT.dev, 1u, &VT.fence, VK_TRUE, UINT64_MAX));
    const double t1 = mono_ms();
    VKA_CHECK(vkResetFences(VT.dev, 1u, &VT.fence));
    uint64_t ts[2] = {0, 0};
    VKA_CHECK(vkGetQueryPoolResults(VT.dev, VT.qp, 0u, 2u, sizeof ts, ts, sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const double gpu_ms = (double)(ts[1] - ts[0]) * VT.ts_period_ns * 1e-6;
    double hot2 = -1.0;
    const int hs2 = vkt_read_hotspot_status(&hot2);
    if (out_hot_status) *out_hot_status = hs2;
    if (out_hot_c) *out_hot_c = hot2;
    return (VkaCompTrial){t0, t1, gpu_ms, t1 - t0};
}

static int chain_slot(uint32_t chain_n) {
    return (chain_n == 1u) ? 0 : 1;
}

static uint32_t job_iters(int job, uint32_t chain_n) {
    return vka_job_iters(job, chain_n);
}

static int verify_comp_job(const CompRes * c, int job, uint32_t chain_n, uint32_t iters) {
    int bad = 0;
    if (job == JOB_FP16) {
        const float * got_all = (const float *)c->fp16_out.map;
        for (uint32_t wg = 0; wg < CHECK_WGS; ++wg) {
            for (uint32_t lid = 0; lid < 256u; ++lid) {
                const float * got_alu = got_all + (wg * 256u + lid) * 8u + 4u;
                if (!vkt_verify_mix_alu_lane(lid, 64u, iters, 1, got_alu[0], got_alu[1], 0, NULL)) {
                    ++bad;
                }
            }
        }
        return bad;
    }
    if (job == JOB_SAD) {
        const uint32_t * got = (const uint32_t *)c->sad_out.map;
        // Verify all 256 lanes of WG 0 plus 16 strided lanes of WGs 1..3 (304 threads total).
        for (uint32_t wg = 0; wg < CHECK_WGS; ++wg) {
            const uint32_t step = (wg == 0u) ? 1u : 16u;
            for (uint32_t lid = 0; lid < 256u; lid += step) {
                const uint32_t want = vka_expected_sad_chain(wg, lid, iters, chain_n, 0u);
                if (got[wg * 256u + lid] != want) ++bad;
            }
        }
        return bad;
    }
    if (job == JOB_TEX) {
        const float * got_all = (const float *)c->tex_out.map;
        for (uint32_t wg = 0; wg < CHECK_WGS; ++wg) {
            for (uint32_t lid = 0; lid < 256u; ++lid) {
                float want[4];
                vkt_expected_thread(VKT_FMT_RGBA16_SFLOAT, VKT_OP_LOD, 1, c->tex_img.w, c->tex_img.h,
                                    wg, lid, iters, 1u, 0u, want);
                const float * got = got_all + (wg * 256u + lid) * 4u;
                for (int k = 0; k < 4; ++k) {
                    if (got[k] != want[k]) ++bad;
                }
            }
        }
        return bad;
    }
    // JOB_STREAM: copy 8 slices (start & end of all 4 x 32 MiB quarters) to host-visible check buffer.
    VkBufferCopy regs[STREAM_CHK_BLOCKS];
    uint32_t starts[STREAM_CHK_BLOCKS];
    for (uint32_t q = 0; q < 4u; ++q) {
        const uint32_t qbase = q * VKA_STREAM_QUARTER_VEC4;
        starts[q * 2u + 0u] = qbase;
        starts[q * 2u + 1u] = qbase + VKA_STREAM_QUARTER_VEC4 - STREAM_CHK_BLOCK_VEC4;
    }
    for (uint32_t b = 0; b < STREAM_CHK_BLOCKS; ++b) {
        regs[b].srcOffset = (VkDeviceSize)starts[b] * 16ull;
        regs[b].dstOffset = (VkDeviceSize)(b * STREAM_CHK_BLOCK_VEC4) * 16ull;
        regs[b].size = (VkDeviceSize)STREAM_CHK_BLOCK_VEC4 * 16ull;
    }
    VKA_CHECK(vkResetCommandBuffer(c->copy_cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKA_CHECK(vkBeginCommandBuffer(c->copy_cb, &bi));
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1u;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(c->copy_cb, &di);
    vkCmdCopyBuffer(c->copy_cb, c->stream_dst.buf, c->stream_chk.buf, STREAM_CHK_BLOCKS, regs);
    VKA_CHECK(vkEndCommandBuffer(c->copy_cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1u;
    si.pCommandBuffers = &c->copy_cb;
    VKA_CHECK(submit_queue_safe(VT.q, &si, c->copy_fence));
    VKA_CHECK(vkWaitForFences(VT.dev, 1u, &c->copy_fence, VK_TRUE, UINT64_MAX));
    VKA_CHECK(vkResetFences(VT.dev, 1u, &c->copy_fence));

    const uint32_t * got = (const uint32_t *)c->stream_chk.map;
    for (uint32_t b = 0; b < STREAM_CHK_BLOCKS; ++b) {
        for (uint32_t k = 0; k < STREAM_CHK_BLOCK_VEC4; ++k) {
            uint32_t want[4];
            vka_stream_expected_vec4(starts[b] + k, chain_n, iters, VKA_STREAM_BASE_SALT, 0u, want);
            const uint32_t * g = got + (b * STREAM_CHK_BLOCK_VEC4 + k) * 4u;
            if (g[0] != want[0] || g[1] != want[1] || g[2] != want[2] || g[3] != want[3]) ++bad;
        }
    }
    return bad;
}

static int check_all_aco(const GfxRes * g, const CompRes * c) {
    int fails = 0;
    VktLoopStats sf = vkt_pipe_analyze_loop(&c->p_fp16, "v_pk_fm");
    const int ok_fp16 = (sf.pk_fma_f16 >= 256);
    if (!ok_fp16) ++fails;
    printf("ACO %-24s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d pk_fma_f16=%3d check=%s\n",
           c->p_fp16.name, c->p_fp16.sg_size, c->p_fp16.vgprs, c->p_fp16.sg_per_simd,
           sf.total_insts, sf.pk_fma_f16, ok_fp16 ? "PASS" : "FAIL");

    VktLoopStats ss = vkt_pipe_analyze_loop(&c->p_sad, "v_");
    const int ok_sad = (ss.int_alu >= 64 && ss.fma_f32 == 0 && ss.pk_fma_f16 == 0 && ss.v_msad == 0);
    if (!ok_sad) ++fails;
    printf("ACO %-24s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d int_alu=%3d v_msad=%d (no msad in GLSL) check=%s\n",
           c->p_sad.name, c->p_sad.sg_size, c->p_sad.vgprs, c->p_sad.sg_per_simd,
           ss.total_insts, ss.int_alu, ss.v_msad, ok_sad ? "PASS" : "FAIL");

    VktLoopStats st = vkt_pipe_analyze_loop(&c->p_tex, "image_");
    const int ok_tex = (st.image_sample >= 16);
    if (!ok_tex) ++fails;
    printf("ACO %-24s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d img_samp=%3d check=%s\n",
           c->p_tex.name, c->p_tex.sg_size, c->p_tex.vgprs, c->p_tex.sg_per_simd,
           st.total_insts, st.image_sample, ok_tex ? "PASS" : "FAIL");

    VktLoopStats sm = vkt_pipe_analyze_loop(&c->p_stream, "_b128");
    const int ok_stream = (sm.buf_load_b128 >= 4 && sm.buf_store_b128 >= 4);
    if (!ok_stream) ++fails;
    printf("ACO %-24s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d ld_b128=%2d st_b128=%2d check=%s\n",
           c->p_stream.name, c->p_stream.sg_size, c->p_stream.vgprs, c->p_stream.sg_per_simd,
           sm.total_insts, sm.buf_load_b128, sm.buf_store_b128, ok_stream ? "PASS" : "FAIL");

    VktLoopStats sg = vkt_pipe_analyze_loop(&g->aco_info, "fma");
    const int gfma = sg.fma_f32 + 2 * sg.dual_fmac_f32;
    const int ok_gfx = (gfma >= 2);
    if (!ok_gfx) ++fails;
    printf("ACO %-24s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d fma_f32=%2d dual_fmac=%2d check=%s\n",
           g->aco_info.name, g->aco_info.sg_size, g->aco_info.vgprs, g->aco_info.sg_per_simd,
           sg.total_insts, sg.fma_f32, sg.dual_fmac_f32, ok_gfx ? "PASS" : "FAIL");
    fflush(stdout);
    return fails;
}

static int wait_cooldown_logged(const char * label, double target_c, double settle_c,
                                int max_wait_ms, int settle_max_ms, VkaCooldownResult * out) {
    VkaCooldownResult res;
    const int st = vka_wait_cooldown_file(VT.hwmon_hot_path, VKA_THERMAL_STOP_C,
                                          target_c, settle_c,
                                          max_wait_ms, settle_max_ms,
                                          VKA_COOLDOWN_POLL_MS, &res);
    const int fan_rpm = vkt_read_fan_rpm();
    const char * sstr = (st == VKA_COOL_OK) ? "OK" :
                        (st == VKA_COOL_TIMEOUT ? "TIMEOUT" : "SENSOR_ERR");
    printf("COOLDOWN label=%s status=%s start_c=%.1f end_c=%.1f waited_ms=%d polls=%d fan_rpm=%d target_c=%.1f\n",
           label, sstr, res.start_hot_c, res.end_hot_c, res.waited_ms, res.polls, fan_rpm, target_c);
    fflush(stdout);
    if (out) *out = res;
    if (st == VKA_COOL_SENSOR_ERR) {
        fprintf(stderr, "THERMAL ABORT (fail-closed) in cooldown(%s): unreadable sensor (%s)\n",
                label, VT.hwmon_hot_path[0] ? VT.hwmon_hot_path : "(missing)");
        fflush(stderr);
        exit(4);
    }
    return st;
}

// Bounded moderate-load fan pre-warm (PACE-60 duty, never writes any fan/power/clock setting).
// Stops early as soon as fan_rpm >= FAN_PREWARM_TARGET_RPM or hotspot >= 78 C or max_ms elapses.
static int run_fan_prewarm(VkaThermalState * therm, int use_gp, VkQueueGlobalPriority prio, int smoke) {
    VKA_CHECK(vkt_init_async(1, use_gp, prio, 0));
    VT.out_dir[0] = '\0';

    double hot0 = -1.0;
    const int hs0 = vkt_read_hotspot_status(&hot0);
    const int fan0 = vkt_read_fan_rpm();
    if (hs0 == VKA_HOT_SENSOR_ERR) {
        vkt_shutdown();
        fprintf(stderr, "THERMAL ABORT (fail-closed) in fan_prewarm_start: unreadable sensor\n");
        fflush(stderr);
        exit(4);
    }
    if (hs0 == VKA_HOT_OVER_LIMIT) {
        const int act = vka_thermal_on_event(therm, hs0);
        printf("FAN_PREWARM status=THERMAL_STOP limit_c=%.0f frames=0 elapsed_ms=0 hot_start=%.1f hot_max=%.1f hot_end=%.1f fan_start=%d fan_end=%d\n",
               VKA_THERMAL_STOP_C, hot0, hot0, hot0, fan0, fan0);
        fflush(stdout);
        vkt_shutdown();
        return act;
    }
    if (fan0 >= FAN_PREWARM_TARGET_RPM) {
        printf("FAN_PREWARM status=SKIP_ALREADY_SPINNING frames=0 elapsed_ms=0 hot_start=%.1f hot_max=%.1f hot_end=%.1f fan_start=%d fan_end=%d\n",
               hot0, hot0, hot0, fan0, fan0);
        fflush(stdout);
        vkt_shutdown();
        return VKA_THERM_NONE;
    }
    if (hot0 >= FAN_PREWARM_SOFT_CAP_C) {
        printf("FAN_PREWARM status=SKIP_ALREADY_WARM frames=0 elapsed_ms=0 hot_start=%.1f hot_max=%.1f hot_end=%.1f fan_start=%d fan_end=%d\n",
               hot0, hot0, hot0, fan0, fan0);
        fflush(stdout);
        vkt_shutdown();
        return VKA_THERM_NONE;
    }

    GfxRes g;
    gfx_init_res(&g, 0);
    record_gfx_frame(&g, g.cbs[0], g.qps[0], GFX_PACE_ITERS);

    const double max_ms = smoke ? FAN_PREWARM_SMOKE_MS : FAN_PREWARM_MAX_MS;
    const int max_frames = smoke ? 120 : 240;
    const double t_start = mono_ms();
    double next_sub = t_start;
    double hot_max = hot0;
    double hot_end = hot0;
    int fan_end = fan0;
    int frames = 0;
    const char * status = "BOUND_REACHED";
    int therm_action = VKA_THERM_NONE;

    for (int i = 0; i < max_frames; ++i) {
        const double now = mono_ms();
        if (now - t_start >= max_ms) {
            status = "TIME_BOUND";
            break;
        }
        double hot = -1.0;
        const int hs = vkt_read_hotspot_status(&hot);
        if (hs != VKA_HOT_OK) {
            hot_end = hot;
            if (hot > hot_max) hot_max = hot;
            therm_action = vka_thermal_on_event(therm, hs);
            status = (hs == VKA_HOT_SENSOR_ERR) ? "SENSOR_ERR" : "THERMAL_STOP";
            break;
        }
        if (hot >= FAN_PREWARM_SOFT_CAP_C) {
            hot_end = hot;
            if (hot > hot_max) hot_max = hot;
            status = "SOFT_CAP_78C";
            break;
        }
        if (i > 0 && now < next_sub) {
            usleep((useconds_t)((next_sub - now) * 1000.0));
        }
        next_sub = mono_ms() + GFX_PACE_MS;

        VKA_CHECK(vkResetFences(VT.dev, 1u, &g.fns[0]));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1u;
        si.pCommandBuffers = &g.cbs[0];
        VKA_CHECK(submit_queue_safe(VT.gfx_q, &si, g.fns[0]));
        VKA_CHECK(vkWaitForFences(VT.dev, 1u, &g.fns[0], VK_TRUE, UINT64_MAX));
        ++frames;

        const int hs2 = vkt_read_hotspot_status(&hot);
        fan_end = vkt_read_fan_rpm();
        hot_end = hot;
        if (hot > hot_max) hot_max = hot;
        if (hs2 != VKA_HOT_OK) {
            therm_action = vka_thermal_on_event(therm, hs2);
            status = (hs2 == VKA_HOT_SENSOR_ERR) ? "SENSOR_ERR" : "THERMAL_STOP";
            break;
        }
        if (fan_end >= FAN_PREWARM_TARGET_RPM) {
            status = "FAN_SPINNING";
            break;
        }
    }

    const double elapsed = mono_ms() - t_start;
    printf("FAN_PREWARM status=%s limit_c=%.0f frames=%d elapsed_ms=%.0f hot_start=%.1f hot_max=%.1f hot_end=%.1f fan_start=%d fan_end=%d\n",
           status, VKA_THERMAL_STOP_C, frames, elapsed, hot0, hot_max, hot_end, fan0, fan_end);
    fflush(stdout);
    gfx_destroy_res(&g);
    vkt_shutdown();
    return therm_action;
}

static int run_gfx_alone_baseline(const GfxRes * g, int reg, int n_frames, double * out_hot_c) {
    const uint32_t iters = (reg == REG_SAT) ? GFX_SAT_ITERS : GFX_PACE_ITERS;
    const int in_flight = (reg == REG_SAT) ? 2 : 1;
    const double pace_ms = (reg == REG_SAT) ? 0.0 : GFX_PACE_MS;

    GfxWorker gw;
    memset(&gw, 0, sizeof gw);
    gw.gfx = g;
    gw.iters = iters;
    gw.in_flight = in_flight;
    gw.pace_ms = pace_ms;
    gw.warmup_frames = GFX_WARMUP_FRAMES;
    gw.max_frames = n_frames + GFX_WARMUP_FRAMES;
    gw.frame_cap = n_frames + 16;
    atomic_init(&gw.stop_req, 0);
    atomic_init(&gw.thermal_status, VKA_HOT_OK);
    atomic_init(&gw.thermal_hot_mc, -1);
    gw.frames = (VkaGfxFrame *)calloc((size_t)gw.frame_cap, sizeof(VkaGfxFrame));
    pthread_mutex_init(&gw.mu, NULL);
    pthread_cond_init(&gw.cv, NULL);

    vkt_metrics_start();
    pthread_t thr;
    pthread_create(&thr, NULL, gfx_worker_thread, &gw);
    pthread_join(thr, NULL);
    VktMetrics mt = vkt_metrics_stop();

    const int hs = atomic_load(&gw.thermal_status);
    const int hmc = atomic_load(&gw.thermal_hot_mc);
    if (out_hot_c) *out_hot_c = (hmc > 0) ? ((double)hmc * 1e-3) : mt.hot_max_c;
    if (hs != VKA_HOT_OK) {
        pthread_mutex_destroy(&gw.mu);
        pthread_cond_destroy(&gw.cv);
        free(gw.frames);
        return hs;
    }

    const int skip = (gw.frame_count > 6) ? GFX_WARMUP_FRAMES : 0;
    const int nf = gw.frame_count - skip;
    double * dts = (double *)calloc((size_t)(nf > 1 ? nf - 1 : 1), sizeof(double));
    double * gpus = (double *)calloc((size_t)(nf > 0 ? nf : 1), sizeof(double));
    double * walls = (double *)calloc((size_t)(nf > 0 ? nf : 1), sizeof(double));
    double dt_sum = 0.0;
    int ndt = 0;
    for (int i = 0; i < nf; ++i) {
        gpus[i] = gw.frames[skip + i].gpu_ms;
        walls[i] = gw.frames[skip + i].wall_ms;
        if (i > 0) {
            double dt = gw.frames[skip + i].t_done_ms - gw.frames[skip + i - 1].t_done_ms;
            dts[ndt++] = dt;
            dt_sum += dt;
        }
    }
    printf("BASELINE_GFX regime=%-7s frames=%d dt_avg=%.3f dt_p50=%.3f dt_p95=%.3f gpu_p50=%.3f wall_p50=%.3f sclk=%.0fMHz uclk=%.0fMHz hot=%.0fC\n",
           k_reg_names[reg], nf,
           ndt > 0 ? dt_sum / (double)ndt : NAN,
           vka_percentile(dts, ndt, 0.50),
           vka_percentile(dts, ndt, 0.95),
           vka_percentile(gpus, nf, 0.50),
           vka_percentile(walls, nf, 0.50),
           mt.sclk_med_mhz, mt.uclk_med_mhz, mt.hot_max_c);
    fflush(stdout);
    free(dts);
    free(gpus);
    free(walls);
    pthread_mutex_destroy(&gw.mu);
    pthread_cond_destroy(&gw.cv);
    free(gw.frames);
    return VKA_HOT_OK;
}

int main(int argc, char ** argv) {
    int dump_asm = 0, do_break = 0, smoke = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-asm") == 0) dump_asm = 1;
        else if (strcmp(argv[i], "--break") == 0) do_break = 1;
        else if (strcmp(argv[i], "--smoke") == 0) smoke = 1;
        else {
            fprintf(stderr, "usage: vk_async [--dump-asm] [--break] [--smoke]\n");
            return 2;
        }
    }

    // 1. Report device, all queue families, and probe VK_KHR_global_priority without root.
    int allowed_idx[4] = {0};
    int n_allowed = 0;
    {
        VkResult r0 = vkt_init_async(0, 0, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM, 1);
        if (r0 != VK_SUCCESS) {
            printf("VERDICT vk_async: FAIL device init rc=%d\n", (int)r0);
            return 1;
        }
        const int has_gp = VT.has_global_prio;
        vkt_shutdown();

        for (int i = 0; i < 4; ++i) {
            if (!has_gp) {
                printf("PRIO_PROBE name=%-8s val=%4d status=EXT_UNAVAILABLE vk_result=%d\n",
                       k_prio_names[i], (int)k_all_prios[i], (int)VK_ERROR_EXTENSION_NOT_PRESENT);
                continue;
            }
            VkResult rp = vkt_init_async(1, 1, k_all_prios[i], 0);
            if (rp == VK_SUCCESS) {
                printf("PRIO_PROBE name=%-8s val=%4d status=ALLOWED vk_result=0\n",
                       k_prio_names[i], (int)k_all_prios[i]);
                allowed_idx[n_allowed++] = i;
                vkt_shutdown();
            } else if (rp == VK_ERROR_NOT_PERMITTED) {
                printf("PRIO_PROBE name=%-8s val=%4d status=REFUSED_NOT_PERMITTED vk_result=%d\n",
                       k_prio_names[i], (int)k_all_prios[i], (int)rp);
            } else {
                printf("PRIO_PROBE name=%-8s val=%4d status=REFUSED_OTHER vk_result=%d\n",
                       k_prio_names[i], (int)k_all_prios[i], (int)rp);
            }
        }
        fflush(stdout);
    }

    // Print preregistered N=1 vs N=80 total work per job so every log proves equal work.
    for (int j = 0; j < JOB_COUNT; ++j) {
        const uint64_t w1 = vka_job_total_work(j, 1u);
        const uint64_t w80 = vka_job_total_work(j, VKA_CHAIN_SAT);
        if (j == JOB_STREAM) {
            printf("WORK_TOTAL job=%-6s n1_iters=%3u n80_iters=%2u total_n1=%llu total_n80=%llu "
                   "unit=%s (%.0f MiB = %.1f GiB) equal=%s\n",
                   k_job_names[j], job_iters(j, 1u), job_iters(j, VKA_CHAIN_SAT),
                   (unsigned long long)w1, (unsigned long long)w80, vka_job_work_unit(j),
                   (double)w1 / (1024.0 * 1024.0), (double)w1 / (1024.0 * 1024.0 * 1024.0),
                   (w1 == w80) ? "YES" : "NO");
        } else {
            printf("WORK_TOTAL job=%-6s n1_iters=%3u n80_iters=%2u total_n1=%llu total_n80=%llu "
                   "unit=%s equal=%s\n",
                   k_job_names[j], job_iters(j, 1u), job_iters(j, VKA_CHAIN_SAT),
                   (unsigned long long)w1, (unsigned long long)w80, vka_job_work_unit(j),
                   (w1 == w80) ? "YES" : "NO");
        }
    }
    fflush(stdout);

    // Pick baseline priority: MEDIUM if allowed, else first allowed, else default (no global prio ext).
    int base_use_gp = 0;
    VkQueueGlobalPriority base_prio = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;
    if (n_allowed > 0) {
        base_use_gp = 1;
        base_prio = k_all_prios[allowed_idx[0]];
        for (int i = 0; i < n_allowed; ++i) {
            if (k_all_prios[allowed_idx[i]] == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM) {
                base_prio = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;
                break;
            }
        }
    }

    // 2. --dump-asm: create pipelines, capture & check ACO assembly, exit without GPU submissions.
    if (dump_asm) {
        VKA_CHECK(vkt_init_async(1, base_use_gp, base_prio, 0));
        GfxRes g;
        CompRes c;
        gfx_init_res(&g, 1);
        comp_init_res(&c, 0, 0);
        const int aco_fail = check_all_aco(&g, &c);
        comp_destroy_res(&c);
        gfx_destroy_res(&g);
        vkt_shutdown();
        printf("VERDICT vk_async_asm: %s (aco_fail=%d)\n", aco_fail == 0 ? "PASS" : "FAIL", aco_fail);
        return aco_fail == 0 ? 0 : 1;
    }

    // 3. --break: deliberate break on ALL 4 compute jobs; every job MUST fail exact CPU check.
    if (do_break) {
        VKA_CHECK(vkt_init_async(1, base_use_gp, base_prio, 0));
        thermal_guard_or_abort("break_start");
        CompRes c;
        comp_init_res(&c, 1, 1 /* break_fp16: perturb GPU kbuf[0].z */);
        int fails = 0;
        for (int j = 0; j < JOB_COUNT; ++j) {
            const uint32_t it = (j == JOB_FP16) ? 11u : (j == JOB_SAD ? 8u : (j == JOB_TEX ? 8u : 1u));
            record_comp_chain(&c, j, 1u, it, 1);
            int chs = VKA_HOT_OK;
            double chot = -1.0;
            (void)run_comp_chain_once(&chs, &chot);
            if (chs != VKA_HOT_OK) {
                fprintf(stderr, "THERMAL ABORT (fail-closed) in break(%s): hot=%.1f C (limit=%.0f C)\n",
                        k_job_names[j], chot, VKA_THERMAL_STOP_C);
                comp_destroy_res(&c);
                vkt_shutdown();
                return 4;
            }
            const int bad = verify_comp_job(&c, j, 1u, it);
            if (bad > 0) ++fails;
            printf("CONTROL vk_async_break job=%-6s bad=%d exact=%s\n",
                   k_job_names[j], bad, bad == 0 ? "ok" : "FAIL");
        }
        fflush(stdout);
        comp_destroy_res(&c);
        vkt_shutdown();
        const int all_caught = (fails == JOB_COUNT);
        printf("VERDICT vk_async: %s (fails=%d/%d, deliberate break)\n",
               all_caught ? "FAIL" : "UNEXPECTED_PASS", fails, JOB_COUNT);
        return all_caught ? 1 : 0;
    }

    // 4. Baseline device: ACO check + graphics alone (PACE-60 then SAT-100) + compute alone (N=1, 80).
    VkaThermalState therm;
    vka_thermal_init(&therm, VKA_THERMAL_MAX_STOPS);
    int total_fails = 0;
    double comp_alone_p50[JOB_COUNT][2];
    memset(comp_alone_p50, 0, sizeof comp_alone_p50);

    {
        VKA_CHECK(vkt_init_async(1, base_use_gp, base_prio, 0));
        VkaCooldownResult cr;
        const int cst = wait_cooldown_logged("baseline_start", VKA_COOLDOWN_TARGET_C, 0.0,
                                             VKA_COOLDOWN_MAX_MS, 0, &cr);
        const int cact = vka_precell_cooldown_action(cst, cr.end_hot_c, VKA_THERMAL_STOP_C);
        if (cact != VKA_PRECELL_RUN) {
            ++therm.cooldown_timeouts;
            if (cact == VKA_PRECELL_ABORT_HOT) ++therm.cooldown_hot_aborts;
            vkt_shutdown();
            printf("VERDICT vk_async: %s (cooldown timeout at baseline_start end_c=%.1f)\n",
                   cact == VKA_PRECELL_ABORT_HOT ? "THERMAL_ABORT" : "FAIL", cr.end_hot_c);
            return cact;
        }
        thermal_guard_or_abort("baseline_start");

        GfxRes g;
        CompRes c;
        gfx_init_res(&g, 1);
        comp_init_res(&c, 1, 0);

        total_fails += check_all_aco(&g, &c);

        // PACE-60 graphics baseline first, then cooldown + SAT-100 baseline (16 frames, 8 smoke).
        const int base_regs[2] = {REG_PACE, REG_SAT};
        for (int bi = 0; bi < 2; ++bi) {
            const int reg = base_regs[bi];
            char clbl[64];
            snprintf(clbl, sizeof clbl, "baseline_gfx_%s", k_reg_names[reg]);
            const int bcs = wait_cooldown_logged(clbl, VKA_COOLDOWN_TARGET_C,
                                                 reg == REG_SAT ? VKA_COOLDOWN_SETTLE_C : 0.0,
                                                 VKA_COOLDOWN_MAX_MS,
                                                 reg == REG_SAT ? VKA_COOLDOWN_SETTLE_MS : 0, &cr);
            const int bact = vka_precell_cooldown_action(bcs, cr.end_hot_c, VKA_THERMAL_STOP_C);
            if (bact != VKA_PRECELL_RUN) {
                ++therm.cooldown_timeouts;
                ++total_fails;
                printf("BASELINE_STOPPED_COOLDOWN regime=%s end_c=%.1f action=%s\n",
                       k_reg_names[reg], cr.end_hot_c,
                       bact == VKA_PRECELL_ABORT_HOT ? "ABORT_HOT" : "FAIL_CELL");
                fflush(stdout);
                if (bact == VKA_PRECELL_ABORT_HOT) {
                    ++therm.cooldown_hot_aborts;
                    comp_destroy_res(&c);
                    gfx_destroy_res(&g);
                    vkt_shutdown();
                    printf("VERDICT vk_async: THERMAL_ABORT (fails=%d thermal_stops=%d sensor_errors=%d cooldown_timeouts=%d)\n",
                           total_fails, therm.stop_count, therm.sensor_errors, therm.cooldown_timeouts);
                    return 4;
                }
                continue;
            }
            const int nf = (reg == REG_SAT) ? (smoke ? 8 : 16) : (smoke ? 6 : 12);
            double bhot = -1.0;
            const int bhs = run_gfx_alone_baseline(&g, reg, nf, &bhot);
            if (bhs != VKA_HOT_OK) {
                const int act = vka_thermal_on_event(&therm, bhs);
                printf("BASELINE_STOPPED_THERMAL regime=%s hot=%.1fC stop_num=%d/%d action=%s\n",
                       k_reg_names[reg], bhot, therm.stop_count, therm.max_stops,
                       act == VKA_THERM_ABORT_RUN ? "ABORT_RUN" : "CONTINUE_NEXT_ARM");
                fflush(stdout);
                if (act == VKA_THERM_ABORT_RUN) {
                    comp_destroy_res(&c);
                    gfx_destroy_res(&g);
                    vkt_shutdown();
                    printf("VERDICT vk_async: THERMAL_ABORT (fails=%d thermal_stops=%d sensor_errors=%d cooldown_timeouts=%d)\n",
                           total_fails, therm.stop_count, therm.sensor_errors, therm.cooldown_timeouts);
                    return 4;
                }
            }
        }

        const uint32_t base_chains[2] = {1u, VKA_CHAIN_SAT};
        int abort_baseline = 0;
        for (int j = 0; j < JOB_COUNT && !abort_baseline; ++j) {
            if (smoke && j != JOB_FP16 && j != JOB_TEX) continue;
            for (int ci = 0; ci < 2 && !abort_baseline; ++ci) {
                const uint32_t cn = base_chains[ci];
                char clbl[64];
                snprintf(clbl, sizeof clbl, "baseline_comp_%s_n%u", k_job_names[j], cn);
                const int ccs = wait_cooldown_logged(clbl, VKA_COOLDOWN_TARGET_C, 0.0,
                                                     VKA_COOLDOWN_MAX_MS, 0, &cr);
                const int cact = vka_precell_cooldown_action(ccs, cr.end_hot_c, VKA_THERMAL_STOP_C);
                if (cact != VKA_PRECELL_RUN) {
                    ++therm.cooldown_timeouts;
                    ++total_fails;
                    printf("BASELINE_STOPPED_COOLDOWN job=%s chain=%u end_c=%.1f action=%s\n",
                           k_job_names[j], cn, cr.end_hot_c,
                           cact == VKA_PRECELL_ABORT_HOT ? "ABORT_HOT" : "FAIL_CELL");
                    fflush(stdout);
                    if (cact == VKA_PRECELL_ABORT_HOT) {
                        ++therm.cooldown_hot_aborts;
                        abort_baseline = 1;
                        break;
                    }
                    continue;
                }
                const uint32_t it = job_iters(j, cn);
                record_comp_chain(&c, j, cn, it, 0);
                int chs = VKA_HOT_OK;
                double chot = -1.0;
                (void)run_comp_chain_once(&chs, &chot);  // warmup
                if (chs != VKA_HOT_OK) {
                    const int act = vka_thermal_on_event(&therm, chs);
                    printf("BASELINE_STOPPED_THERMAL job=%s chain=%u hot=%.1fC stop_num=%d/%d action=%s\n",
                           k_job_names[j], cn, chot, therm.stop_count, therm.max_stops,
                           act == VKA_THERM_ABORT_RUN ? "ABORT_RUN" : "CONTINUE_NEXT_ARM");
                    fflush(stdout);
                    if (act == VKA_THERM_ABORT_RUN) abort_baseline = 1;
                    break;
                }
                vkt_metrics_start();
                double gms[3], wms[3];
                const int nt = smoke ? 2 : 3;
                int comp_stop = 0;
                for (int r = 0; r < nt; ++r) {
                    VkaCompTrial tr = run_comp_chain_once(&chs, &chot);
                    if (chs != VKA_HOT_OK) {
                        comp_stop = 1;
                        break;
                    }
                    gms[r] = tr.gpu_ms;
                    wms[r] = tr.wall_ms;
                }
                VktMetrics mt = vkt_metrics_stop();
                if (comp_stop) {
                    const int act = vka_thermal_on_event(&therm, chs);
                    printf("BASELINE_STOPPED_THERMAL job=%s chain=%u hot=%.1fC stop_num=%d/%d action=%s\n",
                           k_job_names[j], cn, chot, therm.stop_count, therm.max_stops,
                           act == VKA_THERM_ABORT_RUN ? "ABORT_RUN" : "CONTINUE_NEXT_ARM");
                    fflush(stdout);
                    if (act == VKA_THERM_ABORT_RUN) abort_baseline = 1;
                    break;
                }
                const int bad = verify_comp_job(&c, j, cn, it);
                if (bad != 0) ++total_fails;
                const double gp50 = vka_percentile(gms, nt, 0.50);
                const double wp50 = vka_percentile(wms, nt, 0.50);
                comp_alone_p50[j][chain_slot(cn)] = gp50;
                const uint64_t tw = vka_job_total_work(j, cn);
                printf("BASELINE_COMP job=%-6s chain=%2u iters=%4u total_work=%llu(%s) gpu_p50=%.3f wall_p50=%.3f sclk=%.0fMHz uclk=%.0fMHz hot=%.0fC exact=%s(bad=%d)\n",
                       k_job_names[j], cn, it, (unsigned long long)tw, vka_job_work_unit(j),
                       gp50, wp50, mt.sclk_med_mhz, mt.uclk_med_mhz, mt.hot_max_c,
                       bad == 0 ? "ok" : "FAIL", bad);
                fflush(stdout);
            }
        }
        comp_destroy_res(&c);
        gfx_destroy_res(&g);
        vkt_shutdown();
        if (abort_baseline) {
            printf("VERDICT vk_async: THERMAL_ABORT (fails=%d thermal_stops=%d sensor_errors=%d cooldown_timeouts=%d)\n",
                   total_fails, therm.stop_count, therm.sensor_errors, therm.cooldown_timeouts);
            return 4;
        }
    }

    // 5. Build ABBA priority schedule across all allowed priorities.
    int abba_seq[8] = {0};
    int abba_len = 0;
    if (n_allowed > 0) {
        if (smoke) {
            abba_seq[0] = allowed_idx[0];
            abba_len = 1;
            if (n_allowed >= 2) {
                abba_seq[1] = allowed_idx[n_allowed - 1];
                abba_len = 2;
            }
        } else {
            abba_len = vka_build_abba_schedule(allowed_idx, n_allowed, abba_seq);
        }
    } else {
        abba_seq[0] = -1;
        abba_seq[1] = -1;
        abba_len = smoke ? 1 : 2;
    }

    printf("ABBA_SCHEDULE len=%d order=", abba_len);
    for (int i = 0; i < abba_len; ++i) {
        printf("%s%s", i ? "," : "", (abba_seq[i] >= 0) ? k_prio_names[abba_seq[i]] : "DEFAULT");
    }
    printf(" regimes=PACE-60,SAT-100 sat_chain=%u pace_chain=%u\n", VKA_CHAIN_SAT, VKA_CHAIN_PACE);
    fflush(stdout);

    // 6. Run PACE-60 regime FIRST across all ABBA priority arms, then bounded fan pre-warm,
    //    then SAT-100 across all ABBA priority arms. Each arm uses a fresh VkDevice; every arm
    //    and every repeat waits for hotspot < 60 C (bounded, logged) before submitting GPU work.
    const int reg_order[REG_COUNT] = {REG_PACE, REG_SAT};
    int abort_run = 0;

    for (int ri = 0; ri < REG_COUNT && !abort_run; ++ri) {
        const int reg = reg_order[ri];
        if (reg == REG_SAT) {
            const int pw_act = run_fan_prewarm(&therm, base_use_gp, base_prio, smoke);
            if (pw_act == VKA_THERM_ABORT_RUN) {
                printf("VERDICT vk_async: THERMAL_ABORT (fails=%d thermal_stops=%d sensor_errors=%d cooldown_timeouts=%d)\n",
                       total_fails, therm.stop_count, therm.sensor_errors, therm.cooldown_timeouts);
                return 4;
            }
        }

        const uint32_t reg_chains[2] = {1u, (reg == REG_SAT) ? VKA_CHAIN_SAT : VKA_CHAIN_PACE};
        const int ntrials = smoke ? 1 : ((reg == REG_SAT) ? 2 : 3);
        const int max_gfx_frames = (reg == REG_SAT) ? GFX_SAT_REPEAT_MAX_FRAMES : GFX_PACE_REPEAT_MAX_FRAMES;
        const double settle_c = (reg == REG_SAT) ? VKA_COOLDOWN_SETTLE_C : 0.0;
        const int settle_ms = (reg == REG_SAT) ? VKA_COOLDOWN_SETTLE_MS : 0;

        for (int pass = 0; pass < abba_len && !abort_run; ++pass) {
            const int pidx = abba_seq[pass];
            const int use_gp = (pidx >= 0) ? 1 : 0;
            const VkQueueGlobalPriority prio = (pidx >= 0) ? k_all_prios[pidx] : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;
            const char * pname = (pidx >= 0) ? k_prio_names[pidx] : "DEFAULT";

            VKA_CHECK(vkt_init_async(1, use_gp, prio, 0));
            VT.out_dir[0] = '\0';  // ACO listings already captured in baseline pass.

            char arm_lbl[96];
            snprintf(arm_lbl, sizeof arm_lbl, "arm_%s_p%d_%s", k_reg_names[reg], pass, pname);
            VkaCooldownResult arm_cr;
            const int arm_cs = wait_cooldown_logged(arm_lbl, VKA_COOLDOWN_TARGET_C, settle_c,
                                                    VKA_COOLDOWN_MAX_MS, settle_ms, &arm_cr);
            const int arm_act = vka_precell_cooldown_action(arm_cs, arm_cr.end_hot_c, VKA_THERMAL_STOP_C);
            if (arm_act != VKA_PRECELL_RUN) {
                ++therm.cooldown_timeouts;
                ++total_fails;
                printf("ARM_STOPPED_COOLDOWN regime=%s pass=%d prio=%s end_c=%.1f timeouts=%d action=%s\n",
                       k_reg_names[reg], pass, pname, arm_cr.end_hot_c, therm.cooldown_timeouts,
                       arm_act == VKA_PRECELL_ABORT_HOT ? "ABORT_HOT" : "FAIL_CELL");
                fflush(stdout);
                vkt_shutdown();
                if (arm_act == VKA_PRECELL_ABORT_HOT) {
                    ++therm.cooldown_hot_aborts;
                    abort_run = 1;
                }
                continue;
            }

            GfxRes g;
            CompRes c;
            gfx_init_res(&g, 0);
            comp_init_res(&c, 1, 0);

            int arm_stopped = 0;
            for (int j = 0; j < JOB_COUNT && !arm_stopped && !abort_run; ++j) {
                if (smoke && j != JOB_FP16 && j != JOB_TEX) continue;
                for (int ci = 0; ci < 2 && !arm_stopped && !abort_run; ++ci) {
                    const uint32_t cn = reg_chains[ci];
                    const uint32_t it = job_iters(j, cn);
                    record_comp_chain(&c, j, cn, it, 0);

                    VkaOverlapAccum acc;
                    vka_overlap_accum_init(&acc);
                    double gms[8] = {0}, wms[8] = {0};
                    double sclks[8] = {0}, uclks[8] = {0};
                    int n_clk = 0;
                    double hot_max = 0.0;
                    int cell_bad = 0;
                    int cell_cooldown_failed = 0;
                    int completed_r = 0;

                    for (int r = 0; r < ntrials; ++r) {
                        char rep_lbl[128];
                        snprintf(rep_lbl, sizeof rep_lbl, "rep_%s_p%d_%s_%s_n%u_r%d",
                                 k_reg_names[reg], pass, pname, k_job_names[j], cn, r);
                        VkaCooldownResult rep_cr;
                        const int rep_cs = wait_cooldown_logged(rep_lbl, VKA_COOLDOWN_TARGET_C, settle_c,
                                                                VKA_COOLDOWN_MAX_MS, settle_ms, &rep_cr);
                        const int rep_act = vka_precell_cooldown_action(rep_cs, rep_cr.end_hot_c, VKA_THERMAL_STOP_C);
                        if (rep_act != VKA_PRECELL_RUN) {
                            ++therm.cooldown_timeouts;
                            ++total_fails;
                            cell_cooldown_failed = 1;
                            printf("RESULT vk_async pass=%d prio=%-8s regime=%-7s job=%-6s chain=%2u repeat=%d "
                                   "SKIPPED_COOLDOWN_TIMEOUT end_c=%.1f action=%s exact=FAIL\n",
                                   pass, pname, k_reg_names[reg], k_job_names[j], cn, r,
                                   rep_cr.end_hot_c,
                                   rep_act == VKA_PRECELL_ABORT_HOT ? "ABORT_HOT" : "FAIL_CELL");
                            fflush(stdout);
                            if (rep_act == VKA_PRECELL_ABORT_HOT) {
                                ++therm.cooldown_hot_aborts;
                                arm_stopped = 1;
                                abort_run = 1;
                            }
                            break;
                        }

                        GfxWorker gw;
                        memset(&gw, 0, sizeof gw);
                        gw.gfx = &g;
                        gw.iters = (reg == REG_SAT) ? GFX_SAT_ITERS : GFX_PACE_ITERS;
                        gw.in_flight = (reg == REG_SAT) ? 2 : 1;
                        gw.pace_ms = (reg == REG_SAT) ? 0.0 : GFX_PACE_MS;
                        gw.warmup_frames = GFX_WARMUP_FRAMES;
                        gw.max_frames = max_gfx_frames;
                        gw.frame_cap = max_gfx_frames + 16;
                        atomic_init(&gw.stop_req, 0);
                        atomic_init(&gw.thermal_status, VKA_HOT_OK);
                        atomic_init(&gw.thermal_hot_mc, -1);
                        gw.frames = (VkaGfxFrame *)calloc((size_t)gw.frame_cap, sizeof(VkaGfxFrame));
                        pthread_mutex_init(&gw.mu, NULL);
                        pthread_cond_init(&gw.cv, NULL);

                        pthread_t gthr;
                        pthread_create(&gthr, NULL, gfx_worker_thread, &gw);

                        pthread_mutex_lock(&gw.mu);
                        while (!gw.ready) pthread_cond_wait(&gw.cv, &gw.mu);
                        pthread_mutex_unlock(&gw.mu);

                        int comp_hs = VKA_HOT_OK;
                        double comp_hot = -1.0;
                        VkaCompTrial tr = {0.0, 0.0, 0.0, 0.0};
                        VktMetrics mt;
                        memset(&mt, 0, sizeof mt);

                        if (atomic_load(&gw.thermal_status) == VKA_HOT_OK) {
                            if (reg == REG_PACE && r > 0) {
                                // Phase-sweep across 16.6 ms cadence (0, 5.5, 11.0 ms).
                                usleep((useconds_t)(r * 5500));
                            }
                            vkt_metrics_start();
                            tr = run_comp_chain_once(&comp_hs, &comp_hot);
                            mt = vkt_metrics_stop();
                        }

                        atomic_store(&gw.stop_req, 1);
                        pthread_join(gthr, NULL);

                        const int gw_hs = atomic_load(&gw.thermal_status);
                        const int gw_hmc = atomic_load(&gw.thermal_hot_mc);
                        const int combined_hs = vka_thermal_merge_status(gw_hs, comp_hs);

                        if (combined_hs != VKA_HOT_OK) {
                            double stop_hot = comp_hot;
                            if (gw_hmc > 0 && (double)gw_hmc * 1e-3 > stop_hot) stop_hot = (double)gw_hmc * 1e-3;
                            if (mt.hot_max_c > stop_hot) stop_hot = mt.hot_max_c;
                            const int act = vka_thermal_on_event(&therm, combined_hs);
                            printf("ARM_STOPPED_THERMAL regime=%-7s pass=%d prio=%-8s job=%-6s chain=%2u repeat=%d "
                                   "status=%s hot=%.1fC stop_num=%d/%d action=%s\n",
                                   k_reg_names[reg], pass, pname, k_job_names[j], cn, r,
                                   combined_hs == VKA_HOT_SENSOR_ERR ? "SENSOR_ERR" : "OVER_LIMIT",
                                   stop_hot, therm.stop_count, therm.max_stops,
                                   act == VKA_THERM_ABORT_RUN ? "ABORT_RUN" : "CONTINUE_NEXT_ARM");
                            fflush(stdout);
                            pthread_mutex_destroy(&gw.mu);
                            pthread_cond_destroy(&gw.cv);
                            free(gw.frames);
                            arm_stopped = 1;
                            if (act == VKA_THERM_ABORT_RUN) abort_run = 1;
                            break;
                        }

                        vka_overlap_accum_add_batch(&acc, gw.frames, gw.frame_count, &tr, 1);
                        for (int fi = 0; fi < gw.frame_count; ++fi) {
                            if (gw.frames[fi].hot_c > hot_max) hot_max = gw.frames[fi].hot_c;
                        }
                        if (mt.hot_max_c > hot_max) hot_max = mt.hot_max_c;
                        if (comp_hot > hot_max) hot_max = comp_hot;
                        if (mt.n_samples > 0 && n_clk < 8) {
                            sclks[n_clk] = mt.sclk_med_mhz;
                            uclks[n_clk] = mt.uclk_med_mhz;
                            ++n_clk;
                        }
                        pthread_mutex_destroy(&gw.mu);
                        pthread_cond_destroy(&gw.cv);
                        free(gw.frames);

                        cell_bad += verify_comp_job(&c, j, cn, it);
                        gms[completed_r] = tr.gpu_ms;
                        wms[completed_r] = tr.wall_ms;
                        ++completed_r;
                    }

                    if (!arm_stopped && !cell_cooldown_failed && completed_r == ntrials) {
                        if (cell_bad != 0) ++total_fails;
                        const VkaGfxOverlapStats ov = vka_overlap_accum_finish(&acc);
                        const double gp50 = vka_percentile(gms, completed_r, 0.50);
                        const double gp95 = vka_percentile(gms, completed_r, 0.95);
                        const double gmax = vka_percentile(gms, completed_r, 1.00);
                        const double wp50 = vka_percentile(wms, completed_r, 0.50);
                        const double sclk_med = (n_clk > 0) ? vka_percentile(sclks, n_clk, 0.50) : NAN;
                        const double uclk_med = (n_clk > 0) ? vka_percentile(uclks, n_clk, 0.50) : NAN;
                        const double base_ms = comp_alone_p50[j][chain_slot(cn)];
                        const double slow = (base_ms > 1e-6) ? (gp50 / base_ms) : NAN;
                        const uint64_t tw = vka_job_total_work(j, cn);

                        printf("RESULT vk_async pass=%d prio=%-8s regime=%-7s job=%-6s chain=%2u iters=%3u total_work=%llu "
                               "comp_gpu_p50=%.3f comp_gpu_p95=%.3f comp_gpu_max=%.3f comp_wall_p50=%.3f "
                               "slow_vs_alone=%.2fx gfx_dt_avg=%.3f gfx_dt_p50=%.3f gfx_dt_p95=%.3f "
                               "gfx_gpu_p50=%.3f gfx_ov_frames=%d comp_strict_ov=%d/%d "
                               "sclk=%.0fMHz uclk=%.0fMHz hot=%.0fC exact=%s(bad=%d)\n",
                               pass, pname, k_reg_names[reg], k_job_names[j], cn, it, (unsigned long long)tw,
                               gp50, gp95, gmax, wp50, slow,
                               ov.dt_avg_ms, ov.dt_p50_ms, ov.dt_p95_ms, ov.gfx_gpu_p50_ms,
                               ov.n_ov_frames, ov.n_strict_ov, completed_r,
                               sclk_med, uclk_med, hot_max,
                               cell_bad == 0 ? "ok" : "FAIL", cell_bad);
                        fflush(stdout);
                    }
                    vka_overlap_accum_free(&acc);
                }
            }

            comp_destroy_res(&c);
            gfx_destroy_res(&g);
            vkt_shutdown();
        }
    }

    const int exit_rc = vka_thermal_exit_code(&therm, total_fails);
    const char * vstr = (exit_rc == 0) ? "PASS" : (exit_rc == 4 ? "THERMAL_ABORT" : "FAIL");
    printf("VERDICT vk_async: %s (fails=%d thermal_stops=%d sensor_errors=%d cooldown_timeouts=%d)\n",
           vstr, total_fails, therm.stop_count, therm.sensor_errors, therm.cooldown_timeouts);
    return exit_rc;
}
