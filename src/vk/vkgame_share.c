// P2: compute beside drawing. The draw load is an offscreen 2560x1440 RGBA16F target (no window, no swapchain) filled
// by one full-screen triangle whose pixel shader runs an adjustable number of dependent FMA steps; a "frame" is one
// command buffer with that pass, submitted on the graphics queue at a set cadence. The compute probe is the study's
// one-group dispatch (vksched.c), submitted and waited for back to back on the graphics queue in separate submissions
// or on the compute-only queue; or (cq=in) a 20-dispatch list recorded into the frame's own command buffer after the
// pass, as a game records post-processing into its graphics list.
//   lone [--swap-labels]               calibration: both queues interleaved, then each queue alone (vksched's design)
//   share name=..,fps=..,frame=..,cq=none|g|c|in,dur=..,draw=on|off
// After review (heat incident of the first development run): a draw whose pixel shader runs FMA steps (frame > 0) is
// refused; only draw=off arms and the pass without ALU (E7, mode rt) run. The busy share of a draw arm is computed
// from the measured pass times while it runs; above 0.40 the arm ends at 0.1 s.
//   rt name=..,sizes=2560x1440:4096x1536:4096x4096:8192x4096,rounds=5,passes=20     E7: write rate against target size
//   slow name=..,n=300,rounds=6,cpu=3                  E6: blocking wait against spin-polling, pinned against unpinned
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <string.h>

#include "vkgame.h"

#define DW 2560
#define DH 1440
#define RING 64

static VkImage g_img;
static VkDeviceMemory g_imem;
static VkImageView g_iview;
static VkPipeline g_dpipe;
static VkPipelineLayout g_dpl;
static int g_draw_ready;
static VkPipeline draw_pipe(uint32_t w, uint32_t h, const char *label);
static void target_layout(VkImage img);

static uint32_t mem_type_dl(uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    extern VkPhysicalDevice g_pd;
    vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    return 0;
}

void draw_setup(int report) {
    (void)report;
    if (g_draw_ready) return;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ii.extent = (VkExtent3D){DW, DH, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VK_CHECK(vkCreateImage(g_dev, &ii, NULL, &g_img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, g_img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type_dl(mr.memoryTypeBits);
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &g_imem));
    VK_CHECK(vkBindImageMemory(g_dev, g_img, g_imem, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = g_img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ii.format;
    vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(g_dev, &vi, NULL, &g_iview));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &g_dpl));
    g_dpipe = draw_pipe(DW, DH, "gdraw");
    target_layout(g_img);
    g_draw_ready = 1;
}

// The full-screen pass's pipeline for a w x h target (static viewport), with its machine code captured.
static VkPipeline draw_pipe(uint32_t w, uint32_t h, const char *label) {
    VkPipeline pipe;
    VkShaderModule vs = mkmod("gdraw_vert.spv"), fs = mkmod("gdraw_frag.spv");
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fs, st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = vps.scissorCount = 1;
    vps.pViewports = &vp;
    vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &fmt;
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
    gi.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    gi.stageCount = 2;
    gi.pStages = st;
    gi.pVertexInputState = &vin;
    gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vps;
    gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms;
    gi.pColorBlendState = &cb;
    gi.layout = g_dpl;
    VK_CHECK(vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &pipe));
    vkDestroyShaderModule(g_dev, vs, NULL);
    vkDestroyShaderModule(g_dev, fs, NULL);
    Pipe pinfo = {pipe, "", 0};
    snprintf(pinfo.name, sizeof pinfo.name, "%s", label);
    capture(&pinfo);
    return pipe;
}

// The target goes to the attachment layout once.
static void target_layout(VkImage img) {
    VkCommandBuffer c = cb_alloc(&QU[0]);
    cb_begin(c);
    VkImageMemoryBarrier2 ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    ib.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    ib.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img;
    ib.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(c, &di);
    VK_CHECK(vkEndCommandBuffer(c));
    run_once(&QU[0], c);
    vkFreeCommandBuffers(g_dev, QU[0].pool, 1, &c);
}

// One pass over the whole target; a write-after-write dependency on the previous frame's pass first.
static void record_draw(VkCommandBuffer cb, uint32_t iters) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = g_iview;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = (VkRect2D){{0, 0}, {DW, DH}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_dpipe);
    vkCmdPushConstants(cb, g_dpl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &iters);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

// GPU time of one pass at `iters`, the median of `n` frames after 3 unmeasured ones.
static double draw_time_us(uint32_t iters, int n) {
    VkCommandBuffer cb = cb_alloc(&QU[0]);
    VkQueryPool qp = mkqp(2);
    double t[64];
    for (int i = 0; i < n + 3 && i < 64; i++) {
        cb_begin(cb);
        vkCmdResetQueryPool(cb, qp, 0, 2);
        ts_top(cb, qp, 0);
        record_draw(cb, iters);
        ts_bot(cb, qp, 1);
        VK_CHECK(vkEndCommandBuffer(cb));
        run_once(&QU[0], cb);
        t[i] = (double)(qget(qp, 1) - qget(qp, 0)) * g_tsp / 1e3;
    }
    vkDestroyQueryPool(g_dev, qp, NULL);
    vkFreeCommandBuffers(g_dev, QU[0].pool, 1, &cb);
    return pctl(t + 3, n, .5);
}

// ---------------------------------------------------------------------------------------------------------------
// the one-group probe

typedef struct {
    double tsub, tdone;  // host ns
    uint64_t top, bot, r0, r1;
    int q, ok, inframe, busy_at_sub;
} Tiny;

typedef struct {
    Pipe p;
    Buf res, rec;
    VkDescriptorSet ds;
    VkQueryPool qp[2];
    VkCommandBuffer cb[2][RING];
    VkFence f[2];
    int j[2];
} Probe;

// The first timestamp is at BOTTOM_OF_PIPE, as in vksched.c (VKG_TINY_TS=top: TOP_OF_PIPE; which one is used did not
// decide the compute queue's slow path in the development runs). VKG_TINY_SPV=<file>: another one-group shader (for
// example the study's vksched_one.spv; then there are no REALTIME stamps).
static void probe_init(Probe *pr) {
    const char *spv = getenv("VKG_TINY_SPV"), *tsm = getenv("VKG_TINY_TS");
    if (spv && !*spv) spv = NULL;
    memset(pr, 0, sizeof *pr);
    const int bottom_first = !(tsm && !strcmp(tsm, "top"));
    pr->p = mkcomp(spv ? spv : "gtiny.spv", spv ? "tiny_alt" : "gtiny", 64, 0);
    printf("INFO probe shader %s, first timestamp %s\n", spv ? spv : "gtiny.spv", bottom_first ? "BOTTOM_OF_PIPE" : "TOP_OF_PIPE");
    pr->res = mkbuf(RING * 64 * 4, 2);
    pr->rec = mkbuf(RING * 16, 2);
    pr->ds = mkset(&pr->res, &pr->rec, NULL, NULL);
    for (int q = 0; q < 2; q++) {
        pr->qp[q] = mkqp(2 * RING);
        pr->f[q] = mkfence();
        for (int j = 0; j < RING; j++) {
            VkCommandBuffer cb = pr->cb[q][j] = cb_alloc(&QU[q]);
            cb_begin(cb);
            vkCmdResetQueryPool(cb, pr->qp[q], 2 * j, 2);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pr->p.p);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &pr->ds, 0, NULL);
            uint32_t pc[2] = {1000u + 1000u * (uint32_t)q, (uint32_t)j};
            vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
            if (bottom_first) ts_bot(cb, pr->qp[q], 2 * j);
            else ts_top(cb, pr->qp[q], 2 * j);
            vkCmdDispatch(cb, 1, 1, 1);
            ts_bot(cb, pr->qp[q], 2 * j + 1);
            VK_CHECK(vkEndCommandBuffer(cb));
        }
    }
}

// Submits one probe on queue q and waits; the slot's results are poisoned first so a dispatch that did not run fails.
static void probe_one(Probe *pr, int q, Tiny *t) {
    const int j = pr->j[q];
    pr->j[q] = (j + 1) % RING;
    volatile uint32_t *res = (uint32_t *)pr->res.map + j * 64;
    for (int i = 0; i < 64; i++) res[i] = 0xFFFFFFFFu;
    t->q = q;
    t->tsub = mono_ns();
    submit(&QU[q], pr->cb[q][j], pr->f[q]);
    fwait(pr->f[q]);
    t->tdone = mono_ns();
    t->top = qget(pr->qp[q], 2 * j);
    t->bot = qget(pr->qp[q], 2 * j + 1);
    volatile uint32_t *r = (uint32_t *)pr->rec.map + 4 * j;
    t->r0 = r[0] | (uint64_t)r[1] << 32;
    t->r1 = r[2] | (uint64_t)r[3] << 32;
    int ok = 1;
    for (int i = 0; i < 64; i++) ok &= res[i] == 1000u + 1000u * (uint32_t)q + (uint32_t)i;
    t->ok = ok;
}

// Distribution lines for the probes of queue q in [a, b), split by "slow" (wall above 60 us), with the timeline of
// each sample cut into: submit -> TOP timestamp -> shader entry -> shader exit -> BOTTOM timestamp -> host wakes.
static void tiny_report(const char *tag, const Tiny *t, int a, int b, int q, const Map *m, int cls) {
    int n = 0;
    for (int i = a; i < b; i++) n += t[i].q == q && (cls < 0 || t[i].inframe == cls);
    if (!n) return;
    double *w = malloc(n * 8), *s1 = malloc(n * 8), *s2 = malloc(n * 8), *s3 = malloc(n * 8), *s4 = malloc(n * 8),
           *s5 = malloc(n * 8), *g = malloc(n * 8);
    int k = 0, bad = 0;
    for (int i = a; i < b; i++) {
        if (t[i].q != q || (cls >= 0 && t[i].inframe != cls)) continue;
        w[k] = (t[i].tdone - t[i].tsub) / 1e3;
        s1[k] = (map_ns(m, t[i].top) - t[i].tsub) / 1e3;
        s2[k] = (double)(int64_t)(t[i].r0 - t[i].top) * g_tsp / 1e3;
        s3[k] = (double)(int64_t)(t[i].r1 - t[i].r0) * g_tsp / 1e3;
        s4[k] = (double)(int64_t)(t[i].bot - t[i].r1) * g_tsp / 1e3;
        s5[k] = (t[i].tdone - map_ns(m, t[i].bot)) / 1e3;
        g[k] = (double)(int64_t)(t[i].bot - t[i].top) * g_tsp / 1e3;
        bad += !t[i].ok;
        k++;
    }
    char tg[128];
    snprintf(tg, sizeof tg, "%s-%s%s", tag, QU[q].name, cls < 0 ? "" : cls ? "-submitted_during_frame" : "-submitted_between_frames");
    dist(tg, "wall_us(submit_to_fence)", w, n);
    dist(tg, "gpu_us(top_to_bottom_ts)", g, n);
    const double med = pctl(w, n, .5);
    printf("RESULT %s n=%d wall_median_us=%.2f share_above_60us=%.4f share_above_2x_median=%.4f wrong_results=%d\n", tg, n,
           med, frac_above(w, n, 60), frac_above(w, n, 2 * med), bad);
    // The timeline pieces for fast and slow samples separately.
    for (int sl = 0; sl < 2; sl++) {
        double p[5][4096];
        int c = 0;
        for (int i = 0; i < n && c < 4096; i++)
            if ((w[i] > 60) == sl) p[0][c] = s1[i], p[1][c] = s2[i], p[2][c] = s3[i], p[3][c] = s4[i], p[4][c] = s5[i], c++;
        if (!c) continue;
        printf("PIECES %s %s n=%d median_us: submit->first_ts %.2f, first_ts->entry %.2f, entry->exit %.2f, exit->bottom_ts "
               "%.2f, bottom_ts->host_wakes %.2f | p95: %.2f %.2f %.2f %.2f %.2f\n",
               tg, sl ? "slow(>60us)" : "fast", c, pctl(p[0], c, .5), pctl(p[1], c, .5), pctl(p[2], c, .5),
               pctl(p[3], c, .5), pctl(p[4], c, .5), pctl(p[0], c, .95), pctl(p[1], c, .95), pctl(p[2], c, .95),
               pctl(p[3], c, .95), pctl(p[4], c, .95));
    }
    free(w), free(s1), free(s2), free(s3), free(s4), free(s5), free(g);
}

static void tiny_csv(const char *name, const Tiny *t, int n, const Map *m, double c0) {
    if (!g_out[0]) return;
    char path[900], nm[160];
    snprintf(nm, sizeof nm, "tiny-%s.csv", name);
    out_path(path, sizeof path, nm);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "arm,i,queue,t_submit_us,wall_us,submit_to_first_ts_us,first_ts_to_entry_us,entry_to_exit_us,exit_to_bottom_us,"
               "bottom_to_host_us,submitted_during_frame,ok\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d\n", name, i, QU[t[i].q].name, (t[i].tsub - c0) / 1e3,
                (t[i].tdone - t[i].tsub) / 1e3, (map_ns(m, t[i].top) - t[i].tsub) / 1e3,
                (double)(int64_t)(t[i].r0 - t[i].top) * g_tsp / 1e3, (double)(int64_t)(t[i].r1 - t[i].r0) * g_tsp / 1e3,
                (double)(int64_t)(t[i].bot - t[i].r1) * g_tsp / 1e3, (t[i].tdone - map_ns(m, t[i].bot)) / 1e3,
                t[i].inframe, t[i].ok);
    fclose(f);
}

int mode_lone(int argc, char **argv) {
    int swap = argc > 0 && !strcmp(argv[0], "--swap-labels");
    Probe pr;
    probe_init(&pr);
    const char *ne = getenv("VKG_LONE_N");  // fewer samples for the command-stream dump job
    const int N = ne ? atoi(ne) : 2000, W = ne ? 2 : 50;
    Tiny *t = calloc(4 * N, sizeof(Tiny)), tw;
    Map m;
    double c0 = mono_ns();
    // interleaved, order alternating each round (vksched "p0ms")
    map_begin(&m);
    for (int i = 0; i < W; i++) probe_one(&pr, 0, &tw), probe_one(&pr, 1, &tw);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 2; k++) probe_one(&pr, (i & 1) ? 1 - k : k, &t[2 * i + k]);
    map_end(&m);
    map_report("lone-interleaved", &m);
    tiny_report("lone-interleaved", t, 0, 2 * N, 0, &m, -1);
    tiny_report("lone-interleaved", t, 0, 2 * N, 1, &m, -1);
    tiny_csv("lone-interleaved", t, 2 * N, &m, c0);
    // alone: compute first, then graphics (vksched "alone")
    Map m2;
    map_begin(&m2);
    Tiny *u = t + 2 * N;
    for (int q = 1; q >= 0; q--) {
        for (int i = 0; i < W; i++) probe_one(&pr, q, &tw);
        for (int i = 0; i < N; i++) probe_one(&pr, q, &u[(1 - q) * N + i]);
    }
    map_end(&m2);
    map_report("lone-alone", &m2);
    tiny_report("lone-alone", u, 0, 2 * N, 0, &m2, -1);
    tiny_report("lone-alone", u, 0, 2 * N, 1, &m2, -1);
    tiny_csv("lone-alone", u, 2 * N, &m2, c0);
    state_line("lone", c0, mono_ns());
    // Calibration against scheduling item 6 and SC-12 (READ): alone, the compute queue has 25 to 42 % of its
    // submissions in a slow peak near 100 us, the graphics queue one peak near 30 us (p90 35.8, p99 58).
    double wq[2][2000];
    int bad = 0;
    for (int q = 0; q < 2; q++)
        for (int i = 0; i < N; i++) {
            const Tiny *s = &u[(1 - q) * N + i];
            wq[q][i] = (s->tdone - s->tsub) / 1e3;
            bad += !s->ok;
        }
    // Judged here, against what the study documents as stable (scheduling item 6, READ): the graphics queue has one
    // peak near 30 us (median 32.1 us, 1.0 % above 60 us), and a dispatch takes longer on the GPU on the compute queue
    // (0.52-0.56 us against 0.28 us). The share of slow compute submissions is reported, not judged: it changed from
    // run to run in the development runs (0.02 to 1.00 per process); the study's own vksched is the calibration job.
    const int gq = swap ? 1 : 0, cq = swap ? 0 : 1;
    double fg = frac_above(wq[gq], N, 60), mg = pctl(wq[gq], N, .5), gp[2][2000];
    for (int q = 0; q < 2; q++)
        for (int i = 0; i < N; i++) {
            const Tiny *x = &u[(1 - q) * N + i];
            gp[q][i] = (double)(int64_t)(x->bot - x->top) * g_tsp / 1e3;
        }
    const double ratio = pctl(gp[cq], N, .5) / pctl(gp[gq], N, .5);
    printf("INFO lone share of compute-queue submissions above 60 us: %.4f (study vksched alone: 0.300-0.424; SC-12: "
           "0.253-0.303)\n", frac_above(wq[1], N, 60));
    if (ne) {
        printf("INFO lone: %d samples per queue; calibration not judged\n", N);
        free(t);
        return 0;
    }
    verdict("lone_calibration", fg <= 0.03 && mg >= 15 && mg <= 45 && ratio >= 1.5 && !bad,
            "%squeue labelled graphics: median %.1f us (study 32.1; limit 15..45), %.3f above 60 us (study 0.010; limit "
            "0.03); GPU time of the dispatch, queue labelled compute over queue labelled graphics: %.2f (study "
            "0.52-0.56 / 0.28 = 1.9-2.0; limit at least 1.5); wrong results %d",
            swap ? "DELIBERATE BREAK, labels swapped: " : "", mg, fg, ratio, bad);
    free(t);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// share: draw thread + compute probe thread

typedef struct {
    double tsub;
    uint64_t top, rend, bot;  // TOP before the pass, BOTTOM after the pass, BOTTOM after the in-frame list
    uint64_t ctop[20];  // TOP before each dispatch of the in-frame list
} Frame;

typedef struct {
    // spec
    char name[64], cq[8];
    double fps, frame_ms, dur;
    int draw;
    uint32_t iters;
    // state
    volatile int stop, hot85, duty_cut;
    double busy_ns;
    double t0, t1;
    Frame *fr;
    int nfr, maxfr;
    // in-frame list
    Pipe chain;
    Buf rec, chk, err;
    VkDescriptorSet ds;
} Share;

#define FSLOTS 3
#define FQ 64

static void record_frame(Share *s, VkCommandBuffer cb, VkQueryPool qp, int slot) {
    cb_begin(cb);
    const uint32_t qb = (uint32_t)slot * FQ;
    vkCmdResetQueryPool(cb, qp, qb, FQ);
    ts_top(cb, qp, qb);
    record_draw(cb, s->iters);
    ts_bot(cb, qp, qb + 1);
    if (!strcmp(s->cq, "in")) {
        // The study's P1 base list: 20 dispatches of 64 groups spinning 3000 cycles, compute barriers between.
        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->chain.p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &s->ds, 0, NULL);
        for (int k = 0; k < 20; k++) {
            vkCmdPipelineBarrier2(cb, &di);
            if (k == 0) {
                mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            }
            ts_top(cb, qp, qb + 3 + k);
            uint32_t pc[7] = {(uint32_t)k + 1, 3000u, 1u, (uint32_t)((slot * 20 + k) * 64), 0u, 64u * 64u, 0u};
            vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            vkCmdDispatch(cb, 64, 1, 1);
        }
        ts_bot(cb, qp, qb + 2);
    }
    VK_CHECK(vkEndCommandBuffer(cb));
}

static void collect_frame(Share *s, VkQueryPool qp, int slot, Frame *f) {
    const uint32_t qb = (uint32_t)slot * FQ;
    f->top = qget(qp, qb);
    f->rend = qget(qp, qb + 1);
    if (!strcmp(s->cq, "in")) {
        f->bot = qget(qp, qb + 2);
        for (int k = 0; k < 20; k++) f->ctop[k] = qget(qp, qb + 3 + k);
    } else {
        f->bot = f->rend;
    }
}

static void *draw_main(void *arg) {
    Share *s = arg;
    VkQueryPool qp = mkqp(FSLOTS * FQ);
    VkCommandBuffer cb[FSLOTS];
    VkFence f[FSLOTS];
    int fidx[FSLOTS];
    for (int i = 0; i < FSLOTS; i++) cb[i] = cb_alloc(&QU[0]), f[i] = mkfence(), fidx[i] = -1;
    const double period = s->fps > 0 ? 1e9 / s->fps : 0;
    double next = s->t0;
    int inflight_max = s->fps > 0 ? FSLOTS : 2;
    for (int fi = 0; !s->stop && fi < s->maxfr; fi++) {
        if (period > 0) {
            until_ns(next);
            next += period;
        }
        if (mono_ns() >= s->t1 || g_hot_stop) break;
        // The draw load heats the card fast (development run: 60 to 95 C in 0.15 s back to back), so it stops at 85 C.
        if (g_nsmp && g_smp[g_nsmp - 1].hot >= 85.0) {
            s->hot85 = 1;
            break;
        }
        const int slot = fi % inflight_max;
        if (fidx[slot] >= 0) {
            fwait(f[slot]);
            collect_frame(s, qp, slot, &s->fr[fidx[slot]]);
            const Frame *d = &s->fr[fidx[slot]];
            s->busy_ns += (double)(int64_t)(d->rend - d->top) * g_tsp;
            const double el = mono_ns() - s->t0;
            if (el > 0.1e9 && s->busy_ns / el > 0.40) {
                s->duty_cut = 1;
                break;
            }
        }
        record_frame(s, cb[slot], qp, slot);
        s->fr[fi].tsub = mono_ns();
        submit(&QU[0], cb[slot], f[slot]);
        fidx[slot] = fi;
        s->nfr = fi + 1;
    }
    for (int i = 0; i < inflight_max; i++)
        if (fidx[i] >= 0) {
            fwait(f[i]);
            collect_frame(s, qp, i, &s->fr[fidx[i]]);
        }
    for (int i = 0; i < FSLOTS; i++) vkDestroyFence(g_dev, f[i], NULL);
    vkDestroyQueryPool(g_dev, qp, NULL);
    return NULL;
}

static uint32_t g_iters_per_ms;

static int run_share(const char *spec, Probe *pr) {
    Share s;
    memset(&s, 0, sizeof s);
    char b[64];
    kvs(spec, "name", "share", s.name, sizeof s.name);
    kvs(spec, "cq", "c", s.cq, sizeof s.cq);
    s.fps = kvd(spec, "fps", 120);
    s.frame_ms = kvd(spec, "frame", 3);
    s.dur = kvd(spec, "dur", 1.0);
    s.draw = strcmp(kvs(spec, "draw", "on", b, sizeof b), "off") != 0;
    // Thermal rule after review: no pixel-shader FMA load at all; the planned busy share from the nominal frame is a
    // first cap only, the measured pass times decide while the arm runs (draw_main).
    if (s.draw && s.frame_ms > 0) {
        printf("NOT RUN %s: a full-screen pixel-shader FMA load is not allowed after the heat incident\n", s.name);
        return 0;
    }
    const double duty = s.draw ? (s.fps > 0 ? s.frame_ms * s.fps / 1000.0 : 1.0) : 0;
    if (duty > 0.40 && s.dur > 0.1) s.dur = 0.1;
    if (s.dur > 1.0) s.dur = 1.0;
    if (g_hot_stop || hotspot() >= 85.0) {
        printf("THERMAL not starting %s: hotspot %.0f C\n", s.name, hotspot());
        return 1;
    }
    s.iters = (uint32_t)(g_iters_per_ms * s.frame_ms);
    s.maxfr = (int)(s.dur * (s.fps > 0 ? s.fps : 2000)) + 8;
    s.fr = calloc(s.maxfr, sizeof(Frame));
    if (!strcmp(s.cq, "in")) {
        s.chain = mkcomp("gchain.spv", "gchain_l64_inframe", 64, 0);
        s.rec = mkbuf(FSLOTS * 20 * 64 * 16, 0);
        s.chk = mkbuf(64 * 64 * 4, 0);
        s.err = mkbuf(4096, 0);
        s.ds = mkset(&s.rec, &s.chk, &s.err, NULL);
    }
    printf("ARM %s spec=%s draw=%d fps=%.1f frame_target_ms=%.2f iters=%u duty=%.2f dur_s=%.2f compute=%s\n", s.name,
           spec, s.draw, s.fps, s.frame_ms, s.iters, duty, s.dur, s.cq);
    Map m;
    map_begin(&m);
    s.t0 = mono_ns() + 2e6;
    s.t1 = s.t0 + s.dur * 1e9;
    pthread_t th;
    if (s.draw) pthread_create(&th, NULL, draw_main, &s);
    const int maxt = 200000;
    Tiny *t = calloc(maxt, sizeof(Tiny));
    int nt = 0;
    until_ns(s.t0);
    const int cq = !strcmp(s.cq, "g") ? 0 : !strcmp(s.cq, "c") ? 1 : -1;
    while (cq >= 0 && mono_ns() < s.t1 && nt < maxt && !g_hot_stop) probe_one(pr, cq, &t[nt++]);
    if (cq < 0) until_ns(s.t1);
    s.stop = 1;
    if (s.draw) pthread_join(th, NULL);
    map_end(&m);
    map_report(s.name, &m);

    // frames: GPU time of the pass, submit -> TOP, gap between frames
    const int nf = s.nfr;
    double *fd = malloc((nf + 1) * 8), *fs = malloc((nf + 1) * 8), *fgap = malloc((nf + 1) * 8),
           *ftail = malloc((nf + 1) * 8);
    for (int i = 0; i < nf; i++) {
        Frame *f = &s.fr[i];
        fd[i] = f->top && f->rend ? (double)(int64_t)(f->rend - f->top) * g_tsp / 1e3 : NAN;
        fs[i] = f->top ? (map_ns(&m, f->top) - f->tsub) / 1e3 : NAN;
        fgap[i] = i && f->top && s.fr[i - 1].bot ? (double)(int64_t)(f->top - s.fr[i - 1].bot) * g_tsp / 1e3 : NAN;
        ftail[i] = f->bot && f->rend ? (double)(int64_t)(f->bot - f->rend) * g_tsp / 1e3 : NAN;
    }
    if (s.draw) {
        dist(s.name, "frame_pass_gpu_us", fd, nf);
        dist(s.name, "frame_submit_to_top_us", fs, nf);
        dist(s.name, "frame_gap_us(previous_end_to_top)", fgap, nf);
        if (!strcmp(s.cq, "in")) {
            dist(s.name, "inframe_list_after_pass_us(pass_end_to_list_end)", ftail, nf);
            double g0[4096], gk[4096 * 19];
            int n0 = 0, nk = 0;
            for (int i = 0; i < nf && n0 < 4096; i++) {
                Frame *f = &s.fr[i];
                if (!f->ctop[0]) continue;
                g0[n0++] = (double)(int64_t)(f->ctop[0] - f->rend) * g_tsp / 1e3;
                for (int k = 1; k < 20; k++) gk[nk++] = (double)(int64_t)(f->ctop[k] - f->ctop[k - 1]) * g_tsp / 1e3;
            }
            dist(s.name, "inframe_pass_end_to_first_dispatch_top_us", g0, n0);
            dist(s.name, "inframe_top_to_next_top_us(dispatch_period)", gk, nk);
        }
    }
    // classify each probe: was a frame on the GPU when it was submitted (from the frames' TOP and BOTTOM)?
    int fi = 0;
    for (int i = 0; i < nt; i++) {
        t[i].inframe = 0;
        while (fi < nf && s.fr[fi].bot && map_ns(&m, s.fr[fi].bot) < t[i].tsub) fi++;
        if (fi < nf && s.fr[fi].top && map_ns(&m, s.fr[fi].top) <= t[i].tsub && s.fr[fi].bot &&
            map_ns(&m, s.fr[fi].bot) >= t[i].tsub)
            t[i].inframe = 1;
    }
    // did the probe's shader run while a pass was running (REALTIME entry inside a frame's [TOP, BOTTOM])?
    int during = 0, during_n = 0;
    for (int i = 0, f2 = 0; i < nt; i++) {
        while (f2 < nf && s.fr[f2].bot && s.fr[f2].bot < t[i].r0) f2++;
        if (t[i].inframe) {
            during_n++;
            during += f2 < nf && s.fr[f2].top <= t[i].r0 && t[i].r0 <= s.fr[f2].rend;
        }
    }
    if (cq >= 0) {
        tiny_report(s.name, t, 0, nt, cq, &m, -1);
        if (s.draw) {
            tiny_report(s.name, t, 0, nt, cq, &m, 1);
            tiny_report(s.name, t, 0, nt, cq, &m, 0);
            printf("RESULT %s probes_submitted_during_a_pass=%d of %d; of those, shader entry inside the pass: %d\n",
                   s.name, during_n, nt, during);
        }
        tiny_csv(s.name, t, nt, &m, s.t0);
    }
    // frames CSV
    if (g_out[0] && s.draw) {
        char path[900], nm[160];
        snprintf(nm, sizeof nm, "frames-%s.csv", s.name);
        out_path(path, sizeof path, nm);
        FILE *fo = fopen(path, "w");
        if (fo) {
            fprintf(fo, "arm,frame,t_submit_us,submit_to_top_us,pass_gpu_us,gap_from_previous_us,after_pass_us\n");
            for (int i = 0; i < nf; i++)
                fprintf(fo, "%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", s.name, i, (s.fr[i].tsub - s.t0) / 1e3, fs[i], fd[i],
                        fgap[i], ftail[i]);
            fclose(fo);
        }
    }
    state_line(s.name, s.t0, s.t1);
    int bad = 0;
    for (int i = 0; i < nt; i++) bad += !t[i].ok;
    if (cq >= 0) verdict("probe_results", bad == 0, "%s: %d of %d probe dispatches left wrong values", s.name, bad, nt);
    if (s.draw) {
        // Control: the draw load ran as planned (the number of frames). The pass time follows the clock, which this
        // load changes, so it is reported, not judged.
        const double want = s.fps > 0 ? s.dur * s.fps : 0;
        printf("INFO %s pass GPU time median %.3f ms (sized for %.2f ms at the clock of the sizing passes)%s; measured "
               "busy share %.3f%s\n", s.name, pctl(fd, nf, .5) / 1e3, s.frame_ms, s.hot85 ? "; STOPPED at 85 C" : "",
               s.busy_ns / (s.t1 - s.t0), s.duty_cut ? " (ended at 0.1 s: measured busy share above 0.40)" : "");
        verdict("frames", s.hot85 ? 0 : s.fps > 0 ? fabs(nf - want) <= 0.1 * want + 1 : nf >= 5,
                "%s: %d frames%s (expected %.0f +-10 %% at %.0f per second; back to back: at least 5)", s.name, nf,
                s.hot85 ? ", stopped at 85 C" : "", want, s.fps);
    }
    free(t), free(s.fr), free(fd), free(fs), free(fgap), free(ftail);
    if (!strcmp(s.cq, "in")) {
        vkFreeDescriptorSets(g_dev, g_dpool, 1, &s.ds);
        rmbuf(&s.rec), rmbuf(&s.chk), rmbuf(&s.err);
        vkDestroyPipeline(g_dev, s.chain.p, NULL);
    }
    return g_hot_stop ? 1 : 0;
}

int mode_share(int argc, char **argv) {
    draw_setup(1);
    Probe pr;
    probe_init(&pr);
    // The pass with no ALU writes 2560 x 1440 x 8 bytes = 29.5 MB; uncompressed, the memory bus (lane 2: 640-644 GB/s
    // at most) needs at least 45.8 us. Reported, not judged: the development run wrote it faster than that (E7).
    double t0 = draw_time_us(0, 20);
    printf("INFO draw pass without ALU: %.1f us for 29.5 MB = %.0f GB/s (bus at most 644 GB/s, 45.8 us)\n", t0,
           29.4912e6 / (t0 * 1e3));
    // No sizing of a pixel-shader FMA load: such arms are refused after review.
    g_iters_per_ms = 0;
    int rc = 0;
    for (int i = 0; i < argc && !rc; i++) rc = run_share(argv[i], &pr);
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// E7: the pass without ALU (iters 0: no FMA step runs) into targets of several sizes, interleaved in rounds. Each pass
// is one submission, waited for, with 1 ms of idle after it (low duty). Is the 29.5 MB target's write rate above the
// memory bus because the target fits the 64 MB memory-side cache, or because of colour compression? A target larger
// than that cache separates the two readings only partly (compression would still apply); the size is the one change.

typedef struct {
    uint32_t w, h;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    VkPipeline pipe;
    VkDeviceSize bytes;
} Target;

static void mk_target(Target *t) {
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ii.extent = (VkExtent3D){t->w, t->h, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VK_CHECK(vkCreateImage(g_dev, &ii, NULL, &t->img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, t->img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type_dl(mr.memoryTypeBits);
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &t->mem));
    VK_CHECK(vkBindImageMemory(g_dev, t->img, t->mem, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t->img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ii.format;
    vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(g_dev, &vi, NULL, &t->view));
    char label[64];
    snprintf(label, sizeof label, "gdraw_%ux%u", t->w, t->h);
    t->pipe = draw_pipe(t->w, t->h, label);
    target_layout(t->img);
    t->bytes = (VkDeviceSize)t->w * t->h * 8;
    printf("TARGET %ux%u RGBA16F: %.1f MB of pixels, allocation %.1f MB\n", t->w, t->h, t->bytes / 1e6, mr.size / 1e6);
}

static double rt_pass_us(Target *t, VkCommandBuffer cb, VkQueryPool qp) {
    cb_begin(cb);
    vkCmdResetQueryPool(cb, qp, 0, 2);
    ts_top(cb, qp, 0);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = t->view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = (VkRect2D){{0, 0}, {t->w, t->h}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, t->pipe);
    const uint32_t iters = 0;
    vkCmdPushConstants(cb, g_dpl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &iters);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    ts_bot(cb, qp, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    run_once(&QU[0], cb);
    return (double)(qget(qp, 1) - qget(qp, 0)) * g_tsp / 1e3;
}

int mode_rt(int argc, char **argv) {
    const char *spec = argc > 0 ? argv[0] : "";
    char name[64], sizes[256];
    kvs(spec, "name", "rt", name, sizeof name);
    kvs(spec, "sizes", "2560x1440:4096x1536:4096x4096:8192x4096", sizes, sizeof sizes);
    const int rounds = (int)kvd(spec, "rounds", 5), passes = (int)kvd(spec, "passes", 20);
    draw_setup(1);
    Target t[8];
    int nt = 0;
    for (char *s = sizes; *s && nt < 8;) {
        memset(&t[nt], 0, sizeof t[nt]);
        t[nt].w = (uint32_t)strtoul(s, &s, 10);
        if (*s == 'x') s++;
        t[nt].h = (uint32_t)strtoul(s, &s, 10);
        if (*s == ':') s++;
        if (t[nt].w && t[nt].h) mk_target(&t[nt++]);
    }
    VkCommandBuffer cb = cb_alloc(&QU[0]);
    VkQueryPool qp = mkqp(2);
    double *us = calloc((size_t)nt * rounds * passes, sizeof(double));
    char path[900], nm[160];
    snprintf(nm, sizeof nm, "rt-%s.csv", name);
    out_path(path, sizeof path, nm);
    FILE *f = g_out[0] ? fopen(path, "w") : NULL;
    if (f) fprintf(f, "round,target,w,h,bytes,pass,gpu_us,sclk_MHz\n");
    const double c0 = mono_ns();
    for (int r = 0; r < rounds && !g_hot_stop; r++)
        for (int k = 0; k < nt; k++) {
            const int ti = (k + r) % nt;  // the order rotates each round
            for (int p = 0; p < passes && !g_hot_stop; p++) {
                const double v = rt_pass_us(&t[ti], cb, qp);
                us[((size_t)ti * rounds + r) * passes + p] = v;
                const Smp *sm = smp_at(mono_ns());
                if (f) fprintf(f, "%d,%d,%u,%u,%llu,%d,%.3f,%d\n", r, ti, t[ti].w, t[ti].h, (unsigned long long)t[ti].bytes,
                               p, v, sm ? sm->sclk : -1);
                until_ns(mono_ns() + 1e6);
            }
        }
    if (f) fclose(f);
    for (int k = 0; k < nt; k++) {
        char tag[96];
        snprintf(tag, sizeof tag, "%s-%ux%u", name, t[k].w, t[k].h);
        // the first pass of each round's block is reported apart (a first touch after the other target)
        double *v = us + (size_t)k * rounds * passes, rest[4096], first[64], rate[4096];
        int nr = 0, nf = 0;
        for (int r = 0; r < rounds; r++)
            for (int p = 0; p < passes; p++) {
                const double x = v[r * passes + p];
                if (!(x > 0)) continue;
                if (p == 0) first[nf++] = x;
                else rest[nr] = x, rate[nr++] = t[k].bytes / (x * 1e3);
            }
        dist(tag, "pass_gpu_us(passes_2+)", rest, nr);
        dist(tag, "pass_gpu_us(first_pass_of_each_block)", first, nf);
        dist(tag, "write_GB_per_s(passes_2+)", rate, nr);
        printf("RESULT %s bytes=%.1fMB median_us=%.2f median_GB_per_s=%.0f (memory bus at most 644 GB/s; 64 MB "
               "memory-side cache)\n", tag, t[k].bytes / 1e6, pctl(rest, nr, .5), pctl(rate, nr, .5));
    }
    state_line(name, c0, mono_ns());
    free(us);
    return g_hot_stop ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// E6: is the compute queue's late completion a host-side wait effect? The same one-group probes on the compute-only
// queue, alone, in four arms interleaved in rounds (the order rotates): blocking wait (vkWaitForFences) or
// spin-polling (vkGetFenceStatus in a loop), the submitting thread pinned to one CPU or not. The CPU the thread runs
// on is recorded at submission and at wake-up.

static void probe_one_wait(Probe *pr, Tiny *t, int spin, int *cpu0, int *cpu1) {
    const int q = 1, j = pr->j[q];
    pr->j[q] = (j + 1) % RING;
    volatile uint32_t *res = (uint32_t *)pr->res.map + j * 64;
    for (int i = 0; i < 64; i++) res[i] = 0xFFFFFFFFu;
    t->q = q;
    *cpu0 = sched_getcpu();
    t->tsub = mono_ns();
    submit(&QU[q], pr->cb[q][j], pr->f[q]);
    if (spin) {
        VkResult r;
        while ((r = vkGetFenceStatus(g_dev, pr->f[q])) == VK_NOT_READY) {
        }
        VK_CHECK(r);
        t->tdone = mono_ns();
        VK_CHECK(vkResetFences(g_dev, 1, &pr->f[q]));
    } else {
        VK_CHECK(vkWaitForFences(g_dev, 1, &pr->f[q], VK_TRUE, 10ull * 1000000000ull));
        t->tdone = mono_ns();
        VK_CHECK(vkResetFences(g_dev, 1, &pr->f[q]));
    }
    *cpu1 = sched_getcpu();
    t->top = qget(pr->qp[q], 2 * j);
    t->bot = qget(pr->qp[q], 2 * j + 1);
    volatile uint32_t *rr = (uint32_t *)pr->rec.map + 4 * j;
    t->r0 = rr[0] | (uint64_t)rr[1] << 32;
    t->r1 = rr[2] | (uint64_t)rr[3] << 32;
    int ok = 1;
    for (int i = 0; i < 64; i++) ok &= res[i] == 2000u + (uint32_t)i;
    t->ok = ok;
}

int mode_slow(int argc, char **argv) {
    const char *spec = argc > 0 ? argv[0] : "";
    char name[64];
    kvs(spec, "name", "slow", name, sizeof name);
    const int n = (int)kvd(spec, "n", 300), rounds = (int)kvd(spec, "rounds", 6), cpu = (int)kvd(spec, "cpu", 3);
    Probe pr;
    probe_init(&pr);
    cpu_set_t all, one;
    pthread_getaffinity_np(pthread_self(), sizeof all, &all);
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    static const char *an[4] = {"block_unpinned", "spin_unpinned", "block_pinned", "spin_pinned"};
    Tiny *t = calloc((size_t)4 * rounds * n, sizeof(Tiny)), tw;
    int *c0 = calloc((size_t)4 * rounds * n, sizeof(int)), *c1 = calloc((size_t)4 * rounds * n, sizeof(int)), d0, d1;
    Map m;
    map_begin(&m);
    const double t0 = mono_ns();
    for (int r = 0; r < rounds && !g_hot_stop; r++)
        for (int k = 0; k < 4; k++) {
            const int a = (k + r) % 4, spin = a & 1, pin = a >> 1;
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), pin ? &one : &all);
            for (int i = 0; i < 20; i++) probe_one_wait(&pr, &tw, spin, &d0, &d1);  // settle after a switch
            for (int i = 0; i < n; i++) {
                const size_t x = ((size_t)a * rounds + r) * n + i;
                probe_one_wait(&pr, &t[x], spin, &c0[x], &c1[x]);
            }
        }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &all);
    map_end(&m);
    map_report(name, &m);
    char path[900], nm[160];
    snprintf(nm, sizeof nm, "slow-%s.csv", name);
    out_path(path, sizeof path, nm);
    FILE *f = g_out[0] ? fopen(path, "w") : NULL;
    if (f)
        fprintf(f, "arm,round,i,t_submit_us,wall_us,submit_to_first_ts_us,first_ts_to_entry_us,entry_to_exit_us,"
                   "exit_to_bottom_us,bottom_to_host_us,cpu_submit,cpu_wake,ok\n");
    int bad = 0;
    for (int a = 0; a < 4; a++) {
        double *w = calloc((size_t)rounds * n, 8), *bh = calloc((size_t)rounds * n, 8);
        int k = 0, moved = 0;
        for (int r = 0; r < rounds; r++) {
            double wr[4096];
            int kr = 0;
            for (int i = 0; i < n; i++) {
                const size_t x = ((size_t)a * rounds + r) * n + i;
                const Tiny *s = &t[x];
                if (!s->tsub) continue;
                w[k] = (s->tdone - s->tsub) / 1e3;
                bh[k] = (s->tdone - map_ns(&m, s->bot)) / 1e3;
                wr[kr++] = w[k];
                moved += c0[x] != c1[x];
                bad += !s->ok;
                if (f)
                    fprintf(f, "%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d\n", an[a], r, i, (s->tsub - t0) / 1e3,
                            w[k], (map_ns(&m, s->top) - s->tsub) / 1e3, (double)(int64_t)(s->r0 - s->top) * g_tsp / 1e3,
                            (double)(int64_t)(s->r1 - s->r0) * g_tsp / 1e3, (double)(int64_t)(s->bot - s->r1) * g_tsp / 1e3,
                            bh[k], c0[x], c1[x], s->ok);
                k++;
            }
            printf("ROUND %s %s round=%d n=%d share_above_60us=%.4f wall_median_us=%.2f\n", name, an[a], r, kr,
                   frac_above(wr, kr, 60), pctl(wr, kr, .5));
        }
        char tag[128];
        snprintf(tag, sizeof tag, "%s-%s", name, an[a]);
        dist(tag, "wall_us(submit_to_host_sees_fence)", w, k);
        dist(tag, "bottom_ts_to_host_us", bh, k);
        double sl[16384], fa[16384];
        int ns = 0, nf = 0;
        for (int i = 0; i < k && i < 16384; i++) {
            if (w[i] > 60) sl[ns++] = bh[i];
            else fa[nf++] = bh[i];
        }
        printf("RESULT %s n=%d share_above_60us=%.4f bottom_ts_to_host_us median: slow %.2f (n %d), fast %.2f (n %d); "
               "probes whose thread changed CPU between submit and wake: %d\n",
               tag, k, frac_above(w, k, 60), pctl(sl, ns, .5), ns, pctl(fa, nf, .5), nf, moved);
        free(w), free(bh);
    }
    if (f) fclose(f);
    state_line(name, t0, mono_ns());
    verdict("probe_results", bad == 0, "%s: %d probe dispatches left wrong values", name, bad);
    free(t), free(c0), free(c1);
    return 0;
}
