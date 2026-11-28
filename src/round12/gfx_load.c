#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define VENDOR 0x1002u
#define DEVICE 0x7550u
#define DW 2560
#define DH 1440
#define RING 4

#define VK_CHECK(x)                                                                                 \
    do {                                                                                            \
        VkResult r_ = (x);                                                                          \
        if (r_ != VK_SUCCESS) {                                                                     \
            fprintf(stderr, "VK_CHECK failed %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);  \
            exit(3);                                                                                \
        }                                                                                           \
    } while (0)

static double mono_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static char g_hw[1024];
static char g_hw_file[1100];
static int g_thermal_stop = 0;

static void find_hwmon(const VkPhysicalDevicePCIBusInfoPropertiesEXT *pci) {
    const char *env_path = getenv("HOTSPOT_PATH");
    if (env_path != NULL) {
        snprintf(g_hw_file, sizeof g_hw_file, "%s", env_path);
        return;
    }
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x", pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    DIR *d = opendir("/sys/class/drm");
    struct dirent *e;
    char drm[512] = "";
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char p[300], real[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, real)) continue;
        const char *base = strrchr(real, '/');
        if (base && !strcmp(base + 1, want)) snprintf(drm, sizeof drm, "%s", p);
    }
    if (d) closedir(d);
    if (!drm[0]) return;
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", drm);
    d = opendir(hp);
    while (d && (e = readdir(d))) {
        if (!strncmp(e->d_name, "hwmon", 5)) snprintf(g_hw, sizeof g_hw, "%s/%s", hp, e->d_name);
    }
    if (d) closedir(d);
    if (g_hw[0]) snprintf(g_hw_file, sizeof g_hw_file, "%s/temp2_input", g_hw);
}

static double hotspot_c(void) {
    if (!g_hw_file[0]) return -1.0;
    FILE *f = fopen(g_hw_file, "r");
    if (!f) return -1.0;
    char buf[64];
    char *s = fgets(buf, sizeof buf, f);
    fclose(f);
    if (!s) return -1.0;
    char *end = NULL;
    double v = strtod(buf, &end);
    if (end == buf || !isfinite(v) || v <= 0.0) return -1.0;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') return -1.0;
    return v * 1e-3;
}

static char *slurp(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); exit(2); }
    fclose(f);
    *sz = (size_t)n;
    return b;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double pctl(double *v, int n, double p) {
    if (n <= 0) return NAN;
    double *c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    int idx = (int)llround(p * (n - 1));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    double r = c[idx];
    free(c);
    return r;
}

typedef struct {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue q;
    uint32_t qf;
    double tsp_ns;
    VkCommandPool pool;
    VkImage img;
    VkDeviceMemory imem;
    VkImageView iview;
    VkPipelineLayout pl;
    VkPipeline pipe;
} Gfx;

static Gfx G;

static void gfx_init(const char *spv_dir) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "gfx_load", 1, "rdna4", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VK_CHECK(vkCreateInstance(&ici, NULL, &G.inst));

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    VkPhysicalDevice *devs = calloc(nd, sizeof *devs);
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == VENDOR && p.deviceID == DEVICE) { G.pd = devs[i]; break; }
    }
    free(devs);
    if (!G.pd) { fprintf(stderr, "No RX 9070 XT found\n"); exit(1); }

    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT, NULL, 0,0,0,0};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pci, {0}};
    vkGetPhysicalDeviceProperties2(G.pd, &p2);
    G.tsp_ns = p2.properties.limits.timestampPeriod;
    find_hwmon(&pci);
    // Fail closed before the first submission too (Kimi R12 review round 2).
    double h0 = hotspot_c();
    if (h0 < 0 || h0 >= 92.0) {
        fprintf(stderr, "THERMAL STOP %.1f C before init%s\n", h0, h0 < 0 ? " (sensor unreadable: fail closed)" : "");
        exit(4);
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, qp);
    G.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits) { G.qf = i; break; }
    }
    free(qp);
    if (G.qf == UINT32_MAX) { fprintf(stderr, "No graphics queue\n"); exit(1); }

    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, NULL};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13, {0}};
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, G.qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci, 0, NULL, 0, NULL, NULL};
    VK_CHECK(vkCreateDevice(G.pd, &dci, NULL, &G.dev));
    vkGetDeviceQueue(G.dev, G.qf, 0, &G.q);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, G.qf};
    VK_CHECK(vkCreateCommandPool(G.dev, &cpi, NULL, &G.pool));

    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT,
                            {DW, DH, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHECK(vkCreateImage(G.dev, &ii, NULL, &G.img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(G.dev, G.img, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(G.pd, &mp);
    uint32_t mtype = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mtype = i; break; }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mtype};
    VK_CHECK(vkAllocateMemory(G.dev, &ai, NULL, &G.imem));
    VK_CHECK(vkBindImageMemory(G.dev, G.img, G.imem, 0));

    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, G.img, VK_IMAGE_VIEW_TYPE_2D,
                                VK_FORMAT_R16G16B16A16_SFLOAT, {0,0,0,0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VK_CHECK(vkCreateImageView(G.dev, &vi, NULL, &G.iview));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr};
    VK_CHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, &G.pl));

    char vp[600], fp[600];
    snprintf(vp, sizeof vp, "%s/gdraw_vert.spv", spv_dir);
    snprintf(fp, sizeof fp, "%s/gdraw_frag.spv", spv_dir);
    size_t vsz = 0, fsz = 0;
    char *vcode = slurp(vp, &vsz), *fcode = slurp(fp, &fsz);
    VkShaderModule vs, fs;
    VkShaderModuleCreateInfo smv = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, vsz, (const uint32_t *)vcode};
    VkShaderModuleCreateInfo smf = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, fsz, (const uint32_t *)fcode};
    VK_CHECK(vkCreateShaderModule(G.dev, &smv, NULL, &vs));
    VK_CHECK(vkCreateShaderModule(G.dev, &smf, NULL, &fs));
    free(vcode); free(fcode);

    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL}
    };
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport vpt = {0, 0, (float)DW, (float)DH, 0, 1};
    VkRect2D sc = {{0, 0}, {DW, DH}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vpt, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0,0,0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, NULL, VK_FALSE, VK_FALSE};
    VkPipelineColorBlendAttachmentState cba = {VK_FALSE, 0,0,0,0,0,0, 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba, {0,0,0,0}};
    VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &fmt, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, st, &vin, &ia, NULL, &vps, &rs, &ms, NULL, &cb, NULL, G.pl, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VK_CHECK(vkCreateGraphicsPipelines(G.dev, VK_NULL_HANDLE, 1, &gi, NULL, &G.pipe));
    vkDestroyShaderModule(G.dev, vs, NULL);
    vkDestroyShaderModule(G.dev, fs, NULL);

    // Transition image once
    VkCommandBuffer c;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, G.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHECK(vkAllocateCommandBuffers(G.dev, &cai, &c));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    VK_CHECK(vkBeginCommandBuffer(c, &bi));
    VkImageMemoryBarrier2 ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, G.img,
                                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &ib};
    vkCmdPipelineBarrier2(c, &di);
    VK_CHECK(vkEndCommandBuffer(c));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &c, 0, NULL};
    VK_CHECK(vkQueueSubmit(G.q, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(G.q));
    vkFreeCommandBuffers(G.dev, G.pool, 1, &c);
}

static void record_frame(VkCommandBuffer cb, VkQueryPool qp, uint32_t iters) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, NULL};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0, 2);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &mb, 0, NULL, 0, NULL};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, qp, 0);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, G.iview,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE,
                                    VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {DW, DH}}, 1, 0, 1, &ca, NULL, NULL};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, G.pipe);
    vkCmdPushConstants(cb, G.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &iters);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, qp, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
}

static double measure_one(uint32_t iters, int warm, int n, double *out_p95, double *out_max) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, G.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHECK(vkAllocateCommandBuffers(G.dev, &cai, &cb));
    VkQueryPool qp;
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VK_CHECK(vkCreateQueryPool(G.dev, &qpi, NULL, &qp));
    VkFence f;
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VK_CHECK(vkCreateFence(G.dev, &fci, NULL, &f));
    double vals[256];
    int m = 0;
    for (int i = 0; i < warm + n && m < 256; i++) {
        double hot = hotspot_c();
        if (hot < 0 || hot >= 92.0) {
            fprintf(stderr, "THERMAL STOP %.1f C%s\n", hot, hot < 0 ? " (sensor unreadable: fail closed)" : "");
            g_thermal_stop = 1;
            break;
        }
        record_frame(cb, qp, iters);
        vkResetFences(G.dev, 1, &f);
        double ts = mono_ms();
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
        VK_CHECK(vkQueueSubmit(G.q, 1, &si, f));
        VK_CHECK(vkWaitForFences(G.dev, 1, &f, VK_TRUE, UINT64_MAX));
        double te = mono_ms();
        uint64_t qv[2] = {0, 0};
        VK_CHECK(vkGetQueryPoolResults(G.dev, qp, 0, 2, sizeof qv, qv, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double gpu_ms = (double)(qv[1] - qv[0]) * G.tsp_ns * 1e-6;
        if (i >= warm) {
            vals[m++] = gpu_ms;
            printf("GFX_FRAME mode=alone idx=%d t_start_ms=%.3f t_end_ms=%.3f gpu_ms=%.3f wall_ms=%.3f hot_c=%.1f\n",
                   m - 1, ts, te, gpu_ms, te - ts, hotspot_c());
        }
    }
    vkDestroyFence(G.dev, f, NULL);
    vkDestroyQueryPool(G.dev, qp, NULL);
    vkFreeCommandBuffers(G.dev, G.pool, 1, &cb);
    if (out_p95) *out_p95 = pctl(vals, m, 0.95);
    if (out_max) *out_max = pctl(vals, m, 1.0);
    return pctl(vals, m, 0.5);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gfx_load <spv_dir> cal <target_ms> | alone <iters> <n> | serve <iters> <ready> <stop> <max_frames> <in_flight>\n");
        return 2;
    }
    gfx_init(argv[1]);
    const char *mode = argv[2];
    if (!strcmp(mode, "cal")) {
        double target = atof(argv[3]);
        uint32_t iters = 1500;
        double t = measure_one(iters, 3, 5, NULL, NULL);
        if (g_thermal_stop) return 4;
        iters = (uint32_t)(iters * (target / (t > 0.1 ? t : 1.0)));
        if (iters < 10) iters = 10;
        double p95 = 0, mx = 0;
        double t2 = measure_one(iters, 2, 5, &p95, &mx);
        printf("GFX_CAL target_ms=%.2f iters=%u p50_ms=%.3f p95_ms=%.3f max_ms=%.3f hot_c=%.1f\n",
               target, iters, t2, p95, mx, hotspot_c());
        return g_thermal_stop ? 4 : 0;
    }
    if (!strcmp(mode, "alone")) {
        uint32_t iters = (uint32_t)atoi(argv[3]);
        int n = atoi(argv[4]);
        double p95 = 0, mx = 0;
        double p50 = measure_one(iters, 3, n, &p95, &mx);
        printf("GFX_SUMMARY mode=alone n=%d iters=%u p50_ms=%.3f p95_ms=%.3f max_ms=%.3f hot_c=%.1f\n",
               n, iters, p50, p95, mx, hotspot_c());
        return g_thermal_stop ? 4 : 0;
    }
    if (!strcmp(mode, "serve")) {
        uint32_t iters = (uint32_t)atoi(argv[3]);
        const char *ready_path = argv[4];
        const char *stop_path = argv[5];
        int max_frames = atoi(argv[6]);
        int in_flight = atoi(argv[7]);
        double pace_ms = (argc > 8) ? atof(argv[8]) : 0.0;
        if (in_flight < 1) in_flight = 1;
        if (in_flight > RING) in_flight = RING;

        VkCommandBuffer cbs[RING];
        VkQueryPool qps[RING];
        VkFence fns[RING];
        double tsub[RING] = {0};
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, G.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RING};
        VK_CHECK(vkAllocateCommandBuffers(G.dev, &cai, cbs));
        for (int i = 0; i < RING; i++) {
            VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
            VK_CHECK(vkCreateQueryPool(G.dev, &qpi, NULL, &qps[i]));
            VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
            VK_CHECK(vkCreateFence(G.dev, &fci, NULL, &fns[i]));
            record_frame(cbs[i], qps[i], iters);
        }

        int submitted = 0, completed = 0, ready_sent = 0, thermal_stop = 0;
        double next_sub_ms = mono_ms();
        for (;;) {
            int stop_now = (access(stop_path, F_OK) == 0) || (submitted >= max_frames);
            double hot = hotspot_c();
            if (hot < 0 || hot >= 92.0) {
                fprintf(stderr, "GFX_SERVE THERMAL STOP %.1f C at frame %d%s\n", hot, completed, hot < 0 ? " (sensor unreadable: fail closed)" : "");
                thermal_stop = 1;
                stop_now = 1;
            }
            if (!stop_now && (submitted - completed) < in_flight) {
                if (pace_ms > 0.0 && submitted >= 3) {
                    double now = mono_ms();
                    if (now < next_sub_ms) usleep((useconds_t)((next_sub_ms - now) * 1000.0));
                    next_sub_ms = mono_ms() + pace_ms;
                }
                int slot = submitted % RING;
                vkResetFences(G.dev, 1, &fns[slot]);
                tsub[slot] = mono_ms();
                VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cbs[slot], 0, NULL};
                VK_CHECK(vkQueueSubmit(G.q, 1, &si, fns[slot]));
                submitted++;
            }
            if (completed < submitted && ((submitted - completed) >= in_flight || stop_now)) {
                int slot = completed % RING;
                VK_CHECK(vkWaitForFences(G.dev, 1, &fns[slot], VK_TRUE, UINT64_MAX));
                double td = mono_ms();
                uint64_t qv[2] = {0, 0};
                VK_CHECK(vkGetQueryPoolResults(G.dev, qps[slot], 0, 2, sizeof qv, qv, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                double gpu_ms = (double)(qv[1] - qv[0]) * G.tsp_ns * 1e-6;
                printf("GFX_FRAME mode=serve idx=%d t_start_ms=%.3f t_end_ms=%.3f gpu_ms=%.3f wall_ms=%.3f hot_c=%.1f\n",
                       completed, tsub[slot], td, gpu_ms, td - tsub[slot], hotspot_c());
                fflush(stdout);
                completed++;
                if (!ready_sent && completed >= 3) {
                    FILE *rf = fopen(ready_path, "w");
                    if (rf) { fprintf(rf, "ready\n"); fclose(rf); }
                    ready_sent = 1;
                }
            }
            if (stop_now && completed >= submitted) break;
        }
        if (!ready_sent) {
            FILE *rf = fopen(ready_path, "w");
            if (rf) { fprintf(rf, "ready\n"); fclose(rf); }
        }
        for (int i = 0; i < RING; i++) {
            vkDestroyFence(G.dev, fns[i], NULL);
            vkDestroyQueryPool(G.dev, qps[i], NULL);
        }
        vkFreeCommandBuffers(G.dev, G.pool, RING, cbs);
        return thermal_stop ? 4 : 0;
    }
    fprintf(stderr, "Unknown mode %s\n", mode);
    return 2;
}
