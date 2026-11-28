// Round 11 item 3: the graphics fixed blocks of the RX 9070 XT through Vulkan (RADV), headless: pixel output (ROPs), texture filtering,
// rasteriser, surface compression. The graphics pipeline, image and barrier code follows src/vk/vkgame_share.c (dynamic rendering,
// synchronization2); no window or swapchain.
//
//   r11_gfx --plan [--quick]            the cells and the card time, no Vulkan
//   r11_gfx --cpu-checks                files, cell list and CPU reference only, no Vulkan, no card
//   r11_gfx --run [--out DIR] [--reps N] [--burst-ms MS] [--only TEXT[,TEXT]] [--quick]
//
// Every timed burst is one command buffer, one fenced wait; the burst length is calibrated to --burst-ms (default 100, never above 250: the
// plan's 0.25 s rule) and the GPU time comes from timestamps. Every cell runs its control first (a failed control means the cell is not timed):
//   fill   occlusion count over 3 passes == 3 x W x H x samples exactly; read-back of two 64 x 4 pixel regions against the CPU colour of the last draw
//   raster occlusion count == triangles x exact covered pixels (r11g_tri_cover); the culled variant must count 0
//   texture a constant texture sampled by the cell's filter returns the constant (channel by channel)
// Deliberate breaks (each must make its control FAIL): (1) the last draw scissored to half the target; (2) the last draw skipped; (3) a
// compression read-back checked against an image that never asked for the switch; (4) a nearest sampler checked against the bilinear
// analytic values; (5) the culled pipeline's count checked against the visible expectation.
// Compression arm: VK_EXT_image_compression_control. An image created with VK_IMAGE_COMPRESSION_DISABLED_EXT is read back through
// VkImageCompressionPropertiesEXT; the switch counts as available only when the disabled image reads back disabled and an ordinary image does not.
// RADV's GFX8-GFX11.7 DCC debug variables are NOT used (this card is GFX12). Without the extension or without a visible change the
// arm reports "no switch available" and the compressible-against-random comparison stands alone.
// Exit: 0 all controls pass and breaks failed as required; 1 a control or break failed; 2 usage; 3 device or extension missing;
// 4 hotspot at the stop temperature; 5 a submit did not finish in time (device probably hung; nothing is retried); 7 hotspot stayed hot.
#define _GNU_SOURCE
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "r11_gfx_ref.h"

// ---- the cell list (no Vulkan needed) ---------------------------------------------------------------------------------------------
enum { C_FILL, C_RAST, C_TEX, C_COMP };
enum { RV_CM0, RV_DEPTH, RV_CULL };
enum { TF_FETCH, TF_NEAREST, TF_BILINEAR, TF_TRILINEAR, TF_ANISO4, TF_ANISO16, TF_N };
static const char * const tf_name[TF_N] = {"texelFetch", "nearest", "bilinear", "trilinear", "aniso4", "aniso16"};
typedef struct {
    char name[96];
    int kind;
    int fmt;          // fill: R11G_*; texture: R11T_*
    uint32_t mib;     // fill, compression: target size
    int blend, samples;
    uint32_t a256;    // raster: leg in 1/256 pixel
    int var;          // raster: RV_*; texture: TF_*; compression: data (0 constant, 1 random)
    int cc;           // compression: 1 = compression disabled requested
    uint32_t size;    // texture edge
} Cell;
#define MAXCELLS 400
static Cell g_cells[MAXCELLS];
static int g_ncells;
static const uint32_t k_mibs[4] = {1, 4, 32, 512};
static const uint32_t k_legs[7] = {128, 256, 512, 1024, 2048, 4096, 8192};
static const char * const rv_name[3] = {"colormask0", "depthonly", "culled"};

static Cell * add_cell(int kind) {
    Cell * c = &g_cells[g_ncells++];
    memset(c, 0, sizeof *c);
    c->kind = kind;
    return c;
}
static void build_cell_list(int quick) {
    g_ncells = 0;
    for (int f = 0; f < R11G_NFMT; ++f)
        for (int m = 0; m < 4; ++m) {
            uint32_t w, h;
            if (!r11g_target_dims((uint64_t) k_mibs[m] << 20, r11g_fmt_bpp[f], 16384, &w, &h)) continue;  // R8 at 512 MiB exceeds a 16384 image
            Cell * c = add_cell(C_FILL);
            c->fmt = f, c->mib = k_mibs[m], c->samples = 1;
            snprintf(c->name, sizeof c->name, "fill_%s_%uMiB", r11g_fmt_name[f], c->mib);
        }
    static const int bf[2] = {R11G_RGBA8, R11G_RGBA16F};
    static const uint32_t bm[2] = {4, 512};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            Cell * c = add_cell(C_FILL);
            c->fmt = bf[i], c->mib = bm[j], c->blend = 1, c->samples = 1;
            snprintf(c->name, sizeof c->name, "fill_%s_blend_%uMiB", r11g_fmt_name[c->fmt], c->mib);
        }
    static const uint32_t mm[2] = {4, 32};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            Cell * c = add_cell(C_FILL);
            c->fmt = bf[i], c->mib = mm[j], c->samples = 4;
            snprintf(c->name, sizeof c->name, "fill_%s_ms4_%uMiB", r11g_fmt_name[c->fmt], c->mib);
        }
    for (int v = 0; v < 3; ++v)
        for (int l = 0; l < 7; ++l) {
            Cell * c = add_cell(C_RAST);
            c->a256 = k_legs[l], c->var = v;
            snprintf(c->name, sizeof c->name, "raster_leg%u_%s", c->a256, rv_name[v]);
        }
    static const uint32_t ts[2] = {256, 8192};
    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < R11T_NFMT; ++f)
            for (int t = 0; t < TF_N; ++t) {
                if (quick && ts[s] == 8192 && (f == R11T_RGBA32F || f == R11T_RGBA16F)) continue;
                Cell * c = add_cell(C_TEX);
                c->fmt = f, c->size = ts[s], c->var = t;
                snprintf(c->name, sizeof c->name, "tex_%s_%u_%s", r11t_name[f], c->size, tf_name[t]);
            }
    static const uint32_t cm[2] = {32, 512};
    for (int m = 0; m < 2; ++m)
        for (int d = 0; d < 2; ++d)
            for (int cc = 0; cc < 2; ++cc) {
                Cell * c = add_cell(C_COMP);
                c->mib = cm[m], c->var = d, c->cc = cc, c->fmt = R11G_RGBA8, c->samples = 1;
                snprintf(c->name, sizeof c->name, "comp_%uMiB_%s_%s", c->mib, d ? "random" : "solid", cc ? "disabled" : "default");
            }
}

// ---- Vulkan -------------------------------------------------------------------------------------------------------------------------
#define VK_CHECK(x)                                                                                     \
    do {                                                                                                \
        VkResult r_ = (x);                                                                              \
        if (r_ != VK_SUCCESS) {                                                                         \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int) r_, __FILE__, __LINE__, #x);       \
            printf("VERDICT r11_gfx: FAIL Vulkan error %d (%s)%s\n", (int) r_, #x, r_ == VK_ERROR_DEVICE_LOST ? " DEVICE LOST" : ""); \
            exit(r_ == VK_ERROR_DEVICE_LOST ? 5 : 3);                                                   \
        }                                                                                               \
    } while (0)

static VkInstance g_inst;
static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_q;
static uint32_t g_qf;
static VkCommandPool g_pool;
static VkCommandBuffer g_cb;
static VkFence g_fence;
static VkQueryPool g_qp, g_oq;
static double g_ts_ns, g_hottest, g_max_aniso;
static int g_have_cc;
static VkPhysicalDeviceMemoryProperties g_mp;
static VkDescriptorSetLayout g_dsl;
static VkPipelineLayout g_cpl, g_gpl;  // compute (one set, 32 bytes of push constants), graphics (32 bytes, vertex + fragment)
static VkDescriptorPool g_dp;
static VkDescriptorSet g_ds;
static char g_drm[512], g_hwmon[1024], g_out[512] = "", g_spv[512] = "";
static double g_target_ms = 100.0;
static int g_fail;
static int g_burst_over, g_spread_flagged, g_strict_spread;
static PFN_vkGetImageSubresourceLayout2EXT pGetSubLayout2;

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static int cmp_d(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
static double median(const double * v, int n) {
    double * c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    const double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}
static double read_num(const char * dir, const char * file, double scale) {
    char p[1300];
    snprintf(p, sizeof p, "%s/%s", dir, file);
    FILE * f = fopen(p, "r");
    if (!f) return -1;
    double v = -1;
    if (fscanf(f, "%lf", &v) != 1) v = -1;
    fclose(f);
    return v < 0 ? -1 : v * scale;
}
static double hotspot_c(void) { return g_hwmon[0] ? read_num(g_hwmon, "temp2_input", 1e-3) : -1; }

// Thermal pacing. On this card's zero-RPM fan curve the fan stays at 0 rpm for tens of seconds while back-to-back bursts heat the hotspot by
// about 1.3 C per second (31 C -> 96 C in 50 s in the first full run), so a gate per group is not enough. --cool-c C (default 72) sets the
// per-cell ceiling: a cell does not start, and a burst series does not continue, at or above C; the wait ends at C - 6. A group waits for C - 12
// (60 C by default). Waits are up to 120 s; the 95 C stop is unchanged and the guard in the runner is a second line behind it.
static double g_cool_c = 72.0;
static void thermal_wait(double target, const char * what) {
    const double t = hotspot_c();
    if (t < 0) { printf("VERDICT thermal: FAIL hotspot sensor not found; refusing whole-card work\n"); exit(1); }
    if (t > g_hottest) g_hottest = t;
    if (t >= 95.0) { printf("THERMAL hotspot %.0f C: stopping\n", t); exit(4); }
    if (t < target) return;
    const double t0 = now_ns();
    double cur = t;
    while ((cur = hotspot_c()) >= target && now_ns() - t0 < 120e9) { if (cur >= 95.0) exit(4); usleep(250000); }
    if (cur >= target) { printf("THERMAL hotspot still %.0f C after 120 s (%s): stopping\n", cur, what); exit(7); }
    printf("THERMAL paused %.1f s before %s: %.0f C -> %.0f C (target below %.0f C)\n", (now_ns() - t0) * 1e-9, what, t, cur, target);
}
// Before a group of cells.
static void thermal_gate(void) { thermal_wait(g_cool_c - 12.0, "group"); }
// Before every cell and between the bursts of a cell: only waits when the hotspot is at the ceiling, and then until 6 C below it.
static void thermal_cell(const char * what) {
    const double t = hotspot_c();
    if (t < 0) return;
    if (t > g_hottest) g_hottest = t;
    if (t >= g_cool_c) thermal_wait(g_cool_c - 6.0, what);
}

static int has_ext(const VkExtensionProperties * ex, uint32_t n, const char * name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(ex[i].extensionName, name)) return 1;
    return 0;
}
static void find_sysfs(const VkPhysicalDevicePCIBusInfoPropertiesEXT * pci) {
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x", pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    DIR * d = opendir("/sys/class/drm");
    struct dirent * e;
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char p[300], real[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, real)) continue;
        const char * base = strrchr(real, '/');
        if (base && !strcmp(base + 1, want)) snprintf(g_drm, sizeof g_drm, "%s", p);
    }
    if (d) closedir(d);
    if (!g_drm[0]) return;
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", g_drm);
    d = opendir(hp);
    while (d && (e = readdir(d))) if (!strncmp(e->d_name, "hwmon", 5)) snprintf(g_hwmon, sizeof g_hwmon, "%s/%s", hp, e->d_name);
    if (d) closedir(d);
}

static void vk_init(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdna4-r11-gfx";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &g_inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_inst, &n, NULL);
    VkPhysicalDevice * devs = calloc(n, sizeof *devs);
    vkEnumeratePhysicalDevices(g_inst, &n, devs);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == 0x1002 && p.deviceID == 0x7550) { g_pd = devs[i]; break; }
    }
    free(devs);
    if (!g_pd) { printf("VERDICT device: FAIL no Vulkan device 1002:7550 (RX 9070 XT); refusing to run elsewhere\n"); exit(3); }
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(g_pd, &p2);
    g_ts_ns = p2.properties.limits.timestampPeriod;
    g_max_aniso = p2.properties.limits.maxSamplerAnisotropy;
    find_sysfs(&pci);
    printf("DEVICE %s (%04x:%04x) driver %s %s; sysfs %s hwmon %s; max image dimension %u, max anisotropy %.0f\n", p2.properties.deviceName,
           p2.properties.vendorID, p2.properties.deviceID, drv.driverName, drv.driverInfo, g_drm[0] ? g_drm : "(not found)", g_hwmon[0] ? g_hwmon : "(not found)",
           p2.properties.limits.maxImageDimension2D, g_max_aniso);
    if (strcmp(drv.driverName, "radv")) { printf("VERDICT device: FAIL driver is %s, not radv\n", drv.driverName); exit(3); }
    vkGetPhysicalDeviceMemoryProperties(g_pd, &g_mp);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    VkQueueFamilyProperties * qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    g_qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qp[i].timestampValidBits) { g_qf = i; break; }
    free(qp);
    if (g_qf == UINT32_MAX) { printf("VERDICT device: FAIL no graphics+compute queue family with timestamps\n"); exit(3); }

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, NULL);
    VkExtensionProperties * ex = calloc(ne, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, ex);
    const char * names[3] = {"VK_KHR_pipeline_executable_properties", "VK_EXT_image_compression_control", NULL};
    if (!has_ext(ex, ne, names[0])) { printf("VERDICT extension: FAIL %s not supported by this device\n", names[0]); exit(3); }
    g_have_cc = has_ext(ex, ne, names[1]);
    free(ex);

    // what the device supports (queried, then only the needed features are enabled)
    VkPhysicalDeviceImageCompressionControlFeaturesEXT qcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR qexe = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, g_have_cc ? (void *) &qcc : NULL};
    VkPhysicalDeviceVulkan13Features q13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &qexe};
    VkPhysicalDeviceFeatures2 q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q13};
    vkGetPhysicalDeviceFeatures2(g_pd, &q2);
    if (!q13.dynamicRendering || !q13.synchronization2 || !qexe.pipelineExecutableInfo || !q2.features.samplerAnisotropy || !q2.features.textureCompressionBC || !q2.features.occlusionQueryPrecise) {
        printf("VERDICT extension: FAIL a needed feature is missing (dynamicRendering %d synchronization2 %d pipelineExecutableInfo %d samplerAnisotropy %d textureCompressionBC %d occlusionQueryPrecise %d)\n",
               q13.dynamicRendering, q13.synchronization2, qexe.pipelineExecutableInfo, q2.features.samplerAnisotropy, q2.features.textureCompressionBC, q2.features.occlusionQueryPrecise);
        exit(3);
    }
    if (g_have_cc && !qcc.imageCompressionControl) g_have_cc = 0;
    VkPhysicalDeviceImageCompressionControlFeaturesEXT fcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT};
    fcc.imageCompressionControl = VK_TRUE;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, g_have_cc ? (void *) &fcc : NULL};
    fexe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fexe};
    f13.dynamicRendering = f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    f2.features.samplerAnisotropy = f2.features.textureCompressionBC = f2.features.occlusionQueryPrecise = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_qf, qci.queueCount = 1, qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = g_have_cc ? 2 : 1, dci.ppEnabledExtensionNames = names;
    VK_CHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
    if (g_have_cc) {
        pGetSubLayout2 = (PFN_vkGetImageSubresourceLayout2EXT) vkGetDeviceProcAddr(g_dev, "vkGetImageSubresourceLayout2EXT");
        if (!pGetSubLayout2) pGetSubLayout2 = (PFN_vkGetImageSubresourceLayout2EXT) vkGetDeviceProcAddr(g_dev, "vkGetImageSubresourceLayout2");
        if (!pGetSubLayout2) g_have_cc = 0;
    }
    printf("COMPRESSION_CONTROL extension %s\n", g_have_cc ? "present and enabled" : "NOT available on this device");

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, cpi.queueFamilyIndex = g_qf;
    VK_CHECK(vkCreateCommandPool(g_dev, &cpi, NULL, &g_pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g_pool, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(g_dev, &fci, NULL, &g_fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP, qpi.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(g_dev, &qpi, NULL, &g_qp));
    qpi.queryType = VK_QUERY_TYPE_OCCLUSION, qpi.queryCount = 1;
    VK_CHECK(vkCreateQueryPool(g_dev, &qpi, NULL, &g_oq));

    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                         {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2, dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &g_dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1, pli.pSetLayouts = &g_dsl, pli.pushConstantRangeCount = 1, pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &g_cpl));
    VkPushConstantRange gpr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
    VkPipelineLayoutCreateInfo gli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    gli.pushConstantRangeCount = 1, gli.pPushConstantRanges = &gpr;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &gli, NULL, &g_gpl));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1, dpi.poolSizeCount = 2, dpi.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &g_dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_dp, dai.descriptorSetCount = 1, dai.pSetLayouts = &g_dsl;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &dai, &g_ds));
}

// ---- command buffer, buffers, images -----------------------------------------------------------------------------------------------
static void cb_begin(void) {
    VK_CHECK(vkResetCommandBuffer(g_cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_cb, &bi));
}
// One submit, one fenced wait of at most 20 s. A timeout ends the run (no retry on a hung card).
static void cb_submit(void) {
    VK_CHECK(vkEndCommandBuffer(g_cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &g_cb;
    VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
    VK_CHECK(vkQueueSubmit(g_q, 1, &si, g_fence));
    const VkResult r = vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, 20ull * 1000000000ull);
    if (r == VK_TIMEOUT) { printf("VERDICT r11_gfx: FAIL a submit did not finish in 20 s (device probably hung); stopping, nothing retried\n"); exit(5); }
    VK_CHECK(r);
}
static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = g_mp.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}
typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; void * map; } Buf;
static Buf buf_make(VkDeviceSize size, VkBufferUsageFlags usage, int host) {
    Buf b = {0};
    b.size = size ? size : 16;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b.size, bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(g_dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
    uint32_t t;
    if (host) {
        t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    } else {
        t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    }
    if (t == UINT32_MAX) { fprintf(stderr, "no memory type\n"); exit(3); }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = t;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
    if (host) VK_CHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}
static void buf_free(Buf * b) {
    if (!b->buf) return;
    if (b->map) vkUnmapMemory(g_dev, b->mem);
    vkDestroyBuffer(g_dev, b->buf, NULL);
    vkFreeMemory(g_dev, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

static const VkFormat k_vkfmt[R11G_NFMT] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16_SFLOAT,
                                            VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_D32_SFLOAT};
static const VkFormat k_vktex[R11T_NFMT] = {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK};

typedef struct { VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h, mips; VkFormat fmt; VkImageAspectFlags aspect; } Img;

static void bar_img(VkImage im, VkImageAspectFlags asp, uint32_t mips, VkImageLayout o, VkImageLayout n) {
    VkImageMemoryBarrier2 ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.srcStageMask = ib.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    ib.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    ib.oldLayout = o, ib.newLayout = n;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = im;
    ib.subresourceRange = (VkImageSubresourceRange){asp, 0, mips, 0, 1};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(g_cb, &di);
}
static void bar_mem(void) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(g_cb, &di);
}

// True when the device can make this image (format, usage, size, sample count); used to skip a cell cleanly instead of failing.
static int img_supported(VkFormat f, VkImageUsageFlags u, uint32_t w, uint32_t h, VkSampleCountFlagBits s) {
    VkImageFormatProperties p;
    if (vkGetPhysicalDeviceImageFormatProperties(g_pd, f, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u, 0, &p) != VK_SUCCESS) return 0;
    return w <= p.maxExtent.width && h <= p.maxExtent.height && (p.sampleCounts & s);
}
// cc: 0 no request, 1 VK_IMAGE_COMPRESSION_DISABLED_EXT requested (only when the extension is enabled).
static Img img_make(uint32_t w, uint32_t h, uint32_t mips, VkFormat fmt, VkSampleCountFlagBits s, VkImageUsageFlags usage, int cc) {
    Img m = {0};
    m.w = w, m.h = h, m.mips = mips, m.fmt = fmt;
    m.aspect = fmt == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCompressionControlEXT ctl = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT};
    ctl.flags = VK_IMAGE_COMPRESSION_DISABLED_EXT;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, (cc && g_have_cc) ? (const void *) &ctl : NULL};
    ii.imageType = VK_IMAGE_TYPE_2D, ii.format = fmt, ii.extent = (VkExtent3D){w, h, 1};
    ii.mipLevels = mips, ii.arrayLayers = 1, ii.samples = s, ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = usage, ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(g_dev, &ii, NULL, &m.img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, m.img, &mr);
    const uint32_t t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = t == UINT32_MAX ? mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0) : t;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &m.mem));
    VK_CHECK(vkBindImageMemory(g_dev, m.img, m.mem, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = m.img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt;
    vi.subresourceRange = (VkImageSubresourceRange){m.aspect, 0, mips, 0, 1};
    VK_CHECK(vkCreateImageView(g_dev, &vi, NULL, &m.view));
    return m;
}
static void img_free(Img * m) {
    if (!m->img) return;
    vkDestroyImageView(g_dev, m->view, NULL);
    vkDestroyImage(g_dev, m->img, NULL);
    vkFreeMemory(g_dev, m->mem, NULL);
    memset(m, 0, sizeof *m);
}
// What the driver says it applied: VkImageCompressionPropertiesEXT through vkGetImageSubresourceLayout2. 0xffffffff when not readable.
static uint32_t cc_readback(const Img * m) {
    if (!g_have_cc) return 0xffffffffu;
    VkImageCompressionPropertiesEXT pr = {VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT};
    VkSubresourceLayout2 lay = {VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2, &pr};
    VkImageSubresource2 sub = {VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2};
    sub.imageSubresource = (VkImageSubresource){m->aspect, 0, 0};
    pGetSubLayout2(g_dev, m->img, &sub, &lay);
    return (uint32_t) pr.imageCompressionFlags;
}

// ---- shaders and pipelines -----------------------------------------------------------------------------------------------------
static char * read_file(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * d = malloc(n + 1);
    if (fread(d, 1, n, f) != (size_t) n) { fclose(f); free(d); return NULL; }
    fclose(f);
    d[n] = 0;
    *len = n;
    return d;
}
static VkShaderModule mod(const char * name) {
    char path[800];
    snprintf(path, sizeof path, "%s/%s.spv", g_spv, name);
    size_t len = 0;
    char * code = read_file(path, &len);
    if (!code) { printf("VERDICT r11_gfx: FAIL cannot read %s\n", path); exit(1); }
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = len, smi.pCode = (const uint32_t *) code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(g_dev, &smi, NULL, &sm));
    free(code);
    return sm;
}

// The compiler listing of a pipeline (the "Assembly" representation of every executable), once per name, to <out>/asm/<name>.s.
static void capture_asm(VkPipeline p, const char * name) {
    static char seen[64][64];
    static int nseen;
    for (int i = 0; i < nseen; ++i) if (!strcmp(seen[i], name)) return;
    if (nseen < 64) snprintf(seen[nseen++], 64, "%s", name);
    PFN_vkGetPipelineExecutablePropertiesKHR gprops = (PFN_vkGetPipelineExecutablePropertiesKHR) vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR) vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    if (!gprops || !greps) return;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p;
    uint32_t ne = 0;
    gprops(g_dev, &pi, &ne, NULL);
    if (!ne || ne > 8) return;
    VkPipelineExecutablePropertiesKHR ep[8];
    for (uint32_t i = 0; i < ne; i++) ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    gprops(g_dev, &pi, &ne, ep);
    size_t cap = 1 << 16, len = 0;
    char * txt = malloc(cap);
    txt[0] = 0;
    int asm_found = 0;
    for (uint32_t e = 0; e < ne; e++) {
        VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = p, ei.executableIndex = e;
        uint32_t nr = 0;
        greps(g_dev, &ei, &nr, NULL);
        VkPipelineExecutableInternalRepresentationKHR * ir = calloc(nr ? nr : 1, sizeof *ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        greps(g_dev, &ei, &nr, ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].pData = calloc(1, ir[i].dataSize + 1);
        greps(g_dev, &ei, &nr, ir);
        for (uint32_t i = 0; i < nr; i++) {
            if (strstr(ir[i].name, "Assembly") || strstr(ir[i].name, "assembly")) {
                const size_t need = strlen(ir[i].pData) + strlen(ep[e].name) + 96;
                while (len + need + 1 > cap) txt = realloc(txt, cap *= 2);
                len += (size_t) snprintf(txt + len, cap - len, "# pipeline %s, executable \"%s\" (%s), subgroup size %u, representation %s\n%s\n", name, ep[e].name, ep[e].description, ep[e].subgroupSize, ir[i].name, (char *) ir[i].pData);
                asm_found = 1;
            }
            free(ir[i].pData);
        }
        free(ir);
    }
    if (g_out[0] && asm_found) {
        char path[800];
        snprintf(path, sizeof path, "%s/asm", g_out);
        mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/asm/%s.s", g_out, name);
        FILE * f = fopen(path, "w");
        if (f) { fputs(txt, f); fclose(f); }
    }
    printf("LIST %s: %u executable(s), %zu bytes of assembly listing%s\n", name, ne, len, asm_found ? "" : " (NO ASSEMBLY REPRESENTATION RETURNED)");
    free(txt);
}

typedef struct {
    const char * vs, * fs;       // shader names without .spv
    VkFormat col, dep;           // VK_FORMAT_UNDEFINED: no such attachment
    VkSampleCountFlagBits ms;
    int blend, mode;             // blend on; fragment specialisation constant 0
    VkCullModeFlags cull;
    VkColorComponentFlags mask;
    int dtest;                   // depth test ALWAYS and depth write on (else both off)
    const char * cap;
} PD;
static VkPipeline gpipe(const PD * d) {
    VkShaderModule vs = mod(d->vs), fs = mod(d->fs);
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &d->mode};
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fs, st[1].pName = "main", st[1].pSpecializationInfo = &si;
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL, rs.cullMode = d->cull, rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = d->ms;
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = d->dtest ? VK_TRUE : VK_FALSE, ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = d->mask;
    if (d->blend) {
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp = cba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = d->col != VK_FORMAT_UNDEFINED, cb.pAttachments = &cba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dsi = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = 2, dsi.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = d->col != VK_FORMAT_UNDEFINED, rci.pColorAttachmentFormats = &d->col, rci.depthAttachmentFormat = d->dep;
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
    gi.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    gi.stageCount = 2, gi.pStages = st, gi.pVertexInputState = &vin, gi.pInputAssemblyState = &ia, gi.pViewportState = &vps, gi.pRasterizationState = &rs;
    gi.pMultisampleState = &ms, gi.pDepthStencilState = &ds, gi.pColorBlendState = &cb, gi.pDynamicState = &dsi, gi.layout = g_gpl;
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &p));
    vkDestroyShaderModule(g_dev, vs, NULL);
    vkDestroyShaderModule(g_dev, fs, NULL);
    if (d->cap) capture_asm(p, d->cap);
    return p;
}
static VkPipeline cpipe(const char * spv, int mode, const char * cap) {
    VkShaderModule sm = mod(spv);
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &mode};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = sm, ci.stage.pName = "main", ci.stage.pSpecializationInfo = &si;
    ci.layout = g_cpl;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
    vkDestroyShaderModule(g_dev, sm, NULL);
    if (cap) capture_asm(p, cap);
    return p;
}

// ---- a render target and drawing into it ---------------------------------------------------------------------------------------------
typedef struct { VkImageView col, dep; uint32_t w, h; } Rnd;
typedef struct { float color[4]; float z; uint32_t seed, u2, u3; } FillPC;

static void begin_render(const Rnd * r) {
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = r->col, ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    da.imageView = r->dep, da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, da.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // the depth writes are the work being measured: never DONT_CARE
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = (VkRect2D){{0, 0}, {r->w, r->h}}, ri.layerCount = 1;
    ri.colorAttachmentCount = r->col ? 1 : 0, ri.pColorAttachments = &ca, ri.pDepthAttachment = r->dep ? &da : NULL;
    vkCmdBeginRendering(g_cb, &ri);
}
static void set_vp(const Rnd * r, uint32_t scissor_h) {
    VkViewport vp = {0, 0, (float) r->w, (float) r->h, 0, 1};
    VkRect2D sc = {{0, 0}, {r->w, scissor_h ? scissor_h : r->h}};
    vkCmdSetViewport(g_cb, 0, 1, &vp);
    vkCmdSetScissor(g_cb, 0, 1, &sc);
}

// Reads two 64 x 4 pixel regions (top-left and bottom-right) of an image in COLOR/DEPTH attachment layout into `host` (8 KiB per region at 16 B/px max),
// leaving the image in attachment layout again. Recorded into the open command buffer.
#define RB_W 64u
#define RB_H 4u
#define RB_REGION (RB_W * RB_H * 16u)
static void rec_readback(const Img * m, Buf * host) {
    const VkImageLayout from = m->aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar_img(m->img, m->aspect, 1, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy rg[2];
    memset(rg, 0, sizeof rg);
    for (int i = 0; i < 2; ++i) {
        rg[i].bufferOffset = (VkDeviceSize) i * RB_REGION, rg[i].bufferRowLength = RB_W, rg[i].bufferImageHeight = RB_H;
        rg[i].imageSubresource = (VkImageSubresourceLayers){m->aspect, 0, 0, 1};
        rg[i].imageOffset = (VkOffset3D){i ? (int32_t) (m->w - RB_W) : 0, i ? (int32_t) (m->h - RB_H) : 0, 0};
        rg[i].imageExtent = (VkExtent3D){RB_W, RB_H, 1};
    }
    vkCmdCopyImageToBuffer(g_cb, m->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host->buf, 2, rg);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(g_cb, &di);
    bar_img(m->img, m->aspect, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, from);
}
// Pixels of the two regions that differ from the CPU's expectation. mode 0: constant colour c; mode 1: the per-pixel hash of `seed` (RGBA8 only).
static char g_rb_first[160];
static uint32_t check_readback(const Buf * host, int fmt, int mode, const float c[4], uint32_t seed, uint32_t w, uint32_t h) {
    uint8_t want[16];
    r11g_px_expect(fmt, c, want);
    const uint32_t bpp = r11g_fmt_bpp[fmt];
    uint32_t bad = 0;
    for (int r = 0; r < 2; ++r)
        for (uint32_t y = 0; y < RB_H; ++y)
            for (uint32_t x = 0; x < RB_W; ++x) {
                const uint8_t * got = (const uint8_t *) host->map + r * RB_REGION + (y * RB_W + x) * bpp;
                if (mode == 1) {
                    const uint32_t px = (r ? w - RB_W : 0) + x, py = (r ? h - RB_H : 0) + y, hh = r11g_px_hash(px, py, seed);
                    const uint8_t b[4] = {(uint8_t) hh, (uint8_t) (hh >> 8), (uint8_t) (hh >> 16), (uint8_t) (hh >> 24)};
                    bad += memcmp(got, b, 4) != 0;
                } else if (!r11g_px_equal(fmt, got, want)) {
                    if (!bad) {  // keep the first mismatch's bytes: the log then shows which channel or rounding differs
                        int n = snprintf(g_rb_first, sizeof g_rb_first, "region %d px(%u,%u) fmt=%s got=0x", r, x, y, r11g_fmt_name[fmt]);
                        for (int i = (int) bpp - 1; i >= 0 && n < (int) sizeof g_rb_first - 24; --i) n += snprintf(g_rb_first + n, sizeof g_rb_first - (size_t) n, "%02x", got[i]);
                        n += snprintf(g_rb_first + n, sizeof g_rb_first - (size_t) n, " want=0x");
                        for (int i = (int) bpp - 1; i >= 0 && n < (int) sizeof g_rb_first - 4; --i) n += snprintf(g_rb_first + n, sizeof g_rb_first - (size_t) n, "%02x", want[i]);
                    }
                    ++bad;
                }
            }
    return bad;
}

// ---- timed bursts ---------------------------------------------------------------------------------------------------------------------
typedef void (*RecFn)(void * ctx, uint32_t k);
static double burst_ms(RecFn rec, void * ctx, uint32_t k) {
    cb_begin();
    vkCmdResetQueryPool(g_cb, g_qp, 0, 2);
    vkCmdWriteTimestamp(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qp, 0);
    rec(ctx, k);
    vkCmdWriteTimestamp(g_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, 1);
    cb_submit();
    uint64_t ts[2];
    VK_CHECK(vkGetQueryPoolResults(g_dev, g_qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double) (ts[1] - ts[0]) * g_ts_ns * 1e-6;
}
typedef struct { double med, mn, mx, longest; uint32_t k; int over; } Meas;
static FILE * g_csv;
static Meas measure(const char * name, RecFn rec, void * ctx, uint32_t k0, uint32_t kmax, int reps) {
    Meas m = {0};
    uint32_t k = k0;
    double t;
    for (;;) {
        thermal_cell(name);
        t = burst_ms(rec, ctx, k);
        m.longest = r11g_longest(m.longest, t);
        if (t >= 5.0 || k >= kmax) break;
        k = k * 8 > kmax ? kmax : k * 8;
    }
    double kk = (double) k * (g_target_ms / (t > 0.001 ? t : 0.001));
    if (kk > kmax) kk = kmax;
    if (kk < 1) kk = 1;
    m.k = (uint32_t) kk;
    // Warm-up at the final length, and re-aim: the first calibration bursts run on a cold target (a fresh image is cheap to touch, and a blend
    // pass over a cleared surface is faster than over written data), so k from them can be 2.5x too long (the 512 MiB blend cells ran 252 and 274 ms).
    // Up to three more tries until the warm burst is within +-15% of the target; k stays at 1 or kmax when the rule cannot be met by k alone.
    for (int it = 0; it < 4; ++it) {
        thermal_cell(name);
        const double w = burst_ms(rec, ctx, m.k);
        m.longest = r11g_longest(m.longest, w);
        if (it == 3 || (w >= 0.85 * g_target_ms && w <= 1.15 * g_target_ms)) break;
        double nk = (double) m.k * (g_target_ms / (w > 0.001 ? w : 0.001));
        if (nk > kmax) nk = kmax;
        if (nk < 1) nk = 1;
        if ((uint32_t) nk == m.k) break;
        m.k = (uint32_t) nk;
    }
    double v[64];
    if (reps > 64) reps = 64;
    for (int i = 0; i < reps; ++i) {
        thermal_cell(name);
        v[i] = burst_ms(rec, ctx, m.k);
        m.longest = r11g_longest(m.longest, v[i]);
        if (g_csv) fprintf(g_csv, "%s,%d,%.4f,%u\n", name, i, v[i], m.k);
    }
    m.med = median(v, reps), m.mn = m.mx = v[0];
    for (int i = 1; i < reps; ++i) { if (v[i] < m.mn) m.mn = v[i]; if (v[i] > m.mx) m.mx = v[i]; }
    m.over = !r11g_burst_ok(m.longest);
    return m;
}
// --only TEXT[,TEXT...]: a cell runs when its name contains any of the pieces.
static int want_cell(const char * name, const char * only) {
    if (!only) return 1;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", only);
    for (char * t = strtok(buf, ","); t; t = strtok(NULL, ",")) if (strstr(name, t)) return 1;
    return 0;
}
// A burst over the 250 ms desktop-safety limit (any burst of the cell, not only the timed ones) is a failed control (expect-3.md control 6).
// A spread above 10 % is counted and shown in the verdict; it fails the run only with --strict-spread (auto-boost clocks move between repeats).
static void tail_flags(const Meas * m) {
    const int spread = r11g_spread_flag(m->mn, m->mx);
    if (m->over) printf(" BURSTOVER");
    if (spread) printf(" SPREAD>10%%");
    printf("\n");
    if (m->over) {
        ++g_burst_over;
        ++g_fail;
        printf("CTRLFAIL BURSTOVER: the longest burst of the cell above was %.1f ms, over the %.0f ms safety limit\n", m->longest, R11G_BURST_MAX_MS);
    }
    if (spread) {
        ++g_spread_flagged;
        if (g_strict_spread) {
            ++g_fail;
            printf("CTRLFAIL SPREAD: the timed repeats of the cell above differ by more than 10 %% (--strict-spread)\n");
        }
    }
    fflush(stdout);
}

// ---- fill cells (pixel output) --------------------------------------------------------------------------------------------------------
typedef struct { const Rnd * r; VkPipeline pipe; FillPC pc; } FillRec;
static void rec_fill(void * ctx, uint32_t k) {
    FillRec * f = ctx;
    begin_render(f->r);
    set_vp(f->r, 0);
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, f->pipe);
    vkCmdPushConstants(g_cb, g_gpl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof f->pc, &f->pc);
    vkCmdDraw(g_cb, 6, k, 0, 0);
    vkCmdEndRendering(g_cb);
}

static const float k_c1[4] = {0.2f, 0.4f, 0.6f, 0.8f};
// The control of a fill cell: draw A twice then B once inside an occlusion query, read back, compare. brk 1: B scissored to half; brk 2: B skipped.
// Returns 0 when the count AND the read-back are right. count_ok / rb_bad are reported through the pointers.
static int fill_control(const Img * img, int fmt, VkPipeline pipe, int mode, int samples, int blend, int brk, int * count_ok, uint32_t * rb_bad) {
    const int depth = fmt == R11G_D32;
    Rnd r = {depth ? VK_NULL_HANDLE : img->view, depth ? img->view : VK_NULL_HANDLE, img->w, img->h};
    Buf host = buf_make(2 * RB_REGION, 0, 1);
    memset(host.map, 0xee, 2 * RB_REGION);
    float c0[4], c1[4];
    r11g_ctrl_colour(fmt, 0, c0), r11g_ctrl_colour(fmt, 1, c1);
    FillPC a = {{c0[0], c0[1], c0[2], c0[3]}, 0.25f, 0x1111, 0, 0}, b = {{c1[0], c1[1], c1[2], c1[3]}, 0.375f, 0x2222, 0, 0};
    cb_begin();
    vkCmdResetQueryPool(g_cb, g_oq, 0, 1);
    begin_render(&r);
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    set_vp(&r, 0);
    vkCmdBeginQuery(g_cb, g_oq, 0, VK_QUERY_CONTROL_PRECISE_BIT);
    vkCmdPushConstants(g_cb, g_gpl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof a, &a);
    vkCmdDraw(g_cb, 6, 2, 0, 0);
    if (brk != 2) {
        if (brk == 1) set_vp(&r, img->h / 2);
        vkCmdPushConstants(g_cb, g_gpl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof b, &b);
        vkCmdDraw(g_cb, 6, 1, 0, 0);
    }
    vkCmdEndQuery(g_cb, g_oq, 0);
    vkCmdEndRendering(g_cb);
    const int do_rb = samples == 1 && !blend;
    if (do_rb) rec_readback(img, &host);
    cb_submit();
    uint64_t occ = 0;
    VK_CHECK(vkGetQueryPoolResults(g_dev, g_oq, 0, 1, sizeof occ, &occ, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    *count_ok = r11g_count_ok(occ, r11g_fill_expected_count(3, img->w, img->h, (uint32_t) samples));
    *rb_bad = 0;
    if (do_rb) {
        const float cz[4] = {0.375f, 0, 0, 0};
        *rb_bad = check_readback(&host, fmt, mode, depth ? cz : c1, 0x2222, img->w, img->h);
    }
    buf_free(&host);
    const uint64_t want_cnt = r11g_fill_expected_count(3, img->w, img->h, (uint32_t) samples);
    printf("  control count=%llu expected=%llu readback_bad=%u%s\n", (unsigned long long) occ, (unsigned long long) want_cnt, *rb_bad, do_rb ? "" : " (readback n/a)");
    if (!*count_ok && samples > 1)  // facts for the next reader: how far short, and in units of whole rows of 3 passes
        printf("  ms_count_diag ratio=%.5f count_div_3W=%.3f H=%u (spec: passes x W x H x samples; the formula is not the thing that is wrong)\n", (double) occ / (double) want_cnt, (double) occ / (3.0 * img->w), img->h);
    if (*rb_bad) printf("  readback_first_bad %s\n", g_rb_first);
    return !*count_ok || *rb_bad;
}

static FillPC fill_pc(int fmt, int mode) {
    FillPC p = {{k_c1[0], k_c1[1], k_c1[2], k_c1[3]}, fmt == R11G_D32 ? 0.375f : 0.5f, 0x2222, 0, 0};
    (void) mode;
    return p;
}
static VkPipeline fill_pipe(int fmt, int samples, int blend, int mode, const char * cap) {
    const int depth = fmt == R11G_D32;
    PD d = {"gfx_fill_vert", depth ? "gfx_depth_frag" : "gfx_fill_frag", depth ? VK_FORMAT_UNDEFINED : k_vkfmt[fmt], depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED,
            samples == 4 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT, blend, mode, VK_CULL_MODE_NONE, 0xf, depth, cap};
    return gpipe(&d);
}

static double g_fill_rate[R11G_NFMT][4];   // GPix/s of the plain fill cells at 1, 4, 32, 512 MiB
static double g_comp_gbs[2][2][2][2];      // [size 32/512][data][cc][op write/read]
static int g_comp_applied[2][2][2];        // [size][data][cc]: 1 setting applied (or not requested)
static uint32_t g_cc_default_rb = 0xffffffffu, g_cc_disabled_rb = 0xffffffffu;
static int g_switch_verified;

// One fill measurement on an existing image. `label` names the CELL line; extra is appended to it.
static int fill_measure(const Img * img, int fmt, int samples, int blend, int mode, const char * label, const char * cap, const char * extra, int reps, double * gpix, double * gbs) {
    VkPipeline pipe = fill_pipe(fmt, samples, blend, mode, cap);
    int cok;
    uint32_t rb;
    const int bad = fill_control(img, fmt, pipe, mode, samples, blend, 0, &cok, &rb);
    if (bad) {
        printf("CTRLFAIL %s: occlusion count %s, read-back mismatches %u; not timed\n", label, cok ? "ok" : "WRONG", rb);
        ++g_fail;
        vkDestroyPipeline(g_dev, pipe, NULL);
        return 1;
    }
    printf("CONTROL fill %s PASS\n", label);
    const int depth = fmt == R11G_D32;
    const Rnd r = {depth ? VK_NULL_HANDLE : img->view, depth ? img->view : VK_NULL_HANDLE, img->w, img->h};
    FillRec fr = {&r, pipe, fill_pc(fmt, mode)};
    fr.pc.seed = 0x2222;
    const Meas m = measure(label, rec_fill, &fr, 16, 400000, reps);
    const double px = (double) img->w * img->h * m.k, bytes = r11g_fill_bytes(px, r11g_fmt_bpp[fmt], samples);
    *gpix = px / (m.med * 1e6);
    *gbs = bytes / (m.med * 1e6);
    printf("CELL %s kind=fill fmt=%s w=%u h=%u samples=%d blend=%d mode=%s k=%u median_ms=%.3f min_ms=%.3f max_ms=%.3f median_GPix_s=%.2f median_GB_s=%.1f%s", label,
           r11g_fmt_name[fmt], img->w, img->h, samples, blend, mode ? "random" : "solid", m.k, m.med, m.mn, m.mx, *gpix, *gbs, extra);
    tail_flags(&m);
    vkDestroyPipeline(g_dev, pipe, NULL);
    return 0;
}

static void run_fill_cell(const Cell * c, int reps) {
    uint32_t w, h;
    r11g_target_dims((uint64_t) c->mib << 20, r11g_fmt_bpp[c->fmt], 16384, &w, &h);
    const int depth = c->fmt == R11G_D32;
    VkImageUsageFlags u = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) | (c->samples == 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    if (!img_supported(k_vkfmt[c->fmt], u, w, h, c->samples == 4 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT)) { printf("SKIP %s: image not supported by the device\n", c->name); return; }
    Img img = img_make(w, h, 1, k_vkfmt[c->fmt], c->samples == 4 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT, u, 0);
    cb_begin();
    bar_img(img.img, img.aspect, 1, VK_IMAGE_LAYOUT_UNDEFINED, depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    cb_submit();
    double gpix, gbs;
    char cap[64] = "";
    if (!c->blend && c->samples == 1 && c->mib == 4 && (c->fmt == R11G_RGBA8 || c->fmt == R11G_RGBA16F || c->fmt == R11G_D32)) snprintf(cap, sizeof cap, "fill_%s", r11g_fmt_name[c->fmt]);
    if (!fill_measure(&img, c->fmt, c->samples, c->blend, 0, c->name, cap[0] ? cap : NULL, "", reps, &gpix, &gbs) && !c->blend && c->samples == 1)
        for (int m = 0; m < 4; ++m) if (k_mibs[m] == c->mib) g_fill_rate[c->fmt][m] = gpix;
    img_free(&img);
}

// ---- raster cells -------------------------------------------------------------------------------------------------------------------
#define RT 2048u
typedef struct { const Rnd * r; VkPipeline pipe; uint32_t push[8]; uint32_t nvert; } RastRec;
static void rec_rast(void * ctx, uint32_t k) {
    RastRec * f = ctx;
    begin_render(f->r);
    set_vp(f->r, 0);
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, f->pipe);
    vkCmdPushConstants(g_cb, g_gpl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof f->push, f->push);
    vkCmdDraw(g_cb, f->nvert, k, 0, 0);
    vkCmdEndRendering(g_cb);
}
static void rast_push(uint32_t * p, uint32_t a256) {
    const float leg = (float) a256 / 256.0f, stride = (float) r11g_tri_stride(a256), inv = 2.0f / (float) RT, z = 0.5f;
    memcpy(&p[0], &leg, 4), memcpy(&p[1], &stride, 4), memcpy(&p[2], &inv, 4), memcpy(&p[3], &inv, 4);
    p[4] = r11g_tri_gridw(a256, RT);
    memcpy(&p[5], &z, 4);
    p[6] = p[7] = 0;
}
static VkPipeline rast_pipe(int var, const char * cap) {
    PD d = {"gfx_tri_vert", var == RV_DEPTH ? "gfx_depth_frag" : "gfx_tri_frag", var == RV_DEPTH ? VK_FORMAT_UNDEFINED : VK_FORMAT_R8_UNORM, VK_FORMAT_D32_SFLOAT,
            VK_SAMPLE_COUNT_1_BIT, 0, 0, var == RV_CULL ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE, 0, var == RV_DEPTH, cap};
    return gpipe(&d);
}
// Occlusion count of `passes` passes of the triangle grid.
static uint64_t rast_count(const Rnd * r, VkPipeline pipe, uint32_t a256, uint32_t passes) {
    RastRec fr = {r, pipe, {0}, 3u * (uint32_t) r11g_tri_per_pass(a256, RT)};
    rast_push(fr.push, a256);
    cb_begin();
    vkCmdResetQueryPool(g_cb, g_oq, 0, 1);
    begin_render(r);
    set_vp(r, 0);
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdPushConstants(g_cb, g_gpl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof fr.push, fr.push);
    vkCmdBeginQuery(g_cb, g_oq, 0, VK_QUERY_CONTROL_PRECISE_BIT);
    vkCmdDraw(g_cb, fr.nvert, passes, 0, 0);
    vkCmdEndQuery(g_cb, g_oq, 0);
    vkCmdEndRendering(g_cb);
    cb_submit();
    uint64_t occ = 0;
    VK_CHECK(vkGetQueryPoolResults(g_dev, g_oq, 0, 1, sizeof occ, &occ, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return occ;
}

typedef struct { Img col, dep; Rnd rc, rd; } RastTgt;
static void rast_tgt_make(RastTgt * t) {
    t->col = img_make(RT, RT, 1, VK_FORMAT_R8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0);
    t->dep = img_make(RT, RT, 1, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0);
    cb_begin();
    bar_img(t->col.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    bar_img(t->dep.img, VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    cb_submit();
    t->rc = (Rnd){t->col.view, t->dep.view, RT, RT};  // colour-mask-0 and culled cells: colour and an idle depth attachment (test and write off)
    t->rd = (Rnd){VK_NULL_HANDLE, t->dep.view, RT, RT};  // depth-only cells
}
static void rast_tgt_free(RastTgt * t) { img_free(&t->col), img_free(&t->dep); }

static void run_rast_cell(const Cell * c, RastTgt * t, int reps) {
    const Rnd * r = c->var == RV_DEPTH ? &t->rd : &t->rc;
    char cap[64] = "";
    if (c->a256 == 256) snprintf(cap, sizeof cap, "raster_%s", rv_name[c->var]);
    VkPipeline pipe = rast_pipe(c->var, cap[0] ? cap : NULL);
    const uint64_t want = c->var == RV_CULL ? 0 : r11g_tri_expected_count(c->a256, RT, 2), got = rast_count(r, pipe, c->a256, 2);
    printf("  control count=%llu expected=%llu\n", (unsigned long long) got, (unsigned long long) want);
    if (got != want) {
        printf("CTRLFAIL %s: occlusion count %llu, exact expectation %llu; not timed\n", c->name, (unsigned long long) got, (unsigned long long) want);
        ++g_fail;
        vkDestroyPipeline(g_dev, pipe, NULL);
        return;
    }
    printf("CONTROL raster %s PASS\n", c->name);
    RastRec fr = {r, pipe, {0}, 3u * (uint32_t) r11g_tri_per_pass(c->a256, RT)};
    rast_push(fr.push, c->a256);
    const Meas m = measure(c->name, rec_rast, &fr, 4, 200000, reps);
    const double ntri = (double) r11g_tri_per_pass(c->a256, RT) * m.k;
    printf("CELL %s kind=raster leg_px=%.3f area_px2=%.4f tri_per_pass=%llu covered_px_per_tri=%llu k=%u median_ms=%.3f min_ms=%.3f max_ms=%.3f median_GTri_s=%.3f median_GPix_s=%.3f", c->name,
           c->a256 / 256.0, 0.5 * c->a256 / 256.0 * c->a256 / 256.0, (unsigned long long) r11g_tri_per_pass(c->a256, RT), (unsigned long long) r11g_tri_cover(c->a256), m.k, m.med, m.mn, m.mx,
           ntri / (m.med * 1e6), ntri * (double) r11g_tri_cover(c->a256) / (m.med * 1e6));
    tail_flags(&m);
    vkDestroyPipeline(g_dev, pipe, NULL);
}

// ---- texture cells -----------------------------------------------------------------------------------------------------------------------
typedef struct { Img img; VkSampler smp[TF_N]; } TexSet;
static VkSampler make_sampler(int filt) {
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    const int lin = filt != TF_NEAREST && filt != TF_FETCH;
    si.magFilter = si.minFilter = lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = (filt >= TF_TRILINEAR) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = VK_LOD_CLAMP_NONE;
    if (filt == TF_ANISO4 || filt == TF_ANISO16) si.anisotropyEnable = VK_TRUE, si.maxAnisotropy = filt == TF_ANISO4 ? 4.0f : (g_max_aniso < 16 ? (float) g_max_aniso : 16.0f);
    VkSampler s;
    VK_CHECK(vkCreateSampler(g_dev, &si, NULL, &s));
    return s;
}
static void desc_tex(VkImageView v, VkSampler s, const Buf * out) {
    VkDescriptorImageInfo ii = {s, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo bi = {out->buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2];
    memset(w, 0, sizeof w);
    w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = g_ds;
    w[0].dstBinding = 0, w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w[0].pImageInfo = &ii;
    w[1].dstBinding = 1, w[1].descriptorCount = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, w[1].pBufferInfo = &bi;
    vkUpdateDescriptorSets(g_dev, 2, w, 0, NULL);
}

// A mip-mapped texture whose every texel (every level) holds the format's constant, uploaded by copies from one staging buffer.
static Img tex_make_const(int f, uint32_t size) {
    uint32_t mips = 1;
    for (uint32_t s = size; s > 1; s >>= 1) ++mips;
    Img m = img_make(size, size, mips, k_vktex[f], VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    const uint64_t bytes = r11t_mip0_bytes(f, size);
    const int bc = f == R11T_BC1 || f == R11T_BC7;
    Buf st = buf_make(bytes, 0, bc);
    if (bc) {
        const uint32_t bs = f == R11T_BC1 ? 8 : 16;
        uint8_t blk[16];
        if (f == R11T_BC1) r11t_bc1_block(blk); else r11t_bc7_block(blk);
        for (uint64_t o = 0; o < bytes; o += bs) memcpy((uint8_t *) st.map + o, blk, bs);
    }
    VkBufferImageCopy rg[16];
    memset(rg, 0, sizeof rg);
    for (uint32_t l = 0; l < mips; ++l) {
        rg[l].imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
        const uint32_t e = (size >> l) ? (size >> l) : 1;
        rg[l].imageExtent = (VkExtent3D){e, e, 1};
    }
    cb_begin();
    if (!bc) vkCmdFillBuffer(g_cb, st.buf, 0, VK_WHOLE_SIZE, r11t_pattern[f]);
    bar_mem();
    bar_img(m.img, VK_IMAGE_ASPECT_COLOR_BIT, mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(g_cb, st.buf, m.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips, rg);
    bar_img(m.img, VK_IMAGE_ASPECT_COLOR_BIT, mips, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cb_submit();
    buf_free(&st);
    return m;
}

typedef struct { VkPipeline pipe; uint32_t groups; uint32_t push[8]; } CompRec;
static void rec_comp(void * ctx, uint32_t k) {
    CompRec * c = ctx;
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe);
    vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &g_ds, 0, NULL);
    c->push[1] = k;  // gfx_tex.comp: iterations per thread
    vkCmdPushConstants(g_cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof c->push, c->push);
    vkCmdDispatch(g_cb, c->groups, 1, 1);
}
static void tex_grads(int filt, float inv, float * gx, float * gy) {
    switch (filt) {
        case TF_TRILINEAR: *gx = *gy = 1.5f * inv; break;
        case TF_ANISO4: *gx = 4.0f * inv, *gy = inv; break;
        case TF_ANISO16: *gx = 16.0f * inv, *gy = inv; break;
        default: *gx = *gy = inv;
    }
}
// Runs gfx_texctl over n positions of a texture with the given sampler and grads; results to `out` (host).
static void texctl_run(VkPipeline pipe, VkImageView v, VkSampler s, Buf * out, uint32_t n, float gx, float gy) {
    desc_tex(v, s, out);
    cb_begin();
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &g_ds, 0, NULL);
    uint32_t pc[8] = {n, 0, 0, 0, 0, 0, 0, 0};
    memcpy(&pc[1], &gx, 4), memcpy(&pc[2], &gy, 4);
    vkCmdPushConstants(g_cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(g_cb, (n + 63) / 64, 1, 1);
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(g_cb, &di);
    cb_submit();
}
// The gradient control: a 16 x 1 RGBA8 gradient, 64 samples; counts the points (of 64) off by more than 2/255 from the closed form of `expect_filter`.
static uint32_t grad_check(VkPipeline texctl, const Img * grad, VkSampler s, int expect_bilinear, Buf * out) {
    texctl_run(texctl, grad->view, s, out, 64, 1.0f / 16.0f, 1.0f / 16.0f);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        const double u = (i + 0.5) / 64.0, want = expect_bilinear ? r11t_grad_bilinear(u) : r11t_grad_nearest(u);
        bad += !r11t_close(((const float *) out->map)[4 * i], want, 2.0);
    }
    return bad;
}
static Img grad_make(void) {
    Img m = img_make(16, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0);
    Buf st = buf_make(64, 0, 1);
    uint8_t * p = st.map;
    for (int x = 0; x < 16; ++x) p[4 * x] = p[4 * x + 1] = p[4 * x + 2] = (uint8_t) (17 * x), p[4 * x + 3] = 255;
    VkBufferImageCopy rg = {0};
    rg.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, rg.imageExtent = (VkExtent3D){16, 1, 1};
    cb_begin();
    bar_img(m.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(g_cb, st.buf, m.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);
    bar_img(m.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cb_submit();
    buf_free(&st);
    return m;
}

static VkPipeline g_ptex[2], g_ptexctl;
static Buf g_texout;
static int tex_const_ok(const Buf * out, int f, uint32_t n) {
    double c[4];
    r11t_const(f, c);
    for (uint32_t i = 0; i < n; ++i)
        for (int ch = 0; ch < 4; ++ch)
            if (!isnan(c[ch]) && !r11t_close(((const float *) out->map)[4 * i + ch], c[ch], 1.5)) return 0;
    return 1;
}

static void run_tex_cell(const Cell * c, TexSet * ts, int reps) {
    float gx, gy;
    const float inv = 1.0f / (float) c->size;
    tex_grads(c->var, inv, &gx, &gy);
    const VkSampler s = ts->smp[c->var];
    texctl_run(g_ptexctl, ts->img.view, s, &g_texout, 64, gx, gy);
    if (!tex_const_ok(&g_texout, c->fmt, 64)) {
        printf("CTRLFAIL %s: the constant texture did not come back as the constant (filter path not live); not timed\n", c->name);
        ++g_fail;
        return;
    }
    printf("CONTROL texture %s PASS (constant returned)\n", c->name);
    desc_tex(ts->img.view, s, &g_texout);
    CompRec cr = {g_ptex[c->var == TF_FETCH], 4096, {c->size - 1, 0, 0, 0, 0, 0x5eed, 0, 0}};
    memcpy(&cr.push[2], &inv, 4), memcpy(&cr.push[3], &gx, 4), memcpy(&cr.push[4], &gy, 4);
    const Meas m = measure(c->name, rec_comp, &cr, 16, 1u << 18, reps);
    const double samples = 4096.0 * 64.0 * m.k;
    printf("CELL %s kind=texture fmt=%s size=%u filter=%s iters=%u median_ms=%.3f min_ms=%.3f max_ms=%.3f median_GSample_s=%.2f", c->name, r11t_name[c->fmt], c->size, tf_name[c->var], m.k, m.med, m.mn, m.mx,
           samples / (m.med * 1e6));
    tail_flags(&m);
}

// ---- compression arm ---------------------------------------------------------------------------------------------------------------------
typedef struct { uint32_t w, h, per, total; VkPipeline pipe; } ReadRec;
static void rec_read(void * ctx, uint32_t k) {
    ReadRec * r = ctx;
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe);
    vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &g_ds, 0, NULL);
    const uint32_t pc[8] = {r->w, r->per, r->total, 0, 0, 0, 0, 0};
    vkCmdPushConstants(g_cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    for (uint32_t i = 0; i < k; ++i) vkCmdDispatch(g_cb, r->total / 64, 1, 1);
}
static void run_comp_cell(const Cell * c, int reps, int mi) {
    uint32_t w, h;
    r11g_target_dims((uint64_t) c->mib << 20, 4, 16384, &w, &h);
    const VkImageUsageFlags u = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (c->cc && !g_have_cc) { printf("SKIP %s: no switch available (VK_EXT_image_compression_control missing on this device)\n", c->name); return; }
    if (!img_supported(VK_FORMAT_R8G8B8A8_UNORM, u, w, h, VK_SAMPLE_COUNT_1_BIT)) { printf("SKIP %s: image not supported\n", c->name); return; }
    Img img = img_make(w, h, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, u, c->cc);
    cb_begin();
    bar_img(img.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    cb_submit();
    const uint32_t rb = cc_readback(&img);
    const int applied = r11g_comp_applied(c->cc, rb);
    g_comp_applied[mi][c->var][c->cc] = applied;
    if (!c->cc) g_cc_default_rb = rb; else g_cc_disabled_rb = rb;
    printf("CC %s requested=%s readback_flags=0x%x setting_%s\n", c->name, c->cc ? "disabled" : "default", rb, applied ? "applied" : "NOT_APPLIED");
    char extra[64], lbl[128];
    snprintf(extra, sizeof extra, " cc=%s setting_applied=%d", c->cc ? "disabled" : "default", applied);
    snprintf(lbl, sizeof lbl, "%.100s_write", c->name);
    double gpix, gbs;
    const int skip = fill_measure(&img, R11G_RGBA8, 1, 0, c->var, lbl, c->mib == 32 && !c->cc && !c->var ? "comp_fill_solid" : (c->mib == 32 && !c->cc && c->var ? "comp_fill_random" : NULL), extra, reps, &gpix, &gbs);
    if (!skip) {
        g_comp_gbs[mi][c->var][c->cc][0] = gbs;
        cb_begin();
        bar_img(img.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        cb_submit();
        VkSampler smp = make_sampler(TF_NEAREST);
        desc_tex(img.view, smp, &g_texout);
        VkPipeline rp = cpipe("gfx_readsum_comp", 0, c->mib == 32 && !c->cc && !c->var ? "comp_read" : NULL);
        ReadRec rr = {w, 0, 0, 32768u * 64u, rp};
        rr.per = (w * h) / rr.total;
        const Meas m = measure(c->name, rec_read, &rr, 4, 20000, reps);
        const double rbytes = (double) w * h * 4 * m.k;
        g_comp_gbs[mi][c->var][c->cc][1] = rbytes / (m.med * 1e6);
        printf("CELL %s_read kind=compression_read fmt=RGBA8 w=%u h=%u mode=%s k=%u median_ms=%.3f min_ms=%.3f max_ms=%.3f median_GB_s=%.1f%s", c->name, w, h, c->var ? "random" : "solid", m.k, m.med, m.mn, m.mx,
               rbytes / (m.med * 1e6), extra);
        tail_flags(&m);
        vkDestroyPipeline(g_dev, rp, NULL);
        vkDestroySampler(g_dev, smp, NULL);
    }
    img_free(&img);
}

// ---- plan, cpu checks ---------------------------------------------------------------------------------------------------------------
static double est_seconds(int quick) {
    double s = 0;
    const int reps = quick ? 2 : 5;
    for (int i = 0; i < g_ncells; ++i) {
        const double burst = g_target_ms * 1e-3 * (quick ? 0.5 : 1.0);
        double per = 0.05 + 0.3 * burst + (reps + 1) * burst;  // control + calibration (about 3 short runs) + warm-up + repeats
        if (g_cells[i].kind == C_COMP) per = 2 * per + 0.05;  // write and read
        s += per;
    }
    return s + 3.0;  // controls and breaks
}
static void do_plan(int quick) {
    build_cell_list(quick);
    int n[4] = {0};
    for (int i = 0; i < g_ncells; ++i) n[g_cells[i].kind]++;
    printf("PLAN r11_gfx: %d cells: %d pixel-output (9 formats x 4 sizes, blend, MSAA 4), %d raster (7 triangle sizes x colormask0/depthonly/culled), %d texture (6 formats x 2 sizes x 6 filters), %d compression groups (each writes and reads)\n",
           g_ncells, n[C_FILL], n[C_RAST], n[C_TEX], n[C_COMP]);
    printf("PLAN estimated card load %.0f s (about %.1f min) with bursts of %.0f ms, 5 repeats%s; wall time is longer: every group waits for a hotspot below %.0f C and every cell and burst series pauses at %.0f C until %.0f C (--cool-c; up to 120 s each wait)\n", est_seconds(quick), est_seconds(quick) / 60, g_target_ms * (quick ? 0.5 : 1.0), quick ? " (quick: 2 repeats, no 8192 RGBA16F/RGBA32F textures)" : "", g_cool_c - 12.0, g_cool_c, g_cool_c - 6.0);
    printf("PLAN bursts: calibrated to --burst-ms (default 100), never above 250 ms; a burst that ran over is marked BURSTOVER (after calibration every cell is re-aimed on a warm burst to within 15%% of the target)\n");
    printf("PLAN controls: occlusion count == passes x W x H x samples; read-back of two pixel regions; raster count == triangles x exact covered pixels, culled == 0; constant texture returns its constant\n");
    printf("PLAN breaks: (1) last draw scissored to half; (2) last draw skipped; (3) compression read-back checked on an image that never asked for the switch; (4) nearest sampler vs bilinear analytic values; (5) culled pipeline checked against the visible expectation\n");
    printf("PLAN compression: VK_EXT_image_compression_control (VK_IMAGE_COMPRESSION_DISABLED_EXT per image, read back through VkImageCompressionPropertiesEXT); RADV GFX8-GFX11.7 DCC variables are not used; no extension or no visible change: \"no switch available\"\n");
    printf("PLAN not done: MSAA 8, blend on depth, resolve, texture arrays and 3D textures, shader-side ROP export formats (integer targets)\n");
    printf("PLAN desktop: draws share the graphics queue with the desktop compositor; the owner is told before the window (bursts of at most 0.25 s)\n");
    for (int i = 0; i < g_ncells; ++i) {
        if (g_cells[i].kind == C_COMP) printf("PLANCELL %s_write\nPLANCELL %s_read\n", g_cells[i].name, g_cells[i].name);
        else printf("PLANCELL %s\n", g_cells[i].name);
    }
}
static int cpu_checks(void) {
    int bad = 0;
    build_cell_list(0);
    if (g_ncells < 100) { printf("cpu-check FAIL: too few cells (%d)\n", g_ncells); bad++; }
    for (int i = 0; i < g_ncells; ++i)
        for (int j = i + 1; j < g_ncells; ++j)
            if (!strcmp(g_cells[i].name, g_cells[j].name)) { printf("cpu-check FAIL: duplicate cell name %s\n", g_cells[i].name); bad++; }
    const char * spvs[] = {"gfx_fill_vert", "gfx_fill_frag", "gfx_depth_frag", "gfx_tri_vert", "gfx_tri_frag", "gfx_tex_comp", "gfx_texctl_comp", "gfx_readsum_comp"};
    for (int i = 0; i < 8; ++i) {
        char p[800];
        snprintf(p, sizeof p, "%s/%s.spv", g_spv[0] ? g_spv : ".", spvs[i]);
        size_t n;
        char * d = read_file(p, &n);
        if (!d || n < 20 || ((uint32_t *) d)[0] != 0x07230203u) { printf("cpu-check FAIL: %s is not a SPIR-V file\n", p); bad++; }
        else printf("cpu-check ok: %s (%zu bytes, SPIR-V magic)\n", p, n);
        free(d);
    }
    // a few reference facts, so a header edit that breaks them is seen here too
    if (r11g_tri_cover(1024) != 10 || r11g_hash32(1) != 0x688990c0u || r11g_f32_to_f16(0.2f) != 0x3266) { printf("cpu-check FAIL: reference header disagrees with its known values\n"); bad++; }
    for (int i = 0; i < g_ncells; ++i) {
        const Cell * c = &g_cells[i];
        if (c->kind == C_FILL) {
            uint32_t w, h;
            if (!r11g_target_dims((uint64_t) c->mib << 20, r11g_fmt_bpp[c->fmt], 16384, &w, &h)) { printf("cpu-check FAIL: %s has no valid target shape\n", c->name); bad++; }
        }
        if (c->kind == C_RAST && r11g_tri_per_pass(c->a256, 2048) * 3 > 0xffffffffull) { printf("cpu-check FAIL: %s vertex count overflows\n", c->name); bad++; }
    }
    int cells[4] = {0};
    for (int i = 0; i < g_ncells; ++i) cells[g_cells[i].kind]++;
    printf("cpu-check: %d cells (fill %d, raster %d, texture %d, compression groups %d), %d problem(s)\n", g_ncells, cells[0], cells[1], cells[2], cells[3], bad);
    return bad ? 1 : 0;
}

// ---- main ----------------------------------------------------------------------------------------------------------------------------------
static void run_controls_and_breaks(void) {
    printf("CONTROLS and deliberate breaks (1 MiB RGBA8 target, texture gradient, raster grid)\n");
    {  // fill: clean control must pass; breaks 1 and 2 must fail
        uint32_t w, h;
        r11g_target_dims(1u << 20, 4, 16384, &w, &h);
        Img img = img_make(w, h, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
        cb_begin();
        bar_img(img.img, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        cb_submit();
        VkPipeline pipe = fill_pipe(R11G_RGBA8, 1, 0, 0, NULL);
        int cok;
        uint32_t rb;
        int bad = fill_control(&img, R11G_RGBA8, pipe, 0, 1, 0, 0, &cok, &rb);
        printf("CONTROL fill_clean %s\n", bad ? "FAIL" : "PASS");
        if (bad) ++g_fail;
        for (int b = 1; b <= 2; ++b) {
            bad = fill_control(&img, R11G_RGBA8, pipe, 0, 1, 0, b, &cok, &rb);
            const int good = !cok && rb > 0;  // both detectors must see it
            printf("BREAK %s: count %s, read-back %s -> %s\n", b == 1 ? "scissor_half" : "skip_last_draw", cok ? "ok" : "FAIL", rb ? "FAIL" : "ok", good ? "ok (FAIL as required)" : "PROBE INVALID (a detector missed it)");
            if (!good) ++g_fail;
        }
        vkDestroyPipeline(g_dev, pipe, NULL);
        img_free(&img);
    }
    {  // texture: gradient with the right filters passes; nearest against the bilinear closed form must fail
        VkPipeline ctl = g_ptexctl;
        Img grad = grad_make();
        VkSampler sn = make_sampler(TF_NEAREST), sb = make_sampler(TF_BILINEAR);
        const uint32_t bn = grad_check(ctl, &grad, sn, 0, &g_texout), bb = grad_check(ctl, &grad, sb, 1, &g_texout), brk = grad_check(ctl, &grad, sn, 1, &g_texout);
        printf("CONTROL texture_gradient nearest_vs_nearest_bad=%u bilinear_vs_bilinear_bad=%u %s\n", bn, bb, (bn == 0 && bb == 0) ? "PASS" : "FAIL");
        if (bn || bb) ++g_fail;
        printf("BREAK nearest_vs_bilinear_analytic: %u of 64 points off -> %s\n", brk, brk >= 16 ? "ok (FAIL as required)" : "PROBE INVALID (the check cannot tell the filters apart)");
        if (brk < 16) ++g_fail;
        vkDestroySampler(g_dev, sn, NULL), vkDestroySampler(g_dev, sb, NULL);
        img_free(&grad);
    }
    {  // raster: visible count exact, culled count 0, and the culled pipeline against the visible expectation must fail
        RastTgt t;
        rast_tgt_make(&t);
        VkPipeline pv = rast_pipe(RV_CM0, NULL), pc = rast_pipe(RV_CULL, NULL);
        const uint32_t leg = 1024;
        const uint64_t vis = rast_count(&t.rc, pv, leg, 2), cul = rast_count(&t.rc, pc, leg, 2), want = r11g_tri_expected_count(leg, RT, 2);
        printf("CONTROL raster_exact visible=%llu expected=%llu culled=%llu %s\n", (unsigned long long) vis, (unsigned long long) want, (unsigned long long) cul, (vis == want && cul == 0) ? "PASS" : "FAIL");
        if (vis != want || cul != 0) ++g_fail;
        printf("BREAK culled_vs_visible_expectation: culled count %llu against %llu -> %s\n", (unsigned long long) cul, (unsigned long long) want, cul != want ? "ok (FAIL as required)" : "PROBE INVALID");
        if (cul == want) ++g_fail;
        vkDestroyPipeline(g_dev, pv, NULL), vkDestroyPipeline(g_dev, pc, NULL);
        rast_tgt_free(&t);
    }
    {  // compression switch: an image that asked for nothing, checked as if it had asked for DISABLED, must be flagged "setting not applied"
        if (!g_have_cc) {
            printf("BREAK compression_not_applied: n/a, VK_EXT_image_compression_control not available (arm reports \"no switch available\")\n");
            printf("COMPSWITCH NO SWITCH AVAILABLE (VK_EXT_image_compression_control absent); the solid and random default-compression cells still run, only the disabled arm and break 5 are n/a\n");
        } else {
            const VkImageUsageFlags u = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            Img plain = img_make(512, 512, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, u, 0), off = img_make(512, 512, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, u, 1);
            const uint32_t prb = cc_readback(&plain), orb = cc_readback(&off);
            g_cc_default_rb = prb, g_cc_disabled_rb = orb;
            g_switch_verified = r11g_comp_switch_verified(orb, prb);
            printf("CONTROL compression_readback default_flags=0x%x disabled_flags=0x%x disabled_image_applied=%s\n", prb, orb, r11g_comp_applied(1, orb) ? "yes" : "NO");
            const int flagged = !r11g_comp_applied(1, prb);
            printf("BREAK compression_not_applied: an ordinary image checked as 'disabled' reads 0x%x -> %s\n", prb, flagged ? "ok (flagged setting not applied, FAIL as required)" : "UNVERIFIED (the ordinary image also reads back disabled: the read-back cannot show the switch)");
            printf("COMPSWITCH %s\n", g_switch_verified ? "verified: the disabled image reads back disabled and an ordinary image does not" : "NO SWITCH AVAILABLE (not verifiable); the compressible-against-random comparison stands alone");
            img_free(&plain), img_free(&off);
        }
    }
    fflush(stdout);
}

static void verdicts(void) {
    for (int f = 0; f < R11G_NFMT; ++f)
        if (g_fill_rate[f][0] > 0 && g_fill_rate[f][3] > 0)
            printf("SUMMARY bound %s: %.1f GPix/s at 1 MiB, %.1f at 512 MiB -> %s\n", r11g_fmt_name[f], g_fill_rate[f][0], g_fill_rate[f][3],
                   (const char *[]){"flat with size (ROP or shader bound)", "mixed", "memory-limited at 512 MiB"}[r11g_bound_class(g_fill_rate[f][0], g_fill_rate[f][3])]);
    for (int mi = 0; mi < 2; ++mi) {
        for (int op = 0; op < 2; ++op) {
            for (int d = 0; d < 2; ++d) {
                const double on = g_comp_gbs[mi][d][0][op], off = g_comp_gbs[mi][d][1][op];
                if (on > 0 && off > 0 && g_comp_applied[mi][d][1])
                    printf("SUMMARY compression %uMiB %s %s: default %.1f GB/s, disabled %.1f GB/s -> %s\n", mi ? 512 : 32, d ? "random" : "solid", op ? "read" : "write", on, off,
                           (const char *[]){"no change", "compression on is faster", "compression on is slower"}[r11g_comp_effect(on, off, 0.05) == 0 ? 0 : r11g_comp_effect(on, off, 0.05) == 1 ? 1 : 2]);
            }
            const double s = g_comp_gbs[mi][0][0][op], r = g_comp_gbs[mi][1][0][op];
            if (s > 0 && r > 0) printf("SUMMARY compressible_vs_random %uMiB %s: solid %.1f GB/s, random %.1f GB/s (ratio %.2f)\n", mi ? 512 : 32, op ? "read" : "write", s, r, s / r);
        }
    }
    if (!g_switch_verified) printf("SUMMARY compression disabled arm: n/a, no switch available (the default-compression solid and random cells above are the whole comparison)\n");
    // memory anchor: the 512 MiB uncompressed write must not exceed the bus. Random data, compression disabled when the switch is verified.
    const double a = g_switch_verified ? g_comp_gbs[1][1][1][0] : g_comp_gbs[1][1][0][0];
    if (a > 0) {
        const int ok = r11g_anchor_ok(a);
        printf("CONTROL memory_anchor %s 512MiB random write %.1f GB/s (bus %.0f GB/s stock, %.1f at 1408 MHz) %s\n", g_switch_verified ? "compression-disabled" : "default-compression", a, R11G_BUS_STOCK_GBS, R11G_BUS_OC_GBS, ok ? "PASS" : "FAIL");
        if (!ok) { ++g_fail; printf("CTRLFAIL memory anchor: an incompressible fill above the bus; the compression and size results are not interpreted\n"); }
    } else {
        printf("CONTROL memory_anchor not measured (the 512 MiB compression cell did not run)\n");
    }
}

int main(int argc, char ** argv) {
    int mode = 0, reps = 5, quick = 0;
    const char * only = NULL;
    const char * e = getenv("R11_OUT");
    if (e) snprintf(g_out, sizeof g_out, "%s", e);
    e = getenv("R11_SPV");
    snprintf(g_spv, sizeof g_spv, "%s", e ? e : "spv");
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--plan")) mode = 1;
        else if (!strcmp(argv[i], "--cpu-checks")) mode = 2;
        else if (!strcmp(argv[i], "--run")) mode = 3;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) snprintf(g_out, sizeof g_out, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--burst-ms") && i + 1 < argc) g_target_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--cool-c") && i + 1 < argc) g_cool_c = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--strict-spread")) g_strict_spread = 1;
        else { fprintf(stderr, "usage: r11_gfx --plan | --cpu-checks | --run [--out DIR] [--reps N] [--burst-ms MS] [--cool-c C] [--only TEXT] [--quick] [--strict-spread]\n"); return 2; }
    }
    if (g_cool_c < 50.0 || g_cool_c > 85.0) { fprintf(stderr, "--cool-c must be between 50 and 85 (the stop is 95 C)\n"); return 2; }
    if (g_target_ms > 200.0 || g_target_ms < 5.0) { fprintf(stderr, "--burst-ms must be between 5 and 200 (the rule is 250 ms at most, with calibration slack)\n"); return 2; }
    if (reps < 1) reps = 1;
    if (quick) reps = reps > 2 ? 2 : reps, g_target_ms *= 0.5;
    if (mode == 1) { if (quick) g_target_ms *= 2; do_plan(quick); return 0; }
    if (mode == 2) return cpu_checks();
    if (mode != 3) { fprintf(stderr, "usage: r11_gfx --plan | --cpu-checks | --run [...]\n"); return 2; }

    setvbuf(stdout, NULL, _IOLBF, 0);
    build_cell_list(quick);
    vk_init();
    if (g_out[0]) mkdir(g_out, 0755);
    if (g_out[0]) { char p[800]; snprintf(p, sizeof p, "%s/gfx_repeats.csv", g_out); g_csv = fopen(p, "w"); if (g_csv) fprintf(g_csv, "cell,repeat,gpu_ms,k\n"); }
    g_texout = buf_make(4u << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);  // host visible: the control results are read from it; the timed shaders never write it
    g_ptex[0] = cpipe("gfx_tex_comp", 0, "tex_filter");
    g_ptex[1] = cpipe("gfx_tex_comp", 1, "tex_fetch");
    g_ptexctl = cpipe("gfx_texctl_comp", 0, NULL);
    printf("CLOCKS before: hotspot %.0f C, power %.0f W (clocks come from r11_sampler)\n", hotspot_c(), read_num(g_hwmon, "power1_average", 1e-6));
    printf("BURST target %.0f ms, %d repeats, limit %.0f ms\n", g_target_ms, reps, R11G_BURST_MAX_MS);

    thermal_gate();
    run_controls_and_breaks();
    if (g_fail) {
        printf("VERDICT r11_gfx: FAIL a first control or break did not behave; no timing cell is read\n");
        return 1;
    }

    int group = -1;
    RastTgt rt;
    memset(&rt, 0, sizeof rt);
    TexSet ts;
    memset(&ts, 0, sizeof ts);
    int ts_fmt = -1;
    uint32_t ts_size = 0;
    for (int i = 0; i < g_ncells; ++i) {
        const Cell * c = &g_cells[i];
        if (!want_cell(c->name, only)) continue;
        thermal_cell(c->name);
        if (c->kind == C_FILL) {
            if (group != c->fmt * 100 + c->blend * 10 + c->samples) { thermal_gate(); group = c->fmt * 100 + c->blend * 10 + c->samples; }
            run_fill_cell(c, reps);
        } else if (c->kind == C_RAST) {
            if (!rt.col.img) { thermal_gate(); rast_tgt_make(&rt); }
            if (group != 1000 + c->var * 10 + (int) c->a256 / 128) { group = 1000 + c->var * 10 + (int) c->a256 / 128; }
            run_rast_cell(c, &rt, reps);
        } else if (c->kind == C_TEX) {
            if (rt.col.img) rast_tgt_free(&rt);
            if (ts_fmt != c->fmt || ts_size != c->size) {
                if (ts.img.img) { img_free(&ts.img); for (int t = 0; t < TF_N; ++t) vkDestroySampler(g_dev, ts.smp[t], NULL); }
                thermal_gate();
                VkFormatProperties props;
                vkGetPhysicalDeviceFormatProperties(g_pd, k_vktex[c->fmt], &props);
                const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
                if ((props.optimalTilingFeatures & need) != need) { printf("SKIP texture %s %u: format not filterable on this device\n", r11t_name[c->fmt], c->size); ts_fmt = -1; continue; }
                ts.img = tex_make_const(c->fmt, c->size);
                for (int t = 0; t < TF_N; ++t) ts.smp[t] = make_sampler(t);
                ts_fmt = c->fmt, ts_size = c->size;
            }
            run_tex_cell(c, &ts, reps);
        } else {
            if (ts.img.img) { img_free(&ts.img); for (int t = 0; t < TF_N; ++t) vkDestroySampler(g_dev, ts.smp[t], NULL); memset(&ts, 0, sizeof ts); ts_fmt = -1; }
            thermal_gate();
            run_comp_cell(c, reps, c->mib == 512);
        }
    }
    if (rt.col.img) rast_tgt_free(&rt);
    verdicts();
    if (g_csv) fclose(g_csv);
    printf("CLOCKS after: hotspot %.0f C (highest seen %.0f C), power %.0f W\n", hotspot_c(), g_hottest, read_num(g_hwmon, "power1_average", 1e-6));
    printf("SUMMARY controls_or_breaks_failed=%d burst_over=%d spread_flagged=%d strict_spread=%d\n", g_fail, g_burst_over, g_spread_flagged, g_strict_spread);
    if (g_fail) printf("VERDICT r11_gfx: FAIL see CTRLFAIL / BREAK lines\n");
    else if (g_spread_flagged) printf("VERDICT r11_gfx: PASS WITH SPREAD WARNINGS (%d cells >10%% between repeats; all controls passed and every deliberate break failed as required)\n", g_spread_flagged);
    else printf("VERDICT r11_gfx: PASS all controls passed and every deliberate break failed as required\n");
    return g_fail ? 1 : 0;
}
