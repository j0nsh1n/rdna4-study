// Round 11 item 1: the ray tracing units through Vulkan ray queries (VK_KHR_ray_query, VK_KHR_acceleration_structure) on RADV.
// Timing, in-shader cycles (VK_KHR_shader_clock) and the compiler listing only. It never reads the bytes of an acceleration
// structure (open question 3 of PLAN-r11.md: default no).
//
//   r11_rt --plan                    the cells and the card time, no Vulkan
//   r11_rt --cpu-checks              files and cell list only, no Vulkan, no card
//   r11_rt --run [--out DIR] [--reps N] [--only TEXT] [--quick] [--big]
//
// Controls (each decides whether the cells after it are read): every scene gets a CPU brute-force check of a 4096-ray sample against
// the card's committed hits (hit/miss exact, distance 1e-4 relative); the one-triangle scene has a hand-counted analytic hit count;
// every lattice rate cell's hit count must equal the CPU count exactly; a chain cell must be >= 3x slower than the same rays on
// a whole wave; cells whose core clock moved more than 3 % between first and last repeat are flagged.
// Deliberate breaks (each must make its control FAIL): a triangle moved on the card only; the chain run with all lanes active;
// one extra instance against the hand count. The plan's unbuilt-TLAS idea is replaced by a built, EMPTY top level (valid in the
// API); querying a TLAS that was never built is undefined and could hang the card.
// Exit: 0 all controls pass and breaks failed as required; 1 a control or break failed; 2 usage; 3 device or extension missing;
// 4 hotspot at the stop temperature; 5 a dispatch did not finish in time (device probably hung; nothing is retried).
#define _GNU_SOURCE
#include <ctype.h>
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

#include "r11_rt_ref.h"

// ---- the cell list (no Vulkan needed) -------------------------------------------------------------------------------------
enum { K_S0, K_S1, K_S2D, K_S2A, K_S3, K_EMPTY };
typedef struct {
    char name[96];
    int scene;       // K_*
    uint32_t n;      // triangles / instances (S0: legs of the triangle in cells)
    int rot;         // S3: 180-degree instance transform
    uint32_t flags;  // 1 first hit, 2 shuffled
    float tmax;
    uint32_t threads;
    int floor_cell;  // tmax < 1: every ray must miss
} RateCell;
typedef struct {
    char name[64];
    uint32_t n;      // scene size for the chain
} ChainCell;

#define MAXCELLS 256
static RateCell g_cells[MAXCELLS];
static int g_ncells;
static ChainCell g_chain[16];
static int g_nchain;

static void add_rate(int scene, uint32_t n, int rot, uint32_t flags, float tmax, uint32_t threads, const char * tag) {
    RateCell * c = &g_cells[g_ncells++];
    static const char * sn[] = {"S0", "S1", "S2diag", "S2axis", "S3", "EMPTY"};
    snprintf(c->name, sizeof c->name, "%s_n%u%s_%s_%s_%s_t%uk%s%s", sn[scene], n, rot ? "_rot" : "", flags & 1 ? "first" : "closest",
             flags & 2 ? "shuf" : "coh", tmax < 1.0f ? "tmaxfloor" : tmax < 100.0f ? "tmaxshort" : "tmaxlong", threads / 1024,
             tag[0] ? "_" : "", tag);
    c->scene = scene, c->n = n, c->rot = rot, c->flags = flags, c->tmax = tmax, c->threads = threads, c->floor_cell = tmax < 1.0f;
}

static void build_cell_list(int big) {
    g_ncells = g_nchain = 0;
    const uint32_t T = 262144;
    add_rate(K_S0, 64, 0, 0, 1e4f, T, "");
    uint32_t sizes[8] = {8, 64, 512, 4096, 32768, 262144, 2097152}, ns = big ? 7 : 6;
    for (uint32_t i = 0; i < ns; ++i)
        for (uint32_t f = 0; f < 2; ++f)        // 0 closest, 1 first hit
            for (uint32_t sh = 0; sh < 2; ++sh)  // 0 coherent, 1 shuffled
                for (int tl = 0; tl < 2; ++tl) add_rate(K_S1, sizes[i], 0, f | (sh << 1), tl ? 1e4f : 1.01f, T, "");
    for (uint32_t i = 0; i < 6; i += 2) add_rate(K_S1, sizes[i + 1], 0, 0, 0.5f, T, "");  // floor: 64, 4096, 262144
    add_rate(K_EMPTY, 0, 0, 0, 1e4f, T, "");
    static const uint32_t s2n[2] = {4096, 262144}, s3n[4] = {1, 16, 256, 4096}, occ[3] = {4096, 16384, 65536};
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j) add_rate(k ? K_S2A : K_S2D, s2n[j], 0, 2, 1e4f, T, "");
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 4; ++j) add_rate(K_S3, s3n[j], r, 2, 1e4f, T, "");
    for (int j = 0; j < 3; ++j) add_rate(K_S1, 4096, 0, 2, 1e4f, occ[j], "occ");
    uint32_t cs[7] = {1, 8, 64, 512, 4096, 32768, 262144};
    for (int i = 0; i < 7; ++i) {
        snprintf(g_chain[g_nchain].name, sizeof g_chain[0].name, "chain_n%u", cs[i]);
        g_chain[g_nchain++].n = cs[i];
    }
}

// ---- Vulkan --------------------------------------------------------------------------------------------------------------
#define VK_CHECK(x)                                                                                     \
    do {                                                                                                \
        VkResult r_ = (x);                                                                              \
        if (r_ != VK_SUCCESS) {                                                                         \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int) r_, __FILE__, __LINE__, #x);       \
            printf("VERDICT r11_rt: FAIL Vulkan error %d (%s)%s\n", (int) r_, #x, r_ == VK_ERROR_DEVICE_LOST ? " DEVICE LOST" : "");    \
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
static VkQueryPool g_qp;
static double g_ts_ns;
static VkPhysicalDeviceMemoryProperties g_mp;
static VkDescriptorSetLayout g_dsl;
static VkPipelineLayout g_pl;
static VkDescriptorPool g_dp;
static VkDescriptorSet g_ds;
static VkDeviceSize g_scratch_align = 128;
static char g_drm[512], g_hwmon[1024], g_out[512] = "", g_spv[512] = "";
static double g_hottest;

static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkDestroyAccelerationStructureKHR pDestroyAS;
static PFN_vkGetAccelerationStructureBuildSizesKHR pBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuild;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pASAddr;

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static int cmp_d(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return x < y ? -1 : x > y; }
static double median(double * v, int n) {
    double * c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
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

// Waits below 60 C (up to 120 s) before a group of cells; stops (exit 4) at 95 C; no sensor means no whole-card work.
static void thermal_gate(void) {
    double t = hotspot_c();
    if (t < 0) { printf("VERDICT thermal: FAIL hotspot sensor not found; refusing whole-card work\n"); exit(1); }
    if (t > g_hottest) g_hottest = t;
    if (t >= 95.0) { printf("THERMAL hotspot %.0f C: stopping\n", t); exit(4); }
    double t0 = now_ns(), cur = t;
    while ((cur = hotspot_c()) >= 60.0 && now_ns() - t0 < 120e9) { if (cur >= 95.0) exit(4); usleep(250000); }
    if (cur >= 60.0) { printf("THERMAL hotspot still %.0f C after 120 s: stopping\n", cur); exit(7); }
    if (t >= 60.0) printf("THERMAL paused %.1f s: %.0f C -> %.0f C\n", (now_ns() - t0) * 1e-9, t, cur);
}

static int has_ext(VkExtensionProperties * ex, uint32_t n, const char * name) {
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
    app.pApplicationName = "rdna4-r11-rt";
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
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT, &asp};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(g_pd, &p2);
    g_ts_ns = p2.properties.limits.timestampPeriod;
    g_scratch_align = asp.minAccelerationStructureScratchOffsetAlignment ? asp.minAccelerationStructureScratchOffsetAlignment : 128;
    find_sysfs(&pci);
    printf("DEVICE %s (%04x:%04x) driver %s %s; sysfs %s hwmon %s; AS limits: max geometries %llu, max primitives %llu, max instances %llu\n",
           p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID, drv.driverName, drv.driverInfo,
           g_drm[0] ? g_drm : "(not found)", g_hwmon[0] ? g_hwmon : "(not found)", (unsigned long long) asp.maxGeometryCount,
           (unsigned long long) asp.maxPrimitiveCount, (unsigned long long) asp.maxInstanceCount);
    if (strcmp(drv.driverName, "radv")) { printf("VERDICT device: FAIL driver is %s, not radv\n", drv.driverName); exit(3); }
    vkGetPhysicalDeviceMemoryProperties(g_pd, &g_mp);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    VkQueueFamilyProperties * qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    g_qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)  // a compute-only family keeps the probe off the desktop's graphics queue
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits) { g_qf = i; break; }
    if (g_qf == UINT32_MAX)
        for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qf = i; break; }
    free(qp);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, NULL);
    VkExtensionProperties * ex = calloc(ne, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, ex);
    const char * need[] = {"VK_KHR_acceleration_structure", "VK_KHR_ray_query", "VK_KHR_deferred_host_operations", "VK_KHR_shader_clock",
                           "VK_KHR_pipeline_executable_properties"};
    int missing = 0;
    for (size_t i = 0; i < sizeof need / sizeof *need; i++)
        if (!has_ext(ex, ne, need[i])) { printf("VERDICT extension: FAIL %s not supported by this device\n", need[i]); missing = 1; }
    free(ex);
    if (missing) exit(3);

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, &fexe};
    fclk.shaderSubgroupClock = fclk.shaderDeviceClock = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR frq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &fclk};
    frq.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR fas = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &frq};
    fas.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fas};
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    f2.features.shaderInt64 = VK_TRUE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_qf, qci.queueCount = 1, qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = sizeof need / sizeof *need, dci.ppEnabledExtensionNames = need;
    VK_CHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
    pCreateAS = (PFN_vkCreateAccelerationStructureKHR) vkGetDeviceProcAddr(g_dev, "vkCreateAccelerationStructureKHR");
    pDestroyAS = (PFN_vkDestroyAccelerationStructureKHR) vkGetDeviceProcAddr(g_dev, "vkDestroyAccelerationStructureKHR");
    pBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR) vkGetDeviceProcAddr(g_dev, "vkGetAccelerationStructureBuildSizesKHR");
    pCmdBuild = (PFN_vkCmdBuildAccelerationStructuresKHR) vkGetDeviceProcAddr(g_dev, "vkCmdBuildAccelerationStructuresKHR");
    pASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR) vkGetDeviceProcAddr(g_dev, "vkGetAccelerationStructureDeviceAddressKHR");
    if (!pCreateAS || !pDestroyAS || !pBuildSizes || !pCmdBuild || !pASAddr) { printf("VERDICT extension: FAIL acceleration-structure entry points missing\n"); exit(3); }

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, cpi.queueFamilyIndex = g_qf;
    VK_CHECK(vkCreateCommandPool(g_dev, &cpi, NULL, &g_pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g_pool, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, &g_cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(g_dev, &fci, NULL, &g_fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP, qpi.queryCount = 8;
    VK_CHECK(vkCreateQueryPool(g_dev, &qpi, NULL, &g_qp));

    VkDescriptorSetLayoutBinding b[4];
    b[0] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    for (int i = 1; i < 4; i++) b[i] = (VkDescriptorSetLayoutBinding){(uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 4, dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &g_dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1, pli.pSetLayouts = &g_dsl, pli.pushConstantRangeCount = 1, pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &g_pl));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1, dpi.poolSizeCount = 2, dpi.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &g_dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_dp, dai.descriptorSetCount = 1, dai.pSetLayouts = &g_dsl;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &dai, &g_ds));
}

// ---- buffers ------------------------------------------------------------------------------------------------------------
typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; void * map; VkDeviceAddress addr; } XBuf;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = g_mp.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}
static XBuf xbuf(VkDeviceSize size, VkBufferUsageFlags usage, int host) {
    XBuf b = {0};
    b.size = size ? size : 16;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b.size;
    bi.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = t;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
    if (host) VK_CHECK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    VkBufferDeviceAddressInfo da = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    da.buffer = b.buf;
    b.addr = vkGetBufferDeviceAddress(g_dev, &da);
    return b;
}
static void xfree(XBuf * b) {
    if (!b->buf) return;
    if (b->map) vkUnmapMemory(g_dev, b->mem);
    vkDestroyBuffer(g_dev, b->buf, NULL);
    vkFreeMemory(g_dev, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

// One command buffer, one submit, one fenced wait of at most 20 s. A timeout is reported and ends the run (no retry on a hung card).
static void cb_begin(void) {
    VK_CHECK(vkResetCommandBuffer(g_cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_cb, &bi));
}
static void cb_submit(void) {
    VK_CHECK(vkEndCommandBuffer(g_cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &g_cb;
    VK_CHECK(vkResetFences(g_dev, 1, &g_fence));
    VK_CHECK(vkQueueSubmit(g_q, 1, &si, g_fence));
    VkResult r = vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, 20ull * 1000000000ull);
    if (r == VK_TIMEOUT) { printf("VERDICT r11_rt: FAIL a submit did not finish in 20 s (device probably hung); stopping, nothing retried\n"); exit(5); }
    VK_CHECK(r);
}
static void upload(XBuf * dst, const void * src, VkDeviceSize n) {
    XBuf st = xbuf(n, 0, 1);
    memcpy(st.map, src, n);
    cb_begin();
    VkBufferCopy c = {0, 0, n};
    vkCmdCopyBuffer(g_cb, st.buf, dst->buf, 1, &c);
    cb_submit();
    xfree(&st);
}

// ---- acceleration structures --------------------------------------------------------------------------------------------
typedef struct { VkAccelerationStructureKHR as; XBuf buf; VkDeviceAddress addr; } AS;

static AS as_build(int top, VkDeviceAddress data_addr, uint32_t count, double * build_ms) {
    VkAccelerationStructureGeometryKHR g = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    if (top) {
        g.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        g.geometry.instances = (VkAccelerationStructureGeometryInstancesDataKHR){VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        g.geometry.instances.data.deviceAddress = data_addr;
    } else {
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        g.geometry.triangles = (VkAccelerationStructureGeometryTrianglesDataKHR){VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        g.geometry.triangles.vertexData.deviceAddress = data_addr;
        g.geometry.triangles.vertexStride = 12;
        g.geometry.triangles.maxVertex = 3 * count - 1;
        g.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }
    VkAccelerationStructureBuildGeometryInfoKHR bi = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1, bi.pGeometries = &g;
    VkAccelerationStructureBuildSizesInfoKHR sz = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pBuildSizes(g_dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &count, &sz);
    AS a = {0};
    a.buf = xbuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, 0);
    VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.buf.buf, ci.size = sz.accelerationStructureSize, ci.type = bi.type;
    VK_CHECK(pCreateAS(g_dev, &ci, NULL, &a.as));
    XBuf scratch = xbuf(sz.buildScratchSize + g_scratch_align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
    bi.dstAccelerationStructure = a.as;
    bi.scratchData.deviceAddress = (scratch.addr + g_scratch_align - 1) / g_scratch_align * g_scratch_align;
    VkAccelerationStructureBuildRangeInfoKHR range = {count, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR * pr = &range;
    const double t0 = now_ns();
    cb_begin();
    pCmdBuild(g_cb, 1, &bi, &pr);
    cb_submit();
    if (build_ms) *build_ms = (now_ns() - t0) * 1e-6;
    xfree(&scratch);
    VkAccelerationStructureDeviceAddressInfoKHR ai = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    ai.accelerationStructure = a.as;
    a.addr = pASAddr(g_dev, &ai);
    return a;
}
static void as_free(AS * a) {
    if (a->as) pDestroyAS(g_dev, a->as, NULL);
    xfree(&a->buf);
    memset(a, 0, sizeof *a);
}

typedef struct {
    AS blas, tlas;
    XBuf vtx, inst;
    R11V3 * cpu;      // the CPU copy of every triangle in world space (what brute force uses)
    uint32_t ntri;    // triangles in the CPU copy
    uint32_t nx, ny, ncell;
    int rot;
    double blas_ms, tlas_ms;
} Scene;

static void tlas_from_instances(Scene * s, const VkAccelerationStructureInstanceKHR * inst, uint32_t n) {
    s->inst = xbuf(n ? sizeof *inst * n : 64, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 0);
    if (n) upload(&s->inst, inst, sizeof *inst * n);
    s->tlas = as_build(1, s->inst.addr, n, &s->tlas_ms);
}
static VkAccelerationStructureInstanceKHR make_inst(VkDeviceAddress blas, float m[12]) {
    VkAccelerationStructureInstanceKHR i = {0};
    memcpy(&i.transform, m, sizeof m[0] * 12);
    i.mask = 0xff;
    i.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    i.accelerationStructureReference = blas;
    return i;
}

// vertices: ntri x 3 positions for the card; cpu: the copy brute force sees (the same pointer unless a break moved one on the card only).
static void scene_triangles(Scene * s, const R11V3 * card_vtx, const R11V3 * cpu_vtx, uint32_t ntri) {
    s->ntri = ntri;
    s->cpu = malloc(sizeof(R11V3) * 3 * (size_t) ntri);
    memcpy(s->cpu, cpu_vtx, sizeof(R11V3) * 3 * (size_t) ntri);
    s->vtx = xbuf(sizeof(R11V3) * 3 * (VkDeviceSize) ntri, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 0);
    upload(&s->vtx, card_vtx, sizeof(R11V3) * 3 * (size_t) ntri);
    s->blas = as_build(0, s->vtx.addr, ntri, &s->blas_ms);
    float id[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    VkAccelerationStructureInstanceKHR i = make_inst(s->blas.addr, id);
    tlas_from_instances(s, &i, 1);
}

static void scene_free(Scene * s) {
    as_free(&s->tlas);
    as_free(&s->blas);
    xfree(&s->vtx);
    xfree(&s->inst);
    free(s->cpu);
    memset(s, 0, sizeof *s);
}

// kind of scene by cell definition; moved >= 0 moves that triangle by (+0.5, +0.5) in the card's vertex buffer only (deliberate break).
static void scene_make(Scene * s, int scene, uint32_t n, int rot, int moved) {
    memset(s, 0, sizeof *s);
    if (scene == K_S0) {
        const R11V3 t[3] = {{0, 0, 0}, {(float) n, 0, 0}, {0, (float) n, 0}};
        scene_triangles(s, t, t, 1);
        s->nx = s->ny = n, s->ncell = 1;
    } else if (scene == K_S1 || scene == K_S2D || scene == K_S2A) {
        r11_grid_dims(n, &s->nx, &s->ny);
        s->ncell = n;
        R11V3 *cpu = malloc(sizeof(R11V3) * 3 * (size_t) n), *card = malloc(sizeof(R11V3) * 3 * (size_t) n);
        for (uint32_t k = 0; k < n; ++k) {
            if (scene == K_S1) r11_cell_tri(k, s->nx, 0, cpu + 3 * k);
            else r11_sliver_tri(k, s->nx, scene == K_S2A, cpu + 3 * k);
        }
        memcpy(card, cpu, sizeof(R11V3) * 3 * (size_t) n);
        if (moved >= 0) for (int v = 0; v < 3; ++v) card[3 * moved + v].x += 0.5f, card[3 * moved + v].y += 0.5f;
        scene_triangles(s, card, cpu, n);
        free(cpu), free(card);
    } else if (scene == K_S3) {
        // n instances of one unit triangle; rot = 180 degrees about each cell's centre (hit rule a + b >= 16).
        r11_grid_dims(n, &s->nx, &s->ny);
        s->ncell = n, s->rot = rot;
        const R11V3 unit[3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        s->vtx = xbuf(sizeof unit, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 0);
        upload(&s->vtx, unit, sizeof unit);
        s->blas = as_build(0, s->vtx.addr, 1, &s->blas_ms);
        VkAccelerationStructureInstanceKHR * in = calloc(n, sizeof *in);
        s->cpu = malloc(sizeof(R11V3) * 3 * (size_t) n);
        s->ntri = n;
        for (uint32_t k = 0; k < n; ++k) {
            const float cx = (float) (k % s->nx), cy = (float) (k / s->nx);
            float m[12] = {1, 0, 0, cx, 0, 1, 0, cy, 0, 0, 1, 0};
            if (rot) { m[0] = -1; m[5] = -1; m[3] = cx + 1; m[7] = cy + 1; }
            in[k] = make_inst(s->blas.addr, m);
            in[k].instanceCustomIndex = k & 0xffffff;
            r11_cell_tri(k, s->nx, rot, s->cpu + 3 * k);
        }
        tlas_from_instances(s, in, n);
        free(in);
    } else {  // K_EMPTY: a built top level with no instances; the BLAS exists only so the scene has the same shape
        s->nx = s->ny = 1, s->ncell = 1;
        s->ntri = 0;
        s->cpu = malloc(sizeof(R11V3));
        tlas_from_instances(s, NULL, 0);
    }
}

// ---- pipelines and their machine code ------------------------------------------------------------------------------------
typedef struct { VkPipeline p; char name[64]; char * asm_text; } Pipe;

static char * read_file(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * d = malloc(n + 1);
    if (fread(d, 1, n, f) != (size_t) n) { fclose(f); free(d); return NULL; }
    fclose(f);
    d[n] = 0;
    *len = n;
    return d;
}

static void capture_asm(Pipe * p) {
    PFN_vkGetPipelineExecutablePropertiesKHR gprops = (PFN_vkGetPipelineExecutablePropertiesKHR) vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR) vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    if (!gprops || !greps) return;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p->p;
    uint32_t ne = 0;
    gprops(g_dev, &pi, &ne, NULL);
    if (!ne) return;
    VkPipelineExecutablePropertiesKHR ep[4];
    for (int i = 0; i < 4; i++) ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (ne > 4) ne = 4;
    gprops(g_dev, &pi, &ne, ep);
    VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = p->p;
    uint32_t nr = 0;
    greps(g_dev, &ei, &nr, NULL);
    VkPipelineExecutableInternalRepresentationKHR * ir = calloc(nr ? nr : 1, sizeof *ir);
    for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    greps(g_dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; i++) ir[i].pData = calloc(1, ir[i].dataSize + 1);
    greps(g_dev, &ei, &nr, ir);
    size_t cap = 1 << 16, len = 0;
    char * txt = malloc(cap);
    txt[0] = 0;
    len = (size_t) snprintf(txt, cap, "# pipeline %s, executable \"%s\" (%s), subgroup size %u\n", p->name, ep[0].name, ep[0].description, ep[0].subgroupSize);
    for (uint32_t i = 0; i < nr; i++) {
        if (strstr(ir[i].name, "Assembly") || strstr(ir[i].name, "assembly")) {
            size_t need = strlen(ir[i].pData) + 64;
            while (len + need + 1 > cap) txt = realloc(txt, cap *= 2);
            len += (size_t) snprintf(txt + len, cap - len, "# representation: %s\n%s\n", ir[i].name, (char *) ir[i].pData);
        }
        free(ir[i].pData);
    }
    free(ir);
    p->asm_text = txt;
    if (g_out[0]) {
        char path[800];
        snprintf(path, sizeof path, "%s/asm", g_out);
        mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/asm/%s.s", g_out, p->name);
        FILE * f = fopen(path, "w");
        if (f) { fputs(txt, f); fclose(f); }
    }
    // The four instruction kinds the plan asks about, counted by name over the whole listing (loop weighting is the analysis step).
    static const char * kinds[] = {"image_bvh_intersect_ray", "image_bvh64_intersect_ray", "image_bvh8_intersect_ray",
                                   "image_bvh_dual_intersect_ray", "ds_bvh_stack"};
    printf("LIST %s:", p->name);
    for (size_t k = 0; k < sizeof kinds / sizeof *kinds; ++k) {
        int c = 0;
        for (const char * s = txt; (s = strstr(s, kinds[k])); s += strlen(kinds[k])) ++c;
        printf(" %s=%d", kinds[k], c);
    }
    printf(" (all 'bvh' mentions: ");
    int tot = 0;
    for (const char * s = txt; (s = strstr(s, "bvh")); s += 3) ++tot;
    printf("%d)%s\n", tot, tot ? "" : " NO RAY INSTRUCTION FOUND: the listing may not be the final machine code");
}

static Pipe pipe_make(const char * spv, const char * name) {
    Pipe p = {0};
    snprintf(p.name, sizeof p.name, "%s", name);
    char path[800];
    snprintf(path, sizeof path, "%s/%s", g_spv, spv);
    size_t len = 0;
    char * code = read_file(path, &len);
    if (!code) { printf("VERDICT %s: FAIL cannot read %s\n", name, path); exit(1); }
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = len, smi.pCode = (const uint32_t *) code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(g_dev, &smi, NULL, &sm));
    free(code);
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = sm, ci.stage.pName = "main";
    ci.layout = g_pl;
    VkResult r = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &p.p);
    vkDestroyShaderModule(g_dev, sm, NULL);
    if (r != VK_SUCCESS) { printf("VERDICT %s: FAIL pipeline creation returned %d\n", name, (int) r); exit(1); }
    capture_asm(&p);
    return p;
}

// ---- running a shader ------------------------------------------------------------------------------------------------------
static void bind(const AS * tlas, XBuf * b1, XBuf * b2) {
    VkWriteDescriptorSetAccelerationStructureKHR wa = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    wa.accelerationStructureCount = 1, wa.pAccelerationStructures = &tlas->as;
    VkDescriptorBufferInfo bi[3];
    VkWriteDescriptorSet w[4];
    memset(w, 0, sizeof w);
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, w[0].pNext = &wa, w[0].dstSet = g_ds, w[0].dstBinding = 0;
    w[0].descriptorCount = 1, w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    XBuf * bufs[3] = {b1, b2, b2};
    for (int i = 0; i < 3; i++) {
        bi[i] = (VkDescriptorBufferInfo){bufs[i]->buf, 0, VK_WHOLE_SIZE};
        w[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, w[i + 1].dstSet = g_ds, w[i + 1].dstBinding = i + 1;
        w[i + 1].descriptorCount = 1, w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, w[i + 1].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(g_dev, 4, w, 0, NULL);
}

// Binds (tlas, b1 at binding 1, b2 at bindings 2 and 3), optionally zero-fills the first `zero_bytes` of b1, dispatches `groups`, and returns
// the GPU time of the dispatch in ms from timestamps.
static double dispatch(const Pipe * p, const AS * tlas, XBuf * b1, XBuf * b2, uint32_t groups, const void * pc, size_t pcn, size_t zero_bytes) {
    bind(tlas, b1, b2);
    cb_begin();
    vkCmdResetQueryPool(g_cb, g_qp, 0, 2);  // in the command buffer: hostQueryReset is not enabled
    if (zero_bytes) {
        vkCmdFillBuffer(g_cb, b1->buf, 0, zero_bytes, 0);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(g_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->p);
    vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl, 0, 1, &g_ds, 0, NULL);
    vkCmdPushConstants(g_cb, g_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t) pcn, pc);
    vkCmdWriteTimestamp(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qp, 0);
    vkCmdDispatch(g_cb, groups, 1, 1);
    vkCmdWriteTimestamp(g_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, 1);
    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(g_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);
    cb_submit();
    uint64_t ts[2];
    VK_CHECK(vkGetQueryPoolResults(g_dev, g_qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double) (ts[1] - ts[0]) * g_ts_ns * 1e-6;
}

// ---- controls on a scene -----------------------------------------------------------------------------------------------------
static uint32_t g_nsample = 4096;
static int g_fail;
static int g_clock_cells, g_clock_flat_failed, g_strict_clock;

// Builds the 4096-ray sample: even rays at a triangle's centroid, odd rays at a hash point inside its cell box; `force` (>= 0) gets
// rays 0 and 1 (the triangle a break moved).
static uint32_t check_rays(const Scene * s, float * rays, int force) {
    const uint32_t n = g_nsample;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t ti = (uint32_t) (((uint64_t) r11_hash32(i * 7u + 3u)) % (s->ntri ? s->ntri : 1));
        if (s->ntri <= n) ti = i % (s->ntri ? s->ntri : 1);
        if (force >= 0 && i < 2) ti = (uint32_t) force;
        float x, y;
        if (s->ntri == 0) { x = 0.5f; y = 0.5f; }
        else if (!(i & 1) || (force >= 0 && i == 1)) {
            const R11V3 * v = s->cpu + 3 * (size_t) ti;
            x = (v[0].x + v[1].x + v[2].x) / 3.0f, y = (v[0].y + v[1].y + v[2].y) / 3.0f;
        } else {
            const R11V3 * v = s->cpu + 3 * (size_t) ti;
            const float minx = fminf(v[0].x, fminf(v[1].x, v[2].x)), maxx = fmaxf(v[0].x, fmaxf(v[1].x, v[2].x));
            const float miny = fminf(v[0].y, fminf(v[1].y, v[2].y)), maxy = fmaxf(v[0].y, fmaxf(v[1].y, v[2].y));
            x = minx + (maxx - minx) * ((r11_hash32(i * 2u + 11u) & 0xffffu) / 65536.0f);
            y = miny + (maxy - miny) * ((r11_hash32(i * 2u + 12u) & 0xffffu) / 65536.0f);
        }
        float * r = rays + 8 * (size_t) i;
        r[0] = x, r[1] = y, r[2] = 1.0f, r[3] = 0.0f, r[4] = 0.0f, r[5] = 0.0f, r[6] = -1.0f, r[7] = 100.0f;
    }
    return n;
}

static Pipe g_pcheck, g_prate, g_pchain;
static XBuf g_rays, g_res, g_out_buf, g_zero;

// Returns the number of rays on which the card and the CPU brute force disagree; first_bad gets the first one.
static uint32_t scene_check(const Scene * s, int force, uint32_t * first_bad) {
    const uint32_t n = g_nsample;
    float * rays = malloc(sizeof(float) * 8 * n);
    check_rays(s, rays, force);
    memcpy(g_rays.map, rays, sizeof(float) * 8 * n);
    struct { uint32_t n, first; } pc = {n, 0};
    dispatch(&g_pcheck, &s->tlas, &g_rays, &g_res, (n + 63) / 64, &pc, sizeof pc, 0);
    double * want = malloc(sizeof(double) * n);
    for (uint32_t i = 0; i < n; ++i) {
        const double o[3] = {rays[8 * i], rays[8 * i + 1], rays[8 * i + 2]}, d[3] = {0, 0, -1};
        int prim;
        want[i] = r11_brute(s->cpu, s->ntri, o, d, 0.0, 100.0, &prim);
    }
    uint32_t hits_cpu = 0;
    for (uint32_t i = 0; i < n; ++i) hits_cpu += want[i] >= 0;
    const uint32_t bad = r11_compare((const float *) g_res.map, want, n, first_bad);
    printf("CHECK scene rays=%u cpu_hits=%u mismatches=%u\n", n, hits_cpu, bad);
    free(rays), free(want);
    return bad;
}

// ---- cells ----------------------------------------------------------------------------------------------------------------------
typedef struct {
    uint32_t nx, ny, ncell, rpt, flags, seed;
    float tmax;
    uint32_t kind;
    float ext;
    uint32_t total_rays;
} RatePC;

static int rate_cell(const RateCell * c, const Scene * s, int reps, int quick, FILE * csv) {
    const uint32_t threads = c->threads, groups = threads / 64;
    RatePC pc = {s->nx, s->ny, s->ncell ? s->ncell : 1, 1, c->flags, 0x9e3779b9u, c->tmax, c->scene == K_S2D || c->scene == K_S2A ? 1u : 0u,
                 (float) s->nx, 0};
    pc.total_rays = pc.ncell * 256u;
    // Calibrate: double rays per invocation until one dispatch takes >= 50 ms (or 4096 per invocation); a first dispatch is never long.
    double ms = 0;
    for (;;) {
        ms = dispatch(&g_prate, &s->tlas, &g_out_buf, &g_zero, groups, &pc, sizeof pc, 16);
        if (ms >= 50.0 || pc.rpt >= 4096 || (quick && pc.rpt >= 4)) break;
        pc.rpt = ms < 1.0 ? pc.rpt * 16 : pc.rpt * 2;
        if (pc.rpt > 4096) pc.rpt = 4096;
    }
    R11Rate rr = {pc.nx, pc.ny, pc.ncell, pc.rpt, pc.flags, pc.seed, pc.kind, pc.total_rays, threads, s->rot, c->tmax};
    const int exact_ok = pc.kind == 0 && (c->scene == K_S1 || c->scene == K_S3);  // S0 has its own analytic control
    const uint64_t want = c->scene == K_EMPTY ? 0 : exact_ok ? r11_rate_expected_hits(&rr) : 0;
    double t[256], clk[256];
    int bad_hits = 0;
    uint32_t last_hits = 0;
    if (reps > 250) reps = 250;
    if (reps < 1) reps = 1;
    for (int r = 0; r < reps; ++r) {
        t[r] = dispatch(&g_prate, &s->tlas, &g_out_buf, &g_zero, groups, &pc, sizeof pc, 16);
        const uint32_t * o = (const uint32_t *) g_out_buf.map;
        last_hits = o[0];
        const uint64_t cyc = (uint64_t) o[4] | ((uint64_t) o[5] << 32);
        clk[r] = o[6] ? cyc * 100.0 / o[6] : -1;  // cycles / (ticks x 10 ns) in MHz
        if ((exact_ok || c->scene == K_EMPTY) && last_hits != want) ++bad_hits;
    }
    const double rays = (double) threads * pc.rpt, med = median(t, reps);
    double mn = t[0], mx = t[0];
    for (int r = 1; r < reps; ++r) { mn = t[r] < mn ? t[r] : mn; mx = t[r] > mx ? t[r] : mx; }
    const int flat = r11_clock_flat(clk[0], clk[reps - 1]);
    const char * hv = (exact_ok || c->scene == K_EMPTY) ? (bad_hits ? "FAIL" : "PASS") : "n/a";
    printf("CELL %s rays=%.0f rpt=%u reps=%d median_ms=%.3f min_ms=%.3f max_ms=%.3f median_Mrays_s=%.1f best_Mrays_s=%.1f worst_Mrays_s=%.1f "
           "hits_card=%u hits_expected=%s%llu hits_exact=%s clock_first_MHz=%.0f clock_last_MHz=%.0f clock_flat=%s blas_ms=%.2f tlas_ms=%.2f\n",
           c->name, rays, pc.rpt, reps, med, mn, mx, rays / med / 1e3, rays / mx / 1e3, rays / mn / 1e3, last_hits, exact_ok || c->scene == K_EMPTY ? "" : "(none) ",
           (unsigned long long) want, hv, clk[0], clk[reps - 1], flat ? "yes" : "NO", s->blas_ms, s->tlas_ms);
    if (csv) for (int r = 0; r < reps; ++r) fprintf(csv, "%s,%d,%.6f,%.1f\n", c->name, r, t[r], clk[r]);
    ++g_clock_cells;
    if (!flat) ++g_clock_flat_failed;
    if (r11_clock_cell_fail(flat, g_strict_clock)) { ++g_fail; printf("CTRLFAIL %s: core clock moved more than 3 %% between the first and last repeat (--strict-clock)\n", c->name); }
    if (bad_hits) { ++g_fail; printf("CTRLFAIL %s: the card's hit count differs from the CPU's exact count on %d repeat(s)\n", c->name, bad_hits); }
    return bad_hits == 0;
}

typedef struct { uint32_t n, dep, lanes, nx, ny, ncell, seed; float tmax; } ChainPC;

// One chain cell: returns the realtime ticks (10 ns each) of the loop on the one measuring lane.
static double chain_run(const Scene * s, uint32_t n, uint32_t lanes, uint32_t dep, double * clock_mhz, uint32_t * hits) {
    ChainPC pc = {n, dep, lanes, s->nx, s->ny, s->ncell, 0x51, 1e4f};
    dispatch(&g_pchain, &s->tlas, &g_out_buf, &g_zero, 1, &pc, sizeof pc, 16);
    const uint32_t * o = (const uint32_t *) g_out_buf.map;
    const uint64_t cyc = (uint64_t) o[0] | ((uint64_t) o[1] << 32);
    *clock_mhz = o[2] ? cyc * 100.0 / o[2] : -1;
    *hits = o[3];
    return (double) o[2];
}

static int chain_cell(const ChainCell * c, const Scene * s, int reps) {
    const uint32_t N = 4096;
    double ch[64], wv[64], clk = 0;
    uint32_t h;
    if (reps > 64) reps = 64;
    if (reps < 1) reps = 1;
    chain_run(s, N, 1, 1, &clk, &h);  // warm
    for (int r = 0; r < reps; ++r) ch[r] = chain_run(s, N, 1, 1, &clk, &h);
    const double clk_chain = clk;
    for (int r = 0; r < reps; ++r) wv[r] = chain_run(s, N / 32, 32, 0, &clk, &h);  // the same 4096 rays, independent, on one whole wave
    const double mc = median(ch, reps), mw = median(wv, reps);
    const int ok = r11_chain_ok(mc, mw);
    printf("CELL %s rays=%u reps=%d chain_ns_per_ray=%.1f chain_cycles_per_ray=%.0f chain_clock_MHz=%.0f wave_ns_per_ray=%.2f chain_over_wave=%.2f chain_check=%s\n",
           c->name, N, reps, mc * 10.0 / N, mc * 10.0 / N * clk_chain / 1e3, clk_chain, mw * 10.0 / N, mw > 0 ? mc / mw : -1, ok ? "PASS" : "FAIL");
    if (!ok) { ++g_fail; printf("CTRLFAIL %s: chain is not >= 3x the wave: the dependency broke, latency cells void\n", c->name); }
    // Deliberate break (plan 2): the same "chain" with every lane active (32 independent chains in one wave).
    if (c->n == 4096) {
        double bw[64];
        for (int r = 0; r < reps; ++r) bw[r] = chain_run(s, N / 32, 32, 1, &clk, &h);
        const int broke_ok = r11_chain_ok(median(bw, reps), mw);
        printf("BREAK chain_all_lanes: control reported %s -> %s\n", broke_ok ? "PASS" : "FAIL", broke_ok ? "PROBE INVALID (the chain check cannot see it)" : "ok (FAIL as required)");
        if (broke_ok) ++g_fail;
    }
    return ok;
}

static void do_plan(int big, int quick) {
    build_cell_list(big);
    int chain_reps = 15;
    double est = 0;
    for (int i = 0; i < g_ncells; ++i) est += 1.5 + (quick ? 5 : 30) * 0.075;  // calibration + reps x about 75 ms
    est += g_nchain * (2.0 * chain_reps * 0.002 + 0.5);
    printf("PLAN r11_rt: %d rate cells + %d chain cells (+ 4 control scenes: S0 analytic, moved-triangle break, extra-instance break, empty top level)\n", g_ncells, g_nchain);
    printf("PLAN estimated card load %.0f s (about %.1f min) at 30 repeats; wall time is longer because every scene group waits for a hotspot below 60 C (up to 120 s each)\n", est, est / 60);
    printf("PLAN timing: GPU timestamps around each dispatch plus in-shader shader-clock cycles and the 100 MHz realtime counter\n");
    printf("PLAN no acceleration-structure bytes are read (open question 3: default no); the listing is the pipeline's Assembly representation\n");
    for (int i = 0; i < g_ncells; ++i) printf("PLANCELL %s\n", g_cells[i].name);
    for (int i = 0; i < g_nchain; ++i) printf("PLANCELL %s\n", g_chain[i].name);
}

static int cpu_checks(void) {
    int bad = 0;
    build_cell_list(0);
    if (g_ncells < 50) { printf("cpu-check FAIL: too few cells (%d)\n", g_ncells); bad++; }
    for (int i = 0; i < g_ncells; ++i)
        for (int j = i + 1; j < g_ncells; ++j)
            if (!strcmp(g_cells[i].name, g_cells[j].name)) { printf("cpu-check FAIL: duplicate cell name %s\n", g_cells[i].name); bad++; }
    const char * spvs[] = {"rt_rate.spv", "rt_check.spv", "rt_chain.spv"};
    for (int i = 0; i < 3; ++i) {
        char p[800];
        snprintf(p, sizeof p, "%s/%s", g_spv[0] ? g_spv : ".", spvs[i]);
        size_t n;
        char * d = read_file(p, &n);
        if (!d || n < 20 || ((uint32_t *) d)[0] != 0x07230203u) { printf("cpu-check FAIL: %s is not a SPIR-V file\n", p); bad++; }
        else printf("cpu-check ok: %s (%zu bytes, SPIR-V magic)\n", p, n);
        free(d);
    }
    printf("cpu-check: %d cells, %d chain cells, %d problem(s)\n", g_ncells, g_nchain, bad);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    int mode = 0, reps = 30, quick = 0, big = 0;
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
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--quick")) quick = 1, g_nsample = 512;
        else if (!strcmp(argv[i], "--big")) big = 1;
        else if (!strcmp(argv[i], "--strict-clock")) g_strict_clock = 1;
        else { fprintf(stderr, "usage: r11_rt --plan | --cpu-checks | --run [--out DIR] [--reps N] [--only TEXT] [--quick] [--big] [--strict-clock]\n"); return 2; }
    }
    if (mode == 1) { do_plan(big, quick); return 0; }
    if (mode == 2) return cpu_checks();
    if (mode != 3) { fprintf(stderr, "usage: r11_rt --plan | --cpu-checks | --run [...]\n"); return 2; }

    build_cell_list(big);
    vk_init();
    if (g_out[0]) mkdir(g_out, 0755);
    g_pcheck = pipe_make("rt_check.spv", "rt_check");
    g_prate = pipe_make("rt_rate.spv", "rt_rate");
    g_pchain = pipe_make("rt_chain.spv", "rt_chain");
    g_rays = xbuf(sizeof(float) * 8 * 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    g_res = xbuf(sizeof(float) * 4 * 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    g_out_buf = xbuf(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    g_zero = xbuf(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1);
    memset(g_zero.map, 0, 4096);  // the chain shader's run-time zero: the compiler cannot know it is 0.0
    FILE * csv = NULL;
    if (g_out[0]) { char p[800]; snprintf(p, sizeof p, "%s/rt_repeats.csv", g_out); csv = fopen(p, "w"); if (csv) fprintf(csv, "cell,repeat,gpu_ms,clock_MHz\n"); }
    printf("CLOCKS before: hotspot %.0f C, power %.0f W (clocks come from r11_sampler)\n", hotspot_c(), read_num(g_hwmon, "power1_average", 1e-6));

    // Controls first. 1) S0 analytic count. 2) The moved-triangle break. 3) The extra-instance break. Their failure ends the run.
    thermal_gate();
    {
        Scene s;
        scene_make(&s, K_S0, 64, 0, -1);
        float rays[8 * 4096];
        for (uint32_t j = 0; j < 64; ++j)
            for (uint32_t i = 0; i < 64; ++i) {
                float * r = rays + 8 * (j * 64 + i);
                r[0] = i + 0.25f, r[1] = j + 0.25f, r[2] = 1, r[3] = 0, r[4] = 0, r[5] = 0, r[6] = -1, r[7] = 100;
            }
        memcpy(g_rays.map, rays, sizeof rays);
        struct { uint32_t n, first; } pc = {4096, 0};
        dispatch(&g_pcheck, &s.tlas, &g_rays, &g_res, 64, &pc, sizeof pc, 0);
        uint32_t hits = 0;
        for (int i = 0; i < 4096; ++i) hits += ((const float *) g_res.map)[4 * i] >= 0;
        const int ok = hits == r11_s0_expected(64);
        printf("CONTROL S0_analytic hits=%u expected=%llu %s\n", hits, (unsigned long long) r11_s0_expected(64), ok ? "PASS" : "FAIL");
        if (!ok) ++g_fail;
        scene_free(&s);
    }
    {
        Scene s;
        scene_make(&s, K_S1, 4096, 0, 1234);  // card scene: triangle 1234 moved; CPU scene: original
        uint32_t first = 0;
        const uint32_t bad = scene_check(&s, 1234, &first);
        printf("BREAK moved_triangle: control reported %s -> %s\n", bad ? "FAIL" : "PASS", bad ? "ok (FAIL as required)" : "PROBE INVALID (the CPU comparison cannot see it)");
        if (!bad) ++g_fail;
        scene_free(&s);
    }
    {
        Scene s;
        const uint32_t M = 256;
        scene_make(&s, K_S3, M + 1, 0, -1);  // M + 1 instances; the hand count is M
        float rays[8 * 4096];
        uint32_t n = M + 1;
        for (uint32_t k = 0; k < n; ++k) {
            float * r = rays + 8 * k;
            r[0] = (float) (k % s.nx) + 0.25f, r[1] = (float) (k / s.nx) + 0.25f, r[2] = 1, r[3] = 0, r[4] = 0, r[5] = 0, r[6] = -1, r[7] = 100;
        }
        memcpy(g_rays.map, rays, sizeof(float) * 8 * n);
        struct { uint32_t n, first; } pc = {n, 0};
        dispatch(&g_pcheck, &s.tlas, &g_rays, &g_res, 5, &pc, sizeof pc, 0);
        uint32_t hits = 0;
        for (uint32_t i = 0; i < n; ++i) hits += ((const float *) g_res.map)[4 * i] >= 0;
        const int analytic_ok = hits == M;
        printf("BREAK extra_instance: card hits %u, hand count %u: control reported %s -> %s\n", hits, M, analytic_ok ? "PASS" : "FAIL", analytic_ok ? "PROBE INVALID" : "ok (FAIL as required)");
        if (analytic_ok) ++g_fail;
        scene_free(&s);
    }
    if (g_fail) {
        printf("VERDICT r11_rt: FAIL a first control or break did not behave; no timing cell is read\n");
        return 1;
    }

    // Scene groups: one scene built once per group, its check, then every cell that uses it.
    int last_scene = -1;
    uint32_t last_n = 0;
    int last_rot = -1;
    Scene s;
    memset(&s, 0, sizeof s);
    for (int i = 0; i < g_ncells; ++i) {
        const RateCell * c = &g_cells[i];
        if (only && !strstr(c->name, only)) continue;
        if (c->scene != last_scene || c->n != last_n || c->rot != last_rot) {
            if (last_scene >= 0) scene_free(&s);
            thermal_gate();
            scene_make(&s, c->scene, c->n, c->rot, -1);
            last_scene = c->scene, last_n = c->n, last_rot = c->rot;
            if (c->scene != K_EMPTY) {
                uint32_t first = 0;
                const uint32_t bad = scene_check(&s, -1, &first);
                printf("CONTROL scene_check %s n=%u %s\n", c->scene == K_S0 ? "S0" : c->name, c->n, bad ? "FAIL" : "PASS");
                if (bad) { ++g_fail; printf("CTRLFAIL scene n=%u: card and CPU disagree (first ray %u); its cells are not interpreted\n", c->n, first); }
            }
        }
        rate_cell(c, &s, reps, quick, csv);
    }
    if (last_scene >= 0) scene_free(&s);
    for (int i = 0; i < g_nchain; ++i) {
        const ChainCell * c = &g_chain[i];
        if (only && !strstr(c->name, only)) continue;
        thermal_gate();
        scene_make(&s, c->n == 1 ? K_S0 : K_S1, c->n == 1 ? 64 : c->n, 0, -1);
        chain_cell(c, &s, reps < 15 ? reps : 15);
        scene_free(&s);
    }
    if (csv) fclose(csv);
    printf("CLOCKS after: hotspot %.0f C (highest seen %.0f C), power %.0f W\n", hotspot_c(), g_hottest, read_num(g_hwmon, "power1_average", 1e-6));
    printf("SUMMARY controls_or_breaks_failed=%d clock_flat_failed=%d/%d strict_clock=%d\n", g_fail, g_clock_flat_failed, g_clock_cells, g_strict_clock);
    if (g_fail) printf("VERDICT r11_rt: FAIL see CTRLFAIL / BREAK lines\n");
    else if (g_clock_flat_failed) printf("VERDICT r11_rt: PASS (unpinned clock ramp: %d/%d cells >3%% between first and last repeat; rates are at a drifting clock, read the sampler record)\n", g_clock_flat_failed, g_clock_cells);
    else printf("VERDICT r11_rt: PASS all controls passed and every deliberate break failed as required\n");
    return g_fail ? 1 : 0;
}
