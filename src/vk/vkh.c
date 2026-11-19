#define _GNU_SOURCE
#include "vkh.h"

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

Ctx G;

double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

double median(double *v, int n) {
    if (n <= 0) return NAN;
    double *c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

double vmin(const double *v, int n) {
    double m = INFINITY;
    for (int i = 0; i < n; i++) m = v[i] < m ? v[i] : m;
    return m;
}

double vmax(const double *v, int n) {
    double m = -INFINITY;
    for (int i = 0; i < n; i++) m = v[i] > m ? v[i] : m;
    return m;
}

uint64_t rng_next(uint64_t *s) {
    // splitmix64: fixed seeds give the same data on every run, so result files can be compared across runs.
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static double read_num(const char *path, double scale) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    double v = -1;
    if (fscanf(f, "%lf", &v) != 1) v = -1;
    fclose(f);
    return v < 0 ? -1 : v * scale;
}

static int read_star_mhz(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[128];
    int mhz = -1;
    while (fgets(line, sizeof line, f)) {
        if (strchr(line, '*')) {
            char *c = strchr(line, ':');
            if (c) mhz = atoi(c + 1);
        }
    }
    fclose(f);
    return mhz;
}

static void find_sysfs(const VkPhysicalDevicePCIBusInfoPropertiesEXT *pci) {
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x", pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    G.drm_dev[0] = G.hwmon[0] = 0;
    DIR *d = opendir("/sys/class/drm");
    struct dirent *e;
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char p[300], real[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, real)) continue;
        const char *base = strrchr(real, '/');
        if (base && !strcmp(base + 1, want)) snprintf(G.drm_dev, sizeof G.drm_dev, "%s", p);
    }
    if (d) closedir(d);
    if (!G.drm_dev[0]) return;
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", G.drm_dev);
    d = opendir(hp);
    while (d && (e = readdir(d))) {
        if (!strncmp(e->d_name, "hwmon", 5)) snprintf(G.hwmon, sizeof G.hwmon, "%s/%s", hp, e->d_name);
    }
    if (d) closedir(d);
}

static int has_ext(VkExtensionProperties *ex, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(ex[i].extensionName, name)) return 1;
    return 0;
}

void vkh_init(int quiet) {
    memset(&G, 0, sizeof G);
    const char *o = getenv("VKS_OUT");
    snprintf(G.out_dir, sizeof G.out_dir, "%s", o ? o : "");
    const char *s = getenv("VKS_SPV");
    snprintf(G.spv_dir, sizeof G.spv_dir, "%s", s ? s : "build/spv");

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdna4-lane9";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &G.inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(G.inst, &n, NULL);
    VkPhysicalDevice *devs = calloc(n, sizeof *devs);
    vkEnumeratePhysicalDevices(G.inst, &n, devs);
    G.pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == VKH_VENDOR && p.deviceID == VKH_DEVICE) {
            G.pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!G.pd) {
        printf("VERDICT device: FAIL no Vulkan device %04x:%04x (RX 9070 XT); refusing to run elsewhere\n", VKH_VENDOR,
               VKH_DEVICE);
        exit(1);
    }
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(G.pd, &p2);
    G.ts_period_ns = p2.properties.limits.timestampPeriod;
    find_sysfs(&pci);
    if (!quiet)
        printf("DEVICE %s (%04x:%04x) PCI %04x:%02x:%02x.%x driver %s %s; sysfs %s hwmon %s\n",
               p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID, pci.pciDomain, pci.pciBus,
               pci.pciDevice, pci.pciFunction, drv.driverName, drv.driverInfo, G.drm_dev[0] ? G.drm_dev : "(not found)",
               G.hwmon[0] ? G.hwmon : "(not found)");
    if (strcmp(drv.driverName, "radv")) {
        printf("VERDICT device: FAIL driver is %s, not radv\n", drv.driverName);
        exit(1);
    }
    vkGetPhysicalDeviceMemoryProperties(G.pd, &G.memprops);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, qp);
    // A compute-only family keeps the probes off the desktop's graphics queue.
    G.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            qp[i].timestampValidBits) {
            G.qf = i;
            break;
        }
    if (G.qf == UINT32_MAX)
        for (uint32_t i = 0; i < nq; i++)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                G.qf = i;
                break;
            }
    free(qp);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(G.pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(G.pd, NULL, &ne, ex);
    const char *want[] = {"VK_KHR_cooperative_matrix", "VK_EXT_shader_float8", "VK_KHR_shader_bfloat16",
                          "VK_KHR_shader_clock", "VK_KHR_pipeline_executable_properties"};
    const char *en[8];
    uint32_t nen = 0;
    for (size_t i = 0; i < sizeof want / sizeof *want; i++) {
        if (has_ext(ex, ne, want[i]))
            en[nen++] = want[i];
        else
            printf("WARN extension %s not supported\n", want[i]);
    }
    free(ex);

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, &fexe};
    fclk.shaderSubgroupClock = fclk.shaderDeviceClock = VK_TRUE;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR fbf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                                                      &fclk};
    fbf.shaderBFloat16Type = fbf.shaderBFloat16DotProduct = fbf.shaderBFloat16CooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceShaderFloat8FeaturesEXT ff8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &fbf};
    ff8.shaderFloat8 = ff8.shaderFloat8CooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                                                        &ff8};
    fcm.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fcm};
    f13.subgroupSizeControl = f13.computeFullSubgroups = f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderFloat16 = f12.shaderInt8 = f12.storageBuffer8BitAccess = f12.vulkanMemoryModel =
        f12.vulkanMemoryModelDeviceScope = f12.shaderSubgroupExtendedTypes = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderInt64 = f2.features.shaderInt16 = VK_TRUE;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = G.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nen;
    dci.ppEnabledExtensionNames = en;
    VK_CHECK(vkCreateDevice(G.pd, &dci, NULL, &G.dev));
    vkGetDeviceQueue(G.dev, G.qf, 0, &G.q);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = G.qf;
    VK_CHECK(vkCreateCommandPool(G.dev, &cpi, NULL, &G.pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = G.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(G.dev, &cai, &G.cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(G.dev, &fci, NULL, &G.fence));
    G.qp_count = 512;
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = G.qp_count;
    VK_CHECK(vkCreateQueryPool(G.dev, &qpi, NULL, &G.qp));

    VkDescriptorSetLayoutBinding b[VKH_BINDINGS];
    for (int i = 0; i < VKH_BINDINGS; i++) {
        b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                              VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    }
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = VKH_BINDINGS;
    dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(G.dev, &dli, NULL, &G.dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, VKH_PUSH_BYTES};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &G.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, &G.pl));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VKH_BINDINGS};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = G.dp;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &G.dsl;
    VK_CHECK(vkAllocateDescriptorSets(G.dev, &dai, &G.ds));
    G.dummy = buf_create(4096, 0);
    bind_buffers(NULL, NULL, NULL, NULL);
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < G.memprops.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = G.memprops.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

Buf buf_create(VkDeviceSize size, int host_visible) {
    Buf b = {0};
    b.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(G.dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(G.dev, b.buf, &mr);
    uint32_t t;
    if (host_visible) {
        // Host-visible buffers are only staging/readback: system memory, cached when possible.
        t = mem_type(mr.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (t == UINT32_MAX)
            t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         0);
    } else {
        t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    }
    if (t == UINT32_MAX) {
        fprintf(stderr, "no memory type\n");
        exit(3);
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = t;
    VK_CHECK(vkAllocateMemory(G.dev, &ai, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(G.dev, b.buf, b.mem, 0));
    if (host_visible) VK_CHECK(vkMapMemory(G.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

void buf_destroy(Buf *b) {
    if (!b->buf) return;
    if (b->map) vkUnmapMemory(G.dev, b->mem);
    vkDestroyBuffer(G.dev, b->buf, NULL);
    vkFreeMemory(G.dev, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

static void submit_wait(void) {
    VK_CHECK(vkEndCommandBuffer(G.cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &G.cb;
    VK_CHECK(vkQueueSubmit(G.q, 1, &si, G.fence));
    VK_CHECK(vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(G.dev, 1, &G.fence));
}

static void begin_cb(void) {
    VK_CHECK(vkResetCommandBuffer(G.cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(G.cb, &bi));
}

#define STAGE_CHUNK (64ull << 20)

void buf_upload(Buf *dst, const void *src, VkDeviceSize size) {
    Buf st = buf_create(size < STAGE_CHUNK ? size : STAGE_CHUNK, 1);
    for (VkDeviceSize off = 0; off < size; off += STAGE_CHUNK) {
        VkDeviceSize n = size - off < STAGE_CHUNK ? size - off : STAGE_CHUNK;
        memcpy(st.map, (const char *)src + off, n);
        begin_cb();
        VkBufferCopy c = {0, off, n};
        vkCmdCopyBuffer(G.cb, st.buf, dst->buf, 1, &c);
        submit_wait();
    }
    buf_destroy(&st);
}

void buf_download(const Buf *src, void *dst, VkDeviceSize size) {
    Buf st = buf_create(size < STAGE_CHUNK ? size : STAGE_CHUNK, 1);
    for (VkDeviceSize off = 0; off < size; off += STAGE_CHUNK) {
        VkDeviceSize n = size - off < STAGE_CHUNK ? size - off : STAGE_CHUNK;
        begin_cb();
        VkBufferCopy c = {off, 0, n};
        vkCmdCopyBuffer(G.cb, src->buf, st.buf, 1, &c);
        submit_wait();
        memcpy((char *)dst + off, st.map, n);
    }
    buf_destroy(&st);
}

void bind_buffers(Buf *b0, Buf *b1, Buf *b2, Buf *b3) {
    Buf *bs[VKH_BINDINGS] = {b0, b1, b2, b3};
    VkDescriptorBufferInfo bi[VKH_BINDINGS];
    VkWriteDescriptorSet w[VKH_BINDINGS];
    VK_CHECK(vkDeviceWaitIdle(G.dev));
    for (int i = 0; i < VKH_BINDINGS; i++) {
        Buf *b = bs[i] ? bs[i] : &G.dummy;
        bi[i] = (VkDescriptorBufferInfo){b->buf, 0, VK_WHOLE_SIZE};
        w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = G.ds;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(G.dev, VKH_BINDINGS, w, 0, NULL);
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *d = malloc(n + 1);
    if (fread(d, 1, n, f) != (size_t)n) {
        fclose(f);
        free(d);
        return NULL;
    }
    d[n] = 0;
    fclose(f);
    *len = n;
    return d;
}

static void capture(Pipe *p) {
    PFN_vkGetPipelineExecutablePropertiesKHR gprops =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(G.dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            G.dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gstats =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(G.dev, "vkGetPipelineExecutableStatisticsKHR");
    if (!gprops || !greps) return;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p->pipe;
    uint32_t ne = 0;
    gprops(G.dev, &pi, &ne, NULL);
    if (!ne) return;
    VkPipelineExecutablePropertiesKHR ep[4];
    for (int i = 0; i < 4; i++) ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (ne > 4) ne = 4;
    gprops(G.dev, &pi, &ne, ep);
    p->sg_size = ep[0].subgroupSize;
    VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = p->pipe;
    ei.executableIndex = 0;

    size_t cap = 1 << 16, len = 0;
    char *txt = malloc(cap);
    txt[0] = 0;
#define APPEND(...)                                                                                                    \
    do {                                                                                                               \
        int need_ = snprintf(NULL, 0, __VA_ARGS__);                                                                    \
        while (len + need_ + 1 > cap) txt = realloc(txt, cap *= 2);                                                    \
        len += snprintf(txt + len, cap - len, __VA_ARGS__);                                                            \
    } while (0)
    APPEND("# pipeline %s, executable \"%s\" (%s), subgroup size %u\n", p->name, ep[0].name, ep[0].description,
           ep[0].subgroupSize);
    if (gstats) {
        uint32_t ns = 0;
        gstats(G.dev, &ei, &ns, NULL);
        VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof *st);
        for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        gstats(G.dev, &ei, &ns, st);
        for (uint32_t i = 0; i < ns; i++) {
            if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
                APPEND("# stat %s = %llu\n", st[i].name, (unsigned long long)st[i].value.u64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
                APPEND("# stat %s = %lld\n", st[i].name, (long long)st[i].value.i64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
                APPEND("# stat %s = %u\n", st[i].name, st[i].value.b32);
            else
                APPEND("# stat %s = %g\n", st[i].name, st[i].value.f64);
        }
        free(st);
    }
    uint32_t nr = 0;
    greps(G.dev, &ei, &nr, NULL);
    VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr, sizeof *ir);
    for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    greps(G.dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; i++) {
        ir[i].pData = calloc(1, ir[i].dataSize + 1);
    }
    greps(G.dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; i++) {
        // Only the final machine code; NIR and ACO IR are long and not needed for these checks.
        if (strstr(ir[i].name, "Assembly") || strstr(ir[i].name, "assembly")) {
            APPEND("# representation: %s\n%s\n", ir[i].name, (char *)ir[i].pData);
        }
        free(ir[i].pData);
    }
    free(ir);
    p->asm_text = txt;
    if (G.out_dir[0]) {
        char path[800];
        snprintf(path, sizeof path, "%s/asm", G.out_dir);
        mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/asm/%s.s", G.out_dir, p->name);
        FILE *f = fopen(path, "w");
        if (f) {
            fputs(txt, f);
            fclose(f);
        }
    }
#undef APPEND
}

Pipe pipe_create(const char *spv, const char *label, uint32_t local_size, uint32_t req_sg, int full_subgroups) {
    Pipe p = {0};
    snprintf(p.name, sizeof p.name, "%s", label);
    p.local_size = local_size;
    char path[800];
    snprintf(path, sizeof path, "%s/%s", G.spv_dir, spv);
    size_t len = 0;
    char *code = read_file(path, &len);
    if (!code) {
        printf("VERDICT %s: FAIL cannot read %s\n", label, path);
        exit(1);
    }
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = len;
    smi.pCode = (const uint32_t *)code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(G.dev, &smi, NULL, &sm));
    free(code);
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo spi = {1, &me, 4, &local_size};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = req_sg;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.pNext = req_sg ? &rss : NULL;
    ci.stage.flags = full_subgroups ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = sm;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = &spi;
    ci.layout = G.pl;
    VkResult r = vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p.pipe);
    vkDestroyShaderModule(G.dev, sm, NULL);
    if (r != VK_SUCCESS) {
        printf("VERDICT %s: FAIL pipeline creation returned %d\n", label, (int)r);
        exit(1);
    }
    capture(&p);
    return p;
}

void pipe_destroy(Pipe *p) {
    if (p->pipe) vkDestroyPipeline(G.dev, p->pipe, NULL);
    free(p->asm_text);
    memset(p, 0, sizeof *p);
}

// ACO prints "BB<n>:" labels and branches "s_cbranch_* BB<n>" / "s_branch BB<n>". A loop is a branch back to an
// earlier label; its body is every instruction line between that label and the branch.
typedef struct {
    char mnem[48];
    int count;
} Count;
static Count g_last[64];
static int g_nlast;

int loop_count(const char *prefix) {
    int n = 0;
    for (int q = 0; q < g_nlast; q++)
        if (!strncmp(g_last[q].mnem, prefix, strlen(prefix))) n += g_last[q].count;
    return n;
}

static int mnemonic(const char *line, char *out, size_t cap) {
    while (*line == ' ' || *line == '\t') line++;
    if (!*line || *line == '#' || *line == ';' || *line == '\n') return 0;
    size_t i = 0;
    while (line[i] && !isspace((unsigned char)line[i]) && i + 1 < cap) {
        out[i] = line[i];
        i++;
    }
    out[i] = 0;
    if (i && out[i - 1] == ':') return 0; // label
    return i > 0;
}

int pipe_loop_report(const Pipe *p, const char *must, int print_body) {
    if (!p->asm_text) {
        printf("LOOP %s: no machine code captured\n", p->name);
        return -1;
    }
    // split into lines
    int nl = 0, capl = 1024;
    char **lines = malloc(sizeof(char *) * capl);
    char *dup = strdup(p->asm_text);
    for (char *s = strtok(dup, "\n"); s; s = strtok(NULL, "\n")) {
        if (nl == capl) lines = realloc(lines, sizeof(char *) * (capl *= 2));
        lines[nl++] = s;
    }
    int best_must = -1, best_a = -1, best_b = -1;
    for (int i = 0; i < nl; i++) {
        char m[48];
        if (!mnemonic(lines[i], m, sizeof m)) continue;
        if (strncmp(m, "s_cbranch", 9) && strcmp(m, "s_branch")) continue;
        const char *bb = strstr(lines[i], "BB");
        if (!bb) continue;
        char lab[32];
        int k = 0;
        while (bb[k] && (isalnum((unsigned char)bb[k]) || bb[k] == '_') && k < 30) {
            lab[k] = bb[k];
            k++;
        }
        lab[k] = ':';
        lab[k + 1] = 0;
        int start = -1;
        for (int j = 0; j < i; j++) {
            const char *t = lines[j];
            while (*t == ' ' || *t == '\t') t++;
            if (!strncmp(t, lab, strlen(lab))) start = j;
        }
        if (start < 0) continue;
        int nmust = 0;
        for (int j = start; j <= i; j++) {
            char mm[48];
            if (mnemonic(lines[j], mm, sizeof mm) && strstr(mm, must)) nmust++;
        }
        // innermost loop with the most `must` instructions: prefer the shortest on ties
        if (nmust > best_must || (nmust == best_must && best_b - best_a > i - start)) {
            best_must = nmust;
            best_a = start;
            best_b = i;
        }
    }
    g_nlast = 0;
    if (best_a < 0) {
        printf("LOOP %s: no loop found\n", p->name);
        free(lines);
        free(dup);
        return 0;
    }
    Count c[64];
    int nc = 0, total = 0;
    for (int j = best_a; j <= best_b; j++) {
        char m[48];
        if (!mnemonic(lines[j], m, sizeof m)) continue;
        total++;
        int f = -1;
        for (int q = 0; q < nc; q++)
            if (!strcmp(c[q].mnem, m)) f = q;
        if (f < 0 && nc < 64) {
            f = nc++;
            snprintf(c[f].mnem, sizeof c[f].mnem, "%s", m);
            c[f].count = 0;
        }
        if (f >= 0) c[f].count++;
    }
    memcpy(g_last, c, sizeof(Count) * nc);
    g_nlast = nc;
    printf("LOOP %s: %d instructions, %d x %s; ", p->name, total, best_must, must);
    for (int q = 0; q < nc; q++) printf("%s%s %d", q ? ", " : "", c[q].mnem, c[q].count);
    printf("\n");
    if (print_body) {
        for (int j = best_a; j <= best_b; j++) printf("  | %s\n", lines[j]);
    }
    free(lines);
    free(dup);
    return best_must;
}

double run_dispatch(const Pipe *p, uint32_t groups, const void *push, size_t push_bytes, const Fill *fills, int nfill,
                    double *wall_ns) {
    return run_dispatch_warm(p, groups, NULL, push, push_bytes, fills, nfill, wall_ns);
}

double run_dispatch_warm(const Pipe *p, uint32_t groups, const void *warm_push, const void *push, size_t push_bytes,
                         const Fill *fills, int nfill, double *wall_ns) {
    begin_cb();
    vkCmdResetQueryPool(G.cb, G.qp, 0, 2);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdBindPipeline(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pl, 0, 1, &G.ds, 0, NULL);
    // The warm-up dispatch raises the core clock from idle in the same submission, so the timed dispatch does not
    // include the clock ramp; the fills after it reset everything the timed dispatch reports.
    for (int pass = warm_push ? 0 : 1; pass < 2; pass++) {
        for (int i = 0; i < nfill; i++) vkCmdFillBuffer(G.cb, fills[i].buf->buf, 0, VK_WHOLE_SIZE, fills[i].value);
        vkCmdPipelineBarrier(G.cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL,
                             0, NULL);
        if (pass == 0) {
            vkCmdPushConstants(G.cb, G.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_bytes, warm_push);
            vkCmdDispatch(G.cb, groups, 1, 1);
            vkCmdPipelineBarrier(G.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb,
                                 0, NULL, 0, NULL);
        }
    }
    if (push_bytes) vkCmdPushConstants(G.cb, G.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_bytes, push);
    vkCmdWriteTimestamp(G.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.qp, 0);
    vkCmdDispatch(G.cb, groups, 1, 1);
    vkCmdWriteTimestamp(G.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.qp, 1);
    VK_CHECK(vkEndCommandBuffer(G.cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &G.cb;
    double t0 = now_ns();
    VK_CHECK(vkQueueSubmit(G.q, 1, &si, G.fence));
    VK_CHECK(vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, UINT64_MAX));
    double t1 = now_ns();
    VK_CHECK(vkResetFences(G.dev, 1, &G.fence));
    if (wall_ns) *wall_ns = t1 - t0;
    uint64_t ts[2];
    VK_CHECK(vkGetQueryPoolResults(G.dev, G.qp, 0, 2, sizeof ts, ts, 8,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double)(ts[1] - ts[0]) * G.ts_period_ns;
}

// ---- sampler and thermal rule ----

static pthread_t s_thr;
static volatile int s_run;
static double s_sclk[20000], s_pw[20000], s_t[20000];
static int s_n;
static double s_hottest;

double hotspot_c(void) {
    if (!G.hwmon[0]) return -1;
    char p[1100];
    snprintf(p, sizeof p, "%s/temp2_input", G.hwmon);
    return read_num(p, 1e-3);
}

void note_hotspot(double t) {
    if (t > s_hottest) s_hottest = t;
}

double highest_hotspot(void) { return s_hottest; }

static void thermal_stop(double t) {
    note_hotspot(t);
    printf("THERMAL hotspot %.0f C is at or above 95 C: stopping at once\n", t);
    printf("THERMAL highest %.0f C\n", s_hottest);
    fflush(stdout);
    _exit(4);
}

static void *sampler_main(void *arg) {
    (void)arg;
    char ps[1100], pp[1100];
    snprintf(ps, sizeof ps, "%s/pp_dpm_sclk", G.drm_dev);
    snprintf(pp, sizeof pp, "%s/power1_average", G.hwmon);
    while (s_run) {
        if (s_n < 20000) {
            s_sclk[s_n] = read_star_mhz(ps);
            s_pw[s_n] = read_num(pp, 1e-6);
            s_t[s_n] = hotspot_c();
            if (s_t[s_n] >= 95.0) thermal_stop(s_t[s_n]);
            s_n++;
        }
        usleep(50000);
    }
    return NULL;
}

void sampler_start(void) {
    s_n = 0;
    s_run = 1;
    pthread_create(&s_thr, NULL, sampler_main, NULL);
}

SampleStats sampler_stop(void) {
    s_run = 0;
    pthread_join(s_thr, NULL);
    SampleStats st = {0};
    st.n = s_n;
    if (s_n) {
        st.sclk_min = vmin(s_sclk, s_n);
        st.sclk_med = median(s_sclk, s_n);
        st.sclk_max = vmax(s_sclk, s_n);
        st.pw_min = vmin(s_pw, s_n);
        st.pw_max = vmax(s_pw, s_n);
        st.t_min = vmin(s_t, s_n);
        st.t_max = vmax(s_t, s_n);
        note_hotspot(st.t_max);
    }
    return st;
}

void thermal_gate(void) {
    double t = hotspot_c();
    if (t < 0) {
        printf("VERDICT thermal: FAIL hotspot sensor not found; refusing whole-card work\n");
        exit(1);
    }
    if (t >= 95.0) thermal_stop(t);
    note_hotspot(t);
    double t0 = now_ns(), cur = t;
    while ((cur = hotspot_c()) >= 60.0 && now_ns() - t0 < 120e9) usleep(250000);
    if (cur >= 60.0) {
        printf("THERMAL hotspot still %.0f C after 120 s: stopping\n", cur);
        printf("THERMAL highest %.0f C\n", s_hottest);
        exit(7);
    }
    if (t >= 60.0) printf("THERMAL paused %.1f s: %.0f C -> %.0f C\n", (now_ns() - t0) * 1e-9, t, cur);
}

void print_clocks(const char *when) {
    char ps[1100], pm[1100], pp[1100];
    snprintf(ps, sizeof ps, "%s/pp_dpm_sclk", G.drm_dev);
    snprintf(pm, sizeof pm, "%s/pp_dpm_mclk", G.drm_dev);
    snprintf(pp, sizeof pp, "%s/power1_average", G.hwmon);
    double t = hotspot_c();
    note_hotspot(t);
    printf("CLOCKS %s: sclk %d MHz mclk %d MHz power %.0f W hotspot %.0f C\n", when, read_star_mhz(ps),
           read_star_mhz(pm), read_num(pp, 1e-6), t);
}
