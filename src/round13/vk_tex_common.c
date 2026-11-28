// Shared Vulkan 1.3 compute + sampled-image harness implementation for Round 13 Probes B & C.
#define _GNU_SOURCE
#include "vk_tex_common.h"
#include "vk_tex_ref.h"
#include "vk_async_ref.h"

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

VktCtx VT;

#define VKT_CHECK(x)                                                                                 \
    do {                                                                                             \
        VkResult r_ = (x);                                                                           \
        if (r_ != VK_SUCCESS) {                                                                      \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);      \
            exit(3);                                                                                 \
        }                                                                                            \
    } while (0)

double vkt_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static int cmp_dbl(const void * a, const void * b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return x < y ? -1 : x > y;
}

double vkt_median(double * v, int n) {
    if (n <= 0) return NAN;
    double * c = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(c, v, sizeof(double) * (size_t)n);
    qsort(c, (size_t)n, sizeof(double), cmp_dbl);
    double m = (n % 2) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

double vkt_vmin(const double * v, int n) {
    double m = INFINITY;
    for (int i = 0; i < n; ++i) if (v[i] < m) m = v[i];
    return m;
}

double vkt_vmax(const double * v, int n) {
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) if (v[i] > m) m = v[i];
    return m;
}

static void find_sysfs(const VkPhysicalDevicePCIBusInfoPropertiesEXT * pci) {
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x",
             pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    VT.drm_dev[0] = VT.metrics_path[0] = VT.hwmon_hot_path[0] = VT.hwmon_fan_path[0] = '\0';
    DIR * d = opendir("/sys/class/drm");
    struct dirent * e;
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) != 0 || strchr(e->d_name, '-')) continue;
        char p[300], realp[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, realp)) continue;
        const char * base = strrchr(realp, '/');
        if (base && strcmp(base + 1, want) == 0) {
            snprintf(VT.drm_dev, sizeof VT.drm_dev, "%s", p);
            break;
        }
    }
    if (d) closedir(d);
    if (!VT.drm_dev[0]) return;
    snprintf(VT.metrics_path, sizeof VT.metrics_path, "%s/gpu_metrics", VT.drm_dev);
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", VT.drm_dev);
    d = opendir(hp);
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5) == 0) {
            snprintf(VT.hwmon_hot_path, sizeof VT.hwmon_hot_path, "%s/%s/temp2_input", hp, e->d_name);
            snprintf(VT.hwmon_fan_path, sizeof VT.hwmon_fan_path, "%s/%s/fan1_input", hp, e->d_name);
            break;
        }
    }
    if (d) closedir(d);
}

static int has_ext(const VkExtensionProperties * ex, uint32_t n, const char * name) {
    for (uint32_t i = 0; i < n; ++i) {
        if (strcmp(ex[i].extensionName, name) == 0) return 1;
    }
    return 0;
}

double vkt_hotspot_c(void) {
    if (!VT.hwmon_hot_path[0]) return -1.0;
    FILE * f = fopen(VT.hwmon_hot_path, "r");
    if (!f) return -1.0;
    double mc = -1.0;
    if (fscanf(f, "%lf", &mc) != 1) mc = -1.0;
    fclose(f);
    return mc > 0.0 ? mc * 1e-3 : -1.0;
}

// vk_async raises the stop to the study limit (owner 2026-11-27); the texture probes keep 90 C.
static double vkt_stop_c(void) {
    return VT.thermal_stop_c > 0.0 ? VT.thermal_stop_c : VKT_THERMAL_STOP_C;
}


int vkt_check_hotspot_fail_closed(double * out_hot_c) {
    return vka_check_hotspot_file(VT.hwmon_hot_path, vkt_stop_c(), out_hot_c);
}

int vkt_read_hotspot_status(double * out_hot_c) {
    return vka_read_hotspot_status(VT.hwmon_hot_path, vkt_stop_c(), out_hot_c);
}

int vkt_read_fan_rpm(void) {
    if (!VT.hwmon_fan_path[0]) return -1;
    FILE * f = fopen(VT.hwmon_fan_path, "r");
    if (!f) return -1;
    char buf[64];
    size_t n = fread(buf, 1, sizeof buf - 1u, f);
    fclose(f);
    buf[n] = '\0';
    while (n > 0u && (buf[n - 1u] == '\n' || buf[n - 1u] == '\r' ||
                      buf[n - 1u] == ' ' || buf[n - 1u] == '\t')) {
        buf[--n] = '\0';
    }
    if (n == 0u) return -1;
    char * endp = NULL;
    long rpm = strtol(buf, &endp, 10);
    if (!endp || *endp != '\0' || rpm < 0L || rpm > 20000L) return -1;
    return (int)rpm;
}

// Reads amdgpu gpu_metrics v1_3 (120 bytes): offset 6 hotspot, offset 54 current_gfxclk, offset 58 current_uclk.
int vkt_read_gpu_metrics(int * gfxclk_mhz, int * uclk_mhz, double * hot_c) {
    if (gfxclk_mhz) *gfxclk_mhz = -1;
    if (uclk_mhz) *uclk_mhz = -1;
    if (hot_c) *hot_c = vkt_hotspot_c();
    if (!VT.metrics_path[0]) return 0;
    FILE * f = fopen(VT.metrics_path, "rb");
    if (!f) return 0;
    unsigned char b[128];
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    if (n < 60) return 0;
    uint16_t gfx = 0, uclk = 0, mhot = 0;
    memcpy(&mhot, b + 6, 2);
    memcpy(&gfx, b + 54, 2);
    memcpy(&uclk, b + 58, 2);
    if (gfxclk_mhz) *gfxclk_mhz = (int)gfx;
    if (uclk_mhz) *uclk_mhz = (int)uclk;
    if (hot_c && *hot_c < 0.0 && mhot > 0 && mhot < 200) *hot_c = (double)mhot;
    return 1;
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < VT.memprops.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = VT.memprops.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

void vkt_init(int quiet) {
    memset(&VT, 0, sizeof VT);
    const char * o = getenv("VKS_OUT");
    snprintf(VT.out_dir, sizeof VT.out_dir, "%s", o ? o : "");
    const char * s = getenv("VKS_SPV");
    snprintf(VT.spv_dir, sizeof VT.spv_dir, "%s", s ? s : "build/round13/spv");

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdna4-r13-probes";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKT_CHECK(vkCreateInstance(&ici, NULL, &VT.inst));

    uint32_t n = 0;
    VKT_CHECK(vkEnumeratePhysicalDevices(VT.inst, &n, NULL));
    VkPhysicalDevice * devs = (VkPhysicalDevice *)calloc(n ? n : 1, sizeof *devs);
    VKT_CHECK(vkEnumeratePhysicalDevices(VT.inst, &n, devs));
    VT.pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == VKT_VENDOR && p.deviceID == VKT_DEVICE) {
            VT.pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!VT.pd) {
        printf("VERDICT device: FAIL no Vulkan device %04x:%04x (RX 9070 XT)\n", VKT_VENDOR, VKT_DEVICE);
        exit(1);
    }

    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(VT.pd, &p2);
    VT.ts_period_ns = p2.properties.limits.timestampPeriod;
    find_sysfs(&pci);
    if (!quiet) {
        printf("DEVICE %s (%04x:%04x) PCI %04x:%02x:%02x.%x driver %s %s; sysfs %s\n",
               p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID,
               pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction,
               drv.driverName, drv.driverInfo, VT.drm_dev[0] ? VT.drm_dev : "(not found)");
    }
    if (strcmp(drv.driverName, "radv") != 0) {
        printf("VERDICT device: FAIL driver is %s, not radv\n", drv.driverName);
        exit(1);
    }
    vkGetPhysicalDeviceMemoryProperties(VT.pd, &VT.memprops);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(VT.pd, &nq, NULL);
    VkQueueFamilyProperties * qp = (VkQueueFamilyProperties *)calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(VT.pd, &nq, qp);
    VT.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) {
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            qp[i].timestampValidBits) {
            VT.qf = i;
            break;
        }
    }
    if (VT.qf == UINT32_MAX) {
        for (uint32_t i = 0; i < nq; ++i) {
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                VT.qf = i;
                break;
            }
        }
    }
    free(qp);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(VT.pd, NULL, &ne, NULL);
    VkExtensionProperties * ex = (VkExtensionProperties *)calloc(ne, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(VT.pd, NULL, &ne, ex);
    const char * en[4];
    uint32_t nen = 0;
    if (has_ext(ex, ne, "VK_KHR_pipeline_executable_properties")) {
        en[nen++] = "VK_KHR_pipeline_executable_properties";
        VT.has_exec_props = 1;
    }
    if (has_ext(ex, ne, "VK_KHR_shader_clock")) {
        en[nen++] = "VK_KHR_shader_clock";
        VT.has_shader_clock = 1;
    }
    free(ex);

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VT.has_exec_props ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
        VT.has_exec_props ? (void *)&fexe : NULL};
    fclk.shaderSubgroupClock = VK_TRUE;
    fclk.shaderDeviceClock = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        VT.has_shader_clock ? (void *)&fclk : (VT.has_exec_props ? (void *)&fexe : NULL)};
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderSubgroupExtendedTypes = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = VT.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nen;
    dci.ppEnabledExtensionNames = en;
    VKT_CHECK(vkCreateDevice(VT.pd, &dci, NULL, &VT.dev));
    VT.fail_closed_thermal = 1;
    VT.thermal_stop_c = VKT_THERMAL_STOP_C;
    vkGetDeviceQueue(VT.dev, VT.qf, 0, &VT.q);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = VT.qf;
    VKT_CHECK(vkCreateCommandPool(VT.dev, &cpi, NULL, &VT.pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = VT.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VKT_CHECK(vkAllocateCommandBuffers(VT.dev, &cai, &VT.cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKT_CHECK(vkCreateFence(VT.dev, &fci, NULL, &VT.fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 4;
    VKT_CHECK(vkCreateQueryPool(VT.dev, &qpi, NULL, &VT.qp));

    VkDescriptorSetLayoutBinding bindings[3] = {
        {0u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3;
    dli.pBindings = bindings;
    VKT_CHECK(vkCreateDescriptorSetLayout(VT.dev, &dli, NULL, &VT.dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0u, VKT_PUSH_BYTES};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &VT.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VKT_CHECK(vkCreatePipelineLayout(VT.dev, &pli, NULL, &VT.pl));

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2u},
    };
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    VKT_CHECK(vkCreateDescriptorPool(VT.dev, &dpi, NULL, &VT.dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = VT.dp;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &VT.dsl;
    VKT_CHECK(vkAllocateDescriptorSets(VT.dev, &dai, &VT.ds));

    VT.dummy_buf = vkt_buf_create(4096, 1);
}

static const char * prio_label(VkQueueGlobalPriority p) {
    switch (p) {
        case VK_QUEUE_GLOBAL_PRIORITY_LOW: return "LOW";
        case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM: return "MEDIUM";
        case VK_QUEUE_GLOBAL_PRIORITY_HIGH: return "HIGH";
        case VK_QUEUE_GLOBAL_PRIORITY_REALTIME: return "REALTIME";
        default: return "UNKNOWN";
    }
}

static void fmt_qflags(VkQueueFlags f, char * buf, size_t cap) {
    size_t n = 0;
    buf[0] = '\0';
    if (f & VK_QUEUE_GRAPHICS_BIT) n += (size_t)snprintf(buf + n, cap - n, "%sG", n ? "|" : "");
    if (f & VK_QUEUE_COMPUTE_BIT) n += (size_t)snprintf(buf + n, cap - n, "%sC", n ? "|" : "");
    if (f & VK_QUEUE_TRANSFER_BIT) n += (size_t)snprintf(buf + n, cap - n, "%sT", n ? "|" : "");
    if (f & VK_QUEUE_SPARSE_BINDING_BIT) n += (size_t)snprintf(buf + n, cap - n, "%sS", n ? "|" : "");
    if (f & VK_QUEUE_PROTECTED_BIT) n += (size_t)snprintf(buf + n, cap - n, "%sP", n ? "|" : "");
    if (n == 0) snprintf(buf, cap, "NONE");
}

VkResult vkt_init_async(int quiet, int use_global_prio, VkQueueGlobalPriority comp_prio, int dump_qfamilies) {
    memset(&VT, 0, sizeof VT);
    const char * o = getenv("VKS_OUT");
    snprintf(VT.out_dir, sizeof VT.out_dir, "%s", o ? o : "");
    const char * s = getenv("VKS_SPV");
    snprintf(VT.spv_dir, sizeof VT.spv_dir, "%s", s ? s : "build/round13/spv");

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdna4-r13-vk-async";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult res = vkCreateInstance(&ici, NULL, &VT.inst);
    if (res != VK_SUCCESS) return res;

    uint32_t n = 0;
    VKT_CHECK(vkEnumeratePhysicalDevices(VT.inst, &n, NULL));
    VkPhysicalDevice * devs = (VkPhysicalDevice *)calloc(n ? n : 1, sizeof *devs);
    VKT_CHECK(vkEnumeratePhysicalDevices(VT.inst, &n, devs));
    VT.pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == VKT_VENDOR && p.deviceID == VKT_DEVICE) {
            VT.pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!VT.pd) {
        printf("VERDICT device: FAIL no Vulkan device %04x:%04x (RX 9070 XT)\n", VKT_VENDOR, VKT_DEVICE);
        vkDestroyInstance(VT.inst, NULL);
        memset(&VT, 0, sizeof VT);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(VT.pd, &p2);
    VT.ts_period_ns = p2.properties.limits.timestampPeriod;
    find_sysfs(&pci);
    if (!quiet) {
        printf("DEVICE %s (%04x:%04x) PCI %04x:%02x:%02x.%x driver %s %s; sysfs %s\n",
               p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID,
               pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction,
               drv.driverName, drv.driverInfo, VT.drm_dev[0] ? VT.drm_dev : "(not found)");
    }
    if (strcmp(drv.driverName, "radv") != 0) {
        printf("VERDICT device: FAIL driver is %s, not radv\n", drv.driverName);
        vkDestroyInstance(VT.inst, NULL);
        memset(&VT, 0, sizeof VT);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    vkGetPhysicalDeviceMemoryProperties(VT.pd, &VT.memprops);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(VT.pd, NULL, &ne, NULL);
    VkExtensionProperties * ex = (VkExtensionProperties *)calloc(ne ? ne : 1, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(VT.pd, NULL, &ne, ex);
    const int has_khr_gp = has_ext(ex, ne, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    const int has_ext_gp = has_ext(ex, ne, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME);
    const int has_ext_gpq = has_ext(ex, ne, VK_EXT_GLOBAL_PRIORITY_QUERY_EXTENSION_NAME);
    VT.has_global_prio = (has_khr_gp || has_ext_gp) ? 1 : 0;

    const char * en[8];
    uint32_t nen = 0;
    if (has_ext(ex, ne, "VK_KHR_pipeline_executable_properties")) {
        en[nen++] = "VK_KHR_pipeline_executable_properties";
        VT.has_exec_props = 1;
    }
    if (has_ext(ex, ne, "VK_KHR_shader_clock")) {
        en[nen++] = "VK_KHR_shader_clock";
        VT.has_shader_clock = 1;
    }
    if (use_global_prio) {
        if (has_khr_gp) {
            en[nen++] = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
        } else if (has_ext_gp) {
            en[nen++] = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
        } else {
            free(ex);
            vkDestroyInstance(VT.inst, NULL);
            memset(&VT, 0, sizeof VT);
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    free(ex);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(VT.pd, &nq, NULL);
    VkQueueFamilyProperties2 * qp2 = (VkQueueFamilyProperties2 *)calloc(nq ? nq : 1, sizeof *qp2);
    VkQueueFamilyGlobalPriorityProperties * gpp =
        (VkQueueFamilyGlobalPriorityProperties *)calloc(nq ? nq : 1, sizeof *gpp);
    const int can_query_gp = (has_khr_gp || has_ext_gpq) ? 1 : 0;
    for (uint32_t i = 0; i < nq; ++i) {
        qp2[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        if (can_query_gp) {
            gpp[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES;
            qp2[i].pNext = &gpp[i];
        }
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(VT.pd, &nq, qp2);

    VT.gfx_qf = UINT32_MAX;
    VT.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) {
        const VkQueueFamilyProperties * q = &qp2[i].queueFamilyProperties;
        if (dump_qfamilies) {
            char flg[32], plist[128] = "";
            fmt_qflags(q->queueFlags, flg, sizeof flg);
            if (can_query_gp && gpp[i].priorityCount > 0) {
                size_t pn = 0;
                for (uint32_t k = 0; k < gpp[i].priorityCount && k < VK_MAX_GLOBAL_PRIORITY_SIZE; ++k) {
                    pn += (size_t)snprintf(plist + pn, sizeof plist - pn, "%s%s(%d)",
                                           k ? "," : "", prio_label(gpp[i].priorities[k]),
                                           (int)gpp[i].priorities[k]);
                }
            } else {
                snprintf(plist, sizeof plist, "n/a");
            }
            printf("QFAMILY idx=%u flags=0x%x(%s) count=%u ts_bits=%u global_prios=[%s]\n",
                   i, q->queueFlags, flg, q->queueCount, q->timestampValidBits, plist);
        }
        if (VT.gfx_qf == UINT32_MAX && (q->queueFlags & VK_QUEUE_GRAPHICS_BIT) && q->timestampValidBits) {
            VT.gfx_qf = i;
        }
        if (VT.qf == UINT32_MAX && (q->queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(q->queueFlags & VK_QUEUE_GRAPHICS_BIT) && q->timestampValidBits) {
            VT.qf = i;
        }
    }
    if (VT.gfx_qf == UINT32_MAX) {
        for (uint32_t i = 0; i < nq; ++i) {
            if (qp2[i].queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                VT.gfx_qf = i;
                break;
            }
        }
    }
    if (VT.qf == UINT32_MAX) {
        for (uint32_t i = 0; i < nq; ++i) {
            if ((qp2[i].queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                qp2[i].queueFamilyProperties.timestampValidBits) {
                VT.qf = i;
                break;
            }
        }
    }
    if (VT.gfx_qf == UINT32_MAX || VT.qf == UINT32_MAX) {
        free(qp2);
        free(gpp);
        vkDestroyInstance(VT.inst, NULL);
        memset(&VT, 0, sizeof VT);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VT.has_separate_compute = (VT.qf != VT.gfx_qf) ? 1 : 0;
    const uint32_t gfx_qcount = qp2[VT.gfx_qf].queueFamilyProperties.queueCount;
    free(qp2);
    free(gpp);

    if (dump_qfamilies) {
        printf("QUEUE_PLAN gfx_qf=%u comp_qf=%u separate_compute=%d ext_khr_gp=%d ext_ext_gp=%d\n",
               VT.gfx_qf, VT.qf, VT.has_separate_compute, has_khr_gp, has_ext_gp);
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VT.has_exec_props ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
        VT.has_exec_props ? (void *)&fexe : NULL};
    fclk.shaderSubgroupClock = VK_TRUE;
    fclk.shaderDeviceClock = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        VT.has_shader_clock ? (void *)&fclk : (VT.has_exec_props ? (void *)&fexe : NULL)};
    f13.dynamicRendering = VK_TRUE;
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderSubgroupExtendedTypes = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;

    VkDeviceQueueGlobalPriorityCreateInfo gpci = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO, NULL, comp_prio};
    float prio1 = 1.0f;
    float prio2[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qcis[2];
    memset(qcis, 0, sizeof qcis);
    uint32_t nqci = 0;

    if (VT.has_separate_compute) {
        qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[0].queueFamilyIndex = VT.gfx_qf;
        qcis[0].queueCount = 1u;
        qcis[0].pQueuePriorities = &prio1;

        qcis[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[1].pNext = use_global_prio ? &gpci : NULL;
        qcis[1].queueFamilyIndex = VT.qf;
        qcis[1].queueCount = 1u;
        qcis[1].pQueuePriorities = &prio1;
        nqci = 2u;
    } else {
        qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[0].pNext = use_global_prio ? &gpci : NULL;
        qcis[0].queueFamilyIndex = VT.gfx_qf;
        qcis[0].queueCount = (gfx_qcount >= 2u) ? 2u : 1u;
        qcis[0].pQueuePriorities = prio2;
        nqci = 1u;
    }

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = nqci;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = nen;
    dci.ppEnabledExtensionNames = en;

    res = vkCreateDevice(VT.pd, &dci, NULL, &VT.dev);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(VT.inst, NULL);
        memset(&VT, 0, sizeof VT);
        return res;
    }

    VT.used_global_prio = use_global_prio ? 1 : 0;
    VT.comp_global_prio = comp_prio;
    VT.fail_closed_thermal = 1;
    VT.thermal_stop_c = VKA_THERMAL_STOP_C;

    if (VT.has_separate_compute) {
        vkGetDeviceQueue(VT.dev, VT.gfx_qf, 0u, &VT.gfx_q);
        vkGetDeviceQueue(VT.dev, VT.qf, 0u, &VT.q);
        VT.same_queue_handle = 0;
    } else if (gfx_qcount >= 2u) {
        vkGetDeviceQueue(VT.dev, VT.gfx_qf, 0u, &VT.gfx_q);
        vkGetDeviceQueue(VT.dev, VT.gfx_qf, 1u, &VT.q);
        VT.same_queue_handle = 0;
    } else {
        vkGetDeviceQueue(VT.dev, VT.gfx_qf, 0u, &VT.gfx_q);
        VT.q = VT.gfx_q;
        VT.same_queue_handle = 1;
    }

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = VT.qf;
    VKT_CHECK(vkCreateCommandPool(VT.dev, &cpi, NULL, &VT.pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = VT.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1u;
    VKT_CHECK(vkAllocateCommandBuffers(VT.dev, &cai, &VT.cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKT_CHECK(vkCreateFence(VT.dev, &fci, NULL, &VT.fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 4u;
    VKT_CHECK(vkCreateQueryPool(VT.dev, &qpi, NULL, &VT.qp));

    VkDescriptorSetLayoutBinding bindings[3] = {
        {0u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3u;
    dli.pBindings = bindings;
    VKT_CHECK(vkCreateDescriptorSetLayout(VT.dev, &dli, NULL, &VT.dsl));

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0u, VKT_PUSH_BYTES};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1u;
    pli.pSetLayouts = &VT.dsl;
    pli.pushConstantRangeCount = 1u;
    pli.pPushConstantRanges = &pcr;
    VKT_CHECK(vkCreatePipelineLayout(VT.dev, &pli, NULL, &VT.pl));

    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8u},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16u},
    };
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8u;
    dpi.poolSizeCount = 2u;
    dpi.pPoolSizes = ps;
    VKT_CHECK(vkCreateDescriptorPool(VT.dev, &dpi, NULL, &VT.dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = VT.dp;
    dai.descriptorSetCount = 1u;
    dai.pSetLayouts = &VT.dsl;
    VKT_CHECK(vkAllocateDescriptorSets(VT.dev, &dai, &VT.ds));

    VT.dummy_buf = vkt_buf_create(4096, 1);
    return VK_SUCCESS;
}

void vkt_shutdown(void) {
    if (!VT.dev) return;
    vkDeviceWaitIdle(VT.dev);
    vkt_buf_destroy(&VT.dummy_buf);
    if (VT.dp) vkDestroyDescriptorPool(VT.dev, VT.dp, NULL);
    if (VT.pl) vkDestroyPipelineLayout(VT.dev, VT.pl, NULL);
    if (VT.dsl) vkDestroyDescriptorSetLayout(VT.dev, VT.dsl, NULL);
    if (VT.qp) vkDestroyQueryPool(VT.dev, VT.qp, NULL);
    if (VT.fence) vkDestroyFence(VT.dev, VT.fence, NULL);
    if (VT.pool) vkDestroyCommandPool(VT.dev, VT.pool, NULL);
    vkDestroyDevice(VT.dev, NULL);
    vkDestroyInstance(VT.inst, NULL);
    memset(&VT, 0, sizeof VT);
}

int vkt_format_supported(VkFormat fmt) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(VT.pd, fmt, &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (fp.optimalTilingFeatures & need) == need;
}

VktBuf vkt_buf_create(VkDeviceSize size, int host_visible) {
    VktBuf b = {0};
    b.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKT_CHECK(vkCreateBuffer(VT.dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(VT.dev, b.buf, &mr);
    uint32_t t;
    if (host_visible) {
        t = mem_type(mr.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (t == UINT32_MAX) {
            t = mem_type(mr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        }
    } else {
        t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    }
    if (t == UINT32_MAX) {
        fprintf(stderr, "no Vulkan memory type for buffer\n");
        exit(3);
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = t;
    VKT_CHECK(vkAllocateMemory(VT.dev, &ai, NULL, &b.mem));
    VKT_CHECK(vkBindBufferMemory(VT.dev, b.buf, b.mem, 0));
    if (host_visible) VKT_CHECK(vkMapMemory(VT.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

void vkt_buf_destroy(VktBuf * b) {
    if (!b->buf) return;
    if (b->map) vkUnmapMemory(VT.dev, b->mem);
    vkDestroyBuffer(VT.dev, b->buf, NULL);
    vkFreeMemory(VT.dev, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

static void begin_cb(void) {
    VKT_CHECK(vkResetCommandBuffer(VT.cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKT_CHECK(vkBeginCommandBuffer(VT.cb, &bi));
}

static void submit_wait(void) {
    VKT_CHECK(vkEndCommandBuffer(VT.cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &VT.cb;
    VKT_CHECK(vkQueueSubmit(VT.q, 1, &si, VT.fence));
    VKT_CHECK(vkWaitForFences(VT.dev, 1, &VT.fence, VK_TRUE, UINT64_MAX));
    VKT_CHECK(vkResetFences(VT.dev, 1, &VT.fence));
}

VktImg vkt_img_create(VkFormat fmt, uint32_t w, uint32_t h, VkFilter filter) {
    VktImg im = {0};
    im.format = fmt;
    im.w = w;
    im.h = h;

    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = (VkExtent3D){w, h, 1u};
    ii.mipLevels = 1u;
    ii.arrayLayers = 1u;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKT_CHECK(vkCreateImage(VT.dev, &ii, NULL, &im.img));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(VT.dev, im.img, &mr);
    uint32_t t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = t;
    VKT_CHECK(vkAllocateMemory(VT.dev, &ai, NULL, &im.mem));
    VKT_CHECK(vkBindImageMemory(VT.dev, im.img, im.mem, 0));

    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    VKT_CHECK(vkCreateImageView(VT.dev, &vi, NULL, &im.view));

    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = filter;
    si.minFilter = filter;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;
    VKT_CHECK(vkCreateSampler(VT.dev, &si, NULL, &im.sampler));
    return im;
}

void vkt_img_destroy(VktImg * im) {
    if (!im->img) return;
    if (im->sampler) vkDestroySampler(VT.dev, im->sampler, NULL);
    if (im->view) vkDestroyImageView(VT.dev, im->view, NULL);
    vkDestroyImage(VT.dev, im->img, NULL);
    vkFreeMemory(VT.dev, im->mem, NULL);
    memset(im, 0, sizeof *im);
}

void vkt_img_fill_ref(VktImg * im, int ref_fmt, int exact_mode) {
    const uint32_t bpp = vkt_fmt_bpp[ref_fmt];
    const VkDeviceSize row_bytes = (VkDeviceSize)im->w * bpp;
    const VkDeviceSize max_chunk = 64ull << 20;
    uint32_t band_rows = (uint32_t)(max_chunk / row_bytes);
    if (band_rows < 1u) band_rows = 1u;
    if (band_rows > im->h) band_rows = im->h;
    VktBuf st = vkt_buf_create((VkDeviceSize)band_rows * row_bytes, 1);

    // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
    begin_cb();
    VkImageMemoryBarrier b0 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.image = im->img;
    b0.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    b0.srcAccessMask = 0;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(VT.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b0);
    submit_wait();

    for (uint32_t y0 = 0; y0 < im->h; y0 += band_rows) {
        uint32_t rows = (im->h - y0 < band_rows) ? (im->h - y0) : band_rows;
        uint8_t * dst = (uint8_t *)st.map;
        for (uint32_t dy = 0; dy < rows; ++dy) {
            const uint32_t y = y0 + dy;
            uint8_t * rowp = dst + (size_t)dy * (size_t)row_bytes;
            for (uint32_t x = 0; x < im->w; ++x) {
                uint8_t px[16];
                vkt_encode_texel(ref_fmt, exact_mode, x, y, px);
                memcpy(rowp + (size_t)x * bpp, px, bpp);
            }
        }
        begin_cb();
        VkBufferImageCopy reg = {0};
        reg.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
        reg.imageOffset = (VkOffset3D){0, (int32_t)y0, 0};
        reg.imageExtent = (VkExtent3D){im->w, rows, 1u};
        vkCmdCopyBufferToImage(VT.cb, st.buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
        submit_wait();
    }

    // Transition TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    begin_cb();
    VkImageMemoryBarrier b1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.image = im->img;
    b1.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(VT.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b1);
    submit_wait();
    vkt_buf_destroy(&st);
}

void vkt_bind(const VktImg * im, const VktBuf * out_buf, const VktBuf * in_buf) {
    VKT_CHECK(vkDeviceWaitIdle(VT.dev));
    VkDescriptorImageInfo di = {im->sampler, im->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo db1 = {(out_buf ? out_buf : &VT.dummy_buf)->buf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo db2 = {(in_buf ? in_buf : &VT.dummy_buf)->buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[3];
    memset(w, 0, sizeof w);
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = VT.ds;
    w[0].dstBinding = 0u;
    w[0].descriptorCount = 1u;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &di;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = VT.ds;
    w[1].dstBinding = 1u;
    w[1].descriptorCount = 1u;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &db1;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = VT.ds;
    w[2].dstBinding = 2u;
    w[2].descriptorCount = 1u;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &db2;
    vkUpdateDescriptorSets(VT.dev, 3, w, 0, NULL);
}

static char * read_bin_file(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * d = (char *)malloc((size_t)n + 1u);
    if (fread(d, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(d);
        return NULL;
    }
    d[n] = '\0';
    fclose(f);
    *len = (size_t)n;
    return d;
}

static void capture_aco(VktPipe * p) {
    p->vgprs = -1;
    p->sg_per_simd = -1;
    if (!VT.has_exec_props) return;
    PFN_vkGetPipelineExecutablePropertiesKHR gprops =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(VT.dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            VT.dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gstats =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(VT.dev, "vkGetPipelineExecutableStatisticsKHR");
    if (!gprops || !greps) return;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p->pipe;
    uint32_t ne = 0;
    gprops(VT.dev, &pi, &ne, NULL);
    if (!ne) return;
    VkPipelineExecutablePropertiesKHR ep[4];
    for (int i = 0; i < 4; ++i) ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (ne > 4) ne = 4;
    gprops(VT.dev, &pi, &ne, ep);
    uint32_t exec_idx = 0;
    for (uint32_t i = 0; i < ne; ++i) {
        if (ep[i].stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
            exec_idx = i;
            break;
        }
    }
    p->sg_size = ep[exec_idx].subgroupSize;

    VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = p->pipe;
    ei.executableIndex = exec_idx;

    size_t cap = 1u << 16, len = 0;
    char * txt = (char *)malloc(cap);
    txt[0] = '\0';
#define APPEND(...)                                                                 \
    do {                                                                            \
        int need_ = snprintf(NULL, 0, __VA_ARGS__);                                 \
        while (len + (size_t)need_ + 1u > cap) txt = (char *)realloc(txt, cap *= 2); \
        len += (size_t)snprintf(txt + len, cap - len, __VA_ARGS__);                 \
    } while (0)
    APPEND("# pipeline %s, executable \"%s\" (%s), subgroup size %u\n",
           p->name, ep[exec_idx].name, ep[exec_idx].description, ep[exec_idx].subgroupSize);
    if (gstats) {
        uint32_t ns = 0;
        gstats(VT.dev, &ei, &ns, NULL);
        VkPipelineExecutableStatisticKHR * st =
            (VkPipelineExecutableStatisticKHR *)calloc(ns ? ns : 1, sizeof *st);
        for (uint32_t i = 0; i < ns; ++i) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        gstats(VT.dev, &ei, &ns, st);
        for (uint32_t i = 0; i < ns; ++i) {
            if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) {
                APPEND("# stat %s = %llu\n", st[i].name, (unsigned long long)st[i].value.u64);
                if (strcmp(st[i].name, "VGPRs") == 0) p->vgprs = (int)st[i].value.u64;
                if (strcmp(st[i].name, "Subgroups per SIMD") == 0) p->sg_per_simd = (int)st[i].value.u64;
            } else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) {
                APPEND("# stat %s = %lld\n", st[i].name, (long long)st[i].value.i64);
            } else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR) {
                APPEND("# stat %s = %u\n", st[i].name, st[i].value.b32);
            } else {
                APPEND("# stat %s = %g\n", st[i].name, st[i].value.f64);
            }
        }
        free(st);
    }
    uint32_t nr = 0;
    greps(VT.dev, &ei, &nr, NULL);
    VkPipelineExecutableInternalRepresentationKHR * ir =
        (VkPipelineExecutableInternalRepresentationKHR *)calloc(nr ? nr : 1, sizeof *ir);
    for (uint32_t i = 0; i < nr; ++i) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    greps(VT.dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; ++i) ir[i].pData = calloc(1, ir[i].dataSize + 1u);
    greps(VT.dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; ++i) {
        if (strstr(ir[i].name, "Assembly") || strstr(ir[i].name, "assembly")) {
            APPEND("# representation: %s\n%s\n", ir[i].name, (char *)ir[i].pData);
        }
        free(ir[i].pData);
    }
    free(ir);
    p->asm_text = txt;
    if (VT.out_dir[0]) {
        char path[800];
        snprintf(path, sizeof path, "%s/asm", VT.out_dir);
        mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/asm/%s.s", VT.out_dir, p->name);
        FILE * f = fopen(path, "w");
        if (f) {
            fputs(txt, f);
            fclose(f);
        }
    }
#undef APPEND
}

VktPipe vkt_pipe_create(const char * spv, const char * label, uint32_t local_size, uint32_t req_sg) {
    VktPipe p = {0};
    snprintf(p.name, sizeof p.name, "%s", label);
    p.local_size = local_size;
    char path[800];
    snprintf(path, sizeof path, "%s/%s", VT.spv_dir, spv);
    size_t len = 0;
    char * code = read_bin_file(path, &len);
    if (!code) {
        fprintf(stderr, "cannot read SPIR-V %s\n", path);
        exit(1);
    }
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = len;
    smi.pCode = (const uint32_t *)code;
    VkShaderModule sm;
    VKT_CHECK(vkCreateShaderModule(VT.dev, &smi, NULL, &sm));
    free(code);

    VkSpecializationMapEntry me = {0u, 0u, 4u};
    VkSpecializationInfo spi = {1u, &me, 4u, &local_size};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = req_sg;

    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    if (VT.has_exec_props) {
        ci.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR |
                   VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.pNext = req_sg ? &rss : NULL;
    ci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = sm;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = &spi;
    ci.layout = VT.pl;
    VKT_CHECK(vkCreateComputePipelines(VT.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p.pipe));
    vkDestroyShaderModule(VT.dev, sm, NULL);
    capture_aco(&p);
    return p;
}

void vkt_capture_pipe_aco(VkPipeline pipe, const char * label, VktPipe * out) {
    memset(out, 0, sizeof *out);
    out->pipe = pipe;
    snprintf(out->name, sizeof out->name, "%s", label);
    capture_aco(out);
}

void vkt_pipe_destroy(VktPipe * p) {
    if (p->pipe) vkDestroyPipeline(VT.dev, p->pipe, NULL);
    free(p->asm_text);
    memset(p, 0, sizeof *p);
}

static int parse_mnem(const char * line, char * out, size_t cap) {
    while (*line == ' ' || *line == '\t') ++line;
    if (!*line || *line == '#' || *line == ';' || *line == '\n') return 0;
    size_t i = 0;
    while (line[i] && !isspace((unsigned char)line[i]) && i + 1 < cap) {
        out[i] = line[i];
        ++i;
    }
    out[i] = '\0';
    if (i > 0 && out[i - 1] == ':') return 0;
    return i > 0;
}

VktLoopStats vkt_pipe_analyze_loop(const VktPipe * p, const char * primary_prefix) {
    VktLoopStats st = {0};
    if (!p->asm_text) return st;
    int nl = 0, capl = 1024;
    char ** lines = (char **)malloc(sizeof(char *) * (size_t)capl);
    char * dup = strdup(p->asm_text);
    for (char * s = strtok(dup, "\n"); s; s = strtok(NULL, "\n")) {
        if (nl == capl) lines = (char **)realloc(lines, sizeof(char *) * (size_t)(capl *= 2));
        lines[nl++] = s;
    }
    int best_score = -1, best_a = -1, best_b = -1;
    for (int i = 0; i < nl; ++i) {
        char m[64];
        if (!parse_mnem(lines[i], m, sizeof m)) continue;
        if (strncmp(m, "s_cbranch", 9) != 0 && strcmp(m, "s_branch") != 0) continue;
        const char * bb = strstr(lines[i], "BB");
        if (!bb) continue;
        char lab[32];
        int k = 0;
        while (bb[k] && (isalnum((unsigned char)bb[k]) || bb[k] == '_') && k < 30) {
            lab[k] = bb[k];
            ++k;
        }
        lab[k] = ':';
        lab[k + 1] = '\0';
        int start = -1;
        for (int j = 0; j < i; ++j) {
            const char * t = lines[j];
            while (*t == ' ' || *t == '\t') ++t;
            if (strncmp(t, lab, strlen(lab)) == 0) start = j;
        }
        if (start < 0) continue;
        int score = 0;
        for (int j = start; j <= i; ++j) {
            char mm[64];
            if (parse_mnem(lines[j], mm, sizeof mm) && primary_prefix && strstr(mm, primary_prefix)) {
                ++score;
            }
        }
        if (score > best_score || (score == best_score && best_a >= 0 && (i - start) < (best_b - best_a))) {
            best_score = score;
            best_a = start;
            best_b = i;
        }
    }
    if (best_a < 0) {
        best_a = 0;
        best_b = nl - 1;
    }
    for (int j = best_a; j <= best_b; ++j) {
        char m[64];
        if (!parse_mnem(lines[j], m, sizeof m)) continue;
        ++st.total_insts;
        if (strncmp(m, "image_load", 10) == 0) ++st.image_load;
        if (strncmp(m, "image_sample", 12) == 0) ++st.image_sample;
        if (strncmp(m, "image_gather", 12) == 0) ++st.image_gather;
        if (strncmp(m, "v_pk_fma_f16", 12) == 0 || strncmp(m, "v_pk_fmac_f16", 13) == 0) ++st.pk_fma_f16;
        if (strncmp(m, "v_fma_f32", 9) == 0 || strncmp(m, "v_fmac_f32", 10) == 0 ||
            strncmp(m, "v_fmaak_f32", 11) == 0 || strncmp(m, "v_fmamk_f32", 11) == 0) ++st.fma_f32;
        if (strncmp(m, "v_dual_fmac_f32", 15) == 0 || strncmp(m, "v_dual_fmaak_f32", 16) == 0 ||
            strncmp(m, "v_dual_fmamk_f32", 16) == 0) ++st.dual_fmac_f32;
        if (strncmp(m, "v_max_u32", 9) == 0 || strncmp(m, "v_min_u32", 9) == 0 ||
            strncmp(m, "v_sub_nc_u32", 12) == 0 || strncmp(m, "v_sub_u32", 9) == 0 ||
            strncmp(m, "v_subrev_nc_u32", 15) == 0 || strncmp(m, "v_subrev_u32", 12) == 0 ||
            strncmp(m, "v_add_nc_u32", 12) == 0 || strncmp(m, "v_add_u32", 9) == 0 ||
            strncmp(m, "v_add3_u32", 10) == 0 || strncmp(m, "v_lshl_add_u32", 14) == 0 ||
            strncmp(m, "v_dual_add_nc_u32", 17) == 0 || strncmp(m, "v_dual_sub_nc_u32", 17) == 0 ||
            strncmp(m, "v_dual_subrev_nc_u32", 20) == 0 || strncmp(m, "v_dual_and_b32", 14) == 0 ||
            strncmp(m, "v_bfe_u32", 9) == 0 || strncmp(m, "v_and_b32", 9) == 0 ||
            strncmp(m, "v_lshrrev_b32", 13) == 0 || strncmp(m, "v_sad_", 6) == 0 ||
            strncmp(m, "v_absdiff", 9) == 0) {
            ++st.int_alu;
        }
        if (strncmp(m, "v_msad_", 7) == 0) ++st.v_msad;
        if (strncmp(m, "buffer_load_b128", 16) == 0 || strncmp(m, "global_load_b128", 16) == 0 ||
            strncmp(m, "buffer_load_dwordx4", 19) == 0 || strncmp(m, "global_load_dwordx4", 19) == 0) {
            ++st.buf_load_b128;
        }
        if (strncmp(m, "buffer_store_b128", 17) == 0 || strncmp(m, "global_store_b128", 17) == 0 ||
            strncmp(m, "buffer_store_dwordx4", 20) == 0 || strncmp(m, "global_store_dwordx4", 20) == 0) {
            ++st.buf_store_b128;
        }
    }
    free(lines);
    free(dup);
    return st;
}

int vkt_pipe_has_shader_clock(const VktPipe * p) {
    if (!VT.has_exec_props) return 1;
    if (!p->asm_text) return 0;
    return vkt_asm_clock_order_ok(p->asm_text);
}

double vkt_dispatch(const VktPipe * p, uint32_t groups_x, const void * push, size_t push_bytes, double * wall_ns) {
    begin_cb();
    vkCmdResetQueryPool(VT.cb, VT.qp, 0, 2);
    vkCmdBindPipeline(VT.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(VT.cb, VK_PIPELINE_BIND_POINT_COMPUTE, VT.pl, 0, 1, &VT.ds, 0, NULL);
    if (push && push_bytes) {
        vkCmdPushConstants(VT.cb, VT.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u, (uint32_t)push_bytes, push);
    }
    vkCmdWriteTimestamp2(VT.cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VT.qp, 0u);
    vkCmdDispatch(VT.cb, groups_x, 1u, 1u);
    vkCmdWriteTimestamp2(VT.cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VT.qp, 1u);
    VKT_CHECK(vkEndCommandBuffer(VT.cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &VT.cb;
    const double t0 = vkt_now_ns();
    VKT_CHECK(vkQueueSubmit(VT.q, 1, &si, VT.fence));
    VKT_CHECK(vkWaitForFences(VT.dev, 1, &VT.fence, VK_TRUE, UINT64_MAX));
    const double t1 = vkt_now_ns();
    VKT_CHECK(vkResetFences(VT.dev, 1, &VT.fence));
    if (wall_ns) *wall_ns = t1 - t0;
    uint64_t ts[2] = {0, 0};
    VKT_CHECK(vkGetQueryPoolResults(VT.dev, VT.qp, 0, 2, sizeof ts, ts, 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double)(ts[1] - ts[0]) * VT.ts_period_ns;
}

// ---- Background gpu_metrics sampler (2 ms cadence) with the running probe's fail-closed stop (vkt_stop_c: 90 C, vk_async 95 C) ----
enum { VKT_METRICS_CAP = 16384 };
static pthread_t g_mthr;
static volatile int g_mrun = 0;
static double g_win_t0 = 0.0;
static double g_win_t1 = 0.0;
static double g_ts[VKT_METRICS_CAP];
static double g_sclk[VKT_METRICS_CAP], g_uclk[VKT_METRICS_CAP], g_hot[VKT_METRICS_CAP];
static int g_mn = 0;

static void * metrics_thread(void * arg) {
    (void)arg;
    while (g_mrun) {
        int sclk = -1, uclk = -1;
        double hot = -1.0;
        const double t_samp = vkt_now_ns();
        vkt_read_gpu_metrics(&sclk, &uclk, &hot);
        if (VT.fail_closed_thermal) {
            double fc_hot = -1.0;
            if (!vkt_check_hotspot_fail_closed(&fc_hot)) {
                printf("THERMAL ABORT (fail-closed): hotspot %.1f C (limit %.0f C, sensor=%s)\n",
                       fc_hot, vkt_stop_c(), VT.hwmon_hot_path[0] ? VT.hwmon_hot_path : "(missing)");
                fflush(stdout);
                _exit(4);
            }
            hot = fc_hot;
        } else if (hot >= vkt_stop_c()) {
            printf("THERMAL ABORT: hotspot %.1f C >= %.0f C\n", hot, vkt_stop_c());
            fflush(stdout);
            _exit(4);
        }
        if (g_mrun && g_mn < VKT_METRICS_CAP && sclk > 0) {
            g_ts[g_mn] = t_samp;
            g_sclk[g_mn] = (double)sclk;
            g_uclk[g_mn] = (double)uclk;
            g_hot[g_mn] = hot;
            ++g_mn;
        }
        usleep(2000);  // 2 ms dense sampling inside the timed window
    }
    return NULL;
}

void vkt_metrics_start(void) {
    g_mn = 0;
    g_win_t0 = vkt_now_ns();
    g_win_t1 = 0.0;
    g_mrun = 1;
    pthread_create(&g_mthr, NULL, metrics_thread, NULL);
}

VktMetrics vkt_metrics_stop(void) {
    g_win_t1 = vkt_now_ns();
    g_mrun = 0;
    pthread_join(g_mthr, NULL);
    VktMetrics m = {0};
    int n_in = 0;
    double sclk_sum = 0.0;
    for (int i = 0; i < g_mn; ++i) {
        if (g_ts[i] >= g_win_t0 && g_ts[i] <= g_win_t1 && g_sclk[i] > 0.0) {
            g_sclk[n_in] = g_sclk[i];
            g_uclk[n_in] = g_uclk[i];
            g_hot[n_in] = g_hot[i];
            sclk_sum += g_sclk[i];
            ++n_in;
        }
    }
    m.n_samples = n_in;
    if (n_in > 0) {
        m.sclk_avg_mhz = sclk_sum / (double)n_in;
        m.sclk_med_mhz = vkt_median(g_sclk, n_in);
        m.sclk_min_mhz = vkt_vmin(g_sclk, n_in);
        m.sclk_max_mhz = vkt_vmax(g_sclk, n_in);
        m.uclk_med_mhz = vkt_median(g_uclk, n_in);
        m.hot_max_c = vkt_vmax(g_hot, n_in);
    } else {
        // Never fabricate an in-window SCLK sample after the burst ends; still report live hotspot & UCLK.
        int sclk = -1, uclk = -1;
        double hot = -1.0;
        vkt_read_gpu_metrics(&sclk, &uclk, &hot);
        if (uclk > 0) m.uclk_med_mhz = (double)uclk;
        if (hot > 0.0) m.hot_max_c = hot;
    }
    return m;
}

int vkt_thermal_guard_and_cool(double target_c, int max_wait_ms) {
    double hot = -1.0;
    if (!vkt_check_hotspot_fail_closed(&hot)) {
        printf("THERMAL ABORT (fail-closed): hotspot %.1f C (limit %.0f C, sensor=%s)\n",
               hot, vkt_stop_c(), VT.hwmon_hot_path[0] ? VT.hwmon_hot_path : "(missing)");
        fflush(stdout);
        exit(4);
    }
    if (hot >= target_c) {
        VkaCooldownResult cr;
        int cst = vka_wait_cooldown_file(VT.hwmon_hot_path, vkt_stop_c(), target_c, 0.0,
                                         max_wait_ms, 0, VKA_COOLDOWN_POLL_MS, &cr);
        const int act = vka_precell_cooldown_action(cst, cr.end_hot_c, vkt_stop_c());
        printf("COOLDOWN start=%.1fC end=%.1fC waited=%dms status=%s\n",
               cr.start_hot_c, cr.end_hot_c, cr.waited_ms,
               act == VKA_PRECELL_RUN ? "ok" : (cst == VKA_COOL_SENSOR_ERR ? "SENSOR_ERR" : "TIMEOUT"));
        fflush(stdout);
        if (act == VKA_PRECELL_ABORT_HOT) {
            printf("THERMAL ABORT (fail-closed): cooldown ended at %.1f C (limit %.0f C, sensor=%s)\n",
                   cr.end_hot_c, vkt_stop_c(), VT.hwmon_hot_path[0] ? VT.hwmon_hot_path : "(missing)");
            fflush(stdout);
            exit(4);
        }
        if (act == VKA_PRECELL_FAIL_CELL) {
            return 0;
        }
    }
    return 1;
}
