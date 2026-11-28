// Round 13 (P1): Vulkan compute pipeline compiler and ACO stats/disassembly extractor.
// Creates compute pipelines ONLY (no command buffers, no buffers/images, no dispatches)
// with VK_KHR_pipeline_executable_properties on RX 9070 XT (1002:7550, radv).
//
// Modes:
//   vk_pipestats <spv> [--entry main] [--sg 0|32|64] [--asm-out FILE] [--json-out FILE]
//   vk_pipestats --batch <manifest.tsv> [--jsonl-out FILE]
//   vk_pipestats --reflect-only <spv> [--entry main]
//
// Manifest TSV columns (5 tab-separated fields, '#' comments and blank lines skipped):
//   id<TAB>spv_path<TAB>entry<TAB>req_sg<TAB>asm_out_path
#define _GNU_SOURCE

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define TARGET_VENDOR 0x1002u
#define TARGET_DEVICE 0x7550u
#define HOTSPOT_LIMIT_MC 90000

#define SPV_MAGIC 0x07230203u
#define SPV_OP_ENTRY_POINT 15u
#define SPV_OP_TYPE_IMAGE 25u
#define SPV_OP_TYPE_SAMPLER 26u
#define SPV_OP_TYPE_SAMPLED_IMAGE 27u
#define SPV_OP_TYPE_ARRAY 28u
#define SPV_OP_TYPE_RUNTIME_ARRAY 29u
#define SPV_OP_TYPE_STRUCT 30u
#define SPV_OP_TYPE_POINTER 32u
#define SPV_OP_CONSTANT 43u
#define SPV_OP_SPEC_CONSTANT 50u
#define SPV_OP_VARIABLE 59u
#define SPV_OP_DECORATE 71u

#define SPV_EXEC_GLCOMPUTE 5u
#define SPV_SC_UNIFORM_CONSTANT 0u
#define SPV_SC_UNIFORM 2u
#define SPV_SC_PUSH_CONSTANT 9u
#define SPV_SC_STORAGE_BUFFER 12u

#define SPV_DEC_BLOCK 2u
#define SPV_DEC_BUFFER_BLOCK 3u
#define SPV_DEC_BINDING 33u
#define SPV_DEC_DESCRIPTOR_SET 34u

#define MAX_BINDINGS 256
#define MAX_SETS 16

typedef struct {
    uint32_t set;
    uint32_t binding;
    VkDescriptorType type;
    uint32_t count;
    int runtime_array;
} ReflectedBinding;

typedef struct {
    ReflectedBinding bindings[MAX_BINDINGS];
    uint32_t nbindings;
    uint32_t max_set;
    int has_push_constants;
    int entry_found;
    uint32_t spv_version;
} SpvReflection;

typedef struct {
    uint32_t op;
    uint32_t a, b, c, d, e; // generic fields per opcode
    uint32_t set;
    uint32_t binding;
    int has_set;
    int has_binding;
    int is_buffer_block;
} IdInfo;

static int read_file_bytes(const char *path, uint8_t **out_data, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    uint8_t *buf = malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_data = buf;
    *out_len = (size_t)n;
    return 0;
}

static int spv_reflect(const uint8_t *bytes, size_t len, const char *want_entry, SpvReflection *out, char *err,
                       size_t err_sz) {
    memset(out, 0, sizeof *out);
    if (len < 20 || (len % 4) != 0) {
        snprintf(err, err_sz, "invalid SPIR-V size %zu", len);
        return -1;
    }
    const uint32_t *w = (const uint32_t *)bytes;
    size_t nw = len / 4;
    if (w[0] != SPV_MAGIC) {
        snprintf(err, err_sz, "bad SPIR-V magic 0x%08x", w[0]);
        return -1;
    }
    out->spv_version = w[1];
    uint32_t bound = w[3];
    if (bound == 0 || bound > (1u << 22)) {
        snprintf(err, err_sz, "unreasonable SPIR-V id bound %u", bound);
        return -1;
    }
    IdInfo *ids = calloc(bound, sizeof *ids);
    if (!ids) {
        snprintf(err, err_sz, "OOM allocating id table (%u)", bound);
        return -1;
    }

    size_t i = 5;
    while (i < nw) {
        uint32_t hdr = w[i];
        uint32_t wc = hdr >> 16;
        uint32_t op = hdr & 0xffffu;
        if (wc == 0 || i + wc > nw) {
            snprintf(err, err_sz, "truncated SPIR-V instruction at word %zu (wc=%u)", i, wc);
            free(ids);
            return -1;
        }
        if (op == SPV_OP_ENTRY_POINT && wc >= 4) {
            uint32_t model = w[i + 1];
            const char *name = (const char *)&w[i + 3];
            if (model == SPV_EXEC_GLCOMPUTE && strcmp(name, want_entry) == 0) out->entry_found = 1;
        } else if (op == SPV_OP_DECORATE && wc >= 3) {
            uint32_t tid = w[i + 1];
            uint32_t dec = w[i + 2];
            if (tid < bound) {
                if (dec == SPV_DEC_BINDING && wc >= 4) {
                    ids[tid].binding = w[i + 3];
                    ids[tid].has_binding = 1;
                } else if (dec == SPV_DEC_DESCRIPTOR_SET && wc >= 4) {
                    ids[tid].set = w[i + 3];
                    ids[tid].has_set = 1;
                } else if (dec == SPV_DEC_BUFFER_BLOCK) {
                    ids[tid].is_buffer_block = 1;
                }
            }
        } else if (op == SPV_OP_TYPE_IMAGE && wc >= 9) {
            uint32_t rid = w[i + 1];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 3]; // dim
                ids[rid].b = w[i + 7]; // sampled
            }
        } else if (op == SPV_OP_TYPE_SAMPLER && wc >= 2) {
            uint32_t rid = w[i + 1];
            if (rid < bound) ids[rid].op = op;
        } else if (op == SPV_OP_TYPE_SAMPLED_IMAGE && wc >= 3) {
            uint32_t rid = w[i + 1];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 2];
            }
        } else if (op == SPV_OP_TYPE_ARRAY && wc >= 4) {
            uint32_t rid = w[i + 1];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 2]; // elem type
                ids[rid].b = w[i + 3]; // length const id
            }
        } else if (op == SPV_OP_TYPE_RUNTIME_ARRAY && wc >= 3) {
            uint32_t rid = w[i + 1];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 2]; // elem type
            }
        } else if (op == SPV_OP_TYPE_STRUCT && wc >= 2) {
            uint32_t rid = w[i + 1];
            if (rid < bound) ids[rid].op = op;
        } else if (op == SPV_OP_TYPE_POINTER && wc >= 4) {
            uint32_t rid = w[i + 1];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 2]; // storage class
                ids[rid].b = w[i + 3]; // pointee type
            }
        } else if ((op == SPV_OP_CONSTANT || op == SPV_OP_SPEC_CONSTANT) && wc >= 4) {
            uint32_t rid = w[i + 2];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = w[i + 3]; // 32-bit literal value
            }
        } else if (op == SPV_OP_VARIABLE && wc >= 4) {
            uint32_t rty = w[i + 1];
            uint32_t rid = w[i + 2];
            uint32_t sc = w[i + 3];
            if (rid < bound) {
                ids[rid].op = op;
                ids[rid].a = rty;
                ids[rid].b = sc;
            }
        }
        i += wc;
    }

    if (!out->entry_found) {
        snprintf(err, err_sz, "compute entry point '%s' not found in SPIR-V", want_entry);
        free(ids);
        return -1;
    }

    for (uint32_t id = 0; id < bound; id++) {
        if (ids[id].op != SPV_OP_VARIABLE) continue;
        uint32_t sc = ids[id].b;
        if (sc == SPV_SC_PUSH_CONSTANT) {
            out->has_push_constants = 1;
            continue;
        }
        if (sc != SPV_SC_UNIFORM_CONSTANT && sc != SPV_SC_UNIFORM && sc != SPV_SC_STORAGE_BUFFER) continue;
        if (!ids[id].has_binding) continue;

        uint32_t set = ids[id].has_set ? ids[id].set : 0u;
        uint32_t binding = ids[id].binding;
        if (set >= MAX_SETS) {
            snprintf(err, err_sz, "descriptor set %u exceeds MAX_SETS (%d)", set, MAX_SETS);
            free(ids);
            return -1;
        }
        uint32_t ptr_ty = ids[id].a;
        if (ptr_ty >= bound || ids[ptr_ty].op != SPV_OP_TYPE_POINTER) continue;
        uint32_t ty = ids[ptr_ty].b;
        uint32_t count = 1;
        int is_rt = 0;
        if (ty < bound && ids[ty].op == SPV_OP_TYPE_ARRAY) {
            uint32_t len_id = ids[ty].b;
            if (len_id < bound && (ids[len_id].op == SPV_OP_CONSTANT || ids[len_id].op == SPV_OP_SPEC_CONSTANT) &&
                ids[len_id].a > 0)
                count = ids[len_id].a;
            else
                count = 16;
            ty = ids[ty].a;
        } else if (ty < bound && ids[ty].op == SPV_OP_TYPE_RUNTIME_ARRAY) {
            count = 1024;
            is_rt = 1;
            ty = ids[ty].a;
        }
        if (ty >= bound) continue;

        VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (ids[ty].op == SPV_OP_TYPE_SAMPLER) {
            dtype = VK_DESCRIPTOR_TYPE_SAMPLER;
        } else if (ids[ty].op == SPV_OP_TYPE_SAMPLED_IMAGE) {
            dtype = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        } else if (ids[ty].op == SPV_OP_TYPE_IMAGE) {
            uint32_t dim = ids[ty].a;
            uint32_t sampled = ids[ty].b;
            if (dim == 5u) // DimBuffer
                dtype = (sampled == 1u) ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            else
                dtype = (sampled == 1u) ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else if (ids[ty].op == SPV_OP_TYPE_STRUCT) {
            if (sc == SPV_SC_STORAGE_BUFFER || ids[ty].is_buffer_block)
                dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            else
                dtype = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else {
            continue;
        }

        // Deduplicate if the same (set, binding) appears twice (e.g., aliased views in a shader).
        int dup = 0;
        for (uint32_t k = 0; k < out->nbindings; k++) {
            if (out->bindings[k].set == set && out->bindings[k].binding == binding) {
                if (count > out->bindings[k].count) out->bindings[k].count = count;
                if (is_rt) out->bindings[k].runtime_array = 1;
                dup = 1;
                break;
            }
        }
        if (!dup) {
            if (out->nbindings >= MAX_BINDINGS) {
                snprintf(err, err_sz, "too many bindings (>%d)", MAX_BINDINGS);
                free(ids);
                return -1;
            }
            out->bindings[out->nbindings++] = (ReflectedBinding){set, binding, dtype, count, is_rt};
            if (set > out->max_set) out->max_set = set;
        }
    }

    free(ids);
    return 0;
}

static const char *dtype_name(VkDescriptorType t) {
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLER: return "SAMPLER";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "COMBINED_IMAGE_SAMPLER";
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "SAMPLED_IMAGE";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "STORAGE_IMAGE";
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: return "UNIFORM_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return "STORAGE_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UNIFORM_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "STORAGE_BUFFER";
    default: return "OTHER";
    }
}

typedef struct {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    PFN_vkGetPipelineExecutablePropertiesKHR gprops;
    PFN_vkGetPipelineExecutableStatisticsKHR gstats;
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps;
    char device_name[256];
    char driver_name[256];
    char driver_info[256];
    char hwmon_temp2[1024];
    uint32_t min_sg;
    uint32_t max_sg;
} VkCtx;

static void find_hwmon_temp2(const VkPhysicalDevicePCIBusInfoPropertiesEXT *pci, char *out, size_t out_sz) {
    out[0] = 0;
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x", pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    DIR *d = opendir("/sys/class/drm");
    if (!d) return;
    struct dirent *e;
    char drm_dev[512] = {0};
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char p[300], real[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, real)) continue;
        const char *base = strrchr(real, '/');
        if (base && strcmp(base + 1, want) == 0) {
            snprintf(drm_dev, sizeof drm_dev, "%s", p);
            break;
        }
    }
    closedir(d);
    if (!drm_dev[0]) return;
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", drm_dev);
    d = opendir(hp);
    if (!d) return;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5) == 0) {
            snprintf(out, out_sz, "%s/%s/temp2_input", hp, e->d_name);
            break;
        }
    }
    closedir(d);
}

// Fail closed: a missing, unreadable or invalid sensor stops like an over-limit one.
static int check_hotspot_ok(const VkCtx *ctx) {
    if (!ctx->hwmon_temp2[0]) {
        fprintf(stderr, "ABORT: hotspot sensor not found\n");
        return 0;
    }
    FILE *f = fopen(ctx->hwmon_temp2, "r");
    if (!f) {
        fprintf(stderr, "ABORT: hotspot sensor unreadable (%s)\n", ctx->hwmon_temp2);
        return 0;
    }
    long mc = -1;
    if (fscanf(f, "%ld", &mc) != 1) mc = -1;
    fclose(f);
    if (mc <= 0) {
        fprintf(stderr, "ABORT: hotspot sensor reading invalid (%s)\n", ctx->hwmon_temp2);
        return 0;
    }
    if (mc >= HOTSPOT_LIMIT_MC) {
        fprintf(stderr, "ABORT: hotspot %.1f C >= %.1f C (%s)\n", mc / 1000.0, HOTSPOT_LIMIT_MC / 1000.0,
                ctx->hwmon_temp2);
        return 0;
    }
    return 1;
}

static int has_ext(const VkExtensionProperties *ex, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (strcmp(ex[i].extensionName, name) == 0) return 1;
    return 0;
}

static int vk_ctx_init(VkCtx *ctx) {
    memset(ctx, 0, sizeof *ctx);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vk_pipestats";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, NULL, &ctx->inst);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", (int)r);
        return -1;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(ctx->inst, &n, NULL);
    VkPhysicalDevice *devs = calloc(n ? n : 1, sizeof *devs);
    vkEnumeratePhysicalDevices(ctx->inst, &n, devs);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == TARGET_VENDOR && p.deviceID == TARGET_DEVICE) {
            ctx->pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!ctx->pd) {
        fprintf(stderr, "FAIL: no Vulkan device %04x:%04x (RX 9070 XT); refusing to run elsewhere\n", TARGET_VENDOR,
                TARGET_DEVICE);
        return -1;
    }

    VkPhysicalDeviceSubgroupSizeControlProperties sgprops = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
                                                   &sgprops};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(ctx->pd, &p2);

    snprintf(ctx->device_name, sizeof ctx->device_name, "%s", p2.properties.deviceName);
    snprintf(ctx->driver_name, sizeof ctx->driver_name, "%s", drv.driverName);
    snprintf(ctx->driver_info, sizeof ctx->driver_info, "%s", drv.driverInfo);
    ctx->min_sg = sgprops.minSubgroupSize;
    ctx->max_sg = sgprops.maxSubgroupSize;
    find_hwmon_temp2(&pci, ctx->hwmon_temp2, sizeof ctx->hwmon_temp2);

    if (strcmp(ctx->driver_name, "radv") != 0) {
        fprintf(stderr, "FAIL: driver is '%s', expected 'radv'\n", ctx->driver_name);
        return -1;
    }
    if (!check_hotspot_ok(ctx)) return -1;

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->pd, &nq, qp);
    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            qf = i;
            break;
        }
    }
    if (qf == UINT32_MAX) {
        for (uint32_t i = 0; i < nq; i++) {
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qf = i;
                break;
            }
        }
    }
    free(qp);
    if (qf == UINT32_MAX) {
        fprintf(stderr, "FAIL: no compute queue family\n");
        return -1;
    }

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(ctx->pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne ? ne : 1, sizeof *ex);
    vkEnumerateDeviceExtensionProperties(ctx->pd, NULL, &ne, ex);
    if (!has_ext(ex, ne, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        fprintf(stderr, "FAIL: VK_KHR_pipeline_executable_properties not supported\n");
        free(ex);
        return -1;
    }
    const char *en[4];
    uint32_t nen = 0;
    en[nen++] = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
    free(ex);

    // Query supported features so we only enable what the device exposes.
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR sup_exe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features sup13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &sup_exe};
    VkPhysicalDeviceVulkan12Features sup12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &sup13};
    VkPhysicalDeviceVulkan11Features sup11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &sup12};
    VkPhysicalDeviceFeatures2 sup2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &sup11};
    vkGetPhysicalDeviceFeatures2(ctx->pd, &sup2);

    if (!sup_exe.pipelineExecutableInfo || !sup13.subgroupSizeControl) {
        fprintf(stderr, "FAIL: required features missing (pipelineExecutableInfo=%u subgroupSizeControl=%u)\n",
                sup_exe.pipelineExecutableInfo, sup13.subgroupSizeControl);
        return -1;
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VK_TRUE;

    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fexe};
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = sup13.computeFullSubgroups;
    f13.synchronization2 = sup13.synchronization2;

    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.shaderFloat16 = sup12.shaderFloat16;
    f12.shaderInt8 = sup12.shaderInt8;
    f12.storageBuffer8BitAccess = sup12.storageBuffer8BitAccess;
    f12.uniformAndStorageBuffer8BitAccess = sup12.uniformAndStorageBuffer8BitAccess;
    f12.shaderSubgroupExtendedTypes = sup12.shaderSubgroupExtendedTypes;
    f12.shaderBufferInt64Atomics = sup12.shaderBufferInt64Atomics;
    f12.shaderSharedInt64Atomics = sup12.shaderSharedInt64Atomics;
    f12.descriptorIndexing = sup12.descriptorIndexing;
    f12.runtimeDescriptorArray = sup12.runtimeDescriptorArray;
    f12.descriptorBindingPartiallyBound = sup12.descriptorBindingPartiallyBound;
    f12.descriptorBindingVariableDescriptorCount = sup12.descriptorBindingVariableDescriptorCount;
    f12.shaderSampledImageArrayNonUniformIndexing = sup12.shaderSampledImageArrayNonUniformIndexing;
    f12.shaderStorageImageArrayNonUniformIndexing = sup12.shaderStorageImageArrayNonUniformIndexing;
    f12.shaderStorageBufferArrayNonUniformIndexing = sup12.shaderStorageBufferArrayNonUniformIndexing;
    f12.vulkanMemoryModel = sup12.vulkanMemoryModel;
    f12.vulkanMemoryModelDeviceScope = sup12.vulkanMemoryModelDeviceScope;

    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    f11.storageBuffer16BitAccess = sup11.storageBuffer16BitAccess;
    f11.uniformAndStorageBuffer16BitAccess = sup11.uniformAndStorageBuffer16BitAccess;

    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.shaderInt64 = sup2.features.shaderInt64;
    f2.features.shaderInt16 = sup2.features.shaderInt16;
    f2.features.shaderStorageImageReadWithoutFormat = sup2.features.shaderStorageImageReadWithoutFormat;
    f2.features.shaderStorageImageWriteWithoutFormat = sup2.features.shaderStorageImageWriteWithoutFormat;
    f2.features.shaderSampledImageArrayDynamicIndexing = sup2.features.shaderSampledImageArrayDynamicIndexing;
    f2.features.shaderStorageImageArrayDynamicIndexing = sup2.features.shaderStorageImageArrayDynamicIndexing;
    f2.features.shaderStorageBufferArrayDynamicIndexing = sup2.features.shaderStorageBufferArrayDynamicIndexing;
    f2.features.shaderUniformBufferArrayDynamicIndexing = sup2.features.shaderUniformBufferArrayDynamicIndexing;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nen;
    dci.ppEnabledExtensionNames = en;

    r = vkCreateDevice(ctx->pd, &dci, NULL, &ctx->dev);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDevice failed: %d\n", (int)r);
        return -1;
    }

    ctx->gprops = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(
        ctx->dev, "vkGetPipelineExecutablePropertiesKHR");
    ctx->gstats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(
        ctx->dev, "vkGetPipelineExecutableStatisticsKHR");
    ctx->greps = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
        ctx->dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    if (!ctx->gprops || !ctx->gstats || !ctx->greps) {
        fprintf(stderr, "FAIL: pipeline executable properties function pointers missing\n");
        return -1;
    }

    printf("DEVICE %s (%04x:%04x) driver=%s \"%s\" sg=[%u,%u]\n", ctx->device_name, TARGET_VENDOR, TARGET_DEVICE,
           ctx->driver_name, ctx->driver_info, ctx->min_sg, ctx->max_sg);
    return 0;
}

static void vk_ctx_destroy(VkCtx *ctx) {
    if (ctx->dev) vkDestroyDevice(ctx->dev, NULL);
    if (ctx->inst) vkDestroyInstance(ctx->inst, NULL);
    memset(ctx, 0, sizeof *ctx);
}

static void json_escape(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"') fputs("\\\"", f);
        else if (*p == '\\') fputs("\\\\", f);
        else if (*p == '\n') fputs("\\n", f);
        else if (*p == '\r') fputs("\\r", f);
        else if (*p == '\t') fputs("\\t", f);
        else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

static int compile_and_dump(const VkCtx *ctx, const char *id, const char *spv_path, const char *entry, uint32_t req_sg,
                            const char *asm_out, FILE *json_f) {
    if (!check_hotspot_ok(ctx)) return -1;

    uint8_t *code = NULL;
    size_t len = 0;
    if (read_file_bytes(spv_path, &code, &len) != 0) {
        fprintf(stderr, "FAIL [%s]: cannot read '%s': %s\n", id, spv_path, strerror(errno));
        return -1;
    }

    SpvReflection refl;
    char err[256] = {0};
    if (spv_reflect(code, len, entry, &refl, err, sizeof err) != 0) {
        fprintf(stderr, "FAIL [%s]: SPIR-V reflect failed: %s\n", id, err);
        free(code);
        return -1;
    }

    uint32_t nsets = refl.nbindings ? (refl.max_set + 1) : 0;
    VkDescriptorSetLayout dsls[MAX_SETS];
    memset(dsls, 0, sizeof dsls);

    for (uint32_t s = 0; s < nsets; s++) {
        VkDescriptorSetLayoutBinding bs[MAX_BINDINGS];
        VkDescriptorBindingFlags bflags[MAX_BINDINGS];
        uint32_t nb = 0;
        int has_rt = 0;
        for (uint32_t k = 0; k < refl.nbindings; k++) {
            if (refl.bindings[k].set != s) continue;
            bs[nb] = (VkDescriptorSetLayoutBinding){
                .binding = refl.bindings[k].binding,
                .descriptorType = refl.bindings[k].type,
                .descriptorCount = refl.bindings[k].count,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = NULL,
            };
            bflags[nb] = refl.bindings[k].runtime_array ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0u;
            if (refl.bindings[k].runtime_array) has_rt = 1;
            nb++;
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfci = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        bfci.bindingCount = nb;
        bfci.pBindingFlags = bflags;

        VkDescriptorSetLayoutCreateInfo dci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.pNext = has_rt ? &bfci : NULL;
        dci.bindingCount = nb;
        dci.pBindings = bs;
        VkResult r = vkCreateDescriptorSetLayout(ctx->dev, &dci, NULL, &dsls[s]);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "FAIL [%s]: vkCreateDescriptorSetLayout(set=%u) -> %d\n", id, s, (int)r);
            for (uint32_t j = 0; j < s; j++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[j], NULL);
            free(code);
            return -1;
        }
    }

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = nsets;
    plci.pSetLayouts = nsets ? dsls : NULL;
    plci.pushConstantRangeCount = refl.has_push_constants ? 1u : 0u;
    plci.pPushConstantRanges = refl.has_push_constants ? &pcr : NULL;

    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkResult r = vkCreatePipelineLayout(ctx->dev, &plci, NULL, &pl);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL [%s]: vkCreatePipelineLayout -> %d\n", id, (int)r);
        for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[s], NULL);
        free(code);
        return -1;
    }

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = len;
    smci.pCode = (const uint32_t *)code;
    VkShaderModule sm = VK_NULL_HANDLE;
    r = vkCreateShaderModule(ctx->dev, &smci, NULL, &sm);
    free(code);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL [%s]: vkCreateShaderModule -> %d\n", id, (int)r);
        vkDestroyPipelineLayout(ctx->dev, pl, NULL);
        for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[s], NULL);
        return -1;
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgci = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    sgci.requiredSubgroupSize = req_sg;

    VkPipelineShaderStageCreateInfo stci = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stci.pNext = (req_sg != 0) ? &sgci : NULL;
    stci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stci.module = sm;
    stci.pName = entry;

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                 VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    cpci.stage = stci;
    cpci.layout = pl;

    VkPipeline pipe = VK_NULL_HANDLE;
    r = vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
    vkDestroyShaderModule(ctx->dev, sm, NULL);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL [%s]: vkCreateComputePipelines -> %d\n", id, (int)r);
        vkDestroyPipelineLayout(ctx->dev, pl, NULL);
        for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[s], NULL);
        return -1;
    }

    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = pipe;
    uint32_t ne = 0;
    ctx->gprops(ctx->dev, &pi, &ne, NULL);
    if (ne == 0) {
        fprintf(stderr, "FAIL [%s]: 0 pipeline executables returned\n", id);
        vkDestroyPipeline(ctx->dev, pipe, NULL);
        vkDestroyPipelineLayout(ctx->dev, pl, NULL);
        for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[s], NULL);
        return -1;
    }
    VkPipelineExecutablePropertiesKHR ep = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    uint32_t one = 1;
    ctx->gprops(ctx->dev, &pi, &one, &ep);

    VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = pipe;
    ei.executableIndex = 0;

    uint32_t ns = 0;
    ctx->gstats(ctx->dev, &ei, &ns, NULL);
    VkPipelineExecutableStatisticKHR *st = calloc(ns ? ns : 1, sizeof *st);
    for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    if (ns) ctx->gstats(ctx->dev, &ei, &ns, st);

    uint32_t nr = 0;
    ctx->greps(ctx->dev, &ei, &nr, NULL);
    VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr ? nr : 1, sizeof *ir);
    for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    if (nr) ctx->greps(ctx->dev, &ei, &nr, ir);
    for (uint32_t i = 0; i < nr; i++) {
        if (ir[i].dataSize > 0) ir[i].pData = calloc(1, ir[i].dataSize + 1);
    }
    if (nr) ctx->greps(ctx->dev, &ei, &nr, ir);

    printf("PIPE %s sg_req=%u sg_actual=%u exec=\"%s\" bindings=%u", id, req_sg, ep.subgroupSize, ep.name,
           refl.nbindings);
    for (uint32_t i = 0; i < ns; i++) {
        if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
            printf(" [%s=%llu]", st[i].name, (unsigned long long)st[i].value.u64);
        else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
            printf(" [%s=%lld]", st[i].name, (long long)st[i].value.i64);
        else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
            printf(" [%s=%u]", st[i].name, st[i].value.b32);
        else
            printf(" [%s=%g]", st[i].name, st[i].value.f64);
    }
    printf("\n");

    int saved_asm = 0;
    size_t asm_bytes = 0;
    char asm_ir_name[256] = {0};
    for (uint32_t i = 0; i < nr; i++) {
        if (!ir[i].isText || !ir[i].pData) continue;
        if (strstr(ir[i].name, "Assembly") || strstr(ir[i].name, "assembly") || strstr(ir[i].name, "ISA")) {
            snprintf(asm_ir_name, sizeof asm_ir_name, "%s", ir[i].name);
            asm_bytes = ir[i].dataSize;
            if (asm_out && asm_out[0]) {
                FILE *af = fopen(asm_out, "wb");
                if (!af) {
                    fprintf(stderr, "FAIL [%s]: cannot write asm '%s': %s\n", id, asm_out, strerror(errno));
                    saved_asm = -1;
                } else {
                    fprintf(af, "# id: %s\n# executable: %s\n# sg_req: %u\n# sg_actual: %u\n# representation: %s\n", id,
                            ep.name, req_sg, ep.subgroupSize, ir[i].name);
                    for (uint32_t k = 0; k < ns; k++) {
                        if (st[k].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
                            fprintf(af, "# stat %s = %llu\n", st[k].name, (unsigned long long)st[k].value.u64);
                        else if (st[k].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
                            fprintf(af, "# stat %s = %lld\n", st[k].name, (long long)st[k].value.i64);
                        else if (st[k].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
                            fprintf(af, "# stat %s = %u\n", st[k].name, st[k].value.b32);
                        else
                            fprintf(af, "# stat %s = %g\n", st[k].name, st[k].value.f64);
                    }
                    fprintf(af, "\n%s\n", (const char *)ir[i].pData);
                    fclose(af);
                    saved_asm = 1;
                }
            } else {
                saved_asm = 1;
            }
            break;
        }
    }

    if (json_f) {
        fputs("{\"id\":", json_f);
        json_escape(json_f, id);
        fprintf(json_f, ",\"sg_req\":%u,\"sg_actual\":%u,\"bindings\":%u,\"asm_bytes\":%zu,\"asm_ir\":", req_sg,
                ep.subgroupSize, refl.nbindings, asm_bytes);
        json_escape(json_f, asm_ir_name);
        fputs(",\"driver\":", json_f);
        json_escape(json_f, ctx->driver_info);
        fputs(",\"stats\":{", json_f);
        for (uint32_t i = 0; i < ns; i++) {
            if (i) fputc(',', json_f);
            json_escape(json_f, st[i].name);
            fputc(':', json_f);
            if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
                fprintf(json_f, "%llu", (unsigned long long)st[i].value.u64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
                fprintf(json_f, "%lld", (long long)st[i].value.i64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
                fprintf(json_f, "%u", st[i].value.b32);
            else
                fprintf(json_f, "%.6g", st[i].value.f64);
        }
        fputs("}}\n", json_f);
        fflush(json_f);
    }

    for (uint32_t i = 0; i < nr; i++) free(ir[i].pData);
    free(ir);
    free(st);
    vkDestroyPipeline(ctx->dev, pipe, NULL);
    vkDestroyPipelineLayout(ctx->dev, pl, NULL);
    for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(ctx->dev, dsls[s], NULL);

    if (saved_asm <= 0) {
        fprintf(stderr, "FAIL [%s]: no assembly IR captured or write failed (saved_asm=%d)\n", id, saved_asm);
        return -1;
    }
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <spv> [--entry main] [--sg 0|32|64] [--asm-out FILE] [--json-out FILE]\n"
            "  %s --batch <manifest.tsv> [--jsonl-out FILE]\n"
            "  %s --reflect-only <spv> [--entry main]\n",
            prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    if (strcmp(argv[1], "--reflect-only") == 0) {
        if (argc < 3) {
            usage(argv[0]);
            return 2;
        }
        const char *spv = argv[2];
        const char *entry = "main";
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) entry = argv[++i];
        }
        uint8_t *code = NULL;
        size_t len = 0;
        if (read_file_bytes(spv, &code, &len) != 0) {
            fprintf(stderr, "REFLECT FAIL: cannot read '%s'\n", spv);
            return 1;
        }
        SpvReflection refl;
        char err[256] = {0};
        int rc = spv_reflect(code, len, entry, &refl, err, sizeof err);
        free(code);
        if (rc != 0) {
            fprintf(stderr, "REFLECT FAIL: %s\n", err);
            return 1;
        }
        printf("REFLECT PASS spv_ver=0x%x entry=%s bindings=%u max_set=%u push=%d\n", refl.spv_version, entry,
               refl.nbindings, refl.max_set, refl.has_push_constants);
        for (uint32_t k = 0; k < refl.nbindings; k++) {
            printf("  set=%u binding=%u type=%s count=%u rt=%d\n", refl.bindings[k].set, refl.bindings[k].binding,
                   dtype_name(refl.bindings[k].type), refl.bindings[k].count, refl.bindings[k].runtime_array);
        }
        return 0;
    }

    if (strcmp(argv[1], "--batch") == 0) {
        if (argc < 3) {
            usage(argv[0]);
            return 2;
        }
        const char *manifest = argv[2];
        const char *jsonl_path = NULL;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--jsonl-out") == 0 && i + 1 < argc) jsonl_path = argv[++i];
        }
        FILE *mf = fopen(manifest, "r");
        if (!mf) {
            fprintf(stderr, "FAIL: cannot open manifest '%s': %s\n", manifest, strerror(errno));
            return 1;
        }
        FILE *jf = NULL;
        if (jsonl_path) {
            jf = fopen(jsonl_path, "w");
            if (!jf) {
                fprintf(stderr, "FAIL: cannot open '%s': %s\n", jsonl_path, strerror(errno));
                fclose(mf);
                return 1;
            }
        }
        VkCtx ctx;
        if (vk_ctx_init(&ctx) != 0) {
            if (jf) fclose(jf);
            fclose(mf);
            return 1;
        }
        char line[4096];
        int total = 0, failed = 0;
        while (fgets(line, sizeof line, mf)) {
            char *p = line;
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == 0 || *p == '#') continue;
            size_t l = strlen(p);
            while (l > 0 && (p[l - 1] == '\n' || p[l - 1] == '\r')) p[--l] = 0;
            char *f1 = strsep(&p, "\t");
            char *f2 = strsep(&p, "\t");
            char *f3 = strsep(&p, "\t");
            char *f4 = strsep(&p, "\t");
            char *f5 = strsep(&p, "\t");
            if (!f1 || !f2 || !f3 || !f4 || !f5) {
                fprintf(stderr, "FAIL: malformed manifest line near id='%s'\n", f1 ? f1 : "");
                failed++;
                continue;
            }
            uint32_t req_sg = (uint32_t)strtoul(f4, NULL, 10);
            total++;
            if (compile_and_dump(&ctx, f1, f2, f3, req_sg, f5, jf) != 0) failed++;
        }
        vk_ctx_destroy(&ctx);
        if (jf) fclose(jf);
        fclose(mf);
        printf("BATCH SUMMARY total=%d ok=%d failed=%d\n", total, total - failed, failed);
        return failed ? 1 : 0;
    }

    const char *spv = argv[1];
    const char *entry = "main";
    uint32_t req_sg = 0;
    const char *asm_out = NULL;
    const char *json_out = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) entry = argv[++i];
        else if (strcmp(argv[i], "--sg") == 0 && i + 1 < argc) req_sg = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--asm-out") == 0 && i + 1 < argc) asm_out = argv[++i];
        else if (strcmp(argv[i], "--json-out") == 0 && i + 1 < argc) json_out = argv[++i];
        else {
            usage(argv[0]);
            return 2;
        }
    }

    VkCtx ctx;
    if (vk_ctx_init(&ctx) != 0) return 1;
    FILE *jf = NULL;
    if (json_out) {
        jf = fopen(json_out, "w");
        if (!jf) {
            fprintf(stderr, "FAIL: cannot open '%s': %s\n", json_out, strerror(errno));
            vk_ctx_destroy(&ctx);
            return 1;
        }
    }
    int rc = compile_and_dump(&ctx, spv, spv, entry, req_sg, asm_out, jf);
    if (jf) fclose(jf);
    vk_ctx_destroy(&ctx);
    return rc ? 1 : 0;
}
