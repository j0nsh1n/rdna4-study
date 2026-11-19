#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define VENDOR 0x1002u
#define DEVICE 0x7550u
#define SYSFS "/sys/class/drm/card1/device/"
#define NLANES 64
#define NWARM 50
#define MAXN 2000

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult r_ = (x);                                                                                             \
        if (r_ != VK_SUCCESS) {                                                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);                        \
            exit(2);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct {
    const char *name;
    uint32_t family, valid_bits;
    VkQueue q;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    VkQueryPool qp;
    VkBuffer buf;
    VkDeviceMemory mem;
    uint32_t *map;
    VkDescriptorSet ds;
    uint32_t base;
} Lane;

static VkDevice dev;
static VkPipeline pipe;
static VkPipelineLayout pl;
static double ts_period_ns;
static int fail_count;
static FILE *csv;

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static void pause_ms(int ms) {
    if (ms <= 0) return;
    struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double *sorted_copy(const double *v, int n) {
    double *c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    return c;
}

static double pick(const double *s, int n, double q) { return s[(int)(q * (n - 1) + 0.5)]; }

static double median_sorted(const double *s, int n) { return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }

static void verdict(const char *name, int ok, const char *text) {
    printf("VERDICT vksched %s: %s %s\n", name, ok ? "PASS" : "FAIL", text);
    if (!ok) fail_count++;
}

static int read_sclk_mhz(void) {
    FILE *f = fopen(SYSFS "pp_dpm_sclk", "r");
    if (!f) return -1;
    char line[128];
    int mhz = -1;
    while (fgets(line, sizeof line, f))
        if (strchr(line, '*')) {
            char *c = strchr(line, ':');
            if (c) mhz = atoi(c + 1);
        }
    fclose(f);
    return mhz;
}

static int read_busy(void) {
    FILE *f = fopen(SYSFS "gpu_busy_percent", "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static void clocks(const char *mode, const char *when) {
    printf("CLOCKS %s %s: sclk %d MHz gpu_busy %d %%\n", mode, when, read_sclk_mhz(), read_busy());
}

static uint32_t mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// One sample: wall time from just before submit to the fence, and GPU time from the two timestamps.
static void sample(Lane *l, double *wall_us, double *gpu_us) {
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &l->cb;
    double t0 = now_ns();
    VK_CHECK(vkQueueSubmit(l->q, 1, &si, l->fence));
    VK_CHECK(vkWaitForFences(dev, 1, &l->fence, VK_TRUE, UINT64_MAX));
    double t1 = now_ns();
    VK_CHECK(vkResetFences(dev, 1, &l->fence));
    uint64_t ts[2];
    VK_CHECK(vkGetQueryPoolResults(dev, l->qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    uint64_t mask = l->valid_bits >= 64 ? UINT64_MAX : ((1ull << l->valid_bits) - 1);
    *wall_us = (t1 - t0) / 1e3;
    *gpu_us = (double)((ts[1] - ts[0]) & mask) * ts_period_ns / 1e3;
}

typedef struct {
    double wall[MAXN], gpu[MAXN];
    int n;
} Series;

static void report(const char *mode, const char *qname, const Series *s, int *any_zero_gpu, int *any_gpu_ge_wall,
                   double *gpu_med_out) {
    int n = s->n;
    double *w = sorted_copy(s->wall, n), *g = sorted_copy(s->gpu, n);
    double wm = median_sorted(w, n), gm = median_sorted(g, n);
    printf("RESULT %s %s wall_us %.2f %.2f %.2f %.2f %.2f %.2f %d\n", mode, qname, wm, pick(w, n, 0.10), pick(w, n, 0.90),
           pick(w, n, 0.99), w[0], w[n - 1], n);
    printf("RESULT %s %s gpu_us %.3f %.3f %.3f %.3f\n", mode, qname, gm, pick(g, n, 0.10), pick(g, n, 0.90), g[n - 1]);
    int slow = 0;
    double *sl = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
        if (s->wall[i] > 2 * wm) sl[slow++] = s->wall[i];
    double *sls = sorted_copy(sl, slow ? slow : 1);
    printf("RESULT %s %s slow_frac=%.4f slow_median_us=%.2f\n", mode, qname, (double)slow / n,
           slow ? median_sorted(sls, slow) : 0.0);
    int hist[20] = {0};
    for (int i = 0; i < n; i++) {
        int b = s->wall[i] <= 1.0 ? 0 : (int)(log10(s->wall[i]) * 5.0);
        hist[b < 0 ? 0 : b > 19 ? 19 : b]++;
    }
    for (int b = 0; b < 20; b++) printf("HIST %s %s %.2f %d\n", mode, qname, pow(10.0, b / 5.0), hist[b]);
    for (int i = 0; i < n; i++) {
        if (!(s->gpu[i] > 0)) *any_zero_gpu = 1;
        if (!(s->gpu[i] < s->wall[i])) *any_gpu_ge_wall = 1;
    }
    *gpu_med_out = gm;
    free(w);
    free(g);
    free(sl);
    free(sls);
}

// Leftovers C, SC-12: a second thread resubmits the graphics lane's dispatch back to back (fence-waited), so the
// card is never idle, while the main thread times the compute lane.
static volatile int g_keep_run;
static long g_keep_n;
static void *keepalive(void *arg) {
    Lane *l = arg;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &l->cb;
    while (g_keep_run) {
        VK_CHECK(vkQueueSubmit(l->q, 1, &si, l->fence));
        VK_CHECK(vkWaitForFences(dev, 1, &l->fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &l->fence));
        g_keep_n++;
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *spv_path = "build/vk/vksched_one.spv", *out_dir = NULL;
    int brk = 0, keep = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--spv") && i + 1 < argc) spv_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--break")) brk = 1;
        else if (!strcmp(argv[i], "--keepalive")) keep = 1;
        else {
            fprintf(stderr, "usage: vksched [--spv path] [--out dir] [--break] [--keepalive]\n");
            return 2;
        }
    }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vksched";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice *devs = calloc(nd ? nd : 1, sizeof *devs);
    vkEnumeratePhysicalDevices(inst, &nd, devs);
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < nd; i++) {
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.vendorID == VENDOR && props.deviceID == DEVICE) {
            pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!pd) {
        printf("no Vulkan device %04x:%04x (RX 9070 XT); refusing to run on any other GPU\n", VENDOR, DEVICE);
        return 3;
    }
    vkGetPhysicalDeviceProperties(pd, &props);
    ts_period_ns = props.limits.timestampPeriod;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("DEVICE %s (%04x:%04x) timestampPeriod %.3f ns\n", props.deviceName, props.vendorID, props.deviceID, ts_period_ns);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t gfam = UINT32_MAX, cfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        printf("QUEUEFAMILY %u flags 0x%x%s%s%s%s count %u timestampValidBits %u\n", i, qp[i].queueFlags,
               qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ? " GRAPHICS" : "",
               qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT ? " COMPUTE" : "",
               qp[i].queueFlags & VK_QUEUE_TRANSFER_BIT ? " TRANSFER" : "",
               qp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT ? " SPARSE" : "", qp[i].queueCount,
               qp[i].timestampValidBits);
        if (gfam == UINT32_MAX && (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            qp[i].timestampValidBits)
            gfam = i;
        if (cfam == UINT32_MAX && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            qp[i].timestampValidBits)
            cfam = i;
    }
    if (gfam == UINT32_MAX) {
        printf("no graphics+compute family with timestamps; nothing to test\n");
        return 3;
    }
    if (cfam == UINT32_MAX) printf("NOTE no compute-only family exists; testing graphics only\n");
    int nl = cfam == UINT32_MAX ? 1 : 2;
    Lane L[2] = {{.name = "graphics", .family = gfam, .base = 1000u}, {.name = "compute", .family = cfam, .base = 2000000u}};
    L[0].valid_bits = qp[gfam].timestampValidBits;
    if (nl == 2) L[1].valid_bits = qp[cfam].timestampValidBits;
    free(qp);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci[2] = {{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    qci[0].queueFamilyIndex = gfam;
    qci[0].queueCount = 1;
    qci[0].pQueuePriorities = &prio;
    qci[1].queueFamilyIndex = cfam;
    qci[1].queueCount = 1;
    qci[1].pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = (uint32_t)nl;
    dci.pQueueCreateInfos = qci;
    VK_CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &b;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

    FILE *f = fopen(spv_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", spv_path);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)len);
    if (fread(code, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "short read of %s\n", spv_path);
        return 2;
    }
    fclose(f);
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = (size_t)len;
    smi.pCode = code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    free(code);
    VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkDescriptorPool dp;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));

    for (int i = 0; i < nl; i++) {
        Lane *l = &L[i];
        vkGetDeviceQueue(dev, l->family, 0, &l->q);
        VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = l->family;
        VK_CHECK(vkCreateCommandPool(dev, &cpi, NULL, &l->pool));
        VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = l->pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &l->cb));
        VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev, &fci, NULL, &l->fence));
        VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(dev, &qpi, NULL, &l->qp));

        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = 4096;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &l->buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, l->buf, &mr);
        // Host-visible so the result can be read back without a copy; prefer the device-local BAR heap.
        VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t t = mem_type(&mp, mr.memoryTypeBits, hv | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (t == UINT32_MAX) t = mem_type(&mp, mr.memoryTypeBits, hv);
        if (t == UINT32_MAX) {
            fprintf(stderr, "no host-visible memory type\n");
            return 2;
        }
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = t;
        VK_CHECK(vkAllocateMemory(dev, &mai, NULL, &l->mem));
        VK_CHECK(vkBindBufferMemory(dev, l->buf, l->mem, 0));
        VK_CHECK(vkMapMemory(dev, l->mem, 0, VK_WHOLE_SIZE, 0, (void **)&l->map));
        memset(l->map, 0xff, 4096);

        VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dp;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &l->ds));
        VkDescriptorBufferInfo dbi = {l->buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = l->ds;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

        VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(l->cb, &cbi));
        vkCmdResetQueryPool(l->cb, l->qp, 0, 2);
        vkCmdBindPipeline(l->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(l->cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &l->ds, 0, NULL);
        vkCmdPushConstants(l->cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &l->base);
        vkCmdWriteTimestamp(l->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, l->qp, 0);
        vkCmdDispatch(l->cb, 1, 1, 1);
        vkCmdWriteTimestamp(l->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, l->qp, 1);
        VK_CHECK(vkEndCommandBuffer(l->cb));
    }

    if (out_dir) {
        char p[1024];
        snprintf(p, sizeof p, "%s/samples.csv", out_dir);
        csv = fopen(p, "w");
        if (!csv) {
            fprintf(stderr, "cannot write %s\n", p);
            return 2;
        }
        fprintf(csv, "mode,queue,i,wall_us,gpu_us\n");
    }

    if (keep) {
        // SC-12: compute lane alone, blocks of 500 samples, alternating without and with the keep-alive thread
        // (4 blocks each); slow = above twice the median of the no-keep-alive samples.
        if (nl < 2) {
            verdict("keepalive_calib", 0, "no compute-only queue family");
            return 1;
        }
        static Series K[2];
        K[0].n = K[1].n = 0;
        double w, g;
        for (int i = 0; i < NWARM; i++) sample(&L[1], &w, &g);
        long keep_total = 0;
        for (int blk = 0; blk < 8; blk++) {
            const int on = blk & 1;
            pthread_t th;
            if (on) {
                g_keep_run = 1;
                g_keep_n = 0;
                pthread_create(&th, NULL, keepalive, &L[0]);
                pause_ms(5);
            }
            for (int i = 0; i < 250 && K[on].n < MAXN; i++) {
                sample(&L[1], &K[on].wall[K[on].n], &K[on].gpu[K[on].n]);
                K[on].n++;
            }
            if (on) {
                g_keep_run = 0;
                pthread_join(th, NULL);
                keep_total += g_keep_n;
            }
        }
        int az = 0, ag = 0;
        double gm;
        report("keepalive_off", "compute", &K[0], &az, &ag, &gm);
        report("keepalive_on", "compute", &K[1], &az, &ag, &gm);
        double *w0 = sorted_copy(K[0].wall, K[0].n);
        const double med0 = median_sorted(w0, K[0].n);
        int slow0 = 0, slow1 = 0;
        for (int i = 0; i < K[0].n; i++) slow0 += K[0].wall[i] > 2 * med0;
        for (int i = 0; i < K[1].n; i++) slow1 += K[1].wall[i] > 2 * med0;
        const double f0 = (double)slow0 / K[0].n, f1 = (double)slow1 / K[1].n;
        printf("RESULT keepalive graphics-lane submissions during the on blocks: %ld\n", keep_total);
        printf("INFO keepalive slow share (above 2 x %.1f us): off %.3f on %.3f\n", med0, f0, f1);
        // calibration: scheduling item 6, compute queue alone, 30 to 42 %% of submissions slow (+-10 points);
        // --break claims 0 %% when alone
        const int ok = brk ? f0 <= 0.01 : (f0 >= 0.20 && f0 <= 0.52);
        char msg[256];
        snprintf(msg, sizeof msg, "compute queue alone: %.3f of submissions above twice the median (%s); %ld keep-alive "
                 "submissions", f0, brk ? "break: claims 0" : "scheduling item 6: 0.30 to 0.42, +-0.10", keep_total);
        verdict("keepalive_calib", ok && keep_total > 0, msg);
        free(w0);
        return fail_count ? 1 : 0;
    }

    static Series S[2];
    int any_zero = 0, any_ge = 0;
    double gpu_med_min = 1e30, gpu_med_max = 0, wall_med_compute_p0 = -1;
    double run_t0 = now_ns();

    struct { const char *name; int pause, n; } modes[] = {
        {"p0ms", 0, 2000}, {"p1ms", 1, 2000}, {"p10ms", 10, 600}, {"alone", 0, 2000}};
    for (int m = 0; m < 4; m++) {
        const char *mode = modes[m].name;
        int n = modes[m].n, pause = modes[m].pause;
        printf("MODE %s pause %d ms samples_per_queue %d%s\n", mode, pause, n,
               m == 2 ? " (reduced from 2000 to keep the run short)" : "");
        clocks(mode, "start");
        S[0].n = S[1].n = n;
        double w, g;
        if (m < 3) {
            for (int i = 0; i < NWARM; i++)
                for (int k = 0; k < nl; k++) {
                    sample(&L[k], &w, &g);
                    pause_ms(pause);
                }
            for (int i = 0; i < n; i++)
                for (int k = 0; k < nl; k++) {
                    int idx = (i & 1) ? nl - 1 - k : k;
                    sample(&L[idx], &S[idx].wall[i], &S[idx].gpu[i]);
                    pause_ms(pause);
                }
        } else {
            // compute first, then graphics, so each queue runs on its own with no neighbour submitting.
            for (int k = nl - 1; k >= 0; k--) {
                for (int i = 0; i < NWARM; i++) sample(&L[k], &w, &g);
                for (int i = 0; i < n; i++) sample(&L[k], &S[k].wall[i], &S[k].gpu[i]);
            }
        }
        clocks(mode, "end");
        for (int k = 0; k < nl; k++) {
            double gm;
            report(mode, L[k].name, &S[k], &any_zero, &any_ge, &gm);
            if (gm < gpu_med_min) gpu_med_min = gm;
            if (gm > gpu_med_max) gpu_med_max = gm;
            if (m == 0 && k == 1) {
                double *w2 = sorted_copy(S[k].wall, n);
                wall_med_compute_p0 = median_sorted(w2, n);
                free(w2);
            }
            if (csv)
                for (int i = 0; i < n; i++)
                    fprintf(csv, "%s,%s,%d,%.3f,%.4f\n", mode, L[k].name, i, S[k].wall[i], S[k].gpu[i]);
        }
    }
    if (csv) fclose(csv);
    printf("RUNTIME %.1f s\n", (now_ns() - run_t0) / 1e9);

    int bad = 0;
    for (int k = 0; k < nl; k++)
        for (int i = 0; i < NLANES; i++)
            if (L[k].map[i] != L[k].base + (uint32_t)i) bad++;
    char msg[256];
    snprintf(msg, sizeof msg, "%d of %d buffer values wrong", bad, nl * NLANES);
    verdict("buffer_content", bad == 0, msg);
    snprintf(msg, sizeof msg, "gpu time > 0 for all samples: %s; gpu time < wall time for all samples: %s",
             any_zero ? "no" : "yes", any_ge ? "no" : "yes");
    verdict("gpu_below_wall", !any_zero && !any_ge, msg);
    snprintf(msg, sizeof msg, "timestampPeriod %.3f ns, GPU dispatch medians %.3f..%.3f us (want 0.05..50 us)",
             ts_period_ns, gpu_med_min, gpu_med_max);
    verdict("gpu_time_sane", ts_period_ns > 0.5 && ts_period_ns < 1000 && gpu_med_min >= 0.05 && gpu_med_max <= 50.0, msg);
    if (brk) {
        snprintf(msg, sizeof msg, "deliberate break: compute wall median %.2f us claimed under 1 us", wall_med_compute_p0);
        verdict("break_compute_wall_under_1us", wall_med_compute_p0 >= 0 && wall_med_compute_p0 < 1.0, msg);
    }
    return fail_count ? 1 : 0;
}
