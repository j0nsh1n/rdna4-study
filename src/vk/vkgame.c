// Gaming-round probes on the RX 9070 XT through plain Vulkan (RADV): game-like work and what the card and the driver
// do with it. Modes:
//   info                 device, queue families, calibrated-timestamp domains, machine code of every pipeline, and
//                        the check that the shader's REALTIME counter and the queue timestamps share one clock
//   chain <spec>...      P1: lists of short compute dispatches in one command buffer (or as separate submissions);
//                        per dispatch: queue timestamps and, from inside the shader, REALTIME and the cycle counter
//                        of the first wave of stamped groups at entry and exit
//   lone                 P2 calibration: the study's one-group dispatch on both queues (vksched.c's design)
//   share <spec>...      P2: compute submissions beside an offscreen draw load on the graphics queue
//   burst <spec>...      E1-E3: the dense FMA load (gdense.comp) as one dispatch after idle or as bursts at a cadence
//   rt <spec>            E7: the draw pass without ALU into targets of two sizes, interleaved
//   slow <spec>          E6: the one-group probe on the compute queue, blocking wait against spin-polling the fence,
//                        submitting thread pinned to one CPU against unpinned, interleaved in rounds
//   listing              machine code of every pipeline only (no dispatch)
// Every mode samples sysfs and gpu_metrics every 5 ms from a host thread (read-only), and the active power profile
// every 10 ms. At 88 C hotspot it stops submitting and sets the dense load's stop word (gaming-round thermal rule
// after review). Environment: VKG_OUT (run folder for CSV files and listings), VKG_SPV (folder of the .spv files),
// VKG_TAG (file name prefix), VKG_FAMILIES (both | g | c: which queue families the device creates; E4),
// VKG_HOT_STOP_C (F5: a lower stop limit, either absolute in C, at most 88, or "+N": N C above the hotspot read when a
// dense arm's load starts; the end-to-end test of the stop path).
// No window, no swapchain, no display connection.
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <vulkan/vulkan.h>

#include "vkgame.h"



// ---------------------------------------------------------------------------------------------------------------
// time and statistics

double mono_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

// Sleeps until 0.3 ms before t, then spins: host timing within a few microseconds.
void until_ns(double t) {
    for (;;) {
        double d = t - mono_ns();
        if (d <= 0) return;
        if (d > 300e3) {
            double s = d - 300e3;
            struct timespec ts = {(time_t)(s / 1e9), (long)fmod(s, 1e9)};
            nanosleep(&ts, NULL);
        }
    }
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// Nearest-rank percentile of the finite values (p in 0..1).
double pctl(const double *v, int n, double p) {
    double *c = malloc(sizeof(double) * (n ? n : 1));
    int m = 0;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i])) c[m++] = v[i];
    double r = NAN;
    if (m) {
        qsort(c, m, sizeof(double), cmp_d);
        int i = (int)llround(p * (m - 1));
        r = c[i < 0 ? 0 : i >= m ? m - 1 : i];
    }
    free(c);
    return r;
}

int nfinite(const double *v, int n) {
    int m = 0;
    for (int i = 0; i < n; i++) m += isfinite(v[i]) != 0;
    return m;
}

// One line with the whole distribution: n, min, median, p95, p99, max and the index of the largest sample.
void dist(const char *tag, const char *what, const double *v, int n) {
    int m = nfinite(v, n), at = -1;
    double mx = -INFINITY;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i]) && v[i] > mx) mx = v[i], at = i;
    if (!m) {
        printf("DIST %s %s n=0\n", tag, what);
        return;
    }
    printf("DIST %s %s n=%d min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f max_at=%d\n", tag, what, m, pctl(v, n, 0),
           pctl(v, n, .5), pctl(v, n, .95), pctl(v, n, .99), mx, at);
}

double frac_above(const double *v, int n, double lim) {
    int m = 0, k = 0;
    for (int i = 0; i < n; i++)
        if (isfinite(v[i])) {
            m++;
            k += v[i] > lim;
        }
    return m ? (double)k / m : NAN;
}

int g_fail;
void verdict(const char *name, int ok, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#include <stdarg.h>
void verdict(const char *name, int ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("VERDICT vkgame %s: %s ", name, ok ? "PASS" : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!ok) g_fail++;
}

// ---------------------------------------------------------------------------------------------------------------
// sysfs (read-only)

char g_drm[512], g_hw[1024], g_out[512], g_spv[512], g_tag[128];

double read_num(const char *path, double scale) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    double v = -1;
    if (fscanf(f, "%lf", &v) != 1) v = -1;
    fclose(f);
    return v < 0 ? -1 : v * scale;
}

int star_mhz(const char *path) {
    FILE *f = fopen(path, "r");
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

double hotspot(void) {
    char p[1100];
    snprintf(p, sizeof p, "%s/temp2_input", g_hw);
    return read_num(p, 1e-3);
}

// gpu_metrics format 1, content 3 (struct gpu_metrics_v1_3, 120 bytes; the study checked the layout of bytes 68, 106
// and 112 on this card). Offsets used here: 16 average_gfx_activity, 22 average_socket_power, 32 system_clock_counter
// (ns), 40 average_gfxclk_frequency, 54 current_gfxclk, 68 throttle_status, 96 firmware_timestamp (10 ns), 106
// voltage_gfx, 112 indep_throttle_status.

static void read_metrics(Smp *s) {
    char p[600];
    snprintf(p, sizeof p, "%s/gpu_metrics", g_drm);
    FILE *f = fopen(p, "rb");
    unsigned char b[128];
    s->metrics_ok = 0;
    if (!f) return;
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    if (n < 120 || b[0] != 120 || b[1] != 0 || b[2] != 1 || b[3] != 3) return;
    uint16_t u;
    memcpy(&u, b + 16, 2), s->act = u;
    memcpy(&u, b + 22, 2), s->sock_w = u;
    memcpy(&s->sysns, b + 32, 8);
    memcpy(&u, b + 40, 2), s->avg_gfx = u;
    memcpy(&u, b + 54, 2), s->cur_gfx = u;
    memcpy(&s->thr, b + 68, 4);
    memcpy(&s->fwts, b + 96, 8);
    memcpy(&u, b + 106, 2), s->vgfx = u;
    memcpy(&s->ithr, b + 112, 8);
    s->metrics_ok = 1;
}

Smp *g_smp;
volatile int g_nsmp, g_srun, g_hot_stop;
volatile uint32_t *volatile g_stop_word;
#define HOT_STOP_C HOT_STOP_LIMIT_C
// The limit the sampler applies (F5 lowers it by VKG_HOT_STOP_C); g_hot_rel > 0: set per arm to its start hotspot + rel.
volatile double g_hot_stop_c = HOT_STOP_C;
double g_hot_rel;
double g_hottest;
static pthread_t g_sthr;

// The active power profile: the index on the line whose name carries '*' (pp_power_profile_mode, read-only).
static int profile_mode(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int idx = -1;
    while (fgets(line, sizeof line, f))
        if (strchr(line, '*')) {
            idx = atoi(line);
            break;
        }
    fclose(f);
    return idx;
}

static void *sampler_main(void *arg) {
    (void)arg;
    char ps[600], pb[600], pp[1100];
    snprintf(ps, sizeof ps, "%s/pp_dpm_sclk", g_drm);
    snprintf(pb, sizeof pb, "%s/gpu_busy_percent", g_drm);
    snprintf(pp, sizeof pp, "%s/power1_average", g_hw);
    char pm[600];
    snprintf(pm, sizeof pm, "%s/pp_power_profile_mode", g_drm);
    int prof = -1;
    double t0 = mono_ns();
    for (long i = 1; g_srun; i++) {
        if (g_nsmp < MAXSMP) {
            Smp s = {0};
            s.t = mono_ns();
            s.sclk = star_mhz(ps);
            s.busy = (int)read_num(pb, 1);
            s.hot = hotspot();
            s.pw = read_num(pp, 1e-6);
            read_metrics(&s);
            if (i % 2 == 1) prof = profile_mode(pm);
            s.prof = prof;
            g_smp[g_nsmp] = s;
            g_nsmp++;
            if (s.hot > g_hottest) g_hottest = s.hot;
            if (s.hot >= g_hot_stop_c) {
                if (!g_hot_stop)
                    printf("THERMAL hotspot %.0f C: stopping the load (sampler's tripwire, limit %.0f C) at mono_ns=%.0f "
                           "stop_word_armed=%d\n",
                           s.hot, g_hot_stop_c, s.t, g_stop_word != NULL);
                g_hot_stop = 1;
                volatile uint32_t *w = g_stop_word;
                if (w) stop_all(w);
            }
            if (s.hot >= 100.0) {
                printf("THERMAL hotspot %.0f C: ending the process\n", s.hot);
                fflush(stdout);
                _exit(4);
            }
        }
        until_ns(t0 + i * 5e6);
    }
    return NULL;
}

// VKG_SAMPLER=off leaves the sampler thread out (the thermal stop then rests on the script's checks).
static void sampler_start(void) {
    g_smp = calloc(MAXSMP, sizeof(Smp));
    const char *hs = getenv("VKG_HOT_STOP_C");
    if (hs && *hs) {
        const double v = atof(hs + (*hs == '+'));
        if (*hs == '+' && v > 0 && v <= 20) g_hot_rel = v;  // 88 C until a dense arm's load starts
        else if (*hs != '+' && v > 0 && v <= HOT_STOP_C) g_hot_stop_c = v;
        else {
            printf("VERDICT vkgame hot_stop_option: FAIL VKG_HOT_STOP_C=%s (absolute at most 88, or +N with N up to 20)\n", hs);
            exit(2);
        }
        printf("INFO hotspot stop limit lowered by VKG_HOT_STOP_C=%s: %.0f C now%s\n", hs, g_hot_stop_c,
               g_hot_rel > 0 ? "; lowered to the start hotspot + N when each dense arm's load starts" : "");
    }
    const char *e = getenv("VKG_SAMPLER");
    if (e && !strcmp(e, "off")) {
        printf("INFO sampler off\n");
        return;
    }
    g_srun = 1;
    pthread_create(&g_sthr, NULL, sampler_main, NULL);
}

static void sampler_stop_write(void) {
    if (!g_srun) return;
    g_srun = 0;
    pthread_join(g_sthr, NULL);
    if (!g_out[0]) return;
    char p[800];
    snprintf(p, sizeof p, "%s/%s-sysfs.csv", g_out, g_tag);
    FILE *f = fopen(p, "w");
    if (!f) return;
    fprintf(f, "t_mono_ns,sclk_MHz,busy_pct,hotspot_C,power_W_reported,metrics_ok,gfx_activity,socket_power_W,"
               "avg_gfxclk_MHz,cur_gfxclk_MHz,vgfx_mV,throttle_status,indep_throttle_status,fw_timestamp,sys_clock_ns,"
               "power_profile\n");
    for (int i = 0; i < g_nsmp; i++) {
        Smp *s = &g_smp[i];
        fprintf(f, "%.0f,%d,%d,%.1f,%.0f,%d,%d,%d,%d,%d,%d,0x%x,0x%llx,%llu,%llu,%d\n", s->t, s->sclk, s->busy, s->hot,
                s->pw, s->metrics_ok, s->act, s->sock_w, s->avg_gfx, s->cur_gfx, s->vgfx, s->thr,
                (unsigned long long)s->ithr, (unsigned long long)s->fwts, (unsigned long long)s->sysns, s->prof);
    }
    fclose(f);
}

// State of the card over a host-time window, from the 5 ms samples.
void state_line(const char *tag, double t0, double t1) {
    double sc[MAXSMP / 4], cg[MAXSMP / 4], bu[MAXSMP / 4], ho[MAXSMP / 4], pw[MAXSMP / 4];
    int n = 0;
    uint64_t ithr = 0;
    unsigned profs = 0;
    for (int i = 0; i < g_nsmp && n < MAXSMP / 4; i++)
        if (g_smp[i].t >= t0 && g_smp[i].t <= t1) {
            if (g_smp[i].prof >= 0 && g_smp[i].prof < 31) profs |= 1u << g_smp[i].prof;
            sc[n] = g_smp[i].sclk;
            cg[n] = g_smp[i].metrics_ok ? g_smp[i].cur_gfx : NAN;
            bu[n] = g_smp[i].busy;
            ho[n] = g_smp[i].hot;
            pw[n] = g_smp[i].pw;
            ithr |= g_smp[i].ithr;
            n++;
        }
    if (!n) {
        printf("STATE %s no 5 ms sample in the window (%.2f ms)\n", tag, (t1 - t0) / 1e6);
        return;
    }
    printf("STATE %s samples=%d sclk_MHz(min/med/max)=%.0f/%.0f/%.0f cur_gfxclk_MHz(min/med/max)=%.0f/%.0f/%.0f "
           "busy_pct(min/med/max)=%.0f/%.0f/%.0f hotspot_C(min/max)=%.0f/%.0f power_W_reported(min/max)=%.0f/%.0f "
           "indep_throttle_or=0x%llx power_profiles_seen=0x%x\n",
           tag, n, pctl(sc, n, 0), pctl(sc, n, .5), pctl(sc, n, 1), pctl(cg, n, 0), pctl(cg, n, .5), pctl(cg, n, 1),
           pctl(bu, n, 0), pctl(bu, n, .5), pctl(bu, n, 1), pctl(ho, n, 0), pctl(ho, n, 1), pctl(pw, n, 0),
           pctl(pw, n, 1), (unsigned long long)ithr, profs);
}

// The 5 ms sample nearest to host time t.
const Smp *smp_at(double t) {
    const Smp *best = NULL;
    double bd = 1e18;
    for (int i = 0; i < g_nsmp; i++) {
        double d = fabs(g_smp[i].t - t);
        if (d < bd) bd = d, best = &g_smp[i];
    }
    return best;
}

// ---------------------------------------------------------------------------------------------------------------
// device


VkInstance g_inst;
VkPhysicalDevice g_pd;
VkDevice g_dev;
double g_tsp;  // ns per timestamp tick
Qu QU[2];      // 0: graphics family, 1: compute-only family
VkPhysicalDeviceMemoryProperties g_mp;
PFN_vkGetCalibratedTimestampsKHR p_cal;
VkDescriptorSetLayout g_dsl;
VkPipelineLayout g_cpl;
VkDescriptorPool g_dpool;

static void find_sysfs(const VkPhysicalDevicePCIBusInfoPropertiesEXT *pci) {
    char want[32];
    snprintf(want, sizeof want, "%04x:%02x:%02x.%x", pci->pciDomain, pci->pciBus, pci->pciDevice, pci->pciFunction);
    g_drm[0] = g_hw[0] = 0;
    DIR *d = opendir("/sys/class/drm");
    struct dirent *e;
    while (d && (e = readdir(d))) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char p[300], real[PATH_MAX];
        snprintf(p, sizeof p, "/sys/class/drm/%s/device", e->d_name);
        if (!realpath(p, real)) continue;
        const char *base = strrchr(real, '/');
        if (base && !strcmp(base + 1, want)) snprintf(g_drm, sizeof g_drm, "%s", p);
    }
    if (d) closedir(d);
    if (!g_drm[0]) return;
    char hp[600];
    snprintf(hp, sizeof hp, "%s/hwmon", g_drm);
    d = opendir(hp);
    while (d && (e = readdir(d)))
        if (!strncmp(e->d_name, "hwmon", 5)) snprintf(g_hw, sizeof g_hw, "%s/%s", hp, e->d_name);
    if (d) closedir(d);
}

static void dev_init(int report) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdna4-gaming-probes";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, NULL, &g_inst));
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_inst, &n, NULL);
    VkPhysicalDevice *devs = calloc(n ? n : 1, sizeof *devs);
    vkEnumeratePhysicalDevices(g_inst, &n, devs);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == VENDOR && p.deviceID == DEVICE) {
            g_pd = devs[i];
            break;
        }
    }
    free(devs);
    if (!g_pd) {
        printf("VERDICT vkgame device: FAIL no Vulkan device %04x:%04x (RX 9070 XT); refusing to run elsewhere\n", VENDOR,
               DEVICE);
        exit(1);
    }
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &pci};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(g_pd, &p2);
    g_tsp = p2.properties.limits.timestampPeriod;
    find_sysfs(&pci);
    printf("DEVICE %s (%04x:%04x) PCI %04x:%02x:%02x.%x driver %s %s; timestampPeriod %.3f ns; sysfs %s; hwmon %s\n",
           p2.properties.deviceName, p2.properties.vendorID, p2.properties.deviceID, pci.pciDomain, pci.pciBus,
           pci.pciDevice, pci.pciFunction, drv.driverName, drv.driverInfo, g_tsp, g_drm[0] ? g_drm : "(not found)",
           g_hw[0] ? g_hw : "(not found)");
    if (strcmp(drv.driverName, "radv") || !strstr(p2.properties.deviceName, "9070 XT")) {
        printf("VERDICT vkgame device: FAIL driver %s, device %s: not RADV on the RX 9070 XT\n", drv.driverName,
               p2.properties.deviceName);
        exit(1);
    }
    if (!g_hw[0] || hotspot() < 0) {
        printf("VERDICT vkgame device: FAIL hotspot sensor not found; refusing to run\n");
        exit(1);
    }
    vkGetPhysicalDeviceMemoryProperties(g_pd, &g_mp);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    QU[0].fam = QU[1].fam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if (report)
            printf("QUEUEFAMILY %u flags 0x%x%s%s%s count %u timestampValidBits %u\n", i, qp[i].queueFlags,
                   qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ? " GRAPHICS" : "",
                   qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT ? " COMPUTE" : "",
                   qp[i].queueFlags & VK_QUEUE_TRANSFER_BIT ? " TRANSFER" : "", qp[i].queueCount,
                   qp[i].timestampValidBits);
        if (QU[0].fam == UINT32_MAX && (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits)
            QU[0].fam = i, QU[0].valid = qp[i].timestampValidBits;
        if (QU[1].fam == UINT32_MAX && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits)
            QU[1].fam = i, QU[1].valid = qp[i].timestampValidBits;
    }
    free(qp);
    if (QU[0].fam == UINT32_MAX || QU[1].fam == UINT32_MAX) {
        printf("VERDICT vkgame queues: FAIL need a graphics family and a compute-only family with timestamps\n");
        exit(1);
    }
    QU[0].name = "graphics";
    QU[1].name = "compute";

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexe = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fexe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceShaderClockFeaturesKHR fclk = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, &fexe};
    fclk.shaderSubgroupClock = fclk.shaderDeviceClock = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fclk};
    f13.subgroupSizeControl = f13.computeFullSubgroups = f13.synchronization2 = f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.hostQueryReset = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    f2.features.shaderInt64 = VK_TRUE;
    const char *ext[] = {"VK_KHR_shader_clock", "VK_KHR_pipeline_executable_properties", "VK_KHR_calibrated_timestamps"};
    float prio = 1.0f;
    // VKG_FAMILIES=g or c creates only that family (E4: does creating both change the compute queue's timing?).
    const char *fe = getenv("VKG_FAMILIES");
    const int want[2] = {!fe || strcmp(fe, "c"), !fe || strcmp(fe, "g")};
    VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO},
                                      {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    uint32_t nqci = 0;
    for (int i = 0; i < 2; i++) {
        if (!want[i]) {
            QU[i].valid = 0;
            continue;
        }
        qci[nqci].queueFamilyIndex = QU[i].fam;
        qci[nqci].queueCount = 1;
        qci[nqci].pQueuePriorities = &prio;
        nqci++;
    }
    printf("FAMILIES created:%s%s\n", want[0] ? " graphics" : "", want[1] ? " compute" : "");
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = nqci;
    dci.pQueueCreateInfos = qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = ext;
    VK_CHECK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    p_cal = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(g_dev, "vkGetCalibratedTimestampsKHR");
    for (int i = 0; i < 2; i++) {
        if (!QU[i].valid) continue;
        vkGetDeviceQueue(g_dev, QU[i].fam, 0, &QU[i].q);
        VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = QU[i].fam;
        VK_CHECK(vkCreateCommandPool(g_dev, &cpi, NULL, &QU[i].pool));
        pthread_mutex_init(&QU[i].mu, NULL);
    }
    VkDescriptorSetLayoutBinding b[4];
    for (int i = 0; i < 4; i++)
        b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                              VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 4;
    dli.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &g_dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g_dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &g_cpl));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 16;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &g_dpool));
}

// One (device tick, CLOCK_MONOTONIC ns) pair; returns the driver's maximum deviation in ns.
double cal_pair(uint64_t *dev_ticks, double *mono) {
    VkCalibratedTimestampInfoKHR ci[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR},
                                          {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}};
    ci[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
    ci[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
    uint64_t ts[2], dev = 0;
    VK_CHECK(p_cal(g_dev, 2, ci, ts, &dev));
    *dev_ticks = ts[0];
    *mono = (double)ts[1];
    return (double)dev;
}

// Maps device ticks to CLOCK_MONOTONIC ns with two calibration pairs (start and end of a window).
void map_begin(Map *m) { m->dev0 = cal_pair(&m->d0, &m->m0); }
void map_end(Map *m) { m->dev1 = cal_pair(&m->d1, &m->m1); }
double map_ns(const Map *m, uint64_t ticks) {
    double rate = (m->d1 > m->d0) ? (m->m1 - m->m0) / (double)(m->d1 - m->d0) : g_tsp;
    return m->m0 + ((double)(int64_t)(ticks - m->d0)) * rate;
}
void map_report(const char *tag, const Map *m) {
    double rate = (m->m1 - m->m0) / (double)(m->d1 - m->d0);
    printf("CALIB %s device_ticks_per_host_ns=%.7f (timestampPeriod %.3f) window_ms=%.1f max_deviation_ns=%.0f/%.0f\n",
           tag, 1.0 / rate, g_tsp, (m->m1 - m->m0) / 1e6, m->dev0, m->dev1);
}

// ---------------------------------------------------------------------------------------------------------------
// buffers, command buffers, descriptor sets


static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = g_mp.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

// kind 0: device-local; 1: host-visible cached system memory (readback); 2: device-local and host-visible (BAR)
Buf mkbuf(VkDeviceSize size, int kind) {
    Buf b = {0};
    b.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(g_dev, &bi, NULL, &b.b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b.b, &mr);
    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t t = UINT32_MAX;
    if (kind == 0) t = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (kind == 1) t = mem_type(mr.memoryTypeBits, hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (kind == 2) t = mem_type(mr.memoryTypeBits, hv | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (t == UINT32_MAX) t = mem_type(mr.memoryTypeBits, kind ? hv : 0, 0);
    if (t == UINT32_MAX) {
        printf("VERDICT vkgame memory: FAIL no memory type of kind %d\n", kind);
        exit(3);
    }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = t;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &b.m));
    VK_CHECK(vkBindBufferMemory(g_dev, b.b, b.m, 0));
    if (kind) VK_CHECK(vkMapMemory(g_dev, b.m, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

void rmbuf(Buf *b) {
    if (!b->b) return;
    if (b->map) vkUnmapMemory(g_dev, b->m);
    vkDestroyBuffer(g_dev, b->b, NULL);
    vkFreeMemory(g_dev, b->m, NULL);
    memset(b, 0, sizeof *b);
}

VkCommandBuffer cb_alloc(Qu *q) {
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = q->pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    return cb;
}

void cb_begin(VkCommandBuffer cb) {
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
}

VkFence mkfence(void) {
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    VK_CHECK(vkCreateFence(g_dev, &fci, NULL, &f));
    return f;
}

void submit(Qu *q, VkCommandBuffer cb, VkFence f) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    pthread_mutex_lock(&q->mu);
    VK_CHECK(vkQueueSubmit(q->q, 1, &si, f));
    pthread_mutex_unlock(&q->mu);
}

void fwait(VkFence f) {
    VK_CHECK(vkWaitForFences(g_dev, 1, &f, VK_TRUE, 10ull * 1000000000ull));
    VK_CHECK(vkResetFences(g_dev, 1, &f));
}

void run_once(Qu *q, VkCommandBuffer cb) {
    VkFence f = mkfence();
    submit(q, cb, f);
    fwait(f);
    vkDestroyFence(g_dev, f, NULL);
}

// The queue for copies and fills: the compute-only family when this process created it.
Qu *xq(void) { return QU[1].valid ? &QU[1] : &QU[0]; }

void download(Buf *src, void *dst, VkDeviceSize size) {
    Buf st = mkbuf(size, 1);
    VkCommandBuffer cb = cb_alloc(xq());
    cb_begin(cb);
    VkBufferCopy c = {0, 0, size};
    vkCmdCopyBuffer(cb, src->b, st.b, 1, &c);
    VK_CHECK(vkEndCommandBuffer(cb));
    run_once(xq(), cb);
    memcpy(dst, st.map, size);
    vkFreeCommandBuffers(g_dev, xq()->pool, 1, &cb);
    rmbuf(&st);
}

void fill_zero(Buf *b) {
    VkCommandBuffer cb = cb_alloc(xq());
    cb_begin(cb);
    vkCmdFillBuffer(cb, b->b, 0, VK_WHOLE_SIZE, 0);
    VK_CHECK(vkEndCommandBuffer(cb));
    run_once(xq(), cb);
    vkFreeCommandBuffers(g_dev, xq()->pool, 1, &cb);
}

VkDescriptorSet mkset(Buf *b0, Buf *b1, Buf *b2, Buf *b3) {
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g_dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &dai, &ds));
    Buf *bs[4] = {b0, b1, b2, b3};
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet w[4];
    int n = 0;
    for (int i = 0; i < 4; i++) {
        if (!bs[i]) continue;
        bi[n] = (VkDescriptorBufferInfo){bs[i]->b, 0, VK_WHOLE_SIZE};
        w[n] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[n].dstSet = ds;
        w[n].dstBinding = (uint32_t)i;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    vkUpdateDescriptorSets(g_dev, (uint32_t)n, w, 0, NULL);
    return ds;
}

VkQueryPool mkqp(uint32_t n) {
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = n;
    VkQueryPool qp;
    VK_CHECK(vkCreateQueryPool(g_dev, &qpi, NULL, &qp));
    vkResetQueryPool(g_dev, qp, 0, n);
    return qp;
}

uint64_t qget(VkQueryPool qp, uint32_t i) {
    uint64_t v = 0;
    VK_CHECK(vkGetQueryPoolResults(g_dev, qp, i, 1, 8, &v, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return v;
}

void ts_top(VkCommandBuffer cb, VkQueryPool qp, uint32_t i) {
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, qp, i);
}
void ts_bot(VkCommandBuffer cb, VkQueryPool qp, uint32_t i) {
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, qp, i);
}

// ---------------------------------------------------------------------------------------------------------------
// pipelines and their machine code


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
    fclose(f);
    *len = n;
    return d;
}

VkShaderModule mkmod(const char *spv) {
    char path[800];
    snprintf(path, sizeof path, "%s/%s", g_spv, spv);
    size_t len = 0;
    char *code = read_file(path, &len);
    if (!code) {
        printf("VERDICT vkgame spv: FAIL cannot read %s\n", path);
        exit(1);
    }
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = len;
    smi.pCode = (const uint32_t *)code;
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(g_dev, &smi, NULL, &sm));
    free(code);
    return sm;
}

// Writes every executable's statistics and ACO assembly to <out>/asm/<name>.s and prints a PIPE line per executable
// (stage, subgroup size, a few statistics, instruction counts of interest).
void capture(Pipe *p) {
    PFN_vkGetPipelineExecutablePropertiesKHR gprops =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            g_dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gstats =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableStatisticsKHR");
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p->p;
    uint32_t ne = 0;
    gprops(g_dev, &pi, &ne, NULL);
    VkPipelineExecutablePropertiesKHR ep[4];
    for (int i = 0; i < 4; i++)
        ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (ne > 4) ne = 4;
    gprops(g_dev, &pi, &ne, ep);
    FILE *f = NULL;
    if (g_out[0]) {
        char path[800];
        snprintf(path, sizeof path, "%s/asm", g_out);
        mkdir(path, 0755);
        snprintf(path, sizeof path, "%s/asm/%s.s", g_out, p->name);
        f = fopen(path, "w");
    }
    for (uint32_t e = 0; e < ne; e++) {
        if (e == 0) p->sg = ep[e].subgroupSize;
        VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = p->p;
        ei.executableIndex = e;
        char stats[512] = "";
        if (f) fprintf(f, "# pipeline %s, executable %u \"%s\" (%s), subgroup size %u\n", p->name, e, ep[e].name,
                       ep[e].description, ep[e].subgroupSize);
        uint32_t ns = 0;
        gstats(g_dev, &ei, &ns, NULL);
        VkPipelineExecutableStatisticKHR *st = calloc(ns ? ns : 1, sizeof *st);
        for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        gstats(g_dev, &ei, &ns, st);
        for (uint32_t i = 0; i < ns; i++) {
            long long v = st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR ? st[i].value.i64
                          : st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
                              ? (long long)st[i].value.u64
                              : (long long)st[i].value.b32;
            if (f) fprintf(f, "# stat %s = %lld\n", st[i].name, v);
            if (!strcmp(st[i].name, "VGPRs") || !strcmp(st[i].name, "SGPRs") || !strcmp(st[i].name, "Instructions") ||
                !strcmp(st[i].name, "Code size") || !strcmp(st[i].name, "Subgroups per SIMD") ||
                !strcmp(st[i].name, "LDS size") || !strcmp(st[i].name, "Scratch size")) {
                char nm[64];
                snprintf(nm, sizeof nm, "%s", st[i].name);
                for (char *c = nm; *c; c++)
                    if (*c == ' ') *c = '_';
                size_t l = strlen(stats);
                snprintf(stats + l, sizeof stats - l, " %s=%lld", nm, v);
            }
        }
        free(st);
        uint32_t nr = 0;
        greps(g_dev, &ei, &nr, NULL);
        VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr ? nr : 1, sizeof *ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        greps(g_dev, &ei, &nr, ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].pData = calloc(1, ir[i].dataSize + 1);
        greps(g_dev, &ei, &nr, ir);
        int nreal = 0, nclk = 0, nsleep = 0, nwait = 0;
        for (uint32_t i = 0; i < nr; i++) {
            if (strstr(ir[i].name, "ssembly")) {
                const char *t = ir[i].pData;
                if (f) fprintf(f, "# representation: %s\n%s\n", ir[i].name, t);
                for (const char *s = t; (s = strstr(s, "MSG_RTN_GET_REALTIME")); s++) nreal++;
                for (const char *s = t; (s = strstr(s, "SHADER_CYCLES")); s++) nclk++;
                for (const char *s = t; (s = strstr(s, "s_sleep")); s++) nsleep++;
                for (const char *s = t; (s = strstr(s, "s_wait_kmcnt")); s++) nwait++;
            }
            free(ir[i].pData);
        }
        free(ir);
        printf("PIPE %s exe=%u stage=%s subgroup=%u%s realtime_reads=%d cycle_reads=%d s_sleep=%d s_wait_kmcnt=%d\n",
               p->name, e, ep[e].name, ep[e].subgroupSize, stats, nreal, nclk, nsleep, nwait);
    }
    if (f) fclose(f);
}

Pipe mkcomp(const char *spv, const char *label, uint32_t local, uint32_t req_sg) {
    Pipe p = {0};
    snprintf(p.name, sizeof p.name, "%s", label);
    VkShaderModule sm = mkmod(spv);
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo spi = {1, &me, 4, &local};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = req_sg;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.pNext = req_sg ? &rss : NULL;
    ci.stage.flags = req_sg ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = sm;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = &spi;
    ci.layout = g_cpl;
    VK_CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &p.p));
    vkDestroyShaderModule(g_dev, sm, NULL);
    capture(&p);
    return p;
}

// ---------------------------------------------------------------------------------------------------------------
// option parsing: "key=value,key=value"

const char *kv(const char *spec, const char *key, char *buf, size_t cap) {
    size_t kl = strlen(key);
    const char *s = spec;
    while (s && *s) {
        if (!strncmp(s, key, kl) && s[kl] == '=') {
            const char *v = s + kl + 1;
            size_t n = strcspn(v, ",");
            if (n >= cap) n = cap - 1;
            memcpy(buf, v, n);
            buf[n] = 0;
            return buf;
        }
        s = strchr(s, ',');
        if (s) s++;
    }
    return NULL;
}
double kvd(const char *spec, const char *key, double def) {
    char b[64];
    return kv(spec, key, b, sizeof b) ? atof(b) : def;
}
const char *kvs(const char *spec, const char *key, const char *def, char *buf, size_t cap) {
    if (kv(spec, key, buf, cap)) return buf;
    snprintf(buf, cap, "%s", def);
    return buf;
}

void out_path(char *p, size_t cap, const char *name) { snprintf(p, cap, "%s/%s-%s", g_out, g_tag, name); }


// ---------------------------------------------------------------------------------------------------------------
// info: machine code of every pipeline, and whether REALTIME and the queue timestamps share one clock

static int mode_info(void) {
    Pipe pc64 = mkcomp("gchain.spv", "gchain_l64", 64, 0);
    Pipe pc32 = mkcomp("gchain.spv", "gchain_l64_sg32", 64, 32);
    Pipe pc256 = mkcomp("gchain.spv", "gchain_l256", 256, 0);
    Pipe pt = mkcomp("gtiny.spv", "gtiny", 64, 0);
    Pipe pb = mkcomp("gburst.spv", "gburst_l64", 64, 0);  // the load of the first P4 runs, kept for its listing
    Pipe pd32 = mkcomp("gdense.spv", "gdense_l32_w32", 32, 32), pd1k = mkcomp("gdense.spv", "gdense_l1024_w32", 1024, 32);
    draw_setup(1);
    (void)pc64, (void)pc32, (void)pc256, (void)pb, (void)pd32, (void)pd1k;
    // REALTIME against the queue timestamps: one-group dispatches on each queue; TOP timestamp, REALTIME at the
    // shader's entry and exit, BOTTOM timestamp. One clock if every REALTIME lies between its two timestamps.
    Buf res = mkbuf(64 * 64 * 4, 2), rec = mkbuf(64 * 16, 2);
    VkDescriptorSet ds = mkset(&res, &rec, NULL, NULL);
    VkQueryPool qp = mkqp(2);
    int bad = 0, tot = 0;
    for (int qi = 0; qi < 2; qi++) {
        VkCommandBuffer cb = cb_alloc(&QU[qi]);
        double a[100], b[100], w[100];
        for (int i = 0; i < 100; i++) {
            cb_begin(cb);
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pt.p);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
            uint32_t pcv[2] = {7u, 0u};
            vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pcv);
            ts_top(cb, qp, 0);
            vkCmdDispatch(cb, 1, 1, 1);
            ts_bot(cb, qp, 1);
            VK_CHECK(vkEndCommandBuffer(cb));
            run_once(&QU[qi], cb);
            uint64_t t0 = qget(qp, 0), t1 = qget(qp, 1);
            volatile uint32_t *r = rec.map;
            uint64_t r0 = r[0] | (uint64_t)r[1] << 32, r1 = r[2] | (uint64_t)r[3] << 32;
            a[i] = ((double)(int64_t)(r0 - t0)) * g_tsp / 1e3;
            b[i] = ((double)(int64_t)(t1 - r1)) * g_tsp / 1e3;
            w[i] = ((double)(int64_t)(t1 - t0)) * g_tsp / 1e3;
            tot++;
            bad += !(a[i] >= 0 && a[i] < 100 && b[i] >= 0 && b[i] < 100);
        }
        char tag[64];
        snprintf(tag, sizeof tag, "tsdomain-%s", QU[qi].name);
        dist(tag, "realtime_entry_minus_top_ts_us", a, 100);
        dist(tag, "bottom_ts_minus_realtime_exit_us", b, 100);
        dist(tag, "bottom_minus_top_us", w, 100);
    }
    verdict("realtime_domain", bad <= tot / 20,
            "shader REALTIME lies between the dispatch's TOP and BOTTOM timestamps (0..100 us) in %d of %d dispatches "
            "(one clock if at least 95 %%)",
            tot - bad, tot);
    // REALTIME rate against host time over 200 ms, and the device timestamp rate from calibrated pairs.
    Map m;
    map_begin(&m);
    uint64_t rr[2];
    double hh[2];
    for (int k = 0; k < 2; k++) {
        if (k) until_ns(mono_ns() + 200e6);
        VkCommandBuffer cb = cb_alloc(&QU[1]);
        cb_begin(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pt.p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
        uint32_t pcv[2] = {7u, 0u};
        vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pcv);
        vkCmdDispatch(cb, 1, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cb));
        double h0 = mono_ns();
        run_once(&QU[1], cb);
        double h1 = mono_ns();
        volatile uint32_t *r = rec.map;
        rr[k] = r[0] | (uint64_t)r[1] << 32;
        hh[k] = 0.5 * (h0 + h1);
    }
    map_end(&m);
    map_report("info", &m);
    double mhz = (double)(rr[1] - rr[0]) / ((hh[1] - hh[0]) / 1e3);
    verdict("realtime_rate", fabs(mhz - 99.97) < 0.1,
            "REALTIME runs at %.3f MHz against host time over %.0f ms (study: 99.970 MHz; limit +-0.1)", mhz,
            (hh[1] - hh[0]) / 1e6);
    double rate = (double)(m.d1 - m.d0) / ((m.m1 - m.m0) / 1e3);
    verdict("timestamp_rate", fabs(rate - 99.97) < 0.1,
            "calibrated device timestamps run at %.3f MHz against CLOCK_MONOTONIC (limit 99.97 +-0.1)", rate);
    return 0;
}

// Machine code of every pipeline, no dispatch.
static int mode_listing(void) {
    mkcomp("gchain.spv", "gchain_l64", 64, 0);
    mkcomp("gtiny.spv", "gtiny", 64, 0);
    mkcomp("gburst.spv", "gburst_l64", 64, 0);
    mkcomp("gdense.spv", "gdense_l32_w32", 32, 32);
    mkcomp("gdense.spv", "gdense_l1024_w32", 1024, 32);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vkgame info | listing | chain <spec>... | lone [--swap-labels] | share <spec>... | "
                        "burst <spec>... | rt <spec> | slow <spec>\n");
        return 2;
    }
    const char *o = getenv("VKG_OUT"), *s = getenv("VKG_SPV"), *t = getenv("VKG_TAG");
    snprintf(g_out, sizeof g_out, "%s", o ? o : "");
    snprintf(g_spv, sizeof g_spv, "%s", s ? s : "build/spv");
    snprintf(g_tag, sizeof g_tag, "%s", t ? t : argv[1]);
    dev_init(!strcmp(argv[1], "info"));
    sampler_start();
    double t0 = mono_ns();
    printf("START hotspot %.0f C\n", hotspot());
    int rc = 2;
    if (!strcmp(argv[1], "info")) rc = mode_info();
    else if (!strcmp(argv[1], "chain")) rc = mode_chain(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "lone")) rc = mode_lone(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "share")) rc = mode_share(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "burst")) rc = mode_burst(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "listing")) rc = mode_listing();
    else if (!strcmp(argv[1], "rt")) rc = mode_rt(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "slow")) rc = mode_slow(argc - 2, argv + 2);
    else fprintf(stderr, "unknown mode %s\n", argv[1]);
    vkDeviceWaitIdle(g_dev);
    double t1 = mono_ns();
    sampler_stop_write();
    state_line("whole-job", t0, t1);
    // The thermal rule asks for a hotspot reading every 5 ms during the load; a busy host stretches the sampler.
    if (g_nsmp > 2) {
        double *iv = malloc(sizeof(double) * g_nsmp);
        for (int i = 1; i < g_nsmp; i++) iv[i - 1] = (g_smp[i].t - g_smp[i - 1].t) / 1e6;
        dist("sampler", "interval_ms", iv, g_nsmp - 1);
        const double p95 = pctl(iv, g_nsmp - 1, .95), mx = pctl(iv, g_nsmp - 1, 1);
        verdict("sampler_interval", p95 <= 6.0 && mx <= 20.0,
                "5 ms sysfs sampler: interval p95 %.2f ms (limit 6), largest %.2f ms (limit 20)", p95, mx);
        free(iv);
    }
    printf("THERMAL highest %.0f C (5 ms samples); end %.0f C; hot_stop=%d\n", g_hottest, hotspot(), g_hot_stop);
    if (g_hot_stop && (g_hot_rel > 0 || g_hot_stop_c < HOT_STOP_C) && g_hottest < HOT_STOP_C)
        printf("INFO thermal: the load was stopped at a lowered limit (VKG_HOT_STOP_C, the stop path's test); highest "
               "%.0f C\n", g_hottest);
    else if (g_hot_stop) verdict("thermal", 0, "hotspot reached 88 C; load stopped");
    if (rc) return rc;
    return g_fail ? 1 : 0;
}
