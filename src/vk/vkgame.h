// Shared declarations of the gaming-round Vulkan probe (vkgame.c and its mode files).
#ifndef VKGAME_H
#define VKGAME_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define VENDOR 0x1002u
#define DEVICE 0x7550u
#define MAXSMP 60000
#define HOT_STOP_LIMIT_C 88.0  // the thermal rule's in-load stop; VKG_HOT_STOP_C may only lower it

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult r_ = (x);                                                                                             \
        if (r_ != VK_SUCCESS) {                                                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);                        \
            printf("VERDICT vkgame vulkan_call: FAIL error %d at %s:%d\n", (int)r_, __FILE__, __LINE__);               \
            fflush(stdout);                                                                                            \
            exit(3);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct {
    double t;  // CLOCK_MONOTONIC ns
    int sclk, busy;
    double hot, pw;
    int act, sock_w, avg_gfx, cur_gfx, vgfx;
    uint32_t thr;
    uint64_t ithr, fwts, sysns;
    int metrics_ok, prof;
} Smp;

typedef struct {
    const char *name;
    uint32_t fam, valid;
    VkQueue q;
    VkCommandPool pool;
    pthread_mutex_t mu;
} Qu;

// Device ticks to CLOCK_MONOTONIC ns, from calibrated pairs at the start and the end of a window.
typedef struct {
    uint64_t d0, d1;
    double m0, m1, dev0, dev1;
} Map;

typedef struct {
    VkBuffer b;
    VkDeviceMemory m;
    VkDeviceSize size;
    void *map;
} Buf;

typedef struct {
    VkPipeline p;
    char name[96];
    uint32_t sg;
} Pipe;

extern int g_fail;
extern char g_drm[512], g_hw[1024], g_out[512], g_spv[512], g_tag[128];
extern Smp *g_smp;
extern volatile int g_nsmp, g_srun, g_hot_stop;
extern volatile uint32_t *volatile g_stop_word;  // the dense load's stop word while an arm runs (88 C tripwire)
extern double g_hottest;
extern volatile double g_hot_stop_c;  // the sampler's stop limit (88 C unless lowered for F5)
extern double g_hot_rel;               // F5: > 0 sets the limit per dense arm to its start hotspot + g_hot_rel
extern VkDevice g_dev;
extern double g_tsp;
extern Qu QU[2];  // 0: graphics family, 1: compute-only family
extern VkDescriptorSetLayout g_dsl;
extern VkPipelineLayout g_cpl;
extern VkDescriptorPool g_dpool;

double mono_ns(void);
void until_ns(double t);
double pctl(const double *v, int n, double p);
int nfinite(const double *v, int n);
void dist(const char *tag, const char *what, const double *v, int n);
double frac_above(const double *v, int n, double lim);
void verdict(const char *name, int ok, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
double hotspot(void);
void state_line(const char *tag, double t0, double t1);
const Smp *smp_at(double t);

double cal_pair(uint64_t *dev_ticks, double *mono);
void map_begin(Map *m);
void map_end(Map *m);
double map_ns(const Map *m, uint64_t ticks);
void map_report(const char *tag, const Map *m);

Buf mkbuf(VkDeviceSize size, int kind);  // 0 device-local, 1 host-cached system memory, 2 device-local host-visible
void rmbuf(Buf *b);
VkCommandBuffer cb_alloc(Qu *q);
void cb_begin(VkCommandBuffer cb);
VkFence mkfence(void);
void submit(Qu *q, VkCommandBuffer cb, VkFence f);
void fwait(VkFence f);
void run_once(Qu *q, VkCommandBuffer cb);
Qu *xq(void);
void download(Buf *src, void *dst, VkDeviceSize size);
void fill_zero(Buf *b);
VkDescriptorSet mkset(Buf *b0, Buf *b1, Buf *b2, Buf *b3);
VkQueryPool mkqp(uint32_t n);
uint64_t qget(VkQueryPool qp, uint32_t i);
void ts_top(VkCommandBuffer cb, VkQueryPool qp, uint32_t i);
void ts_bot(VkCommandBuffer cb, VkQueryPool qp, uint32_t i);
VkShaderModule mkmod(const char *spv);
void capture(Pipe *p);
Pipe mkcomp(const char *spv, const char *label, uint32_t local, uint32_t req_sg);

const char *kv(const char *spec, const char *key, char *buf, size_t cap);
double kvd(const char *spec, const char *key, double def);
const char *kvs(const char *spec, const char *key, const char *def, char *buf, size_t cap);
void out_path(char *p, size_t cap, const char *name);

// modes
int mode_chain(int argc, char **argv);
int mode_lone(int argc, char **argv);
int mode_share(int argc, char **argv);
int mode_burst(int argc, char **argv);
void stop_all(volatile uint32_t *w);
int mode_rt(int argc, char **argv);
int mode_slow(int argc, char **argv);
void draw_setup(int report);

#endif
