// Small Vulkan compute harness for the lane 9 probes: device selection (RX 9070 XT only), buffers, pipelines
// with captured ACO machine code, timed dispatches, and a 50 ms clock/power/hotspot sampler.
#ifndef VKH_H
#define VKH_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define VKH_VENDOR 0x1002u
#define VKH_DEVICE 0x7550u
#define VKH_BINDINGS 4
#define VKH_PUSH_BYTES 64

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult r_ = (x);                                                                                             \
        if (r_ != VK_SUCCESS) {                                                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n", (int)r_, __FILE__, __LINE__, #x);                        \
            exit(3);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void *map; // only for host-visible buffers
} Buf;

typedef struct {
    VkPipeline pipe;
    char name[96];
    uint32_t local_size;
    uint32_t sg_size;  // from the pipeline executable statistics / properties
    char *asm_text;    // ACO disassembly, owned
} Pipe;

typedef struct {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue q;
    uint32_t qf;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;
    VkQueryPool qp;
    uint32_t qp_count;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkDescriptorPool dp;
    VkDescriptorSet ds;
    double ts_period_ns;
    VkPhysicalDeviceMemoryProperties memprops;
    char drm_dev[512]; // /sys/class/drm/cardN/device of this card
    char hwmon[1024];
    char out_dir[512]; // run folder for asm listings and bit dumps ("" = do not write)
    char spv_dir[512];
    Buf dummy;
} Ctx;

extern Ctx G;

// Selects the RX 9070 XT by vendor/device id and its sysfs node by PCI address; exits on anything else.
void vkh_init(int quiet);
void vkh_print_device_report(void);

Buf buf_create(VkDeviceSize size, int host_visible);
void buf_destroy(Buf *b);
void buf_upload(Buf *dst, const void *src, VkDeviceSize size);
void buf_download(const Buf *src, void *dst, VkDeviceSize size);

// spv: file name under spv_dir. req_sg: 0 = driver choice, else VkPipelineShaderStageRequiredSubgroupSize.
Pipe pipe_create(const char *spv, const char *label, uint32_t local_size, uint32_t req_sg, int full_subgroups);
void pipe_destroy(Pipe *p);
// Counts the instructions of the innermost loop(s) of the captured disassembly; prints LOOP lines.
// Returns the number of times `must` appears in the hottest loop (the loop containing the most `must` lines).
int pipe_loop_report(const Pipe *p, const char *must, int print_body);
// Sum of the counts of mnemonics starting with prefix in the loop of the last pipe_loop_report call.
int loop_count(const char *prefix);

void bind_buffers(Buf *b0, Buf *b1, Buf *b2, Buf *b3);

// Records: optional fills (buf, value) before the dispatch, then ts, dispatch, ts. Returns GPU ns of the dispatch.
typedef struct {
    Buf *buf;
    uint32_t value;
} Fill;
double run_dispatch(const Pipe *p, uint32_t groups, const void *push, size_t push_bytes, const Fill *fills, int nfill,
                    double *wall_ns);
// Same, with a warm-up dispatch (push constants warm_push) in the same command buffer before the fills and the
// timed dispatch.
double run_dispatch_warm(const Pipe *p, uint32_t groups, const void *warm_push, const void *push, size_t push_bytes,
                         const Fill *fills, int nfill, double *wall_ns);

// Host-side clock/power/hotspot sampler (50 ms).
typedef struct {
    double sclk_min, sclk_med, sclk_max, pw_min, pw_max, t_min, t_max;
    int n;
} SampleStats;
void sampler_start(void);
SampleStats sampler_stop(void);
double hotspot_c(void);
double highest_hotspot(void);
void note_hotspot(double t);
// Waits below 60 C (up to 120 s) before a whole-card job; exits 4 at 95 C.
void thermal_gate(void);
void print_clocks(const char *when);

double now_ns(void);
double median(double *v, int n);
double vmin(const double *v, int n);
double vmax(const double *v, int n);
uint64_t rng_next(uint64_t *s);

#endif
