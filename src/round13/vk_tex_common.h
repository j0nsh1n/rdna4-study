// Shared Vulkan 1.3 compute + sampled-image harness for Round 13 Probes B (vk_tex) and C (vk_tex_fp16).
// Selects RX 9070 XT (1002:7550) on RADV only, captures ACO disassembly & statistics via
// VK_KHR_pipeline_executable_properties, reads per-run SCLK/UCLK/hotspot from amdgpu gpu_metrics
// (v1_3) + hwmon temp2_input with a 90 C thermal stop, and supports --dump-asm without GPU execution.
#ifndef VK_TEX_COMMON_H
#define VK_TEX_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define VKT_VENDOR 0x1002u
#define VKT_DEVICE 0x7550u
#define VKT_PUSH_BYTES 64u
#define VKT_THERMAL_STOP_C 90.0

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void * map;
} VktBuf;

typedef struct {
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    VkSampler sampler;
    VkFormat format;
    uint32_t w;
    uint32_t h;
} VktImg;

typedef struct {
    VkPipeline pipe;
    char name[96];
    uint32_t local_size;
    uint32_t sg_size;
    int vgprs;
    int sg_per_simd;
    char * asm_text;
} VktPipe;

typedef struct {
    int total_insts;
    int image_load;
    int image_sample;
    int image_gather;
    int pk_fma_f16;
    int fma_f32;
    int dual_fmac_f32;
    int int_alu;
    int v_msad;
    int buf_load_b128;
    int buf_store_b128;
} VktLoopStats;

typedef struct {
    double sclk_avg_mhz;
    double sclk_med_mhz;
    double sclk_min_mhz;
    double sclk_max_mhz;
    double uclk_med_mhz;
    double hot_max_c;
    int n_samples;
} VktMetrics;

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
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkDescriptorPool dp;
    VkDescriptorSet ds;
    VkPhysicalDeviceMemoryProperties memprops;
    double ts_period_ns;
    char drm_dev[512];
    char metrics_path[600];
    char hwmon_hot_path[1100];
    char hwmon_fan_path[1100];
    char out_dir[512];
    char spv_dir[512];
    VktBuf dummy_buf;
    int has_exec_props;
    int has_shader_clock;
    VkQueue gfx_q;
    uint32_t gfx_qf;
    int has_separate_compute;
    int same_queue_handle;
    int has_global_prio;
    int used_global_prio;
    VkQueueGlobalPriority comp_global_prio;
    int fail_closed_thermal;
    // Hotspot stop for this program; 0 means unset and falls back to VKT_THERMAL_STOP_C.
    double thermal_stop_c;
} VktCtx;

extern VktCtx VT;

void vkt_init(int quiet);
VkResult vkt_init_async(int quiet, int use_global_prio, VkQueueGlobalPriority comp_prio, int dump_qfamilies);
void vkt_shutdown(void);
int vkt_format_supported(VkFormat fmt);
int vkt_check_hotspot_fail_closed(double * out_hot_c);
int vkt_read_hotspot_status(double * out_hot_c);
int vkt_read_fan_rpm(void);
void vkt_capture_pipe_aco(VkPipeline pipe, const char * label, VktPipe * out);

VktBuf vkt_buf_create(VkDeviceSize size, int host_visible);
void vkt_buf_destroy(VktBuf * b);

VktImg vkt_img_create(VkFormat fmt, uint32_t w, uint32_t h, VkFilter filter);
void vkt_img_destroy(VktImg * im);
void vkt_img_fill_ref(VktImg * im, int ref_fmt, int exact_mode);

void vkt_bind(const VktImg * im, const VktBuf * out_buf, const VktBuf * in_buf);

VktPipe vkt_pipe_create(const char * spv, const char * label, uint32_t local_size, uint32_t req_sg);
void vkt_pipe_destroy(VktPipe * p);
VktLoopStats vkt_pipe_analyze_loop(const VktPipe * p, const char * primary_prefix);
int vkt_pipe_has_shader_clock(const VktPipe * p);

double vkt_dispatch(const VktPipe * p, uint32_t groups_x, const void * push, size_t push_bytes, double * wall_ns);

void vkt_metrics_start(void);
VktMetrics vkt_metrics_stop(void);
int vkt_thermal_guard_and_cool(double target_c, int max_wait_ms);
double vkt_hotspot_c(void);
int vkt_read_gpu_metrics(int * gfxclk_mhz, int * uclk_mhz, double * hot_c);

double vkt_now_ns(void);
double vkt_median(double * v, int n);
double vkt_vmin(const double * v, int n);
double vkt_vmax(const double * v, int n);

#endif
