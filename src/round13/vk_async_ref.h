// CPU reference and pure-C helpers for Round 13 Probe P4 (vk_async):
//   - Job (b) emulated 4-byte integer SAD exact reference (GLSL has no msad4 / v_msad_u8)
//   - Job (d) 128 MiB DRAM streaming copy+XOR exact reference
//   - Fail-closed 95 C hotspot file reader (unreadable / invalid = STOP)
//   - Bounded < 60 C cooldown wait with optional soft settle & double-stop thermal policy
//   - ABBA schedule builder between priority arms
//   - Graphics frame completion interval overlap statistics (single-burst and multi-repeat)
// No Vulkan dependency: included by vk_async.c and vk_async_selftest.c.
#ifndef VK_ASYNC_REF_H
#define VK_ASYNC_REF_H

#include "vk_tex_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VKA_THERMAL_STOP_C     95.0  /* study limit (README); owner 2026-11-27, was 90 */
#define VKA_THERMAL_MAX_STOPS  2
#define VKA_COOLDOWN_TARGET_C  60.0
#define VKA_COOLDOWN_SETTLE_C  50.0
#define VKA_COOLDOWN_MAX_MS    15000
#define VKA_COOLDOWN_SETTLE_MS 2000
#define VKA_COOLDOWN_POLL_MS   200

// Chain lengths: preregistered N=1 monolithic vs N=80 chained in BOTH SAT-100 and PACE-60.
// 80 is a multiple of 4 so Job (d) sweeps all 4 x 32 MiB quarters evenly (5120 MiB = 5.0 GiB).
#define VKA_CHAIN_SAT           80u
#define VKA_CHAIN_PACE          80u

// Stream geometry: 128 MiB = 8,388,608 uvec4 elements, split into 4 x 32 MiB quarters.
// Total read+write traffic: N=1 (1 * 20 * 4 * 32 MiB * 2) == N=80 (80 * 1 * 1 * 32 MiB * 2) == 5120 MiB (5.0 GiB).
#define VKA_STREAM_TOTAL_VEC4   (8u * 1024u * 1024u)
#define VKA_STREAM_QUARTER_VEC4 (2u * 1024u * 1024u)
#define VKA_STREAM_GROUPS       2048u
#define VKA_STREAM_LOCAL        256u
#define VKA_STREAM_N1_ITERS     20u
#define VKA_STREAM_N80_ITERS    1u
#define VKA_STREAM_BASE_SALT    0xa5a50000u
#define VKA_STREAM_TOTAL_RW_BYTES (5120ull * 1024ull * 1024ull)

// Job (b) SAD workgroup/iteration sizing: N=1 (1 * 560) == N=80 (80 * 7) = 560 outer iters.
#define VKA_SAD_GROUPS          1024u
#define VKA_SAD_N1_ITERS        560u
#define VKA_SAD_N80_ITERS       7u

// Job (a) FP16 workgroup/iteration sizing: N=1 (1 * 880) == N=80 (80 * 11) = 880 outer iters.
// vk_tex_fp16.comp adds a post-loop parity step of 5 FMAs (one per chain) on even iters so N=1 stays non-trivial.
#define VKA_FP16_GROUPS         1024u
#define VKA_FP16_N1_ITERS       880u
#define VKA_FP16_N80_ITERS      11u

// Job (c) Texture (RGBA16F MALL 32 MiB = 2048x2048) sizing: N=1 (1 * 80) == N=80 (80 * 1) = 80 outer iters,
// the nearest equal pair to the registered monolithic size of 64 (review round 2).
#define VKA_TEX_GROUPS          2048u
#define VKA_TEX_N1_ITERS        80u
#define VKA_TEX_N80_ITERS       1u
// Registered pre-fix monolithic sizes; a resize may not exceed them by more than 25 %.
#define VKA_FP16_N1_REGISTERED  801u
#define VKA_SAD_N1_REGISTERED   550u
#define VKA_TEX_N1_REGISTERED   64u

enum {
    VKA_JOB_FP16   = 0,
    VKA_JOB_SAD    = 1,
    VKA_JOB_TEX    = 2,
    VKA_JOB_STREAM = 3,
    VKA_JOB_COUNT  = 4
};

static inline uint32_t vka_job_iters(int job, uint32_t chain_n) {
    if (job == VKA_JOB_FP16)   return (chain_n == 1u) ? VKA_FP16_N1_ITERS : VKA_FP16_N80_ITERS;
    if (job == VKA_JOB_SAD)    return (chain_n == 1u) ? VKA_SAD_N1_ITERS : VKA_SAD_N80_ITERS;
    if (job == VKA_JOB_TEX)    return (chain_n == 1u) ? VKA_TEX_N1_ITERS : VKA_TEX_N80_ITERS;
    return (chain_n == 1u) ? VKA_STREAM_N1_ITERS : VKA_STREAM_N80_ITERS;
}

// Total work across an entire `chain_n`-dispatch chain (must be equal for N=1 and N=80 on every job).
static inline uint64_t vka_job_total_work(int job, uint32_t chain_n) {
    const uint64_t iters = (uint64_t)vka_job_iters(job, chain_n);
    if (job == VKA_JOB_FP16) {
        // vk_tex_fp16.comp adds one 5-FMA parity step per chain when iters is even.
        const uint64_t per_thread = iters * 256ull + (((iters & 1ull) == 0ull) ? 5ull : 0ull);
        return (uint64_t)chain_n * (uint64_t)VKA_FP16_GROUPS * 256ull * per_thread;
    }
    if (job == VKA_JOB_SAD) {
        return (uint64_t)chain_n * (uint64_t)VKA_SAD_GROUPS * 256ull * iters * 16ull;
    }
    if (job == VKA_JOB_TEX) {
        return (uint64_t)chain_n * (uint64_t)VKA_TEX_GROUPS * 256ull * iters * 16ull;
    }
    const uint64_t chunks = (chain_n <= 1u) ? 4ull : 1ull;
    return (uint64_t)chain_n * iters * chunks * (uint64_t)VKA_STREAM_QUARTER_VEC4 * 16ull * 2ull;
}

static inline const char * vka_job_work_unit(int job) {
    if (job == VKA_JOB_FP16) return "pk_fma_f16";
    if (job == VKA_JOB_SAD)  return "sad4_ops";
    if (job == VKA_JOB_TEX)  return "tex_samples";
    return "rw_bytes";
}

// ---- Job (b): Exact 4-byte integer SAD reference matching vk_async_sad.comp ----
static inline uint32_t vka_sad4_u32(uint32_t x, uint32_t y) {
    uint32_t x0 = x & 0xffu, y0 = y & 0xffu;
    uint32_t x1 = (x >> 8u) & 0xffu, y1 = (y >> 8u) & 0xffu;
    uint32_t x2 = (x >> 16u) & 0xffu, y2 = (y >> 16u) & 0xffu;
    uint32_t x3 = (x >> 24u) & 0xffu, y3 = (y >> 24u) & 0xffu;
    uint32_t d0 = (x0 > y0) ? (x0 - y0) : (y0 - x0);
    uint32_t d1 = (x1 > y1) ? (x1 - y1) : (y1 - x1);
    uint32_t d2 = (x2 > y2) ? (x2 - y2) : (y2 - x2);
    uint32_t d3 = (x3 > y3) ? (x3 - y3) : (y3 - x3);
    return d0 + d1 + d2 + d3;
}

static inline void vka_sad_lane_input(uint32_t lid, uint32_t out_kv[4]) {
    out_kv[0] = vkt_hash32(lid * 4u + 0x101u);
    out_kv[1] = vkt_hash32(lid * 4u + 0x202u);
    out_kv[2] = (vkt_hash32(lid * 4u + 0x303u) | 1u);
    out_kv[3] = (vkt_hash32(lid * 4u + 0x404u) | 1u);
}

static inline uint32_t vka_expected_sad_pass(uint32_t wgid, uint32_t lid, uint32_t iters,
                                             uint32_t pass_idx, uint32_t break_mask) {
    uint32_t kv[4];
    vka_sad_lane_input(lid, kv);
    const uint32_t salt = (wgid * 0x9e3779b9u) ^ (pass_idx * 0x7f4a7c15u) ^ break_mask;
    uint32_t x0 = kv[0] ^ salt;
    uint32_t y0 = kv[1] ^ (salt << 1u);
    uint32_t x1 = (kv[0] + 0x13579bdfu) ^ salt;
    uint32_t y1 = (kv[1] + 0x2468ace0u) ^ (salt >> 1u);
    uint32_t x2 = (kv[0] ^ 0x55aa33ccu) + salt;
    uint32_t y2 = (kv[1] ^ 0xaa55cc33u) + salt;
    uint32_t x3 = (kv[0] + 0x31415927u) ^ (salt << 3u);
    uint32_t y3 = (kv[1] + 0x27182819u) ^ (salt >> 3u);
    const uint32_t inc0 = kv[2];
    const uint32_t inc1 = kv[3];

    uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
    for (uint32_t it = 0u; it < iters; ++it) {
        for (uint32_t u = 0u; u < 4u; ++u) {
            s0 += vka_sad4_u32(x0, y0); x0 += inc0; y0 += inc1;
            s1 += vka_sad4_u32(x1, y1); x1 += inc0; y1 += inc1;
            s2 += vka_sad4_u32(x2, y2); x2 += inc0; y2 += inc1;
            s3 += vka_sad4_u32(x3, y3); x3 += inc0; y3 += inc1;
        }
    }
    return s0 + s1 + s2 + s3;
}

static inline uint32_t vka_expected_sad_chain(uint32_t wgid, uint32_t lid, uint32_t iters,
                                              uint32_t chain_n, uint32_t break_mask) {
    uint32_t acc = 0u;
    for (uint32_t p = 0u; p < chain_n; ++p) {
        acc += vka_expected_sad_pass(wgid, lid, iters, p, break_mask);
    }
    return acc;
}

// ---- Job (d): Exact 128 MiB streaming copy+XOR reference matching vk_async_stream*.comp ----
static inline void vka_stream_init_vec4(uint32_t idx, uint32_t out[4]) {
    const uint32_t base = idx * 4u;
    out[0] = vkt_hash32(base + 0u);
    out[1] = vkt_hash32(base + 1u);
    out[2] = vkt_hash32(base + 2u);
    out[3] = vkt_hash32(base + 3u);
}

// Returns the final XOR mask written to element `idx` after a chain of `chain_n` dispatches
// with `iters` sweeps per dispatch.
// For chain_n == 1: single dispatch writes all 4 quarters with salt = base_salt ^ 0; last iter is iters - 1.
// For chain_n > 1 (e.g. 80): pass p writes quarter q = (p & 3) with salt = base_salt ^ p; last pass for
// quarter q is the largest p < chain_n with (p & 3) == q.
static inline uint32_t vka_stream_final_mask(uint32_t idx, uint32_t chain_n, uint32_t iters,
                                             uint32_t base_salt, uint32_t break_mask) {
    const uint32_t last_it = (iters > 0u) ? (iters - 1u) : 0u;
    if (chain_n <= 1u) {
        return (base_salt ^ 0u) ^ last_it ^ break_mask;
    }
    const uint32_t q = (idx / VKA_STREAM_QUARTER_VEC4) & 3u;
    uint32_t last_p = q;
    for (uint32_t p = q; p < chain_n; p += 4u) last_p = p;
    return (base_salt ^ last_p) ^ last_it ^ break_mask;
}

static inline void vka_stream_expected_vec4(uint32_t idx, uint32_t chain_n, uint32_t iters,
                                            uint32_t base_salt, uint32_t break_mask, uint32_t out[4]) {
    vka_stream_init_vec4(idx, out);
    const uint32_t m = vka_stream_final_mask(idx, chain_n, iters, base_salt, break_mask);
    for (int c = 0; c < 4; ++c) out[c] ^= m;
}

// ---- Fail-closed 95 C hotspot check (unreadable or invalid sensor = STOP) ----
enum {
    VKA_HOT_SENSOR_ERR = -1,
    VKA_HOT_OVER_LIMIT =  0,
    VKA_HOT_OK         =  1,
};

// Returns VKA_HOT_OK (1) ONLY when `path` is readable, holds a pure integer millidegree
// value in (0, 200000), and is strictly below `stop_c`.
// Returns VKA_HOT_OVER_LIMIT (0) when readable and valid but >= `stop_c`.
// Returns VKA_HOT_SENSOR_ERR (-1) when missing, unreadable, empty, non-numeric, or out of range.
static inline int vka_read_hotspot_status(const char * path, double stop_c, double * out_hot_c) {
    if (out_hot_c) *out_hot_c = -1.0;
    if (!path || !path[0]) return VKA_HOT_SENSOR_ERR;
    FILE * f = fopen(path, "r");
    if (!f) return VKA_HOT_SENSOR_ERR;
    char buf[64];
    size_t n = fread(buf, 1, sizeof buf - 1u, f);
    fclose(f);
    buf[n] = '\0';
    while (n > 0u && (buf[n - 1u] == '\n' || buf[n - 1u] == '\r' ||
                      buf[n - 1u] == ' ' || buf[n - 1u] == '\t')) {
        buf[--n] = '\0';
    }
    if (n == 0u) return VKA_HOT_SENSOR_ERR;
    char * endp = NULL;
    long mc = strtol(buf, &endp, 10);
    if (!endp || *endp != '\0' || mc <= 0L || mc >= 200000L) return VKA_HOT_SENSOR_ERR;
    const double c = (double)mc * 1e-3;
    if (out_hot_c) *out_hot_c = c;
    return c < stop_c ? VKA_HOT_OK : VKA_HOT_OVER_LIMIT;
}

static inline int vka_check_hotspot_file(const char * path, double stop_c, double * out_hot_c) {
    return vka_read_hotspot_status(path, stop_c, out_hot_c) == VKA_HOT_OK ? 1 : 0;
}

// ---- Bounded cooldown wait (< 60 C required, optional soft settle < 50 C) ----
enum {
    VKA_COOL_SENSOR_ERR = -1,
    VKA_COOL_TIMEOUT    =  0,
    VKA_COOL_OK         =  1,
};

typedef struct {
    int status;
    double start_hot_c;
    double end_hot_c;
    int waited_ms;
    int polls;
} VkaCooldownResult;

typedef int (*VkaHotReadFn)(void * ctx, double * out_hot_c);
typedef void (*VkaSleepMsFn)(void * ctx, int ms);

// Waits up to `max_wait_ms` for hotspot < `target_c`. Fails closed immediately on sensor error.
// Once < `target_c`, if `settle_c > 0 && settle_max_ms > 0 && hot >= settle_c`, waits up to
// `settle_max_ms` more for `hot < settle_c` (never fails as long as `hot < target_c`).
static inline int vka_wait_cooldown_fn(VkaHotReadFn read_fn, VkaSleepMsFn sleep_fn, void * ctx,
                                       double target_c, double settle_c,
                                       int max_wait_ms, int settle_max_ms, int poll_ms,
                                       VkaCooldownResult * out) {
    VkaCooldownResult res;
    res.status = VKA_COOL_SENSOR_ERR;
    res.start_hot_c = -1.0;
    res.end_hot_c = -1.0;
    res.waited_ms = 0;
    res.polls = 0;
    if (poll_ms <= 0) poll_ms = 100;
    if (max_wait_ms < 0) max_wait_ms = 0;
    if (settle_max_ms < 0) settle_max_ms = 0;

    double hot = -1.0;
    int st = read_fn(ctx, &hot);
    res.polls = 1;
    if (st == VKA_HOT_SENSOR_ERR) {
        if (out) *out = res;
        return VKA_COOL_SENSOR_ERR;
    }
    res.start_hot_c = hot;
    res.end_hot_c = hot;

    while (hot >= target_c) {
        if (res.waited_ms >= max_wait_ms) {
            res.status = VKA_COOL_TIMEOUT;
            if (out) *out = res;
            return VKA_COOL_TIMEOUT;
        }
        int step = max_wait_ms - res.waited_ms;
        if (step > poll_ms) step = poll_ms;
        sleep_fn(ctx, step);
        res.waited_ms += step;
        st = read_fn(ctx, &hot);
        ++res.polls;
        if (st == VKA_HOT_SENSOR_ERR) {
            res.status = VKA_COOL_SENSOR_ERR;
            res.end_hot_c = -1.0;
            if (out) *out = res;
            return VKA_COOL_SENSOR_ERR;
        }
        res.end_hot_c = hot;
    }

    if (settle_c > 0.0 && settle_max_ms > 0 && hot >= settle_c) {
        int sw = 0;
        int flat_polls = 0;
        double prev_hot = hot;
        while (hot >= settle_c && sw < settle_max_ms) {
            int step = settle_max_ms - sw;
            if (step > poll_ms) step = poll_ms;
            sleep_fn(ctx, step);
            res.waited_ms += step;
            sw += step;
            st = read_fn(ctx, &hot);
            ++res.polls;
            if (st == VKA_HOT_SENSOR_ERR) {
                res.status = VKA_COOL_SENSOR_ERR;
                res.end_hot_c = -1.0;
                if (out) *out = res;
                return VKA_COOL_SENSOR_ERR;
            }
            res.end_hot_c = hot;
            if (hot < prev_hot - 0.1) {
                flat_polls = 0;
            } else {
                ++flat_polls;
                if (flat_polls >= 2) break;
            }
            prev_hot = hot;
        }
        if (hot >= target_c) {
            res.status = VKA_COOL_TIMEOUT;
            if (out) *out = res;
            return VKA_COOL_TIMEOUT;
        }
    }

    res.status = VKA_COOL_OK;
    if (out) *out = res;
    return VKA_COOL_OK;
}

typedef struct {
    const char * path;
    double stop_c;
} VkaFileHotCtx;

static inline int vka_file_hot_read_cb(void * ctx, double * out_hot_c) {
    const VkaFileHotCtx * c = (const VkaFileHotCtx *)ctx;
    return vka_read_hotspot_status(c->path, c->stop_c, out_hot_c);
}

static inline void vka_nanosleep_ms_cb(void * ctx, int ms) {
    (void)ctx;
    if (ms <= 0) return;
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static inline int vka_wait_cooldown_file(const char * path, double stop_c,
                                         double target_c, double settle_c,
                                         int max_wait_ms, int settle_max_ms, int poll_ms,
                                         VkaCooldownResult * out) {
    VkaFileHotCtx ctx = {path, stop_c};
    return vka_wait_cooldown_fn(vka_file_hot_read_cb, vka_nanosleep_ms_cb, &ctx,
                                target_c, settle_c, max_wait_ms, settle_max_ms, poll_ms, out);
}

// ---- Hard pre-cell < 60 C cooldown policy ----
// A missed < 60 C cooldown MUST NOT run the cell:
//   - VKA_PRECELL_RUN (0):        cooldown OK (< target_c) and < stop_c -> proceed
//   - VKA_PRECELL_FAIL_CELL (1):  cooldown TIMEOUT with end_hot_c < stop_c -> skip submit, fail cell & verdict (exit 1)
//   - VKA_PRECELL_ABORT_HOT (4):  sensor error or end_hot_c >= stop_c -> skip submit, abort run (exit 4)
enum {
    VKA_PRECELL_RUN       = 0,
    VKA_PRECELL_FAIL_CELL = 1,
    VKA_PRECELL_ABORT_HOT = 4,
};

static inline int vka_precell_cooldown_action(int cool_status, double end_hot_c, double stop_c) {
    if (cool_status == VKA_COOL_SENSOR_ERR) return VKA_PRECELL_ABORT_HOT;
    if (cool_status == VKA_COOL_OK) {
        if (!(end_hot_c > 0.0) || end_hot_c >= stop_c) return VKA_PRECELL_ABORT_HOT;
        return VKA_PRECELL_RUN;
    }
    return (end_hot_c >= stop_c) ? VKA_PRECELL_ABORT_HOT : VKA_PRECELL_FAIL_CELL;
}

// ---- Double-stop thermal policy (continue after 1st >= 95 C stop, abort run on 2nd or sensor error) ----
enum {
    VKA_THERM_NONE              = 0,
    VKA_THERM_CONTINUE_NEXT_ARM = 1,
    VKA_THERM_ABORT_RUN         = 2,
};

typedef struct {
    int stop_count;
    int sensor_errors;
    int cooldown_timeouts;
    int cooldown_hot_aborts;
    int max_stops;
} VkaThermalState;

static inline void vka_thermal_init(VkaThermalState * st, int max_stops) {
    st->stop_count = 0;
    st->sensor_errors = 0;
    st->cooldown_timeouts = 0;
    st->cooldown_hot_aborts = 0;
    st->max_stops = (max_stops > 0) ? max_stops : VKA_THERMAL_MAX_STOPS;
}

static inline int vka_thermal_merge_status(int a, int b) {
    if (a == VKA_HOT_SENSOR_ERR || b == VKA_HOT_SENSOR_ERR) return VKA_HOT_SENSOR_ERR;
    if (a == VKA_HOT_OVER_LIMIT || b == VKA_HOT_OVER_LIMIT) return VKA_HOT_OVER_LIMIT;
    return VKA_HOT_OK;
}

static inline int vka_thermal_on_event(VkaThermalState * st, int hot_status) {
    if (hot_status == VKA_HOT_OK) return VKA_THERM_NONE;
    if (hot_status == VKA_HOT_SENSOR_ERR) {
        ++st->sensor_errors;
        return VKA_THERM_ABORT_RUN;
    }
    ++st->stop_count;
    return (st->stop_count >= st->max_stops) ? VKA_THERM_ABORT_RUN : VKA_THERM_CONTINUE_NEXT_ARM;
}

static inline int vka_thermal_exit_code(const VkaThermalState * st, int total_fails) {
    if (st->sensor_errors > 0 || st->cooldown_hot_aborts > 0 || st->stop_count >= st->max_stops) return 4;
    if (st->stop_count > 0 || st->cooldown_timeouts > 0 || total_fails > 0) return 1;
    return 0;
}

// ---- ABBA schedule builder ----
// Given `n_allowed` priority indices in `allowed[0..n_allowed-1]`, fills `out_seq` with the
// palindrome ABBA sequence [P0, P1, ..., P_{k-1}, P_{k-1}, ..., P1, P0] of length 2 * n_allowed.
static inline int vka_build_abba_schedule(const int * allowed, int n_allowed, int * out_seq) {
    if (n_allowed <= 0) return 0;
    for (int i = 0; i < n_allowed; ++i) {
        out_seq[i] = allowed[i];
        out_seq[2 * n_allowed - 1 - i] = allowed[i];
    }
    return 2 * n_allowed;
}

// ---- Graphics frame completion interval overlap statistics ----
typedef struct {
    double t_sub_ms;
    double t_done_ms;
    double gpu_ms;
    double wall_ms;
    double hot_c;
} VkaGfxFrame;

typedef struct {
    double t_start_ms;
    double t_end_ms;
    double gpu_ms;
    double wall_ms;
} VkaCompTrial;

typedef struct {
    int n_strict_ov;
    int n_ov_frames;
    int n_dt;
    double dt_avg_ms;
    double dt_p50_ms;
    double dt_p95_ms;
    double dt_max_ms;
    double gfx_gpu_p50_ms;
    double gfx_wall_p50_ms;
} VkaGfxOverlapStats;

static inline int vka_cmp_dbl(const void * a, const void * b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// Percentile matching src/round12/gfx_load.c:90-101 (nearest-rank rounding).
static inline double vka_percentile(const double * v, int n, double p) {
    if (n <= 0) return NAN;
    double * c = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(c, v, sizeof(double) * (size_t)n);
    qsort(c, (size_t)n, sizeof(double), vka_cmp_dbl);
    int idx = (int)llround(p * (double)(n - 1));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    const double r = c[idx];
    free(c);
    return r;
}

static inline VkaGfxOverlapStats vka_compute_gfx_overlap(const VkaGfxFrame * frames, int n_frames,
                                                         const VkaCompTrial * trials, int n_trials) {
    VkaGfxOverlapStats st;
    memset(&st, 0, sizeof st);
    st.dt_avg_ms = NAN;
    st.dt_p50_ms = NAN;
    st.dt_p95_ms = NAN;
    st.dt_max_ms = NAN;
    st.gfx_gpu_p50_ms = NAN;
    st.gfx_wall_p50_ms = NAN;
    if (n_frames <= 0 || n_trials <= 0) return st;

    const double g_lo = frames[0].t_sub_ms;
    const double g_hi = frames[n_frames - 1].t_done_ms;
    for (int t = 0; t < n_trials; ++t) {
        if (trials[t].t_start_ms >= g_lo && trials[t].t_end_ms <= g_hi) {
            ++st.n_strict_ov;
        }
    }

    double * dts = (double *)malloc(sizeof(double) * (size_t)n_frames);
    double * gpus = (double *)malloc(sizeof(double) * (size_t)n_frames);
    double * walls = (double *)malloc(sizeof(double) * (size_t)n_frames);
    int n_ov = 0, n_dt = 0;
    double dt_sum = 0.0;

    for (int i = 0; i < n_frames; ++i) {
        int ov = 0;
        for (int t = 0; t < n_trials; ++t) {
            if (frames[i].t_done_ms >= trials[t].t_start_ms &&
                frames[i].t_sub_ms <= trials[t].t_end_ms) {
                ov = 1;
                break;
            }
        }
        if (!ov) continue;
        gpus[n_ov] = frames[i].gpu_ms;
        walls[n_ov] = frames[i].wall_ms;
        ++n_ov;
        if (i > 0) {
            const double dt = frames[i].t_done_ms - frames[i - 1].t_done_ms;
            dts[n_dt++] = dt;
            dt_sum += dt;
        }
    }

    st.n_ov_frames = n_ov;
    st.n_dt = n_dt;
    if (n_dt > 0) {
        st.dt_avg_ms = dt_sum / (double)n_dt;
        st.dt_p50_ms = vka_percentile(dts, n_dt, 0.50);
        st.dt_p95_ms = vka_percentile(dts, n_dt, 0.95);
        st.dt_max_ms = vka_percentile(dts, n_dt, 1.00);
    }
    if (n_ov > 0) {
        st.gfx_gpu_p50_ms = vka_percentile(gpus, n_ov, 0.50);
        st.gfx_wall_p50_ms = vka_percentile(walls, n_ov, 0.50);
    }
    free(dts);
    free(gpus);
    free(walls);
    return st;
}

// Multi-repeat overlap accumulator: aggregates overlapping graphics frames across independent
// per-repeat GfxWorker bursts without ever taking a frame-interval delta across cooldown gaps.
typedef struct {
    double * dts;
    double * gpus;
    double * walls;
    int n_dt;
    int cap_dt;
    int n_ov;
    int cap_ov;
    int n_strict_ov;
    double dt_sum;
} VkaOverlapAccum;

static inline void vka_overlap_accum_init(VkaOverlapAccum * a) {
    memset(a, 0, sizeof *a);
}

static inline void vka_overlap_accum_free(VkaOverlapAccum * a) {
    free(a->dts);
    free(a->gpus);
    free(a->walls);
    memset(a, 0, sizeof *a);
}

static inline void vka_overlap_accum_add_batch(VkaOverlapAccum * a,
                                               const VkaGfxFrame * frames, int n_frames,
                                               const VkaCompTrial * trials, int n_trials) {
    if (n_frames <= 0 || n_trials <= 0) return;
    const double g_lo = frames[0].t_sub_ms;
    const double g_hi = frames[n_frames - 1].t_done_ms;
    for (int t = 0; t < n_trials; ++t) {
        if (trials[t].t_start_ms >= g_lo && trials[t].t_end_ms <= g_hi) {
            ++a->n_strict_ov;
        }
    }
    for (int i = 0; i < n_frames; ++i) {
        int ov = 0;
        for (int t = 0; t < n_trials; ++t) {
            if (frames[i].t_done_ms >= trials[t].t_start_ms &&
                frames[i].t_sub_ms <= trials[t].t_end_ms) {
                ov = 1;
                break;
            }
        }
        if (!ov) continue;
        if (a->n_ov >= a->cap_ov) {
            const int ncap = a->cap_ov ? a->cap_ov * 2 : 64;
            a->gpus = (double *)realloc(a->gpus, sizeof(double) * (size_t)ncap);
            a->walls = (double *)realloc(a->walls, sizeof(double) * (size_t)ncap);
            a->cap_ov = ncap;
        }
        a->gpus[a->n_ov] = frames[i].gpu_ms;
        a->walls[a->n_ov] = frames[i].wall_ms;
        ++a->n_ov;
        if (i > 0) {
            if (a->n_dt >= a->cap_dt) {
                const int ncap = a->cap_dt ? a->cap_dt * 2 : 64;
                a->dts = (double *)realloc(a->dts, sizeof(double) * (size_t)ncap);
                a->cap_dt = ncap;
            }
            const double dt = frames[i].t_done_ms - frames[i - 1].t_done_ms;
            a->dts[a->n_dt++] = dt;
            a->dt_sum += dt;
        }
    }
}

static inline VkaGfxOverlapStats vka_overlap_accum_finish(const VkaOverlapAccum * a) {
    VkaGfxOverlapStats st;
    memset(&st, 0, sizeof st);
    st.n_strict_ov = a->n_strict_ov;
    st.n_ov_frames = a->n_ov;
    st.n_dt = a->n_dt;
    st.dt_avg_ms = (a->n_dt > 0) ? (a->dt_sum / (double)a->n_dt) : NAN;
    st.dt_p50_ms = vka_percentile(a->dts, a->n_dt, 0.50);
    st.dt_p95_ms = vka_percentile(a->dts, a->n_dt, 0.95);
    st.dt_max_ms = vka_percentile(a->dts, a->n_dt, 1.00);
    st.gfx_gpu_p50_ms = vka_percentile(a->gpus, a->n_ov, 0.50);
    st.gfx_wall_p50_ms = vka_percentile(a->walls, a->n_ov, 0.50);
    return st;
}

#endif
