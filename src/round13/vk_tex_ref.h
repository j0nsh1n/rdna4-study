// CPU reference for Round 13 Probes B (vk_tex) and C (vk_tex_fp16):
// format definitions, exact dyadic texel generators, exact bilinear/gather/fetch evaluators,
// FP16/B10G11R11 encoders, exact FP16/FP32 FMA oscillation reference, and bandwidth/rate math.
// No Vulkan dependency: included by vk_tex.c, vk_tex_fp16.c, and vk_tex_selftest.c.
#ifndef VK_TEX_REF_H
#define VK_TEX_REF_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VKT_FMT_R8_UNORM = 0,
    VKT_FMT_RGBA8_UNORM = 1,
    VKT_FMT_R16G16_SFLOAT = 2,
    VKT_FMT_RGBA16_SFLOAT = 3,
    VKT_FMT_RGBA32_SFLOAT = 4,
    VKT_FMT_B10G11R11_UFLOAT = 5,
    VKT_FMT_COUNT = 6
};

enum {
    VKT_OP_FETCH = 0,
    VKT_OP_LOD = 1,
    VKT_OP_GATHER = 2,
    VKT_OP_COUNT = 3
};

enum {
    VKT_LVL_L0 = 0,    // 16 KiB
    VKT_LVL_L2 = 1,    // 2 MiB
    VKT_LVL_MALL = 2,  // 32 MiB
    VKT_LVL_DRAM = 3,  // 512 MiB (256 MiB for 1 B/texel R8_UNORM at maxImageDimension2D=16384)
    VKT_LVL_COUNT = 4
};

#define VKT_MAX_DIM_2D       16384u
#define VKT_MIN_CLK_SAMPLES  5
#define VKT_CLK_SLOTS        64u
#define VKT_RT_TICK_NS       10.0   // 100 MHz MSG_RTN_GET_REALTIME on gfx1201
#define VKT_MIN_CLK_RT_NS    1000.0 // >= 1.0 us (100 ticks @ 100 MHz) rejects reordered/1-tick spans
#define VKT_MIN_CLK_CYCLES   2000ull
#define VKT_WARM_TARGET_MS   20.0
#define VKT_WARM_MIN_REPS    3
#define VKT_WARM_MAX_REPS    40
#define VKT_TIMED_TARGET_MS  30.0
#define VKT_TIMED_MIN_REPS   8
#define VKT_TIMED_MAX_REPS   48
#define VKT_TIMED_MAX_WALL_MS 60.0

static const char * const vkt_fmt_names[VKT_FMT_COUNT] = {
    "R8_UNORM",
    "RGBA8_UNORM",
    "R16G16_SFLOAT",
    "R16G16B16A16_SFLOAT",
    "R32G32B32A32_SFLOAT",
    "B10G11R11_UFLOAT"
};

static const uint32_t vkt_fmt_bpp[VKT_FMT_COUNT] = {1u, 4u, 4u, 8u, 16u, 4u};
static const uint32_t vkt_fmt_channels[VKT_FMT_COUNT] = {1u, 4u, 2u, 4u, 4u, 3u};

static const char * const vkt_op_names[VKT_OP_COUNT] = {"texelFetch", "textureLod", "textureGather"};
static const char * const vkt_lvl_names[VKT_LVL_COUNT] = {"L0", "L2", "MALL", "DRAM"};
static const uint64_t vkt_lvl_bytes[VKT_LVL_COUNT] = {
    16ull << 10,
    2ull << 20,
    32ull << 20,
    512ull << 20
};

// Target footprint bytes for (fmt, lvl), capped by single-2D-image hardware limit 16384x16384*bpp
// (256 MiB = 4x MALL for 1 B/texel R8_UNORM; 512 MiB = 8x MALL for all bpp >= 4 formats).
static inline uint64_t vkt_level_bytes(int fmt, int lvl) {
    const uint64_t max_2d = (uint64_t)VKT_MAX_DIM_2D * (uint64_t)VKT_MAX_DIM_2D * (uint64_t)vkt_fmt_bpp[fmt];
    const uint64_t want = vkt_lvl_bytes[lvl];
    return (want <= max_2d) ? want : max_2d;
}

// Power-of-two 2D dimensions (w >= 32, h >= 32, w <= 16384, h <= 16384) matching target bytes.
static inline int vkt_level_dims(int fmt, int lvl, uint32_t * out_w, uint32_t * out_h) {
    const uint64_t bytes = vkt_level_bytes(fmt, lvl);
    const uint64_t texels = bytes / vkt_fmt_bpp[fmt];
    int lg = 0;
    while ((1ull << lg) < texels) ++lg;
    uint32_t w = 1u << ((lg + 1) / 2);
    uint32_t h = (uint32_t)(texels / w);
    if (w < 32u || h < 32u || w > VKT_MAX_DIM_2D || h > VKT_MAX_DIM_2D ||
        (uint64_t)w * h * vkt_fmt_bpp[fmt] != bytes) {
        return 0;
    }
    *out_w = w;
    *out_h = h;
    return 1;
}

static inline uint32_t vkt_hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline uint32_t vkt_tex_hash(uint32_t x, uint32_t y, uint32_t ch) {
    return vkt_hash32(((y & 0xffffu) << 16) ^ (x & 0xffffu) ^ (ch * 0x9e3779b9u));
}

// IEEE-754 float -> half (round-to-nearest-even, finite positive values).
static inline uint16_t vkt_f32_to_f16(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        const uint32_t mm = mant | 0x800000u;
        const int sh = 14 - exp;
        uint32_t h = mm >> sh;
        const uint32_t rem = mm & ((1u << sh) - 1u);
        const uint32_t mid = 1u << (sh - 1);
        if (rem > mid || (rem == mid && (h & 1u))) ++h;
        return (uint16_t)(sign | h);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t)(sign | h);
}

static inline float vkt_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t u = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            u = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            u = sign | ((uint32_t)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    memcpy(&f, &u, 4);
    return f;
}

// Unsigned small float encoder for VK_FORMAT_B10G11R11_UFLOAT_PACK32 (non-negative finite values).
static inline uint32_t vkt_f32_to_uf11(float f) {
    if (f <= 0.0f) return 0u;
    uint32_t x = 0;
    memcpy(&x, &f, 4);
    const int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return 0u;
    if (exp >= 31) return 0x7c0u;
    return ((uint32_t)exp << 6) | (mant >> 17);
}

static inline uint32_t vkt_f32_to_uf10(float f) {
    if (f <= 0.0f) return 0u;
    uint32_t x = 0;
    memcpy(&x, &f, 4);
    const int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return 0u;
    if (exp >= 31) return 0x3e0u;
    return ((uint32_t)exp << 5) | (mant >> 18);
}

static inline uint32_t vkt_pack_b10g11r11(float r, float g, float b) {
    return (vkt_f32_to_uf11(r) & 0x7ffu) |
           ((vkt_f32_to_uf11(g) & 0x7ffu) << 11) |
           ((vkt_f32_to_uf10(b) & 0x3ffu) << 22);
}

// Deterministic texel value at (x, y) for format `fmt`.
// exact_mode = 1: guarantees every texel AND every bilinear blend at weights (3/8, 1/8, 3/8, 1/8)
//                 is 100% bit-exact in FP16 and FP32 (for UNORM8 uses {0.0, 1.0}; for B10G11R11
//                 uses [1.0, 1.96875] on 1/32 grid; for FP16/FP32 uses [1.0, 1.9921875] on 1/128 grid).
// exact_mode = 0: for UNORM8 uses full 8-bit random bytes {0..255} (incompressible against DCC);
//                 for all float formats stays on the exact incompressible dyadic grid.
static inline void vkt_texel_rgba(int fmt, int exact_mode, uint32_t x, uint32_t y, float out[4]) {
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    out[3] = 1.0f;
    switch (fmt) {
        case VKT_FMT_R8_UNORM: {
            uint32_t h = vkt_tex_hash(x, y, 0u);
            out[0] = exact_mode ? ((h & 1u) ? 1.0f : 0.0f) : (float)(h & 0xffu) / 255.0f;
            break;
        }
        case VKT_FMT_RGBA8_UNORM: {
            for (uint32_t c = 0; c < 4u; ++c) {
                uint32_t h = vkt_tex_hash(x, y, c);
                out[c] = exact_mode ? ((h & 1u) ? 1.0f : 0.0f) : (float)(h & 0xffu) / 255.0f;
            }
            break;
        }
        case VKT_FMT_R16G16_SFLOAT: {
            for (uint32_t c = 0; c < 2u; ++c) {
                uint32_t h = vkt_tex_hash(x, y, c);
                out[c] = (float)(128u + (h & 127u)) / 128.0f;
            }
            break;
        }
        case VKT_FMT_RGBA16_SFLOAT:
        case VKT_FMT_RGBA32_SFLOAT: {
            for (uint32_t c = 0; c < 4u; ++c) {
                uint32_t h = vkt_tex_hash(x, y, c);
                out[c] = (float)(128u + (h & 127u)) / 128.0f;
            }
            break;
        }
        case VKT_FMT_B10G11R11_UFLOAT: {
            for (uint32_t c = 0; c < 3u; ++c) {
                uint32_t h = vkt_tex_hash(x, y, c);
                out[c] = (float)(32u + (h & 31u)) / 32.0f;
            }
            break;
        }
    }
}

static inline void vkt_encode_texel(int fmt, int exact_mode, uint32_t x, uint32_t y, uint8_t dst[16]) {
    float c[4];
    vkt_texel_rgba(fmt, exact_mode, x, y, c);
    switch (fmt) {
        case VKT_FMT_R8_UNORM:
            dst[0] = (uint8_t)floor(c[0] * 255.0 + 0.5);
            break;
        case VKT_FMT_RGBA8_UNORM:
            for (int i = 0; i < 4; ++i) dst[i] = (uint8_t)floor(c[i] * 255.0 + 0.5);
            break;
        case VKT_FMT_R16G16_SFLOAT: {
            uint16_t h[2] = {vkt_f32_to_f16(c[0]), vkt_f32_to_f16(c[1])};
            memcpy(dst, h, 4);
            break;
        }
        case VKT_FMT_RGBA16_SFLOAT: {
            uint16_t h[4] = {vkt_f32_to_f16(c[0]), vkt_f32_to_f16(c[1]),
                             vkt_f32_to_f16(c[2]), vkt_f32_to_f16(c[3])};
            memcpy(dst, h, 8);
            break;
        }
        case VKT_FMT_RGBA32_SFLOAT:
            memcpy(dst, c, 16);
            break;
        case VKT_FMT_B10G11R11_UFLOAT: {
            uint32_t p = vkt_pack_b10g11r11(c[0], c[1], c[2]);
            memcpy(dst, &p, 4);
            break;
        }
    }
}

// Exact evaluation of one sampling op at integer base (ix, iy) on a W x H power-of-two texture.
// Coordinates used by textureLod and textureGather are u = (ix + 0.75)/W, v = (iy + 1.0)/H,
// giving exact bilinear weights wx = 0.25, wy = 0.50 -> (0.375, 0.125, 0.375, 0.125).
static inline void vkt_eval_sample(int fmt, int op, int exact_mode, uint32_t w, uint32_t h,
                                   uint32_t ix, uint32_t iy, float out[4]) {
    const uint32_t mx = w - 1u;
    const uint32_t my = h - 1u;
    const uint32_t x0 = ix & mx;
    const uint32_t y0 = iy & my;
    if (op == VKT_OP_FETCH) {
        vkt_texel_rgba(fmt, exact_mode, x0, y0, out);
        return;
    }
    const uint32_t x1 = (x0 + 1u) & mx;
    const uint32_t y1 = (y0 + 1u) & my;
    float t00[4], t10[4], t01[4], t11[4];
    vkt_texel_rgba(fmt, exact_mode, x0, y0, t00);
    vkt_texel_rgba(fmt, exact_mode, x1, y0, t10);
    vkt_texel_rgba(fmt, exact_mode, x0, y1, t01);
    vkt_texel_rgba(fmt, exact_mode, x1, y1, t11);
    if (op == VKT_OP_LOD) {
        for (int c = 0; c < 4; ++c) {
            out[c] = 0.375f * t00[c] + 0.125f * t10[c] + 0.375f * t01[c] + 0.125f * t11[c];
        }
    } else {
        // Vulkan textureGather(tex, uv, 0): CCW from bottom-left (x0,y1), (x1,y1), (x1,y0), (x0,y0).
        out[0] = t01[0];
        out[1] = t11[0];
        out[2] = t10[0];
        out[3] = t00[0];
    }
}

// Compute workgroup starting tile (wg_tx, wg_ty) for non-overlapping horizontal strips of `taps`
// 16x16 tiles, guaranteeing 100% texture footprint coverage across all cache levels and formats.
static inline void vkt_wg_origin(uint32_t w, uint32_t h, uint32_t taps, uint32_t wgid,
                                 uint32_t * out_tx, uint32_t * out_ty) {
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (taps < tiles_x) ? taps : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    *out_tx = (wgid & (strips_x - 1u)) * strip;
    *out_ty = (wgid / strips_x) & (tiles_y - 1u);
}

// Compute expected accumulated vec4 for invocation (wgid, lid) running `iters` outer iterations
// of 16 taps matching src/round13/shaders/vk_tex.comp.
static inline void vkt_expected_thread(int fmt, int op, int exact_mode,
                                       uint32_t w, uint32_t h,
                                       uint32_t wgid, uint32_t lid,
                                       uint32_t iters, uint32_t y_tile_step,
                                       uint32_t break_dx, float acc[4]) {
    const uint32_t lane = lid & 31u;
    const uint32_t sub = lid >> 5u;
    const uint32_t lx = (lane & 7u) + ((sub & 1u) << 3u);
    const uint32_t ly = ((lane >> 3u) & 3u) + ((sub >> 1u) << 2u);
    uint32_t wg_tx = 0u, wg_ty = 0u;
    vkt_wg_origin(w, h, 16u, wgid, &wg_tx, &wg_ty);
    const uint32_t bx0 = ((wg_tx << 4u) + lx + break_dx) & (w - 1u);
    uint32_t by = ((wg_ty << 4u) + ly) & (h - 1u);

    acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
    for (uint32_t it = 0; it < iters; ++it) {
        uint32_t ix = bx0;
        for (uint32_t k = 0; k < 16u; ++k) {
            float s[4];
            vkt_eval_sample(fmt, op, exact_mode, w, h, ix, by, s);
            for (int c = 0; c < 4; ++c) acc[c] += s[c];
            ix = (ix + 16u) & (w - 1u);
        }
        by = (by + (y_tile_step << 4u)) & (h - 1u);
    }
}

// Exact oscillating FMA reference used by Probe C (vk_tex_fp16):
// x_{n+1} = fma(x_n, -1.0, add_k) = add_k - x_n.
// Period-2 exact in both FP16 and FP32 for arbitrarily many steps with zero overflow or rounding.
static inline float vkt_osc_fma_ref(float x0, float add_k, uint64_t steps) {
    return (steps & 1ull) ? (add_k - x0) : x0;
}

// Per-lane oscillating FMA constants for Probe C (all exact in FP16 and FP32).
static inline void vkt_osc_lane_k(uint32_t lid, float * ak_lo, float * ak_hi) {
    *ak_lo = 1.5f + (float)(lid & 7u) * 0.125f;
    *ak_hi = 1.75f + (float)((lid >> 3u) & 7u) * 0.125f;
}

// Expected sum of 5 independent oscillating chains (a0..a4) after `total_fmas` sequential steps
// distributed round-robin (step s updates chain s % 5). Returns (sum_lo, sum_hi) for f16vec2
// (for scalar FP32, use sum_lo only).
static inline void vkt_expected_osc5_sum(uint32_t lid, uint64_t total_fmas, float * sum_lo, float * sum_hi) {
    static const float init_lo[5] = {0.25f, 0.375f, 0.5f, 0.625f, 0.75f};
    static const float init_hi[5] = {0.5f, 0.625f, 0.75f, 0.875f, 1.0f};
    float ak_lo = 0.0f, ak_hi = 0.0f;
    vkt_osc_lane_k(lid, &ak_lo, &ak_hi);
    float sl = 0.0f, sh = 0.0f;
    for (uint64_t c = 0; c < 5ull; ++c) {
        const uint64_t steps = (total_fmas > c) ? ((total_fmas - 1ull - c) / 5ull + 1ull) : 0ull;
        sl += vkt_osc_fma_ref(init_lo[c], ak_lo, steps);
        sh += vkt_osc_fma_ref(init_hi[c], ak_hi, steps);
    }
    *sum_lo = sl;
    *sum_hi = sh;
}

#define VKT_OSC5_INIT_SUM_LO 2.5f   // 0.25 + 0.375 + 0.5 + 0.625 + 0.75
#define VKT_OSC5_INIT_SUM_HI 3.75f  // 0.5 + 0.625 + 0.75 + 0.875 + 1.0

// Expected 5-chain oscillating FMA sum for Probe C (vk_tex_fp16.comp) and Probe D (vk_async fp16).
// Each outer iteration executes 4 * n_fma round-robin steps across chains 0..4; when n_fma > 0
// and iters is positive and even (e.g. visit-once DRAM), the shader executes one extra step per
// chain after the loop so every chain has an odd total step count and non-trivial parity.
static inline void vkt_expected_mix_alu(uint32_t lid, uint32_t n_fma, uint32_t iters,
                                        float * want_lo, float * want_hi) {
    static const float init_lo[5] = {0.25f, 0.375f, 0.5f, 0.625f, 0.75f};
    static const float init_hi[5] = {0.5f, 0.625f, 0.75f, 0.875f, 1.0f};
    uint64_t per_iter[5] = {0, 0, 0, 0, 0};
    const uint32_t iter_fmas = 4u * n_fma;
    for (uint32_t s = 0u; s < iter_fmas; ++s) ++per_iter[s % 5u];
    const uint64_t extra = (n_fma > 0u && iters > 0u && (iters & 1u) == 0u) ? 1ull : 0ull;
    float ak_lo = 0.0f, ak_hi = 0.0f;
    vkt_osc_lane_k(lid, &ak_lo, &ak_hi);
    float wl = 0.0f, wh = 0.0f;
    for (int c = 0; c < 5; ++c) {
        const uint64_t steps = per_iter[c] * (uint64_t)iters + extra;
        wl += vkt_osc_fma_ref(init_lo[c], ak_lo, steps);
        wh += vkt_osc_fma_ref(init_hi[c], ak_hi, steps);
    }
    *want_lo = wl;
    *want_hi = wh;
}

// Verify one lane's stored ALU output (got_lo, got_hi) against vkt_expected_mix_alu.
// Returns 1 if exact AND non-trivial (fails whenever n_fma > 0 && iters > 0 and either the
// stored ALU result or the expected value equals the untouched initial accumulator sum).
static inline int vkt_verify_mix_alu_lane(uint32_t lid, uint32_t n_fma, uint32_t iters,
                                          int use_fp16, float got_lo, float got_hi,
                                          int break_alu, float * out_err) {
    float want_lo = 0.0f, want_hi = 0.0f;
    vkt_expected_mix_alu(lid, n_fma, iters, &want_lo, &want_hi);
    if (break_alu) want_lo += 0.25f;
    const float elo = fabsf(got_lo - want_lo);
    const float ehi = use_fp16 ? fabsf(got_hi - want_hi) : 0.0f;
    const float err = (elo > ehi) ? elo : ehi;
    if (out_err) *out_err = err;
    if (n_fma > 0u && iters > 0u) {
        const int got_is_init = (got_lo == VKT_OSC5_INIT_SUM_LO) &&
                                (!use_fp16 || got_hi == VKT_OSC5_INIT_SUM_HI);
        const int want_is_init = (want_lo == VKT_OSC5_INIT_SUM_LO) &&
                                 (!use_fp16 || want_hi == VKT_OSC5_INIT_SUM_HI);
        if (got_is_init || want_is_init) return 0;
    }
    return (elo == 0.0f && ehi == 0.0f) ? 1 : 0;
}

// Compute expected accumulated vec4 for 4-tap-per-iter texture loop in Probe C (vk_tex_fp16.comp).
static inline void vkt_expected_thread_4tap(int fmt, int exact_mode,
                                            uint32_t w, uint32_t h,
                                            uint32_t wgid, uint32_t lid,
                                            uint32_t iters, uint32_t y_tile_step,
                                            uint32_t break_dx, float acc[4]) {
    const uint32_t lane = lid & 31u;
    const uint32_t sub = lid >> 5u;
    const uint32_t lx = (lane & 7u) + ((sub & 1u) << 3u);
    const uint32_t ly = ((lane >> 3u) & 3u) + ((sub >> 1u) << 2u);
    uint32_t wg_tx = 0u, wg_ty = 0u;
    vkt_wg_origin(w, h, 4u, wgid, &wg_tx, &wg_ty);
    const uint32_t bx0 = ((wg_tx << 4u) + lx + break_dx) & (w - 1u);
    uint32_t by = ((wg_ty << 4u) + ly) & (h - 1u);

    acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
    for (uint32_t it = 0; it < iters; ++it) {
        uint32_t ix = bx0;
        for (uint32_t k = 0; k < 4u; ++k) {
            float s[4];
            vkt_eval_sample(fmt, VKT_OP_LOD, exact_mode, w, h, ix, by, s);
            for (int c = 0; c < 4; ++c) acc[c] += s[c];
            ix = (ix + 16u) & (w - 1u);
        }
        by = (by + (y_tile_step << 4u)) & (h - 1u);
    }
}

// Metric helpers: texels/clock/CU and GB/s (64 CUs on RX 9070 XT).
static inline double vkt_samp_per_clk_per_cu(uint64_t total_samples, double gpu_ns, double sclk_mhz) {
    if (gpu_ns <= 0.0 || sclk_mhz <= 0.0) return 0.0;
    const double s_per_sec = (double)total_samples / (gpu_ns * 1e-9);
    return s_per_sec / (64.0 * sclk_mhz * 1e6);
}

static inline double vkt_gbs(uint64_t total_samples, uint32_t bpp, double gpu_ns) {
    if (gpu_ns <= 0.0) return 0.0;
    return ((double)total_samples * (double)bpp) / gpu_ns;
}

// Compute visit-once DRAM dispatch geometry (groups, iters, y_tile_step) for format `fmt`
// and taps-per-iter `taps` (16 for vk_tex, 4 for vk_tex_fp16).
// Guarantees groups * 256 * iters * taps == w * h with zero strip wrap or tile reuse.
static inline int vkt_dram_schedule(int fmt, uint32_t taps,
                                    uint32_t * out_groups, uint32_t * out_iters,
                                    uint32_t * out_y_step) {
    uint32_t w = 0u, h = 0u;
    if (!vkt_level_dims(fmt, VKT_LVL_DRAM, &w, &h)) return 0;
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    if (taps == 0u || tiles_x < taps || (tiles_x % taps) != 0u) return 0;
    const uint32_t strips_x = tiles_x / taps;
    uint32_t iters = 8u;
    if (taps == 16u) {
        iters = (vkt_fmt_bpp[fmt] == 1u) ? 16u : 8u;
    } else if (taps == 4u) {
        iters = 16u;
    }
    while (iters > 1u && (tiles_y % iters) != 0u) iters >>= 1u;
    const uint32_t row_span = tiles_y / iters;
    if (row_span == 0u || row_span * iters != tiles_y) return 0;
    const uint32_t groups = strips_x * row_span;
    if (groups == 0u || groups > 65535u) return 0;
    if (out_groups) *out_groups = groups;
    if (out_iters) *out_iters = iters;
    if (out_y_step) *out_y_step = row_span;
    return 1;
}

typedef struct {
    uint64_t total_items;
    uint64_t total_visits;
    uint64_t unvisited;
    uint64_t revisited;
    uint32_t max_visits;
    int lid_bijection_ok;
} VktVisitStats;

// Verify local invocation mapping lid in [0..255] -> (lx, ly) in [0..15]x[0..15] is a bijection
// AND every 16x16 tile of a W x H image is visited strictly once by the exact shader coordinate loop.
static inline int vkt_verify_visit_once(uint32_t w, uint32_t h, uint32_t taps,
                                        uint32_t groups, uint32_t iters, uint32_t y_tile_step,
                                        VktVisitStats * out) {
    VktVisitStats st;
    memset(&st, 0, sizeof st);
    if (out) *out = st;
    uint8_t lid_cnt[256];
    memset(lid_cnt, 0, sizeof lid_cnt);
    for (uint32_t lid = 0u; lid < 256u; ++lid) {
        const uint32_t lane = lid & 31u;
        const uint32_t sub = lid >> 5u;
        const uint32_t lx = (lane & 7u) + ((sub & 1u) << 3u);
        const uint32_t ly = ((lane >> 3u) & 3u) + ((sub >> 1u) << 2u);
        if (lx < 16u && ly < 16u) ++lid_cnt[ly * 16u + lx];
    }
    st.lid_bijection_ok = 1;
    for (uint32_t i = 0u; i < 256u; ++i) {
        if (lid_cnt[i] != 1u) st.lid_bijection_ok = 0;
    }

    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (taps < tiles_x) ? taps : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    const uint32_t strips_x_mask = strips_x - 1u;
    uint32_t strips_x_shift = 0u;
    for (uint32_t v = strips_x; v > 1u; v >>= 1u) ++strips_x_shift;
    const uint32_t tiles_y_mask = tiles_y - 1u;
    const uint32_t mask_x = w - 1u;
    const uint32_t mask_y = h - 1u;

    const uint64_t n_tiles = (uint64_t)tiles_x * (uint64_t)tiles_y;
    uint8_t * visits = (uint8_t *)calloc((size_t)n_tiles, 1u);
    if (!visits) return 0;

    st.total_items = n_tiles;
    for (uint32_t wgid = 0u; wgid < groups; ++wgid) {
        const uint32_t wg_tx = (wgid & strips_x_mask) * strip;
        const uint32_t wg_ty = (wgid >> strips_x_shift) & tiles_y_mask;
        const uint32_t bx0 = (wg_tx << 4u) & mask_x;
        uint32_t by = (wg_ty << 4u) & mask_y;
        for (uint32_t it = 0u; it < iters; ++it) {
            uint32_t ix = bx0;
            for (uint32_t k = 0u; k < taps; ++k) {
                const uint32_t tx = ix >> 4u;
                const uint32_t ty = by >> 4u;
                const uint64_t idx = (uint64_t)ty * (uint64_t)tiles_x + (uint64_t)tx;
                if (visits[idx] < 255u) ++visits[idx];
                ++st.total_visits;
                ix = (ix + 16u) & mask_x;
            }
            by = (by + (y_tile_step << 4u)) & mask_y;
        }
    }

    for (uint64_t i = 0u; i < n_tiles; ++i) {
        if (visits[i] == 0u) ++st.unvisited;
        else if (visits[i] > 1u) ++st.revisited;
        if (visits[i] > st.max_visits) st.max_visits = visits[i];
    }
    free(visits);
    if (out) *out = st;
    return st.lid_bijection_ok && st.unvisited == 0u && st.revisited == 0u &&
           st.max_visits == 1u && st.total_visits == n_tiles;
}

// Direct texel-by-texel verification executing every (wgid, lid, it, k) coordinate of the shader
// into a W*H byte visit map and confirming every texel (x, y) is visited strictly once.
static inline int vkt_verify_visit_once_texels(uint32_t w, uint32_t h, uint32_t taps,
                                               uint32_t groups, uint32_t iters, uint32_t y_tile_step,
                                               VktVisitStats * out) {
    VktVisitStats st;
    memset(&st, 0, sizeof st);
    if (out) *out = st;
    st.lid_bijection_ok = 1;
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (taps < tiles_x) ? taps : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    const uint32_t strips_x_mask = strips_x - 1u;
    uint32_t strips_x_shift = 0u;
    for (uint32_t v = strips_x; v > 1u; v >>= 1u) ++strips_x_shift;
    const uint32_t tiles_y_mask = tiles_y - 1u;
    const uint32_t mask_x = w - 1u;
    const uint32_t mask_y = h - 1u;

    const uint64_t n_texels = (uint64_t)w * (uint64_t)h;
    uint8_t * visits = (uint8_t *)calloc((size_t)n_texels, 1u);
    if (!visits) return 0;

    st.total_items = n_texels;
    uint32_t lx_lut[256], ly_lut[256];
    for (uint32_t lid = 0u; lid < 256u; ++lid) {
        const uint32_t lane = lid & 31u;
        const uint32_t sub = lid >> 5u;
        lx_lut[lid] = (lane & 7u) + ((sub & 1u) << 3u);
        ly_lut[lid] = ((lane >> 3u) & 3u) + ((sub >> 1u) << 2u);
    }

    for (uint32_t wgid = 0u; wgid < groups; ++wgid) {
        const uint32_t wg_tx = (wgid & strips_x_mask) * strip;
        const uint32_t wg_ty = (wgid >> strips_x_shift) & tiles_y_mask;
        for (uint32_t lid = 0u; lid < 256u; ++lid) {
            const uint32_t bx0 = ((wg_tx << 4u) + lx_lut[lid]) & mask_x;
            uint32_t by = ((wg_ty << 4u) + ly_lut[lid]) & mask_y;
            for (uint32_t it = 0u; it < iters; ++it) {
                uint32_t ix = bx0;
                for (uint32_t k = 0u; k < taps; ++k) {
                    const uint64_t idx = (uint64_t)by * (uint64_t)w + (uint64_t)ix;
                    if (visits[idx] < 255u) ++visits[idx];
                    ++st.total_visits;
                    ix = (ix + 16u) & mask_x;
                }
                by = (by + (y_tile_step << 4u)) & mask_y;
            }
        }
    }

    for (uint64_t i = 0u; i < n_texels; ++i) {
        if (visits[i] == 0u) ++st.unvisited;
        else if (visits[i] > 1u) ++st.revisited;
        if (visits[i] > st.max_visits) st.max_visits = visits[i];
    }
    free(visits);
    if (out) *out = st;
    return st.unvisited == 0u && st.revisited == 0u && st.max_visits == 1u &&
           st.total_visits == n_texels;
}

// Pack out_stride_wg push constant (backward-compatible with vk_async which passes plain out_wg=4 -> clk_slots=0):
//   bits [7:0]   = out_wg    (workgroups writing exactness results, <= 255)
//   bits [19:8]  = clk_slots (workgroup clock slots appended after exactness output, 0..4095)
//   bits [24:20] = clk_shift (log2 workgroup stride: WG writes slot = wgid >> clk_shift when low bits == 0)
static inline uint32_t vkt_pack_out_stride(uint32_t out_wg, uint32_t clk_slots, uint32_t clk_shift) {
    return (out_wg & 0xffu) | ((clk_slots & 0xfffu) << 8u) | ((clk_shift & 0x1fu) << 20u);
}

static inline uint32_t vkt_clk_stride_shift(uint32_t groups, uint32_t max_slots) {
    uint32_t sh = 0u;
    while (max_slots > 0u && (groups >> sh) > max_slots && sh < 31u) ++sh;
    return sh;
}

// Decode `n_slots` raw uvec4 (dc_lo, dc_hi, dr_lo, dr_hi) records from GL_ARB_shader_clock /
// GL_EXT_shader_realtime_clock (100 MHz realtime -> 10 ns/tick) written by one dispatch.
// Rejects spans shorter than VKT_MIN_CLK_RT_NS / VKT_MIN_CLK_CYCLES to guard against reordered
// or 1-tick clock reads. Returns arithmetic mean SCLK in MHz over valid samples (50..6000 MHz).
static inline double vkt_decode_shader_clk_mhz(const uint32_t * raw_uvec4, uint32_t n_slots,
                                               int * out_valid, double * out_max_rt_ns) {
    double sum_mhz = 0.0;
    double max_rt_ns = 0.0;
    int valid = 0;
    for (uint32_t i = 0u; i < n_slots; ++i) {
        const uint32_t * q = raw_uvec4 + i * 4u;
        const uint64_t dc = (uint64_t)q[0] | ((uint64_t)q[1] << 32u);
        const uint64_t dr = (uint64_t)q[2] | ((uint64_t)q[3] << 32u);
        if (dc < VKT_MIN_CLK_CYCLES || dr == 0ull) continue;
        const double rt_ns = (double)dr * VKT_RT_TICK_NS;
        if (rt_ns < VKT_MIN_CLK_RT_NS) continue;
        const double mhz = ((double)dc / rt_ns) * 1e3;
        if (mhz >= 50.0 && mhz <= 6000.0 && isfinite(mhz)) {
            sum_mhz += mhz;
            if (rt_ns > max_rt_ns) max_rt_ns = rt_ns;
            ++valid;
        }
    }
    if (out_valid) *out_valid = valid;
    if (out_max_rt_ns) *out_max_rt_ns = max_rt_ns;
    return (valid > 0) ? (sum_mhz / (double)valid) : 0.0;
}

// Verify ACO assembly places start shader/realtime clocks BEFORE the inner loop and end
// shader/realtime clocks strictly AFTER the loop AND after a post-loop buffer/global reload.
static inline int vkt_asm_clock_order_ok(const char * asm_text) {
    if (!asm_text) return 0;
    int nl = 0, cap = 256;
    const char ** lstart = (const char **)malloc(sizeof(const char *) * (size_t)cap);
    size_t * llen = (size_t *)malloc(sizeof(size_t) * (size_t)cap);
    if (!lstart || !llen) {
        free(lstart);
        free(llen);
        return 0;
    }
    const char * p = asm_text;
    while (*p) {
        const char * e = strchr(p, '\n');
        const size_t len = e ? (size_t)(e - p) : strlen(p);
        if (nl == cap) {
            cap *= 2;
            lstart = (const char **)realloc(lstart, sizeof(const char *) * (size_t)cap);
            llen = (size_t *)realloc(llen, sizeof(size_t) * (size_t)cap);
        }
        lstart[nl] = p;
        llen[nl] = len;
        ++nl;
        if (!e) break;
        p = e + 1;
    }

    int loop_a = -1, loop_b = -1;
    for (int i = 0; i < nl; ++i) {
        char buf[256];
        size_t n = llen[i] < sizeof(buf) - 1u ? llen[i] : sizeof(buf) - 1u;
        memcpy(buf, lstart[i], n);
        buf[n] = '\0';
        const char * s = buf;
        while (*s == ' ' || *s == '\t') ++s;
        if (strncmp(s, "s_cbranch", 9) != 0 && strncmp(s, "s_branch", 8) != 0) continue;
        const char * bb = strstr(s, "BB");
        if (!bb) continue;
        char lab[32];
        int k = 0;
        while (bb[k] && ((bb[k] >= 'A' && bb[k] <= 'Z') || (bb[k] >= 'a' && bb[k] <= 'z') ||
                         (bb[k] >= '0' && bb[k] <= '9') || bb[k] == '_') && k < 30) {
            lab[k] = bb[k];
            ++k;
        }
        lab[k] = ':';
        lab[k + 1] = '\0';
        for (int j = 0; j < i; ++j) {
            const char * t = lstart[j];
            size_t tl = llen[j];
            while (tl > 0u && (*t == ' ' || *t == '\t')) { ++t; --tl; }
            if (tl >= (size_t)(k + 1) && memcmp(t, lab, (size_t)(k + 1)) == 0) {
                if (loop_a < 0 || (i - j) > (loop_b - loop_a)) {
                    loop_a = j;
                    loop_b = i;
                }
            }
        }
    }

    int pre_cyc = 0, pre_rt = 0;
    int post_reload_line = -1;
    int post_cyc = 0, post_rt = 0;
    if (loop_a > 0 && loop_b > loop_a) {
        for (int i = 0; i < nl; ++i) {
            char buf[256];
            size_t n = llen[i] < sizeof(buf) - 1u ? llen[i] : sizeof(buf) - 1u;
            memcpy(buf, lstart[i], n);
            buf[n] = '\0';
            const int has_cyc = (strstr(buf, "SHADER_CYCLES") != NULL || strstr(buf, "s_memtime") != NULL);
            const int has_rt = (strstr(buf, "MSG_RTN_GET_REALTIME") != NULL || strstr(buf, "s_memrealtime") != NULL);
            // Only a vector memory load can be the accumulator reload; scalar loads do not count.
            const char *t = buf;
            while (*t == ' ' || *t == '\t') ++t;
            const int has_reload = (strncmp(t, "buffer_load", 11) == 0 || strncmp(t, "global_load", 11) == 0 ||
                                    strncmp(t, "flat_load", 9) == 0);
            if (i < loop_a) {
                if (has_cyc) ++pre_cyc;
                if (has_rt) ++pre_rt;
            } else if (i > loop_b) {
                if (post_reload_line < 0 && has_reload) post_reload_line = i;
                if (post_reload_line >= 0 && i > post_reload_line) {
                    if (has_cyc) ++post_cyc;
                    if (has_rt) ++post_rt;
                }
            }
        }
    }

    free(lstart);
    free(llen);
    return (loop_a > 0 && pre_cyc >= 1 && pre_rt >= 1 && post_reload_line > loop_b &&
            post_cyc >= 1 && post_rt >= 1) ? 1 : 0;
}

// Clock-sample gate: refuse clock-normalised numbers whenever fewer than VKT_MIN_CLK_SAMPLES (5)
// valid samples fell inside the timed window.
static inline int vkt_clk_gate_ok(int n_clk_samples, double sclk_mhz) {
    return (n_clk_samples >= VKT_MIN_CLK_SAMPLES) && (sclk_mhz > 0.0) && isfinite(sclk_mhz);
}

static inline void vkt_fmt_clk_norm(char * dst, size_t cap, int clk_ok, const char * fmt, double val) {
    if (!clk_ok || !isfinite(val)) {
        snprintf(dst, cap, "N/A");
    } else {
        snprintf(dst, cap, fmt, val);
    }
}

// Plan bounded timed-window dispatch count from a warmed dispatch duration (ns).
static inline int vkt_plan_timed_reps(double warm_gpu_ns, double target_ms, int min_reps, int max_reps) {
    if (min_reps < VKT_MIN_CLK_SAMPLES) min_reps = VKT_MIN_CLK_SAMPLES;
    if (max_reps < min_reps) max_reps = min_reps;
    if (!(warm_gpu_ns > 0.0) || !isfinite(warm_gpu_ns)) return min_reps;
    const double target_ns = target_ms * 1e6;
    int reps = (int)ceil(target_ns / warm_gpu_ns);
    if (reps < min_reps) reps = min_reps;
    if (reps > max_reps) reps = max_reps;
    return reps;
}

#endif
