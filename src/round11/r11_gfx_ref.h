// CPU reference for round 11 item 3 (graphics fixed blocks): everything the probe's controls and verdicts rest on, with no Vulkan.
// r11_gfx.c includes it for its expected values; r11_gfx_selftest.c proves each rule and each deliberate break.
#ifndef R11_GFX_REF_H
#define R11_GFX_REF_H
#include <math.h>
#include <stdint.h>
#include <string.h>

// ---- numbers from the plan and the study (cited in scripts/round11/expect/expect-3.md) --------------------------------------
#define R11G_BURST_MAX_MS 250.0     // PLAN-r11: bursts of at most 0.25 s
#define R11G_BUS_STOCK_GBS 640.0    // 256-bit GDDR6 at 20 Gbps (stock memory clock 1258 MHz)
#define R11G_BUS_OC_GBS 720.9       // the same bus at the requested 1408 MHz memory clock (22.5 Gbps), when the display does not pin it
#define R11G_ANCHOR_SLACK 1.02      // an uncompressed fill above bus * 1.02 is a probe fault, not a result

// ---- shared hash (rt_common.glsl uses the same lowbias32; gfx_common.glsl must keep these constants identical) -------------------
static inline uint32_t r11g_hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
// Per-pixel pseudorandom colour of the incompressible arm: x, y below 16384; same text as px_hash in gfx_common.glsl.
static inline uint32_t r11g_px_hash(uint32_t x, uint32_t y, uint32_t seed) { return r11g_hash32(((y << 14) | x) ^ seed); }

// ---- render-target formats ---------------------------------------------------------------------------------------------------
enum { R11G_R8, R11G_RG8, R11G_RGBA8, R11G_RGB10A2, R11G_R16F, R11G_RGBA16F, R11G_R32F, R11G_RGBA32F, R11G_D32, R11G_NFMT };
static const char * const r11g_fmt_name[R11G_NFMT] = {"R8", "RG8", "RGBA8", "RGB10A2", "R16F", "RGBA16F", "R32F", "RGBA32F", "D32"};
static const uint32_t r11g_fmt_bpp[R11G_NFMT] = {1, 2, 4, 4, 2, 8, 4, 16, 4};

static inline uint32_t r11g_unorm(double v, int bits) {
    const double m = (double) ((1u << bits) - 1u);
    const double r = floor(v * m + 0.5);
    return (uint32_t) (r < 0 ? 0 : r > m ? m : r);
}
// float -> half, round to nearest even, no NaN payload care.
static inline uint16_t r11g_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
    const int ebits = (int) ((x >> 23) & 0xffu);
    if (ebits == 0xff) return (uint16_t) (sign | 0x7c00u | (m ? 0x200u : 0u));
    const int e = ebits - 127 + 15;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        const uint32_t mm = m | 0x800000u;
        const int sh = 14 - e;
        uint32_t half = mm >> sh;
        const uint32_t rem = mm & ((1u << sh) - 1u), mid = 1u << (sh - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return (uint16_t) (sign | half);
    }
    uint32_t half = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return (uint16_t) (sign | half);
}
// The bytes one pixel of format `fmt` holds after drawing colour c (depth: c[0]); 16 bytes at most. RGB10A2 is A2B10G10R10_UNORM_PACK32.
static inline void r11g_px_expect(int fmt, const float c[4], uint8_t out[16]) {
    memset(out, 0, 16);
    uint16_t h;
    uint32_t u;
    switch (fmt) {
        case R11G_R8: out[0] = (uint8_t) r11g_unorm(c[0], 8); break;
        case R11G_RG8: out[0] = (uint8_t) r11g_unorm(c[0], 8), out[1] = (uint8_t) r11g_unorm(c[1], 8); break;
        case R11G_RGBA8:
            for (int i = 0; i < 4; ++i) out[i] = (uint8_t) r11g_unorm(c[i], 8);
            break;
        case R11G_RGB10A2:
            u = r11g_unorm(c[0], 10) | (r11g_unorm(c[1], 10) << 10) | (r11g_unorm(c[2], 10) << 20) | (r11g_unorm(c[3], 2) << 30);
            memcpy(out, &u, 4);
            break;
        case R11G_R16F: h = r11g_f32_to_f16(c[0]), memcpy(out, &h, 2); break;
        case R11G_RGBA16F:
            for (int i = 0; i < 4; ++i) h = r11g_f32_to_f16(c[i]), memcpy(out + 2 * i, &h, 2);
            break;
        case R11G_R32F: case R11G_D32: memcpy(out, &c[0], 4); break;
        case R11G_RGBA32F: memcpy(out, c, 16); break;
    }
}
// Colours of the fill control: which 0 = the two earlier draws, which 1 = the last draw that the read-back checks. The read-back colour must lie
// exactly on the format's UNORM grid (every channel * (2^bits - 1) an integer): a colour between two grid points makes the stored value depend on
// the hardware's rounding of the export (RGB10A2 at 0.2 * 1023 = 204.6 did not read back as the CPU's round-to-nearest, 512 of 512 pixels), which
// is a statement about rounding, not about the fill. Other formats keep the original colours (already on the grid, or float formats with a 1-ulp window).
static inline void r11g_ctrl_colour(int fmt, int which, float c[4]) {
    static const float old0[4] = {0.9f, 0.1f, 0.3f, 0.5f}, old1[4] = {0.2f, 0.4f, 0.6f, 0.8f};
    if (fmt == R11G_RGB10A2) {
        static const float g0[4] = {800.0f / 1023.0f, 100.0f / 1023.0f, 300.0f / 1023.0f, 1.0f / 3.0f};
        static const float g1[4] = {205.0f / 1023.0f, 409.0f / 1023.0f, 614.0f / 1023.0f, 2.0f / 3.0f};
        memcpy(c, which ? g1 : g0, 16);
    } else {
        memcpy(c, which ? old1 : old0, 16);
    }
}
// 1 when every channel of c sits within 1e-3 grid units of an integer on the UNORM grid of `fmt` (float and depth formats: always).
static inline int r11g_on_grid(int fmt, const float c[4]) {
    int bits[4] = {0, 0, 0, 0};
    switch (fmt) {
        case R11G_R8: bits[0] = 8; break;
        case R11G_RG8: bits[0] = bits[1] = 8; break;
        case R11G_RGBA8: bits[0] = bits[1] = bits[2] = bits[3] = 8; break;
        case R11G_RGB10A2: bits[0] = bits[1] = bits[2] = 10, bits[3] = 2; break;
        default: return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (!bits[i]) continue;
        const double v = (double) c[i] * (double) ((1u << bits[i]) - 1u);
        if (fabs(v - floor(v + 0.5)) > 1e-3) return 0;
    }
    return 1;
}
// Exact for UNORM; within one unit in the last place for the float formats (positive values: the bit patterns order like the numbers).
static inline int r11g_px_equal(int fmt, const uint8_t * got, const uint8_t * want) {
    const uint32_t bpp = r11g_fmt_bpp[fmt];
    if (fmt <= R11G_RGB10A2) return memcmp(got, want, bpp) == 0;
    if (fmt == R11G_R16F || fmt == R11G_RGBA16F)
        for (uint32_t i = 0; i < bpp / 2; ++i) {
            uint16_t a, b;
            memcpy(&a, got + 2 * i, 2), memcpy(&b, want + 2 * i, 2);
            if ((a > b ? a - b : b - a) > 1) return 0;
        }
    else
        for (uint32_t i = 0; i < bpp / 4; ++i) {
            uint32_t a, b;
            memcpy(&a, got + 4 * i, 4), memcpy(&b, want + 4 * i, 4);
            if ((a > b ? a - b : b - a) > 1) return 0;
        }
    return 1;
}

// Target shape for `bytes` of a format: width and height powers of two, both at most `maxdim`; 0 when it does not fit (R8 at 512 MiB).
static inline int r11g_target_dims(uint64_t bytes, uint32_t bpp, uint32_t maxdim, uint32_t * w, uint32_t * h) {
    const uint64_t px = bytes / bpp;
    int lg = 0;
    while ((1ull << lg) < px) ++lg;
    *w = 1u << ((lg + 1) / 2);
    *h = (uint32_t) (px / *w);
    return *w <= maxdim && *h <= maxdim && (uint64_t) *w * *h * bpp == bytes;
}
// Hardware-counted fact: an occlusion query over `passes` full-target passes counts exactly this many samples.
static inline uint64_t r11g_fill_expected_count(uint32_t passes, uint32_t w, uint32_t h, uint32_t samples) { return (uint64_t) passes * w * h * samples; }
static inline int r11g_count_ok(uint64_t got, uint64_t want) { return got == want; }

// ---- rasteriser: right triangles with vertex 0 at (cell + 77/256, cell + 77/256) and legs a256/256 pixels ---------------------------------
// With the 77/256 offset and legs in multiples of 1/256 no pixel centre can lie on an edge, so the covered count is exact and rule-free.
#define R11G_TRI_OFF256 77u
static inline uint64_t r11g_tri_cover(uint32_t a256) {
    uint64_t n = 0;
    for (uint32_t j = 0; 256u * j + 102u < a256; ++j)
        for (uint32_t i = 0; 256u * (i + j) + 102u < a256; ++i) ++n;
    return n;
}
static inline uint32_t r11g_tri_stride(uint32_t a256) { return (a256 + 255u) / 256u + 1u; }
static inline uint32_t r11g_tri_gridw(uint32_t a256, uint32_t target_px) { return target_px / r11g_tri_stride(a256); }
static inline uint64_t r11g_tri_per_pass(uint32_t a256, uint32_t target_px) {
    const uint64_t g = r11g_tri_gridw(a256, target_px);
    return g * g;
}
static inline uint64_t r11g_tri_expected_count(uint32_t a256, uint32_t target_px, uint32_t passes) {
    return r11g_tri_per_pass(a256, target_px) * r11g_tri_cover(a256) * passes;
}

// ---- textures ----------------------------------------------------------------------------------------------------------------
enum { R11T_R8, R11T_RGBA8, R11T_RGBA16F, R11T_RGBA32F, R11T_BC1, R11T_BC7, R11T_NFMT };
static const char * const r11t_name[R11T_NFMT] = {"R8", "RGBA8", "RGBA16F", "RGBA32F", "BC1", "BC7"};
static const uint32_t r11t_pattern[R11T_NFMT] = {0x80808080u, 0x80808080u, 0x38003800u, 0x3f000000u, 0, 0};  // non-BC: one repeated 32-bit word per texel row
// Bits per texel x 8 (bytes per texel x 8): BC1 4 bits per texel, BC7 8.
static inline uint64_t r11t_mip0_bytes(int f, uint32_t size) {
    const uint64_t px = (uint64_t) size * size;
    static const uint32_t bpp[R11T_NFMT] = {1, 4, 8, 16, 0, 0};
    if (f == R11T_BC1) return px / 2;
    if (f == R11T_BC7) return px;
    return px * bpp[f];
}
// The constant every texel decodes to, per channel (NAN: channel not defined by the format).
static inline void r11t_const(int f, double c[4]) {
    switch (f) {
        case R11T_R8: c[0] = 128.0 / 255, c[1] = c[2] = c[3] = NAN; break;
        case R11T_RGBA8: c[0] = c[1] = c[2] = c[3] = 128.0 / 255; break;
        case R11T_RGBA16F: case R11T_RGBA32F: c[0] = c[1] = c[2] = c[3] = 0.5; break;
        case R11T_BC1: c[0] = 132.0 / 255, c[1] = 130.0 / 255, c[2] = 132.0 / 255, c[3] = 1.0; break;  // colour 0x8410 in 5:6:5, replicated to 8 bits
        case R11T_BC7: c[0] = c[1] = c[2] = c[3] = 128.0 / 255; break;                               // mode 6, 7-bit endpoint 0x40, p-bit 0
    }
}
// One BC1 block of the constant: both endpoints 0x8410 (3-colour mode), indices 0.
static inline void r11t_bc1_block(uint8_t b[8]) {
    const uint8_t blk[8] = {0x10, 0x84, 0x10, 0x84, 0, 0, 0, 0};
    memcpy(b, blk, 8);
}
// One BC7 mode-6 block of the constant: bits LSB first: 7 mode bits (0b1000000), then R0 R1 G0 G1 B0 B1 A0 A1 at 7 bits each (all 0x40), p-bits 0 0, 63 index bits 0.
static inline void r11t_bc7_block(uint8_t b[16]) {
    memset(b, 0, 16);
    int pos = 0;
    uint32_t fields[9] = {0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40};
    for (int k = 0; k < 9; ++k)
        for (int i = 0; i < 7; ++i, ++pos)
            if ((fields[k] >> i) & 1u) b[pos >> 3] |= (uint8_t) (1u << (pos & 7));
}
// Reads a 7-bit field back out of a BC7 block (bit position in the LSB-first stream); used by the selftest as an independent decode of the layout.
static inline uint32_t r11t_bc7_field(const uint8_t b[16], int bitpos) {
    uint32_t v = 0;
    for (int i = 0; i < 7; ++i) v |= (uint32_t) ((b[(bitpos + i) >> 3] >> ((bitpos + i) & 7)) & 1u) << i;
    return v;
}
// The 16-texel-wide RGBA8 gradient (texel x holds 17 x / 255 = x / 15): the value bilinear and nearest sampling must return at u, clamp-to-edge.
static inline double r11t_grad_bilinear(double u) {
    double t = u * 16.0 - 0.5;
    t = t < 0 ? 0 : t > 15 ? 15 : t;
    return t / 15.0;
}
static inline double r11t_grad_nearest(double u) {
    int i = (int) floor(u * 16.0);
    i = i < 0 ? 0 : i > 15 ? 15 : i;
    return i / 15.0;
}
// Sampled value within tol (in 1/255 units) of want.
static inline int r11t_close(double got, double want, double tol255) { return fabs(got - want) * 255.0 <= tol255; }

// ---- compression switch ------------------------------------------------------------------------------------------------------
#define R11G_CC_DISABLED 4u  // VK_IMAGE_COMPRESSION_DISABLED_EXT
// "Setting applied": an image created with compression disabled must read back as disabled; an image that asked for nothing makes no claim.
static inline int r11g_comp_applied(int requested_disabled, uint32_t readback_flags) { return requested_disabled ? (readback_flags & R11G_CC_DISABLED) != 0 : 1; }
// The switch is real only when the disabled image reads back disabled AND an ordinary image does not (else the read-back says nothing).
static inline int r11g_comp_switch_verified(uint32_t disabled_rb, uint32_t default_rb) { return (disabled_rb & R11G_CC_DISABLED) && !(default_rb & R11G_CC_DISABLED); }
// Effect of compression on a rate: +1 compression on is faster, -1 slower, 0 within tol (fraction).
static inline int r11g_comp_effect(double rate_on, double rate_off, double tol) {
    if (rate_on > rate_off * (1.0 + tol)) return 1;
    if (rate_on < rate_off * (1.0 - tol)) return -1;
    return 0;
}

// ---- verdicts and ceilings ---------------------------------------------------------------------------------------------------
// The study's cache-level method: the same ROP count with the memory stopping it at different sizes. rate_small / rate_big (same format).
// 0 flat (not memory-stopped, ROP/shader bound), 1 mixed, 2 memory-limited (the big target is much slower than the small).
static inline int r11g_bound_class(double rate_small, double rate_big) {
    const double r = rate_big / rate_small;
    return r >= 0.9 ? 0 : r >= 0.7 ? 1 : 2;
}
static inline int r11g_anchor_ok(double write_gbs) { return write_gbs <= R11G_BUS_OC_GBS * R11G_ANCHOR_SLACK; }  // uncompressed 512 MiB fill
static inline int r11g_burst_ok(double ms) { return ms <= R11G_BURST_MAX_MS; }
// The 250 ms limit applies to every burst of a cell (calibration, warm re-aim and timed repeats); keep the longest one seen.
static inline double r11g_longest(double cur, double ms) { return ms > cur ? ms : cur; }
// A cell whose timed repeats differ by more than 10 % (min to max) is flagged SPREAD.
static inline int r11g_spread_flag(double mn, double mx) { return mx > mn * 1.10; }
// Raw attachment bytes of a fill burst: pixels x bytes per pixel x samples per pixel.
static inline double r11g_fill_bytes(double px, uint32_t bpp, int samples) { return px * bpp * (double) samples; }
static inline double r11g_ceil_gpix(double rops, double mhz) { return rops * mhz * 1e-3; }
static inline double r11g_ceil_gbs_write(double rate_gpix, uint32_t bpp) { return rate_gpix * bpp; }
static inline double r11g_ceil_gtri(double engines, double per_clock, double mhz) { return engines * per_clock * mhz * 1e-3; }
#endif
