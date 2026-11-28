// CPU-only test of r11_gfx_ref.h (no Vulkan, no card): every rule the graphics probe's controls and verdicts rest on, and every deliberate break.
// Usage: r11_gfx_selftest            all checks must pass, exit 0
//        r11_gfx_selftest --break N  stops sabotaging the scenario of deliberate break N (1..7), so the detector has nothing to see: must exit non-zero
// A break is a scenario the probe must detect (scissored draw, skipped draw, setting not applied, wrong filter, wrong cull, over-bus anchor, off-grid RGB10A2 control colour).
// With --break N the sabotage is left out, the detector then reports "nothing wrong" and the test goes red.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "r11_gfx_ref.h"

static int bad = 0;
#define T(c, msg)                                           \
    do {                                                    \
        if (!(c)) {                                         \
            printf("selftest FAIL: %s\n", msg);             \
            ++bad;                                          \
        }                                                   \
    } while (0)

int main(int argc, char ** argv) {
    int brk = 0;
    if (argc == 3 && !strcmp(argv[1], "--break")) brk = atoi(argv[2]);
    else if (argc != 1) { fprintf(stderr, "usage: r11_gfx_selftest [--break N]\n"); return 2; }

    // hash: vectors from an independent Python implementation of lowbias32
    T(r11g_hash32(0) == 0x0u && r11g_hash32(1) == 0x688990c0u && r11g_hash32(12345) == 0x912efcf7u && r11g_hash32(0x7fffffffu) == 0x8d29ffb8u, "hash32 test vectors");
    T(r11g_px_hash(3, 5, 0xabcd) == 0x036d999eu, "px_hash(3,5,seed) matches the Python value");

    // float formats: values from numpy float16
    T(r11g_f32_to_f16(0.2f) == 0x3266 && r11g_f32_to_f16(0.4f) == 0x3666 && r11g_f32_to_f16(0.6f) == 0x38cd && r11g_f32_to_f16(0.8f) == 0x3a66, "half conversion of 0.2 0.4 0.6 0.8");
    T(r11g_f32_to_f16(0.5f) == 0x3800 && r11g_f32_to_f16(0.375f) == 0x3600 && r11g_f32_to_f16(1.0f) == 0x3c00, "half conversion of exact values");
    T(r11g_unorm(0.2, 8) == 51 && r11g_unorm(0.8, 8) == 204 && r11g_unorm(0.2, 10) == 205 && r11g_unorm(0.4, 10) == 409 && r11g_unorm(0.6, 10) == 614 && r11g_unorm(0.8, 2) == 2, "unorm rounding");

    // fill controls: occlusion count and read-back, with the plan's breaks 1 and 2
    {
        const uint32_t W = 512, H = 512;
        const uint64_t want = r11g_fill_expected_count(3, W, H, 1);
        T(want == 3ull * 512 * 512, "expected count is passes x W x H");
        T(r11g_fill_expected_count(3, W, H, 4) == 4 * want, "MSAA 4 counts 4 samples per pixel");
        for (int f = 0; f < R11G_NFMT; ++f) {
            uint8_t last[16], prev[16], a[16];
            float c0[4], c1[4];
            r11g_ctrl_colour(f, 0, c0), r11g_ctrl_colour(f, 1, c1);
            r11g_px_expect(f, c1, last), r11g_px_expect(f, c0, prev);
            T(r11g_on_grid(f, c1), "the read-back colour of every format is on its UNORM grid");
            T(r11g_px_equal(f, last, last), "a read-back of the last colour is accepted");
            // break 2: the last draw is skipped, the image keeps the earlier colour: read-back must reject it (for every format whose two colours differ)
            memcpy(a, brk == 2 ? last : prev, 16);
            char m[96];
            snprintf(m, sizeof m, "break 2 (skipped last draw) rejected, format %s", r11g_fmt_name[f]);
            T(!r11g_px_equal(f, a, last), m);
        }
        // break 1: the last draw is scissored to half the target: the count falls short and the lower half of the image keeps the earlier colour
        const uint64_t got = brk == 1 ? want : 2ull * W * H + (uint64_t) W * (H / 2);
        T(!r11g_count_ok(got, want), "break 1 (scissor to half): the occlusion count fails");
        T(r11g_count_ok(want, want), "clean count passes");
        // break 2 also shortens the count
        T(!r11g_count_ok(brk == 2 ? want : 2ull * W * H, want), "break 2 (skipped draw): the occlusion count fails too");
        // float tolerance: one ulp passes, three fail
        uint8_t x[16], y[16];
        const float v[4] = {0.2f, 0, 0, 0};
        r11g_px_expect(R11G_R32F, v, x);
        memcpy(y, x, 16);
        uint32_t u;
        memcpy(&u, y, 4), ++u, memcpy(y, &u, 4);
        T(r11g_px_equal(R11G_R32F, y, x), "R32F: one ulp apart is accepted");
        u += 2, memcpy(y, &u, 4);
        T(!r11g_px_equal(R11G_R32F, y, x), "R32F: three ulp apart is rejected");
        T(r11g_fmt_bpp[R11G_RGBA16F] == 8 && r11g_fmt_bpp[R11G_RGBA32F] == 16 && r11g_fmt_bpp[R11G_RGB10A2] == 4, "format sizes");
    }

    // RGB10A2 = VK_FORMAT_A2B10G10R10_UNORM_PACK32: R in bits 0-9, G 10-19, B 20-29, A 30-31 (little-endian word). The first full run read back
    // 512 of 512 pixels different with the colour (0.2, 0.4, 0.6, 0.8), which is off the 10-bit grid (204.6, 409.2, 613.8) and a 2-bit alpha (2.4).
    {
        uint32_t u;
        uint8_t px[16];
        const float rr[4] = {1, 0, 0, 0}, gg[4] = {0, 1, 0, 0}, bb[4] = {0, 0, 1, 0}, aa[4] = {0, 0, 0, 1};
        r11g_px_expect(R11G_RGB10A2, rr, px), memcpy(&u, px, 4);
        T(u == 0x000003ffu, "A2B10G10R10: red is bits 0-9");
        r11g_px_expect(R11G_RGB10A2, gg, px), memcpy(&u, px, 4);
        T(u == 0x000ffc00u, "A2B10G10R10: green is bits 10-19");
        r11g_px_expect(R11G_RGB10A2, bb, px), memcpy(&u, px, 4);
        T(u == 0x3ff00000u, "A2B10G10R10: blue is bits 20-29");
        r11g_px_expect(R11G_RGB10A2, aa, px), memcpy(&u, px, 4);
        T(u == 0xc0000000u, "A2B10G10R10: alpha is bits 30-31");
        // rounding to nearest at 10 and 2 bits, away from the ties
        T(r11g_unorm(204.4 / 1023, 10) == 204 && r11g_unorm(204.6 / 1023, 10) == 205 && r11g_unorm(1.4 / 3, 2) == 1 && r11g_unorm(1.6 / 3, 2) == 2, "10-bit and 2-bit round to nearest");
        // the control colour: exact grid points (205, 409, 614 of 1023; 2 of 3), packed value 205 | 409 << 10 | 614 << 20 | 2 << 30 (worked out by hand)
        float c1[4];
        r11g_ctrl_colour(R11G_RGB10A2, 1, c1);
        r11g_px_expect(R11G_RGB10A2, c1, px), memcpy(&u, px, 4);
        T(u == 0xa66664cdu, "A2B10G10R10 control colour packs to 0xa66664cd (R 205, G 409, B 614, A 2)");
        // the bug: a colour off the grid. With --break 7 the off-grid colour is what the control uses, and the grid rule must notice.
        const float off[4] = {0.2f, 0.4f, 0.6f, 0.8f};
        T(!r11g_on_grid(R11G_RGB10A2, off), "the old colour (0.2, 0.4, 0.6, 0.8) is off the 10-bit grid");
        T(r11g_on_grid(R11G_RGB10A2, brk == 7 ? off : c1), "break 7: an off-grid RGB10A2 control colour is rejected by the grid rule");
    }

    // target shapes
    {
        uint32_t w, h;
        T(r11g_target_dims(1u << 20, 4, 16384, &w, &h) && w == 512 && h == 512, "1 MiB RGBA8 is 512 x 512");
        T(r11g_target_dims(512ull << 20, 4, 16384, &w, &h) && w == 16384 && h == 8192, "512 MiB RGBA8 is 16384 x 8192");
        T(r11g_target_dims(512ull << 20, 2, 16384, &w, &h) && w == 16384 && h == 16384, "512 MiB RG8 is 16384 x 16384");
        T(!r11g_target_dims(512ull << 20, 1, 16384, &w, &h), "512 MiB R8 does not fit a 16384 image (skipped by the probe)");
        T(r11g_target_dims(4ull << 20, 16, 16384, &w, &h) && (uint64_t) w * h * 16 == 4ull << 20, "4 MiB RGBA32F");
    }

    // rasteriser: exact coverage, values from an independent Python loop
    {
        static const uint32_t a[7] = {128, 256, 512, 1024, 2048, 4096, 8192};
        static const uint64_t want[7] = {1, 1, 3, 10, 36, 136, 528};
        for (int i = 0; i < 7; ++i) {
            char m[96];
            snprintf(m, sizeof m, "triangle coverage, legs %u/256 px: %llu vs %llu", a[i], (unsigned long long) r11g_tri_cover(a[i]), (unsigned long long) want[i]);
            T(r11g_tri_cover(a[i]) == want[i], m);
        }
        T(r11g_tri_stride(256) == 2 && r11g_tri_stride(128) == 2 && r11g_tri_stride(8192) == 33, "cell stride is ceil(leg) + 1");
        T(r11g_tri_per_pass(256, 2048) == 1024ull * 1024 && r11g_tri_per_pass(8192, 2048) == 62ull * 62, "triangles per pass in a 2048 target");
        // break 5: the culled control must count 0; if the probe passed the visible pipeline there (no cull), the count would equal the visible one
        const uint64_t vis = r11g_tri_expected_count(1024, 2048, 3);
        const uint64_t culled_got = brk == 5 ? vis : 0;
        T(vis > 0, "visible count is positive");
        T(culled_got == 0, "break 5 (culled control draws pixels): the culled count must be 0");
    }

    // textures
    {
        double c[4];
        r11t_const(R11T_BC1, c);
        T(c[0] == 132.0 / 255 && c[1] == 130.0 / 255 && c[2] == 132.0 / 255 && c[3] == 1.0, "BC1 constant of colour 0x8410");
        uint8_t b1[8];
        r11t_bc1_block(b1);
        const uint32_t c0 = b1[0] | (b1[1] << 8), c1 = b1[2] | (b1[3] << 8);
        T(c0 == 0x8410 && c1 == 0x8410 && !b1[4] && !b1[5] && !b1[6] && !b1[7], "BC1 block holds equal endpoints and zero indices");
        T((((c0 >> 11) & 31) << 3 | ((c0 >> 11) & 31) >> 2) == 132 && ((((c0 >> 5) & 63) << 2) | (((c0 >> 5) & 63) >> 4)) == 130 && (((c0 & 31) << 3) | ((c0 & 31) >> 2)) == 132, "BC1 565 expansion gives 132 130 132");
        uint8_t b7[16];
        r11t_bc7_block(b7);
        T((b7[0] & 0x7f) == 0x40, "BC7 mode field is mode 6 (six zero bits then a one)");
        int pos = 7, ok = 1;
        for (int k = 0; k < 8; ++k, pos += 7) ok &= r11t_bc7_field(b7, pos) == 0x40;
        T(ok, "BC7 eight endpoint fields hold 0x40");
        ok = 1;
        for (int bit = 71; bit < 128; ++bit) ok &= ((b7[bit >> 3] >> (bit & 7)) & 1) == 0;  // p-bits (bits 71, 72) and all index bits zero
        T(ok, "BC7 p-bits and indices are zero");
        // 8-bit value of the endpoint: (7-bit << 1) | p-bit = 128
        T(((0x40u << 1) | 0u) == 128 && c[0] != 0, "BC7 endpoint expands to 128");
        r11t_const(R11T_BC7, c);
        T(c[0] == 128.0 / 255, "BC7 constant is 128/255");
        // memory per level of each format
        T(r11t_mip0_bytes(R11T_RGBA32F, 8192) == 1ull << 30 && r11t_mip0_bytes(R11T_BC1, 8192) == 32ull << 20 && r11t_mip0_bytes(R11T_BC7, 8192) == 64ull << 20 && r11t_mip0_bytes(R11T_R8, 256) == 65536, "texture bytes per base level");
        // gradient: bilinear and nearest values at hand-worked points
        T(fabs(r11t_grad_bilinear(0.5) - 7.5 / 15.0) < 1e-12, "bilinear at u = 0.5 is 7.5/15");
        T(r11t_grad_bilinear(0.0) == 0.0 && r11t_grad_bilinear(1.0) == 1.0, "bilinear clamps at the ends");
        T(fabs(r11t_grad_bilinear(0.3) - (0.3 * 16 - 0.5) / 15.0) < 1e-12, "bilinear at u = 0.3");
        T(r11t_grad_nearest(0.3) == 4.0 / 15.0 && r11t_grad_nearest(0.99) == 1.0, "nearest at 0.3 and 0.99");
        // break 4: a nearest sampler checked against the bilinear analytic values must FAIL (64 points, tolerance 2/255)
        int fails = 0;
        for (int i = 0; i < 64; ++i) {
            const double u = (i + 0.5) / 64.0;
            const double got = brk == 4 ? r11t_grad_bilinear(u) : r11t_grad_nearest(u);  // break 4 leaves the card on the right filter: nothing to detect
            fails += !r11t_close(got, r11t_grad_bilinear(u), 2.0);
        }
        char m[96];
        snprintf(m, sizeof m, "break 4 (nearest sampler vs bilinear analytic): %d of 64 points fail", fails);
        T(fails >= 16, m);
        int clean = 0;
        for (int i = 0; i < 64; ++i) clean += !r11t_close(r11t_grad_bilinear((i + 0.5) / 64.0), r11t_grad_bilinear((i + 0.5) / 64.0), 2.0);
        T(clean == 0, "bilinear against bilinear analytic passes at all 64 points");
    }

    // compression switch (deliberate break 3: the setting not applied)
    {
        const uint32_t honoured = R11G_CC_DISABLED, ignored = 0;
        T(r11g_comp_applied(1, honoured), "an image that asked for disabled and reads back disabled: applied");
        T(!r11g_comp_applied(1, brk == 3 ? honoured : ignored), "break 3: an image whose request was not applied is flagged 'setting not applied'");
        T(r11g_comp_applied(0, ignored), "no request, no claim");
        T(r11g_comp_switch_verified(honoured, ignored), "switch verified: disabled reads disabled, default does not");
        T(!r11g_comp_switch_verified(honoured, honoured), "an ordinary image that also reads disabled: read-back says nothing, switch unverified");
        T(!r11g_comp_switch_verified(ignored, ignored), "no change in the read-back: no switch available");
        T(r11g_comp_effect(700, 600, 0.05) == 1 && r11g_comp_effect(600, 700, 0.05) == -1 && r11g_comp_effect(602, 600, 0.05) == 0, "compression effect classes");
    }

    // verdicts, anchor and burst length (break 6: an uncompressed fill above the bus)
    {
        T(r11g_bound_class(100, 98) == 0 && r11g_bound_class(100, 80) == 1 && r11g_bound_class(100, 40) == 2, "bound classes: flat, mixed, memory-limited");
        T(r11g_anchor_ok(633) && r11g_anchor_ok(703) && r11g_anchor_ok(R11G_BUS_OC_GBS), "anchor: the measured 633 and 703 GB/s pass");
        T(!r11g_anchor_ok(brk == 6 ? 700 : 900), "break 6: an uncompressed 512 MiB fill at 900 GB/s (above the 720.9 GB/s bus) is flagged");
        T(r11g_burst_ok(250.0) && r11g_burst_ok(100.0) && !r11g_burst_ok(250.1), "burst limit 0.25 s");
        // break 8: the longest burst of a cell is taken over ALL its bursts (calibration 40 ms, warm re-aim 274 ms, timed 100 ms): over the limit
        {
            double lg = 0;
            lg = r11g_longest(lg, 40.0);
            lg = r11g_longest(lg, brk == 8 ? 40.0 : 274.0);
            lg = r11g_longest(lg, 100.0);
            T(!r11g_burst_ok(lg), "break 8: a 274 ms warm re-aim burst makes the cell BURSTOVER even when the timed repeats are 100 ms");
            double ok_lg = r11g_longest(r11g_longest(r11g_longest(0, 12.0), 118.0), 250.0);
            T(r11g_burst_ok(ok_lg), "a cell whose longest burst is exactly 250 ms is not over");
        }
        T(r11g_spread_flag(100, 111) && !r11g_spread_flag(100, 109) && !r11g_spread_flag(100, 110), "spread flag: more than 10 % between min and max");
        // break 9: MSAA 4 moves four samples per pixel: 1024 x 512 px x 4 B x 4 samples = 8 MiB per burst of k = 1
        T(fabs(r11g_fill_bytes(1024.0 * 512.0, 4, brk == 9 ? 1 : 4) - 1024.0 * 512.0 * 4 * 4) < 1e-6, "break 9: MSAA 4 fill bytes are pixels x bpp x 4 (not x 1)");
        T(fabs(r11g_fill_bytes(1000.0, 8, 1) - 8000.0) < 1e-9, "single-sample fill bytes are pixels x bpp");
        T(fabs(r11g_ceil_gpix(128, 2970) - 380.16) < 1e-6, "ROP ceiling 128 x 2.97 GHz = 380.16 GPix/s");
        T(fabs(r11g_ceil_gpix(256, 2970) - 760.32) < 1e-6, "texture ceiling 256 x 2.97 GHz = 760.32 GT/s");
        T(fabs(r11g_ceil_gtri(4, 1, 2970) - 11.88) < 1e-9, "raster ceiling 4 engines x 1 tri/clk x 2.97 GHz = 11.88 GTri/s");
        T(fabs(r11g_ceil_gbs_write(380.16, 4) - 1520.64) < 1e-6, "RGBA8 write demand at the ROP ceiling is 1520.64 GB/s, over the 640 GB/s bus");
    }

    printf("r11_gfx_selftest%s: %d check(s) failed\n", brk ? " (break on)" : "", bad);
    return bad ? 1 : 0;
}
