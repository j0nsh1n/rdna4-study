// CPU-only self-test for src/round13/vk_tex_ref.h (no Vulkan, no GPU).
// Usage: vk_tex_selftest              all checks must pass (exit 0)
//        vk_tex_selftest --break N    deliberate break N (1..12) must exit non-zero
#include "vk_tex_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_bad = 0;

#define CHK(cond, msg)                                         \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("vk_tex_selftest FAIL: %s\n", (msg));       \
            ++g_bad;                                           \
        }                                                      \
    } while (0)

int main(int argc, char ** argv) {
    int brk = 0;
    if (argc == 3 && strcmp(argv[1], "--break") == 0) {
        brk = atoi(argv[2]);
        if (brk < 1 || brk > 12) {
            fprintf(stderr, "break index must be 1..12\n");
            return 2;
        }
    } else if (argc != 1) {
        fprintf(stderr, "usage: vk_tex_selftest [--break 1..12]\n");
        return 2;
    }

    // 1. All 6 formats x 4 cache levels produce valid power-of-two dims <= 16384 matching exact bytes.
    for (int f = 0; f < VKT_FMT_COUNT; ++f) {
        for (int l = 0; l < VKT_LVL_COUNT; ++l) {
            uint32_t w = 0, h = 0;
            int ok = vkt_level_dims(f, l, &w, &h);
            CHK(ok && (w & (w - 1u)) == 0u && (h & (h - 1u)) == 0u, "power-of-two level dims");
            CHK((uint64_t)w * h * vkt_fmt_bpp[f] == vkt_level_bytes(f, l), "exact footprint bytes");
        }
    }

    // 2. FP16 and B10G11R11 encoding exactness for every generated texel value.
    //    Also hand-checked B10G11R11 packing: R=1.0 (0x3c0), G=1.5 (0x3e0), B=1.25 (0x1e8).
    {
        const uint32_t p = vkt_pack_b10g11r11(1.0f, 1.5f, 1.25f);
        const uint32_t want = 0x3c0u | (0x3e0u << 11) | (0x1e8u << 22);
        CHK(p == (brk == 1 ? (want ^ 1u) : want), "B10G11R11 hand-computed bit packing");
    }
    for (int f = 0; f < VKT_FMT_COUNT; ++f) {
        int bad_rt = 0, bad_bilin = 0;
        for (uint32_t y = 0; y < 8u; ++y) {
            for (uint32_t x = 0; x < 8u; ++x) {
                float c[4];
                vkt_texel_rgba(f, 1, x, y, c);
                for (int k = 0; k < 4; ++k) {
                    if (vkt_f16_to_f32(vkt_f32_to_f16(c[k])) != c[k]) ++bad_rt;
                }
                float b[4];
                vkt_eval_sample(f, VKT_OP_LOD, 1, 64u, 64u, x, y, b);
                for (int k = 0; k < 4; ++k) {
                    const float rt = vkt_f16_to_f32(vkt_f32_to_f16(b[k]));
                    if (rt != (brk == 2 ? (b[k] + 0.001f) : b[k])) ++bad_bilin;
                }
            }
        }
        CHK(bad_rt == 0, "exact_mode texel round-trips bit-exact through FP16");
        CHK(bad_bilin == 0, "exact bilinear result round-trips bit-exact through FP16");
    }

    // 3. Coordinate-shift break test: shifting x by +1 must change accumulated result on all formats & ops.
    for (int f = 0; f < VKT_FMT_COUNT; ++f) {
        for (int op = 0; op < VKT_OP_COUNT; ++op) {
            int diff = 0;
            for (uint32_t lid = 0; lid < 16u; ++lid) {
                float good[4], shifted[4];
                vkt_expected_thread(f, op, 1, 64u, 64u, 0u, lid, 8u, 1u, 0u, good);
                vkt_expected_thread(f, op, 1, 64u, 64u, 0u, lid, 8u, 1u, brk == 3 ? 0u : 1u, shifted);
                for (int c = 0; c < 4; ++c) {
                    if (fabsf(good[c] - shifted[c]) > 1e-6f) diff = 1;
                }
            }
            CHK(diff, "wrong-coordinate break changes expected output");
        }
    }

    // 4. Verify exact FP32 accumulation over 4096 taps (256 iters x 16 taps) matches double precision bit-for-bit.
    {
        float acc_f[4];
        vkt_expected_thread(VKT_FMT_RGBA16_SFLOAT, VKT_OP_LOD, 1, 64u, 32u, 3u, 19u, 256u, 1u, 0u, acc_f);
        double acc_d[4] = {0, 0, 0, 0};
        const uint32_t lane = 19u & 31u, sub = 19u >> 5u;
        const uint32_t lx = (lane & 7u) + ((sub & 1u) << 3u);
        const uint32_t ly = ((lane >> 3u) & 3u) + ((sub >> 1u) << 2u);
        const uint32_t bx0 = ((3u << 4u) + lx) & 63u;
        uint32_t by = ly & 31u;
        for (uint32_t it = 0; it < 256u; ++it) {
            uint32_t ix = bx0;
            for (uint32_t k = 0; k < 16u; ++k) {
                float s[4];
                vkt_eval_sample(VKT_FMT_RGBA16_SFLOAT, VKT_OP_LOD, 1, 64u, 32u, ix, by, s);
                for (int c = 0; c < 4; ++c) acc_d[c] += (double)s[c];
                ix = (ix + 16u) & 63u;
            }
            by = (by + 16u) & 31u;
        }
        for (int c = 0; c < 4; ++c) {
            const float want = (float)acc_d[c] + (brk == 4 ? 0.001f : 0.0f);
            CHK(acc_f[c] == want, "4096-tap FP32 bilinear accumulation is bit-exact against double");
        }
    }

    // 5. Oscillating FP16/FP32 FMA reference + metric helpers.
    {
        float x = 0.25f;
        const float add_k = 1.75f;
        for (int s = 0; s < 1001; ++s) {
            x = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(x, -1.0f, add_k)));
        }
        const float want = vkt_osc_fma_ref(0.25f, add_k, brk == 5 ? 1000ull : 1001ull);
        CHK(x == want, "oscillating FP16 FMA matches closed-form period-2 reference");

        // 256 TMUs at 3.0 GHz = 768 GSamp/s -> 4.0 samp/clk/CU; at 4 B/texel = 3072 GB/s.
        const double spc = vkt_samp_per_clk_per_cu(768000000ull, 1e6, 3000.0);
        const double gbs = vkt_gbs(768000000ull, 4u, 1e6);
        CHK(fabs(spc - 4.0) < 1e-9, "samp/clk/CU formula: 768 GSamp/s @ 3 GHz = 4.0/clk/CU");
        CHK(fabs(gbs - 3072.0) < 1e-9, "GB/s formula: 768M x 4B in 1ms = 3072 GB/s");
    }

    // 6. Probe C 5-chain oscillating FMA parity check: for every N in {2,4,8,16,32,64} with odd iters=31,
    //    the final 5-chain sum must differ from the 0-FMA initial sum and match iterative FP16 evaluation bit-for-bit.
    {
        static const uint32_t ns[6] = {2u, 4u, 8u, 16u, 32u, 64u};
        static const float init_lo[5] = {0.25f, 0.375f, 0.5f, 0.625f, 0.75f};
        static const float init_hi[5] = {0.5f, 0.625f, 0.75f, 0.875f, 1.0f};
        const float init_sum_lo = 0.25f + 0.375f + 0.5f + 0.625f + 0.75f;
        for (int ni = 0; ni < 6; ++ni) {
            const uint32_t n = ns[ni];
            float ak_lo = 0.0f, ak_hi = 0.0f;
            vkt_osc_lane_k(13u, &ak_lo, &ak_hi);
            float a_lo[5], a_hi[5];
            memcpy(a_lo, init_lo, sizeof a_lo);
            memcpy(a_hi, init_hi, sizeof a_hi);
            for (uint32_t it = 0; it < 31u; ++it) {
                for (uint32_t s = 0; s < 4u * n; ++s) {
                    uint32_t c = s % 5u;
                    a_lo[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_lo[c], -1.0f, ak_lo)));
                    a_hi[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_hi[c], -1.0f, ak_hi)));
                }
            }
            float sim_lo = 0.0f, sim_hi = 0.0f;
            for (int c = 0; c < 5; ++c) { sim_lo += a_lo[c]; sim_hi += a_hi[c]; }
            uint64_t per_iter[5] = {0, 0, 0, 0, 0};
            for (uint32_t s = 0; s < 4u * n; ++s) ++per_iter[s % 5u];
            float ref_lo = 0.0f, ref_hi = 0.0f;
            for (int c = 0; c < 5; ++c) {
                ref_lo += vkt_osc_fma_ref(init_lo[c], ak_lo, per_iter[c] * 31ull);
                ref_hi += vkt_osc_fma_ref(init_hi[c], ak_hi, per_iter[c] * 31ull);
            }
            if (brk == 6 && ni == 0) ref_lo = init_sum_lo;
            CHK(sim_lo == ref_lo && sim_hi == ref_hi, "Probe C 5-chain FP16 oscillation matches closed form");
            CHK(fabsf(ref_lo - init_sum_lo) > 1e-4f, "Probe C 5-chain odd-iters sum differs from 0-FMA init");
        }
        // Also verify 4-tap texture helper matches 1-iter manual sum.
        float t4[4], manual[4] = {0, 0, 0, 0};
        vkt_expected_thread_4tap(VKT_FMT_RGBA16_SFLOAT, 1, 64u, 32u, 0u, 0u, 1u, 1u, 0u, t4);
        for (uint32_t k = 0; k < 4u; ++k) {
            float s[4];
            vkt_eval_sample(VKT_FMT_RGBA16_SFLOAT, VKT_OP_LOD, 1, 64u, 32u, k * 16u, 0u, s);
            for (int c = 0; c < 4; ++c) manual[c] += s[c];
        }
        for (int c = 0; c < 4; ++c) CHK(t4[c] == manual[c], "Probe C 4-tap expected helper matches manual sum");

        // 7. Strict DRAM visit-once proof (tile-level across all formats + direct texel-level
        //    on full 512 MiB textures, plus negative controls rejecting old Slot 2 wrapping bugs).
        for (int f = 0; f < VKT_FMT_COUNT; ++f) {
            uint32_t w = 0u, h = 0u, groups = 0u, iters = 0u, y_step = 0u;
            CHK(vkt_level_dims(f, VKT_LVL_DRAM, &w, &h), "Probe B DRAM dims valid");
            CHK(vkt_dram_schedule(f, 16u, &groups, &iters, &y_step), "Probe B DRAM schedule valid");
            const uint64_t bytes = (uint64_t)w * (uint64_t)h * (uint64_t)vkt_fmt_bpp[f];
            const uint64_t min_bytes = (vkt_fmt_bpp[f] == 1u) ? (256ull << 20) : (512ull << 20);
            CHK(bytes >= min_bytes, "Probe B DRAM footprint >= 512 MiB (256 MiB max-2D for R8)");
            if (brk == 7 && f == VKT_FMT_RGBA8_UNORM) iters *= 2u;  // deliberate strip-wrap break
            VktVisitStats vst;
            const int once = vkt_verify_visit_once(w, h, 16u, groups, iters, y_step, &vst);
            CHK(once && vst.unvisited == 0u && vst.revisited == 0u && vst.max_visits == 1u,
                "Probe B DRAM visits every 16x16 tile strictly once (zero reuse)");
        }
        for (int fi = 0; fi < 2; ++fi) {
            const int f = (fi == 0) ? VKT_FMT_RGBA16_SFLOAT : VKT_FMT_RGBA8_UNORM;
            uint32_t w = 0u, h = 0u, groups = 0u, iters = 0u, y_step = 0u;
            CHK(vkt_level_dims(f, VKT_LVL_DRAM, &w, &h), "Probe C DRAM dims valid");
            CHK(vkt_dram_schedule(f, 4u, &groups, &iters, &y_step), "Probe C DRAM schedule valid");
            CHK((uint64_t)w * (uint64_t)h * (uint64_t)vkt_fmt_bpp[f] >= (512ull << 20),
                "Probe C DRAM footprint >= 512 MiB");
            VktVisitStats vst;
            const int once = vkt_verify_visit_once(w, h, 4u, groups, iters, y_step, &vst);
            CHK(once && vst.unvisited == 0u && vst.revisited == 0u && vst.max_visits == 1u,
                "Probe C DRAM visits every 16x16 tile strictly once (zero reuse)");
        }
        // Full texel-by-texel visit-once check on full 512 MiB DRAM textures (Probe B RGBA32F
        // 8192x4096 = 33.5M texels, Probe C RGBA16F 8192x8192 = 67.1M texels).
        {
            uint32_t w = 0u, h = 0u, groups = 0u, iters = 0u, y_step = 0u;
            vkt_level_dims(VKT_FMT_RGBA32_SFLOAT, VKT_LVL_DRAM, &w, &h);
            vkt_dram_schedule(VKT_FMT_RGBA32_SFLOAT, 16u, &groups, &iters, &y_step);
            VktVisitStats tv;
            CHK(vkt_verify_visit_once_texels(w, h, 16u, groups, iters, y_step, &tv),
                "Probe B RGBA32F 512 MiB direct texel-level visit-once proof");

            vkt_level_dims(VKT_FMT_RGBA16_SFLOAT, VKT_LVL_DRAM, &w, &h);
            vkt_dram_schedule(VKT_FMT_RGBA16_SFLOAT, 4u, &groups, &iters, &y_step);
            CHK(vkt_verify_visit_once_texels(w, h, 4u, groups, iters, y_step, &tv),
                "Probe C RGBA16F 512 MiB direct texel-level visit-once proof");
        }
        // Negative controls: old Slot 2 wrapping DRAM configs MUST fail visit-once check.
        {
            VktVisitStats bad1 = {0}, bad2 = {0};
            const int old_rgba8_ok = vkt_verify_visit_once(8192u, 8192u, 16u, 4096u, 16u, 128u, &bad1);
            const int old_rgba32f_ok = vkt_verify_visit_once(4096u, 4096u, 16u, 4096u, 4u, 1u, &bad2);
            CHK(!old_rgba8_ok && bad1.revisited > 0u && bad1.max_visits == 4u,
                "visit-once checker catches old Slot 2 RGBA8 4x strip wrap");
            CHK(!old_rgba32f_ok && bad2.revisited > 0u && bad2.max_visits == 4u,
                "visit-once checker catches old Slot 2 RGBA32F 4x band overlap");
        }
    }

    // 8. In-shader clock (GL_ARB_shader_clock + GL_EXT_shader_realtime_clock) decoder check.
    {
        uint32_t raw[8 * 4];
        memset(raw, 0, sizeof raw);
        // Slot 0: 3200 MHz (dc = 32,000,000 cycles, dr = 1,000,000 ticks @ 100 MHz = 10 ms)
        // Slot 1: 3000 MHz using 64-bit hi-word test (dc = 4,500,000,000 cycles, dr = 150,000,000 ticks)
        // Slot 2: zero (unwritten slot, must be skipped)
        // Slot 3: 3400 MHz (dc = 34,000,000, dr = 1,000,000)
        // Mean of slots 0, 1, 3 = (3200 + 3000 + 3400) / 3 = 3200.0 MHz.
        const uint64_t dc0 = 32000000ull, dr0 = 1000000ull;
        const uint64_t dc1 = 4500000000ull, dr1 = 150000000ull;
        const uint64_t dc3 = 34000000ull, dr3 = (brk == 8) ? 2000000ull : 1000000ull;
        raw[0] = (uint32_t)dc0; raw[1] = (uint32_t)(dc0 >> 32u);
        raw[2] = (uint32_t)dr0; raw[3] = (uint32_t)(dr0 >> 32u);
        raw[4] = (uint32_t)dc1; raw[5] = (uint32_t)(dc1 >> 32u);
        raw[6] = (uint32_t)dr1; raw[7] = (uint32_t)(dr1 >> 32u);
        raw[12] = (uint32_t)dc3; raw[13] = (uint32_t)(dc3 >> 32u);
        raw[14] = (uint32_t)dr3; raw[15] = (uint32_t)(dr3 >> 32u);
        int n_valid = 0;
        double max_rt_ns = 0.0;
        const double avg_mhz = vkt_decode_shader_clk_mhz(raw, 8u, &n_valid, &max_rt_ns);
        CHK(n_valid == 3, "shader-clock decoder counts only valid non-zero slots");
        CHK(fabs(avg_mhz - 3200.0) < 1e-6, "shader-clock decoder computes exact 64-bit average MHz");
        CHK(fabs(max_rt_ns - 1.5e9) < 1.0, "shader-clock decoder tracks max realtime span");
    }

    // 9. Refusal gate when fewer than 5 clock samples fell inside the timed window.
    {
        for (int n = 0; n < VKT_MIN_CLK_SAMPLES; ++n) {
            const int ok = (brk == 9 && n == 4) ? 1 : vkt_clk_gate_ok(n, 3200.0);
            char buf[32];
            vkt_fmt_clk_norm(buf, sizeof buf, ok, "%.3f", 3.812);
            CHK(!ok && strcmp(buf, "N/A") == 0, "refuse clock-normalised value when clk_n < 5");
        }
        CHK(!vkt_clk_gate_ok(10, 0.0) && !vkt_clk_gate_ok(10, NAN), "refuse non-positive/NaN clock");
        const int ok5 = vkt_clk_gate_ok(VKT_MIN_CLK_SAMPLES, 3200.0);
        char buf5[32];
        vkt_fmt_clk_norm(buf5, sizeof buf5, ok5, "%.3f", 3.812);
        CHK(ok5 && strcmp(buf5, "3.812") == 0, "allow clock-normalised value when clk_n >= 5");
    }

    // 10. Bounded timed-window rep planner + backward-compatible out_stride_wg packing for vk_async.
    {
        CHK(vkt_pack_out_stride(4u, 0u, 0u) == 4u, "out_stride_wg(4,0,0) == 4 preserves vk_async ABI");
        const uint32_t sh4096 = vkt_clk_stride_shift(4096u, VKT_CLK_SLOTS);
        const uint32_t packed = vkt_pack_out_stride(4u, VKT_CLK_SLOTS, sh4096);
        CHK((packed & 0xffu) == 4u && ((packed >> 8u) & 0xfffu) == VKT_CLK_SLOTS &&
                ((packed >> 20u) & 0x1fu) == 6u,
            "out_stride_wg round-trips (out_wg=4, clk_slots=64, clk_shift=6)");
        const int r_fast = vkt_plan_timed_reps(0.4e6, VKT_TIMED_TARGET_MS, VKT_TIMED_MIN_REPS, VKT_TIMED_MAX_REPS);
        const int r_mid = vkt_plan_timed_reps(1.5e6, VKT_TIMED_TARGET_MS, VKT_TIMED_MIN_REPS, VKT_TIMED_MAX_REPS);
        const int r_slow =
            (brk == 10) ? 4 : vkt_plan_timed_reps(10.0e6, VKT_TIMED_TARGET_MS, VKT_TIMED_MIN_REPS, VKT_TIMED_MAX_REPS);
        CHK(r_fast == VKT_TIMED_MAX_REPS && r_mid == 20 && r_slow >= VKT_MIN_CLK_SAMPLES &&
                r_slow == VKT_TIMED_MIN_REPS,
            "timed-window rep planner stays within [min>=5, max=48]");
    }

    // 11. Probe C DRAM even-iters odd-parity check & dropped-FMA rejection:
    //     visit-once DRAM keeps even iters (e.g. 16), shader executes +1 step per chain so every
    //     chain has odd parity, and a "dropped FMA" output (untouched init sum) MUST fail.
    {
        static const uint32_t ns[6] = {2u, 4u, 8u, 16u, 32u, 64u};
        static const float init_lo[5] = {0.25f, 0.375f, 0.5f, 0.625f, 0.75f};
        static const float init_hi[5] = {0.5f, 0.625f, 0.75f, 0.875f, 1.0f};
        for (int fi = 0; fi < 2; ++fi) {
            const int f = (fi == 0) ? VKT_FMT_RGBA16_SFLOAT : VKT_FMT_RGBA8_UNORM;
            uint32_t groups = 0u, iters = 0u, y_step = 0u;
            CHK(vkt_dram_schedule(f, 4u, &groups, &iters, &y_step), "Probe C DRAM schedule valid for ALU parity");
            for (int ni = 0; ni < 6; ++ni) {
                const uint32_t n = ns[ni];
                for (uint32_t lid = 0u; lid < 64u; lid += 9u) {
                    float ak_lo = 0.0f, ak_hi = 0.0f;
                    vkt_osc_lane_k(lid, &ak_lo, &ak_hi);
                    float a_lo[5], a_hi[5];
                    memcpy(a_lo, init_lo, sizeof a_lo);
                    memcpy(a_hi, init_hi, sizeof a_hi);
                    for (uint32_t it = 0u; it < iters; ++it) {
                        for (uint32_t s = 0u; s < 4u * n; ++s) {
                            const uint32_t c = s % 5u;
                            a_lo[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_lo[c], -1.0f, ak_lo)));
                            a_hi[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_hi[c], -1.0f, ak_hi)));
                        }
                    }
                    if (iters > 0u && (iters & 1u) == 0u) {
                        for (int c = 0; c < 5; ++c) {
                            a_lo[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_lo[c], -1.0f, ak_lo)));
                            a_hi[c] = vkt_f16_to_f32(vkt_f32_to_f16(fmaf(a_hi[c], -1.0f, ak_hi)));
                        }
                    }
                    float sim_lo = 0.0f, sim_hi = 0.0f;
                    for (int c = 0; c < 5; ++c) { sim_lo += a_lo[c]; sim_hi += a_hi[c]; }
                    CHK(vkt_verify_mix_alu_lane(lid, n, iters, 1, sim_lo, sim_hi, 0, NULL) == 1,
                        "Probe C DRAM FP16 even-iters odd-parity output passes");
                    CHK(vkt_verify_mix_alu_lane(lid, n, iters, 0, sim_lo, 0.0f, 0, NULL) == 1,
                        "Probe C DRAM FP32 even-iters odd-parity output passes");
                    // Dropped FMA loop storing untouched initial accumulators MUST fail.
                    const int dropped_fp16 =
                        vkt_verify_mix_alu_lane(lid, n, iters, 1, VKT_OSC5_INIT_SUM_LO, VKT_OSC5_INIT_SUM_HI, 0, NULL);
                    const int dropped_fp32 =
                        vkt_verify_mix_alu_lane(lid, n, iters, 0, VKT_OSC5_INIT_SUM_LO, 0.0f, 0, NULL);
                    CHK(dropped_fp16 == (brk == 11 ? 1 : 0),
                        "Probe C DRAM FP16 rejects dropped-FMA init output");
                    CHK(dropped_fp32 == 0,
                        "Probe C DRAM FP32 rejects dropped-FMA init output");
                }
            }
            // N=0 legitimately stores untouched init sum.
            CHK(vkt_verify_mix_alu_lane(0u, 0u, iters, 1, VKT_OSC5_INIT_SUM_LO, VKT_OSC5_INIT_SUM_HI, 0, NULL) == 1,
                "Probe C N=0 accepts initial accumulator sum");
        }
    }

    // 12. Short/reordered shader-clock span rejection & ACO clock-order verifier.
    {
        // 1-tick (10 ns) and 50-tick (500 ns) spans at valid 3200 MHz ratio must be rejected.
        uint32_t short_raw[3 * 4];
        memset(short_raw, 0, sizeof short_raw);
        short_raw[0] = 32u;   short_raw[2] = 1u;   // 1 tick = 10 ns, 3200 MHz
        short_raw[4] = 1600u; short_raw[6] = 50u;  // 50 ticks = 500 ns, 3200 MHz
        short_raw[8] = 1999u; short_raw[10] = 100u; // 1000 ns, but dc < VKT_MIN_CLK_CYCLES (2000)
        int n_short = -1;
        double short_rt = -1.0;
        const double short_mhz = vkt_decode_shader_clk_mhz(short_raw, 3u, &n_short, &short_rt);
        CHK((n_short == 0 && short_mhz == 0.0) == (brk != 12),
            "shader-clock decoder rejects 1-tick and sub-microsecond spans");

        // Minimum valid span (dc = 3200, dr = 100 -> 1000 ns, 3200 MHz) is accepted.
        uint32_t ok_raw[4] = {3200u, 0u, 100u, 0u};
        int n_ok = 0;
        const double ok_mhz = vkt_decode_shader_clk_mhz(ok_raw, 1u, &n_ok, NULL);
        CHK(n_ok == 1 && fabs(ok_mhz - 3200.0) < 1e-6,
            "shader-clock decoder accepts >= 1000 ns and >= 2000 cycles span");

        // ACO clock ordering check: start clocks before loop, reload + end clocks after loop.
        const char * good_asm =
            "BB1:\n"
            "\ts_getreg_b32 s1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)\n"
            "BB9:\n"
            "\timage_sample_lz v[0:3], v[4:5], s[0:7], s[8:11]\n"
            "\ts_cbranch_scc0 BB9\n"
            "BB20:\n"
            "\tbuffer_store_b128 v[0:3], off, s[12:15], s8\n"
            "\tbuffer_load_b128 v[0:3], off, s[12:15], s8\n"
            "\ts_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[6:7], sendmsg(MSG_RTN_GET_REALTIME)\n";
        const char * hoisted_asm =
            "BB1:\n"
            "\ts_getreg_b32 s1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)\n"
            "\ts_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[6:7], sendmsg(MSG_RTN_GET_REALTIME)\n"
            "BB9:\n"
            "\timage_sample_lz v[0:3], v[4:5], s[0:7], s[8:11]\n"
            "\ts_cbranch_scc0 BB9\n";
        const char * no_reload_asm =
            "BB1:\n"
            "\ts_getreg_b32 s1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[2:3], sendmsg(MSG_RTN_GET_REALTIME)\n"
            "BB9:\n"
            "\timage_sample_lz v[0:3], v[4:5], s[0:7], s[8:11]\n"
            "\ts_cbranch_scc0 BB9\n"
            "BB20:\n"
            "\tbuffer_store_b128 v[0:3], off, s[12:15], s8\n"
            "\ts_getreg_b32 s4, hwreg(HW_REG_SHADER_CYCLES_LO)\n"
            "\ts_sendmsg_rtn_b64 s[6:7], sendmsg(MSG_RTN_GET_REALTIME)\n";
        CHK(vkt_asm_clock_order_ok(good_asm) == 1, "ACO clock-order check accepts post-loop reload + end clocks");
        CHK(vkt_asm_clock_order_ok(hoisted_asm) == 0, "ACO clock-order check rejects hoisted end clocks");
        CHK(vkt_asm_clock_order_ok(no_reload_asm) == 0, "ACO clock-order check rejects end clocks without reload");
    }

    printf("VERDICT vk_tex_selftest: %s (bad=%d%s)\n",
           g_bad == 0 ? "PASS" : "FAIL", g_bad, brk ? ", deliberate break" : "");
    return g_bad == 0 ? 0 : 1;
}
