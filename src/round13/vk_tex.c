// Round 13 Probe B: Vulkan compute texture sampling throughput on gfx1201.
// Sweeps formats (R8_UNORM, RGBA8_UNORM, R16G16_SFLOAT, R16G16B16A16_SFLOAT, R32G32B32A32_SFLOAT,
// B10G11R11_UFLOAT if supported), ops (texelFetch, textureLod bilinear, textureGather),
// footprints (L0 16 KiB, L2 2 MiB, MALL 32 MiB, DRAM 512 MiB [256 MiB R8_UNORM max 2D], visit-once),
// and wave32 vs wave64.
//
// Reports texels/clock/CU and GB/s using in-shader core clock (GL_ARB_shader_clock /
// GL_EXT_shader_realtime_clock) averaged over a warmed timed window, cross-checked against 2 ms
// amdgpu gpu_metrics. Refuses to print clock-normalised values when fewer than 5 clock samples fall
// inside the timed window. Enforces fail-closed 90 C thermal stop and < 60 C cooldowns.
// Controls: exact FP16-representable bilinear/gather/fetch CPU verification + ACO image_* check.
// Deliberate break (--break): shifts shader x-coordinate by +1 texel; exactness control must FAIL.
//
// Usage: vk_tex [--dump-asm] [--break] [--smoke]
#define _GNU_SOURCE
#include "vk_tex_common.h"
#include "vk_tex_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t w;
    uint32_t h;
    uint32_t mask_x;
    uint32_t mask_y;
    uint32_t strips_x_mask;
    uint32_t strips_x_shift;
    uint32_t tiles_y_mask;
    uint32_t y_tile_step;
    uint32_t iters;
    uint32_t break_dx;
    uint32_t out_stride_wg;
    uint32_t strip_tiles;
    float inv_w;
    float inv_h;
    float du;
    float dv;
} TexPC;

static const VkFormat k_vk_fmts[VKT_FMT_COUNT] = {
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
};

static const char * k_spv_ops[VKT_OP_COUNT] = {
    "vk_tex_fetch.spv",
    "vk_tex_lod.spv",
    "vk_tex_gather.spv",
};

static uint32_t ctz_u32(uint32_t v) {
    uint32_t s = 0;
    while (v > 1u) { v >>= 1; ++s; }
    return s;
}

static int check_aco_op(int op, const VktLoopStats * ls) {
    if (op == VKT_OP_FETCH) return ls->image_load >= 16 && ls->image_sample == 0 && ls->image_gather == 0;
    if (op == VKT_OP_LOD) return ls->image_sample >= 16 && ls->image_load == 0 && ls->image_gather == 0;
    return ls->image_gather >= 16 && ls->image_load == 0 && ls->image_sample == 0;
}

static TexPC make_pc(uint32_t w, uint32_t h, uint32_t iters, uint32_t y_tile_step,
                     uint32_t break_dx, uint32_t out_wg, uint32_t clk_slots, uint32_t clk_shift) {
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (16u < tiles_x) ? 16u : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    TexPC pc = {0};
    pc.w = w;
    pc.h = h;
    pc.mask_x = w - 1u;
    pc.mask_y = h - 1u;
    pc.strips_x_mask = strips_x - 1u;
    pc.strips_x_shift = ctz_u32(strips_x);
    pc.tiles_y_mask = tiles_y - 1u;
    pc.y_tile_step = y_tile_step;
    pc.iters = iters;
    pc.break_dx = break_dx;
    pc.out_stride_wg = vkt_pack_out_stride(out_wg, clk_slots, clk_shift);
    pc.strip_tiles = strip;
    pc.inv_w = 1.0f / (float)w;
    pc.inv_h = 1.0f / (float)h;
    pc.du = 16.0f / (float)w;
    pc.dv = (float)(y_tile_step << 4u) / (float)h;
    return pc;
}

typedef struct {
    double med_ns;
    double lo_ns;
    double hi_ns;
    double sclk_avg_mhz;
    double gm_sclk_avg_mhz;
    double uclk_med_mhz;
    double hot_max_c;
    int reps;
    int clk_n;
    int gm_n;
    int clk_ok;
    int cooldown_failed;
} TexCellRun;

static TexCellRun run_timed_tex_cell(const VktPipe * pipe, uint32_t groups, uint32_t w, uint32_t h,
                                     uint32_t iters, uint32_t y_step, uint32_t check_wg,
                                     VktBuf * out_buf) {
    TexCellRun out;
    memset(&out, 0, sizeof out);
    if (!vkt_thermal_guard_and_cool(60.0, 15000)) {
        out.cooldown_failed = 1;
        out.hot_max_c = vkt_hotspot_c();
        return out;
    }

    // Bounded warmup (~20 ms, 3..40 reps) so auto DPM stabilizes before timing starts.
    TexPC pc_warm = make_pc(w, h, iters, y_step, 0u, check_wg, 0u, 0u);
    const double t_w0 = vkt_now_ns();
    double last_warm_ns = 0.0;
    for (int r = 0; r < VKT_WARM_MAX_REPS; ++r) {
        double wall = 0.0;
        last_warm_ns = vkt_dispatch(pipe, groups, &pc_warm, sizeof pc_warm, &wall);
        if (r + 1 >= VKT_WARM_MIN_REPS && (vkt_now_ns() - t_w0) >= VKT_WARM_TARGET_MS * 1e6) break;
    }

    const int target_reps =
        vkt_plan_timed_reps(last_warm_ns, VKT_TIMED_TARGET_MS, VKT_TIMED_MIN_REPS, VKT_TIMED_MAX_REPS);
    const uint32_t clk_slots = VT.has_shader_clock ? VKT_CLK_SLOTS : 0u;
    const uint32_t clk_shift = vkt_clk_stride_shift(groups, clk_slots);
    TexPC pc_timed = make_pc(w, h, iters, y_step, 0u, check_wg, clk_slots, clk_shift);
    uint32_t * clk_raw = (uint32_t *)out_buf->map + (size_t)check_wg * 256u * 4u;

    double gns[VKT_TIMED_MAX_REPS];
    int n_reps = 0;
    double sh_clk_sum = 0.0;
    int sh_clk_n = 0;

    vkt_metrics_start();
    const double t_m0 = vkt_now_ns();
    for (int r = 0; r < target_reps; ++r) {
        if (clk_slots > 0u) memset(clk_raw, 0, (size_t)clk_slots * 4u * sizeof(uint32_t));
        double wall = 0.0;
        gns[n_reps] = vkt_dispatch(pipe, groups, &pc_timed, sizeof pc_timed, &wall);
        if (clk_slots > 0u) {
            int wg_valid = 0;
            double max_rt_ns = 0.0;
            const double dclk = vkt_decode_shader_clk_mhz(clk_raw, clk_slots, &wg_valid, &max_rt_ns);
            if (wg_valid >= VKT_MIN_CLK_SAMPLES && dclk > 0.0) {
                sh_clk_sum += dclk;
                ++sh_clk_n;
            }
        }
        ++n_reps;
        if (n_reps >= VKT_MIN_CLK_SAMPLES && (vkt_now_ns() - t_m0) >= VKT_TIMED_MAX_WALL_MS * 1e6) break;
    }
    VktMetrics mt = vkt_metrics_stop();

    out.reps = n_reps;
    out.med_ns = vkt_median(gns, n_reps);
    out.lo_ns = vkt_vmin(gns, n_reps);
    out.hi_ns = vkt_vmax(gns, n_reps);
    out.gm_sclk_avg_mhz = mt.sclk_avg_mhz;
    out.gm_n = mt.n_samples;
    out.uclk_med_mhz = mt.uclk_med_mhz;
    out.hot_max_c = mt.hot_max_c;
    if (VT.has_shader_clock) {
        out.clk_n = sh_clk_n;
        out.sclk_avg_mhz = (sh_clk_n > 0) ? (sh_clk_sum / (double)sh_clk_n) : 0.0;
    } else {
        out.clk_n = mt.n_samples;
        out.sclk_avg_mhz = mt.sclk_avg_mhz;
    }
    out.clk_ok = vkt_clk_gate_ok(out.clk_n, out.sclk_avg_mhz);
    return out;
}

static int verify_output(int fmt, int op, int exact_mode, uint32_t w, uint32_t h,
                         uint32_t iters, uint32_t y_tile_step, uint32_t check_wg,
                         const float * got_buf, float * max_err_out) {
    const float tol = exact_mode ? 0.0f : (0.004f * (float)(iters * 16u));
    float max_err = 0.0f;
    int bad = 0;
    for (uint32_t wg = 0; wg < check_wg; ++wg) {
        for (uint32_t lid = 0; lid < 256u; ++lid) {
            float want[4];
            vkt_expected_thread(fmt, op, exact_mode, w, h, wg, lid, iters, y_tile_step, 0u, want);
            const float * got = got_buf + (wg * 256u + lid) * 4u;
            for (int c = 0; c < 4; ++c) {
                float e = fabsf(got[c] - want[c]);
                if (e > max_err) max_err = e;
                if (e > tol) ++bad;
            }
        }
    }
    if (max_err_out) *max_err_out = max_err;
    return bad;
}

int main(int argc, char ** argv) {
    int dump_asm = 0, do_break = 0, smoke = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-asm") == 0) dump_asm = 1;
        else if (strcmp(argv[i], "--break") == 0) do_break = 1;
        else if (strcmp(argv[i], "--smoke") == 0) smoke = 1;
        else {
            fprintf(stderr, "usage: vk_tex [--dump-asm] [--break] [--smoke]\n");
            return 2;
        }
    }

    vkt_init(0);
    printf("CLOCK_SRC vk_tex primary=%s gpu_metrics_poll=2ms min_clk_samples=%d warm_ms=%.0f timed_ms=%.0f\n",
           VT.has_shader_clock ? "shader(GL_ARB_shader_clock+GL_EXT_shader_realtime_clock)" : "gpu_metrics",
           VKT_MIN_CLK_SAMPLES, VKT_WARM_TARGET_MS, VKT_TIMED_TARGET_MS);

    // Create all 6 pipelines (3 ops x wave32/wave64) and verify ACO image_* and shader-clock instructions.
    VktPipe pipes[VKT_OP_COUNT][2];
    VktLoopStats lstats[VKT_OP_COUNT][2];
    int aco_fail = 0;
    const uint32_t sgs[2] = {32u, 64u};
    for (int op = 0; op < VKT_OP_COUNT; ++op) {
        for (int si = 0; si < 2; ++si) {
            char label[64];
            snprintf(label, sizeof label, "vk_tex_%s_w%u", vkt_op_names[op], sgs[si]);
            pipes[op][si] = vkt_pipe_create(k_spv_ops[op], label, 256u, sgs[si]);
            lstats[op][si] = vkt_pipe_analyze_loop(&pipes[op][si], "image_");
            const int clk_aco_ok = !VT.has_shader_clock || vkt_pipe_has_shader_clock(&pipes[op][si]);
            const int ok = check_aco_op(op, &lstats[op][si]) && clk_aco_ok;
            if (!ok) ++aco_fail;
            printf("ACO %-26s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d img_load=%2d img_samp=%2d img_gath=%2d sh_clk=%s check=%s\n",
                   label, pipes[op][si].sg_size, pipes[op][si].vgprs, pipes[op][si].sg_per_simd,
                   lstats[op][si].total_insts, lstats[op][si].image_load, lstats[op][si].image_sample,
                   lstats[op][si].image_gather, clk_aco_ok ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
        }
    }
    fflush(stdout);

    if (dump_asm) {
        for (int op = 0; op < VKT_OP_COUNT; ++op)
            for (int si = 0; si < 2; ++si) vkt_pipe_destroy(&pipes[op][si]);
        vkt_shutdown();
        printf("VERDICT vk_tex_asm: %s (aco_fail=%d)\n", aco_fail == 0 ? "PASS" : "FAIL", aco_fail);
        return aco_fail == 0 ? 0 : 1;
    }

    if (!vkt_thermal_guard_and_cool(60.0, 15000)) {
        for (int op = 0; op < VKT_OP_COUNT; ++op)
            for (int si = 0; si < 2; ++si) vkt_pipe_destroy(&pipes[op][si]);
        vkt_shutdown();
        printf("VERDICT vk_tex: FAIL (fails=%d, pre-run cooldown timeout)\n", aco_fail + 1);
        return 1;
    }

    const uint32_t kCheckWGs = 4u;
    VktBuf out_buf = vkt_buf_create(((size_t)kCheckWGs * 256u + VKT_CLK_SLOTS) * 4u * sizeof(float), 1);

    int fails = aco_fail;

    // 1. Exact bit-for-bit FP16-weight control for every supported format and all 3 ops (L0 exact_mode=1).
    for (int f = 0; f < VKT_FMT_COUNT; ++f) {
        if (!vkt_format_supported(k_vk_fmts[f])) {
            printf("CONTROL vk_tex_exact fmt=%-20s SKIP unsupported\n", vkt_fmt_names[f]);
            continue;
        }
        uint32_t w = 0, h = 0;
        vkt_level_dims(f, VKT_LVL_L0, &w, &h);
        VktImg im = vkt_img_create(k_vk_fmts[f], w, h, VK_FILTER_LINEAR);
        vkt_img_fill_ref(&im, f, 1);
        vkt_bind(&im, &out_buf, NULL);
        for (int op = 0; op < VKT_OP_COUNT; ++op) {
            TexPC pc = make_pc(w, h, 8u, 1u, do_break ? 1u : 0u, kCheckWGs, 0u, 0u);
            double wall = 0.0;
            vkt_dispatch(&pipes[op][0], kCheckWGs, &pc, sizeof pc, &wall);
            float max_err = 0.0f;
            int bad = verify_output(f, op, 1, w, h, 8u, 1u, kCheckWGs, (const float *)out_buf.map, &max_err);
            if (bad != 0) ++fails;
            printf("CONTROL vk_tex_exact fmt=%-20s op=%-13s bad=%d max_err=%.6g exact=%s\n",
                   vkt_fmt_names[f], vkt_op_names[op], bad, max_err, bad == 0 ? "ok" : "FAIL");
        }
        vkt_img_destroy(&im);
    }
    fflush(stdout);

    if (do_break) {
        vkt_buf_destroy(&out_buf);
        for (int op = 0; op < VKT_OP_COUNT; ++op)
            for (int si = 0; si < 2; ++si) vkt_pipe_destroy(&pipes[op][si]);
        vkt_shutdown();
        printf("VERDICT vk_tex: %s (fails=%d, deliberate break)\n", fails == 0 ? "PASS" : "FAIL", fails);
        return fails == 0 ? 0 : 1;
    }

    // 2. Full sweep (or --smoke subset): formats x footprints x ops x wave32/wave64.
    const int max_lvl = smoke ? 1 : VKT_LVL_COUNT;
    for (int f = 0; f < VKT_FMT_COUNT; ++f) {
        if (smoke && f > 2) break;
        if (!vkt_format_supported(k_vk_fmts[f])) {
            printf("RESULT vk_tex fmt=%-20s SKIP unsupported\n", vkt_fmt_names[f]);
            continue;
        }
        const int timed_exact = (f >= VKT_FMT_R16G16_SFLOAT) ? 1 : 0;
        for (int lvl = 0; lvl < max_lvl; ++lvl) {
            uint32_t w = 0, h = 0;
            vkt_level_dims(f, lvl, &w, &h);
            VktImg im = vkt_img_create(k_vk_fmts[f], w, h, VK_FILTER_LINEAR);
            vkt_img_fill_ref(&im, f, timed_exact);
            vkt_bind(&im, &out_buf, NULL);

            uint32_t groups = 1024u;
            uint32_t iters = 128u;
            uint32_t y_step = 1u;
            if (lvl == VKT_LVL_L0) {
                groups = 1024u;
                iters = (vkt_fmt_bpp[f] >= 8u) ? 64u : 128u;
                y_step = 1u;
            } else if (lvl == VKT_LVL_L2) {
                groups = 2048u;
                iters = (vkt_fmt_bpp[f] >= 8u) ? 32u : 64u;
                y_step = 1u;
            } else if (lvl == VKT_LVL_MALL) {
                groups = 4096u;
                iters = (vkt_fmt_bpp[f] >= 8u) ? 16u : 32u;
                y_step = 4u;
            } else {
                // DRAM (512 MiB for bpp >= 4, 256 MiB for R8_UNORM): strict visit-once schedule,
                // zero strip wrap and zero tile reuse within a dispatch.
                if (!vkt_dram_schedule(f, 16u, &groups, &iters, &y_step)) {
                    fprintf(stderr, "ERROR: invalid DRAM visit-once schedule for fmt=%d\n", f);
                    exit(3);
                }
            }

            for (int op = 0; op < VKT_OP_COUNT; ++op) {
                for (int si = 0; si < 2; ++si) {
                    TexCellRun run = run_timed_tex_cell(&pipes[op][si], groups, w, h, iters, y_step,
                                                        kCheckWGs, &out_buf);
                    if (run.cooldown_failed) {
                        ++fails;
                        printf("RESULT vk_tex fmt=%-20s op=%-13s lvl=%-4s w=%2u dim=%ux%u "
                               "SKIPPED_COOLDOWN_TIMEOUT hot=%.0fC exact=FAIL\n",
                               vkt_fmt_names[f], vkt_op_names[op], vkt_lvl_names[lvl], sgs[si], w, h,
                               run.hot_max_c);
                        fflush(stdout);
                        continue;
                    }

                    float max_err = 0.0f;
                    int bad = verify_output(f, op, timed_exact, w, h, iters, y_step, kCheckWGs,
                                            (const float *)out_buf.map, &max_err);
                    if (bad != 0) ++fails;

                    const uint64_t total_samp = (uint64_t)groups * 256ull * (uint64_t)iters * 16ull;
                    const double gsamp = (double)total_samp / run.med_ns;
                    const double gbs = vkt_gbs(total_samp, vkt_fmt_bpp[f], run.med_ns);
                    const double spc = run.clk_ok
                                           ? vkt_samp_per_clk_per_cu(total_samp, run.med_ns, run.sclk_avg_mhz)
                                           : NAN;
                    char spc_s[32], sclk_s[32], gm_sclk_s[32];
                    vkt_fmt_clk_norm(spc_s, sizeof spc_s, run.clk_ok, "%.3f", spc);
                    vkt_fmt_clk_norm(sclk_s, sizeof sclk_s, run.clk_ok, "%.0fMHz", run.sclk_avg_mhz);
                    vkt_fmt_clk_norm(gm_sclk_s, sizeof gm_sclk_s,
                                     vkt_clk_gate_ok(run.gm_n, run.gm_sclk_avg_mhz), "%.0fMHz",
                                     run.gm_sclk_avg_mhz);
                    const char * spread = (run.hi_ns > run.lo_ns * 1.10) ? " SPREAD>10%" : "";
                    const char * clk_warn = run.clk_ok ? "" : " INSUFFICIENT_CLK_SAMPLES";

                    printf("RESULT vk_tex fmt=%-20s op=%-13s lvl=%-4s w=%2u dim=%ux%u "
                           "samp/clk/CU=%s GSamp/s=%.2f GB/s=%.1f gpu_ms=%.3f [%.3f-%.3f] reps=%d "
                           "sclk=%s clk_n=%d gm_sclk=%s gm_n=%d uclk=%.0fMHz hot=%.0fC exact=%s(err=%.2g)%s%s\n",
                           vkt_fmt_names[f], vkt_op_names[op], vkt_lvl_names[lvl], sgs[si], w, h,
                           spc_s, gsamp, gbs, run.med_ns * 1e-6, run.lo_ns * 1e-6, run.hi_ns * 1e-6,
                           run.reps, sclk_s, run.clk_n, gm_sclk_s, run.gm_n, run.uclk_med_mhz,
                           run.hot_max_c, bad == 0 ? "ok" : "FAIL", max_err, spread, clk_warn);
                    fflush(stdout);
                }
            }
            vkt_img_destroy(&im);
        }
    }

    vkt_buf_destroy(&out_buf);
    for (int op = 0; op < VKT_OP_COUNT; ++op)
        for (int si = 0; si < 2; ++si) vkt_pipe_destroy(&pipes[op][si]);
    vkt_shutdown();

    printf("VERDICT vk_tex: %s (fails=%d)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
