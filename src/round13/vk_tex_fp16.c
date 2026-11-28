// Round 13 Probe C: Texture sampling overlapped with packed FP16 FMAs (f16vec2 -> v_pk_fma_f16)
// and FP32 FMAs (float -> v_fma_f32 / v_fmac_f32 / v_dual_fmac_f32), using the shared vk_tex harness.
//
// Sweeps FMA_PER_TEX in {0, 2, 4, 8, 16, 32, 64} plus pure ALU baselines (DO_TEX=0) at {4, 16, 64},
// across formats (RGBA16F, RGBA8), cache levels (L0 16 KiB, L2 2 MiB, DRAM 512 MiB visit-once),
// and wave32/wave64. Computes overlap ratios T_mix / max(T_tex, T_alu) and T_mix / (T_tex + T_alu)
// using in-shader core clock (GL_ARB_shader_clock / GL_EXT_shader_realtime_clock) averaged over a
// warmed timed window, cross-checked against 2 ms gpu_metrics. Refuses to print clock-normalised
// values when fewer than 5 clock samples fall inside the timed window. Enforces fail-closed 90 C
// thermal stop and < 60 C cooldowns.
//
// Controls: exact CPU check of both texture sum and 5-chain oscillating FMA sum, plus ACO verification
// that v_pk_fma_f16 (FP16) and image_sample (texture) are emitted in the inner loop.
// Deliberate break (--break): shifts shader x-coordinate by +1 texel and corrupts expected ALU parity.
//
// Usage: vk_tex_fp16 [--dump-asm] [--break] [--smoke]
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
} MixPC;

enum { kNumN = 7 };
static const uint32_t kNList[kNumN] = {0u, 2u, 4u, 8u, 16u, 32u, 64u};

static uint32_t ctz_u32(uint32_t v) {
    uint32_t s = 0;
    while (v > 1u) { v >>= 1; ++s; }
    return s;
}

static MixPC make_pc(uint32_t w, uint32_t h, uint32_t iters, uint32_t y_tile_step,
                     uint32_t break_dx, uint32_t out_wg, uint32_t clk_slots, uint32_t clk_shift) {
    const uint32_t tiles_x = w >> 4u;
    const uint32_t tiles_y = h >> 4u;
    const uint32_t strip = (4u < tiles_x) ? 4u : tiles_x;
    const uint32_t strips_x = tiles_x / strip;
    MixPC pc = {0};
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
} MixCellRun;

static MixCellRun run_timed_mix_cell(const VktPipe * pipe, uint32_t groups, uint32_t w, uint32_t h,
                                     uint32_t iters, uint32_t y_step, uint32_t check_wg,
                                     VktBuf * out_buf) {
    MixCellRun out;
    memset(&out, 0, sizeof out);
    if (!vkt_thermal_guard_and_cool(60.0, 15000)) {
        out.cooldown_failed = 1;
        out.hot_max_c = vkt_hotspot_c();
        return out;
    }

    MixPC pc_warm = make_pc(w, h, iters, y_step, 0u, check_wg, 0u, 0u);
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
    MixPC pc_timed = make_pc(w, h, iters, y_step, 0u, check_wg, clk_slots, clk_shift);
    uint32_t * clk_raw = (uint32_t *)out_buf->map + (size_t)check_wg * 256u * 8u;

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

static int verify_mix_output(int fmt, int exact_mode, int do_tex, int use_fp16,
                             uint32_t w, uint32_t h, uint32_t iters, uint32_t y_tile_step,
                             uint32_t n_fma, uint32_t check_wg, const float * got_buf,
                             int break_alu, float * max_tex_err, float * max_alu_err) {
    const float tex_tol = exact_mode ? 0.0f : (0.004f * (float)(iters * 4u));
    float mte = 0.0f, mae = 0.0f;
    int bad = 0;

    for (uint32_t wg = 0; wg < check_wg; ++wg) {
        for (uint32_t lid = 0; lid < 256u; ++lid) {
            const float * got_tex = got_buf + (wg * 256u + lid) * 8u;
            const float * got_alu = got_tex + 4u;
            if (do_tex) {
                float want_tex[4];
                vkt_expected_thread_4tap(fmt, exact_mode, w, h, wg, lid, iters, y_tile_step, 0u, want_tex);
                for (int c = 0; c < 4; ++c) {
                    float e = fabsf(got_tex[c] - want_tex[c]);
                    if (e > mte) mte = e;
                    if (e > tex_tol) ++bad;
                }
            }
            float alu_e = 0.0f;
            if (!vkt_verify_mix_alu_lane(lid, n_fma, iters, use_fp16, got_alu[0], got_alu[1],
                                         (break_alu && wg == 0u && lid == 0u), &alu_e)) {
                ++bad;
            }
            if (alu_e > mae) mae = alu_e;
        }
    }
    if (max_tex_err) *max_tex_err = mte;
    if (max_alu_err) *max_alu_err = mae;
    return bad;
}

static int check_aco_mix(int do_tex, int use_fp16, uint32_t n_fma, const VktLoopStats * ls) {
    const int need_tex = do_tex ? 4 : 0;
    const int need_fma = (int)(4u * n_fma);
    if (do_tex && ls->image_sample < need_tex) return 0;
    if (!do_tex && ls->image_sample != 0) return 0;
    if (n_fma == 0u) return 1;
    if (use_fp16) return ls->pk_fma_f16 >= need_fma;
    return (ls->fma_f32 + 2 * ls->dual_fmac_f32) >= need_fma;
}

int main(int argc, char ** argv) {
    int dump_asm = 0, do_break = 0, smoke = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-asm") == 0) dump_asm = 1;
        else if (strcmp(argv[i], "--break") == 0) do_break = 1;
        else if (strcmp(argv[i], "--smoke") == 0) smoke = 1;
        else {
            fprintf(stderr, "usage: vk_tex_fp16 [--dump-asm] [--break] [--smoke]\n");
            return 2;
        }
    }

    vkt_init(0);
    printf("CLOCK_SRC vk_tex_fp16 primary=%s gpu_metrics_poll=2ms min_clk_samples=%d warm_ms=%.0f timed_ms=%.0f\n",
           VT.has_shader_clock ? "shader(GL_ARB_shader_clock+GL_EXT_shader_realtime_clock)" : "gpu_metrics",
           VKT_MIN_CLK_SAMPLES, VKT_WARM_TARGET_MS, VKT_TIMED_TARGET_MS);

    // Pipelines: [prec: 0=fp16, 1=fp32][sg: 0=32, 1=64][n_idx: 0..6] for DO_TEX=1,
    // plus pure ALU baselines [prec][sg][0=N4, 1=N16, 2=N64] for DO_TEX=0.
    const uint32_t sgs[2] = {32u, 64u};
    const char * pnames[2] = {"fp16", "fp32"};
    const uint32_t alu_ns[3] = {4u, 16u, 64u};

    VktPipe mix_pipes[2][2][kNumN];
    VktPipe alu_pipes[2][2][3];
    int aco_fail = 0;

    for (int pr = 0; pr < 2; ++pr) {
        const int use_fp16 = (pr == 0);
        for (int si = 0; si < 2; ++si) {
            for (int ni = 0; ni < kNumN; ++ni) {
                char spv[64], label[64];
                snprintf(spv, sizeof spv, "vk_mix_%s_n%u.spv", pnames[pr], kNList[ni]);
                snprintf(label, sizeof label, "mix_%s_n%u_w%u", pnames[pr], kNList[ni], sgs[si]);
                mix_pipes[pr][si][ni] = vkt_pipe_create(spv, label, 256u, sgs[si]);
                VktLoopStats ls = vkt_pipe_analyze_loop(&mix_pipes[pr][si][ni], "image_sample");
                const int clk_aco_ok = !VT.has_shader_clock || vkt_pipe_has_shader_clock(&mix_pipes[pr][si][ni]);
                int ok = check_aco_mix(1, use_fp16, kNList[ni], &ls) && clk_aco_ok;
                if (!ok) ++aco_fail;
                printf("ACO %-20s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d samp=%2d pk_f16=%3d f32=%3d dual_f32=%3d sh_clk=%s check=%s\n",
                       label, mix_pipes[pr][si][ni].sg_size, mix_pipes[pr][si][ni].vgprs,
                       mix_pipes[pr][si][ni].sg_per_simd, ls.total_insts, ls.image_sample,
                       ls.pk_fma_f16, ls.fma_f32, ls.dual_fmac_f32, clk_aco_ok ? "PASS" : "FAIL",
                       ok ? "PASS" : "FAIL");
            }
            for (int ai = 0; ai < 3; ++ai) {
                char spv[64], label[64];
                snprintf(spv, sizeof spv, "vk_alu_%s_n%u.spv", pnames[pr], alu_ns[ai]);
                snprintf(label, sizeof label, "alu_%s_n%u_w%u", pnames[pr], alu_ns[ai], sgs[si]);
                alu_pipes[pr][si][ai] = vkt_pipe_create(spv, label, 256u, sgs[si]);
                VktLoopStats ls = vkt_pipe_analyze_loop(&alu_pipes[pr][si][ai], use_fp16 ? "v_pk_fma" : "fma");
                const int clk_aco_ok = !VT.has_shader_clock || vkt_pipe_has_shader_clock(&alu_pipes[pr][si][ai]);
                int ok = check_aco_mix(0, use_fp16, alu_ns[ai], &ls) && clk_aco_ok;
                if (!ok) ++aco_fail;
                printf("ACO %-20s sg=%2u VGPR=%3d aco_sg=%2d insts=%3d samp=%2d pk_f16=%3d f32=%3d dual_f32=%3d sh_clk=%s check=%s\n",
                       label, alu_pipes[pr][si][ai].sg_size, alu_pipes[pr][si][ai].vgprs,
                       alu_pipes[pr][si][ai].sg_per_simd, ls.total_insts, ls.image_sample,
                       ls.pk_fma_f16, ls.fma_f32, ls.dual_fmac_f32, clk_aco_ok ? "PASS" : "FAIL",
                       ok ? "PASS" : "FAIL");
            }
        }
    }
    fflush(stdout);

    if (dump_asm) {
        for (int pr = 0; pr < 2; ++pr)
            for (int si = 0; si < 2; ++si) {
                for (int ni = 0; ni < kNumN; ++ni) vkt_pipe_destroy(&mix_pipes[pr][si][ni]);
                for (int ai = 0; ai < 3; ++ai) vkt_pipe_destroy(&alu_pipes[pr][si][ai]);
            }
        vkt_shutdown();
        printf("VERDICT vk_tex_fp16_asm: %s (aco_fail=%d)\n", aco_fail == 0 ? "PASS" : "FAIL", aco_fail);
        return aco_fail == 0 ? 0 : 1;
    }

    if (!vkt_thermal_guard_and_cool(60.0, 15000)) {
        for (int pr = 0; pr < 2; ++pr)
            for (int si = 0; si < 2; ++si) {
                for (int ni = 0; ni < kNumN; ++ni) vkt_pipe_destroy(&mix_pipes[pr][si][ni]);
                for (int ai = 0; ai < 3; ++ai) vkt_pipe_destroy(&alu_pipes[pr][si][ai]);
            }
        vkt_shutdown();
        printf("VERDICT vk_tex_fp16: FAIL (fails=%d, pre-run cooldown timeout)\n", aco_fail + 1);
        return 1;
    }

    const uint32_t kCheckWGs = 4u;
    VktBuf out_buf = vkt_buf_create(((size_t)kCheckWGs * 256u * 2u + VKT_CLK_SLOTS) * 4u * sizeof(float), 1);
    VktBuf k_buf = vkt_buf_create(256u * 4u * sizeof(float), 1);
    {
        float * kp = (float *)k_buf.map;
        for (uint32_t lid = 0; lid < 256u; ++lid) {
            float ak_lo = 0.0f, ak_hi = 0.0f;
            vkt_osc_lane_k(lid, &ak_lo, &ak_hi);
            kp[lid * 4u + 0u] = -1.0f;
            kp[lid * 4u + 1u] = -1.0f;
            kp[lid * 4u + 2u] = ak_lo;
            kp[lid * 4u + 3u] = ak_hi;
        }
    }

    int fails = aco_fail;

    // 1. Exact control + deliberate break check (L0 RGBA16F exact_mode=1, odd iters=31, N=8).
    {
        uint32_t w = 0, h = 0;
        vkt_level_dims(VKT_FMT_RGBA16_SFLOAT, VKT_LVL_L0, &w, &h);
        VktImg im = vkt_img_create(VK_FORMAT_R16G16B16A16_SFLOAT, w, h, VK_FILTER_LINEAR);
        vkt_img_fill_ref(&im, VKT_FMT_RGBA16_SFLOAT, 1);
        vkt_bind(&im, &out_buf, &k_buf);
        for (int pr = 0; pr < 2; ++pr) {
            MixPC pc = make_pc(w, h, 31u, 1u, do_break ? 1u : 0u, kCheckWGs, 0u, 0u);
            double wall = 0.0;
            vkt_dispatch(&mix_pipes[pr][0][3], kCheckWGs, &pc, sizeof pc, &wall);
            float mte = 0.0f, mae = 0.0f;
            int bad = verify_mix_output(VKT_FMT_RGBA16_SFLOAT, 1, 1, pr == 0, w, h, 31u, 1u, 8u,
                                        kCheckWGs, (const float *)out_buf.map, do_break, &mte, &mae);
            if (bad != 0) ++fails;
            printf("CONTROL vk_tex_fp16_exact prec=%s bad=%d tex_err=%.3g alu_err=%.3g exact=%s\n",
                   pnames[pr], bad, mte, mae, bad == 0 ? "ok" : "FAIL");
        }
        vkt_img_destroy(&im);
    }
    fflush(stdout);

    if (do_break) {
        vkt_buf_destroy(&out_buf);
        vkt_buf_destroy(&k_buf);
        for (int pr = 0; pr < 2; ++pr)
            for (int si = 0; si < 2; ++si) {
                for (int ni = 0; ni < kNumN; ++ni) vkt_pipe_destroy(&mix_pipes[pr][si][ni]);
                for (int ai = 0; ai < 3; ++ai) vkt_pipe_destroy(&alu_pipes[pr][si][ai]);
            }
        vkt_shutdown();
        printf("VERDICT vk_tex_fp16: %s (fails=%d, deliberate break)\n", fails == 0 ? "PASS" : "FAIL", fails);
        return fails == 0 ? 0 : 1;
    }

    // 2. Measure pure ALU rate (ns per FMA-inst across the grid at N=64, scaled per N) for each prec & wave size.
    double alu_ns_per_inst_unit[2][2] = {{0}};  // GPU ns per (1024 WGs x 1 iter x 4 taps x 1 FMA/tap)
    double alu_sclk_mhz[2][2] = {{0}};
    int alu_clk_ok[2][2] = {{0}};
    {
        uint32_t w = 64u, h = 32u;
        VktImg dummy_im = vkt_img_create(VK_FORMAT_R16G16B16A16_SFLOAT, w, h, VK_FILTER_LINEAR);
        vkt_img_fill_ref(&dummy_im, VKT_FMT_RGBA16_SFLOAT, 1);
        vkt_bind(&dummy_im, &out_buf, &k_buf);
        const uint32_t alu_groups = 1024u;
        for (int pr = 0; pr < 2; ++pr) {
            for (int si = 0; si < 2; ++si) {
                for (int ai = 0; ai < 3; ++ai) {
                    const uint32_t n = alu_ns[ai];
                    // Odd iters for 5-chain parity; sized to ~0.4-1.2 ms/dispatch so timed window hits ~25-35 ms.
                    const uint32_t iters = (n == 64u) ? 127u : ((n == 16u) ? 255u : 511u);
                    MixCellRun run = run_timed_mix_cell(&alu_pipes[pr][si][ai], alu_groups, w, h,
                                                        iters, 1u, kCheckWGs, &out_buf);
                    if (run.cooldown_failed) {
                        ++fails;
                        printf("RESULT vk_alu_base prec=%s w=%2u N=%2u SKIPPED_COOLDOWN_TIMEOUT hot=%.0fC exact=FAIL\n",
                               pnames[pr], sgs[si], n, run.hot_max_c);
                        fflush(stdout);
                        continue;
                    }
                    float mte = 0.0f, mae = 0.0f;
                    int bad = verify_mix_output(VKT_FMT_RGBA16_SFLOAT, 1, 0, pr == 0, w, h, iters, 1u, n,
                                                kCheckWGs, (const float *)out_buf.map, 0, &mte, &mae);
                    if (bad != 0) ++fails;
                    const uint64_t total_inst = (uint64_t)alu_groups * 256ull * (uint64_t)iters * 4ull * (uint64_t)n;
                    const double flops_per_inst = (pr == 0) ? 4.0 : 2.0;
                    const double tflops = ((double)total_inst * flops_per_inst) / (run.med_ns * 1e3);
                    const double inst_per_clk_cu =
                        run.clk_ok ? vkt_samp_per_clk_per_cu(total_inst, run.med_ns, run.sclk_avg_mhz) : NAN;
                    if (ai == 2) {
                        alu_ns_per_inst_unit[pr][si] = run.med_ns / ((double)iters * (double)n);
                        alu_sclk_mhz[pr][si] = run.sclk_avg_mhz;
                        alu_clk_ok[pr][si] = run.clk_ok;
                    }
                    char ipc_s[32], sclk_s[32], gm_sclk_s[32];
                    vkt_fmt_clk_norm(ipc_s, sizeof ipc_s, run.clk_ok, "%.2f", inst_per_clk_cu);
                    vkt_fmt_clk_norm(sclk_s, sizeof sclk_s, run.clk_ok, "%.0fMHz", run.sclk_avg_mhz);
                    vkt_fmt_clk_norm(gm_sclk_s, sizeof gm_sclk_s,
                                     vkt_clk_gate_ok(run.gm_n, run.gm_sclk_avg_mhz), "%.0fMHz",
                                     run.gm_sclk_avg_mhz);
                    const char * clk_warn = run.clk_ok ? "" : " INSUFFICIENT_CLK_SAMPLES";
                    printf("RESULT vk_alu_base prec=%s w=%2u N=%2u gpu_ms=%.3f TFLOPS=%.2f inst/clk/CU=%s "
                           "reps=%d sclk=%s clk_n=%d gm_sclk=%s gm_n=%d hot=%.0fC exact=%s%s\n",
                           pnames[pr], sgs[si], n, run.med_ns * 1e-6, tflops, ipc_s,
                           run.reps, sclk_s, run.clk_n, gm_sclk_s, run.gm_n, run.hot_max_c,
                           bad == 0 ? "ok" : "FAIL", clk_warn);
                    fflush(stdout);
                }
            }
        }
        vkt_img_destroy(&dummy_im);
    }

    // 3. Interleaved texture + ALU sweep: formats {RGBA16F, RGBA8} x levels {L0, L2, DRAM} x prec x wave x N.
    const int fmts[2] = {VKT_FMT_RGBA16_SFLOAT, VKT_FMT_RGBA8_UNORM};
    const VkFormat vk_fmts[2] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM};
    const int lvls[3] = {VKT_LVL_L0, VKT_LVL_L2, VKT_LVL_DRAM};
    const int n_fmts = smoke ? 1 : 2;
    const int n_lvls = smoke ? 1 : 3;

    for (int fi = 0; fi < n_fmts; ++fi) {
        const int f = fmts[fi];
        const int exact_mode = (f == VKT_FMT_RGBA16_SFLOAT) ? 1 : 0;
        for (int li = 0; li < n_lvls; ++li) {
            const int lvl = lvls[li];
            uint32_t w = 0, h = 0;
            vkt_level_dims(f, lvl, &w, &h);
            VktImg im = vkt_img_create(vk_fmts[fi], w, h, VK_FILTER_LINEAR);
            vkt_img_fill_ref(&im, f, exact_mode);
            vkt_bind(&im, &out_buf, &k_buf);

            // Odd iters on L0/L2 so 5-chain oscillating parity is non-trivial;
            // strict visit-once power-of-two geometry on 512 MiB DRAM so every texel is read once.
            uint32_t groups = 1024u;
            uint32_t iters = 127u;
            uint32_t y_step = 1u;
            if (lvl == VKT_LVL_L0) {
                groups = 1024u;
                iters = 127u;
                y_step = 1u;
            } else if (lvl == VKT_LVL_L2) {
                groups = 2048u;
                iters = 63u;
                y_step = 1u;
            } else {
                if (!vkt_dram_schedule(f, 4u, &groups, &iters, &y_step)) {
                    fprintf(stderr, "ERROR: invalid DRAM visit-once schedule for fmt=%d\n", f);
                    exit(3);
                }
            }

            for (int pr = 0; pr < 2; ++pr) {
                for (int si = 0; si < 2; ++si) {
                    double t_tex_ns = 0.0;
                    for (int ni = 0; ni < kNumN; ++ni) {
                        const uint32_t n = kNList[ni];
                        MixCellRun run = run_timed_mix_cell(&mix_pipes[pr][si][ni], groups, w, h,
                                                            iters, y_step, kCheckWGs, &out_buf);
                        if (run.cooldown_failed) {
                            ++fails;
                            printf("RESULT vk_tex_fp16 fmt=%-19s lvl=%-4s prec=%s w=%2u N=%2u "
                                   "SKIPPED_COOLDOWN_TIMEOUT hot=%.0fC exact=FAIL\n",
                                   vkt_fmt_names[f], vkt_lvl_names[lvl], pnames[pr], sgs[si], n,
                                   run.hot_max_c);
                            fflush(stdout);
                            continue;
                        }

                        float mte = 0.0f, mae = 0.0f;
                        int bad = verify_mix_output(f, exact_mode, 1, pr == 0, w, h, iters, y_step, n,
                                                    kCheckWGs, (const float *)out_buf.map, 0, &mte, &mae);
                        if (bad != 0) ++fails;

                        if (ni == 0) t_tex_ns = run.med_ns;
                        const int norm_ok = run.clk_ok && alu_clk_ok[pr][si] && (t_tex_ns > 0.0);
                        double t_alu_est_ns = NAN, vs_max = NAN, vs_sum = NAN;
                        if (norm_ok) {
                            const double clk_scale = alu_sclk_mhz[pr][si] / run.sclk_avg_mhz;
                            t_alu_est_ns = alu_ns_per_inst_unit[pr][si] * clk_scale *
                                           ((double)groups / 1024.0) * (double)iters * (double)n;
                            const double t_max_ns = (t_tex_ns > t_alu_est_ns) ? t_tex_ns : t_alu_est_ns;
                            const double t_sum_ns = t_tex_ns + t_alu_est_ns;
                            vs_max = (t_max_ns > 0.0) ? (run.med_ns / t_max_ns) : NAN;
                            vs_sum = (t_sum_ns > 0.0) ? (run.med_ns / t_sum_ns) : NAN;
                        }

                        const uint64_t total_samp = (uint64_t)groups * 256ull * (uint64_t)iters * 4ull;
                        const double spc =
                            run.clk_ok ? vkt_samp_per_clk_per_cu(total_samp, run.med_ns, run.sclk_avg_mhz) : NAN;
                        const double gbs = vkt_gbs(total_samp, vkt_fmt_bpp[f], run.med_ns);

                        char talu_s[32], vmax_s[32], vsum_s[32], spc_s[32], sclk_s[32], gm_sclk_s[32];
                        vkt_fmt_clk_norm(talu_s, sizeof talu_s, norm_ok, "%.3f", t_alu_est_ns * 1e-6);
                        vkt_fmt_clk_norm(vmax_s, sizeof vmax_s, norm_ok, "%.3f", vs_max);
                        vkt_fmt_clk_norm(vsum_s, sizeof vsum_s, norm_ok, "%.3f", vs_sum);
                        vkt_fmt_clk_norm(spc_s, sizeof spc_s, run.clk_ok, "%.3f", spc);
                        vkt_fmt_clk_norm(sclk_s, sizeof sclk_s, run.clk_ok, "%.0fMHz", run.sclk_avg_mhz);
                        vkt_fmt_clk_norm(gm_sclk_s, sizeof gm_sclk_s,
                                         vkt_clk_gate_ok(run.gm_n, run.gm_sclk_avg_mhz), "%.0fMHz",
                                         run.gm_sclk_avg_mhz);
                        const char * spread = (run.hi_ns > run.lo_ns * 1.10) ? " SPREAD>10%" : "";
                        const char * clk_warn = norm_ok ? "" : " INSUFFICIENT_CLK_SAMPLES";

                        printf("RESULT vk_tex_fp16 fmt=%-19s lvl=%-4s prec=%s w=%2u N=%2u "
                               "gpu_ms=%.3f T_tex=%.3f T_alu_est=%s vs_max=%s vs_sum=%s "
                               "samp/clk/CU=%s GB/s=%.1f reps=%d sclk=%s clk_n=%d gm_sclk=%s gm_n=%d "
                               "uclk=%.0fMHz hot=%.0fC exact=%s%s%s\n",
                               vkt_fmt_names[f], vkt_lvl_names[lvl], pnames[pr], sgs[si], n,
                               run.med_ns * 1e-6, t_tex_ns * 1e-6, talu_s, vmax_s, vsum_s,
                               spc_s, gbs, run.reps, sclk_s, run.clk_n, gm_sclk_s, run.gm_n,
                               run.uclk_med_mhz, run.hot_max_c, bad == 0 ? "ok" : "FAIL", spread, clk_warn);
                        fflush(stdout);
                    }
                }
            }
            vkt_img_destroy(&im);
        }
    }

    vkt_buf_destroy(&out_buf);
    vkt_buf_destroy(&k_buf);
    for (int pr = 0; pr < 2; ++pr)
        for (int si = 0; si < 2; ++si) {
            for (int ni = 0; ni < kNumN; ++ni) vkt_pipe_destroy(&mix_pipes[pr][si][ni]);
            for (int ai = 0; ai < 3; ++ai) vkt_pipe_destroy(&alu_pipes[pr][si][ai]);
        }
    vkt_shutdown();

    printf("VERDICT vk_tex_fp16: %s (fails=%d)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
