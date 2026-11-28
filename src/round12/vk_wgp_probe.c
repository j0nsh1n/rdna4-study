#define _GNU_SOURCE
#include "../vk/vkh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t mode;
    uint32_t k1;
    uint32_t k2;
    uint32_t reps;
} PC;

static void dump_exec_props(const Pipe *p) {
    PFN_vkGetPipelineExecutablePropertiesKHR gprops =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(G.dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR greps =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            G.dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gstats =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(G.dev, "vkGetPipelineExecutableStatisticsKHR");
    if (!gprops || !greps || !gstats) return;
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = p->pipe;
    uint32_t ne = 0;
    gprops(G.dev, &pi, &ne, NULL);
    VkPipelineExecutablePropertiesKHR ep = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (ne > 0) {
        uint32_t one = 1;
        gprops(G.dev, &pi, &one, &ep);
        printf("EXEC_PROPS %s: name=\"%s\" sg=%u\n", p->name, ep.name, ep.subgroupSize);
    }
    VkPipelineExecutableInfoKHR ei = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = p->pipe;
    ei.executableIndex = 0;
    uint32_t ns = 0;
    gstats(G.dev, &ei, &ns, NULL);
    VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof *st);
    for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    gstats(G.dev, &ei, &ns, st);
    printf("  STATS(%u):", ns);
    for (uint32_t i = 0; i < ns; i++) {
        if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR)
            printf(" [%s=%llu]", st[i].name, (unsigned long long)st[i].value.u64);
        else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR)
            printf(" [%s=%lld]", st[i].name, (long long)st[i].value.i64);
        else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR)
            printf(" [%s=%u]", st[i].name, st[i].value.b32);
        else
            printf(" [%s=%g]", st[i].name, st[i].value.f64);
    }
    printf("\n");
    free(st);
    uint32_t nr = 0;
    greps(G.dev, &ei, &nr, NULL);
    VkPipelineExecutableInternalRepresentationKHR *ir = calloc(nr, sizeof *ir);
    for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    greps(G.dev, &ei, &nr, ir);
    printf("  IR_NAMES(%u):", nr);
    for (uint32_t i = 0; i < nr; i++) printf(" \"%s\"(%zuB)", ir[i].name, ir[i].dataSize);
    printf("\n");
    free(ir);
}

static double run_bar_case(const char *spv, const char *label, uint32_t lsz, uint32_t req_sg,
                           Buf *bout, Buf *bin, Buf *bsink, int dump_props) {
    Pipe p = pipe_create(spv, label, lsz, req_sg, 1);
    if (dump_props) dump_exec_props(&p);
    bind_buffers(bout, &G.dummy, bin, bsink);
    uint32_t nwaves = lsz / p.sg_size;
    uint32_t reps = 7;
    PC pc = {1u, 64u, 320u, reps};
    // Warm up once + run timed
    double wall_ns = 0;
    run_dispatch(&p, 1, &pc, sizeof pc, NULL, 0, &wall_ns);
    run_dispatch(&p, 1, &pc, sizeof pc, NULL, 0, &wall_ns);
    uint64_t *res = (uint64_t *)bout->map;
    double slopes[128];
    int n = 0;
    for (uint32_t w = 0; w < nwaves; w++) {
        for (uint32_t r = 1; r < reps; r++) {
            uint64_t d1 = res[w * 24 + 2 * r];
            uint64_t d2 = res[w * 24 + 2 * r + 1];
            double sl = (double)(d2 - d1) / (double)(pc.k2 - pc.k1);
            if (n < 128) slopes[n++] = sl;
        }
    }
    double med = median(slopes, n), lo = vmin(slopes, n), hi = vmax(slopes, n);
    // Estimate shader clock from wave 0 rep 0
    uint64_t d1_0 = res[0], d2_0 = res[1], rt1_0 = res[16], rt2_0 = res[17];
    double ghz = (rt2_0 > rt1_0) ? ((double)(d2_0 - d1_0) / ((double)(rt2_0 - rt1_0) * 10.0)) : NAN;
    int bar_ins = 0;
    if (p.asm_text) {
        const char *q = p.asm_text;
        while ((q = strstr(q, "s_barrier"))) { bar_ins++; q += 9; }
    }
    printf("BARRIER %-26s lsz=%3u sg=%2u waves=%2u bar_ops=%d cyc/bar=%.2f (%.2f-%.2f) ~sclk=%.2fGHz\n",
           label, lsz, p.sg_size, nwaves, bar_ins, med, lo, hi, ghz);
    pipe_destroy(&p);
    return med;
}

int main(int argc, char **argv) {
    int do_break = (argc > 1 && !strcmp(argv[1], "--break"));
    vkh_init(0);
    print_clocks("start");

    Buf bout = buf_create(4096 * 24 * sizeof(uint64_t), 1);
    Buf bin = buf_create(1024 * sizeof(float), 1);
    Buf bsink = buf_create(4096 * sizeof(float), 1);
    float *fin = (float *)bin.map;
    for (int i = 0; i < 128; i++) fin[i] = 1.0f;

    double cal_fma = 0.0;
    // 1. Calibration: 16 independent FMAs on 1 wave (32 threads, sg=32)
    {
        Pipe p = pipe_create("bar_lds0.spv", "cal_fma_w32", 32, 32, 1);
        dump_exec_props(&p);
        bind_buffers(&bout, &G.dummy, &bin, &bsink);
        PC pc = {0u, 128u, 384u, 7u};
        double wall = 0;
        run_dispatch(&p, 1, &pc, sizeof pc, NULL, 0, &wall);
        run_dispatch(&p, 1, &pc, sizeof pc, NULL, 0, &wall);
        uint64_t *res = (uint64_t *)bout.map;
        double sl[16];
        for (int r = 1; r < 7; r++)
            sl[r - 1] = (double)(res[2 * r + 1] - res[2 * r]) / (double)((pc.k2 - pc.k1) * 16u);
        int dual = 0;
        if (p.asm_text && strstr(p.asm_text, "v_dual_fmac")) dual = 1;
        cal_fma = median(sl, 6);
        printf("CALIB fma_16x_w32: cyc/FMA=%.3f (%.3f-%.3f) dual_issue=%d\n",
               cal_fma, vmin(sl, 6), vmax(sl, 6), dual);
        pipe_destroy(&p);
    }

    // 2. Barrier cost sweep across workgroup sizes, subgroup sizes, and LDS allocations (0-48 KiB)
    double b_w1 = run_bar_case("bar_lds0.spv", "lds0_wg32_sg32", 32, 32, &bout, &bin, &bsink, 0);
    double b_w4_0 = run_bar_case("bar_lds0.spv", "lds0_wg128_sg32", 128, 32, &bout, &bin, &bsink, 0);
    double b_w4_16 = run_bar_case("bar_lds16k.spv", "lds16k_wg128_sg32", 128, 32, &bout, &bin, &bsink, 1);
    double b_w8_0 = run_bar_case("bar_lds0.spv", "lds0_wg256_sg32", 256, 32, &bout, &bin, &bsink, 0);
    double b_w8_16 = run_bar_case("bar_lds16k.spv", "lds16k_wg256_sg32", 256, 32, &bout, &bin, &bsink, 0);
    double b_w8_28 = run_bar_case("bar_lds28k.spv", "lds28k_wg256_sg32", 256, 32, &bout, &bin, &bsink, 0);
    double b_w8_32 = run_bar_case("bar_lds32k.spv", "lds32k_wg256_sg32", 256, 32, &bout, &bin, &bsink, 1);
    double b_w8_48 = run_bar_case("bar_lds48k.spv", "lds48k_wg256_sg32", 256, 32, &bout, &bin, &bsink, 0);
    double b_w4_64 = run_bar_case("bar_lds16k.spv", "lds16k_wg256_sg64", 256, 64, &bout, &bin, &bsink, 0);
    double b_w16_16 = run_bar_case("bar_lds16k.spv", "lds16k_wg512_sg32", 512, 32, &bout, &bin, &bsink, 0);
    (void)b_w4_0; (void)b_w4_16; (void)b_w8_0; (void)b_w8_16; (void)b_w8_28; (void)b_w8_48; (void)b_w4_64; (void)b_w16_16;

    // 3. Whole-card LDS 32-bit read bandwidth (2nd independent CU vs WGP test: >64 B/cyc/CU requires CU mode)
    double b_per_cyc_cu = 0.0;
    {
        Pipe p = pipe_create("bar_lds16k.spv", "lds_bw_16k_wg256_sg32", 256, 32, 1);
        bind_buffers(&bout, &G.dummy, &bin, &bsink);
        uint32_t groups = 128; // 2 groups/CU = 4 groups/WGP, 1024 waves total
        uint32_t iters = 2000;
        PC pc = {2u, iters, 0u, 1u};
        double wall = 0;
        run_dispatch(&p, groups, &pc, sizeof pc, NULL, 0, &wall); // warm
        double d_ns[5];
        for (int r = 0; r < 5; r++) d_ns[r] = run_dispatch(&p, groups, &pc, sizeof pc, NULL, 0, &wall);
        double med_ns = median(d_ns, 5);
        // 8 x 4-byte loads per thread per iter
        double total_bytes = (double)groups * 256.0 * (double)iters * 8.0 * 4.0;
        double tb_s = total_bytes / (med_ns * 1e3); // B / (ns * 1e3) = TB/s
        // Compute SCLK from wave-0 cycles/realtime and B/cyc/CU from whole-dispatch gpu_ms
        uint64_t *res = (uint64_t *)bout.map;
        double cyc_w0 = (double)res[0];
        double rt_ns = (double)res[1] * 10.0;
        double sclk_ghz = cyc_w0 / rt_ns;
        b_per_cyc_cu = (total_bytes / 64.0) / (med_ns * sclk_ghz);
        int ds_cnt = 0;
        if (p.asm_text) {
            const char *q = p.asm_text;
            while ((q = strstr(q, "ds_load"))) { ds_cnt++; q += 7; }
        }
        printf("LDS_BW_16k_wg256_sg32: ds_load_ops=%d gpu_ms=%.3f TB/s=%.2f (est sclk=%.2fGHz, %.1f B/cyc/CU)\n",
               ds_cnt, med_ns * 1e-6, tb_s, sclk_ghz, b_per_cyc_cu);
        pipe_destroy(&p);
    }

    print_clocks("end");
    // Excess over 1-wave baseline removes per-iteration loop/empty-barrier overhead:
    // CU mode: delta_8w ~27 cyc (<45) AND whole-card LDS >70 B/cyc/CU (WGP cap is 64 B/cyc/CU).
    // WGP mode: delta_8w >60 cyc AND whole-card LDS <=66 B/cyc/CU.
    double delta_8w = b_w8_32 - b_w1;
    const char *mode_guess = (delta_8w < 45.0 && b_per_cyc_cu > 70.0) ? "CU_MODE"
                           : ((delta_8w > 60.0 && b_per_cyc_cu <= 66.0) ? "WGP_MODE" : "AMBIGUOUS");
    // 16 FMAs/iter + ~15-cycle scalar loop branch = ~1.81 cyc/FMA (still rules out 100 MHz realtime clock ~0.09)
    int cal_ok = (cal_fma >= 0.45 && cal_fma <= 2.20);
    if (do_break) {
        // Deliberate break: assert inverted WGP-mode thresholds (must fail on CU-mode hardware)
        int break_pass = (delta_8w > 60.0 && b_per_cyc_cu <= 66.0);
        printf("VERDICT vk_wgp_probe: %s (deliberate break: asserted WGP_MODE delta>60 & BW<=66, got delta=%.2f BW=%.1f B/cyc/CU)\n",
               break_pass ? "PASS" : "FAIL", delta_8w, b_per_cyc_cu);
        buf_destroy(&bout); buf_destroy(&bin); buf_destroy(&bsink);
        return break_pass ? 0 : 1;
    }
    int ok = cal_ok && (strcmp(mode_guess, "AMBIGUOUS") != 0);
    printf("VERDICT vk_wgp_probe: %s inferred=%s (1-wave=%.2f, wg256_lds32k=%.2f, delta=%.2f cyc/bar, lds=%.1f B/cyc/CU, cal_fma=%.3f)\n",
           ok ? "PASS" : "FAIL", mode_guess, b_w1, b_w8_32, delta_8w, b_per_cyc_cu, cal_fma);
    buf_destroy(&bout); buf_destroy(&bin); buf_destroy(&bsink);
    return ok ? 0 : 1;
}
