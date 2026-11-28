// Blind Spot #10 host runner: Vulkan RADV wave32 vs wave64 loads beside FP8 WMMA.
// Reuses src/vk/vkh.c (device selection, pipeline executable stats & ACO disassembly capture).
#define _GNU_SOURCE
#include "../vk/vkh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t iters;
    uint32_t tile_mask;
    uint32_t pad0;
    uint32_t pad1;
} PC;

static int count_substr(const char * s, const char * sub) {
    if (!s) return 0;
    int c = 0;
    size_t n = strlen(sub);
    while ((s = strstr(s, sub))) { c++; s += n; }
    return c;
}

static int parse_stat(const char * asm_text, const char * key) {
    if (!asm_text) return -1;
    char pat[128];
    snprintf(pat, sizeof pat, "# stat %s = ", key);
    const char * p = strstr(asm_text, pat);
    return p ? atoi(p + strlen(pat)) : -1;
}

static int run_cell(const char * spv, const char * label, const char * lvl, uint32_t req_sg,
                    const char * mode_name, uint32_t groups, uint32_t lsz,
                    uint32_t tile_mask, uint32_t iters, int do_break,
                    Buf * brec, Buf * ba, Buf * bb) {
    char pname[96];
    snprintf(pname, sizeof pname, "sg%u_%s_%s", req_sg, mode_name, label);
    Pipe p = pipe_create(spv, pname, lsz, req_sg, 1);
    bind_buffers(brec, &G.dummy, ba, bb);
    uint32_t nwaves = groups * (lsz / p.sg_size);
    double w_per_simd = (groups > 1u) ? ((double) nwaves / 128.0) : 1.0;

    int vgprs = parse_stat(p.asm_text, "VGPRs");
    int sg_simd = parse_stat(p.asm_text, "Subgroups per SIMD");
    int n_wmma = count_substr(p.asm_text, "v_wmma_f32_16x16x16_fp8_fp8");
    int n_tr = count_substr(p.asm_text, "load_tr");

    PC pc = {iters, tile_mask, 0, 0};
    double wall = 0;
    // 1 warm + 3 timed
    run_dispatch(&p, groups, &pc, sizeof pc, NULL, 0, &wall);
    double cyc_mm[3], disp_cyc_mm[3], gpu_ms_v[3], ghz_v[3];
    int bad = 0;
    // Expected: 8 WMMAs/iter * 16.0 (E4M3 1.0*1.0) = 128.0 * iters (sum across 4 accumulators)
    float expect = 128.0f * (float) iters + (do_break ? 1.0f : 0.0f);
    for (int r = 0; r < 3; r++) {
        double gns = run_dispatch(&p, groups, &pc, sizeof pc, NULL, 0, &wall);
        gpu_ms_v[r] = gns * 1e-6;
        uint64_t * rec = (uint64_t *) brec->map;
        double wcycs[2048];
        uint32_t lim = nwaves < 2048 ? nwaves : 2048;
        for (uint32_t w = 0; w < lim; w++) {
            uint32_t bits = (uint32_t) rec[w * 4 + 2];
            float got = 0;
            memcpy(&got, &bits, 4);
            if (fabsf(got - expect) > 0.5f) bad++;
            wcycs[w] = (double) rec[w * 4 + 0];
        }
        double med_cyc = median(wcycs, (int) lim);
        double med_rt_ns = (double) rec[1] * 10.0;
        double ghz = med_rt_ns > 0 ? (wcycs[0] / med_rt_ns) : 0.0;
        // Total WMMAs per SIMD = w_per_simd * 8 * iters
        double simd_mm = w_per_simd * 8.0 * (double) iters;
        cyc_mm[r] = med_cyc / simd_mm;
        disp_cyc_mm[r] = (gns * ghz) / simd_mm;
        ghz_v[r] = ghz;
    }
    double m = median(cyc_mm, 3), lo = vmin(cyc_mm, 3), hi = vmax(cyc_mm, 3);
    double dm = median(disp_cyc_mm, 3);
    printf("RESULT vk_ld_mat2 sg=%2u mode=%-7s w/simd=%.0f lvl=%-2s %-12s disp_cyc/mm=%.2f wave_cyc/mm=%.2f (%.2f-%.2f) gpu_ms=%.3f sclk=%.2fGHz VGPR=%d aco_sg=%d wmma=%d tr=%d exact=%s\n",
           p.sg_size, mode_name, w_per_simd, lvl, label, dm, m, lo, hi, median(gpu_ms_v, 3), median(ghz_v, 3),
           vgprs, sg_simd, n_wmma, n_tr, bad == 0 ? "ok" : "FAIL");
    fflush(stdout);
    pipe_destroy(&p);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    int do_break = (argc > 1 && !strcmp(argv[1], "--break"));
    vkh_init(0);
    print_clocks("vk_ld_mat2-start");

    const size_t kBufBytes = 2u << 20; // 2 MiB (8192 x 256-byte tiles)
    Buf brec = buf_create(4096 * 4 * sizeof(uint64_t), 1);
    Buf ba = buf_create(kBufBytes, 1);
    Buf bb = buf_create(kBufBytes, 1);
    memset(ba.map, 0x38, kBufBytes); // E4M3 1.0f
    memset(bb.map, 0x38, kBufBytes);

    int fail = 0;
    const uint32_t iters = 1024;
    // Brief SCLK warmup (~40 ms) so sg=32 and sg=64 measure at matched boosted SCLK
    if (!do_break) {
        Pipe wp = pipe_create("vk_lm_ref.spv", "warm", 256, 32, 1);
        bind_buffers(&brec, &G.dummy, &ba, &bb);
        PC wpc = {iters, 31u, 0u, 0u};
        double wall = 0;
        for (int w = 0; w < 120; w++) run_dispatch(&wp, 128u, &wpc, sizeof wpc, NULL, 0, &wall);
        pipe_destroy(&wp);
    }
    // Compare sg=32 (1w, card 8w/SIMD) against sg=64 (1w, card_4w = equal threads, card_8w = matched 8w/SIMD!)
    const struct { uint32_t sg; const char * mode; uint32_t groups; uint32_t lsz; } cfgs[] = {
        {32u, "one",     1u,   32u},
        {32u, "card_8w", 128u, 256u}, // 1024 wave32 = 8 waves/SIMD
        {64u, "one",     1u,   64u},
        {64u, "card_4w", 128u, 256u}, // 512 wave64 = 4 waves/SIMD (equal thread count)
        {64u, "card_8w", 256u, 256u}, // 1024 wave64 = 8 waves/SIMD (matched wave occupancy!)
    };
    for (size_t ci = 0; ci < sizeof(cfgs) / sizeof(cfgs[0]); ci++) {
        uint32_t sg = cfgs[ci].sg, groups = cfgs[ci].groups, lsz = cfgs[ci].lsz;
        const char * mname = cfgs[ci].mode;
        fail += run_cell("vk_lm_ref.spv", "ref", "RG", sg, mname, groups, lsz, 31u, iters, do_break, &brec, &ba, &bb);
        // L0: 8 KiB = 32 tiles (mask 31); L2: 1 MiB = 4096 tiles (mask 4095)
        fail += run_cell("vk_lm_k1.spv", "K1_L0", "L0", sg, mname, groups, lsz, 31u, iters, do_break, &brec, &ba, &bb);
        fail += run_cell("vk_lm_k8.spv", "K8_L0", "L0", sg, mname, groups, lsz, 31u, iters, do_break, &brec, &ba, &bb);
        fail += run_cell("vk_lm_k1.spv", "K1_L2", "L2", sg, mname, groups, lsz, 4095u, iters, do_break, &brec, &ba, &bb);
        fail += run_cell("vk_lm_k8.spv", "K8_L2", "L2", sg, mname, groups, lsz, 4095u, iters, do_break, &brec, &ba, &bb);
    }

    buf_destroy(&brec);
    buf_destroy(&ba);
    buf_destroy(&bb);
    print_clocks("vk_ld_mat2-end");
    printf("VERDICT vk_ld_mat2: %s (fail=%d%s)\n", fail == 0 ? "PASS" : "FAIL", fail,
           do_break ? ", deliberate break" : "");
    return fail == 0 ? 0 : 1;
}
