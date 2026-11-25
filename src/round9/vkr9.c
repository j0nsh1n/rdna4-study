// Round 9, items C and D: Vulkan probes built on the gaming-round harness (src/vk/vkgame.c is included here, so its
// device set-up, 5 ms sysfs sampler, buffers and pipelines are the ones every earlier probe used; nothing in src/vk
// is edited).
//
//   cbar <spec>...   item C: does the data one dispatch wrote sit in L2 for the next dispatch, per barrier kind?
//   clk  <spec>...   item D: one dispatch of a fixed length at a fixed cadence, called and waited on; which core clock?
//
// C spec keys: name=, q=g|c, kind=none|exec|b2|legacy|vv7|full|event, order=normal|race|cold|fused, reps=20,
//   steps=1024, warm=150 (ms of r9_spin before the first repetition: the core clock leaves idle), walk=random|page (random: the LCG walk of round 9 item C; page: the increasing walk, next = slot + 1, in
//   address order, item C2). One command buffer per repetition: evict (reads 256 MiB so L2 and the memory-side cache hold nothing of
//   the region), b2 barrier, then
//     normal: producer writes the 4 MiB region, barrier of `kind`, consumer chases `steps` dependent loads through it
//     race:   consumer first, producer second, nothing between (the deliberate break: the consumer's pointer check
//             must fail, else the check cannot see stale data)
//     cold:   consumer chases a region the producer wrote once, before the loop (a pointer chain like the others, followed
//             by loaded links, as r9_cl2's cold cell does), evicted by the evict dispatch of every repetition (control: GDDR6)
//     fused:  one kernel writes the region and then chases it (control: L0 or L2)
// D spec keys: name=, q=g|c, ms=1.0, cad=16.7, dur=20, groups=256, local=64, chunk=4096, idle=1, calib=1.
//   Every wave works until REALTIME has passed its own start + ms; wave 0 of group 0 records (REALTIME, cycles) at every
//   chunk; the clock is cycles over REALTIME x 100 MHz. After the cadence run, calib=1 adds three back-to-back 30 ms
//   dispatches, whose clock is also taken from the queue's own timestamps (a second counter).
// Every producer write carries a stamp of its repetition in the upper 17 bits of dword 0 (the walk may stay the same in all
// repetitions); the consumer adds up the stamps of the values it loads, and a repetition is ok only if the final slot and the sum
// both match: stale data from the previous repetition fails the sum even when its links are the same.
// A cell also reports the core clock of its repetitions (cycles over REALTIME, from the chase itself) and fails if it moves by
// more than 10 % of its median (the first version's first Vulkan cell ran at 1.19 GHz for 14 repetitions, then at 3.2 GHz).
// Output: C prints ROW lines (one per repetition) and RESULT / CELL lines on stdout; D writes VKG_OUT/VKG_TAG-clk-<cell>.csv; scripts/round9/analyse_*.py
// read them. The card is only used through Vulkan; the memory clock is read from sysfs (read only).
#define _GNU_SOURCE
#define main vkgame_original_main
#include "../vk/vkgame.c"
#undef main

#define NSLOT 32768u
#define REG_BYTES (NSLOT * 128u)
#define EV_BYTES (256u << 20)
#define LCG_A 1664525u
#define MAXREC 1024u

static int uclk_now(void) {
    char p[600];
    snprintf(p, sizeof p, "%s/gpu_metrics", g_drm);
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    unsigned char b[64];
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    return n >= 60 ? (int)(b[58] | (b[59] << 8)) : -1;
}

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double med(double *v, int n) {
    if (n <= 0) return NAN;
    double *c = malloc(n * sizeof *c);
    memcpy(c, v, n * sizeof *c);
    qsort(c, n, sizeof *c, cmp_dbl);
    double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

// ---------------------------------------------------------------------------------------------------------------
// barrier kinds: the same recordings as vkgame_chain.c (static there, copied here), so "kind" means the same thing.

static VkDependencyInfo dep_for(const char *kind, VkMemoryBarrier2 *mb) {
    *mb = (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb->srcStageMask = mb->dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (!strcmp(kind, "b2") || !strcmp(kind, "event")) {
        mb->srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb->dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    } else if (!strcmp(kind, "full")) {
        mb->srcStageMask = mb->dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb->srcAccessMask = mb->dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = mb;
    return di;
}

static int known_kind(const char *k) {
    static const char *kinds[] = {"none", "exec", "b2", "legacy", "vv7", "full", "event"};
    for (unsigned i = 0; i < sizeof kinds / sizeof *kinds; i++)
        if (!strcmp(k, kinds[i])) return 1;
    return 0;
}

// `ev` is only used for kind event: the producer side sets it (set_event), the consumer side waits (barrier).
static void barrier(VkCommandBuffer cb, const char *kind, VkEvent ev) {
    if (!strcmp(kind, "none")) return;
    if (!strcmp(kind, "legacy") || !strcmp(kind, "vv7")) {
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = !strcmp(kind, "vv7") ? VK_ACCESS_SHADER_WRITE_BIT
                                                : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             NULL, 0, NULL);
        return;
    }
    VkMemoryBarrier2 mb;
    VkDependencyInfo di = dep_for(kind, &mb);
    if (!strcmp(kind, "event")) vkCmdWaitEvents2(cb, 1, &ev, &di);
    else vkCmdPipelineBarrier2(cb, &di);
}

static void set_event(VkCommandBuffer cb, const char *kind, VkEvent ev) {
    if (strcmp(kind, "event")) return;
    VkMemoryBarrier2 mb;
    VkDependencyInfo di = dep_for("event", &mb);
    vkCmdSetEvent2(cb, ev, &di);
}

// ---------------------------------------------------------------------------------------------------------------
// item C

static Pipe g_prod, g_chase, g_fused, g_evict;
static int g_cpipes;

static void c_pipes(void) {
    if (g_cpipes) return;
    g_prod = mkcomp("r9_prod.spv", "r9_prod", 256, 32);
    g_chase = mkcomp("r9_chase.spv", "r9_chase", 32, 32);
    g_fused = mkcomp("r9_fused.spv", "r9_fused", 256, 32);
    g_evict = mkcomp("r9_evict.spv", "r9_evict", 256, 32);
    g_cpipes = 1;
}

static void dispatch(VkCommandBuffer cb, const Pipe *p, VkDescriptorSet ds, const uint32_t *pc, uint32_t npc,
                     uint32_t groups) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, npc * 4, pc);
    vkCmdDispatch(cb, groups, 1, 1);
}

static Pipe g_spin;
static int g_spin_made;

// One r9_spin dispatch of `ms` milliseconds (every wave works until REALTIME has passed its start + ms), so the core clock has
// left idle when the first timed repetition of a cell starts. Its records are not used.
static void warm_up(Qu *q, double ms) {
    if (ms <= 0) return;
    const uint32_t groups = 256, local = 64, wpg = local / 32, waves = groups * wpg;
    if (!g_spin_made) g_spin = mkcomp("r9_spin.spv", "r9_spin", local, 32), g_spin_made = 1;
    Buf res = mkbuf((size_t)groups * local * 4, 0), rec = mkbuf(16u * (MAXREC + 2), 2), grp = mkbuf((size_t)waves * 32, 2);
    VkDescriptorSet ds = mkset(&res, &rec, &grp, NULL);
    VkCommandBuffer c = cb_alloc(q);
    VkFence f = mkfence();
    const uint64_t lim = (uint64_t)llround(ms * 1e5);
    const uint32_t pc[8] = {(uint32_t)lim, (uint32_t)(lim >> 32), MAXREC, 0u, 0u, wpg, 4096u, 0u};
    cb_begin(c);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_spin.p);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(c, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(c, groups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(c));
    submit(q, c, f);
    fwait(f);
    vkDestroyFence(g_dev, f, NULL);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &c);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
    rmbuf(&res), rmbuf(&rec), rmbuf(&grp);
}

// the stamp of a repetition: the upper 17 bits of dword 0 of every slot the producer of that repetition wrote
static uint32_t stamp_of(uint32_t salt) { return (salt >> 15) | 1u; }

static uint32_t lcg_final(uint32_t a, uint32_t start, uint32_t steps, uint32_t c) {
    uint32_t p = start;
    for (uint32_t i = 0; i < steps; i++) p = (a * p + c) & (NSLOT - 1);
    return p;
}

static int run_cbar(const char *spec) {
    char name[64], qs[8], kind[16], order[16];
    kvs(spec, "name", "c", name, sizeof name);
    kvs(spec, "q", "c", qs, sizeof qs);
    kvs(spec, "kind", "b2", kind, sizeof kind);
    kvs(spec, "order", "normal", order, sizeof order);
    char walk[16];
    kvs(spec, "walk", "random", walk, sizeof walk);
    const int page = !strcmp(walk, "page");
    const int reps = (int)kvd(spec, "reps", 20);
    const uint32_t steps = (uint32_t)kvd(spec, "steps", 1024);
    const double warm = kvd(spec, "warm", 150);
    Qu *q = &QU[!strcmp(qs, "c")];
    if (!q->valid || !known_kind(kind) || warm < 0 || warm > 500 || steps < 16 || steps > NSLOT || reps < 1 || reps > 4000 ||
        (strcmp(order, "normal") && strcmp(order, "race") && strcmp(order, "cold") && strcmp(order, "fused")) ||
        (!page && strcmp(walk, "random"))) {
        printf("VERDICT vkr9 cbar_spec: FAIL bad cell %s (queue %s valid=%d kind %s order %s)\n", spec, q->name, q->valid,
               kind, order);
        g_fail++;
        return 1;
    }
    if (g_hot_stop || hotspot() >= 70.0) {
        printf("NOT RUN %s: hotspot %.0f C\n", name, hotspot());
        return 1;
    }
    c_pipes();
    const int normal = !strcmp(order, "normal"), race = !strcmp(order, "race"), cold = !strcmp(order, "cold"),
              fused = !strcmp(order, "fused");
    Buf ra = mkbuf(REG_BYTES, 0), rb = mkbuf(REG_BYTES, 0), ev = mkbuf(EV_BYTES, 0), sink = mkbuf(4096, 0);
    Buf out = mkbuf(16u * (size_t)reps, 2);
    VkDescriptorSet dsa = mkset(&ra, &out, NULL, NULL), dsb = mkset(&rb, &out, NULL, NULL),
                    dse = mkset(&ev, NULL, &sink, NULL);
    VkCommandBuffer cb = cb_alloc(q);
    VkFence fence = mkfence();
    uint32_t *o = out.map;
    const uint32_t mask = NSLOT - 1;
    // the cold cell: the producer writes region B once, before the loop; every repetition's evict dispatch pushes it out
    const uint32_t cold_a = page ? 1u : LCG_A, cold_c = page ? 1u : 1013904223u, cold_salt = 0x1234567u;
    if (cold) {
        const uint32_t ppB[4] = {cold_a, cold_c, mask, cold_salt};
        cb_begin(cb);
        dispatch(cb, &g_prod, dsb, ppB, 4, NSLOT / 8);
        VK_CHECK(vkEndCommandBuffer(cb));
        submit(q, cb, fence);
        fwait(fence);
    }
    warm_up(q, warm);
    int uc0 = uclk_now();
    printf("CELL %s queue=%s kind=%s order=%s walk=%s reps=%d steps=%u warm_ms=%.0f uclk_start=%d hotspot=%.0f\n", name, q->name,
           kind, order, walk, reps, steps, warm, uc0, hotspot());
    double *cps = calloc(reps, sizeof *cps), *nps = calloc(reps, sizeof *nps), *clk = calloc(reps, sizeof *clk);
    int okn = 0, ncps = 0;
    for (int r = 0; r < reps; r++) {
        // walk=page: the increasing walk (next = slot + 1, whole 4 MiB region in address order). The data is the same every
        // repetition, so a stale region cannot be told from a fresh one by the pointer check; the race cell therefore
        // alternates the direction (next = slot - 1 on odd repetitions) so that its stale data differs from its fresh data.
        const uint32_t a = page ? 1u : LCG_A;
        const uint32_t c = page ? ((race && (r & 1)) ? mask : 1u) : 1013904223u + 2u * (uint32_t)r;
        const uint32_t salt = 0x5bd1e995u * (uint32_t)(r + 1);
        const uint32_t start = ((uint32_t)r * 7919u + 1u) & mask;
        VkEvent evt = VK_NULL_HANDLE;
        if (!strcmp(kind, "event")) {
            VkEventCreateInfo eci = {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
            eci.flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT;
            VK_CHECK(vkCreateEvent(g_dev, &eci, NULL, &evt));
        }
        for (int k = 0; k < 4; k++) o[4 * r + k] = 0xdeadbeefu;
        const uint32_t pe[2] = {EV_BYTES / 16, 512u * 256u};
        const uint32_t pp[4] = {a, c, mask, salt};
        const uint32_t pcA[8] = {start, steps, a, c, mask, 1u, (uint32_t)r, 0u};
        const uint32_t pcB[8] = {start, steps, cold_a, cold_c, mask, 1u, (uint32_t)r, 0u};
        const uint32_t pf[8] = {a, c, mask, salt, start, steps, (uint32_t)r, 0u};
        cb_begin(cb);
        dispatch(cb, &g_evict, dse, pe, 2, 512);
        barrier(cb, "b2", VK_NULL_HANDLE);
        if (normal) {
            dispatch(cb, &g_prod, dsa, pp, 4, NSLOT / 8);
            set_event(cb, kind, evt);
            barrier(cb, kind, evt);
            dispatch(cb, &g_chase, dsa, pcA, 8, 1);
        } else if (race) {
            dispatch(cb, &g_chase, dsa, pcA, 8, 1);
            dispatch(cb, &g_prod, dsa, pp, 4, NSLOT / 8);
        } else if (cold) {
            dispatch(cb, &g_chase, dsb, pcB, 8, 1);
        } else if (fused) {
            dispatch(cb, &g_fused, dsa, pf, 8, 1);
        }
        VK_CHECK(vkEndCommandBuffer(cb));
        submit(q, cb, fence);
        fwait(fence);
        if (evt) vkDestroyEvent(g_dev, evt, NULL);
        // out dwords: cycles (low 32 bits), sum of the stamps seen, REALTIME ticks, final slot
        const uint64_t cyc = o[4 * r];
        const uint32_t acc = o[4 * r + 1], rt = o[4 * r + 2], fin = o[4 * r + 3];
        const uint32_t exp = cold ? lcg_final(cold_a, start, steps, cold_c) : lcg_final(a, start, steps, c);
        const uint32_t expacc = steps * stamp_of(cold ? cold_salt : salt);
        const int written = o[4 * r] != 0xdeadbeefu || o[4 * r + 3] != 0xdeadbeefu;
        const int ok = written && fin == exp && acc == expacc;
        const double mhz = rt ? (double)cyc / rt * 100.0 : 0.0;
        printf("ROW c api=vulkan cell=%s queue=%s kind=%s order=%s walk=%s rep=%d cycles=%llu rt=%u final=%u expect=%u stamps=%u "
               "expect_stamps=%u ok=%d steps=%u clk_mhz=%.0f\n",
               name, q->name, kind, order, walk, r, (unsigned long long)cyc, rt, fin, exp, acc, expacc, ok, steps, mhz);
        if (r > 0 && written) {  // repetition 0 loads the code and builds the evict state: kept in the file, not counted
            cps[ncps] = (double)cyc / steps;
            nps[ncps] = (double)rt * 10.0 / steps;
            clk[ncps] = mhz;
            ncps++;
            okn += ok;
        }
    }
    int uc1 = uclk_now();
    double cmin = 1e30, cmax = 0;
    for (int i = 0; i < ncps; i++) cmin = fmin(cmin, clk[i]), cmax = fmax(cmax, clk[i]);
    const double cmed = med(clk, ncps);
    printf("RESULT vkr9 cbar name=%s queue=%s kind=%s order=%s reps=%d counted=%d cyc_per_step_med=%.1f ns_per_step_med=%.1f "
           "pointer_ok=%d/%d clk_mhz_med=%.0f min=%.0f max=%.0f uclk_start=%d uclk_end=%d\n",
           name, q->name, kind, order, reps, ncps, med(cps, ncps), med(nps, ncps), okn, ncps, cmed, ncps ? cmin : 0.0, cmax, uc0, uc1);
    verdict("cell_clock", ncps > 0 && (cmax - cmin) <= 0.10 * cmed,
            "%s: core clock of the counted repetitions %.0f-%.0f MHz, median %.0f (the spread must stay within 10 %% of the median)",
            name, ncps ? cmin : 0.0, cmax, cmed);
    free(cps), free(nps), free(clk);
    vkDestroyFence(g_dev, fence, NULL);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &cb);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &dsa);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &dsb);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &dse);
    rmbuf(&ra), rmbuf(&rb), rmbuf(&ev), rmbuf(&sink), rmbuf(&out);
    return 0;
}

static int mode_cbar(int argc, char **argv) {
    int rc = 0;
    for (int i = 0; i < argc && !g_hot_stop; i++) rc |= run_cbar(argv[i]);
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// item D

static long long real_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}


// Records one dispatch and returns after its fence; rec/grp are the host-visible result buffers.
static void d_row(FILE *csv, const char *kind, int idx, double tsub, double tdone, uint64_t ttop, uint64_t tbot,
                  const uint32_t *rec, const uint32_t *grp, uint32_t waves, uint64_t lim_ticks) {
    const uint32_t count = rec[0];
    const uint64_t rt0 = rec[4] | (uint64_t)rec[5] << 32, cy0 = rec[6] | (uint64_t)rec[7] << 32;
    uint64_t rt1 = rt0, cy1 = cy0, rtq = rt0, cyq = cy0, rt3 = rt0, cy3 = cy0;
    int gotq = 0, got3 = 0;
    for (uint32_t k = 1; k < count; k++) {
        const uint32_t *w = rec + 4 * (1 + k);
        const uint64_t r = w[0] | (uint64_t)w[1] << 32, c = w[2] | (uint64_t)w[3] << 32;
        if (!gotq && r - rt0 >= lim_ticks / 4) rtq = r, cyq = c, gotq = 1;
        if (!got3 && r - rt0 >= 3 * lim_ticks / 4) rt3 = r, cy3 = c, got3 = 1;
        rt1 = r, cy1 = c;
    }
    double wmin = 1e30, wmax = 0;
    for (uint32_t w = 0; w < waves; w++) {
        const uint32_t *a = grp + 8 * w, *b = a + 4;
        const uint64_t ra = a[0] | (uint64_t)a[1] << 32, ca = a[2] | (uint64_t)a[3] << 32;
        const uint64_t rb = b[0] | (uint64_t)b[1] << 32, cb2 = b[2] | (uint64_t)b[3] << 32;
        if (rb > ra) {
            const double mhz = (double)(cb2 - ca) / (double)(rb - ra) * 100.0;
            if (mhz < wmin) wmin = mhz;
            if (mhz > wmax) wmax = mhz;
        }
    }
    fprintf(csv, "%s,%d,%.0f,%.0f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%.1f,%.1f\n", kind, idx, tsub, tdone,
            (unsigned long long)ttop, (unsigned long long)tbot, (unsigned long long)rt0, (unsigned long long)cy0,
            (unsigned long long)rt1, (unsigned long long)cy1, (unsigned long long)rtq, (unsigned long long)cyq,
            (unsigned long long)rt3, (unsigned long long)cy3, count, wmin, wmax);
}

static int run_clk(const char *spec) {
    char name[64], qs[8];
    kvs(spec, "name", "d", name, sizeof name);
    kvs(spec, "q", "g", qs, sizeof qs);
    const double ms = kvd(spec, "ms", 1.0), cad = kvd(spec, "cad", 16.7), dur = kvd(spec, "dur", 20), idle = kvd(spec, "idle", 1);
    const uint32_t groups = (uint32_t)kvd(spec, "groups", 256), local = (uint32_t)kvd(spec, "local", 64),
                   chunk = (uint32_t)kvd(spec, "chunk", 4096);
    const int calib = (int)kvd(spec, "calib", 1);
    Qu *q = &QU[!strcmp(qs, "c")];
    // thermal rule: FMA-class load, planned busy share well under 0.40 (checked here), 20 s at most per cell
    if (!q->valid || ms < 0.05 || ms > 5.0 || cad < 4 * ms || cad < 5 || dur > 30 || groups < 1 || groups > 4096 ||
        local < 32 || local > 256 || local % 32) {
        printf("VERDICT vkr9 clk_spec: FAIL bad cell %s (busy share limit 0.25, dispatch 0.05-5 ms, at most 30 s)\n", spec);
        g_fail++;
        return 1;
    }
    if (g_hot_stop || hotspot() >= 58.0) {
        printf("NOT RUN %s: hotspot %.0f C (a cell starts below 58 C)\n", name, hotspot());
        return 1;
    }
    const uint32_t wpg = local / 32, waves = groups * wpg;
    Pipe sp = mkcomp("r9_spin.spv", "r9_spin", local, 32);
    Buf res = mkbuf((size_t)groups * local * 4, 0), rec = mkbuf(16u * (MAXREC + 2), 2), grp = mkbuf((size_t)waves * 32, 2);
    VkDescriptorSet ds = mkset(&res, &rec, &grp, NULL);
    VkQueryPool qp = mkqp(2);
    VkCommandBuffer cb = cb_alloc(q), cbc = cb_alloc(q);
    VkFence fence = mkfence();
    const uint64_t lim = (uint64_t)llround(ms * 1e5), limc = (uint64_t)llround(30.0 * 1e5);
    for (int pass = 0; pass < 2; pass++) {
        const uint64_t l = pass ? limc : lim;
        VkCommandBuffer c = pass ? cbc : cb;
        const uint32_t pc[8] = {(uint32_t)l, (uint32_t)(l >> 32), MAXREC, 0u, 0u, wpg, chunk, 0u};
        cb_begin(c);
        vkCmdResetQueryPool(c, qp, 0, 2);
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, sp.p);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(c, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        ts_top(c, qp, 0);
        vkCmdDispatch(c, groups, 1, 1);
        ts_bot(c, qp, 1);
        VK_CHECK(vkEndCommandBuffer(c));
    }
    const int nd = (int)floor(dur * 1000.0 / cad);
    char path[900];
    snprintf(path, sizeof path, "%s/%s-clk-%s.csv", g_out[0] ? g_out : ".", g_tag, name);
    FILE *csv = fopen(path, "w");
    if (!csv) {
        printf("VERDICT vkr9 clk_output: FAIL cannot write %s\n", path);
        g_fail++;
        return 1;
    }
    fprintf(csv, "# api=vulkan queue=%s ms=%.3f cad=%.3f groups=%u local=%u chunk=%u tsp_ns=%.6f\n", q->name, ms, cad, groups, local,
            chunk, g_tsp);
    fprintf(csv, "kind,idx,t_submit_mono_ns,t_done_mono_ns,ts_top,ts_bot,rt0,cyc0,rt1,cyc1,rtq1,cycq1,rtq3,cycq3,nrec,"
                 "wave_mhz_min,wave_mhz_max\n");
    printf("CELL %s api=vulkan queue=%s dispatch_ms=%.3f cadence_ms=%.3f groups=%u local=%u dispatches=%d uclk_start=%d "
           "hotspot=%.0f planned_busy_share=%.3f\n",
           name, q->name, ms, cad, groups, local, nd, uclk_now(), hotspot(), ms / cad);
    printf("INFO %s load_realtime_ns start=%lld (host CLOCK_REALTIME, for the sampler's clock if needed)\n", name, real_ns());
    until_ns(mono_ns() + idle * 1e9);
    const double t0 = mono_ns();
    int done = 0;
    uint32_t *recm = rec.map, *grpm = grp.map;
    double *lat = calloc(nd ? nd : 1, sizeof *lat);
    for (int i = 0; i < nd && !g_hot_stop; i++, done++) {
        until_ns(t0 + i * cad * 1e6);
        const double ts = mono_ns();
        submit(q, cb, fence);
        fwait(fence);
        const double td = mono_ns();
        lat[i] = (ts - (t0 + i * cad * 1e6)) / 1e3;
        d_row(csv, "call", i, ts, td, qget(qp, 0), qget(qp, 1), recm, grpm, waves, lim);
    }
    if (calib && !g_hot_stop)
        for (int i = 0; i < 3; i++) {
            const double ts = mono_ns();
            submit(q, cbc, fence);
            fwait(fence);
            const double td = mono_ns();
            d_row(csv, "calib", i, ts, td, qget(qp, 0), qget(qp, 1), recm, grpm, waves, limc);
        }
    const double t1 = mono_ns();
    fclose(csv);
    printf("INFO %s load_realtime_ns end=%lld mono_start=%.0f mono_end=%.0f\n", name, real_ns(), t0, t1);
    dist(name, "submission_late_us", lat, done);
    verdict("cadence", pctl(lat, done, .95) <= 200 && pctl(lat, done, 1) <= 1000,
            "%s: submissions late against the cadence p95 %.1f us (limit 200), largest %.1f us (limit 1000)", name,
            pctl(lat, done, .95), pctl(lat, done, 1));
    printf("RESULT vkr9 clk name=%s queue=%s dispatch_ms=%.3f dispatches=%d of %d uclk_end=%d\n", name, q->name, ms, done, nd,
           uclk_now());
    state_line(name, t0, t1);
    free(lat);
    vkDestroyFence(g_dev, fence, NULL);
    vkDestroyQueryPool(g_dev, qp, NULL);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &cb);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &cbc);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
    rmbuf(&res), rmbuf(&rec), rmbuf(&grp);
    return g_hot_stop;
}

static int mode_clk(int argc, char **argv) {
    int rc = 0;
    for (int i = 0; i < argc && !rc; i++) rc = run_clk(argv[i]);
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) {
        fprintf(stderr, "usage: vkr9 cbar <spec>... | clk <spec>...\n");
        return 2;
    }
    const char *o = getenv("VKG_OUT"), *s = getenv("VKG_SPV"), *t = getenv("VKG_TAG");
    snprintf(g_out, sizeof g_out, "%s", o ? o : "");
    snprintf(g_spv, sizeof g_spv, "%s", s ? s : "build/round9/spv");
    snprintf(g_tag, sizeof g_tag, "%s", t ? t : argv[1]);
    dev_init(0);
    sampler_start();
    const double t0 = mono_ns();
    printf("START hotspot %.0f C\n", hotspot());
    int rc = 2;
    if (!strcmp(argv[1], "cbar")) rc = mode_cbar(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "clk")) rc = mode_clk(argc - 2, argv + 2);
    else fprintf(stderr, "unknown mode %s\n", argv[1]);
    vkDeviceWaitIdle(g_dev);
    const double t1 = mono_ns();
    sampler_stop_write();
    state_line("whole-job", t0, t1);
    printf("THERMAL highest %.0f C (5 ms samples); end %.0f C; hot_stop=%d\n", g_hottest, hotspot(), g_hot_stop);
    if (g_hot_stop) verdict("thermal", 0, "hotspot reached %.0f C; load stopped", (double)g_hot_stop_c);
    if (rc) return rc;
    return g_fail ? 1 : 0;
}
