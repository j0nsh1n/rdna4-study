// P1: lists of short compute dispatches. A condition is "key=value,..." with
//   name   label                          q      g (graphics family) | c (compute-only family)
//   bar    none | b2 (vkCmdPipelineBarrier2, compute write -> compute read/write) | legacy (vkCmdPipelineBarrier, the
//          same) | exec (execution dependency only, no memory access) | event (vkCmdSetEvent2 / vkCmdWaitEvents2) |
//          full (ALL_COMMANDS -> ALL_COMMANDS, memory read/write) | vv7 (vulkan-view item 7: vkCmdPipelineBarrier,
//          shader write -> shader write only)
//   grid   groups per dispatch            grid2  if set, odd positions use this grid (order: small-large-small)
//   local  threads per group (64)         sg     required subgroup size (0 = driver's choice)
//   body   cycles every wave spins        n      dispatches per list           reps  lists
//   ts     0: timestamps only at the list's start and end; 1: BOTTOM after each dispatch; 2: TOP before and BOTTOM
//          after each dispatch
//   stride stamp every stride-th group (-1: so that at most 64 groups per dispatch are stamped; 0: no stamps)
//   subs   0: one command buffer; 1: one submission per dispatch, submitted back to back, fence on the last;
//          2: one submission per dispatch, each waited for
//   check  1: dependency check (each dispatch reads what another group wrote at the end of the previous one)
//   gap    microseconds of host idle before each list          cal   1: the vulkan-view item 7 known answer
//   fit    name of a group of conditions whose list time is fitted against n
//   push   1: push constants before every dispatch (default); 0: once before the list (every dispatch then gets k = 1)
//   spv    shader file (default gchain.spv; another one-group shader, for example the study's one.spv, for the
//          calibration; it takes no push constants)
//   ts0    top (default) | bot: the list's first timestamp at TOP_OF_PIPE or BOTTOM_OF_PIPE (E4: vkstudy uses BOTTOM)
//   bar0   1: a barrier before the first dispatch too (E4: vkstudy has one before each of its 100 dispatches)
//   explore 1: the dependency check's outcome is reported (RESULT), not judged (E5: what each barrier kind guarantees)
//   pre    F4: cycles of a warm-up dispatch (64 groups, every wave spinning, no stamps) recorded in each list's command
//          buffer before the list's first timestamp, followed by a b2 barrier (0: none)
//   probe  F4: 1 = a one-group clock probe (2,000 cycles, stamped) just before the list's first timestamp and one just
//          after its last, each behind a b2 barrier; their clocks are the list's clock before and after
#define _GNU_SOURCE
#include <math.h>
#include <string.h>

#include "vkgame.h"

typedef struct {
    char name[64], bar[16], fit[32], spv[64];
    int push, q, grid, grid2, local, body, n, reps, ts, stride, subs, check, sg, cal, ts0bot, bar0, explore, probe;
    double gap_us, pre;
} Spec;

static void parse(const char *s, Spec *p) {
    char b[64];
    memset(p, 0, sizeof *p);
    kvs(s, "name", "chain", p->name, sizeof p->name);
    kvs(s, "bar", "b2", p->bar, sizeof p->bar);
    kvs(s, "fit", "", p->fit, sizeof p->fit);
    kvs(s, "spv", "gchain.spv", p->spv, sizeof p->spv);
    p->push = (int)kvd(s, "push", 1);
    p->q = !strcmp(kvs(s, "q", "g", b, sizeof b), "c");
    p->grid = (int)kvd(s, "grid", 64);
    p->grid2 = (int)kvd(s, "grid2", 0);
    p->local = (int)kvd(s, "local", 64);
    p->body = (int)kvd(s, "body", 3000);
    p->n = (int)kvd(s, "n", 20);
    p->reps = (int)kvd(s, "reps", 200);
    p->ts = (int)kvd(s, "ts", 2);
    p->stride = (int)kvd(s, "stride", -1);
    p->subs = (int)kvd(s, "subs", 0);
    p->check = (int)kvd(s, "check", 0);
    p->sg = (int)kvd(s, "sg", 0);
    p->cal = (int)kvd(s, "cal", 0);
    p->gap_us = kvd(s, "gap", 0);
    p->ts0bot = !strcmp(kvs(s, "ts0", "top", b, sizeof b), "bot");
    p->bar0 = (int)kvd(s, "bar0", 0);
    p->explore = (int)kvd(s, "explore", 0);
    p->pre = kvd(s, "pre", 0);
    p->probe = (int)kvd(s, "probe", 0);
    if (p->pre > 4e9) p->pre = 4e9;
    int gmax = p->grid2 > p->grid ? p->grid2 : p->grid;
    if (p->stride < 0) p->stride = gmax <= 64 ? 1 : (gmax + 63) / 64;
}

static int grid_at(const Spec *p, int k) { return (p->grid2 && (k & 1)) ? p->grid2 : p->grid; }

static Pipe g_pipes[8];
static int g_npipes;
static const Pipe *pipe_for(const char *spv, int local, int sg) {
    char name[96];
    snprintf(name, sizeof name, "%.*s_l%d%s%s", (int)strcspn(spv, "."), spv, local, sg ? "_sg" : "",
             sg ? (sg == 32 ? "32" : "64") : "");
    for (int i = 0; i < g_npipes; i++)
        if (!strcmp(g_pipes[i].name, name)) return &g_pipes[i];
    g_pipes[g_npipes] = mkcomp(spv, name, (uint32_t)local, (uint32_t)sg);
    return &g_pipes[g_npipes++];
}

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

// Before dispatch k (k >= 1).
static void barrier(VkCommandBuffer cb, const char *kind, VkEvent *ev, int k) {
    if (!strcmp(kind, "none")) return;
    if (!strcmp(kind, "legacy") || !strcmp(kind, "vv7")) {
        // vv7: vulkan-view item 7's exact barrier (shader write -> shader write only).
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = !strcmp(kind, "vv7") ? VK_ACCESS_SHADER_WRITE_BIT
                                                : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, NULL, 0, NULL);
        return;
    }
    VkMemoryBarrier2 mb;
    VkDependencyInfo di = dep_for(kind, &mb);
    if (!strcmp(kind, "event")) vkCmdWaitEvents2(cb, 1, &ev[k - 1], &di);
    else vkCmdPipelineBarrier2(cb, &di);
}

// One condition's results kept for the fit across conditions.
typedef struct {
    char fit[32], name[64];
    int n;
    double list_med_us;
} FitPt;
static FitPt g_fit[64];
static int g_nfit;

static uint64_t u64(const uint32_t *w) { return w[0] | (uint64_t)w[1] << 32; }

static int run_condition(const char *specstr) {
    Spec p;
    parse(specstr, &p);
    if (g_hot_stop || hotspot() >= 85.0) {
        printf("THERMAL not starting %s: hotspot %.0f C (limit 85) or the 88 C stop is set\n", p.name, hotspot());
        return 1;
    }
    if (!QU[p.q].valid) {
        printf("VERDICT vkgame queue: FAIL %s asks for a queue family this process did not create\n", p.name);
        return 1;
    }
    const int n = p.n, reps = p.reps, gmax = p.grid2 > p.grid ? p.grid2 : p.grid;
    const int S = p.stride ? (gmax + p.stride - 1) / p.stride : 0;
    const Pipe *pp = pipe_for(p.spv, p.local, p.sg);
    Qu *q = &QU[p.q];
    printf("COND %s spec=%s queue=%s subgroup=%u stamped_per_dispatch=%d\n", p.name, specstr, q->name, pp->sg, S);

    const size_t nrec = (size_t)reps * n * (S ? S : 1), nrec_all = nrec + (p.probe ? 2 * (size_t)reps : 0);
    if ((p.pre > 0 || p.probe) && p.subs) {
        printf("VERDICT vkgame chain_options: FAIL %s: pre and probe need one command buffer per list (subs=0)\n", p.name);
        return 1;
    }
    Buf rec = mkbuf(nrec_all * 16, 0), chk = mkbuf((size_t)gmax * p.local * 4, 0), err = mkbuf((size_t)reps * 4 + 64, 0);
    fill_zero(&rec);
    fill_zero(&err);
    VkDescriptorSet ds = mkset(&rec, &chk, &err, NULL);
    const uint32_t qper = 2 * n + 2;
    VkQueryPool qp = mkqp((uint32_t)reps * qper);
    VkEvent *ev = calloc(n, sizeof *ev);
    if (!strcmp(p.bar, "event"))
        for (int k = 0; k < n; k++) {
            VkEventCreateInfo eci = {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
            eci.flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT;
            VK_CHECK(vkCreateEvent(g_dev, &eci, NULL, &ev[k]));
        }
    const int ncb = p.subs ? reps * n : reps;
    VkCommandBuffer *cbs = calloc(ncb, sizeof *cbs);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = q->pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = (uint32_t)ncb;
    VK_CHECK(vkAllocateCommandBuffers(g_dev, &cai, cbs));

    for (int r = 0; r < reps; r++) {
        const uint32_t qb = (uint32_t)r * qper;
        for (int k = 0; k < n; k++) {
            VkCommandBuffer cb = p.subs ? cbs[r * n + k] : cbs[r];
            if (p.subs || k == 0) {
                cb_begin(cb);
                if (p.subs) {
                    vkCmdResetQueryPool(cb, qp, qb + 1 + 2 * k, 2);
                } else {
                    vkCmdResetQueryPool(cb, qp, qb, qper);
                    if (p.check) {
                        vkCmdFillBuffer(cb, chk.b, 0, VK_WHOLE_SIZE, 0);
                        VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.memoryBarrierCount = 1;
                        di.pMemoryBarriers = &mb;
                        vkCmdPipelineBarrier2(cb, &di);
                    }
                    if (!strcmp(p.bar, "event"))
                        for (int e = 0; e < n; e++) vkCmdResetEvent2(cb, ev[e], VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                }
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp->p);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
                if (!p.subs && p.pre > 0) {
                    // F4: the warm-up (64 groups spinning, no stamps, no check), then a barrier
                    uint32_t pw[7] = {0, (uint32_t)p.pre, 0, 0, 0, (uint32_t)(64 * p.local), (uint32_t)r};
                    vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pw, pw);
                    vkCmdDispatch(cb, 64, 1, 1);
                    barrier(cb, "b2", ev, 1);
                }
                if (!p.subs && p.probe) {
                    uint32_t pb[7] = {0, 2000, 1, (uint32_t)(nrec + 2 * (size_t)r), 0, (uint32_t)p.local, (uint32_t)r};
                    vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pb, pb);
                    vkCmdDispatch(cb, 1, 1, 1);
                    barrier(cb, "b2", ev, 1);
                }
                if (!p.subs) {
                    if (p.ts0bot) ts_bot(cb, qp, qb);
                    else ts_top(cb, qp, qb);
                }
            }
            if (!p.subs && k > 0) barrier(cb, p.bar, ev, k);
            if (!p.subs && k == 0 && p.bar0 && strcmp(p.bar, "event")) barrier(cb, p.bar, ev, 1);
            if (p.ts == 2 || p.subs) ts_top(cb, qp, qb + 1 + 2 * k);
            const int g = grid_at(&p, k);
            uint32_t pc[7] = {(uint32_t)k + 1, (uint32_t)p.body, (uint32_t)p.stride,
                              (uint32_t)(((size_t)r * n + k) * (S ? S : 1)), (uint32_t)p.check,
                              (uint32_t)(g * p.local), (uint32_t)r};
            if (p.push || k == 0) vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            vkCmdDispatch(cb, (uint32_t)g, 1, 1);
            if (p.ts >= 1 || p.subs) ts_bot(cb, qp, qb + 2 + 2 * k);
            if (!p.subs && !strcmp(p.bar, "event") && k < n - 1) {
                VkMemoryBarrier2 mb;
                VkDependencyInfo di = dep_for("event", &mb);
                vkCmdSetEvent2(cb, ev[k], &di);
            }
            if (p.subs || k == n - 1) {
                if (!p.subs) ts_bot(cb, qp, qb + qper - 1);
                if (!p.subs && p.probe) {
                    barrier(cb, "b2", ev, 1);
                    uint32_t pb[7] = {0, 2000, 1, (uint32_t)(nrec + 2 * (size_t)r + 1), 0, (uint32_t)p.local, (uint32_t)r};
                    vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pb, pb);
                    vkCmdDispatch(cb, 1, 1, 1);
                }
                VK_CHECK(vkEndCommandBuffer(cb));
            }
        }
    }

    double *tsub = calloc((size_t)reps * n, sizeof(double)), *tdone = calloc(reps, sizeof(double));
    VkFence f = mkfence();
    Map map;
    map_begin(&map);
    const double c0 = mono_ns();
    int done = 0;
    for (int r = 0; r < reps && !g_hot_stop; r++, done++) {
        if (p.gap_us > 0) until_ns(mono_ns() + p.gap_us * 1e3);
        if (!p.subs) {
            tsub[r * n] = mono_ns();
            submit(q, cbs[r], f);
            fwait(f);
        } else {
            for (int k = 0; k < n; k++) {
                tsub[r * n + k] = mono_ns();
                submit(q, cbs[r * n + k], (p.subs == 2 || k == n - 1) ? f : VK_NULL_HANDLE);
                if (p.subs == 2) fwait(f);
            }
            if (p.subs == 1) fwait(f);
        }
        tdone[r] = mono_ns();
    }
    const double c1 = mono_ns();
    map_end(&map);
    vkDestroyFence(g_dev, f, NULL);

    // ---- read back ----
    uint32_t *R = malloc(nrec_all * 16), *E = malloc((size_t)reps * 4 + 64);
    download(&rec, R, nrec_all * 16);
    download(&err, E, (size_t)reps * 4 + 64);
    const size_t nd = (size_t)done * n;
    uint64_t *top = calloc(nd, 8), *bot = calloc(nd, 8), *l0 = calloc(done, 8), *l1 = calloc(done, 8);
    uint64_t *fs = calloc(nd, 8), *ls = calloc(nd, 8), *fe = calloc(nd, 8), *le = calloc(nd, 8);
    double *clk = calloc(nd, sizeof(double)), *dc = calloc(nd, sizeof(double));
    int *nst = calloc(nd, sizeof(int));
    int missing = 0, misorder = 0;
    long errors = 0;
    for (int r = 0; r < done; r++) {
        const uint32_t qb = (uint32_t)r * qper;
        if (!p.subs) l0[r] = qget(qp, qb), l1[r] = qget(qp, qb + qper - 1);
        errors += E[r];
        for (int k = 0; k < n; k++) {
            const size_t d = (size_t)r * n + k;
            if (p.ts == 2 || p.subs) top[d] = qget(qp, qb + 1 + 2 * k);
            if (p.ts >= 1 || p.subs) bot[d] = qget(qp, qb + 2 + 2 * k);
            if (top[d] && bot[d] && bot[d] < top[d]) misorder++;
            const int g = grid_at(&p, k), sk = S ? (g + p.stride - 1) / p.stride : 0;
            double cl[256], cy[256];
            int nc = 0, ny = 0;
            fs[d] = UINT64_MAX;
            for (int s = 0; s < sk; s++) {
                const uint32_t *w = R + 4 * (d * S + s);
                uint64_t r0 = u64(w), r1 = r0 + w[2];
                if (!r0) {
                    missing++;
                    continue;
                }
                nst[d]++;
                if (r0 < fs[d]) fs[d] = r0;
                if (r0 > ls[d]) ls[d] = r0;
                if (!fe[d] || r1 < fe[d]) fe[d] = r1;
                if (r1 > le[d]) le[d] = r1;
                if (ny < 256) cy[ny++] = w[3];
                if (w[2] >= 50 && nc < 256) cl[nc++] = (double)w[3] / w[2] * 100.0;
            }
            if (!nst[d]) fs[d] = 0;
            clk[d] = nc ? pctl(cl, nc, .5) : NAN;
            dc[d] = ny ? pctl(cy, ny, .5) : NAN;
        }
    }

    // ---- derived quantities (us) ----
    const double tk = g_tsp / 1e3;  // us per tick (REALTIME and the timestamps share one 100 MHz clock: info mode)
    double *dur = malloc(nd * 8), *ramp = malloc(nd * 8), *drain = malloc(nd * 8), *gap = malloc(nd * 8),
           *tsd = malloc(nd * 8), *tsg = malloc(nd * 8), *lau = malloc(nd * 8), *tail = malloc(nd * 8);
    double *per = malloc(done * 8), *tot_rt = malloc(done * 8), *wall = malloc(done * 8), *s2s = malloc(done * 8),
           *s2top = malloc(done * 8), *gap0 = malloc(done * 8), *dur0 = malloc(done * 8), *clk0 = malloc(done * 8);
    for (size_t d = 0; d < nd; d++) {
        const int k = (int)(d % n);
        const int has = nst[d] > 0, hasn = k < n - 1 && nst[d + 1] > 0;
        dur[d] = has ? (double)(le[d] - fs[d]) * tk : NAN;
        ramp[d] = has ? (double)(ls[d] - fs[d]) * tk : NAN;
        drain[d] = has ? (double)(le[d] - fe[d]) * tk : NAN;
        gap[d] = has && hasn ? (double)(int64_t)(fs[d + 1] - le[d]) * tk : NAN;
        tsd[d] = top[d] && bot[d] ? (double)(int64_t)(bot[d] - top[d]) * tk : NAN;
        tsg[d] = k < n - 1 && bot[d] && top[d + 1] ? (double)(int64_t)(top[d + 1] - bot[d]) * tk : NAN;
        lau[d] = has && top[d] ? (double)(int64_t)(fs[d] - top[d]) * tk : NAN;
        tail[d] = has && bot[d] ? (double)(int64_t)(bot[d] - le[d]) * tk : NAN;
        if (has && le[d] < fs[d]) misorder++;
    }
    for (int r = 0; r < done; r++) {
        const size_t a = (size_t)r * n, z = a + n - 1;
        uint64_t b0 = p.subs ? top[a] : l0[r], b1 = p.subs ? bot[z] : l1[r];
        per[r] = b0 && b1 ? (double)(int64_t)(b1 - b0) * tk / n : NAN;
        tot_rt[r] = nst[a] && nst[z] ? (double)(le[z] - fs[a]) * tk : NAN;
        wall[r] = (tdone[r] - tsub[a]) / 1e3;
        s2s[r] = nst[a] ? (map_ns(&map, fs[a]) - tsub[a]) / 1e3 : NAN;
        s2top[r] = b0 ? (map_ns(&map, b0) - tsub[a]) / 1e3 : NAN;
        gap0[r] = gap[a];
        dur0[r] = dur[a];
        clk0[r] = clk[a];
    }
    map_report(p.name, &map);
    if (p.probe) {
        // F4: the clock just before the list's first timestamp and just after its last, per list
        double *cpre = malloc(done * 8), *cpost = malloc(done * 8);
        int lo = 0;
        for (int r = 0; r < done; r++) {
            const uint32_t *a = R + 4 * (nrec + 2 * (size_t)r), *b = a + 4;
            cpre[r] = a[2] ? (double)a[3] / a[2] * 100.0 : NAN;
            cpost[r] = b[2] ? (double)b[3] / b[2] * 100.0 : NAN;
            printf("PROBE %s list=%d clock_before_MHz=%.0f clock_after_MHz=%.0f list_ts_per_dispatch_us=%.3f\n", p.name, r,
                   cpre[r], cpost[r], per[r]);
            lo += !(cpre[r] >= 3000 && cpost[r] >= 3000);
        }
        dist(p.name, "probe_clock_before_list_MHz", cpre, done);
        dist(p.name, "probe_clock_after_list_MHz", cpost, done);
        if (p.pre > 0)
            verdict("warm_clock", lo == 0,
                    "%s: %d of %d lists with a clock probe below 3000 MHz before or after the list (the warm-up must hold the "
                    "high clock through the list)",
                    p.name, lo, done);
        free(cpre), free(cpost);
    }
    dist(p.name, "dispatch_dur_us(first_entry_to_last_exit)", dur, (int)nd);
    dist(p.name, "gap_to_next_us(next_first_entry_minus_last_exit)", gap, (int)nd);
    dist(p.name, "start_spread_us(last_entry_minus_first_entry)", ramp, (int)nd);
    dist(p.name, "drain_us(last_exit_minus_first_exit)", drain, (int)nd);
    dist(p.name, "ts_top_to_bottom_us", tsd, (int)nd);
    dist(p.name, "ts_bottom_to_next_top_us", tsg, (int)nd);
    dist(p.name, "top_ts_to_first_entry_us", lau, (int)nd);
    dist(p.name, "last_exit_to_bottom_ts_us", tail, (int)nd);
    dist(p.name, "clock_MHz(cycles/REALTIME_per_group)", clk, (int)nd);
    dist(p.name, "group_cycles(entry_to_exit)", dc, (int)nd);
    dist(p.name, "list_ts_per_dispatch_us", per, done);
    dist(p.name, "list_first_entry_to_last_exit_us", tot_rt, done);
    dist(p.name, "list_wall_submit_to_fence_us", wall, done);
    dist(p.name, "submit_to_first_entry_us(calibrated)", s2s, done);
    dist(p.name, "submit_to_list_top_ts_us(calibrated)", s2top, done);
    dist(p.name, "first_dispatch_dur_us", dur0, done);
    dist(p.name, "first_gap_us(k0_to_k1)", gap0, done);
    dist(p.name, "first_dispatch_clock_MHz", clk0, done);
    // Position in the list: the median over all lists of each quantity at each position.
    for (int k = 0; k < n; k++) {
        double a[4][4096];
        int m = 0;
        for (int r = 0; r < done && m < 4096; r++, m++) {
            const size_t d = (size_t)r * n + k;
            a[0][m] = dur[d], a[1][m] = gap[d], a[2][m] = clk[d], a[3][m] = tsg[d];
        }
        printf("POS %s k=%d grid=%d dur_us_p50=%.3f p95=%.3f gap_us_p50=%.3f p95=%.3f clock_MHz_p50=%.0f "
               "ts_bottom_to_next_top_us_p50=%.3f\n",
               p.name, k, grid_at(&p, k), pctl(a[0], m, .5), pctl(a[0], m, .95), pctl(a[1], m, .5), pctl(a[1], m, .95),
               pctl(a[2], m, .5), pctl(a[3], m, .5));
    }
    // Lists 0 to 2 (the first after the job's start) and the slowest list, dispatch by dispatch.
    int slow = 0;
    for (int r = 1; r < done; r++)
        if (isfinite(tot_rt[r]) && (!isfinite(tot_rt[slow]) || tot_rt[r] > tot_rt[slow])) slow = r;
    int show[4] = {0, 1, 2, slow};
    for (int i = 0; i < 4 && done; i++) {
        int r = show[i];
        if (r >= done) continue;
        const size_t a = (size_t)r * n;
        const double t_rel = (tsub[a] - c0) / 1e6;
        const Smp *sm = smp_at(tsub[a]);
        printf("TL %s list=%d%s t_since_condition_start_ms=%.2f wall_us=%.1f submit_to_first_entry_us=%.1f state: sclk=%d "
               "cur_gfxclk=%d busy=%d hotspot=%.0f power=%.0f\n",
               p.name, r, r == slow ? " (slowest)" : "", t_rel, wall[r], s2s[r], sm ? sm->sclk : -1,
               sm ? sm->cur_gfx : -1, sm ? sm->busy : -1, sm ? sm->hot : -1, sm ? sm->pw : -1);
        for (int k = 0; k < n; k++) {
            const size_t d = a + k;
            printf("TL %s list=%d k=%d entry_us=%.2f exit_us=%.2f dur=%.2f gap_next=%.2f spread=%.2f drain=%.2f "
                   "top=%.2f bottom=%.2f clock_MHz=%.0f stamped=%d\n",
                   p.name, r, k, nst[d] ? (double)(int64_t)(fs[d] - fs[a]) * tk : NAN,
                   nst[d] ? (double)(int64_t)(le[d] - fs[a]) * tk : NAN, dur[d], gap[d], ramp[d], drain[d],
                   top[d] && nst[a] ? (double)(int64_t)(top[d] - fs[a]) * tk : NAN,
                   bot[d] && nst[a] ? (double)(int64_t)(bot[d] - fs[a]) * tk : NAN, clk[d], nst[d]);
        }
    }
    // Outliers of the gap: above 1.5 x median + 1 us (the study's disturbance rule, with 1 us for the 64 cycles).
    {
        double med = pctl(gap, (int)nd, .5);
        int no = 0;
        for (size_t d = 0; d < nd; d++)
            if (isfinite(gap[d]) && gap[d] > 1.5 * fabs(med) + 1.0) {
                if (no < 12) {
                    const Smp *sm = smp_at(tsub[(d / n) * n]);
                    printf("OUTLIER %s list=%zu k=%zu gap_us=%.2f (median %.2f) dur_us=%.2f clock_MHz=%.0f "
                           "t_ms=%.2f busy=%d sclk=%d\n",
                           p.name, d / n, d % n, gap[d], med, dur[d], clk[d], (tsub[(d / n) * n] - c0) / 1e6,
                           sm ? sm->busy : -1, sm ? sm->sclk : -1);
                }
                no++;
            }
        printf("RESULT %s gap_outliers=%d of %d (above 1.5 x median + 1 us)\n", p.name, no, nfinite(gap, (int)nd));
    }
    state_line(p.name, c0, c1);
    printf("RESULT %s lists=%d dispatches=%zu list_ts_per_dispatch_us_p50=%.3f gap_us_p50=%.3f dur_us_p50=%.3f "
           "clock_MHz_p50=%.0f dependency_errors=%ld\n",
           p.name, done, nd, pctl(per, done, .5), pctl(gap, (int)nd, .5), pctl(dur, (int)nd, .5), pctl(clk, (int)nd, .5),
           errors);

    // ---- checks ----
    if (S) verdict("stamps", missing == 0, "%s: %d stamped groups missing of %zu", p.name, missing, nd * S);
    verdict("order", misorder == 0, "%s: %d dispatches with end before start (timestamps or REALTIME)", p.name, misorder);
    if (p.check && p.explore) {
        long lists_bad = 0;
        for (int r = 0; r < done; r++) lists_bad += E[r] != 0;
        printf("RESULT %s dependency_check barrier=%s queue=%s stale_reads=%ld lists_with_stale_reads=%ld of %d "
               "(gap p50 %.2f us: negative = the dispatches overlapped)\n",
               p.name, p.bar, q->name, errors, lists_bad, done, pctl(gap, (int)nd, .5));
    } else if (p.check)
        verdict("dependency", errors == 0,
                "%s: %ld stale reads in %d lists (each dispatch reads what another group of the previous one wrote; "
                "barrier %s)",
                p.name, errors, done, p.bar);
    if (p.body && S) {
        double m = pctl(dc, (int)nd, .5);
        verdict("body_length", m >= p.body && m <= p.body + 2000,
                "%s: median cycles from a stamped group's entry to its exit %.0f for a spin of %d cycles (limit %d..%d)",
                p.name, m, p.body, p.body, p.body + 2000);
    }
    if (p.cal) {
        // Not judged: in the development runs this list gave about 1.7 us while the study's own vkstudy, run the same
        // minute, gave 1.12 us; the script runs vkstudy as the calibration job.
        printf("INFO %s: %.3f us per dispatch (vulkan-view item 7: 1.13 us with vkstudy's own list)\n", p.name,
               pctl(per, done, .5));
    }
    if (p.fit[0] && g_nfit < 64) {
        FitPt *f = &g_fit[g_nfit++];
        snprintf(f->fit, sizeof f->fit, "%s", p.fit);
        snprintf(f->name, sizeof f->name, "%s", p.name);
        f->n = n;
        f->list_med_us = pctl(per, done, .5) * n;
    }

    // ---- raw files ----
    if (g_out[0]) {
        char path[900];
        char nm[160];
        snprintf(nm, sizeof nm, "chain-%s.csv", p.name);
        out_path(path, sizeof path, nm);
        FILE *fo = fopen(path, "w");
        const uint64_t ref = nst[0] ? fs[0] : (l0[0] ? l0[0] : top[0]);
        if (fo) {
            fprintf(fo, "cond,list,k,grid,t_submit_us,list_wall_us,top_tick,bottom_tick,first_entry_tick,last_entry_tick,"
                        "first_exit_tick,last_exit_tick,clock_MHz,stamped,list_top_tick,list_bottom_tick,errors\n");
            for (size_t d = 0; d < nd; d++) {
                const int r = (int)(d / n), k = (int)(d % n);
                fprintf(fo, "%s,%d,%d,%d,%.3f,%.3f,%lld,%lld,%lld,%lld,%lld,%lld,%.1f,%d,%lld,%lld,%u\n", p.name, r, k,
                        grid_at(&p, k), ((p.subs ? tsub[d] : tsub[(size_t)r * n]) - c0) / 1e3, wall[r],
                        top[d] ? (long long)(top[d] - ref) : -1, bot[d] ? (long long)(bot[d] - ref) : -1,
                        nst[d] ? (long long)(fs[d] - ref) : -1, nst[d] ? (long long)(ls[d] - ref) : -1,
                        nst[d] ? (long long)(fe[d] - ref) : -1, nst[d] ? (long long)(le[d] - ref) : -1, clk[d], nst[d],
                        l0[r] ? (long long)(l0[r] - ref) : -1, l1[r] ? (long long)(l1[r] - ref) : -1, E[r]);
            }
            fclose(fo);
        }
        snprintf(nm, sizeof nm, "chain-%s-groups.csv", p.name);
        out_path(path, sizeof path, nm);
        fo = S ? fopen(path, "w") : NULL;
        if (fo) {
            fprintf(fo, "cond,list,k,stamp,entry_tick,exit_tick,cycles\n");
            for (int i = 0; i < 4; i++) {
                int r = show[i];
                if (r >= done || (i == 3 && slow <= 2)) continue;
                for (int k = 0; k < n; k++)
                    for (int s = 0; s < S; s++) {
                        const uint32_t *w = R + 4 * (((size_t)r * n + k) * S + s);
                        if (u64(w))
                            fprintf(fo, "%s,%d,%d,%d,%lld,%lld,%u\n", p.name, r, k, s, (long long)(u64(w) - ref),
                                    (long long)(u64(w) + w[2] - ref), w[3]);
                    }
            }
            fclose(fo);
        }
    }

    vkFreeCommandBuffers(g_dev, q->pool, (uint32_t)ncb, cbs);
    if (!strcmp(p.bar, "event"))
        for (int k = 0; k < n; k++) vkDestroyEvent(g_dev, ev[k], NULL);
    vkDestroyQueryPool(g_dev, qp, NULL);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
    rmbuf(&rec), rmbuf(&chk), rmbuf(&err);
    free(cbs), free(ev), free(tsub), free(tdone), free(R), free(E), free(top), free(bot), free(l0), free(l1);
    free(fs), free(ls), free(fe), free(le), free(clk), free(dc), free(nst), free(dur), free(ramp), free(drain);
    free(gap), free(tsd), free(tsg), free(lau), free(tail), free(per), free(tot_rt), free(wall), free(s2s), free(s2top);
    free(gap0), free(dur0), free(clk0);
    return g_hot_stop ? 1 : 0;
}

int mode_chain(int argc, char **argv) {
    int rc = 0;
    for (int i = 0; i < argc && !rc; i++) rc = run_condition(argv[i]);
    // Least-squares fit of the list time against the number of dispatches, per fit group.
    for (int a = 0; a < g_nfit; a++) {
        int seen = 0;
        for (int b = 0; b < a; b++) seen |= !strcmp(g_fit[a].fit, g_fit[b].fit);
        if (seen) continue;
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        int m = 0;
        for (int b = a; b < g_nfit; b++)
            if (!strcmp(g_fit[a].fit, g_fit[b].fit)) {
                double x = g_fit[b].n, y = g_fit[b].list_med_us;
                sx += x, sy += y, sxx += x * x, sxy += x * y, syy += y * y, m++;
                printf("FITPT %s %s n=%d list_us=%.3f\n", g_fit[a].fit, g_fit[b].name, g_fit[b].n, y);
            }
        if (m < 3) continue;
        double sl = (m * sxy - sx * sy) / (m * sxx - sx * sx), ic = (sy - sl * sx) / m;
        double r2 = pow(m * sxy - sx * sy, 2) / ((m * sxx - sx * sx) * (m * syy - sy * sy));
        printf("FIT %s list_us = %.3f + %.3f x n (R^2 %.4f, %d points)\n", g_fit[a].fit, ic, sl, r2, m);
    }
    return rc;
}
