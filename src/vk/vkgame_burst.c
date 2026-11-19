// E1-E3 (and P4's bursts after review): the dense FMA load of gdense.comp, either as one dispatch after an idle time
// (a ramp) or as bursts at a fixed cadence. Every wave works until REALTIME has passed its own start + the burst
// length, so every group is busy for the whole burst whatever the clock; subgroup 0 of every group records its start
// and end, and subgroup 0 of every recstride-th group records (REALTIME, cycles) at every check. Raw samples go to
// CSV files; scripts/gprobe_dense.py computes the clocks and the verdicts from them.
//   burst name=..,q=g|c,groups=64,local=32,every=64,idle=1,cont=60          one dispatch of 60 ms after 1 s idle
//   burst name=..,q=g|c,groups=..,local=..,blen=2,cad=16.7|gap=..,dur=1,idle=1   bursts of 2 ms
//   options: recstride=8 (0: no timeline), stopat=MS (the host sets the stop word MS after the first submission:
//   the tripwire's known answer), skew=1 (deliberate break: odd groups stop after a tenth of the burst), minfill=F
//   (the analysis judges every dispatch's fill, FMA per SIMD per cycle over every wave, against F; 0: not judged)
// Several arms may follow one another in one process (each with its own idle time).
// Thermal rule: the load is FMA-class. A planned busy share above 0.40 caps the arm at 0.1 s; at or below, at 1 s.
// The sampler sets the stop word at 88 C, which ends a running dispatch within one check (a few microseconds).
// A host-side guard (second fix round) reads the bursts' timestamps while the arm runs and, once 100 ms have run with
// a measured busy share above 0.40, stops submitting and sets the stop word: the cap follows measured times, not only
// the planned share.
#define _GNU_SOURCE
#include <math.h>
#include <string.h>
#include <time.h>

#include "vkgame.h"

typedef struct {
    char name[64];
    int q;
    uint32_t groups, local, every, recstride, skew, recevery;
    double blen_ms, cad_ms, dur_s, idle_s, cont_ms, stopat_ms, minfill;
} Arm;

static long long real_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Both timestamps of a burst without waiting; 0 when the burst has not finished.
static int burst_ticks(VkQueryPool qp, int i, uint64_t *t, uint64_t *b) {
    uint64_t v[4];
    if (vkGetQueryPoolResults(g_dev, qp, (uint32_t)(2 * i), 2, sizeof v, v, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS ||
        !v[1] || !v[3])
        return 0;
    *t = v[0], *b = v[2];
    return 1;
}

static int g_not_run;

// The stop word in each of the 4,096 lines the shader reads from (gdense.comp, Ctl).
void stop_all(volatile uint32_t *w) {
    for (int i = 0; i < 4096; i++) w[i * 32] = 1;
}
static Pipe g_dense[2];  // [0]: local 32 x n (built per local size below), cached by local size
static uint32_t g_dense_local[2];

static const Pipe *dense_pipe(uint32_t local) {
    for (int i = 0; i < 2; i++)
        if (g_dense_local[i] == local) return &g_dense[i];
    for (int i = 0; i < 2; i++)
        if (!g_dense_local[i]) {
            char nm[64];
            snprintf(nm, sizeof nm, "gdense_l%u_w32", local);
            g_dense[i] = mkcomp("gdense.spv", nm, local, 32);
            g_dense_local[i] = local;
            return &g_dense[i];
        }
    printf("VERDICT vkgame dense_pipeline: FAIL more than two local sizes in one process\n");
    exit(2);
}

static void parse_arm(const char *s, Arm *a) {
    char b[16];
    memset(a, 0, sizeof *a);
    kvs(s, "name", "dense", a->name, sizeof a->name);
    a->q = !strcmp(kvs(s, "q", "g", b, sizeof b), "c");
    a->groups = (uint32_t)kvd(s, "groups", 64);
    a->local = (uint32_t)kvd(s, "local", 32);
    a->every = (uint32_t)kvd(s, "every", 64);
    a->recstride = (uint32_t)kvd(s, "recstride", 8);
    a->skew = (uint32_t)kvd(s, "skew", 0);
    a->recevery = (uint32_t)kvd(s, "recevery", 1);
    if (!a->recevery) a->recevery = 1;
    a->blen_ms = kvd(s, "blen", 2);
    a->cont_ms = kvd(s, "cont", 0);
    const double gap = kvd(s, "gap", -1);
    a->cad_ms = gap >= 0 ? a->blen_ms + gap : kvd(s, "cad", 16.7);
    a->dur_s = kvd(s, "dur", 1);
    a->idle_s = kvd(s, "idle", 1);
    a->stopat_ms = kvd(s, "stopat", 0);
    a->minfill = kvd(s, "minfill", 0);
}

static void csv_name(char *path, size_t cap, const char *arm, const char *what) {
    char nm[200];
    snprintf(nm, sizeof nm, "dense-%s-%s.csv", arm, what);
    out_path(path, cap, nm);
}

static int run_arm(const char *spec) {
    Arm a;
    parse_arm(spec, &a);
    if (a.cont_ms > 100) a.cont_ms = 100;  // FMA-class continuous load: at most 0.1 s
    const double planned = a.cont_ms > 0 ? 1.0 : a.blen_ms / a.cad_ms;
    double dur = a.dur_s;
    // At 0.40 planned the measured share comes out a little above (launch and drain): the 0.1 s cap applies from 0.40.
    if (planned >= 0.40 && dur > 0.1) dur = 0.1;
    if (dur > 1.0) dur = 1.0;
    // the last burst ends by dur: (nb - 1) cadences plus one burst
    const int nb = a.cont_ms > 0 ? 1 : (int)floor((dur * 1000.0 - a.blen_ms) / a.cad_ms + 1e-9) + 1;
    const double run_ms = a.cont_ms > 0 ? a.cont_ms : a.blen_ms;
    const uint32_t nrecw = a.recstride ? (a.groups + a.recstride - 1) / a.recstride : 0;
    // a wave that the SIMD's arbitration favours checks about every 0.7 us at every=8 (every 5 us at every=64)
    const uint32_t maxrec = (uint32_t)(run_ms * 1000.0 * (a.every * a.recevery < 32 ? 2.0 : 0.5)) + 64;
    Qu *q = &QU[a.q];
    if (!q->valid) {
        printf("VERDICT vkgame queue: FAIL arm %s asks for the %s queue, which this process did not create\n", a.name,
               a.q ? "compute" : "graphics");
        return 1;
    }
    const Pipe *pp = dense_pipe(a.local);
    printf("ARM %s spec=%s queue=%s groups=%u local=%u subgroup=%u waves=%u every=%u rec_every=%u burst_ms=%.3f cadence_ms=%.3f "
           "planned_busy=%.3f dur_s=%.3f bursts=%d recorders=%u maxrec=%u idle_s=%.2f min_fill=%.2f\n",
           a.name, spec, q->name, a.groups, a.local, pp->sg, a.groups * (a.local / (pp->sg ? pp->sg : 32)), a.every, a.recevery,
           run_ms, a.cont_ms > 0 ? 0 : a.cad_ms, planned, a.cont_ms > 0 ? a.cont_ms / 1e3 : dur, nb, nrecw, maxrec,
           a.idle_s, a.minfill);
    if (g_hot_stop || hotspot() >= 58.0) {
        printf("NOT RUN %s: hotspot %.0f C (an arm starts below 58 C)\n", a.name, hotspot());
        g_not_run++;
        return g_hot_stop ? 1 : 0;
    }
    // every dispatch writes its own per-wave slice (gdense.comp), so every burst's fill is kept
    Buf res = mkbuf(((size_t)nb * a.groups * a.local + 128) * 4, 2);
    float *ab = res.map;
    for (int i = 0; i < 64; i++) ab[2 * i] = 0.999f, ab[2 * i + 1] = 0.001f;
    Buf rec = mkbuf((size_t)nb * (nrecw ? nrecw : 1) * maxrec * 16, 0);
    Buf grp = mkbuf((size_t)nb * a.groups * 48, 0);
    Buf ctl = mkbuf(4096 * 128, 1);
    fill_zero(&rec);
    fill_zero(&grp);
    volatile uint32_t *stopw = ctl.map;
    for (int i = 0; i < 4096 * 32; i++) stopw[i] = 0;
    VkDescriptorSet ds = mkset(&res, &rec, &grp, &ctl);
    VkQueryPool qp = mkqp((uint32_t)nb * 2);
    VkCommandBuffer *cbs = calloc(nb, sizeof *cbs);
    const uint64_t lim = (uint64_t)llround(run_ms * 1e5);  // REALTIME runs at 100 MHz
    for (int i = 0; i < nb; i++) {
        cbs[i] = cb_alloc(q);
        cb_begin(cbs[i]);
        vkCmdResetQueryPool(cbs[i], qp, 2 * i, 2);
        vkCmdBindPipeline(cbs[i], VK_PIPELINE_BIND_POINT_COMPUTE, pp->p);
        vkCmdBindDescriptorSets(cbs[i], VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
        uint32_t pc[10] = {0x7fffffffu, a.every, (uint32_t)lim, (uint32_t)(lim >> 32), a.recstride,
                           (uint32_t)((size_t)i * (nrecw ? nrecw : 1) * maxrec), maxrec, (uint32_t)i * a.groups, a.skew,
                           a.recevery};
        vkCmdPushConstants(cbs[i], g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        ts_top(cbs[i], qp, 2 * i);
        vkCmdDispatch(cbs[i], a.groups, 1, 1);
        ts_bot(cbs[i], qp, 2 * i + 1);
        VK_CHECK(vkEndCommandBuffer(cbs[i]));
    }
    double *tsub = calloc(nb, sizeof(double));
    // Idle, then the arm. The stop word is armed for the sampler's 88 C tripwire during the arm only.
    until_ns(mono_ns() + a.idle_s * 1e9);
    Map m;
    map_begin(&m);
    if (g_hot_rel > 0) {
        g_hot_stop_c = fmin(HOT_STOP_LIMIT_C, hotspot() + g_hot_rel);
        printf("INFO %s hotspot stop limit for this arm %.0f C (start hotspot + %.0f; VKG_HOT_STOP_C)\n", a.name,
               g_hot_stop_c, g_hot_rel);
    }
    g_stop_word = stopw;
    const long long rt_start = real_ns();
    const double t0 = mono_ns();
    int done = 0, forced = 0, guard = 0, gchk = 0;
    uint64_t gfirst = 0, glast = 0;
    double gbusy = 0;
    for (int i = 0; i < nb && !g_hot_stop; i++, done++) {
        // the busy-share guard over the bursts that have finished so far
        uint64_t bt, bb;
        while (gchk < i && burst_ticks(qp, gchk, &bt, &bb)) {
            if (!gfirst) gfirst = bt;
            glast = bb;
            gbusy += (double)(bb - bt);
            gchk++;
        }
        const double gspan_ms = glast > gfirst ? (glast - gfirst) * g_tsp / 1e6 : 0;
        if (gspan_ms >= 100.0 && gbusy / ((glast - gfirst) + 1e-9) > 0.40) {
            stop_all(stopw);
            guard = 1;
            printf("DUTY GUARD %s: measured busy share %.3f over %.1f ms after %d bursts: no further submissions, stop "
                   "word set\n",
                   a.name, gbusy / (glast - gfirst), gspan_ms, gchk);
            break;
        }
        until_ns(t0 + i * a.cad_ms * 1e6);
        tsub[i] = mono_ns();
        submit(q, cbs[i], VK_NULL_HANDLE);
    }
    if (a.stopat_ms > 0) {
        until_ns(t0 + a.stopat_ms * 1e6);
        stop_all(stopw);
        forced = 1;
        printf("INFO %s stop word set by the host %.3f ms after the first submission (known-answer test)\n", a.name,
               (mono_ns() - t0) / 1e6);
    }
    VK_CHECK(vkQueueWaitIdle(q->q));
    g_stop_word = NULL;
    const double t1 = mono_ns();
    const long long rt_end = real_ns();
    if (g_hot_rel > 0) g_hot_stop_c = HOT_STOP_LIMIT_C;
    printf("INFO %s load_realtime_ns start=%lld end=%lld (host CLOCK_REALTIME: first submission, queue idle)\n", a.name,
           rt_start, rt_end);
    map_end(&m);
    map_report(a.name, &m);
    const int stopped = *stopw != 0;

    // raw files
    uint32_t *R = NULL, *G = malloc((size_t)nb * a.groups * 48);
    download(&grp, G, (size_t)nb * a.groups * 48);
    if (nrecw) {
        R = malloc((size_t)nb * nrecw * maxrec * 16);
        download(&rec, R, (size_t)nb * nrecw * maxrec * 16);
    }
    char path[900];
    FILE *f;
    csv_name(path, sizeof path, a.name, "map");
    if ((f = fopen(path, "w"))) {
        fprintf(f, "d0,m0,d1,m1\n%llu,%.0f,%llu,%.0f\n", (unsigned long long)m.d0, m.m0, (unsigned long long)m.d1, m.m1);
        fclose(f);
    }
    double busy = 0, span0 = INFINITY, span1 = 0;
    csv_name(path, sizeof path, a.name, "bursts");
    if ((f = fopen(path, "w"))) {
        fprintf(f, "burst,t_submit_mono_ns,top_tick,bottom_tick\n");
        for (int i = 0; i < done; i++) {
            const uint64_t t = qget(qp, 2 * i), b = qget(qp, 2 * i + 1);
            fprintf(f, "%d,%.0f,%llu,%llu\n", i, tsub[i], (unsigned long long)t, (unsigned long long)b);
            busy += (double)(b - t);
            if (t < span0) span0 = (double)t;
            if (b > span1) span1 = (double)b;
        }
        fclose(f);
    }
    // every wave's chunk count of every dispatch (lane 0 of each wave writes it; the buffer is host-visible)
    csv_name(path, sizeof path, a.name, "waves");
    if ((f = fopen(path, "w"))) {
        const uint32_t *wv = (const uint32_t *)res.map + 128, wpg = a.local / 32;
        fprintf(f, "burst,group,wave,chunks\n");
        for (int i = 0; i < done; i++)
            for (uint32_t g = 0; g < a.groups; g++)
                for (uint32_t w = 0; w < wpg; w++)
                    fprintf(f, "%d,%u,%u,%u\n", i, g, w, wv[((size_t)i * a.groups + g) * a.local + w * 32]);
        fclose(f);
    }
    csv_name(path, sizeof path, a.name, "groups");
    if ((f = fopen(path, "w"))) {
        fprintf(f, "burst,group,start_rt,start_cyc,end_rt,end_cyc,chunks,records\n");
        for (int i = 0; i < done; i++)
            for (uint32_t g = 0; g < a.groups; g++) {
                const uint32_t *w = G + ((size_t)i * a.groups + g) * 12;
                fprintf(f, "%d,%u,%llu,%llu,%llu,%llu,%u,%u\n", i, g, (unsigned long long)(w[0] | (uint64_t)w[1] << 32),
                        (unsigned long long)(w[2] | (uint64_t)w[3] << 32),
                        (unsigned long long)(w[4] | (uint64_t)w[5] << 32),
                        (unsigned long long)(w[6] | (uint64_t)w[7] << 32), w[8], w[9]);
            }
        fclose(f);
    }
    if (nrecw) {
        csv_name(path, sizeof path, a.name, "rec");
        if ((f = fopen(path, "w"))) {
            fprintf(f, "burst,recorder,group,i,rt,cyc\n");
            for (int i = 0; i < done; i++)
                for (uint32_t r = 0; r < nrecw; r++) {
                    const uint32_t *w = R + ((size_t)i * nrecw + r) * maxrec * 4;
                    for (uint32_t k = 0; k < maxrec && (w[4 * k] | w[4 * k + 1]); k++)
                        fprintf(f, "%d,%u,%u,%u,%llu,%llu\n", i, r, r * a.recstride, k,
                                (unsigned long long)(w[4 * k] | (uint64_t)w[4 * k + 1] << 32),
                                (unsigned long long)(w[4 * k + 2] | (uint64_t)w[4 * k + 3] << 32));
                }
            fclose(f);
        }
    }
    const double wall_ms = (span1 - span0) * g_tsp / 1e6, share = busy / (span1 - span0 + 1e-9);
    printf("DUTY %s bursts=%d measured_busy_share=%.3f (GPU time between each burst's timestamps over the span from the "
           "first TOP to the last BOTTOM, %.1f ms) planned=%.3f\n",
           a.name, done, share, wall_ms, planned);
    if (a.cont_ms <= 0) {
        verdict("duty_rule", share <= 0.40 || wall_ms <= 100.5,
                "%s: measured busy share %.3f over %.1f ms (rule: above 0.40 at most 0.1 s)", a.name, share, wall_ms);
        // the host kept the cadence: submission time against the plan
        double *err = calloc(done ? done : 1, sizeof(double));
        for (int i = 0; i < done; i++) err[i] = (tsub[i] - (t0 + i * a.cad_ms * 1e6)) / 1e3;
        dist(a.name, "submission_late_us", err, done);
        const double p95 = pctl(err, done, .95), mx = pctl(err, done, 1);
        verdict("cadence", p95 <= 100 && mx <= 500,
                "%s: submissions late against the planned cadence p95 %.1f us (limit 100), largest %.1f us (limit 500)",
                a.name, p95, mx);
        free(err);
    }
    printf("INFO %s stop word at the end: %d (%s)\n", a.name, stopped,
           forced  ? "set by the host on purpose"
           : guard ? "set by the busy-share guard"
           : stopped ? "set by the sampler's hotspot tripwire"
                     : "never set");
    state_line(a.name, t0, t1);
    vkFreeCommandBuffers(g_dev, q->pool, (uint32_t)nb, cbs);
    vkDestroyQueryPool(g_dev, qp, NULL);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
    rmbuf(&res), rmbuf(&rec), rmbuf(&grp), rmbuf(&ctl);
    free(cbs), free(tsub), free(R), free(G);
    // a stop at a lowered limit (F5) is the expected outcome of that test, not a failure; at 88 C it is one
    return g_hot_stop && !(g_hot_rel > 0 && g_hottest < HOT_STOP_LIMIT_C) ? 1 : 0;
}

int mode_burst(int argc, char **argv) {
    int rc = 0;
    for (int i = 0; i < argc && !rc; i++) rc = run_arm(argv[i]);
    verdict("arms_run", g_not_run == 0, "%d of %d arms not started (hotspot at or above 58 C before the arm)", g_not_run,
            argc);
    return rc;
}
