// CPU-only self-test for src/round13/vk_async_ref.h (no Vulkan, no GPU).
// Usage: vk_async_selftest              all checks must pass (exit 0)
//        vk_async_selftest --break N    deliberate break N (1..12) must exit non-zero
#define _GNU_SOURCE
#include "vk_async_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_bad = 0;

#define CHK(cond, msg)                                           \
    do {                                                         \
        if (!(cond)) {                                           \
            printf("vk_async_selftest FAIL: %s\n", (msg));       \
            ++g_bad;                                             \
        }                                                        \
    } while (0)

static void write_text(const char * path, const char * txt) {
    FILE * f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(2);
    }
    fputs(txt, f);
    fclose(f);
}

typedef struct {
    const double * seq;
    int len;
    int idx;
    int slept_ms;
} MockHotCtx;

static int mock_hot_read(void * ctx, double * out_hot_c) {
    MockHotCtx * m = (MockHotCtx *)ctx;
    int i = (m->idx < m->len) ? m->idx : (m->len - 1);
    ++m->idx;
    const double v = m->seq[i];
    if (v <= 0.0 || v >= 200.0) {
        if (out_hot_c) *out_hot_c = -1.0;
        return VKA_HOT_SENSOR_ERR;
    }
    if (out_hot_c) *out_hot_c = v;
    return v < VKA_THERMAL_STOP_C ? VKA_HOT_OK : VKA_HOT_OVER_LIMIT;
}

static void mock_sleep_ms(void * ctx, int ms) {
    MockHotCtx * m = (MockHotCtx *)ctx;
    m->slept_ms += ms;
}

int main(int argc, char ** argv) {
    int brk = 0;
    if (argc == 3 && strcmp(argv[1], "--break") == 0) {
        brk = atoi(argv[2]);
        if (brk < 1 || brk > 12) {
            fprintf(stderr, "break index must be 1..12\n");
            return 2;
        }
    } else if (argc != 1) {
        fprintf(stderr, "usage: vk_async_selftest [--break 1..12]\n");
        return 2;
    }

    // 1. Hand-computed 4-byte integer SAD exactness + chained N=80 accumulation & break sensitivity.
    {
        // Bytes (0x00, 0x10, 0x80, 0xff) vs (0xff, 0x05, 0x90, 0x00) -> 255 + 11 + 16 + 255 = 537
        const uint32_t d = vka_sad4_u32(0xff801000u, 0x009005ffu);
        const uint32_t want = (brk == 1) ? 536u : 537u;
        CHK(d == want, "vka_sad4_u32 hand-computed 4-byte absolute difference");
        CHK(vka_sad4_u32(0xffffffffu, 0x00000000u) == 1020u, "vka_sad4_u32 max 4x255=1020");
        CHK(vka_sad4_u32(0x12345678u, 0x12345678u) == 0u, "vka_sad4_u32 identical words = 0");

        const uint32_t c_ok = vka_expected_sad_chain(2u, 17u, 14u, 80u, 0u);
        const uint32_t c_brk = vka_expected_sad_chain(2u, 17u, 14u, 80u, 1u);
        uint32_t manual = 0u;
        for (uint32_t p = 0u; p < 80u; ++p) {
            manual += vka_expected_sad_pass(2u, 17u, 14u, p, 0u);
        }
        CHK(c_ok == manual && c_ok != 0u, "vka_expected_sad_chain matches 80-pass sum");
        CHK(c_ok != c_brk, "SAD break_mask=1 changes 80-pass expected output");
    }

    // 2. 128 MiB streaming copy+XOR reference: N=1 (20 sweeps) and N=80 (quarter cycling) + break check.
    {
        uint32_t a1[4], b1[4];
        vka_stream_expected_vec4(12345u, 1u, VKA_STREAM_N1_ITERS, VKA_STREAM_BASE_SALT, 0u, a1);
        vka_stream_init_vec4(12345u, b1);
        const uint32_t m1 = VKA_STREAM_BASE_SALT ^ (VKA_STREAM_N1_ITERS - 1u);
        for (int c = 0; c < 4; ++c) {
            const uint32_t want = b1[c] ^ m1 ^ ((brk == 2 && c == 0) ? 1u : 0u);
            CHK(a1[c] == want, "stream N=1 final XOR mask matches last sweep");
        }
        // For SAT and PACE (both N=80): quarters 0..3 are last written by passes (N-4)..(N-1).
        const uint32_t cns[2] = {VKA_CHAIN_SAT, VKA_CHAIN_PACE};
        for (int ci = 0; ci < 2; ++ci) {
            const uint32_t cn = cns[ci];
            for (uint32_t q = 0u; q < 4u; ++q) {
                const uint32_t idx = q * VKA_STREAM_QUARTER_VEC4 + 77u;
                const uint32_t got_m = vka_stream_final_mask(idx, cn, 1u, VKA_STREAM_BASE_SALT, 0u);
                const uint32_t want_m = VKA_STREAM_BASE_SALT ^ ((cn - 4u) + q);
                CHK(got_m == want_m, "stream chained quarter last-writer pass index exact");
                uint32_t ok_v[4], brk_v[4];
                vka_stream_expected_vec4(idx, cn, 1u, VKA_STREAM_BASE_SALT, 0u, ok_v);
                vka_stream_expected_vec4(idx, cn, 1u, VKA_STREAM_BASE_SALT, 1u, brk_v);
                CHK(ok_v[0] == (brk_v[0] ^ 1u), "stream break_mask=1 flips output bit 0");
            }
        }
    }

    // 3. Fail-closed 95 C hotspot guard: valid <95 passes, >=95 stops, unreadable/corrupt stops.
    {
        const char * base = getenv("TMPDIR");
        if (!base || !base[0]) base = "build/round13";
        char dir[512];
        snprintf(dir, sizeof dir, "%s/vka-hot-test-%ld", base, (long)getpid());
        mkdir(dir, 0755);

        char p_ok[600], p_edge[600], p_90[600], p_hot[600], p_empty[600], p_bad[600], p_neg[600], p_miss[600];
        snprintf(p_ok, sizeof p_ok, "%s/ok", dir);
        snprintf(p_edge, sizeof p_edge, "%s/edge", dir);
        snprintf(p_90, sizeof p_90, "%s/at90", dir);
        snprintf(p_hot, sizeof p_hot, "%s/hot", dir);
        snprintf(p_empty, sizeof p_empty, "%s/empty", dir);
        snprintf(p_bad, sizeof p_bad, "%s/bad", dir);
        snprintf(p_neg, sizeof p_neg, "%s/neg", dir);
        snprintf(p_miss, sizeof p_miss, "%s/does_not_exist", dir);

        write_text(p_ok, "85000\n");
        write_text(p_edge, "94999\n");
        write_text(p_90, "95000\n");
        write_text(p_hot, "99000\n");
        write_text(p_empty, "");
        write_text(p_bad, "not_a_temp\n");
        write_text(p_neg, "-5000\n");

        double hc = 0.0;
        CHK(vka_check_hotspot_file(p_ok, VKA_THERMAL_STOP_C, &hc) == 1 && fabs(hc - 85.0) < 1e-6,
            "hotspot 85 C passes");
        CHK(vka_check_hotspot_file(p_edge, VKA_THERMAL_STOP_C, &hc) == 1 && fabs(hc - 94.999) < 1e-6,
            "hotspot 94.999 C passes");
        CHK(vka_check_hotspot_file(p_90, VKA_THERMAL_STOP_C, &hc) == (brk == 3 ? 1 : 0),
            "hotspot 95.0 C stops");
        CHK(vka_check_hotspot_file(p_hot, VKA_THERMAL_STOP_C, &hc) == 0,
            "hotspot 99.0 C stops");
        CHK(vka_check_hotspot_file(p_miss, VKA_THERMAL_STOP_C, &hc) == (brk == 4 ? 1 : 0),
            "missing/unreadable hotspot sensor fails closed (stops)");
        CHK(vka_check_hotspot_file(p_empty, VKA_THERMAL_STOP_C, &hc) == 0,
            "empty hotspot file fails closed (stops)");
        CHK(vka_check_hotspot_file(p_bad, VKA_THERMAL_STOP_C, &hc) == 0,
            "non-numeric hotspot file fails closed (stops)");
        CHK(vka_check_hotspot_file(p_neg, VKA_THERMAL_STOP_C, &hc) == 0,
            "negative hotspot file fails closed (stops)");
        CHK(vka_check_hotspot_file(NULL, VKA_THERMAL_STOP_C, &hc) == 0,
            "NULL hotspot path fails closed (stops)");
        CHK(vka_read_hotspot_status(p_90, VKA_THERMAL_STOP_C, &hc) == VKA_HOT_OVER_LIMIT,
            "hotspot 95.0 C status is OVER_LIMIT");
        CHK(vka_read_hotspot_status(p_miss, VKA_THERMAL_STOP_C, &hc) == VKA_HOT_SENSOR_ERR,
            "missing hotspot status is SENSOR_ERR");

        // 3b. Real-file bounded cooldown wait (< 60 C): cool passes, warm/hot times out cleanly, missing fails closed.
        char p_cool[600], p_warm[600];
        snprintf(p_cool, sizeof p_cool, "%s/cool", dir);
        snprintf(p_warm, sizeof p_warm, "%s/warm", dir);
        write_text(p_cool, "45000\n");
        write_text(p_warm, "75000\n");

        VkaCooldownResult cr;
        int cs = vka_wait_cooldown_file(p_cool, VKA_THERMAL_STOP_C, VKA_COOLDOWN_TARGET_C, 0.0, 60, 0, 20, &cr);
        CHK(cs == VKA_COOL_OK && cr.waited_ms == 0 && cr.polls == 1 && fabs(cr.end_hot_c - 45.0) < 1e-6,
            "cooldown at 45 C returns OK immediately");

        cs = vka_wait_cooldown_file(p_warm, VKA_THERMAL_STOP_C, VKA_COOLDOWN_TARGET_C, 0.0, 60, 0, 20, &cr);
        CHK(cs == (brk == 7 ? VKA_COOL_OK : VKA_COOL_TIMEOUT) && cr.waited_ms == 60 && fabs(cr.end_hot_c - 75.0) < 1e-6,
            "cooldown stuck at 75 C times out cleanly at max_wait_ms");

        cs = vka_wait_cooldown_file(p_hot, VKA_THERMAL_STOP_C, VKA_COOLDOWN_TARGET_C, 0.0, 40, 0, 20, &cr);
        CHK(cs == VKA_COOL_TIMEOUT && cr.waited_ms == 40 && fabs(cr.end_hot_c - 99.0) < 1e-6,
            "cooldown stuck at 99 C times out cleanly without false OK");

        cs = vka_wait_cooldown_file(p_miss, VKA_THERMAL_STOP_C, VKA_COOLDOWN_TARGET_C, 0.0, 60, 0, 20, &cr);
        CHK(cs == VKA_COOL_SENSOR_ERR && cr.waited_ms == 0,
            "cooldown on missing sensor fails closed immediately");

        unlink(p_cool); unlink(p_warm);
        unlink(p_ok); unlink(p_edge); unlink(p_90); unlink(p_hot);
        unlink(p_empty); unlink(p_bad); unlink(p_neg);
        rmdir(dir);
    }

    // 3c. Synthetic cooling sequences (cooling past 60 C + soft settle to 50 C vs warm ambient 57 C).
    {
        const double seq_cool[6] = {78.0, 68.0, 61.0, 58.0, 52.0, 48.0};
        MockHotCtx m1 = {seq_cool, 6, 0, 0};
        VkaCooldownResult cr;
        int cs = vka_wait_cooldown_fn(mock_hot_read, mock_sleep_ms, &m1,
                                      VKA_COOLDOWN_TARGET_C, VKA_COOLDOWN_SETTLE_C, 500, 200, 50, &cr);
        CHK(cs == VKA_COOL_OK && fabs(cr.start_hot_c - 78.0) < 1e-6 && fabs(cr.end_hot_c - 48.0) < 1e-6 && cr.waited_ms == 250,
            "cooldown drops below 60 C and soft-settles to 48 C");

        const double seq_amb[8] = {72.0, 58.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0};
        MockHotCtx m2 = {seq_amb, 8, 0, 0};
        cs = vka_wait_cooldown_fn(mock_hot_read, mock_sleep_ms, &m2,
                                  VKA_COOLDOWN_TARGET_C, VKA_COOLDOWN_SETTLE_C, 500, 500, 50, &cr);
        CHK(cs == VKA_COOL_OK && fabs(cr.end_hot_c - 57.0) < 1e-6 && cr.waited_ms == 200,
            "warm ambient 57 C exits soft settle early on flat plateau without false timeout");

        const double seq_err[3] = {72.0, -1.0, 50.0};
        MockHotCtx m3 = {seq_err, 3, 0, 0};
        cs = vka_wait_cooldown_fn(mock_hot_read, mock_sleep_ms, &m3,
                                  VKA_COOLDOWN_TARGET_C, VKA_COOLDOWN_SETTLE_C, 500, 100, 50, &cr);
        CHK(cs == VKA_COOL_SENSOR_ERR, "sensor drop mid-cooldown fails closed");
    }

    // 3d. Double-stop thermal policy: 1st >= 95 C stops arm & continues (non-zero exit), 2nd aborts run (exit 4).
    {
        VkaThermalState ts;
        vka_thermal_init(&ts, VKA_THERMAL_MAX_STOPS);
        CHK(vka_thermal_on_event(&ts, VKA_HOT_OK) == VKA_THERM_NONE, "OK event produces no stop");
        CHK(vka_thermal_exit_code(&ts, 0) == 0, "clean thermal state exits 0");

        const int a1 = vka_thermal_on_event(&ts, VKA_HOT_OVER_LIMIT);
        CHK(a1 == VKA_THERM_CONTINUE_NEXT_ARM && ts.stop_count == 1,
            "1st >= 95 C stop records arm stop and continues to next arm");
        CHK(vka_thermal_exit_code(&ts, 0) == 1, "single thermal stop still exits non-zero (1)");

        const int a2 = vka_thermal_on_event(&ts, VKA_HOT_OVER_LIMIT);
        const int want_a2 = (brk == 8) ? VKA_THERM_CONTINUE_NEXT_ARM : VKA_THERM_ABORT_RUN;
        CHK(a2 == want_a2 && ts.stop_count == 2, "2nd >= 95 C stop aborts whole run");
        CHK(vka_thermal_exit_code(&ts, 0) == 4, "double thermal stop exits 4");

        VkaThermalState ts_err;
        vka_thermal_init(&ts_err, VKA_THERMAL_MAX_STOPS);
        CHK(vka_thermal_on_event(&ts_err, VKA_HOT_SENSOR_ERR) == VKA_THERM_ABORT_RUN,
            "unreadable sensor aborts whole run immediately on 1st occurrence");
        CHK(vka_thermal_exit_code(&ts_err, 0) == 4, "sensor error exits 4");
    }

    // 4. ABBA schedule symmetry for k = 1, 2, 3, 4 allowed priorities.
    {
        for (int k = 1; k <= 4; ++k) {
            int allowed[4] = {128, 256, 512, 1024};
            int seq[8] = {0};
            const int n = vka_build_abba_schedule(allowed, k, seq);
            if (brk == 5 && k == 2) seq[3] = 999;
            CHK(n == 2 * k, "ABBA schedule length is 2*k");
            int sym = 1;
            for (int i = 0; i < n; ++i) {
                if (seq[i] != seq[n - 1 - i]) sym = 0;
            }
            CHK(sym, "ABBA schedule is exact palindrome");
        }
    }

    // 5. Synthetic graphics frame completion interval overlap computation.
    {
        // Frames complete at t = 10, 14, 18, 24, 30, 34 ms -> dt = [4, 4, 6, 6, 4]
        // Compute trial spans [12, 29] -> overlaps frames 1 (10..14), 2 (14..18), 3 (18..24), 4 (24..30)
        // Overlapping frame dts: i=1 (4), i=2 (4), i=3 (6), i=4 (6) -> avg = 5.0, p50 = 6.0 (or 4.0 nearest), p95 = 6.0.
        VkaGfxFrame frames[6] = {
            { 6.0, 10.0, 3.8, 4.0, 60.0},
            {10.0, 14.0, 3.9, 4.0, 61.0},
            {14.0, 18.0, 4.0, 4.0, 61.0},
            {18.0, 24.0, 5.8, 6.0, 62.0},
            {24.0, 30.0, 5.9, 6.0, 62.0},
            {30.0, 34.0, 3.8, 4.0, 61.0},
        };
        VkaCompTrial trials[1] = {
            {12.0, 29.0, 16.5, 17.0},
        };
        VkaGfxOverlapStats ov = vka_compute_gfx_overlap(frames, 6, trials, 1);
        const double want_avg = (brk == 6) ? 4.0 : 5.0;
        CHK(ov.n_strict_ov == 1, "strict overlap count == 1");
        CHK(ov.n_ov_frames == 4, "overlapping graphics frame count == 4");
        CHK(ov.n_dt == 4, "overlapping dt interval count == 4");
        CHK(fabs(ov.dt_avg_ms - want_avg) < 1e-6, "overlapping dt_avg_ms == 5.0");
        CHK(fabs(ov.dt_p95_ms - 6.0) < 1e-6, "overlapping dt_p95_ms == 6.0");
        CHK(fabs(ov.dt_max_ms - 6.0) < 1e-6, "overlapping dt_max_ms == 6.0");

        // 5b. Multi-repeat overlap accumulator: two bursts separated by a 2000 ms cooldown gap.
        VkaGfxFrame frames_b2[6];
        for (int i = 0; i < 6; ++i) {
            frames_b2[i] = frames[i];
            frames_b2[i].t_sub_ms += 2000.0;
            frames_b2[i].t_done_ms += 2000.0;
        }
        VkaCompTrial trials_b2[1] = {{2012.0, 2029.0, 16.5, 17.0}};
        VkaOverlapAccum acc;
        vka_overlap_accum_init(&acc);
        vka_overlap_accum_add_batch(&acc, frames, 6, trials, 1);
        vka_overlap_accum_add_batch(&acc, frames_b2, 6, trials_b2, 1);
        VkaGfxOverlapStats ov2 = vka_overlap_accum_finish(&acc);
        vka_overlap_accum_free(&acc);
        const double want_max2 = (brk == 9) ? 2000.0 : 6.0;
        CHK(ov2.n_strict_ov == 2, "multi-repeat strict overlap count == 2");
        CHK(ov2.n_ov_frames == 8 && ov2.n_dt == 8, "multi-repeat overlapping frame count == 8");
        CHK(fabs(ov2.dt_avg_ms - 5.0) < 1e-6, "multi-repeat dt_avg_ms == 5.0 (no cooldown gap leakage)");
        CHK(fabs(ov2.dt_max_ms - want_max2) < 1e-6, "multi-repeat dt_max_ms == 6.0 (no 2000 ms gap)");
    }

    // 6. Preregistered SAT-100 & PACE-60 chain lengths (both N=80) and equal total work per job
    //    between N=1 and N=80 (stream = 5120 MiB = 5.0 GiB; fp16 N=1 even iters stays non-trivial).
    {
        const uint32_t sat_n = (brk == 10) ? 40u : VKA_CHAIN_SAT;
        CHK(sat_n == 80u && VKA_CHAIN_PACE == 80u, "SAT and PACE both use preregistered N=80 chains");
        for (int j = 0; j < VKA_JOB_COUNT; ++j) {
            const uint64_t w1 = vka_job_total_work(j, 1u);
            const uint64_t w_sat = vka_job_total_work(j, sat_n);
            const uint64_t w_pace = vka_job_total_work(j, VKA_CHAIN_PACE);
            // fp16 differs only by the 5-FMA parity step on the even N=1 chain (< 0.01 %).
            const uint64_t tol = (j == VKA_JOB_FP16) ? w1 / 10000ull : 0ull;
            const uint64_t d_sat = (w1 > w_sat) ? w1 - w_sat : w_sat - w1;
            const uint64_t d_pace = (w1 > w_pace) ? w1 - w_pace : w_pace - w1;
            CHK(w1 > 0ull && d_sat <= tol && d_pace <= tol, "N=1 total work equals N=80 total work for every job");
        }
        {
            const uint32_t reg[3] = { VKA_FP16_N1_REGISTERED, VKA_SAD_N1_REGISTERED, VKA_TEX_N1_REGISTERED };
            const uint32_t n1[3] = { VKA_FP16_N1_ITERS, VKA_SAD_N1_ITERS, (brk == 12) ? 160u : VKA_TEX_N1_ITERS };
            for (int k = 0; k < 3; ++k)
                CHK(n1[k] >= reg[k] && n1[k] * 4u <= reg[k] * 5u, "N=1 job sizes stay within +25 % of the registered monolithic sizes");
        }
        CHK(vka_job_total_work(VKA_JOB_STREAM, 1u) == VKA_STREAM_TOTAL_RW_BYTES,
            "stream N=1 traffic is exactly 5120 MiB (5.0 GiB)");
        CHK(vka_job_total_work(VKA_JOB_STREAM, VKA_CHAIN_SAT) == VKA_STREAM_TOTAL_RW_BYTES,
            "stream N=80 traffic is exactly 5120 MiB (5.0 GiB)");
        // Verify fp16 N=1 (even iters=880) has non-trivial parity and rejects dropped-FMA init sum.
        float wlo = 0.0f, whi = 0.0f;
        vkt_expected_mix_alu(7u, 64u, VKA_FP16_N1_ITERS, &wlo, &whi);
        CHK(vkt_verify_mix_alu_lane(7u, 64u, VKA_FP16_N1_ITERS, 1, wlo, whi, 0, NULL) == 1,
            "fp16 N=1 even-iters expected output passes non-trivial parity check");
        CHK(vkt_verify_mix_alu_lane(7u, 64u, VKA_FP16_N1_ITERS, 1, VKT_OSC5_INIT_SUM_LO, VKT_OSC5_INIT_SUM_HI, 0, NULL) == 0,
            "fp16 N=1 rejects dropped-FMA initial accumulator sum");
    }

    // 7. Hard pre-cell < 60 C cooldown policy: missed cooldown never runs the cell;
    //    exit 4 if wait ended >= 95 C (or sensor err), exit 1 otherwise.
    {
        CHK(vka_precell_cooldown_action(VKA_COOL_OK, 55.0, VKA_THERMAL_STOP_C) == VKA_PRECELL_RUN,
            "pre-cell cooldown OK at 55 C allows running cell");
        const int act_warm = vka_precell_cooldown_action(VKA_COOL_TIMEOUT, 75.0, VKA_THERMAL_STOP_C);
        CHK(act_warm == (brk == 11 ? VKA_PRECELL_RUN : VKA_PRECELL_FAIL_CELL),
            "missed <60 C cooldown ending at 75 C skips cell and fails verdict (exit 1)");
        CHK(vka_precell_cooldown_action(VKA_COOL_TIMEOUT, 94.9, VKA_THERMAL_STOP_C) == VKA_PRECELL_FAIL_CELL,
            "missed <60 C cooldown ending just below the 95 C stop fails the cell (exit 1), not abort");
        CHK(vka_precell_cooldown_action(VKA_COOL_TIMEOUT, 95.0, VKA_THERMAL_STOP_C) == VKA_PRECELL_ABORT_HOT,
            "missed <60 C cooldown ending at 95 C aborts with exit 4");
        CHK(vka_precell_cooldown_action(VKA_COOL_SENSOR_ERR, -1.0, VKA_THERMAL_STOP_C) == VKA_PRECELL_ABORT_HOT,
            "pre-cell cooldown sensor error aborts with exit 4");

        VkaThermalState t_fail;
        vka_thermal_init(&t_fail, VKA_THERMAL_MAX_STOPS);
        ++t_fail.cooldown_timeouts;
        CHK(vka_thermal_exit_code(&t_fail, 1) == 1, "cooldown timeout <95 C exits 1");

        VkaThermalState t_hot;
        vka_thermal_init(&t_hot, VKA_THERMAL_MAX_STOPS);
        ++t_hot.cooldown_timeouts;
        ++t_hot.cooldown_hot_aborts;
        CHK(vka_thermal_exit_code(&t_hot, 1) == 4, "cooldown timeout >=95 C exits 4 immediately (even on 1st stop)");
    }

    printf("VERDICT vk_async_selftest: %s (bad=%d%s)\n",
           g_bad == 0 ? "PASS" : "FAIL", g_bad, brk ? ", deliberate break" : "");
    return g_bad == 0 ? 0 : 1;
}
