// Lane 7 item 1: launch overhead under HIP, the effect of idleness, and the clock ramp seen from inside a kernel.
//
// Usage: sched_launch <calib|launch|idle|ramp> [options]
//   calib   one wave, 16 independent v_fma_f32 per block, U = 4 and 8 blocks per loop iteration: cycles per FMA must
//           be 1.00 (lane 1); REALTIME against host time must be 100 MHz within 0.5 %. --expect-fma X changes the
//           expected FMA cost (the script's deliberate break uses 2).
//   launch  wall time of launch + completion for an empty kernel and a one-wave kernel against grid and block size;
//           back to back without waiting, with hipDeviceSynchronize, with events; first launch against later ones.
//   idle    the same launch after 0, 1, 10, 100, 1000 ms of idle, with the clock the kernel sees.
//   ramp    after --idle-ms of idle, a kernel runs for --ms milliseconds and wave 0 records its cycle counter against
//           REALTIME every few microseconds; --card loads every SIMD (one 1024-thread group per WGP-pair, thermal
//           gate first), otherwise one wave. Output: clock per window and the time to reach 95 % of the final clock.
#include "sched_common.h"

using namespace sched;

namespace {

constexpr int CAL_ITERS = 256;

#define FMA1(i) "v_fma_f32 %" #i ", %16, %17, %" #i "\n"
#define FMA16 FMA1(0) FMA1(1) FMA1(2) FMA1(3) FMA1(4) FMA1(5) FMA1(6) FMA1(7) FMA1(8) FMA1(9) FMA1(10) FMA1(11) \
    FMA1(12) FMA1(13) FMA1(14) FMA1(15)
#define X16 "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), \
    "+v"(x[8]), "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])

__device__ __forceinline__ void fma_block(float (&x)[16], float a, float b) {
    asm volatile(FMA16 : X16 : "v"(a), "v"(b));
}

template <int U>
__global__ __launch_bounds__(32) void k_calib(const float * in, uint64_t * out) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0] + i;
    }
    const float a = in[1], b = in[2];
    for (int rep = 0; rep < 4; ++rep) {
        const uint64_t c0 = cyc();
#pragma unroll 1
        for (int it = 0; it < CAL_ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                fma_block(x, a, b);
            }
        }
        const uint64_t c1 = cyc();
        if (threadIdx.x == 0) {
            out[rep] = c1 - c0;
        }
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (s == 1234.5f) {
        out[8] = 1;
    }
}

// Spins `ticks` REALTIME ticks; records REALTIME and cycles at start and end.
__global__ void k_spin_rt(uint64_t ticks, uint64_t * out) {
    const uint64_t r0 = rt(), c0 = cyc();
    uint64_t r1;
    do {
        r1 = rt();
    } while (r1 - r0 < ticks);
    const uint64_t c1 = cyc();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = r0;
        out[1] = r1;
        out[2] = c1 - c0;
    }
}

__global__ void k_empty() {}
__global__ void k_empty_first() {}
// One wave of real work: 64 FMA blocks (1024 FMAs per lane) and one store; also the cycles and REALTIME it took.
__global__ __launch_bounds__(32) void k_one(const float * in, uint64_t * out) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0] + i;
    }
    const float a = in[1], b = in[2];
    const uint64_t r0 = rt(), c0 = cyc();
#pragma unroll 1
    for (int it = 0; it < 64; ++it) {
        fma_block(x, a, b);
    }
    const uint64_t c1 = cyc(), r1 = rt();
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = (uint64_t) (s == 1234.5f);
    }
}
__global__ __launch_bounds__(32) void k_one_first(const float * in, uint64_t * out) {
    if (threadIdx.x == 0) {
        out[3] = (uint64_t) in[0];
    }
}

// Ramp: every wave runs chunks of 8 FMA blocks (128 FMAs per lane, about 130 cycles alone on a SIMD);
// wave 0 of group 0 stores (cycles, REALTIME) after every `every` chunks.
__global__ __launch_bounds__(1024) void k_ramp(const float * in, uint64_t * trace, int nrec, int every,
                                               uint64_t stop_ticks) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0] + i;
    }
    const float a = in[1], b = in[2];
    const bool rec = blockIdx.x == 0 && threadIdx.x < 32;
    const uint64_t r0 = rt();
    int n = 0;
    uint64_t r = r0;
    while (r - r0 < stop_ticks) {
#pragma unroll 1
        for (int e = 0; e < every; ++e) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                fma_block(x, a, b);
            }
        }
        const uint64_t c = cyc();
        r = rt();
        if (rec && threadIdx.x == 0 && n < nrec) {
            trace[2 * n] = c;
            trace[2 * n + 1] = r;
        }
        ++n;
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (s == 1234.5f) {
        trace[0] = 0;
    }
    if (rec && threadIdx.x == 0) {
        trace[2 * nrec] = (uint64_t) n;
    }
}

using clk = std::chrono::steady_clock;
inline double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }

float * d_in;
uint64_t * d_out;

void sleep_ms(double ms) {
    if (ms > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds((long) (ms * 1000)));
    }
}

int mode_calib(int argc, char ** argv) {
    const double expect = std::atof(opt(argc, argv, "--expect-fma", "1.0"));
    std::vector<double> per;
    for (int l = 0; l < 5; ++l) {
        uint64_t a[4], b[4];
        hipLaunchKernelGGL(k_calib<4>, dim3(1), dim3(32), 0, 0, d_in, d_out);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(a, d_out, sizeof a, hipMemcpyDeviceToHost));
        hipLaunchKernelGGL(k_calib<8>, dim3(1), dim3(32), 0, 0, d_in, d_out);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(b, d_out, sizeof b, hipMemcpyDeviceToHost));
        for (int r = 1; r < 4; ++r) {
            per.push_back(((double) b[r] - (double) a[r]) / (4.0 * CAL_ITERS * 16));
        }
    }
    const double med = pct(per, 0.5);
    std::printf("RESULT calib fma_cycles %s\n", dist(per, "%.4f").c_str());
    char t[200];
    std::snprintf(t, sizeof t, "cycles per independent v_fma_f32 = %.4f (expected %.2f +- 0.02)", med, expect);
    const bool ok1 = std::fabs(med - expect) <= 0.02;
    verdict("sched_launch calib_fma", ok1 ? "PASS" : "FAIL", t);
    // REALTIME against host time: 3 spins of 0.2 s.
    std::vector<double> mhz, clkmhz;
    for (int i = 0; i < 3; ++i) {
        uint64_t h[3];
        const auto t0 = clk::now();
        hipLaunchKernelGGL(k_spin_rt, dim3(1), dim3(32), 0, 0, (uint64_t) 20000000, d_out);
        CHECK(hipDeviceSynchronize());
        const double wall_us = us_since(t0);
        CHECK(hipMemcpy(h, d_out, sizeof h, hipMemcpyDeviceToHost));
        // The wall time includes launch and completion (tens of microseconds): bounded error under 0.1 %.
        mhz.push_back((double) (h[1] - h[0]) / wall_us);
        clkmhz.push_back((double) h[2] / ((double) (h[1] - h[0]) / 100.0));
    }
    std::printf("RESULT calib realtime_MHz_vs_host %s\n", dist(mhz, "%.3f").c_str());
    std::printf("RESULT calib one_wave_spin_clock_MHz %s\n", dist(clkmhz, "%.0f").c_str());
    const double m = pct(mhz, 0.5);
    std::snprintf(t, sizeof t, "REALTIME %.3f MHz against host time (expected 100 within 0.5 %%)", m);
    const bool ok2 = std::fabs(m - 100.0) <= 0.5;
    verdict("sched_launch calib_realtime", ok2 ? "PASS" : "FAIL", t);
    return ok1 && ok2 ? 0 : 1;
}

struct LaunchStats {
    std::vector<double> wall, gpu;
};

// launch + hipDeviceSynchronize, wall time on the host and event time on the GPU, `n` samples after `warm`.
template <class F>
LaunchStats sample(F launch, int n, int warm, double idle_ms = 0) {
    LaunchStats s;
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    for (int i = 0; i < warm; ++i) {
        launch();
    }
    CHECK(hipDeviceSynchronize());
    for (int i = 0; i < n; ++i) {
        sleep_ms(idle_ms);
        const auto t0 = clk::now();
        CHECK(hipEventRecord(e0));
        launch();
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        s.wall.push_back(us_since(t0));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        s.gpu.push_back(ms * 1000.0);
    }
    CHECK(hipEventDestroy(e0));
    CHECK(hipEventDestroy(e1));
    return s;
}

int mode_launch() {
    bool ok = true;
    // First launch of a kernel never launched before, then the same kernel again.
    {
        auto t0 = clk::now();
        hipLaunchKernelGGL(k_empty_first, dim3(1), dim3(32), 0, 0);
        CHECK(hipDeviceSynchronize());
        const double first = us_since(t0);
        std::vector<double> later;
        for (int i = 0; i < 200; ++i) {
            t0 = clk::now();
            hipLaunchKernelGGL(k_empty_first, dim3(1), dim3(32), 0, 0);
            CHECK(hipDeviceSynchronize());
            later.push_back(us_since(t0));
        }
        t0 = clk::now();
        hipLaunchKernelGGL(k_one_first, dim3(1), dim3(32), 0, 0, d_in, d_out);
        CHECK(hipDeviceSynchronize());
        const double first2 = us_since(t0);
        std::printf("RESULT launch first_empty_us %.1f first_onewave_us %.1f later_empty_us %s\n", first, first2,
                    dist(later, "%.1f").c_str());
    }
    // Grid and block size, launch + hipDeviceSynchronize.
    std::printf("TABLE grid block wall_us(median p10 p90 max) gpu_event_us(median p10 p90)\n");
    for (int block : {32, 256, 1024}) {
        for (int grid : {1, 16, 256, 4096, 65536, 1 << 20}) {
            const LaunchStats s = sample([&] { hipLaunchKernelGGL(k_empty, dim3(grid), dim3(block), 0, 0); },
                                         grid >= 65536 ? 50 : 200, 5);
            std::printf("RESULT launch empty grid=%d block=%d wall_us %s | gpu_us %s\n", grid, block,
                        dist(s.wall, "%.1f").c_str(), dist(s.gpu, "%.1f").c_str());
        }
    }
    {
        const LaunchStats s = sample([&] { hipLaunchKernelGGL(k_one, dim3(1), dim3(32), 0, 0, d_in, d_out); }, 300, 5);
        uint64_t h[2];
        CHECK(hipMemcpy(h, d_out, sizeof h, hipMemcpyDeviceToHost));
        std::printf("RESULT launch onewave wall_us %s | gpu_us %s | kernel_body cycles=%llu realtime_us=%.2f\n",
                    dist(s.wall, "%.1f").c_str(), dist(s.gpu, "%.1f").c_str(), (unsigned long long) h[0],
                    h[1] / 100.0);
    }
    // Synchronize only (no events) and back to back without waiting.
    for (int which = 0; which < 2; ++which) {
        std::vector<double> sync_only, b2b;
        for (int rep = 0; rep < 7; ++rep) {
            for (int i = 0; i < 100; ++i) {
                const auto t0 = clk::now();
                if (which == 0) {
                    hipLaunchKernelGGL(k_empty, dim3(1), dim3(32), 0, 0);
                } else {
                    hipLaunchKernelGGL(k_one, dim3(1), dim3(32), 0, 0, d_in, d_out);
                }
                CHECK(hipDeviceSynchronize());
                sync_only.push_back(us_since(t0));
            }
            const int N = 2000;
            const auto t0 = clk::now();
            for (int i = 0; i < N; ++i) {
                if (which == 0) {
                    hipLaunchKernelGGL(k_empty, dim3(1), dim3(32), 0, 0);
                } else {
                    hipLaunchKernelGGL(k_one, dim3(1), dim3(32), 0, 0, d_in, d_out);
                }
            }
            const double enq = us_since(t0) / N;
            CHECK(hipDeviceSynchronize());
            b2b.push_back(us_since(t0) / N);
            if (rep == 0) {
                std::printf("RESULT launch %s enqueue_only_us_per_launch %.2f\n", which ? "onewave" : "empty", enq);
            }
        }
        std::printf("RESULT launch %s sync_each_us %s\n", which ? "onewave" : "empty", dist(sync_only, "%.1f").c_str());
        std::printf("RESULT launch %s back_to_back_us_per_launch %s (7 reps of 2000)\n", which ? "onewave" : "empty",
                    dist(b2b, "%.2f").c_str());
    }
    // Events between back-to-back kernels: GPU-side spacing without host waits.
    {
        const int N = 200;
        std::vector<hipEvent_t> ev(N + 1);
        for (auto & e : ev) {
            CHECK(hipEventCreate(&e));
        }
        CHECK(hipEventRecord(ev[0]));
        for (int i = 0; i < N; ++i) {
            hipLaunchKernelGGL(k_empty, dim3(1), dim3(32), 0, 0);
            CHECK(hipEventRecord(ev[i + 1]));
        }
        CHECK(hipDeviceSynchronize());
        std::vector<double> d;
        for (int i = 0; i < N; ++i) {
            float ms;
            CHECK(hipEventElapsedTime(&ms, ev[i], ev[i + 1]));
            d.push_back(ms * 1000);
        }
        std::printf("RESULT launch empty event_spacing_back_to_back_us %s\n", dist(d, "%.2f").c_str());
        for (auto & e : ev) {
            CHECK(hipEventDestroy(e));
        }
        ok = ok && pct(d, 0.5) > 0;
    }
    verdict("sched_launch launch", ok ? "PASS" : "FAIL", "all launches completed; event spacing positive");
    return ok ? 0 : 1;
}

int mode_idle() {
    std::printf("TABLE idle_ms kernel wall_us(median p10 p90 max) gpu_us body_clock_MHz(median min max)\n");
    for (double idle : {0.0, 1.0, 10.0, 100.0, 1000.0}) {
        const int n = idle >= 1000 ? 8 : idle >= 100 ? 30 : 100;
        std::vector<double> body;
        LaunchStats s;
        hipEvent_t e0, e1;
        CHECK(hipEventCreate(&e0));
        CHECK(hipEventCreate(&e1));
        for (int i = 0; i < 3; ++i) {
            hipLaunchKernelGGL(k_one, dim3(1), dim3(32), 0, 0, d_in, d_out);
        }
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < n; ++i) {
            sleep_ms(idle);
            const auto t0 = clk::now();
            CHECK(hipEventRecord(e0));
            hipLaunchKernelGGL(k_one, dim3(1), dim3(32), 0, 0, d_in, d_out);
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            s.wall.push_back(us_since(t0));
            float ms;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            s.gpu.push_back(ms * 1000);
            uint64_t h[2];
            CHECK(hipMemcpy(h, d_out, sizeof h, hipMemcpyDeviceToHost));
            if (h[1] > 0) {
                body.push_back((double) h[0] / ((double) h[1] / 100.0));
            }
        }
        CHECK(hipEventDestroy(e0));
        CHECK(hipEventDestroy(e1));
        std::printf("RESULT idle %.0f_ms onewave wall_us %s | gpu_us %s | body_clock_MHz %s\n", idle,
                    dist(s.wall, "%.1f").c_str(), dist(s.gpu, "%.1f").c_str(), dist(body, "%.0f").c_str());
    }
    verdict("sched_launch idle", "PASS", "all launches completed");
    return 0;
}

int mode_ramp(int argc, char ** argv) {
    const double idle = std::atof(opt(argc, argv, "--idle-ms", "1000"));
    const double run_ms = std::atof(opt(argc, argv, "--ms", "150"));
    const bool card = has_flag(argc, argv, "--card");
    const int every = 8;  // about 8 x 130 cycles = 0.33 us per record alone on a SIMD
    const int nrec = 1 << 20;
    uint64_t * d_tr;
    CHECK(hipMalloc(&d_tr, (2 * (size_t) nrec + 1) * 8));
    // Warm the code path, then idle.
    hipLaunchKernelGGL(k_ramp, dim3(1), dim3(32), 0, 0, d_in, d_tr, nrec, every, (uint64_t) 1000);
    CHECK(hipDeviceSynchronize());
    if (card && !gate(60.0)) {
        verdict("sched_launch ramp", "FAIL", "hotspot did not fall below 60 C");
        return 1;
    }
    Sampler smp(5);
    smp.start();
    sleep_ms(idle);
    const double t_launch = smp.now();
    // --card: 64 groups of 1024 threads (2048 waves, up to 16 per SIMD): every SIMD busy; wave 0 of group 0 records.
    const dim3 g(card ? 64 : 1), b(card ? 1024 : 32);
    hipLaunchKernelGGL(k_ramp, g, b, 0, 0, d_in, d_tr, nrec, every, (uint64_t) (run_ms * 1e5));
    CHECK(hipDeviceSynchronize());
    const double t_done = smp.now();
    sleep_ms(30);
    smp.stop();
    std::vector<uint64_t> h(2 * (size_t) nrec + 1);
    CHECK(hipMemcpy(h.data(), d_tr, h.size() * 8, hipMemcpyDeviceToHost));
    CHECK(hipFree(d_tr));
    const int n = (int) std::min<uint64_t>(h[2 * nrec], nrec);
    // Clock per 1 ms window from (cycles, REALTIME) pairs; the first record is t = 0.
    const uint64_t c0 = h[0], r0 = h[1];
    std::printf("RAMP %s idle_ms=%.0f records=%d kernel_wall_ms=%.1f\n", card ? "card" : "onewave", idle, n,
                (t_done - t_launch) * 1e3);
    std::vector<std::pair<double, double>> win;  // (ms, MHz)
    int j = 0;
    for (double ms = 0; ms < run_ms; ms += (ms < 10 ? 0.25 : 1.0)) {
        const double step = ms < 10 ? 0.25 : 1.0;
        const uint64_t ra = r0 + (uint64_t) (ms * 1e5), rb = r0 + (uint64_t) ((ms + step) * 1e5);
        while (j < n && h[2 * j + 1] < ra) {
            ++j;
        }
        int k = j;
        while (k < n && h[2 * k + 1] < rb) {
            ++k;
        }
        if (k - 1 > j) {
            const double dc = (double) (h[2 * (k - 1)] - h[2 * j]);
            const double dr = (double) (h[2 * (k - 1) + 1] - h[2 * j + 1]);
            win.push_back({ms, dc / (dr / 100.0)});
        }
    }
    double fin = 0;
    int cnt = 0;
    for (auto & w : win) {
        if (w.first >= run_ms * 0.8) {
            fin += w.second;
            ++cnt;
        }
    }
    fin = cnt ? fin / cnt : 0;
    double t95 = -1;
    for (size_t i = 0; i < win.size(); ++i) {
        bool all = true;
        for (size_t k = i; k < win.size(); ++k) {
            all = all && win[k].second >= 0.95 * fin;
        }
        if (all) {
            t95 = win[i].first;
            break;
        }
    }
    for (auto & w : win) {
        if (w.first < 30 || std::fmod(w.first, 5.0) < 0.5) {
            std::printf("RAMPWIN %s idle_ms=%.0f t_ms=%.2f clock_MHz=%.0f\n", card ? "card" : "onewave", idle, w.first,
                        w.second);
        }
    }
    std::printf("RESULT ramp %s idle_ms=%.0f first_window_MHz=%.0f final_MHz=%.0f time_to_95pct_ms=%.2f "
                "total_cycles=%.0f\n",
                card ? "card" : "onewave", idle, win.empty() ? 0 : win[0].second, fin, t95,
                (double) (h[2 * (n - 1)] - c0));
    for (const Snap & s : smp.copy()) {
        if (s.t >= t_launch - 0.02 && s.t <= t_done + 0.03) {
            std::printf("SYSFS t_ms_from_launch=%.0f sclk=%d power_W_reported=%.0f hotspot=%.0f busy=%d\n",
                        (s.t - t_launch) * 1e3, s.sclk, s.power, s.hot, s.busy);
        }
    }
    const bool ok = n > 100 && fin > 1000;
    verdict("sched_launch ramp", ok ? "PASS" : "FAIL", ok ? "trace recorded" : "trace too short or clock implausible");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: sched_launch <calib|launch|idle|ramp> [options]\n");
        return 2;
    }
    const float hin[4] = {0.5f, 0.9999f, 1e-4f, 0};
    CHECK(hipMalloc(&d_in, sizeof hin));
    CHECK(hipMemcpy(d_in, hin, sizeof hin, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_out, 64 * 8));
    const std::string m = argv[1];
    int rc = 2;
    if (m == "calib") {
        rc = mode_calib(argc, argv);
    } else if (m == "launch") {
        rc = mode_launch();
    } else if (m == "idle") {
        rc = mode_idle();
    } else if (m == "ramp") {
        rc = mode_ramp(argc, argv);
    }
    std::printf("HOTTEST %.0f C\n", hottest().load());
    return rc;
}
