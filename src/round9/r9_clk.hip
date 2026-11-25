// Round 9, item D through HIP: one dispatch of a fixed length, called and waited on at a fixed cadence.
//
// Usage: r9_clk <out.csv> [ms=1.0] [cad=16.7] [dur=20] [blocks=256] [idle=1] [calib=1]
// Every block works until the GPU wall clock (100 MHz, wall_clock64) has passed its own start + ms, in chunks of a
// dependent FMA chain; thread 0 of block 0 records (wall clock, clock64) at every chunk. The in-kernel clock is cycles
// over wall-clock ticks x 100 MHz, as round 5's spin.hip and round 7. Block size 64 (two wave32 waves). Each block's
// thread 0 also records its own start and end pair, so the blocks' clocks can be compared. After the cadence run,
// calib adds three back-to-back 30 ms dispatches whose clock is also taken from HIP events (a second timer).
// Planned busy share must stay at or below 0.25; the cell stops if the hotspot (HWMON temp2_input) reaches 88 C.
// Output CSV: same columns as vkr9 clk (ts_top / ts_bot hold the HIP event times in microseconds x 1000, i.e. ns).
#include <hip/hip_runtime.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr unsigned MAXREC = 1024;

struct Rec {
    unsigned long long w, c;
};

__global__ __launch_bounds__(64) void spin(unsigned long long ticks, Rec * rec, Rec * grp, float * sink, int chunk) {
    float a = 1.0f + threadIdx.x, b = 0.999f, c = 0.001f;
    const unsigned long long w0 = wall_clock64();
    const unsigned long long c0 = clock64();
    unsigned long long w1 = w0;
    const bool first = blockIdx.x == 0 && threadIdx.x == 0;
    unsigned k = 0;
    if (first) rec[1] = {w0, c0};
    while (w1 - w0 < ticks) {
        for (int i = 0; i < chunk; i++) a = fmaf(a, b, c);
        w1 = wall_clock64();
        ++k;
        if (first && k < MAXREC) rec[1 + k] = {w1, (unsigned long long) clock64()};
    }
    const unsigned long long c1 = clock64();
    if (first) rec[0] = {k + 1 < MAXREC ? k + 1 : MAXREC, 0};
    if (threadIdx.x == 0) {
        grp[2 * blockIdx.x] = {w0, c0};
        grp[2 * blockIdx.x + 1] = {w1, c1};
    }
    if (a == 12345.0f) sink[threadIdx.x] = a;
}

static double mono_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static void until_ns(double t) {
    for (;;) {
        const double d = t - mono_ns();
        if (d <= 0) return;
        if (d > 300e3) {
            const double s = d - 300e3;
            timespec ts = {(time_t) (s / 1e9), (long) std::fmod(s, 1e9)};
            nanosleep(&ts, nullptr);
        }
    }
}

static double hotspot_c() {
    const char * hw = std::getenv("HWMON");
    if (!hw) return -1;
    FILE * f = std::fopen((std::string(hw) + "/temp2_input").c_str(), "r");
    if (!f) return -1;
    double v = -1;
    if (std::fscanf(f, "%lf", &v) != 1) v = -1;
    std::fclose(f);
    return v / 1000.0;
}

static int uclk_now() {
    const char * d = std::getenv("DRM_DEVICE");
    if (!d) return -1;
    FILE * f = std::fopen((std::string(d) + "/gpu_metrics").c_str(), "rb");
    if (!f) return -1;
    unsigned char b[64];
    const size_t n = std::fread(b, 1, sizeof b, f);
    std::fclose(f);
    return n >= 60 ? (int) (b[58] | (b[59] << 8)) : -1;
}

static void row(FILE * csv, const char * kind, int idx, double tsub, double tdone, double ev_ns, const Rec * rec, const Rec * grp,
                int blocks, unsigned long long lim) {
    const unsigned count = (unsigned) rec[0].w;
    const unsigned long long rt0 = rec[1].w, cy0 = rec[1].c;
    unsigned long long rt1 = rt0, cy1 = cy0, rtq = rt0, cyq = cy0, rt3 = rt0, cy3 = cy0;
    bool gq = false, g3 = false;
    for (unsigned k = 1; k < count; ++k) {
        const Rec & r = rec[1 + k];
        if (!gq && r.w - rt0 >= lim / 4) rtq = r.w, cyq = r.c, gq = true;
        if (!g3 && r.w - rt0 >= 3 * lim / 4) rt3 = r.w, cy3 = r.c, g3 = true;
        rt1 = r.w, cy1 = r.c;
    }
    double wmin = 1e30, wmax = 0;
    for (int b = 0; b < blocks; ++b) {
        const Rec &a = grp[2 * b], &e = grp[2 * b + 1];
        if (e.w > a.w) {
            const double mhz = (double) (e.c - a.c) / (double) (e.w - a.w) * 100.0;
            wmin = std::min(wmin, mhz);
            wmax = std::max(wmax, mhz);
        }
    }
    std::fprintf(csv, "%s,%d,%.0f,%.0f,0,%.0f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%.1f,%.1f\n", kind, idx, tsub, tdone,
                 ev_ns, rt0, cy0, rt1, cy1, rtq, cyq, rt3, cy3, count, wmin, wmax);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: r9_clk <out.csv> [ms] [cad] [dur] [blocks] [idle] [calib]\n");
        return 2;
    }
    const double ms = argc > 2 ? std::atof(argv[2]) : 1.0, cad = argc > 3 ? std::atof(argv[3]) : 16.7,
                 dur = argc > 4 ? std::atof(argv[4]) : 20, idle = argc > 6 ? std::atof(argv[6]) : 1;
    const int blocks = argc > 5 ? std::atoi(argv[5]) : 256, calib = argc > 7 ? std::atoi(argv[7]) : 1, chunk = 4096;
    if (ms < 0.05 || ms > 5.0 || cad < 4 * ms || cad < 5 || dur > 30 || blocks < 1 || blocks > 4096) {
        std::fprintf(stderr, "bad cell: busy share limit 0.25, dispatch 0.05-5 ms, at most 30 s\n");
        return 2;
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::string(prop.gcnArchName).rfind("gfx1201", 0) != 0) {
        std::fprintf(stderr, "device 0 is %s: refusing to run (RX 9070 XT only)\n", prop.gcnArchName);
        return 3;
    }
    if (hotspot_c() >= 58.0) {
        std::printf("NOT RUN: hotspot %.0f C (a cell starts below 58 C)\n", hotspot_c());
        return 4;
    }
    Rec *rec, *grp;
    float * sink;
    CHECK(hipHostMalloc((void **) &rec, sizeof(Rec) * (MAXREC + 2)));
    CHECK(hipHostMalloc((void **) &grp, sizeof(Rec) * 2 * blocks));
    CHECK(hipMalloc(&sink, 4 * 64));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    FILE * csv = std::fopen(argv[1], "w");
    if (!csv) return 2;
    std::fprintf(csv, "# api=hip queue=hip ms=%.3f cad=%.3f groups=%d local=64 chunk=%d tsp_ns=1\n", ms, cad, blocks, chunk);
    std::fprintf(csv, "kind,idx,t_submit_mono_ns,t_done_mono_ns,ts_top,ts_bot,rt0,cyc0,rt1,cyc1,rtq1,cycq1,rtq3,cycq3,nrec,"
                      "wave_mhz_min,wave_mhz_max\n");
    const int nd = (int) std::floor(dur * 1000.0 / cad);
    std::printf("CELL api=hip dispatch_ms=%.3f cadence_ms=%.3f blocks=%d dispatches=%d uclk_start=%d hotspot=%.0f "
                "planned_busy_share=%.3f\n", ms, cad, blocks, nd, uclk_now(), hotspot_c(), ms / cad);
    const unsigned long long lim = (unsigned long long) std::llround(ms * 1e5), limc = 3000000ull;
    // module load and one short dispatch before the clock starts
    hipLaunchKernelGGL(spin, dim3(blocks), dim3(64), 0, 0, 100000ull, rec, grp, sink, chunk);
    CHECK(hipDeviceSynchronize());
    until_ns(mono_ns() + idle * 1e9);
    const double t0 = mono_ns();
    std::vector<double> late;
    int done = 0, hot = 0;
    for (int i = 0; i < nd; ++i, ++done) {
        if (hotspot_c() >= 88.0) {
            hot = 1;
            break;
        }
        until_ns(t0 + i * cad * 1e6);
        const double ts = mono_ns();
        CHECK(hipEventRecord(e0));
        hipLaunchKernelGGL(spin, dim3(blocks), dim3(64), 0, 0, lim, rec, grp, sink, chunk);
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        const double td = mono_ns();
        float em;
        CHECK(hipEventElapsedTime(&em, e0, e1));
        late.push_back((ts - (t0 + i * cad * 1e6)) / 1e3);
        row(csv, "call", i, ts, td, em * 1e6, rec, grp, blocks, lim);
    }
    if (calib && !hot)
        for (int i = 0; i < 3; ++i) {
            const double ts = mono_ns();
            CHECK(hipEventRecord(e0));
            hipLaunchKernelGGL(spin, dim3(blocks), dim3(64), 0, 0, limc, rec, grp, sink, chunk);
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            const double td = mono_ns();
            float em;
            CHECK(hipEventElapsedTime(&em, e0, e1));
            row(csv, "calib", i, ts, td, em * 1e6, rec, grp, blocks, limc);
        }
    const double t1 = mono_ns();
    std::fclose(csv);
    std::sort(late.begin(), late.end());
    const double p95 = late.empty() ? 0 : late[(size_t) (0.95 * (late.size() - 1))], mx = late.empty() ? 0 : late.back();
    std::printf("INFO mono_start=%.0f mono_end=%.0f\n", t0, t1);
    std::printf("VERDICT r9_clk cadence: %s submissions late p95 %.1f us (limit 200), largest %.1f us (limit 1000)\n",
                p95 <= 200 && mx <= 1000 ? "PASS" : "FAIL", p95, mx);
    if (hot) std::printf("VERDICT r9_clk thermal: FAIL hotspot reached 88 C, cell stopped after %d dispatches\n", done);
    std::printf("RESULT r9_clk dispatch_ms=%.3f dispatches=%d of %d uclk_end=%d\n", ms, done, nd, uclk_now());
    return hot ? 1 : 0;
}
