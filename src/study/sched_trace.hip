// Lane 7 item 5: core clock, memory clock, reported power, temperatures and fan against load and time (read-only).
//
// Usage: sched_trace <f16|fp8|fma|stream|light|idle> --out <prefix> [--max-s 20] [--pre-s 1] [--post-s 30]
//   One trace: sysfs sampled every 50 ms from --pre-s before the load to --post-s after it; the load is a series of
//   launches of about 50 ms each, ended at --max-s seconds or when the hotspot reaches 95 C (checked between
//   launches, so the load ends within about 50 ms of the sample). Every launch records each wave's placement, cycle
//   counter and REALTIME, so the clock is also seen from inside.
//   f16, fp8  register-only matrix load as lane 4's mat_card (64 groups x 64 threads, 40 KiB unused LDS, CU mode:
//             one wave on each of the 128 SIMDs), 16 WMMA per iteration over 4 accumulators.
//   fma       16 independent FMA chains per lane in the accumulate form (v_dual_fmac_f32), 8 waves per SIMD.
//   stream    1 GiB of 16-byte loads, grid-stride, 4096 groups of 256.
//   light     the fma kernel on 4 groups (4 CUs of 64).
//   idle      no load: the card at rest for --max-s seconds.
// Files: <prefix>-sysfs.csv (50 ms rows), <prefix>-launch.csv (one row per launch). Build with -mcumode.
//
// Leftovers C, SC-5: sched_trace <fma|fp8|f16|idle> --burst-ms L --sample-ms S --out <prefix> [--break]
//   One short load of L ms (launches of about 10 ms; at most 200 ms, 100 ms for fma and fp8), started only below
//   55 C (else exit 7 without load) and stopped at 90 C; a host thread reads gpu_metrics' throttle bits, the hotspot
//   and the reported power every S ms (5 to 10) from 0.2 s before to 0.5 s after. Writes <prefix>-burst.csv.
//   Verdicts: at least 20 samples in the load window and at least 3 changes of the metrics' content (the table is
//   refreshed often enough to be read this fast); idle: no throttle bit in any sample (--break expects one).
#include "sched_common.h"

#include <functional>
#include <type_traits>

using namespace sched;

namespace {

typedef int i2v __attribute__((ext_vector_type(2)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i8v __attribute__((ext_vector_type(8)));

struct Rec {
    uint64_t pl, cyc, rt0, rt1;
};

#define WM(OP, ACC) OP " %" #ACC ", %4, %5, %" #ACC "\n"
#define WM4(OP) WM(OP, 0) WM(OP, 1) WM(OP, 2) WM(OP, 3)
#define WM16(OP) WM4(OP) WM4(OP) WM4(OP) WM4(OP)

template <int F>
using Frag = typename std::conditional<F == 0, i4v, i2v>::type;

__device__ __forceinline__ int wave_index() { return blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); }

template <int F>
__global__ __launch_bounds__(1024) void k_mat(const unsigned * __restrict__ ab, unsigned * __restrict__ sink,
                                              Rec * __restrict__ rec, int iters) {
    Frag<F> a, b;
    for (int k = 0; k < (int) (sizeof(Frag<F>) / 4); ++k) {
        a[k] = (int) ab[k];
        b[k] = (int) ab[4 + k];
    }
    asm volatile("" : "+v"(a), "+v"(b));
    i8v c0 = {}, c1 = {}, c2 = {}, c3 = {};
    const uint32_t pl = placement();
    const uint64_t cy0 = cyc(), rt0 = rt();
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        if constexpr (F == 0) {
            asm volatile(WM16("v_wmma_f32_16x16x16_f16") : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b));
        } else {
            asm volatile(WM16("v_wmma_f32_16x16x16_fp8_fp8") : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3) : "v"(a), "v"(b));
        }
    }
    const uint64_t cy1 = cyc(), rt1 = rt();
    unsigned x = 0;
    for (int k = 0; k < 8; ++k) {
        x ^= (unsigned) (c0[k] ^ c1[k] ^ c2[k] ^ c3[k]);
    }
    if (x == 0x7fc0ffeeu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cy1 - cy0, rt0, rt1};
    }
}

__global__ __launch_bounds__(1024) void k_fma(const float * __restrict__ k, float * __restrict__ sink,
                                              Rec * __restrict__ rec, int iters) {
    float k0 = k[0], k1 = k[1];
    asm volatile("" : "+v"(k0), "+v"(k1));
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = 0.5f + (float) ((threadIdx.x + i) & 63) * 1e-3f;
    }
    const uint32_t pl = placement();
    const uint64_t cy0 = cyc(), rt0 = rt();
#pragma unroll 4
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            x[i] = fmaf(k0, k1, x[i]);
        }
    }
    const uint64_t cy1 = cyc(), rt1 = rt();
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (s == -12345.0f) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cy1 - cy0, rt0, rt1};
    }
}

__global__ __launch_bounds__(256) void k_stream(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink,
                                                Rec * __restrict__ rec, int passes) {
    unsigned acc = 0;
    const uint32_t pl = placement();
    const uint64_t cy0 = cyc(), rt0 = rt();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
            const uint4 v = src[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    const uint64_t cy1 = cyc(), rt1 = rt();
    if (acc == 0xdeadbeefu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[wave_index()] = {pl, cy1 - cy0, rt0, rt1};
    }
}

using clk = std::chrono::steady_clock;

struct LaunchRow {
    double t0, t1, ms, rate, clk_med;
    int simds, minw, maxw;
};

}  // namespace

// SC-5: the fast sampler of a burst.
struct BurstRow {
    double t;
    double hot, power;
    int sclk;
    uint32_t thr;
    uint64_t ithr;
    int vgfx;
};

int run_burst(const std::string & mode, const std::string & out, double burst_ms, double sample_ms, bool brk,
              const std::function<void(int)> & launch, int iters_per_10ms) {
    if (burst_ms <= 0 || burst_ms > 200 || ((mode == "fma" || mode == "fp8") && burst_ms > 100) || sample_ms < 1 ||
        sample_ms > 10) {
        std::fprintf(stderr, "burst: length beyond the thermal rule (0.2 s; 0.1 s for fma and fp8) or sample period not 1..10 ms\n");
        return 2;
    }
    const double h0 = hotspot();
    if (h0 >= 55.0 && mode != "idle") {
        std::printf("NOT STARTED: hotspot %.0f C at the start (limit 55 C)\n", h0);
        return 7;
    }
    std::vector<BurstRow> rows;
    rows.reserve(4096);
    std::atomic<bool> run{true}, stop{false};
    const auto t0 = clk::now();
    std::thread th([&] {
        int k = 0;
        while (run) {
            Snap s{};
            read_metrics(s);
            BurstRow r{std::chrono::duration<double>(clk::now() - t0).count(), hotspot(),
                       read_num(paths().hw + "/power1_average", 1e-6), star_mhz(paths().drm + "/pp_dpm_sclk"), s.thr, s.ithr,
                       s.vgfx_mv};
            note_hot(r.hot);
            if (r.hot >= 90.0) {
                stop = true;
            }
            rows.push_back(r);
            ++k;
            std::this_thread::sleep_until(t0 + std::chrono::microseconds((long) (sample_ms * 1000 * k)));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const double la = std::chrono::duration<double>(clk::now() - t0).count();
    int launches = 0;
    if (mode != "idle") {
        while (std::chrono::duration<double>(clk::now() - t0).count() - la < burst_ms * 1e-3 && !stop) {
            launch(iters_per_10ms);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            ++launches;
        }
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds((long) burst_ms));
    }
    const double lb = std::chrono::duration<double>(clk::now() - t0).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    run = false;
    th.join();
    FILE * f = std::fopen((out + "-burst.csv").c_str(), "w");
    if (f) {
        std::fprintf(f, "t_from_load_start_s,hotspot_C,power_W_reported,sclk_MHz,throttle_status,indep_throttle_status,vgfx_mV\n");
        for (const BurstRow & r : rows) {
            std::fprintf(f, "%.4f,%.1f,%.0f,%d,0x%x,0x%llx,%d\n", r.t - la, r.hot, r.power, r.sclk, r.thr,
                         (unsigned long long) r.ithr, r.vgfx);
        }
        std::fclose(f);
    }
    int in_load = 0, changes = 0, with_bits = 0;
    uint64_t thr_or = 0;
    std::map<int, double> first_seen;
    for (size_t i = 0; i < rows.size(); ++i) {
        const BurstRow & r = rows[i];
        const bool inl = r.t >= la && r.t <= lb;
        in_load += inl;
        if (i > 0 && (r.ithr != rows[i - 1].ithr || r.vgfx != rows[i - 1].vgfx || r.thr != rows[i - 1].thr)) {
            ++changes;
        }
        if (r.t >= la && r.ithr) {
            ++with_bits;
            thr_or |= r.ithr;
            for (int b = 0; b < 64; ++b) {
                if (((r.ithr >> b) & 1) && !first_seen.count(b)) {
                    first_seen[b] = r.t - la;
                }
            }
        }
    }
    std::string fs;
    for (auto & kv : first_seen) {
        char b[64];
        std::snprintf(b, sizeof b, " %s@%.3fs", throttle_names(1ull << kv.first).c_str(), kv.second);
        fs += b;
    }
    double hmax = 0, pmax = 0;
    for (const BurstRow & r : rows) {
        hmax = std::max(hmax, r.hot);
        pmax = std::max(pmax, r.power);
    }
    std::printf("RESULT burst %s load_ms=%.1f launches=%d samples=%zu in_load=%d metric_changes=%d samples_with_bits_from_load_start=%d "
                "bits=%s(0x%llx) first_seen:%s hotspot_start=%.0f hotspot_max=%.0f power_max=%.0f stopped_at_90C=%d\n",
                mode.c_str(), (lb - la) * 1e3, launches, rows.size(), in_load, changes, with_bits,
                throttle_names(thr_or).c_str(), (unsigned long long) thr_or, fs.empty() ? " none" : fs.c_str(), h0, hmax,
                pmax, (int) stop.load());
    bool ok;
    char t[200];
    if (mode == "idle") {
        ok = brk ? with_bits > 0 : with_bits == 0;
        std::snprintf(t, sizeof t, "idle: %d samples with a throttle bit (%s)", with_bits, brk ? "break: expects some" : "expected 0");
    } else {
        ok = in_load >= 20 && changes >= 3;
        std::snprintf(t, sizeof t, "%d samples in the %.0f ms load window (>= 20), %d changes of the metrics' content (>= 3)",
                      in_load, (lb - la) * 1e3, changes);
    }
    verdict(("sched_trace burst_" + mode).c_str(), ok ? "PASS" : "FAIL", t);
    std::printf("HOTTEST %.0f C\n", hottest().load());
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: sched_trace <f16|fp8|fma|stream|light|idle> --out <prefix> [--max-s 20]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const std::string out = opt(argc, argv, "--out", "trace");
    const double max_s = std::min(20.0, std::atof(opt(argc, argv, "--max-s", "20")));
    const double pre_s = std::atof(opt(argc, argv, "--pre-s", "1"));
    const double post_s = std::atof(opt(argc, argv, "--post-s", "30"));

    unsigned *d_ab, *d_sink;
    float * d_k;
    Rec * d_rec;
    uint4 * d_src = nullptr;
    const unsigned hab[8] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u,
                             0x00003c00u, 0x3c000000u, 0x3c003c00u, 0x3c003c00u};
    const float hk[2] = {0.9999f, 1e-4f};
    CHECK(hipMalloc(&d_ab, sizeof hab));
    CHECK(hipMemcpy(d_ab, hab, sizeof hab, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_k, sizeof hk));
    CHECK(hipMemcpy(d_k, hk, sizeof hk, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_sink, 4096 * 256 * 4));
    const int maxw = 4096 * 8;
    CHECK(hipMalloc(&d_rec, maxw * sizeof(Rec)));
    const size_t bytes = (size_t) 1 << 30;

    // ops per launch per iteration unit, waves, launch function
    int nwaves = 0;
    double ops_per_iter = 0;  // FLOP (matrix: 8192 per instruction; FMA: 2) or bytes (stream, per pass)
    const char * unit = "TFLOPS";
    std::function<void(int)> launch;
    int iters = 0;
    double expect_rate = 0;  // lane 4 whole-card figures, same units
    if (mode == "f16" || mode == "fp8") {
        nwaves = 128;
        ops_per_iter = 128.0 * 16 * 8192;
        if (mode == "f16") {
            launch = [&](int n) { hipLaunchKernelGGL(k_mat<0>, dim3(64), dim3(64), 40 * 1024, 0, d_ab, d_sink, d_rec, n); };
            iters = 600000;
            expect_rate = 211.4;
        } else {
            launch = [&](int n) { hipLaunchKernelGGL(k_mat<1>, dim3(64), dim3(64), 40 * 1024, 0, d_ab, d_sink, d_rec, n); };
            iters = 1200000;
            expect_rate = 419.5;
        }
    } else if (mode == "fma" || mode == "light") {
        const int g = mode == "fma" ? 64 : 4;
        nwaves = g * 16;
        ops_per_iter = (double) g * 512 * 16 * 2;
        launch = [&, g](int n) { hipLaunchKernelGGL(k_fma, dim3(g), dim3(512), 40 * 1024, 0, d_k, (float *) d_sink, d_rec, n); };
        iters = 2300000;
        expect_rate = mode == "fma" ? 2 * 24.7 : 0;
    } else if (mode == "stream") {
        CHECK(hipMalloc(&d_src, bytes));
        CHECK(hipMemset(d_src, 1, bytes));
        nwaves = 4096 * 8;
        ops_per_iter = (double) bytes;
        unit = "TB/s";
        launch = [&](int n) { hipLaunchKernelGGL(k_stream, dim3(4096), dim3(256), 0, 0, d_src, bytes / 16, d_sink, d_rec, n); };
        iters = 30;
        expect_rate = 0.633;
    } else if (mode != "idle") {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    if (mode != "idle" && has_flag(argc, argv, "--burst-ms") && hotspot() >= 55.0) {
        std::printf("NOT STARTED: hotspot %.0f C at the start (limit 55 C)\n", hotspot());
        return 7;
    }
    if (mode != "idle") {
        // Short calibration launch (a tenth), then scale to about 50 ms.
        launch(std::max(1, iters / 10));
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        hipEvent_t a, b;
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
        CHECK(hipEventRecord(a));
        launch(std::max(1, iters / 10));
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms;
        CHECK(hipEventElapsedTime(&ms, a, b));
        iters = std::max(1, (int) (std::max(1, iters / 10) * 50.0 / std::max(ms, 0.01f)));
        std::printf("CALIB %s iters=%d (a tenth took %.2f ms)\n", mode.c_str(), iters, ms);
    }
    if (has_flag(argc, argv, "--burst-ms")) {
        // about 10 ms per launch (the calibration above scaled to 50 ms)
        return run_burst(mode, out, std::atof(opt(argc, argv, "--burst-ms", "100")), std::atof(opt(argc, argv, "--sample-ms", "5")),
                         has_flag(argc, argv, "--break"), launch, mode == "idle" ? 0 : std::max(1, iters / 5));
    }
    if (mode != "light" && mode != "idle" && !gate(mode == "stream" ? 60.0 : 55.0)) {
        verdict("sched_trace start", "FAIL", "hotspot did not fall below the start limit");
        return 1;
    }
    std::printf("START %s hotspot=%.0f C\n", mode.c_str(), hotspot());

    Sampler smp(50);
    smp.start();
    std::this_thread::sleep_for(std::chrono::milliseconds((long) (pre_s * 1000)));
    const double load_t0 = smp.now();
    std::vector<LaunchRow> rows;
    std::vector<Rec> h(std::max(nwaves, 1));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    bool stopped_hot = false;
    double hot_at = -1;
    if (mode == "idle") {
        std::this_thread::sleep_for(std::chrono::milliseconds((long) (max_s * 1000)));
    } else {
        while (smp.now() - load_t0 < max_s) {
            if (smp.hot) {
                stopped_hot = true;
                hot_at = smp.now() - load_t0;
                break;
            }
            LaunchRow r{};
            r.t0 = smp.now() - load_t0;
            CHECK(hipEventRecord(e0));
            launch(iters);
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            float ms;
            CHECK(hipEventElapsedTime(&ms, e0, e1));
            r.t1 = smp.now() - load_t0;
            r.ms = ms;
            r.rate = ops_per_iter * iters / (ms * 1e-3) / 1e12;
            CHECK(hipMemcpy(h.data(), d_rec, nwaves * sizeof(Rec), hipMemcpyDeviceToHost));
            std::vector<double> c;
            std::map<uint32_t, int> per;
            for (const Rec & x : h) {
                if (x.rt1 > x.rt0) {
                    c.push_back((double) x.cyc / ((double) (x.rt1 - x.rt0) / 1e8) / 1e6);
                }
                ++per[pl_simd((uint32_t) x.pl)];
            }
            r.clk_med = pct(c, 0.5);
            r.simds = (int) per.size();
            r.minw = 1 << 30;
            r.maxw = 0;
            for (auto & kv : per) {
                r.minw = std::min(r.minw, kv.second);
                r.maxw = std::max(r.maxw, kv.second);
            }
            rows.push_back(r);
        }
    }
    const double load_t1 = smp.now() - load_t0;
    std::this_thread::sleep_for(std::chrono::milliseconds((long) (post_s * 1000)));
    smp.stop();
    const std::vector<Snap> S = smp.copy();

    FILE * f = std::fopen((out + "-sysfs.csv").c_str(), "w");
    if (f) {
        std::fprintf(f, "t_from_load_start_s,");
        print_snap_header(f);
        for (const Snap & s : S) {
            std::fprintf(f, "%.3f,", s.t - load_t0);
            print_snap(f, s);
        }
        std::fclose(f);
    }
    f = std::fopen((out + "-launch.csv").c_str(), "w");
    if (f) {
        std::fprintf(f, "t0_s,t1_s,ms,rate_%s,counter_clock_MHz,simds,min_waves,max_waves\n", unit);
        for (const LaunchRow & r : rows) {
            std::fprintf(f, "%.3f,%.3f,%.3f,%.4f,%.1f,%d,%d,%d\n", r.t0, r.t1, r.ms, r.rate, r.clk_med, r.simds, r.minw,
                         r.maxw);
        }
        std::fclose(f);
    }

    // Summaries from the 50 ms rows. t is relative to the load start.
    auto in = [&](double a, double b, auto get) {
        std::vector<double> v;
        for (const Snap & s : S) {
            const double t = s.t - load_t0;
            if (t >= a && t < b) {
                v.push_back(get(s));
            }
        }
        return v;
    };
    auto sclk = [](const Snap & s) { return (double) s.sclk; };
    auto mclk = [](const Snap & s) { return (double) s.mclk; };
    auto pw = [](const Snap & s) { return s.power; };
    auto hot = [](const Snap & s) { return s.hot; };
    auto edge = [](const Snap & s) { return s.edge; };
    auto fan = [](const Snap & s) { return s.fan; };
    auto busy = [](const Snap & s) { return (double) s.busy; };
    const double L = load_t1;
    std::printf("RESULT trace %s load_s=%.2f launches=%zu stopped_at_95C=%s%s\n", mode.c_str(), L, rows.size(),
                stopped_hot ? "yes" : "no", stopped_hot ? (" after " + std::to_string(hot_at) + " s").c_str() : "");
    std::printf("RESULT trace %s idle_before: sclk %s | mclk %s | power_W_reported %s | hotspot %s | edge %s | fan %s | busy %s\n",
                mode.c_str(), dist(in(-pre_s, 0, sclk), "%.0f").c_str(), dist(in(-pre_s, 0, mclk), "%.0f").c_str(),
                dist(in(-pre_s, 0, pw), "%.0f").c_str(), dist(in(-pre_s, 0, hot), "%.0f").c_str(),
                dist(in(-pre_s, 0, edge), "%.0f").c_str(), dist(in(-pre_s, 0, fan), "%.0f").c_str(),
                dist(in(-pre_s, 0, busy), "%.0f").c_str());
    if (mode == "idle") {
        std::printf("RESULT trace idle rest: sclk %s | mclk %s | power_W_reported %s | hotspot %s | edge %s | fan %s | busy %s\n",
                    dist(in(0, L, sclk), "%.0f").c_str(), dist(in(0, L, mclk), "%.0f").c_str(),
                    dist(in(0, L, pw), "%.0f").c_str(), dist(in(0, L, hot), "%.0f").c_str(),
                    dist(in(0, L, edge), "%.0f").c_str(), dist(in(0, L, fan), "%.0f").c_str(),
                    dist(in(0, L, busy), "%.0f").c_str());
        verdict("sched_trace idle", "PASS", "idle trace recorded");
        std::printf("HOTTEST %.0f C\n", hottest().load());
        return 0;
    }
    const std::vector<double> steady = in(std::min(1.0, L / 2), L, sclk);
    const double sclk_med = pct(steady, 0.5);
    double t_full = -1;
    for (const Snap & s : S) {
        if (s.t - load_t0 >= 0 && s.sclk >= 0.97 * sclk_med) {
            t_full = s.t - load_t0;
            break;
        }
    }
    std::printf("RESULT trace %s time_to_97pct_of_steady_sclk_s=%.3f (50 ms sampling)\n", mode.c_str(), t_full);
    for (double a = 0; a < L - 0.05; a += (L > 4 ? 2.0 : 1.0)) {
        const double b = std::min(L, a + (L > 4 ? 2.0 : 1.0));
        std::vector<double> lc, lr;
        for (const LaunchRow & r : rows) {
            if (r.t0 >= a && r.t0 < b) {
                lc.push_back(r.clk_med);
                lr.push_back(r.rate);
            }
        }
        uint64_t thr_or = 0;
        int thr_n = 0, rows_n = 0;
        for (const Snap & s : S) {
            const double t = s.t - load_t0;
            if (t >= a && t < b) {
                thr_or |= s.ithr;
                thr_n += s.ithr != 0;
                ++rows_n;
            }
        }
        std::printf("WINDOW %s %.0f-%.0f s: sclk_med=%.0f counter_clk_med=%.0f rate_%s=%.3f power_W_reported_med=%.0f "
                    "hotspot_max=%.0f edge_max=%.0f mem_C_max=%.0f fan_rpm_med=%.0f mclk_med=%.0f vgfx_mV_med=%.0f "
                    "throttle=%s(0x%llx, in %d of %d samples)\n",
                    mode.c_str(), a, b, pct(in(a, b, sclk), 0.5), pct(lc, 0.5), unit, pct(lr, 0.5),
                    pct(in(a, b, pw), 0.5), pct(in(a, b, hot), 1.0), pct(in(a, b, edge), 1.0),
                    pct(in(a, b, [](const Snap & s) { return s.memt; }), 1.0), pct(in(a, b, fan), 0.5),
                    pct(in(a, b, mclk), 0.5), pct(in(a, b, [](const Snap & s) { return (double) s.vgfx_mv; }), 0.5),
                    throttle_names(thr_or).c_str(), (unsigned long long) thr_or, thr_n, rows_n);
    }
    for (double d : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0}) {
        if (d <= post_s + 0.01) {
            const auto v = in(L + d - 0.1, L + d + 0.1, hot);
            const auto e = in(L + d - 0.1, L + d + 0.1, edge);
            const auto c = in(L + d - 0.1, L + d + 0.1, sclk);
            const auto p = in(L + d - 0.1, L + d + 0.1, pw);
            std::printf("COOL %s +%.1f s: hotspot=%.0f edge=%.0f sclk=%.0f power_W_reported=%.0f fan=%.0f\n", mode.c_str(),
                        d, pct(v, 0.5), pct(e, 0.5), pct(c, 0.5), pct(p, 0.5),
                        pct(in(L + d - 0.1, L + d + 0.1, fan), 0.5));
        }
    }
    double below60 = -1, below55 = -1;
    for (const Snap & s : S) {
        const double t = s.t - load_t0 - L;
        if (t >= 0 && below60 < 0 && s.hot < 60) {
            below60 = t;
        }
        if (t >= 0 && below55 < 0 && s.hot < 55) {
            below55 = t;
        }
    }
    std::printf("RESULT trace %s cool_down_s below_60C=%.2f below_55C=%.2f\n", mode.c_str(), below60, below55);
    const double hot_max = pct(in(0, L + 0.2, hot), 1.0);
    std::vector<double> clk_all, rate_all, rate_first;
    int simds_min = 1 << 30, wmin = 1 << 30, wmax = 0;
    for (const LaunchRow & r : rows) {
        clk_all.push_back(r.clk_med);
        rate_all.push_back(r.rate);
        if (r.t0 < 1.5) {
            rate_first.push_back(r.rate);
        }
        simds_min = std::min(simds_min, r.simds);
        wmin = std::min(wmin, r.minw);
        wmax = std::max(wmax, r.maxw);
    }
    const double pmed = pct(in(1.0, L, pw), 0.5);
    const double rmed = pct(rate_all, 0.5);
    // Reported power / rate: pJ per FLOP (or per byte). The power is the driver's reading, scaled down by the mod.
    std::printf("RESULT trace %s steady: sclk_median=%.0f counter_clock_median=%.0f rate_median_%s=%.3f "
                "rate_first_1.5s=%.3f power_W_reported_median=%.0f hotspot_max=%.0f pJ_per_%s_reported=%.3f "
                "placement simds_min=%d waves_per_simd=%d..%d\n",
                mode.c_str(), sclk_med, pct(clk_all, 0.5), unit, rmed, pct(rate_first, 0.5), pmed, hot_max,
                mode == "stream" ? "byte" : "flop", pmed / (rmed * 1e12) * 1e12, simds_min, wmin, wmax);
    bool ok = true;
    char t[220];
    // Same window for both: launches that start after the first second (pp_dpm_sclk lags the real clock by a few
    // hundred ms after a step, so the ramp is left out), and the 50 ms rows of that span.
    const double wa = std::min(1.0, L / 2);
    std::vector<double> clk_w;
    for (const LaunchRow & r : rows) {
        if (r.t0 >= wa) {
            clk_w.push_back(r.clk_med);
        }
    }
    const double ratio = pct(clk_w, 0.5) / sclk_med;
    std::snprintf(t, sizeof t, "counter clock %.0f MHz against pp_dpm_sclk median %.0f MHz over %.1f-%.1f s (ratio %.4f, within 1 %%)",
                  pct(clk_w, 0.5), sclk_med, wa, L, ratio);
    const bool c1 = std::fabs(ratio - 1.0) <= 0.01;
    verdict(("sched_trace " + mode + " counter_vs_sclk").c_str(), c1 ? "PASS" : "FAIL", t);
    ok = ok && c1;
    if (expect_rate > 0) {
        const double r1 = pct(rate_first, 0.5) / expect_rate;
        const bool c2 = std::fabs(r1 - 1.0) <= 0.05 && simds_min == 128;
        std::snprintf(t, sizeof t, "first 1.5 s rate %.3f %s against lane 4's %.3f (ratio %.3f, within 5 %%), %d SIMDs placed",
                      pct(rate_first, 0.5), unit, expect_rate, r1, simds_min);
        verdict(("sched_trace " + mode + " rate").c_str(), c2 ? "PASS" : "FAIL", t);
        ok = ok && c2;
    }
    std::printf("HOTTEST %.0f C\n", hottest().load());
    return ok ? 0 : 1;
}
