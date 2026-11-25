// Round 9, item E: single-wave dependent-load latency at chosen buffer sizes, each labelled with the memory clock
// (gpu_metrics current_uclk, sysfs read only) read right before and right after it.
//
// Usage: r9_lat lat  <KiB[,KiB...]> [trials=5] [walk=random] [min_ms=0]
//                                                 steady points: random walk over the whole buffer, a warming pass of
//                                                 min(nodes, 524288) steps first, as mem_latency.hip's `chase`; sizes of
//                                                 64 MiB and above also get the "fresh" point (4096-step warm-up).
//                                                 walk: random, page (increasing order, next = node + 1) or both (item E2).
//                                                 min_ms: repeat the trials of one point until at least that long (a first
//                                                 trial is thrown away to time one); every RESULT line carries t0_ns and
//                                                 t1_ns (CLOCK_MONOTONIC, the window of the timed trials, for labelling a
//                                                 point by the sampler's memory-clock readings during it)
//        r9_lat cold <KiB[,KiB...]>              one trial per size straight after its chain is built, short warm-up
//                                                 (min(nodes, 262144); 4096 from 128 MiB up), for processes started on an idle
//                                                 card: uclk_process_start is the reading before the first allocation
// The chase kernel, the chain builder and the pointer check are mem_latency.hip's, included as they are (its main is
// renamed). RESULT lines use the keys of the analysis (scripts/round9/analyse_hier.py).
#include <math.h>
#include <time.h>
#define main r9_mem_latency_main
#include "../study/mem_latency.hip"
#undef main

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

static long long mono_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long) t.tv_sec * 1000000000LL + t.tv_nsec;
}

static std::vector<double> parse_list(const char * s) {
    std::vector<double> v;
    for (const char * p = s; *p;) {
        v.push_back(std::atof(p));
        const char * q = std::strchr(p, ',');
        if (!q) break;
        p = q + 1;
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: r9_lat lat|cold <KiB,...> [trials]\n");
        return 2;
    }
    const int uc_start = uclk_now();
    const hipDeviceProp_t prop = require_arch();
    const std::string mode = argv[1];
    const std::vector<double> sizes = parse_list(argv[2]);
    const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
    const std::string walks = argc > 4 ? argv[4] : "random";
    const double min_ms = argc > 5 ? std::atof(argv[5]) : 0.0;
    if (walks != "random" && walks != "page" && walks != "both") {
        std::fprintf(stderr, "walk must be random, page or both\n");
        return 2;
    }
    if (mode != "lat" && mode != "cold") {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    std::printf("device %s (%s), r9_lat %s uclk_process_start %d\n", prop.name, prop.gcnArchName, mode.c_str(), uc_start);
    Ctx c;
    double maxkib = 0;
    for (double k : sizes) maxkib = std::max(maxkib, k);
    c.bytes = (size_t) (maxkib * 1024);
    CHECK(hipMalloc(&c.buf, c.bytes));
    CHECK(hipMalloc(&c.out, 64));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    mem::clocks("r9_lat-start");
    const uint32_t timed = 8192;
    for (double kib : sizes) {
        const size_t nodes = (size_t) (kib * 1024) / 128;
        for (int wi = 0; wi < 2; ++wi) {
            const bool page = wi == 1;
            if ((walks == "random" && page) || (walks == "page" && !page)) {
                continue;
            }
            std::vector<uint32_t> order;
            const uint64_t start = mem::build_chase(c.buf, nodes, 128, !page, 7, &order);
            uint32_t warm = (uint32_t) std::min<size_t>((nodes + 15) / 16 * 16, mode == "cold" ? 262144 : 524288);
            if (mode == "cold" && kib >= 131072) warm = 4096;
            mem::cool_down();
            for (int pass = 0; pass < (mode == "lat" && kib >= 65536 && !page ? 2 : 1); ++pass) {
                const uint32_t w = pass ? 4096 : warm;
                const uint64_t expect = (uint64_t) c.buf + (uint64_t) order[(w + timed) % nodes] * 128;
                int n_trials = mode == "cold" ? 1 : trials;
                if (min_ms > 0 && mode == "lat") {
                    const long long a0 = mono_ns();
                    run_chase(c, start, expect, w, timed, 1);  // thrown away: times one trial
                    const double one_ms = (mono_ns() - a0) * 1e-6;
                    n_trials = std::max(trials, (int) std::min(400.0, std::ceil(min_ms / std::max(one_ms, 1e-3))));
                }
                const int u0 = uclk_now();
                const long long t0 = mono_ns();
                const Lat l = run_chase(c, start, expect, w, timed, n_trials);
                const long long t1 = mono_ns();
                const int u1 = uclk_now();
                std::printf("RESULT r9_lat mode=%s walk=%s KiB=%.0f cycles=%.1f ns=%.2f pointer=%s uclk_before=%d uclk_after=%d "
                            "uclk_process_start=%d trials=%d t0_ns=%lld t1_ns=%lld\n", mode.c_str(),
                            page ? "page" : (pass ? "random_fresh" : "random"), kib, l.cyc, l.ns, l.ok ? "ok" : "WRONG", u0, u1,
                            uc_start, n_trials, t0, t1);
                std::fflush(stdout);
                c.fail |= !l.ok;
            }
        }
    }
    mem::clocks("r9_lat-end");
    mem::report_hottest();
    return c.fail ? 1 : 0;
}
