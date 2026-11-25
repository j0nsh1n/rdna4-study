// Round 9, item E: whole-card read bandwidth at chosen buffer sizes ("split": every thread rereads one buffer of S bytes,
// passes inside the kernel; mem_stream.hip's `size:split` method at selected sizes only), each labelled with the memory
// clock read right before and right after (gpu_metrics current_uclk, sysfs read only).
//
// Usage: r9_bw <KiB[,KiB...]>
// mem_stream.hip's kernels and sweep() are included as they are (its main is renamed); each size is one sweep() call,
// which cools the card first, runs 2 untimed and 5 timed launches and prints its own RESULT line. The r9_bw line carries
// the sweep's own CLOCK_MONOTONIC window (t0_ns, t1_ns), so the analysis can apply the 90 % rule to the sampler rows inside it.
#include <time.h>
#define main r9_mem_stream_main
#include "../study/mem_stream.hip"
#undef main

static long long mono_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long) t.tv_sec * 1000000000ll + t.tv_nsec;
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: r9_bw <KiB,...>\n");
        return 2;
    }
    Ctx c;
    c.prop = require_arch();
    std::vector<double> sizes;
    double maxkib = 0;
    for (const char * p = argv[1]; *p;) {
        sizes.push_back(std::atof(p));
        maxkib = std::max(maxkib, sizes.back());
        const char * q = std::strchr(p, ',');
        if (!q) break;
        p = q + 1;
    }
    const int uc_start = uclk_now();
    CHECK(hipMalloc(&c.sink, 1 << 22));
    std::printf("device %s (%s), %d CUs, r9_bw uclk_process_start %d\n", c.prop.name, c.prop.gcnArchName,
                c.prop.multiProcessorCount, uc_start);
    alloc(c, (size_t) (maxkib * 1024), mem::RANDOM);
    mem::clocks("r9_bw-start");
    for (double kib : sizes) {
        const int u0 = uclk_now();
        const long long t0 = mono_ns();
        sweep<SPLIT>(c, "split", 512, {kib});
        const long long t1 = mono_ns();
        std::printf("RESULT r9_bw KiB=%.0f uclk_before=%d uclk_after=%d uclk_process_start=%d t0_ns=%lld t1_ns=%lld\n", kib, u0, uclk_now(),
                    uc_start, t0, t1);
        std::fflush(stdout);
    }
    mem::clocks("r9_bw-end");
    mem::report_hottest();
    CHECK(hipFree(c.buf));
    CHECK(hipFree(c.sink));
    return c.fail ? 1 : 0;
}
