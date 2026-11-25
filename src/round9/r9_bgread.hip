// Round 9, item E2: a light background streaming read that holds the memory clock up while a single wave chases beside it.
//
// Usage: r9_bgread <seconds> <ready-file> <blocks> [MiB=1024] [iters=2048]
// A second process: its own 1 GiB buffer (a different one from the chase's), `blocks` work-groups of 256 threads, each thread
// reading 16 bytes per step with the groups spread over the buffer, `iters` steps per launch, three launches in flight. A few
// blocks (2 to 32 of the card's 64 CUs) keep the load light: the stream asks for about blocks x 256 x 16 B per memory latency,
// far below the 700 GB/s of the whole card, so the chase's path is not queued behind it. Its own mean bandwidth is printed
// (RESULT r9_bgread ... gbps=) and the analysis refuses a background above 10 % of 700 GB/s as "not light".
// Writes <ready-file> once the first launch has finished; runs until <seconds> (at most 60) or TERM/INT; stops at once if the
// hotspot reads 88 C or more.
#include <hip/hip_runtime.h>
#include <signal.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

// Thread t of block b reads 16 B at (b * 256 + t + phase + step * stride) mod words (stride = the grid's threads): one launch
// reads one contiguous window of blocks x 256 x 16 B x iters bytes once, and the next launch (phase) reads the next window, so the
// 1 GiB buffer is swept round and again and no line is read twice within 1 GiB: the traffic is GDDR6 traffic, not MALL hits.
__global__ __launch_bounds__(256) void k_stream(const uint4 * __restrict__ buf, uint32_t words, uint32_t iters, uint32_t phase,
                                                unsigned * __restrict__ sink) {
    const uint32_t stride = gridDim.x * 256u;
    uint32_t idx = (blockIdx.x * 256u + threadIdx.x + phase) % words;
    unsigned acc = 0;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint4 v = buf[idx];
        acc += v.x ^ v.y ^ v.z ^ v.w;
        idx += stride;
        if (idx >= words) idx -= words;
    }
    if (acc == 0x12345678u) sink[threadIdx.x] = acc;
}

static double mono_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
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

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: r9_bgread <seconds> <ready-file> <blocks> [MiB] [iters]\n");
        return 2;
    }
    const double secs = std::atof(argv[1]);
    const int blocks = std::atoi(argv[3]);
    const size_t mib = argc > 4 ? (size_t) std::atoll(argv[4]) : 1024;
    const uint32_t iters = argc > 5 ? (uint32_t) std::atoi(argv[5]) : 2048;
    if (secs <= 0 || secs > 60 || blocks < 1 || blocks > 512 || mib < 64 || mib > 2048 || iters < 16) {
        std::fprintf(stderr, "r9_bgread: bad arguments (at most 60 s, 1-512 blocks, 64-2048 MiB)\n");
        return 2;
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::string(prop.gcnArchName).rfind("gfx1201", 0) != 0) {
        std::fprintf(stderr, "device 0 is %s: refusing to run\n", prop.gcnArchName);
        return 3;
    }
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    uint4 * buf;
    unsigned * sink;
    const size_t bytes = mib << 20;
    CHECK(hipMalloc(&buf, bytes));
    CHECK(hipMemset(buf, 0x5a, bytes));
    CHECK(hipMalloc(&sink, 1024));
    const uint32_t words = (uint32_t) (bytes / 16);
    hipEvent_t ev[3], t_first;
    for (auto & e : ev) CHECK(hipEventCreate(&e));
    CHECK(hipEventCreate(&t_first));
    const double t0 = mono_s();
    int n = 0;
    bool announced = false;
    double t_ready = 0;
    while (!g_stop && mono_s() - t0 < secs && hotspot_c() < 88.0) {
        hipLaunchKernelGGL(k_stream, dim3(blocks), dim3(256), 0, 0, buf, words, iters, (uint32_t) (((uint64_t) n * blocks * 256u * iters) % words), sink);  // each launch reads the next window
        CHECK(hipEventRecord(ev[n % 3]));
        ++n;
        if (n >= 3) CHECK(hipEventSynchronize(ev[(n - 3) % 3]));
        if (!announced && n >= 3) {
            FILE * f = std::fopen(argv[2], "w");
            if (f) std::fclose(f);
            announced = true;
            t_ready = mono_s();
        }
    }
    CHECK(hipDeviceSynchronize());
    const double t1 = mono_s();
    // bandwidth over the steady part: launches after the first three, bytes = blocks x 256 x 16 B x iters per launch
    const double per_launch = (double) blocks * 256.0 * 16.0 * iters;
    const double steady_s = t1 - t_ready;
    const int steady_n = n > 3 ? n - 3 : 0;
    const double gbps = steady_s > 0 ? steady_n * per_launch / steady_s / 1e9 : 0.0;
    std::printf("RESULT r9_bgread blocks=%d MiB=%zu iters=%u launches=%d seconds=%.2f gbps=%.1f hotspot_C=%.0f\n", blocks, mib, iters, n,
                t1 - t0, gbps, hotspot_c());
    return 0;
}
