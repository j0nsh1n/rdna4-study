// Round 9, item E: a background FP16 matrix loop that keeps the compute side busy with almost no memory traffic, the load
// that held the memory clock near 772 MHz in round 7 (memload.hip's wmma_kernel, same shape: four independent WMMA f16
// accumulators per wave, 256-thread blocks, three launches in flight).
//
// Usage: r9_bg <seconds> <ready-file> [blocks=128] [iters=20000] [cap C=76]
// Writes <ready-file> once the first launch has finished, then keeps launching until <seconds> have passed or TERM/INT
// arrives (the launches already enqueued finish first). Throttled (r9_bg_throttle.h): it pauses while the hotspot is at the
// cap and resumes 4 C below it, so the loop is still alive when the probes beside it run (the first version ran flat out,
// hit its tripwire after 2 s and was gone before the chases). The hard tripwire at 88 C ends it with exit status 5.
// <ready-file>.count holds the number of launches so far (rewritten about every 100 ms): the driver checks that it moves.
#include <hip/hip_runtime.h>
#include <signal.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "r9_bg_throttle.h"

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

__global__ __launch_bounds__(256) void wmma_kernel(const uint32_t * __restrict__ in, float * __restrict__ sink, int iters) {
    v8h a;
    uint32_t w[4];
    for (int i = 0; i < 4; ++i) w[i] = in[(threadIdx.x & 31) * 4 + i];
    __builtin_memcpy(&a, w, 16);
    v8f acc[4];
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;
    asm volatile("" : "+v"(a));
#if defined(__GFX12__)
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, a, acc[j]);
    }
#endif
    float s = 0;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 8; ++i) s += acc[j][i];
    if (s == 1234.5f) sink[threadIdx.x] = s;
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
    if (argc < 3) {
        std::fprintf(stderr, "usage: r9_bg <seconds> <ready-file> [blocks] [iters] [cap C]\n");
        return 2;
    }
    const double secs = std::atof(argv[1]);
    const int blocks = argc > 3 ? std::atoi(argv[3]) : 128, iters = argc > 4 ? std::atoi(argv[4]) : 20000;
    const double cap = argc > 5 ? std::atof(argv[5]) : 76.0;
    if (secs <= 0 || secs > 60 || cap < 50 || cap > 84) {
        std::fprintf(stderr, "r9_bg: at most 60 s, cap between 50 and 84 C\n");
        return 2;
    }
    const std::string count_path = std::string(argv[2]) + ".count";
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::string(prop.gcnArchName).rfind("gfx1201", 0) != 0) {
        std::fprintf(stderr, "device 0 is %s: refusing to run\n", prop.gcnArchName);
        return 3;
    }
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    uint32_t * in;
    float * sink;
    CHECK(hipMalloc(&in, 4096));
    CHECK(hipMemset(in, 0x3c, 4096));
    CHECK(hipMalloc(&sink, 1024));
    hipEvent_t ev[3];
    for (auto & e : ev) CHECK(hipEventCreate(&e));
    const double t0 = mono_s();
    int n = 0;
    bool announced = false, paused = false, tripped = false;
    double paused_s = 0, last_count = -1;
    while (!g_stop && mono_s() - t0 < secs) {
        const double hot = hotspot_c();
        const BgAct act = bg_act(hot, paused, cap);
        if (act == BG_TRIP) {
            tripped = true;
            break;
        }
        if (act == BG_PAUSE) {
            paused = true;
            timespec ts = {0, 20 * 1000 * 1000};
            nanosleep(&ts, nullptr);
            paused_s += 0.02;
            continue;
        }
        paused = false;
        hipLaunchKernelGGL(wmma_kernel, dim3(blocks), dim3(256), 0, 0, in, sink, iters);
        CHECK(hipEventRecord(ev[n % 3]));
        ++n;
        if (n >= 3) CHECK(hipEventSynchronize(ev[(n - 3) % 3]));
        if (!announced && n >= 3) {
            FILE * f = std::fopen(argv[2], "w");
            if (f) std::fclose(f);
            announced = true;
        }
        if (mono_s() - last_count > 0.1) {
            last_count = mono_s();
            FILE * f = std::fopen((count_path + ".tmp").c_str(), "w");
            if (f) {
                std::fprintf(f, "%d\n", n);
                std::fclose(f);
                std::rename((count_path + ".tmp").c_str(), count_path.c_str());
            }
        }
    }
    CHECK(hipDeviceSynchronize());
    if (FILE * f = std::fopen(count_path.c_str(), "w")) {
        std::fprintf(f, "%d\n", n);
        std::fclose(f);
    }
    std::printf("r9_bg: %d launches in %.2f s (paused %.2f s at the %.0f C cap), hotspot %.0f C%s\n", n, mono_s() - t0, paused_s, cap,
                hotspot_c(), tripped ? ", TRIPWIRE (88 C): the loop ended early" : "");
    return tripped ? 5 : 0;
}
