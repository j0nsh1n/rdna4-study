// Heavy FP8 WMMA HIP async-overlap probe:
// Pre-init handshake, boosted-SCLK calibration & logging (reads amdgpu gpu_metrics current_gfxclk),
// stream modes: norm (0), high (-1), cu32 (16 WGPs = 32 CUs), cu32_alt (16 alternating WGPs),
// cu16 (8 WGPs = 16 CUs), graph_norm, graph_high; chain=1 (monolithic) and chain=80 (multi-dispatch);
// plus explicit `break` subcommand that verifies the exactness check fails on corrupted expectation.
#include <ctype.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define HIP_CHECK(x)                                                                              \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_ != hipSuccess) {                                                                   \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)e_, hipGetErrorString(e_),       \
                    __FILE__, __LINE__);                                                          \
            exit(3);                                                                              \
        }                                                                                         \
    } while (0)

static double mono_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static double read_hotspot(const char *p) {
    if (!p || !p[0]) return -1.0;
    FILE *f = fopen(p, "r");
    if (!f) return -1.0;
    char buf[64];
    char *s = fgets(buf, sizeof buf, f);
    fclose(f);
    if (!s) return -1.0;
    char *end = nullptr;
    double v = strtod(buf, &end);
    if (end == buf || !isfinite(v) || v <= 0.0) return -1.0;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') return -1.0;
    return v * 1e-3;
}

static int read_gfxclk(const char *metrics_path) {
    if (!metrics_path || !metrics_path[0]) return -1;
    FILE *f = fopen(metrics_path, "rb");
    if (!f) return -1;
    unsigned char b[128];
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    if (n < 60) return -1;
    uint16_t u = 0;
    memcpy(&u, b + 54, 2); // current_gfxclk (MHz)
    return (int)u;
}

__global__ __launch_bounds__(256) void k_fp8_wmma(const i32x2 *__restrict__ A,
                                                  const i32x2 *__restrict__ B,
                                                  float *__restrict__ out,
                                                  uint32_t iters) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    i32x2 a = A[tid & 65535u];
    i32x2 b = B[tid & 65535u];
    f32x8 c0 = {0,0,0,0,0,0,0,0}, c1 = {0,0,0,0,0,0,0,0};
    f32x8 c2 = {0,0,0,0,0,0,0,0}, c3 = {0,0,0,0,0,0,0,0};
#pragma unroll 1
    for (uint32_t i = 0; i < iters; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c3);
    }
    if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0) {
        out[blockIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
    }
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double pctl(double *v, int n, double p) {
    if (n <= 0) return NAN;
    double *c = (double *)malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmp_d);
    int idx = (int)llround(p * (n - 1));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    double r = c[idx];
    free(c);
    return r;
}

static int run_one(hipStream_t s, hipGraphExec_t gexec, hipEvent_t e0, hipEvent_t e1,
                   const i32x2 *dA, const i32x2 *dB, float *dOut, float *hOut,
                   uint32_t grid, uint32_t chain, uint32_t iters, int do_break,
                   double *t_start, double *t_end, double *wall_ms, double *gpu_ms) {
    double ts = mono_ms();
    HIP_CHECK(hipEventRecord(e0, s));
    if (gexec) {
        HIP_CHECK(hipGraphLaunch(gexec, s));
    } else {
        for (uint32_t c = 0; c < chain; c++) {
            hipLaunchKernelGGL(k_fp8_wmma, dim3(grid), dim3(256), 0, s, dA, dB, dOut, iters);
        }
    }
    HIP_CHECK(hipEventRecord(e1, s));
    HIP_CHECK(hipStreamSynchronize(s));
    double te = mono_ms();
    float ems = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ems, e0, e1));
    HIP_CHECK(hipMemcpy(hOut, dOut, grid * sizeof(float), hipMemcpyDeviceToHost));
    float expect = 64.0f * (float)iters + (do_break ? 1.0f : 0.0f);
    int ok = 1;
    for (uint32_t i = 0; i < grid; i++) {
        if (fabsf(hOut[i] - expect) > 1e-3f) { ok = 0; break; }
    }
    *t_start = ts; *t_end = te; *wall_ms = te - ts; *gpu_ms = (double)ems;
    return ok;
}

// Thermal stop, fail closed: an unreadable sensor counts as too hot (Kimi R12 review #3).
static const char *g_hwtemp = nullptr;
static const double HOT_STOP_C = 92.0;
static bool too_hot(const char *where) {
    double hot = read_hotspot(g_hwtemp);
    if (hot < 0) { fprintf(stderr, "HIP THERMAL STOP sensor unreadable (HOTSPOT_PATH) at %s\n", where); return true; }
    if (hot >= HOT_STOP_C) { fprintf(stderr, "HIP THERMAL STOP %.1f C at %s\n", hot, where); return true; }
    return false;
}

static void boost_sclk(hipStream_t s, const i32x2 *dA, const i32x2 *dB, float *dOut, uint32_t grid, int ms) {
    if (ms <= 0) return;
    double end_t = mono_ms() + ms;
    while (mono_ms() < end_t) {
        if (too_hot("boost_sclk")) exit(4);
        hipLaunchKernelGGL(k_fp8_wmma, dim3(grid), dim3(256), 0, s, dA, dB, dOut, 15000u);
        HIP_CHECK(hipStreamSynchronize(s));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const char *hwtemp = getenv("HOTSPOT_PATH");
    g_hwtemp = hwtemp;
    // Before any HIP call: a missing or bad sensor must stop every subcommand, info included.
    if (too_hot("startup")) return 4;
    const char *mpath = getenv("METRICS_PATH");
    int least = 0, greatest = 0;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));

    const uint32_t grid = 256;
    const uint32_t N = 65536;
    i32x2 *hA = (i32x2 *)malloc(N * sizeof(i32x2));
    i32x2 *hB = (i32x2 *)malloc(N * sizeof(i32x2));
    for (uint32_t i = 0; i < N; i++) { hA[i] = {0x38383838, 0x38383838}; hB[i] = {0x38383838, 0x38383838}; }
    i32x2 *dA = nullptr, *dB = nullptr;
    float *dOut = nullptr, *hOut = (float *)malloc(grid * sizeof(float));
    HIP_CHECK(hipMalloc(&dA, N * sizeof(i32x2)));
    HIP_CHECK(hipMalloc(&dB, N * sizeof(i32x2)));
    HIP_CHECK(hipMalloc(&dOut, grid * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA, N * sizeof(i32x2), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, N * sizeof(i32x2), hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    if (!strcmp(argv[1], "info")) {
        hipStream_t sn, sh;
        HIP_CHECK(hipStreamCreateWithPriority(&sn, hipStreamNonBlocking, 0));
        HIP_CHECK(hipStreamCreateWithPriority(&sh, hipStreamNonBlocking, greatest));
        int pn = 999, ph = 999;
        HIP_CHECK(hipStreamGetPriority(sn, &pn));
        HIP_CHECK(hipStreamGetPriority(sh, &ph));
        const char *ign = getenv("DEBUG_HIP_IGNORE_STREAM_PRIORITY");
        printf("HIP_INFO prio_range=[least=%d,greatest=%d] s_norm=%d s_high=%d DEBUG_HIP_IGNORE_STREAM_PRIORITY=%s\n",
               least, greatest, pn, ph, ign ? ign : "(unset)");
        return 0;
    }

    if (!strcmp(argv[1], "break") || !strcmp(argv[1], "--break")) {
        hipStream_t s;
        HIP_CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, 0));
        double ts, te, wm, gm;
        if (too_hot("break")) return 4;
        int ok = run_one(s, nullptr, e0, e1, dA, dB, dOut, hOut, grid, 1, 64, 1, &ts, &te, &wm, &gm);
        printf("VERDICT hip_wmma_prio: %s (deliberate break: expect+1.0, exact=%s)\n",
               ok ? "PASS" : "FAIL", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    if (!strcmp(argv[1], "cal")) {
        uint32_t chain = (uint32_t)atoi(argv[2]);
        double target = atof(argv[3]);
        hipStream_t s;
        HIP_CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, 0));
        boost_sclk(s, dA, dB, dOut, grid, 80);
        uint32_t iters = (chain == 1) ? 35000u : 450u;
        double ts, te, wm, gm, gs[5];
        for (int i = 0; i < 5; i++) {
            if (too_hot("cal")) return 4;
            run_one(s, nullptr, e0, e1, dA, dB, dOut, hOut, grid, chain, iters, 0, &ts, &te, &wm, &gm);
            gs[i] = gm;
        }
        double gmed = pctl(gs, 5, 0.5);
        iters = (uint32_t)((double)iters * (target / (gmed > 0.1 ? gmed : 1.0)));
        if (iters < 4) iters = 4;
        for (int i = 0; i < 5; i++) {
            if (too_hot("cal")) return 4;
            run_one(s, nullptr, e0, e1, dA, dB, dOut, hOut, grid, chain, iters, 0, &ts, &te, &wm, &gm);
            gs[i] = gm;
        }
        printf("HIP_CAL chain=%u target_ms=%.2f iters=%u gpu_p50_ms=%.3f sclk=%d hot_c=%.1f\n",
               chain, target, iters, pctl(gs, 5, 0.5), read_gfxclk(mpath), read_hotspot(hwtemp));
        return 0;
    }

    if (!strcmp(argv[1], "run")) {
        // run <tag> <norm|high|cu32|cu32_alt|cu16|graph_norm|graph_high> <chain> <iters> <ntrials> <gap_ms> <warm_ms> <ready_file|-> <go_file|->
        const char *tag = argv[2];
        const char *mode = argv[3];
        uint32_t chain = (uint32_t)atoi(argv[4]);
        uint32_t iters = (uint32_t)atoi(argv[5]);
        int ntrials = atoi(argv[6]);
        int gap_ms = atoi(argv[7]);
        int warm_ms = atoi(argv[8]);
        const char *ready_f = (argc > 9) ? argv[9] : "-";
        const char *go_f = (argc > 10) ? argv[10] : "-";

        hipStream_t s = nullptr;
        int use_graph = 0;
        if (!strcmp(mode, "norm") || !strcmp(mode, "graph_norm")) {
            HIP_CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, 0));
            use_graph = !strcmp(mode, "graph_norm");
        } else if (!strcmp(mode, "high") || !strcmp(mode, "graph_high")) {
            HIP_CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, greatest));
            use_graph = !strcmp(mode, "graph_high");
        } else if (!strcmp(mode, "cu32")) {
            // gfx1201 ROCclr cuMask is in WGP units (32 WGPs = 64 CUs): 16 WGPs = 32 CUs
            uint32_t mask[1] = {0x0000ffffu};
            HIP_CHECK(hipExtStreamCreateWithCUMask(&s, 1, mask));
        } else if (!strcmp(mode, "cu32_alt")) {
            // 16 alternating WGPs across all 4 SEs = 32 CUs
            uint32_t mask[1] = {0x55555555u};
            HIP_CHECK(hipExtStreamCreateWithCUMask(&s, 1, mask));
        } else if (!strcmp(mode, "cu16")) {
            // 8 WGPs = 16 CUs
            uint32_t mask[1] = {0x000000ffu};
            HIP_CHECK(hipExtStreamCreateWithCUMask(&s, 1, mask));
        } else {
            fprintf(stderr, "Unknown mode: %s\n", mode);
            return 2;
        }

        uint32_t got_mask[2] = {0, 0};
        HIP_CHECK(hipExtStreamGetCUMask(s, 2, got_mask));
        int prio_val = 999;
        HIP_CHECK(hipStreamGetPriority(s, &prio_val));
        printf("STREAM_INFO tag=%s mode=%s prio=%d cumask=0x%08x_%08x graph=%d\n",
               tag, mode, prio_val, got_mask[1], got_mask[0], use_graph);
        fflush(stdout);

        hipGraphExec_t gexec = nullptr;
        hipGraph_t graph = nullptr;
        if (use_graph) {
            HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
            for (uint32_t c = 0; c < chain; c++) {
                hipLaunchKernelGGL(k_fp8_wmma, dim3(grid), dim3(256), 0, s, dA, dB, dOut, iters);
            }
            HIP_CHECK(hipStreamEndCapture(s, &graph));
            HIP_CHECK(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
        }

        // Pre-compile/warm driver context BEFORE signalling ready
        double ts, te, wm, gm;
        if (too_hot("pre-ready warm")) return 4;
        run_one(s, gexec, e0, e1, dA, dB, dOut, hOut, grid, 1, use_graph ? iters : 16, 0, &ts, &te, &wm, &gm);
        if (strcmp(ready_f, "-") != 0) {
            FILE *rf = fopen(ready_f, "w");
            if (rf) { fprintf(rf, "ready\n"); fclose(rf); }
        }
        if (strcmp(go_f, "-") != 0) {
            for (int w = 0; w < 500; w++) {
                if (access(go_f, F_OK) == 0) break;
                usleep(5000);
            }
        }
        if (warm_ms > 0) boost_sclk(s, dA, dB, dOut, grid, warm_ms);

        double walls[128], gpus[128];
        int n = 0, bad = 0, thermal_stop = 0;
        for (int i = 0; i < ntrials && n < 128; i++) {
            if (too_hot("trial")) {
                fprintf(stderr, "HIP THERMAL STOP at trial %d\n", i);
                thermal_stop = 1;
                break;
            }
            if (gap_ms > 0 && i > 0) usleep((useconds_t)gap_ms * 1000u);
            int ok = run_one(s, gexec, e0, e1, dA, dB, dOut, hOut, grid, chain, iters, 0, &ts, &te, &wm, &gm);
            if (!ok) bad++;
            walls[n] = wm; gpus[n] = gm; n++;
            printf("HIP_TRIAL tag=%s mode=%s prio=%d chain=%u idx=%d t_start_ms=%.3f t_end_ms=%.3f wall_ms=%.3f gpu_ms=%.3f sclk=%d hot_c=%.1f exact=%s\n",
                   tag, mode, prio_val, chain, i, ts, te, wm, gm, read_gfxclk(mpath), read_hotspot(hwtemp), ok ? "OK" : "FAIL");
            fflush(stdout);
        }
        printf("HIP_SUMMARY tag=%s mode=%s prio=%d chain=%u n=%d bad=%d wall_p50=%.3f wall_p95=%.3f wall_max=%.3f gpu_p50=%.3f gpu_p95=%.3f gpu_max=%.3f\n",
               tag, mode, prio_val, chain, n, bad,
               pctl(walls, n, 0.5), pctl(walls, n, 0.95), pctl(walls, n, 1.0),
               pctl(gpus, n, 0.5), pctl(gpus, n, 0.95), pctl(gpus, n, 1.0));
        if (thermal_stop) return 4;
        return bad ? 1 : 0;
    }
    return 2;
}
