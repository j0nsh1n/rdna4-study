// Shared helpers for the first-hour probes: error checks, the arch guard, event timing, the wave cycle counter.
// Every program is built for one arch with -DHARNESS_ARCH="gfxNNNN" (scripts/first-hour.sh does this).
#pragma once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef HARNESS_ARCH
#error "build with -DHARNESS_ARCH=\"gfx1100\" or \"gfx1201\""
#endif

#define CHECK(x)                                                                                    \
    do {                                                                                            \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

// Exit code 3 means "wrong card": the script reports it as a refusal, not as a measurement failure.
// The card record's ARCH (exported as CARD_ARCH by the script) must also agree, so a binary from build/gfx1100 is
// never timed on a gfx1201 card or the other way round.
inline hipDeviceProp_t require_arch() {
    int n = 0;
    CHECK(hipGetDeviceCount(&n));
    if (n < 1) {
        std::fprintf(stderr, "no HIP device visible (check HIP_VISIBLE_DEVICES)\n");
        std::exit(3);
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    const size_t len = std::strlen(HARNESS_ARCH);
    const char * want = std::getenv("CARD_ARCH");
    if (std::strncmp(prop.gcnArchName, HARNESS_ARCH, len) != 0 ||
        (prop.gcnArchName[len] != '\0' && prop.gcnArchName[len] != ':')) {
        std::fprintf(stderr, "device 0 is %s (%s) but this program was built for %s; refusing to run\n", prop.name,
                     prop.gcnArchName, HARNESS_ARCH);
        std::exit(3);
    }
    if (want && std::strcmp(want, HARNESS_ARCH) != 0) {
        std::fprintf(stderr, "card record says %s but this program was built for %s; refusing to run\n", want,
                     HARNESS_ARCH);
        std::exit(3);
    }
    return prop;
}

struct Stats {
    double median, min, max;
};

inline Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

// Times `launch` with HIP events: `warm` untimed runs, then `reps` timed ones. Returns milliseconds per run.
template <class F>
std::vector<double> time_ms(F launch, int reps, int warm = 5) {
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int i = 0; i < warm; ++i) {
        launch();
    }
    CHECK(hipDeviceSynchronize());
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        CHECK(hipEventRecord(t0));
        launch();
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float m;
        CHECK(hipEventElapsedTime(&m, t0, t1));
        ms.push_back(m);
    }
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    return ms;
}

// Shader clock counter of the wave. gfx11 reads the 20-bit HW_REG_SHADER_CYCLES (so differences are taken modulo
// 2^20 and a timed region must stay well under a million cycles); gfx12 reads the 64-bit SHADER_CYCLES_HI/LO pair.
// The empty volatile asm keeps the compiler from moving the read across the timed instructions.
__device__ __forceinline__ uint64_t wave_cycles() {
    asm volatile("" ::: "memory");
    const uint64_t c = __builtin_readcyclecounter();
    asm volatile("" ::: "memory");
    return c;
}

__device__ __forceinline__ uint64_t cycle_diff(uint64_t t0, uint64_t t1) {
#if defined(__GFX11__)
    return (t1 - t0) & 0xFFFFFu;
#else
    return t1 - t0;
#endif
}

inline bool arch_is_gfx11() { return std::strncmp(HARNESS_ARCH, "gfx11", 5) == 0; }

// One grep-able result line per probe: "VERDICT <probe>: PASS|FAIL|INFO <text>".
inline void verdict(const char * probe, const char * status, const char * text) {
    std::printf("VERDICT %s: %s %s\n", probe, status, text);
    std::fflush(stdout);
}
