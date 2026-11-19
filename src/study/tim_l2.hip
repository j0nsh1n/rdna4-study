// Timing study, part A: what makes one wave's L2 latency step up (mem_latency chase saw ~240 cycles up to 0.75-3 MiB
// although the L2 is 8 MiB, and the same step with in-page orders).
//
// Usage: tim_l2 chase [--break] [--only strides|offsets|odd]
//   chase  one wave (32 lanes, one address) follows a pointer cycle, each load waited for before the next. Warm-up: one
//          full cycle inside the same launch, then 8192 timed steps; the final pointer is checked against the host's
//          walk. Three questions, each point printed with the number of distinct lines AND the byte span:
//          1. node spacing 128, 256, 512, 1024, 4096, 65536 bytes against 2048..131072 lines (steps of about x1.4):
//             does the step follow the line count (capacity), the byte span (translation reach), or does it come early
//             for power-of-two spacings (set conflicts)? Spans above 4 GiB are skipped.
//          2. at 128-byte spacing, the chain's base offset by 0, 4 KiB, 64 KiB and 1 MiB + 128 B in a larger allocation.
//             (an extra series, --only odd, uses 384, 640, 1152 and 3072 bytes to tell set conflicts from capacity.)
//          3. random order against linear order (every point of 1 and 2 is run in both).
//          Verdict: calibration only (16 KiB random chase = L0 hits, about 81 cycles per load, +-10 %) and every final
//          pointer correct. The findings are the RESULT and STEP lines. --break runs the calibration point with
//          independent loads instead of a dependent chain: the verdict must FAIL.
#include "mem_common.h"

#define CH1 "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
#define CH16 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1

__device__ __forceinline__ uint64_t chase(uint64_t p, uint32_t steps) {
#pragma unroll 1
    for (uint32_t i = 0; i < steps / 16; ++i) {
        asm volatile(CH16 : "+v"(p)::"memory");
    }
    return p;
}

// out: [0] cycles, [1] REALTIME ticks, [2] final pointer.
__global__ __launch_bounds__(32) void k_chase(uint64_t start, uint32_t warm, uint32_t timed, uint64_t * out) {
    uint64_t p = chase(start, warm);
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
    p = chase(p, timed);
    const uint64_t c1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = p;
    }
}

// Deliberate break: independent loads from the start node, one wait per 16, so the time per "step" is not a latency.
__global__ __launch_bounds__(32) void k_chase_broken(uint64_t start, uint32_t timed, uint64_t * out) {
    uint64_t p = start, d = 0;
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
#define BR1 "global_load_b64 %1, %0, off\n\t"
#pragma unroll 1
    for (uint32_t i = 0; i < timed / 16; ++i) {
        asm volatile(BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 "s_wait_loadcnt 0"
                     : "+v"(p), "=&v"(d)::"memory");
    }
#undef BR1
    const uint64_t c1 = wave_cycles();
    const uint64_t r1 = mem::realtime();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = r1 - r0;
        out[2] = p ^ (d & 0);
    }
}

struct Pt {
    double cyc, cyc_min, cyc_max, ns, ns_min, ns_max;
    bool ok;
};

constexpr uint32_t TIMED = 8192;
constexpr int REPS = 5;

static uint64_t * g_out;

// One chain (nodes at `stride` from `dev`), REPS launches; median and range per load. `broken` swaps in the break kernel.
static Pt measure(char * dev, size_t nodes, size_t stride, bool random, bool broken) {
    std::vector<uint32_t> order;
    const uint64_t start = mem::build_chase(dev, nodes, stride, random, 7, &order);
    const uint32_t warm = (uint32_t) ((nodes + 15) / 16 * 16);
    const uint64_t expect = (uint64_t) dev + (uint64_t) order[(warm + TIMED) % nodes] * stride;
    std::vector<double> cy, ns;
    bool ok = true;
    for (int t = 0; t < REPS; ++t) {
        if (broken) {
            hipLaunchKernelGGL(k_chase_broken, dim3(1), dim3(32), 0, 0, start, TIMED, g_out);
        } else {
            hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, warm, TIMED, g_out);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[3];
        CHECK(hipMemcpy(o, g_out, sizeof o, hipMemcpyDeviceToHost));
        cy.push_back((double) o[0] / TIMED);
        ns.push_back((double) o[1] * 10.0 / TIMED);
        ok &= broken || o[2] == expect;
    }
    const Stats c = stats_of(cy), n = stats_of(ns);
    return {c.median, c.min, c.max, n.median, n.min, n.max, ok};
}

static bool g_all_ok = true;

static void print_pt(const char * sweep, const char * order, size_t stride, size_t lines, size_t off, const Pt & p) {
    std::printf("RESULT tim_l2 chase sweep %s order %s stride %zu lines %zu span_KiB %.0f base_off %zu cycles %.1f (%.1f-%.1f) "
                "ns %.1f (%.1f-%.1f) pointer %s\n",
                sweep, order, stride, lines, (double) lines * stride / 1024.0, off, p.cyc, p.cyc_min, p.cyc_max, p.ns,
                p.ns_min, p.ns_max, p.ok ? "ok" : "WRONG");
    std::fflush(stdout);
    g_all_ok &= p.ok;
}

// First line count at which the latency is at least 1.3x the lowest median seen for the same series.
static void print_step(const char * sweep, const char * order, size_t stride, size_t off, const std::vector<size_t> & lines,
                       const std::vector<double> & cyc) {
    double base = 1e18;
    for (double c : cyc) {
        base = std::min(base, c);
    }
    size_t at = 0;
    for (size_t i = 0; i < cyc.size(); ++i) {
        if (cyc[i] >= 1.3 * base) {
            at = i + 1;
            break;
        }
    }
    if (at) {
        std::printf("STEP tim_l2 sweep %s order %s stride %zu base_off %zu base %.0f cycles; first point >= 1.3x base: "
                    "%zu lines = %.0f KiB span (%.0f cycles)\n",
                    sweep, order, stride, off, base, lines[at - 1], (double) lines[at - 1] * stride / 1024.0, cyc[at - 1]);
    } else {
        std::printf("STEP tim_l2 sweep %s order %s stride %zu base_off %zu base %.0f cycles; no point >= 1.3x base "
                    "(up to %zu lines = %.0f KiB span)\n",
                    sweep, order, stride, off, base, lines.back(), (double) lines.back() * stride / 1024.0);
    }
    std::fflush(stdout);
}

static std::vector<size_t> line_counts() {
    std::vector<size_t> v;
    for (double x = 2048; x < 131072 * 0.99; x *= 1.4) {
        v.push_back((size_t) (x / 64 + 0.5) * 64);
    }
    v.push_back(131072);
    return v;
}

static void sweep(char * buf, const char * name, size_t stride, size_t off, size_t max_span) {
    for (int lin = 0; lin < 2; ++lin) {
        const char * order = lin ? "linear" : "random";
        std::vector<size_t> ls;
        std::vector<double> cy;
        for (size_t lines : line_counts()) {
            if (lines * stride > max_span) {
                continue;
            }
            const Pt p = measure(buf + off, lines, stride, !lin, false);
            print_pt(name, order, stride, lines, off, p);
            ls.push_back(lines);
            cy.push_back(p.cyc);
        }
        print_step(name, order, stride, off, ls, cy);
    }
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2 || std::strcmp(argv[1], "chase") != 0) {
        std::fprintf(stderr, "usage: tim_l2 chase [--break] [--only strides|offsets|odd]\n");
        return 2;
    }
    const bool brk = mem::has_flag(argc, argv, "--break");
    bool do_strides = true, do_offsets = true, do_odd = true;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--only") == 0) {
            do_strides = std::strcmp(argv[i + 1], "strides") == 0;
            do_offsets = std::strcmp(argv[i + 1], "offsets") == 0;
            do_odd = std::strcmp(argv[i + 1], "odd") == 0;
        }
    }
    std::printf("device %s (%s), mode chase%s\n", prop.name, prop.gcnArchName, brk ? " --break" : "");
    constexpr size_t SPAN = 4ull << 30, EXTRA = 2ull << 20;
    char * buf;
    CHECK(hipMalloc(&buf, SPAN + EXTRA));
    CHECK(hipMalloc(&g_out, 64));
    mem::clocks("tim_l2-start");

    // calibration: 128 lines of 128 B (16 KiB), random: every load hits L0
    const Pt cal = measure(buf, 128, 128, true, brk);
    print_pt("calibration", "random", 128, 128, 0, cal);
    const bool cal_ok = std::fabs(cal.cyc - 81.0) <= 8.1;

    if (!brk) {
        if (do_strides) {
            for (size_t stride : {128, 256, 512, 1024, 4096, 65536}) {
                sweep(buf, "stride", stride, 0, SPAN);
            }
        }
        if (do_odd) {  // spacings that are not powers of two: if only powers of two stepped early, these would not
            for (size_t stride : {384, 640, 1152, 3072}) {
                sweep(buf, "oddstride", stride, 0, SPAN);
            }
        }
        if (do_offsets) {
            for (size_t off : {0ull, 4096ull, 65536ull, (1ull << 20) + 128}) {
                sweep(buf, "offset", 128, off, SPAN);
            }
        }
    }
    char t[300];
    std::snprintf(t, sizeof t, "L0 calibration %.1f cycles (%.1f ns) per dependent load, known 81 +-10%%: %s; final pointers %s%s",
                  cal.cyc, cal.ns, cal_ok ? "yes" : "no", g_all_ok ? "all ok" : "WRONG",
                  brk ? " (deliberate break: independent loads)" : "");
    const bool pass = cal_ok && g_all_ok;
    verdict("tim_l2_chase", pass ? "PASS" : "FAIL", t);
    mem::clocks("tim_l2-end");
    mem::report_hottest();
    CHECK(hipFree(buf));
    CHECK(hipFree(g_out));
    return pass ? 0 : 1;
}
