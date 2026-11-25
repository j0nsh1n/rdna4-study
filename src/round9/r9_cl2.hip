// Round 9, item C through HIP: does the data one kernel wrote sit in L2 for the next launch?
//
// Usage: r9_cl2 [reps] [steps] [random|page]   (default 20 repetitions, 1024 dependent loads, as the Vulkan cells of vkr9 cbar;
//        page: the increasing walk, next = slot + 1, in address order (item C2); the data is then the same every repetition, so
//        the race cell alternates the direction (next = slot - 1 on odd repetitions) to make its stale data differ)
// Cells, each repetition after a 512 MiB read that pushes everything out of L2 and the memory-side cache:
//   twolaunch  a producer kernel writes the 4 MiB region (32,768 lines of 128 bytes, dword pair 0 of each line is the
//              64-bit address of the next line: a full-cycle walk), the next launch chases `steps` loads through it
//   cold       the consumer chases a region nobody wrote (control: GDDR6)
//   fused      one block writes the region and then its wave 0 chases it (control: L0 or L2)
//   race       the consumer is launched before the producer (the deliberate break: the final pointer must be wrong)
// The consumer is mem_latency.hip's k_chase (inline-assembly loads, each waited), so cycles and REALTIME are those of
// the memory probes. Output: ROW lines (one per repetition), RESULT lines, same keys as vkr9 cbar.
#include <time.h>
#define main r9_mem_latency_main
#include "../study/mem_latency.hip"
#undef main

constexpr uint32_t NSLOT = 32768, LCG_A = 1664525u;
static uint32_t g_a = LCG_A;  // the walk multiplier of this run (1 for the page walk)

static int uclk_now() {
    const char * d = std::getenv("DRM_DEVICE");
    if (!d) return -1;
    std::string p = std::string(d) + "/gpu_metrics";
    FILE * f = std::fopen(p.c_str(), "rb");
    if (!f) return -1;
    unsigned char b[64];
    const size_t n = std::fread(b, 1, sizeof b, f);
    std::fclose(f);
    return n >= 60 ? (int) (b[58] | (b[59] << 8)) : -1;
}

// Line s: dwords 0-1 = address of line next(s); the other 30 dwords filler.
__device__ __forceinline__ void write_line(char * base, uint32_t slot, uint32_t lane, uint32_t mul, uint32_t c, uint32_t salt) {
    const uint32_t nxt = (mul * slot + c) & (NSLOT - 1);
    uint32_t * line = (uint32_t *) (base + (size_t) slot * 128);
    const uint64_t a = (uint64_t) base + (uint64_t) nxt * 128;
    line[lane] = lane == 0 ? (uint32_t) a : lane == 1 ? (uint32_t) (a >> 32) : (slot ^ salt) + lane;
}

__global__ __launch_bounds__(256) void k_prod(char * base, uint32_t a, uint32_t c, uint32_t salt) {
    const uint32_t t = blockIdx.x * 256 + threadIdx.x;
    write_line(base, t >> 5, t & 31, a, c, salt);
}

// One block of 256: writes everything, then wave 0 chases (cycles, REALTIME, final pointer as k_chase).
__global__ __launch_bounds__(256) void k_fused(char * base, uint32_t a, uint32_t c, uint32_t salt, uint64_t start,
                                               uint32_t steps, uint64_t * out) {
    for (uint32_t slot = threadIdx.x >> 5; slot < NSLOT; slot += 8) write_line(base, slot, threadIdx.x & 31, a, c, salt);
    __threadfence_block();
    __syncthreads();
    if (threadIdx.x < 32) {
        const uint64_t r0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
        const uint64_t p = chase(start, steps);
        const uint64_t c1 = wave_cycles();
        const uint64_t r1 = mem::realtime();
        if (threadIdx.x == 0) {
            out[0] = c1 - c0;
            out[1] = r1 - r0;
            out[2] = p;
        }
    }
}

static uint32_t lcg_final(uint32_t a, uint32_t start, uint32_t steps, uint32_t c) {
    uint32_t p = start;
    for (uint32_t i = 0; i < steps; ++i) p = (a * p + c) & (NSLOT - 1);
    return p;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const uint32_t steps = argc > 2 ? (uint32_t) std::atoi(argv[2]) / 16 * 16 : 1024;
    const bool page = argc > 3 && std::strcmp(argv[3], "page") == 0;
    if (argc > 3 && !page && std::strcmp(argv[3], "random") != 0) {
        std::fprintf(stderr, "walk must be random or page\n");
        return 2;
    }
    g_a = page ? 1u : LCG_A;
    const uint32_t c_b = page ? 1u : 1013904223u;  // the walk written once into region B
    std::printf("device %s (%s), r9_cl2 reps %d steps %u walk %s\n", prop.name, prop.gcnArchName, reps, steps, page ? "page" : "random");
    Ctx c;
    c.bytes = 8u << 20;
    CHECK(hipMalloc(&c.buf, c.bytes));
    char * ra = c.buf;
    char * rb = c.buf + (4u << 20);
    CHECK(hipMalloc(&c.flush, 512u << 20));
    mem::fill(c.flush, 512u << 20, mem::RANDOM);
    CHECK(hipMalloc(&c.out, 64));
    CHECK(hipMalloc(&c.sink, 512 * 256 * 4));
    // region B needs only to exist: its content is never used (cold) but the chase follows pointers, so it holds a
    // valid walk written once, before the first flush; the flush then pushes it out of the caches.
    hipLaunchKernelGGL(k_prod, dim3(NSLOT / 8), dim3(256), 0, 0, rb, g_a, c_b, 0x1234567u);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const char * cells[4] = {"twolaunch", "cold", "fused", "race"};
    for (int ci = 0; ci < 4; ++ci) {
        mem::cool_down(70.0);
        const int uc0 = uclk_now();
        std::vector<double> cps, nps;
        int okn = 0;
        for (int r = 0; r < reps; ++r) {
            const uint32_t cc = page ? ((ci == 3 && (r & 1)) ? NSLOT - 1 : 1u) : 1013904223u + 2u * (uint32_t) r, salt = 0x5bd1e995u * (uint32_t) (r + 1);
            const uint32_t start_slot = ((uint32_t) r * 7919u + 1u) & (NSLOT - 1);
            uint64_t o[4] = {0, 0, 0, 0};
            uint64_t expect;
            do_flush(c);
            if (ci == 0) {  // producer, then consumer
                hipLaunchKernelGGL(k_prod, dim3(NSLOT / 8), dim3(256), 0, 0, ra, g_a, cc, salt);
                const uint64_t start = (uint64_t) ra + (uint64_t) start_slot * 128;
                hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, 0u, steps, c.out);
                expect = (uint64_t) ra + (uint64_t) lcg_final(g_a, start_slot, steps, cc) * 128;
            } else if (ci == 1) {  // region B: its walk is the one written with c = 1013904223
                const uint64_t start = (uint64_t) rb + (uint64_t) start_slot * 128;
                hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, 0u, steps, c.out);
                expect = (uint64_t) rb + (uint64_t) lcg_final(g_a, start_slot, steps, c_b) * 128;
            } else if (ci == 2) {  // fused
                const uint64_t start = (uint64_t) ra + (uint64_t) start_slot * 128;
                hipLaunchKernelGGL(k_fused, dim3(1), dim3(256), 0, 0, ra, g_a, cc, salt, start, steps, c.out);
                expect = (uint64_t) ra + (uint64_t) lcg_final(g_a, start_slot, steps, cc) * 128;
            } else {  // race: consumer first, then the producer that should have come before it
                const uint64_t start = (uint64_t) ra + (uint64_t) start_slot * 128;
                hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, start, 0u, steps, c.out);
                CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
                hipLaunchKernelGGL(k_prod, dim3(NSLOT / 8), dim3(256), 0, 0, ra, g_a, cc, salt);
                expect = (uint64_t) ra + (uint64_t) lcg_final(g_a, start_slot, steps, cc) * 128;
            }
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            if (ci != 3) CHECK(hipMemcpy(o, c.out, sizeof o, hipMemcpyDeviceToHost));
            const int ok = o[2] == expect;
            std::printf("ROW c api=hip cell=%s queue=hip kind=launch order=%s walk=%s rep=%d cycles=%llu rt=%llu final=%llu expect=%llu ok=%d "
                        "steps=%u\n",
                        cells[ci], cells[ci], page ? "page" : "random", r, (unsigned long long) o[0], (unsigned long long) o[1], (unsigned long long) o[2],
                        (unsigned long long) expect, ok, steps);
            if (r > 0) {
                cps.push_back((double) o[0] / steps);
                nps.push_back((double) o[1] * 10.0 / steps);
                okn += ok;
            }
        }
        std::printf("RESULT r9_cl2 cell=%s reps=%d counted=%zu cyc_per_step_med=%.1f ns_per_step_med=%.1f pointer_ok=%d/%zu "
                    "uclk_start=%d uclk_end=%d\n",
                    cells[ci], reps, cps.size(), stats_of(cps).median, stats_of(nps).median, okn, cps.size(), uc0, uclk_now());
        std::fflush(stdout);
    }
    mem::report_hottest();
    return 0;
}
