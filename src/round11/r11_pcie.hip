// Round 11 item 4b: how fast and how late is the path between host memory and the card (HIP).
//   Copy speed: host-to-device, device-to-host and both at once, from pinned, pageable and pageable-misaligned-by-one-byte host
//   memory, 4 KiB to 1 GiB in steps of 4x, median / min / max of many repeats. Every cell is verified with an exact 64-bit hash
//   computed on the card, and the destination is zeroed before the timed loop, so a copy that did not happen is seen.
//   Latency: 4 B and 64 B round trips by copy-and-wait, and a flag ping-pong through coherent pinned host memory (card-timed by
//   the 100 MHz wall clock, CPU-timed by CLOCK_MONOTONIC).
//   Controls: rate above the PCIe ceiling rejects the cell; a card-to-card copy inside VRAM must read near the 633 GB/s anchor.
//   Deliberate breaks (each must make its control FAIL): one byte flipped on the card after the copy; the copy skipped; the
//   ping-pong with no responder.
// Usage: r11_pcie [--pci 0000:03:00.0] [--gts 32] [--lanes 16] [--max-mib 1024] [--reps 30] [--lat-iters 2000] [--pp-iters 5000]
//                 [--skip-xfer] [--skip-lat] [--skip-pp]
// The arithmetic of the ceiling, the hash and the breaks is in r11_pcie_logic.h and is tested on the CPU.
// Exit: 0 all controls pass and every break failed as required; 1 a control or break failed; 3 wrong card.
#include <chrono>
#include <thread>

#include "harness.h"
#include "r11_pcie_logic.h"

using r11::hash_words;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- card side ----------------------------------------------------------------------------------------------------------
__global__ void hash_kernel(const uint64_t * __restrict__ w, size_t n, unsigned long long * out) {
    uint64_t acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        acc += r11::word_term(w[i], i);
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down((long long) acc, o);
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(out, (unsigned long long) acc);
}

__global__ void flip_kernel(unsigned char * p, size_t at, unsigned char mask) { p[at] ^= mask; }

// Flag ping-pong. gpu_first: the card writes a=i, waits for b==i (the CPU answers), per-iteration ticks of the 100 MHz wall
// clock go to ticks[i]. !gpu_first: the CPU writes b=i, the card waits for b==i and answers a=i. Every wait is bounded.
__global__ void pingpong(uint32_t * a, uint32_t * b, int iters, int gpu_first, unsigned long long limit_ticks, uint32_t * ticks,
                         int * done) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int ok = 0;
    for (int i = 1; i <= iters; ++i) {
        const unsigned long long t0 = wall_clock64();
        if (gpu_first) __hip_atomic_store(a, (uint32_t) i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(b, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != (uint32_t) i) {
            if (wall_clock64() - t0 > limit_ticks) {
                *done = ok;
                return;
            }
        }
        if (!gpu_first) __hip_atomic_store(a, (uint32_t) i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        ticks[i - 1] = (uint32_t) (wall_clock64() - t0);
        ok = i;
    }
    *done = ok;
}

static uint64_t card_hash(const void * dev, size_t bytes, unsigned long long * dsum) {
    CHECK(hipMemset(dsum, 0, sizeof *dsum));
    hash_kernel<<<1024, 256>>>((const uint64_t *) dev, bytes / 8, dsum);
    CHECK(hipGetLastError());
    unsigned long long h = 0;
    CHECK(hipMemcpy(&h, dsum, sizeof h, hipMemcpyDeviceToHost));
    return h;
}

// ---- host memory kinds ----------------------------------------------------------------------------------------------------
enum Kind { PINNED, PAGEABLE, MISALIGNED };
static const char * kind_name(Kind k) { return k == PINNED ? "pinned" : k == PAGEABLE ? "pageable" : "pageable_off1"; }

struct HostBuf {
    void * base = nullptr;
    unsigned char * p = nullptr;  // the address handed to hipMemcpy
    size_t n = 0;
    Kind k = PINNED;
    bool alloc(size_t bytes, Kind kind) {
        k = kind;
        n = bytes;
        if (kind == PINNED) {
            if (hipHostMalloc(&base, bytes) != hipSuccess) return false;
        } else {
            base = std::malloc(bytes + 64);
            if (!base) return false;
        }
        p = (unsigned char *) base + (kind == MISALIGNED ? 1 : 0);
        std::memset(p, 0, bytes);
        return true;
    }
    void release() {
        if (!base) return;
        if (k == PINNED) (void) hipHostFree(base); else std::free(base);
        base = nullptr;
    }
};

static int g_bad = 0;       // controls that failed
static int g_rejected = 0;  // cells rejected above the ceiling
static int g_skipped = 0;   // sizes skipped because an allocation failed: the sweep is shorter than planned, which fails the verdict
static double g_ceiling = 0;

static void fail_cell(const char * why, const char * cell) {
    std::printf("CELLFAIL %s: %s\n", cell, why);
    ++g_bad;
}

struct Cell {
    const char * dir;
    Kind k;
    size_t bytes;
    int reps;
};

// One transfer of one kind, size and direction, verified before and after the timed loop.
static void run_xfer(const Cell & c, HostBuf & src, HostBuf & rb, void * dA, void * dB, unsigned long long * dsum,
                     uint64_t want_hash, hipStream_t s1, hipStream_t s2) {
    char name[160];
    std::snprintf(name, sizeof name, "%s %s %zu", c.dir, kind_name(c.k), c.bytes);
    const size_t n = c.bytes;
    const bool bidir = !std::strcmp(c.dir, "bidir");
    const bool h2d = !std::strcmp(c.dir, "h2d");
    // Prepare: the destination is zeroed so a copy that did not happen shows in the hash. For d2h and bidir the card holds the pattern.
    CHECK(hipMemset(dA, 0, n));
    CHECK(hipMemset(dB, 0, n));
    CHECK(hipMemcpy(dB, src.p, n, hipMemcpyHostToDevice));  // dB: the pattern, for d2h and the d2h half of bidir
    std::memset(rb.p, 0, n);
    CHECK(hipDeviceSynchronize());
    auto once = [&]() {
        if (bidir) {
            CHECK(hipMemcpyAsync(dA, src.p, n, hipMemcpyHostToDevice, s1));
            CHECK(hipMemcpyAsync(rb.p, dB, n, hipMemcpyDeviceToHost, s2));
            CHECK(hipStreamSynchronize(s1));
            CHECK(hipStreamSynchronize(s2));
        } else if (h2d) {
            CHECK(hipMemcpy(dA, src.p, n, hipMemcpyHostToDevice));
            CHECK(hipDeviceSynchronize());
        } else {
            CHECK(hipMemcpy(rb.p, dB, n, hipMemcpyDeviceToHost));
            CHECK(hipDeviceSynchronize());
        }
    };
    for (int i = 0; i < 3; ++i) once();
    // zero the destinations again so only the timed loop can fill them
    CHECK(hipMemset(dA, 0, n));
    std::memset(rb.p, 0, n);
    CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < c.reps; ++i) {
        const double a = now_s();
        once();
        t.push_back(now_s() - a);
    }
    bool ok = true;
    if (h2d || bidir) {
        if (card_hash(dA, n, dsum) != want_hash) { fail_cell("card does not hold the pattern after the host-to-device copies", name); ok = false; }
    }
    if (!h2d) {
        if (hash_words((const uint64_t *) rb.p, n / 8) != want_hash) { fail_cell("host does not hold the pattern after the device-to-host copies", name); ok = false; }
    }
    const double moved = bidir ? 2.0 * n : (double) n;
    std::vector<double> rate;
    for (double x : t) rate.push_back(r11::rate_gbps(moved, x));
    const double med = r11::median_of(rate), best = *std::max_element(rate.begin(), rate.end()),
                 worst = *std::min_element(rate.begin(), rate.end());
    // Ceiling: one direction for h2d / d2h, both for bidir.
    const double ceil_here = bidir ? 2.0 * g_ceiling : g_ceiling;
    const bool rejected = r11::above_ceiling(best, ceil_here);
    if (rejected) ++g_rejected;
    std::printf("CELL dir=%s mem=%s bytes=%zu reps=%d median_GBps=%.3f min_GBps=%.3f max_GBps=%.3f median_us=%.2f verify=%s%s\n", c.dir,
                kind_name(c.k), n, c.reps, med, worst, best, r11::median_of(t) * 1e6, ok ? "PASS" : "FAIL",
                rejected ? " REJECTED_ABOVE_CEILING" : "");
}

static void xfer_all(size_t max_bytes, int reps) {
    hipStream_t s1, s2;
    CHECK(hipStreamCreate(&s1));
    CHECK(hipStreamCreate(&s2));
    unsigned long long * dsum;
    CHECK(hipMalloc(&dsum, sizeof *dsum));
    for (Kind k : {PINNED, PAGEABLE, MISALIGNED}) {
        for (size_t bytes = 4096; bytes <= max_bytes; bytes *= 4) {
            HostBuf src, rb;
            void *dA = nullptr, *dB = nullptr;
            if (!src.alloc(bytes, k) || !rb.alloc(bytes, k) || hipMalloc(&dA, bytes) != hipSuccess ||
                hipMalloc(&dB, bytes) != hipSuccess) {
                std::printf("CELLSKIP mem=%s bytes=%zu: allocation failed (host or card memory); the planned sweep is cut short: FAIL\n", kind_name(k), bytes);
                ++g_skipped;
                ++g_bad;
                src.release();
                rb.release();
                break;
            }
            r11::fill_pattern((uint64_t *) src.p, bytes / 8, 0x11 + bytes);
            const uint64_t want = hash_words((const uint64_t *) src.p, bytes / 8);
            const int r = bytes >= (256u << 20) ? std::min(reps, 15) : reps;
            for (const char * d : {"h2d", "d2h", "bidir"}) {
                if (k == MISALIGNED && !std::strcmp(d, "bidir")) continue;
                run_xfer({d, k, bytes, r}, src, rb, dA, dB, dsum, want, s1, s2);
            }
            (void) hipFree(dA);
            (void) hipFree(dB);
            src.release();
            rb.release();
        }
    }
    (void) hipFree(dsum);
    (void) hipStreamDestroy(s1);
    (void) hipStreamDestroy(s2);
}

// Card-to-card copy inside card memory: the known-answer cross-check of the timer and the size arithmetic.
static void d2d_control(size_t bytes, int reps) {
    void *a, *b;
    CHECK(hipMalloc(&a, bytes));
    CHECK(hipMalloc(&b, bytes));
    CHECK(hipMemset(a, 0x5a, bytes));
    std::vector<double> t;
    for (int i = 0; i < reps + 3; ++i) {
        const double x = now_s();
        CHECK(hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice));
        CHECK(hipDeviceSynchronize());
        if (i >= 3) t.push_back(now_s() - x);
    }
    const double rw = 2.0 * bytes / r11::median_of(t) / 1e9;
    const bool ok = r11::d2d_in_band(rw);
    std::printf("CONTROL d2d bytes=%zu median_us=%.1f read_plus_write_GBps=%.1f anchor=633 %s\n", bytes, r11::median_of(t) * 1e6, rw,
                ok ? "PASS" : "FAIL");
    if (!ok) ++g_bad;
    (void) hipFree(a);
    (void) hipFree(b);
}

static void latency(int iters) {
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    void * d;
    CHECK(hipMalloc(&d, 4096));
    HostBuf pin, pg;
    pin.alloc(4096, PINNED);
    pg.alloc(4096, PAGEABLE);
    for (Kind k : {PINNED, PAGEABLE}) {
        HostBuf & h = k == PINNED ? pin : pg;
        for (size_t n : {(size_t) 4, (size_t) 64}) {
            for (const char * d_ : {"h2d", "d2h"}) {
                const bool is_h2d = d_[0] == 'h';
                std::vector<double> t;
                for (int i = 0; i < iters + 50; ++i) {
                    const double a = now_s();
                    if (is_h2d) CHECK(hipMemcpyAsync(d, h.p, n, hipMemcpyHostToDevice, s));
                    else CHECK(hipMemcpyAsync(h.p, d, n, hipMemcpyDeviceToHost, s));
                    CHECK(hipStreamSynchronize(s));
                    if (i >= 50) t.push_back((now_s() - a) * 1e6);
                }
                std::printf("LAT copy dir=%s mem=%s bytes=%zu iters=%d median_us=%.2f p95_us=%.2f min_us=%.2f\n", d_, kind_name(k), n,
                            iters, r11::median_of(t), r11::percentile_of(t, 0.95), *std::min_element(t.begin(), t.end()));
            }
        }
    }
    // zero-size copy: informational (the runtime may accept it as a no-op)
    const hipError_t z = hipMemcpy(d, pin.p, 0, hipMemcpyHostToDevice);
    std::printf("INFO zero-size hipMemcpy returned %d (%s)\n", (int) z, hipGetErrorString(z));
    (void) hipGetLastError();
    pin.release();
    pg.release();
    (void) hipFree(d);
    (void) hipStreamDestroy(s);
}

// Returns the iterations completed; `answer` false is the deliberate break (nobody answers, the card must time out).
static int pingpong_run(int iters, bool gpu_first, bool answer, double * med_us, double * p95_us, double * min_us, double limit_s) {
    uint32_t *a, *b, *ticks;
    int* done;
    CHECK(hipHostMalloc((void **) &a, 4, hipHostMallocCoherent));
    CHECK(hipHostMalloc((void **) &b, 4, hipHostMallocCoherent));
    CHECK(hipHostMalloc((void **) &ticks, sizeof(uint32_t) * iters, hipHostMallocCoherent));
    CHECK(hipHostMalloc((void **) &done, 4, hipHostMallocCoherent));
    __atomic_store_n(a, 0u, __ATOMIC_SEQ_CST);
    __atomic_store_n(b, 0u, __ATOMIC_SEQ_CST);
    *done = -1;
    const unsigned long long limit_ticks = (unsigned long long) (limit_s * 100e6);  // wall_clock64 runs at 100 MHz
    pingpong<<<1, 1>>>(a, b, iters, gpu_first ? 1 : 0, limit_ticks, ticks, done);
    CHECK(hipGetLastError());
    std::vector<double> cpu_us;
    int cpu_done = 0;
    if (answer) {
        for (int i = 1; i <= iters; ++i) {
            const double t0 = now_s();
            if (!gpu_first) __atomic_store_n(b, (uint32_t) i, __ATOMIC_RELAXED);
            while ((gpu_first ? __atomic_load_n(a, __ATOMIC_RELAXED) : __atomic_load_n(a, __ATOMIC_RELAXED)) != (uint32_t) i) {
                if (now_s() - t0 > limit_s) goto out;
            }
            if (gpu_first) __atomic_store_n(b, (uint32_t) i, __ATOMIC_RELAXED);
            cpu_us.push_back((now_s() - t0) * 1e6);
            cpu_done = i;
        }
    }
out:
    CHECK(hipDeviceSynchronize());
    {
        std::vector<double> us;
        if (gpu_first) {
            for (int i = 0; i < *done; ++i) us.push_back(ticks[i] / 100.0);  // 100 MHz: 10 ns per tick
        } else {
            us = cpu_us;
        }
        // Drop the first 10% (launch skew, cold caches).
        if (us.size() > 50) us.erase(us.begin(), us.begin() + us.size() / 10);
        *med_us = us.empty() ? -1 : r11::median_of(us);
        *p95_us = us.empty() ? -1 : r11::percentile_of(us, 0.95);
        *min_us = us.empty() ? -1 : *std::min_element(us.begin(), us.end());
    }
    const int d = *done;
    (void) cpu_done;
    (void) hipHostFree(a);
    (void) hipHostFree(b);
    (void) hipHostFree(ticks);
    (void) hipHostFree(done);
    return d;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const char * pci_want = "0000:03:00.0";
    double gts = 32;
    int lanes = 16, reps = 30, lat_iters = 2000, pp_iters = 5000;
    size_t max_mib = 1024;
    bool skip_xfer = false, skip_lat = false, skip_pp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : (char *) "0"; };
        if (a == "--pci") pci_want = next();
        else if (a == "--gts") gts = std::atof(next());
        else if (a == "--lanes") lanes = std::atoi(next());
        else if (a == "--max-mib") max_mib = std::strtoull(next(), nullptr, 10);
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--lat-iters") lat_iters = std::atoi(next());
        else if (a == "--pp-iters") pp_iters = std::atoi(next());
        else if (a == "--skip-xfer") skip_xfer = true;
        else if (a == "--skip-lat") skip_lat = true;
        else if (a == "--skip-pp") skip_pp = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    char pci[64];
    CHECK(hipDeviceGetPCIBusId(pci, sizeof pci, 0));
    // hip prints the bus id in lower case with a 4-digit domain, as sysfs does; compare case-insensitively.
    if (strcasecmp(pci, pci_want) != 0) {
        std::fprintf(stderr, "device 0 is at %s, the card record says %s; refusing to run\n", pci, pci_want);
        return 3;
    }
    g_ceiling = r11::pcie_ceiling_gbps(gts, lanes);
    const char * sdma = std::getenv("HSA_ENABLE_SDMA");
    std::printf("DEVICE %s (%s) pci %s HSA_ENABLE_SDMA=%s ceiling_GBps_per_direction=%.3f (%.1f GT/s x%d)\n", prop.name, prop.gcnArchName,
                pci, sdma ? sdma : "(unset)", g_ceiling, gts, lanes);
    if (!skip_xfer) {
        xfer_all(max_mib << 20, reps);
        d2d_control(256u << 20, 20);
        // Deliberate break 1: one byte flipped on the card after a verified copy.
        {
            const size_t n = 64u << 20;
            HostBuf src;
            void * d;
            unsigned long long * dsum;
            CHECK(hipMalloc(&d, n));
            CHECK(hipMalloc(&dsum, sizeof *dsum));
            src.alloc(n, PINNED);
            r11::fill_pattern((uint64_t *) src.p, n / 8, 99);
            const uint64_t want = hash_words((const uint64_t *) src.p, n / 8);
            CHECK(hipMemcpy(d, src.p, n, hipMemcpyHostToDevice));
            const bool clean = card_hash(d, n, dsum) == want;
            std::printf("BREAKPREP flip: clean copy verifies: %s\n", clean ? "yes" : "NO");
            if (!clean) ++g_bad;
            flip_kernel<<<1, 1>>>((unsigned char *) d, 12345, 0x01);
            CHECK(hipDeviceSynchronize());
            if (!r11::break_ok("flip", card_hash(d, n, dsum) == want)) ++g_bad;
            // Deliberate break 2: the copy skipped. 256 MiB with no copy takes microseconds: above the ceiling, flagged.
            const size_t big = 256u << 20;
            const double a = now_s();
            CHECK(hipDeviceSynchronize());
            const double rate = r11::rate_gbps((double) big, now_s() - a);
            std::printf("BREAKPREP skip: 256 MiB 'copied' in %.2f us = %.0f GB/s (ceiling %.1f)\n", (now_s() - a) * 1e6, rate, g_ceiling);
            if (!r11::break_ok("skip", !r11::above_ceiling(rate, g_ceiling))) ++g_bad;
            src.release();
            (void) hipFree(d);
            (void) hipFree(dsum);
        }
    }
    if (!skip_lat) latency(lat_iters);
    if (!skip_pp) {
        for (int gf = 1; gf >= 0; --gf) {
            double m, p, mn;
            const int done = pingpong_run(pp_iters, gf, true, &m, &p, &mn, 0.5);
            std::printf("LAT pingpong first=%s iters=%d completed=%d roundtrip_median_us=%.3f p95_us=%.3f min_us=%.3f oneway_median_us=%.3f %s\n",
                        gf ? "card(card-timed)" : "cpu(cpu-timed)", pp_iters, done, m, p, mn, m / 2, done == pp_iters ? "PASS" : "FAIL");
            if (done != pp_iters) ++g_bad;
        }
        // Deliberate break 3: nobody answers; the card's wait must time out and report no iteration completed.
        double m, p, mn;
        const int done = pingpong_run(3, true, false, &m, &p, &mn, 0.05);
        std::printf("BREAKPREP pingpong: no responder, card completed %d of 3\n", done);
        if (!r11::break_ok("pingpong_no_responder", done == 3)) ++g_bad;
    }
    std::printf("SUMMARY controls_failed=%d cells_rejected_above_ceiling=%d cells_skipped=%d\n", g_bad, g_rejected, g_skipped);
    const bool pass = g_bad == 0 && g_rejected == 0;
    verdict("r11_pcie", pass ? "PASS" : "FAIL", pass ? "all controls passed and every deliberate break failed as required" : "see CELLFAIL / CONTROL / BREAK lines");
    return pass ? 0 : 1;
}
