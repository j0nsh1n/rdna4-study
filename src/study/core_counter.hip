// Probe: the gfx12 shader cycle counter (HW_REG_SHADER_CYCLES_LO/HI) of one wave.
//
// Measures: (1) width and wrap, and the cost of one s_getreg read; (2) the rate against the fixed REALTIME counter
// (checked against host wall time) and so the core clock; (3) whether the counter keeps running while the wave waits
// on dependent memory loads; (4) while it sleeps (s_sleep); (5) whether different SIMDs' counters are offset.
// All counter reads are raw s_getreg / s_sendmsg_rtn in inline assembly, so the compiler cannot merge or move them.
// Run from the project root (it reads runs/latest/card.env for the sysfs folder with pp_dpm_sclk). --break expects
// REALTIME at 50 MHz instead of 100 MHz, to show the verdict can fail.
#include "../harness.h"

#include <atomic>
#include <cmath>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <map>

namespace {

__device__ __forceinline__ uint32_t rd_lo() {
    uint32_t r;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_LO)" : "=s"(r));
    return r;
}
__device__ __forceinline__ uint32_t rd_hi() {
    uint32_t r;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)" : "=s"(r));
    return r;
}
// HI, LO, HI again; retry until HI did not change, so the pair is one consistent 64-bit value.
__device__ __forceinline__ uint64_t rd_c64() {
    uint32_t h1, l, h2;
    do {
        h1 = rd_hi();
        l = rd_lo();
        h2 = rd_hi();
    } while (h1 != h2);
    return ((uint64_t) h1 << 32) | l;
}
__device__ __forceinline__ uint64_t rd_rt() {
    uint64_t r;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0" : "=s"(r));
    return r;
}

// 256 independent-chain v_fma_f32 (8 chains x 32): pure VALU work that the compiler cannot delete.
__device__ __forceinline__ void spin_chunk(float (&a)[8], float b, float c) {
#pragma unroll 1
    for (int k = 0; k < 32; ++k) {
        asm volatile(
            "v_fma_f32 %0, %0, %8, %9\n\tv_fma_f32 %1, %1, %8, %9\n\tv_fma_f32 %2, %2, %8, %9\n\t"
            "v_fma_f32 %3, %3, %8, %9\n\tv_fma_f32 %4, %4, %8, %9\n\tv_fma_f32 %5, %5, %8, %9\n\t"
            "v_fma_f32 %6, %6, %8, %9\n\tv_fma_f32 %7, %7, %8, %9"
            : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3]), "+v"(a[4]), "+v"(a[5]), "+v"(a[6]), "+v"(a[7])
            : "v"(b), "v"(c));
    }
}

__device__ __forceinline__ float fold8(const float (&a)[8]) {
    return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
}

// Spins for `ticks` REALTIME ticks (or 4e9 cycles, whichever first). o[0..3] = rt0, c0, c1, rt1.
__device__ void spin_ticks(uint64_t ticks, float seed, uint64_t * o, float & sink) {
    float a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = seed + (float) i;
    }
    const float b = 0.999f, c = 0.001f;
    const uint64_t rt0 = rd_rt();
    const uint64_t c0 = rd_c64();
    const uint32_t lo0 = rd_lo();
    uint64_t rt;
    do {
        spin_chunk(a, b, c);
        rt = rd_rt();
    } while (rt - rt0 < ticks && (rd_lo() - lo0) < 4000000000u);
    const uint64_t c1 = rd_c64();
    const uint64_t rt1 = rd_rt();
    o[0] = rt0;
    o[1] = c0;
    o[2] = c1;
    o[3] = rt1;
    sink += fold8(a);
}

// ---- Item 1: width, wrap, read cost. buf = (HI, LO, HI) triples; info[0] = sample count, info[1] = REALTIME ticks run.
constexpr int CAP = 16384;
constexpr int READS = 17;

__global__ __launch_bounds__(32) void k_wrap(uint32_t * buf, uint64_t * info, uint32_t * rdcost, uint64_t max_ticks,
                                             float seed) {
    float a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = seed + (float) i;
    }
    const float b = 0.999f, c = 0.001f;
    // cost of one read: 17 back-to-back reads of LO, then the differences
    uint32_t r[READS];
#define R(i) r[i] = rd_lo();
    R(0) R(1) R(2) R(3) R(4) R(5) R(6) R(7) R(8) R(9) R(10) R(11) R(12) R(13) R(14) R(15) R(16)
#undef R
    if (threadIdx.x == 0) {
        for (int i = 0; i < READS - 1; ++i) {
            rdcost[i] = r[i + 1] - r[i];
        }
    }
    const uint64_t rt0 = rd_rt();
    uint32_t n = 0, post = 0, first_hi = 0;
    bool wrapped = false;
    while (n < CAP) {
        const uint32_t h1 = rd_hi();
        const uint32_t l = rd_lo();
        const uint32_t h2 = rd_hi();
        if (n == 0) {
            first_hi = h1;
        }
        if (threadIdx.x == 0) {
            buf[3 * n] = h1;
            buf[3 * n + 1] = l;
            buf[3 * n + 2] = h2;
        }
        ++n;
        if (h2 != first_hi) {
            wrapped = true;
        }
        if (wrapped && ++post >= 32) {
            break;
        }
        if (rd_rt() - rt0 > max_ticks) {
            break;
        }
        // about 1e6 cycles of VALU work to the next sample
        do {
            spin_chunk(a, b, c);
        } while (rd_lo() - l < 1000000u);
    }
    const uint64_t rt1 = rd_rt();
    if (threadIdx.x == 0) {
        info[0] = n;
        info[1] = rt1 - rt0;
        info[2] = (uint64_t) __float_as_uint(fold8(a));
    }
}

// ---- Item 2: rate. out[0..3] = rt0, c0, c1, rt1.
__global__ __launch_bounds__(32) void k_rate(uint64_t * out, uint64_t ticks, float seed) {
    uint64_t o[4];
    float sink = 0.f;
    spin_ticks(ticks, seed, o, sink);
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            out[i] = o[i];
        }
        out[4] = (uint64_t) __float_as_uint(sink);
    }
}

// ---- Items 3 and 4: per trial a VALU control (2 ms), then the test. out[t*10 ..]: 4 control values, 4 test values,
// final pointer, sink.
constexpr int TRIALS = 8;
constexpr int STEPS = 4096;

__global__ __launch_bounds__(32) void k_chase(const uint64_t * starts, uint64_t * out, uint64_t ctrl_ticks, float seed) {
    float sink = 0.f;
    for (int t = 0; t < TRIALS; ++t) {
        uint64_t o[4];
        spin_ticks(ctrl_ticks, seed, o, sink);
        uint64_t p = starts[t];
        const uint64_t rt0 = rd_rt();
        const uint64_t c0 = rd_c64();
#pragma unroll 1
        for (int i = 0; i < STEPS / 16; ++i) {
#define L "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
            asm volatile(L L L L L L L L L L L L L L L L : "+v"(p)::"memory");
#undef L
        }
        const uint64_t c1 = rd_c64();
        const uint64_t rt1 = rd_rt();
        if (threadIdx.x == 0) {
            uint64_t * q = out + t * 10;
            q[0] = o[0];
            q[1] = o[1];
            q[2] = o[2];
            q[3] = o[3];
            q[4] = rt0;
            q[5] = c0;
            q[6] = c1;
            q[7] = rt1;
            q[8] = p;
            q[9] = (uint64_t) __float_as_uint(sink);
        }
    }
}

constexpr int SLEEP_TRIALS = 3;
constexpr int SLEEP_BLOCKS = 2000;  // x 16 s_sleep 127 = 32000 sleeps, about 0.1 s

__global__ __launch_bounds__(32) void k_sleep(uint64_t * out, uint64_t ctrl_ticks, float seed) {
    float sink = 0.f;
    for (int t = 0; t < SLEEP_TRIALS; ++t) {
        uint64_t o[4];
        spin_ticks(ctrl_ticks, seed, o, sink);
        const uint64_t rt0 = rd_rt();
        const uint64_t c0 = rd_c64();
#pragma unroll 1
        for (int i = 0; i < SLEEP_BLOCKS; ++i) {
#define S "s_sleep 127\n\t"
            asm volatile(S S S S S S S S S S S S S S S S ::: "memory");
#undef S
        }
        const uint64_t c1 = rd_c64();
        const uint64_t rt1 = rd_rt();
        if (threadIdx.x == 0) {
            uint64_t * q = out + t * 10;
            q[0] = o[0];
            q[1] = o[1];
            q[2] = o[2];
            q[3] = o[3];
            q[4] = rt0;
            q[5] = c0;
            q[6] = c1;
            q[7] = rt1;
            q[8] = 0;
            q[9] = (uint64_t) __float_as_uint(sink);
        }
    }
}

// ---- Item 5: one wave per work-group. out[b*4 ..] = rt0, c0, c1, rt1 around a 1 ms spin.
__global__ __launch_bounds__(32) void k_simd(uint64_t * out, float seed) {
    uint64_t o[4];
    float sink = 0.f;
    spin_ticks(100000, seed, o, sink);
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            out[blockIdx.x * 4 + i] = o[i];
        }
        if (sink == 12345.f) {
            out[blockIdx.x * 4] = 0;
        }
    }
}

// ---------------------------------------------------------------- host
using clk = std::chrono::steady_clock;

double median_of(std::vector<double> v) { return stats_of(v).median; }

std::string drm_device() {
    std::ifstream f("runs/latest/card.env");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("DRM_DEVICE=", 0) == 0) {
            std::string v = line.substr(11);
            if (!v.empty() && v.front() == '\'') {
                v = v.substr(1, v.size() - 2);
            }
            return v;
        }
    }
    return "";
}

// Current level of pp_dpm_sclk: the line marked '*', e.g. "1: 1303Mhz *" -> 1303. Returns -1 when unreadable.
int read_sclk(const std::string & path, std::string * raw) {
    std::ifstream f(path);
    if (!f) {
        return -1;
    }
    std::string line, all;
    int mhz = -1;
    while (std::getline(f, line)) {
        all += line + "|";
        if (line.find('*') != std::string::npos) {
            const size_t c = line.find(':');
            if (c != std::string::npos) {
                mhz = std::atoi(line.c_str() + c + 1);
            }
        }
    }
    if (raw) {
        *raw = all;
    }
    return mhz;
}

struct Wrap {
    bool ok = false, wrap_seen = false;
    uint64_t vmax = 0;
    size_t n = 0;
    int wraps = 0, bad = 0;
    double ticks_s = 0, span_cycles = 0, med_step = 0;
};

Wrap analyse_wrap(const std::vector<uint32_t> & b, size_t n) {
    Wrap w;
    w.n = n;
    std::vector<uint64_t> v(n);
    std::vector<double> steps;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h1 = b[3 * i], l = b[3 * i + 1], h2 = b[3 * i + 2];
        if (h2 < h1 || h2 - h1 > 1) {
            ++w.bad;
        }
        // HI changed between its two reads: LO belongs to the new HI only if it is small (just wrapped)
        const uint32_t h = (h1 != h2 && l < 0x80000000u) ? h2 : h1;
        v[i] = ((uint64_t) h << 32) | l;
    }
    for (size_t i = 1; i < n; ++i) {
        if (v[i] <= v[i - 1]) {
            ++w.bad;  // not monotonic
        }
        const uint32_t hp = (uint32_t) (v[i - 1] >> 32), hn = (uint32_t) (v[i] >> 32);
        const uint32_t lp = (uint32_t) v[i - 1], ln = (uint32_t) v[i];
        // HI must step by exactly one when LO wraps, and by zero when it does not
        if ((ln < lp) != (hn == hp + 1) || (hn != hp && hn != hp + 1)) {
            ++w.bad;
        }
        if (hn == hp + 1) {
            ++w.wraps;
        }
        steps.push_back((double) (v[i] - v[i - 1]));
        w.vmax = std::max(w.vmax, v[i]);
    }
    w.vmax = std::max(w.vmax, v[0]);
    w.wrap_seen = w.wraps > 0;
    w.span_cycles = n > 1 ? (double) (v[n - 1] - v[0]) : 0;
    w.med_step = steps.empty() ? 0 : median_of(steps);
    w.ok = w.bad == 0 && n > 2;
    return w;
}

}  // namespace

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const bool brk = argc > 1 && std::strcmp(argv[1], "--break") == 0;
    const double expect_hz = brk ? 50e6 : 100e6;
    std::printf("counter probe on %s (%s)%s\n", prop.name, prop.gcnArchName,
                brk ? " --break: expecting REALTIME at 50 MHz" : "");
    bool pass = true;
    std::string why;
    auto fail = [&](const std::string & s) {
        pass = false;
        why += (why.empty() ? "" : "; ") + s;
    };

    // ---- 1. width, wrap, read cost
    {
        uint32_t * d_buf;
        uint64_t * d_info;
        uint32_t * d_rd;
        CHECK(hipMalloc(&d_buf, CAP * 3 * 4));
        CHECK(hipMalloc(&d_info, 64));
        CHECK(hipMalloc(&d_rd, 64));
        CHECK(hipMemset(d_buf, 0, CAP * 3 * 4));
        hipLaunchKernelGGL(k_wrap, dim3(1), dim3(32), 0, 0, d_buf, d_info, d_rd, (uint64_t) 280000000, 1.0f);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t info[3];
        uint32_t rd[READS - 1];
        CHECK(hipMemcpy(info, d_info, sizeof info, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(rd, d_rd, sizeof rd, hipMemcpyDeviceToHost));
        std::vector<uint32_t> buf(info[0] * 3);
        CHECK(hipMemcpy(buf.data(), d_buf, buf.size() * 4, hipMemcpyDeviceToHost));
        Wrap w = analyse_wrap(buf, info[0]);
        w.ticks_s = (double) info[1] / expect_hz;
        std::vector<double> rc(rd, rd + READS - 1);
        const Stats rs = stats_of(rc);
        std::printf("RESULT counter wrap samples=%zu run_s=%.3f span_cycles=%.4g median_step=%.0f wraps_seen=%d "
                    "bad=%d max_value=0x%016llx (%llu) max_hi=%u\n",
                    w.n, w.ticks_s, w.span_cycles, w.med_step, w.wraps, w.bad, (unsigned long long) w.vmax,
                    (unsigned long long) w.vmax, (unsigned) (w.vmax >> 32));
        std::printf("RESULT counter read_cost cycles_per_s_getreg median=%.1f min=%.0f max=%.0f\n", rs.median, rs.min,
                    rs.max);
        if (w.wrap_seen) {
            for (size_t i = 1; i < w.n; ++i) {
                const uint32_t hp = buf[3 * (i - 1)], hn = buf[3 * i];
                if (hn == hp + 1) {
                    std::printf("DATA counter wrap_at sample=%zu lo_before=%u lo_after=%u hi_before=%u hi_after=%u\n", i,
                                buf[3 * (i - 1) + 1], buf[3 * i + 1], hp, hn);
                    break;
                }
            }
        } else {
            std::printf("RESULT counter wrap NOT_SEEN within the %.1f s run (the counter did not pass a multiple of 2^32)\n",
                        w.ticks_s);
        }
        if (!w.ok) {
            fail("HI/LO not consistent (bad=" + std::to_string(w.bad) + ")");
        }
        CHECK(hipFree(d_buf));
        CHECK(hipFree(d_info));
        CHECK(hipFree(d_rd));
    }

    // ---- 2. rate against REALTIME and host wall time, with the sysfs clock polled from a second thread
    double rate_med = 0, clk_nom_med = 0;
    {
        const std::string dev = drm_device();
        const std::string sclk_path = dev.empty() ? "" : dev + "/pp_dpm_sclk";
        if (sclk_path.empty()) {
            std::printf("RESULT counter sclk NOT_MEASURED: no DRM_DEVICE in runs/latest/card.env (run from the project root)\n");
        }
        uint64_t * d_out;
        CHECK(hipMalloc(&d_out, 64));
        // warm-up: loads the code object
        hipLaunchKernelGGL(k_rate, dim3(1), dim3(32), 0, 0, d_out, (uint64_t) 1000000, 1.0f);
        CHECK(hipDeviceSynchronize());
        std::vector<double> rate, clk_nom, clk_meas;
        for (int rep = 0; rep < 3; ++rep) {
            // 0.3 s launch first so the clock has ramped up, then the timed one
            hipLaunchKernelGGL(k_rate, dim3(1), dim3(32), 0, 0, d_out, (uint64_t) 30000000, 1.0f);
            CHECK(hipDeviceSynchronize());
            std::atomic<bool> run{true};
            std::map<int, int> seen;
            std::string raw;
            std::thread poll([&] {
                while (run.load()) {
                    if (!sclk_path.empty()) {
                        std::string r;
                        const int m = read_sclk(sclk_path, &r);
                        if (raw.empty()) {
                            raw = r;
                        }
                        ++seen[m];
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
            const auto t0 = clk::now();
            hipLaunchKernelGGL(k_rate, dim3(1), dim3(32), 0, 0, d_out, (uint64_t) 100000000, 1.0f);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const auto t1 = clk::now();
            run = false;
            poll.join();
            uint64_t o[5];
            CHECK(hipMemcpy(o, d_out, sizeof o, hipMemcpyDeviceToHost));
            const double wall = std::chrono::duration<double>(t1 - t0).count();
            const double drt = (double) (o[3] - o[0]), dc = (double) (o[2] - o[1]);
            rate.push_back(drt / wall);
            clk_nom.push_back(dc / (drt / expect_hz) / 1e6);
            clk_meas.push_back(dc / wall / 1e6);
            std::string sc;
            for (auto & kv : seen) {
                sc += (sc.empty() ? "" : ",") + std::to_string(kv.first) + "MHz:x" + std::to_string(kv.second);
            }
            std::printf("DATA counter rate_rep=%d wall_s=%.6f drealtime=%.0f dcycles=%.0f sclk_readings=%s\n", rep, wall,
                        drt, dc, sc.c_str());
            if (rep == 0) {
                std::printf("DATA counter sclk_raw %s\n", raw.c_str());
            }
        }
        const Stats r = stats_of(rate), cn = stats_of(clk_nom), cm = stats_of(clk_meas);
        rate_med = r.median;
        clk_nom_med = cn.median;
        std::printf("RESULT counter realtime_rate_per_host_s median=%.0f min=%.0f max=%.0f Hz (expected %.0f, off by %+.3f %%)\n",
                    r.median, r.min, r.max, expect_hz, (r.median / expect_hz - 1) * 100);
        std::printf("RESULT counter core_clock_MHz (cycles / REALTIME at %.0f Hz) median=%.1f min=%.1f max=%.1f\n",
                    expect_hz, cn.median, cn.min, cn.max);
        std::printf("RESULT counter core_clock_MHz (cycles / host wall time) median=%.1f min=%.1f max=%.1f\n", cm.median,
                    cm.min, cm.max);
        if (std::fabs(r.median / expect_hz - 1) > 0.02) {
            fail("REALTIME runs at " + std::to_string(r.median / 1e6) + " MHz against host time, not within 2% of " +
                 std::to_string(expect_hz / 1e6) + " MHz");
        }
        // Plausibility bound only: pp_dpm_sclk read up to about 3590 MHz on this card under a one-wave load.
        if (cn.median < 500 || cn.median > 4500) {
            fail("derived core clock " + std::to_string(cn.median) + " MHz outside 500..4500");
        }
        CHECK(hipFree(d_out));
    }

    // ---- 3. memory wait: dependent pointer chase through 512 MB
    {
        const size_t BYTES = 512ull << 20, NODE = 128, N = BYTES / NODE;
        void * d_buf;
        CHECK(hipMalloc(&d_buf, BYTES));
        const uint64_t base = (uint64_t) d_buf;
        std::vector<uint32_t> next(N);
        for (size_t i = 0; i < N; ++i) {
            next[i] = (uint32_t) i;
        }
        std::mt19937_64 rng(12345);
        for (size_t i = N - 1; i >= 1; --i) {  // Sattolo: one cycle through every node
            std::swap(next[i], next[rng() % i]);
        }
        {
            std::vector<uint64_t> hb(BYTES / 8, 0);
            for (size_t i = 0; i < N; ++i) {
                hb[i * (NODE / 8)] = base + (uint64_t) next[i] * NODE;
            }
            CHECK(hipMemcpy(d_buf, hb.data(), BYTES, hipMemcpyHostToDevice));
        }
        std::vector<uint64_t> starts(TRIALS), expect(TRIALS);
        for (int t = 0; t < TRIALS; ++t) {
            size_t node = rng() % N;
            starts[t] = base + node * NODE;
            for (int s = 0; s < STEPS; ++s) {
                node = next[node];
            }
            expect[t] = base + node * NODE;
        }
        uint64_t *d_starts, *d_out;
        CHECK(hipMalloc(&d_starts, TRIALS * 8));
        CHECK(hipMalloc(&d_out, TRIALS * 10 * 8));
        CHECK(hipMemcpy(d_starts, starts.data(), TRIALS * 8, hipMemcpyHostToDevice));
        for (int pass = 0; pass < 2; ++pass) {  // first pass loads the code object; its numbers are dropped
            hipLaunchKernelGGL(k_chase, dim3(1), dim3(32), 0, 0, d_starts, d_out, (uint64_t) 200000, 1.0f);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
        std::vector<uint64_t> o(TRIALS * 10);
        CHECK(hipMemcpy(o.data(), d_out, o.size() * 8, hipMemcpyDeviceToHost));
        std::vector<double> cpl, mhz, ns, ctrl;
        int wrong = 0;
        for (int t = 0; t < TRIALS; ++t) {
            const uint64_t * q = &o[t * 10];
            if (q[8] != expect[t]) {
                ++wrong;
            }
            const double dc = (double) (q[6] - q[5]), drt = (double) (q[7] - q[4]);
            cpl.push_back(dc / STEPS);
            ns.push_back(drt / expect_hz / STEPS * 1e9);
            mhz.push_back(dc / (drt / expect_hz) / 1e6);
            ctrl.push_back((double) (q[2] - q[1]) / ((double) (q[3] - q[0]) / expect_hz) / 1e6);
        }
        if (wrong) {
            std::printf("RESULT counter chase NOT_MEASURED: %d of %d chains ended on the wrong node (loads not done)\n",
                        wrong, TRIALS);
        } else {
            const Stats a = stats_of(cpl), b = stats_of(mhz), c = stats_of(ns), d = stats_of(ctrl);
            std::printf("RESULT counter chase loads=%d trials=%d cycles_per_load median=%.1f min=%.1f max=%.1f\n", STEPS,
                        TRIALS, a.median, a.min, a.max);
            std::printf("RESULT counter chase ns_per_load median=%.1f min=%.1f max=%.1f (REALTIME)\n", c.median, c.min,
                        c.max);
            std::printf("RESULT counter chase MHz_during_chase median=%.1f min=%.1f max=%.1f ; VALU control just before: "
                        "median=%.1f min=%.1f max=%.1f ; chase/control=%.3f\n",
                        b.median, b.min, b.max, d.median, d.min, d.max, b.median / d.median);
        }
        CHECK(hipFree(d_buf));
        CHECK(hipFree(d_starts));
        CHECK(hipFree(d_out));
    }

    // ---- 4. sleep
    {
        uint64_t * d_out;
        CHECK(hipMalloc(&d_out, SLEEP_TRIALS * 10 * 8));
        for (int pass = 0; pass < 2; ++pass) {
            hipLaunchKernelGGL(k_sleep, dim3(1), dim3(32), 0, 0, d_out, (uint64_t) 200000, 1.0f);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
        std::vector<uint64_t> o(SLEEP_TRIALS * 10);
        CHECK(hipMemcpy(o.data(), d_out, o.size() * 8, hipMemcpyDeviceToHost));
        std::vector<double> cps, mhz, us, ctrl;
        const double sleeps = 16.0 * SLEEP_BLOCKS;
        for (int t = 0; t < SLEEP_TRIALS; ++t) {
            const uint64_t * q = &o[t * 10];
            const double dc = (double) (q[6] - q[5]), drt = (double) (q[7] - q[4]);
            cps.push_back(dc / sleeps);
            us.push_back(drt / expect_hz / sleeps * 1e6);
            mhz.push_back(dc / (drt / expect_hz) / 1e6);
            ctrl.push_back((double) (q[2] - q[1]) / ((double) (q[3] - q[0]) / expect_hz) / 1e6);
        }
        const Stats a = stats_of(cps), b = stats_of(mhz), c = stats_of(us), d = stats_of(ctrl);
        std::printf("RESULT counter sleep sleeps=%.0f cycles_per_s_sleep_127 median=%.1f min=%.1f max=%.1f (guide: 64*127 = 8128)\n",
                    sleeps, a.median, a.min, a.max);
        std::printf("RESULT counter sleep us_per_s_sleep_127 median=%.3f min=%.3f max=%.3f (REALTIME)\n", c.median, c.min,
                    c.max);
        std::printf("RESULT counter sleep MHz_during_sleep median=%.1f min=%.1f max=%.1f ; VALU control just before: "
                    "median=%.1f min=%.1f max=%.1f ; sleep/control=%.3f\n",
                    b.median, b.min, b.max, d.median, d.min, d.max, b.median / d.median);
        CHECK(hipFree(d_out));
    }

    // ---- 5. one wave per work-group: are the counters of different SIMDs offset?
    {
        const int B = 128;
        uint64_t * d_out;
        CHECK(hipMalloc(&d_out, B * 4 * 8));
        for (int pass = 0; pass < 2; ++pass) {
            hipLaunchKernelGGL(k_simd, dim3(B), dim3(32), 0, 0, d_out, 1.0f);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
        std::vector<uint64_t> o(B * 4);
        CHECK(hipMemcpy(o.data(), d_out, o.size() * 8, hipMemcpyDeviceToHost));
        int ref = 0;
        for (int b = 1; b < B; ++b) {
            if (o[b * 4] < o[ref * 4]) {
                ref = b;
            }
        }
        std::vector<double> cpt;
        for (int b = 0; b < B; ++b) {
            cpt.push_back((double) (o[b * 4 + 2] - o[b * 4 + 1]) / (double) (o[b * 4 + 3] - o[b * 4]));
        }
        const double cyc_per_tick = median_of(cpt);
        std::vector<double> off, rt_rel;
        uint64_t rtmax = 0, cmin = ~0ull, cmax = 0;
        for (int b = 0; b < B; ++b) {
            const double drt = (double) o[b * 4] - (double) o[ref * 4];
            const double dcc = (double) o[b * 4 + 1] - (double) o[ref * 4 + 1];
            off.push_back(dcc - drt * cyc_per_tick);
            rt_rel.push_back(drt);
            rtmax = std::max(rtmax, o[b * 4]);
            cmin = std::min(cmin, o[b * 4 + 1]);
            cmax = std::max(cmax, o[b * 4 + 1]);
        }
        const Stats so = stats_of(off), sr = stats_of(rt_rel), sc = stats_of(cpt);
        int near = 0;
        for (double x : off) {
            near += std::fabs(x - so.median) < 2000;
        }
        std::printf("RESULT counter simd waves=%d start_spread_ticks=%.0f (%.2f us) raw_c0_spread_cycles=%.4g\n", B,
                    sr.max - sr.min, (sr.max - sr.min) / expect_hz * 1e6, (double) (cmax - cmin));
        std::printf("RESULT counter simd cycles_per_realtime_tick median=%.4f min=%.4f max=%.4f (x%.0f Hz = %.0f MHz)\n",
                    sc.median, sc.min, sc.max, expect_hz, sc.median * expect_hz / 1e6);
        std::printf("RESULT counter simd offset_vs_wave%d_after_removing_start_skew_cycles min=%.4g median=%.4g max=%.4g "
                    "within_2000_of_median=%d/%d\n",
                    ref, so.min, so.median, so.max, near, B);
        for (int b = 0; b < B; ++b) {
            std::printf("DATA counter simd block=%d rt0_rel=%.0f offset_cycles=%.0f\n", b, rt_rel[b], off[b]);
        }
        CHECK(hipFree(d_out));
    }

    (void) rate_med;
    (void) clk_nom_med;
    char text[1024];
    std::snprintf(text, sizeof text, "%s", pass ? "REALTIME rate within 2%, core clock in 500..4500 MHz, HI/LO consistent" : why.c_str());
    verdict("core_counter", pass ? "PASS" : "FAIL", text);
    return pass ? 0 : 1;
}
