// Lane 7, item 3: what a work-group barrier costs on gfx1201, whether a wave blocked in a barrier takes issue cycles
// from the other waves on its SIMD, and what contended LDS / global atomics cost.
//
// Method (same as lane 1): one wave alone on its SIMD gives v_fma_f32 independent = 1.00 cycle, measured with a
// differential over the unroll factor U = 4 and 8 so the loop control cancels. Barrier cost is the slope of the
// wave's own shader-cycle counter against the number of barriers (K2 against K1 iterations), so the counter reads and
// the first-barrier skew cancel. Placement (SE/SA/WGP/CU/SIMD/slot) is recorded by every wave in every launch.
//
// Build twice: default (WGP mode, work-group spreads over the 4 SIMDs of a WGP) and with -mcumode -DCUMODE=1 (the
// group stays on one CU, 2 SIMDs).
// Usage: sched_barrier [--break] [--quick]. --break claims the calibration is 2.00 cycles per FMA; it must FAIL.
#include "sched_common.h"

#include <cstdarg>

#ifndef CUMODE
#define CUMODE 0
#endif

namespace {

constexpr int REC = 32;     // u64 per wave record: [0] placement, [1..] values
constexpr int MAXWAVES = 32;
constexpr int MAXREPS = 12;

#define F1(i) "v_fma_f32 %" #i ", %" #i ", %16, %17\n"
#define FMA16 F1(0) F1(1) F1(2) F1(3) F1(4) F1(5) F1(6) F1(7) F1(8) F1(9) F1(10) F1(11) F1(12) F1(13) F1(14) F1(15)
#define XOPS                                                                                                      \
    "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), "+v"(x[8]), \
        "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])
// 16 independent v_fma_f32 on 16 registers; the compiler only picks registers.
#define FMA_BLOCK() asm volatile(FMA16 : XOPS : "v"(k0), "v"(k1))

__device__ __forceinline__ bool lane0() { return (threadIdx.x & 31) == 0; }

__device__ __forceinline__ void init_x(const float * in, float (&x)[16], float & k0, float & k1) {
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0];
    }
    k0 = in[1];
    k1 = in[2];
    // the compiler must not re-materialise the constants inside the timed loop
    asm volatile("" : "+v"(k0), "+v"(k1));
}
__device__ __forceinline__ float sink_of(const float (&x)[16]) {
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Calibration: one wave, the timed loop is ITERS iterations of U blocks of 16 independent FMAs.
constexpr int CAL_ITERS = 128;
constexpr int CAL_REPS = 8;

template <int U>
__global__ __launch_bounds__(32) void k_calib(const float * in, uint64_t * out, float * sink) {
    float x[16], k0, k1;
    init_x(in, x, k0, k1);
    for (int rep = 0; rep < CAL_REPS; ++rep) {
        const uint64_t t0 = sched::cyc();
#pragma unroll 1
        for (int it = 0; it < CAL_ITERS; ++it) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                FMA_BLOCK();
            }
        }
        const uint64_t t1 = sched::cyc();
        if (lane0()) {
            out[1 + rep] = t1 - t0;
        }
    }
    if (lane0()) {
        out[0] = sched::placement();
    }
    sink[threadIdx.x] = sink_of(x);
}

// ---------------------------------------------------------------------------------------------------------------
// Barrier only: per rep, K1 barriers then K2 barriers, each timed by the wave's own counter.
__global__ __launch_bounds__(1024) void k_bar(uint64_t * out, int K1, int K2, int reps) {
    const int w = threadIdx.x >> 5;
    const uint32_t pl = sched::placement();
    for (int rep = 0; rep < reps; ++rep) {
        __syncthreads();
        const uint64_t t0 = sched::cyc();
#pragma unroll 1
        for (int i = 0; i < K1; ++i) {
            __syncthreads();
        }
        const uint64_t t1 = sched::cyc();
        __syncthreads();
        const uint64_t t2 = sched::cyc();
#pragma unroll 1
        for (int i = 0; i < K2; ++i) {
            __syncthreads();
        }
        const uint64_t t3 = sched::cyc();
        if (lane0()) {
            out[w * REC + 1 + 2 * rep] = t1 - t0;
            out[w * REC + 2 + 2 * rep] = t3 - t2;
        }
    }
    if (lane0()) {
        out[w * REC] = pl;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Work then barrier: each iteration is NB blocks of 16 FMAs (W = 16 NB) and, if BAR, one barrier. imb: only wave 0
// does the work.
template <int NB, int BAR>
__global__ __launch_bounds__(1024) void k_work(const float * in, uint64_t * out, float * sink, int K1, int K2, int reps,
                                               int imb) {
    float x[16], k0, k1;
    init_x(in, x, k0, k1);
    const int w = threadIdx.x >> 5;
    const bool doit = !imb || w == 0;
    const uint32_t pl = sched::placement();
    for (int rep = 0; rep < reps; ++rep) {
        __syncthreads();
        const uint64_t t0 = sched::cyc();
#pragma unroll 1
        for (int i = 0; i < K1; ++i) {
            if (NB > 0 && doit) {
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    FMA_BLOCK();
                }
            }
            if (BAR) {
                __syncthreads();
            }
        }
        const uint64_t t1 = sched::cyc();
        __syncthreads();
        const uint64_t t2 = sched::cyc();
#pragma unroll 1
        for (int i = 0; i < K2; ++i) {
            if (NB > 0 && doit) {
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    FMA_BLOCK();
                }
            }
            if (BAR) {
                __syncthreads();
            }
        }
        const uint64_t t3 = sched::cyc();
        if (lane0()) {
            out[w * REC + 1 + 2 * rep] = t1 - t0;
            out[w * REC + 2 + 2 * rep] = t3 - t2;
        }
    }
    if (lane0()) {
        out[w * REC] = pl;
    }
    sink[threadIdx.x] = sink_of(x);
}

// ---------------------------------------------------------------------------------------------------------------
// Does a wave blocked in a barrier cost the other wave on its SIMD issue cycles?
// All waves of the group take a rank on their SIMD (LDS atomic on a per-SIMD counter); rank 0 is the worker and times
// FMA loops (differential over the iteration count; U picks the block count so U = 4 / 8 cancel the loop control).
// Partners (rank >= 1) do one of: 0 = leave at once, 1 = wait in a barrier that the worker reaches only after its
// timing, 2 = run the same FMA loop (positive control: the probe must be able to see sharing).
// The worker's result is out[w * REC + 1 + 2 * rep] = cycles of n1 iterations, +2 = cycles of n2 iterations.
template <int U>
__global__ __launch_bounds__(1024) void k_wait(const float * in, uint64_t * out, float * sink, int mode, int n1, int n2,
                                               int reps) {
    __shared__ int cnt[4];
    float x[16], k0, k1;
    init_x(in, x, k0, k1);
    const int w = threadIdx.x >> 5;
    if (threadIdx.x < 4) {
        cnt[threadIdx.x] = 0;
    }
    __syncthreads();
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
    // CU mode: the group may land on either CU of the WGP, so the SIMD is the bit within the CU
    const int simd = CUMODE ? ((hw >> 9) & 1) : ((hw >> 8) & 3);
    int rank = 0;
    if (lane0()) {
        rank = atomicAdd(&cnt[simd], 1);
    }
    rank = __builtin_amdgcn_readfirstlane(rank);
    const uint32_t pl = sched::placement();
    if (lane0()) {
        out[w * REC] = pl;
        out[w * REC + 20] = rank;
    }
    __syncthreads();
    const bool worker = rank == 0;
    if (!worker && mode == 0) {
        return;
    }
    if (!worker && mode == 1) {
        __syncthreads();  // blocks until the workers have finished their timing
        return;
    }
    uint64_t first = 0, last = 0;
    for (int rep = 0; rep < reps; ++rep) {
        const uint64_t t0 = sched::cyc();
        if (rep == 0) {
            first = t0;
        }
#pragma unroll 1
        for (int i = 0; i < n1; ++i) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                FMA_BLOCK();
            }
        }
        const uint64_t t1 = sched::cyc();
#pragma unroll 1
        for (int i = 0; i < n2; ++i) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                FMA_BLOCK();
            }
        }
        const uint64_t t2 = sched::cyc();
        last = t2;
        if (lane0()) {
            out[w * REC + 1 + 2 * rep] = t1 - t0;
            out[w * REC + 2 + 2 * rep] = t2 - t1;
        }
    }
    if (lane0()) {
        out[w * REC + 24] = first;
        out[w * REC + 25] = last;
    }
    if (worker && mode == 1) {
        __syncthreads();
    }
    sink[threadIdx.x] = sink_of(x);
}

// ---------------------------------------------------------------------------------------------------------------
// Dependent atomic adds: the next addend depends on the value returned by the previous one.
// LDS: ds_add_rtn_u32 through inline asm (so the compiler's atomic optimizer cannot collapse lanes). The address is the
// same for every lane (SAME) or one word per thread. Global: __hip_atomic_fetch_add on one word or one word per thread.
template <int GLOBAL, int SAME>
__global__ __launch_bounds__(1024) void k_atom(uint64_t * out, int * gmem, int K1, int K2, int reps) {
    __shared__ int sh[1024];
    const int w = threadIdx.x >> 5;
    sh[threadIdx.x] = 0;
    gmem[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t pl = sched::placement();
    const uint32_t laddr = SAME ? 0u : (uint32_t) threadIdx.x * 4u;
    int * gp = SAME ? gmem + 1024 : gmem + 2048 + threadIdx.x;
    auto run = [&](int K) {
        int v = 1;
#pragma unroll 1
        for (int i = 0; i < K; ++i) {
            int old;
            if (GLOBAL) {
                old = __hip_atomic_fetch_add(gp, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                asm volatile("ds_add_rtn_u32 %0, %1, %2\n\ts_wait_dscnt 0" : "=&v"(old) : "v"(laddr), "v"(v) : "memory");
            }
            v = 1 | (old >> 31);
        }
        return v;
    };
    int sink = 0;
    for (int rep = 0; rep < reps; ++rep) {
        __syncthreads();
        const uint64_t t0 = sched::cyc();
        sink += run(K1);
        const uint64_t t1 = sched::cyc();
        __syncthreads();
        const uint64_t t2 = sched::cyc();
        sink += run(K2);
        const uint64_t t3 = sched::cyc();
        if (lane0()) {
            out[w * REC + 1 + 2 * rep] = t1 - t0;
            out[w * REC + 2 + 2 * rep] = t3 - t2;
        }
    }
    if (lane0()) {
        out[w * REC] = pl;
    }
    if (sink == 0x7fffffff) {
        gmem[0] = sink;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host

uint64_t * d_out;
float * d_in;
float * d_sink;
int * d_g;

std::vector<uint64_t> fetch(int nwaves) {
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> h((size_t) nwaves * REC);
    CHECK(hipMemcpy(h.data(), d_out, h.size() * 8, hipMemcpyDeviceToHost));
    return h;
}
void clear_out() { CHECK(hipMemset(d_out, 0, (size_t) MAXWAVES * REC * 8)); }

struct Span {
    int simds, cus, wgps;
};
Span span_of(const std::vector<uint64_t> & h, int n) {
    std::vector<uint32_t> s, c, g;
    for (int w = 0; w < n; ++w) {
        const uint32_t p = (uint32_t) h[(size_t) w * REC];
        s.push_back(sched::pl_simd(p));
        c.push_back(sched::pl_cu(p));
        g.push_back(sched::pl_wgpkey(p));
    }
    auto u = [](std::vector<uint32_t> & v) {
        std::sort(v.begin(), v.end());
        return (int) (std::unique(v.begin(), v.end()) - v.begin());
    };
    return {u(s), u(c), u(g)};
}

struct Verdicts {
    int fails = 0;
    void put(const char * name, bool ok, const std::string & text) {
        if (!ok) {
            ++fails;
        }
        std::string t = std::string("sched_barrier ") + name;
        std::printf("VERDICT %s: %s %s\n", t.c_str(), ok ? "PASS" : "FAIL", text.c_str());
        std::fflush(stdout);
    }
};

std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char * f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

const char * MODE = CUMODE ? "CU" : "WGP";
std::vector<std::string> table;

// Slopes per wave and rep: (cycles(K2) - cycles(K1)) / (K2 - K1), from records laid out as in k_bar.
struct Slopes {
    std::vector<double> all;           // every wave, rep, launch
    std::vector<double> launch_med;    // median per launch
    std::vector<double> launch_wavemax;  // max over waves of the per-wave median, per launch
    std::vector<Span> spans;
    std::vector<std::vector<double>> per_wave_med;  // [launch][wave]
};
void add_launch(Slopes & s, const std::vector<uint64_t> & h, int n, int reps, int K1, int K2) {
    std::vector<double> lm, wm;
    for (int w = 0; w < n; ++w) {
        std::vector<double> v;
        for (int r = 0; r < reps; ++r) {
            const double a = (double) h[(size_t) w * REC + 1 + 2 * r], b = (double) h[(size_t) w * REC + 2 + 2 * r];
            v.push_back((b - a) / (K2 - K1));
        }
        for (double d : v) {
            lm.push_back(d);
            s.all.push_back(d);
        }
        wm.push_back(sched::pct(v, 0.5));
    }
    s.launch_med.push_back(sched::pct(lm, 0.5));
    s.launch_wavemax.push_back(*std::max_element(wm.begin(), wm.end()));
    s.per_wave_med.push_back(wm);
    s.spans.push_back(span_of(h, n));
}
// A launch is flagged disturbed when its median is more than 25 % above the median of the launch medians.
int disturbed(const std::vector<double> & lm) {
    const double m = sched::pct(lm, 0.5);
    int d = 0;
    for (double v : lm) {
        if (v > 1.25 * m) {
            ++d;
        }
    }
    return d;
}
std::string span_text(const std::vector<Span> & sp) {
    int smin = 99, smax = 0, cmin = 99, cmax = 0, wmin = 99, wmax = 0;
    for (const Span & s : sp) {
        smin = std::min(smin, s.simds);
        smax = std::max(smax, s.simds);
        cmin = std::min(cmin, s.cus);
        cmax = std::max(cmax, s.cus);
        wmin = std::min(wmin, s.wgps);
        wmax = std::max(wmax, s.wgps);
    }
    auto r = [](int a, int b) { return a == b ? fmt("%d", a) : fmt("%d-%d", a, b); };
    return fmt("%s SIMDs, %s CUs, %s WGPs", r(smin, smax).c_str(), r(cmin, cmax).c_str(), r(wmin, wmax).c_str());
}

constexpr int LAUNCHES = 6;

}  // namespace

#define RUN_WORK(NBV, BARV, N, K1, K2, REPS, IMB)                                                                  \
    hipLaunchKernelGGL((k_work<NBV, BARV>), dim3(1), dim3(32 * (N)), 0, 0, d_in, d_out, d_sink, K1, K2, REPS, IMB)

static void launch_work(int nb, int bar, int n, int K1, int K2, int reps, int imb) {
    switch (nb * 2 + bar) {
        case 0: RUN_WORK(0, 0, n, K1, K2, reps, imb); break;
        case 1: RUN_WORK(0, 1, n, K1, K2, reps, imb); break;
        case 2: RUN_WORK(1, 0, n, K1, K2, reps, imb); break;
        case 3: RUN_WORK(1, 1, n, K1, K2, reps, imb); break;
        case 8: RUN_WORK(4, 0, n, K1, K2, reps, imb); break;
        case 9: RUN_WORK(4, 1, n, K1, K2, reps, imb); break;
        case 32: RUN_WORK(16, 0, n, K1, K2, reps, imb); break;
        case 33: RUN_WORK(16, 1, n, K1, K2, reps, imb); break;
        default: std::fprintf(stderr, "bad nb\n"); std::exit(2);
    }
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    const bool brk = sched::has_flag(argc, argv, "--break");
    const bool quick = sched::has_flag(argc, argv, "--quick");
    Verdicts V;
    std::printf("# sched_barrier (%s mode) on %s, warpSize %d\n", MODE, prop.gcnArchName, prop.warpSize);

    CHECK(hipMalloc(&d_out, (size_t) MAXWAVES * REC * 8));
    CHECK(hipMalloc(&d_in, 64 * 4));
    CHECK(hipMalloc(&d_sink, 1024 * 4));
    CHECK(hipMalloc(&d_g, 4096 * 4 + 8192));
    {
        float h[64] = {1.0f, 1.0f, 0.0f};  // x = 1, k0 = 1, k1 = 0: every FMA is exact and stays 1
        CHECK(hipMemcpy(d_in, h, sizeof h, hipMemcpyHostToDevice));
    }

    // ---------------- calibration -----------------------------------------------------------------------------
    {
        std::vector<double> per;
        for (int pair = 0; pair < 7; ++pair) {
            double med[2];
            for (int k = 0; k < 2; ++k) {
                clear_out();
                if (k == 0) {
                    hipLaunchKernelGGL(k_calib<4>, dim3(1), dim3(32), 0, 0, d_in, d_out, d_sink);
                } else {
                    hipLaunchKernelGGL(k_calib<8>, dim3(1), dim3(32), 0, 0, d_in, d_out, d_sink);
                }
                const auto h = fetch(1);
                std::vector<double> v;
                for (int r = 1; r < CAL_REPS; ++r) {  // the first rep warms the instruction cache
                    v.push_back((double) h[1 + r]);
                }
                med[k] = sched::pct(v, 0.5);
            }
            per.push_back((med[1] - med[0]) / (4.0 * CAL_ITERS * 16));
        }
        const double m = sched::pct(per, 0.5);
        const double want = brk ? 2.00 : 1.00;
        std::printf("RESULT sched_barrier calib_fma cycles_per_fma %s\n", sched::dist(per, "%.3f").c_str());
        V.put("calib_fma", std::fabs(m - want) <= 0.02,
              fmt("one wave alone, 16 independent v_fma_f32 per block, U=4/8: %.3f cycles per FMA (expected %.2f +- 0.02; "
                  "range %.3f..%.3f over 7 pairs)",
                  m, want, sched::pct(per, 0), sched::pct(per, 1)));
        table.push_back(fmt("| calibration (1 wave, FMA) | %s | 1 | %.3f | %.3f..%.3f | cycles per FMA |", MODE, m,
                            sched::pct(per, 0), sched::pct(per, 1)));
    }

    // ---------------- barrier cost ----------------------------------------------------------------------------
    const int Ns[] = {1, 2, 4, 8, 16, 32};
    const int K1 = 64, K2 = 320, BREPS = 6;
    std::vector<std::string> bar_rows;
    double bar_med[6] = {};
    for (int ni = 0; ni < 6; ++ni) {
        const int n = Ns[ni];
        Slopes s;
        for (int l = 0; l < LAUNCHES; ++l) {
            clear_out();
            hipLaunchKernelGGL(k_bar, dim3(1), dim3(32 * n), 0, 0, d_out, K1, K2, BREPS);
            add_launch(s, fetch(n), n, BREPS, K1, K2);
        }
        const int dis = disturbed(s.launch_med);
        bar_med[ni] = sched::pct(s.all, 0.5);
        std::printf("RESULT sched_barrier barrier mode %s N %d cycles_per_barrier %s | %s | disturbed %d of %d\n", MODE,
                    n, sched::dist(s.all).c_str(), span_text(s.spans).c_str(), dis, LAUNCHES);
        bar_rows.push_back(fmt("| %d | %s | %.1f | %.1f | %.1f | %s | %d/%d |", n, MODE, bar_med[ni],
                               sched::pct(s.all, 0), sched::pct(s.all, 1), span_text(s.spans).c_str(), dis, LAUNCHES));
    }
    {
        bool sane = true;
        for (int i = 0; i < 6; ++i) {
            sane = sane && bar_med[i] > 0 && bar_med[i] < 5000;
        }
        V.put("barrier_cost", sane,
              fmt("%s mode cycles per barrier per wave, N=1,2,4,8,16,32: %.1f %.1f %.1f %.1f %.1f %.1f", MODE,
                  bar_med[0], bar_med[1], bar_med[2], bar_med[3], bar_med[4], bar_med[5]));
    }

    // ---------------- barrier with work -----------------------------------------------------------------------
    std::vector<std::string> work_rows;
    {
        const int wk1 = 16, wk2 = 48, wreps = 5;
        const int nbs[] = {0, 1, 4, 16};
        const int wns[] = {4, 8};
        double worst_neg = 0;
        for (int n : wns) {
            for (int nb : nbs) {
                double med[2], mx[2];
                std::vector<Span> sp;
                int dis = 0;
                for (int bar = 0; bar < 2; ++bar) {
                    Slopes s;
                    for (int l = 0; l < LAUNCHES; ++l) {
                        clear_out();
                        launch_work(nb, bar, n, wk1, wk2, wreps, 0);
                        add_launch(s, fetch(n), n, wreps, wk1, wk2);
                    }
                    med[bar] = sched::pct(s.all, 0.5);
                    mx[bar] = sched::pct(s.launch_wavemax, 0.5);
                    dis += disturbed(s.launch_med);
                    if (bar) {
                        sp = s.spans;
                    }
                }
                std::printf("RESULT sched_barrier work mode %s N %d W %d cycles_per_iter nobar %.1f bar %.1f overhead_median %.1f "
                            "overhead_slowest_wave %.1f | %s | disturbed %d of %d\n",
                            MODE, n, nb * 16, med[0], med[1], med[1] - med[0], mx[1] - mx[0], span_text(sp).c_str(), dis,
                            2 * LAUNCHES);
                work_rows.push_back(fmt("| %d | %s | %d | %.1f | %.1f | %.1f | %.1f | %s |", n, MODE, nb * 16, med[0],
                                        med[1], med[1] - med[0], mx[1] - mx[0], span_text(sp).c_str()));
                worst_neg = std::min(worst_neg, med[1] - med[0]);
            }
        }
        V.put("barrier_work", worst_neg > -20, fmt("barrier overhead per iteration, W=0,16,64,256, N=4 and 8, %s mode (table below)", MODE));
    }

    // ---------------- imbalance -------------------------------------------------------------------------------
    std::vector<std::string> imb_rows;
    {
        const int wk1 = 16, wk2 = 48, wreps = 5;
        bool all_wait = true;
        for (int n : {4, 8}) {
            Slopes s;
            for (int l = 0; l < LAUNCHES; ++l) {
                clear_out();
                launch_work(16, 1, n, wk1, wk2, wreps, 1);
                add_launch(s, fetch(n), n, wreps, wk1, wk2);
            }
            Slopes a;
            for (int l = 0; l < LAUNCHES; ++l) {
                clear_out();
                launch_work(16, 0, n, wk1, wk2, wreps, 1);
                add_launch(a, fetch(n), n, wreps, wk1, wk2);
            }
            std::vector<double> w0, oth;
            for (const auto & pw : s.per_wave_med) {
                w0.push_back(pw[0]);
                for (size_t i = 1; i < pw.size(); ++i) {
                    oth.push_back(pw[i]);
                }
            }
            const double m0 = sched::pct(w0, 0.5), mo = sched::pct(oth, 0.5);
            const double alone = sched::pct(a.all, 0.5);  // only wave 0 does work, no barrier: its own cost
            std::printf("RESULT sched_barrier imbalance mode %s N %d wave0 %.1f others_median %.1f others_min %.1f others_max "
                        "%.1f wave0_nobarrier(all waves incl. idle) %.1f\n",
                        MODE, n, m0, mo, sched::pct(oth, 0), sched::pct(oth, 1), alone);
            imb_rows.push_back(fmt("| %d | %s | %.1f | %.1f | %.1f..%.1f |", n, MODE, m0, mo, sched::pct(oth, 0),
                                   sched::pct(oth, 1)));
            all_wait = all_wait && std::fabs(mo - m0) < 0.05 * m0;
        }
        V.put("imbalance_wait", all_wait,
              fmt("%s mode: waves with no work take as many cycles per iteration as wave 0 (they wait in the barrier)", MODE));
    }

    // ---------------- does a waiting wave cost its SIMD partner? ---------------------------------------------
    std::vector<std::string> wait_rows;
    {
        const int n1 = 32, n2 = 160, wreps = 5;
        const int nw = CUMODE ? 4 : 8;
        const char * mname[3] = {"partner leaves", "partner waits in barrier", "partner runs FMA (control)"};
        double fma_c[3] = {};
        int paired_launches[3] = {};
        double ctl_max = 0, ctl_span = 0;
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<double> per;   // worker cycles per FMA, one value per worker per launch
            std::vector<double> partner, span_v;
            for (int l = 0; l < LAUNCHES; ++l) {
                double cyc_it[2][MAXWAVES];
                std::vector<uint64_t> hh[2];
                for (int u = 0; u < 2; ++u) {
                    clear_out();
                    if (u == 0) {
                        hipLaunchKernelGGL(k_wait<4>, dim3(1), dim3(32 * nw), 0, 0, d_in, d_out, d_sink, mode, n1, n2, wreps);
                    } else {
                        hipLaunchKernelGGL(k_wait<8>, dim3(1), dim3(32 * nw), 0, 0, d_in, d_out, d_sink, mode, n1, n2, wreps);
                    }
                    hh[u] = fetch(nw);
                    for (int w = 0; w < nw; ++w) {
                        std::vector<double> v;
                        for (int r = 0; r < wreps; ++r) {
                            v.push_back(((double) hh[u][(size_t) w * REC + 2 + 2 * r] - (double) hh[u][(size_t) w * REC + 1 + 2 * r]) /
                                        (n2 - n1));
                        }
                        cyc_it[u][w] = sched::pct(v, 0.5);
                    }
                }
                // rank per wave comes from the U=8 launch; placement identical in spirit but may differ between launches,
                // so ranks are taken from the same launch as the timing (U=8 minus U=4 pairs workers by wave index)
                int nworkers = 0, withpartner = 0;
                std::map<uint32_t, int> per_simd;
                for (int w = 0; w < nw; ++w) {
                    per_simd[sched::pl_simd((uint32_t) hh[1][(size_t) w * REC])]++;
                }
                for (int w = 0; w < nw; ++w) {
                    if (hh[1][(size_t) w * REC + 20] == 0) {
                        ++nworkers;
                        if (per_simd[sched::pl_simd((uint32_t) hh[1][(size_t) w * REC])] >= 2) {
                            ++withpartner;
                        }
                    }
                }
                if (nworkers > 0 && withpartner == nworkers) {
                    ++paired_launches[mode];
                }
                // worker or partner identity can differ between the U=4 and U=8 launches, so pair the sorted lists:
                // workers are the waves whose record has rank 0 in that launch
                std::vector<double> w4, w8, p4, p8;
                for (int u = 0; u < 2; ++u) {
                    for (int w = 0; w < nw; ++w) {
                        const bool isw = hh[u][(size_t) w * REC + 20] == 0;
                        const double c = cyc_it[u][w];
                        if (c <= 0) {
                            continue;
                        }
                        (u == 0 ? (isw ? w4 : p4) : (isw ? w8 : p8)).push_back(c);
                    }
                }
                if (!w4.empty() && !w8.empty()) {
                    per.push_back((sched::pct(w8, 0.5) - sched::pct(w4, 0.5)) / (4.0 * 16));
                }
                if (mode == 2) {
                    // control, U=8 launch: slowest single loop of any wave (cycles per FMA, loop control included) and the
                    // SIMD-level span of both waves from first start to last end over all their FMAs
                    std::map<uint32_t, std::pair<uint64_t, uint64_t>> sp;
                    std::map<uint32_t, int> nact;
                    for (int w = 0; w < nw; ++w) {
                        if (hh[1][(size_t) w * REC + 25] == 0) {
                            continue;
                        }
                        const uint32_t key = sched::pl_simd((uint32_t) hh[1][(size_t) w * REC]);
                        const uint64_t a = hh[1][(size_t) w * REC + 24], b = hh[1][(size_t) w * REC + 25];
                        auto it = sp.find(key);
                        if (it == sp.end()) {
                            sp[key] = {a, b};
                        } else {
                            it->second.first = std::min(it->second.first, a);
                            it->second.second = std::max(it->second.second, b);
                        }
                        ++nact[key];
                        for (int r = 0; r < wreps; ++r) {
                            ctl_max = std::max(ctl_max, (double) hh[1][(size_t) w * REC + 1 + 2 * r] / (n1 * 8.0 * 16));
                            ctl_max = std::max(ctl_max, (double) hh[1][(size_t) w * REC + 2 + 2 * r] / (n2 * 8.0 * 16));
                        }
                    }
                    for (auto & kv : sp) {
                        if (nact[kv.first] == 2) {
                            span_v.push_back((double) (kv.second.second - kv.second.first) /
                                             (2.0 * wreps * (n1 + n2) * 8 * 16));
                        }
                    }
                }
                if (mode == 2 && !p4.empty() && !p8.empty()) {
                    partner.push_back((sched::pct(p8, 0.5) - sched::pct(p4, 0.5)) / (4.0 * 16));
                }
            }
            fma_c[mode] = sched::pct(per, 0.5);
            std::printf("RESULT sched_barrier waiting mode %s %s worker_cycles_per_fma %s paired_launches %d of %d%s\n", MODE,
                        mname[mode], sched::dist(per, "%.3f").c_str(), paired_launches[mode], LAUNCHES,
                        mode == 2 ? fmt(" partner_cycles_per_fma %s", sched::dist(partner, "%.3f").c_str()).c_str() : "");
            wait_rows.push_back(fmt("| %s | %s | %s | %.3f | %.3f..%.3f | %d/%d |", mname[mode], MODE,
                                    mode == 2 ? fmt("%.3f (partner)", sched::pct(partner, 0.5)).c_str() : "-", fma_c[mode],
                                    sched::pct(per, 0), sched::pct(per, 1), paired_launches[mode], LAUNCHES));
            if (mode == 2) {
                ctl_span = sched::pct(span_v, 0.5);
                std::printf("RESULT sched_barrier waiting_control mode %s slowest_single_loop_cycles_per_fma %.3f simd_span_cycles_per_fma_of_both_waves %s\n",
                            MODE, ctl_max, sched::dist(span_v, "%.3f").c_str());
                wait_rows.push_back(fmt("| control: slowest single loop / SIMD span per FMA of both waves | %s | %.3f | %.3f | - | - |", MODE,
                                        ctl_max, ctl_span));
            }
        }
        V.put("waiting_costs_nothing", std::fabs(fma_c[1] - 1.0) <= 0.02 && std::fabs(fma_c[0] - 1.0) <= 0.02,
              fmt("%s mode: FMA wave with a partner blocked in s_barrier_wait on its SIMD: %.3f cycles per FMA (partner gone: %.3f; "
                  "expected 1.00 +- 0.02); paired launches %d/%d",
                  MODE, fma_c[1], fma_c[0], paired_launches[1], LAUNCHES));
        V.put("waiting_control_sees_sharing", ctl_max > 1.3 && ctl_span > 0.9 && ctl_span < 1.1,
              fmt("%s mode: with the partner also running FMAs the slowest single loop shows %.3f cycles per FMA (must exceed 1.3 "
                  "or the probe cannot see sharing) and the SIMD span over both waves is %.3f cycles per FMA (1.0 = one FMA "
                  "per cycle per SIMD, no doubling)",
                  MODE, ctl_max, ctl_span));
    }

    // ---------------- atomics ---------------------------------------------------------------------------------
    std::vector<std::string> at_rows;
    if (!quick) {
        const int ak1 = 16, ak2 = 48, areps = 5;
        const char * nm[4] = {"LDS distinct", "LDS same", "global distinct", "global same"};
        for (int kind = 0; kind < 4; ++kind) {
            for (int n : Ns) {
                Slopes s;
                for (int l = 0; l < LAUNCHES; ++l) {
                    clear_out();
                    CHECK(hipMemset(d_g, 0, 4096 * 4 + 8192));
                    switch (kind) {
                        case 0: hipLaunchKernelGGL((k_atom<0, 0>), dim3(1), dim3(32 * n), 0, 0, d_out, d_g, ak1, ak2, areps); break;
                        case 1: hipLaunchKernelGGL((k_atom<0, 1>), dim3(1), dim3(32 * n), 0, 0, d_out, d_g, ak1, ak2, areps); break;
                        case 2: hipLaunchKernelGGL((k_atom<1, 0>), dim3(1), dim3(32 * n), 0, 0, d_out, d_g, ak1, ak2, areps); break;
                        default: hipLaunchKernelGGL((k_atom<1, 1>), dim3(1), dim3(32 * n), 0, 0, d_out, d_g, ak1, ak2, areps); break;
                    }
                    add_launch(s, fetch(n), n, areps, ak1, ak2);
                }
                const int dis = disturbed(s.launch_med);
                std::printf("RESULT sched_barrier atomic mode %s %s N %d cycles_per_atomic %s | disturbed %d of %d\n", MODE, nm[kind],
                            n, sched::dist(s.all, "%.1f").c_str(), dis, LAUNCHES);
                at_rows.push_back(fmt("| %s | %s | %d | %.1f | %.1f | %.1f | %d/%d |", nm[kind], MODE, n,
                                      sched::pct(s.all, 0.5), sched::pct(s.all, 0), sched::pct(s.all, 1), dis, LAUNCHES));
            }
        }
        V.put("atomics_ran", true, fmt("%s mode: dependent atomic add sweeps done (table below)", MODE));
    }

    // ---------------- tables ----------------------------------------------------------------------------------
    std::printf("\n## Barrier cost (cycles per barrier per wave, median / min / max over waves, reps, launches)\n");
    std::printf("| N waves | mode | median | min | max | placement span | disturbed |\n|---|---|---|---|---|---|---|\n");
    for (auto & r : bar_rows) {
        std::printf("%s\n", r.c_str());
    }
    std::printf("\n## Barrier with work (cycles per iteration)\n| N | mode | W FMAs | no barrier | barrier | overhead (median) | overhead (slowest wave) | span |\n|---|---|---|---|---|---|---|---|\n");
    for (auto & r : work_rows) {
        std::printf("%s\n", r.c_str());
    }
    std::printf("\n## Imbalance (wave 0 does 256 FMAs per iteration, others none; cycles per iteration)\n| N | mode | wave 0 | others median | others range |\n|---|---|---|---|---|\n");
    for (auto & r : imb_rows) {
        std::printf("%s\n", r.c_str());
    }
    std::printf("\n## Waiting wave (cycles per FMA of the worker wave)\n| partner | mode | partner cycles/FMA | worker median | range | launches with a partner on the SIMD |\n|---|---|---|---|---|---|\n");
    for (auto & r : wait_rows) {
        std::printf("%s\n", r.c_str());
    }
    if (!at_rows.empty()) {
        std::printf("\n## Dependent atomic add (cycles per atomic per wave)\n| kind | mode | N waves | median | min | max | disturbed |\n|---|---|---|---|---|---|---|\n");
        for (auto & r : at_rows) {
            std::printf("%s\n", r.c_str());
        }
    }
    std::printf("\n## Calibration\n| test | mode | waves | median | range | unit |\n|---|---|---|---|---|---|\n%s\n", table[0].c_str());
    return V.fails ? 1 : 0;
}
