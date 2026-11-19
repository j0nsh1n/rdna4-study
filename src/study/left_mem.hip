// Leftovers C, lane A: memory items with one wave (measurements/leftovers-c-a.md).
//
// Usage: left_mem ns [--break]
//   ns  MS-3 and MS-4. One wave, one launch per trial after a 512 MiB flush read. Inside the launch, in this order,
//       segments that each record the shader cycles AND the REALTIME counter (100 MHz) at their start and end, so
//       every segment's latency is known in cycles and in nanoseconds at the clock of that moment:
//         chase    a dependent pointer chase over fresh lines (random inside 64 KiB blocks of a 256 MiB buffer; all
//                  32 lanes read the same address; memory-system 2b "random in 64 KiB blocks")
//         old N    batches of N loads of 4 bytes per lane in the layout of mem_inflight (load k of batch b reads the
//                  128-byte line b*N + k: consecutive loads on consecutive 128-byte lines, so two loads share one
//                  256-byte L2 line), N = 1, 2, 4, 8
//         d256 N   the same with every load on its own 256-byte line (load stride 256 bytes)
//         d4k N    every load 4 KiB from the next
//         w16 old  16 bytes per lane (512 bytes per load), consecutive (mem_inflight's 772-cycle row); w16 d4k: 4 KiB
//                  apart
//         chase    again at the end
//       Every region is fresh (never read since the flush). The cycles of every batch are kept (in LDS), so a mix of
//       L2 hits and memory reads shows as two groups. Verdicts: mem_ns_calib (L0 chase, 16 KiB random, 81 cycles
//       +-10 %, tim_l2; and the old layout's batch-cost ratios N=2/N=1 = 1.54 and N=4/N=1 = 1.42 of mem_inflight
//       within 8 %, ratios because the cycle counts of different jobs depend on the clock); mem_distinct_lines (lane
//       2's rule, memory-system item 5: a batch costs about one latency up to N = 63, so with distinct lines
//       N = 2 <= 1.10 x N = 1 and N = 4 <= 1.10 x N = 2). The clock explanation of MS-3 is INFO.
//   tlb    MS-5: a dependent chase (tim_l2's method: one full cycle of warm-up in the same launch, then 8,192 timed
//          steps, final pointer checked) over N = 64 to 4,096 nodes at spacings of 4 KiB to 2 MiB (spans up to 4 GiB):
//          where the latency first rises 1.3 x above the L2 plateau (the value at 512 nodes), up to 2,048 nodes, in
//          nodes and in bytes. Calibrations: L0 16 KiB random 81 cycles +-10 %; 2,048 nodes at 4 KiB within 5 % of the
//          one-wave L2 plateau of memory-system 2b (231-256); at 64 KiB 463 (tim_l2) +-10 %.
//   multi  MS-2a: W = 1, 2, 4, 16 one-wave groups at once (CU mode, 40 KiB of LDS each, so one per CU), each chasing
//          its own region at 256-byte spacing, N = 2,048 to 30,226 nodes: the step of each W (first N at 1.3 x the
//          lowest median). The one-wave step is at 4,000 to 11,000 nodes; W = 1 must step in 2,867 to 15,421.
//   l0card W  MS-9: whole card, 64 groups (one per CU: 40 KiB of LDS each, CU mode) of W waves (1, 2, 4, 8 or 16);
//          each wave re-reads its own 2 KiB slice (16 KiB per CU at most W = 8, inside the 32 KiB L0) in batches of 8
//          global_load_b128 (4 KiB) and one wait, for 0.1 s of REALTIME: one burst per job (the script runs three jobs
//          per W and waits outside the GPU lock until the hotspot is below 55 C; a burst that would start at 55 C or
//          above is skipped, never waited for while holding the lock). W = 0 is the calibration job. Bytes per cycle per CU = bytes / REALTIME span / 64 / clock
//          (clock from the waves' own counters). Verdicts: mem_l0card_calib (W = 0, a short job: one wave per CU,
//          batches of 8 and of 16 loads on different 512-byte chunks; each extra 16-byte load costs 4.0 cycles +-10 %,
//          memory-system item 5's L0 rows (188.1 - 156.1) / 8: the one-wave 128 bytes per cycle);
//          mem_l0card_ceiling (the burst ran and is not above 1.05 x 128 bytes per cycle per CU); the rate is INFO.
//          --break claims the ceiling at 64 bytes per cycle: FAIL.
//   --break the L0 calibration chase is replaced by independent loads, so the calibration verdict must FAIL.
#include "mem_common.h"

#include <chrono>
#include <cstdarg>
#include <map>
#include <set>
#include <thread>

namespace {

#define CH1 "global_load_b64 %0, %0, off\n\ts_wait_loadcnt 0\n\t"
#define CH16 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1 CH1

__device__ __forceinline__ uint64_t chase(uint64_t p, uint32_t steps) {
#pragma unroll 1
    for (uint32_t i = 0; i < steps / 16; ++i) {
        asm volatile(CH16 : "+v"(p)::"memory");
    }
    return p;
}

// L0 calibration as tim_l2: one full warm cycle, then `timed` steps. out: [0] cycles, [1] final pointer.
__global__ __launch_bounds__(32) void k_cal(uint64_t start, uint32_t warm, uint32_t timed, uint64_t * out) {
    uint64_t p = chase(start, warm);
    const uint64_t c0 = wave_cycles();
    p = chase(p, timed);
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = p;
    }
}

// the break: independent loads from the start node, one wait per 16
__global__ __launch_bounds__(32) void k_cal_broken(uint64_t start, uint32_t timed, uint64_t * out) {
    uint64_t p = start, d = 0;
    const uint64_t c0 = wave_cycles();
#define BR1 "global_load_b64 %1, %0, off\n\t"
#pragma unroll 1
    for (uint32_t i = 0; i < timed / 16; ++i) {
        asm volatile(BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 BR1 "s_wait_loadcnt 0"
                     : "+v"(p), "=&v"(d)::"memory");
    }
#undef BR1
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[0] = c1 - c0;
        out[1] = p ^ (d & 0);
    }
}

// One segment of the ns kernel.
struct Seg {
    uint32_t kind;        // 0 chase, 1 batches of 4-byte loads, 2 batches of 16-byte loads, 3 chase stamped per step
    uint32_t n;           // loads per batch (kind 1, 2) or chase steps (kind 0, multiple of 16)
    uint32_t count;       // batches
    uint32_t load_step;   // bytes between the loads of one batch
    uint64_t base;        // device address of the segment's first line, or the chase start
};
constexpr int MAXSEG = 24;
constexpr int MAXB = 128;

typedef unsigned u4 __attribute__((ext_vector_type(4)));

template <int N>
__device__ __forceinline__ uint32_t batch4(uint64_t base, uint32_t step, uint32_t lane) {
    uint64_t a[N];
    unsigned r[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        a[k] = base + (uint64_t) k * step + lane * 4;
        asm volatile("" : "+v"(a[k]));  // the addresses are ready before the first counter read
    }
    const uint64_t t0 = wave_cycles();
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("global_load_b32 %0, %1, off" : "=v"(r[k]) : "v"(a[k]) : "memory");
    }
    asm volatile("s_wait_loadcnt 0" ::: "memory");
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" : "+v"(r[k]));
    }
    const uint64_t t1 = wave_cycles();
#pragma unroll
    for (int k = 0; k < N; ++k) {
        asm volatile("" ::"v"(r[k]));
    }
    return (uint32_t) (t1 - t0);
}

__device__ __forceinline__ uint32_t batch16(uint64_t base, uint32_t lane) {
    uint64_t a = base + lane * 16;
    asm volatile("" : "+v"(a));
    u4 r;
    const uint64_t t0 = wave_cycles();
    asm volatile("global_load_b128 %0, %1, off\n\ts_wait_loadcnt 0" : "=v"(r) : "v"(a) : "memory");
    const uint64_t t1 = wave_cycles();
    asm volatile("" ::"v"(r));
    return (uint32_t) (t1 - t0);
}

// out per segment s: [s*4 + 0] cycles, [1] REALTIME ticks, [2] final pointer (chase); then per-batch cycles at
// out[MAXSEG*4 + s*MAXB + b].
__global__ __launch_bounds__(32) void k_ns(const Seg * segs, uint32_t nseg, uint64_t * out) {
    __shared__ uint32_t bc[MAXSEG * MAXB];
    const uint32_t lane = threadIdx.x % 32;
    for (uint32_t s = 0; s < nseg; ++s) {
        const Seg g = segs[s];
        uint64_t p = g.base;
        const uint64_t r0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
        if (g.kind == 0) {
            p = chase(p, g.n);
        } else if (g.kind == 3) {
            for (uint32_t b = 0; b < g.count; ++b) {
                const uint64_t t0 = wave_cycles();
                asm volatile(CH1 : "+v"(p)::"memory");
                const uint64_t t1 = wave_cycles();
                if (lane == 0) {
                    bc[s * MAXB + b] = (uint32_t) (t1 - t0);
                }
            }
        } else {
            for (uint32_t b = 0; b < g.count; ++b) {
                const uint64_t base = g.base + (uint64_t) b * g.n * g.load_step;
                uint32_t c = 0;
                if (g.kind == 2) {
                    c = batch16(base, lane);
                } else {
                    switch (g.n) {
                        case 1: c = batch4<1>(base, g.load_step, lane); break;
                        case 2: c = batch4<2>(base, g.load_step, lane); break;
                        case 4: c = batch4<4>(base, g.load_step, lane); break;
                        default: c = batch4<8>(base, g.load_step, lane); break;
                    }
                }
                if (lane == 0) {
                    bc[s * MAXB + b] = c;
                }
            }
        }
        const uint64_t c1 = wave_cycles();
        const uint64_t r1 = mem::realtime();
        if (lane == 0) {
            out[s * 4 + 0] = c1 - c0;
            out[s * 4 + 1] = r1 - r0;
            out[s * 4 + 2] = p;
        }
    }
    __syncthreads();
    for (uint32_t i = lane; i < nseg * MAXB; i += 32) {
        out[MAXSEG * 4 + i] = bc[i];
    }
}

__global__ void k_flush(const u4 * p, uint32_t n, unsigned * sink) {
    unsigned acc = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        acc += p[i].x;
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? NAN : v[v.size() / 2];
}
double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v.empty() ? NAN : v[std::min(v.size() - 1, (size_t) (q * v.size()))];
}

int failures = 0;
void check(const char * name, bool ok, const std::string & text) {
    verdict(name, ok ? "PASS" : "FAIL", text.c_str());
    failures += !ok;
}
std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char * f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

int mode_ns(bool brk) {
    // L0 calibration: 128 nodes of 128 bytes, random (tim_l2's calibration point)
    char * cal;
    uint64_t * out;
    CHECK(hipMalloc(&cal, 1 << 20));
    CHECK(hipMalloc(&out, (MAXSEG * 4 + MAXSEG * MAXB) * 8));
    std::vector<uint32_t> order;
    const uint64_t cstart = mem::build_chase(cal, 128, 128, true, 7, &order);
    std::vector<double> calv;
    for (int t = 0; t < 5; ++t) {
        if (brk) {
            hipLaunchKernelGGL(k_cal_broken, dim3(1), dim3(32), 0, 0, cstart, 8192u, out);
        } else {
            hipLaunchKernelGGL(k_cal, dim3(1), dim3(32), 0, 0, cstart, 128u, 8192u, out);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
        calv.push_back((double) o[0] / 8192);
    }
    const double cal_cyc = median(calv);
    std::printf("RESULT ns calibration L0 chase %.1f cycles per load (5 launches %.1f-%.1f)%s\n", cal_cyc,
                *std::min_element(calv.begin(), calv.end()), *std::max_element(calv.begin(), calv.end()),
                brk ? " (deliberate break: independent loads)" : "");

    // buffers: 256 MiB chase (random inside 64 KiB blocks), 512 MiB stream, 512 MiB flush
    constexpr size_t CSPAN = 256ull << 20, SSPAN = 512ull << 20, FSPAN = 512ull << 20;
    char *cbuf, *cbuf2, *sbuf, *fbuf;
    unsigned * sink;
    CHECK(hipMalloc(&cbuf, CSPAN));
    CHECK(hipMalloc(&cbuf2, CSPAN));
    CHECK(hipMalloc(&sbuf, SSPAN));
    CHECK(hipMalloc(&fbuf, FSPAN));
    CHECK(hipMalloc(&sink, 512 * 256 * 4));
    mem::fill(sbuf, SSPAN, mem::RAMP);
    mem::fill(fbuf, FSPAN, mem::RANDOM);
    const size_t cnodes = CSPAN / 128;
    std::vector<uint32_t> corder;
    mem::build_chase(cbuf, cnodes, 128, true, 11, &corder, 512);
    // the same blocks of 64 KiB with one node per 256 bytes: every node on its own L2 line
    const size_t cnodes2 = CSPAN / 256;
    std::vector<uint32_t> corder2;
    mem::build_chase(cbuf2, cnodes2, 256, true, 13, &corder2, 256);
    Seg * dsegs;
    CHECK(hipMalloc(&dsegs, MAXSEG * sizeof(Seg)));

    struct Def {
        std::string name;
        uint32_t kind, n, load_step;  // load_step of a chase: its node spacing
    };
    std::vector<Def> defs = {{"chase_first", 0, 1024, 128}, {"chase128_st", 3, 1, 128}, {"chase256_st", 3, 1, 256},
                             {"chase256", 0, 1024, 256}};
    for (uint32_t n : {1u, 2u, 4u, 8u}) defs.push_back({"old_N" + std::to_string(n), 1, n, 128});
    for (uint32_t n : {1u, 2u, 4u, 8u}) defs.push_back({"d256_N" + std::to_string(n), 1, n, 256});
    for (uint32_t n : {1u, 2u, 4u, 8u}) defs.push_back({"d4k_N" + std::to_string(n), 1, n, 4096});
    defs.push_back({"w16_old_N1", 2, 1, 512});
    defs.push_back({"w16_d4k_N1", 2, 1, 4096});
    defs.push_back({"chase_last", 0, 1024, 128});
    const uint32_t nseg = (uint32_t) defs.size();

    constexpr int TRIALS = 7;
    std::map<std::string, std::vector<double>> cyc, ns, clk, batch_all;
    size_t soff = 0;
    size_t cblock = 7, cblock2 = 7;  // chase start: the first node of a 64 KiB block not used before
    bool ptr_ok = true;
    for (int t = 0; t < TRIALS; ++t) {
        std::vector<Seg> segs;
        std::vector<uint64_t> expect;
        for (const Def & d : defs) {
            Seg g{d.kind, d.n, MAXB, d.load_step, 0};
            if (d.kind == 0 || d.kind == 3) {
                const uint32_t steps = d.kind == 0 ? d.n : MAXB;
                const bool w = d.load_step == 256;
                const size_t per_block = w ? 256 : 512, nn = w ? cnodes2 : cnodes;
                size_t & cb = w ? cblock2 : cblock;
                const std::vector<uint32_t> & od = w ? corder2 : corder;
                char * cbase = w ? cbuf2 : cbuf;
                const size_t k0 = cb * per_block;
                cb += (steps + per_block - 1) / per_block + 2;  // the blocks this chase covers, and a gap
                g.base = (uint64_t) cbase + (uint64_t) od[k0 % nn] * d.load_step;
                expect.push_back((uint64_t) cbase + (uint64_t) od[(k0 + steps) % nn] * d.load_step);
            } else {
                g.base = (uint64_t) sbuf + soff;
                soff += (size_t) MAXB * d.n * d.load_step + 65536;
                expect.push_back(0);
            }
            segs.push_back(g);
        }
        if (soff > SSPAN || cblock * 512 > cnodes || cblock2 * 256 > cnodes2) {
            std::fprintf(stderr, "buffers too small\n");
            return 2;
        }
        CHECK(hipMemcpy(dsegs, segs.data(), segs.size() * sizeof(Seg), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_flush, dim3(512), dim3(256), 0, 0, (const u4 *) fbuf, (uint32_t) (FSPAN / 16), sink);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        hipLaunchKernelGGL(k_ns, dim3(1), dim3(32), 0, 0, dsegs, nseg, out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> o(MAXSEG * 4 + MAXSEG * MAXB);
        CHECK(hipMemcpy(o.data(), out, o.size() * 8, hipMemcpyDeviceToHost));
        for (uint32_t s = 0; s < nseg; ++s) {
            const Def & d = defs[s];
            const double per = d.kind == 0 ? (double) d.n : (double) MAXB;  // segment mean, loop overhead included
            const double c = (double) o[s * 4] / per, n = (double) o[s * 4 + 1] * 10.0 / per;
            cyc[d.name].push_back(c);
            ns[d.name].push_back(n);
            clk[d.name].push_back((double) o[s * 4] / ((double) o[s * 4 + 1] * 10.0));
            if (d.kind == 0 || d.kind == 3) {
                ptr_ok &= o[s * 4 + 2] == expect[s];
            }
            if (d.kind != 0) {
                for (int b = 0; b < MAXB; ++b) {
                    batch_all[d.name].push_back((double) o[MAXSEG * 4 + s * MAXB + b]);
                }
            }
        }
    }
    std::printf("RESULT ns table: per load (chase) or per batch; medians of %d launches (one launch per trial, each "
                "after a flush); clock = segment cycles / segment ns\n", TRIALS);
    std::map<std::string, double> mc, mn, mb, bmean, bns;
    for (const Def & d : defs) {
        mc[d.name] = median(cyc[d.name]);
        mn[d.name] = median(ns[d.name]);
        const std::vector<double> & b = batch_all[d.name];
        mb[d.name] = median(b);
        bmean[d.name] = b.empty() ? NAN : std::accumulate(b.begin(), b.end(), 0.0) / b.size();
        bns[d.name] = mb[d.name] / median(clk[d.name]);
        std::printf("RESULT ns %-11s segment_mean_cycles %7.1f (%.1f-%.1f) ns %6.1f (%.1f-%.1f) clock_GHz %.3f", d.name.c_str(),
                    mc[d.name], *std::min_element(cyc[d.name].begin(), cyc[d.name].end()),
                    *std::max_element(cyc[d.name].begin(), cyc[d.name].end()), mn[d.name],
                    *std::min_element(ns[d.name].begin(), ns[d.name].end()),
                    *std::max_element(ns[d.name].begin(), ns[d.name].end()), median(clk[d.name]));
        if (!b.empty()) {
            // share of batches below 450 cycles: L2 hits (L2 latency 230-340 cycles, memory 485 and more)
            size_t low = 0;
            for (double x : b) low += x < 450;
            std::printf(" per_batch mean %.1f p10 %.0f p50 %.0f p90 %.0f below_450 %.2f p50_ns %.1f", bmean[d.name],
                        pct(b, 0.1), pct(b, 0.5), pct(b, 0.9), (double) low / b.size(), bns[d.name]);
        }
        std::printf("\n");
    }
    // ratios of per-batch means (the statistic of mem_inflight: its cycles per batch are a mean over 65,536 batches)
    const double r2 = bmean["old_N2"] / bmean["old_N1"], r4 = bmean["old_N4"] / bmean["old_N1"];
    const bool cal_ok = std::fabs(cal_cyc - 81.0) <= 8.1 && std::fabs(r2 / 1.540 - 1) <= 0.08 &&
                        std::fabs(r4 / 1.415 - 1) <= 0.08 && ptr_ok;
    check("mem_ns_calib", cal_ok,
          fmt("L0 chase %.1f cycles (known 81 +-10 %%%s); old layout per-batch mean ratios N2/N1 %.3f (mem_inflight 845.3/549.1 = "
              "1.540) and N4/N1 %.3f (776.7/549.1 = 1.415), each within 8 %%; chase final pointers %s",
              cal_cyc, brk ? ", deliberate break: independent loads" : "", r2, r4, ptr_ok ? "all right" : "WRONG"));
    if (brk) {
        return failures ? 1 : 0;
    }
    // per-batch medians (STUDY-PLAN: medians, not means, of short samples)
    const double a1 = mb["d256_N1"], a2 = mb["d256_N2"], a4 = mb["d256_N4"];
    const double b1 = mb["d4k_N1"], b2 = mb["d4k_N2"], b4 = mb["d4k_N4"];
    check("mem_distinct_lines", a2 <= 1.10 * a1 && a4 <= 1.10 * a2 && b2 <= 1.10 * b1 && b4 <= 1.10 * b2,
          fmt("every load on its own 256-byte line: N=1 %.0f, N=2 %.0f, N=4 %.0f, N=8 %.0f cycles per batch (median); 4 KiB "
              "apart: %.0f, %.0f, %.0f, %.0f; old layout (two loads per L2 line): %.0f, %.0f, %.0f, %.0f (lane 2's rule: "
              "N=2 <= 1.10 x N=1 and N=4 <= 1.10 x N=2 for distinct lines)",
              a1, a2, a4, mb["d256_N8"], b1, b2, b4, mb["d4k_N8"], mb["old_N1"], mb["old_N2"], mb["old_N4"], mb["old_N8"]));
    std::printf("VERDICT mem_clock_explanation: INFO in one launch (clock %.3f GHz): chase with 128-byte nodes %.0f / %.0f "
                "cycles per step (first / last; mean) = %.1f / %.1f ns, stamped per step: median %.0f, %.0f %% below 450; "
                "chase with 256-byte nodes %.0f cycles = %.1f ns, stamped median %.0f, %.0f %% below 450; one 4-byte load "
                "on its own line (median) %.0f cycles = %.1f ns (256 B apart), %.0f = %.1f ns (4 KiB apart); 16-byte "
                "loads %.0f = %.1f ns (consecutive), %.0f = %.1f ns (4 KiB apart); old layout N=1 median %.0f, mean %.0f, "
                "%.0f %% of its batches below 450 cycles\n",
                median(clk["chase_first"]), mc["chase_first"], mc["chase_last"], mn["chase_first"], mn["chase_last"],
                mb["chase128_st"],
                100.0 * [&] {
                    size_t low = 0;
                    for (double x : batch_all["chase128_st"]) low += x < 450;
                    return (double) low / batch_all["chase128_st"].size();
                }(),
                mc["chase256"], mn["chase256"], mb["chase256_st"],
                100.0 * [&] {
                    size_t low = 0;
                    for (double x : batch_all["chase256_st"]) low += x < 450;
                    return (double) low / batch_all["chase256_st"].size();
                }(),
                a1, bns["d256_N1"], b1, bns["d4k_N1"], mb["w16_old_N1"], bns["w16_old_N1"], mb["w16_d4k_N1"],
                bns["w16_d4k_N1"], mb["old_N1"], bmean["old_N1"],
                100.0 * [&] {
                    size_t low = 0;
                    for (double x : batch_all["old_N1"]) low += x < 450;
                    return (double) low / batch_all["old_N1"].size();
                }());
    CHECK(hipFree(cbuf));
    CHECK(hipFree(cbuf2));
    CHECK(hipFree(sbuf));
    CHECK(hipFree(fbuf));
    return failures ? 1 : 0;
}

// ---- MS-5 and MS-2a -----------------------------------------------------------------------------------------------
struct Pt {
    double cyc;
    bool ok;
};

// One chain of `nodes` nodes `stride` bytes apart at dev (random order); median of 5 launches of the tim_l2 method.
Pt chase_point(char * dev, size_t nodes, size_t stride, bool brk, uint64_t * out, uint32_t seed = 7) {
    std::vector<uint32_t> order;
    const uint64_t start = mem::build_chase(dev, nodes, stride, true, seed, &order);
    const uint32_t warm = (uint32_t) ((nodes + 15) / 16 * 16), timed = 8192;
    const uint64_t expect = (uint64_t) dev + (uint64_t) order[(warm + timed) % nodes] * stride;
    std::vector<double> cy;
    bool ok = true;
    for (int t = 0; t < 5; ++t) {
        if (brk) {
            hipLaunchKernelGGL(k_cal_broken, dim3(1), dim3(32), 0, 0, start, timed, out);
        } else {
            hipLaunchKernelGGL(k_cal, dim3(1), dim3(32), 0, 0, start, warm, timed, out);
        }
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o[2];
        CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
        cy.push_back((double) o[0] / timed);
        ok &= brk || o[1] == expect;
    }
    return {median(cy), ok};
}

int mode_tlb(bool brk) {
    uint64_t * out;
    CHECK(hipMalloc(&out, 64));
    constexpr size_t SPAN = 4ull << 30;
    char * buf;
    CHECK(hipMalloc(&buf, SPAN + (1 << 20)));
    const Pt cal = chase_point(buf, 128, 128, brk, out);
    if (brk) {
        check("mem_tlb_calib", false, fmt("L0 calibration %.1f cycles (deliberate break: independent loads)", cal.cyc));
        return 1;
    }
    bool ptr_ok = cal.ok;
    const size_t strides[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 20, 2 << 20};
    // up to 256 nodes the lines fit the 32 KiB L0 (81 cycles); the translation step is judged on the L2 plateau from
    // 512 nodes on, and only up to 2,048 nodes: from about 4,000 lines one wave also leaves L2 (memory-system 2b-2)
    std::vector<size_t> ns = {64, 128, 256, 512, 768, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 3072, 4096};
    std::map<size_t, std::map<size_t, double>> got;
    for (size_t st : strides) {
        for (size_t n : ns) {
            if (n * st > SPAN) continue;
            // the fine steps only where the first run saw the step (64 KiB and more); 3,072 only below
            const bool fine = n > 1024 && n < 2048;
            if (fine && st < 65536) continue;
            if (n == 3072 && st >= 65536) continue;
            const Pt p = chase_point(buf, n, st, false, out);
            ptr_ok &= p.ok;
            got[st][n] = p.cyc;
            std::printf("RESULT tlb stride_KiB %5zu nodes %5zu span_MiB %7.1f pages64K %6.0f cycles %.1f\n", st >> 10, n,
                        (double) n * st / 1048576.0, (double) n * std::max<size_t>(st, 65536) / 65536.0 / std::max<size_t>(1, 65536 / st), p.cyc);
            std::fflush(stdout);
        }
        const double base = got[st][512];
        size_t step = 0;
        for (auto & kv : got[st]) {
            if (kv.first > 512 && kv.first <= 2048 && kv.second >= 1.3 * base) {
                step = kv.first;
                break;
            }
        }
        if (step) {
            std::printf("STEP tlb stride_KiB %zu plateau %.0f cycles at 512 nodes: first point at 1.3 x up to 2,048 nodes: "
                        "%zu nodes = %.1f MiB span\n",
                        st >> 10, base, step, (double) step * st / 1048576.0);
        } else {
            std::printf("STEP tlb stride_KiB %zu plateau %.0f cycles at 512 nodes: no point at 1.3 x up to 2,048 nodes\n",
                        st >> 10, base);
        }
    }
    const double a = got[4096][2048], b = got[65536][2048];
    // 2,048 nodes at 4 KiB: memory-system 2b gives one wave's L2 plateau as 231-256 cycles over all spacings (tim_l2:
    // 233 at this point), accepted within 5 % of that range; 64 KiB: 463 (tim_l2) +-10 %
    check("mem_tlb_calib", std::fabs(cal.cyc - 81) <= 8.1 && a >= 0.95 * 231 && a <= 1.05 * 256 && std::fabs(b / 463 - 1) <= 0.10 && ptr_ok,
          fmt("L0 chase %.1f cycles (81 +-10 %%); 2,048 nodes at 4 KiB %.1f (memory-system 2b plateau 231-256, +-5 %%), at "
              "64 KiB %.1f (tim_l2 463 +-10 %%); final pointers %s",
              cal.cyc, a, b, ptr_ok ? "all right" : "WRONG"));
    CHECK(hipFree(buf));
    return failures ? 1 : 0;
}

// W one-wave groups chase at once, each its own chain. out per group: [0] cycles per load x 1000, [1] placement,
// [2] final pointer.
__global__ __launch_bounds__(32) void k_multi(const uint64_t * starts, uint32_t warm, uint32_t timed, uint64_t * out) {
    extern __shared__ uint32_t pad[];  // 40 KiB: one group per CU in CU mode
    uint64_t p = chase(starts[blockIdx.x], warm);
    const uint64_t c0 = wave_cycles();
    p = chase(p, timed);
    const uint64_t c1 = wave_cycles();
    const uint32_t pl = mem::placement();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 4 + 0] = c1 - c0;
        out[blockIdx.x * 4 + 1] = pl;
        out[blockIdx.x * 4 + 2] = p;
        pad[0] = (uint32_t) p;
    }
}

int mode_multi(bool brk) {
    uint64_t * out;
    CHECK(hipMalloc(&out, 64 * 4 * 8));
    // the calibration point and the break as in tlb
    char * cal;
    CHECK(hipMalloc(&cal, 1 << 20));
    const Pt c = chase_point(cal, 128, 128, brk, out);
    if (brk) {
        check("mem_multi_calib", false, fmt("L0 calibration %.1f cycles (deliberate break: independent loads)", c.cyc));
        return 1;
    }
    constexpr int WMAX = 16;
    constexpr size_t REGION = 8ull << 20;  // 30,226 nodes x 256 B = 7.4 MiB
    char * buf;
    CHECK(hipMalloc(&buf, WMAX * REGION));
    uint64_t * dstarts;
    CHECK(hipMalloc(&dstarts, WMAX * 8));
    std::vector<size_t> ns;
    for (double x = 2048; x < 31000; x *= 1.4) ns.push_back((size_t) (x / 64 + 0.5) * 64);
    std::map<int, std::map<size_t, double>> got;
    bool ok = c.ok;
    std::map<int, std::string> places;
    for (int W : {1, 2, 4, 8, 16}) {
        for (size_t n : ns) {
            std::vector<double> med;
            std::set<uint32_t> ses;
            size_t min_cus = 99;
            for (int seed = 0; seed < 3; ++seed) {
            std::vector<uint64_t> starts(W), expect(W);
            const uint32_t warm = (uint32_t) ((n + 15) / 16 * 16), timed = 8192;
            for (int w = 0; w < W; ++w) {
                std::vector<uint32_t> order;
                char * dev = buf + w * REGION;
                starts[w] = mem::build_chase(dev, n, 256, true, 7 + w + 100 * seed, &order);
                expect[w] = (uint64_t) dev + (uint64_t) order[(warm + timed) % n] * 256;
            }
            CHECK(hipMemcpy(dstarts, starts.data(), W * 8, hipMemcpyHostToDevice));
            for (int t = 0; t < 3; ++t) {
                std::set<uint32_t> cus;
                hipLaunchKernelGGL(k_multi, dim3(W), dim3(32), 40 << 10, 0, dstarts, warm, timed, out);
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                std::vector<uint64_t> o(W * 4);
                CHECK(hipMemcpy(o.data(), out, W * 4 * 8, hipMemcpyDeviceToHost));
                std::vector<double> per;
                for (int w = 0; w < W; ++w) {
                    per.push_back((double) o[w * 4] / timed);
                    ok &= o[w * 4 + 2] == expect[w];
                    cus.insert(mem::pl_cu((uint32_t) o[w * 4 + 1]));
                    ses.insert(mem::pl_se((uint32_t) o[w * 4 + 1]));
                }
                med.push_back(median(per));
                min_cus = std::min(min_cus, cus.size());
            }
            }
            got[W][n] = median(med);
            std::printf("RESULT multi W %2d nodes %5zu lines_in_all %6zu cycles %.1f (median wave; median of 3 chain seeds "
                        "x 3 launches: %.0f-%.0f) fewest CUs in a launch %zu, SEs %zu\n",
                        W, n, n * W, got[W][n], *std::min_element(med.begin(), med.end()),
                        *std::max_element(med.begin(), med.end()), min_cus, ses.size());
            std::fflush(stdout);
            if (min_cus < (size_t) W) places[W] = " SHARED CUs";
        }
    }
    std::map<int, size_t> step;
    for (auto & wv : got) {
        double lo = 1e9;
        for (auto & kv : wv.second) lo = std::min(lo, kv.second);
        for (auto & kv : wv.second) {
            if (kv.second >= 1.3 * lo) {
                step[wv.first] = kv.first;
                break;
            }
        }
        const auto pf = places.find(wv.first);
        std::printf("STEP multi W %d lowest %.0f cycles, first point at 1.3 x: %zu nodes per wave (%zu lines in all)%s\n",
                    wv.first, lo, step[wv.first], step[wv.first] * wv.first, pf == places.end() ? "" : pf->second.c_str());
    }
    check("mem_multi_calib",
          std::fabs(c.cyc - 81) <= 8.1 && step[1] >= 2867 && step[1] <= 15421 && ok && places.empty(),
          fmt("L0 chase %.1f cycles (81 +-10 %%); W = 1 steps at %zu nodes (the one-wave step: 2,867 to 15,421 accepted); "
              "final pointers %s; every wave on its own CU: %s",
              c.cyc, step[1], ok ? "all right" : "WRONG", places.empty() ? "yes" : "NO"));
    std::printf("VERDICT mem_multi_rule: INFO step per wave: W=1 %zu, W=2 %zu, W=4 %zu, W=8 %zu, W=16 %zu nodes (the same for "
                "every W means a limit per wave; falling as 1/W means a shared one; L2 capacity, 8 MiB = 32,768 lines, is "
                "reached at 4,096 nodes per wave for W = 8 and 2,048 for W = 16)\n",
                step[1], step[2], step[4], step[8], step[16]);
    CHECK(hipFree(buf));
    return failures ? 1 : 0;
}

// ---- MS-9 ---------------------------------------------------------------------------------------------------------
// out per wave (index = group * 16 + wave): [0] batches, [1] cycles, [2] REALTIME start, [3] REALTIME end
// the 8 loads of a batch read the wave's 2 KiB slice twice (16 waves x 2 KiB fill the 32 KiB of the group's region)
__global__ __launch_bounds__(512) void k_l0card(const char * buf, uint64_t ticks, uint64_t * out) {
    extern __shared__ uint32_t pad[];
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const char * slice = buf + (size_t) blockIdx.x * 32768 + w * 2048;
    uint64_t a = (uint64_t) slice + lane * 16;
    asm volatile("" : "+v"(a));
    const uint64_t r0 = mem::realtime();
    const uint64_t c0 = wave_cycles();
    uint64_t r = r0, n = 0;
    u4 v0, v1, v2, v3, v4, v5, v6, v7;
    while (r - r0 < ticks) {
        for (int i = 0; i < 64; ++i) {
            asm volatile("global_load_b128 %0, %8, off\n global_load_b128 %1, %8, off offset:512\n"
                         " global_load_b128 %2, %8, off offset:1024\n global_load_b128 %3, %8, off offset:1536\n"
                         " global_load_b128 %4, %8, off\n global_load_b128 %5, %8, off offset:512\n"
                         " global_load_b128 %6, %8, off offset:1024\n global_load_b128 %7, %8, off offset:1536\n"
                         " s_wait_loadcnt 0"
                         : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3), "=&v"(v4), "=&v"(v5), "=&v"(v6), "=&v"(v7)
                         : "v"(a)
                         : "memory");
        }
        n += 64;
        r = mem::realtime();
    }
    const uint64_t c1 = wave_cycles();
    if (lane == 0) {
        const int i = blockIdx.x * 16 + w;
        out[i * 4 + 0] = n;
        out[i * 4 + 1] = c1 - c0;
        out[i * 4 + 2] = r0;
        out[i * 4 + 3] = r;
        pad[0] = (uint32_t) (v0.x ^ v1.x ^ v2.x ^ v3.x ^ v4.x ^ v5.x ^ v6.x ^ v7.x);
    }
}

// the calibration: one wave, batches of NL loads of 16 bytes per lane on NL different 512-byte chunks, one wait;
// out per wave: [0] batches, [1] cycles
template <int NL>
__global__ __launch_bounds__(32) void k_l0cal(const char * buf, uint32_t batches, uint64_t * out) {
    extern __shared__ uint32_t pad[];
    uint64_t a = (uint64_t) (buf + (size_t) blockIdx.x * 32768) + threadIdx.x * 16;
    asm volatile("" : "+v"(a));
    u4 v;
    const uint64_t c0 = wave_cycles();
    for (uint32_t b = 0; b < batches; ++b) {
        asm volatile(".set l0off, 0\n .rept %c2\n global_load_b128 %0, %1, off offset:l0off\n .set l0off, l0off + 512\n"
                     " .endr\n s_wait_loadcnt 0"
                     : "=&v"(v)
                     : "v"(a), "i"(NL)
                     : "memory");
    }
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 4 + 0] = batches;
        out[blockIdx.x * 4 + 1] = c1 - c0;
        pad[0] = v.x;
    }
}

int mode_l0card(int Wsel, bool brk) {
    char * buf;
    uint64_t * out;
    CHECK(hipMalloc(&buf, 64 * 32768));
    CHECK(hipMalloc(&out, 64 * 16 * 4 * 8));
    mem::fill(buf, 64 * 32768, mem::RANDOM);
    std::map<int, std::vector<double>> per_cu, batch_cyc;
    double hottest = 0;
    bool stopped = false;
    const bool cal = Wsel == 0;  // the calibration job: one wave per CU, 8 and 16 loads per batch
    if (cal) {
        double c[2];
        for (int i = 0; i < 2; ++i) {
            std::vector<double> v;
            for (int t = 0; t < 3; ++t) {
                if (i == 0) {
                    hipLaunchKernelGGL(k_l0cal<8>, dim3(64), dim3(32), 40 << 10, 0, buf, 4096u, out);
                } else {
                    hipLaunchKernelGGL(k_l0cal<16>, dim3(64), dim3(32), 40 << 10, 0, buf, 4096u, out);
                }
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                std::vector<uint64_t> o(64 * 4);
                CHECK(hipMemcpy(o.data(), out, o.size() * 8, hipMemcpyDeviceToHost));
                std::vector<double> per;
                for (int g = 0; g < 64; ++g) per.push_back((double) o[g * 4 + 1] / (double) o[g * 4]);
                v.push_back(median(per));
            }
            c[i] = median(v);
        }
        const double inc = (c[1] - c[0]) / 8.0;
        std::printf("RESULT l0card calibration one wave per CU: %.1f cycles per batch of 8 loads, %.1f of 16; %.2f per extra "
                    "16-byte load\n", c[0], c[1], inc);
        check("mem_l0card_calib", std::fabs(inc / 4.0 - 1) <= 0.10,
              fmt("one wave per CU, cycles per extra 16-byte L0 load %.2f (memory-system item 5: (188.1 - 156.1) / 8 = "
                  "4.0, +-10 %%; 128 bytes per cycle)", inc));
        CHECK(hipFree(buf));
        return failures ? 1 : 0;
    }
    int ran = 0;
    for (int W : {Wsel}) {
        for (int b = 0; b < 1 && !stopped; ++b) {
            // thermal gate: a burst starts only below 55 C; no waiting here (the job holds the GPU lock)
            if (mem::hotspot_c() >= 55) {
                std::printf("THERMAL %.0f C: burst %d skipped\n", mem::hotspot_c(), b);
                continue;
            }
            ++ran;
            CHECK(hipMemset(out, 0, 64 * 16 * 4 * 8));
            hipLaunchKernelGGL(k_l0card, dim3(64), dim3(32 * W), 40 << 10, 0, buf, (uint64_t) 10000000, out);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const double hot = mem::hotspot_c();
            hottest = std::max(hottest, hot);
            if (hot >= 95) {
                std::printf("THERMAL %.0f C after a burst: stopping\n", hot);
                stopped = true;
            }
            std::vector<uint64_t> o(64 * 16 * 4);
            CHECK(hipMemcpy(o.data(), out, o.size() * 8, hipMemcpyDeviceToHost));
            uint64_t rmin = ~0ull, rmax = 0;
            double bytes = 0;
            std::vector<double> clk, bc;
            for (int g = 0; g < 64; ++g) {
                for (int w = 0; w < W; ++w) {
                    const uint64_t * q = &o[(g * 16 + w) * 4];
                    rmin = std::min(rmin, q[2]);
                    rmax = std::max(rmax, q[3]);
                    bytes += (double) q[0] * 8 * 512;
                    clk.push_back((double) q[1] / ((double) (q[3] - q[2]) * 10.0));
                    bc.push_back((double) q[1] / (double) q[0]);
                }
            }
            const double ns = (double) (rmax - rmin) * 10.0, ghz = median(clk);
            const double pcu = bytes / ns / ghz / 64.0;
            per_cu[W].push_back(pcu);
            batch_cyc[W].push_back(median(bc));
            std::printf("RESULT l0card W %2d burst %d bytes_per_cycle_per_CU %.1f TB/s %.2f clock_GHz %.3f "
                        "cycles_per_batch_per_wave %.1f hotspot_after %.0f C\n",
                        W, b, pcu, bytes / ns / 1000.0, ghz, median(bc), hot);
            std::fflush(stdout);
        }
    }
    std::printf("THERMAL highest hotspot after a burst %.0f C\n", hottest);
    double top = 0;
    for (auto & kv : per_cu) top = std::max(top, *std::max_element(kv.second.begin(), kv.second.end()));
    const double ceil = brk ? 64.0 : 128.0;
    check("mem_l0card_ceiling", ran == 1 && top <= 1.05 * ceil,
          fmt("W = %d: %.1f bytes per cycle per CU (ceiling %.0f%s, limit 1.05 x)", Wsel, top, ceil,
              brk ? ": deliberate break, 64 claimed" : ""));
    std::printf("VERDICT mem_l0card_rate: INFO W = %d: %.1f bytes per cycle per CU (%.2f of 128)\n", Wsel,
                ran ? median(per_cu[Wsel]) : NAN, ran ? median(per_cu[Wsel]) / 128.0 : NAN);
    CHECK(hipFree(buf));
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = mem::has_flag(argc, argv, "--break");
    mem::clocks("start");
    int rc = 2;
    if (mode == "ns") {
        rc = mode_ns(brk);
    } else if (mode == "tlb") {
        rc = mode_tlb(brk);
    } else if (mode == "multi") {
        rc = mode_multi(brk);
    } else if (mode == "l0card") {
        rc = mode_l0card(argc > 2 ? std::atoi(argv[2]) : 1, brk);
    } else {
        std::fprintf(stderr, "usage: left_mem <ns|tlb|multi> [--break]\n");
    }
    mem::clocks("end");
    return rc;
}
