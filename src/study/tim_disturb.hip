// Timing leftovers, item 1: how disturbed single-wave timings look, with numbers.
//
// Usage: tim_disturb <stats|ramp> [--break]
//   stats  one wave, after 30 ms of warm-up on the same work (the clock ramp, lane 7 item 1, is over by then), times
//          the same calibration loop thousands of times and keeps every sample: a short loop (512 independent FMAs,
//          about 500 cycles), a medium one (10,240, about 10,000) and a long one (1,024,000). Each sample also keeps
//          its start counter and REALTIME, so the spacing of the disturbed samples in time is known. Reports the
//          share of samples the lane 7 rule flags (above 1.5 x median + 64 cycles), their excess, the median and
//          worst with and without the rule, and for the long loop (where one 3,900-cycle stall is far below the
//          rule's threshold) the excess over the best sample in units of one stall.
//          --break claims an FMA costs 2 cycles: the calibration verdict must FAIL.
//   ramp   the card idles 0.5 s, then one wave times chunks of 4,096 FMAs back to back for about 40 ms with no
//          warm-up, recording cycles and REALTIME per chunk: cycles per FMA against the clock during the ramp.
#include "tim_common.h"

#include <chrono>
#include <thread>

namespace {

constexpr int FMA_PER_IT = 256;  // 16 x TIM_FMA16: a 2 KB loop body (above 4 KB the loop itself costs more, tim_core size)

__global__ __launch_bounds__(32) void k_samples(uint32_t n_iter, uint32_t n_samples, uint32_t warm_ticks,
                                                uint32_t * out) {
    float r[18];
    for (int i = 0; i < 16; ++i) {
        r[i] = 1.0f;
    }
    r[16] = r[17] = 0.5f;
    uint32_t id0, id1;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(id0));
    const uint64_t rs = mem::realtime();
    while (mem::realtime() - rs < warm_ticks) {
        uint32_t t0, t1, cnt;
        asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], 8\n 2:\n"
                     ".rept 16\n" TIM_FMA16 ".endr\n"
                     " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"
                     " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                     : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), TIM_FMA_REGS(r)
                     :
                     : "scc", "memory");
    }
    for (uint32_t s = 0; s < n_samples; ++s) {
        const uint32_t rt = (uint32_t) mem::realtime();
        uint32_t t0, t1, cnt;
        asm volatile("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)\n s_mov_b32 %[cnt], %[n]\n 2:\n"
                     ".rept 16\n" TIM_FMA16 ".endr\n"
                     " s_add_co_i32 %[cnt], %[cnt], -1\n s_cmp_lg_u32 %[cnt], 0\n s_cbranch_scc1 2b\n"
                     " s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)\n"
                     : [t0] "=&s"(t0), [t1] "=&s"(t1), [cnt] "=&s"(cnt), TIM_FMA_REGS(r)
                     : [n] "s"(n_iter)
                     : "scc", "memory");
        if (threadIdx.x == 0) {
            out[3 * s + 0] = t0;
            out[3 * s + 1] = t1 - t0;
            out[3 * s + 2] = rt;
        }
    }
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(id1));
    float acc = 0;
    for (int i = 0; i < 16; ++i) {
        acc += r[i];
    }
    if (threadIdx.x == 0) {
        out[3 * n_samples + 0] = id0;
        out[3 * n_samples + 1] = id1;
        out[3 * n_samples + 2] = __float_as_uint(acc);
    }
}

struct Run {
    std::vector<double> cyc, t0, rt;  // per sample: cycles, start counter, start REALTIME (ticks of 10 ns)
    bool moved;
};

Run launch(uint32_t n_iter, uint32_t n_samples, uint32_t warm_ticks) {
    uint32_t * d = tim::dev_u32(3 * n_samples + 3);
    hipLaunchKernelGGL(k_samples, dim3(1), dim3(32), 0, 0, n_iter, n_samples, warm_ticks, d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    const std::vector<uint32_t> h = tim::host_u32(d, 3 * n_samples + 3);
    CHECK(hipFree(d));
    Run r;
    uint32_t c0 = h[0], q0 = h[2];
    double cacc = 0, racc = 0;
    for (uint32_t s = 0; s < n_samples; ++s) {
        // unwrap the 32-bit counters into running totals
        if (s > 0) {
            cacc += (double) (uint32_t) (h[3 * s] - c0);
            racc += (double) (uint32_t) (h[3 * s + 2] - q0);
            c0 = h[3 * s];
            q0 = h[3 * s + 2];
        }
        r.cyc.push_back(h[3 * s + 1]);
        r.t0.push_back(cacc);
        r.rt.push_back(racc);
    }
    r.moved = h[3 * n_samples] != h[3 * n_samples + 1];
    return r;
}

double quant(std::vector<double> v, double q) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (q * v.size()))];
}

int mode_stats(bool brk) {
    struct Len {
        const char * name;
        uint32_t n_iter, n_samples;
    };
    const Len lens[3] = {{"short", 2, 200000}, {"medium", 40, 30000}, {"long", 4000, 400}};
    double med[3] = {0, 0, 0}, all_med[3] = {0, 0, 0}, best[3] = {0, 0, 0};
    double medium_excess = NAN, medium_spacing_ms = NAN;
    for (int li = 0; li < 3; ++li) {
        const Len & L = lens[li];
        const Run r = launch(L.n_iter, L.n_samples, 3000000);
        const tim::Split s = tim::split(r.cyc);
        med[li] = s.median_clean;
        all_med[li] = s.median_all;
        best[li] = s.best;
        const double span_ticks = r.rt.back() - r.rt.front();
        const double ghz = (r.t0.back() - r.t0.front()) / span_ticks / 10.0;
        double busy = 0;
        for (double c : r.cyc) {
            busy += c;
        }
        const double busy_ms = busy / ghz / 1e6;
        // excess of the flagged samples, and their spacing in time
        std::vector<double> ex, gaps;
        double last = -1;
        const double thr = 1.5 * s.median_all + 64;
        for (size_t i = 0; i < r.cyc.size(); ++i) {
            if (r.cyc[i] > thr) {
                ex.push_back(r.cyc[i] - s.median_clean);
                if (last >= 0) {
                    gaps.push_back((r.rt[i] - last) / 1e5);  // ms
                }
                last = r.rt[i];
            }
        }
        std::printf("RESULT disturb %s fma %u samples %zu clock_GHz %.3f timed_ms %.1f span_ms %.1f moved %d\n", L.name,
                    L.n_iter * FMA_PER_IT, s.n, ghz, busy_ms, span_ticks / 1e5, r.moved ? 1 : 0);
        std::printf("RESULT disturb %s all: median %.0f mean %.1f worst %.0f | rule: dropped %zu (%.4f %%) "
                    "median %.0f mean %.1f worst %.0f | best %.0f\n",
                    L.name, s.median_all, s.mean_all, s.worst_all, s.dropped, 100.0 * s.dropped / s.n, s.median_clean,
                    s.mean_clean, s.worst_clean, s.best);
        std::printf("RESULT disturb %s dropped excess cycles p10 %.0f p50 %.0f p90 %.0f max %.0f | spacing ms p10 %.2f "
                    "p50 %.2f p90 %.2f | dropped per ms of timed work %.3f\n",
                    L.name, quant(ex, 0.1), quant(ex, 0.5), quant(ex, 0.9), quant(ex, 1.0), quant(gaps, 0.1),
                    quant(gaps, 0.5), quant(gaps, 0.9), s.dropped / busy_ms);
        // excess over the best sample in units of one regular stall (about 3,900 cycles): the rule cannot see a
        // single stall in a sample this much longer than the stall, this view can
        int hist[6] = {0, 0, 0, 0, 0, 0};
        for (double c : r.cyc) {
            const int k = (int) std::lround((c - s.best) / 3900.0);
            hist[std::min(5, std::max(0, k))]++;
        }
        std::printf("RESULT disturb %s excess_over_best_in_stalls 0:%d 1:%d 2:%d 3:%d 4:%d 5+:%d\n", L.name, hist[0],
                    hist[1], hist[2], hist[3], hist[4], hist[5]);
        if (li == 1) {
            medium_excess = quant(ex, 0.5);
            medium_spacing_ms = quant(gaps, 0.5);
        }
    }
    // Cycles per loop iteration (256 FMAs + the loop control) from the differences of the clean medians; expected
    // 256 x 1.00 + 20 (loop control: s_add/s_cmp/s_cbranch with clang's waits, 19.9 cycles in tim_core size and lane 1
    // section 5).
    const double it1 = (med[1] - med[0]) / 38.0, it2 = (med[2] - med[1]) / 3960.0;
    const double want = brk ? 2.0 * FMA_PER_IT + 20 : FMA_PER_IT + 20.0;
    tim::check("tim_disturb_calib", std::fabs(it1 - want) < 3 && std::fabs(it2 - want) < 3,
               "%.2f cycles per iteration (medium - short) and %.2f (long - medium), clean medians; expected %.0f "
               "(%d independent FMAs at %s cycle + 20 of loop control)%s",
               it1, it2, want, FMA_PER_IT, brk ? "2" : "1", brk ? " (deliberate break)" : "");
    // The statistic to quote: the plain median must not move when the rule drops the disturbed samples, and in the
    // long loop (where every sample is likely to contain a stall) it must stay within 0.5 % of the best sample.
    tim::check("tim_disturb_median",
               std::fabs(all_med[0] - med[0]) <= 0.001 * med[0] && std::fabs(all_med[1] - med[1]) <= 0.001 * med[1] &&
                   all_med[2] <= 1.005 * best[2],
               "median of all samples against the rule-cleaned median: short %.0f / %.0f, medium %.0f / %.0f; long: "
               "median %.0f, best %.0f (+%.2f %%)",
               all_med[0], med[0], all_med[1], med[1], all_med[2], best[2], 100.0 * (all_med[2] / best[2] - 1));
    std::printf("RESULT disturb medium_loop_flagged_excess_median %.0f spacing_ms_median %.2f\n", medium_excess,
                medium_spacing_ms);
    return 0;
}

int mode_ramp() {
    launch(16, 16, 0);  // loads the code object, so the timed launch below does not include it
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const uint32_t n = 24000;
    const Run r = launch(16, n, 0);
    const double fma = 16.0 * FMA_PER_IT;
    // per 0.5 ms bin: clock from the start counters and REALTIME, cycles per FMA (median, max), flagged samples
    std::printf("RESULT ramp moved %d samples %u span_ms %.2f\n", r.moved ? 1 : 0, n, (r.rt.back() - r.rt.front()) / 1e5);
    std::vector<double> late, early;
    size_t i = 0;
    double worst_early = 0, gmin = 1e9, gmax = 0;
    while (i + 1 < r.cyc.size()) {
        const double b0 = r.rt[i];
        size_t j = i;
        std::vector<double> per;
        while (j + 1 < r.cyc.size() && r.rt[j] - b0 < 50000) {
            per.push_back(r.cyc[j] / fma);
            ++j;
        }
        if (j == i) {
            break;
        }
        const double ghz = (r.t0[j] - r.t0[i]) / (r.rt[j] - r.rt[i]) / 10.0;
        const double tms = r.rt[i] / 1e5;
        gmin = std::min(gmin, ghz);
        gmax = std::max(gmax, ghz);
        const tim::Split s = tim::split(per);
        std::printf("RESULT ramp t_ms %.1f clock_GHz %.3f cycles_per_fma median %.4f max %.4f samples %zu\n", tms, ghz,
                    s.median_all, s.worst_all, per.size());
        for (double x : per) {
            (tms >= 20 ? late : early).push_back(x);
        }
        if (tms < 20) {
            worst_early = std::max(worst_early, s.median_all);
        }
        i = j;
    }
    const double ml = tim::median_of(late), me = tim::median_of(early);
    // per chunk: 16 iterations of 256 FMAs + 20 cycles of loop control (tim_disturb stats: 277 per iteration)
    tim::check("tim_ramp_calib", std::fabs(ml * FMA_PER_IT - 276) < 3 && late.size() > 1000,
               "after 20 ms: %.2f cycles per iteration (median of %zu chunks); expected 276 (256 FMAs + 20 of loop control)",
               ml * FMA_PER_IT, late.size());
    // the claim of item 1: the clock ramp changes the clock, not the cycles per instruction
    // only decisive if the clock did ramp during the launch
    tim::check("tim_ramp_cycles", std::fabs(me - ml) < 0.001 && worst_early - ml < 0.001 && gmax > 1.1 * gmin,
               "cycles per FMA (loop control included) during the first 20 ms: median %.4f, worst 0.5 ms bin median "
               "%.4f; after 20 ms %.4f (claim: the same within 0.001) while the clock went from %.3f to %.3f GHz "
               "(needs a ramp of at least 10 %%)",
               me, worst_early, ml, gmin, gmax);
    std::printf("RESULT ramp first_20ms median %.4f worst_bin_median %.4f chunks %zu\n", tim::median_of(early),
                worst_early, early.size());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = tim::flag(argc, argv, "--break");
    mem::clocks("start");
    if (mode == "stats") {
        mode_stats(brk);
    } else if (mode == "ramp") {
        mode_ramp();
    } else {
        std::fprintf(stderr, "usage: tim_disturb <stats|ramp> [--break]\n");
        return 2;
    }
    mem::clocks("end");
    return tim::failures() ? 1 : 0;
}
