// Leftovers C, lane A, SC-2 and SC-1: is the regular card-wide stall (about 1.1 us every 2.8 ms,
// scheduling-and-clocks item 7) seen when the wave reads no REALTIME message, and how well does the disturbance rule
// (a step above 1.5 x median + 64 cycles is disturbed) find real and injected gaps?
//
// Usage: left_spin [--break]
//   One wave (32 lanes) runs steps of 256 independent FMAs (as sched_share spin: about 480 cycles per step) and keeps
//   the cycles of every step. Three variants, launched in turn R, N, Q, R, N, Q (about 1.5 s each, one job):
//     R  reads REALTIME after every step (s_sendmsg_rtn_b64, as sched_share spin) and stores [step cycles, REALTIME]
//     N  stores the same 8 bytes per step but reads no REALTIME (the store traffic is the same as R's)
//     Q  reads no REALTIME and stores only the steps above the limit (no store per step)
//   A fourth launch (variant N) injects `s_sleep k` after chosen steps (about 200 steps for each k in 2, 3, 4, 5, 6, 8,
//   32, 127; s_sleep k waits 64 k cycles, core_counter: 8,128 for k = 127) and the detector runs on the trace. The
//   steps are chosen by a hash of the step number computed in the kernel (scalar ALU, no memory read per step), and
//   the host recomputes the same hash.
//   Verdicts: spin_detector (synthetic trace: clean gives 0 events, one injected 50 us gap gives 1; s_sleep 127 alone
//   8,128 +-64 cycles), spin_events_seen (every R launch sees at least 50 events per second; earlier runs 390 to
//   1,023), spin_inject_recall (the rule flags a step whose excess over the median step is above 0.5 x median + 64
//   cycles: an injected sleep 128 cycles or more above that excess must be found in at least 99 % of its steps (a
//   sleep overlaps the work still in flight, so its real loss is a little below 64 k: measured 60 to 90 cycles less),
//   one 48 cycles or more below it in at most 1 %; the sleeps between are reported). The R against N and Q
//   comparison and the false-flag rate are INFO.
//   --break: expects 0 events in the R launches and the 128-cycle sleeps (k = 2) to be found: both verdicts FAIL.
#include "sched_common.h"

#include <cstdarg>
#include <set>

using namespace sched;

namespace {

#define FMA1(i) "v_fma_f32 %" #i ", %16, %17, %" #i "\n"
#define FMA16 FMA1(0) FMA1(1) FMA1(2) FMA1(3) FMA1(4) FMA1(5) FMA1(6) FMA1(7) FMA1(8) FMA1(9) FMA1(10) FMA1(11) \
    FMA1(12) FMA1(13) FMA1(14) FMA1(15)
#define X16 "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), \
    "+v"(x[8]), "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])

struct Rec {
    uint32_t dc, rt;
};

// slot j of the injection schedule (inj_slot below) -> s_sleep KS[j]; a switch of branches, not a table in memory,
// so an injected step reads nothing from memory (checked in the listing: no load in the loop)
__device__ __forceinline__ void do_sleep(uint32_t j) {
    switch (j) {
        case 0: asm volatile("s_sleep 2" ::: "memory"); break;
        case 1: asm volatile("s_sleep 3" ::: "memory"); break;
        case 2: asm volatile("s_sleep 4" ::: "memory"); break;
        case 3: asm volatile("s_sleep 5" ::: "memory"); break;
        case 4: asm volatile("s_sleep 6" ::: "memory"); break;
        case 5: asm volatile("s_sleep 8" ::: "memory"); break;
        case 6: asm volatile("s_sleep 32" ::: "memory"); break;
        default: asm volatile("s_sleep 127" ::: "memory"); break;
    }
}

// The injection schedule: step n gets s_sleep KS[j] when hash(n) falls in slot j of a 1-in-INJ_EVERY window.
constexpr uint32_t INJ_EVERY = 6000, NK = 8, INJ_FROM = 200000;
__host__ __device__ inline uint32_t inj_hash(uint32_t n) {
    uint32_t x = n * 0x9E3779B9u ^ 0x5bd1e995u;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
// slot + 1 of step n, or 0 for no injection
__host__ __device__ inline uint32_t inj_slot(uint32_t n) {
    const uint32_t h = inj_hash(n);
    return n >= INJ_FROM && h % INJ_EVERY == 0 ? 1 + (h / INJ_EVERY) % NK : 0;
}
const uint32_t KS[NK] = {2, 3, 4, 5, 6, 8, 32, 127};

// variant: 0 R, 1 N, 2 Q. out_n: [0] steps, [1] events stored (Q).
template <int VAR, bool INJ>
__global__ __launch_bounds__(32) void k_spin(const float * in, Rec * tr, uint32_t nsteps, uint32_t lim,
                                             uint32_t * out_n) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0] + i;
    }
    const float a = in[1], b = in[2];
    uint64_t prev = cyc();
    uint32_t nev = 0;
    for (uint32_t n = 0; n < nsteps; ++n) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            asm volatile(FMA16 : X16 : "v"(a), "v"(b));
        }
        if (INJ) {
            const uint32_t j = __builtin_amdgcn_readfirstlane(inj_slot(n));
            if (j) {
                do_sleep(j - 1);
            }
        }
        const uint64_t c = cyc();
        uint32_t r = 0;
        if (VAR == 0) {
            r = (uint32_t) rt();
        }
        const uint32_t dc = (uint32_t) (c - prev);
        prev = c;
        if (threadIdx.x == 0) {
            if (VAR == 2) {
                if (dc > lim) {
                    tr[nev] = {dc, n};
                    ++nev;
                }
            } else {
                tr[n] = {dc, r};
            }
        }
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (threadIdx.x == 0) {
        out_n[0] = nsteps;
        out_n[1] = nev;
        if (s == 1234.5f) {
            out_n[2] = 1;
        }
    }
}

__global__ __launch_bounds__(32) void k_sleep_alone(uint64_t * out) {
    for (int i = 0; i < 8; ++i) {
        const uint64_t t0 = cyc();
        asm volatile("s_sleep 127" ::: "memory");
        const uint64_t t1 = cyc();
        if (threadIdx.x == 0) {
            out[i] = t1 - t0;
        }
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? NAN : v[v.size() / 2];
}

// Events: runs of consecutive steps above lim. Returns the first step index of each event and its extra cycles.
struct Ev {
    size_t first, last;
    double extra;
};
std::vector<Ev> detect(const std::vector<uint32_t> & dc, double base) {
    const double lim = 1.5 * base + 64;
    std::vector<Ev> ev;
    for (size_t i = 0; i < dc.size(); ++i) {
        if (dc[i] > lim) {
            if (!ev.empty() && ev.back().last + 1 == i) {
                ev.back().extra += dc[i] - base;
                ev.back().last = i;
            } else {
                ev.push_back({i, i, dc[i] - base});
            }
        }
    }
    return ev;
}

double base_of(const std::vector<uint32_t> & dc) {
    std::vector<double> s;
    for (size_t i = 0; i < dc.size(); i += std::max<size_t>(1, dc.size() / 200000)) s.push_back(dc[i]);
    return median(s);
}

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1200];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    bool brk = false;
    for (int i = 1; i < argc; ++i) brk |= std::strcmp(argv[i], "--break") == 0;
    constexpr uint32_t NSTEPS = 10000000;  // about 1.5 s at 480 cycles per step and 3.2 GHz
    float * d_in;
    Rec * d_tr;
    uint32_t * d_n;
    uint64_t * d_s;
    CHECK(hipMalloc(&d_in, 16));
    CHECK(hipMalloc(&d_tr, (size_t) NSTEPS * sizeof(Rec)));
    CHECK(hipMalloc(&d_n, 16));
    CHECK(hipMalloc(&d_s, 64));
    const float hin[4] = {1.0f, 1.0f, 0.0f, 0.0f};  // x = x * 1 + 0 stays put
    CHECK(hipMemcpy(d_in, hin, 16, hipMemcpyHostToDevice));

    // detector self-check on a synthetic trace (290..305 cycles per step), clean and with one 50 us gap
    {
        std::vector<uint32_t> syn(200000);
        uint64_t xs = 12345;
        for (auto & v : syn) {
            xs = xs * 6364136223846793005ull + 1442695040888963407ull;
            v = 290 + (uint32_t) (xs >> 60);
        }
        const size_t e0 = detect(syn, base_of(syn)).size();
        syn[100000] += 160000;  // 50 us at 3.2 GHz
        const size_t e1 = detect(syn, base_of(syn)).size();
        hipLaunchKernelGGL(k_sleep_alone, dim3(1), dim3(32), 0, 0, d_s);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t sl[8];
        CHECK(hipMemcpy(sl, d_s, sizeof sl, hipMemcpyDeviceToHost));
        std::vector<double> slv(sl + 1, sl + 8);
        const double s127 = median(slv);
        check("spin_detector", e0 == 0 && e1 == 1 && std::fabs(s127 - 8128) <= 64,
              "synthetic trace: clean %zu events (expect 0), one 50 us gap %zu (expect 1); s_sleep 127 alone %.0f cycles "
              "(core_counter 8,128; limit +-64)",
              e0, e1, s127);
    }

    // first a short R launch to find the median step for Q's limit
    hipLaunchKernelGGL((k_spin<1, false>), dim3(1), dim3(32), 0, 0, d_in, d_tr, 200000u, 0u, d_n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<Rec> h(NSTEPS);
    CHECK(hipMemcpy(h.data(), d_tr, 200000 * sizeof(Rec), hipMemcpyDeviceToHost));
    std::vector<uint32_t> pre(200000);
    for (size_t i = 0; i < pre.size(); ++i) pre[i] = h[i].dc;
    const double pre_base = base_of(pre);
    const uint32_t qlim = (uint32_t) (1.5 * pre_base + 64);
    std::printf("RESULT spin pre-run median step %.0f cycles; limit for Q %u\n", pre_base, qlim);

    struct Row {
        int var;
        double base, per_s, per_mstep, extra_med, secs;
        size_t events;
    };
    std::vector<Row> rows;
    const char * vn[3] = {"R (REALTIME every step)", "N (no REALTIME, store every step)", "Q (no REALTIME, store events only)"};
    for (int round = 0; round < 2; ++round) {
        for (int var = 0; var < 3; ++var) {
            const auto w0 = std::chrono::steady_clock::now();
            if (var == 0) {
                hipLaunchKernelGGL((k_spin<0, false>), dim3(1), dim3(32), 0, 0, d_in, d_tr, NSTEPS, 0u, d_n);
            } else if (var == 1) {
                hipLaunchKernelGGL((k_spin<1, false>), dim3(1), dim3(32), 0, 0, d_in, d_tr, NSTEPS, 0u, d_n);
            } else {
                hipLaunchKernelGGL((k_spin<2, false>), dim3(1), dim3(32), 0, 0, d_in, d_tr, NSTEPS, qlim, d_n);
            }
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            uint32_t on[2];
            CHECK(hipMemcpy(on, d_n, 8, hipMemcpyDeviceToHost));
            Row r{var, 0, 0, 0, 0, 0, 0};
            std::vector<double> extra;
            if (var < 2) {
                CHECK(hipMemcpy(h.data(), d_tr, (size_t) NSTEPS * sizeof(Rec), hipMemcpyDeviceToHost));
                // drop the first 200,000 steps (about 30 ms: clock ramp)
                std::vector<uint32_t> dc(NSTEPS - 200000);
                for (size_t i = 0; i < dc.size(); ++i) dc[i] = h[i + 200000].dc;
                r.base = base_of(dc);
                const auto ev = detect(dc, r.base);
                r.events = ev.size();
                for (const Ev & e : ev) extra.push_back(e.extra);
                if (var == 0) {
                    r.secs = (double) (uint32_t) (h[NSTEPS - 1].rt - h[200000].rt) / 1e8;
                }
            } else {
                std::vector<Rec> evr(on[1]);
                CHECK(hipMemcpy(evr.data(), d_tr, on[1] * sizeof(Rec), hipMemcpyDeviceToHost));
                r.base = pre_base;
                size_t last = SIZE_MAX;
                for (const Rec & e : evr) {
                    if (e.rt < 200000) continue;
                    if (last != SIZE_MAX && e.rt == last + 1) {
                        extra.back() += e.dc - r.base;
                    } else {
                        ++r.events;
                        extra.push_back(e.dc - r.base);
                    }
                    last = e.rt;
                }
            }
            r.per_mstep = r.events / ((NSTEPS - 200000) / 1e6);
            // time base for N and Q: the wall time of the launch scaled to the timed steps (no REALTIME read)
            if (var != 0) r.secs = wall * (NSTEPS - 200000) / (double) NSTEPS;
            r.per_s = r.events / r.secs;
            r.extra_med = median(extra);
            rows.push_back(r);
            std::printf("RESULT spin round %d %-36s median_step %.0f events %zu per_s %.1f per_million_steps %.2f "
                        "extra_cycles_median %.0f seconds %.3f%s\n",
                        round, vn[var], r.base, r.events, r.per_s, r.per_mstep, r.extra_med, r.secs,
                        var == 0 ? " (REALTIME)" : " (launch wall time)");
            std::fflush(stdout);
        }
    }
    double minR = 1e9;
    for (const Row & r : rows) {
        if (r.var == 0) minR = std::min(minR, r.per_s);
    }
    check("spin_events_seen", brk ? minR == 0 : minR >= 50,
          "R launches: at least %.1f events per second (expect %s; earlier runs 390 to 1,023)", minR,
          brk ? "0: deliberate break" : ">= 50");
    auto avg = [&](int var, double Row::*f) {
        double s = 0;
        int n = 0;
        for (const Row & r : rows) {
            if (r.var == var) {
                s += r.*f;
                ++n;
            }
        }
        return s / n;
    };
    std::printf("VERDICT spin_no_rt: INFO events per million steps R %.2f, N %.2f, Q %.2f (ratio N/R %.2f, Q/R %.2f; the "
                "plan's band 0.7 to 1.4); median extra cycles per event R %.0f, N %.0f, Q %.0f (ratio N/R %.2f, Q/R "
                "%.2f; band 0.85 to 1.15)\n",
                avg(0, &Row::per_mstep), avg(1, &Row::per_mstep), avg(2, &Row::per_mstep),
                avg(1, &Row::per_mstep) / avg(0, &Row::per_mstep), avg(2, &Row::per_mstep) / avg(0, &Row::per_mstep),
                avg(0, &Row::extra_med), avg(1, &Row::extra_med), avg(2, &Row::extra_med),
                avg(1, &Row::extra_med) / avg(0, &Row::extra_med), avg(2, &Row::extra_med) / avg(0, &Row::extra_med));

    // SC-1: injected sleeps
    const int ks[NK] = {2, 3, 4, 5, 6, 8, 32, 127};
    std::vector<uint8_t> inj(NSTEPS, 0);
    for (uint32_t i = 0; i < NSTEPS; ++i) {
        const uint32_t j = inj_slot(i);
        inj[i] = (uint8_t) (j ? KS[j - 1] : 0);
    }
    hipLaunchKernelGGL((k_spin<1, true>), dim3(1), dim3(32), 0, 0, d_in, d_tr, NSTEPS, 0u, d_n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h.data(), d_tr, (size_t) NSTEPS * sizeof(Rec), hipMemcpyDeviceToHost));
    std::vector<uint32_t> dc(NSTEPS);
    for (size_t i = 0; i < NSTEPS; ++i) dc[i] = h[i].dc;
    // the base from steps after the ramp, without injected ones
    std::vector<uint32_t> plain;
    for (size_t i = 200000; i < NSTEPS; i += 50) {
        if (!inj[i]) plain.push_back(dc[i]);
    }
    const double base = base_of(plain);
    const double lim = 1.5 * base + 64;
    std::map<int, int> found, total;
    std::map<int, std::vector<double>> ext;
    // the injected sleep shows in the step it follows (the counter read after the sleep)
    for (size_t i = 200000; i < NSTEPS; ++i) {
        if (inj[i]) {
            found[inj[i]] += dc[i] > lim;
            ++total[inj[i]];
            ext[inj[i]].push_back((double) dc[i] - base);
        }
    }
    size_t flagged_other = 0, small_other = 0, other_steps = 0;
    for (size_t i = 200000; i < NSTEPS; ++i) {
        if (inj[i]) continue;
        ++other_steps;
        if (dc[i] > lim) {
            ++flagged_other;
            small_other += dc[i] - base < 1500;
        }
    }
    std::string tab, above, below, near;
    bool rec_ok = true;
    const double lim_ex = lim - base;
    for (int k : ks) {
        char b[160];
        const double fr = total[k] ? (double) found[k] / total[k] : NAN;
        std::snprintf(b, sizeof b, " k=%d (%d cycles): found %d/%d, extra median %.0f;", k, 64 * k, found[k], total[k],
                      median(ext[k]));
        tab += b;
        std::snprintf(b, sizeof b, " %d:%d/%d", 64 * k, found[k], total[k]);
        if (brk && k == 2) {
            rec_ok &= fr >= 0.99;  // deliberate break: the 128-cycle sleeps claimed to be found
            below += b;
        } else if (64 * k >= lim_ex + 128) {
            rec_ok &= total[k] >= 50 && fr >= 0.99;
            above += b;
        } else if (64 * k <= lim_ex - 48) {
            rec_ok &= total[k] >= 50 && fr <= 0.01;
            below += b;
        } else {
            near += b;
        }
    }
    std::printf("RESULT spin inject median_step %.0f limit %.0f (excess above the median %.0f);%s\n", base, lim,
                lim_ex, tab.c_str());
    check("spin_inject_recall", rec_ok,
          "limit: excess above %.0f cycles over the median step %.0f; sleeps (cycles: found/injected) 128 or more above "
          "it, expect >= 99 %%:%s; 48 or more below, expect <= 1 %%%s:%s; between (reported):%s",
          lim_ex, base, above.c_str(), brk ? " (deliberate break: 128 cycles expected found)" : "", below.c_str(),
          near.c_str());
    std::printf("VERDICT spin_false_flags: INFO steps flagged outside the injected ones: %zu of %zu (%.2f per 100,000 "
                "steps), of which %zu with an excess below 1,500 cycles (%.2f per 100,000); the ground truth for these "
                "is not known (desktop drawing is real disturbance)\n",
                flagged_other, other_steps, 1e5 * flagged_other / other_steps, small_other, 1e5 * small_other / other_steps);
    return fails ? 1 : 0;
}
