// Lane 7 item 7: sharing the card with the desktop and with a second compute process.
//
// Usage: sched_share <spin|selftest|pair|multi|gate> [options]
//   spin     --seconds S --out <prefix>: one wave runs fixed steps of 256 independent FMAs and records, after every
//            step, its cycle counter, REALTIME and HW_ID1. Interruptions are found by the detector below and related
//            to gpu_busy_percent and to the graphics-engine time of other processes (/proc/*/fdinfo, sampled every
//            20 ms). A gap is injected into a clean stretch of the real trace: the detector must find exactly it.
//   selftest [--expect-clean N] [--expect-injected N]: synthetic traces (no interruption; one injected 50 us gap);
//            the detector must report 0 and 1. The script's deliberate breaks claim 1 and 0, so the verdicts FAIL.
//   pair     --role A|B --kind big|small --start-ms <CLOCK_REALTIME ms> --seconds S --out <prefix>: one of two
//            processes started together by the script inside one locked job. big: whole-card FMA launches of about
//            5 ms, every group records its CU and REALTIME start/end; small: the spin trace. scripts/sched_pair.py
//            merges the two files on REALTIME (chip-wide, so both processes share the time base).
//   multi    [--seconds S]: 16 single-wave groups spin at once; are their interruptions simultaneous within a
//            shader engine and across shader engines? Also checks HW_ID1 bits 19:18 against the SE message.
//   gate     waits until the hotspot is below 60 C (used before the two-process whole-card job).
//   preempt  --role A|B|AB --start-ms <CLOCK_REALTIME ms> --out <prefix> [--prio high|normal] [--trials N]
//            [--ta-ms 4,16,64,180]: leftovers C item SC-11. Process A launches a kernel of 64 groups x 2
//            waves with 40 KiB of LDS each (one group per CU in CU mode, so no second such group fits anywhere) whose
//            waves sleep in short s_sleep steps until T_A has passed; process B launches the same shape for 1 ms,
//            1 ms after A's launch. Every wave records REALTIME at start and end and its longest gap between two
//            consecutive REALTIME reads (a wave that was saved and restored shows a gap). Every fifth trial A
//            launches nothing: B alone is the control (launch to first group start). AB: both kernels from one
//            process, A on the lowest- and B on the highest-priority stream. scripts/sched_pair.py merges the files.
//            Gaming round P3 (options for A only; B keeps its shape): --a-groups N (default 64), --a-lds-kib K (default
//            40; 0 lets B's groups fit beside A's on the same CUs), --a-hold sleep|fma (fma: the waves run a dependent
//            FMA chain between REALTIME reads instead of sleeping: busy waves at low power), --a-inject-sleeps K
//            (gaming-round E8, the gap detector's known answer: wave 0 of A's group 0, once past half of T_A, runs K
//            s_sleep 127 steps without reading REALTIME, a gap of about K x 8,000 cycles).
//
// Detector (also the rule for finding a disturbed cycle-counter sample from inside a kernel): a step is disturbed
// when its cycle count exceeds 1.5 x the median step plus 64 cycles, or when HW_ID1 changed. Consecutive disturbed
// steps form one event. A step whose REALTIME length grows without extra cycles is a clock change, not a
// disturbance of a cycle count, and is counted separately.
#include "sched_common.h"

#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

#include <ctime>
#include <set>

using namespace sched;

namespace {

struct Step {
    uint64_t cyc;
    uint32_t rt;  // low 32 bits of REALTIME (wraps after 42.9 s; traces are shorter)
    uint32_t hw;
};

#define FMA1(i) "v_fma_f32 %" #i ", %16, %17, %" #i "\n"
#define FMA16 FMA1(0) FMA1(1) FMA1(2) FMA1(3) FMA1(4) FMA1(5) FMA1(6) FMA1(7) FMA1(8) FMA1(9) FMA1(10) FMA1(11) \
    FMA1(12) FMA1(13) FMA1(14) FMA1(15)
#define X16 "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), \
    "+v"(x[8]), "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])

__global__ __launch_bounds__(32) void k_spin(const float * in, Step * tr, int nmax, uint64_t ticks,
                                             volatile uint64_t * host_flag, int * count) {
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = in[0] + i;
    }
    const float a = in[1], b = in[2];
    const uint64_t r0 = rt();
    // Several single-wave groups (mode multi) each write their own trace; group 0 also maps host time.
    tr += (size_t) blockIdx.x * nmax;
    const uint32_t se = se_aid();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *host_flag = r0;  // lets the host map its clock to REALTIME
        __threadfence_system();
    }
    int n = 0;
    uint64_t r = r0;
    while (n < nmax && r - r0 < ticks) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            asm volatile(FMA16 : X16 : "v"(a), "v"(b));
        }
        const uint64_t c = cyc();
        r = rt();
        const uint32_t h = hwid1();
        if (threadIdx.x == 0) {
            tr[n] = {c, (uint32_t) r, h};
        }
        ++n;
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (threadIdx.x == 0) {
        count[blockIdx.x] = n;
        count[gridDim.x + blockIdx.x] = (int) se;
        if (s == 1234.5f) {
            tr[0].hw = 0;
        }
    }
}

// Whole-card FMA load for the pair test: 64 groups of 512 threads (one per CU with 40 KiB LDS, CU mode).
struct GRec {
    uint32_t pl, blk;
    uint64_t r0, r1;
};
__global__ __launch_bounds__(512) void k_big(const float * k, float * sink, GRec * rec, int iters) {
    float k0 = k[0], k1 = k[1];
    asm volatile("" : "+v"(k0), "+v"(k1));
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = 0.5f + (float) ((threadIdx.x + i) & 63) * 1e-3f;
    }
    const uint32_t pl = placement();
    const uint64_t r0 = rt();
#pragma unroll 4
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            x[i] = fmaf(k0, k1, x[i]);
        }
    }
    const uint64_t r1 = rt();
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    if (s == -12345.0f) {
        sink[threadIdx.x] = s;
    }
    // Every wave records: the oldest wave keeps the vector unit and ends first, so wave 0 alone understates the
    // group's time (seen in a first version: 0.8 ms of a 4 ms launch).
    if ((threadIdx.x & 31) == 0) {
        rec[blockIdx.x * 16 + (threadIdx.x >> 5)] = {pl, blockIdx.x, r0, r1};
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Detector

struct Event {
    size_t first, n;
    double start_us;      // REALTIME of the step before the event, from the trace start
    double dur_us;        // REALTIME length of the disturbed steps minus their normal length
    double extra_cycles;  // cycles beyond the normal step count
    bool hw_changed;
};

struct Detect {
    double base_cyc = 0, base_rt = 0;
    std::vector<Event> ev;
    size_t clock_steps = 0;  // REALTIME long, cycles normal
};

Detect detect(const std::vector<Step> & s) {
    Detect d;
    if (s.size() < 3) {
        return d;
    }
    std::vector<double> dc, dr;
    dc.reserve(s.size());
    for (size_t i = 1; i < s.size(); ++i) {
        dc.push_back((double) (s[i].cyc - s[i - 1].cyc));
        dr.push_back((double) (uint32_t) (s[i].rt - s[i - 1].rt));
    }
    // Median from a subsample keeps this fast on 10^7 steps.
    std::vector<double> sc, sr;
    for (size_t i = 0; i < dc.size(); i += std::max<size_t>(1, dc.size() / 200000)) {
        sc.push_back(dc[i]);
        sr.push_back(dr[i]);
    }
    d.base_cyc = pct(sc, 0.5);
    d.base_rt = pct(sr, 0.5);
    const double lim = 1.5 * d.base_cyc + 64;
    const uint32_t r0 = s[0].rt;
    for (size_t i = 0; i < dc.size(); ++i) {
        const bool hw = s[i + 1].hw != s[i].hw;
        if (dc[i] > lim || hw) {
            if (!d.ev.empty() && d.ev.back().first + d.ev.back().n == i) {
                Event & e = d.ev.back();
                ++e.n;
                e.dur_us += (dr[i] - d.base_rt) / 100.0;
                e.extra_cycles += dc[i] - d.base_cyc;
                e.hw_changed = e.hw_changed || hw;
            } else {
                d.ev.push_back({i, 1, (double) (uint32_t) (s[i].rt - r0) / 100.0, (dr[i] - d.base_rt) / 100.0,
                                dc[i] - d.base_cyc, hw});
            }
        } else if (dr[i] > 1.5 * d.base_rt + 2) {
            ++d.clock_steps;
        }
    }
    return d;
}

// Shifts every step from index k on by `cycles` and the matching REALTIME, as if the wave lost that time.
void inject(std::vector<Step> & s, size_t k, uint64_t cycles, double mhz) {
    const uint32_t ticks = (uint32_t) (cycles / mhz * 100.0);
    for (size_t i = k; i < s.size(); ++i) {
        s[i].cyc += cycles;
        s[i].rt += ticks;
    }
}

std::vector<Step> synthetic(size_t n, uint64_t seed) {
    std::vector<Step> s(n);
    uint64_t c = 1000000, x = seed;
    double rtf = 5000;
    for (size_t i = 0; i < n; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        const uint64_t step = 290 + (x >> 60);  // 290..305 cycles: the measured step plus a little jitter
        c += step;
        rtf += step / 32.0;  // 3.2 GHz: 32 cycles per 10 ns tick
        s[i] = {c, (uint32_t) rtf, 0x2000u};
    }
    return s;
}

int mode_selftest(int argc, char ** argv) {
    const int ec = std::atoi(opt(argc, argv, "--expect-clean", "0"));
    const int ei = std::atoi(opt(argc, argv, "--expect-injected", "1"));
    std::vector<Step> s = synthetic(200000, 12345);
    const Detect a = detect(s);
    inject(s, 100000, 160000, 3200);  // 50 us lost
    const Detect b = detect(s);
    char t[200];
    const bool ok1 = (int) a.ev.size() == ec;
    std::snprintf(t, sizeof t, "clean synthetic trace (200000 steps): %zu events, expected %d", a.ev.size(), ec);
    verdict("sched_share detector_clean", ok1 ? "PASS" : "FAIL", t);
    const bool ok2 = (int) b.ev.size() == ei && (b.ev.empty() || std::fabs(b.ev[0].dur_us - 50.0) < 1.0);
    std::snprintf(t, sizeof t, "same trace with one 50 us gap injected: %zu events (%.1f us), expected %d", b.ev.size(),
                  b.ev.empty() ? 0.0 : b.ev[0].dur_us, ei);
    verdict("sched_share detector_injected", ok2 ? "PASS" : "FAIL", t);
    return ok1 && ok2 ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Host sampling of other processes' graphics-engine time on this card.

struct FdRef {
    int pid;
    std::string path, comm;
};

std::vector<FdRef> card_fds(const std::string & pdev) {
    std::vector<FdRef> out;
    const int self = getpid();
    DIR * proc = opendir("/proc");
    if (!proc) {
        return out;
    }
    while (dirent * e = readdir(proc)) {
        const int pid = std::atoi(e->d_name);
        if (pid <= 0 || pid == self) {
            continue;
        }
        const std::string fdi = std::string("/proc/") + e->d_name + "/fdinfo";
        DIR * d = opendir(fdi.c_str());
        if (!d) {
            continue;
        }
        std::set<std::string> clients;
        while (dirent * f = readdir(d)) {
            if (f->d_name[0] == '.') {
                continue;
            }
            const std::string p = fdi + "/" + f->d_name;
            const std::string txt = slurp(p);
            if (txt.find("drm-pdev:\t" + pdev) == std::string::npos) {
                continue;
            }
            const size_t c = txt.find("drm-client-id:");
            const std::string id = c == std::string::npos ? p : txt.substr(c, txt.find('\n', c) - c);
            if (clients.insert(id).second) {  // one fd per DRM client, or its time is counted twice
                std::string comm = slurp(std::string("/proc/") + e->d_name + "/comm");
                if (!comm.empty() && comm.back() == '\n') {
                    comm.pop_back();
                }
                out.push_back({pid, p, comm});
            }
        }
        closedir(d);
    }
    closedir(proc);
    return out;
}

double gfx_ns(const std::string & path) {
    const std::string t = slurp(path);
    const size_t c = t.find("drm-engine-gfx:");
    if (c == std::string::npos) {
        return 0;
    }
    return std::atof(t.c_str() + c + 15);
}

struct HostRow {
    double t;  // host seconds since the kernel's first REALTIME stamp was seen
    int busy;
    std::map<std::string, double> gfx;  // per command name, cumulative ns
};

int mode_spin(int argc, char ** argv) {
    const double secs = std::atof(opt(argc, argv, "--seconds", "4"));
    const std::string out = opt(argc, argv, "--out", "spin");
    const std::string pdev = opt(argc, argv, "--pdev", "0000:03:00.0");
    const int nmax = (int) std::min(40e6, secs * 7e6);
    float * d_in;
    const float hin[3] = {0.5f, 0.9999f, 1e-4f};
    CHECK(hipMalloc(&d_in, sizeof hin));
    CHECK(hipMemcpy(d_in, hin, sizeof hin, hipMemcpyHostToDevice));
    Step * d_tr;
    int * d_n;
    CHECK(hipMalloc(&d_tr, (size_t) nmax * sizeof(Step)));
    CHECK(hipMalloc(&d_n, 2 * sizeof(int)));
    uint64_t * flag;
    CHECK(hipHostMalloc(&flag, 8, hipHostMallocCoherent));
    *flag = 0;
    // Warm-up launch (code load, clock up).
    hipLaunchKernelGGL(k_spin, dim3(1), dim3(32), 0, 0, d_in, d_tr, nmax, (uint64_t) 2000000, flag, d_n);
    CHECK(hipDeviceSynchronize());
    *flag = 0;
    const std::vector<FdRef> fds = card_fds(pdev);
    std::vector<HostRow> host;
    std::atomic<bool> run{true};
    std::atomic<bool> seen{false};
    std::chrono::steady_clock::time_point t_seen;
    hipLaunchKernelGGL(k_spin, dim3(1), dim3(32), 0, 0, d_in, d_tr, nmax, (uint64_t) (secs * 1e8), flag, d_n);
    CHECK(hipGetLastError());
    while (*(volatile uint64_t *) flag == 0) {
    }
    t_seen = std::chrono::steady_clock::now();
    const uint64_t rt_seen = *flag;
    {
        // Which power profile the driver selected while this ROCm queue is busy (KFD is expected to pick COMPUTE).
        std::ifstream pf(paths().drm + "/pp_power_profile_mode");
        std::string line, sel = "?";
        while (std::getline(pf, line)) {
            if (line.find('*') != std::string::npos) {
                sel = line;
            }
        }
        std::printf("RESULT spin power_profile_selected_during_kernel=\"%s\" perf_level=%s\n", sel.c_str(),
                    slurp(paths().drm + "/power_dpm_force_performance_level").c_str());
    }
    std::thread th([&] {
        int k = 0;
        while (run) {
            HostRow r;
            r.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_seen).count();
            r.busy = (int) read_num(paths().drm + "/gpu_busy_percent", 1);
            for (const FdRef & f : fds) {
                r.gfx[f.comm] += gfx_ns(f.path);
            }
            host.push_back(r);
            ++k;
            std::this_thread::sleep_until(t_seen + std::chrono::milliseconds(20 * k));
        }
    });
    CHECK(hipDeviceSynchronize());
    const double t_end = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_seen).count();
    run = false;
    th.join();
    (void) seen;
    int n = 0;
    CHECK(hipMemcpy(&n, d_n, sizeof n, hipMemcpyDeviceToHost));
    std::vector<Step> s(n);
    CHECK(hipMemcpy(s.data(), d_tr, (size_t) n * sizeof(Step), hipMemcpyDeviceToHost));
    const Detect d = detect(s);
    const double span_s = (double) (uint32_t) (s.back().rt - s.front().rt) / 1e8;
    // Clock over the whole trace (the counter keeps counting through gaps); one step's REALTIME is too coarse.
    const double mhz = (double) (s.back().cyc - s.front().cyc) / (span_s * 1e6);
    std::printf("RESULT spin steps=%d span_s=%.3f step_cycles_median=%.0f step_us_median=%.3f clock_MHz=%.0f "
                "clock_change_steps=%zu\n",
                n, span_s, d.base_cyc, d.base_rt / 100.0, mhz, d.clock_steps);
    std::vector<double> durs, extra;
    size_t hwch = 0, cyc_ran = 0;
    for (const Event & e : d.ev) {
        durs.push_back(e.dur_us);
        extra.push_back(e.extra_cycles);
        hwch += e.hw_changed;
        // Did the cycle counter keep counting through the gap at the shader clock?
        if (std::fabs(e.extra_cycles / (std::max(e.dur_us, 0.01) * mhz) - 1.0) < 0.1) {
            ++cyc_ran;
        }
    }
    double lost = 0;
    for (double x : durs) {
        lost += x;
    }
    std::printf("RESULT spin events=%zu per_s=%.1f lost_us_total=%.1f lost_fraction=%.5f hw_id_changed=%zu "
                "counter_ran_through_gap(within 10%%)=%zu\n",
                d.ev.size(), d.ev.size() / span_s, lost, lost * 1e-6 / span_s, hwch, cyc_ran);
    std::printf("RESULT spin event_us %s\n", dist(durs, "%.2f").c_str());
    std::printf("RESULT spin event_extra_cycles %s\n", dist(extra, "%.0f").c_str());
    // Gaps between event starts (period of the interruptions).
    std::vector<double> period;
    for (size_t i = 1; i < d.ev.size(); ++i) {
        period.push_back(d.ev[i].start_us - d.ev[i - 1].start_us);
    }
    std::printf("RESULT spin event_spacing_us %s\n", dist(period, "%.1f").c_str());
    // Duration histogram.
    const double edges[] = {0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1e9};
    for (int b = 0; b + 1 < (int) (sizeof edges / sizeof edges[0]); ++b) {
        int c = 0;
        for (double x : durs) {
            c += x >= edges[b] && x < edges[b + 1];
        }
        std::printf("HIST spin event_us [%g, %g): %d\n", edges[b], edges[b + 1], c);
    }
    for (size_t i = 0; i < d.ev.size() && i < 40; ++i) {
        const Event & e = d.ev[i];
        std::printf("EVENT %zu at_us=%.1f steps=%zu dur_us=%.2f extra_cycles=%.0f hw_changed=%d hw_before=0x%x "
                    "hw_after=0x%x\n",
                    i, e.start_us, e.n, e.dur_us, e.extra_cycles, (int) e.hw_changed, s[e.first].hw,
                    s[e.first + e.n].hw);
    }
    // Per 20 ms host window: events and lost time, busy percent, other processes' graphics time.
    const uint32_t rt0 = s.front().rt;
    const double off = (double) (int32_t) (rt0 - (uint32_t) rt_seen) / 1e8;  // trace start after the flag
    // Second mapping from the end: completion seen on the host against the last step (launch latency aside).
    const double off_end = t_end - (double) (uint32_t) (s.back().rt - rt0) / 1e8;
    std::printf("RESULT spin time_map flag_to_first_step_s=%.6f host_end_minus_trace_span_s=%.6f\n", off, off_end);
    FILE * f = std::fopen((out + "-windows.csv").c_str(), "w");
    std::map<std::string, double> tot;
    std::vector<double> lost_w, busy_w;
    if (f) {
        std::fprintf(f, "t_s,busy_pct,events,lost_us,other_gfx_ms_in_window,top_process\n");
    }
    for (size_t k = 1; k < host.size(); ++k) {
        const double a = host[k - 1].t - off, b = host[k].t - off;
        int ne = 0;
        double lu = 0;
        for (const Event & e : d.ev) {
            const double t = e.start_us * 1e-6;
            if (t >= a && t < b) {
                ++ne;
                lu += e.dur_us;
            }
        }
        double other = 0, best = 0;
        std::string top = "-";
        for (auto & kv : host[k].gfx) {
            const double dd = (kv.second - host[k - 1].gfx[kv.first]) / 1e6;
            other += dd;
            tot[kv.first] += dd;
            if (dd > best) {
                best = dd;
                top = kv.first;
            }
        }
        lost_w.push_back(lu);
        busy_w.push_back(host[k].busy);
        if (f) {
            std::fprintf(f, "%.3f,%d,%d,%.1f,%.3f,%s\n", b, host[k].busy, ne, lu, other, top.c_str());
        }
    }
    if (f) {
        std::fclose(f);
    }
    std::vector<std::pair<double, std::string>> ranked;
    for (auto & kv : tot) {
        ranked.push_back({kv.second, kv.first});
    }
    std::sort(ranked.rbegin(), ranked.rend());
    std::printf("RESULT spin other_processes_gfx_ms_during_trace:");
    for (size_t i = 0; i < ranked.size() && i < 8; ++i) {
        std::printf(" %s=%.1f", ranked[i].second.c_str(), ranked[i].first);
    }
    std::printf(" (DRM clients on the card: %zu)\n", fds.size());
    std::printf("RESULT spin gpu_busy_pct %s\n", dist(busy_w, "%.0f").c_str());
    // Correlation between lost time and busy percent per window.
    {
        double ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
        const size_t m = lost_w.size();
        for (size_t i = 0; i < m; ++i) {
            ma += lost_w[i];
            mb += busy_w[i];
        }
        ma /= std::max<size_t>(m, 1);
        mb /= std::max<size_t>(m, 1);
        for (size_t i = 0; i < m; ++i) {
            sab += (lost_w[i] - ma) * (busy_w[i] - mb);
            saa += (lost_w[i] - ma) * (lost_w[i] - ma);
            sbb += (busy_w[i] - mb) * (busy_w[i] - mb);
        }
        std::printf("RESULT spin correlation(lost_us per 20 ms window, gpu_busy_pct)=%.3f windows=%zu\n",
                    sab / std::sqrt(std::max(saa * sbb, 1e-30)), m);
    }
    // Self-check on the real trace: find a clean stretch of 20000 steps, inject one 50 us gap, expect exactly one.
    bool ok = n > 100000;
    size_t start = 0;
    for (const Event & e : d.ev) {
        if (e.first > start + 20002) {
            break;
        }
        start = e.first + e.n + 1;
    }
    if (start + 20002 <= s.size()) {
        std::vector<Step> w(s.begin() + start, s.begin() + start + 20000);
        const Detect c = detect(w);
        inject(w, 10000, (uint64_t) (50 * mhz), mhz);
        const Detect g = detect(w);
        char t[200];
        const bool c_ok = c.ev.empty() && g.ev.size() == 1 && std::fabs(g.ev[0].dur_us - 50) < 2;
        std::snprintf(t, sizeof t, "clean stretch of the real trace: %zu events (expected 0); with a 50 us gap injected: %zu "
                      "events (expected 1, %.1f us)", c.ev.size(), g.ev.size(), g.ev.empty() ? 0.0 : g.ev[0].dur_us);
        verdict("sched_share detector_real", c_ok ? "PASS" : "FAIL", t);
        ok = ok && c_ok;
    } else {
        verdict("sched_share detector_real", "FAIL", "no clean stretch of 20000 steps found");
        ok = false;
    }
    FILE * g = std::fopen((out + "-events.csv").c_str(), "w");
    if (g) {
        std::fprintf(g, "at_us,steps,dur_us,extra_cycles,hw_changed\n");
        for (const Event & e : d.ev) {
            std::fprintf(g, "%.2f,%zu,%.2f,%.0f,%d\n", e.start_us, e.n, e.dur_us, e.extra_cycles, (int) e.hw_changed);
        }
        std::fclose(g);
    }
    verdict("sched_share spin", n > 100000 ? "PASS" : "FAIL", "trace recorded");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
// pair

void sleep_until_ms(long long ms) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const long long now = (long long) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    if (ms > now) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms - now));
    }
}

int mode_pair(int argc, char ** argv) {
    const std::string role = opt(argc, argv, "--role", "A");
    const std::string kind = opt(argc, argv, "--kind", "big");
    const long long start = std::atoll(opt(argc, argv, "--start-ms", "0"));
    const double secs = std::atof(opt(argc, argv, "--seconds", "1.5"));
    const std::string out = opt(argc, argv, "--out", "pair");
    float * d_k;
    const float hk[3] = {0.9999f, 1e-4f, 0};
    CHECK(hipMalloc(&d_k, sizeof hk));
    CHECK(hipMemcpy(d_k, hk, sizeof hk, hipMemcpyHostToDevice));
    Sampler smp(50);
    if (kind == "big" || kind == "part") {
        // part: 16 groups (16 CUs), so two processes could fit on the card side by side.
        const int G = kind == "big" ? 64 : 16;
        float * d_sink;
        GRec * d_rec;
        CHECK(hipMalloc(&d_sink, 4096));
        const int maxl = 4000;
        CHECK(hipMalloc(&d_rec, (size_t) maxl * 1024 * sizeof(GRec)));
        const int iters = 100000;  // about 2 ms alone at 3.2 GHz with 8 waves per SIMD
        hipLaunchKernelGGL(k_big, dim3(G), dim3(512), 40 * 1024, 0, d_k, d_sink, d_rec, 1000);
        CHECK(hipDeviceSynchronize());
        if (start == 0 && !gate(60.0)) {  // alone (no partner to keep in step with): wait for the card to cool
            return 1;
        }
        sleep_until_ms(start);
        smp.start();
        const auto t0 = std::chrono::steady_clock::now();
        int l = 0;
        // Launches are queued a few at a time so the process keeps the GPU fed without waiting on each.
        while (l < maxl && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs &&
               !smp.hot) {
            for (int q = 0; q < 4 && l < maxl; ++q, ++l) {
                hipLaunchKernelGGL(k_big, dim3(G), dim3(512), 40 * 1024, 0, d_k, d_sink, d_rec + (size_t) l * 1024, iters);
            }
            CHECK(hipDeviceSynchronize());
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        smp.stop();
        std::vector<GRec> w((size_t) l * 1024);
        CHECK(hipMemcpy(w.data(), d_rec, w.size() * sizeof(GRec), hipMemcpyDeviceToHost));
        std::vector<GRec> h((size_t) l * G);
        for (int li = 0; li < l; ++li) {
            for (int g = 0; g < G; ++g) {
                const GRec * src = &w[(size_t) li * 1024 + g * 16];
                GRec & d = h[(size_t) li * G + g];
                d = src[0];
                for (int k = 1; k < 16; ++k) {
                    d.r0 = std::min(d.r0, src[k].r0);
                    d.r1 = std::max(d.r1, src[k].r1);
                }
            }
        }
        FILE * f = std::fopen((out + "-" + role + ".csv").c_str(), "w");
        if (!f) {
            return 2;
        }
        std::fprintf(f, "role,kind,launch,group,cu,r0,r1\n");
        for (int i = 0; i < l; ++i) {
            for (int g = 0; g < G; ++g) {
                const GRec & r = h[(size_t) i * G + g];
                std::fprintf(f, "%s,big,%d,%d,%u,%llu,%llu\n", role.c_str(), i, g, pl_cu(r.pl),
                             (unsigned long long) r.r0, (unsigned long long) r.r1);
            }
        }
        std::fclose(f);
        std::printf("RESULT pair %s %s groups=%d launches=%d wall_s=%.3f ms_per_launch=%.3f hot_stop=%d\n", role.c_str(), kind.c_str(), G, l, wall,
                    wall * 1e3 / std::max(l, 1), (int) smp.hot.load());
        verdict(("sched_share pair_" + role).c_str(), l > 0 ? "PASS" : "FAIL", "launches recorded");
        std::printf("HOTTEST %.0f C\n", hottest().load());
        return l > 0 ? 0 : 1;
    }
    // small: the spin trace, written as steps of disturbed events plus every 1000th step.
    const int nmax = (int) (secs * 7e6);
    Step * d_tr;
    int * d_n;
    uint64_t * flag;
    CHECK(hipMalloc(&d_tr, (size_t) nmax * sizeof(Step)));
    CHECK(hipMalloc(&d_n, 2 * sizeof(int)));
    CHECK(hipHostMalloc(&flag, 8, hipHostMallocCoherent));
    hipLaunchKernelGGL(k_spin, dim3(1), dim3(32), 0, 0, d_k, d_tr, nmax, (uint64_t) 100000, flag, d_n);
    CHECK(hipDeviceSynchronize());
    sleep_until_ms(start);
    hipLaunchKernelGGL(k_spin, dim3(1), dim3(32), 0, 0, d_k, d_tr, nmax, (uint64_t) (secs * 1e8), flag, d_n);
    CHECK(hipDeviceSynchronize());
    int n = 0;
    CHECK(hipMemcpy(&n, d_n, sizeof n, hipMemcpyDeviceToHost));
    std::vector<Step> s(n);
    CHECK(hipMemcpy(s.data(), d_tr, (size_t) n * sizeof(Step), hipMemcpyDeviceToHost));
    const Detect d = detect(s);
    FILE * f = std::fopen((out + "-" + role + ".csv").c_str(), "w");
    if (!f) {
        return 2;
    }
    // REALTIME is stored as its low 32 bits; the big process's files hold full values: the script compares low bits.
    std::fprintf(f, "role,kind,step,cycles,rt_lo,hw,disturbed\n");
    std::set<size_t> dist_steps;
    for (const Event & e : d.ev) {
        for (size_t i = e.first; i <= e.first + e.n; ++i) {
            dist_steps.insert(i);
        }
    }
    for (size_t i = 0; i < s.size(); ++i) {
        const bool dd = dist_steps.count(i) > 0;
        if (dd || i % 1000 == 0) {
            std::fprintf(f, "%s,small,%zu,%llu,%u,%u,%d\n", role.c_str(), i, (unsigned long long) s[i].cyc, s[i].rt,
                         s[i].hw, (int) dd);
        }
    }
    std::fclose(f);
    double lost = 0;
    std::vector<double> durs;
    for (const Event & e : d.ev) {
        lost += e.dur_us;
        durs.push_back(e.dur_us);
    }
    const double span_s = n > 1 ? (double) (uint32_t) (s.back().rt - s.front().rt) / 1e8 : 0;
    std::printf("RESULT pair %s small steps=%d span_s=%.3f step_cycles=%.0f events=%zu lost_us=%.1f lost_fraction=%.4f "
                "event_us %s\n",
                role.c_str(), n, span_s, d.base_cyc, d.ev.size(), lost, lost * 1e-6 / std::max(span_s, 1e-9),
                dist(durs, "%.2f").c_str());
    verdict(("sched_share pair_" + role).c_str(), n > 1000 ? "PASS" : "FAIL", "trace recorded");
    return n > 1000 ? 0 : 1;
}

// multi: 16 single-wave groups spin at once (spread over the shader engines by the dispatcher); are their
// interruptions simultaneous, within one shader engine or across the card?
int mode_multi(int argc, char ** argv) {
    const double secs = std::atof(opt(argc, argv, "--seconds", "0.5"));
    const int G = 16;
    const int nmax = (int) (secs * 7e6);
    float * d_in;
    const float hin[3] = {0.5f, 0.9999f, 1e-4f};
    CHECK(hipMalloc(&d_in, sizeof hin));
    CHECK(hipMemcpy(d_in, hin, sizeof hin, hipMemcpyHostToDevice));
    Step * d_tr;
    int * d_n;
    uint64_t * flag;
    CHECK(hipMalloc(&d_tr, (size_t) G * nmax * sizeof(Step)));
    CHECK(hipMalloc(&d_n, 2 * G * sizeof(int)));
    CHECK(hipHostMalloc(&flag, 8, hipHostMallocCoherent));
    hipLaunchKernelGGL(k_spin, dim3(G), dim3(32), 0, 0, d_in, d_tr, nmax, (uint64_t) 100000, flag, d_n);
    CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(k_spin, dim3(G), dim3(32), 0, 0, d_in, d_tr, nmax, (uint64_t) (secs * 1e8), flag, d_n);
    CHECK(hipDeviceSynchronize());
    std::vector<int> cnt(2 * G);
    CHECK(hipMemcpy(cnt.data(), d_n, cnt.size() * sizeof(int), hipMemcpyDeviceToHost));
    std::vector<std::vector<uint32_t>> ev(G);
    std::vector<uint32_t> hw(G);
    bool se_bits_match = true;
    std::vector<Step> s;
    for (int g = 0; g < G; ++g) {
        s.resize(cnt[g]);
        CHECK(hipMemcpy(s.data(), d_tr + (size_t) g * nmax, (size_t) cnt[g] * sizeof(Step), hipMemcpyDeviceToHost));
        const Detect d = detect(s);
        for (const Event & e : d.ev) {
            ev[g].push_back(s[e.first].rt);
        }
        hw[g] = s.empty() ? 0 : s[0].hw;
        se_bits_match = se_bits_match && (int) ((hw[g] >> 18) & 3) == cnt[G + g];
        std::printf("RESULT multi wave=%d se=%d hw=0x%x steps=%d events=%zu\n", g, cnt[G + g], hw[g], cnt[g], ev[g].size());
    }
    // For each pair of waves: share of the first wave's events within 1 us (100 ticks) of an event of the second.
    std::vector<double> same_se, diff_se;
    for (int a = 0; a < G; ++a) {
        for (int b = 0; b < G; ++b) {
            if (a == b || ev[a].empty() || ev[b].empty()) {
                continue;
            }
            int hit = 0;
            for (uint32_t t : ev[a]) {
                auto it = std::lower_bound(ev[b].begin(), ev[b].end(), t - 100u);
                hit += it != ev[b].end() && (uint32_t) (*it - t + 100u) <= 200u;
            }
            const double f = (double) hit / ev[a].size();
            (cnt[G + a] == cnt[G + b] ? same_se : diff_se).push_back(f);
        }
    }
    std::printf("RESULT multi coincidence_within_1us same_SE_pairs %s\n", dist(same_se, "%.3f").c_str());
    std::printf("RESULT multi coincidence_within_1us different_SE_pairs %s\n", dist(diff_se, "%.3f").c_str());
    std::printf("RESULT multi HW_ID1_bits_19_18_equal_SE_from_message_for_all_waves=%s\n", se_bits_match ? "yes" : "no");
    verdict("sched_share multi", "PASS", "traces recorded");
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// preempt (leftovers C, SC-11)

struct HRec {
    uint32_t pl, blk;
    uint64_t r0, r1;
    uint32_t maxgap, pad;
};

// Waves sleep (near-idle power) until REALTIME passes r0 + ticks; the 40 KiB of dynamic LDS is what keeps a second
// group of this shape off the CU. The sleep between REALTIME reads is long (s_sleep 127, about 8,000 cycles): with
// s_sleep 4 one wave's polling held the other wave of its group at its first REALTIME read for the whole 1 ms.
__global__ __launch_bounds__(64) void k_hold(HRec * rec, uint64_t ticks, uint32_t inject) {
    const uint32_t pl = placement();
    const uint64_t r0 = rt();
    uint64_t r = r0, prev = r0, gap = 0;
    bool injected = inject == 0 || blockIdx.x != 0 || threadIdx.x >= 32;
    while (r - r0 < ticks) {
        asm volatile("s_sleep 127" ::: "memory");
        if (!injected && r - r0 >= ticks / 2) {
            for (uint32_t k = 0; k < inject; ++k) {
                asm volatile("s_sleep 127" ::: "memory");
            }
            injected = true;
        }
        r = rt();
        gap = r - prev > gap ? r - prev : gap;
        prev = r;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[blockIdx.x * 2 + (threadIdx.x >> 5)] = {pl, blockIdx.x, r0, r, (uint32_t) gap, 0};
    }
}

// The same hold with busy waves: 2,048 dependent FMAs (about 10,000 cycles, the spacing of s_sleep 127) between two
// REALTIME reads. The chain's value goes into the record so the compiler keeps it.
__global__ __launch_bounds__(64) void k_hold_fma(HRec * rec, uint64_t ticks, float a, float b) {
    const uint32_t pl = placement();
    const uint64_t r0 = rt();
    uint64_t r = r0, prev = r0, gap = 0;
    float x = (float) threadIdx.x * 1e-3f;
    while (r - r0 < ticks) {
        for (int i = 0; i < 2048; ++i) {
            x = __builtin_fmaf(x, a, b);
        }
        r = rt();
        gap = r - prev > gap ? r - prev : gap;
        prev = r;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[blockIdx.x * 2 + (threadIdx.x >> 5)] = {pl, blockIdx.x, r0, r, (uint32_t) gap, __float_as_uint(x)};
    }
}

// Writes REALTIME into host memory; the host notes its own clock when it sees the value (host time = REALTIME x 10 ns
// + offset; the smallest offset of several tries is the best, as the write takes a moment to arrive).
__global__ void k_stamp(volatile uint64_t * f) {
    if (threadIdx.x == 0) {
        *f = rt();
        __threadfence_system();
    }
}

long long host_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Returns host_ns - REALTIME * 10 (ns), the minimum over `n` tries.
long long rt_offset(volatile uint64_t * flag, int n) {
    long long best = 0;
    for (int i = 0; i < n; ++i) {
        *flag = 0;
        hipLaunchKernelGGL(k_stamp, dim3(1), dim3(32), 0, 0, (uint64_t *) flag);
        uint64_t v;
        while ((v = *flag) == 0) {
        }
        const long long h = host_ns();
        CHECK(hipDeviceSynchronize());
        const long long off = h - (long long) v * 10;
        if (i == 0 || off < best) {
            best = off;
        }
    }
    return best;
}

int mode_preempt(int argc, char ** argv) {
    const std::string role = opt(argc, argv, "--role", "A");
    const long long start = std::atoll(opt(argc, argv, "--start-ms", "0"));
    const std::string out = opt(argc, argv, "--out", "preempt");
    const std::string prio = opt(argc, argv, "--prio", "normal");
    const int trials = std::atoi(opt(argc, argv, "--trials", "20"));
    std::vector<int> tas;
    {
        std::stringstream ss(opt(argc, argv, "--ta-ms", "4,16,64,180"));
        std::string t;
        while (std::getline(ss, t, ',')) {
            tas.push_back(std::atoi(t.c_str()));
        }
    }
    const bool doA = role == "A" || role == "AB", doB = role == "B" || role == "AB";
    int lo = 0, hi = 0;
    CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
    hipStream_t sa, sb;
    // AB: A on the lowest priority, B on the highest. Two processes: B's stream as asked, A's the default priority.
    CHECK(hipStreamCreateWithPriority(&sa, hipStreamNonBlocking, role == "AB" ? lo : 0));
    CHECK(hipStreamCreateWithPriority(&sb, hipStreamNonBlocking, (prio == "high" || role == "AB") ? hi : 0));
    std::printf("RESULT preempt role=%s stream_priority_range least=%d greatest=%d B_stream=%s\n", role.c_str(), lo, hi,
                (prio == "high" || role == "AB") ? "greatest" : "default");
    const int G = 64, WPG = 2;
    const size_t lds = 40 * 1024;
    // A's shape and hold (gaming round P3); B keeps G groups of 40 KiB sleeping for 1 ms.
    const int GA = std::atoi(opt(argc, argv, "--a-groups", "64"));
    const size_t lds_a = (size_t) std::atoi(opt(argc, argv, "--a-lds-kib", "40")) * 1024;
    const bool fma_a = std::string(opt(argc, argv, "--a-hold", "sleep")) == "fma";
    const uint32_t inject = (uint32_t) std::atoi(opt(argc, argv, "--a-inject-sleeps", "0"));
    std::printf("RESULT preempt A_shape groups=%d waves_per_group=%d lds_KiB=%zu hold=%s inject_sleeps=%u; B_shape groups=%d "
                "lds_KiB=%zu hold=sleep 1 ms\n", GA, WPG, lds_a / 1024, fma_a ? "fma" : "sleep", inject, G, lds / 1024);
    auto launch_a = [&](HRec * rec, uint64_t ticks) {
        if (fma_a) {
            hipLaunchKernelGGL(k_hold_fma, dim3(GA), dim3(64), lds_a, sa, rec, ticks, 0.999f, 0.001f);
        } else {
            hipLaunchKernelGGL(k_hold, dim3(GA), dim3(64), lds_a, sa, rec, ticks, inject);
        }
    };
    const int ntr = (int) tas.size() * trials;
    HRec *d_a, *d_b;
    CHECK(hipMalloc(&d_a, (size_t) ntr * G * WPG * sizeof(HRec)));
    CHECK(hipMalloc(&d_b, (size_t) ntr * G * WPG * sizeof(HRec)));
    CHECK(hipMemset(d_a, 0, (size_t) ntr * G * WPG * sizeof(HRec)));
    CHECK(hipMemset(d_b, 0, (size_t) ntr * G * WPG * sizeof(HRec)));
    uint64_t * flag;
    CHECK(hipHostMalloc(&flag, 8, hipHostMallocCoherent));
    // Warm-up: code load and queue creation on both streams.
    launch_a(d_a, (uint64_t) 1000);
    hipLaunchKernelGGL(k_hold, dim3(G), dim3(64), lds, sb, d_b, (uint64_t) 1000, 0u);
    CHECK(hipDeviceSynchronize());
    const long long off0 = rt_offset(flag, 20);
    const long long hm0 = host_ns();
    Sampler smp(50);
    smp.start();
    const long long start_ms = start > 0 ? start : hm0 / 1000000 + 500;
    sleep_until_ms(start_ms);
    // Trial k starts at start + sum of earlier periods; period = T_A + 40 ms. B launches 1 ms after A's launch time.
    std::vector<long long> launchA(ntr, 0), launchB(ntr, 0);
    std::vector<int> taof(ntr), ctrl(ntr);
    long long t = start_ms * 1000000LL;
    int k = 0;
    bool hot = false;
    for (size_t ti = 0; ti < tas.size(); ++ti) {
        for (int r = 0; r < trials; ++r, ++k) {
            taof[k] = tas[ti];
            ctrl[k] = r % 5 == 4;  // every fifth trial: A launches nothing (B alone)
            while (host_ns() < t) {
            }
            if (smp.hot) {
                hot = true;
            }
            if (doA && !ctrl[k] && !hot) {
                launchA[k] = host_ns();
                launch_a(d_a + (size_t) k * G * WPG, (uint64_t) tas[ti] * 100000);
            }
            while (host_ns() < t + 1000000LL) {
            }
            if (doB && !hot) {
                launchB[k] = host_ns();
                hipLaunchKernelGGL(k_hold, dim3(G), dim3(64), lds, sb, d_b + (size_t) k * G * WPG, (uint64_t) 100000, 0u);
            }
            CHECK(hipDeviceSynchronize());
            t += (long long) (tas[ti] + 40) * 1000000LL;
        }
    }
    smp.stop();
    const long long off1 = rt_offset(flag, 20);
    const long long hm1 = host_ns();
    std::printf("RESULT preempt clock_map offset_start_ns=%lld offset_end_ns=%lld drift_ns=%lld\n", off0, off1, off1 - off0);
    std::vector<HRec> ha((size_t) ntr * G * WPG), hb((size_t) ntr * G * WPG);
    CHECK(hipMemcpy(ha.data(), d_a, ha.size() * sizeof(HRec), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hb.data(), d_b, hb.size() * sizeof(HRec), hipMemcpyDeviceToHost));
    // The file holds one row per trial and kernel: first start, last end, longest gap of any wave, distinct CUs,
    // and the launch time mapped to REALTIME (offset interpolated between the start and end mappings).
    FILE * f = std::fopen((out + "-" + role + ".csv").c_str(), "w");
    if (!f) {
        return 2;
    }
    std::fprintf(f, "role,kernel,trial,ta_ms,control,launch_rt,first_start_rt,last_start_rt,last_end_rt,max_gap_ticks,cus,"
                    "waves\n");
    int bad = 0;
    for (int i = 0; i < ntr; ++i) {
        for (int which = 0; which < 2; ++which) {
            const bool isA = which == 0;
            if ((isA && (!doA || ctrl[i] || !launchA[i])) || (!isA && (!doB || !launchB[i]))) {
                continue;
            }
            const HRec * rr = (isA ? ha.data() : hb.data()) + (size_t) i * G * WPG;
            uint64_t fs = ~0ull, ls = 0, le = 0;
            uint32_t mg = 0;
            std::set<uint32_t> cus;
            int waves = 0;
            for (int w = 0; w < G * WPG; ++w) {
                if (rr[w].r1 == 0) {
                    continue;
                }
                ++waves;
                fs = std::min(fs, rr[w].r0);
                ls = std::max(ls, rr[w].r0);
                le = std::max(le, rr[w].r1);
                mg = std::max(mg, rr[w].maxgap);
                cus.insert(pl_cu(rr[w].pl));
            }
            // Every wave must have recorded. The number of distinct CUs is reported (file column cus), not judged:
            // in the gaming round's development run, B beside a 16-group A put two of its 40 KiB groups on one CU.
            const int gk = isA ? GA : G;
            bad += waves != gk * WPG;
            const long long lh = isA ? launchA[i] : launchB[i];
            const double frac = (double) (lh - hm0) / (double) std::max(1LL, hm1 - hm0);
            const long long off = off0 + (long long) ((off1 - off0) * frac);
            const long long lrt = (lh - off) / 10;
            std::fprintf(f, "%s,%s,%d,%d,%d,%lld,%llu,%llu,%llu,%u,%zu,%d\n", role.c_str(), isA ? "A" : "B", i, taof[i],
                         ctrl[i], lrt, (unsigned long long) fs, (unsigned long long) ls, (unsigned long long) le, mg,
                         cus.size(), waves);
        }
    }
    std::fclose(f);
    // --dump K: every wave of trial K (start, end and longest gap relative to the kernel's first start, placement).
    const int dump = std::atoi(opt(argc, argv, "--dump", "-1"));
    for (int which = 0; which < 2 && dump >= 0 && dump < ntr; ++which) {
        const HRec * rr = (which == 0 ? ha.data() : hb.data()) + (size_t) dump * G * WPG;
        uint64_t fs = ~0ull;
        for (int w = 0; w < G * WPG; ++w) {
            if (rr[w].r1) {
                fs = std::min(fs, rr[w].r0);
            }
        }
        for (int w = 0; w < G * WPG; ++w) {
            if (rr[w].r1) {
                std::printf("WAVE %s trial=%d wave=%d grp=%u cu=0x%x simd=%d slot=%d start_us=%.2f end_us=%.2f gap_us=%.2f\n",
                            which == 0 ? "A" : "B", dump, w, rr[w].blk, pl_cu(rr[w].pl), pl_simdincu(rr[w].pl),
                            pl_slot(rr[w].pl), (rr[w].r0 - fs) / 100.0, (rr[w].r1 - fs) / 100.0, rr[w].maxgap / 100.0);
            }
        }
    }
    std::printf("RESULT preempt role=%s trials=%d kernels_with_missing_waves_or_cus=%d hot_stop=%d\n", role.c_str(), ntr,
                bad, (int) hot);
    std::printf("HOTTEST %.0f C\n", hottest().load());
    const std::string vtext = "every launched kernel recorded all its waves (A: " + std::to_string(GA * WPG) +
                              ", B: 128)";
    verdict(("sched_share preempt_" + role).c_str(), bad == 0 && !hot ? "PASS" : "FAIL", vtext.c_str());
    return bad == 0 && !hot ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sched_share <spin|selftest|pair|gate|preempt> [options]\n");
        return 2;
    }
    const std::string m = argv[1];
    if (m == "selftest") {
        return mode_selftest(argc, argv);
    }
    if (m == "gate") {
        return gate(60.0) ? 0 : 1;
    }
    require_arch();
    int rc = 2;
    if (m == "spin") {
        rc = mode_spin(argc, argv);
    } else if (m == "pair") {
        rc = mode_pair(argc, argv);
    } else if (m == "multi") {
        rc = mode_multi(argc, argv);
    } else if (m == "preempt") {
        rc = mode_preempt(argc, argv);
    }
    return rc;
}
