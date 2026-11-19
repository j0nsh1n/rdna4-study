// Round 3, Q5: are the sub-microsecond kernel overlaps that rocprofv3 reported between neighbouring llama.cpp dispatches real
// or a timestamp artifact? An independent clock decides: every wave of every kernel reads the 100 MHz REALTIME counter
// (s_memrealtime, 10 ns per tick) as its first action and, after its stores completed, as its last action.
//
//   kernel A  writes a region of a buffer (tag = pair number), fence, last stamp
//   kernel B  (same stream, launched right after A) reads the region A's LAST blocks wrote (reversed block order) and checks every
//             value against the tag; a mismatch means B read before A's stores landed (or A of the next pair overwrote early)
// Sequence A0 B0 A1 B1 ...: (A_i, B_i) is a dependent pair (like a KV-cache write followed by the attention that reads it),
// (B_i, A_(i+1)) an independent neighbour pair. Per kernel the host reduces the waves' stamps to first_min (the earliest wave start),
// last_max (the last wave's end) and writes them to the dump file; scripts/llm3_pairs.py joins that with a rocprofv3 kernel trace.
//
// Usage: pair_overlap <mode> <KiB per kernel> <pairs> [dump.csv] [break]
//   mode plain      n pairs launched back to back on one stream
//   mode graph      the same, captured into a HIP graph of 100 pairs (200 kernel nodes), launched pairs/100 times (host sync between launches)
//   mode hostsync   NEGATIVE control: a host synchronise between A and B (and B and the next A); no overlap on any clock
//   mode twostreams POSITIVE control: two independent spin kernels of 200 us on two streams, started together; must overlap
//   mode clock      known answer for the counter: ticks per host second must be 100e6 (1 %)
// Checks (exit 1 if a VERDICT line says FAIL):
//  1. clock known answer, always run first: the REALTIME counter ticks 100e6 per second against the host's steady clock (1 %).
//  2. data check: every B reads every value A wrote; 0 mismatches. The argument "break" makes B expect the wrong tag: it must FAIL.
//  3. controls (modes hostsync and twostreams): VERDICT negative-control / positive-control PASS when the in-kernel clock shows
//     what it must (no overlap / overlap in every round).
//  4. the ISA of the two kernels (order of the first stamp, the stores, the wait and the last stamp) is checked by scripts/llm3_pair_isacheck.py.
// Resolution: one tick is 10 ns; pairs whose gap is within one tick (|gap| <= 1) are AMBIGUOUS and counted apart.
#include "harness.h"

#include <chrono>
#include <cmath>
#include <string>
#include <thread>

constexpr int BLOCK = 256;
constexpr int WAVES_PER_BLOCK = BLOCK / 32;

__global__ void pair_a(uint4 * __restrict__ buf, unsigned tag, int items, unsigned long long * __restrict__ first, unsigned long long * __restrict__ last) {
    const unsigned long long t0 = __builtin_readsteadycounter();
    const size_t base = (size_t) blockIdx.x * BLOCK * items;
    for (int j = 0; j < items; ++j) {
        const size_t idx = base + (size_t) j * BLOCK + threadIdx.x;
        buf[idx] = make_uint4(tag, (unsigned) idx, ~tag, (unsigned) idx ^ (tag * 2654435761u));
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");  // the stores of this wave are complete before the last stamp
    const unsigned long long t1 = __builtin_readsteadycounter();
    if ((threadIdx.x & 31) == 0) {
        const int w = blockIdx.x * WAVES_PER_BLOCK + (threadIdx.x >> 5);
        first[w] = t0;
        last[w] = t1;
    }
}

__global__ void pair_b(const uint4 * __restrict__ buf, unsigned tag, int items, unsigned * __restrict__ err, unsigned long long * __restrict__ first,
                       unsigned long long * __restrict__ last) {
    const unsigned long long t0 = __builtin_readsteadycounter();
    const size_t base = (size_t) (gridDim.x - 1 - blockIdx.x) * BLOCK * items;  // the region of the block A launched last, first
    unsigned bad = 0;
    for (int j = 0; j < items; ++j) {
        const size_t idx = base + (size_t) j * BLOCK + threadIdx.x;
        const uint4 v = buf[idx];
        bad += (v.x != tag) | (v.y != (unsigned) idx) | (v.z != ~tag) | (v.w != ((unsigned) idx ^ (tag * 2654435761u)));
    }
    if (bad) {
        atomicAdd(err, bad);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    const unsigned long long t1 = __builtin_readsteadycounter();
    if ((threadIdx.x & 31) == 0) {
        const int w = blockIdx.x * WAVES_PER_BLOCK + (threadIdx.x >> 5);
        first[w] = t0;
        last[w] = t1;
    }
}

// independent work for the positive control: spins on the counter for `ticks`, no memory traffic besides the stamps
__global__ void spin_k(unsigned long long ticks, unsigned long long * __restrict__ first, unsigned long long * __restrict__ last) {
    const unsigned long long t0 = __builtin_readsteadycounter();
    unsigned long long t = t0;
    while (t - t0 < ticks) {
        t = __builtin_readsteadycounter();
    }
    if ((threadIdx.x & 31) == 0) {
        const int w = blockIdx.x * WAVES_PER_BLOCK + (threadIdx.x >> 5);
        first[w] = t0;
        last[w] = t;
    }
}

__global__ void stamp_k(unsigned long long * out) {
    if (threadIdx.x == 0) {
        *out = __builtin_readsteadycounter();
    }
}

struct Agg {
    unsigned long long first_min, first_max, last_min, last_max;
};

static Agg reduce_waves(const unsigned long long * first, const unsigned long long * last, int waves) {
    Agg a{~0ull, 0, ~0ull, 0};
    for (int w = 0; w < waves; ++w) {
        a.first_min = std::min(a.first_min, first[w]);
        a.first_max = std::max(a.first_max, first[w]);
        a.last_min = std::min(a.last_min, last[w]);
        a.last_max = std::max(a.last_max, last[w]);
    }
    return a;
}

struct Gaps {
    long long ambiguous = 0, overlap = 0, ok = 0, n = 0;
    std::vector<double> gap_ns;  // first_min(next) - last_max(prev) in ns, all pairs
};

static void add_gap(Gaps & g, const Agg & prev, const Agg & next) {
    const long long gap = (long long) next.first_min - (long long) prev.last_max;  // ticks
    g.n++;
    if (gap < -1) {
        g.overlap++;
    } else if (gap <= 1) {
        g.ambiguous++;
    } else {
        g.ok++;
    }
    g.gap_ns.push_back(10.0 * gap);
}

static void report_gaps(const char * name, Gaps & g) {
    std::sort(g.gap_ns.begin(), g.gap_ns.end());
    auto q = [&](double p) { return g.gap_ns.empty() ? 0.0 : g.gap_ns[std::min(g.gap_ns.size() - 1, (size_t) (p * g.gap_ns.size()))]; };
    std::printf("RESULT %s pairs %lld inkernel_overlap %lld ambiguous_within_1_tick %lld separated %lld gap_ns_min %.0f p01 %.0f median %.0f p99 %.0f max %.0f\n", name, g.n,
                g.overlap, g.ambiguous, g.ok, q(0), q(0.01), q(0.5), q(0.99), g.gap_ns.empty() ? 0.0 : g.gap_ns.back());
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 4) {
        std::fprintf(stderr, "usage: pair_overlap <plain|graph|hostsync|twostreams|clock> <KiB per kernel> <pairs> [dump.csv] [break]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const size_t kib = std::strtoull(argv[2], nullptr, 10);
    const int pairs = std::atoi(argv[3]);
    const char * dump = (argc > 4 && std::strcmp(argv[4], "-") != 0 && std::strcmp(argv[4], "break") != 0) ? argv[4] : nullptr;
    bool do_break = false;
    for (int i = 4; i < argc; ++i) {
        do_break |= std::strcmp(argv[i], "break") == 0;
    }
    std::printf("device %s (%s), mode %s, %zu KiB per kernel, %d pairs%s\n", prop.name, prop.gcnArchName, mode.c_str(), kib, pairs, do_break ? ", BREAK" : "");
    // 1. clock known answer
    {
        unsigned long long *d, h0, h1;
        CHECK(hipMalloc(&d, 8));
        stamp_k<<<1, 64>>>(d);
        CHECK(hipDeviceSynchronize());
        const auto c0 = std::chrono::steady_clock::now();
        CHECK(hipMemcpy(&h0, d, 8, hipMemcpyDeviceToHost));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        stamp_k<<<1, 64>>>(d);
        CHECK(hipDeviceSynchronize());
        const auto c1 = std::chrono::steady_clock::now();
        CHECK(hipMemcpy(&h1, d, 8, hipMemcpyDeviceToHost));
        const double secs = std::chrono::duration<double>(c1 - c0).count();
        const double rate = (double) (h1 - h0) / secs;
        const bool ok = std::fabs(rate - 100e6) <= 0.01 * 100e6;
        std::printf("RESULT clock_known_answer ticks %llu host_s %.6f ticks_per_s %.0f (expected 100000000 +- 1 %%)\n", h1 - h0, secs, rate);
        std::printf("VERDICT clock-known-answer %s\n", ok ? "PASS" : "FAIL");
        CHECK(hipFree(d));
        if (!ok) {
            std::printf("VERDICT pair_overlap FAIL: the counter is not the 100 MHz one assumed\n");
            return 1;
        }
        if (mode == "clock") {
            std::printf("VERDICT pair_overlap PASS\n");
            return 0;
        }
    }
    const int items = 2;  // 16-byte stores per thread
    const int blocks = (int) std::max<size_t>(16, (kib * 1024) / (16 * BLOCK * items));
    const size_t buf_bytes = (size_t) blocks * BLOCK * items * 16;
    const int waves = blocks * WAVES_PER_BLOCK;
    const int per_graph = 100;
    const int total_kernels = 2 * pairs;
    std::printf("RESULT shape blocks %d x %d threads, %d items of 16 B per thread, buffer %zu KiB, %d waves per kernel\n", blocks, BLOCK, items, buf_bytes >> 10, waves);

    uint4 * buf;
    unsigned * err;
    unsigned long long *first, *last;
    CHECK(hipMalloc(&buf, buf_bytes));
    CHECK(hipMalloc(&err, sizeof(unsigned)));
    const size_t nstamps = (size_t) (mode == "graph" ? 2 * per_graph : total_kernels) * waves;
    CHECK(hipMalloc(&first, nstamps * 8));
    CHECK(hipMalloc(&last, nstamps * 8));
    CHECK(hipMemset(buf, 0, buf_bytes));
    CHECK(hipMemset(err, 0, sizeof(unsigned)));
    hipStream_t s1, s2;
    CHECK(hipStreamCreate(&s1));
    CHECK(hipStreamCreate(&s2));

    std::vector<Agg> agg;  // per kernel in launch order (A0 B0 A1 B1 ...)
    std::vector<unsigned long long> hf(nstamps), hl(nstamps);
    auto tag_of = [](int pair) { return 1000u + (unsigned) pair; };
    auto launch_pair = [&](int pair, int slot, hipStream_t s) {
        const unsigned tag = tag_of(pair);
        pair_a<<<blocks, BLOCK, 0, s>>>(buf, tag, items, first + (size_t) (2 * slot) * waves, last + (size_t) (2 * slot) * waves);
        pair_b<<<blocks, BLOCK, 0, s>>>(buf, do_break ? tag + 1 : tag, items, err, first + (size_t) (2 * slot + 1) * waves, last + (size_t) (2 * slot + 1) * waves);
    };
    auto fetch = [&](int nk) {
        CHECK(hipMemcpy(hf.data(), first, (size_t) nk * waves * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(hl.data(), last, (size_t) nk * waves * 8, hipMemcpyDeviceToHost));
        for (int k = 0; k < nk; ++k) {
            agg.push_back(reduce_waves(&hf[(size_t) k * waves], &hl[(size_t) k * waves], waves));
        }
    };
    const char * kname = "AB";
    Gaps dep, indep;  // A_i -> B_i, B_i -> A_(i+1)
    std::vector<char> kinds;
    if (mode == "plain" || mode == "hostsync") {
        for (int i = 0; i < 20; ++i) {  // warm up
            launch_pair(i, 0, s1);
        }
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemset(err, 0, sizeof(unsigned)));
        for (int i = 0; i < pairs; ++i) {
            const unsigned tag = tag_of(i);
            pair_a<<<blocks, BLOCK, 0, s1>>>(buf, tag, items, first + (size_t) (2 * i) * waves, last + (size_t) (2 * i) * waves);
            if (mode == "hostsync") {
                CHECK(hipStreamSynchronize(s1));
            }
            pair_b<<<blocks, BLOCK, 0, s1>>>(buf, do_break ? tag + 1 : tag, items, err, first + (size_t) (2 * i + 1) * waves, last + (size_t) (2 * i + 1) * waves);
            if (mode == "hostsync") {
                CHECK(hipStreamSynchronize(s1));
            }
        }
        CHECK(hipDeviceSynchronize());
        fetch(total_kernels);
        for (int i = 0; i < pairs; ++i) {
            add_gap(dep, agg[2 * i], agg[2 * i + 1]);
            if (i + 1 < pairs) {
                add_gap(indep, agg[2 * i + 1], agg[2 * i + 2]);
            }
            kinds.push_back('A');
            kinds.push_back('B');
        }
    } else if (mode == "graph") {
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));
        for (int p = 0; p < per_graph; ++p) {
            launch_pair(p, p, s1);
        }
        CHECK(hipStreamEndCapture(s1, &graph));
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        const int launches = std::max(1, pairs / per_graph);
        for (int w = 0; w < 3; ++w) {  // warm up
            CHECK(hipGraphLaunch(exec, s1));
            CHECK(hipStreamSynchronize(s1));
        }
        CHECK(hipMemset(err, 0, sizeof(unsigned)));
        for (int l = 0; l < launches; ++l) {
            CHECK(hipGraphLaunch(exec, s1));
            CHECK(hipStreamSynchronize(s1));
            std::vector<Agg> one;
            CHECK(hipMemcpy(hf.data(), first, (size_t) 2 * per_graph * waves * 8, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(hl.data(), last, (size_t) 2 * per_graph * waves * 8, hipMemcpyDeviceToHost));
            for (int k = 0; k < 2 * per_graph; ++k) {
                one.push_back(reduce_waves(&hf[(size_t) k * waves], &hl[(size_t) k * waves], waves));
            }
            for (int p = 0; p < per_graph; ++p) {
                add_gap(dep, one[2 * p], one[2 * p + 1]);
                if (p + 1 < per_graph) {
                    add_gap(indep, one[2 * p + 1], one[2 * p + 2]);
                }
            }
            for (const Agg & a : one) {
                agg.push_back(a);
            }
            for (int k = 0; k < per_graph; ++k) {
                kinds.push_back('A');
                kinds.push_back('B');
            }
        }
    } else if (mode == "twostreams") {
        const int rounds = pairs;
        const unsigned long long ticks = 20000;  // 200 us
        const int spin_blocks = 16;
        Gaps both;
        std::vector<Agg> p_agg, q_agg;
        for (int r = -5; r < rounds; ++r) {  // 5 warm-up rounds
            spin_k<<<spin_blocks, BLOCK, 0, s1>>>(ticks, first, last);
            spin_k<<<spin_blocks, BLOCK, 0, s2>>>(ticks, first + (size_t) waves, last + (size_t) waves);
            CHECK(hipDeviceSynchronize());
            if (r < 0) {
                continue;
            }
            CHECK(hipMemcpy(hf.data(), first, (size_t) 2 * waves * 8, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(hl.data(), last, (size_t) 2 * waves * 8, hipMemcpyDeviceToHost));
            const Agg p = reduce_waves(&hf[0], &hl[0], spin_blocks * WAVES_PER_BLOCK);
            const Agg q = reduce_waves(&hf[waves], &hl[waves], spin_blocks * WAVES_PER_BLOCK);
            agg.push_back(p);
            agg.push_back(q);
            kinds.push_back('P');
            kinds.push_back('Q');
            add_gap(both, p, q);  // gap < 0: Q started before P's last wave ended
        }
        report_gaps("twostreams_Q_start_minus_P_end", both);
        const bool ok = both.overlap == both.n;
        std::printf("VERDICT positive-control %s (in-kernel clock: Q starts before P ends in %lld of %lld rounds)\n", ok ? "PASS" : "FAIL", both.overlap, both.n);
        if (dump) {
            std::FILE * f = std::fopen(dump, "w");
            std::fprintf(f, "idx,kind,first_min,first_max,last_min,last_max\n");
            for (size_t k = 0; k < agg.size(); ++k) {
                std::fprintf(f, "%zu,%c,%llu,%llu,%llu,%llu\n", k, kinds[k], agg[k].first_min, agg[k].first_max, agg[k].last_min, agg[k].last_max);
            }
            std::fclose(f);
        }
        if (!ok) {
            std::printf("VERDICT pair_overlap FAIL\n");
            return 1;
        }
        std::printf("VERDICT pair_overlap PASS\n");
        return 0;
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    (void) kname;
    unsigned herr = 0;
    CHECK(hipMemcpy(&herr, err, sizeof(unsigned), hipMemcpyDeviceToHost));
    report_gaps("dependent_A_to_B", dep);
    report_gaps("neighbour_B_to_nextA", indep);
    std::printf("RESULT data_check mismatching_values %u\n", herr);
    // kernel durations as the in-kernel clock sees them (first_min to last_max), for the size of the pairs
    std::vector<double> dur;
    for (const Agg & a : agg) {
        dur.push_back(10.0 * (double) (a.last_max - a.first_min));
    }
    std::sort(dur.begin(), dur.end());
    std::printf("RESULT kernel_span_ns median %.0f min %.0f max %.0f\n", dur[dur.size() / 2], dur.front(), dur.back());
    bool pass = true;
    if (do_break) {
        // the deliberate break: B expects the wrong tag, so the data check must see mismatches
        std::printf("VERDICT data-check %s (deliberate break: B expects tag+1, must FAIL)\n", herr == 0 ? "PASS" : "FAIL");
        pass = herr == 0;
    } else {
        std::printf("VERDICT data-check %s\n", herr == 0 ? "PASS" : "FAIL");
        pass = herr == 0;
    }
    if (mode == "hostsync") {
        const bool ok = dep.overlap == 0 && indep.overlap == 0 && dep.ambiguous == 0 && indep.ambiguous == 0;
        std::printf("VERDICT negative-control %s (host sync between kernels: in-kernel overlaps %lld, ambiguous %lld)\n", ok ? "PASS" : "FAIL", dep.overlap + indep.overlap,
                    dep.ambiguous + indep.ambiguous);
        pass = pass && ok;
    }
    if (dump) {
        std::FILE * f = std::fopen(dump, "w");
        std::fprintf(f, "idx,kind,first_min,first_max,last_min,last_max\n");
        for (size_t k = 0; k < agg.size(); ++k) {
            std::fprintf(f, "%zu,%c,%llu,%llu,%llu,%llu\n", k, kinds[k], agg[k].first_min, agg[k].first_max, agg[k].last_min, agg[k].last_max);
        }
        std::fclose(f);
    }
    std::printf("VERDICT pair_overlap %s\n", pass ? "PASS" : "FAIL");
    CHECK(hipFree(buf));
    CHECK(hipFree(err));
    CHECK(hipFree(first));
    CHECK(hipFree(last));
    return pass ? 0 : 1;
}
