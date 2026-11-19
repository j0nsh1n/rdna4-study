// Lane 7 items 2 and 8: where and when work-groups and waves start and end, how many run at once per SIMD, CU and
// work-group processor, the empty time between one group ending and the next starting on a CU, whether the pattern
// repeats from launch to launch, and throughput against the number of compute units in use.
// Built twice: sched_place_wgp (HIP default, WGP mode) and sched_place_cu (-mcumode -DCUMODE=1).
//
// Usage: sched_place <place|regs|ids|scale> [--break]
//   place  grid x block sweep; every wave spins about 20 us on REALTIME and records HW_ID1, SE, REALTIME start/end.
//   regs   the same with 256-thread groups and kernels that keep 8 to 240 registers live: concurrent waves per SIMD
//          against the reported register count, and against the rule min(16, floor(1536 / VGPRs rounded up to 24)).
//   ids    item 8: the distinct values of each hardware-ID field over a launch that covers the whole card.
//   scale  throughput against compute units in use (one group per CU forced by 40 KiB of unused LDS): an FMA-bound
//          and a memory-bound kernel, G = 1 .. 64 groups. Whole card: thermal gate and 95 C stop.
// --break (place): expects the 16-group grid of 32-thread groups to occupy 32 SIMDs, more than 16 single-wave
// groups can occupy, so the placement verdict must FAIL.
#include "sched_common.h"

#include <algorithm>
#include <functional>
#include <set>

using namespace sched;

namespace {

struct Rec {
    uint32_t hw, se, blk, w;
    uint64_t r0, r1;
};

// The wait spins on the wave's own cycle counter (a register read), not on REALTIME: polling REALTIME (a message)
// from many waves delayed other waves' start stamps by whole spin lengths in the first version.
template <int NR, int LB = 1024>
__global__ __launch_bounds__(LB) void k_place(Rec * rec, uint64_t ticks, float seed) {
    float v[NR > 0 ? NR : 1];
    if constexpr (NR > 0) {
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            v[i] = seed * (float) (i + threadIdx.x);
            asm volatile("" : "+v"(v[i]));
        }
    }
    const uint32_t hw = hwid1(), se = se_aid();
    const uint64_t r0 = rt();
    const uint64_t c0 = cyc(), cycles = ticks * 32;  // about `ticks` REALTIME ticks at 3.2 GHz
    while (cyc() - c0 < cycles) {
    }
    const uint64_t r1 = rt();
    float s = 0;
    if constexpr (NR > 0) {
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            asm volatile("" : "+v"(v[i]));
            s += v[i];
        }
    }
    if ((threadIdx.x & 31) == 0) {
        const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        rec[wid] = {hw, se, blockIdx.x, threadIdx.x >> 5, r0, r1};
        if (s == 1234.5f) {
            rec[wid].w = 999;
        }
    }
}

inline uint32_t pl_of(const Rec & r) {
    const uint32_t id1 = r.hw;
    const uint32_t slot = id1 & 31u, simd = (id1 >> 8) & 3u, wgp = (id1 >> 10) & 15u, sa = (id1 >> 16) & 1u;
    return (r.se & 15u) | (sa << 4) | (wgp << 5) | ((simd & 1u) << 9) | ((simd >> 1) << 10) | (slot << 11);
}

// Maximum number of [r0, r1) intervals open at once per key.
template <class KeyF>
int max_conc(const std::vector<Rec> & v, KeyF key, bool per_group) {
    std::map<uint32_t, std::vector<std::pair<uint64_t, int>>> ev;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint64_t, uint64_t>> grp;
    if (per_group) {
        for (const Rec & r : v) {
            auto k = std::make_pair(key(r), r.blk);
            auto it = grp.find(k);
            if (it == grp.end()) {
                grp[k] = {r.r0, r.r1};
            } else {
                it->second.first = std::min(it->second.first, r.r0);
                it->second.second = std::max(it->second.second, r.r1);
            }
        }
        for (auto & g : grp) {
            ev[g.first.first].push_back({g.second.first, 1});
            ev[g.first.first].push_back({g.second.second, -1});
        }
    } else {
        for (const Rec & r : v) {
            ev[key(r)].push_back({r.r0, 1});
            ev[key(r)].push_back({r.r1, -1});
        }
    }
    int best = 0;
    for (auto & kv : ev) {
        std::sort(kv.second.begin(), kv.second.end(), [](auto a, auto b) {
            return a.first != b.first ? a.first < b.first : a.second < b.second;  // ends before starts at a tie
        });
        int c = 0;
        for (auto & e : kv.second) {
            c += e.second;
            best = std::max(best, c);
        }
    }
    return best;
}

struct Summary {
    int ses, wgps, cus, simds, conc_simd, conc_cu_groups, conc_wgp_groups, waves_per_group_simds;
    double gap_med, gap_p90, start_spread_us, span_us;
    int ngaps;
};

Summary summarize(const std::vector<Rec> & v) {
    Summary s{};
    std::set<uint32_t> se, wgp, cu, simd;
    std::map<uint32_t, std::set<uint32_t>> grp_simds;
    uint64_t mn = ~0ull, mx = 0;
    for (const Rec & r : v) {
        const uint32_t p = pl_of(r);
        se.insert(pl_se(p));
        wgp.insert(pl_wgpkey(p));
        cu.insert(pl_cu(p));
        simd.insert(pl_simd(p));
        grp_simds[r.blk].insert(pl_simd(p));
        mn = std::min(mn, r.r0);
        mx = std::max(mx, r.r1);
    }
    s.ses = (int) se.size();
    s.wgps = (int) wgp.size();
    s.cus = (int) cu.size();
    s.simds = (int) simd.size();
    s.waves_per_group_simds = (int) grp_simds.begin()->second.size();
    s.conc_simd = max_conc(v, [](const Rec & r) { return pl_simd(pl_of(r)); }, false);
    s.conc_cu_groups = max_conc(v, [](const Rec & r) { return pl_cu(pl_of(r)); }, true);
    s.conc_wgp_groups = max_conc(v, [](const Rec & r) { return pl_wgpkey(pl_of(r)); }, true);
    s.span_us = (double) (mx - mn) / 100.0;
    // Group intervals per CU (a WGP-mode group spans two CUs: keyed by its first wave's CU).
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> g;
    std::map<uint32_t, uint32_t> gcu;
    for (const Rec & r : v) {
        auto it = g.find(r.blk);
        if (it == g.end()) {
            g[r.blk] = {r.r0, r.r1};
            gcu[r.blk] = pl_cu(pl_of(r));
        } else {
            it->second.first = std::min(it->second.first, r.r0);
            it->second.second = std::max(it->second.second, r.r1);
        }
    }
    std::vector<double> starts;
    for (auto & kv : g) {
        starts.push_back((double) (kv.second.first - mn) / 100.0);
    }
    std::sort(starts.begin(), starts.end());
    s.start_spread_us = starts.empty() ? 0 : starts[std::min(starts.size() - 1, (size_t) 63)];
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> per;
    for (auto & kv : g) {
        per[gcu[kv.first]].push_back(kv.second);
    }
    std::vector<double> gaps;
    for (auto & kv : per) {
        auto & iv = kv.second;
        std::sort(iv.begin(), iv.end());
        for (size_t i = 0; i < iv.size(); ++i) {
            if (iv[i].first <= mn + 100) {
                continue;  // started with the launch, not into a freed slot
            }
            uint64_t last_end = 0;
            for (size_t k = 0; k < iv.size(); ++k) {
                if (k != i && iv[k].second <= iv[i].first) {
                    last_end = std::max(last_end, iv[k].second);
                }
            }
            if (last_end) {
                gaps.push_back((double) (iv[i].first - last_end) / 100.0);
            }
        }
    }
    s.ngaps = (int) gaps.size();
    s.gap_med = pct(gaps, 0.5);
    s.gap_p90 = pct(gaps, 0.9);
    return s;
}

using KFun = void (*)(Rec *, uint64_t, float);

std::vector<Rec> run(KFun k, int grid, int block, uint64_t ticks, Rec * d) {
    const int nw = grid * (block / 32);
    CHECK(hipMemset(d, 0, nw * sizeof(Rec)));
    hipLaunchKernelGGL(k, dim3(grid), dim3(block), 0, 0, d, ticks, 1.0f);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<Rec> h(nw);
    CHECK(hipMemcpy(h.data(), d, nw * sizeof(Rec), hipMemcpyDeviceToHost));
    return h;
}

#ifdef CUMODE
constexpr const char * MODE = "cu";
#else
constexpr const char * MODE = "wgp";
#endif

int mode_place(bool brk) {
    Rec * d;
    CHECK(hipMalloc(&d, (size_t) 4096 * 32 * sizeof(Rec)));
    bool ok = true;
    run(k_place<0>, 64, 64, 100, d);  // warm-up
    std::printf("TABLE mode grid block | SEs WGPs CUs SIMDs | max_waves_per_SIMD max_groups_per_CU max_groups_per_WGP "
                "SIMDs_per_group | first64_start_us span_us | slot_gap_us(median p90 n) | same_CU_as_launch0\n");
    for (int block : {32, 64, 256, 1024}) {
        for (int grid : {16, 64, 128, 512, 4096}) {
            std::vector<std::vector<Rec>> L;
            for (int l = 0; l < 5; ++l) {
                L.push_back(run(k_place<0>, grid, block, 2000, d));
            }
            const Summary s = summarize(L[0]);
            // Repeatability: share of groups whose first wave lands on the same CU as in launch 0, launches 1-4.
            std::map<uint32_t, uint32_t> cu0;
            for (const Rec & r : L[0]) {
                if (r.w == 0) {
                    cu0[r.blk] = pl_cu(pl_of(r));
                }
            }
            int same = 0, tot = 0;
            bool all_ok = true;
            for (int l = 1; l < 5; ++l) {
                for (const Rec & r : L[l]) {
                    if (r.w == 0) {
                        same += cu0[r.blk] == pl_cu(pl_of(r));
                        ++tot;
                    }
                }
            }
            // Every record present (each wave wrote its slot) in every launch.
            for (auto & v : L) {
                for (const Rec & r : v) {
                    all_ok = all_ok && r.r1 > r.r0;
                }
            }
            ok = ok && all_ok;
            std::printf("LAUNCHES %s grid=%d block=%d first64_start_us/span_us:", MODE, grid, block);
            for (auto & v : L) {
                const Summary q = summarize(v);
                std::printf(" %.2f/%.1f", q.start_spread_us, q.span_us);
            }
            std::printf("\n");
            std::printf("RESULT place %s grid=%d block=%d | %d %d %d %d | %d %d %d %d | %.2f %.1f | %.2f %.2f %d | %.2f\n",
                        MODE, grid, block, s.ses, s.wgps, s.cus, s.simds, s.conc_simd, s.conc_cu_groups,
                        s.conc_wgp_groups, s.waves_per_group_simds, s.start_spread_us, s.span_us, s.gap_med,
                        s.gap_p90, s.ngaps, tot ? (double) same / tot : 0);
            // Placement verdict, checked in every launch: a 16-group grid of single-wave groups can occupy at most 16
            // SIMDs; claim 16 (or 32 for --break) and require exactly that many distinct SIMDs.
            if (grid == 16 && block == 32) {
                const int want = brk ? 32 : 16;
                for (auto & v : L) {
                    const int got = summarize(v).simds;
                    if (got != want) {
                        char t[160];
                        std::snprintf(t, sizeof t, "16 groups of one wave: %d distinct SIMDs, expected %d", got, want);
                        verdict("sched_place placement16", "FAIL", t);
                        ok = false;
                        goto next;
                    }
                }
                verdict("sched_place placement16", "PASS", "16 groups of one wave occupied 16 distinct SIMDs in all 5 launches");
            next:;
            }
            // Fill order of the 4096-group grids: the first 24 groups by start time.
            if (grid == 128 && (block == 32 || block == 1024)) {
                std::vector<std::pair<uint64_t, uint32_t>> st;
                std::set<uint32_t> seen;
                for (const Rec & r : L[0]) {
                    if (r.w == 0) {
                        st.push_back({r.r0, pl_of(r)});
                    }
                }
                std::sort(st.begin(), st.end());
                std::printf("ORDER %s grid=%d block=%d (start_ticks_after_first SE.SA.WGP.CU.SIMD):", MODE, grid, block);
                for (size_t i = 0; i < st.size() && i < 40; ++i) {
                    const uint32_t p = st[i].second;
                    std::printf(" %llu:%d.%d.%d.%d.%d", (unsigned long long) (st[i].first - st[0].first), pl_se(p),
                                pl_sa(p), pl_wgp(p), pl_cuinwgp(p), pl_simdincu(p));
                }
                std::printf("\n");
            }
        }
    }
    verdict("sched_place records", ok ? "PASS" : "FAIL", "every wave wrote its record in every launch");
    return ok ? 0 : 1;
}

template <int NR>
bool regs_row(Rec * d) {
    hipFuncAttributes a;
    CHECK(hipFuncGetAttributes(&a, reinterpret_cast<const void *>(k_place<NR, 256>)));
    const auto v = run(k_place<NR, 256>, 4096, 256, 2000, d);
    const Summary s = summarize(v);
    const int regs = a.numRegs;
    const int rounded = (regs + 23) / 24 * 24;
    const int rule = std::min(16, 1536 / std::max(rounded, 1));
    // Groups are placed whole: a 256-thread group adds 8 / (SIMDs it spans) waves to each SIMD, so the measured
    // count is the rule rounded down to that step.
    const int step = 8 / std::max(1, s.waves_per_group_simds);
    const int want = rule / step * step;
    std::printf("RESULT regs %s live_floats=%d numRegs=%d rounded=%d rule_waves_per_SIMD=%d group_step=%d "
                "expected=%d measured_max_waves_per_SIMD=%d max_groups_per_CU=%d\n",
                MODE, NR, regs, rounded, rule, step, want, s.conc_simd, s.conc_cu_groups);
    return s.conc_simd == want;
}

int mode_regs() {
    Rec * d;
    CHECK(hipMalloc(&d, (size_t) 4096 * 8 * sizeof(Rec)));
    run(k_place<0>, 64, 64, 100, d);
    bool ok = regs_row<0>(d);
    ok = regs_row<40>(d) && ok;
    ok = regs_row<64>(d) && ok;
    ok = regs_row<88>(d) && ok;
    ok = regs_row<120>(d) && ok;
    ok = regs_row<150>(d) && ok;
    ok = regs_row<180>(d) && ok;
    ok = regs_row<240>(d) && ok;
    verdict("sched_regs", ok ? "PASS" : "FAIL",
            ok ? "measured waves per SIMD equal min(16, floor(1536 / VGPRs rounded up to 24)), rounded down to whole groups, in every row"
               : "a row differs from the rule (see RESULT regs)");
    return ok ? 0 : 1;
}

int mode_ids() {
    Rec * d;
    CHECK(hipMalloc(&d, (size_t) 4096 * 32 * sizeof(Rec)));
    const auto v = run(k_place<0>, 4096, 1024, 500, d);
    std::map<std::string, std::set<uint32_t>> f;
    std::set<uint32_t> raw_hi;
    for (const Rec & r : v) {
        f["SE (GET_SE_AID_ID 3:0)"].insert(r.se & 15);
        f["SE_AID upper bits (31:4)"].insert(r.se >> 4);
        f["WAVE_ID (HW_ID1 4:0)"].insert(r.hw & 31);
        f["HW_ID1 7:5"].insert((r.hw >> 5) & 7);
        f["SIMD_ID (HW_ID1 9:8)"].insert((r.hw >> 8) & 3);
        f["WGP_ID (HW_ID1 13:10)"].insert((r.hw >> 10) & 15);
        f["HW_ID1 15:14"].insert((r.hw >> 14) & 3);
        f["SA_ID (HW_ID1 16)"].insert((r.hw >> 16) & 1);
        f["HW_ID1 31:17"].insert(r.hw >> 17);
        f["SE.SA.WGP"].insert(pl_wgpkey(pl_of(r)));
        f["SE.SA.WGP.CU"].insert(pl_cu(pl_of(r)));
        f["SE.SA.WGP.CU.SIMD"].insert(pl_simd(pl_of(r)));
    }
    // HW_ID1 bits 19:18 took four values in a first run: do they equal the shader engine from the message?
    size_t eq = 0;
    for (const Rec & r : v) {
        eq += ((r.hw >> 18) & 3u) == (r.se & 15u);
    }
    std::printf("RESULT ids %s HW_ID1_bits_19_18_equal_SE_from_GET_SE_AID_ID=%zu_of_%zu\n", MODE, eq, v.size());
    for (auto & kv : f) {
        std::printf("RESULT ids %s field=\"%s\" distinct=%zu values=", MODE, kv.first.c_str(), kv.second.size());
        int n = 0;
        for (uint32_t x : kv.second) {
            if (n++ < 20) {
                std::printf("%s0x%x", n > 1 ? "," : "", x);
            }
        }
        std::printf("%s\n", kv.second.size() > 20 ? ",..." : "");
    }
    // WGPs per SA per SE.
    std::map<uint32_t, std::set<uint32_t>> wps;
    for (const Rec & r : v) {
        const uint32_t p = pl_of(r);
        wps[(pl_se(p) << 1) | pl_sa(p)].insert(pl_wgp(p));
    }
    for (auto & kv : wps) {
        std::printf("RESULT ids %s SE=%u SA=%u wgp_ids=", MODE, kv.first >> 1, kv.first & 1);
        for (uint32_t x : kv.second) {
            std::printf("%u ", x);
        }
        std::printf("\n");
    }
    const bool ok = f["SE.SA.WGP.CU.SIMD"].size() == 128 && f["SE.SA.WGP.CU"].size() == 64;
    char t[160];
    std::snprintf(t, sizeof t, "%zu distinct SIMDs and %zu CUs seen (driver topology: 128 and 64)",
                  f["SE.SA.WGP.CU.SIMD"].size(), f["SE.SA.WGP.CU"].size());
    verdict("sched_ids topology", ok ? "PASS" : "FAIL", t);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
// scale: FMA-bound (accumulate form, paired by clang into v_dual_fmac_f32, as lane 4's mat_card fma) and memory-bound.
__global__ __launch_bounds__(1024) void k_fma(const float * __restrict__ k, float * __restrict__ sink, Rec * rec,
                                              int iters) {
    float k0 = k[0], k1 = k[1];
    asm volatile("" : "+v"(k0), "+v"(k1));
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = 0.5f + (float) ((threadIdx.x + i) & 63) * 1e-3f;
    }
    const uint32_t hw = hwid1(), se = se_aid();
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
        sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = {hw, se, blockIdx.x, threadIdx.x >> 5, r0, r1};
    }
}

__global__ __launch_bounds__(1024) void k_stream(const uint4 * __restrict__ src, size_t n, unsigned * __restrict__ sink,
                                                 Rec * rec) {
    const uint32_t hw = hwid1(), se = se_aid();
    const uint64_t r0 = rt();
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    const uint64_t r1 = rt();
    if (acc == 0xdeadbeefu) {
        sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    }
    if ((threadIdx.x & 31) == 0) {
        rec[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = {hw, se, blockIdx.x, threadIdx.x >> 5, r0, r1};
    }
}

double ev_ms(const std::function<void()> & f) {
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    CHECK(hipEventRecord(a));
    f();
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms;
    CHECK(hipEventElapsedTime(&ms, a, b));
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    return ms;
}

int mode_scale(int argc, char ** argv) {
    const double expect_tfma = std::atof(opt(argc, argv, "--expect-tfma", "24.7"));  // lane 4, mat_card fma W = 8
    float hk[2] = {0.9999f, 1e-4f};
    float *d_k, *d_sink;
    Rec * d_rec;
    CHECK(hipMalloc(&d_k, sizeof hk));
    CHECK(hipMemcpy(d_k, hk, sizeof hk, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_sink, 64 * 1024 * 4));
    CHECK(hipMalloc(&d_rec, 64 * 32 * sizeof(Rec)));
    const size_t bytes = (size_t) 512 << 20;
    uint4 * d_src;
    CHECK(hipMalloc(&d_src, bytes));
    CHECK(hipMemset(d_src, 1, bytes));
    if (!gate(60.0)) {
        verdict("sched_scale", "FAIL", "hotspot did not fall below 60 C");
        return 1;
    }
    Sampler smp(50);
    smp.start();
    bool ok = true;
    double whole = 0;
    const int iters = 1200000;  // about 30 ms per launch: short launches ran at 2.5-2.8 GHz in a first version
    // Warm-up: 0.2 s of whole-card load so every row runs at the loaded clock.
    for (int w = 0; w < 6 && !smp.hot; ++w) {
        hipLaunchKernelGGL(k_fma, dim3(64), dim3(512), 40 * 1024, 0, d_k, d_sink, d_rec, iters);
        CHECK(hipDeviceSynchronize());
    }
    for (int G : {1, 2, 4, 8, 16, 32, 48, 64}) {
        if (smp.hot) {
            break;
        }
        // 512 threads + 40 KiB LDS: one group per CU (16 waves, 8 per SIMD in CU mode).
        auto f = [&] { hipLaunchKernelGGL(k_fma, dim3(G), dim3(512), 40 * 1024, 0, d_k, d_sink, d_rec, iters); };
        f();
        CHECK(hipDeviceSynchronize());
        std::vector<double> ms;
        for (int r = 0; r < 3; ++r) {
            ms.push_back(ev_ms(f));
        }
        std::vector<Rec> h(G * 16);
        CHECK(hipMemcpy(h.data(), d_rec, h.size() * sizeof(Rec), hipMemcpyDeviceToHost));
        const Summary s = summarize(h);
        const double rate = (double) G * 512 * 16 * iters / (pct(ms, 0.5) * 1e-3) / 1e12;
        std::printf("RESULT scale %s fma groups=%d CUs=%d SIMDs=%d max_waves_per_SIMD=%d ms=%.3f TFMA_s=%.3f per_CU=%.4f\n",
                    MODE, G, s.cus, s.simds, s.conc_simd, pct(ms, 0.5), rate, rate / G);
        ok = ok && s.cus == G;
        if (G == 64) {
            whole = rate;
        }
    }
    for (int G : {1, 2, 4, 8, 16, 32, 48, 64}) {
        auto f = [&] {
            hipLaunchKernelGGL(k_stream, dim3(G), dim3(1024), 40 * 1024, 0, d_src, bytes / 16, (unsigned *) d_sink, d_rec);
        };
        f();
        CHECK(hipDeviceSynchronize());
        std::vector<double> ms;
        for (int r = 0; r < 3; ++r) {
            ms.push_back(ev_ms(f));
        }
        std::vector<Rec> h(G * 32);
        CHECK(hipMemcpy(h.data(), d_rec, h.size() * sizeof(Rec), hipMemcpyDeviceToHost));
        const Summary s = summarize(h);
        const double gbs = (double) bytes / (pct(ms, 0.5) * 1e-3) / 1e9;
        std::printf("RESULT scale %s stream groups=%d CUs=%d ms=%.2f GB_s=%.1f per_CU=%.2f\n", MODE, G, s.cus,
                    pct(ms, 0.5), gbs, gbs / G);
        ok = ok && s.cus == G;
    }
    smp.stop();
    double smax = 0, hmax = 0;
    for (const Snap & s : smp.copy()) {
        smax = std::max(smax, (double) s.sclk);
        hmax = std::max(hmax, s.hot);
    }
    std::vector<double> sc;
    for (const Snap & q : smp.copy()) {
        if (q.sclk > 1000) {
            sc.push_back(q.sclk);
        }
    }
    std::printf("RESULT scale sysfs sclk_loaded_median=%.0f sclk_max=%.0f hotspot_max=%.0f\n", pct(sc, 0.5), smax, hmax);
    char t[200];
    std::snprintf(t, sizeof t, "one group per CU in every row: %s", ok ? "yes" : "NO");
    verdict("sched_scale placement", ok ? "PASS" : "FAIL", t);
    const bool ok2 = std::fabs(whole / expect_tfma - 1.0) <= 0.05;
    std::snprintf(t, sizeof t, "whole-card FMA %.2f TFMA/s against lane 4's %.1f (within 5 %%)", whole, expect_tfma);
    verdict("sched_scale whole_card_fma", ok2 ? "PASS" : "FAIL", t);
    return ok && ok2 ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    if (argc < 2) {
        std::fprintf(stderr, "usage: sched_place <place|regs|ids|scale> [--break]\n");
        return 2;
    }
    const std::string m = argv[1];
    int rc = 2;
    if (m == "place") {
        rc = mode_place(has_flag(argc, argv, "--break"));
    } else if (m == "regs") {
        rc = mode_regs();
    } else if (m == "ids") {
        rc = mode_ids();
    } else if (m == "scale") {
        rc = mode_scale(argc, argv);
    }
    std::printf("HOTTEST %.0f C\n", hottest().load());
    return rc;
}
