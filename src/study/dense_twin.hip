// Gaming round, E1: the HIP twin of src/vk/shaders/gdense.comp, so that the clock after idle can be compared between
// HIP and Vulkan with the same load. One chunk is 8 blocks of 16 independent v_fma_f32 (x = x * a + b; a and b per
// lane in VGPRs, written here as inline assembly in the form ACO emits for the GLSL); every `every` chunks every wave
// reads REALTIME and its cycle counter, ends when REALTIME has passed its own start + the time bound or when the
// host has set the stop word (the 88 C tripwire). Wave 0 of every group writes its start and end; wave 0 of every
// rec_stride-th group writes a (REALTIME, cycles) record at every check. The files have the Vulkan probe's layout,
// and scripts/gprobe_dense.py reads both.
//
// Usage: dense_twin --out <prefix> [--idle-ms 1000] [--stop-after-ms X] [--hot-stop-c +N] arm...
//   arm = name,groups,threads,every,ms[,rec_stride[,rec_every]]   (wave32; threads per group a multiple of 32;
//   a recording wave writes a record at every rec_every-th check)
// Each arm runs after its own idle time. A host thread samples sysfs every 5 ms (read-only), the active power profile
// every 10 ms, and sets the stop word at 88 C. --stop-after-ms: the host sets the stop word X ms after the launch (the
// tripwire's known answer). --hot-stop-c +N (F5, the stop path's end-to-end test): the sampler's limit is lowered to
// N C above the hotspot read just before the launch (88 C before and after); a stop there is the expected outcome.
#include "sched_common.h"

#include <ctime>
#include <string>

using namespace sched;

namespace {

#define DF1(i) "v_fma_f32 %" #i ", %" #i ", %16, %17\n"
#define DF16 DF1(0) DF1(1) DF1(2) DF1(3) DF1(4) DF1(5) DF1(6) DF1(7) DF1(8) DF1(9) DF1(10) DF1(11) DF1(12) DF1(13) \
    DF1(14) DF1(15)
#define DX16 "+v"(x[0]), "+v"(x[1]), "+v"(x[2]), "+v"(x[3]), "+v"(x[4]), "+v"(x[5]), "+v"(x[6]), "+v"(x[7]), \
    "+v"(x[8]), "+v"(x[9]), "+v"(x[10]), "+v"(x[11]), "+v"(x[12]), "+v"(x[13]), "+v"(x[14]), "+v"(x[15])

__device__ __forceinline__ void dense_block(float (&x)[16], float a, float b) {
    asm volatile(DF16 : DX16 : "v"(a), "v"(b));
}

__global__ __launch_bounds__(1024) void k_dense(const float * ab, float * res, uint4 * rec, uint4 * grp,
                                                const uint32_t * stopw, uint32_t chunks, uint32_t every, uint64_t lim,
                                                uint32_t rec_stride, uint32_t maxrec, uint32_t rec_every) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    const float a = ab[2 * (threadIdx.x & 63u)], b = ab[2 * (threadIdx.x & 63u) + 1];
    float x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = (float) lane * 1e-3f + 0.01f * (float) i;
    }
    const bool w0 = wave == 0;
    const bool recw = w0 && rec_stride != 0 && blockIdx.x % rec_stride == 0;
    const uint32_t ridx = (rec_stride ? blockIdx.x / rec_stride : 0) * maxrec;
    const uint64_t r0 = rt(), c0 = cyc();
    uint32_t n = 0, left = every, rleft = rec_every, done = 0;
    for (uint32_t ch = 0; ch < chunks; ++ch) {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            dense_block(x, a, b);
        }
        ++done;
        if (--left == 0) {
            left = every;
            const uint64_t r = rt(), c = cyc();
            if (--rleft == 0) {
                rleft = rec_every;
                if (recw && n < maxrec) {
                    if (lane == 0) {
                        rec[ridx + n] = make_uint4((uint32_t) r, (uint32_t) (r >> 32), (uint32_t) c, (uint32_t) (c >> 32));
                    }
                    ++n;
                }
            }
            // The stop word as in gdense.comp: the line of the current 82 us REALTIME slot, a device-scope load from
            // host memory the GPU may cache (a system-scope load from every wave at every check went over PCIe each
            // time and starved the whole-card load: development run dev2-232334).
            if ((lim != 0 && r - r0 >= lim) ||
                __hip_atomic_load(stopw + (((uint32_t) (r >> 13) & 4095u) * 32u), __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT) != 0) {
                break;
            }
        }
    }
    const uint64_t r1 = rt(), c1 = cyc();
    if (w0 && lane == 0) {
        const uint32_t g = blockIdx.x;
        grp[3 * g] = make_uint4((uint32_t) r0, (uint32_t) (r0 >> 32), (uint32_t) c0, (uint32_t) (c0 >> 32));
        grp[3 * g + 1] = make_uint4((uint32_t) r1, (uint32_t) (r1 >> 32), (uint32_t) c1, (uint32_t) (c1 >> 32));
        grp[3 * g + 2] = make_uint4(done, n, 1, 0);
    }
    float s = 0;
    for (int i = 0; i < 16; ++i) {
        s += x[i];
    }
    // lane 0 of every wave: its chunk count (the load's density over all waves); other lanes: the chains' sum
    res[blockIdx.x * blockDim.x + threadIdx.x] = lane == 0 ? __uint_as_float(done) : s;
}

// Writes REALTIME into host memory (the host-to-REALTIME mapping, as sched_share's rt_offset).
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

int profile_mode() {
    std::ifstream f(paths().drm + "/pp_power_profile_mode");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find('*') != std::string::npos) {
            return std::atoi(line.c_str());
        }
    }
    return -1;
}

struct Row {
    long long t;
    Snap s;
    int prof;
};

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string out = opt(argc, argv, "--out", "dense_twin");
    const double idle_ms = std::atof(opt(argc, argv, "--idle-ms", "1000"));
    const double stop_after = std::atof(opt(argc, argv, "--stop-after-ms", "0"));
    const char * hs = opt(argc, argv, "--hot-stop-c", "");
    double hot_rel = 0;
    if (*hs) {
        hot_rel = std::atof(hs + (*hs == '+'));
        if (*hs != '+' || hot_rel <= 0 || hot_rel > 20) {
            std::printf("VERDICT dense_twin hot_stop_option: FAIL --hot-stop-c %s (only +N with N up to 20)\n", hs);
            return 2;
        }
        std::printf("INFO hotspot stop limit: the hotspot read before each launch + %.0f C (88 C outside the load)\n",
                    hot_rel);
    }
    std::vector<std::string> arms;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            ++i;
            continue;
        }
        arms.push_back(a);
    }
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("DEVICE %s %s wave %d CUs %d\n", prop.name, prop.gcnArchName, prop.warpSize, prop.multiProcessorCount);
    if (std::string(prop.gcnArchName).rfind("gfx1201", 0) != 0 || prop.warpSize != 32) {
        std::printf("VERDICT dense_twin device: FAIL not gfx1201 in wave32\n");
        return 1;
    }
    // 4,096 lines of 128 bytes, written by the host in full; write-combined so the GPU's reads see memory, not a CPU
    // cache, and non-coherent so the GPU may keep a line in L2 (as RADV's host-visible memory of the Vulkan probe).
    uint32_t * stopw;
    CHECK(hipHostMalloc(&stopw, 4096 * 128, hipHostMallocWriteCombined | hipHostMallocNonCoherent));
    auto stop_all = [&](uint32_t v) {
        for (int i = 0; i < 4096; ++i) {
            __atomic_store_n(stopw + i * 32, v, __ATOMIC_SEQ_CST);
        }
    };
    uint64_t * flag;
    CHECK(hipHostMalloc(&flag, 8, hipHostMallocCoherent));
    float * d_ab;
    CHECK(hipMalloc(&d_ab, 128 * 4));
    {
        std::vector<float> ab(128);
        for (int i = 0; i < 64; ++i) {
            ab[2 * i] = 0.999f;
            ab[2 * i + 1] = 0.001f;
        }
        CHECK(hipMemcpy(d_ab, ab.data(), 128 * 4, hipMemcpyHostToDevice));
    }
    int rc = 0;
    for (const std::string & spec : arms) {
        char name[64] = "";
        unsigned groups = 1, threads = 32, every = 64, rs = 8, re = 1;
        double ms = 60;
        if (std::sscanf(spec.c_str(), "%63[^,],%u,%u,%u,%lf,%u,%u", name, &groups, &threads, &every, &ms, &rs, &re) < 5 ||
            re == 0 ||
            threads % 32 || threads > 1024 || ms > 100) {
            std::printf("VERDICT dense_twin arm: FAIL bad arm '%s' (name,groups,threads,every,ms[,rec_stride]; ms at most "
                        "100)\n", spec.c_str());
            return 2;
        }
        // as the Vulkan probe: a favoured wave checks about every 0.7 us at every=8, every 5 us at every=64
        const uint32_t nrecw = rs ? (groups + rs - 1) / rs : 0,
                       maxrec = (uint32_t) (ms * 1000.0 * (every * re < 32 ? 2.0 : 0.5)) + 64;
        std::printf("ARM %s groups=%u threads=%u waves=%u every=%u ms=%.3f recorders=%u maxrec=%u idle_ms=%.0f\n", name,
                    groups, threads, groups * threads / 32, every, ms, nrecw, maxrec, idle_ms);
        if (hotspot() >= 58.0) {
            std::printf("THERMAL not starting %s: hotspot %.0f C (want below 58)\n", name, hotspot());
            return 1;
        }
        float * d_res;
        uint4 *d_rec, *d_grp;
        CHECK(hipMalloc(&d_res, (size_t) groups * threads * 4));
        CHECK(hipMalloc(&d_rec, (size_t) (nrecw ? nrecw : 1) * maxrec * 16));
        CHECK(hipMalloc(&d_grp, (size_t) groups * 48));
        CHECK(hipMemset(d_rec, 0, (size_t) (nrecw ? nrecw : 1) * maxrec * 16));
        CHECK(hipMemset(d_grp, 0, (size_t) groups * 48));
        // Warm the code path (a 10 us launch), then the mapping, then idle.
        stop_all(0);
        hipLaunchKernelGGL(k_dense, dim3(1), dim3(32), 0, 0, d_ab, d_res, d_rec, d_grp, stopw, 0x7fffffffu, 8u,
                           (uint64_t) 1000, 0u, 1u, 1u);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemset(d_rec, 0, (size_t) (nrecw ? nrecw : 1) * maxrec * 16));
        CHECK(hipMemset(d_grp, 0, (size_t) groups * 48));
        CHECK(hipDeviceSynchronize());
        const long long off0 = rt_offset(flag, 20);
        // The sampler: 5 ms, sets the stop word at 88 C; runs from before the idle to after the kernel.
        std::vector<Row> rows;
        rows.reserve(4096);
        std::atomic<bool> run{true}, hot{false};
        std::atomic<double> hottest_c{0}, limit_c{88.0};
        std::thread th([&] {
            const long long t0 = host_ns();
            int prof = -1;
            for (long i = 0; run; ++i) {
                Row r;
                r.t = host_ns();
                r.s = snap(0);
                if (i % 2 == 1) {
                    prof = profile_mode();
                }
                r.prof = prof;
                if (r.s.hot > hottest_c) {
                    hottest_c = r.s.hot;
                }
                if (r.s.hot >= limit_c) {
                    if (!hot) {
                        std::printf("THERMAL hotspot %.0f C: stopping the load (sampler's tripwire, limit %.0f C) at "
                                    "host_ns=%lld\n", r.s.hot, limit_c.load(), r.t);
                    }
                    hot = true;
                    stop_all(1);
                }
                rows.push_back(r);
                const long long next = t0 + (i + 1) * 5000000LL;
                while (host_ns() < next) {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            }
        });
        std::this_thread::sleep_for(std::chrono::microseconds((long long) (idle_ms * 1000)));
        if (hot_rel > 0) {
            limit_c = std::min(88.0, hotspot() + hot_rel);
            std::printf("INFO %s hotspot stop limit for this arm %.0f C\n", name, limit_c.load());
        }
        const long long tl = host_ns();
        hipLaunchKernelGGL(k_dense, dim3(groups), dim3(threads), 0, 0, d_ab, d_res, d_rec, d_grp, stopw, 0x7fffffffu,
                           (uint32_t) every, (uint64_t) std::llround(ms * 1e5), (uint32_t) rs, maxrec, (uint32_t) re);
        bool forced = false;
        if (stop_after > 0) {
            while (host_ns() < tl + (long long) (stop_after * 1e6)) {
            }
            stop_all(1);
            forced = true;
            std::printf("INFO %s stop word set by the host %.3f ms after the launch (known-answer test)\n", name,
                        (host_ns() - tl) / 1e6);
        }
        CHECK(hipDeviceSynchronize());
        const long long td = host_ns();
        limit_c = 88.0;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        run = false;
        th.join();
        const long long off1 = rt_offset(flag, 20);
        std::vector<uint4> hr((size_t) (nrecw ? nrecw : 1) * maxrec), hg((size_t) groups * 3);
        std::vector<uint32_t> hres((size_t) groups * threads);
        CHECK(hipMemcpy(hr.data(), d_rec, hr.size() * 16, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(hg.data(), d_grp, hg.size() * 16, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(hres.data(), d_res, hres.size() * 4, hipMemcpyDeviceToHost));
        CHECK(hipFree(d_res));
        CHECK(hipFree(d_rec));
        CHECK(hipFree(d_grp));
        std::printf("INFO %s launch to synchronize %.3f ms; stop word at the end %u (%s); host-to-REALTIME offset drift "
                    "%lld ns\n", name, (td - tl) / 1e6, *stopw,
                    forced ? "set by the host on purpose" : *stopw ? "set by the sampler's hotspot tripwire" : "never set",
                    off1 - off0);
        stop_all(0);
        const std::string base = out + "-dense-" + name;
        FILE * f = std::fopen((base + "-map.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "host_off0_ns,host_off1_ns,launch_host_ns,sync_host_ns\n%lld,%lld,%lld,%lld\n", off0, off1, tl,
                         td);
            std::fclose(f);
        }
        f = std::fopen((base + "-waves.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "burst,group,wave,chunks\n");
            for (unsigned g = 0; g < groups; ++g) {
                for (unsigned w = 0; w < threads / 32; ++w) {
                    std::fprintf(f, "0,%u,%u,%u\n", g, w, hres[(size_t) g * threads + w * 32]);
                }
            }
            std::fclose(f);
        }
        f = std::fopen((base + "-groups.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "burst,group,start_rt,start_cyc,end_rt,end_cyc,chunks,records\n");
            for (unsigned g = 0; g < groups; ++g) {
                const uint4 *w = &hg[3 * g];
                std::fprintf(f, "0,%u,%llu,%llu,%llu,%llu,%u,%u\n", g,
                             (unsigned long long) w[0].x | (unsigned long long) w[0].y << 32,
                             (unsigned long long) w[0].z | (unsigned long long) w[0].w << 32,
                             (unsigned long long) w[1].x | (unsigned long long) w[1].y << 32,
                             (unsigned long long) w[1].z | (unsigned long long) w[1].w << 32, w[2].x, w[2].y);
            }
            std::fclose(f);
        }
        if (nrecw) {
            f = std::fopen((base + "-rec.csv").c_str(), "w");
            if (f) {
                std::fprintf(f, "burst,recorder,group,i,rt,cyc\n");
                for (uint32_t r = 0; r < nrecw; ++r) {
                    for (uint32_t k = 0; k < maxrec; ++k) {
                        const uint4 & w = hr[(size_t) r * maxrec + k];
                        if (!(w.x | w.y)) {
                            break;
                        }
                        std::fprintf(f, "0,%u,%u,%u,%llu,%llu\n", r, r * rs, k,
                                     (unsigned long long) w.x | (unsigned long long) w.y << 32,
                                     (unsigned long long) w.z | (unsigned long long) w.w << 32);
                    }
                }
                std::fclose(f);
            }
        }
        f = std::fopen((base + "-sysfs.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "t_host_ns,sclk_MHz,busy_pct,hotspot_C,power_W_reported,indep_throttle_status,vgfx_mV,"
                            "power_profile\n");
            for (const Row & r : rows) {
                std::fprintf(f, "%lld,%d,%d,%.1f,%.0f,0x%llx,%d,%d\n", r.t, r.s.sclk, r.s.busy, r.s.hot, r.s.power,
                             (unsigned long long) r.s.ithr, r.s.vgfx_mv, r.prof);
            }
            std::fclose(f);
        }
        std::printf("THERMAL %s highest %.0f C (5 ms samples)\n", name, hottest_c.load());
        {
            // the thermal rule asks for a hotspot reading every 5 ms; a busy host stretches the sampler
            std::vector<double> iv;
            for (size_t i = 1; i < rows.size(); ++i) {
                iv.push_back((rows[i].t - rows[i - 1].t) / 1e6);
            }
            const double p95 = iv.empty() ? 1e9 : pct(iv, 0.95), mx = iv.empty() ? 1e9 : pct(iv, 1.0);
            std::printf("VERDICT dense_twin sampler_interval: %s %s: interval p95 %.2f ms (limit 6), largest %.2f ms "
                        "(limit 20), %zu samples\n", p95 <= 6 && mx <= 20 ? "PASS" : "FAIL", name, p95, mx, rows.size());
            if (!(p95 <= 6 && mx <= 20)) {
                rc = 1;
            }
        }
        note_hot(hottest_c.load());
        if (hot) {
            // a stop at the lowered limit is the F5 test's expected outcome; at 88 C it ends the run as a failure
            if (hot_rel > 0 && hottest_c.load() < 88.0) {
                std::printf("INFO thermal: %s was stopped at the lowered limit (--hot-stop-c, the stop path's test); "
                            "highest %.0f C\n", name, hottest_c.load());
            } else {
                rc = 1;
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::printf("HOTTEST %.0f C\n", hottest().load());
    return rc;
}
