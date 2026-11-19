// Shared parts of the lane 7 probes (scheduling, launch, clocks): device-side counter and placement reads, host-side
// read-only sysfs access (clocks, reported power, temperatures, fan, busy), a sampler thread with the 95 C stop, and
// the hotspot gate. Build with -I$STUDY_REPO/src -I$STUDY_REPO/src/study (harness.h comes from the study repository).
#pragma once
#include "harness.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace sched {

// ---------------------------------------------------------------------------------------------------------------
// Device side

__device__ __forceinline__ uint64_t rt() {
    uint64_t t;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0" : "=s"(t)::"memory");
    return t;
}
// HI, LO, HI; retried until HI is stable so the pair is one 64-bit value (the counter is per SIMD).
__device__ __forceinline__ uint64_t cyc() {
    uint32_t h1, l, h2;
    do {
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)" : "=s"(h1)::"memory");
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_LO)" : "=s"(l)::"memory");
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)" : "=s"(h2)::"memory");
    } while (h1 != h2);
    return ((uint64_t) h1 << 32) | l;
}
__device__ __forceinline__ uint32_t hwid1() {
    uint32_t r;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(r)::"memory");
    return r;
}
__device__ __forceinline__ uint32_t se_aid() {
    uint32_t r;
    asm volatile("s_sendmsg_rtn_b32 %0, sendmsg(MSG_RTN_GET_SE_AID_ID)\n\ts_wait_kmcnt 0" : "=s"(r)::"memory");
    return r;
}
// Packed placement, same layout as the study's mem::placement: bits 3:0 SE, 4 SA, 8:5 WGP, 9 CU in WGP,
// 10 SIMD in CU, 15:11 wave slot.
__device__ __forceinline__ uint32_t placement() {
    const uint32_t id1 = hwid1(), se = se_aid();
    const uint32_t slot = id1 & 31u, simd = (id1 >> 8) & 3u, wgp = (id1 >> 10) & 15u, sa = (id1 >> 16) & 1u;
    return (se & 15u) | (sa << 4) | (wgp << 5) | ((simd & 1u) << 9) | ((simd >> 1) << 10) | (slot << 11);
}

// ---------------------------------------------------------------------------------------------------------------
// Host side: placement fields

inline int pl_se(uint32_t p) { return p & 15; }
inline int pl_sa(uint32_t p) { return (p >> 4) & 1; }
inline int pl_wgp(uint32_t p) { return (p >> 5) & 15; }
inline int pl_cuinwgp(uint32_t p) { return (p >> 9) & 1; }
inline int pl_simdincu(uint32_t p) { return (p >> 10) & 1; }
inline int pl_slot(uint32_t p) { return (p >> 11) & 31; }
inline uint32_t pl_wgpkey(uint32_t p) { return p & 0x1FFu; }  // SE, SA, WGP
inline uint32_t pl_cu(uint32_t p) { return p & 0x3FFu; }       // + CU in WGP
inline uint32_t pl_simd(uint32_t p) { return p & 0x7FFu; }     // + SIMD in CU

// ---------------------------------------------------------------------------------------------------------------
// Host side: read-only sysfs

inline std::string slurp(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
inline double read_num(const std::string & path, double scale) {
    std::ifstream f(path);
    double v;
    if (!(f >> v)) {
        return -1;
    }
    return v * scale;
}
// MHz of the '*' line of pp_dpm_sclk / pp_dpm_mclk.
inline int star_mhz(const std::string & path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find('*') != std::string::npos) {
            const size_t c = line.find(':');
            return std::atoi(line.c_str() + (c == std::string::npos ? 0 : c + 1));
        }
    }
    return -1;
}

struct Paths {
    std::string drm, hw;
};
// DRM_DEVICE and HWMON from the environment (the script exports them from the study's card record).
inline const Paths & paths() {
    static Paths p = [] {
        Paths q;
        const char * d = std::getenv("DRM_DEVICE");
        const char * h = std::getenv("HWMON");
        q.drm = d ? d : "";
        q.hw = h ? h : "";
        return q;
    }();
    return p;
}

struct Snap {
    double t;  // seconds since the sampler started
    int sclk, mclk;
    double power, edge, hot, memt, fan;
    int busy;
    // From gpu_metrics (format 1, content 3, 120 bytes; layout checked against hwmon and pp_dpm_sclk on this card):
    // throttle_status (ASIC-specific bits), indep_throttle_status (driver's ASIC-independent bits), gfx voltage.
    uint32_t thr;
    uint64_t ithr;
    int vgfx_mv;
};

inline bool read_metrics(Snap & s) {
    std::ifstream f(paths().drm + "/gpu_metrics", std::ios::binary);
    unsigned char b[120];
    if (!f.read(reinterpret_cast<char *>(b), sizeof b)) {
        return false;
    }
    if (b[0] != 120 || b[1] != 0 || b[2] != 1 || b[3] != 3) {
        return false;
    }
    std::memcpy(&s.thr, b + 68, 4);
    std::memcpy(&s.ithr, b + 112, 8);
    uint16_t v;
    std::memcpy(&v, b + 106, 2);
    s.vgfx_mv = v;
    return true;
}

// Names of the driver's ASIC-independent throttler bits (SMU_THROTTLER_*_BIT in amdgpu_smu.h, compared with Linux
// master on 2026-11-17; the raw value is always printed next to the names).
inline std::string throttle_names(uint64_t m) {
    static const std::pair<int, const char *> k[] = {
        {0, "PPT0"},        {1, "PPT1"},        {2, "PPT2"},       {3, "PPT3"},        {4, "SPL"},
        {5, "FPPT"},        {6, "SPPT"},        {7, "SPPT_APU"},   {16, "TDC_GFX"},    {17, "TDC_SOC"},
        {18, "TDC_MEM"},    {19, "TDC_VDD"},    {20, "TDC_CVIP"},  {21, "EDC_CPU"},    {22, "EDC_GFX"},
        {23, "APCC"},       {32, "TEMP_GPU"},   {33, "TEMP_CORE"}, {34, "TEMP_MEM"},   {35, "TEMP_EDGE"},
        {36, "TEMP_HOTSPOT"}, {37, "TEMP_SOC"}, {38, "TEMP_VR_GFX"}, {39, "TEMP_VR_SOC"}, {40, "TEMP_VR_MEM0"},
        {41, "TEMP_VR_MEM1"}, {42, "TEMP_LIQUID0"}, {43, "TEMP_LIQUID1"}, {44, "VRHOT0"}, {45, "VRHOT1"},
        {46, "PROCHOT_CPU"}, {47, "PROCHOT_GFX"}, {56, "PPM"},     {57, "FIT"}};
    std::string r;
    for (auto & e : k) {
        if (m & (1ull << e.first)) {
            r += (r.empty() ? "" : "+");
            r += e.second;
        }
    }
    return r.empty() ? "none" : r;
}

inline Snap snap(double t) {
    const Paths & p = paths();
    Snap s{};
    s.t = t;
    s.sclk = star_mhz(p.drm + "/pp_dpm_sclk");
    s.mclk = star_mhz(p.drm + "/pp_dpm_mclk");
    s.busy = (int) read_num(p.drm + "/gpu_busy_percent", 1);
    s.power = read_num(p.hw + "/power1_average", 1e-6);
    s.edge = read_num(p.hw + "/temp1_input", 1e-3);
    s.hot = read_num(p.hw + "/temp2_input", 1e-3);
    s.memt = read_num(p.hw + "/temp3_input", 1e-3);
    s.fan = read_num(p.hw + "/fan1_input", 1);
    read_metrics(s);
    return s;
}
inline double hotspot() { return read_num(paths().hw + "/temp2_input", 1e-3); }

inline std::atomic<double> & hottest() {
    static std::atomic<double> h{0};
    return h;
}
inline void note_hot(double t) {
    double cur = hottest().load();
    while (t > cur && !hottest().compare_exchange_weak(cur, t)) {
    }
}

// Waits until the hotspot is below `below` C (at most max_s seconds); returns false if it did not get there.
inline bool gate(double below, double max_s = 180) {
    const auto t0 = std::chrono::steady_clock::now();
    const double h0 = hotspot();
    double h = h0;
    while ((h = hotspot()) >= below) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > max_s) {
            std::printf("THERMAL gate: hotspot still %.0f C after %.0f s (wanted below %.0f)\n", h, max_s, below);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (h0 >= below) {
        std::printf("THERMAL gate: waited %.1f s, %.0f C -> %.0f C\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), h0, h);
    }
    return true;
}

// Samples every period_ms from a host thread. At 95 C it sets `hot`; the load loop must check it between launches
// (each launch is kept to about 100 ms or less). A hotspot of 100 C ends the process at once as a last resort.
class Sampler {
  public:
    explicit Sampler(int period_ms = 50) : period_(period_ms) {}
    ~Sampler() {
        if (th_.joinable()) {
            stop();
        }
    }
    void start() {
        run_ = true;
        t0_ = std::chrono::steady_clock::now();
        th_ = std::thread([this] {
            while (run_) {
                const Snap s = snap(now());
                if (s.hot > 0) {
                    note_hot(s.hot);
                    if (s.hot >= 95.0) {
                        hot = true;
                    }
                    if (s.hot >= 100.0) {
                        std::printf("THERMAL hotspot %.0f C: ending the process\n", s.hot);
                        std::fflush(stdout);
                        std::_Exit(4);
                    }
                }
                {
                    std::lock_guard<std::mutex> g(mu_);
                    rows.push_back(s);
                }
                std::this_thread::sleep_until(t0_ + std::chrono::milliseconds(period_ * (long) rows.size()));
            }
        });
    }
    void stop() {
        run_ = false;
        th_.join();
    }
    double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
    std::vector<Snap> copy() {
        std::lock_guard<std::mutex> g(mu_);
        return rows;
    }
    std::atomic<bool> hot{false};
    std::vector<Snap> rows;

  private:
    int period_;
    std::atomic<bool> run_{false};
    std::thread th_;
    std::mutex mu_;
    std::chrono::steady_clock::time_point t0_;
};

inline void print_snap_header(FILE * f) {
    std::fprintf(f, "t_s,sclk_MHz,mclk_MHz,power_W_reported,edge_C,hotspot_C,mem_C,fan_rpm,busy_pct,vgfx_mV,"
                    "throttle_status,indep_throttle_status\n");
}
inline void print_snap(FILE * f, const Snap & s) {
    std::fprintf(f, "%.3f,%d,%d,%.0f,%.1f,%.1f,%.1f,%.0f,%d,%d,0x%x,0x%llx\n", s.t, s.sclk, s.mclk, s.power, s.edge,
                 s.hot, s.memt, s.fan, s.busy, s.vgfx_mv, s.thr, (unsigned long long) s.ithr);
}

// Percentile of a copy (p in 0..1, nearest rank).
inline double pct(std::vector<double> v, double p) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    size_t i = (size_t) std::llround(p * (double) (v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}
inline std::string dist(const std::vector<double> & v, const char * fmt = "%.2f") {
    char b[256], f[160];
    std::snprintf(f, sizeof f, "median %s p10 %s p90 %s min %s max %s n %%zu", fmt, fmt, fmt, fmt, fmt);
    std::snprintf(b, sizeof b, f, pct(v, 0.5), pct(v, 0.1), pct(v, 0.9), pct(v, 0.0), pct(v, 1.0), v.size());
    return b;
}

inline bool has_flag(int argc, char ** argv, const char * f) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return true;
        }
    }
    return false;
}
inline const char * opt(int argc, char ** argv, const char * f, const char * def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

}  // namespace sched
