// Shared parts of the timing-leftover probes (tim_*.hip, scripts/study-leftovers-timing.sh).
//
// Every timed region is one inline-assembly statement that reads HW_REG_SHADER_CYCLES_LO at its start and end, so
// nothing the compiler adds can land inside it (the method of mat_share.hip). The 32-bit low half is enough: no
// region here lasts 2^32 cycles. Registers are named explicitly (v16 upward for the timed data, s[40:47] for
// scalar work) so the instruction text and the register banks are fixed by the source, not by the allocator.
//
// Disturbance rule (lane 7, measurements/scheduling-and-clocks.md item 7): a sample above 1.5 x the median sample
// + 64 cycles is disturbed. Every probe applies it to its own samples and reports how many it dropped.
#pragma once
#include "mem_common.h"

#include <cmath>
#include <cstdarg>
#include <string>
#include <vector>

namespace tim {

inline int & failures() {
    static int n = 0;
    return n;
}

inline void check(const std::string & name, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
inline void check(const std::string & name, bool ok, const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    verdict(name.c_str(), ok ? "PASS" : "FAIL", buf);
    if (!ok) {
        ++failures();
    }
}

inline bool flag(int argc, char ** argv, const char * f) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], f) == 0) {
            return true;
        }
    }
    return false;
}

inline double median_of(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Samples split by the disturbance rule.
struct Split {
    double median_all, worst_all, mean_all;
    double median_clean, worst_clean, mean_clean;
    double best;
    size_t n, dropped;
    double excess_median;  // median of (sample - clean median) over the dropped samples
};

inline Split split(const std::vector<double> & v) {
    Split s{};
    s.n = v.size();
    if (v.empty()) {
        return s;
    }
    s.median_all = median_of(v);
    const double thr = 1.5 * s.median_all + 64;
    std::vector<double> clean, excess;
    double sum = 0, sumc = 0;
    s.best = v[0];
    for (double x : v) {
        sum += x;
        s.worst_all = std::max(s.worst_all, x);
        s.best = std::min(s.best, x);
        if (x > thr) {
            excess.push_back(x);
        } else {
            clean.push_back(x);
            sumc += x;
            s.worst_clean = std::max(s.worst_clean, x);
        }
    }
    s.mean_all = sum / v.size();
    s.dropped = excess.size();
    s.median_clean = median_of(clean);
    s.mean_clean = clean.empty() ? NAN : sumc / clean.size();
    for (double & x : excess) {
        x -= s.median_clean;
    }
    s.excess_median = median_of(excess);
    return s;
}

// Median of the clean samples (rule applied), with the count dropped; the usual statistic of these probes.
struct Clean {
    double med, lo, hi;
    size_t n, dropped;
};

inline Clean clean_of(const std::vector<double> & v) {
    const Split s = split(v);
    double lo = s.worst_clean, hi = 0;
    const double thr = 1.5 * s.median_all + 64;
    for (double x : v) {
        if (x <= thr) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    }
    return {s.median_clean, lo, hi, s.n, s.dropped};
}

inline uint32_t * dev_u32(size_t n) {
    uint32_t * p;
    CHECK(hipMalloc(&p, n * 4));
    CHECK(hipMemset(p, 0, n * 4));
    return p;
}

inline std::vector<uint32_t> host_u32(const uint32_t * d, size_t n) {
    std::vector<uint32_t> h(n);
    CHECK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
    return h;
}

}  // namespace tim

#define TIM_STR2(x) #x
#define TIM_STR(x) TIM_STR2(x)

// 16 independent FMA chains in v16..v31 with inline constants (one VGPR read per instruction, 8 bytes each): one
// FMA per cycle (lane 1's calibration, sp_fma3nolit). The same with the constants in v32/v33 (TIM_FMA16_VV) costs
// 17 cycles per 16 (measured here, item 3), so it is not a calibration.
#define TIM_FMA16 "v_fma_f32 v16, v16, 0.5, 0.5\nv_fma_f32 v17, v17, 0.5, 0.5\nv_fma_f32 v18, v18, 0.5, 0.5\nv_fma_f32 v19, v19, 0.5, 0.5\nv_fma_f32 v20, v20, 0.5, 0.5\nv_fma_f32 v21, v21, 0.5, 0.5\nv_fma_f32 v22, v22, 0.5, 0.5\nv_fma_f32 v23, v23, 0.5, 0.5\nv_fma_f32 v24, v24, 0.5, 0.5\nv_fma_f32 v25, v25, 0.5, 0.5\nv_fma_f32 v26, v26, 0.5, 0.5\nv_fma_f32 v27, v27, 0.5, 0.5\nv_fma_f32 v28, v28, 0.5, 0.5\nv_fma_f32 v29, v29, 0.5, 0.5\nv_fma_f32 v30, v30, 0.5, 0.5\nv_fma_f32 v31, v31, 0.5, 0.5\n"
#define TIM_FMA16_VV "v_fma_f32 v16, v16, v32, v33\nv_fma_f32 v17, v17, v32, v33\nv_fma_f32 v18, v18, v32, v33\nv_fma_f32 v19, v19, v32, v33\nv_fma_f32 v20, v20, v32, v33\nv_fma_f32 v21, v21, v32, v33\nv_fma_f32 v22, v22, v32, v33\nv_fma_f32 v23, v23, v32, v33\nv_fma_f32 v24, v24, v32, v33\nv_fma_f32 v25, v25, v32, v33\nv_fma_f32 v26, v26, v32, v33\nv_fma_f32 v27, v27, v32, v33\nv_fma_f32 v28, v28, v32, v33\nv_fma_f32 v29, v29, v32, v33\nv_fma_f32 v30, v30, v32, v33\nv_fma_f32 v31, v31, v32, v33\n"

// Binds float r[18] to v16..v33, so the values stay in those registers from one statement to the next. Set r[0..15]
// to 1.0 and r[16], r[17] to 0.5 (x * 0.5 + 0.5 stays finite).
#define TIM_FMA_REGS(r) \
    "+{v16}"(r[0]), "+{v17}"(r[1]), "+{v18}"(r[2]), "+{v19}"(r[3]), "+{v20}"(r[4]), "+{v21}"(r[5]), "+{v22}"(r[6]), "+{v23}"(r[7]), "+{v24}"(r[8]), "+{v25}"(r[9]), "+{v26}"(r[10]), "+{v27}"(r[11]), "+{v28}"(r[12]), "+{v29}"(r[13]), "+{v30}"(r[14]), "+{v31}"(r[15]), "+{v32}"(r[16]), "+{v33}"(r[17])
