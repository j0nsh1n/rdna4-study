// Shared parts of the lane-3 probes (notes/STUDY-PLAN.md lane 3: LDS, scalar loads, special loads).
//
// Reuses lane 1's cycle method (core_common.h: one wave alone on its SIMD, differential cost of U = 8 against U = 4
// blocks) for the calibration, and lane 2's device and host helpers (mem_common.h: placement, clocks, hotspot).
// Every probe calls lds::calibrate() first: 16 independent v_fma_f32 must cost 1.00 cycle each (lane 1's answer).
#pragma once
#include "core_common.h"
#include "mem_common.h"

#include <chrono>
#include <cstdarg>
#include <thread>

#define LDS_P_FMA(op, i) op " %" #i ", %" #i ", %16, %17\n"

CORE_KERNEL(lds_calib_k, float, float, "+v", "v", CORE_B16(LDS_P_FMA, "v_fma_f32"), "")

namespace lds {

inline int & failures() {
    static int f = 0;
    return f;
}

// VERDICT line; a FAIL is counted so main can return 1.
inline void check(const char * name, bool ok, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    verdict(name, ok ? "PASS" : "FAIL", buf);
    if (!ok) {
        ++failures();
    }
}

// Known answer: 1.00 cycle per independent FMA. brk claims 2.00, so the verdict must fail.
inline double calibrate(bool brk) {
    core::set_init({core::f32(1.0f), core::f32(1.0f), core::f32(0.0f)});
    const core::Cost c = core::pattern_cost(lds_calib_k<core::ULO>, lds_calib_k<core::UMID>, lds_calib_k<core::UHI>);
    const double want = brk ? 2.0 : 1.0;
    std::printf("RESULT calib fma cycles %.3f lo %.3f hi %.3f mid_dev %.2f\n", c.per, c.lo, c.hi, c.mid);
    check("calib", std::fabs(c.per - want) <= 0.06 * want && std::fabs(c.mid) < 1.5,
          "independent v_fma_f32 %.3f cycles (expected %.2f; lane 1: 1.00), U=6 off midpoint by %.2f cycles/iter", c.per,
          want, c.mid);
    return c.per;
}

// Thermal rule of this lane for whole-card jobs: start below 65 C; give up (and fail) at 95 C.
inline bool thermal_gate(const char * tag) {
    double t = mem::hotspot_c();
    mem::hottest() = std::max(mem::hottest(), t);
    if (t >= 95) {
        std::printf("THERMAL %s: %.0f C, stopping\n", tag, t);
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (t >= 65) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        t = mem::hotspot_c();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(120)) {
            std::printf("THERMAL %s: still %.0f C after 120 s\n", tag, t);
            return false;
        }
    }
    return true;
}

inline void after_full(const char * tag) {
    const double t = mem::hotspot_c();
    mem::hottest() = std::max(mem::hottest(), t);
    if (t >= 95) {
        std::printf("THERMAL %s: reached %.0f C\n", tag, t);
    }
}

inline int finish() {
    mem::report_hottest();
    return failures() ? 1 : 0;
}

}  // namespace lds
