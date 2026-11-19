// Lane 6: GPU-side helpers shared by the conversion and dot-product probes: MODE register writes with a vector-ALU
// control that shows the write took effect, and the settings the probes sweep.
#pragma once
#include "harness.h"
#include "cvt_ref.h"

#include <string>

namespace cvt {

// MODE fields replaced by a setting: FP_ROUND [3:0], FP_DENORM [7:4], FP16_OVFL [23].
constexpr uint32_t MODE_MASK = 0x008000FFu;

struct Setting {
    const char * name;
    bool write;
    uint32_t mode;  // value of the masked bits
};
// Rounding fields are set together (FP32 [1:0] and FP16/FP64 [3:2]); 0xF0 keeps denormals for both.
static const Setting SETTINGS[] = {
    {"plain", false, 0},       {"ovfl", true, 0xF0u | (1u << 23)}, {"rup", true, 0xF5u},
    {"rdn", true, 0xFAu},      {"rtz", true, 0xFFu},              {"noden", true, 0x00u},
    {"ovfl_rtz", true, 0xFFu | (1u << 23)},
};
constexpr int NSETTINGS = sizeof(SETTINGS) / sizeof(SETTINGS[0]);

inline int setting_index(const std::string & s) {
    for (int i = 0; i < NSETTINGS; ++i) {
        if (s == SETTINGS[i].name) {
            return i;
        }
    }
    return -1;
}

__device__ __forceinline__ uint32_t mode_set(uint32_t mode) {
    uint32_t old, nw;
    asm volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_MODE)\n"
        "s_and_b32 %1, %0, %3\n"
        "s_or_b32 %1, %1, %2\n"
        "s_setreg_b32 hwreg(HW_REG_MODE), %1\n"
        "s_nop 7\ns_nop 7"
        : "=&s"(old), "=&s"(nw)
        : "s"(mode & MODE_MASK), "s"(~MODE_MASK)
        : "scc");
    return old;
}

__device__ __forceinline__ void mode_restore(uint32_t old) {
    asm volatile("s_nop 7\ns_setreg_b32 hwreg(HW_REG_MODE), %0\ns_nop 7" ::"s"(old));
}

__device__ __forceinline__ uint32_t mode_read() {
    uint32_t m;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE)" : "=s"(m));
    return m;
}

// Control results: [0] 1 + 2^-24 (differs under round up), [1] -1 - 2^-24 (differs under round down),
// [2] 1 + 3 * 2^-25 (differs under round toward zero), [3] 2^-126 * 0.5 (0 when FP32 denormals are flushed),
// [4] v_add_f16 65504 + 65504 (0x7BFF instead of 0x7C00 when FP16_OVFL clamps, or under rounding down or
// toward zero as IEEE has it), [5] MODE as read back.
constexpr int NCTRL = 6;

__device__ __forceinline__ void control(uint32_t * out) {
    float a = 1.0f, b = 0x1p-24f, c = -1.0f, d = -0x1p-24f, e = 1.0f, f = 0x3p-25f, g = 0x1p-126f, h = 0.5f;
    uint32_t hx = 0x7BFFu, hy = 0x7BFFu, r0, r1, r2, r3, r4;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(r0) : "v"(a), "v"(b));
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(r1) : "v"(c), "v"(d));
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(r2) : "v"(e), "v"(f));
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r3) : "v"(g), "v"(h));
    asm volatile("v_add_f16 %0, %1, %2" : "=v"(r4) : "v"(hx), "v"(hy));
    out[0] = r0;
    out[1] = r1;
    out[2] = r2;
    out[3] = r3;
    out[4] = r4 & 0xFFFFu;
    out[5] = mode_read();
}

// What the control must show for a setting, given the MODE value the kernel was launched with.
inline bool control_ok(const Setting & s, const uint32_t * c, uint32_t launch_mode, std::string & why) {
    const uint32_t mode = s.write ? ((launch_mode & ~MODE_MASK) | s.mode) : launch_mode;
    const int rnd = mode & 3, rnd16 = (mode >> 2) & 3;
    const bool den = ((mode >> 4) & 3) == 3, ovfl = (mode >> 23) & 1;
    const uint32_t want[5] = {rnd == 1 ? 0x3F800001u : 0x3F800000u, rnd == 2 ? 0xBF800001u : 0xBF800000u,
                              rnd == 0 ? 0x3F800001u : (rnd == 1 ? 0x3F800001u : 0x3F800000u),
                              den ? 0x00400000u : 0u,
                              (ovfl || rnd16 >= 2) ? 0x7BFFu : 0x7C00u};
    bool ok = c[5] == mode;
    char buf[200];
    std::snprintf(buf, sizeof buf, "MODE read 0x%08x (want 0x%08x); control 0x%08x 0x%08x 0x%08x 0x%08x 0x%04x", c[5],
                  mode, c[0], c[1], c[2], c[3], c[4]);
    why = buf;
    for (int i = 0; i < 5; ++i) {
        if (c[i] != want[i]) {
            ok = false;
            std::snprintf(buf, sizeof buf, " [control %d want 0x%08x]", i, want[i]);
            why += buf;
        }
    }
    return ok;
}

}  // namespace cvt
