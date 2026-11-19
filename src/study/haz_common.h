// Lane 8 (compiler and hazards): shared helpers. One 16x16x16 FP16-into-FP32 matrix problem per wave, host-side
// register images and the CPU reference (the bit-exact model of src/study/arith_model.h).
//
// Register images (gfx1201, wave32, from src/study/arith_gpu.h):
//   A row r, slot s (0..15): lane r + 16 * (s >> 3), half (s & 7) of the lane's 8-half vector.
//   B column c, slot s:      lane c + 16 * (s >> 3), half (s & 7).
//   D (r, c):                lane 16 * (r >> 3) + c, float (r & 7).
#pragma once
#include "../harness.h"
#include "arith_model.h"

#include <cmath>
#include <cstdarg>
#include <string>
#include <vector>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

namespace haz {

inline uint16_t h16(float f) {
    const _Float16 h = (_Float16) f;
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
inline uint32_t f32b(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float bf32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Per-lane images: a[32*8], b[32*8] halves, c[32*8] floats (bits).
struct Problem {
    std::vector<uint16_t> a, b;
    std::vector<uint32_t> c;
    Problem() : a(256), b(256), c(256) {}
};

// Small integers: every product and sum is exact, so any rounding question stays out and the model is exact too.
inline Problem make_problem(uint32_t seed) {
    Problem p;
    uint32_t x = seed * 2654435761u + 12345u;
    auto rnd = [&]() {
        x = x * 1664525u + 1013904223u;
        return (int) ((x >> 16) % 7) - 3;
    };
    for (int i = 0; i < 256; ++i) {
        p.a[i] = h16((float) rnd());
        p.b[i] = h16((float) rnd());
        p.c[i] = f32b((float) rnd());
    }
    return p;
}

// D image (per lane, 8 floats) that the instruction must produce from the register images, all 32 lanes' operands.
inline std::vector<uint32_t> reference(const Problem & p, int form = am::F_F16) {
    const am::Params P = am::measured_params(form);
    std::vector<uint32_t> d(256);
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            uint32_t as[16], bs[16];
            for (int s = 0; s < 16; ++s) {
                as[s] = p.a[(r + 16 * (s >> 3)) * 8 + (s & 7)];
                bs[s] = p.b[(c + 16 * (s >> 3)) * 8 + (s & 7)];
            }
            const int lane = 16 * (r >> 3) + c, el = r & 7;
            d[lane * 8 + el] = am::model_f32(form, as, bs, p.c[lane * 8 + el], P);
        }
    }
    return d;
}

// Lanes that own any of the D elements a lane's operands feed: lane l's A (row l&15) and B (column l&15).
inline bool d_depends_on_lane(int dlane, int el, int srclane) {
    const int r = 8 * (dlane >> 4) + el, c = dlane & 15;
    return (srclane & 15) == r || (srclane & 15) == c;
}

template <class T>
T * dev_copy(const std::vector<T> & h) {
    T * d = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> host_copy(const T * d, size_t n) {
    std::vector<T> h(n);
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

struct Verdicts {
    int fails = 0;
    void put(const char * name, bool ok, const std::string & text) {
        verdict(name, ok ? "PASS" : "FAIL", text.c_str());
        if (!ok) {
            ++fails;
        }
    }
    void info(const char * name, const std::string & text) { verdict(name, "INFO", text.c_str()); }
    int exit_code() const { return fails ? 1 : 0; }
};

inline std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char * f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace haz
