// CPU-only: cr::encode8 for BF8 (E5M2) against a search over the sorted code values, on ALL 2^32 FP32 inputs, in
// every mode cvt_refcheck checks (rounding nearest even / up / down / toward zero, saturating or not). The search
// shares no code with cvt_ref.h: it finds the neighbouring code values of |x| in a sorted table and decides by
// comparing distances. The floor index is found once per input and shared by the eight modes.
// Usage: tim_bf8search [--break]. --break makes the reference under test round ties away from zero; the check must
// then fail.
#include "cvt_ref.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static float fl(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// Positive BF8 values in code order 0..0x7B, plus the next value the format would have.
static std::vector<double> grid() {
    std::vector<double> v;
    for (int c = 0; c <= 0x7B + 1; ++c) {
        const int e = c >> 2, m = c & 3;
        v.push_back(e == 0 ? std::ldexp((double) m, 1 - 15 - 2) : std::ldexp((double) (m + 4), e - 15 - 2));
    }
    return v;
}

struct Search {
    std::vector<double> v;
    uint32_t maxcode;
    Search() : v(grid()), maxcode((uint32_t) v.size() - 2) {}

    // Floor index of |x| in the table (finite x only).
    size_t floor_index(double ax) const { return (size_t) (std::upper_bound(v.begin(), v.end(), ax) - v.begin()) - 1; }

    uint32_t code(size_t lo, double ax, bool neg, bool sat, int rnd) const {
        const uint32_t over = 0x7Cu;
        uint32_t c;
        if (lo >= v.size() - 1) {
            const bool toward_zero = rnd == cr::RTZ || (rnd == cr::RUP && neg) || (rnd == cr::RDN && !neg);
            c = toward_zero ? maxcode : maxcode + 1;
        } else if (v[lo] == ax) {
            c = (uint32_t) lo;
        } else {
            const size_t hi = lo + 1;
            bool up;
            if (rnd == cr::RNE) {
                const double dl = ax - v[lo], dh = v[hi] - ax;
                up = dh < dl || (dh == dl && (hi & 1) == 0);
            } else if (rnd == cr::RTZ) {
                up = false;
            } else {
                up = (rnd == cr::RUP) != neg;
            }
            c = (uint32_t) (up ? hi : lo);
        }
        if (c > maxcode) {
            c = sat ? maxcode : over;
        }
        return (neg ? 0x80u : 0u) | c;
    }
};

int main(int argc, char ** argv) {
    const bool ta = argc > 1 && std::strcmp(argv[1], "--break") == 0;
    const Search S;
    std::atomic<unsigned long long> bad[8] = {};
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t) {
        th.emplace_back([t, ta, &S, &bad] {
            unsigned long long b[8] = {};
            for (uint64_t x = (uint64_t) t << 29; x < (uint64_t) (t + 1) << 29; ++x) {
                const uint32_t bits = (uint32_t) x;
                const bool neg = bits >> 31;
                const uint32_t a = bits & 0x7FFFFFFFu;
                const bool special = a >= 0x7F800000u;
                size_t lo = 0;
                if (!special) {
                    lo = S.floor_index(std::fabs((double) fl(bits)));
                }
                for (int sat = 0; sat < 2; ++sat) {
                    for (int rnd = 0; rnd < 4; ++rnd) {
                        uint32_t want;
                        if (a > 0x7F800000u) {
                            want = (neg ? 0x80u : 0u) | 0x7Eu;
                        } else if (a == 0x7F800000u) {
                            want = (neg ? 0x80u : 0u) | (sat ? S.maxcode : 0x7Cu);
                        } else {
                            want = S.code(lo, std::fabs((double) fl(bits)), neg, sat, rnd);
                        }
                        const uint32_t got = cr::encode8(cr::BF8, bits, sat, rnd, ta && rnd == cr::RNE);
                        b[sat * 4 + rnd] += want != got;
                    }
                }
            }
            for (int i = 0; i < 8; ++i) {
                bad[i] += b[i];
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    static const char * const rn[4] = {"nearest-even", "up", "down", "toward-zero"};
    unsigned long long total = 0;
    for (int sat = 0; sat < 2; ++sat) {
        for (int rnd = 0; rnd < 4; ++rnd) {
            const unsigned long long m = bad[sat * 4 + rnd];
            std::printf("RESULT BF8 encode vs search, all 2^32 inputs, sat %d, %-12s: %llu mismatches\n", sat, rn[rnd], m);
            total += m;
        }
    }
    std::printf("RESULT wall time %.1f s on 8 threads\n", secs);
    std::printf("VERDICT tim_bf8search: %s BF8 encode8 vs sorted-value search, 2^32 inputs x 8 modes, %llu mismatches%s\n",
                total ? "FAIL" : "PASS", total, ta ? " (--break: ties away)" : "");
    return total ? 1 : 0;
}
