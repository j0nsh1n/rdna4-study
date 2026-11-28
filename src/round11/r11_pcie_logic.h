// Round 11 item 4b: the probe's arithmetic and its controls, with no HIP and no card, so a CPU test can break each one on purpose.
// Used by r11_pcie.hip (the probe) and r11_pcie_logic_test.cpp (CPU only).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef __HIPCC__
#define R11_HD __host__ __device__
#else
#define R11_HD
#endif

namespace r11 {

// PCIe payload ceiling per direction in GB/s (1e9 bytes/s): GT/s x lanes x 128/130 / 8 for gen 3 and later (128b/130b line code).
// 256-byte packet and flow-control overhead is not in this number; it is a ceiling, not a prediction.
inline double pcie_ceiling_gbps(double gts, int lanes) {
#ifdef R11_BREAK_CEILING
    return gts * lanes / 8.0;  // deliberate break: forgets the 128b/130b line code
#else
    return gts * lanes * (128.0 / 130.0) / 8.0;
#endif
}

inline double rate_gbps(double bytes, double seconds) { return seconds > 0 ? bytes / seconds / 1e9 : 1e30; }

// A rate above the ceiling means the probe did not move the bytes (or the timer is wrong): the cell is rejected.
inline bool above_ceiling(double rate_gbps_, double ceiling_gbps) { return rate_gbps_ > ceiling_gbps; }

// The card-to-card control: a copy inside card memory counts each byte read and written, and must land near the measured
// 633 GB/s of memory-system.md. The window is 0.70 to 1.05 of that number (fixed in expect-r11.md before any run).
inline bool d2d_in_band(double rw_gbps) { return rw_gbps >= 0.70 * 633.0 && rw_gbps <= 1.05 * 633.0; }

// splitmix64: the fill pattern and the mixing of the hash.
R11_HD inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Exact 64-bit hash of a buffer of 8-byte words: the sum over words of mix64(word + index * odd constant). A sum is order
// independent, so the card can add partial sums in any order and still match. One flipped bit anywhere changes it.
R11_HD inline uint64_t word_term(uint64_t w, uint64_t idx) {
#ifdef R11_BREAK_HASH
    w &= ~0xffull;  // deliberate break: ignores the lowest byte of every word
#endif
    return mix64(w + idx * 0xd6e8feb86659fd93ull);
}
inline uint64_t hash_words(const uint64_t * w, size_t n) {
    uint64_t h = 0;
    for (size_t i = 0; i < n; ++i) h += word_term(w[i], i);
    return h;
}
inline void fill_pattern(uint64_t * w, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; ++i) w[i] = mix64(i * 0x2545f4914f6cdd1dull + seed);
}

inline double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
inline double percentile_of(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t i = (size_t) (p * (v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

// The two deliberate breaks. `control_passed` is what the control reported on the broken input: it must be FALSE.
inline bool break_ok(const char * name, bool control_passed) {
    std::printf("BREAK %s: control reported %s -> %s\n", name, control_passed ? "PASS" : "FAIL",
                control_passed ? "PROBE INVALID (the control cannot see this fault)" : "ok (FAIL as required)");
    return !control_passed;
}

}  // namespace r11
