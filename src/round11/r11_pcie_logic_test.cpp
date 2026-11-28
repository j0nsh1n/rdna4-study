// CPU-only test of r11_pcie_logic.h. Exit 0 if every check passes. Build with -DR11_BREAK_CEILING or -DR11_BREAK_HASH to
// confirm the matching check goes red (the scripts do this and require a non-zero exit).
#include <cmath>
#include <cstdio>

#include "r11_pcie_logic.h"

static int bad = 0;
#define CHECK_T(c, msg)                                          \
    do {                                                         \
        if (!(c)) {                                              \
            std::printf("test FAIL: %s (%s)\n", msg, #c);        \
            ++bad;                                               \
        }                                                        \
    } while (0)

int main() {
    // Ceilings worked out by hand: 32 GT/s x 16 x 128/130 / 8 = 63.0154; 16 GT/s: 31.5077; 8 GT/s x 4: 3.9385.
    CHECK_T(std::fabs(r11::pcie_ceiling_gbps(32, 16) - 63.0154) < 0.001, "gen5 x16 ceiling is 63.015 GB/s");
    CHECK_T(std::fabs(r11::pcie_ceiling_gbps(16, 16) - 31.5077) < 0.001, "gen4 x16 ceiling is 31.508 GB/s");
    CHECK_T(std::fabs(r11::pcie_ceiling_gbps(8, 4) - 3.9385) < 0.001, "gen3 x4 ceiling is 3.939 GB/s");

    const double c5 = r11::pcie_ceiling_gbps(32, 16);
    CHECK_T(!r11::above_ceiling(55.0, c5), "55 GB/s on gen5 x16 is accepted");
    CHECK_T(r11::above_ceiling(63.5, c5), "63.5 GB/s on gen5 x16 is rejected");
    // Deliberate break 2 of the plan: a skipped 256 MiB copy that takes 20 us reads as 13400 GB/s and must be rejected.
    CHECK_T(r11::above_ceiling(r11::rate_gbps(256.0 * 1048576, 20e-6), c5), "skipped copy is flagged");
    CHECK_T(r11::rate_gbps(1e9, 0.0) > 1e20, "zero time reads as impossible, not as zero");
    CHECK_T(std::fabs(r11::rate_gbps(1e9, 0.02) - 50.0) < 1e-9, "1 GB in 20 ms is 50 GB/s");

    CHECK_T(r11::d2d_in_band(633.0) && r11::d2d_in_band(500.0), "card-to-card 633 and 500 GB/s are in the window");
    CHECK_T(!r11::d2d_in_band(60.0) && !r11::d2d_in_band(900.0), "card-to-card 60 and 900 GB/s are outside it");

    // Hash: stable, order independent, and sensitive to a flip of any single bit in any byte of a word.
    std::vector<uint64_t> w(4096);
    r11::fill_pattern(w.data(), w.size(), 7);
    const uint64_t h0 = r11::hash_words(w.data(), w.size());
    CHECK_T(h0 == r11::hash_words(w.data(), w.size()), "hash is repeatable");
    CHECK_T(h0 != 0, "hash of the pattern is not zero");
    for (int byte = 0; byte < 8; ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            auto v = w;
            unsigned char * p = (unsigned char *) &v[1234];
            p[byte] ^= (unsigned char) (1u << bit);
            if (r11::hash_words(v.data(), v.size()) == h0) {
                std::printf("test FAIL: flip of byte %d bit %d not seen by the hash\n", byte, bit);
                ++bad;
            }
        }
    }
    {  // deliberate break 1 of the plan, as the probe runs it: one byte flipped after the copy
        auto v = w;
        ((unsigned char *) v.data())[8 * 17 + 3] ^= 0x40;
        CHECK_T(r11::break_ok("flip", r11::hash_words(v.data(), v.size()) == h0), "flip break reported FAIL");
    }
    {  // two words swapped: a plain sum would miss it; the index in the term must not
        auto v = w;
        std::swap(v[10], v[11]);
        CHECK_T(r11::hash_words(v.data(), v.size()) != h0, "swapped words change the hash");
    }
    CHECK_T(std::fabs(r11::median_of({3, 1, 2}) - 2) < 1e-12 && std::fabs(r11::median_of({4, 1, 2, 3}) - 2.5) < 1e-12, "median");
    CHECK_T(std::fabs(r11::percentile_of({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.95) - 10) < 1e-12, "p95 of 1..10 is 10");
    std::printf("r11_pcie_logic_test: %d check(s) failed\n", bad);
    return bad ? 1 : 0;
}
