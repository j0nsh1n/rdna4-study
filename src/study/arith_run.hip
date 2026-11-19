// Lane 5: generic runner for the arithmetic experiments (scripts/arith_explore.py drives it).
//
// Usage: arith_run IN OUT
// IN (little-endian uint32): form, modifier, steps, use_mode, mode, n_elem, colshift, then per element:
//   steps x (16 A slot bits, 16 B slot bits), then the C bits.
// Element i runs in wave i / 16 as A row j = i % 16 and B column (j + colshift) % 16; its result is D at that
// (row, column). OUT: n_elem result words, then the MODE value read back in wave 0 (0xFFFFFFFF if not written).
#include "arith_gpu.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: arith_run IN OUT\n");
        return 2;
    }
    require_arch();
    FILE * f = std::fopen(argv[1], "rb");
    if (!f) {
        std::perror(argv[1]);
        return 2;
    }
    uint32_t h[7];
    if (std::fread(h, 4, 7, f) != 7) {
        std::fprintf(stderr, "short header\n");
        return 2;
    }
    const int form = (int) h[0], mod = (int) h[1], steps = (int) h[2], n = (int) h[5], shift = (int) h[6];
    std::vector<uint32_t> el((size_t) n * (steps * 32 + 1));
    if (std::fread(el.data(), 4, el.size(), f) != el.size()) {
        std::fprintf(stderr, "short element data\n");
        return 2;
    }
    std::fclose(f);
    arith::Batch bt;
    bt.mod = mod;
    bt.init(form, (n + 15) / 16, steps);
    const size_t per = (size_t) steps * 32 + 1;
    for (int i = 0; i < n; ++i) {
        const int w = i / 16, r = i % 16, col = (r + shift) % 16;
        const uint32_t * e = &el[i * per];
        for (int s = 0; s < steps; ++s) {
            for (int k = 0; k < 16; ++k) {
                bt.set_a(w, s, r, k, e[s * 32 + k]);
                bt.set_b(w, s, col, k, e[s * 32 + 16 + k]);
            }
        }
        bt.set_c(w, r, col, e[steps * 32]);
    }
    arith::run(bt, h[3] != 0, h[4]);
    std::vector<uint32_t> out(n + 1);
    for (int i = 0; i < n; ++i) {
        const int r = i % 16;
        out[i] = bt.get_d(i / 16, r, (r + shift) % 16);
    }
    out[n] = bt.mode_seen.empty() ? 0xFFFFFFFFu : bt.mode_seen[0];
    FILE * o = std::fopen(argv[2], "wb");
    if (!o || std::fwrite(out.data(), 4, out.size(), o) != out.size()) {
        std::perror(argv[2]);
        return 2;
    }
    std::fclose(o);
    return 0;
}
