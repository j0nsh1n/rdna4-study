// Cross-check of two lanes: the result bits a Vulkan cooperative-matrix multiply-add produced through RADV
// (lane 9, file coopmat-f16-check-*.txt in a study-vulkan run folder) against the CPU model of lane 5, with the
// matrix's K index mapped to operand slots in both published 16-bit orders. CPU only.
// Build: g++ -O2 -std=c++17 src/study/arith_vkcheck.cpp -o build/arith/bin/arith_vkcheck. Usage: arith_vkcheck FILE.
#include "arith_model.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    FILE * f = argc > 1 ? std::fopen(argv[1], "r") : nullptr;
    if (!f) {
        std::fprintf(stderr, "usage: arith_vkcheck FILE\n");
        return 2;
    }
    char line[4096];
    std::vector<uint32_t> A, B, C, D;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] != 'A' && line[0] != 'B' && line[0] != 'C' && line[0] != 'D') continue;
        std::vector<uint32_t> & v = line[0] == 'A' ? A : line[0] == 'B' ? B : line[0] == 'C' ? C : D;
        char * p = line + 3;
        for (int i = 0; i < 16; ++i) v.push_back((uint32_t) std::strtoul(p, &p, 16));
    }
    const int sets = (int) D.size() / 256;
    std::printf("sets %d (A %zu B %zu C %zu D %zu words)\n", sets, A.size(), B.size(), C.size(), D.size());
    const am::Params P = am::measured_params(0);
    const char * names[2] = {"slot = K (sample-code order)", "guide order (slots 4-7 hold K 8-11)"};
    long best = 0, total = 0;
    for (int order = 0; order < 2; ++order) {
        long match = 0, exact_match = 0, n = 0;
        for (int s = 0; s < sets; ++s)
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) {
                    uint32_t a[16], b[16];
                    for (int slot = 0; slot < 16; ++slot) {
                        int k = slot;
                        if (order == 1) k = slot < 4 ? slot : slot < 8 ? slot + 4 : slot < 12 ? slot - 4 : slot;
                        a[slot] = A[s * 256 + r * 16 + k];
                        b[slot] = B[s * 256 + k * 16 + c];
                    }
                    const uint32_t acc = C[s * 256 + r * 16 + c], d = D[s * 256 + r * 16 + c];
                    match += am::model_f32(0, a, b, acc, P) == d;
                    exact_match += am::exact_round(0, a, b, acc, am::F32) == d;
                    ++n;
                }
        std::printf("RESULT vkcheck %s: model matches %ld of %ld; exact sum rounded once matches %ld\n", names[order],
                    match, n, exact_match);
        best = std::max(best, match);
        total = n;
    }
    const bool pass = total > 0 && best == total;
    std::printf("VERDICT arith_vkcheck: %s the model reproduces %ld of %ld Vulkan result words under one K order\n",
                pass ? "PASS" : "FAIL", best, total);
    return pass ? 0 : 1;
}
