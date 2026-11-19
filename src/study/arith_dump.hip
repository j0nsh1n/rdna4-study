// Lane 5: runs random problems on the card and writes inputs and results for offline model fitting
// (arith_fit.cpp). Usage: arith_dump FORM DIST NPROBLEMS SEED OUT
//   FORM: f16 bf16 fp8_fp8 fp8_bf8 bf8_fp8 bf8_bf8;  DIST: wide narrow cancel bigsmall allcodes
// OUT: per problem A[16][16], B[16][16], C[16][16], D[16][16] as uint32 (A row-major by row then slot, B by
// column then slot, C and D by row then column).
#include "arith_data.h"
#include "arith_gpu.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: arith_dump FORM DIST N SEED OUT\n");
        return 2;
    }
    require_arch();
    int form = -1, dist = -1;
    for (int f = 0; f < 6; ++f) {
        form = std::string(argv[1]) == arith::form_name[f] ? f : form;
    }
    for (int d = 0; d < ad::NDIST; ++d) {
        dist = std::string(argv[2]) == ad::dist_name[d] ? d : dist;
    }
    if (form < 0 || dist < 0) {
        std::fprintf(stderr, "unknown form or distribution\n");
        return 2;
    }
    const int n = std::atoi(argv[3]);
    std::mt19937_64 rng(std::strtoull(argv[4], nullptr, 10));
    std::vector<ad::Problem> ps(n);
    for (auto & p : ps) {
        p = ad::make(form, (ad::Dist) dist, rng);
    }
    arith::Batch bt;
    bt.init(form, n, 1);
    for (int w = 0; w < n; ++w) {
        for (int r = 0; r < 16; ++r) {
            for (int s = 0; s < 16; ++s) {
                bt.set_a(w, 0, r, s, ps[w].a[r][s]);
                bt.set_b(w, 0, r, s, ps[w].b[r][s]);
            }
            for (int c = 0; c < 16; ++c) {
                bt.set_c(w, r, c, ps[w].c[r][c]);
            }
        }
    }
    arith::run(bt);
    FILE * o = std::fopen(argv[5], "wb");
    if (!o) {
        std::perror(argv[5]);
        return 2;
    }
    for (int w = 0; w < n; ++w) {
        uint32_t d[16][16];
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                d[r][c] = bt.get_d(w, r, c);
            }
        }
        std::fwrite(ps[w].a, 4, 256, o);
        std::fwrite(ps[w].b, 4, 256, o);
        std::fwrite(ps[w].c, 4, 256, o);
        std::fwrite(d, 4, 256, o);
    }
    std::fclose(o);
    std::printf("wrote %d problems (%d elements) of %s/%s\n", n, n * 256, argv[1], argv[2]);
    return 0;
}
