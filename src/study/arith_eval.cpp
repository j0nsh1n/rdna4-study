// Lane 5: the CPU model on an arith_run input file (CPU only), so exploration scripts can compare model and
// hardware on the same elements. Usage: arith_eval IN OUT [k=v ...] (parameters as in arith_fit).
// OUT: per element the model's FP32 bits, then the exact sum rounded once to FP32 (2 words per element).
#include "arith_model.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: arith_eval IN OUT [k=v ...]\n");
        return 2;
    }
    FILE * f = std::fopen(argv[1], "rb");
    uint32_t h[7];
    if (!f || std::fread(h, 4, 7, f) != 7) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }
    // runner forms 6 and 7 are FP16 into FP16 and BF16 into BF16: the 16-bit inputs' model form with the measured
    // 16-bit accumulator rounding (every step; BF16 through FP32, then toward zero)
    const int gform = (int) h[0], steps = (int) h[2], n = (int) h[5];
    const bool acc16 = gform == 6 || gform == 7;
    const int form = acc16 ? gform - 6 : gform;
    am::Params P = acc16 ? am::acc16_params(form, 3) : am::measured_params(form);
    if (acc16 && form == am::F_BF16) {
        P.rtz16 = 2;
    }
    for (int i = 3; i < argc; ++i) {
        const std::string kv = argv[i];
        const std::string k = kv.substr(0, kv.find('='));
        const int v = std::stoi(kv.substr(kv.find('=') + 1));
        if (k == "width") P.width = v;
        else if (k == "prod_off") P.prod_off = v;
        else if (k == "acc_off") P.acc_off = v;
        else if (k == "acc_floor_small") P.acc_floor_small = v;
        else if (k == "group") P.group = v;
        else if (k == "prod_each") P.prod_each = v;
        else if (k == "sub") P.sub = v;
        else if (k == "local_floor") P.local_floor = v;
        else if (k == "min_unit") P.min_unit = v;
        else if (k == "contig" && v) for (int j = 0; j < 16; ++j) P.order[j] = j;
    }
    std::vector<uint32_t> el((size_t) n * (steps * 32 + 1));
    if (std::fread(el.data(), 4, el.size(), f) != el.size()) {
        std::fprintf(stderr, "short input\n");
        return 2;
    }
    std::fclose(f);
    std::vector<uint32_t> out;
    for (int i = 0; i < n; ++i) {
        const uint32_t * e = &el[(size_t) i * (steps * 32 + 1)];
        uint32_t acc = e[steps * 32];
        uint32_t ex = 0;
        for (int s = 0; s < steps; ++s) {
            if (acc16) {
                acc = am::model_acc16(form, e + s * 32, e + s * 32 + 16, acc, P);
                continue;
            }
            ex = am::exact_round(form, e + s * 32, e + s * 32 + 16, acc, am::F32);
            acc = am::model_f32(form, e + s * 32, e + s * 32 + 16, acc, P);
        }
        out.push_back(acc);
        out.push_back(ex);
    }
    FILE * o = std::fopen(argv[2], "wb");
    std::fwrite(out.data(), 4, out.size(), o);
    std::fclose(o);
    return 0;
}
