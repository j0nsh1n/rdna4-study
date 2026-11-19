// Lane 5: compares the CPU model (arith_model.h) with hardware results written by arith_dump (CPU only).
// Usage: arith_fit FORM FILE... [--show N] [--param name=value ...]
//   params: order=guide|contig, group=G, prod_off=P, width=W, acc_floor_small=0|1, zero_anchor=0|1,
//   prod_trunc=0|1, acc_off=A. Prints agreement per file, against the exact sum too, and the first N mismatches.
#include "arith_model.h"

#include <cstdio>
#include <string>
#include <vector>

static const char * forms[] = {"f16", "bf16", "fp8_fp8", "fp8_bf8", "bf8_fp8", "bf8_bf8"};

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: arith_fit FORM FILE... [--show N] [--param k=v]\n");
        return 2;
    }
    int form = -1;
    for (int f = 0; f < 6; ++f) {
        form = std::string(argv[1]) == forms[f] ? f : form;
    }
    am::Params P = am::measured_params(form);
    int show = 5;
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--show") {
            show = std::atoi(argv[++i]);
        } else if (a == "--param") {
            const std::string kv = argv[++i];
            const std::string k = kv.substr(0, kv.find('=')), v = kv.substr(kv.find('=') + 1);
            if (k == "order") {
                const int contig[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
                if (v == "contig") {
                    std::copy(contig, contig + 16, P.order);
                }
            } else if (k == "group") {
                P.group = std::stoi(v);
            } else if (k == "prod_off") {
                P.prod_off = std::stoi(v);
            } else if (k == "width") {
                P.width = std::stoi(v);
            } else if (k == "acc_floor_small") {
                P.acc_floor_small = std::stoi(v);
            } else if (k == "zero_anchor") {
                P.zero_products_anchor = std::stoi(v);
            } else if (k == "prod_trunc") {
                P.prod_trunc = std::stoi(v);
            } else if (k == "local_floor") {
                P.local_floor = std::stoi(v);
            } else if (k == "sub") {
                P.sub = std::stoi(v);
            } else if (k == "prod_each") {
                P.prod_each = std::stoi(v);
            } else if (k == "acc_off") {
                P.acc_off = std::stoi(v);
            } else {
                std::fprintf(stderr, "unknown param %s\n", k.c_str());
                return 2;
            }
        } else {
            files.push_back(a);
        }
    }
    const am::Fmt fa = am::fmt_a(form), fb = am::fmt_b(form);
    long all = 0, all_bad = 0;
    for (const std::string & fn : files) {
        FILE * f = std::fopen(fn.c_str(), "rb");
        if (!f) {
            std::perror(fn.c_str());
            return 2;
        }
        std::vector<uint32_t> buf(1024);
        long n = 0, bad = 0, exact_same = 0;
        int shown = 0;
        while (std::fread(buf.data(), 4, 1024, f) == 1024) {
            const uint32_t *A = &buf[0], *B = &buf[256], *C = &buf[512], *D = &buf[768];
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    const uint32_t got = D[r * 16 + c];
                    const uint32_t want = am::model_f32(form, A + r * 16, B + c * 16, C[r * 16 + c], P);
                    const uint32_t ex = am::exact_round(form, A + r * 16, B + c * 16, C[r * 16 + c], am::F32);
                    exact_same += got == ex;
                    ++n;
                    if (got != want) {
                        ++bad;
                        if (shown < show) {
                            ++shown;
                            std::printf("mismatch %s r%d c%d: hw %08x model %08x exact %08x  C=%08x (%a)\n", fn.c_str(),
                                        r, c, got, want, ex, C[r * 16 + c],
                                        (double) __builtin_bit_cast(float, C[r * 16 + c]));
                            for (int s = 0; s < 16; ++s) {
                                const double va = am::value(fa, A[r * 16 + s]), vb = am::value(fb, B[c * 16 + s]);
                                if (va * vb != 0 || std::isnan(va * vb)) {
                                    std::printf("   slot %2d a=%a b=%a p=%a\n", s, va, vb, va * vb);
                                }
                            }
                            std::printf("   hw %a model %a exact %a\n", (double) __builtin_bit_cast(float, got),
                                        (double) __builtin_bit_cast(float, want), (double) __builtin_bit_cast(float, ex));
                        }
                    }
                }
            }
        }
        std::fclose(f);
        std::printf("FIT %s %s: %ld elements, model mismatches %ld (%.6f %%), hardware equals exact sum on %.4f %%\n",
                    argv[1], fn.c_str(), n, bad, 100.0 * bad / std::max(1L, n), 100.0 * exact_same / std::max(1L, n));
        all += n;
        all_bad += bad;
    }
    std::printf("FIT total %ld elements, %ld mismatches\n", all, all_bad);
    return all_bad ? 1 : 0;
}
