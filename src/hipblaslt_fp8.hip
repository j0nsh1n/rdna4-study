// hipBLASLt FP8 GEMM (E4M3 x E4M3 -> F32, F32 compute) against a CPU reference, and its speed next to hipBLAS F16
// GEMM (F16 x F16 -> F32) on the same shape. gfx1201 only: hipBLASLt 7.1.1 ships FP8 kernels for gfx1201 but none
// for gfx1100.
// Layout follows the usual FP8 GEMM convention ("TN"): column-major, A transposed (stored K x M), B not (K x N),
// D is M x N. No scale pointers (scales are 1).
// Usage: hipblaslt_fp8 [M N K reps]. Defaults 4096 4096 512 50.
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <random>

#include "e4m3_ref.h"
#include "harness.h"

#define CHECK_BLAS(x)                                                                         \
    do {                                                                                      \
        hipblasStatus_t s_ = (x);                                                             \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                                   \
            std::fprintf(stderr, "%s:%d %s: hipBLAS status %d\n", __FILE__, __LINE__, #x, (int) s_); \
            verdict("hipblaslt_fp8", "FAIL", "a hipBLAS or hipBLASLt call failed (see above)");     \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

int main(int argc, char ** argv) {
    require_arch();
    const int M = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int N = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int K = argc > 3 ? std::atoi(argv[3]) : 512;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 50;

    std::mt19937 rng(99);
    std::vector<uint8_t> ha((size_t) K * M), hb((size_t) K * N);
    for (auto * v : {&ha, &hb}) {
        for (auto & x : *v) {
            do {
                x = (uint8_t) rng();
            } while ((x & 0x7F) == 0x7F || (x & 0x78) > 0x50);  // finite, |x| <= 4: sums stay well inside F32
        }
    }
    std::vector<__half> ha16(ha.size()), hb16(hb.size());
    for (size_t i = 0; i < ha.size(); ++i) {
        ha16[i] = __float2half(e4m3_decode(ha[i]));
    }
    for (size_t i = 0; i < hb.size(); ++i) {
        hb16[i] = __float2half(e4m3_decode(hb[i]));
    }

    uint8_t *da, *db;
    __half *da16, *db16;
    float *dd, *dd16;
    void * ws;
    const size_t ws_size = 64u << 20;
    CHECK(hipMalloc(&da, ha.size()));
    CHECK(hipMalloc(&db, hb.size()));
    CHECK(hipMalloc(&da16, ha.size() * 2));
    CHECK(hipMalloc(&db16, hb.size() * 2));
    CHECK(hipMalloc(&dd, (size_t) M * N * 4));
    CHECK(hipMalloc(&dd16, (size_t) M * N * 4));
    CHECK(hipMalloc(&ws, ws_size));
    CHECK(hipMemcpy(da, ha.data(), ha.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, hb.data(), hb.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(da16, ha16.data(), ha.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db16, hb16.data(), hb.size() * 2, hipMemcpyHostToDevice));

    // hipBLASLt FP8
    hipblasLtHandle_t lt;
    CHECK_BLAS(hipblasLtCreate(&lt));
    hipblasLtMatmulDesc_t desc;
    CHECK_BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN));
    hipblasLtMatrixLayout_t la, lb, ld;
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_8F_E4M3, K, M, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_8F_E4M3, K, N, K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, M, N, M));
    hipblasLtMatmulPreference_t pref;
    CHECK_BLAS(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_BLAS(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size,
                                                     sizeof ws_size));
    hipblasLtMatmulHeuristicResult_t heur[8];
    int found = 0;
    CHECK_BLAS(hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 8, heur, &found));
    std::printf("hipBLASLt: %d FP8 algorithms offered for M=%d N=%d K=%d\n", found, M, N, K);
    if (found == 0) {
        verdict("hipblaslt_fp8", "FAIL", "hipBLASLt offered no FP8 algorithm for this shape");
        return 1;
    }
    const float alpha = 1.0f, beta = 0.0f;
    auto lt_run = [&] {
        CHECK_BLAS(hipblasLtMatmul(lt, desc, &alpha, da, la, db, lb, &beta, dd, ld, dd, ld, &heur[0].algo, ws, ws_size, 0));
    };

    // hipBLAS F16 on the same values
    hipblasHandle_t hb_h;
    CHECK_BLAS(hipblasCreate(&hb_h));
    auto f16_run = [&] {
        CHECK_BLAS(hipblasGemmEx(hb_h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &alpha, da16, HIP_R_16F, K, db16, HIP_R_16F,
                                 K, &beta, dd16, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    };

    lt_run();
    f16_run();
    CHECK(hipDeviceSynchronize());
    std::vector<float> got((size_t) M * N), got16((size_t) M * N);
    CHECK(hipMemcpy(got.data(), dd, got.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(got16.data(), dd16, got16.size() * 4, hipMemcpyDeviceToHost));

    // CPU reference on a sample of outputs: every 7th column, all rows (float64 sums of exact products)
    std::vector<float> av(ha.size()), bv(hb.size());
    for (size_t i = 0; i < ha.size(); ++i) {
        av[i] = e4m3_decode(ha[i]);
    }
    for (size_t i = 0; i < hb.size(); ++i) {
        bv[i] = e4m3_decode(hb[i]);
    }
    double max_rel = 0, max_rel16 = 0;
    long checked = 0, bit_equal = 0;
    for (int n = 0; n < N; n += 7) {
        for (int m = 0; m < M; ++m) {
            double s = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                const double p = (double) av[(size_t) m * K + k] * bv[(size_t) n * K + k];
                s += p;
                mag += std::fabs(p);
            }
            const float g = got[(size_t) n * M + m], g16 = got16[(size_t) n * M + m];
            max_rel = std::max(max_rel, std::fabs(g - s) / (mag + 1e-30));
            max_rel16 = std::max(max_rel16, std::fabs(g16 - s) / (mag + 1e-30));
            bit_equal += g == g16;
            ++checked;
        }
    }

    const double flop = 2.0 * M * N * K;
    const Stats t8 = stats_of(time_ms(lt_run, reps));
    const Stats t16 = stats_of(time_ms(f16_run, reps));
    std::printf("RESULT hipblaslt_fp8 median_ms %.4f TFLOPS %.1f\n", t8.median, flop / (t8.median * 1e9));
    std::printf("RESULT hipblas_f16 median_ms %.4f TFLOPS %.1f\n", t16.median, flop / (t16.median * 1e9));
    std::printf("RESULT accuracy outputs_checked %ld max_rel_fp8 %.2e max_rel_f16 %.2e fp8_equals_f16 %.4f\n", checked,
                max_rel, max_rel16, (double) bit_equal / checked);

    // Products are exact in both paths; only F32 accumulation order differs, which stays far below 1e-5 of
    // sum|a*b| for K = 512. A wrong layout or type lands near 1.
    const bool ok = max_rel < 1e-5;
    char text[256];
    std::snprintf(text, sizeof text, "FP8 GEMM %dx%dx%d max_rel %.1e, %.1f TFLOPS against F16 hipBLAS %.1f TFLOPS", M, N,
                  K, max_rel, flop / (t8.median * 1e9), flop / (t16.median * 1e9));
    verdict("hipblaslt_fp8", ok ? "PASS" : "FAIL", text);
    return ok ? 0 : 1;
}
