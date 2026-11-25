// Round 5 item A, state P: a bit-check that runs on the GPU for the whole time a heavy burst runs beside it.
// Every wave executes chunks of 256 WMMA (f16 or fp8, K = 16) with all-A elements x and all-B elements y (small integers, exact in the
// format), accumulating in FP32 from zero: every output element must equal 16 * sum(x*y), an integer below 2^24, so the expected
// value is exact. The sequence of (x, y) depends on the chunk number mod 64; the host computes the 64 expected values with integer
// arithmetic. A wave compares all 8 accumulator values after every chunk against the table and counts mismatches.
// Usage: bitloop <f16|fp8> <seconds> <waves> [--break]   (--break: the host table is wrong by one in entry 5: must FAIL)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
__host__ __device__ inline int xv(int c, int n) { return ((c * 5 + n * 3) % 7) + 1; }
__host__ __device__ inline int yv(int c, int n) { return ((c * 11 + n) % 5) + 1; }
__device__ inline uint32_t fp8byte(int v) { const uint32_t t[8] = {0, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E}; return t[v]; }
template <int KIND>
__global__ __launch_bounds__(32) void loop(unsigned long long ticks, const float *tab, unsigned long long *bad, unsigned long long *chunks, unsigned long long *first) {
    const unsigned long long w0 = wall_clock64();
    unsigned long long nbad = 0, nch = 0;
    for (int c = 0;; ++c) {
        v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int n = 0; n < 256; ++n) {
            const int x = xv(c & 63, n), y = yv(c & 63, n);
            if constexpr (KIND == 0) {
                v8h a, b;
                for (int i = 0; i < 8; ++i) { a[i] = (_Float16)x; b[i] = (_Float16)y; }
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
            } else {
                const uint32_t xa = fp8byte(x) * 0x01010101u, yb = fp8byte(y) * 0x01010101u;
                const v2i a = {(int)xa, (int)xa}, b = {(int)yb, (int)yb};
                acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
            }
        }
        const float e = tab[c & 63];
        for (int i = 0; i < 8; ++i) if (acc[i] != e) { if (!nbad && !threadIdx.x) { first[0] = c; first[1] = __float_as_uint(acc[i]); first[2] = __float_as_uint(e); } ++nbad; }
        ++nch;
        if (wall_clock64() - w0 >= ticks) break;
    }
    atomicAdd(bad, nbad); if (!threadIdx.x) atomicAdd(chunks, nch);
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: bitloop <f16|fp8> <seconds> <waves> [--break]\n"); return 2; }
    const int kind = !strcmp(argv[1], "fp8"); const double sec = atof(argv[2]); const int waves = atoi(argv[3]);
    const bool brk = argc > 4 && !strcmp(argv[4], "--break");
    float tab[64];
    for (int c = 0; c < 64; ++c) { long s = 0; for (int n = 0; n < 256; ++n) s += 16L * xv(c, n) * yv(c, n); tab[c] = (float)s; }
    if (brk) tab[5] += 1.0f;
    float *dt; unsigned long long *dm; hipMalloc(&dt, sizeof tab); hipMalloc(&dm, 8 * 8);
    hipMemcpy(dt, tab, sizeof tab, hipMemcpyHostToDevice); hipMemset(dm, 0, 64);
    const unsigned long long ticks = (unsigned long long)(sec * 1e8);
    if (kind) loop<1><<<waves, 32>>>(ticks, dt, dm, dm + 1, dm + 2); else loop<0><<<waves, 32>>>(ticks, dt, dm, dm + 1, dm + 2);
    unsigned long long h[8]; hipMemcpy(h, dm, 64, hipMemcpyDeviceToHost);
    const double wmma = (double)h[1] * 256;
    printf("RESULT bitloop %s waves %d: chunks %llu (%.3g WMMA), mismatching elements %llu, first bad chunk %llu got %g expected %g\n", argv[1], waves, h[1], wmma, h[0], h[2], *(float *)&h[3], *(float *)&h[4]);
    const bool ok = h[0] == 0 && h[1] > 0;
    printf("VERDICT bitloop_%s: %s %llu mismatches over %llu chunks%s\n", argv[1], ok ? "PASS" : "FAIL", h[0], h[1], brk ? " (broken on purpose)" : "");
    return ok ? 0 : 1;
}
