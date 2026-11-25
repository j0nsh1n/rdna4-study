// Round 5 companion load: keeps `waves` single-wave work-groups busy with FMAs for <seconds> of wall time, so the card
// sits in the light-load high-clock state while another process runs the checks. Prints the in-kernel core clock
// (cycle counter against the 100 MHz REALTIME counter) of wave 0.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__global__ void spin(unsigned long long ticks, unsigned long long *out, float *sink) {
    float a = 1.0f + threadIdx.x, b = 0.999f, c = 0.001f;
    unsigned long long w0 = wall_clock64(), c0 = clock64();
    unsigned long long w1 = w0;
    while (w1 - w0 < ticks) {
        for (int i = 0; i < 4096; i++) a = fmaf(a, b, c);
        w1 = wall_clock64();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = clock64() - c0; out[1] = w1 - w0; }
    if (a == 12345.0f) sink[0] = a;
}
int main(int argc, char **argv) {
    double sec = argc > 1 ? atof(argv[1]) : 5; int blocks = argc > 2 ? atoi(argv[2]) : 1;
    unsigned long long *out; float *sink;
    hipMalloc(&out, 16); hipMalloc(&sink, 4);
    spin<<<blocks, 32>>>((unsigned long long)(sec * 1e8), out, sink);
    unsigned long long h[2] = {0, 0};
    hipMemcpy(h, out, 16, hipMemcpyDeviceToHost);
    printf("spin %d wave(s) %.2f s: in-kernel clock %.0f MHz\n", blocks, h[1] / 1e8, h[0] / (h[1] / 100.0));
    return 0;
}
