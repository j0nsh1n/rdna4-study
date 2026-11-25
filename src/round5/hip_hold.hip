// Round 5 item G: a HIP process that opens the card, creates a compute queue (one tiny kernel), then sits idle for <seconds>.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
__global__ void k(int *p) { p[threadIdx.x] = threadIdx.x; }
int main(int argc, char **argv) {
    double sec = argc > 1 ? atof(argv[1]) : 10;
    int *d; hipMalloc(&d, 256);
    k<<<1, 32>>>(d); hipDeviceSynchronize();
    printf("hip_hold: queue created, idling %.1f s\n", sec); fflush(stdout);
    usleep((useconds_t)(sec * 1e6));
    hipFree(d);
    return 0;
}
