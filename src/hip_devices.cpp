// Lists the GPUs the HIP runtime sees, one line each: "index <i> pci <domain:bus:dev.fn> arch <gfx...> name <...>".
// Run without HIP_VISIBLE_DEVICES; the script matches the PCI address of the card record against it to confirm the
// device index. Host code only; it creates no queue and loads no kernel.
#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount failed\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p;
        char pci[32] = "?";
        if (hipGetDeviceProperties(&p, i) != hipSuccess) {
            return 1;
        }
        if (hipDeviceGetPCIBusId(pci, sizeof pci, i) != hipSuccess) {
            pci[0] = '?';
            pci[1] = 0;
        }
        std::printf("index %d pci %s arch %s name %s\n", i, pci, p.gcnArchName, p.name);
    }
    return 0;
}
