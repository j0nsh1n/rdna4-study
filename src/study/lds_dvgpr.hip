// Lane 3, item 7: can an ordinary HIP kernel on gfx1201 use dynamic VGPRs (S_ALLOC_VGPR, STATUS.DYN_VGPR_EN)?
//
// Usage: lds_dvgpr probe [--break]
//   probe   one wave in an ordinary kernel (no dynamic-VGPR attribute: the compiler has none for HIP, see the log).
//           It reads STATUS (hwreg WAVE_STATUS: DYN_VGPR_EN is bit 30) and STATE_PRIV (SCC is bit 0), then runs
//           s_alloc_vgpr with 64, 256 and 512 VGPRs, setting SCC=1 just before and saving SCC (s_cselect_b32 right
//           after) and STATUS each time.
//           The guide (A 3.3.3) says allocation-changing instructions are ignored when the wave is not in dynamic
//           mode, and that a failed grow sets SCC=0.
//   --break the check expects DYN_VGPR_EN = 1 (or 0 when the bit reads 1), so VERDICT dvgpr must FAIL.
// VERDICT dvgpr is INFO unless something contradicts the guide: here, a wave that reports DYN_VGPR_EN = 1 without
// having been launched in the mode, or an s_alloc_vgpr that changes STATUS.
#include "mem_common.h"

// SCC must be read right after the instruction: the asm is one statement, so nothing is scheduled between.
__device__ __forceinline__ void alloc_probe(uint32_t n, uint32_t & scc, uint32_t & status, uint32_t & priv) {
    uint32_t s, st, pv;
    asm volatile(
        "s_cmp_eq_u32 %3, %3\n\t"  // SCC = 1 first, so an ignored instruction (SCC kept) differs from a failed one (SCC = 0)
        "s_alloc_vgpr %3\n\t"
        "s_cselect_b32 %0, 1, 0\n\t"
        "s_getreg_b32 %1, hwreg(HW_REG_WAVE_STATUS)\n\t"
        "s_getreg_b32 %2, hwreg(HW_REG_WAVE_STATE_PRIV)"
        : "=s"(s), "=s"(st), "=s"(pv)
        : "s"(n)
        : "scc", "memory");
    scc = s;
    status = st;
    priv = pv;
}

// out: [0] status before, [1] state_priv before, then for each request {scc, status, state_priv}, [last] marker
__global__ __launch_bounds__(32) void k_probe(uint32_t n0, uint32_t n1, uint32_t n2, uint32_t * out) {
    uint32_t st, pv;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_STATUS)" : "=s"(st));
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_WAVE_STATE_PRIV)" : "=s"(pv));
    uint32_t r[9];
    alloc_probe(n0, r[0], r[1], r[2]);
    alloc_probe(n1, r[3], r[4], r[5]);
    alloc_probe(n2, r[6], r[7], r[8]);
    // ordinary VGPR work after the requests: if a request had really shrunk the allocation this would fault
    const uint32_t v = threadIdx.x * 3u + 7u;
    if (threadIdx.x == 0) {
        out[0] = st;
        out[1] = pv;
        for (int i = 0; i < 9; ++i) {
            out[2 + i] = r[i];
        }
        out[11] = 0xC0FFEE00u;
    }
    out[16 + threadIdx.x] = v;
}

int main(int argc, char ** argv) {
    const hipDeviceProp_t prop = require_arch();
    if (argc < 2 || std::strcmp(argv[1], "probe") != 0) {
        std::fprintf(stderr, "usage: lds_dvgpr probe [--break]\n");
        return 2;
    }
    const bool brk = mem::has_flag(argc, argv, "--break");
    std::printf("device %s (%s), mode probe%s\n", prop.name, prop.gcnArchName, brk ? " --break" : "");
    std::fflush(stdout);
    uint32_t * d;
    CHECK(hipMalloc(&d, 64 * 4));
    CHECK(hipMemset(d, 0, 64 * 4));
    const uint32_t req[3] = {64, 256, 512};
    hipLaunchKernelGGL(k_probe, dim3(1), dim3(32), 0, 0, req[0], req[1], req[2], d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    uint32_t h[48];
    CHECK(hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost));
    std::printf("RESULT lds_dvgpr before status=0x%08X dyn_vgpr_en=%u state_priv=0x%08X wave64=%u\n", h[0], (h[0] >> 30) & 1,
                h[1], (h[0] >> 29) & 1);
    bool changed = false;
    for (int i = 0; i < 3; ++i) {
        const uint32_t scc = h[2 + 3 * i], st = h[3 + 3 * i], pv = h[4 + 3 * i];
        std::printf("RESULT lds_dvgpr alloc vgprs=%u scc=%u status=0x%08X dyn_vgpr_en=%u state_priv=0x%08X\n", req[i], scc, st,
                    (st >> 30) & 1, pv);
        changed |= st != h[0];
    }
    const bool done = h[11] == 0xC0FFEE00u && h[16 + 31] == 31u * 3 + 7;
    std::printf("RESULT lds_dvgpr kernel_completed=%d\n", done ? 1 : 0);
    const uint32_t bit = (h[0] >> 30) & 1;
    if (brk) {
        const uint32_t want = bit ? 0u : 1u;  // the wrong value on purpose
        if (bit != want) {
            verdict("dvgpr", "FAIL", bit ? "break: expected DYN_VGPR_EN=0 but it reads 1" : "break: expected DYN_VGPR_EN=1 but it reads 0");
            return 1;
        }
    }
    if (!done) {
        verdict("dvgpr", "INFO", "kernel did not complete normally; see the exit code");
        return 0;
    }
    if (bit == 0 && !changed) {
        verdict("dvgpr", "INFO", "ordinary kernel: DYN_VGPR_EN=0, s_alloc_vgpr left STATUS unchanged and the kernel completed; "
                                 "consistent with the guide (allocation instructions ignored when not in dynamic mode)");
    } else {
        verdict("dvgpr", "INFO", "CONTRADICTS the guide: the wave reports dynamic mode or s_alloc_vgpr changed STATUS");
    }
    return 0;
}
