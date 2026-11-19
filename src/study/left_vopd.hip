// Leftovers C, lane A, EC-3: what a dual-issue (VOPD) pair does when it breaks the guide's independence rule
// ("OPX must not overwrite sources of OPY; OPY may overwrite sources of OPX", AMD's RDNA 4 instruction set guide
// 7.8; the RDNA 3 guide added that both halves execute at the same time and reads see the old value, the RDNA 4 guide
// dropped that sentence). The assembler accepts the forbidden pairs (it checks register parity and banks only).
//
// Usage: left_vopd [--break]
//   One wave, every lane its own data, REPS repetitions of each variant; per lane and repetition the result of the
//   reading half is classified as the value from before the pair ("old"), the value the other half writes ("new"),
//   or neither ("other"). Registers are fixed in the text: v21 = lane + 1, v22 = 2, v23 = 100 + lane / 2, v10 and v11
//   start at 1000 + lane and 3000 + lane (all exact in FP32).
//   Variants (X :: Y):
//     fwd_src0   v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23   (Y reads X's destination as src0: forbidden)
//     fwd_src1   v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v23, v10   (as src1: forbidden)
//     fwd_add    v_dual_add_f32 v10, v21, v22 :: v_dual_mul_f32 v11, v10, v23 (forbidden)
//     fwd_fmac   v_dual_fmac_f32 v10, v21, v22 :: v_dual_fmac_f32 v11, v10, v23 (forbidden)
//     fwd_mov    v_dual_mov_b32 v10, v21 :: v_dual_mov_b32 v11, v10 (forbidden; Y's mov uses the SRC2 port)
//     fwd_src0_fresh  fwd_src0 with v10 written by the instruction just before the pair (no s_nop between)
//     rev_allowed v_dual_add_f32 v10, v21, v22 :: v_dual_mov_b32 v21, v23 (Y overwrites X's source: allowed; the
//                guide's rule says X reads the old v21, so X's result must be old v21 + 2)
//   Control: the forbidden fwd_src0 as two ordinary instructions (v_mov_b32 then v_add_f32) must give the new value.
//   Verdicts: vopd_control (control new in every lane and repetition; the allowed reverse case old everywhere),
//   vopd_deterministic (every variant one answer in every lane and repetition); which answer is INFO.
//   --break expects the old value from the sequential control, so vopd_control must FAIL.
#include "../harness.h"

#include <cmath>
#include <string>

namespace {

constexpr int REPS = 1000;
constexpr int NV = 8;

#define INIT                                                                                                         \
    "v_mov_b32 v21, %[a]\n v_mov_b32 v22, %[b]\n v_mov_b32 v23, %[c]\n v_mov_b32 v10, %[o]\n v_mov_b32 v11, %[p]\n"
#define SETTLE "s_nop 7\n s_nop 7\n"
#define OUTS "v_mov_b32 %[r0], v10\n v_mov_b32 %[r1], v11\n v_mov_b32 %[r2], v21\n"

#define VARIANT(BODY)                                                                                             \
    asm volatile(BODY : [r0] "=&v"(r0), [r1] "=&v"(r1), [r2] "=&v"(r2) : [a] "v"(a), [b] "v"(b), [c] "v"(c),       \
                 [o] "v"(o), [p] "v"(pp) : "v10", "v11", "v21", "v22", "v23")

__global__ __launch_bounds__(32) void k_vopd(float * out) {
    const int lane = threadIdx.x;
    const float a = lane + 1.0f, b = 2.0f, c = 100.0f + 0.5f * lane, o = 1000.0f + lane, pp = 3000.0f + lane;
    for (int rep = 0; rep < REPS; ++rep) {
        float r0, r1, r2;
        float * q = out + ((size_t) rep * NV) * 32 * 3 + lane * 3;
#define STORE(v)                    \
    q[(v) * 32 * 3 + 0] = r0;       \
    q[(v) * 32 * 3 + 1] = r1;       \
    q[(v) * 32 * 3 + 2] = r2;
        VARIANT(INIT SETTLE "v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23\n" OUTS);
        STORE(0)
        VARIANT(INIT SETTLE "v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v23, v10\n" OUTS);
        STORE(1)
        VARIANT(INIT SETTLE "v_dual_add_f32 v10, v21, v22 :: v_dual_mul_f32 v11, v10, v23\n" OUTS);
        STORE(2)
        VARIANT(INIT SETTLE "v_dual_fmac_f32 v10, v21, v22 :: v_dual_fmac_f32 v11, v10, v23\n" OUTS);
        STORE(3)
        VARIANT(INIT SETTLE "v_dual_mov_b32 v10, v21 :: v_dual_mov_b32 v11, v10\n" OUTS);
        STORE(4)
        VARIANT(INIT "v_dual_mov_b32 v10, v21 :: v_dual_add_f32 v11, v10, v23\n" OUTS);
        STORE(5)
        VARIANT(INIT SETTLE "v_dual_add_f32 v10, v21, v22 :: v_dual_mov_b32 v21, v23\n" OUTS);
        STORE(6)
        VARIANT(INIT SETTLE "v_mov_b32 v10, v21\n v_add_f32 v11, v10, v23\n" OUTS);
        STORE(7)
#undef STORE
    }
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    bool brk = false;
    for (int i = 1; i < argc; ++i) brk |= std::strcmp(argv[i], "--break") == 0;
    float * d;
    const size_t n = (size_t) REPS * NV * 32 * 3;
    CHECK(hipMalloc(&d, n * 4));
    CHECK(hipMemset(d, 0xff, n * 4));
    hipLaunchKernelGGL(k_vopd, dim3(1), dim3(32), 0, 0, d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> h(n);
    CHECK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
    const char * names[NV] = {"fwd_src0", "fwd_src1", "fwd_add", "fwd_fmac", "fwd_mov", "fwd_src0_fresh", "rev_allowed",
                              "control_seq"};
    int cnt[NV][3] = {};
    bool single[NV];
    for (int v = 0; v < NV; ++v) {
        for (int rep = 0; rep < REPS; ++rep) {
            for (int lane = 0; lane < 32; ++lane) {
                const float a = lane + 1.0f, b = 2.0f, c = 100.0f + 0.5f * lane, o = 1000.0f + lane,
                            p = 3000.0f + lane;
                const float * q = &h[(((size_t) rep * NV + v) * 32 + lane) * 3];
                // value the reading half saw, recovered from its result, against old and new
                float got, oldv, newv;
                switch (v) {
                    case 0: case 5: case 7: got = q[1] - c; oldv = o; newv = a; break;
                    case 1: got = q[1] - c; oldv = o; newv = a; break;
                    case 2: got = q[1] / c; oldv = o; newv = a + b; break;
                    case 3: got = (q[1] - p) / c; oldv = o; newv = o + a * b; break;
                    case 4: got = q[1]; oldv = o; newv = a; break;
                    default: got = q[0] - b; oldv = a; newv = c; break;  // X's view of v21 (X: v10 = v21 + 2)
                }
                cnt[v][got == oldv ? 0 : got == newv ? 1 : 2]++;
            }
        }
        int kinds = 0;
        for (int k = 0; k < 3; ++k) kinds += cnt[v][k] > 0;
        single[v] = kinds == 1 && cnt[v][2] == 0;
        std::printf("RESULT vopd %-15s old %6d new %6d other %6d (of %d lane results)\n", names[v], cnt[v][0], cnt[v][1],
                    cnt[v][2], REPS * 32);
    }
    const int all = REPS * 32;
    const bool ctl = (brk ? cnt[7][0] == all : cnt[7][1] == all) && cnt[6][0] == all;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "sequential control: new in %d of %d lane results (expect %s); allowed reverse pair (OPY overwrites "
                  "OPX's source): X saw the old value in %d of %d (guide: old)",
                  brk ? cnt[7][0] : cnt[7][1], all, brk ? "old: deliberate break" : "new", cnt[6][0], all);
    verdict("vopd_control", ctl ? "PASS" : "FAIL", buf);
    bool det = true;
    std::string ans;
    for (int v = 0; v < 6; ++v) {
        det &= single[v];
        ans += std::string(" ") + names[v] + "=" + (cnt[v][0] == all ? "old" : cnt[v][1] == all ? "new" : "mixed");
    }
    std::snprintf(buf, sizeof buf, "every forbidden variant one answer in all lanes and repetitions: %s;%s",
                  det ? "yes" : "NO", ans.c_str());
    verdict("vopd_deterministic", det ? "PASS" : "FAIL", buf);
    return ctl && det ? 0 : 1;
}
