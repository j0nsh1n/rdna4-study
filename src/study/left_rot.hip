// Leftovers pass, matrix unit, item 8 (matrix-timing.md item 5d): with a vector instruction between matrix
// instructions, rotating the A or B registers cost one cycle more (19 against 18 cycles per FP16 pair) where it
// cost nothing without the vector instruction. This probe varies the number of A and B register sets, which
// operand rotates, the kind of instruction between (none, v_mov reading a VGPR, v_mov of a constant, v_nop, s_nop,
// v_fma reading three VGPRs) and the pattern of changes, to find what triggers the cycle.
//
// One wave alone on its SIMD. Accumulators rotate over 8 (v[0:63], so no matrix instruction waits for C); A sets at
// v[64:95], B sets at v[96:127] (4 VGPRs each; the FP8 form uses the first 2 of each); the instruction between
// writes v130. One period of 8 blocks (block = optional instruction + matrix instruction) repeated R = 4 and R = 8
// times per loop iteration; cycles per block = (T(8) - T(4)) / (4 x 8 x iterations), so the loop's own cost cancels.
//
// Leftovers C (MT-6), variants 32-36: B shifted by 2 registers against B (partial overlap); the v_mov only before every
// second matrix instruction; the v_mov in VOP3 form; v_nop in VOP3 form (v_nop_e64); B rotating while A and B
// both rotate in step but A also shifted. (Every A and B set holds the same values, so a "copy with the same values" is
// what every rotating variant already uses.) Exploratory: only the known answers are judged.
//
// Usage: left_rot [--break]. Known answers judged (matrix-timing.md item 5d, where A and B rotated together): no
// instruction between, A and B fixed: 16 cycles; v_mov between, A and B fixed: 18; v_mov between, A and B rotating
// over 4: 19. --break expects 18 for the last, so the verdict must FAIL.
#include "../harness.h"

#include <algorithm>
#include <string>
#include <vector>

#define ACC0 "v[0:7]"
#define ACC1 "v[8:15]"
#define ACC2 "v[16:23]"
#define ACC3 "v[24:31]"
#define ACC4 "v[32:39]"
#define ACC5 "v[40:47]"
#define ACC6 "v[48:55]"
#define ACC7 "v[56:63]"
#define A0 "v[64:67]"
#define A1 "v[68:71]"
#define A2 "v[72:75]"
#define A3 "v[76:79]"
#define A4 "v[80:83]"
#define A5 "v[84:87]"
#define A6 "v[88:91]"
#define A7 "v[92:95]"
#define B0 "v[96:99]"
#define B1 "v[100:103]"
#define B2 "v[104:107]"
#define B3 "v[108:111]"
#define B4 "v[112:115]"
#define B5 "v[116:119]"
#define B6 "v[120:123]"
#define B7 "v[124:127]"
// 8-bit operands: the first two VGPRs of each set
#define A0h "v[64:65]"
#define A1h "v[68:69]"
#define A2h "v[72:73]"
#define A3h "v[76:77]"
#define B0h "v[96:97]"
#define B1h "v[100:101]"
#define B2h "v[104:105]"
#define B3h "v[108:109]"

#define F16(ACC, A, B) "v_wmma_f32_16x16x16_f16 " ACC ", " A ", " B ", " ACC "\n"
#define FP8(ACC, A, B) "v_wmma_f32_16x16x16_fp8_fp8 " ACC ", " A ", " B ", " ACC "\n"
#define PER(OP, X, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7)                                    \
    X OP(ACC0, a0, b0) X OP(ACC1, a1, b1) X OP(ACC2, a2, b2) X OP(ACC3, a3, b3) X OP(ACC4, a4, b4) X OP(ACC5, a5, b5) \
        X OP(ACC6, a6, b6) X OP(ACC7, a7, b7)

// the operand lists are macros of 8 names; PERX expands them before PER counts its arguments
#define PERX(...) PER(__VA_ARGS__)
#define XNONE ""
#define XMOV "v_mov_b32 v130, v131\n"
#define XMOVK "v_mov_b32 v130, 0\n"
#define XVNOP "v_nop\n"
#define XSNOP "s_nop 0\n"
#define XFMA "v_fma_f32 v130, v131, v132, v133\n"
#define XMOVA "v_mov_b32 v130, v64\n"
#define XMOV64 "v_mov_b32_e64 v130, v131\n"
#define XVNOP64 "v_nop_e64\n"
#define B0s "v[98:101]"
#define BS2 B0, B0s, B0, B0s, B0, B0s, B0, B0s
// the v_mov before every second block only (B rotating over 4)
#define PER_ALT(OP, X)                                                                                                 \
    X OP(ACC0, A0, B0) OP(ACC1, A0, B1) X OP(ACC2, A0, B2) OP(ACC3, A0, B3) X OP(ACC4, A0, B0) OP(ACC5, A0, B1)       \
        X OP(ACC6, A0, B2) OP(ACC7, A0, B3)

#define AFIX A0, A0, A0, A0, A0, A0, A0, A0
#define AR2 A0, A1, A0, A1, A0, A1, A0, A1
#define AR4 A0, A1, A2, A3, A0, A1, A2, A3
#define AR8 A0, A1, A2, A3, A4, A5, A6, A7
#define AP2 A0, A0, A1, A1, A2, A2, A3, A3
#define AP4 A0, A0, A0, A0, A1, A1, A1, A1
#define BFIX B0, B0, B0, B0, B0, B0, B0, B0
#define BR4 B0, B1, B2, B3, B0, B1, B2, B3
#define BR2 B0, B1, B0, B1, B0, B1, B0, B1
#define BR8 B0, B1, B2, B3, B4, B5, B6, B7
#define BP2 B0, B0, B1, B1, B2, B2, B3, B3
#define BR4h B0h, B1h, B2h, B3h, B0h, B1h, B2h, B3h
#define AFIXh A0h, A0h, A0h, A0h, A0h, A0h, A0h, A0h
#define AR4h A0h, A1h, A2h, A3h, A0h, A1h, A2h, A3h
#define BFIXh B0h, B0h, B0h, B0h, B0h, B0h, B0h, B0h

struct Var {
    const char * name;
};
// The variants: X macro (name, period text).
#define VARIANTS(V)                                                                                                    \
    V(0, "f16, nothing between, A B fixed", PERX(F16, XNONE, AFIX, BFIX))                                              \
    V(1, "f16, nothing between, A rotates over 4", PERX(F16, XNONE, AR4, BFIX))                                        \
    V(2, "f16, v_mov between, A B fixed", PERX(F16, XMOV, AFIX, BFIX))                                                 \
    V(3, "f16, v_mov between, A rotates over 2", PERX(F16, XMOV, AR2, BFIX))                                           \
    V(4, "f16, v_mov between, A rotates over 4", PERX(F16, XMOV, AR4, BFIX))                                           \
    V(5, "f16, v_mov between, A rotates over 8", PERX(F16, XMOV, AR8, BFIX))                                           \
    V(6, "f16, v_mov between, B rotates over 4", PERX(F16, XMOV, AFIX, BR4))                                           \
    V(7, "f16, v_mov between, A and B rotate over 4", PERX(F16, XMOV, AR4, BR4))                                       \
    V(8, "f16, v_mov between, A changes every 2nd", PERX(F16, XMOV, AP2, BFIX))                                        \
    V(9, "f16, v_mov between, A changes every 4th", PERX(F16, XMOV, AP4, BFIX))                                        \
    V(10, "f16, v_mov of a constant, A fixed", PERX(F16, XMOVK, AFIX, BFIX))                                           \
    V(11, "f16, v_mov of a constant, A rotates over 4", PERX(F16, XMOVK, AR4, BFIX))                                   \
    V(12, "f16, v_nop between, A fixed", PERX(F16, XVNOP, AFIX, BFIX))                                                 \
    V(13, "f16, v_nop between, A rotates over 4", PERX(F16, XVNOP, AR4, BFIX))                                         \
    V(14, "f16, s_nop 0 between, A fixed", PERX(F16, XSNOP, AFIX, BFIX))                                               \
    V(15, "f16, s_nop 0 between, A rotates over 4", PERX(F16, XSNOP, AR4, BFIX))                                       \
    V(16, "f16, v_fma (3 VGPR reads) between, A fixed", PERX(F16, XFMA, AFIX, BFIX))                                   \
    V(17, "f16, v_fma (3 VGPR reads) between, A rotates over 4", PERX(F16, XFMA, AR4, BFIX))                           \
    V(18, "f16, v_mov reading A set 0, A fixed", PERX(F16, XMOVA, AFIX, BFIX))                                         \
    V(19, "f16, v_mov reading A set 0, A rotates over 4", PERX(F16, XMOVA, AR4, BFIX))                                 \
    V(20, "fp8, nothing between, A fixed", PERX(FP8, XNONE, AFIXh, BFIXh))                                             \
    V(21, "fp8, v_mov between, A fixed", PERX(FP8, XMOV, AFIXh, BFIXh))                                                \
    V(22, "fp8, v_mov between, A rotates over 4", PERX(FP8, XMOV, AR4h, BFIXh))                                      \
    V(23, "f16, nothing between, B rotates over 4", PERX(F16, XNONE, AFIX, BR4))                                      \
    V(24, "f16, v_mov between, B rotates over 2", PERX(F16, XMOV, AFIX, BR2))                                         \
    V(25, "f16, v_mov between, B rotates over 8", PERX(F16, XMOV, AFIX, BR8))                                         \
    V(26, "f16, v_mov between, B changes every 2nd", PERX(F16, XMOV, AFIX, BP2))                                      \
    V(27, "f16, v_nop between, B rotates over 4", PERX(F16, XVNOP, AFIX, BR4))                                        \
    V(28, "f16, s_nop 0 between, B rotates over 4", PERX(F16, XSNOP, AFIX, BR4))                                      \
    V(29, "f16, v_mov of a constant, B rotates over 4", PERX(F16, XMOVK, AFIX, BR4))                                  \
    V(30, "fp8, nothing between, B rotates over 4", PERX(FP8, XNONE, AFIXh, BR4h))                                    \
    V(31, "fp8, v_mov between, B rotates over 4", PERX(FP8, XMOV, AFIXh, BR4h))                                      \
    V(32, "f16, v_mov between, B shifted by 2 VGPRs (overlap)", PERX(F16, XMOV, AFIX, BS2))                           \
    V(33, "f16, v_mov before every 2nd, B rotates over 4", PER_ALT(F16, XMOV))                                        \
    V(34, "f16, v_mov (VOP3) between, B rotates over 4", PERX(F16, XMOV64, AFIX, BR4))                                \
    V(35, "f16, v_nop (VOP3) between, B rotates over 4", PERX(F16, XVNOP64, AFIX, BR4))                               \
    V(36, "f16, v_nop (VOP3) between, A B fixed", PERX(F16, XVNOP64, AFIX, BFIX))
#define NVAR 37

#define CLOBBERS "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61", "v62", "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135"
#define INIT "v_mov_b32 v0, 0\n""v_mov_b32 v1, 0\n""v_mov_b32 v2, 0\n""v_mov_b32 v3, 0\n""v_mov_b32 v4, 0\n""v_mov_b32 v5, 0\n""v_mov_b32 v6, 0\n""v_mov_b32 v7, 0\n""v_mov_b32 v8, 0\n""v_mov_b32 v9, 0\n""v_mov_b32 v10, 0\n""v_mov_b32 v11, 0\n""v_mov_b32 v12, 0\n""v_mov_b32 v13, 0\n""v_mov_b32 v14, 0\n""v_mov_b32 v15, 0\n""v_mov_b32 v16, 0\n""v_mov_b32 v17, 0\n""v_mov_b32 v18, 0\n""v_mov_b32 v19, 0\n""v_mov_b32 v20, 0\n""v_mov_b32 v21, 0\n""v_mov_b32 v22, 0\n""v_mov_b32 v23, 0\n""v_mov_b32 v24, 0\n""v_mov_b32 v25, 0\n""v_mov_b32 v26, 0\n""v_mov_b32 v27, 0\n""v_mov_b32 v28, 0\n""v_mov_b32 v29, 0\n""v_mov_b32 v30, 0\n""v_mov_b32 v31, 0\n""v_mov_b32 v32, 0\n""v_mov_b32 v33, 0\n""v_mov_b32 v34, 0\n""v_mov_b32 v35, 0\n""v_mov_b32 v36, 0\n""v_mov_b32 v37, 0\n""v_mov_b32 v38, 0\n""v_mov_b32 v39, 0\n""v_mov_b32 v40, 0\n""v_mov_b32 v41, 0\n""v_mov_b32 v42, 0\n""v_mov_b32 v43, 0\n""v_mov_b32 v44, 0\n""v_mov_b32 v45, 0\n""v_mov_b32 v46, 0\n""v_mov_b32 v47, 0\n""v_mov_b32 v48, 0\n""v_mov_b32 v49, 0\n""v_mov_b32 v50, 0\n""v_mov_b32 v51, 0\n""v_mov_b32 v52, 0\n""v_mov_b32 v53, 0\n""v_mov_b32 v54, 0\n""v_mov_b32 v55, 0\n""v_mov_b32 v56, 0\n""v_mov_b32 v57, 0\n""v_mov_b32 v58, 0\n""v_mov_b32 v59, 0\n""v_mov_b32 v60, 0\n""v_mov_b32 v61, 0\n""v_mov_b32 v62, 0\n""v_mov_b32 v63, 0\n""v_mov_b32 v64, 0x3c003c00\n""v_mov_b32 v65, 0x3c003c00\n""v_mov_b32 v66, 0x3c003c00\n""v_mov_b32 v67, 0x3c003c00\n""v_mov_b32 v68, 0x3c003c00\n""v_mov_b32 v69, 0x3c003c00\n""v_mov_b32 v70, 0x3c003c00\n""v_mov_b32 v71, 0x3c003c00\n""v_mov_b32 v72, 0x3c003c00\n""v_mov_b32 v73, 0x3c003c00\n""v_mov_b32 v74, 0x3c003c00\n""v_mov_b32 v75, 0x3c003c00\n""v_mov_b32 v76, 0x3c003c00\n""v_mov_b32 v77, 0x3c003c00\n""v_mov_b32 v78, 0x3c003c00\n""v_mov_b32 v79, 0x3c003c00\n""v_mov_b32 v80, 0x3c003c00\n""v_mov_b32 v81, 0x3c003c00\n""v_mov_b32 v82, 0x3c003c00\n""v_mov_b32 v83, 0x3c003c00\n""v_mov_b32 v84, 0x3c003c00\n""v_mov_b32 v85, 0x3c003c00\n""v_mov_b32 v86, 0x3c003c00\n""v_mov_b32 v87, 0x3c003c00\n""v_mov_b32 v88, 0x3c003c00\n""v_mov_b32 v89, 0x3c003c00\n""v_mov_b32 v90, 0x3c003c00\n""v_mov_b32 v91, 0x3c003c00\n""v_mov_b32 v92, 0x3c003c00\n""v_mov_b32 v93, 0x3c003c00\n""v_mov_b32 v94, 0x3c003c00\n""v_mov_b32 v95, 0x3c003c00\n""v_mov_b32 v96, 0x3c003c00\n""v_mov_b32 v97, 0x3c003c00\n""v_mov_b32 v98, 0x3c003c00\n""v_mov_b32 v99, 0x3c003c00\n""v_mov_b32 v100, 0x3c003c00\n""v_mov_b32 v101, 0x3c003c00\n""v_mov_b32 v102, 0x3c003c00\n""v_mov_b32 v103, 0x3c003c00\n""v_mov_b32 v104, 0x3c003c00\n""v_mov_b32 v105, 0x3c003c00\n""v_mov_b32 v106, 0x3c003c00\n""v_mov_b32 v107, 0x3c003c00\n""v_mov_b32 v108, 0x3c003c00\n""v_mov_b32 v109, 0x3c003c00\n""v_mov_b32 v110, 0x3c003c00\n""v_mov_b32 v111, 0x3c003c00\n""v_mov_b32 v112, 0x3c003c00\n""v_mov_b32 v113, 0x3c003c00\n""v_mov_b32 v114, 0x3c003c00\n""v_mov_b32 v115, 0x3c003c00\n""v_mov_b32 v116, 0x3c003c00\n""v_mov_b32 v117, 0x3c003c00\n""v_mov_b32 v118, 0x3c003c00\n""v_mov_b32 v119, 0x3c003c00\n""v_mov_b32 v120, 0x3c003c00\n""v_mov_b32 v121, 0x3c003c00\n""v_mov_b32 v122, 0x3c003c00\n""v_mov_b32 v123, 0x3c003c00\n""v_mov_b32 v124, 0x3c003c00\n""v_mov_b32 v125, 0x3c003c00\n""v_mov_b32 v126, 0x3c003c00\n""v_mov_b32 v127, 0x3c003c00\n" "v_mov_b32 v131, 1.0\n"
template <int VAR, int R>
__global__ __launch_bounds__(32) void rot(uint32_t * out, int n) {
    uint32_t t0, t1, cnt;
#define ROT_CASE(ID, NAME, TEXT)                                                                                       \
    if constexpr (VAR == ID) {                                                                                         \
        if constexpr (R == 4) {                                                                                        \
            asm volatile(INIT "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                      \
                         "s_mov_b32 %2, %3\n"                                                                          \
                         "2:\n" TEXT TEXT TEXT TEXT "s_add_co_i32 %2, %2, -1\ns_cmp_lg_u32 %2, 0\ns_cbranch_scc1 2b\n"   \
                         "s_getreg_b32 %1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                           \
                         : "=&s"(t0), "=&s"(t1), "=&s"(cnt) : "s"(n) : "scc", CLOBBERS);                              \
        } else {                                                                                                       \
            asm volatile(INIT "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                      \
                         "s_mov_b32 %2, %3\n"                                                                          \
                         "2:\n" TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT                                                \
                         "s_add_co_i32 %2, %2, -1\ns_cmp_lg_u32 %2, 0\ns_cbranch_scc1 2b\n"                            \
                         "s_getreg_b32 %1, hwreg(HW_REG_SHADER_CYCLES_LO)\n"                                           \
                         : "=&s"(t0), "=&s"(t1), "=&s"(cnt) : "s"(n) : "scc", CLOBBERS);                              \
        }                                                                                                              \
    }
    VARIANTS(ROT_CASE)
    if (threadIdx.x == 0) {
        out[0] = t1 - t0;
    }
}

typedef void (*RK)(uint32_t *, int);
#define ROT_K(ID, NAME, TEXT) {rot<ID, 4>, rot<ID, 8>, NAME},
struct RVar {
    RK k4, k8;
    const char * name;
};
static const RVar rvars[NVAR] = {VARIANTS(ROT_K)};

int main(int argc, char ** argv) {
    const bool brk = argc > 1 && std::string(argv[1]) == "--break";
    require_arch();
    uint32_t * d;
    CHECK(hipMalloc(&d, 64));
    const int n = 16, launches = 9;
    std::vector<double> cpb(NVAR);
    for (int v = 0; v < NVAR; ++v) {
        std::vector<double> x;
        for (int l = 0; l < launches; ++l) {
            uint32_t c4, c8;
            hipLaunchKernelGGL(rvars[v].k4, dim3(1), dim3(32), 0, 0, d, n);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(&c4, d, 4, hipMemcpyDeviceToHost));
            hipLaunchKernelGGL(rvars[v].k8, dim3(1), dim3(32), 0, 0, d, n);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy(&c8, d, 4, hipMemcpyDeviceToHost));
            x.push_back(((double) c8 - (double) c4) / (4.0 * 8 * n));
        }
        std::sort(x.begin(), x.end());
        cpb[v] = x[x.size() / 2];
        std::printf("RESULT rot %2d %-52s cycles per block median %.3f (min %.3f max %.3f, %d launches)\n", v, rvars[v].name, cpb[v],
                    x.front(), x.back(), launches);
    }
    CHECK(hipFree(d));
    const bool ok = std::abs(cpb[0] - 16.0) < 0.3 && std::abs(cpb[2] - 18.0) < 0.3 &&
                    std::abs(cpb[7] - (brk ? 18.0 : 19.0)) < 0.3;
    std::printf("VERDICT left_rot: %s known answers: nothing between %.2f (16), v_mov A B fixed %.2f (18), v_mov A and B rotating "
                "over 4 %.2f (19)%s\n",
                ok ? "PASS" : "FAIL", cpb[0], cpb[2], cpb[7], brk ? " (broken on purpose)" : "");
    return ok ? 0 : 1;
}
