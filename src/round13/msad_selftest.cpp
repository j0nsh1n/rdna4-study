// CPU-only self-test for src/round13/msad_ref.h (no HIP, no GPU).
// Usage: msad_selftest              all checks must pass (exit 0)
//        msad_selftest --break N    deliberate break N (1..6) must exit non-zero
#include "msad_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_bad = 0;

#define CHK(cond, msg)                                       \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("msad_selftest FAIL: %s\n", (msg));  \
            ++g_bad;                                         \
        }                                                    \
    } while (0)

int main(int argc, char ** argv) {
    int brk = 0;
    if (argc == 3 && std::strcmp(argv[1], "--break") == 0) {
        brk = std::atoi(argv[2]);
        if (brk < 1 || brk > 6) {
            std::fprintf(stderr, "break index must be 1..6\n");
            return 2;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: msad_selftest [--break 1..6]\n");
        return 2;
    }

    // 1. Hand-computed exact vectors for all four SAD/MSAD instructions.
    // s0 = [0x00, 0x10, 0x80, 0xff] (bytes 0..3)
    // s1 = [0x05, 0x00, 0x60, 0x00] (bytes 0..3: bytes 1 and 3 are 0 -> masked in msad_u8!)
    // Byte diffs:
    //   byte 0: |0x00 - 0x05| = 5   (s1.b0 != 0 -> kept in msad_u8!)
    //   byte 1: |0x10 - 0x00| = 16  (s1.b1 == 0 -> masked in msad_u8!)
    //   byte 2: |0x80 - 0x60| = 32  (s1.b2 != 0 -> kept in msad_u8!)
    //   byte 3: |0xff - 0x00| = 255 (s1.b3 == 0 -> masked in msad_u8!)
    const uint32_t s0 = 0xff801000u;
    const uint32_t s1 = 0x00600005u;
    const uint32_t s2 = 0x00001234u;

    // Break 1: pretend msad_u8 masks on src0 instead of src1 (must be caught).
    const uint32_t msad_got = (brk == 1) ? msad_ref::msad_u8(s1, s0, s2) : msad_ref::msad_u8(s0, s1, s2);
    CHK(msad_got == (s2 + 5u + 32u), "msad_u8 masks src1==0 bytes, keeps src0==0 bytes");

    // sad_u8 includes all 4 bytes: 5 + 16 + 32 + 255 = 308
    CHK(msad_ref::sad_u8(s0, s1, s2) == (s2 + 308u), "sad_u8 sums all 4 byte abs diffs");

    // sad_hi_u8 shifts byte sum left by 16 and preserves low 16 bits of s2
    const uint32_t sad_hi_got = (brk == 2) ? msad_ref::sad_u8(s0, s1, s2) : msad_ref::sad_hi_u8(s0, s1, s2);
    CHK(sad_hi_got == (s2 + (308u << 16)), "sad_hi_u8 adds (sum << 16) while preserving low 16 bits");

    // sad_u16: lo16 = |0x1000 - 0x0005| = 4091; hi16 = |0xff80 - 0x0060| = 65312; sum = 69403
    CHK(msad_ref::sad_u16(s0, s1, s2) == (s2 + 4091u + 65312u), "sad_u16 sums two u16 abs diffs");

    // 2. 32-bit wrap-around check (VOP3 clamp=0 wraps mod 2^32)
    CHK(msad_ref::sad_u8(0xffffffffu, 0x00000000u, 0xfffffff0u) == (0xfffffff0u + 4u * 255u),
        "sad_u8 wraps cleanly mod 2^32");
    CHK(msad_ref::msad_u8(0xffffffffu, 0x00000000u, 0xdeadbeefu) == 0xdeadbeefu,
        "msad_u8 with all-zero src1 leaves accumulator unchanged");
    CHK(msad_ref::msad_u8(0x00000000u, 0xffffffffu, 0x00000000u) == 4u * 255u,
        "msad_u8 with all-zero src0 and 0xff src1 sums 4*255");

    // 3. Closed-form accumulator chain matches iterative execution for all 4 ops and all 64 lanes.
    for (int op = 0; op < 4; ++op) {
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t a = msad_ref::lane_s0(lane);
            const uint32_t b = msad_ref::lane_s1(lane);
            const uint32_t x0 = msad_ref::lane_init_acc(lane, 3u);
            uint32_t iter = x0;
            for (int k = 0; k < 257; ++k) iter = msad_ref::eval_sad_op(op, a, b, iter);
            const uint32_t closed = msad_ref::run_acc_chain(op, a, b, x0, 257u);
            CHK(iter == closed, "closed-form acc chain matches iterative evaluation");
        }
    }

    // 4. Exact dyadic FP32 FMA chain: f_{N} == init + N * 0.0625f bit-for-bit.
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const float f0 = msad_ref::lane_init_f(lane, 5u);
        const uint32_t steps = 16384u;
        const float got = msad_ref::run_fma_chain(f0, 1.0f, 0.0625f, steps);
        const float want = f0 + static_cast<float>(steps) * 0.0625f + (brk == 3 ? 0.0625f : 0.0f);
        CHK(got == want, "dyadic FP32 FMA chain is bit-exact over 16384 steps");
    }

    // 5. Differential timing and linearity math (U = 4, 6, 8; ITERS = 128; PATTERNS = 16).
    {
        const double ovh = 20.0;
        const double per = 1.25;
        const int iters = 128;
        const double lo = iters * (ovh + 4.0 * 16.0 * per);
        const double mid = iters * (ovh + 6.0 * 16.0 * per + (brk == 4 ? 3.0 : 0.0));
        const double hi = iters * (ovh + 8.0 * 16.0 * per);
        const msad_ref::DiffCost dc = msad_ref::diff_cost(lo, mid, hi, 4, 8, iters, 16.0);
        CHK(std::fabs(dc.per - per) < 1e-9, "diff_cost recovers exact cycles/pattern");
        CHK(std::fabs(dc.ovh - ovh) < 1e-9, "diff_cost recovers exact loop overhead");
        CHK(dc.linear, "diff_cost linearity check catches 3 cyc/iter non-linearity");
    }

    // 6. Preregistered FMA calibration gate: wave32 accepts 1.00 +- 0.10 only; wave64 accepts 1.00 or 2.00 +- 0.10;
    //    both still require exact bits and linearity.
    {
        const msad_ref::DiffCost lin16 = {1.00, 0.0, 20.0, true};
        const msad_ref::DiffCost lin1  = {5.00, 0.0, 20.0, true};
        const msad_ref::DiffCost nonlin = {1.00, 3.0, 20.0, false};

        CHK(msad_ref::fma_calib_ok(32, 0, lin16, lin1, 1.00), "wave32 accepts 1.00 cyc/op");
        CHK(msad_ref::fma_calib_ok(32, 0, lin16, lin1, 0.92), "wave32 accepts 0.92 cyc/op (within +-0.10)");
        CHK(msad_ref::fma_calib_ok(32, 0, lin16, lin1, 1.08), "wave32 accepts 1.08 cyc/op (within +-0.10)");
        CHK(msad_ref::fma_calib_ok(32, 0, lin16, lin1, 2.00) == (brk == 6),
            "wave32 rejects 2.00 cyc/op");
        CHK(!msad_ref::fma_calib_ok(32, 1, lin16, lin1, 1.00), "wave32 rejects bad bits");
        CHK(!msad_ref::fma_calib_ok(32, 0, nonlin, lin1, 1.00), "wave32 rejects non-linear c16");
        CHK(!msad_ref::fma_calib_ok(32, 0, lin16, nonlin, 1.00), "wave32 rejects non-linear c1");

        CHK(msad_ref::fma_calib_ok(64, 0, lin16, lin1, 1.00), "wave64 accepts 1.00 cyc/op");
        CHK(msad_ref::fma_calib_ok(64, 0, lin16, lin1, 2.00) == (brk != 5),
            "wave64 accepts preregistered 2.00 cyc/op");
        CHK(msad_ref::fma_calib_ok(64, 0, lin16, lin1, 1.92), "wave64 accepts 1.92 cyc/op (within +-0.10)");
        CHK(msad_ref::fma_calib_ok(64, 0, lin16, lin1, 2.08), "wave64 accepts 2.08 cyc/op (within +-0.10)");
        CHK(!msad_ref::fma_calib_ok(64, 0, lin16, lin1, 1.50), "wave64 rejects 1.50 cyc/op");
        CHK(!msad_ref::fma_calib_ok(64, 0, lin16, lin1, 2.20), "wave64 rejects 2.20 cyc/op");
        CHK(!msad_ref::fma_calib_ok(64, 0, lin16, lin1, 5.00), "wave64 rejects 5.00 cyc/op (break case)");
        CHK(!msad_ref::fma_calib_ok(64, 1, lin16, lin1, 2.00), "wave64 rejects bad bits at 2.00");
        CHK(!msad_ref::fma_calib_ok(64, 0, nonlin, lin1, 2.00), "wave64 rejects non-linear c16 at 2.00");
        CHK(!msad_ref::fma_calib_ok(64, 0, lin16, nonlin, 2.00), "wave64 rejects non-linear c1 at 2.00");
    }

    std::printf("VERDICT msad_selftest: %s (bad=%d%s)\n",
                g_bad == 0 ? "PASS" : "FAIL", g_bad, brk ? ", deliberate break" : "");
    return g_bad == 0 ? 0 : 1;
}
