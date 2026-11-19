// Lane 8, item 2: hazards the compiler fixes on gfx12, and attempts to get one past it from plain HIP.
// Matrix hazard: a matrix instruction whose A or B is the previous matrix instruction's result needs one VALU
// (V_NOP) between them (AMD's guide, LLVM fixWMMAHazards). Each kernel below makes that dependency in a different
// source shape; the card result is compared with the CPU model. scripts/haz_hazscan.py checks the listing.
// The break (mode "break") runs the same dependency in inline assembly without the V_NOP; it must give stale values.
// Also: the SGPR read hazard (s_wait_alu) and v_cmpx before v_permlane16, with their cost.
// Usage: haz_hazard [all|break]
#include "haz_common.h"

using namespace haz;

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

#define WMMA16(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12((a), (b), (c))
#define WMMA32(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

// Straight line: D1 (16-bit result) is the next instruction's A.
__global__ void h_straight(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h d1 = WMMA16(A[l], B[l], C16[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// D1 as B.
__global__ void h_bsrc(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h d1 = WMMA16(A[l], B[l], C16[l]);
    D[l] = WMMA32(B2[l], d1, C2[l]);
}

// Across a loop back-edge: the result is the next iteration's A.
__global__ void h_loop(const v8h * A, const v8h * B, const v8h * C16, int n, v8h * D16) {
    const int l = threadIdx.x;
    v8h x = A[l];
    const v8h b = B[l], c = C16[l];
    for (int i = 0; i < n; ++i) {
        x = WMMA16(x, b, c);
    }
    D16[l] = x;
}

// Across basic blocks: a wave-uniform branch with only a load in it between the two matrix instructions.
__global__ void h_branch(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D,
                         int flag) {
    const int l = threadIdx.x;
    const v8h d1 = WMMA16(A[l], B[l], C16[l]);
    v8f c2 = C2[l];
    if (flag) {
        c2 = C2[32 + l];
    }
    D[l] = WMMA32(d1, B2[l], c2);
}

// Across a call: the first matrix instruction is the last thing in a function that is not inlined.
__device__ __attribute__((noinline)) v8h first_mm(v8h a, v8h b, v8h c) { return WMMA16(a, b, c); }
__global__ void h_call(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h d1 = first_mm(A[l], B[l], C16[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// Across a call, with the result left in the return registers (C is the first argument, so D needs no copy).
__device__ __attribute__((noinline)) v8h first_mm_tied(v8h c, v8h a, v8h b) { return WMMA16(a, b, c); }
__global__ void h_call2(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, B2[l], C2[l]);
}

// As h_call2, with the second instruction's other operands loaded before the call, so nothing is left to do
// between the return and the second matrix instruction.
__global__ void h_call3(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v8h b2 = B2[l];
    v8f c2 = C2[l];
    asm volatile("" : "+v"(b2), "+v"(c2));
    const v8h d1 = first_mm_tied(C16[l], A[l], B[l]);
    const v8f d = WMMA32(d1, b2, c2);
    D[l] = d;
}

// The same in plain C++: operands read before the call (an opaque call may write memory, so they stay there).
__global__ void h_call4(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h b2 = B2[l];
    const v8f c2 = C2[l];
    const v8h d1 = first_mm_tied(C16[l], A[l], B[l]);
    D[l] = WMMA32(d1, b2, c2);
}

// Mixed types: the FP16 result's bits used as BF16 A.
__global__ void h_mixed(const v8h * A, const v8h * B, const v8h * C16, const v8bf * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    const v8h d1 = WMMA16(A[l], B[l], C16[l]);
    D[l] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(v8bf, d1), B2[l], C2[l]);
}

// Inline assembly: the same as h_straight with the V_NOP, and without it (forbidden; the break).
__global__ void h_asm_nop(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v8h d1 = C16[l];
    v8f d2;
    asm volatile(
        "v_wmma_f16_16x16x16_f16 %0, %2, %3, %0\n\t"
        "v_nop\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %0, %4, %5"
        : "+&v"(d1), "=&v"(d2)
        : "v"(A[l]), "v"(B[l]), "v"(B2[l]), "v"(C2[l]));
    D[l] = d2;
}
__global__ void h_asm_raw(const v8h * A, const v8h * B, const v8h * C16, const v8h * B2, const v8f * C2, v8f * D) {
    const int l = threadIdx.x;
    v8h d1 = C16[l];
    v8f d2;
    asm volatile(
        "v_wmma_f16_16x16x16_f16 %0, %2, %3, %0\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %0, %4, %5"
        : "+&v"(d1), "=&v"(d2)
        : "v"(A[l]), "v"(B[l]), "v"(B2[l]), "v"(C2[l]));
    D[l] = d2;
}

// SGPR read hazard: a scalar value read by VALU, rewritten by SALU each iteration, read again by VALU.
__global__ void s_hazard(const float * X, float * Y, uint32_t seed, int n, uint64_t * cyc) {
    const int l = threadIdx.x;
    float x = X[l];
    uint32_t s = seed;
    const uint64_t t0 = wave_cycles();
    for (int i = 0; i < n; ++i) {
        x = fmaf(x, 0.5f, __uint_as_float((s & 0x007FFFFFu) | 0x3F800000u));
        s = s * 1664525u + 1013904223u;
    }
    const uint64_t t1 = wave_cycles();
    Y[l] = x;
    if (l == 0) {
        *cyc = cycle_diff(t0, t1);
    }
}

// The same hazard sequence in inline assembly (VALU reads s20; SALU rewrites s20; VALU reads s20), with the wait
// the compiler would insert (W = 1) or without it (W = 0).
#define S_HAZARD_ASM(NAME, WAIT)                                                                       \
    __global__ void NAME(const float * X, float * Y, uint32_t seed, int n, uint64_t * cyc) {           \
        const int l = threadIdx.x;                                                                     \
        float x = X[l];                                                                                \
        uint32_t s = seed;                                                                             \
        int k = n;                                                                                     \
        const uint64_t t0 = wave_cycles();                                                             \
        asm volatile("1:\n\t"                                                                         \
                     "s_and_b32 s20, %[s], 0x7fffff\n\t"                                             \
                     "s_or_b32 s20, s20, 0x3f800000\n\t" WAIT                                         \
                     "v_fma_f32 %[x], %[x], 0.5, s20\n\t"                                             \
                     "s_mul_i32 %[s], %[s], 0x19660d\n\t"                                             \
                     "s_add_co_i32 %[s], %[s], 0x3c6ef35f\n\t"                                        \
                     "s_add_co_i32 %[k], %[k], -1\n\t"                                                \
                     "s_cmp_lg_u32 %[k], 0\n\t"                                                       \
                     "s_cbranch_scc1 1b"                                                               \
                     : [x] "+v"(x), [s] "+s"(s), [k] "+s"(k)                                           \
                     :                                                                                 \
                     : "s20", "scc");                                                                  \
        const uint64_t t1 = wave_cycles();                                                             \
        Y[l] = x;                                                                                      \
        if (l == 0) {                                                                                  \
            *cyc = cycle_diff(t0, t1);                                                                 \
        }                                                                                              \
    }
S_HAZARD_ASM(s_hazard_wait, "s_wait_alu depctr_sa_sdst(0)\n\t")
S_HAZARD_ASM(s_hazard_nowait, "")

// v_cmpx before v_permlane16: a per-lane trip count (the loop-entry test became v_cmpx in haz_mask's k_loop) with
// the lane permute as the loop's first instruction. Lane l runs t = X[64 + l] rounds of x = permlane16(x) (row
// reversal, fetch-inactive set); x is computed by every lane before the loop.
__global__ void p_cmpx(const uint32_t * X, uint32_t * Y, int n) {
    const int l = threadIdx.x;
    uint32_t x = X[l];
    asm volatile("" : "+v"(x));
    const int t = (int) X[64 + l];
    (void) n;
    for (int i = 0; i < t; ++i) {
        x = __builtin_amdgcn_permlane16(x, x, 0x89ABCDEFu, 0x01234567u, true, false);
    }
    Y[l] = x;
}

namespace {

// Exact FP16-accumulator result for small integers (every value exact in FP16, so rounding cannot enter).
std::vector<uint16_t> ref_acc16(const std::vector<uint16_t> & a, const std::vector<uint16_t> & b,
                                const std::vector<uint16_t> & c, bool & exact) {
    auto hv = [](uint16_t u) {
        _Float16 h;
        std::memcpy(&h, &u, 2);
        return (double) h;
    };
    std::vector<uint16_t> d(256);
    exact = true;
    for (int r = 0; r < 16; ++r) {
        for (int col = 0; col < 16; ++col) {
            const int lane = 16 * (r >> 3) + col, el = r & 7;
            double s = hv(c[lane * 8 + el]);
            for (int k = 0; k < 16; ++k) {
                s += hv(a[(r + 16 * (k >> 3)) * 8 + (k & 7)]) * hv(b[(col + 16 * (k >> 3)) * 8 + (k & 7)]);
            }
            exact = exact && std::fabs(s) < 2048 && s == std::floor(s);
            d[lane * 8 + el] = h16((float) s);
        }
    }
    return d;
}

template <class T>
T * dalloc(size_t n) {
    T * p;
    CHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    require_arch();
    Verdicts v;
    const Problem p1 = make_problem(3), p2 = make_problem(5);
    std::vector<uint16_t> c16(256);
    for (int i = 0; i < 256; ++i) {
        c16[i] = h16((float) ((int) (i * 7 % 5) - 2));
    }
    bool exact;
    const auto d1 = ref_acc16(p1.a, p1.b, c16, exact);
    v.put("hazard_ref_exact", exact, "first product is exact in FP16");
    Problem qa = p2, qb = p2;  // second instruction: D1 as A, or D1 as B
    qa.a = d1;
    qb.a = p2.b;
    qb.b = d1;
    const auto ref_a = reference(qa), ref_b = reference(qb);
    Problem qs = p2;  // stale read: A (or B) = C16, the registers' old contents
    qs.a = c16;
    const auto ref_stale = reference(qs);
    Problem qm = p2;  // mixed: D1's bits as BF16, B2 = BF16 bits of small integers
    qm.a = d1;
    for (int i = 0; i < 256; ++i) {
        const float f = (float) ((int) (i * 5 % 7) - 3);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        qm.b[i] = (uint16_t) (u >> 16);
    }
    const auto ref_m = reference(qm, am::F_BF16);

    v8h * A = (v8h *) dev_copy(p1.a);
    v8h * B = (v8h *) dev_copy(p1.b);
    v8h * C16 = (v8h *) dev_copy(c16);
    v8h * B2 = (v8h *) dev_copy(p2.b);
    v8bf * B2m = (v8bf *) dev_copy(qm.b);
    std::vector<uint32_t> c2x(512);
    for (int i = 0; i < 256; ++i) {
        c2x[i] = p2.c[i];
        c2x[256 + i] = p2.c[i];
    }
    v8f * C2 = (v8f *) dev_copy(c2x);
    v8f * D = dalloc<v8f>(32);
    auto get = [&]() {
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        return host_copy((const uint32_t *) D, 256);
    };
    auto diff = [](const std::vector<uint32_t> & x, const std::vector<uint32_t> & y) {
        int w = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            w += x[i] != y[i];
        }
        return w;
    };

    if (mode == "break") {
        // Fix removed: no V_NOP. The documents call this forbidden; lane 5 measured that it reads stale registers.
        hipLaunchKernelGGL(h_asm_raw, dim3(1), dim3(32), 0, 0, A, B, C16, B2, C2, D);
        const auto d = get();
        v.put("hazard_asm_no_vnop", diff(d, ref_a) == 0,
              fmt("no V_NOP: %d of 256 differ from the model; %d of 256 equal the stale-read prediction (A = old "
                  "registers)",
                  diff(d, ref_a), 256 - diff(d, ref_stale)));
        return v.exit_code();
    }

    struct K {
        const char * name;
        void (*k)(const v8h *, const v8h *, const v8h *, const v8h *, const v8f *, v8f *);
        const v8h * b2;
        const std::vector<uint32_t> * ref;
    } ks[] = {{"hazard_straight", h_straight, B2, &ref_a},
              {"hazard_bsrc", h_bsrc, (const v8h *) dev_copy(p2.b), &ref_b},
              {"hazard_call", h_call, B2, &ref_a},
              {"hazard_call_tied", h_call2, B2, &ref_a},

              {"hazard_asm_vnop", h_asm_nop, B2, &ref_a}};
    for (const K & k : ks) {
        hipLaunchKernelGGL(k.k, dim3(1), dim3(32), 0, 0, A, B, C16, k.b2, C2, D);
        const auto d = get();
        v.put(k.name, diff(d, *k.ref) == 0, fmt("%d of 256 differ from the model", diff(d, *k.ref)));
    }
    {
        // Missed case: the callee ends with the matrix instruction and returns, the caller's next instruction is the
        // matrix instruction reading that result as A; no V_NOP on either side (haz_hazscan.py: RETURN and CALL).
        // The check passes while the miss reproduces: the result is wrong, and equal bit for bit to the forbidden
        // no-V_NOP sequence in inline assembly. If a compiler update fixes the miss, this fails and the write-up
        // must change.
        hipLaunchKernelGGL(h_asm_raw, dim3(1), dim3(32), 0, 0, A, B, C16, B2, C2, D);
        const auto raw = get();
        void (*kc[2])(const v8h *, const v8h *, const v8h *, const v8h *, const v8f *, v8f *) = {h_call3, h_call4};
        const char * nc[2] = {"with an empty asm ordering the loads", "plain C++"};
        for (int k = 0; k < 2; ++k) {
            hipLaunchKernelGGL(kc[k], dim3(1), dim3(32), 0, 0, A, B, C16, B2, C2, D);
            const auto d = get();
            v.put("hazard_call_miss", diff(d, ref_a) > 0 && diff(d, raw) == 0,
                  fmt("%s: %d of 256 differ from the model; %d of 256 differ from the no-V_NOP inline-assembly "
                      "result (the miss reproduces when this is 0 and the first is not)",
                      nc[k], diff(d, ref_a), diff(d, raw)));
        }
        hipLaunchKernelGGL(h_asm_raw, dim3(1), dim3(32), 0, 0, A, B, C16, B2, C2, D);
        const auto raw2 = get();
        v.info("hazard_asm_no_vnop", fmt("no V_NOP (inline assembly): %d of 256 differ from the model; repeat equal: %s",
                                         diff(raw, ref_a), raw == raw2 ? "yes" : "no"));
    }
    // h_bsrc computes B2 x D1 with B2 = p2.b as A: qb has a = p2.b, b = d1.
    for (int flag = 0; flag < 2; ++flag) {
        hipLaunchKernelGGL(h_branch, dim3(1), dim3(32), 0, 0, A, B, C16, B2, C2, D, flag);
        const auto d = get();
        v.put("hazard_branch", diff(d, ref_a) == 0, fmt("flag=%d: %d of 256 differ from the model", flag, diff(d, ref_a)));
    }
    {
        hipLaunchKernelGGL(h_mixed, dim3(1), dim3(32), 0, 0, A, B, C16, B2m, C2, D);
        const auto d = get();
        v.put("hazard_mixed", diff(d, ref_m) == 0, fmt("FP16 result as BF16 A: %d of 256 differ from the model", diff(d, ref_m)));
    }
    {
        // Loop: B = a permutation matrix, so every iteration's result stays a small exact integer.
        std::vector<uint16_t> bp(256, 0);
        for (int col = 0; col < 16; ++col) {
            const int s = (col * 5 + 3) & 15;
            bp[(col + 16 * (s >> 3)) * 8 + (s & 7)] = h16(1.0f);
        }
        v8h * BP = (v8h *) dev_copy(bp);
        v8h * D16 = dalloc<v8h>(32);
        for (int n : {1, 2, 5}) {
            std::vector<uint16_t> x = p1.a;
            bool ex = true;
            for (int i = 0; i < n; ++i) {
                bool e;
                x = ref_acc16(x, bp, c16, e);
                ex = ex && e;
            }
            hipLaunchKernelGGL(h_loop, dim3(1), dim3(32), 0, 0, A, BP, C16, n, D16);
            CHECK(hipDeviceSynchronize());
            const auto d = host_copy((const uint16_t *) D16, 256);
            int w = 0;
            for (int i = 0; i < 256; ++i) {
                w += d[i] != x[i];
            }
            v.put("hazard_loop", ex && w == 0, fmt("n=%d (result is next A): %d of 256 differ from the exact result", n, w));
        }
    }
    {
        // SGPR read hazard: right result, and cost per iteration (n = 2048 against 1024).
        std::vector<float> X(32);
        for (int i = 0; i < 32; ++i) {
            X[i] = (float) i;
        }
        float * dX = dev_copy(X);
        float * dY = dalloc<float>(32);
        uint64_t * cyc = dalloc<uint64_t>(1);
        struct SK {
            const char * name;
            void (*k)(const float *, float *, uint32_t, int, uint64_t *);
        } sks[] = {{"compiler", s_hazard}, {"asm_wait", s_hazard_wait}, {"asm_nowait", s_hazard_nowait}};
        for (const SK & sk : sks) {
        double c[2];
        for (int j = 0; j < 2; ++j) {
            const int n = 1024 << j;
            double best = 1e30;
            for (int r = 0; r < 5; ++r) {
                hipLaunchKernelGGL(sk.k, dim3(1), dim3(32), 0, 0, dX, dY, 12345u, n, cyc);
                CHECK(hipDeviceSynchronize());
                best = std::min(best, (double) host_copy(cyc, 1)[0]);
            }
            c[j] = best;
            const auto y = host_copy(dY, 32);
            int w = 0;
            for (int l = 0; l < 32; ++l) {
                float x = X[l];
                uint32_t s = 12345u;
                for (int i = 0; i < n; ++i) {
                    float k;
                    const uint32_t u = (s & 0x007FFFFFu) | 0x3F800000u;
                    std::memcpy(&k, &u, 4);
                    x = std::fma(x, 0.5f, k);
                    s = s * 1664525u + 1013904223u;
                }
                w += f32b(x) != f32b(y[l]);
            }
            const bool must = std::string(sk.name) != "asm_nowait";
            if (must) {
                v.put("hazard_sgpr", w == 0, fmt("%s n=%d: %d of 32 lanes differ from the CPU", sk.name, n, w));
            } else {
                v.info("hazard_sgpr_nowait", fmt("%s n=%d: %d of 32 lanes differ from the CPU", sk.name, n, w));
            }
        }
        v.info("hazard_sgpr_time", fmt("%s: %.2f cycles per loop iteration (n=2048 minus n=1024)", sk.name,
                                       (c[1] - c[0]) / 1024));
        }
    }
    {
        std::vector<uint32_t> X(96);
        for (int i = 0; i < 32; ++i) {
            X[i] = 100 + i;
            X[64 + i] = i;
        }
        uint32_t * dX = dev_copy(X);
        uint32_t * dY = dalloc<uint32_t>(32);
        for (int n : {7, 16, 31}) {
            for (int l = 0; l < 32; ++l) {
                X[64 + l] = l < n ? 1 : 0;
            }
            CHECK(hipMemcpy(dX, X.data(), X.size() * 4, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(p_cmpx, dim3(1), dim3(32), 0, 0, dX, dY, n);
            CHECK(hipDeviceSynchronize());
            const auto y = host_copy(dY, 32);
            int w = 0;
            for (int l = 0; l < 32; ++l) {
                // Lanes < n run one round and read their row partner (lane (l & 16) + 15 - (l & 15)), also when
                // that partner has left the loop (fetch-inactive); the others keep their own x.
                const uint32_t want = l < n ? X[(l & 16) + 15 - (l & 15)] : X[l];
                w += y[l] != want;
            }
            v.put("hazard_cmpx_permlane", w == 0, fmt("n=%d: %d of 32 lanes wrong", n, w));
        }
    }
    return v.exit_code();
}
