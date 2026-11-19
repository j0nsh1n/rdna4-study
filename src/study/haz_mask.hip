// Lane 8, item 1: matrix instructions under a lane mask.
// Part A (compiler): natural HIP kernels with the gfx12 matrix builtin where only some lanes are valid. The listing
// scan (scripts/haz_maskscan.py) says which of them run the matrix instruction under a partial EXEC; this program
// says what the card computes in each, against the CPU model.
// Part B (hardware rule): the matrix instruction in inline assembly with EXEC set to a partial mask, to see which
// lanes' operands are read, which result lanes are written, and whether the time changes.
// Usage: haz_mask [all|break]. "break" runs the asm probe with the result check pointed at the wrong lanes.
#include "haz_common.h"

using namespace haz;

#define WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))

// The result is stored only by lanes below n; operands are loaded by every lane.
__global__ void k_sink(const v8h * A, const v8h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    const v8h a = A[l], b = B[l];
    const v8f c = C[l];
    const v8f d = WMMA(a, b, c);
    if (l < n) {
        D[l] = d;
    }
}

// Everything, loads included, under a per-lane if.
__global__ void k_ifload(const v8h * A, const v8h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    if (l < n) {
        D[l] = WMMA(A[l], B[l], C[l]);
    }
}

// Early return for lanes past the end of a partly filled tile.
__global__ void k_early(const v8h * A, const v8h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    if (l >= n) {
        return;
    }
    D[l] = WMMA(A[l], B[l], C[l]);
}

// Per-lane trip count through the accumulator.
__global__ void k_loop(const v8h * A, const v8h * B, const v8f * C, v8f * D, const int * trip) {
    const int l = threadIdx.x;
    v8f c = C[l];
    const int t = trip[l];
    for (int i = 0; i < t; ++i) {
        c = WMMA(A[i * 32 + l], B[i * 32 + l], c);
    }
    D[l] = c;
}

// The partial-tile pattern written safely: padding by select (every lane runs the matrix instruction), store by all.
__global__ void k_select(const v8h * A, const v8h * B, const v8f * C, v8f * D, int n) {
    const int l = threadIdx.x;
    const bool ok = (l & 15) < n;
    const v8h z = {};
    const v8h a = ok ? A[l] : z;
    const v8f d = WMMA(a, B[l], C[l]);
    D[l] = d;
}

// The early-return pattern with a wave-uniform bound: the branch is scalar, EXEC stays full.
__global__ void k_uniform(const v8h * A, const v8h * B, const v8f * C, v8f * D, int ntiles) {
    if ((int) blockIdx.x >= ntiles) {
        return;
    }
    const int l = threadIdx.x;
    D[l] = WMMA(A[l], B[l], C[l]);
}

// Part B. EXEC is set to `mask` around one matrix instruction; D starts as a sentinel image so unwritten lanes show.
__global__ void k_asm(const v8h * A, const v8h * B, const v8f * C, v8f * D, uint32_t mask) {
    const int l = threadIdx.x;
    const v8h a = A[l], b = B[l];
    const v8f c = C[l];
    v8f d = D[l];
    asm volatile(
        "s_mov_b32 s96, exec_lo\n\t"
        "s_mov_b32 exec_lo, %4\n\t"
        "v_wmma_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        "s_mov_b32 exec_lo, s96"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(c), "s"(mask)
        : "s96");
    D[l] = d;
}

#define W16(x) x x x x x x x x x x x x x x x x
// Timing: 64 chained matrix instructions under `mask`, cycle counter around them (one wave alone on its SIMD).
__global__ void k_asm_time(const v8h * A, const v8h * B, const v8f * C, v8f * D, uint32_t mask, uint64_t * cyc) {
    const int l = threadIdx.x;
    const v8h a = A[l], b = B[l];
    v8f d = C[l];
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 4; ++rep) {
        const uint64_t t0 = wave_cycles();
        asm volatile(
            "s_mov_b32 s96, exec_lo\n\t"
            "s_mov_b32 exec_lo, %3\n\t"
            W16("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n\t")
            W16("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n\t")
            W16("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n\t")
            W16("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0\n\t")
            "s_mov_b32 exec_lo, s96\n\t"
            "v_nop"
            : "+v"(d)
            : "v"(a), "v"(b), "s"(mask)
            : "s96");
        const uint64_t t1 = wave_cycles();
        best = cycle_diff(t0, t1) < best ? cycle_diff(t0, t1) : best;
    }
    D[l] = d;
    if (l == 0) {
        *cyc = best;
    }
}

namespace {

struct Bufs {
    v8h *A, *B;
    v8f *C, *D;
    int * trip;
    uint64_t * cyc;
};

const uint32_t SENT = 0x7FBADBADu;

// Runs a part-A kernel `runs` times; D starts as the sentinel each time. Returns the D images.
template <class Launch>
std::vector<std::vector<uint32_t>> run_a(Bufs & b, Launch launch, int runs) {
    std::vector<std::vector<uint32_t>> out;
    for (int r = 0; r < runs; ++r) {
        std::vector<uint32_t> s(256 * 4, SENT);
        CHECK(hipMemcpy(b.D, s.data(), 256 * 4, hipMemcpyHostToDevice));
        launch();
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> d(256);
        CHECK(hipMemcpy(d.data(), b.D, 256 * 4, hipMemcpyDeviceToHost));
        out.push_back(d);
    }
    return out;
}

// Wrong elements among the lanes in `lanes` (bit mask), against ref.
int wrong_in(const std::vector<uint32_t> & d, const std::vector<uint32_t> & ref, uint32_t lanes) {
    int w = 0;
    for (int l = 0; l < 32; ++l) {
        if (!((lanes >> l) & 1)) {
            continue;
        }
        for (int e = 0; e < 8; ++e) {
            w += d[l * 8 + e] != ref[l * 8 + e];
        }
    }
    return w;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    require_arch();
    Verdicts v;
    const Problem p = make_problem(7);
    const std::vector<uint32_t> ref = reference(p);
    Bufs b{};
    b.A = (v8h *) dev_copy(std::vector<uint16_t>(p.a.begin(), p.a.end()));
    b.B = (v8h *) dev_copy(std::vector<uint16_t>(p.b.begin(), p.b.end()));
    b.C = (v8f *) dev_copy(p.c);
    CHECK(hipMalloc(&b.D, 256 * 4));
    CHECK(hipMalloc(&b.cyc, 8));

    if (mode == "all") {
        // Part A. Lane masks lane < n for n in {8, 16, 20, 31}; valid output = stored lanes; reference = all lanes'
        // operands as they are in memory (what the programmer of a partly filled tile means).
        const int ns[] = {8, 16, 20, 31, 32};
        struct K {
            const char * name;
            void (*k)(const v8h *, const v8h *, const v8f *, v8f *, int);
        } ks[] = {{"sink", k_sink}, {"ifload", k_ifload}, {"early", k_early}};
        for (const K & k : ks) {
            for (int n : ns) {
                auto outs = run_a(b, [&] { hipLaunchKernelGGL(k.k, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, n); }, 3);
                const uint32_t lanes = n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1);
                const int w = wrong_in(outs[0], ref, lanes);
                const bool det = outs[0] == outs[1] && outs[1] == outs[2];
                v.info(("mask_" + std::string(k.name)).c_str(),
                       fmt("n=%d: %d of %d stored results differ from the CPU model; deterministic over 3 runs: %s",
                           n, w, 8 * __builtin_popcount(lanes), det ? "yes" : "no"));
            }
        }
        // Select padding and wave-uniform bound: must be right (the reference uses zero A for padded rows).
        for (int n : {5, 16}) {
            Problem q = p;
            for (int l = 0; l < 32; ++l) {
                if ((l & 15) >= n) {
                    for (int e = 0; e < 8; ++e) {
                        q.a[l * 8 + e] = 0;
                    }
                }
            }
            const auto rq = reference(q);
            auto outs = run_a(b, [&] { hipLaunchKernelGGL(k_select, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, n); }, 2);
            const int w = wrong_in(outs[0], rq, 0xFFFFFFFFu);
            v.put("mask_select", w == 0, fmt("rows >= %d padded by select: %d of 256 wrong", n, w));
        }
        {
            auto outs = run_a(b, [&] { hipLaunchKernelGGL(k_uniform, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, 1); }, 2);
            const int w = wrong_in(outs[0], ref, 0xFFFFFFFFu);
            v.put("mask_uniform", w == 0, fmt("wave-uniform early return: %d of 256 wrong", w));
        }
        // Per-lane trip count: lanes 0..15 run 2 iterations, lanes 16..31 one. Data: iteration 0 = p, iteration 1
        // = a second problem's A and B. Compare with: (u) every lane runs 2 iterations; (s) the second iteration
        // reads the short lanes' stale iteration-0 operands (registers keep their old values).
        {
            const Problem p2 = make_problem(11);
            std::vector<uint16_t> A2(512), B2(512);
            for (int i = 0; i < 256; ++i) {
                A2[i] = p.a[i];
                B2[i] = p.b[i];
                A2[256 + i] = p2.a[i];
                B2[256 + i] = p2.b[i];
            }
            v8h * dA = (v8h *) dev_copy(A2);
            v8h * dB = (v8h *) dev_copy(B2);
            for (int shape = 0; shape < 2; ++shape) {
                std::vector<int> trip(32);
                for (int l = 0; l < 32; ++l) {
                    trip[l] = shape == 0 ? 2 : (l < 16 ? 2 : 1);
                }
                int * dt = dev_copy(trip);
                auto outs = run_a(b, [&] { hipLaunchKernelGGL(k_loop, dim3(1), dim3(32), 0, 0, dA, dB, b.C, b.D, dt); }, 3);
                Problem u = p2;
                u.c = ref;
                const auto ru = reference(u);
                Problem s = u;
                for (int l = 16; l < 32; ++l) {
                    for (int e = 0; e < 8; ++e) {
                        s.a[l * 8 + e] = p.a[l * 8 + e];
                        s.b[l * 8 + e] = p.b[l * 8 + e];
                    }
                }
                const auto rs = reference(s);
                const bool det = outs[0] == outs[1] && outs[1] == outs[2];
                if (shape == 0) {
                    v.put("mask_loop_uniform", wrong_in(outs[0], ru, 0xFFFFFFFFu) == 0,
                          fmt("all lanes 2 trips: %d of 256 wrong", wrong_in(outs[0], ru, 0xFFFFFFFFu)));
                } else {
                    v.info("mask_loop_perlane",
                           fmt("lanes 0-15 two trips, 16-31 one: lanes 0-15 differ from 'all lanes run 2' in %d of "
                               "128, from 'stale operands of lanes 16-31' in %d of 128; lanes 16-31 equal one-trip "
                               "result in %d of 128 and the two-lane-group stale result in %d of 128; deterministic: %s",
                               wrong_in(outs[0], ru, 0xFFFFu), wrong_in(outs[0], rs, 0xFFFFu),
                               128 - wrong_in(outs[0], ref, 0xFFFF0000u), 128 - wrong_in(outs[0], rs, 0xFFFF0000u),
                               det ? "yes" : "no"));
                }
                CHECK(hipFree(dt));
            }
        }
    }

    // Part B: the hardware rule.
    const uint32_t masks[] = {0xFFFFFFFFu, 0x0000FFFFu, 0xFFFF0000u, 0x55555555u, 0x11111111u, 0x00000001u, 0u};
    for (uint32_t m : masks) {
        auto outs = run_a(b, [&] { hipLaunchKernelGGL(k_asm, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, m); }, 3);
        // Rule under test (AMD's guide: the instruction runs with EXEC forced to all ones): every lane's operands
        // are read and every lane's result is written, whatever EXEC is. The break tests the opposite rule
        // (switched-off lanes keep their old result registers, as for an ordinary VALU), which must fail.
        const int wrong_all = wrong_in(outs[0], ref, 0xFFFFFFFFu);
        int written_off = 0;
        for (int l = 0; l < 32; ++l) {
            for (int e = 0; ((m >> l) & 1) == 0 && e < 8; ++e) {
                written_off += outs[0][l * 8 + e] != SENT;
            }
        }
        const int off = 8 * (32 - __builtin_popcount(m));
        const bool det = outs[0] == outs[1] && outs[1] == outs[2];
        const bool ok = mode == "break" ? (written_off == 0 && wrong_in(outs[0], ref, m) == 0)
                                        : (wrong_all == 0 && written_off == off);
        v.put(mode == "break" ? "mask_asm_rule_break" : "mask_asm_rule", ok && det,
              fmt("EXEC=0x%08x: %d of 256 results differ from the full-wave model; switched-off lanes written %d "
                  "of %d; deterministic %s%s",
                  m, wrong_all, written_off, off, det ? "yes" : "no",
                  mode == "break" ? " (break: expects off lanes untouched)" : ""));
    }
    if (mode == "all") {
        for (uint32_t m : masks) {
            double best = 1e30;
            for (int r = 0; r < 3; ++r) {
                hipLaunchKernelGGL(k_asm_time, dim3(1), dim3(32), 0, 0, b.A, b.B, b.C, b.D, m, b.cyc);
                CHECK(hipDeviceSynchronize());
                uint64_t c;
                CHECK(hipMemcpy(&c, b.cyc, 8, hipMemcpyDeviceToHost));
                best = std::min(best, (double) c);
            }
            v.info("mask_asm_time", fmt("EXEC=0x%08x: %.2f cycles per chained matrix instruction (64 in a row)", m,
                                        best / 64.0));
        }
    }
    return v.exit_code();
}
