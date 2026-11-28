#!/usr/bin/env python3
"""Writes src/round10/ld_mat2.hip: round 10, item 1. Loads beside FP8 matrix work, one wave and the whole card.

Usage: python3 -I gen_ld_mat2.py [output .hip]      (standard library only; default src/round10/ld_mat2.hip)

The .hip file is generated whole (probe code, analysis code and the instruction streams); do not edit it by hand.

Stream layout (ld_mat_k's, round 9; it read 8.064 for matrix only), one asm statement per block, kernel k_<name>:
  - a loop body of NMM matrix instructions v_wmma_f32_16x16x16_fp8_fp8, chained through the accumulator (8.0 cycles each).
    NMM = 256 (divisible by K = 1, 4, 8, 32), except K = 32, which uses NMM = 512 so that the ring can be deep enough
    (D up to 14 groups of 32 instructions = 3584 cycles ahead). NMM was 128 first: the loop edge (3 dependent VALU, the
    counter and the branch) costs 14 cycles on gfx1201, so the matrix-only reference read (128 x 8 + 14) / 128 = 8.109,
    0.009 above the one-wave band 7.9-8.1 (Slot 1 part 2, job one-L0L2, exit 3). At 256 the edge adds 14 / 256 = 0.055: 8.055.
    The band is unchanged; the edge instructions are unchanged; only the iteration is twice as long.
  - one load group (global_load_b64, 256 bytes per wave) per K matrix instructions. At the first instruction of group j:
    the load for group j + D, then s_wait_loadcnt D (group j has landed). NO address arithmetic between matrix
    instructions: the load address is a base register plus an immediate offset: field.
  - two address registers (this iteration's window base A, next iteration's base B). Groups 0 .. GN-1 read from A, the
    lookahead groups GN .. GN+D-1 (next iteration's first D groups) read from B with offset (g - GN) * 256. Once per
    iteration, at the loop edge: A <- B; B <- (B + window) & mask. So every load stays inside the level's span
    (round 9's ld_mat_k read up to D x 256 bytes past the span: the footprint check now fails on that).
  - window = GN x 256 bytes (GN = NMM / K groups), a power of two.
  - fixed blocks (suffix _f), made for K whose window is at least the L0 span (8 KiB): the base stays put and the
    offsets are (g x 256) mod 8 KiB, so the footprint is exactly 8 KiB. Used when window >= span (the host picks it).
  - ring: the smallest divisor of GN that is at least D + 2 (so slot g mod ring is the same on every iteration).
    Registers v16..v23 accumulator, v24:25 B fragment, v26 .. the ring, v(R-1) clobbered so the kernel is padded to R.
  - counters: SHADER_CYCLES read as HI, LO, HI at the start and the end of the timed region (64 bit, cross-checked).
  - ref: the same instructions and the same loop-edge update with no loads and no waits (matrix only).

R = 209 (448 threads per group: 7 waves per SIMD on the whole card, one wave in the one-wave runs) and R = 51
(1024 threads per group: 16 waves per SIMD). Largest D: 60 at R 209 (the load counter holds 63), 6 at R 51 (ring of 8
pairs fits v26..v41).
"""
import os
import sys

STEP = 256
NMM_REF = 256
RING0 = 26
SPAN_L0 = 8192
TPB = {209: 448, 51: 1024}
# R -> K -> list of lookaheads
CELLS = {
    209: {1: [0, 1, 2, 4, 8, 24, 54, 60], 4: [0, 2, 6, 12, 30], 8: [0, 2, 6, 8, 12, 14], 32: [0, 1, 2, 4, 8, 14]},
    51: {1: [0, 2, 4, 6], 8: [0, 2, 4, 6], 32: [0, 2, 6]},
}


def nmm_of(k):
    return 512 if k == 32 else 256


def ring_of(gn, d):
    for r in range(d + 2, gn + 1):
        if gn % r == 0:
            return r
    raise ValueError("no ring size")


def fixed_offset(g):
    return (g * STEP) % SPAN_L0


def stream(k, d, r_regs, ref, fixed):
    nmm = nmm_of(k)
    gn = nmm // k
    win = gn * STEP
    kr = 2 if ref else ring_of(gn, d)
    assert RING0 + 2 * kr <= r_regs, "ring does not fit the registers"
    assert d <= 60
    L = []
    for r in range(16, 24):
        L.append(f"v_mov_b32 v{r}, 0")
    L.append("v_mov_b32 v24, 0x24242424")
    L.append("v_mov_b32 v25, 0x24242424")
    for s in range(kr):
        L.append(f"v_mov_b32 v{RING0 + 2 * s}, 0x24242424")
        L.append(f"v_mov_b32 v{RING0 + 2 * s + 1}, 0x24242424")
    L.append("s_getreg_b32 %[h0a], hwreg(HW_REG_SHADER_CYCLES_HI)")
    L.append("s_getreg_b32 %[l0], hwreg(HW_REG_SHADER_CYCLES_LO)")
    L.append("s_getreg_b32 %[h0b], hwreg(HW_REG_SHADER_CYCLES_HI)")
    L.append("s_mov_b32 %[cnt], %[n]")

    def load(g):
        reg = RING0 + 2 * (g % kr)
        if fixed:
            vr, off = "%[va]", fixed_offset(g)
        elif g < gn:
            vr, off = "%[va]", g * STEP
        else:
            vr, off = "%[vb]", (g - gn) * STEP
        return f"global_load_b64 v[{reg}:{reg + 1}], {vr}, %[base] offset:{off}"

    if not ref:
        for g in range(d):
            L.append(load(g))
    L.append("2:")
    for p in range(nmm):
        j, rem = divmod(p, k)
        if not ref and rem == 0:
            L.append(load(j + d))
            L.append(f"s_wait_loadcnt {d}")
        a = RING0 + 2 * (j % kr)
        L.append(f"v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[{a}:{a + 1}], v[24:25], v[16:23]")
    # the loop edge: the only address arithmetic of the stream, once per NMM matrix instructions
    L.append("v_mov_b32 %[va], %[vb]")
    L.append(f"v_add_nc_u32 %[vb], {win}, %[vb]")
    L.append("v_and_b32 %[vb], %[mask], %[vb]")
    L.append("s_add_co_i32 %[cnt], %[cnt], -1")
    L.append("s_cmp_lg_u32 %[cnt], 0")
    L.append("s_cbranch_scc1 2b")
    if not ref:
        L.append("s_wait_loadcnt 0")
    L.append("s_getreg_b32 %[h1a], hwreg(HW_REG_SHADER_CYCLES_HI)")
    L.append("s_getreg_b32 %[l1], hwreg(HW_REG_SHADER_CYCLES_LO)")
    L.append("s_getreg_b32 %[h1b], hwreg(HW_REG_SHADER_CYCLES_HI)")
    return kr, L


def blocks():
    out = []  # (name, k, d, r, nmm, fixed, nslots, lines)
    for r_regs, per_k in CELLS.items():
        out.append((f"ref_r{r_regs}", 0, -1, r_regs, NMM_REF, 0, 0, stream(1, 0, r_regs, True, False)[1]))
        for k, ds in per_k.items():
            for d in ds:
                nmm = nmm_of(k)
                out.append((f"k{k}_d{d}_r{r_regs}", k, d, r_regs, nmm, 0, 0, stream(k, d, r_regs, False, False)[1]))
                if nmm // k * STEP >= SPAN_L0:
                    slots = len({fixed_offset(g) for g in range(nmm // k + d)})
                    assert slots * STEP == SPAN_L0, "fixed block footprint is not the L0 span"
                    out.append((f"k{k}_d{d}_r{r_regs}_f", k, d, r_regs, nmm, 1, slots,
                                stream(k, d, r_regs, False, True)[1]))
    return out


TEMPLATE = r'''// Round 10, item 1: loads beside FP8 matrix work, one wave and the whole card ("ld_mat2"). GENERATED by
// scripts/round10/gen_ld_mat2.py: do not edit; change the generator and run it again.
//
// Question (round 9's whole-card attempt failed its own reference, 15 against 8): what does one 256-byte load group per
// K matrix instructions cost per instruction, at L0 and L2 (and later the memory-side cache and GDDR6, pinned memory
// clock), with D groups of lookahead, with one wave and with 7 or 16 waves per SIMD on every SIMD of the card.
//
// Usage: ld_mat2 --mode one|card --r 209|51 --levels L0,L2 --cells ref,1:0,1:max,8:6 [--bursts 3] [--ms 50]
//                [--aim LEVEL] [--pinned LABEL] [--clk-tol 0.03]
//   cells: ref (matrix only; ALWAYS run first as the gate: 7.9-8.1 one wave, 7.9-8.3 whole card, else nothing is
//   interpreted and the cells are not run) or K:D (K matrix instructions per load group, D groups of lookahead; D may be
//   `max`, the largest the registers of --r allow).
//   Levels: L0 8 KiB, L2 1 MiB, MALL 32 MiB, GDDR 1 GiB. MALL and GDDR are gated on a pinned memory clock (item 2a, not
//   yet run): they refuse to start unless --pinned LABEL names the pinned level, and the label is on every line.
//   --aim LEVEL: the deliberate break: the loads go to the span of LEVEL while the cell keeps its label.
//   Every launch (calibration launches too) starts with the hotspot at or below 58 C, else NOT STARTED, exit 7.
//   Counters: SHADER_CYCLES as HI, LO, HI at both ends of every wave's region (64 bit); the analysis flags and rejects
//   a burst with a wave clock outside 1.5-3.6 GHz, a counter that went backwards, a SIMD whose clock differs from the
//   median SIMD by more than --clk-tol, or a SIMD spread above 20 %; flagged bursts print every raw stamp (STAMPS lines).
// Output: RESULT ld_mat2 launch|burst|cell ..., CONTROL <id> PASS|FAIL ..., STAMPS ..., CLOCKS, THERMAL lines.
// Exit: 0 ok; 1 a burst was not placed; 2 usage; 3 a control failed (reference: nothing run, others: raw numbers); 7 not started (hot); 4 stopped at 95 C.
//
// With -DLDM2_CPU_ONLY the file compiles with a plain C++ compiler (no HIP) into the analysis self-test only.
#ifndef LDM2_CPU_ONLY
#include "../study/tim_common.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// Analysis: plain C++, also compiled alone by the CPU self-test.
namespace ld2 {

struct Rec {
    uint32_t pl;
    uint32_t s0[3], s1[3];  // SHADER_CYCLES HI, LO, HI at the start and at the end
    uint32_t pad;
    uint64_t rt0, rt1;  // REALTIME (100 MHz)
};
static_assert(sizeof(Rec) == 48, "record layout");

inline uint32_t simd_of(uint32_t pl) { return pl & 2047u; }

// HI, LO, HI to one 64-bit value. If HI changed between the two reads, LO was read around the wrap: a LO with its top bit
// set was read before it (HI first), a LO without it after it (HI second). bad: HI moved by more than one.
inline uint64_t stamp(const uint32_t s[3], bool & bad) {
    if (s[0] == s[2]) {
        return ((uint64_t) s[0] << 32) | s[1];
    }
    if (s[2] != s[0] + 1u) {
        bad = true;
    }
    return ((uint64_t) ((s[1] & 0x80000000u) ? s[0] : s[2]) << 32) | s[1];
}

struct Params {
    int W = 1;             // waves per SIMD
    bool card = false;
    uint32_t iters = 1;
    int nmm = 256;
    double clk_lo = 1.5, clk_hi = 3.6, clk_tol = 0.03, spread = 0.20;
};

struct Burst {
    bool placed = false, flagged = false;
    size_t simds = 0;
    double cyc_med = NAN, cyc_min = NAN, cyc_max = NAN, clk_med = NAN;
    int waves = 0, bad_clock = 0, backwards = 0, bad_hi = 0, simd_clk_off = 0;
    bool spread = false;
    std::string why;
};

inline double median(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

inline Burst analyse_burst(const std::vector<Rec> & rec, const Params & p) {
    Burst o;
    struct Sd {
        int n = 0;
        uint64_t c0min = ~0ull, c1max = 0, rt_lo = ~0ull, rt_hi = 0;
    };
    std::map<uint32_t, Sd> per;
    for (const Rec & x : rec) {
        if (!(x.rt1 > x.rt0)) {
            continue;
        }
        ++o.waves;
        bool bad = false;
        const uint64_t c0 = stamp(x.s0, bad), c1 = stamp(x.s1, bad);
        o.bad_hi += bad;
        if (c1 < c0) {
            ++o.backwards;
            std::printf("COUNTER BACKWARDS: pl 0x%x start 0x%llx end 0x%llx\n", x.pl, (unsigned long long) c0, (unsigned long long) c1);
        } else {
            const double ghz = (double) (c1 - c0) / ((double) (x.rt1 - x.rt0) / 1e8) / 1e9;
            if (!(ghz >= p.clk_lo && ghz <= p.clk_hi)) {
                ++o.bad_clock;
            }
        }
        Sd & d = per[simd_of(x.pl)];
        ++d.n;
        d.c0min = std::min(d.c0min, c0);
        d.c1max = std::max(d.c1max, c1);
        d.rt_lo = std::min(d.rt_lo, x.rt0);
        d.rt_hi = std::max(d.rt_hi, x.rt1);
    }
    o.simds = per.size();
    o.placed = (p.card ? per.size() == 128 : per.size() == 1);
    std::vector<double> cyc, clk;
    for (auto & kv : per) {
        const Sd & d = kv.second;
        o.placed = o.placed && d.n == p.W;
        if (d.c1max < d.c0min) {
            continue;  // counted as backwards per wave
        }
        const double span = (double) (d.c1max - d.c0min);
        cyc.push_back(span / ((double) d.n * p.iters * p.nmm));
        clk.push_back(span / ((double) (d.rt_hi - d.rt_lo) / 1e8) / 1e9);
    }
    if (!cyc.empty()) {
        o.cyc_med = median(cyc);
        o.cyc_min = *std::min_element(cyc.begin(), cyc.end());
        o.cyc_max = *std::max_element(cyc.begin(), cyc.end());
        o.clk_med = median(clk);
        for (double c : clk) {
            o.simd_clk_off += std::fabs(c / o.clk_med - 1.0) > p.clk_tol;
        }
        o.spread = o.cyc_max > o.cyc_med * (1 + p.spread) || o.cyc_min < o.cyc_med * (1 - p.spread);
    }
    o.flagged = o.bad_clock || o.backwards || o.bad_hi || o.simd_clk_off || o.spread || cyc.empty();
    char b[200];
    std::snprintf(b, sizeof b, "bad_clock_waves %d backwards %d bad_hi %d simd_clk_off %d spread %d", o.bad_clock, o.backwards, o.bad_hi,
                  o.simd_clk_off, (int) o.spread);
    o.why = b;
    return o;
}

struct Band {
    double lo, hi;
};
inline bool in_band(double v, Band b) { return v >= b.lo && v <= b.hi; }
inline Band ref_gate(bool card) { return card ? Band{7.9, 8.3} : Band{7.9, 8.1}; }

// Control 7: any flagged (rejected) burst fails the job; a clean job has none.
inline bool c7_ok(int flagged_total) { return flagged_total == 0; }

// Footprint of a block against a span, as the host and the listing check both compute it: a fixed block reads the
// slots (g x 256) mod 8 KiB, so its footprint is its slot count x 256 and the span must equal that; a walking block
// reads one window per iteration and the window steps by its own size through the span, so the footprint is the span
// if the window divides it (and is not larger).
inline bool footprint_ok(bool fixed, uint32_t nslots, uint32_t window, uint64_t span) {
    if (fixed) {
        return (uint64_t) nslots * 256 == span;
    }
    return window <= span && span % window == 0;
}

}  // namespace ld2

#ifdef LDM2_CPU_ONLY
// ---------------------------------------------------------------------------------------------------------------
// CPU-only self-test of the analysis (no GPU, no HIP): a clean burst passes, and every deliberate break must trip.
namespace {
std::vector<ld2::Rec> synth(int simds, int W, double ghz, uint64_t base_cycles, uint32_t iters, int nmm, uint64_t c0_start) {
    std::vector<ld2::Rec> v;
    const double per_wave_cycles = 8.0 * nmm * iters * W;  // matrix-bound: the SIMD's waves share one pipe
    const uint64_t rt_ticks = (uint64_t) (per_wave_cycles / ghz / 1e9 * 1e8);
    for (int s = 0; s < simds; ++s) {
        for (int w = 0; w < W; ++w) {
            ld2::Rec r{};
            r.pl = (uint32_t) s;
            const uint64_t a = c0_start, b = c0_start + (uint64_t) per_wave_cycles;
            r.s0[0] = r.s0[2] = (uint32_t) (a >> 32);
            r.s0[1] = (uint32_t) a;
            r.s1[0] = r.s1[2] = (uint32_t) (b >> 32);
            r.s1[1] = (uint32_t) b;
            r.rt0 = base_cycles + 1000;
            r.rt1 = r.rt0 + rt_ticks;
            v.push_back(r);
        }
    }
    return v;
}
int fails = 0;
void expect(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    fails += !ok;
}
}  // namespace

int main() {
    ld2::Params p;
    p.W = 7;
    p.card = true;
    p.iters = 1000;
    p.nmm = 256;
    const uint64_t start = 5000000000ull;
    auto clean = synth(128, 7, 2.9, 1000, p.iters, p.nmm, start);
    ld2::Burst b = ld2::analyse_burst(clean, p);
    expect("clean burst: placed, not flagged, 8.0 cycles per instruction", b.placed && !b.flagged && std::fabs(b.cyc_med - 8.0) < 0.01);
    expect("clean burst: reference gate accepts it", ld2::in_band(b.cyc_med, ld2::ref_gate(true)));
    expect("break: reference gate refuses 15.0", !ld2::in_band(15.0, ld2::ref_gate(true)) && !ld2::in_band(15.1, ld2::ref_gate(false)));

    auto shifted = clean;  // the 58 GHz case: one end stamp 2^32 too large
    shifted[100].s1[0] += 1;
    shifted[100].s1[2] += 1;
    ld2::Burst bs = ld2::analyse_burst(shifted, p);
    expect("break: one end stamp shifted by 2^32 is flagged (bad clock wave)", bs.flagged && bs.bad_clock >= 1);

    auto back = clean;  // counter went backwards
    back[7].s1[0] = back[7].s1[2] = 0;
    back[7].s1[1] = 5;
    ld2::Burst bb = ld2::analyse_burst(back, p);
    expect("break: a counter that went backwards is flagged", bb.flagged && bb.backwards >= 1);

    auto wrapped = synth(128, 7, 2.9, 1000, p.iters, p.nmm, 0xFFFFF000ull);  // the start is just below 2^32
    ld2::Burst bw = ld2::analyse_burst(wrapped, p);
    expect("a plain wrap of the low word (HI 0 to 1, consistent HI) is not flagged and reads 8.0", !bw.flagged && std::fabs(bw.cyc_med - 8.0) < 0.01);

    auto torn = clean;  // HI changed between the two HI reads: LO high, so the HI read first is the right one
    for (auto & r : torn) {
        r.s0[0] = 1;
        r.s0[2] = 2;
        r.s0[1] = 0xFFFFFF00u;
        r.s1[0] = 2;
        r.s1[2] = 2;
        r.s1[1] = (uint32_t) (0xFFFFFF00ull + (uint64_t) (8.0 * p.nmm * p.iters * p.W));
    }
    ld2::Burst bt = ld2::analyse_burst(torn, p);
    expect("torn HI/LO/HI read (LO high: the earlier HI) resolves to the right 64-bit value", !bt.flagged && bt.bad_hi == 0);
    auto torn_bad = torn;
    torn_bad[3].s0[2] = 9;  // HI moved by more than one: a broken read
    expect("break: HI moving by more than one is flagged", ld2::analyse_burst(torn_bad, p).bad_hi >= 1);

    auto slow = clean;  // one SIMD 5 % off the others in clock
    for (int w = 0; w < 7; ++w) {
        slow[50 * 7 + w].rt1 += (slow[50 * 7 + w].rt1 - slow[50 * 7 + w].rt0) / 20;
    }
    expect("break: one SIMD 5 % off in clock is flagged (3 % rule)", ld2::analyse_burst(slow, p).flagged);

    auto unplaced = clean;
    unplaced.erase(unplaced.begin() + 20);
    ld2::Burst bu = ld2::analyse_burst(unplaced, p);
    expect("break: a missing wave is not placed", !bu.placed);

    expect("footprint: a walking 32 KiB window fits 1 MiB and fails 8 KiB", ld2::footprint_ok(false, 0, 32768, 1u << 20) && !ld2::footprint_ok(false, 0, 32768, 8192));
    expect("footprint: a fixed block with 32 slots is exactly 8 KiB, with 31 it is not", ld2::footprint_ok(true, 32, 0, 8192) && !ld2::footprint_ok(true, 31, 0, 8192));
    expect("control 7: zero flagged bursts passes", ld2::c7_ok(0));
    expect("break: one flagged burst fails control 7", !ld2::c7_ok(1) && !ld2::c7_ok(5));
    std::printf("ld_mat2 analysis self-test: %s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
#else
// ---------------------------------------------------------------------------------------------------------------
// Instruction streams (generated) and kernels
@@STREAMS@@

namespace {

using ld2::Rec;

#define LDM2_KERNEL(NAME, K, D, R, TPB, NMM, FIXED, NSLOTS)                                                               \
    __global__ __launch_bounds__(TPB) void k_##NAME(const char * base, uint32_t n_iter, uint32_t log2span, uint32_t win,  \
                                                    int nact, int only0, Rec * out) {                                     \
        __shared__ int cnt[2];                                                                                            \
        if (threadIdx.x < 2) {                                                                                            \
            cnt[threadIdx.x] = 0;                                                                                         \
        }                                                                                                                 \
        __syncthreads();                                                                                                  \
        const uint32_t pl = mem::placement();                                                                             \
        const int simd = (pl >> 10) & 1;                                                                                  \
        int rank = 0;                                                                                                     \
        if ((threadIdx.x % 32) == 0) {                                                                                    \
            rank = atomicAdd(&cnt[simd], 1);                                                                              \
        }                                                                                                                 \
        rank = __builtin_amdgcn_readfirstlane(rank);                                                                      \
        __syncthreads();                                                                                                  \
        if ((only0 && simd != 0) || rank >= nact) {                                                                       \
            return;                                                                                                       \
        }                                                                                                                 \
        const uint32_t wave = blockIdx.x * (TPB / 32) + threadIdx.x / 32;                                                 \
        const uint32_t mask = (1u << log2span) - 1u;                                                                      \
        /* the window of this wave: aligned to its own size, so the windows tile the span; a window that is not smaller */ \
        /* than the span (fixed blocks) gives phase 0 */                                                                  \
        const uint32_t phase = (((wave * 0x9E3779B9u) >> (32 - log2span)) & ~(win - 1u)) & mask;                          \
        uint32_t va = phase + (threadIdx.x % 32) * 8u;                                                                    \
        uint32_t vb = (va + win) & mask;                                                                                  \
        uint32_t h0a, l0, h0b, h1a, l1, h1b, c;                                                                           \
        const uint64_t rt0 = mem::realtime();                                                                             \
        asm volatile(LDM2_TXT_##NAME : [h0a] "=&s"(h0a), [l0] "=&s"(l0), [h0b] "=&s"(h0b), [h1a] "=&s"(h1a),             \
                                       [l1] "=&s"(l1), [h1b] "=&s"(h1b), [cnt] "=&s"(c), [va] "+v"(va), [vb] "+v"(vb) \
                     : [n] "s"(n_iter), [base] "s"(base), [mask] "s"(mask)                                                \
                     : LDM2_CLOB_##R);                                                                                    \
        const uint64_t rt1 = mem::realtime();                                                                             \
        if ((threadIdx.x % 32) == 0) {                                                                                    \
            out[wave] = Rec{pl, {h0a, l0, h0b}, {h1a, l1, h1b}, 0, rt0, rt1};                                             \
        }                                                                                                                 \
    }
LDM2_TABLE(LDM2_KERNEL)

using Kern = void (*)(const char *, uint32_t, uint32_t, uint32_t, int, int, Rec *);
struct Cell {
    const char * name;
    int K, D, R, TPB, NMM, fixed;
    uint32_t nslots;
    Kern k;
};
#define LDM2_ROW(NAME, K, D, R, TPB, NMM, FIXED, NSLOTS) {#NAME, K, D, R, TPB, NMM, FIXED, NSLOTS, k_##NAME},
const Cell cells[] = {LDM2_TABLE(LDM2_ROW)};

struct Level {
    const char * name;
    uint32_t log2span;
    bool gated;
};
const Level levels[] = {{"L0", 13, false}, {"L2", 20, false}, {"MALL", 25, true}, {"GDDR", 30, true}};

const Level * find_level(const std::string & n) {
    for (const Level & l : levels) {
        if (n == l.name) {
            return &l;
        }
    }
    return nullptr;
}

char * g_buf = nullptr;
Rec * g_rec = nullptr;
constexpr int GROUPS = 64;
constexpr int MAX_RECS = GROUPS * 32;
std::string g_mclk_label = "unpinned";
double g_clk_tol = 0.03;
int g_launches = 0, g_hot_bad = 0, g_flagged_total = 0;

const Cell * find_ref(int R) {
    for (const Cell & c : cells) {
        if (c.R == R && c.D < 0) {
            return &c;
        }
    }
    return nullptr;
}
int max_d(int K, int R) {
    int m = -1;
    for (const Cell & c : cells) {
        if (c.R == R && c.K == K && !c.fixed) {
            m = std::max(m, c.D);
        }
    }
    return m;
}
// the block for (K, D, R); the fixed variant when the window is at least the (aimed) span
const Cell * find_cell(int K, int D, int R, uint64_t span) {
    const uint64_t window = (uint64_t) (NMM_OF(K) / K) * 256;
    const bool want_fixed = window >= span;
    for (const Cell & c : cells) {
        if (c.R == R && c.K == K && c.D == D && (bool) c.fixed == want_fixed) {
            return &c;
        }
    }
    return nullptr;
}

std::atomic<bool> g_run{false};
struct Guard {
    std::thread th;
    double tmax = 0;
    void start() {
        g_run = true;
        tmax = 0;
        th = std::thread([this] {
            while (g_run) {
                const double t = mem::hotspot_c();
                if (t >= 0) {
                    tmax = std::max(tmax, t);
                    if (t >= 95.0) {
                        std::printf("THERMAL hotspot %.0f C is at or above 95 C: stopping at once\n", t);
                        std::fflush(stdout);
                        std::_Exit(4);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    void stop() {
        g_run = false;
        th.join();
    }
};

constexpr double HOT_LIMIT = 58.0;
// idle until the hotspot is at or below the limit; false if it cannot be read or does not get there within 120 s
bool cool_below(double & t) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        t = mem::hotspot_c();
        if (t < 0) {
            return false;  // unreadable: never a pass
        }
        if (t <= HOT_LIMIT) {
            return true;
        }
        if (t >= 95.0 || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(120)) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}

struct Run {
    bool card;
    const Cell * c;
    uint32_t log2span;
    uint32_t win;
    int R;
};

double launch_ms(const Run & r, uint32_t n_iter) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    const int groups = r.card ? GROUPS : 1;
    const int nact = r.card ? r.c->TPB / 64 : 1;
    CHECK(hipMemset(g_rec, 0, sizeof(Rec) * MAX_RECS));
    CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(r.c->k, dim3(groups), dim3(r.c->TPB), 0, 0, (const char *) g_buf, n_iter, r.log2span, r.win, nact,
                       r.card ? 0 : 1, g_rec);
    CHECK(hipGetLastError());
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    CHECK(hipEventDestroy(e0));
    CHECK(hipEventDestroy(e1));
    return ms;
}

// One launch with the 58 C start rule and the 95 C guard. kind: "calib" or "timed". Returns false: NOT STARTED.
bool guarded_launch(const Run & r, uint32_t n_iter, const char * kind, int rep, const char * cellname, double & ms, double & hot0, double & hmax) {
    if (!cool_below(hot0)) {
        std::printf("NOT STARTED: hotspot %.0f C (limit %.0f C, -1 is unreadable) after 120 s of cool-down, %s launch of %s\n", hot0, HOT_LIMIT, kind, cellname);
        std::fflush(stdout);
        return false;
    }
    Guard g;
    g.start();
    ms = launch_ms(r, n_iter);
    g.stop();
    hmax = g.tmax;
    ++g_launches;
    g_hot_bad += !(hot0 <= HOT_LIMIT);
    std::printf("RESULT ld_mat2 launch kind %s cell %s rep %d iters %u ms %.3f hot_start %.0f hot_max %.0f\n", kind, cellname, rep, n_iter, ms,
                hot0, hmax);
    std::fflush(stdout);
    return true;
}

bool calibrate(const Run & r, double target_ms, uint32_t & iters, const char * cellname) {
    const uint32_t n1 = r.card ? 16 : 256, n2 = r.card ? 64 : 2048;
    double ms, h0, hm;
    if (!guarded_launch(r, n1, "calib", 0, cellname, ms, h0, hm)) return false;  // code load, first touches
    double t1, t2;
    if (!guarded_launch(r, n1, "calib", 1, cellname, t1, h0, hm)) return false;
    if (!guarded_launch(r, n2, "calib", 2, cellname, t2, h0, hm)) return false;
    double per = (t2 - t1) / (n2 - n1);
    if (!(per > 0)) {
        per = t2 / n2;
    }
    iters = (uint32_t) std::min(std::max(target_ms / per, 32.0), 5e7);
    return true;
}

void dump_stamps(const std::vector<Rec> & rec, const char * cell, int rep, const char * why) {
    for (size_t i = 0; i < rec.size(); ++i) {
        const Rec & x = rec[i];
        if (!(x.rt1 > x.rt0)) {
            continue;
        }
        std::printf("STAMPS cell %s rep %d why [%s] wave %zu pl 0x%x s0 %08x %08x %08x s1 %08x %08x %08x rt0 %llu rt1 %llu\n", cell, rep, why, i, x.pl,
                    x.s0[0], x.s0[1], x.s0[2], x.s1[0], x.s1[1], x.s1[2], (unsigned long long) x.rt0, (unsigned long long) x.rt1);
    }
    std::fflush(stdout);
}

struct CellResult {
    double med = NAN;
    size_t used = 0;
    bool agree = false;
    int unplaced = 0;
};

// one cell: calibrate, then `bursts` timed launches, analysed. fixed label/aimed names for the lines.
bool run_cell(const Run & run, const std::string & level, const std::string & aimed, const std::string & label, int bursts, double ms_target, CellResult & res,
              int & refused) {
    uint32_t iters = 0;
    if (!calibrate(run, ms_target, iters, run.c->name)) return false;
    const int W = run.card ? run.c->TPB / 64 : 1;
    ld2::Params p;
    p.W = W;
    p.card = run.card;
    p.iters = iters;
    p.nmm = run.c->NMM;
    p.clk_tol = g_clk_tol;
    std::vector<double> used;
    int flagged = 0;
    for (int b = 0; b < bursts; ++b) {
        double ms, h0, hm;
        if (!guarded_launch(run, iters, "timed", b, run.c->name, ms, h0, hm)) return false;
        std::vector<Rec> rec(MAX_RECS);
        CHECK(hipMemcpy(rec.data(), g_rec, sizeof(Rec) * MAX_RECS, hipMemcpyDeviceToHost));
        const ld2::Burst o = ld2::analyse_burst(rec, p);
        refused += !o.placed;
        res.unplaced += !o.placed;
        const bool rejected = o.flagged || !o.placed;
        if (rejected) {
            ++flagged;
            ++g_flagged_total;
            std::printf("FLAGGED burst cell %s rep %d: %s placed %d simds %zu waves %d, rejected from the medians; raw stamps follow\n", run.c->name, b,
                        o.why.c_str(), (int) o.placed, o.simds, o.waves);
            dump_stamps(rec, run.c->name, b, o.why.c_str());
        } else {
            used.push_back(o.cyc_med);
        }
        std::printf("RESULT ld_mat2 burst mode %s R %d W %d level %s aimed %s cell %s K %d D %d rep %d iters %u ms %.2f cyc_per_mm_med %.3f min %.3f "
                    "max %.3f clock_ghz %.3f placed %d simds %zu flagged %d hot_start %.0f hot_max %.0f mclk %s\n",
                    run.card ? "card" : "one", run.R, W, label.c_str(), aimed.c_str(), run.c->name, run.c->K, run.c->D, b, iters, ms, o.cyc_med,
                    o.cyc_min, o.cyc_max, o.clk_med, (int) o.placed, o.simds, (int) rejected, h0, hm, g_mclk_label.c_str());
        std::fflush(stdout);
    }
    res.med = ld2::median(used);
    res.used = used.size();
    res.agree = false;
    if (used.size() >= 2) {
        const double lo = *std::min_element(used.begin(), used.end()), hi = *std::max_element(used.begin(), used.end());
        res.agree = (hi - lo) <= 0.05 * res.med;
    }
    std::printf("RESULT ld_mat2 cell mode %s R %d level %s aimed %s cell %s K %d D %d cyc_per_mm_median_of_bursts %.3f bursts_used %zu of %d flagged %d "
                "agree5pct %d mclk %s\n",
                run.card ? "card" : "one", run.R, label.c_str(), aimed.c_str(), run.c->name, run.c->K, run.c->D, res.med, res.used, bursts, flagged,
                (int) res.agree, g_mclk_label.c_str());
    std::printf("CONTROL c9_placed_%s_%s %s %d of %d bursts not placed (every SIMD with its waves); %d flagged and rejected by the counter checks\n", run.c->name,
                label.c_str(), res.unplaced == 0 ? "PASS" : "FAIL", res.unplaced, bursts, flagged);
    std::printf("CONTROL c10_agree_%s_%s %s %zu bursts used\n", run.c->name, label.c_str(), res.agree ? "PASS" : "FAIL", res.used);
    std::fflush(stdout);
    return true;
}

std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(',', a);
        v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    std::string mode, lv = "L0,L2", cs = "ref", aim;
    int R = 0, bursts = 3;
    double ms_target = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--mode") mode = argv[i + 1];
        else if (k == "--r") R = std::atoi(argv[i + 1]);
        else if (k == "--levels") lv = argv[i + 1];
        else if (k == "--cells") cs = argv[i + 1];
        else if (k == "--bursts") bursts = std::atoi(argv[i + 1]);
        else if (k == "--ms") ms_target = std::atof(argv[i + 1]);
        else if (k == "--aim") aim = argv[i + 1];
        else if (k == "--pinned") g_mclk_label = argv[i + 1];
        else if (k == "--clk-tol") g_clk_tol = std::atof(argv[i + 1]);
        else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return 2;
        }
    }
    if ((mode != "one" && mode != "card") || !find_ref(R) || ms_target > 200 || bursts < 1 || bursts > 10 || g_clk_tol <= 0 || g_clk_tol > 0.2) {
        std::fprintf(stderr, "usage: ld_mat2 --mode one|card --r 209|51 --levels L0,L2 --cells ref,K:D,.. [--bursts 1-10] [--ms <=200] [--aim LEVEL] [--pinned LABEL]\n");
        return 2;
    }
    const bool card = mode == "card";
    for (const std::string & ln : split(lv)) {
        const Level * l = find_level(ln);
        if (!l) {
            std::fprintf(stderr, "unknown level %s\n", ln.c_str());
            return 2;
        }
        if (l->gated && g_mclk_label == "unpinned") {
            std::fprintf(stderr, "level %s is gated on a pinned memory clock (item 2a): pass --pinned <label>\n", l->name);
            return 2;
        }
    }
    const size_t alloc = (1ull << 30) + (64u << 10);
    CHECK(hipMalloc(&g_buf, alloc));
    CHECK(hipMemset(g_buf, 0x24, alloc));
    CHECK(hipMalloc(&g_rec, sizeof(Rec) * MAX_RECS));
    mem::clocks("start");

    int refused = 0, control_fail = 0;
    // 1. the reference, always first: nothing is interpreted when it does not read 8.0
    {
        const Cell * ref = find_ref(R);
        const Run run{card, ref, 20, 32768, R};
        CellResult cr;
        if (!run_cell(run, "-", "-", "-", std::max(bursts, 3), ms_target, cr, refused)) return 7;
        const ld2::Band g = ld2::ref_gate(card);
        const bool ok = cr.used >= 1 && ld2::in_band(cr.med, g);
        std::printf("CONTROL ref_%s_R%d %s matrix-only %.3f against %.1f-%.1f (%zu bursts used)\n", mode.c_str(), R, ok ? "PASS" : "FAIL", cr.med, g.lo, g.hi,
                    cr.used);
        if (!ok) {
            std::printf("REFERENCE FAILED: nothing in this job is interpreted; the cells were not run. Read the listing first.\n");
            std::fflush(stdout);
            mem::clocks("end");
            return 3;
        }
    }
    // 2. the cells
    std::map<std::string, double> got;  // "<level>/<aimed>/<K>:<D>" -> median
    for (const std::string & ln : split(lv)) {
        const Level * lvl = find_level(ln);
        const Level * aimed = aim.empty() ? lvl : find_level(aim);
        if (!aimed) {
            std::fprintf(stderr, "unknown --aim level %s\n", aim.c_str());
            return 2;
        }
        for (const std::string & spec : split(cs)) {
            if (spec == "ref") {
                continue;
            }
            int K = 0;
            char dtxt[16] = {0};
            if (std::sscanf(spec.c_str(), "%d:%15s", &K, dtxt) != 2) {
                std::fprintf(stderr, "bad cell %s (want K:D or K:max)\n", spec.c_str());
                return 2;
            }
            const int D = std::strcmp(dtxt, "max") == 0 ? max_d(K, R) : std::atoi(dtxt);
            const Cell * c = find_cell(K, D, R, 1ull << aimed->log2span);
            if (!c) {
                std::fprintf(stderr, "no kernel for K %d D %d R %d at span 2^%u\n", K, D, R, aimed->log2span);
                return 2;
            }
            const uint64_t span = 1ull << aimed->log2span;
            const uint32_t window = (uint32_t) (c->NMM / c->K) * 256;
            if (!ld2::footprint_ok(c->fixed, c->nslots, window, span)) {
                std::fprintf(stderr, "footprint of %s is not the span of %s: refused\n", c->name, aimed->name);
                return 2;
            }
            const Run run{card, c, aimed->log2span, window, R};
            CellResult cr;
            if (!run_cell(run, lvl->name, aimed->name, lvl->name, bursts, ms_target, cr, refused)) return 7;
            char key[64];
            std::snprintf(key, sizeof key, "%s/%d:%d", lvl->name, K, D);
            got[key] = cr.med;
            // anchors of round 9's ld_mat_k (one wave, L2, matching label and aim): raw numbers if they fail
            if (!card && lvl == aimed && std::strcmp(lvl->name, "L2") == 0) {
                struct Anchor { int K, D; double v, tol; const char * id; };
                static const Anchor an[] = {{1, 54, 33.65, 1.0, "c3"}, {4, 12, 14.4, 0.5, "c4_k4"}, {8, 12, 11.4, 0.4, "c4_k8"}};
                for (const Anchor & a : an) {
                    if (a.K == K && a.D == D) {
                        const bool ok = std::fabs(cr.med - a.v) <= a.tol;
                        control_fail += !ok;
                        std::printf("CONTROL %s_anchor_L2 %s K %d D %d %.3f against %.2f +- %.2f (ld_mat_k, measurements/round9.md follow-up A)\n", a.id,
                                    ok ? "PASS" : "FAIL", K, D, cr.med, a.v, a.tol);
                    }
                }
            }
            if (card && lvl == aimed && std::strcmp(lvl->name, "GDDR") == 0 && K == 1) {
                const bool ok = cr.med >= 130 && cr.med <= 155;
                control_fail += !ok;
                std::printf("CONTROL c11_bandwidth_bound_GDDR %s K 1 D %d %.3f against 130-155 (mclk %s)\n", ok ? "PASS" : "FAIL", D, cr.med,
                            g_mclk_label.c_str());
            }
        }
    }
    const bool c7 = ld2::c7_ok(g_flagged_total);
    control_fail += !c7;
    std::printf("CONTROL c7_counters %s %d bursts flagged and rejected in this job (raw stamps above)\n", c7 ? "PASS" : "FAIL", g_flagged_total);
    std::printf("CONTROL c8_hot %s %d of %d launches (calibration included) started at or below %.0f C\n", g_hot_bad == 0 ? "PASS" : "FAIL",
                g_launches - g_hot_bad, g_launches, HOT_LIMIT);
    std::fflush(stdout);
    mem::clocks("end");
    mem::report_hottest();
    if (g_hot_bad) return 3;
    if (control_fail) return 3;
    return refused ? 1 : 0;
}
#endif
'''


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "src", "round10", "ld_mat2.hip")
    bl = blocks()
    rs = sorted({b[3] for b in bl})
    s = []
    s.append("// Generated streams. LDM2_TABLE(X): X(name, K, D, R, TPB, NMM, FIXED, NSLOTS); LDM2_TXT_<name> the asm text.")
    for r in rs:
        clob = ", ".join(f'"v{i}"' for i in range(16, r)) + ', "scc", "memory"'
        s.append(f"#define LDM2_CLOB_{r} {clob}")
    for name, k, d, r, nmm, fixed, nslots, lines in bl:
        text = " ".join(f'"{ln}\\n"' for ln in lines)
        s.append(f"#define LDM2_TXT_{name} {text}")
    s.append("#define LDM2_TABLE(X) \\")
    for name, k, d, r, nmm, fixed, nslots, lines in bl:
        s.append(f"    X({name}, {k}, {d}, {r}, {TPB[r]}, {nmm}, {fixed}, {nslots}) \\")
    s.append("")
    s.append("constexpr int NMM_OF(int k) { return k == 32 ? 512 : %d; }" % nmm_of(1))
    text = TEMPLATE.replace("@@STREAMS@@", "\n".join(s))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {out}: {len(bl)} blocks")


if __name__ == "__main__":
    main()
