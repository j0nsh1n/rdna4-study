#!/usr/bin/env python3
"""Writes src/round9/ld_mat_blocks.h: the instruction streams of ld_mat.hip (round 9, item A), one per cell shape.

Usage: python3 -I gen_ld_mat.py <output header>        (standard library only)

One stream is one inline-assembly statement of ld_mat.hip, kernel k_<name>, name d<D>_r<R> or ref_r<R>:
  - R: the register count the kernel is padded to (51, 90, 167, 209; the study's rule gives 16, 16, 9, 7 waves per SIMD:
    1536 / (R rounded up to a multiple of 24)), TPB: the threads of one work-group, 64 x waves per SIMD, so that one group
    of 2 x W waves fills the two SIMDs of one CU and the register file of both (R > 96 cannot use 1024 threads: 104 registers
    with 1024-thread groups were refused in round 7). The launch is 64 such groups, one per CU.
  - D: the lookahead. Position i = 0..NMM-1 of the loop body is
        v_add_nc_u32 + v_and_b32                                 (the load's address: (voff + 256 g) masked into the span)
        global_load_b64 for the matrix instruction i + D        (issued D matrix instructions ahead of its use)
        s_wait_loadcnt D                                         (the load of position i has arrived; D newer ones may fly)
        v_wmma_f32_16x16x16_fp8_fp8 for position i               (A = the two registers that load wrote)
    so with D = 0 the load is issued and waited at once, and with D = 8 eight matrix instructions (64 cycles at 8 cycles
    each) sit between a load and its use. The loads of the next iteration's first D positions are issued at the end of
    this one (the lookahead does not drop at the loop edge); D loads before the loop start the pipeline.
  - NMM, the body length: the smallest multiple of K that is at least 114 (PR 57's pass 9 has 114 matrix instructions).
    The ring slot of load g is g mod K, so the slot of a load for the next iteration equals the slot of the same position
    in that iteration only if K divides NMM; with NMM = 114 and K = D + 2 it did not for 5 of 6 lookaheads (the first
    version of this probe: the load for position 114 overwrote the pair that position 62 still had to read). The ref
    has the same address arithmetic (written to a register nothing reads), and NMM = 114.
  - ref: the same matrix instructions and address arithmetic with no loads and no waits (the matrix-only reference).
  - "all upfront": the largest D that the registers of the cell allow (at most 60: the load counter holds 63), named by
    its value: d10_r51, d30_r90, d60_r167, d60_r209.
Registers: v16..v23 accumulator (C = D, a chain through C costs 8.0 cycles per instruction, measured in round 7), v24:25 the B
fragment, v26.. a ring of K = D + 2 pairs for A (one 64-bit load per lane, an FP8 A fragment is 2 registers), and v(R-1)
clobbered so the kernel is padded to R registers. The compiler's own values live in v0..v15 (the address temporary
%[atmp] is one of them). Every load reads at (voff + 256 g) & mask, with voff a per-lane register that moves NMM x 256 bytes
per iteration, and mask the buffer span minus 1 (a power of two): the whole footprint of the cell stays inside the level's
span (before this, only voff was masked and the immediate offsets, up to 43 KiB, left the 8 KiB span of the L0 cell).
"""
import sys

NMM = 114          # matrix instructions per iteration (PR 57's pass 9)
STEP = 256         # bytes per load (32 lanes x 8 bytes)
CELLS = [  # R, threads per group, D values (None: all upfront = the largest that fits)
    (51, 1024, [0, 1, 2, 4, 8, None]),
    (90, 1024, [0, 1, 2, 4, 8, None]),
    (167, 576, [0, 1, 2, 4, 8, None]),
    (209, 448, [0, 1, 2, 4, 8, None]),
]
RING0 = 26         # first register of the A ring
DMAX = 60


def d_all(R):
    k = (R - RING0) // 2
    return min(DMAX, k - 2)


def body_len(K):
    return -(-NMM // K) * K


def stream(D, R, ref):
    K = (d_all(R) if D is None else D) + 2
    nb = NMM if ref else body_len(K)
    advance = nb * STEP
    L = []
    for r in range(16, 24):
        L.append(f"v_mov_b32 v{r}, 0")
    L.append("v_mov_b32 v24, 0x24242424")
    L.append("v_mov_b32 v25, 0x24242424")
    for k in range(K):
        L.append(f"v_mov_b32 v{RING0 + 2 * k}, 0x24242424")
        L.append(f"v_mov_b32 v{RING0 + 2 * k + 1}, 0x24242424")
    L.append("s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)")
    L.append("s_mov_b32 %[cnt], %[n]")
    d = 0 if ref else (d_all(R) if D is None else D)

    def addr(g):
        return [f"v_add_nc_u32 %[atmp], {g * STEP}, %[voff]", "v_and_b32 %[atmp], %[mask], %[atmp]"]

    def load(g):
        reg = RING0 + 2 * (g % K)
        return addr(g) + [f"global_load_b64 v[{reg}:{reg + 1}], %[atmp], %[base]"]

    if not ref:
        for g in range(d):
            L.extend(load(g))
    L.append("2:")
    for i in range(nb):
        if not ref:
            L.extend(load(i + d))
            L.append(f"s_wait_loadcnt {d}")
        else:
            L.extend(addr(i + d))
        a = RING0 + 2 * (i % K)
        L.append(f"v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[{a}:{a + 1}], v[24:25], v[16:23]")
    L.append(f"v_add_nc_u32 %[voff], {advance}, %[voff]")
    L.append("v_and_b32 %[voff], %[mask], %[voff]")
    L.append("s_add_co_i32 %[cnt], %[cnt], -1")
    L.append("s_cmp_lg_u32 %[cnt], 0")
    L.append("s_cbranch_scc1 2b")
    if not ref:
        L.append("s_wait_loadcnt 0")
    L.append("s_getreg_b32 %[t1], hwreg(HW_REG_SHADER_CYCLES_LO)")
    return K, d, nb, L


def main():
    out = sys.argv[1]
    blocks = []
    for R, tpb, ds in CELLS:
        for D in ds:
            K, d, nb, lines = stream(D, R, False)
            blocks.append((f"d{d}_r{R}", d, K, R, tpb, nb, lines))
        K, d, nb, lines = stream(0, R, True)
        blocks.append((f"ref_r{R}", -1, K, R, tpb, nb, lines))
    with open(out, "w") as f:
        f.write("// Generated by scripts/round9/gen_ld_mat.py: do not edit. Instruction streams of ld_mat.hip (round 9, item A).\n")
        f.write("// X(name, D, K, R, TPB, NMM, text): D lookahead (-1: matrix only), K ring pairs, R registers, TPB threads per group, NMM matrix instructions per iteration.\n")
        for name, d, K, R, tpb, nb, lines in blocks:
            clob = ", ".join(f'"v{i}"' for i in range(16, R)) + ', "scc", "memory"'
            f.write(f"#define LDM_CLOB_{name} {clob}\n")
        f.write("#define LD_MAT_BLOCKS(X) \\\n")
        for name, d, K, R, tpb, nb, lines in blocks:
            text = " ".join(f'"{ln}\\n"' for ln in lines)
            f.write(f"    X({name}, {d}, {K}, {R}, {tpb}, {nb}, {text}) \\\n")
        f.write("\n")


if __name__ == "__main__":
    main()
