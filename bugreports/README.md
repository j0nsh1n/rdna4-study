# Compiler bug-report bundles (AMDGPU / HIP)

Three standalone bug-report packages for LLVM `backend:AMDGPU` / ROCm, taken from the RX 9070 XT (gfx1201) study.
**These are local drafts. Nothing here has been filed upstream.** Each bundle has its own README written as an issue
text, a minimal reproducer in plain C++ (no study headers), an LLVM IR reproducer with `RUN:` / `CHECK:` lines, and a
CPU-only `verify_cpu.sh` with deliberate breaks. No GPU is needed or opened by anything in this directory.

| Bundle | Problem | Symptom on hardware |
|---|---|---|
| [01-wmma-call-boundary-hazard](01-wmma-call-boundary-hazard/README.md) | No `v_nop` between a WMMA that ends a non-inlined callee and the caller's next WMMA that reads its result as A (`v_wmma ; s_setpc_b64` / `s_swappc_b64 ; v_wmma`). gfx12 and gfx11. | gfx1201: 64 of 256 results wrong, bit-identical to the no-`v_nop` assembly (0 of 256 with `v_nop`). gfx1100: 44 of 256 wrong. |
| [02-wmma-perlane-if-exec](02-wmma-perlane-if-exec/README.md) | WMMA/SWMMAC builtins under a per-lane `if`, early `return` or per-lane loop compile silently inside the `s_and_saveexec` / `v_cmpx` region, although the instruction ignores EXEC. | gfx1201: up to 100 % of stored results wrong (n = 8: 64 of 64; n = 16: 128 of 128; n = 20: 144 of 160; n = 31: 23 of 248), unstable between runs; per-lane loop: 128 of 128 inactive-lane results overwritten. |
| [03-divergent-lane-permute-source](03-divergent-lane-permute-source/README.md) | `permlane16` / `permlanex16` with fetch-inactive inside a divergent branch: the IR sink passes move the source computation into the branch, so switched-off partner lanes never define it. | gfx1201: 100 % of the lanes that have a switched-off partner get a stale value (3 of 5 mask shapes per permute); 0 % wrong when the source is pinned before the branch. |

## Summary across compilers, targets and hardware

CPU results below come from `verify_all_cpu.sh` (listing scans plus FileCheck on the IR reproducers). A cell says the
bug appears in the generated code. Hardware counts were measured with the study's equivalent kernels, which have the
same source shape and listing as the bundle reproducers; the bundle files themselves were not run on a GPU.

| Compiler | Opt | Targets compiled | 01 call boundary | 02 WMMA under partial EXEC | 03 permute source sunk |
|---|---|---|---|---|---|
| ROCm 10.1.0 (AMD clang 24.0.0git d6f6cb69) | -O3 | gfx1201, gfx1200 (01 also gfx1151, gfx1150) | yes | yes | yes |
| ROCm 10.2.0 (AMD clang 24.0.0git bdc35841) | -O3 | same | yes | yes | yes |
| ROCm 10.2.0 | -O1 | same | yes | yes | yes |
| Fedora clang 22.1.8-4.fc44 (`-nogpulib`) | -O3 | same | yes | yes | yes |

| Hardware | 01 call boundary | 02 WMMA under partial EXEC | 03 permute source sunk |
|---|---|---|---|
| RX 9070 XT (gfx1201), ROCm 10.1 | 64 of 256 wrong (-O1, -O2, -O3, -Os) | wrong, counts above (-O3) | 100 % of affected lanes wrong (-O3); pinned right (-O1 to -O3) |
| RX 9070 XT (gfx1201), ROCm 10.2 | 64 of 256 wrong (-O3, -O1) | same counts as ROCm 10.1 (-O3, -O1) | not run |
| RX 7900 XT (gfx1100), ROCm 10.1 / 10.2 / system clang 20 | 44 of 256 wrong | same shapes fail, other counts (n = 16 happens to be right) | not run |
| gfx1200, gfx1150, gfx1151 | listing only | listing only | listing only |

The RX 9070 XT ROCm 10.1 hardware runs used LACT profile "Testing" (stock core clock and voltage, 1258 MHz memory,
317 W cap with a 1.2x regulator mod). Only the 2026-11-26 ROCm 10.2 follow-up for bundles 01 and 02 ran on the
overclocked profile (+150 MHz core, -50 mV, 1409 MHz memory request, 374 W cap with the 1.2x mod); bundle 03 ran on
"Testing" only. Correctness counts do not depend on the clock; wall-clock numbers are not part of these reports.

## Run the checks

```
bash bugreports/verify_all_cpu.sh [--out DIR]
```

- `--out DIR`: where compiler output, listings and per-bundle `matrix.md` files go. The default is a new `mktemp`
  directory; nothing is written inside `bugreports/`.
- Each bundle's `verify_cpu.sh` can also be run alone: `bash bugreports/0N-.../verify_cpu.sh [--out DIR]`.
- Exit status 0 means: every bundle's compile matrix shows the bug, every deliberate break turns its detector RED (or is
  flagged) and the untouched input is GREEN again, and the hygiene scan is clean.
- `--skip-bundles` runs only the hygiene scan and its self-test.
- Needs `bash`, `python3` and the compilers. Defaults: ROCm 10.1 and 10.2 under `$HOME/.local/opt/rocm-10.1.0` and
  `rocm-10.2.0`, `/usr/bin/clang++`, `/usr/bin/clang`. Override with `ROCM_101`, `ROCM_102`, `FEDORA_CXX`,
  `FEDORA_CLANG`. A missing compiler is skipped and reported; at least one must exist. `FileCheck` and `opt` are
  taken from the ROCm trees (or `PATH`); without them the IR tests fall back to the scanners.
- `llc` is not shipped in these ROCm trees. The IR reproducers carry `llc` `RUN:` lines for upstream; the scripts
  compile them with the compilers' clang driver (`clang --target=amdgcn-amd-amdhsa -mcpu=<gpu> -x ir -S -O2`), which runs
  the same backend.

The deliberate breaks, in short: bundle 01 inserts a `v_nop` before `s_setpc_b64` or after `s_swappc_b64` in a copy of the
listing (detector goes RED); bundle 02 moves the `v_wmma` of `bad_if` out of the mask (RED) and injects one into
`good_select` (flagged); bundle 03 moves the sunk source definition in front of the mask (RED) and redefines the source
inside the mask of `good_pinned` (flagged). The hygiene scan has its own six breaks on a copy (home path, macOS path,
login name, `__pycache__`, stray listing, emoji).

## Before any of these is filed

- Rebuild with a current upstream LLVM main (the Fedora clang 22.1.8 is the newest upstream compiler tried here) and
  search the LLVM issue tracker for duplicates.
- Run the bundle reproducers themselves on a gfx12 card (and, for 01, a gfx11 card); the hardware numbers here are from
  the study's kernels.
- Remove the "Provenance" sections, which point at files of the local study.
