# first-hour.sh

The first checks on a card, the same way on the RX 7900 XT (gfx1100) and the RX 9070 XT (gfx1201).

1. `scripts/first-hour.sh info` (read-only, no GPU load). It writes the card record `card.env` (arch, device index, cache sizes, cache-trap size) to a new `runs/<UTC time>-<arch>/`. Read `summary.md` there.
2. `scripts/first-hour.sh build` compiles `src/` for that arch into `build/<arch>/`. `ARCH=gfx1201 scripts/first-hour.sh build` compiles without the card.
3. `scripts/first-hour.sh bandwidth probes decode llama libs` runs the GPU checks into the same run folder. `all` runs every phase in order, starting with `info` (roughly 20 to 30 minutes of GPU time, not yet measured).
4. Each GPU job takes the shared GPU lock (`GPU_LOCK` in `rdna3-fp8/build/local.env`) and runs under `nice` with the device index from the card record. It does not start below 8 GB of free memory and is stopped below 3 GB. It logs `pp_dpm_sclk` and `pp_dpm_mclk` before and after.
5. A failed step does not stop the run. `summary.md` marks it FAILED with its log path, and the exit status is 1.
6. Probe results are the `VERDICT` lines in the logs. `wmma_exact` first checks the operand layout with identity matrices and reports nothing else if that check fails.
7. The 8B model needs `MODELS="llama32-3b llama31-8b"`. `DRY_RUN=1` writes the GPU commands to the logs without running them.
8. hipBLASLt headers: install `hipblaslt-devel`. Until then, `build/third_party/hipblaslt-7.1.1/` holds a fetched copy (see its `SOURCE.txt`).
