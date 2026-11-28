# Expectations, round 11 item 1: ray tracing units through Vulkan ray queries (written before the first GPU job)

Labels: READ (notes/research-8-architecture-and-family.md section 4, Chips and Cheese 2025-04-14, Wccftech deep dive 2025-02-28, ServeTheHome Hot Chips 2025-08-25), INFERRED.
Scope: timing, shader-clock cycles and the compiler listing only. The probe does not read acceleration-structure bytes (PLAN-r11 open question 3, default no).
RADV is the library the README names (Mesa 27.0.0-devel; `vulkaninfo` driverInfo, READ); the run folder records it again.

## Claims (READ) and what the probe can say about each
- Two intersection engines per ray accelerator: 8 box or 2 triangle tests per cycle (RDNA 2/3: 4 and 1). Not separable from a shader-driven traversal; the box/triangle
  mix is varied (S1 vs S2 slivers) and the listing says which instruction runs.
- `IMAGE_BVH8_INTERSECT_RAY`, `IMAGE_BVH_DUAL_INTERSECT_RAY`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`: the probe counts every `bvh` mnemonic in the ACO listing of the three pipelines
  (`LIST` lines, `asm/*.s`). If the BVH8 instruction is absent, the "BVH8" question is answered by the listing and the timing is labelled "BVH4 path".
- "2x ray traversal performance vs RDNA 3 at equal clocks" (AMD slide): RDNA 3 is not on this card; reported as rays/s and cycles per ray only.
- Oriented boxes "10 %": S2 diagonal (loose axis boxes) against S2 axis-aligned (tight boxes) is the nearest black-box contrast; whether the driver uses oriented boxes is NOT tested
  (that needs the bytes). Say so in the write-up.
- Traversal is the shader's loop (inline ray query), so rays/s is a shader-plus-unit rate, not the unit's own. Results are for RADV's builder, not AMD's.

## Bands
- Ceilings (INFERRED, not predictions): 64 ray accelerators x 2.9 GHz: 1.5 T box tests/s (8 per cycle), 0.37 T triangle tests/s (2 per cycle), 0.19 T node visits/s (one per cycle).
- No prior band for rays/s or ns per ray on this card was read: **no prior band**, first-run result, read only after the controls pass.
- Shapes that must hold if the probe works (INFERRED): floor cells (tMax 0.5, every ray misses at the root) faster than any hitting cell of the same scene; per-ray chain time grows
  with N by one tree level per factor of 8 (slope check, not a number); first-hit not slower than closest-hit; shuffled order not faster than coherent.
  A shape that fails is reported, not forced.

## Controls (a failed control stops the cells after it)
1. Analytic S0: a 64 x 64 lattice of rays (i + 1/4, j + 1/4) over a right triangle with legs 64 hits exactly 64 x 65 / 2 = 2080 rays.
2. Per scene, a 4096-ray sample: card (hit/miss exact, distance 1e-4 relative) against CPU brute force over every triangle (rays at triangle centroids and random points in the cell).
3. Rate cells (S1, S3): the card's hit count equals the CPU's exact lattice count (the rays sit on a 1/16-cell lattice with a 1/64 offset, so no ray is on an edge); floor and empty-TLAS cells must hit 0.
4. Chain: the one-lane dependent chain >= 3x the same 4096 rays independent on one wave; otherwise the latency cells are void.
5. Core clock (cycles over the 100 MHz realtime counter) of the first and last repeat within 3 %, else the cell is flagged `clock_flat=NO`. Default (unpinned) policy: a flagged cell does not fail the run; the verdict is qualified (`PASS (unpinned clock ramp: N/M cells ...)`), `SUMMARY` carries `clock_flat_failed=N/M`, and the rates are read at a drifting clock. With `--strict-clock` a flagged cell prints `CTRLFAIL` and the run fails. `run_r11_rt.sh` `analyse()` requires the `clock_flat_failed=N/M` field and cross-checks it against the cells.
6. Occupancy cells (4k, 16k, 64k invocations) stand in for the 1/4/16 waves-per-SIMD cells of the draft (register-count based cells are a follow-up).

## Deliberate breaks (each must make its control FAIL)
1. A triangle moved on the card only (vertex buffer), CPU copy unchanged: the 4096-ray check must report mismatches (a ray is aimed at the moved triangle).
2. The chain run with all 32 lanes active: the chain check must FAIL. (The draft's "drop the zero dependency" variant is also built as shader parameter `dep` and reported, but a single lane's traversal loop
   is serial either way, so it cannot be the break.)
3. One extra instance against the hand count: must FAIL (card hits 257 against 256).
Replaced: the "unbuilt TLAS" idea is undefined use and could hang the card; a built, EMPTY top level (valid) is a floor cell that must hit 0.
CPU proof so far (no GPU): `r11_rt_selftest` is green and goes red under each of its 6 breaks (break 6: a drifting cell must fail under `--strict-clock`); the integer hit rule matches brute-force geometry for both triangle forms and both ray orders. `run_r11_rt.sh --cpu-checks` also feeds `analyse()` synthetic logs: a flat clock passes, a drifting clock passes with a warning, `--strict-clock` with a drifting cell fails, and a missing or wrong `clock_flat_failed` count fails.

## Estimate
68 rate cells + 7 chain cells + 3 control scenes: about 4.5 min of card load at 30 repeats; wall 20 to 30 min because each scene waits for the hotspot below 60 C (up to 120 s each).
Host memory under 2 GB; disk under 100 MB (listings, repeats CSV).
