# Item C expected outcome (written before any GPU job of this item)

Question: after a producer dispatch writes a 4 MiB region, does the consumer's first read of it come from L2 (about 250
cycles per dependent load) or from the memory-side cache (about 360-500) or GDDR6 (485-545), per Vulkan barrier kind, on
the compute and the graphics queue? Same through HIP (two launches).

Sources: `measurements/memory-system.md` 2b/2c (L0 81 cycles, L2 230-255, memory-side cache 360-500, GDDR6 485-545);
`measurements/gaming-probes.md` near line 137 (what RADV records per barrier kind). Everything below is INFERRED
unless it says READ or MEASURED; the point of writing it first is to see where it is wrong.

| Cell | Expected cycles per step | Reasoning |
|---|---|---|
| cold (region nobody wrote, after a 256 MiB read), Vulkan g and c, HIP | 450-600 (GDDR6) | MEASURED 485-545 in 2b for one wave chasing a large buffer. Page-table misses on a first-touch region may add a little. |
| fused (the kernel that wrote it chases it), Vulkan g and c, HIP | 80-330 (L0 or L2) | The wave that reads shares a CU with the writers. Data just written by the same group. |
| HIP two launches (producer kernel, then consumer kernel) | 330-540 (memory-side cache) | MEASURED in 2c: L2 does not keep data across launches. A write-back at the end of a kernel leaves the lines in the memory-side cache. |
| Vulkan normal, kind b2, legacy, full on the compute queue (GL2 write-back and invalidate, 0xc180) | same as the HIP two-launch cell | The L2 is invalidated over its whole range (READ in `gaming-probes.md`), so the consumer cannot hit L2. |
| Vulkan normal, kind vv7 (write to write only, 0xc000, L2 part only) | same as b2 | INFERRED. The L2 part is the one that matters here. |
| Vulkan normal, kind full on the graphics queue (RELEASE_MEM, GCR 0x704) | possibly L2 hits (150-330); not known | This is what the item is for. |
| Vulkan normal, kind none, exec | possibly L2 hits, and the pointer check may fail (no memory dependency is recorded) | exec is only CS_PARTIAL_FLUSH. Reported, not judged. |
| Vulkan normal, kind event | like b2 | Not known; reported. |

Controls that can fail (judged by `analyse_cbar.py`):
1. cold: median in 440-600 cycles, per queue and for HIP.
2. fused: median at most 330 and under 0.7 of the cold cell.
3. HIP two launches: median in 330-540. If it is not, the probe or the card state differs from 2c.
4. pointer check: kind b2 on the compute queue must have every counted repetition ok (the barrier a program would use).
5. race break (consumer before producer, nothing between): at least 90 % of the counted repetitions must FAIL the pointer
   check. If they pass, the check cannot see stale data and no other verdict of this item counts.
6. repeats: the medians of one cell over the 3 process repeats agree within 10 %.

Parts of the design that can mislead (listed so the review looks at them):
- The consumer's loads are vector loads on purpose (the index carries `lane & zero`, which the compiler cannot fold);
  a uniform address could have become a scalar load, which uses another cache.
- The cold region may be in the memory-side cache if the 256 MiB read did not push it out; the cold control would then read
  low and FAIL, which is the intended detection.
- All 32,768 lines are written, only 1,024 are read, so the chase touches 128 KiB of the 4 MiB.
- Memory clock: each cell prints uclk_start and uclk_end; cells at different levels are not merged.
