# Item C2 expected outcome (page-order walk), written before any GPU job of this item

Question: item C walked the 4 MiB region in a random order and read 505 cycles after every cache-doing barrier, 711 through HIP
two launches and 770 for a region nobody wrote. The expectations of item C took the page-order rows of `memory-system.md`
(GDDR6 485-545, memory-side cache 360-390) while the walk was random, so two controls failed on a wrong band. C2 repeats the
cold, fused, two-launch and b2 cells with the walk in page order (increasing: next = slot + 1, the 4 MiB region in address
order, 1,024 steps = 128 KiB), so the page-order rows apply. Which band the b2 and two-launch cells fall in settles what the
505 and 711 of item C were.

Sources: `measurements/memory-system.md` 2b (page-order columns: memory-side cache 358-394 cycles, GDDR6 512-544 at 512 MiB-1 GiB,
the task's band 485-545), `measurements/round9.md` item C (MEASURED: random walk: cold 768-772 / 699, fused 293-296 / 231, b2
and the other cache kinds 502-509, two launches 711). Everything below is INFERRED; nothing has been measured with the page walk.

| Cell | Expected cycles per step | Reasoning |
|---|---|---|
| cold (region nobody wrote, after a 512 MiB read), Vulkan g and c, HIP | 485-545 (band judged 470-560) | GDDR6 in page order. Item C's 770 was the random-walk row (700-820). |
| fused (same kernel writes and chases), Vulkan g and c, HIP | 230-300 (judged: at most 330 and under 0.7 x cold) | Same-kernel level; the order of the walk should matter little (item C: 293, 296, 231). |
| HIP two launches (producer kernel, then consumer kernel) | 485-545 (GDDR6) | Item C's 711 sits in the random GDDR6 row (700-820): the data came from GDDR6. If it falls at 360-390 instead, the data was in the memory-side cache and the random walk's extra was page changes. |
| Vulkan normal, kind b2, compute and graphics queue | 360-390 (memory-side cache) | Item C's 505 lies between the random memory-side-cache row (420-500) and the cold 770, so it reads as memory-side cache. If b2 falls at 485-545 the data came from GDDR6 after all. |
| race (consumer before producer) | whatever; judged only by the pointer check | The break. See the note on the pointer check. |

Controls that can fail (judged by `analyse_cbar.py --page`; self-test runs first):
1. cold: median in 470-560 cycles, Vulkan g, Vulkan c and HIP.
2. fused: median at most 330 and under 0.7 of cold.
3. HIP two launches: median in 350-400 (memory-side cache) or 470-560 (GDDR6). Either passes; which one is the finding.
4. race break: at least 90 % of the counted repetitions fail the pointer check, per queue and HIP. The race cell alternates the
   walk's direction on odd repetitions so its stale data differs from its fresh data (see below).
5. repeats: the medians of one control cell over the 3 process repeats agree within 10 %.
6. b2 on the compute queue and on the graphics queue: pointer check ok in every counted repetition (trivially true, see below).

Findings (not judged): the BAND lines print where every cell sits: memory-side cache (350-400), GDDR6 (470-560), same-kernel
(at most 330), or between the bands.

Limits of the design (listed so the review looks at them):
- With one walk the data of a region is the same in every repetition, so the pointer check of the normal, cold and fused cells cannot
  tell a stale region from a fresh one (stale data holds the same pointers). The race cell alternates the direction (next = slot + 1
  on even repetitions, slot - 1 on odd ones) so that the check can be seen to fail there. The check of the normal cells is therefore only as
  strong as the same check in item C's random walk (57 of 57 correct); the latency is the finding here, not the pointers.
- A walk of 1,024 steps in address order touches 128 KiB of the 4 MiB region. Hardware prefetch could hide part of the miss
  latency in page order; the cold cell would then read below 485 and the control would FAIL, which is the intended detection.
- Memory clock: each cell prints uclk_start and uclk_end; page-order rows of `memory-system.md` were measured at memory clock
  1258, this probe runs at whatever the card does (456 to 1408 within one job); read the uclk values before quoting.
