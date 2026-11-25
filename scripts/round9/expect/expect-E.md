# Item E expected outcome (written before any GPU job of this item)

Question: at the memory clock the card now reaches (1408 MHz under a streaming load), what are the latency and bandwidth of
L0, L2, the memory-side cache and GDDR6, and does a cache level's latency depend on the memory clock state (456, 772, 875,
1124, 1408)? A compute-only load kept the memory clock at 772 in round 7 (FP16 loop).

Sources: `measurements/memory-system.md` 2a-2c; `runs/20261124T053703Z-r7mem/RESULTS.md` (700.3 GB/s read at 1408 MHz,
bandwidth about 0.5031 GB/s per MHz); `round7.md` A3. INFERRED unless marked.

| Point | Expected | Reasoning |
|---|---|---|
| L0, 16 KiB chase | 81 cycles, unchanged by the memory clock | MEASURED 81 in 2b. The memory clock does not feed L0. |
| L2, 1-4 MiB chase | 230-255 cycles, unchanged | MEASURED 230-255. L2 sits on the core clock side. |
| memory-side cache, 32 MiB chase | 360-500 cycles; the same in cycles at every memory clock, or more ns at 456 | Not known whether the memory-side cache follows the memory clock; that is the new finding if it does. |
| GDDR6, 512 MiB and 1 GiB chase (fresh) | ns falls a little at 1408 (a faster controller), cycles about the same | The 137 ns known answer is from an older profile. |
| bandwidth, 1 GiB split | 700.3 GB/s +-1.5 % at 1408; bandwidth over the clock reading about 0.5031 GB/s per MHz at 1124 and above | MEASURED in round 7. |
| bandwidth, 32 MiB split | above the GDDR6 value (served from the memory-side cache) | MEASURED in 2a for the old tables. |
| cold chases at process start | labelled by the reading just before the chase; split by 456 / 772 / 875 / 1124 / 1408 | The reading is a 10 ms table refresh, so a label can be one step stale. |
| chases with a background FP16 loop | the memory clock sits at 772 (round 7); L0 and L2 as above | The reading says which level the chase saw. |

Controls that can fail (judged by `analyse_hier.py`):
1. L0 chase between 75 and 90 cycles, every job.
2. L2 chase (1-4 MiB) between 215 and 270 cycles, every job.
3. GDDR6 read bandwidth at the 1408 label: 700.3 +-1.5 %. Judged only if a job carries that label; otherwise the verdict
   says SKIP (not judged) and the report must say the control was open.
4. Label against bandwidth: for every 1 GiB job labelled 1124 or above, bandwidth over the label within 3 % of 0.5031.
   Break: shifting every label by one level must be caught (verdict FAIL on the shifted data), as round 7's control 2.
5. Every pointer check ok.
6. A job's label is the level that holds at least 90 % of the sampler's samples in the job's window; a job below that is
   labelled "mixed" and kept apart.

Not expected to be clean: the cold cells depend on what the clock does between jobs; the driver waits for 456 before
half of them and not before the others, on purpose.
