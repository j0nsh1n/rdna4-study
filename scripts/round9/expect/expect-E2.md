# Item E2 expected outcome (latency at the 1124 and 1408 memory-clock labels), written before any GPU job of this item

Question: item E measured latency at the 456 and 772 (and one 875) cold-start labels only, because a latency chase does not raise
the memory clock itself. What are the L2, memory-side-cache and GDDR6 latencies when a background streaming read holds the memory
clock at 1124 or 1408, and do they move with the memory clock? Each chase is labelled by the memory clock the 5 ms sampler read
during it (the samples inside the chase's own monotonic window; at least 3 samples and at least 90 % in one level), not by the table
reading before and after it.

Design: a second process, `r9_bgread`, reads its own 1 GiB buffer (not the chase's) with a few work-groups (2, 8, 32 blocks of 256
threads, windows of 64 MiB or more swept round the buffer, so the traffic is GDDR6 traffic and not memory-side-cache hits). The
chase is one wave (`r9_lat`, mem_latency.hip's chase kernel), at 1 MiB (L2), 32 MiB (memory-side cache, random and page order) and
1 GiB (GDDR6, random, "fresh" and page order), repeated until the point lasts 60 ms (the sampler needs a few samples). Two idle jobs
(no background) are the comparison at the idle clock. Whether the label that results is 1124 or 1408 is not chosen by this
driver: the card's own policy decides, and the analysis says which labels were reached.

Why the background is light (evidence, not a promise): (1) its own rate is printed (`RESULT r9_bgread ... gbps=`) and the analysis
requires at most 70 GB/s (10 % of the card's 700 GB/s) in every background job; (2) the L2 chase at 1 MiB must stay in 215-270
cycles and within 5 % of the idle jobs' (the stream's lines share the L2 and the fabric but not the chase's lines); (3) the 32 MiB
page-order chase must stay below 470 cycles, so the memory-side cache has not been emptied of the chase's lines by the stream; (4) where
two jobs reach the same label with background rates 2x apart, the 1 GiB page-order chase must agree within 5 % (the background's
rate does not change the answer). Each of the four is a control that can fail.

Sources: `measurements/round9.md` item E (MEASURED: L2 1 MiB 245.9 cycles at 456; 32 MiB random 552 at 456, 454 at 772, 578 at
875; 1 GiB random fresh 467-483, random after a long walk 852-879; those are cold-start labels, random walk) and
`measurements/memory-system.md` 2b (page order: memory-side cache 358-394 cycles, GDDR6 512-544). INFERRED unless marked.

| Point | Expected | Reasoning |
|---|---|---|
| L2 1 MiB, random | 215-270 cycles at every label | Item E: unchanged at 456, 772 and 875. The L2 is on the core clock side. |
| 32 MiB, page order | 360-420 cycles at every label (flat within 10 % from 456 to 1408) | `memory-system.md` 2b page order 358-394. Item E's random 32 MiB moved with the label (552 to 454), so a drop of 10-20 % at the higher labels is possible; if it moves more than that, that is the finding. |
| 32 MiB, random | 420-560 cycles; falls with the memory clock as in item E | Item E: 552 at 456, 454 at 772. |
| 1 GiB, page order | 485-545 cycles at 456; at 1124 and 1408 not known: this is what the item is for | The 137 ns old answer came from an older profile; a faster memory clock should shorten the DRAM part, not the cache part. |
| 1 GiB, random fresh | 458-483 cycles at 456 to 875; not known above | Item E. |
| 1 GiB, random after a long walk | 830-890 cycles | Item E: 837-885 (translation entries pushed out, not the memory clock). |

Controls that can fail (judged by `analyse_E2.py`; its self-test with breaks runs first):
1. every pointer ok.
2. a clean chase at the 1124 label and one at the 1408 label. If a label is not reached the item stays open at that label and the
   report says so (no value is invented). The driver's list of blocks can be changed (R9_E2_BLOCKS) for another try.
3. background at most 70 GB/s in every background job.
4. L2 1 MiB random within 215-270 and within 5 % of the idle jobs.
5. 32 MiB page order below 470 cycles.
6. 1 GiB page order agrees within 5 % between two jobs at one label whose background rates differ 2x (SKIP when no such pair).

Findings (not judged): the table of median cycles and ns per size, walk and label; the ratio between labels per size (above 1.10 in
cycles or ns: depends on the memory clock); the comparison with the 456 and 772 cold-start values above.

Limits of the design (for the reviewer):
- The label is a window of up to a few hundred ms during which the clock can still move; a chase is "mixed" below 90 % in one level
  and is kept apart, so some points will be lost, and a label with few points has a wide error.
- Cycles are core cycles and the core clock moves with the load; ns (from REALTIME) are the better figure across labels. Both are printed.
- The background shares the memory controller with the 1 GiB chase even when light; control 6 is the check, and only where a pair exists.
- The sampler's memory-clock reading is the driver's table reading (10 ms refresh); a chase can be labelled one step stale.
