# Item D expected outcome (written before any GPU job of this item)

Question: with one dispatch of 0.3, 0.6, 1.0 or 1.7 ms issued every 16.7 ms and waited on (call, wait), which core clock does
the shader see, per dispatch, over 20 s, and how much does it differ between runs? PR 57 saw about 1475 MHz on Windows with
20-40 % swings between sessions.

Sources: `measurements/scheduling-and-clocks.md` section 5, `gaming-probes.md` (the clock-step rule), `round7.md` A (in-kernel
clock about 3.07 GHz on profile "Default" under full load). INFERRED unless marked.

| Cell | Expected in-kernel clock | Reasoning |
|---|---|---|
| any cell, 6 to 10 % duty (1.0 ms of 16.7 ms is 6 %) | a low-state level between 1.5 and 2.8 GHz; may differ from 1475 MHz because that is the Windows driver | The value of the item is the spread and the dependence on dispatch length. |
| 0.3 ms and 0.6 ms against 1.0 and 1.7 ms | the longer dispatches sit higher or step up within the dispatch | The rule in `gaming-probes.md`: the clock steps up when the gaps are short; here the gap is 15-16 ms. |
| the same cell repeated 4 times (Vulkan 1.0 ms) | spread of the run medians under 20 %; above 20 % would confirm PR 57's swing on Linux | |
| in-kernel clock against the sysfs reported clock | not equal; the gap of round 7 ("The clock gap") | The sysfs value is an average over the table refresh and omits the offset (open in round 5 G). |
| first dispatch after 1 s idle | lowest of the run | The card starts from its idle state. |

Controls that can fail (judged by `analyse_clk.py`):
1. calibration: three back-to-back 30 ms dispatches at the end of every cell; the clock from cycles over REALTIME agrees
   within 1.5 % with cycles over the queue's own timestamps (Vulkan) or the HIP events (HIP).
2. waves agree: median over the dispatches of (highest wave clock / lowest wave clock) at most 1.03.
3. dispatch length: median REALTIME length of wave 0 within 3 % (or 0.03 ms) of the planned length.
4. cadence: median submission spacing within 1 % of 16.7 ms; the program also judges lateness (p95 200 us).
5. realtime shift break: the analysis recomputes every clock with REALTIME ticks scaled by 1.05; the median clock must move
   by a factor of 1/1.05 within 0.5 %. If the pipeline does not respond, the verdict FAILs.
6. at least 90 % of the planned dispatches ran and carry two or more records.

No profile switch is part of this item (the "Testing" comparison, G2, is not approved). The gap rule re-check at 2 ms and
5 ms gaps is not run on "Default": `cad=` accepts it, but no expectation is claimed.
