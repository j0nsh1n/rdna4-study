# Second fix round: expectations for F1 to F5, written before any F run

Written 2026-11-18 at 01:16 UTC (file time 01:16:53 UTC), before the first F job (development jobs included); this
line was corrected at 01:20 UTC from a guessed time window, nothing else changed. Every F job's log
starts with its own UTC time, so the order can be checked. Nothing in this file is changed after the runs; what the
runs showed goes into `gaming-probes.md`.

Common: the load is `gdense.comp` (128 independent `v_fma_f32` per chunk). "Low level" means the 1.7-1.8 GHz plateau
seen under Vulkan in E1 to E3, "step" the rise to about 3.2 GHz. The fill is FMA instructions per SIMD per cycle over
every wave of a dispatch (128 SIMDs; at most 1), from the per-wave chunk counts; from this round on every burst's
per-wave counts are kept, not only the last burst's.

## F1. Whole-card bursts that are really dense

Design. 64 groups of 1,024 threads (2,048 waves), graphics queue, the time bound checked every 8 chunks (as E1's
whole-card ramps, which reached 0.86 to 1.0); 2 ms bursts every 16.7, 11.1, 8.3 and 6 ms and 4 ms bursts every 16.7 ms,
1 s each after 1 s idle (all planned at or below 0.33 busy), one arm per job, 5 rounds in rotating order. First one
job with 2 ms every 16.7 ms only, then power and hotspot are read before the rest runs. A host-side guard ends an
arm whose measured busy share passes 0.40 once 100 ms have run.

Expectation. A dense whole-card burst should raise the fill to about 0.85-1.0 per burst. Two outcomes are open and I
do not favour one strongly:
- (a) the clock stays at the low level in every burst at every cadence, as E2's light and 64-wave bursts did: the
  rule is "about 27 ms of unbroken work", whatever the density. Most likely at 12 to 24 % busy.
- (b) at 6 ms cadence (about 0.33 busy) or with 4 ms bursts the clock leaves the low level, because power or
  activity counts and not only unbroken time.
What contradicts (a): any burst above 2 GHz in a clean run. What contradicts (b): no burst above 2 GHz in any run at
6 ms cadence or 4 ms bursts. Either way a fill below 0.8 in a burst means the arm did not test what it claims, and it
is reported as such. Expected starved-wave overrun at 8 chunks per check: under about 0.3 ms per 2 ms burst
(INFERRED from E2's 0.6 to 1.8 ms at 64 chunks).

## F2. The order effect

Design. Targets: one wave and 64 one-wave groups (per CU), each as the only arm of its own process (1 s idle, then
60 ms), on the Vulkan graphics and compute-only queues; for the reverse question the same two shapes through HIP.
Predecessor, in the same locked job: (a) the HIP twin, one wave, 60 ms, followed by a pause of 1, 5 or 20 s before
the target process starts; (b) the Vulkan probe, one wave, 60 ms, graphics queue, pause 1 s; (c) nothing for at
least 20 s (the lock is held through that pause, so no lock-respecting job can run in it). HIP targets get (b) and
(c). 6 runs per combination, all jobs in one randomised order (the seed and the order are recorded). Measured: the
target's first 0.25 ms window, its 1-10 ms plateau and the step time (first window more than 5 % above the 1-5 ms
plateau), and the time from the predecessor's load end to the target's load start.

Expectation (from E1's 3 of 3 against 0 of 2, and the lead's rerun, where the same 3 rounds repeated it):
- Vulkan one-wave target after a HIP job with a 1 s pause steps early (about 10 to 13 ms) in most runs (at least 4 of
  6 on the graphics queue); after a Vulkan job or nothing it steps at about 26 to 28 ms. Contradicted by an early
  step in 2 or more of 6 runs after a Vulkan job or nothing, or by fewer than 3 of 6 early steps after HIP at 1 s.
- The start level does not change with the predecessor (about 1.80-1.82 GHz for one wave): contradicted by a start
  level after HIP that differs by more than 30 MHz from the others in most runs.
- How long it lasts: I expect the effect to fade with the pause, gone by 20 s (weak expectation; E1's gaps were a few
  seconds). Contradicted by early steps at 20 s as often as at 1 s.
- Per-CU target: E1 never saw an early step for 64 waves, but it never ran the per-CU arm first after HIP. No strong
  expectation; I lean to "no early step" (HIP's own 64-wave step is at 34-35 ms, later than its one-wave step).
- Compute queue: no expectation beyond "if the effect is state left by the HIP process on the card, it shows on both
  queues".
- Reverse: a Vulkan predecessor does not change HIP's start (2.77-2.79 GHz) or its step (11-12 ms one wave, 34-35 ms
  per CU). Contradicted by a HIP start below 2.6 GHz or a step that moves by more than 3 ms in most runs.

## F3. Gaps with a load as heavy as real frames

Design. 2.8 ms bursts (the sibling lane's frames were about 2.8 ms) with gaps of 0.5, 1, 2 and 3 ms, on both Vulkan
queues; (i) 64 groups made dense by more waves per group (64 groups of 128 threads, 256 waves, checked every 8
chunks; if a development job shows a fill below 0.6 per burst, 64 groups of 256 threads are used instead, decided
before the final run and recorded), (ii) the whole card (64 x 1,024, every 8 chunks). All arms are above 0.40 busy, so
each runs at most 0.1 s (about 17 to 30 bursts); one arm per job, 5 rounds, order rotating.

Expectation. With 0.5 ms gaps both loads step after about 25 to 45 ms (as E3). The question is 1 and 2 ms gaps:
- If what the sibling's DIRECT queue saw at 1.1 and 2.1 ms gaps comes from the frames being heavy, the dense loads
  here step at 1 ms gaps and probably 2 ms gaps on the graphics queue.
- If it comes from the route (D3D12 through vkd3d-proton, its queue setup or submission pattern), plain Vulkan does
  not step at 1 to 3 ms gaps even with the whole card dense.
I lean slightly to the second (E2 and E3 showed no dependence on load at all so far). Contradicted by a step (a burst
above 2 GHz) at 1 or 2 ms gaps in most clean runs of a dense arm. No difference between the queues is expected (E3).

## F4. The calibration gap at the high clock

Design. The probe's 100-dispatch list (E4's base: one group, body 0, write-to-write barrier, compute-only queue),
in the same command buffer right after a warm-up dispatch (64 groups spinning about 100 million cycles, about 45 ms,
which steps the clock), with a one-group clock probe (2,000 cycles, stamped) just before the list's first timestamp
and one just after its last; the same list and probes without the warm-up (the low clock); same binary, rounds
alternating; and vkstudy in the same rounds.

Expectation. If the 1.73 against 1.12 us gap is the clock, the per-dispatch time scales with 1/clock: at 3.2 GHz
about 1.73 x 1.8 / 3.2 = 0.97 us if all of it is clock-bound, between 0.97 and 1.73 if only part is. I expect most of
the gap to close (0.95 to 1.25 us at the high clock), which would make vkstudy's 1.12 a high-clock figure.
Contradicted by a high-clock time above 1.5 us with the probes reading above 3 GHz. If the probes do not read above
3 GHz, the arm did not test the question and is reported as such.

## F5. The in-load stop, end to end

Design. The stop limit lowered by an option (`VKG_HOT_STOP_C=+2` for the Vulkan probe, `--hot-stop-c +2` for the HIP
twin: 2 C above the hotspot read when the arm starts), on a 100 ms load of 64 one-wave groups... or, if a development
job shows that load does not warm the hotspot by 2 C within 100 ms, the whole card for 60 ms (decided before the final
run). Graphics queue, compute-only queue and the HIP twin; a control with the normal 88 C limit in each.

Expectation. With the lowered limit, the sampler thread sees the limit crossed, sets the stop word, and every wave
ends within about one sampler interval plus one check (under about 6 ms) after the crossing sample; the log says the
stop word was set by the sampler and every group's run is shorter than planned. The control runs to its full length
with the stop word never set. Contradicted by a load that runs its full length although the sampler logged the
crossing, or by a control that stops.
