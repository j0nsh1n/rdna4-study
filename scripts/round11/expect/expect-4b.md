# Expectations, round 11 item 4b: PCIe copy speed, latency, resizable BAR (written before the first GPU job)

Labels: READ (opened or measured earlier), INFERRED (arithmetic here, not observed). Every wall-clock number is for the overclocked card
(+150 MHz, -50 mV, 374 W cap); copy speed is bus-bound, so the profile matters little, but say it.

## Facts read today (sysfs, no GPU job; scripts/round11/run_r11_pcie.sh --cpu-checks)
- Card 0000:03:00.0 (0x7550): max link 32.0 GT/s x16, current 32.0 GT/s x16 at the read (READ). BAR0 is 16384 MiB and the CPU-visible card memory equals
  the total (17,095,983,104 bytes): resizable BAR is ON (READ).
- `pp_dpm_pcie` has three levels: 2.5 GT/s, 16.0 GT/s, 32.0 GT/s (READ). `gpu_metrics` reported 2.5 GT/s (pcie_link_speed 25, x16) while the card was nearly idle
  (READ, r11_sampler 1.2 s sample). The link can be below 32 GT/s at the start of a job: the first repeats may be slow. The probe makes 3 warm-up copies
  and the sampler's `pcie_gts_x10` column records the level during the job; a cell is read at the level the sampler shows.

## Ceilings (INFERRED arithmetic, per direction, payload only)
| link | formula | ceiling |
|---|---|---|
| Gen5 x16, 32 GT/s | 32 x 16 x 128/130 / 8 | 63.015 GB/s |
| Gen4 x16, 16 GT/s | 16 x 16 x 128/130 / 8 | 31.508 GB/s |
| Gen1 x16, 2.5 GT/s | 2.5 x 16 x 8/10 / 8 | 4.0 GB/s (8b/10b coding; the probe does not model this level) |
Both directions at once: twice that. A rate above the ceiling of the card's MAX link rejects its cell (the probe was not moving the bytes).

## Bands
- Pinned host memory, 64 MiB and larger, Gen5 x16: expect 50 to 58 GB/s per direction (INFERRED: 256-byte packets and flow control cost 8 to 20 % of the ceiling).
  No public number for this CPU/board class was fetched in this slice: "no prior band", first-run result, read only after the controls pass.
- Pageable host memory: lower than pinned (the runtime stages through a pinned buffer); no band, first run.
- Pageable off by one byte: not faster than aligned pageable; no band.
- Both directions at once: below twice one direction; no band.
- Latency (4 B, 64 B copy-and-wait): a few to tens of microseconds; no band, first run. Flag ping-pong: round trip of a few microseconds (INFERRED: two
  PCIe writes); half is one way only if the two directions are equal (INFERRED).
- `HSA_ENABLE_SDMA=0` job: copies run on blit kernels; speed may differ; the difference is the result.
- Card-to-card copy 256 MiB: read plus write between 443 and 665 GB/s (0.70 to 1.05 of the measured 633 GB/s read, `measurements/memory-system.md`: 633 read, 581
  write, 547 copy). The window is fixed here, before any run.

## Controls (a failed control means the cells are not interpreted)
1. Every cell: destination zeroed, copy, exact 64-bit hash (card side after h2d, host side after d2h) equal to the host's. The probe's arithmetic is tested on
   the CPU (`r11_pcie_logic_test`): ceilings, rate rejection, hash sensitivity to every bit of a word and to a swap.
2. Rate above ceiling: cell rejected.
3. Card-to-card copy inside the window above.
4. Link and BAR facts and the sampler's link column recorded for the job.

## Deliberate breaks (each must make its control FAIL; a break that stays green makes the probe invalid)
1. One byte flipped on the card after a verified 64 MiB copy: the hash comparison must FAIL.
2. The copy skipped (256 MiB "copied" in microseconds): the rate must exceed the ceiling and be flagged.
3. Flag ping-pong with no responder: the card's bounded wait must time out and complete 0 of 3 round trips.
CPU proof so far (no GPU): the logic test is green; with the ceiling formula broken (`-DR11_BREAK_CEILING`) and with the hash blind to a byte (`-DR11_BREAK_HASH`)
it goes red.

## Not in this slice
Vulkan CPU-write/read timing of the BAR window (the sysfs BAR size answers the resizable-BAR question; the timing arm is a follow-up), forcing the link speed (needs root).
