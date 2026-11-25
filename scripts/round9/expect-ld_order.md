# Expectations for ld_order (round 9, item B), written before any GPU job

Source: `DRAFT-r9-plan.md` section B, `measurements/memory-system.md` sections 5 and 6, `measurements/matrix-timing.md` 8a.
All numbers are cycles of the in-kernel counter, minus the `empty` pattern (two counter reads back to back, a few
cycles). INFERRED unless marked; nothing here has been measured with this probe.

## Question

With a far load (GDDR6) issued first and a near load (L0) second, does `s_wait_loadcnt 1` return after the far one
(in-order retirement, which PR 57's scan assumes), or can the near load's data be used early? Do the other counters
(`dscnt` for LDS, `kmcnt` for scalar loads) run on their own?

## Numbers expected

| Pattern | Expected cost (cycles) | Why |
|---|---|---|
| `pure_hit` (H, wait 0) | **corrected after run 1: 77 (gate 69-85)**; first draft 129-135 (gate 124-140), kept here as written | 77 = L0 hit with its wait, `matrix-timing.md` section 8a (line 354), timed by differential method like this probe. The draft took 129.1 (4 B, N = 1) of `memory-system.md` section 5, a repeating loop whose period includes loop and address arithmetic; its 135.1 is the 16-byte row, not 4-byte |
| `pure_miss` (M, wait 0) | 550-770 | memory batch, `memory-system.md` section 5 (550-870) and the draft |
| `mh_w1` [miss, hit], wait 1 | the miss: 550-770 | in order: the count falls to 1 when the miss returns |
| `hm_w1` [hit, miss], wait 1 | the hit: about 130 | the count falls to 1 when the hit returns |
| `mh_w0`, `hm_w0`, `mm_w0` | the miss, plus a few cycles per extra load issued | wait 0 needs everything |
| `mm_w1` | the miss | two misses fly in parallel, the first one is waited |
| `mhhhh_w1..4` | the miss | the first load is a miss and every wait covers it |
| `hhhhm_w1..4` | the hit (about 130-145) | the first 4 loads are hits; w1 waits for 4 of 5 |
| `hmhmh_w1`, `_w2`, `_w3` | the miss | the first covered loads include a miss |
| `hmhmh_w4` | the hit | only the first load is covered |
| `lds_only` | 30-70 | LDS latency (`matrix-timing.md` 8a: 32 cycles with the wait) |
| `s_hit` | 55-110 | scalar-cache hit (`memory-system.md`: 70 cycles at 16 steps ahead of a scalar prefetch) |
| `s_far` | 650-760 | scalar load, far (`memory-system.md`: 723) |
| `gl_dswait` first snapshot | the LDS latency, far data not yet in the register | dscnt is its own counter |
| `gl_dswait` second snapshot | the miss | |
| `m_s_kmwait` first / second | the scalar hit / the miss | kmcnt is its own counter |
| wrong words in every `must` register | 0 | |
| `brk_wrongcnt`, `brk_wrongkm` | wrong words in at least 90 % of the lane checks (about 100 %: the far data cannot have arrived after a 30-70 cycle wait) | the deliberate break: waits the wrong counter and claims the global register valid |

The second snapshot of a two-snapshot pattern includes about 8 cycles for the copies made after the first.

## Verdict rule

- Expected (INFERRED): `ORDER IN ORDER`. [miss, hit] wait 1 costs the miss, [hit, miss] wait 1 costs the hit.
- If the first two cost the same, or the hit overtakes the miss (`mh_w1` near 130): **stop and report**: the in-order
  assumption fails and PR 57's scan is wrong about what a wait on the newest load costs.
- If `brk_*` shows fewer than 90 % wrong words the probe cannot see an early return and every `bad_zero` result is void.
- If `pure_hit` or `pure_miss` leaves its gate, the probe is not measuring what `mem_inflight` measured: report, do not
  interpret the rest.

## What the controls prove

| Control | Proves |
|---|---|
| `pure_hit`, `pure_miss` | the near and far loads are really an L0 hit and a GDDR6 miss (against `mem_inflight`) |
| `bad_zero` | no early return was seen in any correct wait: data read after `s_wait_*` was always there |
| `break_wrongcnt`, `break_wrongkm` | the register read can see "not yet": the sentinel survives when the wrong counter is waited |
| `order_probe_can_tell` | `mh_w1` and `hm_w1` differ enough for the question to have an answer |
| `mix_in_order` | the five-load mixes follow the in-order model; the out-of-order fit is printed beside it |
| `dscnt_independent`, `kmcnt_independent` | the other counters retire on their own |
| listing check (`build_round9.sh`) | the code object holds the pattern text as written (opcodes, order, registers) and runs one wave in wave32, CU mode |
