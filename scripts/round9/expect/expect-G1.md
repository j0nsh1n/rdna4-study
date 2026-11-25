# Item G1 expected outcome (written before any GPU job of this item)

Question: is `power_dpm_force_performance_level = profile_standard` steady to about +-0.3 % in the in-kernel core clock, as PR 57's
Windows stable mode (1868 MHz) is, and how does that compare with the same 10 runs under `auto`? Approved by the owner
(steady clocks with `profile_standard`, set back to `auto` at once). G2 (the "Testing" profile switch) is not approved and not
part of this script.

Sources: `round6.md` (profile_standard gave about 1.65 GHz core and 1124 MHz memory). INFERRED.

| Measure | Expected |
|---|---|
| in-kernel clock under profile_standard, 10 runs of the 1.0 ms call-then-wait cell | about 1.6-1.7 GHz, run medians within +-0.5 %; the 0.3 % target may or may not hold |
| the same 10 runs under auto | the spread of item D; probably several percent or more |
| level after the script | `auto`, whatever happened (trap) |

Controls that can fail (judged by `analyse_stable.py`):
1. the level read before is the level read after (restore worked); the script's own trap restores `auto` on every exit.
2. the level read inside the job right after the write is profile_standard (the write took effect).
3. each run has at least 90 % usable dispatches (item D's record check).
4. steadiness verdict: (max - min) / 2 / median of the 10 run medians at most 0.3 %. Break: synthetic runs with a 10 % swing
   must FAIL it.

The comparison with auto (ratio of the medians, spread of the auto runs) is a finding, not a control.
