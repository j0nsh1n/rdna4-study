# Round 5: overclock bit-exact checks, the engine sampler, FP8 decode power, a vLLM decode gap, and the start clock

Run folder `runs/20261119T235127Z-r5/` (git-ignored), 2026-11-19 23:52Z to 2026-11-20 01:29Z, 590 GPU jobs, each with a
`.meta`, a 20 ms guard and a 20 ms sampler (`watch/<job>/samples.csv`). Every job ran on the LACT profile "Default"
(374 W cap, +150 MHz core, -50 mV). The memory clock read 1258 MHz before and after all 590 jobs (`uclk_before`,
`uclk_after` in each `.meta`: MEASURED). The kernel log held 25 lines matching "GPU reset" or "ring ... timeout" at the
start (`journal-count-start.txt`) and 25 when I counted again while writing this file: no reset during the round.
Items I and K and the ROCm 10.2 install ran later the same night, each in its own run folder (named in its section), and item K
changed the LACT profile and item I the performance level (both restored); the 590 jobs above made no setting change.
Labels as elsewhere: MEASURED (a job is named), READ (a source file or a file of this repository), INFERRED (worked out,
not observed). The round's own results summary (`RESULTS.md` in the run folder) was checked against the logs and the files it names, number by number; where a log and that file
disagreed the log is used and the difference is listed at the end ("Corrections made while writing this up").

## Plain summary

- **The card computed correctly on the overclock profile.** 522 gated jobs of the arithmetic and conversion checks at two
  clock states, plus three whole-card load bursts beside a GPU-resident bit check (3.0 x 10^10 matrix instructions) and
  one 2 s run (2.8 x 10^10), gave zero mismatches, and the results were identical from run to run. This makes the
  undervolt a less likely cause of the card's earlier resets, but it does not exclude it: the checks are exact-value
  checks of short runs, and a sustained heavy load was not tested.
- **The engine-busy sampler reads about 0.04 to 0.13 ms per frame too high.** The offset does not grow with the work in a frame. A
  HIP (compute-only, KFD) program does not appear in it at all. For a game through DXVK or vkd3d the counters work, if
  they are read while the process lives.
- **Real FP8 decode kernel against Q8_0: +47 W at the 374 W cap**, in a cache-resident loop, so a lower bound on the
  demand. FP8 sat at the cap; Q8_0 did not.
- **The vLLM gap between 32 and 128 generated tokens did not reproduce.** In one warmed engine the step time was 6.82 ms
  at every context length; the old gap came from the method or the engine's state, not from context length.
- **The low Vulkan start clock is not what the kernel source says it is.** On "Default" the same probe starts at 2.8 GHz,
  with or without an idle HIP process. What moves the start clock between "Testing" and "Default" is open.
- **The dead profiler counters were the power state, not the ROCm version.** With the performance level set to
  `profile_standard`, 15 counters count on ROCm 10.1 and 14 on 10.2, against 4 on both in `auto` (item I, run later the
  same night). Checked against the known work, 7 and 6 of them are exact, 7 and 7 plausible, and `VALUBusy` is 4 times
  too high as a percent of SIMD cycles. SPM still aborts on 10.2.
- **The PyTorch FP8 decode command that hung twice on the overclock profile ran clean 3 of 3 times on the stock
  profile** (item K). That points at the profile, from small numbers; it does not test the NaNs, and nothing ran on
  PyTorch FP8 under ROCm 10.2 (installed beside 10.1, below).
- **Three thermal stops** (95 C) and **two dropped jobs** are listed plainly below.

## Thermal stops and dropped jobs

Three jobs ended at the 95 C hotspot stop (guard exit 8, job exit 143). No two were in a row without a clean job between
them. All three were whole-card loads started from an idle card with the fan stopped.

| Job | What | Stopped at | Where its log is |
|---|---|---|---|
| `A-cal-heavy` | `mat_card reg:fp8`, a calibration job for item A | 95 C at 1.34 s of the guard | `A-cal-heavy.log`, counted in the table |
| `A-H1-cvt_sr-rule_fp8` | `cvt_sr rule fp8` with the spinner beside it (state H, first rerun) | 95 C at 3.26 s | `superseded/`, not counted |
| `F-test-f8` | first attempt of item F on the unmodified tree, 5 shapes for about 12 s | 95 C at 5.60 s | `F-test-f8.log` (counted) and `superseded/F-test-f8-thermal-stop.log` |

The **two `cvt_sr rule` jobs were dropped from the repetitions**. `cvt_sr rule fp8` and `cvt_sr rule bf8` ran once alone
(job `A-L1-...`: hotspot 88 and 94 C, exit 0, VERDICT PASS, 0 mismatches of 59,259,224,064 and 118,493,282,304
conversions). With the spinner beside it, `cvt_sr rule fp8` hit the stop (second row above), so neither was repeated in
the other 4 runs of state L or in state H. The 36 other conversion jobs were repeated 5 times in each state. `cvt_time`
is a timing probe and was left out of item A. A first set of "state P" jobs with `arith_check random` beside the
bursts was set aside (`superseded/`): that check's GPU time is about 1 s of 20.6 s, so it did not overlap the bursts.

## A. Does the card compute correctly on the overclock profile?

**Question.** The card reset five times on this profile (four during PyTorch FP8 jobs, once with no study job running;
`llm-runtimes.md`). Does it compute wrongly on the profile, at the clocks it reaches?

**What ran.** The unchanged lane binaries of `matrix-arithmetic.md` (16 GPU jobs from `arith_check` and `arith_more`) and
`conversions-and-dots.md` (36 GPU jobs from `cvt_exact`, `cvt_sr`, `cvt_widen`, `cvt_dot`, `cvt_f16`, job lists in
`scripts/round5/arith_jobs.txt` and `conv_jobs.txt`), each as its own gated, guarded, sampled job, 5 repetitions in each
state:

| State | Companion load | Core clock while busy (sampler, busy samples) | Reported power |
|---|---|---|---|
| L | none | arithmetic jobs: 2.81 GHz (p50 and max 2.812); conversion jobs: p50 2.81, max 3.34 | arithmetic at most 64 W; conversions up to 325 W |
| H | `spin`, a one-wave FMA spinner, 30 s beside each arithmetic job and 60 s beside each conversion job | p50 3.434, max 3.455 | arithmetic up to 112 W; conversions up to 369 W |
| P ("heavy bursts") | `mat_card burst f16 --w 16 --ms 200`, 3 bursts per job, beside a GPU-resident bit check (`bitloop`) | 2.74 to 3.43 GHz in the sampler; the bursts' in-kernel clock 3.09 to 3.16 GHz | up to 281 W in a job |

The jobs in state L start at 2.8 GHz and the whole-card conversion jobs in it climbed to 3.34 GHz in a few samples, so
"L" means "started low", not "ran low". Hotspot maxima: L 94 C (`cvt_sr rule bf8`), H 88 C, P 82 C, the 2 s job 92 C.
Memory 1258 MHz throughout.

**Answer (MEASURED).** **Zero mismatches in 522 gated jobs (262 in state L, 260 in state H): every job exited 0 and every
VERDICT line was PASS** (702 PASS lines, 0 FAIL, 0 jobs without a verdict). Between runs, for every job the md5 of all its
RESULT lines is identical across the 5 L runs, the 5 H runs, and between L and H (`anaA.out`: 106 of 106 job rows
identical; two of them are the single-run `cvt_sr rule` jobs); `arith_check determ` reports 0 repeat, wave and row/column differences in all 10 runs.

**State P, `bitloop.hip`** (WMMA f16 or fp8 on exact small integers, expected values from integer arithmetic on the host;
a wave compares all 8 accumulators with the table after every 256-instruction chunk):

| Job | Bit check | Chunks | WMMA | Mismatches | Hotspot max | Bursts' power max |
|---|---|---|---|---|---|---|
| `A-P1-bitloop_f16` | f16, 16 waves | 40,216,004 | 1.03 x 10^10 | 0 | 77 C | 271, 234, 236 W |
| `A-P2-bitloop_fp8` | fp8, 16 waves | 37,234,518 | 9.53 x 10^9 | 0 | 78 C | 244, 257, 257 W |
| `A-P3-bitloop_f16` | f16, 16 waves | 40,218,358 | 1.03 x 10^10 | 0 | 82 C | 278, 236, 237 W |
| `A-Pb1-bitloop_f16_w4_2s` | f16, 512 waves (4 per SIMD) for 2 s, no bursts | 109,464,043 | 2.8 x 10^10 | 0 | 92 C | sampler max 307 W; core 2.71 to 3.39 GHz, p50 3.28 |

In `A-P2` the bursts' own placement check printed FAIL (the bit check's fp8 waves share the SIMDs, so the burst saw 96 SIMDs
and 16 to 32 waves per SIMD, half the f16 rate) but the load ran and the bit check passed. The 4 s version of
`A-Pb1` was not run: the 2 s job peaked above 85 C, which was this round's rule for not going longer.

**Known-answer breaks (each must FAIL, each did, exit 1; 9 jobs).** `arith_check random --n 200000 --break` (job
`A-break-random`), `arith_check determ --break`, `arith_check special --break`, `arith_more chain8 --break`,
`cvt_exact encode fp8 plain --break` (256 mismatches), `cvt_sr sweep fp8 plain 4 --break` (167,939,720),
`cvt_widen --break` (67,108,864), and `bitloop f16 --break` and `bitloop fp8 --break` (first bad value 48928 against
48929). The two clean bit-loop controls `bl-f16` and `bl-fp8` passed (0 mismatches).

**Open.** The checks are exact-value checks. The card's resets and PyTorch's four hangs are not shown to be arithmetic
errors, nor shown not to be. Heavy bursts were 0.2 s and 2 s; a sustained heavy load was not tested (it reaches the guard
within about a second from an idle start). So the undervolt is less likely, not excluded, as a cause of the resets.

## B. Calibrating the outside-view tool (`scripts/round5/engine_sampler.py`, 0.2 s period)

**Question.** The engine sampler reads `drm-engine-*` busy time from `/proc/<pid>/fdinfo` for each process that holds the
card. How far does it agree with a load whose GPU time is known from the probe's own timestamps?

**What ran.** `vkgame burst` (dense FMA, 64 groups of 128, a 60 Hz cadence, 1 s and 60 bursts per arm; source in
`src/vk`, built unchanged) with the sampler beside it (`scripts/round5/bwrap.sh`), once on the graphics queue and once on a
compute-only queue (async compute). The probe's time per frame is (bottom minus top timestamp) x 10 ns from its own CSV
(`B/*-bursts.csv`): 2.803 to 2.812 ms with 2.8 ms set and 1.005 to 1.011 ms with 1.0 ms set. The sampler's busy time per
frame is (counter at the plateau after an arm minus the plateau before it) / 60 (`scripts/round5/bana.py`). The first
arm is not used (the first sample comes after it started).

**Answer (MEASURED).** The sampler reads a **fixed offset above the probe: +0.04 to +0.10 ms per frame on the graphics
engine and +0.08 to +0.13 ms on the compute engine**, about the same for 2.8 ms and 1.0 ms of work per frame
(graphics: +0.04 to +0.07 ms at 2.8 ms, +0.09 to +0.10 ms at 1.0 ms).

| Queue, work per frame | Jobs | Sampler minus probe, per arm (ms) |
|---|---|---|
| graphics, 2.8 ms | `B2-g-1..3`, 6 arms | +0.058, +0.041, +0.065, +0.070, +0.064, +0.055 |
| graphics, 2.8 ms (first test) | `B-g-1`, 1 arm | +0.074 |
| graphics, 1.0 ms | `B3-g-1ms`, 2 arms | +0.097, +0.089 |
| compute, 2.8 ms | `B2-c-1..3`, 9 arms | +0.122, +0.103, +0.110, +0.086, +0.129, +0.082, +0.079, +0.089, +0.087 |
| compute, 1.0 ms | `B3-c-1ms`, 3 arms | +0.132, +0.127, +0.110 |

The analysis script's own limit is 0.1 ms per arm. Every graphics arm is inside it. On the compute queue 3 of the 4 jobs
(`B2-c-1`, `B2-c-2`, `B3-c-1ms`) print VERDICT FAIL because at least one arm is over 0.1 ms (largest +0.132); `B2-c-3`
passes. So the offset is "within 0.1 ms on graphics, at or a little over 0.1 ms on compute", not "within 0.1 ms
everywhere".

**What it covers.**
- A Vulkan graphics-queue process shows its work as `drm-engine-gfx`; a compute-only-queue process as
  `drm-engine-compute`. Desktop processes (the compositor, Xwayland, the browser, and so on) show only `gfx`.
- **A HIP (KFD) process does not appear in the sampler at all** (`B4-hip`: `bitloop` ran about 2 s, 30 sampler
  periods, no row for it): the sampler reads only fdinfo files that carry a `drm-pdev` line. A game through DXVK or
  vkd3d-proton is a Vulkan client and is covered.
- Counters vanish when the process exits (the last arm's busy time is lost), so the counter must be read while the
  process lives and at a plateau.
- Deliberate break: the same analysis on the wrong PID (the compositor's) FAILs ("no complete arm"; checked again when
  writing this up).

**What this means for tonight's FSR 4 measurement** (`fsr4-predictions.md` is not edited; its commit time is its
evidence). A game frame measured through the sampler's busy counter carries a fixed offset of about 0.04 to 0.13 ms per frame
on top of the true GPU time, so a window of N frames carries about N x (0.04 to 0.13) ms of overhead. Against AMD's
352 us (1080p) and 1316 us (4K) that is 3 % of the 4K pass at the least and 37 % of the 1080p pass at the most, so a difference of FSR 4 on against off should be read
from the counter, not from the sampler's per-0.2-s differences, and the offset should be subtracted per frame, not per
window. Any HIP or KFD work (a compute-only tool run beside the game, for instance) will not appear in the per-process DRM
counters, so it is invisible to this method; the card-level `gpu_busy_percent` and power still see it.

**Open.** The cause of the offset (the ring's job start-to-finish time against the shader's own timestamps) is INFERRED,
not tested. Whether queueing behind the compositor adds to it in a real game is not measured. The test of the game route
itself was not run: its staging folder (the Wine prefix and `gr.sh`) was not on this disk.

## F. Real FP8 against Q8_0 decode kernels, kernel only

**Question.** `llm-workload-3.md` Q1 left open how much of the real 57 W extra draw of FP8 decode the instruction mix
accounts for; the matched probes drew 30 to 33 W more under conditions unlike real decode. Does the real (patched) decode
kernel loop, run alone, show the extra draw?

**What ran.** `test-backend-ops perf -b ROCm0 -o MUL_MAT` on the patched llama.cpp tree (`llama.cpp-latest-patched`, whose
FP8 `mul_mat_vec_f` kernel is the one real decode uses), case `type_a=<f8_e4m3 or q8_0>,type_b=f32,m=4096,n=1,k=14336`
(the 8B down projection), 6 interleaved pairs (`Fp-01..12`), each pair of one FP8 and one Q8_0 job, power averaged over
busy samples (at least 90 % busy, over 120 W) after the first 0.3 s (`fana.py`). Profile "Default", memory 1258 MHz.

**Answer (MEASURED).** **The real FP8 decode kernel loop draws +45.3 to +52.4 W (mean +47.4 W) more than Q8_0 and sits at
the power cap; the gap is a lower bound on demand.**

| Pair | FP8 W (MHz) | Q8_0 W (MHz) | Difference W |
|---|---|---|---|
| 1 | 373.4 (2843) | 325.8 (3221) | +47.7 |
| 2 | 373.4 (2830) | 328.1 (3219) | +45.3 |
| 3 | 375.0 (2828) | 326.8 (3219) | +48.2 |
| 4 | 372.6 (2830) | 326.9 (3222) | +45.7 |
| 5 | 373.2 (2829) | 328.0 (3220) | +45.3 |
| 6 | 375.0 (2830) | 322.6 (3222) | +52.4 |

FP8 sat at the 374 W cap: in `Fp-01` the power-limit bit (PPT0, bit 0 of `indep_throttle_status`) was set in 42 of 52 busy
samples, and the clock was 2.83 GHz against Q8_0's 3.22 GHz. Q8_0 never set a flag (0 of 49 in `Fp-02`).
The kernel alone, without the rest of a token, already reaches the cap.

**Caveat 1: cache-resident.** The weights (58.7 MB for FP8, 62.4 MB for Q8_0, worked out from 4096 x 14336 elements) fit in
the 64 MB cache, and the loop is not DRAM-bound (34.4 us per run against 40.8 us in `Fp-01` and `Fp-02`), so both draw more
than real decode does (294 W for FP8 and 240 W for Q8_0 as medians in `llm-runtimes.md` R4; the real decode streams
7.5 GB per token). The difference is a lower bound, not an estimate of the real one, because FP8 is held at the cap and
its uncapped draw is higher by an amount that was not measured.

**Caveat 2: the "round 2 test tree" is the unmodified llama.cpp.** `llama.cpp-llm2-shapes` has llama.cpp's stock FP8 path
(dequantise first), not the patched kernel. On its 8B gate/up shape (`F-01..12`, m = 14336, k = 4096) FP8 is slower (187.5 us
against 50.1 us) and draws **less** than Q8_0: 36 to 41 W less (mean -37.6 W; 286 W against 323 W, no throttle flag).
That is not the real kernel and does not bear on decode. The first attempt (5 shapes for about 12 s, `F-test-f8`) hit 95 C
at 5.6 s: the n = 1 loops heat the hotspot by about 30 C in 0.3 s from an idle start, so each later job is one 1 s case.

**Across profiles (READ from `llm-workload-3.md` and `llm-runtimes.md`).** Rounds 1 and 3, profile "Testing", 317 W cap: 8B
patched FP8 decode at the cap (PPT0 in 1007 of 1157 samples), 316 against 259 to 260 W for Q8_0, a gap of 56 to 57 W (a lower
bound). Round 4, profile "Default", 374 W cap: the same decode, 294 against 240 W (medians), no throttle flag, "about 54 W". So
the extra draw is real on both profiles; round 5's loop shows the kernel alone reaches the 374 W cap.

**Open.** A DRAM-streaming kernel-only loop for the real kernel: it needs a patched-tree test build with the shapes patch
(m = 128256, 525 MB of weights), not built, because it is a long compile while the machine is in use. The shapes patch exists
only for the unmodified tree.

## H. vLLM decode at 32 against 128 generated tokens (3B FP8, one warmed engine)

**Question.** Round 4 (`llm-runtimes.md` R2) measured one vLLM stream at 111.7 tokens/s with 128 generated tokens and
145.7 with 32. Is that a context-length effect?

**What ran.** `h_vllm.py` (in `scripts/round5/`), vLLM in one engine that stays warm, 4 request kinds interleaved, 8 rounds,
2 runs (`H-vllm-1`, `H-vllm-2`). Kind A: 128-token prompt and 32 new (KV 129 to 160). Kind B: 128 and 128 (KV 129 to 256). Kind
C: 224 and 32 (KV 225 to 256, the same KV range as B's last 32 steps). Kind D: 256 and 32 (KV 257 to 288). The host step time
is one engine step per token timed on the host; "old method" is (N-token request time minus 1-token request time) / N.

**Answer (MEASURED).** **Not explained by context length, and not reproduced.** The host step time was 6.82 ms (146.5
tokens/s) at every context length; the two runs agree to 0.1 %.

| Kind | Host step time (run 1 / run 2) ms | Old method tokens/s (run 1 / run 2) |
|---|---|---|
| A, 128 + 32 | 6.821 / 6.820 | 151.5 / 151.6 |
| B, 128 + 128 | 6.825 / 6.824 | 147.7 / 147.7 |
| C, 224 + 32 | 6.827 / 6.825 | 151.4 / 151.4 |
| D, 256 + 32 | 6.831 / 6.830 | 151.4 / 151.4 |

From A to D (128 more tokens of KV) the step time rose by 0.01 ms, 0.15 %. The 2.5 % gap between A and B in the old method
is the method itself: N tokens counted over N - 1 extra steps (32 / 31 against 128 / 127), not the engine.

**Instrument check.** A deliberate delay inside the timed step loop (run 1: 5 ms; run 2: 20 ms). The 5 ms delay added
-0.01 ms (hidden behind the GPU step); the 20 ms delay raised the step time from 6.87 to 21.05 ms (+14.2 ms). So the timer sees
host delays larger than the GPU step.

**Conditions.** Core clock p50 3.42 GHz while busy, reported power median 162 to 165 W (peak samples around 420 W in
the engine's start-up), profile "Default", memory 1258 MHz.

**What it says about round 4.** Round 4's 111.7 and 145.7 ran at 1258 MHz as well (`llm-runtimes.md` R2: "at the same
clock"; its 158.7 was at 1408 MHz), so the memory clock does not explain the difference. Neither number appears in round 5's
engine, and the 23 % gap between them is not reproduced. INFERRED: the old gap came from the state of that run's engine or
the requests before the timed ones (its decode loop followed prefill tests in the same process), not from context length.
Not tested: that protocol itself (`llm4_vllm.py bench` was not rerun).

## G. Why HIP work starts near 2.7 GHz and Vulkan near 1.7 GHz

**Question.** `gaming-probes.md` E1 found the same loop starting at 2.7 to 2.8 GHz through HIP and 1.7 to 1.8 GHz through
Vulkan, on "Testing". Does the driver ask for a different power setting (a workload profile) for the two paths, and does an
open HIP process raise the Vulkan start clock?

### Kernel source reading (READ)

Upstream: <https://github.com/torvalds/linux>, tag **v7.3**, files under `drivers/gpu/drm/amd/`. The machine runs 7.3.4;
the distribution's patches were not checked, so the lines below are the upstream v7.3 lines. The files were read as local
copies kept in the run folder (`src/`); they are not copied into this repository. Links have the form
`https://github.com/torvalds/linux/blob/v7.3/drivers/gpu/drm/amd/<file>#L<line>`.

1. **KFD (HIP) asks for COMPUTE when a process creates its first queue, not when it binds.** `pqm_create_queue()`,
   `amdkfd/kfd_process_queue_manager.c` lines 381 to 383 ("Register process if this is the first queue") calls
   `dqm->ops.register_process`, which is `register_process()`, `amdkfd/kfd_device_queue_manager.c` line 1653, which calls
   `kfd_inc_compute_active()` at line 1687. `kfd_inc_compute_active()`, `amdkfd/kfd_device.c` lines 1474 to 1478, counts
   `compute_profile` and the first user calls `amdgpu_amdkfd_set_compute_idle(adev, false)`.
   `amdgpu_amdkfd_set_compute_idle()`, `amdgpu/amdgpu_amdkfd.c` lines 751 to 770, for graphics IP version 12 calls
   `amdgpu_gfx_off_ctrl(adev, idle)` (GFXOFF off) at line 758 and at line 768
   `amdgpu_dpm_switch_power_profile(adev, PP_SMC_POWER_PROFILE_COMPUTE, !idle)`. **There is no overdrive check on this
   path.** Release: `unregister_process()`, `kfd_device_queue_manager.c` line 1692 (the decrement at line 1720), from
   `pqm_destroy_queue()` (`kfd_process_queue_manager.c` line 518; the call at line 583).
2. **The graphics and compute rings (Vulkan, desktop) ask for a profile in `amdgpu_gfx_profile_ring_begin_use()`,
   `amdgpu/amdgpu_gfx.c` line 2791** (FULLSCREEN3D when the device has a graphics ring, else COMPUTE, lines 2801 and 2803;
   released one second after the last submission by `amdgpu_gfx_profile_idle_work_handler()`, line 2758). It is called from
   `gfx_v12_0_ring_begin_use()` (`amdgpu/gfx_v12_0.c` lines 5325 to 5329, hooked at lines 5401 and 5439). **It returns at once
   at line 2797, `if (amdgpu_dpm_is_overdrive_enabled(adev)) return;`**, and the same test is at line 2831 in
   `amdgpu_gfx_profile_ring_end_use()`. `amdgpu_dpm_is_overdrive_enabled()`, `pm/amdgpu_dpm.c` lines 1698 to 1703, returns
   `smu->od_enabled`, set at `pm/swsmu/amdgpu_smu.c` lines 760 to 761 from `adev->pm.pp_feature & PP_OVERDRIVE_MASK` (0x4000,
   `include/amd_shared.h`, line 230 as the results summary has it; that header was not among the source copies re-read for this file), which is
   the amdgpu `ppfeaturemask` module parameter at boot. On this machine
   `/sys/module/amdgpu/parameters/ppfeaturemask` reads `0xfff7ffff` (MEASURED, read again when writing this up), so bit
   0x4000 is set. INFERRED: with this boot setting the kernel never requests a workload profile for Vulkan or desktop
   submissions, on any LACT profile (the flag does not change with the profile), while HIP's KFD path does.
3. `smu_switch_power_profile()`, `pm/swsmu/amdgpu_smu.c` line 2525, counts users per profile; `smu_bump_power_profile_mode()`,
   line 2394, ORs the counted profiles into `workload_mask` and the driver for this chip, `smu_v14_0_2_set_power_profile_mode()`,
   `pm/swsmu/smu14/smu_v14_0_2_ppt.c` line 1847, sends `SMU_MSG_SetWorkloadMask` (line 1893) and turns deep sleep off when
   COMPUTE is in the mask (lines 1859 to 1862). It is ignored when the forced performance level is MANUAL or PERF_DETERMINISM
   (`auto` here).
4. **`pp_power_profile_mode` shows only the user's choice.** The `*` marks `smu->power_profile_mode`
   (`amdgpu_smu.c` lines 1359 to 1362, changed in the write handler near line 3310); the counted mask is not printed. READ here:
   `power_profile_mode_active: 0:BOOTUP_DEFAULT` in the headers at the start and end of every watched job (594 headers,
   HIP jobs among them), and `power_profiles_seen=0x1` from the `vkgame` G jobs, which read the same file during the job. So the
   sentence in `gaming-probes.md` E1 that the profile shown did not change is not evidence that the active workload did not.

### Ramp test (MEASURED)

`G-t-alone-1..5` and `G-t-hold-1..5`, interleaved: `vkgame burst` with one wave, one 60 ms dispatch after 1 s idle, on the
graphics queue (`name=target,q=g,groups=1,local=32,cont=60,idle=1`), read by `scripts/gprobe_dense.py`. "hold" is the same
with `hip_hold` (`src/round5/hip_hold.hip`) open: an idle HIP process with a compute queue (`G-kfdlist` shows it listed as a
KFD process in `/sys/class/kfd/kfd/proc` while it ran, and none before).

**The Vulkan ramp starts at 2804 to 2805 MHz in the first window with and without the idle HIP process, steps up at
34 ms, reaches 95 % of the final clock at 35 ms and ends at 3232 to 3234 MHz, in all 10 jobs** (in-kernel clock from the
cycle counter against the 100 MHz realtime counter, 0.25 ms windows; the sysfs `sclk` column reached 3060 at 2 ms and 3233 at 8 ms in
`G-t-alone-1`, as it averages). So the ramp does not start higher with HIP open, because it already starts at 2.8 GHz
alone on "Default".

**The 1.7 to 1.8 GHz Vulkan start in `gaming-probes.md` was measured on "Testing". On "Default" the same probe starts at
2.8 GHz.** The source reading does not explain that difference: the overdrive flag is the same under both LACT profiles. So
**the low Vulkan start is not shown to come from the kernel's workload-profile requests, and what differs between "Testing"
and "Default" that moves the start clock is open.** Candidates, untested: the +150 MHz offset or the -50 mV offset changing
the firmware's starting level; the 317 W against the 374 W cap; a LACT setting that is not visible in sysfs. No profile switch
was made in this round. Memory clock 1258 MHz in all jobs.

## I. Profiler counters: ROCm 10.1 against 10.2, in `auto` and in `profile_standard`

**Question.** Since round 2, 10 of the 13 counter sets read 0 on this card (`llm-workload-2.md` item 4,
`llm-workload-3.md` Q4). Do they start counting with ROCm 10.2's profiler, or in a fixed power state?

**What ran** (run folder `runs/20261120T014728Z-r5i/`, 2026-11-20 01:49Z to 02:08Z, 10 gated jobs per state (4
programs and 1 SPM try for each version); LACT profile "Default" in every job, read before and after each one in `out/jobs-<state>.tsv`). Two
known-answer programs, each built with the matching version's `hipcc` and run with that version's `rocprofv3`
(1.4.1 for 10.1, 1.4.2 for 10.2): `valu_loop` (8192 waves, each running 4096 or 8192 trips of 8 dependent
`v_fma_f32`) and the bandwidth probe's `read_kernel` (128 and 256 MiB, 16-byte loads). 13 counter sets, one `--pmc`
group per pass. The lead set `power_dpm_force_performance_level` to `profile_standard` (read back), the script
`counters.sh profile_standard` ran, and the level was set back to `auto` (read back; the script itself refuses to
run when the level is not the one it is told; the saved logs end before the restore, which is the lead's report). A counter "works" in the run's table if it is non-zero and scales
with the work (a doubled run reads 1.7 to 2.3 times as much).

**Answer (MEASURED).** **The dead counters were the power state, not the ROCm version.**

| Level | ROCm 10.1 | ROCm 10.2 |
|---|---|---|
| `auto` | 4 work, 14 read 0 | the same 4 work, the same 14 read 0 |
| `profile_standard` | 15 work, 0 read 0 | 14 work, 0 read 0 |

In `auto` the working four are `GRBM_COUNT`, `GRBM_GUI_ACTIVE`, `SQ_BUSY_CYCLES`, `SIMD_UTILIZATION`, as in round 2;
10.2's profiler adds nothing there. In `profile_standard` the SQ instruction, L2, TCP and memory counters count
on both versions. Before each job the card's clock read 1.64 to 1.65 GHz at 82 to 84 W in `profile_standard` (`.meta`),
against 120 to 190 MHz idle in `auto`. **Which part of the state matters** (the pinned clock, power gating
off, something else) **is not tested** (INFERRED that it is a power-management effect). Each state was run once,
`auto` first, with no repeat.

### Audit: are the values right, or only non-zero?

"Works" in the table means non-zero and scaling. I checked each counter judged WORKS in `profile_standard` against
what the work must produce. Rule: **right** = within 3 % of a figure worked out from the program, or a clean
dispatch equal to it; **plausible** = non-zero, scales, inside a limit, but no exact known answer; **wrong** =
contradicts the known answer. Nothing was run for this: it uses the saved CSVs (`out/*/*/pass_*/`), the programs'
sources and a disassembly of `read_kernel` built with the 10.1 compiler (CPU only; the loop has 7 vector and 16 total
instructions per trip, 9 vector instructions outside it).

Known answers. `valu_loop`: 8192 waves x 4096 x 8 = 2.684e8 vector FMAs (A), 5.369e8 (B); 8192 x 4096 x 11 = 3.691e8
instructions in all (8 FMAs plus 3 loop-control instructions per trip, INFERRED from the source). `read_kernel`:
128 MiB = 8,388,608 16-byte loads = 262,144 wave loads of 512 B; 32 trips per thread (64 at 256 MiB), so per wave
9 + 7 x 32 = 233 vector instructions (457 at 256 MiB) and 16 x 32 + 35 = 547 instructions in all (1059).

| Counter | ROCm | Value (valu A / read 128 MiB) | Known answer | Verdict |
|---|---|---|---|---|
| `GRBM_COUNT`, `GRBM_GUI_ACTIVE` | both | 8.91e6 / 9.1e5 cycles | cycles divided by the dispatch's own start-to-end time are 1.628 to 1.693 GHz (all 44 dispatches checked) against 1.642 to 1.647 GHz logged: -1 % to +3 % | right |
| `SQ_INSTS_VALU` | 10.1 | 2.685e8 / min 1.90874e6 | A: 2.684e8 (+0.02 %); read: 233 x 8192 = 1,908,736 and 457 x 8192 = 3,743,744, both equal to the lowest dispatch | right (see note) |
| `SQ_WAVE32_INSTS` | both | 3.692e8 / 4.48102e6 | A: 3.691e8 (+0.03 %); B 7.383e8 against 7.381e8; read: 547 x 8192 = 4,481,024 and 1059 x 8192 = 8,675,328, exact in all 16 dispatches | right |
| `FETCH_SIZE` | both | 131072 KiB | 128 MiB = 131072 KiB (262144 at 256 MiB, both exact) | right |
| `TCP_REQ`, `TCP_REQ_MISS` | both | 1,048,576 | 262,144 wave loads x 4 lines of 128 B = 1,048,576 (2,097,152 at 256 MiB); nothing is read twice, so every request misses | right |
| `SQ_BUSY_CYCLES` | both | 2.82e8 | 28.8 to 38.5 times `GRBM_COUNT` (31.7 on A); the card's 32 workgroup processors, if it sums one per processor (INFERRED) | plausible |
| `SQ_WAVE_CYCLES` | both | 1.56e10 | divided by `GRBM_COUNT`: 1683 to 2035 waves resident on average against a ceiling of 128 SIMDs x 16 = 2048 slots (INFERRED limit), except two dispatches at 2283 and 3736 | plausible |
| `SQ_INST_CYCLES_VALU` | 10.1 | 2.685e8 / 2.21e6 | A: equals the instruction count (one cycle each); read: 270 per wave, 8 per trip against 7 instructions (a 64-bit compare may count twice, INFERRED) | plausible |
| `SQ_INST_CYCLES_VMEM` | 10.2 | 524,288 | exactly 2 per wave load (262,144); the unit cannot be checked | plausible |
| `GL2C_EA_RDREQ_sum`, `GL2C_MISS_sum` | both | 524,289 | the analysis expected 1.0e6 to 2.1e6 (64 to 128 B requests); the value is exactly half of 128 MiB / 128 B, i.e. 256 B per request, which `FETCH_SIZE` (exact) supports; hits plus misses (524,319 + 524,289) equal `TCP_REQ`, which would explain the 50.00 % `L2CacheHit` on a streaming read. The request size is not independently known | plausible |
| `MemUnitBusy` | both | 94.9 to 97.7 % | profiled `read_kernel` moves 128 MiB in 0.54 ms (about 0.25 TB/s); no known answer | plausible |
| `SIMD_UTILIZATION` | both | 0.9985 / 0.9999 | reads about 1.0 on a vector-bound and on a memory-bound kernel alike, so it is not a vector-issue fraction; no known answer | plausible |
| `VALUBusy` | both | 94.1 % / 7.6 % | A: 2.684e8 / 128 SIMDs = 2.097e6 vector instructions per SIMD in 8.909e6 cycles = 23.5 % of SIMD cycles; reported 94.1 % is 4.0 times that. It equals `SQ_INST_CYCLES_VALU` / (32 x `GRBM_GUI_ACTIVE`) to within 0.1 point, i.e. a per-processor ratio, not a SIMD issue fraction (read: 1.6 % against 7.6 %) | wrong as a percent of SIMD cycles |

**Count of the 15 (10.1) and 14 (10.2) counters the table judged WORKS: 10.1 right 7, plausible 7, wrong 1; 10.2
right 6, plausible 7, wrong 1.** Not one of them is nonsense, but "works" overstates it for `VALUBusy`, and for the
plausible ones it means "responds", not "correct".

What the audit found beyond the count (MEASURED unless marked):

- **The table's medians are contaminated and cost it some verdicts.** Over the 8 dispatches of a 128 MiB run,
  `SQ_INSTS_VALU` ranges from the exact 1.90874e6 up to 1.5e7; the run's table takes the median (2.16e6 on 10.1,
  4.94e6 on 10.2) and so shows 10.2's `SQ_INSTS_VALU` (and `SQ_INSTS_WAVE32_VALU` on both) as NONSCALING when the
  clean dispatches are exact (also 3.74374e6 at 256 MiB). The same happens to `SQ_WAVES`: exact (8192) in clean
  dispatches, but a median of 14,797 (10.1) and 333,578 (10.2) on `valu_loop` A. Counters that must read 0 also show
  stray values (`TCP_REQ` 7.18e6 on 10.2's `valu_loop` A, which loads nothing; `SQ_INSTS_TEX_STORE` 30). INFERRED: the
  counters also pick up work that is not this program's (the desktop, the runtime's own kernels); not tested.
- **`GL2C_EA_RDREQ_128B_sum` is wrong or measures a class this kernel does not produce.** It reads 0 to 28 where
  128 B requests would give 1.05e6; the table already calls it PARTIAL, and it was not counted above.
- **The profiled kernels are slower than the card can run them.** `valu_loop` took 4.25 cycles per vector
  instruction per SIMD (8.909e6 cycles for 2.097e6), against 1.00 for independent FMAs and 5.00 for one dependent
  chain (`execution-core.md`); with about 14 waves resident per SIMD a dependent chain should issue about once per
  cycle. This is not explained; the counts are exact, so it concerns time, not the counters. `read_kernel` ran at
  0.2 to 0.25 TB/s under the profiler against 0.64 to 0.68 TB/s in the plain probe (`llm-workload.md`). Percent-busy
  counters describe the profiled run.
- Not audited: `SQ_INSTS_FLAT`, `SQ_INSTS_LDS`, `GL2C_EA_WRREQ_*` (0 where 0 is expected; they prove nothing about
  whether the counters count).

### What still does not work (MEASURED)

- **SPM** (the sampled counter mode that ROCm 10.2's notes list for gfx1201): on 10.2 the tool aborts at once with
  "Invalid input parameter" (`tool.cpp:2212`, exit 134), in `auto` and in `profile_standard`, with no output; earlier
  probes with `sclk_cycles`, two counters and a smaller interval did the same (3 variants, then stopped at the
  debugging limit). On 10.1 the argument parser rejects `refclk_cycles` ("choose from 'sclk_cycles'", exit 2); that
  option is what 10.2 adds.
- The 10.0 profiler's abort (`llm-workload-3.md` Q4) was not revisited.

## K. PyTorch FP8 decode on the stock profile (the command of hang 2)

**Question.** Hangs 2 and 3 of round 4 (`llm-runtimes.md`, "GPU hangs") were both 4-stream decode runs of the 3B FP8
model on "Default" (+150 MHz, -50 mV); a fifth reset had a desktop app as the guilty process on the same profile. Does
the same command hang on the stock profile?

**What ran** (`runs/20261120T020831Z-r5k/`, 2026-11-20 02:08Z to 02:13Z). The exact command of `r4t-tg-3b-fp8-a`
(`llm4_torch.py bench`, same venv with torch 2.15.0 + ROCm 10.0, `--fp8 --prefill --streams 1 2 3 4 --reps 3 --steps
128`) three times, each as a gated, guarded job, 60 s apart, on the LACT profile "Testing" (stock; its power cap was
317 W by item G above, not recorded in these logs). The script stopped at the first failure or reset; the profile
was set back to "Default" afterwards (read back).

**Answer (MEASURED).** **Three clean runs of three: exit 0, no thermal stop, no guard loss, and the kernel log's reset
count stayed at 5 before and after each run** (`K-SUMMARY.txt`). Medians of 3 repetitions, tokens/s (milliseconds per
step) at 1, 2, 3, 4 streams:

| Run | 1 | 2 | 3 | 4 | Hotspot max |
|---|---|---|---|---|---|
| `k-...-stock-1` | 57.9 (17.26) | 115.9 (17.25) | 161.2 (18.61) | 223.0 (17.93) | 70 C |
| `k-...-stock-2` | 57.7 (17.32) | 115.2 (17.36) | 166.3 (18.04) | 223.3 (17.91) | 68 C |
| `k-...-stock-3` | 58.2 (17.19) | 113.6 (17.61) | 164.1 (18.28) | 222.4 (17.99) | 69 C |

Hang 2's own job printed 59.8, 117.0 and 171.8 tokens/s at 1, 2 and 3 streams on "Default" before it hung at the
fourth, so stock is about 3 % lower at 1 stream and 3 to 6 % at 3 (one Default run, so a rough ratio only). Every
`.meta` has the memory clock at 1258 MHz before and after, as in all of round 5.

**What this shows.** On "Default" the 4-stream decode of this command hung in 2 of the 2 runs of it (hang 2 in the same
venv; hang 3 on another library stack, torch 2.13 + ROCm 10.1, with streams up to 8). On stock it ran clean 3 of 3.
That **implicates the overclock and undervolt profile** (INFERRED; the numbers are small: 3 and 2 runs).

**What it does not show.** It does not test the NaN results or the run-to-run differences of round 4: this command
runs no perplexity and no outputs were compared. Its `prefill` field was empty (`"prefill": {}`), so the prompt
path that hang 1 and hang 4 died in was not run here either. Nothing was run on ROCm 10.2's PyTorch FP8. The hangs
remain unexplained: three clean runs say a stock card does not hang on this command, not why a tuned one does.

## ROCm 10.2 installed beside 10.1

Record: `runs/20261120T013902Z-rocm102/INSTALL.md`, 2026-11-19. Official sources only: the ROCm 10.2.0 tarball (HIP
7.17, 11 GB, in the home folder; the `rocm` link still points at 10.1), a PyTorch venv (torch 2.15.0+rocm10.2.0,
6.5 GB) and a vLLM venv (AMD's vllm 0.32.1.dev0 build with torch 2.14.0+rocm10.2.0, 14 GB). Nothing in the 10.1
folders was touched. AMD's table pairs vLLM 0.32 with torch 2.14, and its vllm wheel pins a triton that torch 2.14's
build contradicts, so the venv uses an override: a version mix AMD did not state.

- **Works (MEASURED, `rocminfo102`, `hello102`, `torch102`):** `rocminfo` lists the card as gfx1201; a HIP program built
  with the 10.2 `hipcc` ran (1024 values correct); torch sees one device and a bf16 matmul matched the CPU (maximum error
  0.25, bf16 rounding).
- **vLLM did not detect the card at first.** vLLM finds ROCm through `amdsmi`; after `import torch` in the 10.2 venv
  `amdsmi` saw 0 devices (the 10.1 venv sees 2 in the same order). Two fixes failed (the debugging limit). The lead then
  added `amdsmi` from PyPI to the 10.2 vLLM venv (INSTALL.md names 27.2.0 as the candidate; the job log does not print the
  installed version): job `vllm102-amdsmi` printed `torch ... devices 1`, `amdsmi handles 2`, `vllm platform RocmPlatform
  is_rocm True`, exit 0.
- **No model ran on 10.2 and no FP8 job in PyTorch on 10.2.** The counter runs (item I) used 10.2's `rocprofv3` on
  small HIP programs only.

## Table of GPU jobs (from the `.meta` files)

| Group | Jobs | Exit 0 | Exit 1 | Peak hotspot | Thermal stops |
|---|---|---|---|---|---|
| A state L (arithmetic 80, conversions 182) | 262 | 262 | 0 | 94 C (`cvt_sr rule bf8`) | 0 |
| A state H (arithmetic 80, conversions 180) | 260 | 260 | 0 | 88 C | 0 |
| A state P (3 burst jobs, 1 two-second job) | 4 | 4 | 0 | 92 C (`A-Pb1`) | 0 |
| A calibration (`A-cal-spin`, `A-cal-heavy`) and `C-test-enc-fp8-plain` | 3 | 2 | 0 | 95 C | 1 (`A-cal-heavy`, exit 143) |
| deliberate breaks (exit 1 means FAIL as intended; includes `bl-brk-*`) | 9 | 0 | 9 | 78 C | 0 |
| bit-loop checks `bl-f16`, `bl-fp8` | 2 | 2 | 0 | 56 C | 0 |
| B (vkgame and sampler, HIP) | 10 | 10 | 0 | 59 C | 0 |
| F patched tree (`Fp-test` and 12) | 13 | 13 | 0 | 81 C | 0 |
| F unmodified tree (`F-test-f8` and 12) | 13 | 12 | 0 | 95 C | 1 (`F-test-f8`, exit 143) |
| H (vLLM, 2 runs) | 2 | 2 | 0 | 78 C | 0 |
| G (10 ramps, KFD list, calibration) | 12 | 12 | 0 | 51 C | 0 |
| total | 590 | 579 | 9 | | 2 |

The `A-H1-cvt_sr-rule_fp8` stop is in `superseded/` and is not in this table (590 jobs with a `.meta` counted by
`ls *.meta`, which leaves out `superseded/`). The `.meta` count of exit 143 is 2; a third stop is in `superseded/`.

## The round's own copies of the probes and scripts

`src/round5/` and `scripts/round5/` hold **this round's own copies** of the programs and wrappers the round used, with
private paths replaced by environment variables or repository-relative paths. They are not the run folder's files byte
for byte (only the paths changed) and they build and run like the rest of `src/` and `scripts/`.

| File | What it is |
|---|---|
| `src/round5/spin.hip` | The one-wave FMA spinner (state H companion load); prints the in-kernel clock. |
| `src/round5/bitloop.hip` | The GPU-resident bit check for state P (`bitloop <f16\|fp8> <seconds> <waves> [--break]`). |
| `src/round5/hip_hold.hip` | The idle HIP process with a compute queue (item G). |
| `scripts/round5/g.sh` | One gated, guarded, watched GPU job (calls `scripts/study-llm.sh`). |
| `scripts/round5/runA.sh`, `runC.sh`, `arith_jobs.txt`, `conv_jobs.txt` | The arithmetic and conversion job lists, states L and H. |
| `scripts/round5/with_spin.sh`, `with_hold.sh`, `pwrap.sh` | Companion load wrappers (spinner, idle HIP process, bursts beside the bit check). |
| `scripts/round5/bwrap.sh`, `bana.py`, `engine_sampler.py` | Item B: the sampler beside a job, the per-arm analysis, and the sampler (unchanged). |
| `scripts/round5/fwrap.sh`, `fwrap2.sh`, `fana.py` | Item F: one `test-backend-ops` case, unmodified and patched tree, and the paired analysis. |
| `scripts/round5/anaA.py`, `h_vllm.py` | Item A's table and item H's vLLM timing script. |

Set `R_RUN` to the run folder (it must hold `bin/` with the built programs, and `watch/`), `REPO` is found from the script
location, `HOTSPOT_INPUT` to the card's hwmon `temp2_input` (for `pwrap.sh`) and `LLM_BUILD_ROOT` to the folder of the
llama.cpp builds (for `fwrap*.sh`). The Vulkan probes (`vkgame`, `mat_card`) are built from `src/vk` and `src/study`
unchanged. `fana.py` and `anaA.py` read `watch/<job>/samples.csv` under `R_RUN`.

## Corrections made while writing this up

The results summary and the logs disagreed in these places; the log is used above.

- **Item I, SPM exit codes:** `out/jobs-<state>.tsv` records exit 1 for both SPM jobs; the `.meta` files and the logs say
  exit 2 (10.1, argument error) and 134 (10.2, abort). The tsv holds the gate's own code; the logs are used.
- **Item I, "works":** the run's tables count a counter as working on the median of all dispatches; the audit above shows
  those medians include stray counts, so a few verdicts (`SQ_INSTS_VALU` on 10.2, `SQ_INSTS_WAVE32_VALU`, `SQ_WAVES`)
  say NONSCALING where the clean dispatches are exact. The run's `README.md` also still says `profile_standard` was
  untested; it was written before that run.
- **Item K:** the stock profile's 317 W cap is not in the K logs; it comes from item G's text.

- **Deliberate breaks:** the summary's table counts 7 including the `bl-brk-*` jobs; the `.meta` files show 9
  jobs with exit 1 (7 named breaks plus `bl-brk-f16` and `bl-brk-fp8`), and 590 only adds up with 9.
- **State P bursts:** The results summary gives the bursts' power as 234 to 243 W and the sampler's median clock as 3.21 to 3.22 GHz; the
  `BURST` lines of the logs give a maximum power of 234 to 278 W (f16) and 244 to 257 W (fp8) and a median clock of 3.21
  to 3.33 GHz.
- **State L:** The results summary says p50 and max 2.81 GHz and at most 64 W; that holds for the arithmetic jobs only (max 2.812 GHz).
  The conversion jobs reached 3.34 GHz and 325 W in a few samples.
- **Item B:** The results summary says "+0.06 to +0.09 ms on graphics, within 0.1 ms"; the per-arm values run from +0.041 to +0.097 ms,
  and 3 of 4 compute jobs fail the script's own 0.1 ms limit (the largest arm is +0.132).
- **Item F:** the cache-resident run times are 34.4 and 40.8 us in `Fp-01` and `Fp-02` (summary: 34.5 and 40.7); the PPT0
  count of `Fp-01` is 42 of 52 busy samples (summary: 42 of 50).
- **Item H:** The results summary puts round 4's 111.7 and 145.7 at memory 1408 MHz. `llm-runtimes.md` says they were at the same
  clock as each other, 1258 MHz, and that 158.7 was at 1408 MHz, so that sentence of the summary is not used. The 5 ms and 20 ms
  delay checks were in two different runs (`H-vllm-1`, `H-vllm-2`), not both in one. The reported power while decoding is
  a median of 162 to 165 W in the busy samples, not 160 to 175 W.

## Parked, skipped and open

- **Which part of `profile_standard` makes the counters count** (item I): pinned clock, power gating or something else;
  not tested. Whether the same holds on a second boot: not tested (one run per state, `auto` first).
- **Why the profiled `valu_loop` takes 4.25 cycles per vector instruction** and why the table's medians include stray
  counts (item I): not examined.
- **SPM on 10.2** ("Invalid input parameter"): stopped at the debugging limit.
- **The hangs' cause on the overclock profile** (item K): the stock-profile result is 3 clean runs of one command;
  prompt processing, the NaNs and ROCm 10.2's PyTorch FP8 were not run on stock.

- **Sustained heavy load on the profile** (item A): not tested; it reaches the 95 C guard in about a second from idle.
- **Whether the arithmetic checks relate to the resets or the PyTorch hangs:** not shown either way.
- **The sampler's offset cause, and its offset in a running game** (item B): not tested. The game-route network run was
  not done (the staging folder was not on this disk).
- **A DRAM-streaming kernel-only loop for the real FP8 decode kernel** (item F): needs a patched-tree build with the shapes
  patch; not built.
- **The old vLLM protocol** (`llm4_vllm.py bench`): not rerun, so the origin of round 4's 23 % gap stays INFERRED.
- **What differs between "Testing" and "Default" that moves the Vulkan start clock** (item G): open; no profile switch was
  made this round.
- **`include/amd_shared.h` line 230** (`PP_OVERDRIVE_MASK`): taken from the results summary, not re-read.
- The two `cvt_sr rule` jobs were not repeated (above); `cvt_time` was not run as a check.
