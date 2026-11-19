# RX 9070 XT through the players' route: Model 1 and Model 2 (route lane, gaming round)

Status: first pass 2026-11-17, 21:20-22:12 UTC; corrected after an independent review (FIX-PASS) and extended by
follow-up clock experiments, 22:25-23:30 UTC (fix round). The fix round rewrote the statements the review found
false or unsupported in place; every change is listed in the last section, "Corrections after review
(2026-11-17)", with the file the new figure comes from. The new clock experiments are in "Follow-up experiments:
what decides the clock step (E1-E4)". Sections are otherwise in the order the work was done; the summary comes first.

Labels: MEASURED (seen in a run here, folder named under `G/route/runs/`), READ (from a file, path named),
INFERRED (reasoning). G = a staging folder outside the repository, R = the upscaler project's own work folder (also outside it). Wall-clock
numbers are for this card on the LACT "Testing" profile (stock clocks, 317 W cap; the 1.2x regulator mod is in the
card, so reported watts are scaled down).

## Where this was made, and where its files are

This round was run in a staging folder outside the repository (called G
below), because it works on copies of another project's material (the upscaler's runtime, shaders, hosts and test
cases, and a Wine prefix). Those copies, the run scripts built around them (`G/route/gr.sh`, `series.sh`, the
analysis scripts) and every run folder (`G/route/runs/`, with the raw clock samples and per-dispatch records) stay
in G and are not in this repository. Only this write-up and the three tables of section G5
(`measurements/gaming-tables/`) were copied in by the study lead after two independent reviews. The lead reran the
headline arms himself (`G/route/runs/leadrerun`). The second lane of the round (plain-Vulkan probes) is
`measurements/gaming-probes.md`.

## In plain words (added by the study lead after the re-check)

- On the route games use, this card does not run short pieces of work at full speed. Work starts at a low clock
  (about 1.7 to 1.8 GHz) and the card raises it to about 3.1 to 3.2 GHz only after roughly 15 to 38 ms of almost
  unbroken work. With pauses of 3 ms or more between frames it stayed at the low clock; a Model 1 frame then takes
  about 2.8 to 3.6 ms and not 1.6 ms. Our own low-level (HIP) probes do not show this: they start near 2.7 GHz.
- This was seen with headless test runs on a card that was otherwise idle. Whether a real game is affected was not
  tested. The 1.62 ms measured in a game matches the full-clock time.
- At full clock Model 1 takes 1.61 ms per frame; its time follows the clock, so it is not held back by memory speed.
  Those full-clock runs heated the chip's hotspot to 83 to 84 C within a tenth of a second and ran with the
  firmware's hotspot flag set.
- Open: why a separate compute queue raises the clock at longer pauses than the game-style queue, why the low clock
  has two levels, and whether real games see any of this.

## In short

- **On this route the card starts every job at a low clock and steps up only if the idle gaps stay short.** Frames
  start at about 1.65-1.80 GHz (the gauge kernel's time; two persistent card states about 100 MHz apart). With idle
  gaps of about 1 ms between ~2.8 ms frames the clock steps to 3.1 GHz after about 15-38 ms of back-to-back frames
  (study's HIP probes: 2.7-2.8 GHz start, full clock after 14-15 ms). On the DIRECT queue every run with gaps of
  3 ms or more stayed low (55 runs, shares 14-50 %), every run with gaps of 2.1 ms or less stepped. Busy share does
  not decide it: on the COMPUTE queue the card stepped with 4-6.7 ms gaps at 32-42 % share, where DIRECT never did,
  and at about 43 % share small frames with 1 ms gaps stepped while large frames with 3-4 ms gaps did not. A second
  of real idle before the frames changed nothing. Model 1's frame takes about 2.8 ms at the low clock and 1.6 ms at
  full clock. MEASURED. Whether a game shows the low state was not tested; the in-game Model 1 time on this card
  (1.62 ms) is the full-clock time (READ / INFERRED).
- **The output kernel follows the shader clock; one small kernel does not.** In a same-host test where only the
  idle gap (and with it the clock, ratio 1.76) changed, `m1_out_c` took 1.80 times longer, the network
  stages 1.62-1.84, but `m1_out_a` only 1.05 times (memory and fabric clocks unchanged). So the output stage's
  main kernel is not bound by memory bandwidth or fabric (answering the upscaler developers' puzzle: 29.5 MB is 0.1 ms
  of bandwidth, the stage takes 0.5 ms); per SIMD it costs about what it costs on the RX 7900 XT (3,400 against
  3,300 cycles per wave), and the RX 9070 XT has fewer SIMDs. MEASURED (times, clocks) / INFERRED (the comparison).
- **The "full clock" figures were taken with a throttler flagged.** Without the profiler, 421 of 500 own-buffer model 1
  frames carry `TEMP_HOTSPOT` (and some `PPT1`); the hotspot reached 84 C within about 70 ms while the edge sensor
  read 53 C, reported power reached 400 W (459 W for Model 2) against a 317 W cap, and the clock held 3.13-3.19 GHz.
  The first pass's thermal stop could not trigger (it read the hotspot after the run); it is fixed and tested.
  MEASURED.
- **Occupancy curve of the output kernel:** from 9 to 8 waves per SIMD (computed from the study's rule) costs 1.6 %,
  to 4 waves 33 %, to 2 waves 126 %; the same code as wave32 is 28 % slower (RDNA 3 also measured its wave32 form
  slower). MEASURED. Whether it is limited by instruction issue or by exposed latency is not separated by this curve
  (INFERRED either way).
- **Headline times (own-buffer DIRECT, full clock, the owner's DLL pair):** Model 1 frame 1.609 ms on the GPU
  (input 0.362, network 0.766, output 0.478; in game 1.62: 0.32 / 0.76 / 0.53). Model 2 kernels 2.52 ms on the
  COMPUTE queue; against the RX 7900 XT's emulated-FP8 route (minimum against minimum) six of its seven big
  kernels are 1.5-4.6 times faster, but its post kernel is 1.62 times slower (610.6 against 378 us, same source;
  unexplained). MEASURED.
- **COMPUTE queue against DIRECT:** no two-speed split (RDNA 3 had one). COMPUTE costs 0.18-0.20 ms more per model 1
  frame by the runtime's parts (0.15 ms by the profiler span): about 2 x 49 us waits where the runtime puts timing
  marks, and 1-8 us more on each of the other dispatches. On DIRECT, three small dispatches are followed by a gap
  of about 24 us before the next record starts, where on COMPUTE the same dispatches are simply longer and no gap
  follows (38.0 us + 24.4 us against 63.7 us + 0.4 us). MEASURED. Two readings fit and were not separated: the
  DIRECT end timestamp comes early (then per-kernel DIRECT times under-count), or there is a real idle bubble of
  about 24 us on DIRECT.
- **The compiler matters a lot:** ACO with no instruction scheduling makes Model 2 30 % slower (one kernel +160 %),
  Model 1 9 %. Without value numbering one network kernel gets 192 instead of 256 registers and runs steady at 206-210
  us instead of wandering 233-340 us (MEASURED, 2 runs per arm); its code also changes in other ways (11 % more
  instructions, 3 times the copies), so occupancy is a candidate cause, not a shown one.
- **The players' route itself:** GE-Proton11-11's own vkd3d-proton cannot create Model 1's output kernel (dxil-spirv
  lacks AGS opcode 0x1c); the owner's pair (1) and the unpatched game build (2) produce byte-identical machine code
  for all 99 kernels and the same times and output bytes (MEASURED). RADV honours every declared wave size; no
  kernel spills (READ, listings).

## The route as run

- Copies (all under G): a Wine prefix `G/pfx` copied from the upscaler project's no-display prefix (`Graphics=null`,
  `ShowCrashDialog=0`; the original `user.reg` is kept as `route/state-before.txt`); the upscaler's source at
  revision 2 in `G/src` (git archive); hosts, runtimes, cases under `G/route/bin`, `G/route/cases`, `G/route/mbin`,
  `G/route/mcases`. Unedited originals of the copied scripts are packed in `G/route/orig.tar.gz`.
- Hosts: `route_host.exe` (game route: the runtime's exported upscaler backend records into the host's DIRECT
  command list, exactly as in game; source in the upscaler project's capture-host folder, read in full: line 3 and
  129-132 DIRECT queue and list; no capture start and no capture folder option, so nothing is captured;
  it writes `f<t>_out.bin` per frame, so frames 3+ get read-only empty placeholders and the write fails silently);
  `buffer_host.exe` (own-buffer path, the runtime's own API; source in the upscaler project's work folder);
  `gap_host.exe` (the upscaler's prebuilt gap host, byte-equal to the copy here; not built or edited in this lane;
  source read: an earlier `buffer_host` plus a gap option (`GAP_MS`), a QPC spin of that many milliseconds after
  every time-loop frame. It is a different binary from `buffer_host.exe`, so the gap0 arm and `od_p` are not the same
  host; their step times agree, see the clock section);
  `m2_host.exe` (model 2: the upscaler's own-buffer harness host, one of its tuning builds); `pso_sweep.exe` (the
  upscaler project's pipeline sweep tool).
- Runtime for model 1: a developer build of the upscaler's runtime DLL (checked against its checksum list):
  embedded gfx12 kernels of main (revision 1), carries a switch for the own-buffer path that selects the DIRECT
  queue (`QUEUE=direct` below). Runtime for model 2: the runtime of the owner's game test setup (embedded
  gfx12 set; its post kernel source equals revision 1's, not revision 2's).
  No gfx11 switch set; every report's `notes` line is empty (no gfx11 note). MEASURED.
- DLL pairs (d3d12.dll / d3d12core.dll):
  - 1, reference (what ran in game): the owner's game folder snapshot (vkd3d-proton 5b1089c8 + dxil-spirv 58ea644b,
    extended emulation off; READ from the upscaler's developers via the study lead, checksums verified here).
  - 2, second arm: a build from the same sources with extended emulation on.
  - A: GE-Proton11-11 `files/lib/wine/vkd3d-proton/x86_64-windows` (vkd3d-proton build 0bb64c56). Cannot create
    the kernels (below).
  - C, control: a patched RDNA 3 pair from the upscaler project's build output; run with `DXIL_SPIRV_CONFIG` unset.
- gfx12 DXIL on disk (for `pso_sweep`): the upscaler project's build output, 100 of 100 files equal to its checksum
  list (main's set, revision 1). MEASURED.
- Mesa: RADV "radv 26.99.99" (vkd3d log), device string "AMD Radeon RX 9070 XT (RADV GFX1201)". Wine: system
  wine-staging (`/opt/wine-staging`). DXGI in the prefix is DXVK's (v3.1.1-134), kept as in the RDNA 3 runs.
- Card state at every job (READ, each run's `state.txt`): `power_dpm_force_performance_level` auto,
  `pp_power_profile_mode` BOOTUP_DEFAULT selected, memory clock 1258 MHz, LACT profile "Testing".
- Guards (`G/route/gr.sh`, every GPU job): LACT profile must read "Testing"; card1 must be PCI 1002:7550; no game
  process (`/proc/*/comm`: Cyberpunk2077*, StarRail*, *Launcher*, reaper, gamescope*, any `.exe` outside G's
  prefix); >= 8 GB available; hotspot (`temp2_input`) < 60 C and `gpu_busy_percent` < 10 three times, checked
  before and again under the lock (the lock is given back if either fails); the game-process check is repeated
  under the lock since the fix round (before it, it ran only before the lock); refuses the runtime's gfx11 switches,
  `DXIL_SPIRV_CONFIG`, `RADV_PERFTEST`, `MESA_VK_DEVICE_SELECT`, the upscaler's approximation and softmax-test options, and paths outside G;
  `flock` on the shared lock, `systemd-run` scope `MemoryMax=6G MemorySwapMax=0`, `nice -n 10`, `timeout`;
  Wine with `DISPLAY`/`WAYLAND_DISPLAY` unset, scratch `XDG_RUNTIME_DIR`, `SDL_VIDEO_DRIVER=offscreen`,
  `winedbg.exe=d`, sound drivers disabled (`winealsa/winepulse/winejack/wineoss/winecoreaudio.drv=d`); a watcher
  (`watch_run.py`) checks every 0.5 s for display variables, a foreign `XDG_RUNTIME_DIR`, wayland/X11/pulse/pipewire
  sockets of the Wine processes and files in the scratch runtime folder; a 2 ms sampler (`sample_gpu.py`) and a
  device-client sampler run per job. Thermal stop: in the first pass the hotspot was read after the job, when the
  card had already cooled (it falls below 60 C within 0.15 s, study), so its 85 C stop could never trigger; the
  in-run maximum of those runs reached 84 C. **All 162 runs of the first pass were made without a working
  thermal stop.** Since the fix round the stop uses the in-run maximum of the run's own sampler file (88 C, or no
  reading at all, writes the STOP file and ends the series); tested on the card (Checks).
- Adapter check, every run: the profile's `adapter` field, or the runtime's log line naming the D3D12 device, or
  (runs with neither) the host process's DRM engine time per PCI device from `/proc/<pid>/fdinfo`: the RX 9070 XT
  (0000:03:00.0) must have engine time and the iGPU (0000:7c:00.0) none. The Wine process opens both cards' nodes
  while it enumerates adapters, so "which node it holds" is no evidence (the first version of the check used it and
  correctly failed). MEASURED.

## Smoke (step 1), what was settled by trying

MEASURED, folders `runs/smoke/*`, `runs/dump/B_all`:

- Pair A cannot run Model 1: `m1_out_a` fails at pipeline creation, vkd3d log "dxil-spirv: Unsupported AGS magic
  instruction 0x1c (immediate 1)", then E_INVALIDARG (`runs/smoke/k1_A_compute`). Opcode 0x1c is the F32 to F16
  conversion in model 1's output kernel source. READ from the upscaler's developers: no released Proton carries
  dxil-spirv PR #362 (the AGS opcodes the upscaler needs) yet; in game the owner's pair 1 sat in the game folder.
- Adapter: vkd3d-proton picked the RX 9070 XT in every run (the iGPU and llvmpipe are listed; llvmpipe skipped).
- Runtime: with no gfx11 switch the developer DLLs run main's gfx12 kernels; report `notes` empty.
- Softmax self-check: passes (the mixed softmax form; one `m1_probe` and one `m1_probe__rodd` launch at create, then only
  the mix forms in frames).
- DIRECT switch on the own-buffer path works (log "native D3D12: command queue DIRECT"; absent in COMPUTE runs).
- The game route launches other input/output kernels than the own-buffer host: `m1_in_d_direct`,
  `m1_out_a_direct`, `m1_out_c_direct` (texture-reading forms, GPU-side uniforms) instead of
  `m1_in_e`, `m1_out_a`, `m1_out_c`.
- At create the runtime launches the small "wide" stages `m1_net_15`, `m1_net_16`, `m1_net_17` (and their `__rodd` forms) dozens of
  times with 15, 16, 18, 20 and 24 groups, then keeps one group count: 20/8/20 on the own-buffer host, 24/6/24 on the
  game route in the first runs. INFERRED: a launch-size tuning at create.
- Model 2 (`m2_host`, own-buffer, COMPUTE) runs: 8 dispatches a frame (exposure + the 7 big kernels named in the
  plan); the network kernels' group counts change from frame to frame (`m2_net_01` 3283 then 3267), so
  their grids depend on the frame's data.
- Machine-code dump: `pso_sweep` + `RADV_DEBUG=nocache,shaders,shaderstats` creates all 99 non-texture kernels (78
  buffer, 21 direct root signatures), 99 OK (`runs/dump/B_all`).

## G1. Model 1's input and output stages (1706x960 -> 2560x1440, synthetic model and inputs)

### What the card did (observation first)

Arms (pair 1 unless named; "own-buffer" = `buffer_host.exe`, the runtime's own queue and buffers, 10 warm-up + 50 timed
frames of the same input; "game route" = `route_host.exe`, the runtime records into the host's DIRECT list, frames
alternate the two inputs, one submit and wait per frame, then a 29.5 MB read-back):

| arm | runs x frames | GPU sum min / median / p95 / p99 / max (ms) | input / network / output medians | clock |
|---|---|---|---|---|
| own-buffer DIRECT, no profiler, pair 1 (`runs/g1s/od_n_r*`) | 4 x 50 | 1.569 / 1.609 / 1.653 / 2.018 / 2.039 | 0.362 / 0.766 / 0.478 | 3.18 GHz |
| own-buffer DIRECT, no profiler, pair 2 (`runs/g1/od_n_r*`) | 6 x 50 | 1.558 / 1.607 / 1.644 / 1.788 / 1.998 | 0.362 / 0.764 / 0.477 | 3.18 GHz |
| own-buffer DIRECT, profiler on, pair 1 | 4 x 50 | 1.612 / 1.666 / 1.695 / 1.698 / 1.701 | 0.375 / 0.786 / 0.502 | 3.1 GHz |
| own-buffer COMPUTE, no profiler, pair 1 | 4 x 50 | 1.679 / 1.792 / 1.824 / 2.012 / 2.305 | 0.413 / 0.817 / 0.524 (exposure 0.049) | 3.17 GHz |
| game route, runtime's 300-frame timing log, pair 1 (`runs/g1s/grt_r*`) | 4 x 299 (means; max) | mean 2.827-2.833, max 2.864-2.874 | 0.606-0.609 / 1.333-1.336 / 0.881-0.883 | 1.73-1.76 GHz |
| game route, pair 2 (`runs/g1/grt_r*`) | 6 x 299 | mean 2.828-3.024 (two clock groups: the two card states of the E section) | 0.608-0.643 / 1.332-1.433 / 0.882-0.943 | 1.63-1.76 GHz |
| in game, Cyberpunk, RX 9070 XT (READ, the upscaler project's handoff note, s6) | median | 1.62 | 0.32 / 0.76 / 0.53 | not recorded |

- The slowest samples of the no-profiler arms are the first timed frame of a run (frame 0 at a sampled 1.67-1.70 GHz):
  the clock had not finished its step (see "Clock" below). Without them the distributions are narrow (p95 within
  3 % of the median). MEASURED.
- The "full clock" rows ran with a throttler active. In the no-profiler own-buffer runs the driver's throttle words
  are non-zero in 421 of the 500 timed frames (162 of 200 pair 1 frames, `ana/g1p1_od_n_frames.csv`, and 259 of 300 pair 2
  frames, `ana/g1p2_od_n_frames.csv`, column `clk_thr`): `TEMP_HOTSPOT` in all of them, `PPT1` as well in some; the
  in-run hotspot maximum was 83-84 C and reported power up to 389-400 W (section "Hotspot, power and throttle
  bits"). What the throttle bits cost cannot be read from these runs: every frame from the 15th on carries a bit
  (140 of 140 for pair 1, 210 of 210 for pair 2), and the frames without one are the first 12 of a run, during the
  clock step (pair 1: median 1.603 ms with a bit, 1.626 without), so no full-clock frame without a bit exists to compare
  with. The clock in samples with a bit is not lower
  (median 3,164 MHz against 3,144 without, all runs). So "full clock" here means the 3.1-3.2 GHz level the card
  held with `TEMP_HOTSPOT` set; whether it would hold a higher clock without that throttler was not tested (the
  study's light loads reach 3.42 GHz). MEASURED.
- The per-dispatch profiler adds about 0.05 ms per frame (1.666 against 1.609). MEASURED.
- Game-route host frame time (dispatch + submit + wait, host clock, frames 2 onward): median 3.57-3.74 ms, p99
  3.66-4.66, max 3.71-5.38 ms (pair 1, `runs/g1s/gr_r*`, `grt_r*`). MEASURED.

Per dispatch, own-buffer DIRECT, profiler on, 200 frames (pair 1) / 300 (pair 2), microseconds, median (min / p99 / max):

| kernel | groups x threads (wave) | pair 1 | pair 2 |
|---|---|---|---|
| `m1_out_a` | 12795 x 128 (64) | 61.7 | 61.8 (60.8 / 62.7 / 63.0 in the first 100) |
| `m1_in_e` | 3840 x 256 (64) | 305.9 | 304.7 (300.9 / 309.5 / 310.5) |
| `m1_out_c` | 28800 x 128 (64) | 497.0 | 493.8 (490.0 / 502.1 / 502.7) |
| `m1_net_06` (reference) | 1985 x 128 (32) | 128.3 | 127.7 |

Full per-kernel tables: `G/route/ana/g1p1_od_p_*.csv`, `g1p2_od_p_*.csv` (every dispatch of every frame with its gap).

### The clock exponent (does the time follow the shader clock?)

Prediction from the plan: if the output stage were limited by DRAM traffic (29.5 MB written plus history reads, about
0.1 ms at 581-633 GB/s), its time would not change with the shader clock (exponent near 0). If it is limited by
instruction issue or by on-chip latency (L0, L2, LDS, all clocked with the shader clock), time x clock is constant
(exponent 1). Contradicted if the exponent is below 0.5.

**Clean test (fix round; replaces the 12-run fit below as the evidence).** One host (`gap_host.exe`), one queue
(DIRECT), one kernel set, pair 1, profiler on, runs `runs/clk`: high clock = the 3 gap0 runs, frames 25 onward
(111 frames); low clock = the 9 gap2, gap4 and gap8 runs, frames 5 onward (513 frames). The only thing that differs
is the idle gap, which the card answers with a different shader clock. Fabric clock 2016 MHz and memory clock 1258
MHz in every awake sample of both sets. Clock ratio high/low: 1.757 by gpu_metrics, 1.723 by `pp_dpm_sclk`, 1.767
by the gauge kernel. (`fix/exponent.py`, `ana/fix_exponent_clk.txt`; MEASURED.)

| kernel | high clock, median us | low clock, median us | ratio | exponent (gpu_metrics clock ratio 1.757) |
|---|---|---|---|---|
| whole frame span (ms) | 1.66 | 2.84 | 1.71 | 0.95 |
| `m1_out_c` | 497.6 | 895.0 | 1.80 | 1.04 |
| `m1_in_e` | 306.7 | 518.8 | 1.69 | 0.93 |
| `m1_net_06` (the gauge itself) | 128.6 | 227.3 | 1.77 | 1.01 |
| `m1_net_09` | 130.7 | 240.9 | 1.84 | 1.09 |
| `m1_out_a` | 61.6 | 64.7 | **1.05** | **0.09** |
| `m1_net_15` | 35.4 | 57.1 | 1.62 | 0.85 |
| `m1_net_17` | 48.2 | 79.5 | 1.65 | 0.89 |

So the output kernel's time does follow the shader clock (exponent about 1), and `m1_out_a` hardly depends on it.
What this supports: the output stage is not bound by memory bandwidth or by the fabric (neither clock moved while
its time changed 1.8 times). That the memory clock was 1258 MHz in all busy samples is weak evidence on its own (it
never varied, so nothing was tested by it); the clock scaling is the evidence. It does not say which on-chip limit
(issue, latency, LDS) acts.

The first pass's fit, kept for the record: the same kernels ran at 1.63-1.76 GHz (game route) and 3.18 GHz
(own-buffer). That compares different hosts, kernel forms (`_direct` against buffer forms) and fabric clocks (2268
MHz on the game route, 2016 on the own-buffer host), so it is not a one-variable test. Memory clock 1258 MHz in every
sample. MEASURED.

| stage | exponent, DIRECT-queue runs, per-run medians (12 runs) | exponent inside the game route alone (1.635 -> 1.753 GHz) | time x clock, ms x MHz |
|---|---|---|---|
| input | 0.87 | 0.79 | 1,068-1,151 |
| network | 0.94 | 1.02 | 2,338-2,430 |
| output | 1.03 | 0.93 | 1,515-1,548 |
| whole GPU frame | 0.95 (residual 0.3 %) | | 4,932-5,119 |

(The game route runs the `_direct` input and output kernels, the own-buffer host the buffer forms; the per-run fit
mixes the two; the in-route column does not.) A per-frame fit gave 0.45 only because the 2 ms sampler cannot place
a clock on one 1.6 ms frame: some frames got a reading from before the step. That fit is discarded; the clean test
above replaced the per-run fit as the evidence. (The in-route column compares the two persistent card states of
the E section, 1.635 and 1.753 GHz.) File `G/route/ana/clock_exponent_p2.txt`.

Answer (MEASURED, clean test above): the input kernel, the network stages and `m1_out_c` follow the shader
clock (exponents 0.85-1.09); `m1_out_a` does not (0.09). The output stage's big kernel is not bound by memory
bandwidth or fabric. The upscaler developers' puzzle (29.5 MB is ~0.1 ms of bandwidth, yet the stage takes 0.53 ms in
game, no faster than on the RX 7900 XT): `m1_out_c` costs about a fixed number of shader cycles per wave;
`m1_out_a` (about 12 % of the output stage) does not.

### Cycles per wave, against the listing

`t x clock x 128 SIMDs / waves` (own-buffer DIRECT, 3.13 GHz):

| kernel | SIMD cycles per wave, measured | listing: issue cycles per wave without the f64 blocks (study prices, wave64 1.235 per VALU) | ratio |
|---|---|---|---|
| `m1_out_c` | 3,438 | 7,303 | 0.47 |
| `m1_in_e` | 7,966 | 13,227 | 0.60 |
| `m1_out_a` | 969 | 646 | 1.50 |

- RX 7900 XT, same kernel: 0.415 ms (READ, the upscaler project's I/O profile note, line 14); 84 CUs (READ,
  the upscaler project's takeaways note, line 29, "count blocks against 84 CUs"; 168 SIMDs is INFERRED as 2 per CU);
  in-frame clock about 2.75 GHz (READ, the upscaler project's confirmation note, line 81: per-run shader-clock medians 2,694-2,792
  MHz); RX 9070 XT 64 CUs = 128 SIMDs (READ, study `scheduling-and-clocks.md` line 3). With those: 3,329 cycles per
  wave. Almost the same per-SIMD cost on both cards. INFERRED: the RX 9070 XT has
  fewer SIMDs (128 against 168) at a higher clock, so its output stage lands at about the RDNA 3 time; the network
  (matrix instructions) is where RDNA 4 gains. The plan's arithmetic used 192 SIMDs for the 7900 XT; that is the
  XTX's count.
- The output kernel's measured cost is half of what its listing would cost if every non-f64 block ran once per wave.
  Even at an impossible 1 cycle per wave64 VALU instruction, the non-f64 code (~4,180 VALU + 91 transcendental
  instructions) would need ~4,500 cycles. So a large part of that code does not run on these frames. READ from the
  listing: a wave-uniform branch on the HDR flag (`s_bitcmp1_b32 s18, 0` / `s_cbranch_scc0`; the flag is bit 0)
  guards 1,692 vector instructions (272 f64), and inside it an `s_cbranch_execz` region (BB149 to BB515, 1,628
  instructions) is skipped when no lane of the wave needs it. Which of those regions run on these frames is not
  settled by the listing (no instruction counters through this route). MEASURED hint: on the reset frame
  `m1_out_c` costs about 13 % fewer cycles than on steady frames (ratio to `m1_net_06`: 3.38 against 3.87).
- `m1_out_a` costs more than its issue time: 969 shader cycles per wave at 3.13 GHz for 7 loads and 3 stores. That
  is not a fixed cycle count: in the clean test its time is 61.6 against 64.7 us at clock ratio 1.76 (exponent
  0.09), so at the low clock it takes about 580 cycles per wave. Its time is set by something that does not run on
  the shader clock. INFERRED candidates: memory or fabric latency or bandwidth; not separated here.

## Clock behaviour on the route (G6, observed from the route's own frames)

Not planned as a route item; it showed up in the first smoke run and changes every wall-clock number of this lane.

Time axis (fix round): the profiler's GPU ticks are put on the sampler's clock (CLOCK_MONOTONIC) per run, with an
offset bracketed between "submit start minus GPU start" and "fence-wait end minus GPU end" of every frame; the
bracket is 0.8-0.9 ms wide on DIRECT and about 2.9 ms on COMPUTE, with no drift inside a run (the host's steady clock
under Wine equals CLOCK_MONOTONIC: the last fence-wait end of `clk/gap0_r1` is 0.2 ms before `times.txt` appears).
Clock per frame: the `m1_net_06` gauge (size g; 3130 x 127.8 / gauge us). Frames: frame 0 and frame 1 are the spec's
two frames, each followed by an output download and a 44 MB file write by the host; frames 2 onward are the time
loop (10 warm-up + 50 timed, or 10 + 30 in the fix round). Tool: `route/fix/tl.py`; per-frame files
`route/ana/fix_tl/*.csv` and `fix_frames.csv` in every fix-round run folder. MEASURED unless marked.

- **Before frame 0 the card is not at rest.** For about 2 s before frame 0 (runtime creation: kernel creation, test
  launches of the wide stages, the softmax self-check) the sampler shows the card awake (`pp_dpm_sclk` > 1 GHz) in
  a median 32 % (1-2 s before frame 0) and 53 % (last second) of samples, in 100-440 separate bursts per run, at a
  median 1.82 GHz; the last awake sample is 0-5 ms before frame 0. Before Wine starts the share is a median 3 %
  (21 runs of `runs/clk` and `runs/leadrerun`).
- **Frame 0 runs at about 1.75 GHz** (gauge 1.58-1.80 GHz over 92 own-buffer runs of sizes s, m and g; most
  1.74-1.78), **then the host pauses**: the download and file write after frame 0 leave the card idle for 60-145 ms
  at size g (all but two runs 60-97 ms; 17-21 ms at size m, 2 ms at size s), and about 10-15 ms after frame 1. So
  "time from frame 0" includes that pause; the first write-up's "100-160 ms" and "109-159 ms" step times were
  measured from frame 0 and are wrong as a description of busy time.
- **Measured from the start of the loop (frame 2)**, every own-buffer run whose idle gaps stay near 1 ms steps,
  mostly after **28-38 ms** of back-to-back frames: DIRECT g 15-38 ms (48 runs: clock, lead's rerun, `od_p`, size,
  variant and ACO arms), sizes m and s 14-34 ms (8 runs), COMPUTE g 28-30 ms (10 runs); from frame 1, 16-57 ms. The
  lane's earlier run-by-run figures (gap0 r1-r3 of `runs/clk`): 36.7, 20.0, 32.1 ms from the loop start (50.0,
  32.7, 45.6 ms from frame 1, as the review computed). The step itself takes one to two frames
  (gauge 1.78 -> 2.4-2.8 -> 3.1 GHz, about 3-6 ms). The study's HIP probes saw a different start: 2.69-2.79 GHz for
  8 ms, 95 % at 14-15 ms (scheduling-and-clocks.md s1, READ). So on this route the card spends roughly twice as long
  at a much lower level before it steps.
- Whether the step happens depends on the idle gap between frames (one input changed: `GAP_MS`, a host spin
  after each frame; `runs/clk/`, 3 runs each, plus the lead's rerun `runs/leadrerun`, 2 runs each; low-clock busy
  share = sum of frame spans / wall time over the loop frames before the clock starts to rise):

| idle gap between frames (on the GPU) | low-clock busy share | runs | what the clock did (gauge) |
|---|---|---|---|
| 0.95-1.24 ms (no added gap) | 66-72 % | 3 + 2 | stepped to 3.08-3.14 GHz, 20-37 ms after the loop started |
| 2.2-2.3 ms (1 ms added) | 54 % | 3 | two runs reached > 2.8 GHz after 73 and 81 ms, then fell back and wandered 2.2-2.7 GHz; the third rose to 2.75 GHz after about 70 ms and wandered 2.2-2.5 GHz. No steady level in 60 frames |
| 3.1-3.3 ms | 45-48 % | 3 + 2 | stayed 1.75-1.77 GHz |
| 4.0 ms | 42 % | 0 + 2 | stayed 1.76 GHz |
| 5.3 ms | 35 % | 3 | stayed 1.75-1.77 GHz |
| 9.0-9.3 ms | 23-24 % | 3 + 2 | stayed 1.75-1.76 GHz |
| 17.1-17.3 ms | 14 % | 3 + 2 | stayed 1.75-1.76 GHz |
| ~68 ms (inputs uploaded and 44 MB read back every frame) | 5 % | 3 | stayed ~1.75 GHz |

- **Busy share alone does not decide it** (earlier runs, before the follow-up experiments): the COMPUTE-queue
  profiler arm (`g1/oc_p`, 6 runs; `g1s/oc_p`, 4 runs) has idle gaps of about 4.7 ms between its low-clock frames
  and a low-clock busy share of about 40 % (39-41 % by the definition above), and still steps to about 3.05 GHz, 28-30
  ms after the loop starts, after 3 frames at the low clock (at the 5th loop frame in every run). DIRECT arms at
  42-48 % never step. Sizes s and m (no added gap; pieces of 0.8 and 1.1 ms, gaps of about 1 ms, shares 42-44 % and
  51-52 %) step after 14-34 ms, while size g at 45-48 % does not. The follow-up experiments E1-E4 below test this.
- The game-route host (`route_host`): the profiler is off on that route, so no busy share from GPU timestamps
  exists. Its frames (dispatch, submit, wait: 3.42-3.87 ms, median 3.56, lead's rerun) run at a sampled clock near
  1.75 GHz; in no game-route run (10 runs of 300 frames, 10 of 60) did the clock step. Two of the six pair-B
  300-frame runs (`g1/grt_r1`, `r2`, the first two of that arm) ran at 1.61-1.63 GHz instead of 1.74-1.75 GHz; they
  also read 4-6 mV lower graphics voltage over the whole run (gpu_metrics); why is not known. Its firmware activity
  is in E4 below. The first write-up's "it sits at that threshold" is withdrawn: it was not measured.
- For players (INFERRED; not tested in a game): a game whose frames leave the GPU idle for a few milliseconds
  between upscaler runs may keep the card at about 1.75 GHz, where Model 1 takes about 2.8 ms instead of 1.6 ms.
  But the in-game Model 1 time on this card, 1.62 ms (READ, the upscaler project's handoff note, s6), matches the full-clock
  time here (1.61 ms), so in the game that was measured the card was at full clock.
- Two clock readings, one source. `pp_dpm_sclk` and `gpu_metrics` are both the firmware's metrics: the
  `gpu_metrics` file changes only every 6 ms (three identical 2 ms samples in a row), `pp_dpm_sclk` at nearly every
  sample, and each time `gpu_metrics` changes its clock equals that sample's `pp_dpm_sclk` (MEASURED, e.g.
  `g1/od_n_r6`; `fix/metrics_cadence.py`). Over all busy samples of all runs they agree exactly in 34 % of
  samples; the median difference is 0 but the 90th percentile of the absolute difference is 1,744 MHz (one reads a
  burst, the other the idle between). The first write-up's check "median difference 0 MHz" is therefore neither a
  second source nor a meaningful agreement, and it is withdrawn. A frame of 1.6-2.8 ms cannot be given a clock
  from either file; the gauge can.
- Gauge against sampler at the low level: the gauge reads 1,757-1,760 MHz; the sampler reads 1,784 MHz
  (`gpu_metrics`, awake samples inside the frames of the gap2/4/8 runs), 1,811 (`pp_dpm_sclk`, same samples) or
  1,821 (all awake samples of the 9.3 ms-gap runs, the first write-up's figure). The first two give gauge time x clock
  constant within 0.6 % between the two levels (the figure `analyze_clk.py`'s docstring quoted); the last gives
  3.6 %. Both are stated; the difference is in which samples count, and the sampler cannot place a level within
  2-3 % on frames this short. The gauge is calibrated from the same sampler, so it is not an independent clock,
  and it is valid for size g only (the gauge kernel takes 39 us at s and 47 us at m at full clock; `analyze_clk.py` now skips
  other sizes).
- Hotspot, power and throttle: see the next section. (The first write-up's line "No throttle bit was set in any
  sample of any run; hotspot max 69 C (gap 0 runs), power reported 47-87 W" was false and is withdrawn.)

## Hotspot, power and throttle bits (fix round, from every run's `clk.txt`)

Source: the 2 ms sampler file of every run (`gpu_metrics` fields: `temp_hotspot`, `temp_edge`, average socket power,
`throttle_status` = the legacy word, `indep_throttle_status` = the independent word). Script
`route/fix/thermal_scan.py`, output `route/ana/fix_thermal_all.txt` (per run and per arm). Names of the independent
word's bits: the study's table of the kernel's `SMU_THROTTLER_*_BIT` list (READ, rdna4-fp8 master
`src/study/sched_common.h`): bit 1 `PPT1`, bit 36 `TEMP_HOTSPOT`. The legacy word's bits are not named in the
study's notes; MEASURED pairing: legacy 0x2 appears exactly with independent 0x1000000000 (`TEMP_HOTSPOT`), and
legacy 0x8002 exactly with 0x1000000002 (`PPT1` + `TEMP_HOTSPOT`), in every sample of every run, so legacy bit 1
travels with `TEMP_HOTSPOT` and legacy bit 15 with `PPT1`; their own names are not known here.

- 162 runs of the first pass: **in-run hotspot maximum 84 C** (`g1/od_n_r1`, `r3`, `r6`, `mod2/mod2_n_r1`); reported
  power up to **459 W** (`mod2/mod2_n_r1`) and up to 400 W in the full-clock own-buffer model 1 runs; a throttle word non-zero
  in **57 of 162 runs** (lead's rerun: 3 of 14). The edge sensor never exceeded 55 C.
- Per arm (pooled samples; hotspot max, power max, samples with a throttle bit, which bits):

| arm (runs) | hotspot max C | power max W | samples with a bit | bits |
|---|---|---|---|---|
| model 1 own-buffer DIRECT, no profiler, pair 2 (`g1/od_n`, 6) | 84 | 400 | 405 | `TEMP_HOTSPOT` 371, + `PPT1` 34 |
| same, pair 1 (`g1s/od_n`, 4) | 83 | 389 | 256 | `TEMP_HOTSPOT` 249, + `PPT1` 7 |
| model 1 own-buffer COMPUTE, no profiler, pair 2 / pair 1 (`oc_n`, 6 / 4) | 81 / 80 | 367 / 362 | 394 / 244 | `TEMP_HOTSPOT` only |
| model 1 own-buffer DIRECT, profiler, pair 2 / pair 1 (`od_p`, 6 / 4) | 73 / 73 | 320 / 330 | 56 / 12 | `TEMP_HOTSPOT` only |
| model 1 own-buffer COMPUTE, profiler, pair 2 / pair 1 (`oc_p`, 6 / 4) | 66 / 65 | 305 / 284 | 0 / 0 | - |
| Model 2, no profiler (`mod2/mod2_n`, 6) | 84 | 459 | 322 | `TEMP_HOTSPOT` 193, + `PPT1` 129 |
| Model 2, profiler (`mod2/mod2_p`, 6) | 69 | 362 | 0 | - |
| model 1 variants (`var/*`, 15) and ACO model 1 arms (`aco/aco1_*`, 8) | 71-74 | 279-346 | 0-131 per arm | `TEMP_HOTSPOT` only |
| clock arms gap0 (`clk/gap0`, 3) | 70 | 320 | 0 | - |
| clock arms gap1-gap16, xfer (`clk`, 18) | 53-58 | 128-185 | 0 | - |
| game route model 1 (`g1*/gr`, `grt`, 20) | 55-58 | 129-136 | 0 | - |
| game route model 2 (`mod2g`, 8) | 62-68 | 291-388 | 0 | - |
| size m / s (`size/od_pm`, `od_ps`, 8) | 60 / 58 | 198 / 139 | 0 | - |
| lead's rerun gap0 / od_p (2 / 2) | 74 / 74 | 334 / 315 | 28 / 18 | `TEMP_HOTSPOT` only |

- When the bits appear: `TEMP_HOTSPOT` first appears in a run at a hotspot reading of 65-72 C (57 runs), and stays
  set while the reading falls (lowest reading with the bit: 57 C). It is not a threshold on the reported hotspot:
  1,511 samples at 68 C or more (clock > 2.9 GHz) carry no bit. `PPT1` appears with reported power of 375 W median
  (min 111 W in a trailing 6 ms snapshot, max 459 W). Reported power above the 317 W cap (`power1_cap`, study) in
  short bursts is itself a surprise: the study's FP8 load held at 317 W under `PPT0`; here `PPT0` never appears and
  bursts read up to 459 W. INFERRED: the reported value is an average over a window shorter than the limit the cap
  acts on; not tested.
- One run's timeline (`g1/od_n_r6`, model 1 own-buffer DIRECT without profiler, pair 2; 6 ms `gpu_metrics` snapshots, times
  relative to the clock step):
  - before the step: clock 1.68-1.80 GHz, power 103-119 W, hotspot 54-56 C, edge 52 C, no bit;
  - step (0 to 12 ms): 2.29 then 3.07 GHz, power 229 then 310 W, hotspot 58 then 63 C;
  - 12 ms: `TEMP_HOTSPOT` set at hotspot 68 C, clock 3.13 GHz, 342 W;
  - 12-48 ms: clock 3.17-3.19 GHz, power 361-395 W, hotspot climbs 71, 73, 75, 77, 79 C; at 48 ms `PPT1` joins
    (legacy 0x8002) at 388 W, hotspot 80 C;
  - 48-84 ms: clock 3.16-3.18 GHz, hotspot 80 -> 84 C, power 359-395 W, `PPT1` on and off between snapshots;
  - the job ends: power 112 W, hotspot falls 84 -> 74 C within 12 ms, bits still set in the last snapshots.
  The hotspot rose from 63 to 84 C in about 66 ms while the edge sensor read 52-53 C, and the clock did not drop
  while either throttler was flagged.
- What this means for the "full clock" figures: every full-clock model 1 and model 2 number without the profiler (and some with
  it) was taken while the firmware flagged `TEMP_HOTSPOT`, and in the hottest runs `PPT1`. The clock held 3.13-3.19
  GHz during those frames; no full-clock frame without a flag exists, so what the flag costs is not known; the figures describe what
  the card does under these short loads, not an unthrottled ceiling. These runs (up to 60 frames back to back, about
  0.1-0.2 s) were within the lane's thermal rule of continuous busy time but reached 84 C, 1 C below the old stop
  and 4 C below the new one.

## G3 (observation part): COMPUTE queue against DIRECT queue, Model 1 own-buffer frames

- No two-speed split on this card. RDNA 3 split 0.24 ms into two piles on the COMPUTE queue (the upscaler project's slow-frames note).
  Here the COMPUTE arm's GPU sum is one pile: pair 2 300 frames, median 1.790, p95 1.820, p99 1.850 ms; pair 1 200 frames,
  1.792 / 1.824 / 2.012 (the 2.0 values are first frames at the low clock). MEASURED.
- COMPUTE costs more per frame than DIRECT; the figure depends on what is measured (pair 2, full-clock timed
  frames; `fix/queue_sum.py` -> `ana/fix_queue_sum_p2.txt`, `fix/queue_s2s.py` -> `ana/fix_queue_s2s_p2.txt`):
  - the runtime's four parts (exposure + input + network + output, between its own timestamp marks, `times.txt`):
    +0.183 ms without the profiler (1.790 against 1.606, 294 frames each) and +0.195 ms with it (1.852 against
    1.658);
  - the profiler's frame span (first record start to last record stop): +0.152 ms (1.808 against 1.656, 300
    frames each). The span splits exactly per frame into dispatch durations, barrier-record durations and gaps
    (no record overlaps another on either queue): medians 1,562 + 7 + 82 us on DIRECT, 1,704 + 12 + 95 us on
    COMPUTE; differences +142 (dispatches), +5 (barriers), +13 (gaps) us;
  - so the first write-up's "per-dispatch differences 142.8 us plus ~94 us at the two marks" double counts, and
    its "~0.1 ms of the 0.18 at the timing marks" mixed a profiler-on gap with a profiler-off total.
- Why the gaps differ so little: DIRECT has gaps of its own that COMPUTE does not. On DIRECT `m1_net_03` and
  `m1_net_10` are followed by a gap before the next barrier record of median 24.4 us (p95 25.1-25.2), and
  `m1_net_11` by one of median 4.0 us (p95 28.1); e.g. 23.9, 27.8 and 25.0 us in `g1/od_p_r2` fence 309; every other barrier
  record starts 1.3-1.9 us after the dispatch before it; on COMPUTE those gaps are 0.4 us, but the same dispatches
  take that much longer (`m1_net_03` 37.6 + 23.9 us on DIRECT against 64.0 + 0.4 on COMPUTE). INFERRED: on the
  DIRECT queue a dispatch's end timestamp can be written before all of its work has finished, so per-kernel times
  on DIRECT under-count small kernels; not tested. Measured robustly, from one dispatch's start to the next
  dispatch's start (medians, us; DIRECT -> COMPUTE):
  - the two places with timing marks: `m1_in_e` + the wait before `m1_net_06` 308.1 -> 363.3
    (+55), `m1_net_09` + the wait before `m1_out_c` 133.4 -> 190.5 (+57); of each, about 49 us is the
    wait (gap) on COMPUTE;
  - the small and medium network stages: +1 to +8.5 us each (+2 to +20 %; `m1_net_03` +2.3 us, +3.6 %, not +68 %;
    `m1_net_10` +8.5 us, +6.7 %, not +31 %; the wide-4 stages +5.8 to +6.8 us, +11 to +20 %);
  - the large I/O kernels: `m1_out_a` +3.5, `m1_out_c` +10.3 us (+2 to +5 %);
  - sum of these start-to-start medians: 1,653 -> 1,830 us (+177 us). Medians of parts do not add up to the median of
    sums (the span medians differ by 152 us), so each figure is given with its definition.
  Cause of the ~49 us waits not established. INFERRED candidates: the timing marks force a wait on the compute
  queue (the study's compute queue showed a slow submit path of ~70 us, scheduling-and-clocks.md s6); small grids
  losing more suggests fewer wave slots or slower wave launch on the compute queue.
- The players' queue (DIRECT) is what the game uses (the upscaler project's latency note, F1 and F7), so the in-game numbers follow the DIRECT
  rows.

## G2 (observation part): gaps inside one model 1 frame

Own-buffer DIRECT, profiler on (every dispatch is followed by a UAV barrier record): median gap between the end of
one record and the start of the next dispatch 0.16 us (p99 0.20-0.24) for 12 of the 16 dispatches; 1.4-1.9 us where
the runtime's timing marks sit (before `m1_net_06`, `m1_warm`, `m1_out_c`) and 3.0-4.6 us before `m1_out_a`.
Correction (fix round): that counted only the gaps before dispatches. Counting every gap between records, a DIRECT
frame spends a median 82 us in gaps out of 1,656 (the gaps before barrier records: 1.3-1.9 us after most dispatches
and a median 24.4 us after `m1_net_03` and `m1_net_10`; see G3), not 7-9 us; a COMPUTE frame 95 us.
MEASURED (`ana/fix_queue_sum_p2.txt`; the first figure from `ana/g1p1_od_p_dispatch.csv`). The "wide" stages of 8-24
groups take 32-48 us each at 3.1 GHz (RDNA 3: about 70 us at any size).

## G1 continued: size sweep (one input changed: the frame size)

Own-buffer DIRECT, pair 1, profiler on, 4 runs x 50 timed frames per size (`runs/size/`). Fix round: medians of the
full-clock timed frames only (size-specific gauge above 2.8 GHz; `fix/size_full.py` -> `ana/fix_size_full.txt`).
Of the 200 timed frames, 19 at size s and 21 at size m were still at the low clock (the loop's first timed frames:
at those sizes the step comes after frame 12, where timing starts); none at size g. The first write-up's medians over
all timed frames differ by at most 0.4 us, but its p95 values at s and m were low-clock frames (e.g. `m1_out_c`
p95 12.6 us at s, 227.5 at m).

| case (render -> output) | `m1_out_c` waves (pixels / 64) | `m1_out_c` | input kernel | `m1_net_06` | `m1_net_17` | frame span |
|---|---|---|---|---|---|---|
| s (130x78 -> 200x120) | 375 | 8.0 us | 70.3 us | 39.0 us | 43.4 us | 0.465 ms |
| m (853x481 -> 1280x720) | 14,400 | 128.2 us | 104.2 us | 47.3 us | 48.3 us | 0.667 ms |
| g (1706x960 -> 2560x1440) | 57,600 | 492.6 us | 304.9 us | 127.6 us | 48.1 us | 1.655 ms |

- `m1_out_c` against waves: a line through m and g gives 8.44 ns per wave and 6.7 us at zero (two points always
  fit a line, so that is not a test). The third point, s, falls 18 % below that line (8.0 us measured, 9.8 us
  predicted). So the time is not linear in waves down to the small size; with three points nothing more can be
  said about the shape. The first write-up's "about 480 waves" for s was wrong (200 x 120 / 64 = 375), and its
  "H-dispatch-size contradicted" rested on the two-point line; it is withdrawn as a test result. MEASURED.
- The input kernel takes 70 us even at the tiny size: one wave's path through it is about 220,000 cycles long, about
  16 times its listing's issue time. A lone wave is latency-bound; at full size the SIMD gets through a wave every
  ~8,000 cycles because many waves overlap. MEASURED (time) / INFERRED (reading).
- The four wide-4 stages and the wide-m stage take 43-48 us at every size (RDNA 3: ~70 us at any size, the upscaler project's simulation note). They are
  fixed-cost stages on both cards. MEASURED.

## G1b. Model 2 beside Model 1

`m2_host.exe` (the upscaler's model 2 harness host: own buffers, COMPUTE queue), the model 2 runtime (revision 1 gfx12 set), pair 1,
the synthetic model 2 case (`case.json`: render 1706x960, output 2560x1440, 3 frames, `"seed": 1`, read here),
3 setup frames + 30 timed frames, 6 runs with the profiler (`runs/mod2/mod2_p_r*`) and 6 without (`mod2_n_r*`). Launch order,
read from the trace: `m2_expo_a`, `m2_in_a`, `m2_net_01`, `m2_net_04`,
`m2_net_12`, `m2_net_09`, `m2_net_14`,
`m2_post_c` (the plan's list; default fusion setting, one of its options off at this size).
The network kernels' group counts change from frame to frame (e.g. 3283 and 3267 for `m2_net_01`): their grids depend on the
data. `m2_net_12` is 128 groups of 256 (persistent).

Per kernel, frames at full clock (162 of the 180 timed frames = 6 runs x 30; the other 18 timed frames, 3 per run,
ran at the low clock: the step had not happened yet when timing started; they are not the setup frames),
microseconds (`ana/mod2_split.txt`). The RX 7900 XT column is a minimum over 99 frames of a tuned build (READ,
the upscaler project's speed note D, line 77: "min, ms, over 99 frames = 3 runs x 33"), so the like-for-like ratio is RX 9070 XT
minimum / RX 7900 XT minimum (fix round; the first write-up divided medians by minima):

| kernel | min | median | p95 | p99 | max | RX 7900 XT D1, minimum (READ, speed note D, s3.3; emulated FP8) | ratio of minima |
|---|---|---|---|---|---|---|---|
| `m2_in_a` | 210.4 | 213.6 | 217.1 | 218.7 | 219.4 | 307 | 0.69 |
| `m2_net_01` | 275.2 | 286.8 | 294.0 | 295.1 | 296.6 | 1252 | 0.22 |
| `m2_net_04` | 223.5 | 265.0 | 310.1 | 322.0 | 337.3 | 657 | 0.34 |
| `m2_net_12` | 443.4 | 486.0 | 537.8 | 565.9 | 580.2 | 1942 | 0.23 |
| `m2_net_09` | 194.3 | 203.7 | 207.4 | 210.0 | 224.3 | 895 | 0.22 |
| `m2_net_14` | 434.1 | 439.1 | 442.6 | 443.7 | 446.7 | 1432 | 0.30 |
| `m2_post_c` | 610.6 | 615.6 | 618.2 | 619.6 | 622.5 | 378 | **1.62** |
| sum of kernels per frame | 2,425 | 2,517 | 2,575 | | 2,601 | 6,870 (sum of per-kernel minima) | 0.348 (sum of per-kernel minima, 2,391.5 / 6,870); 0.353 with the per-frame minimum 2,425 |

- The repository's in-game direct-path figure for this size is 2.256 ms (READ, a runtime source comment, card not
  named); this harness (COMPUTE queue, profiler on) gives 2.52 ms. Not a like-for-like comparison: queue and profiler
  differ.
- Surprise: the post kernel is 1.62 times slower (minima; 1.63 median against minimum) than on the RX 7900 XT with
  the same source (revision 1 post kernel source). Everything else got 1.45-4.6 times faster (minima). It has no matrix instructions; it runs wave64, 512-thread groups, 84 VGPRs,
  30,720 bytes of LDS per group, and its main loop holds 213 `global_load_b32` and 100 waits on loads per pass
  (census, READ). Per SIMD it costs about 41,000 cycles per wave here against about 28,400 on the 7900 XT (INFERRED
  from the times, 128 against 168 SIMDs). Its low-clock frames are 1.69 times slower for a 1.78 clock ratio, so it
  follows the shader clock too. Cause not established; candidates for a later probe: wave64 load issue, L0 hit
  latency, LDS-limited occupancy (30 KB per 8-wave group). Not tested here.
- Model 2 on the game route (`route_host.exe`, DIRECT, a model file written from `synthetic_tensors(seed=1)`, inputs
  converted to RGBA 32-bit float; `runs/mod2g/`, 4 runs x 299 frames with the runtime's stage log): network GPU mean
  3.47-3.58 ms, max 4.36-4.37 (stage 1 0.28, stage 2 1.76-1.81, stage 3 0.59-0.61, stage 4 0.84-0.87); host frame time median
  4.13-4.40 ms. The clock wandered 1.63-3.03 GHz inside every run (5-95 % of busy samples): a model 2 frame with the
  host's gaps puts it where the clock does not settle (as in the DIRECT runs with ~2 ms gaps, which wandered 2.2-2.7 GHz), so these are not full-clock numbers; why the clock wanders there is not settled. The game route
  runs `m2_post_c_direct`. MEASURED.
- Matrix floor check (plan G1b step 8): FP8 matrix instructions per frame (static counts x launches) are not known
  exactly, because four of the kernels loop over a data-dependent count; not computed. The census shows the matrix
  instructions sit mostly in accumulator chains and that 15-52 % of them wait directly on an LDS or memory operand
  (G4 below), which points to operand latency, not matrix issue, as the limit. INFERRED; not tested by experiment.

## G1 continued: variant kernels, and the verdict for the output stage

Built on copies (`G/srcvar`, the upscaler's DXIL build script with its dxc shim, CPU only; the unchanged revision 2 build first
passed the gate below) and launched by giving each variant its own kernel folder whose manifest entry for
`m1_out_c` points at the variant file (`G/route/bin/V_*`; runtime: a developer build of the upscaler's runtime DLL
that reads kernels from disk; its unchanged run `vbase` writes the same bytes as the model 1 developer runtime).
The runtime's kernel-substitution switch was tried first and does nothing for Model 1: model 1 launches through its own runtime launch function
(model 1 runtime source, lines 960-969), which skips the model 2 runtime's substitution hook. The no-store and no-load
variants of the plan were not built: removing the history stores changes the next frame's data and so its branches,
which would confound the timing.

Variants (listings in `runs/dump/V_var2/isa`): wave32 build of the same source (96 VGPR); and unused LDS padding,
indexed by a run-time value so the compiler cannot shrink it (the first attempt with a thread index let the
compiler bound it to 512 bytes), bringing the group to 8, 16, 32 KB. The compiler also raised the VGPR count when
LDS became the occupancy limit (96, 144, 144), so the padded arms change scheduling a little as well as occupancy.

`runs/var/`, 3 interleaved rounds x 50 frames, own-buffer DIRECT, pair 1, all 15 runs write byte-identical
`f0_out.bin` / `f1_out.bin` (equal to the unchanged kernel's):

| `m1_out_c` form | waves per SIMD (computed from the study's register/LDS rule, not observed) | median us | min / p99 / max | vs original |
|---|---|---|---|---|
| original, wave64, 84 VGPR, 1 KB LDS | 9 | 491.7 | 484.6 / 499.1 / 500.0 | - |
| 8 KB LDS, 96 VGPR | 8 | 499.6 | 492.9 / 506.1 / 507.0 | +1.6 % |
| 16 KB LDS, 144 VGPR | 4 | 651.9 | 647.6 / 657.1 / 657.4 | +33 % |
| 32 KB LDS, 144 VGPR | 2 | 1,110.0 | 1,103.8 / 1,118.6 / 1,120.7 | +126 % |
| wave32, 96 VGPR | 16 (wave32) | 627.6 | 620.9 / 635.0 / 635.8 | +28 % |

Prediction written before the variant runs (from the study): wave32 can pair instructions (VOPD) and wave64 issues
VALU at only 0.81 per cycle, so a wave32 form could be up to 1.2-1.6 times faster if VALU-bound (plan G4 item 4);
contradicted if wave32 is not faster. Result: contradicted, wave32 is 1.28 times slower. RDNA 3 had already measured
the same direction for this kernel: a wave32 form the compiler could pack ran 721 against 640 us (READ,
the I/O profile note, line 132, quoting the output kernel source comment at its lines 25-26). The listings explain the
direction: the wave32 build has only 177 VOPD pairs; issue cycles per pixel over every non-f64 block are 114
(wave64, 1.235 cycles per instruction) against 178 (wave32), a predicted ratio of 1.56 (`ana/variant_issue_model.txt`).
INFERRED: wave64 wins because one wave64 VALU instruction covers 64 lanes in 1.235 cycles where wave32 needs two
instructions; the measured 1.28 sits below 1.56 because not all of the time is VALU issue.

What was measured for `m1_out_c`: its time follows the shader clock (exponent 1.04 in the clean test); 8
instead of 9 computed waves per SIMD changes the time by 1.6 %, 4 by 33 %, 2 by 126 %; the wave32 build, with twice
the resident waves, is 28 % slower. (The first write-up's "time linear in waves, intercept ~7 us" rested on a
two-point line; see the size sweep.) INFERRED, not shown: "limited by instruction issue, with latency hidden at
the real occupancy". Exposed latency would give the same occupancy curve (flat down to some occupancy, then
rising), so the curve does not separate the two. What would separate them: an issue-rate change at fixed
occupancy (for example the same code with independent filler VALU instructions added: issue-bound time grows by
their issue cycles, latency-bound time hardly at all), or per-wave instruction and stall counters, which this route
does not give. The plan's store and load variants were not run. The input kernel was not varied; its size sweep says
one wave alone is heavily latency-bound (70 us at the tiny size), and its clean-test exponent is 0.93.

## G4. Code generation through dxil-spirv on gfx12

All from `runs/dump/p1_all/isa` (pso_sweep, pair 1, `RADV_DEBUG=nocache,shaders,shaderstats`), checked against the code
the real runs compiled (see Checks).

- Wave size: declared equals the size in the listing for all 95 kernels the listing decides (61 wave32, 34 wave64);
  4 undecided (three tiny probes, one `__a1` form). RADV honours `[WaveSize]` through vkd3d-proton. READ (from the
  listings).
- No spills, no scratch in any kernel (READ, listings). Model 2's network kernels: 192-256 VGPRs (5-8 waves per SIMD); RDNA 3 had 65
  (`m2_net_01`) and 165 (`m2_net_12`) spilled VGPRs on gfx11 (READ, speed note D).
- ACO's "Subgroups per SIMD" for wave64 kernels is twice the study's measured rule (e.g. 18 for 84 VGPRs, where the
  rule gives 9); the variant runs agree with the study's rule (8 against 9 waves changes almost nothing; see G1).
  INFERRED that ACO counts wave64 like wave32 there.
- Matrix code (`census/wmma_sites.txt`): Model 2's network kernels use `v_wmma_f32_16x16x16_fp8_fp8` (200-648 per
  kernel) with a few FP16 forms; Model 1's network stages use `v_wmma_f32_16x16x16_f16` only (110-420). No `v_nop`
  anywhere. A matrix result used directly as an A/B operand (the study's 19/11-cycle case): 0-4 sites per kernel.
  Accumulator chains (no extra cost in the study): 75-90 % of the matrix instructions. A memory or LDS wait directly
  before a matrix instruction: 15-52 % of them (`m2_net_12` 341 of 656). `s_delay_alu` directly before one: 6-17 %.
  READ (listing); the expected consequence (operand latency, not matrix issue, limits these kernels) is INFERRED and
  not tested.
- Behaviour test with Mesa's ACO switches (`runs/aco/`, 2 runs per arm, profiler on; model 1 own-buffer DIRECT, model 2 own-buffer
  COMPUTE; `ana/aco_summary.txt`):

| switch | Model 1, 9 kernels | Model 2, 7 kernels | largest single change | RDNA 3 model 2 frame (READ, speed note E, s2.4) |
|---|---|---|---|---|
| `nosched` | +8.6 % | +29.7 % | `m2_net_09` +160 % | +21 % |
| `nosched-ilp` | +3.7 % | +12.8 % | `m2_net_04` +53 % | +12 % |
| `novn` | 0.0 % | -0.6 % | `m2_net_04` -20 % | +9 % |

  The plan's prediction ("contradicted if kernel times do not move > 3 %") holds: scheduling matters on gfx12,
  more than the RDNA 3 static model ever predicted. MEASURED.
- The `novn` case explained (`runs/dump/m2_base`, `m2_novn`): with value numbering off, `m2_net_04`
  gets 192 VGPRs instead of 256 (8 waves per SIMD instead of 5; its 26 KB of LDS per 8-wave group allows 8). Its time
  goes from 233-340 us, wandering frame to frame on identical input, to a steady 206-210 us (MEASURED, 2 runs per
  arm). But without value numbering the code differs by more than registers (READ, `runs/dump/m2_base/isa/index.json`
  against `m2_novn`): instructions 4,443 -> 4,960, copies 337 -> 969, VALU 2,622 -> 3,110, VOPD 293 -> 444, VMEM
  clauses 35 -> 45. So occupancy is one candidate, not the shown cause. INFERRED candidate: at 5 waves per SIMD an
  8-wave group cannot spread evenly over a WGP's 4 SIMDs, and where groups land changes between launches (the study
  saw 0-16 % of groups land on the same CU as before, scheduling-and-clocks.md s2). Not tested; a test would change
  occupancy alone (for example an LDS pad that holds the novn build at 5 waves).
- Not done: LDS bank-conflict model of the network (attention) kernels and the model 1 mix softmax, and the padding variants for
  them (cut; see "Not measured"); CU against WGP mode (RADV's choice for these kernels was not read).

## G5. Table for the speed predictor (gfx1201)

Files in `G/tables/` (no tool of the upscaler project changed):
- `gfx1201_classes.csv`: 28 rows, the per-class issue and latency figures the predictor's tables need (VALU, packed,
  dot, cvt, trans, quarter, half, f64, permlane, bpermute, VOPD, wave64 VALU 1.235, occupancy rule, dependent chain
  with `s_delay_alu`, WMMA FP16/FP8 and result-as-operand, LDS, scalar loads, L0 / MALL / DRAM latency, launch spacing).
  Each row carries the study file and the verbatim line it was copied from; `route/make_g5_classes.py` refuses to
  write the table if any line is missing (study commit in `route/study/MASTER-SHA.txt`). Fix round: source lines are
  verbatim including their line breaks (CSV-quoted), `issue_cycles` and `latency_cycles` hold numbers only (the
  launch row's microseconds are in its note), latency ranges are in the new columns `latency_low` and
  `latency_high`, and the script refuses text in a numeric column. Values that replace RDNA 3
  constants: wave64 VALU 1.235 (was `WAVE64_VALU_X = 2`), trans issue 4 overlapping VALU, `permlanex16` 1 cycle,
  dependency penalty as the `s_delay_alu` interleave row.
- `kernels_gfx1201.csv`: 99 kernels in the column shape of `kernels.csv` (ACO stats, static class counts, in-loop
  counts, wave declared and observed, FP8/F16 matrix counts).
- `kernel_times_gfx1201.csv`: 524 rows in the shape of `kernel_times.csv` (per kernel per run: n, median, min, max,
  sampled clock), model 1 at three sizes on DIRECT, model 1 on COMPUTE (pair 2), model 2 on COMPUTE. Caveat: the per-frame clock
  columns come from the 2 ms sampler and misplace some frames (see G1); use the run medians.
- Missing for the predictor (READ from the plan's list, still missing): microbenchmarks in `micro.csv` shape for TRANS
  next to WMMA and WMMA mixed with `v_fma_mix`; a gfx12 class function for `isasim.py` (it parses gfx11 mnemonics);
  the per-dispatch fixed cost `beta` (G2 isolation, lane B). New here: an occupancy curve on a real kernel (G1
  variants) and the clock behaviour, which the predictor's fixed `F_CLK` cannot represent.

## Follow-up experiments: what decides the clock step (E1-E4, fix round)

Series `fix-e` (22:34-22:56 UTC), pair 1, own-buffer gap host `gap_host.exe` with the profiler on, spec
`gap_fix40.txt` (the two spec frames + 40 loop frames), 20 arms, 5 interleaved rounds (start rotated each round),
100 runs, all with the corrected guard (in-run hotspot maximum 73 C, no stop, no window alert). Expectations and
what would contradict them were written before the runs: `route/fix/EXPECTATIONS.md`. Per run: `fix_frames.csv` in
the run folder (per frame: GPU start and end on the sampler's clock, span, gap, gauge time and gauge clock, sampler clock,
power, hotspot, throttle words); summaries `route/ana/fix_e_analysis.txt` (with the per-run timelines, gauge clock
of every loop frame) and `route/ana/fix_e_tl.txt`; tool `route/fix/analyze_fix.py`. "Low-clock share": sum of frame
spans over the loop frames before the clock starts to rise, divided by their wall time; "step": first frame above
2.8 GHz, time from the loop start. Firmware activity: median `gpu_busy_percent` (and `gfx_act`) in the sampler
samples of that low-clock stretch. All MEASURED unless marked.

### E1. Queue against busy share and gap (size g; DIRECT `fgd<ms>`, COMPUTE `fgc<ms>`, same binary)

The gap binary cannot pick the queue; the runtime's queue switch (`QUEUE=direct`) does, so both queues ran
the same host binary with the same `GAP_MS` values. COMPUTE adds about 3.4 ms of idle per frame on its own
(its gaps at `GAP_MS=0` are 3.9-4.7 ms), so the two queues cover overlapping shares and gaps at different
settings.

| arm | low-clock share | idle gap | stepped | step time from loop start | high level | firmware `gpu_busy` in the low stretch |
|---|---|---|---|---|---|---|
| DIRECT, 0 ms added | 71-72 % | 1.07-1.12 ms | 5/5 | 15, 31, 31, 34, 34 ms | 3.13 GHz | 92-99 % (med 95) |
| DIRECT, 1 | 56-59 % | 2.06-2.14 | 5/5 | 36, 59, 59, 62, 63 | 3.07 | 72-80 (73) |
| DIRECT, 2 | 48-50 % | 3.01-3.08 | 0/5 | - | - | 62-74 (69) |
| DIRECT, 3 | 42-43 % | 4.03-4.07 | 0/5 | - | - | 46-60 (50) |
| DIRECT, 4 | 36-38 % | 5.05-5.06 | 0/5 | - | - | 36-60 (45) |
| DIRECT, 6 | 29-30 % | 7.01-7.09 | 0/5 | - | - | 19-41 (28) |
| DIRECT, 8 | 24-25 % | 9.02-9.10 | 0/5 | - | - | 9-33 (14) |
| COMPUTE, 0 | 40-42 % | 3.92-4.71 | 5/5 | 13, 23, 28, 28, 28 | 3.05 | 95-100 (99) |
| COMPUTE, 1 | 36 % | 5.44-5.71 | 5/5 | 33, 33, 33, 33, 39 | 3.04 | 81-100 (89) |
| COMPUTE, 2 | 32 % | 6.45-6.70 | 5/5 | 37, 37, 47, 54, 54 | 3.04 | 71-99 (88) |
| COMPUTE, 3 | 29 % | 7.42-7.76 | 0/5 | - | - | 68-99 (80) |
| COMPUTE, 4 | 27 % | 8.49-8.70 | 0/5 | - | - | 71-95 (80) |
| COMPUTE, 6 | 23 % | 10.5-10.7 | 0/5 | - | - | 51-58 (56) |
| COMPUTE, 8 | 20-21 % | 12.4-12.7 | 0/5 | - | - | 41-59 (41) |

- "Stepped" in the table means the gauge kernel's estimate passed 2.8 GHz in the loop. By that rule, at equal busy
  share the queue decides: at 36-38 % DIRECT did not step (0/5) and COMPUTE did (5/5); at equal gap too: DIRECT
  with 5.05 ms gaps 0/5, COMPUTE with 5.4-6.7 ms gaps 10/10. DIRECT stops stepping between 2.1 and 3.0 ms of idle
  (share 57 -> 49 %); COMPUTE between 6.7 and 7.4 ms (share 32 -> 29 %), but not as a cliff: in the COMPUTE arm with
  7.4-7.8 ms gaps (`fgc3`) all five runs rise part of the way (gauge peaks 2.15-2.63 GHz; the sampler's 90th
  percentile of awake samples is 1.99-2.10 GHz there against 1.71-1.83 GHz in `fgc4` and in the DIRECT no-step
  arm `fgd3`; lead's recount from `clk.txt`), so the response is graded there. Not reconciled: the profiler shows
  COMPUTE's extra 3.4 ms per frame as an empty gap, while the firmware reads the card as 80-99 % busy in it and the
  host-to-GPU bracket is 2.9 ms against 0.8 ms on DIRECT. Whatever fills that time is the main unknown in this
  queue comparison. The expectation held; neither contradiction (same behaviour at equal share; COMPUTE losing its step at DIRECT's no-step gaps) was
  seen.
- The firmware's own activity reading is much higher on COMPUTE for the same profiler share (share 27-36 %: 80-89 %
  `gpu_busy`; DIRECT at 36-42 %: 45-50 %). The second expectation held. But the firmware reading does not give one
  threshold across the queues either: DIRECT stepped at a median of 73 % and did not at 69 %, while COMPUTE did not
  step at 80 % (3 and 4 ms added). The samples per stretch are few (the gpu_metrics snapshot changes every 6 ms) and
  spread widely (68-99 % within one arm).
- What E1 rules out: busy share alone, idle gap alone, and the firmware's `gpu_busy` alone as the deciding quantity
  across the two queues. What stays open: what on the COMPUTE queue makes the card step at gaps where DIRECT does
  not. INFERRED candidates, untested: the compute queue stays "active" to the firmware between submissions (it
  reads 80-99 % busy at 27-36 % kernel time), or the power manager treats the queue types differently.
- Levels: COMPUTE's high level is 3.04-3.05 GHz, DIRECT's 3.07-3.13 GHz, in the same rounds; COMPUTE's gap0 runs
  drew less power (max 287 W against 316 W) and stayed cooler (65 against 72 C, no throttle bit against
  `TEMP_HOTSPOT` in 4 of 5 DIRECT runs).

### E2. From rest (`fpz`: 1 s wrapper pause after "frame 1 done"; control DIRECT 0 ms added, same rounds)

The wrapper stopped the host process (SIGSTOP, by PID) 11-15 us after "frame 1 done" appeared in its log and
continued it 1.000 s later (`pause.txt` in each run); the GPU saw 1,007-1,010 ms of idle before the first loop
frame (frame 2), against 8-10 ms in the control.

| | runs | first loop frame (gauge) | low level | step from loop start | high level |
|---|---|---|---|---|---|
| after 1 s idle | 5 | 1,653-1,761 MHz | 1,672-1,782 | 15, 16, 34, 35, 35 ms | 3.12-3.15 GHz |
| control | 5 | 1,658-1,775 | 1,672-1,795 | 15, 31, 31, 34, 34 ms | 3.13-3.14 |

- After a second of real idle the clock starts at the same low level and steps after the same time as without it.
  The expectation held. The creation-phase bursts before frame 0 and the 70-100 ms pause after frame 0 are
  therefore not what puts the card at 1.65-1.78 GHz; the study's HIP start (2.7-2.8 GHz) was not seen after 1 s
  idle either. What stays open: why this route's start level differs from the study's HIP start (queue type,
  driver, the Vulkan/RADV path or the power profile were not varied).
- The step times fall into two groups: 15-16 ms (4 runs here) and 31-35 ms (most runs, old and new). INFERRED,
  untested: the decision is made at a fixed interval, and the loop start falls at a different phase of it.

### E3. Length of the busy piece at a similar share (DIRECT; sizes s `fsd`, m `fmd`, g from E1)

| arm | piece (frame span) | idle gap | low-clock share | stepped | step time |
|---|---|---|---|---|---|
| s, 0 ms added | 0.83 ms | 0.99-1.07 ms | 43-45 % | 5/5 | 27-30 ms |
| m, 0.5 ms added | 1.17 ms | 1.51-1.65 ms | 41-43 % | 5/5 | 65-77 ms |
| g, 2 and 3 ms added (E1) | 2.84 ms | 3.0-4.1 ms | 42-50 % | 0/10 | - |
| s, 2.3 ms added | 0.84 ms | 3.28-3.40 ms | 19-21 % | 0/5 | - |
| m, 2.3 ms added | 1.12 ms | 3.29-3.38 ms | 25-27 % | 0/5 | - |
| g, 2 ms added (E1) | 2.84 ms | 3.01-3.08 ms | 48-50 % | 0/5 | - |

- At about 42-45 % share, short pieces with short gaps step (s, m) and long pieces with long gaps do not (g): the
  step does not follow the share. At the same 3.0-3.4 ms gap, none of the three sizes steps, whatever its share
  (20-50 %) and piece length (0.8-2.8 ms). On DIRECT the idle gap's length predicts the step better than the share
  or the piece: every DIRECT run with gaps of 3.0 ms or more stayed low (55 runs in this lane: 35 here, 20 earlier;
  40-60 loop frames of 0.8-2.8 ms each), and every run with gaps of 2.1 ms or less stepped; at 2.2-2.3 ms (the
  earlier gap1 arm) two of three passed 2.8 GHz and fell back, and all three wandered at 2.2-2.7 GHz. The time to step is not set by the gap alone
  (m with 1.5 ms gaps took 65-77 ms; g with 2.1 ms gaps 36-63 ms). The expectation held, and so did its second
  branch (s and m stay low at 3.3 ms gaps).
- What E3 cannot separate: at a fixed share, piece length and gap length change together; with the gap binary the
  two cannot be varied independently (no spec can alternate frames and gaps, and a host edit was not allowed). The
  other lane's plain-Vulkan burst probe can.

### E4. The game-route host, from the sampler alone (`gr`, 5 runs of 60 frames; also the earlier runs)

- Firmware activity over the last 150 ms before `times.txt` appeared (inside the frame loop): `gpu_busy` median 72 %
  (67-78), `gfx_act` 70 % (33-77), card awake (`pp_dpm_sclk` > 1 GHz) in 97-100 % of samples, at 1.61-1.76 GHz;
  never a step. Earlier game-route runs with the same window: `gpu_busy` 30-82 % (22 runs: 60-frame runs 45-82 %,
  300-frame runs 30-73 %).
  Host frame (dispatch, submit, wait) median 3.54-3.74 ms.
- Measured the same way, the own-buffer DIRECT arms read 95 % (1 ms gaps, steps), 73 % (2.1 ms, steps), 69 %
  (3.0 ms, never), 50 % and below (longer gaps, never). So by the firmware's reading the game route sits between
  the DIRECT arms that step and those that do not, and on the queue (DIRECT) where that reading is closest to the
  kernel share. Its kernel share itself cannot be recovered from the sampler: the card stays awake through the
  read-back between frames, and the length of the awake stretch before `times.txt` gave 6.6-13 ms per frame in
  different runs, which cannot be told apart from creation-phase activity. Not resolved. The expectation ("near the
  DIRECT no-step arms of gaps 2-4 ms") held in the sense that it did not step at a firmware reading inside the range
  where DIRECT runs went either way.
- One 300-frame run is about 1-4 s of low-duty work at about 1.75 GHz; the frame loop never stepped (10 runs).
  Before the loop it does reach the high clock once: in all five `fix-e/gr` runs the sampler shows one stretch of
  10 to 26 ms above 2.8 GHz (about 3.23 GHz, card 100 % busy) about 3.4 s into the sampler's window, during the
  creation phase, after which the clock falls back to about 1.8 GHz (lead's recount from `clk.txt`). Every
  own-buffer run has the same kind of stretch (8 to 18 samples above 2.8 GHz). This is direct support for "a long
  enough unbroken load raises the clock", and it means the card is not coming from rest when the loop starts.

### A card state found on the way: two low levels

Every arm's runs sit at one of two low levels, about 100 MHz apart, and the level persists over consecutive runs
for minutes, whatever the arm (`fix-e`, run order in `series/fix-e.log`): DIRECT g 1.65-1.67 or 1.75-1.80 GHz,
COMPUTE g 1.62-1.64 or 1.73-1.74, sizes m and s 1.70-1.71 or 1.80-1.83 (gauge). In order of the 95 own-buffer runs:
H x6, L x3, H x17, L x10, H x32, L x4, H x3, L x20 (the last round entirely low). The sampler sees the same two
states independently of the gauge: `gpu_metrics` clock 1,701-1,715 MHz with graphics voltage 755-759 mV in the low
state against 1,816-1,835 MHz and 760-767 mV in the high state; edge temperature (51-55 C), fan (stopped), fabric
(2,016 MHz) and SoC clock (1,280 MHz) the same in both. The first pass's two pair-B game-route "clock groups"
(1.61-1.63 against 1.74-1.75 GHz, 4-6 mV apart) are the same pair of states. What switches it is not known
(nothing was changed between runs; INFERRED candidates: a firmware voltage-frequency adjustment, or a desktop
client's load on the card; not tested). It moves every low-clock time in this lane by about 6 %.

## Surprises

1. The clock on this route: frames start at 1.65-1.80 GHz, not at the 2.7-2.8 GHz of the study's HIP probes, and a
   second of real idle before them changes nothing. The step needs about 15-38 ms of frames with short idle gaps;
   on DIRECT, idle gaps of 3 ms stop it at any share, while on COMPUTE gaps up to 6.7 ms still allow it (E1-E3).
2. The low level comes in two states about 100 MHz (and 5 mV) apart that persist for minutes and switch between
   runs with nothing changed (E section).
3. The firmware reads the COMPUTE queue as 80-99 % busy when kernels fill 27-36 % of the time (E1).
4. Hotspot 63 -> 84 C in about 66 ms at an edge reading of 52-53 C; reported power up to 459 W against a 317 W cap;
   `TEMP_HOTSPOT` and `PPT1` flagged while the clock held 3.16-3.19 GHz.
5. After frame 0 the host pauses 60-145 ms (output download and file write); the first write-up measured the step
   from frame 0 across that pause.
6. On the DIRECT queue three small dispatches are followed by a gap of about 24 us before the next record (an early
   end timestamp or a real idle bubble: not separated); COMPUTE loses ~49 us at each of the two timing marks (G3). RDNA 3's two-speed
   split is absent.
7. Model 2's post kernel is 1.62 times slower than on the RX 7900 XT with the same source, while everything else
   got faster (G1b).
8. One network kernel's time wanders by 40 % between identical frames at its default register count (G4).
9. ACO's waves-per-SIMD figure for wave64 is twice the study's measured register rule (G4, G1 variants).
10. At create the runtime test-launches the small "wide" stages dozens of times and keeps different group counts on
   the two hosts (20/8/20 own-buffer, 24/6/24 game route).
11. The Wine process opens both cards' device nodes while it enumerates adapters, so "which node it holds" does not
   tell which card ran the work; the kernel's per-client engine time does.
12. `gpu_metrics` changes only every 6 ms, while `pp_dpm_sclk` gives fresher readings of the same firmware data.

## Checks and deliberate breaks

| check (known answer) | result | deliberate break (must FAIL) | break result |
|---|---|---|---|
| guards refuse forbidden variables / options / paths (`gr.sh` as it is now; rerun in the fix round, `route/fix/guard_test.sh` -> `route/fix/guard_test.txt`) | dry-run controls pass: a clean job, a job passing the runtime DLL as an argument (`route_host`), an `.exe` whose working folder is in G | `DXIL_SPIRV_CONFIG`, two of the runtime's gfx11 switches, `RADV_PERFTEST`, a spec with the softmax-test option, an option with the approximation switch, a path into the upscaler, a diagnostic key without its label | all 8 refused (exit 8/9) |
| game-process guard, before the lock and (since the fix round) under it | a process named `reaper` and a foreign `fakegame.exe` (copies of `/usr/bin/sleep`, working folder outside G) refused before the lock (exit 6); the foreign `.exe` refused under the shared lock by `gr.sh --inner` before any sampler started (exit 6, run folder left empty) | the same cases | FAIL as intended. The first write-up's `gr.log` also holds a refusal naming the runtime DLL under `bin/A/`, which the current guard cannot produce (it greps only `.txt` arguments): the guard was narrowed after that test; the rerun above replaces those results |
| thermal stop on the in-run hotspot maximum (fix round) | known answers of the reader: `g1/od_n_r6` 84, `mod2/mod2_n_r1` 84, `clk/gap8_r1` 54, a missing file "none" (= stop); the follow-up series ran with the 88 C limit and did not stop | series `fix-guard`, 2 rounds of a light arm with `GR_HOT_STOP=50`: run 1 in-run max 56 C | STOP written, round 2 not run (FAIL as intended; STOP file kept as `runs/fix-guard/STOP.from-hot-stop-test`) |
| adapter, every run (profile field, runtime log, or DRM engine time per PCI device) | RX 9070 XT in all runs; iGPU engine time 0 | first version (device node held) | failed on a real run: both nodes are held |
| DIRECT queue switch logs "command queue DIRECT" | present in DIRECT runs | COMPUTE runs | line absent |
| sampler `current_gfxclk` against starred `pp_dpm_sclk` | withdrawn in the fix round: both are the firmware's metrics (gpu_metrics refreshes every 6 ms, `pp_dpm_sclk` reads fresher values; they agree exactly in 34 % of busy samples), so this is not a second source | - | - |
| profiler timestamps against host time over 49 frames | ratio 1.0004-1.0020 (`calib/timer_calib.txt`) | - | - |
| census against the study's kept listings (16 WMMA + 4 loop instructions; splat adds 4 moves + 1 delay) | PASS both (`calib/census_calib.txt`) | expectation with 15 WMMA | FAIL as intended |
| revision 2 DXIL build against revision 1 hashes: only the post kernels may differ | 88 same, 12 post kernels differ: PASS (`calib/gate_rev2.txt`) | one byte of `m1_out_c.dxil` flipped | caught |
| pso_sweep listings equal the code the real runs compiled | 16 of 16 (own-buffer), 6 of 6 (game route) (`calib/listing_gate_*.txt`); Model 1 kernels only: Model 2's and the variants' listings are not gated | another kernel's listing under `m1_out_c`'s name | FAIL as intended |
| G5 values found verbatim in the study | 28 of 28 (`calib/g5_pass.txt`; fix round `calib/g5_pass_fix.txt`: source lines now verbatim with their line breaks, numeric columns numbers only, ranges in `latency_low`/`latency_high`) | VALU latency 4.60 (RDNA 3's figure); fix round also text in a numeric column (`--break-num`) | both FAIL as intended, table untouched |
| output bytes: pairs 1, 2, C, DIRECT and COMPUTE, profiler on and off, variants | identical (model 1 own-buffer f0/f1, game route f0-f2, 15 variant runs) | - | - |
| synthetic model 2 | `synthetic_tensors(seed=1)` written as raw equals the copied case's model files, 128 of 128 files | - | - |
| window and sound watcher, every run | 162 watch files, no alert; no wayland/X11/pulse/pipewire socket (fix round: every run again, no alert) | - | - |
| GPU-to-host time alignment (fix round) | per run, offset bracket 0.8-0.9 ms (DIRECT), ~2.9 ms (COMPUTE), no drift inside a run; last fence wait ends 0.2 ms before `times.txt` appears | - | - |

Plan checks not done: the plain-Vulkan calibration probes (streaming write, FMA kernel, classification of known
controls) belong to lane B; the per-arm "classification script" was replaced by the plan's verdict rules applied by
hand above.

## How to rerun

All from `G/route` (each GPU job goes through `gr.sh`, which takes the shared lock and waits for a cool, idle card):
- `PAIR=1 ./series.sh <name> <rounds> <arm> ...`; arms in `series.sh` (`od_p od_n oc_p oc_n gr grt` for G1/G3,
  `gap0..gap16 xfer` for the clock, `mod2_p mod2_n mod2gr mod2grt` for model 2, `od_ps od_pm` sizes, `aco1_* aco2_*` for ACO,
  `vbase v_w32 v_l8 v_l16 v_l32` variants, `kdump gdump` listing gate). `chain1.sh`, `chain2.sh` hold the orders used.
- Analysis: `analyze_m1.py <prefix> <run dirs>` (timelines, distributions), `analyze_gr.py`, `analyze_clk.py`,
  `census.py`, `wmma_sites.py`, `listing_gate.py`, `split_dump.py`, `make_g5_classes.py`, `make_g5_tables.py`.
- DXIL: `build_rev2.sh <out>` (CPU; `SRC=G/srcvar` for the variants; it executes the upscaler's dxc shim).
- Fix round: arms `fgd<ms>` / `fgc<ms>` (DIRECT / COMPUTE, gap host, size g, 2 spec frames + 40 loop frames),
  `fmd<ms>` / `fsd<ms>` (sizes m / s, DIRECT), `fpz` (1 s wrapper pause after "frame 1 done"); series
  `PAIR=1 WR_TIMEOUT=120 ./series.sh fix-e 5 fgd0 fgc0 ... gr` (the order used is in `series/fix-e.log`).
  Analysis: `fix/tl.py` (time axis, per-frame CSV), `fix/analyze_fix.py` (E1-E4), `fix/thermal_scan.py`,
  `fix/exponent.py`, `fix/queue_sum.py`, `fix/queue_s2s.py`, `fix/size_full.py`, `fix/metrics_cadence.py`; guard
  tests `fix/guard_test.sh`.

## Not measured, and why

- G2 isolation lists, G3 draw load and tiny-dispatch probe, G6 burst probe in plain Vulkan: lane B's work.
- LDS bank-conflict model and padding variants for the network kernels and the model 1 mix softmax: cut (first cut in the
  brief's order after the DLL arms, which were done).
- Store / load-removal variants of the output kernel: not built (they change the next frame's data; see G1).
- Model 2 with revision 2's post kernel: not timed (the model 2 runtimes here embed revision 1's set, which is the RDNA 3
  like-for-like; a disk-kernel model 2 runtime was not set up). Fusion arms of model 2: not run.
- Per-frame GPU times on the game route: the runtime's profiler is off when it records into the caller's list
  (`profile_begin` returns early when `external_` is set, the D3D12 device source), so the game route gives 300-frame stage
  means and maxima and host frame times only.
- Whether a real game shows the low-clock state: needs the game and the owner.
- Why the COMPUTE queue loses 49 us at timing marks, and why model 2's post kernel is slower than on RDNA 3: open.
- Output storage: G holds about 26 GB by apparent size (19 GB on disk; the lead's `du`, 23:10 UTC) after the fix round (83 GB free on the disk), mostly `f*_out.bin` images kept
  for the byte checks (`route/runs/*`; the fix round's `runs/fix-e` adds about 9 GB of them). Nothing was deleted;
  the lead may want to remove them.

## Changes to the draft plan

- Reference DLL pair: 1 (the owner's pair from the game folder snapshot), not GE-Proton11-11's (A), which
  cannot create the kernels. 2 second arm, C one control.
- Players' queue: the game-route host `route_host.exe` (the upscaler developers' suggestion) instead of the DIRECT switch alone;
  the switch was still used for per-dispatch timing on the own-buffer path.
- The game route runs the `_direct` input and output kernels, not the plan's buffer forms; both were timed.
- The plan's 192 SIMDs for the RX 7900 XT are the XTX's; the XT has 168.
- Thermal rule read as continuous GPU-busy time (lead's decision). The first write-up's "hotspot never above 69 C in
  any sample" was false: the in-run maximum was 84 C, and the stop that should have caught 85 C could not work (see
  "Hotspot, power and throttle bits"). What was done under the rule: the longest stretch of `gpu_busy` at 70 % or
  more was 334-344 ms in the six `g1/oc_p` and two `g1s/oc_p` runs (COMPUTE, profiler on), whose loop frames hold
  about 112 ms of kernel time in a 318 ms loop (`g1s/oc_p_r3`); the gap0 clock runs hold 109 ms of kernel time in a
  184 ms loop; the game-route runs of 300 frames (about 1.1 s at 3.6 ms per frame plus read-backs) were treated as
  low duty.
- Idle-gap arms came from the existing gap host (`gap_host.exe`, `GAP_MS`), no host edit.
- Model 2 on the game route needed a model file: generated from `synthetic_tensors(seed=1)` with its
  synthetic-model script (from the copy in G), after checking its raw form equals the copied model.
- The runtime's kernel-substitution switch was replaced by per-variant kernel folders (it does not reach model 1's launches).

## Questions to the upscaler side

See `G/route/QUESTIONS.md` (question 3 answered; nothing open).

## Corrections after review (2026-11-17)

The independent review returned FIX-PASS. Each finding, what was done, and where the figure now comes from. Where
my reading of the raw files differs from the review's, both are given. Tools added in the fix round are under
`route/fix/` (all new; no file of the upscaler project changed, no host built or edited).

1. **Hotspot, power, throttle (fixed).** The statements "No throttle bit was set in any sample of any run; hotspot
   max 69 C (gap 0 runs), power reported 47-87 W", "hotspot never above 69 C in any sample" and the summary's
   omission were false. From every run's `clk.txt` (`fix/thermal_scan.py` -> `ana/fix_thermal_all.txt`): in-run
   hotspot maximum 84 C (`g1/od_n_r1`, `r3`, `r6`, `mod2/mod2_n_r1`), power up to 459 W (`mod2/mod2_n_r1`), throttle words
   non-zero in 57 of 162 runs; legacy 0x2 / 0x8002 always paired with independent 0x1000000000 (`TEMP_HOTSPOT`) /
   0x1000000002 (`PPT1` + `TEMP_HOTSPOT`). The review's reading reproduces exactly; one detail differs: the 421
   flagged frames out of 500 are the pair 1 file (162 of 200, `ana/g1p1_od_n_frames.csv`) and the pair 2 file (259 of 300,
   `ana/g1p2_od_n_frames.csv`) together, not the pair 1 file alone. New section "Hotspot, power and throttle bits" with
   the per-arm table, when the bits appear, one run's timeline (`g1/od_n_r6`), and what it means for the full-clock
   figures; note added under the G1 table.
2. **Thermal guard (fixed and tested).** `gr.sh` now decides the stop on the in-run maximum of the run's own
   `clk.txt` (88 C, or no reading, writes STOP and ends the series); the old after-run reading is still logged.
   Control: a series with the limit at 50 C stopped after its first run (56 C) and did not start the second. The
   game-process check now also runs under the lock (refusal tested with a fake process under the shared lock). The
   guard tests were rerun on the current `gr.sh` (`fix/guard_test.txt`: 15 PASS, 0 FAIL) and replace the first
   write-up's results; the old `gr.log` refusal naming the runtime DLL under `bin/A/` came from an earlier guard
   (it grepped every file argument) that was narrowed after its test. The text now says that all 162 first-pass
   runs were made without a working stop. Before-fix copies: `route/fix/gr.sh.before-fix`.
3. **Clock timeline (fixed).** Rebuilt on the sampler's time axis with frames numbered from the profile
   (`fix/tl.py`). The review's figures reproduce: frame 1 starts 108.6 ms after frame 0 in `clk/gap0_r1`; from
   frame 1 the step takes 50.0, 32.7, 45.6 ms (gap0 r1-r3) and 85.9, 94.1 ms (gap1 r2, r3; gap1_r1 never passes
   2.8 GHz but rises to 2.75 GHz and wanders). New: measured from the loop start (frame 2, after a further 10-15 ms
   pause), stepping runs step after 15-38 ms (48 DIRECT g runs), close to but about twice the study's 14-15 ms.
   Creation phase described (card awake in a median 32-53 % of samples over the 2 s before frame 0, at 1.82 GHz).
   The lead's rerun is included (5 more gap0/od_p runs, 8 no-step runs).
4. **"Busy more than about half" (fixed: rewritten as observations and an open question, then tested in E1-E4).**
   Low-clock shares recomputed with one definition (sum of frame spans over the loop frames before the rise):
   gap0 66-72 %, gap1 54 %, gap2 45-48 %, gap4 35 % (review: 70, 55, 46, 35 %); COMPUTE profiler arms about 40 %
   (39-41 %; the review's 39 % used median span and gap) and they step. "The game-route host sits at that
   threshold" withdrawn. The two pair-B clock groups are named (`g1/grt_r1`, `r2` at 1.61-1.63 GHz, with 4-6 mV
   lower graphics voltage; unexplained). The in-game 1.62 ms (full-clock time) is stated beside the caution for
   players. The follow-up experiments then tested what decides the step (section E1-E4): the queue matters at equal
   share and equal gap, on DIRECT the idle gap's length predicts the step better than share or piece length, a
   second of idle changes nothing, and the game route's firmware activity sits where DIRECT runs went either way.
   They also found the two persistent low-clock states, which explain the pair-B "clock groups".
5. **Clock evidence (fixed).** `pp_dpm_sclk` and `gpu_metrics` shown to be one firmware source (the 6 ms refresh);
   the "median difference 0 MHz" check withdrawn. The gauge restricted to size g in `analyze_clk.py` (skips other
   sizes; before-fix copy kept) and in the text. The 0.6 % and 3.6 % are both stated and explained (which sampler
   values are compared: 1,784 vs 1,821 MHz). `pp_power_profile_mode` BOOTUP_DEFAULT added to the conditions.
6. **Clock exponent (fixed).** The clean same-host test (gap0 frames 25+ against gap2/4/8 frames 5+, `runs/clk`)
   replaces the 12-run fit as the evidence (`fix/exponent.py` -> `ana/fix_exponent_clk.txt`). Same numbers as the
   review: `m1_out_c` 497.6 / 895.0 us (1.80), gauge 1.77, input 1.69, `m1_out_a` 61.6 / 64.7 (1.05), clock
   ratio 1.72-1.77. "969 cycles per wave" for `m1_out_a` is now marked as not a fixed count; "not memory-bound"
   narrowed to "not bound by memory bandwidth or fabric", with the clock scaling as the evidence.
7. **RX 7900 XT comparison (fixed).** Ratios now minimum against minimum: `m2_net_04` 0.34, sum 0.348 (sum of
   per-kernel minima; 0.353 with the per-frame minimum, the review's figure), post kernel 1.62 (1.63 before). The
   wave32 result now cites RDNA 3's own measurement (the I/O profile note, line 132, quoting the output kernel source
   lines 25-26: wave32 721 against 640 us; the output kernel source file itself is not in the revision 2 archive and was not
   read). Sources for 2.75 GHz (the confirmation note, line 81), 84 CUs (the takeaways note, line 29; 168 SIMDs
   INFERRED as 2 per CU) and 128 SIMDs (study) added. The review cited "the I/O profile note, lines 25 to 26"; those
   lines are in the output kernel source, quoted at line 132 of that note.
8. **COMPUTE minus DIRECT (fixed).** Reconciled per frame (`fix/queue_sum.py`, `fix/queue_s2s.py`): span difference
   152 us (median of the per-frame differences) against dispatches +142, barriers +5, gaps +13 (medians of the
   parts, which sum to 160: medians of parts need not add up to the median of the whole; per frame the parts do
   add up once all gaps are counted).
   The review's 235 us double-counted because DIRECT has about 75 us of gaps of its own after three small
   dispatches, where COMPUTE instead shows longer dispatches. Each figure is given with its definition (parts sum
   +0.183 ms profiler off, +0.195 on; span +0.152; start-to-start sum +0.177). New finding: the per-kernel "+68 %"
   (`m1_net_03`) and "+31 %" (`m1_net_10`) on COMPUTE are mostly where DIRECT writes its end timestamps
   (start-to-start: +3.6 % and +6.7 %). The G2 statement "7-9 us in gaps per frame" was also wrong (82 us when
   gaps before barrier records count) and is corrected.
9. **Size sweep (fixed).** Full-clock frames only (`fix/size_full.py`): s 8.0, m 128.2, g 492.6 us; 19 (s) and
   21 (m) of 200 timed frames were at the low clock (the review's "low-clock minority" confirmed). Waves at s:
   375, not about 480. The third point falls 18 % below the m-g line (review: about 26 % below a prediction of
   about 11 us; with full-clock medians and 375 waves the prediction is 9.8 us). "Linear in waves" and the
   H-dispatch-size verdict withdrawn as tests.
10. **Labels (fixed).** Wave size and spills: READ (listings). Waves per SIMD in the variant table: computed from the
   study's rule, not observed. "Limited by instruction issue": INFERRED, with what would separate it from exposed
   latency. Value numbering: occupancy is one candidate (INFERRED); the code differs by more than registers
   (instructions 4,443 -> 4,960, copies 337 -> 969, VALU 2,622 -> 3,110; `runs/dump/M_*/isa/index.json`), 2 runs
   per arm. "In short" rewritten: MEASURED and INFERRED parts separated; "decides the upscaler's speed more than
   anything else here" removed.
11. **Hidden or wrong items (fixed).** The busy rule is stated in the body (longest `gpu_busy` >= 70 % stretch
   334-344 ms in the COMPUTE profiler runs; 112 ms of kernel time in a 318 ms loop in `g1s/oc_p_r3`, measured over
   loop frames 2-61; the review's "414 ms loop" probably included the spec frames and their pause; 300-frame
   game-route runs as low duty). `build_rev2.sh` executes the upscaler's dxc shim (it points the build at the upscaler's
   shim and runs the upscaler's DXIL build script from the archive copy); its header comment says "read-only", meaning
   nothing in the upscaler was written; "executed" is the right word. `gap_host.exe` is described as the upscaler's
   prebuilt gap host. The Model 2 "3 setup frames" sentence is corrected (the 18 are low-clock timed
   frames). The listing gate is stated to cover Model 1 only. G5 table: the two source lines (`wmma` FP16 row,
   `lds` row) are verbatim again with their line breaks; the launch row's "2.13 us / 3.36 us" moved out of the
   numeric column (into the note); the three latency ranges moved into new numeric columns `latency_low` and
   `latency_high`; `make_g5_classes.py` now refuses text in a numeric column (`--break-num` fails as intended).
   Surprises extended (host pause, COMPUTE step, throttle readings, timestamps on DIRECT).

## Corrections after the re-check of the fix round (2026-11-17, by the study lead; text only, no new run)

A second independent reader went through the fix round (verdict FIX-PASS). What changed here, each point checked
by the lead against the raw files where a figure is given:

- E1: the COMPUTE arm with 7.4-7.8 ms gaps rises part of the way in all five runs, so "0/5" and "stops between 6.7
  and 7.4 ms" hid a graded response; the rule for "stepped" is now stated; the unreconciled 3.4 ms of the COMPUTE
  frame (empty to the profiler, busy to the firmware) is named as the main unknown.
- Correction 8: "152 = 142 + 5 + 13" did not add up (160); stated as medians. The 24 us on DIRECT is a gap with two
  readings, in "In short", G3 and "Surprises".
- "Frames with a throttle bit were not slower" withdrawn as evidence: no full-clock frame without a bit exists.
- E4 and the creation phase: every run reaches about 3.2 GHz once for 10 to 26 ms before the loop.
- What the fix round's runs did against the 0.2 s rule: by the firmware's busy reading (70 % or more) the five
  `fgc0` runs have stretches of 206 to 235 ms and `fgc2` up to 276 ms above 2.8 GHz; by kernel time they are about
  90 ms. Hotspot at most 65 C in those runs, start temperatures 52 to 56 C. The rule was read as kernel-busy time.
- The guard tests: "15 pass" is 14 scripted lines plus one thermal line written by hand; `runs/fix-guard` also
  holds one real run (`fpz_r1`) made after the STOP file was moved aside, which the text did not mention; the run
  count of 102 includes it. Two weaknesses of the guard are left as they are and recorded: a sampler that dies
  after a few rows would pass the in-run stop, and process names are cut to 15 characters, so a long game
  executable name would slip past the game check.
- Storage figures corrected. Nothing was deleted.
- Not in the tables: the thermal table leaves out `ctlC` and `size/od_p` (73 C, 46 flagged samples).
- The two low clock levels: the re-checker confirms them in the raw samples (about 1,701-1,715 MHz at 755-759 mV
  and 1,816-1,828 MHz at 760-764 mV, three runs at 1,834 MHz and 767 mV, a possible third level), with switches
  after both 10 s and 26 s between runs and no pattern in arm or temperature. The same ten or so desktop clients
  hold the card in every run, in both states.
- What the sibling lane adds (`G/probes/measurements/gaming-probes.md`, plain Vulkan on RADV, the clock read by a
  cycle counter inside the shader): the low start (about 1.70 GHz) and the step after a few tens of ms are there
  without Wine, vkd3d-proton or Model 1, while the study's HIP probe in the same job list starts at about 2.7 GHz
  and is at full clock after 14 ms. So the low start belongs to the Vulkan submission path on this card, not to
  this route's layers (which part of that path is not known). It saw the same two low levels. Its compute-queue
  bursts with 6.3 ms gaps stayed low where this lane's COMPUTE arms still step at 6.5-6.7 ms gaps, which suggests
  the COMPUTE effect seen here belongs to this host and runtime path and not to compute queues in general; that
  lane's bursts were a light load, and it is being repeated there with a dense one.
- The time in the header of `fix/EXPECTATIONS.md` (22:35) is later than its file time (22:33:51 UTC, before the
  first run at 22:34:04).


## Follow-ups from the upscaler work (2026-11-17 to 2026-11-19)

Added 2026-11-19. These come from the upscaler project's own work on the same card and route (Wine, pair 1, RADV),
done after this write-up; the runs and their notes are in that project's work folders, not here. Labels as above;
"MEASURED (upscaler)" means measured there, with interleaved arms and byte-identical output checks. Clock profiles are
named where its notes record them. Each item answers or narrows an open point of this file.

- **RADV's CU or WGP choice for these kernels** (G4 "Not done"): READ, Mesa `radv_should_use_wgp_mode`: compute
  shaders that use LDS run in CU mode ("LDS is faster with CU mode"). So two 256-lane wave32 workgroups fit on a CU only
  at 192 VGPRs or fewer (8 waves on its 2 SIMDs), and a kernel just above 192 loses half its workgroups per CU. INFERRED
  from the rule and the listings.
- **`m2_net_04`'s register count was the cause** (G4: "occupancy is one candidate, not the shown cause"): ACO's VGPR
  demand for it before scheduling was 194, two above that step, so it was compiled for one workgroup per CU and grew to
  256. A source change that only moves a table load one step later (no arithmetic changed) brings it to 192 under the
  normal ACO, and its time from 260 to 194 us, steady from frame to frame (MEASURED (upscaler), busy-sample clock
  3.07-3.14 GHz, profile not recorded). That is the `novn` arm's 206-210 us without `novn`'s other code changes, so
  occupancy was the cause.
- **Why `m2_post_c` is slower than on the RX 7900 XT** (G1b, "Not measured": open), narrowed: two diagnostic arms that
  skip the history samples (wrong output, timing only) put the history look-ups at about 212 of 596 us; without them the
  kernel takes about 383 us, close to the RX 7900 XT's whole-kernel 378 us. Each wave runs up to four dependent
  12-texel history reads per step. INFERRED: the RDNA 4 slowdown sits in those dependent reads; the RX 7900 XT has not
  been run with the same diagnostics. About a third of the remaining gap is workgroup count against WGPs (768 large
  workgroups, 4 per WGP; 32 WGPs here, 42 on the RX 7900 XT; INFERRED). Reading the history taps as 8-byte pieces
  (`Load2` at an address with the low 3 bits cleared, an identity because every offset is a multiple of 8) gave
  616 -> 596 us (MEASURED (upscaler)). Overlapping the two history candidates needs 132-144 VGPRs against 84 (round 3 of
  that work times it; not settled).
- **Load width through dxil-spirv and ACO** (new): an HLSL `RWByteAddressBuffer.Load4` is known to be 4-byte aligned
  only, and on this route each one compiles to four `global_load_b32` with four address additions (READ, listings). When
  the address is in fact 16-byte aligned, clearing its low 4 bits (`(base + off) & ~15`) lets ACO emit one 16-byte load
  instead: model 1's narrow network stages lost 10-16 % of their instructions (READ, listings), VGPRs unchanged, and the
  model 1 frame's kernel sum went from 1540 to 1516 us at 1706x960 -> 2560x1440 (-24 us, -1.6 %; 240 interleaved frames
  per arm, busy-sample clock about 2.97 GHz, overclocked "Default" profile; MEASURED (upscaler)). This is the D3D12
  counterpart of the masked-address result for HIP in `compiler-and-hazards.md`. Open: in other kernels Mesa did not
  merge masked 8- or 16-byte loads (the Swin kernels', round 1 of that work); why is not known.
- **GE-Proton 11-12** (pair A's row): its own vkd3d-proton creates exactly the same kernels as 11-11's: 42 of the
  upscaler's 157 compute kernels; 85 are refused with `0x80070057` and 30 crash the process (on both). MEASURED (upscaler), `pso_sweep` per kernel, system wine 11.21, only the D3D12 pair changed per arm.
  Pair 1 creates 155 of 157 (two reference-route kernels crash dxil-spirv on every pair).
