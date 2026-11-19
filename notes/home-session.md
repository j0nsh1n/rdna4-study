# Home session: tests that need you at the machine

Everything here is ready to run. Each test is one command in a terminal in the study folder
(`cd` to the root of this repository). A small recorder runs during every test. It reads the card's sensor files and never
opens the card itself; at its start and end it also runs `rpm` (the Mesa version), `date` and
`lact cli profile get`, all read-only. Results go into one folder per sitting, with one summary file that collects
every result line.

The recorder can still change what it measures a little. At rest, two recordings saw the card awake 3.3 % of the
time when read every 2 ms and 0.8 % when read every 20 ms. The likely reason (inferred, not checked) is that a read
of the card's metrics table makes the driver ask the card for fresh numbers once its 1 ms copy is old. So for the
capped and light game situations, where the question is how the card rests between frames, a second recording at
20 ms is required (Part A).

Nothing changes a card setting unless you start it with an "OK" flag for that one test. Without the flag, a test
that needs a setting changed stops and tells you what to do. With the flag it checks that the LACT profile exists,
switches to it, checks the changed setting and that nothing else differs from Testing, runs, and puts the card back
on your "Testing" profile. If it cannot confirm that, it prints **RESTORE FAILED**, the settings that differ, and
what to do. Only one step runs at a time: a second one refuses to start. Steps that put load on the card (Parts B,
C and D) also refuse to start while a game is running: close the game first.

## The short list (about 45 minutes)

1. Part A, Model 1 only: the three game situations without the upscaler, then with it, with the 20 ms repeats of
   the capped and light cases (about 30 minutes, including the launch-option change and a game restart).
2. Part D1, the memory clock check (about 6 minutes, including making one LACT profile).
3. Part B1, the power mode test (about 7 minutes, including making one LACT profile).

## Before you start (1 minute)

```
scripts/study-home.sh check
```

It shows the card's settings, the LACT profiles, whether another step or a game is running, and the programs that
use the card. "stock Testing settings: yes" means you can start. If it says NO, open LACT, choose the profile Testing, and
run the check again. Run it once at the start of each sitting: the memory clock test (D1) needs its record.

## Making a LACT profile

The labels below are LACT 0.10.1's own English text (read from the program on this machine). Where they sit on the
screen was not checked.

1. In the **Settings Profile** list, choose **Add new profile**. In **Create Profile**, type the **Name**, set
   **Copy settings from:** to **Testing**, and press **Create**.
2. Make sure the profile list now shows the new profile: changes go to the profile that is chosen.
3. Change the one setting named in the step, press **Apply**, and in **Confirm Settings** ("Do you want to keep
   the new settings?") press **Confirm** before the countdown ends.
4. Choose **Testing** again in the profile list (except for Home-Mem, D1) and run `scripts/study-home.sh check`.

The profiles:

| Profile | Where | The one change |
|---|---|---|
| Home-Power | **Overclocking** page | **Performance Level**: **Manual**, then **Power Profile Mode:** 3D_FULL_SCREEN (LACT shows the mode only with Manual) |
| Home-High | **Overclocking** page | **Performance Level**: **Highest Clocks** |
| Home-Fan100 | **Thermals** page, **Fan Control** | the **Static** tab, **Static Speed (%)** 100 |
| Home-Mem | **Overclocking** page | **Maximum VRAM Clock (MHz)**: 1409 |

If a profile differs from Testing in anything else, the step names the setting, switches back at once and runs no
load on it.

## Part A. A real game (most important, about 30 minutes for Model 1)

**What it tells us.** Our test programs found that the card starts short bursts of game-style work at a low
clock (about 1.7 to 1.8 GHz) and only goes to full speed (3.1 to 3.2 GHz) after 15 to 38 ms of almost unbroken
work. With pauses of 3 ms or more between frames it stayed low, and the upscaler pass then took about 2.8 to
3.6 ms instead of 1.6 ms. Those were test programs on an idle card. This part checks whether a real game sees it.

### A1. Without the upscaler (about 13 minutes)

Play Cyberpunk 2077 with the upscaler off. Hold each situation steady for the whole 2 minutes. Use the same game
settings for all of them.

The frame limit for the capped case comes from OptiScaler, not from the game: at your settings the game's own
limiter leaves the card busy, so it creates no rest between frames. Set OptiScaler's frame limit (`FramerateLimit`
in OptiScaler.ini, or in its overlay) to about 30. Turn it off again for the uncapped and light cases.

| Situation | What to do in the game | Command (start it, then play) |
|---|---|---|
| Uncapped | Normal play or a busy street, no frame limit | `scripts/study-home.sh game uncapped` |
| Capped | The same place, OptiScaler's frame limit about 30 | `scripts/study-home.sh game cap30` |
| Capped, 20 ms (required) | The same, right after | `scripts/study-home.sh game cap30-slow --period-ms 20` |
| Capped at 45 (optional) | The same place, OptiScaler's frame limit 45 | `scripts/study-home.sh game cap45` |
| Light | A menu or a quiet indoor scene, no frame limit | `scripts/study-home.sh game light` |
| Light, 20 ms (required) | The same, right after | `scripts/study-home.sh game light-slow --period-ms 20` |

Each recording takes 2 minutes and prints "done" at the end. Ctrl-C ends one early; it still saves. Closing the
terminal also still saves it. The game step starts nothing on the card, but it does not start while another step
of this kit is running. If it finds no running game it says so (it looks for the game's traces in the system's
process list, since under Proton no process carries the game's name); start the game first.

**How to read it.** After each recording the script prints a few lines. `AWAKE_SHARES` gives the share of the
awake time at 1.65 to 1.9 GHz (the low clock) and above 2.9 GHz (full clock). `HIGH_RUNS` and `LOW_RUNS` say how
often and for how long the clock stayed high or low. `HOTSPOT_C`, `MEM_C` and the `THROTTLE` line show heat and
limits.

- The clock shares alone do not show a problem. A capped or light scene may rightly rest at the low clock: the
  card has spare time. A problem is the low clock while the frame or the upscaler needs the time (A2).
- If the 2 ms and 20 ms recordings of the same case differ clearly, the recorder's own reading changed the result;
  then trust the 20 ms one.

### A2. With the upscaler (about 17 minutes per model)

The steps below are as given by the upscaler project, not tested here.

1. In Steam: Cyberpunk 2077, Properties, Launch Options. Paste one line: the upscaler project's own launch options
   (its runtime options variable, set to switch its timing log on, with model 1 chosen for the model 1 runs;
   the exact line comes from the upscaler project and is not in this repository), plus
   `VKD3D_SHADER_CACHE_PATH=<a shader cache folder for this card>` and `WINEDLLOVERRIDES="dxgi=n,b"`, then `%command%`.
   - Model 1: the line with model 1 chosen.
   - Model 2: the same line without that choice. Model 2 is the default.
2. Keep the game folder as it is now (your own d3d12 files). Do not set any other test switches. Ray
   Reconstruction off. Use the same game settings in all situations.
3. In the game, use the same upscaler entry you normally use (you use DLSS); keep it the same for every
   recording. If OptiScaler's overlay shows "FSR 2.1.2", the upscaler did not start and FSR took over: that run
   does not count (it writes no timing lines). The overlay saying "model 2" even for model 1 is expected.
4. Run the situations below, 2 minutes each, held steady. Change no resolution or quality setting during a
   recording (that restarts the upscaler). When the script says "Recording done. Keep playing 5 more seconds",
   keep playing until it prints "done": the last log line arrives a moment later.

| Situation | Command |
|---|---|
| Uncapped | `scripts/study-home.sh game model1-uncapped --upscaler` |
| Capped (OptiScaler, about 30) | `scripts/study-home.sh game model1-cap30 --upscaler` |
| Capped, 20 ms (required) | `scripts/study-home.sh game model1-cap30-slow --upscaler --period-ms 20` |
| Capped at 45 (optional) | `scripts/study-home.sh game model1-cap45 --upscaler` |
| Light | `scripts/study-home.sh game model1-light --upscaler` |
| Light, 20 ms (required) | `scripts/study-home.sh game model1-light-slow --upscaler --period-ms 20` |

For Model 2, use the model 2 launch line, restart the game, and use the labels `model2-uncapped` and so on.

The script finds the upscaler's log through Steam's library list. It only reads the log and opens no other file in
the game folder. If it cannot find the log, add `--log` and the log's path. The log's path inside the game folder is set by `STUDY_UPSCALER_LOG` in `local.env` (see `local.env.example`).

**How to read it.** The log gives one line per 300 frames (about 5 seconds at 60 frames per second). The script
takes the lines written during the recording, drops the first one (its 300 frames can include some from before the
situation was set up) and says so, and prints:

- `UPSCALER model 1 (stage names ...)`: the model, told apart by the stage names of each timing line
  (the stage names and the names printed for the models are settings in `local.env`, see `local.env.example`), and
  the upscaler's self-check result (`selfcheck (self-check passed)` with the default tag).
  `selfcheck(unchecked)` (default tag) means the self-check did not run.
- one `UPSCALER_LINE` per log line: the pass time (mean, and the maximum in brackets), its stages, `total_ms` (the
  upscaler's whole share of the game's GPU work per frame), `resume_ms` (from the end of its input part to the start
  of its output part, as the game sees it) and `cpu_dispatch_ms` (processor time per frame to record its work).
- `UPSCALER pass_ms`: the mean of those lines. Below that, the recorder's clock shares for the same 2 minutes, and
  the earlier numbers: Model 1 takes about 1.61 ms at full clock and about 2.8 to 3.6 ms at the low clock.
- "no timing lines in this recording" means what it says: nothing from another session is used instead. Check the
  launch line and the overlay.

So "pass 1.6 ms, mostly above 2.9 GHz" means full speed. "Pass about 3 ms, mostly at 1.65 to 1.9 GHz" means the
low clock costs the upscaler time in a real game. Tie every conclusion to the pass time, not to the clock shares
alone.

**The pattern to expect** (your session of 2026-11-18, 01:46 to 02:03 UTC, MEASURED, stock profile, Model 1):

| Case | Frames | Clock | Model 1 pass |
|---|---|---|---|
| Uncapped | about 63 per second | above 2.9 GHz 91 % of the time; power at the cap | 1.65 to 1.68 ms |
| OptiScaler limit about 30 | about 30 per second | low clock (1.65 to 1.9 GHz) 54 % of the awake time, above 2.9 GHz 13 %; the card still reads 96 % busy | 2.15 to 2.79 ms (median 2.37) |

A new recording that lands near these numbers repeats that result; one far from them (for example a capped case
at full clock, or a pass near 1.6 ms while mostly at the low clock) is worth a second look at the settings. When the reader has to match lines by the time of day (no byte positions recorded), it says so: the log has
no date, so a line from another day at the same time would match too.

To read a recording's log again later: `scripts/study-home.sh upscaler <folder of that recording>`.

## Part B. What controls the low clock (about 20 minutes)

**What it tells us.** Which setting makes the card start low and wait before it speeds up. Each test runs short
tests with one setting changed, and runs them on your normal settings before and after for comparison:

- the study's own ramp test (HIP): full card work for 0.1 s after 1 s of rest;
- the gaming round's Vulkan tests: a 60 ms load after 1 s of rest, and 2 ms bursts 60 times a second for 1 s.

The Vulkan tests are not in the study folder yet. To include them, start each command in Part B with
`HOME_VKGAME=<probe build folder>/vkgame HOME_VKSPV=<probe build folder>/spv`
(one line, before `scripts/...`). Without that, they are skipped with a note and the HIP test still runs.

Your normal settings first (30 seconds): `scripts/study-home.sh short-tests`

### B1. The driver's power mode (about 7 minutes)

The driver offers these modes: BOOTUP_DEFAULT (in use now), 3D_FULL_SCREEN, POWER_SAVING, VIDEO, VR, COMPUTE,
CUSTOM and WINDOW_3D. Try **3D_FULL_SCREEN** first. It is the gaming mode, and its clock-control numbers differ
most from the default. The default has an extra "booster" entry at 800 MHz that 3D_FULL_SCREEN does not have.
What those numbers do is not documented, so this is a guess to test.

1. Make the profile **Home-Power** (see "Making a LACT profile"). Choose Testing again.
2. Run `scripts/study-home.sh power-mode --owner-ok=power-mode`.

It runs the tests on Testing, switches to Home-Power, checks that the power mode reads 3D_FULL_SCREEN and that only
the power mode and the performance level differ, runs the tests, switches back, checks, and runs them once more.

Likely on the first try: RESTORE FAILED at the end, after the tests ran. Your Testing profile does not name a power
mode, so switching back to it may leave 3D_FULL_SCREEN in place. If that happens, do not edit Testing. Open LACT,
choose the RX 9070 XT, set Power Profile Mode back to BOOTUP_DEFAULT (and Performance Level to Automatic) and press
Apply, then run `scripts/study-home.sh check`. If it still differs, reboot with Testing chosen. The results of the
step are kept either way.

Second mode (optional, another 7 minutes): make a profile **Home-Compute** with COMPUTE, then run
`scripts/study-home.sh power-mode --mode COMPUTE --profile Home-Compute --owner-ok=power-mode`.
Its table in the driver has different clock-control numbers again (what they mean is not documented).

### B2. Performance level pinned high (about 7 minutes)

1. Make the profile **Home-High**. Choose Testing again.
2. Run `scripts/study-home.sh perf-high --owner-ok=perf-high`.

While on Home-High the card sits at its top clock even at rest, so it runs warmer. Each short test starts only
below 60 C. If the hotspot stays above that for 3 minutes, the step prints why it stopped, switches back and runs
nothing more.

### B3. Desktop programs closed (about 3 minutes)

The recorder saw these programs using the card (besides the desktop itself): a chat program, Steam and its web helper,
T3 Code, a chat bot, a dictation program, a planner program, a tray program, a Python program, Xwayland and a web view.
`scripts/study-home.sh check` shows the current list. Close what you can (leave the desktop), then run
`scripts/study-home.sh short-tests apps-closed`. The agent that runs this study needs T3 Code, so close it last or
leave it.

**How to read Part B.** For each test the summary has the probe's own result line (`RAMP ...`) and the recorder's
lines. Compare the first rows with the changed ones:

- the HIP ramp's first clock (normally about 2.7 GHz) and its time to full clock (normally 14 to 15 ms);
- the Vulkan ramp's low plateau (normally about 1.7 GHz for about 27 ms);
- whether the 60-per-second bursts ever get above 1.9 GHz (normally never).

A setting that removes the low start for Vulkan is a strong lead. "No change" is also an answer.

### How each setting can be changed on this machine without sudo (read, not tried)

| Setting | Driver file (owned by root) | Through `lact cli` | Through a LACT profile |
|---|---|---|---|
| Power mode | `pp_power_profile_mode` | no command for it (LACT 0.10.1: list, info, stats, snapshot, power-limit, profile, detach, reattach) | yes: `power_profile_mode_index`; LACT needs the performance level `manual` for it |
| Performance level | `power_dpm_force_performance_level` | no | yes (`performance_level`, now `auto` in every profile) |
| Fan at 100 % | `gpu_od/fan_ctrl/fan_curve` and `fan_zero_rpm_enable` | no | yes (fan mode `static`, `static_speed`; now `curve`). On this card LACT writes a flat curve at that speed and turns zero RPM off (its log texts "setting static curve" and "could not disable zero RPM mode for static fan control"); the card has no `pwm1_enable`, and `pwm1` is read-only |
| Memory clock request | `pp_od_clk_voltage` | no | yes (`max_memory_clock`; the overclock profile asks for 1409) |
| Power cap | `power1_cap` | yes, `lact cli power-limit set` (not used here) | yes (`power_cap`) |
| Desktop programs | - | - | close them by hand |

LACT's daemon accepts commands from the group `wheel`, and this account is in it, so `lact cli profile set` needs no
sudo. Without `-g`, `lact cli` picks the processor's built-in graphics (Radeon 610M), not the RX 9070 XT. Profile
commands are not tied to one card, but any per-card command needs `-g 1002:7550-1043:061A-0000:03:00.0`.

## Part C. Heat, power and long loads (about 30 minutes)

**What it tells us.** How hot the chip's hottest point gets under steady full loads, how much a faster fan helps,
and how far the driver's power reading is from the real draw. The power-limit mod in the card makes the driver read
low.

### C1. Normal fan curve (about 9 minutes)

`scripts/study-home.sh heat normal`. Watch the screen; the hotspot is printed after each load.

Loads, each at least 60 s after the last and starting below 55 C: FP16 matrix 20 s, FP8 matrix 20 s, then a light
load and memory streaming, each as three 20 s runs in one recording. These three runs are not one unbroken minute:

- Each run is a new process. It first starts the card's driver and runs a short calibration load (two launches of
  a tenth of the normal size) before its 20 s.
- The light load follows at once, so the gap is that start-up only (about a second or two; not measured here).
- The streaming load also waits until the hotspot is below 60 C before each run (the trace program's own rule, at
  most 3 minutes). In the study's earlier 20 s streaming traces the hotspot peaked at 60 C and fell to 51 C within
  half a second, so the wait is usually short, but it is not guaranteed.

The summary prints how long each run took ("the parts took 22.1 s, ..."): anything beyond 20 s is start-up,
calibration or waiting, and any wait of the trace program is printed with it. The FMA load is left out: it reached
95 C in 2.7 s before.

### C2. Fan at 100 % (about 10 minutes, loud)

1. Make the profile **Home-Fan100**. Choose Testing again.
2. Run `scripts/study-home.sh heat fan100 --owner-ok=fan100`.

It checks that the fan reads 100 % (a flat 100 % curve, the minimum fan speed at 100, or a fixed PWM of 255) and
that nothing but the fan settings differ, runs the loads, and switches back. The switch back must also bring back
your fan curve and zero RPM, or it says RESTORE FAILED.

**How to read C.** For each load: `HOTSPOT_C` (maximum, and seconds above 80 and 90 C), `MEM_C`, the `THROTTLE`
line, and the probe's own `WINDOW` lines (clock and rate every 2 s). Compare normal and fan100 load by load. If the
hotspot is lower by more than a few degrees and the throttle bits disappear, heat was the limit. If not, heat was
not.

### C3. Wall meter (about 4 minutes, only if you have one)

`scripts/study-home.sh wall-meter`. Plug the PC into the meter. For each of three 20 s stretches (rest, memory
streaming, FP16 matrix) the script says "read the wall meter now". Note the steady number and type it when asked.
It prints, for streaming and matrix, the wall watts per watt the driver reports, above rest. That ratio still
includes the power supply's losses and any change in the rest of the PC, so it is an upper bound on the card's own
scale.

## Part D. The overclock profile (about 20 minutes)

### D1. Memory clock set by hand (about 6 minutes)

Found on 2026-11-18: the memory overclock works, but it takes effect only when the displays blank. With the
monitors on, the memory kept running at 1258 MHz; after switching the monitors off for about 15 s and on again, it
ran at 1408 MHz and reads rose from 633 to about 690 to 700 GB/s. So this step is now a repeat that checks it.

1. On Testing, run `scripts/study-home.sh check` (it records the Testing settings this step compares with).
2. Make the profile **Home-Mem**, press Apply, and **stay on Home-Mem**.
3. Switch your monitors off for about 15 seconds and on again.
4. Run `scripts/study-home.sh memclock --owner-ok=memclock`. If it says the memory still runs at 1258 MHz, repeat
   step 3 and run it again.

It checks that Home-Mem differs from Testing only in the memory request, records the running memory level, runs a
10 s streaming test and the short read test three times, and switches back to Testing. **How to read it:** the
line with "running memory clock" (read from the card's own metrics: 1258 means not applied, 1408 means
applied; the driver's level list then marks no level, which is expected) and "bandwidth read GB/s":
632.6 to 632.9 on stock. If the clock really rose, expect more, up to about 12 % (1409 against 1258 MHz).

### D2. The overclock comparison script (about 4 minutes)

`scripts/study-home.sh overclock --owner-ok=overclock`

It runs the study's comparison (stock, overclock, stock, in short bursts) with its own checks. It switches LACT to
your Default profile and back by itself. This is that script's first real run since it was reworked and tested
with a stand-in, so stay at the machine. Its result: exit 0 is done, 5 is RESTORE FAILED. Its summary is linked in
the line it prints.

### D3. 20-second loads on the overclock profile (about 7 minutes, watch the hotspot)

`scripts/study-home.sh oc-traces --owner-ok=oc-traces`

It switches to Default and runs the light, streaming, FP16 and FP8 loads for 20 s each. Each load is stopped at
**88 C**, not 95, because this profile reached 102 C under a long load before the fan curve was changed. A stop
ends the step: no further load runs, and it switches back. **How to read it:** clock (`WINDOW` lines), hotspot and
throttle bits per load, against the same loads on stock (C1, or the study's earlier traces: FP16 3.32 GHz at 306 W,
FP8 3.18 GHz at the 317 W limit).

## Which stop guards which load

Every load the script starts itself has the recorder beside it as a temperature guard, and most programs have stops
of their own. Any temperature stop ends the step's remaining loads; the switch back to Testing still runs. The
exception is D2: there the study's own comparison script runs the loads, and the recorder only records (see its row).

| Guard | Loads | What it does | How fast |
|---|---|---|---|
| The recorder | every load of the script | reads the hotspot and memory temperature every 2 ms; at 95 C hotspot (88 C in D3), 90 C memory (the driver's own limit is 108 C), or a value it cannot read: TERM to the load, and KILL 1 s later if it still runs | the next sample (2 ms) plus the read (0.13 ms typical, 1.8 ms at worst in 99 of 100) plus however old the driver's value is: the driver's metrics table refreshes about every 8 ms (measured at idle and under load), and on a busy desktop the recorder once missed samples for up to 34 ms; so count on about 10 ms, at worst a few tens of ms; work already sent to the card (one launch) may still finish. The same process records and stops: if it is starved of CPU, the stop is late, and the job's time limit is the last backstop |
| The script | every load | starts a load only below 60 C (55 C for C and D traces), waits at most 3 minutes; no further load after a hotspot of 95 C read after a load | between loads |
| Trace program (`sched_trace`) | C, D1 streaming, D3, C3 | checks every 50 ms; at 95 C ends its load between launches (each about 50 ms); at 100 C ends the process | about 50 to 100 ms |
| HIP ramp (`sched_launch`) | B | one 100 ms kernel; checks every 5 ms, but only 100 C ends it | the kernel runs to its end |
| Read test (`bandwidth`) | D1 | no stop of its own; a fraction of a second per run | the recorder only |
| D2 (`overclock`): the study's comparison script | its short matrix bursts and read tests | its own checks: it starts a load only below 60 C and stops the run after a load that ends at 95 C or more; the matrix program has its own 95 C stop, the read test none. The kit's recorder records but does NOT stop anything in D2 | the matrix program between launches; otherwise between loads |
| Vulkan tests (`vkgame`) | B, if given | checks every 5 ms; at 88 C stops submitting and sets a stop flag that ends a running dispatch within one of its checks (a few microseconds, its source says); 100 C ends the process; each test starts only below 58 C and lasts at most 0.1 s (busy share above 0.4) or 1 s | about 5 ms |

A full-clock burst on this card went from 56 to 84 C in under 0.1 s, and one pixel load reached 95 C in 0.15 s.
At that rate one 2 ms sample is about half a degree, so the recorder stops a load within about a degree of its
limit plus the driver's delay; a burst that ends within 0.1 s is mostly over before any stop can act, and for those
the programs' own limits and the short burst length are what keep it safe.

## If something goes wrong

- **RESTORE FAILED**: the message names the settings that differ. In LACT, choose the profile **Testing first**.
  LACT remembers the chosen profile and applies it again at boot, so rebooting while another profile is chosen
  brings that profile back (for Default: +150 MHz, -50 mV, 374 W). Then check by hand: power limit 317 W,
  performance level Automatic, power profile mode BOOTUP_DEFAULT, no clock or voltage offset, memory clock
  1259 MHz (no maximum set), fan: your usual curve with Zero RPM on. Only then, if the driver still shows a
  difference, reboot. Afterwards run `scripts/study-home.sh check`.
- **Ctrl-C while it switches back**: it is not acted on during the switch back. That usually takes seconds to about
  2 minutes, longer if LACT itself hangs (each LACT call may wait up to 20 s). Wait for the line "restored" (or
  RESTORE FAILED).
- **The terminal died, or the script was killed hard (SIGKILL, "force quit")**: then it cannot switch back, and the
  card stays on the changed profile. The running load and the recorder are ended as the script ends. Run
  `scripts/study-home.sh check`, and choose Testing in LACT if it says NO.
- **A load gets hot**: see the table above. The step stops and switches back. Ctrl-C also stops a test: the load
  ends first, then the profile goes back to Testing.
- **"a game seems to be running"**: a step that loads the card does not start beside a game. Close the game, wait
  until Steam shows it stopped, and start the step again.
- **"another step of this kit is running"**: a step is still running in another terminal. Let it finish, then
  start yours again.
- **"the hotspot stayed at ... for 3 minutes"**: the card did not cool down enough to start the next load. Let it
  cool, check that the fan turns, and start the step again.
- **"not free within 60 s"**: another program is using the card through the shared lock. Wait a minute and run the
  step again.
- **"below 8 GB available memory"**: close something large and run the step again.
- **"LACT has no profile named ..."**: make that profile first.

## Where the results are

`runs/<date and time>-home/`: one folder per step and `summary.md` with every result line. Steps run within 12 hours
share one folder. In each recording folder: `summary.txt`, the raw `samples.csv` (one row every 2 ms), `header.txt`
(settings at the start and end) and the lists of programs that used the card.

## For the study lead: the kit and its checks

- Recorder: `src/study/watch_card.cpp` with `watch_card_sysfs.h` (host only, built with g++ by
  `scripts/study-home.sh build`). It reads `pp_dpm_sclk`/`pp_dpm_mclk` (active line; a garbled line reads as
  missing, never 0 MHz), `gpu_metrics` (format 1, content 3: current graphics clock, both throttle words kept as
  64-bit integers, graphics voltage, activity, socket power), hwmon power, edge, hotspot and memory temperatures,
  fan, busy and VRAM. `firmware_timestamp` reads all ones on this card, so a fresh metrics table is found by a hash
  of its content (without the driver's per-read time). The table refreshes every 8 ms at idle (MEASURED, 1,335
  tables in 10 s). The `gfxclk_avg_MHz` column (offset 40) equalled the current clock in all 6,330 real samples of
  the three control recordings (MEASURED): it is not an independent average on this card, and the summary counts
  the matches. Percentiles and shares are weighted by the time each sample stands for. It handles INT, TERM and HUP
  (files finished), refuses a missing or non-numeric option value, writes `ready` once the first sample is on disk,
  and ends with its parent. As guard it sends TERM, then KILL to the load's process group after 1 s.
- Timing honesty (MEASURED, idle card, desktop in use): at 2 ms, 4,990 samples in 10 s, median period 2.000 ms,
  99th percentile 2.033, longest gap 4.0 ms, 10 slots missed. One read of all files takes 132 us median, 1.8 ms at
  the 99th percentile. The first dry run on a busier desktop had a longest gap of 34 ms; the summary always
  reports it. At 20 ms every read misses the driver's 1 ms cache and costs 0.8 ms median.
- Controls (`runs/20261118T004354Z-home-controls/`): idle 10 s, asleep 96.7 % of the time, wake-ups only at
  1.82 GHz, no throttle bit, only its own output files open (no device). Load: `bandwidth 1024 50` under the
  shared lock. The recorder saw the load start at 2.78 GHz, pass 2.9 GHz 12 ms later and hold 3.22 GHz for 318 ms,
  with GPU busy 100 %, memory busy up to 99 %, 144 W reported, memory clock 1258 MHz and no throttle bit. The probe
  read 632.8 GB/s (study 632.6 to 632.9).
- Fan files on this card (read 2026-11-18): hwmon has `pwm1` (read-only, 0), `pwm1_min` 0, `pwm1_max` 255,
  `fan1_input` 0 and `fan1_target` 0 (fan stopped at rest), `fan1_min` 0, `fan1_max` 6000, `fan1_enable` (cannot be
  read), and no `pwm1_enable`. `gpu_od/fan_ctrl`: the curve 55C 30 %, 58C 58 %, 62C 73 %, 66C 94 %, 73C 100 %;
  zero RPM on (stop temperature 55 C); minimum PWM 30; target temperature 76 C; acoustic limit 6000 and target
  1000 rpm. The kit compares all of these and `pwm1_enable` (plus `pwm1` when it is 1).
- Game guard: load steps refuse while a game runs (exit 7). Under Proton no process is named like the game, so the
  kit looks in /proc only: a card client named GameThread, a card client whose command line runs from
  `steamapps/common`, or any process whose command line holds `steamapps/common` or a Proton
  `compatibilitytools.d` path, with either slash. With Cyberpunk 2077 running (2026-11-18 02:07 UTC) it found
  GameThread (command line `S:\steamapps\common\...`, a Windows path), xalia.exe, wineserver and steam.exe.
- Self-test: `scripts/study-home-selftest.sh` runs the kit in self-test mode (`HOME_SELFTEST=1`; without it the
  test hooks are ignored and named) with a stand-in for `lact`, a fake sensor tree, a fake /proc, fake probes and
  its own lock files. It touches no setting and no load. 46 tests, all PASS in
  `runs/20261118T020951Z-home-selftest` (commit 87c7ef1). 13 deliberate break runs, together 15 failing tests,
  each break FAILs on its tests (runs `20261118T021342Z` to `021656Z`; a 14th file there is the passing control): wrongexp, upscaler (expected values); noflag, norestore, gamelock, nokitlock,
  nofanfields, thermalstop, hooks, extrafield, nokill, nopdeath, nogameguard (properties). The same tests run
  against the previous kit (49f7a25, before the game guard) failed 30 of 45, every test of a fix among them; the
  older behaviour tests passed there too.
- Not tested: the overclock step with its flag (it calls the unchanged `scripts/study-overclock.sh`, whose own
  self-test covers it); a real LACT profile switch to any Home-* profile, and what LACT's switch back to Testing
  does to a power mode, performance level or static fan that Testing does not name (the restore check catches it
  and says RESTORE FAILED); the LACT labels' places on screen; how old the driver's hotspot value is when read;
  the game guard on the real machine with the game closed (it ran only while the game was running, so a leftover
  Steam or Proton process that would block load steps was not looked for).
