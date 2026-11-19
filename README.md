# RDNA 4 measurement study: the AMD RX 9070 XT (gfx1201)

This repository measures how the AMD Radeon RX 9070 XT behaves. Small probes time or inspect one part of the card at a
time, from matrix instructions to memory caches and clock changes. A separate workload study uses large language models
to see how those hardware rules appear in real programs.

The goal is to understand the hardware, not to speed up a program. Each measurement file starts with a plain-language
summary, then gives the tables, method, and rerun commands.

## Key findings

Each bullet states the result. The linked file contains the evidence, spread, and limits of the measurement.

- **Matrix instruction time depends on the input type, not the data.** FP16 and BF16 inputs take 16 cycles. FP8, BF8,
  and 8-bit integer inputs take 8 cycles. Chaining through the accumulator adds no time, while feeding a result back as
  a multiplicand adds 3 cycles. Whole-card rates are about 211 TFLOPS for FP16 and BF16, and 420 TFLOPS for FP8 and
  8-bit integer work. Sparse forms do twice the work in the same time.
  ([matrix-timing.md](measurements/matrix-timing.md))
- **CPU models reproduce the matrix instructions bit for bit, and show that their additions are not exact.** All 11
  dense and all 11 sparse forms match the models. The card adds 16-bit-input products in 4 steps of 4, and 8-bit-input
  products in 2 steps of 8. Each fixed-point step cuts low bits toward minus infinity, then rounds the sum to FP32.
  A small term next to a large one can vanish.
  ([matrix-arithmetic.md](measurements/matrix-arithmetic.md))
- **The matrix adder's error rarely survives rounding to FP8.** Only 2 of 1,446,000 model-shaped outputs differed from
  the exact sum after E4M3 rounding. With FP8 weights, FP16 output differed in 0.004 % of cases at K = 16 and 0.18 % at
  K = 4096.
  ([errors-after-quantizing.md](measurements/errors-after-quantizing.md))
- **FP8 conversions are exact, but their edge cases need care.** Widening to FP32 is exact for every code. Narrowing
  rounds to nearest even over all 2^32 FP32 inputs. Narrowing a NaN always gives 0xFF, overflow gives NaN, and infinity
  still gives NaN with saturation enabled. Widening costs 1.0 cycle per weight to FP32 and 1.5 cycles to FP16 pairs.
  ([conversions-and-dots.md](measurements/conversions-and-dots.md))
- **Most vector instructions issue once per cycle, but a dependent instruction waits 5 cycles.** Every dual-issue pair
  allowed by the guide runs at two operations per cycle. FP64 operations take 32 cycles. Branches cost 9 cycles when
  not taken and 15 when taken.
  ([execution-core.md](measurements/execution-core.md))
- **Streaming reads nearly fill the memory bus, and the cache levels are distinct.** Plain reads reach 633 GB/s against
  a 640 GB/s bus limit regardless of data content. The card has a 32 KB first-level cache per compute unit, 8 MB of L2
  emptied at each kernel launch, a 64 MB memory-side cache, and 16 GB of GDDR6. A first-level hit takes 81 cycles;
  a GDDR6 read takes 485 to 545.
  ([memory-system.md](measurements/memory-system.md))
- **Shared memory answers 2.5 times faster than the first-level cache.** A 32-bit LDS load issues in one cycle per wave
  and returns data after 32 cycles. HIP's default work-group-processor mode halves bandwidth per compute unit: the whole
  card reads 11.5 TB/s in that mode and 21.9 TB/s in compute-unit mode.
  ([lds-and-special-loads.md](measurements/lds-and-special-loads.md))
- **The whole card pauses about every 2.8 ms.** A spinning wave loses about 1.1 us, and 96 % of stall clusters coincide
  across waves on all 4 shader engines. The cycle counter keeps running during the pause, so timing code needs short
  regions or medians.
  ([scheduling-and-clocks.md](measurements/scheduling-and-clocks.md))
- **Vulkan reaches the same hardware limits as HIP.** Register-fed matrix work measured 210.6 TFLOPS for FP16 and 421.0
  for FP8 through Vulkan, against 211 and 420 through HIP. Vulkan read 636 GB/s against HIP's 633. RADV uses wave64 for
  compute shaders with 64 or more invocations, which costs nothing for register-fed matrix code.
  ([vulkan-view.md](measurements/vulkan-view.md))
- **Matrix instructions ignore the lane mask, and the compiler can miss the consequences.** All lanes are read and
  written in wave32 and wave64 across 20 tested forms. A matrix builtin inside a per-lane `if` can therefore give wrong
  results. The compiler also omitted a required hazard fix across a non-inlined call, producing 64 wrong results among
  256 in the probe.
  ([compiler-and-hazards.md](measurements/compiler-and-hazards.md),
  [leftovers-c-b.md](measurements/leftovers-c-b.md))
- **Game-style work starts at a lower clock through Vulkan than through HIP and steps up only when idle gaps are short.**
  The same loop starts at 1.7 to 1.8 GHz through Vulkan and 2.7 to 2.8 through HIP. An upscaler's neural network took
  about 2.8 ms per frame at the low clock and 1.6 ms at full clock. On the game route's graphics queue, gaps of 2.1 ms
  or less always allowed a step up, while gaps of 3 ms or more always stayed low. These were headless tests, not a
  measurement of a running game.
  ([gaming-route.md](measurements/gaming-route.md), [gaming-probes.md](measurements/gaming-probes.md))
- **llama.cpp's FP8 prompts are slower because of the code path, not the FP8 format.** FP8 cannot use the integer matrix
  kernel available to Q8_0, so llama.cpp first widens each weight matrix to FP16. In a 3B prompt of 512 tokens, the
  matrix operations took 26.9 ms for FP8 and 25.6 ms for Q8_0. The 17.8 ms gap came from the extra memory passes for
  widening, activation conversion, and scaling. Q8_0 forced through the same path was as slow as FP8.
  ([llm-workload.md](measurements/llm-workload.md))
- **Generation leaves part of the memory bus unused, and FP8 decode has a real power cost.** Generation streamed 71 to
  83 % of the read rate at a 1408 MHz memory clock. Prompt loops sat at the 317 W reported cap. Patched FP8 generation
  used 56 to 57 W more than Q8_0 at nearly the same bandwidth. Matched test kernels explain 30 to 33 W through the
  instruction mix and block shape; the rest is open.
  ([llm-workload.md](measurements/llm-workload.md), [llm-workload-3.md](measurements/llm-workload-3.md))
- **PyTorch and vLLM use the FP8 matrix instruction for prompts, but PyTorch's path hung this card.** Their large matrix
  products run hipBLASLt kernels containing `v_wmma_f32_16x16x16_fp8_fp8`, reaching 236 to 259 TFLOPS on the overclocked
  profile. vLLM's one-token decode instead uses the FP8 dot product on the vector unit. PyTorch's FP8 path hung the
  overclocked GPU four times and is parked.
  ([llm-runtimes.md](measurements/llm-runtimes.md))
- **Against the true 3B weights, llama.cpp's FP8 and Q8_0 files have similar error.** Mean KL divergence from the 16-bit
  reference was 3.7e-3 for FP8 and 4.3e-3 for Q8_0, against 2e-5 for llama.cpp's own 16-bit file. The earlier, much
  larger ratio compared both files with the exact FP8 checkpoint weights, not with the true model.
  ([llm-runtimes.md](measurements/llm-runtimes.md), [llm-workload-2.md](measurements/llm-workload-2.md))
- **Most profiler counters are dead on this stack, and its sub-microsecond overlaps are timestamp artifacts in the probe.**
  Only 4 of 13 counter sets returned numbers under ROCm 10.1; memory-side and instruction counters read zero. An
  in-kernel 100 MHz clock showed no overlap in 323,982 dependent kernel pairs, while two kernels that really ran at
  the same time overlapped on both clocks. The profiler reported
  3 overlaps, but the in-kernel clock placed the kernels 3.4 us apart. llama.cpp's overlaps were not retimed, so applying
  the artifact finding to them remains INFERRED.
  ([llm-workload-2.md](measurements/llm-workload-2.md), [llm-workload-3.md](measurements/llm-workload-3.md))

## Card and setup

- Card: ASUS Prime RX 9070 XT, 16 GB Samsung GDDR6, 64 compute units, gfx1201, wave32 under HIP.
- Compiler and runtime: ROCm 10.1, HIP 7.16, and AMD clang 24. The Vulkan probes use Mesa RADV 27.0.0-devel.
- Operating system: Linux kernel 7.3.4.

The configuration matters for every wall-clock result:

- Most hardware probes used the LACT profile "Testing" with stock core clock and voltage, a 317 W cap, and memory at
  1258 MHz. The first-hour run and lane 1's cycle counts used an earlier overclocked profile. Cycle counts come from
  the wave's own counter and do not depend on the clock frequency.
- LLM rounds 1 to 3 used "Testing" with the memory clock at 1408 MHz.
- LLM round 4 and the exact 8B step used the overclocked "Default" profile: a 374 W cap, +150 MHz core offset, -50 mV,
  and a requested memory maximum of 1409 MHz. Round 4 read 1408 MHz until a GPU reset and 1258 MHz afterward. The exact
  8B step read 1408 MHz throughout. Each wall-clock table in those files states its memory clock.
- A 1.2x regulator-level power-limit mod is installed. Reported watts are scaled down on the main rail and are not true
  board power. The study notes document the mod; these probes did not measure it.

Wall-clock numbers in GB/s, TFLOPS, GHz, and watts describe this card in these configurations, not a stock card. Later
rounds correct earlier interpretations where the files say "Updated in round N"; the findings above use the corrected
results.

A new memory overclock took effect only after the displays blanked once. Before that, the active memory level stayed at
1258 MHz. After switching the displays off and on, `gpu_metrics` reported 1408 MHz. Read `current_uclk` from
`gpu_metrics` after a change rather than relying on the listed levels in `pp_dpm_mclk`.
([scheduling-and-clocks.md](measurements/scheduling-and-clocks.md))

## How results are labelled

- **MEASURED** means observed in a run on this card. The write-up names the run folder.
- **READ** means taken from a document, source file, machine-code listing, or driver dump.
- **INFERRED** means reasoned from measurements rather than observed directly.

A probe is trusted only after it reproduces a known answer from AMD's documentation, an earlier run, or a CPU model.
Expectations are written before the runs and kept in the `*-expectations.md` files. Probes print `VERDICT` lines and
include deliberate breaks that must report `FAIL`. A check that cannot fail proves nothing. The write-ups report the
spread as well as the median.

AMD's official [RDNA 4 Instruction Set Architecture: Reference Guide](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
is the main instruction reference.

## Repository layout

| Folder | What it holds |
|---|---|
| `measurements/` | Measurement write-ups, expectations written before runs, and `gaming-tables/` with three derived CSV tables. |
| `notes/` | The study plan, first-hour record, session guide, and reading notes on the hardware and software stack. |
| `notes/plans/` | The matrix-arithmetic test plan. |
| `scripts/` | One `study-*.sh` entry point per study lane, Python checkers and table builders, and expected machine-code patterns. |
| `src/` | HIP probes, Vulkan host programs, and Vulkan shaders. |

`runs/`, `build/`, `toolchains/`, and `sources/` are git-ignored. Raw run data is not in the repository. Each write-up
names its run folder and the script and mode that produced it. The committed tables preserve the results; rerun the
script to check them against new raw data.

## Measurement files

| File | Subject |
|---|---|
| `execution-core.md` | Instruction issue and latency, dual issue, scalar operations, branches, and waves sharing a SIMD. |
| `memory-system.md` | Global-memory bandwidth and latency, cache levels, loads in flight, stores, and cache hints. |
| `lds-and-special-loads.md` | Shared memory, bank conflicts, scalar loads, block loads, and transposing loads. |
| `matrix-timing.md` | Cost of each matrix form, operand hazards, sharing, and whole-card rates. |
| `matrix-arithmetic.md` | Bit-exact models of the matrix instructions and their addition rules. |
| `errors-after-quantizing.md` | How often matrix-addition error survives rounding to FP16 or FP8. |
| `conversions-and-dots.md` | Small-format conversions, dot products, and the cost of widening FP8. |
| `scheduling-and-clocks.md` | Launch cost, work placement, barriers, clocks, power, the regular pause, and process sharing. |
| `compiler-and-hazards.md` | Hazard handling, lane masks, and wrong or slow compiler output. |
| `vulkan-view.md` | The same hardware through Vulkan compute and RADV. |
| `leftovers-c-a.md` | Follow-up probes of the execution core, memory, LDS, and compiler hazards. |
| `leftovers-c-b.md` | Follow-up probes of matrix rules, scheduling, special loads, Vulkan, and arithmetic error. |
| `gaming-route.md` | An upscaler's neural network on the headless vkd3d-proton, dxil-spirv, and RADV route. |
| `gaming-probes.md` | Vulkan and HIP probes of dispatch gaps, queues, bursts, clock steps, and preemption. |
| `gaming-probes-f-expectations.md` | Expectations for the final gaming probe round, written before its runs. |
| `llm-workload.md` | LLM round 1: llama.cpp kernel traces, speed, quality, clocks, power, and bandwidth. |
| `llm-workload-expectations.md` | Expectations for LLM round 1. |
| `llm-workload-2.md` | LLM round 2: exact-weight references, matrix shapes, profiler counters, cache behavior, and streams. |
| `llm-workload-2-expectations.md` | Expectations for the first part of LLM round 2. |
| `llm-workload-2-expectations-7.md` | Expectations for the stream and matrix-shape items in LLM round 2. |
| `llm-workload-3.md` | LLM round 3: FP8 decode power, the exact 8B reference attempt, cache state, counters, and overlap timestamps. |
| `llm-workload-3-expectations.md` | Expectations for LLM round 3. |
| `llm-workload-3-expectations-q1d.md` | Expectations for the random-data power follow-up. |
| `llm-runtimes.md` | LLM round 4: PyTorch, vLLM, and llama.cpp on the overclocked profile, including the PyTorch hangs. |
| `llm-runtimes-expectations.md` | Expectations for LLM round 4. |
| `llm-8b-exact.md` | The 8B files against the FP8 checkpoint's own weights converted exactly. |
| `llm-8b-exact-expectations.md` | Expectations for the exact 8B step. |

Two earlier follow-up passes have no separate measurement file. Their results appear in new sections of the lane files.

## Running the probes

Read the usage text at the top of the relevant `scripts/study-*.sh` file first. Most hardware lanes accept `build`, `run`,
`breaks`, or `all`. The `all` mode builds, runs the probes, then runs the deliberate breaks.

```sh
scripts/study-core.sh all
scripts/study-memory.sh all
scripts/study-vulkan.sh all
```

The scripts read `local.env` in the repository root when that file exists. Copy the committed `local.env.example` and
set the paths for your machine:

| Setting | Purpose |
|---|---|
| `STUDY_GPU_LOCK` | Lock file that serializes GPU jobs. The default is `runs/gpu.lock`. |
| `LLM_BUILD_ROOT` | Folder containing the llama.cpp builds and the perplexity text. |
| `LLM_MODELS` | Folder containing the GGUF models. |
| `RDNA3_DIR` | Optional: a checkout of the RDNA 3 companion repository, used by `scripts/first-hour.sh`. |
| `STUDY_UPSCALER_LOG` | Optional: the path of an upscaler's log inside a game folder, read by `scripts/study-home.sh`. |

ROCm 10.1 is found through `ROCM_PATH` in the environment, not through `local.env`.

The Vulkan probes also need Mesa RADV and `glslc`. The LLM study needs the external builds, models, and runtime
environments described in its measurement files.

Run `scripts/first-hour.sh info` before the other probes. This read-only step writes `runs/<time>-gfx1201/card.env`;
later scripts read the card record through `runs/latest/card.env`.

Every GPU job takes the shared lock, runs under `nice`, refuses to start below 8 GB of free memory, and logs the clocks
before and after; whole-card jobs keep to a hotspot limit. The LLM study's script (`scripts/study-llm.sh`) adds more:
a game check, a `systemd-run` scope with a 10 GB host-memory cap and no swap, and a 20 ms thermal guard that stops the
job at a 95 C hotspot or 100 C memory temperature, if a sensor becomes unreadable, or if the guard itself is lost.

Whole-card loads can heat this card quickly, especially on an overclock. Keep added loads short and do not bypass the
thermal guard. `scripts/study-overclock.sh` and some `scripts/study-home.sh` steps change card settings; they are not
part of a normal probe run.

## Status and open work

The nine hardware lanes and their follow-up passes are measured, reviewed, and rerun. The gaming studies and four LLM
rounds are complete and reviewed. The exact 8B comparison is also complete.

The open lists in the measurement files still include sustained loads and some profile changes, measurements during a
running game, questions not reachable through HIP or Vulkan, and the causes of the dead profiler counters. The remaining
part of FP8 decode's extra power is unexplained. PyTorch's FP8 hangs, NaNs, and non-reproducibility remain parked rather
than diagnosed. Other GPUs, multi-GPU runs, and real chat workloads are outside the measured scope.
