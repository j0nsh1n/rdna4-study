# Round 11: PCIe, ray tracing, video, and graphics on the RX 9070 XT (2026-11-26)

Card RX 9070 XT (`1002:7550`, gfx1201), RADV, ROCm. Slot 1 raw logs are in the companion checkout's git-ignored build tree, folder `afk-r13/slot1-results/`. Slot 2 raw logs are in `afk-r13/slot2-results/`. Both were recorded 2026-11-26. Labels: MEASURED (a number in one of those logs), READ (a file or an earlier note of this study), INFERRED (worked out from those, not printed by the probe).

**Profile (READ, Slot 1 `env-manifest.json`, generated 2026-11-26T12:51:01Z).** LACT track `overclock`: core offset +150 MHz, voltage offset -50 mV, memory-clock request 1409 MHz, performance level `auto`, power cap 374 W, power profile `BOOTUP_DEFAULT`. VBIOS `115-G295BP00-100` build `00150471`. Host: Ryzen 7 9700X, Linux `7.3.4-202`. Manifest toolchain: ROCm 10.1.0, AMD clang `d6f6cb69`, HIP 7.16.26385. The Vulkan probes print driver `radv Mesa 27.0.0-devel`. The installed package recorded in the manifest is `mesa-vulkan-drivers-git-freeworld-26.4.0-1.git.20261029.1c3a936.fc44.x86_64`. Git commit in the manifest is null. A 1.2x regulator-level power-limit mod is installed on this card (READ, earlier notes of this study; the manifest has no field for it), so reported watts are low on the main rail and are not board power. Slot 2's manifest (2026-11-26T14:39:42Z) records the same overclock fields, the same VBIOS, the same kernel string, the same package string, and the same toolchain commit prefix.

Every wall-clock GB/s, Mrays/s, watt, and GHz below is this overclock profile on this card. Cycle counts, hit counts, and the instruction names do not depend on the clock. Achieved clocks are in each section: the PCIe copies did reach 1408 MHz memory, and the ray-tracing job's memory clock stayed at 96 MHz.

## Plain summary

1. **PCIe copies on a Gen5 x16 link, ReBAR on, finished PASS (MEASURED).** A 1 GiB pinned copy moved at 55.279 GB/s host-to-device, 56.518 GB/s device-to-host, and 59.769 GB/s both ways. With `HSA_ENABLE_SDMA=0`, a 1 GiB pageable buffer offset by one byte dropped to 44.894 and 34.987 GB/s. A 4 to 64 byte pinned copy waits 19.18 to 19.79 us. Card-timed ping-pong is 1.560 us round trip, 0.780 us one way.
2. **The link-speed fields disagree (MEASURED).** Sysfs reads 32.0 GT/s x16 at the start and the end of both copy jobs, including at about 11 W. The 100 ms `gpu_metrics` field stayed at 2.5 GT/s x16 for every sample while the copies ran. The achieved rates sit above the 4.0 GB/s Gen1 ceiling written in `expect.md` and under the 63.015 GB/s Gen5 per-direction ceiling.
3. **Ray queries through RADV finished with a legacy, unqualified PASS (MEASURED).** The ray-hit correctness controls and the deliberate breaks all passed, but 60 of the 68 rate cells had `clock_flat=NO` under the unpinned auto-boost, so the rates are not flat-clock results. The printed `VERDICT r11_rt: PASS` is legacy pre-fix output: Slot 1 ran the binary from an earlier commit, before a later fix added the `clock_flat_failed=N/M` qualification and `--strict-clock`. The current `r11_rt` qualifies an unpinned ramp verdict and fails a drifting cell under `--strict-clock`, and the current `run_r11_rt.sh` `analyse()` would reject this old log for lacking the `clock_flat_failed=N/M` field. An empty top level and a root miss sit near 106 Grays/s (the 64-triangle floor is 106420.7 Mrays/s, 9.40 ps/ray, INFERRED as `1e6 / median_Mrays_s`). Eight coherent triangles, first hit, short tMax, are 47616.4 Mrays/s (21.00 ps/ray). At 262,144 triangles the closest-hit long-tMax cells are 19601.2 Mrays/s coherent and 13024.4 Mrays/s shuffled. A serial chain of dependent rays runs from 677.4 ns/ray at 1 triangle to 1730.8 ns/ray at 32,768, and 1612.2 ns/ray at 262,144.
4. **RADV emitted RDNA 4 BVH8 instructions (READ, the three listings).** Each of `rt_check.s`, `rt_rate.s`, and `rt_chain.s` has one `image_bvh8_intersect_ray` and one `ds_bvh_stack_push8_pop1_rtn_b32`. The plain, 64-bit, and dual BVH image opcodes are absent. Extra `bvh` mentions in the listing scan are `s_wait_bvhcnt`.
5. **The ray-tracing shader clock ramped inside the repeats, and the memory clock stayed at 96 MHz (MEASURED).** `clock_flat=yes` on 8 of 68 rate cells and `NO` on 60. Among the 873 sampler samples at or above 2000 MHz, the shader clock averages 3144.8 MHz (2004 to 3349) and reported power averages 348.6 W, peak 464 W. Peak hotspot over all 5548 samples is 93 C. Memory clock is 96 MHz on 5545 samples and 456 MHz on 3.
6. **Slot 2 video decode and encode passed (MEASURED).** 72 of 72 cells printed `activity=ok failed=0`. Six decode frame-md5 checks and six encode PSNR checks passed, each encode against a 26 dB floor. The 100 ms sampler's hotspot is 35 to 48 C. `--break-device` was not passed, so the iGPU decode break has no hardware result.
7. **Slot 2 graphics finished, and the verdict is FAIL (MEASURED).** The 95 C guard did not fire. Sampler hotspot is 35 to 86 C; the probe's `CLOCKS after` line says highest seen 87 C. `fill_RGB10A2` passed at 1, 4, 32, and 512 MiB. The summary is `controls_or_breaks_failed=8 burst_over=3 spread_flagged=29`: four 4x MSAA occlusion counts are wrong with a matching read-back, three bursts ran past 250 ms, and the 512 MiB random write printed 855.3 GB/s above the probe's bus ceiling. Fill and compression rates from this run are not hardware limits. Slot 1's full graphics run had stopped at the 95 C guard; that run's rates are not results either.
8. **Still open.** The 2-million-triangle scene (`--big`) and the ray-tracing pipeline were not run. The video iGPU break was not run. The long thermal load was not run.

## Item 4b: PCIe Gen5 x16, ReBAR, SDMA, and host-GPU latency (MEASURED, PASS)

Script `scripts/round11/run_r11_pcie.sh --run`: two locked jobs, the second with `HSA_ENABLE_SDMA=0`. Logs `afk-r13/slot1-results/r11-pcie/pcie-default.log` and `pcie-sdma0.log`. `analysis.txt` reprints 4 KiB, 1 MiB, 64 MiB, and 1 GiB only. The tables here are every `CELL` line in the two logs (80 cells each, all `verify=PASS`). Repeats are 30 at 64 MiB and below, and 15 at 256 MiB and 1 GiB.

`card-state.log` stamps 13:04:10 through 13:04:42. The slot run sheet places this job at about 05:04-05:05 PST, so those stamps line up with UTC. Both jobs exited 0. There is no `kernel-faults.txt` in this folder. The run sheet records 0 amdgpu faults for the job (READ).

### Link, BAR, and the copy ceiling

| Fact | Where | Value |
|---|---|---|
| Device line | both logs | AMD Radeon RX 9070 XT (gfx1201), PCI `0000:03:00.0` |
| Per-direction ceiling | both logs | 63.015 GB/s at 32.0 GT/s x16 (`32 x 16 x 128/130 / 8`) |
| Both directions together | `expect.md` (READ) | 126.0 GB/s |
| Cells rejected for exceeding the ceiling | both `SUMMARY` lines | 0 |
| ReBAR | `bar.txt` | ON. BAR0 16384 MiB. CPU-visible card memory 17095983104 bytes of 17095983104 |
| Sysfs link, start and end, both jobs | `card-state.log`, `bar.txt` | `32.0 GT/s PCIe x16 (max 32.0 GT/s PCIe x16)`, including the start at 11 W, shader clock 0, memory 96 MHz |
| `pp_dpm_pcie` levels | `expect.md` (READ, before this job) | 2.5, 16.0, and 32.0 GT/s |
| Live `gpu_metrics` during the copies | both sampler CSVs | `pcie_gts_x10` 25 and width 16 on every sample (110 samples, span 10.900 s; then 112 samples, span 11.100 s) |

The highest bidirectional maximum in the default job is 63.883 GB/s. That is above the one-direction ceiling and below the 126 GB/s two-direction figure in the plan. A bidirectional GB/s number is the sum of the two directions. The probe rejected 0 cells.

`HSA_ENABLE_SDMA` is `(unset)` in the first job and `0` in the second.

Sampler, default job: memory clock mean 1048 MHz, median 1124 MHz, max 1408 MHz, shader clock max 3255 MHz, reported power max 81 W, hotspot max 34 C, fan 0 rpm on all 110 samples, throttle words 0 on all samples. SDMA=0 job: memory clock mean 1063 MHz, median 1124 MHz, max 1408 MHz, shader clock max 3255 MHz, reported power max 90 W, hotspot max 36 C, fan 0 rpm, throttle words 0.

### Controls and deliberate breaks

Both jobs: `controls_failed=0`, `VERDICT r11_pcie: PASS`. A zero-size `hipMemcpy` returned 0 in both.

| Check | Default job | `HSA_ENABLE_SDMA=0` |
|---|---|---|
| Device-to-device 256 MiB | PASS, median 839.5 us, read+write 639.5 GB/s, anchor 633 | PASS, median 843.8 us, read+write 636.3 GB/s, anchor 633 |
| Break: flip one byte after a verified copy | FAIL as required | FAIL as required |
| Break: skip the copy | FAIL as required. 256 MiB reported as copied in 19.00 us = 14196 GB/s | FAIL as required. 19.93 us = 13537 GB/s |
| Break: ping-pong with no responder | FAIL as required. Card completed 0 of 3 | FAIL as required. Card completed 0 of 3 |

### Default copy engine (`HSA_ENABLE_SDMA` unset)

**Pinned host memory.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H | BIDIR |
|---|---:|---:|---:|---:|
| 4 KiB | 30 | 0.098 (0.088-0.111) | 0.114 (0.085-0.124) | 0.221 (0.197-0.227) |
| 16 KiB | 30 | 0.463 (0.431-0.492) | 0.463 (0.370-0.505) | 0.846 (0.760-0.893) |
| 64 KiB | 30 | 2.206 (1.524-8.657) | 2.861 (1.708-8.868) | 3.549 (3.122-5.795) |
| 256 KiB | 30 | 8.006 (5.003-18.192) | 9.531 (0.745-15.312) | 23.112 (1.981-25.182) |
| 1 MiB | 30 | 19.714 (15.546-22.795) | 30.941 (27.755-32.334) | 38.729 (34.669-45.082) |
| 4 MiB | 30 | 46.389 (13.614-46.599) | 47.279 (44.516-47.685) | 54.382 (49.587-56.969) |
| 16 MiB | 30 | 52.884 (31.956-53.521) | 54.009 (51.515-54.227) | 59.929 (57.777-63.037) |
| 64 MiB | 30 | 54.632 (54.181-55.126) | 55.580 (55.169-55.988) | 62.123 (59.840-63.530) |
| 256 MiB | 15 | 55.431 (54.675-55.540) | 56.203 (56.011-56.350) | 62.440 (60.561-63.883) |
| 1 GiB | 15 | 55.279 (54.812-55.379) | 56.518 (55.947-56.601) | 59.769 (42.906-60.194) |

**Pageable host memory.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H | BIDIR |
|---|---:|---:|---:|---:|
| 4 KiB | 30 | 0.095 (0.086-0.117) | 0.098 (0.092-0.119) | 0.158 (0.132-0.169) |
| 16 KiB | 30 | 0.370 (0.071-0.463) | 0.371 (0.316-0.450) | 0.562 (0.469-0.597) |
| 64 KiB | 30 | 3.870 (3.225-4.068) | 4.553 (3.566-4.917) | 7.609 (6.215-8.484) |
| 256 KiB | 30 | 13.523 (9.389-14.903) | 14.852 (14.056-15.669) | 10.420 (9.024-14.117) |
| 1 MiB | 30 | 19.228 (18.039-22.830) | 25.050 (22.211-33.109) | 24.401 (24.094-27.289) |
| 4 MiB | 30 | 41.726 (31.486-45.890) | 43.209 (40.801-47.223) | 42.488 (41.131-45.850) |
| 16 MiB | 30 | 51.528 (49.892-53.256) | 52.583 (47.283-54.217) | 52.053 (50.157-53.070) |
| 64 MiB | 30 | 54.214 (51.842-54.720) | 55.278 (53.742-55.727) | 54.618 (51.064-55.045) |
| 256 MiB | 15 | 54.880 (52.084-55.107) | 55.753 (55.483-55.941) | 55.423 (54.430-55.567) |
| 1 GiB | 15 | 55.367 (54.960-55.467) | 56.081 (55.597-56.215) | 55.609 (55.389-55.699) |

**Pageable host memory, offset by 1 byte.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H |
|---|---:|---:|---:|
| 4 KiB | 30 | 0.101 (0.075-0.128) | 0.104 (0.098-0.127) |
| 16 KiB | 30 | 0.381 (0.364-0.490) | 0.394 (0.361-0.502) |
| 64 KiB | 30 | 3.417 (2.890-4.473) | 4.708 (3.480-4.991) |
| 256 KiB | 30 | 10.300 (8.898-11.900) | 13.576 (4.101-15.189) |
| 1 MiB | 30 | 21.406 (16.237-23.432) | 31.989 (27.285-33.608) |
| 4 MiB | 30 | 45.034 (33.833-47.085) | 47.530 (45.149-48.305) |
| 16 MiB | 30 | 52.916 (49.772-53.494) | 54.118 (51.418-54.391) |
| 64 MiB | 30 | 54.147 (52.521-54.636) | 55.323 (52.397-55.735) |
| 256 MiB | 15 | 54.127 (26.895-54.986) | 55.750 (55.293-55.814) |
| 1 GiB | 15 | 54.780 (53.086-54.913) | 55.964 (55.436-56.120) |

### `HSA_ENABLE_SDMA=0`

**Pinned host memory.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H | BIDIR |
|---|---:|---:|---:|---:|
| 4 KiB | 30 | 0.100 (0.080-0.121) | 0.111 (0.089-0.121) | 0.197 (0.177-0.221) |
| 16 KiB | 30 | 0.451 (0.418-0.483) | 0.458 (0.312-0.486) | 0.863 (0.700-0.890) |
| 64 KiB | 30 | 2.325 (2.189-2.726) | 2.695 (2.501-2.873) | 4.320 (3.828-4.656) |
| 256 KiB | 30 | 9.002 (8.139-9.309) | 9.650 (9.027-10.102) | 14.796 (13.133-15.683) |
| 1 MiB | 30 | 15.793 (12.346-21.056) | 21.026 (17.427-25.513) | 30.460 (5.940-34.150) |
| 4 MiB | 30 | 35.487 (20.962-39.347) | 37.495 (35.410-39.927) | 46.100 (42.601-48.745) |
| 16 MiB | 30 | 47.781 (28.927-49.925) | 50.650 (42.424-51.899) | 50.361 (46.668-52.268) |
| 64 MiB | 30 | 52.432 (48.904-53.940) | 55.045 (51.475-55.796) | 53.384 (50.700-55.560) |
| 256 MiB | 15 | 51.367 (49.378-52.711) | 55.323 (50.640-56.547) | 53.080 (51.337-54.659) |
| 1 GiB | 15 | 52.894 (50.953-53.546) | 56.061 (54.657-56.651) | 53.332 (50.107-54.731) |

**Pageable host memory.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H | BIDIR |
|---|---:|---:|---:|---:|
| 4 KiB | 30 | 0.094 (0.084-0.122) | 0.096 (0.090-0.125) | 0.154 (0.034-0.167) |
| 16 KiB | 30 | 0.376 (0.351-0.471) | 0.395 (0.202-0.476) | 0.603 (0.493-0.646) |
| 64 KiB | 30 | 2.400 (2.056-2.477) | 2.480 (2.094-2.609) | 3.552 (2.715-3.967) |
| 256 KiB | 30 | 7.018 (6.441-8.926) | 7.643 (7.076-9.312) | 7.823 (7.488-8.810) |
| 1 MiB | 30 | 17.503 (14.790-20.871) | 17.622 (15.279-21.261) | 19.368 (16.339-21.259) |
| 4 MiB | 30 | 35.731 (28.463-39.805) | 36.784 (33.733-41.668) | 36.558 (32.572-39.808) |
| 16 MiB | 30 | 48.663 (42.506-50.935) | 48.845 (41.629-51.420) | 48.849 (38.590-50.329) |
| 64 MiB | 30 | 51.989 (48.035-53.544) | 52.692 (49.188-54.296) | 51.901 (46.612-53.228) |
| 256 MiB | 15 | 52.009 (49.429-52.794) | 51.635 (44.498-54.069) | 52.367 (50.662-52.981) |
| 1 GiB | 15 | 52.559 (51.780-52.902) | 53.333 (52.208-54.170) | 52.506 (51.241-52.994) |

**Pageable host memory, offset by 1 byte.** Median GB/s, min-max in parentheses. Every cell prints `verify=PASS`.

| Size | reps | H2D | D2H |
|---|---:|---:|---:|
| 4 KiB | 30 | 0.091 (0.082-0.111) | 0.094 (0.079-0.118) |
| 16 KiB | 30 | 0.361 (0.295-0.464) | 0.375 (0.318-0.455) |
| 64 KiB | 30 | 2.403 (2.084-2.646) | 2.404 (2.047-2.582) |
| 256 KiB | 30 | 7.020 (6.292-8.314) | 7.471 (6.877-7.592) |
| 1 MiB | 30 | 17.933 (15.475-21.175) | 17.248 (16.110-18.615) |
| 4 MiB | 30 | 30.651 (27.071-32.517) | 35.181 (32.148-37.286) |
| 16 MiB | 30 | 39.530 (32.948-41.933) | 26.467 (24.245-28.593) |
| 64 MiB | 30 | 42.581 (30.433-44.246) | 33.605 (31.116-35.177) |
| 256 MiB | 15 | 43.608 (42.577-44.597) | 34.626 (32.996-35.315) |
| 1 GiB | 15 | 44.894 (44.157-45.496) | 34.987 (33.747-35.384) |

The odd-offset pageable path is the one that falls away with SDMA off. At 1 GiB it is 44.894 GB/s host-to-device (44.157-45.496) and 34.987 GB/s device-to-host (33.747-35.384), against 54.780 and 55.964 in the default job. At 64 MiB with SDMA off it is 42.581 (30.433-44.246) and 33.605 (31.116-35.177). Aligned pageable and pinned copies also sit lower with SDMA off, and they do not show this drop.

### Small copies and ping-pong

2000 iterations for the copies. 5000 completed round trips for ping-pong. Figures are the log's median, p95, and min, in microseconds.

| Job | Kind | 4 B H2D | 4 B D2H | 64 B H2D | 64 B D2H |
|---|---|---:|---:|---:|---:|
| Default | pinned | 19.79 (21.16, 18.74) | 19.20 (20.89, 18.41) | 19.49 (21.18, 18.74) | 19.18 (21.22, 18.44) |
| Default | pageable | 19.85 (21.37, 19.07) | 32.62 (35.99, 31.08) | 19.81 (27.58, 18.93) | 33.02 (36.07, 30.75) |
| SDMA=0 | pinned | 19.77 (21.51, 18.80) | 19.21 (21.25, 18.46) | 19.58 (22.13, 18.77) | 19.24 (21.43, 18.41) |
| SDMA=0 | pageable | 20.03 (22.67, 19.14) | 32.44 (39.00, 31.07) | 19.53 (27.64, 18.84) | 32.21 (37.74, 30.73) |

| Job | Timer | Completed | RTT median us | RTT p95 | RTT min | One way us | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Default | card | 5000/5000 | 1.560 | 2.280 | 1.320 | 0.780 | PASS |
| Default | CPU | 5000/5000 | 1.510 | 2.010 | 0.910 | 0.755 | PASS |
| SDMA=0 | card | 5000/5000 | 1.560 | 1.720 | 1.320 | 0.780 | PASS |
| SDMA=0 | CPU | 5000/5000 | 1.480 | 1.710 | 0.850 | 0.740 | PASS |

## Item 1: ray tracing fixed blocks, BVH8 throughput, latency, and ACO listings (MEASURED; controls PASS, clocks not flat)

Script `scripts/round11/run_r11_rt.sh --run`. `R11_RT_REPS` defaults to 30. Log `afk-r13/slot1-results/r11-rt/rt.log`. Repeats file `rt/rt_repeats.csv` (68 cells, 30 rows each; the chain is not in that file). Listings in `rt/asm/`. The probe is Vulkan ray queries (`VK_KHR_ray_query`), not the ray-tracing pipeline.

`card-state.log`: start 13:08:04, hotspot 28 C, 8 W; end 13:17:20, exit 0, hotspot 37 C, 8 W. The run sheet places the job at 05:07-05:17 PST. Sampler span 555.208 s, 5548 samples. No `kernel-faults.txt` in this folder. The run sheet records 0 amdgpu faults for the job (READ). `analysis.txt` records 24 thermal pauses of about 0.3 s, with the hotspot falling from the 70s-90s C into the 30s-40s C.

Device line: AMD Radeon RX 9070 XT (RADV GFX1201) (`1002:7550`), driver `radv Mesa 27.0.0-devel`. Acceleration-structure limits: max geometries 16777215, max primitives 536870911, max instances 16777215. `CLOCKS before`: hotspot 28 C, power 10 W. `CLOCKS after`: hotspot 38 C (highest seen 93 C), power 24 W.

Rate cells launch 1073741824 rays, 4096 rays per thread, 30 repeats, except the occupancy cells. Spread in the tables is `(max_ms - min_ms) / median_ms`, from the log's own min, max, and median. The log's `best_Mrays_s` is the rate of the slowest repeat (the one with `max_ms`), and `worst_Mrays_s` is the rate of the fastest. The tables use the median. Hit percent is `hits_card / rays`, to three decimals, so 570425344 / 1073741824 stays 53.125%. A two-decimal format of that fraction rounds to 53.12. `hits_exact=PASS` on every rate cell that has an expected count. S0 and the S2 slivers print `hits_expected=(none)`.

The median of `rt_repeats.csv` matches `median_ms` on all 68 cells. On 57 of those 68 cells, repeat 0 is the slowest repeat. For `S1_n8_first_coh_tmaxshort_t256k`, repeat 0 is 30.549 ms at 1811 MHz and repeat 1 is already 22.829 ms at 3204 MHz. `clock_flat=NO` means the first and last shader-clock samples of that cell differ by more than the probe's 3% gate. The median is then the later, boosted repeats, and the first repeats are the cold start. The eight `clock_flat=yes` cells are `S0_n64_closest_coh_tmaxlong_t256k`, `S1_n4096_closest_coh_tmaxfloor_t256k`, `S1_n262144_closest_coh_tmaxfloor_t256k`, `EMPTY_n0_closest_coh_tmaxlong_t256k`, `S2diag_n4096_closest_shuf_tmaxlong_t256k`, `S2diag_n262144_closest_shuf_tmaxlong_t256k`, `S2axis_n4096_closest_shuf_tmaxlong_t256k`, `S2axis_n262144_closest_shuf_tmaxlong_t256k`.

### Controls and deliberate breaks

`SUMMARY controls_or_breaks_failed=0`. `VERDICT r11_rt: PASS`. This is legacy pre-fix output: Slot 1 ran the binary from an earlier commit, before a later fix added the `clock_flat_failed=N/M` qualification and `--strict-clock`. Read it as: all ray-hit correctness controls and deliberate breaks passed, while 60 of 68 rate cells had `clock_flat=NO` under the unpinned auto-boost (the legacy binary printed an unqualified PASS; current `r11_rt` qualifies the verdict and rejects `clock_flat=NO` under `--strict-clock`). The log has no `clock_flat_failed=N/M` field and none was added to it.

| Check | Result |
|---|---|
| `CONTROL S0_analytic` | hits 2080, expected 2080, PASS. The plan's hand count is 64 x 65 / 2 |
| Scene CPU checks, 4096 rays | 24 `CHECK` lines. 23 have `mismatches=0`. The other line is the moved-triangle break (cpu_hits 3026, mismatches 3). The following clean check of that scene is cpu_hits 3025, mismatches 0 |
| Clean scene checks named in the log | S1 family cpu_hits 3025; S2 diagonal cpu_hits 2098; S2 axis-aligned cpu_hits 3083; S3 rotated cpu_hits 3119. Each mismatches 0 |
| Empty top level | `EMPTY_n0` hits_card 0, hits_expected 0, `hits_exact=PASS` |
| Break: triangle moved on the card only | FAIL as required |
| Break: one extra instance | FAIL as required. Card hits 257, hand count 256 |
| Break: chain with all lanes active | FAIL as required |
| Chain value checks | `chain_check=PASS` on all 7 sizes |

### One triangle, root miss, and an empty top level

Picoseconds per ray are INFERRED: `1e6 / median_Mrays_s`. The 64-triangle floor is 106420.7 Mrays/s, 9.40 ps/ray. The eight-triangle first-hit coherent short-tMax cell is 47616.4 Mrays/s, 21.00 ps/ray. The four root-miss cells (three floors and the empty top level) sit between 106034.5 and 106455.0 Mrays/s.

| Cell | Median Mrays/s | Median ms | Min-max ms | Spread | Hits | BLAS ms | TLAS ms | Shader clock MHz | Flat |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `S0_n64_closest_coh_tmaxlong_t256k` | 47053.4 | 22.820 | 22.548-22.868 | 1.4% | 1073741824 (100.000%) | 0.15 | 0.20 | 3201->3237 | yes |
| `S1_n64_closest_coh_tmaxfloor_t256k` | 106420.7 | 10.090 | 9.826-10.368 | 5.4% | 0 (0.000%) | 0.29 | 0.13 | 3096->3220 | NO |
| `S1_n4096_closest_coh_tmaxfloor_t256k` | 106455.0 | 10.086 | 9.820-10.244 | 4.2% | 0 (0.000%) | 0.50 | 0.12 | 3128->3215 | yes |
| `S1_n262144_closest_coh_tmaxfloor_t256k` | 106034.5 | 10.126 | 9.958-10.287 | 3.2% | 0 (0.000%) | 3.01 | 0.13 | 3127->3211 | yes |
| `EMPTY_n0_closest_coh_tmaxlong_t256k` | 106345.4 | 10.097 | 9.849-10.264 | 4.1% | 0 (0.000%) | 0.00 | 0.22 | 3111->3128 | yes |

### Triangle count, 8 to 262,144

Eight cells at each count: closest or first hit, coherent or shuffled order, short tMax (1.01) or long tMax (1e4). Coherent S1 hit count is 570425344 on every one of these cells, 53.125%. Shuffled hit counts change with the triangle count. Build times are `blas_ms` and `tlas_ms` on each row.

Closest hit, long tMax, coherent, in Mrays/s: 47579.1 at 8 triangles, 39271.0 at 64 triangles, 26183.2 at 512 triangles, 27539.6 at 4096 triangles, 17288.2 at 32768 triangles, 19601.2 at 262144 triangles. The shuffled closest long-tMax cell at 262,144 triangles is 13024.4 Mrays/s. First-hit coherent long tMax at 262,144 is 19959.9 Mrays/s. `S1_n262144_first_coh_tmaxshort_t256k` ends at 2132 MHz, so its last clock is not the boosted clock the other cells reach.

| Cell | Median Mrays/s | Median ms | Min-max ms | Spread | Hits | BLAS ms | TLAS ms | Shader clock MHz | Flat |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `S1_n8_closest_coh_tmaxshort_t256k` | 47066.9 | 22.813 | 22.460-33.206 | 47.1% | 570425344 (53.125%) | 0.26 | 0.19 | 1717->3250 | NO |
| `S1_n8_closest_coh_tmaxlong_t256k` | 47579.1 | 22.567 | 22.343-31.380 | 40.0% | 570425344 (53.125%) | 0.26 | 0.19 | 1759->3274 | NO |
| `S1_n8_closest_shuf_tmaxshort_t256k` | 45389.1 | 23.656 | 23.415-32.833 | 39.8% | 483393536 (45.020%) | 0.26 | 0.19 | 1789->3281 | NO |
| `S1_n8_closest_shuf_tmaxlong_t256k` | 45310.0 | 23.698 | 23.335-38.120 | 62.4% | 483393536 (45.020%) | 0.26 | 0.19 | 1666->3256 | NO |
| `S1_n8_first_coh_tmaxshort_t256k` | 47616.4 | 22.550 | 22.390-30.549 | 36.2% | 570425344 (53.125%) | 0.26 | 0.19 | 1811->3288 | NO |
| `S1_n8_first_coh_tmaxlong_t256k` | 47845.1 | 22.442 | 22.095-31.609 | 42.4% | 570425344 (53.125%) | 0.26 | 0.19 | 1788->3297 | NO |
| `S1_n8_first_shuf_tmaxshort_t256k` | 45367.9 | 23.667 | 23.332-31.538 | 34.7% | 483393536 (45.020%) | 0.26 | 0.19 | 1754->3271 | NO |
| `S1_n8_first_shuf_tmaxlong_t256k` | 45313.6 | 23.696 | 23.506-34.333 | 45.7% | 483393536 (45.020%) | 0.26 | 0.19 | 1783->3270 | NO |
| `S1_n64_closest_coh_tmaxshort_t256k` | 39241.6 | 27.362 | 26.978-33.251 | 22.9% | 570425344 (53.125%) | 0.86 | 0.20 | 2042->3291 | NO |
| `S1_n64_closest_coh_tmaxlong_t256k` | 39271.0 | 27.342 | 27.081-38.390 | 41.4% | 570425344 (53.125%) | 0.86 | 0.20 | 1804->3288 | NO |
| `S1_n64_closest_shuf_tmaxshort_t256k` | 37366.8 | 28.735 | 28.146-37.148 | 31.3% | 560529408 (52.203%) | 0.86 | 0.20 | 1752->3210 | NO |
| `S1_n64_closest_shuf_tmaxlong_t256k` | 37359.9 | 28.741 | 28.364-37.777 | 32.8% | 560529408 (52.203%) | 0.86 | 0.20 | 1762->3218 | NO |
| `S1_n64_first_coh_tmaxshort_t256k` | 38563.2 | 27.844 | 27.599-37.473 | 35.5% | 570425344 (53.125%) | 0.86 | 0.20 | 1776->3258 | NO |
| `S1_n64_first_coh_tmaxlong_t256k` | 38968.0 | 27.554 | 27.105-37.158 | 36.5% | 570425344 (53.125%) | 0.86 | 0.20 | 1817->3294 | NO |
| `S1_n64_first_shuf_tmaxshort_t256k` | 37275.8 | 28.805 | 27.928-37.265 | 32.4% | 560529408 (52.203%) | 0.86 | 0.20 | 1760->3210 | NO |
| `S1_n64_first_shuf_tmaxlong_t256k` | 37292.3 | 28.793 | 28.326-38.071 | 33.8% | 560529408 (52.203%) | 0.86 | 0.20 | 1801->3210 | NO |
| `S1_n512_closest_coh_tmaxshort_t256k` | 25732.6 | 41.727 | 40.742-50.853 | 24.2% | 570425344 (53.125%) | 0.54 | 0.13 | 1814->3279 | NO |
| `S1_n512_closest_coh_tmaxlong_t256k` | 26183.2 | 41.009 | 40.364-48.915 | 20.9% | 570425344 (53.125%) | 0.54 | 0.13 | 1864->3318 | NO |
| `S1_n512_closest_shuf_tmaxshort_t256k` | 24669.2 | 43.526 | 42.553-50.602 | 18.5% | 550772736 (51.295%) | 0.54 | 0.13 | 1878->3225 | NO |
| `S1_n512_closest_shuf_tmaxlong_t256k` | 24621.4 | 43.610 | 43.235-59.259 | 36.7% | 550772736 (51.295%) | 0.54 | 0.13 | 1658->3229 | NO |
| `S1_n512_first_coh_tmaxshort_t256k` | 26179.6 | 41.014 | 40.201-55.459 | 37.2% | 570425344 (53.125%) | 0.54 | 0.13 | 1552->3248 | NO |
| `S1_n512_first_coh_tmaxlong_t256k` | 26653.4 | 40.285 | 39.890-46.638 | 16.8% | 570425344 (53.125%) | 0.54 | 0.13 | 2054->3302 | NO |
| `S1_n512_first_shuf_tmaxshort_t256k` | 25065.9 | 42.837 | 41.912-49.559 | 17.9% | 550772736 (51.295%) | 0.54 | 0.13 | 1917->3216 | NO |
| `S1_n512_first_shuf_tmaxlong_t256k` | 25094.9 | 42.787 | 42.169-51.251 | 21.2% | 550772736 (51.295%) | 0.54 | 0.13 | 1892->3222 | NO |
| `S1_n4096_closest_coh_tmaxshort_t256k` | 27471.6 | 39.086 | 38.892-48.986 | 25.8% | 570425344 (53.125%) | 0.49 | 0.10 | 1763->3278 | NO |
| `S1_n4096_closest_coh_tmaxlong_t256k` | 27539.6 | 38.989 | 38.490-48.697 | 26.2% | 570425344 (53.125%) | 0.49 | 0.10 | 1808->3286 | NO |
| `S1_n4096_closest_shuf_tmaxshort_t256k` | 24929.4 | 43.071 | 42.041-50.682 | 20.1% | 570185728 (53.103%) | 0.49 | 0.10 | 1786->3148 | NO |
| `S1_n4096_closest_shuf_tmaxlong_t256k` | 24914.9 | 43.096 | 42.079-49.908 | 18.2% | 570185728 (53.103%) | 0.49 | 0.10 | 1863->3125 | NO |
| `S1_n4096_first_coh_tmaxshort_t256k` | 27333.5 | 39.283 | 38.577-50.010 | 29.1% | 570425344 (53.125%) | 0.49 | 0.10 | 1784->3289 | NO |
| `S1_n4096_first_coh_tmaxlong_t256k` | 27575.6 | 38.938 | 38.687-49.850 | 28.7% | 570425344 (53.125%) | 0.49 | 0.10 | 1791->3288 | NO |
| `S1_n4096_first_shuf_tmaxshort_t256k` | 24944.9 | 43.045 | 42.015-47.431 | 12.6% | 570185728 (53.103%) | 0.49 | 0.10 | 2116->3145 | NO |
| `S1_n4096_first_shuf_tmaxlong_t256k` | 24900.7 | 43.121 | 42.090-49.978 | 18.3% | 570185728 (53.103%) | 0.49 | 0.10 | 1818->3129 | NO |
| `S1_n32768_closest_coh_tmaxshort_t256k` | 17257.2 | 62.220 | 61.953-70.679 | 14.0% | 570425344 (53.125%) | 1.83 | 0.15 | 2134->3235 | NO |
| `S1_n32768_closest_coh_tmaxlong_t256k` | 17288.2 | 62.108 | 61.550-71.483 | 16.0% | 570425344 (53.125%) | 1.83 | 0.15 | 2018->3226 | NO |
| `S1_n32768_closest_shuf_tmaxshort_t256k` | 15089.9 | 71.156 | 69.314-76.524 | 10.1% | 567289984 (52.833%) | 1.83 | 0.15 | 2167->3134 | NO |
| `S1_n32768_closest_shuf_tmaxlong_t256k` | 15084.7 | 71.181 | 69.149-78.698 | 13.4% | 567289984 (52.833%) | 1.83 | 0.15 | 1991->3126 | NO |
| `S1_n32768_first_coh_tmaxshort_t256k` | 17782.5 | 60.382 | 59.823-69.415 | 15.9% | 570425344 (53.125%) | 1.83 | 0.15 | 2064->3241 | NO |
| `S1_n32768_first_coh_tmaxlong_t256k` | 17843.0 | 60.177 | 59.584-71.023 | 19.0% | 570425344 (53.125%) | 1.83 | 0.15 | 1913->3232 | NO |
| `S1_n32768_first_shuf_tmaxshort_t256k` | 15477.6 | 69.374 | 67.368-75.977 | 12.4% | 567289984 (52.833%) | 1.83 | 0.15 | 2038->3129 | NO |
| `S1_n32768_first_shuf_tmaxlong_t256k` | 15486.8 | 69.333 | 67.607-78.158 | 15.2% | 567289984 (52.833%) | 1.83 | 0.15 | 1910->3131 | NO |
| `S1_n262144_closest_coh_tmaxshort_t256k` | 19418.6 | 55.295 | 54.642-62.360 | 14.0% | 570425344 (53.125%) | 5.89 | 0.88 | 2148->3256 | NO |
| `S1_n262144_closest_coh_tmaxlong_t256k` | 19601.2 | 54.779 | 54.426-64.939 | 19.2% | 570425344 (53.125%) | 5.89 | 0.88 | 1923->3279 | NO |
| `S1_n262144_closest_shuf_tmaxshort_t256k` | 13137.9 | 81.729 | 78.947-90.417 | 14.0% | 570575968 (53.139%) | 5.89 | 0.88 | 1945->3210 | NO |
| `S1_n262144_closest_shuf_tmaxlong_t256k` | 13024.4 | 82.441 | 79.557-87.633 | 9.8% | 570575968 (53.139%) | 5.89 | 0.88 | 2118->3204 | NO |
| `S1_n262144_first_coh_tmaxshort_t256k` | 19858.1 | 54.071 | 53.376-63.093 | 18.0% | 570425344 (53.125%) | 5.89 | 0.88 | 1914->2132 | NO |
| `S1_n262144_first_coh_tmaxlong_t256k` | 19959.9 | 53.795 | 53.311-63.935 | 19.7% | 570425344 (53.125%) | 5.89 | 0.88 | 1860->3234 | NO |
| `S1_n262144_first_shuf_tmaxshort_t256k` | 13061.2 | 82.209 | 80.026-87.216 | 8.7% | 570575968 (53.139%) | 5.89 | 0.88 | 2117->3206 | NO |
| `S1_n262144_first_shuf_tmaxlong_t256k` | 13155.8 | 81.618 | 79.794-86.185 | 7.8% | 570575968 (53.139%) | 5.89 | 0.88 | 2216->3130 | NO |

### Slivers and instances

S2 is shuffled, closest hit, long tMax: diagonal slivers, then axis-aligned slivers. The 4096-ray CPU checks are the control sample (2098 and 3083 hits), not the rate-cell hit count. Whether the driver builds oriented boxes was not tested (`expect.md`).

| Cell | Median Mrays/s | Median ms | Min-max ms | Spread | Hits | BLAS ms | TLAS ms | Shader clock MHz | Flat |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `S2diag_n4096_closest_shuf_tmaxlong_t256k` | 24609.3 | 43.632 | 42.592-47.451 | 11.1% | 33133568 (3.086%) | 3.84 | 6.70 | 3195->3190 | yes |
| `S2diag_n262144_closest_shuf_tmaxlong_t256k` | 13470.6 | 79.710 | 78.349-88.497 | 12.7% | 33112448 (3.084%) | 6.05 | 1.68 | 3193->3193 | yes |
| `S2axis_n4096_closest_shuf_tmaxlong_t256k` | 34697.2 | 30.946 | 30.685-40.862 | 32.9% | 17085440 (1.591%) | 0.46 | 0.10 | 3163->3184 | yes |
| `S2axis_n262144_closest_shuf_tmaxlong_t256k` | 21803.6 | 49.246 | 48.747-57.501 | 17.8% | 20968736 (1.953%) | 2.66 | 0.15 | 3187->3199 | yes |

S3 is one triangle instanced 1, 16, 256, or 4096 times, plain or rotated 180 degrees, shuffled, closest hit, long tMax. The plain scene's 4096-ray check is cpu_hits 3025. The rotated scene's is 3119.

| Cell | Median Mrays/s | Median ms | Min-max ms | Spread | Hits | BLAS ms | TLAS ms | Shader clock MHz | Flat |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `S3_n1_closest_shuf_tmaxlong_t256k` | 43891.7 | 24.463 | 24.226-34.206 | 40.8% | 566231040 (52.734%) | 1.33 | 1.80 | 2162->3174 | NO |
| `S3_n16_closest_shuf_tmaxlong_t256k` | 37320.4 | 28.771 | 28.291-38.416 | 35.2% | 546045952 (50.854%) | 0.75 | 2.12 | 1759->3236 | NO |
| `S3_n256_closest_shuf_tmaxlong_t256k` | 30205.9 | 35.547 | 35.077-45.146 | 28.3% | 567672832 (52.869%) | 2.09 | 0.22 | 1766->3244 | NO |
| `S3_n4096_closest_shuf_tmaxlong_t256k` | 24481.3 | 43.860 | 43.053-54.182 | 25.4% | 570185728 (53.103%) | 1.20 | 3.70 | 1763->3146 | NO |
| `S3_n1_rot_closest_shuf_tmaxlong_t256k` | 44006.2 | 24.400 | 23.891-32.769 | 36.4% | 507510784 (47.266%) | 0.84 | 0.68 | 1751->3237 | NO |
| `S3_n16_rot_closest_shuf_tmaxlong_t256k` | 37362.4 | 28.739 | 28.209-36.075 | 27.4% | 527695872 (49.146%) | 1.63 | 1.75 | 1801->3209 | NO |
| `S3_n256_rot_closest_shuf_tmaxlong_t256k` | 30317.0 | 35.417 | 35.049-45.729 | 30.2% | 506068992 (47.131%) | 0.17 | 0.27 | 1766->3213 | NO |
| `S3_n4096_rot_closest_shuf_tmaxlong_t256k` | 24403.3 | 44.000 | 43.050-47.430 | 10.0% | 503556096 (46.897%) | 0.21 | 0.26 | 2324->3127 | NO |

### Occupancy

Same 4096-triangle shuffled closest-hit long-tMax scene, fewer threads. Invocations are `rays / rpt`. `blas_ms` is 0.32 and `tlas_ms` is 1.96 on all three. `clock_flat=NO` on all three.

| Cell | Rays | rpt | Invocations | Median Mrays/s | Median ms (min-max) | Spread | Hits | Shader clock MHz | Flat |
|---|---:|---:|---:|---:|---|---:|---:|---|---|
| `S1_n4096_closest_shuf_tmaxlong_t4k_occ` | 16777216 | 4096 | 4096 | 2211.0 | 7.588 (7.543-12.678) | 67.7% | 8909152 | 1924->3214 | NO |
| `S1_n4096_closest_shuf_tmaxlong_t16k_occ` | 67108864 | 4096 | 16384 | 8199.4 | 8.185 (8.141-14.084) | 72.6% | 35636608 | 1855->3190 | NO |
| `S1_n4096_closest_shuf_tmaxlong_t64k_occ` | 268435456 | 4096 | 65536 | 25410.1 | 10.564 (10.425-19.034) | 81.5% | 142546432 | 1781->3200 | NO |

### Serial dependent-ray chain

One lane, 4096 rays whose origin depends on the previous hit, 15 repeats. The 677 to 1731 ns/ray range is the series minimum (677.4 at 1 triangle, 677.3 at 8) and the series maximum (1730.8 at 32,768). The 262,144-triangle point is 1612.2 ns/ray, 5203 shader cycles, at 3227 MHz. Cycles are the probe's shader-clock cycles at the clock on that row.

| Triangles | ns/ray | Shader cycles/ray | Shader clock MHz | Wave ns/ray | Chain / wave | Check |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 677.4 | 2185 | 3226 | 22.82 | 29.68 | PASS |
| 8 | 677.3 | 2186 | 3227 | 22.71 | 29.83 | PASS |
| 64 | 802.7 | 2590 | 3226 | 27.22 | 29.49 | PASS |
| 512 | 1041.1 | 3359 | 3226 | 44.80 | 23.24 | PASS |
| 4096 | 1069.8 | 3452 | 3227 | 45.26 | 23.63 | PASS |
| 32768 | 1730.8 | 5585 | 3227 | 115.56 | 14.98 | PASS |
| 262144 | 1612.2 | 5203 | 3227 | 99.49 | 16.20 | PASS |

### ACO listings

The probe's own count, then the opcode lines in the listing.

| Listing | `image_bvh8_intersect_ray` | plain / bvh64 / dual | `ds_bvh_stack` | All `bvh` mentions |
|---|---:|---:|---:|---:|
| `rt_check.s` | 1 | 0 / 0 / 0 | 1 | 5 |
| `rt_rate.s` | 1 | 0 / 0 / 0 | 1 | 4 |
| `rt_chain.s` | 1 | 0 / 0 / 0 | 1 | 4 |

Each file contains one `image_bvh8_intersect_ray` and one `ds_bvh_stack_push8_pop1_rtn_b32`. `rt_check.s` has three `s_wait_bvhcnt` lines; `rt_rate.s` and `rt_chain.s` have two. No `image_bvh_intersect_ray`, `image_bvh64_intersect_ray`, or `image_bvh_dual_intersect_ray`.

### Clock and temperature

The all-sample shader-clock mean is 512 MHz and the median is 0 MHz, because most of the 5548 samples are the idle gaps between cells. The figures in the plain summary are the 873 samples at or above 2000 MHz: mean 3144.8 MHz, 2004 to 3349; mean reported power 348.6 W. Peak reported power over every sample is 464 W, and `sock_power_W` peaks at the same 464 W. Both are the driver's reported power under the 1.2x regulator mod, not board power. Peak hotspot is 93 C.

A non-zero `throttle_status` or `indep_throttle_status` is set on 875 of 5548 samples (15.77%). Inside the samples at or above 2000 MHz it is 825 of 873 (94.5%). The legacy words seen are 0x4000 (279), 0x2 (276), 0x4002 (155), 0xc000 (93), 0xc002 (49), 0x8002 (22), and 0x8000 (1). This file does not rename those bits. `pcie_gts_x10` stayed 25 and width stayed 16 for all 5548 samples. Fan rpm reached 14696. The `fan_pct` column holds the same numbers as `fan_rpm` (it also reaches 14696), so it is not a percent in this CSV.

### Not run

No rate cell has more than 262144 triangles, so `--big` (the 2-million-triangle scene) was not used. `scripts/round11/run_r11_rt.sh` lists the ray-tracing pipeline, an unbuilt top level, acceleration-structure byte reads, and the 2M scene without `--big` under `PLAN not done` (READ). `--smoke` is a shorter job; this folder is the full `--run` without `--big`.

## Item 4a: video engine, VCN decode and encode (MEASURED, Slot 2)

Folder `afk-r13/slot2-results/r11-video/`. `card-state.log`: start 14:43:36 UTC, end 14:55:15 UTC, exit 0. Those stamps are 06:43:36-06:55:15 PST. The console does not contain `--break-device`. `run_r11_video.sh --run` without that flag does not decode on the iGPU node, so deliberate break 1 has no hardware result in this slot. The copied `expect.md` records the other two breaks as CPU checks (a corrupted byte, and PSNR against a different source). This write-up did not re-run those checks.

`analysis.txt` has one idle line and 72 `CELL` lines (36 decode, 36 encode). Every cell prints `activity=ok failed=0`. `video.log` has the 12 controls: 6 `decode_md5` PASS, 6 `encode_check` PASS. `video.journal` is empty. The runner prints a kernel-log fault and returns 13 when it sees one, and returns 12 when the log cannot be read (READ, `r11_common.sh`). This job exited 0, so that check returned clean.

### Idle baseline and encode quality

`IDLE baseline (max vclk0 dclk0 vclk1 dclk1 mm_act rows): 25 25 25 25 0 600`. Over those 600 rows the highest video clocks are 25 MHz and `mm_act` stays 0.

The six encode controls are one 8 Mbit/s encode per codec and size (READ, `verify_hw` in `run_r11_video.sh` passes `-b:v 8M`). They are not the throughput cells. Each line prints `floor=26`. The copied `expect.md` sets that floor from the CPU reference measured before any hardware encode: the lowest CPU figure is 29.06 dB (AV1, 1080p, 8 Mbit/s), and 26 is about 3 dB under it. That same file still says a built-in default of 35 would fail the CPU encoders. The script as it stands defaults `R11_PSNR_FLOOR` to 26. The Slot 2 log shows the floor that was applied. It does not show which default the process inherited.

| Check | Hardware PSNR Y (dB) | Floor | CPU reference, 8 Mbit/s (READ, `expect.md`) |
|---|---:|---:|---:|
| H.264 1080p | 34.841266 | 26 | 33.93 |
| HEVC 1080p | 35.375551 | 26 | 34.85 |
| AV1 1080p | 29.198040 | 26 | 29.06 |
| H.264 4k | 33.575672 | 26 | 33.40 |
| HEVC 4k | 34.313944 | 26 | 33.84 |
| AV1 4k | 30.735967 | 26 | 30.67 |

CPU reference at the other rates, same file, not a hardware control: 1080p 40 Mbit/s is 35.68 / 36.12 / 29.22, and 4k 40 Mbit/s is 33.99 / 35.01 / 30.84 (H.264 / HEVC / AV1).

### Decode throughput

`fps_total`, with `fps_per_proc` in parentheses. `hw` is the VA-API device path. `sys` is the other path the cell name prints.

| Codec | Size | Path | 1 stream | 2 streams | 3 streams |
|---|---|---|---:|---:|---:|
| H.264 | 1080p | hw | 371.5 (371.5) | 373.1 (186.6) | 327.1 (109.0) |
| H.264 | 1080p | sys | 254.6 (254.6) | 428.9 (214.4) | 436.5 (145.5) |
| H.264 | 4k | hw | 266.6 (266.6) | 274.1 (137.1) | 274.5 (91.5) |
| H.264 | 4k | sys | 138.5 (138.5) | 194.0 (97.0) | 199.9 (66.6) |
| HEVC | 1080p | hw | 397.0 (397.0) | 407.2 (203.6) | 408.9 (136.3) |
| HEVC | 1080p | sys | 230.7 (230.7) | 571.9 (285.9) | 557.2 (185.7) |
| HEVC | 4k | hw | 318.6 (318.6) | 322.9 (161.4) | 319.2 (106.4) |
| HEVC | 4k | sys | 146.2 (146.2) | 264.8 (132.4) | 326.6 (108.9) |
| AV1 | 1080p | hw | 389.1 (389.1) | 394.3 (197.2) | 391.7 (130.6) |
| AV1 | 1080p | sys | 209.1 (209.1) | 274.7 (137.3) | 349.6 (116.5) |
| AV1 | 4k | hw | 288.6 (288.6) | 292.5 (146.2) | 293.0 (97.7) |
| AV1 | 4k | sys | 107.7 (107.7) | 184.2 (92.1) | 232.2 (77.4) |

### Encode throughput

Same units. H.264 has `qdefault` and `q0`. HEVC and AV1 cells are `qdefault` only. 4k cells are 40 Mbit/s only. There is no 4k 8 Mbit/s throughput row.

| Codec | Size | Rate | Quality | 1 stream | 2 streams | 3 streams |
|---|---|---|---|---:|---:|---:|
| H.264 | 1080p | 8 Mbit/s | qdefault | 337.9 (337.9) | 382.5 (191.2) | 392.9 (131.0) |
| H.264 | 1080p | 8 Mbit/s | q0 | 335.8 (335.8) | 383.7 (191.8) | 392.5 (130.8) |
| H.264 | 1080p | 40 Mbit/s | qdefault | 310.4 (310.4) | 360.6 (180.3) | 367.2 (122.4) |
| H.264 | 1080p | 40 Mbit/s | q0 | 316.8 (316.8) | 357.8 (178.9) | 367.0 (122.3) |
| H.264 | 4k | 40 Mbit/s | qdefault | 103.1 (103.1) | 110.8 (55.4) | 111.4 (37.1) |
| H.264 | 4k | 40 Mbit/s | q0 | 103.6 (103.6) | 110.6 (55.3) | 111.0 (37.0) |
| HEVC | 1080p | 8 Mbit/s | qdefault | 589.1 (589.1) | 804.6 (402.3) | 806.6 (268.9) |
| HEVC | 1080p | 40 Mbit/s | qdefault | 487.6 (487.6) | 597.3 (298.6) | 619.7 (206.6) |
| HEVC | 4k | 40 Mbit/s | qdefault | 223.6 (223.6) | 248.6 (124.3) | 252.8 (84.3) |
| AV1 | 1080p | 8 Mbit/s | qdefault | 287.8 (287.8) | 321.8 (160.9) | 327.9 (109.3) |
| AV1 | 1080p | 40 Mbit/s | qdefault | 258.1 (258.1) | 284.2 (142.1) | 289.8 (96.6) |
| AV1 | 4k | 40 Mbit/s | qdefault | 89.2 (89.2) | 94.2 (47.1) | 95.0 (31.7) |

### Telemetry

`video.sampler/samples.csv`: 6993 rows, period 100 ms, span 699.200 s, `gpu_metrics` v1_3. `power_W` is hwmon `power1_average` (READ, the CSV comment). Reported watts are the 1.2x-mod figures, not board power.

| Field | Min | Median | Max |
|---|---:|---:|---:|
| Hotspot C | 35 | 43 | 48 |
| Reported power W | 6 | 34 | 47 |
| `vclk0` and `vclk1` MHz | 25 | 2826 | 2933 |
| `dclk0` and `dclk1` MHz | 25 | 2119 | 2200 |
| `mm_act` percent | 0 | 100 | 100 |
| Shader clock MHz | 0 | 0 | 1986 |
| Memory clock MHz | 96 | 772 | 772 |
| Fan rpm | 0 | 0 | 0 |

`mm_act` is non-zero on 5526 of 6993 rows. Memory-clock values in the file are 96, 456, and 772 MHz. `throttle_status` and `indep_throttle_status` are `0x0` on every row. `pcie_gts_x10` is 25 and `pcie_w` is 16 on every row, which is the same 2.5 GT/s x16 field the PCIe section reads. `card-state.log` at both ends reads `32.0 GT/s PCIe x16`, hotspot 37 C then 39 C, power 8 W then 10 W, shader clock 0, memory 96 MHz. The two link readings disagree, as they did on the copy job. The fan stayed at 0 rpm for the whole sampler file.

### What was not run

`--break-device` (decode one clip on the iGPU node `renderD129`, device 0x13c0, and require the dGPU activity check to fail) was not invoked. The copied expectation describes that break. It is untested on this card in this slot.

## Item 3: graphics fixed blocks (MEASURED)

### Slot 1 (smoke PASS; full run stopped at 95 C)

Smoke folder `afk-r13/slot1-results/r11-gfx-smoke/`. `VERDICT r11_gfx: PASS`, `controls_or_breaks_failed=0`. `COMPRESSION_CONTROL` extension not available. Controls that passed: `fill_clean`, `texture_gradient`, `raster_exact` (visible 3345620), `fill_RGBA8_1MiB`, `raster_leg256_colormask0`, `tex_RGBA8_256_bilinear`. Breaks that printed FAIL as required: `scissor_half`, `skip_last_draw`, `nearest_vs_bilinear` (60 of 64 points), `culled_vs_visible`. `compression_not_applied` is n/a because `VK_EXT_image_compression_control` is missing. `card-state.log` start and end are both stamped 13:19:23, exit 0. `r11-gfx-smoke.console` ends `DRIVER exit=0`. No `kernel-faults.txt` in the folder. The run sheet records 0 amdgpu faults (READ).

Full run folder `afk-r13/slot1-results/r11-gfx/`. `guard.log`: `GUARD: hotspot 95 C at the 95 C limit, stopping job 427248`. `card-state.log`: start 13:19:40, hotspot 31 C, 11 W; end 13:20:30, exit 143, hotspot 55 C, 12 W (50 s of wall clock, already cooling). `r11-gfx.console` ends `DRIVER exit=8`. The same compression extension is absent. Eight controls failed before any timing and are not rates:

- `fill_RGB10A2` at 1, 4, 32, and 512 MiB: occlusion count ok, read-back mismatches 512.
- `fill_RGBA8_ms4` at 4 and 32 MiB, and `fill_RGBA16F_ms4` at 4 and 32 MiB: occlusion count wrong, read-back mismatches 0.

Sampler: 492 samples, span 49.211 s. Hotspot 31 to 96 C. The fan is 0 rpm for the first 321 samples, through t=32.11 s. In that window the hotspot reaches 82 C, reported power reaches 332 W, and the shader clock reaches 3566 MHz. Those three peaks are different samples. The shader-clock maximum of the whole file is 3566 MHz at t=30.11 s, fan 0 rpm, hotspot 52 C, power 125 W. The first sample with the fan moving is t=32.21 s, 149 rpm, hotspot 83 C, power 310 W, shader clock 3345 MHz. Reported power then peaks at 407 W at t=39.81 s, fan 3556 rpm, hotspot 87 C, shader clock 3302 MHz. Hotspot peaks at 96 C at t=33.11 s, fan 2073 rpm, power 398 W, shader clock 3315 MHz. Memory clock median 96 MHz, max 1408 MHz (350 of 492 samples at 96). Throttle words non-zero on 74 of 492 samples (15.04%).

Fill rates from this stopped run are not claims. The run sheet says a fix was in progress for Slot 2 (READ).

### Slot 2 check, then the full run (VERDICT FAIL, rates uninterpreted)

Short check `afk-r13/slot2-results/r11-gfx-check/`. `card-state.log`: 14:55:46 to 14:55:49 UTC, exit 1 (06:55:46-06:55:49 PST). Console ends `DRIVER exit=14`. The runner returns 14 when the probe's own status is 1 (READ, `r11_locked_job`). `VERDICT r11_gfx: FAIL`, `controls_or_breaks_failed=1 burst_over=0 spread_flagged=1 strict_spread=0`. `VK_EXT_image_compression_control` is absent. Passed: `fill_clean`, `texture_gradient`, `raster_exact` (visible 3345620), `fill_RGBA8_1MiB`, `fill_RGB10A2_1MiB`, `raster_leg256_colormask0`, `tex_RGBA8_256_bilinear`. Breaks that failed as required: `scissor_half`, `skip_last_draw`, `nearest_vs_bilinear` (60 of 64), `culled_vs_visible`. `compression_not_applied` is n/a. The one CTRLFAIL is `fill_RGBA8_ms4_4MiB` (occlusion count wrong, read-back mismatches 0, not timed). The one `SPREAD>10%` cell is `tex_RGBA8_256_bilinear`. `memory_anchor` was not measured, because the 512 MiB cell did not run. `CLOCKS before` hotspot 36 C, 10 W. `CLOCKS after` hotspot 40 C, "highest seen 72 C", 8 W. `card-state.log` itself reads 12 W at the start (hotspot 36 C) and 12 W at the end (hotspot 40 C, shader clock 585 MHz). The check sampler (26 rows, span 2.500 s) peaks at hotspot 64 C, reported power 282 W, shader clock 3231 MHz. The probe's "highest seen 72 C" is above that sampler maximum. `gfx.journal` is empty.

Full run `afk-r13/slot2-results/r11-gfx/`. Device line: AMD Radeon RX 9070 XT (RADV GFX1201), driver `radv Mesa 27.0.0-devel`. `card-state.log`: start 14:56:14 UTC, hotspot 36 C, 11 W, link `32.0 GT/s PCIe x16`, memory 96 MHz; end 14:58:12 UTC, exit 1, hotspot 49 C, 53 W, memory level 4 at 1124 MHz. That is 06:56:14-06:58:12 PST. Console ends `DRIVER exit=14`, the same probe-status mapping as the check. No `guard.log`. The 95 C guard did not stop this job. `gfx.journal` is empty, and the runner reaches exit 14 only after the kernel-log check returns clean (READ, the check runs first and returns 12 or 13 on a fault). `gfx.log` has 140 `CELL` lines, 29 of them marked `SPREAD>10%`, and 8 `CTRLFAIL` lines.

`analysis.txt`: `BURST target 100 ms, 5 repeats, limit 250 ms`. `SUMMARY controls_or_breaks_failed=8 burst_over=3 spread_flagged=29 strict_spread=0`. `VERDICT r11_gfx: FAIL`. `COMPRESSION_CONTROL` extension not available, so break 5 and the disabled-compression arm are n/a (`COMPSWITCH NO SWITCH AVAILABLE`). Four of five deliberate breaks failed as required: `scissor_half`, `skip_last_draw`, `nearest_vs_bilinear_analytic` (60 of 64 points), `culled_vs_visible_expectation` (culled count 0 against 3345620).

The runner's default `--cool-c` is 72 (READ, `R11_GFX_COOL_C`). The log's pause lines match that default: `72 C -> ... (target below 66 C)` before cells, and `target below 60 C` before groups. One cell line is `THERMAL paused 0.3 s before fill_RGBA8_32MiB: 72 C -> 39 C (target below 66 C)`. The suite's sampler spans 118.000 s (1181 rows).

**What passed.** `CONTROL fill fill_RGB10A2_1MiB`, `_4MiB`, `_32MiB`, and `_512MiB` are all PASS. Slot 1 failed these four with 512 read-back mismatches. `r11_gfx_ref.h` says a colour between two RGB10A2 grid points (its example is 0.2, which is 204.6 on a 10-bit grid) read back wrong on all 512 pixels, and the control colours for that format are now required to sit on the grid (READ). The Slot 2 controls pass.

**Why the verdict is FAIL.** Eight control-or-break failures, three of them also `burst_over`, and 29 cells flagged for spread. The fill and compression GB/s lines in the analysis are not hardware limits. The analysis says so on the memory-anchor failure: `the compression and size results are not interpreted`.

1. **Four 4x MSAA occlusion queries, kept untimed.** Read-back mismatches are 0. The count does not match the expected count. Printed diagnostic (`count`, `expected`, `ratio`, `count_div_3W`, `H`):

| Cell | Count | Expected | Ratio | count / (3 W) | H |
|---|---:|---:|---:|---:|---:|
| `fill_RGBA8_ms4_4MiB` | 878592 | 12582912 | 0.06982 | 286 | 1024 |
| `fill_RGBA8_ms4_32MiB` | 6660096 | 100663296 | 0.06616 | 542 | 2048 |
| `fill_RGBA16F_ms4_4MiB` | 485376 | 6291456 | 0.07715 | 158 | 512 |
| `fill_RGBA16F_ms4_32MiB` | 3330048 | 50331648 | 0.06616 | 542 | 2048 |

`count / (3 W)` is the log's `count_div_3W`. W is not printed. From `count / (3 * count_div_3W)` it is 1024, 4096, 1024, and 2048 (INFERRED). For each row, `H/4 + 30` equals `count_div_3W` exactly, so the count equals `3 * W * (H/4 + 30)` (INFERRED). The log's own words for the expected count are `passes x W x H x samples`. The printed ratios are 0.06616 to 0.07715. One sixteenth is 0.0625, and none of the four ratios is that value. A standalone query repro is not in this folder. These four cells were not timed.

2. **Memory anchor.** `CONTROL memory_anchor default-compression 512MiB random write 855.3 GB/s (bus 640 GB/s stock, 720.9 at 1408 MHz) FAIL`. The same summary prints the 512 MiB pair beside it: write solid 3286.2 GB/s and random 855.3 GB/s (ratio 3.84); read solid 395.1 GB/s and random 110.7 GB/s (ratio 3.57). The random read is the third `BURSTOVER` below. One reading of the 855.3 GB/s write, above both ceilings the control names, is that repeated full-screen draws to one target inside a single render pass, with no barrier between them, let the ROP or its cache combine writes across passes (INFERRED). The control's rule is the one to follow: those rates are not a measured bus limit.

3. **Three bursts over the 250 ms limit.** The CTRLFAIL lines name the longest burst, which is not the cell's kept-repeat `max_ms`:

| Cell | Longest burst named by CTRLFAIL | Kept repeats, min-max ms | Median ms |
|---|---:|---|---:|
| `fill_RGBA8_blend_512MiB` | 284.4 ms | 34.707-34.884 | 34.722 |
| `fill_RGBA16F_blend_512MiB` | 284.9 ms | 34.764-34.847 | 34.833 |
| `comp_512MiB_random_default_read` | 265.9 ms | 100.281-102.838 | 101.875 |

The blend cells' fill controls printed PASS before the timing line. The failure is the burst length.

4. **29 cells `SPREAD>10%`.** The analysis warning says the repeats are more than 10% apart under unpinned auto-boost clocks, and that this is not an unqualified clean pass. `strict_spread=0`.

**Clocks and temperature, full run.** `CLOCKS before`: hotspot 35 C, 9 W. `CLOCKS after`: hotspot 51 C, "highest seen 87 C", power 163 W. Sampler, which is the during-job record: hotspot 35 to 86 C, median 56 C; reported power 6 to 411 W, median 135 W (`sock_power_W` matches); shader clock 0 to 3583 MHz, median 3377; memory clock 96 to 1408 MHz, median 96, values 96, 456, 772, 875, 1124, 1408; video clocks stuck at 25 MHz; `mm_act` 0; fan 0 to 20897 rpm, median 1702; `pcie_gts_x10` 25 and width 16 on all 1181 rows. `throttle_status` is `0x2` on 242 rows and `0x0` on the other 939. `indep_throttle_status` is `0x1000000000` on those same 242 rows. This note does not rename those bits. The probe's "highest seen 87 C" is one degree above the sampler maximum of 86 C. Both numbers are printed. The card-state end snapshot is 49 C and 53 W, which is the end of the job, not the sampler peak.

The 32 MiB and 512 MiB compression summaries, and the `SUMMARY bound` GPix/s lines, stay in the log. They are not repeated here as rates. The anchor failure is why.

## Item 2: long thermal load (parked)

Not run in either slot. It needs the owner present. No duration was measured.

## Rerun

These commands are the runners' own usage. They take the GPU lock. This write-up did not launch them.

```sh
scripts/round11/run_r11_pcie.sh --run
scripts/round11/run_r11_rt.sh --run
scripts/round11/run_r11_rt.sh --cpu-checks
scripts/round11/run_r11_gfx.sh --run --smoke
scripts/round11/run_r11_gfx.sh --run
scripts/round11/run_r11_video.sh --make-inputs
scripts/round11/run_r11_video.sh --cpu-reference
scripts/round11/run_r11_video.sh --run
scripts/round11/run_r11_video.sh --run --break-device
```

`run_r11_rt.sh --big` adds the 2-million-triangle scene. `--smoke` is the short control job. The PCIe runner's second job sets `HSA_ENABLE_SDMA=0`. `--break-device` with `--run` decodes one clip on the iGPU node and requires the dGPU activity check to fail. Slot 2 did not pass it. The graphics runner's default `--cool-c` is 72.

## Open

- Displays-on against displays-off for the card-wide pauses, same binary. Until a displays-on run records events, the 16-wave chance figure stays open. See `ext-review-checks.md`.
- Which link-speed field is live during a copy, and during the Slot 2 video job. Sysfs says 32 GT/s. The 100 ms `gpu_metrics` field says 2.5 GT/s, including while the video cells ran.
- Ray-tracing rates at a raised memory clock. The Slot 1 ray-tracing job's memory clock stayed at 96 MHz.
- `--break-device` on the video engine. Slot 2's 72 cells and 12 controls passed without it.
- The four 4x MSAA occlusion counts on RADV GFX1201. Read-back matches. The query count does not. No standalone repro is in the Slot 2 folder.
- Graphics fill and compression rates at a pinned memory clock. The Slot 2 run's rates are uninterpreted: the memory anchor failed, three bursts ran past 250 ms, and 29 cells spread by more than 10% on unpinned clocks.
- The long thermal load, parked until the owner is present.
