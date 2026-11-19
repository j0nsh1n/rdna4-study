# RDNA 4 research, part 5: memory system, wave scheduling and one-stream speed

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 deep research: memory system and
decode speed"); items marked CHECKED were re-read by the lead. READ = stated in the source, seen by the helper only.
INFERRED = reasoning. COMMUNITY = forum or issue-tracker report, one user's setup.

Main sources:
- [HC] AMD, "AMD RDNA 4 Radeon 9000 series GPU", Hot Chips 2025 slides (August 2025),
  <https://hc2025.hotchips.org/assets/program/conference/day1/8_amd_pomianowski_final.pdf> (copy in `sources/`).
- [CC1] Chips and Cheese, "AMD's RDNA4 Architecture Video", 2025-03-05,
  <https://chipsandcheese.com/p/amds-rdna4-architecture-video>
- [CC2] Chips and Cheese, "RDNA 4's Out-of-Order Memory Accesses", 2025-03-23,
  <https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses>
- [CC3] Chips and Cheese, "Dynamic Register Allocation on AMD's RDNA 4 GPU Architecture", 2025-04-05,
  <https://chipsandcheese.com/p/dynamic-register-allocation-on-amds>
- [CC4] Chips and Cheese, "AMD's RDNA4 GPU Architecture at Hot Chips 2025", 2025-09-13,
  <https://old.chipsandcheese.com/2025/09/13/amds-rdna4-gpu-architecture-at-hot-chips-2025/>

## What one work-group processor contains (CHECKED, AMD's "Compute Engine" slide [HC])

Per SIMD (there are two pairs per work-group processor in the diagram):

| Block | Size |
|---|---|
| Vector register file | 192 KB |
| Scalar register file | 8 KB |
| Arithmetic units | 32 FMA units plus 32 FMA/integer units |
| Transcendental units | 8 |
| Matrix accelerator | one per SIMD ("4x 64 unit matrix accelerator" on the AI slide) |

Shared in the work-group processor: shared memory (LDS) 128 KB, L0 data cache 32 KB (two shown), instruction cache
32 KB, scalar (constant) cache 16 KB, texture units, ray accelerators, load/store.

- INFERRED: 192 KB / (32 lanes x 4 bytes) = 1536 registers per SIMD, the "1536 physical VGPRs" in LLVM (part 1).
- The slide draws the matrix accelerator as its own block. A comment from the author of llama.cpp's RDNA 4 kernels
  says the same: on RDNA 3 the matrix instructions use "the same silicon of vector instructions", while RDNA 4
  "redesigns the tensor core and makes it more like CDNA" (CHECKED, llama.cpp PR 17077, 2025-11-08). AMD's
  calculator and LLVM still say matrix and vector instructions do not run at the same time on RDNA 4 (part 1). Whether
  a separate block means they can overlap is the first thing to measure.

## Caches and the memory path

| Level | RDNA 4 (Navi 48) | Source |
|---|---|---|
| L0, per compute unit | 32 KB | [HC], CHECKED |
| "Aggregate CU cache" | 2 MB in total | [HC], CHECKED. 64 CU x 32 KB = 2 MB, so this is the sum of the L0s (INFERRED) |
| L1 | gone; described as "a read/write-coalescing buffer and no longer a dedicated cache level" | [CC1], READ |
| L2 | 8 MB, in 4 shader engines; banks per slice doubled | [HC] CHECKED for size; [CC1] READ |
| Infinity Cache (L3, memory side) | 64 MB = 16 blocks of 4 MB, each with its own memory controller | [HC] CHECKED for size and the 16 units; [CC4] READ |
| DRAM | 16 channels of 16 bits = 256 bits, GDDR6 at 20 Gbps = 640 GB/s | [HC] diagram CHECKED; arithmetic INFERRED |

- CHECKED [HC]: the fabric between the compute side and the memory side carries "1KBytes/clock" at 1.5 to 2.5 GHz, so
  up to about 2.5 TB/s. INFERRED: the fabric is about 4 times wider than the DRAM bus; data that sits in the Infinity
  Cache is not bandwidth-limited, only the 64 MB size limits it.
- READ [CC4]: a hit in the Infinity Cache costs "more than 50 ns of extra latency over a L2 hit", well over 100 clock
  cycles. DRAM is slower again. So a streaming kernel needs many loads in flight per wave to hide the latency.
- READ [CC4]: the cache line is 128 bytes (stated in the memory compression discussion). The guide's prefetch
  instructions also work in 128-byte chunks (part 1). A community post claims 192 bytes; unconfirmed and probably
  wrong.
- CHECKED [HC]: memory compression and decompression is central and fully in hardware ("~25% fabric bandwidth
  reduction ... for a few workloads", "~15% performance improvement for some raster workloads"). INFERRED: it targets
  render targets, not plain buffers of weights.

## How waves wait for memory

- CHECKED [HC]: "RDNA3 data returns were strictly ordered with respect to when the requests were made"; RDNA 4
  "introduces additional out-of-order queues for memory requests".
- READ [CC2], with their own microbenchmark: on RDNA 3 a wave waiting on its loads also waited behind other waves'
  slow loads; on RDNA 4 "the two waves don't care what the other is doing". Inside one wave, loads tracked by the
  vector-memory counter still return in order.
- READ [CC1]: the counters are split (global memory, texture sampling, ray intersection; scalar loads and LDS), as in
  the guide. The number of loads a wave can have outstanding is not published.
- INFERRED: on RDNA 4, running more waves per SIMD should help a memory-bound kernel more than on RDNA 3.

## Registers and how many waves fit

- CHECKED [CC3]: "RDNA 4 desktop GPUs have a 192 KB register file per SIMD, so a GPU kernel can use all thread slots
  (achieve maximum occupancy) if it uses 96 or fewer vector registers." 16 wave slots per SIMD. Registers are
  allocated "in 24 entry blocks" in the normal (non-dynamic) mode.
- INFERRED formula: waves per SIMD = min(16, floor(1536 / registers rounded up to a multiple of 24)). 96 registers
  give 16 waves, 120 give 12, 144 give 10, 192 give 8.
- The same article says RDNA 3's high-end chips also have the 192 KB file. On the RX 7900 XT, kernels above about 64
  registers were measurably slower. By this formula that was not a loss of wave slots, so something else (compiler
  spills, load scheduling) caused it. Worth re-testing on RDNA 4 at 64, 96, 120 and 144 registers.
- READ [CC3]: dynamic register allocation is for wave32 compute shaders only and takes over a whole work-group
  processor; the author found no use of it in Linux or LLVM at the time.

## One-stream generation in llama.cpp (COMMUNITY numbers)

Llama 2 7B Q4_0, tokens per second generated (tg128):

| Card | ROCm / HIP | Vulkan |
|---|---|---|
| RX 9070 XT, 2025 build | 97 to 101 | 137 to 142 |
| RX 9070 XT, July 2026 build, ROCm 7.1.1 | 122 | 132 |
| RX 9070 (non-XT) | 114 to 115 | |
| RX 7900 XT | 112 to 116 | 123 |

Sources: <https://github.com/ggml-org/llama.cpp/discussions/15021>, <https://github.com/ggml-org/llama.cpp/discussions/10879>,
<https://github.com/ggml-org/llama.cpp/issues/20218>. Different people, builds and systems.

- INFERRED: 122 tokens/s x 3.56 GiB is about 467 GB/s, 73 % of 640 GB/s; Vulkan's 132 is about 505 GB/s, 79 %. On the
  7900 XT our FP8 kernel read at 754 of 800 GB/s (94 %). So there is room above what llama.cpp reaches today, on
  either backend.
- No maintainer explanation was found for HIP being slower than Vulkan at generation on this card. A maintainer's
  view in 2025: "there is zero reason to expect 9070(xt) to perform better than the xtx" for generation.
- COMMUNITY (<https://github.com/ggml-org/llama.cpp/discussions/21043>, 2026-03, Radeon AI PRO R9700, same chip):
  dense-model decode on Vulkan reached 506 to 517 GB/s, "79-83% BW utilization"; the author blames redundant wait
  instructions in the Vulkan compiler's output and block sizes that waste part of each cache line.

## Clocks and power (COMMUNITY; matters for every benchmark)

- Setting `power_dpm_force_performance_level` to `high` changed generation speed noticeably on two R9700 cards.
- With the stock power limit one user saw core clocks rarely above 2.4 GHz under HIP load, about 2.8 GHz with a
  raised limit. Another reports automatic boost to 3.4 GHz on compute.
- Memory clock: "MCLK won't boost to 1258 MHz on kernel 6.17 ... Works fine on 6.19.8." An Arch Linux thread
  (2026-01) reports the memory clock stuck one level below maximum when the performance level is forced manually.
  The top level shows as 1258 MHz in `pp_dpm_mclk`.
- A vLLM issue on the same chip: raising the memory clock from 1258 to 1375 MHz gave about 5 % more speed.
- PCIe power saving (ASPM) set to performance gave about 10 % more decode speed on Vulkan for one user.
- Rule for our runs: read and log the core and memory clock (`pp_dpm_sclk`, `pp_dpm_mclk`), the power profile and
  the performance level in every run, and confirm the memory clock is at the top level before trusting a
  bandwidth number.

## Measured bandwidth on RDNA 4 (COMMUNITY)

- `memtest_vulkan` on an RX 9070 XT (Linux 7.2.8, Mesa RADV, 2026-10): written 551 to 553 GB/s, checked (read) 585
  GB/s, which is 91 % of 640. Another user got 142 to 154 GB/s on a misconfigured system.
  (<https://discuss.cachyos.org/t/9070xt-poor-vkmark-memtest-vulkan-on-fresh-install/36582>)
- No STREAM-style or OpenCL bandwidth measurement was found. A read-only HIP measurement would be new.

## What this means for a decode kernel (INFERRED)

1. The ceiling is about 0.8 of the 7900 XT's. If the card reads at 94 % of its bus like the 7900 XT did, that is
   about 600 GB/s; community tools show 585.
2. Keep registers at or below 96 for all 16 wave slots; test whether the old 64-register limit still matters.
3. Use test weights well above 64 MB.
4. Have each wave keep several independent loads in flight; test non-temporal against regular loads again.
5. Lay data out so each lane's load covers whole 128-byte lines where possible.
6. Pin and log clocks.

## Not found

- The number of outstanding loads per wave or per compute unit.
- Per-level cache latency and bandwidth figures, apart from the two quoted.
- Why HIP generation is slower than Vulkan on this card.
