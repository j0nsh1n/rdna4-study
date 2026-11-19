# RDNA 4 research, part 3: hardware and architecture

Date: 2026-11-16. Reading only. Gathered by a Sonnet helper (child thread "RDNA 4 research: hardware and
architecture"); items marked CHECKED were re-read by the lead against the source page. READ = stated in the source.
INFERRED = arithmetic or reasoning. AMD's product pages, TechPowerUp, Phoronix and the Hot Chips slides could not be
opened by the helper, so most raw specs come from press and Wikipedia, not from AMD.

## The card against the RX 7900 XT

| | RX 9070 XT (Navi 48) | RX 7900 XT (Navi 31) | Ratio |
|---|---|---|---|
| Compute units | 64 (32 work-group processors) | 84 | 0.76 |
| Memory | 16 GB GDDR6, 256-bit, 20 Gbps | 20 GB GDDR6, 320-bit, 20 Gbps | 0.8 |
| Theoretical bandwidth | 256 / 8 x 20 = 640 GB/s | 320 / 8 x 20 = 800 GB/s | 0.8 |
| Infinity Cache (L3) | 64 MB | 80 MB | 0.8 |
| L2 | 8 MB (4 x 2 MB) | not looked up | |
| Die, process | 356.5 mm2, TSMC 4 nm, 53.9 B transistors | 300 mm2 N5 plus memory dies | |
| Board power | 304 W | 315 W | |

Sources: PC Games Hardware test of 2025-03-05; Wccftech's summary of AMD's architecture deck (2025-02-28);
Wikipedia "Radeon RX 9000 series" and "Radeon RX 7000 series". All READ, none from AMD directly. The RX 9070 has 56
compute units and the same memory. The RX 9060 XT (Navi 44) has 32 compute units, a 128-bit bus (320 GB/s) and a
32 MB Infinity Cache.

- INFERRED: decode is memory-bound, so the ceiling for one-stream decode on the 9070 XT is about 0.8 of the 7900
  XT's. The measured read ceiling on the 7900 XT was 754 GB/s of its 800; the same fraction of 640 would be about
  600 GB/s. To be measured.
- INFERRED: the cache trap is smaller. Benchmarks need weights well above 64 MB (the RDNA 3 rule of 4 times the cache
  gives 256 MB).
- 16 GB against 20 GB: the 8B FP8 model (8.1 GB) fits; nothing in the sources quantifies the limit further.

## Cache and memory changes (Chips and Cheese, READ)

- The L1 cache level of earlier generations "no longer exists" on RDNA 4; L2 banks per slice were doubled
  (<https://chipsandcheese.com/p/amds-rdna4-architecture-video>, 2025-03-05).
- The Infinity Cache is 16 blocks of 4 MB on the memory side, and a hit there costs "more than 50 ns of extra
  latency over a L2 hit" (Hot Chips 2025 coverage, 2025-09-13).
- Out-of-order memory access: the wait counter is split, and waves no longer wait for each other's slow loads
  ("On RDNA 4, the two waves don't care what the other is doing", 2025-03-23). This matches the split
  `S_WAIT_*` counters in the instruction guide (part 1).
- Registers: 192 KB register file per SIMD, 16 wave slots; dynamic register allocation for wave32 compute shaders
  (2025-04-05). Matches section 3.3.3 of the guide.

## Matrix throughput according to AMD

CHECKED, <https://gpuopen.com/learn/using_matrix_core_amd_rdna4/> (2025-07-11), theoretical FLOPS per clock per
compute unit:

| Type | RDNA 2 | RDNA 3 | RDNA 4 |
|---|---|---|---|
| FP16 | 256 | 512 | 1024 |
| BF16 | not listed | 512 | 1024 |
| I8 | 512 | 512 | 2048 |

- These ratios agree with the cycle counts in AMD's instruction calculator (part 1): FP16 WMMA 32 -> 16 cycles,
  8-bit 32 -> 8.
- READ (Chips and Cheese, same article as above): FP8 and BF8 run "at the same throughput as INT8", and 4:2
  structured sparsity doubles the rate again. AMD's deck: "2x-16b & 4x-8b/4b dense matrix rates".
- INFERRED: whole card, FP16 dense: 64 CU x 1024 x 2.95 GHz = about 193 TFLOPS; the 7900 XT at 84 CU x 512 x
  about 2.4 GHz = about 103 TFLOPS. So about 1.9x for FP16 and about 3.7x for 8-bit types, before memory limits.
  The clocks are press figures; treat the result as a rough bound.
- READ: the RDNA 4 operand layout is not backward compatible with RDNA 3 (no replicated halves).

## AMD's RDNA 4 WMMA guide (CHECKED, three parts, 2026-06-02)

<https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/> (and part-2, part-3).

- One layout for all data types: A and B are stored K-major with 8 consecutive elements per lane; D is M-major, so D
  is the transpose of A's layout.
- FP16 fills a 128-bit load per lane. FP8 and INT8 use only 64-bit loads, INT4 32-bit. AMD's fix: fuse two WMMA
  instructions into one "double-K" step so the main loop uses 128-bit loads, "effectively doubling memory
  throughput" for the 8-bit types. The article says the results match hipBLAS bit for bit and that the techniques
  are "used in Llama.cpp".
- Part 3 says RDNA 4 lacks "shared-memory transpose loading and in-register matrix transpose" and builds a transpose
  from a WMMA with an identity matrix. (The instruction guide's transposing load reads from global memory, not from
  the local data share, so the two statements do not conflict.)
- No timing numbers in any part.

## Independent measurements

- None found for matrix instruction throughput, FP8 or measured memory bandwidth on RDNA 4. This is open ground.
- llama.cpp community numbers, Llama 2 7B Q4_0 (READ by the helper; the Vulkan row for the 9070 XT, 5603.63 pp512 /
  142.00 tg128 with flash attention, is CHECKED in <https://github.com/ggml-org/llama.cpp/discussions/10879>):

| Backend | RX 9070 XT pp512 / tg128 | RX 7900 XT pp512 / tg128 |
|---|---|---|
| Vulkan (RADV) | 5201.65 / 136.56 | 2941.58 / 123.18 |
| ROCm (HIP) | 5055.19 / 101.27 | 3098.38 / 116.15 |

  The rows come from different people, builds and dates, so they are not controlled comparisons. They suggest
  prompt processing is about 1.6 to 1.8x faster on the 9070 XT and one-stream generation about the same or slower
  on HIP. A comment in the HIP thread says RDNA 4 WMMA support arrived with ROCm 7
  (<https://github.com/ggml-org/llama.cpp/discussions/15021>).
- The Vulkan log line in that thread reports "warp size: 64" and "matrix cores: KHR_coopmat" for the 9070 XT on RADV.

## Conflicts and gaps

- Process: N4P (PC Games Hardware, Wikipedia) against N4C (a TechPowerUp snippet, not opened).
- Bandwidth 640 to 645 GB/s, from 20.0 against 20.1 Gbps.
- AMD's "3.5x ML uplift per compute unit" is against RDNA 2 and does not equal the 4x in the table above.
- Not found: AMD's own spec pages, any per-instruction timing measurement, Stable Diffusion numbers.
