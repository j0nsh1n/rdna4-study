# RDNA 4 global memory system (lane 2)

Date: 2026-11-17. Card: Radeon RX 9070 XT (gfx1201, 64 CUs in 32 WGPs, 8 shader arrays, 4 shader engines, wave32,
16 GB GDDR6 on a 256-bit bus), ROCm 10.1.0 (hipcc, clang 24), kernel 7.3.4.
Run folder: `runs/20261117T121101Z-memory/` (`scripts/study-memory.sh all`, commit eeeea18). Every number is
MEASURED in that run unless it is labelled READ (from a document) or INFERRED (reasoning); a few diagnostic points
come from development runs in `runs/dev-memory/` and say so. Two earlier full runs of the same script
(`20261117T115726Z-memory`, `20261117T120410Z-memory`) agree with it within the spreads given. After an independent
review (Sonnet 5.5) the lead corrected several explanations in this file, rewrote one verdict and added time
limits to the script; reruns `20261117T122024Z-memory` (before the fixes) and `20261117T122956Z-memory` (after)
reproduce the numbers. The `mem_inflight_waits` verdict line in the main run's log predates the rewritten verdict; the
current one is in `20261117T122956Z-memory`.

Card configuration for all runs here: LACT profile "Testing" (stock core clock and voltage, the driver's default
317 W power cap, memory maximum 1259 MHz, performance level auto). The regulator-level power-limit mod (rdna4p,
1.2x) is still in the card, so the driver's power reading is scaled down on the main rail. The script logged
`pp_od_clk_voltage` and `power1_cap` before every job; they did not change. The memory clock read 1258 MHz
(`pp_dpm_mclk` top level) before and after every job, so the bus limit is 640 GB/s (READ: 16 channels x 16 bits x
20 Gbps; 644 GB/s if 1258 MHz is taken exactly). The memory is Samsung GDDR6 (driver's `mem_info_vram_vendor`). The
regulator mod is the owner's statement, not something measured here. The core clock under load read 3.2 to 3.4 GHz. The first-hour numbers quoted for comparison (read 634 GB/s,
copy 550 GB/s, latency 137 ns) were taken on the earlier overclocked profile. Cycles are shader-clock cycles from the
wave's own counter; nanoseconds come from the 100 MHz REALTIME counter.

Round 7 (`round7.md`): the memory clock is no longer held at 1258 MHz. Under any streaming load it now reads 1408 MHz (this run made no display change), and a read pass runs at 700.3 GB/s (write 638.7, copy 608.9), about 10 to 11 % above the 1258 MHz
figures in this file. MEASURED.

## In short

The memory system as one wave or one kernel sees it:

| Level | Size | Who shares it | Latency, one wave (cycles / ns) | Bandwidth, whole card |
|---|---|---|---|---|
| L0 vector cache | 32 KB | one CU (not the other CU of the WGP) | 81 / 24-29 | about 19 TB/s on data all waves read (whole card, L0 and request merging together); separately, one wave alone gets 128 bytes per cycle |
| "L1" 256 KB per shader array (driver table) | - | - | no latency level of its own | a small drop for data shared by all CUs between 288 and 512 KB, nothing else |
| L2 | 8 MB for the card; one wave alone sees L2 latency only up to about 4,000-11,000 lines of 256 bytes (1-2.75 MB if L2 lines are 256 bytes, which the stride sweeps support and the offset series does not; it follows the line count, not the span, section 2b-2) | all CUs; emptied at every kernel launch | 230-255 / 71-79 | 7.4-9.5 TB/s |
| Memory-side cache (Infinity Cache, MALL) | 64 MB | whole card, kept across launches | 360-390 / 105-118 in page order; 420-500 / 130-149 random | about 2.0 TB/s |
| GDDR6 | 16 GB | | 485-545 / 142-158; 700-820 / 200-246 random after a long walk | 633 GB/s read (98 to 99 % of the 640-644 GB/s bus limit), 581 write, 547 copy |

- Reading plain buffers reaches 633 GB/s whatever the data: zeros, a repeated byte, a ramp and random data all read
  at 633 GB/s, and at the same speed from the memory-side cache. The hardware's memory compression does not change
  plain-buffer bandwidth.
- What limits a streaming kernel is the bus and nothing else, as long as enough data is in flight: about 200 KB for
  the card (INFERRED from 640 GB/s x the loaded latency), which is 3 waves per SIMD with one 16-byte load each, or one
  wave per SIMD with 4 loads each. Waves per SIMD, register count (24 to 192), loads in flight beyond that, cache
  hints and the address form changed nothing measurable: everything lands at 619-632 GB/s. Four 1024-thread
  work-groups (4 WGPs, one per shader engine) already fill the bus; one gets 224 GB/s.
- Surprises: (1) L2 contents do not survive a kernel launch: data read by one launch is found in the memory-side
  cache, not in L2, by the next. (2) One wave alone sees L2 latency only up to about 4,000-11,000 lines (1-2.75 MB), not 8 MB; the limit follows the
  number of lines, not the span, the page count, power-of-two strides or the launch's L2 flush, and its cause is not
  known (section 2b-2). (3) A random walk
  through memory gets about 55 % slower after some 25-50 MB of other lines have passed (address translation, INFERRED). (4)
  Writing half of each 128-byte line costs 4.5 times more per byte than writing whole lines, but two half-line stores
  back to back are combined. (5) Nine or more waves waiting on memory on the neighbouring SIMD of a CU make a
  wave that only hits L0 two to four times slower; slow waves on its own SIMD do not. It happens with L2 hits as well
  as misses and without translation misses; it is neither a fixed count of waves nor of outstanding loads (section 7a).
- What a kernel author should do: load 16 bytes per lane from addresses aligned to 128 bytes; keep about 4 loads in
  flight per wave (more only matters for a single wave, which stops at 63); do not fear 96 or more registers in a
  memory-bound kernel; read streamed data with non-temporal loads (`__builtin_nontemporal_load`, `th:TH_LOAD_NT`): in
  the memory-side cache it neither stays nor pushes other data out (plain loads do both), while in L2 any stream
  larger than L2 pushes other data out whatever the hint, so reuse data soon after loading it;
  never write partial lines in scattered stores; for benchmarks use buffers above the 64 MB memory-side cache (memory speed shows from 96 MiB) and make each timed
  launch last at least 1 ms or time many launches back to back (a single short launch carries about 10 us of overhead); use `s_prefetch_data`, which works under ROCm, for latency-bound walks with known addresses.

## Method

Five probes, built and run by `scripts/study-memory.sh` (build, then every mode as its own GPU job, then the
deliberate breaks). Each timed loop was checked in the `-save-temps` assembly by `scripts/mem_loopcheck.py` (which
reuses lane 1's loop finder) against `scripts/mem-expect/<probe>.txt`: which load and store instructions, how many,
which cache-policy bits, the register count and no scratch. Whole-card wall-clock figures are medians of 5 to 20
launches timed with HIP events after warm-up launches; the range in brackets is slowest to fastest launch. Cycle
figures are medians of 3 to 5 launches of one wave (or one CU). Long tables are complete; the text below each gives
the reading. Data is random unless the item is about data content.

| Probe | Items | What it does |
|---|---|---|
| `src/study/mem_stream.hip` | 1, 2a, 3, 4, 8 | whole-card reads: calibration, data content, size sweeps (five ways), the cache trap, load width and shape, load hints |
| `src/study/mem_latency.hip` | 2b, 8, 9 | one-wave pointer chase against size, which level is shared by whom, persistence across launches, hints against L2 and the memory-side cache, software prefetch |
| `src/study/mem_inflight.hip` | 5, 7 | one wave with N loads before a wait; a fast wave next to slow waves (built with `-mcumode`, one work-group on one CU) |
| `src/study/mem_occupancy.hip` | 5, 6 | waves that fit on a SIMD against registers; streaming against waves per SIMD, loads in flight and registers |
| `src/study/mem_store.hip` | 10, 11 | stores with each policy, partial-line stores, copies, read-modify-write; bandwidth against work-group count and size |

Thermal and safety: whole-card jobs started only below 70 C hotspot and at least 20 s after the previous one; each
lasted under 20 s and the programs paused between configurations above 80 C (never triggered). Highest hotspot seen
in the cited run: 69 C (the two earlier full runs: 66 C and 70 C). Memory available was checked (at least 8 GB)
before each job; device buffers stayed under 5 GB.

## 1. Calibration and data content

| Read, 1 GiB, first-hour kernel and launch (`read_ref`) | GB/s |
|---|---|
| zero-filled (first hour: 634, overclocked profile) | 633 (630-634) |
| byte 0x55 | 633 (630-634) |
| ramp (word i = i) | 633 (632-634) |
| random | 633 (632-634) |
| 32 MiB, repeated launches (memory-side cache), zero / byte / ramp / random | 1067 (683-1081) / 1069 (1013-1085) / 1071 (1019-1080) / 1069 (1007-1084) |

The known answer reproduces (within 0.3 % of 634) on the stock profile. Data content changes nothing at either size.
INFERRED: the memory compression AMD describes (READ, research-5: "for a few workloads") is not applied to plain
compute buffers, or does not change bandwidth when it is.

## 2. The cache hierarchy

### 2a. Bandwidth against buffer size (whole card)

Five views, because the way waves share a buffer decides what the caches see:

- pass1: one 16-byte load per thread, the grid covers the buffer once per launch, launches back to back (the
  2.5 us empty-launch time is subtracted). Every launch sweeps the buffer in order.
- win_shared / win_private: 2048 waves (16 per SIMD), each reads a window in 4 KiB steps with the fewest possible
  instructions; shared = every wave reads the same window, private = each wave its own (footprint 2048 x window).
- private: 1024 waves each read their own region repeatedly (footprint 1024 x region).
- shared: 128 work-groups each read the whole buffer repeatedly (each CU sees the whole buffer).
- split: the whole grid reads one buffer repeatedly, each thread the same addresses every pass. Waves race ahead of
  each other (lane 1: the oldest wave keeps the SIMD), so this view overstates capacities; it is listed for that
  reason only.

| View | Size per reader | Footprint | GB/s |
|---|---|---|---|
| pass1 | 1 MiB | 1 MiB | 105 (102-105) (discard; INFERRED: first job on a cold clock, 2.0 GHz at start; the same in all five runs) |
| pass1 | 2 MiB | 2 MiB | 2081 (2017-2086) (launch-bound, overstated: see note) |
| pass1 | 4 MiB | 4 MiB | 1714 (1630-1770) (launch-bound) |
| pass1 | 6 MiB | 6 MiB | 1924 (1866-1973) |
| pass1 | 7 MiB | 7 MiB | 1918 (1890-1968) |
| pass1 | 8 MiB | 8 MiB | 1942 (1932-1952) |
| pass1 | 9 MiB | 9 MiB | 1998 (1951-2012) |
| pass1 | 10 MiB | 10 MiB | 1882 (1871-1911) |
| pass1 | 12 MiB | 12 MiB | 1936 (1912-1948) |
| pass1 | 14 MiB | 14 MiB | 1904 (1882-1931) |
| pass1 | 16 MiB | 16 MiB | 1932 (1925-1944) |
| pass1 | 20 MiB | 20 MiB | 1955 (1940-1958) |
| pass1 | 24 MiB | 24 MiB | 1952 (1941-1966) |
| pass1 | 32 MiB | 32 MiB | 1972 (1962-1973) |
| pass1 | 48 MiB | 48 MiB | 1997 (1988-2002) |
| pass1 | 56 MiB | 56 MiB | 1978 (1970-1980) |
| pass1 | 64 MiB | 64 MiB | 1984 (1981-1984) |
| pass1 | 72 MiB | 72 MiB | 710 (709-711) |
| pass1 | 80 MiB | 80 MiB | 664 (663-666) |
| pass1 | 96 MiB | 96 MiB | 636 (636-636) |
| pass1 | 112 MiB | 112 MiB | 636 (636-636) |
| pass1 | 128 MiB | 128 MiB | 636 (636-636) |
| pass1 | 160 MiB | 160 MiB | 637 (636-637) |
| pass1 | 192 MiB | 192 MiB | 637 (636-637) |
| pass1 | 256 MiB | 256 MiB | 637 (637-637) |
| pass1 | 512 MiB | 512 MiB | 634 (634-635) |
| pass1 | 1 GiB | 1 GiB | 612 (601-612) |
| pass1 | 4 GiB | 4 GiB | 632 (626-638) |
| win_shared | 4 KiB | 4 KiB | 19072 (18831-19381) |
| win_shared | 8 KiB | 8 KiB | 19235 (17642-19284) |
| win_shared | 16 KiB | 16 KiB | 19249 (19146-19312) |
| win_shared | 24 KiB | 24 KiB | 19045 (17783-19099) |
| win_shared | 32 KiB | 32 KiB | 19249 (19153-19326) |
| win_shared | 48 KiB | 48 KiB | 19180 (18950-19388) |
| win_shared | 64 KiB | 64 KiB | 19119 (18844-19236) |
| win_shared | 128 KiB | 128 KiB | 19201 (18897-19277) |
| win_shared | 192 KiB | 192 KiB | 19051 (18837-19167) |
| win_shared | 256 KiB | 256 KiB | 19133 (18957-19305) |
| win_shared | 320 KiB | 320 KiB | 19236 (19208-19339) |
| win_shared | 384 KiB | 384 KiB | 19078 (18977-19326) |
| win_shared | 512 KiB | 512 KiB | 19249 (19126-19360) |
| win_shared | 1 MiB | 1 MiB | 19078 (18957-19167) |
| win_shared | 2 MiB | 2 MiB | 19943 (19665-20708) |
| win_shared | 4 MiB | 4 MiB | 11522 (10899-11747) |
| win_shared | 8 MiB | 8 MiB | 12185 (11689-12669) |
| win_private | 4 KiB | 8 MiB | 9532 (9151-9548) |
| win_private | 8 KiB | 16 MiB | 2377 (2348-2439) |
| win_private | 16 KiB | 32 MiB | 2110 (2109-2114) |
| win_private | 32 KiB | 64 MiB | 1980 (1978-1984) |
| win_private | 64 KiB | 128 MiB | 667 (666-668) |
| win_private | 128 KiB | 256 MiB | 627 (627-629) |
| win_private | 256 KiB | 512 MiB | 627 (626-628) |
| win_private | 512 KiB | 1 GiB | 628 (628-629) |
| private | 1 KiB | 1 MiB | 12692 (12641-12740) |
| private | 2 KiB | 2 MiB | 12689 (12561-13752) |
| private | 3 KiB | 3 MiB | 9349 (9342-10010) |
| private | 4 KiB | 4 MiB | 7433 (7426-7503) |
| private | 6 KiB | 6 MiB | 7724 (7571-8202) |
| private | 8 KiB | 8 MiB | 7799 (7745-7871) |
| private | 10 KiB | 10 MiB | 3250 (3234-3358) |
| private | 12 KiB | 12 MiB | 1981 (1956-1992) |
| private | 16 KiB | 16 MiB | 1959 (1945-1975) |
| private | 24 KiB | 24 MiB | 1977 (1951-1978) |
| private | 32 KiB | 32 MiB | 1969 (1944-1970) |
| private | 64 KiB | 64 MiB | 1949 (1556-1962) |
| private | 128 KiB | 128 MiB | 636 (597-637) |
| private | 256 KiB | 256 MiB | 636 (594-637) |
| private | 1 MiB | 1 GiB | 635 (593-636) |
| shared | 4 KiB | 4 KiB | 12874 (12865-13345) |
| shared | 8 KiB | 8 KiB | 12865 (12402-12893) |
| shared | 16 KiB | 16 KiB | 13285 (13272-14988) |
| shared | 24 KiB | 24 KiB | 13489 (13240-13500) |
| shared | 28 KiB | 28 KiB | 13981 (13773-15196) |
| shared | 32 KiB | 32 KiB | 14309 (14294-14328) |
| shared | 36 KiB | 36 KiB | 14562 (14279-16648) |
| shared | 40 KiB | 40 KiB | 14727 (14485-14796) |
| shared | 48 KiB | 48 KiB | 14678 (14426-15003) |
| shared | 64 KiB | 64 KiB | 14482 (14409-14851) |
| shared | 96 KiB | 96 KiB | 13081 (12967-13379) |
| shared | 128 KiB | 128 KiB | 13276 (12993-14963) |
| shared | 192 KiB | 192 KiB | 11828 (11663-11969) |
| shared | 224 KiB | 224 KiB | 12575 (12383-14074) |
| shared | 256 KiB | 256 KiB | 12686 (12416-13826) |
| shared | 288 KiB | 288 KiB | 12319 (12222-12411) |
| shared | 320 KiB | 320 KiB | 9377 (9241-10453) |
| shared | 352 KiB | 352 KiB | 12657 (12345-12699) |
| shared | 384 KiB | 384 KiB | 9008 (8848-9551) |
| shared | 448 KiB | 448 KiB | 12382 (11630-13554) |
| shared | 512 KiB | 512 KiB | 9276 (9120-9434) |
| shared | 768 KiB | 768 KiB | 9167 (8904-9921) |
| shared | 1 MiB | 1 MiB | 8676 (8641-8801) |
| shared | 2 MiB | 2 MiB | 8730 (8635-9430) |
| shared | 4 MiB | 4 MiB | 8735 (8527-8943) |
| shared | 8 MiB | 8 MiB | 9023 (8999-9505) |
| split | 256 KiB | 256 KiB | 8565 (8352-8710) |
| split | 512 KiB | 512 KiB | 13005 (12921-14676) |
| split | 1 MiB | 1 MiB | 16637 (14794-16872) |
| split | 1.5 MiB | 1.5 MiB | 14557 (13581-15060) |
| split | 2 MiB | 2 MiB | 13880 (13790-14259) |
| split | 2.5 MiB | 2.5 MiB | 15558 (15090-17773) |
| split | 3 MiB | 3 MiB | 15628 (14527-17487) |
| split | 4 MiB | 4 MiB | 13766 (13706-14139) |
| split | 6 MiB | 6 MiB | 13152 (12911-16034) |
| split | 7 MiB | 7 MiB | 13613 (12664-14357) |
| split | 8 MiB | 8 MiB | 13970 (13476-15326) |
| split | 9 MiB | 9 MiB | 12490 (12300-12983) |
| split | 10 MiB | 10 MiB | 12069 (11966-12137) |
| split | 12 MiB | 12 MiB | 10408 (9055-11111) |
| split | 16 MiB | 16 MiB | 8711 (7854-8734) |
| split | 18 MiB | 18 MiB | 7334 (6639-7529) |
| split | 20 MiB | 20 MiB | 3509 (3462-3630) |
| split | 22 MiB | 22 MiB | 2868 (2580-2913) |
| split | 24 MiB | 24 MiB | 2337 (2323-2370) |
| split | 32 MiB | 32 MiB | 2017 (1966-2017) |
| split | 40 MiB | 40 MiB | 1992 (1988-1994) |
| split | 48 MiB | 48 MiB | 1977 (1974-1980) |
| split | 56 MiB | 56 MiB | 1964 (1964-1965) |
| split | 64 MiB | 64 MiB | 1952 (1950-1953) |
| split | 72 MiB | 72 MiB | 1932 (1928-1935) |
| split | 80 MiB | 80 MiB | 1898 (1896-1899) |
| split | 96 MiB | 96 MiB | 1856 (1846-1860) |
| split | 128 MiB | 128 MiB | 1754 (1754-1760) |
| split | 160 MiB | 160 MiB | 772 (756-806) |
| split | 192 MiB | 192 MiB | 648 (628-650) |
| split | 256 MiB | 256 MiB | 647 (617-648) |
| split | 384 MiB | 384 MiB | 637 (615-637) |
| split | 512 MiB | 512 MiB | 632 (616-633) |
| split | 1 GiB | 1 GiB | 630 (617-631) |
| split | 4 GiB | 4 GiB | 632 (632-633) |

Reading of the steps (MEASURED unless marked):
- L0: per-wave private data stays at the top rate (12.7 TB/s in this loop) up to 2 KiB per wave, 32 KiB per CU
  (16 waves per CU), and falls at 3 KiB per wave (48 KiB per CU). Shared windows show no step at 32 KB because all
  waves of a CU read the same lines.
- 256 KB level: data that every CU reads drops from about 12.7 to 9 TB/s between 288 and 512 KB, with an irregular
  pattern (320 and 384 KB slow, 352 and 448 KB fast). No such level appears in latency (2b). INFERRED: the 256 KB
  per shader array in the driver's table is the read-coalescing buffer research-5 describes (READ: "no longer a
  dedicated cache level"); it merges requests but does not keep data.
- L2: private data keeps 7.4-9.5 TB/s up to 8 MiB in total and falls by 10 to 12 MiB; data shared by every wave falls
  between 2 and 4 MiB (19.9 to 11.5 TB/s).
- Memory-side cache: about 2.0 TB/s up to 64 MiB in pass1, gone by 96 MiB (pass1: 710 GB/s at
  72 MiB, 636 at 96 MiB). The split view is an exception: it still shows 1.75-1.93 TB/s at 72-128 MiB and falls to 772
  GB/s only at 160 MiB (it overstates capacities, as said above).
- Note on pass1: it subtracts the 2.5 us empty-launch time from each launch. Below about 4 MiB the kernel is shorter
  than a launch, so the subtraction overstates the rate (2 MiB: 598 GB/s raw against 2081 net). The 32-64 MiB points
  and the in-kernel views support about 2.0 TB/s for the memory-side cache.
- pass1 never shows L2 speed: 6 to 64 MiB all read at about 1.9-2.0 TB/s (the 2 and 4 MiB rows are launch-bound). With section 2c, the reason is that every
  launch starts with an empty L2.

### 2b. Latency against buffer size (one wave)

One wave chases a pointer cycle, one node per 128 bytes, each load waited for before the next. Three orders: random
over the whole buffer; increasing; random inside 64 KiB blocks visited in order (the same lines with few page
changes). Above 64 MiB the random walk is also timed after a short warm-up ("fresh"); see the note below the table.

| Size | random cycles (ns) | random, fresh | increasing | random in 64 KiB blocks |
|---|---|---|---|---|
| 4 KiB | 81 (29) | - | 81 (24) | - |
| 8 KiB | 81 (29) | - | 81 (24) | - |
| 12 KiB | 81 (29) | - | 81 (24) | - |
| 16 KiB | 81 (29) | - | 81 (24) | - |
| 20 KiB | 81 (29) | - | 81 (24) | - |
| 24 KiB | 81 (29) | - | 81 (24) | - |
| 28 KiB | 81 (29) | - | 81 (24) | - |
| 30 KiB | 81 (29) | - | 81 (24) | - |
| 32 KiB | 81 (29) | - | 81 (24) | - |
| 34 KiB | 247 (88) | - | 240 (72) | - |
| 36 KiB | 242 (82) | - | 255 (76) | - |
| 40 KiB | 238 (77) | - | 246 (74) | - |
| 48 KiB | 231 (71) | - | 252 (76) | - |
| 56 KiB | 237 (73) | - | 234 (70) | - |
| 64 KiB | 246 (76) | - | 241 (72) | - |
| 96 KiB | 253 (78) | - | 237 (71) | - |
| 128 KiB | 249 (77) | - | 244 (73) | - |
| 160 KiB | 253 (78) | - | 237 (71) | - |
| 192 KiB | 249 (77) | - | 243 (73) | - |
| 224 KiB | 256 (79) | - | 237 (71) | - |
| 240 KiB | 252 (78) | - | 253 (76) | - |
| 256 KiB | 249 (77) | - | 252 (76) | - |
| 272 KiB | 237 (73) | - | 246 (74) | - |
| 288 KiB | 249 (77) | - | 252 (76) | - |
| 320 KiB | 234 (72) | - | 244 (73) | - |
| 384 KiB | 237 (74) | - | 251 (75) | - |
| 448 KiB | 252 (78) | - | 253 (76) | - |
| 512 KiB | 238 (74) | - | 249 (75) | - |
| 768 KiB | 298 (92) | - | 331 (99) | - |
| 1 MiB | 268 (83) | - | 252 (76) | 246 (72) |
| 2 MiB | 297 (92) | - | 358 (107) | 338 (99) |
| 3 MiB | 443 (137) | - | 375 (113) | 368 (107) |
| 4 MiB | 418 (129) | - | 362 (108) | 371 (108) |
| 5 MiB | 432 (134) | - | 365 (109) | 379 (110) |
| 6 MiB | 419 (130) | - | 390 (117) | 372 (108) |
| 7 MiB | 441 (137) | - | 377 (113) | 369 (108) |
| 7.5 MiB | 436 (135) | - | 394 (118) | 381 (111) |
| 8 MiB | 436 (135) | - | 369 (110) | 374 (109) |
| 8.5 MiB | 443 (137) | - | 369 (110) | 358 (104) |
| 9 MiB | 441 (137) | - | 386 (115) | 368 (107) |
| 10 MiB | 450 (139) | - | 369 (110) | 369 (108) |
| 12 MiB | 451 (140) | - | 373 (111) | 381 (111) |
| 16 MiB | 450 (139) | - | 363 (108) | 374 (109) |
| 24 MiB | 467 (144) | - | 365 (108) | 372 (108) |
| 32 MiB | 461 (142) | - | 378 (112) | 366 (107) |
| 40 MiB | 472 (144) | - | 358 (106) | 375 (109) |
| 48 MiB | 471 (143) | - | 362 (106) | 375 (109) |
| 56 MiB | 475 (144) | - | 388 (113) | 380 (111) |
| 60 MiB | 477 (143) | - | 369 (107) | 382 (111) |
| 64 MiB | 499 (149) | 469 (140) | 372 (108) | 386 (112) |
| 68 MiB | 561 (166) | 490 (145) | 537 (156) | 537 (156) |
| 72 MiB | 595 (175) | 495 (145) | 538 (157) | 530 (154) |
| 80 MiB | 625 (182) | 484 (141) | 529 (154) | 538 (157) |
| 96 MiB | 666 (194) | 491 (143) | 537 (156) | 524 (153) |
| 128 MiB | 698 (203) | 495 (144) | 523 (152) | 530 (154) |
| 192 MiB | 733 (214) | 489 (142) | 537 (156) | 518 (151) |
| 256 MiB | 744 (217) | 490 (143) | 534 (156) | 524 (153) |
| 512 MiB | 758 (222) | 488 (142) | 544 (158) | 512 (149) |
| 1 GiB | 771 (226) | 483 (142) | 531 (155) | 525 (154) |
| 4 GiB | 820 (246) | 535 (161) | - | - |

Final pointers wrong: 0 (every walk ended on the node the host predicted when 0).

- L0: 80.6 cycles (24 to 29 ns: the clock under this light load was 2.8 to 3.4 GHz) up to 32 KiB, in every order; 34
  KiB is already L2. So the L0 is 32 KB with 128-byte lines (the node spacing), as the driver says (READ: 32 KB,
  128-byte lines).
- L2 for one wave: 231-256 cycles (71-79 ns) from 34 KiB to 512 KiB in every run; the rise starts between 768 KiB
  and 3 MiB depending on the run (this run: 298-331 at 768 KiB, 297-358 at 2 MiB; run 1: 253 at 2 MiB), and from
  3 MiB on it is 358-450 cycles in all three orders. Because the in-page orders show the same step, it is not address
  translation. Whole-card private data
  keeps L2 speed to 8 MiB (2a). Unexplained: a reading as four 2 MB parts, one per shader engine (research-5 READ: "8 MB, in 4 shader
  engines"), does not fit section 2a, where every CU reading a shared 8 MiB window still gets 9-12 TB/s, nor 2c,
  where the first reader on a new shader engine found a 16 KiB buffer at L2 latency (n = 3).
- Memory-side cache: 358-394 cycles (105-118 ns) in page order from 3 to 64 MiB, 418-499 random.
- Memory: 512-544 cycles (149-158 ns) in page order; random 483-495 "fresh" (535 at 4 GiB), rising to 700-820
  after long random walks.
- The random-walk penalty: at 512 MiB, latency rises from 145 ns to 230 ns when the walk is longer (MEASURED in
  development, `runs/dev-memory/`: 4096 warm-up steps 150 ns; 65 Ki steps 148; 98 Ki 185; 131 Ki 196; 196 Ki and more
  227-231). The memory and fabric clocks stayed at 1258 and about 2026 MHz throughout (sampled every 50 ms). Page-order
  walks of the same lines do not rise. INFERRED: address-translation entries are pushed out of the caches by the
  25-50 MB of lines walked through, and every later page change pays a table walk from memory.
- Against lane 1: 512 MiB random, fresh: 142 ns here against 137 ns in lane 1 (core_counter, earlier profile;
  rerun today: 149 ns). The latency verdict uses the fresh figure.

### 2b-2. What the single-wave L2 limit follows (timing leftovers, `tim_l2`)

Written by a Sonnet helper from a sealed brief (`src/study/tim_l2.hip`), read, rebuilt and rerun by the lane owner;
run `runs/20261117T175731Z-timing/`, MEASURED. One wave, dependent chase as in 2b (one full cycle of warm-up inside the same
launch, then 8,192 timed steps, final pointer checked: 0 wrong), with the node spacing, the number of nodes and the
start address varied independently. Calibration: 16 KiB random chase 80.6 cycles (L0, lane 2: 81).

| Nodes | 128 B | 256 B | 512 B | 1024 B | 4096 B | 65536 B |
|---|---|---|---|---|---|---|
| 2,048 | 246 | 252 | 234 | 236 | 233 | 463 |
| 5,632 | 244 | 296 | 255 | 254 | 244 | 453 |
| 7,872 | 243 | 348 | 412 | 248 | 392 | 671 |
| 11,008 | 293 | 368 | 359 | 411 | 470 | 696 |
| 21,568 | 338 | 462 | 478 | 466 | 476 | 756 |
| 42,304 | 433 | 478 | 486 | 474 | 690 | 818 |
| 131,072 | 458 | 478 | 505 | 760 | 667 | - |

(median cycles per load, random order; the full tables, linear order and the odd spacings are in the run's
`tim_l2-chase.log`.)

- The step from about 240 to 460-480 cycles follows the number of lines the wave cycles through, not the bytes it
  spans: at spacings of 256 to 4,096 bytes the first point 1.3 times above the plateau lies between about 4,000 and
  11,000 nodes (this run: 4,032 at 256 and 512 bytes, 11,008 at 1,024, 7,872 at 4,096; the helper's three runs:
  7,900 to 11,000), while the span there ranges from 1 MiB to 31 MiB. At 4,096-byte spacing, 5,632 nodes on 5,632
  different pages (22 MiB) still run at 244 cycles, so the reach of address translation is not the limit. Spacings of
  384, 640, 1,152 and 3,072 bytes step at 4,000 to 11,000 nodes like the powers of two, so set conflicts are excluded;
  moving the start by 4 KiB, 64 KiB or 1 MiB + 128 B moves the step between 5,632 and 21,568 nodes in one run
  (5,632 to 30,208 over the three runs), which is the same factor of several that separate runs show without any
  offset, so no effect of the start alignment can be told apart from that wobble. The warm-up runs
  inside the same launch, so the per-launch L2 flush is excluded too.
- At 128-byte spacing two nodes share one of the L2's 256-byte lines (section 2b's stride test), and the step comes
  at about twice the node count (11,000 to 22,000 nodes). That is the usual case, not every case: over the three runs with this probe the 128-byte stride sweep stepped
  at 21,568, 11,008, 30,208, 5,632, 11,008 and 11,008 nodes and the 256-byte one at 4,032, 4,032 and 7,872 (four
  times), so five of the six 128-byte steps lie above every 256-byte step and one (5,632, run `180204Z`) lies
  below; the start-offset series at 128-byte spacing gives 11,008 to 30,208. So the 256-byte-line reading is
  supported, not proven. Counted in 256-byte L2 lines the step is at about 4,000 to
  11,000 lines at every spacing, 1 to 2.75 MiB of L2, an eighth to a third of its 8 MB (INFERRED from the two
  observations above). Why one wave cannot use more of L2 is not established: no hit counters or physical addresses
  are reachable here, and between 4,000 and 16,000 nodes the result wobbles between processes by up to 100 cycles
  (INFERRED: where the buffer lands in memory), so the step is known only to within a factor of about five (5,632 to 30,208 nodes at 128-byte spacing).
- Address translation is a separate, second effect: at 65,536-byte spacing (one node per page) there is no 240-cycle
  plateau at all (455 cycles from 2,048 nodes), and at 4,096 bytes a second rise starts between 30,000 and 42,000 nodes
  (467 cycles at 30,208 nodes, 690 at 42,304: 120 to 165 MiB), in line with the random-walk penalty above.
- What would have contradicted "lines, not span": steps at equal spans for different spacings, or a step at the same
  page count for every spacing (fewer than 700 pages at 128 bytes, 7,900 at 4,096).

### 2c. Which level is shared by whom, and does L2 survive a launch

128 one-wave work-groups take turns, each chasing the same buffer once and then again; the first pass of each reader
is grouped by the nearest earlier reader. Cycles per load, median (range).

| Buffer | Nearest earlier reader | Readers | First pass | Second pass |
|---|---|---|---|---|
| 16 KiB | same cu | 64 | 101.3 (99.1-103.0) | 80.8 (80.8-80.8) |
| 16 KiB | same wgp | 32 | 241.6 (228.1-255.6) | 100.9 (99.2-102.7) |
| 16 KiB | same sa | 24 | 244.6 (225.1-256.6) | 101.3 (98.9-102.8) |
| 16 KiB | same se | 4 | 239.1 (235.1-239.1) | 100.5 (100.0-100.5) |
| 16 KiB | other se | 3 | 234.7 (226.7-235.7) | 100.0 (98.9-100.0) |
| 128 KiB | same cu | 64 | 242.9 (224.9-331.1) | 242.6 (224.6-358.8) |
| 128 KiB | same wgp | 32 | 242.9 (227.9-351.5) | 242.6 (227.6-365.3) |
| 128 KiB | same sa | 24 | 244.4 (230.7-349.9) | 244.1 (230.6-283.8) |
| 128 KiB | same se | 4 | 234.4 (224.7-238.4) | 234.1 (224.6-238.1) |
| 128 KiB | other se | 3 | 253.2 (224.7-255.2) | 253.1 (224.6-255.1) |
| 1024 KiB | same cu | 64 | 346.7 (230.7-382.8) | 246.3 (224.6-367.4) |
| 1024 KiB | same wgp | 32 | 351.4 (233.7-371.2) | 245.6 (230.3-313.4) |
| 1024 KiB | same sa | 24 | 353.3 (241.2-371.5) | 247.1 (227.6-309.4) |
| 1024 KiB | same se | 4 | 348.8 (235.2-355.2) | 235.1 (227.7-275.1) |
| 1024 KiB | other se | 3 | 353.5 (230.7-370.9) | 235.3 (230.6-252.1) |

16 KiB: readers on 64 CUs, 8 shader arrays, 4 shader engines, timeouts 0, wrong pointers 0, first reader 499.7 cycles;
128 KiB: readers on 64 CUs, 8 shader arrays, 4 shader engines, timeouts 0, wrong pointers 0, first reader 469.3
cycles; 1024 KiB: readers on 64 CUs, 8 shader arrays, 4 shader engines, timeouts 0, wrong pointers 0, first reader
506.0 cycles.

- 16 KiB: a reader on a CU that an earlier reader used finds the data in L0 (101 cycles first pass); a reader on the
  other CU of the same WGP, in the same shader array or anywhere else finds it in L2 (235-245). So the L0 belongs to
  one CU, not to the WGP, and nothing between L0 and L2 holds data for a shader array.
- 128 KiB (too big for L0): 234-253 for everyone on both passes: no 256 KB level.
- 1 MiB: second passes 235-247 (L2); first passes 347-354 in every group. INFERRED: the first pass pays address
  translation on a CU that has not used these pages.

Persistence across launches (one wave, a different CU each launch in practice), cycles per load:

| Buffer | Flushed (memory) | Second pass, same launch | Next launch |
|---|---|---|---|
| 16 KiB | 471.7 | 80.8 | 353.9 |
| 1 MiB | 507.7 | 227.6 | 357.8 |
| 4 MiB | 714.8 | 403.7 | 446.5 |
| 32 MiB | 769.0 | 461.3 | 463.6 |

A 1 MiB buffer read at L2 latency (228 cycles) in the same launch costs 358 in the next launch, the memory-side
cache's latency; 16 KiB loses L0 and L2 the same way (81 to 354). So L2 is emptied (or invalidated) between kernel
launches; the memory-side cache keeps its contents. INFERRED: the
system-scope acquire at the start of every HIP launch (READ, research-10 section 4 and the guide's table 16: an
invalidate at system scope acts on the CU caches and L2). Consequence: a kernel never finds the previous kernel's
data in L2; consecutive kernels exchange data through the memory-side cache at about 2 TB/s.

## 3. The cache trap

The first-hour read kernel launched repeatedly over one buffer, as a naive benchmark does:

| Buffer | random data GB/s | zero data GB/s |
|---|---|---|
| 32 MiB | 1037 (970-1047) | 1070 (992-1077) |
| 48 MiB | 1301 (1291-1309) | 1333 (1289-1343) |
| 64 MiB | 1427 (1196-1444) | 1452 (1439-1464) |
| 80 MiB | 623 (609-655) | 636 (622-654) |
| 96 MiB | 600 (596-609) | 601 (577-611) |
| 112 MiB | 602 (582-605) | 600 (581-604) |
| 128 MiB | 594 (576-605) | 597 (573-605) |
| 160 MiB | 612 (608-615) | 612 (609-615) |
| 192 MiB | 618 (602-622) | 618 (607-620) |
| 256 MiB | 620 (615-622) | 620 (615-622) |
| 384 MiB | 626 (622-627) | 626 (620-627) |
| 512 MiB | 628 (623-630) | 629 (626-630) |
| 1024 MiB | 633 (630-634) | 633 (630-634) |
| 2048 MiB | 634 (632-635) | 634 (632-635) |

- Up to 64 MiB the benchmark measures the memory-side cache (1.0-1.46 TB/s, inflated up to 2.3x).
- From 80 MiB to 512 MiB it reads 1 to 6 % below the 2 GiB value. INFERRED (reviewer's arithmetic, checked by the
  lead): this is fixed per-launch overhead, not the cache. Each launch here is timed alone with events; converting
  the table to extra time over 634 GB/s gives a nearly constant 8 to 14 us at every size from 96 MiB to 512 MiB (and
  about 3 us at 1 GiB), and the pass1 view, which runs launches back to back, reads 636-637 GB/s at 96-256 MiB.
- Rule for this card: put the buffer above the 64 MB memory-side cache (memory speed shows from 96 MiB), and make
  each timed launch last at least 1 ms (about 640 MB read per launch) or time many launches back to back; a single
  short launch carries about 10 us of overhead. The RDNA 3 rule, 4 times the last-level cache (256 MB here), is
  enough for the cache; it reads 2.5 % low in this table only because one launch of 256 MiB lasts 0.4 ms. Data
  content does not matter (zeros and random data agree within 1 %).

## 4. Load width and shape (whole card, 1 GiB random, inline-assembly loads, 8 per wave before a wait)

| Test | Bytes per lane | Address | Lane stride | Row offset | Order | Useful GB/s | Counting 128-byte lines touched |
|---|---|---|---|---|---|---|---|
| width | 4 | saddr | 4 | 0 | sequential | 585 (582-591) | 585 |
| width | 8 | saddr | 8 | 0 | sequential | 609 (606-611) | 609 |
| width | 12 | saddr | 12 | 0 | sequential | 624 (624-626) | 624 |
| width | 16 | saddr | 16 | 0 | sequential | 630 (629-631) | 630 |
| addr | 4 | vaddr64 | 4 | 0 | sequential | 588 (586-589) | 588 |
| addr | 8 | vaddr64 | 8 | 0 | sequential | 619 (618-621) | 619 |
| addr | 12 | vaddr64 | 12 | 0 | sequential | 626 (623-628) | 626 |
| addr | 16 | vaddr64 | 16 | 0 | sequential | 630 (629-631) | 630 |
| align | 16 | saddr | 16 | 0 | sequential | 630 (628-631) | 630 |
| align | 16 | saddr | 16 | 4 | sequential | 550 (511-574) | 550 |
| align | 16 | saddr | 16 | 8 | sequential | 548 (522-601) | 548 |
| align | 16 | saddr | 16 | 16 | sequential | 602 (592-616) | 602 |
| align | 16 | saddr | 16 | 32 | sequential | 602 (595-615) | 602 |
| align | 16 | saddr | 16 | 64 | sequential | 601 (583-614) | 601 |
| align | 16 | saddr | 16 | 96 | sequential | 600 (594-615) | 600 |
| align | 4 | saddr | 4 | 2 | sequential | 434 (415-446) | 434 |
| align | 4 | saddr | 4 | 64 | sequential | 588 (561-598) | 588 |
| stride | 16 | saddr | 16 | 0 | sequential | 630 (628-631) | 630 |
| stride | 16 | saddr | 32 | 0 | sequential | 315 (315-316) | 631 |
| stride | 16 | saddr | 64 | 0 | sequential | 157 (157-158) | 629 |
| stride | 16 | saddr | 128 | 0 | sequential | 79 (79-79) | 631 |
| stride | 16 | saddr | 256 | 0 | sequential | 40 (39-40) | 316 |
| stride | 16 | saddr | 512 | 0 | sequential | 39 (38-39) | 310 |
| stride | 4 | saddr | 4 | 0 | sequential | 587 (585-590) | 587 |
| stride | 4 | saddr | 8 | 0 | sequential | 311 (309-312) | 621 |
| stride | 4 | saddr | 16 | 0 | sequential | 157 (157-157) | 628 |
| stride | 4 | saddr | 32 | 0 | sequential | 79 (79-79) | 630 |
| stride | 4 | saddr | 64 | 0 | sequential | 39 (39-39) | 629 |
| stride | 4 | saddr | 128 | 0 | sequential | 20 (20-20) | 630 |
| random_lane | 16 | saddr | 16 | 0 | random lanes | 42 (42-42) | 337 |
| random_lane | 4 | saddr | 4 | 0 | random lanes | 11 (11-11) | 338 |
| random_lane | 16 | vaddr64 | 16 | 0 | random lanes | 42 (42-42) | 338 |
| random_row | 16 | saddr | 16 | 0 | random rows | 415 (415-416) | 415 |
| random_row | 4 | saddr | 4 | 0 | random rows | 105 (105-105) | 105 |

- Width: 16 bytes per lane is the fastest (630 GB/s); 4 bytes reaches 585 (7 % less), 8 bytes 609, 12 bytes 624.
- Address form: no consistent difference between an SGPR base with a 32-bit VGPR offset and a 64-bit VGPR address
  (this run: the 64-bit form 0 to 2 % faster; run 1: up to 4 % slower).
- Alignment of each 512-byte row: 128-byte aligned 630; aligned to 16, 32, 64 or 96 but not 128: 600-602 (5 % less; the cause
  is not established: a shifted contiguous stream touches the same number of lines overall); misaligned by 4 or 8 bytes: 548-550 (13 % less); 4-byte loads off by
  2 bytes: 434 (31 % less).
- Lane stride: the bus moves whole lines. Counting 128 bytes per line touched, strides up to 128 bytes run at
  627-631 GB/s; at 256 and 512 bytes the same count gives 311-315, half. INFERRED: memory is read in 256-byte units
  (the driver's L2 line size, READ), so touching one 128-byte line in every 256 bytes wastes half the bus. Random
  lanes (each lane a random line): 42 GB/s of useful 16-byte data, 337 GB/s counting 128-byte lines. (Counting 256-byte
  units would give about 670, above the bus limit, so that count overstates: random lanes touch some lines twice.) Random 512-byte rows at 4 KiB-aligned places: 415 GB/s.
- Cycles per load from L0, one wave (from item 5): about 4 cycles per 16-byte load (512 bytes, so 128 bytes per
  cycle per CU) and 1.3 cycles per 4-byte load (98 bytes per cycle).

## 5. Loads in flight

One wave, N independent loads (4 or 16 bytes per lane) then one `s_wait_loadcnt 0`, repeated. Cycles per batch;
"bad" counts values in the last batch that differ from the data (a wait that returned early would show here).

| Data | Bytes per lane | N | Cycles per batch | Cycles per load | Bytes per cycle | bad |
|---|---|---|---|---|---|---|
| memory | 4 | 1 | 549.1 (517.4-549.5) | 549.08 | 0.2 | 0 |
| memory | 4 | 2 | 845.3 (814.1-846.1) | 422.64 | 0.3 | 0 |
| memory | 4 | 4 | 776.7 (767.1-798.5) | 194.18 | 0.7 | 0 |
| memory | 4 | 8 | 778.0 (777.4-778.5) | 97.25 | 1.3 | 0 |
| memory | 4 | 16 | 810.3 (795.1-830.5) | 50.65 | 2.5 | 0 |
| memory | 4 | 24 | 823.3 (816.8-823.8) | 34.31 | 3.7 | 0 |
| memory | 4 | 32 | 825.3 (822.3-826.2) | 25.79 | 5.0 | 0 |
| memory | 4 | 48 | 856.3 (846.0-857.4) | 17.84 | 7.2 | 0 |
| memory | 4 | 56 | 865.7 (863.9-866.1) | 15.46 | 8.3 | 0 |
| memory | 4 | 63 | 873.1 (869.5-882.2) | 13.86 | 9.2 | 0 |
| memory | 4 | 64 | 985.6 (973.1-1025.7) | 15.40 | 8.3 | 0 |
| memory | 4 | 72 | 1496.1 (1482.3-1496.9) | 20.78 | 6.2 | 0 |
| memory | 4 | 80 | 1520.5 (1519.3-1539.8) | 19.01 | 6.7 | 0 |
| memory | 4 | 96 | 1541.3 (1537.7-1558.5) | 16.05 | 8.0 | 0 |
| memory | 16 | 1 | 772.0 (771.5-800.8) | 771.98 | 0.7 | 0 |
| memory | 16 | 2 | 788.8 (788.2-819.2) | 394.42 | 1.3 | 0 |
| memory | 16 | 4 | 806.9 (799.6-807.4) | 201.73 | 2.5 | 0 |
| memory | 16 | 8 | 818.5 (818.2-825.9) | 102.31 | 5.0 | 0 |
| memory | 16 | 16 | 858.3 (851.1-858.6) | 53.64 | 9.5 | 0 |
| memory | 16 | 24 | 918.6 (912.8-923.0) | 38.27 | 13.4 | 0 |
| L2 | 4 | 1 | 337.3 (333.3-388.7) | 337.30 | 0.4 | 0 |
| L2 | 4 | 2 | 323.8 (319.2-325.2) | 161.89 | 0.8 | 0 |
| L2 | 4 | 4 | 316.3 (310.4-337.3) | 79.06 | 1.6 | 0 |
| L2 | 4 | 8 | 301.1 (292.5-334.4) | 37.63 | 3.4 | 0 |
| L2 | 4 | 16 | 312.0 (306.0-321.1) | 19.50 | 6.6 | 0 |
| L2 | 4 | 24 | 327.1 (321.1-332.1) | 13.63 | 9.4 | 0 |
| L2 | 4 | 32 | 339.2 (328.8-344.1) | 10.60 | 12.1 | 0 |
| L2 | 4 | 48 | 364.8 (355.8-370.8) | 7.60 | 16.8 | 0 |
| L2 | 4 | 56 | 377.8 (371.8-437.0) | 6.75 | 19.0 | 0 |
| L2 | 4 | 63 | 373.4 (372.3-378.3) | 5.93 | 21.6 | 0 |
| L2 | 4 | 64 | 549.1 (539.1-551.1) | 8.58 | 14.9 | 0 |
| L2 | 4 | 72 | 546.9 (534.9-580.3) | 7.60 | 16.9 | 0 |
| L2 | 4 | 80 | 553.9 (541.9-655.6) | 6.92 | 18.5 | 0 |
| L2 | 4 | 96 | 587.2 (569.6-603.6) | 6.12 | 20.9 | 0 |
| L2 | 16 | 1 | 313.7 (299.0-335.2) | 313.74 | 1.6 | 0 |
| L2 | 16 | 2 | 316.4 (306.5-318.1) | 158.22 | 3.2 | 0 |
| L2 | 16 | 4 | 328.2 (322.2-341.6) | 82.05 | 6.2 | 0 |
| L2 | 16 | 8 | 329.9 (320.4-335.9) | 41.23 | 12.4 | 0 |
| L2 | 16 | 16 | 372.0 (357.0-382.1) | 23.25 | 22.0 | 0 |
| L2 | 16 | 24 | 406.7 (401.0-407.0) | 16.94 | 30.2 | 0 |
| L0 | 4 | 1 | 129.1 (129.1-129.1) | 129.08 | 1.0 | 0 |
| L0 | 4 | 2 | 130.1 (130.1-130.1) | 65.04 | 2.0 | 0 |
| L0 | 4 | 4 | 132.1 (132.1-132.1) | 33.01 | 3.9 | 0 |
| L0 | 4 | 8 | 137.1 (137.1-137.1) | 17.13 | 7.5 | 0 |
| L0 | 4 | 16 | 147.1 (147.1-147.1) | 9.19 | 13.9 | 0 |
| L0 | 4 | 24 | 161.1 (161.1-161.1) | 6.71 | 19.1 | 0 |
| L0 | 4 | 32 | 167.1 (167.1-167.1) | 5.22 | 24.5 | 0 |
| L0 | 4 | 48 | 191.1 (191.1-191.1) | 3.98 | 32.2 | 0 |
| L0 | 4 | 56 | 201.1 (201.1-201.1) | 3.59 | 35.6 | 0 |
| L0 | 4 | 63 | 210.1 (210.1-210.1) | 3.33 | 38.4 | 0 |
| L0 | 4 | 64 | 207.1 (207.1-207.1) | 3.24 | 39.6 | 0 |
| L0 | 4 | 72 | 221.1 (221.1-221.1) | 3.07 | 41.7 | 0 |
| L0 | 4 | 80 | 231.1 (231.1-231.1) | 2.89 | 44.3 | 0 |
| L0 | 4 | 96 | 251.1 (251.1-251.1) | 2.62 | 48.9 | 0 |
| L0 | 16 | 1 | 135.1 (135.1-135.1) | 135.08 | 3.8 | 0 |
| L0 | 16 | 2 | 139.1 (139.1-139.1) | 69.54 | 7.4 | 0 |
| L0 | 16 | 4 | 147.1 (147.1-147.1) | 36.77 | 13.9 | 0 |
| L0 | 16 | 8 | 156.1 (156.1-156.1) | 19.51 | 26.2 | 0 |
| L0 | 16 | 16 | 188.1 (188.1-245.7) | 11.76 | 43.6 | 0 |
| L0 | 16 | 24 | 231.1 (231.1-231.1) | 9.63 | 53.2 | 0 |

- Memory: a batch costs about the latency (550-870 cycles) for N up to 63, so throughput grows with N: 4-byte loads
  reach 9.2 bytes per cycle at N = 63, 16-byte loads 13.4 bytes per cycle at N = 24 (the largest without spills).
- N = 64: the batch costs 13 % more than N = 63 at memory and 47 % more at L2; at 72 and above it costs two round
  trips (1496-1541 cycles). So a wave holds at most 63 loads in flight; the 64th waits for the first to return. No
  wrong values at any N: the counter does not wrap, the wave just stalls.
- L2: the same pattern at about 300-375 cycles per batch, up to 22 bytes per cycle (4-byte loads, N = 63) and 30
  (16-byte, N = 24).
- L0: 129-135 cycles for one load, then about 1.3 cycles per extra 4-byte load and 4 per extra 16-byte load.
- Many waves (whole card, 512 MiB stream, item 6 table): 1 load in flight per wave gives 256-274 GB/s with 1 wave
  per SIMD, 530-540 with 2, 616-619 with 3 and 624-628 with 4; 4 or 16 loads in flight reach 624-630 GB/s already at
  1 wave per SIMD. INFERRED: the bus is full with about 384 loads of 512 bytes in flight (200 KB), which matches
  640 GB/s x about 300 ns.

## 6. Occupancy

Waves that fit on a SIMD (4096 one-wave work-groups spinning 200 us, each counting itself in and out per SIMD; all
128 SIMDs reached the same maximum):

| Registers (descriptor) | Waves per SIMD, measured | Rule |
|---|---|---|
| 24 | 16 (all 128 SIMDs) | 16 |
| 25 | 16 (all 128 SIMDs) | 16 |
| 48 | 16 (all 128 SIMDs) | 16 |
| 49 | 16 (all 128 SIMDs) | 16 |
| 64 | 16 (all 128 SIMDs) | 16 |
| 72 | 16 (all 128 SIMDs) | 16 |
| 73 | 16 (all 128 SIMDs) | 16 |
| 96 | 16 (all 128 SIMDs) | 16 |
| 97 | 12 (all 128 SIMDs) | 12 |
| 120 | 12 (all 128 SIMDs) | 12 |
| 128 | 10 (all 128 SIMDs) | 10 |
| 144 | 10 (all 128 SIMDs) | 10 |
| 160 | 9 (all 128 SIMDs) | 9 |
| 168 | 9 (all 128 SIMDs) | 9 |
| 192 | 8 (all 128 SIMDs) | 8 |
| 193 | 7 (all 128 SIMDs) | 7 |
| 256 | 5 (all 128 SIMDs) | 5 |

The rule "waves per SIMD = min(16, floor(1536 / registers rounded up to 24)))" (READ, research-5 and LLVM) holds for
all 17 register counts, including the boundaries (96 gives 16, 97 gives 12; 192 gives 8, 193 gives 7; 256 gives 5).

Streaming bandwidth (512 MiB) against waves per SIMD (w), loads in flight per wave (N) and registers:

| Registers | N | w=1 | w=2 | w=3 | w=4 | w=6 | w=8 | w=10 | w=12 | w=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1 | 256 | 530 | 616 | 625 | 628 | 626 | 626 | 626 | 622 |
| 24 | 4 | 630 | 629 | 632 | 626 | 629 | 626 | 628 | 628 | 619 |
| 68 | 16 | 629 | 626 | 629 | 629 | 627 | 626 | 625 | 624 | 623 |
| 48 | 1 | 272 | 538 | 619 | 628 | 628 | 628 | 626 | 629 | 623 |
| 48 | 4 | 628 | 629 | 629 | 626 | 626 | 626 | 626 | 628 | 620 |
| 68 | 16 | 629 | 629 | 629 | 626 | 627 | 625 | 625 | 624 | 624 |
| 72 | 1 | 271 | 534 | 618 | 627 | 625 | 626 | 626 | 629 | 622 |
| 72 | 4 | 624 | 625 | 626 | 626 | 626 | 628 | 625 | 628 | 619 |
| 72 | 16 | 629 | 629 | 629 | 626 | 624 | 625 | 624 | 624 | 623 |
| 96 | 1 | 271 | 534 | 618 | 625 | 625 | 626 | 626 | 626 | 623 |
| 96 | 4 | 625 | 628 | 629 | 629 | 629 | 629 | 628 | 626 | 619 |
| 96 | 16 | 628 | 629 | 626 | 628 | 624 | 628 | 627 | 627 | 623 |
| 120 | 1 | 273 | 535 | 617 | 624 | 628 | 626 | 628 | 631 | - |
| 120 | 4 | 628 | 626 | 629 | 629 | 629 | 628 | 626 | 626 | - |
| 120 | 16 | 627 | 627 | 628 | 626 | 626 | 626 | 630 | 626 | - |
| 144 | 1 | 274 | 536 | 618 | 628 | 625 | 628 | 628 | - | - |
| 144 | 4 | 629 | 626 | 626 | 626 | 626 | 629 | 625 | - | - |
| 144 | 16 | 629 | 626 | 629 | 629 | 628 | 628 | 628 | - | - |
| 192 | 1 | 274 | 540 | 619 | 627 | 625 | 629 | - | - | - |
| 192 | 4 | 625 | 629 | 626 | 629 | 626 | 628 | - | - | - |
| 192 | 16 | 629 | 629 | 626 | 626 | 625 | 625 | - | - | - |

GB/s, medians of 5 launches. With 1 wave per SIMD the dispatcher used 102-128 of the 128 SIMDs; from 2 waves on, all
128 (per-SIMD counts in the log).

- For a bus-bound streaming read, register count costs nothing beyond the wave slots it takes: at equal waves per
  SIMD, 24 and 192 registers read at the same speed. (Rows labelled 24 and 48 registers with 16 loads in flight really
  use 68 registers, because 16 loads need that many.) Kernels limited by something other than the bus were not tested. The RDNA 3 observation that kernels above about 64 registers were slower is not reproduced for a
  plain streaming loop; INFERRED: it came from what the compiler did with the extra registers there, not from the
  register file.
- The only occupancy that matters for a streaming read is enough loads in flight (section 5). At 16 waves per SIMD
  the rate drops 1 % (619-624 against 624-632).

## 7. Do waves wait for each other?

One work-group on one CU. A fast wave does batches of 8 loads that hit L0 (16 KiB) or L2 (1 MiB); slow waves follow a
random pointer chain through 1 GiB (about 770 cycles per load). Fast wave's cycles per batch and the ratio to running
alone:

| Fast wave's data | Slow waves, same SIMD | Slow waves, other SIMD | Fast cycles per batch of 8 | Ratio to alone | Slow wave cycles per load |
|---|---|---|---|---|---|
| L0 (16 KiB) | 0 | 0 | 159.0 (159.0-159.0) | 1.000 | 0 |
| L0 (16 KiB) | 1 | 0 | 159.4 (159.3-160.9) | 1.002 | 780 |
| L0 (16 KiB) | 0 | 1 | 159.4 (159.3-161.3) | 1.002 | 785 |
| L0 (16 KiB) | 15 | 0 | 160.8 (160.6-162.5) | 1.011 | 785 |
| L0 (16 KiB) | 0 | 4 | 159.5 (159.5-161.2) | 1.003 | 784 |
| L0 (16 KiB) | 0 | 8 | 159.9 (159.8-161.3) | 1.006 | 802 |
| L0 (16 KiB) | 0 | 10 | 628.9 (620.3-647.5) | 3.955 | 786 |
| L0 (16 KiB) | 0 | 12 | 632.5 (618.9-651.2) | 3.977 | 790 |
| L0 (16 KiB) | 0 | 14 | 630.9 (625.1-631.6) | 3.967 | 788 |
| L0 (16 KiB) | 0 | 16 | 631.6 (618.4-656.7) | 3.972 | 789 |
| L0 (16 KiB) | 4 | 4 | 160.0 (159.9-161.8) | 1.006 | 782 |
| L0 (16 KiB) | 8 | 8 | 161.2 (161.0-162.8) | 1.013 | 777 |
| L0 (16 KiB) | 15 | 8 | 162.8 (162.4-164.3) | 1.024 | 783 |
| L0 (16 KiB) | 12 | 12 | 625.5 (619.7-653.7) | 3.933 | 781 |
| L0 (16 KiB) | 15 | 16 | 635.9 (623.5-658.4) | 3.999 | 794 |
| L2 (1 MiB) | 0 | 0 | 320.0 (308.4-363.2) | 1.000 | 0 |
| L2 (1 MiB) | 1 | 0 | 314.2 (308.1-322.6) | 0.982 | 763 |
| L2 (1 MiB) | 0 | 1 | 328.0 (308.1-393.0) | 1.025 | 770 |
| L2 (1 MiB) | 15 | 0 | 318.2 (315.1-366.0) | 0.994 | 765 |
| L2 (1 MiB) | 0 | 4 | 314.3 (308.6-314.6) | 0.982 | 764 |
| L2 (1 MiB) | 0 | 8 | 340.0 (314.8-364.4) | 1.063 | 763 |
| L2 (1 MiB) | 0 | 10 | 628.4 (623.5-646.5) | 1.964 | 776 |
| L2 (1 MiB) | 0 | 12 | 625.8 (619.7-646.9) | 1.956 | 771 |
| L2 (1 MiB) | 0 | 14 | 624.8 (623.2-645.7) | 1.953 | 771 |
| L2 (1 MiB) | 0 | 16 | 630.0 (614.3-649.8) | 1.969 | 777 |
| L2 (1 MiB) | 4 | 4 | 314.9 (308.9-344.6) | 0.984 | 769 |
| L2 (1 MiB) | 8 | 8 | 339.0 (309.7-366.1) | 1.059 | 771 |
| L2 (1 MiB) | 15 | 8 | 331.7 (311.3-332.6) | 1.037 | 770 |
| L2 (1 MiB) | 12 | 12 | 624.9 (618.7-642.6) | 1.953 | 769 |
| L2 (1 MiB) | 15 | 16 | 623.3 (618.9-657.1) | 1.948 | 768 |

- The public test reproduces: one slow wave on the same SIMD does not hold the fast wave back (ratio 1.00-1.01),
  as research-5 reports for RDNA 4 (READ: "the two waves don't care"). Up to 15 slow waves on its own SIMD and up to
  8 on the other SIMD also cost at most 2 % (L0) or 6 % (L2, noisier).
- New: 10 or more slow waves on the other SIMD of the CU make the fast wave 4.0 times slower on L0 data and 2.0 times
  on L2 data, wherever the other slow waves are; 8 do not. The fast wave's batch then costs about 625 cycles whatever
  its data, close to one memory round trip. INFERRED: a structure shared by the two SIMDs of a CU (a miss queue of
  roughly 9 to 10 entries for the neighbouring SIMD's requests, or the texture-address unit's arbitration) blocks
  hits behind misses once it is full; why the fast wave's own SIMD can hold 15 slow waves without effect is not
  explained. Another reading fits as well: the slow waves walk 1 GiB, so every slow load is also an
  address-translation miss, and a queue of translation misses would behave the same. Only 8 and 10 slow waves were
  tested (not 9), on one CU.

### 7a. What the slow-wave threshold follows (timing leftovers, `tim_slow`)

Written by a Sonnet helper from a sealed brief (`src/study/tim_slow.hip`), read, rebuilt and rerun by the lane
owner; run `runs/20261117T175731Z-timing/`, MEASURED. One work-group on one CU (CU mode). The fast wave does batches of 8 L0
loads (alone: 159.0 cycles, as above); slow waves only on the other SIMD, each with C independent pointer chains
(one load per chain, then a wait for all), each chain starting at its own node. Footprints: 1 GiB random (every load
also a translation miss), 16 MiB random (translations cached after the first touch, about half the loads miss L2),
1 MiB random (L2 hits, warmed inside the launch). Fast wave's slowdown against alone, median of 5:

| Slow waves x chains (outstanding loads) | 1 GiB | 16 MiB | 1 MiB |
|---|---|---|---|
| 2 to 5 x 1 | 1.00-1.01 | 1.01-1.01 | 1.01-1.02 |
| 8 x 1 | 1.01 | 1.04 | 1.33 |
| 9 x 1 | 2.76 | 1.89 | 1.77 |
| 12 x 1 | 2.79 | 2.13 | 1.98 |
| 4 x 2 / 2 x 4 (8) | 1.01 / 1.01 | 1.03 / 1.02 | 1.18 / 1.17 |
| 5 x 2 (10) / 3 x 4 (12) | 1.02 / 1.02 | 1.06 / 1.12 | 1.98 / 1.98 |
| 6 x 2 (12) | 1.03 | 1.37 | 1.98 |
| 4 x 4 (16) / 5 x 4 (20) / 6 x 4 (24) | 1.05 / 1.24 / 2.03 | 1.96 / 2.03 / 2.02 | 2.01 / 1.99 / 2.04 |

- The onset with one chain per wave is at 9 slow waves for all three footprints, so translation misses are not
  needed. 8 slow waves are harmless at 16 MiB and 1 GiB; at 1 MiB they already cost 1.2 to 1.3x (1.33 in the named
  run, with a range of 169 to 212 cycles). With several chains per wave the answer splits: where the slow loads hit L2 (1 MiB) the slowdown
  follows the number of outstanding loads (1.2-1.3x at 8, 1.8-2.0x from 9 to 10, whether 9 x 1, 5 x 2 or 3 x 4); where they miss (16 MiB,
  1 GiB) waves with several chains need many more outstanding loads (12 to 16 at 16 MiB, 20 to 24 at 1 GiB) while
  single-chain waves still trigger at 9. So the trigger is neither a fixed number of waves nor a fixed number of
  outstanding misses; something per wave counts too (a wave's only load in flight weighs more than a wave's extra
  independent ones), not isolated. Which shared structure it is (a queue in the texture path, or arbitration between
  the two SIMDs of the CU) this probe cannot tell.
- Not reproduced: section 7's 4x for 10 to 16 slow waves; here 12 single-chain waves on 1 GiB give 2.8x (all slow
  waves sharing one start node, as in `mem_inflight`: 2.9x). INFERRED candidate, not tested: the slow waves here
  check their stop flag after every load, in `mem_inflight` after four.

## 8. Cache hints

All hints reached through inline assembly (`th:` and `scope:` fields; the encodings were checked with `llvm-mc` and
in the `.lto.s`). `__builtin_nontemporal_load` compiles to `th:TH_LOAD_NT` and `__builtin_nontemporal_store` to
`th:TH_STORE_NT` (MEASURED in the assembly). Last use (`TH_LOAD_LU`) has no source-level builtin (READ, research-10);
it was reached only through inline assembly.

Streaming 1 GiB with each hint, whole card (GB/s):

| Hint | GB/s |
|---|---|
| rt | 630 (630-631) |
| nt | 632 (632-634) |
| ht | 630 (629-631) |
| lu | 632 (632-633) |
| nt_rt | 631 (629-633) |
| rt_nt | 630 (630-632) |
| nt_ht | 631 (631-633) |
| rt_se | 630 (629-631) |
| rt_dev | 630 (630-632) |
| rt_sys | 632 (629-632) |
| nt_sys | 632 (632-633) |

No hint changes streaming bandwidth (629-633).

Where a hinted read leaves data, and whether a hinted stream evicts other data. Memory-side cache view (across
launches, whole card; cycles per load of a later one-wave chase; about 400-480 = in the memory-side cache, 630-760 =
in memory):

References (cycles): 4 MiB: flushed 634.5, read twice normally 405.3; 32 MiB: flushed 751.5, read twice normally 461.9.

| H | Hint | Chase after a hinted read of H | Chase after a hinted 256 MiB stream |
|---|---|---|---|
| 4 MiB | rt | 405.1 | 650.9 |
| 4 MiB | nt | 637.5 | 418.6 |
| 4 MiB | ht | 460.7 | 718.7 |
| 4 MiB | lu | 704.9 | 460.8 |
| 4 MiB | nt_rt | 458.6 | 726.3 |
| 4 MiB | rt_nt | 694.1 | 427.5 |
| 4 MiB | nt_ht | 430.6 | 687.5 |
| 4 MiB | rt_se | 457.0 | 713.8 |
| 4 MiB | rt_dev | 437.0 | 722.3 |
| 4 MiB | rt_sys | 457.4 | 694.7 |
| 4 MiB | nt_sys | 695.0 | 440.9 |
| 32 MiB | rt | 483.9 | 749.6 |
| 32 MiB | nt | 740.0 | 454.2 |
| 32 MiB | ht | 485.6 | 765.6 |
| 32 MiB | lu | 737.7 | 450.7 |
| 32 MiB | nt_rt | 480.5 | 749.3 |
| 32 MiB | rt_nt | 766.3 | 451.0 |
| 32 MiB | nt_ht | 450.6 | 739.5 |
| 32 MiB | rt_se | 457.6 | 766.0 |
| 32 MiB | rt_dev | 463.3 | 747.4 |
| 32 MiB | rt_sys | 467.7 | 736.5 |
| 32 MiB | nt_sys | 739.6 | 450.7 |

L2 view (one launch, one wave, H = 1 MiB, stream 16 MiB; about 240 = in L2, about 370-480 = memory-side cache):

| Hint | Regular chase after a hinted chase of H | Chase of H after a hinted 16 MiB stream |
|---|---|---|
| rt | 251.9 | 348.3 |
| nt | 315.8 | 351.4 |
| ht | 250.2 | 371.9 |
| lu | 230.7 | 345.3 |
| nt_rt | 252.2 | 344.9 |
| rt_nt | 233.7 | 414.4 |
| nt_ht | 238.2 | 325.3 |
| rt_se | 269.9 | 377.1 |
| rt_dev | 255.2 | 375.0 |
| rt_sys | 348.6 | 345.2 |
| nt_sys | 528.8 | 354.5 |

- Memory-side cache: NT, LU, RT_NT and NT with system scope do not allocate (a hinted read leaves the data in memory
  only) and do not evict (other data survives a 256 MiB hinted stream). RT, HT, NT_RT, NT_HT and RT at any scope
  allocate and evict. So the "far" half of the two-part hints is the memory-side cache (READ, guide: "NT_RT:
  non-temporal near, regular far").
- L2: no hint kept H in L2 through a 16 MiB stream from the same wave: 325-414 cycles afterwards for every hint,
  against 228 for H left in L2 (persistence table; a different job, so a different shader clock: take the comparison
  as a level, L2 against memory-side cache, not as exact cycles). After a hinted chase of H, H is still in L2 for RT, HT, LU,
  NT_RT, RT_NT, NT_HT and SE or DEV scope (230-270); NT leaves less of it there (316); system scope keeps it out of
  L2 (349, the memory-side cache), and NT with system scope out of both caches (529). INFERRED: in L2, NT lowers how
  long a line stays but does not bypass L2, and a stream larger than one wave's L2 share pushes everything out.
  Development runs with H = 2 MiB gave different L2 answers from run to run because 2 MiB sits on the one-wave L2
  step (2b); only this run's 1 MiB figures are used.
- For a decode kernel: stream weights with NT loads; activations then stay in the memory-side cache. In L2 the
  weight stream pushes them out whatever the hint, so use each activation soon after loading it, or hold it in
  registers or LDS. The RDNA 3 lesson (README: non-temporal weights) holds for the memory-side cache.

## 9. Software prefetch

`MODE` read in a kernel under ROCm: 0x010000f0, so `SCALAR_PREFETCH_EN` (bit 24, READ: guide) is already set. It can
be cleared and set from a kernel with `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 24, 1)`. One wave, a latency-bound walk
(one 128-byte load per 512-byte step through flushed memory, each waited for), cycles per step:

| Loop | Steps ahead | Cycles per step | MODE bit 24 during the loop |
|---|---|---|---|
| vector plain | 0 | 694.0 (686.6-718.4) | 1 |
| vector sprefetch mode as is | 4 | 266.0 (263.0-290.3) | 1 |
| vector sprefetch mode bit set | 4 | 269.8 (269.0-272.0) | 1 |
| vector sprefetch mode bit clear | 4 | 767.5 (762.5-786.0) | 0 |
| vector loadprefetch | 4 | 375.8 (374.1-377.9) | 1 |
| vector sprefetch mode as is | 16 | 275.8 (266.7-297.2) | 1 |
| vector sprefetch mode bit set | 16 | 275.7 (269.5-296.5) | 1 |
| vector sprefetch mode bit clear | 16 | 763.5 (759.6-770.3) | 0 |
| vector loadprefetch | 16 | 376.6 (375.5-380.1) | 1 |
| scalar plain | 0 | 723.2 (710.0-738.4) | 1 |
| scalar sprefetch mode as is | 4 | 194.4 (193.4-195.3) | 1 |
| scalar sprefetch mode bit set | 4 | 195.1 (194.8-196.0) | 1 |
| scalar sprefetch mode bit clear | 4 | 730.6 (724.1-755.0) | 0 |
| scalar sprefetch mode as is | 16 | 70.0 (70.0-70.2) | 1 |
| scalar sprefetch mode bit set | 16 | 70.0 (69.7-70.1) | 1 |
| scalar sprefetch mode bit clear | 16 | 746.4 (726.6-756.4) | 0 |

- `s_prefetch_data` (`__builtin_amdgcn_s_prefetch_data`) works under ROCm: 4 steps ahead it cuts a vector load from
  694 to 266-276 cycles (about the L2 latency: the prefetch fills L2), and a scalar load from 723 to 194-195 cycles,
  and to 70 cycles at 16 steps ahead (INFERRED: scalar loads then hit the scalar cache). With the MODE bit cleared
  the instruction does nothing (764-768 and 731-746 cycles, the plain cost plus the instruction).
- A vector load issued as a prefetch helps less (376 cycles): the wave must still track it.

## 10. Stores and copies (whole card, 1 GiB)

| Test | GB/s (data moved) | Lines per second |
|---|---|---|
| store st_rt | 581 (538-587) | 4.54 G |
| store st_nt | 573 (568-577) | 4.48 G |
| store st_ht | 582 (576-586) | 4.54 G |
| store st_wb | 582 (541-586) | 4.54 G |
| store st_sys | 585 (543-589) | 4.57 G |
| store st_b32 | 530 (522-538) | 4.14 G |
| store st_half | 128 (127-128) | 2.00 G |
| store st_evenodd | 572 (567-577) | 4.47 G |
| store st_strided | 130 (129-133) | 8.15 G |
| copy cp_rt | 547 (545-550) | |
| copy cp_nt | 583 (581-584) | |
| copy cp_ntload | 586 (585-587) | |
| copy rmw | 544 (543-546) | |
| copy rd | 631 (596-631) | |

st_rt: plain 16-byte stores; st_nt, st_ht, st_wb: TH_STORE_NT, _HT, _WB; st_sys: system scope; st_b32: 4-byte stores,
full lines; st_half: every other 4-byte word; st_evenodd: even words then odd words, two stores back to back;
st_strided: one 16-byte store per line. cp_rt: copy; cp_nt: NT load and NT store; cp_ntload: NT load, plain store;
rmw: x + 1 in place; rd: read only. Copy and rmw count read plus written bytes.

- Full-line stores run at 581 GB/s (91 % of the bus) with any policy (RT, NT, HT, WB, system scope: 573-585);
  4-byte stores that fill whole lines reach 530.
- Write combining: two half-line stores issued back to back (even words, then odd words) run at 572 GB/s, as fast as
  full lines; so partial writes to one line are merged before memory.
- Partial lines that stay partial are expensive: writing every other word (half of each line) moves 128 GB/s of data,
  2.0 G lines per second against 4.5 for whole lines: 4.5 times the cost per byte. Writing 16 bytes of each line (one
  aligned 16-byte piece) reaches 8.2 G lines per second. INFERRED: memory writes in 32-byte units; a unit that is only
  partly written costs a read-modify-write, so scattered 4-byte holes in every unit are the worst case.
- Copy 547 GB/s (first hour, overclocked profile: 550); with non-temporal loads 583-586 (7 % more), and NT stores do
  not add more. In-place read-modify-write 544.

## 11. The whole card against a few work-groups (1 GiB or 256 MiB read, 4 loads in flight per lane)

| Sweep | Work-groups | Threads each | WGPs / shader arrays / shader engines used | GB/s |
|---|---|---|---|---|
| A | 1 | 1024 | 1 / 1 / 1 | 224 (218-227) |
| A | 2 | 1024 | 2 / 2 / 2 | 461 (451-464) |
| A | 4 | 1024 | 4 / 4 / 4 | 625 (624-626) |
| A | 8 | 1024 | 8 / 5 / 4 | 632 (630-636) |
| A | 12 | 1024 | 12 / 6 / 4 | 632 (630-635) |
| A | 16 | 1024 | 16 / 6 / 4 | 632 (623-636) |
| A | 24 | 1024 | 24 / 8 / 4 | 632 (628-635) |
| A | 32 | 1024 | 32 / 8 / 4 | 632 (630-635) |
| A | 48 | 1024 | 32 / 8 / 4 | 631 (627-635) |
| A | 64 | 1024 | 32 / 8 / 4 | 626 (622-626) |
| B | 1 | 256 | 1 / 1 / 1 | 61 (61-63) |
| B | 2 | 256 | 2 / 2 / 2 | 123 (122-124) |
| B | 4 | 256 | 4 / 4 / 4 | 243 (240-245) |
| B | 8 | 256 | 8 / 5 / 4 | 463 (462-476) |
| B | 16 | 256 | 16 / 8 / 4 | 623 (622-626) |
| B | 32 | 256 | 32 / 8 / 4 | 632 (630-635) |
| B | 64 | 256 | 32 / 8 / 4 | 632 (630-635) |
| B | 96 | 256 | 32 / 8 / 4 | 632 (628-635) |
| B | 128 | 256 | 32 / 8 / 4 | 632 (630-635) |
| B | 256 | 256 | 32 / 8 / 4 | 625 (623-626) |
| C | 2048 | 32 | 32 / 8 / 4 | 619 (617-620) |
| C | 1024 | 64 | 32 / 8 / 4 | 624 (622-625) |
| C | 512 | 128 | 32 / 8 / 4 | 625 (622-626) |
| C | 256 | 256 | 32 / 8 / 4 | 626 (624-626) |
| C | 128 | 512 | 32 / 8 / 4 | 625 (623-626) |
| C | 64 | 1024 | 32 / 8 / 4 | 626 (623-627) |

- One 1024-thread work-group (one WGP, 32 waves) reads 224 GB/s; two 461; four 625, the full bus. The dispatcher
  puts consecutive work-groups on different shader engines (2 groups: 2 shader engines; 4: 4).
- 256-thread groups: 61 GB/s each, linear to 8 groups (463), full at 16.
- At the whole card the work-group size (32 to 1024 threads) changes nothing (619-626 GB/s).
- The verdict first asked for 64 groups to give 3 times one group; one group already reads a third of the bus, so the
  check was changed to "rising from 1 to 4 groups, and the whole card within 10 % of 634 GB/s" (run 1 recorded the
  old FAIL: 2.82 times).

## Deliberate breaks

Five verdicts were broken on purpose; each must FAIL, and did (`scripts/study-memory.sh breaks`, output in the run
folder). The shape, hint and CU-scaling tables have no deliberate break, and the store check is wide (300 to 720
GB/s), so those rest on the assembly check and on the calibration alone:

- `mem_stream-calib-break.log` (this and the next three lines are quoted from run `20261117T121101Z-memory`; later
  runs give the same verdicts with slightly different numbers): `VERDICT mem_stream_calib: FAIL 1 GiB zero-filled read 145100.2 GB/s; known answer 634
  GB/s (first hour, overclocked profile) +-5%: no; at or below the 640 GB/s bus limit: no (deliberate break: loads
  deleted by the compiler)`
- `mem_latency-chase-break.log`: `VERDICT mem_latency_chase: FAIL 512 MiB random chase (4096-step warm-up) 3.1 ns per
  load; known answer 137 ns (lane 1, earlier profile) +-15%: no; 16 KiB faster than half of that: n/a; final pointers
  WRONG (deliberate break: independent loads)`
- `mem_inflight-waits-break.log` (verdict rewritten after review; run `20261117T122956Z-memory`): `VERDICT
  mem_inflight_waits: FAIL fast wave alone 159.0 cycles per batch of 8 L0 loads; with one slow wave on its SIMD x1.002
  (must be <= 1.20); worst with many slow waves x4.06 (deliberate break: claimed <= 1.20)`. The normal verdict now
  rests on two measured ratios (one slow wave on the same SIMD within 20 %, many slow waves at least 2x); the first
  version compared a number with itself and could not fail.
- `mem_occupancy-rule-break.log`: `VERDICT mem_occupancy_rule: FAIL 3 of 17 register counts give the waves per SIMD of
  min(16, 1024 / (regs rounded up to 24)) (deliberate break: wrong register file size)`
- `mem_store-copy-break.log`: `VERDICT mem_store_copy: FAIL cp_rt median 547.6 GB/s, expected 1100 GB/s within 10
  percent (990.0 to 1210.0)`

## Where this disagrees with research-5 (public sources)

- "L1 gone, a read/write-coalescing buffer": agrees; the driver's 256 KB per shader array shows no latency level.
- "8 MB L2": the card holds 8 MB of L2-speed data in total, but one wave alone sees L2 latency only up to about
  4,000-11,000 L2 lines (1-2.75 MB; it varied between runs and processes). It follows the line count (section 2b-2);
  the cause is not known.
- "An Infinity Cache hit costs more than 50 ns over an L2 hit": measured +30 to 35 ns in page order (about 75 against
  105-118 ns) and +55 to 70 ns in random order (130-149 ns). Consistent once address translation is counted; I believe
  the page-order figure is the cache's own cost because the random walk adds page changes.
- "128-byte lines": true for L0 (latency step at 32 KiB with 128-byte nodes); memory traffic moves in 256-byte units
  (stride test), matching the driver's 256-byte L2 line.
- "Waves don't care what the other is doing": true for one slow wave; false once nine or more slow waves sit on the
  neighbouring SIMD (sections 7 and 7a).
- The INFERRED occupancy formula is correct for all 17 register counts tried.
- Community bandwidth (memtest_vulkan read 585 GB/s, READ): HIP reads at 633 GB/s here.

## Not measured

Still open:

- The cause of the single-wave L2 limit and of the slow-wave threshold. Narrowed by the leftovers pass (sections
  2b-2 and 7a): the L2 limit follows the number of lines (excluded: byte span and translation reach, power-of-two set
  conflicts, start alignment, the per-launch flush); the threshold needs no translation misses and is neither a fixed
  number of waves nor of outstanding loads. Which structure sets either is not identified (no counters reachable).
  The slow-wave threshold is not reachable (leftovers C plan, MS-2b: no counters).
  Answered in part (leftovers C, MS-2a, `measurements/leftovers-c-a.md`): two limits show. A total near 45,000 lines:
  eight waves step at 5,632 nodes each and sixteen at 2,880 (45,056 and 46,080 lines) in all three runs. And a
  one-wave limit somewhere between about 4,000 and 11,000 lines that moves from run to run (one, two or four waves
  slow down well before that total). So one wave alone does not get all of L2 at L2 latency, and many waves together
  do. MEASURED. Open: whether the limit belongs to the wave, its CU or its shader array (the dispatcher spread the
  waves over 2 to 4 shader engines, so these were not separated; "per wave" is a reading); whether the 45,000-line
  total is L2's capacity (it lies above the 32,768 256-byte lines of the 8 MiB L2; how much of each node L2 keeps was
  not measured).
- Why memory latency in cycles differs between probes: the prefetch probe's 694 cycles was not rerun and its access
  shape was not re-read (INFERRED: one read per line at about 2.8 GHz).
  Answered in part (leftovers C, MS-3, `measurements/leftovers-c-a.md`): the access shape decides, not the clock alone:
  in one launch at one clock a chase with 128-byte nodes costs 463 cycles per step and one with 256-byte nodes 660; a
  single load on its own 256-byte line costs 667 to 724 cycles. With 128-byte nodes two nodes share a 256-byte L2 line,
  so half the steps find it already fetched (about 245 cycles) and half go to memory (about 700). Every shape with one
  read per L2 line costs 237 to 258 ns at one load in flight, at any clock; the in-flight probe's 772 cycles is that at
  about 3.08 GHz (INFERRED: that job's clock was not logged per segment). MEASURED. Compare memory latencies in
  nanoseconds and count L2 lines of 256 bytes.
- In the in-flight probe, 2 loads in flight to one L2 line being slower than 4: why.
  Answered in part (leftovers C, MS-4, `measurements/leftovers-c-a.md`): the 4-byte old-layout row with 1 load in
  flight is a mix: exactly 50 % of its batches are L2 hits, because consecutive batches read the two halves of one
  256-byte line. With each load on its own 256-byte line, 2 loads cost no more than 1 (680 against 724 cycles) and 4 no
  more than 2: lane 2's rule holds. The old layout's 2 loads (both in one L2 line) stay slower than 4 (738 against
  680) and about 55 cycles (20 ns) above the distinct-line case; why is not established (INFERRED: the second request
  to a line being filled waits for the fill). MEASURED.
- Translation-buffer sizes and page size: only their effect on random walks was seen.
  Answered in part (leftovers C, MS-5, `measurements/leftovers-c-a.md`): at spacings of 64 KiB to 1 MiB, 1,024 nodes run
  at L2 latency and 1,152 miss on every step (about 210 cycles more); at 32 KiB spacing 2,048 nodes still hit. The
  step is bracketed only between 1,024 and 1,152 nodes. INFERRED: about 64 MiB of translation reach, about 1,024
  entries of 64 KiB (sizes inferred; two points do not fit: 16 KiB spacing slows at 3,072 nodes, 2 MiB spacing already
  at 1,024). Open: those two points; where between 1,024 and 1,152 nodes the step lies; the second rise of 2b-2 above
  120 MiB.
- Last use through the compiler's `!amdgpu.last.use` metadata: needs LLVM IR; reached only through inline assembly.
  Low value (leftovers C plan, MS-7).
  Answered in part (leftovers C, MS-7, `measurements/leftovers-c-a.md`): LLVM IR with `!amdgpu.last.use` compiles to
  `global_load_b32 ... th:TH_LOAD_LU` (listing, CPU only). Open: its effect on the card through this path was not run
  (the instruction is the one MS-6 measured).
- Whether memory compression is enabled for these allocations at all: no control or counter for it is reachable from
  HIP; only its absence of effect was measured. Not reachable (leftovers C plan, MS-8).
- New from leftovers C: why one and two waves read faster at 15,424 nodes than at 11,008 (W = 1: 340, 352, 336 cycles
  against 398, 416, 413 in the three runs; MS-2a, `measurements/leftovers-c-a.md`).
- New from leftovers C: why the L0 burst's batch (4 distinct 512-byte chunks, each read twice) costs 113 cycles against
  the calibration's 137 for 8 distinct chunks (MS-9, `measurements/leftovers-c-a.md`).

Answered since (leftovers C, 2026-11-17):

- Answered in scheduling-and-clocks.md, item 5a: the 1409 MHz memory request is accepted into the overdrive table but
  not applied (the driver keeps the memory clock at 1258 MHz). Why, and whether setting it by hand works, stays open
  there (the owner will test it; leftovers C plan, MS-1 and SC-14).
  Round 7 (`round7.md`): the request now applies; loads of 0.05 s to 20 s read 1408 MHz (no 1258 MHz reading in 27 traced loads).
- Answered (leftovers C, MS-6, `measurements/leftovers-c-a.md`): hints through the buffer-load builtins' cache-policy
  operand: the operand maps one to one onto the instruction's hint and scope fields (32 of 32 policy values checked in
  the listing), and each hint has the same effect as through inline assembly within 2 %: after an RT or HT read the
  data is in the memory-side cache (435 and 467 cycles in the chase), after an NT or LU read it is not (764 and 754).
  The scope bits do not change where a read leaves data. MEASURED (effect, listing), READ (bit layout).
- Answered (leftovers C, MS-9, `measurements/leftovers-c-a.md`): per-level bandwidth of L0 on its own at whole-card
  level: every CU's L0 delivers 128 bytes per cycle with all 64 CUs busy, 25.8 TB/s at 3.15 GHz (4 waves per CU reach
  126.4 bytes per cycle, 8 waves 128.0, 16 waves 127.9; one wave per CU 36.2 and is latency-bound). So the 19 TB/s of
  2a was below the L0 limit, not above it. MEASURED. Half of those loads repeat an address; that they were not served
  for free is INFERRED.

## How to rerun

```
scripts/study-memory.sh all      # build, every job, every break; about 6 minutes with the thermal pauses
scripts/study-memory.sh build    # build and loop checks only
build/study/bin/mem_latency chase   # one mode by hand: set HIP_VISIBLE_DEVICES=0, CARD_ARCH=gfx1201,
                                    # DRM_DEVICE and HWMON from runs/latest/card.env, and take the GPU lock
```

Results go to `runs/<UTC time>-memory/`: one log per job (clocks before and after, the card configuration, RESULT and
VERDICT lines), `loopcheck-<probe>.txt`, `build-<probe>.log` and `summary.md`. Diagnostic switches of `mem_latency
chase`: `CHASE_ONLY_512M=1` (only the 512 MiB point), `CHASE_WARM=<steps>` (warm-up length).
