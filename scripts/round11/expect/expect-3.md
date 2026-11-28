# Expectations, round 11 item 3: graphics fixed blocks (pixel output, texture filtering, rasteriser, surface compression), written before the first GPU job

Labels: READ (PLAN-r11.md, DRAFT-r11-plan.md item 3, notes/research-8-architecture-and-family.md section 3, AMD Hot Chips 2025 slide as reported by Wccftech 2025-08-26 and
ServeTheHome 2025-08-25), MEASURED (memory-system.md: 633 GB/s compute write/read anchor, 687-703 GB/s after the memory-clock fix), INFERRED (RDNA 3 layout carried over, not read for RDNA 4).
Scope: timing, occlusion counts, read-backs and the compiler listing. Everything runs through Vulkan on RADV (the Mesa the README names), headless, one offscreen image, no window.
The card is overclocked with a raised power limit (core clock 2.6 to 3.5 GHz under load, memory 1258 or 1408 MHz depending on whether the second monitor pins it), so every
rate below is a ceiling at a stated clock, and the run folder's sampler record says which clock the cell really had.

## Claims (READ) and what the probe can say about each
- 128 ROPs and 256 texture units (Guru3D, TechPowerUp, AMD page); peak pixel rate 380.2 GPix/s and peak texture rate 760.3 GT/s at the 2970 MHz boost clock (my arithmetic agrees).
  One outlet lists 96 ROPs and others 730.3 GT/s and 190.1 GP/s: conflict. The probe can settle the ROP count: a 1 MiB RGBA8 fill sits in the L2 and is not memory-stopped, so
  (GPix/s) / (core clock in GHz) from the sampler is pixels per clock. 128 -> 128; 96 -> 96; if it comes out near 64 the export path or the shader, not the ROP count, is the limit (reported, not forced).
- Compression: "about 25 % less fabric bandwidth" and "about 15 % performance in some raster workloads" (Hot Chips slide as reported); transparent to software (STH). The probe's compressible-
  against-random contrast and, when the switch is verified, compression default against disabled are the only black-box tests of it.
- Kernel side (READ, research-9): `gmc_v12_0_get_dcc_alignment()` exists for GC 12.0.0/12.0.1, so DCC is a graphics-surface feature with alignment tied to the L2 slice count.
- RADV's DCC debug options cover GFX8-GFX11.7 (RADV's own help text, PLAN-r11 corrections); this card is GFX12, so they are NOT used. The off switch is `VK_EXT_image_compression_control`
  (`VK_IMAGE_COMPRESSION_DISABLED_EXT` per image), with `VkImageCompressionPropertiesEXT` read back as the "setting applied" check. No extension, or a read-back that does not change: reported as
  "no switch available" and only the compressible-against-random comparison is read.
- Rate by format (full rate for 32-bit, slower for 64 and 128-bit) and the rasteriser's one primitive per clock per shader engine: INFERRED from RDNA 3, not read for RDNA 4.
  4 shader engines is MEASURED (earlier probes).

## Bands (ceilings, not predictions)
- Pixel output: 128 ROPs x 2.97 GHz = 380.2 GPix/s (448 GPix/s at 3.5 GHz). Write demand at that rate: R8 380 GB/s, RGBA8 1521 GB/s, RGBA16F 3041 GB/s, RGBA32F 6083 GB/s, against a bus of
  640 GB/s (stock 20 Gbps) or 720.9 GB/s (1408 MHz memory). So at 512 MiB (GDDR6) an uncompressed fill is memory-stopped: RGBA8 at most 160 GPix/s (180 at 1408 MHz), RGBA16F 80 (90),
  RGBA32F 40 (45), D32 160 (180); R8 would be ROP-bound but does not fit a 16384-wide image at 512 MiB and is skipped. Reported rates above these, with data that cannot compress, are a probe fault.
  At 1 MiB (inside the 8 MB L2) the rate is the ROP/export limit; at 4 MiB mostly L2; at 32 MiB (inside the 64 MB memory-side cache) between. Wide formats export slower (INFERRED: 64-bit half, 128-bit quarter rate).
- Shapes that must hold if the probe works (INFERRED): rate flat with size for R8 and RG8 until 32 MiB; falling 3x or more from 1 to 512 MiB for RGBA16F and RGBA32F (class "memory-limited");
  blending never faster than no blending; MSAA 4 at equal pixel count not faster per pixel than 1 sample; depth-only at least as fast as RGBA8 at 1 MiB. A shape that fails is reported, not forced.
- Texture: 256 TMUs x 2.97 GHz = 760 GT/s for a bilinear 32-bit-texel sample (448 raw units: 3.5 GHz gives 896). Per format and filter (INFERRED): nearest and bilinear RGBA8 at the full rate; RGBA16F half;
  RGBA32F quarter; trilinear half of bilinear; 4x and 16x anisotropic at most a quarter and a sixteenth of bilinear when the footprint is long; texelFetch at or above the nearest rate.
  The 8192 x 8192 textures read one new row per wave and move along it: they can fall toward the memory rate for the wide formats (RGBA32F: 16 B per sample at 640 GB/s is 40 GSample/s). BC1 and BC7 carry
  fewer bytes per texel (0.5 and 1) so they should stay near the filter rate where RGBA32F falls.
- Rasteriser: 4 shader engines x 1 primitive per clock x 2.97 GHz = 11.9 GTri/s (INFERRED, RDNA 3 layout). Triangles of 0.125 and 0.5 pixel^2 cover one pixel centre each (the 77/256 offset makes the count exact);
  a 1-pixel triangle still shades a 2x2 quad (INFERRED), so the pixel rate of small triangles sits far under the fill rate. The culled variant stops at setup and must be at least as fast as the visible one
  at every size; for the smallest triangles visible and culled should be near equal if the rate is primitive-limited, and diverge from about 8 pixel^2 up as pixel work grows.
- Compression (the DCC question): if the card's compression works for render targets, a solid-colour RGBA8 fill at 512 MiB writes more than the bus carries (above 640 to 721 GB/s, up to the ROP
  ceiling of about 1.5 TB/s) while the per-pixel-random fill cannot (at or under the bus); with the switch verified, solid and random disabled should both sit at the bus. A solid read-back above the bus means the
  texture path reads the compressed surface. If all four cases are equal, as compute buffers were, that is the result: "no measurable effect on this probe".
  No prior band for any rate on this card was read: first-run result, read only after the controls pass.

## Controls (a failed control stops its cell; the first-control group stops the run)
1. Occlusion count of every fill cell: three passes (two of colour A, one of colour B) count exactly 3 x W x H x samples. A hardware-counted fact, exact.
2. Read-back of two 64 x 4 pixel regions (top-left and bottom-right) against the CPU's pixels of colour B: exact for UNORM formats, within one unit in the last place for the float formats; depth 0.375 exact;
   the random arm checks every pixel against the CPU hash (the shader's and the C hash share constants; `--cpu-checks` greps both). Blend and MSAA cells have the count only.
3. Raster: occlusion count equals triangles per pass x exact covered pixels x passes (coverage by an integer rule that cannot tie: 1, 1, 3, 10, 36, 136, 528 pixels for the seven sizes); culled pipeline counts 0.
4. Texture: a constant texture sampled by the cell's own filter returns the constant (the filter path is live); a 16-texel RGBA8 gradient returns the closed-form bilinear and nearest values at 64 points (within 2/255).
5. Memory anchor: the 512 MiB RGBA8 random write (compression disabled when the switch is verified, else default) is at or under 720.9 GB/s x 1.02. The measured compute anchor is 633 GB/s (703 after the memory-clock fix).
6. Bursts: calibrated to 100 ms, never above 250 ms (PLAN rule). Every burst of a cell counts, not only the timed repeats: the calibration bursts, the warm re-aim bursts and the repeats. A longer one is marked BURSTOVER, prints `CTRLFAIL BURSTOVER`, counts in `SUMMARY burst_over=N`, and fails the verdict and the runner's analysis. Core clock, power and temperatures come from `r11_sampler` beside the job.
7. Spread: timed repeats of one cell more than 10 % apart (min to max) are marked `SPREAD>10%` and counted in `SUMMARY spread_flagged=N`. The clocks are not pinned (auto boost), so this is a warning, not a control: with N > 0 the verdict reads `PASS WITH SPREAD WARNINGS (N cells >10% between repeats ...)` and the runner prints that it is not an unqualified clean pass. `--strict-spread` turns each such cell into a `CTRLFAIL SPREAD` and fails the run.

## Deliberate breaks (each must make its control FAIL; checked on the card, and on the CPU for the logic)
1. The last draw scissored to half the target: the occlusion count and the read-back both fail.
2. The last draw skipped: the read-back fails (and the count).
3. Compression: an ordinary image (never asked for the switch) is checked as if it had asked for "disabled": the "setting applied" check must flag it. If an ordinary image also reads back disabled the
   read-back cannot show the switch and the arm is reported unverified, not as a result.
4. A nearest sampler checked against the bilinear closed form: at least 16 of 64 points must fail.
5. The culled pipeline's count checked against the visible expectation: must fail (the count control can tell a drawing pipeline from a culling one).
6. (CPU) an uncompressed 512 MiB fill at 900 GB/s is flagged as over the bus.
CPU proof so far (no GPU): `r11_gfx_selftest` is green and goes red under each of its nine breaks (7 off-grid colour, 8 a 274 ms warm burst ignored, 9 MSAA 4 bytes counted once); hash, half-float, UNORM, triangle coverage and BC1/BC7 constants are checked against an independent Python run.

## Estimate
144 cells (43 pixel output, 21 raster, 72 texture, 8 compression groups that each write and read) = 152 results, about 2 to 3 minutes of card load at 5 repeats of 100 ms bursts; wall about 30 minutes
(hotspot below 60 C before every group; every cell and every burst series also pauses at 72 C until 66 C, `--cool-c`, up to 120 s each wait; the first full run, with a gate per group only, climbed 31 C to 96 C in 50 s on the zero-RPM fan curve and tripped the 95 C guard). Host memory under 2 GB (the 8192 x 8192 BC textures stage 32 and 64 MiB on the host); card memory up to about 2.4 GiB at the
RGBA32F 8192 texture, 1 GiB staging plus the image. Disk under 50 MB (listings, repeats CSV). The desktop shares the graphics queue: tell the owner before the window; bursts are at most 0.25 s.

## Findings from the first full run (Slot 1 Part 4) and what changed
- RGB10A2 read-back: all 512 sampled pixels differed at every size. The control colour (0.2, 0.4, 0.6, 0.8) is off the 10-bit grid (204.6, 409.2, 613.8; alpha 2.4 of 3), so the stored value depends on the hardware's export rounding; the 8-bit colours were exact grid points and never tested rounding. The control now uses exact grid points (205, 409, 614 of 1023; 2 of 3), the log prints the first mismatching pixel's bytes (`readback_first_bad`), and the selftest pins the A2B10G10R10 bit layout and the grid rule (break 7). If it still fails, the printed bytes say which channel differs. Not yet confirmed on the card.
- MSAA 4 occlusion count: the formula (passes x W x H x samples) is the Vulkan rule and stays. The card returned 3 x W x (H/4 + 30) on all four cells (4 MiB RGBA8 878592, 32 MiB RGBA8 6660096, 4 MiB RGBA16F 485376, 32 MiB RGBA16F 3330048), about 1/16 of the expectation. Cause not known (driver count or hardware behaviour with 4 samples and no depth buffer); the cells stay untimed rather than weakened, and `ms_count_diag` lines record the ratio. A read-only follow-up on the card is a separate decision.
- BURSTOVER (512 MiB blend, 252 and 274 ms): calibration ran on a cold target, so k came out about 2.5 times too long. Every cell now re-aims k on a warm burst (up to three tries, within 15% of the target).
- No `VK_EXT_image_compression_control` (this driver): the solid and random default-compression cells run and read back as normal; only the disabled arm (`comp_*_disabled`) and break 5 report n/a, with `COMPSWITCH NO SWITCH AVAILABLE` and a `SUMMARY compression disabled arm: n/a` line.
