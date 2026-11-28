# Expectations, round 11 item 4a: video engine through VA-API (written before the first GPU job)

Labels: READ, INFERRED. ffmpeg 8.1.3 with `h264_vaapi hevc_vaapi av1_vaapi vp9_vaapi`, `-hwaccel vaapi`, filters `psnr`, software encoders libx264 libx265
libsvtav1 present (READ, `run_r11_video.sh --cpu-checks`, encoder lists only; the device was not opened).

## Claims to test (READ in notes/research-8-architecture-and-family.md, section 6)
- Two media engines on Navi 48 (ServeTheHome, Hot Chips 2025-08-25). Test: does 2 parallel processes give about twice one process's rate, and 3 no more than 2?
- AV1 B-frames, lower encode latency; 25 % better H.264 low-latency quality, 11 % HEVC quality (relative to the previous generation, which this card cannot test:
  reported with settings, not as a hardware claim).
- No frames-per-second number for this card exists in the local notes and none was fetched here: **no prior band**. First-run result, read only after the controls pass.
  The measured numbers are the finding; the sampler's video clocks (`vclk0 dclk0 vclk1 dclk1`) and `power1_average` give the power cost.

## What the sampler shows at idle (READ, 1.2 s sysfs sample, gpu_metrics v1_3)
`vclk0 dclk0 vclk1 dclk1` read 25 MHz each at idle, `mm_act` 0; fan 0 RPM (fan stop). So "video activity" means above the card's own idle baseline (25), not above 0.
The video-clock fields are the table's; whether they rise under VCN load on this firmware is itself a result of the first run (if they stay at 25 and `mm_act` stays 0
during hardware jobs, the activity control cannot tell dGPU work from iGPU work and the cells are void).

## Controls
1. Decode: `-pix_fmt yuv420p` frame md5 of the hardware decode equals the CPU decoder's for every frame, all three codecs, both sizes (bit-exact by standard, INFERRED).
2. Encode: the hardware stream decodes on the CPU with the right frame count and size, and PSNR (Y) against the source >= the floor. Floor: 26 dB, set from the CPU
   reference measured BEFORE any hardware encode (`run_r11_video.sh --cpu-reference`, 300 frames, synthetic testsrc2 + noise, CPU encoders at their script presets, PSNR Y in dB):
   | | h264 (libx264 veryfast) | hevc (libx265 veryfast) | av1 (SVT-AV1 preset 10, 9 at 4k) |
   |---|---|---|---|
   | 1080p 8 Mbit/s | 33.93 | 34.85 | 29.06 |
   | 1080p 40 Mbit/s | 35.68 | 36.12 | 29.22 |
   | 4k 8 Mbit/s | 33.40 | 33.84 | 30.67 |
   | 4k 40 Mbit/s | 33.99 | 35.01 | 30.84 |
   Lowest 29.06 (AV1, 1080p, 8 Mbit/s); floor = 26 dB, about 3 dB below it (run with `R11_PSNR_FLOOR=26`; the script's built-in default of 35 would fail the CPU encoders themselves).
   The noise in the source caps the PSNR of any encoder at these rates, and SVT-AV1 barely moves between 8 and 40 Mbit/s (its own rate control), so this floor only catches a broken or
   wrong-source encode, not a quality ranking; the quality comparison of the hardware encoders is a result of the run, not a control.
3. Device: during every hardware cell the dGPU's video fields or `mm_act` exceed the first 60 s idle baseline; otherwise the cell is void.
4. Render node found by device id (0x7550 -> renderD128 today; the iGPU 0x13c0 is renderD129): never given by hand.
Content caveat: fixed-function speed depends little on content, bit rate depends a lot (synthetic testsrc2 + noise inputs).

## Deliberate breaks
1. `--break-device`: decode on the iGPU node: the dGPU activity check must FAIL.
2. One byte of a stream corrupted: the md5 comparison must FAIL (proved on the CPU, tiny clip: FAILED as required).
3. PSNR against a different source: must FAIL the floor (proved on the CPU: 11 dB against a floor of 30).
Also proved on the CPU: wrong frame count FAILS; unreadable PSNR FAILS; an idle window and an empty window FAIL the activity check; the analysis marks the break cell FAIL as required.

## Estimate
72 cells (36 decode, 36 encode) at about 6 s each, plus 2 x 60 s idle baselines and the controls: about 8 to 10 min of card time; host memory under 4 GB; scratch about 8 GB.
