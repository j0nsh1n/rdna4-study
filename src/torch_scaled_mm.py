"""torch._scaled_mm with float8_e4m3fn on the card: does it run, is it right, how fast next to an F16 matmul.

Run with the PyTorch venv's python and -I. Device 0 must be the card under test (the script sets
HIP_VISIBLE_DEVICES). Prints RESULT lines and one VERDICT line; exits 1 on a failure.
Shape: M x K times K x N with M = N = 4096, K = 512, F32 output, scales of 1.
"""
import os
import sys

import torch


def verdict(status, text):
    print(f"VERDICT torch_scaled_mm: {status} {text}", flush=True)


def time_ms(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def main():
    want = os.environ.get("CARD_ARCH", "")
    if not torch.cuda.is_available():
        verdict("FAIL", "no GPU visible to PyTorch")
        return 1
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"torch {torch.__version__}, device 0 {torch.cuda.get_device_name(0)} ({arch})")
    if want and arch != want:
        print(f"device 0 is {arch}, card record says {want}; refusing to run", file=sys.stderr)
        return 3
    m, n, k = 4096, 4096, 512
    g = torch.Generator(device="cuda").manual_seed(5)
    # |x| <= 4 so every value is exact in E4M3 after the cast and the sums stay small
    a = (torch.rand(m, k, device="cuda", generator=g) * 8 - 4).to(torch.float8_e4m3fn)
    b_t = (torch.rand(n, k, device="cuda", generator=g) * 8 - 4).to(torch.float8_e4m3fn)
    b = b_t.t()  # column-major K x N, as _scaled_mm requires
    one = torch.tensor(1.0, device="cuda")
    try:
        out = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
    except Exception as e:  # report the reason in the run log, the phase records the failure
        verdict("FAIL", f"_scaled_mm raised {type(e).__name__}: {e}")
        return 1
    ref = a.double() @ b.double()
    mag = a.double().abs() @ b.double().abs()
    max_rel = ((out.double() - ref).abs() / (mag + 1e-30)).max().item()
    a16, b16 = a.half(), b.half()
    t8 = time_ms(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32))
    t16 = time_ms(lambda: a16 @ b16)
    flop = 2.0 * m * n * k
    print(f"RESULT torch_scaled_mm median_ms {t8:.4f} TFLOPS {flop / (t8 * 1e9):.1f}")
    print(f"RESULT torch_matmul_f16 median_ms {t16:.4f} TFLOPS {flop / (t16 * 1e9):.1f}")
    print(f"RESULT accuracy max_rel {max_rel:.2e}")
    ok = max_rel < 1e-5
    verdict("PASS" if ok else "FAIL",
            f"_scaled_mm FP8 {m}x{n}x{k} max_rel {max_rel:.1e}, {flop / (t8 * 1e9):.1f} TFLOPS against F16 matmul "
            f"{flop / (t16 * 1e9):.1f} TFLOPS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
