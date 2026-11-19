#!/usr/bin/env python3
"""Round 4 debugging probe: PyTorch's scaled_dot_product_attention alone, at the shapes the HF Llama 3B prefill uses (24 heads, head size 128,
causal, bf16), at growing lengths; prints before each call so a GPU hang names the length. Optional: --backend math|flash|efficient."""
import sys, torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
be = {"default": None, "math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}[sys.argv[1]]
for L in [int(x) for x in sys.argv[2:]]:
    q = torch.randn(1, 24, L, 128, device="cuda", dtype=torch.bfloat16)
    print("sdpa", sys.argv[1], L, flush=True)
    if be is None:
        y = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    else:
        with sdpa_kernel(be):
            y = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    print("ok", L, float(y.float().abs().mean()), flush=True)
