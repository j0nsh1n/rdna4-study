#!/usr/bin/env python3
"""Round 4: are the downloaded FP8 checkpoint's weights the ones the 3B FP8 GGUF was made from?
Run with the convert venv:  llm4_origin.py compare <checkpoint dir> <gguf>   |   selftest
Compares every tensor bit for bit (FP8 weights: raw bytes, with the converter's q/k row reorder; scales and norms: values)."""
import json, os, sys
import numpy as np
import torch
import gguf
from safetensors import safe_open

HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
      "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NORM = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight"}


def permute(w, n_head):
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def compare(ckpt, gguf_path, damage=False):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    f = safe_open(os.path.join(ckpt, "model.safetensors"), "pt")
    r = gguf.GGUFReader(gguf_path)
    res = {"fp8_equal": 0, "fp8_diff": [], "scale_equal": 0, "scale_diff": [], "other_equal": 0, "other_diff": [], "skipped": []}
    for t in r.tensors:
        p = t.name.split(".")
        if p[0] == "blk" and len(p) == 4 and p[2] in HF and p[3] == "weight":
            hf = f"model.layers.{p[1]}.{HF[p[2]]}.weight"
            raw = f.get_tensor(hf).view(torch.uint8).numpy()
            if p[2] == "attn_q":
                raw = permute(raw, cfg["num_attention_heads"])
            elif p[2] == "attn_k":
                raw = permute(raw, cfg["num_key_value_heads"])
            got = np.asarray(t.data).view(np.uint8).reshape(raw.shape)
            if damage and not res['fp8_diff'] and res['fp8_equal'] == 5:
                raw = raw.copy(); raw[3, 3] ^= 1  # deliberate break: one bit of the sixth weight tensor
            if np.array_equal(raw, got): res["fp8_equal"] += 1
            else: res["fp8_diff"].append(t.name)
        elif p[0] == "blk" and len(p) == 4 and p[2] in HF and p[3] in ("scale", "input_scale"):
            hf = f"model.layers.{p[1]}.{HF[p[2]]}." + ("weight_scale" if p[3] == "scale" else "input_scale")
            a = f.get_tensor(hf).float().numpy().reshape(-1); b = np.asarray(t.data).astype(np.float32).reshape(-1)
            if a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32)): res["scale_equal"] += 1
            else: res["scale_diff"].append(t.name)
        else:
            hf = TOP.get(t.name) or (f"model.layers.{p[1]}.{NORM[p[2]]}.weight" if p[0] == "blk" and len(p) == 4 and p[2] in NORM else None)
            if hf is None:
                res["skipped"].append(f"{t.name} ({t.tensor_type.name}; no checkpoint twin: rope_freqs from the converter, output.weight is Q8_0 made by llama-quantize from lm_head)")
                continue
            a = f.get_tensor(hf).float().numpy()
            d = np.asarray(t.data)
            if t.tensor_type.name == "BF16":
                b = (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(a.shape)
            else:
                b = d.astype(np.float32).reshape(a.shape)
            if np.array_equal(a.view(np.uint32), b.view(np.uint32)): res["other_equal"] += 1
            else: res["other_diff"].append(t.name)
    return res


def main():
    if sys.argv[1] == "selftest":
        return selftest()
    res = compare(sys.argv[2], sys.argv[3], damage=len(sys.argv) > 4 and sys.argv[4] == "--damage")
    for k, v in res.items():
        print(k, v if not isinstance(v, list) else (len(v), v[:6]))
    ok = not res["fp8_diff"] and not res["scale_diff"] and not res["other_diff"] and res["fp8_equal"] == 196 and res["scale_equal"] == 392
    print("VERDICT", "PASS identical" if ok else "FAIL not identical")
    return 0 if ok else 1


def selftest():
    # permute known answer: 2 heads, 4 rows: rows reorder (0,2,1,3) within each head
    w = np.arange(8).reshape(8, 1)
    out = permute(w, 2).reshape(-1).tolist()
    assert out == [0, 2, 1, 3, 4, 6, 5, 7], out
    a = np.zeros(4, np.uint8); b = a.copy(); b[2] = 1
    assert np.array_equal(a, a) and not np.array_equal(a, b)
    print("VERDICT llm4_origin selftest PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
