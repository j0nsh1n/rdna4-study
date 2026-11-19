#!/usr/bin/env python3
"""Exact-weights reference files for the round-2 quality runs. Run with the convert venv's python (gguf, numpy, torch,
safetensors): toolchains/convert-venv/bin/python -I scripts/llm_ref.py <command>.

  exactness <fp8.gguf>              how many FP8 x scale products each tensor holds exactly in F16, BF16 and F32
  make3b <fp8.gguf> <out.gguf> [f32|f16]   every F8_E4M3 tensor becomes F32 = fp8 value x scale (the .scale tensors are dropped);
                                    every other tensor and all metadata are copied byte for byte
  compare3b <fp8.gguf> <ref.gguf> [rows]   bit compare: sampled rows of every FP8 tensor against fp8 x scale in float64,
                                    whole other tensors byte for byte
  compare8b <checkpoint dir> <ref.gguf> [rows]   the same against the safetensors checkpoint (q and k rows in the converter's order)
  selftest                          deliberate breaks on a tiny canned file
"""
import json
import os
import sys
import tempfile

import numpy as np

import gguf
import torch

F8 = gguf.GGMLQuantizationType.F8_E4M3
F32 = gguf.GGMLQuantizationType.F32


def fp8_to_f64(raw_u8):
    return torch.from_numpy(np.ascontiguousarray(raw_u8)).view(torch.float8_e4m3fn).double().numpy()


def products(raw_u8, scale):
    """float64 products fp8 x scale; the scale has at most 24 significant bits and an E4M3 value 4, so float64 holds them."""
    return fp8_to_f64(raw_u8) * np.float64(scale)


def exactness_row(p):
    """Fraction of finite products that survive a round trip through F32, F16 and BF16."""
    fin = np.isfinite(p)
    p = p[fin]
    f32 = float((p.astype(np.float32).astype(np.float64) == p).mean())
    f16 = float((p.astype(np.float16).astype(np.float64) == p).mean())
    bf = torch.from_numpy(p.astype(np.float64)).to(torch.bfloat16).double().numpy()
    return f32, f16, float((bf == p).mean()), int((~fin).sum())


def tensors_by_name(r):
    return {t.name: t for t in r.tensors}


def cmd_exactness(path):
    r = gguf.GGUFReader(path)
    byname = tensors_by_name(r)
    worst = [1.0, 1.0, 1.0]
    best16 = 0.0
    n = 0
    for t in r.tensors:
        if t.tensor_type != F8:
            continue
        sc = byname[t.name.replace(".weight", ".scale")].data[0]
        f32, f16, bf, bad = exactness_row(products(np.asarray(t.data), sc))
        if bad:
            print(f"FAIL {t.name}: {bad} non-finite products")
            return 1
        n += 1
        worst = [min(worst[0], f32), min(worst[1], f16), min(worst[2], bf)]
        best16 = max(best16, f16)
    print(f"exactness over {n} FP8 tensors: lowest share exact F32 {worst[0]:.6f}, F16 {worst[1]:.4f} (highest {best16:.4f}), BF16 {worst[2]:.4f}")
    if worst[0] != 1.0:
        print("FAIL: F32 is not exact for every product")
        return 1
    if best16 == 1.0 and worst[1] == 1.0:
        print("NOTE: F16 is exact for every tensor")
    return 0


def copy_fields(r, w):
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types
        if t[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(name, f.contents(), t[0], t[1])
        else:
            w.add_key_value(name, f.contents(), t[0])


def cmd_make3b(src, dst, dtype="f32"):
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    byname = tensors_by_name(r)
    w = gguf.GGUFWriter(dst, arch)
    copy_fields(r, w)
    plan = []
    for t in r.tensors:
        if t.tensor_type == F32 and t.name.endswith(".scale") and t.name.replace(".scale", ".weight") in byname:
            if byname[t.name.replace(".scale", ".weight")].tensor_type == F8:
                continue
        if t.tensor_type == F8:
            shape = tuple(int(x) for x in t.data.shape)
            if dtype == "f16":
                w.add_tensor_info(t.name, shape, np.dtype(np.float16), int(np.prod(shape)) * 2, gguf.GGMLQuantizationType.F16)
            else:
                w.add_tensor_info(t.name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4, F32)
            plan.append((t, True))
        else:
            data = t.data
            w.add_tensor_info(t.name, tuple(int(x) for x in data.shape), data.dtype, int(t.n_bytes), t.tensor_type)
            plan.append((t, False))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.write_padding(w.fout[0], w.fout[0].tell())
    inexact = total = 0
    worst = 0.0
    for t, conv in plan:
        if conv:
            p = products(np.asarray(t.data), byname[t.name.replace(".weight", ".scale")].data[0])
            arr = p.astype(np.float16 if dtype == "f16" else np.float32)
            if dtype == "f16":
                inexact += int((arr.astype(np.float64) != p).sum())
                total += p.size
                worst = max(worst, float(np.max(np.abs(arr.astype(np.float64) - p) / np.maximum(np.abs(p), 1e-300))))
            elif not np.array_equal(arr.astype(np.float64), p):
                print(f"FAIL {t.name}: products not exact in F32")
                return 1
            raw = arr.tobytes()
        else:
            raw = np.ascontiguousarray(t.data).tobytes()
        w.fout[0].write(raw)
        w.write_padding(w.fout[0], len(raw))
    w.fout[0].flush()
    w.close()
    print(f"wrote {dst}: {len(plan)} tensors, {os.path.getsize(dst)} bytes")
    if dtype == "f16":
        print(f"F16 rounding of the FP8 x scale products: {inexact} of {total} ({100 * inexact / total:.2f} %) changed, worst relative error {worst:.3e}")
    return 0


def sample_rows(n_rows, k):
    if n_rows >= k:
        return np.arange(k)
    rng = np.random.default_rng(7)
    return np.unique(np.concatenate([[0, k - 1], rng.choice(k, n_rows, replace=False)]))


def compare_fp8_tensor(name, ref_t, fp8_rows_fn, nrows, scale, bad):
    """Row sample of one FP8 tensor: float64 products must equal the F32 reference exactly (bit compare of the F32 value)."""
    k = int(ref_t.data.shape[0])
    ref = np.asarray(ref_t.data)
    for i in sample_rows(nrows, k):
        want = np.float64(1) * fp8_rows_fn(int(i)) * np.float64(scale)
        got = ref[i]
        if got.dtype != np.float32 or not np.array_equal(want.astype(np.float32).view(np.uint32), got.view(np.uint32)) or \
                not np.array_equal(want, got.astype(np.float64)):
            bad.append(f"{name} row {i}")
            return


def cmd_compare3b(src, ref_path, nrows=8):
    r = gguf.GGUFReader(src)
    ref = gguf.GGUFReader(ref_path)
    a, b = tensors_by_name(r), tensors_by_name(ref)
    bad, rows, whole = [], 0, 0
    for name, t in a.items():
        if name.endswith(".scale") and name.replace(".scale", ".weight") in a and a[name.replace(".scale", ".weight")].tensor_type == F8:
            if name in b:
                bad.append(f"{name} should not be in the reference")
            continue
        if name not in b:
            bad.append(f"{name} missing in the reference")
            continue
        if t.tensor_type == F8:
            if b[name].tensor_type != F32:
                bad.append(f"{name} is not F32 in the reference")
                continue
            raw = np.asarray(t.data)
            sc = a[name.replace(".weight", ".scale")].data[0]
            compare_fp8_tensor(name, b[name], lambda i, raw=raw: fp8_to_f64(raw[i]), nrows, sc, bad)
            rows += len(sample_rows(nrows, raw.shape[0]))
        else:
            if t.tensor_type != b[name].tensor_type or not np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(b[name].data).view(np.uint8)):
                bad.append(f"{name} differs")
            whole += 1
    extra = set(b) - set(a)
    if extra:
        bad.append(f"extra tensors in the reference: {sorted(extra)[:3]}")
    return report_compare("3B", bad, rows, whole)


def report_compare(label, bad, rows, whole):
    if bad:
        print(f"FAIL {label} reference is not exact: {len(bad)} problems, first: {bad[:3]}")
        return 1
    print(f"PASS {label} reference: {rows} sampled FP8 rows bit-identical to fp8 x scale (float64 equal, F32 bits equal); {whole} other tensors byte-identical")
    return 0


def permute(w, n_head):
    """The converter's Llama q/k row reorder (convert_hf_to_gguf.py `permute`): applied to the checkpoint rows to get the GGUF order."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
      "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


OTHER8 = {"token_embd.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight", "output_norm.weight": "model.norm.weight"}
OTHER8_BLK = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def same_values(ck, arr):
    """The checkpoint tensor (BF16 or F32) widened to float32 equals the reference array bit for bit."""
    a = ck.float().numpy()
    return a.shape == arr.shape and arr.dtype == np.float32 and np.array_equal(a.view(np.uint32), arr.view(np.uint32))


def cmd_compare8b(ckpt, ref_path, nrows=8):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    ref = gguf.GGUFReader(ref_path)
    bad, rows, whole = [], 0, 0

    def get(k):
        with safe_open(os.path.join(ckpt, idx[k]), "pt") as f:
            return f.get_tensor(k)

    for t in ref.tensors:
        parts = t.name.split(".")
        if len(parts) == 4 and parts[0] == "blk" and parts[2] in HF:
            hf = f"model.layers.{parts[1]}.{HF[parts[2]]}.weight"
            w = get(hf)
            sc = float(get(hf + "_scale").float().item())
            if w.dtype != torch.float8_e4m3fn or t.tensor_type != F32:
                bad.append(f"{t.name}: checkpoint {w.dtype}, reference {t.tensor_type.name}")
                continue
            raw = w.view(torch.uint8).numpy()
            if parts[2] == "attn_q":
                raw = permute(raw, cfg["num_attention_heads"])
            elif parts[2] == "attn_k":
                raw = permute(raw, cfg["num_key_value_heads"])
            ref_arr = np.asarray(t.data)
            for i in sample_rows(nrows, ref_arr.shape[0]):
                want = fp8_to_f64(raw[i]) * np.float64(sc)
                got = ref_arr[i]
                if not np.array_equal(want.astype(np.float32).view(np.uint32), got.view(np.uint32)) or not np.array_equal(want, got.astype(np.float64)):
                    bad.append(f"{t.name} row {i}")
                    break
            rows += len(sample_rows(nrows, ref_arr.shape[0]))
        else:
            hf = OTHER8.get(t.name) or (f"model.layers.{parts[1]}.{OTHER8_BLK[parts[2]]}.weight" if len(parts) == 4 and parts[0] == "blk" and parts[2] in OTHER8_BLK else None)
            if t.name == "rope_freqs.weight":
                continue  # generated by the converter from the rope settings, not in the checkpoint
            if hf is None:
                bad.append(f"{t.name}: no checkpoint name known")
                continue
            if not same_values(get(hf), np.asarray(t.data)):
                bad.append(f"{t.name} differs from {hf}")
            whole += 1
    return report_compare("8B", bad, rows, whole)


def canned(path, damage=False, f16_exact_scale=False):
    """A tiny FP8 GGUF: one fp8 weight with a BF16-class scale, one norm, one BF16 matrix."""
    w = gguf.GGUFWriter(path, "llama")
    raw = (np.arange(64, dtype=np.uint8) % 0x78).reshape(8, 8)  # finite E4M3 codes
    scale = np.float32(2.0 if f16_exact_scale else 2.0 ** -9 * 1.0078125)
    w.add_tensor("blk.0.attn_q.weight", raw, raw_dtype=F8)
    w.add_tensor("blk.0.attn_q.scale", np.array([scale], dtype=np.float32))
    w.add_tensor("blk.0.attn_norm.weight", np.ones(8, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def selftest():
    bad = 0

    def expect(name, want, got):
        nonlocal bad
        ok = want == got
        print(f"{'ok  ' if ok else 'BAD '}{name}: want {want}, got {got}")
        bad += not ok

    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", None)) as d:
        a, b = os.path.join(d, "fp8.gguf"), os.path.join(d, "ref.gguf")
        canned(a)
        # the canned file's scale has 8 significant bits, so F16 cannot hold every product but F32 can
        expect("make3b on the canned file", 0, cmd_make3b(a, b))
        expect("control: the exact reference passes the bit compare", 0, cmd_compare3b(a, b, 100))
        expect("exactness check passes (F32 exact)", 0, cmd_exactness(a))
        # break 1: one reference element changed by one bit in the last row
        data = bytearray(open(b, "rb").read())
        ref = gguf.GGUFReader(b)
        t = [x for x in ref.tensors if x.name == "blk.0.attn_q.weight"][0]
        off = int(t.data_offset) + 4 * 63
        data[off] ^= 1
        b2 = os.path.join(d, "ref-damaged.gguf")
        open(b2, "wb").write(data)
        expect("break: one bit of the last row changed is caught", 1, cmd_compare3b(a, b2, 100))
        # break 2: a reference made with an F16 round trip is not exact and is caught
        arr = np.asarray(t.data).astype(np.float16).astype(np.float32)
        data = bytearray(open(b, "rb").read())
        data[int(t.data_offset):int(t.data_offset) + arr.nbytes] = arr.tobytes()
        b3 = os.path.join(d, "ref-f16.gguf")
        open(b3, "wb").write(data)
        expect("break: a reference rounded through F16 is caught", 1, cmd_compare3b(a, b3, 100))
        # break 3: the sample misses the damaged row when only row 0 is checked? (rows 0 and k-1 are always included)
        data = bytearray(open(b, "rb").read())
        data[off - 4 * 8 * 3] ^= 1  # a row in the middle
        b4 = os.path.join(d, "ref-mid.gguf")
        open(b4, "wb").write(data)
        expect("break: damage in a middle row is caught when all rows are sampled", 1, cmd_compare3b(a, b4, 100))
        # the exactness check reports F16 share below 1 for the canned scale
        r = gguf.GGUFReader(a)
        p = products(np.asarray([x for x in r.tensors if x.name.endswith("attn_q.weight")][0].data), 2.0 ** -9 * 1.0078125)
        f32, f16, bf, _ = exactness_row(p)
        expect("exactness: F32 all exact", 1.0, f32)
        expect("exactness: F16 not all exact for an 8-bit scale", True, f16 < 1.0)
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        expect("same_values: equal tensors", True, same_values(torch.from_numpy(x.copy()), x))
        y = x.copy()
        y[1, 2] = np.nextafter(y[1, 2], np.float32(10))
        expect("break: one ulp apart is caught", False, same_values(torch.from_numpy(x.copy()), y))
        # break 4: a scale with 24 significant bits makes F32 inexact and the check says so
        p2 = products(np.asarray([x for x in r.tensors if x.name.endswith("attn_q.weight")][0].data), 2.0 ** -9 * (1 + 2.0 ** -23))
        expect("break: a 24-bit scale is detected as not exact in F32", True, exactness_row(p2)[0] < 1.0)
    print("selftest:", "FAILED" if bad else "all passed")
    return 1 if bad else 0


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    c = a[0]
    if c == "exactness" and len(a) == 2:
        return cmd_exactness(a[1])
    if c == "make3b" and len(a) in (3, 4):
        return cmd_make3b(a[1], a[2], a[3] if len(a) == 4 else "f32")
    if c == "compare3b" and len(a) in (3, 4):
        return cmd_compare3b(a[1], a[2], int(a[3]) if len(a) == 4 else 8)
    if c == "compare8b" and len(a) in (3, 4):
        return cmd_compare8b(a[1], a[2], int(a[3]) if len(a) == 4 else 8)
    if c == "selftest":
        return selftest()
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
