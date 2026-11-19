#!/usr/bin/env python3
"""Round 3, Q2: which FP8 tensors of a GGUF hold every FP8 x scale product exactly in F16, by computation, and what a
mixed reference file (F16 where exact, else F32) would weigh. Run with the convert venv's python:
toolchains/convert-venv/bin/python -I scripts/llm3_mixed_ref.py <command>

  plan <fp8.gguf>     per FP8 tensor: scale, significand bits of the scale, the product set of the codes PRESENT in the tensor,
                      exact-in-F16 test (significand bits at most 11 AND exponent range, by round trip through numpy float16),
                      then counts and bytes: F16 tensors, F32 tensors, other tensors as stored, total
  build <fp8.gguf> <out.gguf>   the mixed reference itself (round 4 follow-up): every F8_E4M3 tensor becomes fp8 x scale, stored as F16 when
                      the plan says every product is exact in F16, else F32 (the build refuses a tensor with a NaN code or a product that F32 cannot hold exactly, and checks every element: stored value == float64 product); the .scale tensors
                      are dropped; every other tensor and all metadata are copied byte for byte. One tensor at a time (host memory stays small).
  compare <fp8.gguf> <ref.gguf> [rows]   bit compare: sampled rows (first, last and random ones) of every FP8 tensor against fp8 x scale in
                      float64 (F16 rows widened, F32 rows by their bits), whole other tensors byte for byte, the scale tensors absent
  selftest            deliberate breaks on canned scales and codes, and on a canned file through build and compare
Exit 1 if a check fails.
"""
import os
import sys
import tempfile

import numpy as np

import gguf
import torch

F8 = gguf.GGMLQuantizationType.F8_E4M3
GB = 1e9


def decode_all():
    """All 256 E4M3 codes as float64 (NaN codes included), by torch's float8_e4m3fn."""
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double().numpy()


def sig_bits(x):
    """Significant bits of a finite nonzero float64 (position of the highest set bit minus the lowest)."""
    m, e = np.frexp(abs(float(x)))
    mant = int(m * (1 << 53))
    low = (mant & -mant).bit_length() - 1
    return 53 - low


def f16_exact(products):
    """True if every finite product survives a round trip through float16 (covers significand AND exponent range)."""
    p = products[np.isfinite(products)]
    with np.errstate(over="ignore"):
        return bool((p.astype(np.float16).astype(np.float64) == p).all())


def tensor_row(raw_u8, scale):
    present = np.flatnonzero(np.bincount(raw_u8.reshape(-1), minlength=256))
    vals = decode_all()[present]
    fin = np.isfinite(vals)
    prods = vals[fin] * np.float64(scale)
    nz = prods[prods != 0]
    ok16 = f16_exact(prods)
    ok32 = bool((prods.astype(np.float32).astype(np.float64) == prods).all())
    maxabs = float(np.abs(prods).max()) if prods.size else 0.0
    minabs = float(np.abs(nz).min()) if nz.size else 0.0
    bits = int(max(sig_bits(p) for p in nz)) if nz.size else 0
    return {"present_codes": int(present.size), "nan_codes": int((~fin).sum()), "f16": ok16, "f32": ok32, "max_bits": bits,
            "maxabs": maxabs, "minabs": minabs, "scale_bits": sig_bits(scale) if scale != 0 else 0}


def cmd_plan(path):
    r = gguf.GGUFReader(path)
    byname = {t.name: t for t in r.tensors}
    n16 = n32 = 0
    b16 = b32 = b_other = 0
    e16 = e32 = 0
    bad = 0
    print("name\telements\tscale\tscale_bits\tmax_product_bits\tminabs\tmaxabs\tf16_exact\tf32_exact\tpresent_codes")
    for t in r.tensors:
        if t.tensor_type != F8:
            b_other += int(t.n_bytes)
            continue
        sc = float(byname[t.name.replace(".weight", ".scale")].data[0])
        row = tensor_row(np.asarray(t.data), sc)
        n = int(np.prod(t.shape))
        if not row["f32"] or row["nan_codes"]:
            bad += 1
        if row["f16"]:
            n16 += 1; e16 += n; b16 += 2 * n
        else:
            n32 += 1; e32 += n; b32 += 4 * n
        print(f"{t.name}\t{n}\t{sc:.6g}\t{row['scale_bits']}\t{row['max_bits']}\t{row['minabs']:.4g}\t{row['maxabs']:.4g}\t{int(row['f16'])}\t{int(row['f32'])}\t{row['present_codes']}")
    # scale tensors are dropped from the reference (as in round 2's make3b)
    scale_bytes = sum(int(t.n_bytes) for t in r.tensors if t.name.endswith(".scale"))
    b_other -= scale_bytes
    print(f"# FP8 tensors: {n16 + n32}; exact in F16 (bits and range): {n16} tensors, {e16} elements; need F32: {n32} tensors, {e32} elements; products not exact in F32 or NaN codes: {bad}")
    print(f"# mixed file: F16 tensors {b16 / GB:.3f} GB, F32 tensors {b32 / GB:.3f} GB, other tensors as stored (embedding, output, norms; scales dropped) {b_other / GB:.3f} GB, total {(b16 + b32 + b_other) / GB:.3f} GB")
    print(f"# all-F32 file would be {(4 * (e16 + e32) + b_other) / GB:.3f} GB, all-F16 {(2 * (e16 + e32) + b_other) / GB:.3f} GB")
    return 1 if bad else 0


def products64(raw_u8, scale):
    return decode_all()[np.asarray(raw_u8)] * np.float64(scale)


def is_scale_of_fp8(name, byname):
    return name.endswith(".scale") and byname.get(name.replace(".scale", ".weight")) is not None and byname[name.replace(".scale", ".weight")].tensor_type == F8


def copy_fields(r, w):
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types
        if t[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(name, f.contents(), t[0], t[1])
        else:
            w.add_key_value(name, f.contents(), t[0])


def cmd_build(src, dst):
    r = gguf.GGUFReader(src)
    byname = {t.name: t for t in r.tensors}
    w = gguf.GGUFWriter(dst, r.fields["general.architecture"].contents())
    copy_fields(r, w)
    plan = []
    n16 = n32 = 0
    for t in r.tensors:
        if is_scale_of_fp8(t.name, byname):
            continue
        if t.tensor_type == F8:
            sc = float(byname[t.name.replace(".weight", ".scale")].data[0])
            row = tensor_row(np.asarray(t.data), sc)
            if not row["f32"] or row["nan_codes"]:
                print(f"FAIL {t.name}: products not exact in F32 or NaN codes present")
                return 1
            shape = tuple(int(x) for x in t.data.shape)
            if row["f16"]:
                w.add_tensor_info(t.name, shape, np.dtype(np.float16), int(np.prod(shape)) * 2, gguf.GGMLQuantizationType.F16)
                n16 += 1
            else:
                w.add_tensor_info(t.name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4, gguf.GGMLQuantizationType.F32)
                n32 += 1
            plan.append((t, sc, row["f16"]))
        else:
            data = t.data
            w.add_tensor_info(t.name, tuple(int(x) for x in data.shape), data.dtype, int(t.n_bytes), t.tensor_type)
            plan.append((t, None, None))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.write_padding(w.fout[0], w.fout[0].tell())
    for t, sc, as16 in plan:
        if sc is None:
            raw = np.ascontiguousarray(t.data).tobytes()
        else:
            p = products64(np.asarray(t.data), sc)
            arr = p.astype(np.float16 if as16 else np.float32)
            if not np.array_equal(arr.astype(np.float64), p):
                print(f"FAIL {t.name}: the stored values are not exactly fp8 x scale")
                return 1
            raw = arr.tobytes()
        w.fout[0].write(raw)
        w.write_padding(w.fout[0], len(raw))
    w.fout[0].flush()
    w.close()
    print(f"wrote {dst}: {len(plan)} tensors ({n16} F16, {n32} F32 from FP8), {os.path.getsize(dst)} bytes")
    return 0


def sample_rows(n_rows, k):
    if n_rows >= k:
        return np.arange(k)
    rng = np.random.default_rng(7)
    return np.unique(np.concatenate([[0, k - 1], rng.choice(k, n_rows, replace=False)]))


def cmd_compare(src, ref_path, nrows=8):
    r = gguf.GGUFReader(src)
    ref = gguf.GGUFReader(ref_path)
    a = {t.name: t for t in r.tensors}
    b = {t.name: t for t in ref.tensors}
    bad, rows, whole, n16, n32 = [], 0, 0, 0, 0
    for name, t in a.items():
        if is_scale_of_fp8(name, a):
            if name in b:
                bad.append(f"{name} should not be in the reference")
            continue
        if name not in b:
            bad.append(f"{name} missing in the reference")
            continue
        if t.tensor_type == F8:
            bt = b[name].tensor_type
            if bt not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                bad.append(f"{name} is {bt.name} in the reference")
                continue
            sc = np.float64(a[name.replace(".weight", ".scale")].data[0])
            raw = np.asarray(t.data)
            got = np.asarray(b[name].data)
            n16 += bt == gguf.GGMLQuantizationType.F16
            n32 += bt == gguf.GGMLQuantizationType.F32
            for i in sample_rows(nrows, raw.shape[0]):
                want = products64(raw[i], sc)
                have = got[i]
                if bt == gguf.GGMLQuantizationType.F32:
                    same = np.array_equal(want.astype(np.float32).view(np.uint32), have.view(np.uint32)) and np.array_equal(want, have.astype(np.float64))
                else:
                    same = np.array_equal(want.astype(np.float16).view(np.uint16), have.view(np.uint16)) and np.array_equal(want, have.astype(np.float64))
                if not same:
                    bad.append(f"{name} row {i}")
                    break
            rows += len(sample_rows(nrows, raw.shape[0]))
        else:
            if t.tensor_type != b[name].tensor_type or not np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(b[name].data).view(np.uint8)):
                bad.append(f"{name} differs")
            whole += 1
    extra = set(b) - {n for n in a if not is_scale_of_fp8(n, a)}
    if extra:
        bad.append(f"extra tensors in the reference: {sorted(extra)[:3]}")
    if bad:
        print(f"FAIL mixed reference is not exact: {len(bad)} problems, first: {bad[:3]}")
        return 1
    print(f"PASS mixed reference: {rows} sampled rows of {n16 + n32} FP8 tensors ({n16} F16, {n32} F32) bit-identical to fp8 x scale; {whole} other tensors byte-identical")
    return 0


def canned(path):
    """A tiny FP8 GGUF: an F16-exact tensor (scale 2), an F32-only tensor (an 8-bit scale), one norm, one BF16-class matrix."""
    w = gguf.GGUFWriter(path, "llama")
    raw = (np.arange(64, dtype=np.uint8) % 0x78).reshape(8, 8)
    w.add_tensor("blk.0.attn_q.weight", raw, raw_dtype=F8)
    w.add_tensor("blk.0.attn_q.scale", np.array([2.0], dtype=np.float32))
    w.add_tensor("blk.0.attn_k.weight", raw, raw_dtype=F8)
    w.add_tensor("blk.0.attn_k.scale", np.array([255 / 128 * 2.0 ** -9], dtype=np.float32))
    w.add_tensor("blk.0.attn_norm.weight", np.ones(8, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def selftest_build(expect):
    with tempfile.TemporaryDirectory(dir=os.environ.get("LLM_RUN", None)) as d:
        a, b = os.path.join(d, "fp8.gguf"), os.path.join(d, "ref.gguf")
        canned(a)
        expect("build on the canned file", cmd_build(a, b), 0)
        ref = gguf.GGUFReader(b)
        ty = {t.name: t.tensor_type.name for t in ref.tensors}
        expect("the 1-bit-scale tensor is stored F16, the 8-bit-scale one F32, the scales are dropped",
               (ty.get("blk.0.attn_q.weight"), ty.get("blk.0.attn_k.weight"), "blk.0.attn_q.scale" in ty), ("F16", "F32", False))
        expect("control: the exact reference passes the compare", cmd_compare(a, b, 100), 0)
        for name, ofs in (("blk.0.attn_k.weight", 4 * 63), ("blk.0.attn_k.weight", 4 * 20), ("blk.0.attn_q.weight", 2 * 63)):
            t = [x for x in ref.tensors if x.name == name][0]
            data = bytearray(open(b, "rb").read())
            data[int(t.data_offset) + ofs] ^= 1
            bd = os.path.join(d, "damaged.gguf")
            open(bd, "wb").write(data)
            expect(f"break: one bit of {name} at byte {ofs} is caught", cmd_compare(a, bd, 100), 1)
        # a reference rounded through F16 where F32 was needed is caught
        t = [x for x in ref.tensors if x.name == "blk.0.attn_k.weight"][0]
        arr = np.asarray(t.data).astype(np.float16).astype(np.float32)
        data = bytearray(open(b, "rb").read())
        data[int(t.data_offset):int(t.data_offset) + arr.nbytes] = arr.tobytes()
        bd = os.path.join(d, "rounded.gguf")
        open(bd, "wb").write(data)
        expect("break: the F32 tensor rounded through F16 is caught", cmd_compare(a, bd, 100), 1)
        # the norm tensor changed by one bit (a copied tensor must be byte for byte)
        t = [x for x in ref.tensors if x.name == "blk.0.attn_norm.weight"][0]
        data = bytearray(open(b, "rb").read())
        data[int(t.data_offset)] ^= 1
        bd = os.path.join(d, "norm.gguf")
        open(bd, "wb").write(data)
        expect("break: a copied tensor changed by one bit is caught", cmd_compare(a, bd, 100), 1)


def selftest():
    fails = 0

    def expect(name, got, want):
        nonlocal fails
        ok = got == want
        print(("PASS" if ok else "FAIL (selftest broken)") + f" {name}: got {got}, want {want}")
        fails += not ok

    allc = decode_all()
    codes = np.arange(256, dtype=np.uint8)
    finite_codes = codes[np.isfinite(allc)]
    # known answers worked out by hand: E4M3 values have at most 4 significant bits
    expect("scale 1.0 (1 bit): all codes exact in F16", tensor_row(finite_codes, 1.0)["f16"], True)
    expect("scale 0x7f/64 (7 bits): 11-bit products exact in F16", tensor_row(finite_codes, 127 / 64)["f16"], True)
    expect("scale 255/128 (8 bits): a 12-bit product (15 x 255) is not exact in F16", tensor_row(finite_codes, 255 / 128)["f16"], False)
    expect("scale 2^-20 (1 bit): products below 2^-24 do not fit by range", tensor_row(finite_codes, 2.0 ** -20)["f16"], False)
    expect("scale 2^-20 is still exact in F32", tensor_row(finite_codes, 2.0 ** -20)["f32"], True)
    expect("scale 2^8 (1 bit): 448 x 256 = 114688 passes the F16 maximum 65504? no: out of range", tensor_row(finite_codes, 256.0)["f16"], False)
    # only the codes present count: a tensor holding only 0x00 and 0x38 (1.0) with an 8-bit scale is exact
    only = np.array([0x00, 0x38, 0x38], dtype=np.uint8)
    expect("tensor holding only 0.0 and 1.0 with an 8-bit scale: exact in F16", tensor_row(only, 255 / 128)["f16"], True)
    expect("significand bits of 255/128", sig_bits(255 / 128), 8)
    expect("significand bits of 15 * 255 / 128", sig_bits(15 * 255 / 128), 12)
    # break: a test that would say 'exact' for everything must be caught by the 12-bit case above
    wrong = f16_exact(np.array([15 * 255 / 128 * 2.0 ** -3]))
    expect("a 12-bit product is not exact (the check would be broken if this said True)", wrong, False)
    selftest_build(expect)
    print("VERDICT llm3_mixed_ref selftest " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


def main():
    a = sys.argv[1:]
    if a[:1] == ["plan"] and len(a) == 2:
        return cmd_plan(a[1])
    if a[:1] == ["build"] and len(a) == 3:
        return cmd_build(a[1], a[2])
    if a[:1] == ["compare"] and len(a) in (3, 4):
        return cmd_compare(a[1], a[2], int(a[3]) if len(a) == 4 else 8)
    if a[:1] == ["selftest"]:
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
