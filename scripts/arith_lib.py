"""Lane 5 helpers: format encoding, exact values and the GPU runner call (build/arith/bin/arith_run).

Exploration scripts import this; every GPU call takes the shared lock. Values are Python Fractions (exact).
"""
import os
import struct
import subprocess
from fractions import Fraction as Fr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "build/arith/bin/arith_run")
SCRATCH = os.path.join(REPO, "build/arith/explore")
def _local_setting(name, default):
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(os.path.join(REPO, "local.env")) as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key == name:
                    return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return default


LOCK = _local_setting("STUDY_GPU_LOCK", os.path.join(REPO, "runs", "gpu.lock"))
FORMS = {"f16": 0, "bf16": 1, "fp8_fp8": 2, "fp8_bf8": 3, "bf8_fp8": 4, "bf8_bf8": 5, "f16_f16": 6,
         "bf16_bf16": 7, "iu8": 8, "iu4": 9}
# element format of A and B per form
AB_FMT = {"f16": ("f16", "f16"), "bf16": ("bf16", "bf16"), "fp8_fp8": ("e4m3", "e4m3"),
          "fp8_bf8": ("e4m3", "e5m2"), "bf8_fp8": ("e5m2", "e4m3"), "bf8_bf8": ("e5m2", "e5m2"),
          "f16_f16": ("f16", "f16"), "bf16_bf16": ("bf16", "bf16")}
FMT = {  # exponent bits, mantissa bits, bias, kind of top code
    "f16": (5, 10, 15, "ieee"), "bf16": (8, 7, 127, "ieee"), "f32": (8, 23, 127, "ieee"),
    "e4m3": (4, 3, 7, "e4m3"), "e5m2": (5, 2, 15, "ieee")}


def decode(fmt, bits):
    """Exact value as Fraction, or the strings 'nan', 'inf', '-inf'."""
    e_bits, m_bits, bias, kind = FMT[fmt]
    sign = (bits >> (e_bits + m_bits)) & 1
    e = (bits >> m_bits) & ((1 << e_bits) - 1)
    m = bits & ((1 << m_bits) - 1)
    if kind == "e4m3" and e == 15 and m == 7:
        return "nan"
    if kind == "ieee" and e == (1 << e_bits) - 1:
        return "nan" if m else ("-inf" if sign else "inf")
    if e == 0:
        v = Fr(m, 1 << m_bits) * Fr(2) ** (1 - bias)
    else:
        v = (1 + Fr(m, 1 << m_bits)) * Fr(2) ** (e - bias)
    return -v if sign else v


def encode(fmt, v):
    """Bits of an exactly representable value (raises if not representable)."""
    e_bits, m_bits, bias, kind = FMT[fmt]
    sign = 0
    if v < 0 or (v == 0 and isinstance(v, float) and str(v).startswith("-")):
        sign = 1
    v = abs(Fr(v))
    if v == 0:
        return sign << (e_bits + m_bits)
    e = 0
    while v >= 2 ** (e + 1):
        e += 1
    while v < Fr(2) ** e:
        e -= 1
    be = e + bias
    if be <= 0:
        m = v / (Fr(2) ** (1 - bias)) * (1 << m_bits)
        be = 0
    else:
        m = (v / Fr(2) ** e - 1) * (1 << m_bits)
    top = (1 << e_bits) - (1 if kind == "ieee" else 0)
    if m.denominator != 1 or be >= top or (kind == "e4m3" and be == 15 and m == 7):
        raise ValueError(f"{v} not representable in {fmt}")
    return (sign << (e_bits + m_bits)) | (be << m_bits) | int(m)


def f32_bits(v):
    if isinstance(v, str):
        return {"nan": 0x7FC00000, "inf": 0x7F800000, "-inf": 0xFF800000}[v]
    return encode("f32", v)


def neg0(fmt):
    e_bits, m_bits, _, _ = FMT[fmt]
    return 1 << (e_bits + m_bits)


def run(form, elements, mod=0, mode=None, shift=0, steps=1, tag="x"):
    """elements: list of (a_bits[16 * steps], b_bits[16 * steps], c_bits). Returns (result bits list, mode seen)."""
    os.makedirs(SCRATCH, exist_ok=True)
    n = len(elements)
    hdr = [FORMS[form], mod, steps, 1 if mode is not None else 0, mode or 0, n, shift]
    buf = bytearray(struct.pack("<7I", *hdr))
    for a, b, c in elements:
        assert len(a) == 16 * steps and len(b) == 16 * steps
        for s in range(steps):
            buf += struct.pack("<16I", *a[16 * s:16 * s + 16])
            buf += struct.pack("<16I", *b[16 * s:16 * s + 16])
        buf += struct.pack("<I", c & 0xFFFFFFFF)
    fin = os.path.join(SCRATCH, f"{tag}.in")
    fout = os.path.join(SCRATCH, f"{tag}.out")
    with open(fin, "wb") as f:
        f.write(buf)
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0", CARD_ARCH="gfx1201")
    subprocess.run(["flock", LOCK, "nice", "-n", "10", "timeout", "120", RUNNER, fin, fout], check=True, env=env)
    with open(fout, "rb") as f:
        out = list(struct.unpack(f"<{n + 1}I", f.read()))
    return out[:n], out[n]


def val32(bits):
    return decode("f32", bits)


def bits_str(bits):
    v = val32(bits)
    return f"0x{bits:08x} ({v if isinstance(v, str) else float(v)!r})"


EVAL = os.path.join(REPO, "build/arith/bin/arith_eval")


def model(form, elements, steps=1, tag="m", params=()):
    """CPU model (build/arith/bin/arith_eval) on the same element format as run(): list of (model, exact) bits."""
    os.makedirs(SCRATCH, exist_ok=True)
    n = len(elements)
    buf = bytearray(struct.pack("<7I", FORMS[form], 0, steps, 0, 0, n, 0))
    for a, b, c in elements:
        for s in range(steps):
            buf += struct.pack("<16I", *a[16 * s:16 * s + 16])
            buf += struct.pack("<16I", *b[16 * s:16 * s + 16])
        buf += struct.pack("<I", c & 0xFFFFFFFF)
    fin = os.path.join(SCRATCH, f"{tag}.min")
    fout = os.path.join(SCRATCH, f"{tag}.mout")
    with open(fin, "wb") as f:
        f.write(buf)
    subprocess.run([EVAL, fin, fout, *params], check=True)
    with open(fout, "rb") as f:
        w = struct.unpack(f"<{2 * n}I", f.read())
    return [(w[2 * i], w[2 * i + 1]) for i in range(n)]
