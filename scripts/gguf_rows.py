#!/usr/bin/env python3
"""Minimal GGUF v3 reader plus a raw row extractor (standard library only).

Usage:
  gguf_rows.py FILE list
  gguf_rows.py FILE probe TENSOR            byte statistics of the first rows of one tensor
  gguf_rows.py FILE extract TENSOR OUT ROWS [FIRST_ROW]
"""
import struct
import sys
from collections import Counter

# Value types of GGUF metadata.
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        n = self.unpack("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            return self.unpack(SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            et = self.unpack("<I")
            n = self.unpack("<Q")
            # Large arrays (vocab) are skipped element by element but not kept.
            vals = [self.value(et) for _ in range(n)]
            return vals if n <= 16 else "<array of %d>" % n
        raise ValueError("unknown metadata type %d" % t)


def read_gguf(path):
    f = open(path, "rb")
    r = Reader(f)
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.unpack("<I")
    if version != 3:
        raise ValueError("only GGUF v3 handled, got %d" % version)
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.unpack("<I")
        meta[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.unpack("<I")
        dims = [r.unpack("<Q") for _ in range(nd)]
        ttype = r.unpack("<I")
        off = r.unpack("<Q")
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = meta.get("general.alignment", 32)
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return f, meta, tensors, data_start


def cmd_list(path):
    f, meta, tensors, data_start = read_gguf(path)
    print("data_start", data_start, "tensors", len(tensors))
    for k, v in meta.items():
        print("META", k, "=", v if not isinstance(v, str) or len(v) < 200 else v[:200] + "...")
    for t in tensors:
        print("TENSOR %-36s type %2d dims %s offset %d" % (t["name"], t["type"], t["dims"], t["offset"]))
    cnt = Counter(t["type"] for t in tensors)
    print("types in file (type id: tensor count):", dict(cnt))
    # Size from consecutive offsets exposes the bytes per element of each type.
    order = sorted(tensors, key=lambda t: t["offset"])
    sizes = {}
    for a, b in zip(order, order[1:]):
        sizes[a["name"]] = b["offset"] - a["offset"]
    seen = {}
    for t in order[:-1]:
        n = 1
        for d in t["dims"]:
            n *= d
        seen.setdefault(t["type"], (t["name"], sizes[t["name"]] / n))
    for ty, (nm, bpe) in sorted(seen.items()):
        print("type %d: bytes per element (from offsets, incl. padding) %.4f, e.g. %s" % (ty, bpe, nm))


def find(tensors, name):
    for t in tensors:
        if t["name"] == name:
            return t
    raise SystemExit("no tensor " + name)


def cmd_probe(path, name):
    f, meta, tensors, ds = read_gguf(path)
    t = find(tensors, name)
    cols = t["dims"][0]
    f.seek(ds + t["offset"])
    buf = f.read(64 * cols)
    print("tensor", name, "dims", t["dims"], "type", t["type"])
    print("first 96 bytes hex:", buf[:96].hex(" "))
    h = Counter(buf)
    print("byte histogram, top 20 (byte: count):", h.most_common(20))
    exp = Counter((b >> 3) & 15 for b in buf)
    print("exponent field of low 7 bits (E4M3 reading), counts 0..15:", [exp.get(i, 0) for i in range(16)])
    print("sign bit share %.4f, zero bytes %.4f" % (sum(1 for b in buf if b & 0x80) / len(buf), h.get(0, 0) / len(buf)))
    print("NaN codes 0x7f/0xff:", h.get(0x7F, 0), h.get(0xFF, 0))


def cmd_extract(path, name, out, rows, first=0):
    f, meta, tensors, ds = read_gguf(path)
    t = find(tensors, name)
    cols = t["dims"][0]
    nrows = t["dims"][1] if len(t["dims"]) > 1 else 1
    rows = min(rows, nrows - first)
    # The per-tensor scale lives in a separate F32 tensor "<layer>.scale" next to the weight.
    scale = 0.0
    kind = 0
    if name.endswith(".weight"):
        sc = [x for x in tensors if x["name"] == name[:-len(".weight")] + ".scale"]
        if sc:
            f.seek(ds + sc[0]["offset"])
            scale = struct.unpack("<f", f.read(4))[0]
            kind = 1
    f.seek(ds + t["offset"] + first * cols)  # 1 byte per element: checked by the probe
    buf = f.read(rows * cols)
    with open(out, "wb") as o:
        # Header: magic, rows u32, cols u32, scale kind u32 (0 none, 1 one F32 per tensor), ggml type u32,
        # scale f32; then rows * cols raw E4M3 codes, row-major.
        o.write(b"E4M3ROWS")
        o.write(struct.pack("<IIIIf", rows, cols, kind, t["type"], scale))
        o.write(buf)
    print("wrote %s: %d rows x %d cols from %s (first row %d), per-tensor scale %r (kind %d)" % (out, rows, cols, name, first, scale, kind))


if __name__ == "__main__":
    a = sys.argv
    if len(a) >= 3 and a[2] == "list":
        cmd_list(a[1])
    elif len(a) >= 4 and a[2] == "probe":
        cmd_probe(a[1], a[3])
    elif len(a) >= 6 and a[2] == "extract":
        cmd_extract(a[1], a[3], a[4], int(a[5]), int(a[6]) if len(a) > 6 else 0)
    else:
        print(__doc__)
        sys.exit(2)
