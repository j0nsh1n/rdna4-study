#!/usr/bin/env python3
"""Round 4 R1: which matrix instructions are in the machine code of the kernels a trace named. CPU only; python3 -I.
  llm4_isa.py hipblaslt <names file> <library dir> <cache dir for the unbundled code objects> [--objdump PATH]   disassemble each named hipBLASLt (Tensile) kernel from the code object that holds it
  llm4_isa.py triton <dir with .amdgcn files>                          count matrix instructions in Triton's saved assembly
  llm4_isa.py selftest
Output per kernel: the distinct v_wmma_* opcodes and their counts. FP8 means an opcode with fp8 or bf8 in its name."""
import collections, glob, os, re, subprocess, sys

WMMA = re.compile(r"\b(v_(?:s?wmma|swmmac)\w*)")
OBJDUMP = os.path.expanduser("~/.local/opt/rocm/lib/llvm/bin/llvm-objdump")


def count_wmma(text):
    return collections.Counter(WMMA.findall(text))


BUNDLER = os.path.expanduser("~/.local/opt/rocm/lib/llvm/bin/clang-offload-bundler")


def unbundled(co, cache):
    """hipBLASLt's .co files are compressed offload bundles (magic CCOB); the code object for gfx1201 is cut out into the cache folder."""
    out = os.path.join(cache, os.path.basename(co)[:-3] + ".hsaco")
    if not os.path.exists(out):
        subprocess.run([BUNDLER, "--unbundle", "--type=o", "--targets=hipv4-amdgcn-amd-amdhsa--gfx1201", f"--input={co}", f"--output={out}"],
                       capture_output=True)
    return out if os.path.exists(out) else None


def find_co(name, libdir, cache=None):
    key = name.encode()
    cos = sorted(glob.glob(os.path.join(libdir, "*.co")))
    if cache is None:
        return None
    os.makedirs(cache, exist_ok=True)
    tok = name.split("_")[3] if name.count("_") > 3 else ""
    first = [c for c in cos if ("F8F8" in c) == tok.startswith("F8")]
    for f in first + [c for c in cos if c not in first]:
        o = unbundled(f, cache)
        if o:
            with open(o, "rb") as fh:
                if key in fh.read():
                    return o
    return None


def disasm(co, name, objdump):
    """The kernel symbols have size 0 in these code objects, so the code runs from the symbol's address to the next function symbol's."""
    readelf = os.path.join(os.path.dirname(objdump), "llvm-readelf")
    syms = subprocess.run([readelf, "-sW", co], capture_output=True, text=True).stdout.splitlines()
    funcs, start = [], None
    for l in syms:
        f = l.split()
        if len(f) >= 8 and f[3] == "FUNC":
            addr = int(f[1], 16)
            funcs.append(addr)
            if f[7] == name:
                start = addr
    if start is None:
        return "", "symbol not found"
    stop = min([a for a in funcs if a > start], default=None)
    if stop is None:
        return "", "no following symbol"
    p = subprocess.run([objdump, "-d", "--mcpu=gfx1201", f"--start-address={start}", f"--stop-address={stop}", co], capture_output=True, text=True)
    return p.stdout, p.stderr


def cmd_hipblaslt(names_file, libdir, cache, objdump):
    ok = True
    for name in [l.strip() for l in open(names_file) if l.strip()]:
        co = find_co(name, libdir, cache)
        short = name[:60]
        if co is None:
            print(f"NOTFOUND {short}")
            ok = False
            continue
        txt, err = disasm(co, name, objdump)
        c = count_wmma(txt)
        fp8 = sum(v for k, v in c.items() if "fp8" in k or "bf8" in k)
        print(f"{os.path.basename(co)[:70]} :: {short}...: instructions {len(txt.splitlines())} lines; wmma {dict(c)}; fp8-wmma {fp8}")
    return ok


def cmd_triton(d):
    for f in sorted(glob.glob(os.path.join(d, "**", "*.amdgcn"), recursive=True)):
        c = count_wmma(open(f).read())
        if c:
            print(os.path.basename(f), dict(c))


def selftest():
    t = "v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11], v[0:7]\nv_wmma_f32_16x16x16_fp8_fp8 v[0:7]\nv_wmma_f32_16x16x16_bf16 v[0:7]\nv_add_f32 v1, v2, v3\n"
    c = count_wmma(t)
    assert c == {"v_wmma_f32_16x16x16_fp8_fp8": 2, "v_wmma_f32_16x16x16_bf16": 1}, c
    assert count_wmma("v_add_f32 v1, v2, v3") == {}
    import tempfile
    assert find_co("nonexistent_kernel_xyz", "/nonexistent", tempfile.mkdtemp()) is None
    print("VERDICT llm4_isa selftest PASS")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "selftest":
        selftest()
    elif a[0] == "hipblaslt":
        sys.exit(0 if cmd_hipblaslt(a[1], a[2], a[3], a[a.index("--objdump") + 1] if "--objdump" in a else OBJDUMP) else 1)
    elif a[0] == "triton":
        cmd_triton(a[1])
