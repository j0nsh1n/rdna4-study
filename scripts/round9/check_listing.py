#!/usr/bin/env python3
"""Listing checks for the round 9 probes (CPU only; standard library; run with python3 -I).

Usage:
  check_listing.py ld_order <disassembly> <notes> <ld_order_blocks.h> [--wavefront N] [--cumode 0|1] (default 1: built with -mcumode)
  check_listing.py ld_mat   <disassembly> <notes> <ld_mat_blocks.h>   [--wavefront N] [--cumode 0|1]
  check_listing.py ld_mat_k <disassembly> <notes> <ld_mat_k_blocks.h> [--wavefront N] [--cumode 0|1]   (same checks as ld_mat)
  check_listing.py selftest <ld_order|ld_mat|ld_mat_k> <disassembly> <notes> <header>
Disassembly: llvm-objdump -d --mcpu=gfx1201 of the device code object taken out of the binary that will run.
Notes: llvm-readelf --notes of the same code object (the descriptor: .vgpr_count, .wavefront_size,
.max_flat_workgroup_size, .workgroup_processor_mode).

What is checked for every kernel of the header:
  - the instruction text of the asm statement (prelude, pattern, drain), as written in the header and in the probe's
    source, appears in the code object as one unbroken run of instructions (same opcodes, same order, same registers; a
    compiler-chosen register is a wildcard), at every copy of the statement the compiler made, and the number of copies
    that start with the statement's first instruction equals the number of complete copies;
  - the descriptor: .wavefront_size, .workgroup_processor_mode (0 is CU mode, -mcumode), the register count and the
    largest work-group the descriptor accepts (ld_mat: the groups of the cell must fit the register file of one CU).
Exit 1 on any failure. `selftest` runs the check on the real header (must pass) and on deliberately wrong ones (every
one must fail): an opcode changed, a register changed, a wait count changed, the wrong wavefront size, the wrong group
size.
"""
import re
import sys

# ---------------------------------------------------------------------------------------------------------------
# text handling


def norm(t):
    t = t.split("//")[0].strip().lower()
    t = re.sub(r"^([a-z0-9_]+?)_e(?:32|64)\b", r"\1", t)
    t = re.sub(r"0x[0-9a-f]+", lambda m: str(int(m.group(0), 16)), t)
    t = re.sub(r"\s+offset:0(?!\d)", "", t)  # the disassembler leaves out a zero offset
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t)
    return t


WILD = r"(?:s\d+|s\[\d+:\d+\]|v\d+|v\[\d+:\d+\])"


def to_re(line):
    n = norm(line)
    n = re.sub(r"%\[\w+\]", "@", n)
    n = re.sub(r"\b2b$", "~", n)  # a branch back to the loop label: the listing shows the offset
    return re.compile("^" + re.escape(n).replace("@", WILD).replace("~", r"-?\d+") + "$")


def read_dis(path):
    """name -> list of normalized instruction strings."""
    out, cur = {}, None
    for ln in open(path, errors="replace"):
        m = re.match(r"^[0-9a-f]+ <(\S+)>:", ln)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is not None and ln.startswith("\t"):
            t = ln.split("//")[0].strip()
            if t:
                out[cur].append(norm(t))
    return out


def sym_name(sym):
    """_ZN12_GLOBAL__N_17k_mh_w1EPKc... -> k_mh_w1 (length-prefixed)."""
    m = re.search(r"_GLOBAL__N_1(\d+)(k_\w+)", sym)
    if not m:
        m = re.match(r"_Z(\d+)(k_\w+)", sym)
    if not m:
        return None
    return m.group(2)[: int(m.group(1))]


def read_notes(path):
    kernels, cur = {}, None
    for ln in open(path, errors="replace"):
        if re.match(r"^  - \.", ln):
            cur = {}
        m = re.match(r"^\s{4}(\.\w+):\s*(.*)$", ln) if cur is not None else None
        if m:
            cur[m.group(1)] = m.group(2).strip()
            if m.group(1) == ".name":
                kernels[m.group(2).strip()] = cur
    return kernels


def parse_header(path, nfields=0):
    """Header entries X(name, f1, ..., "text" "text" ...): returns name -> (fields, text lines)."""
    s = open(path).read()
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    out = {}
    pos = 0
    while True:
        m = re.compile(r"(?<![\w])X\(").search(s, pos)
        if not m:
            break
        i = m.end()
        items = []
        while i < len(s):
            while i < len(s) and (s[i].isspace() or s[i] == "\\"):
                i += 1
            if s[i] == ")":
                i += 1
                break
            if s[i] == ",":
                i += 1
                continue
            if s[i] == '"':
                parts = []
                while True:
                    j = i + 1
                    while s[j] != '"':
                        j += 2 if s[j] == "\\" else 1
                    parts.append(s[i + 1:j])
                    i = j + 1
                    k = i
                    while k < len(s) and (s[k].isspace() or s[k] == "\\"):
                        k += 1
                    if k < len(s) and s[k] == '"':
                        i = k
                        continue
                    break
                items.append(("str", "".join(parts)))
            else:
                j = i
                while s[j] not in ",)":
                    j += 1
                items.append(("tok", s[i:j].strip()))
                i = j
        pos = i
        name = items[0][1]
        fields = [v for _, v in items[1:-1]]
        text = items[-1][1].replace("\\n", "\n")
        out[name] = (fields, [ln for ln in text.splitlines() if ln.strip()])
    return out


def match_runs(instrs, pats):
    """(copies starting with the first expected instruction, complete copies)."""
    starts = [i for i in range(len(instrs)) if pats[0].match(instrs[i])]
    full = 0
    for i in starts:
        if i + len(pats) <= len(instrs) and all(pats[j].match(instrs[i + j]) for j in range(len(pats))):
            full += 1
    return len(starts), full


# ---------------------------------------------------------------------------------------------------------------
# ld_order

LDO_PRE = [
    "v_mov_b32 v40, 0xdeadbeef", "v_mov_b32 v41, 0xdeadbeef", "v_mov_b32 v42, 0xdeadbeef",
    "v_mov_b32 v43, 0xdeadbeef", "v_mov_b32 v44, 0xdeadbeef", "v_mov_b32 v45, 0xdeadbeef",
    "s_mov_b32 s48, 0xdeadbeef",
    "global_load_b32 v50, v62, %[nbase]", "s_load_b32 s50, %[sn], 0x0",
    "s_wait_loadcnt 0", "s_wait_kmcnt 0",
    "s_getreg_b32 %[t0], hwreg(HW_REG_SHADER_CYCLES_LO)",
]
LDO_POST = ["s_wait_loadcnt 0", "s_wait_dscnt 0", "s_wait_kmcnt 0"]


def check_ld_order(dis, notes, hdr, wf=32, cumode=1, quiet=False):
    bad = 0
    pats = parse_header(hdr, 6)
    kern = {sym_name(k): v for k, v in read_dis(dis).items() if sym_name(k)}
    desc = {sym_name(k.replace(".kd", "")): v for k, v in read_notes(notes).items() if sym_name(k)}
    if not pats:
        print("FAIL: no patterns found in the header")
        return 1
    for name, (fields, body) in sorted(pats.items()):
        k = "k_" + name
        if k not in kern:
            print(f"FAIL {name}: no kernel {k} in the listing")
            bad += 1
            continue
        want = [to_re(x) for x in LDO_PRE + body + LDO_POST]
        starts, full = match_runs(kern[k], want)
        if full < 1 or starts != full:
            print(f"FAIL {name}: {full} complete copies of the asm statement for {starts} copies that start like it")
            bad += 1
        d = desc.get(k)
        if d is None:
            print(f"FAIL {name}: no descriptor")
            bad += 1
            continue
        if int(d[".wavefront_size"]) != wf:
            print(f"FAIL {name}: .wavefront_size {d['.wavefront_size']} (want {wf})")
            bad += 1
        if int(d.get(".workgroup_processor_mode", "0")) != (0 if cumode else 1):
            print(f"FAIL {name}: .workgroup_processor_mode {d.get('.workgroup_processor_mode')} (cumode {cumode})")
            bad += 1
        if int(d[".max_flat_workgroup_size"]) != 32:
            print(f"FAIL {name}: .max_flat_workgroup_size {d['.max_flat_workgroup_size']} (want 32: one wave)")
            bad += 1
        if int(d[".vgpr_spill_count"]) != 0 or int(d[".sgpr_spill_count"]) != 0:
            print(f"FAIL {name}: register spills")
            bad += 1
    if not quiet:
        print(f"ld_order listing: {len(pats)} patterns checked, {bad} failures")
    return bad


# ---------------------------------------------------------------------------------------------------------------
# ld_mat. Header: X(name, D, K, R, TPB, "text") where text is one iteration of the loop body (the probe's source adds
# the prelude and the loop control, listed below); name = d<D>_r<R> or ref_r<R>.

LDM_HEAD_PROLOGUE = "prologue"  # the prologue loads are part of the header text (they differ with D)
LDM_TAIL = [
    "s_add_co_i32 %[cnt], %[cnt], -1", "s_cmp_lg_u32 %[cnt], 0", "s_cbranch_scc1 2b",
]


def ring_and_footprint(body):
    """Static check of one ld_mat / ld_mat_k stream (the header text): every matrix instruction reads the pair that the load
    for exactly that position wrote, over three iterations (prologue loads, then the body with the loop edge), so a load
    that lands on a pair a later matrix instruction still has to read is found; and where each load's address comes from.
    Returns (errors, nmm, unmasked_footprint_bytes, all_masked): unmasked_footprint is the largest immediate offset + 256 of a
    load that uses voff directly (ld_mat_k's form); all_masked is true when every load takes a v_add + v_and masked address."""
    err = []
    pro, loop = [], []
    cur = pro
    for x in body:
        t = x.strip()
        if re.match(r"^\d+:$", t):
            cur = loop
        elif cur is loop and t.startswith(("s_add_co_i32", "s_cmp", "s_cbranch")):
            continue
        else:
            cur.append(t)
    nmm = sum(1 for t in loop if t.startswith("v_wmma"))
    gn = sum(1 for t in loop if t.startswith("global_load"))  # load groups per iteration
    if nmm == 0:
        return ["no matrix instruction in the loop body"], 0, 0, False
    if gn == 0:  # the matrix-only reference: no loads, nothing to clobber
        return [], nmm, 0, True
    per = nmm // gn  # matrix instructions per load group (1 in ld_mat)
    reg = {}  # first register of the pair -> tag (global number of the position the pair is loaded for)
    raw = {}  # address register state: ("raw", imm) after v_add, ("masked", imm) after v_and
    foot, masked = 0, True

    def one(t, it, n):
        nonlocal foot, masked
        m = re.match(r"v_add_nc_u32 (%\[atmp\]), (\d+), %\[voff\]$", t)
        if m:
            raw["a"] = ("raw", int(m.group(2)))
            return n
        if t == "v_and_b32 %[atmp], %[mask], %[atmp]":
            if raw.get("a", ("", 0))[0] == "raw":
                raw["a"] = ("masked", raw["a"][1])
            else:
                err.append("v_and without the v_add before it")
            return n
        m = re.match(r"global_load_b64 v\[(\d+):(\d+)\], (%\[atmp\]|%\[voff\]), %\[base\](?: offset:(\d+))?$", t)
        if m:
            if m.group(3) == "%[atmp]":
                st, imm = raw.get("a", ("none", 0))
                if st != "masked":
                    err.append(f"a load whose address was not masked into the span ({t})")
                    masked = False
            else:
                imm = int(m.group(4) or 0)
                masked = False
                foot = max(foot, imm + 256)
            reg[int(m.group(1))] = it * gn + imm // 256
            return n
        m = re.match(r"v_wmma_f32_16x16x16_fp8_fp8 v\[16:23\], v\[(\d+):\d+\], v\[24:25\], v\[16:23\]$", t)
        if m:
            a = int(m.group(1))
            want = it * gn + n // per
            if reg.get(a) != want:
                err.append(f"matrix instruction {want} reads v{a}, which holds the load for position {reg.get(a)} (a load wrote over it, or its load is missing)")
            return n + 1
        return n

    for t in pro:
        one(t, 0, 0)
    for it in range(3):
        n = 0
        for t in loop:
            n = one(t, it, n)
    return err[:3], nmm, foot, masked


def selftest_ring(hdr, which):
    """The static ring check must pass on the real header and fail on deliberately broken copies of its streams."""
    pats = parse_header(hdr, 6)
    ok = True
    real = {n: ring_and_footprint(b) for n, (f, b) in pats.items()}
    bad_real = [n for n, r in real.items() if r[0]]
    print(f"selftest: ring check on the real header: {'PASS' if not bad_real else 'FAIL ' + ','.join(bad_real)}")
    ok &= not bad_real
    # a ring that does not divide the body: the first block that has loads, its last load aimed at the pair of the first
    name = next(n for n, (f, b) in sorted(pats.items()) if any(x.startswith("global_load") for x in b))
    body = list(pats[name][1])
    idx = [i for i, x in enumerate(body) if x.startswith("global_load")]
    first = re.search(r"v\[(\d+):", body[idx[0]]).group(1)
    last = body[idx[-1]]
    brk = list(body)
    brk[idx[-1]] = re.sub(r"v\[\d+:\d+\]", f"v[{first}:{int(first) + 1}]", last)
    r = ring_and_footprint(brk)
    print(f"selftest: deliberate break (last load of {name} aimed at the first load's pair): {'FAILED as it must' if r[0] else 'DID NOT FAIL'}")
    ok &= bool(r[0])
    if which == "ld_mat":
        brk = list(body)
        j = next(i for i, x in enumerate(brk) if x.startswith("v_and_b32 %[atmp]"))
        brk[j] = "s_nop 0"
        r = ring_and_footprint(brk)
        print(f"selftest: deliberate break (one address not masked into the span): {'FAILED as it must' if r[0] else 'DID NOT FAIL'}")
        ok &= bool(r[0])
    return ok


def fits(vgprs, threads):
    """Does a work-group of `threads` threads (wave32, CU mode: waves split over the two SIMDs) fit the register file?
    Each SIMD holds 1536 registers per lane, allocated in granules of 24 (the rule of mem_occupancy.hip)."""
    alloc = (vgprs + 23) // 24 * 24
    per_simd = (threads // 32 + 1) // 2
    return per_simd * alloc <= 1536


def check_ld_mat(dis, notes, hdr, wf=32, cumode=1, quiet=False, which="ld_mat"):
    bad = 0
    pats = parse_header(hdr, 6)
    kern = {sym_name(k): v for k, v in read_dis(dis).items() if sym_name(k)}
    desc = {sym_name(k.replace(".kd", "")): v for k, v in read_notes(notes).items() if sym_name(k)}
    if not pats:
        print("FAIL: no blocks found in the header")
        return 1
    for name, (fields, body) in sorted(pats.items()):
        D, K, R, TPB = (int(x) for x in fields[:4])
        k = "k_" + name
        if k not in kern:
            print(f"FAIL {name}: no kernel {k} in the listing")
            bad += 1
            continue
        # body = prologue + loop label body + counter updates, as in the probe: the first line of the loop is the
        # first line of the iteration; the loop control follows. Expected: the whole statement as one run.
        want = [to_re(x) for x in body if not re.match(r"^\d+:$", x.strip())]  # labels are not instructions
        starts, full = match_runs(kern[k], want)
        if full < 1 or starts != full:
            print(f"FAIL {name}: {full} complete copies of the asm statement for {starts} copies that start like it")
            bad += 1
        d = desc.get(k)
        if d is None:
            print(f"FAIL {name}: no descriptor")
            bad += 1
            continue
        if int(d[".wavefront_size"]) != wf:
            print(f"FAIL {name}: .wavefront_size {d['.wavefront_size']} (want {wf})")
            bad += 1
        if int(d.get(".workgroup_processor_mode", "0")) != (0 if cumode else 1):
            print(f"FAIL {name}: .workgroup_processor_mode {d.get('.workgroup_processor_mode')} (cumode {cumode})")
            bad += 1
        vg = int(d[".vgpr_count"])
        if vg != R:
            print(f"FAIL {name}: .vgpr_count {vg}, the cell is padded to {R}")
            bad += 1
        mx = int(d[".max_flat_workgroup_size"])
        if mx != TPB:
            print(f"FAIL {name}: .max_flat_workgroup_size {mx}, the cell launches {TPB} threads per group")
            bad += 1
        # a group of TPB threads has TPB/32 waves on one CU (two SIMDs): each SIMD needs waves x allocation <= 1536
        errs, nmm, foot, masked = ring_and_footprint(body)
        for e in errs:
            print(f"FAIL {name}: {e}")
            bad += 1
        if which == "ld_mat" and not masked:
            print(f"FAIL {name}: a load reads at voff plus an immediate offset (footprint {foot} bytes), not masked into the level's span")
            bad += 1
        if which == "ld_mat_k" and foot > 8192 and not quiet:
            print(f"NOTE {name}: loads read at voff plus an immediate offset: footprint {foot / 1024:.2f} KiB, more than the 8 KiB L0 span "
                  f"(the L0 cell of this block is not an L0-resident test)")
        if not fits(vg, TPB):
            print(f"FAIL {name}: {TPB} threads with {vg} registers do not fit the register file of a CU: the launch is refused")
            bad += 1
        if int(d[".vgpr_spill_count"]) != 0 or int(d[".sgpr_spill_count"]) != 0 or d.get(".private_segment_fixed_size", "0") != "0":
            print(f"FAIL {name}: spills or scratch")
            bad += 1
    if not quiet:
        print(f"ld_mat listing: {len(pats)} blocks checked, {bad} failures")
    return bad


# ---------------------------------------------------------------------------------------------------------------


def selftest(which, dis, notes, hdr):
    fn = check_ld_order if which == "ld_order" else (lambda *a, **k: check_ld_mat(*a, which=which, **k))
    ok = True
    r = fn(dis, notes, hdr, quiet=True)
    print(f"selftest: real header: {'PASS' if r == 0 else 'FAIL'} ({r} failures)")
    ok &= r == 0
    src = open(hdr).read()
    if which.startswith("ld_mat"):
        rule = fits(104, 1024) is False and fits(90, 1024) and fits(167, 576) and fits(209, 448) and not fits(209, 1024) and not fits(167, 1024)
        print(f"selftest: register-file rule (104 registers with 1024 threads must be refused, as in round 7; 90/1024, 167/576, 209/448 must fit): {'PASS' if rule else 'FAIL'}")
        ok &= bool(rule)

    def broken(label, text, **kw):
        nonlocal ok
        tmp = hdr + ".bad"
        open(tmp, "w").write(text)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            f = fn(dis, notes, tmp, quiet=True, **kw)
        print(f"selftest: deliberate break ({label}): {'FAILED as it must' if f else 'DID NOT FAIL'}")
        ok &= f > 0

    # the first wait or first load of the first pattern with an unusual opcode
    if which == "ld_order":
        broken("opcode: s_wait_loadcnt 1 in mh_w1 changed to s_wait_dscnt 1", src.replace("s_wait_loadcnt 1", "s_wait_dscnt 1", 1))
        broken("wait count: s_wait_loadcnt 1 changed to 2", src.replace("s_wait_loadcnt 1", "s_wait_loadcnt 2", 1))
        broken("register: first v_mov_b32 v64, v40 changed to v65", src.replace("v_mov_b32 v64, v40", "v_mov_b32 v65, v40", 1))
        broken("wavefront size 64 expected", src, wf=64)
        broken("workgroup-processor mode expected where CU mode is built", src, cumode=0)
    else:
        broken("opcode: first s_wait_loadcnt changed to s_wait_dscnt", src.replace("s_wait_loadcnt", "s_wait_dscnt", 1))
        broken("register: first v_wmma operand changed", src.replace("v_wmma_f32_16x16x16_fp8_fp8 v[16:23]", "v_wmma_f32_16x16x16_fp8_fp8 v[24:31]", 1))
        broken("wavefront size 64 expected", src, wf=64)
        if which == "ld_mat_k":
            broken("wait count: first s_wait_loadcnt 12 changed to 11", src.replace("s_wait_loadcnt 12", "s_wait_loadcnt 11", 1))
            broken("an extra instruction inserted into the first stream that reads v28:29", src.replace("\\n\" \"v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[28:29]", "\\n\" \"s_nop 0\\n\" \"v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[28:29]", 1))
        broken("group size: TPB of the first block doubled", re.sub(r"(X\(\w+, \d+, \d+, \d+, )(\d+)", lambda m: m.group(1) + str(int(m.group(2)) * 2), src, count=1))
    if which.startswith("ld_mat"):
        ok &= selftest_ring(hdr, which)
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2

    def opt(name, default):
        return int(a[a.index(name) + 1]) if name in a else default

    if a[0] == "selftest":
        return selftest(a[1], a[2], a[3], a[4])
    wf = opt("--wavefront", 32)
    if a[0] == "ld_order":
        return 1 if check_ld_order(a[1], a[2], a[3], wf, opt("--cumode", 1)) else 0
    if a[0] in ("ld_mat", "ld_mat_k"):
        return 1 if check_ld_mat(a[1], a[2], a[3], wf, opt("--cumode", 1), which=a[0]) else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
