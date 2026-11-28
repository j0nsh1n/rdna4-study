#!/usr/bin/env python3
"""Listing check for round 10's ld_mat2 (CPU only; standard library; run with python3 -I).

Usage:
  check_ld_mat2.py check    <disassembly> [--notes FILE]
  check_ld_mat2.py selftest <disassembly> <out folder> [--notes FILE]
Disassembly: llvm-objdump -d --mcpu=gfx1201 of the device code object taken out of the binary that will run (the .s listing).
Notes: llvm-readelf --notes of the same code object (optional; checks register count, no spills, wave32, CU mode).

For every kernel k_ref_r<R> and k_k<K>_d<D>_r<R>[_f] the check reads only the listing (it does not trust the generator):
  1. ADDRESS ARITHMETIC: between two consecutive matrix instructions (v_wmma_f32_16x16x16_fp8_fp8) there is nothing but
     global_load_b64 and s_wait_loadcnt; at group starts exactly [global_load_b64, s_wait_loadcnt D], elsewhere nothing.
     The one place with other instructions is the loop edge (once per iteration): v_mov_b32, v_add_nc_u32, v_and_b32,
     s_add_co_i32, s_cmp_lg_u32, s_cbranch_scc1, in that order and no others (the three VALU there are reported).
  2. STRUCTURE: NMM matrix instructions per iteration (256; 512 for K = 32), NMM / K load groups, every wait counts D, the A
     registers of the matrix instructions and the destination registers of the loads follow the ring (smallest divisor of the
     group count that is at least D + 2), D loads before the loop, the loop closes on its first load.
  3. FOOTPRINT: from the immediates in the listing, per memory level the block can be asked for (L0 8 KiB, L2 1 MiB, MALL 32 MiB,
     GDDR 1 GiB; the host picks the fixed block when the window is not smaller than the span): a walking block reads exactly the
     slots 0 .. GN-1 once per iteration (A for the iteration's own groups, B with the next iteration's offsets), its window
     GN x 256 equals the constant the loop edge adds, and the window divides the span: footprint = span. A fixed block reads
     (g x 256) mod 8 KiB: 32 distinct slots, footprint 8 KiB, every offset below 8 KiB.
  4. NOTES (with --notes): vgpr_count = R, no spills, no scratch, wavefront 32, workgroup_processor_mode 0 (CU mode),
     max_flat_workgroup_size >= the group size.
Exit 1 on any failure. `selftest` runs the check on the real listing (must pass) and on deliberately corrupted copies written
to the output folder (every one must fail): an address instruction between two matrix instructions, one added to the
reference, a block whose offsets are doubled (span doubled), a fixed block with an offset at 8 KiB, a wait count changed, a
matrix instruction removed, a load added to the reference.
"""
import re
import sys

sys.dont_write_bytecode = True

SPAN = {"L0": 8192, "L2": 1 << 20, "MALL": 32 << 20, "GDDR": 1 << 30}
STEP = 256
NMM_REF = 256
RING0 = 26
TPB = {209: 448, 51: 1024}
EDGE = ["v_mov_b32", "v_add_nc_u32", "v_and_b32", "s_add_co_i32", "s_cmp_lg_u32", "s_cbranch_scc1"]
WMMA = "v_wmma_f32_16x16x16_fp8_fp8"
ALLOWED_GAP = {"global_load_b64", "s_wait_loadcnt"}
LOAD_RE = re.compile(r"^global_load_b64 v\[(\d+):(\d+)\], (v\d+), (s\[\d+:\d+\])(?: offset:(\d+))?$")


def nmm_of(k):
    return 512 if k == 32 else 256


def ring_of(gn, d):
    for r in range(d + 2, gn + 1):
        if gn % r == 0:
            return r
    return None


def mnem(t):
    m = t.split()[0]
    return re.sub(r"_e(32|64)$", "", m)


def norm(t):
    t = re.sub(r"0x[0-9a-fA-F]+", lambda m: str(int(m.group(0), 16)), t)
    t = re.sub(r"_e(32|64)\b", "", t, count=1)
    return re.sub(r"\s+", " ", t.replace(" :: ", " ")).strip()


def read_dis(path):
    """kernel name (k_...) -> list of (address, normalized text)."""
    out, cur = {}, None
    for ln in open(path, errors="replace"):
        m = re.match(r"^[0-9a-f]+ <(\S+)>:", ln)
        if m:
            cur = None
            s = re.search(r"_GLOBAL__N_1(\d+)(k_\w+)", m.group(1))
            if s:
                cur = s.group(2)[: int(s.group(1))]
                out[cur] = []
            continue
        if cur is not None and ln.startswith("\t"):
            body, _, cm = ln.partition("//")
            body = body.strip()
            a = re.match(r"\s*([0-9A-Fa-f]+):", cm)
            if body:
                out[cur].append((int(a.group(1), 16) if a else -1, norm(body)))
    return out


def read_notes(path):
    kernels, cur = {}, None
    for ln in open(path, errors="replace"):
        if re.match(r"^  - \.", ln):
            cur = {}
        m = re.match(r"^\s{4}(\.\w+):\s*(.*)$", ln) if cur is not None else None
        if m:
            cur[m.group(1)] = m.group(2).strip()
            if m.group(1) == ".name":
                s = re.search(r"_GLOBAL__N_1(\d+)(k_\w+)", m.group(2))
                if s:
                    kernels[s.group(2)[: int(s.group(1))]] = cur
    return kernels


def parse_name(name):
    m = re.match(r"^k_ref_r(\d+)$", name)
    if m:
        return {"ref": True, "K": 1, "D": 0, "R": int(m.group(1)), "fixed": False}
    m = re.match(r"^k_k(\d+)_d(\d+)_r(\d+)(_f)?$", name)
    if m:
        return {"ref": False, "K": int(m.group(1)), "D": int(m.group(2)), "R": int(m.group(3)), "fixed": bool(m.group(4))}
    return None


def check_kernel(name, ins):
    """returns (errors, infos)"""
    err, info = [], []
    p = parse_name(name)
    if p is None:
        return ["unknown kernel name"], info
    K, D, R, ref, fixed = p["K"], p["D"], p["R"], p["ref"], p["fixed"]
    nmm = nmm_of(K) if not ref else NMM_REF
    gn = nmm // K
    kr = 2 if ref else ring_of(gn, D)
    if kr is None or RING0 + 2 * kr > R:
        return ["ring does not fit"], info
    mn = [mnem(t) for _, t in ins]
    wi = [i for i, m in enumerate(mn) if m == WMMA]
    if not wi:
        return ["no matrix instructions"], info
    # the loop: the backward branch after the last matrix instruction
    b = next((i for i in range(wi[-1] + 1, len(mn)) if mn[i] == "s_cbranch_scc1"), None)
    if b is None:
        return ["no loop branch after the last matrix instruction"], info
    off = int(ins[b][1].split()[-1])
    off = off - 65536 if off >= 32768 else off
    target = ins[b][0] + 4 + off * 4
    ls = next((i for i in range(len(ins)) if ins[i][0] == target), None)
    if ls is None or ls > wi[0]:
        return ["loop target not found before the first matrix instruction"], info
    if len(wi) != nmm:
        err.append(f"{len(wi)} matrix instructions in the loop, want {nmm}")
        return err, info
    # 1. address arithmetic between matrix instructions, and the group pattern
    viol = []
    for n in range(nmm):
        lo = ls if n == 0 else wi[n - 1] + 1
        gap = [mn[i] for i in range(lo, wi[n])]
        want = [] if (ref or n % K) else ["global_load_b64", "s_wait_loadcnt"]
        if gap != want:
            viol.append((n, gap))
    if viol:
        n, gap = viol[0]
        bad = [g for g in gap if g not in ALLOWED_GAP]
        err.append(f"before matrix instruction {n}: {gap}, want {[] if (ref or n % K) else ['global_load_b64', 's_wait_loadcnt']}"
                   + (f" ADDRESS ARITHMETIC {bad}" if bad else "") + f" ({len(viol)} places)")
    edge = mn[wi[-1] + 1 : b + 1]
    if edge != EDGE:
        err.append(f"loop edge {edge}, want {EDGE}")
    valu = sum(1 for m in mn[ls : b + 1] if m.startswith("v_") and m != WMMA)
    info.append(f"VALU in the loop apart from the matrix instructions: {valu} (all at the edge)")
    if valu != 3:
        err.append(f"{valu} VALU instructions in the loop, want the 3 of the edge")
    # the edge registers and constant
    try:
        mv, ad, an = ins[wi[-1] + 1][1], ins[wi[-1] + 2][1], ins[wi[-1] + 3][1]
        ma = re.match(r"^v_mov_b32 (v\d+), (v\d+)$", mv)
        mb = re.match(r"^v_add_nc_u32 (v\d+), (\d+), (v\d+)$", ad)
        mc = re.match(r"^v_and_b32 (v\d+), (s\d+), (v\d+)$", an)
        A, B, win = ma.group(1), ma.group(2), int(mb.group(2))
        if not (mb.group(1) == B == mb.group(3) == mc.group(1) == mc.group(3)):
            err.append("edge: A <- B, B <- (B + W) & mask not on one register B")
    except (AttributeError, IndexError):
        return err + ["edge instructions unreadable"], info
    # 2. loads and waits: ring, offsets, counts
    loads = []
    for i in range(ls, wi[-1]):
        if mn[i] == "global_load_b64":
            m = LOAD_RE.match(ins[i][1])
            if not m:
                err.append(f"load not in the expected form: {ins[i][1]}")
                continue
            loads.append((int(m.group(1)), m.group(3), int(m.group(5) or 0), i))
        if mn[i] == "s_wait_loadcnt" and int(ins[i][1].split()[-1]) != D:
            err.append(f"wait count {ins[i][1].split()[-1]}, want {D}")
    pre = [i for i in range(0, ls) if mn[i] == "global_load_b64"]
    if ref:
        if loads or pre or any(m == "s_wait_loadcnt" for m in mn[ls:b]):
            err.append("the reference has loads or waits")
    else:
        if len(loads) != gn:
            err.append(f"{len(loads)} loads per iteration, want {gn}")
        if len(pre) != D:
            err.append(f"{len(pre)} loads before the loop, want {D}")
        exp = []
        for j in range(gn):
            g = j + D
            reg = RING0 + 2 * (g % kr)
            if fixed:
                exp.append((reg, A, (g * STEP) % SPAN["L0"]))
            elif g < gn:
                exp.append((reg, A, g * STEP))
            else:
                exp.append((reg, B, (g - gn) * STEP))
        got = [(l[0], l[1], l[2]) for l in loads]
        if got != exp:
            first = next((i for i in range(min(len(got), len(exp))) if got[i] != exp[i]), None)
            err.append(f"loads differ from the stream at load {first}: got {got[first] if first is not None else '-'}, want {exp[first] if first is not None else '-'}")
        for g, i in enumerate(pre):
            m = LOAD_RE.match(ins[i][1])
            want = (RING0 + 2 * (g % kr), A, (g * STEP) % SPAN["L0"] if fixed else g * STEP)
            if not m or (int(m.group(1)), m.group(3), int(m.group(5) or 0)) != want:
                err.append(f"preamble load {g}: {ins[i][1]}, want reg {want}")
                break
    # matrix A registers follow the ring
    for n in range(nmm):
        m = re.match(r"^v_wmma_f32_16x16x16_fp8_fp8 v\[16:23\], v\[(\d+):(\d+)\], v\[24:25\], v\[16:23\]$", ins[wi[n]][1])
        a = RING0 + 2 * ((n // K) % kr)
        if not m or (int(m.group(1)), int(m.group(2))) != (a, a + 1):
            err.append(f"matrix instruction {n}: {ins[wi[n]][1]}, want A v[{a}:{a + 1}]")
            break
    # 3. footprint
    if not ref:
        offs_a = [l[2] for l in loads if l[1] == A]
        offs_b = [l[2] for l in loads if l[1] == B]
        if fixed:
            slots = set(offs_a)
            ok = (not offs_b and all(o % STEP == 0 and o < SPAN["L0"] for o in offs_a) and len(slots) * STEP == SPAN["L0"])
            for lv, sp in SPAN.items():
                if win >= sp:  # the host uses the fixed block when the window is not smaller than the span
                    if not ok or len(slots) * STEP != sp:
                        err.append(f"FOOTPRINT {lv}: fixed block touches {len(slots) * STEP} B (offsets {sorted(slots)[:2]}..{max(offs_a)}), span {sp}")
                    else:
                        info.append(f"FOOTPRINT {lv} {sp} B == span (fixed block, {len(slots)} slots)")
        else:
            slots = sorted([o // STEP for o in offs_a] + [o // STEP for o in offs_b])
            ok = slots == list(range(gn)) and all(o % STEP == 0 for o in offs_a + offs_b)
            if not ok:
                err.append(f"FOOTPRINT: slots read per iteration are not 0..{gn - 1} once each (max offset {max(offs_a + offs_b)} B)")
            if win != gn * STEP:
                err.append(f"FOOTPRINT: the loop edge adds {win} B but the window is {gn * STEP} B")
            for lv, sp in SPAN.items():
                if sp > win:  # the host uses the walking block when the span is larger than the window
                    if ok and win == gn * STEP and sp % win == 0:
                        info.append(f"FOOTPRINT {lv} {sp} B == span (window {win} B steps through it)")
                    else:
                        err.append(f"FOOTPRINT {lv}: window {win} B does not tile the span {sp}")
    return err, info


def check_notes(notes, names):
    err = []
    for n in names:
        p = parse_name(n)
        k = notes.get(n)
        if k is None:
            err.append(f"{n}: not in the notes")
            continue
        want = {".vgpr_count": str(p["R"]), ".vgpr_spill_count": "0", ".sgpr_spill_count": "0", ".private_segment_fixed_size": "0",
                ".wavefront_size": "32", ".workgroup_processor_mode": "0"}
        for key, v in want.items():
            if k.get(key) != v:
                err.append(f"{n}: {key} {k.get(key)}, want {v}")
        if int(k.get(".max_flat_workgroup_size", "0")) < TPB[p["R"]]:
            err.append(f"{n}: max_flat_workgroup_size {k.get('.max_flat_workgroup_size')} below {TPB[p['R']]}")
    return err


def run_check(dis_path, notes_path, quiet=False):
    kernels = {k: v for k, v in read_dis(dis_path).items() if parse_name(k)}
    if not kernels:
        print("FAIL no ld_mat2 kernels in the listing")
        return False
    ok = True
    for name in sorted(kernels):
        err, info = check_kernel(name, kernels[name])
        if err:
            ok = False
        if not quiet or err:
            print(("FAIL " if err else "PASS ") + name + (": " + "; ".join(err) if err else ""))
            if not quiet and not err:
                for i in info:
                    print("     " + i)
    if notes_path:
        ne = check_notes(read_notes(notes_path), kernels)
        for e in ne:
            print("FAIL notes " + e)
        ok = ok and not ne
        if not ne and not quiet:
            print(f"PASS notes: {len(kernels)} kernels: vgpr count = R, no spills, no scratch, wave32, CU mode, group size fits")
    print(f"{'PASS' if ok else 'FAIL'}: {len(kernels)} kernels checked")
    return ok


def selftest(dis_path, outdir, notes_path):
    import os
    os.makedirs(outdir, exist_ok=True)
    text = open(dis_path, errors="replace").read().split("\n")
    names = [k for k in read_dis(dis_path) if parse_name(k)]

    def region(kernel):
        s = e = None
        for i, ln in enumerate(text):
            m = re.match(r"^[0-9a-f]+ <(\S+)>:", ln)
            if m:
                sm = re.search(r"_GLOBAL__N_1(\d+)(k_\w+)", m.group(1))
                nm = sm.group(2)[: int(sm.group(1))] if sm else None
                if nm == kernel:
                    s = i
                elif s is not None:
                    e = i
                    break
        return s, e if e is not None else len(text)

    def find(kernel, pat, nth=0):
        s, e = region(kernel)
        c = 0
        for i in range(s, e):
            if re.search(pat, text[i]):
                if c == nth:
                    return i
                c += 1
        raise SystemExit(f"selftest: no {pat} in {kernel}")

    fake = lambda m: f"\t{m}                                          // 000000000000: 00000000"
    breaks = []
    b = list(text)

    def make(tag, edit):
        t = list(text)
        edit(t)
        path = f"{outdir}/break-{len(breaks) + 1}-{tag}.s"
        open(path, "w").write("\n".join(t))
        breaks.append((tag, path))

    def ins_between(t, kernel, instr):
        i = find(kernel, WMMA, 5)
        t.insert(i + 1, fake(instr))

    make("v_add_between_matrix_instructions", lambda t: ins_between(t, "k_k8_d2_r209", "v_add_nc_u32_e32 v5, 1, v5"))
    make("v_and_in_reference", lambda t: ins_between(t, "k_ref_r209", "v_and_b32_e32 v5, v5, v5"))

    def double_offsets(t):
        s, e = region("k_k8_d2_r209")
        for i in range(s, e):
            t[i] = re.sub(r"offset:(\d+)", lambda m: f"offset:{int(m.group(1)) * 2}", t[i])

    make("one_block_with_the_span_doubled", double_offsets)

    def fixed_8k(t):
        i = find("k_k1_d2_r209_f", r"offset:7936")
        t[i] = t[i].replace("offset:7936", "offset:8192")

    make("fixed_block_offset_at_8KiB", fixed_8k)

    def wait_count(t):
        i = find("k_k4_d6_r209", r"s_wait_loadcnt 0x6", 3)
        t[i] = t[i].replace("s_wait_loadcnt 0x6", "s_wait_loadcnt 0x7")

    make("wait_count_changed", wait_count)

    def drop_wmma(t):
        i = find("k_k1_d4_r51", WMMA, 9)
        del t[i]

    make("matrix_instruction_removed", drop_wmma)

    def load_in_ref(t):
        i = find("k_ref_r51", WMMA, 3)
        t.insert(i + 1, fake("global_load_b64 v[26:27], v0, s[14:15] offset:256"))

    make("load_added_to_the_reference", load_in_ref)

    print("## the real listing (must pass)")
    good = run_check(dis_path, notes_path, quiet=True)
    allfail = True
    print("## deliberate breaks (every one must FAIL)")
    for tag, path in breaks:
        r = run_check(path, None, quiet=True)
        print(f"{'ok   ' if not r else 'NOT CAUGHT '} break {tag}: the check {'FAILED as it must' if not r else 'PASSED (it must fail)'}  [{path}]")
        allfail = allfail and not r
    print(f"{'PASS' if good and allfail else 'FAIL'}: self-test ({len(breaks)} breaks)")
    return good and allfail


def main(argv):
    notes = None
    if "--notes" in argv:
        i = argv.index("--notes")
        notes = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if len(argv) >= 3 and argv[1] == "check":
        return 0 if run_check(argv[2], notes) else 1
    if len(argv) >= 4 and argv[1] == "selftest":
        return 0 if selftest(argv[2], argv[3], notes) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
