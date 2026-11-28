#!/usr/bin/env python3
"""Round 13 (P1): Static ISA instruction classifier for RDNA 4 (gfx1201) ACO disassembly.

Counts STATIC instructions (not dynamic/runtime execution counts) into 15 mutually
exclusive classes (plus 'other' for anything unrecognized, expected to be 0):
  1. packed_fp16_valu   : v_pk_*_f16 (packed dual-FP16 VALU)
  2. scalar_fp16_valu   : scalar-lane FP16 VALU (v_*_f16 / v_fma_mix{lo,hi}_f16)
  3. fp32_valu          : FP32/FP64 VALU (v_*_f32, v_dual_*_f32, v_fma_mix_f32)
  4. int_valu           : integer / bitwise / move / lane / compare VALU (incl. v_pk_*_{i,u}16)
  5. msad_sad           : v_msad_u8 / v_sad_* / v_qsad_* / v_mqsad_*
  6. dot                : v_dot* / v_dual_dot2acc*
  7. wmma               : v_wmma_* / v_swmmac_*
  8. conversions        : v_cvt_* / v_pk_cvt_* / v_pack_b32_f16
  9. image_sample       : image_sample*
 10. image_gather4      : image_gather4*
 11. image_load_store   : image_load* / image_store* / image_atomic_* / image_get_* / image_bvh*
 12. buffer_global_mem  : buffer_* / global_* / flat_* / scratch_*
 13. lds                : ds_*
 14. scalar             : s_* (SALU + SMEM, excluding waits/barriers/branches/delays)
 15. waits_branches     : s_wait* / s_barrier* / s_delay_alu / s_nop / s_clause / s_branch /
                          s_cbranch* / s_setpc* / s_swappc* / s_call* / s_endpgm / s_code_end

Usage:
  isa_classify.py --self-test
  isa_classify.py <asm_file> [--json]
  isa_classify.py --dir <asm_dir> [--json-out FILE]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

CLASSES = [
    "packed_fp16_valu",
    "scalar_fp16_valu",
    "fp32_valu",
    "int_valu",
    "msad_sad",
    "dot",
    "wmma",
    "conversions",
    "image_sample",
    "image_gather4",
    "image_load_store",
    "buffer_global_mem",
    "lds",
    "scalar",
    "waits_branches",
    "other",
]

# Strip encoding suffix (_e32, _e64, _dpp, _sdwa, _t16) when matching arithmetic suffix.
_ENC_SUFFIX = re.compile(r"(_e32|_e64|_dpp|_sdwa|_t16)+$")
_WAITS_BRANCHES_PREFIXES = (
    "s_wait",
    "s_barrier",
    "s_branch",
    "s_cbranch",
    "s_setpc",
    "s_swappc",
    "s_call",
)
_WAITS_BRANCHES_EXACT = {
    "s_delay_alu",
    "s_nop",
    "s_clause",
    "s_endpgm",
    "s_code_end",
    "s_sethalt",
    "s_sleep",
    "s_wakeup",
    "s_trap",
    "s_rfe_b64",
}

_LABEL_RE = re.compile(r"^[A-Za-z0-9_.$@]+:$")
_OP_RE = re.compile(r"^([a-z][a-z0-9_]*)(?:\s|$)")
_REPEAT_RE = re.compile(r"^\(then repeated (\d+) times\)$")


def strip_comment(line: str) -> str:
    # ACO comments start with ';', '//', or '#' (our header metadata).
    # Strip /* ... */ inline blocks first if present.
    s = re.sub(r"/\*.*?\*/", "", line)
    for tok in (";", "//", "#"):
        pos = s.find(tok)
        if pos != -1:
            s = s[:pos]
    return s.strip()


def classify_single_op(op: str) -> str:
    if not op:
        return "other"

    if op.startswith("ds_"):
        return "lds"

    if op.startswith("image_sample"):
        return "image_sample"
    if op.startswith("image_gather4"):
        return "image_gather4"
    if op.startswith("image_"):
        return "image_load_store"

    if op.startswith(("buffer_", "global_", "flat_", "scratch_")):
        return "buffer_global_mem"

    if op.startswith("s_"):
        if op in _WAITS_BRANCHES_EXACT or op.startswith(_WAITS_BRANCHES_PREFIXES):
            return "waits_branches"
        return "scalar"

    if op.startswith("v_"):
        if op.startswith(("v_wmma_", "v_swmmac_")):
            return "wmma"
        if op.startswith(("v_dot", "v_dual_dot")):
            return "dot"
        if op.startswith(("v_msad_", "v_sad_", "v_qsad_", "v_mqsad_")):
            return "msad_sad"
        if op.startswith(("v_cvt_", "v_pk_cvt_")) or op == "v_pack_b32_f16":
            return "conversions"

        base = _ENC_SUFFIX.sub("", op)

        if base.startswith("v_pk_"):
            if base.endswith("_f16"):
                return "packed_fp16_valu"
            return "int_valu"

        if base in ("v_fma_mixlo_f16", "v_fma_mixhi_f16"):
            return "scalar_fp16_valu"
        if base == "v_fma_mix_f32":
            return "fp32_valu"

        if base.endswith("_f16") or "_f16_" in base:
            return "scalar_fp16_valu"
        if base.endswith(("_f32", "_f64")) or "_f32_" in base or "_f64_" in base:
            return "fp32_valu"

        return "int_valu"

    return "other"


def classify_instruction_line(stripped: str):
    """Return (primary_class, op_list, is_vopd) for a stripped non-empty instruction line."""
    if "::" in stripped:
        left, right = [p.strip() for p in stripped.split("::", 1)]
        ml = _OP_RE.match(left)
        mr = _OP_RE.match(right)
        ops = []
        if ml:
            ops.append(ml.group(1))
        if mr:
            ops.append(mr.group(1))
        if len(ops) == 2:
            c0 = classify_single_op(ops[0])
            c1 = classify_single_op(ops[1])
            # A VOPD pair is 1 static machine instruction executing 2 VALU ops.
            # Assign line class by compute priority (dot > packed_fp16 > scalar_fp16 > fp32 > int),
            # while callers also get the exact sub-op pair breakdown via vopd_subops.
            prio = {
                "dot": 5,
                "packed_fp16_valu": 4,
                "scalar_fp16_valu": 3,
                "fp32_valu": 2,
                "int_valu": 1,
            }
            chosen = c0 if prio.get(c0, 0) >= prio.get(c1, 0) else c1
            return chosen, ops, True

    m = _OP_RE.match(stripped)
    if not m:
        return None, [], False
    op = m.group(1)
    return classify_single_op(op), [op], False


def classify_text(text: str) -> dict:
    counts = {k: 0 for k in CLASSES}
    vopd_lines = 0
    vopd_subops = collections.Counter()
    unknown_ops = collections.Counter()
    image_atomics = 0
    buffer_global_atomics = 0
    lds_atomics = 0
    top_ops = collections.Counter()

    in_meta = False
    last_item = None  # (cls, ops, is_vopd)
    for raw in text.splitlines():
        s = strip_comment(raw)
        if not s:
            continue
        m_rep = _REPEAT_RE.match(s)
        if m_rep and last_item is not None:
            rep = int(m_rep.group(1))
            cls, ops, is_vopd = last_item
            counts[cls] += rep
            if is_vopd:
                vopd_lines += rep
                for op in ops:
                    vopd_subops[classify_single_op(op)] += rep
                    top_ops[op] += rep
            else:
                op = ops[0]
                top_ops[op] += rep
                if cls == "other":
                    unknown_ops[op] += rep
                if op.startswith("image_atomic_"):
                    image_atomics += rep
                elif op.startswith(("buffer_atomic_", "global_atomic_", "flat_atomic_")):
                    buffer_global_atomics += rep
                elif op.startswith("ds_") and any(
                    k in op for k in ("_add_", "_sub_", "_inc_", "_dec_", "_min_", "_max_", "_and_", "_or_", "_xor_", "_cmpswap", "_swap")
                ):
                    lds_atomics += rep
            continue
        if s.startswith("."):
            if s.startswith(".amdgpu_metadata"):
                in_meta = True
            elif s.startswith(".end_amdgpu_metadata"):
                in_meta = False
            continue
        if in_meta:
            continue
        if _LABEL_RE.match(s):
            last_item = None
            continue
        first_tok = s.split()[0]
        if first_tok.endswith(":"):
            last_item = None
            continue

        cls, ops, is_vopd = classify_instruction_line(s)
        if cls is None:
            last_item = None
            continue
        last_item = (cls, ops, is_vopd)
        counts[cls] += 1
        if is_vopd:
            vopd_lines += 1
            for op in ops:
                vopd_subops[classify_single_op(op)] += 1
                top_ops[op] += 1
        else:
            op = ops[0]
            top_ops[op] += 1
            if cls == "other":
                unknown_ops[op] += 1
            if op.startswith("image_atomic_"):
                image_atomics += 1
            elif op.startswith(("buffer_atomic_", "global_atomic_", "flat_atomic_")):
                buffer_global_atomics += 1
            elif op.startswith("ds_") and any(
                k in op for k in ("_add_", "_sub_", "_inc_", "_dec_", "_min_", "_max_", "_and_", "_or_", "_xor_", "_cmpswap", "_swap")
            ):
                lds_atomics += 1

    total = sum(counts.values())
    valu_total = (
        counts["packed_fp16_valu"]
        + counts["scalar_fp16_valu"]
        + counts["fp32_valu"]
        + counts["int_valu"]
        + counts["msad_sad"]
        + counts["dot"]
        + counts["wmma"]
        + counts["conversions"]
    )
    fp16_total = counts["packed_fp16_valu"] + counts["scalar_fp16_valu"]
    shares = {k: (counts[k] / total if total else 0.0) for k in CLASSES}
    return {
        "static_instruction_count": total,
        "counts": counts,
        "shares": shares,
        "fp16_valu_total": fp16_total,
        "fp16_share_of_total": (fp16_total / total if total else 0.0),
        "fp16_share_of_valu": (fp16_total / valu_total if valu_total else 0.0),
        "valu_total": valu_total,
        "vopd_lines": vopd_lines,
        "vopd_subops": dict(vopd_subops),
        "image_atomics": image_atomics,
        "buffer_global_atomics": buffer_global_atomics,
        "lds_atomics": lds_atomics,
        "unknown_ops": dict(unknown_ops),
        "top_ops": top_ops.most_common(12),
    }


def run_self_test() -> int:
    snippet = """
    # Synthetic RDNA 4 (gfx1201) ACO disassembly covering all 15 instruction classes
    BB0:
        s_load_b128 s[4:7], s[0:1], 0x0
        s_wait_kmcnt 0x0
        v_pk_fma_f16 v0, v1, v2, v3
        v_pk_add_f16 v4, v0, v1
        v_add_f16_t16 v5.l, v0.l, v1.l
        v_rcp_f16_e32 v6.l, v5.l
        v_fma_mixlo_f16 v7, v0, v1, v2
        v_fmac_f32_e32 v8, v9, v10
        v_dual_fmac_f32 v11, v12, v13 :: v_dual_mov_b32 v14, 0
        v_add_nc_u32_e32 v15, v16, v17
        v_pk_add_i16 v18, v19, v20
        v_msad_u8 v21, v22, v23, v24
        v_sad_u8 v25, v26, v27, v28
        v_dot2_f32_f16 v29, v30, v31, v29
        v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[40:41], v[42:43], v[32:39]
        v_cvt_pk_rtz_f16_f32 v44, v8, v11
        v_pack_b32_f16 v45, v5.l, v6.l
        image_sample_lz v[46:49], v[0:1], s[4:11], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D
        image_gather4_lz v[50:53], v[0:1], s[4:11], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
        image_load v54, v[0:1], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
        image_store v54, v[0:1], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
        image_atomic_umax v55, v[0:1], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
        global_load_b128 v[56:59], v[60:61], off
        buffer_store_b32 v0, v1, s[4:7], 0 offen
        ds_load_b32 v62, v63
        ds_add_u32 v63, v0
        s_barrier_signal -1
        s_barrier_wait -1
        s_nop 0
        (then repeated 2 times)
        s_cbranch_scc0 BB0
        s_endpgm
    """
    expected = {
        "packed_fp16_valu": 2,
        "scalar_fp16_valu": 3,
        "fp32_valu": 2,
        "int_valu": 2,
        "msad_sad": 2,
        "dot": 1,
        "wmma": 1,
        "conversions": 2,
        "image_sample": 1,
        "image_gather4": 1,
        "image_load_store": 3,
        "buffer_global_mem": 2,
        "lds": 2,
        "scalar": 1,
        "waits_branches": 8,
        "other": 0,
    }
    res = classify_text(snippet)
    ok = True
    for k, exp in expected.items():
        got = res["counts"][k]
        if got != exp:
            print(f"SELF-TEST FAIL: class '{k}' expected {exp}, got {got}", file=sys.stderr)
            ok = False
    if res["static_instruction_count"] != sum(expected.values()):
        print(
            f"SELF-TEST FAIL: total {res['static_instruction_count']} != {sum(expected.values())}",
            file=sys.stderr,
        )
        ok = False
    if res["vopd_lines"] != 1 or res["vopd_subops"] != {"fp32_valu": 1, "int_valu": 1}:
        print(f"SELF-TEST FAIL: unexpected vopd metadata: {res['vopd_lines']} {res['vopd_subops']}", file=sys.stderr)
        ok = False
    if res["image_atomics"] != 1 or res["lds_atomics"] != 1:
        print(
            f"SELF-TEST FAIL: atomics mismatch img={res['image_atomics']} lds={res['lds_atomics']}",
            file=sys.stderr,
        )
        ok = False

    # Deliberate wrong-label break case: assert v_msad_u8 is mislabeled as packed_fp16_valu -> MUST FAIL.
    wrong_snippet = "    v_msad_u8 v0, v1, v2, v3\n"
    wrong_res = classify_text(wrong_snippet)
    wrong_expect_pass = wrong_res["counts"]["packed_fp16_valu"] == 1
    if wrong_expect_pass:
        print("SELF-TEST FAIL: deliberate wrong-label check unexpectedly passed!", file=sys.stderr)
        ok = False
    else:
        print("DELIBERATE-BREAK check (mislabeled v_msad_u8 as packed_fp16_valu): FAIL as expected (caught)")

    if ok:
        print(
            f"VERDICT isa_classify self-test: PASS ({res['static_instruction_count']} static instructions, "
            f"15/15 classes exact, other=0, deliberate-break caught)"
        )
        return 0
    print("VERDICT isa_classify self-test: FAIL", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify static RDNA 4 ACO disassembly instructions.")
    ap.add_argument("path", nargs="?", help="Single .asm file to classify")
    ap.add_argument("--self-test", action="store_true", help="Run built-in unit tests + deliberate break")
    ap.add_argument("--dir", help="Classify every *.asm file in directory")
    ap.add_argument("--json", action="store_true", help="Emit JSON for single file")
    ap.add_argument("--json-out", help="Write JSON map for --dir mode")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if args.dir:
        d = Path(args.dir)
        out = {}
        unknown_total = collections.Counter()
        for p in sorted(d.glob("*.asm")):
            r = classify_text(p.read_text(encoding="utf-8", errors="replace"))
            out[p.stem] = r
            for op, cnt in r["unknown_ops"].items():
                unknown_total[op] += cnt
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"Classified {len(out)} disassembly files (STATIC instruction counts); unknown_ops={dict(unknown_total)}")
        return 0

    if not args.path:
        ap.print_help(sys.stderr)
        return 2

    res = classify_text(Path(args.path).read_text(encoding="utf-8", errors="replace"))
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        n = res["static_instruction_count"]
        print(f"{args.path}: {n} STATIC instructions (not runtime execution count)")
        for k in CLASSES:
            c = res["counts"][k]
            if c or k == "other":
                print(f"  {k:20s} {c:5d}  ({100.0 * c / n if n else 0.0:5.1f}%)")
        print(
            f"  FP16 VALU total: {res['fp16_valu_total']} "
            f"({100.0 * res['fp16_share_of_total']:.1f}% of total, "
            f"{100.0 * res['fp16_share_of_valu']:.1f}% of VALU)"
        )
        if res["vopd_lines"]:
            print(f"  VOPD dual-issue lines: {res['vopd_lines']} (subops: {res['vopd_subops']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
