#!/usr/bin/env python3
"""Round 6's own copy (the model folder comes from LLM_MODELS). analyse.py RUN_DIR STATE -> markdown tables on stdout (ctr.sh writes them to table-<STATE>.md).

Per workload the run has two jobs, N=16 and N=32 generated tokens, each with one rocprofv3 pass per counter group. Per counter the
values of every dispatch are summed per kernel; the per-token number is (sum at N=32 - sum at N=16) / DIV, so the prompt,
start-up and one-off kernels cancel. DIV is 16 for llama.cpp (-n 16 / 32) and 32 for vLLM (the trace task decodes the N tokens twice:
a warm-up pass and a second pass). A counter 'scales' when its sum grows from N=16 to N=32.
Bytes = FETCH_SIZE (KiB) x 1024. Expected bytes: weights of the matrix tensors the decode matrix kernels read (GGUF or safetensors
header) and the KV cache read at the mean context of the counted tokens (K and V, f16/bf16, no padding).
"""
import csv, glob, json, os, re, struct, sys, collections

run, state = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(run)), "..", "scripts"))
sys.path.insert(0, os.path.abspath(os.path.join(run, "..", "..", "scripts")))
M = os.environ["LLM_MODELS"]  # the model folder (GGUF files and the two vLLM model folders)
CTRS = ["FETCH_SIZE", "TCP_REQ", "TCP_REQ_MISS", "SQ_INSTS_VALU", "SQ_WAVE32_INSTS", "SQ_WAVES", "GRBM_GUI_ACTIVE", "GRBM_COUNT"]
# workload: (label, div, model path, kind, KV bytes per element, prompt tokens)
WORK = [("ll-q8", "llama.cpp 3B Q8_0", 16, "llama32-3b-q8_0.gguf"),
        ("ll-fp8", "llama.cpp 3B FP8 (patched)", 16, "llama32-3b-fp8-outq8.gguf"),
        ("ll-f16", "llama.cpp 3B auto (FP8 layers + bf16 lm head; NOT 16-bit, see README)", 16, "llama32-3b-auto.gguf"),
        ("vl-fp8", "vLLM 3B FP8 (eager)", 32, "llama32-3b-redhat-fp8"),
        ("vl-bf16", "vLLM 3B bf16 (eager)", 32, "llama32-3b-instruct-16bit")]
FAM = [("matvec", re.compile(r"mul_mat_vec|wvSplitK|LLGemm|gemv|Cijk_|gemm|sgemm|hipblas|rocblas|Tensile|skinny", re.I)),
       ("attention", re.compile(r"flash_attn|paged_attention|attention|unified_attn|attn", re.I))]
LAYER_W = re.compile(r"blk\.\d+\.ffn_(gate|up|down)\.scale$|blk\.\d+\.(attn_q|attn_k|attn_v|attn_output)\.scale$|(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$|(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(weight|weight_scale)$")
NL, NKV, HD = 28, 8, 128   # Llama 3.2 3B (config.json of both vLLM models)


def family(name):
    for f, rx in FAM:
        if rx.search(name):
            return f
    return "other"


def load(job):
    """{kernel: {counter: sum}}, {kernel: dispatches (rows of GRBM_COUNT)}"""
    s = collections.defaultdict(lambda: collections.defaultdict(float)); n = collections.Counter()
    for f in glob.glob(f"{run}/out/{state}/{job}/pass_*/p_counter_collection.csv"):
        for r in csv.DictReader(open(f, newline="")):
            s[r["Kernel_Name"]][r["Counter_Name"]] += float(r["Counter_Value"])
            if r["Counter_Name"] == "GRBM_COUNT":
                n[r["Kernel_Name"]] += 1
    return s, n


def gguf_expected(path):
    import gguf_rows
    f, meta, tensors, ds = gguf_rows.read_gguf(path)
    size = os.path.getsize(path) - ds
    order = sorted(tensors, key=lambda t: t["offset"])
    by = {}
    for a, b in zip(order, order[1:] + [{"offset": size}]):
        by[a["name"]] = b["offset"] - a["offset"]
    w = sum(v for k, v in by.items() if LAYER_W.search(k))
    head = by.get("output.weight", by.get("token_embd.weight"))
    types = collections.Counter(t["type"] for t in tensors if LAYER_W.search(t["name"]))
    return w + head, w, head, f"types of layer matrices {dict(types)}; lm head tensor {'output.weight' if 'output.weight' in by else 'token_embd.weight (tied)'}"


def st_expected(d):
    j = {}
    for fn in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = open(fn, "rb"); n = struct.unpack("<Q", h.read(8))[0]; part = json.loads(h.read(n)); part.pop("__metadata__", None); j.update(part)
    by = {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in j.items()}
    w = sum(v for k, v in by.items() if LAYER_W.search(k))
    hk = "lm_head.weight" if "lm_head.weight" in by else "model.embed_tokens.weight"
    dt = collections.Counter(v["dtype"] for k, v in j.items() if LAYER_W.search(k))
    return w + by[hk], w, by[hk], f"layer tensor dtypes {dict(dt)}; lm head tensor {hk}"


def kv_bytes(ctx, elt=2):
    return 2 * NL * NKV * HD * ctx * elt


def fmt(x, d=0):
    return "-" if x is None else (f"{x:,.{d}f}")


print(f"State: {state}. Per generated token, from the difference of the N=32 and N=16 runs (see the top of analyse.py). Dispatch counts are per token too.\n")
for key, title, div, mdl in WORK:
    s16, n16 = load(f"{key}-n16"); s32, n32 = load(f"{key}-n32")
    print(f"### {title}\n")
    if not s16 or not s32:
        print(f"NO DATA: out/{state}/{key}-n16 or -n32 is missing or has no counter rows (the job did not run or rocprofv3 could not attach).\n")
        continue
    # family and per-kernel tables
    fam = collections.defaultdict(lambda: collections.defaultdict(float)); famn = collections.Counter(); fam16 = collections.defaultdict(lambda: collections.defaultdict(float))
    ker = collections.defaultdict(lambda: collections.defaultdict(float)); kn = collections.Counter()
    for k in set(s16) | set(s32):
        f = family(k)
        for c in CTRS:
            d = (s32[k].get(c, 0.0) - s16[k].get(c, 0.0)) / div if k in s32 else 0.0
            fam[f][c] += d; ker[k][c] += d; fam16[f][c] += s16[k].get(c, 0.0) if k in s16 else 0.0
        dn = (n32.get(k, 0) - n16.get(k, 0)) / div
        famn[f] += dn; kn[k] += dn
    allzero = all(fam[f]["FETCH_SIZE"] == 0 and fam[f]["SQ_INSTS_VALU"] == 0 for f in fam)
    if allzero:
        print("**All memory-side and instruction counters read 0 in this state (FETCH_SIZE, TCP_REQ, TCP_REQ_MISS, SQ_INSTS_VALU, SQ_WAVE32_INSTS);** only GRBM and SQ_WAVES count. Expected in `auto`.\n")
    print("| family | dispatches/token | FETCH_SIZE bytes/token | TCP_REQ/token | TCP_REQ_MISS/token | miss share | SQ_INSTS_VALU/token | SQ_WAVE32_INSTS/token | waves/token | GRBM_GUI_ACTIVE cycles/token | scales (FETCH N=16 -> N=32 total) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in ("matvec", "attention", "other"):
        v = fam[f]; req = v["TCP_REQ"]
        ms = f"{100 * v['TCP_REQ_MISS'] / req:.1f}%" if req else "-"
        t16 = fam16[f]["FETCH_SIZE"]; t32 = t16 + v["FETCH_SIZE"] * div
        sc = ("yes" if t32 > t16 > 0 else "no") + f" ({t16 * 1024:.3g} -> {t32 * 1024:.3g})"
        print(f"| {f} | {fmt(famn[f], 1)} | {fmt(v['FETCH_SIZE'] * 1024)} | {fmt(req)} | {fmt(v['TCP_REQ_MISS'])} | {ms} | {fmt(v['SQ_INSTS_VALU'])} | {fmt(v['SQ_WAVE32_INSTS'])} | {fmt(v['SQ_WAVES'])} | {fmt(v['GRBM_GUI_ACTIVE'])} | {sc} |")
    # expected bytes
    path = os.path.join(M, mdl)
    try:
        tot, w, head, note = gguf_expected(path) if mdl.endswith(".gguf") else st_expected(path)
    except Exception as e:
        tot = None; note = f"expected bytes unavailable: {e}"
    ctx = 128 + (16 + 32 + 1) / 2 if div == 16 else 128 + (16 + 32 + 1) / 2   # prompt 128 + mean position of tokens 17..32
    kv = kv_bytes(ctx, 2)
    mv, at = fam["matvec"]["FETCH_SIZE"] * 1024, fam["attention"]["FETCH_SIZE"] * 1024
    print(f"\nExpected: weights read by the matrix kernels {fmt(tot)} bytes ({note}); KV cache at mean context {ctx:.1f}: {fmt(kv)} bytes (f16/bf16, K and V, 28 layers x 8 heads x 128).")
    if tot and mv:
        print(f"Matrix kernels FETCH_SIZE / weight bytes = **{mv / tot:.3f}**. Attention FETCH_SIZE / KV bytes = {at / kv:.3f}. (matvec+attention) / (weights+KV) = {(mv + at) / (tot + kv):.3f}.")
    elif tot:
        print("Ratio not computed: matrix-kernel FETCH_SIZE is 0 or missing.")
    print("\nPer kernel (top 8 by GRBM_GUI_ACTIVE):\n")
    print("| kernel | family | dispatches/token | FETCH_SIZE bytes/token | TCP_REQ/token | miss share | SQ_INSTS_VALU/token | waves/token | GRBM_GUI_ACTIVE/token |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(ker.items(), key=lambda x: -x[1]["GRBM_GUI_ACTIVE"])[:8]:
        ms = f"{100 * v['TCP_REQ_MISS'] / v['TCP_REQ']:.1f}%" if v["TCP_REQ"] else "-"
        print(f"| `{re.sub(r'[|]', '/', k)[:78]}` | {family(k)} | {fmt(kn[k], 1)} | {fmt(v['FETCH_SIZE'] * 1024)} | {fmt(v['TCP_REQ'])} | {ms} | {fmt(v['SQ_INSTS_VALU'])} | {fmt(v['SQ_WAVES'])} | {fmt(v['GRBM_GUI_ACTIVE'])} |")
    print()
