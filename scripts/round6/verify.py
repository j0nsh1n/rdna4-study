# Independent recheck of table-profile_standard.md: plain sums per job, no kernel-family code shared with analyse.py.
import csv, glob, collections, re, sys
S = "out/profile_standard"
def tot(job, pat, ctr):
    t = 0.0; rows = collections.Counter()
    for f in glob.glob(f"{S}/{job}/pass_*/p_counter_collection.csv"):
        for r in csv.DictReader(open(f)):
            if r["Counter_Name"] == "GRBM_COUNT": rows[f.split('/')[-2]] += 1
            if r["Counter_Name"] == ctr and re.search(pat, r["Kernel_Name"]): t += float(r["Counter_Value"])
    return t, dict(rows)
checks = [("ll-q8", "mul_mat_vec", 16), ("ll-fp8", "mul_mat_vec", 16), ("ll-f16", "mul_mat_vec", 16), ("vl-fp8", "wvSplitK", 32), ("vl-bf16", "wvSplitK", 32)]
for w, pat, div in checks:
    out = []
    for c in ("FETCH_SIZE", "SQ_INSTS_VALU", "SQ_WAVE32_INSTS", "TCP_REQ", "TCP_REQ_MISS"):
        a, ra = tot(f"{w}-n16", pat, c); b, rb = tot(f"{w}-n32", pat, c)
        out.append(f"{c}={(b - a) / div * (1024 if c == 'FETCH_SIZE' else 1):.6g}")
    print(w, " ".join(out), "| GRBM_COUNT rows/pass n16", sorted(set(ra.values())), "n32", sorted(set(rb.values())), "passes", len(ra), len(rb))
for w in ("ll-q8", "ll-fp8", "ll-f16", "vl-fp8", "vl-bf16"):  # attention bytes per token and tile kernel alone
    pat = "flash_attn_tile" if w.startswith("ll") else "paged_attention_ll4mi_QKV"
    a, _ = tot(f"{w}-n16", pat, "FETCH_SIZE"); b, _ = tot(f"{w}-n32", pat, "FETCH_SIZE")
    print(w, "attention main kernel FETCH bytes/token", f"{(b - a) / (16 if w.startswith('ll') else 32) * 1024:.6g}")
