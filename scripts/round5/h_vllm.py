"""Round 5 item H: vLLM 3B FP8, one warmed engine, decode at 32 against 128 generated tokens.
Interleaves requests of different prompt length and generated length (one stream), times (a) whole generate() calls (the
round-4 method: time of N tokens minus time of 1 token), (b) every engine step on the host, with the KV length of each step.
Usage: h_vllm.py --model DIR --text FILE [--rounds 8] [--out JSON]"""
import argparse, json, os, statistics, sys, time, random
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--text"); ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--out", default="h_vllm.json"); ap.add_argument("--gpu-util", type=float, default=0.5)
ap.add_argument("--inject-ms", type=float, default=5.0)
a = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(open(a.text, encoding="utf-8").read(), add_special_tokens=True)["input_ids"]
llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False, seed=0,
          max_num_seqs=16, max_num_batched_tokens=4096)
eng = llm.llm_engine

def sp(n):
    return SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)

def prompt(P, off):
    return ids[1 + off:1 + off + P]

def gen_time(P, n, off=0):
    t = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": prompt(P, off)}], sp(n), use_tqdm=False)
    dt = time.perf_counter() - t
    assert len(out[0].outputs[0].token_ids) == n, (n, len(out[0].outputs[0].token_ids))
    return dt

_rid = [0]
def step_trace(P, n, off=0, inject_ms=0.0):
    """Per-step host times of one request: [first step (prefill+1st token), then n-1 decode steps]. Also the total."""
    _rid[0] += 1
    rid = f"h{_rid[0]}"
    eng.add_request(rid, {"prompt_token_ids": prompt(P, off)}, sp(n))
    steps = []; ntok = 0; extra = 0.0
    t_all = time.perf_counter(); t_prev = t_all
    while eng.has_unfinished_requests():
        outs = eng.step()
        if inject_ms:
            time.sleep(inject_ms / 1000)
        t = time.perf_counter()
        prev = ntok
        for o in outs:
            if o.request_id == rid:
                ntok = len(o.outputs[0].token_ids)
        if ntok > prev:
            steps.append(t - t_prev)       # a step that produced a token
        else:
            extra += t - t_prev            # a step with no new token (finishing): not a decode step, kept in the total
        t_prev = t
    assert ntok == n, (ntok, n)
    assert len(steps) == n, ("token-producing steps", len(steps), "tokens", n)
    return steps, time.perf_counter() - t_all

# warm-up: both lengths, several times (graph capture, kernel load)
for _ in range(3):
    gen_time(128, 128); gen_time(224, 32); step_trace(128, 32)

kinds = {  # name: (prompt length, new tokens)
    "A": (128, 32),    # round 4's short protocol; KV 129..160
    "B": (128, 128),   # round 4's long protocol; KV 129..256
    "C": (224, 32),    # 32 new tokens whose KV range (225..256) overlaps B's last 32 steps
    "D": (256, 32),    # KV 257..288, beyond B's range
}
res = {"gen_one": {}, "gen_n": {}, "steps": {}, "order": []}
order_base = list(kinds)
for r in range(a.rounds):
    ordr = order_base[r % 4:] + order_base[:r % 4]
    random.Random(r).shuffle(ordr) if r >= 4 else None
    res["order"].append(ordr)
    for k in ordr:
        P, n = kinds[k]
        off = 7 * r
        t1 = gen_time(P, 1, off)
        tn = gen_time(P, n, off)
        st, tot = step_trace(P, n, off)
        res["gen_one"].setdefault(k, []).append(t1)
        res["gen_n"].setdefault(k, []).append(tn)
        res["steps"].setdefault(k, []).append({"first": st[0], "decode": st[1:], "total": tot})
# instrument check: 5 ms injected inside the timed step loop must show up as +5 ms per step
inj = [step_trace(128, 32, 0, a.inject_ms) for _ in range(3)]
base = [step_trace(128, 32, 0, 0.0) for _ in range(3)]
d_inj = statistics.median(statistics.mean(s[1:]) for s, _ in inj); d_base = statistics.median(statistics.mean(s[1:]) for s, _ in base)
res["inject"] = {"ms": a.inject_ms, "decode_step_ms_with": d_inj * 1e3, "decode_step_ms_without": d_base * 1e3, "delta_ms": (d_inj - d_base) * 1e3}
json.dump(res, open(a.out, "w"))
print("RESULT written", a.out)
for k, (P, n) in kinds.items():
    t1 = statistics.median(res["gen_one"][k]); tn = statistics.median(res["gen_n"][k])
    dec = [statistics.mean(s["decode"]) for s in res["steps"][k]]
    print(f"kind {k} P={P} N={n}: generate-subtraction {n / (tn - t1):.1f} tok/s (t1 {t1*1e3:.1f} ms, tN {tn*1e3:.1f} ms); "
          f"host decode step {statistics.median(dec)*1e3:.3f} ms = {1/statistics.median(dec):.1f} tok/s; first step {statistics.median(s['first'] for s in res['steps'][k])*1e3:.1f} ms")
print("inject check:", res["inject"])
