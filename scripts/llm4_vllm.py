#!/usr/bin/env python3
"""Round 4: the same models through vLLM (offline engine, in one process so the profiler sees the kernels).

Run with toolchains/vllm-rocm101-venv through the gated runner (scripts/study-llm.sh r4*).
  llm4_vllm.py <task> --model DIR [--max-len N] [options]
Tasks: bench (prefill at 128/512/2048, decode streams 1..8), ppl (perplexity and, with a llama.cpp logits file, the distance of the
target-token log-probability), trace (a marked region for rocprofv3), selftest (no GPU).
vLLM returns log-probabilities of the actual next token (prompt_logprobs=0), not full distributions, so quality here is perplexity and
the mean absolute difference of the target-token log-probability to the reference; there is no KL divergence for vLLM.
"""
import argparse, json, math, os, statistics, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

MARK_A, MARK_B = 1234567, 2345678


def load_ids(model_dir, text_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(open(text_path, encoding="utf-8").read(), add_special_tokens=True)["input_ids"]
    return ids, tok.bos_token_id


def make_llm(a):
    from vllm import LLM
    kw = dict(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False, seed=0,
              max_num_seqs=a.max_seqs, max_num_batched_tokens=a.batched_tokens)
    if a.eager:
        kw["enforce_eager"] = True
    if a.cudagraph_mode or a.compilation_mode:
        from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
        cc_kw = {}
        if a.compilation_mode:
            cc_kw["mode"] = getattr(CompilationMode, a.compilation_mode)
        if a.cudagraph_mode:
            cc_kw["cudagraph_mode"] = getattr(CUDAGraphMode, a.cudagraph_mode)
        kw["compilation_config"] = CompilationConfig(**cc_kw)
    return LLM(**kw)


def gen(llm, prompts, sp):
    t = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    return time.perf_counter() - t, out


def task_bench(a):
    from vllm import SamplingParams
    ids, bos = load_ids(a.model, a.text)
    llm = make_llm(a)
    res = {
        "task": "bench",
        "model": os.path.basename(a.model),
        "eager": bool(a.eager),
        "compilation_mode": a.compilation_mode,
        "cudagraph_mode": a.cudagraph_mode,
        "env_aot": os.environ.get("VLLM_USE_AOT_COMPILE"),
        "env_standalone": os.environ.get("VLLM_USE_STANDALONE_COMPILE"),
        "env_disable_cache": os.environ.get("VLLM_DISABLE_COMPILE_CACHE"),
        "prefill": {},
        "decode": {},
    }
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    for n in a.prefill:
        p = ids[1:n + 1]
        gen(llm, [p], sp1)
        ts = [gen(llm, [p], sp1)[0] for _ in range(a.reps)]
        res["prefill"][n] = {"tok_s_median": n / statistics.median(ts), "tok_s_all": [n / t for t in ts]}
        print(f"prefill {n}: {n / statistics.median(ts):.0f} tokens/s", flush=True)
    for b in a.streams:
        prompts = [ids[1 + 7 * i:129 + 7 * i] for i in range(b)]
        sp_long = SamplingParams(max_tokens=a.steps + 1, temperature=0.0, ignore_eos=True, detokenize=not a.no_detok)
        gen(llm, prompts, sp_long)
        # the decode time is the time of (prompt + steps + 1 tokens) minus the time of the prompt with one new token, same batch
        tps = []
        for _ in range(a.reps):
            t1, _ = gen(llm, prompts, sp1)
            t2, out = gen(llm, prompts, sp_long)
            assert all(len(o.outputs[0].token_ids) == a.steps + 1 for o in out)
            tps.append(b * a.steps / (t2 - t1))
        res["decode"][b] = {"tok_s_median": statistics.median(tps), "tok_s_all": tps}
        print(f"decode {b} streams: {statistics.median(tps):.1f} tokens/s", flush=True)
    print("RESULT " + json.dumps(res))


def task_ppl(a):
    from vllm import SamplingParams
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ids, bos = load_ids(a.model, a.text)
    n_ctx, first = a.ctx, a.ctx // 2
    llm = make_llm(a)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    chunks = []
    for i in range(a.chunks):
        c = list(ids[i * n_ctx:(i + 1) * n_ctx])
        c[0] = bos
        chunks.append(c)
    _, out = gen(llm, chunks, sp)
    nll, count = 0.0, 0
    lps = []
    for c, o in zip(chunks, out):
        pl = o.prompt_logprobs  # entry j is the log-probability of prompt token j given the tokens before it
        row = []
        for j in range(first + 1, n_ctx):
            lp = pl[j][c[j]].logprob
            nll -= lp
            count += 1
            row.append(lp)
        lps.append(row)
    res = {"task": "ppl", "model": os.path.basename(a.model), "chunks": a.chunks, "n_ctx": n_ctx, "scored_tokens": count, "ppl": math.exp(nll / count)}
    if a.ref:
        import numpy as np
        with open(a.ref, "rb") as f:
            assert f.read(8) == b"_logits_"
            rn_ctx, rn_vocab, rn_chunk = np.frombuffer(f.read(12), dtype=np.int32)
            rtok = np.frombuffer(f.read(4 * int(rn_ctx) * int(rn_chunk)), dtype=np.int32)
            off = f.tell()
        nt = a.chunks * n_ctx
        if list(rtok[:nt]) != list(ids[:nt]):
            raise SystemExit("reference tokens differ from this tokenizer's")
        nv = 2 * ((int(rn_vocab) + 1) // 2) + 4
        per = n_ctx - 1 - first
        mm = np.memmap(a.ref, dtype=np.uint16, mode="r", offset=off)
        d, n = 0.0, 0
        for i, row in enumerate(lps):
            blk = mm[i * per * nv:(i + 1) * per * nv].reshape(per, nv)
            hdr = np.ascontiguousarray(blk[:, :4]).view(np.float32)
            for r in range(per):
                tgt = chunks[i][first + 1 + r]
                ref_lp = hdr[r, 0] * float(blk[r, 4 + tgt]) + hdr[r, 1]
                d += abs(ref_lp - row[r])
                n += 1
        res["mean_abs_dlogp_target"] = float(d / n)
    print("RESULT " + json.dumps(res))


def task_loop(a):
    from vllm import SamplingParams
    ids, bos = load_ids(a.model, a.text)
    llm = make_llm(a)
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    per = []
    if a.kind == "prefill":
        p = [ids[1:a.tokens + 1]] * a.streams
        gen(llm, p, sp1)
        gen(llm, p, sp1)
        t_end = time.perf_counter() + a.seconds
        while time.perf_counter() < t_end:
            per.append(gen(llm, p, sp1)[0])
        toks = a.tokens * a.streams
    else:
        prompts = [ids[1 + 7 * i:129 + 7 * i] for i in range(a.streams)]
        sp_long = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True)
        gen(llm, prompts, sp_long)
        t_end = time.perf_counter() + a.seconds
        while time.perf_counter() < t_end:
            per.append(gen(llm, prompts, sp_long)[0])
        toks = 256 * a.streams
    print("RESULT " + json.dumps({"task": "loop", "kind": a.kind, "passes": len(per), "tok_s_median_incl_prompt": toks / statistics.median(per)}))


def task_trace(a):
    import torch
    from vllm import SamplingParams
    ids, bos = load_ids(a.model, a.text)
    llm = make_llm(a)
    sp1 = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)

    def mark(numel):
        torch.cuda.synchronize()
        torch.ones(numel, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
    if a.kind == "prefill":
        p = [ids[1:a.tokens + 1]] * a.streams
        gen(llm, p, sp1)
        gen(llm, p, sp1)
        mark(MARK_A)
        for _ in range(a.reps):
            gen(llm, p, sp1)
        mark(MARK_B)
    else:
        prompts = [ids[1 + 7 * i:129 + 7 * i] for i in range(a.streams)]
        sp_long = SamplingParams(max_tokens=a.reps + 1, temperature=0.0, ignore_eos=True)
        gen(llm, prompts, sp_long)
        mark(MARK_A)
        gen(llm, prompts, sp_long)
        mark(MARK_B)
    print("RESULT " + json.dumps({"task": "trace", "kind": a.kind, "reps": a.reps, "tokens": a.tokens, "streams": a.streams, "mark_a": MARK_A, "mark_b": MARK_B}))


def selftest():
    # decode time arithmetic: 2 streams, 128 steps, (t2 - t1) = 0.5 s -> 512 tokens/s
    assert abs(2 * 128 / (1.0 - 0.5) - 512.0) < 1e-9
    ap = build_parser()
    a = ap.parse_args(["bench", "--cudagraph-mode", "PIECEWISE", "--compilation-mode", "VLLM_COMPILE"])
    assert a.cudagraph_mode == "PIECEWISE" and a.compilation_mode == "VLLM_COMPILE" and not a.eager
    print("VERDICT llm4_vllm selftest PASS")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["bench", "ppl", "trace", "loop", "selftest"])
    ap.add_argument("--model")
    ap.add_argument("--text")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--batched-tokens", type=int, default=4096)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--cudagraph-mode", choices=["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"], default=None)
    ap.add_argument("--compilation-mode", choices=["NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE"], default=None)
    ap.add_argument("--no-detok", action="store_true", help="do not detokenise the generated tokens (host-side work per step)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prefill", type=int, nargs="*", default=[128, 512, 2048])
    ap.add_argument("--streams", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--ref")
    ap.add_argument("--kind", default="prefill")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--tokens", type=int, default=512)
    return ap


def main():
    a = build_parser().parse_args()
    if a.task == "selftest":
        return selftest()
    if a.task in ("trace", "loop"):
        a.streams = a.streams[0] if isinstance(a.streams, list) else a.streams
    {"bench": task_bench, "ppl": task_ppl, "trace": task_trace, "loop": task_loop}[a.task](a)


if __name__ == "__main__":
    main()
