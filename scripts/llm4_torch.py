#!/usr/bin/env python3
"""Round 4: Llama through PyTorch (HuggingFace modules, eager) with real FP8 matrix maths.

Run with the GPU PyTorch venv (toolchains/torch-rocm10-venv) through the gated runner (scripts/study-llm.sh r4*).
  llm4_torch.py <task> --model DIR --fp8|--bf16 [options]
Tasks: bench (prefill at 128/512/2048 and decode streams), gemm (the FP8 and bf16 GEMMs at the model's shapes), ppl (perplexity on the
study text, llama-perplexity's chunking), kl (KL divergence of this run against a saved reference), trace (a marked region for rocprofv3),
selftest (no GPU).
The FP8 path: every Linear of the decoder layers is an FP8Linear. Weights stay FP8 (E4M3) with the checkpoint's per-tensor scale; the
activation is quantised with the checkpoint's static per-tensor input scale and torch._scaled_mm does the matrix product (hipBLASLt on
this card). lm_head, embeddings and norms stay BF16 as in the checkpoint. The 16-bit path is the same modules in BF16.
"""
import argparse, json, math, os, statistics, sys, time

if os.environ.get("LLM4_NO_TRITON") == "1":
    # rocprofv3 crashes when triton's bundled LLVM is imported next to its own ("registered more than once"); the eager HF path needs no triton
    sys.modules["triton"] = None

import torch

FP8 = torch.float8_e4m3fn
SYNC_AFTER_MM = os.environ.get("LLM4_SYNC_AFTER_MM") == "1"
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class FP8Linear(torch.nn.Module):
    def __init__(self, w, w_scale, in_scale, pad_rows=16):
        super().__init__()
        self.weight = w  # [out, in] float8_e4m3fn, row-major; w.t() is the column-major operand _scaled_mm wants
        self.w_scale = w_scale.float().reshape(()).to(w.device)
        self.in_scale = in_scale.float().reshape(()).to(w.device)
        self.inv_in = (1.0 / self.in_scale).reshape(())
        self.out_features, self.in_features = w.shape
        self.pad_rows = pad_rows

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        m = x2.shape[0]
        # static activation quantisation: x / s, clamped to the E4M3 range, rounded to nearest even by the cast
        xq = (x2.float() * self.inv_in).clamp_(-448.0, 448.0).to(FP8)
        if m % self.pad_rows:
            xq = torch.nn.functional.pad(xq, (0, 0, 0, self.pad_rows - m % self.pad_rows))
        y = torch._scaled_mm(xq, self.weight.t(), scale_a=self.in_scale, scale_b=self.w_scale, out_dtype=torch.bfloat16)
        if SYNC_AFTER_MM:  # debugging switch: tests whether the NaN and GPU-hang problem is an ordering hazard around the GEMM
            torch.cuda.synchronize()
        if y.shape[0] != m:
            y = y[:m]
        return y.reshape(*shp[:-1], self.out_features)


def _shards(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


def load_model(d, fp8, device="cuda"):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.llama import modeling_llama as ml
    cfg = AutoConfig.from_pretrained(d)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    cfg.tie_word_embeddings = False
    with torch.device("meta"):
        model = ml.LlamaForCausalLM(cfg).to(torch.bfloat16)
    tensors = {}
    for s in _shards(d):
        with safe_open(s, "pt", device=device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    n_fp8 = 0
    rest = {}
    for k, t in tensors.items():
        if k.endswith("_scale"):
            continue
        if t.dtype == FP8:
            if not fp8:
                raise SystemExit(f"{k} is FP8 but --bf16 was asked")
            mod_name = k[: -len(".weight")]
            parent = model.get_submodule(mod_name.rsplit(".", 1)[0])
            leaf = mod_name.rsplit(".", 1)[1]
            setattr(parent, leaf, FP8Linear(t, tensors[mod_name + ".weight_scale"], tensors[mod_name + ".input_scale"]))
            n_fp8 += 1
        else:
            rest[k] = t
    if "lm_head.weight" not in rest:  # tied embeddings (Meta's file has no lm_head)
        rest["lm_head.weight"] = rest["model.embed_tokens.weight"]
    missing = model.load_state_dict(rest, strict=False, assign=True)
    unexpected = [k for k in missing.unexpected_keys]
    miss = [k for k in missing.missing_keys]
    if unexpected or miss:
        raise SystemExit(f"load mismatch: missing {miss[:4]} unexpected {unexpected[:4]}")
    model.model.rotary_emb = ml.LlamaRotaryEmbedding(cfg).to(device)
    model.eval()
    if fp8 and n_fp8 != 7 * cfg.num_hidden_layers:
        raise SystemExit(f"expected {7 * cfg.num_hidden_layers} FP8 linears, got {n_fp8}")
    if not fp8 and n_fp8:
        raise SystemExit("FP8 tensors in a bf16 run")
    return model, cfg, n_fp8


def ppl_tokens(tokenizer_dir, text_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    text = open(text_path, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=True)["input_ids"]
    return ids, tok.bos_token_id


def chunk_inputs(ids, bos, n_ctx, i):
    c = list(ids[i * n_ctx:(i + 1) * n_ctx])
    c[0] = bos  # llama-perplexity puts BOS in front of every chunk
    return c


def nll_for_chunk(logits_f32, chunk, first):
    """llama-perplexity's scoring: positions first .. n_ctx-2 predict the next token (the second half of the chunk)."""
    lp = torch.log_softmax(logits_f32[first:-1], dim=-1)
    tgt = torch.tensor(chunk[first + 1:], device=lp.device)
    return -lp.gather(1, tgt[:, None])[:, 0], lp


class RefLogits:
    """llama-perplexity's saved-logits file (--kl-divergence-base): header, the chunks' tokens, then per scored position a scale, a minimum
    log-probability and one 16-bit code per vocabulary entry (log p = scale * code + minimum). Read with a memory map."""

    def __init__(self, path, n_ctx):
        import numpy as np
        self.np = np
        with open(path, "rb") as f:
            if f.read(8) != b"_logits_":
                raise SystemExit("not a llama.cpp logits file")
            self.n_ctx, self.n_vocab, self.n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
            self.tokens = np.frombuffer(f.read(4 * int(self.n_ctx) * int(self.n_chunk)), dtype=np.int32)
            self.offset = f.tell()
        if int(self.n_ctx) != n_ctx:
            raise SystemExit(f"logits file n_ctx {self.n_ctx} != {n_ctx}")
        self.nv = 2 * ((int(self.n_vocab) + 1) // 2) + 4
        self.per_chunk = (n_ctx - 1 - n_ctx // 2)
        self.mm = np.memmap(path, dtype=np.uint16, mode="r", offset=self.offset)

    def chunk_logprobs(self, i):
        np = self.np
        a = self.mm[i * self.per_chunk * self.nv:(i + 1) * self.per_chunk * self.nv].reshape(self.per_chunk, self.nv)
        hdr = np.ascontiguousarray(a[:, :4]).view(np.float32)  # scale, min_log_prob
        codes = torch.from_numpy(np.ascontiguousarray(a[:, 4:4 + int(self.n_vocab)]).astype(np.int32)).to("cuda")
        sc = torch.from_numpy(hdr[:, 0:1].copy()).to("cuda")
        mn = torch.from_numpy(hdr[:, 1:2].copy()).to("cuda")
        return codes.float() * sc + mn


def task_ppl(a):
    model, cfg, _ = load_model(a.model, a.fp8)
    ids, bos = ppl_tokens(a.model, a.text)
    n_ctx, first = a.ctx, a.ctx // 2
    nll_all, count = [], 0
    kl_sum, kl_max, top_same, kl_n, dlp = 0.0, 0.0, 0, 0, 0.0
    ref = RefLogits(a.ref, n_ctx) if a.ref else None
    if ref is not None:
        nt = a.chunks * n_ctx
        if a.chunks > int(ref.n_chunk) or list(ref.tokens[:nt]) != list(ids[:nt]):
            raise SystemExit("the reference file's tokens differ from this tokenizer's (or it has fewer chunks)")
    for i in range(a.chunks):
        c = chunk_inputs(ids, bos, n_ctx, i)
        with torch.no_grad():
            logits = model(torch.tensor([c], device="cuda"), use_cache=False).logits[0].float()
        nll, lp = nll_for_chunk(logits, c, first)
        nll_all.append(nll.double().sum().item())
        count += nll.numel()
        if ref is not None:
            lq = ref.chunk_logprobs(i)
            # llama-perplexity counts only reference entries above -16 (it stores nothing below max - 16); the clipped tail would add a bias
            kl = (lq.exp() * (lq - lp) * (lq > -16.0)).sum(-1)
            kl_sum += kl.double().sum().item()
            kl_n += kl.numel()
            kl_max = max(kl_max, kl.max().item())
            top_same += (lq.argmax(-1) == lp.argmax(-1)).sum().item()
            tgt = torch.tensor(c[first + 1:], device="cuda")[:, None]
            dlp += (lq.gather(1, tgt) - lp.gather(1, tgt)).abs().double().sum().item()
        print(f"chunk {i + 1}: ppl so far {math.exp(sum(nll_all) / count):.4f}", flush=True)
    res = {"task": "ppl", "model": os.path.basename(a.model), "fp8": a.fp8, "chunks": a.chunks, "n_ctx": n_ctx, "scored_tokens": count,
           "n_tokens_text": len(ids), "ppl": math.exp(sum(nll_all) / count)}
    if ref is not None:
        res.update({"kl_mean": kl_sum / kl_n, "kl_max": kl_max, "same_top_pct": 100.0 * top_same / kl_n, "kl_n": kl_n,
                    "mean_abs_dlogp_target": dlp / kl_n})
    print("RESULT " + json.dumps(res))


def _ev():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        s, e = _ev()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / 1000.0)
    return ts


def task_bench(a):
    from transformers import DynamicCache
    model, cfg, _ = load_model(a.model, a.fp8)
    ids, bos = ppl_tokens(a.model, a.text)
    res = {"task": "bench", "model": os.path.basename(a.model), "fp8": a.fp8, "prefill": {}, "decode": {}}

    def prefill(n, b=1):
        x = torch.tensor([ids[1:n + 1]] * b, device="cuda")
        with torch.no_grad():
            model(x, use_cache=False, logits_to_keep=1)

    for n in a.prefill:
        prefill(n)
        ts = timed(lambda: prefill(n), a.reps)
        res["prefill"][n] = {"tok_s_median": n / statistics.median(ts), "tok_s_all": [n / t for t in ts]}
        print(f"prefill {n}: {n / statistics.median(ts):.0f} tokens/s", flush=True)
    for b in a.streams:
        def run_decode(nsteps):
            x = torch.tensor([ids[1:129]] * b, device="cuda")
            cache = DynamicCache(config=cfg)
            with torch.no_grad():
                out = model(x, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                s, e = _ev()
                s.record()
                for _ in range(nsteps):
                    out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                e.record()
                e.synchronize()
            return s.elapsed_time(e) / 1000.0
        run_decode(8)
        ts = [run_decode(a.steps) for _ in range(a.reps)]
        tps = [b * a.steps / t for t in ts]
        res["decode"][b] = {"tok_s_median": statistics.median(tps), "tok_s_all": tps, "ms_per_step_median": 1000 * statistics.median(ts) / a.steps}
        print(f"decode {b} streams: {statistics.median(tps):.1f} tokens/s ({res['decode'][b]['ms_per_step_median']:.2f} ms/step)", flush=True)
    print("RESULT " + json.dumps(res))


def task_gemm(a):
    """The GEMMs of the model's layers (M = weight rows, K = inputs, N = tokens) timed alone with CUDA events."""
    import itertools
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model)
    h, i_, kv = cfg.hidden_size, cfg.intermediate_size, cfg.num_key_value_heads * cfg.head_dim
    shapes = {"q": (h, h), "k_v": (kv, h), "o": (h, h), "gate_up": (i_, h), "down": (h, i_)}
    res = {"task": "gemm", "shapes": {}}
    for name, (m, k) in shapes.items():
        for n in a.ns:
            w8 = (torch.randn(m, k, device="cuda") * 0.5).to(FP8)
            x8 = (torch.randn(n, k, device="cuda") * 0.5).to(FP8)
            wb = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            xb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            one = torch.ones((), device="cuda")
            f8 = lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
            fb = lambda: xb @ wb.t()
            row = {}
            for label, fn in (("fp8", f8), ("bf16", fb)):
                if n % 16 and label == "fp8":
                    continue
                for _ in range(5):
                    fn()
                ts = timed(fn, a.reps)
                t = statistics.median(ts)
                row[label] = {"ms": 1000 * t, "tflops": 2.0 * m * n * k / t / 1e12, "min_ms": 1000 * min(ts), "max_ms": 1000 * max(ts)}
            res["shapes"][f"{name}:M{m}:K{k}:N{n}"] = row
            print(name, m, k, n, {k_: round(v["tflops"], 1) for k_, v in row.items()}, flush=True)
    print("RESULT " + json.dumps(res))


def task_loop(a):
    """A load of about --seconds s for the sampler: prompts of --tokens tokens back to back, or decode steps of --streams streams."""
    from transformers import DynamicCache
    model, cfg, _ = load_model(a.model, a.fp8)
    ids, bos = ppl_tokens(a.model, a.text)
    x = torch.tensor([ids[1:(a.tokens if a.kind == "prefill" else 128) + 1]] * a.streams, device="cuda")
    n_done, per = 0, []
    with torch.no_grad():
        if a.kind == "prefill":
            for _ in range(3):
                model(x, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            t_end = time.perf_counter() + a.seconds
            while time.perf_counter() < t_end:
                per.append(timed(lambda: model(x, use_cache=False, logits_to_keep=1), 1)[0])
            toks = a.tokens * a.streams
        else:
            cache = DynamicCache(config=cfg)
            out = model(x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(8):
                out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            t_end = time.perf_counter() + a.seconds
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                per.append(time.perf_counter() - t0)
            toks = a.streams
    print("RESULT " + json.dumps({"task": "loop", "kind": a.kind, "passes": len(per), "tok_s_median": toks / statistics.median(per)}))


def task_detgemm(a):
    """Is the GEMM bit-reproducible? The same inputs are multiplied again and again (FP8 _scaled_mm and bf16 matmul); every result is compared
    bit for bit with the first, and NaNs are counted. Shapes as in the model (3B gate/up and down) at N tokens."""
    res = {"task": "detgemm", "rows": {}}
    for name, (m, k) in {"gate_up": (8192, 3072), "down": (3072, 8192), "q": (3072, 3072)}.items():
        n = a.tokens
        g = torch.Generator(device="cuda").manual_seed(1)
        w8 = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(FP8)
        x8 = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(FP8)
        wb = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        xb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
        one = torch.ones((), device="cuda")
        for label, fn in (("fp8", lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)), ("bf16", lambda: xb @ wb.t())):
            ref = fn()
            torch.cuda.synchronize()
            bad, nan = 0, 0
            t_end = time.perf_counter() + a.seconds
            calls = 0
            while time.perf_counter() < t_end:
                y = fn()
                bad += int(not torch.equal(y.view(torch.int16), ref.view(torch.int16)))
                nan += int(torch.isnan(y).any().item())
                calls += 1
            res["rows"][f"{name}:{label}"] = {"calls": calls, "differing_results": bad, "results_with_nan": nan}
            print(name, label, calls, "differ", bad, "nan", nan, flush=True)
    print("RESULT " + json.dumps(res))


def mark(numel):
    """A marker kernel with a grid size no other kernel of the run has (the trace analysis finds the region between two of them)."""
    torch.cuda.synchronize()
    torch.ones(numel, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()


MARK_A, MARK_B = 1234567, 2345678


def task_trace(a):
    from transformers import DynamicCache
    model, cfg, _ = load_model(a.model, a.fp8)
    ids, bos = ppl_tokens(a.model, a.text)
    x = torch.tensor([ids[1:a.tokens + 1]] * a.streams, device="cuda")
    with torch.no_grad():
        if a.kind == "prefill":
            for _ in range(2):
                model(x, use_cache=False, logits_to_keep=1)
            mark(MARK_A)
            for _ in range(a.reps):
                model(x, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            mark(MARK_B)
        else:
            cache = DynamicCache(config=cfg)
            out = model(x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(8):
                out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            mark(MARK_A)
            for _ in range(a.reps):
                out = model(nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            mark(MARK_B)
    print("RESULT " + json.dumps({"task": "trace", "kind": a.kind, "reps": a.reps, "tokens": a.tokens, "streams": a.streams, "mark_a": MARK_A, "mark_b": MARK_B}))


def selftest():
    # the scoring window: for n_ctx 8 the scored positions are 4, 5, 6 (3 tokens, targets chunk[5], chunk[6], chunk[7])
    logits = torch.zeros(8, 5)
    chunk = [0, 1, 2, 3, 4, 0, 1, 2]
    nll, _ = nll_for_chunk(logits, chunk, 4)
    assert nll.numel() == 3 and abs(nll[0].item() - math.log(5)) < 1e-6, nll
    c = chunk_inputs(list(range(20)), 99, 8, 1)
    assert c == [99, 9, 10, 11, 12, 13, 14, 15], c
    # the activation quantiser: a value above the E4M3 range is clamped (448), not turned into NaN
    v = (torch.tensor([1000.0, -1000.0, 1.0]).clamp_(-448.0, 448.0)).to(FP8).float()
    assert v.tolist() == [448.0, -448.0, 1.0], v
    # KL of identical distributions is 0, of different ones positive
    p = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1)
    q = torch.log_softmax(torch.tensor([[3.0, 2.0, 1.0]]), -1)
    assert (p.exp() * (p - p)).sum() == 0 and (p.exp() * (p - q)).sum() > 0.5
    # the -16 cut: a reference entry below -16 contributes nothing, as in llama-perplexity
    lq = torch.tensor([[-0.01, -20.0]]); lp2 = torch.tensor([[-0.01, -30.0]])
    assert float((lq.exp() * (lq - lp2) * (lq > -16.0)).sum()) == 0.0 and float((lq.exp() * (lq - lp2)).sum()) > 0.0
    # the reference-file reader on a canned file: 1 chunk of n_ctx 8 (scored positions 4..6 -> 3), vocab 4 (nv = 8)
    import numpy as np, tempfile
    if torch.cuda.is_available() and "--gpu" in sys.argv:
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "ref.bin")
            nv = 2 * ((4 + 1) // 2) + 4
            body = np.zeros((3, nv), np.uint16)
            for r in range(3):
                body[r, :4] = np.array([0.5, -3.0], np.float32).view(np.uint16)  # scale, min_log_prob
                body[r, 4:8] = [0, 2, 4, 6]
            with open(fn, "wb") as f:
                f.write(b"_logits_" + np.array([8, 4, 1], np.int32).tobytes() + np.arange(8, dtype=np.int32).tobytes() + body.tobytes())
            r = RefLogits(fn, 8)
            lp = r.chunk_logprobs(0).cpu()
            assert lp.shape == (3, 4) and lp[0].tolist() == [-3.0, -2.0, -1.0, 0.0], lp
            assert r.tokens.tolist() == list(range(8))
    print("VERDICT llm4_torch selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["bench", "gemm", "ppl", "trace", "loop", "detgemm", "selftest"])
    ap.add_argument("--model")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fp8", action="store_true")
    g.add_argument("--bf16", action="store_true")
    ap.add_argument("--text")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prefill", type=int, nargs="*", default=[128, 512, 2048])
    ap.add_argument("--streams", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--ns", type=int, nargs="*", default=[16, 128, 512, 2048])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--ref")
    ap.add_argument("--kind", default="prefill")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--streams-trace", dest="streams_one", type=int, default=1)
    a = ap.parse_args()
    if a.task == "selftest":
        return selftest()
    if a.task in ("trace", "loop"):
        a.streams = a.streams_one
    if a.task != "detgemm" and (a.model is None or not (a.fp8 or a.bf16 or a.task == "gemm")):
        ap.error("--model and --fp8 or --bf16 are needed")
    torch.manual_seed(0)
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)} model {a.model} fp8={a.fp8}", flush=True)
    {"bench": task_bench, "gemm": task_gemm, "ppl": task_ppl, "trace": task_trace, "loop": task_loop, "detgemm": task_detgemm}[a.task](a)


if __name__ == "__main__":
    main()
