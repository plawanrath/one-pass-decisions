"""C1 and Q1 on one host: latency of every served path as K grows, on one fine-tuned model.

    .venv/bin/python scripts/latency_sweep.py --size base --task guardrail --threads 8 --K 1 5 20 50 --S 512 2048
Paths: torch one-seq (full), torch naive batch (full), engine cold (block, packed), engine hit (cached state),
ORT fp32 / int8 one-seq (full), ORT packed (block). Prefill-only decoder and GLiNER cells are separate scripts.
Requests are synthesized from the test pool: a state of about S tokens and K questions drawn from that task's pool.
"""
from __future__ import annotations

import argparse
import copy

import torch

from onepass.bench.timing import TimingCell, compare, thread_setup, time_cell
from onepass.cli import RESULTS, WINDOW_TOKENS, host_tag, model_dir, pool_dir, state_side, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter, layout_request, pack_batch
from onepass.paths.torch_eager import naive_batch, one_seq
from onepass.tasks.pools import load_pool
from onepass.bench.requests import make_requests


def parse_cells(cells: list[str], S: list[int], K: list[int]) -> list[tuple[int, int]]:
    """The (S, K) cells to time: explicit `S:K` pairs (trace-shaped cells) or the full S x K grid."""
    if cells:
        return [(int(c.split(":")[0]), int(c.split(":")[1])) for c in cells]
    return [(s, k) for s in S for k in K]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="guardrail")
    ap.add_argument("--mode", default="block")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--K", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--S", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-at-S", nargs="*", default=[], help="per-S request-count overrides as S:n, e.g. 2048:50")
    ap.add_argument("--n-slow", type=int, default=20, help="requests per cell for the slow baselines (decoder readout, schema encoder), whose latency is 10 to 100x the others")
    ap.add_argument("--warmup-slow", type=int, default=3)
    ap.add_argument("--hot-states", type=int, default=8, help="distinct resident states the cached-state cell cycles through")
    ap.add_argument("--skip", nargs="*", default=[], help="cells to leave out, as S<S>_K<K> (a rented host drops the slowest corner)")
    ap.add_argument("--cells", nargs="*", default=[], help="explicit S:K cells instead of the S x K grid (the traces' median and 95th-percentile shapes)")
    ap.add_argument("--out-tag", default="", help="suffix of the results directory (a second grid such as the trace-shaped cells)")
    ap.add_argument("--engine-kernel", default="sdpa", choices=["sdpa", "flex", "split", "cascade"], help="attention kernel of the engine paths (flex: the block mask as a FlexAttention block mask; cells suffixed -flex)")
    ap.add_argument("--stock", action="store_true", help="time the pretrained encoder with an untrained head instead of the fine-tune: latency does not depend on the weight values, so this times a model size whose fine-tuned weights are not on this host (results tagged -stock)")
    ap.add_argument("--paths", nargs="+", default=["one-seq", "naive-batch", "engine-cold", "engine-hit", "ort-one-seq", "ort-int8-one-seq", "ort-packed"],
                    help="also: compile-one-seq (the one-seq path through torch.compile), prefill (decoder readout, one prefill per question), gliner (second workload model, one call per request)")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda for the PyTorch paths")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="weights and activations of the PyTorch encoder paths")
    ap.add_argument("--decoder", default="smol1.7", help="decoder for the readout paths (keys of onepass.cli.DECODERS)")
    ap.add_argument("--no-cudnn-sdp", action="store_true", help="disable the cuDNN attention backend (CUDA): with it, every first-seen sequence length builds an execution plan (about 30 ms on an H100), which dominates a small encoder's forward")
    ap.add_argument("--cap-state", action="store_true", help="one-seq paths keep only the first S tokens of each state (the truncated-context operating points; without it a state is whole pool states concatenated to at least S tokens)")
    ap.add_argument("--engine-compiled", action="store_true", help="engine paths with each layer's attention-free work compiled (EngineConfig.compiled; cells suffixed -compiled)")
    ap.add_argument("--warm-shapes", action="store_true", help="serve every timed request once before timing, so each cell measures shapes the process has already seen (a server that pads to a fixed set of lengths)")
    a = ap.parse_args()
    ksuf = ("" if a.engine_kernel == "sdpa" else f"-{a.engine_kernel}") + ("-compiled" if a.engine_compiled else "")
    thread_setup(a.threads)
    if a.no_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    md = model_dir(a.size, a.mode, a.task)
    if a.stock:
        import tempfile
        from pathlib import Path

        from onepass.cli import ENCODERS
        from onepass.model.encoder import load_tokenizer_adapter

        enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval().to(a.device)
        _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
        md = Path(tempfile.mkdtemp(prefix="onepass-stock-"))  # ONNX exports of the stock model go here, never beside a fine-tune
        a.out_tag = "-".join(x for x in (a.out_tag, "stock") if x)
    else:
        enc = OnePassEncoder.load(md).eval().to(a.device)
        tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
    recs = load_pool(pool_dir(a.task), "test")
    if a.dtype == "bf16":
        enc = enc.to(torch.bfloat16)
    late_models = {}
    for kind in ("late", "bi"):
        if kind in a.paths:
            from onepass.cli import MODELS
            from onepass.model.late import LateEncoder

            for root in (MODELS, MODELS / "_from_h100", MODELS / "cuda"):
                d = root / f"modernbert-{a.size}-{kind}-{a.task}"
                if d.exists():
                    late_models[kind] = LateEncoder.load(d).eval().to(a.device)
                    break
    enc_hf, hf_tag = None, ""
    if any(x in a.paths for x in ("hf-one-seq", "hf-batched")):
        # the stock forward at strength: FlashAttention-2 unpadded where available, SDPA otherwise (recorded in the path name)
        try:
            import flash_attn  # noqa: F401
            impl, hf_tag = "flash_attention_2", "fa2"
        except ImportError:
            impl, hf_tag = "sdpa", "sdpa"
        from onepass.cli import ENCODERS

        enc_hf = OnePassEncoder.from_pretrained(*ENCODERS[a.size], attn_implementation=impl) if a.stock else OnePassEncoder.load(md, attn_implementation=impl)
        enc_hf = enc_hf.eval().to(torch.bfloat16 if (a.dtype == "bf16" or impl == "flash_attention_2") else torch.float32).to(a.device)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("" if a.dtype == "fp32" else f"-{a.dtype}")
    sync = (torch.cuda.synchronize if a.device == "cuda" else (torch.mps.synchronize if a.device == "mps" else None))
    out_dir = RESULTS / "latency" / (f"{host}-{a.size}-{a.task}-t{a.threads}" + (f"-{a.out_tag}" if a.out_tag else ""))
    pf = gl = None
    if any(x in a.paths for x in ("prefill", "prefill-cached", "prefill-uncached")):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from onepass.cli import DECODERS
        from onepass.paths.prefill_only import PrefillOnly

        repo, rev = DECODERS[a.decoder]
        pf = PrefillOnly(AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32), AutoTokenizer.from_pretrained(repo, revision=rev), device=a.device)
    if "gliner" in a.paths:
        from onepass.cli import GLINER
        from onepass.paths.gliner import GLiNERDecide

        gl = GLiNERDecide(GLINER[0], GLINER[1], device=a.device)
    ort = {}
    if any(p.startswith("ort") for p in a.paths):
        from onepass.paths.ort import OrtPath, export_packed, quantize_int8

        sample = pack_batch([layout_request(tok, recs[0].state, [recs[0].question], "block")], tok.pad_id, "block")
        p32 = export_packed(enc, sample, md / "packed.onnx")
        ort["fp32"] = OrtPath(p32, enc, threads=a.threads, name="ort-fp32")
        if "ort-int8-one-seq" in a.paths:
            ort["int8"] = OrtPath(quantize_int8(p32, md / "packed-int8.onnx"), enc, threads=a.threads, name="ort-int8")
    grid = {}
    enc_c = None
    n_override = {int(x.split(":")[0]): int(x.split(":")[1]) for x in a.n_at_S}
    for S, K in parse_cells(a.cells, a.S, a.K):
        if True:
            if f"S{S}_K{K}" in a.skip:
                print(f"S={S} K={K} skipped", flush=True)
                continue
            n_here = n_override.get(S, a.n)
            reqs = make_requests(recs, tok, S, K, n_here + a.warmup, seed=S * 1000 + K)
            # every contract tokenized once, outside the timed region, for every path: the engine caches token ids per
            # state, so a baseline that tokenized inside its timed call was charged work the engine was not (an earlier
            # version did, which made the baseline's latencies bimodal by contract length)
            state_ids = [tok.encode(st) for st, _ in reqs]
            N = int(sum(len(q.options) for _, qs in reqs for q in qs) / sum(len(qs) for _, qs in reqs))
            cells = {}
            kw = dict(host=host, threads=a.threads, S=S, K=K, N=N, n_requests=n_here, n_warmup=a.warmup, sync=sync)
            kw_slow = dict(kw, n_requests=min(n_here, a.n_slow), n_warmup=min(a.warmup, a.warmup_slow))

            def cell(name, fn, slow=False):
                """Time one path for this (S, K) unless its cell file exists (a sweep resumes cell by cell)."""
                f = out_dir / f"cell_S{S}_K{K}_{name}.json"
                if f.exists():
                    cells[name] = TimingCell.load(f)
                    return
                try:
                    if a.warm_shapes:
                        for i in range(len(reqs)):
                            fn(i)
                    cells[name] = time_cell(fn, path=name, **(kw_slow if slow else kw))
                except torch.cuda.OutOfMemoryError:
                    # a path that does not fit the accelerator at this shape is a result, not a crash of the sweep
                    torch.cuda.empty_cache()
                    write_json(out_dir / f"oom_S{S}_K{K}_{name}.json", {"path": name, "S": S, "K": K, "out_of_memory": True})
                    print(f"S={S} K={K} {name}: out of memory", flush=True)
                    return
                cells[name].save(f)
            if "one-seq" in a.paths:
                cap = {"max_state_tokens": S, "state_side": state_side(a.task)} if a.cap_state else {}
                cell("torch-one-seq-full", lambda i: one_seq(enc, tok, *reqs[i], **(cap or {"max_state_tokens": min(max(S, 4096), 8190)}), state_ids=state_ids[i]))
            if "compile-one-seq" in a.paths:
                # the tuned PyTorch baseline: the same one-seq path through torch.compile (inductor, dynamic shapes),
                # compiled once per process; the warm-up requests absorb compilation
                if enc_c is None:
                    enc_c = torch.compile(enc, dynamic=True)
                cell("torch-compile-one-seq-full", lambda i: one_seq(enc_c, tok, *reqs[i], max_state_tokens=min(max(S, 4096), 8190), state_ids=state_ids[i]))
            if enc_hf is not None and "hf-one-seq" in a.paths:
                from onepass.paths.torch_eager import hf_full
                cell(f"hf-{hf_tag}-one-seq-full", lambda i: hf_full(enc_hf, tok, *reqs[i], batched=False, state_ids=state_ids[i]))
            if enc_hf is not None and "hf-batched" in a.paths:
                from onepass.paths.torch_eager import hf_full
                cell(f"hf-{hf_tag}-batched-full", lambda i: hf_full(enc_hf, tok, *reqs[i], batched=True, state_ids=state_ids[i]))
            for kind, lm in late_models.items():
                # served with the context encoded once per request (its cache) and every option of every question
                # encoded against it; the question+option texts are not cached across requests
                from onepass.model.late import option_texts, tokenize as ltok

                def late_req(i, lm=lm):
                    state, qs = reqs[i]
                    with torch.no_grad():
                        ci, cm = ltok(tok._hf, [state], 1024)
                        ctx = lm.encode(ci.to(a.device), cm.to(a.device))
                        for q in qs:
                            oi, om = ltok(tok._hf, option_texts(q), 96)
                            lm.score(ctx, cm.to(a.device), lm.encode(oi.to(a.device), om.to(a.device)), om.to(a.device).float())
                cell(f"{kind}-encoder", late_req)
            # states are built to at least S tokens; every path reads up to max(S, 4,096) state tokens (one window
            # at most for full attention: the encoders' positions end at 8,192). Every baseline tokenizes the state once
            # per request inside the timed call, as the engine does (re-tokenizing it per question cost up to 0.9 s of a
            # 1.6 s GPU request at 8k tokens and 41 questions)
            st = {"max_state_tokens": min(max(S, 4096), 8190)}
            if "varlen-one-seq" in a.paths or "varlen-batched" in a.paths:
                from onepass.paths.varlen import default_kernel, varlen_paths
                vk = default_kernel(a.device)
                if "varlen-one-seq" in a.paths:
                    cell(f"varlen-{vk}-one-seq-full", lambda i: varlen_paths(enc, tok, *reqs[i], batched=False, kernel=vk, state_ids=state_ids[i], **st))
                if "varlen-batched" in a.paths:
                    cell(f"varlen-{vk}-batched-full", lambda i: varlen_paths(enc, tok, *reqs[i], batched=True, kernel=vk, state_ids=state_ids[i], **st))
            if "varlen-batched-compiled" in a.paths:
                # the baseline with the engine's compiled attention-free pieces: a compiled engine against an equally
                # compiled baseline (compiled once per process; the warm-up requests absorb compilation)
                from onepass.paths.varlen import default_kernel, varlen_paths
                vk = default_kernel(a.device)
                cell(f"varlen-{vk}-batched-compiled-full", lambda i: varlen_paths(enc, tok, *reqs[i], batched=True, kernel=vk, state_ids=state_ids[i], compiled=True, **st))
            if "naive-batch" in a.paths:
                cell("torch-naive-batch-full", lambda i: naive_batch(enc, tok, *reqs[i], state_ids=state_ids[i], **st))
            if "engine-cold" in a.paths:
                eng = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel=a.engine_kernel, compiled=a.engine_compiled, **st))
                cell("engine-cold" + ksuf, lambda i: eng.decide(*reqs[i]))
            for fused in (True, False):
                # a cold request that also leaves its state in the cache: from the same pass (fused) or a second one
                base = "engine-cold-fill" + ("" if fused else "-unfused")
                name = base + ksuf
                if base in a.paths:
                    engf = OnePassEngine(enc, tok, EngineConfig(use_cache=True, fuse_fill=fused, cache_bytes=8 << 30, attn_kernel=a.engine_kernel, compiled=a.engine_compiled, **st))

                    def cold_fill(i, e=engf):
                        e.cache._d.clear()  # every request arrives cold; clearing a dict is not part of the work timed
                        e.cache.bytes = 0
                        return e.decide(*reqs[i])
                    cell(name, cold_fill)
            if "engine-windowed-cold" in a.paths or "engine-windowed-hit" in a.paths:
                # the windowed state: exactly S content tokens in independent windows of WINDOW_TOKENS (S may exceed 8,192)
                wcfg = dict(mask="windowed", window_tokens=WINDOW_TOKENS, max_state_tokens=S + 2 * -(-S // (WINDOW_TOKENS - 2)))
                if "engine-windowed-cold" in a.paths:
                    engw = OnePassEngine(enc, tok, EngineConfig(use_cache=False, attn_kernel=a.engine_kernel, compiled=a.engine_compiled, **wcfg))
                    cell("engine-windowed-cold" + ksuf, lambda i: engw.decide(*reqs[i]))
                if "engine-windowed-hit" in a.paths:
                    hotw = [s for s, _ in reqs[: a.hot_states]]
                    reqs_hw = [(hotw[i % len(hotw)], qs) for i, (_, qs) in enumerate(reqs)]
                    engw2 = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=24 << 30, attn_kernel=a.engine_kernel, compiled=a.engine_compiled, **wcfg))
                    for s_ in hotw:
                        engw2.decide(s_, reqs[0][1][:1])
                    cell("engine-windowed-hit" + ksuf, lambda i: engw2.decide(*reqs_hw[i]))
            if "engine-hit" in a.paths:
                # The cached-state path is measured on resident states: requests cycle through a small set of hot
                # states (as system prompts and long-lived agent transcripts do), each primed once before timing. With
                # one distinct state per request the working set (about 277 MB per 2,048-token state in fp32) would
                # overflow any bounded cache and the cell would measure LRU thrash, not the hit path.
                hot = [s for s, _ in reqs[: a.hot_states]]
                reqs_hit = [(hot[i % len(hot)], qs) for i, (_, qs) in enumerate(reqs)]
                eng2 = OnePassEngine(enc, tok, EngineConfig(use_cache=True, cache_bytes=8 << 30, attn_kernel=a.engine_kernel, compiled=a.engine_compiled, **st))
                for s in hot:
                    eng2.decide(s, reqs[0][1][:1])
                cell("engine-hit" + ksuf, lambda i: eng2.decide(*reqs_hit[i]))
            fits = pf is not None and all(len(q.options) <= 26 for _, qs in reqs for q in qs)
            dsuf = "" if a.decoder == "smol1.7" else f"-{a.decoder}"
            if fits and "prefill" in a.paths:
                cell("prefill-only-decoder" + dsuf, lambda i: [pf.decide(reqs[i][0], q) for q in reqs[i][1]], slow=True)
            if fits and "prefill-uncached" in a.paths:
                cell("prefill-uncached-decoder" + dsuf, lambda i: pf.decide_uncached(*reqs[i]), slow=True)
            if fits and "prefill-cached" in a.paths:
                cell("prefill-cached-decoder" + dsuf, lambda i: pf.decide_cached(*reqs[i]), slow=True)
            if gl is not None:
                cell("gliner-decide", lambda i: gl.decide(*reqs[i]), slow=True)
            if "ort-one-seq" in a.paths:
                def ort_one_seq(i, path=ort["fp32"]):
                    s, qs = reqs[i]
                    for q in qs:
                        b = pack_batch([layout_request(tok, s, [q], "full")], tok.pad_id, "full")
                        path.decisions(b, [[q]])
                cell("ort-fp32-one-seq-full", ort_one_seq)
            if "ort-int8-one-seq" in a.paths and "int8" in ort:
                def ort_one_seq8(i, path=ort["int8"]):
                    s, qs = reqs[i]
                    for q in qs:
                        b = pack_batch([layout_request(tok, s, [q], "full")], tok.pad_id, "full")
                        path.decisions(b, [[q]])
                cell("ort-int8-one-seq-full", ort_one_seq8)
            if "ort-packed" in a.paths:
                def ort_packed(i, path=ort["fp32"]):
                    s, qs = reqs[i]
                    b = pack_batch([layout_request(tok, s, qs, "block")], tok.pad_id, "block")
                    path.decisions(b, [qs])
                cell("ort-fp32-packed-block", ort_packed)
            ref = cells.get("torch-one-seq-full") or next(iter(cells.values()))
            grid[f"S{S}_K{K}"] = {"N": N, "p50_ms": {k: c.p50_ms for k, c in cells.items()},
                                  "per_decision_p50_ms": {k: c.per_decision_p50_ms for k, c in cells.items()},
                                  "speedup_vs_one_seq": {k: compare(c, ref) for k, c in cells.items()}}
            print(f"S={S} K={K}", {k: round(v, 1) for k, v in grid[f'S{S}_K{K}']["p50_ms"].items()}, flush=True)
    write_json(out_dir / "summary.json", {"host": host, "threads": a.threads, "size": a.size, "task": a.task, "torch": torch.__version__, "grid": grid})
