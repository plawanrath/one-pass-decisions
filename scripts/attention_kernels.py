"""Which attention kernel each engine path actually runs on the GPU, and whether the FlexAttention path agrees.

    .venv/bin/python scripts/attention_kernels.py --size base --task contractnli --cell 4096:17 --device cuda
(1) Agreement: on a few real requests, the engine with masked SDPA and with FlexAttention give the same decisions (bf16:
    the same argmax and probabilities within 2e-2), cold and cached.
(2) Kernels: one cold request through each engine under the profiler; the CUDA kernels by total time, so the paper can
    say which SDPA backend an arbitrary additive mask selects and what replaced it.
(3) Utilization: the engine's cold request in model FLOPs per second (linear layers plus the attention its mask allows),
    for both kernels and for the unpadded baseline, from the median of `--n` requests.
Writes results/attention_kernels/<host>-<size>-<task>-S<S>-K<K>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from onepass.cli import ENCODERS, REPO, RESULTS, host_tag, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.model.mask import allowed_matrix, window_matrix
from onepass.paths.varlen import default_kernel, varlen_paths
from onepass.schema import Question


def model_flops(enc, lay, mode: str) -> float:
    """Forward FLOPs of one packed request: 2 x parameters x tokens for the linear layers, plus 4 x head_dim x heads x
    (allowed query-key pairs) per layer for attention (scores and weighted values), counting each layer type's mask."""
    import torch as t

    n = len(lay.ids)
    params = sum(p.numel() for name, p in enc.backbone.named_parameters() if "embeddings" not in name and "embed_tokens" not in name)
    lin = 2.0 * params * n
    blk = t.tensor([lay.block_ids]); pad = t.ones_like(blk, dtype=t.bool); pos = t.tensor([lay.positions])
    cids = t.tensor([lay.chunk_ids]) if lay.chunk_ids is not None else None
    allowed = allowed_matrix(blk, pad, mode, cids)
    pairs_full = float(allowed.sum())
    pairs_local = float((allowed & window_matrix(pos, enc.half_window)).sum())
    cfg = enc.backbone.config
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    layers = [enc.ops.layer_type(l) for l in enc.ops.layers(enc.backbone)]
    att = sum(4.0 * width * (pairs_local if ty == "sliding_attention" else pairs_full) for ty in layers)
    return lin + att


def timed(fn, n, sync):
    for _ in range(3):
        fn()
    sync()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--task", default="contractnli")
    ap.add_argument("--cell", default="4096:17")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--agree-requests", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix of the output file")
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    enc = OnePassEncoder.from_pretrained(*ENCODERS[a.size]).eval().to(a.device).to(torch.bfloat16)
    _, tok = load_tokenizer_adapter(*ENCODERS[a.size])
    S, K = (int(x) for x in a.cell.split(":"))
    raw = json.loads((REPO / "data" / "cache" / "requests" / f"{a.task}.json").read_text())[a.cell]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]]) for r in raw]
    cap = min(max(S, 4096), 8190)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    out = {"size": a.size, "task": a.task, "S": S, "K": K, "agreement": {}, "kernels": {}, "ms": {}, "tflops": {}}
    with torch.no_grad():
        engines = {k: OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, attn_kernel=k, cache_bytes=8 << 30)) for k in ("sdpa", "flex", "split", "cascade")}
        # (1) agreement, cold then cached
        agree = {k: [0.0, 0, 0] for k in ("split", "cascade")}  # worst prob diff, same argmax, total; against sdpa
        for s, qs in reqs[: a.agree_requests]:
            for _ in range(2):
                d = {k: e.decide(s, qs) for k, e in engines.items()}
                for k in agree:
                    for x, y in zip(d["sdpa"], d[k]):
                        agree[k][0] = max(agree[k][0], max(abs(p - q) for p, q in zip(x.probs, y.probs)))
                        agree[k][1] += int(x.index == y.index)
                        agree[k][2] += 1
        out["agreement"] = {"pair": "split vs sdpa", "max_prob_diff": agree["split"][0], "argmax_same": agree["split"][1], "total": agree["split"][2],
                            "cascade": {"pair": "cascade vs sdpa", "max_prob_diff": agree["cascade"][0], "argmax_same": agree["cascade"][1], "total": agree["cascade"][2]}}
        for k, (w, same, n) in agree.items():
            print(f"[kernels] agreement {k} vs sdpa: argmax {same}/{n}, max prob diff {w:.4f}", flush=True)
        # (2) kernels under the profiler, one cold request each
        from torch.profiler import ProfilerActivity, profile

        for k in ("sdpa", "split", "cascade"):
            cold = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, attn_kernel=k, use_cache=False))
            for _ in range(3):
                cold.decide(*reqs[0])
            sync()
            with profile(activities=[ProfilerActivity.CUDA] if a.device == "cuda" else [ProfilerActivity.CPU]) as prof:
                cold.decide(*reqs[0])
                sync()
            rows = sorted(((e.key, e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total) for e in prof.key_averages()),
                          key=lambda r: -r[1])[:12]
            out["kernels"][k] = [{"kernel": n, "us": float(t)} for n, t in rows]
            print(f"[kernels] {k}: " + "; ".join(f"{n[:60]} {t / 1000:.1f} ms" for n, t in rows[:5]), flush=True)
        # (3) utilization: median cold latency and model FLOPs per second
        lay = layout_request(tok, reqs[0][0], reqs[0][1], "block", cap, 24, state_ids=tok.encode(reqs[0][0]))
        fl = model_flops(enc, lay, "block")
        out["flops_per_request"] = fl
        for k in ("sdpa", "split", "cascade"):
            cold = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=cap, attn_kernel=k, use_cache=False))
            ms = timed(lambda: cold.decide(*reqs[1]), a.n, sync)
            out["ms"][f"engine-cold-{k}"] = ms
            out["tflops"][f"engine-cold-{k}"] = fl / (ms / 1000) / 1e12
        kern = default_kernel(a.device)
        ms = timed(lambda: varlen_paths(enc, tok, *reqs[1], batched=True, kernel=kern, state_ids=tok.encode(reqs[1][0]), max_state_tokens=cap), a.n, sync)
        base_fl = sum(model_flops(enc, layout_request(tok, reqs[1][0], [q], "full", cap, 24, state_ids=tok.encode(reqs[1][0])), "full") for q in reqs[1][1])
        out["ms"]["varlen-batched"] = ms
        out["tflops"]["varlen-batched"] = base_fl / (ms / 1000) / 1e12
        print("[kernels] " + ", ".join(f"{k} {v:.1f} ms ({out['tflops'][k]:.0f} TFLOP/s)" for k, v in out["ms"].items()), flush=True)
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}")
    write_json(RESULTS / "attention_kernels" / f"{host}-{a.size}-{a.task}-S{S}-K{K}{a.tag}.json", out)
