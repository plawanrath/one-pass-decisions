"""Diagnose a cascade-kernel disagreement on the GPU. (1) Does the variable-length FlashAttention kernel's log-sum-exp
equal logsumexp(scale * q.k) in natural log, for bf16 inputs, in the shapes the cascade kernel uses (queries longer
than keys, keys longer than queries, a window)? (2) Per contract: the cascade engine's plain cold path and its cold path
with the cache fill, each against masked SDPA in bf16 (argmax agreement and max probability difference).

    .venv/bin/python scripts/cascade_diag.py
"""
from __future__ import annotations

import json
import math

import torch

from onepass.cli import ENCODERS, REPO
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.schema import Question

if __name__ == "__main__":
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    torch.manual_seed(0)
    dev, H, D, sc = "cuda", 12, 64, 1 / math.sqrt(64)
    for lq, lk, win in ((700, 4096, -1), (40, 40, -1), (40, 104, 64), (4096, 4096, 64)):
        q = torch.randn(lq, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(lk, H, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(lk, H, D, device=dev, dtype=torch.bfloat16)
        cq = torch.tensor([0, lq], dtype=torch.int32, device=dev)
        ck = torch.tensor([0, lk], dtype=torch.int32, device=dev)
        o, lse = varlen_attn(q, k, v, cq, ck, lq, lk, return_aux=AuxRequest(lse=True), scale=sc, window_size=(win, win))
        s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * sc
        if win >= 0:
            iq = torch.arange(lq, device=dev)[:, None] + (lk - lq)
            ik = torch.arange(lk, device=dev)[None]
            s = s.masked_fill(((iq - ik).abs() > win)[None], float("-inf"))
        ref = torch.logsumexp(s, -1)
        oref = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v.float())
        print(f"[diag] lse lq={lq} lk={lk} win={win}: shape {tuple(lse.shape)} dtype {lse.dtype}; max |lse-ref| {(lse.float() - ref).abs().max():.4f}; "
              f"ratio ref/lse {(ref / lse.float()).median():.4f}; max |o-oref| {(o.float() - oref).abs().max():.4f}", flush=True)
    enc = OnePassEncoder.from_pretrained(*ENCODERS["base"]).eval().to(dev).to(torch.bfloat16)
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())["4096:17"][:10]
    reqs = [(r["state"], [Question(x["kind"], x["text"], x["options"], qid=x["qid"]) for x in r["questions"]]) for r in raw]
    torch.backends.cuda.enable_cudnn_sdp(False)
    with torch.no_grad():
        for i, (st, qs) in enumerate(reqs):
            ref = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=4096, use_cache=False)).decide(st, qs)
            row = []
            for name, cfg in (("plain", dict(use_cache=False)), ("fill", dict(use_cache=True, cache_bytes=8 << 30))):
                d = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=4096, attn_kernel="cascade", **cfg)).decide(st, qs)
                same = sum(x.index == y.index for x, y in zip(ref, d))
                diff = max(max(abs(p - q) for p, q in zip(x.probs, y.probs)) for x, y in zip(ref, d))
                row.append(f"{name} {same}/{len(d)} {diff:.3f}")
            print(f"[diag] req {i}: " + "; ".join(row), flush=True)
