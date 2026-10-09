"""What the state cache costs and what reading it costs (ADR-0006, E12).

    .venv/bin/python scripts/cache_study.py --device cuda --dtype bf16
(1) Bytes per cached token for every encoder (all layers; and trimmed: local-attention layers keep only the state tail
    a later query can see) and, for comparison, the decoders' prefix caches, from the model configs; bytes per
    contract at the contract lengths of the document tasks.
(2) Measured on the device with the contract-review model: for states of S tokens and K = 17 questions, the cache
    entry's bytes, the engine's cold latency (fused fill) and hit latency, trimmed and untrimmed, and the hit path's
    effective read bandwidth (entry bytes / hit latency); the windowed engine at 32,768 tokens (stock encoder).
Writes results/cache_study/<host>/summary.json (and cells/ with the raw latencies).
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from onepass.bench.requests import make_requests
from onepass.cli import DECODERS, ENCODERS, RESULTS, SYSTEM2, host_tag, model_dir, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool


def encoder_bytes(cfg, dtype_bytes: int, trimmed: bool, tokens: int) -> int:
    """Bytes of one cached state of `tokens` tokens (keys and values of every layer)."""
    t = cfg.model_type
    heads = cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", None) or heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    per_layer_tok = 2 * kv * hd * dtype_bytes
    L = cfg.num_hidden_layers
    if t == "modernbert" and trimmed:
        every = cfg.global_attn_every_n_layers
        n_global = len([i for i in range(L) if i % every == 0])
        half = cfg.local_attention // 2
        return n_global * per_layer_tok * tokens + (L - n_global) * per_layer_tok * min(tokens, half)
    return L * per_layer_tok * tokens


def decoder_bytes_per_token(cfg, dtype_bytes: int = 2) -> int:
    from onepass.decoder import kv_bytes_per_token

    return kv_bytes_per_token(cfg, dtype_bytes)


def analytic() -> dict:
    from transformers import AutoConfig

    out = {"encoders": {}, "decoders": {}}
    for name in ("base", "large", "eurobert-base", "eurobert-large"):
        repo, rev = ENCODERS[name]
        cfg = AutoConfig.from_pretrained(repo, revision=rev, trust_remote_code=False)
        row = {}
        for dt, b in (("bf16", 2), ("fp32", 4)):
            row[f"bytes_per_token_{dt}"] = encoder_bytes(cfg, b, False, 1)
            row[f"bytes_4k_{dt}"] = encoder_bytes(cfg, b, False, 4096)
            row[f"bytes_8k_{dt}"] = encoder_bytes(cfg, b, False, 8192)
            row[f"bytes_32k_{dt}"] = encoder_bytes(cfg, b, False, 32768)
            row[f"trimmed_bytes_4k_{dt}"] = encoder_bytes(cfg, b, True, 4096)
            row[f"trimmed_bytes_8k_{dt}"] = encoder_bytes(cfg, b, True, 8192)
            row[f"trimmed_bytes_32k_{dt}"] = encoder_bytes(cfg, b, True, 32768)
            row[f"trimmed_bytes_per_token_{dt}"] = (encoder_bytes(cfg, b, True, 1 << 20) - encoder_bytes(cfg, b, True, 0)) / (1 << 20)
        out["encoders"][name] = row
    for name, (repo, rev) in {**DECODERS, "mistral-small-3.2": SYSTEM2}.items():
        try:
            cfg = AutoConfig.from_pretrained(repo, revision=rev)
        except Exception as e:  # noqa: BLE001  (a config that does not load is reported, not fatal)
            out["decoders"][name] = {"error": str(e)[:200]}
            continue
        bpt = decoder_bytes_per_token(cfg, 2)
        out["decoders"][name] = {"bytes_per_token_bf16": bpt, "bytes_4k_bf16": 4096 * bpt, "bytes_8k_bf16": 8192 * bpt, "bytes_32k_bf16": 32768 * bpt}
    return out


def timed(fn, n: int, warm: int, sync) -> list[float]:
    for i in range(warm):
        fn(i)
        sync()
    lat = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(warm + i)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--S", type=int, nargs="+", default=[1024, 2048, 4096, 8190])
    ap.add_argument("--K", type=int, default=17)
    ap.add_argument("--windowed-S", type=int, default=32768)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--analytic-only", action="store_true")
    a = ap.parse_args()
    summary = {"analytic": analytic()}
    host = host_tag() + ("" if a.device == "cpu" else f"-{a.device}") + ("" if a.dtype == "fp32" else f"-{a.dtype}")
    out = RESULTS / "cache_study" / host
    if not a.analytic_only:
        torch.backends.cuda.enable_cudnn_sdp(False)
        sync = torch.cuda.synchronize if a.device == "cuda" else (torch.mps.synchronize if a.device == "mps" else (lambda: None))
        md = model_dir("base", "block", "contractnli")
        enc = OnePassEncoder.load(md).eval().to(a.device)
        tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(md))
        if a.dtype == "bf16":
            enc = enc.to(torch.bfloat16)
        recs = load_pool(pool_dir("contractnli"), "test")
        meas = {}
        with torch.no_grad():
            cells = [(S, "block", enc, tok, recs) for S in a.S]
            if a.windowed_S:
                senc = OnePassEncoder.from_pretrained(*ENCODERS["base"]).eval().to(a.device)
                senc = senc.to(torch.bfloat16) if a.dtype == "bf16" else senc
                _, stok = load_tokenizer_adapter(*ENCODERS["base"])
                cells.append((a.windowed_S, "windowed", senc, stok, load_pool(pool_dir("cuad"), "test")))
            for S, mask, e, t, rr in cells:
                reqs = make_requests(rr, t, S, a.K, a.n + a.warmup, seed=S * 1000 + a.K)
                row = {}
                for trim in (False, True):
                    cfg = EngineConfig(mask=mask, max_state_tokens=S, trim_local=trim, cache_bytes=64 << 30)
                    eng = OnePassEngine(e, t, cfg)
                    cold, hit, nb, nt = [], [], [], []
                    for i in range(a.warmup + a.n):
                        # each request cold (the fused pass fills the cache), then the same state again (hit), then the
                        # cache is emptied: only one entry is ever resident, so long states cannot exhaust the device
                        sync()
                        t0 = time.perf_counter()
                        eng.decide(*reqs[i])
                        sync()
                        t1 = time.perf_counter()
                        eng.decide(reqs[i][0], reqs[i][1][::-1])
                        sync()
                        t2 = time.perf_counter()
                        ent = next(iter(eng.cache._d.values()))
                        if i >= a.warmup:
                            cold.append((t1 - t0) * 1000)
                            hit.append((t2 - t1) * 1000)
                            nb.append(ent.n_bytes)
                            nt.append(ent.state_len)
                        eng.cache._d.clear()
                        eng.cache.bytes = 0
                    del eng
                    if a.device == "cuda":
                        torch.cuda.empty_cache()
                    nbytes, ntok = statistics.median(nb), statistics.median(nt)
                    hit_p50 = float(np.percentile(hit, 50))
                    row["trimmed" if trim else "full"] = {"cold_p50_ms": float(np.percentile(cold, 50)), "hit_p50_ms": hit_p50,
                                                          "entry_bytes": nbytes, "state_tokens": ntok, "bytes_per_token": nbytes / ntok,
                                                          "hit_read_GBps": nbytes / (hit_p50 / 1000) / 1e9, "cold_ms": cold, "hit_ms": hit}
                    print(f"S={S} {mask} {'trimmed' if trim else 'full   '}: cold {row['trimmed' if trim else 'full']['cold_p50_ms']:.1f} ms, "
                          f"hit {hit_p50:.1f} ms, entry {nbytes / 1e6:.1f} MB ({nbytes / ntok / 1024:.1f} KB/token)", flush=True)
                meas[f"{mask}-{S}"] = row
        summary["measured"] = {"device": a.device, "dtype": a.dtype, "K": a.K, "n": a.n, "cells": meas}
    write_json(out / "summary.json", summary)
    for k, v in summary["analytic"]["encoders"].items():
        print(f"{k}: {v['bytes_per_token_bf16'] / 1024:.1f} KB/token bf16 (trimmed {v['trimmed_bytes_per_token_bf16'] / 1024:.1f}); 4k contract {v['bytes_4k_bf16'] / 1e9:.3f} GB (trimmed {v['trimmed_bytes_4k_bf16'] / 1e9:.3f})")
    for k, v in summary["analytic"]["decoders"].items():
        print(f"{k}: " + (f"{v['bytes_per_token_bf16'] / 1024:.1f} KB/token bf16; 4k {v['bytes_4k_bf16'] / 1e9:.3f} GB" if "error" not in v else v["error"]))
