"""Reuse across requests on two documented review patterns, measured end to end (not argued from cells).

  two-question-sets  each contract is reviewed twice: the 17 ContractNLI hypotheses, then CUAD's 41 clause questions
                     (the two public contract-review question sets), as a tool that runs an NLI check and a clause
                     extraction over the same document would;
  added-questions    a reviewer asks the 17 hypotheses in four batches (5, 4, 4, 4, in the dataset's order), as a
                     session that adds questions after reading the first answers would.
Every request is served once on every path before timing (first-seen shapes cost extra). For each contract session, the summed wall time of its requests on three paths: the engine with its cache (the first
request cold, filling the cache; later requests resident), the engine with every request cold (the cache cleared before
each request), and unpadded FlashAttention batching (one sequence per question; nothing to reuse).

    .venv/bin/python scripts/session_replay.py --device cuda --contracts 30
Contracts are the first N of the ContractNLI test pool (truncated to 4,096 tokens); the encoder is stock ModernBERT
base (latency does not depend on the weights). Writes results/session_replay/<host>.json.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import OrderedDict

import torch

from onepass.cli import ENCODERS, RESULTS, host_tag, pool_dir, write_json
from onepass.engine.engine import EngineConfig, OnePassEngine
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.paths.varlen import varlen_paths
from onepass.tasks.pools import load_pool


def contracts(n: int):
    by: OrderedDict = OrderedDict()
    for r in load_pool(pool_dir("contractnli"), "test"):
        by.setdefault(r.state, []).append(r.question)
    return list(by.items())[:n]


def clause_questions():
    seen: OrderedDict = OrderedDict()
    for r in load_pool(pool_dir("cuad"), "test"):
        seen.setdefault(r.question.qid, r.question)
        if len(seen) == 41:
            break
    return list(seen.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--contracts", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--kernel", default="cascade", help="engine attention kernel (sdpa on CPUs)")
    ap.add_argument("--max-state-tokens", type=int, default=4096)
    a = ap.parse_args()
    if a.device == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    enc = OnePassEncoder.from_pretrained(*ENCODERS["base"]).eval().to(a.device)
    if a.device == "cuda":
        enc = enc.to(torch.bfloat16)
    _, tok = load_tokenizer_adapter(*ENCODERS["base"])
    S = a.max_state_tokens
    cached = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=S, attn_kernel=a.kernel, cache_bytes=16 << 30))
    cold = OnePassEngine(enc, tok, EngineConfig(max_state_tokens=S, attn_kernel=a.kernel, cache_bytes=16 << 30))
    docs = contracts(a.contracts + a.warmup)
    clauses = clause_questions()
    patterns = {"two-question-sets": lambda qs: [qs, clauses], "added-questions": lambda qs: [qs[:5], qs[5:9], qs[9:13], qs[13:17]]}

    def run_cached(state, reqs):
        cached.cache._d.clear()  # a new contract arrives cold; its later requests find it resident
        out = []
        for qs in reqs:
            sync(); t0 = time.perf_counter(); cached.decide(state, qs); sync(); out.append((time.perf_counter() - t0) * 1000)
        return out

    def run_cold(state, reqs):
        out = []
        for qs in reqs:
            cold.cache._d.clear()  # every request re-encodes the contract
            sync(); t0 = time.perf_counter(); cold.decide(state, qs); sync(); out.append((time.perf_counter() - t0) * 1000)
        return out

    def run_baseline(state, reqs):
        out = []
        for qs in reqs:
            sync(); t0 = time.perf_counter(); varlen_paths(enc, tok, state, qs, batched=True, max_state_tokens=S, state_ids=cached._ids(state)); sync()
            out.append((time.perf_counter() - t0) * 1000)
        return out

    paths = {"engine, cached": run_cached, "engine, every request cold": run_cold, "unpadded batching": run_baseline}
    # every request is served once, untimed, on every path before any timing: a first-seen sequence length costs extra
    # (kernel selection and allocation), and without this pass the first path timed pays it for all of them
    with torch.no_grad():
        for state, qs in docs:
            for split in patterns.values():
                for fn in paths.values():
                    fn(state, split(qs[:17]))
    res = {"device": a.device, "kernel": a.kernel, "contracts": a.contracts, "max_state_tokens": S, "patterns": {}}
    with torch.no_grad():
        for pname, split in patterns.items():
            res["patterns"][pname] = {}
            for path, fn in paths.items():
                per = []
                for i, (state, qs) in enumerate(docs):
                    reqs = split(qs[:17])
                    t = fn(state, reqs)
                    if i >= a.warmup:
                        per.append(t)
                tot = [sum(t) for t in per]
                res["patterns"][pname][path] = {"session_ms_median": statistics.median(tot), "session_ms": tot,
                                                "per_request_ms_median": [statistics.median(t[j] for t in per) for j in range(len(per[0]))]}
                print(f"[session] {pname:18s} {path:28s} session median {statistics.median(tot):8.1f} ms; per request "
                      + ", ".join(f"{x:.1f}" for x in res["patterns"][pname][path]["per_request_ms_median"]), flush=True)
            c = res["patterns"][pname]
            c["speedup_cached_vs_cold"] = c["engine, every request cold"]["session_ms_median"] / c["engine, cached"]["session_ms_median"]
            c["speedup_cached_vs_baseline"] = c["unpadded batching"]["session_ms_median"] / c["engine, cached"]["session_ms_median"]
    write_json(RESULTS / "session_replay" / f"{host_tag()}-{a.device}.json", res)
